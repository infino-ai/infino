// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Per-write-op cost cells — the write-side equivalent of the per-query
//! read cost cells ([`crate::cost::warm_cell_per_million`] /
//! [`crate::cost::cold_cell_per_million`]).
//!
//! The read path reports a cost cell per query *shape*, so the spread
//! across shapes is visible. The write path only had one whole-run
//! aggregate (`write_per_million_docs`, dollars per doc averaged over
//! ingest, drain and compaction alike), which hides the two axes that
//! dominate per-op write cost:
//!
//! - **Batch size.** A 1-row append pays the full fixed commit cost —
//!   superfile build overhead, manifest json, pointer CAS — that a
//!   100k-row append amortizes 100 000 ways. Per-row those differ by
//!   orders of magnitude.
//! - **Modality.** A vector append encodes and places 1024-dim
//!   embeddings; a text append builds postings; the same byte count
//!   does very different work.
//!
//! This diagnostic measures each write shape as an individual committed
//! op against a seeded table, prices it with the same instance cost
//! model the read cells use (compute at the binding CPU/RAM leg +
//! object-store request dollars), and reports $/write, $/1M writes and
//! $/1M rows per shape.
//!
//! **Direct cost only, deliberately.** Deferred maintenance a write
//! later causes (hidden-index drain merges, compaction) is excluded
//! here; it is already inside the amortized `write_per_million_docs`
//! figure. These cells exist to compare shapes against each other on
//! the work the op itself performed. Engine-attributed counters (rows, planned
//! PUTs vs actual PUTs) are reported alongside so the priced billing
//! legs can be reconciled against the measured dollars.
//!
//! Corpus text/vectors come from the standard synthetic generator
//! ([`SequentialSyntheticCorpus`], same seeds and distributions as the
//! main bench corpus). SQL-schema appends are deferred to a follow-up;
//! the FTS and vector shapes carry the modality spread.
//!
//! Invoked as `cargo bench -- write-diag`.

use std::{env, sync::Arc, time::Duration};

use arrow_array::{LargeStringArray, RecordBatch};
use datafusion::prelude::{col, lit};
use infino::{
    runtime_metrics::op_stats::{OpStats, with_op_stats},
    storage::{LocalFsStorageProvider, StorageProvider},
    supertable::Supertable,
};
use tempfile::TempDir;

use crate::{
    corpus::{self, SequentialSyntheticCorpus},
    cost, cpu,
    ingest::supertable::{
        CORPUS_VEC_SEED, Modality, TEXT_COLUMN, options_for, schema_for, vector_array,
        vector_filter_bucket_term,
    },
    markdown::{fmt_count, fmt_time},
    report::{Better, Block, Cell, Report, Section, metric, text},
    rss::{self, PeakSampler},
    storage_meter::{self, MeteredStorage, ObjectStoreMeter},
};

/// Rows pre-loaded into each table before any measured op, so shapes run
/// against a real manifest rather than an empty table. Override with
/// `INFINO_BENCH_WRITE_DIAG_SEED_DOCS`.
const DEFAULT_SEED_DOCS: usize = 100_000;

/// Largest measured append, in rows. The ladder is fixed at
/// {1, 1_000, this} so the small/large amortization cliff is always
/// visible. Override with `INFINO_BENCH_WRITE_DIAG_DOCS`.
const DEFAULT_TOP_APPEND_ROWS: usize = 100_000;

/// Mid rung of the append ladder.
const MID_APPEND_ROWS: usize = 1_000;

/// Per-commit row cap while seeding. The ingest working set scales with
/// the commit's doc count, not the table's (see the ingest path's
/// MAX_DOCS_PER_COMMIT rationale), so a large seed lands as several
/// modest commits — which also matches how a real table gets to that
/// size. Measured ops are never chunked: an append shape's batch size
/// IS the thing being measured.
const SEED_CHUNK_ROWS: usize = 250_000;

/// Single-row mutations averaged per mutation cell — enough iterations
/// that the process-CPU sampler sees a measurable delta over the loop.
const N_MUTATIONS: usize = 8;

/// Text seed for the streamed corpus — must match the vector seed policy
/// of the main corpus (both derive from the same base seed).
const CORPUS_TEXT_SEED: u64 = 1;

/// Nanoseconds per second, for latency markdown.
const NS_PER_SEC: f64 = 1e9;

/// Bytes per GiB, as a float for the per-GiB cost division.
const BYTES_PER_GIB_F64: f64 = (1u64 << 30) as f64;

fn seed_docs() -> usize {
    env::var("INFINO_BENCH_WRITE_DIAG_SEED_DOCS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_SEED_DOCS)
}

fn top_append_rows() -> usize {
    env::var("INFINO_BENCH_WRITE_DIAG_DOCS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_TOP_APPEND_ROWS)
}

/// One measured write op (or averaged loop of ops): wall, CPU, peak RSS,
/// object-store window, and the engine's own per-op counters.
struct OpMeasure {
    wall: Duration,
    cpu_s: Option<f64>,
    peak_rss_bytes: u64,
    io: ObjectStoreMeter,
    stats: OpStats,
}

/// Bracket `f` with the same instruments the read cells use — process
/// CPU, wall, peak RSS, the provider's request meter — plus the engine's
/// per-op work scope.
fn measure(ms: &MeteredStorage, f: impl FnOnce()) -> OpMeasure {
    let before = ms.snapshot();
    let sampler = PeakSampler::start_default();
    let (((), stats), wall, cpu_s) = cpu::timed(|| with_op_stats(f));
    let peak_rss_bytes = sampler.stop_stats().peak_rss_bytes;
    let io = ms.snapshot().since(&before);
    OpMeasure {
        wall,
        cpu_s,
        peak_rss_bytes,
        io,
        stats,
    }
}

/// Fresh streamed generator positioned at doc 0.
fn corpus_stream(total_docs: usize) -> DocStream {
    DocStream {
        stream: SequentialSyntheticCorpus::new(
            corpus::n_cent(total_docs),
            CORPUS_VEC_SEED,
            CORPUS_TEXT_SEED,
            true,
        ),
        next_doc: 0,
    }
}

/// The synthetic stream plus its running doc position — the vector
/// schema's filter-bucket terms derive from each row's global doc id,
/// exactly as the bulk ingest path stamps them.
struct DocStream {
    stream: SequentialSyntheticCorpus,
    next_doc: usize,
}

/// Next `len` rows off the stream as a batch for `modality` (Fts => the
/// text column; Vector => bucket + embedding, matching `schema_for`'s
/// field order).
fn next_batch(src: &mut DocStream, modality: Modality, len: usize) -> RecordBatch {
    let mut titles: Vec<String> = Vec::new();
    let mut flat: Vec<f32> = Vec::new();
    let (gen_text, gen_vec) = match modality {
        Modality::Fts => (true, false),
        Modality::Vector => (false, true),
        _ => unreachable!("write-diag drives Fts and Vector shapes only"),
    };
    src.stream
        .fill_chunk_modality(len, &mut titles, &mut flat, gen_text, gen_vec);
    let doc_base = src.next_doc;
    src.next_doc += len;
    let schema = schema_for(modality);
    match modality {
        Modality::Fts => {
            let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
            RecordBatch::try_new(schema, vec![Arc::new(LargeStringArray::from(refs))])
                .expect("fts RecordBatch")
        }
        Modality::Vector => {
            let buckets: Vec<String> = (doc_base..doc_base + len)
                .map(vector_filter_bucket_term)
                .collect();
            let bucket_refs: Vec<&str> = buckets.iter().map(String::as_str).collect();
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(LargeStringArray::from(bucket_refs)),
                    vector_array(&flat),
                ],
            )
            .expect("vector RecordBatch")
        }
        _ => unreachable!(),
    }
}

/// A seeded table of `modality` on a metered local store.
fn seeded_table(
    modality: Modality,
    stream: &mut DocStream,
    n_seed: usize,
) -> (TempDir, MeteredStorage, Supertable) {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let ms = storage_meter::wrap(Arc::clone(&storage));
    let st = Supertable::create(options_for(modality, Some(storage))).expect("create supertable");
    let mut remaining = n_seed;
    while remaining > 0 {
        let chunk = remaining.min(SEED_CHUNK_ROWS);
        let batch = next_batch(stream, modality, chunk);
        st.append(&batch).expect("seed append");
        remaining -= chunk;
    }
    (dir, ms, st)
}

/// Report cells for one measured shape. `ops` is how many ops the
/// measurement window covered (1 for the appends, `N_MUTATIONS` for the
/// averaged mutation loops); every per-op figure divides by it.
fn shape_row(label: &str, rows_per_op: usize, ops: usize, m: &OpMeasure) -> Vec<Cell> {
    let ops_f = ops as f64;
    let wall_s = m.wall.as_secs_f64() / ops_f;
    let cpu_s = m.cpu_s.map(|c| c / ops_f);
    let per_op_usd =
        cost::write_op_usd(m.cpu_s, m.wall.as_secs_f64(), Some(m.peak_rss_bytes), &m.io)
            .map(|total| total / ops_f);
    let per_million_rows = per_op_usd.map(|usd| usd / (rows_per_op.max(1) as f64) * 1e6);
    // Logical payload this op ingested — the caller's own bytes, before
    // any index or replication expansion. Carrying it here is what lets a
    // per-op cost convert to $/GiB, so cost per byte is comparable across
    // shapes whose row sizes differ.
    let logical_bytes = m
        .stats
        .scalar_bytes_written
        .saturating_add(m.stats.vector_bytes_written) as f64
        / ops_f;
    let per_gib = per_op_usd
        .and_then(|usd| (logical_bytes > 0.0).then(|| usd / (logical_bytes / BYTES_PER_GIB_F64)));
    let wall_ns = wall_s * NS_PER_SEC;
    vec![
        text(label),
        metric(rows_per_op as f64, fmt_count(rows_per_op), Better::Higher),
        metric(wall_ns, fmt_time(wall_ns), Better::Lower),
        match cpu_s {
            Some(c) => metric(c, format!("{c:.4}"), Better::Lower),
            None => text("—"),
        },
        metric(
            m.peak_rss_bytes as f64,
            rss::fmt_bytes(m.peak_rss_bytes),
            Better::Lower,
        ),
        metric(
            m.io.put_count as f64 / ops_f,
            format!("{:.0}", m.io.put_count as f64 / ops_f),
            Better::Lower,
        ),
        metric(
            m.stats.planned_write_requests as f64 / ops_f,
            format!("{:.0}", m.stats.planned_write_requests as f64 / ops_f),
            Better::Lower,
        ),
        metric(
            m.io.get_count as f64 / ops_f,
            format!("{:.0}", m.io.get_count as f64 / ops_f),
            Better::Lower,
        ),
        match per_op_usd {
            Some(usd) => metric(usd, cost::write_cell(usd), Better::Lower),
            None => text("—"),
        },
        match per_million_rows {
            Some(usd) => metric(usd, cost::usd_text(usd), Better::Lower),
            None => text("—"),
        },
        metric(
            logical_bytes,
            rss::fmt_bytes(logical_bytes as u64),
            Better::Higher,
        ),
        match per_gib {
            Some(usd) => metric(usd, cost::usd_text(usd), Better::Lower),
            None => text("—"),
        },
    ]
}

pub fn run() {
    let n_seed = seed_docs();
    let top = top_append_rows();
    let append_ladder: [usize; 3] = [1, MID_APPEND_ROWS, top];
    let total_docs = n_seed + append_ladder.iter().sum::<usize>() + N_MUTATIONS;
    eprintln!(
        "[write-diag] seed {} docs/table, appends {:?}, {} single-row mutations...",
        fmt_count(n_seed),
        append_ladder,
        N_MUTATIONS
    );

    let mut rows: Vec<Vec<Cell>> = Vec::new();

    // ── Appends: the batch-size ladder, per modality ──────────────────
    for modality in [Modality::Fts, Modality::Vector] {
        let label = match modality {
            Modality::Fts => "fts",
            Modality::Vector => "vector",
            _ => unreachable!(),
        };
        let mut stream = corpus_stream(total_docs);
        let (_dir, ms, st) = seeded_table(modality, &mut stream, n_seed);
        for &n_rows in &append_ladder {
            let batch = next_batch(&mut stream, modality, n_rows);
            let m = measure(&ms, || {
                st.append(&batch).expect("measured append");
            });
            assert_eq!(
                m.stats.rows_written, n_rows as u64,
                "the engine's per-op counter must see exactly the appended rows"
            );
            rows.push(shape_row(
                &format!("append_{label}_{}_rows", fmt_count(n_rows)),
                n_rows,
                1,
                &m,
            ));
        }
    }

    // ── Single-row mutations, averaged over N_MUTATIONS ops ───────────
    // Marker rows appended (uncounted) so predicates hit exactly one row
    // each against the seeded corpus table.
    let mut stream = corpus_stream(total_docs);
    let (_dir, ms, st) = seeded_table(Modality::Fts, &mut stream, n_seed);
    let markers: Vec<String> = (0..N_MUTATIONS)
        .map(|i| format!("write-diag-marker-{i:04}"))
        .collect();
    {
        let refs: Vec<&str> = markers.iter().map(String::as_str).collect();
        let batch = RecordBatch::try_new(
            schema_for(Modality::Fts),
            vec![Arc::new(LargeStringArray::from(refs))],
        )
        .expect("marker batch");
        st.append(&batch).expect("marker append");
    }

    let update_m = measure(&ms, || {
        for marker in &markers {
            let replacement = RecordBatch::try_new(
                schema_for(Modality::Fts),
                vec![Arc::new(LargeStringArray::from(vec![marker.as_str()]))],
            )
            .expect("replacement batch");
            st.update(col(TEXT_COLUMN).eq(lit(marker.as_str())), &replacement)
                .expect("measured update");
        }
    });
    rows.push(shape_row("update_1_row", 1, N_MUTATIONS, &update_m));

    let delete_m = measure(&ms, || {
        for marker in &markers {
            st.delete(col(TEXT_COLUMN).eq(lit(marker.as_str())))
                .expect("measured delete");
        }
    });
    rows.push(shape_row("delete_1_row", 1, N_MUTATIONS, &delete_m));

    let mut report = Report::load("write-diag");
    report.emit(&Section {
        anchor: "bench/write_diag/shapes".into(),
        title: format!(
            "Per-write-op cost by shape ({} seed docs/table)",
            fmt_count(n_seed)
        ),
        note: "Each shape is one committed op (mutations averaged over a small loop) against a \
               seeded local-store table, priced with the same instance cost model as the read \
               cells: compute at the binding CPU/RAM leg + request dollars. Direct cost only — \
               deferred drain/compaction is excluded here and amortized into the \
               `write_per_million_docs` anchor. `PUT plan` is the engine's priceable \
               `planned_write_requests`; `PUT act` is the provider meter's actual count. \
               `Logical` is the payload the caller sent, so `$/GiB logical` makes cost \
               per byte comparable across shapes with different row sizes. \
               Δ is vs the previous run."
            .into(),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "Shape".into(),
                "Rows/op".into(),
                "Wall/op".into(),
                "CPU s/op".into(),
                "Peak RSS".into(),
                "PUT act".into(),
                "PUT plan".into(),
                "GET".into(),
                "$/write".into(),
                "$/1M rows".into(),
                "Logical".into(),
                "$/GiB logical".into(),
            ],
            rows,
        }],
    });
    report.save();
}
