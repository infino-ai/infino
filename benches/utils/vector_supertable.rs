//! Infino-only vector bench for the supertable layer:
//!
//!   ingest timing (10M × 384, sharded into [`N_SEGMENTS`] superfiles)
//! + calibrated kNN search at recall targets {0.90, 0.95, 0.99}
//! + correctness gate (`recall@10 ≥ 0.80` at high-recall config)
//!
//! Multi-segment shape: the corpus is sharded into [`N_SEGMENTS`]
//! commits with `n_cent_per_segment = n_cent(N_DOCS) / N_SEGMENTS`. A
//! supertable query's per-segment `nprobe` applies to every segment, so
//! the effective probe count is `nprobe × n_superfiles`.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench supertable_all -- supertable_all_build   # shared FTS + vector ingest only
//! cargo bench --bench supertable_all -- supertable_vec_search  # vector search only
//! cargo bench --bench supertable_all -- supertable_fts_search  # FTS search only
//! ```

use std::hint::black_box;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::corpus::{self, Calibrated, DIM};
use crate::tiers::{self, Tier};
use crate::{markdown, rss};
use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, Throughput, criterion_group};
use infino::superfile::builder::{FtsConfig, VectorConfig};
use infino::superfile::fts::reader::BoolMode;
use infino::superfile::fts::tokenize::Tokenizer;
use infino::superfile::vector::distance::Metric;
use infino::supertable::query::SuperfileHit;
use infino::supertable::query::vector::VectorSearchOptions;
use infino::supertable::{Supertable, SupertableOptions};
use infino::test_helpers::default_tokenizer;

// ─── Constants ────────────────────────────────────────────────────────

/// Doc count for every vector-supertable bench. Supertable is the
/// scale-out shape → 10M.
const N_DOCS: usize = corpus::SUPERTABLE_DOCS;

const N_SEGMENTS: usize = 16;
const TOP_K: usize = 10;

const TEXT_COLUMN: &str = "title";
const VEC_COLUMN: &str = "emb";

const N_CORRECTNESS_QUERIES: usize = 20;
const N_CALIBRATION_QUERIES: usize = 100;

const RECALL_TARGETS: &[f32] = &[0.90, 0.95, 0.99];
const BUILD_MEASUREMENT_TIME: Duration = Duration::from_secs(60 * 60);

/// Per-segment probe grid. The supertable applies a single `nprobe`
/// to every segment, so the effective probe count is `nprobe ×
/// n_superfiles`. Each segment carries `n_cent(N_DOCS) / N_SEGMENTS`
/// clusters with ~2.5× the docs-per-cluster of the 1M single-superfile
/// grid, so the upper end has to climb well past the old 16-probe cap
/// to reach the same recall — otherwise calibration reports an
/// artificially low ceiling that's just "under-probed", not a real
/// quality limit.
const SUPERTABLE_PROBES_PER_SEG: &[usize] = &[1, 2, 4, 8, 12, 16, 32, 64, 128];
const SUPERTABLE_REFINES: &[usize] = &[4, 16, 64, 256, 1024];

const CORRECTNESS_RECALL_FLOOR: f32 = 0.80;
const CORRECTNESS_SUPERTABLE_NPROBE: usize = 16;
const CORRECTNESS_SUPERTABLE_RERANK_MULT: usize = 256;

const FTS_SEARCH_IDS: &[&str] = &[
    "single_rare_supertable_top10",
    "single_common_supertable_top10",
    "two_term_or_supertable_top10",
    "three_wide_or_supertable_top10",
    "three_similar_or_supertable_top10",
    "five_term_or_supertable_top10",
    "prefix_supertable_top10",
];

// ─── Fixtures ────────────────────────────────────────────────────────

static VECTORS: OnceLock<corpus::MmapVectorCorpus> = OnceLock::new();
static TEXT_CORPUS: OnceLock<corpus::MmapTextCorpus> = OnceLock::new();
static QUERIES_CORRECTNESS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
static QUERIES_CALIBRATION: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
static GROUND_TRUTH_CORRECTNESS: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
static GROUND_TRUTH_CALIBRATION: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
static SUPERTABLE: OnceLock<Supertable> = OnceLock::new();
/// Wall-clock nanoseconds of the single shared supertable build,
/// recorded the first (and only) time `SUPERTABLE` is initialized. The
/// ingest bench reports this one measurement rather than rebuilding the
/// 10M corpus per criterion sample.
static SUPERTABLE_BUILD_NS: OnceLock<f64> = OnceLock::new();
static CALIBRATIONS: OnceLock<Calibrations> = OnceLock::new();

/// Producer has committed the 10M combined FTS + vector supertable to object storage.
struct S3VectorCommitted {
    storage: Arc<dyn infino::supertable::storage::StorageProvider>,
    storage_label: &'static str,
}
static S3_VECTOR: OnceLock<S3VectorCommitted> = OnceLock::new();

fn vectors() -> &'static [f32] {
    VECTORS
        .get_or_init(|| {
            // Raw corpus fixture only. Ingestion below still goes through
            // RecordBatch -> writer.append() -> commit(); the mmap prevents
            // the synthetic 10M x 384 source corpus from living as heap RAM.
            corpus::MmapVectorCorpus::generate(N_DOCS, corpus::n_cent(N_DOCS), 1, true)
        })
        .as_slice()
}

fn text_corpus() -> &'static corpus::MmapTextCorpus {
    TEXT_CORPUS.get_or_init(|| corpus::MmapTextCorpus::generate(N_DOCS, 1))
}

fn queries_correctness() -> &'static [Vec<f32>] {
    QUERIES_CORRECTNESS.get_or_init(|| {
        corpus::generate_realistic_queries(vectors(), N_DOCS, N_CORRECTNESS_QUERIES, 17, true, 0.05)
    })
}

fn queries_calibration() -> &'static [Vec<f32>] {
    QUERIES_CALIBRATION.get_or_init(|| {
        corpus::generate_realistic_queries(vectors(), N_DOCS, N_CALIBRATION_QUERIES, 99, true, 0.05)
    })
}

fn ground_truth_correctness() -> &'static [Vec<u32>] {
    GROUND_TRUTH_CORRECTNESS
        .get_or_init(|| corpus::ground_truth(vectors(), N_DOCS, queries_correctness(), TOP_K))
}

fn ground_truth_calibration() -> &'static [Vec<u32>] {
    GROUND_TRUTH_CALIBRATION
        .get_or_init(|| corpus::ground_truth(vectors(), N_DOCS, queries_calibration(), TOP_K))
}

fn supertable() -> &'static Supertable {
    SUPERTABLE.get_or_init(|| {
        let t0 = Instant::now();
        let st = build_supertable();
        let _ = SUPERTABLE_BUILD_NS.set(t0.elapsed().as_secs_f64() * 1e9);
        st
    })
}

fn ensure_supertable(reason: &str) -> &'static Supertable {
    if SUPERTABLE.get().is_none() {
        eprintln!(
            "[supertable_all] initializing shared combined FTS + vector supertable ({N_DOCS} docs × {N_SEGMENTS} superfiles) for {reason}..."
        );
    }
    supertable()
}

// ─── Builder ──────────────────────────────────────────────────────────

fn supertable_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(TEXT_COLUMN, DataType::LargeUtf8, false),
        Field::new(
            VEC_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]))
}

fn vector_supertable_options(
    storage: Option<Arc<dyn infino::supertable::storage::StorageProvider>>,
) -> SupertableOptions {
    let n_cent_total = corpus::n_cent(N_DOCS);
    let n_cent_per_segment = (n_cent_total / N_SEGMENTS).max(1);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get().max(1))
            .build()
            .expect("pool"),
    );
    let tk: Arc<dyn Tokenizer> = default_tokenizer();
    let mut opts = SupertableOptions::new(
        supertable_schema(),
        vec![FtsConfig {
            column: TEXT_COLUMN.into(),
        }],
        vec![VectorConfig {
            column: VEC_COLUMN.into(),
            dim: DIM,
            n_cent: n_cent_per_segment,
            rot_seed: 7,
            metric: Metric::Cosine,
            rerank_codec: infino::superfile::vector::rerank_codec::RerankCodec::Sq8Residual,
        }],
        Some(tk),
    )
    .expect("opts")
    .with_reader_pool(pool.clone())
    .with_commit_threshold_size_mb(1024)
    .with_writer_pool(pool);
    if let Some(s) = storage {
        opts = opts.with_storage(s);
    }
    opts
}

fn build_supertable() -> Supertable {
    build_supertable_with_options(vector_supertable_options(None))
}

/// Commit the 10M combined FTS + vector supertable to object storage (producer path).
fn build_supertable_on_storage(
    storage: Arc<dyn infino::supertable::storage::StorageProvider>,
) -> Supertable {
    build_supertable_with_options(vector_supertable_options(Some(storage)))
}

fn build_supertable_with_options(opts: SupertableOptions) -> Supertable {
    let st = Supertable::create(opts);
    let mut w = st.writer().expect("writer");
    let chunk_size = N_DOCS.div_ceil(N_SEGMENTS);
    let v = vectors();
    let text = text_corpus();
    for start in (0..N_DOCS).step_by(chunk_size) {
        let end = (start + chunk_size).min(N_DOCS);
        let len = end - start;
        let titles = LargeStringArray::from(text.chunk_strs(start, len));
        let flat: Vec<f32> = v[start * DIM..end * DIM].to_vec();
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Float32Array::from(flat);
        let fsl = FixedSizeListArray::try_new(
            item_field,
            DIM as i32,
            Arc::new(values) as Arc<dyn Array>,
            None,
        )
        .expect("FSL");
        let batch =
            RecordBatch::try_new(supertable_schema(), vec![Arc::new(titles), Arc::new(fsl)])
                .expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    st
}

fn s3_vector_committed() -> &'static S3VectorCommitted {
    S3_VECTOR.get_or_init(|| {
        eprintln!(
            "[supertable_all] committing combined FTS + vector supertable ({N_DOCS} docs) to object storage for warm/cold tiers..."
        );
        let fixture = tiers::block_on(tiers::supertable_storage_fixture());
        let producer = build_supertable_on_storage(fixture.storage.clone());
        eprintln!(
            "[supertable_all] object-store commit OK: manifest_id={} ({})",
            producer.manifest_id(),
            fixture.storage_label
        );
        drop(producer);
        S3VectorCommitted {
            storage: fixture.storage,
            storage_label: fixture.storage_label,
        }
    })
}

/// Run a supertable kNN search and resolve per-superfile hits to
/// global doc-ids via cumulative-`n_docs` offsets in manifest order.
/// `commit()` row-shards into `min(writer_pool.threads, total_rows)`
/// superfiles, so the bench can't assume "one superfile per append
/// batch." Prefix-sum gives the global base for each superfile.
fn supertable_topk(
    st: &Supertable,
    query: &[f32],
    k: usize,
    options: VectorSearchOptions,
) -> Vec<u32> {
    let r = st.reader();
    let hits: Vec<SuperfileHit> =
        corpus::block_on_inmem(r.vector_search(VEC_COLUMN, query, k, options))
            .expect("vector_search");
    let manifest = r.manifest();
    let mut offsets: Vec<u32> = Vec::with_capacity(manifest.superfiles.len());
    let mut acc: u32 = 0;
    for entry in manifest.superfiles.iter() {
        offsets.push(acc);
        acc = acc.saturating_add(entry.n_docs as u32);
    }
    hits.into_iter()
        .map(|h| {
            let seg_idx = manifest
                .superfiles
                .iter()
                .position(|e| e.uri == h.segment)
                .expect("superfile in manifest");
            offsets[seg_idx] + h.local_doc_id
        })
        .collect()
}

async fn supertable_topk_async(
    st: &Supertable,
    query: &[f32],
    k: usize,
    options: VectorSearchOptions,
) -> Vec<u32> {
    let r = st.reader();
    let hits: Vec<SuperfileHit> = r
        .vector_search(VEC_COLUMN, query, k, options)
        .await
        .expect("vector_search");
    let manifest = r.manifest();
    let mut offsets: Vec<u32> = Vec::with_capacity(manifest.superfiles.len());
    let mut acc: u32 = 0;
    for entry in manifest.superfiles.iter() {
        offsets.push(acc);
        acc = acc.saturating_add(entry.n_docs as u32);
    }
    hits.into_iter()
        .map(|h| {
            let seg_idx = manifest
                .superfiles
                .iter()
                .position(|e| e.uri == h.segment)
                .expect("superfile in manifest");
            offsets[seg_idx] + h.local_doc_id
        })
        .collect()
}

fn mean_recall_supertable(
    st: &Supertable,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
    options: VectorSearchOptions,
) -> f32 {
    let mut sum = 0f32;
    for (q, t) in queries.iter().zip(truths) {
        let hits = supertable_topk(st, q, TOP_K, options);
        let truth_set: std::collections::HashSet<u32> = t.iter().copied().collect();
        let recall = if t.is_empty() {
            1.0
        } else {
            hits.iter().filter(|id| truth_set.contains(id)).count() as f32 / t.len() as f32
        };
        sum += recall;
    }
    sum / queries.len() as f32
}

// ─── Correctness ──────────────────────────────────────────────────────

fn assert_supertable_self_consistent(st: &Supertable) -> f32 {
    let opts = VectorSearchOptions::new()
        .with_nprobe(CORRECTNESS_SUPERTABLE_NPROBE)
        .with_rerank_mult(CORRECTNESS_SUPERTABLE_RERANK_MULT);
    let mean_recall =
        mean_recall_supertable(st, queries_correctness(), ground_truth_correctness(), opts);
    assert!(
        mean_recall >= CORRECTNESS_RECALL_FLOOR,
        "supertable mean recall@{TOP_K} at correctness config \
         (p={CORRECTNESS_SUPERTABLE_NPROBE}, r={CORRECTNESS_SUPERTABLE_RERANK_MULT}) \
         below floor: {mean_recall:.3} < {CORRECTNESS_RECALL_FLOOR:.3}"
    );
    mean_recall
}

fn assert_fts_self_consistent(st: &Supertable) {
    let r = st.reader();
    let probe_doc_id = (N_DOCS / 2) as u32;
    let probe_token = format!("doc{probe_doc_id:07}");
    let hits =
        corpus::block_on_inmem(r.bm25_search(TEXT_COLUMN, &probe_token, TOP_K, BoolMode::Or))
            .expect("bm25");
    assert_eq!(
        hits.len(),
        1,
        "df=1 token {probe_token:?} should return exactly one hit; got {}",
        hits.len()
    );
    assert!(
        hits[0].score > 0.0,
        "df=1 score must be positive; got {}",
        hits[0].score
    );

    let hits = corpus::block_on_inmem(r.bm25_search(TEXT_COLUMN, "term00001", TOP_K, BoolMode::Or))
        .expect("bm25");
    assert_eq!(hits.len(), TOP_K, "common term should fill top-{TOP_K}");
    for w in hits.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "results must be sorted by score desc; got {} then {}",
            w[0].score,
            w[1].score
        );
    }
}

// ─── Calibration ──────────────────────────────────────────────────────

struct Calibrations {
    supertable: [Option<Calibrated>; 3],
}

fn calibrate_supertable_at_target(
    st: &Supertable,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
    target_recall: f32,
) -> Option<Calibrated> {
    let mut best: Option<Calibrated> = None;
    let mut peak_recall = 0f32;
    for &probe in SUPERTABLE_PROBES_PER_SEG {
        for &refine in SUPERTABLE_REFINES {
            let opts = VectorSearchOptions::new()
                .with_nprobe(probe)
                .with_rerank_mult(refine);
            let recall = mean_recall_supertable(st, queries, truths, opts);
            if recall > peak_recall {
                peak_recall = recall;
            }
            if recall < target_recall {
                continue;
            }
            let q = &queries[0];
            let n_iter = 21;
            let mut samples = Vec::with_capacity(n_iter);
            for _ in 0..n_iter {
                let t0 = Instant::now();
                let _ = supertable_topk(st, q, TOP_K, opts);
                samples.push(t0.elapsed().as_secs_f32() * 1e6);
            }
            samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p50 = samples[samples.len() / 2];
            let cand = Calibrated {
                probe,
                refine,
                recall,
                p50_micros: p50,
            };
            best = match best {
                None => Some(cand),
                Some(b) if cand.p50_micros < b.p50_micros => Some(cand),
                Some(b) => Some(b),
            };
        }
    }
    if best.is_none() {
        eprintln!(
            "    [supertable] no point hit recall ≥ {target_recall:.2}; peak observed = {peak_recall:.3}"
        );
    }
    best
}

fn calibrations() -> &'static Calibrations {
    CALIBRATIONS.get_or_init(|| {
        let st = ensure_supertable("vector calibration");
        let qs = queries_calibration();
        let gt = ground_truth_calibration();

        eprintln!(
            "[supertable_vec_search] calibrating vector search on shared combined supertable at recall targets {RECALL_TARGETS:?}..."
        );
        let mut s: [Option<Calibrated>; 3] = [None; 3];
        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            s[i] = calibrate_supertable_at_target(st, qs, gt, target);
            eprintln!("  recall ≥ {target:.2} | vector search: {:?}", s[i]);
        }
        Calibrations { supertable: s }
    })
}

// ─── Bench: ingest (group: supertable_all_build) ──────────────────────
//
// Build once, test many times. The 10M × 384 supertable is far too
// costly to resample, so this group builds it exactly once into the
// shared `SUPERTABLE` fixture and reports that single measured build
// for every criterion iteration (via `iter_custom`). Every later phase
// — correctness, calibration, and hot/warm/cold search — reuses that
// same instance, so only one 10M supertable is ever resident (which
// also keeps peak RSS to a single build, not two simultaneous corpora).
// Registered ahead of `bench_search` in `criterion_group!` so the build
// happens here, inside this group's RSS window.

fn bench_ingest(c: &mut Criterion) {
    let _ = vectors();
    let _ = text_corpus();

    let mut g = c.benchmark_group("supertable_all_build");
    g.sample_size(10);
    g.measurement_time(BUILD_MEASUREMENT_TIME);
    g.throughput(Throughput::Elements(N_DOCS as u64));

    let rss_sample = rss::PeakSampler::start_default();
    let bench_id = format!("supertable_{N_DOCS}docs_{N_SEGMENTS}superfiles");
    g.bench_function(bench_id.clone(), |b| {
        b.iter_custom(|iters| {
            // First call builds the shared supertable once (slow) and
            // records its wall-clock time; subsequent calls hit the
            // initialized `OnceLock` and return the same measurement
            // without rebuilding.
            let _ = ensure_supertable("ingest timing");
            let ns = *SUPERTABLE_BUILD_NS
                .get()
                .expect("supertable build time recorded on first build");
            Duration::from_nanos(ns as u64) * (iters as u32)
        });
    });

    g.finish();
    let stats = rss_sample.stop_stats();
    let _ = rss::write_rss_stats(group_name::SUPERTABLE_ALL_BUILD, &bench_id, stats);

    emit_ingest_markdown();
}

// ─── Bench: search (group: supertable_vec_search) ─────────────────────

fn bench_search(c: &mut Criterion) {
    let st = ensure_supertable("vector correctness/search");
    eprintln!(
        "[supertable_vec] correctness: using shared combined FTS + vector supertable ({N_DOCS} docs × {N_SEGMENTS} superfiles)..."
    );
    let recall = assert_supertable_self_consistent(st);
    eprintln!(
        "[supertable_vec] correctness OK: vector recall@{TOP_K} = {recall:.3} (≥ {:.2})",
        CORRECTNESS_RECALL_FLOOR
    );

    let cal = calibrations();
    let qs = queries_calibration();

    let mut g = c.benchmark_group(tiers::search_group_name("supertable_vec", Tier::Hot, None));
    g.sample_size(10);
    let rss_sample = rss::PeakSampler::start_default();

    for (i, &target) in RECALL_TARGETS.iter().enumerate() {
        let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
        if let Some(c_st) = cal.supertable[i] {
            // Stable bench id: the calibrated (probe, refine) lives in
            // the markdown table, not the criterion id, so criterion can
            // match this row against its prior baseline and print its
            // own improved/regressed delta on subsequent runs.
            let (p, r) = (c_st.probe, c_st.refine);
            g.bench_function(format!("supertable_{label}"), |b| {
                let q = &qs[0];
                let opts = VectorSearchOptions::new()
                    .with_nprobe(p)
                    .with_rerank_mult(r);
                b.iter(|| {
                    let hits = supertable_topk(st, black_box(q), TOP_K, opts);
                    black_box(hits)
                });
            });
        }
    }

    g.finish();
    let stats = rss_sample.stop_stats();
    for (i, &target) in RECALL_TARGETS.iter().enumerate() {
        let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
        if cal.supertable[i].is_some() {
            let bid = format!("supertable_{label}");
            let _ = rss::write_rss_stats(group_name::SUPERTABLE_VEC_SEARCH, &bid, stats);
        }
    }

    bench_search_object_store_tiers(c, &cal, qs);

    emit_search_markdown();
}

fn bench_fts_search(c: &mut Criterion) {
    let st = ensure_supertable("FTS correctness/search");

    eprintln!("[supertable_fts_search] correctness check on shared combined supertable...");
    assert_fts_self_consistent(st);
    eprintln!("[supertable_fts_search] correctness OK: infino self-consistent");

    let r = st.reader();

    let queries: &[(&str, &str)] = &[
        ("single_rare", "term09999"),
        ("single_common", "term00001"),
        ("two_term_or", "term00001 term00050"),
        ("three_wide_or", "term00001 term00050 term00100"),
        ("three_similar_or", "term00050 term00051 term00052"),
        (
            "five_term_or",
            "term00050 term00051 term00052 term00053 term00054",
        ),
    ];

    let mut g = c.benchmark_group(tiers::search_group_name("supertable_fts", Tier::Hot, None));
    g.sample_size(10);
    let rss_sample = rss::PeakSampler::start_default();

    for (name, q) in queries {
        let name = *name;
        let q = *q;
        g.bench_function(format!("{name}_supertable_top10"), |b| {
            b.iter(|| {
                let hits = corpus::block_on_inmem(r.bm25_search(
                    black_box(TEXT_COLUMN),
                    black_box(q),
                    TOP_K,
                    BoolMode::Or,
                ))
                .expect("bm25");
                black_box(hits)
            });
        });
    }

    g.bench_function("prefix_supertable_top10", |b| {
        b.iter(|| {
            let hits = corpus::block_on_inmem(r.bm25_search_prefix(
                black_box(TEXT_COLUMN),
                black_box("term0009"),
                TOP_K,
            ))
            .expect("bm25_prefix");
            black_box(hits)
        });
    });

    g.finish();
    let stats = rss_sample.stop_stats();
    for bid in FTS_SEARCH_IDS {
        let _ = rss::write_rss_stats(group_name::SUPERTABLE_FTS_SEARCH, bid, stats);
    }

    bench_fts_search_object_store_tiers(c, queries);
    emit_fts_search_markdown();
}

/// Warm/cold search over the 10M supertable on object storage
/// (`with_storage` + `with_disk_cache`). Hot rows stay in-memory above.
fn bench_search_object_store_tiers(c: &mut Criterion, cal: &Calibrations, qs: &[Vec<f32>]) {
    let committed = s3_vector_committed();
    let q = &qs[0];

    for tier in [Tier::Warm, Tier::Cold] {
        let mut g = c.benchmark_group(tiers::search_group_name(
            "supertable_vec",
            tier,
            Some(committed.storage_label),
        ));
        g.sample_size(10);
        // Cold rebuilds a fresh cache + full S3 cold open per sample; widen
        // only the cold groups so criterion stops warning it can't fit 10
        // samples in the 5s default (warm/hot are sub-ms).
        if tier == Tier::Cold {
            g.measurement_time(Duration::from_secs(30));
        }

        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            let Some(c_st) = cal.supertable[i] else {
                continue;
            };
            let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
            let (p, r) = (c_st.probe, c_st.refine);
            let opts = VectorSearchOptions::new()
                .with_nprobe(p)
                .with_rerank_mult(r);
            let bench_id = format!("supertable_{label}");

            match tier {
                Tier::Warm => {
                    let storage = Arc::clone(&committed.storage);
                    let (cache_dir, cache) = tiers::fresh_disk_cache(storage.clone());
                    let consumer_opts = tiers::consumer_options(
                        vector_supertable_options(None),
                        storage,
                        cache.clone(),
                    );
                    let st = tiers::block_on(tiers::open_consumer(consumer_opts));
                    let query = q.clone();
                    tiers::block_on(async {
                        let _ = supertable_topk_async(&st, &query, TOP_K, opts).await;
                        st.wait_until_warm(Duration::from_secs(600))
                            .await
                            .expect("supertable warm promotion");
                    });
                    g.bench_function(&bench_id, |b| {
                        let query = q.clone();
                        b.iter(|| {
                            let hits = tiers::block_on(supertable_topk_async(
                                &st,
                                black_box(&query),
                                TOP_K,
                                opts,
                            ));
                            black_box(hits)
                        });
                    });
                    drop(st);
                    drop(cache);
                    drop(cache_dir);
                }
                Tier::Cold => {
                    let storage = Arc::clone(&committed.storage);
                    let query = q.clone();
                    g.bench_function(&bench_id, |b| {
                        b.iter_custom(|iters| {
                            let mut total = Duration::ZERO;
                            for _ in 0..iters {
                                let (cache_dir, cache) =
                                    tiers::fresh_disk_cache(Arc::clone(&storage));
                                let consumer_opts = tiers::consumer_options(
                                    vector_supertable_options(None),
                                    Arc::clone(&storage),
                                    cache.clone(),
                                );
                                let t0 = Instant::now();
                                tiers::block_on(async {
                                    let st = tiers::open_consumer(consumer_opts).await;
                                    let _ = supertable_topk_async(&st, &query, TOP_K, opts).await;
                                });
                                total += t0.elapsed();
                                drop(cache);
                                drop(cache_dir);
                            }
                            total
                        });
                    });
                }
                Tier::Hot => {}
            }
        }
        g.finish();
    }
}

fn bench_fts_search_object_store_tiers(c: &mut Criterion, queries: &[(&str, &str)]) {
    let committed = s3_vector_committed();

    for tier in [Tier::Warm, Tier::Cold] {
        let mut g = c.benchmark_group(tiers::search_group_name(
            "supertable_fts",
            tier,
            Some(committed.storage_label),
        ));
        g.sample_size(10);
        if tier == Tier::Cold {
            g.measurement_time(Duration::from_secs(30));
        }

        for (name, q) in queries {
            let bench_id = format!("{name}_supertable_top10");
            match tier {
                Tier::Warm => {
                    let storage = Arc::clone(&committed.storage);
                    let (cache_dir, cache) = tiers::fresh_disk_cache(storage.clone());
                    let consumer_opts = tiers::consumer_options(
                        vector_supertable_options(None),
                        storage,
                        cache.clone(),
                    );
                    let st = tiers::block_on(tiers::open_consumer(consumer_opts));
                    let query = *q;
                    tiers::block_on(async {
                        let _ = st
                            .reader()
                            .bm25_search(TEXT_COLUMN, query, TOP_K, BoolMode::Or)
                            .await
                            .expect("warm prewarm bm25");
                        st.wait_until_warm(Duration::from_secs(600))
                            .await
                            .expect("supertable warm promotion");
                    });
                    g.bench_function(&bench_id, |b| {
                        b.iter(|| {
                            let hits = tiers::block_on(async {
                                st.reader()
                                    .bm25_search(TEXT_COLUMN, q, TOP_K, BoolMode::Or)
                                    .await
                                    .expect("bm25")
                            });
                            black_box(hits)
                        });
                    });
                    drop(st);
                    drop(cache);
                    drop(cache_dir);
                }
                Tier::Cold => {
                    let storage = Arc::clone(&committed.storage);
                    let query = *q;
                    g.bench_function(&bench_id, |b| {
                        b.iter_custom(|iters| {
                            let mut total = Duration::ZERO;
                            for _ in 0..iters {
                                let (cache_dir, cache) =
                                    tiers::fresh_disk_cache(Arc::clone(&storage));
                                let consumer_opts = tiers::consumer_options(
                                    vector_supertable_options(None),
                                    Arc::clone(&storage),
                                    cache.clone(),
                                );
                                let t0 = Instant::now();
                                tiers::block_on(async {
                                    let st = tiers::open_consumer(consumer_opts).await;
                                    let _ = st
                                        .reader()
                                        .bm25_search(TEXT_COLUMN, query, TOP_K, BoolMode::Or)
                                        .await
                                        .expect("cold bm25");
                                });
                                total += t0.elapsed();
                                drop(cache);
                                drop(cache_dir);
                            }
                            total
                        });
                    });
                }
                Tier::Hot => {}
            }
        }

        g.bench_function("prefix_supertable_top10", |b| match tier {
            Tier::Warm => {
                let storage = Arc::clone(&committed.storage);
                let (cache_dir, cache) = tiers::fresh_disk_cache(storage.clone());
                let consumer_opts = tiers::consumer_options(
                    vector_supertable_options(None),
                    storage,
                    cache.clone(),
                );
                let st = tiers::block_on(tiers::open_consumer(consumer_opts));
                tiers::block_on(async {
                    let _ = st
                        .reader()
                        .bm25_search_prefix(TEXT_COLUMN, "term0009", TOP_K)
                        .await
                        .expect("warm prewarm prefix");
                    st.wait_until_warm(Duration::from_secs(600))
                        .await
                        .expect("supertable warm promotion");
                });
                b.iter(|| {
                    let hits = tiers::block_on(async {
                        st.reader()
                            .bm25_search_prefix(TEXT_COLUMN, "term0009", TOP_K)
                            .await
                            .expect("prefix")
                    });
                    black_box(hits)
                });
                drop(st);
                drop(cache);
                drop(cache_dir);
            }
            Tier::Cold => {
                let storage = Arc::clone(&committed.storage);
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let (cache_dir, cache) = tiers::fresh_disk_cache(Arc::clone(&storage));
                        let consumer_opts = tiers::consumer_options(
                            vector_supertable_options(None),
                            Arc::clone(&storage),
                            cache.clone(),
                        );
                        let t0 = Instant::now();
                        tiers::block_on(async {
                            let st = tiers::open_consumer(consumer_opts).await;
                            let _ = st
                                .reader()
                                .bm25_search_prefix(TEXT_COLUMN, "term0009", TOP_K)
                                .await
                                .expect("cold prefix");
                        });
                        total += t0.elapsed();
                        drop(cache);
                        drop(cache_dir);
                    }
                    total
                });
            }
            Tier::Hot => {}
        });

        g.finish();
    }
}

// ─── Markdown summary emitters ────────────────────────────────────────

mod group_name {
    pub const SUPERTABLE_ALL_BUILD: &str = "supertable_all_build";
    pub const SUPERTABLE_VEC_SEARCH: &str = "supertable_vec_hot_search";
    pub const SUPERTABLE_FTS_SEARCH: &str = "supertable_fts_hot_search";
}

fn emit_ingest_markdown() {
    use markdown::{MarkdownSection, fmt_throughput, fmt_time, read_mean_ns};

    let group = group_name::SUPERTABLE_ALL_BUILD;
    let bench = format!("supertable_{N_DOCS}docs_{N_SEGMENTS}superfiles");
    let ns = read_mean_ns(group, &bench);
    let peak_rss = rss::read_peak_rss_bytes(group, &bench);

    let mut body = String::new();
    body.push_str(&format!(
        "### Supertable combined FTS + vector — ingest ({N_DOCS} docs × dim={DIM}, sharded into {N_SEGMENTS} superfiles)\n\n"
    ));
    body.push_str(
        "| Engine | Time | Throughput | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |\n",
    );
    body.push_str(
        "|--------|------|------------|----------|------------|---------|------------|\n",
    );
    let time = ns.map(fmt_time).unwrap_or_else(|| "—".into());
    let thrpt = ns
        .map(|n| fmt_throughput((N_DOCS as f64) / (n / 1e9)))
        .unwrap_or_else(|| "—".into());
    let rss_cell = peak_rss.map(rss::fmt_bytes).unwrap_or_else(|| "—".into());
    let median_rss = rss::fmt_median_rss(group, &bench);
    let p90_rss = rss::fmt_p90_rss(group, &bench);
    let rss_delta = rss::fmt_peak_rss_delta(group, &bench);
    body.push_str(&format!(
        "| supertable | {time} | {thrpt} | {rss_cell} | {median_rss} | {p90_rss} | {rss_delta} |\n"
    ));

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/vector/supertable/ingest".into(),
        body,
    });
}

fn emit_search_markdown() {
    use markdown::{MarkdownSection, fmt_time, read_mean_ns};

    let cal = calibrations();
    let group = group_name::SUPERTABLE_VEC_SEARCH;

    let mut body = String::new();
    body.push_str(&format!(
        "### Supertable vector — search ({N_DOCS} docs × dim={DIM}, calibrated at recall targets)\n\n"
    ));
    body.push_str(
        "Hot = in-memory; warm/cold = object storage + disk cache (s3s-fs or `INFINO_REAL_S3_BUCKET`).\n\n",
    );
    body.push_str(
        "| Recall target | (p/seg, r) | hot | warm | cold | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |\n",
    );
    body.push_str(
        "|---------------|------------|-----|------|------|----------|------------|---------|------------|\n",
    );

    for (i, &target) in RECALL_TARGETS.iter().enumerate() {
        let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
        let row_target = format!("{target:.2}");
        let bid = format!("supertable_{label}");
        let (cell, hot, warm, cold, rss_cell, median_rss, p90_rss, rss_delta) =
            match cal.supertable[i] {
                Some(c) => {
                    let peak = rss::read_peak_rss_bytes(group, &bid);
                    (
                        format!("(p={}, r={})", c.probe, c.refine),
                        read_mean_ns(group, &bid),
                        markdown::read_tier_mean_ns("supertable_vec", "warm", &bid),
                        markdown::read_tier_mean_ns("supertable_vec", "cold", &bid),
                        peak.map(rss::fmt_bytes).unwrap_or_else(|| "—".into()),
                        rss::fmt_median_rss(group, &bid),
                        rss::fmt_p90_rss(group, &bid),
                        rss::fmt_peak_rss_delta(group, &bid),
                    )
                }
                None => (
                    "—".into(),
                    None,
                    None,
                    None,
                    "—".into(),
                    "—".into(),
                    "—".into(),
                    "—".into(),
                ),
            };
        body.push_str(&format!(
            "| {row_target} | {cell} | {} | {} | {} | {rss_cell} | {median_rss} | {p90_rss} | {rss_delta} |\n",
            hot.map(fmt_time).unwrap_or_else(|| "—".into()),
            warm.map(fmt_time).unwrap_or_else(|| "—".into()),
            cold.map(fmt_time).unwrap_or_else(|| "—".into()),
        ));
    }

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/vector/supertable/search".into(),
        body,
    });
}

fn emit_fts_search_markdown() {
    use markdown::{MarkdownSection, fmt_time, read_mean_ns};

    let group = group_name::SUPERTABLE_FTS_SEARCH;
    let mut body = String::new();
    body.push_str(&format!(
        "### Supertable FTS — search ({N_DOCS} docs, shared combined supertable)\n\n"
    ));
    body.push_str("Hot = in-memory; warm/cold = object storage + disk cache.\n\n");
    body.push_str(
        "| Query          | hot        | warm       | cold       | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |\n",
    );
    body.push_str(
        "|----------------|------------|------------|------------|-----------|------------|-----------|------------|\n",
    );

    let rows = [
        "single_rare",
        "single_common",
        "two_term_or",
        "three_wide_or",
        "three_similar_or",
        "five_term_or",
        "prefix",
    ];
    for q in rows {
        let bid = format!("{q}_supertable_top10");
        let hot = read_mean_ns(group, &bid);
        let warm = markdown::read_tier_mean_ns("supertable_fts", "warm", &bid);
        let cold = markdown::read_tier_mean_ns("supertable_fts", "cold", &bid);
        let rss_cell = rss::read_peak_rss_bytes(group, &bid)
            .map(rss::fmt_bytes)
            .unwrap_or_else(|| "—".into());
        let median_rss = rss::fmt_median_rss(group, &bid);
        let p90_rss = rss::fmt_p90_rss(group, &bid);
        let rss_delta = rss::fmt_peak_rss_delta(group, &bid);
        body.push_str(&format!(
            "| {q:14} | {} | {} | {} | {rss_cell:9} | {median_rss:10} | {p90_rss:9} | {rss_delta:10} |\n",
            hot.map(fmt_time).unwrap_or_else(|| "—".into()),
            warm.map(fmt_time).unwrap_or_else(|| "—".into()),
            cold.map(fmt_time).unwrap_or_else(|| "—".into()),
        ));
    }

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/fts/supertable/search".into(),
        body,
    });
}

criterion_group!(benches, bench_ingest, bench_search, bench_fts_search);
