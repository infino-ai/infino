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
//! cargo bench --bench vector -- supertable_vec_build       # ingest only
//! cargo bench --bench vector -- supertable_vec_search      # search only
//! ```

use std::hint::black_box;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::corpus::{self, Calibrated, DIM};
use crate::{markdown, rss};
use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, Throughput, criterion_group};
use infino::superfile::builder::VectorConfig;
use infino::superfile::vector::distance::Metric;
use infino::supertable::query::SuperfileHit;
use infino::supertable::query::vector::VectorSearchOptions;
use infino::supertable::{Supertable, SupertableOptions};

// ─── Constants ────────────────────────────────────────────────────────

/// Doc count for every vector-supertable bench. Pinned to 10M — the
/// supertable is the scale-out shape.
const N_DOCS: usize = 10_000_000;

const N_SEGMENTS: usize = 4;
const TOP_K: usize = 10;

const N_CORRECTNESS_QUERIES: usize = 20;
const N_CALIBRATION_QUERIES: usize = 100;

const RECALL_TARGETS: &[f32] = &[0.90, 0.95, 0.99];
const BUILD_MEASUREMENT_TIME: Duration = Duration::from_secs(60 * 60);

/// Per-segment probe grid. The supertable applies a single `nprobe`
/// to every segment, so the effective probe count is `nprobe ×
/// n_superfiles`. This grid approximates the single-superfile probe
/// grid scaled down by [`N_SEGMENTS`].
const SUPERTABLE_PROBES_PER_SEG: &[usize] = &[1, 2, 4, 8, 12, 16];
const SUPERTABLE_REFINES: &[usize] = &[4, 16, 64, 256, 1024];

const CORRECTNESS_RECALL_FLOOR: f32 = 0.80;
const CORRECTNESS_SUPERTABLE_NPROBE: usize = 16;
const CORRECTNESS_SUPERTABLE_RERANK_MULT: usize = 256;

// ─── Fixtures ────────────────────────────────────────────────────────

static VECTORS: OnceLock<corpus::MmapCorpus> = OnceLock::new();
static QUERIES_CORRECTNESS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
static QUERIES_CALIBRATION: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
static GROUND_TRUTH_CORRECTNESS: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
static GROUND_TRUTH_CALIBRATION: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
static SUPERTABLE: OnceLock<Supertable> = OnceLock::new();
static CALIBRATIONS: OnceLock<Calibrations> = OnceLock::new();

fn vectors() -> &'static [f32] {
    VECTORS
        .get_or_init(|| {
            // Raw corpus fixture only. Ingestion below still goes through
            // RecordBatch -> writer.append() -> commit(); the mmap prevents
            // the synthetic 10M x 384 source corpus from living as heap RAM.
            corpus::MmapCorpus::generate(N_DOCS, corpus::n_cent(N_DOCS), 1, true)
        })
        .as_slice()
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
    SUPERTABLE.get_or_init(build_supertable)
}

// ─── Builder ──────────────────────────────────────────────────────────

fn supertable_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "emb",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            DIM as i32,
        ),
        false,
    )]))
}

fn build_supertable() -> Supertable {
    let n_cent_total = corpus::n_cent(N_DOCS);
    let n_cent_per_segment = (n_cent_total / N_SEGMENTS).max(1);

    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get().max(1))
            .build()
            .expect("pool"),
    );
    let opts = SupertableOptions::new(
        supertable_schema(),
        vec![],
        vec![VectorConfig {
            column: "emb".into(),
            dim: DIM,
            n_cent: n_cent_per_segment,
            rot_seed: 7,
            metric: Metric::Cosine,
        }],
        None,
    )
    .expect("opts")
    .with_writer_pool(pool);

    let st = Supertable::create(opts);
    let mut w = st.writer().expect("writer");

    let chunk_size = N_DOCS / N_SEGMENTS;
    let v = vectors();
    for chunk_idx in 0..N_SEGMENTS {
        let start = chunk_idx * chunk_size;
        let end = start + chunk_size;
        // Materialize one append batch from the disk-backed corpus, then feed
        // it through the normal production ingestion API.
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
        let batch = RecordBatch::try_new(supertable_schema(), vec![Arc::new(fsl)]).expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    st
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
    let hits: Vec<SuperfileHit> = r
        .vector_search("emb", query, k, options)
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
        let st = supertable();
        let qs = queries_calibration();
        let gt = ground_truth_calibration();

        eprintln!(
            "[supertable_vec_search] calibrating supertable at recall targets {RECALL_TARGETS:?}..."
        );
        let mut s: [Option<Calibrated>; 3] = [None; 3];
        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            s[i] = calibrate_supertable_at_target(st, qs, gt, target);
            eprintln!("  recall ≥ {target:.2} | supertable: {:?}", s[i]);
        }
        Calibrations { supertable: s }
    })
}

// ─── Bench entry ──────────────────────────────────────────────────────

fn bench(c: &mut Criterion) {
    eprintln!(
        "[supertable_vec] correctness: building supertable ({N_DOCS} docs × {N_SEGMENTS} superfiles)..."
    );
    let st = supertable();
    let recall = assert_supertable_self_consistent(st);
    eprintln!(
        "[supertable_vec] correctness OK: supertable recall@{TOP_K} = {recall:.3} (≥ {:.2})",
        CORRECTNESS_RECALL_FLOOR
    );

    // ---- Ingest sub-bench (group: supertable_vec_build) ------------
    {
        let v = vectors();
        let mut g = c.benchmark_group("supertable_vec_build");
        g.sample_size(10);
        g.measurement_time(BUILD_MEASUREMENT_TIME);
        g.throughput(Throughput::Elements(N_DOCS as u64));

        let rss_sample = rss::PeakSampler::start_default();
        let bench_id = format!("supertable_{N_DOCS}docs_{N_SEGMENTS}superfiles");
        g.bench_function(bench_id.clone(), |b| {
            b.iter_with_large_drop(|| {
                let _ = black_box(v);
                build_supertable()
            });
        });

        g.finish();
        let peak = rss_sample.stop();
        let _ = rss::write_peak_rss(group_name::SUPERTABLE_VEC_BUILD, &bench_id, peak);

        emit_ingest_markdown();
    }

    // ---- Search sub-bench (group: supertable_vec_search) -----------
    {
        let cal = calibrations();
        let qs = queries_calibration();

        let mut g = c.benchmark_group("supertable_vec_search");
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
        let peak = rss_sample.stop();
        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
            if cal.supertable[i].is_some() {
                let bid = format!("supertable_{label}");
                let _ = rss::write_peak_rss(group_name::SUPERTABLE_VEC_SEARCH, &bid, peak);
            }
        }

        emit_search_markdown();
    }
}

// ─── Markdown summary emitters ────────────────────────────────────────

mod group_name {
    pub const SUPERTABLE_VEC_BUILD: &str = "supertable_vec_build";
    pub const SUPERTABLE_VEC_SEARCH: &str = "supertable_vec_search";
}

fn emit_ingest_markdown() {
    use markdown::{MarkdownSection, fmt_throughput, fmt_time, read_mean_ns};

    let group = group_name::SUPERTABLE_VEC_BUILD;
    let bench = format!("supertable_{N_DOCS}docs_{N_SEGMENTS}superfiles");
    let ns = read_mean_ns(group, &bench);
    let peak_rss = rss::read_peak_rss_bytes(group, &bench);

    let mut body = String::new();
    body.push_str(&format!(
        "### Supertable vector — ingest ({N_DOCS} docs × dim={DIM}, sharded into {N_SEGMENTS} superfiles)\n\n"
    ));
    body.push_str("| Engine | Time | Throughput | Peak RSS | Peak RSS Δ |\n");
    body.push_str("|--------|------|------------|----------|------------|\n");
    let time = ns.map(fmt_time).unwrap_or_else(|| "—".into());
    let thrpt = ns
        .map(|n| fmt_throughput((N_DOCS as f64) / (n / 1e9)))
        .unwrap_or_else(|| "—".into());
    let rss_cell = peak_rss.map(rss::fmt_bytes).unwrap_or_else(|| "—".into());
    let rss_delta = rss::fmt_peak_rss_delta(group, &bench);
    body.push_str(&format!(
        "| supertable | {time} | {thrpt} | {rss_cell} | {rss_delta} |\n"
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
        "| Recall target | supertable (probe/seg, refine) | supertable p50 | Peak RSS | Peak RSS Δ |\n",
    );
    body.push_str(
        "|---------------|--------------------------------|----------------|----------|------------|\n",
    );

    for (i, &target) in RECALL_TARGETS.iter().enumerate() {
        let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
        let row_target = format!("{target:.2}");
        let (cell, ns, rss_cell, rss_delta) = match cal.supertable[i] {
            Some(c) => {
                let bid = format!("supertable_{label}");
                let ns = read_mean_ns(group, &bid);
                let peak = rss::read_peak_rss_bytes(group, &bid);
                let rss_cell = peak.map(rss::fmt_bytes).unwrap_or_else(|| "—".into());
                let rss_delta = rss::fmt_peak_rss_delta(group, &bid);
                (
                    format!("(p={}, r={})", c.probe, c.refine),
                    ns,
                    rss_cell,
                    rss_delta,
                )
            }
            None => ("—".into(), None, "—".into(), "—".into()),
        };
        let t = ns.map(fmt_time).unwrap_or_else(|| "—".into());
        body.push_str(&format!(
            "| {row_target} | {cell} | {t} | {rss_cell} | {rss_delta} |\n"
        ));
    }

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/vector/supertable/search".into(),
        body,
    });
}

criterion_group!(benches, bench);
