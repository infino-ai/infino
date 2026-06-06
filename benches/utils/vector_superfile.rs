//! Infino-only vector bench for the superfile layer:
//!
//!   ingest timing (1M × 384 Gaussian planted clusters, cosine)
//! + calibrated kNN search at recall targets {0.90, 0.95, 0.99}
//! + nprobe/rerank sweeps
//! + correctness gate (`recall@10 ≥ 0.80` at high-recall config)
//!
//! Every phase uses the production path: [`SuperfileBuilder`] →
//! [`SuperfileReader`] → [`SuperfileReader::vector_search`]. Hot
//! opens the finished `.parquet` in memory; cold commits the same bytes
//! to object storage and reads through [`DiskCacheStore::reader`].
//!
//! Pinned to 1M × 384. Supertable scale (10M × 384, sharded into N
//! superfiles) lives in `benches/vector/supertable.rs`.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench superfile_vector -- superfile_vec_build      # ingest only
//! cargo bench --bench superfile_vector -- superfile_vec_search     # search only
//! ```

use std::hint::black_box;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use std::sync::Arc as ArrowArc;

use crate::tiers::{self, Tier};
use arrow_array::{Decimal128Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use rayon::prelude::*;
// Calibrated `(probe, refine)` is intentionally not in the criterion ID:
// criterion's improved/regressed/noise panel only fires on exact-ID matches
// against the previous run's baseline, and `(probe, refine)` drifts as
// calibration re-picks the lowest-p50 point. The chosen point is reported
// in the markdown table instead.
use crate::corpus::{self, Calibrated, DIM};
use crate::fixture;
use crate::{markdown, rss};
use infino::superfile::SuperfileReader;
use infino::superfile::builder::{BuilderOptions, SuperfileBuilder, VectorConfig};
use infino::superfile::reader::VectorSearchOptions;
use infino::superfile::vector::distance::Metric;
use infino::superfile::vector::rerank_codec::RerankCodec;
use infino::supertable::SuperfileUri;

// ─── Constants ────────────────────────────────────────────────────────

const N_DOCS: usize = corpus::SUPERFILE_DOCS;
const TOP_K: usize = 10;
const N_CORRECTNESS_QUERIES: usize = 20;
const N_CALIBRATION_QUERIES: usize = 100;

/// Recall floor for the correctness gate. Any infino regression that
/// drops below this fails the bench.
const CORRECTNESS_RECALL_FLOOR: f32 = 0.80;

/// High-recall config used as the correctness probe.
const CORRECTNESS_NPROBE: usize = 64;
const CORRECTNESS_RERANK_MULT: usize = 256;

/// Default options for the user-facing "what does it cost in
/// production?" baseline reported in the search markdown.
const DEFAULT_NPROBE: usize = 8;
const DEFAULT_RERANK_MULT: usize = 20;

const RECALL_TARGETS: &[f32] = &[0.90, 0.95, 0.99];

/// (probe, refine) calibration grids. The lowest-p50 point clearing
/// each recall target is what the search table reports.
const PROBES: &[usize] = &[1, 5, 10, 25, 50, 100, 200, 400, 800];
const REFINES: &[usize] = &[1, 4, 16, 64, 256, 1024];

const ID_COLUMN: &str = "doc_id";
const VEC_COLUMN: &str = "v";

// ─── Fixtures ────────────────────────────────────────────────────────

static VECTORS: OnceLock<corpus::MmapVectorCorpus> = OnceLock::new();
static QUERIES_CORRECTNESS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
static QUERIES_CALIBRATION: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
static GROUND_TRUTH_CORRECTNESS: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
static GROUND_TRUTH_CALIBRATION: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
static SUPERFILE_BYTES: OnceLock<Vec<u8>> = OnceLock::new();
static CALIBRATIONS: OnceLock<Calibrations> = OnceLock::new();
static SUPERFILE_OBJECT: OnceLock<tiers::SuperfileCommitted> = OnceLock::new();

fn superfile_bytes() -> &'static [u8] {
    SUPERFILE_BYTES.get_or_init(|| build_superfile_bytes(vectors()))
}

fn superfile_object() -> &'static tiers::SuperfileCommitted {
    SUPERFILE_OBJECT.get_or_init(|| {
        let blob = Bytes::from(superfile_bytes().to_vec());
        eprintln!(
            "[superfile_vec] committing {N_DOCS} docs to object storage for the cold tier \
             (production .parquet, {} MiB)...",
            blob.len() / (1024 * 1024)
        );
        tiers::block_on(tiers::commit_superfile(&blob))
    })
}

fn vectors() -> &'static [f32] {
    VECTORS
        .get_or_init(|| {
            // Raw corpus fixture only. Build/search still exercise Infino's
            // normal vector builder/reader paths; the mmap avoids pinning the
            // synthetic source corpus as heap RAM.
            corpus::MmapVectorCorpus::generate(N_DOCS, corpus::n_cent(N_DOCS), 1, true)
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

fn superfile_reader() -> SuperfileReader {
    SuperfileReader::open(Bytes::from(superfile_bytes().to_vec())).expect("open superfile")
}

fn search_opts(nprobe: usize, rerank_mult: usize) -> VectorSearchOptions {
    VectorSearchOptions::new()
        .with_nprobe(nprobe)
        .with_rerank_mult(rerank_mult)
}

// ─── Builder (production SuperfileBuilder) ───────────────────────────

/// Same path as supertable commit: Arrow id column + vector slices →
/// unified `.parquet` (Parquet body + embedded vector blob + `inf.*` KV).
fn build_superfile_bytes(vectors: &[f32]) -> Vec<u8> {
    build_superfile_bytes_range(vectors, 0, N_DOCS, 0)
}

/// Build one self-contained superfile from the `[start_doc,
/// start_doc + len)` slice of the corpus, assigning doc ids from
/// `id_base`. `n_cent` is sized to this shard's doc count so each shard
/// is a valid standalone superfile — the multi-segment shape supertable
/// commit produces.
fn build_superfile_bytes_range(
    vectors: &[f32],
    start_doc: usize,
    len: usize,
    id_base: usize,
) -> Vec<u8> {
    let n_cent = corpus::n_cent(len);
    let schema = ArrowArc::new(Schema::new(vec![Field::new(
        ID_COLUMN,
        DataType::Decimal128(38, 0),
        false,
    )]));
    let opts = BuilderOptions::new(
        schema.clone(),
        ID_COLUMN,
        vec![],
        vec![VectorConfig {
            column: VEC_COLUMN.into(),
            dim: DIM,
            n_cent,
            rot_seed: 7,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8Residual,
        }],
        None,
    );
    let mut builder = SuperfileBuilder::new(opts).expect("SuperfileBuilder::new");
    const CHUNK: usize = 65_536;
    let mut local = 0;
    while local < len {
        let batch_len = CHUNK.min(len - local);
        let abs_start = start_doc + local;
        let ids: Decimal128Array = ((id_base + local) as u64..(id_base + local + batch_len) as u64)
            .map(|i| Some(i as i128))
            .collect::<Decimal128Array>()
            .with_precision_and_scale(38, 0)
            .expect("decimal128");
        let batch =
            RecordBatch::try_new(schema.clone(), vec![ArrowArc::new(ids)]).expect("RecordBatch");
        builder
            .add_batch(
                &batch,
                &[&vectors[abs_start * DIM..(abs_start + batch_len) * DIM]],
            )
            .expect("add_batch");
        local += batch_len;
    }
    builder.finish().expect("SuperfileBuilder::finish")
}

/// Rayon-sharded parallel build: each shard runs its own
/// [`SuperfileBuilder`] over its slice of the corpus and emits a
/// self-contained `.parquet` — the same multi-segment shape supertable
/// commit produces. Mirrors the FTS bench's `build_superfiles_rayon` so
/// the vector ingest table reports the same 1-thread vs. rayon
/// comparison.
fn build_superfiles_rayon(vectors: &[f32]) -> Vec<Vec<u8>> {
    let n_shards = rayon::current_num_threads();
    let docs_per_shard = N_DOCS.div_ceil(n_shards);
    (0..n_shards)
        .into_par_iter()
        .filter_map(|shard_idx| {
            let start = shard_idx * docs_per_shard;
            if start >= N_DOCS {
                return None;
            }
            let len = docs_per_shard.min(N_DOCS - start);
            Some(build_superfile_bytes_range(vectors, start, len, start))
        })
        .collect()
}

// ─── Correctness ──────────────────────────────────────────────────────

fn assert_infino_self_consistent(reader: &SuperfileReader) -> f32 {
    let qs = queries_correctness();
    let gt = ground_truth_correctness();
    let opts = search_opts(CORRECTNESS_NPROBE, CORRECTNESS_RERANK_MULT);
    let mut total_recall = 0.0_f32;
    for (q, truth) in qs.iter().zip(gt.iter()) {
        let hits = corpus::block_on_inmem(async {
            reader.vector_search(VEC_COLUMN, q, TOP_K, opts).await
        })
        .expect("vector_search");
        assert_eq!(
            hits.len(),
            TOP_K,
            "infino kNN should fill top-{TOP_K}; got {}",
            hits.len()
        );
        total_recall += corpus::recall_at_k(&hits, truth);
    }
    let mean_recall = total_recall / (qs.len() as f32);
    assert!(
        mean_recall >= CORRECTNESS_RECALL_FLOOR,
        "infino mean recall@{TOP_K} at correctness config \
         (p={CORRECTNESS_NPROBE}, r={CORRECTNESS_RERANK_MULT}) \
         below floor: {mean_recall:.3} < {CORRECTNESS_RECALL_FLOOR:.3}"
    );
    mean_recall
}

// ─── Calibration ──────────────────────────────────────────────────────

struct Calibrations {
    infino: [Option<Calibrated>; 3],
}

fn calibrations() -> &'static Calibrations {
    CALIBRATIONS.get_or_init(|| {
        let reader = superfile_reader();
        let qs = queries_calibration();
        let gt = ground_truth_calibration();

        eprintln!(
            "[superfile_vec_search] calibrating superfile vector_search at recall targets {RECALL_TARGETS:?}..."
        );
        let mut inf: [Option<Calibrated>; 3] = [None; 3];
        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            inf[i] = corpus::calibrate_superfile(
                &reader,
                VEC_COLUMN,
                qs,
                gt,
                target,
                PROBES,
                REFINES,
                21,
                TOP_K,
            );
            eprintln!("  recall ≥ {target:.2} | infino: {:?}", inf[i]);
        }
        Calibrations { infino: inf }
    })
}

// ─── Bench entry ──────────────────────────────────────────────────────

fn bench(c: &mut Criterion) {
    let run_build = fixture::supertable::criterion_filter_selects(
        &["superfile_vec", "superfile_vector", "superfile_vec_build"],
        &["superfile_vec_build"],
    );
    let run_search = fixture::supertable::criterion_filter_selects(
        &["superfile_vec", "superfile_vector", "superfile_vec_search"],
        &["superfile_vec_hot_search", "superfile_vec_cold_search"],
    );
    if !run_build && !run_search {
        return;
    }

    // ---- Ingest sub-bench (group: superfile_vec_build) -------------
    if run_build {
        let v = vectors();
        let mut g = c.benchmark_group("superfile_vec_build");
        g.sample_size(10);
        g.throughput(Throughput::Elements(N_DOCS as u64));

        // Two variants, mirroring the FTS ingest bench: a single-thread
        // build (one superfile) and the rayon-sharded build (one
        // superfile per shard). Peak-RSS samplers bound each closure so
        // the recorded RSS reflects that variant, not setup or the other
        // variant.
        let one_thread_id = format!("infino_1thread_{N_DOCS}docs");
        let rayon_id = format!("infino_rayon_default_threads_{N_DOCS}docs");

        let rss_sample = rss::PeakSampler::start_default();
        g.bench_function(one_thread_id.clone(), |b| {
            b.iter_with_large_drop(|| build_superfile_bytes(black_box(v)));
        });
        let stats = rss_sample.stop_stats();
        let _ = rss::write_rss_stats(group_name::SUPERFILE_VEC_BUILD, &one_thread_id, stats);

        let rss_sample = rss::PeakSampler::start_default();
        g.bench_function(rayon_id.clone(), |b| {
            b.iter_with_large_drop(|| build_superfiles_rayon(black_box(v)));
        });
        let stats = rss_sample.stop_stats();
        let _ = rss::write_rss_stats(group_name::SUPERFILE_VEC_BUILD, &rayon_id, stats);

        g.finish();

        emit_ingest_markdown();
    }
    if !run_search {
        return;
    }

    artifact_report(N_DOCS, corpus::n_cent(N_DOCS), vectors());

    eprintln!(
        "[superfile_vec] correctness: building shared superfile for correctness/search ({N_DOCS} docs)..."
    );
    let reader = superfile_reader();
    let recall = assert_infino_self_consistent(&reader);
    eprintln!(
        "[superfile_vec] correctness OK: infino recall@{TOP_K} = {recall:.3} (≥ {:.2})",
        CORRECTNESS_RECALL_FLOOR
    );

    // ---- Search sub-bench (group: superfile_vec_search) ------------
    {
        let cal = calibrations();
        let qs = queries_calibration();

        let mut g = c.benchmark_group(tiers::search_group_name("superfile_vec", Tier::Hot, None));
        g.sample_size(10);
        let rss_sample = rss::PeakSampler::start_default();

        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
            if let Some(c_inf) = cal.infino[i] {
                // Stable bench id: the calibrated (probe, refine) lives in
                // the markdown table, not the criterion id, so criterion can
                // match this row against its prior baseline and print its
                // own improved/regressed delta on subsequent runs.
                let (p, r) = (c_inf.probe, c_inf.refine);
                let opts = search_opts(p, r);
                g.bench_function(format!("infino_{label}"), |b| {
                    let q = &qs[0];
                    b.iter(|| {
                        let hits = corpus::block_on_inmem(async {
                            reader
                                .vector_search(VEC_COLUMN, black_box(q), TOP_K, opts)
                                .await
                        })
                        .expect("vector_search");
                        black_box(hits)
                    });
                });
            }
        }

        // Default-options baseline (what users get with no tuning).
        let q = &qs[0];
        let default_opts = search_opts(DEFAULT_NPROBE, DEFAULT_RERANK_MULT);
        g.bench_function("infino_default_options_top10", |b| {
            b.iter(|| {
                let hits = corpus::block_on_inmem(async {
                    reader
                        .vector_search(VEC_COLUMN, black_box(q), TOP_K, default_opts)
                        .await
                })
                .expect("vector_search");
                black_box(hits)
            });
        });

        // nprobe sweep (rerank fixed at default)
        let n_cent = corpus::n_cent(N_DOCS);
        for &nprobe in &[1, 4, 8, 16, 32, 64, 128] {
            if nprobe > n_cent {
                continue;
            }
            let opts = search_opts(nprobe, DEFAULT_RERANK_MULT);
            g.bench_with_input(
                BenchmarkId::new("infino_nprobe_sweep_rerank20", nprobe),
                &nprobe,
                |b, _| {
                    b.iter(|| {
                        let hits = corpus::block_on_inmem(async {
                            reader
                                .vector_search(VEC_COLUMN, black_box(q), TOP_K, opts)
                                .await
                        })
                        .expect("vector_search");
                        black_box(hits)
                    });
                },
            );
        }

        // rerank_mult sweep (nprobe fixed at default)
        for &rerank in &[1, 5, 10, 20, 50, 100] {
            let opts = search_opts(DEFAULT_NPROBE, rerank);
            g.bench_with_input(
                BenchmarkId::new("infino_rerank_sweep_nprobe8", rerank),
                &rerank,
                |b, _| {
                    b.iter(|| {
                        let hits = corpus::block_on_inmem(async {
                            reader
                                .vector_search(VEC_COLUMN, black_box(q), TOP_K, opts)
                                .await
                        })
                        .expect("vector_search");
                        black_box(hits)
                    });
                },
            );
        }

        g.finish();
        let stats = rss_sample.stop_stats();
        // Single sampler covers every (probe, refine) point in this
        // group; record the same peak against each criterion id so the
        // markdown lookup matches the bench id verbatim.
        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
            if cal.infino[i].is_some() {
                let bid = format!("infino_{label}");
                let _ = rss::write_rss_stats(group_name::SUPERFILE_VEC_SEARCH, &bid, stats);
            }
        }
        let _ = rss::write_rss_stats(
            group_name::SUPERFILE_VEC_SEARCH,
            "infino_default_options_top10",
            stats,
        );

        bench_superfile_vec_storage_tiers(c, cal, qs);

        emit_search_markdown();
    }
}

fn bench_superfile_vec_storage_tiers(c: &mut Criterion, cal: &Calibrations, qs: &[Vec<f32>]) {
    let committed = superfile_object();
    let uri: SuperfileUri = committed.uri;
    let q = &qs[0];

    for tier in [Tier::Cold] {
        let mut g = c.benchmark_group(tiers::search_group_name(
            "superfile_vec",
            tier,
            Some(committed.storage_label),
        ));
        g.sample_size(10);
        // Cold rebuilds a fresh cache + full S3 cold open per sample, so a
        // single sample can exceed the 5s default and criterion warns it
        // can't fit 10 samples. Give it room.
        if tier == Tier::Cold {
            g.measurement_time(Duration::from_secs(30));
        }

        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            let Some(c_inf) = cal.infino[i] else {
                continue;
            };
            let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
            let (p, r) = (c_inf.probe, c_inf.refine);
            let bench_id = format!("infino_{label}");

            match tier {
                Tier::Cold => {
                    let storage = Arc::clone(&committed.storage);
                    let query = q.clone();
                    let opts = search_opts(p, r);
                    g.bench_function(&bench_id, |b| {
                        b.iter_custom(|iters| {
                            let mut total = Duration::ZERO;
                            for _ in 0..iters {
                                let (cache_dir, cache) =
                                    tiers::fresh_superfile_cache(Arc::clone(&storage));
                                let t0 = Instant::now();
                                tiers::block_on(async {
                                    let reader = cache.reader(&uri).await.expect("cold reader");
                                    let _ = reader
                                        .vector_search(VEC_COLUMN, &query, TOP_K, opts)
                                        .await
                                        .expect("cold vector_search");
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

        let bench_id = "infino_default_options_top10";
        let default_opts = search_opts(DEFAULT_NPROBE, DEFAULT_RERANK_MULT);
        match tier {
            Tier::Cold => {
                let storage = Arc::clone(&committed.storage);
                let query = q.clone();
                g.bench_function(bench_id, |b| {
                    b.iter_custom(|iters| {
                        let mut total = Duration::ZERO;
                        for _ in 0..iters {
                            let (cache_dir, cache) =
                                tiers::fresh_superfile_cache(Arc::clone(&storage));
                            let t0 = Instant::now();
                            tiers::block_on(async {
                                let reader = cache.reader(&uri).await.expect("reader");
                                let _ = reader
                                    .vector_search(VEC_COLUMN, &query, TOP_K, default_opts)
                                    .await
                                    .expect("vector_search");
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

        g.finish();
    }
}

// ─── Markdown summary emitters ────────────────────────────────────────

mod group_name {
    pub const SUPERFILE_VEC_BUILD: &str = "superfile_vec_build";
    pub const SUPERFILE_VEC_SEARCH: &str = "superfile_vec_hot_search";
}

fn emit_ingest_markdown() {
    use markdown::{MarkdownSection, fmt_bandwidth, fmt_throughput, fmt_time, read_mean_ns};

    let group = group_name::SUPERFILE_VEC_BUILD;
    let one_thread_id = format!("infino_1thread_{N_DOCS}docs");
    let rayon_id = format!("infino_rayon_default_threads_{N_DOCS}docs");
    let one_thread = read_mean_ns(group, &one_thread_id);
    let rayon = read_mean_ns(group, &rayon_id);
    let one_thread_rss = rss::read_peak_rss_bytes(group, &one_thread_id);
    let rayon_rss = rss::read_peak_rss_bytes(group, &rayon_id);

    // Logical input payload: the raw f32 vectors (`n_docs × dim × 4`),
    // identical across both variants.
    let input_bytes = (N_DOCS * DIM * std::mem::size_of::<f32>()) as f64;

    let mut body = String::new();
    body.push_str(&format!(
        "### Superfile vector — ingest ({N_DOCS} docs × dim={DIM}, Gaussian planted clusters, cosine)\n\n"
    ));
    body.push_str(
        "Build path: `SuperfileBuilder` → unified `.parquet` (same as production supertable commit). \
         `infino_1thread` builds one superfile (the vector builder still uses intra-segment rayon \
         for rotation/encode); `infino_rayon_default_threads` shards the corpus across the rayon \
         pool into one superfile per shard (the multi-segment shape supertable commit produces). \
         Bandwidth (MB/s) is over the raw f32 vector payload.\n\n",
    );
    body.push_str(
        "| Engine                       | Time       | Throughput | Bandwidth  | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |\n",
    );
    body.push_str(
        "|------------------------------|------------|------------|------------|-----------|------------|-----------|------------|\n",
    );

    let row = |label: &str, bench_id: &str, ns: Option<f64>, peak: Option<u64>| -> String {
        let time = ns.map(fmt_time).unwrap_or_else(|| "—".into());
        let thrpt = ns
            .map(|n| fmt_throughput((N_DOCS as f64) / (n / 1e9)))
            .unwrap_or_else(|| "—".into());
        let bandwidth = ns
            .map(|n| fmt_bandwidth(input_bytes / (n / 1e9)))
            .unwrap_or_else(|| "—".into());
        let rss_cell = peak.map(rss::fmt_bytes).unwrap_or_else(|| "—".into());
        let median_rss = rss::fmt_median_rss(group, bench_id);
        let p90_rss = rss::fmt_p90_rss(group, bench_id);
        let rss_delta = rss::fmt_peak_rss_delta(group, bench_id);
        format!(
            "| {label:28} | {time:10} | {thrpt:10} | {bandwidth:10} | {rss_cell:9} | {median_rss:10} | {p90_rss:9} | {rss_delta:10} |\n"
        )
    };

    body.push_str(&row(
        "infino_1thread",
        &one_thread_id,
        one_thread,
        one_thread_rss,
    ));
    body.push_str(&row(
        "infino_rayon_default_threads",
        &rayon_id,
        rayon,
        rayon_rss,
    ));

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/vector/superfile/ingest".into(),
        body,
    });
}

fn emit_search_markdown() {
    use markdown::{MarkdownSection, fmt_time, read_mean_ns};

    let group = group_name::SUPERFILE_VEC_SEARCH;
    let cal = calibrations();

    let mut body = String::new();
    body.push_str(&format!(
        "### Superfile vector — search ({N_DOCS} docs × dim={DIM}, calibrated at recall targets)\n\n"
    ));
    body.push_str(
        "Hot = `SuperfileReader::open` in memory; cold = same `.parquet` on object storage via \
         `DiskCacheStore::reader` → `vector_search` (production cold path).\n\n",
    );
    body.push_str(
        "| Recall target | (p, r)     | hot        | cold       | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |\n",
    );
    body.push_str(
        "|---------------|------------|------------|------------|----------|------------|---------|------------|\n",
    );

    for (i, &target) in RECALL_TARGETS.iter().enumerate() {
        let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
        let row_target = format!("{target:.2}");
        if let Some(c_inf) = cal.infino[i] {
            let id = format!("infino_{label}");
            let hot = read_mean_ns(group, &id);
            let cold = markdown::read_tier_mean_ns("superfile_vec", "cold", &id);
            let peak = rss::read_peak_rss_bytes(group, &id);
            let rss_cell = peak.map(rss::fmt_bytes).unwrap_or_else(|| "—".into());
            let median_rss = rss::fmt_median_rss(group, &id);
            let p90_rss = rss::fmt_p90_rss(group, &id);
            let rss_delta = rss::fmt_peak_rss_delta(group, &id);
            body.push_str(&format!(
                "| {row_target:13} | (p={}, r={}) | {} | {} | {rss_cell} | {median_rss} | {p90_rss} | {rss_delta} |\n",
                c_inf.probe,
                c_inf.refine,
                hot.map(fmt_time).unwrap_or_else(|| "—".into()),
                cold.map(fmt_time).unwrap_or_else(|| "—".into()),
            ));
        } else {
            body.push_str(&format!(
                "| {row_target:13} | — | — | — | — | — | — | — |\n"
            ));
        }
    }

    body.push('\n');
    body.push_str(
    "**infino default options** (`nprobe=8, rerank_mult=20` — user-facing latency baseline):\n\n",
  );
    body.push_str("| Metric | Value |\n");
    body.push_str("|--------|-------|\n");
    let def = read_mean_ns(group, "infino_default_options_top10");
    let def_cold =
        markdown::read_tier_mean_ns("superfile_vec", "cold", "infino_default_options_top10");
    let def_s = def.map(fmt_time).unwrap_or_else(|| "—".into());
    let def_rss = rss::read_peak_rss_bytes(group, "infino_default_options_top10")
        .map(rss::fmt_bytes)
        .unwrap_or_else(|| "—".into());
    let def_median = rss::fmt_median_rss(group, "infino_default_options_top10");
    let def_p90 = rss::fmt_p90_rss(group, "infino_default_options_top10");
    body.push_str(&format!(
        "| infino_default_options_top10 (hot) | {def_s} |\n"
    ));
    body.push_str(&format!(
        "| infino_default_options_top10 (cold) | {} |\n",
        def_cold.map(fmt_time).unwrap_or_else(|| "—".into())
    ));
    body.push_str(&format!(
        "| infino_default_options_top10_peak_rss | {def_rss} |\n"
    ));
    body.push_str(&format!(
        "| infino_default_options_top10_median_rss | {def_median} |\n"
    ));
    body.push_str(&format!(
        "| infino_default_options_top10_p90_rss | {def_p90} |\n"
    ));

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/vector/superfile/search".into(),
        body,
    });
}

// ─── Artifact size + first-query report ──────────────────────────────

fn artifact_report(n: usize, n_cent: usize, vectors: &[f32]) {
    // Build once and time the cold open + first query so the
    // user-visible "first-query latency" number isn't hidden inside
    // criterion's warm-up loop.
    use std::time::Instant;

    let t0 = Instant::now();
    let blob = build_superfile_bytes(vectors);
    let build_elapsed = t0.elapsed();

    let size_mib = blob.len() as f64 / (1024.0 * 1024.0);

    let t0 = Instant::now();
    let reader = SuperfileReader::open(Bytes::from(blob)).expect("open superfile");
    let open_elapsed = t0.elapsed();

    let q = &queries_calibration()[0];
    let opts = search_opts(DEFAULT_NPROBE, DEFAULT_RERANK_MULT);
    let t0 = Instant::now();
    let _ =
        corpus::block_on_inmem(async { reader.vector_search(VEC_COLUMN, q, TOP_K, opts).await })
            .expect("vector_search");
    let first_q_elapsed = t0.elapsed();

    eprintln!(
        "\n--- artifact-size + cold-load report ({n} docs, {n_cent} clusters, dim={DIM}) ---"
    );
    eprintln!(
        "infino:  build {:>7.2}s  size {size_mib:>6.2} MiB  open {:>6.2} ms  first-query {:>5.2} ms",
        build_elapsed.as_secs_f64(),
        open_elapsed.as_secs_f64() * 1e3,
        first_q_elapsed.as_secs_f64() * 1e3,
    );
}

criterion_group!(benches, bench);
