// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Filtered vector search: recall@10 and cost at several selectivities.
//!
//! Builds a drained table, then runs filtered kNN with an allow-set that
//! keeps 1%, 10% and 50% of the rows, and reports per query:
//!
//! - recall@10 against brute-force exact search over the allowed rows,
//! - warm p50 latency,
//! - plan counters (cells scanned, 1-bit rows scanned, ranges, rows
//!   reranked),
//! - cold GETs and bytes (a fresh handle with an empty cache per query).
//!
//! Three data shapes:
//!
//! - **planted, spread**: the bench's synthetic corpus, with allowed rows
//!   picked by a hash of the row number, so they are spread evenly over
//!   the vectors.
//! - **planted, every Nth**: the same corpus, allowed rows `0, N, 2N, …`.
//!   The corpus puts row `i` in cluster `i % n_cent`, so this filter keeps
//!   only some clusters: a filter correlated with the vectors.
//! - **layered, spread**: groups of nearby sub-clusters with uneven sizes
//!   and per-cluster spread, rows assigned to clusters at random, and
//!   queries that are new points from the same model (never inserted).
//!
//! The allow-set is prepared once per filter, so latency covers the
//! vector search only, not evaluating the filter.
//!
//! Invoked as `cargo bench -- filtered-vector`. Table size:
//! `INFINO_BENCH_FILTERED_VECTOR_DOCS` (default 1M).

use std::{
    collections::{HashMap, HashSet},
    env,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    Metric,
    runtime_metrics::op_stats::with_op_stats,
    storage::{LocalFsStorageProvider, StorageProvider, io_counters},
    superfile::{VectorSearchOptions, builder::VectorConfig},
    supertable::{Supertable, SupertableOptions},
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use tempfile::TempDir;

use crate::{corpus, tiers};

/// Default table size.
const DEFAULT_DOCS: usize = 1_000_000;
/// Env override for the table size.
const DOCS_ENV: &str = "INFINO_BENCH_FILTERED_VECTOR_DOCS";
/// Rows per commit while building.
const DOCS_PER_COMMIT: usize = 100_000;
/// Queries per filter for recall and warm latency.
const N_QUERIES: usize = 100;
/// Queries per filter for the cold measurement (fresh handle each).
const N_COLD_QUERIES: usize = 20;
/// Requested top-k.
const K: usize = 10;
/// Allowed rows: one in this many (1%, 10%, 50%).
const KEEP_ONE_IN: &[usize] = &[100, 10, 2];
const VEC_COLUMN: &str = "emb";
const ROT_SEED: u64 = 7;
/// Planted corpus and query seeds (the supertable bench's values).
const PLANTED_CORPUS_SEED: u64 = 1;
const PLANTED_QUERY_SEED: u64 = 17;
/// Query noise around a corpus row (the supertable bench's value).
const PLANTED_QUERY_SIGMA: f32 = 0.05;

/// Dimensions of the layered corpus (a common text-embedding width).
const LAYERED_DIM: usize = 768;
/// Top-level groups; each holds several nearby sub-clusters.
const LAYERED_GROUPS: usize = 64;
/// Sub-clusters in total.
const LAYERED_CLUSTERS: usize = 1024;
/// How far a sub-cluster centre sits from its group centre.
const LAYERED_CLUSTER_OFFSET: f32 = 0.6;
/// Range of per-cluster row spread around the sub-cluster centre.
const LAYERED_SPREAD_MIN: f32 = 0.3;
const LAYERED_SPREAD_MAX: f32 = 0.9;
/// Power-law exponent of cluster sizes (larger = more uneven).
const LAYERED_SIZE_EXPONENT: f64 = 0.8;
/// Rows generated per parallel chunk.
const LAYERED_CHUNK_ROWS: usize = 16_384;
/// Odd multiplier that gives each chunk its own seed.
const LAYERED_CHUNK_SEED_MIX: u64 = 0x9E37_79B9;
/// Seeds of the layered model, its rows, and its queries.
const LAYERED_MODEL_SEED: u64 = 7_768;
const LAYERED_ROW_SEED: u64 = 11;
const LAYERED_QUERY_SEED: u64 = 23;

/// SplitMix64 constants: the row-number hash that picks spread rows.
const SPLITMIX_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_MUL_1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_MUL_2: u64 = 0x94D0_49BB_1331_11EB;

const MIB: f64 = 1024.0 * 1024.0;
const MS_PER_SEC: f64 = 1e3;
const PERCENT: f64 = 100.0;

/// How the allowed rows are picked.
#[derive(Clone, Copy)]
enum Pick {
    /// By a hash of the row number: spread evenly over the vectors.
    Spread,
    /// Rows `0, N, 2N, …`.
    EveryNth,
}

/// Unit-length Gaussian vector.
fn unit(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| StandardNormal.sample(rng)).collect();
    normalize(&mut v);
    v
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
}

/// The layered model: sub-cluster centres near their group centre, a
/// spread per cluster, and cumulative size weights.
struct Layered {
    centres: Vec<Vec<f32>>,
    spread: Vec<f32>,
    cumulative: Vec<f64>,
}

impl Layered {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let groups: Vec<Vec<f32>> = (0..LAYERED_GROUPS)
            .map(|_| unit(&mut rng, LAYERED_DIM))
            .collect();
        let centres = (0..LAYERED_CLUSTERS)
            .map(|_| {
                let group = &groups[rng.random_range(0..LAYERED_GROUPS)];
                let offset = unit(&mut rng, LAYERED_DIM);
                let mut c: Vec<f32> = group
                    .iter()
                    .zip(&offset)
                    .map(|(g, o)| g + LAYERED_CLUSTER_OFFSET * o)
                    .collect();
                normalize(&mut c);
                c
            })
            .collect();
        let spread = (0..LAYERED_CLUSTERS)
            .map(|_| rng.random_range(LAYERED_SPREAD_MIN..LAYERED_SPREAD_MAX))
            .collect();
        let mut weights: Vec<f64> = (0..LAYERED_CLUSTERS)
            .map(|rank| 1.0 / ((rank + 1) as f64).powf(LAYERED_SIZE_EXPONENT))
            .collect();
        // Shuffle so a cluster's id does not tell its size.
        for i in (1..weights.len()).rev() {
            weights.swap(i, rng.random_range(0..=i));
        }
        let total: f64 = weights.iter().sum();
        let mut acc = 0.0;
        let cumulative = weights
            .iter()
            .map(|w| {
                acc += w / total;
                acc
            })
            .collect();
        Self {
            centres,
            spread,
            cumulative,
        }
    }

    /// `n` rows: each picks a cluster by size, then its centre plus noise
    /// at the cluster's spread.
    fn rows(&self, n: usize, seed: u64) -> Vec<f32> {
        let mut out = vec![0f32; n * LAYERED_DIM];
        out.par_chunks_mut(LAYERED_CHUNK_ROWS * LAYERED_DIM)
            .enumerate()
            .for_each(|(chunk, rows)| {
                let mut rng = StdRng::seed_from_u64(
                    seed ^ (chunk as u64 + 1).wrapping_mul(LAYERED_CHUNK_SEED_MIX),
                );
                for row in rows.chunks_mut(LAYERED_DIM) {
                    let u: f64 = rng.random_range(0.0..1.0);
                    let c = self
                        .cumulative
                        .partition_point(|&x| x < u)
                        .min(LAYERED_CLUSTERS - 1);
                    let noise = unit(&mut rng, LAYERED_DIM);
                    for ((r, centre), e) in row.iter_mut().zip(&self.centres[c]).zip(&noise) {
                        *r = centre + self.spread[c] * e;
                    }
                    normalize(row);
                }
            });
        out
    }
}

/// SplitMix64 of the row number.
fn row_hash(row: usize) -> u64 {
    let mut x = (row as u64).wrapping_add(SPLITMIX_INCREMENT);
    x = (x ^ (x >> 30)).wrapping_mul(SPLITMIX_MUL_1);
    x = (x ^ (x >> 27)).wrapping_mul(SPLITMIX_MUL_2);
    x ^ (x >> 31)
}

/// Allowed row numbers: about one in `keep_one_in`.
fn allowed_rows(n_docs: usize, keep_one_in: usize, pick: Pick) -> Vec<usize> {
    match pick {
        Pick::Spread => (0..n_docs)
            .filter(|&row| row_hash(row).is_multiple_of(keep_one_in as u64))
            .collect(),
        Pick::EveryNth => (0..n_docs).step_by(keep_one_in).collect(),
    }
}

fn options(storage: Arc<dyn StorageProvider>, dim: usize) -> SupertableOptions {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let schema = Arc::new(Schema::new(vec![Field::new(
        VEC_COLUMN,
        DataType::FixedSizeList(item, dim as i32),
        false,
    )]));
    SupertableOptions::new(
        schema,
        vec![],
        vec![VectorConfig {
            provided_centroids: None,
            column: VEC_COLUMN.into(),
            dim,
            rot_seed: ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: corpus::bench_rerank_codec(Metric::Cosine),
        }],
    )
    .expect("options")
    .with_storage(storage)
}

/// Commit `vecs` in chunks, then drain into the hidden index.
fn build(storage: &Arc<dyn StorageProvider>, vecs: &[f32], dim: usize) -> Supertable {
    let n_docs = vecs.len() / dim;
    let started = Instant::now();
    let st = Supertable::create(options(Arc::clone(storage), dim)).expect("create");
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    for start in (0..n_docs).step_by(DOCS_PER_COMMIT) {
        let end = (start + DOCS_PER_COMMIT).min(n_docs);
        let fsl = FixedSizeListArray::try_new(
            Arc::clone(&item),
            dim as i32,
            Arc::new(Float32Array::from(vecs[start * dim..end * dim].to_vec())) as ArrayRef,
            None,
        )
        .expect("vectors");
        let batch =
            RecordBatch::try_new(st.options().schema.clone(), vec![Arc::new(fsl)]).expect("batch");
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    st.drain_vectors_to_cells_sync().expect("drain");
    eprintln!(
        "[filtered_vector] built + drained {n_docs} rows in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    st
}

/// Exact top-k row numbers among `rows` (cosine on unit vectors = dot).
fn exact_top_k(vecs: &[f32], dim: usize, rows: &[usize], query: &[f32]) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = rows
        .iter()
        .map(|&row| {
            let v = &vecs[row * dim..(row + 1) * dim];
            (v.iter().zip(query).map(|(a, b)| a * b).sum::<f32>(), row)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.truncate(K);
    scored.into_iter().map(|(_, row)| row).collect()
}

/// Per-query means for one filter.
struct Measured {
    recall: f64,
    warm_p50: Duration,
    cells: f64,
    rows_scanned: f64,
    ranges: f64,
    reranked: f64,
    cold_gets: f64,
    cold_mib: f64,
}

fn mean(total: u64, n: usize) -> f64 {
    total as f64 / n as f64
}

fn measure(
    st: &Supertable,
    storage: &Arc<dyn StorageProvider>,
    dim: usize,
    row_ids: &[i128],
    allowed: &[usize],
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
) -> Measured {
    let opts = VectorSearchOptions::new();
    let allow_ids: Arc<Vec<i128>> = Arc::new(allowed.iter().map(|&row| row_ids[row]).collect());
    let reader = st.reader().expect("reader");
    let prepared =
        tiers::block_on(reader.prepare_vector_stable_allow_async(Arc::clone(&allow_ids)))
            .expect("prepare allow-set");
    // Untimed pass so the measured window is steady-state warm.
    for q in queries {
        tiers::block_on(
            reader.vector_hits_prepared_global_allow_async(VEC_COLUMN, q, K, opts, &prepared),
        )
        .expect("prewarm");
    }
    let mut latencies = Vec::with_capacity(queries.len());
    let (mut recall, mut cells, mut rows_scanned, mut ranges, mut reranked) = (0.0, 0, 0, 0, 0);
    for (q, gt) in queries.iter().zip(truth) {
        // The reader is minted inside the scope so the scope sees its work.
        let ((hits, elapsed), stats) = with_op_stats(|| {
            let r = st.reader().expect("reader");
            let started = Instant::now();
            let hits = tiers::block_on(
                r.vector_hits_prepared_global_allow_async(VEC_COLUMN, q, K, opts, &prepared),
            )
            .expect("filtered query");
            (hits, started.elapsed())
        });
        latencies.push(elapsed);
        let got: HashSet<i128> = hits.iter().filter_map(|h| h.stable_id).collect();
        recall += gt
            .iter()
            .filter(|&&row| got.contains(&row_ids[row]))
            .count() as f64
            / K as f64;
        cells += stats.vector_cells_scanned;
        rows_scanned += stats.vector_candidates_scanned;
        ranges += stats.planned_read_ranges;
        reranked += stats.vector_rows_reranked;
    }
    latencies.sort_unstable();
    let (mut cold_gets, mut cold_bytes) = (0, 0);
    for q in queries.iter().take(N_COLD_QUERIES) {
        let fresh = Supertable::open(options(Arc::clone(storage), dim)).expect("open cold");
        let r = fresh.reader().expect("reader");
        let p = tiers::block_on(r.prepare_vector_stable_allow_async(Arc::clone(&allow_ids)))
            .expect("prepare cold allow-set");
        let _ = io_counters::take();
        tiers::block_on(r.vector_hits_prepared_global_allow_async(VEC_COLUMN, q, K, opts, &p))
            .expect("cold query");
        let (gets, bytes, _, _) = io_counters::take();
        cold_gets += gets;
        cold_bytes += bytes;
    }
    let n = queries.len();
    Measured {
        recall: recall / n as f64,
        warm_p50: latencies[latencies.len() / 2],
        cells: mean(cells, n),
        rows_scanned: mean(rows_scanned, n),
        ranges: mean(ranges, n),
        reranked: mean(reranked, n),
        cold_gets: mean(cold_gets, N_COLD_QUERIES),
        cold_mib: mean(cold_bytes, N_COLD_QUERIES) / MIB,
    }
}

/// Build one table from `vecs` and print a row per filter and pick.
fn run_shape(name: &str, vecs: &[f32], dim: usize, queries: &[Vec<f32>], picks: &[Pick]) {
    let n_docs = vecs.len() / dim;
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    // Query through a freshly opened handle: the build leaves the writing
    // handle's cache holding what the drain just wrote, which is not the
    // state a reader starts from.
    drop(build(&storage, vecs, dim));
    let st = Supertable::open(options(Arc::clone(&storage), dim)).expect("open");
    let id_to_row: HashMap<i128, u32> = corpus::engine_id_to_dense(&st, n_docs);
    let mut row_ids = vec![0i128; n_docs];
    for (&id, &row) in &id_to_row {
        row_ids[row as usize] = id;
    }
    println!();
    println!(
        "### {name} ({n_docs} rows × dim={dim}, k={K}, {} queries)",
        queries.len()
    );
    println!();
    println!(
        "| filter | kept | recall@10 | warm p50 ms | cells scanned | 1-bit rows scanned | ranges | rows reranked | cold GETs | cold MiB |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for &pick in picks {
        let label = match pick {
            Pick::Spread => "spread",
            Pick::EveryNth => "every Nth",
        };
        for &keep_one_in in KEEP_ONE_IN {
            let allowed = allowed_rows(n_docs, keep_one_in, pick);
            let truth: Vec<Vec<usize>> = queries
                .par_iter()
                .map(|q| exact_top_k(vecs, dim, &allowed, q))
                .collect();
            let m = measure(&st, &storage, dim, &row_ids, &allowed, queries, &truth);
            println!(
                "| {label} | {:.0}% | {:.3} | {:.2} | {:.1} | {:.0} | {:.1} | {:.0} | {:.1} | {:.1} |",
                PERCENT * allowed.len() as f64 / n_docs as f64,
                m.recall,
                m.warm_p50.as_secs_f64() * MS_PER_SEC,
                m.cells,
                m.rows_scanned,
                m.ranges,
                m.reranked,
                m.cold_gets,
                m.cold_mib,
            );
        }
    }
}

pub fn run() {
    let n_docs = env::var(DOCS_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_DOCS);

    let planted = corpus::MmapVectorCorpus::generate(
        n_docs,
        corpus::n_cent(n_docs),
        PLANTED_CORPUS_SEED,
        true,
    );
    let planted_vecs = planted.as_slice();
    let planted_queries = corpus::generate_realistic_queries(
        planted_vecs,
        n_docs,
        N_QUERIES,
        PLANTED_QUERY_SEED,
        true,
        PLANTED_QUERY_SIGMA,
    );
    run_shape(
        "planted clusters",
        planted_vecs,
        corpus::dim(),
        &planted_queries,
        &[Pick::Spread, Pick::EveryNth],
    );
    drop(planted);

    let model = Layered::new(LAYERED_MODEL_SEED);
    let layered_vecs = model.rows(n_docs, LAYERED_ROW_SEED);
    let layered_queries: Vec<Vec<f32>> = model
        .rows(N_QUERIES, LAYERED_QUERY_SEED)
        .chunks(LAYERED_DIM)
        .map(<[f32]>::to_vec)
        .collect();
    run_shape(
        "layered clusters",
        &layered_vecs,
        LAYERED_DIM,
        &layered_queries,
        &[Pick::Spread],
    );
}
