// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Standalone recall/latency microbench for the per-superfile vector path.
//!
//! Builds one in-memory superfile, computes brute-force ground truth once,
//! then sweeps `(nprobe, rerank_mult)` — printing each row as it completes
//! (no full-bench calibration grid, no cold tier).
//!
//! Run:
//!   cargo run --release -p infino-bench-utils --bin profile_superfile
//!   cargo run --release -p infino-bench-utils --bin profile_superfile -- 1000000 1024 sweep-down
//!
//! `sweep-down` walks **down** from the calibrated 0.99 point `(p=5, r=256)`:
//! halve `r` at fixed `p=5`, then halve `p` at the last `r`.

use std::sync::Arc;
use std::time::Instant;

use futures::executor::block_on;
use infino::roaring::RoaringBitmap;
use infino::superfile::reader::VectorSearchOptions;
use infino_bench_utils::corpus::{self, DIM};

const SEED: u64 = 1;
const QUERY_SEED: u64 = 17;
const N_QUERIES: usize = 20;
const TOP_K: usize = 10;
const SIGMA: f32 = 0.05;

/// Default superfile doc count when no CLI arg.
const DEFAULT_N_DOCS: usize = 1_000_000;
/// Default IVF centroid count when no CLI arg (matches `corpus::n_cent(1M)`).
const DEFAULT_N_CENT: usize = 1024;
/// Warm-up nprobe used to touch pages before timing.
const WARMUP_NPROBE: usize = 16;
/// Warm-up rerank multiplier.
const WARMUP_RERANK_MULT: usize = 64;
/// Seconds-to-milliseconds factor for latency output.
const MS_PER_SEC: f64 = 1e3;
/// Recall floor used by the main vector bench default-config gate.
const RECALL_FLOOR: f32 = 0.80;
/// Filtered-search allow-set stride (~10% selectivity), matching the bench.
const FILTER_KEEP_EVERY: usize = 10;
/// Starting point for `sweep-down` mode (lowest-p50 0.99 calibrated point).
const SWEEP_DOWN_PROBE: usize = 5;
const SWEEP_DOWN_RERANK: usize = 256;

fn opts(nprobe: usize, rerank_mult: usize) -> VectorSearchOptions {
    VectorSearchOptions::new()
        .with_nprobe(nprobe)
        .with_rerank_mult(rerank_mult)
}

/// `(nprobe, rerank_mult)` points for the legacy differential grid.
fn legacy_grid() -> Vec<(usize, usize)> {
    vec![
        (1, 4),
        (4, 4),
        (16, 4),
        (64, 4),
        (128, 4),
        (16, 16),
        (16, 64),
        (16, 256),
        (16, 1024),
        (64, 256),
        (128, 256),
        (128, 1024),
    ]
}

/// Downward walk from `(SWEEP_DOWN_PROBE, SWEEP_DOWN_RERANK)`: halve `r`, then halve `p`.
fn sweep_down_grid() -> Vec<(usize, usize)> {
    let mut grid = Vec::new();
    let mut r = SWEEP_DOWN_RERANK;
    while r >= 1 {
        grid.push((SWEEP_DOWN_PROBE, r));
        if r == 1 {
            break;
        }
        r /= 2;
    }
    let last_r = grid.last().expect("non-empty sweep-down grid").1;
    let mut p = SWEEP_DOWN_PROBE;
    while p >= 1 {
        if p != SWEEP_DOWN_PROBE {
            grid.push((p, last_r));
        }
        if p == 1 {
            break;
        }
        p /= 2;
    }
    grid
}

fn filtered_allow(n_docs: usize) -> Arc<RoaringBitmap> {
    let mut allow = RoaringBitmap::new();
    for i in (0..n_docs as u32).step_by(FILTER_KEEP_EVERY) {
        allow.insert(i);
    }
    Arc::new(allow)
}

fn filtered_ground_truth(
    vectors: &[f32],
    queries: &[Vec<f32>],
    allow: &RoaringBitmap,
) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(f32, u32)> = allow
                .iter()
                .map(|id| {
                    let row = &vectors[id as usize * DIM..(id as usize + 1) * DIM];
                    let dot: f32 = row.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
                    (-dot, id)
                })
                .collect();
            dists.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            dists.truncate(TOP_K);
            dists.into_iter().map(|(_, id)| id).collect()
        })
        .collect()
}

fn mean_recall_unfiltered(
    reader: &infino::superfile::SuperfileReader,
    queries: &[Vec<f32>],
    gt: &[Vec<u32>],
    nprobe: usize,
    rerank_mult: usize,
) -> f32 {
    let mut sum = 0f32;
    for (q, truth) in queries.iter().zip(gt) {
        let hits = block_on(reader.vector_hits_async("emb", q, TOP_K, opts(nprobe, rerank_mult)))
            .expect("vector_hits");
        sum += corpus::recall_at_k(&hits, truth);
    }
    sum / queries.len() as f32
}

fn mean_recall_filtered(
    reader: &infino::superfile::SuperfileReader,
    queries: &[Vec<f32>],
    gt: &[Vec<u32>],
    allow: &Arc<RoaringBitmap>,
    nprobe: usize,
    rerank_mult: usize,
) -> f32 {
    let mut sum = 0f32;
    for (q, truth) in queries.iter().zip(gt) {
        let hits = block_on(reader.vector_hits_filtered_async(
            "emb",
            q,
            TOP_K,
            opts(nprobe, rerank_mult),
            Some(Arc::clone(allow)),
        ))
        .expect("vector_hits_filtered");
        sum += corpus::recall_at_k(&hits, truth);
    }
    sum / queries.len() as f32
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n_docs: usize = args
        .next()
        .and_then(|a| a.parse().ok())
        .unwrap_or(DEFAULT_N_DOCS);
    let n_cent: usize = args
        .next()
        .and_then(|a| a.parse().ok())
        .unwrap_or(DEFAULT_N_CENT);
    let mode = args.next().unwrap_or_else(|| "sweep-down".into());

    eprintln!(
        "[profile] building 1 superfile: {n_docs} docs, n_cent={n_cent}, dim={DIM}, mode={mode}"
    );
    let t = Instant::now();
    let vectors = corpus::generate_vector_corpus(n_docs, n_cent, SEED, true);
    let docs = corpus::generate_text_corpus(n_docs, SEED);
    let blob = corpus::build_superfile(&docs, &vectors, n_cent);
    let reader = corpus::open_superfile(blob);
    eprintln!(
        "[profile] build+open took {:.1}s  (docs/cluster ≈ {})",
        t.elapsed().as_secs_f64(),
        n_docs / n_cent.max(1)
    );

    let queries =
        corpus::generate_realistic_queries(&vectors, n_docs, N_QUERIES, QUERY_SEED, true, SIGMA);

    eprintln!("[profile] computing unfiltered ground truth ({N_QUERIES} queries)...");
    let gt: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| corpus::brute_force_topk_cosine(&vectors, n_docs, q, TOP_K))
        .collect();

    let allow = filtered_allow(n_docs);
    eprintln!("[profile] computing filtered ground truth (~10% allow-set)...");
    let filtered_gt = filtered_ground_truth(&vectors, &queries, &allow);

    for q in &queries {
        let _ = block_on(reader.vector_hits_async(
            "emb",
            q,
            TOP_K,
            opts(WARMUP_NPROBE, WARMUP_RERANK_MULT),
        ));
    }

    let grid: Vec<(usize, usize)> = if mode == "sweep-down" {
        sweep_down_grid()
    } else {
        legacy_grid()
    };

    println!("\n nprobe  rerank_mult   p50_ms   recall@10  filt@10  floor");
    println!("----------------------------------------------------------------");
    for &(nprobe, rerank_mult) in &grid {
        eprintln!("[profile] measuring p={nprobe}, r={rerank_mult}...");
        let mut lats = Vec::with_capacity(N_QUERIES);
        for q in &queries {
            let t = Instant::now();
            let _ = block_on(reader.vector_hits_async("emb", q, TOP_K, opts(nprobe, rerank_mult)))
                .expect("search");
            lats.push(t.elapsed().as_secs_f64() * MS_PER_SEC);
        }
        lats.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p50 = lats[lats.len() / 2];

        let recall = mean_recall_unfiltered(&reader, &queries, &gt, nprobe, rerank_mult);
        let filt =
            mean_recall_filtered(&reader, &queries, &filtered_gt, &allow, nprobe, rerank_mult);
        let pass = if recall >= RECALL_FLOOR && filt >= RECALL_FLOOR {
            "ok"
        } else {
            "FAIL"
        };
        println!(
            " {nprobe:>6}  {rerank_mult:>11}   {p50:>6.2}   {recall:>8.3}   {filt:>7.3}   {pass}"
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
}
