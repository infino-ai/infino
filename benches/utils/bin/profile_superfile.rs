// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Standalone profiling harness for the per-superfile vector query path.
//!
//! Builds ONE supertable-superfile-sized superfile (default 2.5M docs,
//! `n_cent = 1024`, dim 384, Sq8 / Cosine — i.e. one of the four
//! 10M-supertable superfiles) and times the warm in-memory
//! `SuperfileReader::search` across an `(nprobe, rerank_mult)` grid,
//! reporting per-config latency and recall@10.
//!
//! It deliberately skips the 10M build, the full calibration
//! sweep, and the cold/warm tier machinery — none of which a
//! query-path profile needs — so it runs in minutes instead of hours.
//!
//! The grid isolates the two suspected costs by differential timing:
//!   * sweep `nprobe` at small `rerank_mult` -> coarse 1-bit scan cost
//!   * sweep `rerank_mult` at fixed `nprobe` -> Sq8 rerank cost
//!
//! and the recall column shows whether per-superfile recall keeps
//! climbing past the old 16-probe cap (the ceiling question).
//!
//! To A/B the within-superfile rayon parallelism, flip
//! `PARALLEL_SCAN_MIN` in `src/superfile/vector/reader.rs`
//! (`usize::MAX` forces serial, `0` forces parallel) and rerun.
//!
//! Run:
//!   cargo run --release -p infino-bench-utils --bin profile_superfile
//!   cargo run --release -p infino-bench-utils --bin profile_superfile -- 625000 256

use std::collections::HashSet;
use std::time::Instant;

use futures::executor::block_on;
use infino::superfile::reader::VectorSearchOptions;
use infino_bench_utils::corpus::{self, DIM};

const SEED: u64 = 42;
const N_QUERIES: usize = 30;
const TOP_K: usize = 10;
const SIGMA: f32 = 0.1;

/// Default superfile doc count (one 10M-shard slice) when no CLI arg.
const DEFAULT_N_DOCS: usize = 2_500_000;
/// Default IVF centroid count when no CLI arg (matches a supertable superfile).
const DEFAULT_N_CENT: usize = 1024;
/// XOR mask decorrelating the query seed from the corpus seed.
const QUERY_SEED_XOR: u64 = 0x9e37;
/// Warm-up nprobe used to touch pages before timing.
const WARMUP_NPROBE: usize = 16;
/// Warm-up rerank_mult (unused by the current opts closure, but kept
/// for parity with the timed grid).
const WARMUP_RERANK_MULT: usize = 64;
/// Seconds-to-milliseconds factor for latency output.
const MS_PER_SEC: f64 = 1e3;

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

    eprintln!(
        "[profile] building 1 superfile: {n_docs} docs, n_cent={n_cent}, dim={DIM}, Sq8/Cosine"
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

    let queries = corpus::generate_realistic_queries(
        &vectors,
        n_docs,
        N_QUERIES,
        SEED ^ QUERY_SEED_XOR,
        true,
        SIGMA,
    );

    eprintln!("[profile] computing brute-force ground truth ({N_QUERIES} queries)...");
    let gt: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| corpus::brute_force_topk_cosine(&vectors, n_docs, q, TOP_K))
        .collect();

    let opts = |nprobe: usize, _rerank_mult: usize| VectorSearchOptions::new().with_nprobe(nprobe);

    // Warm the reader (touch pages, settle the allocator) before timing.
    for q in &queries {
        let _ = block_on(reader.vector_hits_async(
            "emb",
            q,
            TOP_K,
            opts(WARMUP_NPROBE, WARMUP_RERANK_MULT),
        ));
    }

    // First block sweeps nprobe at small rerank (coarse-scan cost),
    // second sweeps rerank at fixed nprobe (rerank cost), the last
    // block is the wide-probe high-recall region the supertable grid
    // now reaches.
    let grid: &[(usize, usize)] = &[
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
    ];

    println!("\n nprobe  rerank_mult   p50_ms   mean_ms   recall@10");
    println!("-------------------------------------------------------");
    for &(nprobe, rerank_mult) in grid {
        let mut lats = Vec::with_capacity(N_QUERIES);
        let mut recall_sum = 0.0f64;
        for (qi, q) in queries.iter().enumerate() {
            let t = Instant::now();
            let hits =
                block_on(reader.vector_hits_async("emb", q, TOP_K, opts(nprobe, rerank_mult)))
                    .expect("search");
            lats.push(t.elapsed().as_secs_f64() * MS_PER_SEC);
            let got: HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
            let hit = gt[qi].iter().filter(|d| got.contains(d)).count();
            recall_sum += hit as f64 / TOP_K as f64;
        }
        lats.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p50 = lats[lats.len() / 2];
        let mean = lats.iter().sum::<f64>() / lats.len() as f64;
        let recall = recall_sum / N_QUERIES as f64;
        println!(" {nprobe:>6}  {rerank_mult:>11}   {p50:>6.2}   {mean:>7.2}   {recall:>8.3}");
    }
}
