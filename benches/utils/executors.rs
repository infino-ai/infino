// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared bench executors.
//!
//! One implementation of each benchmark's query battery, query
//! execution, warm/cold measurement, and report rendering. Both the
//! superfile (single-superfile, in-memory) and supertable (multi-superfile,
//! object-store) runners call these functions; the only thing each tier
//! supplies is a *reader* (and, for cold, a way to open a fresh one).
//! The reader type is an implementation detail hidden behind the
//! per-modality trait here, so the measured + reported surface can never
//! drift between the two tiers again.

use std::time::{Duration, Instant};

/// p50 of a sample set (lower-median; matches the historical bench
/// definition shared by every runner).
pub fn p50(samples: &mut [Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort_unstable();
    samples[(samples.len() - 1) / 2]
}

/// Cold timings for one query, split at the open/search boundary:
/// `open` is the fresh-consumer open (consumer + manifest + every
/// superfile reader), `search` is the first query over the opened but
/// data-cold table. Timed separately so cold search latency never
/// bills the one-time open bookkeeping — the same cold-open vs
/// cold-first-search split the quick-iter object-store harness uses.
#[derive(Clone, Copy)]
pub struct ColdTiming {
    pub open: Duration,
    pub search: Duration,
}

/// Force-open every superfile reader on the consumer's pinned snapshot —
/// the "cold open" phase of a cold iteration. Runs the same per-superfile
/// open the query fan-out would lazily trigger (in-memory tier → disk
/// cache admit → lazy range-GET fallback), concurrently like the query
/// path, so the subsequent timed search pays only the search work.
pub fn open_all_superfiles(consumer: &infino::supertable::Supertable) {
    let reader = consumer.reader();
    let manifest = reader.manifest();
    let store = &manifest.options.store;
    let disk_cache = manifest.options.disk_cache.as_ref();
    let storage = manifest.options.storage.as_ref();
    crate::tiers::block_on(async {
        futures::future::try_join_all(manifest.superfiles.iter().map(|e| {
            infino::supertable::query::superfile_reader::superfile_reader(
                store,
                disk_cache,
                storage,
                &e.uri,
                e.subsection_offsets.as_ref(),
            )
        }))
        .await
        .expect("cold open: open superfile readers");
    });
}

pub mod fts {
    use super::*;
    use std::collections::HashMap;

    use infino::superfile::SuperfileReader;
    use infino::superfile::fts::reader::BoolMode as InfinoBoolMode;
    use infino::supertable::SupertableReader;

    use crate::harness::{BoolMode, FtsQuery};
    use crate::markdown::fmt_time;
    use crate::report::{Better, Block, Cell, Report, Section, metric, text};
    use crate::rss::{self, PeakSampler, RssStats};

    /// Nanoseconds per second, for time-cell formatting.
    const NS_PER_SEC: f64 = 1e9;

    /// The full FTS query battery — single source of truth for both
    /// tiers' warm + cold search and the cross-engine recall grading.
    pub const FTS_BATTERY: &[FtsQuery] = &[
        FtsQuery {
            name: "single_rare",
            terms: &["term09999"],
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "single_df1",
            terms: &["doc0500000"],
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "single_common",
            terms: &["term00001"],
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "two_term_or",
            terms: &["term00001", "term00050"],
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "three_wide_or",
            terms: &["term00001", "term00050", "term00100"],
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "three_similar_or",
            terms: &["term00050", "term00051", "term00052"],
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "five_term_or",
            terms: &[
                "term00050",
                "term00051",
                "term00052",
                "term00053",
                "term00054",
            ],
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "ten_term_or",
            terms: &[
                "term00050",
                "term00051",
                "term00052",
                "term00053",
                "term00054",
                "term00055",
                "term00056",
                "term00057",
                "term00058",
                "term00059",
            ],
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "two_term_and",
            terms: &["term00001", "term00050"],
            mode: BoolMode::And,
        },
        FtsQuery {
            name: "three_wide_and",
            terms: &["term00001", "term00050", "term00100"],
            mode: BoolMode::And,
        },
        FtsQuery {
            name: "three_similar_and",
            terms: &["term00050", "term00051", "term00052"],
            mode: BoolMode::And,
        },
        FtsQuery {
            name: "five_term_and",
            terms: &[
                "term00050",
                "term00051",
                "term00052",
                "term00053",
                "term00054",
            ],
            mode: BoolMode::And,
        },
        FtsQuery {
            name: "ten_term_and",
            terms: &[
                "term00050",
                "term00051",
                "term00052",
                "term00053",
                "term00054",
                "term00055",
                "term00056",
                "term00057",
                "term00058",
                "term00059",
            ],
            mode: BoolMode::And,
        },
    ];

    /// OR query names, in table order.
    pub const OR_QUERIES: &[&str] = &[
        "single_rare",
        "single_df1",
        "single_common",
        "two_term_or",
        "three_wide_or",
        "three_similar_or",
        "five_term_or",
        "ten_term_or",
    ];

    /// AND query names, in table order.
    pub const AND_QUERIES: &[&str] = &[
        "two_term_and",
        "three_wide_and",
        "three_similar_and",
        "five_term_and",
        "ten_term_and",
    ];

    pub fn to_infino_mode(mode: BoolMode) -> InfinoBoolMode {
        match mode {
            BoolMode::Or => InfinoBoolMode::Or,
            BoolMode::And => InfinoBoolMode::And,
        }
    }

    /// Correctness gate run on **both tiers** after the artifact is built.
    /// The corpus plants a per-doc-unique `doc{id:07}` token, so a df=1
    /// lookup must return exactly one hit, and a common term must return
    /// at least one — i.e. the FTS index is present and resolving.
    pub fn assert_correct<R: FtsRead>(reader: &R, column: &str, n_docs: usize, log_prefix: &str) {
        let mid = n_docs / 2;
        let df1 = format!("doc{mid:07}");
        let got = reader.bm25_rows(column, &df1, 10, InfinoBoolMode::Or);
        assert_eq!(
            got, 1,
            "[{log_prefix}] correctness: df=1 token {df1:?} returned {got} hits, expected 1"
        );
        let common = reader.bm25_rows(column, "term00001", 10, InfinoBoolMode::Or);
        assert!(
            common >= 1,
            "[{log_prefix}] correctness: common term returned 0 hits (empty index?)"
        );
        eprintln!("[{log_prefix}] correctness OK: df=1 -> 1 hit, common -> {common} hits");
    }

    /// A reader the FTS executor can run a BM25 query against. Hides
    /// whether the bytes are an in-memory superfile or an object-store
    /// supertable consumer.
    pub trait FtsRead {
        /// Run one BM25 search; return the materialized hit/row count
        /// (used as a black-box sink so the search is not optimized out).
        fn bm25_rows(&self, column: &str, query: &str, k: usize, mode: InfinoBoolMode) -> usize;
    }

    impl FtsRead for SuperfileReader {
        fn bm25_rows(&self, column: &str, query: &str, k: usize, mode: InfinoBoolMode) -> usize {
            crate::tiers::block_on(self.bm25_hits_async(column, query, k, mode))
                .expect("superfile bm25_search")
                .len()
        }
    }

    impl FtsRead for SupertableReader {
        fn bm25_rows(&self, column: &str, query: &str, k: usize, mode: InfinoBoolMode) -> usize {
            self.bm25_search(column, query, k, mode, None)
                .expect("supertable bm25_search")
                .iter()
                .map(|b| b.num_rows())
                .sum()
        }
    }

    /// Warm p50 (+ per-query RSS) for one query.
    #[derive(Clone, Debug)]
    pub struct FtsQueryStat {
        pub name: &'static str,
        pub p50: Duration,
        pub rss: RssStats,
    }

    /// Measure the warm battery against an already-warm reader: one
    /// untimed prewarm per query, then `iters` timed iterations.
    pub fn measure_warm<R: FtsRead>(
        reader: &R,
        battery: &[FtsQuery],
        column: &str,
        k: usize,
        iters: usize,
        log_prefix: &str,
    ) -> Vec<FtsQueryStat> {
        battery
            .iter()
            .map(|q| {
                eprintln!("[{log_prefix}] warm: query {}...", q.name);
                let query = q.terms.join(" ");
                let mode = to_infino_mode(q.mode);
                let _ = reader.bm25_rows(column, &query, k, mode);
                let sampler = PeakSampler::start_default();
                let mut samples = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let t = Instant::now();
                    let rows = reader.bm25_rows(column, &query, k, mode);
                    samples.push(t.elapsed());
                    std::hint::black_box(rows);
                }
                let rss = sampler.stop_stats();
                FtsQueryStat {
                    name: q.name,
                    p50: p50(&mut samples),
                    rss,
                }
            })
            .collect()
    }

    /// Measure the cold battery: for each query, `iters` fresh-reader
    /// opens, timing the open and one search **separately** (see
    /// [`ColdTiming`]). `open_fresh` returns a guard that both
    /// implements [`FtsRead`] and owns the cache/consumer resources it
    /// must drop after the timed read; the guard's constructor performs
    /// the full open (consumer + superfile readers).
    pub fn measure_cold<G: FtsRead>(
        open_fresh: impl Fn() -> G,
        battery: &[FtsQuery],
        column: &str,
        k: usize,
        iters: usize,
        log_prefix: &str,
    ) -> HashMap<&'static str, ColdTiming> {
        let mut out = HashMap::new();
        for q in battery {
            eprintln!(
                "[{log_prefix}] cold: query {} — {iters} fresh-cache iters...",
                q.name
            );
            let query = q.terms.join(" ");
            let mode = to_infino_mode(q.mode);
            let mut open_samples = Vec::with_capacity(iters);
            let mut search_samples = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t_open = Instant::now();
                let guard = open_fresh();
                open_samples.push(t_open.elapsed());
                let t = Instant::now();
                let rows = guard.bm25_rows(column, &query, k, mode);
                search_samples.push(t.elapsed());
                std::hint::black_box(rows);
                drop(guard);
            }
            out.insert(
                q.name,
                ColdTiming {
                    open: p50(&mut open_samples),
                    search: p50(&mut search_samples),
                },
            );
        }
        out
    }

    fn warm_cells(stat: Option<&FtsQueryStat>) -> Vec<Cell> {
        match stat {
            Some(q) => {
                let ns = q.p50.as_secs_f64() * NS_PER_SEC;
                vec![
                    metric(ns, fmt_time(ns), Better::Lower),
                    metric(
                        q.rss.peak_rss_bytes as f64,
                        rss::fmt_bytes(q.rss.peak_rss_bytes),
                        Better::Lower,
                    ),
                    metric(
                        q.rss.median_rss_bytes as f64,
                        rss::fmt_bytes(q.rss.median_rss_bytes),
                        Better::Lower,
                    ),
                    metric(
                        q.rss.p90_rss_bytes as f64,
                        rss::fmt_bytes(q.rss.p90_rss_bytes),
                        Better::Lower,
                    ),
                ]
            }
            None => vec![text("—"), text("—"), text("—"), text("—")],
        }
    }

    fn search_row(
        name: &'static str,
        warm: Option<&HashMap<&'static str, FtsQueryStat>>,
        cold: Option<&HashMap<&'static str, ColdTiming>>,
    ) -> Vec<Cell> {
        let mut cells = vec![text(name)];
        if let Some(warm) = warm {
            cells.extend(warm_cells(warm.get(&name)));
        }
        if let Some(cold) = cold {
            match cold.get(&name) {
                Some(t) => {
                    let open_ns = t.open.as_secs_f64() * NS_PER_SEC;
                    let search_ns = t.search.as_secs_f64() * NS_PER_SEC;
                    cells.push(metric(open_ns, fmt_time(open_ns), Better::Lower));
                    cells.push(metric(search_ns, fmt_time(search_ns), Better::Lower));
                }
                None => {
                    cells.push(text("—"));
                    cells.push(text("—"));
                }
            }
        }
        cells
    }

    /// Render the OR/AND search table for either tier. `warm`/`cold` are
    /// each optional so a warm-only or cold-only run renders just its
    /// columns; `probes` is the infino-only per-algorithm block (passed
    /// only by the superfile runner).
    #[allow(clippy::too_many_arguments)]
    pub fn emit_search(
        report: &mut Report,
        anchor: &str,
        title: String,
        note: &str,
        warm: Option<&[FtsQueryStat]>,
        cold: Option<&HashMap<&'static str, ColdTiming>>,
        probes: Option<&[(&'static str, Duration, Duration)]>,
    ) {
        let warm_map: Option<HashMap<&'static str, FtsQueryStat>> =
            warm.map(|w| w.iter().map(|q| (q.name, q.clone())).collect());

        let mut header_cols = vec!["Query".to_string()];
        if warm_map.is_some() {
            header_cols.extend(
                ["warm", "Peak RSS", "Median RSS", "P90 RSS"]
                    .iter()
                    .map(|s| s.to_string()),
            );
        }
        if cold.is_some() {
            header_cols.push("cold open".to_string());
            header_cols.push("cold search".to_string());
        }

        let or_block = Block {
            subtitle: "OR queries".into(),
            headers: header_cols.clone(),
            rows: OR_QUERIES
                .iter()
                .map(|&n| search_row(n, warm_map.as_ref(), cold))
                .collect(),
        };
        let and_block = Block {
            subtitle: "AND queries".into(),
            headers: header_cols,
            rows: AND_QUERIES
                .iter()
                .map(|&n| search_row(n, warm_map.as_ref(), cold))
                .collect(),
        };
        let mut blocks = vec![or_block, and_block];
        if let Some(probes) = probes {
            blocks.push(Block {
                subtitle: "Per-algorithm probes (WAND+BMW vs MaxScore+BMM)".into(),
                headers: vec!["Shape".into(), "WAND+BMW".into(), "MaxScore+BMM".into()],
                rows: probes
                    .iter()
                    .map(|(shape, wand, bmm)| {
                        let w = wand.as_secs_f64() * NS_PER_SEC;
                        let b = bmm.as_secs_f64() * NS_PER_SEC;
                        vec![
                            text(*shape),
                            metric(w, fmt_time(w), Better::Lower),
                            metric(b, fmt_time(b), Better::Lower),
                        ]
                    })
                    .collect(),
            });
        }

        report.emit(&Section {
            anchor: anchor.into(),
            title,
            note: note.into(),
            blocks,
        });
    }
}

pub mod vector {
    use super::*;
    use std::hint::black_box;

    use infino::superfile::SuperfileReader;
    use infino::superfile::reader::VectorSearchOptions;
    use infino::supertable::Supertable;

    use crate::corpus::{self, Calibrated};
    use crate::markdown::fmt_time;
    use crate::report::{Better, Block, Cell, Report, Section, metric, text};
    use crate::rss::{self, PeakSampler, RssStats};

    /// Recall correctness gate (shared by both tiers).
    pub const CORRECTNESS_RECALL_FLOOR: f32 = 0.80;
    pub const CORRECTNESS_NPROBE: usize = 64;
    pub const CORRECTNESS_RERANK_MULT: usize = 256;
    pub const N_CORRECTNESS_QUERIES: usize = 20;
    /// Calibration battery + p50 reps per timed grid point.
    pub const N_CALIBRATION_QUERIES: usize = 100;
    pub const CALIBRATION_P50_ITERS: usize = 7;
    /// Recall targets reported (lowest-p50 point clearing each) + `default`.
    pub const RECALL_TARGETS: &[f32] = &[0.90, 0.95, 0.99];
    /// (probe, refine) calibration grid — one shape for both tiers.
    pub const PROBES: &[usize] = &[1, 5, 10, 25, 50, 100, 200, 400, 800];
    pub const REFINES: &[usize] = &[1, 4, 16, 64, 256, 1024];
    /// Query-generation seeds (must match the ingested corpus seed).
    pub const QUERY_CORRECTNESS_SEED: u64 = 17;
    pub const QUERY_CALIBRATION_SEED: u64 = 99;
    pub const QUERY_SIGMA: f32 = 0.05;

    const NS_PER_SEC: f64 = 1e9;

    pub fn search_opts(nprobe: usize, rerank_mult: usize) -> VectorSearchOptions {
        VectorSearchOptions::new()
            .with_nprobe(nprobe)
            .with_rerank_mult(rerank_mult)
    }

    /// A reader the vector executor runs kNN against, returning **global**
    /// `(doc_id, score)` hits so recall can be graded against brute-force
    /// ground truth regardless of how many superfiles back the reader.
    pub trait VectorRead {
        fn topk_global(
            &self,
            column: &str,
            query: &[f32],
            k: usize,
            nprobe: usize,
            rerank: usize,
        ) -> Vec<(u32, f32)>;
    }

    impl VectorRead for SuperfileReader {
        fn topk_global(
            &self,
            column: &str,
            query: &[f32],
            k: usize,
            nprobe: usize,
            rerank: usize,
        ) -> Vec<(u32, f32)> {
            // Single superfile: local_doc_id == global id.
            crate::tiers::block_on(self.vector_hits_async(
                column,
                query,
                k,
                search_opts(nprobe, rerank),
            ))
            .expect("superfile vector_search")
        }
    }

    impl VectorRead for Supertable {
        fn topk_global(
            &self,
            column: &str,
            query: &[f32],
            k: usize,
            nprobe: usize,
            rerank: usize,
        ) -> Vec<(u32, f32)> {
            let reader = self.reader();
            let hits = reader
                .vector_hits(column, query, k, search_opts(nprobe, rerank))
                .expect("supertable vector_hits");
            let manifest = reader.manifest();
            // Per-superfile global-id base offsets in manifest order.
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
                        .position(|e| e.uri == h.superfile)
                        .expect("hit superfile present in manifest");
                    (offsets[seg_idx] + h.local_doc_id, h.score)
                })
                .collect()
        }
    }

    /// Mean recall@k of a reader's hits vs brute-force ground truth.
    pub fn mean_recall<R: VectorRead>(
        reader: &R,
        column: &str,
        queries: &[Vec<f32>],
        truths: &[Vec<u32>],
        k: usize,
        nprobe: usize,
        rerank: usize,
    ) -> f32 {
        let mut sum = 0f32;
        for (q, t) in queries.iter().zip(truths) {
            let hits = reader.topk_global(column, q, k, nprobe, rerank);
            sum += corpus::recall_at_k(&hits, t);
        }
        sum / queries.len() as f32
    }

    /// Lowest-p50 `(probe, refine)` clearing `target_recall`; `None` if no
    /// grid point reaches it. Timing is p50 over a single query.
    pub fn calibrate<R: VectorRead>(
        reader: &R,
        column: &str,
        queries: &[Vec<f32>],
        truths: &[Vec<u32>],
        target_recall: f32,
        k: usize,
        log_prefix: &str,
    ) -> Option<Calibrated> {
        let mut best: Option<Calibrated> = None;
        let mut peak = 0f32;
        for &probe in PROBES {
            for &refine in REFINES {
                let recall = mean_recall(reader, column, queries, truths, k, probe, refine);
                peak = peak.max(recall);
                if recall < target_recall {
                    continue;
                }
                let q0 = &queries[0];
                let p50 = corpus::p50_micros(
                    || {
                        let _ = reader.topk_global(column, q0, k, probe, refine);
                    },
                    CALIBRATION_P50_ITERS,
                );
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
                "    [{log_prefix}] no point hit recall ≥ {target_recall:.2}; peak = {peak:.3}"
            );
        }
        best
    }

    /// Warm p50 (+ RSS) for one config on an already-warm reader.
    #[derive(Clone, Copy)]
    pub struct VecTiming {
        pub p50_ns: f64,
        pub rss: RssStats,
    }

    pub fn measure_warm<R: VectorRead>(
        reader: &R,
        column: &str,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank: usize,
    ) -> VecTiming {
        let sampler = PeakSampler::start_default();
        let _ = reader.topk_global(column, query, k, nprobe, rerank);
        let mut samples = Vec::with_capacity(CALIBRATION_P50_ITERS);
        for _ in 0..CALIBRATION_P50_ITERS {
            let t0 = Instant::now();
            let hits = reader.topk_global(column, query, k, nprobe, rerank);
            samples.push(t0.elapsed());
            black_box(hits);
        }
        let rss = sampler.stop_stats();
        VecTiming {
            p50_ns: p50(&mut samples).as_secs_f64() * NS_PER_SEC,
            rss,
        }
    }

    /// Cold p50s: `iters` fresh-reader opens, timing the open and one
    /// search separately (see [`ColdTiming`]).
    pub fn measure_cold<G: VectorRead>(
        open_fresh: &impl Fn() -> G,
        column: &str,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank: usize,
        iters: usize,
    ) -> ColdTiming {
        let mut open_samples = Vec::with_capacity(iters);
        let mut search_samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t_open = Instant::now();
            let guard = open_fresh();
            open_samples.push(t_open.elapsed());
            let t0 = Instant::now();
            let hits = guard.topk_global(column, query, k, nprobe, rerank);
            search_samples.push(t0.elapsed());
            black_box(hits);
            drop(guard);
        }
        ColdTiming {
            open: p50(&mut open_samples),
            search: p50(&mut search_samples),
        }
    }

    /// One rendered recall-table row.
    pub struct RecallRow {
        pub target: String,
        pub params: String,
        pub recall: String,
        pub warm: Option<VecTiming>,
        pub cold: Option<ColdTiming>,
    }

    fn time_cell(ns: f64) -> Cell {
        if ns.is_finite() {
            metric(ns, fmt_time(ns), Better::Lower)
        } else {
            text("—")
        }
    }

    fn rss_cells(stats: &RssStats) -> Vec<Cell> {
        vec![
            metric(
                stats.peak_rss_bytes as f64,
                rss::fmt_bytes(stats.peak_rss_bytes),
                Better::Lower,
            ),
            metric(
                stats.median_rss_bytes as f64,
                rss::fmt_bytes(stats.median_rss_bytes),
                Better::Lower,
            ),
            metric(
                stats.p90_rss_bytes as f64,
                rss::fmt_bytes(stats.p90_rss_bytes),
                Better::Lower,
            ),
        ]
    }

    /// Render the recall/latency table (same columns for both tiers):
    /// `Recall target | (p, r) | recall | [warm | Peak/Median/P90 RSS] | [cold]`.
    pub fn emit_recall_table(
        report: &mut Report,
        anchor: &str,
        title: String,
        note: &str,
        rows: &[RecallRow],
        include_warm: bool,
        include_cold: bool,
    ) {
        let mut headers = vec![
            "Recall target".to_string(),
            "(p, r)".to_string(),
            "recall".to_string(),
        ];
        if include_warm {
            headers.extend(
                ["warm", "Peak RSS", "Median RSS", "P90 RSS"]
                    .iter()
                    .map(|s| s.to_string()),
            );
        }
        if include_cold {
            headers.push("cold open".to_string());
            headers.push("cold search".to_string());
        }
        let body: Vec<Vec<Cell>> = rows
            .iter()
            .map(|r| {
                let mut cells = vec![text(&r.target), text(&r.params), text(&r.recall)];
                if include_warm {
                    match &r.warm {
                        Some(w) => {
                            cells.push(time_cell(w.p50_ns));
                            cells.extend(rss_cells(&w.rss));
                        }
                        None => cells.extend(std::iter::repeat_with(|| text("—")).take(4)),
                    }
                }
                if include_cold {
                    match r.cold {
                        Some(t) => {
                            cells.push(time_cell(t.open.as_secs_f64() * NS_PER_SEC));
                            cells.push(time_cell(t.search.as_secs_f64() * NS_PER_SEC));
                        }
                        None => {
                            cells.push(text("—"));
                            cells.push(text("—"));
                        }
                    }
                }
                cells
            })
            .collect();
        report.emit(&Section {
            anchor: anchor.into(),
            title,
            note: note.into(),
            blocks: vec![Block {
                subtitle: String::new(),
                headers,
                rows: body,
            }],
        });
    }

    /// Shared search driver: correctness gate, per-target calibration,
    /// warm + cold rows, and table emission. `warm_reader` is the
    /// already-warm reader both correctness and warm timing run against;
    /// `open_cold` yields a fresh cold reader per cold iteration.
    #[allow(clippy::too_many_arguments)]
    pub fn run_search<R: VectorRead, G: VectorRead>(
        report: &mut Report,
        warm_reader: &R,
        open_cold: impl Fn() -> G,
        column: &str,
        k: usize,
        default_nprobe: usize,
        default_rerank: usize,
        q_correct: &[Vec<f32>],
        gt_correct: &[Vec<u32>],
        q_cal: &[Vec<f32>],
        gt_cal: &[Vec<u32>],
        include_warm: bool,
        include_cold: bool,
        cold_iters: usize,
        log_prefix: &str,
        anchor: &str,
        title: String,
        note: &str,
    ) {
        eprintln!(
            "[{log_prefix}] correctness: recall@{k} on {} queries (nprobe={CORRECTNESS_NPROBE}, rerank={CORRECTNESS_RERANK_MULT})...",
            q_correct.len(),
        );
        let recall = mean_recall(
            warm_reader,
            column,
            q_correct,
            gt_correct,
            k,
            CORRECTNESS_NPROBE,
            CORRECTNESS_RERANK_MULT,
        );
        assert!(
            recall >= CORRECTNESS_RECALL_FLOOR,
            "{log_prefix} vector recall@{k} {recall:.3} < floor {CORRECTNESS_RECALL_FLOOR:.2}"
        );
        eprintln!("[{log_prefix}] correctness OK: recall@{k} = {recall:.3}");

        let cal: Vec<Option<Calibrated>> = RECALL_TARGETS
            .iter()
            .map(|&target| {
                eprintln!(
                    "[{log_prefix}] calibrating recall@{target:.2}: grid over probes/refines ({} queries)...",
                    q_cal.len(),
                );
                calibrate(warm_reader, column, q_cal, gt_cal, target, k, log_prefix)
            })
            .collect();

        let q0 = &q_cal[0];
        let mut rows: Vec<RecallRow> = Vec::new();
        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            match cal[i] {
                Some(c) => rows.push(RecallRow {
                    target: format!("{target:.2}"),
                    params: format!("p={}, r={}", c.probe, c.refine),
                    recall: format!("{:.3}", c.recall),
                    warm: include_warm
                        .then(|| measure_warm(warm_reader, column, q0, k, c.probe, c.refine)),
                    cold: include_cold.then(|| {
                        measure_cold(&open_cold, column, q0, k, c.probe, c.refine, cold_iters)
                    }),
                }),
                None => rows.push(RecallRow {
                    target: format!("{target:.2}"),
                    params: "—".into(),
                    recall: "—".into(),
                    warm: None,
                    cold: None,
                }),
            }
        }
        rows.push(RecallRow {
            target: "default".into(),
            params: format!("p={default_nprobe}, r={default_rerank}"),
            recall: "—".into(),
            warm: include_warm
                .then(|| measure_warm(warm_reader, column, q0, k, default_nprobe, default_rerank)),
            cold: include_cold.then(|| {
                measure_cold(
                    &open_cold,
                    column,
                    q0,
                    k,
                    default_nprobe,
                    default_rerank,
                    cold_iters,
                )
            }),
        });

        emit_recall_table(
            report,
            anchor,
            title,
            note,
            &rows,
            include_warm,
            include_cold,
        );
    }
}

pub mod sql {
    use super::*;
    use std::collections::HashMap;
    use std::hint::black_box;

    use infino::supertable::Supertable;

    use crate::harness::{InfinoSqlEngine, InfinoSqlIndex, SqlEngine, SqlQuery};
    use crate::markdown::{fmt_count, fmt_time};
    use crate::report::{Better, Block, Cell, Report, Section, metric, text};
    use crate::rss::{self, PeakSampler, RssStats};

    /// Timed query repetitions per query (after one warmup).
    pub const ITERS: usize = 10;

    const BUCKET_IN_ALL: &str = "('b0','b1','b2','b3','b4','b5','b6','b7','b8','b9')";

    /// Scalar SQL battery — aggregations + count-filters (read + compute,
    /// return few rows). Shared by both tiers' warm and cold paths.
    pub const SQL_BATTERY: &[SqlQuery] = &[
        SqlQuery {
            name: "agg_max_title",
            sql: "SELECT MAX(title) AS m FROM supertable",
        },
        SqlQuery {
            name: "filter_category_count",
            sql: "SELECT COUNT(*) AS n FROM supertable WHERE category = 'rust'",
        },
        SqlQuery {
            name: "filter_rating_count",
            sql: "SELECT COUNT(*) AS n FROM supertable WHERE rating < 10",
        },
        SqlQuery {
            name: "count_star",
            sql: "SELECT COUNT(*) AS n FROM supertable",
        },
        SqlQuery {
            name: "group_by_category",
            sql: "SELECT category, COUNT(*) AS n FROM supertable GROUP BY category",
        },
    ];

    /// Query literals that depend on the built corpus (sample row values).
    pub struct QueryInputs {
        pub qv: String,
        pub sample_title: String,
        pub sample_key: String,
    }

    /// A reader the SQL executor runs `query_sql` against (returns the
    /// materialized row count). Hides whether it's an in-memory superfile
    /// table or an object-store supertable consumer.
    pub trait SqlRead {
        fn query_rows(&self, sql: &str) -> usize;
        /// Run a one-row `SELECT COUNT(*)`-shaped aggregate and return the
        /// scalar `Int64` value — used by the correctness gate.
        fn query_count(&self, sql: &str) -> i64;
    }

    impl SqlRead for InfinoSqlIndex {
        fn query_rows(&self, sql: &str) -> usize {
            InfinoSqlEngine::read(self, sql).rows
        }
        fn query_count(&self, sql: &str) -> i64 {
            scalar_i64(
                &self
                    .table()
                    .reader()
                    .query_sql(sql)
                    .expect("query_sql count"),
            )
        }
    }

    impl SqlRead for Supertable {
        fn query_rows(&self, sql: &str) -> usize {
            self.reader()
                .query_sql(sql)
                .expect("query_sql")
                .iter()
                .map(|b| b.num_rows())
                .sum()
        }
        fn query_count(&self, sql: &str) -> i64 {
            scalar_i64(&self.reader().query_sql(sql).expect("query_sql count"))
        }
    }

    /// Extract the single `Int64` aggregate value from a one-row result.
    fn scalar_i64(batches: &[arrow_array::RecordBatch]) -> i64 {
        use arrow_array::{Array, Int64Array};
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("aggregate column is Int64")
            .value(0)
    }

    /// Correctness gate run on **both tiers** after the artifact is built:
    /// `COUNT(*)` must equal the row count, and the planted `rust`
    /// category (assigned by `doc_id % 4`) must match its expected share.
    pub fn assert_correct<R: SqlRead>(reader: &R, n_docs: usize, log_prefix: &str) {
        let total = reader.query_count("SELECT COUNT(*) AS n FROM supertable");
        assert_eq!(
            total, n_docs as i64,
            "[{log_prefix}] correctness: COUNT(*) {total} != {n_docs}"
        );
        let rust =
            reader.query_count("SELECT COUNT(*) AS n FROM supertable WHERE category = 'rust'");
        let expected = n_docs.div_ceil(4) as i64;
        assert_eq!(
            rust, expected,
            "[{log_prefix}] correctness: rust-category COUNT {rust} != {expected}"
        );
        eprintln!("[{log_prefix}] correctness OK: COUNT(*) == {n_docs}, rust == {rust}");
    }

    #[derive(Clone)]
    pub struct SqlQueryStat {
        pub name: &'static str,
        pub p50: Duration,
        pub rows: usize,
        pub rss: RssStats,
    }

    /// The full set of measured warm SQL query shapes.
    pub struct QuerySets {
        pub scalar: Vec<SqlQueryStat>,
        pub tvf: Vec<SqlQueryStat>,
        pub plain_scan: Vec<SqlQueryStat>,
        pub fts_pushdown: Vec<SqlQueryStat>,
        pub agg_scan: Vec<SqlQueryStat>,
        pub agg_idx: Vec<SqlQueryStat>,
    }

    fn timed<R: SqlRead>(reader: &R, name: &'static str, sql: &str, iters: usize) -> SqlQueryStat {
        let sampler = PeakSampler::start_default();
        let warm_rows = reader.query_rows(sql);
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let r = reader.query_rows(sql);
            samples.push(t0.elapsed());
            black_box(r);
        }
        let rss = sampler.stop_stats();
        SqlQueryStat {
            name,
            p50: p50(&mut samples),
            rows: warm_rows,
            rss,
        }
    }

    /// Measure every warm SQL query shape against `reader`. Identical for
    /// both tiers; only the reader differs.
    pub fn measure_query_sets<R: SqlRead>(
        reader: &R,
        inputs: &QueryInputs,
        iters: usize,
        log_prefix: &str,
    ) -> QuerySets {
        let qv = inputs.qv.as_str();
        let sample_title = inputs.sample_title.as_str();
        let sample_key = inputs.sample_key.as_str();

        eprintln!(
            "[{log_prefix}] scalar SQL battery ({} queries)...",
            SQL_BATTERY.len()
        );
        let scalar = SQL_BATTERY
            .iter()
            .map(|q| timed(reader, q.name, q.sql, iters))
            .collect();

        eprintln!(
            "[{log_prefix}] search table functions (bm25 / vector / hybrid / token / exact)..."
        );
        let tvf = vec![
            timed(
                reader,
                "bm25_search",
                "SELECT _id FROM bm25_search('title', 'term00001', 10)",
                iters,
            ),
            timed(
                reader,
                "vector_search",
                &format!("SELECT _id FROM vector_search('emb', '{qv}', 10)"),
                iters,
            ),
            timed(
                reader,
                "hybrid_search",
                &format!("SELECT _id FROM hybrid_search('title', 'term00001', 'emb', '{qv}', 10)"),
                iters,
            ),
            timed(
                reader,
                "token_match (all rows)",
                "SELECT _id FROM token_match('title', 'term00001 term00002', 'and')",
                iters,
            ),
            timed(
                reader,
                "token_match (selective)",
                "SELECT _id FROM token_match('title', 'doc0500000', 'and')",
                iters,
            ),
            timed(
                reader,
                "exact_match",
                &format!("SELECT _id FROM exact_match('title', '{sample_title}')"),
                iters,
            ),
        ];

        eprintln!(
            "[{log_prefix}] no-index vs FTS-index equality (sorted title vs unsorted key)..."
        );
        let plain_scan = vec![
            timed(
                reader,
                "WHERE title = ?  (sorted col, min/max prunes)",
                &format!("SELECT title FROM supertable WHERE title_noidx = '{sample_title}'"),
                iters,
            ),
            timed(
                reader,
                "WHERE key   = ?  (unsorted col, min/max defeated)",
                &format!("SELECT key FROM supertable WHERE key_noidx = '{sample_key}'"),
                iters,
            ),
        ];
        let fts_pushdown = vec![
            timed(
                reader,
                "WHERE title = ?  (sorted col, min/max prunes)",
                &format!("SELECT title FROM supertable WHERE title = '{sample_title}'"),
                iters,
            ),
            timed(
                reader,
                "WHERE key   = ?  (unsorted col, min/max defeated)",
                &format!("SELECT key FROM supertable WHERE key = '{sample_key}'"),
                iters,
            ),
        ];

        eprintln!(
            "[{log_prefix}] aggregate shapes over a candidate set: DataFusion only vs token_match..."
        );
        let agg_scan = vec![
            timed(
                reader,
                "COUNT(*)            key=? (1 row)",
                &format!("SELECT COUNT(*) AS a FROM supertable WHERE key_noidx = '{sample_key}'"),
                iters,
            ),
            timed(
                reader,
                "SUM(rating)         key=? (1 row)",
                &format!(
                    "SELECT SUM(rating) AS a FROM supertable WHERE key_noidx = '{sample_key}'"
                ),
                iters,
            ),
            timed(
                reader,
                "MAX(rating)         key=? (1 row)",
                &format!(
                    "SELECT MAX(rating) AS a FROM supertable WHERE key_noidx = '{sample_key}'"
                ),
                iters,
            ),
            timed(
                reader,
                "AVG(rating)         key=? (1 row)",
                &format!(
                    "SELECT AVG(rating) AS a FROM supertable WHERE key_noidx = '{sample_key}'"
                ),
                iters,
            ),
            timed(
                reader,
                "SUM(rating) bucket IN all (1M rows)",
                &format!(
                    "SELECT SUM(rating) AS a FROM supertable WHERE bucket_noidx IN {BUCKET_IN_ALL}"
                ),
                iters,
            ),
        ];
        let agg_idx = vec![
            timed(
                reader,
                "COUNT(*)            key=? (1 row)",
                &format!("SELECT COUNT(*) AS a FROM supertable WHERE key = '{sample_key}'"),
                iters,
            ),
            timed(
                reader,
                "SUM(rating)         key=? (1 row)",
                &format!("SELECT SUM(rating) AS a FROM supertable WHERE key = '{sample_key}'"),
                iters,
            ),
            timed(
                reader,
                "MAX(rating)         key=? (1 row)",
                &format!("SELECT MAX(rating) AS a FROM supertable WHERE key = '{sample_key}'"),
                iters,
            ),
            timed(
                reader,
                "AVG(rating)         key=? (1 row)",
                &format!("SELECT AVG(rating) AS a FROM supertable WHERE key = '{sample_key}'"),
                iters,
            ),
            timed(
                reader,
                "SUM(rating) bucket IN all (1M rows)",
                &format!("SELECT SUM(rating) AS a FROM supertable WHERE bucket IN {BUCKET_IN_ALL}"),
                iters,
            ),
        ];

        QuerySets {
            scalar,
            tvf,
            plain_scan,
            fts_pushdown,
            agg_scan,
            agg_idx,
        }
    }

    fn rss_cells(stats: &RssStats) -> Vec<Cell> {
        vec![
            metric(
                stats.peak_rss_bytes as f64,
                rss::fmt_bytes(stats.peak_rss_bytes),
                Better::Lower,
            ),
            metric(
                stats.median_rss_bytes as f64,
                rss::fmt_bytes(stats.median_rss_bytes),
                Better::Lower,
            ),
            metric(
                stats.p90_rss_bytes as f64,
                rss::fmt_bytes(stats.p90_rss_bytes),
                Better::Lower,
            ),
        ]
    }

    fn query_row(stat: &SqlQueryStat) -> Vec<Cell> {
        let ns = stat.p50.as_secs_f64() * 1e9;
        let mut cells = vec![
            text(stat.name),
            metric(ns, fmt_time(ns), Better::Lower),
            text(fmt_count(stat.rows)),
        ];
        cells.extend(rss_cells(&stat.rss));
        cells
    }

    fn query_headers() -> Vec<String> {
        vec![
            "Query".into(),
            "p50".into(),
            "Rows".into(),
            "Peak RSS".into(),
            "Median RSS".into(),
            "P90 RSS".into(),
        ]
    }

    fn block(subtitle: &str, stats: &[SqlQueryStat]) -> Block {
        Block {
            subtitle: subtitle.into(),
            headers: query_headers(),
            rows: stats.iter().map(query_row).collect(),
        }
    }

    /// Render the full warm SQL query table (same blocks for both tiers).
    pub fn emit_query(
        report: &mut Report,
        anchor: &str,
        title: String,
        note: &str,
        sets: &QuerySets,
    ) {
        report.emit(&Section {
            anchor: anchor.into(),
            title,
            note: note.into(),
            blocks: vec![
                block("Aggregations & count-filters (read + compute, return few rows — not the index A/B)", &sets.scalar),
                block("Plain Scan (DataFusion only) — selective equality, 1 row (sorted vs unsorted col)", &sets.plain_scan),
                block("FTS-pushdown (DataFusion + Infino) — SAME equality, 1 row (sorted vs unsorted col)", &sets.fts_pushdown),
                block("Aggregate over FTS candidates — Full Scan (DataFusion only)", &sets.agg_scan),
                block("Aggregate over FTS candidates — FTS-pushdown (DataFusion + Infino token_match)", &sets.agg_idx),
                block("Search table functions (bm25 / vector / hybrid / token / exact)", &sets.tvf),
            ],
        });
    }

    /// Cold scalar-battery p50s: `iters` fresh-reader opens per query,
    /// timing the open and the query separately (see [`ColdTiming`]).
    pub fn measure_cold<G: SqlRead>(
        open_fresh: impl Fn() -> G,
        iters: usize,
        log_prefix: &str,
    ) -> HashMap<&'static str, ColdTiming> {
        let mut out = HashMap::new();
        for q in SQL_BATTERY {
            eprintln!(
                "[{log_prefix}] cold: query {} — {iters} fresh-cache iters...",
                q.name
            );
            let mut open_samples = Vec::with_capacity(iters);
            let mut search_samples = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t_open = Instant::now();
                let guard = open_fresh();
                open_samples.push(t_open.elapsed());
                let t0 = Instant::now();
                let rows = guard.query_rows(q.sql);
                search_samples.push(t0.elapsed());
                black_box(rows);
                drop(guard);
            }
            out.insert(
                q.name,
                ColdTiming {
                    open: p50(&mut open_samples),
                    search: p50(&mut search_samples),
                },
            );
        }
        out
    }

    pub fn emit_cold(
        report: &mut Report,
        anchor: &str,
        title: String,
        note: &str,
        cold: &HashMap<&'static str, ColdTiming>,
    ) {
        let time_cell = |ns: f64| {
            if ns.is_finite() {
                metric(ns, fmt_time(ns), Better::Lower)
            } else {
                text("—")
            }
        };
        report.emit(&Section {
            anchor: anchor.into(),
            title,
            note: note.into(),
            blocks: vec![Block {
                subtitle: String::new(),
                headers: vec!["Query".into(), "cold open".into(), "cold search".into()],
                rows: SQL_BATTERY
                    .iter()
                    .map(|q| {
                        let (open_ns, search_ns) = cold
                            .get(&q.name)
                            .map(|t| (t.open.as_secs_f64() * 1e9, t.search.as_secs_f64() * 1e9))
                            .unwrap_or((f64::NAN, f64::NAN));
                        vec![text(q.name), time_cell(open_ns), time_cell(search_ns)]
                    })
                    .collect(),
            }],
        });
    }
}
