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

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::{Array, RecordBatch};

use crate::{
    cpu,
    markdown::fmt_time,
    report::{Better, Cell, context, metric, text},
    rss::{self, RssStats},
};

/// Rows and logical payload bytes of a query's returned batches — the values
/// the caller actually receives. `ArrayData::get_slice_memory_size` is
/// slice-aware (a batch sliced to its top-k counts only those rows) and sizes
/// the data exactly, so this is neither allocation footprint
/// (`get_array_memory_size`, which counts capacity slack) nor a serialized
/// wire size: the engine returns in-memory `RecordBatch`es and never
/// serializes, so transport framing belongs to whatever protocol the
/// embedding application speaks.
pub fn payload_bytes(batches: &[RecordBatch]) -> (u64, u64) {
    let rows = batches.iter().map(|b| b.num_rows() as u64).sum();
    let bytes = batches
        .iter()
        .flat_map(|batch| batch.columns())
        .map(|column| {
            let data = column.to_data();
            data.get_slice_memory_size()
                .unwrap_or_else(|_| data.get_buffer_memory_size()) as u64
        })
        .sum();
    (rows, bytes)
}

/// A warm-latency cell. All three warm metrics (p50 / p90 / p99) are
/// Δ-tracked equally here; which one *gates* the A/B regression decision is
/// chosen downstream by the summary, not at measurement time.
fn warm_time_cell(ns: f64) -> Cell {
    if ns.is_finite() {
        metric(ns, fmt_time(ns), Better::Lower)
    } else {
        text("—")
    }
}

/// p50 of a sample set (lower-median; matches the historical bench
/// definition shared by every runner).
pub fn p50(samples: &mut [Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort_unstable();
    samples[(samples.len() - 1) / 2]
}

/// Mean of per-iteration on-CPU seconds, or `None` when nothing was
/// sampled (e.g. `/proc/self/task` unavailable).
fn mean_opt(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    Some(samples.iter().sum::<f64>() / samples.len() as f64)
}

/// Per-iteration open/search wall + on-CPU samples for one cold measurement.
///
/// Shared by every tier's `measure_cold` (FTS / vector / SQL) so the p50 +
/// mean reduction and the [`ColdTiming`] shape live in exactly one place. A
/// cold query is mostly object-store I/O wait, so the cost ledger prices its
/// compute from the measured on-CPU means, not the wall p50s.
pub struct ColdSamples {
    open_wall: Vec<Duration>,
    search_wall: Vec<Duration>,
    open_cpu: Vec<f64>,
    search_cpu: Vec<f64>,
    search_get_count: Vec<u64>,
    search_get_bytes: Vec<u64>,
}

impl ColdSamples {
    pub fn with_capacity(iters: usize) -> Self {
        Self {
            open_wall: Vec::with_capacity(iters),
            search_wall: Vec::with_capacity(iters),
            open_cpu: Vec::with_capacity(iters),
            search_cpu: Vec::with_capacity(iters),
            search_get_count: Vec::with_capacity(iters),
            search_get_bytes: Vec::with_capacity(iters),
        }
    }

    /// Record one open's wall duration and (optional) measured on-CPU seconds.
    pub fn push_open(&mut self, wall: Duration, cpu: Option<f64>) {
        self.open_wall.push(wall);
        self.open_cpu.extend(cpu);
    }

    /// Record one first-search's wall duration and measured on-CPU seconds.
    /// The search window includes all on-CPU work during a cold query:
    /// fetch-path decode (TLS, decompress, CRC, cache write) plus IVF scoring.
    /// I/O wait is off-CPU and excluded by schedstat.
    pub fn push_search(&mut self, wall: Duration, cpu: Option<f64>) {
        self.search_wall.push(wall);
        self.search_cpu.extend(cpu);
    }

    /// Record one first-search's object-store GET count and downloaded
    /// bytes (process-default meter delta around the search call only).
    pub fn push_search_io(&mut self, get_count: u64, get_bytes: u64) {
        self.search_get_count.push(get_count);
        self.search_get_bytes.push(get_bytes);
    }

    pub fn finish(mut self) -> ColdTiming {
        ColdTiming {
            open: p50(&mut self.open_wall),
            search: p50(&mut self.search_wall),
            open_cpu_s: mean_opt(&self.open_cpu),
            search_cpu_s: mean_opt(&self.search_cpu),
            search_get_count: p50_u64(&mut self.search_get_count),
            search_get_bytes: p50_u64(&mut self.search_get_bytes),
        }
    }
}

/// Median of a `u64` sample set (sorts in place); `0` when empty.
fn p50_u64(samples: &mut [u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// p50 / p90 / p99 of a timed-sample set.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub p50: Duration,
    pub p90: Duration,
    pub p99: Duration,
}

/// Collect `iters` timings of `op`, ONE call per sample.
///
/// Calls are timed individually rather than in batches. A batched sampler
/// records each window's mean, so the percentiles computed from it are
/// percentiles OF MEANS: with a batch of 100, one slow call is diluted
/// hundredfold and the tail it represents disappears — exactly the signal
/// p90/p99 exist to show. Per-call timing costs one `Instant::now` pair per
/// sample (tens of ns, ~4% on the fastest sub-µs shapes measured here and
/// far less on the rest), which is the honest price of real tail numbers.
pub fn sample_batched<T>(iters: usize, mut op: impl FnMut() -> T) -> Vec<Duration> {
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        std::hint::black_box(op());
        samples.push(t.elapsed());
    }
    samples
}

/// Peak / median / p90 RSS cells. Peak gates; median and p90 are context.
fn rss_cells(stats: &RssStats) -> Vec<Cell> {
    vec![
        metric(
            stats.peak_rss_bytes as f64,
            rss::fmt_bytes(stats.peak_rss_bytes),
            Better::Lower,
        ),
        context(
            stats.median_rss_bytes as f64,
            rss::fmt_bytes(stats.median_rss_bytes),
            Better::Lower,
        ),
        context(
            stats.p90_rss_bytes as f64,
            rss::fmt_bytes(stats.p90_rss_bytes),
            Better::Lower,
        ),
    ]
}

/// Lower-median / nearest-rank p90 / nearest-rank p99 of a sample set
/// (sorts in place). At small sample counts nearest-rank p99 degenerates
/// to the max — the honest tail read for the samples taken.
pub fn summarize(samples: &mut [Duration]) -> Stats {
    let n = samples.len();
    if n == 0 {
        return Stats {
            p50: Duration::ZERO,
            p90: Duration::ZERO,
            p99: Duration::ZERO,
        };
    }
    samples.sort_unstable();
    let p90_rank = (9 * n).div_ceil(10).clamp(1, n);
    let p99_rank = (99 * n).div_ceil(100).clamp(1, n);
    Stats {
        p50: samples[(n - 1) / 2],
        p90: samples[p90_rank - 1],
        p99: samples[p99_rank - 1],
    }
}

/// Like [`sample_batched`], but also returns the amortized on-CPU seconds per
/// call (schedstat over the whole sample loop, divided by total invocations),
/// or `None` when schedstat is unavailable. One warm query is far too short to
/// sample on-CPU time precisely, so we measure the batch and divide — the same
/// figure the cost ledger prices warm and cold query compute from.
pub fn sample_batched_cpu<T>(
    iters: usize,
    mut op: impl FnMut() -> T,
) -> (Vec<Duration>, Option<f64>) {
    let mut samples = Vec::with_capacity(iters);
    let cpu0 = cpu::process_cpu_ns();
    for _ in 0..iters {
        let t = Instant::now();
        std::hint::black_box(op());
        samples.push(t.elapsed());
    }
    let cpu_s = cpu::cpu_seconds_since(cpu0).map(|s| s / iters as f64);
    (samples, cpu_s)
}

/// Cold timings for one query, split at the open/search boundary.
/// `open` measures whatever the caller's guard constructor does: for
/// guards that force-open (the single-superfile FTS guard, the
/// superfile-tier SQL guard via [`open_all_superfiles`]) it is the
/// consumer + manifest + every superfile reader; for the supertable
/// FTS/SQL cold guards and the cost-model cold-store closures it is
/// consumer + manifest CONSTRUCT ONLY (no `open_all_superfiles`), so the
/// query-driven survivor opens land in `search`. `search` is the first
/// query over the opened but data-cold table. Timed separately so cold
/// search latency never bills the one-time open bookkeeping.
#[derive(Clone, Copy)]
pub struct ColdTiming {
    pub open: Duration,
    pub search: Duration,
    /// Measured on-CPU seconds for the table-open window (all-thread schedstat
    /// delta), when sampled.
    pub open_cpu_s: Option<f64>,
    /// Measured on-CPU seconds for the first-search window (all-thread
    /// schedstat delta), when sampled. Includes fetch-path on-CPU work
    /// (decompress, CRC, cache write) plus scoring; excludes I/O wait.
    pub search_cpu_s: Option<f64>,
    /// Median object-store GET count of the first cold search across the
    /// timed iterations (process-default meter delta around the search
    /// call only — not the open). Zero when unmetered. This is what lets
    /// the cost model price each shape's cold request leg from the same
    /// battery the search table reports, instead of one representative.
    pub search_get_count: u64,
    /// Median downloaded bytes of the first cold search across iterations.
    pub search_get_bytes: u64,
}

/// Force-open every superfile reader on the consumer's pinned snapshot —
/// the "cold open" phase of a cold iteration. Runs the same per-superfile
/// open the query fan-out would lazily trigger (in-memory tier → disk
/// cache admit → lazy range-GET fallback), concurrently like the query
/// path, so the subsequent timed search pays only the search work.
pub fn open_all_superfiles(consumer: &infino::supertable::Supertable) {
    consumer.open_all_superfiles();
}

pub mod fts {
    use std::collections::HashMap;

    use infino::{
        storage::io_counters,
        superfile::{
            SuperfileReader,
            fts::{
                reader::BoolMode as InfinoBoolMode,
                tokenize::{AsciiLowerTokenizer, Tokenizer},
            },
        },
        supertable::SupertableReader,
    };

    use super::*;
    use crate::{
        cpu,
        harness::{BoolMode, FtsQuery},
        markdown::{fmt_count, fmt_time},
        report::{Better, Block, Cell, Report, Section, metric, text},
        rss::{PeakSampler, RssStats},
    };

    /// Nanoseconds per second, for time-cell formatting.
    const NS_PER_SEC: f64 = 1e9;

    /// Twenty mid-rank common terms (`term00050`..`term00069`) — a dense
    /// disjunction whose match set covers a large fraction of the corpus.
    /// Exercises the large-union count path, where a naive per-doc k-way
    /// merge degrades super-linearly in the term count.
    const TWENTY_COMMON_TERMS: &[&str] = &[
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
        "term00060",
        "term00061",
        "term00062",
        "term00063",
        "term00064",
        "term00065",
        "term00066",
        "term00067",
        "term00068",
        "term00069",
    ];

    /// Forty mid-rank common terms (`term00050`..`term00089`) — the extreme
    /// large-union shape; the count path's worst case at this scale.
    const FORTY_COMMON_TERMS: &[&str] = &[
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
        "term00060",
        "term00061",
        "term00062",
        "term00063",
        "term00064",
        "term00065",
        "term00066",
        "term00067",
        "term00068",
        "term00069",
        "term00070",
        "term00071",
        "term00072",
        "term00073",
        "term00074",
        "term00075",
        "term00076",
        "term00077",
        "term00078",
        "term00079",
        "term00080",
        "term00081",
        "term00082",
        "term00083",
        "term00084",
        "term00085",
        "term00086",
        "term00087",
        "term00088",
        "term00089",
    ];

    /// The full FTS query battery — single source of truth for both
    /// tiers' warm + cold search and the cross-engine recall grading.
    pub const FTS_BATTERY: &[FtsQuery] = &[
        FtsQuery {
            name: "single_rare",
            terms: &["term09999"],
            mode: BoolMode::Or,
        },
        FtsQuery {
            // doc-unique token for doc 0: df=1 at every scale (the corpus
            // plants doc{id:07} for id in 0..n_docs). A higher fixed id
            // (e.g. doc0500000) is absent below that many docs and would
            // silently measure an empty result at small scales.
            name: "single_df1",
            terms: &["doc0000000"],
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
            name: "twenty_term_or",
            terms: TWENTY_COMMON_TERMS,
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "forty_term_or",
            terms: FORTY_COMMON_TERMS,
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "two_term_and",
            terms: &["term00001", "term00050"],
            mode: BoolMode::And,
        },
        // Small intersection: two mid-rare terms whose overlap is well
        // under a top-1000 request, so the top-k heap never fills and
        // the block-max-AND skip cannot fire — every match is scored.
        // Isolates the raw score-and-collect cost that the large
        // `two_term_and` overlap hides behind pruning.
        FtsQuery {
            name: "two_term_and_small",
            terms: &["term00500", "term01000"],
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
        // Mixed clause shapes (`+must` + bare shoulds under Or): the
        // must intersection drives the walk, shoulds are scoring-only.
        FtsQuery {
            name: "must_common_should_common",
            terms: &["+term00050", "term00001"],
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "must_rare_should_common",
            terms: &["+term09999", "term00001"],
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "must_two_should_two",
            terms: &["+term00050", "+term00051", "term00001", "term00052"],
            mode: BoolMode::Or,
        },
        // Exact phrases (quoted atoms). The Zipfian corpus gives the
        // top terms frequent chance adjacency, so these measure the
        // real intersect-then-verify pipeline with non-empty results.
        FtsQuery {
            name: "phrase_two_common",
            terms: &[r#""term00001 term00002""#],
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "phrase_two_mixed",
            terms: &[r#""term00001 term00500""#],
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "phrase_three_common",
            terms: &[r#""term00001 term00002 term00003""#],
            mode: BoolMode::Or,
        },
        FtsQuery {
            name: "phrase_plus_must_term",
            terms: &[r#"+"term00001 term00002""#, "+term00010"],
            mode: BoolMode::Or,
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
        "twenty_term_or",
        "forty_term_or",
    ];

    /// AND query names, in table order.
    pub const AND_QUERIES: &[&str] = &[
        "two_term_and",
        "two_term_and_small",
        "three_wide_and",
        "three_similar_and",
        "five_term_and",
        "ten_term_and",
    ];

    /// Mixed must/should clause query names, in table order.
    pub const CLAUSE_QUERIES: &[&str] = &[
        "must_common_should_common",
        "must_rare_should_common",
        "must_two_should_two",
    ];

    /// Phrase query names, in table order.
    pub const PHRASE_QUERIES: &[&str] = &[
        "phrase_two_common",
        "phrase_two_mixed",
        "phrase_three_common",
        "phrase_plus_must_term",
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
    ///
    /// Two measurement surfaces per tier, mirroring the search-engine
    /// phases:
    ///
    ///   * [`bm25_rows`](FtsRead::bm25_rows) — the **query phase**:
    ///     id + score, no row materialization. Superfile = the raw
    ///     kernel (`bm25_hits_async`); supertable = the public
    ///     `bm25_search(.., None)` (bare projection — arithmetic `_id`
    ///     resolve, no Parquet).
    ///   * [`bm25_rows_fetched`](FtsRead::bm25_rows_fetched) — the
    ///     **fetch phase**: same search plus materializing the text
    ///     column for the top-k rows. Superfile = kernel +
    ///     `take_by_local_doc_ids`; supertable = the public
    ///     `bm25_search(.., Some([_id, column, score]))`.
    pub trait FtsRead {
        /// Query phase: one BM25 search returning id + score; the hit
        /// count is the black-box sink so the search is not optimized
        /// out.
        fn bm25_rows(&self, column: &str, query: &str, k: usize, mode: InfinoBoolMode) -> usize;

        /// Fetch phase: query + materialize the searched column for
        /// the top-k hits.
        fn bm25_rows_fetched(
            &self,
            column: &str,
            query: &str,
            k: usize,
            mode: InfinoBoolMode,
        ) -> usize;

        /// `(rows, payload_bytes)` of each phase's returned result — the
        /// search phase (id + score) and the fetch phase (+ top-k text) —
        /// so each cost class carries the payload it actually returns.
        fn bm25_payloads(
            &self,
            column: &str,
            query: &str,
            k: usize,
            mode: InfinoBoolMode,
        ) -> ((u64, u64), (u64, u64));

        /// Count phase: the matching-doc count from the dedicated count
        /// primitives — single-term `term_df` (O(1) from the dictionary
        /// header), multi-term `token_match` cardinality — with no BM25
        /// scoring and no row materialization. `query` is the raw query
        /// string so `+must` clause sigils resolve exactly as the
        /// production count path resolves them.
        fn count_matching(&self, column: &str, query: &str, mode: InfinoBoolMode) -> u64;
    }

    /// Fetch-phase measurement for a raw superfile reader: kernel hits,
    /// then materialize the searched column for the top-k rows. Shared
    /// by the warm reader impl and the cold guard so the two tiers of
    /// the superfile battery measure the identical operation.
    pub fn superfile_rows_fetched(
        reader: &SuperfileReader,
        column: &str,
        query: &str,
        k: usize,
        mode: InfinoBoolMode,
    ) -> usize {
        let hits = crate::tiers::block_on(reader.bm25_hits_async(column, query, k, mode))
            .expect("superfile bm25_search");
        if hits.is_empty() {
            return 0;
        }
        let locals: Vec<u32> = hits.iter().map(|&(doc, _)| doc).collect();
        reader
            .take_by_local_doc_ids(&locals, &[column])
            .expect("superfile take rows")
            .num_rows()
    }

    impl FtsRead for SuperfileReader {
        fn bm25_rows(&self, column: &str, query: &str, k: usize, mode: InfinoBoolMode) -> usize {
            crate::tiers::block_on(self.bm25_hits_async(column, query, k, mode))
                .expect("superfile bm25_search")
                .len()
        }

        fn bm25_rows_fetched(
            &self,
            column: &str,
            query: &str,
            k: usize,
            mode: InfinoBoolMode,
        ) -> usize {
            superfile_rows_fetched(self, column, query, k, mode)
        }

        fn bm25_payloads(
            &self,
            column: &str,
            query: &str,
            k: usize,
            mode: InfinoBoolMode,
        ) -> ((u64, u64), (u64, u64)) {
            // Raw-kernel tier: the query phase hands back `(doc, score)`
            // pairs — a caller payload like any other, just not Arrow-shaped.
            // The fetch phase additionally materializes the text column.
            let hits = crate::tiers::block_on(self.bm25_hits_async(column, query, k, mode))
                .expect("superfile bm25_search");
            // Measure the value actually handed back, not a per-hit constant.
            let search = (
                hits.len() as u64,
                std::mem::size_of_val(hits.as_slice()) as u64,
            );
            if hits.is_empty() {
                return (search, (0, 0));
            }
            let locals: Vec<u32> = hits.iter().map(|&(doc, _)| doc).collect();
            let batch = self
                .take_by_local_doc_ids(&locals, &[column])
                .expect("superfile take rows");
            (search, payload_bytes(std::slice::from_ref(&batch)))
        }

        fn count_matching(&self, column: &str, query: &str, mode: InfinoBoolMode) -> u64 {
            crate::tiers::block_on(async {
                // Resolve the match set exactly as the supertable count
                // does: with `+must` clauses, the count is the musts'
                // intersection (shoulds only affect scores, so they
                // never change which docs count); otherwise the bare
                // terms match under `mode`. Phrase atoms take the
                // phrase-aware walk.
                let clauses = AsciiLowerTokenizer.parse(query).into_clauses(mode);
                let has_musts = !clauses.musts.is_empty() || !clauses.must_phrases.is_empty();
                let (terms, phrases, eff_mode) = if has_musts {
                    (clauses.musts, clauses.must_phrases, InfinoBoolMode::And)
                } else {
                    (clauses.shoulds, clauses.should_phrases, mode)
                };
                if !phrases.is_empty() {
                    let refs: Vec<&str> = terms.iter().map(|t| &**t).collect();
                    let owned: Vec<Vec<String>> = phrases
                        .into_iter()
                        .map(|p| p.into_iter().map(|t| t.into_owned()).collect())
                        .collect();
                    return self
                        .atoms_match_count(column, &refs, &owned, eff_mode)
                        .await
                        .expect("superfile atoms_match_count");
                }
                let refs: Vec<&str> = terms.iter().map(|t| &**t).collect();
                // Single term: df is the exact match count, read O(1) from
                // the dictionary header. Multi-term: the dedicated count
                // primitive (union/intersection cardinality, no scoring,
                // no id materialization) — the same path the supertable
                // count uses, not `token_match().len()` (which would
                // materialize the id list through the slower merge walk).
                if refs.len() == 1 {
                    self.term_df(column, refs[0])
                        .await
                        .expect("superfile term_df")
                } else {
                    self.token_match_count(column, &refs, eff_mode)
                        .await
                        .expect("superfile token_match_count")
                }
            })
        }
    }

    impl FtsRead for SupertableReader {
        fn bm25_rows(&self, column: &str, query: &str, k: usize, mode: InfinoBoolMode) -> usize {
            self.bm25_search(
                column,
                query,
                k,
                mode,
                infino::Bm25Stats::PerSuperfile,
                None,
            )
            .expect("supertable bm25_search")
            .iter()
            .map(|b| b.num_rows())
            .sum()
        }

        fn bm25_rows_fetched(
            &self,
            column: &str,
            query: &str,
            k: usize,
            mode: InfinoBoolMode,
        ) -> usize {
            self.bm25_search(
                column,
                query,
                k,
                mode,
                infino::Bm25Stats::PerSuperfile,
                Some(&["_id", column, "score"]),
            )
            .expect("supertable bm25_search fetched")
            .iter()
            .map(|b| b.num_rows())
            .sum()
        }

        fn bm25_payloads(
            &self,
            column: &str,
            query: &str,
            k: usize,
            mode: InfinoBoolMode,
        ) -> ((u64, u64), (u64, u64)) {
            let search = self
                .bm25_search(
                    column,
                    query,
                    k,
                    mode,
                    infino::Bm25Stats::PerSuperfile,
                    None,
                )
                .expect("supertable bm25_search payload");
            let fetched = self
                .bm25_search(
                    column,
                    query,
                    k,
                    mode,
                    infino::Bm25Stats::PerSuperfile,
                    Some(&["_id", column, "score"]),
                )
                .expect("supertable bm25_search fetched payload");
            (payload_bytes(&search), payload_bytes(&fetched))
        }

        fn count_matching(&self, column: &str, query: &str, mode: InfinoBoolMode) -> u64 {
            self.count(column, query, mode).expect("supertable count")
        }
    }

    /// Warm timing (+ RSS) for one query: `warm` is the query phase (id +
    /// score), `fetched` the fetch phase (+ top-k text).
    #[derive(Clone, Debug)]
    pub struct FtsQueryStat {
        pub name: &'static str,
        pub warm: Stats,
        /// Fetch-phase (query + top-k text materialization) latency
        /// percentiles.
        pub fetched: Stats,
        /// Amortized on-CPU seconds of one warm query-phase search — the
        /// query's measured compute (cache hot, 0 GET), the basis for both the
        /// warm and cold query CPU cost.
        pub cpu_s: Option<f64>,
        /// Amortized on-CPU seconds of one warm fetch-phase call (query +
        /// top-k column materialization) — prices the retrieval class.
        pub fetched_cpu_s: Option<f64>,
        /// Return-payload of the query phase (id + score, no text): what a
        /// search-only caller receives.
        pub search_payload_rows: u64,
        pub search_payload_bytes: u64,
        /// Return-payload of the fetch phase (materialized top-k): row count
        /// and logical value bytes — the retrieval result the client
        /// receives, priced for egress.
        pub fetched_payload_rows: u64,
        pub fetched_payload_bytes: u64,
        pub rss: RssStats,
    }

    /// Untimed iterations before sampling, to reach steady state.
    const WARMUP_ITERS: usize = 5;

    /// One warm measurement of a query: query-phase `Stats` + fetch + RSS.
    fn measure_warm_once<R: FtsRead>(
        reader: &R,
        q: &FtsQuery,
        column: &str,
        k: usize,
        iters: usize,
    ) -> FtsQueryStat {
        let query = q.terms.join(" ");
        let mode = to_infino_mode(q.mode);
        for _ in 0..WARMUP_ITERS {
            std::hint::black_box(reader.bm25_rows(column, &query, k, mode));
        }
        let sampler = PeakSampler::start_default();
        let (mut samples, cpu_s) =
            sample_batched_cpu(iters, || reader.bm25_rows(column, &query, k, mode));
        for _ in 0..WARMUP_ITERS {
            std::hint::black_box(reader.bm25_rows_fetched(column, &query, k, mode));
        }
        let (mut fetched_samples, fetched_cpu_s) =
            sample_batched_cpu(iters, || reader.bm25_rows_fetched(column, &query, k, mode));
        // Both phases' return payloads, read from the ENGINE's result ledger
        // (untimed calls). Payload is cache-independent, so these warm figures
        // are the egress payloads for the cold path too.
        let (
            (search_payload_rows, search_payload_bytes),
            (fetched_payload_rows, fetched_payload_bytes),
        ) = reader.bm25_payloads(column, &query, k, mode);
        let rss = sampler.stop_stats();
        FtsQueryStat {
            name: q.name,
            warm: summarize(&mut samples),
            fetched: summarize(&mut fetched_samples),
            cpu_s,
            fetched_cpu_s,
            search_payload_rows,
            search_payload_bytes,
            fetched_payload_rows,
            fetched_payload_bytes,
            rss,
        }
    }

    /// Measure the warm battery against an already-warm reader once per query.
    pub fn measure_warm<R: FtsRead>(
        reader: &R,
        battery: &[FtsQuery],
        column: &str,
        k: usize,
        iters: usize,
        log_prefix: &str,
    ) -> Vec<FtsQueryStat> {
        eprintln!("[{log_prefix}] warm: {} queries...", battery.len());
        battery
            .iter()
            .map(|q| measure_warm_once(reader, q, column, k, iters))
            .collect()
    }

    /// Measure the cold battery: for each query, `iters` fresh-reader
    /// opens, timing the open and one search **separately** (see
    /// [`ColdTiming`]). `open_fresh` returns a guard that both
    /// implements [`FtsRead`] and owns the cache/consumer resources it
    /// must drop after the timed read; the guard's constructor performs
    /// the full open (consumer + superfile readers).
    /// Cold timings for one FTS shape, one per cost class: the query phase
    /// (search: id + score) and the fetch phase (retrieval: + top-k text),
    /// each measured on its OWN fresh opens so neither phase warms the
    /// other's cache.
    #[derive(Clone, Copy)]
    pub struct FtsColdStat {
        pub search: ColdTiming,
        /// `None` when the tier cannot run the fetch phase cold (the
        /// superfile tier's raw cold reader has no lazy row-take path; the
        /// production cold-fetch cost is measured at the supertable tier
        /// through the public `bm25_search` projection).
        pub fetched: Option<ColdTiming>,
    }

    pub fn measure_cold<G: FtsRead>(
        open_fresh: impl Fn() -> G,
        battery: &[FtsQuery],
        column: &str,
        k: usize,
        iters: usize,
        fetch_phase: bool,
        log_prefix: &str,
    ) -> HashMap<&'static str, FtsColdStat> {
        let mut out = HashMap::new();
        for q in battery {
            eprintln!(
                "[{log_prefix}] cold: query {} — {iters} fresh-cache iters × {} phase(s)...",
                q.name,
                if fetch_phase { 2 } else { 1 },
            );
            let query = q.terms.join(" ");
            let mode = to_infino_mode(q.mode);
            // Query phase (search: id + score) on its own fresh opens.
            let mut cold = ColdSamples::with_capacity(iters);
            for _ in 0..iters {
                let (guard, open_wall, open_cpu) = cpu::timed(&open_fresh);
                cold.push_open(open_wall, open_cpu);
                // Meter object-store GETs across the search call only (the
                // process-default meter counts every provider GET), so the
                // cost model can price each shape's cold request leg.
                let io_before = io_counters::snapshot();
                let (rows, search_wall, search_cpu) =
                    cpu::timed(|| guard.bm25_rows(column, &query, k, mode));
                let io = io_counters::snapshot().since(&io_before);
                cold.push_search(search_wall, search_cpu);
                cold.push_search_io(io.get_count, io.get_bytes);
                std::hint::black_box(rows);
                drop(guard);
            }
            // Fetch phase (retrieval: + top-k text) on SEPARATE fresh opens —
            // its cold cost includes the scalar column-page fetches the query
            // phase never pays.
            let fetched = fetch_phase.then(|| {
                let mut cold_fetch = ColdSamples::with_capacity(iters);
                for _ in 0..iters {
                    let (guard, open_wall, open_cpu) = cpu::timed(&open_fresh);
                    cold_fetch.push_open(open_wall, open_cpu);
                    let io_before = io_counters::snapshot();
                    let (rows, search_wall, search_cpu) =
                        cpu::timed(|| guard.bm25_rows_fetched(column, &query, k, mode));
                    let io = io_counters::snapshot().since(&io_before);
                    cold_fetch.push_search(search_wall, search_cpu);
                    cold_fetch.push_search_io(io.get_count, io.get_bytes);
                    std::hint::black_box(rows);
                    drop(guard);
                }
                cold_fetch.finish()
            });
            out.insert(
                q.name,
                FtsColdStat {
                    search: cold.finish(),
                    fetched,
                },
            );
        }
        out
    }

    /// "N / bytes" cell for a cold search's object-store reads.
    fn cold_io_cell(t: Option<&ColdTiming>) -> Cell {
        match t {
            Some(t) => text(format!(
                "{} / {}",
                t.search_get_count,
                crate::rss::fmt_bytes(t.search_get_bytes)
            )),
            None => text("—"),
        }
    }

    fn time_cell_opt(d: Option<Duration>) -> Cell {
        match d {
            Some(d) => {
                let ns = d.as_secs_f64() * NS_PER_SEC;
                metric(ns, fmt_time(ns), Better::Lower)
            }
            None => text("—"),
        }
    }

    /// ONE row per shape. Both cost classes ride as columns — the query phase
    /// (search: id + score) and the fetch phase (`+fetch`: same search plus
    /// materializing the top-k text) — and warm and cold sit side by side, so
    /// a shape's full economics read across a single line.
    fn search_row(
        name: &'static str,
        warm: Option<&HashMap<&'static str, FtsQueryStat>>,
        cold: Option<&HashMap<&'static str, FtsColdStat>>,
        resident: u64,
    ) -> Vec<Cell> {
        let w = warm.and_then(|m| m.get(&name));
        let c = cold.and_then(|m| m.get(&name));
        // Payload is only ever measured on the warm pass; a cold-only run
        // (`w` is `None`) has genuinely never sized this shape's result, so
        // its Payload/Egress cells render "—", never a fabricated "0 B"/"$0"
        // that reads as "this query returns nothing and costs nothing".
        let payloads = w.map(|q| (q.search_payload_bytes, q.fetched_payload_bytes));
        let search_payload = payloads.map(|(s, _)| s).unwrap_or(0);
        let fetch_payload = payloads.map(|(_, f)| f).unwrap_or(0);
        let mut cells = vec![text(name)];

        // ---- search phase (id + score)
        match payloads {
            Some((search_payload, _)) => {
                cells.push(text(crate::rss::fmt_bytes(search_payload)));
                cells.push(text(crate::cost::egress_cell_per_million(search_payload)));
            }
            None => cells.extend([text("—"), text("—")]),
        }
        match w {
            Some(q) => {
                for d in [q.warm.p50, q.warm.p90, q.warm.p99] {
                    cells.push(warm_time_cell(d.as_secs_f64() * NS_PER_SEC));
                }
                cells.push(text(crate::cost::warm_cell_per_million(
                    q.cpu_s,
                    q.warm.p50.as_secs_f64(),
                    resident,
                    search_payload,
                )));
            }
            None => cells.extend([text("—"), text("—"), text("—"), text("—")]),
        }

        // ---- fetch phase (+ top-k text)
        match payloads {
            Some((_, fetch_payload)) => cells.push(text(crate::rss::fmt_bytes(fetch_payload))),
            None => cells.push(text("—")),
        }
        match w {
            Some(q) => {
                cells.push(context(
                    q.fetched.p50.as_secs_f64() * NS_PER_SEC,
                    fmt_time(q.fetched.p50.as_secs_f64() * NS_PER_SEC),
                    Better::Lower,
                ));
                cells.push(text(crate::cost::warm_cell_per_million(
                    q.fetched_cpu_s,
                    q.fetched.p50.as_secs_f64(),
                    resident,
                    fetch_payload,
                )));
            }
            None => cells.extend([text("—"), text("—")]),
        }

        // ---- cold, both phases. RAM-hold window = the same-config warm p50.
        let warm_window = w.map(|q| q.warm.p50.as_secs_f64()).unwrap_or(0.0);
        let fetch_window = w.map(|q| q.fetched.p50.as_secs_f64()).unwrap_or(0.0);
        cells.push(time_cell_opt(c.map(|s| s.search.open)));
        cells.push(time_cell_opt(c.map(|s| s.search.search)));
        cells.push(cold_io_cell(c.map(|s| &s.search)));
        cells.push(match c {
            Some(s) => text(crate::cost::cold_cell_per_million(
                s.search.search_cpu_s,
                if warm_window > 0.0 {
                    warm_window
                } else {
                    s.search.search.as_secs_f64()
                },
                resident,
                s.search.search_get_count,
                search_payload,
            )),
            None => text("—"),
        });
        let cf = c.and_then(|s| s.fetched.as_ref());
        cells.push(time_cell_opt(cf.map(|t| t.search)));
        cells.push(cold_io_cell(cf));
        cells.push(match cf {
            Some(t) => text(crate::cost::cold_cell_per_million(
                t.search_cpu_s,
                if fetch_window > 0.0 {
                    fetch_window
                } else {
                    t.search.as_secs_f64()
                },
                resident,
                t.search_get_count,
                fetch_payload,
            )),
            None => text("—"),
        });
        cells
    }

    /// Render the unified per-family queries + cost table for either tier:
    /// one row per shape, with both cost classes as columns (search = id +
    /// score, `+fetch` = same search plus top-k text) and warm and cold side
    /// by side. Every row carries the payload it returns, its egress, and
    /// full per-query dollars (compute + requests + egress). `warm`/`cold`
    /// are each optional so a warm-only or cold-only run renders "—" in the
    /// other side's columns; `probes` is the infino-only per-algorithm block.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_search(
        report: &mut Report,
        anchor: &str,
        title: String,
        note: &str,
        warm: Option<&[FtsQueryStat]>,
        cold: Option<&HashMap<&'static str, FtsColdStat>>,
        probes: Option<&[(&'static str, Duration, Duration, Duration)]>,
    ) {
        let warm_map: Option<HashMap<&'static str, FtsQueryStat>> =
            warm.map(|w| w.iter().map(|q| (q.name, q.clone())).collect());
        let resident = crate::rss::current_anon_rss_bytes().unwrap_or(0);

        let header_cols: Vec<String> = [
            "Query",
            "Payload",
            "Egress $/1M",
            "warm p50",
            "warm p90",
            "warm p99",
            "Warm $/1M",
            "+fetch Payload",
            "+fetch p50",
            "+fetch $/1M",
            "cold open",
            "cold search",
            "cold GET/bytes",
            "Cold $/1M",
            "+fetch cold",
            "+fetch cold GET/bytes",
            "+fetch cold $/1M",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let family_rows = |names: &[&'static str]| -> Vec<Vec<Cell>> {
            names
                .iter()
                .map(|&n| search_row(n, warm_map.as_ref(), cold, resident))
                .collect()
        };

        let or_block = Block {
            subtitle: "OR queries".into(),
            headers: header_cols.clone(),
            rows: family_rows(OR_QUERIES),
        };
        let and_block = Block {
            subtitle: "AND queries".into(),
            headers: header_cols.clone(),
            rows: family_rows(AND_QUERIES),
        };
        let clause_block = Block {
            subtitle: "Must/should queries (+must, bare should)".into(),
            headers: header_cols.clone(),
            rows: family_rows(CLAUSE_QUERIES),
        };
        let phrase_block = Block {
            subtitle: "Phrase queries (exact adjacency)".into(),
            headers: header_cols,
            rows: family_rows(PHRASE_QUERIES),
        };
        let mut blocks = vec![or_block, and_block, clause_block, phrase_block];
        if let Some(probes) = probes {
            blocks.push(Block {
                subtitle: "Per-algorithm probes (WAND+BMW vs MaxScore+BMM vs windowed union)"
                    .into(),
                headers: vec![
                    "Shape".into(),
                    "WAND+BMW".into(),
                    "MaxScore+BMM".into(),
                    "Windowed union".into(),
                ],
                rows: probes
                    .iter()
                    .map(|(shape, wand, bmm, windowed)| {
                        let w = wand.as_secs_f64() * NS_PER_SEC;
                        let b = bmm.as_secs_f64() * NS_PER_SEC;
                        let u = windowed.as_secs_f64() * NS_PER_SEC;
                        vec![
                            text(*shape),
                            context(w, fmt_time(w), Better::Lower),
                            context(b, fmt_time(b), Better::Lower),
                            context(u, fmt_time(u), Better::Lower),
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

    /// Warm count timing for one query: `p50` is the dedicated count
    /// path's per-call p50; `n` is the matching-doc count it returned.
    #[derive(Clone, Debug)]
    pub struct CountStat {
        pub name: &'static str,
        pub p50: Duration,
        pub n: u64,
    }

    /// Measure the count battery against an already-warm reader: for
    /// each query, `iters` timed iterations of the dedicated count path.
    pub fn measure_count<R: FtsRead>(
        reader: &R,
        battery: &[FtsQuery],
        column: &str,
        iters: usize,
        log_prefix: &str,
    ) -> Vec<CountStat> {
        battery
            .iter()
            .map(|q| {
                eprintln!("[{log_prefix}] count: query {}...", q.name);
                let mode = to_infino_mode(q.mode);
                let query = q.terms.join(" ");
                let n = reader.count_matching(column, &query, mode);
                let mut samples = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let t = Instant::now();
                    let got = reader.count_matching(column, &query, mode);
                    samples.push(t.elapsed());
                    std::hint::black_box(got);
                }
                CountStat {
                    name: q.name,
                    p50: p50(&mut samples),
                    n,
                }
            })
            .collect()
    }

    fn count_row(name: &'static str, stats: &HashMap<&'static str, CountStat>) -> Vec<Cell> {
        match stats.get(&name) {
            Some(c) => {
                let ns = c.p50.as_secs_f64() * NS_PER_SEC;
                vec![
                    text(name),
                    text(fmt_count(c.n as usize)),
                    context(ns, fmt_time(ns), Better::Lower),
                ]
            }
            None => vec![text(name), text("—"), text("—")],
        }
    }

    /// Render the count battery: the dedicated count path's p50 per
    /// query, alongside the matching-doc count. infino-only — the same
    /// table shape for both tiers.
    pub fn emit_count(
        report: &mut Report,
        anchor: &str,
        title: String,
        note: &str,
        counts: &[CountStat],
    ) {
        let map: HashMap<&'static str, CountStat> =
            counts.iter().map(|c| (c.name, c.clone())).collect();
        let headers: Vec<String> = ["Query", "matches", "count()"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let or_block = Block {
            subtitle: "OR queries".into(),
            headers: headers.clone(),
            rows: OR_QUERIES.iter().map(|&n| count_row(n, &map)).collect(),
        };
        let and_block = Block {
            subtitle: "AND queries".into(),
            headers: headers.clone(),
            rows: AND_QUERIES.iter().map(|&n| count_row(n, &map)).collect(),
        };
        let clause_block = Block {
            subtitle: "Must/should queries (count = must intersection)".into(),
            headers: headers.clone(),
            rows: CLAUSE_QUERIES.iter().map(|&n| count_row(n, &map)).collect(),
        };
        let phrase_block = Block {
            subtitle: "Phrase queries (count = verified matches)".into(),
            headers,
            rows: PHRASE_QUERIES.iter().map(|&n| count_row(n, &map)).collect(),
        };
        report.emit(&Section {
            anchor: anchor.into(),
            title,
            note: note.into(),
            blocks: vec![or_block, and_block, clause_block, phrase_block],
        });
    }
}

pub mod vector {
    use std::{collections::HashMap, hint::black_box};

    use infino::{
        storage::io_counters,
        superfile::{SuperfileReader, reader::VectorSearchOptions},
        supertable::{
            manifest::list::PartitionStrategy, query::vector::USER_FINE_RUNS_PER_FRAGMENT,
        },
    };

    use super::*;
    use crate::{
        corpus::{self, Calibrated},
        cpu,
        markdown::fmt_time,
        report::{Better, Block, Cell, Report, Section, metric, text},
        rss::{PeakSampler, RssStats},
    };

    /// Recall correctness gate (high-nprobe sanity check). Temporarily
    /// lowered while the post-drain drain-clustering recall gap is under
    /// investigation (10M post-drain peaks ~0.975 at nprobe=64); restore to
    /// 0.98 once cell assignment is fixed.
    pub const CORRECTNESS_RECALL_FLOOR: f32 = 0.80;
    /// Default `(nprobe, rerank)` config gate — lower bar so large-scale
    /// pre-drain staging runs can complete while routing is tuned.
    pub const DEFAULT_CONFIG_RECALL_FLOOR: f32 = 0.80;
    pub const CORRECTNESS_NPROBE: usize = 64;
    pub const CORRECTNESS_RERANK_MULT: usize = 256;
    pub const N_CORRECTNESS_QUERIES: usize = 20;
    /// Calibration battery + p50 reps per timed grid point.
    pub const N_CALIBRATION_QUERIES: usize = 100;
    pub const CALIBRATION_P50_ITERS: usize = 7;
    /// Recall targets reported (lowest-p50 point clearing each) + `default`.
    pub const RECALL_TARGETS: &[f32] = &[0.90, 0.95, 0.98];
    /// (probe, refine) calibration grid — one shape for both tiers.
    pub const PROBES: &[usize] = &[1, 5, 10, 25, 50, 100, 200, 400, 800];
    pub const REFINES: &[usize] = &[1, 4, 16, 64, 256, 1024];
    /// Query-generation seeds (must match the ingested corpus seed).
    pub const QUERY_CORRECTNESS_SEED: u64 = 17;
    pub const QUERY_CALIBRATION_SEED: u64 = 99;
    pub const QUERY_SIGMA: f32 = 0.05;

    const NS_PER_SEC: f64 = 1e9;

    /// `nprobe` / `rerank` sentinel meaning "engine default — do not
    /// override". Rows measured with this value run
    /// `VectorSearchOptions::default()` exactly, so recorded numbers always
    /// reflect shipped defaults rather than bench-side overrides.
    pub const ENGINE_DEFAULT: usize = 0;

    pub fn default_search_opts() -> VectorSearchOptions {
        VectorSearchOptions::default()
    }

    pub fn search_opts(nprobe: usize, rerank_mult: usize) -> VectorSearchOptions {
        let mut opts = VectorSearchOptions::default();
        if nprobe != ENGINE_DEFAULT {
            opts = opts.with_nprobe(nprobe);
        }
        if rerank_mult != ENGINE_DEFAULT {
            opts = opts.with_rerank_mult(rerank_mult);
        }
        opts
    }

    /// A reader the vector executor runs kNN against, returning global
    /// dense `(doc_id, score)` hits for recall vs brute-force ground truth.
    pub trait VectorRead {
        fn topk_global(
            &self,
            column: &str,
            query: &[f32],
            k: usize,
            nprobe: usize,
            rerank: usize,
        ) -> Vec<(u32, f32)>;

        /// `(rows, payload_bytes)` of the returned id + score result — what a
        /// vector search hands back. The default measures the raw-kernel
        /// result (`(doc, score)` pairs); tiers returning Arrow override it.
        fn topk_payload(
            &self,
            column: &str,
            query: &[f32],
            k: usize,
            nprobe: usize,
            rerank: usize,
        ) -> (u64, u64) {
            let hits = self.topk_global(column, query, k, nprobe, rerank);
            // Measure the returned value itself, not a per-hit constant.
            (
                hits.len() as u64,
                std::mem::size_of_val(hits.as_slice()) as u64,
            )
        }

        /// Parameters the reader actually applies. Supertables may translate
        /// the requested IVF probe count into table-level cell routing.
        fn search_params(&self, nprobe: usize, rerank: usize) -> String {
            format!("p={nprobe}, r={rerank}")
        }

        /// Time the PUBLIC `vector_search` path (routing + rerank + stable
        /// `_id` resolution + Arrow materialization) — the surface a real
        /// caller uses. Returns p50 nanoseconds, or `None` for readers that
        /// only expose `topk_global` (superfile tier, cold guards). Used to
        /// quantify the `_id`-resolution cost that the `topk_global` /
        /// `vector_hits` latency excludes.
        fn full_search_p50_ns(
            &self,
            _column: &str,
            _query: &[f32],
            _k: usize,
            _nprobe: usize,
            _rerank: usize,
        ) -> Option<f64> {
            None
        }
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
            // Single superfile: local_doc_id == dense oracle id.
            crate::tiers::block_on(self.vector_hits_async(
                column,
                query,
                k,
                search_opts(nprobe, rerank),
            ))
            .expect("superfile vector_search")
        }

        fn search_params(&self, nprobe: usize, rerank: usize) -> String {
            let codec = self
                .vec()
                .and_then(|vector| vector.vector_columns_config().next())
                .map(|column| column.rerank_codec.name())
                .unwrap_or("none");
            format!("p={nprobe}, r={rerank}, codec={codec}")
        }
    }

    /// Supertable recall through the public `vector_search` surface: hits come
    /// back as stable `_id` + score; `id_to_dense` (one ordered `SELECT _id`
    /// scan) translates them to the dense oracle rows the brute-force ground
    /// truth speaks.
    pub struct SupertableVectorRead<'a> {
        pub table: &'a infino::supertable::Supertable,
        pub id_to_dense: Arc<HashMap<i128, u32>>,
    }

    impl SupertableVectorRead<'_> {
        pub fn routing_label(&self, nprobe: usize, rerank: usize) -> String {
            let rerank_label = if rerank == ENGINE_DEFAULT {
                "r=default".to_string()
            } else {
                format!("r{rerank}")
            };
            if let Some(hidden) = self.table.vector_index_table() {
                let reader = hidden.pinned_reader();
                let manifest = reader.manifest();
                if !manifest.get_all_superfiles().is_empty()
                    && let PartitionStrategy::VectorCell { routing, .. } =
                        manifest.get_partition_strategy()
                {
                    return format!(
                        "hidden: cells {}..{}, fine {}, {}, {rerank_label}",
                        routing.nprobe_min,
                        routing.nprobe_max,
                        routing.fine_nprobe,
                        hidden.options().vector_columns[0].rerank_codec.name(),
                    );
                }
            }
            let cells_label = if nprobe == ENGINE_DEFAULT {
                "routed 1 (default)".to_string()
            } else {
                format!("{nprobe}+")
            };
            format!(
                "user: cells {cells_label}, fine {USER_FINE_RUNS_PER_FRAGMENT}/fragment, {}, {rerank_label}",
                self.table.options().vector_columns[0].rerank_codec.name(),
            )
        }
    }

    impl VectorRead for SupertableVectorRead<'_> {
        fn topk_global(
            &self,
            column: &str,
            query: &[f32],
            k: usize,
            nprobe: usize,
            rerank: usize,
        ) -> Vec<(u32, f32)> {
            let batches = self
                .table
                .reader()
                .expect("reader")
                .vector_search(column, query, k, search_opts(nprobe, rerank), None, None)
                .expect("supertable vector_search");
            corpus::id_scores_from_vector_search(&batches)
                .into_iter()
                .map(|(id, score)| {
                    let dense = *self
                        .id_to_dense
                        .get(&id)
                        .unwrap_or_else(|| panic!("vector_search returned unknown _id {id}"));
                    (dense, score)
                })
                .collect()
        }

        fn topk_payload(
            &self,
            column: &str,
            query: &[f32],
            k: usize,
            nprobe: usize,
            rerank: usize,
        ) -> (u64, u64) {
            let batches = self
                .table
                .reader()
                .expect("reader")
                .vector_search(column, query, k, search_opts(nprobe, rerank), None, None)
                .expect("supertable vector_search payload");
            super::payload_bytes(&batches)
        }

        fn search_params(&self, nprobe: usize, rerank: usize) -> String {
            self.routing_label(nprobe, rerank)
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

    /// Largest doc count that still calibrates with the exhaustive
    /// 54-point grid sweep per target. Each grid point costs one full
    /// `mean_recall` battery (100 searches), so the sweep is fine on
    /// small corpora and pathological at scale — past this cap the
    /// staircase calibration below exploits recall/latency
    /// monotonicity to evaluate O(P + R) points instead of P × R × 3.
    pub const FULL_CALIBRATION_MAX_DOCS: usize = 1_000_000;

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

    /// Memoized `mean_recall` at one grid point — the unit of work the
    /// staircase walk economizes (one evaluation = a full query
    /// battery against the engine).
    #[allow(clippy::too_many_arguments)]
    fn eval_grid_point<R: VectorRead>(
        reader: &R,
        column: &str,
        queries: &[Vec<f32>],
        truths: &[Vec<u32>],
        k: usize,
        probe: usize,
        refine: usize,
        memo: &mut HashMap<(usize, usize), f32>,
        log_prefix: &str,
    ) -> f32 {
        if let Some(&r) = memo.get(&(probe, refine)) {
            return r;
        }
        // Announce BEFORE the work: one evaluation is a full query
        // battery (minutes at large scale), and a run that logs only
        // on completion is indistinguishable from a hung one.
        eprintln!(
            "    [{log_prefix}] staircase eval p={probe} r={refine} ({} queries)...",
            queries.len()
        );
        let recall = mean_recall(reader, column, queries, truths, k, probe, refine);
        eprintln!("    [{log_prefix}]   → recall {recall:.3}");
        memo.insert((probe, refine), recall);
        recall
    }

    /// Staircase calibration for corpora past
    /// [`FULL_CALIBRATION_MAX_DOCS`] — same outputs as running
    /// [`calibrate`] per target, at a fraction of the evaluations.
    ///
    /// Exploits the two monotonicities of IVF search:
    ///
    ///   * **recall** is non-decreasing in both `nprobe` and `rerank`,
    ///     so (a) one evaluation of the most expensive corner answers
    ///     reachability for every target, and (b) per target, the
    ///     minimum refine that clears is non-increasing as probe grows
    ///     — the clearing boundary is a staircase walkable in
    ///     O(P + R) evaluations instead of P × R;
    ///   * **latency** is increasing in both axes, so the lowest-p50
    ///     clearing point lies on that staircase frontier — only
    ///     frontier points pay the p50 timing loop.
    ///
    /// A memo cache shares evaluations and timings across the three
    /// targets, so the whole calibration costs ~O(P + R) engine
    /// batteries total.
    pub fn calibrate_staircase<R: VectorRead>(
        reader: &R,
        column: &str,
        queries: &[Vec<f32>],
        truths: &[Vec<u32>],
        k: usize,
        log_prefix: &str,
    ) -> Vec<Option<Calibrated>> {
        let mut recall_memo: HashMap<(usize, usize), f32> = HashMap::new();
        let mut p50_memo: HashMap<(usize, usize), f32> = HashMap::new();

        // No upfront reachability probe: it would pre-pay the single
        // most expensive grid point (max probe × max refine). The walk
        // answers reachability on its own — an unreachable target
        // misses across every row and its last evaluation IS that
        // corner; a reachable one never pays it at all.
        let p_max = *PROBES.last().expect("non-empty probe grid");
        let r_max = *REFINES.last().expect("non-empty refine grid");

        RECALL_TARGETS
            .iter()
            .map(|&target| {
                // Walk from (smallest probe, largest refine): a clear
                // step moves refine down (tighter), a miss moves probe
                // up (wider). Each row's minimal clearing refine joins
                // the frontier — at most min(P, R) + 1 points.
                let mut frontier: Vec<(usize, usize, f32)> = Vec::new();
                let mut p_i = 0usize;
                let mut r_i = REFINES.len() - 1;
                let mut row_clear: Option<(usize, f32)> = None;
                while p_i < PROBES.len() {
                    let recall = eval_grid_point(
                        reader,
                        column,
                        queries,
                        truths,
                        k,
                        PROBES[p_i],
                        REFINES[r_i],
                        &mut recall_memo,
                        log_prefix,
                    );
                    if recall >= target {
                        row_clear = Some((r_i, recall));
                        if r_i == 0 {
                            // Can't tighten refine further; wider
                            // probes only add latency at refine 0.
                            break;
                        }
                        r_i -= 1;
                    } else {
                        // Row's minimal clearing refine was the last
                        // clearing step (if any); move to next probe.
                        if let Some((ri, rec)) = row_clear.take() {
                            frontier.push((PROBES[p_i], REFINES[ri], rec));
                        }
                        p_i += 1;
                    }
                }
                if let Some((ri, rec)) = row_clear.take() {
                    frontier.push((PROBES[p_i.min(PROBES.len() - 1)], REFINES[ri], rec));
                }
                if frontier.is_empty() {
                    // No row cleared, so the walk's last evaluation was
                    // the (max probe, max refine) corner — the grid's
                    // recall ceiling.
                    let peak = recall_memo
                        .get(&(p_max, r_max))
                        .copied()
                        .unwrap_or_default();
                    eprintln!(
                        "    [{log_prefix}] no point hit recall ≥ {target:.2}; peak = {peak:.3}"
                    );
                    return None;
                }
                // Lowest-p50 frontier point wins; timings memoized
                // across targets (frontiers overlap heavily).
                let mut best: Option<Calibrated> = None;
                for (probe, refine, recall) in frontier {
                    let p50 = *p50_memo.entry((probe, refine)).or_insert_with(|| {
                        let q0 = &queries[0];
                        corpus::p50_micros(
                            || {
                                let _ = reader.topk_global(column, q0, k, probe, refine);
                            },
                            CALIBRATION_P50_ITERS,
                        )
                    });
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
                best
            })
            .collect()
    }

    /// Warm timing (+ RSS + measured on-CPU) for one config on an
    /// already-warm reader, gated on `warm.p50`. `cpu_s` is the amortized
    /// on-CPU seconds of one warm query — the query's true compute, measured
    /// (not a wall proxy). It's the basis for BOTH warm and cold query CPU
    /// cost: a cold query runs the identical scoring, so its compute equals
    /// this; the cold premium is I/O requests.
    #[derive(Clone, Copy)]
    pub struct VecTiming {
        pub warm: Stats,
        pub cpu_s: Option<f64>,
        /// Return-payload (id + score) of one query: rows and logical value
        /// bytes — the egress quantity. `(0, 0)` at the superfile tier (raw
        /// kernel, no Arrow result materialized).
        pub payload_rows: u64,
        pub payload_bytes: u64,
        pub rss: RssStats,
    }

    /// Untimed iterations before sampling, to reach steady state.
    const WARMUP_ITERS: usize = 5;
    const WARM_SAMPLE_ITERS: usize = 30;

    pub fn measure_warm<R: VectorRead>(
        reader: &R,
        column: &str,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank: usize,
    ) -> VecTiming {
        for _ in 0..WARMUP_ITERS {
            black_box(reader.topk_global(column, query, k, nprobe, rerank));
        }
        let dump_phases = io_counters::phase_enabled();
        let mut phase_sums: HashMap<&'static str, u64> = HashMap::new();
        let sampler = PeakSampler::start_default();
        let (mut samples, cpu_s) = if dump_phases {
            // Sample one query at a time so phase spans attribute to a single
            // warm iteration (batched sampling would merge concurrent queries).
            let mut walls = Vec::with_capacity(WARM_SAMPLE_ITERS);
            let mut cpu_acc = 0.0f64;
            for _ in 0..WARM_SAMPLE_ITERS {
                io_counters::phase_reset();
                let ((), wall, cpu) = cpu::timed(|| {
                    black_box(reader.topk_global(column, query, k, nprobe, rerank));
                });
                walls.push(wall);
                if let Some(c) = cpu {
                    cpu_acc += c;
                }
                for (name, us) in io_counters::phase_take_summed() {
                    *phase_sums.entry(name).or_default() += us;
                }
            }
            let cpu_s = Some(cpu_acc / WARM_SAMPLE_ITERS as f64);
            (walls, cpu_s)
        } else {
            sample_batched_cpu(WARM_SAMPLE_ITERS, || {
                reader.topk_global(column, query, k, nprobe, rerank)
            })
        };
        let rss = sampler.stop_stats();
        // One untimed call; payload comes from the ENGINE's result ledger.
        // Cache-independent, so this warm figure is the cold egress payload too.
        let (payload_rows, payload_bytes) = reader.topk_payload(column, query, k, nprobe, rerank);
        if dump_phases && !phase_sums.is_empty() {
            let n = WARM_SAMPLE_ITERS as f64;
            let mut names: Vec<_> = phase_sums.keys().copied().collect();
            names.sort_unstable();
            let parts: Vec<String> = names
                .into_iter()
                .map(|name| {
                    let avg_us = *phase_sums.get(name).unwrap_or(&0) as f64 / n;
                    format!("{name}={avg_us:.0}µs")
                })
                .collect();
            eprintln!(
                "[vector warm phases] avg over {WARM_SAMPLE_ITERS} queries (Σ across concurrent fan-out units): {}",
                parts.join("  ")
            );
        }
        VecTiming {
            warm: summarize(&mut samples),
            cpu_s,
            payload_rows,
            payload_bytes,
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
        let mut cold = ColdSamples::with_capacity(iters);
        for _ in 0..iters {
            let (guard, open_wall, open_cpu) = cpu::timed(open_fresh);
            cold.push_open(open_wall, open_cpu);
            // Meter the search window's object-store GETs, as the FTS and SQL
            // cold loops do — without this a cold vector query is priced with
            // no request leg at all, which is the dominant cold cost.
            let io_before = io_counters::snapshot();
            let (hits, search_wall, search_cpu) =
                cpu::timed(|| guard.topk_global(column, query, k, nprobe, rerank));
            let io = io_counters::snapshot().since(&io_before);
            cold.push_search(search_wall, search_cpu);
            cold.push_search_io(io.get_count, io.get_bytes);
            black_box(hits);
            drop(guard);
        }
        cold.finish()
    }

    /// One rendered recall-table row.
    pub struct RecallRow {
        pub target: String,
        pub params: String,
        pub recall: String,
        pub warm: Option<VecTiming>,
        pub cold: Option<ColdTiming>,
    }

    /// Gate latency cell (warm p50, cold search).
    fn time_cell(ns: f64) -> Cell {
        if ns.is_finite() {
            metric(ns, fmt_time(ns), Better::Lower)
        } else {
            text("—")
        }
    }

    /// Context latency cell (p90/p99, cold open).
    fn ctx_time_cell(ns: f64) -> Cell {
        if ns.is_finite() {
            context(ns, fmt_time(ns), Better::Lower)
        } else {
            text("—")
        }
    }

    /// Render the recall/latency table (same columns for both tiers):
    /// `Recall target | Search parameters | recall | [warm | Peak/Median/P90 RSS] | [cold]`.
    pub fn emit_recall_table(
        report: &mut Report,
        anchor: &str,
        title: String,
        note: &str,
        rows: &[RecallRow],
        include_warm: bool,
        include_cold: bool,
    ) {
        // Same column contract as the FTS and SQL tables: what the query
        // returns (payload), what that costs to ship (egress), and the full
        // per-query dollars warm and cold — one row per measured config.
        let resident = crate::rss::current_anon_rss_bytes().unwrap_or(0);
        let mut headers = vec![
            "Recall target".to_string(),
            "Search parameters".to_string(),
            "recall".to_string(),
            "Payload".to_string(),
            "Egress $/1M".to_string(),
        ];
        if include_warm {
            headers.extend(
                [
                    "warm p50",
                    "warm p90",
                    "warm p99",
                    "Warm $/1M",
                    "Peak RSS",
                    "Median RSS",
                    "P90 RSS",
                ]
                .iter()
                .map(|s| s.to_string()),
            );
        }
        if include_cold {
            headers.push("cold open".to_string());
            headers.push("cold search".to_string());
            headers.push("cold GET/bytes".to_string());
            headers.push("Cold $/1M".to_string());
        }
        let body: Vec<Vec<Cell>> = rows
            .iter()
            .map(|r| {
                // Payload is only ever measured on the warm pass; a
                // cold-only run (`r.warm` is `None`) never sized this
                // config's result, so Payload/Egress render "—", never a
                // fabricated "0 B"/"$0".
                let payload_measured = r.warm.as_ref().map(|w| w.payload_bytes);
                let payload = payload_measured.unwrap_or(0);
                let warm_window = r
                    .warm
                    .as_ref()
                    .map(|w| w.warm.p50.as_secs_f64())
                    .unwrap_or(0.0);
                let mut cells = vec![text(&r.target), text(&r.params), text(&r.recall)];
                match payload_measured {
                    Some(p) => {
                        cells.push(text(crate::rss::fmt_bytes(p)));
                        cells.push(text(crate::cost::egress_cell_per_million(p)));
                    }
                    None => cells.extend([text("—"), text("—")]),
                }
                if include_warm {
                    match &r.warm {
                        Some(w) => {
                            let p50_ns = w.warm.p50.as_secs_f64() * NS_PER_SEC;
                            let p90_ns = w.warm.p90.as_secs_f64() * NS_PER_SEC;
                            let p99_ns = w.warm.p99.as_secs_f64() * NS_PER_SEC;
                            cells.push(warm_time_cell(p50_ns));
                            cells.push(warm_time_cell(p90_ns));
                            cells.push(warm_time_cell(p99_ns));
                            cells.push(text(crate::cost::warm_cell_per_million(
                                w.cpu_s,
                                w.warm.p50.as_secs_f64(),
                                resident,
                                payload,
                            )));
                            cells.extend(rss_cells(&w.rss));
                        }
                        None => cells.extend(std::iter::repeat_with(|| text("—")).take(7)),
                    }
                }
                if include_cold {
                    match r.cold {
                        Some(t) => {
                            cells.push(ctx_time_cell(t.open.as_secs_f64() * NS_PER_SEC));
                            cells.push(time_cell(t.search.as_secs_f64() * NS_PER_SEC));
                            cells.push(text(format!(
                                "{} / {}",
                                t.search_get_count,
                                crate::rss::fmt_bytes(t.search_get_bytes)
                            )));
                            cells.push(text(crate::cost::cold_cell_per_million(
                                t.search_cpu_s,
                                if warm_window > 0.0 {
                                    warm_window
                                } else {
                                    t.search.as_secs_f64()
                                },
                                resident,
                                t.search_get_count,
                                payload,
                            )));
                        }
                        None => cells.extend(std::iter::repeat_with(|| text("—")).take(4)),
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
        n_docs: usize,
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
        skip_calibration: bool,
        log_prefix: &str,
        anchor: &str,
        title: String,
        note: &str,
    ) -> Vec<RecallRow> {
        // Representative query for the latency probes below. Prefer a
        // calibration query; fall back to the correctness set, which a
        // skip-calibration reopen loads from the oracle bin without any
        // calibration queries (so `q_cal` is legitimately empty there).
        let q0 = q_cal
            .first()
            .or_else(|| q_correct.first())
            .expect("run_search needs at least one held-out query");
        let mut rows: Vec<RecallRow> = Vec::new();
        let default_recall: Option<f32>;
        if skip_calibration {
            // Skip-calibration mode (both tiers' default — see each
            // tier's `RUN_CALIBRATION_GRID`): no recall-target grid; the
            // fixed `(default_nprobe, default_rerank)` recall sample IS
            // the gate, asserted against the floor below.
            if gt_correct.is_empty() {
                // No brute-force ground truth was built (skip-calibration / no
                // corpus). Recall is not measured — render "—", not a bogus 0.000.
                eprintln!(
                    "[{log_prefix}] skip-calibration: {} — no ground truth, recall not measured",
                    warm_reader.search_params(default_nprobe, default_rerank),
                );
                default_recall = None;
            } else {
                eprintln!(
                    "[{log_prefix}] skip-calibration: {} ({} queries)...",
                    warm_reader.search_params(default_nprobe, default_rerank),
                    q_correct.len(),
                );
                let default = mean_recall(
                    warm_reader,
                    column,
                    q_correct,
                    gt_correct,
                    k,
                    default_nprobe,
                    default_rerank,
                );
                eprintln!(
                    "[{log_prefix}] default-config: recall@{k} = {default:.3} (floor {DEFAULT_CONFIG_RECALL_FLOOR:.2})",
                );
                // The printed floor is a real gate, not decoration — this
                // was previously print-only, so a recall collapse in skip
                // mode sailed through green.
                assert!(
                    default >= DEFAULT_CONFIG_RECALL_FLOOR,
                    "{log_prefix} default-config vector recall@{k} {default:.3} < floor \
                     {DEFAULT_CONFIG_RECALL_FLOOR:.2}"
                );
                default_recall = Some(default);
            }
        } else {
            eprintln!(
                "[{log_prefix}] correctness: recall@{k} on {} queries ({})...",
                q_correct.len(),
                warm_reader.search_params(CORRECTNESS_NPROBE, CORRECTNESS_RERANK_MULT),
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

            eprintln!(
                "[{log_prefix}] default-config recall@{k} on {} queries (nprobe={default_nprobe}, rerank={default_rerank})...",
                q_correct.len(),
            );
            let default = mean_recall(
                warm_reader,
                column,
                q_correct,
                gt_correct,
                k,
                default_nprobe,
                default_rerank,
            );
            assert!(
                default >= DEFAULT_CONFIG_RECALL_FLOOR,
                "{log_prefix} default-config vector recall@{k} {default:.3} < floor {DEFAULT_CONFIG_RECALL_FLOOR:.2}"
            );
            eprintln!("[{log_prefix}] default-config OK: recall@{k} = {default:.3}");
            default_recall = Some(default);
            // Small corpora afford the exhaustive grid; past the cap the
            // staircase walk gets the same answers from O(P + R)
            // evaluations (see `calibrate_staircase`).
            let cal: Vec<Option<Calibrated>> = if n_docs <= FULL_CALIBRATION_MAX_DOCS {
                RECALL_TARGETS
                    .iter()
                    .map(|&target| {
                        eprintln!(
                            "[{log_prefix}] calibrating recall@{target:.2}: grid over probes/refines ({} queries)...",
                            q_cal.len(),
                        );
                        calibrate(warm_reader, column, q_cal, gt_cal, target, k, log_prefix)
                    })
                    .collect()
            } else {
                eprintln!(
                    "[{log_prefix}] calibrating {} targets: staircase walk over the (probe, refine) grid ({} queries)...",
                    RECALL_TARGETS.len(),
                    q_cal.len(),
                );
                calibrate_staircase(warm_reader, column, q_cal, gt_cal, k, log_prefix)
            };

            for (i, &target) in RECALL_TARGETS.iter().enumerate() {
                match cal[i] {
                    Some(c) => rows.push(RecallRow {
                        target: format!("{target:.2}"),
                        params: warm_reader.search_params(c.probe, c.refine),
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
        }
        rows.push(RecallRow {
            target: "default".into(),
            params: warm_reader.search_params(default_nprobe, default_rerank),
            recall: default_recall
                .map(|r| format!("{r:.3}"))
                .unwrap_or_else(|| "—".into()),
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

        // `topk_global` times the public `vector_search` path for supertable.
        if include_warm
            && let (Some(full_p50), Some(hits_p50)) = (
                warm_reader.full_search_p50_ns(column, q0, k, default_nprobe, default_rerank),
                rows.last()
                    .and_then(|r| r.warm.as_ref())
                    .map(|w| w.warm.p50.as_secs_f64() * NS_PER_SEC),
            )
            && (full_p50 - hits_p50).abs() > 1.0
        {
            eprintln!(
                "[{log_prefix}] public vector_search (_id-resolved) p50 = {} vs vector_hits p50 = {} (id-resolve delta = {})",
                fmt_time(full_p50),
                fmt_time(hits_p50),
                fmt_time((full_p50 - hits_p50).max(0.0)),
            );
        }

        emit_recall_table(
            report,
            anchor,
            title,
            note,
            &rows,
            include_warm,
            include_cold,
        );
        rows
    }
}

pub mod sql {
    use std::{collections::HashMap, hint::black_box};

    use infino::{storage::io_counters, supertable::Supertable};

    use super::*;
    use crate::{
        cpu,
        harness::{InfinoSqlEngine, InfinoSqlIndex, SqlEngine, SqlQuery},
        markdown::{fmt_count, fmt_time},
        report::{Better, Block, Cell, Report, Section, metric, text},
        rss::{PeakSampler, RssStats},
    };

    /// Timed query repetitions per query (after one warmup).
    pub const ITERS: usize = 30;

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

    /// Realistic row-returning WHERE queries built from the ingested
    /// sample row. Unlike the [`SQL_BATTERY`] aggregates — which the
    /// engine answers from manifest statistics (row count, exact value
    /// frequencies, min/max) and so touch ~0 row data — these SELECT
    /// columns behind a predicate, so they scan and fetch the row data
    /// a real lookup/range pays for. Shared by the warm WHERE-scan block
    /// and the cold battery so warm and cold measure the same shapes.
    /// Bulk row-set shape names — queries whose result scales with the match
    /// set (O(selectivity × corpus)), so their cost is dominated by GB
    /// returned. Named once here so the battery definitions and the serving
    /// family split can never drift apart.
    pub const BULK_RANGE_SCAN: &str = "WHERE rating < N (range scan, returns rows)";
    pub const BULK_TOKEN_MATCH_ALL: &str = "token_match (all rows)";

    /// The one classification of a bulk row-set shape by name. Both the
    /// warm/cold query table (this module) and the serving-cost family
    /// split (`supertable.rs`) call this rather than each re-deriving the
    /// same `name == BULK_RANGE_SCAN || name == BULK_TOKEN_MATCH_ALL` check,
    /// so the two tables can never classify the same shape differently.
    pub fn is_bulk_shape(name: &str) -> bool {
        name == BULK_RANGE_SCAN || name == BULK_TOKEN_MATCH_ALL
    }

    /// Scan-backed aggregates — realistic analytics shapes that provably
    /// DEFEAT the manifest-statistics fold (`covered_agg`), so they price the
    /// real cost of aggregation: column scans (warm CPU; cold data GETs).
    /// The manifest-answered battery ([`SQL_BATTERY`]) is the fold's
    /// best case — every shape there satisfies the fold's preconditions by
    /// corpus construction and touches zero data — so without these the
    /// reported "aggregation cost" is the fast-path floor, not the cost of
    /// aggregation. Fold-defeat mechanisms, per shape:
    ///   * rollup: the grouped fold accepts exactly `[COUNT(*)]` — AVG in the
    ///     aggregate list disqualifies it (full 2-column scan + hash agg);
    ///   * filtered metric: AVG is not the COUNT-only value-count shortcut,
    ///     and every segment's category min/max straddles `'rust'`, so all
    ///     segments classify boundary → the rewrite declines → full scan;
    ///   * title window: COUNT+SUM over a corpus-order range whose edges land
    ///     mid-segment — interior segments fold from stats, the two straddled
    ///     segments scan (the designed O(boundary) regime, otherwise
    ///     unmeasured);
    ///   * crosstab: two group columns — the grouped fold accepts exactly one.
    pub const SCAN_AGG_ROLLUP: &str = "AVG(rating) GROUP BY category (scan rollup)";
    pub const SCAN_AGG_FILTERED_AVG: &str = "AVG(rating) WHERE category=? (scan agg)";
    pub const SCAN_AGG_WINDOW: &str = "COUNT+SUM over title window (boundary scan)";
    pub const SCAN_AGG_CROSSTAB: &str = "COUNT(*) GROUP BY bucket, category (crosstab)";

    /// The boundary window spans the middle half of the corpus, with each
    /// edge offset to an odd multiple of `n_docs / 32`: ingest commits in
    /// `n_docs / 16` chunks, so odd multiples of `n_docs / 32` are chunk
    /// midpoints — the edges provably land inside a segment (boundary scan),
    /// never on a commit boundary (which would fold cleanly).
    const WINDOW_EDGE_32NDS: (usize, usize) = (9, 25);

    /// Search-TVF battery: top-k id + score through the SQL surface.
    pub fn tvf_battery(inputs: &QueryInputs) -> Vec<(&'static str, String)> {
        let qv = inputs.qv.as_str();
        let sample_title = inputs.sample_title.as_str();
        vec![
            (
                "bm25_search",
                "SELECT _id FROM bm25_search('title', 'term00001', 10)".to_string(),
            ),
            (
                "vector_search",
                format!("SELECT _id FROM vector_search('emb', '{qv}', 10)"),
            ),
            (
                "hybrid_search",
                format!("SELECT _id FROM hybrid_search('title', 'term00001', 'emb', '{qv}', 10)"),
            ),
            (
                BULK_TOKEN_MATCH_ALL,
                "SELECT _id FROM token_match('title', 'term00001 term00002', 'and')".to_string(),
            ),
            (
                // doc-unique token for doc 0 — df=1 at any scale (a higher
                // fixed id would match zero rows below that many docs and
                // measure an empty candidate set as a "selective" lookup).
                "token_match (selective)",
                "SELECT _id FROM token_match('title', 'doc0000000', 'and')".to_string(),
            ),
            (
                "exact_match",
                format!("SELECT _id FROM exact_match('title', '{sample_title}')"),
            ),
        ]
    }

    /// Aggregates over an FTS-pushdown candidate set (key=? one-row shapes
    /// plus the all-matching bucket IN scan).
    pub fn agg_candidates_battery(inputs: &QueryInputs) -> Vec<(&'static str, String)> {
        let sample_key = inputs.sample_key.as_str();
        vec![
            (
                "COUNT(*)            key=? (1 row)",
                format!("SELECT COUNT(*) AS a FROM supertable WHERE key = '{sample_key}'"),
            ),
            (
                "SUM(rating)         key=? (1 row)",
                format!("SELECT SUM(rating) AS a FROM supertable WHERE key = '{sample_key}'"),
            ),
            (
                "MAX(rating)         key=? (1 row)",
                format!("SELECT MAX(rating) AS a FROM supertable WHERE key = '{sample_key}'"),
            ),
            (
                "AVG(rating)         key=? (1 row)",
                format!("SELECT AVG(rating) AS a FROM supertable WHERE key = '{sample_key}'"),
            ),
            (
                // `bucket` is b{id % 10}, so `IN (b0..b9)` matches every row:
                // this scans+aggregates the whole table (n_docs rows, not a
                // fixed 1M) — the label states the shape, not a row count.
                "SUM(rating) bucket IN all (whole-table scan)",
                format!("SELECT SUM(rating) AS a FROM supertable WHERE bucket IN {BUCKET_IN_ALL}"),
            ),
        ]
    }

    /// The COMPLETE warm battery as (name, sql) — every shape the warm sets
    /// measure, in QuerySets order. The cold battery runs this verbatim so
    /// warm and cold cover identical shapes (one table, both sides).
    pub fn full_battery(inputs: &QueryInputs) -> Vec<(&'static str, String)> {
        SQL_BATTERY
            .iter()
            .map(|q| (q.name, q.sql.to_string()))
            .chain(scan_agg_battery(inputs.n_docs))
            .chain(scan_battery(&inputs.sample_key, &inputs.sample_title))
            .chain(agg_candidates_battery(inputs))
            .chain(tvf_battery(inputs))
            .collect()
    }

    /// Realistic scan-backed aggregate battery (see the shape consts above).
    /// Titles are `doc{doc_id:07}` in corpus order, so a title range is the
    /// bench's time-window analog. (Zero-padding keeps lexicographic ==
    /// numeric order up to 10M docs; past that the window degrades to an
    /// approximate range but still exercises the boundary-scan regime.)
    pub fn scan_agg_battery(n_docs: usize) -> Vec<(&'static str, String)> {
        let lo = WINDOW_EDGE_32NDS.0 * n_docs / 32;
        let hi = WINDOW_EDGE_32NDS.1 * n_docs / 32;
        vec![
            (
                SCAN_AGG_ROLLUP,
                "SELECT category, AVG(rating) AS avg_rating, COUNT(*) AS n FROM supertable \
                 GROUP BY category"
                    .to_string(),
            ),
            (
                SCAN_AGG_FILTERED_AVG,
                "SELECT AVG(rating) AS avg_rating FROM supertable WHERE category = 'rust'"
                    .to_string(),
            ),
            (
                SCAN_AGG_WINDOW,
                format!(
                    "SELECT COUNT(*) AS n, SUM(rating) AS s FROM supertable \
                     WHERE title >= 'doc{lo:07}' AND title < 'doc{hi:07}'"
                ),
            ),
            (
                SCAN_AGG_CROSSTAB,
                "SELECT bucket, category, COUNT(*) AS n FROM supertable \
                 GROUP BY bucket, category"
                    .to_string(),
            ),
        ]
    }

    pub fn scan_battery(sample_key: &str, sample_title: &str) -> Vec<(&'static str, String)> {
        let k = sample_key.replace('\'', "''");
        let t = sample_title.replace('\'', "''");
        vec![
            (
                "WHERE key = ? (point lookup, unsorted col)",
                format!("SELECT key, title, category, rating FROM supertable WHERE key = '{k}'"),
            ),
            (
                "WHERE title = ? (point lookup, sorted col)",
                format!("SELECT key, rating FROM supertable WHERE title = '{t}'"),
            ),
            (
                BULK_RANGE_SCAN,
                "SELECT title, rating FROM supertable WHERE rating < 10".to_string(),
            ),
        ]
    }

    /// High-cardinality GROUP BY guard, run only on the in-memory superfile
    /// tier (passed as `extra_scalar` to [`measure_query_sets`]). `title` is
    /// near-unique per row, so the group key set is ~n_docs: the shape where
    /// the aggregate's partial phase does almost no dedup yet re-hashes every
    /// key (the case `PARTIAL_AGG_SKIP_PROBE_RATIO` targets). Kept off the
    /// shared battery because a whole-table scan's working set blows the
    /// supertable object-store tier's warm-settle window; the in-memory tier
    /// has no such settle.
    pub const HIGH_CARD_SQL: &[SqlQuery] = &[SqlQuery {
        name: "group_by_title_highcard",
        sql: "SELECT title, COUNT(*) AS n FROM supertable GROUP BY title ORDER BY n DESC LIMIT 10",
    }];

    /// Query literals that depend on the built corpus (sample row values).
    pub struct QueryInputs {
        pub qv: String,
        pub sample_title: String,
        pub sample_key: String,
        /// Corpus row count — sizes the boundary-window aggregate's title
        /// range so its edges land mid-segment at any scale.
        pub n_docs: usize,
    }

    /// A reader the SQL executor runs `query_sql` against (returns the
    /// materialized row count). Hides whether it's an in-memory superfile
    /// table or an object-store supertable consumer.
    pub trait SqlRead {
        fn query_rows(&self, sql: &str) -> usize;
        /// `(rows, payload_bytes)` of the returned result set.
        fn query_payload(&self, sql: &str) -> (u64, u64);
        /// Run a one-row `SELECT COUNT(*)`-shaped aggregate and return the
        /// scalar `Int64` value — used by the correctness gate.
        fn query_count(&self, sql: &str) -> i64;
        /// Settle any background work the preceding (warmup) queries kicked
        /// off, so timed iterations measure the steady state instead of
        /// racing it. No-op for tiers with no background machinery.
        fn settle_warm(&self) {}
    }

    impl SqlRead for InfinoSqlIndex {
        fn query_rows(&self, sql: &str) -> usize {
            InfinoSqlEngine::read(self, sql).rows
        }
        fn query_payload(&self, sql: &str) -> (u64, u64) {
            payload_bytes(
                &self
                    .table()
                    .reader()
                    .expect("reader")
                    .query_sql(sql)
                    .expect("query_sql payload"),
            )
        }
        fn query_count(&self, sql: &str) -> i64 {
            scalar_i64(
                &self
                    .table()
                    .reader()
                    .expect("reader")
                    .query_sql(sql)
                    .expect("query_sql count"),
            )
        }
    }

    impl SqlRead for Supertable {
        fn query_rows(&self, sql: &str) -> usize {
            self.reader()
                .expect("reader")
                .query_sql(sql)
                .expect("query_sql")
                .iter()
                .map(|b| b.num_rows())
                .sum()
        }
        fn query_payload(&self, sql: &str) -> (u64, u64) {
            payload_bytes(
                &self
                    .reader()
                    .expect("reader")
                    .query_sql(sql)
                    .expect("query_sql payload"),
            )
        }
        fn query_count(&self, sql: &str) -> i64 {
            scalar_i64(
                &self
                    .reader()
                    .expect("reader")
                    .query_sql(sql)
                    .expect("query_sql count"),
            )
        }
        fn settle_warm(&self) {
            self.wait_until_warm(WARM_SETTLE_TIMEOUT)
                .expect("settle background fills");
        }
    }

    /// Generous ceiling for settling a single query's background fills;
    /// scoped to the query's own working set, so the typical wait is the
    /// few files that query opened — not the table.
    const WARM_SETTLE_TIMEOUT: Duration = Duration::from_secs(600);

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
        pub warm: Stats,
        pub rows: usize,
        /// Amortized on-CPU seconds of one warm query — the query's measured
        /// compute (cache hot), the basis for both warm and cold query CPU.
        pub cpu_s: Option<f64>,
        /// Logical value bytes of the result set (the egress quantity; row
        /// count is `rows`). Cache-independent, so it is the egress payload
        /// for the cold path too.
        pub payload_bytes: u64,
        pub rss: RssStats,
    }

    /// Untimed iterations before sampling, to reach steady state.
    const WARMUP_ITERS: usize = 5;

    /// The full set of measured warm SQL query shapes. Infino-only: the
    /// DataFusion-only control arms (plain scan, full-scan aggregates) were
    /// dropped so the bench tracks the engine's own FTS-pushdown path.
    pub struct QuerySets {
        pub scalar: Vec<SqlQueryStat>,
        pub tvf: Vec<SqlQueryStat>,
        pub fts_pushdown: Vec<SqlQueryStat>,
        pub agg_idx: Vec<SqlQueryStat>,
        /// Scan-backed aggregates ([`scan_agg_battery`]) — fold-ineligible
        /// shapes pricing real aggregation, vs the manifest-answered `scalar`
        /// battery which folds from statistics without touching data.
        pub agg_scan: Vec<SqlQueryStat>,
    }

    fn timed<R: SqlRead>(reader: &R, name: &'static str, sql: &str, iters: usize) -> SqlQueryStat {
        let mut warm_rows = 0;
        for _ in 0..WARMUP_ITERS {
            warm_rows = reader.query_rows(sql);
        }
        // Warmup opened this query's working set; settle its background
        // fills so the timed iterations measure steady state.
        reader.settle_warm();
        let sampler = PeakSampler::start_default();
        let (mut samples, cpu_s) = sample_batched_cpu(iters, || reader.query_rows(sql));
        let rss = sampler.stop_stats();
        // Result payload from the ENGINE's ledger (one extra untimed call).
        let (_, payload_bytes) = reader.query_payload(sql);
        SqlQueryStat {
            name,
            warm: summarize(&mut samples),
            rows: warm_rows,
            cpu_s,
            payload_bytes,
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
        extra_scalar: &[SqlQuery],
    ) -> QuerySets {
        let sample_title = inputs.sample_title.as_str();
        let sample_key = inputs.sample_key.as_str();

        eprintln!(
            "[{log_prefix}] scalar SQL battery ({} queries)...",
            SQL_BATTERY.len() + extra_scalar.len()
        );
        let scalar = SQL_BATTERY
            .iter()
            .chain(extra_scalar)
            .map(|q| timed(reader, q.name, q.sql, iters))
            .collect();

        eprintln!(
            "[{log_prefix}] search table functions (bm25 / vector / hybrid / token / exact)..."
        );
        let tvf = tvf_battery(inputs)
            .iter()
            .map(|(name, sql)| timed(reader, name, sql, iters))
            .collect::<Vec<_>>();

        eprintln!("[{log_prefix}] FTS-pushdown equality (sorted title vs unsorted key)...");
        // Realistic row-returning WHERE scans (point lookups + range),
        // shared with the cold battery so warm and cold price the same
        // shapes. These SELECT columns behind a predicate, so they scan
        // and fetch row data (unlike the manifest-answered aggregates).
        let fts_pushdown = scan_battery(sample_key, sample_title)
            .iter()
            .map(|(name, sql)| timed(reader, name, sql, iters))
            .collect::<Vec<_>>();

        eprintln!("[{log_prefix}] aggregate shapes over a token_match candidate set...");
        let agg_idx = agg_candidates_battery(inputs)
            .iter()
            .map(|(name, sql)| timed(reader, name, sql, iters))
            .collect::<Vec<_>>();

        eprintln!("[{log_prefix}] scan-backed aggregates (fold-ineligible)...");
        let agg_scan = scan_agg_battery(inputs.n_docs)
            .iter()
            .map(|(name, sql)| timed(reader, name, sql, iters))
            .collect::<Vec<_>>();

        QuerySets {
            scalar,
            tvf,
            fts_pushdown,
            agg_idx,
            agg_scan,
        }
    }

    /// One unified row: latency percentiles, rows, payload, egress, and full
    /// per-query dollars, warm and cold.
    fn query_row(
        stat: &SqlQueryStat,
        cold: Option<&HashMap<&'static str, ColdTiming>>,
        resident: u64,
    ) -> Vec<Cell> {
        let p50_ns = stat.warm.p50.as_secs_f64() * 1e9;
        let p90_ns = stat.warm.p90.as_secs_f64() * 1e9;
        let p99_ns = stat.warm.p99.as_secs_f64() * 1e9;
        let mut cells = vec![
            text(stat.name),
            text(fmt_count(stat.rows)),
            text(crate::rss::fmt_bytes(stat.payload_bytes)),
            text(crate::cost::egress_cell_per_million(stat.payload_bytes)),
            warm_time_cell(p50_ns),
            warm_time_cell(p90_ns),
            warm_time_cell(p99_ns),
            text(crate::cost::warm_cell_per_million(
                stat.cpu_s,
                stat.warm.p50.as_secs_f64(),
                resident,
                stat.payload_bytes,
            )),
        ];
        match cold.and_then(|m| m.get(stat.name)) {
            Some(t) => {
                let open_ns = t.open.as_secs_f64() * 1e9;
                let search_ns = t.search.as_secs_f64() * 1e9;
                cells.push(context(open_ns, fmt_time(open_ns), Better::Lower));
                cells.push(metric(search_ns, fmt_time(search_ns), Better::Lower));
                cells.push(text(format!(
                    "{} / {}",
                    t.search_get_count,
                    crate::rss::fmt_bytes(t.search_get_bytes)
                )));
                cells.push(text(crate::cost::cold_cell_per_million(
                    t.search_cpu_s,
                    stat.warm.p50.as_secs_f64(),
                    resident,
                    t.search_get_count,
                    stat.payload_bytes,
                )));
            }
            None => cells.extend([text("—"), text("—"), text("—"), text("—")]),
        }
        cells
    }

    fn query_headers() -> Vec<String> {
        [
            "Query",
            "Rows",
            "Payload",
            "Egress $/1M",
            "warm p50",
            "warm p90",
            "warm p99",
            "Warm $/1M",
            "cold open",
            "cold search",
            "cold GET/bytes",
            "Cold $/1M",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Render the full unified SQL queries + cost table (same class blocks
    /// for both tiers, warm and cold sides of every shape in one table).
    /// Bulk row-set shapes are split into their own block so no
    /// bounded-result class's rows sit next to a 100+ MiB result.
    pub fn emit_query(
        report: &mut Report,
        anchor: &str,
        title: String,
        note: &str,
        sets: &QuerySets,
        cold: Option<&HashMap<&'static str, ColdTiming>>,
    ) {
        let resident = crate::rss::current_anon_rss_bytes().unwrap_or(0);
        let block = |subtitle: &str, stats: &[&SqlQueryStat]| -> Block {
            Block {
                subtitle: subtitle.into(),
                headers: query_headers(),
                rows: stats.iter().map(|s| query_row(s, cold, resident)).collect(),
            }
        };
        let (bulk_scans, lookups): (Vec<&SqlQueryStat>, Vec<&SqlQueryStat>) = sets
            .fts_pushdown
            .iter()
            .partition(|s| is_bulk_shape(s.name));
        let (bulk_tvfs, tvf_idscore): (Vec<&SqlQueryStat>, Vec<&SqlQueryStat>) =
            sets.tvf.iter().partition(|s| is_bulk_shape(s.name));
        let bulk: Vec<&SqlQueryStat> = bulk_scans.into_iter().chain(bulk_tvfs).collect();
        let scalar: Vec<&SqlQueryStat> = sets.scalar.iter().collect();
        let agg_scan: Vec<&SqlQueryStat> = sets.agg_scan.iter().collect();
        let agg_idx: Vec<&SqlQueryStat> = sets.agg_idx.iter().collect();
        report.emit(&Section {
            anchor: anchor.into(),
            title,
            note: note.into(),
            blocks: vec![
                block(
                    "Analytics — manifest-answered (statistics fold, no scan)",
                    &scalar,
                ),
                block(
                    "Aggregates — scan-backed, fold-ineligible (rollup / filtered metric / window / crosstab)",
                    &agg_scan,
                ),
                block(
                    "Retrieval — point lookups (top-k rows; sorted vs unsorted col)",
                    &lookups,
                ),
                block(
                    "Aggregate over FTS candidates — FTS-pushdown (token_match)",
                    &agg_idx,
                ),
                block(
                    "Search TVFs, id+score (bm25 / vector / hybrid / token / exact)",
                    &tvf_idscore,
                ),
                block("Bulk row sets — GB-returned dominated", &bulk),
            ],
        });
    }

    /// Cold p50s for a `(name, sql)` battery: `iters` fresh-reader opens
    /// per query, timing the open and the query separately (see
    /// [`ColdTiming`]) and metering the search's object-store GETs. The
    /// caller supplies the battery — for SQL that is realistic
    /// row-returning WHERE scans ([`scan_battery`]) plus one labelled
    /// aggregate, not the manifest-answered aggregate battery.
    pub fn measure_cold<G: SqlRead>(
        open_fresh: impl Fn() -> G,
        battery: &[(&'static str, &str)],
        iters: usize,
        log_prefix: &str,
    ) -> HashMap<&'static str, ColdTiming> {
        let mut out = HashMap::new();
        for (name, sql) in battery {
            eprintln!("[{log_prefix}] cold: query {name} — {iters} fresh-cache iters...");
            let mut cold = ColdSamples::with_capacity(iters);
            for _ in 0..iters {
                let (guard, open_wall, open_cpu) = cpu::timed(&open_fresh);
                cold.push_open(open_wall, open_cpu);
                let io_before = io_counters::snapshot();
                let (rows, search_wall, search_cpu) = cpu::timed(|| guard.query_rows(sql));
                let io = io_counters::snapshot().since(&io_before);
                cold.push_search(search_wall, search_cpu);
                cold.push_search_io(io.get_count, io.get_bytes);
                black_box(rows);
                drop(guard);
            }
            out.insert(*name, cold.finish());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{sample_batched, summarize};

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn summarize_picks_median_p90_p99() {
        let mut s = [ms(5), ms(1), ms(3), ms(2), ms(4)];
        let out = summarize(&mut s);
        assert_eq!(out.p50, ms(3)); // lower-median of 5
        assert_eq!(out.p90, ms(5)); // nearest-rank ceil(0.9*5)=5
        assert_eq!(out.p99, ms(5)); // nearest-rank ceil(0.99*5)=5
    }

    #[test]
    fn summarize_single_and_empty() {
        assert_eq!(summarize(&mut [ms(7)]).p90, ms(7));
        let z = summarize(&mut []);
        assert_eq!((z.p50, z.p90, z.p99), (ms(0), ms(0), ms(0)));
    }

    #[test]
    fn sample_batched_returns_requested_count() {
        let s = sample_batched(8, || std::hint::black_box(1 + 1));
        assert_eq!(s.len(), 8);
    }
}
