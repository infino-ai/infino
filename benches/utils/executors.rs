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

use crate::{
    markdown::fmt_time,
    report::{Better, Cell, context, metric, text},
    rss::{self, RssStats},
};

/// A warm-latency cell. All three warm metrics (min / p50 / p90) are
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

/// Min / p50 / p90 of a timed-sample set.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub min: Duration,
    pub p50: Duration,
    pub p90: Duration,
}

/// Batch sub-µs ops up to this span so per-call `Instant::now` overhead
/// (tens of ns) can't dominate the sample.
const MIN_SAMPLE_NS: u64 = 50_000;
/// Cap on the auto-chosen batch size.
const MAX_BATCH: u64 = 100_000;

/// Collect `iters` per-call timings of `op`, batching calls so each timed
/// window spans at least [`MIN_SAMPLE_NS`]. A heavy op runs one call per
/// sample; a sub-µs op runs many and divides out — accurate either way.
pub fn sample_batched<T>(iters: usize, mut op: impl FnMut() -> T) -> Vec<Duration> {
    let probe = Instant::now();
    std::hint::black_box(op());
    let per_call_ns = (probe.elapsed().as_nanos() as u64).max(1);
    let batch = (MIN_SAMPLE_NS / per_call_ns).clamp(1, MAX_BATCH) as u32;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        for _ in 0..batch {
            std::hint::black_box(op());
        }
        samples.push(t.elapsed() / batch);
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

/// Min / lower-median / nearest-rank p90 of a sample set (sorts in place).
pub fn summarize(samples: &mut [Duration]) -> Stats {
    let n = samples.len();
    if n == 0 {
        return Stats {
            min: Duration::ZERO,
            p50: Duration::ZERO,
            p90: Duration::ZERO,
        };
    }
    samples.sort_unstable();
    let p90_rank = (9 * n).div_ceil(10).clamp(1, n);
    Stats {
        min: samples[0],
        p50: samples[(n - 1) / 2],
        p90: samples[p90_rank - 1],
    }
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
    let store = manifest.options.store.clone();
    let disk_cache = manifest.options.disk_cache.clone();
    let storage = manifest.options.storage.clone();
    // Snapshot the per-superfile open inputs up front so each spawned task
    // owns its data ('static). `tokio::spawn` per superfile distributes the
    // per-open CPU parse across the runtime's worker threads — matching the
    // production vector fan-out (`tokio::spawn` per superfile) instead of
    // serializing all the parses on a single `try_join_all` poller.
    let superfiles: Vec<_> = manifest
        .superfiles
        .iter()
        .map(|e| (e.uri, e.subsection_offsets.clone()))
        .collect();
    crate::tiers::block_on(async move {
        let handles: Vec<_> = superfiles
            .into_iter()
            .map(|(uri, offsets)| {
                let store = store.clone();
                let disk_cache = disk_cache.clone();
                let storage = storage.clone();
                tokio::spawn(async move {
                    infino::supertable::query::superfile_reader::superfile_reader(
                        &store,
                        disk_cache.as_ref(),
                        storage.as_ref(),
                        &uri,
                        offsets.as_ref(),
                    )
                    .await
                })
            })
            .collect();
        for h in handles {
            h.await
                .expect("cold open: join superfile open task")
                .expect("cold open: open superfile readers");
        }
    });
}

pub mod fts {
    use std::collections::HashMap;

    use infino::{
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

        fn count_matching(&self, column: &str, query: &str, mode: InfinoBoolMode) -> u64 {
            crate::tiers::block_on(async {
                // Resolve the match set exactly as the supertable count
                // does: with `+must` clauses, the count is the musts'
                // intersection (shoulds only affect scores, so they
                // never change which docs count); otherwise the bare
                // terms match under `mode`.
                let clauses = AsciiLowerTokenizer.parse(query).into_clauses(mode);
                let (terms, eff_mode) = if clauses.musts.is_empty() {
                    (clauses.shoulds, mode)
                } else {
                    (clauses.musts, InfinoBoolMode::And)
                };
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
            self.bm25_search(column, query, k, mode, None)
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
            self.bm25_search(column, query, k, mode, Some(&["_id", column, "score"]))
                .expect("supertable bm25_search fetched")
                .iter()
                .map(|b| b.num_rows())
                .sum()
        }

        fn count_matching(&self, column: &str, query: &str, mode: InfinoBoolMode) -> u64 {
            self.count(column, query, mode).expect("supertable count")
        }
    }

    /// Warm timing (+ RSS) for one query: `warm` is the query phase (id +
    /// score), `fetched_min` the fetch phase (+ top-k text).
    #[derive(Clone, Debug)]
    pub struct FtsQueryStat {
        pub name: &'static str,
        pub warm: Stats,
        pub fetched_min: Duration,
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
        let mut samples = sample_batched(iters, || reader.bm25_rows(column, &query, k, mode));
        for _ in 0..WARMUP_ITERS {
            std::hint::black_box(reader.bm25_rows_fetched(column, &query, k, mode));
        }
        let mut fetched_samples =
            sample_batched(iters, || reader.bm25_rows_fetched(column, &query, k, mode));
        let rss = sampler.stop_stats();
        FtsQueryStat {
            name: q.name,
            warm: summarize(&mut samples),
            fetched_min: summarize(&mut fetched_samples).min,
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
                let min_ns = q.warm.min.as_secs_f64() * NS_PER_SEC;
                let p50_ns = q.warm.p50.as_secs_f64() * NS_PER_SEC;
                let p90_ns = q.warm.p90.as_secs_f64() * NS_PER_SEC;
                let fetched_ns = q.fetched_min.as_secs_f64() * NS_PER_SEC;
                let mut cells = vec![
                    warm_time_cell(min_ns),
                    warm_time_cell(p50_ns),
                    warm_time_cell(p90_ns),
                    context(fetched_ns, fmt_time(fetched_ns), Better::Lower),
                ];
                cells.extend(rss_cells(&q.rss));
                cells
            }
            None => vec![
                text("—"),
                text("—"),
                text("—"),
                text("—"),
                text("—"),
                text("—"),
                text("—"),
            ],
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
                    cells.push(context(open_ns, fmt_time(open_ns), Better::Lower));
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
                [
                    "warm min",
                    "warm p50",
                    "warm p90",
                    "+fetch min",
                    "Peak RSS",
                    "Median RSS",
                    "P90 RSS",
                ]
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
            headers: header_cols.clone(),
            rows: AND_QUERIES
                .iter()
                .map(|&n| search_row(n, warm_map.as_ref(), cold))
                .collect(),
        };
        let clause_block = Block {
            subtitle: "Must/should queries (+must, bare should)".into(),
            headers: header_cols,
            rows: CLAUSE_QUERIES
                .iter()
                .map(|&n| search_row(n, warm_map.as_ref(), cold))
                .collect(),
        };
        let mut blocks = vec![or_block, and_block, clause_block];
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
                            context(w, fmt_time(w), Better::Lower),
                            context(b, fmt_time(b), Better::Lower),
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
            headers,
            rows: CLAUSE_QUERIES.iter().map(|&n| count_row(n, &map)).collect(),
        };
        report.emit(&Section {
            anchor: anchor.into(),
            title,
            note: note.into(),
            blocks: vec![or_block, and_block, clause_block],
        });
    }
}

pub mod vector {
    use std::{collections::HashMap, hint::black_box};

    use infino::{
        superfile::{SuperfileReader, reader::VectorSearchOptions},
        supertable::Supertable,
    };

    use super::*;
    use crate::{
        corpus::{self, Calibrated},
        markdown::fmt_time,
        report::{Better, Block, Cell, Report, Section, metric, text},
        rss::{PeakSampler, RssStats},
    };

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
                .vector_hits(column, query, k, search_opts(nprobe, rerank), None)
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

    /// Warm timing (+ RSS) for one config on an already-warm reader,
    /// gated on `warm.min`.
    #[derive(Clone, Copy)]
    pub struct VecTiming {
        pub warm: Stats,
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
        let sampler = PeakSampler::start_default();
        let mut samples = sample_batched(WARM_SAMPLE_ITERS, || {
            reader.topk_global(column, query, k, nprobe, rerank)
        });
        let rss = sampler.stop_stats();
        VecTiming {
            warm: summarize(&mut samples),
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

    /// Gate latency cell (warm min, cold search).
    fn time_cell(ns: f64) -> Cell {
        if ns.is_finite() {
            metric(ns, fmt_time(ns), Better::Lower)
        } else {
            text("—")
        }
    }

    /// Context latency cell (p50/p90, cold open).
    fn ctx_time_cell(ns: f64) -> Cell {
        if ns.is_finite() {
            context(ns, fmt_time(ns), Better::Lower)
        } else {
            text("—")
        }
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
                [
                    "warm min",
                    "warm p50",
                    "warm p90",
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
        }
        let body: Vec<Vec<Cell>> = rows
            .iter()
            .map(|r| {
                let mut cells = vec![text(&r.target), text(&r.params), text(&r.recall)];
                if include_warm {
                    match &r.warm {
                        Some(w) => {
                            let min_ns = w.warm.min.as_secs_f64() * NS_PER_SEC;
                            let p50_ns = w.warm.p50.as_secs_f64() * NS_PER_SEC;
                            let p90_ns = w.warm.p90.as_secs_f64() * NS_PER_SEC;
                            cells.push(warm_time_cell(min_ns));
                            cells.push(warm_time_cell(p50_ns));
                            cells.push(warm_time_cell(p90_ns));
                            cells.extend(rss_cells(&w.rss));
                        }
                        None => cells.extend(std::iter::repeat_with(|| text("—")).take(6)),
                    }
                }
                if include_cold {
                    match r.cold {
                        Some(t) => {
                            cells.push(ctx_time_cell(t.open.as_secs_f64() * NS_PER_SEC));
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
        let q0 = &q_cal[0];
        let mut rows: Vec<RecallRow> = Vec::new();
        let default_recall: Option<f32>;
        if skip_calibration {
            // Skip-calibration mode (INFINO_BENCH_SKIP_CALIBRATION): no
            // high-recall correctness gate, no recall-target grid — only
            // the fixed `(default_nprobe, default_rerank)` recall sample.
            eprintln!(
                "[{log_prefix}] skip-calibration: default-config recall@{k} at p={default_nprobe}, r={default_rerank} ({} queries)...",
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
                "[{log_prefix}] default-config: recall@{k} = {default:.3} (floor {CORRECTNESS_RECALL_FLOOR:.2})",
            );
            default_recall = Some(default);
        } else {
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
                default >= CORRECTNESS_RECALL_FLOOR,
                "{log_prefix} default-config vector recall@{k} {default:.3} < floor {CORRECTNESS_RECALL_FLOOR:.2}"
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
        }
        rows.push(RecallRow {
            target: "default".into(),
            params: format!("p={default_nprobe}, r={default_rerank}"),
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

    use infino::supertable::Supertable;

    use super::*;
    use crate::{
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
        pub warm: Stats,
        pub rows: usize,
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
    }

    fn timed<R: SqlRead>(reader: &R, name: &'static str, sql: &str, iters: usize) -> SqlQueryStat {
        let mut warm_rows = 0;
        for _ in 0..WARMUP_ITERS {
            warm_rows = reader.query_rows(sql);
        }
        let sampler = PeakSampler::start_default();
        let mut samples = sample_batched(iters, || reader.query_rows(sql));
        let rss = sampler.stop_stats();
        SqlQueryStat {
            name,
            warm: summarize(&mut samples),
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

        eprintln!("[{log_prefix}] FTS-pushdown equality (sorted title vs unsorted key)...");
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

        eprintln!("[{log_prefix}] aggregate shapes over a token_match candidate set...");
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
            fts_pushdown,
            agg_idx,
        }
    }

    fn query_row(stat: &SqlQueryStat) -> Vec<Cell> {
        let min_ns = stat.warm.min.as_secs_f64() * 1e9;
        let p50_ns = stat.warm.p50.as_secs_f64() * 1e9;
        let p90_ns = stat.warm.p90.as_secs_f64() * 1e9;
        let mut cells = vec![
            text(stat.name),
            warm_time_cell(min_ns),
            warm_time_cell(p50_ns),
            warm_time_cell(p90_ns),
            text(fmt_count(stat.rows)),
        ];
        cells.extend(rss_cells(&stat.rss));
        cells
    }

    fn query_headers() -> Vec<String> {
        vec![
            "Query".into(),
            "warm min".into(),
            "warm p50".into(),
            "warm p90".into(),
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
                block(
                    "Aggregations & count-filters (read + compute, return few rows)",
                    &sets.scalar,
                ),
                block(
                    "WHERE equality, FTS-pushdown — selective, 1 row (sorted vs unsorted col)",
                    &sets.fts_pushdown,
                ),
                block(
                    "Aggregate over FTS candidates — FTS-pushdown (token_match)",
                    &sets.agg_idx,
                ),
                block(
                    "Search table functions (bm25 / vector / hybrid / token / exact)",
                    &sets.tvf,
                ),
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{sample_batched, summarize};

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn summarize_picks_min_median_p90() {
        let mut s = [ms(5), ms(1), ms(3), ms(2), ms(4)];
        let out = summarize(&mut s);
        assert_eq!(out.min, ms(1));
        assert_eq!(out.p50, ms(3)); // lower-median of 5
        assert_eq!(out.p90, ms(5)); // nearest-rank ceil(0.9*5)=5
    }

    #[test]
    fn summarize_single_and_empty() {
        assert_eq!(summarize(&mut [ms(7)]).p90, ms(7));
        let z = summarize(&mut []);
        assert_eq!((z.min, z.p50, z.p90), (ms(0), ms(0), ms(0)));
    }

    #[test]
    fn sample_batched_returns_requested_count() {
        let s = sample_batched(8, || std::hint::black_box(1 + 1));
        assert_eq!(s.len(), 8);
    }
}
