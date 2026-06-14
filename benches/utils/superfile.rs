// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Superfile-layer benchmark runners grouped by modality.

use crate::cost;
use crate::report::{Better, Cell, Report, metric, text};
use crate::rss;

/// Shared headers for the single-superfile ingest tables (fts / vector /
/// sql), so the three modalities can't drift. `Corpus` is the raw input
/// payload fed to the build; `Stored` is the built superfile's size on
/// disk (full Parquet data pages + embedded BM25/vector indexes) and its
/// share of the corpus.
fn ingest_headers() -> Vec<String> {
    [
        "Build",
        "Time",
        "Throughput",
        "Bandwidth",
        "Corpus",
        "Stored",
        "Peak RSS",
        "Median RSS",
        "P90 RSS",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// The `(Corpus, Stored)` cell pair shared by every single-superfile
/// ingest row; `Stored` carries its percentage of `Corpus` inline.
fn corpus_stored_cells(corpus_bytes: u64, stored_bytes: u64) -> [Cell; 2] {
    let stored_pct = if corpus_bytes > 0 {
        100.0 * stored_bytes as f64 / corpus_bytes as f64
    } else {
        0.0
    };
    [
        text(rss::fmt_bytes(corpus_bytes)),
        metric(
            stored_bytes as f64,
            format!("{} ({stored_pct:.0}%)", rss::fmt_bytes(stored_bytes)),
            Better::Lower,
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
fn emit_cost_warm(
    report: &mut Report,
    anchor: &str,
    title: String,
    ingest_wall_s: f64,
    writers: u32,
    stored_bytes: u64,
    corpus_bytes: u64,
    n_docs: usize,
    warm: &[(String, f64)],
) {
    if warm.is_empty() {
        return;
    }
    let resident = rss::current_anon_rss_bytes().unwrap_or(0);
    cost::emit(
        report,
        anchor,
        title,
        &cost::CellCost {
            ingest_wall_s,
            writers,
            put_count: 1,
            stored_bytes,
            corpus_bytes,
            n_docs,
            resident_anon_bytes: resident,
            warm,
            cold: None,
            cold_store: None,
            storage_months: None,
            cold_open_amortized: false,
        },
    );
}

pub mod fts {
    // SPDX-License-Identifier: Apache-2.0
    // SPDX-FileCopyrightText: Copyright The Infino Authors

    //! Superfile-layer FTS bench.
    //!
    //! The comparable build + search numbers — the ones the cross-engine
    //! comparison harness also produces — are measured through
    //! the engine-generic harness ([`run_fts::<InfinoFtsEngine>`]), so
    //! infino's own headline numbers and its head-to-head numbers come from
    //! one measurement path, not two.
    //!
    //! Layered on top are the infino-only extras that have no cross-engine
    //! analogue and stay measured directly:
    //!
    //!   - correctness oracle (BMW top-k == brute-force; df=1 + common-term
    //!     ordering),
    //!   - per-algorithm probe (WAND+BMW vs MaxScore+BMM),
    //!   - rayon-sharded parallel build (single-engine ingest-parallelism),
    //!   - cold tier (the same `.parquet` on object storage, read through
    //!     the production `DiskCacheStore` cold path).
    //!
    //! Every phase uses the production path: [`SuperfileBuilder`] → unified
    //! `.parquet` → [`SuperfileReader`].
    //!
    //! Pinned to 1M-doc Zipfian (200 tokens/doc, 10K vocab). The
    //! single-superfile shape is rarely much larger in production — the
    //! supertable bench covers the 10M+ scale.
    //!
    //! ## Invocation
    //!
    //! ```text
    //! cargo bench -- superfile fts                 # build + search
    //! cargo bench -- superfile fts build           # ingest only
    //! cargo bench -- superfile fts search          # search only
    //! INFINO_BENCH_UPDATE_README=1 cargo bench -- superfile fts
    //! ```

    use std::collections::HashMap;
    use std::hint::black_box;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use bytes::Bytes;
    use infino::superfile::SuperfileReader;
    use infino::superfile::fts::reader::{BoolMode as InfinoBoolMode, OrAlgo};

    use crate::corpus::{self, MmapTextCorpus, block_on_inmem};
    use crate::cost;
    use crate::executors::ColdTiming;
    use crate::executors::fts as exec_fts;
    use crate::executors::fts::{FTS_BATTERY, FtsRead};
    use crate::harness::{EngineFtsResult, InfinoFtsEngine, InfinoFtsIndex, run_fts_with_index};
    use crate::markdown::{fmt_bandwidth, fmt_count, fmt_throughput, fmt_time};
    use crate::report::{Better, Block, Cell, Report, Section, metric, text};
    use crate::rss::{self, RssStats};
    use crate::supertable::Phases;
    use crate::tiers;

    // ─── Constants ────────────────────────────────────────────────────────

    // Document count is the malleable superfile-test parameter
    // (`corpus::superfile_docs()`, default 1M, env-overridable). Captured
    // once per run into a local `n_docs`.
    pub const FTS_COLUMN: &str = "title";

    /// Top-k for every search.
    pub const K: usize = 10;
    /// Timed warm-search repetitions per query (after one warmup). `run_fts`
    /// reports the p50 over these.
    pub const WARM_ITERS: usize = 50;
    /// Cold-tier repetitions per query — each pays a fresh cache + full S3
    /// cold open, so this is deliberately small.
    const COLD_ITERS: usize = 10;
    /// Nanoseconds per second, for throughput / bandwidth markdown.
    const NS_PER_SEC: f64 = 1e9;

    // ─── Query battery (shared by warm search, cold tier, recall id grading) ─

    // FTS query battery + OR/AND name lists live in `crate::executors::fts`.

    /// Negation (`-term`) queries, timed through the string `bm25_hits_async`
    /// path — the shared battery is pretokenized and can't carry the
    /// sigil. Mid-frequency positives so scores differentiate; the
    /// negated term is common (long excluded list) or rare.
    const NEGATION_QUERIES: &[(&str, &str, InfinoBoolMode)] = &[
        (
            "mid_pos_common_neg",
            "term00050 -term00005",
            InfinoBoolMode::Or,
        ),
        (
            "mid_pos_rare_neg",
            "term00050 -term09000",
            InfinoBoolMode::Or,
        ),
        (
            "two_mid_or_common_neg",
            "term00050 term00100 -term00005",
            InfinoBoolMode::Or,
        ),
        (
            "two_mid_and_common_neg",
            "term00050 term00100 -term00005",
            InfinoBoolMode::And,
        ),
    ];

    /// Per-algorithm probe shapes (OR-only; WAND+BMW vs MaxScore+BMM). This
    /// is an infino-internal hook with no cross-engine analogue.
    const PROBE_SHAPES: &[(&str, &[&str])] = &[
        ("wide_3_or", &["term00001", "term00050", "term00100"]),
        ("similar_3_or", &["term00050", "term00051", "term00052"]),
        (
            "similar_5_or",
            &[
                "term00050",
                "term00051",
                "term00052",
                "term00053",
                "term00054",
            ],
        ),
        (
            "similar_10_or",
            &[
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
        ),
    ];

    // ─── Correctness (infino-only oracle) ─────────────────────────────────

    fn assert_superfile_self_consistent(reader: &SuperfileReader, n_docs: usize) {
        let probe_doc_id = n_docs / 2;
        let probe_token = format!("doc{probe_doc_id:07}");
        let hits =
            block_on_inmem(reader.bm25_hits_async(FTS_COLUMN, &probe_token, K, InfinoBoolMode::Or))
                .expect("search df=1");
        assert_eq!(hits.len(), 1, "df=1 term should return exactly one hit");
        assert_eq!(
            hits[0].0 as usize, probe_doc_id,
            "{probe_token} should match doc_id {probe_doc_id}"
        );

        let hits =
            block_on_inmem(reader.bm25_hits_async(FTS_COLUMN, "term00001", K, InfinoBoolMode::Or))
                .expect("search common");
        assert_eq!(hits.len(), K, "common term should fill top-k");
        for w in hits.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "results must be sorted by score desc; got {} then {}",
                w[0].1,
                w[1].1
            );
        }
    }

    fn assert_bmw_matches_brute_force(reader: &SuperfileReader) -> usize {
        let battery: &[(&str, &[&str])] = &[
            ("single_rare", &["term09999"]),
            ("single_common", &["term00001"]),
            ("two_term_or", &["term00001", "term00050"]),
            ("three_wide_or", &["term00001", "term00050", "term00100"]),
            ("three_similar_or", &["term00050", "term00051", "term00052"]),
            (
                "five_term_or",
                &[
                    "term00050",
                    "term00051",
                    "term00052",
                    "term00053",
                    "term00054",
                ],
            ),
            (
                "ten_term_or",
                &[
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
            ),
        ];
        const SCORE_EPSILON: f32 = 1e-4;

        for (label, terms) in battery {
            let bmw_top10: Vec<(u32, f32)> = block_on_inmem(reader.bm25_search_pretokenized(
                FTS_COLUMN,
                terms,
                K,
                InfinoBoolMode::Or,
            ))
            .expect("bmw search");
            let mut brute_full = block_on_inmem(reader.bm25_search_pretokenized(
                FTS_COLUMN,
                terms,
                usize::MAX,
                InfinoBoolMode::Or,
            ))
            .expect("brute-force search");
            brute_full.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            let brute_top10: Vec<(u32, f32)> = brute_full.into_iter().take(K).collect();

            assert_eq!(
                bmw_top10.len(),
                brute_top10.len(),
                "result lengths must match on {label}: BMW {} vs brute {}",
                bmw_top10.len(),
                brute_top10.len()
            );
            for i in 0..bmw_top10.len() {
                let (bmw_doc, bmw_score) = bmw_top10[i];
                let (brute_doc, brute_score) = brute_top10[i];
                let diff = (bmw_score - brute_score).abs();
                if diff > SCORE_EPSILON {
                    let bmw_seq: Vec<f32> = bmw_top10.iter().map(|(_, s)| *s).collect();
                    let brute_seq: Vec<f32> = brute_top10.iter().map(|(_, s)| *s).collect();
                    panic!(
                        "BMW vs brute-force score divergence at position {i} on {label} ({terms:?}):\n  \
                         BMW score = {bmw_score} (doc {bmw_doc})\n  \
                         brute score = {brute_score} (doc {brute_doc})\n  \
                         diff = {diff} > epsilon {SCORE_EPSILON}\n  \
                         BMW scores  : {bmw_seq:?}\n  \
                         brute scores: {brute_seq:?}"
                    );
                }
            }
        }
        battery.len()
    }

    // ─── Manual timing helpers (infino-only extras) ───────────────────────

    /// Nearest-rank p50 of a duration set (sorts in place).
    fn p50(samples: &mut [Duration]) -> Duration {
        if samples.is_empty() {
            return Duration::ZERO;
        }
        samples.sort_unstable();
        samples[(samples.len() - 1) / 2]
    }

    /// One warmup call, then `WARM_ITERS` timed calls of `run`; returns
    /// the p50. Shared scaffold for the manual hot-timing paths.
    fn hot_p50<T>(mut run: impl FnMut() -> T) -> Duration {
        black_box(run());
        let mut samples = Vec::with_capacity(WARM_ITERS);
        for _ in 0..WARM_ITERS {
            let t = Instant::now();
            let out = run();
            samples.push(t.elapsed());
            black_box(out);
        }
        p50(&mut samples)
    }

    /// WAND+BMW vs MaxScore+BMM p50 for one OR shape, via the infino
    /// internal per-algorithm hook.
    fn probe_algo_p50(reader: &SuperfileReader, terms: &[&str], algo: OrAlgo) -> Duration {
        let fts = reader.fts().expect("FTS subsection");
        hot_p50(|| {
            block_on_inmem(fts.search_with_algo_for_bench(FTS_COLUMN, terms, K, algo))
                .expect("probe search")
        })
    }

    /// Hot p50 for one negation query, through the string path (which
    /// parses the `-` sigil).
    fn negation_p50(reader: &SuperfileReader, query: &str, mode: InfinoBoolMode) -> Duration {
        hot_p50(|| {
            block_on_inmem(reader.bm25_hits_async(FTS_COLUMN, query, K, mode))
                .expect("negation search")
        })
    }

    /// Negation correctness gate: each query must return hits, and no
    /// hit's doc may contain a negated term (checked against the corpus
    /// text).
    fn assert_negation_excludes(reader: &SuperfileReader, docs: &[(u64, &str)]) {
        for (name, query, mode) in NEGATION_QUERIES {
            let negated: Vec<&str> = query
                .split_whitespace()
                .filter_map(|r| r.strip_prefix('-'))
                .collect();
            let hits = block_on_inmem(reader.bm25_hits_async(FTS_COLUMN, query, K, *mode))
                .expect("negation oracle search");
            assert!(!hits.is_empty(), "{name}: no hits");
            for (doc_id, _) in &hits {
                let text = docs[*doc_id as usize].1;
                for neg in &negated {
                    assert!(
                        !text.split_whitespace().any(|w| w == *neg),
                        "{name}: doc {doc_id} contains negated term {neg:?}"
                    );
                }
            }
        }
    }

    // ─── Entry point ──────────────────────────────────────────────────────

    /// Bench entry point. Invoked by `benches/fts/main.rs`.
    pub fn run(phases: Phases) {
        let n_docs = corpus::superfile_docs();
        eprintln!(
            "[superfile_fts] starting {} docs (build={}, warm={}, cold={})",
            fmt_count(n_docs),
            phases.build,
            phases.warm,
            phases.cold,
        );
        let (corpus, result, index) = build_warm_artifact(n_docs, phases);

        // Run-to-run deltas for every metric below, vs the previous run.
        let mut report = Report::load("superfile_fts");

        if phases.build {
            emit_build(
                &mut report,
                n_docs,
                &corpus,
                &result,
                index.bytes().len() as u64,
            );
        }

        if phases.warm || phases.cold {
            assert_correct(&index, n_docs);
            exec_fts::assert_correct(index.reader(), FTS_COLUMN, n_docs, "superfile_fts");
            let warm = phases.warm.then(|| {
                exec_fts::measure_warm(
                    index.reader(),
                    FTS_BATTERY,
                    FTS_COLUMN,
                    K,
                    WARM_ITERS,
                    "superfile_fts",
                )
            });
            let probes = phases.warm.then(|| measure_warm_probes(&index));
            let negations = phases.warm.then(|| {
                let docs = corpus.rows();
                assert_negation_excludes(index.reader(), &docs);
                eprintln!(
                    "[superfile_fts] negation battery: {} queries × {WARM_ITERS} timed iters...",
                    NEGATION_QUERIES.len(),
                );
                NEGATION_QUERIES
                    .iter()
                    .map(|(name, query, mode)| (*name, negation_p50(index.reader(), query, *mode)))
                    .collect::<Vec<(&'static str, Duration)>>()
            });
            if phases.warm || probes.is_some() {
                exec_fts::emit_search(
                    &mut report,
                    "bench/fts/superfile/search",
                    format!(
                        "Superfile FTS — search, single-superfile / in-memory ({} docs)",
                        fmt_count(n_docs)
                    ),
                    "Warm = `SuperfileReader::open` in memory (per-query p50); cold = same `.parquet` on \
                     object storage via `DiskCacheStore::reader` -> `bm25_search` (production cold path). \
                     Δ is vs the previous run.",
                    warm.as_deref(),
                    None,
                    probes.as_deref(),
                );
            }
            if let Some(rows) = negations {
                report.emit(&Section {
                    anchor: "bench/fts/superfile/negation".into(),
                    title: format!(
                        "Superfile FTS — negation (`-term`), warm ({} docs)",
                        fmt_count(n_docs)
                    ),
                    note: "Through the string `bm25_hits_async` path (parses the `-` sigil); \
                           a correctness gate (no hit contains a negated term) runs before \
                           timing. Δ is vs the previous run."
                        .into(),
                    blocks: vec![Block {
                        subtitle: "Negation queries".into(),
                        headers: vec!["Query".into(), "warm".into()],
                        rows: rows
                            .iter()
                            .map(|(name, d)| {
                                let ns = d.as_secs_f64() * 1e9;
                                vec![text(*name), metric(ns, fmt_time(ns), Better::Lower)]
                            })
                            .collect(),
                    }],
                });
            }
            if phases.warm
                && let Some(ref warm_stats) = warm
            {
                let b = result
                    .builds
                    .last()
                    .expect("harness records at least one build row");
                super::emit_cost_warm(
                    &mut report,
                    "bench/fts/superfile/cost",
                    format!("Superfile FTS — cost model ({} docs)", fmt_count(n_docs)),
                    b.phase.wall.as_secs_f64(),
                    b.writers as u32,
                    index.bytes().len() as u64,
                    corpus.total_bytes(),
                    n_docs,
                    &cost::warm_from_fts(warm_stats),
                );
            }
            if phases.cold {
                let cold = measure_cold_queries(&index);
                exec_fts::emit_search(
                    &mut report,
                    "bench/fts/superfile/cold",
                    format!(
                        "Superfile FTS — cold search, object-store ({} docs)",
                        fmt_count(n_docs)
                    ),
                    "Cold = same `.parquet` committed to object storage, read through \
                     `DiskCacheStore::reader` with a fresh cache per iteration. Δ is vs the previous run.",
                    None,
                    Some(&cold),
                    None,
                );
            }
        }

        report.save();
    }

    /// Build the canonical one-writer superfile and run the warm query
    /// battery through the shared FTS driver. The returned index is the
    /// exact measured artifact used by correctness and cold reads.
    fn build_warm_artifact(
        n_docs: usize,
        phases: Phases,
    ) -> (MmapTextCorpus, EngineFtsResult, InfinoFtsIndex) {
        eprintln!(
            "[superfile_fts] generating {}-doc Zipfian corpus...",
            fmt_count(n_docs)
        );
        let corpus = MmapTextCorpus::generate(n_docs, 1);
        let docs = corpus.rows();

        let run_warm_search = phases.warm;
        if phases.build {
            eprintln!(
                "[superfile_fts] building 1-writer superfile over {} docs...",
                fmt_count(n_docs)
            );
        }
        if run_warm_search {
            eprintln!(
                "[superfile_fts] warm search battery: {} queries × {WARM_ITERS} timed iters...",
                FTS_BATTERY.len(),
            );
        }
        let (result, index) = run_fts_with_index::<InfinoFtsEngine>(
            FTS_COLUMN,
            &docs,
            &[], // warm search measured via crate::executors::fts
            K,
            WARM_ITERS,
            corpus::parallel_writers(),
        );
        (corpus, result, index)
    }

    fn assert_correct(index: &InfinoFtsIndex, n_docs: usize) {
        eprintln!(
            "[superfile_fts] correctness check: self-consistency + BMW==brute-force on measured artifact..."
        );
        let reader = index.reader();
        assert_superfile_self_consistent(reader, n_docs);
        let n_bmw = assert_bmw_matches_brute_force(reader);
        eprintln!(
            "[superfile_fts] correctness OK: self-consistent + {n_bmw} queries BMW==brute-force"
        );
    }

    /// Infino-only warm probes on the measured in-memory artifact.
    fn measure_warm_probes(index: &InfinoFtsIndex) -> Vec<(&'static str, Duration, Duration)> {
        eprintln!(
            "[superfile_fts] per-algorithm probes: {} OR shapes × {WARM_ITERS} iters (WAND+BMW vs MaxScore+BMM)...",
            PROBE_SHAPES.len(),
        );
        let reader = index.reader();
        let mut probes: Vec<(&'static str, Duration, Duration)> = Vec::new();
        for (shape, terms) in PROBE_SHAPES {
            eprintln!("[superfile_fts] probe: {shape}...");
            let wand = probe_algo_p50(reader, terms, OrAlgo::WandBmw);
            let bmm = probe_algo_p50(reader, terms, OrAlgo::Bmm);
            probes.push((shape, wand, bmm));
        }
        probes
    }

    /// Cold tier: commit the same bytes to object storage, then read each
    /// query through the production cold path.
    fn measure_cold_queries(index: &InfinoFtsIndex) -> HashMap<&'static str, ColdTiming> {
        eprintln!(
            "[superfile_fts] uploading measured 1-writer artifact to object storage for cold tier..."
        );
        let committed = tiers::block_on(tiers::commit_superfile(&Bytes::copy_from_slice(
            index.bytes(),
        )));
        eprintln!(
            "[superfile_fts] cold search: {} queries × {COLD_ITERS} fresh-cache iters...",
            FTS_BATTERY.len(),
        );
        let storage = Arc::clone(&committed.storage);
        let uri = committed.uri;
        exec_fts::measure_cold(
            || SuperfileColdGuard::open(Arc::clone(&storage), uri, committed.object_size),
            FTS_BATTERY,
            FTS_COLUMN,
            K,
            COLD_ITERS,
            "superfile_fts",
        )
    }

    /// Cold-tier guard: a fresh disk cache per open. The constructor
    /// performs the cold reader open (footer + section admit through
    /// the production `DiskCacheStore` path), so the timed `bm25_rows`
    /// pays only the cold search — open and search report separately.
    struct SuperfileColdGuard {
        _cache_dir: tempfile::TempDir,
        reader: Arc<infino::superfile::SuperfileReader>,
    }

    impl SuperfileColdGuard {
        fn open(
            storage: Arc<dyn infino::supertable::storage::StorageProvider>,
            uri: infino::supertable::manifest::SuperfileUri,
            known_size: u64,
        ) -> Self {
            let (cache_dir, reader) = tiers::open_superfile_cold_reader(storage, &uri, known_size);
            Self {
                _cache_dir: cache_dir,
                reader,
            }
        }
    }

    impl FtsRead for SuperfileColdGuard {
        fn bm25_rows(&self, column: &str, query: &str, k: usize, mode: InfinoBoolMode) -> usize {
            tiers::block_on(async {
                self.reader
                    .bm25_hits_async(column, query, k, mode)
                    .await
                    .expect("cold bm25")
                    .len()
            })
        }

        fn bm25_rows_fetched(
            &self,
            column: &str,
            query: &str,
            k: usize,
            mode: InfinoBoolMode,
        ) -> usize {
            exec_fts::superfile_rows_fetched(&self.reader, column, query, k, mode)
        }
    }

    // ─── Result rendering (run-to-run deltas via report.rs) ───────────────

    fn ingest_row(
        label: &str,
        n_docs: usize,
        wall: Duration,
        stats: RssStats,
        corpus_bytes: u64,
        stored_bytes: u64,
    ) -> Vec<Cell> {
        let secs = wall.as_secs_f64();
        let ns = secs * NS_PER_SEC;
        let thr = n_docs as f64 / secs;
        let bw = corpus_bytes as f64 / secs;
        let [corpus_cell, stored_cell] = super::corpus_stored_cells(corpus_bytes, stored_bytes);
        vec![
            text(label),
            metric(ns, fmt_time(ns), Better::Lower),
            metric(thr, fmt_throughput(thr), Better::Higher),
            metric(bw, fmt_bandwidth(bw), Better::Higher),
            corpus_cell,
            stored_cell,
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

    fn writer_label(writers: usize) -> String {
        if writers == 1 {
            "1 writer".to_string()
        } else {
            format!("{writers} writers")
        }
    }

    fn emit_build(
        report: &mut Report,
        n_docs: usize,
        corpus: &MmapTextCorpus,
        result: &EngineFtsResult,
        stored_bytes: u64,
    ) {
        // Logical input payload: total corpus text bytes, identical across
        // every writer count (the parallel build shards the same corpus).
        let corpus_bytes = corpus.total_bytes();
        let rows: Vec<Vec<Cell>> = result
            .builds
            .iter()
            .map(|b| {
                ingest_row(
                    &writer_label(b.writers),
                    n_docs,
                    b.phase.wall,
                    b.phase.rss,
                    corpus_bytes,
                    stored_bytes,
                )
            })
            .collect();
        let block = Block {
            subtitle: String::new(),
            headers: super::ingest_headers(),
            rows,
        };
        report.emit(&Section {
            anchor: "bench/fts/superfile/ingest".into(),
            title: format!(
                "Superfile FTS — ingest, single-superfile / in-memory ({} docs, Zipfian, 200 tokens/doc, 10K vocab)",
                fmt_count(n_docs)
            ),
            note: "Build path: `SuperfileBuilder` → unified `.parquet` (same as production supertable \
                   commit), through the engine-generic `run_fts` driver the cross-engine comparison also \
                   uses. Rows are by writer count: `1 writer` is the single-threaded build (and the index \
                   queries run against); `N writers` is the sharded parallel build. Bandwidth is over the \
                   logical input text payload. Δ is vs the previous run."
                .into(),
            blocks: vec![block],
        });
    }

    // `search_row` / `emit_search` now live in `crate::executors::fts::emit_search`.
}

pub mod vector {
    //! Infino-only vector bench for the superfile layer:
    //!
    //!   ingest timing (1M × 384 Gaussian planted clusters, cosine)
    //! + calibrated kNN search at recall targets {0.90, 0.95, 0.99}
    //! + nprobe/rerank sweeps
    //! + correctness gate (`recall@10 ≥ 0.80` at high-recall config)
    //!
    //! Every phase uses the production path: [`SuperfileBuilder`] →
    //! [`SuperfileReader`] → [`SuperfileReader::vector_search`]. Warm
    //! opens the finished `.parquet` in memory; cold commits the same bytes
    //! to object storage and reads through [`DiskCacheStore::reader`].
    //!
    //! Pinned to 1M × 384. Supertable scale (10M × 384, sharded into N
    //! superfiles) lives in `benches/vector/supertable.rs`.
    //!
    //! ## Invocation
    //!
    //! ```text
    //! cargo bench -- superfile vector build              # ingest only
    //! cargo bench -- superfile vector search             # search only
    //! ```

    use std::sync::{Arc, OnceLock};
    use std::time::Duration;

    use crate::corpus::{self, DIM};
    use crate::cost;
    use crate::executors::vector as exec_vec;
    use crate::executors::vector::VectorRead;
    use crate::harness::{
        EngineVectorResult, InfinoVectorEngine, InfinoVectorIndex, VectorMetric, VectorRunConfig,
        run_vector_with_index,
    };
    use crate::markdown::{fmt_bandwidth, fmt_count, fmt_throughput, fmt_time};
    use crate::report::{Better, Block, Cell, Report, Section, metric, text};
    use crate::rss;
    use crate::supertable::Phases;
    use crate::tiers;
    use bytes::Bytes;

    // ─── Constants ────────────────────────────────────────────────────────

    const TOP_K: usize = 10;
    const N_CORRECTNESS_QUERIES: usize = 20;
    const N_CALIBRATION_QUERIES: usize = 100;
    const CALIBRATION_P50_ITERS: usize = 7;

    /// Default options for the user-facing "what does it cost in
    /// production?" baseline reported in the search markdown.
    const DEFAULT_NPROBE: usize = 8;
    const DEFAULT_RERANK_MULT: usize = 20;

    /// Nanoseconds per second, for latency markdown.
    const NS_PER_SEC: f64 = 1e9;
    /// Deterministic rotation seed for the vector corpus fixture.
    const CORPUS_ROT_SEED: u64 = 1;

    const VEC_COLUMN: &str = "v";

    fn n_docs() -> usize {
        corpus::superfile_docs()
    }

    // ─── Fixtures ────────────────────────────────────────────────────────

    static VECTORS: OnceLock<corpus::MmapVectorCorpus> = OnceLock::new();
    static QUERIES_CORRECTNESS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
    static QUERIES_CALIBRATION: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
    static GROUND_TRUTH_CORRECTNESS: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
    static GROUND_TRUTH_CALIBRATION: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
    pub fn vectors() -> &'static [f32] {
        VECTORS
            .get_or_init(|| {
                // Raw corpus fixture only. Build/search still exercise Infino's
                // normal vector builder/reader paths; the mmap avoids pinning the
                // synthetic source corpus as heap RAM.
                let n = n_docs();
                corpus::MmapVectorCorpus::generate(n, corpus::n_cent(n), CORPUS_ROT_SEED, true)
            })
            .as_slice()
    }

    pub fn queries_correctness() -> &'static [Vec<f32>] {
        QUERIES_CORRECTNESS.get_or_init(|| {
            corpus::generate_realistic_queries(
                vectors(),
                n_docs(),
                N_CORRECTNESS_QUERIES,
                17,
                true,
                0.05,
            )
        })
    }

    // Calibration fixtures are `pub` so the cross-engine comparison
    // the external comparison harness can run the identical recall-calibrated
    // protocol against the same queries and ground truth.
    pub fn queries_calibration() -> &'static [Vec<f32>] {
        QUERIES_CALIBRATION.get_or_init(|| {
            corpus::generate_realistic_queries(
                vectors(),
                n_docs(),
                N_CALIBRATION_QUERIES,
                99,
                true,
                0.05,
            )
        })
    }

    pub fn ground_truth_correctness() -> &'static [Vec<u32>] {
        GROUND_TRUTH_CORRECTNESS
            .get_or_init(|| corpus::ground_truth(vectors(), n_docs(), queries_correctness(), TOP_K))
    }

    pub fn ground_truth_calibration() -> &'static [Vec<u32>] {
        GROUND_TRUTH_CALIBRATION
            .get_or_init(|| corpus::ground_truth(vectors(), n_docs(), queries_calibration(), TOP_K))
    }

    // ─── Correctness ──────────────────────────────────────────────────────

    // ─── Custom-harness runner ────────────────────────────────────────────

    fn writer_label(writers: usize) -> String {
        if writers == 1 {
            "1 writer".to_string()
        } else {
            format!("{writers} writers")
        }
    }

    fn build_row(
        label: &str,
        n_docs: usize,
        wall: Duration,
        stats: rss::RssStats,
        corpus_bytes: u64,
        stored_bytes: u64,
    ) -> Vec<Cell> {
        let secs = wall.as_secs_f64();
        let ns = secs * NS_PER_SEC;
        let thr = n_docs as f64 / secs;
        let bw = corpus_bytes as f64 / secs;
        let [corpus_cell, stored_cell] = super::corpus_stored_cells(corpus_bytes, stored_bytes);
        vec![
            text(label),
            metric(ns, fmt_time(ns), Better::Lower),
            metric(thr, fmt_throughput(thr), Better::Higher),
            metric(bw, fmt_bandwidth(bw), Better::Higher),
            corpus_cell,
            stored_cell,
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

    pub fn run(phases: Phases) {
        let n_docs = n_docs();
        eprintln!(
            "[superfile_vec] starting {}×{DIM} (build={}, warm={}, cold={})",
            fmt_count(n_docs),
            phases.build,
            phases.warm,
            phases.cold,
        );
        let (build_result, index) = build_warm_artifact(n_docs);

        let build_rows = build_rows(n_docs, &build_result, index.bytes().len() as u64);
        let mut report = Report::load("superfile_vector");
        if phases.build {
            emit_build(&mut report, n_docs, build_rows);
        }

        if phases.warm || phases.cold {
            let committed = phases.cold.then(|| commit_cold_artifact(&index));
            let open_cold = || {
                let committed = committed.as_ref().expect("cold artifact present");
                SuperfileVecColdGuard::open(
                    Arc::clone(&committed.storage),
                    committed.uri,
                    committed.object_size,
                )
            };
            let recall_rows = exec_vec::run_search(
                &mut report,
                index.reader(),
                open_cold,
                VEC_COLUMN,
                n_docs,
                TOP_K,
                DEFAULT_NPROBE,
                DEFAULT_RERANK_MULT,
                queries_correctness(),
                ground_truth_correctness(),
                queries_calibration(),
                ground_truth_calibration(),
                phases.warm,
                phases.cold,
                3,
                false,
                "superfile_vec",
                "bench/vector/superfile/search",
                format!(
                    "Superfile vector — search, single-superfile / in-memory ({} docs × dim={DIM})",
                    fmt_count(n_docs)
                ),
                "Correctness, warm search, and cold upload reuse the measured 1-writer artifact. Recall rows use the lowest-p50 calibrated point meeting each target; `default` is the user-facing option baseline. Δ is vs the previous run.",
            );
            if phases.warm {
                let b = build_result
                    .builds
                    .last()
                    .expect("harness records at least one build row");
                let corpus_bytes = (n_docs * DIM) as u64 * std::mem::size_of::<f32>() as u64;
                super::emit_cost_warm(
                    &mut report,
                    "bench/vector/superfile/cost",
                    format!(
                        "Superfile vector — cost model ({} docs × dim={DIM})",
                        fmt_count(n_docs)
                    ),
                    b.wall.as_secs_f64(),
                    b.writers as u32,
                    index.bytes().len() as u64,
                    corpus_bytes,
                    n_docs,
                    &cost::warm_from_vector(&recall_rows),
                );
            }

            struct SuperfileVecColdGuard {
                _cache_dir: tempfile::TempDir,
                reader: Arc<infino::superfile::SuperfileReader>,
            }
            impl SuperfileVecColdGuard {
                /// The reader open (footer + KV fetch over the object
                /// store) happens HERE so the cold driver bills it to
                /// the "cold open" leg — mirroring the FTS guard. A
                /// constructor that only builds the cache makes the
                /// open column measure an empty struct while the
                /// search column silently absorbs the open cost.
                fn open(
                    storage: Arc<dyn infino::supertable::storage::StorageProvider>,
                    uri: infino::supertable::manifest::SuperfileUri,
                    known_size: u64,
                ) -> Self {
                    let (cache_dir, reader) =
                        tiers::open_superfile_cold_reader(storage, &uri, known_size);
                    Self {
                        _cache_dir: cache_dir,
                        reader,
                    }
                }
            }
            impl VectorRead for SuperfileVecColdGuard {
                fn topk_global(
                    &self,
                    column: &str,
                    query: &[f32],
                    k: usize,
                    nprobe: usize,
                    rerank: usize,
                ) -> Vec<(u32, f32)> {
                    tiers::block_on(async {
                        self.reader
                            .vector_hits_async(
                                column,
                                query,
                                k,
                                exec_vec::search_opts(nprobe, rerank),
                            )
                            .await
                            .expect("cold vector_search")
                    })
                }
            }
        }
        report.save();
    }

    /// Build the canonical one-writer vector superfile and the build-only
    /// parallel row. The returned index is the exact artifact used by
    /// correctness, warm search, and cold upload.
    fn build_warm_artifact(n_docs: usize) -> (EngineVectorResult, InfinoVectorIndex) {
        eprintln!(
            "[superfile_vec] generating {}×{DIM} planted-cluster vector corpus...",
            fmt_count(n_docs)
        );
        let vectors = vectors();

        eprintln!(
            "[superfile_vec] building 1-writer vector superfile over {} docs...",
            fmt_count(n_docs),
        );
        let empty_queries: [crate::harness::VectorQuery<'_>; 0] = [];
        let (build_result, index) = run_vector_with_index::<InfinoVectorEngine>(
            VectorRunConfig {
                column: VEC_COLUMN,
                dim: DIM,
                metric: VectorMetric::Cosine,
                k: TOP_K,
                iters: CALIBRATION_P50_ITERS,
                parallel: corpus::parallel_writers(),
            },
            vectors,
            &empty_queries,
        );
        (build_result, index)
    }

    /// Upload the exact measured one-writer artifact for cold reads.
    fn commit_cold_artifact(index: &InfinoVectorIndex) -> tiers::SuperfileCommitted {
        eprintln!(
            "[superfile_vec] uploading measured 1-writer artifact to object storage for cold tier..."
        );
        tiers::block_on(tiers::commit_superfile(&Bytes::copy_from_slice(
            index.bytes(),
        )))
    }

    fn build_rows(
        n_docs: usize,
        build_result: &EngineVectorResult,
        stored_bytes: u64,
    ) -> Vec<Vec<Cell>> {
        // Logical input payload: the raw f32 embeddings, identical across
        // every writer count (the parallel build shards the same corpus).
        let corpus_bytes = (n_docs * DIM * size_of::<f32>()) as u64;
        let mut rows = Vec::new();
        for b in &build_result.builds {
            rows.push(build_row(
                &writer_label(b.writers),
                n_docs,
                b.wall,
                b.rss,
                corpus_bytes,
                stored_bytes,
            ));
        }
        rows
    }

    fn emit_build(report: &mut Report, n_docs: usize, build_rows: Vec<Vec<Cell>>) {
        report.emit(&Section {
            anchor: "bench/vector/superfile/ingest".into(),
            title: format!(
                "Superfile vector — ingest, single-superfile / in-memory ({} docs × dim={DIM})",
                fmt_count(n_docs)
            ),
            note: "Build path: `SuperfileBuilder` → unified `.parquet`, through `VectorEngine`. Rows are by writer count; `1 writer` is the canonical artifact used by correctness/search/cold upload. Δ is vs the previous run.".into(),
            blocks: vec![Block {
                subtitle: String::new(),
                headers: super::ingest_headers(),
                rows: build_rows,
            }],
        });
    }
}

pub mod sql {
    // SPDX-License-Identifier: Apache-2.0
    // SPDX-FileCopyrightText: Copyright The Infino Authors

    //! SQL bench (infino-only entry point).
    //!
    //! Build + query numbers are measured through the engine-generic SQL
    //! harness (`run_sql::<InfinoSqlEngine>`) — the same path the cross-engine
    //! comparison uses. The canonical 1-writer build produces the queryable
    //! in-memory `Supertable`; correctness and warm queries run against that
    //! exact artifact. A separate `N writers` build row measures parallel
    //! ingest throughput.
    //!
    //! ## Invocation
    //!
    //! ```text
    //! cargo bench -- superfile sql
    //! INFINO_BENCH_SUPERFILE_DOCS=100000 cargo bench -- superfile sql
    //! INFINO_BENCH_UPDATE_README=1 cargo bench -- superfile sql
    //! ```

    use std::sync::Arc;

    use infino::supertable::Supertable;

    use crate::corpus::{self, MmapTextCorpus};
    use crate::cost;
    use crate::executors::sql as exec_sql;
    use crate::executors::sql::SqlRead;
    use crate::harness::{
        EngineSqlResult, InfinoSqlEngine, InfinoSqlIndex, SqlRow, SqlRunConfig,
        build_supertable_with_options, run_sql_with_index, sample_query_csv, scatter_key,
        sql_options,
    };
    use crate::markdown::{fmt_bandwidth, fmt_count, fmt_throughput, fmt_time};
    use crate::report::{Better, Block, Cell, Report, Section, metric, text};
    use crate::rss::{self, RssStats};
    use crate::supertable::Phases;
    use crate::tiers;

    /// Deterministic category labels assigned round-robin by doc id, so the
    /// planted distribution is exactly known for the correctness gate.
    const CATEGORIES: &[&str] = &["rust", "python", "go", "sql"];

    /// Build the planted `(doc_id, title, category, score)` rows borrowing
    /// titles from the shared mmap corpus. `category` cycles through
    /// [`CATEGORIES`]; `score` is `doc_id % 100`.
    pub fn sql_rows<'a>(corpus_rows: &'a [(u64, &'a str)]) -> Vec<SqlRow<'a>> {
        corpus_rows
            .iter()
            .map(|&(doc_id, title)| SqlRow {
                doc_id,
                title,
                category: CATEGORIES[(doc_id as usize) % CATEGORIES.len()],
                score: (doc_id % 100) as i64,
            })
            .collect()
    }

    // ─── Entry point ──────────────────────────────────────────────────────

    pub fn run(phases: Phases) {
        let n_docs = corpus::superfile_docs();
        eprintln!(
            "[superfile_sql] starting {} rows (build={}, warm={}, cold={})",
            fmt_count(n_docs),
            phases.build,
            phases.warm,
            phases.cold,
        );
        let (corpus, query_inputs, result, index) = build_warm_artifact(n_docs, phases);

        let mut report = Report::load("sql");
        if phases.build {
            let stored = stored_bytes(&index);
            emit_build(&mut report, n_docs, &corpus, &result, stored);
        }
        if phases.warm {
            exec_sql::assert_correct(&index, n_docs, "superfile_sql");
            let sets = exec_sql::measure_query_sets(
                &index,
                &query_inputs,
                exec_sql::ITERS,
                "superfile_sql",
            );
            exec_sql::emit_query(
                &mut report,
                "bench/sql/query",
                format!(
                    "Superfile SQL — query, single superfile / in-memory ({} rows)",
                    fmt_count(n_docs)
                ),
                "Warm p50 over `query_sql` against the canonical 1-writer table, all through infino's own path (the DataFusion-only control arms are not run here). Blocks: aggregations & count-filters, FTS-pushdown equality, aggregates over an FTS candidate set, and the search table functions. `Rows` is the result-set size. Δ is vs the previous run.",
                &sets,
            );
            let b = result
                .builds
                .last()
                .expect("harness records at least one build row");
            super::emit_cost_warm(
                &mut report,
                "bench/sql/superfile/cost",
                format!("Superfile SQL — cost model ({} rows)", fmt_count(n_docs)),
                b.wall.as_secs_f64(),
                b.writers as u32,
                stored_bytes(&index),
                corpus.total_bytes(),
                n_docs,
                &cost::warm_from_sql(&sets),
            );
        }
        if phases.cold {
            let corpus_rows = corpus.rows();
            let rows = sql_rows(&corpus_rows);
            let cold = measure_cold_queries(&rows);
            exec_sql::emit_cold(
                &mut report,
                "bench/sql/superfile/cold",
                format!(
                    "Superfile SQL — cold query, object-store ({} rows)",
                    fmt_count(n_docs)
                ),
                "Cold p50 over `reader().query_sql` after reopening the same SQL table shape from object storage with a fresh disk cache per iteration. Δ is vs the previous run.",
                &cold,
            );
        }
        report.save();
    }

    /// Build the canonical one-writer SQL table and run the warm scalar SQL
    /// battery through the shared SQL driver.
    fn build_warm_artifact(
        n_docs: usize,
        phases: Phases,
    ) -> (
        MmapTextCorpus,
        exec_sql::QueryInputs,
        EngineSqlResult,
        InfinoSqlIndex,
    ) {
        eprintln!(
            "[superfile_sql] generating {}-row Zipfian corpus...",
            fmt_count(n_docs)
        );
        let corpus = MmapTextCorpus::generate(n_docs, 1);
        let corpus_rows = corpus.rows();
        let mid = corpus_rows.len() / 2;
        let query_inputs = exec_sql::QueryInputs {
            qv: sample_query_csv(),
            sample_title: corpus_rows[mid].1.replace('\'', "''"),
            sample_key: scatter_key(corpus_rows[mid].0),
        };
        let rows = sql_rows(&corpus_rows);

        if phases.build {
            eprintln!(
                "[superfile_sql] building 1-writer supertable over {} rows...",
                fmt_count(n_docs),
            );
        }
        let (result, index) = run_sql_with_index::<InfinoSqlEngine>(
            SqlRunConfig {
                iters: exec_sql::ITERS,
                parallel: corpus::parallel_writers(),
            },
            &rows,
            &[], // scalar battery measured via crate::executors::sql
        );
        (corpus, query_inputs, result, index)
    }

    struct ColdSqlArtifact {
        fixture: tiers::StorageFixture,
        n_rows: usize,
        total_index_bytes: u64,
    }

    /// Build the SQL bench shape on the superfile object-store fixture so
    /// cold reads exercise `Supertable::open` + `reader().query_sql`.
    ///
    /// The write is a single commit at superfile scale; this keeps the
    /// default `s3s_fs` fixture usable while still writing the same parquet
    /// superfile format that warm SQL reads.
    fn build_cold_artifact(rows: &[SqlRow<'_>]) -> ColdSqlArtifact {
        eprintln!(
            "[superfile_sql] building object-store SQL artifact for cold reads over {} rows...",
            fmt_count(rows.len())
        );
        let fixture = tiers::block_on(tiers::superfile_storage_fixture());
        let (cache_dir, cache) = tiers::fresh_disk_cache(Arc::clone(&fixture.storage));
        let opts = sql_options(rows.len())
            .with_storage(std::sync::Arc::clone(&fixture.storage))
            .with_disk_cache(cache.clone())
            .with_cache_prepopulation(false);
        let table = build_supertable_with_options(rows, opts, rows.len().max(1));
        let reader = table.reader();
        let total_index_bytes: u64 = reader
            .manifest()
            .superfiles
            .iter()
            .filter_map(|entry| entry.subsection_offsets.as_ref())
            .map(|offsets| offsets.total_size)
            .sum();
        drop(reader);
        drop(table);
        drop(cache);
        drop(cache_dir);
        ColdSqlArtifact {
            fixture,
            n_rows: rows.len(),
            total_index_bytes,
        }
    }

    fn open_cold_consumer(artifact: &ColdSqlArtifact) -> (tempfile::TempDir, Supertable) {
        let (cache_dir, cache) = tiers::fresh_supertable_search_cache(
            std::sync::Arc::clone(&artifact.fixture.storage),
            Some(artifact.total_index_bytes),
        );
        let opts = sql_options(artifact.n_rows)
            .with_storage(std::sync::Arc::clone(&artifact.fixture.storage))
            .with_disk_cache(cache);
        (cache_dir, tiers::open_consumer(opts))
    }

    fn measure_cold_queries(
        rows: &[SqlRow<'_>],
    ) -> std::collections::HashMap<&'static str, crate::executors::ColdTiming> {
        const COLD_ITERS: usize = 5;
        let artifact = build_cold_artifact(rows);
        eprintln!(
            "[superfile_sql] cold queries: {} queries × {COLD_ITERS} fresh-cache iters...",
            exec_sql::SQL_BATTERY.len(),
        );
        let cold = exec_sql::measure_cold(
            || {
                let (cache_dir, table) = open_cold_consumer(&artifact);
                crate::executors::open_all_superfiles(&table);
                SqlColdGuard {
                    _cache_dir: cache_dir,
                    table,
                }
            },
            COLD_ITERS,
            "superfile_sql",
        );
        if let Some(cleanup) = &artifact.fixture.cleanup {
            tiers::cleanup_prefix(cleanup);
        }
        cold
    }

    /// Cold-tier guard: holds the fresh cache dir + reopened table so the
    /// shared SQL executor can time one `query_sql` per fresh open.
    struct SqlColdGuard {
        _cache_dir: tempfile::TempDir,
        table: Supertable,
    }
    impl SqlRead for SqlColdGuard {
        fn query_rows(&self, sql: &str) -> usize {
            self.table.query_rows(sql)
        }
        fn query_count(&self, sql: &str) -> i64 {
            self.table.query_count(sql)
        }
    }

    // ─── Result rendering (run-to-run deltas via report.rs) ───────────────

    fn writer_label(writers: usize) -> String {
        if writers == 1 {
            "1 writer".to_string()
        } else {
            format!("{writers} writers")
        }
    }

    fn rss_cells(stats: RssStats) -> Vec<Cell> {
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

    /// Total on-storage footprint of the built superfile(s): full Parquet
    /// (data pages + embedded indexes), summed from the manifest. Same
    /// derivation as the cold artifact's `total_index_bytes`.
    fn stored_bytes(index: &InfinoSqlIndex) -> u64 {
        index
            .table()
            .reader()
            .manifest()
            .superfiles
            .iter()
            .filter_map(|entry| entry.subsection_offsets.as_ref())
            .map(|offsets| offsets.total_size)
            .sum()
    }

    fn emit_build(
        report: &mut Report,
        n_docs: usize,
        corpus: &MmapTextCorpus,
        result: &EngineSqlResult,
        stored_bytes: u64,
    ) {
        let corpus_bytes = corpus.total_bytes();
        let rows: Vec<Vec<Cell>> = result
            .builds
            .iter()
            .map(|b| {
                let secs = b.wall.as_secs_f64();
                let ns = secs * 1e9;
                let thr = n_docs as f64 / secs;
                let bw = corpus_bytes as f64 / secs;
                let [corpus_cell, stored_cell] =
                    super::corpus_stored_cells(corpus_bytes, stored_bytes);
                let mut cells = vec![
                    text(writer_label(b.writers)),
                    metric(ns, fmt_time(ns), Better::Lower),
                    metric(thr, fmt_throughput(thr), Better::Higher),
                    metric(bw, fmt_bandwidth(bw), Better::Higher),
                    corpus_cell,
                    stored_cell,
                ];
                cells.extend(rss_cells(b.rss));
                cells
            })
            .collect();
        report.emit(&Section {
            anchor: "bench/sql/build".into(),
            title: format!(
                "Superfile SQL — ingest, single superfile / in-memory ({} rows: title + category + score)",
                fmt_count(n_docs)
            ),
            note: "Build path: `SupertableWriter::append` + `commit` into an in-memory supertable, through \
                   the engine-generic `run_sql` driver the cross-engine comparison also uses. Rows are by \
                   writer count: `1 writer` is the canonical build queries run against; `N writers` is the \
                   sharded parallel build. Δ is vs the previous run."
                .into(),
            blocks: vec![Block {
                subtitle: String::new(),
                headers: super::ingest_headers(),
                rows,
            }],
        });
    }
}
