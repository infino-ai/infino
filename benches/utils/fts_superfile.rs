//! Infino-only FTS bench for the superfile layer:
//!
//!   ingest timing (single-thread + rayon-sharded multi-thread)
//! + 7-query search timing
//! + 3 per-algorithm (WAND+BMW vs MaxScore+BMM) probes
//! + correctness gates (BMW-vs-brute-force, df=1 + common-term ordering)
//!
//! Pinned to 1M-doc Zipfian (200 tokens/doc, 10K vocab). The
//! single-superfile shape is rarely much larger in production —
//! `benches/fts/supertable.rs` covers the 10M+ supertable scale.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench fts                            # all FTS benches
//! cargo bench --bench fts -- superfile_fts_build     # only superfile ingest
//! cargo bench --bench fts -- superfile_fts_search    # only superfile search
//! cargo bench --bench fts -- _build                  # ingest across superfile + supertable
//! ```
//!
//! Correctness phase runs unconditionally on every invocation
//! (criterion filters skip timing, not setup), so a filter to
//! `superfile_fts_search` still validates the BMW oracle before
//! timing kicks in.

use crate::{corpus, markdown, rss};
use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group};
use infino::superfile::fts::builder::FtsBuilder;
use infino::superfile::fts::reader::{BoolMode, FtsReader, OrAlgo};
use infino::test_helpers::default_tokenizer;
use rayon::prelude::*;
use std::hint::black_box;
use std::sync::OnceLock;

// ─── Constants ────────────────────────────────────────────────────────

const FTS_COLUMNS_JSON: &str = r#"[{"name":"title","tokenizer":"ascii_lower"}]"#;

/// Doc count for every FTS-superfile bench. Pinned to 1M.
const N_DOCS: usize = 1_000_000;

// ─── Fixtures ────────────────────────────────────────────────────────

static DOCS: OnceLock<Vec<String>> = OnceLock::new();
static INFINO_BLOB: OnceLock<Vec<u8>> = OnceLock::new();

fn docs() -> &'static [String] {
    DOCS.get_or_init(|| corpus::generate_text_corpus(N_DOCS, 1))
        .as_slice()
}

fn infino_reader() -> FtsReader {
    let blob = INFINO_BLOB.get_or_init(|| build_infino_blob_1thread(docs()));
    open_infino(blob)
}

// ─── Builders ─────────────────────────────────────────────────────────

/// Build an FTS blob single-threaded. Used as both correctness fixture
/// and ingest-timing closure body.
fn build_infino_blob_1thread(docs: &[String]) -> Vec<u8> {
    let mut builder = FtsBuilder::new(default_tokenizer());
    builder
        .register_column("title".to_string())
        .expect("register column");
    for (i, text) in docs.iter().enumerate() {
        builder.add_doc(0, i as u32, text).expect("add doc");
    }
    builder.finish()
}

/// Rayon-sharded parallel build. Each shard runs its own `FtsBuilder`
/// and emits a self-contained FTS blob — composes with
/// `SuperfileBuilder::commit()`'s multi-segment output shape.
fn build_infino_blobs_rayon(docs: &[String]) -> Vec<Vec<u8>> {
    let n_shards = rayon::current_num_threads();
    let docs_per_shard = docs.len().div_ceil(n_shards);
    docs.chunks(docs_per_shard)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(build_infino_blob_1thread)
        .collect()
}

fn open_infino(blob: &[u8]) -> FtsReader {
    FtsReader::open(Bytes::from(blob.to_vec()), FTS_COLUMNS_JSON).expect("open FtsReader")
}

// ─── Correctness ──────────────────────────────────────────────────────

/// Self-consistency sanity on an infino blob: a known df=1 token
/// returns exactly one hit at the matching doc_id; a known
/// Zipfian-common term fills top-10 in descending-score order.
fn assert_infino_self_consistent(reader: &FtsReader) {
    let hits = reader
        .search("title", &["doc0500000"], 10, BoolMode::Or)
        .expect("search df=1");
    assert_eq!(hits.len(), 1, "df=1 term should return exactly one hit");
    assert_eq!(hits[0].0, 500_000, "doc0500000 should match doc_id 500000");

    let hits = reader
        .search("title", &["term00001"], 10, BoolMode::Or)
        .expect("search common");
    assert_eq!(hits.len(), 10, "common term should fill top-10");
    for w in hits.windows(2) {
        assert!(
            w[0].1 >= w[1].1,
            "results must be sorted by score desc; got {} then {}",
            w[0].1,
            w[1].1
        );
    }
}

/// BMW correctness oracle: for each query, compare BMW's top-10 against
/// an effectively-brute-force ranking from the same engine. Catches BMW
/// skip bugs, BMM partition bugs, and posting-decode bugs that affect
/// ranking — without needing an external oracle.
///
/// How it works: `search(... k=usize::MAX, BoolMode::Or)` makes BMW's
/// pruning never fire (heap never threshold-tightens because k > df),
/// so the result is the brute-force BM25 ranking. We sort + truncate to
/// top-10 and compare position-by-position.
///
/// Comparing scores not doc_ids: at score ties (common at the top-K
/// boundary on Zipfian corpora), BMW's heap keeps the first-arrived doc
/// while brute-force sort breaks ties by smallest doc_id. Same correct
/// result, different choice from the tied set. Comparing scores
/// sidesteps that.
fn assert_bmw_matches_brute_force(reader: &FtsReader) -> usize {
    let battery: &[(&str, &[&str])] = &[
        ("single_rare", &["term09999"]),
        ("single_common", &["term00001"]),
        ("two_term_or", &["term00001", "term00050"]),
        ("three_wide", &["term00001", "term00050", "term00100"]),
        ("three_similar", &["term00050", "term00051", "term00052"]),
        (
            "five_term",
            &[
                "term00050",
                "term00051",
                "term00052",
                "term00053",
                "term00054",
            ],
        ),
    ];
    const SCORE_EPSILON: f32 = 1e-4;

    for (label, terms) in battery {
        let bmw_top10: Vec<(u32, f32)> = reader
            .search("title", terms, 10, BoolMode::Or)
            .expect("bmw search");
        let mut brute_full = reader
            .search("title", terms, usize::MAX, BoolMode::Or)
            .expect("brute-force search");
        brute_full.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let brute_top10: Vec<(u32, f32)> = brute_full.into_iter().take(10).collect();

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

// ─── Bench helpers ────────────────────────────────────────────────────

fn bench_infino(
    c: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    name: &str,
    r: &FtsReader,
    terms: &'static [&'static str],
    mode: BoolMode,
) {
    c.bench_function(format!("{name}_infino_top10"), |b| {
        b.iter(|| {
            let hits = r
                .search(black_box("title"), black_box(terms), black_box(10), mode)
                .expect("infino search");
            black_box(hits)
        });
    });
}

fn bench_per_algo_probe(
    c: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    name: &str,
    r: &FtsReader,
    terms: &'static [&'static str],
) {
    c.bench_function(format!("{name}_wand_top10"), |b| {
        b.iter(|| {
            let hits = r
                .search_with_algo_for_bench(
                    black_box("title"),
                    black_box(terms),
                    black_box(10),
                    OrAlgo::WandBmw,
                )
                .expect("WAND+BMW search");
            black_box(hits)
        });
    });
    c.bench_function(format!("{name}_bmm_top10"), |b| {
        b.iter(|| {
            let hits = r
                .search_with_algo_for_bench(
                    black_box("title"),
                    black_box(terms),
                    black_box(10),
                    OrAlgo::Bmm,
                )
                .expect("MaxScore+BMM search");
            black_box(hits)
        });
    });
}

// ─── Bench entry ──────────────────────────────────────────────────────

fn bench(c: &mut Criterion) {
    // ---- Correctness phase (runs regardless of criterion filter) ---
    eprintln!("[fts/superfile] correctness: building infino ({N_DOCS} docs)...");
    let r = infino_reader();
    assert_infino_self_consistent(&r);
    let n_bmw = assert_bmw_matches_brute_force(&r);
    eprintln!(
        "[fts/superfile] correctness OK: infino self-consistent + {n_bmw} queries BMW==brute-force"
    );

    // ---- Ingest sub-bench (group: superfile_fts_build) -------------
    {
        let n = N_DOCS;
        let docs_for_ingest = docs();
        let mut g = c.benchmark_group("superfile_fts_build");
        g.sample_size(10);
        g.throughput(Throughput::Elements(n as u64));

        let one_thread_id = format!("infino_1thread_{n}docs");
        let rayon_id = format!("infino_rayon_default_threads_{n}docs");
        let rss_sample = rss::PeakSampler::start_default();
        g.bench_function(one_thread_id.clone(), |b| {
            b.iter_with_large_drop(|| build_infino_blob_1thread(black_box(docs_for_ingest)));
        });
        let peak = rss_sample.stop();
        let _ = rss::write_peak_rss(group_name::SUPERFILE_FTS_BUILD, &one_thread_id, peak);

        let rss_sample = rss::PeakSampler::start_default();
        g.bench_function(rayon_id.clone(), |b| {
            b.iter_with_large_drop(|| build_infino_blobs_rayon(black_box(docs_for_ingest)));
        });
        let peak = rss_sample.stop();
        let _ = rss::write_peak_rss(group_name::SUPERFILE_FTS_BUILD, &rayon_id, peak);

        g.finish();

        emit_ingest_markdown();
    }

    // ---- Search sub-bench (group: superfile_fts_search) ------------
    {
        let mut g = c.benchmark_group("superfile_fts_search");
        let rss_sample = rss::PeakSampler::start_default();

        bench_infino(&mut g, "single_rare", &r, &["term09999"], BoolMode::Or);
        bench_infino(&mut g, "single_df1", &r, &["doc0500000"], BoolMode::Or);
        bench_infino(&mut g, "single_common", &r, &["term00001"], BoolMode::Or);
        bench_infino(
            &mut g,
            "two_term_or",
            &r,
            &["term00001", "term00050"],
            BoolMode::Or,
        );
        bench_infino(
            &mut g,
            "three_wide",
            &r,
            &["term00001", "term00050", "term00100"],
            BoolMode::Or,
        );
        bench_infino(
            &mut g,
            "three_similar",
            &r,
            &["term00050", "term00051", "term00052"],
            BoolMode::Or,
        );
        bench_infino(
            &mut g,
            "five_term",
            &r,
            &[
                "term00050",
                "term00051",
                "term00052",
                "term00053",
                "term00054",
            ],
            BoolMode::Or,
        );
        bench_infino(
            &mut g,
            "two_term_and",
            &r,
            &["term00001", "term00050"],
            BoolMode::And,
        );
        bench_infino(
            &mut g,
            "three_wide_and",
            &r,
            &["term00001", "term00050", "term00100"],
            BoolMode::And,
        );
        bench_infino(
            &mut g,
            "three_similar_and",
            &r,
            &["term00050", "term00051", "term00052"],
            BoolMode::And,
        );
        bench_infino(
            &mut g,
            "five_term_and",
            &r,
            &[
                "term00050",
                "term00051",
                "term00052",
                "term00053",
                "term00054",
            ],
            BoolMode::And,
        );

        // Per-algo probes
        bench_per_algo_probe(
            &mut g,
            "wide_3",
            &r,
            &["term00001", "term00050", "term00100"],
        );
        bench_per_algo_probe(
            &mut g,
            "similar_3",
            &r,
            &["term00050", "term00051", "term00052"],
        );
        bench_per_algo_probe(
            &mut g,
            "similar_5",
            &r,
            &[
                "term00050",
                "term00051",
                "term00052",
                "term00053",
                "term00054",
            ],
        );

        g.finish();
        let peak = rss_sample.stop();
        // Single sampler covers every search closure in this group;
        // record the same peak against each criterion id so the
        // markdown lookup matches the bench id verbatim.
        let search_ids = [
            "single_rare_infino_top10",
            "single_df1_infino_top10",
            "single_common_infino_top10",
            "two_term_or_infino_top10",
            "three_wide_infino_top10",
            "three_similar_infino_top10",
            "five_term_infino_top10",
            "two_term_and_infino_top10",
            "three_wide_and_infino_top10",
            "three_similar_and_infino_top10",
            "five_term_and_infino_top10",
            "wide_3_wand_top10",
            "wide_3_bmm_top10",
            "similar_3_wand_top10",
            "similar_3_bmm_top10",
            "similar_5_wand_top10",
            "similar_5_bmm_top10",
        ];
        for bid in search_ids {
            let _ = rss::write_peak_rss(group_name::SUPERFILE_FTS_SEARCH, bid, peak);
        }

        emit_search_markdown();
    }
}

// ─── Markdown summary emitters ────────────────────────────────────────

mod group_name {
    pub const SUPERFILE_FTS_BUILD: &str = "superfile_fts_build";
    pub const SUPERFILE_FTS_SEARCH: &str = "superfile_fts_search";
}

fn emit_ingest_markdown() {
    use markdown::{MarkdownSection, fmt_throughput, fmt_time, read_mean_ns};

    let mut body = String::new();
    body.push_str(&format!(
        "### Superfile FTS — ingest ({N_DOCS} docs, Zipfian, 200 tokens/doc, 10K vocab)\n\n"
    ));
    body.push_str(
        "| Engine                       | Time       | Throughput | Peak RSS  | Peak RSS Δ |\n",
    );
    body.push_str(
        "|------------------------------|------------|------------|-----------|------------|\n",
    );

    let group = group_name::SUPERFILE_FTS_BUILD;
    let one_thread_id = format!("infino_1thread_{N_DOCS}docs");
    let rayon_id = format!("infino_rayon_default_threads_{N_DOCS}docs");
    let one_thread = read_mean_ns(group, &one_thread_id);
    let rayon = read_mean_ns(group, &rayon_id);
    let one_thread_rss = rss::read_peak_rss_bytes(group, &one_thread_id);
    let rayon_rss = rss::read_peak_rss_bytes(group, &rayon_id);

    let row = |label: &str, bench_id: &str, ns: Option<f64>, peak: Option<u64>| -> String {
        let time = ns.map(fmt_time).unwrap_or_else(|| "—".into());
        let thrpt = ns
            .map(|n| fmt_throughput((N_DOCS as f64) / (n / 1e9)))
            .unwrap_or_else(|| "—".into());
        let rss_cell = peak.map(rss::fmt_bytes).unwrap_or_else(|| "—".into());
        let rss_delta = rss::fmt_peak_rss_delta(group, bench_id);
        format!("| {label:28} | {time:10} | {thrpt:10} | {rss_cell:9} | {rss_delta:10} |\n")
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
        anchor_id: "bench/fts/superfile/ingest".into(),
        body,
    });
}

fn emit_search_markdown() {
    use markdown::{MarkdownSection, fmt_time, read_mean_ns};

    let mut body = String::new();
    body.push_str(&format!("### Superfile FTS — search ({N_DOCS} docs)\n\n"));
    body.push_str("| Query          | infino     | Peak RSS  | Peak RSS Δ |\n");
    body.push_str("|----------------|------------|-----------|------------|\n");

    let group = group_name::SUPERFILE_FTS_SEARCH;
    let queries_or = [
        "single_rare",
        "single_df1",
        "single_common",
        "two_term_or",
        "three_wide",
        "three_similar",
        "five_term",
    ];
    let queries_and = [
        "two_term_and",
        "three_wide_and",
        "three_similar_and",
        "five_term_and",
    ];

    body.push_str("**OR queries:**\n\n");
    for q in queries_or {
        let bid = format!("{q}_infino_top10");
        let inf = read_mean_ns(group, &bid);
        let inf_s = inf.map(fmt_time).unwrap_or_else(|| "—".into());
        let rss_cell = rss::read_peak_rss_bytes(group, &bid)
            .map(rss::fmt_bytes)
            .unwrap_or_else(|| "—".into());
        let rss_delta = rss::fmt_peak_rss_delta(group, &bid);
        body.push_str(&format!(
            "| {q:14} | {inf_s:10} | {rss_cell:9} | {rss_delta:10} |\n"
        ));
    }

    body.push_str("\n**AND queries:**\n\n");
    for q in queries_and {
        let bid = format!("{q}_infino_top10");
        let inf = read_mean_ns(group, &bid);
        let inf_s = inf.map(fmt_time).unwrap_or_else(|| "—".into());
        let rss_cell = rss::read_peak_rss_bytes(group, &bid)
            .map(rss::fmt_bytes)
            .unwrap_or_else(|| "—".into());
        let rss_delta = rss::fmt_peak_rss_delta(group, &bid);
        body.push_str(&format!(
            "| {q:14} | {inf_s:10} | {rss_cell:9} | {rss_delta:10} |\n"
        ));
    }

    body.push('\n');
    body.push_str("**Per-algorithm probes** (WAND+BMW vs MaxScore+BMM):\n\n");
    body.push_str("| Shape         | WAND+BMW   | MaxScore+BMM |\n");
    body.push_str("|---------------|------------|--------------|\n");
    for shape in ["wide_3", "similar_3", "similar_5"] {
        let wand = read_mean_ns(group, &format!("{shape}_wand_top10"));
        let bmm = read_mean_ns(group, &format!("{shape}_bmm_top10"));
        let wand_s = wand.map(fmt_time).unwrap_or_else(|| "—".into());
        let bmm_s = bmm.map(fmt_time).unwrap_or_else(|| "—".into());
        body.push_str(&format!("| {shape:13} | {wand_s:10} | {bmm_s:12} |\n"));
    }

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/fts/superfile/search".into(),
        body,
    });
}

criterion_group!(benches, bench);
