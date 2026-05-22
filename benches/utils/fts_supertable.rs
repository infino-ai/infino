//! Infino-only FTS bench for the supertable layer:
//!
//!   ingest timing (10M docs sharded into [`SEGMENTS`] commits)
//! + 7-query search timing (single rare, single common, OR-2,
//!   wide-3, similar-3, OR-5, prefix-10)
//! + self-consistency correctness gate
//!
//! Multi-segment shape: the corpus is sharded into [`SEGMENTS`]
//! commits. Infino's `commit()` row-shards into
//! `min(writer_pool.threads, total_rows)` superfiles — the writer-pool
//! size doubles as the output-cardinality dial (auto = `cpus/2`;
//! override with `INFINO_SUPERTABLE__WRITER_THREADS=N`).
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench fts -- supertable_fts            # both groups
//! cargo bench --bench fts -- supertable_fts_build      # ingest only
//! cargo bench --bench fts -- supertable_fts_search     # search only
//! INFINO_SUPERTABLE__WRITER_THREADS=32 cargo bench --bench fts -- supertable_fts_build
//! ```

use std::hint::black_box;
use std::sync::{Arc, OnceLock};

use crate::{corpus, markdown, rss};
use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, Throughput, criterion_group};
use infino::superfile::builder::FtsConfig;
use infino::superfile::fts::reader::BoolMode;
use infino::superfile::fts::tokenize::Tokenizer;
use infino::supertable::{Supertable, SupertableOptions};
use infino::test_helpers::default_tokenizer;
use rayon::ThreadPool;

// ─── Constants ────────────────────────────────────────────────────────

/// Doc count for every FTS-supertable bench. Pinned to 10M.
const N_DOCS: usize = 10_000_000;

/// Input chunk count. Drives the `append()`-batching shape; output
/// superfile count is governed by writer_pool threads, not this knob.
const SEGMENTS: usize = 4;

const TOP_K: usize = 10;

// ─── Fixtures ────────────────────────────────────────────────────────

static DOCS: OnceLock<Vec<String>> = OnceLock::new();
static INFINO: OnceLock<Supertable> = OnceLock::new();

fn docs() -> &'static [String] {
    DOCS.get_or_init(|| corpus::generate_text_corpus(N_DOCS, 1))
        .as_slice()
}

fn infino_supertable() -> &'static Supertable {
    INFINO.get_or_init(|| build_supertable_infino(docs(), parallel_pool()))
}

// ─── Shared rayon pool ────────────────────────────────────────────────

/// `num_cpus`-sized pool used as infino's reader pool. Sized to the
/// machine so the supertable's per-segment fan-out doesn't bottleneck
/// on threads.
fn parallel_pool() -> Arc<ThreadPool> {
    static POOL: OnceLock<Arc<ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_cpus::get().max(1))
                .thread_name(|i| format!("supertable-fts-bench-par-{i}"))
                .build()
                .expect("parallel pool"),
        )
    })
    .clone()
}

// ─── Builder ──────────────────────────────────────────────────────────

fn schema_id_title() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "title",
        DataType::LargeUtf8,
        false,
    )]))
}

fn supertable_options(reader_pool: Arc<ThreadPool>) -> SupertableOptions {
    let tk: Arc<dyn Tokenizer> = default_tokenizer();
    SupertableOptions::new(
        schema_id_title(),
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![],
        Some(tk),
    )
    .expect("opts")
    .with_reader_pool(reader_pool)
    // Bench raises the commit-threshold sky-high so `append()` doesn't
    // auto-flush mid-build. With a 1 GiB default at ~4.5 GB / chunk,
    // default options would auto-commit after every single append — but
    // the supertable's `commit()` runs per-shard work in parallel only
    // **within** a commit. By buffering all chunks before the explicit
    // final commit below, `commit()` row-shards across all writer-pool
    // threads in one go.
    .with_commit_threshold_size_mb(0)
}

/// Build an FTS-only supertable from `docs`. Append-many-then-commit-
/// once: each chunk is appended to the writer's buffer; a single
/// `commit()` at the end drains and row-shards across the writer pool.
/// Output superfile count is `min(writer_pool.threads, total_rows)`.
fn build_supertable_infino(docs: &[String], reader_pool: Arc<ThreadPool>) -> Supertable {
    let st = Supertable::create(supertable_options(reader_pool));
    let mut w = st.writer().expect("writer");
    let chunk_size = docs.len().div_ceil(SEGMENTS);
    for chunk in docs.chunks(chunk_size) {
        let titles = LargeStringArray::from(chunk.iter().map(String::as_str).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(schema_id_title(), vec![Arc::new(titles)]).expect("batch");
        w.append(&batch).expect("append");
    }
    w.commit().expect("commit");
    drop(w);
    st
}

// ─── Correctness ──────────────────────────────────────────────────────

/// Self-consistency on the built supertable: the corpus's df=1
/// identifier `doc<id:07>` returns exactly one hit; a Zipfian-common
/// term fills top-10 in score-desc order.
fn assert_infino_self_consistent(st: &Supertable) {
    let r = st.reader();
    let probe_doc_id = (N_DOCS / 2) as u32;
    let probe_token = format!("doc{probe_doc_id:07}");
    let hits = r
        .bm25_search("title", &probe_token, 10, BoolMode::Or)
        .expect("bm25");
    assert_eq!(
        hits.len(),
        1,
        "df=1 token {probe_token:?} should return exactly one hit; got {}",
        hits.len()
    );
    assert!(
        hits[0].score > 0.0,
        "df=1 score must be positive; got {}",
        hits[0].score
    );

    let hits = r
        .bm25_search("title", "term00001", 10, BoolMode::Or)
        .expect("bm25");
    assert_eq!(hits.len(), 10, "common term should fill top-10");
    for w in hits.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "results must be sorted by score desc; got {} then {}",
            w[0].score,
            w[1].score
        );
    }
}

// ─── Bench: ingest (group: supertable_fts_build) ──────────────────────

fn bench_ingest(c: &mut Criterion) {
    eprintln!("[supertable_fts_build] correctness: building infino ({N_DOCS} docs)...");
    let infino = build_supertable_infino(docs(), parallel_pool());
    assert_infino_self_consistent(&infino);
    eprintln!("[supertable_fts_build] correctness OK: infino self-consistent");
    drop(infino);

    let mut g = c.benchmark_group("supertable_fts_build");
    g.sample_size(10);
    g.throughput(Throughput::Elements(N_DOCS as u64));

    // Per-group peak VmRSS — covers the auto-writer-pool build of the
    // 10M-doc supertable end-to-end (append buffers + tokenizer
    // workspaces + per-shard FtsBuilder allocations during commit).
    let rss_sample = rss::PeakSampler::start_default();

    g.bench_function("infino_auto_writer_pool", |b| {
        b.iter_with_large_drop(|| build_supertable_infino(black_box(docs()), parallel_pool()));
    });

    g.finish();
    let peak = rss_sample.stop();
    let _ = rss::write_peak_rss(
        group_name::SUPERTABLE_FTS_BUILD,
        "infino_auto_writer_pool",
        peak,
    );

    emit_ingest_markdown();
}

// ─── Bench: search (group: supertable_fts_search) ─────────────────────

fn bench_search(c: &mut Criterion) {
    let st = infino_supertable();
    let pool = parallel_pool();

    eprintln!("[supertable_fts_search] correctness check...");
    assert_infino_self_consistent(st);
    eprintln!(
        "[supertable_fts_search] correctness OK (rayon pool: {} threads)",
        pool.current_num_threads()
    );

    let r = st.reader();

    let mut g = c.benchmark_group("supertable_fts_search");
    g.sample_size(10);

    // Group-level peak VmRSS for FTS-supertable search workload —
    // primarily the resident size of per-superfile FTS index segments
    // pinned through the read.
    let rss_sample = rss::PeakSampler::start_default();

    let queries: &[(&str, &str)] = &[
        ("single_rare", "term09999"),
        ("single_common", "term00001"),
        ("two_term_or", "term00001 term00050"),
        ("three_wide", "term00001 term00050 term00100"),
        ("three_similar", "term00050 term00051 term00052"),
        (
            "five_term",
            "term00050 term00051 term00052 term00053 term00054",
        ),
    ];
    for (name, q) in queries {
        let name = *name;
        let q = *q;
        g.bench_function(format!("{name}_supertable_top10"), |b| {
            b.iter(|| {
                let hits = r
                    .bm25_search(black_box("title"), black_box(q), TOP_K, BoolMode::Or)
                    .expect("bm25");
                black_box(hits)
            });
        });
    }

    g.bench_function("prefix_supertable_top10", |b| {
        b.iter(|| {
            let hits = r
                .bm25_search_prefix(black_box("title"), black_box("term0009"), TOP_K)
                .expect("bm25_prefix");
            black_box(hits)
        });
    });

    g.finish();
    let peak = rss_sample.stop();
    let search_ids = [
        "single_rare_supertable_top10",
        "single_common_supertable_top10",
        "two_term_or_supertable_top10",
        "three_wide_supertable_top10",
        "three_similar_supertable_top10",
        "five_term_supertable_top10",
        "prefix_supertable_top10",
    ];
    for bid in search_ids {
        let _ = rss::write_peak_rss(group_name::SUPERTABLE_FTS_SEARCH, bid, peak);
    }

    emit_search_markdown();
}

// ─── Markdown summary emitters ────────────────────────────────────────

mod group_name {
    pub const SUPERTABLE_FTS_BUILD: &str = "supertable_fts_build";
    pub const SUPERTABLE_FTS_SEARCH: &str = "supertable_fts_search";
}

fn emit_ingest_markdown() {
    use markdown::{MarkdownSection, fmt_throughput, fmt_time, read_mean_ns};

    let group = group_name::SUPERTABLE_FTS_BUILD;
    let bench = "infino_auto_writer_pool";
    let ns = read_mean_ns(group, bench);
    let peak_rss = rss::read_peak_rss_bytes(group, bench);

    let mut body = String::new();
    body.push_str(&format!(
        "### Supertable FTS — ingest ({N_DOCS} docs, Zipfian, 200 tokens/doc, 10K vocab)\n\n"
    ));
    body.push_str(
        "| Engine                  | Time       | Throughput | Peak RSS  | Peak RSS Δ |\n",
    );
    body.push_str(
        "|-------------------------|------------|------------|-----------|------------|\n",
    );
    let time = ns.map(fmt_time).unwrap_or_else(|| "—".into());
    let thrpt = ns
        .map(|n| fmt_throughput((N_DOCS as f64) / (n / 1e9)))
        .unwrap_or_else(|| "—".into());
    let rss_cell = peak_rss.map(rss::fmt_bytes).unwrap_or_else(|| "—".into());
    let rss_delta = rss::fmt_peak_rss_delta(group, bench);
    body.push_str(&format!(
        "| infino_auto_writer_pool | {time:10} | {thrpt:10} | {rss_cell:9} | {rss_delta:10} |\n"
    ));
    body.push_str(
        "\n*Output cardinality: infino emits `min(writer_pool.threads, total_rows)` superfiles \
         per commit (auto = cpus/2). Override with `INFINO_SUPERTABLE__WRITER_THREADS=N` for a \
         specific shard count.*\n",
    );

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/fts/supertable/ingest".into(),
        body,
    });
}

fn emit_search_markdown() {
    use markdown::{MarkdownSection, fmt_time, read_mean_ns};

    let group = group_name::SUPERTABLE_FTS_SEARCH;
    let mut body = String::new();
    body.push_str(&format!("### Supertable FTS — search ({N_DOCS} docs)\n\n"));
    body.push_str("| Query          | infino     | Peak RSS  | Peak RSS Δ |\n");
    body.push_str("|----------------|------------|-----------|------------|\n");
    let queries = [
        "single_rare",
        "single_common",
        "two_term_or",
        "three_wide",
        "three_similar",
        "five_term",
        "prefix",
    ];
    for q in queries {
        let bid = format!("{q}_supertable_top10");
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

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/fts/supertable/search".into(),
        body,
    });
}

criterion_group!(benches, bench_ingest, bench_search);
