//! Infino-only FTS bench for the supertable layer:
//!
//!   ingest timing (10M docs streamed through bounded append chunks)
//! + 7-query search timing (single rare, single common, OR-2,
//!   wide-3, similar-3, OR-5, prefix-10)
//! + self-consistency correctness gate
//!
//! Multi-segment shape: the mmap corpus is materialized into bounded
//! append chunks. Each commit row-shards into
//! `min(writer_pool.threads, chunk_rows)` superfiles — the writer-pool
//! size and append-chunk count together control output cardinality.
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

/// Input chunk count. Keeps each LargeStringArray materialization bounded
/// instead of building one 20GB Arrow payload for the full 10M corpus.
const APPEND_CHUNKS: usize = 16;

const TOP_K: usize = 10;

// ─── Fixtures ────────────────────────────────────────────────────────

static TEXT_CORPUS: OnceLock<corpus::MmapTextCorpus> = OnceLock::new();
static INFINO: OnceLock<Supertable> = OnceLock::new();

fn text_corpus() -> &'static corpus::MmapTextCorpus {
    TEXT_CORPUS.get_or_init(|| corpus::MmapTextCorpus::generate(N_DOCS, 1))
}

fn infino_supertable() -> &'static Supertable {
    INFINO.get_or_init(|| build_supertable_infino(text_corpus(), parallel_pool()))
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
    .with_commit_threshold_size_mb(1024)
}

/// Build an FTS-only supertable from an mmap-backed text corpus. Each
/// chunk is materialized into an Arrow array, appended, committed, and
/// dropped before the next chunk so the bench does not pin all 10M docs in
/// both the fixture and writer buffer.
fn build_supertable_infino(
    corpus: &corpus::MmapTextCorpus,
    reader_pool: Arc<ThreadPool>,
) -> Supertable {
    let st = Supertable::create(supertable_options(reader_pool));
    let mut w = st.writer().expect("writer");
    let chunk_size = corpus.n_docs().div_ceil(APPEND_CHUNKS);
    for start in (0..corpus.n_docs()).step_by(chunk_size) {
        let titles = LargeStringArray::from(corpus.chunk_strs(start, chunk_size));
        let batch = RecordBatch::try_new(schema_id_title(), vec![Arc::new(titles)]).expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
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
    let infino = build_supertable_infino(text_corpus(), parallel_pool());
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
        b.iter_with_large_drop(|| {
            build_supertable_infino(black_box(text_corpus()), parallel_pool())
        });
    });

    g.finish();
    let stats = rss_sample.stop_stats();
    let _ = rss::write_rss_stats(
        group_name::SUPERTABLE_FTS_BUILD,
        "infino_auto_writer_pool",
        stats,
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
        ("three_wide_or", "term00001 term00050 term00100"),
        ("three_similar_or", "term00050 term00051 term00052"),
        (
            "five_term_or",
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
    let stats = rss_sample.stop_stats();
    let search_ids = [
        "single_rare_supertable_top10",
        "single_common_supertable_top10",
        "two_term_or_supertable_top10",
        "three_wide_or_supertable_top10",
        "three_similar_or_supertable_top10",
        "five_term_or_supertable_top10",
        "prefix_supertable_top10",
    ];
    for bid in search_ids {
        let _ = rss::write_rss_stats(group_name::SUPERTABLE_FTS_SEARCH, bid, stats);
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
        "| Engine                  | Time       | Throughput | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |\n",
    );
    body.push_str(
        "|-------------------------|------------|------------|-----------|------------|-----------|------------|\n",
    );
    let time = ns.map(fmt_time).unwrap_or_else(|| "—".into());
    let thrpt = ns
        .map(|n| fmt_throughput((N_DOCS as f64) / (n / 1e9)))
        .unwrap_or_else(|| "—".into());
    let rss_cell = peak_rss.map(rss::fmt_bytes).unwrap_or_else(|| "—".into());
    let median_rss = rss::fmt_median_rss(group, bench);
    let p90_rss = rss::fmt_p90_rss(group, bench);
    let rss_delta = rss::fmt_peak_rss_delta(group, bench);
    body.push_str(&format!(
        "| infino_auto_writer_pool | {time:10} | {thrpt:10} | {rss_cell:9} | {median_rss:10} | {p90_rss:9} | {rss_delta:10} |\n"
    ));
    body.push_str(&format!(
        "\n*Output cardinality: infino emits `min(writer_pool.threads, chunk_rows)` superfiles \
         per commit across {APPEND_CHUNKS} bounded append chunks (writer auto = cpus/2). \
         Override with `INFINO_SUPERTABLE__WRITER_THREADS=N` for a specific shard count.*\n",
    ));

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
    body.push_str(
        "| Query          | infino     | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |\n",
    );
    body.push_str(
        "|----------------|------------|-----------|------------|-----------|------------|\n",
    );
    let queries = [
        "single_rare",
        "single_common",
        "two_term_or",
        "three_wide_or",
        "three_similar_or",
        "five_term_or",
        "prefix",
    ];
    for q in queries {
        let bid = format!("{q}_supertable_top10");
        let inf = read_mean_ns(group, &bid);
        let inf_s = inf.map(fmt_time).unwrap_or_else(|| "—".into());
        let rss_cell = rss::read_peak_rss_bytes(group, &bid)
            .map(rss::fmt_bytes)
            .unwrap_or_else(|| "—".into());
        let median_rss = rss::fmt_median_rss(group, &bid);
        let p90_rss = rss::fmt_p90_rss(group, &bid);
        let rss_delta = rss::fmt_peak_rss_delta(group, &bid);
        body.push_str(&format!(
            "| {q:14} | {inf_s:10} | {rss_cell:9} | {median_rss:10} | {p90_rss:9} | {rss_delta:10} |\n"
        ));
    }

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/fts/supertable/search".into(),
        body,
    });
}

criterion_group!(benches, bench_ingest, bench_search);
