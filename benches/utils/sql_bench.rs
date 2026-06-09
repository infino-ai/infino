// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! SQL bench (infino-only entry point).
//!
//! Build + query numbers are measured through the engine-generic SQL
//! harness (`run_sql::<InfinoSqlEngine>`) — the same path the cross-engine
//! comparison uses. The canonical 1-writer build produces the queryable
//! in-memory `Supertable`; correctness and hot queries run against that
//! exact artifact. A separate `N writers` build row measures parallel
//! ingest throughput.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench sql
//! INFINO_BENCH_DOC_COUNT=100000 cargo bench --bench sql
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench sql
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use arrow_array::Int64Array;
use infino::supertable::Supertable;

use crate::corpus::{self, MmapTextCorpus};
use crate::harness::{
    EngineSqlResult, InfinoSqlEngine, InfinoSqlIndex, SqlEngine, SqlQuery, SqlRow, SqlRunConfig,
    run_sql_with_index, sample_query_csv, scatter_key,
};
use crate::markdown::{fmt_count, fmt_throughput, fmt_time};
use crate::report::{Better, Block, Cell, Report, Section, metric, text};
use crate::rss::{self, PeakSampler, RssStats};

/// Timed query repetitions per query (after one warmup).
const ITERS: usize = 10;

/// Deterministic category labels assigned round-robin by doc id, so the
/// planted distribution is exactly known for the correctness gate.
const CATEGORIES: &[&str] = &["rust", "python", "go", "sql"];

/// The SQL query battery. `SELECT *` scans the whole table; the filters
/// exercise scalar pushdown on a text column and a numeric column; the
/// aggregates exercise the grouped/counted paths.
// Aggregations + count-based filters: each reads the column(s) but
// collapses to a few rows, so the measurement is read + compute
// throughput — not row materialization. (A bare `SELECT col` returning
// every row would just measure output transfer, so it's deliberately
// absent — analytical benchmarks like ClickBench / TPC-H don't include
// one.)
const SQL_BATTERY: &[SqlQuery] = &[
    // Aggregation over the whole title column (decodes every value,
    // returns one row).
    SqlQuery {
        name: "agg_max_title",
        sql: "SELECT MAX(title) AS m FROM supertable",
    },
    // Selective filters as match counts (process all rows, return one).
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

/// Build the planted `(doc_id, title, category, score)` rows borrowing
/// titles from the shared mmap corpus. `category` cycles through
/// [`CATEGORIES`]; `score` is `doc_id % 100`.
fn sql_rows<'a>(corpus_rows: &'a [(u64, &'a str)]) -> Vec<SqlRow<'a>> {
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

/// Number of rows whose category is `rust` (`doc_id % 4 == 0`).
fn expected_rust(n_docs: usize) -> usize {
    n_docs.div_ceil(CATEGORIES.len())
}

/// Extract the single `COUNT(*)` value from a one-row aggregate result.
fn count_value(table: &Supertable, sql: &str) -> i64 {
    let batches = table.query_sql(sql).expect("query_sql count");
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count column is Int64")
        .value(0)
}

/// One measured infino-only SQL table-function query (bm25 / vector /
/// hybrid). These are reachable through the same `query_sql` read path —
/// hybrid is just another SQL option, not a separate harness.
struct TvfStat {
    name: &'static str,
    p50: Duration,
    rows: usize,
    rss: RssStats,
}

fn p50(samples: &mut [Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort_unstable();
    samples[(samples.len() - 1) / 2]
}

fn timed_tvf(index: &InfinoSqlIndex, name: &'static str, sql: &str) -> TvfStat {
    let sampler = PeakSampler::start_default();
    let warm = InfinoSqlEngine::read(index, sql);
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let out = InfinoSqlEngine::read(index, sql);
        samples.push(t0.elapsed());
        black_box(out);
    }
    let rss = sampler.stop_stats();
    TvfStat {
        name,
        p50: p50(&mut samples),
        rows: warm.rows,
        rss,
    }
}

// ─── Entry point ──────────────────────────────────────────────────────

pub fn run() {
    let n_docs = corpus::superfile_docs();
    eprintln!("[sql] generating {}-row corpus...", fmt_count(n_docs));
    let corpus = MmapTextCorpus::generate(n_docs, 1);
    let corpus_rows = corpus.rows();
    let rows = sql_rows(&corpus_rows);

    eprintln!(
        "[sql] run_sql: build + {ITERS}-iter query battery over {} rows...",
        fmt_count(n_docs)
    );
    let (result, index) = run_sql_with_index::<InfinoSqlEngine>(
        SqlRunConfig {
            iters: ITERS,
            parallel: corpus::parallel_writers(),
        },
        &rows,
        SQL_BATTERY,
    );

    // Correctness gate on the exact 1-writer artifact measured above.
    eprintln!("[sql] correctness: using measured 1-writer artifact...");
    let table = index.table();
    let total = count_value(table, "SELECT COUNT(*) AS n FROM supertable");
    assert_eq!(
        total as usize, n_docs,
        "COUNT(*) must equal the row count; got {total}"
    );
    let rust = count_value(
        table,
        "SELECT COUNT(*) AS n FROM supertable WHERE category = 'rust'",
    );
    assert_eq!(
        rust as usize,
        expected_rust(n_docs),
        "rust-category COUNT(*) must match the planted distribution; got {rust}"
    );
    eprintln!("[sql] correctness OK: COUNT(*) == {n_docs}, rust == {rust}");

    // Infino-only SQL options: table functions on the same `query_sql`
    // resolve through the same `query_sql` read path against the indexed
    // table. Hybrid is just a SQL option, measured here as another query.
    eprintln!(
        "[sql] measuring search table-function queries (bm25 / vector / hybrid / token / exact)..."
    );
    let qv = sample_query_csv();
    let sample_title = corpus_rows[corpus_rows.len() / 2].1.replace('\'', "''");

    let tvf = vec![
        timed_tvf(
            &index,
            "bm25_search",
            "SELECT _id FROM bm25_search('title', 'term00001', 10)",
        ),
        timed_tvf(
            &index,
            "vector_search",
            &format!("SELECT _id FROM vector_search('emb', '{qv}', 10)"),
        ),
        timed_tvf(
            &index,
            "hybrid_search",
            &format!("SELECT _id FROM hybrid_search('title', 'term00001', 'emb', '{qv}', 10)"),
        ),
        // Degenerate: the two most-frequent Zipf terms (rank 1 & 2)
        // occur in ~every doc, so this AND matches the whole table — a
        // worst case dominated by materializing 1M result rows.
        timed_tvf(
            &index,
            "token_match (all rows)",
            "SELECT _id FROM token_match('title', 'term00001 term00002', 'and')",
        ),
        // Realistic: a doc-unique token (df=1) — the selective shape a
        // WHERE predicate actually hits, returning a tiny result.
        timed_tvf(
            &index,
            "token_match (selective)",
            "SELECT _id FROM token_match('title', 'doc0500000', 'and')",
        ),
        timed_tvf(
            &index,
            "exact_match",
            &format!("SELECT _id FROM exact_match('title', '{sample_title}')"),
        ),
    ];

    // Selective equality (one matching row), no-index column vs
    // FTS-indexed column — on TWO column shapes that expose when the
    // index actually beats DataFusion's min/max page pruning:
    //   * `title`  is sorted by ingest order (titles start with
    //     `doc{id:07}`), so its page min/max ranges isolate the value —
    //     DataFusion prunes well on its own and the scan stays cheap.
    //   * `key`    is a high-cardinality hash uncorrelated with row
    //     order, so every page's min/max spans the whole domain —
    //     min/max can prune nothing and DataFusion must scan all pages,
    //     while the FTS index resolves the single row's page directly.
    // The unsorted `key` row is the honest win-case; the sorted `title`
    // row shows the index adds little when min/max already works.
    eprintln!("[sql] measuring no-index vs FTS-index equality (sorted title vs unsorted key)...");
    let sample_key = scatter_key(corpus_rows[corpus_rows.len() / 2].0);
    let plain_scan = vec![
        timed_tvf(
            &index,
            "WHERE title = ?  (sorted col, min/max prunes)",
            &format!("SELECT title FROM supertable WHERE title_noidx = '{sample_title}'"),
        ),
        timed_tvf(
            &index,
            "WHERE key   = ?  (unsorted col, min/max defeated)",
            &format!("SELECT key FROM supertable WHERE key_noidx = '{sample_key}'"),
        ),
    ];
    let fts_pushdown = vec![
        timed_tvf(
            &index,
            "WHERE title = ?  (sorted col, min/max prunes)",
            &format!("SELECT title FROM supertable WHERE title = '{sample_title}'"),
        ),
        timed_tvf(
            &index,
            "WHERE key   = ?  (unsorted col, min/max defeated)",
            &format!("SELECT key FROM supertable WHERE key = '{sample_key}'"),
        ),
    ];

    // Aggregate **shapes** (COUNT / SUM / MAX / AVG, plus a GROUP BY)
    // over the surviving candidate rows of an FTS-resolvable predicate,
    // run two ways: on a non-indexed column (DataFusion full scan) vs the
    // FTS-indexed column (the WHERE resolves through `token_match`). The
    // selective rows use the unsorted `key` (min/max defeated → the
    // honest win-case); the final `SUM … bucket IN (all)` row is the
    // many-matches case where matches saturate every page so no page can
    // be skipped and the index can't win (it just adds overhead — this is
    // the case a selectivity gate must catch ahead of time).
    eprintln!(
        "[sql] measuring aggregate shapes over a candidate set: DataFusion only vs token_match..."
    );
    const BUCKET_IN_ALL: &str = "('b0','b1','b2','b3','b4','b5','b6','b7','b8','b9')";
    let agg_scan = vec![
        timed_tvf(
            &index,
            "COUNT(*)            key=? (1 row)",
            &format!("SELECT COUNT(*) AS a FROM supertable WHERE key_noidx = '{sample_key}'"),
        ),
        timed_tvf(
            &index,
            "SUM(rating)         key=? (1 row)",
            &format!("SELECT SUM(rating) AS a FROM supertable WHERE key_noidx = '{sample_key}'"),
        ),
        timed_tvf(
            &index,
            "MAX(rating)         key=? (1 row)",
            &format!("SELECT MAX(rating) AS a FROM supertable WHERE key_noidx = '{sample_key}'"),
        ),
        timed_tvf(
            &index,
            "AVG(rating)         key=? (1 row)",
            &format!("SELECT AVG(rating) AS a FROM supertable WHERE key_noidx = '{sample_key}'"),
        ),
        timed_tvf(
            &index,
            "SUM(rating) bucket IN all (1M rows)",
            &format!(
                "SELECT SUM(rating) AS a FROM supertable WHERE bucket_noidx IN {BUCKET_IN_ALL}"
            ),
        ),
    ];
    let agg_idx = vec![
        timed_tvf(
            &index,
            "COUNT(*)            key=? (1 row)",
            &format!("SELECT COUNT(*) AS a FROM supertable WHERE key = '{sample_key}'"),
        ),
        timed_tvf(
            &index,
            "SUM(rating)         key=? (1 row)",
            &format!("SELECT SUM(rating) AS a FROM supertable WHERE key = '{sample_key}'"),
        ),
        timed_tvf(
            &index,
            "MAX(rating)         key=? (1 row)",
            &format!("SELECT MAX(rating) AS a FROM supertable WHERE key = '{sample_key}'"),
        ),
        timed_tvf(
            &index,
            "AVG(rating)         key=? (1 row)",
            &format!("SELECT AVG(rating) AS a FROM supertable WHERE key = '{sample_key}'"),
        ),
        timed_tvf(
            &index,
            "SUM(rating) bucket IN all (1M rows)",
            &format!("SELECT SUM(rating) AS a FROM supertable WHERE bucket IN {BUCKET_IN_ALL}"),
        ),
    ];

    let mut report = Report::load("sql");
    emit_build(&mut report, n_docs, &corpus, &result);
    emit_query(
        &mut report,
        n_docs,
        &result,
        &tvf,
        &plain_scan,
        &fts_pushdown,
        &agg_scan,
        &agg_idx,
    );
    report.save();
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

fn emit_build(
    report: &mut Report,
    n_docs: usize,
    corpus: &MmapTextCorpus,
    result: &EngineSqlResult,
) {
    let input_bytes = corpus.total_bytes() as f64;
    let rows: Vec<Vec<Cell>> = result
        .builds
        .iter()
        .map(|b| {
            let secs = b.wall.as_secs_f64();
            let ns = secs * 1e9;
            let thr = n_docs as f64 / secs;
            let mbps = input_bytes / secs / 1e6;
            let mut cells = vec![
                text(writer_label(b.writers)),
                metric(ns, fmt_time(ns), Better::Lower),
                metric(thr, fmt_throughput(thr), Better::Higher),
                metric(mbps, format!("{mbps:.1} MB/s"), Better::Higher),
            ];
            cells.extend(rss_cells(b.rss));
            cells
        })
        .collect();
    report.emit(&Section {
        anchor: "bench/sql/build".into(),
        title: format!(
            "SQL — ingest, in-memory supertable ({} rows: title + category + score)",
            fmt_count(n_docs)
        ),
        note: "Build path: `SupertableWriter::append` + `commit` into an in-memory supertable, through \
               the engine-generic `run_sql` driver the cross-engine comparison also uses. Rows are by \
               writer count: `1 writer` is the canonical build queries run against; `N writers` is the \
               sharded parallel build. Δ is vs the previous run."
            .into(),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "Build".into(),
                "Time".into(),
                "Throughput".into(),
                "Bandwidth".into(),
                "Peak RSS".into(),
                "Median RSS".into(),
                "P90 RSS".into(),
            ],
            rows,
        }],
    });
}

fn query_row(name: &str, p50: Duration, rows: usize, stats: RssStats) -> Vec<Cell> {
    let ns = p50.as_secs_f64() * 1e9;
    let mut cells = vec![
        text(name),
        metric(ns, fmt_time(ns), Better::Lower),
        text(fmt_count(rows)),
    ];
    cells.extend(rss_cells(stats));
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

#[allow(clippy::too_many_arguments)]
fn emit_query(
    report: &mut Report,
    n_docs: usize,
    result: &EngineSqlResult,
    tvf: &[TvfStat],
    plain_scan: &[TvfStat],
    fts_pushdown: &[TvfStat],
    agg_scan: &[TvfStat],
    agg_idx: &[TvfStat],
) {
    let to_rows = |stats: &[TvfStat]| -> Vec<Vec<Cell>> {
        stats
            .iter()
            .map(|t| query_row(t.name, t.p50, t.rows, t.rss))
            .collect()
    };
    let scalar = Block {
        subtitle:
            "Aggregations & count-filters (read + compute, return few rows — not the index A/B)"
                .into(),
        headers: query_headers(),
        rows: result
            .queries
            .iter()
            .map(|q| query_row(q.name, q.p50, q.rows, q.rss))
            .collect(),
    };
    let search = Block {
        subtitle: "Search table functions (bm25 / vector / hybrid / token / exact)".into(),
        headers: query_headers(),
        rows: tvf
            .iter()
            .map(|t| query_row(t.name, t.p50, t.rows, t.rss))
            .collect(),
    };
    // The honest A/B: same selective equality (1 matching row), no index
    // vs FTS index. Two blocks so the labels are unmistakable.
    let plain = Block {
        subtitle:
            "Plain Scan (DataFusion only) — selective equality, 1 row (sorted vs unsorted col)"
                .into(),
        headers: query_headers(),
        rows: to_rows(plain_scan),
    };
    let pushdown = Block {
        subtitle:
            "FTS-pushdown (DataFusion + Infino) — SAME equality, 1 row (sorted vs unsorted col)"
                .into(),
        headers: query_headers(),
        rows: to_rows(fts_pushdown),
    };
    // Aggregate shapes over the candidate set, the two access paths
    // back-to-back. The 1-row `key=?` rows are the win-case (unsorted
    // key → min/max defeated → index reads one page); the `bucket IN all`
    // row is the many-matches case where the index can't help.
    let agg_scan_block = Block {
        subtitle: "Aggregate over FTS candidates — Full Scan (DataFusion only)".into(),
        headers: query_headers(),
        rows: to_rows(agg_scan),
    };
    let agg_idx_block = Block {
        subtitle: "Aggregate over FTS candidates — FTS-pushdown (DataFusion + Infino token_match)"
            .into(),
        headers: query_headers(),
        rows: to_rows(agg_idx),
    };
    report.emit(&Section {
        anchor: "bench/sql/query".into(),
        title: format!(
            "SQL — query, in-memory supertable ({} rows)",
            fmt_count(n_docs)
        ),
        note: "Hot p50 over `Supertable::query_sql` against the canonical 1-writer table. The headline \
               comparison is the last two blocks: the *same* selective equality (one matching row) run \
               against a non-indexed column (Plain Scan — DataFusion decodes + filters) vs the \
               byte-identical FTS-indexed `title` column (FTS-pushdown — infino's token index selects \
               the candidate row, DataFusion verifies). Same predicate, same 1-row result, so the gap \
               is purely the index. The first block is aggregations & count-filters (read + compute, \
               return few rows) — general engine context, not a like-for-like index comparison; there \
               is no bare `SELECT col` row because that only measures row materialization. `Rows` is \
               the result-set size. Δ is vs the previous run."
            .into(),
        // Comparison blocks adjacent: the 1-row equality (Plain Scan vs
        // FTS-pushdown), then the 10%-filter aggregate (Full Scan vs
        // token_match candidate); the bm25 / vector / hybrid TVFs last.
        blocks: vec![
            scalar,
            plain,
            pushdown,
            agg_scan_block,
            agg_idx_block,
            search,
        ],
    });
}
