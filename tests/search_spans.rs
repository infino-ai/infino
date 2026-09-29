// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The spans a search exports at the `info` level, which is what a trace
//! pipeline keeps by default.
//!
//! A search on a table of several superfiles splits its root span into
//! phases (superfile selection, term lookup, the fan-out, the vector legs),
//! each carrying counts rather than a span per superfile. A search on a
//! single superfile keeps only its root. These tests pin both shapes, that
//! every span of a search lands in that search's trace (SQL table functions
//! included, whatever plan DataFusion wraps around them), and that a phase
//! span ends where its phase does.
//!
//! One test function: the capture layer is the process-wide subscriber,
//! and the engine's runtime threads only ever see that one.

#![cfg(feature = "detailed-tracing")]
#![deny(clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, Mutex},
    time::Instant,
};

use infino::{
    Bm25SearchOptions, BoolMode, Connection, IndexSpec, Metric, Supertable,
    arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch},
    arrow_schema::{DataType, Field, Schema, SchemaRef},
    connect,
};
use tempfile::TempDir;
use tracing::{
    Subscriber,
    field::{Field as TraceField, Visit},
    span,
};
use tracing_subscriber::{EnvFilter, Layer, layer::Context, prelude::*, registry::LookupSpan};

/// Embedding width, the engine's minimum.
const DIM: usize = 16;
/// Commits to the table whose searches split into phases. Each writes at
/// least one superfile: a vector-indexed commit writes one per vector cell
/// its rows land in.
const COMMITS: usize = 6;
/// Rows per commit.
const ROWS: usize = 12;
/// Top-K for every search.
const K: usize = 5;
/// Title words; every row carries three of them.
const WORDS: &[&str] = &["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"];

/// A span as the capture layer saw it, once closed.
#[derive(Debug, Clone)]
struct Captured {
    name: &'static str,
    /// Names from the trace root down to this span's parent.
    ancestors: Vec<&'static str>,
    fields: BTreeMap<&'static str, String>,
    start: Instant,
    end: Instant,
}

impl Captured {
    fn root(&self) -> &'static str {
        self.ancestors.first().copied().unwrap_or(self.name)
    }

    fn parent(&self) -> Option<&'static str> {
        self.ancestors.last().copied()
    }

    fn field(&self, name: &str) -> Option<u64> {
        self.fields.get(name).and_then(|v| v.parse().ok())
    }
}

/// What a span holds while open.
struct Open {
    ancestors: Vec<&'static str>,
    fields: BTreeMap<&'static str, String>,
    start: Instant,
}

struct FieldVisitor<'a>(&'a mut BTreeMap<&'static str, String>);

impl Visit for FieldVisitor<'_> {
    fn record_u64(&mut self, field: &TraceField, value: u64) {
        self.0.insert(field.name(), value.to_string());
    }

    fn record_i64(&mut self, field: &TraceField, value: i64) {
        self.0.insert(field.name(), value.to_string());
    }

    fn record_str(&mut self, field: &TraceField, value: &str) {
        self.0.insert(field.name(), value.to_string());
    }

    fn record_debug(&mut self, field: &TraceField, value: &dyn fmt::Debug) {
        self.0.insert(field.name(), format!("{value:?}"));
    }
}

/// Keeps every closed span, in close order.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<Captured>>>);

impl Capture {
    fn take(&self) -> Vec<Captured> {
        std::mem::take(&mut *self.0.lock().expect("capture lock"))
    }
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Capture {
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).expect("new span is registered");
        let mut ancestors: Vec<&'static str> = span.scope().skip(1).map(|s| s.name()).collect();
        ancestors.reverse();
        let mut fields = BTreeMap::new();
        attrs.record(&mut FieldVisitor(&mut fields));
        span.extensions_mut().insert(Open {
            ancestors,
            fields,
            start: Instant::now(),
        });
    }

    fn on_record(&self, id: &span::Id, values: &span::Record<'_>, ctx: Context<'_, S>) {
        let span = ctx.span(id).expect("recorded span is registered");
        if let Some(open) = span.extensions_mut().get_mut::<Open>() {
            values.record(&mut FieldVisitor(&mut open.fields));
        }
    }

    fn on_close(&self, id: span::Id, ctx: Context<'_, S>) {
        let span = ctx.span(&id).expect("closed span is registered");
        let Some(open) = span.extensions_mut().remove::<Open>() else {
            return;
        };
        self.0.lock().expect("capture lock").push(Captured {
            name: span.name(),
            ancestors: open.ancestors,
            fields: open.fields,
            start: open.start,
            end: Instant::now(),
        });
    }
}

/// `title`, plus the `emb` embedding when `vectors`.
fn schema(vectors: bool) -> SchemaRef {
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let mut fields = vec![Field::new("title", DataType::LargeUtf8, false)];
    if vectors {
        fields.push(Field::new(
            "emb",
            DataType::FixedSizeList(item, DIM as i32),
            false,
        ));
    }
    Arc::new(Schema::new(fields))
}

/// A deterministic, well-spread embedding for row `n`.
fn embedding(n: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| (((n * 7 + d * 13) % 17) as f32 - 8.0) / 8.0)
        .collect()
}

/// Commit `commit`'s rows, with embeddings when `vectors`.
fn batch(commit: usize, vectors: bool) -> RecordBatch {
    let rows = commit * ROWS..(commit + 1) * ROWS;
    let titles: Vec<String> = rows
        .clone()
        .map(|n| {
            let w = |i: usize| WORDS[i % WORDS.len()];
            format!("{} {} {}", w(n), w(n / 2), w(n / 3))
        })
        .collect();
    let mut columns: Vec<ArrayRef> = vec![Arc::new(LargeStringArray::from(titles))];
    if vectors {
        let flat: Vec<f32> = rows.flat_map(embedding).collect();
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let values = Arc::new(Float32Array::from(flat));
        columns.push(Arc::new(FixedSizeListArray::new(
            item, DIM as i32, values, None,
        )));
    }
    RecordBatch::try_new(schema(vectors), columns).expect("batch")
}

/// A table of `commits` appends, FTS-indexed on `title` and, when
/// `vectors`, vector-indexed on `emb`.
fn create(db: &Connection, name: &str, commits: usize, vectors: bool) -> Supertable {
    let mut index = IndexSpec::new().fts("title");
    if vectors {
        index = index.vector("emb", DIM, Metric::Cosine);
    }
    let table = db
        .create_table(name, schema(vectors), index)
        .expect("create_table");
    for commit in 0..commits {
        table.append(&batch(commit, vectors)).expect("append");
    }
    table
}

fn rows(batches: &[RecordBatch]) -> u64 {
    batches.iter().map(|b| b.num_rows() as u64).sum()
}

/// A bm25 search on `table`, returning its row count.
fn bm25(table: &Supertable) -> impl Fn() -> u64 + '_ {
    move || {
        let hits = table.bm25_search(
            "title",
            "alpha delta",
            K,
            Bm25SearchOptions::default(),
            None,
        );
        rows(&hits.expect("bm25"))
    }
}

/// A vector search on `table` for `q`, returning its row count.
fn vector<'a>(table: &'a Supertable, q: &'a [f32]) -> impl Fn() -> u64 + 'a {
    move || {
        rows(
            &table
                .vector_search("emb", q, K, None, None)
                .expect("vector"),
        )
    }
}

fn names(spans: &[Captured]) -> Vec<&'static str> {
    let mut names: Vec<_> = spans.iter().map(|s| s.name).collect();
    names.sort_unstable();
    names
}

fn span<'a>(spans: &'a [Captured], name: &str) -> &'a Captured {
    spans
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("no `{name}` span in {:?}", names(spans)))
}

/// The spans of one search: run `op` (warm, after a first run) and keep the
/// spans whose trace root is `root`. Asserts the search's own spans all
/// landed in that trace and that nothing per superfile was exported.
fn trace_of(capture: &Capture, root: &str, op: impl Fn() -> u64) -> (Vec<Captured>, u64) {
    op();
    capture.take();
    let out = op();
    let spans = capture.take();
    for s in &spans {
        assert!(
            !matches!(s.name, "open_reader" | "get_range"),
            "`{}` is a span per superfile or range read; it must stay below `info`",
            s.name
        );
        let ours = [
            "fts.", "vector.", "hybrid.", "tvf.", "search.", "sql.", "scan.",
        ];
        if ours.iter().any(|prefix| s.name.starts_with(prefix)) {
            assert_eq!(
                s.root(),
                root,
                "`{}` (under {:?}) left the `{root}` trace",
                s.name,
                s.ancestors
            );
        }
    }
    let spans: Vec<Captured> = spans.into_iter().filter(|s| s.root() == root).collect();
    let root_span = span(&spans, root);
    assert_eq!(root_span.field("rows_out"), Some(out), "`{root}` rows_out");
    assert!(
        root_span.fields.contains_key("store_gets"),
        "`{root}` records the store delta: {:?}",
        root_span.fields
    );
    (spans, out)
}

/// `span` ends before `next` starts: the phase span stops where its phase
/// does, not where its handle happens to go out of scope.
fn ends_before(spans: &[Captured], first: &str, next: &str) {
    let (a, b) = (span(spans, first), span(spans, next));
    assert!(
        a.end <= b.start,
        "`{first}` must end before `{next}` starts ({:?} after)",
        a.end.saturating_duration_since(b.start)
    );
}

/// The opens a tiered phase counted by cache tier, summed.
fn tier_total(span: &Captured) -> u64 {
    ["memory", "disk", "lazy", "source", "coalesced", "streamed"]
        .iter()
        .map(|tier| {
            span.field(tier)
                .unwrap_or_else(|| panic!("`{}` has no `{tier}`", span.name))
        })
        .sum()
}

#[test]
fn search_spans_split_into_phases_only_where_there_is_something_to_split() {
    let capture = Capture::default();
    tracing_subscriber::registry()
        .with(capture.clone().with_filter(EnvFilter::new("info")))
        .init();

    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let many = create(&db, "many", COMMITS, true);
    // No vector index, so its one commit is one superfile.
    let one = create(&db, "one", 1, false);
    let q = embedding(3);
    let q_csv = q.iter().map(f32::to_string).collect::<Vec<_>>().join(",");
    let db = &db;
    let sql = |text: String| move || rows(&db.query_sql(&text).expect("query_sql"));

    // bm25 on several superfiles: select, term lookup and fan-out as phases
    // under the async root, each ending before the next begins, and the
    // fan-out's opens counted by tier.
    let (spans, _) = trace_of(&capture, "bm25_search", bm25(&many));
    for phase in ["fts.select_superfiles", "fts.term_index", "fts.fanout"] {
        assert_eq!(span(&spans, phase).parent(), Some("bm25_search_async"));
    }
    let select = span(&spans, "fts.select_superfiles");
    let manifest = select
        .field("manifest_superfiles")
        .expect("manifest_superfiles");
    let survivors = select.field("survivors").expect("survivors");
    assert!(manifest >= COMMITS as u64 && survivors <= manifest);
    ends_before(&spans, "fts.select_superfiles", "fts.fanout");
    ends_before(&spans, "fts.term_index", "fts.fanout");
    // Ranked by score ceiling, so a unit that can no longer reach the top
    // `k` is skipped unopened, and uncounted.
    let fanout = span(&spans, "fts.fanout");
    let opened = tier_total(fanout);
    assert!(opened > 0 && Some(opened) <= fanout.field("units"));

    // The same search on one superfile keeps only its root and resolve.
    let (spans, _) = trace_of(&capture, "bm25_search", bm25(&one));
    assert_eq!(
        names(&spans),
        ["bm25_search", "bm25_search_async", "search.resolve"]
    );

    // token_match and exact_match: the same three phases.
    let (spans, _) = trace_of(&capture, "token_match", || {
        let hits = many.token_match("title", "alpha delta", BoolMode::And, None);
        rows(&hits.expect("token_match"))
    });
    ends_before(&spans, "fts.select_superfiles", "fts.fanout");
    let fanout = span(&spans, "fts.fanout");
    assert_eq!(Some(tier_total(fanout)), fanout.field("units"));
    let (spans, _) = trace_of(&capture, "exact_match", || {
        rows(
            &many
                .exact_match("title", "alpha alpha alpha", None)
                .expect("exact_match"),
        )
    });
    ends_before(&spans, "fts.select_superfiles", "fts.fanout");

    // Vector: the route (admit) stage ends before the scan starts, and the
    // scan counts its opens by tier.
    let (spans, _) = trace_of(&capture, "vector_search", vector(&many, &q));
    ends_before(&spans, "vector.route", "vector.scan");
    let scan = span(&spans, "vector.scan");
    assert_eq!(Some(tier_total(scan)), scan.field("units"));

    // Hybrid: one leg span each, under the async root, and the fusion's
    // input and output counts.
    let (spans, out) = trace_of(&capture, "hybrid_search", || {
        let hits = many.hybrid_search("title", "alpha", BoolMode::Or, "emb", &q, K, None);
        rows(&hits.expect("hybrid"))
    });
    assert_eq!(
        span(&spans, "hybrid.bm25").parent(),
        Some("hybrid_search_async")
    );
    assert_eq!(
        span(&spans, "hybrid.vector").parent(),
        Some("hybrid_search_async")
    );
    assert_eq!(span(&spans, "fts.fanout").parent(), Some("hybrid.bm25"));
    assert!(
        span(&spans, "vector.scan")
            .ancestors
            .contains(&"hybrid.vector")
    );
    assert_eq!(span(&spans, "hybrid.fuse").field("rows_out"), Some(out));

    // SQL table functions: a span each under `sql.execute`, with the kernel's
    // phases beneath it, named after the function.
    let tvfs = [
        (
            "tvf.bm25_search",
            format!("bm25_search('many', 'title', 'alpha', {K})"),
        ),
        (
            "tvf.bm25_search_prefix",
            format!("bm25_search_prefix('many', 'title', 'alp', {K})"),
        ),
        (
            "tvf.vector_search",
            format!("vector_search('many', 'emb', '{q_csv}', {K})"),
        ),
        (
            "tvf.hybrid_search",
            format!("hybrid_search('many', 'title', 'alpha', 'emb', '{q_csv}', {K})"),
        ),
        (
            "tvf.token_match",
            "token_match('many', 'title', 'alpha delta', 'and')".to_string(),
        ),
        (
            "tvf.exact_match",
            "exact_match('many', 'title', 'alpha alpha alpha')".to_string(),
        ),
    ];
    for (name, tvf) in &tvfs {
        let (spans, out) = trace_of(&capture, "query_sql", sql(format!("SELECT _id FROM {tvf}")));
        let tvf_span = span(&spans, name);
        assert_eq!(tvf_span.parent(), Some("sql.execute"), "`{name}`");
        assert_eq!(tvf_span.field("rows_out"), Some(out), "`{name}` rows_out");
    }

    // A plan DataFusion wraps around the function (a sort, a join) still
    // runs it inside the query's trace; `trace_of` checks every span's root.
    let bm25_tvf = &tvfs[0].1;
    let vector_tvf = &tvfs[2].1;
    trace_of(
        &capture,
        "query_sql",
        sql(format!("SELECT _id FROM {bm25_tvf} ORDER BY _id")),
    );
    let join = format!("SELECT a._id FROM {bm25_tvf} a JOIN {vector_tvf} b ON a._id = b._id");
    let (spans, _) = trace_of(&capture, "query_sql", sql(join));
    span(&spans, "tvf.bm25_search");
    span(&spans, "tvf.vector_search");

    // The table scan the phases mirror: its selection span also ends where
    // selection does.
    let (spans, _) = trace_of(
        &capture,
        "query_sql",
        sql("SELECT COUNT(*) FROM many WHERE title LIKE '%alpha%'".to_string()),
    );
    ends_before(&spans, "scan.select_superfiles", "scan.open_files");
    let open = span(&spans, "scan.open_files");
    assert_eq!(Some(tier_total(open)), open.field("files"));
}
