// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`Supertable::sorted_search`]: the first `k` rows of the table by one
//! range-indexed column, ties broken by `_id` ascending, without SQL.
//!
//! A range index holds its column in order, and keeps the rows sharing a
//! value in `_id` order, so a superfile's first `k` entries in it are its
//! first `k` rows by the column and `_id`. The answer is the best `k` of
//! those across superfiles: each superfile reads a block or two of its
//! index, the candidates are ordered by key and `_id`, and only the `k`
//! kept are resolved to rows. No statement is planned, so the cost is the
//! index reads and the `k` rows, whatever the table's size.
//!
//! The index answers only when every superfile holds the column in a range
//! index with its ties in `_id` order, and when at least `k` rows have a
//! value (a missing value sorts last and is in no index). Otherwise the
//! same sort runs as SQL, which answers alike, only slower.

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    slice::from_ref,
    sync::Arc,
    time::Instant,
};

use arrow::{
    array::{Array, AsArray},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DFSchema, ScalarValue, tree_node::TreeNode},
    execution::context::ExecutionProps,
    logical_expr::{
        Expr,
        simplify::SimplifyContext,
        utils::{conjunction, split_conjunction},
    },
    optimizer::{
        analyzer::type_coercion::TypeCoercionRewriter, simplify_expressions::ExprSimplifier,
    },
    physical_expr::create_physical_expr,
};
use roaring::RoaringBitmap;
use tracing::{Instrument, debug};

#[cfg(test)]
use crate::supertable::query::provider::walk_probe;
use crate::{
    InfinoError,
    runtime_bridge::run_on_pool,
    runtime_metrics::op_stats::timed_section,
    superfile::{SuperfileReader, fts::tokenize::STANDARD_TOKENIZER, range::KeyKind},
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        manifest::{SuperfileEntry, SuperfileUri, summaries::ExactLiterals},
        query::{
            SuperfileHit,
            candidate::single_range,
            dispatch::{self, tag_hits, tombstone_deny_set},
            exec::common::resolve_hits_named,
            provider::{flatten_or_eq, read_column, scalar_distinct},
            skip::ColumnTypeGuard,
            text_match::{TextMatchMode, text_match_call},
        },
        reader_cache::disk::ForegroundQueryGuard,
        tombstones::SidecarCache,
    },
    utils::trace::detail_span,
};

/// The name the trailing synthesized column has in a search's output.
const SCORE_COLUMN: &str = "score";

/// The table a reader's SQL statements read.
const SQL_TABLE: &str = "supertable";

/// One superfile's candidates: `(key, hit)` for its first rows by the key.
type Candidates = Vec<(u64, SuperfileHit)>;

/// The first `k` of the `candidates` that `keep` accepts, in a sort's order by the key,
/// `descending` or not, and then by `_id` ascending, each hit given its `_id` first. The `_id`s
/// come from the superfiles' id spans or id pages, so no other column is read. It reserves no
/// memory: the walk and the range top-k reserve their candidates before they call it.
pub(crate) async fn by_key_then_id(
    reader: &SupertableReader,
    candidates: Candidates,
    descending: bool,
    k: usize,
    keep: impl Fn(u64, &SuperfileHit) -> bool,
) -> Result<Candidates, QueryError> {
    let (keys, mut hits): (Vec<u64>, Vec<SuperfileHit>) = candidates.into_iter().unzip();
    dispatch::attach_stable_ids_to_hits(reader, &mut hits).await?;
    let mut order: Vec<usize> = (0..hits.len())
        .filter(|&i| keep(keys[i], &hits[i]))
        .collect();
    let compare = |&a: &usize, &b: &usize| {
        let by_key = match descending {
            false => keys[a].cmp(&keys[b]),
            true => keys[b].cmp(&keys[a]),
        };
        by_key.then_with(|| hits[a].stable_id.cmp(&hits[b].stable_id))
    };
    // Only the first `k` are kept: pick them, then sort just those.
    if order.len() > k {
        order.select_nth_unstable_by(k, compare);
        order.truncate(k);
    }
    order.sort_by(compare);
    Ok(order.into_iter().map(|i| (keys[i], hits[i])).collect())
}

/// A top-k walk starts only when it expects to find its rows within 4,096, and reads no window past
/// that after its first; past it the scan reads every row, faster for a filter that keeps so few.
pub(crate) const MAX_WALKED_ROWS: usize = 4096;

/// A window reads twice the rows the walk expects to need, so it usually holds the page at once.
pub(crate) const WINDOW_MARGIN: f64 = 2.0;

/// How much larger a window is than the last when the last kept no row.
const WINDOW_GROWTH: usize = 4;

/// A walk's window holds at most 1M index entries (about 80 MB) across its files when the
/// connection has no memory limit; with one, the window is reserved against that limit instead.
const MAX_WINDOW_ENTRIES: usize = 1 << 20;

/// A filter as a top-k walk reads it: the sort column, the interval of its keys the filter keeps
/// (every key when nothing bounds it), and the rest, checked on the rows the walk reads.
struct SortedFilter {
    column: String,
    interval: (u64, u64),
    /// Whether a filter bounds the column, so a row with no value is out of the answer, not unread.
    bounded: bool,
    /// Whether a filter keeps no row (`false`, or a `NULL` the simplifier made), so the answer is
    /// empty.
    keeps_none: bool,
    rest: Option<Expr>,
}

/// The share of rows a filter is expected to keep ([`SupertableReader::expected_share`]), and
/// whether the statistics prove it keeps none.
struct Share {
    share: f64,
    none: bool,
}

/// Where a walk stands: the last row read (key and `_id`), and per file how many rows read hold
/// that key. The next window reads from that key on and skips the rows at or before the cursor.
struct Cursor {
    key: u64,
    id: Option<i128>,
    at_key: HashMap<SuperfileUri, usize>,
}

impl Cursor {
    /// The cursor after a window read `read`, in order, from `before`. `None` when it read none.
    fn past(before: Option<Cursor>, read: &[(u64, SuperfileHit)]) -> Option<Cursor> {
        let &(key, last) = read.last()?;
        // A window that read only rows of the last key adds to that key's count.
        let mut at_key = match before {
            Some(before) if before.key == key => before.at_key,
            _ => HashMap::new(),
        };
        for (_, hit) in read.iter().rev().take_while(|(at, _)| *at == key) {
            *at_key.entry(hit.superfile).or_default() += 1;
        }
        Some(Cursor {
            key,
            id: last.stable_id,
            at_key,
        })
    }

    /// The rows of `superfile` holding the cursor's key that were read.
    fn read_at_key(&self, superfile: SuperfileUri) -> usize {
        self.at_key.get(&superfile).copied().unwrap_or(0)
    }

    /// The rows holding the cursor's key that were read, in every superfile.
    fn read(&self) -> usize {
        self.at_key.values().sum()
    }

    /// Whether the row of `key` and `id` comes after the cursor: by key, then by `_id` ascending.
    fn passed_by(&self, key: u64, id: Option<i128>, descending: bool) -> bool {
        let by_key = match descending {
            false => key.cmp(&self.key),
            true => self.key.cmp(&key),
        };
        by_key.then(id.cmp(&self.id)) == Ordering::Greater
    }
}

impl SupertableReader {
    /// The first `k` rows by `column` (descending with `descending`), ties
    /// by `_id` ascending, as hits in that order, read from the superfiles'
    /// range indexes. `None` when the indexes cannot answer: a superfile
    /// without the column's index, or without its ties in `_id` order, or
    /// fewer than `k` rows with a value. An error when `column` has no
    /// range index at all.
    pub(crate) async fn sorted_hits_async(
        &self,
        column: &str,
        descending: bool,
        k: usize,
    ) -> Result<Option<Vec<SuperfileHit>>, QueryError> {
        self.sorted_hits_within_async(column, descending, k, None, &self.manifest().superfiles)
            .await
    }

    /// [`Self::sorted_hits_async`] over the rows of `files` with a key in `interval` (encoded, both
    /// ends inclusive). Rows with no value are in no interval, so fewer than `k` rows is the whole
    /// answer.
    async fn sorted_hits_within_async(
        &self,
        column: &str,
        descending: bool,
        k: usize,
        interval: Option<(u64, u64)>,
        files: &[Arc<SuperfileEntry>],
    ) -> Result<Option<Vec<SuperfileHit>>, QueryError> {
        Ok(self
            .sorted_keyed_hits_async(column, descending, k, interval, None, files)
            .await?
            .map(|page| page.into_iter().map(|(_, hit)| hit).collect()))
    }

    /// [`Self::sorted_hits_within_async`] as `(key, hit)` pairs, read from `files` only. With
    /// `after`, it is a walk's next window, which reads no row twice:
    ///  - each file reads its first `k` rows from the cursor's key on, plus the rows of that key it
    ///    gave already, so at least `k` are new.
    ///  - candidates past the merged `k`-th are dropped before their `_id`s are read.
    async fn sorted_keyed_hits_async(
        &self,
        column: &str,
        descending: bool,
        k: usize,
        interval: Option<(u64, u64)>,
        after: Option<&Cursor>,
        files: &[Arc<SuperfileEntry>],
    ) -> Result<Option<Vec<(u64, SuperfileHit)>>, QueryError> {
        let manifest = self.manifest();
        let range_cols = manifest.table_schema().range_column_types();
        let Some((field, data_type)) = range_cols.get(column) else {
            return Err(QueryError::InvalidQuery(format!(
                "sorted_search: column '{column}' has no range index"
            )));
        };
        if KeyKind::of(data_type).is_none() {
            return Err(QueryError::InvalidQuery(format!(
                "sorted_search: column '{column}' is not of a range-indexed type"
            )));
        }
        if k == 0 {
            return Ok(Some(Vec::new()));
        }
        // More rows than the table holds cannot all have a value: the rows
        // without one would fill the page, so the indexes cannot answer.
        if interval.is_none() && u64::try_from(k).map_or(true, |k| k > self.rows_held()) {
            return Ok(None);
        }
        let mut bounds = interval.unwrap_or((0, u64::MAX));
        // The rows before the cursor's key were read already.
        if let Some(after) = after {
            match descending {
                false => bounds.0 = after.key,
                true => bounds.1 = after.key,
            }
        }
        let field = *field;
        let units: Vec<_> = files
            .iter()
            .map(|e| {
                let read = after.map_or(0, |after| after.read_at_key(e.uri));
                (Arc::clone(e), k.saturating_add(read))
            })
            .collect();
        let data_type = data_type.clone();
        let body = move |r: Arc<SuperfileReader>,
                         entry: Arc<SuperfileEntry>,
                         tombstone_cache: Option<Arc<SidecarCache>>,
                         now: Instant,
                         take: usize| {
            let data_type = data_type.clone();
            async move {
                let deleted = match &tombstone_cache {
                    Some(cache) => tombstone_deny_set(cache, entry.superfile_id, now)?,
                    None => None,
                };
                let none = RoaringBitmap::new();
                let ends = r
                    .range_ends(
                        field,
                        &data_type,
                        bounds,
                        take,
                        descending,
                        deleted.as_deref().unwrap_or(&none),
                    )
                    .await
                    .map_err(QueryError::from)?;
                let Some(ends) = ends.filter(|ends| ends.ties_in_id_order) else {
                    return Ok::<Option<Candidates>, QueryError>(None);
                };
                let keys: Vec<u64> = ends.pairs.iter().map(|(key, _)| *key).collect();
                let hits = tag_hits(
                    &entry,
                    ends.pairs.iter().map(|(_, row)| (*row, 0.0)).collect(),
                );
                Ok(Some(keys.into_iter().zip(hits).collect()))
            }
        };
        let per_superfile = dispatch::fanout_with(self, units, true, self.open_intent(), body)
            .instrument(detail_span!("sorted.fanout"))
            .await?;
        let Some(candidates) = per_superfile
            .into_iter()
            .collect::<Option<Vec<Candidates>>>()
        else {
            return Ok(None);
        };
        let mut candidates: Candidates = candidates.into_iter().flatten().collect();
        if candidates.len() < k && interval.is_none() {
            return Ok(None);
        }
        // At most the rows the cursor counts lie at or before it, so the page ends by the merged
        // `reach`-th key; candidates past it are dropped before their `_id`s are read.
        let reach = k.saturating_add(after.map_or(0, Cursor::read));
        if candidates.len() > reach {
            let mut keys: Vec<u64> = candidates.iter().map(|(key, _)| *key).collect();
            let cut = match descending {
                false => *keys.select_nth_unstable(reach - 1).1,
                true => *keys.select_nth_unstable_by(reach - 1, |a, b| b.cmp(a)).1,
            };
            candidates.retain(|(key, _)| match descending {
                false => *key <= cut,
                true => *key >= cut,
            });
        }
        Ok(Some(
            by_key_then_id(self, candidates, descending, k, |key, hit| {
                after.is_none_or(|after| after.passed_by(key, hit.stable_id, descending))
            })
            .await?,
        ))
    }

    /// The rows this snapshot's superfiles hold, deleted ones included: no
    /// sort can return more.
    fn rows_held(&self) -> u64 {
        self.manifest().superfiles.iter().map(|e| e.n_docs).sum()
    }

    /// How `filter` reads in `column`'s order: its literal comparisons on the column, `AND`ed, give
    /// the keys to read, and the rest is checked row by row. `None` with no range index on the
    /// column.
    fn sorted_filter(
        &self,
        column: &str,
        filter: &Expr,
    ) -> Result<Option<SortedFilter>, QueryError> {
        // Cast each literal as the statement does, then simplify (a literal that cannot cast
        // errors): on a `Float32` column, `f != 16777217 AND f = 16777216` keeps no row. Casts go
        // to the stored types of this reader's snapshot, as a statement's `Utf8View` compares as
        // the stored string does and a later schema change does not move them.
        let schema = Arc::new(
            DFSchema::try_from(self.snapshot_sql_schemas().scalar().as_ref().clone())
                .map_err(QueryError::DataFusion)?,
        );
        let simplifier = ExprSimplifier::new(
            SimplifyContext::builder()
                .with_schema(Arc::clone(&schema))
                .build(),
        );
        let Ok(coerced) = filter
            .clone()
            .rewrite(&mut TypeCoercionRewriter::new(&schema))
            .map(|t| t.data)
        else {
            return Ok(None);
        };
        let filter = simplifier
            .simplify(coerced)
            .map_err(|e| QueryError::InvalidQuery(format!("top-k walk: {e}")))?;
        let column = column.to_string();
        let range_cols = self.manifest().table_schema().range_column_types();
        if !range_cols.contains_key(&column) {
            return Ok(None);
        }
        let mut bounds = Vec::new();
        let mut rest = Vec::new();
        let mut keeps_none = false;
        for conjunct in split_conjunction(&filter) {
            match conjunct {
                Expr::Literal(ScalarValue::Boolean(Some(true)), _) => {}
                Expr::Literal(value, _)
                    if value.is_null() || *value == ScalarValue::Boolean(Some(false)) =>
                {
                    keeps_none = true;
                }
                _ if single_range(from_ref(conjunct), &column, &range_cols).is_some() => {
                    bounds.push(conjunct.clone());
                }
                _ => rest.push(conjunct.clone()),
            }
        }
        let interval = match single_range(&bounds, &column, &range_cols) {
            Some((_, _, lo, hi)) => (lo, hi),
            None if keeps_none => (0, u64::MAX),
            None => return Ok(None),
        };
        Ok(Some(SortedFilter {
            column,
            interval,
            bounded: !bounds.is_empty(),
            keeps_none,
            rest: conjunction(rest),
        }))
    }

    /// The share of rows `condition` is expected to keep, from the files' column statistics:
    ///  - `column = literal`: the rows a file's exact value table counts for it, else one row per
    ///    distinct value, as the scan estimates for the planner.
    ///  - `IN`: one such share per value. `AND`: the product of its sides.
    ///  - `text_match`: the rows each file's term counts allow ([`Self::text_match_share`]).
    ///  - anything else, or a column with no statistics: every row; the walk learns the share as it
    ///    reads.
    async fn expected_share(&self, condition: &Expr, files: &[Arc<SuperfileEntry>]) -> Share {
        let mut share = 1.0;
        let mut none = false;
        for conjunct in split_conjunction(condition) {
            let one = match self.text_match_share(conjunct, files).await {
                Some(one) => one,
                None => self.conjunct_share(conjunct, files),
            };
            share *= one.share;
            none |= one.none;
        }
        Share {
            share: share.clamp(0.0, 1.0),
            none,
        }
    }

    /// The rows of each of `files` a `text_match` can keep, from the term index's count of the
    /// rows a file holds a token in: the rarest token's count for `all`, the sum for `any`.
    /// A file the index does not list yet, or a token it holds with no count, counts every row.
    /// `None` when the index cannot answer: no term index, field or token, another tokenizer, or a
    /// failed read.
    async fn text_match_counts(
        &self,
        conjunct: &Expr,
        files: &[Arc<SuperfileEntry>],
    ) -> Option<Vec<u64>> {
        let (column, query, mode) = text_match_call(conjunct)?;
        let tokens = query.tokens();
        let manifest = self.manifest();
        if tokens.is_empty()
            || manifest.try_fts_tokenizer_for(&column)?.name() != STANDARD_TOKENIZER
        {
            return None;
        }
        let field = manifest.field_id(&column)?;
        // A failed index read leaves no estimate; the statement's own read then reports it.
        let index = manifest.term_index().await.ok()??;
        let terms: Vec<&str> = tokens.iter().map(String::as_str).collect();
        // The same runs the statement's own `text_match` reads, which the index then keeps.
        let located = index.locations(field, &terms, files).await.ok()?;
        let mut counts = Vec::with_capacity(files.len());
        for entry in files {
            if !index.is_indexed(&entry.superfile_id) {
                counts.push(entry.n_docs);
                continue;
            }
            let file_terms = located
                .get(&entry.superfile_id)
                .map_or(&[][..], Vec::as_slice);
            // A token the file does not hold counts 0; one held with no count may be in every row.
            let df = |term: &str| match file_terms.iter().find(|(listed, _, _)| listed == term) {
                Some((_, 0, _)) => entry.n_docs,
                Some((_, df, _)) => *df,
                None => 0,
            };
            // A phrase holds no more rows than its rarest term.
            let dfs = query
                .words
                .iter()
                .map(|word| df(word))
                .chain(query.phrases.iter().map(|phrase| {
                    phrase
                        .terms
                        .iter()
                        .map(|term| df(term))
                        .fold(u64::MAX, u64::min)
                }));
            let count = match mode {
                TextMatchMode::All => dfs.fold(u64::MAX, u64::min),
                TextMatchMode::Any => dfs.fold(0, u64::saturating_add),
            };
            counts.push(count.min(entry.n_docs));
        }
        Some(counts)
    }

    /// The share of rows a `text_match` can keep over `files` ([`Self::text_match_counts`]).
    async fn text_match_share(
        &self,
        conjunct: &Expr,
        files: &[Arc<SuperfileEntry>],
    ) -> Option<Share> {
        let matched: u64 = self.text_match_counts(conjunct, files).await?.iter().sum();
        let rows: u64 = files.iter().map(|entry| entry.n_docs).sum();
        Some(Share {
            share: matched as f64 / rows.max(1) as f64,
            none: matched == 0,
        })
    }

    /// The `files` that can hold a row `rest` keeps: a file where the term index counts no row for
    /// one of its `text_match`es holds none, so the walk leaves it out.
    async fn files_holding(
        &self,
        rest: &Expr,
        files: &[Arc<SuperfileEntry>],
    ) -> Vec<Arc<SuperfileEntry>> {
        let mut holding = files.to_vec();
        for conjunct in split_conjunction(rest) {
            if let Some(counts) = self.text_match_counts(conjunct, &holding).await {
                let mut counts = counts.into_iter();
                holding.retain(|_| counts.next().is_some_and(|count| count > 0));
            }
        }
        holding
    }

    /// [`Self::expected_share`] of one conjunct over `files`, the files the walk reads. A value
    /// table counts every row a file holds, deleted ones too, so a literal missing from it is in no
    /// live row there.
    fn conjunct_share(&self, conjunct: &Expr, files: &[Arc<SuperfileEntry>]) -> Share {
        let unknown = Share {
            share: 1.0,
            none: false,
        };
        let schemas = self.snapshot_sql_schemas();
        let (column, literals): (String, Vec<ScalarValue>) = match conjunct {
            // An equality, or the `OR` of equalities the simplifier makes of a short `IN`.
            Expr::BinaryExpr(_) => match flatten_or_eq(conjunct, schemas.scalar()) {
                Some(found) => found,
                None => return unknown,
            },
            Expr::InList(list) if !list.negated => match list.expr.as_ref() {
                Expr::Column(c) => {
                    let mut values = Vec::with_capacity(list.list.len());
                    for v in &list.list {
                        let Expr::Literal(v, _) = v else {
                            return unknown;
                        };
                        values.push(v.clone());
                    }
                    (c.name.clone(), values)
                }
                _ => return unknown,
            },
            _ => return unknown,
        };
        if literals.iter().any(|v| v.is_null()) {
            return unknown;
        }
        // The column in its stored type: the scan's `Utf8View` compares as the stored string does.
        let Ok(column_type) = schemas
            .scalar()
            .field_with_name(&column)
            .map(|f| f.data_type())
        else {
            return unknown;
        };
        let manifest = self.manifest();
        let Some(field) = manifest.table_schema().id_of(&column) else {
            return unknown;
        };
        // Statistics from a type the column has since left no longer compare as the statement does.
        let guard = ColumnTypeGuard::new(manifest, field);
        // The walk's files, not the snapshot's list, which a lazily opened table holds in part.
        let entries = files;
        // `None` when a statement would cast the column to compare it with a literal (a string
        // column against a number): then no table counts the literals and each file falls back.
        let exact = ExactLiterals::new(&literals, column_type);
        // `IN (5, 5)` asks for 5 once: counted after the cast when the literals cast, else as
        // written.
        let asked = exact.as_ref().map_or_else(
            || literals.iter().collect::<HashSet<_>>().len(),
            ExactLiterals::distinct_count,
        );
        if entries.iter().any(|entry| guard.stats_are_stale(entry)) {
            return unknown;
        }
        // Rows the value tables count for the literals, and the rows with a value in files whose
        // tables do not count them.
        let mut counted = 0.0;
        let mut uncounted = 0.0;
        let mut rows = 0.0;
        for entry in entries.iter() {
            rows += entry.n_docs as f64;
            let Some(stats) = entry.scalar_stats.column(field) else {
                return unknown;
            };
            if stats.all_null() {
                continue;
            }
            match exact.as_ref().and_then(|exact| stats.literal_count(exact)) {
                Some(held) => counted += held as f64,
                None => {
                    uncounted +=
                        entry.n_docs.saturating_sub(stats.null_count().unwrap_or(0)) as f64;
                }
            }
        }
        if rows == 0.0 {
            return unknown;
        }
        let mut expected = counted;
        if uncounted > 0.0 {
            // One row per distinct value per literal, from the estimate the scan gives the planner.
            let Some(distinct) = scalar_distinct(&read_column(entries, field)) else {
                return unknown;
            };
            if distinct >= 1 {
                expected += uncounted * (asked as f64 / distinct as f64).min(1.0);
            }
        }
        Share {
            share: expected / rows,
            none: uncounted == 0.0 && expected == 0.0,
        }
    }

    /// Which of `hits` `condition` keeps, evaluated over the hits' rows (only the columns it
    /// names). The rows are reserved against the connection's memory budget; over it, the query is
    /// refused.
    ///  - `None` when it cannot run outside a statement (a subquery, a statement-only function).
    ///  - `None` when it names a column the table does not store.
    async fn kept_by(
        &self,
        hits: &[SuperfileHit],
        condition: &Expr,
    ) -> Result<Option<Vec<bool>>, QueryError> {
        let stored = self.manifest().stored_schema();
        let mut names: Vec<String> = condition
            .column_refs()
            .into_iter()
            .map(|c| c.name.clone())
            .collect();
        names.sort();
        names.dedup();
        if names.is_empty() || names.iter().any(|n| stored.field_with_name(n).is_err()) {
            return Ok(None);
        }
        let projection: Vec<&str> = names.iter().map(String::as_str).collect();
        let rows = resolve_hits_named(self, hits, Some(&projection)).await?;
        let _held = self
            .manifest()
            .options
            .connection_memory_budget
            .try_reserve(rows.get_array_memory_size())
            .map_err(|over| {
                QueryError::OverBudget(format!(
                    "while the top-k walk checks a window of rows, {over}"
                ))
            })?;
        let Ok(schema) = DFSchema::try_from(rows.schema().as_ref().clone()) else {
            return Ok(None);
        };
        let Ok(coerced) = condition
            .clone()
            .rewrite(&mut TypeCoercionRewriter::new(&schema))
            .map(|t| t.data)
        else {
            return Ok(None);
        };
        let Ok(predicate) = create_physical_expr(&coerced, &schema, &ExecutionProps::new()) else {
            return Ok(None);
        };
        // Up to a window of rows: run it on the reader pool, and count its time as the query's.
        let (kept, ns) = run_on_pool(
            Some(&self.manifest().options.reader_pool),
            "sorted search condition",
            move || {
                timed_section(|| {
                    let kept = predicate
                        .evaluate(&rows)
                        .and_then(|v| v.into_array(rows.num_rows()))
                        .ok()?;
                    let kept = kept.as_boolean_opt()?;
                    // A row the condition leaves unknown (nulls) is not kept, as in a WHERE.
                    Some(
                        (0..kept.len())
                            .map(|i| kept.is_valid(i) && kept.value(i))
                            .collect::<Vec<bool>>(),
                    )
                })
            },
        )
        .await
        .map_err(QueryError::internal)?;
        if let Some(stats) = &self.op_stats {
            stats.add_kernel_cpu_ns(ns);
        }
        Ok(kept)
    }

    /// The first `k` rows `filter` keeps in its column's order: the interval read from the range
    /// indexes a window at a time, each from where the last ended, checked against the rest.
    /// `None`:
    ///  - when the indexes cannot answer, or the rest cannot run outside a statement.
    ///  - when [`MAX_WALKED_ROWS`] rows do not find `k`.
    ///  - when nothing bounds the column and its rows with a value run out (nulls sort after,
    ///    unread).
    async fn sorted_hits_where_async(
        &self,
        filter: &SortedFilter,
        descending: bool,
        k: usize,
        files: &[Arc<SuperfileEntry>],
    ) -> Result<Option<Vec<SuperfileHit>>, QueryError> {
        if filter.keeps_none {
            return Ok(Some(Vec::new()));
        }
        let Some(rest) = &filter.rest else {
            if !filter.bounded {
                return self
                    .sorted_hits_within_async(&filter.column, descending, k, None, files)
                    .await;
            }
            return self
                .sorted_hits_within_async(
                    &filter.column,
                    descending,
                    k,
                    Some(filter.interval),
                    files,
                )
                .await;
        };
        if k == 0 {
            return Ok(Some(Vec::new()));
        }
        // Rows to read for `rows` more to pass at a share; a walk expected to pass its limit leaves
        // the rows to the statement before reading them.
        let needed = |share: f64, rows: usize| rows as f64 / share;
        let window_for = |share: f64, rows: usize| {
            ((needed(share, rows) * WINDOW_MARGIN).ceil() as usize).max(rows)
        };
        let holding = self.files_holding(rest, files).await;
        let files = holding.as_slice();
        let Share { share, none } = self.expected_share(rest, files).await;
        if none {
            // The statistics count every value the filter names, and no row holds one: it keeps
            // none.
            return Ok(Some(Vec::new()));
        }
        // No window needs more than the rows the files hold, plus one so reading them all ends it.
        // Under a bounded filter that end answers, so a walk reads no more rows than are left, and
        // one over a few small files starts.
        let rows_held = files.iter().map(|entry| entry.n_docs).sum::<u64>();
        let rows_left = |walked: usize| {
            (rows_held as usize)
                .saturating_sub(walked)
                .saturating_add(1)
        };
        let reads = |rows: f64, walked: usize| match filter.bounded {
            true => rows.min(rows_left(walked) as f64),
            false => rows,
        };
        if share <= 0.0 || reads(needed(share, k), 0) > MAX_WALKED_ROWS as f64 {
            return Ok(None);
        }
        let mut window = window_for(share, k).min(rows_left(0));
        let mut page = Vec::with_capacity(k);
        let mut cursor: Option<Cursor> = None;
        let mut walked = 0usize;
        let mut passed = 0usize;
        loop {
            // The first window reads its full margin; later ones stop at the limit.
            let window_now = match cursor {
                None => window,
                Some(_) => window.min(MAX_WALKED_ROWS - walked),
            };
            // Each superfile reads a window from the cursor's key on, plus the rows of that key it
            // gave already.
            let entries: usize = files
                .iter()
                .map(|e| window_now + cursor.as_ref().map_or(0, |c| c.read_at_key(e.uri)))
                .sum();
            let budget = &self.manifest().options.connection_memory_budget;
            if budget.remaining().is_none() && entries > MAX_WINDOW_ENTRIES {
                return Ok(None);
            }
            let _window = budget
                .try_reserve(entries.saturating_mul(size_of::<(u64, SuperfileHit)>()))
                .map_err(|over| {
                    QueryError::OverBudget(format!(
                        "while the top-k walk reads a window of the range index, {over}"
                    ))
                })?;
            let Some(read) = self
                .sorted_keyed_hits_async(
                    &filter.column,
                    descending,
                    window_now,
                    Some(filter.interval),
                    cursor.as_ref(),
                    files,
                )
                .await?
            else {
                return Ok(None);
            };
            #[cfg(test)]
            walk_probe::window_read(&self.manifest().superfiles, read.len());
            let ended = read.len() < window_now;
            let hits: Vec<SuperfileHit> = read.iter().map(|(_, hit)| *hit).collect();
            let kept = match hits.is_empty() {
                true => Vec::new(),
                false => match self.kept_by(&hits, rest).await? {
                    Some(kept) => kept,
                    None => return Ok(None),
                },
            };
            passed += kept.iter().filter(|kept| **kept).count();
            walked += hits.len();
            let room = k - page.len();
            page.extend(
                hits.iter()
                    .zip(kept.iter().copied())
                    .filter_map(|(hit, kept)| kept.then_some(*hit))
                    .take(room),
            );
            if page.len() == k || (ended && filter.bounded) {
                return Ok(Some(page));
            }
            if ended || walked >= MAX_WALKED_ROWS {
                return Ok(None);
            }
            cursor = Cursor::past(cursor, &read);
            // Size the next window from the share kept so far; past the limit, the statement is
            // faster. With nothing kept yet, grow it, unless walked and grown rows pass the limit.
            let rows = k - page.len();
            window = match passed {
                0 => {
                    let grown = window_now.saturating_mul(WINDOW_GROWTH);
                    if walked as f64 + reads(grown as f64, walked) > MAX_WALKED_ROWS as f64 {
                        return Ok(None);
                    }
                    grown
                }
                _ => {
                    let seen = passed as f64 / walked as f64;
                    if walked as f64 + reads(needed(seen, rows), walked) > MAX_WALKED_ROWS as f64 {
                        return Ok(None);
                    }
                    window_for(seen, rows)
                }
            };
        }
    }

    /// The first `k` rows `predicates` keep in `column`'s order, ties by `_id` ascending, walked
    /// over the files a scan kept. `None`, and the scan reads every row, when the walk cannot
    /// answer, or when rows with no value sort first and no filter on `column` keeps them out.
    pub(crate) async fn sorted_hits_for_scan(
        &self,
        column: &str,
        predicates: &[Expr],
        descending: bool,
        nulls_first: bool,
        k: usize,
        files: &[Arc<SuperfileEntry>],
    ) -> Result<Option<Vec<SuperfileHit>>, QueryError> {
        // A filter the walk cannot read leaves the rows to the scan.
        let filter = conjunction(predicates.to_vec())
            .and_then(|filter| self.sorted_filter(column, &filter).ok().flatten())
            .filter(|filter| filter.bounded || !nulls_first);
        match filter {
            Some(filter) => {
                self.sorted_hits_where_async(&filter, descending, k, files)
                    .await
            }
            None => Ok(None),
        }
    }

    /// Sync [`Self::sorted_hits_async`] through the sync→async bridge.
    pub(crate) fn sorted_hits(
        &self,
        column: &str,
        descending: bool,
        k: usize,
    ) -> Result<Option<Vec<SuperfileHit>>, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(self.sorted_hits_async(column, descending, k))
    }
}

impl Supertable {
    /// The first `k` rows by `column`, a range-indexed column, ascending
    /// or (with `descending`) descending, rows sharing a value ordered by
    /// `_id` ascending, and rows without a value last: what `SELECT … ORDER
    /// BY column, _id LIMIT k` answers, without planning a statement. Read
    /// from the column's range index, a block or two per superfile, however
    /// large the table. Returns Arrow rows in that order like
    /// [`Supertable::bm25_search`], with `score` fixed at `0.0`;
    /// `projection` follows the same rules.
    pub fn sorted_search(
        &self,
        column: &str,
        descending: bool,
        k: usize,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(column, descending, k, "sorted_search");
        let close_out = self.close_out();
        let reader = self.reader()?;
        let context = |e: QueryError| InfinoError::from(e).with_context("sorted_search", None);
        reader.check_projection(projection).map_err(context)?;
        // A page past every row is every row: the statement's limit stays
        // within what the table holds, whatever `k` the caller passed.
        let k = k.min(usize::try_from(reader.rows_held()).unwrap_or(usize::MAX));
        let batches = match reader.sorted_hits(column, descending, k).map_err(context)? {
            Some(hits) => vec![
                self.block_on_query(
                    resolve_hits_named(&reader, &hits, projection)
                        .instrument(detail_span!("search.resolve", hits = hits.len())),
                )
                .map_err(context)?,
            ],
            None => reader
                .run_sql(&sorted_statement(
                    SQL_TABLE,
                    column,
                    descending,
                    k,
                    projection,
                    &reader.options().id_column,
                ))
                .map_err(context)?,
        };
        close_out.finish_batches(&batches);
        Ok(batches)
    }
}

/// The SQL over `table` that answers [`Supertable::sorted_search`] where the
/// range indexes cannot (and over the wire, where a hosted table has no
/// sorted call): the same order and limit, with the same output columns
/// (`_id` and `score` when `projection` names none, `score` as `0.0`).
pub(crate) fn sorted_statement(
    table: &str,
    column: &str,
    descending: bool,
    k: usize,
    projection: Option<&[&str]>,
    id_column: &str,
) -> String {
    let default = [id_column, SCORE_COLUMN];
    let select: Vec<String> = projection
        .unwrap_or(&default)
        .iter()
        .map(|name| match *name {
            SCORE_COLUMN => format!("CAST(0 AS FLOAT) AS {}", quote(SCORE_COLUMN)),
            name => quote(name),
        })
        .collect();
    let direction = if descending { "DESC" } else { "ASC" };
    format!(
        "SELECT {} FROM {} ORDER BY {} {direction} NULLS LAST, {} ASC LIMIT {k}",
        select.join(", "),
        quote(table),
        quote(column),
        quote(id_column),
    )
}

/// `name` as a quoted SQL identifier.
pub(crate) fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

#[cfg(test)]
mod tests {
    use std::{cmp::Ordering, iter::repeat_n, sync::Arc};

    use arrow::compute::cast;
    use arrow_array::{
        Array, Decimal128Array, Float32Array, Float64Array, Int64Array, StringArray,
        TimestampMillisecondArray,
    };
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use datafusion::prelude::{col, lit};
    use tempfile::TempDir;

    use super::*;
    use crate::{
        config::OptimizeOptions,
        memory::ConnectionMemoryBudget,
        runtime_metrics::op_stats::with_op_stats,
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{
            SupertableOptions,
            schema::change::{FieldPatch, SchemaPatch},
        },
    };

    /// [`SupertableReader::sorted_hits_for_scan`] over every file of `st`, as rows. A `None`
    /// `column` is the one column `filter` names.
    fn sorted_search_where(
        st: &Supertable,
        column: Option<&str>,
        filter: &Expr,
        descending: bool,
        k: usize,
        projection: Option<&[&str]>,
    ) -> Result<Option<Vec<RecordBatch>>, InfinoError> {
        let reader = st.reader()?;
        let named = filter.column_refs();
        let column = match column {
            Some(column) => column.to_owned(),
            None => {
                let mut names = named.iter().map(|c| c.name.as_str());
                match (names.next(), names.next()) {
                    (Some(name), None) => name.to_owned(),
                    _ => return Ok(None),
                }
            }
        };
        let predicates: Vec<Expr> = split_conjunction(filter).into_iter().cloned().collect();
        let files = &reader.manifest().superfiles;
        let walked = reader.sorted_hits_for_scan(&column, &predicates, descending, false, k, files);
        let Some(hits) = reader.block_on(walked).map_err(InfinoError::from)? else {
            return Ok(None);
        };
        let rows = reader
            .block_on(resolve_hits_named(&reader, &hits, projection))
            .map_err(InfinoError::from)?;
        Ok(Some(vec![rows]))
    }

    /// Superfiles the table is written as, one commit each.
    const FILES: i64 = 3;
    /// Rows in each.
    const ROWS_PER_FILE: i64 = 3000;
    /// Distinct `key` values, so every cutoff ties on many rows.
    const KEYS: i64 = 50;
    /// Every this-many rows `key` is null.
    const NULL_EVERY: i64 = 17;
    /// Rows the first delete removes: those with `seq` below this.
    const DELETED_BELOW: i64 = 200;
    /// Rows the second delete removes: those with `seq` from this on, a
    /// larger share of the last superfile than the first takes of the first,
    /// so an optimize merges the last superfile first.
    const LATE_DELETED_FROM: i64 = 8_000;
    /// The `(descending, k)` each pass asks for; the last is past the rows
    /// with a value, so the nulls come in and the index cannot answer.
    const ASKS: &[(bool, usize)] = &[
        (false, 1),
        (false, 10),
        (true, 10),
        (true, 125),
        (false, 600),
        (true, 7_700),
    ];

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Int64, true),
        ]))
    }

    /// `key` of row `seq`: not in the order the rows arrive, null now and
    /// then.
    fn key(seq: i64) -> Option<i64> {
        (seq % NULL_EVERY != 0).then_some((seq * 31) % KEYS)
    }

    /// The table: `key` range indexed, written in [`FILES`] commits, its
    /// rows below [`DELETED_BELOW`] and from [`LATE_DELETED_FROM`] deleted.
    fn table(dir: &TempDir) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(schema(), vec![], vec![])
            .expect("options")
            .with_storage(storage)
            .with_range_indexes(&["key".to_string()]);
        let st = Supertable::create(options).expect("create");
        for file in 0..FILES {
            let seqs: Vec<i64> = (file * ROWS_PER_FILE..(file + 1) * ROWS_PER_FILE).collect();
            let keys: Vec<Option<i64>> = seqs.iter().map(|s| key(*s)).collect();
            let batch = RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(Int64Array::from(keys)),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st.delete(col("seq").lt(lit(DELETED_BELOW)))
            .expect("delete");
        st.delete(col("seq").gt_eq(lit(LATE_DELETED_FROM)))
            .expect("delete");
        st
    }

    /// One `Int64` column of `batches`, by name, across them in order.
    fn ints(batches: &[RecordBatch], name: &str) -> Vec<Option<i64>> {
        batches
            .iter()
            .flat_map(|b| {
                let column = b
                    .column_by_name(name)
                    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                    .expect(name)
                    .clone();
                (0..column.len()).map(move |i| column.is_valid(i).then(|| column.value(i)))
            })
            .collect()
    }

    /// The `_id` column of `batches`, in order.
    fn ids(batches: &[RecordBatch]) -> Vec<i128> {
        batches
            .iter()
            .flat_map(|b| {
                let ids = b
                    .column_by_name("_id")
                    .and_then(|c| c.as_any().downcast_ref::<Decimal128Array>())
                    .expect("_id")
                    .clone();
                (0..ids.len()).map(move |i| ids.value(i))
            })
            .collect()
    }

    #[test]
    fn sorted_search_where_answers_as_the_filtered_sql_sort() {
        // One interval of `key`, every ask, before and after optimize.
        //  - Each page matches the filtered SQL sort, row for row, read from the range indexes.
        //  - An interval holds no null, so a short page is the whole answer.
        //  - With no column named, the interval's own column sorts.
        // This test pins the walk's answer for a plain interval.
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        let intervals: [(Expr, &str); 5] = [
            (
                col("key")
                    .gt_eq(lit(10_i64))
                    .and(col("key").lt(lit(20_i64))),
                "key >= 10 AND key < 20",
            ),
            (col("key").gt(lit(45_i64)), "key > 45"),
            (col("key").lt_eq(lit(3_i64)), "key <= 3"),
            (
                col("key").between(lit(7_i64), lit(7_i64)),
                "key BETWEEN 7 AND 7",
            ),
            (col("key").gt(lit(1_000_i64)), "key > 1000"),
        ];
        for pass in ["appended", "optimized"] {
            if pass == "optimized" {
                st.optimize(&OptimizeOptions::default()).expect("optimize");
            }
            let reader = st.reader().expect("reader");
            for (filter, sql) in &intervals {
                for &(descending, k) in ASKS {
                    for column in [Some("key"), None] {
                        let got = sorted_search_where(
                            &st,
                            column,
                            filter,
                            descending,
                            k,
                            Some(&["_id", "seq", "key"]),
                        )
                        .expect("sorted_search_where")
                        .unwrap_or_else(|| panic!("{pass}: {sql}: the index answers"));
                        let direction = if descending { "DESC" } else { "ASC" };
                        let want = reader
                            .query_sql(&format!(
                                "SELECT _id, seq, key FROM supertable WHERE {sql} ORDER BY key {direction} NULLS LAST, _id ASC LIMIT {k}"
                            ))
                            .expect("sql");
                        let what = format!("{pass}: {sql}, descending {descending}, k {k}");
                        assert_eq!(ids(&got), ids(&want), "{what}");
                        assert_eq!(ints(&got, "seq"), ints(&want, "seq"), "{what}");
                        assert_eq!(ints(&got, "key"), ints(&want, "key"), "{what}");
                    }
                }
            }
        }
    }

    /// The asks the walk answers for a filter keeping a third of any stretch of `key`'s order: all
    /// but the one past every row with a value.
    const DENSE_ASKS: &[(bool, usize)] = &[
        (false, 1),
        (false, 10),
        (true, 10),
        (true, 125),
        (false, 600),
    ];

    #[test]
    fn sorted_search_where_checks_the_other_conditions_on_the_rows_in_order() {
        // Filters beside the interval, on other columns or the sorted one, before and after
        // optimize.
        //  - Every page the walk answers matches the filtered SQL sort, row for row.
        //  - Each filter's asks within the walk's reach are answered.
        //  - With nothing bounding the column, a walk past its rows with a value leaves the nulls
        //    to SQL.
        // This test pins that the rest of the filter is checked on the rows in order.
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        // Each filter with the asks the walk must answer:
        //  - the first three keep a third of any stretch of the order, so pages up to 600 fit the
        //    walk.
        //  - keys 3 and 5 lie a few hundred rows into the ascending order, past the limit
        //    descending.
        //  - `seq IN (…)` is expected to keep 3 rows, so only a page of one is walked for.
        //  - `seq < 0` keeps no row, and with no bound on `key` the nulls stay unread: never
        //    answered.
        let filters: [(Expr, &str, &[(bool, usize)]); 6] = [
            (
                col("key")
                    .gt_eq(lit(10_i64))
                    .and((col("seq") % lit(3_i64)).eq(lit(0_i64))),
                "key >= 10 AND seq % 3 = 0",
                DENSE_ASKS,
            ),
            (
                (col("seq") % lit(2_i64)).eq(lit(1_i64)),
                "seq % 2 = 1",
                DENSE_ASKS,
            ),
            (
                col("key").lt(lit(25_i64)).and(
                    col("seq")
                        .gt(lit(4_000_i64))
                        .or(col("seq").lt(lit(1_000_i64))),
                ),
                "key < 25 AND (seq > 4000 OR seq < 1000)",
                DENSE_ASKS,
            ),
            (
                col("key").eq(lit(3_i64)).or(col("key").eq(lit(5_i64))),
                "(key = 3 OR key = 5)",
                &[(false, 1), (false, 10)],
            ),
            (
                col("key").gt(lit(40_i64)).and(
                    col("seq").in_list(vec![lit(301_i64), lit(302_i64), lit(4_444_i64)], false),
                ),
                "key > 40 AND seq IN (301, 302, 4444)",
                &[(false, 1)],
            ),
            (col("seq").lt(lit(0_i64)), "seq < 0", &[]),
        ];
        for pass in ["appended", "optimized"] {
            if pass == "optimized" {
                st.optimize(&OptimizeOptions::default()).expect("optimize");
            }
            let reader = st.reader().expect("reader");
            for (filter, sql, answered) in &filters {
                for &(descending, k) in ASKS {
                    let got = sorted_search_where(
                        &st,
                        Some("key"),
                        filter,
                        descending,
                        k,
                        Some(&["_id", "seq", "key"]),
                    )
                    .expect("sorted_search_where");
                    let what = format!("{pass}: {sql}, descending {descending}, k {k}");
                    let must = answered.contains(&(descending, k));
                    if *sql == "seq < 0" {
                        assert!(got.is_none(), "{what}: left to the statement");
                    }
                    let Some(got) = got else {
                        assert!(!must, "{what}: answered by the walk");
                        continue;
                    };
                    let direction = if descending { "DESC" } else { "ASC" };
                    let want = reader
                        .query_sql(&format!(
                            "SELECT _id, seq, key FROM supertable WHERE {sql} ORDER BY key {direction} NULLS LAST, _id ASC LIMIT {k}"
                        ))
                        .expect("sql");
                    assert_eq!(ids(&got), ids(&want), "{what}");
                    assert_eq!(ints(&got, "seq"), ints(&want, "seq"), "{what}");
                    assert_eq!(ints(&got, "key"), ints(&want, "key"), "{what}");
                }
            }
        }
    }

    #[test]
    fn a_value_no_value_table_holds_keeps_no_row() {
        // A filter on a column whose files keep exact value tables.
        //  - A literal no table holds, alone or in an `IN`, keeps no row: empty page, no row read.
        //  - A literal a table holds is walked and matches the SQL sort.
        // This test pins that the value tables prove a filter empty.
        let dir = TempDir::new().expect("tempdir");
        let schema = Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("tag", DataType::Utf8, false),
        ]));
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(Arc::clone(&schema), vec![], vec![])
            .expect("options")
            .with_storage(storage)
            .with_range_indexes(&["seq".to_string()]);
        let st = Supertable::create(options).expect("create");
        for file in 0..2_i64 {
            let seqs: Vec<i64> = (file * 1000..(file + 1) * 1000).collect();
            let tags: Vec<String> = seqs.iter().map(|s| format!("t{}", s % 7)).collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(arrow_array::StringArray::from(tags)),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        let page = |filter: &Expr| {
            sorted_search_where(&st, Some("seq"), filter, true, 10, Some(&["_id", "seq"]))
                .expect("sorted_search_where")
        };
        for filter in [
            col("tag").eq(lit("absent")),
            col("tag").in_list(vec![lit("absent"), lit("missing")], false),
        ] {
            let got = page(&filter).unwrap_or_else(|| panic!("{filter}: answered"));
            assert_eq!(
                got.iter().map(RecordBatch::num_rows).sum::<usize>(),
                0,
                "{filter}"
            );
        }
        let held = page(&col("tag").eq(lit("t3"))).expect("a held value is walked");
        let reader = st.reader().expect("reader");
        let want = reader
            .query_sql(
                "SELECT _id, seq FROM supertable WHERE tag = 't3' ORDER BY seq DESC NULLS LAST, _id ASC LIMIT 10",
            )
            .expect("sql");
        assert_eq!(ints(&held, "seq"), ints(&want, "seq"));
    }

    /// Distinct `n` values in the repeated-literal test's file: more than its value tables keep.
    const UNTABLED_VALUES: i64 = 20_000;
    /// An `n` value the repeated-literal test asks for; `ASKED + 1` is another one.
    const ASKED: i64 = 500;

    #[test]
    fn a_repeated_literal_is_counted_once() {
        // One file whose `n` holds more distinct values than its value tables keep, so the
        // estimate is one row per distinct value per literal.
        //  - `n IN (a, a)` expects the rows `n = a` does, not those of `n IN (a, a + 1)`.
        // This test pins that the walk's estimate asks for each literal once.
        let dir = TempDir::new().expect("tempdir");
        let schema = Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("n", DataType::Int64, false),
        ]));
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(Arc::clone(&schema), vec![], vec![])
            .expect("options")
            .with_storage(storage)
            .with_range_indexes(&["seq".to_string()]);
        let st = Supertable::create(options).expect("create");
        let values: Vec<i64> = (0..UNTABLED_VALUES).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(values.clone())),
                Arc::new(Int64Array::from(values)),
            ],
        )
        .expect("batch");
        let mut writer = st.writer().expect("writer");
        writer.append(&batch).expect("append");
        writer.commit().expect("commit");
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let field = manifest.table_schema().id_of("n").expect("n");
        let files = &manifest.superfiles;
        assert!(
            files.iter().all(|file| file
                .scalar_stats
                .column(field)
                .is_some_and(|stats| stats.value_counts().is_none())),
            "no file keeps a table for `n`"
        );
        let share = |filter: Expr| reader.conjunct_share(&filter, files).share;
        let one = share(col("n").eq(lit(ASKED)));
        assert!(one < 1.0, "the estimate counts rows: {one}");
        let twice = share(col("n").in_list(vec![lit(ASKED), lit(ASKED)], false));
        assert_eq!(twice, one);
        assert!(share(col("n").in_list(vec![lit(ASKED), lit(ASKED + 1)], false)) > one);
    }

    /// The numeric-tag table has 7 `tag` values, written with a leading zero (`"00"` to `"06"`).
    const NUMERIC_TAGS: i64 = 7;
    /// Every `n` is 2^53 + 1, which reads as 2^53 when cast to `Float64`.
    const BIG: i64 = (1 << 53) + 1;

    /// Range-indexed `seq`, a numeric `tag` string and `n` = [`BIG`], in two commits, so each file
    /// keeps exact value tables for `tag` and `n`.
    fn numeric_tag_table(dir: &TempDir) -> Supertable {
        let schema = Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("tag", DataType::Utf8, false),
            Field::new("n", DataType::Int64, false),
        ]));
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(Arc::clone(&schema), vec![], vec![])
            .expect("options")
            .with_storage(storage)
            .with_range_indexes(&["seq".to_string()]);
        let st = Supertable::create(options).expect("create");
        for file in 0..2_i64 {
            let seqs: Vec<i64> = (file * 1000..(file + 1) * 1000).collect();
            let tags: Vec<String> = seqs
                .iter()
                .map(|s| format!("{:02}", s % NUMERIC_TAGS))
                .collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(StringArray::from(tags)),
                    Arc::new(Int64Array::from(vec![BIG; 1000])),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st
    }

    /// The walk's `seq DESC LIMIT 10` page for `filter` must be answered and match the SQL sort of
    /// `sql`.
    fn assert_page_matches_sql(st: &Supertable, filter: &Expr, sql: &str) {
        let got = sorted_search_where(st, Some("seq"), filter, true, 10, Some(&["_id", "seq"]))
            .expect("sorted_search_where")
            .unwrap_or_else(|| panic!("{sql}: answered"));
        let want = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT _id, seq FROM supertable WHERE {sql} ORDER BY seq DESC NULLS LAST, _id ASC LIMIT 10"
            ))
            .expect("sql");
        assert!(!ints(&want, "seq").is_empty(), "{sql}: SQL keeps rows");
        assert_eq!(ints(&got, "seq"), ints(&want, "seq"), "{sql}");
    }

    #[test]
    fn a_literal_the_column_is_cast_for_is_not_counted() {
        // Literals the statement compares by casting the column.
        //  - `tag = 3` casts `"03"` to 3, and `n = 2^53` casts [`BIG`] to 2^53: both keep rows the
        //    tables hold under another value, so neither is looked up.
        //  - A literal of the column's own type is still counted: `tag = '09'` keeps no row.
        // This test pins that a cast literal is never read as absent.
        let dir = TempDir::new().expect("tempdir");
        let st = numeric_tag_table(&dir);
        assert_page_matches_sql(&st, &col("tag").eq(lit(3_i64)), "tag = 3");
        assert_page_matches_sql(
            &st,
            &col("n").eq(lit((1_i64 << 53) as f64)),
            "n = 9007199254740992.0",
        );
        let absent = sorted_search_where(
            &st,
            Some("seq"),
            &col("tag").eq(lit("09")),
            true,
            10,
            Some(&["_id", "seq"]),
        )
        .expect("sorted_search_where")
        .expect("a value no table holds is answered");
        assert_eq!(absent.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    }

    #[test]
    fn value_tables_in_a_type_the_column_left_are_not_counted() {
        // `tag` converted from strings to `Int64`, its value tables still counting the strings.
        // This test pins that `tag = 3` keeps the `"03"` rows, not finding no 3 in the old tables.
        let dir = TempDir::new().expect("tempdir");
        let st = numeric_tag_table(&dir);
        st.apply_schema(
            &SchemaPatch::new(vec![FieldPatch::named("tag").with_type(DataType::Int64)]),
            None,
        )
        .expect("convert tag");
        assert_page_matches_sql(&st, &col("tag").eq(lit(3_i64)), "tag = 3");
    }

    /// `seq` of the first row by `seq` that `filter` keeps, walked by `reader` over `files`.
    fn first_walked_seq(
        reader: &SupertableReader,
        filter: Expr,
        files: &[Arc<SuperfileEntry>],
    ) -> Vec<Option<i64>> {
        let predicates = vec![col("seq").gt_eq(lit(0_i64)), filter];
        let walked = reader.sorted_hits_for_scan("seq", &predicates, false, false, 1, files);
        let hits = reader
            .block_on(walked)
            .expect("walk")
            .expect("answered by the walk");
        let rows = reader
            .block_on(resolve_hits_named(reader, &hits, Some(&["seq"])))
            .expect("rows");
        ints(&[rows], "seq")
    }

    #[test]
    fn a_walk_casts_its_filter_in_its_snapshot_types() {
        // `tag` is `Int64` in the snapshot a reader pins, then converted to `Float32`.
        //  - In `Int64`, `tag != 2^24 + 1` keeps the row whose `tag` is 2^24.
        //  - In `Float32` the literal would round to 2^24 and drop that row.
        // This test pins that the pinned reader walks in its own snapshot's types.
        let dir = TempDir::new().expect("tempdir");
        let schema = Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("tag", DataType::Int64, false),
        ]));
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(Arc::clone(&schema), vec![], vec![])
            .expect("options")
            .with_storage(storage)
            .with_range_indexes(&["seq".to_string()]);
        let st = Supertable::create(options).expect("create");
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![0, 1])),
                Arc::new(Int64Array::from(vec![FLOAT32_EXACT, 5])),
            ],
        )
        .expect("batch");
        let mut writer = st.writer().expect("writer");
        writer.append(&batch).expect("append");
        writer.commit().expect("commit");
        drop(writer);
        let reader = st.reader().expect("reader");
        st.apply_schema(
            &SchemaPatch::new(vec![FieldPatch::named("tag").with_type(DataType::Float32)]),
            None,
        )
        .expect("convert tag");
        let files = reader.manifest().superfiles.clone();
        let filter = col("tag").not_eq(lit(FLOAT32_EXACT + 1));
        assert_eq!(first_walked_seq(&reader, filter, &files), vec![Some(0)]);
    }

    #[test]
    fn a_value_is_counted_over_the_files_walked_not_the_snapshot_list() {
        // A table opened lazily, then appended to: its snapshot's list holds only the new file.
        //  - The older file holds the one `r` row; the new file holds only `c`.
        //  - `tag = 'r'` over both files finds that row; the new file's table alone lacks `r`.
        // This test pins that the value tables of the files walked prove a value absent.
        let dir = TempDir::new().expect("tempdir");
        let schema = Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("tag", DataType::Utf8, false),
        ]));
        let options = || {
            let storage: Arc<dyn StorageProvider> =
                Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
            SupertableOptions::new(Arc::clone(&schema), vec![], vec![])
                .expect("options")
                .with_storage(storage)
                .with_range_indexes(&["seq".to_string()])
        };
        let append = |st: &Supertable, seq: i64, tag: &str| {
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(vec![seq])),
                    Arc::new(StringArray::from(vec![tag])),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        };
        let created = Supertable::create(options()).expect("create");
        append(&created, 0, "r");
        drop(created);
        let st = Supertable::open(options().with_eager_load_threshold(0)).expect("open lazily");
        append(&st, 1, "c");
        let reader = st.reader().expect("reader");
        assert_eq!(
            reader.manifest().superfiles.len(),
            1,
            "the lazily opened snapshot lists only the appended file"
        );
        let got = reader
            .query_sql(
                "SELECT seq FROM supertable WHERE seq >= 0 AND tag = 'r' ORDER BY seq LIMIT 1",
            )
            .expect("sql");
        assert_eq!(ints(&got, "seq"), vec![Some(0)]);
    }

    /// The CPU test's condition is 200 additions, so evaluating it dwarfs reading its rows.
    const COSTLY_ADDITIONS: usize = 200;
    /// It runs 50 times, as the per-thread CPU clock moves at scheduler events and one short run
    /// can read zero.
    const CPU_RUNS: usize = 50;
    /// Rows per run.
    const CPU_ROWS: usize = 2000;
    /// The checks must report at least half the condition's own CPU: the clock may lose some, never
    /// most.
    const CPU_SHARE_FLOOR: u64 = 2;

    #[test]
    #[cfg_attr(
        not(target_os = "linux"),
        ignore = "per-thread CPU clock is Linux procfs (schedstat); off Linux kernel_cpu_ns is always 0"
    )]
    fn a_condition_checked_on_rows_reports_its_cpu() {
        // A costly condition checked on a walk's rows, against the same condition evaluated alone.
        // This test pins that the checks report their CPU to the query, as a WHERE does.
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        let costly = (0..COSTLY_ADDITIONS)
            .fold(col("seq"), |sum, _| sum + lit(1_i64))
            .gt(lit(0_i64));
        let hits = st
            .reader()
            .expect("reader")
            .sorted_hits("key", false, CPU_ROWS)
            .expect("hits")
            .expect("answered");
        let reader = st.reader().expect("reader");
        let rows = reader
            .block_on(resolve_hits_named(&reader, &hits, Some(&["seq"])))
            .expect("rows");
        let schema = DFSchema::try_from(rows.schema().as_ref().clone()).expect("schema");
        let predicate =
            create_physical_expr(&costly, &schema, &ExecutionProps::new()).expect("predicate");
        let (((), alone), _) = with_op_stats(|| {
            timed_section(|| {
                for _ in 0..CPU_RUNS {
                    predicate.evaluate(&rows).expect("evaluate");
                }
            })
        });
        let mut reported = 0;
        for _ in 0..CPU_RUNS {
            let (kept, stats) = with_op_stats(|| {
                let reader = st.reader().expect("reader");
                reader
                    .block_on(reader.kept_by(&hits, &costly))
                    .expect("kept_by")
            });
            assert!(kept.expect("evaluated").iter().all(|kept| *kept));
            reported += stats.kernel_cpu_ns;
        }
        assert!(alone > 0, "{CPU_RUNS} evaluations take no CPU");
        assert!(
            reported >= alone / CPU_SHARE_FLOOR,
            "checks report {reported} ns of a condition taking {alone} ns"
        );
    }

    /// 2^24 + 1 is the first integer a `Float32` cannot hold: the floats around it are two apart.
    const FLOAT32_EXACT: i64 = 1 << 24;
    /// Rows of each value in [`float32_table`].
    const FLOAT32_ROWS_EACH: usize = 3;

    /// A range-indexed `Float32` column `f` holding the floats from 2^24 - 4 to 2^24 + 4, two
    /// apart.
    fn float32_table(dir: &TempDir) -> Supertable {
        let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float32, false)]));
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(Arc::clone(&schema), vec![], vec![])
            .expect("options")
            .with_storage(storage)
            .with_range_indexes(&["f".to_string()]);
        let st = Supertable::create(options).expect("create");
        let floats: Vec<f32> = (-2..=2_i64)
            .flat_map(|step| repeat_n((FLOAT32_EXACT + 2 * step) as f32, FLOAT32_ROWS_EACH))
            .collect();
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Float32Array::from(floats))])
            .expect("batch");
        let mut writer = st.writer().expect("writer");
        writer.append(&batch).expect("append");
        writer.commit().expect("commit");
        st
    }

    #[test]
    fn a_bound_compares_with_its_literal_cast_to_the_column() {
        // Integer bounds on a `Float32` column, cast as a statement casts them.
        //  - `f = 2^24 + 1` keeps the rows at 2^24, which it rounds to; `f < 2^24 + 1` keeps none
        //    of them.
        //  - Literals a `Float32` holds exactly read as before.
        // This test pins that each page matches the SQL sort.
        let dir = TempDir::new().expect("tempdir");
        let st = float32_table(&dir);
        let reader = st.reader().expect("reader");
        let inexact = FLOAT32_EXACT + 1;
        let cases = [
            (col("f").eq(lit(inexact)), format!("f = {inexact}")),
            (col("f").lt(lit(inexact)), format!("f < {inexact}")),
            (col("f").lt_eq(lit(inexact)), format!("f <= {inexact}")),
            (col("f").gt(lit(inexact)), format!("f > {inexact}")),
            (
                col("f").eq(lit(FLOAT32_EXACT)),
                format!("f = {FLOAT32_EXACT}"),
            ),
            (
                col("f").lt(lit(FLOAT32_EXACT)),
                format!("f < {FLOAT32_EXACT}"),
            ),
        ];
        for (filter, sql) in cases {
            let got = sorted_search_where(&st, Some("f"), &filter, false, 10, Some(&["_id"]))
                .expect("sorted_search_where")
                .unwrap_or_else(|| panic!("{sql}: answered"));
            let want = reader
                .query_sql(&format!(
                    "SELECT _id FROM supertable WHERE {sql} ORDER BY f ASC, _id ASC LIMIT 10"
                ))
                .expect("sql");
            assert!(!ids(&want).is_empty(), "{sql}: SQL keeps rows");
            assert_eq!(ids(&got), ids(&want), "{sql}");
        }
    }

    #[test]
    fn a_compound_filter_is_cast_before_it_is_simplified() {
        // `f != 2^24 + 1` on a `Float32` column reads as `f != 2^24`.
        //  - Beside `f = 2^24` it keeps no row.
        //  - Beside a bound it drops the rows at 2^24.
        // This test pins that a compound filter is cast before it is simplified.
        let dir = TempDir::new().expect("tempdir");
        let st = float32_table(&dir);
        let reader = st.reader().expect("reader");
        let inexact = FLOAT32_EXACT + 1;
        let below = FLOAT32_EXACT - 2;
        let cases = [
            (
                col("f")
                    .not_eq(lit(inexact))
                    .and(col("f").eq(lit(FLOAT32_EXACT))),
                format!("f != {inexact} AND f = {FLOAT32_EXACT}"),
                false,
            ),
            (
                col("f")
                    .not_eq(lit(inexact))
                    .and(col("f").gt_eq(lit(below))),
                format!("f != {inexact} AND f >= {below}"),
                true,
            ),
        ];
        for (filter, sql, keeps_rows) in cases {
            let got = sorted_search_where(&st, Some("f"), &filter, false, 10, Some(&["_id"]))
                .expect("sorted_search_where")
                .unwrap_or_else(|| panic!("{sql}: answered"));
            let want = reader
                .query_sql(&format!(
                    "SELECT _id FROM supertable WHERE {sql} ORDER BY f ASC, _id ASC LIMIT 10"
                ))
                .expect("sql");
            assert_eq!(!ids(&want).is_empty(), keeps_rows, "{sql}: SQL's rows");
            assert_eq!(ids(&got), ids(&want), "{sql}");
        }
    }

    #[test]
    fn a_walk_whose_first_rows_fall_short_reads_its_full_first_window() {
        // 8,000 rows, 20 of them tagged `r`: 5 among the first 4,096 by `seq`, 15 after.
        //  - `k = 10` at that share expects 4,000 rows, within the limit, so the walk starts.
        //  - Its first window reads twice that, finds 20, and answers as the SQL sort.
        // This test pins that the limit does not cut the first window's margin.
        let tagged =
            |s: i64| (s < 4000 && s % 800 == 0) || (4100..7850).step_by(250).any(|t| t == s);
        assert!((0..4096).filter(|s| tagged(*s)).count() < 10);
        let dir = TempDir::new().expect("tempdir");
        let schema = Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("tag", DataType::Utf8, false),
        ]));
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(Arc::clone(&schema), vec![], vec![])
            .expect("options")
            .with_storage(storage)
            .with_range_indexes(&["seq".to_string()]);
        let st = Supertable::create(options).expect("create");
        for file in 0..2_i64 {
            let seqs: Vec<i64> = (file * 4000..(file + 1) * 4000).collect();
            let tags: Vec<&str> = seqs
                .iter()
                .map(|s| if tagged(*s) { "r" } else { "c" })
                .collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(StringArray::from(tags)),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        let filter = col("tag").eq(lit("r"));
        let got = sorted_search_where(&st, Some("seq"), &filter, false, 10, Some(&["_id", "seq"]))
            .expect("sorted_search_where")
            .expect("answered by the walk");
        let want = st
            .reader()
            .expect("reader")
            .query_sql(
                "SELECT _id, seq FROM supertable WHERE tag = 'r' \
                 ORDER BY seq ASC NULLS LAST, _id ASC LIMIT 10",
            )
            .expect("sql");
        assert_eq!(ints(&want, "seq").len(), 10);
        assert_eq!(ints(&got, "seq"), ints(&want, "seq"));
    }

    /// The many-files table: 40 commits of 60 rows, 12 `key` values so each key's rows lie in every
    /// file and a window ends among them, `key` null every 11th row, and `seq` below 100 deleted.
    const MANY_FILES: i64 = 40;
    const MANY_ROWS: i64 = 60;
    const MANY_KEYS: i64 = 12;
    const MANY_NULL_EVERY: i64 = 11;
    const MANY_DELETED_BELOW: i64 = 100;
    /// The `(descending, k)` asked of each filter.
    const MANY_ASKS: &[(bool, usize)] = &[
        (false, 1),
        (true, 1),
        (false, 10),
        (true, 10),
        (false, 100),
        (true, 100),
        (false, 700),
        (true, 700),
    ];

    /// The many-files table, `key` range indexed with its values in no order.
    fn many_table(dir: &TempDir) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(schema(), vec![], vec![])
            .expect("options")
            .with_storage(storage)
            .with_range_indexes(&["key".to_string()]);
        let st = Supertable::create(options).expect("create");
        for file in 0..MANY_FILES {
            let seqs: Vec<i64> = (file * MANY_ROWS..(file + 1) * MANY_ROWS).collect();
            let keys: Vec<Option<i64>> = seqs
                .iter()
                .map(|s| (s % MANY_NULL_EVERY != 0).then_some((s * 7) % MANY_KEYS))
                .collect();
            let batch = RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(Int64Array::from(keys)),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st.delete(col("seq").lt(lit(MANY_DELETED_BELOW)))
            .expect("delete");
        st
    }

    #[test]
    fn sorted_search_where_walks_many_superfiles_in_order() {
        // 40 files whose rows of each key interleave, walked over several windows that end among a
        // key's rows.
        //  - Each page matches the SQL sort with `NULLS LAST`, both directions.
        //  - A page reaching past the rows with a value, with no bound on the column, is left to
        //    SQL.
        // This test pins the walk's cursor across windows and files.
        let dir = TempDir::new().expect("tempdir");
        let st = many_table(&dir);
        let reader = st.reader().expect("reader");
        // Asks the walk must leave to SQL: 697 rows with a value keep `seq % 3 = 0` and 105 keep
        // `seq % 20 = 0`, so their pages of 700 need nulls; the bounded filter's 700 is its 207
        // rows.
        let filters: [(Expr, &str, &[usize]); 3] = [
            (
                (col("seq") % lit(3_i64)).eq(lit(0_i64)),
                "seq % 3 = 0",
                &[700],
            ),
            (
                (col("seq") % lit(20_i64)).eq(lit(0_i64)),
                "seq % 20 = 0",
                &[700],
            ),
            (
                col("key")
                    .lt(lit(6_i64))
                    .and((col("seq") % lit(5_i64)).eq(lit(2_i64))),
                "key < 6 AND seq % 5 = 2",
                &[],
            ),
        ];
        for (filter, sql, declined) in &filters {
            for &(descending, k) in MANY_ASKS {
                let what = format!("{sql}, descending {descending}, k {k}");
                let got = sorted_search_where(
                    &st,
                    Some("key"),
                    filter,
                    descending,
                    k,
                    Some(&["_id", "seq", "key"]),
                )
                .expect("sorted_search_where");
                let direction = if descending { "DESC" } else { "ASC" };
                let want = reader
                    .query_sql(&format!(
                        "SELECT _id, seq, key FROM supertable WHERE {sql} ORDER BY key {direction} NULLS LAST, _id ASC LIMIT {k}"
                    ))
                    .expect("sql");
                if declined.contains(&k) {
                    assert!(
                        ints(&want, "key").iter().any(Option::is_none),
                        "{what}: SQL's page holds nulls"
                    );
                    assert!(got.is_none(), "{what}: left to the statement");
                    continue;
                }
                let got = got.unwrap_or_else(|| panic!("{what}: answered by the walk"));
                assert_eq!(ids(&got), ids(&want), "{what}");
                assert_eq!(ints(&got, "seq"), ints(&want, "seq"), "{what}");
                assert_eq!(ints(&got, "key"), ints(&want, "key"), "{what}");
            }
        }
    }

    #[test]
    fn sorted_search_where_declines_what_the_index_cannot_answer() {
        // Filters the walk must leave to the statement:
        //  - a column with no range index, or two columns with none named.
        //  - a filter keeping no row of the order with nothing bounding the column (nulls unread).
        //  - a condition only a statement can evaluate.
        //  - an equality on a column with thousands of distinct values, which needs more rows than
        //    a walk reads.
        // This test pins that each one answers `None`.
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        let declined: [(Option<&str>, Expr); 5] = [
            (Some("key"), col("seq").eq(lit(5_000_i64))),
            (Some("seq"), col("seq").gt(lit(10_i64))),
            (
                None,
                col("key").gt(lit(10_i64)).and(col("seq").gt(lit(10_i64))),
            ),
            (Some("key"), col("seq").lt(lit(0_i64))),
            (
                Some("key"),
                datafusion::logical_expr::in_subquery(
                    col("seq"),
                    Arc::new(
                        datafusion::logical_expr::LogicalPlanBuilder::empty(true)
                            .project(vec![lit(1_i64)])
                            .and_then(|b| b.build())
                            .expect("plan"),
                    ),
                ),
            ),
        ];
        for (column, filter) in &declined {
            let got = sorted_search_where(&st, *column, filter, false, 10, Some(&["_id", "key"]))
                .expect("sorted_search_where");
            assert!(got.is_none(), "{column:?} {filter}: left to a statement");
        }
    }

    /// Every ask answers as the same sort in SQL does, row for row, before
    /// and after an optimize merges the superfiles out of `_id` order; the
    /// asks within the rows with a value come from the range indexes.
    #[test]
    fn sorted_search_answers_as_the_sql_sort() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        for pass in ["appended", "optimized"] {
            if pass == "optimized" {
                st.optimize(&OptimizeOptions::default()).expect("optimize");
            }
            let reader = st.reader().expect("reader");
            for &(descending, k) in ASKS {
                let got = st
                    .sorted_search("key", descending, k, Some(&["_id", "seq", "key"]))
                    .expect("sorted_search");
                let direction = if descending { "DESC" } else { "ASC" };
                let want = reader
                    .query_sql(&format!(
                        "SELECT _id, seq, key FROM supertable ORDER BY key {direction} NULLS LAST, _id ASC LIMIT {k}"
                    ))
                    .expect("sql");
                assert_eq!(
                    ids(&got),
                    ids(&want),
                    "{pass}: descending {descending}, k {k}"
                );
                assert_eq!(
                    ints(&got, "seq"),
                    ints(&want, "seq"),
                    "{pass}: {descending} {k}"
                );
                assert_eq!(
                    ints(&got, "key"),
                    ints(&want, "key"),
                    "{pass}: {descending} {k}"
                );
                let from_index = reader
                    .sorted_hits("key", descending, k)
                    .expect("hits")
                    .is_some();
                let with_values = ints(&want, "key").iter().all(Option::is_some);
                assert_eq!(from_index, with_values, "{pass}: {descending} {k}");
            }
        }
    }

    /// With no projection the rows are `_id` and a `score` of `0.0`, on the
    /// index path and on the SQL one alike.
    #[test]
    fn a_bare_sorted_search_returns_ids_and_a_zero_score() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        for k in [10, 7_700] {
            let batches = st
                .sorted_search("key", false, k, None)
                .expect("sorted_search");
            let names: Vec<String> = batches[0]
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            assert_eq!(
                names,
                vec!["_id".to_string(), SCORE_COLUMN.to_string()],
                "k {k}"
            );
            let scores = batches[0]
                .column(1)
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("score");
            assert!(scores.values().iter().all(|s| *s == 0.0), "k {k}");
        }
    }

    /// A `k` past every row the table holds returns every row in the sort's
    /// order, as the same sort with no limit does, rather than sizing
    /// anything by `k`.
    #[test]
    fn a_k_past_every_row_returns_every_row() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        let reader = st.reader().expect("reader");
        for descending in [false, true] {
            let got = st
                .sorted_search("key", descending, usize::MAX, Some(&["_id"]))
                .expect("sorted_search");
            let direction = if descending { "DESC" } else { "ASC" };
            let want = reader
                .query_sql(&format!(
                    "SELECT _id FROM supertable ORDER BY key {direction} NULLS LAST, _id ASC"
                ))
                .expect("sql");
            assert_eq!(ids(&got), ids(&want), "descending {descending}");
            assert!(
                reader
                    .sorted_hits("key", descending, usize::MAX)
                    .expect("hits")
                    .is_none(),
                "descending {descending}"
            );
        }
    }

    /// Superfiles of the key-kinds table, one commit each.
    const KIND_FILES: usize = 3;
    /// Floats in IEEE total order's corners, cycled through the rows: both
    /// zeros, both NaNs, both infinities, and ordinary values between.
    const FLOATS: [f64; 9] = [
        -0.0,
        0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1.5,
        -2.5,
        f64::MIN_POSITIVE,
        -f64::MAX,
    ];
    /// Rows in each superfile of the key-kinds table.
    const KIND_ROWS: usize = 40;
    /// Every this-many rows the key-kinds table's columns are null.
    const KIND_NULL_EVERY: usize = 6;
    /// Distinct timestamps, a second apart, so the timestamp cutoffs tie.
    const TIMESTAMPS: i64 = 7;
    /// Milliseconds in a second.
    const MILLIS_PER_SECOND: i64 = 1000;
    /// A column whose name SQL must quote: a space and capitals.
    const QUOTED: &str = "Sort Key";
    /// The `(descending, k)` each key-kinds ask runs, the last past the rows
    /// with a value, so the statement answers it.
    const KIND_ASKS: &[(bool, usize)] = &[
        (false, 3),
        (true, 3),
        (false, 8),
        (true, 8),
        (false, 11),
        (true, 11),
        (false, 115),
        (true, 115),
    ];

    /// A table of a `Float64`, a `Timestamp` and an `Int64` column whose name
    /// needs quoting, each range indexed and null now and then, in
    /// [`KIND_FILES`] commits. The float cycles through [`FLOATS`] with a
    /// negative NaN among them.
    fn kinds_table(dir: &TempDir) -> Supertable {
        let schema = Arc::new(Schema::new(vec![
            Field::new("f", DataType::Float64, true),
            Field::new("ts", DataType::Timestamp(TimeUnit::Millisecond, None), true),
            Field::new(QUOTED, DataType::Int64, true),
        ]));
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(Arc::clone(&schema), vec![], vec![])
            .expect("options")
            .with_storage(storage)
            .with_range_indexes(&["f".to_string(), "ts".to_string(), QUOTED.to_string()]);
        let st = Supertable::create(options).expect("create");
        for file in 0..KIND_FILES {
            let rows: Vec<usize> = (file * KIND_ROWS..(file + 1) * KIND_ROWS).collect();
            let valued = |row: usize| !row.is_multiple_of(KIND_NULL_EVERY);
            let floats: Vec<Option<f64>> = rows
                .iter()
                .map(|&row| {
                    valued(row).then(|| match row % (FLOATS.len() + 1) {
                        // NaN with the sign bit set, which total order puts first.
                        i if i == FLOATS.len() => -f64::NAN,
                        i => FLOATS[i],
                    })
                })
                .collect();
            let times: Vec<Option<i64>> = rows
                .iter()
                .map(|&row| valued(row).then(|| (row as i64 % TIMESTAMPS) * MILLIS_PER_SECOND))
                .collect();
            let quoted: Vec<Option<i64>> = rows
                .iter()
                .map(|&row| valued(row).then(|| (row as i64 * 13) % 5))
                .collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Float64Array::from(floats)),
                    Arc::new(TimestampMillisecondArray::from(times)),
                    Arc::new(Int64Array::from(quoted)),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st
    }

    /// Every row's `_id` and `column` value, read with no sort or limit,
    /// ordered by the value (IEEE total order for a float, descending with
    /// `descending`), rows without one last, ties by `_id` ascending; and how
    /// many rows have a value. Ordered here rather than by an `ORDER BY`, so
    /// the expectation does not rest on the engine's own sort.
    fn expected(reader: &SupertableReader, column: &str, descending: bool) -> (Vec<i128>, usize) {
        let batches = reader
            .query_sql(&format!(
                "SELECT _id, {} AS v FROM supertable",
                quote(column)
            ))
            .expect("rows");
        let mut rows: Vec<(Option<f64>, i128)> = Vec::new();
        for batch in &batches {
            let id_column = batch.column_by_name("_id").expect("_id");
            let ids = id_column
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("_id");
            let values = batch.column_by_name("v").expect("v");
            let floats = match values.data_type() {
                DataType::Float64 => Arc::clone(values),
                _ => cast(
                    &cast(values, &DataType::Int64).expect("int"),
                    &DataType::Float64,
                )
                .expect("float"),
            };
            let floats = floats
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("float");
            for i in 0..batch.num_rows() {
                rows.push((floats.is_valid(i).then(|| floats.value(i)), ids.value(i)));
            }
        }
        rows.sort_by(|(a, a_id), (b, b_id)| {
            let by_value = match (a, b) {
                (Some(a), Some(b)) if descending => b.total_cmp(a),
                (Some(a), Some(b)) => a.total_cmp(b),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            };
            by_value.then(a_id.cmp(b_id))
        });
        let valued = rows.iter().filter(|(v, _)| v.is_some()).count();
        (rows.into_iter().map(|(_, id)| id).collect(), valued)
    }

    /// Float, timestamp and quoted-name keys return the rows in the column's
    /// order, either way, ties by `_id`: the float in IEEE total order
    /// (negative NaN, negative infinity, …, negative zero before zero, …,
    /// infinity, NaN), from the range indexes within the rows with a value
    /// and from the statement past them.
    #[test]
    fn float_timestamp_and_quoted_keys_return_the_rows_in_order() {
        let dir = TempDir::new().expect("tempdir");
        let st = kinds_table(&dir);
        let reader = st.reader().expect("reader");
        for column in ["f", "ts", QUOTED] {
            for &(descending, k) in KIND_ASKS {
                let got = st
                    .sorted_search(column, descending, k, Some(&["_id"]))
                    .expect("sorted_search");
                let (order, valued) = expected(&reader, column, descending);
                let want: Vec<i128> = order.into_iter().take(k).collect();
                assert_eq!(ids(&got), want, "{column}: descending {descending}, k {k}");
                let from_index = reader
                    .sorted_hits(column, descending, k)
                    .expect("hits")
                    .is_some();
                assert_eq!(
                    from_index,
                    k <= valued,
                    "{column}: descending {descending}, k {k}"
                );
            }
        }
    }

    /// Keys the candidates of [`by_key_then_id_keeps_the_first_k_and_reserves_nothing`] take, so
    /// many share each key and the `_id` decides among them.
    const CANDIDATE_KEYS: u32 = 7;
    /// The candidates it keeps.
    const KEPT: usize = 40;

    /// `by_key_then_id` returns only the first `k` candidates by the key and then `_id`, either
    /// way. It reserves no memory, so it answers with the connection's budget full: its callers
    /// reserve the candidates, and a second reservation would count them twice.
    #[test]
    fn by_key_then_id_keeps_the_first_k_and_reserves_nothing() {
        let dir = TempDir::new().expect("tempdir");
        drop(table(&dir));
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let mut options = SupertableOptions::new(schema(), vec![], vec![])
            .expect("options")
            .with_storage(storage)
            .with_range_indexes(&["key".to_string()]);
        options.connection_memory_budget = ConnectionMemoryBudget::with_room();
        let st = Supertable::open(options).expect("open");
        let reader = st.reader().expect("reader");
        let _full = reader
            .manifest()
            .options
            .connection_memory_budget
            .hold_all();
        let candidates: Candidates = reader
            .manifest()
            .superfiles
            .iter()
            .flat_map(|entry| {
                let locals: Vec<u32> = (0..entry.n_docs as u32).collect();
                let keys = locals.iter().map(|local| u64::from(local % CANDIDATE_KEYS));
                let hits = tag_hits(entry, locals.iter().map(|local| (*local, 0.0)).collect());
                keys.zip(hits).collect::<Vec<_>>()
            })
            .collect();
        for descending in [false, true] {
            let mut want: Vec<(u64, Option<i128>)> = candidates
                .iter()
                .map(|(key, hit)| (*key, hit.stable_id))
                .collect();
            want.sort_by(|a, b| match descending {
                false => a.cmp(b),
                true => b.0.cmp(&a.0).then(a.1.cmp(&b.1)),
            });
            want.truncate(KEPT);
            let got: Vec<(u64, Option<i128>)> = reader
                .block_on(by_key_then_id(
                    &reader,
                    candidates.clone(),
                    descending,
                    KEPT,
                    |_, _| true,
                ))
                .expect("ordered")
                .into_iter()
                .map(|(key, hit)| (key, hit.stable_id))
                .collect();
            assert_eq!(got, want, "descending: {descending}");
        }
    }

    /// A column without a range index is refused by name, and a `k` of
    /// zero answers no rows.
    #[test]
    fn sorted_search_needs_a_range_index_and_takes_zero() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        let error = st
            .sorted_search("seq", false, 10, None)
            .expect_err("seq has no range index");
        assert!(error.to_string().contains("seq"), "{error}");
        let none = st.sorted_search("key", true, 0, None).expect("k 0");
        assert_eq!(none.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    }
}
