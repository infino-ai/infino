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

use std::{cmp::Ordering, slice::from_ref, sync::Arc, time::Instant};

use arrow::{
    array::{Array, AsArray},
    datatypes::DataType,
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DFSchema, ScalarValue, tree_node::TreeNode},
    execution::context::ExecutionProps,
    logical_expr::{
        Expr,
        simplify::SimplifyContext,
        type_coercion::binary::comparison_coercion,
        utils::{conjunction, split_conjunction},
    },
    optimizer::{
        analyzer::type_coercion::TypeCoercionRewriter, simplify_expressions::ExprSimplifier,
    },
    physical_expr::create_physical_expr,
};
use roaring::RoaringBitmap;
use tracing::{Instrument, debug};

use crate::{
    InfinoError,
    runtime_bridge::run_on_pool,
    runtime_metrics::op_stats::timed_section,
    superfile::{SuperfileReader, range::KeyKind},
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        manifest::{SuperfileEntry, hll::HllSketch},
        query::{
            SuperfileHit,
            candidate::single_range,
            dispatch::{self, tag_hits, tombstone_deny_set},
            exec::common::resolve_hits_named,
            provider::flatten_or_eq,
            skip::ColumnTypeGuard,
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

/// The most rows a [`Supertable::sorted_search_where`] walk reads in the
/// index's order to find the rows its other conditions keep. Past it the
/// caller's statement answers instead: a condition that keeps so few rows
/// reads them faster from a scan than one window at a time.
const MAX_WALKED_ROWS: usize = 4096;

/// How many times the rows a walk expects to need its windows read, so
/// that a window usually holds the page at the first try.
const WINDOW_MARGIN: f64 = 2.0;

/// How much larger a window is than the last when the last kept no row.
const WINDOW_GROWTH: usize = 4;

/// What [`Supertable::sorted_search_where`] reads a filter as: the column
/// the rows come in order of, the interval of its keys the filter keeps
/// (the whole key space when no condition bounds it), and the conditions
/// left over, which the rows read are checked against.
struct SortedFilter {
    column: String,
    interval: (u64, u64),
    /// Whether a condition bounds the column, so a row without a value is
    /// out of the answer rather than an unread row that might be in it.
    bounded: bool,
    rest: Option<Expr>,
}

/// The share of a table's rows a condition is expected to keep
/// ([`SupertableReader::expected_share`]), and whether the statistics prove
/// it keeps none.
struct Share {
    share: f64,
    none: bool,
}

/// The rows an exact value table's `entries` count for `literals` together,
/// each literal compared with the column as a statement compares them:
/// `None` when a statement would cast the column (`column_type`) to compare
/// it with a literal, as a string column against a number is, when the
/// table's values are not of `column_type`, or when a literal does not
/// take it. The entries are sorted, so each literal is a binary search.
fn literal_counts(
    entries: &[(ScalarValue, u64)],
    literals: &[ScalarValue],
    column_type: &DataType,
) -> Option<u64> {
    let Some((first, _)) = entries.first() else {
        return Some(0);
    };
    if first.data_type() != *column_type {
        return None;
    }
    let mut held = 0u64;
    for literal in literals {
        if comparison_coercion(column_type, &literal.data_type()).as_ref() != Some(column_type) {
            return None;
        }
        let literal = literal.cast_to(column_type).ok()?;
        // Values that do not order map to `Equal`, so they land on the
        // equality check below rather than on a miss.
        match entries
            .binary_search_by(|(value, _)| value.partial_cmp(&literal).unwrap_or(Ordering::Equal))
        {
            Ok(at) if entries[at].0 == literal => {
                held = held.saturating_add(entries[at].1);
            }
            Ok(_) => return None,
            Err(_) => {}
        }
    }
    Some(held)
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
        self.sorted_hits_within_async(column, descending, k, None)
            .await
    }

    /// [`Self::sorted_hits_async`] over the rows whose key lies in
    /// `interval` (keys as the range index encodes them, both ends
    /// inclusive) when one is given. A row without a value lies in no
    /// interval, so fewer than `k` rows in it is the whole answer, not a
    /// page the index cannot fill.
    async fn sorted_hits_within_async(
        &self,
        column: &str,
        descending: bool,
        k: usize,
        interval: Option<(u64, u64)>,
    ) -> Result<Option<Vec<SuperfileHit>>, QueryError> {
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
        let bounds = interval.unwrap_or((0, u64::MAX));
        let field = *field;
        let units: Vec<_> = manifest
            .superfiles
            .iter()
            .cloned()
            .map(|e| (e, ()))
            .collect();
        let data_type = data_type.clone();
        let body = move |r: Arc<SuperfileReader>,
                         entry: Arc<SuperfileEntry>,
                         tombstone_cache: Option<Arc<SidecarCache>>,
                         now: Instant,
                         _: ()| {
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
                        k,
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
        let (keys, mut hits): (Vec<u64>, Vec<SuperfileHit>) =
            candidates.into_iter().flatten().unzip();
        if hits.len() < k && interval.is_none() {
            return Ok(None);
        }
        dispatch::attach_stable_ids_to_hits(self, &mut hits).await?;
        let mut order: Vec<usize> = (0..hits.len()).collect();
        order.sort_by(|&a, &b| {
            let by_key = match descending {
                false => keys[a].cmp(&keys[b]),
                true => keys[b].cmp(&keys[a]),
            };
            by_key.then_with(|| hits[a].stable_id.cmp(&hits[b].stable_id))
        });
        Ok(Some(order.into_iter().take(k).map(|i| hits[i]).collect()))
    }

    /// The rows this snapshot's superfiles hold, deleted ones included: no
    /// sort can return more.
    fn rows_held(&self) -> u64 {
        self.manifest().superfiles.iter().map(|e| e.n_docs).sum()
    }

    /// How `filter` reads in the order of `column` (or, with `None`, of the
    /// one column it names): its comparisons and `BETWEEN`s on that column
    /// against literals, `AND`ed, are the interval of the column's range
    /// index keys to read, and its other conditions are left to check row
    /// by row. `None` when the column has no range index, or when no column
    /// is named and the filter names more than one.
    fn sorted_filter(
        &self,
        column: Option<&str>,
        filter: &Expr,
    ) -> Result<Option<SortedFilter>, QueryError> {
        let schema = Arc::new(
            DFSchema::try_from(self.sql_schemas().scan().as_ref().clone())
                .map_err(QueryError::DataFusion)?,
        );
        let simplifier = ExprSimplifier::new(
            SimplifyContext::builder()
                .with_schema(Arc::clone(&schema))
                .build(),
        );
        let filter = simplifier
            .simplify(filter.clone())
            .map_err(|e| QueryError::InvalidQuery(format!("sorted_search_where: {e}")))?;
        let column = match column {
            Some(column) => column.to_string(),
            None => {
                let named = filter.column_refs();
                let mut names = named.iter().map(|c| c.name.as_str());
                match (names.next(), names.next()) {
                    (Some(name), None) => name.to_string(),
                    _ => return Ok(None),
                }
            }
        };
        let range_cols = self.manifest().table_schema().range_column_types();
        if !range_cols.contains_key(&column) {
            return Ok(None);
        }
        // A bound is read with its literal cast as a statement casts it to
        // compare with the column: `f = 16777217` on a `Float32` column
        // compares with the float `16777216`, not the integer. A condition
        // that is no bound once cast stays as written, to check row by row.
        let mut bounds = Vec::new();
        let mut rest = Vec::new();
        for conjunct in split_conjunction(&filter) {
            if matches!(conjunct, Expr::Literal(ScalarValue::Boolean(Some(true)), _)) {
                continue;
            }
            let coerced = conjunct
                .clone()
                .rewrite(&mut TypeCoercionRewriter::new(&schema))
                .map(|t| t.data)
                .and_then(|c| simplifier.simplify(c));
            match coerced {
                Ok(bound) if single_range(from_ref(&bound), &column, &range_cols).is_some() => {
                    bounds.push(bound);
                }
                _ => rest.push(conjunct.clone()),
            }
        }
        let Some((_, _, lo, hi)) = single_range(&bounds, &column, &range_cols) else {
            return Ok(None);
        };
        Ok(Some(SortedFilter {
            column,
            interval: (lo, hi),
            bounded: !bounds.is_empty(),
            rest: conjunction(rest),
        }))
    }

    /// The share of the table's rows `condition` is expected to keep, from
    /// the superfiles' column statistics: a column equal to a literal keeps
    /// about one row in the column's distinct values (counted exactly where
    /// a superfile keeps a value table, estimated from its distinct sketch
    /// elsewhere), an `IN` list one per value, and an `AND` the product of
    /// its sides. Any other condition, or a column a superfile keeps no
    /// statistic of, counts as keeping every row: the walk then learns the
    /// share from the rows it reads.
    fn expected_share(&self, condition: &Expr) -> Share {
        let mut share = 1.0;
        let mut none = false;
        for conjunct in split_conjunction(condition) {
            let one = self.conjunct_share(conjunct);
            share *= one.share;
            none |= one.none;
        }
        Share {
            share: share.clamp(0.0, 1.0),
            none,
        }
    }

    /// [`Self::expected_share`] of one conjunct. Where a superfile keeps an
    /// exact value table for the column, the literals' own counts in it are
    /// read rather than one row per distinct value. A table counts every row
    /// the superfile holds, deleted ones included, so a literal it lacks is
    /// held by no live row there either.
    fn conjunct_share(&self, conjunct: &Expr) -> Share {
        let unknown = Share {
            share: 1.0,
            none: false,
        };
        let schemas = self.sql_schemas();
        let (column, literals): (String, Vec<ScalarValue>) = match conjunct {
            // An equality, or the equalities `OR`ed the simplifier turns a
            // short `IN` list into.
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
        // The column in its stored type: the scan views a string column as
        // `Utf8View`, which compares as the stored string does.
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
        // Statistics in a type the column has since left describe values
        // the statement no longer compares as they are stored.
        let guard = ColumnTypeGuard::new(manifest, field);
        // Rows expected to hold the literals, summed over the superfiles, and
        // whether every superfile's part of that is a count, not an estimate.
        let mut expected = 0.0;
        let mut rows = 0.0;
        let mut counted = true;
        for entry in manifest.superfiles.iter() {
            rows += entry.n_docs as f64;
            if guard.stats_are_stale(entry) {
                return unknown;
            }
            let Some(stats) = entry.scalar_stats.column(field) else {
                return unknown;
            };
            if stats.all_null() {
                continue;
            }
            let present = entry.n_docs.saturating_sub(stats.null_count().unwrap_or(0)) as f64;
            if let Some(table) = stats.value_counts() {
                if let Some(held) = literal_counts(table.entries(), &literals, column_type) {
                    expected += held as f64;
                    continue;
                }
                counted = false;
                let distinct = table.entries().len() as f64;
                if distinct >= 1.0 {
                    expected += present * (literals.len() as f64 / distinct).min(1.0);
                }
                continue;
            }
            counted = false;
            let Some(sketch) = stats.sketch().and_then(HllSketch::from_bytes) else {
                return unknown;
            };
            let distinct = sketch.estimate();
            if distinct >= 1.0 {
                expected += present * (literals.len() as f64 / distinct).min(1.0);
            }
        }
        if rows == 0.0 {
            return unknown;
        }
        Share {
            share: expected / rows,
            none: counted && expected == 0.0,
        }
    }

    /// Which of `hits` `condition` keeps, read from the hits' own rows: the
    /// columns it names are resolved for them and it is evaluated over
    /// those. `None` when it cannot be evaluated alone, outside a statement
    /// (a subquery, a function only a statement registers), or names a
    /// column the table does not store.
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
        // The condition runs over up to a window of rows: compute, so the
        // reader pool takes it, and its time is the query's as a
        // statement's predicate is.
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
                    // A condition a row's nulls leave unknown does not keep
                    // it, as in a statement's WHERE.
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

    /// The first `k` rows `filter` keeps in its column's order, as hits in
    /// that order: the interval's rows read from the range indexes a window
    /// at a time, each window checked against the conditions the interval
    /// leaves, until `k` are kept or the interval ends. `None` when the
    /// indexes cannot answer, when the conditions cannot be evaluated alone,
    /// or when [`MAX_WALKED_ROWS`] rows do not find `k`; and, with no
    /// condition bounding the column, when its rows with a value run out
    /// first, since the rows without one sort after them unread.
    async fn sorted_hits_where_async(
        &self,
        filter: &SortedFilter,
        descending: bool,
        k: usize,
    ) -> Result<Option<Vec<SuperfileHit>>, QueryError> {
        let Some(rest) = &filter.rest else {
            if !filter.bounded {
                return self.sorted_hits_async(&filter.column, descending, k).await;
            }
            return self
                .sorted_hits_within_async(&filter.column, descending, k, Some(filter.interval))
                .await;
        };
        if k == 0 {
            return Ok(Some(Vec::new()));
        }
        // Rows to read for `k` to pass, at the share the statistics expect;
        // a walk expected to run past its limit is left to the statement
        // before it reads a row.
        let needed = |share: f64| k as f64 / share;
        let window_for = |share: f64| {
            ((needed(share) * WINDOW_MARGIN).ceil() as usize).clamp(k, MAX_WALKED_ROWS)
        };
        let Share { share, none } = self.expected_share(rest);
        if none {
            // The statistics count every value a condition names, and no row
            // holds one: the filter keeps nothing.
            return Ok(Some(Vec::new()));
        }
        if share <= 0.0 || needed(share) > MAX_WALKED_ROWS as f64 {
            return Ok(None);
        }
        let mut window = window_for(share);
        loop {
            let window_now = window.min(MAX_WALKED_ROWS);
            let Some(hits) = self
                .sorted_hits_within_async(
                    &filter.column,
                    descending,
                    window_now,
                    Some(filter.interval),
                )
                .await?
            else {
                return Ok(None);
            };
            let ended = hits.len() < window_now;
            let Some(kept) = self.kept_by(&hits, rest).await? else {
                return Ok(None);
            };
            let passed = kept.iter().filter(|kept| **kept).count();
            let page: Vec<SuperfileHit> = hits
                .iter()
                .zip(kept)
                .filter_map(|(hit, kept)| kept.then_some(*hit))
                .take(k)
                .collect();
            if page.len() == k || (ended && filter.bounded) {
                return Ok(Some(page));
            }
            if ended || window_now == MAX_WALKED_ROWS {
                return Ok(None);
            }
            // The next window from the share the rows read kept; past the
            // limit, the statement answers sooner than the walk would.
            let window_next = match passed {
                0 => window_now.saturating_mul(WINDOW_GROWTH),
                _ => {
                    let seen = passed as f64 / window_now as f64;
                    if needed(seen) > MAX_WALKED_ROWS as f64 {
                        return Ok(None);
                    }
                    window_for(seen)
                }
            };
            window = window_next.max(window_now.saturating_mul(2));
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
    /// The first `k` rows `filter` keeps by `column`, ascending or (with
    /// `descending`) descending, rows sharing a value ordered by `_id`
    /// ascending: what `SELECT … WHERE filter ORDER BY column, _id LIMIT k`
    /// answers, read from the column's range index without planning a
    /// statement, as [`Supertable::sorted_search`] reads it. With `column`
    /// `None` the column is the one `filter` names.
    ///
    /// The filter's comparisons and `BETWEEN`s on the column against
    /// literals, `AND`ed, pick the interval of the index to read; a row
    /// without a value is in no interval, so fewer than `k` rows come back
    /// when fewer are in it. Its other conditions, `AND`ed to those, are
    /// checked on the rows the index gives, a window at a time, until `k`
    /// pass. Every superfile must hold the column's range index with its
    /// ties in `_id` order. `None`, for the caller to plan the statement,
    /// when the column has no range index, when no column is named and the
    /// filter names several, when the other conditions need a statement
    /// to evaluate (a subquery, say), or when they keep too few of the rows
    /// read in order to find `k` within a few thousand. When the superfiles'
    /// exact value tables show that no row holds a value the conditions
    /// require (an equality or `IN` on a column they count), the page is
    /// empty without a row read.
    pub fn sorted_search_where(
        &self,
        column: Option<&str>,
        filter: &Expr,
        descending: bool,
        k: usize,
        projection: Option<&[&str]>,
    ) -> Result<Option<Vec<RecordBatch>>, InfinoError> {
        debug!(?column, descending, k, "sorted_search_where");
        let close_out = self.close_out();
        let reader = self.reader()?;
        let context =
            |e: QueryError| InfinoError::from(e).with_context("sorted_search_where", None);
        reader.check_projection(projection).map_err(context)?;
        let Some(filter) = reader.sorted_filter(column, filter).map_err(context)? else {
            return Ok(None);
        };
        let k = k.min(usize::try_from(reader.rows_held()).unwrap_or(usize::MAX));
        let hits = {
            let _foreground = ForegroundQueryGuard::enter();
            reader
                .block_on(reader.sorted_hits_where_async(&filter, descending, k))
                .map_err(context)?
        };
        let Some(hits) = hits else {
            return Ok(None);
        };
        let batches = vec![
            self.block_on_query(
                resolve_hits_named(&reader, &hits, projection)
                    .instrument(detail_span!("search.resolve", hits = hits.len())),
            )
            .map_err(context)?,
        ];
        close_out.finish_batches(&batches);
        Ok(Some(batches))
    }

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
fn quote(name: &str) -> String {
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
        runtime_metrics::op_stats::with_op_stats,
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{
            SupertableOptions,
            schema::change::{FieldPatch, SchemaPatch},
        },
    };

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

    /// Over one interval of `key`, every ask answers as the same filtered
    /// sort in SQL does, row for row, before and after an optimize, from
    /// the range indexes: an interval holds no null, so a short page is the
    /// whole answer. With no column named, the interval's own column sorts.
    #[test]
    fn sorted_search_where_answers_as_the_filtered_sql_sort() {
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
                        let got = st
                            .sorted_search_where(
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

    /// With conditions beside the interval, on other columns or on the
    /// sorted one, every ask the walk answers is the same filtered sort in
    /// SQL, row for row, before and after an optimize; the asks within a
    /// few thousand rows of the order are answered. With no condition
    /// bounding the column, a walk that runs past its rows with a value
    /// leaves the nulls to the statement.
    #[test]
    fn sorted_search_where_checks_the_other_conditions_on_the_rows_in_order() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        // The first three keep a third or more of the rows in any stretch of
        // the order, so a page of up to 125 is found within the walk.
        let filters: [(Expr, &str); 6] = [
            (
                col("key")
                    .gt_eq(lit(10_i64))
                    .and((col("seq") % lit(3_i64)).eq(lit(0_i64))),
                "key >= 10 AND seq % 3 = 0",
            ),
            ((col("seq") % lit(2_i64)).eq(lit(1_i64)), "seq % 2 = 1"),
            (
                col("key").lt(lit(25_i64)).and(
                    col("seq")
                        .gt(lit(4_000_i64))
                        .or(col("seq").lt(lit(1_000_i64))),
                ),
                "key < 25 AND (seq > 4000 OR seq < 1000)",
            ),
            (
                col("key").eq(lit(3_i64)).or(col("key").eq(lit(5_i64))),
                "(key = 3 OR key = 5)",
            ),
            (
                col("key").gt(lit(40_i64)).and(
                    col("seq").in_list(vec![lit(301_i64), lit(302_i64), lit(4_444_i64)], false),
                ),
                "key > 40 AND seq IN (301, 302, 4444)",
            ),
            (col("seq").lt(lit(0_i64)), "seq < 0"),
        ];
        for pass in ["appended", "optimized"] {
            if pass == "optimized" {
                st.optimize(&OptimizeOptions::default()).expect("optimize");
            }
            let reader = st.reader().expect("reader");
            for (at, (filter, sql)) in filters.iter().enumerate() {
                for &(descending, k) in ASKS {
                    let got = st
                        .sorted_search_where(
                            Some("key"),
                            filter,
                            descending,
                            k,
                            Some(&["_id", "seq", "key"]),
                        )
                        .expect("sorted_search_where");
                    let what = format!("{pass}: {sql}, descending {descending}, k {k}");
                    let Some(got) = got else {
                        assert!(at > 2 || k > 125, "{what}: a dense condition is answered");
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

    /// A condition on a column the superfiles keep exact value tables for
    /// reads its literals' counts: one no table holds, alone or in an `IN`
    /// of such literals, keeps no row, and the page is empty without a row
    /// read; a literal a table holds is walked and answers as the SQL sort.
    #[test]
    fn a_value_no_value_table_holds_keeps_no_row() {
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
            st.sorted_search_where(Some("seq"), filter, true, 10, Some(&["_id", "seq"]))
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

    /// Distinct `tag` values of the numeric-tag table, each written with a
    /// leading zero (`"00"` to `"06"`).
    const NUMERIC_TAGS: i64 = 7;
    /// The one value of the numeric-tag table's `n`: past 2^53, so it reads
    /// as `2^53` when cast to `Float64`.
    const BIG: i64 = (1 << 53) + 1;

    /// A table of a range-indexed `seq`, a `tag` string that holds numbers
    /// with a leading zero, and an `n` that is [`BIG`] on every row, in two
    /// commits, so each superfile keeps exact value tables for `tag` and `n`.
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

    /// `sorted_search_where(seq DESC, 10)` of `filter` and the SQL sort of
    /// `sql`, as `seq` values: the page must be answered, and match.
    fn assert_page_matches_sql(st: &Supertable, filter: &Expr, sql: &str) {
        let got = st
            .sorted_search_where(Some("seq"), filter, true, 10, Some(&["_id", "seq"]))
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

    /// A literal a statement compares by casting the column is not looked
    /// up in the column's value tables: `tag = 3` casts the string `"03"`
    /// to `3`, and `n = 2^53` casts [`BIG`] to the float `2^53`, so both
    /// keep rows the tables hold under another value. A literal of the
    /// column's own type is still counted: `tag = '09'` keeps no row.
    #[test]
    fn a_literal_the_column_is_cast_for_is_not_counted() {
        let dir = TempDir::new().expect("tempdir");
        let st = numeric_tag_table(&dir);
        assert_page_matches_sql(&st, &col("tag").eq(lit(3_i64)), "tag = 3");
        assert_page_matches_sql(
            &st,
            &col("n").eq(lit((1_i64 << 53) as f64)),
            "n = 9007199254740992.0",
        );
        let absent = st
            .sorted_search_where(
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

    /// After `tag` is converted from strings to `Int64`, the superfiles'
    /// value tables still count the strings: `tag = 3` reads `"03"` as `3`
    /// and keeps its rows rather than finding no `3` in the tables.
    #[test]
    fn value_tables_in_a_type_the_column_left_are_not_counted() {
        let dir = TempDir::new().expect("tempdir");
        let st = numeric_tag_table(&dir);
        st.apply_schema(
            &SchemaPatch::new(vec![FieldPatch::named("tag").with_type(DataType::Int64)]),
            None,
        )
        .expect("convert tag");
        assert_page_matches_sql(&st, &col("tag").eq(lit(3_i64)), "tag = 3");
    }

    /// Additions in the condition [`a_condition_checked_on_rows_reports_its_cpu`]
    /// evaluates, so its time dwarfs reading the rows it runs over.
    const COSTLY_ADDITIONS: usize = 200;
    /// Times that test evaluates the condition: the per-thread CPU clock
    /// advances at scheduler events, so one short run can read zero.
    const CPU_RUNS: usize = 50;
    /// Rows the condition is evaluated over in each run.
    const CPU_ROWS: usize = 2000;
    /// The least share of the condition's own CPU the checks must report:
    /// the clock's granularity may lose some, never most.
    const CPU_SHARE_FLOOR: u64 = 2;

    /// A condition checked on the rows a walk reads bills its CPU to the
    /// query, as a statement's predicate does: the checks report at least
    /// [`CPU_SHARE_FLOOR`] of what evaluating the condition alone, over the
    /// same rows, costs on this thread.
    #[test]
    #[cfg_attr(
        not(target_os = "linux"),
        ignore = "per-thread CPU clock is Linux procfs (schedstat); off Linux kernel_cpu_ns is always 0"
    )]
    fn a_condition_checked_on_rows_reports_its_cpu() {
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

    /// The first integer a `Float32` cannot hold is one past this: the
    /// floats around it are two apart.
    const FLOAT32_EXACT: i64 = 1 << 24;
    /// Rows of each value in [`float32_table`].
    const FLOAT32_ROWS_EACH: usize = 3;

    /// A table of one range-indexed `Float32` column `f` holding the floats
    /// from four below [`FLOAT32_EXACT`] to four above it, two apart.
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

    /// A bound's integer literal is cast to the column's `Float32` as a
    /// statement casts it: `f = 2^24 + 1` keeps the rows at `2^24`, which the
    /// literal rounds to, and `f < 2^24 + 1` keeps none of them. Literals a
    /// `Float32` holds exactly read as they did.
    #[test]
    fn a_bound_compares_with_its_literal_cast_to_the_column() {
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
            let got = st
                .sorted_search_where(Some("f"), &filter, false, 10, Some(&["_id"]))
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

    /// A filter the index cannot be read for is left to the caller's
    /// statement: a column without a range index, two columns with none
    /// named, a condition that keeps no row of the order with no condition
    /// bounding the column (the walk ends, and the rows without a value
    /// are unread), a condition only a statement evaluates, and an equality
    /// on a column whose statistics count thousands of distinct values,
    /// which a walk would need more rows than it reads to fill.
    #[test]
    fn sorted_search_where_declines_what_the_index_cannot_answer() {
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
            let got = st
                .sorted_search_where(*column, filter, false, 10, Some(&["_id", "key"]))
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
