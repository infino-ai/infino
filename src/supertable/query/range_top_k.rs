// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`RangeTopK`]: `ORDER BY column LIMIT k` on a range-indexed column reads
//! the `k` rows the range index puts first, instead of every row the range
//! holds.
//!
//! Without it the scan hands every row of the `WHERE`'s range to the sort,
//! which keeps the best `k`: a range holding half the table decodes half the
//! table to return ten rows. The range index already holds the column in
//! order, so the best `k` of each superfile are its first `k` keys inside
//! the range, a block or two away. The `k`-th best key across all of them is
//! the cutoff no returned row can pass, and the scan then decodes only the
//! rows collected below it:
//!
//! ```text
//!   Sort fetch=10: price ASC            Sort fetch=10: price ASC
//!     Filter price > 990                  Filter price > 990
//!       TableScan                           TableScan, top-k hint (price, asc, 10)
//!         every row with price > 990          per superfile: its first 10 keys
//!                                             above 990, cut at the 10th overall
//! ```
//!
//! The rule only marks the scan, with a provider carrying the hint; the
//! provider does the reading (`SupertableProvider::top_k_rows`), since that
//! needs the superfiles. The cutoff holds only when every filter under the
//! sort is a range on the sort column: a filter on another column could
//! reject the rows the index puts first. Any other filter is walked instead
//! (`SupertableReader::sorted_hits_for_scan`), and a walk that cannot answer
//! exactly leaves the scan to read every row. A sort with nulls first and
//! nothing keeping them out is left alone too, as nulls are in no range index
//! and would come first. The sort may order by more keys after the column
//! (`ORDER BY price, id`, the tie-break a paged search adds): those order only
//! the rows tied on the column, so the scan then reads every row tied with the
//! cutoff too. The one exception is a sort whose only later key is the table's
//! `_id`, ascending: a range index that keeps the rows of a tied key in `_id`
//! order makes a superfile's first rows by the column its first by the column
//! and `_id`, and the scan reads those alone. The sort and the filters stay
//! where they are, so the answer is the one they give over the rows the scan
//! returns.

use std::{slice::from_ref, sync::Arc};

use datafusion::{
    catalog::TableProvider,
    common::{Column, tree_node::Transformed},
    datasource::{DefaultTableSource, provider_as_source},
    error::Result as DfResult,
    logical_expr::{
        Expr, Filter, LogicalPlan, Sort, SortExpr, TableScan, TableSource, utils::split_conjunction,
    },
    optimizer::{OptimizerConfig, OptimizerRule, optimizer::ApplyOrder},
};

use crate::{
    superfile::range::KeyKind,
    supertable::query::{
        candidate::{RangeColumns, single_range},
        exec::common::unqualified,
        provider::{SupertableProvider, TopKHint},
        sorted::MAX_WALKED_ROWS,
    },
};

/// The rule. Registered on the `query_sql` session after DataFusion's
/// built-in rules, which push the `LIMIT` into the sort as its fetch.
#[derive(Debug, Default)]
pub(crate) struct RangeTopK;

impl OptimizerRule for RangeTopK {
    fn name(&self) -> &str {
        "range_top_k"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> DfResult<Transformed<LogicalPlan>> {
        match try_rewrite(&plan)? {
            Some(rewritten) => Ok(Transformed::yes(rewritten)),
            None => Ok(Transformed::no(plan)),
        }
    }
}

/// The sort with its scan's provider swapped for one carrying the hint, or
/// `None` when the plan is not the shape the hint is sound for.
fn try_rewrite(plan: &LogicalPlan) -> DfResult<Option<LogicalPlan>> {
    let Some(SortOverScan {
        first: sort_expr,
        column,
        later,
        fetch: k,
        scan,
        mut predicates,
    }) = sort_over_scan(plan)
    else {
        return Ok(None);
    };
    let Some(provider) = provider_of(scan) else {
        return Ok(None);
    };
    if provider.has_top_k() {
        return Ok(None);
    }
    predicates.extend(scan.filters.iter().flat_map(split_conjunction).cloned());
    let range_cols = provider.manifest().table_schema().range_column_types();
    let id_column = provider.manifest().options.id_column.as_str();
    // Later sort keys (an id that breaks ties, say) order only rows the
    // first key ties; the hint then keeps every row tied with its cutoff.
    let ties_by_id = matches!(
        later,
        [only] if only.asc && matches!(&only.expr, Expr::Column(c) if c.name == id_column)
    );
    // One range on the column takes each file's first keys in it; any other filter takes the walk.
    let walk = match single_range(&predicates, &column.name, &range_cols) {
        Some(_) => None,
        None => match walk_predicates(
            &predicates,
            sort_expr,
            later,
            ties_by_id,
            k,
            &column.name,
            &range_cols,
        )? {
            Some(walk) => Some(walk),
            None => return Ok(None),
        },
    };
    if predicates.is_empty() && sort_expr.nulls_first {
        return Ok(None);
    }
    let hinted = Arc::new(provider.with_top_k(TopKHint {
        column: column.name.clone(),
        descending: !sort_expr.asc,
        k,
        ties: !later.is_empty(),
        ties_by_id,
        walk,
        nulls_first: sort_expr.nulls_first,
    }));
    let source = provider_as_source(hinted as Arc<dyn TableProvider>);
    Ok(Some(with_scan_source(plan, &source)?))
}

/// The filters the walk checks, table qualifiers dropped as its rows name bare columns. `None` when
/// the walk cannot give the sort's exact answer:
///  - the column has no range index, or the sort keeps more rows than a walk reads.
///  - a later sort key other than `_id` ascending orders the ties.
///  - a filter is volatile, so the filter above would give another answer.
///  - rows with no value sort first and no filter on the column keeps them out.
fn walk_predicates(
    predicates: &[Expr],
    first: &SortExpr,
    later: &[SortExpr],
    ties_by_id: bool,
    fetch: usize,
    column: &str,
    range_cols: &RangeColumns,
) -> DfResult<Option<Vec<Expr>>> {
    if fetch > MAX_WALKED_ROWS {
        return Ok(None);
    }
    let indexed = range_cols
        .get(column)
        .is_some_and(|(_, data_type)| KeyKind::of(data_type).is_some());
    if !indexed || (!later.is_empty() && !ties_by_id) || predicates.iter().any(Expr::is_volatile) {
        return Ok(None);
    }
    let bounds_column = predicates
        .iter()
        .any(|p| single_range(from_ref(p), column, range_cols).is_some());
    if first.nulls_first && !bounds_column {
        return Ok(None);
    }
    predicates
        .iter()
        .cloned()
        .map(unqualified)
        .collect::<DfResult<Vec<_>>>()
        .map(Some)
}

/// A sort with a fetch whose first key is a plain column, over filters,
/// plain column projections and aliases down to a scan: the shape a rule
/// that hands the scan the sort's order and limit fires on.
pub(crate) struct SortOverScan<'a> {
    /// The sort's first key.
    pub(crate) first: &'a SortExpr,
    /// The column the first key sorts by.
    pub(crate) column: &'a Column,
    /// The sort's other keys, which order only rows the first one ties.
    pub(crate) later: &'a [SortExpr],
    /// The rows the sort keeps.
    pub(crate) fetch: usize,
    /// The scan at the bottom.
    pub(crate) scan: &'a TableScan,
    /// The conjuncts of every filter between the sort and the scan.
    pub(crate) predicates: Vec<Expr>,
}

/// `plan` as a [`SortOverScan`], or `None` when it is not one.
pub(crate) fn sort_over_scan(plan: &LogicalPlan) -> Option<SortOverScan<'_>> {
    let LogicalPlan::Sort(Sort {
        expr,
        input,
        fetch: Some(fetch),
    }) = plan
    else {
        return None;
    };
    let (first, later) = expr.split_first()?;
    let Expr::Column(column) = &first.expr else {
        return None;
    };
    let mut predicates: Vec<Expr> = Vec::new();
    let mut node = input.as_ref();
    let scan = loop {
        match node {
            LogicalPlan::Projection(projection)
                if projection.expr.iter().all(|e| matches!(e, Expr::Column(_))) =>
            {
                node = projection.input.as_ref();
            }
            LogicalPlan::Filter(Filter {
                predicate, input, ..
            }) => {
                predicates.extend(split_conjunction(predicate).into_iter().cloned());
                node = input.as_ref();
            }
            LogicalPlan::SubqueryAlias(alias) => node = alias.input.as_ref(),
            LogicalPlan::TableScan(scan) => break scan,
            _ => return None,
        }
    };
    Some(SortOverScan {
        first,
        column,
        later,
        fetch: *fetch,
        scan,
        predicates,
    })
}

/// Our provider behind `scan`, when it is ours.
pub(crate) fn provider_of(scan: &TableScan) -> Option<&SupertableProvider> {
    let source = scan.source.downcast_ref::<DefaultTableSource>()?;
    source.table_provider.downcast_ref::<SupertableProvider>()
}

/// `plan` with the scan at the bottom of its single-input chain reading
/// from `source`.
pub(crate) fn with_scan_source(
    plan: &LogicalPlan,
    source: &Arc<dyn TableSource>,
) -> DfResult<LogicalPlan> {
    if let LogicalPlan::TableScan(scan) = plan {
        return Ok(LogicalPlan::TableScan(TableScan {
            source: Arc::clone(source),
            ..scan.clone()
        }));
    }
    let inputs = plan
        .inputs()
        .into_iter()
        .map(|input| with_scan_source(input, source))
        .collect::<DfResult<Vec<_>>>()?;
    plan.with_new_exprs(plan.expressions(), inputs)
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread::spawn,
    };

    use arrow::util::display::array_value_to_string;
    use arrow_array::{Array, Int64Array, LargeStringArray, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        common::tree_node::{TreeNode, TreeNodeRecursion},
        logical_expr::{Expr, LogicalPlan},
        prelude::{col, lit},
    };
    use roaring::RoaringBitmap;
    use tempfile::TempDir;
    use tokio::runtime::Builder;
    use uuid::Uuid;

    use crate::{
        config::OptimizeOptions,
        memory::ConnectionMemoryBudget,
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::builder::FtsConfig,
        supertable::{
            Supertable, SupertableOptions, SupertableReader,
            error::QueryError,
            query::{
                column_cache::DecodedColumnCache,
                provider::{TopKHint, WalkFallback, tests::planned_files, walk_probe},
                range_top_k::provider_of,
                scalar_cache::ByteLru,
            },
        },
    };

    /// Superfiles the table is written as, one commit each.
    const FILES: i64 = 3;
    /// Rows in each.
    const ROWS_PER_FILE: i64 = 4000;
    /// Distinct `key` values: each is held by `FILES * ROWS_PER_FILE /
    /// KEYS` rows, so every cutoff ties on many rows.
    const KEYS: i64 = 100;
    /// Rows per row group, so a file has several and the index's rows fall
    /// in some of them only.
    const ROW_GROUP_ROWS: usize = 500;
    /// Characters of `pad`, a column wide beside `key` and `seq`.
    const PAD_LEN: usize = 48;
    /// Rows the delete removes: those with `seq` below this.
    const DELETED_BELOW: i64 = 300;
    /// Kept bytes: room for every column of every row group.
    const KEPT_BYTES: usize = 64 << 20;
    /// Below every `key`: a cutoff that keeps all of them.
    const BELOW_EVERY_KEY: i64 = -1;
    /// Rows the `_id` tie-break test also deletes: those with `seq` from
    /// this on, a larger share of the last superfile than the first delete
    /// takes of the first, so an optimize merges the last superfile first
    /// and the merged rows are not in `_id` order.
    const LATE_DELETED_FROM: i64 = 11_000;
    /// Top-k statements whose ties the table's `_id` breaks, ascending, as
    /// `(key above, descending, limit)`; no key bound means no `WHERE`, as a
    /// search with no query sends.
    const BY_ID: &[(Option<i64>, bool, usize)] = &[
        (Some(40), false, 10),
        (Some(40), true, 25),
        (Some(90), false, 130),
        (None, true, 7),
        (None, false, 45),
        (None, true, 250),
    ];
    /// The tie-broken top-k statements, and the `(key above, descending,
    /// limit)` each asks for.
    const TOP_K: &[(&str, i64, bool, usize)] = &[
        (
            "SELECT seq, key FROM supertable WHERE key > 40 ORDER BY key, seq LIMIT 10",
            40,
            false,
            10,
        ),
        (
            "SELECT seq, key FROM supertable WHERE key > 40 ORDER BY key DESC, seq DESC LIMIT 25",
            40,
            true,
            25,
        ),
        (
            "SELECT seq, key, pad FROM supertable WHERE key > 90 ORDER BY key ASC NULLS LAST, seq ASC NULLS LAST LIMIT 130",
            90,
            false,
            130,
        ),
    ];

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
            Field::new("pad", DataType::LargeUtf8, false),
        ]))
    }

    /// `key` of row `seq`: not in the order the rows arrive.
    fn key(seq: i64) -> i64 {
        (seq * 37) % KEYS
    }

    /// The range-indexed table, keeping decoded columns, its rows below
    /// [`DELETED_BELOW`] deleted.
    fn table(dir: &TempDir) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let mut options = SupertableOptions::new(schema(), vec![], vec![])
            .expect("options")
            .with_storage(storage)
            .with_range_indexes(&["key".to_string()]);
        options.decoded_columns = Some(Arc::new(ByteLru::new(KEPT_BYTES)));
        options.row_group_size = Some(ROW_GROUP_ROWS);
        let st = Supertable::create(options).expect("create");
        for file in 0..FILES {
            let seqs: Vec<i64> = (file * ROWS_PER_FILE..(file + 1) * ROWS_PER_FILE).collect();
            let keys: Vec<i64> = seqs.iter().map(|s| key(*s)).collect();
            let pads: Vec<String> = seqs
                .iter()
                .map(|s| format!("{s:0>width$}", width = PAD_LEN))
                .collect();
            let batch = RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(LargeStringArray::from(pads)),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st.delete(col("seq").lt(lit(DELETED_BELOW)))
            .expect("delete");
        st
    }

    /// The `seq`s a top-k over the live rows answers, in its order.
    fn expected(above: i64, descending: bool, limit: usize) -> Vec<i64> {
        let mut rows: Vec<(i64, i64)> = (DELETED_BELOW..FILES * ROWS_PER_FILE)
            .map(|s| (key(s), s))
            .filter(|(k, _)| *k > above)
            .collect();
        rows.sort();
        if descending {
            rows.reverse();
        }
        rows.into_iter().take(limit).map(|(_, s)| s).collect()
    }

    /// The `seq`s a top-k over the live rows (with those from
    /// [`LATE_DELETED_FROM`] deleted too) answers when it orders by `key`
    /// and then `seq` ascending, whichever way `key` goes.
    fn expected_then_seq(above: i64, descending: bool, limit: usize) -> Vec<i64> {
        let mut rows: Vec<(i64, i64)> = (DELETED_BELOW..LATE_DELETED_FROM)
            .map(|s| (key(s), s))
            .filter(|(k, _)| *k > above)
            .collect();
        rows.sort_by(|a, b| {
            let by_key = match descending {
                false => a.0.cmp(&b.0),
                true => b.0.cmp(&a.0),
            };
            by_key.then(a.1.cmp(&b.1))
        });
        rows.into_iter().take(limit).map(|(_, s)| s).collect()
    }

    /// Assert every superfile of `reader`'s snapshot keeps the rows tied on
    /// `key` in `_id` order in its range index, which is what lets a top-k
    /// tie-broken by `_id` read only each superfile's first rows.
    fn assert_ties_in_id_order(st: &Supertable, reader: &SupertableReader, what: &str) {
        let manifest = reader.manifest();
        let field = manifest.field_id("key").expect("key");
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        for entry in &manifest.superfiles {
            let file = st.options().store.reader(&entry.uri).expect("superfile");
            let ends = runtime
                .block_on(file.range_ends(
                    field,
                    &DataType::Int64,
                    (0, u64::MAX),
                    1,
                    false,
                    &RoaringBitmap::new(),
                ))
                .expect("ends")
                .expect("key is indexed");
            assert!(ends.ties_in_id_order, "{what}: {:?}", entry.uri);
        }
    }

    /// The `seq` column across `batches`, in order.
    fn seqs(batches: &[RecordBatch]) -> Vec<i64> {
        batches
            .iter()
            .flat_map(|b| {
                let seqs = b
                    .column_by_name("seq")
                    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                    .expect("seq")
                    .clone();
                (0..seqs.len()).map(move |i| seqs.value(i))
            })
            .collect()
    }

    /// The `EXPLAIN` of `sql` on `reader`, every plan line joined.
    fn explained(reader: &SupertableReader, sql: &str) -> String {
        reader
            .query_sql(&format!("EXPLAIN {sql}"))
            .expect("explain")
            .iter()
            .flat_map(|b| {
                let plans = b
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("plan column")
                    .clone();
                (0..plans.len()).map(move |i| plans.value(i).to_owned())
            })
            .collect()
    }

    /// A top-k whose ties a later key breaks reads the rows the range index
    /// chose from kept columns, and answers exactly: on the pass that decodes
    /// the columns and on the ones that read them kept, deleted rows left out.
    #[test]
    fn a_tie_broken_top_k_reads_the_chosen_rows_from_kept_columns() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        let reader = st.reader().expect("reader");
        let plan = explained(&reader, TOP_K[0].0);
        assert!(plan.contains("CachedScanExec"), "{plan}");
        for pass in ["decoded", "kept", "kept again"] {
            for (sql, above, descending, limit) in TOP_K {
                assert_eq!(
                    seqs(&reader.query_sql(sql).expect("sql")),
                    expected(*above, *descending, *limit),
                    "{pass}: {sql}"
                );
            }
        }
    }

    /// A top-k whose ties the table's `_id` breaks answers as a sort by the
    /// key and then `seq` does (the rows were appended in `seq` order, so
    /// their ids ascend with it), deleted rows left out, whichever way the
    /// key goes, before and after an optimize merges the superfiles in an
    /// order other than `_id`'s; and every superfile's range index keeps its
    /// tied rows in `_id` order, so the scan reads each one's first rows.
    #[test]
    fn a_top_k_tie_broken_by_the_id_answers_from_each_superfiles_first_rows() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        st.delete(col("seq").gt_eq(lit(LATE_DELETED_FROM)))
            .expect("delete");
        for pass in ["appended", "optimized"] {
            if pass == "optimized" {
                st.optimize(&OptimizeOptions::default()).expect("optimize");
            }
            let reader = st.reader().expect("reader");
            assert_ties_in_id_order(&st, &reader, pass);
            for (above, descending, limit) in BY_ID {
                let filter = above.map_or(String::new(), |a| format!("WHERE key > {a} "));
                let order = if *descending { "DESC" } else { "ASC" };
                let sql = format!(
                    "SELECT seq, key FROM supertable {filter}ORDER BY key {order} NULLS LAST, _id ASC LIMIT {limit}"
                );
                assert_eq!(
                    seqs(&reader.query_sql(&sql).expect("sql")),
                    expected_then_seq(above.unwrap_or(BELOW_EVERY_KEY), *descending, *limit),
                    "{pass}: {sql}"
                );
            }
        }
    }

    /// A top-k over an aliased table answers as over the table itself.
    #[test]
    fn a_top_k_through_a_table_alias_answers_exactly() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        let reader = st.reader().expect("reader");
        let (_, above, descending, limit) = TOP_K[0];
        let sql = format!(
            "SELECT s.seq, s.key FROM supertable AS s WHERE s.key > {above} \
             ORDER BY s.key, s.seq LIMIT {limit}"
        );
        assert_eq!(
            seqs(&reader.query_sql(&sql).expect("sql")),
            expected(above, descending, limit),
            "{sql}"
        );
    }

    // The walk: a sort whose filter is more than one range on its column.

    /// Rows in each superfile of the tagged table.
    const TAGGED_ROWS_PER_FILE: i64 = 2000;
    /// Superfiles of the tagged table.
    const TAGGED_FILES: i64 = 4;
    /// Every `RARE_EVERY`-th row is tagged `rare`: too few for the walk, so the scan reads them.
    const RARE_EVERY: i64 = 337;
    /// Every `KERNEL_EVERY`-th other row is tagged `kernel` (about 14%).
    const KERNEL_EVERY: i64 = 7;
    /// Every `B_EVERY`-th other row is tagged `b`.
    const B_EVERY: i64 = 3;
    /// Rows whose `seq` leaves this remainder by [`NULL_EVERY`] have no `opt`.
    const NULL_AT: i64 = 3;
    /// One row in this many has no `opt`.
    const NULL_EVERY: i64 = 10;
    /// Distinct `opt` values: ties at every cutoff.
    const OPT_KEYS: i64 = 1000;
    /// The tagged table deletes the rows with `seq` below 150, and every multiple of 11.
    const TAGGED_DELETED_BELOW: i64 = 150;
    const DELETED_EVERY: i64 = 11;

    fn tagged_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
            Field::new("opt", DataType::Int64, true),
            Field::new("tag", DataType::LargeUtf8, false),
            Field::new("pad", DataType::LargeUtf8, false),
        ]))
    }

    fn tag(seq: i64) -> &'static str {
        if seq % RARE_EVERY == 0 {
            "rare"
        } else if seq % KERNEL_EVERY == 0 {
            "kernel"
        } else if seq % B_EVERY == 0 {
            "b"
        } else {
            "c"
        }
    }

    fn opt(seq: i64) -> Option<i64> {
        (seq % NULL_EVERY != NULL_AT).then_some((seq * 53) % OPT_KEYS)
    }

    /// Range-indexed on `key` and `opt`, with deleted rows in every file.
    fn tagged_table(dir: &TempDir) -> Supertable {
        tagged_table_with(dir, None, TAGGED_FILES, TAGGED_ROWS_PER_FILE)
    }

    /// [`tagged_table`] keeping decoded columns in `kept` when given.
    fn tagged_table_with(
        dir: &TempDir,
        kept: Option<Arc<DecodedColumnCache>>,
        files: i64,
        rows: i64,
    ) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let mut options = SupertableOptions::new(tagged_schema(), vec![], vec![])
            .expect("options")
            .with_storage(storage)
            .with_range_indexes(&["key".to_string(), "opt".to_string()]);
        options.decoded_columns = kept;
        options.row_group_size = Some(ROW_GROUP_ROWS);
        let st = Supertable::create(options).expect("create");
        for file in 0..files {
            let seqs: Vec<i64> = (file * rows..(file + 1) * rows).collect();
            let batch = RecordBatch::try_new(
                tagged_schema(),
                vec![
                    Arc::new(Int64Array::from(seqs.clone())),
                    Arc::new(Int64Array::from_iter_values(seqs.iter().map(|s| key(*s)))),
                    Arc::new(Int64Array::from_iter(seqs.iter().map(|s| opt(*s)))),
                    Arc::new(LargeStringArray::from_iter_values(
                        seqs.iter().map(|s| tag(*s)),
                    )),
                    Arc::new(LargeStringArray::from_iter_values(
                        seqs.iter()
                            .map(|s| format!("{s:0>width$}", width = PAD_LEN)),
                    )),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st.delete(
            col("seq")
                .lt(lit(TAGGED_DELETED_BELOW))
                .or((col("seq") % lit(DELETED_EVERY)).eq(lit(0))),
        )
        .expect("delete");
        st
    }

    /// The rows the scan under `sql` emitted, from its `EXPLAIN ANALYZE`.
    fn scanned_rows(reader: &SupertableReader, sql: &str) -> u64 {
        let plan = reader
            .query_sql(&format!("EXPLAIN ANALYZE {sql}"))
            .expect("explain analyze");
        let text: String = plan
            .iter()
            .flat_map(|b| {
                let plans = b
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("plan column")
                    .clone();
                (0..plans.len()).map(move |i| plans.value(i).to_owned())
            })
            .collect();
        let line = text
            .lines()
            .find(|l| l.contains("DataSourceExec") || l.contains("CachedScanExec"))
            .expect("a scan node");
        let rows = line.split("output_rows=").nth(1).expect("scan output_rows");
        let number: String = rows
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        let value: f64 = number.parse().expect("row count");
        let scale = match rows[number.len()..].trim_start().chars().next() {
            Some('K') => 1e3,
            Some('M') => 1e6,
            _ => 1.0,
        };
        (value * scale) as u64
    }

    /// The column `name` across `batches` as text, `NULL` for a null.
    fn column_text(batches: &[RecordBatch], name: &str) -> Vec<String> {
        batches
            .iter()
            .flat_map(|b| {
                let column = b.column_by_name(name).expect("column").clone();
                (0..column.len()).map(move |i| match column.is_null(i) {
                    true => "NULL".to_owned(),
                    false => array_value_to_string(&column, i).expect("display"),
                })
            })
            .collect()
    }

    /// Filters for the walk, and whether it reads the rows (`true`) or leaves them to the scan.
    const WALK_FILTERS: &[(&str, bool)] = &[
        ("tag = 'kernel'", true),
        ("tag = 'kernel' AND {col} > 40", true),
        ("tag IN ('kernel', 'b')", true),
        ("(tag = 'kernel' OR seq > 7000) AND {col} >= 10", true),
        ("tag = 'kernel' AND length(pad) = {pad_len}", true),
        ("tag = 'rare'", false),
        ("tag = 'nowhere'", false),
        ("tag = 'kernel' AND random() < 2.0", false),
    ];

    #[test]
    fn a_filtered_top_k_answers_as_the_sort_without_the_walk() {
        // The tagged table (deleted rows, ties, nulls), 8 filters x 5 orders x 5 limits on `key`
        // and `opt`, before and after optimize.
        //  - Each statement answers what the same sort answers with the walk out of the plan
        //    (`col + 0`, which no rule reads from an index).
        //  - Where the walk reads the rows, the scan emits and plans no more than the page.
        // This test pins that the walk changes how a page is read, never what it answers.
        let dir = TempDir::new().expect("tempdir");
        let st = tagged_table(&dir);
        let orders = [
            ("ASC NULLS LAST", ", _id ASC"),
            ("DESC NULLS LAST", ", _id ASC"),
            ("ASC", ""),
            ("DESC", ", _id ASC"),
            ("ASC NULLS LAST", ", seq DESC"),
        ];
        let limits = [
            "LIMIT 1",
            "LIMIT 10",
            "LIMIT 130",
            "LIMIT 10 OFFSET 5",
            "LIMIT 9000",
        ];
        let mut walked = 0;
        for pass in ["appended", "optimized"] {
            if pass == "optimized" {
                st.optimize(&OptimizeOptions::default()).expect("optimize");
            }
            let reader = st.reader().expect("reader");
            for column in ["key", "opt"] {
                for (filter, walks) in WALK_FILTERS {
                    let filter = filter
                        .replace("{col}", column)
                        .replace("{pad_len}", &PAD_LEN.to_string());
                    for (direction, later) in orders {
                        for limit in limits {
                            let sql = |sort: &str| {
                                format!(
                                    "SELECT seq, {column} FROM supertable WHERE {filter} \
                                     ORDER BY {sort} {direction}{later} {limit}"
                                )
                            };
                            let with_walk = sql(column);
                            let without = sql(&format!("({column} + 0)"));
                            let got = reader.query_sql(&with_walk).expect("sql");
                            let want = reader.query_sql(&without).expect("sql");
                            let what = format!("{pass}: {with_walk}");
                            // Ties the statement leaves unbroken may come in any order.
                            match later.is_empty() {
                                true => assert_eq!(
                                    column_text(&got, column),
                                    column_text(&want, column),
                                    "{what}"
                                ),
                                false => assert_eq!(seqs(&got), seqs(&want), "{what}"),
                            }
                            // The walk reads the rows only where its answer is exact:
                            // `_id` (or nothing) after the key, nulls last or kept out.
                            let bounded = filter.contains(column);
                            let takes = *walks
                                && later != ", seq DESC"
                                && (direction.contains("NULLS LAST")
                                    || direction == "ASC"
                                    || bounded)
                                && limit != "LIMIT 9000";
                            if takes {
                                let fetch: u64 = match limit {
                                    "LIMIT 10 OFFSET 5" => 15,
                                    other => other["LIMIT ".len()..].parse().expect("limit"),
                                };
                                let scanned = scanned_rows(&reader, &with_walk);
                                assert!(scanned <= fetch, "{what}: scanned {scanned}");
                                // A superfile holding none of the handed rows is not scanned.
                                let files = planned_files(&reader, &with_walk) as u64;
                                assert!(files <= fetch.max(1), "{what}: planned {files} files");
                                walked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(walked > 0, "no statement went through the walk");
    }

    /// The hint the optimized plan of `sql` puts on its scan, if any.
    fn hint_of(reader: &SupertableReader, sql: &str) -> Option<TopKHint> {
        let ctx = reader.sql_session_context().expect("session");
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let plan = runtime
            .block_on(ctx.state().create_logical_plan(sql))
            .expect("plan");
        let optimized = ctx.state().optimize(&plan).expect("optimize");
        let mut hint = None;
        optimized
            .apply(|node| {
                if let LogicalPlan::TableScan(scan) = node {
                    hint = provider_of(scan).and_then(|p| p.top_k_hint().cloned());
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .expect("walk the plan");
        hint
    }

    #[test]
    fn the_rule_hands_the_walk_only_the_sorts_it_answers_exactly() {
        // One sort the walk answers, and five it cannot answer exactly:
        //  - a volatile filter, a later key other than `_id` ascending, nulls first with nothing
        //    on the column keeping them out, a column with no range index, a limit past the walk's.
        // This test pins that the rule declines those up front, not through the walk's fallback.
        let dir = TempDir::new().expect("tempdir");
        let st = tagged_table(&dir);
        let reader = st.reader().expect("reader");
        let taken = "SELECT seq FROM supertable WHERE tag = 'kernel' \
                     ORDER BY opt ASC NULLS LAST, _id ASC LIMIT 10";
        let hint = hint_of(&reader, taken).expect("a hint");
        assert!(hint.walk.is_some(), "{taken}: {hint:?}");
        for declined in [
            "SELECT seq FROM supertable WHERE tag = 'kernel' AND random() < 2.0 \
             ORDER BY opt ASC NULLS LAST, _id ASC LIMIT 10",
            "SELECT seq FROM supertable WHERE tag = 'kernel' \
             ORDER BY opt ASC NULLS LAST, seq DESC LIMIT 10",
            "SELECT seq FROM supertable WHERE tag = 'kernel' \
             ORDER BY opt DESC NULLS FIRST, _id ASC LIMIT 10",
            "SELECT seq FROM supertable WHERE tag = 'kernel' \
             ORDER BY seq ASC NULLS LAST, _id ASC LIMIT 10",
            "SELECT seq FROM supertable WHERE tag = 'kernel' \
             ORDER BY opt ASC NULLS LAST, _id ASC LIMIT 5000",
        ] {
            assert_eq!(hint_of(&reader, declined), None, "{declined}");
        }
    }

    #[test]
    fn a_walk_that_keeps_no_row_scans_nothing() {
        // This test pins that a walk keeping no row answers empty and plans no file.
        let dir = TempDir::new().expect("tempdir");
        let st = tagged_table(&dir);
        let reader = st.reader().expect("reader");
        // `kernel` rows are the multiples of KERNEL_EVERY, so none leaves 1.
        let sql = format!(
            "SELECT seq, opt FROM supertable WHERE opt BETWEEN 0 AND 50 AND tag = 'kernel' \
             AND seq % {KERNEL_EVERY} = 1 ORDER BY opt ASC NULLS LAST, _id ASC LIMIT 10"
        );
        assert!(
            hint_of(&reader, &sql).is_some_and(|hint| hint.walk.is_some()),
            "{sql}"
        );
        assert!(
            seqs(&reader.query_sql(&sql).expect("sql")).is_empty(),
            "{sql}"
        );
        assert_eq!(planned_files(&reader, &sql), 0, "{sql}");
    }

    #[test]
    fn a_delete_between_the_walk_and_the_scan_leaves_k_rows() {
        // A delete lands after the walk found its rows and before the scan reads them.
        //  - The walk's rows are checked against the scan's deletes; one is gone, so the scan
        //    reads every row.
        //  - The statement answers the same 10 rows as one run after the delete.
        // This test pins that a delete between the walk and the scan never shortens the page.
        let dir = TempDir::new().expect("tempdir");
        let st = tagged_table(&dir);
        let sql = "SELECT seq FROM supertable WHERE tag = 'kernel' \
                   ORDER BY opt ASC NULLS LAST, _id ASC LIMIT 10";
        let before = seqs(&st.reader().expect("reader").query_sql(sql).expect("sql"));
        assert_eq!(before.len(), 10, "{sql}");
        let reader = st.reader().expect("reader");
        let superfile = reader.manifest().superfiles[0].superfile_id;
        let deleting = st.clone();
        let first = before[0];
        let fired = Arc::new(AtomicBool::new(false));
        let fired_in_hook = Arc::clone(&fired);
        walk_probe::watch(superfile);
        walk_probe::after_next_walk(
            superfile,
            Box::new(move || {
                // A thread of its own: the delete drives its own runtime.
                spawn(move || deleting.delete(col("seq").eq(lit(first))))
                    .join()
                    .expect("delete thread")
                    .expect("delete");
                fired_in_hook.store(true, Ordering::SeqCst);
            }),
        );
        let got = seqs(&reader.query_sql(sql).expect("sql"));
        assert!(
            fired.load(Ordering::SeqCst),
            "the walk ran and the delete landed"
        );
        let after = seqs(&st.reader().expect("reader").query_sql(sql).expect("sql"));
        assert!(!after.contains(&first), "{after:?}");
        assert_eq!(got, after, "{sql}");
        assert_eq!(
            walk_probe::fallbacks(superfile),
            vec![WalkFallback::RowDeleted],
            "the check declined the walk's rows once"
        );
    }

    #[test]
    fn a_walk_over_kept_columns_answers_as_the_sort_without_it() {
        // A walked sort over kept decoded columns, on the pass that decodes them and the one that
        // reads them kept.
        //  - Both answer as the sort without the walk.
        //  - A walk that keeps no row answers empty and plans no file.
        // This test pins the walk through the kept-columns scan.
        let dir = TempDir::new().expect("tempdir");
        let kept = Some(Arc::new(ByteLru::new(KEPT_BYTES)));
        let st = tagged_table_with(&dir, kept, TAGGED_FILES, TAGGED_ROWS_PER_FILE);
        let reader = st.reader().expect("reader");
        let sql = |filter: &str, sort: &str| {
            format!(
                "SELECT seq, pad FROM supertable WHERE {filter} \
                 ORDER BY {sort} ASC NULLS LAST, _id ASC LIMIT 10"
            )
        };
        let kernel = "tag = 'kernel'";
        assert!(
            explained(&reader, &sql(kernel, "opt")).contains("CachedScanExec"),
            "{}",
            sql(kernel, "opt")
        );
        let want = seqs(&reader.query_sql(&sql(kernel, "(opt + 0)")).expect("sql"));
        assert_eq!(want.len(), 10);
        for pass in ["decoded", "kept"] {
            let got = seqs(&reader.query_sql(&sql(kernel, "opt")).expect("sql"));
            assert_eq!(got, want, "{pass}");
        }
        let none = format!("opt BETWEEN 0 AND 50 AND {kernel} AND seq % {KERNEL_EVERY} = 1");
        assert!(seqs(&reader.query_sql(&sql(&none, "opt")).expect("sql")).is_empty());
        assert_eq!(planned_files(&reader, &sql(&none, "opt")), 0);
    }

    #[test]
    fn a_walk_that_fails_falls_back_to_the_scan_and_says_why() {
        // The walk is made to fail, first over the memory limit, then with any other error.
        //  - The scan answers as the sort without the walk.
        //  - The fallback names the cause.
        // This test pins that a failed walk costs speed, never the answer.
        let dir = TempDir::new().expect("tempdir");
        let st = tagged_table(&dir);
        let reader = st.reader().expect("reader");
        let superfile = reader.manifest().superfiles[0].superfile_id;
        walk_probe::watch(superfile);
        let sql = |sort: &str| {
            format!(
                "SELECT seq FROM supertable WHERE tag = 'kernel' \
                 ORDER BY {sort} ASC NULLS LAST, _id ASC LIMIT 10"
            )
        };
        let want = seqs(&reader.query_sql(&sql("(opt + 0)")).expect("sql"));
        for (error, cause) in [
            (
                QueryError::OverBudget("injected".to_owned()),
                WalkFallback::OverBudget,
            ),
            (QueryError::internal("injected"), WalkFallback::Failed),
        ] {
            walk_probe::fail_next_walk(superfile, error);
            let got = seqs(&reader.query_sql(&sql("opt")).expect("sql"));
            assert_eq!(got, want, "{cause:?}");
            assert_eq!(walk_probe::fallbacks(superfile).last(), Some(&cause));
        }
    }

    /// The tagged table in `dir`, opened again with a connection memory limit of `bytes`.
    fn reopened_with_budget(dir: &TempDir, bytes: u64) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let mut options = SupertableOptions::new(tagged_schema(), vec![], vec![])
            .expect("options")
            .with_storage(storage)
            .with_range_indexes(&["key".to_string(), "opt".to_string()]);
        options.row_group_size = Some(ROW_GROUP_ROWS);
        options.connection_memory_budget = ConnectionMemoryBudget::with_limit(bytes);
        Supertable::open(options).expect("open")
    }

    #[test]
    fn a_walk_window_the_memory_budget_cannot_hold_is_refused() {
        // The tagged table under a 256 KiB connection memory limit.
        //  - A walk for 10 `kernel` rows (1 row in 7) fits its window in the limit and answers.
        //  - A walk for 4,000 rows, under `MAX_WALKED_ROWS` so only the limit refuses it, is
        //    refused before its window is read.
        // This test pins that a walk reserves its window against the connection's memory limit.
        let dir = TempDir::new().expect("tempdir");
        tagged_table(&dir);
        let st = reopened_with_budget(&dir, 256 << 10);
        let reader = st.reader().expect("reader");
        let files = reader.manifest().superfiles.clone();
        let walk = |k: usize, filter: Expr| {
            reader.block_on(reader.sorted_hits_for_scan("opt", &[filter], false, false, k, &files))
        };
        let small = walk(10, col("tag").eq(lit("kernel")));
        assert!(matches!(small, Ok(Some(_))), "{small:?}");
        let large = walk(4000, col("seq").gt_eq(lit(0_i64)));
        assert!(matches!(large, Err(QueryError::OverBudget(_))), "{large:?}");
    }

    #[test]
    fn a_top_k_no_file_survives_never_starts_the_walk() {
        // This test pins that a filter the statistics prune to no file never starts the walk.
        let dir = TempDir::new().expect("tempdir");
        let st = tagged_table(&dir);
        let reader = st.reader().expect("reader");
        let superfile = reader.manifest().superfiles[0].superfile_id;
        walk_probe::watch(superfile);
        // Past every `seq`: min/max statistics prune every superfile.
        let past_every_seq = TAGGED_FILES * TAGGED_ROWS_PER_FILE;
        let sql = format!(
            "SELECT seq FROM supertable WHERE tag = 'kernel' AND seq > {past_every_seq} \
             ORDER BY opt ASC NULLS LAST, _id ASC LIMIT 10"
        );
        assert!(
            hint_of(&reader, &sql).is_some_and(|hint| hint.walk.is_some()),
            "{sql}"
        );
        assert!(
            seqs(&reader.query_sql(&sql).expect("sql")).is_empty(),
            "{sql}"
        );
        assert!(explained(&reader, &sql).contains("EmptyExec"), "{sql}");
        assert_eq!(walk_probe::walks(superfile), 0, "the walk never started");
    }

    #[test]
    fn a_filtered_top_k_over_many_files_answers_as_the_sort_without_the_walk() {
        // 40 files of 100 rows, each key's rows in every file, with deletes and nulls.
        //  - Each walked statement, both directions, matches the same sort with the walk out of
        //    the plan (`col + 0`).
        //  - The walk answers every one of them; none falls back.
        // This test pins the walk's cursor across many files through SQL.
        let dir = TempDir::new().expect("tempdir");
        let st = tagged_table_with(&dir, None, 40, 100);
        let reader = st.reader().expect("reader");
        let superfile = reader.manifest().superfiles[0].superfile_id;
        walk_probe::watch(superfile);
        for filter in ["tag = 'kernel'", "tag IN ('kernel', 'b')"] {
            for column in ["key", "opt"] {
                for direction in ["ASC NULLS LAST", "DESC NULLS LAST"] {
                    for limit in [1, 10, 100] {
                        let sql = |sort: &str| {
                            format!(
                                "SELECT seq FROM supertable WHERE {filter} \
                                 ORDER BY {sort} {direction}, _id ASC LIMIT {limit}"
                            )
                        };
                        let got = seqs(&reader.query_sql(&sql(column)).expect("sql"));
                        let without = sql(&format!("({column} + 0)"));
                        let want = seqs(&reader.query_sql(&without).expect("sql"));
                        assert_eq!(got.len(), limit, "{}", sql(column));
                        assert_eq!(got, want, "{}", sql(column));
                    }
                }
            }
        }
        assert_eq!(walk_probe::walks(superfile), 24, "every statement walked");
        assert_eq!(walk_probe::fallbacks(superfile), vec![], "none fell back");
    }

    #[test]
    fn explain_says_whether_the_walk_answered_or_why_it_fell_back() {
        // A walked sort, then the same sort with its walk made to fail two ways.
        //  - The scan's `EXPLAIN` line says the walk answered, or that it fell back and why.
        // This test pins the walk's outcome in `EXPLAIN`.
        let dir = TempDir::new().expect("tempdir");
        let st = tagged_table(&dir);
        let reader = st.reader().expect("reader");
        let superfile = reader.manifest().superfiles[0].superfile_id;
        walk_probe::watch(superfile);
        let sql = "SELECT seq FROM supertable WHERE tag = 'kernel' \
                   ORDER BY opt ASC NULLS LAST, _id ASC LIMIT 10";
        let plan = explained(&reader, sql);
        assert!(plan.contains("top-k walk: answered"), "{plan}");
        for (error, note) in [
            (
                QueryError::OverBudget("injected".to_owned()),
                "top-k walk: fell back, a window over the memory limit",
            ),
            (
                QueryError::internal("injected"),
                "top-k walk: fell back, it failed",
            ),
        ] {
            walk_probe::fail_next_walk(superfile, error);
            let plan = explained(&reader, sql);
            assert!(plan.contains(note), "{plan}");
        }
    }

    /// The needle table is 20 files of 500 rows, enough that a rare filter's page is past the
    /// walk's 4,096-row limit.
    const NEEDLE_FILES: i64 = 20;
    const NEEDLE_ROWS: i64 = 500;
    /// `needle` is in one row of each of the last 9 files: their 4,500 rows are past the limit.
    const NEEDLE_FROM_FILE: i64 = 11;
    /// `pair` is in every 10th row of the last 2 files, so it fills a page within them.
    const PAIR_FROM_FILE: i64 = 18;
    /// `lone` is in one row of the last file: its page needs more rows than the limit, but the
    /// file holds only 500.
    const LONE_FILE: i64 = NEEDLE_FILES - 1;
    /// `seq` from here keeps the last 7 files, 3,500 rows, with `lone` 3,250 rows in, ascending.
    const LONE_BOUND: i64 = 6_500;

    /// The needle table: `seq` range indexed, `msg` full-text indexed, `common` every second row.
    fn needle_table(dir: &TempDir) -> Supertable {
        let schema = Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("msg", DataType::LargeUtf8, false),
        ]));
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options =
            SupertableOptions::new(Arc::clone(&schema), vec![FtsConfig::new("msg")], vec![])
                .expect("options")
                .with_storage(storage)
                .with_range_indexes(&["seq".to_string()]);
        let st = Supertable::create(options).expect("create");
        for file in 0..NEEDLE_FILES {
            let seqs: Vec<i64> = (file * NEEDLE_ROWS..(file + 1) * NEEDLE_ROWS).collect();
            let msgs: Vec<String> = seqs
                .iter()
                .map(|s| {
                    let mut msg = format!("w{s}");
                    for (word, holds) in [
                        (
                            "needle",
                            file >= NEEDLE_FROM_FILE && s % NEEDLE_ROWS == NEEDLE_ROWS / 2,
                        ),
                        ("pair", file >= PAIR_FROM_FILE && s % 10 == 0),
                        (
                            "lone",
                            file == LONE_FILE && s % NEEDLE_ROWS == NEEDLE_ROWS / 2,
                        ),
                        ("common", s % 2 == 0),
                    ] {
                        if holds {
                            msg.push(' ');
                            msg.push_str(word);
                        }
                    }
                    msg
                })
                .collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(LargeStringArray::from(msgs)),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st
    }

    /// `seq` of `filter`'s page in `direction`, through the walk and through the sort without it.
    fn needle_pages(reader: &SupertableReader, filter: &str, direction: &str) -> [Vec<i64>; 2] {
        ["seq", "(seq + 0)"].map(|sort| {
            let sql = format!(
                "SELECT seq FROM supertable WHERE {filter} \
                 ORDER BY {sort} {direction} NULLS LAST, _id ASC LIMIT 10"
            );
            seqs(&reader.query_sql(&sql).expect("sql"))
        })
    }

    /// The needle table's reader, with its walks watched.
    fn watched_needle_table(dir: &TempDir) -> (SupertableReader, Uuid) {
        let reader = needle_table(dir).reader().expect("reader");
        let superfile = reader.manifest().superfiles[0].superfile_id;
        walk_probe::watch(superfile);
        (reader, superfile)
    }

    #[test]
    fn a_rare_text_match_is_declined_before_the_walk_reads_a_window() {
        // The term index counts `needle` in 9 rows, one in each of 9 files of 500: a page of 10
        // needs 5,000 rows, and those files hold 4,500, both past the walk's limit.
        //  - Both directions decline before any window is read, and answer as the sort.
        //  - So does `needle` and `common`, counted as the rarer of the two in each file.
        // This test pins that a `text_match` is estimated from the term index.
        let dir = TempDir::new().expect("tempdir");
        let (reader, superfile) = watched_needle_table(&dir);
        for filter in [
            "text_match(msg, 'needle')",
            "text_match(msg, 'needle common', 'and')",
        ] {
            for direction in ["ASC", "DESC"] {
                let [walked, sorted] = needle_pages(&reader, filter, direction);
                assert_eq!(walked, sorted, "{filter} {direction}");
                assert_eq!(walked.len(), 9, "{filter} {direction}");
            }
        }
        assert_eq!(
            walk_probe::windows(superfile),
            (0, 0),
            "declined before reading"
        );
        assert_eq!(
            walk_probe::fallbacks(superfile),
            vec![WalkFallback::CannotAnswer; 4]
        );
    }

    #[test]
    fn a_text_match_in_few_files_walks_only_those_files() {
        // `pair` is in 100 rows, all in the last 2 of 20 files, so first by `seq` descending and
        // last ascending.
        //  - Over those 2 files a page of 10 is one row in 10, so one window of 200 rows answers,
        //    in both directions, as the sort does.
        //  - Over every file it would be one row in 100, and ascending the first window would
        //    pass nothing.
        // This test pins that the walk leaves out files the term index proves hold no match.
        let dir = TempDir::new().expect("tempdir");
        let (reader, superfile) = watched_needle_table(&dir);
        for direction in ["ASC", "DESC"] {
            let [walked, sorted] = needle_pages(&reader, "text_match(msg, 'pair')", direction);
            assert_eq!(walked, sorted, "{direction}");
            assert_eq!(walked.len(), 10, "{direction}");
        }
        assert_eq!(walk_probe::windows(superfile), (2, 400));
        assert_eq!(walk_probe::fallbacks(superfile), vec![]);
    }

    #[test]
    fn a_rare_text_match_in_one_small_file_is_walked_under_a_bounded_filter() {
        // `lone` is in 1 row of the last file of 500: a page of 10 at one row in 500 needs 5,000
        // rows, past the walk's limit, but that file holds only 500.
        //  - With no bound on `seq` a walk that reads that file still cannot answer: declined.
        //  - Under `seq >= 0` both directions read that file in one window and answer as the sort.
        // This test pins that only under a bounded filter the decline is capped at the rows held.
        let dir = TempDir::new().expect("tempdir");
        let (reader, superfile) = watched_needle_table(&dir);
        let [walked, sorted] = needle_pages(&reader, "text_match(msg, 'lone')", "ASC");
        assert_eq!(walked, sorted);
        assert_eq!(walk_probe::windows(superfile), (0, 0), "declined unbounded");
        assert_eq!(
            walk_probe::fallbacks(superfile),
            vec![WalkFallback::CannotAnswer]
        );
        for direction in ["ASC", "DESC"] {
            let filter = "seq >= 0 AND text_match(msg, 'lone')";
            let [walked, sorted] = needle_pages(&reader, filter, direction);
            assert_eq!(walked, sorted, "{direction}");
            assert_eq!(walked.len(), 1, "{direction}");
        }
        assert_eq!(walk_probe::windows(superfile), (2, 1000));
        assert_eq!(
            walk_probe::fallbacks(superfile),
            vec![WalkFallback::CannotAnswer]
        );
    }

    #[test]
    fn a_bounded_walk_reads_on_to_the_rows_its_files_hold() {
        // `seq >= 6500` keeps 7 files of 500, and `LIKE` has no estimate: its one match is 3,250
        // rows in, ascending.
        //  - Windows of 20, 80, 320 and 1,280 pass none; the next would pass the limit, but only
        //    1,800 rows are left, so the walk reads them and answers as the sort.
        // This test pins that a bounded walk's give-up counts only the rows its files hold.
        let dir = TempDir::new().expect("tempdir");
        let (reader, superfile) = watched_needle_table(&dir);
        let filter = format!("seq >= {LONE_BOUND} AND msg LIKE '%lone%'");
        let [walked, sorted] = needle_pages(&reader, &filter, "ASC");
        assert_eq!(walked, sorted);
        assert_eq!(walked.len(), 1);
        assert_eq!(walk_probe::windows(superfile), (5, 3500));
        assert_eq!(walk_probe::fallbacks(superfile), vec![]);
    }

    #[test]
    fn a_filter_with_no_estimate_gives_up_once_its_share_puts_the_page_past_the_limit() {
        // `LIKE` and a function have no estimate, and the `needle` rows are late by `seq`.
        //  - Windows of 20, 80, 320 and 1,280 rows pass none, and the next would be 5,120: past the
        //    limit, so the walk stops at 1,700 rows, not at 4,096.
        //  - The page answers as the sort.
        // This test pins the walk's give-up when nothing passes.
        let dir = TempDir::new().expect("tempdir");
        let (reader, superfile) = watched_needle_table(&dir);
        for filter in ["msg LIKE '%needle%'", "strpos(msg, 'needle') > 0"] {
            let (windows, walked) = walk_probe::windows(superfile);
            let [got, sorted] = needle_pages(&reader, filter, "ASC");
            assert_eq!(got, sorted, "{filter}");
            assert_eq!(
                walk_probe::windows(superfile),
                (windows + 4, walked + 1700),
                "{filter}"
            );
        }
        assert_eq!(
            walk_probe::fallbacks(superfile),
            vec![WalkFallback::CannotAnswer; 2]
        );
    }

    #[test]
    fn a_common_text_match_is_still_walked() {
        // `common` is in every second row of every file.
        // This test pins that the walk still answers it, in both directions, as the sort does.
        let dir = TempDir::new().expect("tempdir");
        let (reader, superfile) = watched_needle_table(&dir);
        for direction in ["ASC", "DESC"] {
            let [walked, sorted] = needle_pages(&reader, "text_match(msg, 'common')", direction);
            assert_eq!(walked, sorted, "{direction}");
            assert_eq!(walked.len(), 10, "{direction}");
        }
        assert_eq!(walk_probe::walks(superfile), 2);
        assert_eq!(walk_probe::fallbacks(superfile), vec![]);
    }
}
