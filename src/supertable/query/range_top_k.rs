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
//! needs the superfiles. It fires only when every filter under the sort is
//! a range on the sort column: a filter on another column could reject the
//! rows the index puts first, and the cutoff would then drop rows that
//! belong in the answer. A sort with nulls first and no filter is left
//! alone too, as nulls are in no range index and would come first. The
//! sort may order by more keys after the column (`ORDER BY price, id`, the
//! tie-break a paged search adds): those order only the rows tied on the
//! column, so the scan then reads every row tied with the cutoff too. The
//! one exception is a sort whose only later key is the table's `_id`,
//! ascending: a range index that keeps the rows of a tied key in `_id`
//! order makes a superfile's first rows by the column its first by the
//! column and `_id`, and the scan reads those alone. The sort and the
//! filters stay where they are, so the answer is the one they give over
//! the rows the scan returns.

use std::sync::Arc;

use datafusion::{
    catalog::TableProvider,
    common::tree_node::Transformed,
    datasource::{DefaultTableSource, provider_as_source},
    error::Result as DfResult,
    logical_expr::{
        Expr, Filter, LogicalPlan, Sort, TableScan, TableSource, utils::split_conjunction,
    },
    optimizer::{OptimizerConfig, OptimizerRule, optimizer::ApplyOrder},
};

use crate::supertable::query::{
    candidate::single_range,
    provider::{SupertableProvider, TopKHint},
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
    let LogicalPlan::Sort(Sort {
        expr,
        input,
        fetch: Some(k),
    }) = plan
    else {
        return Ok(None);
    };
    // Later sort keys (an id that breaks ties, say) order only rows the
    // first key ties; the hint then keeps every row tied with its cutoff.
    let Some((sort_expr, later)) = expr.split_first() else {
        return Ok(None);
    };
    let Expr::Column(column) = &sort_expr.expr else {
        return Ok(None);
    };
    // Walk to the scan through filters and plain column projections that
    // keep the sort column, collecting the filters' predicates.
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
            LogicalPlan::TableScan(scan) => break scan,
            _ => return Ok(None),
        }
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
    let ties_by_id = matches!(
        later,
        [only] if only.asc && matches!(&only.expr, Expr::Column(c) if c.name == id_column)
    );
    if single_range(&predicates, &column.name, &range_cols).is_none() {
        return Ok(None);
    }
    if predicates.is_empty() && sort_expr.nulls_first {
        return Ok(None);
    }
    let hinted = Arc::new(provider.with_top_k(TopKHint {
        column: column.name.clone(),
        descending: !sort_expr.asc,
        k: *k,
        ties: !later.is_empty(),
        ties_by_id,
    }));
    let source = provider_as_source(hinted as Arc<dyn TableProvider>);
    Ok(Some(with_scan_source(plan, &source)?))
}

/// Our provider behind `scan`, when it is ours.
fn provider_of(scan: &TableScan) -> Option<&SupertableProvider> {
    let source = scan.source.downcast_ref::<DefaultTableSource>()?;
    source.table_provider.downcast_ref::<SupertableProvider>()
}

/// `plan` with the scan at the bottom of its single-input chain reading
/// from `source`.
fn with_scan_source(plan: &LogicalPlan, source: &Arc<dyn TableSource>) -> DfResult<LogicalPlan> {
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
    use std::sync::Arc;

    use arrow_array::{Array, Int64Array, LargeStringArray, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{col, lit};
    use roaring::RoaringBitmap;
    use tempfile::TempDir;
    use tokio::runtime::Builder;

    use crate::{
        config::OptimizeOptions,
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{
            Supertable, SupertableOptions, SupertableReader, query::scalar_cache::ByteLru,
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
}
