// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`EmptySortKeys`]: a top-k's leading sort keys on columns no row holds a
//! value in are dropped.
//!
//! By every superfile's statistics such a column is null in every row, so it
//! orders nothing: `ORDER BY empty_column, _id LIMIT k` keeps the same rows
//! in the same order as `ORDER BY _id LIMIT k`. Sorted on its remaining key,
//! the top-k's running bound is a bound on a column with statistics, which a
//! scan of kept decoded columns skips row groups by (see `CachedScanExec`).

use std::sync::Arc;

use datafusion::{
    common::tree_node::Transformed,
    error::Result as DfResult,
    logical_expr::{Expr, LogicalPlan, Sort},
    optimizer::{OptimizerConfig, OptimizerRule, optimizer::ApplyOrder},
};

use crate::supertable::query::{provider::SupertableProvider, range_top_k::provider_of};

/// The rule. Registered on the `query_sql` session after DataFusion's
/// built-in rules, which push the `LIMIT` into the sort as its fetch.
#[derive(Debug, Default)]
pub(crate) struct EmptySortKeys;

impl OptimizerRule for EmptySortKeys {
    fn name(&self) -> &str {
        "empty_sort_keys"
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
        match try_rewrite(&plan) {
            Some(rewritten) => Ok(Transformed::yes(rewritten)),
            None => Ok(Transformed::no(plan)),
        }
    }
}

/// The sort without its leading keys on columns no row holds a value in;
/// `None` when it has none, keeps no key without them, or is not a sort
/// with a fetch over plain column projections, aliases and filters down to
/// a scan of the table.
fn try_rewrite(plan: &LogicalPlan) -> Option<LogicalPlan> {
    let LogicalPlan::Sort(Sort {
        expr,
        input,
        fetch: Some(k),
    }) = plan
    else {
        return None;
    };
    let mut node = input.as_ref();
    let scan = loop {
        match node {
            LogicalPlan::Projection(projection) if projection.expr.iter().all(keeps_its_name) => {
                node = projection.input.as_ref();
            }
            LogicalPlan::SubqueryAlias(alias) => node = alias.input.as_ref(),
            LogicalPlan::Filter(filter) => node = filter.input.as_ref(),
            LogicalPlan::TableScan(scan) => break scan,
            _ => return None,
        }
    };
    let provider = provider_of(scan)?;
    let empty = expr
        .iter()
        .take_while(|key| matches!(&key.expr, Expr::Column(column) if holds_no_value(provider, &column.name)))
        .count();
    if empty == 0 || empty == expr.len() {
        return None;
    }
    Some(LogicalPlan::Sort(Sort {
        expr: expr[empty..].to_vec(),
        input: Arc::clone(input),
        fetch: Some(*k),
    }))
}

/// Whether a projection's expression is a column under its own name: bare,
/// or aliased to the name it has (`"_id" AS "_id"`), so a sort key above it
/// names the same column below it.
fn keeps_its_name(expr: &Expr) -> bool {
    match expr {
        Expr::Column(_) => true,
        Expr::Alias(alias) => {
            matches!(alias.expr.as_ref(), Expr::Column(column) if column.name == alias.name)
        }
        _ => false,
    }
}

/// Whether every superfile's statistics say `column` holds no value in any
/// of its rows. A superfile without statistics for the column says nothing.
fn holds_no_value(provider: &SupertableProvider, column: &str) -> bool {
    let manifest = provider.manifest();
    let Some(field) = manifest.table_schema().id_of(column) else {
        return false;
    };
    !manifest.superfiles.is_empty()
        && manifest.superfiles.iter().all(|entry| {
            entry
                .scalar_stats
                .column(field)
                .is_some_and(|stats| stats.all_null())
        })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Decimal128Array, Int64Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{col, lit};

    use crate::{IndexSpec, connect};

    /// Rows in each of the table's three commits.
    const ROWS_PER_COMMIT: i64 = 400;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("tag", DataType::LargeUtf8, false),
            Field::new("note", DataType::LargeUtf8, true),
        ]))
    }

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

    /// A key on a column no row holds a value in, before `_id`, answers as
    /// `_id` alone: ascending and descending, through same-name aliases,
    /// under a filter the index answers and one it does not, across three
    /// commits and deleted rows.
    #[test]
    fn an_empty_leading_key_orders_nothing() {
        // Deletes need durable storage.
        let root = tempfile::TempDir::new().expect("tempdir");
        let conn = connect(format!("file://{}", root.path().display())).expect("connect");
        let table = conn
            .create_table("logs", schema(), IndexSpec::new().fts("tag"))
            .expect("create");
        for commit in 0..3_i64 {
            let seqs: Vec<i64> =
                (commit * ROWS_PER_COMMIT..(commit + 1) * ROWS_PER_COMMIT).collect();
            let tags: Vec<&str> = seqs
                .iter()
                .map(|s| if s % 3 == 0 { "kernel" } else { "cron" })
                .collect();
            let notes: Vec<Option<&str>> = seqs.iter().map(|_| None).collect();
            table
                .append(
                    &RecordBatch::try_new(
                        schema(),
                        vec![
                            Arc::new(Int64Array::from(seqs)),
                            Arc::new(LargeStringArray::from(tags)),
                            Arc::new(LargeStringArray::from(notes)),
                        ],
                    )
                    .expect("batch"),
                )
                .expect("append");
        }
        table.delete(col("seq").lt(lit(30_i64))).expect("delete");
        let kernel = {
            let mut all = ids(&conn
                .query_sql("SELECT _id FROM logs WHERE tag = 'kernel'")
                .expect("sql"));
            all.sort_unstable();
            all
        };
        assert!(kernel.len() > 300, "the fixture keeps a third of the rows");
        for k in [1_usize, 10, 133, 2000] {
            for (order, descending) in [("ASC", false), ("DESC", true)] {
                let mut want = kernel.clone();
                if descending {
                    want.reverse();
                }
                want.truncate(k);
                for filter in ["tag = 'kernel'", "seq % 3 = 0"] {
                    for select in ["_id", "seq, \"note\" AS \"note\", \"_id\" AS \"_id\""] {
                        let sql = format!(
                            "SELECT {select} FROM logs WHERE {filter} \
                             ORDER BY note ASC NULLS LAST, _id {order} LIMIT {k}"
                        );
                        assert_eq!(ids(&conn.query_sql(&sql).expect("sql")), want, "{sql}");
                    }
                }
            }
        }
    }

    /// A key on a column some row holds a value in stays, and orders.
    #[test]
    fn a_key_on_a_column_with_values_keeps_the_sort() {
        let conn = connect("memory://").expect("connect");
        let table = conn
            .create_table("logs", schema(), IndexSpec::new().fts("tag"))
            .expect("create");
        table
            .append(
                &RecordBatch::try_new(
                    schema(),
                    vec![
                        Arc::new(Int64Array::from(vec![3_i64, 1, 2])),
                        Arc::new(LargeStringArray::from(vec!["kernel", "kernel", "kernel"])),
                        Arc::new(LargeStringArray::from(vec![Some("b"), None, Some("a")])),
                    ],
                )
                .expect("batch"),
            )
            .expect("append");
        let sql = "SELECT seq FROM logs WHERE tag = 'kernel' ORDER BY note ASC NULLS LAST, _id ASC LIMIT 2";
        let batches = conn.query_sql(sql).expect("sql");
        let seqs: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                let c = b
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("seq")
                    .clone();
                (0..c.len()).map(move |i| c.value(i))
            })
            .collect();
        assert_eq!(seqs, [2, 3], "{sql}");
    }
}
