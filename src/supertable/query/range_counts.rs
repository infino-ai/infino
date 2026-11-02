// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! One row of range counts: the buckets of a bucketed range aggregation
//! (`COUNT(*) FILTER (WHERE range)` outputs), each counted from the range
//! index, in one scan.
//!
//! The covered-aggregate rewrite plans the aggregation as a projection over
//! this table, so the plan stays two nodes whatever the number of buckets,
//! where a scan per bucket joined into one row costs the planner a branch
//! per bucket.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    catalog::{Session, TableProvider},
    datasource::memory::MemorySourceConfig,
    error::{DataFusionError, Result as DfResult},
    logical_expr::{Expr, TableType},
    physical_plan::ExecutionPlan,
};
use futures::future::try_join_all;

use crate::supertable::query::provider::{CountedRange, SupertableProvider};

/// One bucket: its output column, the `WHERE` and bucket filter together as
/// conjuncts, and those conjuncts as one interval on a range-indexed column;
/// with no conjuncts and no interval, every live row of the table.
#[derive(Debug)]
pub(crate) struct RangeBucket {
    pub(crate) name: String,
    pub(crate) range: Option<CountedRange>,
    pub(crate) filters: Vec<Expr>,
}

/// A table of one row: each bucket's `COUNT(*)`, under the bucket's name.
#[derive(Debug)]
pub(crate) struct RangeCounts {
    table: SupertableProvider,
    buckets: Vec<RangeBucket>,
    schema: SchemaRef,
}

impl RangeCounts {
    /// The counts of `buckets` over `table`'s snapshot.
    pub(crate) fn new(table: SupertableProvider, buckets: Vec<RangeBucket>) -> Self {
        let fields: Vec<Field> = buckets
            .iter()
            .map(|bucket| Field::new(&bucket.name, DataType::Int64, false))
            .collect();
        Self {
            table,
            buckets,
            schema: Arc::new(Schema::new(fields)),
        }
    }
}

#[async_trait]
impl TableProvider for RangeCounts {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let counts = try_join_all(self.buckets.iter().map(|bucket| {
            self.table
                .counted_rows(state, bucket.range.as_ref(), &bucket.filters)
        }))
        .await?;
        let columns = counts
            .into_iter()
            .map(|count| {
                let count =
                    i64::try_from(count).map_err(|e| DataFusionError::External(e.into()))?;
                Ok(Arc::new(Int64Array::from(vec![count])) as ArrayRef)
            })
            .collect::<DfResult<Vec<_>>>()?;
        let row = RecordBatch::try_new(Arc::clone(&self.schema), columns)?;
        let exec: Arc<dyn ExecutionPlan> = MemorySourceConfig::try_new_exec(
            &[vec![row]],
            Arc::clone(&self.schema),
            projection.cloned(),
        )?;
        Ok(exec)
    }
}
