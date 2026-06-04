//! `SupertableProvider` — a DataFusion [`TableProvider`] that owns
//! segment selection and hands the rest to DataFusion.
//!
//! ## Two-tier pruning
//!
//! This is the SQL counterpart to the dedicated BM25 / vector
//! entry points: **infino decides which segments are relevant;
//! DataFusion executes over them.** Concretely, [`scan`] performs
//! two tiers of skipping:
//!
//!   1. **Segment skip (infino).** The `WHERE` clause's simple
//!      `column <op> literal` conjuncts are lowered to
//!      [`ScalarPredicate`]s and run through
//!      [`scalar_skip`] against each segment's persisted
//!      `ScalarStatsTable` min/max. Definitely-irrelevant segments
//!      are dropped before any bytes are decoded. This is the same
//!      manifest-level skip philosophy as `fts_bloom_skip` /
//!      `vector_centroid_skip`.
//!   2. **Row-group / page skip (DataFusion).** The surviving
//!      segments' Parquet bytes are exposed to a DataFusion
//!      `ParquetSource` (via an in-memory object store), and the
//!      same predicate is handed to it as a physical expression so
//!      DataFusion's own `PruningPredicate` prunes row groups and
//!      pages, then projects + limits. We deliberately do **not**
//!      reimplement this commodity layer.
//!
//! Correctness is independent of either tier: every pushed filter
//! is reported [`TableProviderFilterPushDown::Inexact`], so
//! DataFusion always re-applies the full predicate in a
//! `FilterExec` above the scan. Both skip tiers are pure
//! *conservative* optimizations — they may keep a non-matching
//! segment/row group, never drop a matching one.
//!
//! ## Why an in-memory object store
//!
//! The reader cache already holds each segment's Parquet bytes
//! (`SuperfileReader::parquet_bytes`, an `Arc`-backed `Bytes` —
//! cloning is a refcount bump, not a copy). Registering those
//! bytes into a [`InMemory`] object store lets us reuse
//! DataFusion's full `ParquetSource` (lazy row-group decode,
//! projection/limit pushdown, row-group pruning) without
//! reimplementing any Parquet machinery. This replaces the v1
//! `MemTable` path, which eagerly decoded every row group of every
//! segment regardless of the query.

use std::any::Any;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::DFSchema;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{FileScanConfigBuilder, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::logical_expr::{Expr, Operator, TableProviderFilterPushDown, TableType};
use datafusion::object_store::memory::InMemory;
use datafusion::object_store::path::Path as ObjPath;
use datafusion::object_store::{ObjectStoreExt, PutPayload};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;

use crate::supertable::SuperfileEntry;
use crate::supertable::manifest::Manifest;
use crate::supertable::query::skip::{ScalarOp, ScalarPredicate, scalar_skip};
use crate::supertable::reader_cache::{DiskCacheStore, SuperfileReaderCache};

/// Logical name the supertable is registered under in the
/// DataFusion `SessionContext`. Callers reference it as
/// `FROM supertable`; we also use it as the schema qualifier when
/// resolving filter columns to a physical pruning predicate.
pub(crate) const TABLE_NAME: &str = "supertable";

/// Object-store URL the surviving segments are registered under
/// for the duration of a scan. The authority is arbitrary — it's
/// only a key into the session's object-store registry.
const MEMORY_STORE_URL: &str = "memory://supertable/";

/// A [`TableProvider`] over a pinned supertable snapshot.
///
/// Cheap to build (just `Arc` clones); all real work happens in
/// [`scan`](TableProvider::scan), which is invoked per physical
/// plan. See the module docs for the two-tier pruning model.
pub(crate) struct SupertableProvider {
    /// User-visible scalar schema (`_id` + scalar + FTS columns).
    /// Matches the Parquet body each segment was written with.
    schema: SchemaRef,
    /// Pinned manifest snapshot for this query.
    manifest: Arc<Manifest>,
    /// In-memory segment-bytes tier.
    store: Arc<dyn SuperfileReaderCache>,
    /// Optional disk cache (storage-backed supertables).
    disk_cache: Option<Arc<DiskCacheStore>>,
}

/// Manual `Debug` (required by `TableProvider`): the cache /
/// disk-cache fields are trait objects without a `Debug` bound, so
/// we print a structural summary instead.
impl std::fmt::Debug for SupertableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SupertableProvider")
            .field("schema", &self.schema)
            .field("n_superfiles", &self.manifest.superfiles.len())
            .field("has_disk_cache", &self.disk_cache.is_some())
            .finish()
    }
}

impl SupertableProvider {
    /// Build a provider over a pinned snapshot. The arguments
    /// mirror what `Supertable::query_sql` already pins.
    pub(crate) fn new(
        schema: SchemaRef,
        manifest: Arc<Manifest>,
        store: Arc<dyn SuperfileReaderCache>,
        disk_cache: Option<Arc<DiskCacheStore>>,
    ) -> Self {
        Self {
            schema,
            manifest,
            store,
            disk_cache,
        }
    }

    /// Flatten the pinned manifest into the visible segment list,
    /// honoring a persisted hierarchical `list` when present (both
    /// eager + lazy modes) and falling back to the flat
    /// `manifest.superfiles` view otherwise.
    ///
    /// Mirrors the v1 `build_mem_table` flattening so SQL sees the
    /// exact same segment set it did under the `MemTable` path.
    async fn flatten_segments(&self) -> DfResult<Vec<Arc<SuperfileEntry>>> {
        match self.manifest.list.as_ref() {
            Some(list) => {
                let kept: Vec<_> = list.parts.iter().map(|p| p.part_id).collect();
                crate::supertable::query::hierarchical_iter::load_and_flatten(
                    self.manifest.as_ref(),
                    &kept,
                )
                .await
                .map_err(|e| DataFusionError::Execution(e.to_string()))
            }
            None => Ok(
                crate::supertable::query::hierarchical_iter::fallback_to_flat_segments(
                    self.manifest.as_ref(),
                ),
            ),
        }
    }
}

#[async_trait]
impl TableProvider for SupertableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// Report every filter as `Inexact`: DataFusion hands us the
    /// predicates (for both pruning tiers) **and** keeps a
    /// `FilterExec` above the scan, so correctness never depends on
    /// our conservative pruning. Returning `Unsupported` (the
    /// default) would withhold the filters from [`scan`] entirely,
    /// disabling segment + row-group skip.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let segments = self.flatten_segments().await?;

        // Tier 1 — segment skip from the persisted scalar min/max.
        let predicates = exprs_to_scalar_predicates(filters, &self.schema);
        let keep = scalar_skip(&segments, &predicates);
        let survivors: Vec<&Arc<SuperfileEntry>> = segments
            .iter()
            .zip(keep)
            .filter_map(|(entry, keep)| keep.then_some(entry))
            .collect();

        // Nothing survived (empty table, or every segment pruned):
        // a schema-correct empty scan. EmptyExec yields one
        // partition / zero rows, so `COUNT(*)` is 0 and `SELECT *`
        // returns the right empty shape. The projection must be
        // honored here too — `COUNT(*)` projects zero columns, and
        // DataFusion checks the physical schema against the logical
        // one.
        if survivors.is_empty() {
            let projected = match projection {
                Some(indices) => Arc::new(self.schema.project(indices)?),
                None => Arc::clone(&self.schema),
            };
            return Ok(Arc::new(EmptyExec::new(projected)));
        }

        // Expose the surviving segments' bytes to DataFusion via an
        // in-memory object store. `parquet_bytes()` is Arc-backed,
        // so `clone()` / `PutPayload::from` are refcount bumps.
        let object_store = Arc::new(InMemory::new());
        let mut files: Vec<PartitionedFile> = Vec::with_capacity(survivors.len());
        for entry in &survivors {
            let reader = crate::supertable::query::superfile_reader::superfile_reader(
                &self.store,
                self.disk_cache.as_ref(),
                self.manifest.options.storage.as_ref(),
                &entry.uri,
                entry.subsection_offsets.as_ref(),
            )
            .await
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            let bytes = reader
                .parquet_bytes()
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "SQL scan requires eager-opened superfile bytes; reader for {:?} \
                         was opened via the lazy path which does not materialize the \
                         full segment",
                        entry.uri
                    ))
                })?
                .clone();
            let path = entry.uri.storage_path();
            let size = bytes.len() as u64;
            object_store
                .put(&ObjPath::from(path.clone()), PutPayload::from(bytes))
                .await
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            files.push(PartitionedFile::new(path, size));
        }

        let url = ObjectStoreUrl::parse(MEMORY_STORE_URL)?;
        state
            .runtime_env()
            .register_object_store(url.as_ref(), object_store);

        // Tier 2 — DataFusion-owned row-group / page pruning. Hand
        // the same predicate to ParquetSource as a physical expr;
        // on any lowering failure we simply skip row-group pruning
        // (FilterExec above still guarantees correctness).
        let mut source = ParquetSource::new(Arc::clone(&self.schema));
        if let Some(predicate) = row_group_predicate(state, filters, &self.schema) {
            source = source.with_predicate(predicate);
        }

        // Only push the LIMIT into the scan when there are no
        // filters: with an `Inexact` filter re-applied above, a
        // scan-level limit could stop before enough matching rows
        // are produced. With no filters, DataFusion's own limit and
        // a scan-level limit agree.
        let effective_limit = if filters.is_empty() { limit } else { None };

        let mut builder = FileScanConfigBuilder::new(url, Arc::new(source));
        for file in files {
            builder = builder.with_file(file);
        }
        let config = builder
            .with_projection_indices(projection.cloned())?
            .with_limit(effective_limit)
            .build();

        let plan: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(config);
        Ok(plan)
    }
}

/// Lower a conjunction of DataFusion filter `Expr`s into infino's
/// [`ScalarPredicate`]s for segment skip.
///
/// Each top-level filter is treated as a conjunct; nested `AND`s
/// are flattened. Only `column <op> literal` (and the mirrored
/// `literal <op> column`) shapes over a column present in `schema`
/// are recognized — everything else is silently dropped (it just
/// doesn't contribute pruning; `FilterExec` still applies it).
fn exprs_to_scalar_predicates(filters: &[Expr], schema: &SchemaRef) -> Vec<ScalarPredicate> {
    let mut out = Vec::new();
    for filter in filters {
        collect_conjuncts(filter, schema, &mut out);
    }
    out
}

/// Recurse through `AND` nodes, pushing any recognized
/// `column <op> literal` leaf into `out`.
fn collect_conjuncts(expr: &Expr, schema: &SchemaRef, out: &mut Vec<ScalarPredicate>) {
    if let Expr::BinaryExpr(be) = expr {
        if be.op == Operator::And {
            collect_conjuncts(&be.left, schema, out);
            collect_conjuncts(&be.right, schema, out);
        } else if let Some(p) = leaf_to_predicate(&be.left, be.op, &be.right, schema) {
            out.push(p);
        }
    }
}

/// Convert a single `left <op> right` comparison into a
/// [`ScalarPredicate`] when it's `column <op> literal` or
/// `literal <op> column` over a known column; else `None`.
fn leaf_to_predicate(
    left: &Expr,
    op: Operator,
    right: &Expr,
    schema: &SchemaRef,
) -> Option<ScalarPredicate> {
    let (column, value, scalar_op) = match (left, right) {
        (Expr::Column(c), Expr::Literal(v, _)) => (&c.name, v, map_op(op)?),
        (Expr::Literal(v, _), Expr::Column(c)) => (&c.name, v, flip_op(map_op(op)?)),
        _ => return None,
    };
    // Guard against columns not in the scalar schema (e.g. a typo
    // would already fail planning, but be defensive).
    schema.field_with_name(column).ok()?;
    Some(ScalarPredicate {
        column: column.clone(),
        op: scalar_op,
        value: value.clone(),
    })
}

/// Map a DataFusion comparison [`Operator`] to a [`ScalarOp`].
/// Non-comparison operators return `None` (no pruning).
fn map_op(op: Operator) -> Option<ScalarOp> {
    match op {
        Operator::Eq => Some(ScalarOp::Eq),
        Operator::NotEq => Some(ScalarOp::NotEq),
        Operator::Lt => Some(ScalarOp::Lt),
        Operator::LtEq => Some(ScalarOp::LtEq),
        Operator::Gt => Some(ScalarOp::Gt),
        Operator::GtEq => Some(ScalarOp::GtEq),
        _ => None,
    }
}

/// Flip a comparison so `literal <op> column` becomes the
/// equivalent `column <flipped> literal` (e.g. `5 < x` ⟺ `x > 5`).
fn flip_op(op: ScalarOp) -> ScalarOp {
    match op {
        ScalarOp::Eq => ScalarOp::Eq,
        ScalarOp::NotEq => ScalarOp::NotEq,
        ScalarOp::Lt => ScalarOp::Gt,
        ScalarOp::LtEq => ScalarOp::GtEq,
        ScalarOp::Gt => ScalarOp::Lt,
        ScalarOp::GtEq => ScalarOp::LtEq,
    }
}

/// Lower the conjunction of `filters` into a single physical
/// predicate for DataFusion's row-group pruning, or `None` if the
/// filters are empty or can't be lowered (column-resolution /
/// planning failure → skip pruning, never incorrect).
fn row_group_predicate(
    state: &dyn Session,
    filters: &[Expr],
    schema: &SchemaRef,
) -> Option<Arc<dyn PhysicalExpr>> {
    let combined = filters.iter().cloned().reduce(|a, b| a.and(b))?;
    // Filter columns may arrive qualified (`supertable.col`) or
    // bare depending on the plan; try the qualified schema first,
    // then the unqualified one.
    let df_schema = DFSchema::try_from_qualified_schema(TABLE_NAME, schema.as_ref())
        .or_else(|_| DFSchema::try_from(schema.as_ref().clone()))
        .ok()?;
    state.create_physical_expr(combined, &df_schema).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{col, lit};
    use datafusion::scalar::ScalarValue;

    fn schema_xy() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, true),
            Field::new("y", DataType::Int64, true),
        ]))
    }

    #[test]
    fn col_op_lit_maps_directly() {
        let s = schema_xy();
        let preds = exprs_to_scalar_predicates(&[col("x").gt(lit(5_i64))], &s);
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].column, "x");
        assert_eq!(preds[0].op, ScalarOp::Gt);
        assert_eq!(preds[0].value, ScalarValue::Int64(Some(5)));
    }

    #[test]
    fn lit_op_col_flips_operator() {
        // `5 < x`  ⟺  `x > 5`
        let s = schema_xy();
        let preds = exprs_to_scalar_predicates(&[lit(5_i64).lt(col("x"))], &s);
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].column, "x");
        assert_eq!(preds[0].op, ScalarOp::Gt);
        assert_eq!(preds[0].value, ScalarValue::Int64(Some(5)));
    }

    #[test]
    fn and_is_flattened_into_two_predicates() {
        let s = schema_xy();
        let expr = col("x").gt_eq(lit(5_i64)).and(col("x").lt_eq(lit(8_i64)));
        let preds = exprs_to_scalar_predicates(&[expr], &s);
        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].op, ScalarOp::GtEq);
        assert_eq!(preds[1].op, ScalarOp::LtEq);
    }

    #[test]
    fn multiple_top_level_filters_each_contribute() {
        let s = schema_xy();
        let preds =
            exprs_to_scalar_predicates(&[col("x").gt(lit(1_i64)), col("y").lt(lit(9_i64))], &s);
        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].column, "x");
        assert_eq!(preds[1].column, "y");
    }

    #[test]
    fn col_op_col_is_ignored() {
        let s = schema_xy();
        let preds = exprs_to_scalar_predicates(&[col("x").gt(col("y"))], &s);
        assert!(preds.is_empty());
    }

    #[test]
    fn unknown_column_is_ignored() {
        let s = schema_xy();
        let preds = exprs_to_scalar_predicates(&[col("z").gt(lit(1_i64))], &s);
        assert!(preds.is_empty());
    }

    #[test]
    fn non_comparison_operator_is_ignored() {
        let s = schema_xy();
        // x + 1 (arithmetic) — not a comparison, no predicate.
        let preds = exprs_to_scalar_predicates(&[col("x") + lit(1_i64)], &s);
        assert!(preds.is_empty());
    }
}
