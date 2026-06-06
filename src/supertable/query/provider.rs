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
use datafusion::datasource::physical_plan::parquet::ParquetAccessPlan;
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

use bytes::Bytes;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector};
use roaring::RoaringBitmap;

use crate::superfile::fts::reader::BoolMode;
use crate::supertable::SuperfileEntry;
use crate::supertable::manifest::Manifest;
use crate::supertable::query::skip::{ScalarOp, ScalarPredicate};
use crate::supertable::reader_cache::{DiskCacheStore, SuperfileReaderCache};
use crate::supertable::tombstones::SidecarCache;
use datafusion::scalar::ScalarValue;

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
    /// Per-superfile soft-delete (tombstone) overlay. `None` for
    /// in-memory tables with no WAL/mutation surface. When present,
    /// [`scan`](TableProvider::scan) pushes the tombstoned rows into
    /// each segment's Parquet read as a [`ParquetAccessPlan`] row
    /// selection — the *lazy* delete path: deleted rows are skipped
    /// during decode rather than materialized then dropped. This
    /// keeps the analytical SELECT path's projection/limit/row-group
    /// pushdown intact while still honoring deletes.
    tombstone_cache: Option<Arc<SidecarCache>>,
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
            .field("has_tombstone_cache", &self.tombstone_cache.is_some())
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
        tombstone_cache: Option<Arc<SidecarCache>>,
    ) -> Self {
        Self {
            schema,
            manifest,
            store,
            disk_cache,
            tombstone_cache,
        }
    }

    /// Lower scalar predicates to prune leaves. Each predicate yields a
    /// `Scalar` leaf; additionally, an equality on an FTS-indexed text
    /// column also yields a `TermPresence` leaf so the segment's term
    /// bloom prunes it. Sound: a row matching `col = 'a b'` has a value
    /// whose tokens include every token of the literal, so requiring all
    /// of them possibly-present (`BoolMode::And`) never drops a match —
    /// bloom false positives can only keep a segment, never drop one.
    fn predicates_to_prune_leaves(
        &self,
        predicates: Vec<ScalarPredicate>,
    ) -> Vec<crate::supertable::query::prune::PruneLeaf> {
        use crate::supertable::query::prune::PruneLeaf;
        let opts = &self.manifest.options;
        let mut leaves = Vec::with_capacity(predicates.len());
        for pred in predicates {
            if pred.op == ScalarOp::Eq
                && opts.fts_columns.iter().any(|c| c.column == pred.column)
                && let Some(tok) = opts.tokenizer.as_ref()
                && let Some(literal) = scalar_as_str(&pred.value)
            {
                let terms: Vec<String> = tok.tokenize(literal).collect();
                if !terms.is_empty() {
                    leaves.push(PruneLeaf::TermPresence {
                        column: pred.column.clone(),
                        terms,
                        mode: BoolMode::And,
                    });
                }
            }
            leaves.push(PruneLeaf::Scalar(pred));
        }
        leaves
    }

    /// Lower `filters` to prune leaves and return the segments that
    /// survive the two-tier prune — exactly the inputs the scan hands to
    /// DataFusion.
    async fn select_survivors(&self, filters: &[Expr]) -> DfResult<Vec<Arc<SuperfileEntry>>> {
        let predicates = exprs_to_scalar_predicates(filters, &self.schema);
        let leaves = self.predicates_to_prune_leaves(predicates);
        crate::supertable::query::prune::select_segments(self.manifest.as_ref(), &leaves)
            .await
            .map_err(|e| DataFusionError::Execution(e.to_string()))
    }

    /// Test hook: how many segments survive pruning for `filters` — the
    /// observable behind "did the index prune more than min/max?".
    #[cfg(test)]
    pub(crate) async fn surviving_segment_count(&self, filters: &[Expr]) -> usize {
        self.select_survivors(filters)
            .await
            .expect("select survivors")
            .len()
    }
}

/// Extract a UTF-8 string literal from a scalar value, if it is one.
/// Used to tokenize an equality literal for FTS-bloom pruning.
fn scalar_as_str(v: &ScalarValue) -> Option<&str> {
    match v {
        ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => Some(s.as_str()),
        _ => None,
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
        // Segment selection via the shared two-tier prune (the same
        // path FTS search uses); see `select_survivors`. Survivors go to
        // DataFusion.
        let survivor_entries = self.select_survivors(filters).await?;
        let survivors: Vec<&Arc<SuperfileEntry>> = survivor_entries.iter().collect();

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
        //
        // One `Instant::now()` for the whole scan so every per-segment
        // tombstone lookup shares the same `SidecarCache` TTL
        // reference.
        let now = std::time::Instant::now();

        // Warm every surviving segment's tombstone bitmap in one
        // batched fetch before the per-segment sweep below, mirroring
        // the bm25 / vector fan-out (see `SidecarCache::prefetch`);
        // each `bitmap_for` in the loop then resolves from cache.
        if let Some(cache) = self.tombstone_cache.as_ref() {
            let ids: Vec<_> = survivors.iter().map(|e| e.superfile_id).collect();
            cache.prefetch(&ids, now).await;
        }

        // In-memory object store exposing the surviving segment bytes
        // to DataFusion's Parquet source for the duration of this scan.
        let mem_store = Arc::new(InMemory::new());
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

            // Lazy delete path: translate this segment's tombstone
            // bitmap into a Parquet row selection so deleted rows are
            // never decoded. Absent/empty overlay → full scan, zero
            // overhead. The `local_doc_id` in the bitmap is the row's
            // global position within the segment's Parquet body, which
            // is exactly the coordinate `ParquetAccessPlan` selects on.
            let access_plan = match self.tombstone_cache.as_ref() {
                Some(cache) => {
                    let bitmap = cache
                        .bitmap_for(entry.superfile_id, now)
                        .map_err(|e| DataFusionError::Execution(format!("tombstone cache: {e}")))?;
                    if bitmap.is_empty() {
                        None
                    } else {
                        tombstone_access_plan(&bytes, &bitmap)?
                    }
                }
                None => None,
            };

            mem_store
                .put(&ObjPath::from(path.clone()), PutPayload::from(bytes))
                .await
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            let mut file = PartitionedFile::new(path, size);
            if let Some(plan) = access_plan {
                file = file.with_extensions(Arc::new(plan));
            }
            files.push(file);
        }

        let url = ObjectStoreUrl::parse(MEMORY_STORE_URL)?;
        state
            .runtime_env()
            .register_object_store(url.as_ref(), mem_store);

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

/// Build a [`ParquetAccessPlan`] that skips this segment's
/// tombstoned rows during decode, or `None` if none of the deleted
/// `local_doc_id`s fall inside the file (so a plain full scan is
/// correct and cheaper than attaching an all-`Scan` plan).
///
/// `bitmap` holds the tombstoned `local_doc_id`s, where a row's
/// `local_doc_id` is its 0-based global position within the segment's
/// Parquet body (row groups are laid out in append order, so global
/// position partitions contiguously across them). For each row group
/// we translate the deleted positions into a [`RowSelection`] of
/// alternating select/skip runs; fully-deleted row groups are skipped
/// outright and clean ones are left as `Scan`.
///
/// Parsing the footer via [`ParquetRecordBatchReaderBuilder`] only
/// touches metadata, not column data, and only happens when the
/// segment actually has tombstones — clean tables pay nothing.
fn tombstone_access_plan(
    parquet_bytes: &Bytes,
    bitmap: &RoaringBitmap,
) -> DfResult<Option<ParquetAccessPlan>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_bytes.clone())
        .map_err(|e| DataFusionError::Execution(format!("parquet metadata: {e}")))?;
    let row_groups = builder.metadata().row_groups();
    // Sorted ascending — `RoaringBitmap::iter` yields in order, which
    // lets each row group binary-search its slice of deleted ids.
    let deleted: Vec<u32> = bitmap.iter().collect();

    let mut plan = ParquetAccessPlan::new_all(row_groups.len());
    let mut base: u32 = 0;
    let mut any = false;
    for (idx, rg) in row_groups.iter().enumerate() {
        let n = rg.num_rows() as u32;
        if n == 0 {
            continue;
        }
        let lo = deleted.partition_point(|&x| x < base);
        let hi = deleted.partition_point(|&x| x < base + n);
        let rg_deleted = &deleted[lo..hi];
        if rg_deleted.is_empty() {
            base += n;
            continue;
        }
        any = true;
        if rg_deleted.len() as u32 == n {
            plan.skip(idx);
            base += n;
            continue;
        }
        // Coalesce consecutive deleted positions into single skip runs,
        // emitting the live gaps between them as select runs.
        let mut selectors: Vec<RowSelector> = Vec::new();
        let mut cursor: u32 = 0; // next un-emitted position, relative to row group
        let mut i = 0usize;
        while i < rg_deleted.len() {
            let start_rel = rg_deleted[i] - base;
            if start_rel > cursor {
                selectors.push(RowSelector::select((start_rel - cursor) as usize));
            }
            let mut j = i;
            while j + 1 < rg_deleted.len() && rg_deleted[j + 1] == rg_deleted[j] + 1 {
                j += 1;
            }
            let run = (rg_deleted[j] - rg_deleted[i] + 1) as usize;
            selectors.push(RowSelector::skip(run));
            cursor = (rg_deleted[j] - base) + 1;
            i = j + 1;
        }
        if cursor < n {
            selectors.push(RowSelector::select((n - cursor) as usize));
        }
        plan.scan_selection(idx, RowSelection::from(selectors));
        base += n;
    }

    Ok(any.then_some(plan))
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

    use arrow_array::{Int64Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{col, lit};
    use datafusion::scalar::ScalarValue;

    use crate::superfile::builder::FtsConfig;
    use crate::supertable::{Supertable, SupertableOptions};
    use crate::test_helpers::default_tokenizer;

    /// Build an in-memory Parquet file of `Int64` values `0..total`
    /// split into row groups of `rg_size` rows each.
    fn parquet_with_row_groups(total: i64, rg_size: usize) -> Bytes {
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;

        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let arr = Int64Array::from((0..total).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(arr)]).expect("batch");
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rg_size))
            .build();
        let mut buf = Vec::new();
        {
            let mut w =
                ArrowWriter::try_new(&mut buf, Arc::clone(&schema), Some(props)).expect("writer");
            w.write(&batch).expect("write");
            w.close().expect("close");
        }
        Bytes::from(buf)
    }

    /// Decode `bytes` honoring `plan`'s row-group + row selection and
    /// return the surviving `v` values in order.
    fn read_with_plan(bytes: &Bytes, plan: ParquetAccessPlan) -> Vec<i64> {
        let meta = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .expect("meta")
            .metadata()
            .clone();
        let row_groups = plan.row_group_indexes();
        let selection = plan
            .into_overall_row_selection(meta.row_groups())
            .expect("overall selection");
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .expect("builder")
            .with_row_groups(row_groups);
        if let Some(sel) = selection {
            builder = builder.with_row_selection(sel);
        }
        let reader = builder.build().expect("reader");
        let mut got = Vec::new();
        for b in reader {
            let b = b.expect("batch");
            let c = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("int64 col");
            for i in 0..c.len() {
                got.push(c.value(i));
            }
        }
        got
    }

    #[test]
    fn tombstone_access_plan_none_when_no_deletes_in_file() {
        let bytes = parquet_with_row_groups(12, 4);
        // Tombstone an id past the end of the file → nothing selected.
        let mut bm = RoaringBitmap::new();
        bm.insert(99);
        assert!(
            tombstone_access_plan(&bytes, &bm).expect("plan").is_none(),
            "no deleted id falls inside the file → full scan (None)"
        );
    }

    #[test]
    fn tombstone_access_plan_skips_deleted_across_row_groups() {
        // 3 row groups of 4 rows: rg0=0..4, rg1=4..8, rg2=8..12.
        let bytes = parquet_with_row_groups(12, 4);

        // rg0: delete 0,1 (consecutive run at the start)
        // rg1: delete 4,5,6,7 (whole row group → Skip)
        // rg2: delete 10 (single row mid-group)
        let mut bm = RoaringBitmap::new();
        for id in [0u32, 1, 4, 5, 6, 7, 10] {
            bm.insert(id);
        }

        let plan = tombstone_access_plan(&bytes, &bm)
            .expect("plan")
            .expect("some deletes");

        // Whole-deleted row group is skipped entirely.
        assert!(!plan.should_scan(1), "fully-tombstoned row group 1 skipped");
        assert!(plan.should_scan(0));
        assert!(plan.should_scan(2));

        let survivors = read_with_plan(&bytes, plan);
        assert_eq!(survivors, vec![2, 3, 8, 9, 11]);
    }

    #[test]
    fn tombstone_access_plan_handles_alternating_and_boundary_deletes() {
        // Single row group of 8 rows with an alternating pattern plus
        // the last row deleted (exercises the trailing-select branch).
        let bytes = parquet_with_row_groups(8, 8);
        let mut bm = RoaringBitmap::new();
        for id in [0u32, 2, 4, 7] {
            bm.insert(id);
        }
        let plan = tombstone_access_plan(&bytes, &bm)
            .expect("plan")
            .expect("some deletes");
        let survivors = read_with_plan(&bytes, plan);
        assert_eq!(survivors, vec![1, 3, 5, 6]);
    }

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

    // ---- Segment-prune contrast: index helps vs. doesn't ----------
    //
    // End-to-end through a real multi-segment supertable: count how many
    // segments survive the scan's prune for different predicates. This
    // is the observable proof that the embedded FTS index prunes more
    // than the scalar min/max a plain Parquet scan relies on — and,
    // honestly, where it doesn't (full scans, non-FTS predicates).

    fn cat_title_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("category", DataType::LargeUtf8, false),
            Field::new("title", DataType::LargeUtf8, false),
        ]))
    }

    fn cat_title_opts() -> SupertableOptions {
        // One writer thread → one segment per commit (deterministic
        // segment counts).
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        SupertableOptions::new(
            cat_title_schema(),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        )
        .expect("opts")
        .with_writer_pool(pool)
    }

    fn cat_title_batch(cats: &[&str], titles: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            cat_title_schema(),
            vec![
                Arc::new(LargeStringArray::from(cats.to_vec())),
                Arc::new(LargeStringArray::from(titles.to_vec())),
            ],
        )
        .expect("batch")
    }

    #[test]
    fn segment_prune_index_helps_vs_does_not() {
        let st = Supertable::create(cat_title_opts()).expect("create");
        let mut w = st.writer().expect("writer");
        // Three segments. Every segment's `title` lexicographic range
        // spans "mango", so scalar min/max can prune none of them — but
        // only the middle segment actually holds the token.
        w.append(&cat_title_batch(&["lang", "lang"], &["aardvark", "zebra"]))
            .expect("a1");
        w.commit().expect("c1");
        w.append(&cat_title_batch(&["lang"], &["mango"]))
            .expect("a2");
        w.commit().expect("c2");
        w.append(&cat_title_batch(&["lang", "lang"], &["delta", "sigma"]))
            .expect("a3");
        w.commit().expect("c3");
        assert_eq!(st.reader().n_superfiles(), 3);

        let reader = st.reader();
        let provider = SupertableProvider::new(
            st.options().scalar_schema(),
            reader.manifest().clone(),
            st.options().store.clone(),
            st.options().disk_cache.clone(),
            reader.tombstone_cache.clone(),
        );
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("rt");

        // Index HELPS: the term bloom prunes the two wide-range segments
        // that min/max could not, leaving only the real holder.
        assert_eq!(
            rt.block_on(provider.surviving_segment_count(&[col("title").eq(lit("mango"))])),
            1,
            "FTS bloom prunes to the single token holder"
        );

        // Index can't help a full scan — every segment is read.
        assert_eq!(
            rt.block_on(provider.surviving_segment_count(&[])),
            3,
            "no predicate → full scan, nothing pruned"
        );

        // Non-FTS predicate present in every segment: no bloom to use,
        // and min/max can't prune (all categories equal) → nothing
        // pruned. This is the honest "index doesn't help" case.
        assert_eq!(
            rt.block_on(provider.surviving_segment_count(&[col("category").eq(lit("lang"))])),
            3,
            "non-FTS predicate matching all segments prunes nothing"
        );
    }
}
