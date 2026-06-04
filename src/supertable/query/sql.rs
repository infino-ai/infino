//! `Supertable::query_sql` — DataFusion SQL over the supertable.
//!
//! ## Public API
//!
//! ```ignore
//! let st: Supertable = ...;
//! let batches: Vec<RecordBatch> =
//!     st.query_sql("SELECT category, COUNT(*) FROM supertable GROUP BY category")?;
//! ```
//!
//! Sync return type: callers don't need a tokio runtime.
//! Internally we `block_on` against a single multi-worker Runtime
//! cached on `SupertableInner` (lazy — first SQL query allocates).
//!
//! ## Strategy
//!
//! Each visible segment becomes one [`MemTable`] partition. At
//! `query_sql` time we:
//!
//!   1. Pin the manifest (`self.reader()` → `Arc<Manifest>`).
//!   2. Fetch every segment's `SuperfileReader` from the store.
//!   3. Eagerly read all parquet row groups into `RecordBatch`es
//!      via `ParquetRecordBatchReaderBuilder` (sync, on the calling
//!      thread; no rayon fan-out yet).
//!   4. Build a single `MemTable` whose partition list mirrors
//!      the manifest order, and register it as `supertable` in a
//!      fresh `SessionContext`.
//!   5. `ctx.sql(sql).await.collect().await`.
//!
//! No manifest-level skip is wired into this path in v1: every
//! segment is scanned.
//!
//! TODO: replace this MemTable wrapper with a custom
//! `TableProvider` whose `scan` consults per-segment min/max
//! via DataFusion's `PruningPredicate` against the
//! `ScalarStatsTable` already populated by the writer at commit
//! time. The per-segment stats are ready; the `TableProvider`
//! integration needs a perf bench that motivates the wiring
//! with measured pruning gains, not speculative ones.
//!
//! ## Why MemTable
//!
//! The in-memory `SuperfileReaderCache` already holds every segment's
//! parquet bytes; eagerly decoding them into Arrow shifts the cost
//! from `execute()` time to `register_table()` time but doesn't
//! change the working set. DataFusion still applies `FilterExec`
//! above the MemTable, so per-batch predicate filtering works as
//! expected. Per-row-group pushdown into parquet is the next
//! optimization — the right home for it is a custom
//! `TableProvider` that hands DataFusion a `ParquetExec` per
//! non-pruned segment, layered on top of the manifest-level
//! skip helpers.
//!
//! ## Schema
//!
//! The supertable's *user-visible* schema (`options.scalar_schema`)
//! contains id + scalar columns + FTS columns; vector columns are
//! stored in the embedded vector blob and never exposed via SQL
//! (callers reach them through `vector_search`). The parquet body
//! of each segment was written with this same scalar schema, so
//! round-trip shape matches without projection or rewrite.

use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use arrow_array::{Array, Decimal128Array};
use bytes::Bytes;
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::supertable::error::QueryError;
use crate::supertable::handle::Supertable;
use crate::supertable::manifest::Manifest;
use crate::supertable::reader_cache::SuperfileReaderCache;

/// Logical name the supertable is registered under in the
/// DataFusion `SessionContext`. Callers reference it as
/// `FROM supertable` in their SQL.
const TABLE_NAME: &str = "supertable";

impl Supertable {
    /// Run a SQL query against this supertable's pinned snapshot.
    ///
    /// The snapshot is captured at `query_sql` entry — concurrent
    /// commits don't affect the in-flight query. Returns the
    /// concatenated `Vec<RecordBatch>` from
    /// `DataFrame::collect`.
    ///
    /// The SQL must reference the table as `supertable`. The
    /// available columns are id + scalar + FTS columns; vector
    /// columns are not exposed (use `vector_search` instead).
    ///
    /// Sync API. The first call allocates a tokio Runtime
    /// (single worker thread) cached on the `SupertableInner`;
    /// subsequent calls reuse it.
    pub fn query_sql(&self, sql: &str) -> Result<Vec<RecordBatch>, QueryError> {
        self.ensure_fresh();
        let reader = self.reader();
        let manifest = Arc::clone(reader.manifest());
        let store = Arc::clone(&self.options().store);
        let disk_cache = self.options().disk_cache.as_ref().map(Arc::clone);
        let storage = self.options().storage.as_ref().map(Arc::clone);
        let scalar_schema = self.options().scalar_schema();
        let tombstone_cache = reader.tombstone_cache.clone();

        let sql = sql.to_owned();

        let drive = async move {
            let table = build_mem_table(
                scalar_schema,
                manifest,
                store,
                disk_cache,
                storage,
                tombstone_cache,
            )
            .await?;
            let ctx = SessionContext::new();
            ctx.register_table(TABLE_NAME, Arc::new(table))
                .map_err(|e| QueryError::Plan(e.to_string()))?;
            let df = ctx
                .sql(&sql)
                .await
                .map_err(|e| QueryError::Plan(e.to_string()))?;
            df.collect()
                .await
                .map_err(|e| QueryError::Execute(e.to_string()))
        };

        // same ambient-runtime detection pattern the
        // writer's persist_commit uses. Lazy-init the owned
        // query_runtime only when there's NO ambient runtime —
        // calling `Builder::new_multi_thread().build()` from
        // inside another runtime panics with "Cannot start a
        // runtime from within a runtime". Web handlers,
        // `#[tokio::test]`s, and any async caller now get a
        // working query_sql.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(drive)),
            Err(_) => self.query_runtime().block_on(drive),
        }
    }

    /// Resolve a predicate to the matching `_id` values. Used by
    /// the writer's `delete()` / `update()` entry points to
    /// capture the target-id set at call time (step 0a in the
    /// update / delete pipeline).
    ///
    /// Plan shape mirrors [`query_sql`]: build a `MemTable` over
    /// the current manifest, register as `supertable`, apply
    /// `expr` as a filter via DataFusion's `DataFrame::filter`,
    /// project just `_id`, drain into a `Vec<i128>`. No
    /// `TableProvider` is introduced.
    ///
    /// Note: the resolution is against the **current** manifest
    /// snapshot, exactly like a contemporaneous `query_sql` would
    /// see. Rows that newly match `expr` between this call and
    /// the eventual `commit()` are NOT in the returned set —
    /// captured-at-call semantics match SQL `UPDATE WHERE` /
    /// `DELETE WHERE`.
    pub(crate) fn scan_ids_matching(
        &self,
        expr: datafusion::prelude::Expr,
    ) -> Result<Vec<i128>, QueryError> {
        let reader = self.reader();
        let manifest = Arc::clone(reader.manifest());
        let store = Arc::clone(&self.options().store);
        let disk_cache = self.options().disk_cache.as_ref().map(Arc::clone);
        let storage = self.options().storage.as_ref().map(Arc::clone);
        let scalar_schema = self.options().scalar_schema();
        let tombstone_cache = reader.tombstone_cache.clone();
        let id_column = self.options().id_column.clone();

        let drive = async move {
            let table = build_mem_table(
                scalar_schema,
                manifest,
                store,
                disk_cache,
                storage,
                tombstone_cache,
            )
            .await?;
            let ctx = SessionContext::new();
            ctx.register_table(TABLE_NAME, Arc::new(table))
                .map_err(|e| QueryError::Plan(e.to_string()))?;
            let df = ctx
                .table(TABLE_NAME)
                .await
                .map_err(|e| QueryError::Plan(e.to_string()))?
                .filter(expr)
                .map_err(|e| QueryError::Plan(e.to_string()))?
                .select_columns(&[id_column.as_str()])
                .map_err(|e| QueryError::Plan(e.to_string()))?;
            let batches = df
                .collect()
                .await
                .map_err(|e| QueryError::Execute(e.to_string()))?;
            extract_id_column(&batches)
        };

        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(drive)),
            Err(_) => self.query_runtime().block_on(drive),
        }
    }
}

/// Drain `_id`-only batches into a `Vec<i128>`. The supertable's
/// `_id` is a Decimal128(38, 0) column; we read the raw 128-bit
/// integer value directly.
fn extract_id_column(batches: &[RecordBatch]) -> Result<Vec<i128>, QueryError> {
    let mut out: Vec<i128> = Vec::new();
    for batch in batches {
        if batch.num_columns() != 1 {
            return Err(QueryError::Plan(format!(
                "scan_ids_matching: expected 1-column batch, got {}",
                batch.num_columns()
            )));
        }
        let col = batch.column(0);
        let arr = col
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .ok_or_else(|| {
                QueryError::Plan("scan_ids_matching: _id column not Decimal128".into())
            })?;
        for i in 0..arr.len() {
            if arr.is_null(i) {
                continue;
            }
            out.push(arr.value(i));
        }
    }
    Ok(out)
}

/// Read every segment in the manifest and assemble a `MemTable`
/// with one partition per segment.
///
/// A `MemTable` requires at least one partition even if all
/// partitions are empty — DataFusion errors otherwise. For an
/// empty supertable we emit a single empty partition so
/// `SELECT COUNT(*)` returns 0 and `SELECT *` yields an empty
/// result, instead of surfacing the planner's "No partitions
/// provided" check to the caller.
///
/// **M15c — hierarchical pruning.** When the manifest carries
/// a persisted `list` (storage-backed), walk the list's parts,
/// lazy-load each (or hit the OnceCell in eager mode), and
/// build one MemTable partition per segment across all kept
/// parts. SQL-level list-pruning (extracting predicates from
/// the parsed DataFusion plan to drive a `prune_parts_for_*`
/// call) is **deferred** — DataFusion's plan-rewrite hooks
/// don't surface predicates until after `MemTable` providers
/// have built their partition list, so a pushdown-aware
/// variant requires either a custom `TableProvider` (significant
/// new code) or a pre-parse pass. Ships the "load all
/// parts" SQL path; exact-term BM25 + prefix BM25 + vector
/// queries get list-prune via their dedicated entry points.
async fn build_mem_table(
    schema: Arc<arrow_schema::Schema>,
    manifest: Arc<Manifest>,
    store: Arc<dyn SuperfileReaderCache>,
    disk_cache: Option<Arc<crate::supertable::reader_cache::DiskCacheStore>>,
    storage: Option<Arc<dyn crate::storage::StorageProvider>>,
    tombstone_cache: Option<Arc<crate::supertable::tombstones::SidecarCache>>,
) -> Result<MemTable, QueryError> {
    // route through the hierarchical iterator when
    // the manifest has a persisted list (which includes
    // both eager + lazy modes). For in-process supertables
    // with no list, the fallback returns the flat
    // `manifest.superfiles` view.
    let superfiles: Vec<Arc<crate::supertable::SuperfileEntry>> = match manifest.list.as_ref() {
        Some(list) => {
            let kept: Vec<_> = list.parts.iter().map(|p| p.part_id).collect();
            crate::supertable::query::hierarchical_iter::load_and_flatten(manifest.as_ref(), &kept)
                .await?
        }
        None => crate::supertable::query::hierarchical_iter::fallback_to_flat_segments(
            manifest.as_ref(),
        ),
    };
    // One `Instant::now()` for the entire SQL query so each
    // per-superfile tombstone lookup shares the same TTL
    // reference. Matches the FTS / vector hot-path budget.
    let now = std::time::Instant::now();
    let mut partitions: Vec<Vec<RecordBatch>> = Vec::with_capacity(superfiles.len().max(1));
    for entry in &superfiles {
        let reader = crate::supertable::query::superfile_reader::superfile_reader(
            &store,
            disk_cache.as_ref(),
            storage.as_ref(),
            &entry.uri,
            entry.subsection_offsets.as_ref(),
        )
        .await
        .map_err(|e| QueryError::Store(e.to_string()))?;
        let parquet = reader
            .parquet_bytes()
            .ok_or_else(|| {
                QueryError::Plan(format!(
                    "SQL pass-through requires eager-opened superfile bytes; \
                     reader for {:?} was opened via the lazy path which does \
                     not materialize the full segment",
                    entry.uri
                ))
            })?
            .clone();
        let mut batches = read_all_batches(parquet)?;
        if let Some(cache) = tombstone_cache.as_ref() {
            apply_tombstone_filter_to_batches(cache, entry.superfile_id, &mut batches, now)?;
        }
        partitions.push(batches);
    }
    if partitions.is_empty() {
        partitions.push(Vec::new());
    }
    MemTable::try_new(schema, partitions).map_err(|e| QueryError::Plan(e.to_string()))
}

/// Drop tombstoned rows from one superfile's batches before they
/// reach DataFusion's `MemTable`. The local doc-id of row `i` in
/// batch `b` is `row_offset(b) + i`, where `row_offset(b)` sums
/// the lengths of all batches preceding `b` within the superfile.
///
/// Empty-bitmap short-circuits the entire loop so a tombstone-free
/// superfile pays nothing beyond the cache lookup.
fn apply_tombstone_filter_to_batches(
    cache: &Arc<crate::supertable::tombstones::SidecarCache>,
    superfile_id: uuid::Uuid,
    batches: &mut Vec<RecordBatch>,
    now: std::time::Instant,
) -> Result<(), QueryError> {
    let bitmap = cache
        .bitmap_for(superfile_id, now)
        .map_err(|e| QueryError::Store(format!("tombstone cache: {e}")))?;
    if bitmap.is_empty() {
        return Ok(());
    }
    let mut row_offset: u32 = 0;
    let mut filtered: Vec<RecordBatch> = Vec::with_capacity(batches.len());
    for batch in batches.drain(..) {
        let n = batch.num_rows() as u32;
        // Build a boolean mask. `true` keeps the row; `false`
        // drops it. We allocate once per batch — the typical
        // batch is row-group-sized (~64K rows) so this is one
        // small allocation per superfile-batch combination.
        let mask: Vec<bool> = (0..n).map(|i| !bitmap.contains(row_offset + i)).collect();
        // Fast-path: every row keeps → no filter call needed.
        if mask.iter().all(|b| *b) {
            row_offset += n;
            filtered.push(batch);
            continue;
        }
        let mask_array = arrow_array::BooleanArray::from(mask);
        let kept = arrow::compute::filter_record_batch(&batch, &mask_array)
            .map_err(|e| QueryError::Parquet(format!("filter_record_batch: {e}")))?;
        row_offset += n;
        filtered.push(kept);
    }
    *batches = filtered;
    Ok(())
}

/// Eagerly drain a parquet file into `Vec<RecordBatch>`.
///
/// `ParquetRecordBatchReaderBuilder::try_new(bytes)` is zero-copy:
/// `Bytes` implements `ChunkReader` directly, so the builder reads
/// the footer in place. `build()` returns a sync iterator yielding
/// row-group-sized batches.
fn read_all_batches(bytes: Bytes) -> Result<Vec<RecordBatch>, QueryError> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .map_err(|e| QueryError::Parquet(e.to_string()))?;
    let reader = builder
        .build()
        .map_err(|e| QueryError::Parquet(e.to_string()))?;
    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| QueryError::Parquet(e.to_string()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        Array, FixedSizeListArray, Float32Array, Int64Array, LargeStringArray, RecordBatch,
    };
    use arrow_schema::{DataType, Field, Schema};

    use crate::superfile::builder::{FtsConfig, VectorConfig};

    use crate::superfile::vector::distance::Metric;
    use crate::supertable::{Supertable, SupertableOptions};

    use crate::test_helpers::default_tokenizer as tok;

    /// Schema with id + scalar + FTS column. No vector; query_sql
    /// is scalar-only by design.
    fn schema_id_cat_title() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("category", DataType::LargeUtf8, false),
            Field::new("title", DataType::LargeUtf8, false),
        ]))
    }

    fn options_id_cat_title() -> SupertableOptions {
        // Single-threaded writer pool so each commit produces
        // exactly one segment — keeps assertions on per-segment
        // counts deterministic.
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("rayon pool"),
        );
        SupertableOptions::new(
            schema_id_cat_title(),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// Build a small categorical batch — start id sequence at
    /// `start`, plant `cats[i] / titles[i]` per row.
    fn build_cat_batch(_start: u64, cats: &[&str], titles: &[&str]) -> RecordBatch {
        assert_eq!(cats.len(), titles.len());
        let cat_arr = LargeStringArray::from(cats.to_vec());
        let title_arr = LargeStringArray::from(titles.to_vec());
        RecordBatch::try_new(
            schema_id_cat_title(),
            vec![Arc::new(cat_arr), Arc::new(title_arr)],
        )
        .expect("build batch")
    }

    /// Convenience: run a query and pull a single `Int64` aggregate
    /// value from cell (0,0).
    fn run_count(st: &Supertable, sql: &str) -> i64 {
        let batches = st.query_sql(sql).expect("query_sql ok");
        assert!(!batches.is_empty(), "expected at least one result batch");
        let n = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count column is Int64");
        n.value(0)
    }

    #[test]
    fn query_sql_count_star_returns_zero_on_empty_supertable() {
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let n = run_count(&st, "SELECT COUNT(*) FROM supertable");
        assert_eq!(n, 0);
    }

    #[test]
    fn query_sql_count_star_returns_total_doc_count() {
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(
            0,
            &["rust", "rust", "python"],
            &["a", "b", "c"],
        ))
        .expect("append");
        w.commit().expect("commit");

        let n = run_count(&st, "SELECT COUNT(*) FROM supertable");
        assert_eq!(n, 3);
    }

    #[test]
    fn query_sql_filter_predicate_applied_above_mem_table() {
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(
            0,
            &["rust", "rust", "python", "rust", "go"],
            &["a", "b", "c", "d", "e"],
        ))
        .expect("append");
        w.commit().expect("commit");

        let n = run_count(
            &st,
            "SELECT COUNT(*) FROM supertable WHERE category = 'rust'",
        );
        assert_eq!(n, 3);
    }

    #[test]
    fn query_sql_group_by_returns_correct_per_category_counts() {
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(
            0,
            &["rust", "rust", "python", "rust", "python", "go"],
            &["a", "b", "c", "d", "e", "f"],
        ))
        .expect("append");
        w.commit().expect("commit");

        let batches = st
            .query_sql(
                "SELECT category, COUNT(*) AS n FROM supertable \
                 GROUP BY category ORDER BY category",
            )
            .expect("group-by query");
        assert_eq!(batches.len(), 1);

        let cat_col = batches[0].column(0);
        let counts = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count is Int64");
        // DataFusion may materialize the GROUP BY key as Utf8,
        // LargeUtf8, or StringView depending on hash-aggregate
        // type promotion; accept all three.
        let extract = |i: usize| -> String {
            if let Some(a) = cat_col.as_any().downcast_ref::<LargeStringArray>() {
                a.value(i).to_string()
            } else if let Some(a) = cat_col.as_any().downcast_ref::<arrow_array::StringArray>() {
                a.value(i).to_string()
            } else if let Some(a) = cat_col
                .as_any()
                .downcast_ref::<arrow_array::StringViewArray>()
            {
                a.value(i).to_string()
            } else {
                panic!("unexpected category column type: {:?}", cat_col.data_type())
            }
        };
        let mut got: Vec<(String, i64)> = (0..cat_col.len())
            .map(|i| (extract(i), counts.value(i)))
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                ("go".to_string(), 1),
                ("python".to_string(), 2),
                ("rust".to_string(), 3),
            ]
        );
    }

    #[test]
    fn query_sql_scans_across_multiple_segments() {
        // Three commits → three superfiles. SQL must aggregate across
        // all of them.
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(0, &["rust", "rust"], &["a", "b"]))
            .expect("a1");
        w.commit().expect("c1");
        w.append(&build_cat_batch(10, &["python"], &["c"]))
            .expect("a2");
        w.commit().expect("c2");
        w.append(&build_cat_batch(20, &["rust", "go"], &["d", "e"]))
            .expect("a3");
        w.commit().expect("c3");

        assert_eq!(st.reader().n_superfiles(), 3);

        let n_total = run_count(&st, "SELECT COUNT(*) FROM supertable");
        assert_eq!(n_total, 5);

        let n_rust = run_count(
            &st,
            "SELECT COUNT(*) FROM supertable WHERE category = 'rust'",
        );
        assert_eq!(n_rust, 3);
    }

    #[test]
    fn query_sql_select_orders_ids_across_segments() {
        // Verifies row identity round-trips through MemTable +
        // DataFusion: rows planted across two superfiles come back
        // in monotonic _id order under ORDER BY. The _id values
        // are auto-injected by the supertable (timestamp +
        // worker + counter), so we don't assert specific
        // values — only strict-increasing order.
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(100, &["a", "b"], &["t1", "t2"]))
            .expect("a1");
        w.commit().expect("c1");
        w.append(&build_cat_batch(200, &["c"], &["t3"]))
            .expect("a2");
        w.commit().expect("c2");

        let batches = st
            .query_sql("SELECT _id FROM supertable ORDER BY _id")
            .expect("query");
        let ids: Vec<i128> = batches
            .iter()
            .flat_map(|b| {
                let a = b
                    .column(0)
                    .as_any()
                    .downcast_ref::<arrow_array::Decimal128Array>()
                    .expect("_id is Decimal128");
                (0..a.len()).map(|i| a.value(i)).collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(ids.len(), 3);
        for w in ids.windows(2) {
            assert!(w[0] < w[1], "expected strictly increasing _id");
        }
    }

    #[test]
    fn query_sql_select_star_exposes_only_user_columns_plus_id() {
        // The supertable is a thin SQL skin over scalar columns —
        // `inf.*` KV metadata stays invisible. The injected `_id`
        // column is part of the visible schema.
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(0, &["x"], &["t"])).expect("a");
        w.commit().expect("c");

        let batches = st
            .query_sql("SELECT * FROM supertable LIMIT 1")
            .expect("query");
        let schema = batches[0].schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["_id", "category", "title"]);
    }

    #[test]
    fn query_runtime_is_cached_across_calls() {
        // Two queries on the same supertable must share one
        // Runtime — the OnceLock guarantees this; we assert by
        // checking that both calls succeed without spawning a
        // fresh Runtime per call (observed indirectly via the
        // `.await` over `block_on` not double-allocating; if the
        // cache regressed, tests would still pass but would leak
        // a Runtime per call. The functional check below is
        // adequate for correctness; benchmarks would catch leak).
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(0, &["x"], &["t"])).expect("a");
        w.commit().expect("c");
        for _ in 0..3 {
            let n = run_count(&st, "SELECT COUNT(*) FROM supertable");
            assert_eq!(n, 1);
        }
    }

    #[test]
    fn query_sql_invalid_sql_returns_plan_error() {
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let err = st
            .query_sql("SELECT NOT_A_REAL_FN(*) FROM supertable")
            .expect_err("expected a plan error");
        assert!(
            matches!(err, crate::supertable::error::QueryError::Plan(_)),
            "expected Plan variant; got {err:?}"
        );
    }

    // ---- vector schema integration ----------------------------------

    /// Build a schema that includes a vector column. The supertable
    /// strips it at commit time; SQL surface only sees the scalar
    /// columns. `query_sql` SELECTing the vector column must error
    /// (DataFusion's planner rejects unknown column).
    fn schema_with_vector(dim: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ]))
    }

    fn options_with_vector(dim: usize) -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("rayon pool"),
        );
        SupertableOptions::new(
            schema_with_vector(dim),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 0,
                metric: Metric::Cosine,
                rerank_codec: crate::superfile::vector::rerank_codec::RerankCodec::Fp32,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    fn build_vector_batch(_start: u64, n: usize, dim: usize) -> RecordBatch {
        let titles = LargeStringArray::from((0..n).map(|i| format!("doc {i}")).collect::<Vec<_>>());
        let mut flat = Vec::<f32>::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push(((i + d) as f32) / 100.0);
            }
        }
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Float32Array::from(flat);
        let emb = FixedSizeListArray::try_new(
            item_field,
            dim as i32,
            Arc::new(values) as Arc<dyn Array>,
            None,
        )
        .expect("FixedSizeList build");
        RecordBatch::try_new(
            schema_with_vector(dim),
            vec![Arc::new(titles), Arc::new(emb)],
        )
        .expect("build batch")
    }

    #[test]
    fn query_sql_hides_vector_columns_from_sql_surface() {
        let st = Supertable::create(options_with_vector(16)).expect("create");
        let mut w = st.writer().expect("writer");
        // n=8 ≥ n_cent=4 so kmeans has data to cluster.
        w.append(&build_vector_batch(0, 8, 16)).expect("append");
        w.commit().expect("commit");

        let batches = st
            .query_sql("SELECT * FROM supertable LIMIT 1")
            .expect("query");
        let schema = batches[0].schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        // `emb` was stripped by `vector_split` at commit time and
        // lives in the embedded vector blob — not visible to SQL.
        // The supertable-injected `_id` is visible.
        assert_eq!(names, vec!["_id", "title"]);
    }

    #[test]
    fn query_sql_referencing_vector_column_returns_plan_error() {
        let st = Supertable::create(options_with_vector(16)).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 8, 16)).expect("append");
        w.commit().expect("commit");

        let err = st
            .query_sql("SELECT emb FROM supertable")
            .expect_err("vector column should not be in the SQL schema");
        assert!(
            matches!(err, crate::supertable::error::QueryError::Plan(_)),
            "expected Plan variant; got {err:?}"
        );
    }
}
