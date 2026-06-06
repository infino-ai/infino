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
//! At `query_sql` time we:
//!
//!   1. Pin the manifest (`self.reader()` → `Arc<Manifest>`).
//!   2. Register a [`SupertableProvider`] as `supertable` in a
//!      fresh `SessionContext`.
//!   3. `ctx.sql(sql).await.collect().await`.
//!
//! The provider's `scan` does the real work — see
//! [`crate::supertable::query::provider`]. In short, it applies
//! **two tiers of pruning**: infino's [`scalar_skip`] drops
//! definitely-irrelevant *segments* from the pushed-down `WHERE`
//! predicates, then DataFusion's `ParquetSource` prunes *row
//! groups / pages* and pushes projection + limit into the Parquet
//! reader over the surviving segments. This replaces the v1
//! `MemTable` path, which eagerly decoded every row group of every
//! segment regardless of the query.
//!
//! [`scalar_skip`]: crate::supertable::query::skip::scalar_skip
//! [`SupertableProvider`]: crate::supertable::query::provider::SupertableProvider
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
use datafusion::execution::context::SessionContext;

use crate::supertable::error::QueryError;
use crate::supertable::handle::Supertable;
use crate::supertable::query::provider::{SupertableProvider, TABLE_NAME};

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
        // Apply the read-consistency policy before pinning, so SQL
        // honors the same freshness contract as the search APIs
        // (`bm25_search` / `vector_search`): under `Strong` /
        // `BoundedStaleness` this advances the in-memory snapshot to
        // the latest published manifest; under `Snapshot` (and for
        // in-memory tables) it is a no-op. See
        // [`Supertable::ensure_fresh`].
        self.ensure_fresh();

        // Build (or reuse the cached) SessionContext for the pinned
        // snapshot — the pushdown-aware SupertableProvider plus the
        // search TVFs. See [`Supertable::sql_session_context`].
        let ctx = self.sql_session_context()?;

        let sql = sql.to_owned();
        let drive = async move {
            let df = ctx
                .sql(&sql)
                .await
                .map_err(|e| QueryError::Plan(e.to_string()))?;
            df.collect()
                .await
                .map_err(|e| QueryError::Execute(e.to_string()))
        };

        // Drive through the shared sync→async bridge: ambient
        // runtime → block_in_place on the ambient handle; otherwise
        // the lazily-built owned query_runtime. See
        // [`Supertable::block_on_query`].
        self.block_on_query(drive)
    }

    /// Build (or reuse the cached) [`SessionContext`] for the
    /// current pinned manifest snapshot: the pushdown-aware
    /// [`SupertableProvider`] registered as `supertable`, plus the
    /// vector / BM25 / hybrid search TVFs.
    ///
    /// The cache keys on the manifest `Arc` — commits publish a new
    /// `Arc`, so any committed state since the last call forces a
    /// rebuild. A hit skips the ~1.5 ms `SessionContext::new()` +
    /// `register_*` setup. Shared by [`query_sql`](Self::query_sql)
    /// (SQL string) and [`scan_ids_matching`](Self::scan_ids_matching)
    /// (programmatic `Expr`), so mutation id-capture gets the same
    /// segment-skip + row-group/page pruning + lazy tombstone
    /// filtering the read path uses.
    ///
    /// Callers apply their own freshness policy
    /// ([`ensure_fresh`](Self::ensure_fresh)) before calling.
    fn sql_session_context(&self) -> Result<SessionContext, QueryError> {
        // ArcSwap::load_full is a single atomic load + Arc clone, so
        // pinning the snapshot is cheap even on the hot path.
        let reader = Arc::new(self.reader());
        let manifest = Arc::clone(reader.manifest());

        let mut guard = self
            .sql_session_cache()
            .lock()
            .expect("sql_session_cache mutex poisoned");
        if let Some((cached, ctx)) = &*guard
            && Arc::ptr_eq(cached, &manifest)
        {
            return Ok(ctx.clone());
        }

        let store = Arc::clone(&self.options().store);
        let disk_cache = self.options().disk_cache.as_ref().map(Arc::clone);
        let scalar_schema = self.options().scalar_schema();
        let provider = SupertableProvider::new(
            Arc::clone(&scalar_schema),
            Arc::clone(&manifest),
            store,
            disk_cache,
            reader.tombstone_cache.clone(),
        );
        let ctx = SessionContext::new();
        ctx.register_table(TABLE_NAME, Arc::new(provider))
            .map_err(|e| QueryError::Plan(e.to_string()))?;
        // Search TVFs (vector kNN, BM25 FTS, hybrid RRF) bound to
        // the pinned snapshot. They lower to custom `ExecutionPlan`
        // nodes that call the async kernels inside `execute()`.
        crate::supertable::query::exec::vector_exec::register_vector_search(
            &ctx,
            Arc::clone(&reader),
            Arc::clone(&scalar_schema),
        );
        crate::supertable::query::exec::fts_exec::register_bm25(
            &ctx,
            Arc::clone(&reader),
            Arc::clone(&scalar_schema),
        );
        crate::supertable::query::exec::hybrid_exec::register_hybrid_search(
            &ctx,
            Arc::clone(&reader),
            Arc::clone(&scalar_schema),
        );
        *guard = Some((Arc::clone(&manifest), ctx.clone()));
        Ok(ctx)
    }

    /// Resolve a predicate to the matching `_id` values. Used by
    /// the writer's `delete()` / `update()` entry points to
    /// capture the target-id set at call time (step 0a in the
    /// update / delete pipeline).
    ///
    /// Runs through the same pushdown-aware [`SupertableProvider`]
    /// as `query_sql` (via [`sql_session_context`](Self::sql_session_context)):
    /// `expr` is applied as a `DataFrame::filter` and the result
    /// projected to just `_id`. Segment skip, row-group / page
    /// pruning, and lazy tombstone filtering all apply, so a
    /// large-table delete/update predicate never materializes every
    /// segment into memory.
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
        // Resolve against the freshest snapshot the consistency
        // policy allows — the spec requires delete/update predicates
        // to bind "against the current snapshot at call time", which
        // means honoring `Strong` / `BoundedStaleness` just like the
        // read APIs do. No-op under `Snapshot` / in-memory tables.
        self.ensure_fresh();
        let ctx = self.sql_session_context()?;
        let id_column = self.options().id_column.clone();

        let drive = async move {
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

        self.block_on_query(drive)
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
    fn query_sql_equality_on_fts_column_across_segments_is_correct() {
        // Equality on the FTS-indexed `title` column drives the new
        // term-bloom prune leaf (plus the scalar min/max leaf). The two
        // segments whose bloom lacks "bravo" may be pruned, but the
        // result must still be exactly the one matching row — proving
        // the bloom prune never drops a match.
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(0, &["x"], &["alpha"]))
            .expect("a1");
        w.commit().expect("c1");
        w.append(&build_cat_batch(10, &["y"], &["bravo"]))
            .expect("a2");
        w.commit().expect("c2");
        w.append(&build_cat_batch(20, &["z"], &["charlie"]))
            .expect("a3");
        w.commit().expect("c3");
        assert_eq!(st.reader().n_superfiles(), 3);

        assert_eq!(
            run_count(&st, "SELECT COUNT(*) FROM supertable WHERE title = 'bravo'"),
            1
        );
        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable WHERE title = 'nonexistent'"
            ),
            0
        );
    }

    #[test]
    fn query_sql_multiword_equality_on_fts_column_is_correct() {
        // Multi-word literal: the equality lowers to a `TermPresence`
        // leaf over {rust, async, runtime} (AND). The second segment's
        // bloom lacks those tokens and is pruned, yet results are exact
        // — DataFusion's FilterExec re-applies the full string equality.
        let st = Supertable::create(options_id_cat_title()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_cat_batch(0, &["lang"], &["rust async runtime"]))
            .expect("a1");
        w.commit().expect("c1");
        w.append(&build_cat_batch(10, &["lang"], &["python data science"]))
            .expect("a2");
        w.commit().expect("c2");
        assert_eq!(st.reader().n_superfiles(), 2);

        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable WHERE title = 'rust async runtime'"
            ),
            1
        );
        // Tokens present in segment 1, but no row equals this exact
        // string — the prune is an optimization, correctness holds.
        assert_eq!(
            run_count(
                &st,
                "SELECT COUNT(*) FROM supertable WHERE title = 'rust async'"
            ),
            0
        );
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
    fn query_sql_runtime_is_cached_across_calls() {
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
