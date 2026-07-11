// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Catalog layer — [`Connection`] and the `connect` entry points.
//!
//! A `Connection` is rooted at a URI (local dir, object-store prefix, or
//! `memory://`) and owns a `name → table` catalog. It is the entry point
//! to the public API: open a connection, then create / open / drop / list
//! tables, each of which is a [`Supertable`].
//!
//! The catalog is **validating** — `list_tables` reflects an
//! authoritative `name → record` map (persisted on the root storage for
//! durable backends, in-process for `memory://`), not a raw directory
//! scan, so it never lists a table that can't be opened.

mod index_spec;
mod manifest;
mod options;
mod search_tvf;
mod uri;

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::record_batch::RecordBatch;
use arrow_schema::SchemaRef;
use dashmap::DashMap;
use datafusion::{config::Dialect, error::DataFusionError};
use futures::future::try_join_all;
pub use index_spec::IndexSpec;
use manifest::{
    TableEntry, VectorEntry, commit_catalog, read_catalog, schema_from_ipc, schema_to_ipc,
};
pub use options::{ColdFetchMode, ConnectOptions};
use tokio::runtime::Runtime;
use tracing::{debug, info};
use uri::{Backend, parse_uri};

use crate::{
    InfinoError,
    config::DEFAULT_CONNECTION_BUDGET_BYTES,
    memory::{ConnectionMemoryBudget, budgeted_session_context},
    runtime_bridge::{bridge_on_runtime, bridge_sync_to_async, shared_io_runtime},
    storage::{StorageError, StorageProvider},
    superfile::{
        builder::FtsConfig,
        fts::tokenize::{AsciiLowerTokenizer, Tokenizer},
        vector::{builder::VectorConfig, distance::Metric},
    },
    supertable::{
        Supertable,
        options::{Consistency, SupertableOptions},
        reader_cache::{DiskCacheConfig, DiskCacheError, DiskCacheStore},
    },
};

/// Open (or create) a catalog rooted at `uri`.
///
/// The storage backend is derived from the URI scheme: a bare path or
/// `file://` → local filesystem, `s3://bucket/prefix` → S3,
/// `az://container/prefix` → Azure, `gs://bucket/prefix` → GCS,
/// `memory://` → in-process (non-persistent). Equivalent to
/// [`connect_with`]`(uri, ConnectOptions::default())`.
///
/// ```
/// let db = infino::connect("memory://")?;
/// assert!(db.list_tables()?.is_empty());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn connect(uri: impl AsRef<str>) -> Result<Connection, InfinoError> {
    connect_with(uri, ConnectOptions::default())
}

/// Open (or create) a catalog rooted at `uri` with explicit storage
/// configuration (credentials / region / endpoint the URI can't carry).
///
/// With `ConnectOptions::with_validate(true)`, object-store backends are
/// probed before returning, so bad credentials fail at connect rather
/// than on the first table operation.
///
/// ```
/// use infino::{connect_with, ConnectOptions};
/// let db = connect_with("memory://", ConnectOptions::new())?;
/// # let _ = db;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn connect_with(
    uri: impl AsRef<str>,
    options: ConnectOptions,
) -> Result<Connection, InfinoError> {
    let backend = parse_uri(uri.as_ref())?;
    let store = match &backend {
        Backend::Memory => CatalogStore::Memory(Mutex::new(HashMap::new())),
        _ => {
            let root = backend_to_provider(&backend, &options)?
                .expect("non-memory backend yields a storage provider");
            // Opt-in probe: fail at connect on bad credentials, not first use.
            if options.validate {
                bridge_sync_to_async(read_catalog(root.as_ref()))?;
            }
            CatalogStore::Storage {
                root,
                handles: DashMap::new(),
                building: DashMap::new(),
            }
        }
    };

    // Budget comes from `ConnectOptions`; unset falls back to the engine
    // default (measure-only today). The `config.yaml` default takes a separate
    // path (`apply_config`), so `connect` never reads config.
    let connection_memory_budget = ConnectionMemoryBudget::from_budget_bytes(
        options
            .connection_memory_budget_bytes
            .unwrap_or(DEFAULT_CONNECTION_BUDGET_BYTES),
    );

    debug!(backend = ?backend, validate = options.validate, "catalog connected");
    Ok(Connection {
        inner: Arc::new(ConnectionInner {
            backend,
            options,
            store,
            connection_memory_budget,
        }),
    })
}

/// A catalog connection. Cheap to clone (one `Arc`); clones share the
/// same catalog.
#[derive(Clone)]
pub struct Connection {
    inner: Arc<ConnectionInner>,
}

struct ConnectionInner {
    backend: Backend,
    options: ConnectOptions,
    store: CatalogStore,
    /// Per-connection memory budget, minted once at `connect` and shared
    /// (cloned `Arc`) into every table's `SupertableOptions`. See
    /// [`crate::memory`].
    connection_memory_budget: Arc<ConnectionMemoryBudget>,
}

/// Where the `name → table` map lives. Durable backends persist it on the
/// root storage under optimistic concurrency; `memory://` keeps it (and
/// the tables themselves) in-process.
enum CatalogStore {
    Memory(Mutex<HashMap<String, Supertable>>),
    /// Durable backend. `handles` is the warm cache (name → live `Supertable`);
    /// `building` is a per-name lock guarding the build or evict of an entry.
    /// Both, because the read must be lock-free but the build must be single:
    /// two `Supertable`s for one name would race their cold-fetch finalizers on
    /// the same cache file (a SIGBUS in the mmap path).
    ///
    /// Lifecycle of one name:
    ///   1. `open_table` / `query_sql` checks `handles` first: a hit is a
    ///      lock-free clone, the common path (`memory://` memoizes the same way
    ///      via `Memory`).
    ///   2. A miss takes that name's `building` lock, re-checks `handles` (a
    ///      peer may have just built it), then builds the `Supertable` once and
    ///      inserts it. Same-name openers queue on the lock so exactly one store
    ///      is built; different names build in parallel.
    ///   3. `create_table` inserts under the same lock, `drop_table` evicts
    ///      under it. So build, create, and drop of one name never overlap, and
    ///      a dropped name is never left behind in `handles`.
    Storage {
        root: Arc<dyn StorageProvider>,
        /// Warm cache of live handles. Sharded so concurrent queries on one
        /// `Connection` don't serialize on a lock.
        handles: DashMap<String, Supertable>,
        /// Per-name build/evict lock. Never removed once created: a concurrent
        /// opener may hold or await the `Arc`, so evicting it mid-use would let
        /// two builds proceed.
        ///
        /// One empty `Arc<Mutex<()>>` therefore lingers per distinct name ever
        /// seen. We can bound it with refcount-gated eviction (drop only when no
        /// one holds the `Arc`) later.
        building: DashMap<String, Arc<Mutex<()>>>,
    },
}

impl Connection {
    /// Create a new table named `name` with the given Arrow `schema` and
    /// search `indexes`. Fails with [`InfinoError::AlreadyExists`] if a
    /// table of that name already exists. Returns the open handle.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use infino::arrow_schema::{DataType, Field, Schema};
    /// use infino::{connect, IndexSpec};
    ///
    /// let db = connect("memory://")?;
    /// let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// let posts = db.create_table("posts", schema, IndexSpec::new().fts("body"))?;
    /// assert_eq!(db.list_tables()?, ["posts"]);
    /// # let _ = posts;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn create_table(
        &self,
        name: &str,
        schema: SchemaRef,
        indexes: IndexSpec,
    ) -> Result<Supertable, InfinoError> {
        validate_name(name).map_err(|e| e.with_context("create_table", Some(name)))?;
        let (fts_cfg, vec_cfg) = indexes.to_configs();
        let tokenizer = table_tokenizer(&indexes);

        match &self.inner.store {
            CatalogStore::Memory(map) => {
                let opts = build_options(
                    schema,
                    fts_cfg,
                    vec_cfg,
                    tokenizer,
                    None,
                    Arc::clone(&self.inner.connection_memory_budget),
                )
                .map_err(|e| e.with_context("create_table", Some(name)))?;
                let handle = Supertable::create(opts)
                    .map_err(|e| InfinoError::from(e).with_context("create_table", Some(name)))?;
                let mut map = map.lock().expect("catalog mutex poisoned");
                if map.contains_key(name) {
                    return Err(InfinoError::AlreadyExists(name.to_string())
                        .with_context("create_table", Some(name)));
                }
                map.insert(name.to_string(), handle.clone());
                info!(table = name, backend = "memory", "created table");
                Ok(handle)
            }
            CatalogStore::Storage {
                root,
                handles,
                building,
            } => {
                // Record what was actually used to build the table, so
                // `open_table` reconstructs matching options (the
                // supertable's options-hash check then validates them).
                let vectors: Vec<VectorEntry> = vec_cfg
                    .iter()
                    .map(|vc| VectorEntry {
                        column: vc.column.clone(),
                        dim: vc.dim,
                        n_cent: vc.n_cent,
                        metric: metric_to_str(vc.metric).to_string(),
                    })
                    .collect();
                // Physical subtree is unique per creation, not just the
                // table name. `drop_table` is logical — it unregisters the
                // name but leaves the bytes in place — so reusing `<root>/
                // <name>` would make a same-name re-create silently re-open
                // the dropped table's committed data (or fail the
                // options-hash check on a schema change) instead of
                // yielding a fresh, empty table. The catalog name stays the
                // stable identity; `location` is the storage path.
                let location = unique_location(name);
                let entry = TableEntry {
                    location: location.clone(),
                    schema_ipc: schema_to_ipc(&schema)
                        .map_err(|e| e.with_context("create_table", Some(name)))?,
                    fts: indexes.fts_columns().to_vec(),
                    vectors,
                    created_at_unix: now_unix(),
                };

                let table_storage =
                    backend_to_provider(&self.inner.backend.join(&location), &self.inner.options)
                        .map_err(|e| e.with_context("create_table", Some(name)))?
                        .expect("non-memory backend yields a storage provider");
                // Disk cache is keyed on the stable name (not the unique
                // location) so the producer and a later reopener share one
                // cache directory; superfile keys carry the location, so a
                // re-created table never reads a dropped generation's bytes.
                let disk_cache = build_disk_cache(&self.inner.options, &table_storage, name)
                    .map_err(|e| e.with_context("create_table", Some(name)))?;
                let mut opts = build_options(
                    schema,
                    fts_cfg,
                    vec_cfg,
                    tokenizer,
                    Some(table_storage),
                    Arc::clone(&self.inner.connection_memory_budget),
                )
                .map_err(|e| e.with_context("create_table", Some(name)))?;
                if let Some(cache) = disk_cache {
                    opts = opts.with_disk_cache(cache);
                }

                // Match `open_table`'s memoized handles: Strong keeps every
                // query re-checking the manifest pointer (see `open_table`).
                opts = opts.with_read_consistency(Consistency::Strong);

                // Create the physical table at its unique location, then
                // register the name. A losing racer that also created a
                // (distinct) location just orphans its empty subtree; the
                // catalog OCC below decides the single name winner.
                let handle = Supertable::create(opts)
                    .map_err(|e| InfinoError::from(e).with_context("create_table", Some(name)))?;

                // Gate the commit + memo insert: else a racing `open_table`
                // sees the commit, misses the memo, and builds a rival store.
                let gate = single_flight_gate(building, name);
                let _built = gate.lock().expect("catalog build gate poisoned");

                let name_owned = name.to_string();
                bridge_sync_to_async(commit_catalog(root.as_ref(), move |body| {
                    if body.tables.contains_key(&name_owned) {
                        return Err(InfinoError::AlreadyExists(name_owned.clone()));
                    }
                    body.tables.insert(name_owned.clone(), entry.clone());
                    Ok(())
                }))
                .map_err(|e| e.with_context("create_table", Some(name)))?;

                // Seed the memo: `query_sql` reads back through this same
                // handle, so in-process writes are visible at once.
                handles.insert(name.to_string(), handle.clone());

                info!(table = name, location = %location, "created table");
                Ok(handle)
            }
        }
    }

    /// Open an existing table by name. Fails with
    /// [`InfinoError::NotFound`] if no such table is registered.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use infino::arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, IndexSpec};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # db.create_table("posts", schema, IndexSpec::new().fts("body"))?;
    /// let posts = db.open_table("posts")?;
    /// # let _ = posts;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn open_table(&self, name: &str) -> Result<Supertable, InfinoError> {
        debug!(table = name, "opening table");
        match &self.inner.store {
            CatalogStore::Memory(map) => map
                .lock()
                .expect("catalog mutex poisoned")
                .get(name)
                .cloned()
                .ok_or_else(|| {
                    InfinoError::NotFound(name.to_string()).with_context("open_table", Some(name))
                }),

            CatalogStore::Storage {
                root,
                handles,
                building,
            } => {
                // Warm path: lock-free sharded lookup, no serialization.
                if let Some(handle) = handles.get(name) {
                    return Ok(handle.clone());
                }

                // Cold path: build once under the gate. Blocks here if a
                // same-name peer is mid-build (same `Arc`, same mutex); the
                // winner builds, the rest wake to find a warm `handles`.
                let gate = single_flight_gate(building, name);
                let _built = gate.lock().expect("catalog build gate poisoned");

                // A peer may have built it while we waited on the gate.
                if let Some(handle) = handles.get(name) {
                    return Ok(handle.clone());
                }

                let (body, _etag) = bridge_sync_to_async(read_catalog(root.as_ref()))
                    .map_err(|e| e.with_context("open_table", Some(name)))?;
                let entry = body.tables.get(name).ok_or_else(|| {
                    InfinoError::NotFound(name.to_string()).with_context("open_table", Some(name))
                })?;

                let schema = schema_from_ipc(&entry.schema_ipc)
                    .map_err(|e| e.with_context("open_table", Some(name)))?;
                // Rebuild the index spec from the recorded declarations and
                // lower it through the *same* path `create_table` used, so
                // the defaults it applies (rotation seed, rerank codec) are
                // identical and the table's options-hash check passes.
                let mut spec = IndexSpec::new();
                for column in &entry.fts {
                    spec = spec.fts(column.clone());
                }
                for v in &entry.vectors {
                    spec = spec.vector(
                        v.column.clone(),
                        v.dim,
                        v.n_cent,
                        metric_from_str(&v.metric)
                            .map_err(|e| e.with_context("open_table", Some(name)))?,
                    );
                }
                let (fts_cfg, vec_cfg) = spec.to_configs();
                let tokenizer = table_tokenizer(&spec);

                let table_storage = backend_to_provider(
                    &self.inner.backend.join(&entry.location),
                    &self.inner.options,
                )
                .map_err(|e| e.with_context("open_table", Some(name)))?
                .expect("non-memory backend yields a storage provider");

                // Cache directory is keyed on the stable name, matching
                // `create_table` (the on-storage subtree is `entry.location`).
                let disk_cache = build_disk_cache(&self.inner.options, &table_storage, name)
                    .map_err(|e| e.with_context("open_table", Some(name)))?;
                let mut opts = build_options(
                    schema,
                    fts_cfg,
                    vec_cfg,
                    tokenizer,
                    Some(table_storage),
                    Arc::clone(&self.inner.connection_memory_budget),
                )
                .map_err(|e| e.with_context("open_table", Some(name)))?;
                if let Some(cache) = disk_cache {
                    opts = opts.with_disk_cache(cache);
                }
                // Strong: re-check the manifest pointer per query (cheap,
                // short-circuits when unchanged), matching the old rebuild's
                // freshness without its cost.
                opts = opts.with_read_consistency(Consistency::Strong);
                let handle = Supertable::open(opts)
                    .map_err(|e| InfinoError::from(e).with_context("open_table", Some(name)))?;
                handles.insert(name.to_string(), handle.clone());

                Ok(handle)
            }
        }
    }

    /// Remove a table from the catalog. **Idempotent**: dropping a table that
    /// is not registered is a no-op success, not an error. A caller may retry a
    /// drop whose first attempt committed the removal but whose success was not
    /// observed (a lost response, or a proxy retrying on a timeout); the retry
    /// finds the table already gone and must still succeed. This matches the
    /// object store's own delete semantics (deleting a missing key is not an
    /// error), which the whole engine is built on.
    ///
    /// Unregistering is always logical and O(1): the `name → location`
    /// entry leaves the catalog, and readers pinned to a pre-drop
    /// snapshot keep working. `purge` additionally deletes the table's
    /// storage subtree (its unique per-creation location) after the
    /// catalog commit — the name is gone first, so a crash mid-purge
    /// can only leave unreferenced orphans, never a half-deleted live
    /// table. When the table was already absent there is nothing to purge.
    /// For `memory://`, tables live in-process and free with the
    /// last handle, so `purge` has nothing extra to do.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use infino::arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, IndexSpec};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # db.create_table("posts", schema, IndexSpec::new().fts("body"))?;
    /// db.drop_table("posts", true)?; // purge: reclaim the bytes too
    /// assert!(db.list_tables()?.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn drop_table(&self, name: &str, purge: bool) -> Result<(), InfinoError> {
        info!(table = name, purge, "dropping table");
        match &self.inner.store {
            CatalogStore::Memory(map) => {
                // Idempotent: an absent table is a no-op success, so a retried
                // drop never spuriously fails.
                map.lock().expect("catalog mutex poisoned").remove(name);
                Ok(())
            }
            CatalogStore::Storage {
                root,
                handles,
                building,
            } => {
                // Gate the evict + commit: else a racing `open_table` that read
                // the pre-commit catalog re-inserts the handle after we evict,
                // and the warm path keeps serving the dropped table.
                let gate = single_flight_gate(building, name);
                let _dropping = gate.lock().expect("catalog build gate poisoned");

                // Evict first: a later create/open rebuilds fresh, and this
                // frees the handle's `DiskCacheStore`.
                handles.remove(name);

                // Capture the removed entry's location out of the OCC
                // closure; on a retry the freshest body is re-read, so
                // the last successful attempt's location wins.
                // Idempotent: a missing entry is a no-op (a retried drop whose
                // first attempt already committed the removal must still
                // succeed). `location` then stays `None`, so the purge below is
                // skipped — there is nothing left to reclaim.
                let mut location: Option<String> = None;
                bridge_sync_to_async(commit_catalog(root.as_ref(), |body| {
                    if let Some(entry) = body.tables.remove(name) {
                        location = Some(entry.location);
                    }
                    Ok(())
                }))
                .map_err(|e| e.with_context("drop_table", Some(name)))?;
                if let (true, Some(location)) = (purge, location) {
                    // Delete everything under the table's unique
                    // location. Listing is component-aware, so a sibling
                    // location sharing a string prefix never matches;
                    // deletes are idempotent, so re-running after a
                    // partial failure converges.
                    bridge_sync_to_async(async {
                        let objects = root.list_with_prefix(&location).await?;
                        try_join_all(objects.iter().map(|uri| root.delete(uri))).await?;
                        Ok::<(), StorageError>(())
                    })
                    .map_err(|e| InfinoError::from(e).with_context("drop_table", Some(name)))?;
                }
                Ok(())
            }
        }
    }

    /// List the names of every table registered in this catalog,
    /// alphabetically.
    ///
    /// ```
    /// # let db = infino::connect("memory://")?;
    /// let names: Vec<String> = db.list_tables()?;
    /// # let _ = names;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn list_tables(&self) -> Result<Vec<String>, InfinoError> {
        match &self.inner.store {
            CatalogStore::Memory(map) => {
                let mut names: Vec<String> = map
                    .lock()
                    .expect("catalog mutex poisoned")
                    .keys()
                    .cloned()
                    .collect();
                names.sort();
                Ok(names)
            }
            CatalogStore::Storage { root, .. } => {
                let (body, _etag) = bridge_sync_to_async(read_catalog(root.as_ref()))?;
                Ok(body.tables.into_keys().collect())
            }
        }
    }

    /// Run SQL across the tables in this catalog. Every relation the query
    /// names is resolved through the catalog and registered into one
    /// DataFusion session, so cross-table joins and aggregations work.
    /// Returns the collected result batches.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use infino::arrow_array::{LargeStringArray, RecordBatch};
    /// # use infino::arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, IndexSpec};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # let posts = db.create_table("posts", schema.clone(), IndexSpec::new().fts("body"))?;
    /// # posts.append(&RecordBatch::try_new(schema, vec![Arc::new(LargeStringArray::from(vec!["hello"]))])?)?;
    /// let rows = db.query_sql("SELECT _id, body FROM posts")?;
    /// assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(sql = sql))
    )]
    pub fn query_sql(&self, sql: &str) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(sql, "running sql query");

        // Gate SQL heap on the connection budget: DataFusion allocates the
        // working set (sort / aggregate / join), so its pool is the gate.
        let ctx = budgeted_session_context(&self.inner.connection_memory_budget)
            .map_err(|e| InfinoError::Query(e.to_string()).with_context("query_sql", None))?;

        // Resolve the relations the query names and register each that is a
        // catalog table. Unknown names (CTEs, search TVFs, aliases) are
        // skipped — the planner resolves those by other means or errors.
        let statement = ctx
            .state()
            .sql_to_statement(sql, &Dialect::Generic)
            .map_err(|e| InfinoError::Query(e.to_string()).with_context("query_sql", None))?;
        let refs = ctx
            .state()
            .resolve_table_references(&statement)
            .map_err(|e| InfinoError::Query(e.to_string()).with_context("query_sql", None))?;

        let mut seen = HashSet::new();
        let mut handles: Vec<Supertable> = Vec::new();
        for r in &refs {
            let name = r.table().to_string();
            if !seen.insert(name.clone()) {
                continue;
            }
            match self.open_table(&name) {
                Ok(table) => {
                    table.register_into(&ctx, &name).map_err(|e| {
                        InfinoError::Query(e.to_string()).with_context("query_sql", None)
                    })?;
                    handles.push(table);
                }
                Err(InfinoError::NotFound(_)) => {}
                Err(e) => return Err(e.with_context("query_sql", None)),
            }
        }

        // Search TVFs resolve their leading table-name argument through
        // the catalog at call time (so a table named only inside a TVF —
        // not as a `FROM` relation — still resolves).
        search_tvf::register_search_tvfs(&ctx, self.clone());

        let sql = sql.to_owned();
        let drive = async move {
            let df = ctx
                .sql(&sql)
                .await
                .map_err(|e| InfinoError::Query(e.to_string()).with_context("query_sql", None))?;

            // A RecordBatch carries the schema; an empty Vec does not. Capture
            // the output schema before collect() consumes the DataFrame so a
            // zero-row result returns one empty batch with the projected
            // schema, rather than a schema-less Vec — which the Python binding's
            // Table.from_batches([]) can't build from.
            let output_schema: SchemaRef = df.schema().inner().clone();
            let batches = df
                .collect()
                .await
                .map_err(|e| sql_exec_error(e).with_context("query_sql", None))?;
            if batches.is_empty() {
                Ok(vec![RecordBatch::new_empty(output_schema)])
            } else {
                Ok(batches)
            }
        };
        // A query that names a `FROM` catalog table drives on that table's
        // runtime; otherwise the connection's own. The fallback still has to
        // be multi-thread: a table-free query can be a search TVF, which
        // fans out object-store reads under the hood.
        match handles.first() {
            Some(table) => table
                .block_on_query(drive)
                .map_err(|e: InfinoError| e.with_context("query_sql", None)),
            None => bridge_on_runtime(drive, &self.query_runtime())
                .map_err(|e: InfinoError| e.with_context("query_sql", None)),
        }
    }

    /// Runtime for the table-free `query_sql` fallback.
    fn query_runtime(&self) -> Arc<Runtime> {
        shared_io_runtime()
    }
}

/// Build `SupertableOptions` from a schema + lowered configs, attaching
/// `storage` when present (absent → in-memory table).
fn build_options(
    schema: SchemaRef,
    fts: Vec<FtsConfig>,
    vectors: Vec<VectorConfig>,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    storage: Option<Arc<dyn StorageProvider>>,
    connection_memory_budget: Arc<ConnectionMemoryBudget>,
) -> Result<SupertableOptions, InfinoError> {
    let mut opts = SupertableOptions::new(schema, fts, vectors, tokenizer)?;
    if let Some(s) = storage {
        opts = opts.with_storage(s);
    }
    // Set last so no builder step can reset the shared connection budget.
    opts.connection_memory_budget = connection_memory_budget;
    Ok(opts)
}

/// Map a SQL execution error to the public error: a budget exhaustion becomes
/// [`InfinoError::OverBudget`], anything else a generic query error.
fn sql_exec_error(e: DataFusionError) -> InfinoError {
    match e {
        DataFusionError::ResourcesExhausted(msg) => InfinoError::OverBudget(msg),
        other => InfinoError::Query(other.to_string()),
    }
}

/// The v1 default tokenizer, required iff the spec has FTS columns.
fn table_tokenizer(indexes: &IndexSpec) -> Option<Arc<dyn Tokenizer>> {
    if indexes.has_fts() {
        Some(Arc::new(AsciiLowerTokenizer))
    } else {
        None
    }
}

/// Construct the storage provider for `backend` (None for `memory://`).
fn backend_to_provider(
    backend: &Backend,
    options: &ConnectOptions,
) -> Result<Option<Arc<dyn StorageProvider>>, InfinoError> {
    use crate::storage::{
        AzureStorageProvider, GcsStorageProvider, LocalFsStorageProvider, S3StorageProvider,
    };

    let provider: Option<Arc<dyn StorageProvider>> = match backend {
        Backend::Memory => None,
        Backend::LocalFs { root } => Some(Arc::new(LocalFsStorageProvider::new(root.clone())?)),
        Backend::S3 { bucket, prefix } => Some(Arc::new(S3StorageProvider::new_with_prefix(
            bucket,
            prefix,
            &options.storage_options,
        )?)),
        Backend::Azure { container, prefix } => Some(Arc::new(
            AzureStorageProvider::new_with_prefix(container, prefix, &options.storage_options)?,
        )),
        Backend::Gcs { bucket, prefix } => Some(Arc::new(GcsStorageProvider::new_with_prefix(
            bucket,
            prefix,
            &options.storage_options,
        )?)),
    };
    Ok(provider)
}

/// Build a per-table disk cache from the connection's options, or `None`
/// when no cache directory is configured. Rooted at `<cache_dir>/<name>`
/// so tables don't share cache files; the byte budget applies per table.
fn build_disk_cache(
    options: &ConnectOptions,
    storage: &Arc<dyn StorageProvider>,
    name: &str,
) -> Result<Option<Arc<DiskCacheStore>>, InfinoError> {
    let Some(cache_root) = options.cache_dir.as_ref() else {
        return Ok(None);
    };
    let mut cfg = DiskCacheConfig {
        cache_root: cache_root.join(name),
        cold_fetch_mode: options.cold_fetch_mode.to_internal(),
        ..Default::default()
    };
    if let Some(budget) = options.cache_budget_bytes {
        cfg.disk_budget_bytes = budget;
    }
    let cache = DiskCacheStore::new_unpinned(Arc::clone(storage), cfg).map_err(|e| {
        if let DiskCacheError::Config(msg) = e {
            InfinoError::Config(msg)
        } else {
            InfinoError::Io(e.to_string())
        }
    })?;
    Ok(Some(cache))
}

/// The per-name single-flight gate, created on first use. Returned as an owned
/// `Arc` (not a `DashMap` reference) so the caller locks it *after* the map
/// access returns, never holding a shard across the build's blocking I/O.
fn single_flight_gate(building: &DashMap<String, Arc<Mutex<()>>>, name: &str) -> Arc<Mutex<()>> {
    building
        .entry(name.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

/// Table names are flat, case-sensitive `[A-Za-z0-9_-]+` identifiers
/// that may not start with `_` — they are SQL identifiers and
/// object-store path segments, and the `_`-prefixed namespace is
/// reserved for catalog/table internals (`_catalog/`, `_supertable/`).
fn validate_name(name: &str) -> Result<(), InfinoError> {
    let ok = !name.is_empty()
        && !name.starts_with('_')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if ok {
        Ok(())
    } else {
        Err(InfinoError::Backend(format!(
            "invalid table name {name:?}: use non-empty [A-Za-z0-9_-], not starting with '_'"
        )))
    }
}

/// A unique-per-creation physical subtree for a table. The catalog name
/// is the stable identity; this is only the storage location, made
/// unique so a `drop_table` (logical by default — without `purge` it
/// leaves the bytes in place)
/// followed by a re-create of the same name lands on a fresh subtree
/// rather than re-opening the dropped table's committed data. Stays a
/// single path segment (same depth as the old `<root>/<name>`).
fn unique_location(name: &str) -> String {
    /// Process-local tiebreaker so two creations within the same
    /// nanosecond tick still get distinct locations.
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{name}-{nanos:x}-{seq:x}")
}

/// The metric's lowercased name (`"cosine"` / `"l2sq"` / `"negdot"`),
/// matching the manifest's encoding. An explicit map — not the `Debug`
/// repr — so the on-disk catalog encoding can't drift if `Metric`'s
/// `Debug` ever changes.
fn metric_to_str(m: Metric) -> &'static str {
    match m {
        Metric::Cosine => "cosine",
        Metric::L2Sq => "l2sq",
        Metric::NegDot => "negdot",
    }
}

/// Inverse of [`metric_to_str`].
fn metric_from_str(s: &str) -> Result<Metric, InfinoError> {
    match s {
        "cosine" => Ok(Metric::Cosine),
        "l2sq" => Ok(Metric::L2Sq),
        "negdot" => Ok(Metric::NegDot),
        other => Err(InfinoError::Backend(format!(
            "unknown vector metric {other:?}"
        ))),
    }
}

/// Seconds since the Unix epoch (0 if the clock is before the epoch).
fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path, sync::Arc, thread};

    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};

    use super::*;
    use crate::{
        BoolMode,
        test_helpers::{build_title_batch, schema_id_title},
    };

    const TOP_K: usize = 10;

    /// Total rows across the materialized search batches.
    fn n_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|b| b.num_rows()).sum()
    }

    #[test]
    fn memory_create_open_search_drop() {
        let conn = connect("memory://").expect("connect");
        let table = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table");
        table
            .append(&build_title_batch(&["the quick brown fox"]))
            .expect("append");

        assert_eq!(conn.list_tables().expect("list"), vec!["docs".to_string()]);

        // Re-open by name and search.
        let reopened = conn.open_table("docs").expect("open_table");
        let hits = reopened
            .bm25_search("title", "fox", TOP_K, BoolMode::Or, None)
            .expect("bm25_search");
        assert_eq!(n_rows(&hits), 1, "expected one hit for 'fox'");

        conn.drop_table("docs", false).expect("drop_table");
        assert!(conn.list_tables().expect("list").is_empty());
        assert!(matches!(
            conn.open_table("docs"),
            Err(InfinoError::NotFound(_))
        ));
    }

    #[test]
    fn create_table_range_only_with_cache_dir_is_rejected() {
        let dir = std::env::temp_dir().join(format!("infino-test-ro-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("mkdir");
        let cache_dir = dir.join("cache");
        let opts = ConnectOptions::new()
            .with_cache_dir(&cache_dir)
            .with_cold_fetch_mode(ColdFetchMode::RangeOnly);
        let conn = connect_with(format!("file://{}", dir.display()), opts)
            .expect("connect succeeds; validation is deferred to table creation");
        let err = conn
            .create_table("t", schema_id_title(), IndexSpec::new().fts("title"))
            .expect_err("range_only + cache_dir must be rejected at table creation");
        assert!(
            matches!(err, InfinoError::Config(_)),
            "expected InfinoError::Config, got: {err:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn create_table_cache_dir_with_non_range_only_mode_is_accepted() {
        let dir = std::env::temp_dir().join(format!("infino-test-hybrid-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("mkdir");
        let cache_dir = dir.join("cache");
        let opts = ConnectOptions::new()
            .with_cache_dir(&cache_dir)
            .with_cold_fetch_mode(ColdFetchMode::HybridWithPrefetch);
        let conn = connect_with(format!("file://{}", dir.display()), opts).expect("connect");
        conn.create_table("t", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("cache_dir + HybridWithPrefetch must be accepted");
        let _ = fs::remove_dir_all(&dir);
    }

    /// Regression: on durable storage, `open_table` on a table that was
    /// created but never appended to must succeed and yield an empty,
    /// usable table. `create` leaves no pointer file until the first commit,
    /// so a fresh `open` — any reconnect (another process, a restart) before
    /// the first append — must treat the missing pointer as an empty table
    /// rather than failing. Previously it surfaced a "manifest load error",
    /// and the create handle only worked because it never went through `open`.
    #[test]
    fn durable_open_before_first_append_yields_empty_usable_table() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");

        // Create, but do NOT append through the returned handle.
        let _created = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table");

        // Open fresh — the reconnect path. This must not error.
        let opened = conn
            .open_table("docs")
            .expect("open_table before first append");

        // Starts empty.
        let before = opened
            .bm25_search("title", "fox", TOP_K, BoolMode::Or, None)
            .expect("bm25_search on empty table");
        assert_eq!(n_rows(&before), 0, "freshly opened table starts empty");

        // Fully usable: the first commit lands through the reopened handle,
        // then the query round-trips.
        opened
            .append(&build_title_batch(&["the quick brown fox"]))
            .expect("append via reopened handle");
        let hits = opened
            .bm25_search("title", "fox", TOP_K, BoolMode::Or, None)
            .expect("bm25_search after append");
        assert_eq!(n_rows(&hits), 1, "expected one hit for 'fox' after append");
    }

    /// Row count via SQL — used by the memoization tests below.
    fn count_rows(conn: &Connection, table: &str) -> i64 {
        let batches = conn
            .query_sql(&format!("SELECT COUNT(*) FROM {table}"))
            .expect("count query");
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("COUNT(*) yields Int64")
            .value(0)
    }

    /// The memoized storage handle must reuse its warm disk cache across
    /// `query_sql` calls: the first query cold-fetches the superfile, the
    /// second hits the cache and does no further cold fetch. Guards the
    /// `open_table` handle memoization for durable backends.
    #[test]
    fn storage_query_sql_reuses_warm_disk_cache_across_calls() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = tempfile::tempdir().expect("cache dir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();

        // Writer connection creates + fills the table. Its own writer path
        // populates the in-memory reader tier, so a reader that did NOT write
        // the superfile is needed to exercise the disk-cache cold path.
        let writer = connect(&uri).expect("connect writer");
        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["the quick brown fox"]))
            .expect("append");

        // Fresh reader connection with a disk cache: empty in-memory tier, so
        // the first read cold-fetches through the disk cache. A projecting
        // scan (reads `title`) forces a real superfile read, unlike `COUNT(*)`
        // which is answered from manifest stats.
        let reader = connect_with(
            &uri,
            ConnectOptions::new().with_cache_dir(cache.path().to_path_buf()),
        )
        .expect("connect reader");

        reader.query_sql("SELECT title FROM docs").expect("scan q1");
        let cold_after_q1 = reader
            .open_table("docs")
            .expect("open")
            .stats()
            .n_cold_fetches
            .expect("disk cache attached");
        assert!(
            cold_after_q1 > 0,
            "first query should cold-fetch the superfile"
        );

        // Query 2 reuses the memoized handle: it must hit the warm cache, not
        // re-fetch. Before memoization this rebuilt the store and cold-fetched
        // again.
        reader.query_sql("SELECT title FROM docs").expect("scan q2");
        let cold_after_q2 = reader
            .open_table("docs")
            .expect("open")
            .stats()
            .n_cold_fetches
            .expect("disk cache attached");
        assert_eq!(
            cold_after_q2, cold_after_q1,
            "second query must reuse the warm disk cache, not cold-fetch again"
        );
    }

    /// A server holds one `Connection` and fans out concurrent queries. Many
    /// parallel first-opens of the same table must single-flight: build exactly
    /// one `Supertable`/`DiskCacheStore`, so the one superfile is cold-fetched
    /// once, not once per racing thread. A double-build would spin up rival
    /// stores that each cold-fetch (and race their finalizers on the same cache
    /// file, the SIGBUS in the mmap path).
    #[test]
    fn storage_concurrent_first_opens_build_one_store() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = tempfile::tempdir().expect("cache dir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();

        let writer = connect(&uri).expect("connect writer");
        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["the quick brown fox"]))
            .expect("append");

        let reader = connect_with(
            &uri,
            ConnectOptions::new().with_cache_dir(cache.path().to_path_buf()),
        )
        .expect("connect reader");

        // 8 threads race through open_table for the same, not-yet-open table.
        thread::scope(|s| {
            let joins: Vec<_> = (0..8)
                .map(|_| s.spawn(|| reader.query_sql("SELECT title FROM docs").expect("scan")))
                .collect();
            for j in joins {
                j.join().expect("query thread");
            }
        });

        // One store built (single-flight) and one superfile, so exactly one
        // cold fetch despite 8 concurrent queries. More would mean rival stores.
        let cold = reader
            .open_table("docs")
            .expect("open")
            .stats()
            .n_cold_fetches
            .expect("disk cache attached");
        assert_eq!(
            cold, 1,
            "concurrent first-opens must build one store and cold-fetch the superfile once"
        );
    }

    /// Sequential self-heal: dropping a table clears its memoized handle, so a
    /// later `open_table` reads the catalog and reports `NotFound` rather than
    /// serving the dropped table from the warm memo fast path.
    #[test]
    fn storage_drop_invalidates_memo_then_open_is_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");
        conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["fox"]))
            .expect("append");
        // Warm the memo so the drop must actively evict it.
        conn.open_table("docs").expect("open before drop");

        conn.drop_table("docs", true).expect("drop");

        assert!(
            matches!(conn.open_table("docs"), Err(InfinoError::NotFound(_))),
            "open after drop must be NotFound, not a stale memoized handle"
        );
    }

    /// A retried `drop_table` must be idempotent. In a distributed deployment a
    /// caller retries a drop whose first attempt committed the catalog removal
    /// but whose response was not observed as success (a later purge step
    /// failed, or a proxy retried on a timeout). The retry then finds the table
    /// already gone: it must succeed as a no-op, not hard-error `NotFound`, or
    /// the caller sees a spurious "not found" for a drop that in fact succeeded
    /// (and, under a retry loop, exhausts its budget failing on every attempt).
    #[test]
    fn storage_drop_table_is_idempotent_on_retry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");
        conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["fox"]))
            .expect("append");

        // First drop removes the table from the catalog and purges its bytes.
        conn.drop_table("docs", true).expect("first drop succeeds");
        assert!(
            conn.list_tables().expect("list").is_empty(),
            "the table is gone from the catalog after the first drop"
        );

        // The retry: dropping an already-removed table must be a no-op success,
        // not a NotFound error.
        conn.drop_table("docs", true)
            .expect("a retried drop of an already-removed table must be idempotent");
    }

    /// `drop_table` racing `open_table` on the same name must not leave a stale
    /// memoized handle: an open that read the pre-commit catalog must not
    /// re-insert after the drop evicts. The `building` gate (held across evict +
    /// commit) serializes them, so once the drop settles the table stays gone.
    /// Loop many rounds to hit the window; post-join `open_table` is `NotFound`.
    #[test]
    fn storage_concurrent_drop_and_open_never_serves_dropped() {
        const ROUNDS: usize = 20;
        const OPENERS: usize = 4;

        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");

        for round in 0..ROUNDS {
            let name = format!("t{round}");
            conn.create_table(&name, schema_id_title(), IndexSpec::new().fts("title"))
                .expect("create_table")
                .append(&build_title_batch(&["fox"]))
                .expect("append");
            // Warm the memo, so a racing open can observe (and must not restore)
            // an entry the drop is removing.
            conn.open_table(&name).expect("open before race");

            thread::scope(|s| {
                for _ in 0..OPENERS {
                    s.spawn(|| {
                        // Both Ok (raced before the drop) and NotFound (raced
                        // after) are valid mid-race; neither may panic.
                        let _ = conn.open_table(&name);
                    });
                }
                s.spawn(|| {
                    conn.drop_table(&name, false).expect("drop");
                });
            });

            assert!(
                matches!(conn.open_table(&name), Err(InfinoError::NotFound(_))),
                "round {round}: table must stay dropped, no stale handle in the memo"
            );
        }
    }

    /// `create_table` racing `open_table` on the same name: benign, but must
    /// stay so. The gate serializes the create's commit + memo insert against
    /// the open, so no rival store is memoized. A racing open sees the table or
    /// `NotFound`, never a panic or other error; afterward the table queries.
    #[test]
    fn storage_concurrent_create_and_open_stays_consistent() {
        const ROUNDS: usize = 20;
        const OPENERS: usize = 4;

        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");

        for round in 0..ROUNDS {
            let name = format!("t{round}");

            thread::scope(|s| {
                s.spawn(|| {
                    conn.create_table(&name, schema_id_title(), IndexSpec::new().fts("title"))
                        .expect("create_table")
                        .append(&build_title_batch(&["fox"]))
                        .expect("append");
                });
                for _ in 0..OPENERS {
                    s.spawn(|| {
                        // Pre-commit opens see NotFound; post-commit opens see
                        // the table. Both are fine; a panic or other error is
                        // not.
                        match conn.open_table(&name) {
                            Ok(_) | Err(InfinoError::NotFound(_)) => {}
                            Err(e) => panic!("round {round}: unexpected open error: {e}"),
                        }
                    });
                }
            });

            // After the race the table is present and the appended row is
            // readable through the memoized handle (one store, writes visible).
            assert_eq!(
                count_rows(&conn, &name),
                1,
                "round {round}: created table must be queryable with its row"
            );
        }
    }

    /// A commit made after a table has been queried (so its handle is
    /// memoized and warm) must be visible to the next query. Guards that
    /// memoizing the handle does not serve a stale manifest.
    #[test]
    fn storage_query_sql_sees_commit_after_the_handle_is_memoized() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");
        let table = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table");
        table
            .append(&build_title_batch(&["one"]))
            .expect("append 1");

        // First query memoizes + warms the handle.
        assert_eq!(count_rows(&conn, "docs"), 1);

        // Commit a second row through a handle from the same connection
        // (the memoized one), then re-query: the memoized handle must see it.
        conn.open_table("docs")
            .expect("open")
            .append(&build_title_batch(&["two"]))
            .expect("append 2");
        assert_eq!(
            count_rows(&conn, "docs"),
            2,
            "the memoized handle must reflect the new commit, not a stale snapshot"
        );
    }

    /// Cross-connection freshness: memoizing must not pin the snapshot a handle
    /// first saw. A second connection commits on the same storage; the first
    /// connection's memoized handle sees it on the next query, because it opens
    /// `Strong` and re-reads the manifest pointer. This is the guarantee the old
    /// rebuild-per-query gave for free and the reason memoized handles use
    /// `Strong` rather than the default bounded staleness.
    #[test]
    fn storage_memoized_handle_sees_another_connections_commit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();

        let writer = connect(&uri).expect("connect writer");
        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["one"]))
            .expect("append 1");

        // A separate connection memoizes + warms its own handle for `docs`.
        let reader = connect(&uri).expect("connect reader");
        assert_eq!(count_rows(&reader, "docs"), 1);

        // The other connection commits a second row.
        writer
            .open_table("docs")
            .expect("reopen for append")
            .append(&build_title_batch(&["two"]))
            .expect("append 2");

        assert_eq!(
            count_rows(&reader, "docs"),
            2,
            "memoized handle must reflect another connection's commit, not a pinned snapshot"
        );
    }

    #[test]
    fn connection_memory_budget_is_measured_by_default() {
        let conn = connect("memory://").expect("connect");
        assert_eq!(conn.inner.connection_memory_budget.limit(), None);
    }

    #[test]
    fn with_connection_memory_budget_bytes_mints_a_bounded_budget_at_the_gate() {
        let conn = connect_with(
            "memory://",
            ConnectOptions::new().with_connection_memory_budget_bytes(1000),
        )
        .expect("connect");
        // 90% headroom gate: 1000 configured -> 900 enforced.
        assert_eq!(conn.inner.connection_memory_budget.limit(), Some(900));
    }

    #[test]
    fn zero_connection_memory_budget_is_measured() {
        let conn = connect_with(
            "memory://",
            ConnectOptions::new().with_connection_memory_budget_bytes(0),
        )
        .expect("connect");
        assert_eq!(conn.inner.connection_memory_budget.limit(), None);
    }

    #[test]
    fn all_tables_share_one_connection_memory_budget() {
        let conn = connect_with(
            "memory://",
            ConnectOptions::new().with_connection_memory_budget_bytes(1000),
        )
        .expect("connect");
        let a = conn
            .create_table("a", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create a");
        let b = conn
            .create_table("b", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create b");

        // Every table sees the one budget the connection minted, not a copy.
        assert!(Arc::ptr_eq(
            &a.options().connection_memory_budget,
            &b.options().connection_memory_budget
        ));
        assert!(Arc::ptr_eq(
            &a.options().connection_memory_budget,
            &conn.inner.connection_memory_budget
        ));
    }

    #[test]
    fn reopened_table_shares_the_connection_memory_budget() {
        // open_table threads the same shared budget as create_table.
        let conn = connect_with(
            "memory://",
            ConnectOptions::new().with_connection_memory_budget_bytes(1000),
        )
        .expect("connect");
        conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create");
        let reopened = conn.open_table("docs").expect("open");

        assert!(Arc::ptr_eq(
            &reopened.options().connection_memory_budget,
            &conn.inner.connection_memory_budget
        ));
    }

    #[test]
    fn duplicate_create_is_already_exists() {
        let conn = connect("memory://").expect("connect");
        conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("first create");
        let again = conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"));
        assert!(matches!(again, Err(InfinoError::AlreadyExists(_))));
    }

    #[test]
    fn open_missing_is_not_found() {
        let conn = connect("memory://").expect("connect");
        assert!(matches!(
            conn.open_table("nope"),
            Err(InfinoError::NotFound(_))
        ));
    }

    #[test]
    fn invalid_table_name_rejected() {
        let conn = connect("memory://").expect("connect");
        let bad = conn.create_table("has space", schema_id_title(), IndexSpec::new());
        assert!(bad.is_err());
    }

    #[test]
    fn underscore_prefixed_name_rejected() {
        // The `_`-prefixed namespace is reserved for catalog/table
        // internals (`_catalog/`, `_supertable/`).
        let conn = connect("memory://").expect("connect");
        assert!(
            conn.create_table("_catalog", schema_id_title(), IndexSpec::new())
                .is_err()
        );
        assert!(
            conn.create_table("_hidden", schema_id_title(), IndexSpec::new())
                .is_err()
        );
    }

    #[test]
    fn drop_then_recreate_same_name_is_empty() {
        // `drop_table` is logical (leaves bytes in place); a re-create of
        // the same name must yield a FRESH, empty table — not re-open the
        // dropped generation's committed rows.
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");

        let first = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create");
        first
            .append(&build_title_batch(&["a lazy sleeping fox"]))
            .expect("append");
        assert_eq!(
            n_rows(
                &first
                    .bm25_search("title", "fox", TOP_K, BoolMode::Or, None)
                    .expect("search")
            ),
            1
        );

        conn.drop_table("docs", false).expect("drop");

        // Re-create the same name: the old subtree is orphaned, the new
        // table starts empty.
        let second = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("recreate");
        assert_eq!(
            n_rows(
                &second
                    .bm25_search("title", "fox", TOP_K, BoolMode::Or, None)
                    .expect("search")
            ),
            0,
            "re-created table must not resurrect the dropped table's rows"
        );
    }

    #[test]
    fn create_without_append_reopens_as_empty_table() {
        // A table created but never appended to is still durably
        // registered in the catalog. Reopening it in a fresh connection
        // (i.e. after a program restart) must succeed and yield an empty
        // table — not fail because no manifest pointer was ever written.
        // The pointer is only written on the first commit, so `create`
        // alone leaves the physical table with a catalog entry but no
        // `_supertable/current`; open must tolerate that the same way
        // `create` does when it probes and finds no pointer.
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();

        {
            let conn = connect(&uri).expect("connect");
            conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
                .expect("create");
            // No append/commit: the manifest pointer is never written.
        }

        // Reopen in a fresh connection, simulating a program restart.
        let conn = connect(&uri).expect("reconnect");
        assert_eq!(conn.list_tables().expect("list"), vec!["docs".to_string()]);
        let docs = conn
            .open_table("docs")
            .expect("open a created-but-empty table");
        assert_eq!(
            n_rows(
                &docs
                    .bm25_search("title", "fox", TOP_K, BoolMode::Or, None)
                    .expect("search")
            ),
            0,
            "a created-but-empty table has no hits"
        );
    }

    #[test]
    fn drop_with_purge_reclaims_the_storage_subtree() {
        /// Count regular files under `dir` whose path contains a
        /// component starting with `prefix` (the table's unique
        /// `<name>-<nanos>-<seq>` location).
        fn files_under_location(dir: &Path, prefix: &str) -> usize {
            let mut n = 0;
            let mut stack = vec![dir.to_path_buf()];
            while let Some(d) = stack.pop() {
                let Ok(entries) = fs::read_dir(&d) else {
                    continue;
                };
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        stack.push(path);
                    } else if path
                        .components()
                        .any(|c| c.as_os_str().to_string_lossy().starts_with(prefix))
                    {
                        n += 1;
                    }
                }
            }
            n
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");

        let table = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create");
        table
            .append(&build_title_batch(&["a lazy sleeping fox"]))
            .expect("append");
        assert!(
            files_under_location(dir.path(), "docs-") > 0,
            "committed table must have bytes under its unique location"
        );

        conn.drop_table("docs", true).expect("drop with purge");
        assert!(conn.list_tables().expect("list").is_empty());
        assert_eq!(
            files_under_location(dir.path(), "docs-"),
            0,
            "purge must delete every object under the dropped table's location"
        );
    }

    #[test]
    fn query_sql_resolves_tables_by_catalog_name() {
        use arrow_array::Int64Array;

        let conn = connect("memory://").expect("connect");
        let docs = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create docs");
        docs.append(&build_title_batch(&["the quick brown fox", "a lazy dog"]))
            .expect("append docs");
        let more = conn
            .create_table("more", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create more");
        more.append(&build_title_batch(&["hello world"]))
            .expect("append more");

        // Resolved by catalog name (not the old hardcoded `supertable`).
        let batches = conn
            .query_sql("SELECT COUNT(*) AS n FROM docs")
            .expect("count docs");
        let n = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 count")
            .value(0);
        assert_eq!(n, 2, "docs has two rows");

        // Two catalog tables registered into one query.
        let rows: usize = conn
            .query_sql("SELECT title FROM docs UNION ALL SELECT title FROM more")
            .expect("union across tables")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(rows, 3, "2 from docs + 1 from more");
    }

    // Many distinct group keys force DataFusion's aggregate to build a real
    // hash table, so its memory pool (the connection budget) is exercised.
    fn many_distinct_titles() -> Vec<String> {
        (0..4000)
            .map(|i| format!("distinct title number {i} with some filler text"))
            .collect()
    }

    fn append_titles(conn: &Connection) -> usize {
        let docs = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create");
        let titles = many_distinct_titles();
        let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
        docs.append(&build_title_batch(&refs)).expect("append");
        titles.len()
    }

    /// Ingest the fixture on a measured connection, then return a 0-byte-gate
    /// connection over the same durable store plus the row count. Ingest is
    /// gated by the budget too, so setup runs on a measured connection and only
    /// the query connection carries the gate. Hold the `TempDir`: dropping it
    /// deletes the store.
    fn tiny_budget_conn_after_ingest() -> (tempfile::TempDir, Connection, usize) {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let n = append_titles(&connect(&uri).expect("writer connect"));
        let conn = connect_with(
            &uri,
            ConnectOptions::new().with_connection_memory_budget_bytes(1),
        )
        .expect("query connect");
        (dir, conn, n)
    }

    const HEAVY_GROUP_BY: &str = "SELECT title, COUNT(*) AS n FROM docs GROUP BY title";

    #[test]
    fn query_sql_under_measure_only_default_is_never_refused() {
        // Default budget only measures, so even a heavy aggregate runs.
        let conn = connect("memory://").expect("connect");
        let n = append_titles(&conn);
        let out = conn
            .query_sql(HEAVY_GROUP_BY)
            .expect("measure-only never refuses");
        assert_eq!(n_rows(&out), n);
    }

    #[test]
    fn query_sql_under_a_generous_budget_succeeds() {
        // 1 GiB is far more than the query needs, so the gate never trips.
        let conn = connect_with(
            "memory://",
            ConnectOptions::new().with_connection_memory_budget_bytes(1 << 30),
        )
        .expect("connect");
        let n = append_titles(&conn);
        let out = conn.query_sql(HEAVY_GROUP_BY).expect("well under 1 GiB");
        assert_eq!(n_rows(&out), n);
    }

    #[test]
    fn query_sql_over_a_tiny_budget_is_refused_as_over_budget() {
        // 0-byte gate: the aggregate can't reserve its first byte, and spilling
        // needs memory it doesn't have, so it's refused as OverBudget.
        let (_dir, conn, _n) = tiny_budget_conn_after_ingest();
        let err = conn
            .query_sql(HEAVY_GROUP_BY)
            .expect_err("a 0-byte gate refuses the aggregate");
        assert!(
            matches!(err, InfinoError::OverBudget(_)),
            "expected OverBudget, got {err:?}"
        );
    }

    #[test]
    fn query_sql_streaming_scan_is_not_refused_under_a_tiny_budget() {
        // A projection streams (no buffering), so it reserves nothing and runs
        // even at a 0-byte gate: the budget bounds sort/aggregate/join, not scans.
        let (_dir, conn, n) = tiny_budget_conn_after_ingest();
        let out = conn
            .query_sql("SELECT title FROM docs")
            .expect("a streaming scan is not gated");
        assert_eq!(n_rows(&out), n);
    }

    #[test]
    fn query_sql_zero_row_filter_preserves_projected_schema() {
        let conn = connect("memory://").expect("connect");
        let docs = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create docs");
        docs.append(&build_title_batch(&["alpha", "beta"]))
            .expect("append");

        // Same projection with rows gives the ground-truth schema to compare against.
        let with_rows = conn
            .query_sql("SELECT _id, title FROM docs")
            .expect("query with rows");
        let expected_schema = with_rows[0].schema();

        let batches = conn
            .query_sql("SELECT _id, title FROM docs WHERE title = 'no_match'")
            .expect("zero-row query must not error");
        assert!(
            !batches.is_empty(),
            "must contain at least one (empty) batch"
        );
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 0, "no rows should match");
        assert_eq!(
            batches[0].schema(),
            expected_schema,
            "zero-row schema must match the with-rows schema"
        );
    }

    #[test]
    fn query_sql_zero_row_group_by_preserves_projected_schema() {
        // GROUP BY is a different DataFusion operator path from a filtered scan;
        // zero matching groups must still produce a schema-bearing empty batch.
        let conn = connect("memory://").expect("connect");
        let docs = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create docs");
        docs.append(&build_title_batch(&["alpha", "beta"]))
            .expect("append");

        // The same aggregate over matching rows gives the ground-truth schema:
        // an aggregate's output schema (group keys + aggregate exprs) must be
        // identical whether or not any group forms.
        let with_groups = conn
            .query_sql("SELECT title, COUNT(*) AS n FROM docs GROUP BY title")
            .expect("GROUP BY with rows");
        let expected_schema = with_groups[0].schema();

        let batches = conn
            .query_sql(
                "SELECT title, COUNT(*) AS n FROM docs WHERE title = 'no_match' GROUP BY title",
            )
            .expect("zero-row GROUP BY must not error");
        assert!(
            !batches.is_empty(),
            "must contain at least one (empty) batch"
        );
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 0, "no groups should form");
        assert_eq!(
            batches[0].schema(),
            expected_schema,
            "zero-group schema must match the with-groups schema"
        );
    }

    #[test]
    fn query_sql_bm25_search_tvf_resolves_table() {
        let conn = connect("memory://").expect("connect");
        let docs = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create docs");
        docs.append(&build_title_batch(&["the quick brown fox", "a lazy dog"]))
            .expect("append");

        // Leading table-name argument selects the catalog table.
        let rows: usize = conn
            .query_sql("SELECT _id, score FROM bm25_search('docs', 'title', 'fox', 10)")
            .expect("bm25_search tvf")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(rows, 1, "one doc matches 'fox'");

        // An unknown table in the TVF is a clean planning error.
        assert!(
            conn.query_sql("SELECT _id FROM bm25_search('nope', 'title', 'fox', 10)")
                .is_err()
        );
    }

    #[test]
    fn query_sql_search_tvf_over_storage_does_not_panic() {
        // Regression: a search TVF takes the table-free runtime fallback (it
        // names its table in an argument, not a `FROM` relation). Over a
        // storage backend it fans out object-store reads that need a
        // multi-thread runtime; this panicked before the fix. `memory://`
        // has no such reads, so the bug only showed on localfs.
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");
        let docs = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create docs");
        docs.append(&build_title_batch(&["the quick brown fox", "a lazy dog"]))
            .expect("append");

        let rows: usize = conn
            .query_sql("SELECT _id, score FROM bm25_search('docs', 'title', 'fox', 10)")
            .expect("bm25_search tvf over storage")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(rows, 1, "one doc matches 'fox'");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn connection_drops_cleanly_inside_async_runtime() {
        // The sync API supports being called from inside the caller's
        // runtime (the bridge uses `block_in_place`), and `query_sql` builds
        // the connection runtime eagerly. Dropping the last `Connection`
        // here must not trip tokio's drop-runtime-in-async-context panic.
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");
        let docs = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create docs");
        docs.append(&build_title_batch(&["the quick brown fox"]))
            .expect("append");
        // Table-free TVF → builds the connection runtime on this thread.
        conn.query_sql("SELECT _id FROM bm25_search('docs', 'title', 'fox', 10)")
            .expect("query");

        drop(docs);
        drop(conn); // must not panic
    }

    #[test]
    fn query_sql_match_tvfs_resolve_table() {
        let conn = connect("memory://").expect("connect");
        let docs = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create docs");
        docs.append(&build_title_batch(&[
            "the quick brown fox",
            "a lazy dog",
            "quick thinking",
        ]))
        .expect("append");

        // Unranked token match: rows containing the token, any order.
        let rows: usize = conn
            .query_sql("SELECT _id FROM token_match('docs', 'title', 'quick')")
            .expect("token_match tvf")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(rows, 2, "two docs contain 'quick'");

        // Set algebra over index-bounded candidate sets.
        let rows: usize = conn
            .query_sql(
                "SELECT _id FROM token_match('docs', 'title', 'quick') \
                 EXCEPT \
                 SELECT _id FROM token_match('docs', 'title', 'fox')",
            )
            .expect("EXCEPT over token_match")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(rows, 1, "'quick thinking' has quick but not fox");

        // Exact raw-string match.
        let rows: usize = conn
            .query_sql("SELECT _id FROM exact_match('docs', 'title', 'a lazy dog')")
            .expect("exact_match tvf")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(rows, 1, "one doc equals the raw string exactly");
    }

    /// The remaining catalog-level search TVFs — `bm25_search_prefix`,
    /// `vector_search`, and `hybrid_search` — resolve their leading
    /// table-name argument and forward the rest to the table's search
    /// kernels. Exercises each `*CatalogFunc::call` over a table that
    /// carries both an FTS index and a vector index.
    #[test]
    fn query_sql_prefix_vector_and_hybrid_tvfs_resolve_table() {
        use crate::Metric;

        /// Embedding dimension for the fixture's vector column.
        const DIM: usize = 16;
        /// IVF centroid count; kmeans needs at least this many rows.
        const N_CENT: usize = 4;
        /// Rows in the fixture (one-hot vectors at dims 0..ROWS).
        const ROWS: usize = 4;
        /// Top-k requested by the vector / hybrid queries.
        const TOP_K: usize = 4;

        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                ),
                false,
            ),
        ]));

        // Four docs; doc `i` is one-hot at dim `i`, so a one-hot query
        // at dim 0 is the exact nearest neighbour of doc 0.
        let batch = {
            use arrow_array::{FixedSizeListArray, Float32Array, LargeStringArray};
            let titles = ["rust async", "python data", "rust systems", "go rust"];
            let mut flat = Vec::<f32>::with_capacity(ROWS * DIM);
            for i in 0..ROWS {
                for d in 0..DIM {
                    flat.push(if d == i { 1.0 } else { 0.0 });
                }
            }
            let field = Arc::new(Field::new("item", DataType::Float32, true));
            let list = FixedSizeListArray::new(
                field,
                DIM as i32,
                Arc::new(Float32Array::from(flat)),
                None,
            );
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(LargeStringArray::from(titles.to_vec())),
                    Arc::new(list),
                ],
            )
            .expect("vector batch")
        };

        let conn = connect("memory://").expect("connect");
        let table = conn
            .create_table(
                "vecs",
                schema,
                IndexSpec::new()
                    .fts("title")
                    .vector("emb", DIM, N_CENT, Metric::L2Sq),
            )
            .expect("create table");
        table.append(&batch).expect("append");

        let one_hot_0 = (0..DIM)
            .map(|d| if d == 0 { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join(",");

        // bm25_search_prefix: 'rus' expands to 'rust'.
        let prefix_rows: usize = conn
            .query_sql(&format!(
                "SELECT _id FROM bm25_search_prefix('vecs', 'title', 'rus', {TOP_K})"
            ))
            .expect("bm25_search_prefix tvf")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert!(prefix_rows >= 1, "'rus' prefix should match 'rust' docs");

        // vector_search over the catalog table.
        let vec_rows: usize = conn
            .query_sql(&format!(
                "SELECT _id FROM vector_search('vecs', 'emb', '{one_hot_0}', {TOP_K})"
            ))
            .expect("vector_search tvf")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert!(vec_rows >= 1, "vector_search should return neighbours");

        // hybrid_search fuses the FTS + vector retrievers.
        let hybrid_rows: usize = conn
            .query_sql(&format!(
                "SELECT _id FROM hybrid_search('vecs', 'title', 'rust', 'emb', '{one_hot_0}', {TOP_K})"
            ))
            .expect("hybrid_search tvf")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert!(
            hybrid_rows >= 1,
            "hybrid_search should fuse and return hits"
        );
    }

    #[test]
    fn localfs_with_disk_cache() {
        let root = tempfile::tempdir().expect("tempdir");
        let cache = tempfile::tempdir().expect("cache tempdir");
        let opts = ConnectOptions::new()
            .with_cache_dir(cache.path())
            .with_cold_fetch_mode(ColdFetchMode::HybridWithPrefetch)
            .with_cache_budget_bytes(64 * 1024 * 1024);
        let conn = connect_with(root.path().to_str().expect("utf8"), opts).expect("connect");
        let table = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create");
        table
            .append(&build_title_batch(&["the quick brown fox"]))
            .expect("append");
        let hits = table
            .bm25_search("title", "fox", TOP_K, BoolMode::Or, None)
            .expect("search");
        assert_eq!(n_rows(&hits), 1);
        // The disk cache got a per-table subdirectory.
        assert!(cache.path().join("docs").exists());
    }

    #[test]
    fn connect_with_default_options_yields_empty_memory_catalog() {
        let db = connect_with("memory://", ConnectOptions::new()).expect("connect_with");
        assert!(db.list_tables().expect("list").is_empty());
    }

    #[test]
    fn connect_does_not_probe_by_default() {
        // Default (validate off): a bogus bucket builds a provider but the
        // backend is never touched, so connect succeeds without network.
        connect("s3://no-such-bucket-xyzzy/prefix").expect("offline connect by default");
    }

    #[test]
    fn connect_gcs_uri_builds_offline() {
        // Provider construction must not dial GCS — connect is offline until
        // the first table op, exactly like the S3 case.
        connect("gs://no-such-bucket-xyzzy/prefix").expect("offline gcs connect by default");
    }

    #[test]
    fn connection_clone_shares_one_catalog() {
        let conn = connect("memory://").expect("connect");
        let clone = conn.clone();
        conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create on original");
        // The clone shares the same Arc<ConnectionInner>, so the table
        // is visible through it.
        assert_eq!(clone.list_tables().expect("list"), vec!["docs".to_string()]);
    }

    #[test]
    fn query_sql_table_free_select_uses_shared_bridge() {
        // A query naming no catalog relation falls through to the shared
        // sync->async bridge (the `handles.first()` None arm).
        let conn = connect("memory://").expect("connect");
        let batches = conn
            .query_sql("SELECT 1 AS one")
            .expect("table-free select");
        assert_eq!(n_rows(&batches), 1);
    }

    #[test]
    fn query_sql_invalid_sql_is_query_error() {
        let conn = connect("memory://").expect("connect");
        let err = conn.query_sql("NOT VALID SQL @@@");
        assert!(matches!(err, Err(InfinoError::Query(_))), "got {err:?}");
    }

    #[test]
    fn drop_missing_is_idempotent_no_op() {
        // Dropping a table that was never registered is a no-op success, not an
        // error: drop is idempotent so a retried drop is retry-safe (matches the
        // object store's delete semantics).
        let conn = connect("memory://").expect("connect");
        conn.drop_table("nope", false)
            .expect("dropping an absent table is a no-op success");
        conn.drop_table("nope", true)
            .expect("dropping an absent table with purge is also a no-op success");
    }

    #[test]
    fn empty_table_name_rejected() {
        let conn = connect("memory://").expect("connect");
        assert!(
            conn.create_table("", schema_id_title(), IndexSpec::new())
                .is_err()
        );
    }

    #[test]
    fn vector_index_round_trips_metric_through_storage_catalog() {
        use crate::Metric;

        // Exercises metric_to_str (create) + metric_from_str (open) plus
        // the VectorEntry catalog encoding across a reconnect. A
        // storage-backed catalog records the index spec and rebuilds it
        // on open, so the table's options-hash check must pass.
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();

        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 16),
                false,
            ),
        ]));

        // A FixedSizeList<Float32, 16> column of one all-zero vector,
        // committed so the physical table writes its pointer file (open
        // requires committed state).
        let one_vector = || -> RecordBatch {
            use arrow_array::{FixedSizeListArray, Float32Array, LargeStringArray};
            let values = Float32Array::from(vec![0.0_f32; 16]);
            let field = Arc::new(Field::new("item", DataType::Float32, true));
            let list = FixedSizeListArray::new(field, 16, Arc::new(values), None);
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(LargeStringArray::from(vec!["hello"])),
                    Arc::new(list),
                ],
            )
            .expect("vector batch")
        };

        {
            let conn = connect(&uri).expect("connect");
            let table = conn
                .create_table(
                    "vecs",
                    schema.clone(),
                    IndexSpec::new()
                        .fts("title")
                        .vector("embedding", 16, 4, Metric::L2Sq),
                )
                .expect("create vector table");
            table.append(&one_vector()).expect("append vector row");
        }

        // Reopen: open_table rebuilds the spec via metric_from_str and
        // validates the options hash — a mismatch would error here.
        let conn = connect(&uri).expect("reconnect");
        assert_eq!(conn.list_tables().expect("list"), vec!["vecs".to_string()]);
        conn.open_table("vecs").expect("open vector table");
    }

    /// `metric_to_str` / `metric_from_str` round-trip every `Metric`
    /// variant, and the inverse rejects an unknown name with a typed
    /// `Backend` error (the catalog's on-disk metric encoding).
    #[test]
    fn metric_str_round_trips_all_variants_and_rejects_unknown() {
        for m in [Metric::Cosine, Metric::L2Sq, Metric::NegDot] {
            let s = metric_to_str(m);
            let back = metric_from_str(s).expect("known metric round-trips");
            assert_eq!(back, m, "{m:?} did not survive the string round-trip");
        }
        assert_eq!(metric_to_str(Metric::Cosine), "cosine");
        assert_eq!(metric_to_str(Metric::L2Sq), "l2sq");
        assert_eq!(metric_to_str(Metric::NegDot), "negdot");
        assert!(matches!(
            metric_from_str("euclidean"),
            Err(InfinoError::Backend(_))
        ));
    }

    /// A duplicate `create_table` on a storage-backed (localfs) catalog
    /// hits the OCC closure's `AlreadyExists` guard, distinct from the
    /// in-memory duplicate path.
    #[test]
    fn storage_duplicate_create_is_already_exists() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");
        conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("first create");
        let again = conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"));
        assert!(matches!(again, Err(InfinoError::AlreadyExists(_))));
    }

    /// A `query_sql` statement that names the same table twice resolves
    /// it once: the dedup `continue` in the reference loop fires, and the
    /// self-join still returns the joined rows.
    #[test]
    fn query_sql_dedups_repeated_table_reference() {
        let conn = connect("memory://").expect("connect");
        let docs = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create docs");
        docs.append(&build_title_batch(&["alpha", "beta"]))
            .expect("append");
        let rows: usize = conn
            .query_sql("SELECT a.title FROM docs a JOIN docs b ON a._id = b._id")
            .expect("self-join resolves the repeated reference once")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(rows, 2, "self-join on _id pairs each row with itself");
    }

    #[test]
    fn localfs_persists_across_reconnect() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();

        {
            let conn = connect(&uri).expect("connect");
            let table = conn
                .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
                .expect("create_table");
            table
                .append(&build_title_batch(&["a lazy sleeping fox"]))
                .expect("append");
        }

        // A fresh connection to the same root sees the catalog + data.
        let conn = connect(&uri).expect("reconnect");
        assert_eq!(conn.list_tables().expect("list"), vec!["docs".to_string()]);
        let table = conn.open_table("docs").expect("open_table");
        let hits = table
            .bm25_search("title", "fox", TOP_K, BoolMode::Or, None)
            .expect("bm25_search");
        assert_eq!(
            n_rows(&hits),
            1,
            "expected the persisted doc to be searchable"
        );
    }

    /// Finding #3: public API boundaries prefix operation (+ table when
    /// known) into the InfinoError message so Display carries context.
    #[test]
    fn public_api_errors_carry_operation_and_table_context() {
        use datafusion::prelude::{col, lit};

        // --- Catalog methods know the table name ---
        let conn = connect("memory://").expect("connect");
        let err = conn.open_table("posts").expect_err("missing table");
        assert!(matches!(err, InfinoError::NotFound(_)));
        assert!(err.to_string().contains("open_table(posts):"), "got: {err}");

        conn.create_table("posts", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create posts");
        let err = conn
            .create_table("posts", schema_id_title(), IndexSpec::new().fts("title"))
            .expect_err("duplicate create");
        assert!(matches!(err, InfinoError::AlreadyExists(_)));
        assert!(
            err.to_string().contains("create_table(posts):"),
            "got: {err}"
        );

        // --- Supertable methods: operation only (no catalog name on handle) ---
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path");
        let conn = connect(uri).expect("connect");
        conn.create_table("posts", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create posts");
        let posts = conn.open_table("posts").expect("open");
        posts
            .append(&build_title_batch(&["hello world"]))
            .expect("append one row");

        let err = posts
            .update(
                col("title").eq(lit("hello world")),
                &build_title_batch(&["a", "b"]),
            )
            .expect_err("cardinality mismatch");
        assert!(matches!(err, InfinoError::Cardinality(_)));
        assert!(err.to_string().contains("update:"), "got: {err}");

        let err = posts
            .bm25_search("title", "-onlyneg", TOP_K, BoolMode::Or, None)
            .expect_err("negation-only query");
        assert!(matches!(err, InfinoError::Query(_)));
        assert!(err.to_string().contains("bm25_search:"), "got: {err}");

        let err = conn.query_sql("NOT VALID SQL @@@").expect_err("bad sql");
        assert!(matches!(err, InfinoError::Query(_)));
        assert!(err.to_string().contains("query_sql:"), "got: {err}");
    }
}
