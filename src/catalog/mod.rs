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

mod cancel;
mod index_spec;
mod manifest;
mod options;
#[cfg(feature = "remote")]
mod remote;
mod search_tvf;
mod table;
mod uri;

#[cfg(feature = "graph-index")]
use std::sync::Weak;
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{Error as IoError, ErrorKind},
    ops::ControlFlow,
    panic,
    path::Path,
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::record_batch::RecordBatch;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
pub use cancel::QueryCancel;
use dashmap::DashMap;
use datafusion::{
    common::{
        runtime::SpawnedTask,
        tree_node::{TreeNode, TreeNodeRecursion},
    },
    execution::{context::SQLOptions, session_state::SessionState},
    logical_expr::{BinaryExpr, Operator},
    prelude::{Expr, SessionContext},
    sql::{
        parser::{DFParserBuilder, Statement as DFStatement},
        sqlparser::{
            ast::{
                CastKind, DataType as SqlDataType, ExactNumberInfo, Expr as SqlExpr,
                Statement as SqlStatement, Value as SqlValue, visit_expressions_mut,
                visit_statements,
            },
            dialect::GenericDialect,
            keywords::Keyword,
            tokenizer::{Token, Tokenizer as SqlTokenizer},
        },
    },
};
use futures::future::try_join_all;
pub(crate) use index_spec::DEFAULT_ROT_SEED;
pub use index_spec::{FtsField, IndexSpec};
use manifest::{
    TableEntry, VectorEntry, commit_catalog, read_catalog, schema_from_ipc, schema_to_ipc,
};
pub use options::{ColdFetchMode, ConnectOptions};
pub use table::Supertable;
use tokio::{runtime::Runtime, sync::Semaphore};
use tracing::{Instrument, debug, info, warn};
use uri::{Backend, parse_uri};

/// Most `AND` / `OR` connectives allowed in one SQL statement or mutation predicate: past a few
/// thousand, DataFusion's planner recursion overflows the stack and aborts. `IN (...)` is one
/// node, never counted.
pub(crate) const MAX_PREDICATE_CONNECTIVES: usize = 1024;

/// Fewest bytes one connective can occupy in SQL text; text shorter than
/// `MIN_BYTES_PER_CONNECTIVE * MAX_PREDICATE_CONNECTIVES` cannot reach the cap and skips the scan.
const MIN_BYTES_PER_CONNECTIVE: usize = 3;

/// Most digits a whole-number literal can carry and still be read exactly: `Decimal128`'s
/// precision, which is also the `_id` column's (`DECIMAL(38, 0)`).
const MAX_EXACT_INTEGER_DIGITS: usize = 38;

/// The file in a table's disk-cache directory naming the storage location of
/// the table generation its cached bytes belong to. The directory is keyed on
/// the stable table name, which a drop and re-create reuses; superfile keys
/// carry the location, so a later generation never reads an earlier one's
/// bytes, but nothing else would ever reclaim them. Not a superfile name, so
/// the cache's own scan leaves it alone.
const TABLE_CACHE_LOCATION_FILE: &str = ".table-location";

/// What `query_sql` answers a write with, from either read-only check.
const READ_ONLY_REFUSAL: &str =
    "query_sql is read-only; writes go through the table's append / update / delete API";

/// Most table opens a cancel can cut short that may be in flight at once, process-wide. Each holds
/// a thread of the I/O runtime's blocking pool (tokio's default 512), and an open needs more of
/// that pool for its own work (manifest decode, local file reads), so the bound stays well below
/// it: opens waiting on one table's build can never take every thread and stall the build they
/// wait on. Past it, an open runs on the calling thread, as it did before a cancel could reach it.
const MAX_CANCELLABLE_OPENS: usize = 64;

/// Slots for the opens [`MAX_CANCELLABLE_OPENS`] bounds.
static CANCELLABLE_OPENS: Semaphore = Semaphore::const_new(MAX_CANCELLABLE_OPENS);

#[cfg(feature = "detailed-tracing")]
use crate::utils::trace::OpOrigin;
use crate::{
    InfinoError,
    config::DEFAULT_CONNECTION_BUDGET_BYTES,
    error::{datafusion_error, datafusion_planning_error},
    memory::ConnectionMemoryBudget,
    runtime_bridge::{bridge_on_runtime, bridge_sync_to_async, carry_span, shared_io_runtime},
    runtime_metrics::{
        io::{UsageMeter, UsageSnapshot},
        op_stats,
    },
    storage::{
        AzureStorageProvider, GcsStorageProvider, LocalFsStorageProvider, S3StorageProvider,
        StorageError, StorageProvider,
        gcs::{GCS_BEARER_TOKEN_OPTION, SwappableGcpCredential},
    },
    superfile::{
        builder::FtsConfig,
        fts::{
            analysis::{Stemmer, Stopwords, check_recorded_tokenizer},
            bm25,
            tokenize::STANDARD_TOKENIZER,
        },
        vector::{builder::VectorConfig, distance::Metric},
    },
    supertable::{
        Supertable as SupertableHandle,
        manifest::disk_cache::ManifestDiskCache,
        options::SupertableOptions,
        query::{exec::common::collect_plan_metered, sql::sql_session_context},
        reader_cache::{DiskCacheConfig, DiskCacheError, DiskCacheStore},
        schema::{ColumnIndex, TableSchema, change::SchemaPatch, error::SchemaError},
    },
    utils::trace::{self, CloseOut, detail_span},
};

/// Subdirectory under a tables cache root holding the manifest-part cache.
const MANIFEST_CACHE_SUBDIR: &str = "manifest-parts";
/// budget for a tables content-addressed manifest-part cache.
const MANIFEST_CACHE_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;

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
    // A hosted (remote) target forwards over the wire — it never touches the
    // local storage path below.
    if matches!(backend, Backend::Remote { .. }) {
        return connect_remote(backend, options);
    }
    let usage_meter = UsageMeter::new();
    let gcs_credential = match &backend {
        Backend::Gcs { .. } => options
            .storage_options
            .get(GCS_BEARER_TOKEN_OPTION)
            .map(|bearer| SwappableGcpCredential::new(bearer.clone())),
        _ => None,
    };
    let store = match &backend {
        Backend::Memory => CatalogStore::Memory(Mutex::new(HashMap::new())),
        _ => {
            let root = backend_to_provider(
                &backend,
                &options,
                Arc::clone(&usage_meter),
                gcs_credential.as_ref(),
            )?
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
            usage_meter,
            gcs_credential,
            #[cfg(feature = "graph-index")]
            graph: Mutex::new(None),
        }),
    })
}

/// Build a hosted (remote) connection from a `Backend::Remote` target. The API
/// key comes from `ConnectOptions` or the `INFINO_API_KEY` env var.
#[cfg(feature = "remote")]
fn connect_remote(backend: Backend, options: ConnectOptions) -> Result<Connection, InfinoError> {
    let (base_url, database) = match &backend {
        Backend::Remote { base_url, database } => (base_url.clone(), database.clone()),
        _ => {
            return Err(InfinoError::Backend(
                "connect_remote requires a remote target".to_string(),
            ));
        }
    };
    let remote =
        remote::RemoteCatalog::new(base_url, database, options.api_key().map(str::to_owned))?;
    let connection_memory_budget = ConnectionMemoryBudget::from_budget_bytes(
        options
            .connection_memory_budget_bytes
            .unwrap_or(DEFAULT_CONNECTION_BUDGET_BYTES),
    );
    debug!(backend = ?backend, "catalog connected (remote)");
    Ok(Connection {
        inner: Arc::new(ConnectionInner {
            backend,
            options,
            store: CatalogStore::Remote(Arc::new(remote)),
            connection_memory_budget,
            usage_meter: UsageMeter::new(),
            gcs_credential: None,
            #[cfg(feature = "graph-index")]
            graph: Mutex::new(None),
        }),
    })
}

/// Without the `remote` feature a hosted target cannot be served: report it
/// clearly rather than silently falling through to the local path.
#[cfg(not(feature = "remote"))]
fn connect_remote(_backend: Backend, _options: ConnectOptions) -> Result<Connection, InfinoError> {
    Err(InfinoError::Config(
        "this build has no remote support; rebuild with the `remote` feature".to_string(),
    ))
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
    /// Sole object-store usage ledger for this connection (shared into every
    /// table provider). Benches and billing snapshot this meter.
    usage_meter: Arc<UsageMeter>,
    /// Swappable GCS credential shared by every provider on this connection.
    gcs_credential: Option<Arc<SwappableGcpCredential>>,
    /// A graph attached from another catalog (see [`Connection::attach_graph`]):
    /// the connection it lives in and its edge table, served to this
    /// connection's SQL as `graph_walk` / `graph_rank` with no table argument.
    /// Held weakly: the caller keeps the graph's connection open, and a
    /// strong handle here would let a connection attached to itself, or two
    /// attached to each other, keep both alive for good.
    #[cfg(feature = "graph-index")]
    graph: Mutex<Option<(Weak<ConnectionInner>, String)>>,
}

/// Where the `name → table` map lives. Durable backends persist it on the
/// root storage under optimistic concurrency; `memory://` keeps it (and
/// the tables themselves) in-process.
enum CatalogStore {
    Memory(Mutex<HashMap<String, SupertableHandle>>),
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
        handles: DashMap<String, SupertableHandle>,
        /// Per-name build/evict lock. Never removed once created: a concurrent
        /// opener may hold or await the `Arc`, so evicting it mid-use would let
        /// two builds proceed.
        ///
        /// One empty `Arc<Mutex<()>>` therefore lingers per distinct name ever
        /// seen. We can bound it with refcount-gated eviction (drop only when no
        /// one holds the `Arc`) later.
        building: DashMap<String, Arc<Mutex<()>>>,
    },
    /// Hosted (remote) connection: catalog operations forward over the wire.
    /// There is no local storage provider or handle cache — the endpoint owns
    /// the tables.
    #[cfg(feature = "remote")]
    Remote(Arc<remote::RemoteCatalog>),
}

impl Connection {
    /// Serve `table` of `graph` — an edge table over this connection's
    /// rows that `graph` has indexed with `OptimizeOptions::with_adjacency`
    /// — to this connection's [`query_sql`](Self::query_sql) as
    /// `graph_walk(table, seed_table, seed_ids, hops, k)` and
    /// `graph_rank(...)`, with no edge-table argument. The edge table
    /// stays `graph`'s: it is not listed, opened or scanned here, and a
    /// statement reaches it only through the two functions, which return
    /// rows of this connection's tables. So a graph can live in a catalog
    /// of its own and be walked from another. Attaching again replaces the
    /// earlier graph.
    ///
    /// The graph's connection is held weakly: keep `graph` (or a clone)
    /// open while it is attached. Once every handle to it is dropped, a
    /// statement that calls the functions fails and says so. Behind the
    /// `graph-index` feature, off the curated public surface.
    #[cfg(feature = "graph-index")]
    pub fn attach_graph(&self, graph: &Connection, table: &str) {
        *self
            .inner
            .graph
            .lock()
            .unwrap_or_else(PoisonError::into_inner) =
            Some((Arc::downgrade(&graph.inner), table.to_string()));
    }

    /// Cumulative object-store usage for this connection (read-only snapshot).
    /// Shared by every table provider created through this connection; take
    /// two snapshots and call [`UsageSnapshot::since`] for a window delta.
    /// Visible under `test-helpers`, `metering`, or `cfg(test)` — not part
    /// of the curated default public API.
    #[cfg(any(test, feature = "test-helpers", feature = "metering"))]
    pub fn usage_snapshot(&self) -> UsageSnapshot {
        self.inner.usage_meter.snapshot()
    }

    /// Object-store usage ledger for this connection. Shared by every table
    /// provider; benches and diagnostics hold the `Arc` across phases.
    /// Visible under `test-helpers`, `metering`, or `cfg(test)` — not part
    /// of the curated default public API.
    #[cfg(any(test, feature = "test-helpers", feature = "metering"))]
    pub fn usage_meter(&self) -> Arc<UsageMeter> {
        Arc::clone(&self.inner.usage_meter)
    }

    /// Provision the database this connection targets.
    ///
    /// A connection is bound to a single database — the path segment of a
    /// hosted URL (`https://host/<database>`), or the catalog root for a local
    /// backend. This registers that database so tables can be created in it,
    /// without a separate provisioning step outside the code.
    ///
    /// For a hosted connection it registers the database on the service and
    /// fails with [`InfinoError::AlreadyExists`] if it is already registered.
    /// For a local backend (`file://`, `s3://`, `memory://`, …) the catalog
    /// root *is* the database and comes into being with the first table, so
    /// this is a no-op success — kept on the surface so the same setup code
    /// runs against either target.
    ///
    /// ```
    /// # let db = infino::connect("memory://")?;
    /// db.create_database()?; // no-op for a local backend
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn create_database(&self) -> Result<(), InfinoError> {
        match &self.inner.store {
            // A local catalog root is created lazily by the first table; there
            // is no separate database to register.
            CatalogStore::Memory(_) | CatalogStore::Storage { .. } => Ok(()),
            #[cfg(feature = "remote")]
            CatalogStore::Remote(c) => c.create_database(),
        }
    }

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
        validate_schema(&schema).map_err(|e| e.with_context("create_table", Some(name)))?;
        indexes
            .check_analyzers()
            .map_err(|e| e.with_context("create_table", Some(name)))?;
        indexes
            .check_companions()
            .map_err(|e| e.with_context("create_table", Some(name)))?;
        let (fts_cfg, vec_cfg) = indexes.to_configs();
        // A companion is a table on storage beside this one; a catalog with
        // no storage, or one forwarding to a hosted service, has nowhere to
        // put it.
        if indexes.has_companions() && !matches!(self.inner.store, CatalogStore::Storage { .. }) {
            return Err(InfinoError::Unsupported(
                "companion tables need a storage-backed connection".to_string(),
            )
            .with_context("create_table", Some(name)));
        }

        match &self.inner.store {
            CatalogStore::Memory(map) => {
                let opts = build_options(
                    schema,
                    fts_cfg,
                    vec_cfg,
                    None,
                    Arc::clone(&self.inner.connection_memory_budget),
                )
                .map_err(|e| e.with_context("create_table", Some(name)))?;
                let handle = SupertableHandle::create(opts)
                    .map_err(|e| InfinoError::from(e).with_context("create_table", Some(name)))?;
                let mut map = map.lock().expect("catalog mutex poisoned");
                if map.contains_key(name) {
                    return Err(InfinoError::AlreadyExists(name.to_string())
                        .with_context("create_table", Some(name)));
                }
                map.insert(name.to_string(), handle.clone());
                info!(table = name, backend = "memory", "created table");
                Ok(Supertable::from_local(handle))
            }
            CatalogStore::Storage {
                root,
                handles,
                building,
            } => {
                let (existing, _) =
                    bridge_on_runtime(read_catalog(root.as_ref()), &shared_io_runtime())
                        .map_err(|e| e.with_context("create_table", Some(name)))?;
                if existing.tables.contains_key(name) {
                    return Err(InfinoError::AlreadyExists(name.to_string())
                        .with_context("create_table", Some(name)));
                }

                // Record what was actually used to build the table, so
                // `open_table` reconstructs matching options (the
                // supertable's options-hash check then validates them).
                let vectors: Vec<VectorEntry> = vec_cfg
                    .iter()
                    .map(|vc| VectorEntry {
                        column: vc.column.clone(),
                        dim: vc.dim,
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
                    fts: indexes.fts_columns(),
                    fts_analyzers: vec![
                        STANDARD_TOKENIZER.to_string();
                        indexes.fts_columns().len()
                    ],
                    fts_stopwords: indexes
                        .fts_stopwords()
                        .iter()
                        .map(|s| s.as_str().unwrap_or_default().to_string())
                        .collect(),
                    fts_stemmers: indexes
                        .fts_stemmers()
                        .iter()
                        .map(|s| s.as_str().unwrap_or_default().to_string())
                        .collect(),
                    fts_positions: indexes.fts_positions(),
                    fts_stored: indexes.fts_stored(),
                    fts_k1: indexes.fts_bm25().iter().map(|p| p.k1).collect(),
                    fts_b: indexes.fts_bm25().iter().map(|p| p.b).collect(),
                    vectors,
                    created_at_unix: now_unix(),
                };

                let table_storage = backend_to_provider(
                    &self.inner.backend.join(&location),
                    &self.inner.options,
                    Arc::clone(&self.inner.usage_meter),
                    self.inner.gcs_credential.as_ref(),
                )
                .map_err(|e| e.with_context("create_table", Some(name)))?
                .expect("non-memory backend yields a storage provider");
                // Disk cache is keyed on the stable name (not the unique
                // location) so the producer and a later reopener share one
                // cache directory; superfile keys carry the location, so a
                // re-created table never reads a dropped generation's bytes,
                // and the directory is claimed for this location, so a
                // dropped generation's bytes do not stay on disk either.
                let disk_cache =
                    build_disk_cache(&self.inner.options, &table_storage, name, &location)
                        .map_err(|e| e.with_context("create_table", Some(name)))?;
                let mut opts = build_options(
                    schema,
                    fts_cfg,
                    vec_cfg,
                    Some(table_storage),
                    Arc::clone(&self.inner.connection_memory_budget),
                )
                .map_err(|e| e.with_context("create_table", Some(name)))?;
                if let Some((cache, manifest_cache)) = disk_cache {
                    opts = opts
                        .with_disk_cache(cache)
                        .with_manifest_disk_cache(manifest_cache);
                }

                // Honor the connection's read-consistency policy (default
                // BoundedStaleness); `open_table` applies the same.
                opts = opts
                    .with_read_consistency(self.inner.options.read_consistency)
                    .with_companions(indexes.companion_specs());

                // Create the physical table at its unique location, then
                // register the name. A losing racer that also created a
                // (distinct) location just orphans its empty subtree; the
                // catalog OCC below decides the single name winner.
                let handle = SupertableHandle::create(opts)
                    .map_err(|e| InfinoError::from(e).with_context("create_table", Some(name)))?;

                // Gate the commit + memo insert: else a racing `open_table`
                // sees the commit, misses the memo, and builds a rival store.
                let gate = single_flight_gate(building, name);
                let _built = lock_gate(&gate);

                let name_owned = name.to_string();
                bridge_on_runtime(
                    commit_catalog(root.as_ref(), move |body| {
                        if body.tables.contains_key(&name_owned) {
                            return Err(InfinoError::AlreadyExists(name_owned.clone()));
                        }
                        body.tables.insert(name_owned.clone(), entry.clone());
                        Ok(())
                    }),
                    &shared_io_runtime(),
                )
                .map_err(|e| e.with_context("create_table", Some(name)))?;

                // Seed the memo: `query_sql` reads back through this same
                // handle, so in-process writes are visible at once.
                handles.insert(name.to_string(), handle.clone());

                info!(table = name, location = %location, "created table");
                Ok(Supertable::from_local(handle))
            }
            #[cfg(feature = "remote")]
            CatalogStore::Remote(c) => c.create_table(name, schema, indexes),
        }
    }

    /// Open the concrete engine handle for `name`, building and memoizing it on
    /// first use. Internal: callers needing engine-only methods (`register_into`
    /// for SQL, `reader` for the search TVFs) go through this; the public
    /// [`open_table`](Self::open_table) wraps the result.
    pub(crate) fn open_table_handle(&self, name: &str) -> Result<SupertableHandle, InfinoError> {
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
                // Warm path: lock-free sharded lookup, no serialization. A
                // handle purged elsewhere is dropped here, so the cold path
                // re-resolves it against the catalog.
                if let Some(handle) = live_handle(handles, name) {
                    return Ok(handle);
                }

                // Cold path: build once under the gate. Blocks here if a
                // same-name peer is mid-build (same `Arc`, same mutex); the
                // winner builds, the rest wake to find a warm `handles`.
                let gate = single_flight_gate(building, name);
                let _built = lock_gate(&gate);

                // A peer may have built it while we waited on the gate.
                if let Some(handle) = live_handle(handles, name) {
                    return Ok(handle);
                }

                let (body, _etag) =
                    bridge_on_runtime(read_catalog(root.as_ref()), &shared_io_runtime())
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
                for (i, column) in entry.fts.iter().enumerate() {
                    check_recorded_tokenizer(
                        column,
                        entry.fts_analyzers.get(i).map(String::as_str),
                    )
                    .map_err(|e| {
                        InfinoError::Config(e.to_string()).with_context("open_table", Some(name))
                    })?;
                    // A missing or short list reads as the default.
                    let stored = entry.fts_stored.get(i).copied().unwrap_or(true);
                    let positions = entry.fts_positions.get(i).copied().unwrap_or(false);
                    // A missing or empty filter name means no filter; a
                    // name that does not resolve is different — the
                    // recorded analysis cannot be reproduced, so the
                    // table is unusable rather than usable-with-a-guess.
                    let stopwords = match entry.fts_stopwords.get(i).map(String::as_str) {
                        None | Some("") => Stopwords::None,
                        Some(set) => Stopwords::from_name(set).ok_or_else(|| {
                            InfinoError::Backend(format!(
                                "table '{name}' column {column:?} records unknown stopwords \
                                 {set:?}"
                            ))
                            .with_context("open_table", Some(name))
                        })?,
                    };
                    let stemmer = match entry.fts_stemmers.get(i).map(String::as_str) {
                        None | Some("") => Stemmer::None,
                        Some(stem) => Stemmer::from_name(stem).ok_or_else(|| {
                            InfinoError::Backend(format!(
                                "table '{name}' column {column:?} records unknown stemmer \
                                 {stem:?}"
                            ))
                            .with_context("open_table", Some(name))
                        })?,
                    };
                    // The BM25 fallback is the frozen standard pair, not
                    // the crate default.
                    let k1 = entry.fts_k1.get(i).copied().unwrap_or(bm25::K1);
                    let b = entry.fts_b.get(i).copied().unwrap_or(bm25::B);
                    spec = spec.fts(
                        FtsField::new(column.clone())
                            .stopwords(stopwords)
                            .stemmer(stemmer)
                            .positions(positions)
                            .stored(stored)
                            .bm25(k1, b),
                    );
                }
                for v in &entry.vectors {
                    spec = spec.vector(
                        v.column.clone(),
                        v.dim,
                        metric_from_str(&v.metric)
                            .map_err(|e| e.with_context("open_table", Some(name)))?,
                    );
                }
                let (fts_cfg, vec_cfg) = spec.to_configs();

                let table_storage = backend_to_provider(
                    &self.inner.backend.join(&entry.location),
                    &self.inner.options,
                    Arc::clone(&self.inner.usage_meter),
                    self.inner.gcs_credential.as_ref(),
                )
                .map_err(|e| e.with_context("open_table", Some(name)))?
                .expect("non-memory backend yields a storage provider");

                // Cache directory is keyed on the stable name, matching
                // `create_table` (the on-storage subtree is `entry.location`).
                let disk_cache =
                    build_disk_cache(&self.inner.options, &table_storage, name, &entry.location)
                        .map_err(|e| e.with_context("open_table", Some(name)))?;
                let mut opts = build_options(
                    schema,
                    fts_cfg,
                    vec_cfg,
                    Some(table_storage),
                    Arc::clone(&self.inner.connection_memory_budget),
                )
                .map_err(|e| e.with_context("open_table", Some(name)))?;
                if let Some((cache, manifest_cache)) = disk_cache {
                    opts = opts
                        .with_disk_cache(cache)
                        .with_manifest_disk_cache(manifest_cache);
                }
                // Honor the connection's read-consistency policy. Default is
                // BoundedStaleness(1s): the per-query pointer re-check is
                // amortized across the window rather than paid on every query.
                opts = opts.with_read_consistency(self.inner.options.read_consistency);
                let handle = SupertableHandle::open(opts)
                    .map_err(|e| InfinoError::from(e).with_context("open_table", Some(name)))?;
                handles.insert(name.to_string(), handle.clone());

                Ok(handle)
            }
            // The local handle backs the local SQL / search-TVF paths, which a
            // remote connection never takes (it forwards `query_sql`). Reaching
            // here on a remote connection is an internal invariant violation.
            #[cfg(feature = "remote")]
            CatalogStore::Remote(_) => Err(InfinoError::Backend(
                "internal: no local table handle for a remote connection".to_string(),
            )
            .with_context("open_table", Some(name))),
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
        #[cfg(feature = "remote")]
        if let CatalogStore::Remote(c) = &self.inner.store {
            return c.open_table(name);
        }
        Ok(Supertable::from_local(self.open_table_handle(name)?))
    }

    /// The schema document of table `name`: its live columns with their
    /// ids, types, nullability and indexes, the field cap, and the
    /// `schema_id` a write can compare against. What this returns is also
    /// what [`Connection::apply_schema`] accepts, so a document read back
    /// applies as a no-op.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use infino::arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, IndexSpec};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # db.create_table("posts", schema, IndexSpec::new().fts("body"))?;
    /// let doc = db.schema("posts")?;
    /// assert_eq!(doc.fields()[0].name, "body");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn schema(&self, name: &str) -> Result<TableSchema, InfinoError> {
        // A hosted connection reads the document the endpoint serves; the
        // local handle below is the engine's own.
        #[cfg(feature = "remote")]
        if let CatalogStore::Remote(c) = &self.inner.store {
            return c
                .schema_document(name)
                .map_err(|e| e.with_context("schema", Some(name)));
        }
        let handle = self.open_table_handle(name)?;
        Ok((*handle.table_schema()).clone())
    }

    /// Change the schema of table `name` by merging `patch` into it, or
    /// create the table from `patch` when there is none. The result is the
    /// document the table holds afterwards.
    ///
    /// Each field in the patch is matched to a live column by `id` when it
    /// carries one and by `name` otherwise. An unmatched field adds a
    /// column; a matched field with another name (possible only by id)
    /// renames it; another type changes it; `dropped` retires it; a field
    /// not mentioned is untouched. Nothing is dropped by omission, and a
    /// patch that changes nothing commits nothing. `expected` is an
    /// optional compare-and-set against the current `schema_id`; it fails
    /// with [`InfinoError::Conflict`] when the schema has moved.
    ///
    /// ```
    /// # use infino::arrow_schema::DataType;
    /// # use infino::{connect, FieldPatch, SchemaPatch};
    /// # let db = connect("memory://")?;
    /// let patch = SchemaPatch::new(vec![
    ///     FieldPatch::named("body")
    ///         .with_type(DataType::LargeUtf8)
    ///         .with_nullable(false),
    /// ]);
    /// let doc = db.apply_schema("posts", &patch, None)?;
    /// assert_eq!(doc.schema_id(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn apply_schema(
        &self,
        name: &str,
        patch: &SchemaPatch,
        expected: Option<u32>,
    ) -> Result<TableSchema, InfinoError> {
        // The endpoint creates the table from the patch when there is none,
        // as the local path below does, so a hosted write forwards whole.
        #[cfg(feature = "remote")]
        if let CatalogStore::Remote(c) = &self.inner.store {
            return c
                .apply_schema(name, patch, expected)
                .map_err(|e| e.with_context("apply_schema", Some(name)));
        }
        let handle = match self.open_table_handle(name) {
            Ok(handle) => handle,
            Err(InfinoError::NotFound(_)) => {
                if let Some(expected) = expected {
                    return Err(InfinoError::Conflict(
                        SchemaError::SchemaConflict {
                            expected,
                            current: 0,
                        }
                        .to_string(),
                    )
                    .with_context("apply_schema", Some(name)));
                }
                let (schema, indexes) = seed_from_patch(patch)
                    .map_err(|e| e.with_context("apply_schema", Some(name)))?;
                self.create_table(name, schema, indexes)?;
                // The seed carries the columns; the cap and anything else the
                // patch sets lands by the same merge an existing table takes.
                self.open_table_handle(name)?
            }
            Err(e) => return Err(e),
        };
        Ok((*handle.apply_schema(patch, expected)?).clone())
    }

    /// Rotate the GCS bearer token in place, returning `true` when it was
    /// swapped. Only the bearer (`google_bearer_token`) is honored; any other
    /// key in `storage_options` is ignored on this path. Returns `false` when
    /// the connection has no in-place-rotatable credential (a non-GCS backend)
    /// or no bearer key was supplied — in both cases the caller should reopen,
    /// which applies the full `storage_options`.
    pub fn update_storage_credentials(&self, storage_options: &[(String, String)]) -> bool {
        let Some(credential) = &self.inner.gcs_credential else {
            return false;
        };
        let Some((_, bearer)) = storage_options
            .iter()
            .find(|(key, _)| key == GCS_BEARER_TOKEN_OPTION)
        else {
            return false;
        };
        credential.set_bearer(bearer.clone());
        true
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
    /// The hidden tables kept under the table's location — its vector
    /// index, and any companions declared with [`IndexSpec::companion`] —
    /// are part of those bytes: a drop without `purge` leaves them there,
    /// orphaned with the rest of the table, and `purge` removes them.
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
                let _dropping = lock_gate(&gate);

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
                bridge_on_runtime(
                    commit_catalog(root.as_ref(), |body| {
                        if let Some(entry) = body.tables.remove(name) {
                            location = Some(entry.location);
                        }
                        Ok(())
                    }),
                    &shared_io_runtime(),
                )
                .map_err(|e| e.with_context("drop_table", Some(name)))?;
                if let (true, Some(location)) = (purge, location) {
                    // Delete everything under the table's unique
                    // location. Listing is component-aware, so a sibling
                    // location sharing a string prefix never matches;
                    // deletes are idempotent, so re-running after a
                    // partial failure converges.
                    bridge_on_runtime(
                        async {
                            let objects = root.list_with_prefix(&location).await?;
                            try_join_all(objects.iter().map(|uri| root.delete(uri))).await?;
                            Ok::<(), StorageError>(())
                        },
                        &shared_io_runtime(),
                    )
                    .map_err(|e| InfinoError::from(e).with_context("drop_table", Some(name)))?;
                }
                // The table's local disk cache goes with it: its bytes belong
                // to a generation nothing will read again. Under the gate, so
                // no open of the same name is building a cache over it. The
                // catalog commit above is the drop; a cache that cannot be
                // removed here is cleared by the next open of the name
                // (`claim_table_cache`), so it does not fail the drop.
                if let Some(cache_root) = self.inner.options.cache_dir.as_ref() {
                    remove_table_cache(&cache_root.join(name));
                }
                Ok(())
            }
            #[cfg(feature = "remote")]
            CatalogStore::Remote(c) => c.drop_table(name, purge),
        }
    }

    /// On-storage byte footprint for `name` (user + hidden vector index).
    ///
    /// Loads lazy manifest parts before summing so cold tables are not
    /// under-counted. Visible under `metering` for platform billing / Grafana.
    #[cfg(any(test, feature = "test-helpers", feature = "metering"))]
    pub fn table_storage_bytes(&self, name: &str) -> Result<u64, InfinoError> {
        // `storage_bytes` is a local measurement of the on-storage superfile
        // footprint; a hosted connection has no local storage to measure, so
        // reject it there rather than reaching for a handle that doesn't exist.
        #[cfg(feature = "remote")]
        if matches!(self.inner.store, CatalogStore::Remote(_)) {
            return Err(InfinoError::Backend(
                "table_storage_bytes is a local measurement, not available over the remote transport"
                    .to_string(),
            )
            .with_context("table_storage_bytes", Some(name)));
        }
        // The concrete engine handle carries `storage_bytes`; the public
        // wrapper returned by `open_table` does not.
        self.open_table_handle(name)?
            .storage_bytes()
            .map_err(|e| InfinoError::from(e).with_context("table_storage_bytes", Some(name)))
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
            #[cfg(feature = "remote")]
            CatalogStore::Remote(c) => c.list_tables(),
        }
    }

    /// Run SQL across the tables in this catalog. Every relation the query
    /// names is resolved through the catalog and registered into one
    /// DataFusion session, so cross-table joins and aggregations work.
    /// Returns the collected result batches.
    ///
    /// Read-only, and one statement per call: a query, `EXPLAIN` or
    /// `DESCRIBE`. A write of any kind is refused with [`InfinoError::Query`];
    /// writes go through a table's `append`, `update` and `delete`.
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
    pub fn query_sql(&self, sql: &str) -> Result<Vec<RecordBatch>, InfinoError> {
        self.query_sql_cancellable(sql, &QueryCancel::new())
    }

    /// [`query_sql`](Self::query_sql) that stops when `cancel` fires.
    ///
    /// Fire `cancel` from another thread and the call returns
    /// [`InfinoError::Cancelled`] soon after, whether the query is resolving
    /// its tables, planning (which opens the files it reads) or executing. The
    /// planning and execution it drops stop at their next await point and
    /// give back the cache and memory they reserved. A handle that has
    /// already fired returns at once, without reading anything.
    ///
    /// What a cancel does not cut short:
    /// - Work already handed off, which finishes after the call returns: a
    ///   table being opened or checked for a newer manifest (the table is
    ///   ready for the next query), an object-store read a search function is
    ///   waiting on, CPU work on the reader pool, a manifest part being loaded
    ///   (it lands in the cache for the next query), and the background
    ///   download of a file the query opened, as after any query. Until a
    ///   table open handed off this way is done, its reads count as the
    ///   connection's foreground reads and the call's trace span stays open.
    ///   Under bounded staleness, the open has already taken the table's
    ///   freshness check, so a query right after the cancel is served the
    ///   snapshot from before it until the open lands, as a concurrent query
    ///   would be.
    /// - A table open while many others are in flight across the process: it
    ///   runs on the calling thread, and the call waits for it.
    /// - Execution on the calling thread: the result is gathered there, so a
    ///   plan over cached data that does its work without yielding (one
    ///   partition, no read to wait on) sees the cancel only once it yields.
    /// - A hosted connection's request: the handle is checked only before it
    ///   is sent.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use infino::arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, IndexSpec, InfinoError, QueryCancel};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # db.create_table("posts", schema, IndexSpec::new().fts("body"))?;
    /// let cancel = QueryCancel::new();
    /// // Typically fired from another thread, for example when a deadline passes.
    /// cancel.cancel();
    /// let result = db.query_sql_cancellable("SELECT _id FROM posts", &cancel);
    /// assert!(matches!(result, Err(InfinoError::Cancelled(_))));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg_attr(
        feature = "detailed-tracing",
        // Connection-level entry: no table handle yet, so no `role` — the
        // per-table spans beneath this one carry it. The empty fields are
        // filled in once the query has run: what it returned, the per-op
        // counters the meter collected for it, and the object-store
        // requests and bytes the connection issued meanwhile, so the span,
        // the billed read work and the GETs can be read against each other.
        // Named for `query_sql`, which every caller's trace already searches
        // by; the plain form adds no span of its own.
        tracing::instrument(name = "query_sql", skip_all, fields(
            sql = sql,
            origin = OpOrigin::Query.as_str(),
            rows_out = tracing::field::Empty,
            sql_page_bytes = tracing::field::Empty,
            planned_read_ranges = tracing::field::Empty,
            rows_materialized = tracing::field::Empty,
            kernel_cpu_ns = tracing::field::Empty,
            fts_postings_bytes = tracing::field::Empty,
            vector_cells_scanned = tracing::field::Empty,
            vector_candidates_scanned = tracing::field::Empty,
            vector_rows_reranked = tracing::field::Empty,
            store_heads = tracing::field::Empty,
            store_gets = tracing::field::Empty,
            store_get_bytes = tracing::field::Empty,
            store_bg_gets = tracing::field::Empty,
            store_bg_get_bytes = tracing::field::Empty,
        ))
    )]
    pub fn query_sql_cancellable(
        &self,
        sql: &str,
        cancel: &QueryCancel,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(sql, "running sql query");

        cancel
            .check()
            .map_err(|e| e.with_context("query_sql", None))?;

        // A hosted connection forwards the SQL to the endpoint; the local
        // DataFusion path below runs only for local backends. The request
        // blocks until the endpoint answers, so the cancel above is the only
        // one it sees.
        #[cfg(feature = "remote")]
        if let CatalogStore::Remote(c) = &self.inner.store {
            return c.query_sql(sql);
        }

        // Past the cap the planner aborts instead of erroring; refuse before any planning work.
        ensure_sql_within_connective_cap(sql)?;

        // Gate SQL heap on the connection budget: DataFusion allocates the
        // working set (sort / aggregate / join), so its pool is the gate. The
        // same constructor as a table reader's, so covered aggregates are
        // answered from manifest statistics here too. A session built from
        // our own budget config failing is ours, not the query's.
        let ctx = sql_session_context(&self.inner.connection_memory_budget)
            .map_err(|e| InfinoError::Backend(e.to_string()).with_context("query_sql", None))?;

        // One snapshot of the session for parsing and table lookup; planning
        // takes a fresh one later, after the tables are registered.
        let state = ctx.state();
        let statement =
            read_only_statement(&state, sql).map_err(|e| e.with_context("query_sql", None))?;

        // Resolve the relations the query names and register each that is a
        // catalog table. Unknown names (CTEs, search TVFs, aliases) are
        // skipped — the planner resolves those by other means or errors.
        let refs = state
            .resolve_table_references(&statement)
            .map_err(|e| datafusion_planning_error(&e).with_context("query_sql", None))?;

        let mut seen = HashSet::new();
        let mut handles: Vec<SupertableHandle> = Vec::new();
        for r in &refs {
            let name = r.table().to_string();
            if !seen.insert(name.clone()) {
                continue;
            }
            if let Some(table) = self
                .open_and_register_cancellable(&ctx, &name, cancel)
                .map_err(|e| e.with_context("query_sql", None))?
            {
                handles.push(table);
            }
        }

        // Search TVFs resolve their leading table-name argument through
        // the catalog at call time (so a table named only inside a TVF —
        // not as a `FROM` relation — still resolves).
        search_tvf::register_search_tvfs(&ctx, self.clone());
        // The graph functions walk an edge table of this catalog by name, or
        // the one attached from another catalog with no name.
        #[cfg(feature = "graph-index")]
        {
            let attached = self
                .inner
                .graph
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone();
            match attached {
                Some((graph, table)) => {
                    let graph = graph
                        .upgrade()
                        .map(|inner| Connection { inner })
                        .ok_or_else(|| {
                            InfinoError::Query(
                                "the attached graph's connection was dropped; keep it open while \
                                 it is attached, or attach it again"
                                    .to_string(),
                            )
                        })?;
                    search_tvf::register_graph_tvfs(&ctx, graph, self.clone(), Some(table));
                }
                None => search_tvf::register_graph_tvfs(&ctx, self.clone(), self.clone(), None),
            }
        }
        trace::follow_spans_into_datafusion_tasks();

        // Caller-thread pickup, same as reader mint: the drive future may
        // poll on runtime threads where the scope's slot is invisible.
        let op_stats = op_stats::current();
        // What the span records when the query is done; see `CloseOut`.
        let close_out = CloseOut::begin(
            || op_stats.clone(),
            || Some(Arc::clone(&self.inner.usage_meter)),
        );
        let drive = async move {
            // Plan on this runtime's 16 MiB workers, not the calling thread `block_on` polls on:
            // planner recursion depth must not hang on a stack the engine does not own. A panic
            // surfaces through the join as a query error.
            let planner_ctx = ctx.clone();
            // The planning task runs on another thread, which inherits no
            // span on its own. `sql.plan` is made here, under this call's
            // span, and the task carries it along. The task tracer
            // (`trace::follow_spans_into_datafusion_tasks`) is installed only
            // with `detailed-tracing`, so the span is attached here too, or
            // planning's log lines lose this call's span in every other build.
            let planning = async move {
                // Plan the statement already parsed and checked, check the plan, execute.
                // `SessionContext::sql` would run a DDL or session statement while producing
                // the DataFrame, so the second read-only check sits between planning and
                // execution. It runs on the planned tree and catches what the statement does
                // not show: `SELECT ... INTO` reads like a query and plans to a CREATE TABLE.
                // Planning has no side effects; a refused statement has touched nothing.
                let plan = planner_ctx
                    .state()
                    .statement_to_plan(statement)
                    .await
                    .map_err(|e| datafusion_planning_error(&e))?;

                read_only_sql_options()
                    .verify_plan(&plan)
                    .map_err(|e| InfinoError::Query(format!("{READ_ONLY_REFUSAL} ({e})")))?;

                let df = planner_ctx
                    .execute_logical_plan(plan)
                    .await
                    .map_err(|e| datafusion_error(&e))?;

                // Execute through the physical plan (what `DataFrame::collect`
                // does internally) so the plan handle survives execution and
                // DataFusion's own operator metrics — elapsed compute, scan
                // output rows — can be folded into the per-query stats.
                let task_ctx = planner_ctx.task_ctx();
                let plan = df
                    .create_physical_plan()
                    .await
                    .map_err(|e| datafusion_error(&e))?;
                Ok::<_, InfinoError>((task_ctx, plan))
            }
            .instrument(detail_span!("sql.plan"))
            .in_current_span();
            // A `SpawnedTask` aborts the task when dropped, so a cancel stops
            // the file opens planning is making instead of leaving them to
            // run on, detached.
            let (task_ctx, plan) = SpawnedTask::spawn(planning).await.map_err(|join| {
                // A panic while planning is the engine's fault, never the query's.
                InfinoError::Backend(format!("planning task failed: {join}"))
            })??;
            // The shared meter-collect-harvest step: the root wrapper
            // meters the whole plan (aggregation, sort and join work sits
            // above the scan and is this query's CPU too), the scan
            // wrapper still counts on spawned partitions where the root's
            // thread never runs, and the shared bracket depth keeps a
            // single-partition plan from counting both.
            let batches = collect_plan_metered(&plan, task_ctx, &op_stats)
                .instrument(detail_span!("sql.execute"))
                .await
                .map_err(|e| datafusion_error(&e))?;
            // An empty Vec carries no schema, so hand back one empty batch
            // instead. Its schema comes from the physical plan, not the
            // DataFrame: the scan types scalar strings as `Utf8View`, and
            // `expand_views_at_output` undoes that during optimization,
            // which the DataFrame's logical plan predates.
            let batches = if batches.is_empty() {
                vec![RecordBatch::new_empty(plan.schema())]
            } else {
                batches
            };
            // Field ids are the engine's bookkeeping; a result carries the
            // caller's columns.
            Ok(batches
                .into_iter()
                .map(crate::supertable::schema::strip_field_ids)
                .collect::<Vec<_>>())
        };
        // The cancel races the whole query. Dropping `drive` stops planning
        // (its `SpawnedTask`) and execution (DataFusion aborts its partition
        // tasks on drop).
        let drive = cancel.unless_cancelled(drive);
        // A query that names a `FROM` catalog table drives on that table's
        // runtime; otherwise the connection's own. The fallback still has to
        // be multi-thread: a table-free query can be a search TVF, which
        // fans out object-store reads under the hood. Every error out of
        // `drive` gets its `query_sql` context here, once.
        let result = match handles.first() {
            Some(table) => table
                .block_on_query(drive)
                .map_err(|e: InfinoError| e.with_context("query_sql", None)),
            None => bridge_on_runtime(drive, &self.query_runtime())
                .map_err(|e: InfinoError| e.with_context("query_sql", None)),
        };

        // A failed or cancelled query records what it read too; for a cancelled one, the reads it
        // made before it stopped are the number worth seeing.
        match &result {
            Ok(batches) => close_out.finish_batches(batches),
            Err(_) => close_out.finish_failed(),
        }

        result
    }

    /// [`Self::open_and_register`] for one query, returning `Cancelled` as soon
    /// as `cancel` fires.
    ///
    /// An open that may read storage moves to the I/O runtime's blocking pool
    /// and races the cancel: a table not built yet loads its whole manifest,
    /// which nothing can interrupt midway, and a snapshot due its freshness
    /// check reads the pointer, and a newer manifest if one was published. A
    /// cancelled open still finishes, so the next query finds the table
    /// built. An open that reads nothing stays on the calling thread, since
    /// the hop would cost more than the open, and so does one past
    /// [`MAX_CANCELLABLE_OPENS`].
    fn open_and_register_cancellable(
        &self,
        ctx: &SessionContext,
        name: &str,
        cancel: &QueryCancel,
    ) -> Result<Option<SupertableHandle>, InfinoError> {
        cancel.check()?;
        // A memory catalog opens without reading storage, and a hosted one
        // never opens tables here.
        let CatalogStore::Storage { handles, .. } = &self.inner.store else {
            return self.open_and_register(ctx, name);
        };
        if let Some(table) = live_handle(handles, name).filter(|t| !t.pointer_refresh_pending()) {
            return register(table, ctx, name);
        }
        let Ok(slot) = CANCELLABLE_OPENS.try_acquire() else {
            return self.open_and_register(ctx, name);
        };
        let conn = self.clone();
        let session = ctx.clone();
        let table = name.to_owned();
        // The SQL provider takes the query's work counters from the thread
        // that registers it (`op_stats::current`), so they move with the open.
        let collector = op_stats::current();
        let runtime = shared_io_runtime();
        // Dropping the handle on a cancel leaves the open running, holding
        // its slot until it is done.
        let opening = runtime.spawn_blocking(carry_span(move || {
            let _slot = slot;
            op_stats::attached(collector, || conn.open_and_register(&session, &table))
        }));
        let opened = async {
            match opening.await {
                Ok(registered) => registered,
                // A panic reaches the caller as it does from an inline open.
                Err(join) => panic::resume_unwind(join.into_panic()),
            }
        };
        bridge_on_runtime(cancel.unless_cancelled(opened), &runtime)
    }

    /// Open the catalog table `name` and register it into `ctx`, or `None`
    /// for a table that is not there.
    fn open_and_register(
        &self,
        ctx: &SessionContext,
        name: &str,
    ) -> Result<Option<SupertableHandle>, InfinoError> {
        match self.open_table_handle(name) {
            Ok(table) => register(table, ctx, name),
            Err(InfinoError::NotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Runtime for the table-free `query_sql` fallback.
    fn query_runtime(&self) -> Arc<Runtime> {
        shared_io_runtime()
    }
}

/// Parse `sql` into the one statement `query_sql` runs, refusing a write before it is planned.
///
/// ```text
///  sql ──► parse ──► one statement? ──no──► Query: "runs exactly one SQL statement"
///                          │
///                      reads only? ──no──► Query: read-only refusal
///                          │
///                   planned once from this statement, then the plan is checked again
/// ```
///
/// Checking the statement, not only the plan, is what refuses a write DataFusion cannot plan
/// (`ALTER TABLE`, an `INSERT` inside a CTE): planning fails on those before the plan check
/// runs, and they would otherwise read as SQL a later version might support.
fn read_only_statement(state: &SessionState, sql: &str) -> Result<DFStatement, InfinoError> {
    let recursion_limit = state.config().options().sql_parser.recursion_limit;

    let mut statements = DFParserBuilder::new(sql)
        .with_dialect(&GenericDialect {})
        .with_recursion_limit(recursion_limit)
        .build()
        .and_then(|mut parser| parser.parse_statements())
        .map_err(|e| datafusion_planning_error(&e))?;

    let (Some(mut statement), true) = (statements.pop_front(), statements.is_empty()) else {
        return Err(InfinoError::Query(
            "query_sql runs exactly one SQL statement".to_string(),
        ));
    };

    if !reads_only(&statement) {
        return Err(InfinoError::Query(READ_ONLY_REFUSAL.to_string()));
    }

    exact_wide_integers(&mut statement);
    Ok(statement)
}

/// Read every whole-number literal too wide for 64 bits as the exact decimal it spells.
///
/// DataFusion reads a number literal as an `i64`, else a `u64`, else a `Float64`. A row's `_id`
/// is a 32-digit `DECIMAL(38, 0)`, so `WHERE _id = 3304...` compared the column with a float:
/// the comparison is coerced to `Decimal128(38, 15)`, which overflows and fails the query, and a
/// float would not hold the 32 digits anyway. Each such literal becomes
/// `CAST('<digits>' AS DECIMAL(38, 0))`, which compares with the column exactly. A literal that
/// fits 64 bits, has a fraction or an exponent, or is past `MAX_EXACT_INTEGER_DIGITS` is left as
/// DataFusion reads it.
fn exact_wide_integers(statement: &mut DFStatement) {
    match statement {
        DFStatement::Statement(statement) => {
            let _ = visit_expressions_mut(statement.as_mut(), |expr| {
                if let Some(digits) = wide_integer_digits(expr) {
                    *expr = SqlExpr::Cast {
                        kind: CastKind::Cast,
                        expr: Box::new(SqlExpr::Value(SqlValue::SingleQuotedString(digits).into())),
                        data_type: SqlDataType::Decimal(ExactNumberInfo::PrecisionAndScale(
                            MAX_EXACT_INTEGER_DIGITS as u64,
                            0,
                        )),
                        array: false,
                        format: None,
                    };
                }
                ControlFlow::<()>::Continue(())
            });
        }
        DFStatement::Explain(explain) => exact_wide_integers(&mut explain.statement),
        _ => {}
    }
}

/// The digits of `expr` when it is a whole-number literal DataFusion would read as a float: past
/// `u64` and within `MAX_EXACT_INTEGER_DIGITS`. A sign is a separate unary operator in the AST, so
/// the literal itself is digits alone.
fn wide_integer_digits(expr: &SqlExpr) -> Option<String> {
    let SqlExpr::Value(value) = expr else {
        return None;
    };
    let SqlValue::Number(text, _) = &value.value else {
        return None;
    };
    let whole = !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit());
    (whole && text.len() <= MAX_EXACT_INTEGER_DIGITS && text.parse::<u64>().is_err())
        .then(|| text.clone())
}

/// Whether `statement`, and every statement nested in it, only reads: an `INSERT` can hide in a
/// CTE, in parentheses or under an `EXPLAIN`. An allowlist, so a statement kind nobody listed is
/// refused rather than run.
fn reads_only(statement: &DFStatement) -> bool {
    match statement {
        DFStatement::Statement(statement) => visit_statements(statement.as_ref(), |nested| {
            if is_read(nested) {
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(())
            }
        })
        .is_continue(),

        DFStatement::Explain(explain) => reads_only(&explain.statement),

        _ => false,
    }
}

/// The statements that change nothing: a query, and the ones that describe (`EXPLAIN`,
/// `DESCRIBE`, every `SHOW` form). Listing a read DataFusion cannot plan is deliberate: the
/// planner then reports it as unsupported, not as a write. A statement nested in one of these is
/// checked on its own.
fn is_read(statement: &SqlStatement) -> bool {
    matches!(
        statement,
        SqlStatement::Query(_)
            | SqlStatement::Explain { .. }
            | SqlStatement::ExplainTable { .. }
            | SqlStatement::ShowCatalogs { .. }
            | SqlStatement::ShowCharset(_)
            | SqlStatement::ShowCollation { .. }
            | SqlStatement::ShowColumns { .. }
            | SqlStatement::ShowCreate { .. }
            | SqlStatement::ShowDatabases { .. }
            | SqlStatement::ShowFunctions { .. }
            | SqlStatement::ShowObjects(_)
            | SqlStatement::ShowProcessList { .. }
            | SqlStatement::ShowSchemas { .. }
            | SqlStatement::ShowStatus { .. }
            | SqlStatement::ShowTables { .. }
            | SqlStatement::ShowVariable { .. }
            | SqlStatement::ShowVariables { .. }
            | SqlStatement::ShowViews { .. }
    )
}

/// `query_sql`'s second read-only check: refuse every plan node that acts on data, schema, or
/// session state (DDL, DML and `COPY`, session statements such as `SET`). **The check is on the
/// planned tree, so whatever the planner turns into a write is refused, however it was spelled.**
fn read_only_sql_options() -> SQLOptions {
    SQLOptions::new()
        .with_allow_ddl(false)
        .with_allow_dml(false)
        .with_allow_statements(false)
}

/// Occurrences of `or` / `and` as case-insensitive substrings: an upper bound on the true
/// connective count (`ORDER` or a literal only overcount), provable in one byte pass.
fn connective_upper_bound(sql: &str) -> usize {
    let bytes = sql.as_bytes();
    let mut count = 0usize;
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i].eq_ignore_ascii_case(&b'o') && bytes[i + 1].eq_ignore_ascii_case(&b'r') {
            count += 1;
            i += 2;
        } else if i + 2 < bytes.len()
            && bytes[i].eq_ignore_ascii_case(&b'a')
            && bytes[i + 1].eq_ignore_ascii_case(&b'n')
            && bytes[i + 2].eq_ignore_ascii_case(&b'd')
        {
            count += 1;
            i += 3;
        } else {
            i += 1;
        }
    }
    count
}

/// Register the opened `table` into `ctx` as `name`.
///
/// `None` when the table was purged since it was opened: the name is then
/// skipped like one that was never a table, and the planner reports it, as
/// the search table functions do. Any other failure minting its reader (a
/// storage fault, refused credentials) keeps its cause.
fn register(
    table: SupertableHandle,
    ctx: &SessionContext,
    name: &str,
) -> Result<Option<SupertableHandle>, InfinoError> {
    match table.register_into(ctx, name).map_err(InfinoError::from) {
        Ok(_) => Ok(Some(table)),
        Err(InfinoError::NotFound(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

fn connective_cap_error(surface: &str) -> InfinoError {
    InfinoError::Query(format!(
        "{surface} has more than {MAX_PREDICATE_CONNECTIVES} AND/OR connectives; rewrite long value lists as IN (...)"
    ))
}

/// Refuse SQL with more than [`MAX_PREDICATE_CONNECTIVES`] `AND` / `OR` tokens, before planning.
/// Cheapest stage that can decide wins:
///
/// - shorter than the cap could occupy: pass, no scan,
/// - substring bound under the cap: pass, one byte scan, no lexing,
/// - exact token count decides: a keyword inside a string literal does not count, `BETWEEN`'s
///   `AND` does (refuses sooner, never admits more).
///
/// Text that fails to tokenize passes; the parser reports it right after.
fn ensure_sql_within_connective_cap(sql: &str) -> Result<(), InfinoError> {
    if sql.len() < MIN_BYTES_PER_CONNECTIVE * MAX_PREDICATE_CONNECTIVES {
        return Ok(());
    }

    if connective_upper_bound(sql) <= MAX_PREDICATE_CONNECTIVES {
        return Ok(());
    }

    if connective_count(sql).is_some_and(|count| count > MAX_PREDICATE_CONNECTIVES) {
        return Err(connective_cap_error("query").with_context("query_sql", None));
    }

    Ok(())
}

/// Exact `AND` / `OR` keyword count over the token stream; `None` when the text does not tokenize.
fn connective_count(sql: &str) -> Option<usize> {
    let tokens = SqlTokenizer::new(&GenericDialect {}, sql).tokenize().ok()?;
    Some(
        tokens
            .iter()
            .filter(|token| {
                matches!(token, Token::Word(word) if matches!(word.keyword, Keyword::AND | Keyword::OR))
            })
            .count(),
    )
}

/// Mutation-predicate side of [`ensure_sql_within_connective_cap`]: count `AND` / `OR` nodes in
/// a built `Expr`. The stack-protected `TreeNode` walk is safe on the input it refuses; stops at
/// the first node past the cap.
pub(crate) fn ensure_expr_within_connective_cap(predicate: &Expr) -> Result<(), InfinoError> {
    let mut connectives = 0usize;
    predicate
        .apply(|expr| {
            if let Expr::BinaryExpr(BinaryExpr {
                op: Operator::And | Operator::Or,
                ..
            }) = expr
            {
                connectives += 1;
            }

            Ok(if connectives > MAX_PREDICATE_CONNECTIVES {
                TreeNodeRecursion::Stop
            } else {
                TreeNodeRecursion::Continue
            })
        })
        .expect("invariant: the counting visitor never errors");

    if connectives > MAX_PREDICATE_CONNECTIVES {
        return Err(connective_cap_error("predicate"));
    }

    Ok(())
}

/// Build `SupertableOptions` from a schema + lowered configs, attaching
/// `storage` when present (absent → in-memory table).
fn build_options(
    schema: SchemaRef,
    fts: Vec<FtsConfig>,
    vectors: Vec<VectorConfig>,
    storage: Option<Arc<dyn StorageProvider>>,
    connection_memory_budget: Arc<ConnectionMemoryBudget>,
) -> Result<SupertableOptions, InfinoError> {
    let mut opts = SupertableOptions::new(schema, fts, vectors)?;
    if let Some(s) = storage {
        opts = opts.with_storage(s);
    }
    // Set last so no builder step can reset the shared connection budget.
    opts.connection_memory_budget = connection_memory_budget;
    Ok(opts)
}

/// Construct the storage provider for `backend` (None for `memory://`).
/// Every durable provider records into the connection's `usage_meter`.
fn backend_to_provider(
    backend: &Backend,
    options: &ConnectOptions,
    usage_meter: Arc<UsageMeter>,
    gcs_credential: Option<&Arc<SwappableGcpCredential>>,
) -> Result<Option<Arc<dyn StorageProvider>>, InfinoError> {
    let provider: Option<Arc<dyn StorageProvider>> = match backend {
        Backend::Memory => None,
        Backend::LocalFs { root } => Some(Arc::new(LocalFsStorageProvider::new_with_meter(
            root.clone(),
            usage_meter,
        )?)),
        Backend::S3 { bucket, prefix } => Some(Arc::new(
            S3StorageProvider::new_with_prefix(bucket, prefix, &options.storage_options)?
                .with_usage_meter(usage_meter),
        )),
        Backend::Azure { container, prefix } => Some(Arc::new(
            AzureStorageProvider::new_with_prefix(container, prefix, &options.storage_options)?
                .with_usage_meter(usage_meter),
        )),
        Backend::Gcs { bucket, prefix } => Some(Arc::new(
            GcsStorageProvider::new_with_shared_credential(
                bucket,
                prefix,
                &options.storage_options,
                gcs_credential.cloned(),
            )?
            .with_usage_meter(usage_meter),
        )),
        // A remote (hosted) connection forwards operations over the wire and
        // never opens a local storage provider; `connect_with` routes it away
        // before reaching here.
        Backend::Remote { .. } => {
            return Err(InfinoError::Backend(
                "remote backend has no storage provider".to_string(),
            ));
        }
    };
    // Tests slow or count reads by wrapping what the connection builds.
    #[cfg(test)]
    let provider = match &options.storage_wrapper {
        Some(wrap) => provider.map(|provider| (wrap.0)(provider)),
        None => provider,
    };
    Ok(provider)
}

/// Build a per-table disk cache from the connection's options, or `None`
/// when no cache directory is configured. Rooted at `<cache_dir>/<name>`
/// so tables don't share cache files; the byte budget applies per table.
///
/// Budget semantics: an explicit `cache_budget_bytes` is respected
/// verbatim (the engine warns once if the table's footprint outgrows it).
/// With no explicit budget the cache is marked engine-managed, and the
/// supertable raises the budget to the table's real on-storage footprint
/// — user superfiles plus the hidden vector index — at open and after
/// maintenance, so vector tables don't silently churn a default-sized
/// cache once the drain doubles their working set.
fn build_disk_cache(
    options: &ConnectOptions,
    storage: &Arc<dyn StorageProvider>,
    name: &str,
    location: &str,
) -> Result<Option<(Arc<DiskCacheStore>, Arc<ManifestDiskCache>)>, InfinoError> {
    let Some(cache_root) = options.cache_dir.as_ref() else {
        return Ok(None);
    };
    let table_root = cache_root.join(name);
    claim_table_cache(&table_root, location)?;
    let mut cfg = DiskCacheConfig {
        cache_root: table_root.clone(),
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
    if options.cache_budget_bytes.is_none() {
        cache.mark_budget_auto_sized();
    }
    let manifest_cache = ManifestDiskCache::new(
        table_root.join(MANIFEST_CACHE_SUBDIR),
        MANIFEST_CACHE_BUDGET_BYTES,
    )
    .map_err(|e| InfinoError::Io(e.to_string()))?;
    Ok(Some((cache, manifest_cache)))
}

/// Make `table_root` the disk cache of the table generation stored at
/// `location`. A directory holding another generation's bytes - the table was
/// dropped and re-created under the same name, by this process or another -
/// or bytes of no recorded generation (a cache written before the record
/// existed) is emptied first: those bytes are keyed by locations nothing will
/// read again, and the cache would otherwise count them against its budget
/// and keep them on disk until eviction. Then `location` is recorded.
fn claim_table_cache(table_root: &Path, location: &str) -> Result<(), InfinoError> {
    let record = table_root.join(TABLE_CACHE_LOCATION_FILE);
    let io = |e: IoError| InfinoError::Io(format!("disk cache {}: {e}", table_root.display()));
    match fs::read_to_string(&record) {
        Ok(held) if held == location => return Ok(()),
        Ok(_) => remove_table_cache(table_root),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            let holds_files = fs::read_dir(table_root)
                .map(|mut entries| entries.next().is_some())
                .unwrap_or(false);
            if holds_files {
                remove_table_cache(table_root);
            }
        }
        Err(e) => return Err(io(e)),
    }
    fs::create_dir_all(table_root).map_err(io)?;
    fs::write(&record, location).map_err(io)
}

/// Remove a table's disk-cache directory, if there is one. A failure is
/// logged, not returned: the bytes are a cache, and the next claim of the
/// directory ([`claim_table_cache`]) clears what this left.
fn remove_table_cache(table_root: &Path) {
    match fs::remove_dir_all(table_root) {
        Ok(()) => info!(dir = %table_root.display(), "removed a table's disk cache"),
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => {
            warn!(dir = %table_root.display(), error = %e, "could not remove a table's disk cache")
        }
    }
}

/// The cached handle for `name`, or `None` (after evicting it) if its table was
/// dropped and purged elsewhere — `handles` is per-process, so such a drop never
/// reaches it. The `Ref` is dropped before `remove`, which would else deadlock.
fn live_handle(
    handles: &DashMap<String, SupertableHandle>,
    name: &str,
) -> Option<SupertableHandle> {
    let entry = handles.get(name)?;
    if !entry.pointer_vanished() {
        return Some(entry.clone());
    }
    drop(entry);
    handles.remove(name);
    None
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

/// Grab the build gate. If it's poisoned, ignore that and use it anyway.
///
/// Why ignoring poison is safe: a lock gets poisoned when a thread crashes
/// while holding it, warning "the data might be half-written." But the only
/// shared write under this gate is `handles.insert`, on the last line, after
/// the build has fully succeeded. A crash happens before that, so nothing is
/// half-written and there's nothing to protect.
///
/// What this fixes: the old code crashed on poison instead. The gate is kept
/// forever (one per table name), so once poisoned, every later open of that
/// table crashed on it, restarted, and crashed again: a permanent crash loop.
///
/// Keep the write last: if you add a shared-state write in the middle of the
/// gated section, a crash could leave it half-done and ignoring poison would no
/// longer be safe.
fn lock_gate(gate: &Mutex<()>) -> MutexGuard<'_, ()> {
    gate.lock().unwrap_or_else(PoisonError::into_inner)
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

/// Reject a schema that the engine can build but can never query. Arrow
/// permits a `Schema` with duplicate field names, yet any query against a
/// table built from one fails on the ambiguous column. Catching it here turns
/// a table that would otherwise error on every read into a create-time
/// rejection.
/// The seed `create_table` takes for `patch`: every listed column with its
/// type (a new table's columns all need one), and the indexes they declare.
/// Fields marked `dropped` are left out.
fn seed_from_patch(patch: &SchemaPatch) -> Result<(SchemaRef, IndexSpec), InfinoError> {
    let mut fields = Vec::with_capacity(patch.fields.len());
    let mut indexes = IndexSpec::new();
    for field in patch.fields.iter().filter(|f| !f.dropped) {
        let data_type = field.data_type.clone().ok_or_else(|| {
            InfinoError::Schema(SchemaError::TypeRequired {
                column: field.name.clone(),
            })
        })?;
        match &field.index {
            Some(ColumnIndex::Fts {
                stopwords,
                stemmer,
                positions,
                stored,
                bm25,
            }) => {
                indexes = indexes.fts(
                    FtsField::new(field.name.clone())
                        .stopwords(*stopwords)
                        .stemmer(*stemmer)
                        .positions(*positions)
                        .stored(*stored)
                        .bm25(bm25.k1, bm25.b),
                );
            }
            Some(ColumnIndex::Vector {
                metric,
                rot_seed,
                rerank_codec,
            }) => {
                let DataType::FixedSizeList(_, dim) = &data_type else {
                    return Err(InfinoError::Schema(SchemaError::InvalidIndex {
                        column: field.name.clone(),
                        reason: "a vector index needs a vector column".to_owned(),
                    }));
                };
                // Carry what the document recorded. Taking the defaults
                // instead would accept a patch naming a seed or a codec and
                // silently build the table with different ones, so cloning a
                // table's schema onto a new one would not reproduce it.
                indexes = indexes.vector_as_recorded(
                    field.name.clone(),
                    *dim as usize,
                    *metric,
                    *rot_seed,
                    *rerank_codec,
                );
            }
            None => {}
        }
        fields.push(Field::new(
            &field.name,
            data_type,
            field.nullable.unwrap_or(true),
        ));
    }
    Ok((Arc::new(Schema::new(fields)), indexes))
}

fn validate_schema(schema: &SchemaRef) -> Result<(), InfinoError> {
    let mut seen = HashSet::new();
    for field in schema.fields() {
        if !seen.insert(field.name().as_str()) {
            return Err(InfinoError::Schema(SchemaError::DuplicateColumn {
                name: field.name().clone(),
            }));
        }
    }
    Ok(())
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
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::{
            Arc, Mutex, MutexGuard, PoisonError,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread,
        time::{Duration, Instant},
    };

    use arrow::util::{display::array_value_to_string, pretty::pretty_format_batches};
    use arrow_array::{
        Array, ArrayRef, BooleanArray, Decimal128Array, FixedSizeListArray, Float32Array,
        Float64Array, Int64Array, LargeStringArray, StringViewArray, cast::AsArray,
        types::Int64Type,
    };
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        error::DataFusionError,
        logical_expr::LogicalPlan,
        prelude::{SessionContext, col, lit},
    };
    use proptest::prelude::*;
    use serde_json::{Map, Value, json};

    use super::*;
    use crate::{
        Bm25SearchOptions, BoolMode, CompactionSettings, Consistency, FieldPatch, Metric,
        OptimizeOptions, SchemaPatch, Stemmer, Stopwords,
        catalog::manifest::CATALOG_PATH,
        supertable::{
            manifest::commit::POINTER_PATH,
            query::{provider::TABLE_NAME, sql::value_count_schema},
        },
        test_helpers::{
            build_title_batch,
            fault_storage::{FaultOp, FaultStorage},
            schema_id_title,
            slow_storage::SlowStorage,
        },
    };

    const TOP_K: usize = 10;

    /// Total rows across the materialized search batches.
    fn n_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|b| b.num_rows()).sum()
    }

    /// `create_database` is a no-op success on a local backend (the catalog
    /// root is the database), so the same "provision then create a table" setup
    /// code that a hosted target needs runs unchanged against a durable local
    /// one — it doesn't error, and a table created afterward is queryable.
    #[test]
    fn local_create_database_is_a_noop_then_table_works() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");

        conn.create_database()
            .expect("create_database is a no-op success for a local backend");
        // Idempotent: a second call is still fine.
        conn.create_database().expect("second create_database");

        conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table after create_database")
            .append(&build_title_batch(&["fox"]))
            .expect("append");
        assert_eq!(count_rows(&conn, "docs"), 1);
    }

    /// A durable (Storage-backed) connection over a fresh temp dir. The
    /// returned `TempDir` must stay in scope: dropping it deletes the catalog.
    fn storage_conn() -> (Connection, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");
        (conn, dir)
    }

    /// Borrow the Storage backend's `building` (build gates) and `handles`
    /// (warm memo) maps, or fail loudly if the connection is not durable.
    fn storage_maps(
        conn: &Connection,
    ) -> (
        &DashMap<String, Arc<Mutex<()>>>,
        &DashMap<String, SupertableHandle>,
    ) {
        match &conn.inner.store {
            CatalogStore::Storage {
                building, handles, ..
            } => (building, handles),
            _ => panic!("expected a Storage-backed catalog"),
        }
    }

    /// Poison a name's build gate exactly the way a panicking cold-path build
    /// does: lock the gate on another thread and panic while the guard is held,
    /// which drops the guard mid-unwind and marks the `Mutex` poisoned.
    fn poison_gate(building: &DashMap<String, Arc<Mutex<()>>>, name: &str) {
        let gate = single_flight_gate(building, name);
        let joined = thread::spawn(move || {
            let _guard = gate.lock().expect("lock gate to poison it");
            panic!("simulated build panic under the gate (e.g. rayon EAGAIN)");
        })
        .join();
        assert!(joined.is_err(), "the poisoning thread must have panicked");
        assert!(
            single_flight_gate(building, name).is_poisoned(),
            "the gate must be poisoned after a held-guard panic",
        );
    }

    /// Run `f`, returning `Err(())` if it panicked. The panic hook is silenced
    /// for the call so a caught panic does not spew to stderr; a real assertion
    /// failure elsewhere still prints normally.
    fn without_panic<T>(f: impl FnOnce() -> T) -> Result<T, ()> {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        std::panic::set_hook(prev);
        out.map_err(|_| ())
    }

    // ---- Regression: "catalog build gate poisoned" crash loop -------------
    //
    // A build that panics while holding a name's single-flight gate poisons
    // that gate's `Mutex`. In production the panic came from rayon's thread
    // pool failing to spawn OS threads (`EAGAIN` / "Resource temporarily
    // unavailable") during a cold-path build. The gate `Arc<Mutex<()>>` is
    // cached per name and never evicted, so a propagated `PoisonError` was
    // sticky: every later catalog op on that name re-locked the poisoned mutex
    // and panicked, turning one transient blip into a permanent crash loop.
    //
    // `lock_gate` recovers the poisoned guard instead of propagating it. Each
    // test below poisons a gate the way a panicking build would, then drives
    // one of the three ops that lock it (open / drop / create) and asserts the
    // op does not panic and still produces a correct result.

    #[test]
    fn poisoned_gate_does_not_wedge_open() {
        let (conn, _dir) = storage_conn();
        conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["fox"]))
            .expect("append");

        let (building, handles) = storage_maps(&conn);
        // Evict the warm handle so open takes the cold path and locks the gate,
        // exactly as a fresh worker does on first open.
        handles.remove("docs");
        poison_gate(building, "docs");

        let table = without_panic(|| conn.open_table("docs"))
            .expect("open_table must not panic on a poisoned gate (that was the crash loop)")
            .expect("open_table should rebuild after recovering the poisoned gate");
        assert_eq!(
            n_rows(
                &table
                    .bm25_search("title", "fox", TOP_K, Bm25SearchOptions::new(), None)
                    .expect("bm25_search after recovery"),
            ),
            1,
            "the recovered table must still be queryable",
        );
    }

    #[test]
    fn poisoned_gate_does_not_wedge_drop() {
        let (conn, _dir) = storage_conn();
        conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table");

        let (building, _handles) = storage_maps(&conn);
        poison_gate(building, "docs");

        without_panic(|| conn.drop_table("docs", false))
            .expect("drop_table must not panic on a poisoned gate")
            .expect("drop_table should succeed after recovering the poisoned gate");
        assert!(
            conn.list_tables().expect("list").is_empty(),
            "the table must be unregistered after drop",
        );
    }

    #[test]
    fn poisoned_gate_does_not_wedge_create() {
        let (conn, _dir) = storage_conn();

        // Pre-poison the gate for a name that does not exist yet, then create
        // it: `create_table` commits the memo under this same gate.
        let (building, _handles) = storage_maps(&conn);
        poison_gate(building, "fresh");

        let table = without_panic(|| {
            conn.create_table("fresh", schema_id_title(), IndexSpec::new().fts("title"))
        })
        .expect("create_table must not panic on a poisoned gate")
        .expect("create_table should succeed after recovering the poisoned gate");
        table
            .append(&build_title_batch(&["fox"]))
            .expect("append to the created table");
        assert_eq!(count_rows(&conn, "fresh"), 1, "the created table is usable");
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
            .bm25_search("title", "fox", TOP_K, Bm25SearchOptions::new(), None)
            .expect("bm25_search");
        assert_eq!(n_rows(&hits), 1, "expected one hit for 'fox'");

        conn.drop_table("docs", false).expect("drop_table");
        assert!(conn.list_tables().expect("list").is_empty());
        assert!(matches!(
            conn.open_table("docs"),
            Err(InfinoError::NotFound(_))
        ));
    }

    /// `standard` may be named explicitly; any other analyzer is refused
    /// at create, naming the column.
    #[test]
    fn create_table_accepts_only_the_standard_analyzer() {
        let conn = connect("memory://").expect("connect");
        conn.create_table(
            "pinned",
            schema_id_title(),
            IndexSpec::new().fts(FtsField::new("title").analyzer("standard")),
        )
        .expect("an explicit standard is accepted");

        let err = conn
            .create_table(
                "other",
                schema_id_title(),
                IndexSpec::new().fts(FtsField::new("title").analyzer("ascii_lower")),
            )
            .expect_err("a non-standard analyzer is refused");
        assert!(matches!(err, InfinoError::Config(_)), "got: {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("title") && msg.contains("ascii_lower"),
            "names the column and analyzer: {msg}"
        );
        assert!(
            !conn
                .list_tables()
                .expect("list")
                .contains(&"other".to_string()),
            "a refused create registers nothing"
        );
    }

    /// Stemming, end to end: one inflection finds the others because
    /// both sides of the search run through the same chain.
    #[test]
    fn stemming_folds_inflections_end_to_end() {
        let conn = connect("memory://").expect("connect");
        let stemmed = conn
            .create_table(
                "stemmed",
                schema_id_title(),
                IndexSpec::new().fts(FtsField::new("title").stemmer(Stemmer::English)),
            )
            .expect("create stemmed table");
        stemmed
            .append(&build_title_batch(&[
                "running late",
                "she runs fast",
                "a walk",
            ]))
            .expect("append");

        // Every inflection reaches both documents holding one, whichever
        // one the query spells.
        for query in ["running", "runs", "run"] {
            let hits = stemmed
                .bm25_search("title", query, TOP_K, Bm25SearchOptions::new(), None)
                .expect("bm25_search");
            assert_eq!(
                n_rows(&hits),
                2,
                "{query:?} must reach both inflections on a stemmed column"
            );
        }

        // And the same corpus without the stemmer separates them, so the
        // declaration is what changed the answer.
        let plain = conn
            .create_table("plain", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create plain table");
        plain
            .append(&build_title_batch(&[
                "running late",
                "she runs fast",
                "a walk",
            ]))
            .expect("append");
        let hits = plain
            .bm25_search("title", "run", TOP_K, Bm25SearchOptions::new(), None)
            .expect("bm25_search");
        assert_eq!(
            n_rows(&hits),
            0,
            "an unstemmed column matches the word only"
        );
    }

    /// Stopword removal, end to end, including the consequence worth
    /// being explicit about: once a word is not indexed, no query can
    /// find it.
    #[test]
    fn stopwords_leave_the_index_and_the_query() {
        let conn = connect("memory://").expect("connect");
        let stopped = conn
            .create_table(
                "stopped",
                schema_id_title(),
                IndexSpec::new().fts(FtsField::new("title").stopwords(Stopwords::English)),
            )
            .expect("create stopped table");
        stopped
            .append(&build_title_batch(&["the fox and the hound", "a cat"]))
            .expect("append");

        // The content words still search normally.
        let hits = stopped
            .bm25_search("title", "fox", TOP_K, Bm25SearchOptions::new(), None)
            .expect("bm25_search");
        assert_eq!(n_rows(&hits), 1);

        // A stopword leaves the query too, so a query of nothing but
        // stopwords has no term left to match — not an error, no rows.
        let hits = stopped
            .bm25_search("title", "the and", TOP_K, Bm25SearchOptions::new(), None)
            .expect("a stopword-only query is not an error");
        assert_eq!(n_rows(&hits), 0, "nothing is left of the query to match");

        // And a query mixing the two searches only what survives, so the
        // stopword neither narrows nor widens the result.
        let hits = stopped
            .bm25_search("title", "the fox", TOP_K, Bm25SearchOptions::new(), None)
            .expect("bm25_search");
        assert_eq!(n_rows(&hits), 1);
    }

    /// The hole a removed stopword leaves is what keeps a phrase honest:
    /// `"new york"` must not match `new the york`, and
    /// `"end of the world"` must still match the text it came from.
    #[test]
    fn stopword_holes_keep_phrase_spacing_end_to_end() {
        let conn = connect("memory://").expect("connect");
        let table = conn
            .create_table(
                "phrases",
                schema_id_title(),
                IndexSpec::new().fts(
                    FtsField::new("title")
                        .stopwords(Stopwords::English)
                        .positions(true),
                ),
            )
            .expect("create table");
        table
            .append(&build_title_batch(&[
                "new york city",                // 0: the words are adjacent
                "new the york city",            // 1: a removed word sits between them
                "the end of the world is nigh", // 2
                "end world",                    // 3: no gap where the phrase wants one
            ]))
            .expect("append");

        let phrase = |q: &str| -> usize {
            n_rows(
                &table
                    .bm25_search("title", q, TOP_K, Bm25SearchOptions::new(), None)
                    .expect("phrase search"),
            )
        };

        // Adjacent in the text and adjacent in the phrase: a match. The
        // document with a removed word between them is *not* one — its
        // `york` sits one position further along, exactly where the hole
        // left it.
        assert_eq!(
            phrase("\"new york\""),
            1,
            "only the text whose words are really adjacent"
        );
        // The query's own removed words become the spacing it asks for:
        // `end` and `world` three positions apart, which is where the
        // same chain put them in document 2 — and not in document 3,
        // where they are adjacent.
        assert_eq!(
            phrase("\"end of the world\""),
            1,
            "the phrase asks for the spacing its own stopwords imply"
        );
        // Naming the surviving words as an adjacent phrase finds the
        // document where they *are* adjacent, and only that one.
        assert_eq!(phrase("\"end world\""), 1);
    }

    /// A column with filters survives a reopen: the catalog record
    /// carries the tokenizer and both filters, so query text is
    /// tokenized the same way after reopening as before, and the
    /// table's options-hash still verifies.
    #[test]
    fn a_chained_column_survives_reopen_on_storage() {
        let (conn, _dir) = storage_conn();
        {
            let table = conn
                .create_table(
                    "docs",
                    schema_id_title(),
                    IndexSpec::new().fts(
                        FtsField::new("title")
                            .stopwords(Stopwords::English)
                            .stemmer(Stemmer::English),
                    ),
                )
                .expect("create_table");
            table
                .append(&build_title_batch(&["the running studies"]))
                .expect("append");
        }
        // A fresh connection over the same root, so the spec is rebuilt
        // from the catalog rather than reused from memory.
        let uri = _dir.path().to_str().expect("utf8 path").to_string();
        let reopened = connect(&uri).expect("reconnect");
        let table = reopened.open_table("docs").expect("open_table");
        let hits = table
            .bm25_search(
                "title",
                "the studies",
                TOP_K,
                Bm25SearchOptions::new(),
                None,
            )
            .expect("bm25_search after reopen");
        assert_eq!(
            n_rows(&hits),
            1,
            "the reopened table still stems and still drops stopwords"
        );
    }

    /// The `positions` flag has to round-trip the catalog record, and
    /// the failure is not the obvious one: the flag joins the table's
    /// options-hash, so a record that lost it would make a positional
    /// table fail its **own** hash check on reopen — refusing to open
    /// at all — rather than merely forgetting how to answer a phrase.
    #[test]
    fn a_positional_column_reopens_and_still_answers_phrases() {
        let (conn, dir) = storage_conn();
        {
            let table = conn
                .create_table(
                    "docs",
                    schema_id_title(),
                    IndexSpec::new().fts(FtsField::new("title").positions(true)),
                )
                .expect("create_table");
            table
                .append(&build_title_batch(&["new york city", "york new city"]))
                .expect("append");
            // The phrase works before the reopen, so a failure after it
            // is the round-trip and not the declaration.
            assert_eq!(
                n_rows(
                    &table
                        .bm25_search(
                            "title",
                            "\"new york\"",
                            TOP_K,
                            Bm25SearchOptions::new(),
                            None
                        )
                        .expect("phrase search")
                ),
                1
            );
        }
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let reopened = connect(&uri).expect("reconnect");
        let table = reopened
            .open_table("docs")
            .expect("a positional table must reopen — losing the flag fails the options-hash");
        assert_eq!(
            n_rows(
                &table
                    .bm25_search(
                        "title",
                        "\"new york\"",
                        TOP_K,
                        Bm25SearchOptions::new(),
                        None
                    )
                    .expect("phrase search after reopen")
            ),
            1,
            "the reopened table still records positions"
        );
    }

    /// The other half of exposing `positions`: a column without them
    /// answers a phrase query with an error naming the column, never a
    /// silent bag-of-words fallback that would return documents holding
    /// the words in the wrong order.
    #[test]
    fn a_positionless_column_rejects_a_phrase_query() {
        let conn = connect("memory://").expect("connect");
        let table = conn
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table");
        table
            .append(&build_title_batch(&["york new city"]))
            .expect("append");
        let err = table
            .bm25_search(
                "title",
                "\"new york\"",
                TOP_K,
                Bm25SearchOptions::new(),
                None,
            )
            .expect_err("a phrase on a positionless column must be an error");
        let msg = err.to_string();
        assert!(
            msg.contains("title"),
            "the error must name the column, got: {msg}"
        );
    }

    /// A catalog record naming a filter this engine cannot reproduce
    /// makes the table unusable rather than usable-with-a-guess. Same
    /// rule as the superfile entry: an *absent* filter means off, and a
    /// *present* name that does not resolve is refused — analyzing
    /// without the set the postings were built with is a different
    /// index, not a degraded one.
    #[test]
    fn a_catalog_record_naming_an_unresolvable_filter_is_refused() {
        for (field, bad) in [("fts_stopwords", "german"), ("fts_stemmers", "porter")] {
            let (conn, dir) = storage_conn();
            conn.create_table(
                "docs",
                schema_id_title(),
                IndexSpec::new().fts(FtsField::new("title").stopwords(Stopwords::English)),
            )
            .expect("create_table");
            // Rewrite just that column's filter name in the catalog,
            // leaving everything else intact.
            let path = dir.path().join(CATALOG_PATH);
            let body = std::fs::read_to_string(&path).expect("read catalog");
            let patched = body.replace(
                &format!("\"{field}\":[\"english\"]"),
                &format!("\"{field}\":[\"{bad}\"]"),
            );
            let patched = match patched == body {
                // The stemmer list is empty in this fixture, so inject.
                true => body.replace(
                    &format!("\"{field}\":[\"\"]"),
                    &format!("\"{field}\":[\"{bad}\"]"),
                ),
                false => patched,
            };
            assert_ne!(patched, body, "{field}: fixture did not patch");
            std::fs::write(&path, &patched).expect("write catalog");

            let uri = dir.path().to_str().expect("utf8 path").to_string();
            let reopened = connect(&uri).expect("reconnect");
            let err = reopened
                .open_table("docs")
                .expect_err("an unresolvable filter must be refused");
            let msg = err.to_string();
            assert!(
                msg.contains(bad) && msg.contains("title"),
                "{field}: the error must name the value and the column, got: {msg}"
            );
        }
    }

    #[test]
    fn standard_analyzer_keeps_non_ascii_end_to_end() {
        // A bare declaration is analyzed by `standard`, which keeps
        // non-ASCII, so "café" matches through the full create → append →
        // search path.
        let conn = connect("memory://").expect("connect");
        let table = conn
            .create_table("dflt", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create default table");
        table
            .append(&build_title_batch(&["café latte"]))
            .expect("append");
        let hits = table
            .bm25_search("title", "café", TOP_K, Bm25SearchOptions::new(), None)
            .expect("bm25_search");
        assert_eq!(
            n_rows(&hits),
            1,
            "a bare declaration keeps the non-ASCII term"
        );
    }

    fn schema_title_body() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("body", DataType::LargeUtf8, false),
        ]))
    }

    fn title_body_batch(schema: Arc<Schema>, title: &str, body: &str) -> RecordBatch {
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(LargeStringArray::from(vec![title])),
                Arc::new(LargeStringArray::from(vec![body])),
            ],
        )
        .expect("batch shape matches schema")
    }

    /// Two text columns, different analysis: `title` plain, `body`
    /// stemmed. The same text holds `running` in both, so `run` reaches
    /// it through `body` but not `title` — each column is indexed AND
    /// queried with its own chain, not column 0's for all.
    #[test]
    fn mixed_per_column_analysis_indexes_and_queries_independently() {
        let conn = connect("memory://").expect("connect");
        let schema = schema_title_body();
        let table = conn
            .create_table(
                "docs",
                schema.clone(),
                IndexSpec::new()
                    .fts("title")
                    .fts(FtsField::new("body").stemmer(Stemmer::English)),
            )
            .expect("create_table");
        table
            .append(&title_body_batch(schema, "running late", "running late"))
            .expect("append");

        let search = |column: &str, query: &str| {
            table
                .bm25_search(column, query, TOP_K, Bm25SearchOptions::new(), None)
                .map(|h| n_rows(&h))
                .unwrap_or(0)
        };
        assert_eq!(
            search("body", "run"),
            1,
            "the stemmed column folds the inflection"
        );
        assert_eq!(search("title", "run"), 0, "the plain column does not");
        assert_eq!(search("title", "running"), 1, "the plain column is indexed");
    }

    /// Rewrite table `docs`'s catalog record through `edit`.
    fn edit_catalog_entry(dir: &Path, edit: impl FnOnce(&mut Map<String, Value>)) {
        let catalog_file = dir.join(CATALOG_PATH);
        let mut body: Value =
            serde_json::from_slice(&fs::read(&catalog_file).expect("read catalog"))
                .expect("catalog json");
        edit(body["tables"]["docs"].as_object_mut().expect("table entry"));
        fs::write(
            &catalog_file,
            serde_json::to_vec(&body).expect("encode catalog"),
        )
        .expect("write catalog");
    }

    /// A new record names `standard` per column, and a recorded name is
    /// checked on open: absent or `standard` opens, any other name refuses
    /// the table with the re-create instruction rather than querying it
    /// with `standard`.
    #[test]
    fn open_table_checks_a_recorded_analyzer_name() {
        let (conn, dir) = storage_conn();
        conn.create_table(
            "docs",
            schema_title_body(),
            IndexSpec::new().fts("title").fts("body"),
        )
        .expect("create_table");
        let uri = dir.path().to_str().expect("utf8 path").to_string();

        let mut written = None;
        edit_catalog_entry(dir.path(), |entry| {
            written = entry.get("fts_analyzers").cloned();
        });
        assert_eq!(
            written,
            Some(json!(["standard", "standard"])),
            "a new record names `standard` per column"
        );
        edit_catalog_entry(dir.path(), |entry| {
            entry.remove("fts_analyzers");
        });
        connect(&uri)
            .expect("reconnect")
            .open_table("docs")
            .expect("an absent analyzer list opens");

        edit_catalog_entry(dir.path(), |entry| {
            entry.insert("fts_analyzers".into(), json!(["standard", "standard"]));
        });
        connect(&uri)
            .expect("reconnect")
            .open_table("docs")
            .expect("a recorded standard opens");

        edit_catalog_entry(dir.path(), |entry| {
            entry.insert("fts_analyzers".into(), json!(["standard", "ascii_lower"]));
        });
        let err = connect(&uri)
            .expect("reconnect")
            .open_table("docs")
            .expect_err("a removed analyzer must be refused");
        assert!(matches!(err, InfinoError::Config(_)), "got: {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("docs")
                && msg.contains("body")
                && msg.contains("ascii_lower")
                && msg.contains("re-create"),
            "must name the table, column, analyzer and fix: {msg}"
        );

        // An unrelated catalog commit rewrites every record; the removed
        // analyzer must survive it, or the refusal silently disappears.
        connect(&uri)
            .expect("reconnect")
            .create_table("other", schema_title_body(), IndexSpec::new().fts("title"))
            .expect("an unrelated create_table");
        let err = connect(&uri)
            .expect("reconnect")
            .open_table("docs")
            .expect_err("the refusal survives an unrelated catalog commit");
        assert!(err.to_string().contains("ascii_lower"), "got: {err}");
    }

    /// An index-only column (`FtsField::stored(false)`) survives a
    /// storage-backed reopen: the catalog records the flag, `open_table`
    /// reconstructs it, and the reopened table both searches the column
    /// and keeps rejecting it as a projection target.
    #[test]
    fn index_only_column_survives_reopen() {
        let dir = std::env::temp_dir().join(format!("infino-idxonly-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("mkdir");
        let uri = format!("file://{}", dir.display());
        let schema = schema_title_body();
        {
            let conn = connect(&uri).expect("connect");
            let table = conn
                .create_table(
                    "docs",
                    schema.clone(),
                    IndexSpec::new()
                        .fts("title")
                        .fts(FtsField::new("body").stored(false)),
                )
                .expect("create_table");
            table
                .append(&title_body_batch(
                    schema.clone(),
                    "stored title",
                    "hidden signal text",
                ))
                .expect("append");
        }
        let conn2 = connect(&uri).expect("reconnect");
        let table = conn2.open_table("docs").expect("open_table");
        // schema() keeps the ingest contract, index-only column included.
        assert_eq!(
            table
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect::<Vec<_>>(),
            vec!["title", "body"],
        );
        // Searchable after reopen…
        let hits = table
            .bm25_search("body", "signal", TOP_K, Bm25SearchOptions::new(), None)
            .expect("index-only search after reopen");
        assert_eq!(n_rows(&hits), 1);
        // …and still not readable: projecting it fails with a clean,
        // caller-level error.
        let err = table
            .bm25_search(
                "body",
                "signal",
                TOP_K,
                Bm25SearchOptions::new(),
                Some(&["_id", "body", "score"]),
            )
            .expect_err("index-only column must not be projectable after reopen");
        let msg = err.to_string();
        assert!(msg.contains("body"), "error names the column: {msg}");
        assert!(!msg.contains("DataFusion"), "no engine internals: {msg}");
        // Appends still require the column (it is part of the write
        // contract even though it is never stored).
        let title_only = Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]));
        let short = RecordBatch::try_new(
            title_only,
            vec![Arc::new(LargeStringArray::from(vec!["no body"]))],
        )
        .expect("batch");
        assert!(
            table.append(&short).is_err(),
            "append without the index-only column must be rejected"
        );
        let _ = fs::remove_dir_all(&dir);
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
            .bm25_search("title", "fox", TOP_K, Bm25SearchOptions::new(), None)
            .expect("bm25_search on empty table");
        assert_eq!(n_rows(&before), 0, "freshly opened table starts empty");

        // Fully usable: the first commit lands through the reopened handle,
        // then the query round-trips.
        opened
            .append(&build_title_batch(&["the quick brown fox"]))
            .expect("append via reopened handle");
        let hits = opened
            .bm25_search("title", "fox", TOP_K, Bm25SearchOptions::new(), None)
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
            .open_table_handle("docs")
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
            .open_table_handle("docs")
            .expect("open")
            .stats()
            .n_cold_fetches
            .expect("disk cache attached");
        assert_eq!(
            cold_after_q2, cold_after_q1,
            "second query must reuse the warm disk cache, not cold-fetch again"
        );
    }

    /// The files under a table's disk-cache directory, its location record
    /// aside, by path relative to the directory.
    fn table_cache_files(table_root: &Path) -> HashSet<PathBuf> {
        let mut found = HashSet::new();
        let mut pending = vec![table_root.to_path_buf()];
        while let Some(dir) = pending.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    pending.push(path);
                } else if !path.ends_with(TABLE_CACHE_LOCATION_FILE) {
                    found.insert(
                        path.strip_prefix(table_root)
                            .expect("under root")
                            .to_path_buf(),
                    );
                }
            }
        }
        found
    }

    /// A cached reader over `uri`, with strong reads so it sees a peer's drop.
    fn cached_reader(uri: &str, cache: &Path) -> Connection {
        connect_with(
            uri,
            ConnectOptions::new()
                .with_cache_dir(cache.to_path_buf())
                .with_read_consistency(Consistency::Strong),
        )
        .expect("connect cached reader")
    }

    /// Dropping a table removes its local disk cache: the bytes belong to a
    /// generation nothing will read again, and kept, they count against the
    /// disk the cache shares with every other table.
    #[test]
    fn dropping_a_table_removes_its_disk_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = tempfile::tempdir().expect("cache dir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        connect(&uri)
            .expect("connect writer")
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["the quick brown fox"]))
            .expect("append");
        let reader = cached_reader(&uri, cache.path());
        reader.query_sql("SELECT title FROM docs").expect("scan");
        let table_root = cache.path().join("docs");
        assert!(
            !table_cache_files(&table_root).is_empty(),
            "the scan cached the superfile"
        );

        reader.drop_table("docs", true).expect("drop");
        assert!(!table_root.exists(), "the dropped table's cache is gone");
    }

    /// A table dropped and re-created by another process keeps its name, and
    /// so its cache directory here; opening the new generation clears the old
    /// one's bytes rather than keep them beside its own until eviction.
    #[test]
    fn a_re_created_table_clears_the_dropped_generations_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = tempfile::tempdir().expect("cache dir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let writer = connect_with(
            &uri,
            ConnectOptions::new().with_read_consistency(Consistency::Strong),
        )
        .expect("connect writer");
        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["the quick brown fox"]))
            .expect("append");
        let reader = cached_reader(&uri, cache.path());
        reader.query_sql("SELECT title FROM docs").expect("scan");
        let table_root = cache.path().join("docs");
        let first = table_cache_files(&table_root);
        assert!(!first.is_empty(), "the first generation is cached");

        writer.drop_table("docs", true).expect("drop elsewhere");
        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("re-create")
            .append(&build_title_batch(&["a slow green turtle"]))
            .expect("append");
        // Trip the reader's freshness probe, as a peer's drop is first seen
        // (`storage_recreated_table_after_a_peer_drop_reads_the_new_generation`).
        let _ = reader.query_sql("SELECT COUNT(*) FROM docs");
        reader
            .query_sql("SELECT title FROM docs")
            .expect("scan again");

        let second = table_cache_files(&table_root);
        assert!(!second.is_empty(), "the new generation is cached");
        assert!(
            first.is_disjoint(&second),
            "no file of the dropped generation is left: {first:?} {second:?}"
        );
    }

    /// A cache directory with no location record - written before the record
    /// existed - holds bytes of no known generation, and is emptied when the
    /// table is opened.
    #[test]
    fn a_table_cache_of_no_recorded_generation_is_emptied_on_open() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = tempfile::tempdir().expect("cache dir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        connect(&uri)
            .expect("connect writer")
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["the quick brown fox"]))
            .expect("append");
        let table_root = cache.path().join("docs");
        fs::create_dir_all(&table_root).expect("mkdir");
        let leftover = table_root.join("seg-leftover.sf.parquet");
        fs::write(&leftover, b"bytes of an unknown generation").expect("seed");

        cached_reader(&uri, cache.path())
            .open_table_handle("docs")
            .expect("open");
        assert!(!leftover.exists(), "the unrecorded bytes are gone");
        assert!(
            table_root.join(TABLE_CACHE_LOCATION_FILE).exists(),
            "the directory now records its generation"
        );
    }

    /// Every `_supertable/current` pointer file under `root`, recursively — the
    /// on-storage evidence that a supertable exists.
    fn pointer_files(root: &Path) -> Vec<PathBuf> {
        let mut found = Vec::new();
        let Ok(entries) = fs::read_dir(root) else {
            return found;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                found.extend(pointer_files(&path));
            } else if path.ends_with(POINTER_PATH) {
                found.push(path);
            }
        }
        found
    }

    /// Two connections over one storage root, the shape of a database served by
    /// more than one process. One drops and purges a table; the other has it warm
    /// in its per-process handle cache, which the drop never reaches, and must
    /// stop serving it rather than answer from deleted superfiles forever.
    #[test]
    fn storage_purged_table_is_not_served_from_a_peer_connections_warm_handle() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();

        let writer = connect_with(
            &uri,
            ConnectOptions::new().with_read_consistency(Consistency::Strong),
        )
        .expect("connect writer");
        let peer = connect_with(
            &uri,
            ConnectOptions::new().with_read_consistency(Consistency::Strong),
        )
        .expect("connect peer");

        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["the quick brown fox"]))
            .expect("append");

        // Warm the peer's handle cache, and confirm it really is serving.
        assert_eq!(count_rows(&peer, "docs"), 1, "peer reads the seeded row");

        writer.drop_table("docs", true).expect("drop_table");

        // The next freshness probe discovers the deletion, and it runs inside
        // the query path after the cached handle was taken — so this call is the
        // trigger and recovery lands on the one after it.
        let _ = peer.query_sql("SELECT COUNT(*) FROM docs");

        let err = peer
            .open_table("docs")
            .expect_err("the purged table must not open");
        assert!(
            matches!(err, InfinoError::NotFound(_)),
            "expected NotFound, got {err:?}"
        );
        // `query_sql` reports the planner's failure to resolve the relation,
        // not our typed `NotFound` — an unregistrable name is skipped during
        // registration (it may be a CTE or a TVF argument), so the refusal
        // surfaces one layer up. The typed assertion is on `open_table` above;
        // what matters here is that the message is a missing *table* and not a
        // fetch of a purged superfile off the stale manifest, which is exactly
        // how this failed before.
        let err = peer
            .query_sql("SELECT COUNT(*) FROM docs")
            .expect_err("the purged table must not be queryable");
        let msg = err.to_string();
        assert!(
            msg.contains("docs") && !msg.contains(".sf.parquet"),
            "expected a missing-table error naming docs, got {err:?}"
        );
    }

    /// The purge seen from a table handle the caller is *holding*, rather than
    /// re-resolving by name — and with a disk cache, which is what makes this
    /// the sharpest case.
    ///
    /// Re-resolving recovers, because the catalog drops the dead handle and
    /// rebuilds. A held handle has no name to re-resolve, and the freshness
    /// probe inside it swallows errors by design, so nothing stops the read:
    /// the pinned manifest still names the purged superfiles and the cache
    /// still holds their bytes, so every search answers — correctly shaped,
    /// from a table that no longer exists — for as long as the handle lives.
    /// Storage never gets asked, so the deletion cannot surface on its own.
    #[test]
    fn storage_reads_on_a_purged_handle_refuse_instead_of_serving_cached_rows() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = tempfile::tempdir().expect("cache dir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();

        let writer = connect_with(
            &uri,
            ConnectOptions::new().with_read_consistency(Consistency::Strong),
        )
        .expect("connect writer");
        let peer = connect_with(
            &uri,
            ConnectOptions::new()
                .with_cache_dir(cache.path())
                .with_read_consistency(Consistency::Strong),
        )
        .expect("connect peer");

        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["the quick brown fox"]))
            .expect("append");

        // Read once through the held handle so the superfile bytes are resident
        // in the peer's disk cache — the state that lets a purged table keep
        // answering without ever touching storage again.
        let peer_table = peer.open_table("docs").expect("peer opens the table");
        assert_eq!(
            n_rows(
                &peer_table
                    .bm25_search("title", "quick", TOP_K, Bm25SearchOptions::new(), None)
                    .expect("warm read")
            ),
            1,
            "peer reads the seeded row, warming its cache"
        );

        writer.drop_table("docs", true).expect("drop_table");

        // Every read verb, twice: the first trips the probe that discovers the
        // purge, and the second proves the refusal is latched rather than a
        // one-shot side effect of that discovery.
        for attempt in 0..2 {
            let err = peer_table
                .bm25_search("title", "quick", TOP_K, Bm25SearchOptions::new(), None)
                .expect_err("bm25_search on a purged table must not return rows");
            assert!(
                matches!(err, InfinoError::NotFound(_)),
                "attempt {attempt}: expected NotFound, got {err:?}"
            );
            for err in [
                peer_table
                    .token_match("title", "quick", BoolMode::Or, None)
                    .expect_err("token_match must refuse"),
                peer_table
                    .exact_match("title", "the quick brown fox", None)
                    .expect_err("exact_match must refuse"),
                peer_table
                    .count("title", "quick", BoolMode::Or)
                    .expect_err("count must refuse"),
            ] {
                assert!(
                    matches!(err, InfinoError::NotFound(_)),
                    "attempt {attempt}: expected NotFound, got {err:?}"
                );
            }
        }

        // Mutations refuse at predicate resolution, before writing any WAL
        // state, and report the same missing table rather than a backend fault.
        let err = peer_table
            .delete(col("_id").eq(lit(1_i64)))
            .expect_err("delete on a purged table must refuse");
        assert!(
            matches!(err, InfinoError::NotFound(_)),
            "expected NotFound, got {err:?}"
        );
    }

    /// The same stale handle, written to rather than read from. A commit fences
    /// on the pointer's etag, and an absent pointer used to mean "initial
    /// commit" — republishing one from the stale manifest and resurrecting the
    /// table under a name the catalog no longer lists. Hence the assertion on
    /// storage state, not just on the error.
    #[test]
    fn storage_append_on_a_purged_handle_refuses_and_republishes_no_pointer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();

        let writer = connect(&uri).expect("connect writer");
        let peer = connect(&uri).expect("connect peer");

        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["the quick brown fox"]))
            .expect("append");

        // Write once through the peer, so its handle carries real manifest state.
        let peer_table = peer.open_table("docs").expect("peer opens the table");
        peer_table
            .append(&build_title_batch(&["peer row"]))
            .expect("peer append before the drop");

        writer.drop_table("docs", true).expect("drop_table");
        assert!(
            pointer_files(dir.path()).is_empty(),
            "the purge should leave no pointer behind"
        );

        let err = peer_table
            .append(&build_title_batch(&["after the drop"]))
            .expect_err("appending to a purged table must fail");
        // The same answer the read path gives. It has to survive the commit →
        // build → mutation-commit error hops the append path takes, or a caller
        // sees an indistinguishable backend fault and retries — and every retry
        // uploads another superfile before reaching the fence that refuses it.
        assert!(
            matches!(err, InfinoError::NotFound(_)),
            "expected NotFound, got {err:?}"
        );

        assert!(
            pointer_files(dir.path()).is_empty(),
            "the refused append must not republish a pointer (resurrecting the \
             dropped table as unreachable, unreclaimable data): {err:?}"
        );
        assert!(
            writer.list_tables().expect("list").is_empty(),
            "the table stays dropped"
        );
    }

    /// The write-only twin of the read-path recovery. A handle that has never
    /// served a read has never run a freshness probe, so the commit's pointer
    /// fence is the only thing that can notice the purge — and unless that
    /// observation latches, the catalog goes on serving the dead handle from
    /// cache and every later append fences against a location a re-create has
    /// already replaced. Correctly refusing forever is still broken.
    #[test]
    fn storage_appends_recover_after_a_peer_drop_and_recreate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();

        let writer = connect(&uri).expect("connect writer");
        let peer = connect(&uri).expect("connect peer");

        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["seeded"]))
            .expect("append");

        // The peer only ever writes — no query, so no freshness probe.
        let peer_table = peer.open_table("docs").expect("peer opens the table");
        peer_table
            .append(&build_title_batch(&["peer row"]))
            .expect("peer append before the drop");

        writer.drop_table("docs", true).expect("drop_table");

        let err = peer_table
            .append(&build_title_batch(&["after the drop"]))
            .expect_err("appending to a purged table must refuse");
        assert!(
            matches!(err, InfinoError::NotFound(_)),
            "expected NotFound, got {err:?}"
        );

        // The name comes back at a fresh location. Re-resolving through the
        // connection must rebuild rather than hand back the dead handle.
        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("re-create")
            .append(&build_title_batch(&["new one"]))
            .expect("append to the new generation");

        peer.open_table("docs")
            .expect("peer re-opens the re-created table")
            .append(&build_title_batch(&["peer writes again"]))
            .expect("a write-only peer must recover after the re-create");
        assert_eq!(
            count_rows(&writer, "docs"),
            2,
            "the new generation holds its own row plus the peer's"
        );
    }

    /// Drop-then-recreate through different connections: the name is back, but at
    /// a fresh location, so the peer must rebuild rather than serve the old rows.
    #[test]
    fn storage_recreated_table_after_a_peer_drop_reads_the_new_generation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();

        let writer = connect_with(
            &uri,
            ConnectOptions::new().with_read_consistency(Consistency::Strong),
        )
        .expect("connect writer");
        let peer = connect_with(
            &uri,
            ConnectOptions::new().with_read_consistency(Consistency::Strong),
        )
        .expect("connect peer");

        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["old one", "old two", "old three"]))
            .expect("append");
        assert_eq!(count_rows(&peer, "docs"), 3, "peer warms on the old table");

        writer.drop_table("docs", true).expect("drop_table");
        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("re-create")
            .append(&build_title_batch(&["new one"]))
            .expect("append to the new generation");

        // Trip the peer's freshness probe (see the read-path test above).
        let _ = peer.query_sql("SELECT COUNT(*) FROM docs");
        assert_eq!(
            count_rows(&peer, "docs"),
            1,
            "peer must rebuild against the re-created table, not serve the \
             dropped generation's rows"
        );
    }

    /// The same purge, against a peer handle that has never served a read.
    ///
    /// Freshness is discovered by re-probing the pointer, and the probe carries
    /// the etag of the last one read — which only a previous probe sets. So a
    /// handle built but not yet queried has no etag, and neither does one on a
    /// backend that omits them. Keying "did we have a pointer?" on that etag
    /// therefore reads this deletion as "nothing newer to load", and since the
    /// miss also leaves the etag unset, every later probe repeats it: the
    /// handle serves the purged table off its in-memory manifest for as long
    /// as the process lives, and a re-create never reaches it either. The
    /// pointer's absence is what makes it fatal — not our record of it.
    #[test]
    fn storage_purged_table_is_not_served_from_a_handle_that_never_read() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();

        let writer = connect(&uri).expect("connect writer");
        let peer = connect(&uri).expect("connect peer");

        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["old one", "old two", "old three"]))
            .expect("append");

        // Warm the peer's handle cache *without* querying through it, so no
        // freshness probe has run and no pointer etag has been recorded.
        peer.open_table("docs").expect("peer opens the table");

        writer.drop_table("docs", true).expect("drop_table");

        // Trip the probe (it runs inside the query path, after the cached
        // handle has been taken — so recovery lands on the call after it).
        let _ = peer.query_sql("SELECT COUNT(*) FROM docs");

        let err = peer
            .query_sql("SELECT COUNT(*) FROM docs")
            .expect_err("the purged table must not be queryable");
        assert!(
            !err.to_string().contains(".sf.parquet"),
            "the peer must report the table gone, not fail fetching a purged \
             superfile off its stale manifest: {err:?}"
        );

        // And the handle is genuinely replaced, not just poisoned: a re-create
        // under the same name is visible to this connection.
        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("re-create")
            .append(&build_title_batch(&["new one"]))
            .expect("append to the new generation");
        assert_eq!(
            count_rows(&peer, "docs"),
            1,
            "peer must rebuild against the re-created table"
        );
    }

    /// The premise the read-path check rests on: `create` publishes a pointer
    /// before any writer runs, so a table with nothing appended to it still has
    /// one and reads as empty. That is what makes an *absent* pointer
    /// unambiguous — never "not committed yet", always "deleted under us".
    #[test]
    fn storage_table_with_no_appends_still_reads_as_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();

        let conn = connect(&uri).expect("connect");
        conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table");

        assert_eq!(
            count_rows(&conn, "docs"),
            0,
            "an unwritten table reads empty"
        );
        conn.open_table("docs")
            .expect("an unwritten table still opens");

        // And from a second connection, which builds its handle from scratch.
        let peer = connect(&uri).expect("connect peer");
        assert_eq!(count_rows(&peer, "docs"), 0);
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
            .open_table_handle("docs")
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

        let writer = connect_with(
            &uri,
            ConnectOptions::new().with_read_consistency(Consistency::Strong),
        )
        .expect("connect writer");
        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table")
            .append(&build_title_batch(&["one"]))
            .expect("append 1");

        // A separate connection memoizes + warms its own handle for `docs`.
        let reader = connect_with(
            &uri,
            ConnectOptions::new().with_read_consistency(Consistency::Strong),
        )
        .expect("connect reader");
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
            &a.local_handle().options().connection_memory_budget,
            &b.local_handle().options().connection_memory_budget
        ));
        assert!(Arc::ptr_eq(
            &a.local_handle().options().connection_memory_budget,
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
            &reopened.local_handle().options().connection_memory_budget,
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
    fn duplicate_column_names_rejected() {
        // Arrow accepts a schema with two fields named `a`, but a table built
        // from it can never be queried — every query fails on the ambiguous
        // column. `create_table` rejects it up front rather than yielding a
        // table that errors on every read.
        let conn = connect("memory://").expect("connect");
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("a", DataType::LargeUtf8, true),
        ]));
        let err = conn
            .create_table("dupcol", schema, IndexSpec::new())
            .expect_err("a schema with duplicate column names is rejected");
        assert!(matches!(err, InfinoError::Schema(_)), "got {err:?}");
        assert!(err.to_string().contains("duplicate column name: a"));
    }

    /// A single-column schema whose one column `emb` is a
    /// `FixedSizeList<Float32, dim>` — the vector-column shape.
    fn emb_schema(dim: i32) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(
            "emb",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        )]))
    }

    #[test]
    fn vector_index_dim_out_of_range_rejected_at_create() {
        // A vector index's `dim` is validated when the table is created, not
        // deferred to the first append: a `dim` outside the supported range is
        // rejected up front, so a caller can't create a table whose vector
        // column can never be built.
        let conn = connect("memory://").expect("connect");

        // Zero (below the floor) and an oversized dim both fail at create time.
        for dim in [0_i32, 100_000] {
            let spec = IndexSpec::new().vector("emb", dim as usize, Metric::Cosine);
            let err = conn
                .create_table(&format!("v{dim}"), emb_schema(dim), spec)
                .expect_err("an out-of-range vector dim is rejected");
            assert!(matches!(err, InfinoError::Schema(_)), "got {err:?}");
        }
    }

    #[test]
    fn vector_index_dim_within_range_accepted_at_create() {
        // The complement of the rejection test: both inclusive boundaries of
        // the supported range (16 and 4096) are valid dims, so a table with a
        // vector index at either bound is created successfully.
        let conn = connect("memory://").expect("connect");
        for dim in [16_i32, 4096] {
            let spec = IndexSpec::new().vector("emb", dim as usize, Metric::Cosine);
            conn.create_table(&format!("v{dim}"), emb_schema(dim), spec)
                .expect("an in-range vector dim is accepted");
        }
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
                    .bm25_search("title", "fox", TOP_K, Bm25SearchOptions::new(), None)
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
                    .bm25_search("title", "fox", TOP_K, Bm25SearchOptions::new(), None)
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
                    .bm25_search("title", "fox", TOP_K, Bm25SearchOptions::new(), None)
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
    fn duplicate_create_on_storage_leaks_no_subtree() {
        // Top-level physical roots for the table are its unique
        // `<name>-...` location directories under the catalog root.
        fn location_dirs(dir: &Path, prefix: &str) -> usize {
            fs::read_dir(dir)
                .expect("read catalog root")
                .flatten()
                .filter(|e| {
                    e.path().is_dir() && e.file_name().to_string_lossy().starts_with(prefix)
                })
                .count()
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");

        conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("first create");
        assert_eq!(location_dirs(dir.path(), "docs-"), 1);

        let again = conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"));
        assert!(matches!(again, Err(InfinoError::AlreadyExists(_))));
        assert_eq!(
            location_dirs(dir.path(), "docs-"),
            1,
            "a rejected re-create must not leave an orphaned location"
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

    /// A three-value `title` over two superfiles, so a grouped count merges two superfiles'
    /// value counts.
    fn create_cab_types(conn: &Connection, name: &str) {
        let table = conn
            .create_table(name, schema_id_title(), IndexSpec::new())
            .expect("create");
        table
            .append(&build_title_batch(&["yellow", "green", "yellow"]))
            .expect("append 1");
        table
            .append(&build_title_batch(&["fhv", "yellow"]))
            .expect("append 2");
    }

    /// `(group, count)` rows of a grouped count, sorted by group.
    fn grouped_counts(batches: &[RecordBatch]) -> Vec<(String, i64)> {
        let mut rows = Vec::new();
        for batch in batches {
            let groups = batch
                .column(0)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("LargeUtf8 group");
            let counts = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 count");
            for i in 0..batch.num_rows() {
                rows.push((groups.value(i).to_string(), counts.value(i)));
            }
        }
        rows.sort();
        rows
    }

    /// Whether `EXPLAIN` output plans a scan of any table.
    fn scans(explained: &[RecordBatch]) -> bool {
        pretty_format_batches(explained)
            .expect("format plan")
            .to_string()
            .contains("TableScan")
    }

    /// `GROUP BY col, COUNT(*)` and a single-column filtered `COUNT(*)` are answered from the
    /// manifest's value counts through the public API, with no scan.
    #[test]
    fn query_sql_answers_covered_aggregates_from_manifest_stats() {
        let conn = connect("memory://").expect("connect");
        create_cab_types(&conn, "cabs");

        let grouped = "SELECT title, count(*) FROM cabs GROUP BY title";
        assert_eq!(
            grouped_counts(&conn.query_sql(grouped).expect("grouped")),
            vec![
                ("fhv".to_string(), 1),
                ("green".to_string(), 1),
                ("yellow".to_string(), 3)
            ]
        );
        let filtered = "SELECT count(*) FROM cabs WHERE title = 'yellow'";
        let n = conn.query_sql(filtered).expect("filtered")[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 count")
            .value(0);
        assert_eq!(n, 3);

        for sql in [grouped, filtered] {
            let plan = conn.query_sql(&format!("EXPLAIN {sql}")).expect("explain");
            assert!(
                !scans(&plan),
                "answered from manifest stats, no scan: {sql}"
            );
        }
    }

    /// Distinct values one append of the past-the-cap fixture holds: more than a superfile
    /// records counts for.
    const PAST_THE_FILE_CAP: usize = 300;
    /// Distinct values in each of two appends of the past-the-cap fixture: fewer than a
    /// superfile's cap, more than it between them.
    const UNDER_THE_FILE_CAP: usize = 200;
    /// Superfiles that trigger a merge in the value-count fixture's compaction, whatever
    /// their size: as few as merging takes.
    const MERGE_AT_SUPERFILES: u64 = 2;

    /// `(value, count)` pairs of a grouped count, the value as text (`None` for the null
    /// group), sorted.
    fn value_count_rows(batches: &[RecordBatch]) -> Vec<(Option<String>, i64)> {
        let mut rows = Vec::new();
        for batch in batches {
            let counts = batch.column(1).as_primitive::<Int64Type>();
            for i in 0..batch.num_rows() {
                let value = batch
                    .column(0)
                    .is_valid(i)
                    .then(|| array_value_to_string(batch.column(0), i).expect("value as text"));
                rows.push((value, counts.value(i)));
            }
        }
        rows.sort();
        rows
    }

    /// Where `value_counts` must find its answer.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Answered {
        /// From the statistics alone, with no query planned.
        FromStats,
        /// By running the grouped count.
        ByGroupBy,
    }

    /// `table.value_counts(column)` against the `GROUP BY` it stands for: the same rows under
    /// the same names and types, found where `how` says.
    fn assert_value_counts(conn: &Connection, table: &str, column: &str, how: Answered) {
        let handle = conn.open_table(table).expect("open");
        let direct = handle.value_counts(column).expect("value counts");
        let grouped = conn
            .query_sql(&format!(
                "SELECT \"{column}\", count(*) FROM {table} GROUP BY \"{column}\""
            ))
            .expect("grouped");
        assert_eq!(
            value_count_rows(std::slice::from_ref(&direct)),
            value_count_rows(&grouped),
            "{table}.{column}"
        );
        if let Some(first) = grouped.first() {
            assert_eq!(
                direct.schema().fields(),
                first.schema().fields(),
                "{table}.{column}"
            );
        }
        let schema = value_count_schema(direct.schema().field(0)).expect("schema");
        let from_stats = handle
            .local_handle()
            .reader()
            .expect("reader")
            .value_counts_from_stats(column, &schema)
            .expect("from stats");
        assert_eq!(
            from_stats.is_some(),
            how == Answered::FromStats,
            "{table}.{column}"
        );
    }

    /// A one-column table of `field`, holding `values`.
    fn one_column_table(
        conn: &Connection,
        name: &str,
        field: Field,
        values: ArrayRef,
    ) -> Supertable {
        let schema = Arc::new(Schema::new(vec![field]));
        let table = conn
            .create_table(name, Arc::clone(&schema), IndexSpec::new())
            .expect("create");
        table
            .append(&RecordBatch::try_new(schema, vec![values]).expect("batch"))
            .expect("append");
        table
    }

    /// `n` distinct strings, `prefix` before each.
    fn distinct_strings(prefix: &str, n: usize) -> ArrayRef {
        Arc::new(LargeStringArray::from(
            (0..n).map(|i| format!("{prefix}{i}")).collect::<Vec<_>>(),
        ))
    }

    /// A table's value counts are its `GROUP BY` counts, merged across superfiles.
    #[test]
    fn value_counts_match_the_grouped_count() {
        let conn = connect("memory://").expect("connect");
        create_cab_types(&conn, "cabs");
        assert_value_counts(&conn, "cabs", "title", Answered::FromStats);
    }

    /// Rows without a value count under a null key, as `GROUP BY` puts them in a null group.
    #[test]
    fn value_counts_count_missing_values_under_a_null_key() {
        let conn = connect("memory://").expect("connect");
        let tags = LargeStringArray::from(vec![Some("a"), None, Some("a"), None, Some("b")]);
        one_column_table(
            &conn,
            "tags",
            Field::new("tag", DataType::LargeUtf8, true),
            Arc::new(tags),
        );
        assert_value_counts(&conn, "tags", "tag", Answered::FromStats);
    }

    /// A file whose column holds only nulls counts as its nulls, so a column filled in later
    /// is still answered from the statistics.
    #[test]
    fn value_counts_count_an_all_null_file_as_its_nulls() {
        let conn = connect("memory://").expect("connect");
        let table = one_column_table(
            &conn,
            "tags",
            Field::new("tag", DataType::LargeUtf8, true),
            Arc::new(LargeStringArray::from(vec![None::<&str>, None])),
        );
        assert_value_counts(&conn, "tags", "tag", Answered::FromStats);
        let schema = table.schema();
        table
            .append(
                &RecordBatch::try_new(
                    schema,
                    vec![Arc::new(LargeStringArray::from(vec!["x", "y"]))],
                )
                .expect("batch"),
            )
            .expect("append");
        assert_value_counts(&conn, "tags", "tag", Answered::FromStats);
    }

    /// Integer, boolean and decimal columns are counted from the statistics too, each value
    /// under its own type.
    #[test]
    fn value_counts_answer_non_string_columns() {
        let conn = connect("memory://").expect("connect");
        one_column_table(
            &conn,
            "ints",
            Field::new("n", DataType::Int64, true),
            Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(1), None])),
        );
        one_column_table(
            &conn,
            "flags",
            Field::new("on", DataType::Boolean, true),
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(false),
                Some(true),
            ])),
        );
        one_column_table(
            &conn,
            "prices",
            Field::new("price", DataType::Decimal128(10, 2), true),
            Arc::new(
                Decimal128Array::from(vec![Some(150), Some(150), Some(299)])
                    .with_precision_and_scale(10, 2)
                    .expect("decimal"),
            ),
        );
        assert_value_counts(&conn, "ints", "n", Answered::FromStats);
        assert_value_counts(&conn, "flags", "on", Answered::FromStats);
        assert_value_counts(&conn, "prices", "price", Answered::FromStats);
    }

    /// Past the cap of distinct values, in one superfile or across two, the statistics hold no
    /// exact answer and the grouped count runs.
    #[test]
    fn value_counts_past_the_cap_run_the_grouped_count() {
        let conn = connect("memory://").expect("connect");
        let field = Field::new("tag", DataType::LargeUtf8, true);
        one_column_table(
            &conn,
            "wide",
            field.clone(),
            distinct_strings("w", PAST_THE_FILE_CAP),
        );
        assert_value_counts(&conn, "wide", "tag", Answered::ByGroupBy);

        let table = one_column_table(
            &conn,
            "split",
            field,
            distinct_strings("a", UNDER_THE_FILE_CAP),
        );
        assert_value_counts(&conn, "split", "tag", Answered::FromStats);
        table
            .append(
                &RecordBatch::try_new(
                    table.schema(),
                    vec![distinct_strings("b", UNDER_THE_FILE_CAP)],
                )
                .expect("batch"),
            )
            .expect("append");
        assert_value_counts(&conn, "split", "tag", Answered::ByGroupBy);
    }

    /// A column in the middle of a type change has files counted in its old type, so the
    /// grouped count answers, reading every value as the new type.
    #[test]
    fn value_counts_of_a_column_mid_type_change_run_the_grouped_count() {
        let (conn, _dir) = storage_conn();
        let digits = LargeStringArray::from(vec![Some("1"), Some("x"), Some("1")]);
        one_column_table(
            &conn,
            "codes",
            Field::new("code", DataType::LargeUtf8, true),
            Arc::new(digits),
        );
        conn.apply_schema(
            "codes",
            &SchemaPatch::new(vec![FieldPatch::named("code").with_type(DataType::Int64)]),
            None,
        )
        .expect("retype");
        assert_value_counts(&conn, "codes", "code", Answered::ByGroupBy);
    }

    /// The counts follow the table: an append adds its file's counts, and a compaction that
    /// swaps the files for one leaves the same counts, all from the statistics.
    #[test]
    fn value_counts_follow_appends_and_compaction() {
        let (conn, _dir) = storage_conn();
        create_cab_types(&conn, "cabs");
        assert_value_counts(&conn, "cabs", "title", Answered::FromStats);
        let table = conn.open_table("cabs").expect("open");
        table
            .append(&build_title_batch(&["green", "taxi"]))
            .expect("append");
        assert_value_counts(&conn, "cabs", "title", Answered::FromStats);

        let files = |table: &Supertable| {
            table
                .local_handle()
                .reader()
                .expect("reader")
                .n_superfiles()
        };
        let before = files(&table);
        table
            .optimize(&OptimizeOptions::compact(CompactionSettings {
                min_superfiles_for_merge: MERGE_AT_SUPERFILES,
                ..CompactionSettings::default()
            }))
            .expect("optimize");
        let table = conn.open_table("cabs").expect("reopen");
        let after = files(&table);
        assert!(
            after < before,
            "the compaction merged the files: {before} -> {after}"
        );
        assert_value_counts(&conn, "cabs", "title", Answered::FromStats);
    }

    /// A table with no rows has no value and no null: no rows, from the statistics.
    #[test]
    fn value_counts_of_an_empty_table_are_empty() {
        let conn = connect("memory://").expect("connect");
        conn.create_table("empty", schema_id_title(), IndexSpec::new())
            .expect("create");
        assert_value_counts(&conn, "empty", "title", Answered::FromStats);
        let direct = conn
            .open_table("empty")
            .expect("open")
            .value_counts("title")
            .expect("value counts");
        assert_eq!(direct.num_rows(), 0);
    }

    /// A column named `count` is counted like any other, its counts beside it under
    /// `count(*)`; a column named `count(*)` itself is refused, since its counts would take its
    /// name.
    #[test]
    fn value_counts_keep_a_column_named_count_apart_from_its_counts() {
        let conn = connect("memory://").expect("connect");
        one_column_table(
            &conn,
            "tallies",
            Field::new("count", DataType::Int64, true),
            Arc::new(Int64Array::from(vec![7, 7, 9])),
        );
        assert_value_counts(&conn, "tallies", "count", Answered::FromStats);
        let counts = conn
            .open_table("tallies")
            .expect("open")
            .value_counts("count")
            .expect("value counts");
        let names: Vec<&str> = counts
            .schema_ref()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect();
        assert_eq!(names, vec!["count", "count(*)"]);

        one_column_table(
            &conn,
            "odd",
            Field::new("count(*)", DataType::Int64, true),
            Arc::new(Int64Array::from(vec![1])),
        );
        assert!(
            conn.open_table("odd")
                .expect("open")
                .value_counts("count(*)")
                .is_err()
        );
    }

    /// After a delete the per-file counts still hold the deleted row, so the grouped count
    /// answers; an unknown column is an error.
    #[test]
    fn value_counts_run_the_grouped_count_over_deletes_and_refuse_unknown_columns() {
        let (conn, _dir) = storage_conn();
        create_cab_types(&conn, "cabs");
        let table = conn.open_table("cabs").expect("open");
        table.delete(col("title").eq(lit("green"))).expect("delete");
        assert_value_counts(&conn, "cabs", "title", Answered::ByGroupBy);
        assert!(table.value_counts("no_such_column").is_err());
    }

    /// The catalog and a table's reader plan the same aggregate the same way: same rows, and
    /// neither scans. The two build their SQL context separately, and once drifted apart.
    #[test]
    fn query_sql_plans_aggregates_like_the_table_reader() {
        let conn = connect("memory://").expect("connect");
        create_cab_types(&conn, "cabs");
        let reader = conn
            .open_table_handle("cabs")
            .expect("handle")
            .reader()
            .expect("reader");

        let sql = "SELECT title, count(*) FROM {} GROUP BY title";
        let through_catalog = conn.query_sql(&sql.replace("{}", "cabs")).expect("catalog");
        let through_reader = reader
            .query_sql(&sql.replace("{}", TABLE_NAME))
            .expect("reader");
        assert_eq!(
            grouped_counts(&through_catalog),
            grouped_counts(&through_reader)
        );

        let catalog_plan = conn
            .query_sql(&format!("EXPLAIN {}", sql.replace("{}", "cabs")))
            .expect("catalog explain");
        let reader_plan = reader
            .query_sql(&format!("EXPLAIN {}", sql.replace("{}", TABLE_NAME)))
            .expect("reader explain");
        assert!(!scans(&catalog_plan), "catalog plan scans");
        assert!(!scans(&reader_plan), "reader plan scans");
    }

    /// Deleted rows never reach a covered aggregate: a superfile with tombstones is not clean, so
    /// the rewrite scans it and the scan drops the deleted rows.
    #[test]
    fn query_sql_covered_aggregates_see_deletes() {
        // Deletes need durable storage; `memory://` refuses them.
        let dir = tempfile::tempdir().expect("tempdir");
        let conn = connect(dir.path().to_str().expect("utf8 path")).expect("connect");
        create_cab_types(&conn, "cabs");
        conn.open_table("cabs")
            .expect("open")
            .delete(col("title").eq(lit("green")))
            .expect("delete");

        assert_eq!(
            grouped_counts(
                &conn
                    .query_sql("SELECT title, count(*) FROM cabs GROUP BY title")
                    .expect("grouped")
            ),
            vec![("fhv".to_string(), 1), ("yellow".to_string(), 3)],
            "the deleted group is gone"
        );
        let n = conn
            .query_sql("SELECT count(*) FROM cabs WHERE title = 'green'")
            .expect("filtered")[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 count")
            .value(0);
        assert_eq!(n, 0, "deleted rows are not counted");
    }

    /// A fresh connection to a durable table, as a cold worker opens it: a clean table is
    /// answered from stats, and once rows are deleted the counts still leave them out.
    #[test]
    fn query_sql_covered_aggregates_after_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        create_cab_types(&connect(&uri).expect("connect"), "cabs");
        let grouped = "SELECT title, count(*) FROM cabs GROUP BY title";

        let reopened = connect(&uri).expect("reopen");
        let plan = reopened
            .query_sql(&format!("EXPLAIN {grouped}"))
            .expect("explain");
        assert!(
            !scans(&plan),
            "a clean reopened table is answered from stats"
        );

        reopened
            .open_table("cabs")
            .expect("open")
            .delete(col("title").eq(lit("green")))
            .expect("delete");
        let after_delete = connect(&uri).expect("reopen after delete");
        assert_eq!(
            grouped_counts(&after_delete.query_sql(grouped).expect("grouped")),
            vec![("fhv".to_string(), 1), ("yellow".to_string(), 3)]
        );
    }

    /// A nullable `title` and a nullable `fare` over two superfiles: four non-null fares summing
    /// to 20, and the non-null titles yellow twice, green once and fhv once.
    fn create_fares(conn: &Connection, name: &str) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, true),
            Field::new("fare", DataType::Float64, true),
        ]));
        let table = conn
            .create_table(name, Arc::clone(&schema), IndexSpec::new())
            .expect("create");
        let batch = |titles: Vec<Option<&str>>, fares: Vec<Option<f64>>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(LargeStringArray::from(titles)),
                    Arc::new(Float64Array::from(fares)),
                ],
            )
            .expect("batch")
        };
        table
            .append(&batch(
                vec![Some("yellow"), None, Some("green")],
                vec![Some(10.0), Some(2.5), None],
            ))
            .expect("append 1");
        table
            .append(&batch(
                vec![Some("yellow"), Some("fhv"), None],
                vec![None, Some(7.0), Some(0.5)],
            ))
            .expect("append 2");
    }

    /// The single value in column `column` of a one-row result.
    fn single<A: Array + Clone + 'static>(batches: &[RecordBatch], column: usize) -> A {
        batches[0]
            .column(column)
            .as_any()
            .downcast_ref::<A>()
            .expect("column type")
            .clone()
    }

    /// `COUNT(column)` counts the column's non-null rows from the manifest's null counts, alone
    /// and beside `SUM` (the pair an average is assembled from), with no scan; under a range on
    /// the same column it still counts only non-null rows.
    #[test]
    fn query_sql_answers_column_counts_from_manifest_stats() {
        let conn = connect("memory://").expect("connect");
        create_fares(&conn, "fares");

        let count_only = "SELECT count(fare) AS n FROM fares";
        let counted = conn.query_sql(count_only).expect("count");
        assert_eq!(single::<Int64Array>(&counted, 0).value(0), 4);

        let pair = "SELECT sum(fare) AS s, count(fare) AS n FROM fares";
        let summed = conn.query_sql(pair).expect("sum and count");
        assert_eq!(single::<Float64Array>(&summed, 0).value(0), 20.0);
        assert_eq!(single::<Int64Array>(&summed, 1).value(0), 4);

        for sql in [count_only, pair] {
            let plan = conn.query_sql(&format!("EXPLAIN {sql}")).expect("explain");
            assert!(
                !scans(&plan),
                "answered from manifest stats, no scan: {sql}"
            );
        }

        let ranged = conn
            .query_sql("SELECT count(fare) AS n FROM fares WHERE fare >= 0")
            .expect("ranged count");
        assert_eq!(single::<Int64Array>(&ranged, 0).value(0), 4);
    }

    /// A grouped count filtered to the grouping column's non-null values is answered from the
    /// value counts even though the column holds nulls; unfiltered, the null group needs the scan.
    #[test]
    fn query_sql_answers_not_null_grouped_counts_from_value_counts() {
        let conn = connect("memory://").expect("connect");
        create_fares(&conn, "fares");

        let not_null = "SELECT title, count(*) FROM fares WHERE title IS NOT NULL GROUP BY title";
        assert_eq!(
            grouped_counts(&conn.query_sql(not_null).expect("grouped")),
            vec![
                ("fhv".to_string(), 1),
                ("green".to_string(), 1),
                ("yellow".to_string(), 2)
            ]
        );
        let plan = conn
            .query_sql(&format!("EXPLAIN {not_null}"))
            .expect("explain");
        assert!(!scans(&plan), "answered from value counts, no scan");

        let whole = "SELECT title, count(*) FROM fares GROUP BY title";
        let plan = conn
            .query_sql(&format!("EXPLAIN {whole}"))
            .expect("explain");
        assert!(scans(&plan), "the null group is not in the value counts");
    }

    /// `(group, count)` rows of a grouped count whose group column may be null, sorted.
    fn nullable_grouped_counts(batches: &[RecordBatch]) -> Vec<(Option<String>, i64)> {
        let mut rows = Vec::new();
        for batch in batches {
            let groups = batch
                .column(0)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("LargeUtf8 group");
            let counts = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 count");
            for i in 0..batch.num_rows() {
                let group = (!groups.is_null(i)).then(|| groups.value(i).to_string());
                rows.push((group, counts.value(i)));
            }
        }
        rows.sort();
        rows
    }

    /// Only `IS NOT NULL` on the grouping column itself, alone, is answered from its value
    /// counts. A filter on another column keeps the null group the value counts lack, and a
    /// conjunction narrows the rows below them; both are scanned.
    #[test]
    fn query_sql_scans_grouped_counts_under_any_other_filter() {
        let conn = connect("memory://").expect("connect");
        create_fares(&conn, "fares");

        let other_column =
            "SELECT title, count(*) FROM fares WHERE fare IS NOT NULL GROUP BY title";
        assert_eq!(
            nullable_grouped_counts(&conn.query_sql(other_column).expect("grouped")),
            vec![
                (None, 2),
                (Some("fhv".to_string()), 1),
                (Some("yellow".to_string()), 1)
            ]
        );
        let conjunction = "SELECT title, count(*) FROM fares \
                           WHERE title IS NOT NULL AND fare IS NOT NULL GROUP BY title";
        assert_eq!(
            grouped_counts(&conn.query_sql(conjunction).expect("grouped")),
            vec![("fhv".to_string(), 1), ("yellow".to_string(), 1)]
        );
        for sql in [other_column, conjunction] {
            let plan = conn.query_sql(&format!("EXPLAIN {sql}")).expect("explain");
            assert!(scans(&plan), "must scan: {sql}");
        }
    }

    /// The rewrite only answers an aggregate over one table's scan; an aggregate over a join of
    /// two tables still scans both and counts the joined rows.
    #[test]
    fn query_sql_does_not_rewrite_an_aggregate_over_a_join() {
        let conn = connect("memory://").expect("connect");
        create_cab_types(&conn, "cabs");
        create_cab_types(&conn, "fares");

        let sql = "SELECT a.title, count(*) FROM cabs a JOIN fares b ON a.title = b.title \
                   GROUP BY a.title";
        assert_eq!(
            grouped_counts(&conn.query_sql(sql).expect("join")),
            vec![
                ("fhv".to_string(), 1),
                ("green".to_string(), 1),
                ("yellow".to_string(), 9)
            ]
        );
        let plan = conn.query_sql(&format!("EXPLAIN {sql}")).expect("explain");
        assert!(scans(&plan), "a join is planned as scans");
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
    fn query_sql_exact_ilike_over_a_tiny_budget_is_refused_as_over_budget() {
        // `title` is `standard`-analyzed, so `%filler%` is answered from the
        // dictionary, which charges what it holds to the connection budget
        // while the plan is built. A 0-byte gate refuses it as OverBudget.
        let (_dir, conn, _n) = tiny_budget_conn_after_ingest();
        let err = conn
            .query_sql("SELECT title FROM docs WHERE title ILIKE '%filler%'")
            .expect_err("a 0-byte gate refuses the exact path");
        assert!(
            matches!(&err, InfinoError::OverBudget(msg) if msg.contains("exact ILIKE")),
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
    fn query_sql_sort_over_a_tiny_budget_is_refused_as_over_budget() {
        // The sort wraps the pool's refusal in context of its own ("Not enough
        // memory to continue external sort"); the wrapped refusal is still a
        // budget refusal, not a generic query error.
        let (_dir, conn, _n) = tiny_budget_conn_after_ingest();
        let err = conn
            .query_sql("SELECT title FROM docs ORDER BY title")
            .expect_err("a 0-byte gate refuses the sort");
        assert!(
            matches!(err, InfinoError::OverBudget(_)),
            "expected OverBudget, got {err:?}"
        );
    }

    #[test]
    fn a_wrapped_refusal_is_classified_by_its_root() {
        let wrapped = DataFusionError::ResourcesExhausted("over".into()).context("sorting");
        assert!(matches!(
            datafusion_error(&wrapped),
            InfinoError::OverBudget(msg) if msg == "over"
        ));
        // DataFusion's own `Execution` counts as ours until shown otherwise:
        // see `datafusion_error`.
        assert!(matches!(
            datafusion_error(&DataFusionError::Execution("boom".into())),
            InfinoError::Backend(_)
        ));
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

    /// A zero-row result must expose the caller-facing `LargeUtf8`, never the
    /// scan's `Utf8View`.
    ///
    /// The two zero-row tests above miss this. One uses an FTS column (never
    /// viewed), the other a `GROUP BY` (still emits an empty batch to take the
    /// schema from). A view only escapes when the plan emits no batch at all
    /// and the schema has to come from somewhere else, so this covers the
    /// shapes that emit nothing: an unmatched filter, an empty table, `LIMIT
    /// 0`, and an unmatched join.
    #[test]
    fn query_sql_zero_row_string_schema_is_large_utf8_not_view() {
        let conn = connect("memory://").expect("connect");
        // No FTS on `title`, so it is a plain scalar string and gets viewed.
        let docs = conn
            .create_table("docs", schema_id_title(), IndexSpec::new())
            .expect("create docs");
        docs.append(&build_title_batch(&["alpha", "beta"]))
            .expect("append");
        conn.create_table("blank", schema_id_title(), IndexSpec::new())
            .expect("create blank");

        // Ground truth: the same projection over rows that exist.
        let matched = conn
            .query_sql("SELECT title FROM docs WHERE title = 'alpha'")
            .expect("matching query");
        assert_eq!(n_rows(&matched), 1, "exactly one row matches");

        // Collected rather than asserted per shape, so a regression names every
        // shape it broke instead of stopping at the first.
        let mut leaked: Vec<(&str, DataType)> = Vec::new();
        for sql in [
            "SELECT title FROM docs WHERE title = 'no_such_title'",
            "SELECT title FROM docs LIMIT 0",
            "SELECT title FROM blank",
            "SELECT a.title FROM docs a JOIN docs b ON a.title = b.title \
             WHERE a.title = 'no_such_title'",
        ] {
            let empty = conn.query_sql(sql).expect("zero-row query must not error");
            assert!(
                !empty.is_empty(),
                "{sql}: needs one empty batch to carry the schema"
            );
            assert_eq!(n_rows(&empty), 0, "{sql}: must return no rows");
            let ty = empty[0].schema().field(0).data_type().clone();
            if ty != DataType::LargeUtf8 {
                leaked.push((sql, ty));
            }
        }
        assert!(
            leaked.is_empty(),
            "zero-row shapes not LargeUtf8: {leaked:?}"
        );

        // The plain filter shape also matches the with-rows schema exactly,
        // names and nullability included.
        let empty = conn
            .query_sql("SELECT title FROM docs WHERE title = 'no_such_title'")
            .expect("zero-row query");
        assert_eq!(
            empty[0].schema(),
            matched[0].schema(),
            "zero-row and matching results must carry the same schema"
        );
    }

    /// Public path: a non-FTS `LargeUtf8` column is scanned as `Utf8View`, but
    /// `Connection::query_sql` returns it as `LargeUtf8` (the scan view is
    /// coerced at the plan output), so no `Utf8View` leaks to a caller.
    #[test]
    fn query_sql_public_string_result_is_large_utf8_not_view() {
        let conn = connect("memory://").expect("connect");
        // No FTS on `title`, so it is a plain scalar string and gets viewed
        // (the case the view targets).
        let docs = conn
            .create_table("docs", schema_id_title(), IndexSpec::new())
            .expect("create docs");
        docs.append(&build_title_batch(&["alpha", "beta", "alpha"]))
            .expect("append");

        let batches = conn
            .query_sql("SELECT title FROM docs GROUP BY title ORDER BY title")
            .expect("group-by");
        let col = batches[0].column(0);
        assert_eq!(
            col.data_type(),
            &DataType::LargeUtf8,
            "public result must be LargeUtf8, not Utf8View"
        );
        let titles = col
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title downcasts to LargeStringArray");
        let got: Vec<&str> = (0..titles.len()).map(|i| titles.value(i)).collect();
        assert_eq!(got, vec!["alpha", "beta"], "distinct titles, ordered");
        assert!(
            col.as_any().downcast_ref::<StringViewArray>().is_none(),
            "Utf8View must not leak to the caller"
        );
    }

    /// Ungrouped `MIN`/`MAX` over a viewed string column through the public
    /// path (the shape that regressed before `expand_views_at_output`): must
    /// not error and returns `LargeUtf8`.
    #[test]
    fn query_sql_public_ungrouped_min_max_string() {
        let conn = connect("memory://").expect("connect");
        let docs = conn
            .create_table("docs", schema_id_title(), IndexSpec::new())
            .expect("create docs");
        docs.append(&build_title_batch(&["beta", "alpha", "gamma"]))
            .expect("append");

        let batches = conn
            .query_sql("SELECT MIN(title) lo, MAX(title) hi, COUNT(*) n FROM docs")
            .expect("ungrouped min/max over a viewed column");
        let lo = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("MIN(title) is LargeUtf8");
        assert_eq!(lo.value(0), "alpha");
    }

    #[test]
    fn update_storage_credentials_is_false_without_a_gcs_backend() {
        let conn = connect("memory://").expect("connect");
        assert!(
            !conn.update_storage_credentials(&[(
                "google_bearer_token".to_string(),
                "t1".to_string()
            )])
        );
    }

    /// Cross-table join whose key is a viewed string column: the join key comes
    /// back `LargeUtf8`, not a view.
    #[test]
    fn query_sql_join_across_tables_on_string_column() {
        let conn = connect("memory://").expect("connect");
        // No FTS, so `title` is a plain scalar string (viewed) in both tables.
        let a = conn
            .create_table("a", schema_id_title(), IndexSpec::new())
            .expect("create a");
        let b = conn
            .create_table("b", schema_id_title(), IndexSpec::new())
            .expect("create b");
        a.append(&build_title_batch(&["rust", "go"]))
            .expect("append a");
        b.append(&build_title_batch(&["rust", "go"]))
            .expect("append b");

        let batches = conn
            .query_sql("SELECT a.title FROM a JOIN b ON a.title = b.title ORDER BY a.title")
            .expect("cross-table join on a string key");
        let col = batches[0].column(0);
        assert_eq!(
            col.data_type(),
            &DataType::LargeUtf8,
            "joined string key must be LargeUtf8, not a view"
        );
        let t = col
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title is LargeUtf8");
        let got: Vec<&str> = (0..t.len()).map(|i| t.value(i)).collect();
        assert_eq!(got, vec!["go", "rust"]);
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

    /// Embedding dimension of the [`conn_with_vector_table`] fixture.
    const VEC_DIM: usize = 16;
    /// Titles of the [`conn_with_vector_table`] fixture, one row each.
    const VEC_TITLES: [&str; 4] = ["rust async", "python data", "rust systems", "go rust"];

    /// A `memory://` connection holding table `vecs`: one row per
    /// [`VEC_TITLES`] entry, row `i` one-hot at dim `i` (so a one-hot query
    /// at dim 0 is the exact nearest neighbour of row 0), with a full-text
    /// index on `title` and an L2 vector index on `emb`.
    fn conn_with_vector_table() -> Connection {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(Arc::clone(&item), VEC_DIM as i32),
                false,
            ),
        ]));
        let mut flat = Vec::<f32>::with_capacity(VEC_TITLES.len() * VEC_DIM);
        for i in 0..VEC_TITLES.len() {
            for d in 0..VEC_DIM {
                flat.push(if d == i { 1.0 } else { 0.0 });
            }
        }
        let list = FixedSizeListArray::new(
            item,
            VEC_DIM as i32,
            Arc::new(Float32Array::from(flat)),
            None,
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(LargeStringArray::from(VEC_TITLES.to_vec())),
                Arc::new(list),
            ],
        )
        .expect("vector batch");

        let conn = connect("memory://").expect("connect");
        conn.create_table(
            "vecs",
            schema,
            IndexSpec::new()
                .fts("title")
                .vector("emb", VEC_DIM, Metric::L2Sq),
        )
        .expect("create table")
        .append(&batch)
        .expect("append");
        conn
    }

    /// The [`VEC_DIM`]-wide one-hot vector at `dim`, as the comma-separated
    /// literal the vector TVFs take.
    fn one_hot_csv(dim: usize) -> String {
        (0..VEC_DIM)
            .map(|d| if d == dim { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Two calls of one search table function in a statement are two
    /// searches. DataFusion names every table-function scan after the
    /// function alone and compares scans without their provider, where
    /// the arguments live, so the optimizer used to merge calls that
    /// differed only in arguments: each pair below answered the first
    /// call's count in both columns — across tables as well as terms,
    /// and for every search function.
    #[test]
    fn search_tvf_calls_differing_in_arguments_stay_distinct() {
        let conn = conn_with_vector_table();
        conn.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create docs")
            .append(&build_title_batch(&["rust", "zig"]))
            .expect("append");
        let v = one_hot_csv(0);
        // (what, first call, second call, first count, second count)
        let cases: Vec<(&str, String, String, i64, i64)> = vec![
            (
                "token_match, different terms",
                "token_match('vecs', 'title', 'rust')".into(),
                "token_match('vecs', 'title', 'python')".into(),
                3,
                1,
            ),
            (
                "token_match, different tables",
                "token_match('vecs', 'title', 'rust')".into(),
                "token_match('docs', 'title', 'rust')".into(),
                3,
                1,
            ),
            (
                "exact_match",
                "exact_match('vecs', 'title', 'rust async')".into(),
                "exact_match('vecs', 'title', 'no such title')".into(),
                1,
                0,
            ),
            (
                "bm25_search, different terms",
                "bm25_search('vecs', 'title', 'rust', 10)".into(),
                "bm25_search('vecs', 'title', 'python', 10)".into(),
                3,
                1,
            ),
            (
                "bm25_search, different k",
                "bm25_search('vecs', 'title', 'rust', 1)".into(),
                "bm25_search('vecs', 'title', 'rust', 10)".into(),
                1,
                3,
            ),
            (
                "bm25_search_prefix",
                "bm25_search_prefix('vecs', 'title', 'rus', 10)".into(),
                "bm25_search_prefix('vecs', 'title', 'pyt', 10)".into(),
                3,
                1,
            ),
            (
                "vector_search, different k",
                format!("vector_search('vecs', 'emb', '{v}', 1)"),
                format!("vector_search('vecs', 'emb', '{v}', 3)"),
                1,
                3,
            ),
            (
                "hybrid_search, different k",
                format!("hybrid_search('vecs', 'title', 'rust', 'emb', '{v}', 1)"),
                format!("hybrid_search('vecs', 'title', 'rust', 'emb', '{v}', 3)"),
                1,
                3,
            ),
        ];
        for (what, first, second, want_first, want_second) in cases {
            let batches = conn
                .query_sql(&format!(
                    "SELECT (SELECT count(*) FROM {first}) AS a, \
                            (SELECT count(*) FROM {second}) AS b"
                ))
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let count = |i: usize| {
                batches[0]
                    .column(i)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("count(*) is Int64")
                    .value(0)
            };
            assert_eq!(
                (count(0), count(1)),
                (want_first, want_second),
                "{what}: each call answers its own count"
            );
        }
    }

    /// The remaining catalog-level search TVFs — `bm25_search_prefix`,
    /// `vector_search`, and `hybrid_search` — resolve their leading
    /// table-name argument and forward the rest to the table's search
    /// kernels. Exercises each `*CatalogFunc::call` over a table that
    /// carries both an FTS index and a vector index.
    #[test]
    fn query_sql_prefix_vector_and_hybrid_tvfs_resolve_table() {
        /// Top-k requested by the vector / hybrid queries.
        const TOP_K: usize = 4;

        let conn = conn_with_vector_table();
        let one_hot_0 = one_hot_csv(0);

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
            .bm25_search("title", "fox", TOP_K, Bm25SearchOptions::new(), None)
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
    fn connection_read_consistency_flows_to_table_handles() {
        let dir = std::env::temp_dir().join(format!("infino-consistency-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("mkdir");
        let uri = format!("file://{}", dir.display());

        // Default connection → BoundedStaleness(1s), applied to both a created
        // handle and a freshly opened one.
        let db = connect_with(&uri, ConnectOptions::new()).expect("connect default");
        let created = db
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create");
        assert_eq!(
            created.local_handle().options().read_consistency,
            Consistency::BoundedStaleness(Duration::from_secs(1)),
            "an unset connection defaults to BoundedStaleness(1s)"
        );
        assert_eq!(
            db.open_table("docs")
                .expect("open")
                .local_handle()
                .options()
                .read_consistency,
            Consistency::BoundedStaleness(Duration::from_secs(1)),
            "open_table applies the same policy as create_table"
        );

        // An explicit policy on the connection flows through unchanged.
        let dir2 = std::env::temp_dir().join(format!(
            "infino-consistency-strong-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir2).expect("mkdir");
        let uri2 = format!("file://{}", dir2.display());
        let strong = connect_with(
            &uri2,
            ConnectOptions::new().with_read_consistency(Consistency::Strong),
        )
        .expect("connect strong");
        let created = strong
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create");
        assert_eq!(
            created.local_handle().options().read_consistency,
            Consistency::Strong,
            "with_read_consistency(Strong) reaches the table handle"
        );

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&dir2);
    }

    #[test]
    fn bounded_staleness_default_is_stale_within_window_then_converges() {
        // The default connection is BoundedStaleness(1s): a peer connection's
        // commit is not visible within the staleness window, and becomes visible
        // once it elapses. (Strong would show it immediately; that path is
        // covered by `storage_memoized_handle_sees_another_connections_commit`.)
        let dir = std::env::temp_dir().join(format!("infino-bs-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("mkdir");
        let uri = format!("file://{}", dir.display());

        let writer = connect(&uri).expect("connect writer");
        writer
            .create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create")
            .append(&build_title_batch(&["fox"]))
            .expect("append v1");

        // A separate default (BoundedStaleness) connection. Its first read sees
        // the first commit and stamps the per-window pointer check.
        let reader = connect(&uri).expect("connect reader");
        let hits = |r: &Connection| {
            n_rows(
                &r.open_table("docs")
                    .expect("open")
                    .bm25_search("title", "fox", TOP_K, Bm25SearchOptions::new(), None)
                    .expect("search"),
            )
        };
        assert_eq!(hits(&reader), 1, "reader sees the first commit");

        // The writer commits a second matching doc.
        writer
            .open_table("docs")
            .expect("open")
            .append(&build_title_batch(&["fox"]))
            .expect("append v2");

        // Within the 1s window the reader still serves its pinned snapshot.
        assert_eq!(
            hits(&reader),
            1,
            "within the bounded-staleness window the peer's commit is not yet visible"
        );

        // After the window elapses, the next read re-probes and converges.
        thread::sleep(Duration::from_millis(1_100));
        assert_eq!(
            hits(&reader),
            2,
            "after the 1s window the reader picks up the peer's commit"
        );

        let _ = fs::remove_dir_all(&dir);
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

    /// A connection with one populated table, `docs` (`_id`, `title`). Writes in the gate tests
    /// target a real table so the refusal comes from the gate, not from name resolution.
    fn conn_with_docs() -> Connection {
        let conn = connect("memory://").expect("connect");
        let docs = conn
            .create_table("docs", schema_id_title(), IndexSpec::new())
            .expect("create docs");
        docs.append(&build_title_batch(&["one row"]))
            .expect("append docs");
        conn
    }

    /// `docs` is still the only table and still has its one row: the refused writes changed nothing.
    fn assert_docs_intact(conn: &Connection) {
        assert_eq!(conn.list_tables().expect("list"), vec!["docs".to_owned()]);
        let rows = conn
            .query_sql("SELECT title FROM docs")
            .expect("docs intact");
        assert_eq!(n_rows(&rows), 1);
    }

    /// `sql` was refused by the read-only gate itself, not by a later planning or execution error.
    fn assert_refused_as_write(conn: &Connection, sql: &str) {
        let err = conn.query_sql(sql);
        assert!(
            matches!(&err, Err(InfinoError::Query(msg)) if msg.contains("read-only")),
            "expected read-only refusal for {sql:?}, got {err:?}",
        );
    }

    #[test]
    fn query_sql_refuses_every_plannable_write() {
        // Every write the planner can plan, against a live table.
        //  - DDL, DML, COPY and session statements each plan to a side-effecting node.
        //  - `SELECT INTO` plans to CREATE TABLE; a comment or an EXPLAIN in front of an INSERT
        //    leaves the same DML node underneath.
        //  - all are refused before they run: by the statement check, or for `SELECT INTO`,
        //    which reads like a query, by the plan check after planning.
        // Afterwards the catalog and the table are exactly as created.
        let conn = conn_with_docs();
        for sql in [
            "CREATE TABLE evil (x int)",
            "CREATE EXTERNAL TABLE t STORED AS PARQUET LOCATION 'x'",
            "CREATE VIEW v AS SELECT title FROM docs",
            "DROP TABLE docs",
            "SELECT * INTO newt FROM docs",
            "COPY docs TO 'x.csv'",
            "INSERT INTO docs VALUES (1, 'x')",
            "UPDATE docs SET title = 'x'",
            "DELETE FROM docs",
            "SET datafusion.execution.batch_size = 1",
            "RESET datafusion.execution.batch_size",
            "START TRANSACTION",
            "/* c */ INSERT INTO docs VALUES (1, 'x')",
            "INSERT/**/INTO docs VALUES (1, 'x')",
            "-- c\nINSERT INTO docs VALUES (1, 'x')",
            "EXPLAIN INSERT INTO docs VALUES (1, 'x')",
            "EXPLAIN SELECT * INTO newt FROM docs",
        ] {
            assert_refused_as_write(&conn, sql);
        }
        assert_docs_intact(&conn);
        assert!(
            conn.query_sql("SELECT * FROM evil").is_err(),
            "CREATE TABLE must not have run"
        );
        assert!(
            conn.query_sql("SELECT * FROM newt").is_err(),
            "SELECT INTO must not have run"
        );
    }

    /// A filter that fails on the table's own values is the caller's mistake.
    /// The filter runs in the `FilterExec` above the scan, so the failure is
    /// Arrow's cast error itself; the message check pins that, so a DataFusion
    /// upgrade that rewords it fails here instead of the error silently turning
    /// into an engine fault.
    #[test]
    fn query_sql_reports_a_filter_that_fails_on_the_data_as_the_callers() {
        let conn = conn_with_docs();
        let err = conn.query_sql("SELECT title FROM docs WHERE CAST(title AS BIGINT) = 1");
        assert!(
            matches!(&err, Err(InfinoError::Query(msg)) if msg.contains("Cannot cast string")),
            "got {err:?}"
        );
        // A regex the caller wrote that does not parse: DataFusion returns the
        // regex crate's error, and it is still the caller's.
        let err = conn.query_sql("SELECT title FROM docs WHERE title ~ '('");
        assert!(matches!(err, Err(InfinoError::Query(_))), "got {err:?}");
    }

    /// A valid read the engine does not implement is neither the caller's
    /// mistake nor an engine fault: `Unsupported`, so a client can tell "rewrite
    /// this" from "this query is wrong". A write is never this: it is refused as
    /// one before planning, whether or not DataFusion could plan it.
    #[test]
    fn query_sql_reports_an_unimplemented_read_as_unsupported() {
        let conn = conn_with_docs();
        for sql in [
            "SELECT title FROM docs ORDER BY title FETCH FIRST 1 ROWS WITH TIES",
            // `SHOW` forms DataFusion does not plan: reads, so unsupported, not refused.
            "SHOW SCHEMAS",
            "SHOW DATABASES",
            "SHOW VIEWS",
        ] {
            let err = conn.query_sql(sql);
            assert!(
                matches!(&err, Err(InfinoError::Unsupported(msg)) if msg.contains("not implemented")),
                "{sql:?}: got {err:?}"
            );
        }
    }

    #[test]
    fn query_sql_reports_a_function_argument_rejected_at_planning_as_the_callers() {
        // DataFusion rejects a bad function argument with an `Execution` error
        // while it plans, before anything is read: the caller's mistake.
        let conn = conn_with_docs();
        let err = conn.query_sql("SELECT arrow_cast(title, 'NotAType') FROM docs");
        assert!(matches!(err, Err(InfinoError::Query(_))), "got {err:?}");
    }

    #[test]
    fn query_sql_refuses_a_write_from_its_statement_before_planning() {
        // Writes DataFusion cannot plan, or plans only to refuse: the statement
        // check refuses each before planning, with the same answer as a
        // plannable write, so none reads as SQL a later version might run.
        let conn = conn_with_docs();
        for sql in [
            "ALTER TABLE docs ADD COLUMN y int",
            "TRUNCATE TABLE docs",
            "WITH t AS (SELECT 'x' AS title) INSERT INTO docs (title) SELECT title FROM t",
            "(INSERT INTO docs VALUES (1, 'x'))",
            "WITH t AS (INSERT INTO docs VALUES (1, 'x') RETURNING title) SELECT * FROM t",
            "EXPLAIN WITH t AS (SELECT 'x' AS title) INSERT INTO docs (title) SELECT title FROM t",
        ] {
            assert_refused_as_write(&conn, sql);
        }
        assert_docs_intact(&conn);
    }

    #[test]
    fn query_sql_runs_exactly_one_statement() {
        // A second statement, or none, is the caller's mistake, however
        // harmless each one is; a trailing semicolon is still one statement.
        let conn = conn_with_docs();
        for sql in ["SELECT 1; DROP TABLE docs", "SELECT 1; SELECT 2", ""] {
            let err = conn.query_sql(sql);
            assert!(
                matches!(&err, Err(InfinoError::Query(msg)) if msg.contains("exactly one SQL statement")),
                "{sql:?}: got {err:?}"
            );
        }
        conn.query_sql("SELECT 1;")
            .expect("one statement with a trailing semicolon");
        assert_docs_intact(&conn);
    }

    #[test]
    fn query_sql_allows_read_only_shapes() {
        // Reads in every position the gate inspects: table and table-free reads, CTEs, set
        // operations, subqueries, EXPLAIN. None is refused.
        let conn = conn_with_docs();
        for sql in [
            "SELECT title FROM docs",
            "SELECT COUNT(*) FROM docs",
            "SELECT 1 AS one",
            "VALUES (1), (2)",
            "(SELECT 1)",
            "WITH t AS (SELECT title FROM docs) SELECT title FROM t",
            "SELECT title FROM docs UNION ALL SELECT title FROM docs",
            "SELECT 1 INTERSECT SELECT 1",
            "SELECT 1 EXCEPT SELECT 2",
            "SELECT (SELECT COUNT(*) FROM docs) AS scalar",
            "SELECT * FROM (SELECT title FROM docs) AS sub",
            "EXPLAIN SELECT title FROM docs",
            "WITH ranked AS (SELECT title, ROW_NUMBER() OVER (ORDER BY title) AS rn, COUNT(*) OVER () AS total FROM docs), top AS (SELECT title, rn FROM ranked WHERE rn <= 10 OR total < 100) SELECT title FROM top WHERE rn > 0 AND title <> '' ORDER BY rn",
            "SELECT title, SUM(CHAR_LENGTH(title)) OVER (ORDER BY title ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS w FROM docs ORDER BY title LIMIT 5",
            "DESCRIBE docs",
            // Parsed as sqlparser's own `EXPLAIN`, not DataFusion's.
            "DESCRIBE SELECT title FROM docs",
            "EXPLAIN ANALYZE SELECT title FROM docs",
        ] {
            conn.query_sql(sql)
                .unwrap_or_else(|e| panic!("{sql:?} should be allowed: {e}"));
        }
        // `SHOW` reads too. `query_sql` does not enable `information_schema`,
        // so these fail, but as reads: never refused as a write.
        for sql in [
            "SHOW TABLES",
            "SHOW COLUMNS FROM docs",
            "SHOW CREATE TABLE docs",
            "SHOW FUNCTIONS",
            "SHOW datafusion.execution.batch_size",
        ] {
            let err = conn.query_sql(sql);
            assert!(
                matches!(&err, Err(InfinoError::Query(msg)) if !msg.contains("read-only")),
                "{sql:?}: got {err:?}"
            );
        }
    }

    /// Rows in the first append of the ordering test: under one default
    /// DataFusion batch (8192), so the superfile is smaller than the next.
    const ORDER_FIRST_APPEND_ROWS: usize = 3_000;
    /// Rows in the second append: past one batch, so the two superfiles
    /// together carry enough rows that the planner repartitions above the sort.
    const ORDER_SECOND_APPEND_ROWS: usize = 9_000;
    /// Times the ordered query runs: the bad order depends on which output
    /// partition finishes first, so one run can come back sorted by chance.
    const ORDER_QUERY_RUNS: usize = 8;

    /// Every `_id` in `batches`' single string column, parsed back to its number.
    fn ids_from_strings(batches: &[RecordBatch]) -> Vec<u128> {
        let mut ids = Vec::new();
        for batch in batches {
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("SQL strings come back as LargeUtf8");
            for i in 0..col.len() {
                ids.push(col.value(i).parse().expect("_id renders as an integer"));
            }
        }
        ids
    }

    #[test]
    fn query_sql_order_by_holds_across_superfiles() {
        // Two appends, two superfiles; the second holds the larger ids. The final
        // cast to `LargeUtf8` makes the planner split the sorted stream across
        // partitions above the sort, and collecting those partitions must not
        // undo the order. Every run must return all ids ascending.
        let conn = connect("memory://").expect("connect");
        let t = conn
            .create_table("t", schema_id_title(), IndexSpec::new())
            .expect("create t");
        for rows in [ORDER_FIRST_APPEND_ROWS, ORDER_SECOND_APPEND_ROWS] {
            let titles: Vec<String> = (0..rows).map(|i| format!("row {i}")).collect();
            let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
            t.append(&build_title_batch(&refs)).expect("append");
        }
        let sql = "SELECT CAST(_id AS VARCHAR) AS v FROM t ORDER BY _id";

        // The plan shape, which does not depend on timing: no repartition may
        // sit between the root and the merge that produces the sorted stream.
        let explain = conn.query_sql(&format!("EXPLAIN {sql}")).expect("explain");
        let text = pretty_format_batches(&explain)
            .expect("format explain")
            .to_string();
        let physical = &text[text.find("physical_plan").expect("physical plan row")..];
        let merge_at = physical
            .find("SortPreservingMergeExec")
            .expect("the sorted result comes from a merge");
        assert!(
            !physical[..merge_at].contains("RepartitionExec"),
            "a repartition splits the sorted result:\n{text}"
        );

        let total = ORDER_FIRST_APPEND_ROWS + ORDER_SECOND_APPEND_ROWS;
        for run in 0..ORDER_QUERY_RUNS {
            let batches = conn.query_sql(sql).expect("ordered select");
            let ids = ids_from_strings(&batches);
            assert_eq!(ids.len(), total, "run {run}: row count");
            if let Some(at) = ids.windows(2).position(|w| w[0] >= w[1]) {
                panic!(
                    "run {run}: ids not ascending at row {}: {} then {}",
                    at + 1,
                    ids[at],
                    ids[at + 1]
                );
            }
        }
    }

    /// Exhaustive over `LogicalPlan`, so a DataFusion upgrade that adds a variant fails to compile
    /// here. On that failure:
    ///  - classify the new variant below;
    ///  - if it acts on data, schema or session state, confirm `SQLOptions::verify_plan` refuses it;
    ///  - if upstream does not, `read_only_sql_options` needs its own check for that node.
    fn plan_has_side_effect(plan: &LogicalPlan) -> bool {
        match plan {
            LogicalPlan::Ddl(_)
            | LogicalPlan::Dml(_)
            | LogicalPlan::Copy(_)
            | LogicalPlan::Statement(_) => true,
            LogicalPlan::Projection(_)
            | LogicalPlan::Filter(_)
            | LogicalPlan::Window(_)
            | LogicalPlan::Aggregate(_)
            | LogicalPlan::Sort(_)
            | LogicalPlan::Join(_)
            | LogicalPlan::Repartition(_)
            | LogicalPlan::Union(_)
            | LogicalPlan::TableScan(_)
            | LogicalPlan::EmptyRelation(_)
            | LogicalPlan::Subquery(_)
            | LogicalPlan::SubqueryAlias(_)
            | LogicalPlan::Limit(_)
            | LogicalPlan::Values(_)
            | LogicalPlan::Explain(_)
            | LogicalPlan::Analyze(_)
            | LogicalPlan::Extension(_)
            | LogicalPlan::Distinct(_)
            | LogicalPlan::DescribeTable(_)
            | LogicalPlan::Unnest(_)
            | LogicalPlan::RecursiveQuery(_) => false,
        }
    }

    #[test]
    fn read_only_gate_matches_the_side_effecting_plan_nodes() {
        // One plan per class on a bare DataFusion context; the policy is about plan shape, not the
        // provider under the scan.
        //  - a read and an EXPLAIN pass;
        //  - a DDL, a DML, a COPY and a session statement are refused;
        //  - `plan_has_side_effect` agrees with `verify_plan` on every one.
        let options = read_only_sql_options();
        let ctx = SessionContext::new();
        ctx.register_batch("docs", build_title_batch(&["one row"]))
            .expect("register docs");
        for (sql, side_effect) in [
            ("SELECT title FROM docs", false),
            ("EXPLAIN SELECT title FROM docs", false),
            ("CREATE TABLE t (x int)", true),
            ("INSERT INTO docs VALUES ('x')", true),
            ("COPY docs TO 'x.csv'", true),
            ("SET datafusion.execution.batch_size = 1", true),
        ] {
            let plan = bridge_sync_to_async(ctx.state().create_logical_plan(sql)).expect("plans");
            assert_eq!(plan_has_side_effect(&plan), side_effect, "{sql}");
            assert_eq!(options.verify_plan(&plan).is_err(), side_effect, "{sql}");
        }
    }

    /// `SELECT ... WHERE _id=0 OR _id=1 OR ...`: `terms` equality terms, `terms - 1` connectives.
    fn or_chain(terms: usize) -> String {
        let clause: Vec<String> = (0..terms).map(|i| format!("_id={i}")).collect();
        format!("SELECT title FROM docs WHERE {}", clause.join(" OR "))
    }

    #[test]
    fn query_sql_refuses_a_boolean_chain_over_the_connective_cap() {
        // Twice the cap; without the cap this aborts the test binary, not the assertion.
        let conn = conn_with_docs();
        let err = conn.query_sql(&or_chain(2 * MAX_PREDICATE_CONNECTIVES));
        assert!(
            matches!(&err, Err(InfinoError::Query(msg)) if msg.contains("connectives")),
            "got {err:?}"
        );
    }

    #[test]
    fn query_sql_allows_a_boolean_chain_at_the_connective_cap() {
        // Exactly the cap plans and runs.
        let conn = conn_with_docs();
        conn.query_sql(&or_chain(MAX_PREDICATE_CONNECTIVES + 1))
            .expect("at-cap chain plans");
    }

    #[test]
    fn query_sql_in_list_is_not_capped() {
        // IN is one node however long; ten times the cap plans fine.
        let conn = conn_with_docs();
        let values: Vec<String> = (0..10 * MAX_PREDICATE_CONNECTIVES)
            .map(|i| i.to_string())
            .collect();
        let sql = format!(
            "SELECT title FROM docs WHERE _id IN ({})",
            values.join(", ")
        );
        conn.query_sql(&sql).expect("long IN list plans");
    }

    #[test]
    fn query_sql_connectives_inside_a_string_literal_do_not_count() {
        // Past the length pre-filter, but every OR is inside one literal: none count.
        let conn = conn_with_docs();
        let literal = "x OR ".repeat(MAX_PREDICATE_CONNECTIVES);
        let sql = format!("SELECT title FROM docs WHERE title = '{literal}'");
        conn.query_sql(&sql)
            .expect("literal ORs are not connectives");
    }

    #[test]
    fn query_sql_compares_a_full_width_id_literal_exactly() {
        // A row's `_id` is wider than 64 bits, so DataFusion would read it unquoted as a float and
        // fail the comparison on a Decimal128 overflow. Equality and IN both find the row.
        let conn = conn_with_docs();
        let ids = conn.query_sql("SELECT _id FROM docs").expect("read _id");
        let id = ids[0]
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id is Decimal128")
            .value(0)
            .to_string();
        assert!(
            id.parse::<u64>().is_err(),
            "_id {id} must be past u64 to exercise the fix"
        );
        for sql in [
            format!("SELECT title FROM docs WHERE _id = {id}"),
            format!("SELECT title FROM docs WHERE _id IN ({id}, 1)"),
        ] {
            let rows = conn
                .query_sql(&sql)
                .unwrap_or_else(|e| panic!("{sql}: {e:?}"));
            assert_eq!(n_rows(&rows), 1, "{sql}");
        }
    }

    #[test]
    fn delete_allows_a_predicate_at_the_connective_cap() {
        // Exactly the cap resolves (zero rows match). Storage-backed table: mutations refuse
        // memory:// before the predicate matters.
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = dir.path().to_str().expect("utf8 path").to_string();
        let conn = connect(&uri).expect("connect");
        let docs = conn
            .create_table("docs", schema_id_title(), IndexSpec::new())
            .expect("create docs");
        docs.append(&build_title_batch(&["one row"]))
            .expect("append docs");
        let predicate = (0..MAX_PREDICATE_CONNECTIVES + 1)
            .map(|i| col("title").eq(lit(format!("t{i}"))))
            .reduce(|acc, term| acc.or(term))
            .expect("nonempty chain");
        docs.delete(predicate).expect("at-cap predicate resolves");
    }

    proptest! {
        // Stage two skips the lexer on this inequality: the byte-scan bound never undercounts
        // the true keyword count, on any input.
        #[test]
        fn connective_upper_bound_never_undercounts(sql in ".{0,4096}") {
            if let Some(exact) = connective_count(&sql) {
                prop_assert!(connective_upper_bound(&sql) >= exact);
            }
        }
    }

    #[test]
    fn prefilter_floor_holds_for_the_densest_chain() {
        // Densest lexable chain, `(1)OR(1)...`, spends 5 bytes per connective, so text under the
        // pre-filter length tops out near 614 connectives: under the cap, stage one is sound.
        let limit = MIN_BYTES_PER_CONNECTIVE * MAX_PREDICATE_CONNECTIVES;
        let mut sql = String::from("(1");
        while sql.len() + 6 < limit {
            sql.push_str(")OR(1");
        }
        sql.push(')');
        assert!(sql.len() < limit);
        let count = connective_count(&sql).expect("chain tokenizes");
        assert!(
            count <= MAX_PREDICATE_CONNECTIVES,
            "{count} connectives fit under the pre-filter length"
        );
    }

    #[test]
    fn update_refuses_a_predicate_over_the_connective_cap() {
        // Same gate as delete; refusal comes before storage or batch shape is looked at.
        let conn = conn_with_docs();
        let docs = conn.open_table("docs").expect("open docs");
        let predicate = (0..MAX_PREDICATE_CONNECTIVES + 2)
            .map(|i| col("title").eq(lit(format!("t{i}"))))
            .reduce(|acc, term| acc.or(term))
            .expect("nonempty chain");
        let err = docs.update(predicate, &build_title_batch(&["replacement"]));
        assert!(
            matches!(&err, Err(InfinoError::Query(msg)) if msg.contains("connectives")),
            "got {err:?}"
        );
    }

    #[test]
    fn delete_refuses_a_predicate_over_the_connective_cap() {
        // One connective past the cap, refused before any id capture.
        let conn = conn_with_docs();
        let docs = conn.open_table("docs").expect("open docs");
        let predicate = (0..MAX_PREDICATE_CONNECTIVES + 2)
            .map(|i| col("title").eq(lit(format!("t{i}"))))
            .reduce(|acc, term| acc.or(term))
            .expect("nonempty chain");
        let err = docs.delete(predicate);
        assert!(
            matches!(&err, Err(InfinoError::Query(msg)) if msg.contains("connectives")),
            "got {err:?}"
        );
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
                        .vector("embedding", 16, Metric::L2Sq),
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
            .bm25_search("title", "fox", TOP_K, Bm25SearchOptions::new(), None)
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
            .bm25_search("title", "-onlyneg", TOP_K, Bm25SearchOptions::new(), None)
            .expect_err("negation-only query");
        assert!(matches!(err, InfinoError::Query(_)));
        assert!(err.to_string().contains("bm25_search:"), "got: {err}");

        let err = conn.query_sql("NOT VALID SQL @@@").expect_err("bad sql");
        assert!(matches!(err, InfinoError::Query(_)));
        assert!(err.to_string().contains("query_sql:"), "got: {err}");
    }

    #[test]
    fn usage_meters_are_isolated_across_connections() {
        let a_dir = tempfile::tempdir().expect("a");
        let b_dir = tempfile::tempdir().expect("b");
        let a = connect(a_dir.path().to_str().expect("utf8")).expect("connect a");
        let b = connect(b_dir.path().to_str().expect("utf8")).expect("connect b");
        let before_a = a.usage_snapshot();
        let before_b = b.usage_snapshot();
        a.create_table("docs", schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create a")
            .append(&build_title_batch(&["alpha"]))
            .expect("append a");
        let delta_a = a.usage_snapshot().since(&before_a);
        let delta_b = b.usage_snapshot().since(&before_b);
        assert!(
            delta_a.put_count > 0 || delta_a.get_count > 0 || delta_a.list_count > 0,
            "connection A must record I/O: {delta_a:?}"
        );
        assert!(
            delta_b.is_zero(),
            "connection B must stay quiet while A writes: {delta_b:?}"
        );
    }

    // ---- cancelling a query ----

    /// Superfiles in the cancel fixture: one per append, so a query over it
    /// reads this many files, each with its own reads.
    const CANCEL_FIXTURE_FILES: usize = 48;
    /// How long each read of the slowed store takes.
    const SLOW_READ: Duration = Duration::from_millis(40);
    /// How soon a cancelled query must return. The call does not wait for
    /// the work it drops, so this is generous even for a slow runner.
    const CANCEL_RETURN_BOUND: Duration = Duration::from_secs(2);
    /// How long a test waits for something it expects to happen: a query to
    /// start reading, an uncancelled one to finish, the cache to go quiet.
    /// Generous, since a slow runner pays every read's delay in turn.
    const QUERY_DEADLINE: Duration = Duration::from_secs(30);
    /// How long the test watches for reads once the query has returned:
    /// many slow reads' worth, so work still running would show up.
    const AFTER_CANCEL_WINDOW: Duration = Duration::from_millis(600);
    /// A read only planning makes here: a superfile's tombstone sidecar.
    const SIDECAR: &str = ".tombstones";
    /// A read only execution makes here: a superfile's data. Planning opens
    /// the files from the bytes the manifest carries.
    const SUPERFILE: &str = ".sf.parquet";
    /// Poll cadence while a test waits on a count.
    const CANCEL_POLL: Duration = Duration::from_millis(5);
    /// Files and rows per file in the execution-phase fixture: enough rows
    /// that each file's data lies past what the manifest carries, so the
    /// query reads it, and sorts it, while it executes.
    const LARGE_FILES: usize = 8;
    const ROWS_PER_LARGE_FILE: usize = 5_000;
    /// An odd multiplier that scatters consecutive row numbers across the
    /// 128-bit range, so the titles' hex compresses poorly.
    const SCATTER: u128 = 0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c835;
    /// A query that reads every file: no statistic answers it.
    const READ_EVERY_FILE: &str = "SELECT count(title) FROM docs WHERE title LIKE '%fox%'";
    /// The same rows sorted, so execution holds memory while it reads.
    const SORT_EVERY_FILE: &str = "SELECT title FROM docs WHERE title LIKE '%fox%' ORDER BY title";

    /// The slowed stores a connection built, one per provider in the order it
    /// built them: the catalog root, then each table as it was opened.
    #[derive(Default)]
    struct SlowStores {
        stores: Mutex<Vec<Arc<SlowStorage>>>,
        /// Whether stores built from now on start with their reads held.
        hold_new: AtomicBool,
    }

    impl SlowStores {
        fn wrap(&self, inner: Arc<dyn StorageProvider>) -> Arc<dyn StorageProvider> {
            let slow = SlowStorage::wrap(inner, SLOW_READ);
            if self.hold_new.load(Ordering::SeqCst) {
                slow.hold_reads();
            }
            self.lock().push(Arc::clone(&slow));
            slow
        }

        fn lock(&self) -> MutexGuard<'_, Vec<Arc<SlowStorage>>> {
            self.stores.lock().unwrap_or_else(PoisonError::into_inner)
        }

        /// Hold the reads of every store built from now on, until
        /// [`Self::release_reads`].
        fn hold_new_stores(&self) {
            self.hold_new.store(true, Ordering::SeqCst);
        }

        /// Hold the reads of every store built so far, until
        /// [`Self::release_reads`].
        fn hold_reads(&self) {
            self.lock().iter().for_each(|s| s.hold_reads());
        }

        fn release_reads(&self) {
            self.hold_new.store(false, Ordering::SeqCst);
            self.lock().iter().for_each(|s| s.release_reads());
        }

        /// Providers built so far.
        fn built(&self) -> usize {
            self.lock().len()
        }

        fn reads(&self) -> usize {
            self.lock().iter().map(|s| s.reads()).sum()
        }

        fn reads_of(&self, fragment: &str) -> usize {
            self.lock().iter().map(|s| s.reads_of(fragment)).sum()
        }

        /// Reads started on the `index`th provider built.
        fn reads_on(&self, index: usize) -> usize {
            self.lock().get(index).map_or(0, |s| s.reads())
        }
    }

    /// Poll until `done` holds, or fail naming `what`.
    fn wait_until(what: &str, done: impl Fn() -> bool) {
        let deadline = Instant::now() + QUERY_DEADLINE;
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting until {what}");
            thread::sleep(CANCEL_POLL);
        }
    }

    /// A table named `name` at `conn` holding `files` superfiles, each with
    /// `rows` live titles with "fox" in them and one deleted row. The deletes
    /// give every superfile a tombstone sidecar, which planning fetches from
    /// storage. Each title ends in a hex string that compresses poorly, so
    /// enough rows put a file's data past what the manifest carries for it.
    fn write_fox_table(conn: &Connection, name: &str, files: usize, rows: usize) {
        let table = conn
            .create_table(name, schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create");
        for i in 0..files {
            let mut titles: Vec<String> = (0..rows)
                .map(|j| {
                    let n = (i * rows + j) as u128;
                    format!("fox {i} {:032x}", n.wrapping_mul(SCATTER))
                })
                .collect();
            titles.push(format!("gone {i}"));
            let titles: Vec<&str> = titles.iter().map(String::as_str).collect();
            table.append(&build_title_batch(&titles)).expect("append");
        }
        let deleted = table
            .delete(col("title").like(lit("gone%")))
            .expect("delete");
        assert_eq!(deleted.n_tombstoned(), files);
    }

    /// A catalog at a fresh directory holding the `docs` fox table.
    fn many_file_table() -> tempfile::TempDir {
        let (conn, dir) = storage_conn();
        write_fox_table(&conn, "docs", CANCEL_FIXTURE_FILES, 1);
        dir
    }

    /// A connection to `dir` with an empty disk cache (so every file is cold)
    /// whose reads are slowed, and the slowed stores.
    fn slow_cold_conn(dir: &Path, cache: &Path) -> (Connection, Arc<SlowStores>) {
        slow_cold_conn_with(dir, cache, ConnectOptions::new())
    }

    /// [`slow_cold_conn`] with `options` for everything but the cache and the
    /// slowed stores.
    fn slow_cold_conn_with(
        dir: &Path,
        cache: &Path,
        options: ConnectOptions,
    ) -> (Connection, Arc<SlowStores>) {
        let stores = Arc::new(SlowStores::default());
        let wrapping = Arc::clone(&stores);
        let options = options
            .with_cache_dir(cache)
            .with_storage_wrapper(move |inner| wrapping.wrap(inner));
        let conn = connect_with(dir.to_str().expect("utf8 path"), options).expect("connect");
        (conn, stores)
    }

    /// Run `sql` on its own thread under `cancel`, and hand back the result
    /// with the time the call returned.
    fn query_in_background(
        conn: &Connection,
        sql: &str,
        cancel: &QueryCancel,
    ) -> mpsc::Receiver<(Result<Vec<RecordBatch>, InfinoError>, Instant)> {
        let (sent, received) = mpsc::channel();
        let conn = conn.clone();
        let cancel = cancel.clone();
        let sql = sql.to_owned();
        thread::spawn(move || {
            let result = conn.query_sql_cancellable(&sql, &cancel);
            let _ = sent.send((result, Instant::now()));
        });
        received
    }

    /// Cancel the query `running` and check that the call returned
    /// `Cancelled` within [`CANCEL_RETURN_BOUND`].
    fn cancel_and_expect_cancelled(
        cancel: &QueryCancel,
        running: &mpsc::Receiver<(Result<Vec<RecordBatch>, InfinoError>, Instant)>,
    ) {
        let cancelled_at = Instant::now();
        cancel.cancel();
        let (result, returned_at) = running
            .recv_timeout(CANCEL_RETURN_BOUND)
            .expect("the cancelled query returns");
        assert!(
            matches!(result, Err(InfinoError::Cancelled(_))),
            "got {result:?}"
        );
        assert!(returned_at - cancelled_at < CANCEL_RETURN_BOUND);
    }

    /// No foreground read lands for [`AFTER_CANCEL_WINDOW`]: what the
    /// cancelled query dropped stopped, rather than reading on detached.
    fn assert_no_foreground_read_follows(conn: &Connection) {
        let at_return = conn.usage_snapshot().get_count;
        thread::sleep(AFTER_CANCEL_WINDOW);
        assert_eq!(
            conn.usage_snapshot().get_count,
            at_return,
            "a foreground read landed after the cancelled query returned"
        );
    }

    /// Wait until no read starts for [`AFTER_CANCEL_WINDOW`].
    fn wait_until_quiet(stores: &SlowStores) {
        let deadline = Instant::now() + QUERY_DEADLINE;
        let mut reads = stores.reads();
        loop {
            thread::sleep(AFTER_CANCEL_WINDOW);
            let now = stores.reads();
            if now == reads {
                return;
            }
            assert!(Instant::now() < deadline, "reads never went quiet");
            reads = now;
        }
    }

    /// The disk cache's ledger once nothing is reading any more: every charged
    /// byte is held by something the cache holds.
    fn assert_cache_ledger_settles(conn: &Connection, stores: &SlowStores) {
        wait_until_quiet(stores);
        let table = conn.open_table_handle("docs").expect("open");
        let cache = table.options().disk_cache.clone().expect("disk cache");
        table.block_on_query(cache.block_writes_settled());
        cache.assert_budget_consistent();
    }

    fn fox_count(batches: &[RecordBatch]) -> i64 {
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count yields Int64")
            .value(0)
    }

    /// A query cancelled while planning reads a large table's files stops
    /// reading them: it returns `Cancelled` at once, and no foreground read
    /// lands after it returned. What it reserved in the disk cache comes
    /// back, and the same query then answers in full.
    #[test]
    fn a_query_cancelled_while_planning_stops_reading() {
        let dir = many_file_table();
        let cache = tempfile::tempdir().expect("cache dir");
        let (conn, stores) = slow_cold_conn(dir.path(), cache.path());
        let cancel = QueryCancel::new();

        let running = query_in_background(&conn, READ_EVERY_FILE, &cancel);
        wait_until("planning reads a sidecar", || stores.reads_of(SIDECAR) > 0);
        cancel_and_expect_cancelled(&cancel, &running);
        assert_no_foreground_read_follows(&conn);

        assert_cache_ledger_settles(&conn, &stores);
        let rows = conn
            .query_sql(READ_EVERY_FILE)
            .expect("query after the cancel");
        assert_eq!(fox_count(&rows), CANCEL_FIXTURE_FILES as i64);
        assert_cache_ledger_settles(&conn, &stores);
    }

    /// A query cancelled while it executes, reading the files' data and
    /// holding sort memory, returns at once and gives the memory back.
    ///
    /// Whether the reads stop is the planning test's to show: here every
    /// file's reads start together (one partition each, with read-ahead), so
    /// they are all in flight before the cancel. Execution's partition tasks,
    /// and the read-ahead downloads they own, are DataFusion tasks that are
    /// aborted when the query is dropped.
    #[test]
    fn a_query_cancelled_while_executing_returns_and_frees_its_memory() {
        let (conn, dir) = storage_conn();
        write_fox_table(&conn, "docs", LARGE_FILES, ROWS_PER_LARGE_FILE);
        let cache = tempfile::tempdir().expect("cache dir");
        let (conn, stores) = slow_cold_conn(dir.path(), cache.path());
        let budget = Arc::clone(&conn.inner.connection_memory_budget);
        let cancel = QueryCancel::new();

        let running = query_in_background(&conn, SORT_EVERY_FILE, &cancel);
        wait_until("execution reads data and holds sort memory", || {
            stores.reads_of(SUPERFILE) > 0 && budget.used() > 0
        });
        cancel_and_expect_cancelled(&cancel, &running);
        wait_until("the sort memory comes back", || budget.used() == 0);
        assert_cache_ledger_settles(&conn, &stores);
    }

    /// Cancelling the query that started reading the files does not disturb
    /// another that joined it on the same files: the other answers in full,
    /// and the cache's ledger balances once both are done.
    #[test]
    fn cancelling_one_query_leaves_a_concurrent_one_whole() {
        let dir = many_file_table();
        let cache = tempfile::tempdir().expect("cache dir");
        let (conn, stores) = slow_cold_conn(dir.path(), cache.path());
        let cancel = QueryCancel::new();

        let cancelled = query_in_background(&conn, READ_EVERY_FILE, &cancel);
        wait_until("the first query reads a sidecar", || {
            stores.reads_of(SIDECAR) > 0
        });
        let sidecar_reads = stores.reads_of(SIDECAR);
        let whole = query_in_background(&conn, READ_EVERY_FILE, &QueryCancel::new());
        wait_until("the second query reads too", || {
            stores.reads_of(SIDECAR) > sidecar_reads
        });
        cancel_and_expect_cancelled(&cancel, &cancelled);

        let (result, _) = whole
            .recv_timeout(QUERY_DEADLINE)
            .expect("the other query returns");
        assert_eq!(
            fox_count(&result.expect("the other query")),
            CANCEL_FIXTURE_FILES as i64
        );
        assert_cache_ledger_settles(&conn, &stores);
    }

    /// A handle that has already fired stops the query before it reads
    /// anything, not even the catalog.
    #[test]
    fn a_cancelled_handle_reads_nothing() {
        let (conn, dir) = storage_conn();
        write_fox_table(&conn, "docs", 1, 1);
        let cache = tempfile::tempdir().expect("cache dir");
        let (conn, stores) = slow_cold_conn(dir.path(), cache.path());
        let cancel = QueryCancel::new();
        cancel.cancel();

        let result = conn.query_sql_cancellable(READ_EVERY_FILE, &cancel);
        assert!(
            matches!(result, Err(InfinoError::Cancelled(_))),
            "got {result:?}"
        );
        assert_eq!(stores.reads(), 0, "the cancelled query read nothing");
    }

    /// A cancel that fires while the query opens the first table it names
    /// returns at once, with that table's manifest still loading. The open
    /// finishes on its own, so the next query finds the table built, and the
    /// second table is never opened.
    #[test]
    fn a_query_cancelled_while_opening_a_table_returns_and_opens_no_more() {
        let (conn, dir) = storage_conn();
        write_fox_table(&conn, "docs", 1, 1);
        write_fox_table(&conn, "more", 1, 1);
        let cache = tempfile::tempdir().expect("cache dir");
        let (conn, stores) = slow_cold_conn(dir.path(), cache.path());
        let cancel = QueryCancel::new();

        // Providers are built in order: the catalog root, then `docs` when
        // the query opens it. The open of `docs` is held until the cancelled
        // query has returned.
        stores.hold_new_stores();
        let running = query_in_background(&conn, "SELECT count(*) FROM docs, more", &cancel);
        wait_until("the query opens docs", || stores.reads_on(1) > 0);
        cancel_and_expect_cancelled(&cancel, &running);

        stores.release_reads();
        wait_until_quiet(&stores);
        assert_eq!(stores.built(), 2, "more was never opened");
        let rows = conn
            .query_sql("SELECT count(*) FROM docs")
            .expect("query after the cancel");
        assert_eq!(fox_count(&rows), 1);
        assert_eq!(stores.built(), 2, "the cancelled open built docs");
    }

    /// The bounded-staleness window the freshness tests run under: short, so
    /// a test can wait it out.
    const STALENESS_WINDOW: Duration = Duration::from_millis(50);

    /// Under `consistency`, a cancel that fires while the query checks an
    /// open table for a newer manifest returns at once, with that pointer
    /// read still under way, and the table answers in full afterwards.
    fn cancel_while_checking_for_a_newer_manifest(consistency: Consistency) {
        let (conn, dir) = storage_conn();
        write_fox_table(&conn, "docs", 1, 1);
        let cache = tempfile::tempdir().expect("cache dir");
        let options = ConnectOptions::new().with_read_consistency(consistency);
        let (conn, stores) = slow_cold_conn_with(dir.path(), cache.path(), options);
        conn.query_sql("SELECT count(*) FROM docs")
            .expect("opens docs");
        let cancel = QueryCancel::new();

        stores.hold_reads();
        // Past the window, the next query is due its freshness check.
        thread::sleep(STALENESS_WINDOW);
        let pointer_reads = stores.reads_of(POINTER_PATH);
        let running = query_in_background(&conn, "SELECT count(*) FROM docs", &cancel);
        wait_until("the query reads the pointer", || {
            stores.reads_of(POINTER_PATH) > pointer_reads
        });
        cancel_and_expect_cancelled(&cancel, &running);

        stores.release_reads();
        let rows = conn
            .query_sql("SELECT count(*) FROM docs")
            .expect("query after the cancel");
        assert_eq!(fox_count(&rows), 1);
    }

    /// Strong reads the pointer before every query.
    #[test]
    fn a_strong_query_cancelled_while_checking_for_a_newer_manifest_returns() {
        cancel_while_checking_for_a_newer_manifest(Consistency::Strong);
    }

    /// Bounded staleness reads it once the window has passed.
    #[test]
    fn a_bounded_query_cancelled_while_checking_for_a_newer_manifest_returns() {
        cancel_while_checking_for_a_newer_manifest(Consistency::BoundedStaleness(STALENESS_WINDOW));
    }

    /// A pointer read that fails while a Strong query checks for a newer
    /// manifest fails the query with that storage error, rather than serving
    /// the snapshot it has or reporting the table as missing.
    #[test]
    fn a_failed_freshness_check_fails_a_strong_query() {
        let (conn, dir) = storage_conn();
        write_fox_table(&conn, "docs", 1, 1);
        let faults = Arc::new(Mutex::new(Vec::<Arc<FaultStorage>>::new()));
        let wrapping = Arc::clone(&faults);
        let options = ConnectOptions::new()
            .with_read_consistency(Consistency::Strong)
            .with_storage_wrapper(move |inner| {
                let faulty = FaultStorage::wrap(inner);
                wrapping
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(Arc::clone(&faulty));
                faulty
            });
        let conn = connect_with(dir.path().to_str().expect("utf8 path"), options).expect("connect");
        conn.query_sql("SELECT count(*) FROM docs")
            .expect("opens docs");

        faults
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .for_each(|f| f.fail(FaultOp::Get, POINTER_PATH, usize::MAX));
        let failed = conn
            .query_sql("SELECT count(*) FROM docs")
            .expect_err("the pointer read fails");
        assert!(matches!(failed, InfinoError::Io(_)), "got {failed:?}");
    }

    /// Rows the CPU-bound cancel test counts: many seconds of work in a debug
    /// build if left alone, and still finite if a regression lets it run.
    const COUNTED_ROWS: u64 = 10_000_000_000;
    /// How long that query runs before the test cancels it.
    const EXECUTE_BEFORE_CANCEL: Duration = Duration::from_millis(200);

    /// A query with no read to wait on returns promptly when cancelled too: a
    /// count over generated rows returns `Cancelled` soon after the cancel
    /// instead of counting on. The call returns whichever phase the cancel
    /// lands in, planning on a slow runner or, usually, execution.
    #[test]
    fn a_query_cancelled_while_computing_stops() {
        let conn = connect("memory://").expect("connect");
        let cancel = QueryCancel::new();
        let sql = format!("SELECT count(*) FROM range({COUNTED_ROWS})");
        let running = query_in_background(&conn, &sql, &cancel);
        thread::sleep(EXECUTE_BEFORE_CANCEL);
        cancel_and_expect_cancelled(&cancel, &running);
    }
}
