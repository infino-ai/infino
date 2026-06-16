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

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::record_batch::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::execution::context::SessionContext;

pub use index_spec::IndexSpec;
pub use options::{ColdFetchMode, ConnectOptions};

use crate::InfinoError;
use crate::runtime_bridge::bridge_sync_to_async;
use crate::storage::StorageProvider;
use crate::superfile::builder::FtsConfig;
use crate::superfile::fts::tokenize::{AsciiLowerTokenizer, Tokenizer};
use crate::superfile::vector::builder::VectorConfig;
use crate::superfile::vector::distance::Metric;
use crate::supertable::Supertable;
use crate::supertable::options::SupertableOptions;
use crate::supertable::reader_cache::{DiskCacheConfig, DiskCacheStore};
use manifest::{
    TableEntry, VectorEntry, commit_catalog, read_catalog, schema_from_ipc, schema_to_ipc,
};
use uri::{Backend, parse_uri};

/// Open (or create) a catalog rooted at `uri`.
///
/// The storage backend is derived from the URI scheme: a bare path or
/// `file://` → local filesystem, `s3://bucket/prefix` → S3,
/// `az://container/prefix` → Azure, `memory://` → in-process
/// (non-persistent). Equivalent to
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
            CatalogStore::Storage(root)
        }
    };
    Ok(Connection {
        inner: Arc::new(ConnectionInner {
            backend,
            options,
            store,
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
}

/// Where the `name → table` map lives. Durable backends persist it on the
/// root storage under optimistic concurrency; `memory://` keeps it (and
/// the tables themselves) in-process.
enum CatalogStore {
    Memory(Mutex<HashMap<String, Supertable>>),
    Storage(Arc<dyn StorageProvider>),
}

impl Connection {
    /// Create a new table named `name` with the given Arrow `schema` and
    /// search `indexes`. Fails with [`InfinoError::AlreadyExists`] if a
    /// table of that name already exists. Returns the open handle.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_schema::{DataType, Field, Schema};
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
        validate_name(name)?;
        let (fts_cfg, vec_cfg) = indexes.to_configs();
        let tokenizer = table_tokenizer(&indexes);

        match &self.inner.store {
            CatalogStore::Memory(map) => {
                let opts = build_options(schema, fts_cfg, vec_cfg, tokenizer, None)?;
                let handle = Supertable::create(opts)?;
                let mut map = map.lock().expect("catalog mutex poisoned");
                if map.contains_key(name) {
                    return Err(InfinoError::AlreadyExists(name.to_string()));
                }
                map.insert(name.to_string(), handle.clone());
                Ok(handle)
            }
            CatalogStore::Storage(root) => {
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
                    schema_ipc: schema_to_ipc(&schema)?,
                    fts: indexes.fts_columns().to_vec(),
                    vectors,
                    created_at_unix: now_unix(),
                };

                let table_storage =
                    backend_to_provider(&self.inner.backend.join(&location), &self.inner.options)?
                        .expect("non-memory backend yields a storage provider");
                // Disk cache is keyed on the stable name (not the unique
                // location) so the producer and a later reopener share one
                // cache directory; superfile keys carry the location, so a
                // re-created table never reads a dropped generation's bytes.
                let disk_cache = build_disk_cache(&self.inner.options, &table_storage, name)?;
                let mut opts =
                    build_options(schema, fts_cfg, vec_cfg, tokenizer, Some(table_storage))?;
                if let Some(cache) = disk_cache {
                    opts = opts.with_disk_cache(cache);
                }
                // Create the physical table at its unique location, then
                // register the name. A losing racer that also created a
                // (distinct) location just orphans its empty subtree; the
                // catalog OCC below decides the single name winner.
                let handle = Supertable::create(opts)?;

                let name = name.to_string();
                bridge_sync_to_async(commit_catalog(root.as_ref(), move |body| {
                    if body.tables.contains_key(&name) {
                        return Err(InfinoError::AlreadyExists(name.clone()));
                    }
                    body.tables.insert(name.clone(), entry.clone());
                    Ok(())
                }))?;
                Ok(handle)
            }
        }
    }

    /// Open an existing table by name. Fails with
    /// [`InfinoError::NotFound`] if no such table is registered.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, IndexSpec};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # db.create_table("posts", schema, IndexSpec::new().fts("body"))?;
    /// let posts = db.open_table("posts")?;
    /// # let _ = posts;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn open_table(&self, name: &str) -> Result<Supertable, InfinoError> {
        match &self.inner.store {
            CatalogStore::Memory(map) => map
                .lock()
                .expect("catalog mutex poisoned")
                .get(name)
                .cloned()
                .ok_or_else(|| InfinoError::NotFound(name.to_string())),
            CatalogStore::Storage(root) => {
                let (body, _etag) = bridge_sync_to_async(read_catalog(root.as_ref()))?;
                let entry = body
                    .tables
                    .get(name)
                    .ok_or_else(|| InfinoError::NotFound(name.to_string()))?;

                let schema = schema_from_ipc(&entry.schema_ipc)?;
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
                        metric_from_str(&v.metric)?,
                    );
                }
                let (fts_cfg, vec_cfg) = spec.to_configs();
                let tokenizer = table_tokenizer(&spec);

                let table_storage = backend_to_provider(
                    &self.inner.backend.join(&entry.location),
                    &self.inner.options,
                )?
                .expect("non-memory backend yields a storage provider");
                // Cache directory is keyed on the stable name, matching
                // `create_table` (the on-storage subtree is `entry.location`).
                let disk_cache = build_disk_cache(&self.inner.options, &table_storage, name)?;
                let mut opts =
                    build_options(schema, fts_cfg, vec_cfg, tokenizer, Some(table_storage))?;
                if let Some(cache) = disk_cache {
                    opts = opts.with_disk_cache(cache);
                }
                Ok(Supertable::open(opts)?)
            }
        }
    }

    /// Remove a table from the catalog. Fails with
    /// [`InfinoError::NotFound`] if it isn't registered.
    ///
    /// Unregistering is always logical and O(1): the `name → location`
    /// entry leaves the catalog, and readers pinned to a pre-drop
    /// snapshot keep working. `purge` additionally deletes the table's
    /// storage subtree (its unique per-creation location) after the
    /// catalog commit — the name is gone first, so a crash mid-purge
    /// can only leave unreferenced orphans, never a half-deleted live
    /// table. For `memory://`, tables live in-process and free with the
    /// last handle, so `purge` has nothing extra to do.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, IndexSpec};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # db.create_table("posts", schema, IndexSpec::new().fts("body"))?;
    /// db.drop_table("posts", true)?; // purge: reclaim the bytes too
    /// assert!(db.list_tables()?.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn drop_table(&self, name: &str, purge: bool) -> Result<(), InfinoError> {
        match &self.inner.store {
            CatalogStore::Memory(map) => map
                .lock()
                .expect("catalog mutex poisoned")
                .remove(name)
                .map(|_| ())
                .ok_or_else(|| InfinoError::NotFound(name.to_string())),
            CatalogStore::Storage(root) => {
                // Capture the removed entry's location out of the OCC
                // closure; on a retry the freshest body is re-read, so
                // the last successful attempt's location wins.
                let mut location: Option<String> = None;
                bridge_sync_to_async(commit_catalog(root.as_ref(), |body| {
                    match body.tables.remove(name) {
                        Some(entry) => {
                            location = Some(entry.location);
                            Ok(())
                        }
                        None => Err(InfinoError::NotFound(name.to_string())),
                    }
                }))?;
                if purge {
                    let location =
                        location.expect("catalog commit succeeded => an entry was removed");
                    // Delete everything under the table's unique
                    // location. Listing is component-aware, so a sibling
                    // location sharing a string prefix never matches;
                    // deletes are idempotent, so re-running after a
                    // partial failure converges.
                    bridge_sync_to_async(async {
                        let objects = root.list_with_prefix(&location).await?;
                        futures::future::try_join_all(objects.iter().map(|uri| root.delete(uri)))
                            .await?;
                        Ok::<(), crate::storage::StorageError>(())
                    })?;
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
            CatalogStore::Storage(root) => {
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
    /// # use arrow_array::{LargeStringArray, RecordBatch};
    /// # use arrow_schema::{DataType, Field, Schema};
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
        let ctx = SessionContext::new();

        // Resolve the relations the query names and register each that is a
        // catalog table. Unknown names (CTEs, search TVFs, aliases) are
        // skipped — the planner resolves those by other means or errors.
        let statement = ctx
            .state()
            .sql_to_statement(sql, &datafusion::config::Dialect::Generic)
            .map_err(|e| InfinoError::Query(e.to_string()))?;
        let refs = ctx
            .state()
            .resolve_table_references(&statement)
            .map_err(|e| InfinoError::Query(e.to_string()))?;

        let mut seen = HashSet::new();
        let mut handles: Vec<Supertable> = Vec::new();
        for r in &refs {
            let name = r.table().to_string();
            if !seen.insert(name.clone()) {
                continue;
            }
            match self.open_table(&name) {
                Ok(table) => {
                    table
                        .register_into(&ctx, &name)
                        .map_err(|e| InfinoError::Query(e.to_string()))?;
                    handles.push(table);
                }
                Err(InfinoError::NotFound(_)) => {}
                Err(e) => return Err(e),
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
                .map_err(|e| InfinoError::Query(e.to_string()))?;
            df.collect()
                .await
                .map_err(|e| InfinoError::Query(e.to_string()))
        };
        // Drive on an established per-supertable multi-thread query runtime
        // (any referenced table's `block_on_query`); fall back to the shared
        // bridge for table-free queries such as `SELECT 1`.
        match handles.first() {
            Some(table) => table.block_on_query(drive),
            None => bridge_sync_to_async(drive),
        }
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
) -> Result<SupertableOptions, InfinoError> {
    let mut opts = SupertableOptions::new(schema, fts, vectors, tokenizer)?;
    if let Some(s) = storage {
        opts = opts.with_storage(s);
    }
    Ok(opts)
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
    use crate::storage::{AzureStorageProvider, LocalFsStorageProvider, S3StorageProvider};

    let provider: Option<Arc<dyn StorageProvider>> = match backend {
        Backend::Memory => None,
        Backend::LocalFs { root } => Some(Arc::new(LocalFsStorageProvider::new(root.clone())?)),
        Backend::S3 { bucket, prefix } => {
            let p = match options.s3.as_ref() {
                Some(s3) => S3StorageProvider::new_with_endpoint_and_prefix(
                    &s3.endpoint,
                    bucket,
                    &s3.access_key,
                    &s3.secret_key,
                    &s3.region,
                    prefix,
                )?,
                None => S3StorageProvider::new_with_prefix(bucket, prefix)?,
            };
            Some(Arc::new(p))
        }
        Backend::Azure { container, prefix } => Some(Arc::new(
            AzureStorageProvider::new_with_prefix(container, prefix)?,
        )),
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
    let cache = DiskCacheStore::new_unpinned(Arc::clone(storage), cfg)
        .map_err(|e| InfinoError::Io(e.to_string()))?;
    Ok(Some(cache))
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
    use super::*;
    use crate::BoolMode;
    use crate::test_helpers::{build_title_batch, schema_id_title};

    const TOP_K: usize = 10;

    /// Total rows across the materialized search batches.
    fn n_rows(batches: &[arrow::record_batch::RecordBatch]) -> usize {
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
    fn drop_with_purge_reclaims_the_storage_subtree() {
        /// Count regular files under `dir` whose path contains a
        /// component starting with `prefix` (the table's unique
        /// `<name>-<nanos>-<seq>` location).
        fn files_under_location(dir: &std::path::Path, prefix: &str) -> usize {
            let mut n = 0;
            let mut stack = vec![dir.to_path_buf()];
            while let Some(d) = stack.pop() {
                let Ok(entries) = std::fs::read_dir(&d) else {
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
}
