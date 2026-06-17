// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Node.js bindings for infino (napi-rs).
//!
//! Mirrors the Rust catalog API: `infino.connect(uri)` →
//! `db.createTable(...)` / `db.openTable(...)` / `db.querySql(...)`, and
//! `table.append(...)` / `table.bm25Search(...)` /
//! `table.vectorSearch(...)`. Built standalone — it consumes the core
//! crate's curated public API only (no `test-helpers`), so it is also a
//! public-surface consumer test.
//!
//! ## Sync for v1
//!
//! The surface is synchronous, matching the Rust and Python bindings. A
//! sync native call blocks the libuv thread it runs on (the event loop);
//! a long-running Node server doing S3-backed retrieval should run calls
//! in a `worker_thread`. Async (Promise-returning) methods are an
//! additive follow-up, not v1.
//!
//! ## Arrow interchange
//!
//! Arrow is the logical interchange, but unlike the Python bindings —
//! which get zero-copy pyarrow↔arrow-rs via the Arrow C Data Interface —
//! JS↔Rust has no such free bridge, so bulk data crosses as **Arrow IPC
//! bytes** (a `Buffer`): JS serializes with `tableToIPC`, Rust reads with
//! a `StreamReader` (and the reverse out). Search results come back as
//! plain JS objects `{ id, score }`; query-vector arrays cross as
//! `Float32Array` (by reference, no copy).

use std::io::Cursor;
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow::error::ArrowError;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use datafusion::common::DFSchema;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::Expr;
use infino::{
    BoolMode, ColdFetchMode, CompactionError, CompactionSettings, InfinoError, Metric,
    VectorSearchOptions,
};

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

/// Map a core [`InfinoError`] to a JS error, mirroring the Python
/// bindings' grouping: not-found vs. bad-argument vs. runtime. The bucket
/// is encoded in the napi [`Status`] (surfaced as `err.code` in JS) and
/// the message is preserved.
//
// TODO: refine into distinct JS `Error` subclasses once the surface
// settles; the three-bucket split is the same contract as Python's
// KeyError / ValueError / RuntimeError mapping.
fn map_err(e: InfinoError) -> Error {
    match e {
        InfinoError::NotFound(m) => Error::new(Status::GenericFailure, format!("NotFound: {m}")),
        InfinoError::AlreadyExists(m)
        | InfinoError::Schema(m)
        | InfinoError::Cardinality(m)
        | InfinoError::Query(m) => Error::new(Status::InvalidArg, m),
        InfinoError::Io(m) | InfinoError::Backend(m) => Error::new(Status::GenericFailure, m),
        // `InfinoError` is `#[non_exhaustive]`: future variants fall back
        // to a generic runtime error carrying the message.
        other => Error::new(Status::GenericFailure, other.to_string()),
    }
}

fn arrow_err(e: ArrowError) -> Error {
    Error::new(Status::GenericFailure, e.to_string())
}

/// Map a [`CompactionError`] (a distinct error type from `InfinoError`) to a
/// JS error. `NoStorage` is a bad-argument case (compaction needs durable
/// storage); everything else is a runtime failure carrying the message.
fn compact_err(e: CompactionError) -> Error {
    match e {
        CompactionError::NoStorage => Error::new(
            Status::InvalidArg,
            "compact requires durable storage (not memory://)",
        ),
        other => Error::new(Status::GenericFailure, other.to_string()),
    }
}

/// Parse a metric name (`"cosine"` / `"l2sq"` / `"negdot"`).
fn metric_from_str(s: &str) -> Result<Metric> {
    match s.to_ascii_lowercase().as_str() {
        "cosine" => Ok(Metric::Cosine),
        "l2sq" | "l2" => Ok(Metric::L2Sq),
        "negdot" | "dot" => Ok(Metric::NegDot),
        other => Err(Error::new(
            Status::InvalidArg,
            format!("unknown metric {other:?}; use 'cosine', 'l2sq', or 'negdot'"),
        )),
    }
}

// ---------------------------------------------------------------------------
// Arrow IPC helpers (the JS↔Rust transport)
// ---------------------------------------------------------------------------

/// Read the schema carried by an Arrow IPC stream (JS sends an empty
/// table built with the schema; we only need its schema).
fn read_schema_ipc(bytes: &[u8]) -> Result<Schema> {
    let reader = StreamReader::try_new(Cursor::new(bytes), None).map_err(arrow_err)?;
    Ok(reader.schema().as_ref().clone())
}

/// Read all record batches from an Arrow IPC stream.
fn read_batches_ipc(bytes: &[u8]) -> Result<Vec<RecordBatch>> {
    let reader = StreamReader::try_new(Cursor::new(bytes), None).map_err(arrow_err)?;
    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch.map_err(arrow_err)?);
    }
    Ok(batches)
}

/// Serialize batches to an Arrow IPC stream the JS side reads with
/// `tableFromIPC`. With no batches the stream still carries `schema`, so
/// `Table.schema()` round-trips an empty table with the right schema.
fn write_batches_ipc(schema: &Schema, batches: &[RecordBatch]) -> Result<Buffer> {
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, schema).map_err(arrow_err)?;
        for batch in batches {
            writer.write(batch).map_err(arrow_err)?;
        }
        writer.finish().map_err(arrow_err)?;
    }
    Ok(Buffer::from(buf))
}

/// Serialize a query/search result (`Vec<RecordBatch>`) to an Arrow IPC
/// `Buffer`. Schema comes from the first batch, or an empty schema for an
/// empty result. Shared by `query_sql` and the row-returning searches.
fn batches_to_ipc(batches: &[RecordBatch]) -> Result<Buffer> {
    let schema = match batches.first() {
        Some(batch) => batch.schema(),
        None => Arc::new(Schema::empty()),
    };
    write_batches_ipc(schema.as_ref(), batches)
}

/// Parse a cold-fetch-mode string into a [`ColdFetchMode`]. Short aliases
/// (`"hybrid"` / `"range"` / `"lazy"`) are accepted alongside the full names.
fn cold_fetch_from_str(s: &str) -> Result<ColdFetchMode> {
    match s.to_ascii_lowercase().as_str() {
        "hybrid_with_prefetch" | "hybrid" => Ok(ColdFetchMode::HybridWithPrefetch),
        "range_only" | "range" => Ok(ColdFetchMode::RangeOnly),
        "lazy_foreground_with_background_fill" | "lazy" => {
            Ok(ColdFetchMode::LazyForegroundWithBackgroundFill)
        }
        other => Err(Error::new(
            Status::InvalidArg,
            format!(
                "unknown coldFetchMode {other:?}; use 'hybrid_with_prefetch', 'range_only', \
                 or 'lazy_foreground_with_background_fill'"
            ),
        )),
    }
}

/// Parse a boolean-mode string (`"or"` default, or `"and"`).
fn parse_mode(mode: Option<&str>) -> Result<BoolMode> {
    match mode.unwrap_or("or").to_ascii_lowercase().as_str() {
        "or" => Ok(BoolMode::Or),
        "and" => Ok(BoolMode::And),
        other => Err(Error::new(
            Status::InvalidArg,
            format!("mode must be 'or' or 'and', got {other:?}"),
        )),
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Storage and cache config the `connect` URI can't carry. All fields are
/// optional; omit for local / `memory://` / ambient-credential S3 with no
/// disk cache.
#[napi(object)]
pub struct ConnectOptions {
    /// S3-compatible endpoint; requires `region`, `accessKey`, `secretKey`.
    pub endpoint: Option<String>,
    pub region: Option<String>,
    pub access_key: Option<String>,
    pub secret_key: Option<String>,
    /// Local disk-cache directory for remote-backed tables.
    pub cache_dir: Option<String>,
    /// Disk-cache budget in bytes (a JS number; up to 2^53).
    pub cache_budget_bytes: Option<f64>,
    /// Cold-miss strategy: `"hybrid_with_prefetch"` | `"range_only"` |
    /// `"lazy_foreground_with_background_fill"`.
    pub cold_fetch_mode: Option<String>,
}

/// Tuning for `compact`; all fields optional (omitted ⇒ engine default).
#[napi(object)]
pub struct CompactOptions {
    /// Build-time memory budget, in MB.
    pub max_memory_mb: Option<u32>,
    /// Only compact superfiles below this fill percent (0–100).
    pub min_fill_percent: Option<u32>,
    /// Target merged-superfile size, in MB.
    pub target_superfile_size_mb: Option<u32>,
}

/// Row counts from an `update` / `delete`.
#[napi(object)]
pub struct MutationStats {
    /// Rows the predicate matched.
    pub matched: i64,
    /// Rows tombstoned (removed from the live set).
    pub n_tombstoned: i64,
    /// Matched rows that were not found in any live segment.
    pub n_not_found: i64,
}

impl From<infino::MutationStats> for MutationStats {
    fn from(s: infino::MutationStats) -> Self {
        Self {
            matched: s.matched() as i64,
            n_tombstoned: s.n_tombstoned() as i64,
            n_not_found: s.n_not_found() as i64,
        }
    }
}

/// Declares which columns are full-text (BM25) and which are vector (IVF
/// kNN) indexed. Built fluently:
/// `new IndexSpec().fts("body").vector("emb", 384, 256, "cosine")`.
#[napi]
#[derive(Clone, Default)]
pub struct IndexSpec {
    fts: Vec<String>,
    /// `(column, dim, n_cent, metric)`.
    vectors: Vec<(String, u32, u32, String)>,
}

#[napi]
impl IndexSpec {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `column` (a UTF-8 string column) as full-text indexed.
    #[napi]
    pub fn fts(&self, column: String) -> Self {
        let mut next = self.clone();
        next.fts.push(column);
        next
    }

    /// Mark `column` (a `fixed_size_list<float32, dim>`) as vector
    /// indexed. `nCent` is the IVF centroid count (size it to the table's
    /// scale); `metric` is `"cosine"` / `"l2sq"` / `"negdot"`.
    #[napi]
    pub fn vector(&self, column: String, dim: u32, n_cent: u32, metric: String) -> Self {
        let mut next = self.clone();
        next.vectors.push((column, dim, n_cent, metric));
        next
    }
}

impl IndexSpec {
    /// Lower to the core `IndexSpec` builder.
    fn to_rust(&self) -> Result<infino::IndexSpec> {
        let mut spec = infino::IndexSpec::new();
        for column in &self.fts {
            spec = spec.fts(column.clone());
        }
        for (column, dim, n_cent, metric) in &self.vectors {
            spec = spec.vector(
                column.clone(),
                *dim as usize,
                *n_cent as usize,
                metric_from_str(metric)?,
            );
        }
        Ok(spec)
    }
}

/// Open (or create) a catalog rooted at `uri` (local dir, `memory://`, or
/// object-store prefix). S3-compatible static credentials are passed via
/// `options` (the JS-idiomatic form of the Rust `ConnectOptions`).
#[napi]
pub fn connect(uri: String, options: Option<ConnectOptions>) -> Result<Connection> {
    let inner = match options {
        None => infino::connect(&uri),
        Some(o) => {
            let mut opts = infino::ConnectOptions::new();
            // S3 endpoint: all four fields are required together.
            if let Some(endpoint) = o.endpoint {
                let region = o.region.ok_or_else(|| {
                    Error::new(Status::InvalidArg, "region is required with endpoint")
                })?;
                let access_key = o.access_key.ok_or_else(|| {
                    Error::new(Status::InvalidArg, "accessKey is required with endpoint")
                })?;
                let secret_key = o.secret_key.ok_or_else(|| {
                    Error::new(Status::InvalidArg, "secretKey is required with endpoint")
                })?;
                opts = opts.with_s3_endpoint(endpoint, region, access_key, secret_key);
            }
            if let Some(dir) = o.cache_dir {
                opts = opts.with_cache_dir(dir);
            }
            if let Some(bytes) = o.cache_budget_bytes {
                opts = opts.with_cache_budget_bytes(bytes as u64);
            }
            if let Some(mode) = o.cold_fetch_mode {
                opts = opts.with_cold_fetch_mode(cold_fetch_from_str(&mode)?);
            }
            infino::connect_with(&uri, opts)
        }
    }
    .map_err(map_err)?;
    Ok(Connection { inner })
}

/// Infino's build identifier (version + build hash) from the core crate.
/// Re-exported on the JS side as the `BUILDER_ID` string constant.
#[napi]
pub fn builder_id() -> String {
    infino::BUILDER_ID.to_string()
}

/// A catalog connection. `const db = connect(uri)`.
#[napi]
pub struct Connection {
    inner: infino::Connection,
}

#[napi]
impl Connection {
    /// Create a table from an Arrow `Schema` (sent as an IPC `Buffer` —
    /// an empty `apache-arrow` table built with the schema) and an
    /// `IndexSpec`.
    #[napi]
    pub fn create_table(
        &self,
        name: String,
        schema: Buffer,
        indexes: &IndexSpec,
    ) -> Result<Table> {
        let schema = read_schema_ipc(&schema)?;
        let spec = indexes.to_rust()?;
        let inner = self
            .inner
            .create_table(&name, Arc::new(schema), spec)
            .map_err(map_err)?;
        Ok(Table { inner })
    }

    /// Open an existing table by name.
    #[napi]
    pub fn open_table(&self, name: String) -> Result<Table> {
        let inner = self.inner.open_table(&name).map_err(map_err)?;
        Ok(Table { inner })
    }

    /// Drop (unregister) a table.
    #[napi]
    pub fn drop_table(&self, name: String, purge: Option<bool>) -> Result<()> {
        self.inner.drop_table(&name, purge.unwrap_or(false)).map_err(map_err)
    }

    /// List the catalog's table names.
    #[napi]
    pub fn list_tables(&self) -> Result<Vec<String>> {
        self.inner.list_tables().map_err(map_err)
    }

    /// Run SQL across the catalog's tables; returns an Arrow IPC `Buffer`
    /// the JS side reads with `tableFromIPC`. Search is available in SQL
    /// via the TVFs, e.g.
    /// `SELECT _id, score FROM bm25_search('docs', 'body', 'q', 10)`.
    #[napi]
    pub fn query_sql(&self, sql: String) -> Result<Buffer> {
        let batches = self.inner.query_sql(&sql).map_err(map_err)?;
        batches_to_ipc(&batches)
    }
}

/// A single-table handle.
#[napi]
pub struct Table {
    inner: infino::Supertable,
}

#[napi]
impl Table {
    /// Append data, sent as an Arrow IPC `Buffer` (`tableToIPC` on the JS
    /// side). Durable when this returns — one `append` == one commit ==
    /// one sealed segment, so batch rows per call. Multi-batch streams are
    /// concatenated into one commit; an empty stream is a no-op.
    #[napi]
    pub fn append(&self, data: Buffer) -> Result<()> {
        let batches = read_batches_ipc(&data)?;
        if batches.is_empty() {
            return Ok(());
        }
        self.inner.append(&self.align_batches(batches)?).map_err(map_err)
    }

    /// BM25 search over one FTS column. Returns matching rows as an Arrow
    /// IPC `Buffer` (read with `tableFromIPC`). `mode` is `"or"` (default)
    /// or `"and"`. `projection` selects the returned columns — pass
    /// `["_id", "score"]` for just id + score, or omit for full rows.
    #[napi]
    pub fn bm25_search(
        &self,
        column: String,
        query: String,
        k: u32,
        mode: Option<String>,
        projection: Option<Vec<String>>,
    ) -> Result<Buffer> {
        let mode = parse_mode(mode.as_deref())?;
        let proj: Option<Vec<&str>> =
            projection.as_ref().map(|v| v.iter().map(String::as_str).collect());
        let batches = self
            .inner
            .bm25_search(&column, &query, k as usize, mode, proj.as_deref())
            .map_err(map_err)?;
        batches_to_ipc(&batches)
    }

    /// Vector kNN over one vector column. `query` is a `Float32Array`
    /// (crosses by reference — no copy). Returns matching rows as an Arrow
    /// IPC `Buffer` (read with `tableFromIPC`). `projection` selects the
    /// returned columns (`["_id", "score"]` for just id + score, or omit
    /// for full rows).
    #[napi]
    pub fn vector_search(
        &self,
        column: String,
        query: Float32Array,
        k: u32,
        nprobe: Option<u32>,
        rerank_mult: Option<u32>,
        projection: Option<Vec<String>>,
    ) -> Result<Buffer> {
        let mut opts = VectorSearchOptions::new();
        if let Some(n) = nprobe {
            opts = opts.with_nprobe(n as usize);
        }
        if let Some(m) = rerank_mult {
            opts = opts.with_rerank_mult(m as usize);
        }
        let proj: Option<Vec<&str>> =
            projection.as_ref().map(|v| v.iter().map(String::as_str).collect());
        let batches = self
            .inner
            .vector_search(&column, query.as_ref(), k as usize, opts, None, proj.as_deref())
            .map_err(map_err)?;
        batches_to_ipc(&batches)
    }

    /// Unranked token match over one FTS column — every row whose `column`
    /// matches the query's tokens under `mode` (`"or"` default, `"and"`).
    /// Returns Arrow rows like [`Table::bm25_search`], with `score` = 0.0.
    /// `projection` selects columns (omit for full rows).
    #[napi]
    pub fn token_match(
        &self,
        column: String,
        query: String,
        mode: Option<String>,
        projection: Option<Vec<String>>,
    ) -> Result<Buffer> {
        let mode = parse_mode(mode.as_deref())?;
        let proj: Option<Vec<&str>> =
            projection.as_ref().map(|v| v.iter().map(String::as_str).collect());
        let batches = self
            .inner
            .token_match(&column, &query, mode, proj.as_deref())
            .map_err(map_err)?;
        batches_to_ipc(&batches)
    }

    /// Unranked exact match of `value` against `column`. Returns Arrow rows
    /// like [`Table::bm25_search`], with `score` = 0.0. `projection` selects
    /// columns (omit for full rows).
    #[napi]
    pub fn exact_match(
        &self,
        column: String,
        value: String,
        projection: Option<Vec<String>>,
    ) -> Result<Buffer> {
        let proj: Option<Vec<&str>> =
            projection.as_ref().map(|v| v.iter().map(String::as_str).collect());
        let batches = self
            .inner
            .exact_match(&column, &value, proj.as_deref())
            .map_err(map_err)?;
        batches_to_ipc(&batches)
    }

    /// Delete every row matching a SQL `predicate` (e.g. `"status = 'spam'"`),
    /// returning the mutation counts. Requires durable storage — a `memory://`
    /// table surfaces a clear error.
    #[napi]
    pub fn delete(&self, predicate: String) -> Result<MutationStats> {
        let expr = self.parse_predicate(&predicate)?;
        Ok(self.inner.delete(expr).map_err(map_err)?.into())
    }

    /// Replace every row matching a SQL `predicate` with `rows` (an Arrow IPC
    /// `Buffer`, like `append`), 1:1 — the matched count must equal the
    /// replacement-row count or the engine errors. Requires durable storage.
    #[napi]
    pub fn update(&self, predicate: String, rows: Buffer) -> Result<MutationStats> {
        let expr = self.parse_predicate(&predicate)?;
        let batches = read_batches_ipc(&rows)?;
        let aligned = if batches.is_empty() {
            RecordBatch::new_empty(self.inner.schema())
        } else {
            self.align_batches(batches)?
        };
        Ok(self.inner.update(expr, &aligned).map_err(map_err)?.into())
    }

    /// Merge small / underfilled superfiles into larger ones. `settings` tunes
    /// the memory budget, fill threshold, and target size (omit for engine
    /// defaults).
    #[napi]
    pub fn compact(&self, settings: Option<CompactOptions>) -> Result<()> {
        let mut s = CompactionSettings::default();
        if let Some(o) = settings {
            if let Some(v) = o.max_memory_mb {
                s.max_memory_mb = v as u64;
            }
            if let Some(v) = o.min_fill_percent {
                s.min_fill_percent = v as u8;
            }
            if let Some(v) = o.target_superfile_size_mb {
                s.target_superfile_size_mb = v as u64;
            }
        }
        self.inner.compact(&s).map_err(compact_err)
    }

    /// The user-facing Arrow schema, as an Arrow IPC `Buffer` (an empty
    /// table carrying the schema; read with `tableFromIPC`).
    #[napi]
    pub fn schema(&self) -> Result<Buffer> {
        let declared = self.inner.schema();
        write_batches_ipc(declared.as_ref(), &[])
    }
}

impl Table {
    /// Merge IPC batches into one and re-wrap under the table's declared
    /// schema, so the exact-schema check accepts otherwise-nullable inputs (a
    /// genuine type mismatch still errors). Caller guarantees `batches` is
    /// non-empty. Shared by `append` and `update`.
    fn align_batches(&self, batches: Vec<RecordBatch>) -> Result<RecordBatch> {
        let declared = self.inner.schema();
        let merged = if batches.len() == 1 {
            batches.into_iter().next().expect("len == 1")
        } else {
            let schema = batches[0].schema();
            concat_batches(&schema, &batches).map_err(arrow_err)?
        };
        RecordBatch::try_new(declared, merged.columns().to_vec()).map_err(arrow_err)
    }

    /// Parse a SQL predicate string into a DataFusion `Expr`, resolved against
    /// the table's schema. Keeps Python and Node on the same predicate model:
    /// a SQL `WHERE`-style string rather than a hand-built expression tree.
    fn parse_predicate(&self, predicate: &str) -> Result<Expr> {
        let df_schema = DFSchema::try_from(self.inner.schema().as_ref().clone())
            .map_err(|e| Error::new(Status::InvalidArg, format!("schema: {e}")))?;
        SessionContext::new()
            .parse_sql_expr(predicate, &df_schema)
            .map_err(|e| {
                Error::new(Status::InvalidArg, format!("invalid predicate {predicate:?}: {e}"))
            })
    }
}
