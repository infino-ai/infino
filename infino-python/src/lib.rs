// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Python bindings for infino (PyO3 + maturin).
//!
//! Mirrors the Rust catalog API: `infino.connect(uri)` →
//! `db.create_table(...)` / `db.open_table(...)` / `db.query_sql(...)`,
//! and `table.append(...)` / `table.bm25_search(...)` /
//! `table.vector_search(...)`. Arrow is the interchange — schemas and
//! batches cross the boundary as pyarrow objects via the Arrow C Data
//! Interface; search and SQL results come back as pyarrow `Table`s.
//!
//! Sync for v1 (data-science callers expect sync). Built standalone with
//! maturin — it consumes the core crate's curated public API only (no
//! `test-helpers`), so it is also a public-surface consumer test.

use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use datafusion::common::DFSchema;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::Expr;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use infino::{
    BoolMode, ColdFetchMode, CompactionError, CompactionSettings, ConnectOptions, InfinoError,
    Metric, VectorSearchOptions,
};

/// Map a core [`InfinoError`] to the closest Python exception.
fn py_err(e: InfinoError) -> PyErr {
    match e {
        InfinoError::NotFound(m) => PyKeyError::new_err(m),
        InfinoError::AlreadyExists(m)
        | InfinoError::Schema(m)
        | InfinoError::Cardinality(m)
        | InfinoError::Query(m) => PyValueError::new_err(m),
        InfinoError::Io(m) | InfinoError::Backend(m) => PyRuntimeError::new_err(m),
        // `InfinoError` is `#[non_exhaustive]`: future variants fall back
        // to a generic runtime error carrying the message.
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Compaction has its own error type, so it doesn't go through `py_err`.
/// These are storage / build failures — surfaced as runtime errors, like
/// the mutation paths.
fn compact_err(e: CompactionError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Parse a metric name (`"cosine"` / `"l2sq"` / `"negdot"`).
fn metric_from_str(s: &str) -> PyResult<Metric> {
    match s.to_ascii_lowercase().as_str() {
        "cosine" => Ok(Metric::Cosine),
        "l2sq" | "l2" => Ok(Metric::L2Sq),
        "negdot" | "dot" => Ok(Metric::NegDot),
        other => Err(PyValueError::new_err(format!(
            "unknown metric {other:?}; use 'cosine', 'l2sq', or 'negdot'"
        ))),
    }
}

/// Parse a cold-fetch-mode name into its [`ColdFetchMode`].
fn cold_fetch_from_str(s: &str) -> PyResult<ColdFetchMode> {
    match s.to_ascii_lowercase().as_str() {
        "hybrid_with_prefetch" => Ok(ColdFetchMode::HybridWithPrefetch),
        "range_only" => Ok(ColdFetchMode::RangeOnly),
        "lazy_foreground_with_background_fill" => Ok(ColdFetchMode::LazyForegroundWithBackgroundFill),
        other => Err(PyValueError::new_err(format!(
            "unknown cold_fetch_mode {other:?}; use 'hybrid_with_prefetch', \
             'range_only', or 'lazy_foreground_with_background_fill'"
        ))),
    }
}

/// Open (or create) a catalog rooted at `uri`. Storage config the URI
/// can't carry is passed as keyword arguments: explicit S3 endpoint +
/// static credentials, and the optional local disk cache (`cache_dir`,
/// `cache_budget_bytes`, `cold_fetch_mode`). Omit all for local /
/// `memory://` / ambient-credential S3.
// Flat kwargs are the intended Python API; a config struct would change it.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (uri, *, endpoint=None, region=None, access_key=None, secret_key=None,
                    cache_dir=None, cache_budget_bytes=None, cold_fetch_mode=None))]
fn connect(
    py: Python<'_>,
    uri: &str,
    endpoint: Option<String>,
    region: Option<String>,
    access_key: Option<String>,
    secret_key: Option<String>,
    cache_dir: Option<String>,
    cache_budget_bytes: Option<u64>,
    cold_fetch_mode: Option<String>,
) -> PyResult<Connection> {
    // Opening a connection can touch object storage; release the GIL so
    // other Python threads run during the (blocking) I/O.
    let inner = py.detach(|| {
        let mut opts = ConnectOptions::new();
        let mut has_options = false;
        // The S3 endpoint + credentials are all-or-nothing: any one of them
        // means the caller wants an explicit endpoint, so require the rest
        // rather than silently dropping a partial config back to ambient.
        if endpoint.is_some() || region.is_some() || access_key.is_some() || secret_key.is_some() {
            let endpoint = endpoint
                .ok_or_else(|| PyValueError::new_err("endpoint is required with S3 credentials"))?;
            let region = region
                .ok_or_else(|| PyValueError::new_err("region is required for an S3 endpoint"))?;
            let access_key = access_key
                .ok_or_else(|| PyValueError::new_err("access_key is required for an S3 endpoint"))?;
            let secret_key = secret_key
                .ok_or_else(|| PyValueError::new_err("secret_key is required for an S3 endpoint"))?;
            opts = opts.with_s3_endpoint(endpoint, region, access_key, secret_key);
            has_options = true;
        }
        if let Some(dir) = cache_dir {
            opts = opts.with_cache_dir(dir);
            has_options = true;
        }
        if let Some(bytes) = cache_budget_bytes {
            opts = opts.with_cache_budget_bytes(bytes);
            has_options = true;
        }
        if let Some(mode) = cold_fetch_mode {
            opts = opts.with_cold_fetch_mode(cold_fetch_from_str(&mode)?);
            has_options = true;
        }
        // Preserve the plain `connect(uri)` path when no options are set.
        if has_options {
            infino::connect_with(uri, opts).map_err(py_err)
        } else {
            infino::connect(uri).map_err(py_err)
        }
    })?;
    Ok(Connection { inner })
}

/// Declares which columns are full-text (BM25) and which are vector
/// (IVF kNN) indexed. Built fluently:
/// `IndexSpec().fts("body").vector("emb", 384, 256, "cosine")`.
#[pyclass(name = "IndexSpec", skip_from_py_object)]
#[derive(Clone, Default)]
struct IndexSpec {
    fts: Vec<String>,
    /// `(column, dim, n_cent, metric)`.
    vectors: Vec<(String, usize, usize, String)>,
}

#[pymethods]
impl IndexSpec {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Mark `column` (a UTF-8 string column) as full-text indexed.
    fn fts(&self, column: String) -> Self {
        let mut next = self.clone();
        next.fts.push(column);
        next
    }

    /// Mark `column` (a `fixed_size_list<float32, dim>`) as vector
    /// indexed. `n_cent` is the IVF centroid count (size it to the
    /// table's scale); `metric` is `"cosine"` / `"l2sq"` / `"negdot"`.
    fn vector(&self, column: String, dim: usize, n_cent: usize, metric: String) -> Self {
        let mut next = self.clone();
        next.vectors.push((column, dim, n_cent, metric));
        next
    }
}

impl IndexSpec {
    /// Lower to the core `IndexSpec` builder.
    fn to_rust(&self) -> PyResult<infino::IndexSpec> {
        let mut spec = infino::IndexSpec::new();
        for column in &self.fts {
            spec = spec.fts(column.clone());
        }
        for (column, dim, n_cent, metric) in &self.vectors {
            spec = spec.vector(column.clone(), *dim, *n_cent, metric_from_str(metric)?);
        }
        Ok(spec)
    }
}

/// A catalog connection. `db = infino.connect(uri)`.
#[pyclass]
struct Connection {
    inner: infino::Connection,
}

#[pymethods]
impl Connection {
    /// Create a table from a pyarrow `Schema` and an `IndexSpec`.
    fn create_table(
        &self,
        py: Python<'_>,
        name: &str,
        schema: &Bound<'_, PyAny>,
        indexes: &IndexSpec,
    ) -> PyResult<Table> {
        // pyarrow conversions touch Python (hold the GIL); the table
        // build commits to storage, so drop the GIL for that part.
        let schema = Arc::new(Schema::from_pyarrow_bound(schema)?);
        let spec = indexes.to_rust()?;
        let inner = py
            .detach(|| self.inner.create_table(name, schema, spec))
            .map_err(py_err)?;
        Ok(Table { inner })
    }

    /// Open an existing table by name.
    fn open_table(&self, py: Python<'_>, name: &str) -> PyResult<Table> {
        let inner = py
            .detach(|| self.inner.open_table(name))
            .map_err(py_err)?;
        Ok(Table { inner })
    }

    /// Drop (unregister) a table. `purge=True` also deletes the table's
    /// storage subtree after the catalog commit; the default leaves the
    /// bytes in place (readers pinned to a pre-drop snapshot keep
    /// working).
    #[pyo3(signature = (name, purge=false))]
    fn drop_table(&self, py: Python<'_>, name: &str, purge: bool) -> PyResult<()> {
        py.detach(|| self.inner.drop_table(name, purge))
            .map_err(py_err)
    }

    /// List the catalog's table names.
    fn list_tables(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        py.detach(|| self.inner.list_tables()).map_err(py_err)
    }

    /// Run SQL across the catalog's tables; returns a pyarrow `Table`.
    /// Search is available in SQL via the TVFs, e.g.
    /// `SELECT _id, score FROM bm25_search('docs', 'body', 'q', 10)`.
    fn query_sql<'py>(&self, py: Python<'py>, sql: &str) -> PyResult<Bound<'py, PyAny>> {
        let batches = py
            .detach(|| self.inner.query_sql(sql))
            .map_err(py_err)?;
        batches_to_pyarrow_table(py, batches)
    }
}

/// Row counts returned by `update` / `delete`.
#[pyclass(name = "MutationStats", frozen)]
struct MutationStats {
    #[pyo3(get)]
    matched: usize,
    #[pyo3(get)]
    n_tombstoned: usize,
    #[pyo3(get)]
    n_not_found: usize,
}

impl MutationStats {
    fn from_core(s: &infino::MutationStats) -> Self {
        Self {
            matched: s.matched(),
            n_tombstoned: s.n_tombstoned(),
            n_not_found: s.n_not_found(),
        }
    }
}

#[pymethods]
impl MutationStats {
    fn __repr__(&self) -> String {
        format!(
            "MutationStats(matched={}, n_tombstoned={}, n_not_found={})",
            self.matched, self.n_tombstoned, self.n_not_found
        )
    }
}

/// Tuning for `compact`; omitted fields fall back to engine defaults.
#[pyclass(name = "CompactOptions", skip_from_py_object)]
#[derive(Clone, Default)]
struct CompactOptions {
    max_memory_mb: Option<u64>,
    min_fill_percent: Option<u8>,
    target_superfile_size_mb: Option<u64>,
}

#[pymethods]
impl CompactOptions {
    #[new]
    #[pyo3(signature = (*, max_memory_mb=None, min_fill_percent=None, target_superfile_size_mb=None))]
    fn new(
        max_memory_mb: Option<u64>,
        min_fill_percent: Option<u8>,
        target_superfile_size_mb: Option<u64>,
    ) -> Self {
        Self {
            max_memory_mb,
            min_fill_percent,
            target_superfile_size_mb,
        }
    }
}

/// A single-table handle.
#[pyclass]
struct Table {
    inner: infino::Supertable,
}

#[pymethods]
impl Table {
    /// Append data. Accepts a pyarrow `RecordBatch` or `Table`, a pandas
    /// `DataFrame`, or a `list[dict]` (coerced to Arrow with the table's
    /// declared schema). Durable when this returns — one `append` == one
    /// commit == one sealed superfile, so batch rows per call.
    fn append(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let declared = self.inner.schema();
        let py_schema = declared.as_ref().to_pyarrow(py)?;
        match coerce_to_record_batch(py, data, &py_schema)? {
            Some(batch) => {
                let aligned = align_to_schema(declared, batch)?;
                // Append commits a superfile to storage — release the GIL.
                py.detach(|| self.inner.append(&aligned)).map_err(py_err)
            }
            // Empty input — nothing to append (no empty commit).
            None => Ok(()),
        }
    }

    /// BM25 search over one FTS column. Returns a pyarrow `Table`.
    /// `projection` names the output columns (`_id`, any scalar column,
    /// or the trailing `score` — higher is better); omitting it returns
    /// the engine-native `_id` + `score` pair with no scalar decode.
    /// Materializing row data is an explicit opt-in by naming columns.
    /// `mode` is `"or"` (default) or `"and"`.
    #[pyo3(signature = (column, query, k, mode=None, projection=None))]
    fn bm25_search<'py>(
        &self,
        py: Python<'py>,
        column: &str,
        query: &str,
        k: usize,
        mode: Option<&str>,
        projection: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mode = parse_mode(mode)?;
        let batches = py
            .detach(|| {
                let names = projection_refs(&projection);
                self.inner.bm25_search(column, query, k, mode, names.as_deref())
            })
            .map_err(py_err)?;
        batches_to_pyarrow_table(py, batches)
    }

    /// Vector kNN over one vector column. `query` is a `list[float]`.
    /// Returns a pyarrow `Table`. `projection` names the output columns
    /// (`_id`, any scalar column, or the trailing `score` — distance,
    /// smaller is nearer); omitting it returns the engine-native
    /// `_id` + `score` pair with no scalar decode. Materializing row
    /// data is an explicit opt-in by naming columns.
    #[pyo3(signature = (column, query, k, nprobe=None, projection=None))]
    fn vector_search<'py>(
        &self,
        py: Python<'py>,
        column: &str,
        query: Vec<f32>,
        k: usize,
        nprobe: Option<usize>,
        projection: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut opts = VectorSearchOptions::new();
        if let Some(n) = nprobe {
            opts = opts.with_nprobe(n);
        }
        let batches = py
            .detach(|| {
                let names = projection_refs(&projection);
                self.inner.vector_search(column, &query, k, opts, None, names.as_deref())
            })
            .map_err(py_err)?;
        batches_to_pyarrow_table(py, batches)
    }

    /// Unranked token match over one FTS column. Returns a pyarrow
    /// `Table` like `bm25_search`, but `score` is `0.0` and row order is
    /// unspecified. `mode` is `"or"` (default) or `"and"`; `projection`
    /// follows the same rules as `bm25_search`.
    #[pyo3(signature = (column, query, mode=None, projection=None))]
    fn token_match<'py>(
        &self,
        py: Python<'py>,
        column: &str,
        query: &str,
        mode: Option<&str>,
        projection: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mode = parse_mode(mode)?;
        let batches = py
            .detach(|| {
                let names = projection_refs(&projection);
                self.inner.token_match(column, query, mode, names.as_deref())
            })
            .map_err(py_err)?;
        batches_to_pyarrow_table(py, batches)
    }

    /// Unranked exact match of `value` against `column`. Returns a
    /// pyarrow `Table` like `bm25_search`, with `score` fixed at `0.0`
    /// and unspecified row order. `projection` follows the same rules
    /// as `bm25_search`.
    #[pyo3(signature = (column, value, projection=None))]
    fn exact_match<'py>(
        &self,
        py: Python<'py>,
        column: &str,
        value: &str,
        projection: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let batches = py
            .detach(|| {
                let names = projection_refs(&projection);
                self.inner.exact_match(column, value, names.as_deref())
            })
            .map_err(py_err)?;
        batches_to_pyarrow_table(py, batches)
    }

    /// Delete rows matching a SQL predicate string, e.g. `"status = 'spam'"`.
    /// Needs durable storage — a `memory://` table raises.
    fn delete(&self, py: Python<'_>, predicate: &str) -> PyResult<MutationStats> {
        // Parse and mutate both off the GIL — neither touches Python.
        let stats = py.detach(|| {
            let expr = self.parse_predicate(predicate)?;
            self.inner.delete(expr).map_err(py_err)
        })?;
        Ok(MutationStats::from_core(&stats))
    }

    /// Replace rows matching a SQL predicate with `new_rows` (same shapes as
    /// `append`). Replacement is 1:1 — the match count must equal the number
    /// of rows supplied. Needs durable storage.
    fn update(
        &self,
        py: Python<'_>,
        predicate: &str,
        new_rows: &Bound<'_, PyAny>,
    ) -> PyResult<MutationStats> {
        let declared = self.inner.schema();
        let py_schema = declared.as_ref().to_pyarrow(py)?;
        // Pass an empty batch through rather than short-circuiting like
        // `append` does — we want the engine's cardinality check to run.
        let aligned = match coerce_to_record_batch(py, new_rows, &py_schema)? {
            Some(batch) => align_to_schema(declared, batch)?,
            None => RecordBatch::new_empty(declared),
        };
        // Parse and mutate both off the GIL — neither touches Python.
        let stats = py.detach(|| {
            let expr = self.parse_predicate(predicate)?;
            self.inner.update(expr, &aligned).map_err(py_err)
        })?;
        Ok(MutationStats::from_core(&stats))
    }

    /// Merge small / underfilled superfiles into larger ones. Omit
    /// `settings` for engine defaults.
    #[pyo3(signature = (settings=None))]
    fn compact(&self, py: Python<'_>, settings: Option<&CompactOptions>) -> PyResult<()> {
        let mut s = CompactionSettings::default();
        if let Some(o) = settings {
            if let Some(v) = o.max_memory_mb {
                s.max_memory_mb = v;
            }
            if let Some(v) = o.min_fill_percent {
                s.min_fill_percent = v;
            }
            if let Some(v) = o.target_superfile_size_mb {
                s.target_superfile_size_mb = v;
            }
        }
        py.detach(|| self.inner.compact(&s)).map_err(compact_err)
    }

    /// The user-facing Arrow schema, as a pyarrow `Schema`.
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.schema().as_ref().to_pyarrow(py)
    }
}

impl Table {
    /// Resolve a SQL predicate string into the `Expr` the core mutation
    /// API takes. Column names resolve against the table's own schema.
    fn parse_predicate(&self, predicate: &str) -> PyResult<Expr> {
        let df_schema = DFSchema::try_from(self.inner.schema().as_ref().clone())
            .map_err(|e| PyValueError::new_err(format!("schema: {e}")))?;
        SessionContext::new()
            .parse_sql_expr(predicate, &df_schema)
            .map_err(|e| PyValueError::new_err(format!("invalid predicate {predicate:?}: {e}")))
    }
}

/// Borrow an optional Python projection (`list[str]`) as the `&str`
/// slices the Rust search APIs take. Shared by all four search methods.
fn projection_refs(projection: &Option<Vec<String>>) -> Option<Vec<&str>> {
    projection
        .as_ref()
        .map(|p| p.iter().map(String::as_str).collect())
}

/// Assemble `Vec<RecordBatch>` into a single pyarrow `Table`. Shared by
/// `query_sql` and the row-returning search methods.
fn batches_to_pyarrow_table<'py>(
    py: Python<'py>,
    batches: Vec<RecordBatch>,
) -> PyResult<Bound<'py, PyAny>> {
    let py_batches = batches.to_pyarrow(py)?;
    let pyarrow = py.import("pyarrow")?;
    pyarrow
        .getattr("Table")?
        .call_method1("from_batches", (py_batches,))
}

/// Parse the `"or"` (default) / `"and"` boolean mode argument.
fn parse_mode(mode: Option<&str>) -> PyResult<BoolMode> {
    match mode.unwrap_or("or").to_ascii_lowercase().as_str() {
        "or" => Ok(BoolMode::Or),
        "and" => Ok(BoolMode::And),
        other => Err(PyValueError::new_err(format!(
            "mode must be 'or' or 'and', got {other:?}"
        ))),
    }
}

/// Re-wrap a coerced batch under the table's declared schema. Python
/// sources (pandas, list[dict]) are inherently nullable; this lets the
/// exact-schema check accept them. A genuine type / null mismatch still errors.
fn align_to_schema(declared: Arc<Schema>, batch: RecordBatch) -> PyResult<RecordBatch> {
    RecordBatch::try_new(declared, batch.columns().to_vec())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Coerce append input — a pyarrow `RecordBatch` / `Table`, a pandas
/// `DataFrame`, or a `list[dict]` — into a single `RecordBatch`. `schema`
/// is the table's declared pyarrow `Schema`, used to type the `list` /
/// `DataFrame` conversions so column types match. Returns `None` for
/// empty input (so an empty append is a no-op, not an empty commit).
fn coerce_to_record_batch(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    schema: &Bound<'_, PyAny>,
) -> PyResult<Option<RecordBatch>> {
    let pa = py.import("pyarrow")?;
    let table_cls = pa.getattr("Table")?;
    let record_batch_cls = pa.getattr("RecordBatch")?;

    // A single RecordBatch: convert directly.
    if data.is_instance(&record_batch_cls)? {
        return Ok(Some(RecordBatch::from_pyarrow_bound(data)?));
    }

    // Normalize a Table / list[dict] / DataFrame to a pyarrow Table,
    // typed by the table's own schema so column types line up.
    let table = if data.is_instance(&table_cls)? {
        data.clone()
    } else if data.is_instance_of::<PyList>() {
        let kwargs = PyDict::new(py);
        kwargs.set_item("schema", schema)?;
        table_cls.call_method("from_pylist", (data,), Some(&kwargs))?
    } else {
        // Assume a pandas DataFrame (or anything `from_pandas` accepts).
        let kwargs = PyDict::new(py);
        kwargs.set_item("schema", schema)?;
        kwargs.set_item("preserve_index", false)?;
        table_cls.call_method("from_pandas", (data,), Some(&kwargs))?
    };

    // Collapse the Table's chunks into a single RecordBatch — one append
    // is one commit / one sealed superfile.
    let batches = table
        .call_method0("combine_chunks")?
        .call_method0("to_batches")?;
    let batches = batches.cast::<PyList>()?;
    if batches.is_empty() {
        return Ok(None);
    }
    let mut rust_batches = Vec::with_capacity(batches.len());
    for batch in batches.iter() {
        rust_batches.push(RecordBatch::from_pyarrow_bound(&batch)?);
    }
    if rust_batches.len() == 1 {
        Ok(rust_batches.into_iter().next())
    } else {
        let merged_schema = rust_batches[0].schema();
        concat_batches(&merged_schema, &rust_batches)
            .map(Some)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// Named `infino_ext` (not `infino`) so the generated module item doesn't
// shadow the `infino` crate inside this file; `#[pyo3(name = "infino")]`
// keeps the Python module name `infino` (init symbol `PyInit_infino`).
#[pymodule]
#[pyo3(name = "infino")]
fn infino_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(connect, m)?)?;
    m.add_class::<Connection>()?;
    m.add_class::<Table>()?;
    m.add_class::<IndexSpec>()?;
    m.add_class::<MutationStats>()?;
    m.add_class::<CompactOptions>()?;
    Ok(())
}
