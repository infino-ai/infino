// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Python bindings for infino (PyO3 + maturin).
//!
//! Mirrors the Rust catalog API: `infino.connect(uri)` →
//! `db.create_table(...)` / `db.open_table(...)` / `db.query_sql(...)`,
//! and `table.append(...)` / `table.bm25_search(...)` /
//! `table.vector_search(...)`. Arrow is the interchange — schemas and
//! batches cross the boundary as pyarrow objects via the Arrow C Data
//! Interface; search hits come back as `list[dict]`.
//!
//! Sync for v1 (data-science callers expect sync). Built standalone with
//! maturin — it consumes the core crate's curated public API only (no
//! `test-helpers`), so it is also a public-surface consumer test.

use std::sync::Arc;

use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use infino::{BoolMode, ConnectOptions, InfinoError, Metric, VectorSearchOptions};

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

/// Open (or create) a catalog rooted at `uri`. Storage config the URI
/// can't carry is passed as keyword arguments (Q14 — no separate
/// `connect_with` in Python). Today that is the explicit S3-compatible
/// endpoint + static credentials; omit them for local / `memory://` /
/// ambient-credential S3.
#[pyfunction]
#[pyo3(signature = (uri, *, endpoint=None, region=None, access_key=None, secret_key=None))]
fn connect(
    uri: &str,
    endpoint: Option<String>,
    region: Option<String>,
    access_key: Option<String>,
    secret_key: Option<String>,
) -> PyResult<Connection> {
    let inner = match endpoint {
        Some(endpoint) => {
            let region =
                region.ok_or_else(|| PyValueError::new_err("region is required with endpoint"))?;
            let access_key = access_key
                .ok_or_else(|| PyValueError::new_err("access_key is required with endpoint"))?;
            let secret_key = secret_key
                .ok_or_else(|| PyValueError::new_err("secret_key is required with endpoint"))?;
            let opts =
                ConnectOptions::new().with_s3_endpoint(endpoint, region, access_key, secret_key);
            infino::connect_with(uri, opts)
        }
        None => infino::connect(uri),
    }
    .map_err(py_err)?;
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
        name: &str,
        schema: &Bound<'_, PyAny>,
        indexes: &IndexSpec,
    ) -> PyResult<Table> {
        let schema = Schema::from_pyarrow_bound(schema)?;
        let spec = indexes.to_rust()?;
        let inner = self
            .inner
            .create_table(name, Arc::new(schema), spec)
            .map_err(py_err)?;
        Ok(Table { inner })
    }

    /// Open an existing table by name.
    fn open_table(&self, name: &str) -> PyResult<Table> {
        let inner = self.inner.open_table(name).map_err(py_err)?;
        Ok(Table { inner })
    }

    /// Drop (unregister) a table.
    fn drop_table(&self, name: &str) -> PyResult<()> {
        self.inner.drop_table(name).map_err(py_err)
    }

    /// List the catalog's table names.
    fn list_tables(&self) -> PyResult<Vec<String>> {
        self.inner.list_tables().map_err(py_err)
    }

    /// Run SQL across the catalog's tables; returns a pyarrow `Table`.
    /// Search is available in SQL via the TVFs, e.g.
    /// `SELECT _id, score FROM bm25_search('docs', 'body', 'q', 10)`.
    fn query_sql<'py>(&self, py: Python<'py>, sql: &str) -> PyResult<Bound<'py, PyAny>> {
        let batches = self.inner.query_sql(sql).map_err(py_err)?;
        // `Vec<RecordBatch>` converts to a Python *list* of pyarrow
        // RecordBatches; assemble them into a single pyarrow `Table`.
        let py_batches = batches.to_pyarrow(py)?;
        let pyarrow = py.import("pyarrow")?;
        pyarrow
            .getattr("Table")?
            .call_method1("from_batches", (py_batches,))
    }
}

/// A single-table handle.
#[pyclass]
struct Table {
    inner: infino::Supertable,
}

#[pymethods]
impl Table {
    /// Append a pyarrow `RecordBatch`. Durable when this returns (one
    /// `append` == one commit == one sealed segment).
    fn append(&self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let batch = RecordBatch::from_pyarrow_bound(data)?;
        self.inner.append(&batch).map_err(py_err)
    }

    /// BM25 search over one FTS column. Returns `list[{"_id", "score"}]`,
    /// best first. `mode` is `"or"` (default) or `"and"`.
    #[pyo3(signature = (column, query, k, mode=None))]
    fn bm25_search<'py>(
        &self,
        py: Python<'py>,
        column: &str,
        query: &str,
        k: usize,
        mode: Option<&str>,
    ) -> PyResult<Bound<'py, PyList>> {
        let mode = match mode.unwrap_or("or").to_ascii_lowercase().as_str() {
            "or" => BoolMode::Or,
            "and" => BoolMode::And,
            other => {
                return Err(PyValueError::new_err(format!(
                    "mode must be 'or' or 'and', got {other:?}"
                )));
            }
        };
        let hits = self
            .inner
            .bm25_search(column, query, k, mode)
            .map_err(py_err)?;
        hits_to_pylist(py, &hits)
    }

    /// Vector kNN over one vector column. `query` is a `list[float]`.
    /// Returns `list[{"_id", "score"}]`, nearest first.
    #[pyo3(signature = (column, query, k, nprobe=None))]
    fn vector_search<'py>(
        &self,
        py: Python<'py>,
        column: &str,
        query: Vec<f32>,
        k: usize,
        nprobe: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let mut opts = VectorSearchOptions::new();
        if let Some(n) = nprobe {
            opts = opts.with_nprobe(n);
        }
        let hits = self
            .inner
            .vector_search(column, &query, k, opts)
            .map_err(py_err)?;
        hits_to_pylist(py, &hits)
    }

    /// The user-facing Arrow schema, as a pyarrow `Schema`.
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.schema().as_ref().to_pyarrow(py)
    }
}

/// Convert search hits to a Python `list[{"_id": int, "score": float}]`.
fn hits_to_pylist<'py>(
    py: Python<'py>,
    hits: &[infino::SearchHit],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for hit in hits {
        let row = PyDict::new(py);
        row.set_item("_id", hit.id)?;
        row.set_item("score", hit.score)?;
        list.append(row)?;
    }
    Ok(list)
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
    Ok(())
}
