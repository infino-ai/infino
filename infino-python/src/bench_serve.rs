// SPDX-License-Identifier: Apache-2.0
//! EXPERIMENTAL benchmark serve mode (raw-TCP) — **not a production server**.
//!
//! Wraps the embedded engine behind a minimal raw-TCP wire so a networked
//! benchmark client (e.g. VectorDBBench) can drive it, the same way it drives
//! any server database. No auth, no TLS, no durability, single table. The
//! supported infino interface remains embedded (`connect`/`open_table`); this
//! exists purely to measure warm search throughput under a client-server
//! topology.
//!
//! Two entry points:
//!   - [`bench_serve_tcp`] — SEARCH-ONLY: opens an already-built table and
//!     serves top-k over a fixed request format. Build runs in-process.
//!   - [`bench_serve_build_tcp`] — BUILD+SERVE: an opcode-tagged protocol that
//!     also creates the table, appends rows, and optimizes — so a client on a
//!     separate machine can drive the whole build over the wire (a true
//!     two-machine client/server topology). Search returns the dataset id
//!     directly, mapped from the engine `_id` by an in-memory table the server
//!     builds once at optimize (no per-query scalar projection).

use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, RwLock},
    thread,
};

use arrow_array::{
    Array, Decimal128Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use infino::{ConnectOptions, Connection, IndexSpec, Metric, Supertable, connect_with};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

// ----------------------------------------------------------------------------
// Search-only server (unchanged wire): request u32 k, u32 dim, dim*4 f32 query;
// response u32 n, n*IDSIZE bytes (16 for `_id`, 8 for a projected id column).
// ----------------------------------------------------------------------------

fn handle(
    mut stream: TcpStream,
    st: Arc<Supertable>,
    col: String,
    id_col: Option<String>,
    dim: usize,
) {
    stream.set_nodelay(true).ok();
    let mut hdr = [0u8; 8];
    loop {
        if stream.read_exact(&mut hdr).is_err() {
            break; // client closed
        }
        let k = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
        let req_dim = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
        if req_dim != dim {
            eprintln!(
                "[infino-bench-serve] query dim {req_dim} != table dim {dim}; closing connection"
            );
            break;
        }
        let mut qbuf = vec![0u8; dim * 4];
        if stream.read_exact(&mut qbuf).is_err() {
            break;
        }
        let query: Vec<f32> = qbuf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let projection = id_col.as_deref().map(|c| [c]);
        let proj_ref = projection.as_ref().map(|p| p.as_slice());
        let batches = match st.vector_search(&col, &query, k, None, proj_ref) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[infino-bench-serve] search error: {e}");
                break;
            }
        };
        let mut out: Vec<u8> = Vec::with_capacity(4 + k * 16);
        out.extend_from_slice(&0u32.to_le_bytes()); // n placeholder
        let mut n: u32 = 0;
        for b in &batches {
            if let Some(ic) = id_col.as_deref() {
                let Some(c) = b.column_by_name(ic) else {
                    eprintln!(
                        "[infino-bench-serve] id column {ic:?} absent from result batch; closing"
                    );
                    return;
                };
                let Some(arr) = c.as_any().downcast_ref::<Int64Array>() else {
                    eprintln!(
                        "[infino-bench-serve] id column {ic:?} is not Int64 in result batch; closing"
                    );
                    return;
                };
                for i in 0..arr.len() {
                    out.extend_from_slice(&arr.value(i).to_le_bytes());
                    n += 1;
                }
            } else {
                let Some(c) = b.column_by_name("_id") else {
                    break;
                };
                let Some(dec) = c.as_any().downcast_ref::<Decimal128Array>() else {
                    break;
                };
                for i in 0..dec.len() {
                    out.extend_from_slice(&dec.value(i).to_be_bytes());
                    n += 1;
                }
            }
        }
        out[0..4].copy_from_slice(&n.to_le_bytes());
        if stream.write_all(&out).is_err() {
            break;
        }
    }
}

/// EXPERIMENTAL raw-TCP benchmark serve loop (search only). Opens `table` under
/// `data_path` and serves top-k ids over TCP. Blocks forever. Not a production
/// server (no auth/TLS/durability).
#[pyfunction]
#[pyo3(signature = (data_path, table, col, addr, cache_bytes, id_col=""))]
pub fn bench_serve_tcp(
    py: Python<'_>,
    data_path: &str,
    table: &str,
    col: &str,
    addr: &str,
    cache_bytes: u64,
    id_col: &str,
) -> PyResult<()> {
    eprintln!(
        "[infino-bench-serve] EXPERIMENTAL benchmark serve mode — no auth, no TLS, no durability, \
         single table. NOT production-validated; the supported infino interface is embedded."
    );
    let data_path = data_path.to_string();
    let table = table.to_string();
    let col = col.to_string();
    let addr = addr.to_string();
    let id_col: Option<String> = if id_col.is_empty() {
        None
    } else {
        Some(id_col.to_string())
    };
    py.detach(move || -> Result<(), String> {
        let opts = ConnectOptions::new()
            .with_cache_budget_bytes(cache_bytes)
            .with_cache_dir(format!("{data_path}/cache"));
        let conn = connect_with(&data_path, opts).map_err(|e| e.to_string())?;
        let st = Arc::new(conn.open_table(&table).map_err(|e| e.to_string())?);
        let schema = st.schema();
        let vfield = schema
            .field_with_name(&col)
            .map_err(|e| format!("vector column {col:?} not in schema: {e}"))?;
        let dim = match vfield.data_type() {
            DataType::FixedSizeList(_, n) => *n as usize,
            other => {
                return Err(format!(
                    "vector column {col:?} is {other:?}, expected FixedSizeList<Float32>"
                ));
            }
        };
        if let Some(ic) = id_col.as_deref() {
            let ifield = schema
                .field_with_name(ic)
                .map_err(|e| format!("id column {ic:?} not in schema: {e}"))?;
            if !matches!(ifield.data_type(), DataType::Int64) {
                return Err(format!(
                    "id column {ic:?} is {:?}, expected Int64",
                    ifield.data_type()
                ));
            }
        }
        let warm = vec![0.0f32; dim];
        let _ = st.vector_search(&col, &warm, 10, None, None);
        let listener = TcpListener::bind(&addr).map_err(|e| e.to_string())?;
        eprintln!(
            "[infino-bench-serve] table={table} dim={dim} listening={addr} id_col={} cache={cache_bytes} — ready",
            id_col.as_deref().unwrap_or("_id")
        );
        for stream in listener.incoming() {
            match stream {
                Ok(s) => {
                    let st = st.clone();
                    let col = col.clone();
                    let id_col = id_col.clone();
                    thread::spawn(move || handle(s, st, col, id_col, dim));
                }
                Err(e) => eprintln!("[infino-bench-serve] accept error: {e}"),
            }
        }
        Ok(())
    })
    .map_err(PyRuntimeError::new_err)?;
    Ok(())
}

// ----------------------------------------------------------------------------
// Build+serve server (opcode-tagged wire). Lets a client on a separate machine
// drive create/append/optimize/search over TCP — a true two-machine topology.
//
// Wire (little-endian). Each request is a 1-byte opcode + payload:
//   CREATE  (1): u32 dim, u32 metric_len, metric_len bytes  -> u8 status
//   APPEND  (2): u32 nrows, u32 dim, nrows*(i64 id + dim*f32) -> u8 status, u64 count
//   OPTIMIZE(3): (none)                                       -> u8 status
//   SEARCH  (4): u32 k, u32 dim, dim*f32                      -> u32 n, n*i64 dataset ids
//   DROP    (5): (none)                                       -> u8 status
// status: 0 = ok, 1 = error (followed by u32 msg_len, msg bytes).
//
// A driver issues these in order: CREATE, then all APPENDs, then OPTIMIZE
// (which builds the serving index and the `_id` -> dataset-id map), then
// SEARCHes. SEARCH before OPTIMIZE returns an empty result; a row appended
// after OPTIMIZE is not in the id map and is dropped from results — so the
// build phase must finish before search begins for recall to be exact.
// ----------------------------------------------------------------------------

const OP_CREATE: u8 = 1;
const OP_APPEND: u8 = 2;
const OP_OPTIMIZE: u8 = 3;
const OP_SEARCH: u8 = 4;
const OP_DROP: u8 = 5;

// Sanity bounds on wire-supplied lengths so a garbled header can't drive an
// unbounded allocation before any validation. Generous vs. real embedding dims
// and batch sizes; a header past these is a protocol error, not a workload.
const MAX_DIM: usize = 1 << 16; // 65536
const MAX_METRIC_LEN: usize = 64;
const MAX_APPEND_ROWS: usize = 1 << 24; // ~16.7M rows per APPEND

fn metric_from_str(s: &str) -> Result<Metric, String> {
    match s.to_ascii_lowercase().as_str() {
        "cosine" => Ok(Metric::Cosine),
        "l2sq" | "l2" => Ok(Metric::L2Sq),
        "negdot" | "dot" => Ok(Metric::NegDot),
        other => Err(format!("unknown metric {other:?}")),
    }
}

struct TableState {
    table: Option<Arc<Supertable>>,
    dim: usize,
    /// engine `_id` (i128) -> dataset id (i64), built once at optimize so search
    /// returns the dataset id without a per-query scalar projection. Kept as a
    /// contiguous `Vec` sorted by `_id` (binary-searched per result) rather than
    /// a `HashMap`: half the resident bytes per row and no per-entry allocation,
    /// which matters when the map holds one entry per corpus vector.
    id_map: Option<Arc<Vec<(i128, i64)>>>,
}

struct ServeState {
    conn: Connection,
    table: String,
    col: String,
    id_col: String,
    state: RwLock<TableState>,
}

fn read_u32(stream: &mut TcpStream) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    stream.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_n(stream: &mut TcpStream, n: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    stream.read_exact(&mut buf)?;
    Ok(buf)
}

fn write_status(stream: &mut TcpStream, result: Result<(), String>) -> std::io::Result<()> {
    match result {
        Ok(()) => stream.write_all(&[0u8]),
        Err(e) => {
            let msg = e.as_bytes();
            stream.write_all(&[1u8])?;
            stream.write_all(&(msg.len() as u32).to_le_bytes())?;
            stream.write_all(msg)
        }
    }
}

/// Build the `id`/`emb` schema and a matching [`IndexSpec`] for a vector table.
fn build_schema(id_col: &str, col: &str, dim: usize) -> Arc<Schema> {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let emb_type = DataType::FixedSizeList(item, dim as i32);
    Arc::new(Schema::new(vec![
        Field::new(id_col, DataType::Int64, false),
        Field::new(col, emb_type, false),
    ]))
}

fn do_create(stream: &mut TcpStream, srv: &ServeState) -> std::io::Result<()> {
    let dim = read_u32(stream)? as usize;
    let mlen = read_u32(stream)? as usize;
    // Bound the metric length before allocating; the dim is validated inside the
    // result closure so its error reaches the client as a status message.
    if mlen > MAX_METRIC_LEN {
        return write_status(stream, Err(format!("metric length {mlen} exceeds max")));
    }
    let metric_bytes = read_n(stream, mlen)?;
    let res = (|| -> Result<(), String> {
        if dim == 0 || dim > MAX_DIM {
            return Err(format!("dim {dim} out of range (1..={MAX_DIM})"));
        }
        let metric = metric_from_str(&String::from_utf8_lossy(&metric_bytes))?;
        // Fresh table each run: a reopened build would serve stale vectors.
        let _ = srv.conn.drop_table(&srv.table, true);
        let schema = build_schema(&srv.id_col, &srv.col, dim);
        let spec = IndexSpec::new().vector(srv.col.clone(), dim, metric);
        let table = srv
            .conn
            .create_table(&srv.table, schema, spec)
            .map_err(|e| e.to_string())?;
        let mut g = srv.state.write().unwrap();
        g.table = Some(Arc::new(table));
        g.dim = dim;
        g.id_map = None;
        Ok(())
    })();
    write_status(stream, res)
}

fn do_append(stream: &mut TcpStream, srv: &ServeState) -> std::io::Result<()> {
    let nrows = read_u32(stream)? as usize;
    let dim = read_u32(stream)? as usize;
    // Validate the header against the table's known dimension BEFORE allocating:
    // a mismatched/garbage header would otherwise drive an unbounded read. On a
    // protocol violation the body is left unread, so the stream is desynced —
    // report the error and close the connection rather than try to resync.
    let table_dim = srv.state.read().unwrap().dim;
    if dim != table_dim || nrows > MAX_APPEND_ROWS {
        let _ = write_status(
            stream,
            Err(format!(
                "append header invalid: nrows={nrows}, dim={dim}, table dim={table_dim}"
            )),
        );
        return Err(std::io::Error::other("append header invalid"));
    }
    let row_bytes = 8 + dim * 4;
    let body = read_n(stream, nrows * row_bytes)?;
    let res = (|| -> Result<(), String> {
        let mut ids: Vec<i64> = Vec::with_capacity(nrows);
        let mut vals: Vec<f32> = Vec::with_capacity(nrows * dim);
        for r in body.chunks_exact(row_bytes) {
            ids.push(i64::from_le_bytes(r[0..8].try_into().unwrap()));
            vals.extend(
                r[8..]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
        }
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let emb =
            FixedSizeListArray::new(item, dim as i32, Arc::new(Float32Array::from(vals)), None);
        let schema = build_schema(&srv.id_col, &srv.col, dim);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(ids)), Arc::new(emb)])
                .map_err(|e| e.to_string())?;
        // Writes are serialized under the write lock: the table handle is not
        // multi-writer, and no search overlaps the build phase. The guard is
        // only read here, but the exclusive lock is the point — it bars a
        // concurrent create/drop and any second appender.
        #[allow(clippy::readonly_write_lock)]
        let g = srv.state.write().unwrap();
        let table = g.table.as_ref().ok_or("append before create")?;
        table.append(&batch).map_err(|e| e.to_string())
    })();
    match &res {
        Ok(()) => {
            stream.write_all(&[0u8])?;
            stream.write_all(&(nrows as u64).to_le_bytes())
        }
        Err(_) => write_status(stream, res),
    }
}

fn do_optimize(stream: &mut TcpStream, srv: &ServeState) -> std::io::Result<()> {
    let res = (|| -> Result<(), String> {
        // Hold the write lock across the whole optimize + map build so no
        // concurrent append/create/drop on another connection races the table
        // handle (it is not multi-writer), and so a search cannot observe a
        // half-built id map. Search connections are opened only after optimize
        // completes, so this blocks nothing that should run concurrently.
        let mut g = srv.state.write().unwrap();
        let table = g.table.clone().ok_or("optimize before create")?;
        table
            .optimize(&Default::default())
            .map_err(|e| e.to_string())?;
        // Build the `_id` -> dataset-id map once, here, so every search returns
        // the dataset id from memory instead of projecting the id column
        // (which decodes a parquet row-group per query).
        let sql = format!("SELECT _id, {} FROM {}", srv.id_col, srv.table);
        let batches = srv.conn.query_sql(&sql).map_err(|e| e.to_string())?;
        let mut map: Vec<(i128, i64)> = Vec::new();
        for b in &batches {
            let eng = b
                .column_by_name("_id")
                .and_then(|c| c.as_any().downcast_ref::<Decimal128Array>())
                .ok_or("_id column missing/!decimal128")?;
            let ds = b
                .column_by_name(&srv.id_col)
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                .ok_or("id column missing/!int64")?;
            for i in 0..eng.len() {
                map.push((eng.value(i), ds.value(i)));
            }
        }
        // Sorted by `_id` so do_search resolves each result with a binary search.
        map.sort_unstable_by_key(|&(eng, _)| eng);
        g.id_map = Some(Arc::new(map));
        Ok(())
    })();
    write_status(stream, res)
}

fn do_search(stream: &mut TcpStream, srv: &ServeState) -> std::io::Result<()> {
    let k = read_u32(stream)? as usize;
    let dim = read_u32(stream)? as usize;
    // Bound dim before allocating the query buffer; a garbage header is a
    // protocol error and the stream is already desynced, so close.
    if dim == 0 || dim > MAX_DIM {
        eprintln!("[infino-bench-serve] query dim {dim} out of range; closing");
        return Err(std::io::Error::other("dim out of range"));
    }
    let qbuf = read_n(stream, dim * 4)?;
    let query: Vec<f32> = qbuf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // Build the reply while holding the state read lock, referencing the table
    // and id map in place. An RwLock admits concurrent readers, so parallel
    // searches still proceed together; holding the guard avoids cloning the two
    // Arc handles on every query, whose atomic refcount traffic is a real tail
    // cost under many search threads. The lock is dropped before the network
    // write below.
    let out: Vec<u8> = {
        let g = srv.state.read().unwrap();
        let (Some(table), Some(id_map)) = (g.table.as_ref(), g.id_map.as_ref()) else {
            // Not built/optimized yet — reply empty rather than desync the stream.
            return stream.write_all(&0u32.to_le_bytes());
        };
        if dim != g.dim {
            eprintln!(
                "[infino-bench-serve] query dim {dim} != table dim {}; closing",
                g.dim
            );
            return Err(std::io::Error::other("dim mismatch"));
        }
        let batches = match table.vector_search(&srv.col, &query, k, None, None) {
            Ok(b) => b,
            Err(e) => {
                // A transient engine error fails this one query rather than
                // tearing down the pipelined connection (which would abort the
                // whole run); reply empty so the client scores it a miss.
                eprintln!("[infino-bench-serve] search error (returning empty): {e}");
                return stream.write_all(&0u32.to_le_bytes());
            }
        };
        let mut out: Vec<u8> = Vec::with_capacity(4 + k * 8);
        out.extend_from_slice(&0u32.to_le_bytes());
        let mut n: u32 = 0;
        for b in &batches {
            let Some(dec) = b
                .column_by_name("_id")
                .and_then(|c| c.as_any().downcast_ref::<Decimal128Array>())
            else {
                break;
            };
            for i in 0..dec.len() {
                // Resolve `_id` -> dataset id via the sorted map; drop an id
                // absent from it rather than emit a wrong one (only possible if a
                // row was appended after optimize, which the build protocol avoids).
                if let Ok(pos) = id_map.binary_search_by_key(&dec.value(i), |&(eng, _)| eng) {
                    out.extend_from_slice(&id_map[pos].1.to_le_bytes());
                    n += 1;
                }
            }
        }
        out[0..4].copy_from_slice(&n.to_le_bytes());
        out
    };
    stream.write_all(&out)
}

fn do_drop(stream: &mut TcpStream, srv: &ServeState) -> std::io::Result<()> {
    let res = (|| -> Result<(), String> {
        srv.conn
            .drop_table(&srv.table, true)
            .map_err(|e| e.to_string())?;
        let mut g = srv.state.write().unwrap();
        g.table = None;
        g.id_map = None;
        Ok(())
    })();
    write_status(stream, res)
}

fn handle_build(mut stream: TcpStream, srv: Arc<ServeState>) {
    stream.set_nodelay(true).ok();
    loop {
        let mut op = [0u8; 1];
        if stream.read_exact(&mut op).is_err() {
            break; // client closed
        }
        let r = match op[0] {
            OP_CREATE => do_create(&mut stream, &srv),
            OP_APPEND => do_append(&mut stream, &srv),
            OP_OPTIMIZE => do_optimize(&mut stream, &srv),
            OP_SEARCH => do_search(&mut stream, &srv),
            OP_DROP => do_drop(&mut stream, &srv),
            other => {
                eprintln!("[infino-bench-serve] unknown opcode {other}; closing");
                break;
            }
        };
        if r.is_err() {
            break; // write/read failed — client gone or protocol desync
        }
    }
}

/// EXPERIMENTAL raw-TCP build+serve loop. A client drives create/append/
/// optimize/search over TCP, so build and serve can run on a machine separate
/// from the driving client. Blocks forever. Not a production server.
#[pyfunction]
#[pyo3(signature = (data_path, table, col, id_col, addr, cache_bytes))]
pub fn bench_serve_build_tcp(
    py: Python<'_>,
    data_path: &str,
    table: &str,
    col: &str,
    id_col: &str,
    addr: &str,
    cache_bytes: u64,
) -> PyResult<()> {
    eprintln!(
        "[infino-bench-serve] EXPERIMENTAL build+serve mode — no auth, no TLS, no durability, \
         single table. NOT production-validated; the supported infino interface is embedded."
    );
    let data_path = data_path.to_string();
    let (table, col, id_col, addr) = (
        table.to_string(),
        col.to_string(),
        id_col.to_string(),
        addr.to_string(),
    );
    py.detach(move || -> Result<(), String> {
        let opts = ConnectOptions::new()
            .with_cache_budget_bytes(cache_bytes)
            .with_cache_dir(format!("{data_path}/cache"));
        let conn = connect_with(&data_path, opts).map_err(|e| e.to_string())?;
        let srv = Arc::new(ServeState {
            conn,
            table,
            col,
            id_col,
            state: RwLock::new(TableState {
                table: None,
                dim: 0,
                id_map: None,
            }),
        });
        let listener = TcpListener::bind(&addr).map_err(|e| e.to_string())?;
        eprintln!(
            "[infino-bench-serve] build+serve listening={addr} table={} col={} id_col={} cache={cache_bytes} — ready",
            srv.table, srv.col, srv.id_col
        );
        for stream in listener.incoming() {
            match stream {
                Ok(s) => {
                    let srv = srv.clone();
                    thread::spawn(move || handle_build(s, srv));
                }
                Err(e) => eprintln!("[infino-bench-serve] accept error: {e}"),
            }
        }
        Ok(())
    })
    .map_err(PyRuntimeError::new_err)?;
    Ok(())
}
