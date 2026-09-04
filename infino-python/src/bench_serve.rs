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
//! Two id modes (chosen at start):
//!   - default: return the engine `_id` (16-byte big-endian decimal128); the
//!     client maps it to its dataset id.
//!   - `id_col` set: project that scalar int64 column and return the dataset id
//!     directly (8-byte little-endian) — the client needs no id map.
//!
//! Wire (little-endian, one persistent connection, pipelined requests):
//!   request : u32 k, u32 dim, then dim*4 bytes of f32 query
//!   response: u32 n, then n*IDSIZE bytes  (IDSIZE = 16 for `_id`, 8 for id_col)

use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::Arc,
    thread,
};

use arrow_array::{Array, Decimal128Array, Int64Array};
use arrow_schema::DataType;
use infino::{ConnectOptions, Supertable, connect_with};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

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
        // Validate against the table's known dimension BEFORE allocating: a
        // mismatched/garbage header would otherwise drive an unbounded alloc and
        // then a swallowed search error. Fail loud instead.
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
        // Serialize ids: dataset int64 (8B) when id_col is set, else engine _id (16B).
        let mut out: Vec<u8> = Vec::with_capacity(4 + k * 16);
        out.extend_from_slice(&0u32.to_le_bytes()); // n placeholder
        let mut n: u32 = 0;
        for b in &batches {
            if let Some(ic) = id_col.as_deref() {
                // Column presence + Int64 type are validated at startup, so a miss
                // here is a bug, not a config error — fail loud rather than return a
                // short/empty result the client would score as ~0 recall silently.
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

/// EXPERIMENTAL raw-TCP benchmark serve loop. Opens `table` under `data_path`
/// and serves top-k ids over TCP. Blocks forever. Not a production server
/// (no auth/TLS/durability) — it is invoked by the benchmark client. When
/// `id_col` is non-empty, results are the dataset int64 ids from that column
/// (no client-side id map); otherwise the engine `_id`.
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
    // Serve with the GIL released — the loop is pure Rust (no Python).
    py.detach(move || -> Result<(), String> {
        let opts = ConnectOptions::new()
            .with_cache_budget_bytes(cache_bytes)
            .with_cache_dir(format!("{data_path}/cache"));
        let conn = connect_with(&data_path, opts).map_err(|e| e.to_string())?;
        let st = Arc::new(conn.open_table(&table).map_err(|e| e.to_string())?);
        // Read the fixed vector dimension from the schema so the warm-up query and
        // the per-request length check use the table's real dim (not a hard-coded
        // one), and confirm the id column is the Int64 the wire format assumes —
        // both are startup invariants, so a mismatch should fail here, loudly,
        // not silently mid-benchmark.
        let schema = st.schema();
        let vfield = schema
            .field_with_name(&col)
            .map_err(|e| format!("vector column {col:?} not in schema: {e}"))?;
        let dim = match vfield.data_type() {
            DataType::FixedSizeList(_, n) => *n as usize,
            other => return Err(format!("vector column {col:?} is {other:?}, expected FixedSizeList<Float32>")),
        };
        if let Some(ic) = id_col.as_deref() {
            let ifield = schema
                .field_with_name(ic)
                .map_err(|e| format!("id column {ic:?} not in schema: {e}"))?;
            if !matches!(ifield.data_type(), DataType::Int64) {
                return Err(format!("id column {ic:?} is {:?}, expected Int64", ifield.data_type()));
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
