// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Binary encodings for the per-superfile skip-summary types
//! that ride inside the manifest-part Avro schema as opaque
//! `bytes` fields.
//!
//! The Avro layer doesn't need to introspect these — the
//! aggregate skip pruning at the manifest-list level uses
//! the parent-level aggregates, not the per-superfile bytes;
//! the per-superfile summaries are loaded into memory by the
//! manifest-part decoder and consumed by the superfile-level
//! prune path.
//!
//! Three encodings, all little-endian, all designed for
//! bit-exact round-trip of floats (no `f32 → str → f32`
//! through a decimal representation):
//!
//! - [`encode_scalar_stats`] / [`decode_scalar_stats`] —
//!   Arrow IPC bytes for the per-column min/max table.
//! - [`encode_fts_summary`] / [`decode_fts_summary`] —
//!   custom packed: bloom bytes (already
//!   [`Bloom::to_bytes`] / [`Bloom::from_bytes`] symmetric),
//!   `n_terms_distinct` as LE u32, term-range min and max
//!   as length-prefixed bytes.
//! - [`encode_vector_summary`] / [`decode_vector_summary`] —
//!   custom packed: dim (LE u32), centroid (dim × LE f32),
//!   radius (LE f32).
//!
//! Wrapped variants — [`encode_fts_summary_map`] /
//! [`encode_vector_summary_map`] — emit the
//! `HashMap<String, T>` shape the in-memory `SuperfileEntry`
//! carries.
//!
//! All decode functions return a [`DecodeError`] on shape
//! mismatch; callers (the manifest part decoder) wrap that
//! into [`OpenError::ManifestPartParse`].

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{Field, Schema};
use thiserror::Error;

use crate::supertable::manifest::bloom::Bloom;
use crate::supertable::manifest::list::ScalarStatsAgg;
use crate::supertable::manifest::{FtsSummaryAgg, VectorSummary};

/// Errors from the per-summary binary decoders.
///
/// The manifest-part decoder catches these and wraps them in
/// `OpenError::ManifestPartParse` so the supertable layer
/// surfaces a single uniform parse-error variant.
#[derive(Debug, Error)]
pub enum DecodeError {
    /// Input buffer is shorter than the fixed-width prefix
    /// the encoding requires (e.g., a 4-byte length header).
    #[error("truncated input: needed {needed} bytes for {what}, had {had}")]
    Truncated {
        what: &'static str,
        needed: usize,
        had: usize,
    },

    /// Bloom byte length isn't a valid `n_blocks × BLOCK_BYTES`
    /// power-of-two — see `Bloom::from_bytes` for the rule.
    #[error("invalid bloom layout: {0} bytes")]
    InvalidBloomLayout(usize),

    /// Vector dim or centroid bytes mismatch.
    #[error("invalid vector summary: {0}")]
    InvalidVectorSummary(String),

    /// Term range was inverted (`min_term > max_term`). Both-empty is the
    /// legal "no range" sentinel and an empty min is fine (the empty string
    /// is lex-smallest), but an out-of-order pair is corrupt input that
    /// would break prefix-overlap pruning.
    #[error("invalid term range: min_term > max_term (inverted range)")]
    InvalidTermRange,

    /// Arrow IPC parse failed.
    #[error("arrow ipc parse failed: {0}")]
    ArrowIpc(String),

    /// Arrow IPC stream produced zero batches where one was
    /// expected (or more than one).
    #[error("expected exactly one arrow ipc batch, got {0}")]
    UnexpectedBatchCount(usize),
}

/// Errors from the per-summary binary encoders.
///
/// Surfaced by [`encode_length1_array`] when the Arrow IPC writer rejects
/// an array; the manifest-list encoder wraps it into its own encode error
/// so a commit fails cleanly rather than panicking.
#[derive(Debug, Error)]
pub enum EncodeError {
    /// Arrow IPC serialization failed.
    #[error("arrow ipc encode failed: {0}")]
    ArrowIpc(String),

    /// A length-1 array was expected (an aggregate is a single value) but
    /// the array had a different row count — caught before persisting so a
    /// malformed manifest is never written.
    #[error("expected a length-1 array, got {0} rows")]
    WrongRowCount(usize),
}

// ---------------------------------------------------------
// Scalar stats (`HashMap<String, ScalarStatsAgg>`): arrow-ipc encoding.
// ---------------------------------------------------------
//
// One RecordBatch carries every column's stats as length-1
// columns named by suffix: `<col>__min` / `<col>__max`
// (always, paired), plus optional `<col>__nulls` (UInt64),
// `<col>__sum` (the column's SUM result type) and
// `<col>__hll` (Binary, raw HLL registers). The logical
// schema is reconstructed at decode time by stripping the
// suffixes; data types are preserved by the IPC format
// itself. Decoding tolerates absent optional stats (segments
// written before they existed), never inventing values.

const MIN_SUFFIX: &str = "__min";
const MAX_SUFFIX: &str = "__max";
const NULLS_SUFFIX: &str = "__nulls";
const SUM_SUFFIX: &str = "__sum";
const HLL_SUFFIX: &str = "__hll";

pub fn encode_scalar_stats(stats: &HashMap<String, ScalarStatsAgg>) -> Vec<u8> {
    if stats.is_empty() {
        // Empty table → emit a sentinel zero-length blob.
        // Decode treats that as an empty map.
        return Vec::new();
    }
    // Sort columns for deterministic output. The order
    // doesn't matter for correctness but makes diffs +
    // content-addressing stable.
    let mut keys: Vec<&String> = stats.keys().collect();
    keys.sort();

    let mut fields: Vec<Field> = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for key in keys {
        let agg = &stats[key];
        fields.push(Field::new(
            format!("{key}{MIN_SUFFIX}"),
            agg.min.data_type().clone(),
            true,
        ));
        fields.push(Field::new(
            format!("{key}{MAX_SUFFIX}"),
            agg.max.data_type().clone(),
            true,
        ));
        arrays.push(agg.min.clone());
        arrays.push(agg.max.clone());
        if let Some(nulls) = agg.null_count {
            fields.push(Field::new(
                format!("{key}{NULLS_SUFFIX}"),
                arrow_schema::DataType::UInt64,
                true,
            ));
            arrays.push(Arc::new(arrow_array::UInt64Array::from(vec![nulls])) as ArrayRef);
        }
        if let Some(sum) = &agg.sum {
            fields.push(Field::new(
                format!("{key}{SUM_SUFFIX}"),
                sum.data_type().clone(),
                true,
            ));
            arrays.push(sum.clone());
        }
        if let Some(sketch) = &agg.hll {
            fields.push(Field::new(
                format!("{key}{HLL_SUFFIX}"),
                arrow_schema::DataType::Binary,
                true,
            ));
            arrays.push(
                Arc::new(arrow_array::BinaryArray::from(vec![sketch.as_slice()])) as ArrayRef,
            );
        }
    }
    let schema = Arc::new(Schema::new(fields));
    let batch =
        RecordBatch::try_new(schema.clone(), arrays).expect("schema/array match by construction");

    let mut out = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut out, &schema).expect("ipc writer init");
        writer.write(&batch).expect("ipc write");
        writer.finish().expect("ipc finish");
    }
    out
}

pub fn decode_scalar_stats(bytes: &[u8]) -> Result<HashMap<String, ScalarStatsAgg>, DecodeError> {
    if bytes.is_empty() {
        return Ok(HashMap::new());
    }
    let reader = StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|e| DecodeError::ArrowIpc(e.to_string()))?;
    let batches: Vec<RecordBatch> = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| DecodeError::ArrowIpc(e.to_string()))?;
    if batches.len() != 1 {
        return Err(DecodeError::UnexpectedBatchCount(batches.len()));
    }
    let batch = &batches[0];
    let schema = batch.schema();

    // Bucket fields by stripped base name; min/max must pair up,
    // everything else is optional. Assembled into per-column
    // `ScalarStatsAgg` once every field is seen.
    let mut mins: HashMap<String, ArrayRef> = HashMap::new();
    let mut maxes: HashMap<String, ArrayRef> = HashMap::new();
    let mut null_counts: HashMap<String, u64> = HashMap::new();
    let mut sums: HashMap<String, ArrayRef> = HashMap::new();
    let mut hlls: HashMap<String, Vec<u8>> = HashMap::new();
    for (i, field) in schema.fields().iter().enumerate() {
        let name = field.name();
        let column = batch.column(i);
        if let Some(base) = name.strip_suffix(MIN_SUFFIX) {
            mins.insert(base.to_string(), column.clone());
        } else if let Some(base) = name.strip_suffix(MAX_SUFFIX) {
            maxes.insert(base.to_string(), column.clone());
        } else if let Some(base) = name.strip_suffix(NULLS_SUFFIX) {
            let arr = column
                .as_any()
                .downcast_ref::<arrow_array::UInt64Array>()
                .ok_or_else(|| {
                    DecodeError::ArrowIpc(format!("{name}: __nulls column is not UInt64"))
                })?;
            if !arr.is_empty() && !arr.is_null(0) {
                null_counts.insert(base.to_string(), arr.value(0));
            }
        } else if let Some(base) = name.strip_suffix(SUM_SUFFIX) {
            sums.insert(base.to_string(), column.clone());
        } else if let Some(base) = name.strip_suffix(HLL_SUFFIX) {
            let arr = column
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .ok_or_else(|| {
                    DecodeError::ArrowIpc(format!("{name}: __hll column is not Binary"))
                })?;
            if !arr.is_empty() && !arr.is_null(0) {
                hlls.insert(base.to_string(), arr.value(0).to_vec());
            }
        } else {
            return Err(DecodeError::ArrowIpc(format!(
                "unrecognized stats column suffix: {name}"
            )));
        }
    }
    if mins.len() != maxes.len() {
        return Err(DecodeError::ArrowIpc(format!(
            "unpaired __min/__max columns: {} mins vs {} maxes",
            mins.len(),
            maxes.len()
        )));
    }
    let mut stats: HashMap<String, ScalarStatsAgg> = HashMap::new();
    for (base, min) in mins {
        let max = maxes.remove(&base).ok_or_else(|| {
            DecodeError::ArrowIpc(format!("column {base} has __min but no __max"))
        })?;
        let null_count = null_counts.remove(&base);
        let sum = sums.remove(&base);
        let hll = hlls.remove(&base);
        stats.insert(
            base,
            ScalarStatsAgg {
                min,
                max,
                null_count,
                sum,
                hll,
            },
        );
    }
    // Each matched base was `remove`d from the optional maps above, so a
    // leftover entry is a `__nulls` / `__sum` / `__hll` field whose base
    // column carries no `__min`/`__max` pair. Valid data always pairs
    // optionals with min/max, so an orphan signals corruption — reject it
    // rather than silently dropping it (which would hide bad manifest data
    // and yield incomplete statistics).
    if let Some(base) = null_counts
        .keys()
        .chain(sums.keys())
        .chain(hlls.keys())
        .next()
    {
        return Err(DecodeError::ArrowIpc(format!(
            "orphan optional stat for column {base} with no __min/__max pair"
        )));
    }
    Ok(stats)
}

/// Encode a single length-1 [`ArrayRef`] as Arrow-IPC stream bytes —
/// the `ScalarStatsAgg.{min,max,sum}` wire shape (one batch, one
/// column, one row). `field_name` is recorded in the IPC schema only;
/// decoders read column 0 by position and ignore the name. Output is
/// deterministic for identical inputs, which the manifest list's
/// content-addressing relies on.
///
/// Returns [`EncodeError::WrongRowCount`] if `arr` is not exactly one row
/// (an aggregate is a single value), or [`EncodeError::ArrowIpc`] if the
/// Arrow IPC writer rejects the array (e.g. an unsupported data type). In
/// practice the schema is built from the array's own `data_type`, so the
/// batch always matches — but failures are surfaced rather than panicked so
/// a bad array can't abort a commit.
pub(crate) fn encode_length1_array(
    field_name: &str,
    arr: &ArrayRef,
) -> Result<Vec<u8>, EncodeError> {
    // Enforce the single-value contract before writing — fail fast here
    // rather than persisting a manifest that `decode_length1_array` would
    // later reject.
    if arr.len() != 1 {
        return Err(EncodeError::WrongRowCount(arr.len()));
    }
    let field = Field::new(field_name, arr.data_type().clone(), true);
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![arr.clone()])
        .map_err(|e| EncodeError::ArrowIpc(e.to_string()))?;
    let mut out = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut out, &schema)
            .map_err(|e| EncodeError::ArrowIpc(e.to_string()))?;
        writer
            .write(&batch)
            .map_err(|e| EncodeError::ArrowIpc(e.to_string()))?;
        writer
            .finish()
            .map_err(|e| EncodeError::ArrowIpc(e.to_string()))?;
    }
    Ok(out)
}

/// Decode the bytes produced by [`encode_length1_array`] back into the
/// single length-1 [`ArrayRef`] (column 0 of the one-batch stream).
pub(crate) fn decode_length1_array(bytes: &[u8]) -> Result<ArrayRef, DecodeError> {
    let reader = StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|e| DecodeError::ArrowIpc(e.to_string()))?;
    let batches: Vec<RecordBatch> = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| DecodeError::ArrowIpc(e.to_string()))?;
    if batches.len() != 1 {
        return Err(DecodeError::UnexpectedBatchCount(batches.len()));
    }
    let batch = &batches[0];
    if batch.num_columns() != 1 {
        return Err(DecodeError::ArrowIpc(format!(
            "expected exactly one column, got {}",
            batch.num_columns()
        )));
    }
    // The aggregate is a single value; callers read index 0 as the only
    // element. Reject any other row count so malformed manifest data fails
    // loudly here rather than silently yielding a wrong min/max (which would
    // corrupt list-level prune decisions).
    if batch.num_rows() != 1 {
        return Err(DecodeError::ArrowIpc(format!(
            "expected exactly one row, got {}",
            batch.num_rows()
        )));
    }
    Ok(batch.column(0).clone())
}

// ---------------------------------------------------------
// FtsSummaryAgg: custom packed.
//
// Layout (all LE):
//   u32 bloom_len                  (== n_blocks × BLOCK_BYTES; 0 ⇒ no bloom)
//   [bloom_len bytes]              (Bloom::to_bytes output)
//   u32 n_terms_distinct           (per-superfile count fits u32; widened to
//                                   u64 in memory)
//   u32 min_term_len
//   [min_term bytes]
//   u32 max_term_len
//   [max_term bytes]               (empty min+max ⇒ no range, i.e. None)
//
// ---------------------------------------------------------

pub fn encode_fts_summary(s: &FtsSummaryAgg) -> Vec<u8> {
    let bloom_bytes = s
        .term_bloom
        .as_ref()
        .map(|b| b.to_bytes())
        .unwrap_or_default();
    // `None` range encodes as empty min+max (the 0-term / no-range sentinel).
    let (min_term, max_term): (&[u8], &[u8]) = match &s.term_range {
        Some((mn, mx)) => (mn, mx),
        None => (&[], &[]),
    };
    let cap = 4 + bloom_bytes.len() + 4 + 4 + min_term.len() + 4 + max_term.len();
    let mut out = Vec::with_capacity(cap);
    out.extend_from_slice(&(bloom_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&bloom_bytes);
    // Keep the wire field u32 (a single superfile's distinct count fits) so
    // the part format is unchanged; saturate the u64 in-memory value.
    out.extend_from_slice(
        &u32::try_from(s.n_terms_distinct)
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    out.extend_from_slice(&(min_term.len() as u32).to_le_bytes());
    out.extend_from_slice(min_term);
    out.extend_from_slice(&(max_term.len() as u32).to_le_bytes());
    out.extend_from_slice(max_term);
    out
}

pub fn decode_fts_summary(bytes: &[u8]) -> Result<FtsSummaryAgg, DecodeError> {
    let mut c = Cursor::new(bytes);
    let bloom_len = read_u32(&mut c, "bloom_len")? as usize;
    let bloom_bytes = read_n(&mut c, bloom_len, "bloom_bytes")?;
    // Empty bloom run ⇒ "no bloom info" (None); a non-empty run must be a
    // valid bloom layout.
    let term_bloom = if bloom_bytes.is_empty() {
        None
    } else {
        Some(Bloom::from_bytes(&bloom_bytes).ok_or(DecodeError::InvalidBloomLayout(bloom_len))?)
    };
    let n_terms_distinct = u64::from(read_u32(&mut c, "n_terms_distinct")?);
    let min_len = read_u32(&mut c, "min_term_len")? as usize;
    let min_term = read_n(&mut c, min_len, "min_term")?;
    let max_len = read_u32(&mut c, "max_term_len")? as usize;
    let max_term = read_n(&mut c, max_len, "max_term")?;
    // Both-empty is the "no range" sentinel (None). Otherwise it's a real
    // `[min, max]` — which must be ordered. An empty min is legal (the empty
    // string is lex-smallest), but `min > max` is an inverted, corrupt range
    // that would break prefix-overlap pruning, so reject it.
    let term_range = if min_term.is_empty() && max_term.is_empty() {
        None
    } else if min_term <= max_term {
        Some((min_term, max_term))
    } else {
        return Err(DecodeError::InvalidTermRange);
    };
    Ok(FtsSummaryAgg {
        term_bloom,
        n_terms_distinct,
        term_range,
    })
}

// ---------------------------------------------------------
// VectorSummary: custom packed.
//
// Layout (all LE):
//   u32 dim
//   [dim × f32]   (centroid)
//   f32 radius
// ---------------------------------------------------------

pub fn encode_vector_summary(s: &VectorSummary) -> Vec<u8> {
    let dim = s.centroid.len();
    let cl = &s.clusters;
    let nc = cl.n_cent as usize;
    let cd = cl.dim as usize;
    let mut out = Vec::with_capacity(4 + dim * 4 + 4 + 8 + nc * (4 + 4 + 4) + nc * cd);
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    for &v in &s.centroid {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&s.radius.to_le_bytes());
    // Per-cluster centroid block: n_cent, dim, then counts / mins /
    // scales / Sq8 codes. `n_cent == 0` encodes a superfile with no
    // vector index for the column (empty trailer).
    out.extend_from_slice(&cl.n_cent.to_le_bytes());
    out.extend_from_slice(&cl.dim.to_le_bytes());
    for &c in &cl.counts {
        out.extend_from_slice(&c.to_le_bytes());
    }
    for &m in &cl.mins {
        out.extend_from_slice(&m.to_le_bytes());
    }
    for &sc in &cl.scales {
        out.extend_from_slice(&sc.to_le_bytes());
    }
    out.extend_from_slice(&cl.codes);
    out
}

pub fn decode_vector_summary(bytes: &[u8]) -> Result<VectorSummary, DecodeError> {
    let mut c = Cursor::new(bytes);
    let dim = read_u32(&mut c, "dim")? as usize;
    let mut centroid = Vec::with_capacity(dim);
    for i in 0..dim {
        let b = read_n(&mut c, 4, "centroid_float")?;
        if b.len() != 4 {
            return Err(DecodeError::InvalidVectorSummary(format!(
                "truncated centroid at index {i}"
            )));
        }
        let arr = [b[0], b[1], b[2], b[3]];
        centroid.push(f32::from_le_bytes(arr));
    }
    let rb = read_n(&mut c, 4, "radius")?;
    if rb.len() != 4 {
        return Err(DecodeError::InvalidVectorSummary("truncated radius".into()));
    }
    let radius = f32::from_le_bytes([rb[0], rb[1], rb[2], rb[3]]);

    // Per-cluster centroid block (new-engine format). `n_cent == 0` is
    // a superfile with no vector index for the column.
    let n_cent = read_u32(&mut c, "cluster_n_cent")? as usize;
    let cdim = read_u32(&mut c, "cluster_dim")? as usize;

    let counts_b = read_n(&mut c, n_cent * 4, "cluster_counts")?;
    if counts_b.len() != n_cent * 4 {
        return Err(DecodeError::InvalidVectorSummary(
            "truncated cluster counts".into(),
        ));
    }
    let counts: Vec<u32> = counts_b
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let ms_b = read_n(&mut c, n_cent * 8, "cluster_min_scale")?;
    if ms_b.len() != n_cent * 8 {
        return Err(DecodeError::InvalidVectorSummary(
            "truncated cluster min/scale".into(),
        ));
    }
    let floats: Vec<f32> = ms_b
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let mins = floats[0..n_cent].to_vec();
    let scales = floats[n_cent..2 * n_cent].to_vec();

    let codes_b = read_n(&mut c, n_cent * cdim, "cluster_codes")?;
    if codes_b.len() != n_cent * cdim {
        return Err(DecodeError::InvalidVectorSummary(
            "truncated cluster codes".into(),
        ));
    }
    let codes = codes_b.to_vec();

    Ok(VectorSummary {
        centroid,
        radius,
        clusters: super::ClusterCentroids {
            n_cent: n_cent as u32,
            dim: cdim as u32,
            codes,
            mins,
            scales,
            counts,
            code_moments: std::sync::OnceLock::new(),
        },
    })
}

// ---------------------------------------------------------
// Map-of-summary wrappers.
//
// Layout (all LE):
//   u32 n_entries
//   for each entry:
//     u32 key_len
//     [key_len bytes]    (column name, UTF-8)
//     u32 value_len
//     [value_len bytes]  (encode_<inner>)
// ---------------------------------------------------------

pub fn encode_fts_summary_map(map: &HashMap<String, FtsSummaryAgg>) -> Vec<u8> {
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    let mut out = Vec::new();
    out.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    for k in keys {
        let key_bytes = k.as_bytes();
        let value_bytes = encode_fts_summary(&map[k]);
        out.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(key_bytes);
        out.extend_from_slice(&(value_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&value_bytes);
    }
    out
}

pub fn decode_fts_summary_map(bytes: &[u8]) -> Result<HashMap<String, FtsSummaryAgg>, DecodeError> {
    let mut c = Cursor::new(bytes);
    let n = read_u32(&mut c, "fts_map_n")? as usize;
    let mut out = HashMap::with_capacity(n);
    for _ in 0..n {
        let kl = read_u32(&mut c, "fts_key_len")? as usize;
        let k = read_n(&mut c, kl, "fts_key")?;
        let key = String::from_utf8(k)
            .map_err(|e| DecodeError::ArrowIpc(format!("fts key utf-8: {e}")))?;
        let vl = read_u32(&mut c, "fts_value_len")? as usize;
        let v = read_n(&mut c, vl, "fts_value")?;
        out.insert(key, decode_fts_summary(&v)?);
    }
    Ok(out)
}

pub fn encode_vector_summary_map(map: &HashMap<String, VectorSummary>) -> Vec<u8> {
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    let mut out = Vec::new();
    out.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    for k in keys {
        let key_bytes = k.as_bytes();
        let value_bytes = encode_vector_summary(&map[k]);
        out.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(key_bytes);
        out.extend_from_slice(&(value_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&value_bytes);
    }
    out
}

pub fn decode_vector_summary_map(
    bytes: &[u8],
) -> Result<HashMap<String, VectorSummary>, DecodeError> {
    let mut c = Cursor::new(bytes);
    let n = read_u32(&mut c, "vec_map_n")? as usize;
    let mut out = HashMap::with_capacity(n);
    for _ in 0..n {
        let kl = read_u32(&mut c, "vec_key_len")? as usize;
        let k = read_n(&mut c, kl, "vec_key")?;
        let key = String::from_utf8(k)
            .map_err(|e| DecodeError::ArrowIpc(format!("vec key utf-8: {e}")))?;
        let vl = read_u32(&mut c, "vec_value_len")? as usize;
        let v = read_n(&mut c, vl, "vec_value")?;
        out.insert(key, decode_vector_summary(&v)?);
    }
    Ok(out)
}

// ---------------------------------------------------------
// Cursor helpers.
// ---------------------------------------------------------

fn read_u32(c: &mut Cursor<&[u8]>, what: &'static str) -> Result<u32, DecodeError> {
    let b = read_n(c, 4, what)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_n(c: &mut Cursor<&[u8]>, n: usize, what: &'static str) -> Result<Vec<u8>, DecodeError> {
    let pos = c.position() as usize;
    let buf = *c.get_ref();
    if pos + n > buf.len() {
        return Err(DecodeError::Truncated {
            what,
            needed: n,
            had: buf.len().saturating_sub(pos),
        });
    }
    let out = buf[pos..pos + n].to_vec();
    c.set_position((pos + n) as u64);
    Ok(out)
}

#[cfg(test)]
mod decode_error_tests {
    //! Exercise the error/empty branches of the binary decoders so the
    //! shape-mismatch paths (truncation, bad arrow columns, unpaired
    //! min/max, non-UTF-8 map keys) are covered, not just the happy
    //! round-trips.
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    use super::{
        DecodeError, ScalarStatsAgg, decode_fts_summary, decode_fts_summary_map,
        decode_length1_array, decode_scalar_stats, decode_vector_summary,
        decode_vector_summary_map, encode_length1_array, encode_scalar_stats, read_n, read_u32,
    };

    /// Hand-build a `decode_fts_summary` payload: no bloom, a given
    /// distinct count, then the `(min_term, max_term)` pair verbatim.
    /// Lets a test plant a half-empty range the encoder would never emit.
    fn fts_summary_bytes(n_terms_distinct: u32, min_term: &[u8], max_term: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0u32.to_le_bytes()); // bloom_len = 0 ⇒ no bloom
        out.extend_from_slice(&n_terms_distinct.to_le_bytes());
        out.extend_from_slice(&(min_term.len() as u32).to_le_bytes());
        out.extend_from_slice(min_term);
        out.extend_from_slice(&(max_term.len() as u32).to_le_bytes());
        out.extend_from_slice(max_term);
        out
    }

    /// One length-1 arrow-IPC RecordBatch with the given fields/columns,
    /// matching the on-wire shape `encode_scalar_stats` emits.
    fn ipc_batch(fields: Vec<Field>, arrays: Vec<ArrayRef>) -> Vec<u8> {
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), arrays).expect("batch");
        let mut out = Vec::new();
        {
            let mut w =
                arrow::ipc::writer::StreamWriter::try_new(&mut out, &schema).expect("ipc init");
            w.write(&batch).expect("ipc write");
            w.finish().expect("ipc finish");
        }
        out
    }

    /// An empty blob decodes to an empty table (the zero-length sentinel
    /// `encode_scalar_stats` emits for an empty input).
    #[test]
    fn decode_scalar_stats_empty_is_empty_table() {
        let table = decode_scalar_stats(&[]).expect("empty");
        assert!(table.is_empty());
        // The empty table round-trips back to a zero-length blob.
        assert!(encode_scalar_stats(&HashMap::new()).is_empty());
    }

    /// Full encode → decode round-trip of a populated table, with columns
    /// exercising every optional-field combination (min/max only; +nulls;
    /// +sum; all of them). Confirms the suffixed-column wire format and the
    /// decode assembly reconstruct each per-column `ScalarStatsAgg` exactly,
    /// including which optional stats are present vs absent.
    #[test]
    fn encode_decode_scalar_stats_round_trips_all_optional_field_combos() {
        let i64_arr = |v: i64| Arc::new(Int64Array::from(vec![v])) as ArrayRef;
        let str_arr = |v: &str| Arc::new(StringArray::from(vec![v])) as ArrayRef;

        let mut table: HashMap<String, ScalarStatsAgg> = HashMap::new();
        // Every stat present.
        table.insert(
            "full".into(),
            ScalarStatsAgg {
                min: i64_arr(1),
                max: i64_arr(100),
                null_count: Some(7),
                sum: Some(i64_arr(5050)),
                hll: Some(vec![0xde, 0xad, 0xbe, 0xef]),
            },
        );
        // Min/max only (the `from_min_max` shape).
        table.insert(
            "bounds_only".into(),
            ScalarStatsAgg::from_min_max(str_arr("alpha"), str_arr("omega")),
        );
        // Nulls but no sum/hll (e.g. a non-summable type that still counts nulls).
        table.insert(
            "nulls_no_sum".into(),
            ScalarStatsAgg {
                min: i64_arr(-3),
                max: i64_arr(9),
                null_count: Some(2),
                sum: None,
                hll: None,
            },
        );

        let decoded = decode_scalar_stats(&encode_scalar_stats(&table)).expect("round-trip");
        assert_eq!(decoded, table);
    }

    /// Garbage (non-arrow-IPC) bytes surface an `ArrowIpc` decode error.
    #[test]
    fn decode_scalar_stats_garbage_is_arrow_ipc_error() {
        let err = decode_scalar_stats(b"definitely not arrow ipc").expect_err("garbage");
        assert!(matches!(err, DecodeError::ArrowIpc(_)), "got {err:?}");
    }

    /// A `__nulls` column typed as something other than UInt64 is rejected.
    #[test]
    fn decode_scalar_stats_wrong_nulls_type_errors() {
        let bytes = ipc_batch(
            vec![Field::new("c__nulls", DataType::Int64, true)],
            vec![Arc::new(Int64Array::from(vec![1])) as ArrayRef],
        );
        let err = decode_scalar_stats(&bytes).expect_err("bad nulls type");
        assert!(matches!(err, DecodeError::ArrowIpc(_)), "got {err:?}");
    }

    /// A column whose name carries no recognized stats suffix is rejected.
    #[test]
    fn decode_scalar_stats_unknown_suffix_errors() {
        let bytes = ipc_batch(
            vec![Field::new("c__bogus", DataType::Int64, true)],
            vec![Arc::new(Int64Array::from(vec![1])) as ArrayRef],
        );
        let err = decode_scalar_stats(&bytes).expect_err("bad suffix");
        assert!(matches!(err, DecodeError::ArrowIpc(_)), "got {err:?}");
    }

    /// A single length-1 array round-trips through encode/decode.
    #[test]
    fn decode_length1_array_round_trips_single_row() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![42]));
        let bytes = encode_length1_array("v", &arr).expect("encode");
        let decoded = decode_length1_array(&bytes).expect("decode");
        assert_eq!(decoded.to_data(), arr.to_data());
    }

    /// Encoding a non-length-1 array fails fast, before any bytes are
    /// written, so an invalid aggregate never reaches a persisted manifest.
    #[test]
    fn encode_length1_array_rejects_non_single_row() {
        use super::EncodeError;
        let multi: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        let err = encode_length1_array("v", &multi).expect_err("multi-row");
        assert!(matches!(err, EncodeError::WrongRowCount(2)), "got {err:?}");

        let empty: ArrayRef = Arc::new(Int64Array::from(Vec::<i64>::new()));
        let err = encode_length1_array("v", &empty).expect_err("zero-row");
        assert!(matches!(err, EncodeError::WrongRowCount(0)), "got {err:?}");
    }

    /// A multi-row batch is rejected — callers read index 0 as the only
    /// element, so accepting >1 rows would silently produce a wrong
    /// aggregate min/max and corrupt list-level prune decisions.
    #[test]
    fn decode_length1_array_rejects_multi_row() {
        let bytes = ipc_batch(
            vec![Field::new("v", DataType::Int64, true)],
            vec![Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
        );
        let err = decode_length1_array(&bytes).expect_err("multi-row");
        assert!(matches!(err, DecodeError::ArrowIpc(_)), "got {err:?}");
    }

    /// A zero-row batch is rejected for the same single-value contract.
    #[test]
    fn decode_length1_array_rejects_zero_row() {
        let bytes = ipc_batch(
            vec![Field::new("v", DataType::Int64, true)],
            vec![Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef],
        );
        let err = decode_length1_array(&bytes).expect_err("zero-row");
        assert!(matches!(err, DecodeError::ArrowIpc(_)), "got {err:?}");
    }

    /// A batch with more than one column is rejected — the aggregate is a
    /// single value, read from column 0.
    #[test]
    fn decode_length1_array_rejects_multi_column() {
        let bytes = ipc_batch(
            vec![
                Field::new("a", DataType::Int64, true),
                Field::new("b", DataType::Int64, true),
            ],
            vec![
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                Arc::new(Int64Array::from(vec![2])) as ArrayRef,
            ],
        );
        let err = decode_length1_array(&bytes).expect_err("multi-column");
        assert!(matches!(err, DecodeError::ArrowIpc(_)), "got {err:?}");
    }

    /// A stream carrying more than one batch is rejected with the
    /// dedicated `UnexpectedBatchCount` error.
    #[test]
    fn decode_length1_array_rejects_multi_batch() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![1])) as ArrayRef],
        )
        .expect("batch");
        let mut out = Vec::new();
        {
            let mut w =
                arrow::ipc::writer::StreamWriter::try_new(&mut out, &schema).expect("ipc init");
            w.write(&batch).expect("write 1");
            w.write(&batch).expect("write 2");
            w.finish().expect("finish");
        }
        let err = decode_length1_array(&out).expect_err("two batches");
        assert!(
            matches!(err, DecodeError::UnexpectedBatchCount(2)),
            "got {err:?}"
        );
    }

    /// Garbage (non-arrow-IPC) bytes surface an `ArrowIpc` error from the
    /// length-1 decoder's reader-init path.
    #[test]
    fn decode_length1_array_garbage_is_arrow_ipc_error() {
        let err = decode_length1_array(b"definitely not arrow ipc").expect_err("garbage");
        assert!(matches!(err, DecodeError::ArrowIpc(_)), "got {err:?}");
    }

    /// A `__min` without a matching `__max` is an unpaired-column error.
    #[test]
    fn decode_scalar_stats_unpaired_min_errors() {
        let bytes = ipc_batch(
            vec![Field::new("c__min", DataType::Utf8, true)],
            vec![Arc::new(StringArray::from(vec!["a"])) as ArrayRef],
        );
        let err = decode_scalar_stats(&bytes).expect_err("unpaired min");
        assert!(matches!(err, DecodeError::ArrowIpc(_)), "got {err:?}");
    }

    /// Truncated bytes (here a too-short vector summary) surface a
    /// `Truncated` error via the cursor helpers.
    #[test]
    fn decode_vector_summary_truncated_errors() {
        // Claims dim = 4 (LE u32) but supplies no centroid bytes.
        let bytes = 4u32.to_le_bytes().to_vec();
        let err = decode_vector_summary(&bytes).expect_err("truncated");
        assert!(matches!(err, DecodeError::Truncated { .. }), "got {err:?}");
    }

    /// A map whose key bytes are not valid UTF-8 surfaces a decode error
    /// rather than panicking.
    #[test]
    fn decode_summary_maps_reject_non_utf8_keys() {
        // n_entries = 1, key_len = 1, key = 0xff (invalid UTF-8).
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.push(0xff);
        let fts_err = decode_fts_summary_map(&bytes).expect_err("bad fts key");
        assert!(
            matches!(fts_err, DecodeError::ArrowIpc(_)),
            "got {fts_err:?}"
        );
        let vec_err = decode_vector_summary_map(&bytes).expect_err("bad vec key");
        assert!(
            matches!(vec_err, DecodeError::ArrowIpc(_)),
            "got {vec_err:?}"
        );
    }

    /// An inverted term range (`min_term > max_term`) is corrupt input and
    /// must surface `InvalidTermRange`, not deserialize into a `Some` that
    /// would break prefix-overlap pruning. `min_term = "abc", max_term = ""`
    /// is such a pair (`"" < "abc"`).
    #[test]
    fn decode_fts_summary_inverted_term_range_errors() {
        let inverted = fts_summary_bytes(0, b"abc", b"");
        let err = decode_fts_summary(&inverted).expect_err("min > max");
        assert!(matches!(err, DecodeError::InvalidTermRange), "got {err:?}");
    }

    /// The legal term-range encodings decode as expected: both bounds empty
    /// ⇒ `None`; both present ⇒ `Some`; and an **empty min** with a present
    /// max is valid (the empty string is lex-smallest, so `min ≤ max`).
    #[test]
    fn decode_fts_summary_legal_term_ranges() {
        let none = decode_fts_summary(&fts_summary_bytes(0, b"", b"")).expect("both empty");
        assert_eq!(none.term_range, None);

        let some = decode_fts_summary(&fts_summary_bytes(3, b"alpha", b"omega")).expect("both set");
        assert_eq!(
            some.term_range,
            Some((b"alpha".to_vec(), b"omega".to_vec()))
        );

        // Empty min, present max: a valid ordered range, not an error.
        let empty_min = decode_fts_summary(&fts_summary_bytes(1, b"", b"xyz")).expect("empty min");
        assert_eq!(empty_min.term_range, Some((b"".to_vec(), b"xyz".to_vec())));
    }

    /// An empty map round-trips: both decoders read `n_entries = 0` and
    /// return an empty `HashMap`.
    #[test]
    fn decode_summary_maps_empty() {
        let zero = 0u32.to_le_bytes().to_vec();
        let fts: HashMap<_, _> = decode_fts_summary_map(&zero).expect("empty fts");
        assert!(fts.is_empty());
        let vec: HashMap<_, _> = decode_vector_summary_map(&zero).expect("empty vec");
        assert!(vec.is_empty());
    }

    /// The cursor helpers report `Truncated` (with the field name) when
    /// the buffer is shorter than the requested read.
    #[test]
    fn cursor_helpers_truncate() {
        let mut c = std::io::Cursor::new(&[0u8, 1][..]);
        let err = read_u32(&mut c, "header").expect_err("only 2 bytes");
        assert!(
            matches!(err, DecodeError::Truncated { what: "header", .. }),
            "got {err:?}"
        );
        let mut c = std::io::Cursor::new(&[0u8, 1, 2][..]);
        let err = read_n(&mut c, 8, "body").expect_err("only 3 bytes");
        assert!(
            matches!(
                err,
                DecodeError::Truncated {
                    what: "body",
                    needed: 8,
                    had: 3
                }
            ),
            "got {err:?}"
        );
    }

    /// A scalar-stats stream carrying more than one batch is rejected with
    /// the dedicated `UnexpectedBatchCount` error (the decode_scalar_stats
    /// multi-batch guard).
    #[test]
    fn decode_scalar_stats_rejects_multi_batch() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "c__min",
            DataType::Int64,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![1])) as ArrayRef],
        )
        .expect("batch");
        let mut out = Vec::new();
        {
            let mut w =
                arrow::ipc::writer::StreamWriter::try_new(&mut out, &schema).expect("ipc init");
            w.write(&batch).expect("write 1");
            w.write(&batch).expect("write 2");
            w.finish().expect("finish");
        }
        let err = decode_scalar_stats(&out).expect_err("two batches");
        assert!(
            matches!(err, DecodeError::UnexpectedBatchCount(2)),
            "got {err:?}"
        );
    }

    /// A `__hll` column typed as something other than Binary is rejected
    /// (the hll-column type check), mirroring the `__nulls` type guard.
    #[test]
    fn decode_scalar_stats_wrong_hll_type_errors() {
        let bytes = ipc_batch(
            vec![Field::new("c__hll", DataType::Int64, true)],
            vec![Arc::new(Int64Array::from(vec![1])) as ArrayRef],
        );
        let err = decode_scalar_stats(&bytes).expect_err("bad hll type");
        assert!(matches!(err, DecodeError::ArrowIpc(_)), "got {err:?}");
    }

    /// An optional stat field (`__sum`/`__nulls`/`__hll`) whose base column
    /// has no `__min`/`__max` pair is rejected, not silently dropped — a
    /// stray optional signals corrupted manifest data.
    #[test]
    fn decode_scalar_stats_rejects_orphan_optional_stat() {
        let bytes = ipc_batch(
            vec![
                Field::new("a__min", DataType::Int64, true),
                Field::new("a__max", DataType::Int64, true),
                // `b__sum` has no b__min / b__max — orphaned.
                Field::new("b__sum", DataType::Int64, true),
            ],
            vec![
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                Arc::new(Int64Array::from(vec![2])) as ArrayRef,
                Arc::new(Int64Array::from(vec![3])) as ArrayRef,
            ],
        );
        let err = decode_scalar_stats(&bytes).expect_err("orphan __sum");
        assert!(matches!(err, DecodeError::ArrowIpc(_)), "got {err:?}");
    }

    /// Equal min/max field counts but mismatched base names: a `__min` for
    /// one column and a `__max` for another. The count check passes, so the
    /// per-column assembly surfaces the "has __min but no __max" error.
    #[test]
    fn decode_scalar_stats_mismatched_min_max_bases_errors() {
        let bytes = ipc_batch(
            vec![
                Field::new("a__min", DataType::Int64, true),
                Field::new("b__max", DataType::Int64, true),
            ],
            vec![
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                Arc::new(Int64Array::from(vec![2])) as ArrayRef,
            ],
        );
        let err = decode_scalar_stats(&bytes).expect_err("mismatched bases");
        assert!(matches!(err, DecodeError::ArrowIpc(_)), "got {err:?}");
    }
}

#[cfg(test)]
mod vector_summary_tests {
    use super::{decode_vector_summary, encode_vector_summary};
    use crate::supertable::manifest::{ClusterCentroids, VectorSummary};

    #[test]
    fn round_trips_with_cluster_centroids() {
        // 3 clusters × dim 4, distinct per-cluster value ranges so the
        // per-cluster Sq8 calibration is exercised (incl. a count-0
        // cluster).
        let (n_cent, dim) = (3u32, 4u32);
        let centroids: Vec<f32> = vec![
            0.0, 1.0, 2.0, 3.0, // cluster 0
            -5.0, -2.5, 0.0, 2.5, // cluster 1
            10.0, 10.5, 11.0, 11.5, // cluster 2
        ];
        let counts = vec![100u32, 0, 42];
        let clusters = ClusterCentroids::from_fp32(n_cent, dim, &centroids, counts.clone());
        let s = VectorSummary {
            centroid: vec![1.0, 2.0, 3.0, 4.0],
            radius: 9.0,
            clusters,
        };

        let got = decode_vector_summary(&encode_vector_summary(&s)).expect("decode");
        assert_eq!(got.centroid, s.centroid);
        assert!((got.radius - s.radius).abs() < 1e-9);
        assert_eq!(got.clusters.n_cent, n_cent);
        assert_eq!(got.clusters.dim, dim);
        assert_eq!(got.clusters.counts, counts);
        assert_eq!(got.clusters.codes, s.clusters.codes);
        assert_eq!(got.clusters.mins, s.clusters.mins);
        assert_eq!(got.clusters.scales, s.clusters.scales);

        // Dequantized centroids are within one Sq8 step of the source.
        for c in 0..n_cent as usize {
            let mut out = vec![0f32; dim as usize];
            got.clusters.dequantize_into(c, &mut out);
            let src = &centroids[c * dim as usize..(c + 1) * dim as usize];
            let step = got.clusters.scales[c];
            for (o, e) in out.iter().zip(src) {
                assert!(
                    (o - e).abs() <= step + 1e-6,
                    "cluster {c}: dequant {o} vs {e} (step {step})"
                );
            }
        }
    }

    #[test]
    fn round_trips_with_empty_clusters() {
        let s = VectorSummary {
            centroid: vec![0.5, -0.5],
            radius: 1.0,
            clusters: ClusterCentroids::empty(),
        };
        let got = decode_vector_summary(&encode_vector_summary(&s)).expect("decode");
        assert_eq!(got.centroid, s.centroid);
        assert!(got.clusters.is_empty());
        assert_eq!(got.clusters.n_cent, 0);
    }
}
