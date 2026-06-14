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
use crate::supertable::manifest::{FtsSummary, ScalarStatsTable, VectorSummary};

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

    /// Arrow IPC parse failed.
    #[error("arrow ipc parse failed: {0}")]
    ArrowIpc(String),

    /// Arrow IPC stream produced zero batches where one was
    /// expected (or more than one).
    #[error("expected exactly one arrow ipc batch, got {0}")]
    UnexpectedBatchCount(usize),
}

// ---------------------------------------------------------
// ScalarStatsTable: arrow-ipc encoding.
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

pub fn encode_scalar_stats(stats: &ScalarStatsTable) -> Vec<u8> {
    if stats.cols.is_empty() {
        // Empty table → emit a sentinel zero-length blob.
        // Decode treats that as `ScalarStatsTable::new()`.
        return Vec::new();
    }
    // Sort columns for deterministic output. The order
    // doesn't matter for correctness but makes diffs +
    // content-addressing stable.
    let mut keys: Vec<&String> = stats.cols.keys().collect();
    keys.sort();

    let mut fields: Vec<Field> = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for key in keys {
        let (mn, mx) = &stats.cols[key];
        fields.push(Field::new(
            format!("{key}{MIN_SUFFIX}"),
            mn.data_type().clone(),
            true,
        ));
        fields.push(Field::new(
            format!("{key}{MAX_SUFFIX}"),
            mx.data_type().clone(),
            true,
        ));
        arrays.push(mn.clone());
        arrays.push(mx.clone());
        if let Some(&nulls) = stats.null_counts.get(key) {
            fields.push(Field::new(
                format!("{key}{NULLS_SUFFIX}"),
                arrow_schema::DataType::UInt64,
                true,
            ));
            arrays.push(Arc::new(arrow_array::UInt64Array::from(vec![nulls])) as ArrayRef);
        }
        if let Some(sum) = stats.sums.get(key) {
            fields.push(Field::new(
                format!("{key}{SUM_SUFFIX}"),
                sum.data_type().clone(),
                true,
            ));
            arrays.push(sum.clone());
        }
        if let Some(sketch) = stats.hll.get(key) {
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

pub fn decode_scalar_stats(bytes: &[u8]) -> Result<ScalarStatsTable, DecodeError> {
    if bytes.is_empty() {
        return Ok(ScalarStatsTable::new());
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
    // everything else is optional.
    let mut mins: HashMap<String, ArrayRef> = HashMap::new();
    let mut maxes: HashMap<String, ArrayRef> = HashMap::new();
    let mut stats = ScalarStatsTable::new();
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
                stats.null_counts.insert(base.to_string(), arr.value(0));
            }
        } else if let Some(base) = name.strip_suffix(SUM_SUFFIX) {
            stats.sums.insert(base.to_string(), column.clone());
        } else if let Some(base) = name.strip_suffix(HLL_SUFFIX) {
            let arr = column
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .ok_or_else(|| {
                    DecodeError::ArrowIpc(format!("{name}: __hll column is not Binary"))
                })?;
            if !arr.is_empty() && !arr.is_null(0) {
                stats.hll.insert(base.to_string(), arr.value(0).to_vec());
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
    for (base, mn) in mins {
        let mx = maxes.remove(&base).ok_or_else(|| {
            DecodeError::ArrowIpc(format!("column {base} has __min but no __max"))
        })?;
        stats.cols.insert(base, (mn, mx));
    }
    Ok(stats)
}

// ---------------------------------------------------------
// FtsSummary: custom packed.
//
// Layout (all LE):
//   u32 bloom_len                  (== n_blocks × BLOCK_BYTES)
//   [bloom_len bytes]              (Bloom::to_bytes output)
//   u32 n_terms_distinct
//   u32 min_term_len
//   [min_term bytes]
//   u32 max_term_len
//   [max_term bytes]
// ---------------------------------------------------------

pub fn encode_fts_summary(s: &FtsSummary) -> Vec<u8> {
    let bloom_bytes = s.term_bloom.to_bytes();
    let cap = 4 + bloom_bytes.len() + 4 + 4 + s.term_range.0.len() + 4 + s.term_range.1.len();
    let mut out = Vec::with_capacity(cap);
    out.extend_from_slice(&(bloom_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&bloom_bytes);
    out.extend_from_slice(&s.n_terms_distinct.to_le_bytes());
    out.extend_from_slice(&(s.term_range.0.len() as u32).to_le_bytes());
    out.extend_from_slice(&s.term_range.0);
    out.extend_from_slice(&(s.term_range.1.len() as u32).to_le_bytes());
    out.extend_from_slice(&s.term_range.1);
    out
}

pub fn decode_fts_summary(bytes: &[u8]) -> Result<FtsSummary, DecodeError> {
    let mut c = Cursor::new(bytes);
    let bloom_len = read_u32(&mut c, "bloom_len")? as usize;
    let bloom_bytes = read_n(&mut c, bloom_len, "bloom_bytes")?;
    let term_bloom =
        Bloom::from_bytes(&bloom_bytes).ok_or(DecodeError::InvalidBloomLayout(bloom_len))?;
    let n_terms_distinct = read_u32(&mut c, "n_terms_distinct")?;
    let min_len = read_u32(&mut c, "min_term_len")? as usize;
    let min_term = read_n(&mut c, min_len, "min_term")?;
    let max_len = read_u32(&mut c, "max_term_len")? as usize;
    let max_term = read_n(&mut c, max_len, "max_term")?;
    Ok(FtsSummary {
        term_bloom,
        n_terms_distinct,
        term_range: (min_term, max_term),
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

pub fn encode_fts_summary_map(map: &HashMap<String, FtsSummary>) -> Vec<u8> {
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

pub fn decode_fts_summary_map(bytes: &[u8]) -> Result<HashMap<String, FtsSummary>, DecodeError> {
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
