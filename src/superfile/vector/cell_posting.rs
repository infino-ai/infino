// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Contiguous Sq8+ε cell posting blob for [`super::layout::VectorLayout::CellPosting`].
//!
//! One superfile carries one cell's postings. Cold read = one range GET on
//! `inf.vec.offset..+length`, then scan/rerank in memory.

use std::{cmp::Ordering, collections::BinaryHeap};

use roaring::RoaringBitmap;

use crate::superfile::{
    BuildError,
    builder::VectorConfig,
    format::vec::{METRIC_ID_COSINE, METRIC_ID_L2SQ, METRIC_ID_NEGDOT},
    vector::{
        builder::derive_sq8_quantizer_from_min_max,
        distance::{Metric, SQ8_RESIDUAL_DIVISOR, Sq8ResidualEpsilonKernel, metric_distance_by},
    },
};

const MAGIC: &[u8] = b"infino.cell_posting.v1\n";
const SEGMENTED_MAGIC: &[u8] = b"infino.cell_posting.segments.v1\n";
const SQ8_CODE_MAX: f32 = 255.0;
const EPSILON_I8_CLAMP: f32 = 127.0;
/// Symmetric clamp bound for the Sq8+ε i8 residual leg (matches IVF builder).
const SQ8_RESIDUAL_I8_CLAMP: f32 = 127.0;
const ROW_BYTES_PER_DIM: usize = 2;

fn u32_le(body: &[u8]) -> Result<u32, String> {
    let arr: [u8; 4] = body.try_into().map_err(|_| "truncated u32".to_string())?;
    Ok(u32::from_le_bytes(arr))
}

fn f32_le(body: &[u8]) -> Result<f32, String> {
    let arr: [u8; 4] = body.try_into().map_err(|_| "truncated f32".to_string())?;
    Ok(f32::from_le_bytes(arr))
}

#[derive(Debug, Clone)]
pub struct CellPostingBuilder {
    columns: Vec<ColumnState>,
}

#[derive(Debug, Clone)]
struct ColumnState {
    config: VectorConfig,
    ids: Vec<u32>,
    vectors: Vec<f32>,
    next_local_id: u32,
}

#[derive(Debug, Clone)]
struct DecodedPosting {
    dim: usize,
    metric: Metric,
    ids: Vec<u32>,
    scale: Vec<f32>,
    offset: Vec<f32>,
    rows: Vec<u8>,
    /// Residual-corrected ||x||² per row (L2Sq/Cosine), matching IVF Sq8+ε layout.
    per_doc_norms: Option<Vec<f32>>,
}

impl Default for CellPostingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CellPostingBuilder {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    pub fn register_column(&mut self, config: VectorConfig) -> Result<(), BuildError> {
        if self
            .columns
            .iter()
            .any(|c| c.config.column == config.column)
        {
            return Err(BuildError::DuplicateLogicalName(config.column));
        }
        self.columns.push(ColumnState {
            config,
            ids: Vec::new(),
            vectors: Vec::new(),
            next_local_id: 0,
        });
        Ok(())
    }

    pub fn add(&mut self, col_id: u32, vector: &[f32]) -> Result<(), BuildError> {
        let col = self
            .columns
            .get_mut(col_id as usize)
            .ok_or_else(|| BuildError::VectorSchemaMismatch(format!("column id {col_id}")))?;
        if vector.len() != col.config.dim {
            return Err(BuildError::VectorDimMismatch {
                column: col.config.column.clone(),
                expected: col.config.dim,
                actual: vector.len(),
            });
        }
        col.ids.push(col.next_local_id);
        col.next_local_id += 1;
        col.vectors.extend_from_slice(vector);
        Ok(())
    }

    pub fn finish(self) -> Result<Vec<u8>, BuildError> {
        if self.columns.is_empty() {
            return Ok(Vec::new());
        }
        if self.columns.len() != 1 {
            return Err(BuildError::VectorSchemaMismatch(
                "cell posting superfile supports exactly one vector column".into(),
            ));
        }
        let col = &self.columns[0];
        encode_blob(col.config.metric, col.config.dim, &col.ids, &col.vectors)
            .map_err(BuildError::VectorSchemaMismatch)
    }
}

pub fn encode_blob(
    metric: Metric,
    dim: usize,
    ids: &[u32],
    vectors: &[f32],
) -> Result<Vec<u8>, String> {
    if dim == 0 {
        return Err("cell posting dim must be > 0".into());
    }
    if vectors.len() != ids.len() * dim {
        return Err("cell posting vector length mismatch".into());
    }
    let rows: Vec<usize> = (0..ids.len()).collect();
    let posting = encode_rows(metric, vectors, ids, dim, &rows);
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    out.push(metric_id(metric));
    out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for v in &posting.scale {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in &posting.offset {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&posting.rows);
    for id in &posting.ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    if let Some(norms) = &posting.per_doc_norms {
        for n in norms {
            out.extend_from_slice(&n.to_le_bytes());
        }
    }
    Ok(out)
}

fn open_segments(bytes: &[u8]) -> Result<Vec<DecodedPosting>, String> {
    if bytes.starts_with(SEGMENTED_MAGIC) {
        open_segmented_blob(bytes)
    } else {
        Ok(vec![open_legacy_blob(bytes)?])
    }
}

fn open_legacy_blob(bytes: &[u8]) -> Result<DecodedPosting, String> {
    let body = bytes
        .strip_prefix(MAGIC)
        .ok_or_else(|| "bad cell posting magic".to_string())?;
    let (posting, _) = parse_segment_body(body, true)?;
    Ok(posting)
}

fn open_segmented_blob(bytes: &[u8]) -> Result<Vec<DecodedPosting>, String> {
    let body = bytes
        .strip_prefix(SEGMENTED_MAGIC)
        .ok_or_else(|| "bad segmented cell posting magic".to_string())?;
    if body.len() < 4 + 1 + 4 {
        return Err("segmented cell posting header truncated".into());
    }
    let dim = u32_le(&body[0..4])? as usize;
    let metric = metric_from_id(body[4])?;
    let n_segments = u32_le(&body[5..9])? as usize;
    let mut off = 9;
    let mut segments = Vec::with_capacity(n_segments);
    for _ in 0..n_segments {
        if body.len() < off + 4 {
            return Err("segmented cell posting segment header truncated".into());
        }
        let n_docs = u32_le(&body[off..off + 4])? as usize;
        off += 4;
        let segment_len = dim * 8 + n_docs * dim * ROW_BYTES_PER_DIM + n_docs * 4 + n_docs * 4;
        if body.len() < off + segment_len {
            return Err("segmented cell posting segment body truncated".into());
        }
        let seg_body = [
            &(dim as u32).to_le_bytes()[..],
            &[metric_id(metric)][..],
            &(n_docs as u32).to_le_bytes()[..],
            &body[off..off + segment_len],
        ]
        .concat();
        let (segment, consumed) = parse_segment_body(&seg_body, false)?;
        debug_assert_eq!(consumed, seg_body.len());
        segments.push(segment);
        off += segment_len;
    }
    if off != body.len() {
        return Err("segmented cell posting trailing bytes".into());
    }
    Ok(segments)
}

fn parse_segment_body(
    body: &[u8],
    allow_legacy_missing_norms: bool,
) -> Result<(DecodedPosting, usize), String> {
    if body.len() < 4 + 1 + 4 {
        return Err("cell posting header truncated".into());
    }
    let dim = u32_le(&body[0..4])? as usize;
    let metric = metric_from_id(body[4])?;
    let n_docs = u32_le(&body[5..9])? as usize;
    let header = 9 + dim * 8;
    let rows_len = n_docs * dim * ROW_BYTES_PER_DIM;
    let ids_len = n_docs * 4;
    if body.len() < header + rows_len + ids_len {
        return Err("cell posting body truncated".into());
    }
    let scale_start = 9;
    let offset_start = scale_start + dim * 4;
    let rows_start = header;
    let ids_start = rows_start + rows_len;
    let norms_start = ids_start + ids_len;
    let mut scale = vec![0f32; dim];
    let mut offset = vec![0f32; dim];
    for d in 0..dim {
        scale[d] = f32_le(&body[scale_start + d * 4..scale_start + (d + 1) * 4])?;
        offset[d] = f32_le(&body[offset_start + d * 4..offset_start + (d + 1) * 4])?;
    }
    let ids = decode_ids(&body[ids_start..ids_start + ids_len]);
    let rows = body[rows_start..rows_start + rows_len].to_vec();
    let mut posting = DecodedPosting {
        dim,
        metric,
        ids,
        scale,
        offset,
        rows,
        per_doc_norms: None,
    };
    let consumed = if body.len() >= norms_start + n_docs * 4 {
        let mut norms = Vec::with_capacity(n_docs);
        for i in 0..n_docs {
            let off = norms_start + i * 4;
            norms.push(f32_le(&body[off..off + 4])?);
        }
        posting.per_doc_norms = Some(norms);
        norms_start + n_docs * 4
    } else if allow_legacy_missing_norms && matches!(metric, Metric::L2Sq | Metric::Cosine) {
        posting.per_doc_norms = Some(compute_encoded_norms(&posting));
        norms_start
    } else {
        norms_start
    };
    Ok((posting, consumed))
}

pub fn search_blob(bytes: &[u8], query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, String> {
    let postings = open_segments(bytes)?;
    let Some(first) = postings.first() else {
        return Ok(Vec::new());
    };
    if query.len() != first.dim {
        return Err("cell posting query dim mismatch".into());
    }
    if k == 0 {
        return Ok(Vec::new());
    }
    let mut heap = BinaryHeap::<WorstHit>::new();
    for posting in &postings {
        if posting.dim != first.dim || posting.metric != first.metric {
            return Err("cell posting segment metric/dim mismatch".into());
        }
        let norms_for_kernel = match posting.metric {
            Metric::NegDot => None,
            Metric::L2Sq | Metric::Cosine => Some(
                posting
                    .per_doc_norms
                    .as_deref()
                    .ok_or_else(|| "cell posting missing per_doc_norms".to_string())?,
            ),
        };
        let kernel = Sq8ResidualEpsilonKernel::new(
            posting.metric,
            query,
            &posting.scale,
            &posting.offset,
            SQ8_RESIDUAL_DIVISOR,
            norms_for_kernel,
        );
        let dim = posting.dim;
        for row in 0..posting.ids.len() {
            let base = row * dim * ROW_BYTES_PER_DIM;
            let codes = &posting.rows[base..base + dim];
            let residuals = &posting.rows[base + dim..base + dim + dim];
            let norm = norms_for_kernel.map(|norms| norms[row]);
            let d = kernel.distance_with_norm(codes, residuals, norm);
            let hit = WorstHit((posting.ids[row], d));
            if heap.len() < k {
                heap.push(hit);
            } else if let Some(worst) = heap.peek()
                && cmp_f32(hit.0.1, worst.0.1).is_lt()
            {
                heap.pop();
                heap.push(hit);
            }
        }
    }
    let mut out: Vec<(u32, f32)> = heap.into_iter().map(|h| h.0).collect();
    out.sort_by(|a, b| cmp_f32(a.1, b.1));
    Ok(out)
}

pub fn merge_encoded_blobs(
    inputs: &[(&[u8], Option<&RoaringBitmap>)],
) -> Result<(Vec<u8>, u64), String> {
    if inputs.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let mut merged_segments = Vec::new();
    let mut next_id = 0u32;
    let mut dim = None;
    let mut metric = None;
    for (blob, deleted) in inputs {
        for segment in open_segments(blob)? {
            if let Some(d) = dim {
                if d != segment.dim {
                    return Err("cell posting merge dim mismatch".into());
                }
            } else {
                dim = Some(segment.dim);
            }
            if let Some(m) = metric {
                if m != segment.metric {
                    return Err("cell posting merge metric mismatch".into());
                }
            } else {
                metric = Some(segment.metric);
            }
            let mut ids = Vec::with_capacity(segment.ids.len());
            let mut rows = Vec::with_capacity(segment.rows.len());
            let mut norms = segment
                .per_doc_norms
                .as_ref()
                .map(|_| Vec::with_capacity(segment.ids.len()));
            for (row, &old_id) in segment.ids.iter().enumerate() {
                if deleted.is_some_and(|bm| bm.contains(old_id)) {
                    continue;
                }
                ids.push(next_id);
                next_id = next_id.saturating_add(1);
                let base = row * segment.dim * ROW_BYTES_PER_DIM;
                rows.extend_from_slice(&segment.rows[base..base + segment.dim * ROW_BYTES_PER_DIM]);
                if let (Some(src), Some(dst)) = (segment.per_doc_norms.as_ref(), norms.as_mut()) {
                    dst.push(src[row]);
                }
            }
            if !ids.is_empty() {
                let mut segment = DecodedPosting {
                    dim: segment.dim,
                    metric: segment.metric,
                    ids,
                    scale: segment.scale,
                    offset: segment.offset,
                    rows,
                    per_doc_norms: norms,
                };
                if segment.per_doc_norms.is_none()
                    && matches!(segment.metric, Metric::L2Sq | Metric::Cosine)
                {
                    segment.per_doc_norms = Some(compute_encoded_norms(&segment));
                }
                merged_segments.push(segment);
            }
        }
    }
    let n_docs = u64::from(next_id);
    if n_docs == 0 {
        return Ok((Vec::new(), 0));
    }
    let Some(dim) = dim else {
        return Err("cell posting merge missing dim".into());
    };
    let Some(metric) = metric else {
        return Err("cell posting merge missing metric".into());
    };
    Ok((
        encode_segmented_blob(dim, metric, &merged_segments)?,
        n_docs,
    ))
}

fn encode_segmented_blob(
    dim: usize,
    metric: Metric,
    segments: &[DecodedPosting],
) -> Result<Vec<u8>, String> {
    let mut out = SEGMENTED_MAGIC.to_vec();
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    out.push(metric_id(metric));
    out.extend_from_slice(&(segments.len() as u32).to_le_bytes());
    for segment in segments {
        if segment.dim != dim || segment.metric != metric {
            return Err("cell posting segment metric/dim mismatch".into());
        }
        out.extend_from_slice(&(segment.ids.len() as u32).to_le_bytes());
        for v in &segment.scale {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &segment.offset {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&segment.rows);
        for id in &segment.ids {
            out.extend_from_slice(&id.to_le_bytes());
        }
        let norms = match (&segment.per_doc_norms, metric) {
            (Some(norms), _) => norms.clone(),
            (None, Metric::NegDot) => Vec::new(),
            (None, Metric::L2Sq | Metric::Cosine) => compute_encoded_norms(segment),
        };
        if matches!(metric, Metric::L2Sq | Metric::Cosine) {
            for n in norms {
                out.extend_from_slice(&n.to_le_bytes());
            }
        }
    }
    Ok(out)
}

fn compute_encoded_norms(p: &DecodedPosting) -> Vec<f32> {
    let dim = p.dim;
    let mut norms = Vec::with_capacity(p.ids.len());
    for row in 0..p.ids.len() {
        let base = row * dim * ROW_BYTES_PER_DIM;
        let mut acc = 0.0f64;
        for d in 0..dim {
            let code = p.rows[base + d] as f32;
            let eps = i8::from_le_bytes([p.rows[base + dim + d]]) as f32;
            let step = p.scale[d] / SQ8_RESIDUAL_DIVISOR;
            let x = p.offset[d] + code * p.scale[d] + eps * step;
            acc += (x as f64) * (x as f64);
        }
        norms.push(acc as f32);
    }
    norms
}

fn decode_ids(bytes: &[u8]) -> Vec<u32> {
    // `chunks_exact(4)` only yields 4-byte slices, so `u32_le` never errors here;
    // collect through Result and fall back to empty rather than panicking.
    bytes
        .chunks_exact(4)
        .map(u32_le)
        .collect::<Result<Vec<u32>, _>>()
        .unwrap_or_default()
}

struct EncodedRows {
    ids: Vec<u32>,
    scale: Vec<f32>,
    offset: Vec<f32>,
    rows: Vec<u8>,
    per_doc_norms: Option<Vec<f32>>,
}

fn encode_rows(
    metric: Metric,
    vectors: &[f32],
    ids: &[u32],
    dim: usize,
    rows: &[usize],
) -> EncodedRows {
    if rows.is_empty() {
        return EncodedRows {
            ids: Vec::new(),
            scale: vec![1.0; dim],
            offset: vec![0.0; dim],
            rows: Vec::new(),
            per_doc_norms: None,
        };
    }
    let mut min = vec![f32::INFINITY; dim];
    let mut max = vec![f32::NEG_INFINITY; dim];
    for &row in rows {
        let src = &vectors[row * dim..(row + 1) * dim];
        for d in 0..dim {
            min[d] = min[d].min(src[d]);
            max[d] = max[d].max(src[d]);
        }
    }
    let (scale, offset) = derive_sq8_quantizer_from_min_max(&min, &max);
    let store_norms = matches!(metric, Metric::L2Sq | Metric::Cosine);
    let mut out_ids = Vec::with_capacity(rows.len());
    let mut encoded = Vec::with_capacity(rows.len() * dim * ROW_BYTES_PER_DIM);
    let mut per_doc_norms = store_norms.then(|| Vec::with_capacity(rows.len()));
    for &row in rows {
        out_ids.push(ids[row]);
        let src = &vectors[row * dim..(row + 1) * dim];
        let code_start = encoded.len();
        encoded.resize(code_start + dim * ROW_BYTES_PER_DIM, 0);
        let eps_start = code_start + dim;
        let mut acc = 0.0f64;
        for d in 0..dim {
            let q = if scale[d] > 0.0 {
                ((src[d] - offset[d]) / scale[d])
                    .round()
                    .clamp(0.0, SQ8_CODE_MAX) as u8
            } else {
                0
            };
            let base = offset[d] + q as f32 * scale[d];
            let step = scale[d] / SQ8_RESIDUAL_DIVISOR;
            let eps = if step > 0.0 {
                ((src[d] - base) / step)
                    .round()
                    .clamp(-EPSILON_I8_CLAMP, EPSILON_I8_CLAMP) as i8
            } else {
                0
            };
            encoded[code_start + d] = q;
            encoded[eps_start + d] = eps.to_le_bytes()[0];
            if store_norms {
                let x = base + (eps as f32) * step;
                acc += (x as f64) * (x as f64);
            }
        }
        if let Some(norms) = per_doc_norms.as_mut() {
            norms.push(acc as f32);
        }
    }
    EncodedRows {
        ids: out_ids,
        scale,
        offset,
        rows: encoded,
        per_doc_norms,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct WorstHit((u32, f32));

impl Eq for WorstHit {}
impl Ord for WorstHit {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_f32(self.0.1, other.0.1)
    }
}
impl PartialOrd for WorstHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn cmp_f32(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

fn metric_id(m: Metric) -> u8 {
    // Same metric↔id mapping as the IVF directory entry (format::vec).
    match m {
        Metric::L2Sq => METRIC_ID_L2SQ as u8,
        Metric::Cosine => METRIC_ID_COSINE as u8,
        Metric::NegDot => METRIC_ID_NEGDOT as u8,
    }
}

fn metric_from_id(id: u8) -> Result<Metric, String> {
    match id as u32 {
        METRIC_ID_L2SQ => Ok(Metric::L2Sq),
        METRIC_ID_COSINE => Ok(Metric::Cosine),
        METRIC_ID_NEGDOT => Ok(Metric::NegDot),
        _ => Err(format!("unknown cell posting metric id {id}")),
    }
}

/// One Sq8+ε row carried through SPFresh maintenance without fp32 reconstruction.
///
/// `scale`/`offset` are the per-cluster dequant params (length `dim`), identical
/// for every row in a cluster, so they are stored as a shared `Arc<[f32]>`: all
/// rows decoded from one cluster point at a single backing buffer instead of each
/// carrying its own `dim`-length copy. At dim=1024 that is ~8 KiB/row of
/// duplication removed — material when a drain materializes millions of rows.
#[derive(Debug, Clone)]
pub struct EncodedCellRow {
    pub stable_id: i128,
    pub scale: std::sync::Arc<[f32]>,
    pub offset: std::sync::Arc<[f32]>,
    pub codes: Vec<u8>,
    pub residuals: Vec<u8>,
    pub norm_sq: Option<f32>,
}

/// One IVF row for Sq8-native maintenance rebuilds: preserved 1-bit RaBitQ estimate
/// codes plus the Sq8+ε rerank payload. Re-fed into [`super::builder::VectorBuilder`].
#[derive(Debug, Clone)]
pub struct MaterializedIvfRow {
    pub local_doc_id: u32,
    pub stable_id: i128,
    pub rabitq_code: Vec<u8>,
    pub encoded: EncodedCellRow,
}

/// True when two per-cluster Sq8 quantizers are bitwise identical.
pub(crate) fn sq8_quant_params_equal(
    scale_a: &[f32],
    offset_a: &[f32],
    scale_b: &[f32],
    offset_b: &[f32],
) -> bool {
    scale_a.len() == scale_b.len()
        && offset_a.len() == offset_b.len()
        && scale_a == scale_b
        && offset_a == offset_b
}

/// Per-cluster (scale, offset) taken from the medoid Sq8 row in `bucket`.
pub(crate) fn cluster_quant_from_medoid(
    metric: Metric,
    dim: usize,
    bucket: &[&MaterializedIvfRow],
) -> (Vec<f32>, Vec<f32>) {
    debug_assert!(!bucket.is_empty());
    let encoded: Vec<EncodedCellRow> = bucket.iter().map(|r| r.encoded.clone()).collect();
    let mid = medoid_index_symmetric(metric, dim, &encoded);
    (encoded[mid].scale.to_vec(), encoded[mid].offset.to_vec())
}

/// Copy or Sq8-transcode one row into `out` (`[codes | residuals]`, length `2·dim`).
///
/// When source quant matches the destination cluster quantizer, copies bytes
/// verbatim. Otherwise re-quantizes per dimension from the folded scalar
/// component — no `dim`-length fp32 buffer.
///
/// Returns residual-corrected ||x||² when `store_norm` is true (L2Sq/Cosine).
pub(crate) fn materialize_sq8_residual_row_into_cluster_quant(
    row: &EncodedCellRow,
    dst_scale: &[f32],
    dst_offset: &[f32],
    dim: usize,
    out: &mut [u8],
    store_norm: bool,
) -> Option<f32> {
    debug_assert_eq!(out.len(), dim * ROW_BYTES_PER_DIM);
    let code_off = 0;
    let res_off = dim;
    if sq8_quant_params_equal(&row.scale, &row.offset, dst_scale, dst_offset) {
        out[..dim].copy_from_slice(&row.codes);
        out[dim..].copy_from_slice(&row.residuals);
        return store_norm.then(|| {
            row.norm_sq.unwrap_or_else(|| {
                sq8_residual_norm_sq(dim, dst_scale, dst_offset, &row.codes, &row.residuals)
            })
        });
    }

    let inv_scale: Vec<f32> = dst_scale.iter().map(|s| 1.0 / s).collect();
    let c2: Vec<f32> = dst_scale
        .iter()
        .zip(dst_offset.iter())
        .map(|(s, o)| (-o).mul_add(1.0 / s, 0.5))
        .collect();
    let residual_divisor = SQ8_RESIDUAL_DIVISOR;
    let mut acc = 0.0f64;
    for d in 0..dim {
        let v = encoded_component_at(row, d);
        let q = v.mul_add(inv_scale[d], c2[d]).clamp(0.0, SQ8_CODE_MAX);
        out[code_off + d] = q as u8;
        let base = q.mul_add(dst_scale[d], dst_offset[d]);
        let step = dst_scale[d] / residual_divisor;
        let rq = if step > 0.0 {
            ((v - base) / step)
                .round()
                .clamp(-SQ8_RESIDUAL_I8_CLAMP, SQ8_RESIDUAL_I8_CLAMP) as i8
        } else {
            0
        };
        out[res_off + d] = rq.to_le_bytes()[0];
        if store_norm {
            let x = base + (rq as f32) * step;
            acc += (x as f64) * (x as f64);
        }
    }
    store_norm.then_some(acc as f32)
}

/// ||x||² for one Sq8+ε row, folded inline (no fp32 vector buffer).
pub(crate) fn sq8_residual_norm_sq(
    dim: usize,
    scale: &[f32],
    offset: &[f32],
    codes: &[u8],
    residuals: &[u8],
) -> f32 {
    let inv_div = 1.0 / SQ8_RESIDUAL_DIVISOR;
    (0..dim)
        .map(|d| {
            let v = offset[d]
                + scale[d]
                    * (codes[d] as f32 + (i8::from_le_bytes([residuals[d]]) as f32) * inv_div);
            v * v
        })
        .sum()
}

/// Fold one Sq8+ε component at `d` without materializing the full vector.
#[inline]
pub(crate) fn encoded_component_at(row: &EncodedCellRow, d: usize) -> f32 {
    let inv_div = 1.0 / SQ8_RESIDUAL_DIVISOR;
    row.offset[d]
        + row.scale[d]
            * (row.codes[d] as f32 + (i8::from_le_bytes([row.residuals[d]]) as f32) * inv_div)
}

fn distance_encoded_rows_symmetric(
    metric: Metric,
    dim: usize,
    a: &EncodedCellRow,
    b: &EncodedCellRow,
) -> f32 {
    metric_distance_by(
        metric,
        dim,
        |d| encoded_component_at(a, d),
        |d| encoded_component_at(b, d),
    )
}

/// Index of the medoid row — the one minimizing the summed pairwise distance to
/// all others — under an arbitrary row↔row distance `dist`. Shared by the
/// symmetric kernel here and the asymmetric variant in `supertable::spfresh`.
pub(crate) fn medoid_index_by<F>(shard: &[EncodedCellRow], dist: F) -> usize
where
    F: Fn(&EncodedCellRow, &EncodedCellRow) -> f32,
{
    let mut best_idx = 0usize;
    let mut best_sum = f32::INFINITY;
    for (i, row_i) in shard.iter().enumerate() {
        let sum: f32 = shard.iter().map(|row_j| dist(row_i, row_j)).sum();
        if sum < best_sum {
            best_sum = sum;
            best_idx = i;
        }
    }
    best_idx
}

fn medoid_index_symmetric(metric: Metric, dim: usize, shard: &[EncodedCellRow]) -> usize {
    medoid_index_by(shard, |a, b| {
        distance_encoded_rows_symmetric(metric, dim, a, b)
    })
}

/// Sq8+ε row → `dim` fp32 components for manifest [`ClusterCentroids::from_fp32`]
/// or IVF header centroids only (never a full-corpus decode).
pub(crate) fn manifest_centroid_components_from_row(row: &EncodedCellRow, dim: usize) -> Vec<f32> {
    (0..dim).map(|d| encoded_component_at(row, d)).collect()
}

/// Lloyd k-means on Sq8+ε rows for internal IVF centroid training during maintenance.
pub(crate) fn encoded_ivf_kmeans(
    rows: &[EncodedCellRow],
    metric: Metric,
    k: usize,
    iters: usize,
) -> Vec<f32> {
    if rows.is_empty() || k == 0 {
        return Vec::new();
    }
    let dim = rows[0].codes.len();
    let k = k.min(rows.len());

    let mut medoid_indices: Vec<usize> = Vec::with_capacity(k);
    medoid_indices.push(0);
    while medoid_indices.len() < k {
        let mut best_idx = 0usize;
        let mut best_min = f32::NEG_INFINITY;
        for (i, row) in rows.iter().enumerate() {
            if medoid_indices.contains(&i) {
                continue;
            }
            let min_d = medoid_indices
                .iter()
                .map(|&s| distance_encoded_rows_symmetric(metric, dim, row, &rows[s]))
                .fold(f32::INFINITY, f32::min);
            if min_d > best_min {
                best_min = min_d;
                best_idx = i;
            }
        }
        medoid_indices.push(best_idx);
    }

    let mut assign = vec![0usize; rows.len()];
    for _ in 0..iters {
        for (i, row) in rows.iter().enumerate() {
            let mut best_c = 0usize;
            let mut best_score = f32::INFINITY;
            for (c, &mid) in medoid_indices.iter().enumerate().take(k) {
                let score = distance_encoded_rows_symmetric(metric, dim, row, &rows[mid]);
                if score < best_score {
                    best_score = score;
                    best_c = c;
                }
            }
            assign[i] = best_c;
        }
        for (c, slot) in medoid_indices.iter_mut().enumerate().take(k) {
            let shard: Vec<usize> = assign
                .iter()
                .enumerate()
                .filter(|(_, a)| **a == c)
                .map(|(i, _)| i)
                .collect();
            if shard.is_empty() {
                continue;
            }
            let shard_rows: Vec<EncodedCellRow> = shard.iter().map(|&i| rows[i].clone()).collect();
            let local_m = medoid_index_symmetric(metric, dim, &shard_rows);
            *slot = shard[local_m];
        }
    }

    let mut out = vec![0f32; k * dim];
    for (c, &mid) in medoid_indices.iter().enumerate().take(k) {
        let fp32 = manifest_centroid_components_from_row(&rows[mid], dim);
        out[c * dim..(c + 1) * dim].copy_from_slice(&fp32);
    }
    out
}

/// Load live Sq8+ε rows from a cell-posting blob aligned with scalar `_id` order.
pub fn load_encoded_rows_from_blob(
    bytes: &[u8],
    stable_ids: &[i128],
    deleted: Option<&RoaringBitmap>,
) -> Result<Vec<EncodedCellRow>, String> {
    let postings = open_segments(bytes)?;
    let mut row_idx = 0usize;
    let mut out = Vec::new();
    for posting in &postings {
        // One shared backing per segment — every row in this segment clones the
        // Arc (a refcount bump), not the dim-length scale/offset buffers.
        let scale_arc: std::sync::Arc<[f32]> = std::sync::Arc::from(posting.scale.as_slice());
        let offset_arc: std::sync::Arc<[f32]> = std::sync::Arc::from(posting.offset.as_slice());
        for local_row in 0..posting.ids.len() {
            if deleted.is_some_and(|bm| bm.contains(posting.ids[local_row])) {
                row_idx += 1;
                continue;
            }
            if row_idx >= stable_ids.len() {
                return Err("cell posting row count exceeds scalar _id batch".into());
            }
            let base = local_row * posting.dim * ROW_BYTES_PER_DIM;
            let codes = posting.rows[base..base + posting.dim].to_vec();
            let residuals =
                posting.rows[base + posting.dim..base + posting.dim + posting.dim].to_vec();
            let norm_sq = posting.per_doc_norms.as_ref().map(|norms| norms[local_row]);
            out.push(EncodedCellRow {
                stable_id: stable_ids[row_idx],
                scale: scale_arc.clone(),
                offset: offset_arc.clone(),
                codes,
                residuals,
                norm_sq,
            });
            row_idx += 1;
        }
    }
    if row_idx != stable_ids.len() {
        return Err(format!(
            "cell posting row count mismatch: blob {row_idx} vs scalar {}",
            stable_ids.len()
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::{builder::VectorConfig, vector::rerank_codec::RerankCodec};

    #[test]
    fn roundtrip_and_search() {
        let dim = 8usize;
        let mut ids = Vec::new();
        let mut vecs = Vec::new();
        for i in 0..32u32 {
            ids.push(i);
            for d in 0..dim {
                vecs.push(if d == 0 { i as f32 * 0.01 } else { 0.0 });
            }
        }
        let blob = encode_blob(Metric::L2Sq, dim, &ids, &vecs).expect("encode");
        let mut q = vec![0f32; dim];
        q[0] = 0.31;
        let hits = search_blob(&blob, &q, 5).expect("search");
        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].0, 31);
    }

    #[test]
    fn loaded_rows_share_one_scale_offset_backing_per_segment() {
        // The drain materializes millions of EncodedCellRows; scale/offset are
        // per-cluster (length `dim`), so rows of one segment must share a single
        // Arc<[f32]> backing rather than each carrying its own copy. Per-row
        // copies would give one distinct pointer per row.
        let dim = 8usize;
        let n = 16u32;
        let mut ids = Vec::new();
        let mut vecs = Vec::new();
        for i in 0..n {
            ids.push(i);
            for d in 0..dim {
                vecs.push(if d == 0 { i as f32 * 0.01 } else { 0.0 });
            }
        }
        let blob = encode_blob(Metric::L2Sq, dim, &ids, &vecs).expect("encode");
        let stable_ids: Vec<i128> = (0..n as i128).collect();
        let rows = load_encoded_rows_from_blob(&blob, &stable_ids, None).expect("load");
        assert_eq!(rows.len(), n as usize);
        assert_eq!(rows[0].scale.len(), dim, "scale is per-dim, not per-row sized");

        let distinct_scale: std::collections::HashSet<*const f32> =
            rows.iter().map(|r| r.scale.as_ptr()).collect();
        let distinct_offset: std::collections::HashSet<*const f32> =
            rows.iter().map(|r| r.offset.as_ptr()).collect();
        assert!(
            distinct_scale.len() < rows.len(),
            "scale backing not shared: {} distinct buffers for {} rows",
            distinct_scale.len(),
            rows.len()
        );
        assert!(
            distinct_offset.len() < rows.len(),
            "offset backing not shared: {} distinct buffers for {} rows",
            distinct_offset.len(),
            rows.len()
        );
    }

    #[test]
    fn merge_encoded_blobs_remaps_ids_and_searches_segments() {
        let dim = 4usize;
        let ids_a = vec![0u32, 1, 2];
        let ids_b = vec![0u32, 1];
        let vecs_a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let vecs_b = vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0];
        let blob_a = encode_blob(Metric::L2Sq, dim, &ids_a, &vecs_a).expect("encode a");
        let blob_b = encode_blob(Metric::L2Sq, dim, &ids_b, &vecs_b).expect("encode b");
        let mut deleted = RoaringBitmap::new();
        deleted.insert(1);

        let (merged, n_docs) = merge_encoded_blobs(&[
            (blob_a.as_slice(), Some(&deleted)),
            (blob_b.as_slice(), None),
        ])
        .expect("merge");
        assert_eq!(n_docs, 4);

        let hits = search_blob(&merged, &[2.0, 0.0, 0.0, 0.0], 4).expect("search merged");
        assert_eq!(hits.len(), 4);
        assert_eq!(
            hits[0].0, 3,
            "second blob row id must be remapped after surviving rows"
        );
        assert!(hits.iter().all(|(id, _)| *id < 4));
    }

    #[test]
    fn builder_finish_matches_encode() {
        let cfg = VectorConfig {
            column: "emb".into(),
            dim: 4,
            n_cent: 1,
            rot_seed: 1,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
        };
        let mut b = CellPostingBuilder::new();
        b.register_column(cfg).expect("register");
        b.add(0, &[1.0, 0.0, 0.0, 0.0]).expect("add");
        b.add(0, &[0.0, 1.0, 0.0, 0.0]).expect("add");
        let blob = b.finish().expect("finish");
        let hits = search_blob(&blob, &[1.0, 0.0, 0.0, 0.0], 1).expect("search");
        assert_eq!(hits[0].0, 0);
    }
}
