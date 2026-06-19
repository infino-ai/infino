// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Contiguous Sq8+ε cell posting blob for [`super::layout::VectorLayout::CellPosting`].
//!
//! One superfile carries one cell's postings. Cold read = one range GET on
//! `inf.vec.offset..+length`, then scan/rerank in memory.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use roaring::RoaringBitmap;

use crate::superfile::BuildError;
use crate::superfile::builder::VectorConfig;
use crate::superfile::vector::distance::{
    Metric, SQ8_RESIDUAL_DIVISOR, Sq8ResidualEpsilonKernel,
};

const MAGIC: &[u8] = b"infino.cell_posting.v1\n";
const SEGMENTED_MAGIC: &[u8] = b"infino.cell_posting.segments.v1\n";
const SQ8_CODE_MAX: f32 = 255.0;
const EPSILON_I8_CLAMP: f32 = 127.0;
const ROW_BYTES_PER_DIM: usize = 2;

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

impl CellPostingBuilder {
    pub fn new() -> Self {
        Self { columns: Vec::new() }
    }

    pub fn register_column(&mut self, config: VectorConfig) -> Result<(), BuildError> {
        if self.columns.iter().any(|c| c.config.column == config.column) {
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
        encode_blob(
            col.config.metric,
            col.config.dim,
            &col.ids,
            &col.vectors,
        )
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
    let dim = u32::from_le_bytes(body[0..4].try_into().unwrap()) as usize;
    let metric = metric_from_id(body[4])?;
    let n_segments = u32::from_le_bytes(body[5..9].try_into().unwrap()) as usize;
    let mut off = 9;
    let mut segments = Vec::with_capacity(n_segments);
    for _ in 0..n_segments {
        if body.len() < off + 4 {
            return Err("segmented cell posting segment header truncated".into());
        }
        let n_docs = u32::from_le_bytes(body[off..off + 4].try_into().unwrap()) as usize;
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

fn parse_segment_body(body: &[u8], allow_legacy_missing_norms: bool) -> Result<(DecodedPosting, usize), String> {
    if body.len() < 4 + 1 + 4 {
        return Err("cell posting header truncated".into());
    }
    let dim = u32::from_le_bytes(body[0..4].try_into().unwrap()) as usize;
    let metric = metric_from_id(body[4])?;
    let n_docs = u32::from_le_bytes(body[5..9].try_into().unwrap()) as usize;
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
        scale[d] = f32::from_le_bytes(
            body[scale_start + d * 4..scale_start + (d + 1) * 4]
                .try_into()
                .unwrap(),
        );
        offset[d] = f32::from_le_bytes(
            body[offset_start + d * 4..offset_start + (d + 1) * 4]
                .try_into()
                .unwrap(),
        );
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
            norms.push(f32::from_le_bytes(body[off..off + 4].try_into().unwrap()));
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
            } else if let Some(worst) = heap.peek() {
                if cmp_f32(hit.0.1, worst.0.1).is_lt() {
                    heap.pop();
                    heap.push(hit);
                }
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
            let mut norms = segment.per_doc_norms.as_ref().map(|_| Vec::with_capacity(segment.ids.len()));
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
                if segment.per_doc_norms.is_none() && matches!(segment.metric, Metric::L2Sq | Metric::Cosine) {
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
    Ok((encode_segmented_blob(dim.unwrap(), metric.unwrap(), &merged_segments)?, n_docs))
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
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
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
    let mut scale = vec![1.0; dim];
    let mut offset = vec![0.0; dim];
    for d in 0..dim {
        let span = max[d] - min[d];
        scale[d] = if span > 0.0 { span / SQ8_CODE_MAX } else { 1.0 };
        offset[d] = min[d];
    }
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
    match m {
        Metric::L2Sq => 0,
        Metric::Cosine => 1,
        Metric::NegDot => 2,
    }
}

fn metric_from_id(id: u8) -> Result<Metric, String> {
    match id {
        0 => Ok(Metric::L2Sq),
        1 => Ok(Metric::Cosine),
        2 => Ok(Metric::NegDot),
        _ => Err(format!("unknown cell posting metric id {id}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::builder::VectorConfig;
    use crate::superfile::vector::rerank_codec::RerankCodec;

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
    fn merge_encoded_blobs_remaps_ids_and_searches_segments() {
        let dim = 4usize;
        let ids_a = vec![0u32, 1, 2];
        let ids_b = vec![0u32, 1];
        let vecs_a = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        let vecs_b = vec![
            0.0, 0.0, 0.0, 1.0,
            2.0, 0.0, 0.0, 0.0,
        ];
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
        assert_eq!(hits[0].0, 3, "second blob row id must be remapped after surviving rows");
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
