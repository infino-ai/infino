// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Contiguous Sq8+ε cell posting blob for [`super::layout::VectorLayout::CellPosting`].
//!
//! One superfile carries one cell's postings. Cold read = one range GET on
//! `inf.vec.offset..+length`, then scan/rerank in memory.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::superfile::BuildError;
use crate::superfile::builder::VectorConfig;
use crate::superfile::vector::distance::{Metric, SQ8_RESIDUAL_DIVISOR, distance};

const MAGIC: &[u8] = b"infino.cell_posting.v1\n";
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
    let posting = encode_rows(vectors, ids, dim, &rows);
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
    Ok(out)
}

pub fn open_blob(bytes: &[u8]) -> Result<DecodedPosting, String> {
    let body = bytes
        .strip_prefix(MAGIC)
        .ok_or_else(|| "bad cell posting magic".to_string())?;
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
    Ok(DecodedPosting {
        dim,
        metric,
        ids: decode_ids(&body[ids_start..ids_start + ids_len]),
        scale,
        offset,
        rows: body[rows_start..rows_start + rows_len].to_vec(),
    })
}

pub fn search_blob(bytes: &[u8], query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, String> {
    let posting = open_blob(bytes)?;
    if query.len() != posting.dim {
        return Err("cell posting query dim mismatch".into());
    }
    if k == 0 || posting.ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut heap = BinaryHeap::<WorstHit>::new();
    for row in 0..posting.ids.len() {
        let d = row_distance(&posting, query, row);
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
    let mut out: Vec<(u32, f32)> = heap.into_iter().map(|h| h.0).collect();
    out.sort_by(|a, b| cmp_f32(a.1, b.1));
    Ok(out)
}

/// Decode all vectors from a cell posting blob (for merge/rebuild).
pub fn decode_all_vectors(bytes: &[u8]) -> Result<(Metric, usize, Vec<u32>, Vec<f32>), String> {
    let posting = open_blob(bytes)?;
    Ok((
        posting.metric,
        posting.dim,
        posting.ids.clone(),
        decode_all(&posting),
    ))
}

pub fn merge_blobs(inputs: &[&[u8]]) -> Result<Vec<u8>, String> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let first = open_blob(inputs[0])?;
    let mut all_ids = first.ids.clone();
    let mut all_vecs = decode_all(&first);
    for chunk in inputs.iter().skip(1) {
        let p = open_blob(chunk)?;
        if p.dim != first.dim || p.metric != first.metric {
            return Err("cell posting merge metric/dim mismatch".into());
        }
        all_ids.extend_from_slice(&p.ids);
        all_vecs.extend(decode_all(&p));
    }
    encode_blob(first.metric, first.dim, &all_ids, &all_vecs)
}

fn decode_all(p: &DecodedPosting) -> Vec<f32> {
    let mut out = vec![0f32; p.ids.len() * p.dim];
    for row in 0..p.ids.len() {
        decode_row(p, row, &mut out[row * p.dim..(row + 1) * p.dim]);
    }
    out
}

fn row_distance(p: &DecodedPosting, query: &[f32], row: usize) -> f32 {
    let mut decoded = vec![0f32; p.dim];
    decode_row(p, row, &mut decoded);
    distance(p.metric, query, &decoded)
}

fn decode_row(p: &DecodedPosting, row: usize, out: &mut [f32]) {
    let dim = p.dim;
    let base = row * dim * ROW_BYTES_PER_DIM;
    for d in 0..dim {
        let code = p.rows[base + d] as f32;
        let eps = p.rows[base + dim + d] as i8 as f32;
        let step = p.scale[d] / SQ8_RESIDUAL_DIVISOR;
        out[d] = p.offset[d] + code * p.scale[d] + eps * step;
    }
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
}

fn encode_rows(vectors: &[f32], ids: &[u32], dim: usize, rows: &[usize]) -> EncodedRows {
    if rows.is_empty() {
        return EncodedRows {
            ids: Vec::new(),
            scale: vec![1.0; dim],
            offset: vec![0.0; dim],
            rows: Vec::new(),
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
    let mut out_ids = Vec::with_capacity(rows.len());
    let mut encoded = Vec::with_capacity(rows.len() * dim * ROW_BYTES_PER_DIM);
    for &row in rows {
        out_ids.push(ids[row]);
        let src = &vectors[row * dim..(row + 1) * dim];
        let code_start = encoded.len();
        encoded.resize(code_start + dim * ROW_BYTES_PER_DIM, 0);
        let eps_start = code_start + dim;
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
        }
    }
    EncodedRows {
        ids: out_ids,
        scale,
        offset,
        rows: encoded,
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
        let blob = encode_blob(Metric::Cosine, dim, &ids, &vecs).expect("encode");
        let mut q = vec![0f32; dim];
        q[0] = 0.31;
        let hits = search_blob(&blob, &q, 5).expect("search");
        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].0, 31);
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
