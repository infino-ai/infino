// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Global vector cell index.
//!
//! This is the vector-routing structure from the plan in
//! `claude-plans/todo/024_global_vector_recall.md`: one table-level coarse
//! quantizer plus cell-organized compact postings. It is deliberately separate
//! from base superfiles; the base remains authoritative and this index is a
//! rebuildable acceleration layer.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};

use crate::superfile::vector::distance::{Metric, SQ8_RESIDUAL_DIVISOR, distance};
use crate::superfile::vector::kmeans::kmeans_with_assignments;

/// Lloyd iterations for the global coarse quantizer.
const GLOBAL_KMEANS_ITERS: usize = 5;
/// Maximum encoded Sq8 code value.
const SQ8_CODE_MAX: f32 = 255.0;
/// Symmetric clamp for the epsilon residual byte.
const EPSILON_I8_CLAMP: f32 = 127.0;
/// Serialized magic prefix for the index blob.
const SERIALIZED_MAGIC: &[u8] = b"infino.global_vector_index.v1\n";

/// A single vector-search result from [`GlobalVectorIndex::search`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalVectorHit {
    /// Caller-supplied id for the source vector. In the supertable integration
    /// this will be an indirection into `(superfile_uri, local_doc_id)`.
    pub id: u32,
    /// Distance, with smaller meaning closer for every [`Metric`].
    pub distance: f32,
}

/// Table-level vector router + cell-organized compact postings.
#[derive(Debug, Clone)]
pub struct GlobalVectorIndex {
    dim: usize,
    metric: Metric,
    centroids: Vec<f32>,
    cells: Vec<CellPosting>,
}

#[derive(Debug, Clone)]
struct CellPosting {
    ids: Vec<u32>,
    scale: Vec<f32>,
    offset: Vec<f32>,
    rows: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct IndexDto {
    dim: usize,
    metric: String,
    centroids: Vec<f32>,
    cells: Vec<CellDto>,
}

#[derive(Serialize, Deserialize)]
struct CellDto {
    ids: Vec<u32>,
    scale: Vec<f32>,
    offset: Vec<f32>,
    rows: Vec<u8>,
}

impl GlobalVectorIndex {
    /// Build a global cell index over row-major `vectors` and caller-provided
    /// ids. The physical cells are table-level, not superfile-local.
    pub fn build_with_ids(
        vectors: &[f32],
        ids: &[u32],
        dim: usize,
        n_cells: usize,
        metric: Metric,
        seed: u64,
    ) -> Self {
        assert!(dim > 0, "global vector index dim must be > 0");
        assert_eq!(vectors.len() % dim, 0, "vectors length must be n * dim");
        let n_docs = vectors.len() / dim;
        assert_eq!(ids.len(), n_docs, "ids length must match vector rows");
        assert!(n_docs > 0, "global vector index needs at least one vector");
        let n_cells = n_cells.max(1).min(n_docs);
        let (centroids, assignments) =
            kmeans_with_assignments(vectors, dim, n_cells, GLOBAL_KMEANS_ITERS, seed);

        let mut per_cell: Vec<Vec<usize>> = vec![Vec::new(); n_cells];
        for (row, &cell) in assignments.iter().enumerate() {
            per_cell[cell as usize].push(row);
        }
        let cells = per_cell
            .iter()
            .map(|rows| encode_cell(vectors, ids, dim, rows))
            .collect();
        Self {
            dim,
            metric,
            centroids,
            cells,
        }
    }

    /// Build with ids `0..n`.
    pub fn build(vectors: &[f32], dim: usize, n_cells: usize, metric: Metric, seed: u64) -> Self {
        let ids: Vec<u32> = (0..(vectors.len() / dim) as u32).collect();
        Self::build_with_ids(vectors, &ids, dim, n_cells, metric, seed)
    }

    /// Number of global cells.
    pub fn n_cells(&self) -> usize {
        self.cells.len()
    }

    /// Search `nprobe` global cells and return top-k hits by distance.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        _rerank_mult: usize,
    ) -> Vec<GlobalVectorHit> {
        if k == 0 || self.cells.is_empty() {
            return Vec::new();
        }
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        let nprobe = nprobe.max(1).min(self.cells.len());
        let mut centroid_scores: Vec<(usize, f32)> = (0..self.cells.len())
            .map(|cell| {
                let c = &self.centroids[cell * self.dim..(cell + 1) * self.dim];
                (cell, distance(self.metric, query, c))
            })
            .collect();
        centroid_scores.select_nth_unstable_by(nprobe - 1, |a, b| cmp_f32(a.1, b.1));
        centroid_scores.truncate(nprobe);

        let mut heap = BinaryHeap::<WorstHit>::new();
        for (cell_id, _) in centroid_scores {
            let cell = &self.cells[cell_id];
            for row in 0..cell.ids.len() {
                let d = cell.distance(self.metric, query, row, self.dim);
                let hit = WorstHit(GlobalVectorHit {
                    id: cell.ids[row],
                    distance: d,
                });
                if heap.len() < k {
                    heap.push(hit);
                } else if let Some(worst) = heap.peek() {
                    if cmp_f32(hit.0.distance, worst.0.distance).is_lt() {
                        heap.pop();
                        heap.push(hit);
                    }
                }
            }
        }
        let mut out: Vec<_> = heap.into_iter().map(|h| h.0).collect();
        out.sort_by(|a, b| cmp_f32(a.distance, b.distance));
        out
    }

    /// Append one vector by routing it to the nearest existing global cell and
    /// rebuilding only that cell's compact posting. This is the local-update
    /// primitive; split/merge policy is layered above it.
    pub fn insert(&mut self, id: u32, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim, "insert dim mismatch");
        let cell_id = self.nearest_centroid(vector);
        let mut ids = self.cells[cell_id].ids.clone();
        let mut vectors = self.cells[cell_id].decode_all(self.dim);
        ids.push(id);
        vectors.extend_from_slice(vector);
        let rows: Vec<usize> = (0..ids.len()).collect();
        self.cells[cell_id] = encode_cell(&vectors, &ids, self.dim, &rows);
        self.recompute_centroid(cell_id);
    }

    /// Split the largest cell with local k-means(2). Returns `true` when a
    /// split happened. This is the SPFresh/LIRE-style local rebalance hook.
    pub fn split_largest_cell(&mut self, seed: u64) -> bool {
        let Some((cell_id, cell)) = self
            .cells
            .iter()
            .enumerate()
            .max_by_key(|(_, cell)| cell.ids.len())
        else {
            return false;
        };
        if cell.ids.len() < 2 {
            return false;
        }
        let vectors = cell.decode_all(self.dim);
        let ids = cell.ids.clone();
        let (centroids, assignments) = kmeans_with_assignments(
            &vectors,
            self.dim,
            2,
            GLOBAL_KMEANS_ITERS,
            seed.wrapping_add(cell_id as u64),
        );
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (row, &a) in assignments.iter().enumerate() {
            if a == 0 {
                left.push(row);
            } else {
                right.push(row);
            }
        }
        if left.is_empty() || right.is_empty() {
            return false;
        }
        self.cells[cell_id] = encode_cell(&vectors, &ids, self.dim, &left);
        self.cells.push(encode_cell(&vectors, &ids, self.dim, &right));
        self.centroids[cell_id * self.dim..(cell_id + 1) * self.dim]
            .copy_from_slice(&centroids[..self.dim]);
        self.centroids.extend_from_slice(&centroids[self.dim..]);
        true
    }

    /// Serialize to a stable JSON payload with a magic prefix. This is not the
    /// final object-store posting layout; it is a deterministic bring-up format
    /// for tests and offline validation.
    pub fn serialize(&self) -> Vec<u8> {
        let dto = IndexDto {
            dim: self.dim,
            metric: metric_to_str(self.metric).to_string(),
            centroids: self.centroids.clone(),
            cells: self
                .cells
                .iter()
                .map(|cell| CellDto {
                    ids: cell.ids.clone(),
                    scale: cell.scale.clone(),
                    offset: cell.offset.clone(),
                    rows: cell.rows.clone(),
                })
                .collect(),
        };
        let mut out = SERIALIZED_MAGIC.to_vec();
        out.extend_from_slice(&serde_json::to_vec(&dto).expect("serialize global vector index"));
        out
    }

    /// Open a serialized index.
    pub fn open(bytes: &[u8]) -> Result<Self, String> {
        let body = bytes
            .strip_prefix(SERIALIZED_MAGIC)
            .ok_or_else(|| "bad global vector index magic".to_string())?;
        let dto: IndexDto = serde_json::from_slice(body).map_err(|e| e.to_string())?;
        let metric = metric_from_str(&dto.metric)?;
        if dto.dim == 0 {
            return Err("global vector index dim must be > 0".to_string());
        }
        if dto.centroids.len() != dto.cells.len() * dto.dim {
            return Err("global vector index centroid length mismatch".to_string());
        }
        let mut cells = Vec::with_capacity(dto.cells.len());
        for cell in dto.cells {
            if cell.scale.len() != dto.dim || cell.offset.len() != dto.dim {
                return Err("global vector index cell quantizer dim mismatch".to_string());
            }
            if cell.rows.len() != cell.ids.len() * dto.dim * CELL_ROW_BYTES_PER_DIM {
                return Err("global vector index cell row byte length mismatch".to_string());
            }
            cells.push(CellPosting {
                ids: cell.ids,
                scale: cell.scale,
                offset: cell.offset,
                rows: cell.rows,
            });
        }
        Ok(Self {
            dim: dto.dim,
            metric,
            centroids: dto.centroids,
            cells,
        })
    }

    fn nearest_centroid(&self, vector: &[f32]) -> usize {
        (0..self.cells.len())
            .min_by(|&a, &b| {
                let ca = &self.centroids[a * self.dim..(a + 1) * self.dim];
                let cb = &self.centroids[b * self.dim..(b + 1) * self.dim];
                cmp_f32(
                    distance(self.metric, vector, ca),
                    distance(self.metric, vector, cb),
                )
            })
            .unwrap_or(0)
    }

    fn recompute_centroid(&mut self, cell_id: usize) {
        let vectors = self.cells[cell_id].decode_all(self.dim);
        if vectors.is_empty() {
            return;
        }
        let n = vectors.len() / self.dim;
        let dst = &mut self.centroids[cell_id * self.dim..(cell_id + 1) * self.dim];
        dst.fill(0.0);
        for row in vectors.chunks_exact(self.dim) {
            for d in 0..self.dim {
                dst[d] += row[d];
            }
        }
        for v in dst {
            *v /= n as f32;
        }
    }
}

/// One Sq8 byte + one epsilon byte per dimension.
const CELL_ROW_BYTES_PER_DIM: usize = 2;

impl CellPosting {
    fn distance(&self, metric: Metric, query: &[f32], row: usize, dim: usize) -> f32 {
        let mut decoded = vec![0.0f32; dim];
        self.decode_row(row, dim, &mut decoded);
        distance(metric, query, &decoded)
    }

    fn decode_all(&self, dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.ids.len() * dim];
        for row in 0..self.ids.len() {
            self.decode_row(row, dim, &mut out[row * dim..(row + 1) * dim]);
        }
        out
    }

    fn decode_row(&self, row: usize, dim: usize, out: &mut [f32]) {
        let base = row * dim * CELL_ROW_BYTES_PER_DIM;
        for d in 0..dim {
            let code = self.rows[base + d] as f32;
            let eps = self.rows[base + dim + d] as i8 as f32;
            let step = self.scale[d] / SQ8_RESIDUAL_DIVISOR;
            out[d] = self.offset[d] + code * self.scale[d] + eps * step;
        }
    }
}

fn encode_cell(vectors: &[f32], ids: &[u32], dim: usize, rows: &[usize]) -> CellPosting {
    debug_assert_eq!(vectors.len() % dim, 0);
    if rows.is_empty() {
        return CellPosting {
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
    let mut encoded = Vec::with_capacity(rows.len() * dim * CELL_ROW_BYTES_PER_DIM);
    for &row in rows {
        out_ids.push(ids[row]);
        let src = &vectors[row * dim..(row + 1) * dim];
        let code_start = encoded.len();
        encoded.resize(code_start + dim * CELL_ROW_BYTES_PER_DIM, 0);
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
    CellPosting {
        ids: out_ids,
        scale,
        offset,
        rows: encoded,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct WorstHit(GlobalVectorHit);

impl Eq for WorstHit {}
impl Ord for WorstHit {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_f32(self.0.distance, other.0.distance)
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

fn metric_to_str(metric: Metric) -> &'static str {
    match metric {
        Metric::Cosine => "cosine",
        Metric::L2Sq => "l2sq",
        Metric::NegDot => "negdot",
    }
}

fn metric_from_str(s: &str) -> Result<Metric, String> {
    match s {
        "cosine" => Ok(Metric::Cosine),
        "l2sq" => Ok(Metric::L2Sq),
        "negdot" => Ok(Metric::NegDot),
        other => Err(format!("unknown global vector index metric {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DIM: usize = 32;
    const TEST_CELLS: usize = 16;
    const TEST_DOCS_PER_CELL: usize = 64;
    const TEST_SEED: u64 = 7;
    const TEST_TOP_K: usize = 10;
    const TEST_NPROBE: usize = 2;
    const TEST_RERANK_MULT: usize = 20;
    const RECALL_FLOOR: f32 = 0.99;

    fn clustered_vectors() -> (Vec<f32>, Vec<u32>, Vec<Vec<f32>>) {
        let mut vectors = Vec::with_capacity(TEST_CELLS * TEST_DOCS_PER_CELL * TEST_DIM);
        let mut ids = Vec::with_capacity(TEST_CELLS * TEST_DOCS_PER_CELL);
        let mut queries = Vec::with_capacity(TEST_CELLS);
        for cell in 0..TEST_CELLS {
            let active = cell % TEST_DIM;
            let mut q = vec![0.0f32; TEST_DIM];
            q[active] = 1.0;
            queries.push(q);
            for i in 0..TEST_DOCS_PER_CELL {
                ids.push((cell * TEST_DOCS_PER_CELL + i) as u32);
                for d in 0..TEST_DIM {
                    let v = if d == active {
                        1.0
                    } else if d == (active + 1 + i % (TEST_DIM - 1)) % TEST_DIM {
                        0.001
                    } else {
                        0.0
                    };
                    vectors.push(v);
                }
            }
        }
        (vectors, ids, queries)
    }

    fn exact_topk(vectors: &[f32], ids: &[u32], query: &[f32], k: usize) -> Vec<u32> {
        let mut scored: Vec<_> = ids
            .iter()
            .enumerate()
            .map(|(row, &id)| {
                let v = &vectors[row * TEST_DIM..(row + 1) * TEST_DIM];
                (id, distance(Metric::Cosine, query, v))
            })
            .collect();
        scored.sort_by(|a, b| cmp_f32(a.1, b.1));
        scored.truncate(k);
        scored.into_iter().map(|(id, _)| id).collect()
    }

    fn recall(actual: &[GlobalVectorHit], expected: &[u32]) -> f32 {
        let got: std::collections::HashSet<u32> = actual.iter().map(|h| h.id).collect();
        let hit = expected.iter().filter(|id| got.contains(id)).count();
        hit as f32 / expected.len() as f32
    }

    #[test]
    fn routed_cells_recover_exact_topk_on_clustered_corpus() {
        let (vectors, ids, queries) = clustered_vectors();
        let index = GlobalVectorIndex::build_with_ids(
            &vectors,
            &ids,
            TEST_DIM,
            TEST_CELLS,
            Metric::Cosine,
            TEST_SEED,
        );
        assert_eq!(index.n_cells(), TEST_CELLS);
        for q in &queries {
            let got = index.search(q, TEST_TOP_K, TEST_NPROBE, TEST_RERANK_MULT);
            let expected = exact_topk(&vectors, &ids, q, TEST_TOP_K);
            assert!(
                recall(&got, &expected) >= RECALL_FLOOR,
                "query recall below floor: got={got:?} expected={expected:?}"
            );
        }
    }

    #[test]
    fn serialize_roundtrip_preserves_search_results() {
        let (vectors, ids, queries) = clustered_vectors();
        let index = GlobalVectorIndex::build_with_ids(
            &vectors,
            &ids,
            TEST_DIM,
            TEST_CELLS,
            Metric::Cosine,
            TEST_SEED,
        );
        let reopened = GlobalVectorIndex::open(&index.serialize()).expect("open serialized index");
        let q = &queries[0];
        assert_eq!(
            index.search(q, TEST_TOP_K, TEST_NPROBE, TEST_RERANK_MULT),
            reopened.search(q, TEST_TOP_K, TEST_NPROBE, TEST_RERANK_MULT),
        );
    }

    #[test]
    fn insert_and_split_largest_cell_keep_index_searchable() {
        let (vectors, ids, queries) = clustered_vectors();
        let mut index = GlobalVectorIndex::build_with_ids(
            &vectors,
            &ids,
            TEST_DIM,
            TEST_CELLS,
            Metric::Cosine,
            TEST_SEED,
        );
        let before = index.n_cells();
        index.insert(999_999, &queries[0]);
        assert!(index
            .search(&queries[0], TEST_TOP_K, TEST_NPROBE, TEST_RERANK_MULT)
            .iter()
            .any(|h| h.id == 999_999));
        assert!(index.split_largest_cell(TEST_SEED));
        assert_eq!(index.n_cells(), before + 1);
    }
}
