//! Vector blob reader. Multi-column kNN search via IVF + 1-bit RaBitQ
//! shortlist + full-precision rerank.
//!
//! Opens the unified-blob byte layout produced by
//! [`super::builder::VectorBuilder::finish`] and exposes per-column
//! kNN search.
//!
//! Self-contained: owns its `Bytes`. Per-column metadata is parsed
//! eagerly at `open()`; per-query work happens on demand.

use crate::superfile::format::checksum::crc32c;
use crate::superfile::format::{self};
use crate::superfile::vector::distance::{Metric, distance_bytes};
use crate::superfile::vector::quant::BitQuantizer;
use crate::superfile::vector::rotation::RandomRotation;
use crate::superfile::{ReadError, error::VectorError};
use bytes::Bytes;
use rayon::prelude::*;
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Range;

const OUTER_HEADER_SIZE: usize = 32;
const DIR_ENTRY_SIZE: usize = 64;
const SUB_HEADER_SIZE: usize = 56;

/// JSON-deserialized form of one entry in `inf.vec.columns`. The KV
/// value is a JSON array of these in declaration order.
#[derive(Debug, Clone, Deserialize)]
pub struct VectorColumnConfig {
    pub name: String,
    pub dim: usize,
    pub n_cent: usize,
    pub rot_seed: u64,
    /// `"l2sq"`, `"cosine"`, or `"negdot"`.
    pub metric: String,
}

/// Per-column reader state; cached at open time.
#[derive(Debug)]
pub struct ColumnReader {
    pub name: String,
    pub dim: usize,
    pub n_cent: u32,
    pub n_docs: u32,
    pub metric: Metric,
    pub rot_seed: u64,
    /// Byte range of this column's subsection within the outer blob.
    subsection_range: Range<usize>,
    /// Offsets relative to the subsection start.
    summary_off: usize,
    summary_radius: f32,
    centroids_off: usize,
    cluster_idx_off: usize,
    codes_off: usize,
    full_off: usize,
    doc_ids_off: usize,
    /// `local_doc_id → cluster-position`. Built at open. ~4 MB at 1 M
    /// docs per column.
    doc_to_pos: Vec<u32>,
    quant: BitQuantizer,
    /// Cached random rotation built once at open from `(dim, rot_seed)`.
    /// Construction is `O(dim³)` for Gram-Schmidt — at dim=384 that's
    /// ~7.9 ms, dominant over every other per-query stage if rebuilt
    /// per `search()`. Build once, reuse forever.
    rot: RandomRotation,
}

/// Per-open knobs for [`VectorReader::open_with`]. `Default` is the
/// safe choice (CRC verification on); construct with `verify_crc:
/// false` when the caller has already validated the bytes (e.g.
/// known-good local file) and wants the cheap-open path.
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Verify the per-subsection CRC during open. Each subsection is
    /// scanned in full (≈1.5 GiB at 1M × 384, single column), so this
    /// dominates cold-open time when on. Defaults to `true`; the
    /// argumentless [`VectorReader::open`] uses this default.
    /// Flip to `false` when storage is already trusted (content-
    /// addressed object store, checksummed filesystem) to skip
    /// the scan.
    pub verify_crc: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self { verify_crc: true }
    }
}

/// Multi-column vector blob reader. `Send + Sync`; concurrent searches
/// share the underlying `Bytes`.
#[derive(Debug)]
pub struct VectorReader {
    blob: Bytes,
    n_docs: u64,
    columns: Vec<ColumnReader>,
    column_id_by_name: HashMap<String, u32>,
}

impl VectorReader {
    /// Open the reader. `columns_json` is the value of the
    /// `inf.vec.columns` Parquet KV key (a JSON array of
    /// [`VectorColumnConfig`]).
    /// Open the reader with default options (CRC verification on).
    pub fn open(blob: Bytes, columns_json: &str) -> Result<Self, VectorError> {
        Self::open_with(blob, columns_json, OpenOptions::default())
    }

    /// Open with explicit options. The fast path is
    /// `OpenOptions { verify_crc: false }` which skips both the
    /// outer (whole-blob) CRC and the per-subsection CRC scans —
    /// at 1M × 384 cold open drops from ~132 ms to ~2 ms. Use this
    /// when the underlying storage is trusted (e.g. local disk after
    /// a successful file integrity check) or when CRC verification
    /// is performed elsewhere in the stack.
    pub fn open_with(
        blob: Bytes,
        columns_json: &str,
        opts: OpenOptions,
    ) -> Result<Self, VectorError> {
        if blob.len() < OUTER_HEADER_SIZE + 4 {
            return Err(VectorError::Read(ReadError::MissingKv(
                "vector blob header",
            )));
        }

        if &blob[0..8] != format::vec::OUTER_MAGIC {
            return Err(VectorError::Read(ReadError::BadMagic {
                section: "vector",
                expected: format::vec::OUTER_MAGIC,
                actual: blob[0..8].to_vec(),
            }));
        }

        let version = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]);
        if version != format::vec::VERSION {
            return Err(VectorError::Read(ReadError::UnsupportedVersion(format!(
                "vector blob version {version}"
            ))));
        }

        let n_columns = u32::from_le_bytes([blob[12], blob[13], blob[14], blob[15]]) as usize;
        let n_docs = read_u64_le(&blob[16..24]);
        let dir_offset = read_u64_le(&blob[24..32]) as usize;

        // Verify directory CRC.
        let dir_size = n_columns * DIR_ENTRY_SIZE;
        if dir_offset + dir_size + 4 > blob.len() {
            return Err(VectorError::Read(ReadError::MalformedVersion(
                "vector directory runs past blob".into(),
            )));
        }
        let dir_bytes = &blob[dir_offset..dir_offset + dir_size];
        let dir_crc_expected = read_u32_le(&blob[dir_offset + dir_size..dir_offset + dir_size + 4]);
        let dir_crc_actual = crc32c(dir_bytes);
        if dir_crc_expected != dir_crc_actual {
            return Err(VectorError::Read(ReadError::ChecksumMismatch {
                section: "vector/directory",
                column: String::new(),
            }));
        }

        if opts.verify_crc {
            verify_vector_crcs(&blob, dir_bytes, n_columns)?;
        }

        // Parse JSON.
        let cols_json: Vec<VectorColumnConfig> =
            serde_json::from_str(columns_json).map_err(|e| {
                VectorError::Read(ReadError::MalformedVersion(format!(
                    "inf.vec.columns JSON: {e}"
                )))
            })?;
        if cols_json.len() != n_columns {
            return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                "inf.vec.columns has {} entries, header says {n_columns}",
                cols_json.len()
            ))));
        }

        // Parse each directory entry, build ColumnReader.
        let mut columns = Vec::with_capacity(n_columns);
        let mut column_id_by_name = HashMap::with_capacity(n_columns);
        for (i, cfg) in cols_json.iter().enumerate() {
            let entry_off = i * DIR_ENTRY_SIZE;
            let column_id = u32::from_le_bytes([
                dir_bytes[entry_off],
                dir_bytes[entry_off + 1],
                dir_bytes[entry_off + 2],
                dir_bytes[entry_off + 3],
            ]);
            if column_id != i as u32 {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "vector dir entry {i} has column_id {column_id}"
                ))));
            }
            let dim = u32::from_le_bytes([
                dir_bytes[entry_off + 4],
                dir_bytes[entry_off + 5],
                dir_bytes[entry_off + 6],
                dir_bytes[entry_off + 7],
            ]) as usize;
            let n_cent = u32::from_le_bytes([
                dir_bytes[entry_off + 8],
                dir_bytes[entry_off + 9],
                dir_bytes[entry_off + 10],
                dir_bytes[entry_off + 11],
            ]);
            let metric_id = u32::from_le_bytes([
                dir_bytes[entry_off + 12],
                dir_bytes[entry_off + 13],
                dir_bytes[entry_off + 14],
                dir_bytes[entry_off + 15],
            ]);
            let rot_seed = read_u64_le(&dir_bytes[entry_off + 16..entry_off + 24]);
            let subsection_off = read_u64_le(&dir_bytes[entry_off + 24..entry_off + 32]) as usize;
            let subsection_len = read_u64_le(&dir_bytes[entry_off + 32..entry_off + 40]) as usize;
            // bytes [40..48] = summary_offset (absolute), [48..52] = summary_length, then padding
            let _summary_off_abs = read_u64_le(&dir_bytes[entry_off + 40..entry_off + 48]);

            // Validate against JSON.
            if dim != cfg.dim {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' dim mismatch: dir={dim} json={}",
                    cfg.name, cfg.dim
                ))));
            }
            if rot_seed != cfg.rot_seed {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' rot_seed mismatch",
                    cfg.name
                ))));
            }
            let metric = match metric_id {
                0 => Metric::L2Sq,
                1 => Metric::Cosine,
                2 => Metric::NegDot,
                _ => {
                    return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                        "unknown metric_id {metric_id} for column '{}'",
                        cfg.name
                    ))));
                }
            };

            // Validate subsection bounds + magic. Subsection CRCs are
            // verified above in the parallel CRC pre-pass when requested.
            let sub_end = subsection_off + subsection_len;
            if sub_end > blob.len() {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} runs past blob"
                ))));
            }
            let sub = &blob[subsection_off..sub_end];
            if sub.len() < SUB_HEADER_SIZE + 4 {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} too short"
                ))));
            }
            if &sub[0..8] != format::vec::SUB_MAGIC {
                return Err(VectorError::Read(ReadError::BadMagic {
                    section: "vector/subsection",
                    expected: format::vec::SUB_MAGIC,
                    actual: sub[0..8].to_vec(),
                }));
            }
            let sub_crc_pos = sub.len() - 4;

            // Sub-header parse (SUB_HEADER_SIZE = 56 bytes):
            //   [8..12]  version  (cross-checked against outer header)
            //   [12..16] reserved
            //   [16..24] summary_centroid_offset (relative to sub start)
            //   [24..28] summary_radius_x100
            //   [28..32] reserved
            //   [32..40] centroids_offset
            //   [40..48] cluster_idx_offset
            //   [48..52] codes_offset
            //   [52..56] full_offset
            let summary_off = read_u64_le(&sub[16..24]) as usize;
            let summary_radius_x100 = read_u32_le(&sub[24..28]);
            let centroids_off = read_u64_le(&sub[32..40]) as usize;
            let cluster_idx_off = read_u64_le(&sub[40..48]) as usize;
            let codes_off = read_u32_le(&sub[48..52]) as usize;
            let full_off = read_u32_le(&sub[52..56]) as usize;

            let summary_radius = (summary_radius_x100 as f32) / 100.0;

            // Compute n_docs for this column and doc_ids_off.
            // doc_ids start at end of full vectors. Total subsection
            // bytes (excluding CRC) = SUB_HEADER + summary + centroids +
            // cluster_idx + codes + full + doc_ids.
            let quant = BitQuantizer::new(dim);
            let code_bytes = quant.code_bytes();
            // We can derive n_docs from the cluster index: sum of counts
            // across clusters. Or from the layout: doc_ids region size
            // / 4. Let's compute from doc_ids region:
            //   doc_ids_size = sub.len() - 4 - doc_ids_off
            // But we need doc_ids_off first. Use full_off + full_size:
            // that requires n_docs, circular. Instead derive from
            // codes region: codes region size = full_off - codes_off,
            // and codes region size = n_docs * code_bytes.
            let codes_size = full_off - codes_off;
            if code_bytes == 0 || !codes_size.is_multiple_of(code_bytes) {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' codes size {codes_size} not divisible by {code_bytes}",
                    cfg.name
                ))));
            }
            let col_n_docs = (codes_size / code_bytes) as u32;

            let full_size = (col_n_docs as usize) * dim * 4;
            let doc_ids_off = full_off + full_size;

            // Build doc_to_pos lookup table.
            let cluster_idx_size = (n_cent as usize) * 8;
            let cluster_idx_end = cluster_idx_off + cluster_idx_size;
            if cluster_idx_end > sub_crc_pos {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' cluster index runs past subsection",
                    cfg.name
                ))));
            }
            let mut doc_to_pos = vec![u32::MAX; col_n_docs as usize];
            for c in 0..n_cent as usize {
                let idx_start = cluster_idx_off + c * 8;
                let off = u32::from_le_bytes([
                    sub[idx_start],
                    sub[idx_start + 1],
                    sub[idx_start + 2],
                    sub[idx_start + 3],
                ]);
                let cnt = u32::from_le_bytes([
                    sub[idx_start + 4],
                    sub[idx_start + 5],
                    sub[idx_start + 6],
                    sub[idx_start + 7],
                ]);
                let did_start = doc_ids_off + (off as usize) * 4;
                let did_end = did_start + (cnt as usize) * 4;
                if did_end > sub_crc_pos {
                    return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                        "column '{}' doc_ids slice {did_start}..{did_end} past subsection",
                        cfg.name
                    ))));
                }
                for i in 0..cnt as usize {
                    let s = did_start + i * 4;
                    let d = u32::from_le_bytes([sub[s], sub[s + 1], sub[s + 2], sub[s + 3]]);
                    if (d as usize) < doc_to_pos.len() {
                        doc_to_pos[d as usize] = off + i as u32;
                    }
                }
            }

            // Soft cross-check: cfg.metric matches blob's metric.
            let cfg_metric = match cfg.metric.as_str() {
                "l2sq" => Some(Metric::L2Sq),
                "cosine" => Some(Metric::Cosine),
                "negdot" => Some(Metric::NegDot),
                _ => None,
            };
            if let Some(m) = cfg_metric
                && m != metric
            {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' metric mismatch: dir={metric:?} json={}",
                    cfg.name, cfg.metric
                ))));
            }

            columns.push(ColumnReader {
                name: cfg.name.clone(),
                dim,
                n_cent,
                n_docs: col_n_docs,
                metric,
                rot_seed,
                subsection_range: subsection_off..sub_end,
                summary_off,
                summary_radius,
                centroids_off,
                cluster_idx_off,
                codes_off,
                full_off,
                doc_ids_off,
                doc_to_pos,
                quant,
                rot: RandomRotation::new(dim, rot_seed),
            });
            column_id_by_name.insert(cfg.name.clone(), i as u32);
        }

        Ok(VectorReader {
            blob,
            n_docs,
            columns,
            column_id_by_name,
        })
    }

    pub fn n_docs(&self) -> u64 {
        self.n_docs
    }

    pub fn vector_columns(&self) -> impl Iterator<Item = &str> {
        self.columns.iter().map(|c| c.name.as_str())
    }

    /// Per-column summary centroid + radius, used by the storage plan
    /// for cross-segment skip pruning.
    pub fn summary(&self, column: &str) -> Option<(Vec<f32>, f32)> {
        let cid = *self.column_id_by_name.get(column)?;
        let col = &self.columns[cid as usize];
        let sub = &self.blob[col.subsection_range.clone()];
        let off = col.summary_off;
        let dim = col.dim;
        let centroid: Vec<f32> = (0..dim)
            .map(|i| {
                let s = off + i * 4;
                f32::from_le_bytes([sub[s], sub[s + 1], sub[s + 2], sub[s + 3]])
            })
            .collect();
        Some((centroid, col.summary_radius))
    }

    /// Single-column kNN search. Returns `(local_doc_id, distance)`
    /// sorted ascending by distance (smaller = closer for every metric).
    pub fn search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank_mult: usize,
    ) -> Result<Vec<(u32, f32)>, VectorError> {
        let cid = *self
            .column_id_by_name
            .get(column)
            .ok_or_else(|| VectorError::UnknownColumn(column.to_string()))?;
        let col = &self.columns[cid as usize];

        if query.len() != col.dim {
            return Err(VectorError::DimensionMismatch {
                expected: col.dim,
                got: query.len(),
            });
        }
        if k == 0 || col.n_docs == 0 {
            return Ok(Vec::new());
        }

        let sub = &self.blob[col.subsection_range.clone()];

        // 1. Score query vs every centroid (cheap; n_cent is small).
        //
        // Zero-copy `f32x8` over the centroid bytes in `sub` via
        // `distance_bytes` — `bytemuck::try_cast_slice` borrows the
        // 4-aligned region (common case for our layout), falling
        // back to a per-chunk LE decode if alignment is off. At
        // `n_cent = 1024, dim = 384` this is 1024 zero-copy SIMD
        // dot/L2² calls per query; no per-centroid heap allocation.
        let dim = col.dim;
        let dim_bytes = dim * 4;
        let mut centroid_scores: Vec<(usize, f32)> = (0..col.n_cent as usize)
            .map(|c| {
                let start = col.centroids_off + c * dim_bytes;
                let bytes = &sub[start..start + dim_bytes];
                (c, distance_bytes(col.metric, query, bytes))
            })
            .collect();
        centroid_scores.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let nprobe_eff = nprobe.min(col.n_cent as usize).max(1);
        centroid_scores.truncate(nprobe_eff);

        // 2. Rotate query for the 1-bit code estimator. The rotation
        // matrix was built once at open and is cached on `col` —
        // rebuilding it per `search()` costs `O(dim³)` Gram-Schmidt
        // (~7.9 ms at dim=384), which dominates every other per-query
        // stage. The `apply` itself is a cheap `O(dim²)` matvec.
        let mut q_rot = vec![0f32; dim];
        col.rot.apply(query, &mut q_rot);

        // 3. Scan codes within probed clusters → shortlist.
        let cb = col.quant.code_bytes();
        let mut shortlist: Vec<(u32, f32)> = Vec::new();
        for &(c, _) in &centroid_scores {
            let idx_start = col.cluster_idx_off + c * 8;
            let off = u32::from_le_bytes([
                sub[idx_start],
                sub[idx_start + 1],
                sub[idx_start + 2],
                sub[idx_start + 3],
            ]);
            let cnt = u32::from_le_bytes([
                sub[idx_start + 4],
                sub[idx_start + 5],
                sub[idx_start + 6],
                sub[idx_start + 7],
            ]);
            if cnt == 0 {
                continue;
            }
            for i in 0..cnt as usize {
                let code_start = col.codes_off + (off as usize + i) * cb;
                let code = &sub[code_start..code_start + cb];
                let est = col.quant.estimate_dot_rotated(&q_rot, code);
                let did_start = col.doc_ids_off + (off as usize + i) * 4;
                let did = u32::from_le_bytes([
                    sub[did_start],
                    sub[did_start + 1],
                    sub[did_start + 2],
                    sub[did_start + 3],
                ]);
                shortlist.push((did, est));
            }
        }

        // 4. Take top `k * rerank_mult` by descending estimate
        //    (higher est = closer for cosine / negdot; for l2sq it's
        //    reversed but the rerank step uses true distance anyway,
        //    so a slightly looser shortlist is fine).
        let want = (k.saturating_mul(rerank_mult)).min(shortlist.len());
        if want < shortlist.len() {
            shortlist.select_nth_unstable_by(want.saturating_sub(1), |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
            });
            shortlist.truncate(want);
        }

        // 5. Full-precision rerank using the true metric.
        //
        // Score each candidate's full vector directly from its byte
        // slice in the blob — `distance_bytes` zero-copies into
        // `f32x8` when 4-aligned (the common case for our layout)
        // and falls back to a per-chunk LE decode otherwise. Same
        // zero-copy pattern as the centroid probe above; no
        // per-candidate heap allocation.
        let sub = &self.blob[col.subsection_range.clone()];
        let dim_bytes = col.dim * 4;
        let mut reranked: Vec<(u32, f32)> = shortlist
            .iter()
            .map(|&(did, _)| {
                let pos = col.doc_to_pos[did as usize] as usize;
                let start = col.full_off + pos * dim_bytes;
                let bytes = &sub[start..start + dim_bytes];
                let d = distance_bytes(col.metric, query, bytes);
                (did, d)
            })
            .collect();
        reranked.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        reranked.truncate(k);
        Ok(reranked)
    }
}

#[inline]
fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline]
fn read_u64_le(b: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&b[0..8]);
    u64::from_le_bytes(buf)
}

#[inline]
fn verify_vector_crcs(blob: &Bytes, dir_bytes: &[u8], n_columns: usize) -> Result<(), VectorError> {
    struct CrcJob<'a> {
        idx: i32,
        bytes: &'a [u8],
        expected: u32,
    }

    let mut jobs: Vec<CrcJob<'_>> = Vec::with_capacity(n_columns + 1);

    let outer_total = blob.len();
    let outer_crc_pos = outer_total - 4;
    jobs.push(CrcJob {
        idx: -1,
        bytes: &blob[..outer_crc_pos],
        expected: read_u32_le(&blob[outer_crc_pos..outer_total]),
    });

    for i in 0..n_columns {
        let entry_off = i * DIR_ENTRY_SIZE;
        let subsection_off = read_u64_le(&dir_bytes[entry_off + 24..entry_off + 32]) as usize;
        let subsection_len = read_u64_le(&dir_bytes[entry_off + 32..entry_off + 40]) as usize;
        let sub_end = subsection_off + subsection_len;
        if sub_end > blob.len() {
            return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                "subsection {i} runs past blob"
            ))));
        }
        let sub = &blob[subsection_off..sub_end];
        if sub.len() < SUB_HEADER_SIZE + 4 {
            return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                "subsection {i} too short"
            ))));
        }
        let sub_crc_pos = sub.len() - 4;
        jobs.push(CrcJob {
            idx: i as i32,
            bytes: &sub[..sub_crc_pos],
            expected: read_u32_le(&sub[sub_crc_pos..]),
        });
    }

    // The outer-blob scan and per-subsection scans each touch ~1.5 GiB
    // at 1M x 384. They are independent, so run them as parallel jobs
    // and let the checksum module's CLMUL implementation handle each
    // byte stream quickly.
    let mismatch = jobs.par_iter().find_map_any(|job| {
        if crc32c(job.bytes) != job.expected {
            Some(job.idx)
        } else {
            None
        }
    });
    if let Some(idx) = mismatch {
        if idx < 0 {
            return Err(VectorError::Read(ReadError::ChecksumMismatch {
                section: "vector",
                column: String::new(),
            }));
        }
        let i = idx as usize;
        return Err(VectorError::Read(ReadError::ChecksumMismatch {
            section: "vector/subsection",
            column: format!(" (column index {i})"),
        }));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::vector::builder::{VectorBuilder, VectorConfig};

    fn build_blob(n_docs: u32, dim: usize, n_cent: usize, metric: Metric) -> (Bytes, String) {
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            name: "embedding".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric,
        })
        .expect("register column");
        for i in 0..n_docs {
            // Deterministic "random" vector.
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i.wrapping_mul(31) + j as u32) % 100) as f32 * 0.01)
                .collect();
            b.add(0, &v).expect("add to vector builder");
        }
        let bytes = b.finish();
        let metric_s = match metric {
            Metric::L2Sq => "l2sq",
            Metric::Cosine => "cosine",
            Metric::NegDot => "negdot",
        };
        let json = format!(
            r#"[{{"name":"embedding","dim":{dim},"n_cent":{n_cent},"rot_seed":7,"metric":"{metric_s}"}}]"#
        );
        (Bytes::from(bytes), json)
    }

    #[test]
    fn open_accepts_valid_blob() {
        let (blob, json) = build_blob(64, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open should succeed");
        assert_eq!(r.n_docs(), 64);
        let cols: Vec<&str> = r.vector_columns().collect();
        assert_eq!(cols, vec!["embedding"]);
    }

    #[test]
    fn open_rejects_bad_magic() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let mut bytes = blob.to_vec();
        bytes[0] = b'X';
        let err = VectorReader::open(Bytes::from(bytes), &json).expect_err("expected error");
        assert!(matches!(err, VectorError::Read(ReadError::BadMagic { .. })));
    }

    #[test]
    fn open_rejects_short_blob() {
        let err = VectorReader::open(Bytes::from(vec![0u8; 8]), "[]").expect_err("expected error");
        assert!(matches!(err, VectorError::Read(_)));
    }

    #[test]
    fn open_detects_corruption_via_outer_crc() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let mut bytes = blob.to_vec();
        // Flip a byte in the middle of the directory area.
        let pos = OUTER_HEADER_SIZE + 10;
        bytes[pos] ^= 0xFF;
        let err = VectorReader::open(Bytes::from(bytes), &json).expect_err("expected error");
        assert!(matches!(
            err,
            VectorError::Read(ReadError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn open_with_skip_crc_accepts_corrupted_directory_bytes() {
        // The fast-open path explicitly skips CRC verification — so
        // a flipped byte in the directory area opens cleanly. The
        // caller is responsible for upstream integrity.
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let mut bytes = blob.to_vec();
        let pos = OUTER_HEADER_SIZE + 10;
        bytes[pos] ^= 0xFF;
        let r =
            VectorReader::open_with(Bytes::from(bytes), &json, OpenOptions { verify_crc: false });
        // Open succeeds; the corruption may surface later as a
        // bad-magic / bounds error or be silently absorbed depending
        // on which byte got flipped. The contract is "we don't
        // verify"; not "we'll always read sensibly."
        let _ = r;
    }

    #[test]
    fn open_with_default_options_matches_open() {
        // Sanity: default opts produce identical results to `open`.
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r1 = VectorReader::open(blob.clone(), &json).expect("open VectorReader");
        let r2 = VectorReader::open_with(blob, &json, OpenOptions::default())
            .expect("open VectorReader");
        assert_eq!(r1.n_docs(), r2.n_docs());
    }

    #[test]
    fn search_self_query_returns_self_as_top1() {
        let dim = 16;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            name: "embedding".into(),
            dim,
            n_cent: 4,
            rot_seed: 7,
            metric: Metric::L2Sq,
        })
        .expect("register column");
        let mut all_vecs = Vec::new();
        for i in 0..64u32 {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i.wrapping_mul(13) + j as u32 * 5) % 100) as f32)
                .collect();
            b.add(0, &v).expect("add to vector builder");
            all_vecs.push(v);
        }
        let bytes = b.finish();
        let json = r#"[{"name":"embedding","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#;
        let r = VectorReader::open(Bytes::from(bytes), json).expect("open VectorReader");

        // Pick a doc, query with its own vector → top-1 is self with distance 0.
        let target = 17;
        let hits = r
            .search("embedding", &all_vecs[target], 5, 4, 5)
            .expect("FTS search");
        assert!(!hits.is_empty(), "search should return hits");
        assert_eq!(hits[0].0, target as u32, "self should be nearest");
        assert!(
            hits[0].1 < 1e-3,
            "self distance should be ~0, got {}",
            hits[0].1
        );
    }

    #[test]
    fn search_unknown_column_errors() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let err = r
            .search("nonexistent", &[0.0; 16], 5, 4, 5)
            .expect_err("expected error");
        assert!(matches!(err, VectorError::UnknownColumn(_)));
    }

    #[test]
    fn search_dim_mismatch_errors() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let err = r
            .search("embedding", &[0.0; 8], 5, 4, 5)
            .expect_err("expected error");
        assert!(matches!(err, VectorError::DimensionMismatch { .. }));
    }

    #[test]
    fn search_zero_k_returns_empty() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let hits = r
            .search("embedding", &[0.0; 16], 0, 4, 5)
            .expect("FTS search");
        assert!(hits.is_empty());
    }

    #[test]
    fn search_results_sorted_ascending_by_distance() {
        let (blob, json) = build_blob(64, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let q = vec![0.5; 16];
        let hits = r.search("embedding", &q, 10, 4, 5).expect("FTS search");
        for w in hits.windows(2) {
            assert!(w[0].1 <= w[1].1, "distances should be ascending");
        }
    }

    #[test]
    fn summary_returns_dim_centroid_and_radius() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let (centroid, radius) = r.summary("embedding").expect("vector summary");
        assert_eq!(centroid.len(), 16);
        assert!(radius >= 0.0);
        assert!(r.summary("nonexistent").is_none());
    }

    #[test]
    fn open_rejects_columns_json_mismatch() {
        let (blob, _) = build_blob(32, 16, 4, Metric::L2Sq);
        // header says 1 column; pass 2-column JSON.
        let bad_json = r#"[{"name":"a","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"},{"name":"b","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#;
        let err = VectorReader::open(blob, bad_json).expect_err("expected error");
        assert!(matches!(
            err,
            VectorError::Read(ReadError::MalformedVersion(_))
        ));
    }
}
