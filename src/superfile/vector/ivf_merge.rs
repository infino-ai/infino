// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Byte-splice merge of Sq8+ε IVF subsections for compaction.
//!
//! Concatenates per-cluster blocks across inputs, remapping local doc ids,
//! and Sq8-transcodes rerank rows only when a source cluster's quantizer
//! differs from the destination — no fp32 corpus buffer and no re-kmeans.

use bytemuck::cast_slice;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::superfile::{
    BuildError,
    format::{
        checksum::crc32c,
        CRC_BYTES,
        vec::{sub_hdr, DOC_ID_BYTES, STABLE_ID_BYTES},
    },
    vector::{
        builder::{
            IvfSubsectionLayout, alloc_ivf_subsection_with_header, centroid_storage_order,
            write_ivf_cluster_blocks,
        },
        cell_posting::{
            EncodedCellRow, materialize_sq8_residual_row_into_cluster_quant,
            sq8_quant_params_equal, sq8_residual_norm_sq,
        },
        centroid_block::{self, CentroidBlock},
        distance::Metric,
        quant::BitQuantizer,
        reader::{VectorReader, read_cluster_entry, read_cluster_radius},
        rerank_codec::RerankCodec,
    },
};

/// One input superfile column for byte-splice merge.
pub(crate) struct Sq8IvfMergeInput {
    pub sub: Vec<u8>,
    pub dim: usize,
    pub n_cent: usize,
    pub n_docs: u32,
    pub metric: Metric,
    pub doc_id_offset: u32,
    pub cluster_idx_off: usize,
    /// Byte offset (within `sub`) of the Sq8+ε centroid block's first byte.
    pub centroid_block_off: usize,
    pub per_cluster_blocks_off: usize,
    pub code_bytes: usize,
    pub per_vec_bytes: usize,
    pub stride: usize,
    pub scale: Vec<f32>,
    pub offset: Vec<f32>,
    pub summary_radius_x100: u32,
    /// Inline stable-`_id`s indexed by local doc id when the source has the region.
    pub stable_ids: Option<Vec<i128>>,
}

/// Output of a byte-splice merge, ready for [`super::builder::VectorBuilder::set_prebuilt_subsection`].
pub(crate) struct MergedIvfSubsection {
    pub bytes: Vec<u8>,
    pub n_cent: usize,
    pub n_docs: u32,
    pub summary_offset_in_sub: usize,
    pub codec_meta_offset_in_sub: usize,
    pub codec_meta_size: usize,
}

/// `(doc_off, count)` for cluster `c` in one input, decoded via the shared
/// reader-side [`read_cluster_entry`] (input shape adapted: full subsection
/// buffer + cluster-index offset → the `n_cent × 8` index slice, widened to
/// `usize` for the byte-offset arithmetic here).
fn cluster_entry(sub: &[u8], cluster_idx_off: usize, c: usize) -> (usize, usize) {
    let (doc_off, count) = read_cluster_entry(&sub[cluster_idx_off..], c);
    (doc_off as usize, count as usize)
}

/// Cluster `c`'s covering radius in one input (sibling of [`cluster_entry`]).
/// A spliced output cluster is a verbatim source cluster, so its radius is the
/// source's — copied here so the merged cell stays radius-aware.
fn cluster_radius(sub: &[u8], cluster_idx_off: usize, c: usize) -> f32 {
    read_cluster_radius(&sub[cluster_idx_off..], c)
}

/// Merge Sq8+ε IVF subsections by splicing per-cluster blocks.
pub(crate) fn merge_sq8_ivf_subsections(
    inputs: &[(&VectorReader, &str, u32)],
) -> Result<MergedIvfSubsection, BuildError> {
    if inputs.is_empty() {
        return Err(BuildError::VectorSchemaMismatch(
            "merge requires at least one IVF input".into(),
        ));
    }
    let parsed: Vec<Sq8IvfMergeInput> = inputs
        .iter()
        .map(|(r, col, off)| r.sq8_ivf_merge_input(col, *off))
        .collect::<Result<_, _>>()?;

    let dim = parsed[0].dim;
    let n_cent = parsed[0].n_cent;
    let metric = parsed[0].metric;
    for inp in &parsed[1..] {
        if inp.dim != dim || inp.n_cent != n_cent || inp.metric != metric {
            return Err(BuildError::VectorSchemaMismatch(
                "Sq8 IVF merge inputs must share dim, n_cent, and metric".into(),
            ));
        }
    }

    let n_docs: u32 = parsed.iter().map(|p| p.n_docs).sum();
    let codec = RerankCodec::Sq8Residual;
    let quant = BitQuantizer::new(dim);
    let code_bytes = quant.code_bytes();
    let per_vec_bytes = codec.per_vector_bytes(dim);
    let store_norm = matches!(metric, Metric::L2Sq | Metric::Cosine);

    // Per-input centroid block views. Centroids are Sq8+ε on disk; decode
    // each cluster centroid to fp32 here (the merge boundary, not the query
    // hot path) to count-weight-average across inputs, then re-quantize the
    // result through the output centroid block below.
    let centroid_blocks: Vec<CentroidBlock> = parsed
        .iter()
        .map(|inp| {
            CentroidBlock::new(
                &inp.sub[inp.centroid_block_off..],
                inp.metric,
                inp.dim,
                inp.n_cent,
            )
        })
        .collect();

    let mut out_centroids = vec![0.0f32; n_cent * dim];
    for c in 0..n_cent {
        let mut acc = vec![0.0f64; dim];
        let mut total = 0u64;
        for (inp, block) in parsed.iter().zip(&centroid_blocks) {
            let (_, count) = cluster_entry(&inp.sub, inp.cluster_idx_off, c);
            if count == 0 {
                continue;
            }
            total += count as u64;
            let cv = block.cluster_components(c);
            for (acc_d, &v) in acc.iter_mut().zip(&cv) {
                *acc_d += v as f64 * count as f64;
            }
        }
        if total > 0 {
            let inv = 1.0 / (total as f64);
            for d in 0..dim {
                out_centroids[c * dim + d] = (acc[d] * inv) as f32;
            }
        }
    }

    let mut summary_centroid = vec![0.0f32; dim];
    if n_cent > 0 {
        let mut acc = vec![0.0f64; dim];
        for c in 0..n_cent {
            let cv = &out_centroids[c * dim..(c + 1) * dim];
            for (a, &x) in acc.iter_mut().zip(cv) {
                *a += x as f64;
            }
        }
        let inv = 1.0 / (n_cent as f64);
        for (s, a) in summary_centroid.iter_mut().zip(&acc) {
            *s = (*a * inv) as f32;
        }
    }

    let summary_radius_x100 = parsed
        .iter()
        .map(|p| p.summary_radius_x100)
        .max()
        .unwrap_or(0);

    let mut dst_scale = vec![1.0f32; n_cent * dim];
    let mut dst_offset = vec![0.0f32; n_cent * dim];
    for c in 0..n_cent {
        for inp in &parsed {
            let (_, count) = cluster_entry(&inp.sub, inp.cluster_idx_off, c);
            if count == 0 {
                continue;
            }
            let off = c * dim;
            dst_scale[off..off + dim].copy_from_slice(&inp.scale[off..off + dim]);
            dst_offset[off..off + dim].copy_from_slice(&inp.offset[off..off + dim]);
            break;
        }
    }

    let codec_meta_size = codec.codec_meta_bytes(dim, n_docs as usize, n_cent, metric);
    let cluster_stride = code_bytes + DOC_ID_BYTES + per_vec_bytes;
    let produce_region = parsed.iter().all(|p| p.stable_ids.is_some());
    let stable_ids_region_bytes = if produce_region {
        n_docs as usize * STABLE_ID_BYTES
    } else {
        0
    };
    let layout = IvfSubsectionLayout::compute(
        dim,
        n_cent,
        n_docs as usize,
        cluster_stride,
        codec_meta_size,
        stable_ids_region_bytes,
        metric,
    );

    let mut bytes = alloc_ivf_subsection_with_header(
        &layout,
        codec_meta_size,
        summary_radius_x100,
        metric,
        dim,
        n_cent,
        &summary_centroid,
        &out_centroids,
    );

    let sq8_scale_block_off = layout.codec_meta_off;
    let sq8_offset_block_off = sq8_scale_block_off + n_cent * dim * 4;
    let sq8_norms_block_off = if store_norm {
        Some(sq8_offset_block_off + n_cent * dim * 4)
    } else {
        None
    };

    for c in 0..n_cent {
        let sc_off = sq8_scale_block_off + c * dim * 4;
        bytes[sc_off..sc_off + dim * 4]
            .copy_from_slice(cast_slice(&dst_scale[c * dim..c * dim + dim]));
        let oc_off = sq8_offset_block_off + c * dim * 4;
        bytes[oc_off..oc_off + dim * 4]
            .copy_from_slice(cast_slice(&dst_offset[c * dim..c * dim + dim]));
    }

    let cluster_order = centroid_storage_order(&out_centroids, n_cent, dim);
    // Merged per-cluster row counts (sum across inputs), so the shared
    // cluster-block writer owns the index + cursor + offset math.
    let merged_counts: Vec<u32> = (0..n_cent)
        .map(|c| {
            parsed
                .iter()
                .map(|inp| cluster_entry(&inp.sub, inp.cluster_idx_off, c).1 as u32)
                .sum()
        })
        .collect();
    // Each merged cluster c pools the inputs' cluster c (shared centroid), so its
    // covering radius is the max of the sources' radii.
    let merged_radii: Vec<f32> = (0..n_cent)
        .map(|c| {
            parsed
                .iter()
                .map(|inp| cluster_radius(&inp.sub, inp.cluster_idx_off, c))
                .fold(0.0f32, f32::max)
        })
        .collect();
    let id_bytes = DOC_ID_BYTES;
    let mut row_buf = vec![0u8; dim * 2];
    let stable_ids_region_off = layout.stable_ids_off;

    write_ivf_cluster_blocks(
        &mut bytes,
        &layout,
        &cluster_order,
        &merged_counts,
        &merged_radii,
        code_bytes,
        per_vec_bytes,
        |bytes, centroid_id, blk| {
            let scale_c = &dst_scale[centroid_id * dim..centroid_id * dim + dim];
            let offset_c = &dst_offset[centroid_id * dim..centroid_id * dim + dim];
            let mut out_i = 0usize;

            for inp in &parsed {
                let (doc_off, count) = cluster_entry(&inp.sub, inp.cluster_idx_off, centroid_id);
                if count == 0 {
                    continue;
                }
                let src_scale = &inp.scale[centroid_id * dim..centroid_id * dim + dim];
                let src_offset = &inp.offset[centroid_id * dim..centroid_id * dim + dim];
                let block = inp.per_cluster_blocks_off + doc_off * inp.stride;
                let doc_ids_at = block + count * inp.code_bytes;
                let full_at = block + count * (inp.code_bytes + id_bytes);

                for i in 0..count {
                    bytes[blk.codes_base + out_i * code_bytes
                        ..blk.codes_base + (out_i + 1) * code_bytes]
                        .copy_from_slice(
                            &inp.sub[block + i * inp.code_bytes..block + (i + 1) * inp.code_bytes],
                        );

                    let idb = doc_ids_at + i * id_bytes;
                    let src_local = u32::from_le_bytes([
                        inp.sub[idb],
                        inp.sub[idb + 1],
                        inp.sub[idb + 2],
                        inp.sub[idb + 3],
                    ]);
                    let local_id = src_local + inp.doc_id_offset;
                    let id_off = blk.ids_base + out_i * id_bytes;
                    bytes[id_off..id_off + id_bytes].copy_from_slice(&local_id.to_le_bytes());

                    if let Some(region_off) = stable_ids_region_off {
                        let sid =
                            inp.stable_ids.as_ref().expect("produce_region")[src_local as usize];
                        let p = region_off + (local_id as usize) * STABLE_ID_BYTES;
                        bytes[p..p + STABLE_ID_BYTES].copy_from_slice(&sid.to_le_bytes());
                    }

                    let rowb = full_at + i * inp.per_vec_bytes;
                    let full_off = blk.rerank_base + out_i * per_vec_bytes;
                    let norm_sq =
                        if sq8_quant_params_equal(src_scale, src_offset, scale_c, offset_c) {
                            bytes[full_off..full_off + dim * 2]
                                .copy_from_slice(&inp.sub[rowb..rowb + dim * 2]);
                            store_norm.then(|| {
                                sq8_residual_norm_sq(
                                    dim,
                                    scale_c,
                                    offset_c,
                                    &inp.sub[rowb..rowb + dim],
                                    &inp.sub[rowb + dim..rowb + dim + dim],
                                )
                            })
                        } else {
                            let encoded = EncodedCellRow {
                                stable_id: 0,
                                scale: Arc::from(src_scale.to_vec()),
                                offset: Arc::from(src_offset.to_vec()),
                                codes: inp.sub[rowb..rowb + dim].to_vec(),
                                residuals: inp.sub[rowb + dim..rowb + dim + dim].to_vec(),
                                norm_sq: None,
                            };
                            let n = materialize_sq8_residual_row_into_cluster_quant(
                                &encoded,
                                scale_c,
                                offset_c,
                                dim,
                                &mut row_buf,
                                store_norm,
                            );
                            bytes[full_off..full_off + dim * 2].copy_from_slice(&row_buf);
                            n
                        };

                    if let (Some(norms_off), Some(n_sq)) = (sq8_norms_block_off, norm_sq) {
                        let n_off = norms_off + (blk.first_row + out_i) * 4;
                        bytes[n_off..n_off + 4].copy_from_slice(&n_sq.to_le_bytes());
                    }
                    out_i += 1;
                }
            }
            debug_assert_eq!(out_i, blk.count);
            Ok(())
        },
    )?;

    let crc = crc32c(&bytes);
    bytes.extend_from_slice(&crc.to_le_bytes());

    Ok(MergedIvfSubsection {
        bytes,
        n_cent,
        n_docs,
        summary_offset_in_sub: layout.summary_off,
        codec_meta_offset_in_sub: if codec_meta_size == 0 {
            0
        } else {
            layout.codec_meta_off
        },
        codec_meta_size,
    })
}

/// A byte-spliced subsection for one routed cell, plus the stable `_id`s in the
/// exact `local_doc_id` order the subsection was written in (`stable_ids[local]`).
/// The caller pairs these with a scalar `_id` batch in the same order so the
/// superfile's id pages line up with the IVF subsection's local doc ids.
pub(crate) struct RoutedCellSubsection {
    pub subsection: MergedIvfSubsection,
    pub stable_ids: Vec<i128>,
    /// Max member distance to this cell's centroid over the routed rows (the
    /// recomputed shard radius `route` reported). Travels with the subsection so
    /// the routing pass can stay a pure parallel map — no shared-state side
    /// channel. `0.0` when no row contributed a positive radius.
    pub shard_radius: f32,
    /// Per-output-cluster covering radius (max member distance to the cluster's
    /// own centroid), aligned with the cell's clusters by ordinal. Populated
    /// onto `VectorSummary.clusters.radii` so the within-cell admission can be
    /// radius-aware. Empty when no rows contributed.
    pub cluster_radii: Vec<f32>,
}

/// One contributing source row, identified by `(input, source_cluster, row)`.
struct SourceRowRef {
    input: usize,
    cluster: usize,
    row: usize,
    src_local: u32,
}

/// Routing result for one `(input, source_cluster)` unit: its rows bucketed by
/// destination cell, and the per-cell max member radius. Produced independently
/// per unit by the parallel routing pass, then merged collision-free (each
/// `pair` is unique across units).
struct PairRouting {
    pair: (usize, usize),
    per_cell: HashMap<u32, Vec<SourceRowRef>>,
    radii: HashMap<u32, f32>,
}

fn read_u64_le(sub: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(
        sub[off..off + 8]
            .try_into()
            .expect("8-byte u64 slice"),
    )
}

/// Recover vector dimension from the per-cluster blocks region and doc count.
fn infer_ivf_subsection_dim(sub: &[u8], per_cluster_blocks_off: usize, n_docs: u32) -> Option<usize> {
    if n_docs == 0 || sub.len() < per_cluster_blocks_off + CRC_BYTES {
        return None;
    }
    let cluster_region = sub.len() - per_cluster_blocks_off - CRC_BYTES;
    let stride = cluster_region / (n_docs as usize);
    if stride <= DOC_ID_BYTES {
        return None;
    }
    // stride = dim/8 + DOC_ID_BYTES + 2*dim
    let num = stride.checked_mul(8)?.saturating_sub(DOC_ID_BYTES * 8);
    if num % 17 != 0 {
        return None;
    }
    let dim = num / 17;
    (dim >= 16 && dim <= 4096).then_some(dim)
}

fn infer_ivf_subsection_metric(
    summary_off: usize,
    cluster_idx_off: usize,
    dim: usize,
    n_cent: usize,
) -> Metric {
    for metric in [Metric::L2Sq, Metric::Cosine, Metric::NegDot] {
        if summary_off + centroid_block::centroid_block_bytes(dim, n_cent, metric) == cluster_idx_off
        {
            return metric;
        }
    }
    Metric::L2Sq
}

/// Parse a byte-spliced cell subsection back into a merge input (for incremental
/// drain concat — one superfile at a time).
fn sq8_ivf_merge_input_from_merged_subsection(
    merged: &MergedIvfSubsection,
    stable_ids: Vec<i128>,
) -> Result<Sq8IvfMergeInput, BuildError> {
    let sub = &merged.bytes;
    let n_cent = merged.n_cent;
    let n_docs = merged.n_docs;
    let cluster_idx_off = read_u64_le(sub, sub_hdr::CLUSTER_IDX_OFF_OFF) as usize;
    let per_cluster_blocks_off = read_u64_le(sub, sub_hdr::PER_CLUSTER_BLOCKS_OFF_OFF) as usize;
    let summary_off = read_u64_le(sub, sub_hdr::SUMMARY_OFF_OFF) as usize;
    let summary_radius_x100 = u32::from_le_bytes(
        sub[sub_hdr::SUMMARY_RADIUS_X100_OFF..sub_hdr::SUMMARY_RADIUS_X100_OFF + 4]
            .try_into()
            .expect("summary radius"),
    );
    let codec_meta_off = merged.codec_meta_offset_in_sub;
    let dim = infer_ivf_subsection_dim(sub, per_cluster_blocks_off, n_docs).ok_or_else(|| {
        BuildError::VectorSchemaMismatch("cannot infer IVF dim from merged subsection".into())
    })?;
    let metric = infer_ivf_subsection_metric(summary_off, cluster_idx_off, dim, n_cent);
    let scale_end = codec_meta_off + n_cent * dim * 4;
    let offset_end = scale_end + n_cent * dim * 4;
    let scale = cast_slice(&sub[codec_meta_off..scale_end]).to_vec();
    let offset = cast_slice(&sub[scale_end..offset_end]).to_vec();
    let codec = RerankCodec::Sq8Residual;
    let quant = BitQuantizer::new(dim);
    let code_bytes = quant.code_bytes();
    let per_vec_bytes = codec.per_vector_bytes(dim);
    let stride = code_bytes + DOC_ID_BYTES + per_vec_bytes;
    Ok(Sq8IvfMergeInput {
        sub: sub.to_vec(),
        dim,
        n_cent,
        n_docs,
        metric,
        doc_id_offset: 0,
        cluster_idx_off,
        centroid_block_off: summary_off,
        per_cluster_blocks_off,
        code_bytes,
        per_vec_bytes,
        stride,
        scale,
        offset,
        summary_radius_x100,
        stable_ids: Some(stable_ids),
    })
}

/// Every non-empty `(input, source_cluster)` in `parsed`, with rows in cluster order.
fn pairs_all_clusters(parsed: &[Sq8IvfMergeInput]) -> HashMap<(usize, usize), Vec<SourceRowRef>> {
    let id_bytes = DOC_ID_BYTES;
    let mut pairs = HashMap::new();
    for (ii, inp) in parsed.iter().enumerate() {
        for c in 0..inp.n_cent {
            let (doc_off, count) = cluster_entry(&inp.sub, inp.cluster_idx_off, c);
            if count == 0 {
                continue;
            }
            let block = inp.per_cluster_blocks_off + doc_off * inp.stride;
            let doc_ids_at = block + count * inp.code_bytes;
            let rows: Vec<SourceRowRef> = (0..count)
                .map(|i| {
                    let idb = doc_ids_at + i * id_bytes;
                    let src_local = u32::from_le_bytes([
                        inp.sub[idb],
                        inp.sub[idb + 1],
                        inp.sub[idb + 2],
                        inp.sub[idb + 3],
                    ]);
                    SourceRowRef {
                        input: ii,
                        cluster: c,
                        row: i,
                        src_local,
                    }
                })
                .collect();
            pairs.insert((ii, c), rows);
        }
    }
    pairs
}

/// Build one routed cell subsection from regrouped `(input, source_cluster)` buckets.
fn build_routed_cell_subsection_from_pairs(
    pairs: HashMap<(usize, usize), Vec<SourceRowRef>>,
    parsed: &[Sq8IvfMergeInput],
    stable_ids_per_input: &[Vec<i128>],
    shard_radius: f32,
) -> Result<RoutedCellSubsection, BuildError> {
    let dim = parsed[0].dim;
    let metric = parsed[0].metric;
    for inp in &parsed[1..] {
        if inp.dim != dim || inp.metric != metric {
            return Err(BuildError::VectorSchemaMismatch(
                "Sq8 IVF encode inputs must share dim and metric".into(),
            ));
        }
    }

    let codec = RerankCodec::Sq8Residual;
    let quant = BitQuantizer::new(dim);
    let code_bytes = quant.code_bytes();
    let per_vec_bytes = codec.per_vector_bytes(dim);
    let store_norm = matches!(metric, Metric::L2Sq | Metric::Cosine);
    let id_bytes = DOC_ID_BYTES;
    let produce_region = true;

    let centroid_blocks: Vec<CentroidBlock> = parsed
        .iter()
        .map(|inp| {
            CentroidBlock::new(
                &inp.sub[inp.centroid_block_off..],
                inp.metric,
                inp.dim,
                inp.n_cent,
            )
        })
        .collect();

    let mut pair_keys: Vec<(usize, usize)> = pairs.keys().copied().collect();
    pair_keys.sort_unstable();
    let out_n_cent = pair_keys.len();
    let n_docs: u32 = pairs.values().map(|v| v.len() as u32).sum();

    let mut out_scale = vec![0.0f32; out_n_cent * dim];
    let mut out_offset = vec![0.0f32; out_n_cent * dim];
    let mut out_centroids = vec![0.0f32; out_n_cent * dim];
    for (k, &(ii, c)) in pair_keys.iter().enumerate() {
        out_scale[k * dim..k * dim + dim].copy_from_slice(&parsed[ii].scale[c * dim..c * dim + dim]);
        out_offset[k * dim..k * dim + dim]
            .copy_from_slice(&parsed[ii].offset[c * dim..c * dim + dim]);
        let cv = centroid_blocks[ii].cluster_components(c);
        out_centroids[k * dim..k * dim + dim].copy_from_slice(&cv);
    }

    let mut summary_centroid = vec![0.0f32; dim];
    if out_n_cent > 0 {
        let mut acc = vec![0.0f64; dim];
        for c in 0..out_n_cent {
            let cv = &out_centroids[c * dim..(c + 1) * dim];
            for (a, &x) in acc.iter_mut().zip(cv) {
                *a += x as f64;
            }
        }
        let inv = 1.0 / (out_n_cent as f64);
        for (s, a) in summary_centroid.iter_mut().zip(&acc) {
            *s = (*a * inv) as f32;
        }
    }

    let summary_radius_x100 = pair_keys
        .iter()
        .map(|&(ii, _)| parsed[ii].summary_radius_x100)
        .max()
        .unwrap_or(0);

    let codec_meta_size = codec.codec_meta_bytes(dim, n_docs as usize, out_n_cent, metric);
    let cluster_stride = code_bytes + id_bytes + per_vec_bytes;
    let stable_ids_region_bytes = if produce_region {
        n_docs as usize * STABLE_ID_BYTES
    } else {
        0
    };
    let layout = IvfSubsectionLayout::compute(
        dim,
        out_n_cent,
        n_docs as usize,
        cluster_stride,
        codec_meta_size,
        stable_ids_region_bytes,
        metric,
    );

    let mut bytes = alloc_ivf_subsection_with_header(
        &layout,
        codec_meta_size,
        summary_radius_x100,
        metric,
        dim,
        out_n_cent,
        &summary_centroid,
        &out_centroids,
    );

    let sq8_scale_block_off = layout.codec_meta_off;
    let sq8_offset_block_off = sq8_scale_block_off + out_n_cent * dim * 4;
    let sq8_norms_block_off = if store_norm {
        Some(sq8_offset_block_off + out_n_cent * dim * 4)
    } else {
        None
    };
    for c in 0..out_n_cent {
        let sc_off = sq8_scale_block_off + c * dim * 4;
        bytes[sc_off..sc_off + dim * 4].copy_from_slice(cast_slice(&out_scale[c * dim..c * dim + dim]));
        let oc_off = sq8_offset_block_off + c * dim * 4;
        bytes[oc_off..oc_off + dim * 4].copy_from_slice(cast_slice(&out_offset[c * dim..c * dim + dim]));
    }

    let cluster_order = centroid_storage_order(&out_centroids, out_n_cent, dim);
    let merged_counts: Vec<u32> = pair_keys.iter().map(|k| pairs[k].len() as u32).collect();
    let out_cluster_radii: Vec<f32> = pair_keys
        .iter()
        .map(|&(ii, c)| cluster_radius(&parsed[ii].sub, parsed[ii].cluster_idx_off, c))
        .collect();
    let stable_ids_region_off = layout.stable_ids_off;
    let mut stable_ids_by_local = vec![0i128; n_docs as usize];

    write_ivf_cluster_blocks(
        &mut bytes,
        &layout,
        &cluster_order,
        &merged_counts,
        &out_cluster_radii,
        code_bytes,
        per_vec_bytes,
        |bytes, out_cluster, blk| {
            let (ii, c) = pair_keys[out_cluster];
            let inp = &parsed[ii];
            let scale_c = &out_scale[out_cluster * dim..out_cluster * dim + dim];
            let offset_c = &out_offset[out_cluster * dim..out_cluster * dim + dim];
            debug_assert!(sq8_quant_params_equal(
                &inp.scale[c * dim..c * dim + dim],
                &inp.offset[c * dim..c * dim + dim],
                scale_c,
                offset_c,
            ));

            let (doc_off, src_count) = cluster_entry(&inp.sub, inp.cluster_idx_off, c);
            let block = inp.per_cluster_blocks_off + doc_off * inp.stride;
            let full_at = block + src_count * (inp.code_bytes + id_bytes);

            let rows = &pairs[&(ii, c)];
            for (out_i, row_ref) in rows.iter().enumerate() {
                let i = row_ref.row;
                bytes[blk.codes_base + out_i * code_bytes..blk.codes_base + (out_i + 1) * code_bytes]
                    .copy_from_slice(
                        &inp.sub[block + i * inp.code_bytes..block + (i + 1) * inp.code_bytes],
                    );

                let local_id = (blk.first_row + out_i) as u32;
                let id_off = blk.ids_base + out_i * id_bytes;
                bytes[id_off..id_off + id_bytes].copy_from_slice(&local_id.to_le_bytes());

                let stable_id = stable_ids_per_input[ii][row_ref.src_local as usize];
                stable_ids_by_local[local_id as usize] = stable_id;
                if let Some(region_off) = stable_ids_region_off {
                    let p = region_off + (local_id as usize) * STABLE_ID_BYTES;
                    bytes[p..p + STABLE_ID_BYTES].copy_from_slice(&stable_id.to_le_bytes());
                }

                let rowb = full_at + i * inp.per_vec_bytes;
                let full_off = blk.rerank_base + out_i * per_vec_bytes;
                bytes[full_off..full_off + dim * 2]
                    .copy_from_slice(&inp.sub[rowb..rowb + dim * 2]);

                if let Some(norms_off) = sq8_norms_block_off {
                    let n_sq = sq8_residual_norm_sq(
                        dim,
                        scale_c,
                        offset_c,
                        &inp.sub[rowb..rowb + dim],
                        &inp.sub[rowb + dim..rowb + dim + dim],
                    );
                    let n_off = norms_off + (blk.first_row + out_i) * 4;
                    bytes[n_off..n_off + 4].copy_from_slice(&n_sq.to_le_bytes());
                }
                debug_assert_eq!(row_ref.input, ii);
                debug_assert_eq!(row_ref.cluster, c);
            }
            Ok(())
        },
    )?;

    let crc = crc32c(&bytes);
    bytes.extend_from_slice(&crc.to_le_bytes());

    Ok(RoutedCellSubsection {
        subsection: MergedIvfSubsection {
            bytes,
            n_cent: out_n_cent,
            n_docs,
            summary_offset_in_sub: layout.summary_off,
            codec_meta_offset_in_sub: if codec_meta_size == 0 {
                0
            } else {
                layout.codec_meta_off
            },
            codec_meta_size,
        },
        stable_ids: stable_ids_by_local,
        shard_radius,
        cluster_radii: out_cluster_radii,
    })
}

/// Append `add` onto `acc` in place (incremental drain: one incoming superfile at a time).
pub(crate) fn concat_routed_cell_subsections_in_place(
    acc: &mut RoutedCellSubsection,
    add: RoutedCellSubsection,
) -> Result<(), BuildError> {
    let left_ids = std::mem::take(&mut acc.stable_ids);
    let left_shard = acc.shard_radius;
    let left_sub = std::mem::replace(
        &mut acc.subsection,
        MergedIvfSubsection {
            bytes: Vec::new(),
            n_cent: 0,
            n_docs: 0,
            summary_offset_in_sub: 0,
            codec_meta_offset_in_sub: 0,
            codec_meta_size: 0,
        },
    );
    let left_inp = sq8_ivf_merge_input_from_merged_subsection(&left_sub, left_ids)?;
    let right_inp = sq8_ivf_merge_input_from_merged_subsection(&add.subsection, add.stable_ids)?;
    let parsed = [left_inp, right_inp];
    let stable_ids_per_input: Vec<Vec<i128>> = parsed
        .iter()
        .map(|p| p.stable_ids.clone().unwrap_or_default())
        .collect();
    let pairs = pairs_all_clusters(&parsed);
    *acc = build_routed_cell_subsection_from_pairs(
        pairs,
        &parsed,
        &stable_ids_per_input,
        left_shard.max(add.shard_radius),
    )?;
    Ok(())
}

/// Route every Sq8+ε row across `inputs` to a cell (via `route`) and emit one
/// byte-spliced IVF subsection per touched cell — **without ever re-quantizing**.
///
/// Each distinct `(input, source_cluster)` that contributes at least one row to
/// a cell becomes its own output cluster in that cell's subsection, carrying
/// that source cluster's quantizer (`scale`/`offset`) and centroid verbatim.
/// Because the destination quantizer is, by construction, always equal to the
/// source quantizer, the verbatim code+rerank copy is the only path taken; the
/// re-quant branch from [`merge_sq8_ivf_subsections`] is never reached here (a
/// `debug_assert` proves it). All `code_bytes` of the 1-bit RaBitQ code and all
/// `2 * dim` rerank bytes (Sq8 codes ‖ ε residuals) are copied byte-for-byte
/// from the source — recall is preserved exactly.
///
/// Mirrors [`merge_sq8_ivf_subsections`]'s header / centroid-block / cluster-block
/// / CRC writing; only the cluster grouping (per source pair, not pooled by
/// shared cluster index) and the routing differ.
pub(crate) fn route_and_splice_ivf_subsections<F>(
    inputs: &[(&VectorReader, &str)],
    // Per-input stable `_id`s in local-doc-id order, resolved by the caller.
    // Incoming superfiles are streaming-built and carry NO inline `_id` region,
    // so the ids must come from the scalar `_id` column / span arithmetic — the
    // splice cannot read them from the subsection bytes.
    stable_ids_per_input: &[Vec<i128>],
    route: F,
) -> Result<HashMap<u32, RoutedCellSubsection>, BuildError>
where
    // Returns the SET of destination cells (SPANN closure replication) with each
    // row's distance to that cell's centroid. `Sync` for the rayon `par_iter`.
    F: Fn(&EncodedCellRow) -> Vec<(u32, f32)> + Sync,
{
    if inputs.is_empty() {
        return Err(BuildError::VectorSchemaMismatch(
            "route_and_splice_ivf_subsections requires at least one IVF input".into(),
        ));
    }
    if stable_ids_per_input.len() != inputs.len() {
        return Err(BuildError::VectorSchemaMismatch(
            "route_and_splice_ivf_subsections: stable_ids_per_input must match inputs len".into(),
        ));
    }
    let parsed: Vec<Sq8IvfMergeInput> = inputs
        .iter()
        .map(|(r, col)| r.sq8_ivf_merge_input(col, 0))
        .collect::<Result<_, _>>()?;

    let dim = parsed[0].dim;
    let metric = parsed[0].metric;
    for inp in &parsed[1..] {
        if inp.dim != dim || inp.metric != metric {
            return Err(BuildError::VectorSchemaMismatch(
                "Sq8 IVF encode inputs must share dim and metric".into(),
            ));
        }
    }

    let store_norm = matches!(metric, Metric::L2Sq | Metric::Cosine);
    let id_bytes = DOC_ID_BYTES;

    // Routing pass: reconstruct each row as an `EncodedCellRow` (codes/residuals
    // straight from the rerank bytes, scale/offset from the source cluster's
    // quantizer — identical to the merge else-branch), route it, and bucket the
    // source reference under `cell -> (input, source_cluster) -> [rows]`. The
    // inner key preserving `(input, source_cluster)` is what guarantees each
    // output cluster carries a single, intact source quantizer.
    //
    // The per-row work (nearest-cell over the global centroids) is the drain's
    // dominant cost, so it runs as a rayon `par_iter` over `(input, cluster)`
    // units. This uses the *ambient* rayon pool — the drain invokes this under
    // `writer_pool.install(...)`, whose pool is sized to half the cores — so no
    // new pool is created here. Each unit is a `(input, source_cluster)` pair
    // (always written as one output cluster), so units never share an output
    // bucket and the merge below is a collision-free, deterministic regroup.
    let work: Vec<(usize, usize)> = parsed
        .iter()
        .enumerate()
        .flat_map(|(ii, inp)| (0..inp.n_cent).map(move |c| (ii, c)))
        .collect();

    let routed_pairs: Vec<PairRouting> = work
        .par_iter()
        .map(|&(ii, c)| {
            let inp = &parsed[ii];
            let mut per_cell: HashMap<u32, Vec<SourceRowRef>> = HashMap::new();
            let mut radii: HashMap<u32, f32> = HashMap::new();
            let (doc_off, count) = cluster_entry(&inp.sub, inp.cluster_idx_off, c);
            if count > 0 {
                let src_scale: Arc<[f32]> = Arc::from(inp.scale[c * dim..c * dim + dim].to_vec());
                let src_offset: Arc<[f32]> = Arc::from(inp.offset[c * dim..c * dim + dim].to_vec());
                let block = inp.per_cluster_blocks_off + doc_off * inp.stride;
                let doc_ids_at = block + count * inp.code_bytes;
                let full_at = block + count * (inp.code_bytes + id_bytes);
                for i in 0..count {
                    let idb = doc_ids_at + i * id_bytes;
                    let src_local = u32::from_le_bytes([
                        inp.sub[idb],
                        inp.sub[idb + 1],
                        inp.sub[idb + 2],
                        inp.sub[idb + 3],
                    ]);
                    let rowb = full_at + i * inp.per_vec_bytes;
                    let codes = inp.sub[rowb..rowb + dim].to_vec();
                    let residuals = inp.sub[rowb + dim..rowb + dim + dim].to_vec();
                    let stable_id = stable_ids_per_input[ii][src_local as usize];
                    let norm_sq = store_norm.then(|| {
                        sq8_residual_norm_sq(dim, &src_scale, &src_offset, &codes, &residuals)
                    });
                    let encoded = EncodedCellRow {
                        stable_id,
                        scale: Arc::clone(&src_scale),
                        offset: Arc::clone(&src_offset),
                        codes,
                        residuals,
                        norm_sq,
                    };
                    // SPANN closure replication: the row is written into every
                    // cell `route` returns (interior → 1, boundary → a few).
                    for (cell, radius) in route(&encoded) {
                        per_cell.entry(cell).or_default().push(SourceRowRef {
                            input: ii,
                            cluster: c,
                            row: i,
                            src_local,
                        });
                        let e = radii.entry(cell).or_insert(0.0);
                        if radius > *e {
                            *e = radius;
                        }
                    }
                }
            }
            PairRouting {
                pair: (ii, c),
                per_cell,
                radii,
            }
        })
        .collect();

    // Merge (sequential, cheap — no distance compute). Each `(ii, c)` pair was
    // routed by exactly one unit, so `insert` into the per-cell pair map never
    // collides; row order within a pair is preserved, and `pair_keys` are sorted
    // when each subsection is built, so the output layout is fully deterministic.
    let mut by_cell: HashMap<u32, HashMap<(usize, usize), Vec<SourceRowRef>>> = HashMap::new();
    let mut cell_radius: HashMap<u32, f32> = HashMap::new();
    for pr in routed_pairs {
        for (cell, rows) in pr.per_cell {
            by_cell.entry(cell).or_default().insert(pr.pair, rows);
        }
        for (cell, r) in pr.radii {
            let e = cell_radius.entry(cell).or_insert(0.0);
            if r > *e {
                *e = r;
            }
        }
    }

    let mut out: HashMap<u32, RoutedCellSubsection> = HashMap::with_capacity(by_cell.len());
    for (cell_id, pairs) in by_cell {
        let routed = build_routed_cell_subsection_from_pairs(
            pairs,
            &parsed,
            stable_ids_per_input,
            cell_radius.get(&cell_id).copied().unwrap_or(0.0),
        )?;
        out.insert(cell_id, routed);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Decimal128Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use std::collections::{HashMap, HashSet};

    use crate::superfile::{
        builder::{BuilderOptions, SuperfileBuilder, VectorConfig},
        reader::SuperfileReader,
        vector::cell_posting::{EncodedCellRow, MaterializedIvfRow, encoded_component_at},
    };

    const DIM: usize = 16;
    const N_CENT: usize = 2;
    const N_ROWS: usize = 12;
    const COLUMN: &str = "emb";
    const ID_BASE: i128 = 100;

    fn id_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "doc_id",
            DataType::Decimal128(38, 0),
            false,
        )]))
    }

    fn vector_config() -> VectorConfig {
        VectorConfig {
            column: COLUMN.into(),
            dim: DIM,
            n_cent: N_CENT,
            rot_seed: 0,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
        }
    }

    fn id_batch(ids: &[i128]) -> RecordBatch {
        let arr = Decimal128Array::from_iter_values(ids.iter().copied())
            .with_precision_and_scale(38, 0)
            .expect("valid decimal128");
        RecordBatch::try_new(id_schema(), vec![Arc::new(arr) as ArrayRef]).expect("build id batch")
    }

    /// Deterministic, separable fp32 corpus: rows split cleanly into two
    /// clusters so the IVF build produces non-empty clusters.
    fn corpus() -> (Vec<i128>, Vec<f32>) {
        let mut ids = Vec::with_capacity(N_ROWS);
        let mut vecs = vec![0.0f32; N_ROWS * DIM];
        for r in 0..N_ROWS {
            ids.push(ID_BASE + r as i128);
            let cluster = r % 2;
            for d in 0..DIM {
                let base = if cluster == 0 { 0.0 } else { 10.0 };
                vecs[r * DIM + d] = base + (r as f32) * 0.01 + (d as f32) * 0.1;
            }
        }
        (ids, vecs)
    }

    /// Build a streaming fp32 -> Sq8+ε IVF superfile — the exact shape of a real
    /// drain "incoming" superfile: the streaming build emits NO inline
    /// stable-`_id` region, so the read-back rows carry `stable_id == 0` and the
    /// splice must take its ids from the caller-supplied slice. The planted id
    /// for a row is `ids[local_doc_id]` (scalar `_id` batch is in add order).
    async fn build_streaming_incoming() -> (Arc<SuperfileReader>, Vec<i128>) {
        let (ids, vecs) = corpus();

        let opts = BuilderOptions::new(id_schema(), "doc_id", vec![], vec![vector_config()], None);
        let mut b = SuperfileBuilder::new(opts).expect("new builder");
        b.add_batch(&id_batch(&ids), &[vecs.as_slice()])
            .expect("add_batch");
        let bytes = b.finish().expect("finish incoming");
        let incoming = SuperfileReader::open(Bytes::from(bytes)).expect("open incoming");
        (Arc::new(incoming), ids)
    }

    /// Reconstruct one row's fp32 from its Sq8+ε codes/residuals via the shared
    /// decoder — identical inputs decode to identical vectors.
    fn decode(row: &EncodedCellRow) -> Vec<f32> {
        (0..DIM).map(|d| encoded_component_at(row, d)).collect()
    }

    fn l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Top-`k` stable ids by exact L2 over decoded rows.
    fn topk_ids(rows: &[(i128, Vec<f32>)], query: &[f32], k: usize) -> Vec<i128> {
        let mut scored: Vec<(f32, i128)> = rows.iter().map(|(id, v)| (l2(v, query), *id)).collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }

    #[tokio::test]
    async fn route_and_splice_ivf_subsections_splices_verbatim_and_preserves_recall() {
        let (incoming, ids) = build_streaming_incoming().await;

        // Streaming-built incoming carries no inline region, so the read-back
        // rows have `stable_id == 0`; the true id for a row is `ids[local_doc_id]`.
        // Key the source rows by that true id, holding the raw code+rerank bytes
        // for the verbatim-equality check.
        let src_rows = incoming
            .vec()
            .expect("vec reader")
            .materialized_index_rows_async(COLUMN)
            .await
            .expect("source rows");
        assert_eq!(src_rows.len(), N_ROWS);
        let mut src_by_id: HashMap<i128, &MaterializedIvfRow> = HashMap::new();
        for r in &src_rows {
            src_by_id.insert(ids[r.local_doc_id as usize], r);
        }

        // The caller resolves stable ids out-of-band (here: local-doc order, the
        // same order the scalar `_id` batch was added). This is the regression
        // guard: if the splice ever reads ids from the absent inline region
        // instead of this slice, every output id collapses to 0 and the
        // assertions below fail.
        let stable_ids_per_input: Vec<Vec<i128>> = vec![ids.clone()];

        // Route: even ids to cell 0, odd ids to cell 1 (deterministic split).
        let inputs: Vec<(&VectorReader, &str)> =
            vec![(incoming.vec().expect("vec reader"), COLUMN)];
        let route = |row: &EncodedCellRow| -> Vec<(u32, f32)> {
            if row.stable_id % 2 == 0 {
                vec![(0, 0.0)]
            } else {
                vec![(1, 0.0)]
            }
        };
        let routed = route_and_splice_ivf_subsections(&inputs, &stable_ids_per_input, route)
            .expect("route_and_splice_ivf_subsections");
        assert_eq!(routed.len(), 2, "both cells should be touched");

        // (a)+(b): every spliced row's code + rerank bytes are byte-identical to
        // the source, and the inline stable id maps to the right local id.
        let mut seen_ids: HashSet<i128> = HashSet::new();
        for (&cell, routed_cell) in routed.iter() {
            // Build a full superfile from the spliced subsection (exactly the
            // drain's publish shape) and read it back.
            let cell_ids = &routed_cell.stable_ids;
            let opts =
                BuilderOptions::new(id_schema(), "doc_id", vec![], vec![vector_config()], None);
            let mut cb = SuperfileBuilder::new(opts).expect("cell builder");
            cb.add_batch_ids_only(&id_batch(cell_ids)).expect("add ids");
            // Clone the subsection bytes into a fresh MergedIvfSubsection.
            let sub = MergedIvfSubsection {
                bytes: routed_cell.subsection.bytes.clone(),
                n_cent: routed_cell.subsection.n_cent,
                n_docs: routed_cell.subsection.n_docs,
                summary_offset_in_sub: routed_cell.subsection.summary_offset_in_sub,
                codec_meta_offset_in_sub: routed_cell.subsection.codec_meta_offset_in_sub,
                codec_meta_size: routed_cell.subsection.codec_meta_size,
            };
            cb.set_prebuilt_ivf_subsection(0, sub)
                .expect("set prebuilt");
            let cell_bytes = cb.finish().expect("finish cell");
            let cell_reader = SuperfileReader::open(Bytes::from(cell_bytes)).expect("open cell");
            let cell_rows = cell_reader
                .vec()
                .expect("vec reader")
                .materialized_index_rows_async(COLUMN)
                .await
                .expect("cell rows");

            for row in &cell_rows {
                // Parity matches the routed cell.
                assert_eq!(
                    (row.stable_id % 2 == 0) as u32,
                    1 - cell,
                    "row {} routed to wrong cell {cell}",
                    row.stable_id
                );
                let src = src_by_id.get(&row.stable_id).expect("source row by id");
                // (a) verbatim 1-bit RaBitQ code.
                assert_eq!(row.rabitq_code, src.rabitq_code, "rabitq code not verbatim");
                // (a) verbatim 2*dim rerank bytes (Sq8 codes ‖ ε residuals).
                assert_eq!(
                    row.encoded.codes, src.encoded.codes,
                    "sq8 codes not verbatim"
                );
                assert_eq!(
                    row.encoded.residuals, src.encoded.residuals,
                    "ε residuals not verbatim"
                );
                // (b) inline stable id resolves and lines up with the local id.
                assert_eq!(
                    cell_ids[row.local_doc_id as usize], row.stable_id,
                    "inline stable id / local id mismatch"
                );
                seen_ids.insert(row.stable_id);
            }
        }
        assert_eq!(
            seen_ids.len(),
            N_ROWS,
            "every source row must be routed once"
        );
        for id in &ids {
            assert!(seen_ids.contains(id), "id {id} dropped on splice");
        }

        // (c) brute-force recall@k over the spliced cells equals recall over the
        // original superfile. Bytes are identical, so decoded vectors match and
        // the nearest-neighbor sets are identical (recall == 1.0, no loss).
        let original: Vec<(i128, Vec<f32>)> = src_rows
            .iter()
            .map(|r| (ids[r.local_doc_id as usize], decode(&r.encoded)))
            .collect();
        let mut spliced: Vec<(i128, Vec<f32>)> = Vec::new();
        for routed_cell in routed.values() {
            // Re-derive decoded rows straight from the source map by stable id —
            // identical bytes guarantee identical decode.
            for &id in &routed_cell.stable_ids {
                let src = src_by_id.get(&id).expect("src by id");
                spliced.push((id, decode(&src.encoded)));
            }
        }

        let k = 10.min(N_ROWS);
        for (_, q) in &original {
            let a = topk_ids(&original, q, k);
            let b = topk_ids(&spliced, q, k);
            let sa: HashSet<i128> = a.iter().copied().collect();
            let sb: HashSet<i128> = b.iter().copied().collect();
            let inter = sa.intersection(&sb).count();
            let recall = inter as f32 / a.len() as f32;
            assert_eq!(recall, 1.0, "splice must preserve recall exactly");
        }
    }
}
