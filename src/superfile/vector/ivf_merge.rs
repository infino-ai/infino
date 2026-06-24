// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Byte-splice merge of Sq8+ε IVF subsections for compaction.
//!
//! Concatenates per-cluster blocks across inputs, remapping local doc ids,
//! and Sq8-transcodes rerank rows only when a source cluster's quantizer
//! differs from the destination — no fp32 corpus buffer and no re-kmeans.

use bytemuck::cast_slice;

use crate::superfile::{
    BuildError,
    format::{checksum::crc32c, vec::{DOC_ID_BYTES, STABLE_ID_BYTES}},
    vector::{
        builder::{
            IvfSubsectionLayout, alloc_ivf_subsection_with_header, centroid_storage_order,
            write_ivf_cluster_blocks,
        },
        cell_posting::{
            EncodedCellRow, materialize_sq8_residual_row_into_cluster_quant,
            sq8_quant_params_equal, sq8_residual_norm_sq,
        },
        centroid_block::CentroidBlock,
        distance::Metric,
        quant::BitQuantizer,
        reader::{VectorReader, read_cluster_entry},
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
    let id_bytes = DOC_ID_BYTES;
    let mut row_buf = vec![0u8; dim * 2];
    let stable_ids_region_off = layout.stable_ids_off;

    write_ivf_cluster_blocks(
        &mut bytes,
        &layout,
        &cluster_order,
        &merged_counts,
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
                        let sid = inp.stable_ids.as_ref().expect("produce_region")
                            [src_local as usize];
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
                                scale: src_scale.to_vec(),
                                offset: src_offset.to_vec(),
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
