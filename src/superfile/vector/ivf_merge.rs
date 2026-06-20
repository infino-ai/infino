// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Byte-splice merge of Sq8+ε IVF subsections for compaction.
//!
//! Concatenates per-cluster blocks across inputs, remapping local doc ids,
//! and Sq8-transcodes rerank rows only when a source cluster's quantizer
//! differs from the destination — no fp32 corpus buffer and no re-kmeans.

use crate::superfile::{
    BuildError,
    format::{
        checksum::crc32c,
        vec::{
            CLUSTER_IDX_COUNT_OFFSET, CLUSTER_IDX_ENTRY_BYTES, MAGIC_BYTES, U32_BYTES, U64_BYTES,
            sub_hdr,
        },
    },
    vector::{
        cell_posting::{
            EncodedCellRow, materialize_sq8_residual_row_into_cluster_quant,
            sq8_quant_params_equal, sq8_residual_norm_sq,
        },
        distance::Metric,
        quant::BitQuantizer,
        reader::VectorReader,
        rerank_codec::RerankCodec,
    },
};

const SUB_HEADER_SIZE: usize = crate::superfile::format::vec::SUB_HEADER_SIZE;

/// One input superfile column for byte-splice merge.
pub(crate) struct Sq8IvfMergeInput {
    pub sub: Vec<u8>,
    pub dim: usize,
    pub n_cent: usize,
    pub n_docs: u32,
    pub metric: Metric,
    pub doc_id_offset: u32,
    pub cluster_idx_off: usize,
    pub centroids_off: usize,
    pub per_cluster_blocks_off: usize,
    pub code_bytes: usize,
    pub per_vec_bytes: usize,
    pub stride: usize,
    pub scale: Vec<f32>,
    pub offset: Vec<f32>,
    pub summary_radius_x100: u32,
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

fn cluster_entry(sub: &[u8], cluster_idx_off: usize, c: usize) -> (usize, usize) {
    let e = cluster_idx_off + c * CLUSTER_IDX_ENTRY_BYTES;
    let doc_off = u32::from_le_bytes([sub[e], sub[e + 1], sub[e + 2], sub[e + 3]]) as usize;
    let cb = e + CLUSTER_IDX_COUNT_OFFSET;
    let count = u32::from_le_bytes([sub[cb], sub[cb + 1], sub[cb + 2], sub[cb + 3]]) as usize;
    (doc_off, count)
}

fn centroid_storage_order(centroids: &[f32], n_cent: usize, dim: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n_cent).collect();
    order.sort_by(|&a, &b| {
        let ca = &centroids[a * dim..(a + 1) * dim];
        let cb = &centroids[b * dim..(b + 1) * dim];
        ca.iter()
            .zip(cb)
            .map(|(x, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .find(|o| *o != std::cmp::Ordering::Equal)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
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
    let codec = RerankCodec::Sq8ResidualEpsilon;
    let quant = BitQuantizer::new(dim);
    let code_bytes = quant.code_bytes();
    let per_vec_bytes = codec.per_vector_bytes(dim);
    let store_norm = matches!(metric, Metric::L2Sq | Metric::Cosine);

    let mut out_centroids = vec![0.0f32; n_cent * dim];
    for c in 0..n_cent {
        let mut acc = vec![0.0f64; dim];
        let mut total = 0u64;
        for inp in &parsed {
            let (_, count) = cluster_entry(&inp.sub, inp.cluster_idx_off, c);
            if count == 0 {
                continue;
            }
            total += count as u64;
            let co = inp.centroids_off + c * dim * 4;
            for (d, acc_d) in acc.iter_mut().enumerate().take(dim) {
                let v = f32::from_le_bytes([
                    inp.sub[co + d * 4],
                    inp.sub[co + d * 4 + 1],
                    inp.sub[co + d * 4 + 2],
                    inp.sub[co + d * 4 + 3],
                ]);
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

    let summary_size = dim * 4;
    let centroids_size = n_cent * dim * 4;
    let cluster_idx_size = n_cent * CLUSTER_IDX_ENTRY_BYTES;
    let codec_meta_size = codec.codec_meta_bytes(dim, n_docs as usize, n_cent, metric);
    let per_cluster_blocks_size = n_docs as usize
        * (code_bytes + crate::superfile::format::vec::DOC_ID_BYTES + per_vec_bytes);

    let summary_off = SUB_HEADER_SIZE;
    let centroids_off = summary_off + summary_size;
    let cluster_idx_off = centroids_off + centroids_size;
    let codec_meta_off = cluster_idx_off + cluster_idx_size;
    let per_cluster_blocks_off = codec_meta_off + codec_meta_size;

    let total_size_before_crc = SUB_HEADER_SIZE
        + summary_size
        + centroids_size
        + cluster_idx_size
        + codec_meta_size
        + per_cluster_blocks_size;

    let mut bytes = vec![0u8; total_size_before_crc];

    bytes[0..MAGIC_BYTES].copy_from_slice(crate::superfile::format::vec::SUB_MAGIC);
    bytes[sub_hdr::VERSION_OFF..sub_hdr::VERSION_OFF + U32_BYTES]
        .copy_from_slice(&crate::superfile::format::vec::SUBSECTION_VERSION.to_le_bytes());
    bytes[sub_hdr::CODEC_META_SIZE_OFF..sub_hdr::CODEC_META_SIZE_OFF + U32_BYTES]
        .copy_from_slice(&(codec_meta_size as u32).to_le_bytes());
    bytes[sub_hdr::SUMMARY_OFF_OFF..sub_hdr::SUMMARY_OFF_OFF + U64_BYTES]
        .copy_from_slice(&(summary_off as u64).to_le_bytes());
    bytes[sub_hdr::SUMMARY_RADIUS_X100_OFF..sub_hdr::SUMMARY_RADIUS_X100_OFF + U32_BYTES]
        .copy_from_slice(&summary_radius_x100.to_le_bytes());
    bytes[sub_hdr::CENTROIDS_OFF_OFF..sub_hdr::CENTROIDS_OFF_OFF + U64_BYTES]
        .copy_from_slice(&(centroids_off as u64).to_le_bytes());
    bytes[sub_hdr::CLUSTER_IDX_OFF_OFF..sub_hdr::CLUSTER_IDX_OFF_OFF + U64_BYTES]
        .copy_from_slice(&(cluster_idx_off as u64).to_le_bytes());
    bytes[sub_hdr::PER_CLUSTER_BLOCKS_OFF_OFF..sub_hdr::PER_CLUSTER_BLOCKS_OFF_OFF + U64_BYTES]
        .copy_from_slice(&(per_cluster_blocks_off as u64).to_le_bytes());

    bytes[summary_off..summary_off + summary_size]
        .copy_from_slice(bytemuck::cast_slice(&summary_centroid));
    bytes[centroids_off..centroids_off + centroids_size]
        .copy_from_slice(bytemuck::cast_slice(&out_centroids));

    let sq8_scale_block_off = codec_meta_off;
    let sq8_offset_block_off = sq8_scale_block_off + n_cent * dim * 4;
    let sq8_norms_block_off = if store_norm {
        Some(sq8_offset_block_off + n_cent * dim * 4)
    } else {
        None
    };

    for c in 0..n_cent {
        let sc_off = sq8_scale_block_off + c * dim * 4;
        bytes[sc_off..sc_off + dim * 4]
            .copy_from_slice(bytemuck::cast_slice(&dst_scale[c * dim..c * dim + dim]));
        let oc_off = sq8_offset_block_off + c * dim * 4;
        bytes[oc_off..oc_off + dim * 4]
            .copy_from_slice(bytemuck::cast_slice(&dst_offset[c * dim..c * dim + dim]));
    }

    let cluster_order = centroid_storage_order(&out_centroids, n_cent, dim);
    let cluster_stride = code_bytes + crate::superfile::format::vec::DOC_ID_BYTES + per_vec_bytes;
    let mut block_cursor = 0usize;
    let mut acc_off = 0u32;
    let mut row_buf = vec![0u8; dim * 2];
    let id_bytes = crate::superfile::format::vec::DOC_ID_BYTES;

    for &centroid_id in &cluster_order {
        let mut cluster_count = 0u32;
        for inp in &parsed {
            let (_, count) = cluster_entry(&inp.sub, inp.cluster_idx_off, centroid_id);
            cluster_count += count as u32;
        }

        let idx_base = cluster_idx_off + centroid_id * CLUSTER_IDX_ENTRY_BYTES;
        bytes[idx_base..idx_base + CLUSTER_IDX_COUNT_OFFSET]
            .copy_from_slice(&acc_off.to_le_bytes());
        bytes[idx_base + CLUSTER_IDX_COUNT_OFFSET..idx_base + CLUSTER_IDX_ENTRY_BYTES]
            .copy_from_slice(&cluster_count.to_le_bytes());

        if cluster_count > 0 {
            let cnt = cluster_count as usize;
            let block_base = per_cluster_blocks_off + block_cursor;
            let codes_len = cnt * code_bytes;
            let ids_len = cnt * id_bytes;
            let full_chunk_base = block_base + codes_len + ids_len;
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
                    bytes[block_base + out_i * code_bytes..block_base + (out_i + 1) * code_bytes]
                        .copy_from_slice(
                            &inp.sub[block + i * inp.code_bytes..block + (i + 1) * inp.code_bytes],
                        );

                    let idb = doc_ids_at + i * id_bytes;
                    let local_id = u32::from_le_bytes([
                        inp.sub[idb],
                        inp.sub[idb + 1],
                        inp.sub[idb + 2],
                        inp.sub[idb + 3],
                    ]) + inp.doc_id_offset;
                    let id_off = block_base + codes_len + out_i * id_bytes;
                    bytes[id_off..id_off + id_bytes].copy_from_slice(&local_id.to_le_bytes());

                    let rowb = full_at + i * inp.per_vec_bytes;
                    let full_off = full_chunk_base + out_i * per_vec_bytes;
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
                        let n_off = norms_off + (acc_off as usize + out_i) * 4;
                        bytes[n_off..n_off + 4].copy_from_slice(&n_sq.to_le_bytes());
                    }
                    out_i += 1;
                }
            }
            debug_assert_eq!(out_i, cnt);
            block_cursor += cnt * cluster_stride;
        }
        acc_off += cluster_count;
    }

    let crc = crc32c(&bytes);
    bytes.extend_from_slice(&crc.to_le_bytes());

    Ok(MergedIvfSubsection {
        bytes,
        n_cent,
        n_docs,
        summary_offset_in_sub: summary_off,
        codec_meta_offset_in_sub: if codec_meta_size == 0 {
            0
        } else {
            codec_meta_off
        },
        codec_meta_size,
    })
}
