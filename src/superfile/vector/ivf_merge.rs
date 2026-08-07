// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Byte-splice merge of Sq8+ε IVF subsections for compaction.
//!
//! Concatenates per-cluster blocks across inputs, remapping local doc ids,
//! and Sq8-transcodes rerank rows only when a source cluster's quantizer
//! differs from the destination — no fp32 corpus buffer and no re-kmeans.

use std::collections::HashMap;

use bytemuck::cast_slice;
use rayon::prelude::*;

use crate::superfile::{
    BuildError,
    format::{
        CRC_BYTES,
        checksum::crc32c,
        vec::{
            CLUSTER_IDX_ENTRY_BYTES, DOC_ID_BYTES, STABLE_ID_BYTES, SUB_HEADER_SIZE, U32_BYTES,
            U64_BYTES, sub_hdr,
        },
    },
    vector::{
        builder::{
            IvfSubsectionLayout, alloc_ivf_subsection_with_header, centroid_storage_order,
            fixed_sq8_quantizer, write_ivf_cluster_blocks,
        },
        cell_posting::{EncodedCellRow, sq8_quant_params_equal},
        distance::{
            Metric, add_weighted_f32_to_f64_acc, decode_f32_le_into, decode_f32_le_vec,
            f64_acc_mean_into_f32, mean_f32_cluster_major,
        },
        quant::BitQuantizer,
        reader::{VectorReader, read_cluster_entry},
        rerank_codec::RerankCodec,
    },
};

/// Read a fragment's stable id at `src_local` (a doc id decoded from stored
/// bytes), bounds-checked against the fragment's stable-id table. A corrupt or
/// boundary-replicated short slice can push `src_local` past the table; return
/// a typed `BuildError` rather than panicking the rayon drain worker on an
/// out-of-range index.
#[inline]
fn stable_id_at(sids: &[i128], src_local: u32) -> Result<i128, BuildError> {
    sids.get(src_local as usize).copied().ok_or_else(|| {
        BuildError::VectorSchemaMismatch(
            "cell fragment doc-id out of range for stable-id table".into(),
        )
    })
}

/// One input superfile column for byte-splice merge.
pub(crate) struct Sq8IvfMergeInput {
    pub sub: Vec<u8>,
    pub dim: usize,
    pub n_cent: usize,
    pub n_docs: u32,
    pub metric: Metric,
    pub rerank_codec: RerankCodec,
    pub doc_id_offset: u32,
    pub cluster_idx_off: usize,
    pub centroids_off: usize,
    pub per_cluster_blocks_off: usize,
    pub code_bytes: usize,
    pub per_vec_bytes: usize,
    pub stride: usize,
    pub scale: Vec<f32>,
    pub offset: Vec<f32>,
    /// Inline stable-`_id`s for this input, indexed by its local doc id, when
    /// the source subsection carries the region (materialized/hidden cells).
    /// `None` for region-less sources (streaming/incoming). The merge produces a
    /// merged region only when every input has one.
    pub stable_ids: Option<Vec<i128>>,
}

/// Output of a byte-splice merge, ready for [`super::builder::VectorBuilder::set_prebuilt_subsection`].
pub(crate) struct MergedIvfSubsection {
    pub bytes: Vec<u8>,
    pub n_cent: usize,
    pub n_docs: u32,
    pub rerank_codec: RerankCodec,
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
    merge_sq8_ivf_subsections_from_parsed(&parsed)
}

/// Same as [`merge_sq8_ivf_subsections`], but takes already-parsed cell IVFs
/// (used when packing/unpacking multi-cell v2 blobs per `global_cell_id`).
pub(crate) fn merge_sq8_ivf_subsections_from_parsed(
    parsed: &[Sq8IvfMergeInput],
) -> Result<MergedIvfSubsection, BuildError> {
    if parsed.is_empty() {
        return Err(BuildError::VectorSchemaMismatch(
            "merge requires at least one IVF input".into(),
        ));
    }

    let dim = parsed[0].dim;
    let n_cent = parsed[0].n_cent;
    let metric = parsed[0].metric;
    let codec = parsed[0].rerank_codec;
    for inp in &parsed[1..] {
        if inp.dim != dim
            || inp.n_cent != n_cent
            || inp.metric != metric
            || inp.rerank_codec != codec
        {
            return Err(BuildError::VectorSchemaMismatch(
                "Sq8 IVF merge inputs must share dim, n_cent, metric, and codec".into(),
            ));
        }
    }

    let n_docs: u32 = parsed.iter().map(|p| p.n_docs).sum();
    debug_assert!(codec.is_ivf_mergeable());
    let quant = BitQuantizer::new(dim);
    let code_bytes = quant.code_bytes();
    let per_vec_bytes = codec.per_vector_bytes(dim);
    let store_norm = matches!(metric, Metric::L2Sq | Metric::Cosine);

    let mut out_centroids = vec![0.0f32; n_cent * dim];
    let mut cent_buf = vec![0f32; dim];
    for c in 0..n_cent {
        let mut acc = vec![0.0f64; dim];
        let mut total = 0u64;
        for inp in parsed {
            let (_, count) = cluster_entry(&inp.sub, inp.cluster_idx_off, c);
            if count == 0 {
                continue;
            }
            total += count as u64;
            let co = inp.centroids_off + c * dim * 4;
            decode_f32_le_into(&inp.sub[co..co + dim * 4], &mut cent_buf);
            add_weighted_f32_to_f64_acc(&mut acc, &cent_buf, count as f64);
        }
        if total > 0 {
            f64_acc_mean_into_f32(
                &acc,
                1.0 / total as f64,
                &mut out_centroids[c * dim..(c + 1) * dim],
            );
        }
    }

    let summary_centroid = mean_f32_cluster_major(&out_centroids, dim, n_cent);

    // Seed the merged quantizer table with the pinned constants when the
    // codec's quantizer is fixed: a cluster that is empty in every input
    // keeps its seed slot, and the reader's open-time validation requires
    // every slot — populated or empty — to carry the pinned scale/offset
    // bitwise, exactly as the direct build paths write them. Fitted codecs
    // keep the neutral 1.0/0.0 seed; their empty slots are never read.
    // Only the residual family seeds a per-cluster scale/offset table.
    // Merge inputs are built exclusively for the residual family
    // (`sq8_ivf_merge_input_at` is `is_ivf_mergeable`-gated), so
    // the extra family check is a guard: single-plane Sq16 must never
    // seed the fixed sq8 grid here.
    // Only cluster-quant codecs carry a per-cluster scale/offset table.
    // The fixed-grid single-plane codec (Sq16) quantizes on a fixed grid with no
    // per-cluster quantizer, so its `scale`/`offset` merge inputs are empty and
    // never seeded, sliced, or written to codec_meta below.
    let has_cluster_quant = codec.carries_cluster_quant_meta();
    let (mut dst_scale, mut dst_offset) = if codec.uses_fixed_quantizer() && has_cluster_quant {
        let (scale, offset) = fixed_sq8_quantizer(dim);
        (scale.repeat(n_cent), offset.repeat(n_cent))
    } else {
        (vec![1.0f32; n_cent * dim], vec![0.0f32; n_cent * dim])
    };
    if has_cluster_quant {
        // Copy the first contributing input's per-cluster ruler as the
        // destination grid; per-row transcode below re-encodes the other inputs
        // onto it (clamping any out-of-range component). This matches the
        // existing merge behavior for every fitted/fixed cluster-quant codec.
        // (Sizing the destination grid to cover every input's range so no
        // component clamps is a separate future change, not bundled here.)
        for c in 0..n_cent {
            for inp in parsed {
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
    }

    let codec_meta_size = codec.codec_meta_bytes(dim, n_docs as usize, n_cent, metric);
    let cluster_stride = code_bytes + DOC_ID_BYTES + per_vec_bytes;
    // Carry the inline stable-`_id` region through the splice when every input
    // has one (materialized/hidden cells always do; streaming/incoming sources
    // don't). All-or-nothing: a merged region must cover every merged local id,
    // so a single region-less input means we emit none and the merged cell
    // falls back to the scalar `_id` column (still correct). The region is
    // rewritten in merged local-id order in the cluster-block loop below.
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
    );

    let mut bytes = alloc_ivf_subsection_with_header(
        &layout,
        codec_meta_size,
        &summary_centroid,
        &out_centroids,
    );

    // codec_meta layout comes from the codec's own ops: the residual family
    // lays out `[scale | offset | norms?]`; Sq16 lays out `[norms?]` at the
    // head with no scale/offset arrays. The per-doc norm block offset follows
    // whichever shape the codec declares.
    let ops = codec.ops().expect("mergeable codec has cold-path ops");
    let meta_layout = ops.codec_meta_layout(layout.codec_meta_off, n_cent, dim, metric);
    let norms_block_off = meta_layout.norms_off;

    if let (Some(scale_off), Some(offset_off)) = (meta_layout.scale_off, meta_layout.offset_off) {
        for c in 0..n_cent {
            let sc_off = scale_off + c * dim * 4;
            bytes[sc_off..sc_off + dim * 4]
                .copy_from_slice(cast_slice(&dst_scale[c * dim..c * dim + dim]));
            let oc_off = offset_off + c * dim * 4;
            bytes[oc_off..oc_off + dim * 4]
                .copy_from_slice(cast_slice(&dst_offset[c * dim..c * dim + dim]));
        }
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
    // Relative offset of the merged stable-`_id` region (start of the i128s),
    // `Some` exactly when `produce_region`. Written per row below, indexed by
    // the merged local doc id.
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

            for inp in parsed {
                let (doc_off, count) = cluster_entry(&inp.sub, inp.cluster_idx_off, centroid_id);
                if count == 0 {
                    continue;
                }
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

                    // Carry the stable `_id` to the merged region at the same
                    // (remapped) local id. `produce_region` guarantees every
                    // input has `stable_ids`, so the index is in range.
                    if let Some(region_off) = stable_ids_region_off {
                        let sid = stable_id_at(
                            inp.stable_ids.as_ref().expect("produce_region"),
                            src_local,
                        )?;
                        let p = region_off + (local_id as usize) * STABLE_ID_BYTES;
                        bytes[p..p + STABLE_ID_BYTES].copy_from_slice(&sid.to_le_bytes());
                    }

                    let rowb = full_at + i * inp.per_vec_bytes;
                    let full_off = blk.rerank_base + out_i * per_vec_bytes;
                    let norm_sq = if !has_cluster_quant {
                        // Single-plane fixed-grid codec (Sq16): the body is
                        // portable across cluster changes, so a merge is a
                        // verbatim byte copy with the norm read off the grid.
                        bytes[full_off..full_off + per_vec_bytes]
                            .copy_from_slice(&inp.sub[rowb..rowb + per_vec_bytes]);
                        store_norm.then(|| {
                            ops.decoded_norm_sq(&inp.sub[rowb..rowb + per_vec_bytes], dim, &[], &[])
                        })
                    } else {
                        let src_scale = &inp.scale[centroid_id * dim..centroid_id * dim + dim];
                        let src_offset = &inp.offset[centroid_id * dim..centroid_id * dim + dim];
                        if sq8_quant_params_equal(src_scale, src_offset, scale_c, offset_c) {
                            bytes[full_off..full_off + dim * 2]
                                .copy_from_slice(&inp.sub[rowb..rowb + dim * 2]);
                            store_norm.then(|| {
                                ops.decoded_norm_sq(
                                    &inp.sub[rowb..rowb + dim * 2],
                                    dim,
                                    scale_c,
                                    offset_c,
                                )
                            })
                        } else {
                            // Body layout depends on the codec: the residual
                            // family splits `[codes(dim) ‖ residuals(dim)]`,
                            // while the single-`u16`-plane `Sq16Adaptive` keeps
                            // the whole `dim*2`-byte plane as `codes` with no
                            // residual leg.
                            let (codes, residuals) = if inp.rerank_codec.writes_single_u16_plane() {
                                (inp.sub[rowb..rowb + dim * 2].to_vec(), Vec::new())
                            } else {
                                (
                                    inp.sub[rowb..rowb + dim].to_vec(),
                                    inp.sub[rowb + dim..rowb + dim + dim].to_vec(),
                                )
                            };
                            let encoded = EncodedCellRow {
                                stable_id: 0,
                                rerank_codec: inp.rerank_codec,
                                scale: std::sync::Arc::from(src_scale),
                                offset: std::sync::Arc::from(src_offset),
                                codes,
                                residuals,
                                norm_sq: None,
                            };
                            let n = ops.materialize_row_into_cluster_quant(
                                &encoded,
                                scale_c,
                                offset_c,
                                dim,
                                &mut row_buf,
                                store_norm,
                            )?;
                            bytes[full_off..full_off + dim * 2].copy_from_slice(&row_buf);
                            n
                        }
                    };

                    if let (Some(norms_off), Some(n_sq)) = (norms_block_off, norm_sq) {
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
        rerank_codec: codec,
        summary_offset_in_sub: layout.summary_off,
        codec_meta_offset_in_sub: if codec_meta_size == 0 {
            0
        } else {
            layout.codec_meta_off
        },
        codec_meta_size,
    })
}

/// Stable `_id`s in merged local-doc-id order for inputs that all carry an
/// inline stable-id region (hidden / materialized cells).
pub(crate) fn stable_ids_in_merged_local_order(
    parsed: &[Sq8IvfMergeInput],
) -> Result<Vec<i128>, BuildError> {
    if parsed.is_empty() {
        return Ok(Vec::new());
    }
    if !parsed.iter().all(|p| p.stable_ids.is_some()) {
        return Err(BuildError::VectorSchemaMismatch(
            "multi-cell merge requires inline stable_ids on every cell IVF".into(),
        ));
    }
    let n_docs: usize = parsed.iter().map(|p| p.n_docs as usize).sum();
    let mut ids = vec![0i128; n_docs];
    for inp in parsed {
        let src = inp.stable_ids.as_ref().expect("checked above");
        let base = inp.doc_id_offset as usize;
        for (i, &sid) in src.iter().enumerate() {
            ids[base + i] = sid;
        }
    }
    Ok(ids)
}

/// Splice routed source clusters into one hidden-cell superfile as a
/// **multi-cluster** IVF: each fragment (one source cluster from one input)
/// becomes its own output cluster, copied **verbatim** — its centroid, its Sq8
/// calibration, its code+rerank block. This is what restores *inner pruning*:
/// a query scores the fragment centroids and scans only the near ones. No
/// averaging, no transcode, no decode — each output cluster reuses its own
/// fragment's calibration, so rerank rows copy byte-for-byte.
///
/// Output local doc ids are fresh + contiguous (`0..n`) in cluster-storage
/// order; identity rides the inline stable-`_id` region and is also returned in
/// id-column order. `fragments` empty (or all-empty) ⇒ `None`. Each fragment is
/// `(input, source-cluster index within that input, that input's stable ids)`.
pub(crate) fn splice_fragments_into_cell(
    fragments: &[(&Sq8IvfMergeInput, usize, &[i128])],
) -> Result<Option<(MergedIvfSubsection, Vec<i128>)>, BuildError> {
    if fragments.is_empty() {
        return Ok(None);
    }
    let dim = fragments[0].0.dim;
    let metric = fragments[0].0.metric;
    let codec = fragments[0].0.rerank_codec;
    for (inp, _, _) in &fragments[1..] {
        if inp.dim != dim || inp.metric != metric || inp.rerank_codec != codec {
            return Err(BuildError::VectorSchemaMismatch(
                "fragment splice inputs must share dim, metric, and codec".into(),
            ));
        }
    }

    // The residual family carries a per-cluster quantizer (`[scale|offset]`
    // codec_meta blocks); the single-plane fixed-grid `Sq16` carries only the
    // per-doc norm table. Splice supports both: the rerank bytes are copied
    // verbatim regardless of codec, and the codec_meta write below branches on
    // this to lay out either `[scale|offset|norms]` or norms-only.
    let has_cluster_quant = codec.carries_cluster_quant_meta();

    let out_n_cent = fragments.len();
    let counts: Vec<u32> = fragments
        .iter()
        .map(|(inp, c, _)| cluster_entry(&inp.sub, inp.cluster_idx_off, *c).1 as u32)
        .collect();
    let n_docs: u32 = counts.iter().sum();
    if n_docs == 0 {
        return Ok(None);
    }

    debug_assert!(codec.is_ivf_mergeable());
    let quant = BitQuantizer::new(dim);
    let code_bytes = quant.code_bytes();
    let per_vec_bytes = codec.per_vector_bytes(dim);
    let store_norm = matches!(metric, Metric::L2Sq | Metric::Cosine);
    let id_bytes = DOC_ID_BYTES;

    // Output cluster k = fragment k: copy its centroid + Sq8 calibration verbatim.
    let mut out_centroids = vec![0.0f32; out_n_cent * dim];
    let mut dst_scale = vec![1.0f32; out_n_cent * dim];
    let mut dst_offset = vec![0.0f32; out_n_cent * dim];
    for (k, (inp, c, _)) in fragments.iter().enumerate() {
        let co = inp.centroids_off + c * dim * 4;
        decode_f32_le_into(
            &inp.sub[co..co + dim * 4],
            &mut out_centroids[k * dim..(k + 1) * dim],
        );
        // Sq16 carries no per-cluster quantizer (empty scale/offset); its
        // rerank bytes decode off the fixed grid. Only the residual family has
        // per-cluster scale/offset to carry through.
        if has_cluster_quant {
            dst_scale[k * dim..(k + 1) * dim].copy_from_slice(&inp.scale[c * dim..c * dim + dim]);
            dst_offset[k * dim..(k + 1) * dim].copy_from_slice(&inp.offset[c * dim..c * dim + dim]);
        }
    }

    // Summary centroid = mean of fragment centroids.
    let summary_centroid = mean_f32_cluster_major(&out_centroids, dim, out_n_cent);

    let codec_meta_size = codec.codec_meta_bytes(dim, n_docs as usize, out_n_cent, metric);
    let cluster_stride = code_bytes + id_bytes + per_vec_bytes;
    let stable_ids_region_bytes = n_docs as usize * STABLE_ID_BYTES;
    let layout = IvfSubsectionLayout::compute(
        dim,
        out_n_cent,
        n_docs as usize,
        cluster_stride,
        codec_meta_size,
        stable_ids_region_bytes,
    );

    let mut bytes = alloc_ivf_subsection_with_header(
        &layout,
        codec_meta_size,
        &summary_centroid,
        &out_centroids,
    );

    // codec_meta layout: the residual family writes per-cluster [scale|offset]
    // blocks (one dim-slot per output cluster) then the per-doc norm table;
    // single-plane Sq16 writes only the norm table at the codec_meta head.
    let sq8_norms_block_off = if has_cluster_quant {
        let scale_off = layout.codec_meta_off;
        let offset_off = scale_off + out_n_cent * dim * 4;
        bytes[scale_off..scale_off + out_n_cent * dim * 4].copy_from_slice(cast_slice(&dst_scale));
        bytes[offset_off..offset_off + out_n_cent * dim * 4]
            .copy_from_slice(cast_slice(&dst_offset));
        store_norm.then_some(offset_off + out_n_cent * dim * 4)
    } else {
        store_norm.then_some(layout.codec_meta_off)
    };

    let stable_ids_region_off = layout.stable_ids_off;
    let mut out_stable_ids = vec![0i128; n_docs as usize];
    let cluster_order = centroid_storage_order(&out_centroids, out_n_cent, dim);

    write_ivf_cluster_blocks(
        &mut bytes,
        &layout,
        &cluster_order,
        &counts,
        code_bytes,
        per_vec_bytes,
        |bytes, centroid_id, blk| {
            // Output cluster `centroid_id` = fragment `centroid_id`, verbatim.
            let (inp, src_cluster, sids) = fragments[centroid_id];
            let scale_c = &dst_scale[centroid_id * dim..centroid_id * dim + dim];
            let offset_c = &dst_offset[centroid_id * dim..centroid_id * dim + dim];
            let (doc_off, count) = cluster_entry(&inp.sub, inp.cluster_idx_off, src_cluster);
            let block = inp.per_cluster_blocks_off + doc_off * inp.stride;
            let doc_ids_at = block + count * inp.code_bytes;
            let full_at = block + count * (inp.code_bytes + id_bytes);
            for i in 0..count {
                let out_row = blk.first_row + i; // fresh global local doc id
                bytes[blk.codes_base + i * code_bytes..blk.codes_base + (i + 1) * code_bytes]
                    .copy_from_slice(
                        &inp.sub[block + i * inp.code_bytes..block + (i + 1) * inp.code_bytes],
                    );
                let id_off = blk.ids_base + i * id_bytes;
                bytes[id_off..id_off + id_bytes].copy_from_slice(&(out_row as u32).to_le_bytes());

                let idb = doc_ids_at + i * id_bytes;
                let src_local = u32::from_le_bytes([
                    inp.sub[idb],
                    inp.sub[idb + 1],
                    inp.sub[idb + 2],
                    inp.sub[idb + 3],
                ]);
                let sid = stable_id_at(sids, src_local)?;
                out_stable_ids[out_row] = sid;
                if let Some(region_off) = stable_ids_region_off {
                    let p = region_off + out_row * STABLE_ID_BYTES;
                    bytes[p..p + STABLE_ID_BYTES].copy_from_slice(&sid.to_le_bytes());
                }

                // Rerank: verbatim — the output cluster uses this fragment's own
                // calibration, so no transcode is ever needed.
                let rowb = full_at + i * inp.per_vec_bytes;
                let full_off = blk.rerank_base + i * per_vec_bytes;
                bytes[full_off..full_off + dim * 2].copy_from_slice(&inp.sub[rowb..rowb + dim * 2]);
                if store_norm && let Some(norms_off) = sq8_norms_block_off {
                    let n_sq = codec
                        .ops()
                        .expect("mergeable codec has cold-path ops")
                        .decoded_norm_sq(&inp.sub[rowb..rowb + dim * 2], dim, scale_c, offset_c);
                    let n_off = norms_off + out_row * 4;
                    bytes[n_off..n_off + 4].copy_from_slice(&n_sq.to_le_bytes());
                }
            }
            debug_assert_eq!(count, blk.count);
            Ok(())
        },
    )?;

    let crc = crc32c(&bytes);
    bytes.extend_from_slice(&crc.to_le_bytes());

    Ok(Some((
        MergedIvfSubsection {
            bytes,
            n_cent: out_n_cent,
            n_docs,
            rerank_codec: codec,
            summary_offset_in_sub: layout.summary_off,
            codec_meta_offset_in_sub: if codec_meta_size == 0 {
                0
            } else {
                layout.codec_meta_off
            },
            codec_meta_size,
        },
        out_stable_ids,
    )))
}

/// Route each input's local clusters to their nearest global cell(s) and splice
/// the routed clusters into per-cell **multi-cluster (fragment)** subsections —
/// the structure that preserves inner pruning (vs the flat concat that lost it).
///
/// `route_cluster(local_centroid_fp32) -> dest cells` is caller-supplied (one
/// cell for an interior cluster; several for SPANN boundary replication), so
/// this stays free of the global-cell-grid types. Parses inputs once; the
/// per-cell splice runs in parallel; results are in-memory (no spool).
///
/// Multi-cell (v2) user superfiles expand to one merge input per packed cell.
/// Those subsections carry inline stable-`_id`s; the caller slice is only used
/// for single-cell (v1) inputs that lack an inline region.
pub(crate) fn route_clusters_into_cells<F>(
    inputs: &[(&VectorReader, &str)],
    stable_ids_per_input: &[Vec<i128>],
    route_cluster: F,
) -> Result<HashMap<u32, (MergedIvfSubsection, Vec<i128>)>, BuildError>
where
    F: Fn(&[f32]) -> Vec<u32> + Sync,
{
    if inputs.is_empty() {
        return Err(BuildError::VectorSchemaMismatch(
            "route_clusters_into_cells requires at least one IVF input".into(),
        ));
    }
    if stable_ids_per_input.len() != inputs.len() {
        return Err(BuildError::VectorSchemaMismatch(
            "route_clusters_into_cells: stable_ids_per_input must match inputs len".into(),
        ));
    }

    // One parse entry per IVF subsection. Multi-cell packs contribute one entry
    // per packed cell; v1 contributes one. Stable ids ride the subsection's
    // inline region when present (required for multi-cell), else the caller's
    // per-superfile slice (v1 streaming / no-region).
    let mut parsed: Vec<Sq8IvfMergeInput> = Vec::new();
    let mut stable_ids: Vec<Vec<i128>> = Vec::new();
    for (reader_i, (reader, col)) in inputs.iter().enumerate() {
        if reader.is_multi_cell() {
            let n_cells = reader.packed_cell_ids().len();
            for cell_idx in 0..n_cells {
                let inp = reader.sq8_ivf_merge_input_at(cell_idx, 0)?;
                let ids = inp.stable_ids.clone().ok_or_else(|| {
                    BuildError::VectorSchemaMismatch(format!(
                        "route_clusters_into_cells: multi-cell packed cell {cell_idx} missing inline stable ids"
                    ))
                })?;
                parsed.push(inp);
                stable_ids.push(ids);
            }
        } else {
            let inp = reader.sq8_ivf_merge_input(col, 0)?;
            let ids = match inp.stable_ids.clone() {
                Some(ids) => ids,
                None => stable_ids_per_input[reader_i].clone(),
            };
            parsed.push(inp);
            stable_ids.push(ids);
        }
    }
    if parsed.is_empty() {
        return Err(BuildError::VectorSchemaMismatch(
            "route_clusters_into_cells: no IVF subsections to route".into(),
        ));
    }
    let dim = parsed[0].dim;

    // Route each non-empty (input, local cluster) by its centroid → dest cell(s).
    let mut cell_frags: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
    let mut centroid_buf = vec![0f32; dim];
    for (ii, inp) in parsed.iter().enumerate() {
        for c in 0..inp.n_cent {
            let (_, count) = cluster_entry(&inp.sub, inp.cluster_idx_off, c);
            if count == 0 {
                continue;
            }
            let co = inp.centroids_off + c * dim * 4;
            decode_f32_le_into(&inp.sub[co..co + dim * 4], &mut centroid_buf);
            let dests = route_cluster(&centroid_buf);
            if dests.is_empty() {
                return Err(BuildError::VectorSchemaMismatch(format!(
                    "route_clusters_into_cells: non-empty cluster {c} (count={count}) of \
                     input {ii} routed to zero cells; its {count} rows would be dropped",
                )));
            }
            for cell in dests {
                cell_frags.entry(cell).or_default().push((ii, c));
            }
        }
    }

    // Splice each cell's fragments in parallel into a multi-cluster subsection.
    let cells: Vec<(u32, Vec<(usize, usize)>)> = cell_frags.into_iter().collect();
    let out: Vec<(u32, (MergedIvfSubsection, Vec<i128>))> = cells
        .par_iter()
        .filter_map(|(cell, frags)| {
            let fragments: Vec<(&Sq8IvfMergeInput, usize, &[i128])> = frags
                .iter()
                .map(|&(ii, c)| (&parsed[ii], c, stable_ids[ii].as_slice()))
                .collect();
            match splice_fragments_into_cell(&fragments) {
                Ok(Some(res)) => Some(Ok((*cell, res))),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        })
        .collect::<Result<Vec<_>, BuildError>>()?;
    Ok(out.into_iter().collect())
}

/// Parse a fragment-style (or any Sq8 residual-family) IVF subsection into a
/// merge input. Used when the splice drain accumulates the same cell across
/// batches and must concatenate prior spilled clusters with a new batch's.
pub(crate) fn sq8_ivf_merge_input_from_subsection(
    sub: &[u8],
    dim: usize,
    n_cent: usize,
    n_docs: u32,
    metric: Metric,
    rerank_codec: RerankCodec,
    stable_ids: Option<Vec<i128>>,
) -> Result<Sq8IvfMergeInput, BuildError> {
    if !rerank_codec.is_ivf_mergeable() {
        return Err(BuildError::VectorSchemaMismatch(
            "fragment merge requires an IVF-mergeable subsection (Sq8 residual family or Sq16)"
                .into(),
        ));
    }
    if sub.len() < SUB_HEADER_SIZE + CRC_BYTES {
        return Err(BuildError::VectorSchemaMismatch(
            "subsection too short for fragment merge".into(),
        ));
    }
    let centroids_off = u64::from_le_bytes(
        sub[sub_hdr::CENTROIDS_OFF_OFF..sub_hdr::CENTROIDS_OFF_OFF + U64_BYTES]
            .try_into()
            .expect("8-byte centroids off"),
    ) as usize;
    let cluster_idx_off = u64::from_le_bytes(
        sub[sub_hdr::CLUSTER_IDX_OFF_OFF..sub_hdr::CLUSTER_IDX_OFF_OFF + U64_BYTES]
            .try_into()
            .expect("8-byte cluster idx off"),
    ) as usize;
    let per_cluster_blocks_off = u64::from_le_bytes(
        sub[sub_hdr::PER_CLUSTER_BLOCKS_OFF_OFF..sub_hdr::PER_CLUSTER_BLOCKS_OFF_OFF + U64_BYTES]
            .try_into()
            .expect("8-byte per-cluster blocks off"),
    ) as usize;
    let codec_meta_size = u32::from_le_bytes(
        sub[sub_hdr::CODEC_META_SIZE_OFF..sub_hdr::CODEC_META_SIZE_OFF + U32_BYTES]
            .try_into()
            .expect("4-byte codec meta size"),
    ) as usize;
    let codec_meta_off = cluster_idx_off + n_cent * CLUSTER_IDX_ENTRY_BYTES;
    // The residual family stores per-cluster [scale|offset] blocks in codec_meta;
    // Sq16's codec_meta is norms-only (no scale/offset). Splice reconstructs each
    // row's norm from the codes, so the merge input carries empty scale/offset for
    // Sq16 — parsing the norm table as scale/offset would error (small cell) or
    // silently corrupt the reranker (large cell).
    let (scale, offset) = if rerank_codec.carries_cluster_quant_meta() {
        let so_bytes = n_cent * dim * 4;
        if codec_meta_size < 2 * so_bytes {
            return Err(BuildError::VectorSchemaMismatch(
                "subsection codec meta too small for scale/offset blocks".into(),
            ));
        }
        if sub.len() < codec_meta_off + 2 * so_bytes {
            return Err(BuildError::VectorSchemaMismatch(
                "subsection truncated before scale/offset blocks".into(),
            ));
        }
        (
            decode_f32_le_vec(&sub[codec_meta_off..codec_meta_off + so_bytes]),
            decode_f32_le_vec(&sub[codec_meta_off + so_bytes..codec_meta_off + 2 * so_bytes]),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let quant = BitQuantizer::new(dim);
    let code_bytes = quant.code_bytes();
    let per_vec_bytes = rerank_codec.per_vector_bytes(dim);
    Ok(Sq8IvfMergeInput {
        sub: sub.to_vec(),
        dim,
        n_cent,
        n_docs,
        metric,
        rerank_codec,
        doc_id_offset: 0,
        cluster_idx_off,
        centroids_off,
        per_cluster_blocks_off,
        code_bytes,
        per_vec_bytes,
        stride: code_bytes + DOC_ID_BYTES + per_vec_bytes,
        scale,
        offset,
        stable_ids,
    })
}

/// Concatenate two fragment-style cell subsections by copying every cluster
/// verbatim (multi-batch splice drain).
pub(crate) fn merge_fragment_subsections(
    left: &MergedIvfSubsection,
    left_ids: &[i128],
    right: &MergedIvfSubsection,
    right_ids: &[i128],
    dim: usize,
    metric: Metric,
) -> Result<(MergedIvfSubsection, Vec<i128>), BuildError> {
    if left.rerank_codec != right.rerank_codec {
        return Err(BuildError::VectorSchemaMismatch(
            "fragment merge inputs must share rerank codec".into(),
        ));
    }
    if left_ids.len() != left.n_docs as usize || right_ids.len() != right.n_docs as usize {
        return Err(BuildError::VectorSchemaMismatch(
            "fragment merge stable_ids length must match n_docs".into(),
        ));
    }
    let left_inp = sq8_ivf_merge_input_from_subsection(
        &left.bytes,
        dim,
        left.n_cent,
        left.n_docs,
        metric,
        left.rerank_codec,
        Some(left_ids.to_vec()),
    )?;
    let right_inp = sq8_ivf_merge_input_from_subsection(
        &right.bytes,
        dim,
        right.n_cent,
        right.n_docs,
        metric,
        right.rerank_codec,
        Some(right_ids.to_vec()),
    )?;
    let mut fragments: Vec<(&Sq8IvfMergeInput, usize, &[i128])> =
        Vec::with_capacity(left.n_cent + right.n_cent);
    for c in 0..left.n_cent {
        fragments.push((&left_inp, c, left_ids));
    }
    for c in 0..right.n_cent {
        fragments.push((&right_inp, c, right_ids));
    }
    splice_fragments_into_cell(&fragments)?.ok_or_else(|| {
        BuildError::VectorSchemaMismatch("fragment merge produced an empty cell".into())
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::superfile::vector::{
        builder::{VectorConfig, build_merged_subsection_from_fp32},
        rerank_codec::{SQ8_FIXED_OFFSET, SQ8_FIXED_SCALE},
    };

    /// Dim of the tiny fixture corpus.
    const DIM: usize = 8;
    /// Provided grid width: one populated cluster plus three empty ones.
    const N_CENT: usize = 4;
    /// Rows per merge input.
    const ROWS: usize = 6;

    /// Build one fixed-codec cell subsection whose rows all land in cluster 0
    /// of a provided 4-centroid grid, leaving clusters 1..3 empty (count 0)
    /// by construction — the shape a fine k-means with more centroids than
    /// natural clusters produces at scale.
    fn fixed_subsection_with_empty_clusters(id_base: i128) -> MergedIvfSubsection {
        let mut centroids = vec![0.0f32; N_CENT * DIM];
        for c in 0..N_CENT {
            centroids[c * DIM + c] = 1.0;
        }
        let mut vectors = Vec::with_capacity(ROWS * DIM);
        for r in 0..ROWS {
            let mut row = [0.0f32; DIM];
            row[0] = 1.0;
            row[4 + r % 4] = 0.05 + r as f32 * 0.01;
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            vectors.extend(row.iter().map(|v| v / norm));
        }
        let ids: Vec<i128> = (0..ROWS as i128).map(|i| id_base + i).collect();
        let cfg = VectorConfig {
            column: "emb".into(),
            dim: DIM,
            rot_seed: 7,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8FixedResidual,
            provided_centroids: Some(Arc::from(centroids)),
        };
        build_merged_subsection_from_fp32(cfg, N_CENT, Arc::new(vectors), &ids).expect("cell build")
    }

    /// `merge_fragment_subsections` concatenates two fragment cells verbatim:
    /// the merged cell holds every doc from both inputs and carries all their
    /// stable ids (the multi-batch splice-drain path).
    #[test]
    fn merge_fragment_subsections_concatenates_docs_and_ids() {
        use std::collections::HashSet;

        let left = fixed_subsection_with_empty_clusters(1_000);
        let right = fixed_subsection_with_empty_clusters(2_000);
        let left_ids: Vec<i128> = (0..ROWS as i128).map(|i| 1_000 + i).collect();
        let right_ids: Vec<i128> = (0..ROWS as i128).map(|i| 2_000 + i).collect();

        let (merged, ids) =
            merge_fragment_subsections(&left, &left_ids, &right, &right_ids, DIM, Metric::Cosine)
                .expect("fragment merge");

        assert_eq!(
            merged.n_docs as usize,
            2 * ROWS,
            "merged cell holds every doc from both fragments"
        );
        assert_eq!(ids.len(), 2 * ROWS, "one stable id per merged doc");
        let got: HashSet<i128> = ids.into_iter().collect();
        for id in left_ids.iter().chain(right_ids.iter()) {
            assert!(got.contains(id), "merged ids must include {id}");
        }
    }

    /// Sq16 build helper mirroring [`fixed_subsection_with_empty_clusters`] but
    /// on the single-plane codec, whose codec_meta is norms-only (no per-cluster
    /// scale/offset). Exercises the multi-batch splice read-back for Sq16.
    fn sq16_subsection_with_empty_clusters(id_base: i128) -> MergedIvfSubsection {
        let mut centroids = vec![0.0f32; N_CENT * DIM];
        for c in 0..N_CENT {
            centroids[c * DIM + c] = 1.0;
        }
        let mut vectors = Vec::with_capacity(ROWS * DIM);
        for r in 0..ROWS {
            let mut row = [0.0f32; DIM];
            row[0] = 1.0;
            row[4 + r % 4] = 0.05 + r as f32 * 0.01;
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            vectors.extend(row.iter().map(|v| v / norm));
        }
        let ids: Vec<i128> = (0..ROWS as i128).map(|i| id_base + i).collect();
        let cfg = VectorConfig {
            column: "emb".into(),
            dim: DIM,
            rot_seed: 7,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq16,
            provided_centroids: Some(Arc::from(centroids)),
        };
        build_merged_subsection_from_fp32(cfg, N_CENT, Arc::new(vectors), &ids)
            .expect("Sq16 cell build")
    }

    /// Sq16 multi-batch splice: a cell whose fragments span two drain batches
    /// goes `merge_fragment_subsections` → `sq8_ivf_merge_input_from_subsection`
    /// → `splice_fragments_into_cell`. All three must treat Sq16's codec_meta as
    /// norms-only (no per-cluster scale/offset). Before the read-back fix, the
    /// input parser decoded the norm table as scale/offset — erroring on a small
    /// cell or silently corrupting the reranker on a large one.
    #[test]
    fn sq16_multi_batch_splice_round_trips() {
        let left = sq16_subsection_with_empty_clusters(1_000);
        let right = sq16_subsection_with_empty_clusters(2_000);
        assert_eq!(left.rerank_codec, RerankCodec::Sq16);

        // Read-back must parse norms-only → empty scale/offset (not the norm
        // table misread as a quantizer).
        let inp = sq8_ivf_merge_input_from_subsection(
            &left.bytes,
            DIM,
            left.n_cent,
            left.n_docs,
            Metric::Cosine,
            RerankCodec::Sq16,
            None,
        )
        .expect("Sq16 read-back parses norms-only codec_meta");
        assert!(
            inp.scale.is_empty() && inp.offset.is_empty(),
            "Sq16 carries no per-cluster scale/offset"
        );

        let left_ids: Vec<i128> = (0..ROWS as i128).map(|i| 1_000 + i).collect();
        let right_ids: Vec<i128> = (0..ROWS as i128).map(|i| 2_000 + i).collect();
        let (merged, ids) =
            merge_fragment_subsections(&left, &left_ids, &right, &right_ids, DIM, Metric::Cosine)
                .expect("Sq16 multi-batch splice merge");
        assert_eq!(merged.rerank_codec, RerankCodec::Sq16);
        assert_eq!(
            merged.n_docs as usize,
            2 * ROWS,
            "spliced cell holds every doc from both batches"
        );
        assert_eq!(ids.len(), 2 * ROWS, "one stable id per spliced doc");
    }

    /// The genuine build-path guard for the 16-bit-vs-8-bit fit bug: build the
    /// SAME corpus through the real `Sq16Adaptive` and `Sq8Residual` build/fit
    /// paths and read back each fitted per-cluster ruler. The only difference
    /// must be the code max — `Sq16Adaptive` fits `scale = span/65535`,
    /// `Sq8Residual` fits `span/255` — so the per-dim ratio is exactly
    /// `255/65535`. Before the fix both used `span/255` (ratio 1.0), i.e.
    /// `Sq16Adaptive` was silently an 8-bit quantizer. Rotation-agnostic: both
    /// codecs fit the same rotated min/max, so the raw span never enters.
    #[test]
    fn sq16_adaptive_fit_uses_full_u16_range_not_8bit() {
        const D: usize = 8;
        const NR: usize = 64;
        // One-centroid grid so every row lands in cluster 0; wide, varied
        // per-dim values so the fitted span is comfortably non-degenerate.
        let centroids = vec![0.0f32; D];
        let mut vectors = Vec::with_capacity(NR * D);
        for r in 0..NR {
            for d in 0..D {
                vectors.push(((r * 7 + d * 13) % 97) as f32 - 40.0);
            }
        }
        let ids: Vec<i128> = (0..NR as i128).collect();
        let fitted_scale = |codec: RerankCodec| -> Vec<f32> {
            let cfg = VectorConfig {
                column: "emb".into(),
                dim: D,
                rot_seed: 7,
                metric: Metric::L2Sq,
                rerank_codec: codec,
                provided_centroids: Some(Arc::from(centroids.clone())),
            };
            let sub = build_merged_subsection_from_fp32(cfg, 1, Arc::new(vectors.clone()), &ids)
                .expect("cell build");
            sq8_ivf_merge_input_from_subsection(
                &sub.bytes,
                D,
                1,
                sub.n_docs,
                Metric::L2Sq,
                codec,
                None,
            )
            .expect("parse fitted ruler")
            .scale
        };
        let s8 = fitted_scale(RerankCodec::Sq8Residual);
        let s16 = fitted_scale(RerankCodec::Sq16Adaptive);
        let expected = 255.0f32 / 65535.0;
        for d in 0..D {
            if s8[d] > 0.0 {
                let ratio = s16[d] / s8[d];
                assert!(
                    (ratio - expected).abs() <= expected * 0.05,
                    "dim {d}: Sq16Adaptive ruler must be span/65535 — {expected}x Sq8Residual's \
                     span/255 — got ratio {ratio} (1.0 ⇒ Sq16Adaptive is secretly 8-bit)"
                );
            }
        }
    }

    /// Splice-merging inputs that share an all-empty cluster must leave the
    /// pinned scale/offset constants in that cluster's codec-meta slots: the
    /// open-time validator requires every slot — populated or empty — to be
    /// bitwise-equal to the pinned quantizer, exactly as the direct build
    /// paths write it. The placeholder-seeded merge previously left 1.0/0.0
    /// in all-empty slots, and compaction's own summary open rejected the
    /// merged superfile (first seen at 10M docs / 64 cells, where per-cell
    /// fine k-means is the first shape to produce empty fine clusters).
    #[test]
    fn splice_merge_keeps_pinned_meta_in_empty_clusters() {
        let a = fixed_subsection_with_empty_clusters(1_000);
        let b = fixed_subsection_with_empty_clusters(2_000);
        let parse = |sub: &MergedIvfSubsection| {
            sq8_ivf_merge_input_from_subsection(
                &sub.bytes,
                DIM,
                sub.n_cent,
                sub.n_docs,
                Metric::Cosine,
                RerankCodec::Sq8FixedResidual,
                None,
            )
            .expect("parse merge input")
        };
        let inputs = [parse(&a), parse(&b)];
        let empty_everywhere = (0..N_CENT).any(|c| {
            inputs
                .iter()
                .all(|inp| cluster_entry(&inp.sub, inp.cluster_idx_off, c).1 == 0)
        });
        assert!(
            empty_everywhere,
            "fixture must produce an all-empty cluster"
        );

        let merged = merge_sq8_ivf_subsections_from_parsed(&inputs).expect("splice merge");
        let so_bytes = merged.n_cent * DIM * 4;
        let meta = &merged.bytes
            [merged.codec_meta_offset_in_sub..merged.codec_meta_offset_in_sub + 2 * so_bytes];
        let scale = decode_f32_le_vec(&meta[..so_bytes]);
        let offset = decode_f32_le_vec(&meta[so_bytes..]);
        for (i, value) in scale.iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                SQ8_FIXED_SCALE.to_bits(),
                "scale slot {i} (cluster {}) must stay pinned",
                i / DIM
            );
        }
        for (i, value) in offset.iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                SQ8_FIXED_OFFSET.to_bits(),
                "offset slot {i} (cluster {}) must stay pinned",
                i / DIM
            );
        }
    }

    /// A fragment doc id is stored bytes, not a trusted slice index. Corrupting
    /// it past the stable-id table must return a typed build error instead of
    /// panicking the drain worker.
    #[test]
    fn splice_fragment_rejects_doc_id_past_stable_ids() {
        let subsection = fixed_subsection_with_empty_clusters(1_000);
        let stable_ids: Vec<i128> = (0..ROWS as i128).map(|id| 1_000 + id).collect();
        let mut input = sq8_ivf_merge_input_from_subsection(
            &subsection.bytes,
            DIM,
            subsection.n_cent,
            subsection.n_docs,
            Metric::Cosine,
            RerankCodec::Sq8FixedResidual,
            None,
        )
        .expect("parse merge input");
        let (cluster, doc_off, count) = (0..input.n_cent)
            .find_map(|cluster| {
                let (doc_off, count) = cluster_entry(&input.sub, input.cluster_idx_off, cluster);
                (count > 0).then_some((cluster, doc_off, count))
            })
            .expect("fixture has a populated cluster");
        let block = input.per_cluster_blocks_off + doc_off * input.stride;
        let first_doc_id = block + count * input.code_bytes;
        input.sub[first_doc_id..first_doc_id + DOC_ID_BYTES]
            .copy_from_slice(&(ROWS as u32).to_le_bytes());

        let fragments = [(&input, cluster, stable_ids.as_slice())];
        let error = match splice_fragments_into_cell(&fragments) {
            Err(error) => error,
            Ok(_) => panic!("out-of-range stored doc id must fail"),
        };
        assert!(
            matches!(
                &error,
                BuildError::VectorSchemaMismatch(message)
                    if message.contains("doc-id out of range")
            ),
            "unexpected error: {error}"
        );
    }
}
