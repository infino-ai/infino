// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use bytes::Bytes;
use infino::superfile::vector::builder::{VectorBuilder, VectorConfig};
use infino::superfile::vector::distance::Metric;
use infino::superfile::vector::reader::{OpenOptions, VectorReader};
use infino::superfile::vector::rerank_codec::RerankCodec;
use infino_bench_utils::corpus;

const N_DOCS: usize = 1_000_000;
const TOP_K: usize = 10;
const N_QUERIES: usize = 100;
const QUERY_SEED: u64 = 99;
const QUERY_SIGMA: f32 = 0.05;

/// Shift between an `f32` and its high 16 bits (the bf16 payload).
const F32_HI16_SHIFT: u32 = 16;
/// `f32` absolute-value mask (clears the sign bit) for NaN detection.
const F32_ABS_MASK: u32 = 0x7FFF_FFFF;
/// `f32` bit pattern of +∞; anything greater (abs) is NaN.
const F32_INF_BITS: u32 = 0x7F80_0000;
/// Quiet-NaN payload bit set in the bf16 result so NaN stays NaN.
const BF16_QUIET_NAN_BIT: u32 = 0x0040;
/// Round-to-nearest-even bias added before truncating to bf16.
const BF16_ROUND_BIAS: u32 = 0x7FFF;

fn fp32_to_bf16_debug(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & F32_ABS_MASK) > F32_INF_BITS {
        ((bits >> F32_HI16_SHIFT) | BF16_QUIET_NAN_BIT) as u16
    } else {
        let lsb = (bits >> F32_HI16_SHIFT) & 1;
        let bias = BF16_ROUND_BIAS + lsb;
        (bits.wrapping_add(bias) >> F32_HI16_SHIFT) as u16
    }
}

fn bf16_to_f32_debug(bf: u16) -> f32 {
    f32::from_bits((bf as u32) << F32_HI16_SHIFT)
}

fn read_u32_le(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn read_u64_le(bytes: &[u8], off: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[off..off + 8]);
    u64::from_le_bytes(buf)
}

fn read_f32_le(bytes: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

#[derive(Clone)]
struct IvfLayout {
    dim: usize,
    n_cent: usize,
    n_docs: usize,
    centroids: Vec<f32>,
    cluster_index: Vec<(u32, u32)>,
    doc_ids_by_pos: Vec<u32>,
}

fn parse_ivf_layout_from_fp32_blob(blob: &[u8]) -> IvfLayout {
    let n_docs = read_u64_le(blob, 16) as usize;
    let dir_off = read_u64_le(blob, 24) as usize;
    let dim = read_u32_le(blob, dir_off + 4) as usize;
    let n_cent = read_u32_le(blob, dir_off + 8) as usize;
    let sub_off = read_u64_le(blob, dir_off + 24) as usize;
    let sub_len = read_u64_le(blob, dir_off + 32) as usize;
    let sub = &blob[sub_off..sub_off + sub_len];
    let centroids_off = read_u64_le(sub, 32) as usize;
    let cluster_idx_off = read_u64_le(sub, 40) as usize;
    let full_off = read_u32_le(sub, 52) as usize;
    let doc_ids_off = full_off + n_docs * dim * 4;

    let mut centroids = Vec::with_capacity(n_cent * dim);
    for i in 0..n_cent * dim {
        centroids.push(read_f32_le(sub, centroids_off + i * 4));
    }

    let mut cluster_index = Vec::with_capacity(n_cent);
    for c in 0..n_cent {
        let base = cluster_idx_off + c * 8;
        cluster_index.push((read_u32_le(sub, base), read_u32_le(sub, base + 4)));
    }

    let mut doc_ids_by_pos = Vec::with_capacity(n_docs);
    for i in 0..n_docs {
        doc_ids_by_pos.push(read_u32_le(sub, doc_ids_off + i * 4));
    }

    IvfLayout {
        dim,
        n_cent,
        n_docs,
        centroids,
        cluster_index,
        doc_ids_by_pos,
    }
}

fn build_blob(vectors: &[f32], codec: RerankCodec) -> Vec<u8> {
    let n_cent = corpus::n_cent(N_DOCS);
    let mut b = VectorBuilder::new();
    b.register_column(VectorConfig {
        column: "v".into(),
        dim: corpus::DIM,
        n_cent,
        rot_seed: 7,
        metric: Metric::Cosine,
        rerank_codec: codec,
    })
    .expect("register column");
    for i in 0..N_DOCS {
        let off = i * corpus::DIM;
        b.add(0, &vectors[off..off + corpus::DIM])
            .expect("add vector");
    }
    b.finish().expect("finish vector builder")
}

fn open_reader_from_blob(blob: Vec<u8>) -> VectorReader {
    let n_cent = corpus::n_cent(N_DOCS);
    let json = format!(
        r#"[{{"column":"v","dim":{},"n_cent":{},"rot_seed":7,"metric":"cosine"}}]"#,
        corpus::DIM,
        n_cent
    );
    VectorReader::open_with(Bytes::from(blob), &json, OpenOptions { verify_crc: true })
        .expect("open reader")
}

fn build_reader(vectors: &[f32], codec: RerankCodec) -> VectorReader {
    open_reader_from_blob(build_blob(vectors, codec))
}

async fn search_async(
    reader: &VectorReader,
    query: &[f32],
    k: usize,
    nprobe: usize,
    rerank_mult: usize,
) -> Vec<(u32, f32)> {
    reader
        .search("v", query, k, nprobe, rerank_mult)
        .await
        .expect("search")
}

async fn recall_ids(
    reader: &VectorReader,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
    nprobe: usize,
    rerank_mult: usize,
) -> f32 {
    let mut total = 0f32;
    for (q, truth) in queries.iter().zip(truths) {
        let hits = search_async(reader, q, TOP_K, nprobe, rerank_mult).await;
        total += corpus::recall_at_k(&hits, truth);
    }
    total / queries.len() as f32
}

async fn hit_count(
    reader: &VectorReader,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
    nprobe: usize,
    rerank_mult: usize,
) -> usize {
    let mut hits_total = 0usize;
    for (q, truth) in queries.iter().zip(truths) {
        let hits = search_async(reader, q, TOP_K, nprobe, rerank_mult).await;
        let truth_set: std::collections::HashSet<u32> = truth.iter().copied().collect();
        hits_total += hits.iter().filter(|(id, _)| truth_set.contains(id)).count();
    }
    hits_total
}

fn make_householder_to_e0(x: &[f32]) -> (Vec<f32>, f32) {
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        return (vec![0.0; x.len()], 0.0);
    }
    let mut v: Vec<f32> = x.iter().map(|&a| a / norm).collect();
    v[0] -= 1.0;
    let vv = v.iter().map(|a| a * a).sum::<f32>();
    if vv <= 1e-12 { (v, 0.0) } else { (v, 2.0 / vv) }
}

fn apply_householder(v: &[f32], beta: f32, x: &[f32], out: &mut [f32]) {
    if beta == 0.0 {
        out.copy_from_slice(x);
        return;
    }
    let vx = v.iter().zip(x).map(|(a, b)| a * b).sum::<f32>();
    for d in 0..x.len() {
        out[d] = x[d] - beta * v[d] * vx;
    }
}

struct LocalHouseholderSq8 {
    dim: usize,
    n_docs: usize,
    scales: Vec<f32>,
    offsets: Vec<f32>,
    codes: Vec<u8>,
    norms: Vec<f32>,
    house_v: Vec<f32>,
    house_beta: Vec<f32>,
    cluster_index: Vec<(u32, u32)>,
    doc_ids_by_pos: Vec<u32>,
}

fn build_local_householder_sq8(vectors: &[f32], layout: &IvfLayout) -> LocalHouseholderSq8 {
    let dim = layout.dim;
    let n_cent = layout.n_cent;
    let n_docs = layout.n_docs;
    let mut house_v = Vec::with_capacity(n_cent * dim);
    let mut house_beta = Vec::with_capacity(n_cent);
    for c in 0..n_cent {
        let centroid = &layout.centroids[c * dim..(c + 1) * dim];
        let (v, beta) = make_householder_to_e0(centroid);
        house_v.extend_from_slice(&v);
        house_beta.push(beta);
    }

    let mut mins = vec![f32::INFINITY; n_cent * dim];
    let mut maxs = vec![f32::NEG_INFINITY; n_cent * dim];
    let mut y = vec![0.0f32; dim];

    for c in 0..n_cent {
        let (off, cnt) = layout.cluster_index[c];
        let hv = &house_v[c * dim..(c + 1) * dim];
        let beta = house_beta[c];
        for pos in off as usize..(off + cnt) as usize {
            let doc_id = layout.doc_ids_by_pos[pos] as usize;
            let row = &vectors[doc_id * dim..(doc_id + 1) * dim];
            apply_householder(hv, beta, row, &mut y);
            for (d, &yd) in y.iter().enumerate().take(dim) {
                let idx = c * dim + d;
                mins[idx] = mins[idx].min(yd);
                maxs[idx] = maxs[idx].max(yd);
            }
        }
    }

    let mut scales = vec![0.0f32; n_cent * dim];
    let mut offsets = vec![0.0f32; n_cent * dim];
    for i in 0..n_cent * dim {
        offsets[i] = if mins[i].is_finite() { mins[i] } else { 0.0 };
        scales[i] = if maxs[i].is_finite() && maxs[i] > mins[i] {
            (maxs[i] - mins[i]) / 255.0
        } else {
            1.0
        };
    }

    let mut codes = vec![0u8; n_docs * dim];
    let mut norms = vec![0.0f32; n_docs];
    for c in 0..n_cent {
        let (off, cnt) = layout.cluster_index[c];
        let hv = &house_v[c * dim..(c + 1) * dim];
        let beta = house_beta[c];
        let scale_c = &scales[c * dim..(c + 1) * dim];
        let offset_c = &offsets[c * dim..(c + 1) * dim];
        for pos in off as usize..(off + cnt) as usize {
            let doc_id = layout.doc_ids_by_pos[pos] as usize;
            let row = &vectors[doc_id * dim..(doc_id + 1) * dim];
            apply_householder(hv, beta, row, &mut y);
            let code = &mut codes[pos * dim..(pos + 1) * dim];
            let mut norm_sq = 0.0f64;
            for d in 0..dim {
                let q = ((y[d] - offset_c[d]) / scale_c[d])
                    .round()
                    .clamp(0.0, 255.0) as u8;
                code[d] = q;
                let dec = (q as f32) * scale_c[d] + offset_c[d];
                norm_sq += (dec as f64) * (dec as f64);
            }
            norms[pos] = norm_sq as f32;
        }
    }

    LocalHouseholderSq8 {
        dim,
        n_docs,
        scales,
        offsets,
        codes,
        norms,
        house_v,
        house_beta,
        cluster_index: layout.cluster_index.clone(),
        doc_ids_by_pos: layout.doc_ids_by_pos.clone(),
    }
}

fn push_topk(top: &mut Vec<(u32, f32)>, did: u32, dist: f32) {
    if top.len() < TOP_K {
        top.push((did, dist));
        return;
    }
    let mut worst = 0usize;
    for i in 1..top.len() {
        if top[i].1 > top[worst].1 {
            worst = i;
        }
    }
    if dist < top[worst].1 {
        top[worst] = (did, dist);
    }
}

fn local_householder_topk(index: &LocalHouseholderSq8, query: &[f32]) -> Vec<u32> {
    let dim = index.dim;
    let mut top = Vec::with_capacity(TOP_K);
    let mut qy = vec![0.0f32; dim];
    for c in 0..index.cluster_index.len() {
        let (off, cnt) = index.cluster_index[c];
        let hv = &index.house_v[c * dim..(c + 1) * dim];
        let beta = index.house_beta[c];
        apply_householder(hv, beta, query, &mut qy);
        let scale_c = &index.scales[c * dim..(c + 1) * dim];
        let offset_c = &index.offsets[c * dim..(c + 1) * dim];
        for pos in off as usize..(off + cnt) as usize {
            let code = &index.codes[pos * dim..(pos + 1) * dim];
            let mut dot = 0.0f32;
            for d in 0..dim {
                let dec = (code[d] as f32) * scale_c[d] + offset_c[d];
                dot += qy[d] * dec;
            }
            let norm = index.norms[pos].sqrt();
            let dist = if norm > 0.0 {
                1.0 - dot / norm
            } else {
                1.0 - dot
            };
            push_topk(&mut top, index.doc_ids_by_pos[pos], dist);
        }
    }
    top.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    top.into_iter().map(|(id, _)| id).collect()
}

fn local_householder_recall(
    index: &LocalHouseholderSq8,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
) -> (usize, f32) {
    let mut hits_total = 0usize;
    for (q, truth) in queries.iter().zip(truths) {
        let pred = local_householder_topk(index, q);
        let truth_set: std::collections::HashSet<u32> = truth.iter().copied().collect();
        hits_total += pred.iter().filter(|id| truth_set.contains(id)).count();
    }
    (
        hits_total,
        hits_total as f32 / (queries.len() * TOP_K) as f32,
    )
}

struct LeastSquaresSq8 {
    dim: usize,
    n_docs: usize,
    scales: Vec<f32>,
    offsets: Vec<f32>,
    codes: Vec<u8>,
    norms: Vec<f32>,
    cluster_index: Vec<(u32, u32)>,
    doc_ids_by_pos: Vec<u32>,
}

fn build_least_squares_sq8(vectors: &[f32], layout: &IvfLayout) -> LeastSquaresSq8 {
    let dim = layout.dim;
    let n_cent = layout.n_cent;
    let n_docs = layout.n_docs;

    let mut min_v = vec![f32::INFINITY; n_cent * dim];
    let mut max_v = vec![f32::NEG_INFINITY; n_cent * dim];
    for c in 0..n_cent {
        let (off, cnt) = layout.cluster_index[c];
        for pos in off as usize..(off + cnt) as usize {
            let doc_id = layout.doc_ids_by_pos[pos] as usize;
            let row = &vectors[doc_id * dim..(doc_id + 1) * dim];
            for (d, &rd) in row.iter().enumerate().take(dim) {
                let idx = c * dim + d;
                min_v[idx] = min_v[idx].min(rd);
                max_v[idx] = max_v[idx].max(rd);
            }
        }
    }

    let mut initial_scales = vec![1.0f32; n_cent * dim];
    let mut initial_offsets = vec![0.0f32; n_cent * dim];
    for i in 0..n_cent * dim {
        if min_v[i].is_finite() {
            initial_offsets[i] = min_v[i];
            if max_v[i] > min_v[i] {
                initial_scales[i] = (max_v[i] - min_v[i]) / 255.0;
            }
        }
    }

    let mut codes = vec![0u8; n_docs * dim];
    let mut count = vec![0u64; n_cent * dim];
    let mut sum_q = vec![0.0f64; n_cent * dim];
    let mut sum_x = vec![0.0f64; n_cent * dim];
    let mut sum_qq = vec![0.0f64; n_cent * dim];
    let mut sum_qx = vec![0.0f64; n_cent * dim];

    for c in 0..n_cent {
        let (off, cnt) = layout.cluster_index[c];
        let scale0 = &initial_scales[c * dim..(c + 1) * dim];
        let offset0 = &initial_offsets[c * dim..(c + 1) * dim];
        for pos in off as usize..(off + cnt) as usize {
            let doc_id = layout.doc_ids_by_pos[pos] as usize;
            let row = &vectors[doc_id * dim..(doc_id + 1) * dim];
            let code = &mut codes[pos * dim..(pos + 1) * dim];
            for d in 0..dim {
                let q = ((row[d] - offset0[d]) / scale0[d])
                    .round()
                    .clamp(0.0, 255.0) as u8;
                code[d] = q;
                let idx = c * dim + d;
                let qf = q as f64;
                let xf = row[d] as f64;
                count[idx] += 1;
                sum_q[idx] += qf;
                sum_x[idx] += xf;
                sum_qq[idx] += qf * qf;
                sum_qx[idx] += qf * xf;
            }
        }
    }

    let mut scales = vec![1.0f32; n_cent * dim];
    let mut offsets = vec![0.0f32; n_cent * dim];
    for idx in 0..n_cent * dim {
        let n = count[idx] as f64;
        if n == 0.0 {
            continue;
        }
        let denom = n * sum_qq[idx] - sum_q[idx] * sum_q[idx];
        if denom.abs() > 1e-12 {
            let a = (n * sum_qx[idx] - sum_q[idx] * sum_x[idx]) / denom;
            let b = (sum_x[idx] - a * sum_q[idx]) / n;
            if a.is_finite() && b.is_finite() && a > 0.0 {
                scales[idx] = a as f32;
                offsets[idx] = b as f32;
                continue;
            }
        }
        scales[idx] = initial_scales[idx];
        offsets[idx] = initial_offsets[idx];
    }

    let mut norms = vec![0.0f32; n_docs];
    for c in 0..n_cent {
        let (off, cnt) = layout.cluster_index[c];
        let scale_c = &scales[c * dim..(c + 1) * dim];
        let offset_c = &offsets[c * dim..(c + 1) * dim];
        for pos in off as usize..(off + cnt) as usize {
            let code = &codes[pos * dim..(pos + 1) * dim];
            let mut norm_sq = 0.0f64;
            for d in 0..dim {
                let dec = (code[d] as f32) * scale_c[d] + offset_c[d];
                norm_sq += (dec as f64) * (dec as f64);
            }
            norms[pos] = norm_sq as f32;
        }
    }

    LeastSquaresSq8 {
        dim,
        n_docs,
        scales,
        offsets,
        codes,
        norms,
        cluster_index: layout.cluster_index.clone(),
        doc_ids_by_pos: layout.doc_ids_by_pos.clone(),
    }
}

fn least_squares_topk(index: &LeastSquaresSq8, query: &[f32]) -> Vec<u32> {
    let dim = index.dim;
    let mut top = Vec::with_capacity(TOP_K);
    for c in 0..index.cluster_index.len() {
        let (off, cnt) = index.cluster_index[c];
        let scale_c = &index.scales[c * dim..(c + 1) * dim];
        let offset_c = &index.offsets[c * dim..(c + 1) * dim];
        for pos in off as usize..(off + cnt) as usize {
            let code = &index.codes[pos * dim..(pos + 1) * dim];
            let mut dot = 0.0f32;
            for d in 0..dim {
                let dec = (code[d] as f32) * scale_c[d] + offset_c[d];
                dot += query[d] * dec;
            }
            let norm = index.norms[pos].sqrt();
            let dist = if norm > 0.0 {
                1.0 - dot / norm
            } else {
                1.0 - dot
            };
            push_topk(&mut top, index.doc_ids_by_pos[pos], dist);
        }
    }
    top.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    top.into_iter().map(|(id, _)| id).collect()
}

fn least_squares_recall(
    index: &LeastSquaresSq8,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
) -> (usize, f32) {
    let mut hits_total = 0usize;
    for (q, truth) in queries.iter().zip(truths) {
        let pred = least_squares_topk(index, q);
        let truth_set: std::collections::HashSet<u32> = truth.iter().copied().collect();
        hits_total += pred.iter().filter(|id| truth_set.contains(id)).count();
    }
    (
        hits_total,
        hits_total as f32 / (queries.len() * TOP_K) as f32,
    )
}

fn hadamard_sign(i: usize, seed: u64) -> f32 {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    if (x & 1) == 0 { 1.0 } else { -1.0 }
}

fn apply_block_hadamard<const B: usize>(x: &[f32], out: &mut [f32], seed: u64) {
    debug_assert!(B.is_power_of_two());
    debug_assert_eq!(x.len(), out.len());
    debug_assert_eq!(x.len() % B, 0);
    let inv = 1.0 / (B as f32).sqrt();
    for block_start in (0..x.len()).step_by(B) {
        for j in 0..B {
            let idx = block_start + j;
            out[idx] = x[idx] * hadamard_sign(idx, seed);
        }
        let mut h = 1usize;
        while h < B {
            let step = h * 2;
            for base in (block_start..block_start + B).step_by(step) {
                for j in 0..h {
                    let a = out[base + j];
                    let b = out[base + j + h];
                    out[base + j] = a + b;
                    out[base + j + h] = a - b;
                }
            }
            h = step;
        }
        for j in 0..B {
            out[block_start + j] *= inv;
        }
    }
}

struct BlockHadamardSq8<const B: usize> {
    dim: usize,
    n_docs: usize,
    scales: Vec<f32>,
    offsets: Vec<f32>,
    codes: Vec<u8>,
    norms: Vec<f32>,
    cluster_index: Vec<(u32, u32)>,
    doc_ids_by_pos: Vec<u32>,
    seed: u64,
}

fn build_block_hadamard_sq8<const B: usize>(
    vectors: &[f32],
    layout: &IvfLayout,
    seed: u64,
) -> BlockHadamardSq8<B> {
    let dim = layout.dim;
    let n_cent = layout.n_cent;
    let n_docs = layout.n_docs;
    let mut min_v = vec![f32::INFINITY; n_cent * dim];
    let mut max_v = vec![f32::NEG_INFINITY; n_cent * dim];
    let mut y = vec![0.0f32; dim];

    for c in 0..n_cent {
        let (off, cnt) = layout.cluster_index[c];
        for pos in off as usize..(off + cnt) as usize {
            let doc_id = layout.doc_ids_by_pos[pos] as usize;
            let row = &vectors[doc_id * dim..(doc_id + 1) * dim];
            apply_block_hadamard::<B>(row, &mut y, seed);
            for (d, &yd) in y.iter().enumerate().take(dim) {
                let idx = c * dim + d;
                min_v[idx] = min_v[idx].min(yd);
                max_v[idx] = max_v[idx].max(yd);
            }
        }
    }

    let mut scales = vec![1.0f32; n_cent * dim];
    let mut offsets = vec![0.0f32; n_cent * dim];
    for i in 0..n_cent * dim {
        if min_v[i].is_finite() {
            offsets[i] = min_v[i];
            if max_v[i] > min_v[i] {
                scales[i] = (max_v[i] - min_v[i]) / 255.0;
            }
        }
    }

    let mut codes = vec![0u8; n_docs * dim];
    let mut norms = vec![0.0f32; n_docs];
    for c in 0..n_cent {
        let (off, cnt) = layout.cluster_index[c];
        let scale_c = &scales[c * dim..(c + 1) * dim];
        let offset_c = &offsets[c * dim..(c + 1) * dim];
        for pos in off as usize..(off + cnt) as usize {
            let doc_id = layout.doc_ids_by_pos[pos] as usize;
            let row = &vectors[doc_id * dim..(doc_id + 1) * dim];
            apply_block_hadamard::<B>(row, &mut y, seed);
            let code = &mut codes[pos * dim..(pos + 1) * dim];
            let mut norm_sq = 0.0f64;
            for d in 0..dim {
                let q = ((y[d] - offset_c[d]) / scale_c[d])
                    .round()
                    .clamp(0.0, 255.0) as u8;
                code[d] = q;
                let dec = (q as f32) * scale_c[d] + offset_c[d];
                norm_sq += (dec as f64) * (dec as f64);
            }
            norms[pos] = norm_sq as f32;
        }
    }

    BlockHadamardSq8 {
        dim,
        n_docs,
        scales,
        offsets,
        codes,
        norms,
        cluster_index: layout.cluster_index.clone(),
        doc_ids_by_pos: layout.doc_ids_by_pos.clone(),
        seed,
    }
}

fn block_hadamard_topk<const B: usize>(index: &BlockHadamardSq8<B>, query: &[f32]) -> Vec<u32> {
    let dim = index.dim;
    let mut qy = vec![0.0f32; dim];
    apply_block_hadamard::<B>(query, &mut qy, index.seed);
    let mut top = Vec::with_capacity(TOP_K);
    for c in 0..index.cluster_index.len() {
        let (off, cnt) = index.cluster_index[c];
        let scale_c = &index.scales[c * dim..(c + 1) * dim];
        let offset_c = &index.offsets[c * dim..(c + 1) * dim];
        for pos in off as usize..(off + cnt) as usize {
            let code = &index.codes[pos * dim..(pos + 1) * dim];
            let mut dot = 0.0f32;
            for d in 0..dim {
                let dec = (code[d] as f32) * scale_c[d] + offset_c[d];
                dot += qy[d] * dec;
            }
            let norm = index.norms[pos].sqrt();
            let dist = if norm > 0.0 {
                1.0 - dot / norm
            } else {
                1.0 - dot
            };
            push_topk(&mut top, index.doc_ids_by_pos[pos], dist);
        }
    }
    top.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    top.into_iter().map(|(id, _)| id).collect()
}

fn block_hadamard_recall<const B: usize>(
    index: &BlockHadamardSq8<B>,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
) -> (usize, f32) {
    let mut hits_total = 0usize;
    for (q, truth) in queries.iter().zip(truths) {
        let pred = block_hadamard_topk(index, q);
        let truth_set: std::collections::HashSet<u32> = truth.iter().copied().collect();
        hits_total += pred.iter().filter(|id| truth_set.contains(id)).count();
    }
    (
        hits_total,
        hits_total as f32 / (queries.len() * TOP_K) as f32,
    )
}

struct ClippedSigmaSq8 {
    dim: usize,
    n_docs: usize,
    scales: Vec<f32>,
    offsets: Vec<f32>,
    codes: Vec<u8>,
    norms: Vec<f32>,
    cluster_index: Vec<(u32, u32)>,
    doc_ids_by_pos: Vec<u32>,
    sigma: f32,
}

fn build_clipped_sigma_sq8(vectors: &[f32], layout: &IvfLayout, sigma: f32) -> ClippedSigmaSq8 {
    let dim = layout.dim;
    let n_cent = layout.n_cent;
    let n_docs = layout.n_docs;
    let mut count = vec![0u64; n_cent * dim];
    let mut sum = vec![0.0f64; n_cent * dim];
    let mut sum_sq = vec![0.0f64; n_cent * dim];

    for c in 0..n_cent {
        let (off, cnt) = layout.cluster_index[c];
        for pos in off as usize..(off + cnt) as usize {
            let doc_id = layout.doc_ids_by_pos[pos] as usize;
            let row = &vectors[doc_id * dim..(doc_id + 1) * dim];
            for (d, &rd) in row.iter().enumerate().take(dim) {
                let idx = c * dim + d;
                let x = rd as f64;
                count[idx] += 1;
                sum[idx] += x;
                sum_sq[idx] += x * x;
            }
        }
    }

    let mut scales = vec![1.0f32; n_cent * dim];
    let mut offsets = vec![0.0f32; n_cent * dim];
    for idx in 0..n_cent * dim {
        let n = count[idx] as f64;
        if n == 0.0 {
            continue;
        }
        let mean = sum[idx] / n;
        let var = (sum_sq[idx] / n - mean * mean).max(0.0);
        let std = var.sqrt();
        let lo = mean - (sigma as f64) * std;
        let hi = mean + (sigma as f64) * std;
        offsets[idx] = lo as f32;
        scales[idx] = if hi > lo {
            ((hi - lo) / 255.0) as f32
        } else {
            1.0
        };
    }

    let mut codes = vec![0u8; n_docs * dim];
    let mut norms = vec![0.0f32; n_docs];
    for c in 0..n_cent {
        let (off, cnt) = layout.cluster_index[c];
        let scale_c = &scales[c * dim..(c + 1) * dim];
        let offset_c = &offsets[c * dim..(c + 1) * dim];
        for pos in off as usize..(off + cnt) as usize {
            let doc_id = layout.doc_ids_by_pos[pos] as usize;
            let row = &vectors[doc_id * dim..(doc_id + 1) * dim];
            let code = &mut codes[pos * dim..(pos + 1) * dim];
            let mut norm_sq = 0.0f64;
            for d in 0..dim {
                let q = ((row[d] - offset_c[d]) / scale_c[d])
                    .round()
                    .clamp(0.0, 255.0) as u8;
                code[d] = q;
                let dec = (q as f32) * scale_c[d] + offset_c[d];
                norm_sq += (dec as f64) * (dec as f64);
            }
            norms[pos] = norm_sq as f32;
        }
    }

    ClippedSigmaSq8 {
        dim,
        n_docs,
        scales,
        offsets,
        codes,
        norms,
        cluster_index: layout.cluster_index.clone(),
        doc_ids_by_pos: layout.doc_ids_by_pos.clone(),
        sigma,
    }
}

fn clipped_sigma_topk(index: &ClippedSigmaSq8, query: &[f32]) -> Vec<u32> {
    let dim = index.dim;
    let mut top = Vec::with_capacity(TOP_K);
    for c in 0..index.cluster_index.len() {
        let (off, cnt) = index.cluster_index[c];
        let scale_c = &index.scales[c * dim..(c + 1) * dim];
        let offset_c = &index.offsets[c * dim..(c + 1) * dim];
        for pos in off as usize..(off + cnt) as usize {
            let code = &index.codes[pos * dim..(pos + 1) * dim];
            let mut dot = 0.0f32;
            for d in 0..dim {
                let dec = (code[d] as f32) * scale_c[d] + offset_c[d];
                dot += query[d] * dec;
            }
            let norm = index.norms[pos].sqrt();
            let dist = if norm > 0.0 {
                1.0 - dot / norm
            } else {
                1.0 - dot
            };
            push_topk(&mut top, index.doc_ids_by_pos[pos], dist);
        }
    }
    top.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    top.into_iter().map(|(id, _)| id).collect()
}

fn clipped_sigma_recall(
    index: &ClippedSigmaSq8,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
) -> (usize, f32) {
    let mut hits_total = 0usize;
    for (q, truth) in queries.iter().zip(truths) {
        let pred = clipped_sigma_topk(index, q);
        let truth_set: std::collections::HashSet<u32> = truth.iter().copied().collect();
        hits_total += pred.iter().filter(|id| truth_set.contains(id)).count();
    }
    (
        hits_total,
        hits_total as f32 / (queries.len() * TOP_K) as f32,
    )
}

fn print_quant_error_proxy(vectors: &[f32]) {
    let n_cent = corpus::n_cent(N_DOCS);
    let dim = corpus::DIM;
    let mut min_v = vec![f32::INFINITY; n_cent * dim];
    let mut max_v = vec![f32::NEG_INFINITY; n_cent * dim];

    for i in 0..N_DOCS {
        let c = i % n_cent;
        let row = &vectors[i * dim..(i + 1) * dim];
        for (d, &rd) in row.iter().enumerate().take(dim) {
            let idx = c * dim + d;
            min_v[idx] = min_v[idx].min(rd);
            max_v[idx] = max_v[idx].max(rd);
        }
    }

    let mut bf16_sum_sq = 0.0f64;
    let mut bf16_sum_abs = 0.0f64;
    let mut bf16_max_abs = 0.0f32;
    let mut sq8_sum_sq = 0.0f64;
    let mut sq8_sum_abs = 0.0f64;
    let mut sq8_max_abs = 0.0f32;
    let mut sq8_bucket_sum = 0.0f64;
    let mut n = 0usize;

    for i in 0..N_DOCS {
        let c = i % n_cent;
        let row = &vectors[i * dim..(i + 1) * dim];
        for (d, &rd) in row.iter().enumerate().take(dim) {
            let x = rd;
            let xb = bf16_to_f32_debug(fp32_to_bf16_debug(x));
            let be = (xb - x).abs();
            bf16_sum_abs += be as f64;
            bf16_sum_sq += (be as f64) * (be as f64);
            bf16_max_abs = bf16_max_abs.max(be);

            let idx = c * dim + d;
            let lo = min_v[idx];
            let hi = max_v[idx];
            let scale = if hi > lo { (hi - lo) / 255.0 } else { 1.0 };
            let q = ((x - lo) / scale).round().clamp(0.0, 255.0) as u8;
            let xs = (q as f32) * scale + lo;
            let se = (xs - x).abs();
            sq8_sum_abs += se as f64;
            sq8_sum_sq += (se as f64) * (se as f64);
            sq8_max_abs = sq8_max_abs.max(se);
            sq8_bucket_sum += scale as f64;
            n += 1;
        }
    }

    let nf = n as f64;
    eprintln!(
        "debug: component error proxy over {N_DOCS} docs × {dim}: bf16 mean_abs={:.8} rmse={:.8} max_abs={:.8}; planted-cluster-sq8 mean_abs={:.8} rmse={:.8} max_abs={:.8} mean_bucket_width={:.8}",
        bf16_sum_abs / nf,
        (bf16_sum_sq / nf).sqrt(),
        bf16_max_abs,
        sq8_sum_abs / nf,
        (sq8_sum_sq / nf).sqrt(),
        sq8_max_abs,
        sq8_bucket_sum / nf,
    );
}

#[tokio::test]
#[ignore = "1M-doc exploratory residual recall diagnostic; run explicitly"]
async fn compare_fp32_and_sq8_on_1m_bench_workload() {
    eprintln!(
        "debug: generating mmap corpus N_DOCS={N_DOCS}, dim={}, n_cent={}",
        corpus::DIM,
        corpus::n_cent(N_DOCS)
    );
    let corpus_mmap = corpus::MmapVectorCorpus::generate(N_DOCS, corpus::n_cent(N_DOCS), 1, true);
    let vectors = corpus_mmap.as_slice();

    eprintln!(
        "debug: generating {N_QUERIES} realistic queries with seed={QUERY_SEED}, sigma={QUERY_SIGMA}"
    );
    let queries = corpus::generate_realistic_queries(
        vectors,
        N_DOCS,
        N_QUERIES,
        QUERY_SEED,
        true,
        QUERY_SIGMA,
    );

    eprintln!("debug: computing exact fp32 brute-force ground truth");
    let truths = corpus::ground_truth(vectors, N_DOCS, &queries, TOP_K);
    print_quant_error_proxy(vectors);

    let configs = [
        (64usize, 256usize, "correctness config"),
        (800usize, 1024usize, "bench max grid"),
        (1024usize, 1024usize, "all clusters, bench max refine"),
    ];

    eprintln!("\ndebug: building Fp32 blob for reader + IVF layout parse");
    let fp32_blob = build_blob(vectors, RerankCodec::Fp32);
    let layout = parse_ivf_layout_from_fp32_blob(&fp32_blob);
    let fp32_reader = open_reader_from_blob(fp32_blob);
    for (nprobe, rerank_mult, label) in configs {
        let recall = recall_ids(&fp32_reader, &queries, &truths, nprobe, rerank_mult).await;
        let hits = hit_count(&fp32_reader, &queries, &truths, nprobe, rerank_mult).await;
        eprintln!(
            "debug: codec=Fp32 config={label} nprobe={nprobe} rerank_mult={rerank_mult} recall@{TOP_K}={recall:.4} hits={hits}/{}",
            N_QUERIES * TOP_K
        );
    }
    drop(fp32_reader);

    for codec in [RerankCodec::Sq8Residual, RerankCodec::Sq8Residual] {
        eprintln!("\ndebug: building {codec:?} reader");
        let reader = build_reader(vectors, codec);
        for (nprobe, rerank_mult, label) in configs {
            let recall = recall_ids(&reader, &queries, &truths, nprobe, rerank_mult).await;
            let hits = hit_count(&reader, &queries, &truths, nprobe, rerank_mult).await;
            eprintln!(
                "debug: codec={codec:?} config={label} nprobe={nprobe} rerank_mult={rerank_mult} recall@{TOP_K}={recall:.4} hits={hits}/{}",
                N_QUERIES * TOP_K
            );
        }
        drop(reader);
    }

    eprintln!("\ndebug: building local Householder/tangent Sq8 diagnostic index");
    let local = build_local_householder_sq8(vectors, &layout);
    eprintln!(
        "debug: local Householder/tangent Sq8 built: docs={} dim={} bytes_per_doc=1/dim",
        local.n_docs, local.dim
    );
    let (hits, recall) = local_householder_recall(&local, &queries, &truths);
    eprintln!(
        "debug: codec=LocalHouseholderSq8 scoring=all-docs recall@{TOP_K}={recall:.4} hits={hits}/{}",
        N_QUERIES * TOP_K
    );

    eprintln!("\ndebug: building least-squares-decode Sq8 diagnostic index");
    let ls = build_least_squares_sq8(vectors, &layout);
    eprintln!(
        "debug: least-squares-decode Sq8 built: docs={} dim={} bytes_per_doc=1/dim",
        ls.n_docs, ls.dim
    );
    let (hits, recall) = least_squares_recall(&ls, &queries, &truths);
    eprintln!(
        "debug: codec=LeastSquaresDecodeSq8 scoring=all-docs recall@{TOP_K}={recall:.4} hits={hits}/{}",
        N_QUERIES * TOP_K
    );

    eprintln!("\ndebug: building block-Hadamard Sq8 diagnostic indexes");
    let bh16 = build_block_hadamard_sq8::<16>(vectors, &layout, 0xC05E_2026);
    eprintln!(
        "debug: block-Hadamard Sq8 B=16 built: docs={} dim={} bytes_per_doc=1/dim",
        bh16.n_docs, bh16.dim
    );
    let (hits, recall) = block_hadamard_recall::<16>(&bh16, &queries, &truths);
    eprintln!(
        "debug: codec=BlockHadamardSq8[B=16] scoring=all-docs recall@{TOP_K}={recall:.4} hits={hits}/{}",
        N_QUERIES * TOP_K
    );

    let bh64 = build_block_hadamard_sq8::<64>(vectors, &layout, 0xC05E_2026);
    eprintln!(
        "debug: block-Hadamard Sq8 B=64 built: docs={} dim={} bytes_per_doc=1/dim",
        bh64.n_docs, bh64.dim
    );
    let (hits, recall) = block_hadamard_recall::<64>(&bh64, &queries, &truths);
    eprintln!(
        "debug: codec=BlockHadamardSq8[B=64] scoring=all-docs recall@{TOP_K}={recall:.4} hits={hits}/{}",
        N_QUERIES * TOP_K
    );

    eprintln!("\ndebug: building clipped-sigma Sq8 diagnostic indexes");
    for sigma in [3.0_f32, 4.0, 5.0] {
        let clipped = build_clipped_sigma_sq8(vectors, &layout, sigma);
        eprintln!(
            "debug: clipped-sigma Sq8 k={} built: docs={} dim={} bytes_per_doc=1/dim",
            clipped.sigma, clipped.n_docs, clipped.dim
        );
        let (hits, recall) = clipped_sigma_recall(&clipped, &queries, &truths);
        eprintln!(
            "debug: codec=ClippedSigmaSq8[k={}] scoring=all-docs recall@{TOP_K}={recall:.4} hits={hits}/{}",
            clipped.sigma,
            N_QUERIES * TOP_K
        );
    }
}

fn exact_topn_scores(vectors: &[f32], query: &[f32], n: usize) -> Vec<(u32, f32)> {
    let dim = corpus::DIM;
    let mut scored: Vec<(u32, f32)> = (0..N_DOCS as u32)
        .map(|i| {
            let off = (i as usize) * dim;
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += vectors[off + d] * query[d];
            }
            (i, 1.0 - dot)
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n);
    scored
}

fn exact_distance_for_doc(vectors: &[f32], query: &[f32], doc_id: u32) -> f32 {
    let dim = corpus::DIM;
    let off = doc_id as usize * dim;
    let mut dot = 0.0f32;
    for d in 0..dim {
        dot += vectors[off + d] * query[d];
    }
    1.0 - dot
}

#[tokio::test]
#[ignore = "1M-doc exploratory residual recall diagnostic; run explicitly"]
async fn inspect_sq8_miss_geometry_on_1m_bench_workload() {
    eprintln!(
        "miss-debug: generating mmap corpus N_DOCS={N_DOCS}, dim={}, n_cent={}",
        corpus::DIM,
        corpus::n_cent(N_DOCS)
    );
    let corpus_mmap = corpus::MmapVectorCorpus::generate(N_DOCS, corpus::n_cent(N_DOCS), 1, true);
    let vectors = corpus_mmap.as_slice();

    eprintln!(
        "miss-debug: generating {N_QUERIES} realistic queries with seed={QUERY_SEED}, sigma={QUERY_SIGMA}"
    );
    let queries = corpus::generate_realistic_queries(
        vectors,
        N_DOCS,
        N_QUERIES,
        QUERY_SEED,
        true,
        QUERY_SIGMA,
    );

    eprintln!("miss-debug: building Sq8 reader");
    let sq8 = build_reader(vectors, RerankCodec::Sq8Residual);

    let mut total_hits = 0usize;
    let mut missed_queries = 0usize;
    let mut gaps = Vec::with_capacity(N_QUERIES);
    let mut false_deltas = Vec::new();
    let mut missing_deltas = Vec::new();

    for (qi, q) in queries.iter().enumerate() {
        let exact = exact_topn_scores(vectors, q, 20);
        let exact10: Vec<u32> = exact[..TOP_K].iter().map(|(id, _)| *id).collect();
        let truth_set: std::collections::HashSet<u32> = exact10.iter().copied().collect();
        let rank10_dist = exact[TOP_K - 1].1;
        let rank11_dist = exact[TOP_K].1;
        let gap = rank11_dist - rank10_dist;
        gaps.push(gap);

        let pred = search_async(&sq8, q, TOP_K, corpus::n_cent(N_DOCS), 1024).await;
        let pred_ids: Vec<u32> = pred.iter().map(|(id, _)| *id).collect();
        let pred_set: std::collections::HashSet<u32> = pred_ids.iter().copied().collect();
        let hits = pred_ids.iter().filter(|id| truth_set.contains(id)).count();
        total_hits += hits;

        if hits < TOP_K {
            missed_queries += 1;
            let missing: Vec<u32> = exact10
                .iter()
                .copied()
                .filter(|id| !pred_set.contains(id))
                .collect();
            let false_pos: Vec<u32> = pred_ids
                .iter()
                .copied()
                .filter(|id| !truth_set.contains(id))
                .collect();
            eprintln!(
                "miss-debug: q={qi} hits={hits}/{TOP_K} exact_rank10={rank10_dist:.8} rank11={rank11_dist:.8} gap={gap:.8} missing={missing:?} false_pos={false_pos:?}"
            );
            for id in &missing {
                let d = exact_distance_for_doc(vectors, q, *id);
                let delta = d - rank10_dist;
                missing_deltas.push(delta);
                eprintln!("  missing id={id} exact_dist={d:.8} delta_vs_rank10={delta:.8}");
            }
            for id in &false_pos {
                let d = exact_distance_for_doc(vectors, q, *id);
                let delta = d - rank10_dist;
                false_deltas.push(delta);
                let top20_rank = exact.iter().position(|(eid, _)| eid == id).map(|r| r + 1);
                eprintln!(
                    "  false id={id} exact_dist={d:.8} delta_vs_rank10={delta:.8} exact_top20_rank={top20_rank:?}"
                );
            }
        }
    }

    gaps.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    false_deltas.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    missing_deltas.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!(
        "miss-debug: total_hits={total_hits}/{} recall={:.4} missed_queries={missed_queries}/{N_QUERIES}",
        N_QUERIES * TOP_K,
        total_hits as f32 / (N_QUERIES * TOP_K) as f32
    );
    eprintln!(
        "miss-debug: exact rank10->11 gap min={:.8} p50={:.8} p90={:.8} max={:.8}",
        gaps[0],
        gaps[gaps.len() / 2],
        gaps[(gaps.len() * 90 / 100).min(gaps.len() - 1)],
        gaps[gaps.len() - 1]
    );
    if !false_deltas.is_empty() {
        eprintln!(
            "miss-debug: false-positive exact delta_vs_rank10 min={:.8} p50={:.8} p90={:.8} max={:.8}",
            false_deltas[0],
            false_deltas[false_deltas.len() / 2],
            false_deltas[(false_deltas.len() * 90 / 100).min(false_deltas.len() - 1)],
            false_deltas[false_deltas.len() - 1]
        );
    }
    if !missing_deltas.is_empty() {
        eprintln!(
            "miss-debug: missing-truth exact delta_vs_rank10 min={:.8} p50={:.8} p90={:.8} max={:.8}",
            missing_deltas[0],
            missing_deltas[missing_deltas.len() / 2],
            missing_deltas[(missing_deltas.len() * 90 / 100).min(missing_deltas.len() - 1)],
            missing_deltas[missing_deltas.len() - 1]
        );
    }
}

fn exact_rescore_topk(vectors: &[f32], query: &[f32], candidate_ids: &[u32]) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = candidate_ids
        .iter()
        .copied()
        .map(|id| (id, exact_distance_for_doc(vectors, query, id)))
        .collect();
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(TOP_K);
    scored.into_iter().map(|(id, _)| id).collect()
}

#[tokio::test]
#[ignore = "1M-doc exploratory residual recall diagnostic; run explicitly"]
async fn sq8_oversampled_exact_rescore_recall_on_1m_bench_workload() {
    eprintln!(
        "rescore-debug: generating mmap corpus N_DOCS={N_DOCS}, dim={}, n_cent={}",
        corpus::DIM,
        corpus::n_cent(N_DOCS)
    );
    let corpus_mmap = corpus::MmapVectorCorpus::generate(N_DOCS, corpus::n_cent(N_DOCS), 1, true);
    let vectors = corpus_mmap.as_slice();

    eprintln!(
        "rescore-debug: generating {N_QUERIES} realistic queries with seed={QUERY_SEED}, sigma={QUERY_SIGMA}"
    );
    let queries = corpus::generate_realistic_queries(
        vectors,
        N_DOCS,
        N_QUERIES,
        QUERY_SEED,
        true,
        QUERY_SIGMA,
    );

    eprintln!("rescore-debug: computing exact fp32 brute-force ground truth");
    let truths = corpus::ground_truth(vectors, N_DOCS, &queries, TOP_K);

    eprintln!("rescore-debug: building Sq8 reader");
    let sq8 = build_reader(vectors, RerankCodec::Sq8Residual);

    for m in [10usize, 20, 50, 100, 200] {
        let mut sq8_hits = 0usize;
        let mut rescored_hits = 0usize;
        let mut contains_all_truth = 0usize;
        for (q, truth) in queries.iter().zip(truths.iter()) {
            let truth_set: std::collections::HashSet<u32> = truth.iter().copied().collect();
            let sq8_hits_m = search_async(&sq8, q, m, corpus::n_cent(N_DOCS), 1024).await;
            let sq8_ids: Vec<u32> = sq8_hits_m.iter().map(|(id, _)| *id).collect();
            if m == TOP_K {
                sq8_hits += sq8_ids.iter().filter(|id| truth_set.contains(id)).count();
            }
            if truth.iter().all(|id| sq8_ids.contains(id)) {
                contains_all_truth += 1;
            }
            let rescored = exact_rescore_topk(vectors, q, &sq8_ids);
            rescored_hits += rescored.iter().filter(|id| truth_set.contains(id)).count();
        }
        if m == TOP_K {
            eprintln!(
                "rescore-debug: sq8_final_top10 baseline hits={sq8_hits}/{} recall={:.4}",
                N_QUERIES * TOP_K,
                sq8_hits as f32 / (N_QUERIES * TOP_K) as f32
            );
        }
        eprintln!(
            "rescore-debug: sq8_top{m}_then_exact_rescore_top10 hits={rescored_hits}/{} recall={:.4} queries_containing_all_truth={contains_all_truth}/{N_QUERIES}",
            N_QUERIES * TOP_K,
            rescored_hits as f32 / (N_QUERIES * TOP_K) as f32
        );
    }
}

fn fp32_to_f16_bits_debug(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 255 {
        if mant == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7e00;
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x800000;
        let shift = (14 - half_exp) as u32;
        let mut half_mant = (mantissa >> shift) as u16;
        let round_bit = (mantissa >> (shift - 1)) & 1;
        if round_bit != 0 {
            half_mant = half_mant.wrapping_add(1);
        }
        return sign | half_mant;
    }
    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    if (mant & 0x1000) != 0 {
        half = half.wrapping_add(1);
    }
    half
}

fn f16_bits_to_f32_debug(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut mant_norm = mant;
            let mut exp_norm = -14i32;
            while (mant_norm & 0x0400) == 0 {
                mant_norm <<= 1;
                exp_norm -= 1;
            }
            mant_norm &= 0x03ff;
            sign | (((exp_norm + 127) as u32) << 23) | (mant_norm << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn bf16_rescore_topk(vectors: &[f32], query: &[f32], candidate_ids: &[u32]) -> Vec<u32> {
    let dim = corpus::DIM;
    let mut scored = Vec::with_capacity(candidate_ids.len());
    for &id in candidate_ids {
        let off = id as usize * dim;
        let mut dot = 0.0f32;
        let mut norm_sq = 0.0f32;
        for d in 0..dim {
            let x = bf16_to_f32_debug(fp32_to_bf16_debug(vectors[off + d]));
            dot += query[d] * x;
            norm_sq += x * x;
        }
        let norm = norm_sq.sqrt();
        let dist = if norm > 0.0 {
            1.0 - dot / norm
        } else {
            1.0 - dot
        };
        scored.push((id, dist));
    }
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(TOP_K);
    scored.into_iter().map(|(id, _)| id).collect()
}

fn f16_rescore_topk(vectors: &[f32], query: &[f32], candidate_ids: &[u32]) -> Vec<u32> {
    let dim = corpus::DIM;
    let mut scored = Vec::with_capacity(candidate_ids.len());
    for &id in candidate_ids {
        let off = id as usize * dim;
        let mut dot = 0.0f32;
        let mut norm_sq = 0.0f32;
        for d in 0..dim {
            let x = f16_bits_to_f32_debug(fp32_to_f16_bits_debug(vectors[off + d]));
            dot += query[d] * x;
            norm_sq += x * x;
        }
        let norm = norm_sq.sqrt();
        let dist = if norm > 0.0 {
            1.0 - dot / norm
        } else {
            1.0 - dot
        };
        scored.push((id, dist));
    }
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(TOP_K);
    scored.into_iter().map(|(id, _)| id).collect()
}

fn selected_dim_rescore_topk(
    vectors: &[f32],
    query: &[f32],
    candidate_ids: &[u32],
    dims: &[usize],
) -> Vec<u32> {
    let dim = corpus::DIM;
    let mut scored = Vec::with_capacity(candidate_ids.len());
    for &id in candidate_ids {
        let off = id as usize * dim;
        let mut dot = 0.0f32;
        let mut norm_sq = 0.0f32;
        for &d in dims {
            let x = vectors[off + d];
            dot += query[d] * x;
            norm_sq += x * x;
        }
        let norm = norm_sq.sqrt();
        let dist = if norm > 0.0 {
            1.0 - dot / norm
        } else {
            1.0 - dot
        };
        scored.push((id, dist));
    }
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(TOP_K);
    scored.into_iter().map(|(id, _)| id).collect()
}

fn top_variance_dims(vectors: &[f32], n_dims: usize) -> Vec<usize> {
    let dim = corpus::DIM;
    let mut stats = Vec::with_capacity(dim);
    for d in 0..dim {
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for i in 0..N_DOCS {
            let x = vectors[i * dim + d] as f64;
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / N_DOCS as f64;
        let var = (sum_sq / N_DOCS as f64 - mean * mean).max(0.0);
        stats.push((d, var));
    }
    stats.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    stats.truncate(n_dims);
    stats.into_iter().map(|(d, _)| d).collect()
}

#[tokio::test]
#[ignore = "1M-doc exploratory residual recall diagnostic; run explicitly"]
async fn sq8_top20_cheap_sidecar_rescore_on_1m_bench_workload() {
    eprintln!(
        "sidecar-debug: generating mmap corpus N_DOCS={N_DOCS}, dim={}, n_cent={}",
        corpus::DIM,
        corpus::n_cent(N_DOCS)
    );
    let corpus_mmap = corpus::MmapVectorCorpus::generate(N_DOCS, corpus::n_cent(N_DOCS), 1, true);
    let vectors = corpus_mmap.as_slice();
    let queries = corpus::generate_realistic_queries(
        vectors,
        N_DOCS,
        N_QUERIES,
        QUERY_SEED,
        true,
        QUERY_SIGMA,
    );
    eprintln!("sidecar-debug: computing exact fp32 brute-force ground truth");
    let truths = corpus::ground_truth(vectors, N_DOCS, &queries, TOP_K);
    eprintln!("sidecar-debug: building Sq8 reader");
    let sq8 = build_reader(vectors, RerankCodec::Sq8Residual);

    let top32_dims = top_variance_dims(vectors, 32);
    let top64_dims = top_variance_dims(vectors, 64);
    let top128_dims = top_variance_dims(vectors, 128);

    let mut sq8_hits = 0usize;
    let mut fp32_hits = 0usize;
    let mut bf16_hits = 0usize;
    let mut f16_hits = 0usize;
    let mut sel32_hits = 0usize;
    let mut sel64_hits = 0usize;
    let mut sel128_hits = 0usize;

    for (q, truth) in queries.iter().zip(truths.iter()) {
        let truth_set: std::collections::HashSet<u32> = truth.iter().copied().collect();
        let top20 = search_async(&sq8, q, 20, corpus::n_cent(N_DOCS), 1024).await;
        let ids: Vec<u32> = top20.iter().map(|(id, _)| *id).collect();
        sq8_hits += ids[..TOP_K]
            .iter()
            .filter(|id| truth_set.contains(id))
            .count();
        let fp32 = exact_rescore_topk(vectors, q, &ids);
        let bf16 = bf16_rescore_topk(vectors, q, &ids);
        let f16 = f16_rescore_topk(vectors, q, &ids);
        let sel32 = selected_dim_rescore_topk(vectors, q, &ids, &top32_dims);
        let sel64 = selected_dim_rescore_topk(vectors, q, &ids, &top64_dims);
        let sel128 = selected_dim_rescore_topk(vectors, q, &ids, &top128_dims);
        fp32_hits += fp32.iter().filter(|id| truth_set.contains(id)).count();
        bf16_hits += bf16.iter().filter(|id| truth_set.contains(id)).count();
        f16_hits += f16.iter().filter(|id| truth_set.contains(id)).count();
        sel32_hits += sel32.iter().filter(|id| truth_set.contains(id)).count();
        sel64_hits += sel64.iter().filter(|id| truth_set.contains(id)).count();
        sel128_hits += sel128.iter().filter(|id| truth_set.contains(id)).count();
    }

    let denom = (N_QUERIES * TOP_K) as f32;
    eprintln!(
        "sidecar-debug: sq8_top20_take10 hits={sq8_hits}/{} recall={:.4}",
        N_QUERIES * TOP_K,
        sq8_hits as f32 / denom
    );
    eprintln!(
        "sidecar-debug: sq8_top20_fp32_rescore hits={fp32_hits}/{} recall={:.4} sidecar_bytes_per_candidate={}",
        N_QUERIES * TOP_K,
        fp32_hits as f32 / denom,
        corpus::DIM * 4
    );
    eprintln!(
        "sidecar-debug: sq8_top20_bf16_rescore hits={bf16_hits}/{} recall={:.4} sidecar_bytes_per_candidate={}",
        N_QUERIES * TOP_K,
        bf16_hits as f32 / denom,
        corpus::DIM * 2
    );
    eprintln!(
        "sidecar-debug: sq8_top20_f16_rescore hits={f16_hits}/{} recall={:.4} sidecar_bytes_per_candidate={}",
        N_QUERIES * TOP_K,
        f16_hits as f32 / denom,
        corpus::DIM * 2
    );
    eprintln!(
        "sidecar-debug: sq8_top20_selected32_fp32_rescore hits={sel32_hits}/{} recall={:.4} sidecar_bytes_per_candidate={}",
        N_QUERIES * TOP_K,
        sel32_hits as f32 / denom,
        32 * 4
    );
    eprintln!(
        "sidecar-debug: sq8_top20_selected64_fp32_rescore hits={sel64_hits}/{} recall={:.4} sidecar_bytes_per_candidate={}",
        N_QUERIES * TOP_K,
        sel64_hits as f32 / denom,
        64 * 4
    );
    eprintln!(
        "sidecar-debug: sq8_top20_selected128_fp32_rescore hits={sel128_hits}/{} recall={:.4} sidecar_bytes_per_candidate={}",
        N_QUERIES * TOP_K,
        sel128_hits as f32 / denom,
        128 * 4
    );
}

#[derive(Clone)]
struct Sq8Sidecar {
    dim: usize,
    scales: Vec<f32>,
    offsets: Vec<f32>,
    codes_by_doc: Vec<u8>,
}

fn build_sq8_sidecar_by_doc(vectors: &[f32], layout: &IvfLayout) -> Sq8Sidecar {
    let dim = layout.dim;
    let n_cent = layout.n_cent;
    let n_docs = layout.n_docs;
    let mut mins = vec![f32::INFINITY; n_cent * dim];
    let mut maxs = vec![f32::NEG_INFINITY; n_cent * dim];
    for c in 0..n_cent {
        let (off, cnt) = layout.cluster_index[c];
        for pos in off as usize..(off + cnt) as usize {
            let doc_id = layout.doc_ids_by_pos[pos] as usize;
            let row = &vectors[doc_id * dim..(doc_id + 1) * dim];
            for (d, &rd) in row.iter().enumerate().take(dim) {
                let idx = c * dim + d;
                mins[idx] = mins[idx].min(rd);
                maxs[idx] = maxs[idx].max(rd);
            }
        }
    }
    let mut scales = vec![1.0f32; n_cent * dim];
    let mut offsets = vec![0.0f32; n_cent * dim];
    for idx in 0..n_cent * dim {
        if mins[idx].is_finite() {
            offsets[idx] = mins[idx];
            if maxs[idx] > mins[idx] {
                scales[idx] = (maxs[idx] - mins[idx]) / 255.0;
            }
        }
    }
    let mut codes_by_doc = vec![0u8; n_docs * dim];
    for c in 0..n_cent {
        let (off, cnt) = layout.cluster_index[c];
        let scale_c = &scales[c * dim..(c + 1) * dim];
        let offset_c = &offsets[c * dim..(c + 1) * dim];
        for pos in off as usize..(off + cnt) as usize {
            let doc_id = layout.doc_ids_by_pos[pos] as usize;
            let row = &vectors[doc_id * dim..(doc_id + 1) * dim];
            let code = &mut codes_by_doc[doc_id * dim..(doc_id + 1) * dim];
            for d in 0..dim {
                let q = ((row[d] - offset_c[d]) / scale_c[d])
                    .round()
                    .clamp(0.0, 255.0) as u8;
                code[d] = q;
            }
        }
    }
    Sq8Sidecar {
        dim,
        scales,
        offsets,
        codes_by_doc,
    }
}

fn sq8_cluster_for_doc(layout: &IvfLayout, doc_id: u32) -> usize {
    for c in 0..layout.cluster_index.len() {
        let (off, cnt) = layout.cluster_index[c];
        for pos in off as usize..(off + cnt) as usize {
            if layout.doc_ids_by_pos[pos] == doc_id {
                return c;
            }
        }
    }
    panic!("doc id {doc_id} not found in layout")
}

fn fp16_residual_corrected_rescore_topk(
    vectors: &[f32],
    query: &[f32],
    candidate_ids: &[u32],
    layout: &IvfLayout,
    sq8: &Sq8Sidecar,
) -> Vec<u32> {
    let dim = sq8.dim;
    let mut scored = Vec::with_capacity(candidate_ids.len());
    for &id in candidate_ids {
        let c = sq8_cluster_for_doc(layout, id);
        let scale_c = &sq8.scales[c * dim..(c + 1) * dim];
        let offset_c = &sq8.offsets[c * dim..(c + 1) * dim];
        let code = &sq8.codes_by_doc[id as usize * dim..(id as usize + 1) * dim];
        let row = &vectors[id as usize * dim..(id as usize + 1) * dim];
        let mut dot = 0.0f32;
        let mut norm_sq = 0.0f32;
        for d in 0..dim {
            let base = (code[d] as f32) * scale_c[d] + offset_c[d];
            let residual = row[d] - base;
            let corrected = base + f16_bits_to_f32_debug(fp32_to_f16_bits_debug(residual));
            dot += query[d] * corrected;
            norm_sq += corrected * corrected;
        }
        let norm = norm_sq.sqrt();
        let dist = if norm > 0.0 {
            1.0 - dot / norm
        } else {
            1.0 - dot
        };
        scored.push((id, dist));
    }
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(TOP_K);
    scored.into_iter().map(|(id, _)| id).collect()
}

fn int8_residual_corrected_rescore_topk(
    vectors: &[f32],
    query: &[f32],
    candidate_ids: &[u32],
    layout: &IvfLayout,
    sq8: &Sq8Sidecar,
    multiplier: f32,
) -> Vec<u32> {
    let dim = sq8.dim;
    let mut scored = Vec::with_capacity(candidate_ids.len());
    for &id in candidate_ids {
        let c = sq8_cluster_for_doc(layout, id);
        let scale_c = &sq8.scales[c * dim..(c + 1) * dim];
        let offset_c = &sq8.offsets[c * dim..(c + 1) * dim];
        let code = &sq8.codes_by_doc[id as usize * dim..(id as usize + 1) * dim];
        let row = &vectors[id as usize * dim..(id as usize + 1) * dim];
        let mut dot = 0.0f32;
        let mut norm_sq = 0.0f32;
        for d in 0..dim {
            let base = (code[d] as f32) * scale_c[d] + offset_c[d];
            let step = scale_c[d] / multiplier;
            let residual = row[d] - base;
            let rq = if step > 0.0 {
                (residual / step).round().clamp(-127.0, 127.0)
            } else {
                0.0
            };
            let corrected = base + rq * step;
            dot += query[d] * corrected;
            norm_sq += corrected * corrected;
        }
        let norm = norm_sq.sqrt();
        let dist = if norm > 0.0 {
            1.0 - dot / norm
        } else {
            1.0 - dot
        };
        scored.push((id, dist));
    }
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(TOP_K);
    scored.into_iter().map(|(id, _)| id).collect()
}

#[tokio::test]
#[ignore = "1M-doc exploratory residual recall diagnostic; run explicitly"]
async fn sq8_top20_residual_sidecar_rescore_on_1m_bench_workload() {
    eprintln!(
        "residual-debug: generating mmap corpus N_DOCS={N_DOCS}, dim={}, n_cent={}",
        corpus::DIM,
        corpus::n_cent(N_DOCS)
    );
    let corpus_mmap = corpus::MmapVectorCorpus::generate(N_DOCS, corpus::n_cent(N_DOCS), 1, true);
    let vectors = corpus_mmap.as_slice();
    let queries = corpus::generate_realistic_queries(
        vectors,
        N_DOCS,
        N_QUERIES,
        QUERY_SEED,
        true,
        QUERY_SIGMA,
    );
    eprintln!("residual-debug: computing exact fp32 brute-force ground truth");
    let truths = corpus::ground_truth(vectors, N_DOCS, &queries, TOP_K);
    eprintln!("residual-debug: building Fp32 blob for IVF layout parse");
    let fp32_blob = build_blob(vectors, RerankCodec::Fp32);
    let layout = parse_ivf_layout_from_fp32_blob(&fp32_blob);
    eprintln!("residual-debug: building Sq8 reader + Sq8 sidecar model");
    let sq8_reader = build_reader(vectors, RerankCodec::Sq8Residual);
    let sq8_sidecar = build_sq8_sidecar_by_doc(vectors, &layout);

    let mut fp32_hits = 0usize;
    let mut fp16_residual_hits = 0usize;
    let mut int8_residual_x2_hits = 0usize;
    let mut int8_residual_x4_hits = 0usize;
    let mut int8_residual_x8_hits = 0usize;
    let mut int8_residual_x16_hits = 0usize;
    let mut int8_residual_x32_hits = 0usize;

    for (q, truth) in queries.iter().zip(truths.iter()) {
        let truth_set: std::collections::HashSet<u32> = truth.iter().copied().collect();
        let top20 = search_async(&sq8_reader, q, 20, corpus::n_cent(N_DOCS), 1024).await;
        let ids: Vec<u32> = top20.iter().map(|(id, _)| *id).collect();
        let fp32 = exact_rescore_topk(vectors, q, &ids);
        let fp16_residual =
            fp16_residual_corrected_rescore_topk(vectors, q, &ids, &layout, &sq8_sidecar);
        let int8_residual_x2 =
            int8_residual_corrected_rescore_topk(vectors, q, &ids, &layout, &sq8_sidecar, 2.0);
        let int8_residual_x4 =
            int8_residual_corrected_rescore_topk(vectors, q, &ids, &layout, &sq8_sidecar, 4.0);
        let int8_residual_x8 =
            int8_residual_corrected_rescore_topk(vectors, q, &ids, &layout, &sq8_sidecar, 8.0);
        let int8_residual_x16 =
            int8_residual_corrected_rescore_topk(vectors, q, &ids, &layout, &sq8_sidecar, 16.0);
        let int8_residual_x32 =
            int8_residual_corrected_rescore_topk(vectors, q, &ids, &layout, &sq8_sidecar, 32.0);
        fp32_hits += fp32.iter().filter(|id| truth_set.contains(id)).count();
        fp16_residual_hits += fp16_residual
            .iter()
            .filter(|id| truth_set.contains(id))
            .count();
        int8_residual_x2_hits += int8_residual_x2
            .iter()
            .filter(|id| truth_set.contains(id))
            .count();
        int8_residual_x4_hits += int8_residual_x4
            .iter()
            .filter(|id| truth_set.contains(id))
            .count();
        int8_residual_x8_hits += int8_residual_x8
            .iter()
            .filter(|id| truth_set.contains(id))
            .count();
        int8_residual_x16_hits += int8_residual_x16
            .iter()
            .filter(|id| truth_set.contains(id))
            .count();
        int8_residual_x32_hits += int8_residual_x32
            .iter()
            .filter(|id| truth_set.contains(id))
            .count();
    }

    let denom = (N_QUERIES * TOP_K) as f32;
    eprintln!(
        "residual-debug: sq8_top20_fp32_rescore hits={fp32_hits}/{} recall={:.4} sidecar_bytes_per_candidate={}",
        N_QUERIES * TOP_K,
        fp32_hits as f32 / denom,
        corpus::DIM * 4
    );
    eprintln!(
        "residual-debug: sq8_top20_fp16_residual_rescore hits={fp16_residual_hits}/{} recall={:.4} sidecar_bytes_per_candidate={}",
        N_QUERIES * TOP_K,
        fp16_residual_hits as f32 / denom,
        corpus::DIM * 2
    );
    eprintln!(
        "residual-debug: sq8_top20_int8_residual_x2_rescore hits={int8_residual_x2_hits}/{} recall={:.4} sidecar_bytes_per_candidate={}",
        N_QUERIES * TOP_K,
        int8_residual_x2_hits as f32 / denom,
        corpus::DIM
    );
    eprintln!(
        "residual-debug: sq8_top20_int8_residual_x4_rescore hits={int8_residual_x4_hits}/{} recall={:.4} sidecar_bytes_per_candidate={}",
        N_QUERIES * TOP_K,
        int8_residual_x4_hits as f32 / denom,
        corpus::DIM
    );
    eprintln!(
        "residual-debug: sq8_top20_int8_residual_x8_rescore hits={int8_residual_x8_hits}/{} recall={:.4} sidecar_bytes_per_candidate={}",
        N_QUERIES * TOP_K,
        int8_residual_x8_hits as f32 / denom,
        corpus::DIM
    );
    eprintln!(
        "residual-debug: sq8_top20_int8_residual_x16_rescore hits={int8_residual_x16_hits}/{} recall={:.4} sidecar_bytes_per_candidate={}",
        N_QUERIES * TOP_K,
        int8_residual_x16_hits as f32 / denom,
        corpus::DIM
    );
    eprintln!(
        "residual-debug: sq8_top20_int8_residual_x32_rescore hits={int8_residual_x32_hits}/{} recall={:.4} sidecar_bytes_per_candidate={}",
        N_QUERIES * TOP_K,
        int8_residual_x32_hits as f32 / denom,
        corpus::DIM
    );
}
