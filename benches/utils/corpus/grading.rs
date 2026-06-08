// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Recall grading for supertable vector benches: queries + brute-force top-k.
//!
//! Does **not** mmap the full corpus. Regenerates the same synthetic stream as
//! ingest ([`SequentialSyntheticCorpus`]), collects query seeds, then one streaming
//! pass over all docs with a per-query top-k heap. Optionally caches the small
//! result (~120 queries + top-k ids) to a temp file for faster re-runs.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::corpus::{self, DIM, SUPERTABLE_DOCS, SequentialSyntheticCorpus, normalize};

const MAGIC: &[u8; 8] = b"INFBENCH";
const CACHE_VERSION: u32 = 1;

const VEC_SEED: u64 = 1;
const TEXT_SEED: u64 = 1;

const CORRECTNESS_QUERY_SEED: u64 = 17;
const CALIBRATION_QUERY_SEED: u64 = 99;
const QUERY_SIGMA: f32 = 0.05;

const N_CORRECTNESS_QUERIES: usize = 20;
const N_CALIBRATION_QUERIES: usize = 100;
const TOP_K: usize = 10;

/// Coprime stride spreading query base docs across the corpus
/// (matches `corpus.rs`).
const QUERY_BASE_DOC_STRIDE: usize = 7919;
/// Doc count streamed per chunk when collecting vectors / computing
/// ground truth.
const STREAM_CHUNK_SIZE: usize = 65_536;

/// Cached queries + ground truth for vector search benches.
pub struct SupertableGrading {
    pub correctness_queries: Vec<Vec<f32>>,
    pub correctness_gt: Vec<Vec<u32>>,
    pub calibration_queries: Vec<Vec<f32>>,
    pub calibration_gt: Vec<Vec<u32>>,
}

static GRADING: OnceLock<SupertableGrading> = OnceLock::new();

pub const SUPERTABLE_N_DOCS: usize = SUPERTABLE_DOCS;
pub const VECTOR_DIM: usize = DIM;

pub fn supertable_grading() -> &'static SupertableGrading {
    GRADING.get_or_init(|| {
        eprintln!(
            "[grading] building query set + brute-force top-{TOP_K} (streamed, no full-corpus mmap)..."
        );
        let t0 = std::time::Instant::now();
        let g = load_or_compute(SUPERTABLE_DOCS);
        eprintln!(
            "[grading] OK: {} correctness + {} calibration queries ({:.1}s)",
            g.correctness_queries.len(),
            g.calibration_queries.len(),
            t0.elapsed().as_secs_f64()
        );
        g
    })
}

pub fn realistic_queries(n_docs: usize, n_queries: usize, seed: u64, sigma: f32) -> Vec<Vec<f32>> {
    let bases: HashSet<usize> = query_base_doc_ids(n_docs, n_queries).into_iter().collect();
    let base_vectors = collect_doc_vectors(n_docs, &bases);
    build_queries_from_bases(n_docs, n_queries, seed, sigma, &base_vectors)
}

pub fn ground_truth(n_docs: usize, queries: &[Vec<f32>], top_k: usize) -> Vec<Vec<u32>> {
    streaming_ground_truth(n_docs, queries, top_k)
}

fn load_or_compute(n_docs: usize) -> SupertableGrading {
    let path = cache_path(n_docs);
    if let Ok(g) = read_cache(&path, n_docs) {
        eprintln!("[grading] loaded cached labels from {}", path.display());
        return g;
    }
    let g = compute(n_docs);
    if let Err(e) = write_cache(&path, n_docs, &g) {
        eprintln!("[grading] cache write skipped: {e}");
    } else {
        eprintln!("[grading] wrote cache {}", path.display());
    }
    g
}

fn compute(n_docs: usize) -> SupertableGrading {
    let corr_bases = query_base_doc_ids(n_docs, N_CORRECTNESS_QUERIES);
    let cal_bases = query_base_doc_ids(n_docs, N_CALIBRATION_QUERIES);
    let all_bases: HashSet<usize> = corr_bases.iter().chain(cal_bases.iter()).copied().collect();

    let base_vectors = collect_doc_vectors(n_docs, &all_bases);

    let correctness_queries = build_queries_from_bases(
        n_docs,
        N_CORRECTNESS_QUERIES,
        CORRECTNESS_QUERY_SEED,
        QUERY_SIGMA,
        &base_vectors,
    );
    let calibration_queries = build_queries_from_bases(
        n_docs,
        N_CALIBRATION_QUERIES,
        CALIBRATION_QUERY_SEED,
        QUERY_SIGMA,
        &base_vectors,
    );

    let mut all_queries = correctness_queries.clone();
    all_queries.extend(calibration_queries.clone());
    let all_gt = streaming_ground_truth(n_docs, &all_queries, TOP_K);

    let n_corr = N_CORRECTNESS_QUERIES;
    SupertableGrading {
        correctness_gt: all_gt[..n_corr].to_vec(),
        calibration_gt: all_gt[n_corr..].to_vec(),
        correctness_queries,
        calibration_queries,
    }
}

fn query_base_doc_ids(n_docs: usize, n_queries: usize) -> Vec<usize> {
    (0..n_queries)
        .map(|i| (i * QUERY_BASE_DOC_STRIDE) % n_docs)
        .collect()
}

fn build_queries_from_bases(
    n_docs: usize,
    n_queries: usize,
    seed: u64,
    sigma: f32,
    base_vectors: &HashMap<usize, Vec<f32>>,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = StandardNormal;
    let mut out = Vec::with_capacity(n_queries);
    for i in 0..n_queries {
        let base_idx = (i * QUERY_BASE_DOC_STRIDE) % n_docs;
        let base = base_vectors
            .get(&base_idx)
            .unwrap_or_else(|| panic!("missing base vector for doc {base_idx}"));
        let mut q: Vec<f32> = (0..DIM)
            .map(|d| {
                let s: f64 = dist.sample(&mut rng);
                base[d] + (s as f32) * sigma
            })
            .collect();
        normalize(&mut q);
        out.push(q);
    }
    out
}

/// Stream the synthetic corpus once and retain vectors for the listed doc ids.
fn collect_doc_vectors(n_docs: usize, doc_ids: &HashSet<usize>) -> HashMap<usize, Vec<f32>> {
    let n_cent = corpus::n_cent(n_docs);
    let mut synth = SequentialSyntheticCorpus::new(n_cent, VEC_SEED, TEXT_SEED, true);
    let mut titles = Vec::new();
    let mut flat = Vec::new();
    let chunk_size = STREAM_CHUNK_SIZE.min(n_docs.max(1));
    let mut found = HashMap::with_capacity(doc_ids.len());
    let mut doc_id = 0usize;

    while doc_id < n_docs && found.len() < doc_ids.len() {
        let len = chunk_size.min(n_docs - doc_id);
        synth.fill_chunk(len, &mut titles, &mut flat);
        for local in 0..len {
            let id = doc_id + local;
            if doc_ids.contains(&id) {
                let off = local * DIM;
                found.insert(id, flat[off..off + DIM].to_vec());
            }
        }
        doc_id += len;
    }
    assert_eq!(
        found.len(),
        doc_ids.len(),
        "streamed corpus missing some base doc vectors"
    );
    found
}

/// One pass over all docs; maintain top-k largest dot product per query.
fn streaming_ground_truth(n_docs: usize, queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    let n_cent = corpus::n_cent(n_docs);
    let mut synth = SequentialSyntheticCorpus::new(n_cent, VEC_SEED, TEXT_SEED, true);
    let mut heaps: Vec<BinaryHeap<Reverse<Hit>>> =
        queries.iter().map(|_| BinaryHeap::new()).collect();
    let mut titles = Vec::new();
    let mut flat = Vec::new();
    let chunk_size = STREAM_CHUNK_SIZE.min(n_docs.max(1));
    let mut doc_id = 0usize;

    while doc_id < n_docs {
        let len = chunk_size.min(n_docs - doc_id);
        synth.fill_chunk(len, &mut titles, &mut flat);
        for (qi, q) in queries.iter().enumerate() {
            let heap = &mut heaps[qi];
            for local in 0..len {
                let off = local * DIM;
                let mut dot = 0f32;
                for d in 0..DIM {
                    dot += flat[off + d] * q[d];
                }
                let global = (doc_id + local) as u32;
                if heap.len() < k {
                    heap.push(Reverse(Hit(dot, global)));
                } else if let Some(Reverse(worst)) = heap.peek()
                    && dot > worst.0
                {
                    heap.pop();
                    heap.push(Reverse(Hit(dot, global)));
                }
            }
        }
        doc_id += len;
    }

    heaps
        .into_iter()
        .map(|mut h| {
            let mut v: Vec<(f32, u32)> = h.drain().map(|Reverse(Hit(dot, id))| (dot, id)).collect();
            v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            v.truncate(k);
            v.into_iter().map(|(_, id)| id).collect()
        })
        .collect()
}

#[derive(PartialEq)]
struct Hit(f32, u32);

impl Eq for Hit {}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

fn cache_path(n_docs: usize) -> PathBuf {
    std::env::temp_dir().join(format!(
        "infino_supertable_grading_v{CACHE_VERSION}_{n_docs}_{VEC_SEED}_{TEXT_SEED}.bin"
    ))
}

fn write_cache(path: &PathBuf, n_docs: usize, g: &SupertableGrading) -> std::io::Result<()> {
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&CACHE_VERSION.to_le_bytes());
    buf.extend_from_slice(&(n_docs as u64).to_le_bytes());
    push_queries(&mut buf, &g.correctness_queries);
    push_gt(&mut buf, &g.correctness_gt);
    push_queries(&mut buf, &g.calibration_queries);
    push_gt(&mut buf, &g.calibration_gt);
    fs::write(path, buf)
}

fn read_cache(path: &PathBuf, n_docs: usize) -> std::io::Result<SupertableGrading> {
    let bytes = fs::read(path)?;
    let mut cursor = &bytes[..];
    let mut magic = [0u8; 8];
    read_exact(&mut cursor, &mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad grading cache magic",
        ));
    }
    let version = read_u32(&mut cursor)?;
    if version != CACHE_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "grading cache version mismatch",
        ));
    }
    let stored_docs = read_u64(&mut cursor)? as usize;
    if stored_docs != n_docs {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "grading cache n_docs mismatch",
        ));
    }
    Ok(SupertableGrading {
        correctness_queries: pull_queries(&mut cursor)?,
        correctness_gt: pull_gt(&mut cursor)?,
        calibration_queries: pull_queries(&mut cursor)?,
        calibration_gt: pull_gt(&mut cursor)?,
    })
}

fn push_queries(buf: &mut Vec<u8>, queries: &[Vec<f32>]) {
    buf.extend_from_slice(&(queries.len() as u32).to_le_bytes());
    for q in queries {
        assert_eq!(q.len(), DIM);
        for &v in q {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
}

fn push_gt(buf: &mut Vec<u8>, gt: &[Vec<u32>]) {
    buf.extend_from_slice(&(gt.len() as u32).to_le_bytes());
    for row in gt {
        assert_eq!(row.len(), TOP_K);
        for &id in row {
            buf.extend_from_slice(&id.to_le_bytes());
        }
    }
}

fn pull_queries(cursor: &mut &[u8]) -> std::io::Result<Vec<Vec<f32>>> {
    let n = read_u32(cursor)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut q = vec![0f32; DIM];
        for slot in &mut q {
            *slot = read_f32(cursor)?;
        }
        out.push(q);
    }
    Ok(out)
}

fn pull_gt(cursor: &mut &[u8]) -> std::io::Result<Vec<Vec<u32>>> {
    let n = read_u32(cursor)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row = vec![0u32; TOP_K];
        for id in &mut row {
            *id = read_u32(cursor)?;
        }
        out.push(row);
    }
    Ok(out)
}

fn read_exact(cursor: &mut &[u8], buf: &mut [u8]) -> std::io::Result<()> {
    if cursor.len() < buf.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "grading cache truncated",
        ));
    }
    buf.copy_from_slice(&cursor[..buf.len()]);
    *cursor = &cursor[buf.len()..];
    Ok(())
}

fn read_u32(cursor: &mut &[u8]) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    read_exact(cursor, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(cursor: &mut &[u8]) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    read_exact(cursor, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f32(cursor: &mut &[u8]) -> std::io::Result<f32> {
    let mut b = [0u8; 4];
    read_exact(cursor, &mut b)?;
    Ok(f32::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny corpus size for the stream-vs-brute-force ground-truth test.
    const TEST_N_DOCS: usize = 512;
    /// Query count for the tiny-corpus ground-truth test.
    const TEST_N_QUERIES: usize = 4;
    /// Top-k compared between streaming and brute-force ground truth.
    const TEST_TOP_K: usize = 5;

    #[test]
    fn streaming_gt_matches_brute_force_on_tiny_corpus() {
        let n_docs = TEST_N_DOCS;
        let queries =
            realistic_queries(n_docs, TEST_N_QUERIES, CORRECTNESS_QUERY_SEED, QUERY_SIGMA);
        let stream_gt = streaming_ground_truth(n_docs, &queries, TEST_TOP_K);
        let mmap =
            corpus::MmapVectorCorpus::generate(n_docs, corpus::n_cent(n_docs), VEC_SEED, true);
        let brute = corpus::ground_truth(mmap.as_slice(), n_docs, &queries, TEST_TOP_K);
        assert_eq!(stream_gt, brute);
    }
}
