// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Cached exact recall labels for the supertable vector lifecycle.
//!
//! One document-major pass computes the base, filtered, and post-delta
//! top-k labels together. The result is tiny and deterministic, so subsequent
//! runs load it from `TMPDIR` instead of rescanning the corpus.

use std::{
    env, fs,
    io::{Error, ErrorKind, Result},
    path::{Path, PathBuf},
    process,
    time::Instant,
};

use crate::corpus::{self, DIM, LifecycleGroundTruth};

const CACHE_MAGIC: &[u8; 8] = b"INFLGT02";
const CACHE_VERSION: u32 = 2;
/// Existing lifecycle oracle loaded without fallback or recomputation.
const GROUND_TRUTH_PATH_ENV: &str = "INFINO_BENCH_VECTOR_GROUND_TRUTH_PATH";
/// FNV-1a offset basis used only for deterministic cache filenames.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a prime used only for deterministic cache filenames.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Inputs that define one exact lifecycle oracle.
pub struct LifecycleGradingOptions<'a> {
    pub vectors: &'a [f32],
    pub n_docs: usize,
    pub augmented_docs: usize,
    pub corpus_seed: u64,
    pub normalized_vectors: bool,
    pub filter_keep_every: usize,
    pub top_k: usize,
    pub correctness_query_count: usize,
    pub queries: &'a [Vec<f32>],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CacheKey {
    n_docs: u64,
    augmented_docs: u64,
    dim: u64,
    n_cent: u64,
    corpus_seed: u64,
    normalized_vectors: bool,
    filter_keep_every: u64,
    top_k: u64,
    correctness_query_count: u64,
    query_count: u64,
}

impl CacheKey {
    fn from_options(options: &LifecycleGradingOptions<'_>) -> Self {
        Self {
            n_docs: options.n_docs as u64,
            augmented_docs: options.augmented_docs as u64,
            dim: DIM as u64,
            n_cent: corpus::n_cent(options.n_docs) as u64,
            corpus_seed: options.corpus_seed,
            normalized_vectors: options.normalized_vectors,
            filter_keep_every: options.filter_keep_every as u64,
            top_k: options.top_k as u64,
            correctness_query_count: options.correctness_query_count as u64,
            query_count: options.queries.len() as u64,
        }
    }
}

/// Load exact lifecycle labels from the deterministic cache, or compute and
/// atomically publish them when this corpus/query shape has not been graded.
pub fn lifecycle_ground_truth_cached(options: LifecycleGradingOptions<'_>) -> LifecycleGroundTruth {
    validate_options(&options);
    let key = CacheKey::from_options(&options);
    if let Some(path) = env::var_os(GROUND_TRUTH_PATH_ENV).map(PathBuf::from) {
        let labels = read_cache(&path, key, options.queries).unwrap_or_else(|error| {
            panic!(
                "failed to load or validate {GROUND_TRUTH_PATH_ENV}={}: {error}",
                path.display()
            )
        });
        eprintln!(
            "[vector ground truth] loaded exact lifecycle oracle from {}",
            path.display()
        );
        return labels;
    }

    let path = cache_path(&key, options.queries);
    match read_cache(&path, key, options.queries) {
        Ok(labels) => {
            eprintln!(
                "[vector ground truth] loaded exact lifecycle oracle from {}",
                path.display()
            );
            return labels;
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => {
            eprintln!(
                "[vector ground truth] ignoring stale cache {}: {error}",
                path.display()
            );
        }
    }

    assert_eq!(
        options.vectors.len(),
        options.augmented_docs * DIM,
        "cannot recompute lifecycle ground truth from a base-only vector corpus; \
         set {GROUND_TRUTH_PATH_ENV} to a valid existing oracle cache"
    );
    eprintln!(
        "[vector ground truth] computing one exact pass over {} docs for {} queries...",
        options.augmented_docs,
        options.queries.len()
    );
    let started = Instant::now();
    let labels = corpus::lifecycle_ground_truth(
        options.vectors,
        options.n_docs,
        options.augmented_docs,
        options.queries,
        options.correctness_query_count,
        options.filter_keep_every,
        options.top_k,
    );
    match write_cache(&path, key, options.queries, &labels) {
        Ok(()) => eprintln!(
            "[vector ground truth] cached exact lifecycle oracle at {} ({:.1}s)",
            path.display(),
            started.elapsed().as_secs_f64()
        ),
        Err(error) => eprintln!(
            "[vector ground truth] cache write skipped for {}: {error}",
            path.display()
        ),
    }
    labels
}

fn validate_options(options: &LifecycleGradingOptions<'_>) {
    assert!(options.n_docs > 0);
    assert!(options.n_docs <= options.augmented_docs);
    assert!(
        options.vectors.len() == options.n_docs * DIM
            || options.vectors.len() == options.augmented_docs * DIM,
        "lifecycle vector corpus must contain either the base or augmented row count"
    );
    assert!(options.filter_keep_every > 0);
    assert!(options.top_k > 0);
    assert!(options.correctness_query_count <= options.queries.len());
    for query in options.queries {
        assert_eq!(query.len(), DIM);
    }
}

fn cache_path(key: &CacheKey, queries: &[Vec<f32>]) -> PathBuf {
    let fingerprint = cache_fingerprint(key, queries);
    env::temp_dir().join(format!(
        "infino_vector_lifecycle_gt_v{CACHE_VERSION}_{}_{}_{fingerprint:016x}.bin",
        key.n_docs, key.augmented_docs
    ))
}

fn cache_fingerprint(key: &CacheKey, queries: &[Vec<f32>]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for value in [
        key.n_docs,
        key.augmented_docs,
        key.dim,
        key.n_cent,
        key.corpus_seed,
        u64::from(key.normalized_vectors),
        key.filter_keep_every,
        key.top_k,
        key.correctness_query_count,
        key.query_count,
    ] {
        hash_bytes(&mut hash, &value.to_le_bytes());
    }
    for query in queries {
        for value in query {
            hash_bytes(&mut hash, &value.to_bits().to_le_bytes());
        }
    }
    hash
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn write_cache(
    path: &Path,
    key: CacheKey,
    queries: &[Vec<f32>],
    labels: &LifecycleGroundTruth,
) -> Result<()> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(CACHE_MAGIC);
    bytes.extend_from_slice(&CACHE_VERSION.to_le_bytes());
    push_key(&mut bytes, key);
    push_queries(&mut bytes, queries);
    push_ground_truth(&mut bytes, &labels.base, key.top_k as usize);
    push_ground_truth(&mut bytes, &labels.filtered, key.top_k as usize);
    push_ground_truth(&mut bytes, &labels.augmented, key.top_k as usize);

    let temporary = path.with_extension(format!("{}.tmp", process::id()));
    fs::write(&temporary, bytes)?;
    match fs::rename(&temporary, path) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(error)
        }
    }
}

fn read_cache(
    path: &Path,
    expected_key: CacheKey,
    expected_queries: &[Vec<f32>],
) -> Result<LifecycleGroundTruth> {
    let bytes = fs::read(path)?;
    let mut cursor = bytes.as_slice();
    let mut magic = [0u8; CACHE_MAGIC.len()];
    read_exact(&mut cursor, &mut magic)?;
    if &magic != CACHE_MAGIC {
        return invalid_data("bad lifecycle ground-truth cache magic");
    }
    if read_u32(&mut cursor)? != CACHE_VERSION {
        return invalid_data("lifecycle ground-truth cache version mismatch");
    }
    if pull_key(&mut cursor)? != expected_key {
        return invalid_data("lifecycle ground-truth cache key mismatch");
    }
    validate_queries(&mut cursor, expected_queries)?;
    let top_k = expected_key.top_k as usize;
    let base = pull_ground_truth(&mut cursor, expected_queries.len(), top_k)?;
    let correctness = expected_key.correctness_query_count as usize;
    let filtered = pull_ground_truth(&mut cursor, correctness, top_k)?;
    let augmented = pull_ground_truth(&mut cursor, correctness, top_k)?;
    if !cursor.is_empty() {
        return invalid_data("lifecycle ground-truth cache has trailing bytes");
    }
    Ok(LifecycleGroundTruth {
        base,
        filtered,
        augmented,
    })
}

fn push_key(bytes: &mut Vec<u8>, key: CacheKey) {
    for value in [
        key.n_docs,
        key.augmented_docs,
        key.dim,
        key.n_cent,
        key.corpus_seed,
        key.filter_keep_every,
        key.top_k,
        key.correctness_query_count,
        key.query_count,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.push(u8::from(key.normalized_vectors));
}

fn pull_key(cursor: &mut &[u8]) -> Result<CacheKey> {
    Ok(CacheKey {
        n_docs: read_u64(cursor)?,
        augmented_docs: read_u64(cursor)?,
        dim: read_u64(cursor)?,
        n_cent: read_u64(cursor)?,
        corpus_seed: read_u64(cursor)?,
        filter_keep_every: read_u64(cursor)?,
        top_k: read_u64(cursor)?,
        correctness_query_count: read_u64(cursor)?,
        query_count: read_u64(cursor)?,
        normalized_vectors: read_u8(cursor)? != 0,
    })
}

fn push_queries(bytes: &mut Vec<u8>, queries: &[Vec<f32>]) {
    bytes.extend_from_slice(&(queries.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(DIM as u64).to_le_bytes());
    for query in queries {
        for value in query {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
}

fn validate_queries(cursor: &mut &[u8], expected: &[Vec<f32>]) -> Result<()> {
    if read_u64(cursor)? as usize != expected.len() || read_u64(cursor)? as usize != DIM {
        return invalid_data("lifecycle ground-truth cache query shape mismatch");
    }
    for query in expected {
        for value in query {
            if read_u32(cursor)? != value.to_bits() {
                return invalid_data("lifecycle ground-truth cache query mismatch");
            }
        }
    }
    Ok(())
}

fn push_ground_truth(bytes: &mut Vec<u8>, labels: &[Vec<u32>], top_k: usize) {
    bytes.extend_from_slice(&(labels.len() as u64).to_le_bytes());
    for row in labels {
        assert_eq!(row.len(), top_k);
        for id in row {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
    }
}

fn pull_ground_truth(
    cursor: &mut &[u8],
    expected_rows: usize,
    top_k: usize,
) -> Result<Vec<Vec<u32>>> {
    if read_u64(cursor)? as usize != expected_rows {
        return invalid_data("lifecycle ground-truth cache row count mismatch");
    }
    let mut labels = Vec::with_capacity(expected_rows);
    for _ in 0..expected_rows {
        let mut row = Vec::with_capacity(top_k);
        for _ in 0..top_k {
            row.push(read_u32(cursor)?);
        }
        labels.push(row);
    }
    Ok(labels)
}

fn read_exact(cursor: &mut &[u8], output: &mut [u8]) -> Result<()> {
    if cursor.len() < output.len() {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "lifecycle ground-truth cache truncated",
        ));
    }
    output.copy_from_slice(&cursor[..output.len()]);
    *cursor = &cursor[output.len()..];
    Ok(())
}

fn read_u8(cursor: &mut &[u8]) -> Result<u8> {
    let mut bytes = [0u8; 1];
    read_exact(cursor, &mut bytes)?;
    Ok(bytes[0])
}

fn read_u32(cursor: &mut &[u8]) -> Result<u32> {
    let mut bytes = [0u8; 4];
    read_exact(cursor, &mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(cursor: &mut &[u8]) -> Result<u64> {
    let mut bytes = [0u8; 8];
    read_exact(cursor, &mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn invalid_data<T>(message: &str) -> Result<T> {
    Err(Error::new(ErrorKind::InvalidData, message))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::corpus::{MmapVectorCorpus, generate_realistic_queries};

    const TEST_N_DOCS: usize = 512;
    const TEST_AUGMENTED_DOCS: usize = 576;
    const TEST_N_QUERIES: usize = 4;
    const TEST_N_CORRECTNESS_QUERIES: usize = 3;
    const TEST_TOP_K: usize = 5;
    const TEST_FILTER_KEEP_EVERY: usize = 7;
    const TEST_CORPUS_SEED: u64 = 1;
    const TEST_QUERY_SEED: u64 = 17;
    const TEST_QUERY_SIGMA: f32 = 0.05;

    fn fixture() -> (
        MmapVectorCorpus,
        Vec<Vec<f32>>,
        CacheKey,
        LifecycleGroundTruth,
    ) {
        let vectors = MmapVectorCorpus::generate(
            TEST_AUGMENTED_DOCS,
            corpus::n_cent(TEST_N_DOCS),
            TEST_CORPUS_SEED,
            true,
        );
        let queries = generate_realistic_queries(
            vectors.as_slice(),
            TEST_N_DOCS,
            TEST_N_QUERIES,
            TEST_QUERY_SEED,
            true,
            TEST_QUERY_SIGMA,
        );
        let options = LifecycleGradingOptions {
            vectors: vectors.as_slice(),
            n_docs: TEST_N_DOCS,
            augmented_docs: TEST_AUGMENTED_DOCS,
            corpus_seed: TEST_CORPUS_SEED,
            normalized_vectors: true,
            filter_keep_every: TEST_FILTER_KEEP_EVERY,
            top_k: TEST_TOP_K,
            correctness_query_count: TEST_N_CORRECTNESS_QUERIES,
            queries: &queries,
        };
        let key = CacheKey::from_options(&options);
        let labels = corpus::lifecycle_ground_truth(
            options.vectors,
            options.n_docs,
            options.augmented_docs,
            options.queries,
            options.correctness_query_count,
            options.filter_keep_every,
            options.top_k,
        );
        (vectors, queries, key, labels)
    }

    #[test]
    fn cache_roundtrip_validates_full_key_and_queries() {
        let directory = tempdir().expect("tempdir");
        let path = directory.path().join("grading.bin");
        let (_vectors, queries, key, labels) = fixture();
        write_cache(&path, key, &queries, &labels).expect("write");
        let restored = read_cache(&path, key, &queries).expect("read");
        assert_eq!(restored.base, labels.base);
        assert_eq!(restored.filtered, labels.filtered);
        assert_eq!(restored.augmented, labels.augmented);

        let stale_key = CacheKey {
            filter_keep_every: key.filter_keep_every + 1,
            ..key
        };
        assert_eq!(
            read_cache(&path, stale_key, &queries)
                .expect_err("stale key")
                .kind(),
            ErrorKind::InvalidData
        );

        let mut stale_queries = queries.clone();
        stale_queries[0][0] = f32::from_bits(stale_queries[0][0].to_bits() ^ 1);
        assert_eq!(
            read_cache(&path, key, &stale_queries)
                .expect_err("stale query")
                .kind(),
            ErrorKind::InvalidData
        );
    }
}
