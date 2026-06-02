//! Top-level superfile reader.
//!
//! `SuperfileReader::open(bytes)` parses the Parquet footer's `inf.*`
//! KV metadata, slices out the embedded FTS + vector blobs (zero-copy
//! via `Bytes`), and constructs the unified [`FtsReader`] +
//! [`VectorReader`] for query execution.
//!
//! ## Threading
//!
//! `Send + Sync`. Concurrent searches share the underlying `Bytes`.
//!
//! ## Section laziness
//!
//! Eager at the blob level (both blobs sliced once at `open()`), lazy
//! within each blob (per-(column,term) postings + per-cluster vector
//! codes are read on-demand by the underlying readers). The
//! single-segment SuperfileReader does no I/O after `open()`; a
//! storage layer can layer cold-fetch heuristics on top.

use crate::superfile::ReadError;
use crate::superfile::format::{self, footer, kv};
use crate::superfile::fts::reader::{BoolMode, FtsReader};
use crate::superfile::vector::reader::VectorReader;
use arrow_array::Decimal128Array;
use arrow_schema::Schema;
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::sync::Arc;

/// Per-open knobs for [`SuperfileReader::open_with`]. Defaults to
/// safe behavior (CRC verification on); flip `verify_crc` to `false`
/// to skip the ~132 ms scan at 1M × 384 when storage is trusted.
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Verify all CRC32C checksums on open: the embedded
    /// vector blob's whole-blob + per-subsection CRCs, and
    /// the embedded FTS blob's four per-section CRCs (FST,
    /// postings region, doc-lengths directory, per-column
    /// doc-lengths arrays). Defaults to `true`; the
    /// argumentless [`SuperfileReader::open`] uses this
    /// default. Flip to `false` only when the underlying
    /// storage is already trusted (e.g. a content-addressed
    /// object store that validates checksums on its own) to
    /// skip the checksum scan.
    pub verify_crc: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self { verify_crc: true }
    }
}

pub struct SuperfileReader {
    bytes: Bytes,
    schema: Arc<Schema>,
    id_column: String,
    n_docs: u64,
    fts: Option<FtsReader>,
    vec: Option<VectorReader>,
}

impl std::fmt::Debug for SuperfileReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SuperfileReader")
            .field("id_column", &self.id_column)
            .field("n_docs", &self.n_docs)
            .field("has_fts", &self.fts.is_some())
            .field("has_vec", &self.vec.is_some())
            .field("bytes_len", &self.bytes.len())
            .finish()
    }
}

impl SuperfileReader {
    /// Open from a complete superfile byte buffer (i.e. the bytes
    /// returned by `SuperfileBuilder::finish`, or read from disk / S3
    /// / etc.). CRC verification on by default; use [`open_with`]
    /// for the fast path on trusted storage.
    pub fn open(bytes: Bytes) -> Result<Self, ReadError> {
        Self::open_with(bytes, OpenOptions::default())
    }

    /// Open a superfile via a [`LazyByteSource`].
    ///
    /// Unlike [`open`], `open_lazy` does not require the caller
    /// to have materialized the full segment up-front. The
    /// source pulls bytes on demand. Today's implementation
    /// issues a single full-range fetch through the source
    /// (the polymorphism point), then constructs the reader
    /// through the existing `open()` path — so the per-call
    /// memory cost still scales with segment size. Per-query
    /// laziness ("≤ 3 ranges per query" — fetching only the
    /// footer + the touched FTS posting list + the touched
    /// vector cluster) requires teaching `FtsReader` +
    /// `VectorReader` to fetch posting / cluster bytes through
    /// the source on demand, which the trait shape already
    /// supports but the inner readers don't yet exercise.
    ///
    /// The async signature is what lets the supertable layer
    /// wrap the source with a broadcast / cold-fetch
    /// coordinator (see `ColdFetchMode`) — coalescing multiple
    /// concurrent cold readers onto a single set of
    /// range-GETs, or running the foreground reader in
    /// parallel with a background disk-cache fill.
    ///
    /// [`open`]: SuperfileReader::open
    pub async fn open_lazy(
        source: &dyn crate::superfile::LazyByteSource,
    ) -> Result<Self, ReadError> {
        Self::open_lazy_with(source, OpenOptions::default()).await
    }

    /// Like [`open_lazy`] but with explicit [`OpenOptions`].
    ///
    /// [`open_lazy`]: SuperfileReader::open_lazy
    pub async fn open_lazy_with(
        source: &dyn crate::superfile::LazyByteSource,
        opts: OpenOptions,
    ) -> Result<Self, ReadError> {
        let size = source.size();
        let bytes = source.range(0, size).await.map_err(|e| match e {
            crate::superfile::LazyByteSourceError::Storage(se) => {
                ReadError::MalformedKv(format!("lazy source storage: {se}"))
            }
            crate::superfile::LazyByteSourceError::OutOfBounds { start, len, size } => {
                ReadError::MalformedKv(format!(
                    "lazy source out-of-bounds: start={start} len={len} size={size}"
                ))
            }
        })?;
        Self::open_with(bytes, opts)
    }

    /// Open with explicit options. `OpenOptions { verify_crc: false }`
    /// skips both the whole-blob and per-subsection CRC scans — at
    /// 1M × 384 cold open drops from ~132 ms to ~2 ms. Use this when
    /// the underlying storage is trusted or CRC verification is
    /// performed elsewhere.
    pub fn open_with(bytes: Bytes, opts: OpenOptions) -> Result<Self, ReadError> {
        // 1. Read all KV metadata via the footer module.
        let kv_map = footer::read_kv_metadata(&bytes)?;

        // 2. Validate required keys + format version.
        for k in kv::REQUIRED {
            if !kv_map.contains_key(*k) {
                return Err(ReadError::MissingKv(k));
            }
        }
        let format_value = kv_map.get(kv::FORMAT).expect("checked above");
        if format_value != kv::FORMAT_VALUE {
            return Err(ReadError::MalformedKv(format!(
                "{} expected {:?}, got {:?}",
                kv::FORMAT,
                kv::FORMAT_VALUE,
                format_value
            )));
        }
        let version_str = kv_map.get(kv::FORMAT_VERSION).expect("checked above");
        let version = format::Version::parse(version_str)
            .ok_or_else(|| ReadError::MalformedVersion(version_str.clone()))?;
        if !version.is_compatible_with_current() {
            return Err(ReadError::UnsupportedVersion(version_str.clone()));
        }

        let id_column = kv_map.get(kv::ID_COLUMN).expect("checked above").clone();
        let n_docs: u64 = kv_map
            .get(kv::N_DOCS)
            .expect("checked above")
            .parse()
            .map_err(|_| ReadError::MalformedKv(format!("{} not a u64", kv::N_DOCS)))?;

        // 3. Read Arrow schema from the Parquet metadata via parquet-rs.
        //    Bytes implements ChunkReader directly, so this is zero-copy.
        let parq_builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .map_err(|e| ReadError::Footer(footer::FooterError::Parquet(e)))?;
        let schema = parq_builder.schema().clone();

        // 4. If FTS keys present, slice + open FtsReader.
        let fts = if all_present(&kv_map, kv::FTS_KEYS) {
            let off = parse_u64(&kv_map, kv::FTS_OFFSET)?;
            let len = parse_u64(&kv_map, kv::FTS_LENGTH)?;
            let cols_json = kv_map.get(kv::FTS_COLUMNS).expect("checked");
            let blob = slice_or_err(&bytes, off, len, "FTS")?;
            Some(FtsReader::open_with(
                blob,
                cols_json,
                crate::superfile::fts::reader::OpenOptions {
                    verify_crc: opts.verify_crc,
                },
            )?)
        } else if any_present(&kv_map, kv::FTS_KEYS) {
            return Err(ReadError::MalformedKv(
                "partial inf.fts.* keys present".into(),
            ));
        } else {
            None
        };

        // 5. Vector path mirrors FTS.
        let vec = if all_present(&kv_map, kv::VEC_KEYS) {
            let off = parse_u64(&kv_map, kv::VEC_OFFSET)?;
            let len = parse_u64(&kv_map, kv::VEC_LENGTH)?;
            let cols_json = kv_map.get(kv::VEC_COLUMNS).expect("checked");
            let blob = slice_or_err(&bytes, off, len, "vector")?;
            Some(VectorReader::open_with(
                blob,
                cols_json,
                crate::superfile::vector::reader::OpenOptions {
                    verify_crc: opts.verify_crc,
                },
            )?)
        } else if any_present(&kv_map, kv::VEC_KEYS) {
            return Err(ReadError::MalformedKv(
                "partial inf.vec.* keys present".into(),
            ));
        } else {
            None
        };

        Ok(Self {
            bytes,
            schema,
            id_column,
            n_docs,
            fts,
            vec,
        })
    }

    /// Arrow schema of the user-visible columns (Parquet rows).
    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// Name of the primary-key column (UInt64).
    pub fn id_column(&self) -> &str {
        &self.id_column
    }

    /// Total document count in this superfile.
    pub fn n_docs(&self) -> u64 {
        self.n_docs
    }

    /// FTS column names in declaration order, or empty.
    pub fn fts_columns(&self) -> Vec<&str> {
        match &self.fts {
            Some(r) => r.fts_columns().collect(),
            None => Vec::new(),
        }
    }

    /// Underlying FTS reader. `None` if this superfile has no FTS index.
    pub fn fts(&self) -> Option<&FtsReader> {
        self.fts.as_ref()
    }

    /// Vector column names in declaration order, or empty.
    pub fn vector_columns(&self) -> Vec<&str> {
        match &self.vec {
            Some(r) => r.vector_columns().collect(),
            None => Vec::new(),
        }
    }

    /// Underlying vector reader. `None` if this superfile has no vector index.
    pub fn vec(&self) -> Option<&VectorReader> {
        self.vec.as_ref()
    }

    /// Pass-through to the raw Parquet bytes — the superfile is a
    /// valid Parquet file, so this works as input to any external
    /// Parquet reader (DataFusion, DuckDB, pyarrow, …).
    pub fn parquet_bytes(&self) -> &Bytes {
        &self.bytes
    }

    /// Sequential scan of the `_id` column for an exact `target`
    /// match. Returns the matching row's local doc_id (the row
    /// offset within this superfile, used by tombstones / FTS /
    /// vector indices) or `None` if the target isn't present.
    ///
    /// `_id` is stored as `Decimal128` with the supertable's
    /// fixed precision/scale; we decode each value as `i128`.
    ///
    /// Used by the WAL recovery sweep's `resolve_target_id`
    /// path: given a `target_id`, scan the candidate superfile to
    /// find where the row lives so the tombstone phase can mark
    /// its bit. Currently rebuilds a `ParquetRecordBatchReader`
    /// each call — opportunity for a follow-up that caches
    /// parquet metadata on the reader (see TODO in module-level
    /// comments).
    pub fn id_lookup(&self, target: i128) -> Result<Option<u32>, ReadError> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(self.bytes.clone())
            .map_err(|e| ReadError::Footer(footer::FooterError::Parquet(e)))?;
        let reader = builder
            .build()
            .map_err(|e| ReadError::Footer(footer::FooterError::Parquet(e)))?;

        let mut row_offset: u32 = 0;
        for batch_res in reader {
            // `batch_res` is `Result<RecordBatch, ArrowError>`;
            // funnel through ParquetError so the whole id_lookup
            // path surfaces a single error variant.
            let batch =
                batch_res.map_err(|e| ReadError::Footer(footer::FooterError::Parquet(e.into())))?;
            let id_idx = batch.schema().index_of(&self.id_column).map_err(|_| {
                ReadError::MalformedKv(format!(
                    "id_column {:?} declared in KV metadata but missing from parquet schema",
                    self.id_column
                ))
            })?;
            let arr = batch.column(id_idx);
            let id_arr = arr
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| {
                    ReadError::MalformedKv(format!(
                        "id_column {:?} is not Decimal128",
                        self.id_column
                    ))
                })?;
            for i in 0..id_arr.len() {
                if id_arr.value(i) == target {
                    return Ok(Some(row_offset + i as u32));
                }
            }
            row_offset = row_offset.checked_add(id_arr.len() as u32).ok_or_else(|| {
                ReadError::MalformedKv("row_offset overflow scanning id column".to_string())
            })?;
        }
        Ok(None)
    }

    /// Single-column BM25 search across the unified FTS reader.
    ///
    /// `query` is tokenized by the same v1 tokenizer used at build
    /// time (`AsciiLowerTokenizer`). Returns `(local_doc_id, score)`
    /// ordered by descending score.
    pub fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
        let tok = crate::superfile::fts::tokenize::AsciiLowerTokenizer;
        use crate::superfile::fts::tokenize::Tokenizer as _;
        let term_strings: Vec<String> = tok.tokenize(query).collect();
        let term_refs: Vec<&str> = term_strings.iter().map(|s| s.as_str()).collect();
        self.bm25_search_pretokenized(column, &term_refs, k, mode)
    }

    /// Pre-tokenized variant of [`Self::bm25_search`] — the caller
    /// supplies the already-tokenized term slice and we skip the
    /// `AsciiLowerTokenizer` pass.
    ///
    /// Used by the supertable layer's fan-out: the cross-segment
    /// search tokenizes the query once at the orchestrator (to
    /// compute the bloom-skip mask) and then passes the same
    /// `terms` slice to every per-segment search, avoiding
    /// `(N+1)·T` redundant tokenizations across N segments and
    /// a T-token query.
    ///
    /// Terms must be already lower-cased ASCII alphanumeric tokens
    /// — the FST keys are stored in that form. Callers using the
    /// v1 tokenizer can produce them via
    /// `AsciiLowerTokenizer.tokenize(query)`.
    pub fn bm25_search_pretokenized(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.search(column, terms, k, mode)?)
    }

    /// Prefix-expanded BM25 search.
    ///
    /// Expands `prefix` to the lex-ordered list of indexed terms
    /// in `column` whose tokenized form begins with `prefix`,
    /// then runs `BoolMode::Or` BM25 over that term set. Matches
    /// the v1 tokenizer convention: the FST stores
    /// AsciiLowerTokenizer-tokenized terms, so the prefix is
    /// ASCII-lowercased before expansion. Whitespace inside
    /// `prefix` is **not** split — prefix search is a single
    /// term-level prefix, not a query parser.
    ///
    /// Returns an empty `Vec` if no indexed term begins with
    /// `prefix` or if `k == 0`.
    pub fn bm25_search_prefix(
        &self,
        column: &str,
        prefix: &str,
        k: usize,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let lowered = prefix.to_ascii_lowercase();
        let term_bytes = fts.iter_terms_with_prefix(column, lowered.as_bytes());
        if term_bytes.is_empty() {
            return Ok(Vec::new());
        }
        // FST keys are valid UTF-8 by construction (AsciiLower
        // tokenizer only emits ASCII bytes); the from_utf8 below
        // is a typed pass-through, not a re-validation cost.
        let term_strings: Vec<&str> = term_bytes
            .iter()
            .filter_map(|b| std::str::from_utf8(b).ok())
            .collect();
        Ok(fts.search(column, &term_strings, k, BoolMode::Or)?)
    }

    /// Multi-term OR BM25 search restricted to a doc_id sub-range.
    ///
    /// Mirrors [`Self::bm25_search_pretokenized`] in `BoolMode::Or`
    /// shape but only scores docs in `[doc_id_start, doc_id_end)`.
    /// Used by the supertable layer's intra-segment parallel
    /// fan-out: the supertable splits each segment into N
    /// equal-width sub-ranges, runs one call per sub-range in
    /// parallel on the reader pool, then merges the per-sub-range
    /// top-K heaps.
    ///
    /// Single-term inputs (`terms.len() == 1`) are not optimized
    /// here — they already finish in microseconds via
    /// [`Self::bm25_search_pretokenized`]; the supertable layer
    /// should keep them on the un-ranged path.
    pub fn bm25_search_or_range_pretokenized(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.search_or_range_pretokenized(column, terms, k, doc_id_start, doc_id_end)?)
    }

    /// Prefix-expanded BM25 search restricted to a doc_id sub-range.
    ///
    /// Same expansion logic as [`Self::bm25_search_prefix`] —
    /// AsciiLower the prefix, walk the FST for matching terms, run
    /// BM25 OR over the term set — but only docs in
    /// `[doc_id_start, doc_id_end)` are eligible. Used by the
    /// supertable layer's intra-segment parallel fan-out on prefix
    /// queries; the per-sub-range expansion is identical (same FST,
    /// same column) so each sub-range expands locally rather than
    /// passing pre-expanded terms across the task boundary.
    pub fn bm25_search_prefix_range(
        &self,
        column: &str,
        prefix: &str,
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        if k == 0 || doc_id_start >= doc_id_end {
            return Ok(Vec::new());
        }
        let lowered = prefix.to_ascii_lowercase();
        let term_bytes = fts.iter_terms_with_prefix(column, lowered.as_bytes());
        if term_bytes.is_empty() {
            return Ok(Vec::new());
        }
        let term_strings: Vec<&str> = term_bytes
            .iter()
            .filter_map(|b| std::str::from_utf8(b).ok())
            .collect();
        Ok(fts.search_or_range_pretokenized(column, &term_strings, k, doc_id_start, doc_id_end)?)
    }

    /// Multi-column BM25 search with per-column weights ("most
    /// fields" semantics: per-column scores summed by weight).
    pub fn bm25_search_multi(
        &self,
        columns: &[(&str, f32)],
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.search_multi(columns, query, k, mode)?)
    }

    /// Single-column vector kNN against a named vector index.
    ///
    /// `options` controls the recall-vs-latency tradeoff;
    /// [`VectorSearchOptions::new()`] (or `..Default::default()`)
    /// picks defaults that recover ≥0.9 recall@10 on typical IVF
    /// setups.
    ///
    /// Sync — every public surface in `src/` is sync. Per-range
    /// byte access routes through
    /// [`crate::superfile::vector::reader::Source::get_range`],
    /// which is itself sync and bridges to the underlying
    /// async `LazyByteSource::range` only on cold
    /// `Source::Lazy` misses (via `block_in_place +
    /// Handle::block_on`, same pattern as the
    /// supertable query path's per-segment reader). On
    /// `Source::InMemory` and on warm-cache `Source::Lazy`
    /// (`BytesLazyByteSource`, mmap-backed) every fetch
    /// resolves zero-copy on the sync fast path.
    pub fn vector_search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
        let v = self
            .vec()
            .ok_or_else(|| ReadError::MissingKv(kv::VEC_OFFSET))?;
        Ok(v.search(column, query, k, options.nprobe, options.rerank_mult)?)
    }
}

/// Tuning knobs for [`SuperfileReader::vector_search`]. Defaults
/// are picked so a caller who hasn't profiled the recall-vs-latency
/// tradeoff still gets recall in the 0.9+ range on typical IVF
/// setups.
///
/// - `nprobe`: number of IVF clusters to scan. Higher = better recall,
///   slower. Default `8`, internally clamped to `[1, n_cent]`. For a
///   typical `n_cent ≈ sqrt(n_docs)` setup this means 1/8th of the
///   index per query.
/// - `rerank_mult`: number of `k * rerank_mult` candidates passed
///   from the 1-bit RaBitQ shortlist into the full-precision rerank.
///   Higher = better recall, slower. Default `20`. With smaller
///   values (e.g. `5`) recall@10 drops to ~50% on a 10k×384 corpus
///   because the 1-bit estimate noise drops true neighbors out of
///   the shortlist before rerank can recover them — see
///   `tests/recall.rs` for the measurements behind the default.
///
/// The defaults are deliberately conservative; the bench harness
/// has the measured 1M / 10M recall-vs-latency curves and may
/// motivate a smaller default later.
#[derive(Debug, Clone, Copy)]
pub struct VectorSearchOptions {
    pub nprobe: usize,
    pub rerank_mult: usize,
}

impl VectorSearchOptions {
    /// Builder default: `nprobe = 8`, `rerank_mult = 20`.
    pub const DEFAULT_NPROBE: usize = 8;
    pub const DEFAULT_RERANK_MULT: usize = 20;

    /// Construct with both defaults applied.
    pub fn new() -> Self {
        Self {
            nprobe: Self::DEFAULT_NPROBE,
            rerank_mult: Self::DEFAULT_RERANK_MULT,
        }
    }

    /// Override the IVF probe count.
    pub fn with_nprobe(mut self, n: usize) -> Self {
        self.nprobe = n;
        self
    }

    /// Override the rerank candidate multiplier.
    pub fn with_rerank_mult(mut self, n: usize) -> Self {
        self.rerank_mult = n;
        self
    }
}

impl Default for VectorSearchOptions {
    fn default() -> Self {
        Self::new()
    }
}

fn all_present(map: &footer::KvMap, keys: &[&str]) -> bool {
    keys.iter().all(|k| map.contains_key(*k))
}

fn any_present(map: &footer::KvMap, keys: &[&str]) -> bool {
    keys.iter().any(|k| map.contains_key(*k))
}

fn parse_u64(map: &footer::KvMap, key: &'static str) -> Result<u64, ReadError> {
    map.get(key)
        .ok_or(ReadError::MissingKv(key))?
        .parse()
        .map_err(|_| ReadError::MalformedKv(format!("{key} not a u64")))
}

fn slice_or_err(
    bytes: &Bytes,
    off: u64,
    len: u64,
    section: &'static str,
) -> Result<Bytes, ReadError> {
    let off = off as usize;
    let len = len as usize;
    if off.saturating_add(len) > bytes.len() {
        return Err(ReadError::MalformedKv(format!(
            "{section} blob offset+len out of range"
        )));
    }
    Ok(bytes.slice(off..off + len))
}

// Re-export for convenience: callers want `BoolMode` without diving
// into the FTS submodule.
pub use crate::superfile::fts::reader::BoolMode as FtsBoolMode;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::builder::{BuilderOptions, FtsConfig, SuperfileBuilder};
    use crate::superfile::vector::distance::normalize;
    use crate::test_helpers::{decimal128_ids, default_tokenizer, default_vector_config};
    use arrow_array::{LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field};

    fn schema_with_text() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
        ]))
    }

    fn build_simple_fts_only_superfile() -> Bytes {
        let schema = schema_with_text();
        let opts = BuilderOptions::new(
            schema.clone(),
            "doc_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        );
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let ids = decimal128_ids(vec![10u64, 11, 12, 13]);
        let title = LargeStringArray::from(vec![
            "rust async runtime",
            "python data pipeline",
            "rust embedded system",
            "javascript web frontend",
        ]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(title)])
            .expect("build RecordBatch");
        b.add_batch(&batch, &[]).expect("add_batch");
        Bytes::from(b.finish().expect("finish builder"))
    }

    #[test]
    fn open_reports_n_docs_and_id_column() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        assert_eq!(r.n_docs(), 4);
        assert_eq!(r.id_column(), "doc_id");
    }

    #[test]
    fn open_exposes_arrow_schema() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let s = r.schema();
        assert_eq!(s.fields().len(), 2);
        assert_eq!(s.field(0).name(), "doc_id");
        assert_eq!(s.field(1).name(), "title");
    }

    #[test]
    fn open_reports_fts_columns_when_present() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let cols = r.fts_columns();
        assert_eq!(cols, vec!["title"]);
        assert!(r.vector_columns().is_empty());
        assert!(r.fts().is_some());
        assert!(r.vec().is_none());
    }

    #[test]
    fn bm25_search_finds_matching_docs() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let hits = r
            .bm25_search("title", "rust", 5, BoolMode::Or)
            .expect("BM25 search");
        // docs 0 and 2 contain "rust"; both should appear.
        let doc_ids: std::collections::HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(doc_ids.contains(&0));
        assert!(doc_ids.contains(&2));
    }

    #[test]
    fn bm25_search_errors_when_no_fts() {
        // Build a superfile with no FTS, no vec.
        let schema = schema_with_text();
        let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![], None);
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let ids = decimal128_ids(vec![1u64]);
        let title = LargeStringArray::from(vec!["x"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(title)])
            .expect("build RecordBatch");
        b.add_batch(&batch, &[]).expect("add_batch");
        let bytes = Bytes::from(b.finish().expect("finish builder"));
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let err = r
            .bm25_search("nope", "x", 1, BoolMode::Or)
            .expect_err("expected error");
        assert!(matches!(err, ReadError::MissingKv(_)));
    }

    #[test]
    fn open_rejects_non_parquet_bytes() {
        let err = SuperfileReader::open(Bytes::from(vec![0u8; 16])).expect_err("expected error");
        assert!(matches!(err, ReadError::Footer(_)));
    }

    #[test]
    fn open_rejects_parquet_without_inf_format_kv() {
        // Hand-build a Parquet file with no inf.* keys; it should fail
        // with MissingKv (inf.format).
        use crate::superfile::format::footer::write_parquet_with_blobs;
        use parquet::basic::Compression;
        let schema = schema_with_text();
        let ids = decimal128_ids(vec![1u64]);
        let title = LargeStringArray::from(vec!["x"]);
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(title)])
            .expect("build RecordBatch");
        let parts =
            write_parquet_with_blobs(&schema, &[batch], &[], &[], &[], Compression::SNAPPY, 1024)
                .expect("write parquet with blobs");
        let err = SuperfileReader::open(Bytes::from(parts.bytes)).expect_err("expected error");
        assert!(matches!(err, ReadError::MissingKv(_)));
    }

    fn build_vector_only_superfile() -> Bytes {
        let schema = schema_with_text();
        let opts = BuilderOptions::new(
            schema.clone(),
            "doc_id",
            vec![],
            vec![default_vector_config("emb", 7)],
            None,
        );
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        // 4 unit-norm vectors so cosine is well-defined.
        let mut flat = Vec::<f32>::new();
        for i in 0..4u32 {
            let mut v = vec![0.0f32; 16];
            v[(i % 16) as usize] = 1.0;
            v[((i + 3) % 16) as usize] = 0.5;
            normalize(&mut v);
            flat.extend_from_slice(&v);
        }
        let ids = decimal128_ids(vec![100u64, 101, 102, 103]);
        let title = LargeStringArray::from(vec!["a", "b", "c", "d"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(title)])
            .expect("build RecordBatch");
        b.add_batch(&batch, &[flat.as_slice()]).expect("add_batch");
        Bytes::from(b.finish().expect("finish builder"))
    }

    #[test]
    fn open_loads_vector_reader_when_blob_present() {
        let bytes = build_vector_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        assert!(r.fts().is_none());
        assert!(r.vec().is_some());
        assert_eq!(r.vector_columns(), vec!["emb"]);
    }

    #[test]
    fn vector_search_options_default_values() {
        let opts = VectorSearchOptions::default();
        assert_eq!(opts.nprobe, 8);
        assert_eq!(opts.rerank_mult, 20);
        let opts2 = VectorSearchOptions::new();
        assert_eq!(opts.nprobe, opts2.nprobe);
        assert_eq!(opts.rerank_mult, opts2.rerank_mult);
    }

    #[test]
    fn vector_search_options_builder_chains() {
        let opts = VectorSearchOptions::new()
            .with_nprobe(2)
            .with_rerank_mult(50);
        assert_eq!(opts.nprobe, 2);
        assert_eq!(opts.rerank_mult, 50);
    }

    #[test]
    fn vector_search_with_default_options_succeeds() {
        // Confirms the default options path actually executes without
        // panicking; the recall is exercised in tests/recall.rs.
        let bytes = build_vector_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let mut q = vec![0.0f32; 16];
        q[2] = 1.0;
        q[5] = 0.5;
        normalize(&mut q);
        let hits = r
            .vector_search("emb", &q, 1, VectorSearchOptions::default())
            .expect("vector search");
        assert!(!hits.is_empty());
    }

    #[test]
    fn vector_search_finds_self() {
        let bytes = build_vector_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        // Query equal to doc 2's vector → top hit must be doc 2.
        let mut q = vec![0.0f32; 16];
        q[2] = 1.0;
        q[5] = 0.5;
        normalize(&mut q);
        let hits = r
            .vector_search(
                "emb",
                &q,
                1,
                VectorSearchOptions::new()
                    .with_nprobe(4)
                    .with_rerank_mult(5),
            )
            .expect("vector search");
        assert_eq!(hits[0].0, 2);
    }

    #[test]
    fn parquet_bytes_round_trips_as_parquet() {
        // The whole buffer should still be a valid Parquet file.
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let p = r.parquet_bytes().clone();
        let builder = ParquetRecordBatchReaderBuilder::try_new(p)
            .expect("try_new ParquetRecordBatchReaderBuilder");
        let mut reader = builder.build().expect("build parquet reader");
        let batch = reader.next().expect("one batch").expect("decode batch");
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 2);
    }

    #[test]
    fn unknown_column_in_search_propagates_fts_error() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let err = r
            .bm25_search("nonexistent", "rust", 5, BoolMode::Or)
            .expect_err("expected error");
        assert!(matches!(err, ReadError::Fts(_)));
    }

    #[test]
    fn bm25_search_multi_combines_columns() {
        // Build a 2-FTS-column file.
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("body", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(
            schema.clone(),
            "doc_id",
            vec![
                FtsConfig {
                    column: "title".into(),
                },
                FtsConfig {
                    column: "body".into(),
                },
            ],
            vec![],
            Some(default_tokenizer()),
        );
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let ids = decimal128_ids(vec![1u64, 2, 3]);
        let title = LargeStringArray::from(vec!["rust", "python", "go"]);
        let body = LargeStringArray::from(vec!["systems", "rust ml", "concurrency"]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(title), Arc::new(body)])
                .expect("build RecordBatch");
        b.add_batch(&batch, &[]).expect("add_batch");
        let bytes = Bytes::from(b.finish().expect("finish builder"));
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let hits = r
            .bm25_search_multi(&[("title", 1.0), ("body", 1.0)], "rust", 3, BoolMode::Or)
            .expect("BM25 multi-column search");
        // Both doc 0 (title:rust) and doc 1 (body:rust) hit.
        let doc_ids: std::collections::HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(doc_ids.contains(&0));
        assert!(doc_ids.contains(&1));
    }
}
