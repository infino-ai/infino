//! FTS blob reader. Multi-column BM25 search.
//!
//! Opens the byte layout produced by [`super::builder::FtsBuilder::finish`]
//! and exposes BM25 search per-column or weighted across columns.
//!
//! See `docs/architecture/superfile.md` for the on-disk layout.
//!
//! ## Threading
//!
//! `FtsReader` is `Send + Sync` and immutable after `open()` — concurrent
//! `search` calls share the underlying `Bytes`. The DictReader is
//! constructed per call (cheap; the FST validates its header in O(1) and
//! then it's a borrowed view).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use serde::Deserialize;

use crate::superfile::format::checksum::crc32c;
use crate::superfile::format::{self, FST_SEPARATOR};
use crate::superfile::fts::builder::{DOC_LENGTHS_ENTRY_SIZE, SKIP_ENTRY_SIZE, TERM_META_SIZE};
use crate::superfile::fts::dict::{DictReader, make_key};
use crate::superfile::fts::fst_value::FstValue;
use crate::superfile::fts::posting::{BLOCK_LEN, decode_block};
use crate::superfile::lazy_source::{LazyByteSource, PrefetchedSource, Source};
use crate::superfile::{ReadError, error::FtsError};

/// Boolean-mode for multi-term queries.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BoolMode {
    /// All query terms must match the doc.
    And,
    /// Any query term matching contributes to the doc's score.
    Or,
}

/// Multi-term OR algorithm selector for the bench harness's
/// `search_with_algo_for_bench` entry point. Production code routes
/// through `FtsReader::dispatch_multi_term_or`, which picks
/// automatically; this enum exists so head-to-head bench runs can
/// compare all three under identical inputs.
#[doc(hidden)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum OrAlgo {
    /// Block-Max MaxScore: production default for dominant-term ORs.
    Bmm,
    /// WAND + Block-Max-WAND: historical baseline; retained for
    /// regression comparisons.
    WandBmw,
    /// Exhaustive union walk with SIMD scoring + top-K heap. Wins
    /// when no term dominates (uniform `term_max_bm25` upper bounds)
    /// so BMM/BMW's skip checks rarely trigger and become pure
    /// overhead.
    Exhaustive,
}

/// Per-column metadata, indexed by column_id (declaration order).
#[derive(Debug, Clone)]
pub struct ColumnMeta {
    pub name: String,
    /// Byte range into [`FtsReader::blob`] holding this column's
    /// `u32` doc-lengths array (4 bytes per doc, length × n_docs).
    pub doc_lengths_range: Range<usize>,
    /// Average doc length across this column. `0.0` if the column has
    /// no docs.
    pub avgdl: f32,
    /// Precomputed BM25 denominator constant per doc:
    /// `dl_norm_k1[d] = K1 * (1 - B + B * dl[d] / avgdl)`. The hot
    /// scoring loop multiplies-out to `idf * tf * (K1+1) / (tf +
    /// dl_norm_k1[d])`, so each scoring call shaves a load + mul +
    /// add + mul vs recomputing on the fly. Computed once per
    /// reader at `open` time.
    pub dl_norm_k1: Vec<f32>,
}

/// JSON-deserialized form of one entry in `inf.fts.columns`. The KV
/// value is a JSON array of these, in declaration order.
#[derive(Debug, Clone, Deserialize)]
pub struct FtsColumnConfig {
    pub name: String,
    /// Currently always `"ascii_lower"`. A missing field
    /// deserializes to `"ascii_lower"` too — the only
    /// tokenizer that has ever existed for this format, so
    /// any file written without the field can only have
    /// been emitted with it implicitly.
    #[serde(default = "default_tokenizer")]
    pub tokenizer: String,
}

fn default_tokenizer() -> String {
    "ascii_lower".to_string()
}

/// Per-open knobs for [`FtsReader::open_with`]. Mirrors the
/// vector reader's `OpenOptions` so the superfile layer can
/// pass a single `verify_crc` flag through to both
/// sub-readers.
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Verify the four per-section CRC32C checks (FST,
    /// postings region, doc-lengths directory, per-column
    /// doc-lengths arrays). Defaults to `true`; flip to
    /// `false` only when the underlying storage already
    /// validates checksums (content-addressed object
    /// store, ZFS, etc.) to skip the scan on cold open.
    pub verify_crc: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self { verify_crc: true }
    }
}

impl OpenOptions {
    pub fn for_object_store() -> Self {
        Self { verify_crc: false }
    }
}

/// FTS blob reader. Self-contained — owns its `Bytes` (which the storage
/// layer assembled from mmap / range-fetch / full-read).
#[derive(Debug)]
pub struct FtsReader {
    source: Source,
    n_docs: u32,
    n_terms_total: u32,
    fst_range: Range<usize>,
    postings_range: Range<usize>,
    columns: Vec<ColumnMeta>,
    column_id_by_name: HashMap<String, u32>,
}

impl FtsReader {
    /// Open with default options (CRC verification on).
    pub fn open(blob: Bytes, columns_json: &str) -> Result<Self, FtsError> {
        Self::open_with(blob, columns_json, OpenOptions::default())
    }

    /// Open with explicit options. Pass
    /// `OpenOptions { verify_crc: false }` to skip the
    /// four per-section CRC scans on trusted-storage cold
    /// opens.
    pub fn open_with(blob: Bytes, columns_json: &str, opts: OpenOptions) -> Result<Self, FtsError> {
        Self::open_with_source(Source::InMemory(blob), columns_json, opts)
    }

    /// Open from a range source without materializing the FTS
    /// subsection. Three open-time GETs prefetch the only regions a
    /// reader needs before it can serve queries: the fixed header, the
    /// FST term directory (contiguous after the header), and the
    /// doc-length tables (the trailing region, needed to build BM25
    /// normalization). The postings region stays lazy — each query
    /// term's bytes are fetched on demand by [`Self::fetch_term_postings`],
    /// mirroring how the vector reader fetches only probed clusters.
    pub async fn open_lazy(
        source: Arc<dyn LazyByteSource>,
        columns_json: &str,
        opts: OpenOptions,
    ) -> Result<Self, FtsError> {
        // Length of the FTS subsection itself (≈ `kv::FTS_LENGTH`), not
        // the whole superfile: `source` is the FTS-scoped sub-source.
        let fts_blob_len = source.size() as usize;
        let header = fetch_lazy_range(source.as_ref(), 0..48, "fts header").await?;
        if &header[0..8] != format::fts::MAGIC {
            return Err(FtsError::Read(ReadError::BadMagic {
                section: "fts",
                expected: format::fts::MAGIC,
                actual: header[0..8].to_vec(),
            }));
        }
        let version = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
        if version != format::fts::VERSION {
            return Err(FtsError::Read(ReadError::UnsupportedVersion(format!(
                "fts section version {version}"
            ))));
        }

        let postings_offset = read_u64_le(&header[32..40]) as usize;
        let doc_lengths_table_offset = read_u64_le(&header[40..48]) as usize;

        // Prefetch the FST directory ([48..postings_offset], contiguous
        // after the header) so every later `dict_bytes()` resolves from
        // the overlay instead of a fresh GET per search, and the
        // doc-length tail ([doc_lengths_table_offset..fts_blob_len]) so
        // `open_with_source` builds its BM25 norm tables without
        // touching the source again. The doc-lengths region is the
        // *trailing* region of the FTS blob (it follows the postings),
        // so `..fts_blob_len` is the tail — directory + every per-column
        // doc-length array + their CRCs — fetched in one range GET, not
        // the whole blob (the FST is a separate range above; postings
        // stay lazy).
        //
        // Both ranges are known exactly once the header is parsed and
        // neither depends on the other, so they fire **concurrently**:
        // the FTS open spends 2 serial RTTs (header, then this parallel
        // pair) instead of 3. On a warm/in-memory source both resolve
        // through the sync zero-copy path at no cost. The doc-length
        // tail is fetched whole (one range) rather than dir-then-arrays,
        // keeping the open-time GET count minimal and avoiding
        // per-column range calls during metadata decode.
        let (fst_region, doc_lengths_tail) = futures::try_join!(
            fetch_lazy_range(source.as_ref(), 48..postings_offset, "fts/dict"),
            fetch_lazy_range(
                source.as_ref(),
                doc_lengths_table_offset..fts_blob_len,
                "fts/doc_lengths_tail",
            ),
        )?;

        let mut overlay = PrefetchedSource::new(source);
        overlay.install(0, header);
        overlay.install(48, fst_region);
        overlay.install(doc_lengths_table_offset as u64, doc_lengths_tail);

        Self::open_with_source(Source::Lazy(Arc::new(overlay)), columns_json, opts)
    }

    /// Open over an arbitrary byte source. The eager path wraps a
    /// full subsection as [`Source::InMemory`]; lazy callers can pass
    /// a range-backed source without changing the public search API.
    pub fn open_with_source(
        source: Source,
        columns_json: &str,
        opts: OpenOptions,
    ) -> Result<Self, FtsError> {
        let source_len = source.len();
        if source_len < 48 {
            return Err(FtsError::Read(ReadError::MissingKv("fts header")));
        }
        let header = fetch_source_range(&source, 0..48, "fts header")?;

        // Magic check.
        if &header[0..8] != format::fts::MAGIC {
            return Err(FtsError::Read(ReadError::BadMagic {
                section: "fts",
                expected: format::fts::MAGIC,
                actual: header[0..8].to_vec(),
            }));
        }

        // Version check.
        let version = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
        if version != format::fts::VERSION {
            return Err(FtsError::Read(ReadError::UnsupportedVersion(format!(
                "fts section version {version}"
            ))));
        }

        let n_columns =
            u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;
        let n_docs = read_u32_le(&header[16..20]);
        let n_terms_total = read_u32_le(&header[20..24]);
        let fst_offset = read_u64_le(&header[24..32]) as usize;
        let postings_offset = read_u64_le(&header[32..40]) as usize;
        let doc_lengths_table_offset = read_u64_le(&header[40..48]) as usize;

        // Bounds-check every offset against the blob length before
        // any slice indexing. A single byte flip in the header can
        // corrupt these into multi-GB values; without this check
        // they propagate as out-of-range slice indices and panic
        // before the CRC verification can reject the corruption.
        //
        // The `< +4` checks (rather than `<= +4`) admit the legal
        // empty-region case: when every term takes the df=1 inline-FST
        // short-circuit, the postings region body is zero bytes and
        // only the trailing 4-byte CRC32C(empty) sits between
        // `postings_offset` and `doc_lengths_table_offset`.
        if fst_offset < 48
            || postings_offset < fst_offset + 4
            || doc_lengths_table_offset < postings_offset + 4
            || doc_lengths_table_offset > source_len
        {
            return Err(FtsError::Read(ReadError::MalformedVersion(format!(
                "fts header offsets out of range: fst={fst_offset}, postings={postings_offset}, \
                 doc_lengths={doc_lengths_table_offset}, blob_len={}",
                source_len
            ))));
        }

        // Postings region length: we don't store it explicitly (CRC32C of
        // the body is at postings_offset + len - 4). Compute from the
        // surrounding offsets — postings ends where the doc-lengths
        // directory begins.
        let fst_range = fst_offset..postings_offset.saturating_sub(4); // strip CRC
        let postings_range = postings_offset..doc_lengths_table_offset.saturating_sub(4); // strip CRC

        // Verify FST CRC32C (4 bytes after fst body).
        if opts.verify_crc {
            let fst_crc_bytes = fetch_source_range(
                &source,
                postings_offset.saturating_sub(4)..postings_offset,
                "fts/dict crc",
            )?;
            let fst_crc_expected = read_u32_le(&fst_crc_bytes);
            let fst_bytes = fetch_source_range(&source, fst_range.clone(), "fts/dict")?;
            let fst_crc_actual = crc32c(&fst_bytes);
            if fst_crc_expected != fst_crc_actual {
                return Err(FtsError::Read(ReadError::ChecksumMismatch {
                    section: "fts/dict",
                    column: String::new(),
                }));
            }
        }

        // Verify postings region CRC32C.
        if opts.verify_crc {
            let postings_crc_pos = doc_lengths_table_offset.saturating_sub(4);
            let postings_crc_bytes = fetch_source_range(
                &source,
                postings_crc_pos..doc_lengths_table_offset,
                "fts/postings crc",
            )?;
            let postings_crc_expected = read_u32_le(&postings_crc_bytes);
            let postings_bytes =
                fetch_source_range(&source, postings_range.clone(), "fts/postings")?;
            let postings_crc_actual = crc32c(&postings_bytes);
            if postings_crc_expected != postings_crc_actual {
                return Err(FtsError::Read(ReadError::ChecksumMismatch {
                    section: "fts/postings",
                    column: String::new(),
                }));
            }
        }

        // Parse columns_json.
        let cols: Vec<FtsColumnConfig> = serde_json::from_str(columns_json).map_err(|e| {
            FtsError::Read(ReadError::MalformedVersion(format!(
                "inf.fts.columns JSON: {e}"
            )))
        })?;
        if cols.len() != n_columns {
            return Err(FtsError::Read(ReadError::MalformedVersion(format!(
                "inf.fts.columns has {} entries, header says {}",
                cols.len(),
                n_columns
            ))));
        }

        // Read doc-lengths directory: n_columns × 16-byte entries + 4-byte CRC.
        //
        // On the lazy open path this directory — and every per-column
        // array fetched below — falls inside the
        // `[doc_lengths_table_offset..fts_blob_len]` tail that
        // `open_lazy` already fetched in one GET and installed in the
        // overlay, so these `fetch_source_range` calls resolve from the
        // overlay with **no** per-column GETs. On the eager path the
        // whole subsection is in memory, so they are zero-copy slices.
        let dir_size = n_columns * DOC_LENGTHS_ENTRY_SIZE;
        let dir_end = doc_lengths_table_offset + dir_size;
        if dir_end + 4 > source_len {
            return Err(FtsError::Read(ReadError::MalformedVersion(
                "doc-lengths directory runs past blob end".into(),
            )));
        }
        let dir_region = fetch_source_range(
            &source,
            doc_lengths_table_offset..dir_end + 4,
            "fts/doc_lengths_dir",
        )?;
        let dir_bytes = &dir_region[..dir_size];
        if opts.verify_crc {
            let dir_crc_expected = read_u32_le(&dir_region[dir_size..dir_size + 4]);
            let dir_crc_actual = crc32c(dir_bytes);
            if dir_crc_expected != dir_crc_actual {
                return Err(FtsError::Read(ReadError::ChecksumMismatch {
                    section: "fts/doc_lengths_dir",
                    column: String::new(),
                }));
            }
        }

        // Build ColumnMeta vec + column_id_by_name.
        let mut columns = Vec::with_capacity(n_columns);
        let mut column_id_by_name = HashMap::with_capacity(n_columns);
        for (i, col_cfg) in cols.iter().enumerate() {
            let entry_off = i * DOC_LENGTHS_ENTRY_SIZE;
            let column_id = u32::from_le_bytes([
                dir_bytes[entry_off],
                dir_bytes[entry_off + 1],
                dir_bytes[entry_off + 2],
                dir_bytes[entry_off + 3],
            ]);
            let doc_lengths_offset =
                read_u64_le(&dir_bytes[entry_off + 4..entry_off + 12]) as usize;
            let avgdl_x1000 = read_u32_le(&dir_bytes[entry_off + 12..entry_off + 16]) as u64;

            // Verify column_id matches the JSON's positional column_id.
            if column_id != i as u32 {
                return Err(FtsError::Read(ReadError::MalformedVersion(format!(
                    "doc-lengths directory entry {i} has column_id {column_id}"
                ))));
            }

            // Per-column doc-lengths array: 4 * n_docs bytes + 4-byte CRC.
            // `doc_lengths_offset` lies within the prefetched doc-lengths
            // tail, so on the lazy path this resolves from the overlay
            // (see the directory comment above) — no per-column GET.
            let array_byte_len = 4 * n_docs as usize;
            let array_end = doc_lengths_offset + array_byte_len;
            if array_end + 4 > source_len {
                return Err(FtsError::Read(ReadError::MalformedVersion(format!(
                    "doc-lengths array {i} runs past blob end"
                ))));
            }
            let array_region = fetch_source_range(
                &source,
                doc_lengths_offset..array_end + 4,
                "fts/doc_lengths_array",
            )?;
            if opts.verify_crc {
                let array_crc_expected =
                    read_u32_le(&array_region[array_byte_len..array_byte_len + 4]);
                let array_crc_actual = crc32c(&array_region[..array_byte_len]);
                if array_crc_expected != array_crc_actual {
                    return Err(FtsError::Read(ReadError::ChecksumMismatch {
                        section: "fts/doc_lengths_array",
                        column: format!(" (column '{}')", col_cfg.name),
                    }));
                }
            }

            let avgdl = (avgdl_x1000 as f32) / 1000.0;
            // Precompute per-doc length normalizer:
            //   dl_norm_k1[d] = K1 * (1 - B + B * dl[d] / avgdl)
            // For avgdl == 0 (empty column) leave the table empty;
            // it'll never be indexed since `search` short-circuits.
            let mut dl_norm_k1 = Vec::with_capacity(n_docs as usize);
            if avgdl > 0.0 {
                let inv_avgdl = 1.0_f32 / avgdl;
                for d in 0..(n_docs as usize) {
                    let dl = read_u32_le(&array_region[d * 4..d * 4 + 4]) as f32;
                    let norm = 1.0 - crate::superfile::fts::bm25::B
                        + crate::superfile::fts::bm25::B * dl * inv_avgdl;
                    dl_norm_k1.push(crate::superfile::fts::bm25::K1 * norm);
                }
            }
            columns.push(ColumnMeta {
                name: col_cfg.name.clone(),
                doc_lengths_range: doc_lengths_offset..array_end,
                avgdl,
                dl_norm_k1,
            });
            column_id_by_name.insert(col_cfg.name.clone(), i as u32);
        }

        Ok(FtsReader {
            source,
            n_docs,
            n_terms_total,
            fst_range,
            postings_range,
            columns,
            column_id_by_name,
        })
    }

    pub fn n_docs(&self) -> u32 {
        self.n_docs
    }

    pub fn n_terms(&self) -> u32 {
        self.n_terms_total
    }

    /// FTS column names in declaration order.
    pub fn fts_columns(&self) -> impl Iterator<Item = &str> {
        self.columns.iter().map(|c| c.name.as_str())
    }

    fn dict_bytes(&self) -> Result<Bytes, FtsError> {
        fetch_source_range(&self.source, self.fst_range.clone(), "fts/dict")
    }

    /// Async FST-dictionary fetch for the query path. Resolves
    /// zero-copy for in-memory / warm sources; for a cold `Lazy`
    /// source it `await`s the object-store range on the caller's
    /// runtime (no sync bridge).
    async fn dict_bytes_async(&self) -> Result<Bytes, FtsError> {
        self.source
            .range_async(self.fst_range.clone())
            .await
            .map_err(|e| {
                FtsError::Read(ReadError::MalformedVersion(format!(
                    "fts/dict range fetch failed: {e}"
                )))
            })
    }

    /// Fetch the complete byte range of each requested term — metadata
    /// header (20 bytes) + skip table + encoded posting blocks — in
    /// parallel. `terms` are `(metadata_offset, postings_length)` pairs
    /// stored in the FST (`FstValue::Pfor`); the
    /// returned `Bytes` for term `i` starts at that term's metadata
    /// header (offset 0) and runs to the end of its last block, so a
    /// `TermCursor` can index it directly.
    ///
    /// This is the FTS analog of the vector reader's per-probed-cluster
    /// `Source::get_ranges_parallel` fan-out: a query only ever pulls
    /// the bytes of the terms it actually scores, never the whole
    /// postings region. On an in-memory source every range resolves as
    /// a zero-copy slice; on a lazy (object-store) source the cold
    /// ranges are coalesced under one async bridge and returned in
    /// input order.
    ///
    /// Because the FST value carries the length, this is a single
    /// range batch. The metadata header remains in the returned bytes
    /// for validation and cursor construction.
    async fn fetch_term_postings(&self, terms: &[(usize, usize)]) -> Result<Vec<Bytes>, FtsError> {
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let base = self.postings_range.start;
        let region_len = self.postings_range.len();

        let mut ranges: Vec<Range<usize>> = Vec::with_capacity(terms.len());
        for &(m, postings_length) in terms {
            if postings_length < 20 || m + postings_length > region_len {
                return Err(FtsError::Read(ReadError::MalformedVersion(
                    "term postings range runs past postings region".into(),
                )));
            }
            ranges.push(base + m..base + m + postings_length);
        }
        self.source
            .get_ranges_parallel_async(&ranges)
            .await
            .map_err(|e| {
                FtsError::Read(ReadError::MalformedVersion(format!(
                    "fts/postings term body range fetch failed: {e}"
                )))
            })
    }

    /// Resolve a column name to its dense column_id, or
    /// `FtsError::UnknownColumn` if the column isn't FTS-indexed in
    /// this segment. Shared by every public search entry point.
    fn resolve_column_id(&self, column: &str) -> Result<u32, FtsError> {
        self.column_id_by_name
            .get(column)
            .copied()
            .ok_or_else(|| FtsError::UnknownColumn(column.to_string()))
    }

    /// Walk the FST and collect every term registered under
    /// `column`, in lex order. Used to populate per-segment FTS
    /// skip-pruning summaries (term-presence bloom + lex term
    /// range) at commit time.
    ///
    /// Returns an empty `Vec` if `column` is not registered as
    /// an FTS column in this segment. Cost is O(terms in column)
    /// FST decodes; intended to be called once per (segment,
    /// column) at commit time, not on the query hot path.
    pub fn iter_column_terms(&self, column: &str) -> Result<Vec<Vec<u8>>, FtsError> {
        self.iter_terms_with_prefix(column, b"")
    }

    /// Walk the FST and collect every term registered under
    /// `column` whose bytes begin with `term_prefix`, in lex order.
    ///
    /// Mirrors [`Self::iter_column_terms`] but bounds the walk to a
    /// prefix range instead of the whole column. Used by
    /// [`SuperfileReader::bm25_search_prefix`] to expand a
    /// prefix into the concrete terms list before delegating to
    /// `search` in OR mode.
    ///
    /// `term_prefix` is the prefix as it appears in the FST — the
    /// caller is responsible for any tokenizer-level normalization
    /// (e.g. ASCII-lowercasing for the v1 tokenizer). Returns an
    /// empty `Vec` if `column` is not registered or no terms match
    /// the prefix.
    pub fn iter_terms_with_prefix(
        &self,
        column: &str,
        term_prefix: &[u8],
    ) -> Result<Vec<Vec<u8>>, FtsError> {
        if !self.column_id_by_name.contains_key(column) {
            return Ok(Vec::new());
        }
        let mut full_prefix = column.as_bytes().to_vec();
        full_prefix.push(FST_SEPARATOR);
        let column_prefix_len = full_prefix.len();
        full_prefix.extend_from_slice(term_prefix);
        let fst_bytes = self
            .dict_bytes()
            .expect("FST bytes must be available for term iteration");
        let dict = DictReader::open(&fst_bytes).map_err(|e| {
            FtsError::Read(ReadError::MalformedVersion(format!(
                "FST parse failed: {e}"
            )))
        })?;
        let pairs = dict.iter_prefix(&full_prefix);
        Ok(pairs
            .into_iter()
            .map(|(key, _)| key[column_prefix_len..].to_vec())
            .collect())
    }

    /// Single-column BM25 search.
    ///
    /// `terms` are the *already-tokenized* query terms — caller-tokenized
    /// to match the column's tokenizer. The format currently uses one
    /// tokenizer for all columns, so callers can use the same tokenizer
    /// that was used for indexing.
    pub async fn search(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let column_id = self.resolve_column_id(column)?;

        if terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        // Single-term fast path: BlockMaxWAND-driven block skipping.
        // Walks blocks in order, populating a top-k min-heap. Once the
        // heap is full, blocks whose skip-table-recorded `max_bm25`
        // can't beat the heap's worst score are skipped without
        // decoding.
        if terms.len() == 1 {
            return self.search_single_term_bmw(column_id, terms[0], k).await;
        }

        // Multi-term routing:
        //   OR  → MaxScore+BMM via `dispatch_multi_term_or`. WAND+BMW
        //         remains in-tree for `search_with_algo_for_bench` but
        //         is not on the production path; see the routing-
        //         decision table on `dispatch_multi_term_or`.
        //   AND → leapfrog intersection over the skip table via
        //         `run_and_intersect`. Both share cursor construction
        //         with the OR path so neither pays for cursor work
        //         twice when the bench harness compares them.
        match mode {
            BoolMode::Or => self.dispatch_multi_term_or(column_id, terms, k).await,
            BoolMode::And => {
                // Build cursors; if any term is missing, the
                // intersection is empty.
                let cursors = self.build_term_cursors(column_id, terms).await?;
                if cursors.len() != terms.len() {
                    return Ok(Vec::new());
                }
                self.run_and_intersect(column_id, cursors, k)
            }
        }
    }

    /// Multi-term OR BM25 search constrained to a doc_id sub-range.
    ///
    /// Same scoring semantics as [`Self::search`] in `BoolMode::Or`
    /// for the multi-term case, but only docs whose id falls within
    /// `[doc_id_start, doc_id_end)` are eligible. Used by the
    /// supertable's intra-segment parallel fan-out: when the reader
    /// pool has more threads than segments, each segment is sliced
    /// into N equal-width doc-id sub-ranges and one task per
    /// sub-range runs here in parallel; the caller merges the
    /// per-sub-range top-K heaps.
    ///
    /// Returns `Ok(Vec::new())` for `terms.is_empty()`, `k == 0`, or
    /// a degenerate range (`doc_id_start >= doc_id_end`).
    ///
    /// Single-term inputs (`terms.len() == 1`) are NOT
    /// sub-range-optimized here — single-term queries already
    /// complete in microseconds via [`Self::search`]'s BMW path; the
    /// supertable layer should keep them on the un-ranged call. The
    /// implementation delegates to
    /// [`Self::run_max_score_bmm_range`] which seeks every cursor
    /// to `doc_id_start` and breaks the outer loop when the next
    /// candidate doc_id reaches `doc_id_end`.
    pub async fn search_or_range_pretokenized(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let column_id = self.resolve_column_id(column)?;
        if terms.is_empty() || k == 0 || doc_id_start >= doc_id_end {
            return Ok(Vec::new());
        }
        let cursors = self.build_term_cursors(column_id, terms).await?;
        if cursors.is_empty() {
            return Ok(Vec::new());
        }
        self.run_max_score_bmm_range(column_id, cursors, k, doc_id_start, doc_id_end)
    }

    /// Multi-column BM25 search (most_fields semantics): each
    /// `(column, weight)` runs an OR-mode search; per-column scores are
    /// multiplied by `weight` and summed across columns.
    pub async fn search_multi(
        &self,
        columns: &[(&str, f32)],
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        // One tokenizer for all columns; per-column tokenizers would
        // require splitting this call to use the column's configured
        // tokenizer.
        let tok = crate::superfile::fts::tokenize::AsciiLowerTokenizer;
        use crate::superfile::fts::tokenize::Tokenizer as _;
        let term_strings: Vec<String> = tok.tokenize(query).collect();
        let term_refs: Vec<&str> = term_strings.iter().map(|s| s.as_str()).collect();

        let mut combined: HashMap<u32, f32> = HashMap::new();
        for (col_name, weight) in columns {
            let per_col = self.search(col_name, &term_refs, usize::MAX, mode).await?;
            for (doc_id, s) in per_col {
                *combined.entry(doc_id).or_insert(0.0) += s * weight;
            }
        }
        Ok(top_k(combined, k))
    }

    /// Single-term BM25 search with BlockMaxWAND-driven block skipping.
    ///
    /// Reads the per-(col, term) metadata + skip table, then iterates
    /// blocks in order. Maintains a top-k min-heap of `(score, doc_id)`.
    /// Once the heap is full (`heap.len() == k`), subsequent blocks
    /// whose skip-table `max_bm25` can't beat the heap's current
    /// minimum (= the current kth-best score) are skipped without
    /// decoding. Both the block bytes and the per-doc score loop are
    /// avoided.
    ///
    /// For uniform-dense lists where every block has similar
    /// `max_bm25`, BMW provides zero benefit. Its win shows up on
    /// posting lists with high score variance — e.g. very long lists
    /// where most blocks contain mid-relevance docs and the top-k is
    /// dominated by a few outliers.
    async fn search_single_term_bmw(
        &self,
        column_id: u32,
        term: &str,
        k: usize,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let fst_bytes = self.dict_bytes_async().await?;
        let dict = DictReader::open(&fst_bytes).map_err(|e| {
            FtsError::Read(ReadError::MalformedVersion(format!(
                "FST parse failed: {e}"
            )))
        })?;
        let col_meta = &self.columns[column_id as usize];
        let key = make_key(&col_meta.name, term);
        let Some(packed) = dict.lookup(&key) else {
            return Ok(Vec::new());
        };
        let (metadata_offset, postings_length) = match FstValue::unpack(packed) {
            FstValue::Inline { doc_id, tf } => {
                // df=1 inline path: no postings-region read, no
                // skip-table, no PFOR decode. The single doc's score
                // is the entire result for any k ≥ 1.
                let idf_t = crate::superfile::fts::bm25::idf(self.n_docs as u64, 1);
                let idf_x_k1p1 = idf_t * (crate::superfile::fts::bm25::K1 + 1.0);
                let dl_norm_k1 = col_meta.dl_norm_k1[doc_id as usize];
                let score =
                    crate::superfile::fts::bm25::score_with_dl_norm_k1(idf_x_k1p1, tf, dl_norm_k1);
                return Ok(vec![(doc_id, score)]);
            }
            FstValue::Pfor {
                metadata_offset,
                postings_length,
            } => (metadata_offset as usize, postings_length as usize),
        };
        // Fetch only this term's byte range (metadata header + skip
        // table + blocks). The returned buffer starts at the metadata
        // header, so the region-relative `metadata_offset` rebases to
        // 0 for all indexing below.
        let term_bytes = {
            let mut fetched = self
                .fetch_term_postings(&[(metadata_offset, postings_length)])
                .await?;
            fetched.pop().expect("one fetched range for one PFOR term")
        };
        let postings = term_bytes.as_ref();
        let metadata_offset = 0usize;

        let term_meta = TermMeta::parse(postings, metadata_offset)?;

        let idf_t = crate::superfile::fts::bm25::idf(self.n_docs as u64, term_meta.df);
        let idf_x_k1p1 = idf_t * (crate::superfile::fts::bm25::K1 + 1.0);
        let dl_norm_k1 = col_meta.dl_norm_k1.as_slice();

        // Top-k min-heap; see `TopKEntry` for the reversed ordering
        // that makes `peek()` the current kth-best score.
        let mut heap: BinaryHeap<TopKEntry> =
            BinaryHeap::with_capacity(k.min(term_meta.num_blocks * 128).max(1));
        let mut buf_d = vec![0u32; BLOCK_LEN];
        let mut buf_t = vec![0u32; BLOCK_LEN];

        for i in 0..term_meta.num_blocks {
            // last_doc_id (first tuple slot) is unused here — it serves
            // AND-merge seeks, which single-term never does.
            let (_, block_offset_in_term, block_max_bm25) = term_meta.skip_entry(postings, i);

            // BMW skip: heap full AND this block can't beat the kth-best.
            if heap.len() >= k
                && let Some(TopKEntry(min_score, _)) = heap.peek()
                && block_max_bm25 <= *min_score
            {
                continue;
            }

            // Locate the block's bytes.
            let block_end_in_term = term_meta.block_end_in_term(postings, i);
            let block_bytes = &postings
                [metadata_offset + block_offset_in_term..metadata_offset + block_end_in_term];

            //  Actual number of real docs in that block.
            let n = decode_block(block_bytes, &mut buf_d, &mut buf_t);

            for j in 0..n {
                let doc_id = buf_d[j];
                let tf = buf_t[j];
                let score = crate::superfile::fts::bm25::score_with_dl_norm_k1(
                    idf_x_k1p1,
                    tf,
                    dl_norm_k1[doc_id as usize],
                );
                if heap.len() < k {
                    heap.push(TopKEntry(score, doc_id));
                } else if let Some(TopKEntry(min_score, _)) = heap.peek()
                    && score > *min_score
                {
                    heap.pop();
                    heap.push(TopKEntry(score, doc_id));
                }
            }
        }

        Ok(drain_top_k_desc(heap))
    }

    /// Build one `TermCursor` per term that resolves in the FST.
    /// Missing terms (FST miss) are silently dropped — fine for OR
    /// semantics where a missing term contributes nothing. Returned
    /// `Vec` may be empty (all terms missed) or shorter than `terms`.
    async fn build_term_cursors(
        &self,
        column_id: u32,
        terms: &[&str],
    ) -> Result<Vec<TermCursor>, FtsError> {
        let fst_bytes = self.dict_bytes_async().await?;
        let dict = DictReader::open(&fst_bytes).map_err(|e| {
            FtsError::Read(ReadError::MalformedVersion(format!(
                "FST parse failed: {e}"
            )))
        })?;
        let col_meta = &self.columns[column_id as usize];

        // Resolve each present term to either an inline (df=1) value or
        // a PFOR metadata offset, preserving query order. FST misses
        // are dropped (fine for OR; AND callers length-check). Collect
        // the PFOR offsets so all their byte ranges can be fetched in
        // one parallel fan-out below — never the whole postings region.
        enum Resolved {
            Inline { doc_id: u32, tf: u32 },
            Pfor,
        }
        let mut resolved: Vec<Resolved> = Vec::with_capacity(terms.len());
        let mut pfor_offsets: Vec<(usize, usize)> = Vec::new();
        for term in terms {
            let key = make_key(&col_meta.name, term);
            let Some(packed) = dict.lookup(&key) else {
                continue;
            };
            match FstValue::unpack(packed) {
                FstValue::Inline { doc_id, tf } => {
                    resolved.push(Resolved::Inline { doc_id, tf });
                }
                FstValue::Pfor {
                    metadata_offset,
                    postings_length,
                } => {
                    pfor_offsets.push((metadata_offset as usize, postings_length as usize));
                    resolved.push(Resolved::Pfor);
                }
            }
        }

        let pfor_bytes = self.fetch_term_postings(&pfor_offsets).await?;
        let mut pfor_iter = pfor_bytes.into_iter();

        let mut cursors: Vec<TermCursor> = Vec::with_capacity(resolved.len());
        for r in resolved {
            match r {
                Resolved::Inline { doc_id, tf } => {
                    let dl_norm_k1 = col_meta.dl_norm_k1[doc_id as usize];
                    cursors.push(TermCursor::new_inline(
                        doc_id,
                        tf,
                        self.n_docs as u64,
                        dl_norm_k1,
                    ));
                }
                Resolved::Pfor => {
                    let term_bytes = pfor_iter.next().expect("one fetched range per PFOR term");
                    cursors.push(TermCursor::new(term_bytes, self.n_docs as u64)?);
                }
            }
        }
        Ok(cursors)
    }

    /// Multi-term OR via WAND + BlockMaxWAND.
    ///
    /// Algorithm: maintain a `TermCursor` per query term. Each
    /// iteration sorts cursors by current `doc_id`, computes the
    /// **WAND pivot** (smallest j such that the prefix-sum of
    /// term-level upper bounds exceeds the kth-best score), then
    /// applies the **BMW augmentation** (per-block UBs across the
    /// pivot prefix). If the pivot doc can't beat the threshold even
    /// with full per-block UBs, advance the leftmost cursor past the
    /// smallest block-end among the prefix; otherwise score the doc
    /// and advance.
    ///
    /// Reference: Ding & Suel, "Faster Top-k Document Retrieval Using
    /// Block-Max Indexes", SIGIR 2011.
    ///
    /// Result invariants: top-k by descending BM25 score, ties broken
    /// by ascending doc_id.
    ///
    /// Not on the production path. `dispatch_multi_term_or` always
    /// routes to [`Self::run_max_score_bmm`]; this entry point is
    /// kept for `search_with_algo_for_bench` so the bench harness
    /// can compare algorithms under identical inputs. Cursor
    /// construction is shared with the BMM path.
    fn run_wand_bmw(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = col_meta.dl_norm_k1.as_slice();

        // `search_multi` passes `k = usize::MAX` to gather every
        // matching doc before weighting across columns; cap initial
        // capacity at n_docs (the upper bound on distinct doc_ids in
        // the heap) so we don't try to allocate `usize::MAX * size_of::<TopKEntry>()`.
        // The BinaryHeap grows on demand if needed.
        let initial_cap = k.min(self.n_docs as usize).max(1);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let mut threshold: f32 = 0.0;

        // Reused index buffer to avoid per-iteration allocation.
        let mut idx: Vec<usize> = Vec::with_capacity(cursors.len());

        loop {
            // Drop exhausted cursors. Doing this in-place keeps idx
            // valid for the next iteration without re-allocation.
            cursors.retain(|c| !c.is_exhausted());
            if cursors.is_empty() {
                break;
            }

            // Sort cursor indices ascending by current doc_id.
            idx.clear();
            idx.extend(0..cursors.len());
            // Per-iteration WAND cursor reorder; pdqsort because
            // cursors hold distinct current doc_ids in the heap
            // state used by this scan.
            idx.sort_unstable_by_key(|&i| cursors[i].current_doc_id());

            // WAND pivot: smallest j such that the prefix-sum of
            // *term-level* upper bounds exceeds the threshold.
            let mut accum_term_ub: f32 = 0.0;
            let mut pivot_j: Option<usize> = None;
            for (j, &ci) in idx.iter().enumerate() {
                accum_term_ub += cursors[ci].term_max_bm25;
                if accum_term_ub > threshold {
                    pivot_j = Some(j);
                    break;
                }
            }

            let Some(mut pivot_j) = pivot_j else {
                // Sum of all remaining term UBs ≤ threshold: no
                // future doc can beat the heap. Done.
                break;
            };

            let pivot_doc = cursors[idx[pivot_j]].current_doc_id();

            // Extend the pivot prefix to include any cursors past
            // `pivot_j` that are also at `pivot_doc`. They contribute
            // to both the BMW upper-bound sum and the actual score,
            // so missing them under-counts the BMW UB and could
            // trigger an incorrect skip.
            while pivot_j + 1 < idx.len() && cursors[idx[pivot_j + 1]].current_doc_id() == pivot_doc
            {
                pivot_j += 1;
            }

            // BMW augmentation: sum of per-block upper bounds for the
            // block that would contain `pivot_doc` in each prefix
            // cursor. Lagging cursors' current decoded block is for
            // an earlier doc whose UB doesn't bound their
            // contribution at pivot_doc; `shallow_advance_block_to`
            // moves the lightweight inspect-block pointer to the
            // pivot-doc block without decoding, then
            // `inspect_block_max_bm25` reads that block's UB.
            let mut accum_block_ub: f32 = 0.0;
            for &ci in &idx[..=pivot_j] {
                cursors[ci].shallow_advance_block_to(pivot_doc);
                accum_block_ub += cursors[ci].inspect_block_max_bm25();
            }

            if accum_block_ub <= threshold {
                // No doc in [pivot_doc, smallest_pivot_block_end]
                // can beat the kth-best score. Advance the leftmost
                // cursor to the next interesting doc — either one
                // past the smallest pivot-block-end among the prefix,
                // or a suffix cursor's current doc if that's closer.
                // The suffix cap matters for recall: without it,
                // leftmost can leap multiple blocks past pivot_doc
                // and overshoot a doc one of the suffix cursors is
                // sitting at, leaving that doc with too few cursors
                // ever positioned on it to score correctly.
                let mut target = u32::MAX;
                for &ci in &idx[..=pivot_j] {
                    let last = cursors[ci].inspect_block_last_doc_id();
                    if last < target {
                        target = last;
                    }
                }
                let mut effective_target = target.saturating_add(1);
                for &ci in &idx[pivot_j + 1..] {
                    let d = cursors[ci].current_doc_id();
                    if d < effective_target {
                        effective_target = d;
                    }
                }
                cursors[idx[0]].skip_to(effective_target);
                continue;
            }

            // Align every lagging cursor in the pivot prefix to
            // `pivot_doc` so its contribution is included in this
            // doc's score. If any cursor's posting list doesn't
            // contain `pivot_doc` (the seek lands past it), abandon
            // this pivot — re-sort and re-pivot next iteration. This
            // is the WAND alignment step (Ding & Suel §3); without
            // it, lagging cursors that DO have pivot_doc in their
            // posting list get advanced past it on subsequent
            // iterations without ever contributing to its score,
            // producing under-counted scores and missing top-k hits.
            let mut aligned = true;
            for &ci in &idx[..=pivot_j] {
                if cursors[ci].current_doc_id() < pivot_doc {
                    cursors[ci].skip_to(pivot_doc);
                    if cursors[ci].current_doc_id() != pivot_doc {
                        aligned = false;
                        break;
                    }
                }
            }
            if !aligned {
                continue;
            }

            // All prefix cursors are at pivot_doc. Score it by summing
            // contributions from every cursor at pivot_doc (cursors
            // beyond the prefix may also be at pivot_doc — they
            // contribute too). SIMD-pack up to 4 cursors per scoring
            // call.
            let norm = dl_norm_k1[pivot_doc as usize];
            let mut score: f32 = 0.0;
            let mut idfs = [0.0_f32; 4];
            let mut tfs = [0.0_f32; 4];
            let mut packed = 0;
            for cursor in &cursors {
                if cursor.current_doc_id() == pivot_doc {
                    idfs[packed] = cursor.idf_x_k1p1;
                    tfs[packed] = cursor.current_tf() as f32;
                    packed += 1;
                    if packed == 4 {
                        score += crate::superfile::fts::bm25::score_simd_x4(idfs, tfs, norm);
                        idfs = [0.0; 4];
                        tfs = [0.0; 4];
                        packed = 0;
                    }
                }
            }
            if packed > 0 {
                score += crate::superfile::fts::bm25::score_simd_x4(idfs, tfs, norm);
            }

            // Update heap.
            if heap.len() < k {
                heap.push(TopKEntry(score, pivot_doc));
                if heap.len() == k {
                    threshold = heap.peek().expect("non-empty").0;
                }
            } else if let Some(TopKEntry(min_score, _)) = heap.peek()
                && score > *min_score
            {
                heap.pop();
                heap.push(TopKEntry(score, pivot_doc));
                threshold = heap.peek().expect("non-empty").0;
            }

            // Advance every cursor at pivot_doc (the prefix, plus any
            // cursors past the prefix that happened to be at it).
            for cursor in cursors.iter_mut() {
                if cursor.current_doc_id() == pivot_doc {
                    cursor.next();
                }
            }
        }

        Ok(drain_top_k_desc(heap))
    }

    /// Multi-term OR via Block-Max MaxScore (BMM).
    ///
    /// Algorithm sketch (Turtle & Flood 1995, Strohman & Croft 2007;
    /// the "Block-Max" augmentation per Petri & Moffat 2017):
    ///
    ///   1. Sort cursors in *descending* `term_max_bm25`.
    ///   2. Compute suffix sums: `partial_max[i] = sum_{j>=i} cursors[j].term_max_bm25`.
    ///   3. Partition into **essential** prefix `cursors[0..f]` and
    ///      **non-essential** suffix `cursors[f..n]` where
    ///      `f = min{ i : partial_max[i] <= threshold }`. A doc that
    ///      appears only in non-essential cursors has max-possible
    ///      score `partial_max[f] <= threshold` and can't make top-k.
    ///   4. Find next candidate doc as the smallest `current_doc_id`
    ///      among essential cursors. (Non-essential cursors are
    ///      skipped *to* the candidate, not iterated for new candidates.)
    ///   5. Apply BMW-style block-skip on the leftmost essential: if
    ///      `leftmost_block_ub + sum_other_term_ubs <= threshold`,
    ///      no doc in the leftmost's current block can beat top-k —
    ///      jump leftmost past its block.
    ///   6. Score: sum essential contributions, then run the
    ///      non-essential loop with **block-level** early termination
    ///      using `current_block_max_bm25` of the remaining cursors.
    ///   7. Update heap; recompute `f` from the new threshold; repeat.
    ///
    /// **When is BMM better than WAND+BMW?** When query terms have
    /// similar upper bounds (3+ same-rank Zipfian terms is the
    /// canonical case) — WAND's pivot moves around because no single
    /// cursor dominates, while MaxScore stably partitions essential
    /// vs non-essential. WAND wins when one term has much higher UB
    /// (rare + common); the partition collapses to a single
    /// essential cursor anyway and WAND's pivot is tighter.
    ///
    /// The router [`Self::dispatch_multi_term_or`] picks between
    /// the two using a UB-spread heuristic. Both algorithms share
    /// cursor construction via [`Self::build_term_cursors`] so the
    /// router doesn't pay for cursor work twice.
    fn run_max_score_bmm(
        &self,
        column_id: u32,
        cursors: Vec<TermCursor>,
        k: usize,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        self.run_max_score_bmm_range(column_id, cursors, k, 0, u32::MAX)
    }

    /// Multi-term AND via leapfrog intersection over the skip table.
    ///
    /// The smallest-df cursor is the leader: every matching doc must
    /// be in its posting list. For each leader doc, every other
    /// cursor runs `skip_to(candidate)` — a skip-table-driven jump
    /// that decodes at most one block per call (and zero if the
    /// target lies in the already-decoded block). If any cursor
    /// lands past the candidate, that doc isn't in the intersection;
    /// the candidate is bumped to the new high-water mark and the
    /// remaining cursors re-skip. When all cursors converge on the
    /// same doc, the BM25 contribution from each is summed.
    ///
    /// Cost is bounded by `min_df` leader steps × `n_terms` skip_to
    /// calls, with each skip_to a constant-or-O(log) skip-table walk.
    /// The old `run_and` did a full PFOR decode of every term's full
    /// posting list (dominated by the largest list, e.g. ~hundreds of
    /// K postings for a common Zipfian term) followed by a HashMap
    /// intersection — orders of magnitude more work than this when
    /// any term is rare.
    fn run_and_intersect(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        if cursors.is_empty() {
            return Ok(Vec::new());
        }
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = col_meta.dl_norm_k1.as_slice();

        // Smallest-df cursor at index 0 = leader. The remaining order
        // doesn't matter for correctness but ascending-df reduces the
        // expected number of leapfrog bumps per candidate.
        cursors.sort_by_key(|c| c.block_count());

        let initial_cap = k.min(self.n_docs as usize).max(1);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);

        // 2-term shape gets a specialized flat-merge inner loop: when
        // both cursors sit in their decoded block buffers, we walk the
        // two sorted `block_doc_ids` arrays with two index pointers
        // instead of calling `skip_to` per leader doc. That removes
        // the function-call + within-block linear-scan overhead on the
        // hottest AND case (rare ∧ common). The general path is kept
        // for n >= 3 because flat-merge across N arrays doesn't
        // straightforwardly generalize and the per-doc leapfrog still
        // amortizes well with the block-max pruning below.
        if cursors.len() == 2 {
            self.run_and_intersect_2term(&mut cursors, dl_norm_k1, k, &mut heap);
        } else {
            self.run_and_intersect_general(&mut cursors, dl_norm_k1, k, &mut heap);
        }

        Ok(drain_top_k_desc(heap))
    }

    /// General `n >= 3`-term AND path. Same shape as the 2-term path:
    /// block-max pruning at the top, then a flat-merge over the
    /// leader's decoded `block_doc_ids` against each non-leader's
    /// decoded `block_doc_ids`. For each leader doc, every non-leader's
    /// `pos` is advanced with a tight `pos += 1` scan instead of
    /// `skip_to` — no function-call or within-block linear-scan
    /// overhead per leader doc, just integer comparisons over the
    /// already-decoded buffers. When any cursor exhausts its block,
    /// the outer loop crosses blocks via `next()` and re-aligns.
    fn run_and_intersect_general(
        &self,
        cursors: &mut [TermCursor],
        dl_norm_k1: &[f32],
        k: usize,
        heap: &mut BinaryHeap<TopKEntry>,
    ) {
        'outer: loop {
            if cursors[0].is_exhausted() {
                break;
            }

            // Block-max-AND pruning. After the heap fills, the kth-best
            // score gates further inserts. If the leader's current block
            // can't possibly produce a top-K beating-score, skip the
            // whole block — the safest UB sums leader's block_max with
            // each other cursor's max block_max across all blocks that
            // overlap the leader's block doc-id range.
            if heap.len() >= k {
                let heap_min = heap.peek().expect("heap len == k").0;
                let range_start = cursors[0].current_doc_id();
                let range_end = cursors[0].current_block_last_doc_id();
                let leader_block_max = cursors[0].current_block_max_bm25();
                let mut other_ub = 0.0_f32;
                for c in cursors[1..].iter_mut() {
                    other_ub += c.block_max_in_range(range_start, range_end);
                }
                if leader_block_max + other_ub <= heap_min {
                    cursors[0].skip_to(range_end.saturating_add(1));
                    continue;
                }
            }

            // Align every non-leader cursor to >= leader's current doc.
            // Largest landing-doc becomes the new alignment target if
            // any cursor jumped past leader. If any cursor crossed
            // leader's current block, restart the outer loop so pruning
            // re-fires on leader's new block; otherwise the flat-merge
            // proceeds in the current decoded blocks.
            let leader_doc = cursors[0].current_doc_id();
            let leader_block_end = cursors[0].current_block_last_doc_id();
            let mut max_other = leader_doc;
            let mut crossed_block = false;
            for c in cursors[1..].iter_mut() {
                c.skip_to(leader_doc);
                if c.is_exhausted() {
                    break 'outer;
                }
                let here = c.current_doc_id();
                if here > leader_block_end {
                    crossed_block = true;
                }
                if here > max_other {
                    max_other = here;
                }
            }
            if max_other > leader_doc {
                cursors[0].skip_to(max_other);
                if cursors[0].is_exhausted() {
                    break 'outer;
                }
                if crossed_block {
                    continue;
                }
            }

            // Flat-merge across decoded blocks. Split leader off so
            // both leader and others borrow mutably without overlap;
            // the inner loop reads each cursor's `block_doc_ids` and
            // updates its `pos` directly.
            let (leader_slice, others) = cursors.split_at_mut(1);
            let c0 = &mut leader_slice[0];
            let lb_n = c0.block_n;
            let mut i = c0.pos;
            while i < lb_n {
                let a = c0.block_doc_ids[i];

                // For each non-leader, walk its `pos` forward through
                // the decoded block until block_doc_ids[pos] >= a (or
                // the block exhausts). If any block exhausts, break
                // out to the outer loop's block-crossing step. If any
                // cursor lands above `a`, the leader doc isn't in the
                // intersection — advance leader only.
                let mut block_exhausted = false;
                let mut all_match = true;
                for o in others.iter_mut() {
                    while o.pos < o.block_n && o.block_doc_ids[o.pos] < a {
                        o.pos += 1;
                    }
                    if o.pos >= o.block_n {
                        block_exhausted = true;
                        break;
                    }
                    if o.block_doc_ids[o.pos] != a {
                        all_match = false;
                        break;
                    }
                }
                if block_exhausted {
                    break;
                }
                if all_match {
                    let norm = dl_norm_k1[a as usize];
                    let mut score = crate::superfile::fts::bm25::score_with_dl_norm_k1(
                        c0.idf_x_k1p1,
                        c0.block_tfs[i],
                        norm,
                    );
                    for o in others.iter() {
                        score += crate::superfile::fts::bm25::score_with_dl_norm_k1(
                            o.idf_x_k1p1,
                            o.block_tfs[o.pos],
                            norm,
                        );
                    }
                    and_heap_push(heap, k, score, a);
                    i += 1;
                    for o in others.iter_mut() {
                        o.pos += 1;
                    }
                } else {
                    i += 1;
                }
            }
            c0.pos = i;

            // Cross blocks for whichever cursors exhausted. The outer
            // loop's alignment step re-pulls everyone to the new leader
            // doc on the next iteration.
            if c0.pos >= c0.block_n {
                c0.next();
            }
            for o in others.iter_mut() {
                if o.pos >= o.block_n {
                    o.next();
                }
            }
        }
    }

    /// 2-term specialization. While both cursors share a doc-id region
    /// covered by their respective decoded blocks, do a flat
    /// sorted-merge over the two `block_doc_ids` arrays: no `skip_to`
    /// function calls per leader doc, no per-doc within-block linear
    /// scan — just two index pointers walking forward. When either
    /// block exhausts, the cursor crosses to its next block (decoding
    /// on demand) and the merge resumes.
    fn run_and_intersect_2term(
        &self,
        cursors: &mut [TermCursor],
        dl_norm_k1: &[f32],
        k: usize,
        heap: &mut BinaryHeap<TopKEntry>,
    ) {
        debug_assert_eq!(cursors.len(), 2);
        // Split into two simultaneous mutable refs so the inner loop
        // can read both cursors' decoded buffers and update both
        // positions without borrow-checker contortions.
        let (left, right) = cursors.split_at_mut(1);
        let c0 = &mut left[0];
        let c1 = &mut right[0];

        'outer: loop {
            if c0.is_exhausted() || c1.is_exhausted() {
                break;
            }

            // Block-max-AND pruning at the leader's current block.
            if heap.len() >= k {
                let heap_min = heap.peek().expect("heap len == k").0;
                let range_start = c0.current_doc_id();
                let range_end = c0.current_block_last_doc_id();
                let ub =
                    c0.current_block_max_bm25() + c1.block_max_in_range(range_start, range_end);
                if ub <= heap_min {
                    c0.skip_to(range_end.saturating_add(1));
                    continue;
                }
            }

            // Align c1 with c0 at the current leader doc. After this
            // call both cursors are positioned on doc_ids >= leader.
            // If c1 jumped past the leader's current block we'll bump
            // the leader via the outer loop's next iteration.
            c1.skip_to(c0.current_doc_id());
            if c1.is_exhausted() {
                break 'outer;
            }
            // If c1 sits above c0's pos, pull c0 forward to align.
            // When that pull crosses c0's current block, restart the
            // outer loop so pruning re-fires on c0's new block;
            // otherwise fall through and let the flat-merge handle
            // the within-block divergence inline.
            if c1.current_doc_id() > c0.current_doc_id() {
                let crossed_block = c1.current_doc_id() > c0.current_block_last_doc_id();
                c0.skip_to(c1.current_doc_id());
                if c0.is_exhausted() {
                    break 'outer;
                }
                if crossed_block {
                    continue;
                }
            }

            // Flat sorted-merge within the overlap of the two decoded
            // blocks. Pre-load all locals; the borrow checker is
            // satisfied because c0/c1 are independently mutable refs.
            let lb_n = c0.block_n;
            let rb_n = c1.block_n;
            let mut i = c0.pos;
            let mut j = c1.pos;
            let c0_idf = c0.idf_x_k1p1;
            let c1_idf = c1.idf_x_k1p1;
            while i < lb_n && j < rb_n {
                let a = c0.block_doc_ids[i];
                let b = c1.block_doc_ids[j];
                if a < b {
                    i += 1;
                } else if a > b {
                    j += 1;
                } else {
                    let norm = dl_norm_k1[a as usize];
                    let score = crate::superfile::fts::bm25::score_with_dl_norm_k1(
                        c0_idf,
                        c0.block_tfs[i],
                        norm,
                    ) + crate::superfile::fts::bm25::score_with_dl_norm_k1(
                        c1_idf,
                        c1.block_tfs[j],
                        norm,
                    );
                    and_heap_push(heap, k, score, a);
                    i += 1;
                    j += 1;
                }
            }
            c0.pos = i;
            c1.pos = j;

            // Whichever cursor exhausted its block crosses to its next
            // block; the other holds. The outer loop re-checks
            // is_exhausted and re-aligns on the next iteration.
            if i >= lb_n {
                c0.next();
            }
            if j >= rb_n {
                c1.next();
            }
        }
    }

    /// MaxScore+BMM constrained to the doc_id half-open range
    /// `[doc_id_start, doc_id_end)`. Used by the supertable layer's
    /// intra-segment parallel fan-out: when the reader pool has more
    /// threads than segments, each segment is split into N sub-ranges
    /// and the per-sub-range searches run in parallel, each producing
    /// its own top-K heap that the caller merges.
    ///
    /// Setting `doc_id_start == 0` and `doc_id_end == u32::MAX`
    /// reproduces the un-ranged BMM walk byte-for-byte (the seek is
    /// a no-op and the upper-bound check trivially never fires).
    ///
    /// **Pruning trade**: each sub-range maintains an independent
    /// top-K heap + BMM threshold. The threshold tightens slower than
    /// in the un-ranged walk because each sub-range sees only `1/N`
    /// of the docs, so the per-sub-range BMW block-skip fires less
    /// aggressively. Net wall-time win comes from spreading the
    /// scoring work across more cores; the per-sub-range work loss
    /// from looser pruning is bounded by the bookkeeping path (and
    /// in practice ~10–20% of single-thread serial), well below the
    /// 2× cores-doubled headroom.
    fn run_max_score_bmm_range(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = col_meta.dl_norm_k1.as_slice();

        // Sub-range seek: jump every cursor past any doc_id below
        // the lower bound. Cursors already past the bound stay where
        // they are; cursors whose entire posting list sits below the
        // bound become exhausted. The skip_to walks the skip-table
        // (cross-block) when needed, so we don't decode blocks we'll
        // never score.
        if doc_id_start > 0 {
            for cursor in &mut cursors {
                cursor.skip_to(doc_id_start);
            }
        }

        // Sort descending by term-max UB. Stability isn't required —
        // ties (equal `term_max_bm25` across terms) are rare and the
        // tie-break is arbitrary as long as the prefix-sum invariant
        // holds.
        cursors.sort_unstable_by(|a, b| {
            b.term_max_bm25
                .partial_cmp(&a.term_max_bm25)
                .unwrap_or(Ordering::Equal)
        });

        // Suffix sums of term_max_bm25. partial_max[0] = total UB,
        // partial_max[n] = 0. Monotonically decreasing.
        let n = cursors.len();
        let mut partial_max = vec![0.0_f32; n + 1];
        for i in (0..n).rev() {
            partial_max[i] = partial_max[i + 1] + cursors[i].term_max_bm25;
        }

        let initial_cap = k.min(self.n_docs as usize).max(1);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let mut threshold: f32 = 0.0;

        // Essential boundary: smallest f such that partial_max[f] ≤ threshold.
        // Initially threshold=0; only partial_max[n]=0 satisfies, so f=n
        // (all terms essential). f shrinks as threshold rises.
        let mut f_essential: usize = n;
        let recompute_f = |partial_max: &[f32], threshold: f32| -> usize {
            // Linear scan from the front until partial_max[i] ≤ threshold.
            // For typical N ≤ 8 query terms this is cheaper than a
            // binary search's branch-and-bound overhead.
            let mut f = 0;
            while f < partial_max.len() - 1 && partial_max[f] > threshold {
                f += 1;
            }
            f
        };

        // Total term-level UB. Used for the block-skip bound on
        // essential cursors below.
        let total_term_ub = partial_max[0];

        loop {
            // **f=1 block-batch fast path.** Once threshold rises
            // enough that only `cursors[0]` (highest term_max) is
            // essential, the candidate set is *exactly* `cursors[0]`'s
            // posting list. We can decode one of its blocks and
            // process every doc in the block inline — no per-doc
            // pivot search, no per-doc cursor sort. The outer loop's
            // overhead amortizes over ~128 docs per block instead of
            // 1 doc per iteration. This is the steady state for
            // dominator queries (wide-UB) and for similar-UB queries
            // after the heap fills with multi-term hits.
            if f_essential == 1 {
                if cursors[0].is_exhausted() || cursors[0].current_doc_id() >= doc_id_end {
                    break;
                }
                // Block-skip: if `block_max + sum_others_term_max`
                // can't beat threshold, skip the block.
                let block_ub = cursors[0].current_block_max_bm25()
                    + (total_term_ub - cursors[0].term_max_bm25);
                if block_ub <= threshold {
                    let end = cursors[0].current_block_last_doc_id();
                    cursors[0].skip_to(end.saturating_add(1));
                    continue;
                }

                let block_end = cursors[0].current_block_last_doc_id();
                let mut f_changed = false;
                // Per-doc UB tightening: bound this doc's max possible
                // score by `essential_score + sum_others_term_max`.
                // If even this can't beat the heap threshold, skip
                // the non-essential lookups + heap update entirely
                // — those are the dominant per-doc cost. Only docs
                // where the essential alone is "in striking distance"
                // pay the full lookup price.
                let others_term_ub = total_term_ub - cursors[0].term_max_bm25;
                while !cursors[0].is_exhausted()
                    && cursors[0].current_doc_id() <= block_end
                    && cursors[0].current_doc_id() < doc_id_end
                {
                    let candidate = cursors[0].current_doc_id();
                    let norm = dl_norm_k1[candidate as usize];
                    let essential_score = crate::superfile::fts::bm25::score_with_dl_norm_k1(
                        cursors[0].idf_x_k1p1,
                        cursors[0].current_tf(),
                        norm,
                    );
                    if essential_score + others_term_ub <= threshold {
                        // No combination of non-essential
                        // contributions at `candidate` can push it
                        // above threshold. Skip lookup + heap.
                        cursors[0].next();
                        continue;
                    }
                    // SIMD-pack non-essentials at `candidate`.
                    let mut idfs = [cursors[0].idf_x_k1p1, 0.0, 0.0, 0.0];
                    let mut tfs = [cursors[0].current_tf() as f32, 0.0, 0.0, 0.0];
                    let mut packed = 1;
                    let mut score: f32 = 0.0;
                    for cursor in cursors.iter_mut().skip(1) {
                        cursor.skip_to(candidate);
                        if cursor.current_doc_id() == candidate {
                            idfs[packed] = cursor.idf_x_k1p1;
                            tfs[packed] = cursor.current_tf() as f32;
                            packed += 1;
                            if packed == 4 {
                                score +=
                                    crate::superfile::fts::bm25::score_simd_x4(idfs, tfs, norm);
                                idfs = [0.0; 4];
                                tfs = [0.0; 4];
                                packed = 0;
                            }
                        }
                    }
                    if packed > 0 {
                        score += crate::superfile::fts::bm25::score_simd_x4(idfs, tfs, norm);
                    }

                    if heap.len() < k {
                        heap.push(TopKEntry(score, candidate));
                        if heap.len() == k {
                            threshold = heap.peek().expect("non-empty").0;
                            let new_f = recompute_f(&partial_max, threshold);
                            if new_f != f_essential {
                                f_essential = new_f;
                                f_changed = true;
                            }
                        }
                    } else if score > threshold {
                        heap.pop();
                        heap.push(TopKEntry(score, candidate));
                        threshold = heap.peek().expect("non-empty").0;
                        let new_f = recompute_f(&partial_max, threshold);
                        if new_f != f_essential {
                            f_essential = new_f;
                            f_changed = true;
                        }
                    }

                    cursors[0].next();

                    if f_changed {
                        break;
                    }
                }
                continue;
            }

            // Pick the next candidate doc: smallest current_doc_id
            // among essential cursors. (Non-essential cursors only
            // get probed via skip_to once we have a candidate.)
            // Specialized for f=2 (the most common steady state for
            // similar-UB queries) to avoid the iter loop overhead.
            let (candidate, leftmost_essential) = if f_essential == 2 {
                let d0 = cursors[0].current_doc_id();
                let d1 = cursors[1].current_doc_id();
                if d0 == u32::MAX && d1 == u32::MAX {
                    break;
                }
                if d0 <= d1 { (d0, 0) } else { (d1, 1) }
            } else {
                let mut candidate = u32::MAX;
                let mut leftmost_essential: usize = 0;
                for (i, cursor) in cursors.iter().take(f_essential).enumerate() {
                    let d = cursor.current_doc_id();
                    if d < candidate {
                        candidate = d;
                        leftmost_essential = i;
                    }
                }
                if candidate == u32::MAX {
                    break;
                }
                (candidate, leftmost_essential)
            };
            // Sub-range upper bound: every subsequent candidate is
            // monotonically increasing, so once we cross the bound
            // there's no work left for this sub-range.
            if candidate >= doc_id_end {
                break;
            }

            // **BMW-style block-skip on the leftmost essential.** Bound
            // the score of any doc in `leftmost_essential`'s current
            // block by `current_block_max + sum_of_other_term_UBs`. If
            // that bound can't beat the threshold, no doc in this
            // block can make top-k — skip the cursor past its current
            // block. This is what makes BMM competitive with WAND+BMW
            // on dominant-term queries; without it MaxScore scans
            // every doc in the dominant term's posting list.
            let leftmost_term_ub = cursors[leftmost_essential].term_max_bm25;
            let leftmost_block_ub = cursors[leftmost_essential].current_block_max_bm25();
            // others_ub = sum of OTHER cursors' term UBs (essential + non-essential).
            // We use term-level UBs for the others as a conservative bound; using
            // their per-block UBs would tighten further but require keeping them
            // synced with the candidate, which we only do lazily in the
            // non-essential probe below.
            let others_ub = total_term_ub - leftmost_term_ub;
            if leftmost_block_ub + others_ub <= threshold {
                let last_in_block = cursors[leftmost_essential].current_block_last_doc_id();
                cursors[leftmost_essential].skip_to(last_in_block.saturating_add(1));
                continue;
            }

            // Score essential contributions at the candidate doc.
            // SIMD-pack up to 4 cursors per scoring call. (Essential
            // scoring has no early-bail; non-essential scoring below
            // does, so it stays scalar to keep `score` always
            // up-to-date for the bail check.)
            let norm = dl_norm_k1[candidate as usize];
            let mut score: f32 = 0.0;
            let mut idfs = [0.0_f32; 4];
            let mut tfs = [0.0_f32; 4];
            let mut packed = 0;
            for cursor in cursors.iter().take(f_essential) {
                if cursor.current_doc_id() == candidate {
                    idfs[packed] = cursor.idf_x_k1p1;
                    tfs[packed] = cursor.current_tf() as f32;
                    packed += 1;
                    if packed == 4 {
                        score += crate::superfile::fts::bm25::score_simd_x4(idfs, tfs, norm);
                        idfs = [0.0; 4];
                        tfs = [0.0; 4];
                        packed = 0;
                    }
                }
            }
            if packed > 0 {
                score += crate::superfile::fts::bm25::score_simd_x4(idfs, tfs, norm);
            }

            // Per-doc UB tightening: bound the doc's max possible
            // score by `essential_score + sum_non_essentials_term_max`.
            // If even this can't beat threshold, skip the
            // non-essential probe + heap update entirely. This is
            // looser than the per-non-essential block_ub bound below
            // but spares the `skip_to` cursor advances themselves —
            // those are the dominant per-doc cost.
            let non_essentials_term_ub = partial_max[f_essential];
            if score + non_essentials_term_ub > threshold {
                // Tighter pre-bail using non-essential block_max
                // (which is tighter than term_max). Use shallow
                // advance — moves the lightweight inspect-block
                // pointer to candidate's block without decoding,
                // amortized O(1). If even this tighter UB can't beat
                // threshold, skip the deep skip_to pass entirely.
                let mut remaining_block_ub: f32 = 0.0;
                for cursor in cursors.iter_mut().skip(f_essential) {
                    cursor.shallow_advance_block_to(candidate);
                    remaining_block_ub += cursor.inspect_block_max_bm25();
                }

                if score + remaining_block_ub > threshold {
                    for cursor in cursors.iter_mut().skip(f_essential) {
                        let block_ub = cursor.inspect_block_max_bm25();
                        if score + remaining_block_ub <= threshold {
                            break;
                        }
                        cursor.skip_to(candidate);
                        if cursor.current_doc_id() == candidate {
                            score += crate::superfile::fts::bm25::score_with_dl_norm_k1(
                                cursor.idf_x_k1p1,
                                cursor.current_tf(),
                                norm,
                            );
                        }
                        remaining_block_ub -= block_ub;
                    }
                }
            }
            // (If essential score + remaining_block_ub already ≤ threshold,
            // we don't bother scoring non-essentials — the doc can't beat
            // the kth-best.)

            // Update heap. `threshold` is kept in sync with
            // heap.peek().0 every time we mutate the heap, so we can
            // gate the replace-or-skip decision against the local
            // f32 instead of paying for a heap.peek() per iter.
            if heap.len() < k {
                heap.push(TopKEntry(score, candidate));
                if heap.len() == k {
                    threshold = heap.peek().expect("non-empty").0;
                    f_essential = recompute_f(&partial_max, threshold);
                }
            } else if score > threshold {
                heap.pop();
                heap.push(TopKEntry(score, candidate));
                threshold = heap.peek().expect("non-empty").0;
                f_essential = recompute_f(&partial_max, threshold);
            }

            // Advance every essential cursor that was at the candidate
            // doc. (Non-essential cursors stay where skip_to landed
            // them; the next iteration's skip_to will move them as
            // needed for the next candidate.)
            for cursor in cursors.iter_mut().take(f_essential) {
                if cursor.current_doc_id() == candidate {
                    cursor.next();
                }
            }
        }

        Ok(drain_top_k_desc(heap))
    }

    /// Exhaustive union walk for multi-term OR. No threshold-driven
    /// block skipping — every doc in the union of the cursor postings
    /// is scored and offered to the top-K heap.
    ///
    /// **Not on the production path.** `dispatch_multi_term_or` always
    /// routes to MaxScore+BMM; this function is reachable only via
    /// `search_with_algo_for_bench(OrAlgo::Exhaustive)`. It exists
    /// because the supertable bench surfaced one specific shape where
    /// it narrowly wins, and we want the option available for future
    /// re-routing work without re-implementing it.
    ///
    /// **When this can beat BMM (measured at 10M × 8 segments)**:
    /// - **Prefix expansions over very-rare terms, in parallel mode.**
    ///   E.g., `term0009*` expanding to 10 terms at Zipfian rank
    ///   90–99 (df ≈ 0.1% each). On the supertable parallel bench,
    ///   exhaustive ran at 40.2 ms vs BMM's 54.0 ms — a 26% win. The
    ///   per-segment work is tiny (∼12 K matching docs across 10
    ///   short cursors) so BMM's per-block bookkeeping
    ///   (`f_essential` recomputation, `shallow_advance_block_to`,
    ///   `inspect_block_max_bm25`) dominates over actual scoring
    ///   work.
    ///
    /// **When BMM is strictly better — measured regressions if we
    /// route to exhaustive**:
    /// - **Mid-rank uniform-UB queries.** Five terms at rank 50–54
    ///   (df ≈ 0.4% each): exhaustive serial 174 ms vs BMM 99 ms —
    ///   a **76% regression**. Three terms at rank 50–52: exhaustive
    ///   serial 93 ms vs BMM 61 ms — a **52% regression**. Enough
    ///   matching docs exist that BMM's skip-pruning actually fires
    ///   and amortizes its bookkeeping.
    /// - **Any dominant-term query.** BMM's `f_essential == 1` fast
    ///   path collapses to a block-batch loop on the dominant
    ///   cursor's postings — about as tight as exhaustive could be,
    ///   and with skip on top.
    /// - **Single-term queries.** Don't go through OR dispatch
    ///   anyway; `search_single_term_bmw` handles them.
    ///
    /// **Routing heuristic if revisited**: the obvious-looking
    /// `max(term_max_bm25) / sum(term_max_bm25) < 1.5/n_cursors`
    /// (uniform UB) **over-routes** because it admits mid-rank
    /// queries where BMM wins. A better rule would gate on
    /// *absolute* low total df **and** uniform UB — e.g.,
    /// `σdf < n_docs / 100 AND max_ub/sum_ub < 1.5/n_cursors`.
    /// Empirically that admits the prefix-of-rare-terms shape and
    /// excludes the mid-rank multi-term shapes. Not yet wired up:
    /// the single-query parallel win (26% on prefix) hasn't
    /// justified the routing-heuristic maintenance cost yet.
    ///
    /// Algorithm: classic k-way merge over `TermCursor`s. Each
    /// iteration finds the smallest current `doc_id` among live
    /// cursors, sums BM25 contributions from all cursors at that
    /// doc, advances those cursors, pushes into the top-K min-heap.
    ///
    /// Result invariants match [`Self::run_max_score_bmm`]: top-k by
    /// descending BM25 score, ties broken by ascending doc_id.
    fn run_exhaustive_union(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = col_meta.dl_norm_k1.as_slice();

        let initial_cap = k.min(self.n_docs as usize).max(1);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let mut threshold: f32 = 0.0;

        loop {
            // Find smallest current doc_id across all live cursors —
            // the next candidate to score. Exhausted cursors report
            // `u32::MAX`, which can't be smaller than any live cursor's
            // doc_id, so this terminates naturally when every cursor
            // has been drained.
            let mut candidate = u32::MAX;
            for cursor in &cursors {
                let d = cursor.current_doc_id();
                if d < candidate {
                    candidate = d;
                }
            }
            if candidate == u32::MAX {
                break;
            }

            // Score: sum BM25 from every cursor positioned at the
            // candidate doc. Pack up to 4 cursors per SIMD scoring
            // call, matching the BMM essential-scoring shape.
            let norm = dl_norm_k1[candidate as usize];
            let mut score: f32 = 0.0;
            let mut idfs = [0.0_f32; 4];
            let mut tfs = [0.0_f32; 4];
            let mut packed = 0;
            for cursor in cursors.iter_mut() {
                if cursor.current_doc_id() == candidate {
                    idfs[packed] = cursor.idf_x_k1p1;
                    tfs[packed] = cursor.current_tf() as f32;
                    packed += 1;
                    if packed == 4 {
                        score += crate::superfile::fts::bm25::score_simd_x4(idfs, tfs, norm);
                        idfs = [0.0; 4];
                        tfs = [0.0; 4];
                        packed = 0;
                    }
                    cursor.next();
                }
            }
            if packed > 0 {
                score += crate::superfile::fts::bm25::score_simd_x4(idfs, tfs, norm);
            }

            // Top-K update. `threshold` mirrors `heap.peek().0` so
            // the replace-or-skip branch doesn't re-peek per iter.
            if heap.len() < k {
                heap.push(TopKEntry(score, candidate));
                if heap.len() == k {
                    threshold = heap.peek().expect("non-empty").0;
                }
            } else if score > threshold {
                heap.pop();
                heap.push(TopKEntry(score, candidate));
                threshold = heap.peek().expect("non-empty").0;
            }
        }

        Ok(drain_top_k_desc(heap))
    }

    /// Multi-term OR dispatch. Routes everything to MaxScore+BMM.
    ///
    /// **Routing decision (1M docs, M4 Max — head-to-head WAND+BMW vs MaxScore+BMM):**
    ///
    /// | Query shape                                 | WAND+BMW | MaxScore+BMM |
    /// |---|---|---|
    /// | two-term wide (rank 1 + 50)                 | 1.25 ms  | **0.28 ms**  |
    /// | three-term wide (rank 1 + 50 + 100)         | 17.2 ms  | 18.3 ms      |
    /// | three-term similar UBs (rank 50/51/52)      | 28.3 ms  | **24.7 ms**  |
    /// | five-term similar UBs (rank 50–54)          | 59.1 ms  | **55.1 ms**  |
    ///
    /// BMM wins on most shapes once we have:
    ///   1. A precomputed per-doc length-norm table (no per-call
    ///      `dl/avgdl` work in scoring).
    ///   2. SIMD x4 scoring of all aligned cursors per doc.
    ///   3. A block-batch fast path when only one cursor is essential
    ///      (`f_essential == 1`) — the steady state for wide-UB and
    ///      heap-warmed similar-UB queries.
    ///
    /// **Exhaustive union walk** ([`Self::run_exhaustive_union`]) is
    /// implemented and reachable via `search_with_algo_for_bench`,
    /// but the dispatcher does NOT route to it. Empirically it
    /// regressed mid-rank uniform-UB shapes by 50–80% — see
    /// `run_exhaustive_union`'s doc comment for the cost model and
    /// the one shape (prefix-of-very-rare-terms in parallel mode)
    /// where it narrowly wins. WAND+BMW remains in the codebase
    /// for the same reason — bench-harness comparison only.
    async fn dispatch_multi_term_or(
        &self,
        column_id: u32,
        terms: &[&str],
        k: usize,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let cursors = self.build_term_cursors(column_id, terms).await?;
        if cursors.is_empty() {
            return Ok(Vec::new());
        }
        self.run_max_score_bmm(column_id, cursors, k)
    }

    /// Bench/dev helper: force the multi-term OR path to use a specific
    /// algorithm regardless of the dispatcher's heuristic. Used by
    /// `benches/fts_search.rs` to compare WAND+BMW, MaxScore+BMM, and
    /// exhaustive-union under identical inputs so the heuristic
    /// threshold can be validated against measured numbers.
    ///
    /// **Not part of the stable API** — production code should use
    /// `search`, which routes through `dispatch_multi_term_or`.
    #[doc(hidden)]
    pub async fn search_with_algo_for_bench(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        algo: OrAlgo,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let column_id = self.resolve_column_id(column)?;
        if terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let cursors = self.build_term_cursors(column_id, terms).await?;
        if cursors.is_empty() {
            return Ok(Vec::new());
        }
        match algo {
            OrAlgo::Bmm => self.run_max_score_bmm(column_id, cursors, k),
            OrAlgo::WandBmw => self.run_wand_bmw(column_id, cursors, k),
            OrAlgo::Exhaustive => self.run_exhaustive_union(column_id, cursors, k),
        }
    }
}

/// Top-k min-heap entry `(score, doc_id)`, shared by every search
/// path (single-term BMW, WAND+BMW, MaxScore+BMM, exhaustive union,
/// AND intersection, and the `search_multi` combiner).
///
/// Ordering is **reversed** on purpose: smaller score is "greater",
/// so `BinaryHeap::peek()` returns the smallest-score entry. Once the
/// heap holds k entries, `peek()` is the current kth-best score — the
/// bar a new doc must beat (also the BMW/BMM pruning threshold).
/// Tie-break: larger doc_id is "greater", so on equal scores the
/// smaller doc_id survives in the heap.
#[derive(Debug, Copy, Clone)]
struct TopKEntry(f32, u32);
impl PartialEq for TopKEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl Eq for TopKEntry {}
impl PartialOrd for TopKEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TopKEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .partial_cmp(&self.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.1.cmp(&self.1))
    }
}

/// Drain a top-k min-heap into the public result order: descending
/// score, ascending doc_id on ties.
///
/// pdqsort: entries are unique by `(score, doc_id)` — every search
/// path offers each doc_id to its heap at most once — so an unstable
/// sort has no observable reorderings.
fn drain_top_k_desc(heap: BinaryHeap<TopKEntry>) -> Vec<(u32, f32)> {
    let mut out: Vec<(u32, f32)> = heap.into_iter().map(|TopKEntry(s, d)| (d, s)).collect();
    out.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    out
}

/// Push `(score, doc_id)` into the top-k AND heap with the same
/// tie-break (asc doc_id) the OR paths use, so AND and OR rankings
/// agree on score-tied docs.
#[inline]
fn and_heap_push(heap: &mut BinaryHeap<TopKEntry>, k: usize, score: f32, doc_id: u32) {
    if heap.len() < k {
        heap.push(TopKEntry(score, doc_id));
    } else if let Some(&worst) = heap.peek()
        && (score > worst.0 || (score == worst.0 && doc_id < worst.1))
    {
        heap.pop();
        heap.push(TopKEntry(score, doc_id));
    }
}

/// Merge a `doc_id -> score` map into top-k by descending score, ties
/// broken by ascending doc_id. Used by `search_multi`'s cross-column
/// combiner, where the per-column scores have already been weighted
/// and summed into `scores`.
fn top_k(scores: HashMap<u32, f32>, k: usize) -> Vec<(u32, f32)> {
    // Iterate in ascending doc_id order so ties resolve deterministically
    // (smaller doc_ids enter the heap first; the strict `score > peek`
    // check below means subsequent equal-score entries don't displace
    // them). Without this, HashMap's hash-order iteration would make the
    // tied result non-deterministic and would disagree with the BMW
    // single-term path (which naturally iterates in doc_id order).
    // pdqsort: doc_ids are unique by construction (HashMap keys).
    let mut sorted: Vec<(u32, f32)> = scores.into_iter().collect();
    sorted.sort_unstable_by_key(|(d, _)| *d);

    let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(k.min(sorted.len()).max(1));
    for (doc_id, score) in sorted {
        if heap.len() < k {
            heap.push(TopKEntry(score, doc_id));
        } else if let Some(TopKEntry(top_score, _)) = heap.peek()
            && score > *top_score
        {
            heap.pop();
            heap.push(TopKEntry(score, doc_id));
        }
    }
    drain_top_k_desc(heap)
}

fn fetch_source_range(source: &Source, range: Range<usize>, what: &str) -> Result<Bytes, FtsError> {
    source.get_range(range).map_err(|e| {
        FtsError::Read(ReadError::MalformedVersion(format!(
            "{what} lazy source range fetch failed: {e}"
        )))
    })
}

async fn fetch_lazy_range(
    source: &dyn LazyByteSource,
    range: Range<usize>,
    what: &str,
) -> Result<Bytes, FtsError> {
    source
        .range(range.start as u64, range.len() as u64)
        .await
        .map_err(|e| {
            FtsError::Read(ReadError::MalformedVersion(format!(
                "{what} lazy source range fetch failed: {e}"
            )))
        })
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

/// Parsed per-(column, term) metadata header from the postings
/// region. The byte layout is documented once, on the writer side —
/// see [`TERM_META_SIZE`] in `builder.rs` — this struct is its
/// read-side mirror and must stay in sync with that doc.
///
/// [`TermMeta::parse`] is the single place that validates untrusted
/// offsets (the FST value points here) against the postings region:
/// both the fixed 20-byte header and the skip table it declares are
/// bounds-checked before any caller touches a byte. Both the
/// single-term BMW path and [`TermCursor::new`] go through here, so
/// the header layout is interpreted in exactly one spot.
#[derive(Debug, Copy, Clone)]
struct TermMeta {
    /// Document frequency — number of docs containing the term.
    df: u64,
    /// Byte length of the term's whole region (header + skip table +
    /// blocks), relative to the term's `metadata_offset`.
    postings_length: usize,
    /// Number of PFOR blocks (= number of skip-table entries).
    num_blocks: usize,
    /// Absolute offset (within the postings region) of the first
    /// skip-table entry: `metadata_offset + TERM_META_SIZE`.
    skip_start: usize,
}

impl TermMeta {
    /// Parse + bounds-validate the header and its skip table.
    /// Returns `Err` (never panics) on a corrupt or malicious
    /// `metadata_offset` — the crate-wide "untrusted input yields
    /// `Err`, not a slice-index panic" rule.
    fn parse(postings: &[u8], metadata_offset: usize) -> Result<Self, FtsError> {
        if metadata_offset + TERM_META_SIZE > postings.len() {
            return Err(FtsError::Read(ReadError::MalformedVersion(
                "term metadata offset out of postings region".into(),
            )));
        }
        let df = read_u32_le(&postings[metadata_offset..metadata_offset + 4]) as u64;
        // bytes [4..12] = self-offset (redundant; u64); skip
        let postings_length =
            read_u32_le(&postings[metadata_offset + 12..metadata_offset + 16]) as usize;
        let num_blocks =
            read_u32_le(&postings[metadata_offset + 16..metadata_offset + 20]) as usize;

        let skip_start = metadata_offset + TERM_META_SIZE;
        let skip_end = skip_start + num_blocks * SKIP_ENTRY_SIZE;
        if skip_end > postings.len() {
            return Err(FtsError::Read(ReadError::MalformedVersion(
                "skip table runs past postings region".into(),
            )));
        }
        Ok(Self {
            df,
            postings_length,
            num_blocks,
            skip_start,
        })
    }

    /// Decode skip-table entry `i` into `(last_doc_id,
    /// block_offset_in_term, block_max_bm25)`. `block_offset_in_term`
    /// is relative to the term's `metadata_offset`; `block_max_bm25`
    /// is recovered from the fixed-point `max_bm25_x1000` field. The
    /// reserved field (entry bytes 12..16) is ignored. Per-entry on
    /// purpose — the single-term BMW walk streams entries without
    /// materializing a `Vec`.
    #[inline]
    fn skip_entry(&self, postings: &[u8], i: usize) -> (u32, usize, f32) {
        debug_assert!(i < self.num_blocks, "skip entry {i} >= {}", self.num_blocks);
        let entry_off = self.skip_start + i * SKIP_ENTRY_SIZE;
        let last_doc_id = read_u32_le(&postings[entry_off..entry_off + 4]);
        let block_offset = read_u32_le(&postings[entry_off + 4..entry_off + 8]) as usize;
        let max_bm25_x1000 = read_u32_le(&postings[entry_off + 8..entry_off + 12]);
        (last_doc_id, block_offset, (max_bm25_x1000 as f32) / 1000.0)
    }

    /// End offset (relative to the term's `metadata_offset`) of block
    /// `i`'s bytes. Blocks are concatenated back-to-back, so each
    /// block ends where the next one's `block_offset` begins; the last
    /// block ends at `postings_length`.
    #[inline]
    fn block_end_in_term(&self, postings: &[u8], i: usize) -> usize {
        if i + 1 < self.num_blocks {
            let next_off = self.skip_start + (i + 1) * SKIP_ENTRY_SIZE;
            read_u32_le(&postings[next_off + 4..next_off + 8]) as usize
        } else {
            self.postings_length
        }
    }
}

/// Per-term per-block metadata, parsed once at `TermCursor` construction.
#[derive(Debug, Clone, Copy)]
struct BlockMeta {
    /// Largest doc_id present in this block.
    last_doc_id: u32,
    /// Absolute byte offset (within the FTS postings region) of this
    /// block's encoded bytes.
    block_byte_offset: usize,
    /// Absolute byte offset of the first byte AFTER this block. For
    /// the last block of a term it's `metadata_offset + postings_length`.
    block_byte_end: usize,
    /// Per-block BM25 upper bound, recovered from the skip table's
    /// fixed-point `max_bm25_x1000` field.
    block_max_bm25: f32,
}

/// Per-query-term cursor used by [`FtsReader::run_max_score_bmm`]
/// (and by [`FtsReader::run_wand_bmw`] in the bench-only path).
///
/// State:
///   - `blocks`: parsed skip table — one entry per block, lets us
///     decide whether to decode a block before paying the cost.
///   - `current_block` + `pos`: where we are in the term's posting
///     list. `pos == block_n` is treated as "advance to next block".
///   - `block_doc_ids` / `block_tfs`: decoded buffers for the current
///     block, reused across blocks.
///
/// `current_doc_id() == u32::MAX` is the "exhausted" sentinel; the
/// WAND loop drops cursors that are exhausted at the top of each
/// iteration.
struct TermCursor {
    /// Precomputed `idf * (K1 + 1)` — the score numerator's
    /// per-cursor constant. Computed once at cursor build so the
    /// hot inner loop fits one multiply + add + divide per call.
    /// (The bare `idf` value isn't kept on the cursor — every hot
    /// scoring path uses `score_with_dl_norm_k1` which takes
    /// `idf_x_k1p1` directly.)
    idf_x_k1p1: f32,
    /// Maximum block-max-BM25 across all blocks. Used by the WAND
    /// pivot test (term-level upper bound).
    term_max_bm25: f32,
    /// Per-block metadata.
    blocks: Vec<BlockMeta>,
    /// Decoded buffers for the current block. Reused across decodes.
    block_doc_ids: Vec<u32>,
    block_tfs: Vec<u32>,
    /// Number of valid entries in the decoded block buffers (the
    /// last block may be partial).
    block_n: usize,
    /// Index into `blocks` of the currently-decoded block. Equal to
    /// `blocks.len()` once exhausted.
    current_block: usize,
    /// Position within the currently-decoded block. Always `<
    /// block_n` while not exhausted.
    pos: usize,
    /// Index into `blocks` of the block being inspected by the BMW
    /// upper-bound check. Standard block-cursor split:
    /// `shallow_advance_block_to(pivot_doc)` updates this without
    /// decoding the block, so subsequent BMW UB lookups for
    /// monotonically-increasing pivot docs are amortized O(1). Always
    /// `>= current_block`; synced up whenever `current_block` is
    /// advanced.
    inspect_block: usize,
    /// This term's own postings bytes — the metadata header (offset
    /// 0), skip table, and encoded blocks, fetched as a single
    /// contiguous range by [`FtsReader::fetch_term_postings`]. All
    /// `BlockMeta` byte offsets are relative to the start of this
    /// buffer. Empty for inline (df=1) cursors, which never decode.
    /// Mirrors the vector reader's per-probed-cluster buffers: the
    /// search hot loops index only the bytes this term touches, never
    /// the whole postings region.
    bytes: Bytes,
}

impl TermCursor {
    /// Parse one term's metadata + skip table out of its own postings
    /// byte range and decode its first block. `term_bytes` starts at
    /// the term's 20-byte metadata header (offset 0) and runs to the
    /// end of its last block — the contiguous range
    /// [`FtsReader::fetch_term_postings`] fetched for this term.
    fn new(term_bytes: Bytes, n_docs: u64) -> Result<Self, FtsError> {
        let postings: &[u8] = term_bytes.as_ref();
        let metadata_offset = 0usize;

        let term_meta = TermMeta::parse(postings, metadata_offset)?;
        let idf = crate::superfile::fts::bm25::idf(n_docs, term_meta.df);

        let mut blocks: Vec<BlockMeta> = Vec::with_capacity(term_meta.num_blocks);
        let mut term_max_bm25: f32 = 0.0;
        for i in 0..term_meta.num_blocks {
            let (last_doc_id, block_offset_in_term, block_max_bm25) =
                term_meta.skip_entry(postings, i);
            term_max_bm25 = term_max_bm25.max(block_max_bm25);

            blocks.push(BlockMeta {
                last_doc_id,
                block_byte_offset: metadata_offset + block_offset_in_term,
                block_byte_end: metadata_offset + term_meta.block_end_in_term(postings, i),
                block_max_bm25,
            });
        }

        let mut cursor = Self {
            idf_x_k1p1: idf * (crate::superfile::fts::bm25::K1 + 1.0),
            term_max_bm25,
            blocks,
            block_doc_ids: vec![0u32; BLOCK_LEN],
            block_tfs: vec![0u32; BLOCK_LEN],
            block_n: 0,
            current_block: 0,
            pos: 0,
            inspect_block: 0,
            bytes: term_bytes,
        };
        if !cursor.blocks.is_empty() {
            cursor.decode_current_block();
        }
        Ok(cursor)
    }

    /// Synthesize a cursor for a df=1 inline-encoded term. Skips the
    /// postings-region read entirely — the caller already has
    /// (doc_id, tf) from unpacking the FST value, and BMW upper bound
    /// for a 1-doc term equals that doc's actual BM25 score (only one
    /// doc means min_dl = dl and max_tf = tf, so the per-block UB
    /// formula collapses to the score itself). Computed at query time
    /// since there's no skip-table entry stored for inline terms.
    fn new_inline(doc_id: u32, tf: u32, n_docs: u64, dl_norm_k1: f32) -> Self {
        let idf = crate::superfile::fts::bm25::idf(n_docs, 1);
        let idf_x_k1p1 = idf * (crate::superfile::fts::bm25::K1 + 1.0);
        let block_max_bm25 =
            crate::superfile::fts::bm25::score_with_dl_norm_k1(idf_x_k1p1, tf, dl_norm_k1);

        let blocks = vec![BlockMeta {
            last_doc_id: doc_id,
            // No postings-region bytes back this cursor; the decoded
            // buffer is pre-filled below so `decode_current_block` is
            // never called against these offsets.
            block_byte_offset: 0,
            block_byte_end: 0,
            block_max_bm25,
        }];

        let mut block_doc_ids = vec![0u32; BLOCK_LEN];
        let mut block_tfs = vec![0u32; BLOCK_LEN];
        block_doc_ids[0] = doc_id;
        block_tfs[0] = tf;

        Self {
            idf_x_k1p1,
            term_max_bm25: block_max_bm25,
            blocks,
            block_doc_ids,
            block_tfs,
            block_n: 1,
            current_block: 0,
            pos: 0,
            inspect_block: 0,
            bytes: Bytes::new(),
        }
    }

    fn decode_current_block(&mut self) {
        let block = self.blocks[self.current_block];
        let bytes = self
            .bytes
            .slice(block.block_byte_offset..block.block_byte_end);
        self.block_n = decode_block(&bytes, &mut self.block_doc_ids, &mut self.block_tfs);
        self.pos = 0;
    }

    fn is_exhausted(&self) -> bool {
        self.current_block >= self.blocks.len()
    }

    /// Block count, used as a cheap proxy for df when AND intersection
    /// picks the rarest cursor as the leader. Block count is an exact
    /// upper bound on df: a term's df is `(blocks - 1) * BLOCK_LEN +
    /// last_block_n`, so cursors compare in the same order by block
    /// count as they do by df. Inline cursors return 1.
    #[inline(always)]
    fn block_count(&self) -> usize {
        self.blocks.len()
    }

    #[inline(always)]
    fn current_doc_id(&self) -> u32 {
        if self.is_exhausted() || self.pos >= self.block_n {
            u32::MAX
        } else {
            self.block_doc_ids[self.pos]
        }
    }

    #[inline(always)]
    fn current_tf(&self) -> u32 {
        debug_assert!(!self.is_exhausted() && self.pos < self.block_n);
        self.block_tfs[self.pos]
    }

    #[inline(always)]
    fn current_block_max_bm25(&self) -> f32 {
        if self.is_exhausted() {
            0.0
        } else {
            self.blocks[self.current_block].block_max_bm25
        }
    }

    /// Largest doc_id in the cursor's current block. Used by the BMW
    /// skip step to compute the smallest "next interesting doc_id"
    /// across the prefix.
    #[inline(always)]
    fn current_block_last_doc_id(&self) -> u32 {
        if self.is_exhausted() {
            u32::MAX
        } else {
            self.blocks[self.current_block].last_doc_id
        }
    }

    /// Shallow-advance the inspect-block pointer to the block that
    /// would contain `target`. Does NOT decode and does NOT touch the
    /// doc cursor (`current_block`, `pos`, decoded buffers stay put);
    /// only the lightweight `inspect_block` index moves. Used by the
    /// BMW UB sum at `pivot_doc` for cursors whose current_doc lags
    /// pivot_doc — their relevant block-max is the block containing
    /// pivot_doc, not their current decoded block.
    ///
    /// Monotonically advances; calling this for monotonically-
    /// increasing `target` across WAND iterations gives amortized
    /// O(1) per call.
    fn shallow_advance_block_to(&mut self, target: u32) {
        // Never let inspect_block fall behind current_block — once
        // the doc cursor has decoded past a block, that block's
        // metadata is no longer relevant.
        if self.inspect_block < self.current_block {
            self.inspect_block = self.current_block;
        }
        while self.inspect_block < self.blocks.len()
            && self.blocks[self.inspect_block].last_doc_id < target
        {
            self.inspect_block += 1;
        }
    }

    /// Maximum `block_max_bm25` across all blocks of this cursor whose
    /// doc-id range overlaps `[range_start, range_end]` (inclusive on
    /// both ends). Used by AND block-max pruning to compute a safe
    /// upper bound on this cursor's contribution across the leader's
    /// current block — a single-block lookup at one boundary
    /// underestimates when the leader's range spans multiple
    /// cursor blocks with varying block_max. Uses `inspect_block` as
    /// a hint pointer so monotonically-advancing leader ranges amortize
    /// to O(1) amortized per call.
    fn block_max_in_range(&mut self, range_start: u32, range_end: u32) -> f32 {
        // Advance inspect_block to the first block whose last_doc_id
        // could intersect the range. shallow_advance_block_to lands on
        // the first block with last_doc_id >= range_start, which is
        // exactly the first block that can overlap the range.
        self.shallow_advance_block_to(range_start);
        let mut max: f32 = 0.0;
        let mut i = self.inspect_block;
        while i < self.blocks.len() {
            // Block i starts at the doc right after the previous block's
            // last_doc_id (or doc 0 if i == 0). Once block_start exceeds
            // range_end the rest of the blocks lie strictly past the
            // range; stop walking.
            let block_start = if i == 0 {
                0u32
            } else {
                self.blocks[i - 1].last_doc_id.saturating_add(1)
            };
            if block_start > range_end {
                break;
            }
            let m = self.blocks[i].block_max_bm25;
            if m > max {
                max = m;
            }
            i += 1;
        }
        max
    }

    /// Block-max-BM25 at the inspect-block pointer. Pair with
    /// `shallow_advance_block_to(pivot_doc)` to bound the cursor's
    /// contribution at pivot_doc.
    fn inspect_block_max_bm25(&self) -> f32 {
        if self.inspect_block >= self.blocks.len() {
            0.0
        } else {
            self.blocks[self.inspect_block].block_max_bm25
        }
    }

    /// Last doc_id in the block at the inspect-block pointer. Used
    /// for the BMW skip target — the smallest "next interesting doc"
    /// across the prefix is one past the smallest such block-end.
    fn inspect_block_last_doc_id(&self) -> u32 {
        if self.inspect_block >= self.blocks.len() {
            u32::MAX
        } else {
            self.blocks[self.inspect_block].last_doc_id
        }
    }

    /// Advance one position. Crosses block boundaries automatically;
    /// decodes the next block on demand.
    fn next(&mut self) {
        if self.is_exhausted() {
            return;
        }
        self.pos += 1;
        if self.pos >= self.block_n {
            self.current_block += 1;
            if self.current_block > self.inspect_block {
                self.inspect_block = self.current_block;
            }
            if self.current_block < self.blocks.len() {
                self.decode_current_block();
            }
        }
    }

    /// Skip forward so `current_doc_id() >= target`. Uses the skip
    /// table to skip whole blocks when the entire block precedes
    /// `target`. Common-case fast path (target lies within the
    /// already-decoded current block) is just an inlined `pos++`
    /// scan — no re-decode, no `is_exhausted` rechecks.
    #[inline(always)]
    fn skip_to(&mut self, target: u32) {
        if self.is_exhausted() {
            return;
        }
        let cur_block = self.current_block;
        let cur_block_last = self.blocks[cur_block].last_doc_id;
        if cur_block_last >= target {
            // Fast path: target is in our currently-decoded block.
            // Just scan pos forward. The `current_doc_id() >= target`
            // guard from before is folded into this scan — if pos is
            // already at-or-past, the loop body doesn't execute.
            let n = self.block_n;
            while self.pos < n && self.block_doc_ids[self.pos] < target {
                self.pos += 1;
            }
            if self.pos < n {
                return;
            }
            // Walked off the end of the decoded block (rare under
            // skip-table invariants); fall through to cross-block.
        }
        self.skip_to_cross_block(target);
    }

    /// Cross-block path of `skip_to`: target is past the current
    /// decoded block. Advances `current_block` via the skip table,
    /// decodes the new block (only when crossing), and scans pos.
    /// Pulled out so the within-block fast path stays small enough
    /// to inline at every call site.
    #[cold]
    fn skip_to_cross_block(&mut self, target: u32) {
        while self.current_block < self.blocks.len()
            && self.blocks[self.current_block].last_doc_id < target
        {
            self.current_block += 1;
        }
        if self.current_block > self.inspect_block {
            self.inspect_block = self.current_block;
        }
        if self.is_exhausted() {
            return;
        }
        self.decode_current_block();
        while self.pos < self.block_n && self.block_doc_ids[self.pos] < target {
            self.pos += 1;
        }
        if self.pos >= self.block_n {
            self.current_block += 1;
            if self.current_block > self.inspect_block {
                self.inspect_block = self.current_block;
            }
            if self.current_block < self.blocks.len() {
                self.decode_current_block();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::fts::builder::FtsBuilder;
    use crate::superfile::fts::tokenize::AsciiLowerTokenizer;
    use std::sync::Arc;

    fn build_blob() -> (Bytes, String) {
        // 3 docs, 1 column.
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into()).expect("register column");
        b.add_doc(0, 0, "rust async runtime").expect("add doc");
        b.add_doc(0, 1, "tokio is a rust runtime").expect("add doc");
        b.add_doc(0, 2, "java spring boot").expect("add doc");
        let bytes = b.finish().expect("finish");
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        (Bytes::from(bytes), json.to_string())
    }

    #[test]
    fn open_accepts_valid_blob() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open should succeed");
        assert_eq!(r.n_docs(), 3);
        assert!(r.n_terms() > 0);
        assert_eq!(r.fts_columns().collect::<Vec<_>>(), vec!["body"]);
    }

    #[test]
    fn open_rejects_bad_magic() {
        let (mut blob_vec, json) = build_blob();
        let mut bytes = blob_vec.to_vec();
        bytes[0] = b'X';
        blob_vec = Bytes::from(bytes);
        let err = FtsReader::open(blob_vec, &json).expect_err("expected error");
        assert!(matches!(err, FtsError::Read(ReadError::BadMagic { .. })));
    }

    #[test]
    fn open_rejects_short_blob() {
        let err = FtsReader::open(Bytes::from(vec![0u8; 8]), "[]").expect_err("expected error");
        assert!(matches!(err, FtsError::Read(_)));
    }

    #[test]
    fn open_rejects_columns_json_mismatch() {
        let (blob, _) = build_blob();
        // Header says n_columns=1; pass a 2-column JSON.
        let bad_json = r#"[{"name":"body","tokenizer":"ascii_lower"},{"name":"title","tokenizer":"ascii_lower"}]"#;
        let err = FtsReader::open(blob, bad_json).expect_err("expected error");
        assert!(matches!(
            err,
            FtsError::Read(ReadError::MalformedVersion(_))
        ));
    }

    #[tokio::test]
    async fn search_returns_exact_doc_ids_for_known_term() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["rust"], 10, BoolMode::Or)
            .await
            .expect("FTS search");
        // "rust" appears in doc 0 and doc 1.
        let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(ids.contains(&0), "doc 0 should match");
        assert!(ids.contains(&1), "doc 1 should match");
        assert!(!ids.contains(&2), "doc 2 should not match");
    }

    #[tokio::test]
    async fn exhaustive_and_bmm_agree_on_top_k() {
        // Build a larger blob so multi-term OR queries are
        // interesting (some docs have multiple terms, some have one).
        // Both algorithms must return identical top-K (descending
        // score, ascending doc_id tiebreak).
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into()).expect("register column");
        // 20 docs sprinkled with mixed term combinations.
        let docs = [
            "alpha",
            "beta",
            "gamma",
            "alpha beta",
            "alpha gamma",
            "beta gamma",
            "alpha beta gamma",
            "delta",
            "epsilon",
            "alpha delta",
            "beta epsilon",
            "gamma delta",
            "alpha beta delta",
            "alpha epsilon gamma",
            "delta epsilon",
            "alpha alpha alpha",
            "beta beta beta",
            "gamma gamma",
            "alpha beta gamma delta epsilon",
            "epsilon",
        ];
        for (i, text) in docs.iter().enumerate() {
            b.add_doc(0, i as u32, text).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        // Three terms with similar UBs — the heuristic should pick
        // exhaustive for this shape, but we cross-check by calling
        // both paths directly via the bench harness.
        let terms: &[&str] = &["alpha", "beta", "gamma"];
        let bmm = r
            .search_with_algo_for_bench("body", terms, 5, OrAlgo::Bmm)
            .await
            .expect("bmm");
        let exh = r
            .search_with_algo_for_bench("body", terms, 5, OrAlgo::Exhaustive)
            .await
            .expect("exhaustive");
        assert_eq!(bmm.len(), exh.len(), "result length mismatch");
        for ((d_bmm, s_bmm), (d_exh, s_exh)) in bmm.iter().zip(exh.iter()) {
            assert_eq!(d_bmm, d_exh, "doc_id mismatch");
            assert!(
                (s_bmm - s_exh).abs() < 1e-4,
                "score mismatch: bmm={s_bmm} exhaustive={s_exh}"
            );
        }
    }

    #[tokio::test]
    async fn search_missing_term_or_returns_empty() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["nonexistent"], 10, BoolMode::Or)
            .await
            .expect("search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn search_and_short_circuits_on_missing_term() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["rust", "nonexistent"], 10, BoolMode::And)
            .await
            .expect("search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn search_and_intersects_term_postings() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        // "rust AND runtime" — both in doc 0 and doc 1.
        let hits = r
            .search("body", &["rust", "runtime"], 10, BoolMode::And)
            .await
            .expect("search");
        let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(!ids.contains(&2));
    }

    #[tokio::test]
    async fn search_unknown_column_errors() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let err = r
            .search("title", &["rust"], 10, BoolMode::Or)
            .await
            .expect_err("expected error");
        assert!(matches!(err, FtsError::UnknownColumn(_)));
    }

    #[tokio::test]
    async fn search_empty_terms_returns_empty() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &[], 10, BoolMode::Or)
            .await
            .expect("FTS search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn search_zero_k_returns_empty() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["rust"], 0, BoolMode::Or)
            .await
            .expect("FTS search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn search_results_sorted_by_score_desc() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["rust"], 10, BoolMode::Or)
            .await
            .expect("FTS search");
        for w in hits.windows(2) {
            assert!(w[0].1 >= w[1].1, "scores should be descending");
        }
    }

    #[tokio::test]
    async fn search_limits_to_k() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["rust"], 1, BoolMode::Or)
            .await
            .expect("FTS search");
        assert_eq!(hits.len(), 1);
    }

    /// Build a corpus that exercises both the df=1 inline-encoded
    /// path and the df ≥ 2 PFOR path side-by-side.
    fn build_mixed_df_blob() -> (Bytes, String) {
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into()).expect("register column");
        // `common`     → df = 3 (PFOR form)
        // `rust`       → df = 2 (PFOR form)
        // `uniqzero`  → df = 1 (inline form)
        // `uniqtwo`  → df = 1 (inline form)
        b.add_doc(0, 0, "common rust uniqzero").expect("add doc");
        b.add_doc(0, 1, "common rust").expect("add doc");
        b.add_doc(0, 2, "common uniqtwo").expect("add doc");
        let bytes = b.finish().expect("finish");
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        (Bytes::from(bytes), json.to_string())
    }

    #[test]
    fn df1_inline_form_flag_set_on_fst_value() {
        // Verify the FST values for df=1 terms have bit 0 set
        // (inline form) and df ≥ 2 terms have bit 0 clear (PFOR).
        let (blob, _json) = build_mixed_df_blob();
        // Re-parse the blob enough to reach the FST bytes.
        let header_size = 48usize;
        let fst_off =
            u64::from_le_bytes(blob[24..32].try_into().expect("fst_off slice is 8 bytes")) as usize;
        let postings_off = u64::from_le_bytes(
            blob[32..40]
                .try_into()
                .expect("postings_off slice is 8 bytes"),
        ) as usize;
        // FST bytes occupy [fst_off, postings_off - 4) (last 4 = FST CRC).
        let fst_bytes = &blob[fst_off..postings_off - 4];
        let dict = crate::superfile::fts::dict::DictReader::open(fst_bytes).expect("open dict");
        assert_eq!(header_size, 48);

        let val_common = dict.lookup(b"body\x1Fcommon").expect("common in FST");
        let val_rust = dict.lookup(b"body\x1Frust").expect("rust in FST");
        let val_uniq_d0 = dict.lookup(b"body\x1Funiqzero").expect("uniqzero in FST");
        let val_uniq_d2 = dict.lookup(b"body\x1Funiqtwo").expect("uniqtwo in FST");

        assert_eq!(val_common & 1, 0, "df=3 common term must use PFOR form");
        assert_eq!(val_rust & 1, 0, "df=2 rust term must use PFOR form");
        assert_eq!(val_uniq_d0 & 1, 1, "df=1 uniqzero must use inline form");
        assert_eq!(val_uniq_d2 & 1, 1, "df=1 uniqtwo must use inline form");

        // Decode the inline values and check (doc_id, tf) match.
        match FstValue::unpack(val_uniq_d0) {
            FstValue::Inline { doc_id, tf } => {
                assert_eq!(doc_id, 0);
                assert_eq!(tf, 1);
            }
            FstValue::Pfor { .. } => panic!("expected inline form"),
        }
        match FstValue::unpack(val_uniq_d2) {
            FstValue::Inline { doc_id, tf } => {
                assert_eq!(doc_id, 2);
                assert_eq!(tf, 1);
            }
            FstValue::Pfor { .. } => panic!("expected inline form"),
        }
    }

    #[tokio::test]
    async fn df1_single_term_search_returns_one_doc() {
        let (blob, json) = build_mixed_df_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["uniqzero"], 10, BoolMode::Or)
            .await
            .expect("FTS search");
        assert_eq!(hits.len(), 1, "df=1 term should return exactly one hit");
        assert_eq!(hits[0].0, 0, "uniqzero lives in doc 0");
        assert!(hits[0].1 > 0.0, "score must be positive");
    }

    #[tokio::test]
    async fn df1_in_or_query_combines_with_df_ge_2() {
        let (blob, json) = build_mixed_df_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["uniqtwo", "rust"], 10, BoolMode::Or)
            .await
            .expect("FTS search");
        // uniqtwo → doc 2; rust → docs 0, 1.
        let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[tokio::test]
    async fn df1_in_and_query_intersects_correctly() {
        let (blob, json) = build_mixed_df_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        // uniqzero ∩ rust = {doc 0}.
        let hits = r
            .search("body", &["uniqzero", "rust"], 10, BoolMode::And)
            .await
            .expect("FTS search");
        let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert_eq!(ids, vec![0]);
        // uniqzero ∩ uniqtwo = ∅ (different docs).
        let hits = r
            .search("body", &["uniqzero", "uniqtwo"], 10, BoolMode::And)
            .await
            .expect("FTS search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn df1_missing_term_returns_empty() {
        let (blob, json) = build_mixed_df_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["nonexistentunique"], 10, BoolMode::Or)
            .await
            .expect("FTS search");
        assert!(hits.is_empty());
    }

    #[test]
    fn df1_inline_path_skips_postings_region_writes() {
        // A blob with only df=1 terms should produce a much smaller
        // postings region than a blob with the same term count but
        // df ≥ 2 — the inline form writes nothing for df=1.
        let tok = Arc::new(AsciiLowerTokenizer);

        let mut b_inline = FtsBuilder::new(tok.clone());
        b_inline
            .register_column("body".into())
            .expect("register column");
        for i in 0..20 {
            b_inline
                .add_doc(0, i, &format!("uniq{i:03}"))
                .expect("add doc");
        }
        let blob_inline = b_inline.finish().expect("finish inline");

        let mut b_pfor = FtsBuilder::new(tok);
        b_pfor
            .register_column("body".into())
            .expect("register column");
        // Same 20 terms but all appearing in every doc → df = 20 → PFOR.
        for i in 0..20 {
            let text = (0..20)
                .map(|j| format!("uniq{j:03}"))
                .collect::<Vec<_>>()
                .join(" ");
            b_pfor.add_doc(0, i, &text).expect("add doc");
        }
        let blob_pfor = b_pfor.finish().expect("finish pfor");

        // Extract postings-region sizes from the headers.
        let postings_off_i = u64::from_le_bytes(
            blob_inline[32..40]
                .try_into()
                .expect("postings_off_i slice is 8 bytes"),
        ) as usize;
        let dir_off_i = u64::from_le_bytes(
            blob_inline[40..48]
                .try_into()
                .expect("dir_off_i slice is 8 bytes"),
        ) as usize;
        let postings_size_inline = dir_off_i - postings_off_i;

        let postings_off_p = u64::from_le_bytes(
            blob_pfor[32..40]
                .try_into()
                .expect("postings_off_p slice is 8 bytes"),
        ) as usize;
        let dir_off_p = u64::from_le_bytes(
            blob_pfor[40..48]
                .try_into()
                .expect("dir_off_p slice is 8 bytes"),
        ) as usize;
        let postings_size_pfor = dir_off_p - postings_off_p;

        // Inline-only blob's postings region holds just the trailing
        // CRC32 (4 B). PFOR blob holds 20 terms × (20 B metadata +
        // 16 B skip table × 1 block + ~tens of bytes per PFOR block).
        assert_eq!(
            postings_size_inline, 4,
            "all-df=1 postings region should hold only the trailing CRC32; \
             got {postings_size_inline} bytes"
        );
        assert!(
            postings_size_pfor > 20 * 36,
            "PFOR postings region should be hundreds of bytes; got {postings_size_pfor}"
        );
    }
}
