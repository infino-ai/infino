// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

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
//! single-superfile SuperfileReader does no I/O after `open()`; a
//! storage layer can layer cold-fetch heuristics on top.

use std::{
    borrow::Cow,
    collections::HashMap,
    fmt, io,
    ops::{Deref, Range},
    str,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
    },
};

use arrow::compute::{concat_batches, interleave, take_record_batch};
use arrow_array::{
    Array, ArrayRef, Decimal128Array, LargeStringArray, RecordBatch, RecordBatchOptions,
    RecordBatchReader, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use futures::{
    FutureExt,
    future::{BoxFuture, try_join_all},
};
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{
            ArrowReaderBuilder, ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReader,
            ParquetRecordBatchReaderBuilder, RowSelection, RowSelector,
        },
        async_reader::MetadataFetch,
        parquet_to_arrow_schema,
    },
    errors::{ParquetError, Result as ParquetResult},
    file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader},
};
use rayon::ThreadPool;
use roaring::RoaringBitmap;
use tokio::sync::OnceCell;
use tracing::warn;

#[cfg(test)]
use crate::superfile::fts::reader::TermIndexFact;
use crate::{
    memory::ConnectionMemoryBudget,
    superfile::{
        BytesLazyByteSource, LazyByteSource, LazyByteSourceError, LazySubSource, ReadError,
        error::FtsError,
        format::{self, footer, kv},
        fts::{
            bm25::Bm25Params,
            reader::{
                self as fts_reader, BoolMode, ClauseLists, ContainsRows, FetchedTermMemo,
                FtsReader, MatchWork, OrCursorSet, PreparedClauses, ScoringView, TermPattern,
            },
            tokenize::{Phrase, Tokenizer},
        },
        id_space::{FtsDocId, RowId},
        ids::{self, PackedIds},
        vector::{
            layout::VectorLayout,
            reader::{self as vector_reader, ProbeTally, ScanCandidate, ScanOutcome, VectorReader},
        },
    },
    supertable::{query::provider::tombstone_access_plan, schema::FieldId},
    utils::terms::DictEntry,
};
/// Speculative Parquet-footer tail length for a lazy open. 64 KiB
/// covers a typical superfile footer (its `inf.*` KVs plus a single
/// row group's column metadata — a few KiB to a few tens of KiB) in
/// one range GET, so the cold open usually costs a single round-trip.
pub(crate) const DEFAULT_TAIL_SPECULATIVE_BYTES: u64 = 64 * 1024;

/// A cold `_id` read takes the whole stable-id sidecar once it asks for at
/// least one distinct row in this many. At half the rows nearly every page of
/// the `_id` column is touched, so reading the pages costs more GETs and more
/// bytes than the sidecar's few bytes a row; below it the column's page index
/// reads less.
const SIDECAR_MIN_ROW_SHARE: u64 = 2;

/// Gap between two stable-id sidecar blocks a lookup reads in one span
/// rather than two: past a block's few KB, a round trip costs more than
/// this many bytes of transfer.
const ID_BLOCK_GAP_BYTES: usize = if cfg!(test) { 0 } else { 128 * 1024 };

/// A take of at least one row in this many of the file, over fixed-width
/// columns only, reads each projected column whole and gathers the rows
/// from it, instead of decoding through a row selection. A selection skips
/// the pages that hold none of its rows, which a take this dense rarely
/// can, so it pays nearly the full decode anyway plus the selection's
/// bookkeeping per row: on a million-row file, a take of a fifth of the
/// rows of one float column through a selection measured 13 ms against
/// 1.7 ms for the whole-column read and gather, the aggregate around them
/// included. Below this share the pages a selection skips are worth more
/// than the bookkeeping costs.
///
/// A variable-width column keeps the selection at any density: its decode
/// is its values' bytes rather than a slot per row, so the pages a
/// selection skips cost more than its bookkeeping saves, most of all when
/// the rows cluster (matches close in time, as a log's are) and whole
/// pages hold none of them.
pub(crate) const WHOLE_COLUMN_TAKE_DIVISOR: u64 = 16;

pub(crate) fn vector_layout_from_kv(kv_map: &HashMap<String, String>) -> VectorLayout {
    kv_map
        .get(kv::VEC_LAYOUT)
        .and_then(|s| VectorLayout::from_kv_value(s))
        .unwrap_or(VectorLayout::Ivf)
}

/// Per-open knobs for [`SuperfileReader::open_with`]. Defaults to
/// safe behavior (CRC verification on); flip `verify_crc` to `false`
/// to skip the ~132 ms scan at 1M × 384 when storage is trusted.
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Verify all CRC32C checksums on open: the embedded
    /// vector blob's whole-blob + per-subsection CRCs, and
    /// the embedded FTS blob's four per-section CRCs (term dictionary,
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

/// A [`ScoringView`] owned, as the scored-view memo keys on it.
#[derive(Debug)]
struct ScoringKey {
    params: Option<Bm25Params>,
    avgdl: Option<(String, f32)>,
}

impl ScoringKey {
    /// Whether this key is `scoring`'s, compared without allocating.
    fn matches(&self, scoring: &ScoringView<'_>) -> bool {
        self.params == scoring.params
            && self
                .avgdl
                .as_ref()
                .map(|(column, avgdl)| (column.as_str(), *avgdl))
                == scoring.avgdl
    }

    fn of(scoring: &ScoringView<'_>) -> Self {
        Self {
            params: scoring.params,
            avgdl: scoring
                .avgdl
                .map(|(column, avgdl)| (column.to_owned(), avgdl)),
        }
    }
}

/// The FTS reader a scored query reads through: the one the file baked,
/// or a shared view of it scoring against something else. Derefs to
/// [`FtsReader`] so call sites do not care which they hold.
enum ScoredFts<'a> {
    Baked(&'a FtsReader),
    Derived(Arc<FtsReader>),
}

impl Deref for ScoredFts<'_> {
    type Target = FtsReader;

    fn deref(&self) -> &FtsReader {
        match self {
            ScoredFts::Baked(r) => r,
            ScoredFts::Derived(r) => r,
        }
    }
}

pub struct SuperfileReader {
    /// Full Parquet bytes, `Some` only for the eager [`open`]
    /// path. The lazy [`open_lazy`] path drops the
    /// whole-superfile hold — pass-through SQL / external-Parquet
    /// callers (`parquet_bytes`) see `None` and must take a
    /// different path. Vector + FTS queries work on either,
    /// since each carries its own source under the inner readers.
    bytes: Option<Bytes>,
    /// Parquet metadata parsed once at open (eager path only), reused
    /// by every `take_by_local_doc_ids` so targeted reads never
    /// re-parse the footer. `None` on the lazy path (no resident
    /// bytes). Carries the page index when present so `RowSelection`
    /// can skip whole pages.
    arrow_meta: Option<ArrowReaderMetadata>,
    /// The Parquet footer metadata parsed at open — set on BOTH paths
    /// (eager: shared with `arrow_meta`; lazy: the footer
    /// `open_lazy_with` already fetched). Lets the SQL scan path serve
    /// row-group counts and DataFusion's parquet opener from one
    /// per-open parse instead of re-reading footers on every query.
    parquet_meta: Arc<ParquetMetaData>,
    /// Lazy-reader metadata upgraded with Parquet offset indexes on the first
    /// targeted scalar read. Offset-index bytes flow through the same
    /// block-cached source and are decoded once per reader.
    page_index_meta: OnceCell<Arc<ParquetMetaData>>,
    /// The lazy byte source the reader was opened over, retained only
    /// on the [`open_lazy`] path. `None` on the eager [`open`] path
    /// (resident bytes already cover every range). Lets
    /// [`byte_source`](Self::byte_source) hand callers one uniform
    /// whole-superfile byte source regardless of how the reader opened.
    ///
    /// [`open_lazy`]: SuperfileReader::open_lazy
    /// [`open`]: SuperfileReader::open
    source: Option<Arc<dyn LazyByteSource>>,
    schema: Arc<Schema>,
    id_column: String,
    n_docs: u64,
    fts: Option<FtsReader>,
    /// The most recently derived scored FTS view, kept so a query that
    /// scores against anything other than what the file baked in does
    /// not rebuild one per call.
    ///
    /// Rebuilding is not free: it clones the column metadata and builds
    /// a fresh 256-entry decode table per column, and the scored path
    /// asks for a view more than once per superfile per query. Since
    /// table-wide statistics became the default that happens on every
    /// query rather than only on an explicit override, so a common term
    /// touching every superfile paid it once per superfile per call.
    ///
    /// One slot rather than a map: within a manifest generation every
    /// query on a column asks for the same override, so a single entry
    /// hits on all but the first. A table alternating between two FTS
    /// columns thrashes it and simply pays what it paid before.
    scored_fts: RwLock<Option<(ScoringKey, Arc<FtsReader>)>>,
    vec: Option<VectorReader>,
    /// Where the stable-id sidecar sits in the file, when the superfile
    /// carries one (absent on superfiles written before it existed). The
    /// sidecar holds every row's `_id` in row order, so it answers `_id`
    /// without decoding Parquet pages: a fully resident reader slices it
    /// ([`resident_id_sidecar`](Self::resident_id_sidecar)), and any reader
    /// can fetch it whole in one read
    /// ([`fetch_id_sidecar`](Self::fetch_id_sidecar)).
    id_sidecar: Option<IdSidecarLocation>,
    /// Set by the first read that finds a bad block in the sidecar. Later
    /// reads take `_id` from Parquet, so the sidecar is not retried or warned
    /// about again.
    id_sidecar_broken: AtomicBool,
    /// The `(offset, len)` the resident bytes do not hold, when they are a
    /// local copy with a hole ([`install_resident_parquet`]); every other byte
    /// of them is the file's own.
    ///
    /// [`install_resident_parquet`]: Self::install_resident_parquet
    resident_hole: Option<(u64, u64)>,
}

/// Where a superfile's stable-id sidecar lives and how it is laid out, from
/// the footer key-values. Its range was checked against the file length when
/// it was parsed ([`id_sidecar_location`]).
#[derive(Debug, Clone, Copy)]
struct IdSidecarLocation {
    offset: u64,
    len: u64,
    /// The packed layout (`superfile::ids`) rather than the raw `i128` array.
    packed: bool,
}

impl IdSidecarLocation {
    /// The sidecar, given its bytes.
    fn with_bytes(self, bytes: Bytes, n_docs: u64) -> IdSidecar {
        IdSidecar {
            bytes,
            packed: self.packed,
            n_docs,
        }
    }

    /// The sidecar cut from the whole file's `bytes`.
    fn slice(self, file: &Bytes, n_docs: u64) -> IdSidecar {
        let start = self.offset as usize;
        self.with_bytes(file.slice(start..start + self.len as usize), n_docs)
    }
}

/// A superfile's whole stable-id sidecar: one `_id` per local doc, in
/// Parquet row order (every writer builds it from the same rows, in the same
/// order, as the Parquet body). Decoding it is CPU over the rows asked for,
/// so it is a step of its own that the caller runs where CPU work belongs.
#[derive(Debug)]
pub(crate) struct IdSidecar {
    bytes: Bytes,
    packed: bool,
    n_docs: u64,
}

impl IdSidecar {
    /// Check the sidecar's shape without decoding it: a packed one's header,
    /// a raw one's length (sixteen bytes per doc). An eager open checks this
    /// much; [`ids`](Self::ids) also checks each block it reads.
    fn check(&self) -> Result<(), ReadError> {
        if self.packed {
            return PackedIds::open(&self.bytes, self.n_docs as usize)
                .map(|_| ())
                .map_err(ReadError::MalformedKv);
        }
        let expected = self.n_docs as usize * format::ID_SIDECAR_ENTRY_BYTES;
        if self.bytes.len() != expected {
            return Err(ReadError::MalformedKv(format!(
                "stable-id sidecar length {} != {expected} (16 x n_docs)",
                self.bytes.len()
            )));
        }
        Ok(())
    }

    /// The `_id` of each of `local_doc_ids`, in caller order; repeats and any
    /// order are fine. A doc past the end is the caller's mistake
    /// ([`ReadError::DocIdOutOfRange`]); a sidecar that does not decode is
    /// [`ReadError::MalformedKv`], and the caller can read the same ids from
    /// the Parquet `_id` column instead.
    pub(crate) fn ids(&self, local_doc_ids: &[u32]) -> Result<Vec<i128>, ReadError> {
        /// Width of one raw id, aliased from the format const for the
        /// fixed-size `from_le_bytes` decode.
        const ENTRY: usize = format::ID_SIDECAR_ENTRY_BYTES;
        if let Some(&doc_id) = local_doc_ids
            .iter()
            .find(|&&doc| u64::from(doc) >= self.n_docs)
        {
            return Err(ReadError::DocIdOutOfRange {
                doc_id,
                n_docs: self.n_docs,
            });
        }
        let mut out: Vec<i128> = Vec::with_capacity(local_doc_ids.len());
        if self.packed {
            // Opening the packed layout is its check. A block walk hands its
            // docs over ascending, so a large resolve unpacks each block's two
            // streams once instead of reading two bit fields per doc.
            PackedIds::open(&self.bytes, self.n_docs as usize)
                .map_err(ReadError::MalformedKv)?
                .get_many(local_doc_ids, &mut out)
                .ok_or_else(|| {
                    ReadError::MalformedKv("a block of the stable-id sidecar is malformed".into())
                })?;
            return Ok(out);
        }
        self.check()?;
        // In bounds: every doc is below `n_docs`, and the length is
        // `n_docs * ENTRY`.
        out.extend(local_doc_ids.iter().map(|&doc_id| {
            let start = doc_id as usize * ENTRY;
            let raw: [u8; ENTRY] = self.bytes[start..start + ENTRY]
                .try_into()
                .expect("a raw sidecar entry is ENTRY bytes");
            i128::from_le_bytes(raw)
        }));
        Ok(out)
    }

    /// Every row's `_id`, in row order, checked against what the footer
    /// already records about them. The sidecar carries no checksum, so a
    /// damaged block that still decodes would hand back wrong ids; these
    /// checks catch that before the ids place a hit or a delete:
    ///
    ///   - the ids span exactly `id_range`, the file's lowest and highest
    ///     `_id` as the table recorded them;
    ///   - each row group's ids have that group's `_id` column minimum and
    ///     maximum, where the footer carries them.
    ///
    /// A mismatch is [`ReadError::MalformedKv`], like a block that does not
    /// decode, so the caller reads the column instead.
    pub(crate) fn verified_ids(
        &self,
        metadata: &ParquetMetaData,
        id_column: &str,
        id_range: (i128, i128),
    ) -> Result<Vec<i128>, ReadError> {
        let every: Vec<u32> = (0..self.n_docs as u32).collect();
        let ids = self.ids(&every)?;
        let span = ids.iter().min().zip(ids.iter().max());
        if span.is_some_and(|(&lo, &hi)| (lo, hi) != id_range) {
            return Err(ReadError::MalformedKv(
                "the stable-id sidecar's ids do not span the file's id range".into(),
            ));
        }
        let Some(column) = metadata
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .position(|c| c.name() == id_column)
        else {
            return Ok(ids);
        };
        let mut start = 0usize;
        for (group, row_group) in metadata.row_groups().iter().enumerate() {
            let rows = usize::try_from(row_group.num_rows()).unwrap_or(0);
            let end = start.saturating_add(rows).min(ids.len());
            let slice = &ids[start..end];
            start = end;
            let Some((stat_min, stat_max)) = row_group.column(column).statistics().and_then(|s| {
                Some((
                    decimal_stat(s.min_bytes_opt()?)?,
                    decimal_stat(s.max_bytes_opt()?)?,
                ))
            }) else {
                continue;
            };
            let found = slice.iter().min().zip(slice.iter().max());
            if found != Some((&stat_min, &stat_max)) {
                return Err(ReadError::MalformedKv(format!(
                    "the stable-id sidecar disagrees with row group {group}'s `_id` statistics"
                )));
            }
        }
        Ok(ids)
    }
}

/// Whether `local_doc_ids` names at least one distinct row of a file of
/// `n_docs` in every [`SIDECAR_MIN_ROW_SHARE`]. Counts distinct rows, so a
/// list with repeats is not mistaken for coverage, and rows past the end do
/// not count.
fn covers_row_share(local_doc_ids: &[u32], n_docs: u64) -> bool {
    let needed = n_docs.div_ceil(SIDECAR_MIN_ROW_SHARE);
    if n_docs == 0 || (local_doc_ids.len() as u64) < needed {
        return false;
    }
    let distinct: RoaringBitmap = local_doc_ids
        .iter()
        .copied()
        .filter(|&doc| u64::from(doc) < n_docs)
        .collect();
    distinct.len() >= needed
}

/// A `Decimal128` column statistic as stored in the footer: sixteen
/// big-endian bytes. `None` for any other width (a truncated statistic says
/// nothing exact).
fn decimal_stat(bytes: &[u8]) -> Option<i128> {
    Some(i128::from_be_bytes(bytes.try_into().ok()?))
}

struct LazyMetadataFetch {
    source: Arc<dyn LazyByteSource>,
}

impl MetadataFetch for &mut LazyMetadataFetch {
    fn fetch(&mut self, range: Range<u64>) -> BoxFuture<'_, ParquetResult<Bytes>> {
        let source = Arc::clone(&self.source);
        async move {
            let len = range.end.checked_sub(range.start).ok_or_else(|| {
                ParquetError::General(format!("invalid metadata range {range:?}"))
            })?;
            source
                .range(range.start, len)
                .await
                .map_err(|error| ParquetError::External(Box::new(error)))
        }
        .boxed()
    }
}

impl fmt::Debug for SuperfileReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SuperfileReader")
            .field("id_column", &self.id_column)
            .field("n_docs", &self.n_docs)
            .field("has_fts", &self.fts.is_some())
            .field("has_vec", &self.vec.is_some())
            .field("bytes_len", &self.bytes.as_ref().map(|b| b.len()))
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

    /// Open a superfile via a shared [`LazyByteSource`].
    ///
    /// Cold-open range budget:
    ///
    /// 1. **1-2 GETs** for the Parquet footer (`inf.*` KV
    ///    metadata + Arrow schema), via
    ///    `format::footer::read_parquet_metadata_lazy`.
    /// 2. **3-4 GETs** for the embedded vector subsection, via
    ///    `VectorReader::open_lazy` (outer header, directory + CRC,
    ///    subsection headers, and Sq8 codec_meta when present).
    /// 3. **2-3 GETs** for the embedded FTS subsection, via
    ///    `FtsReader::open_lazy` (header, doc-lengths directory, and the
    ///    doc-id map when the file has one; the dictionary, each column's
    ///    length array and the postings stay lazy until a query needs
    ///    them).
    ///
    /// Total open budget is small exact metadata ranges rather than
    /// whole-subsection/speculative slabs. Subsequent vector queries
    /// fetch centroids, cluster indexes, and per-cluster blocks on
    /// demand via the source.
    ///
    /// The returned reader does **not** hold the full superfile;
    /// `parquet_bytes()` returns `None`. Callers that need the
    /// full Parquet bytes (DataFusion register, DuckDB,
    /// pyarrow) must use the eager [`open`] path.
    ///
    /// [`open`]: SuperfileReader::open
    pub async fn open_lazy(source: Arc<dyn LazyByteSource>) -> Result<Self, ReadError> {
        Self::open_lazy_with(source, OpenOptions::default()).await
    }

    /// Like [`open_lazy`] but with explicit [`OpenOptions`].
    ///
    /// Lazy opens do not run whole-superfile CRC scans. Forcing CRC
    /// verification here would require reading the full superfile through
    /// range GETs, which is exactly what the lazy path is meant to avoid;
    /// the embedded vector/FTS lazy readers therefore use their
    /// object-store options (`verify_crc = false`) while eager cache
    /// promotion can verify after the full superfile is materialized.
    ///
    /// [`open_lazy`]: SuperfileReader::open_lazy
    #[cfg_attr(feature = "detailed-tracing", tracing::instrument(skip_all))]
    pub async fn open_lazy_with(
        source: Arc<dyn LazyByteSource>,
        _opts: OpenOptions,
    ) -> Result<Self, ReadError> {
        // 1. Fetch the Parquet footer (≤ 2 GETs).
        let metadata =
            footer::read_parquet_metadata_lazy(source.as_ref(), DEFAULT_TAIL_SPECULATIVE_BYTES)
                .await
                .map_err(ReadError::Footer)?;
        let kv_map = footer::extract_kv_map(&metadata).map_err(ReadError::Footer)?;

        // 2. Validate required KVs + format version (same
        //    checks as the eager `open_with` path).
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
        let version = format::Version::parse(version_str).ok_or_else(|| {
            ReadError::Malformed(format!("format-version {version_str:?} does not parse"))
        })?;
        if !version.is_compatible_with_current() {
            return Err(ReadError::UnsupportedVersion(version_str.clone()));
        }

        let id_column = kv_map.get(kv::ID_COLUMN).expect("checked above").clone();
        let n_docs: u64 = kv_map
            .get(kv::N_DOCS)
            .expect("checked above")
            .parse()
            .map_err(|_| ReadError::MalformedKv(format!("{} not a u64", kv::N_DOCS)))?;

        // 3. Arrow schema from decoded Parquet metadata — no
        //    extra range GET. The parsed footer is retained on the
        //    reader (`parquet_meta`) so later consumers (the SQL scan
        //    path) never re-fetch or re-parse it.
        let file_meta = metadata.file_metadata();
        let schema = Arc::new(
            parquet_to_arrow_schema(file_meta.schema_descr(), file_meta.key_value_metadata())
                .map_err(|e| ReadError::Footer(footer::FooterError::Parquet(e)))?,
        );
        let parquet_meta = Arc::new(metadata);

        // 4 + 5. Vector + FTS subsections — Tail-fetch path:
        //   fires both subsection fetches **concurrently** via
        //   `futures::try_join!`. The two subsections live at
        //   disjoint offsets (parquet body → fts → vec → footer
        //   in the layout produced by `splice_index_blobs`)
        //   so neither depends on the other; on a network-backed
        //   `LazyByteSource` this collapses two serial RTTs
        //   into one parallel RTT. On warm/in-memory sources
        //   both branches resolve through the sync zero-copy
        //   path with no extra cost.
        //
        // Each branch validates its `inf.{fts,vec}.*` KV
        // shape before starting any I/O so partial-KV
        // misconfigurations fail fast (and don't trigger a
        // wasted range GET).
        let vec_present = all_present(&kv_map, kv::VEC_KEYS);
        if !vec_present && any_present(&kv_map, kv::VEC_KEYS) {
            return Err(ReadError::MalformedKv(
                "partial inf.vec.* keys present".into(),
            ));
        }
        let fts_present = all_present(&kv_map, kv::FTS_KEYS);
        if !fts_present && any_present(&kv_map, kv::FTS_KEYS) {
            return Err(ReadError::MalformedKv(
                "partial inf.fts.* keys present".into(),
            ));
        }

        let vec_layout = vector_layout_from_kv(&kv_map);
        let vec_fut = async {
            if !vec_present {
                return Ok::<_, ReadError>(None);
            }
            if vec_layout == VectorLayout::CellPosting {
                return Ok(None);
            }
            let off = parse_u64(&kv_map, kv::VEC_OFFSET)?;
            let len = parse_u64(&kv_map, kv::VEC_LENGTH)?;
            let cols_json = kv_map.get(kv::VEC_COLUMNS).expect("checked");
            let sub: Arc<dyn LazyByteSource> =
                Arc::new(LazySubSource::new(Arc::clone(&source), off, len));
            let reader = VectorReader::open_lazy(
                sub,
                cols_json,
                vector_reader::OpenOptions::for_object_store(),
            )
            .await?;
            Ok(Some(reader))
        };

        let fts_fut = async {
            if !fts_present {
                return Ok::<_, ReadError>(None);
            }
            let off = parse_u64(&kv_map, kv::FTS_OFFSET)?;
            let len = parse_u64(&kv_map, kv::FTS_LENGTH)?;
            let cols_json = kv_map.get(kv::FTS_COLUMNS).expect("checked");
            let sub: Arc<dyn LazyByteSource> =
                Arc::new(LazySubSource::new(Arc::clone(&source), off, len));
            let reader =
                FtsReader::open_lazy(sub, cols_json, fts_reader::OpenOptions::for_object_store())
                    .await?;
            Ok(Some(reader))
        };

        let (vec, fts) = futures::try_join!(vec_fut, fts_fut)?;
        // Lazy reads use the sidecar only to fetch every `_id` at once, a
        // shortcut with the Parquet column behind it, so a malformed location
        // switches the shortcut off rather than failing the open.
        let id_sidecar = id_sidecar_location(&kv_map, source.size()).unwrap_or_else(|error| {
            warn!(%error, "ignoring the stable-id sidecar of a lazily opened superfile");
            None
        });

        Ok(Self {
            bytes: None,
            arrow_meta: None,
            parquet_meta,
            page_index_meta: OnceCell::new(),
            source: Some(source),
            schema,
            id_column,
            n_docs,
            fts,
            scored_fts: RwLock::new(None),
            vec,
            // No resident bytes to slice on the lazy path; a read of a few
            // `_id`s goes through the Parquet id column, and a read of every
            // one fetches the sidecar through `fetch_id_sidecar`.
            id_sidecar,
            id_sidecar_broken: AtomicBool::new(false),
            resident_hole: None,
        })
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
        let version = format::Version::parse(version_str).ok_or_else(|| {
            ReadError::Malformed(format!("format-version {version_str:?} does not parse"))
        })?;
        if !version.is_compatible_with_current() {
            return Err(ReadError::UnsupportedVersion(version_str.clone()));
        }

        let id_column = kv_map.get(kv::ID_COLUMN).expect("checked above").clone();
        let n_docs: u64 = kv_map
            .get(kv::N_DOCS)
            .expect("checked above")
            .parse()
            .map_err(|_| ReadError::MalformedKv(format!("{} not a u64", kv::N_DOCS)))?;

        // 3. Parse the Parquet metadata once and cache it on the reader.
        //    `Bytes` implements `ChunkReader` directly so this is
        //    zero-copy, and every later `take_by_local_doc_ids` reuses
        //    this `ArrowReaderMetadata` instead of re-parsing the footer
        //    per call.
        //
        //    The page index — a decoded min/max pair per page, per column,
        //    per row group — is deliberately NOT parsed here. It exists to
        //    let `RowSelection` skip pages on targeted reads, and only the
        //    take and SQL-scan paths use it; BM25 and COUNT never touch it.
        //    Parsing it at open put tens of MiB on the heap per superfile
        //    for every reader, including the fan-out readers that will
        //    only ever count. `parquet_metadata_with_page_index` parses it
        //    on first use instead, from these same resident bytes, so the
        //    paths that need it see identical metadata and pay no I/O for
        //    it either way.
        let arrow_meta = ArrowReaderMetadata::load(&bytes, ArrowReaderOptions::new())
            .map_err(|e| ReadError::Footer(footer::FooterError::Parquet(e)))?;
        let schema = arrow_meta.schema().clone();

        // 4. If FTS keys present, slice + open FtsReader.
        let fts = if all_present(&kv_map, kv::FTS_KEYS) {
            let off = parse_u64(&kv_map, kv::FTS_OFFSET)?;
            let len = parse_u64(&kv_map, kv::FTS_LENGTH)?;
            let cols_json = kv_map.get(kv::FTS_COLUMNS).expect("checked");
            let blob = slice_or_err(&bytes, off, len, "FTS")?;
            Some(FtsReader::open_with(
                blob,
                cols_json,
                fts_reader::OpenOptions {
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

        // 5. Vector path mirrors FTS (cell posting blobs are scanned separately).
        let vec_layout = vector_layout_from_kv(&kv_map);
        let vec = if all_present(&kv_map, kv::VEC_KEYS) {
            if vec_layout == VectorLayout::CellPosting {
                None
            } else {
                let off = parse_u64(&kv_map, kv::VEC_OFFSET)?;
                let len = parse_u64(&kv_map, kv::VEC_LENGTH)?;
                let cols_json = kv_map.get(kv::VEC_COLUMNS).expect("checked");
                let blob = slice_or_err(&bytes, off, len, "vector")?;
                Some(VectorReader::open_with(
                    blob,
                    cols_json,
                    vector_reader::OpenOptions {
                        verify_crc: opts.verify_crc,
                    },
                )?)
            }
        } else if any_present(&kv_map, kv::VEC_KEYS) {
            return Err(ReadError::MalformedKv(
                "partial inf.vec.* keys present".into(),
            ));
        } else {
            None
        };

        // 6. Stable-id sidecar (optional): every row's `_id` in row order,
        //    packed (`superfile::ids`) or raw (a little-endian `i128` per
        //    doc). Recorded so `take_by_local_doc_ids` answers `_id` without
        //    decoding the Parquet id pages. Absent on superfiles written
        //    before it existed, which then read the Parquet column. Only its
        //    shape is checked here (a packed header, or a raw length); each
        //    packed block is checked when it is read.
        let id_sidecar = id_sidecar_location(&kv_map, bytes.len() as u64)?;
        if let Some(loc) = id_sidecar {
            loc.slice(&bytes, n_docs).check()?;
        }

        Ok(Self {
            bytes: Some(bytes),
            parquet_meta: Arc::clone(arrow_meta.metadata()),
            // Filled by `parquet_metadata_with_page_index` on first use.
            page_index_meta: OnceCell::new(),
            arrow_meta: Some(arrow_meta),
            source: None,
            schema,
            id_column,
            n_docs,
            fts,
            scored_fts: RwLock::new(None),
            vec,
            id_sidecar,
            id_sidecar_broken: AtomicBool::new(false),
            resident_hole: None,
        })
    }

    /// The Parquet footer metadata parsed at open — page index included
    /// on the eager path. One parse per reader lifetime; the SQL scan
    /// path serves row-group counts and DataFusion's parquet opener
    /// from this instead of re-reading the footer per query.
    pub fn parquet_metadata(&self) -> &Arc<ParquetMetaData> {
        &self.parquet_meta
    }

    /// [`Self::parquet_metadata_with_page_index`] when it needs no read or
    /// decode: the open already loaded the page indexes, or an earlier call
    /// did. `None` means that call would do work.
    pub(crate) fn loaded_parquet_metadata_with_page_index(&self) -> Option<Arc<ParquetMetaData>> {
        if self.parquet_meta.offset_index().is_some() {
            return Some(Arc::clone(&self.parquet_meta));
        }
        self.page_index_meta.get().cloned()
    }

    /// Parquet metadata with the page indexes (column + offset) loaded
    /// when available.
    ///
    /// Eager readers return their open-time metadata. Lazy readers fetch
    /// only the page-index metadata range through their existing byte
    /// source; a [`OnceCell`] coalesces concurrent first callers and keeps
    /// all later targeted row reads local.
    ///
    /// Both indexes load together even though the take path needs only the
    /// offset index: the SQL scan serves this same parse to DataFusion's
    /// parquet opener, whose page pruning short-circuits only when the
    /// column index is present too. An index-complete parse here means the
    /// opener never fetches index bytes through the metered DataFusion
    /// store — which would make `sql_page_bytes` depend on how the reader
    /// happened to be opened (see `SuperfileObjectStore`).
    pub(crate) async fn parquet_metadata_with_page_index(
        &self,
    ) -> Result<Arc<ParquetMetaData>, ReadError> {
        if let Some(loaded) = self.loaded_parquet_metadata_with_page_index() {
            return Ok(loaded);
        }
        let source = self.source.as_ref().map(Arc::clone);
        let bytes = self.bytes.clone();
        let metadata = self
            .page_index_meta
            .get_or_try_init(|| async move {
                let Some(source) = source else {
                    // Eager / mmap reader: the whole file is already
                    // resident, so parse the index straight out of it
                    // rather than going back to storage. Keeping this off
                    // the metered DataFusion store is what makes page-byte
                    // accounting independent of how the reader was opened.
                    let Some(bytes) = bytes else {
                        return Ok::<Arc<ParquetMetaData>, ReadError>(Arc::clone(
                            &self.parquet_meta,
                        ));
                    };
                    let indexed = ArrowReaderMetadata::load(
                        &bytes,
                        ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Optional),
                    )
                    .map_err(|e| ReadError::Footer(footer::FooterError::Parquet(e)))?;
                    return Ok(Arc::clone(indexed.metadata()));
                };
                let mut fetch = LazyMetadataFetch { source };
                let mut loader =
                    ParquetMetaDataReader::new_with_metadata((*self.parquet_meta).clone())
                        .with_column_index_policy(PageIndexPolicy::Optional)
                        .with_offset_index_policy(PageIndexPolicy::Optional);
                loader
                    .load_page_index(&mut fetch)
                    .await
                    .map_err(|error| ReadError::Footer(footer::FooterError::Parquet(error)))?;
                let metadata = loader
                    .finish()
                    .map_err(|error| ReadError::Footer(footer::FooterError::Parquet(error)))?;
                Ok(Arc::new(metadata))
            })
            .await?;
        Ok(Arc::clone(metadata))
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
    /// The name this file knows the column `id` by, for a caller holding
    /// the table's current name for it. A file written before a rename
    /// carries the old label in its FTS and vector blobs, which key their
    /// columns by name; the id is what identifies the column across both.
    /// Falls back to `name` for a file written before ids, whose labels
    /// were the table's at the time.
    pub(crate) fn column_alias<'a>(&'a self, id: Option<FieldId>, name: &'a str) -> &'a str {
        let Some(id) = id else {
            return name;
        };
        let fts = self
            .fts()
            .into_iter()
            .flat_map(|fts| fts.fts_columns_config())
            .find(|c| c.field_id == Some(id))
            .map(|c| c.name.as_str());
        fts.or_else(|| {
            self.vec()
                .into_iter()
                .flat_map(|vec| vec.vector_columns_config())
                .find(|c| c.field_id == Some(id))
                .map(|c| c.name.as_str())
        })
        .unwrap_or(name)
    }

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

    /// Attribute every byte of a resident superfile to a region — the
    /// Parquet row groups (per column), the `_id` sidecar, the vector
    /// index, the FTS blob (itself broken down per term band by
    /// [`FtsReader::size_breakdown`]) and the footer. A report over the
    /// whole file, never a query path; `None` when the reader is not
    /// fully resident (a lazy object-store open), since the total is the
    /// resident length.
    pub fn size_breakdown(&self) -> Result<Option<SuperfileSizeBreakdown>, ReadError> {
        let Some(bytes) = self.bytes.as_ref() else {
            return Ok(None);
        };
        let kv_map = footer::extract_kv_map(&self.parquet_meta).map_err(ReadError::Footer)?;
        let kv_len = |key: &str| -> u64 {
            kv_map
                .get(key)
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0)
        };
        let mut columns: Vec<(String, u64)> = Vec::new();
        let mut row_group_bytes = 0u64;
        for rg in self.parquet_meta.row_groups() {
            for c in rg.columns() {
                let size = c.compressed_size().max(0) as u64;
                row_group_bytes += size;
                let name = c.column_path().string();
                match columns.iter_mut().find(|(n, _)| *n == name) {
                    Some((_, b)) => *b += size,
                    None => columns.push((name, size)),
                }
            }
        }
        let fts = match &self.fts {
            Some(r) => Some(r.size_breakdown().map_err(fts_reader_error_to_read)?),
            None => None,
        };
        Ok(Some(SuperfileSizeBreakdown {
            total_bytes: bytes.len() as u64,
            n_docs: self.n_docs,
            row_groups: self.parquet_meta.row_groups().len() as u64,
            row_group_bytes,
            columns,
            id_sidecar_bytes: self.id_sidecar.map_or(0, |loc| loc.len),
            vec_bytes: kv_len(kv::VEC_LENGTH),
            fts_bytes: kv_len(kv::FTS_LENGTH),
            fts,
        }))
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
    ///
    /// Returns `None` for readers opened via [`open_lazy`] — the
    /// lazy path does not materialize the full superfile,
    /// so external-Parquet pass-throughs need either the eager
    /// [`open`] path or an explicit `LazyByteSource::range(0, size)`
    /// against the source.
    ///
    /// On a promoted hybrid reader ([`install_resident_parquet`]) the
    /// returned bytes leave the embedded **vector blob sparse**; every
    /// parquet-referenced byte and the FTS subsection are real. External
    /// parquet readers never dereference the vector region, but consumers
    /// that need the complete superfile must gate on `is_fully_resident`.
    ///
    /// [`open`]: SuperfileReader::open
    /// [`open_lazy`]: SuperfileReader::open_lazy
    /// [`install_resident_parquet`]: SuperfileReader::install_resident_parquet
    pub fn parquet_bytes(&self) -> Option<&Bytes> {
        self.bytes.as_ref()
    }

    /// Whether synchronous targeted row materialization is ready. A cache
    /// reader can expose resident bytes before its Arrow metadata is installed;
    /// those readers must stay on the async object-store row path.
    pub(crate) fn can_take_by_local_doc_ids(&self) -> bool {
        self.bytes.is_some() && self.arrow_meta.is_some()
    }

    /// Whether every byte of the superfile is resident and real — the eager
    /// [`open`] shape. `false` for lazy readers and for promoted hybrid
    /// readers whose resident bytes leave the vector blob sparse
    /// ([`install_resident_parquet`]); consumers that read vector bytes
    /// synchronously (compaction's Sq8 merge) must gate on this, not on
    /// [`parquet_bytes`] presence.
    ///
    /// [`open`]: SuperfileReader::open
    /// [`parquet_bytes`]: SuperfileReader::parquet_bytes
    /// [`install_resident_parquet`]: SuperfileReader::install_resident_parquet
    pub(crate) fn is_fully_resident(&self) -> bool {
        self.bytes.is_some() && self.source.is_none()
    }

    /// Whether the stable-id sidecar is in the packed layout.
    pub(crate) fn id_sidecar_is_packed(&self) -> bool {
        self.id_sidecar.is_some_and(|loc| loc.packed)
    }

    /// Whole-file bytes, but only when every region of them is valid.
    ///
    /// `bytes` alone is not enough: a hybrid-promoted reader carries a
    /// fill-file mmap whose vector region is left sparse, so a caller
    /// copying blob ranges out of it would copy zeros. Gated on
    /// [`Self::is_fully_resident`], which that promotion never satisfies.
    pub(crate) fn whole_file_bytes(&self) -> Option<&Bytes> {
        self.is_fully_resident().then_some(self.bytes.as_ref())?
    }

    /// Install resident whole-file bytes on a lazily-opened reader so the
    /// synchronous parquet paths (`take_by_local_doc_ids`,
    /// `get_record_batch`, `id_lookup`) run main-style sync decodes instead
    /// of the async stream machinery. Used by the disk cache's hybrid mmap
    /// promotion, where `bytes` is the fill-file mmap with the **vector blob
    /// left sparse**: `hole` is that blob's `(offset, len)`, and every byte
    /// outside it is the file's own. The FTS/vector sub-readers keep the
    /// sources they were opened over; [`byte_source`] keeps preferring the
    /// reader's lazy source, so no consumer can reach the sparse region
    /// through this reader.
    ///
    /// [`byte_source`]: SuperfileReader::byte_source
    pub(crate) fn install_resident_parquet(
        &mut self,
        bytes: Bytes,
        hole: (u64, u64),
    ) -> Result<(), ReadError> {
        self.resident_hole = Some(hole);
        let arrow_meta = ArrowReaderMetadata::load(
            &bytes,
            ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Optional),
        )
        .map_err(|e| ReadError::Footer(footer::FooterError::Parquet(e)))?;
        // Seed the page-index cell too (best effort — a concurrent lazy
        // upgrade may have won; either value serves the same reads).
        let _ = self.page_index_meta.set(Arc::clone(arrow_meta.metadata()));
        self.arrow_meta = Some(arrow_meta);
        self.bytes = Some(bytes);
        Ok(())
    }
    /// Returns a record batch containing all documents with all columns.
    /// Test-only: joining the batches overflows a `Utf8` column past 2 GiB,
    /// so production code reads with `record_batches`.
    #[cfg(test)]
    pub fn get_record_batch(
        &self,
        deleted_docs_bitmap: Option<Arc<RoaringBitmap>>,
    ) -> Result<RecordBatch, ReadError> {
        let reader = self.record_batches(deleted_docs_bitmap)?;
        let read_schema = reader.schema();
        let batches = reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ReadError::Columnar(e.to_string()))?;
        let record_batch = concat_batches(&read_schema, &batches)
            .map_err(|e| ReadError::Columnar(e.to_string()))?;

        Ok(record_batch)
    }

    /// All documents (minus `deleted_docs_bitmap`) with all columns, one
    /// decoded batch at a time. The batches are never joined, so a string column past 2 GiB, more than
    /// one `Utf8` array's 32-bit offsets can hold, still reads.
    pub(crate) fn record_batches(
        &self,
        deleted_docs_bitmap: Option<Arc<RoaringBitmap>>,
    ) -> Result<ParquetRecordBatchReader, ReadError> {
        let bytes = self
            .bytes
            .as_ref()
            .ok_or(ReadError::LazyReaderUnsupported)?;
        let arrow_meta = self
            .arrow_meta
            .as_ref()
            .ok_or(ReadError::LazyReaderUnsupported)?
            .clone();
        let plan = if let Some(deleted_docs_bitmap) = deleted_docs_bitmap {
            tombstone_access_plan(bytes, deleted_docs_bitmap.as_ref())
                .map_err(|e| ReadError::Columnar(e.to_string()))?
        } else {
            None
        };

        let mut builder =
            ParquetRecordBatchReaderBuilder::new_with_metadata(bytes.clone(), arrow_meta);
        if let Some(plan) = plan {
            let row_groups = plan.row_group_indexes();
            let selection = plan
                .into_overall_row_selection(builder.metadata().row_groups())
                .map_err(|e| ReadError::Columnar(e.to_string()))?;

            builder = builder.with_row_groups(row_groups);
            if let Some(selection) = selection {
                builder = builder.with_row_selection(selection);
            }
        }
        builder
            .build()
            .map_err(|e| ReadError::Columnar(e.to_string()))
    }

    /// The whole stable-id sidecar, fetched with one ranged read through
    /// [`byte_source`](Self::byte_source). On a cold reader that is one
    /// request for a few bytes per row, where reading every `_id` from the
    /// Parquet column costs a request per column chunk. `None` when the
    /// superfile carries no sidecar (written before it existed) or an earlier
    /// read found it does not decode
    /// ([`mark_id_sidecar_broken`](Self::mark_id_sidecar_broken)); the caller
    /// then reads `_id` from the Parquet column, which holds the same ids.
    pub(crate) async fn fetch_id_sidecar(&self) -> Result<Option<IdSidecar>, ReadError> {
        let Some(loc) = self.id_sidecar else {
            return Ok(None);
        };
        if self.id_sidecar_broken.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let bytes = self
            .byte_source()
            .range(loc.offset, loc.len)
            .await
            .map_err(|source| ReadError::RangeFetch {
                what: "stable-id sidecar fetch",
                source,
            })?;
        // A short buffer is a failed read, not a damaged sidecar: it must
        // stay in the fetch class, so the caller retries the sidecar on a
        // later read instead of marking a healthy one broken.
        if bytes.len() as u64 != loc.len {
            return Err(ReadError::RangeFetch {
                what: "stable-id sidecar fetch",
                source: LazyByteSourceError::ShortRead {
                    start: loc.offset,
                    requested: loc.len,
                    got: bytes.len() as u64,
                },
            });
        }
        Ok(Some(loc.with_bytes(bytes, self.n_docs)))
    }

    /// Whether a read of `local_doc_ids` from `id_column` should fetch the
    /// stable-id sidecar whole ([`fetch_id_sidecar`](Self::fetch_id_sidecar))
    /// rather than read the Parquet column or the vector blob's inline ids.
    /// All four must hold:
    ///
    ///   - no resident Parquet: a resident reader answers without any read;
    ///   - at least one distinct row in [`SIDECAR_MIN_ROW_SHARE`] of the file
    ///     asked for: the sidecar is fetched whole, while a smaller subset
    ///     reads only its own pages through the column's page index, and at
    ///     that share those pages are nearly all of the column;
    ///   - the table's id column: the only column the sidecar holds;
    ///   - the packed layout: a raw sidecar is sixteen bytes a row, more than
    ///     the compressed column it would replace.
    pub(crate) fn id_sidecar_serves(&self, local_doc_ids: &[u32], id_column: &str) -> bool {
        self.bytes.is_none()
            && id_column == self.id_column
            && self.id_sidecar_is_packed()
            && covers_row_share(local_doc_ids, self.n_docs)
    }

    /// Record that this superfile's sidecar does not decode, so later reads
    /// take `_id` from another source without fetching it again.
    pub(crate) fn mark_id_sidecar_broken(&self) {
        self.id_sidecar_broken.store(true, Ordering::Relaxed);
    }

    /// A [`LazyByteSource`] over the **entire** superfile, regardless of
    /// how the reader was opened. The single byte-access handle the
    /// SQL/DataFusion path reads through -- callers never branch on
    /// storage mode:
    ///
    /// - resident-bytes readers (eager [`open`], disk-cache mmap) wrap
    ///   their bytes in a [`BytesLazyByteSource`]; every `range` is a
    ///   zero-copy `Bytes::slice` (a refcount bump, no copy).
    /// - lazy readers ([`open_lazy`]) return the source they were
    ///   opened over, so ranges stream straight from object storage.
    ///
    /// [`LazyByteSource`]: crate::superfile::LazyByteSource
    /// [`BytesLazyByteSource`]: crate::superfile::BytesLazyByteSource
    /// [`open`]: SuperfileReader::open
    /// [`open_lazy`]: SuperfileReader::open_lazy
    pub fn byte_source(&self) -> Arc<dyn LazyByteSource> {
        match (&self.bytes, &self.source) {
            // Source-first: a promoted hybrid reader has BOTH — its bytes
            // leave the vector blob sparse (`install_resident_parquet`), so
            // whole-file byte access must go through the source, whose
            // vector-hole fallback serves real bytes. Eager readers have no
            // source and keep the zero-copy bytes wrap.
            (_, Some(src)) => Arc::clone(src),
            (Some(bytes), None) => Arc::new(BytesLazyByteSource::new(bytes.clone())),
            (None, None) => {
                unreachable!("a SuperfileReader has either resident bytes or a lazy source")
            }
        }
    }

    /// Resolve in-superfile row offsets to their durable identity.
    ///
    /// Given a slice of `local_doc_id`s — the per-superfile row
    /// offsets produced by [`bm25_search`](Self::bm25_search) /
    /// [`vector_search`](Self::vector_search) and carried in a
    /// `SuperfileHit` — return a [`RecordBatch`] of the requested
    /// `projection` columns at exactly those rows, in the same
    /// order as `local_doc_ids`. This is the bridge from a hit's
    /// `(superfile, local_doc_id)` to the supertable's durable `_id`
    /// plus any projected scalar columns: pass
    /// [`id_column`](Self::id_column) in `projection` to recover the
    /// primary key, and zip the result rows back to the per-hit
    /// scores positionally (row `i` is the superfile row at
    /// `local_doc_ids[i]`).
    ///
    /// Output columns are `projection` in the given order (not file
    /// order), and only those columns are decoded (column-projected
    /// read). Duplicate or repeated offsets are honored as-is.
    ///
    /// # Errors
    /// - [`ReadError::LazyReaderUnsupported`] — opened via
    ///   [`open_lazy`](Self::open_lazy); no materialized bytes.
    /// - [`ReadError::UnknownColumn`] — a name is not in
    ///   [`schema`](Self::schema).
    /// - [`ReadError::DocIdOutOfRange`] — an offset is `>= n_docs()`.
    /// - [`ReadError::Columnar`] — a Parquet/Arrow decode failure.
    pub fn take_by_local_doc_ids(
        &self,
        local_doc_ids: &[u32],
        projection: &[&str],
    ) -> Result<RecordBatch, ReadError> {
        self.take_rows(local_doc_ids, projection, true)
    }

    /// [`take_by_local_doc_ids`](Self::take_by_local_doc_ids) with `_id`
    /// decoded from the Parquet column even where the stable-id sidecar could
    /// answer, for a caller rebuilding that sidecar: its ids must come from
    /// the column, not from the copy being replaced.
    pub(crate) fn take_by_local_doc_ids_from_parquet(
        &self,
        local_doc_ids: &[u32],
        projection: &[&str],
    ) -> Result<RecordBatch, ReadError> {
        self.take_rows(local_doc_ids, projection, false)
    }

    /// The rows behind both takes; `id_from_sidecar` lets the stable-id
    /// sidecar answer `_id` when the reader holds it.
    fn take_rows(
        &self,
        local_doc_ids: &[u32],
        projection: &[&str],
        id_from_sidecar: bool,
    ) -> Result<RecordBatch, ReadError> {
        let bytes = self
            .bytes
            .as_ref()
            .ok_or(ReadError::LazyReaderUnsupported)?
            .clone();
        let arrow_meta = self
            .arrow_meta
            .as_ref()
            .ok_or(ReadError::LazyReaderUnsupported)?
            .clone();

        // 1. Resolve projected names → output fields (in caller order),
        //    validating every name up front. Per-source column selection
        //    (sidecar vs Parquet) is decided below.
        let mut out_fields: Vec<Field> = Vec::with_capacity(projection.len());
        for &name in projection {
            let idx = self
                .schema
                .index_of(name)
                .map_err(|_| ReadError::UnknownColumn(name.to_string()))?;
            out_fields.push(self.schema.field(idx).clone());
        }
        let out_schema = Arc::new(Schema::new(out_fields));

        // 2. Bounds-check every requested offset up front so a
        //    single bad id is a typed error, not a silent
        //    truncation or a confused parquet error later.
        for &doc_id in local_doc_ids {
            if u64::from(doc_id) >= self.n_docs {
                return Err(ReadError::DocIdOutOfRange {
                    doc_id,
                    n_docs: self.n_docs,
                });
            }
        }

        // 3. Empty input short-circuits to an empty batch with the
        //    projected schema — no parquet decode, no allocation
        //    beyond the schema. Preserves the pre-existing contract
        //    for callers that do speculative resolves with k=0 hits.
        if local_doc_ids.is_empty() {
            return Ok(RecordBatch::new_empty(out_schema));
        }

        // On a fully resident reader, `_id` resolves from the stable-id
        // sidecar when the superfile carries one: a slice, no Parquet decode.
        // Every other column still comes from the id-selected Parquet read;
        // when `_id` is the only projection, that read is skipped entirely.
        // A sidecar that can't be read (a bad block) falls back to the Parquet
        // `_id` column, which holds the same ids, so the read still answers.
        // The first failure marks the sidecar broken, so later reads skip it and
        // it warns once. `Relaxed` is enough: a read that misses the mark only
        // retries the sidecar and falls back the same way.
        let resident_sidecar = id_from_sidecar
            .then(|| self.resident_id_sidecar())
            .flatten();
        let sidecar_id = match resident_sidecar {
            Some(sidecar)
                if projection.iter().any(|&n| n == self.id_column)
                    && !self.id_sidecar_broken.load(Ordering::Relaxed) =>
            {
                match self.id_array_from_sidecar(&sidecar, local_doc_ids) {
                    Ok(ids) => Some(ids),
                    Err(error) => {
                        self.mark_id_sidecar_broken();
                        warn!(%error, "reading _id from Parquet, the id sidecar is unreadable");
                        None
                    }
                }
            }
            _ => None,
        };
        let use_sidecar_for_id = sidecar_id.is_some();

        // Columns Parquet must serve: the projection minus the sidecar-served
        // `_id`, deduped in first-seen order for the ProjectionMask. The
        // per-name gather below restores caller order (and any duplicates).
        let mut parquet_names: Vec<&str> = Vec::with_capacity(projection.len());
        for &name in projection {
            if use_sidecar_for_id && name == self.id_column {
                continue;
            }
            if !parquet_names.contains(&name) {
                parquet_names.push(name);
            }
        }

        // 4+5+6. Decode the Parquet-served columns (skipped when the sidecar
        //    covers the whole projection). local_doc_ids is a dense
        //    parquet-row index (one parquet row per doc, in id order —
        //    invariant of the superfile body), so the selection lines up
        //    directly with parquet row offsets; the caller's original order
        //    (including duplicates) is restored via the rank-back step.
        //    Either way the rows come out in the caller's order.
        let parquet_rows = if parquet_names.is_empty() {
            None
        } else {
            let mut col_indices = Vec::with_capacity(parquet_names.len());
            for &name in &parquet_names {
                let idx = self
                    .schema
                    .index_of(name)
                    .map_err(|_| ReadError::UnknownColumn(name.to_string()))?;
                col_indices.push(idx);
            }
            // Metadata-cached read: reuse the `ArrowReaderMetadata` parsed at
            // open (no per-call footer parse), so this targeted read only pays
            // the projected-column page decode. CPU-bound over in-memory
            // bytes — callers fan these across `options.reader_pool`.
            let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(bytes, arrow_meta);
            let mask = ProjectionMask::roots(builder.parquet_schema(), col_indices.iter().copied());
            let builder = builder.with_projection(mask);
            let fixed_width = col_indices
                .iter()
                .all(|&i| self.schema.field(i).data_type().primitive_width().is_some());
            let dense = fixed_width
                && (local_doc_ids.len() as u64).saturating_mul(WHOLE_COLUMN_TAKE_DIVISOR)
                    >= self.n_docs;
            if dense {
                // A take this dense touches nearly every page: read the
                // columns whole, in row order, and gather the rows in the
                // caller's order straight from the decoded batches, with no
                // copy of the whole column made first (see
                // WHOLE_COLUMN_TAKE_DIVISOR).
                let reader = builder
                    .build()
                    .map_err(|e| ReadError::Columnar(e.to_string()))?;
                let read_schema = reader.schema();
                let batches = reader
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| ReadError::Columnar(e.to_string()))?;
                Some(gather_rows(&read_schema, &batches, local_doc_ids)?)
            } else {
                // The page index (when present) lets `RowSelection` seek to
                // the relevant pages.
                let (sorted_ids, selection) = row_selection_for_ids(local_doc_ids);
                let reader = builder
                    .with_row_selection(selection)
                    .build()
                    .map_err(|e| ReadError::Columnar(e.to_string()))?;
                let read_schema = reader.schema();
                let batches = reader
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| ReadError::Columnar(e.to_string()))?;
                // `selected` has exactly sorted_ids.len() rows in sorted_ids
                // order (parquet honors the selection).
                let selected = concat_batches(&read_schema, &batches)
                    .map_err(|e| ReadError::Columnar(e.to_string()))?;
                debug_assert_eq!(
                    selected.num_rows(),
                    sorted_ids.len(),
                    "RowSelection rows ≠ requested distinct doc ids"
                );
                // Rank back into the caller's order via take. Cheap: typical k
                // is 10..1000.
                let indices = rank_back_indices(local_doc_ids, &sorted_ids);
                Some(
                    take_record_batch(&selected, &indices)
                        .map_err(|e| ReadError::Columnar(e.to_string()))?,
                )
            }
        };

        // 7. Gather columns in caller's projection order: `_id` from the
        //    sidecar, everything else from the Parquet read's rows.
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(projection.len());
        for &name in projection {
            if use_sidecar_for_id && name == self.id_column {
                columns.push(Arc::clone(sidecar_id.as_ref().expect("built above")));
                continue;
            }
            let rows = parquet_rows
                .as_ref()
                .expect("non-id projection implies a parquet read");
            let idx = rows
                .schema()
                .index_of(name)
                .map_err(|_| ReadError::UnknownColumn(name.to_string()))?;
            columns.push(Arc::clone(rows.column(idx)));
        }
        // The row count is carried explicitly rather than inferred from the
        // columns: a projection can legitimately resolve to no columns at
        // all — a caller naming only columns this file predates — and a
        // batch with neither columns nor a stated row count cannot say how
        // many rows it has. The caller null-fills from this count.
        RecordBatch::try_new_with_options(
            out_schema,
            columns,
            &RecordBatchOptions::new().with_row_count(Some(local_doc_ids.len())),
        )
        .map_err(|e| ReadError::Columnar(e.to_string()))
    }

    /// The `projection` columns of every row of row group `row_group`, in
    /// file order, decoded once from resident bytes. A scan reading whole
    /// row groups takes this rather than
    /// [`take_by_local_doc_ids`](Self::take_by_local_doc_ids) of every row,
    /// which would sort a list of the row group's rows and copy each column
    /// twice more to put them back in that list's order. Columns come out
    /// in `projection`'s order; the string and binary columns named in
    /// `views` decode straight into views ([`view_read_options`]).
    ///
    /// # Errors
    /// - [`ReadError::LazyReaderUnsupported`] — no resident bytes.
    /// - [`ReadError::UnknownColumn`] — a name is not in the file.
    /// - [`ReadError::Columnar`] — `row_group` is past the file's last, or
    ///   a Parquet/Arrow decode failure.
    pub(crate) fn read_row_group(
        &self,
        row_group: usize,
        projection: &[&str],
        views: &[&str],
    ) -> Result<RecordBatch, ReadError> {
        let bytes = self
            .bytes
            .as_ref()
            .ok_or(ReadError::LazyReaderUnsupported)?
            .clone();
        let mut arrow_meta = self
            .arrow_meta
            .as_ref()
            .ok_or(ReadError::LazyReaderUnsupported)?
            .clone();
        if !views.is_empty() {
            arrow_meta = ArrowReaderMetadata::try_new(
                Arc::clone(arrow_meta.metadata()),
                view_read_options(arrow_meta.schema(), views),
            )
            .map_err(|e| ReadError::Columnar(e.to_string()))?;
        }
        let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(bytes, arrow_meta);
        let (builder, rows) = whole_row_group(builder, row_group, projection)?;
        let reader = builder
            .build()
            .map_err(|e| ReadError::Columnar(e.to_string()))?;
        let read_schema = reader.schema();
        let batches = reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ReadError::Columnar(e.to_string()))?;
        in_projection_order(&read_schema, batches, projection, rows)
    }

    /// The stable-id sidecar sliced from resident bytes, without a read: on a
    /// [fully resident](Self::is_fully_resident) reader, and on a promoted one
    /// whose missing range ([`install_resident_parquet`]) does not overlap
    /// the sidecar. The promotion leaves out only the vector blob, which ends
    /// before the sidecar starts, so a promoted reader's sidecar bytes are the
    /// file's own; the overlap check keeps that true should the hole move.
    ///
    /// [`install_resident_parquet`]: Self::install_resident_parquet
    fn resident_id_sidecar(&self) -> Option<IdSidecar> {
        let loc = self.id_sidecar?;
        let bytes = self.bytes.as_ref()?;
        let holds_sidecar = match self.resident_hole {
            None => self.is_fully_resident(),
            Some((hole, hole_len)) => {
                loc.offset.saturating_add(loc.len) <= hole
                    || loc.offset >= hole.saturating_add(hole_len)
            }
        };
        holds_sidecar.then(|| loc.slice(bytes, self.n_docs))
    }

    /// Build the `_id` column for `local_doc_ids` (in caller order) from
    /// `sidecar`, through [`IdSidecar::ids`]. Direct indexing, so duplicates
    /// and ordering need no rank-back.
    fn id_array_from_sidecar(
        &self,
        sidecar: &IdSidecar,
        local_doc_ids: &[u32],
    ) -> Result<ArrayRef, ReadError> {
        self.id_array(sidecar.ids(local_doc_ids)?)
    }

    /// The `_id` of a few rows from the packed stable-id sidecar of a lazily
    /// opened superfile, reading its directory and only the blocks the rows
    /// fall in — a couple of kilobytes each — rather than the Parquet page
    /// index and the pages that hold them. `None` when the superfile has no
    /// packed sidecar or its sidecar was found broken; a read that fails is
    /// an error the caller answers from Parquet instead, and a block that
    /// does not decode marks the sidecar broken, as a whole-sidecar read
    /// would, so later reads skip it.
    pub(crate) async fn ids_from_sidecar_blocks(
        &self,
        local_doc_ids: &[u32],
    ) -> Result<Option<ArrayRef>, ReadError> {
        let Some(loc) = self.id_sidecar else {
            return Ok(None);
        };
        if !loc.packed || self.id_sidecar_broken.load(Ordering::Relaxed) {
            return Ok(None);
        }
        if let Some(&doc_id) = local_doc_ids
            .iter()
            .find(|&&doc| u64::from(doc) >= self.n_docs)
        {
            return Err(ReadError::DocIdOutOfRange {
                doc_id,
                n_docs: self.n_docs,
            });
        }
        let n_docs = self.n_docs as usize;
        let source = self.byte_source();
        let fetch = |what: &'static str| {
            move |source: LazyByteSourceError| ReadError::RangeFetch { what, source }
        };
        let dir_len = (ids::directory_len(n_docs) as u64).min(loc.len);
        let dir_bytes = source
            .range(loc.offset, dir_len)
            .await
            .map_err(fetch("stable-id sidecar directory fetch"))?;
        let malformed = |what: String| {
            self.mark_id_sidecar_broken();
            ReadError::MalformedKv(what)
        };
        let dir = ids::PackedDirectory::open(&dir_bytes, n_docs).map_err(malformed)?;
        let mut sorted: Vec<u32> = local_doc_ids.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let mut blocks: Vec<usize> = sorted
            .iter()
            .map(|&d| d as usize / ids::BLOCK_DOCS)
            .collect();
        blocks.dedup();
        let sidecar_len = loc.len as usize;
        let ranges: Vec<Range<usize>> = blocks
            .iter()
            .map(|&b| {
                dir.block_range(b, sidecar_len).ok_or_else(|| {
                    malformed(format!("stable-id sidecar block {b} is out of place"))
                })
            })
            .collect::<Result<_, _>>()?;
        // The merge below takes each block to start past the one before it;
        // a directory that says otherwise is corrupt.
        if let Some(w) = ranges.windows(2).find(|w| w[1].start < w[0].end) {
            return Err(malformed(format!(
                "stable-id sidecar block at {}..{} starts before the one before it ends at {}",
                w[1].start, w[1].end, w[0].end
            )));
        }
        // Exact reads, since a block is a few KB and a cache widening each to
        // its read window downloaded hundreds of KB per hit; blocks closer
        // than `ID_BLOCK_GAP_BYTES` share one read, so a superfile with many
        // hits costs a few reads rather than one per block.
        let mut spans: Vec<Range<usize>> = Vec::new();
        for r in &ranges {
            match spans.last_mut() {
                Some(span) if r.start <= span.end + ID_BLOCK_GAP_BYTES => {
                    span.end = span.end.max(r.end)
                }
                _ => spans.push(r.clone()),
            }
        }
        let span_bodies = try_join_all(
            spans
                .iter()
                .map(|r| source.range_exact(loc.offset + r.start as u64, r.len() as u64)),
        )
        .await
        .map_err(fetch("stable-id sidecar block fetch"))?;
        let bodies: Vec<Bytes> = ranges
            .iter()
            .map(|r| {
                let at = spans.partition_point(|span| span.end <= r.start);
                let span = &spans[at];
                span_bodies[at].slice(r.start - span.start..r.end - span.start)
            })
            .collect();
        let mut by_doc: Vec<i128> = Vec::with_capacity(sorted.len());
        let mut at = 0usize;
        for (&b, body) in blocks.iter().zip(&bodies) {
            let end = sorted[at..]
                .iter()
                .position(|&d| d as usize / ids::BLOCK_DOCS != b)
                .map_or(sorted.len(), |n| at + n);
            ids::ids_in_block(
                body,
                b,
                n_docs,
                dir.n_blocks(),
                &sorted[at..end],
                &mut by_doc,
            )
            .ok_or_else(|| malformed(format!("stable-id sidecar block {b} is malformed")))?;
            at = end;
        }
        let ids: Vec<i128> = local_doc_ids
            .iter()
            .map(|d| by_doc[sorted.binary_search(d).expect("every doc was sorted in")])
            .collect();
        self.id_array(ids).map(Some)
    }

    /// Read the packed sidecar's directory into the byte source's cache, so a
    /// later [`Self::ids_from_sidecar_blocks`] costs one round trip for its
    /// blocks. Nothing to do without a packed sidecar; a failed read is left
    /// to that later read to report.
    pub(crate) async fn prewarm_id_directory(&self) {
        let Some(loc) = self.id_sidecar else {
            return;
        };
        // A resident reader resolves ids from its own bytes; the blocks are
        // for a lazy one, and building a byte source here per superfile and
        // query is measurable on a warm table.
        if !loc.packed || self.bytes.is_some() {
            return;
        }
        let dir_len = (ids::directory_len(self.n_docs as usize) as u64).min(loc.len);
        let _ = self.byte_source().range(loc.offset, dir_len).await;
    }

    /// The `_id` column over `ids`, typed as the table declares it.
    fn id_array(&self, ids: Vec<i128>) -> Result<ArrayRef, ReadError> {
        let id_idx = self
            .schema
            .index_of(&self.id_column)
            .map_err(|_| ReadError::UnknownColumn(self.id_column.clone()))?;
        let (precision, scale) = match self.schema.field(id_idx).data_type() {
            DataType::Decimal128(p, s) => (*p, *s),
            other => {
                return Err(ReadError::Columnar(format!(
                    "id column {} is {other:?}, expected Decimal128",
                    self.id_column
                )));
            }
        };
        let array = Decimal128Array::from_iter_values(ids)
            .with_precision_and_scale(precision, scale)
            .map_err(|e| ReadError::Columnar(e.to_string()))?;
        Ok(Arc::new(array))
    }

    /// Sequential scan of the `_id` column resolving many targets
    /// in **one** pass instead of one pass per target. Returns
    /// `(target, local doc_id)` pairs for the targets present in
    /// this superfile, ascending by target; absent targets are
    /// simply missing from the result. A `doc_id` is the row offset
    /// within this superfile, as used by tombstones / FTS / vector
    /// indices.
    ///
    /// `sorted_targets` must be sorted ascending and deduped — the
    /// scan probes it by binary search, merging the (small, sorted)
    /// target side against the (large) id column, so the cost is
    /// `O(rows · log targets)` for the whole batch rather than
    /// `O(targets · rows)`. The scan stops early once every target
    /// has been located.
    ///
    /// `_id` is stored as `Decimal128` with the supertable's
    /// fixed precision/scale; we decode each value as `i128`.
    ///
    /// Used by the WAL tombstone phase's target-resolution sweep:
    /// given the mutation's target ids, scan each candidate
    /// superfile once to find where those rows live so the phase
    /// can mark their bits. Rebuilds a `ParquetRecordBatchReader`
    /// each call (cold recovery path), rather than reusing the
    /// cached `arrow_meta` like `take_by_local_doc_ids` does.
    pub(crate) fn id_lookup_many(
        &self,
        sorted_targets: &[i128],
    ) -> Result<Vec<(i128, u32)>, ReadError> {
        debug_assert!(
            sorted_targets.windows(2).all(|w| w[0] < w[1]),
            "id_lookup_many requires ascending, deduped targets"
        );
        if sorted_targets.is_empty() {
            return Ok(Vec::new());
        }
        let bytes = self.bytes.clone().ok_or_else(|| {
            ReadError::Io(io::Error::other(
                "id_lookup_many requires an eager-opened superfile; this reader was \
                 opened via the lazy path and does not hold the full superfile bytes",
            ))
        })?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .map_err(|e| ReadError::Footer(footer::FooterError::Parquet(e)))?;
        // _id is always at index 0
        let descriptor = builder.parquet_schema().clone();
        let reader = builder
            .with_projection(ProjectionMask::leaves(&descriptor, vec![0]))
            .build()
            .map_err(|e| ReadError::Footer(footer::FooterError::Parquet(e)))?;

        // Parallel to `sorted_targets`: the doc_id each target
        // resolved to, or `None` while it is still unfound.
        let mut hits: Vec<Option<u32>> = vec![None; sorted_targets.len()];
        let mut n_found: usize = 0;

        let mut row_offset: u32 = 0;
        'scan: for batch_res in reader {
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
                // Probe the small sorted target side rather than
                // re-scanning the column per target. Nothing is
                // assumed about the column's own ordering, so this
                // stays correct for merged / rewritten superfiles.
                if let Ok(t) = sorted_targets.binary_search(&id_arr.value(i)) {
                    // First occurrence wins, matching the
                    // single-target scan's semantics.
                    if hits[t].is_none() {
                        hits[t] = Some(row_offset + i as u32);
                        n_found += 1;
                        if n_found == sorted_targets.len() {
                            break 'scan;
                        }
                    }
                }
            }
            row_offset = row_offset.checked_add(id_arr.len() as u32).ok_or_else(|| {
                ReadError::MalformedKv("row_offset overflow scanning id column".to_string())
            })?;
        }

        Ok(sorted_targets
            .iter()
            .zip(hits)
            .filter_map(|(&target, hit)| hit.map(|doc_id| (target, doc_id)))
            .collect())
    }

    /// Single-column BM25 search across the unified FTS reader.
    ///
    /// `query` is tokenized by the same tokenizer the column was built
    /// with (its per-column analyzer). Returns `(local_doc_id, score)`
    /// hits ordered by descending score — this is the hit kernel, not a
    /// row-returning search; row materialization is `take_by_local_doc_ids`.
    ///
    /// ## Clause sigils (`+term`, `-term`)
    ///
    /// A `+`-prefixed term is a **must**: every result contains it. A
    /// `-`-prefixed term is a **must-not**: any doc containing it is
    /// excluded, regardless of score. Bare terms take their polarity
    /// from `mode`, the default operator: `And` requires them like
    /// musts; `Or` makes them scoring-only **shoulds** when a must
    /// exists (`"+climate policy"` matches docs with `climate`, and
    /// docs also mentioning `policy` rank higher) and a plain union
    /// when none does. A query with only negated terms is rejected
    /// (`FtsError::NegationOnly`).
    pub async fn bm25_hits_async(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<(RowId, f32)>, ReadError> {
        // Tokenize with the target column's configured tokenizer so query
        // terms match how the column was indexed (its analysis chain).
        // A column this superfile has no full-text index for fails here,
        // where the reason is still nameable, rather than after a pass with
        // some other column's analyzer.
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        let tok: Arc<dyn Tokenizer> = fts.column_tokenizer(column)?;

        // Split the query into clause lists and resolve the bare
        // tokens' polarity from the default operator. The parsed
        // tokens borrow `query`, so nothing is copied here.
        let clauses = tok.parse(query).into_clauses(mode);
        let musts: Vec<&str> = clauses.musts.iter().map(|t| &**t).collect();
        let shoulds: Vec<&str> = clauses.shoulds.iter().map(|t| &**t).collect();
        let negatives: Vec<&str> = clauses.negatives.iter().map(|t| &**t).collect();
        // `Phrase::map` keeps each term's offset, which is what a
        // phrase on a stopworded column needs: its terms were not
        // adjacent in the query and must not be required adjacent here.
        let own = |phrases: Vec<Phrase<Cow<'_, str>>>| -> Vec<Phrase<String>> {
            phrases.iter().map(|p| p.map(|t| t.to_string())).collect()
        };
        let must_phrases = own(clauses.must_phrases);
        let should_phrases = own(clauses.should_phrases);
        let negative_phrases = own(clauses.negative_phrases);
        self.bm25_search_clauses(
            column,
            ClauseLists {
                musts: &musts,
                shoulds: &shoulds,
                negatives: &negatives,
                must_phrases: &must_phrases,
                should_phrases: &should_phrases,
                negative_phrases: &negative_phrases,
                idf: fts_reader::QueryIdf::ThisFile,
                prefetched: None,
                live_floor: None,
                allow: None,
                exact_count: false,
            },
            k,
            f32::NEG_INFINITY,
        )
        .await
    }

    /// Pre-tokenized variant of [`Self::bm25_hits_async`] — the caller
    /// supplies the already-tokenized term slice and we skip the
    /// tokenizer pass.
    ///
    /// Used by the supertable layer's fan-out: the cross-superfile
    /// search tokenizes the query once at the orchestrator (to
    /// compute the bloom-skip mask) and then passes the same
    /// `terms` slice to every per-superfile search, avoiding
    /// `(N+1)·T` redundant tokenizations across N superfiles and
    /// a T-token query.
    ///
    /// Terms must already be tokenized to the column's dictionary key form —
    /// the column's own tokenizer, e.g. `StandardTokenizer.tokenize(query)`
    /// for a plain `standard` column.
    pub async fn bm25_search_pretokenized(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<(RowId, f32)>, ReadError> {
        self.bm25_search_pretokenized_with_floor(column, terms, k, mode, f32::NEG_INFINITY)
            .await
    }

    /// [`Self::bm25_search_pretokenized`] with a score floor: docs
    /// scoring strictly below `floor` are pruned inside the kernels
    /// (BMW / MaxScore / AND block skips all start from the floor);
    /// docs scoring exactly `floor` are still returned. Used by the
    /// supertable fan-out to share the global kth-best score across
    /// segments. See [`FtsReader::search_with_floor`].
    pub async fn bm25_search_pretokenized_with_floor(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        mode: BoolMode,
        floor: f32,
    ) -> Result<Vec<(RowId, f32)>, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.search_with_floor(column, terms, k, mode, floor).await?)
    }

    /// Unranked token match: the `local_doc_id`s matching the
    /// `tokens` under `mode` (`And` = every token, `Or` = any token),
    /// in ascending order. No BM25 scoring. `tokens` are already
    /// tokenized terms. Delegates to [`FtsReader::token_match`].
    pub async fn token_match(
        &self,
        column: &str,
        tokens: &[&str],
        mode: BoolMode,
    ) -> Result<(Vec<RowId>, MatchWork), ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.token_match(column, tokens, mode).await?)
    }

    /// [`Self::token_match`] with terms a table-level term index already
    /// resolved served from `prefetched`, so the dictionary is read only
    /// for tokens the memo lacks. Delegates to
    /// [`FtsReader::token_match_prefetched`].
    pub(crate) async fn token_match_prefetched(
        &self,
        column: &str,
        tokens: &[&str],
        mode: BoolMode,
        prefetched: Option<&FetchedTermMemo>,
    ) -> Result<(Vec<RowId>, MatchWork), ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts
            .token_match_prefetched(column, tokens, mode, prefetched)
            .await?)
    }

    /// The cursors of an `Or` match over `tokens`, built once for this
    /// superfile's doc-id slices. Delegates to
    /// [`FtsReader::match_cursor_set`].
    pub(crate) async fn match_cursor_set(
        &self,
        column: &str,
        tokens: &[&str],
        prefetched: Option<&FetchedTermMemo>,
    ) -> Result<(OrCursorSet, MatchWork), ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.match_cursor_set(column, tokens, prefetched).await?)
    }

    /// One doc-id slice of an `Or` match against cursors
    /// [`Self::match_cursor_set`] built. Delegates to
    /// [`FtsReader::token_match_range_prebuilt`].
    pub(crate) fn token_match_range_prebuilt(
        &self,
        set: &OrCursorSet,
        start: u32,
        end: u32,
    ) -> Result<(Vec<RowId>, MatchWork), ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.token_match_range_prebuilt(set, start, end)?)
    }

    /// Widen the tokens of one `LIKE` leaf to the indexed terms of
    /// `column` each covers, in one dictionary pass (a slot is `None` when
    /// more than `max_terms` qualify, or when it needs the whole column
    /// walked and `allow_full_walk` is off); `fold` compares terms under
    /// the `ILIKE` folding. Delegates to [`FtsReader::expand_terms`].
    pub(crate) async fn expand_terms(
        &self,
        column: &str,
        patterns: &[TermPattern<'_>],
        fold: bool,
        max_terms: usize,
        allow_full_walk: bool,
        pool: Option<&ThreadPool>,
    ) -> Result<(Vec<Option<Vec<String>>>, MatchWork), ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts
            .expand_terms(column, patterns, fold, max_terms, allow_full_walk, pool)
            .await?)
    }

    /// For each of `needles`, the rows of `column` an `ILIKE '%needle%'`
    /// matches, decided from the dictionary and postings in one walk but
    /// for the rows it marks doubtful; every needle meets the exact rule.
    /// What it holds meanwhile is charged to `budget`. Delegates to
    /// [`FtsReader::contains_rows`].
    pub(crate) async fn contains_rows(
        &self,
        column: &str,
        needles: &[&str],
        pool: Option<&ThreadPool>,
        budget: Option<&Arc<ConnectionMemoryBudget>>,
    ) -> Result<(Vec<ContainsRows>, MatchWork), ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.contains_rows(column, needles, pool, budget).await?)
    }

    /// Unranked token-match **count**: the number of `local_doc_id`s
    /// [`token_match`](Self::token_match) would return under `mode`,
    /// without materializing the id `Vec`. Delegates to
    /// [`FtsReader::token_match_count`].
    pub async fn token_match_count(
        &self,
        column: &str,
        tokens: &[&str],
        mode: BoolMode,
    ) -> Result<(u64, MatchWork), ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.token_match_count(column, tokens, mode).await?)
    }

    /// [`Self::token_match_count`] served from `prefetched`; see
    /// [`Self::token_match_prefetched`].
    pub(crate) async fn token_match_count_prefetched(
        &self,
        column: &str,
        tokens: &[&str],
        mode: BoolMode,
        prefetched: Option<&FetchedTermMemo>,
    ) -> Result<(u64, MatchWork), ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts
            .token_match_count_prefetched(column, tokens, mode, prefetched)
            .await?)
    }

    /// Phrase-aware unranked match: `local_doc_id`s whose `column`
    /// matches the `terms` and exact `phrases` under `mode`,
    /// ascending. Used by the table layer whenever the match set
    /// contains a phrase; plain-token queries keep
    /// [`token_match`](Self::token_match).
    pub async fn atoms_match_ids(
        &self,
        column: &str,
        terms: &[&str],
        phrases: &[Phrase<String>],
        mode: BoolMode,
    ) -> Result<(Vec<RowId>, MatchWork), ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.atoms_match_ids(column, terms, phrases, mode).await?)
    }

    /// Phrase-aware unranked match **count** — the phrase sibling of
    /// [`token_match_count`](Self::token_match_count). `neg_terms` /
    /// `neg_phrases` are counted as a skip-based exclusion gate (a doc
    /// matching any negated clause is not counted); pass empty slices for
    /// an unnegated count.
    pub async fn atoms_match_count(
        &self,
        column: &str,
        terms: &[&str],
        phrases: &[Phrase<String>],
        mode: BoolMode,
        neg_terms: &[&str],
        neg_phrases: &[Phrase<String>],
    ) -> Result<(u64, MatchWork), ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts
            .atoms_match_count(column, terms, phrases, mode, neg_terms, neg_phrases)
            .await?)
    }

    /// Document frequency of `token` in `column` (0 if absent) — a cheap
    /// header-only read used to estimate a predicate's match count
    /// before running `token_match`. Delegates to
    /// [`FtsReader::term_df`].
    pub async fn term_df(&self, column: &str, token: &str) -> Result<(u64, MatchWork), ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.term_df(column, token).await?)
    }

    /// A prefetched-term memo for terms whose dictionary values are already
    /// known, fetching only their postings. Delegates to
    /// [`FtsReader::memo_from_dict_values`].
    pub(crate) async fn term_memo_from_dict_values(
        &self,
        terms: &[(&str, u64, DictEntry)],
    ) -> Result<FetchedTermMemo, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.memo_from_dict_values(terms).await?)
    }

    /// What a table-level term index records about each of `tokens` in
    /// `column`, in input order (`None` for an absent token). Delegates to
    /// [`FtsReader::term_index_facts`].
    #[cfg(test)]
    pub(crate) async fn term_index_facts(
        &self,
        column: &str,
        tokens: &[&str],
    ) -> Result<Vec<Option<TermIndexFact>>, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.term_index_facts(column, tokens).await?)
    }

    /// Document frequency of each of `tokens` in `column`, in input order (0
    /// for any absent token). Batched sibling of [`Self::term_df`]: resolves
    /// the whole set with one dictionary parse and one coalesced header fetch.
    /// Delegates to [`FtsReader::term_dfs`].
    pub async fn term_dfs(
        &self,
        column: &str,
        tokens: &[&str],
    ) -> Result<(Vec<u64>, MatchWork), ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.term_dfs(column, tokens).await?)
    }

    /// Open-wave fetch of `terms`' dictionary slots + postings ranges
    /// with per-term df, for the fused global-stats gather. Delegates to
    /// [`FtsReader::fetch_scored_terms`].
    pub(crate) async fn fetch_scored_terms(
        &self,
        column: &str,
        terms: &[&str],
    ) -> Result<fts_reader::FetchedTermMemo, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.fetch_scored_terms(column, terms).await?)
    }

    /// Two-pass exact match of a **raw string** `value` against
    /// `column`'s stored values. The input is a raw string, **not**
    /// tokens — tokenization is used only to prune candidates, never as
    /// the match:
    ///
    ///   1. **Prune (index).** Tokenize `value` and `token_match` its
    ///      tokens under `And` to get candidate rows that contain all of
    ///      them. (An empty token set — e.g. punctuation-only `value` —
    ///      can't prune, so every row is a candidate.)
    ///   2. **Verify (text).** Decode `column` for those candidates and
    ///      keep the rows whose stored value **equals `value` exactly**.
    ///
    /// Returns the verified `local_doc_id`s in ascending order. Works
    /// for single-word and multi-word strings alike — the token count
    /// only affects pruning, never the raw-string comparison.
    /// Returns the verified ids plus the prune pass's posting-walk work.
    pub async fn exact_match(
        &self,
        column: &str,
        value: &str,
    ) -> Result<(Vec<RowId>, MatchWork), ReadError> {
        // Pass 1 — candidate rows via the index: the term-AND of the
        // string's tokens (a superset of the exact matches). Tokenize
        // with the column's configured tokenizer to match the index; a
        // column with no full-text index here has no dictionary to prune
        // through, so it fails rather than pruning with a foreign analyzer.
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        let tok: Arc<dyn Tokenizer> = fts.column_tokenizer(column)?;
        let tokens: Vec<String> = tok.tokenize(value).collect();
        let (candidates, work): (Vec<RowId>, MatchWork) = if tokens.is_empty() {
            // No tokens to prune with: every row is a candidate.
            (
                (0..self.n_docs() as u32).map(RowId::new).collect(),
                MatchWork::default(),
            )
        } else {
            let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
            self.token_match(column, &refs, BoolMode::And).await?
        };
        if candidates.is_empty() {
            return Ok((Vec::new(), work));
        }

        // Pass 2 — verify raw-string equality on the decoded text.
        let candidate_rows: Vec<u32> = candidates.iter().copied().map(RowId::get).collect();
        let batch = self.take_by_local_doc_ids(&candidate_rows, &[column])?;
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .ok_or_else(|| {
                ReadError::Io(io::Error::other(format!(
                    "exact_match: column '{column}' is not LargeUtf8"
                )))
            })?;
        // `take_by_local_doc_ids` returns rows in `candidates` order, so
        // row `i` is `candidates[i]`; candidates are ascending, so the
        // kept ids stay ascending.
        let mut out = Vec::new();
        for (i, &doc) in candidates.iter().enumerate() {
            if !col.is_null(i) && col.value(i) == value {
                out.push(doc);
            }
        }
        Ok((out, work))
    }

    /// Pre-tokenized BM25 search over explicit clause lists — the
    /// clause sibling of [`Self::bm25_search_pretokenized`]. The
    /// supertable fan-out parses the `+`/`-` sigils and resolves the
    /// default operator once at the orchestrator, then hands every
    /// superfile the split lists through here. `musts` all have to
    /// match; `shoulds` only raise scores; `negatives` exclude. Docs
    /// scoring strictly below `floor` are pruned
    /// (`f32::NEG_INFINITY` disables the floor).
    pub(crate) async fn bm25_search_clauses(
        &self,
        column: &str,
        lists: ClauseLists<'_>,
        k: usize,
        floor: f32,
    ) -> Result<Vec<(RowId, f32)>, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.search_excluding(column, lists, k, floor).await?)
    }

    /// I/O half of [`Self::bm25_search_clauses`] — resolves the column
    /// and fetches every cursor [`Self::run_prepared`] needs to score.
    pub(crate) async fn prepare_clauses(
        &self,
        column: &str,
        lists: ClauseLists<'_>,
        k: usize,
        floor: f32,
        scoring: &ScoringView<'_>,
    ) -> Result<PreparedClauses, ReadError> {
        let fts = self.fts_scored(scoring)?;
        Ok(fts.prepare_clauses(column, lists, k, floor).await?)
    }

    /// CPU half paired with [`Self::prepare_clauses`] — scores the
    /// cursors it fetched.
    ///
    /// `bm25` must be the same override the paired `prepare_clauses`
    /// was given: the cursors carry the bounds they were built with, and
    /// scoring divides by a norm table derived from that same override.
    /// `with_bm25_override` is deterministic, so two separately-derived
    /// views of one pair agree bit for bit.
    #[cfg(test)]
    pub(crate) fn run_prepared(
        &self,
        prep: PreparedClauses,
        scoring: &ScoringView<'_>,
    ) -> Result<Vec<(RowId, f32)>, ReadError> {
        let fts = self.fts_scored(scoring)?;
        Ok(fts.run_prepared(prep)?)
    }

    /// [`Self::run_prepared`] in blob ids, for a caller that translates the
    /// hits itself on its own runtime ([`Self::hits_to_rows_async`]).
    pub(crate) fn run_prepared_in_blob_ids(
        &self,
        prep: PreparedClauses,
        scoring: &ScoringView<'_>,
    ) -> Result<Vec<(FtsDocId, f32)>, ReadError> {
        let fts = self.fts_scored(scoring)?;
        Ok(fts.run_prepared_in_blob_ids(prep)?)
    }

    /// [`Self::run_prepared_with_count`] in blob ids.
    pub(crate) fn run_prepared_with_count_in_blob_ids(
        &self,
        prep: PreparedClauses,
        scoring: &ScoringView<'_>,
    ) -> Result<(Vec<(FtsDocId, f32)>, u64), ReadError> {
        let fts = self.fts_scored(scoring)?;
        Ok(fts.run_prepared_with_count_in_blob_ids(prep)?)
    }

    /// Blob-id hits of this superfile's FTS index as rows, reading only the
    /// doc-id map pages they fall in when the map is not in memory.
    pub(crate) async fn hits_to_rows_async(
        &self,
        hits: Vec<(FtsDocId, f32)>,
    ) -> Result<Vec<(RowId, f32)>, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.hits_to_rows_async(hits).await?)
    }

    /// The FTS reader a scored query should read through: this
    /// superfile's own, or a view of it that scores with `scoring`
    /// instead of what each column baked in.
    ///
    /// Borrowed when there is no override, which is the default path
    /// and costs nothing. An override clones the reader — an `Arc` bump
    /// for the blob plus one 1 KiB decode table per column that
    /// actually moves — and records the factor that keeps each column's
    /// stored bounds upper bounds under what is being scored.
    fn fts_scored(&self, scoring: &ScoringView<'_>) -> Result<ScoredFts<'_>, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        if let Some((cached, view)) = self.scored_fts.read().expect("scored fts memo").as_ref()
            && cached.matches(scoring)
        {
            return Ok(ScoredFts::Derived(Arc::clone(view)));
        }
        let Some(view) = fts.with_scoring(scoring) else {
            return Ok(ScoredFts::Baked(fts));
        };
        let view = Arc::new(view);
        // Last writer wins. Two queries racing here derive the same view
        // from the same inputs — `with_scoring` is deterministic — so which
        // one lands is immaterial.
        *self.scored_fts.write().expect("scored fts memo") =
            Some((ScoringKey::of(scoring), Arc::clone(&view)));
        Ok(ScoredFts::Derived(view))
    }

    /// Multi-term OR BM25 search restricted to a doc_id sub-range.
    ///
    /// Mirrors [`Self::bm25_search_pretokenized`] in `BoolMode::Or`
    /// shape but only scores docs in `[doc_id_start, doc_id_end)`.
    /// Used by the supertable layer's intra-superfile parallel
    /// fan-out: the supertable splits each superfile into N
    /// equal-width sub-ranges, runs one call per sub-range in
    /// parallel on the reader pool, then merges the per-sub-range
    /// top-K heaps.
    ///
    /// Single-term inputs (`terms.len() == 1`) are not optimized
    /// here — they already finish in microseconds via
    /// [`Self::bm25_search_pretokenized`]; the supertable layer
    /// should keep them on the un-ranged path.
    pub async fn bm25_search_or_range_pretokenized(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
    ) -> Result<Vec<(RowId, f32)>, ReadError> {
        self.bm25_search_or_range_pretokenized_with_floor(
            column,
            terms,
            k,
            doc_id_start,
            doc_id_end,
            f32::NEG_INFINITY,
            fts_reader::QueryIdf::ThisFile,
        )
        .await
    }

    /// [`Self::bm25_search_or_range_pretokenized`] with a score floor —
    /// same contract as [`Self::bm25_search_pretokenized_with_floor`].
    pub(crate) async fn bm25_search_or_range_pretokenized_with_floor(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
        floor: f32,
        idf: fts_reader::QueryIdf<'_>,
    ) -> Result<Vec<(RowId, f32)>, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts
            .search_or_range_pretokenized_with_floor(
                column,
                terms,
                k,
                doc_id_start,
                doc_id_end,
                floor,
                idf,
            )
            .await?)
    }

    /// Build the OR cursor set for `terms` once, for reuse across this
    /// superfile's doc-id sub-ranges via
    /// [`Self::bm25_search_or_range_prebuilt`].
    pub(crate) async fn bm25_or_cursor_set(
        &self,
        column: &str,
        terms: &[&str],
        idf: fts_reader::QueryIdf<'_>,
        prefetched: Option<&fts_reader::FetchedTermMemo>,
        scoring: &ScoringView<'_>,
    ) -> Result<OrCursorSet, ReadError> {
        // Built through the same view the set is scored through, so its
        // stored bounds carry that view's scale.
        let fts = self.fts_scored(scoring)?;
        Ok(fts
            .build_or_cursor_set(column, terms, idf, prefetched)
            .await?)
    }

    /// Ranged multi-term OR against prebuilt cursors — see
    /// [`Self::bm25_search_or_range_pretokenized_with_floor`] for the
    /// range and floor contract. Hits are in blob ids, for
    /// [`Self::hits_to_rows_async`].
    pub(crate) fn bm25_search_or_range_prebuilt(
        &self,
        set: &OrCursorSet,
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
        floor: f32,
        scoring: &ScoringView<'_>,
    ) -> Result<Vec<(FtsDocId, f32)>, ReadError> {
        let fts = self.fts_scored(scoring)?;
        Ok(fts.search_or_range_prebuilt_in_blob_ids(set, k, doc_id_start, doc_id_end, floor)?)
    }

    /// Multi-column BM25 search with per-column weights ("most
    /// fields" semantics: per-column scores summed by weight).
    pub async fn bm25_search_multi(
        &self,
        columns: &[(&str, f32)],
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<(RowId, f32)>, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.search_multi(columns, query, k, mode).await?)
    }

    /// Single-column vector kNN against a named vector index.
    ///
    /// `options` controls the recall-vs-latency tradeoff;
    /// [`VectorSearchOptions::new()`] (or `..Default::default()`)
    /// picks defaults that recover ≥0.9 recall@10 on typical IVF
    /// setups.
    ///
    /// Async — consistent with [`Self::bm25_search`]: the reader's
    /// search surfaces are `async` so the supertable fan-out can drive
    /// every superfile concurrently on the shared query runtime. The
    /// public `Supertable` API remains strictly sync; it wraps these
    /// kernels in `block_on_query`. Per-range byte access
    /// routes through
    /// [`crate::superfile::lazy_source::Source::range_async`] /
    /// `get_ranges_parallel_async`, which resolve zero-copy on the sync
    /// fast path for `Source::InMemory` and warm-cache `Source::Lazy`
    /// (`BytesLazyByteSource`, mmap-backed) via `try_get_range_sync`;
    /// only a cold `Source::Lazy` miss actually `await`s an
    /// object-store GET. The CPU steps (centroid + 1-bit code scoring,
    /// rerank) parallelize on the global rayon pool.
    pub async fn vector_hits_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
        Ok(self
            .vector_hits_filtered_async(column, query, k, options, None, None, None, None)
            .await?
            .0)
    }

    /// As [`Self::vector_hits_async`], but restricts the kNN ranking to
    /// the `local_doc_id`s in `allow` (a per-superfile predicate
    /// allow-set). The allow-set is applied *inside* the coarse 1-bit
    /// shortlist, so the returned top-k is the true k-nearest among
    /// matching rows — pushdown, not post-filter, with no underflow.
    /// `allow == None` is identical to [`Self::vector_hits_async`].
    /// Returns the hits plus the probe's work tallies for the per-query
    /// work stats.
    pub async fn vector_hits_filtered_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        allow: Option<Arc<RoaringBitmap>>,
        deny: Option<Arc<RoaringBitmap>>,
        pool: Option<Arc<ThreadPool>>,
        budget: Option<Arc<ConnectionMemoryBudget>>,
    ) -> Result<(Vec<(u32, f32)>, ProbeTally), ReadError> {
        let filtered = allow.is_some();
        let (nprobe, rerank_mult) = options.resolve(filtered);
        let v = self
            .vec()
            .ok_or_else(|| ReadError::MissingKv(kv::VEC_OFFSET))?;
        let rerank_mult = v.public_rerank_mult(column, rerank_mult);
        Ok(v.search_async(
            column,
            query,
            k,
            nprobe,
            rerank_mult,
            allow,
            deny,
            pool,
            budget,
        )
        .await?)
    }

    /// As [`Self::vector_search`], but probes an **externally chosen**
    /// set of IVF cluster ids — selected globally across superfiles from
    /// the manifest's per-cluster centroids — instead of this superfile's
    /// own `nprobe` centroid scoring. `rerank_mult` is still derived from
    /// `options`; `options.nprobe` is unused on this path.
    pub async fn vector_search_clusters(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        clusters: &[u32],
        options: VectorSearchOptions,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
        Ok(self
            .vector_search_clusters_filtered(
                column, query, k, clusters, options, None, None, None, None,
            )
            .await?
            .0)
    }

    /// As [`Self::vector_search_clusters`], but restricts the kNN
    /// ranking to the `local_doc_id`s in `allow` (a per-superfile
    /// predicate allow-set), applied inside the coarse shortlist.
    /// `allow == None` is identical to [`Self::vector_search_clusters`].
    /// Returns the hits plus the probe's work tallies for the per-query
    /// work stats.
    pub async fn vector_search_clusters_filtered(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        clusters: &[u32],
        options: VectorSearchOptions,
        allow: Option<Arc<RoaringBitmap>>,
        deny: Option<Arc<RoaringBitmap>>,
        pool: Option<Arc<ThreadPool>>,
        budget: Option<Arc<ConnectionMemoryBudget>>,
    ) -> Result<(Vec<(u32, f32)>, ProbeTally), ReadError> {
        let filtered = allow.is_some();
        let (_, rerank_mult) = options.resolve(filtered);
        let v = self
            .vec()
            .ok_or_else(|| ReadError::MissingKv(kv::VEC_OFFSET))?;
        let rerank_mult = v.public_rerank_mult(column, rerank_mult);
        Ok(v.search_clusters_async(
            column,
            query,
            k,
            clusters,
            rerank_mult,
            allow,
            deny,
            pool,
            budget,
        )
        .await?)
    }

    /// Deferred-rerank sibling of [`Self::vector_search_clusters_filtered`]
    /// for the supertable's global-shortlist width sweeps: warm cells
    /// return 1-bit estimate survivors (scanned at the undivided
    /// `k × rerank_mult`), cold cells rerank immediately under
    /// `cold_rerank_mult`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn vector_scan_clusters_filtered(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        clusters: &[u32],
        options: VectorSearchOptions,
        cold_rerank_mult: usize,
        allow: Option<Arc<RoaringBitmap>>,
        deny: Option<Arc<RoaringBitmap>>,
        pool: Option<Arc<ThreadPool>>,
        budget: Option<Arc<ConnectionMemoryBudget>>,
    ) -> Result<ScanOutcome, ReadError> {
        let filtered = allow.is_some();
        let (_, rerank_mult) = options.resolve(filtered);
        let v = self
            .vec()
            .ok_or_else(|| ReadError::MissingKv(kv::VEC_OFFSET))?;
        let rerank_mult = v.public_rerank_mult(column, rerank_mult);
        Ok(v.search_clusters_scan_async(
            column,
            query,
            k,
            clusters,
            None,
            rerank_mult,
            cold_rerank_mult,
            false,
            allow,
            deny,
            pool,
            budget,
        )
        .await?)
    }

    /// Rerank this superfile's share of the globally selected shortlist —
    /// phase C of the deferred-rerank width sweep. Returns the hits plus
    /// the rerank kernel's bracketed on-CPU ns.
    pub(crate) async fn vector_rerank_selected(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        selected: Vec<ScanCandidate>,
        pool: Option<Arc<ThreadPool>>,
    ) -> Result<(Vec<(u32, f32)>, u64), ReadError> {
        let v = self
            .vec()
            .ok_or_else(|| ReadError::MissingKv(kv::VEC_OFFSET))?;
        Ok(v.rerank_selected_async(column, query, k, selected, pool)
            .await?)
    }
}

/// Tuning knobs for vector search (`Supertable::vector_search`).
///
/// Each field is an optional override. `None` means “use the engine default”,
/// which depends on whether the query is filtered:
///
/// - unfiltered: `nprobe=6`, `rerank_mult=256` (the supertable user-table
///   path widens its own coarse default independently — see
///   `USER_COARSE_CELLS` in the supertable query layer)
/// - filtered (`filter: Some(...)` or an internal allow-set): `nprobe=8`, `rerank_mult=256`
///
/// Set a field with [`Self::with_nprobe`] / [`Self::with_rerank_mult`].
#[derive(Debug, Clone, Copy, Default)]
pub struct VectorSearchOptions {
    /// IVF probe override. `None` → engine default for the query path.
    pub nprobe: Option<usize>,
    rerank_mult: Option<usize>,
    /// `hnsw_ivf` serve-time beam override. `None` → the stamped k→ef curve
    /// (or the `vector.hnsw_ef_search` config), exactly as today; `Some(ef)`
    /// with `ef > 0` walks this query at that fixed beam. A test-and-bench
    /// instrument (recall/latency sweeps of an already-built graph); ignored
    /// under any non-graph serving path.
    ef: Option<usize>,
}

impl VectorSearchOptions {
    /// Unfiltered default `nprobe` (used when [`Self::nprobe`] is `None`).
    pub const DEFAULT_NPROBE: usize = 6;

    /// Default `rerank_mult` for both paths (used when unset).
    pub const RERANK_MULT: usize = 256;

    /// Filtered default `nprobe` (internal; applied when the query carries a filter).
    pub(crate) const FILTERED_DEFAULT_NPROBE: usize = 8;

    /// Default options — engine-default `nprobe` and rerank multiplier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of IVF partitions to probe. Higher improves recall at
    /// the cost of more work.
    pub fn with_nprobe(mut self, n: usize) -> Self {
        self.nprobe = Some(n);
        self
    }

    /// Set the over-fetch multiplier for the exact-rerank stage. Higher
    /// improves recall at the cost of more work.
    pub fn with_rerank_mult(mut self, n: usize) -> Self {
        self.rerank_mult = Some(n.max(1));
        self
    }

    /// The configured rerank multiplier, if one was set.
    pub fn rerank_mult(&self) -> Option<usize> {
        self.rerank_mult
    }

    /// Set the `hnsw_ivf` serve-time beam (`ef`) for this query. Higher improves
    /// recall at the cost of more work; overrides the stamped k→ef curve for an
    /// already-built graph without a rebuild.
    pub fn with_ef(mut self, ef: usize) -> Self {
        self.ef = Some(ef);
        self
    }

    /// The configured serve-time `ef` beam, if one was set.
    pub(crate) fn ef(&self) -> Option<usize> {
        self.ef
    }

    /// Resolve `(nprobe, rerank_mult)` for this query path.
    pub(crate) fn resolve(&self, filtered: bool) -> (usize, usize) {
        let nprobe = self.nprobe.unwrap_or(if filtered {
            Self::FILTERED_DEFAULT_NPROBE
        } else {
            Self::DEFAULT_NPROBE
        });
        let rerank_mult = self.rerank_mult.unwrap_or(Self::RERANK_MULT);
        (nprobe, rerank_mult)
    }
}

/// The stable-id sidecar's location from the footer key-values, `None` on a
/// superfile written before the sidecar existed. Partial keys, or a range
/// past `file_len`, are malformed.
fn id_sidecar_location(
    kv_map: &footer::KvMap,
    file_len: u64,
) -> Result<Option<IdSidecarLocation>, ReadError> {
    if !all_present(kv_map, kv::IDS_KEYS) {
        if any_present(kv_map, kv::IDS_KEYS) {
            return Err(ReadError::MalformedKv(
                "partial inf.ids.* keys present".into(),
            ));
        }
        return Ok(None);
    }
    let offset = parse_u64(kv_map, kv::IDS_OFFSET)?;
    let len = parse_u64(kv_map, kv::IDS_LENGTH)?;
    if offset.checked_add(len).is_none_or(|end| end > file_len) {
        return Err(ReadError::MalformedKv(
            "stable-id sidecar range out of bounds".into(),
        ));
    }
    let packed = kv_map.get(kv::IDS_LAYOUT).map(String::as_str) == Some(kv::IDS_LAYOUT_PACKED);
    Ok(Some(IdSidecarLocation {
        offset,
        len,
        packed,
    }))
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

/// Reader options that decode the `views` columns of a file whose Arrow
/// schema is `file` as `Utf8View` / `BinaryView` when they hold strings or
/// binaries, and every other column as `file` types it. A view of a
/// dictionary-encoded column points into the dictionary rather than copying
/// each value out, and a scan that works on views would otherwise copy every
/// value a second time to make them.
pub(crate) fn view_read_options(file: &Schema, views: &[&str]) -> ArrowReaderOptions {
    let fields: Vec<Field> = file
        .fields()
        .iter()
        .map(|field| {
            let view = match field.data_type() {
                DataType::Utf8 | DataType::LargeUtf8 => Some(DataType::Utf8View),
                DataType::Binary | DataType::LargeBinary => Some(DataType::BinaryView),
                _ => None,
            };
            match view {
                Some(view) if views.contains(&field.name().as_str()) => {
                    field.as_ref().clone().with_data_type(view)
                }
                _ => field.as_ref().clone(),
            }
        })
        .collect();
    ArrowReaderOptions::new().with_schema(Arc::new(
        Schema::new(fields).with_metadata(file.metadata().clone()),
    ))
}

/// `builder` set to read the `projection` columns of every row of row group
/// `row_group` as one batch, and that row group's row count. Shared by
/// [`SuperfileReader::read_row_group`] (resident bytes) and the supertable's
/// read of a lazily opened superfile's row group.
pub(crate) fn whole_row_group<T>(
    builder: ArrowReaderBuilder<T>,
    row_group: usize,
    projection: &[&str],
) -> Result<(ArrowReaderBuilder<T>, usize), ReadError> {
    let meta = builder
        .metadata()
        .row_groups()
        .get(row_group)
        .ok_or_else(|| {
            ReadError::Columnar(format!(
                "row group {row_group} of a file with {}",
                builder.metadata().num_row_groups()
            ))
        })?;
    let rows = usize::try_from(meta.num_rows()).map_err(|_| {
        ReadError::Columnar(format!("row group {row_group} has a negative row count"))
    })?;
    let mut col_indices = Vec::with_capacity(projection.len());
    for &name in projection {
        let idx = builder
            .schema()
            .index_of(name)
            .map_err(|_| ReadError::UnknownColumn(name.to_string()))?;
        col_indices.push(idx);
    }
    let mask = ProjectionMask::roots(builder.parquet_schema(), col_indices);
    let builder = builder
        .with_projection(mask)
        .with_row_groups(vec![row_group])
        .with_batch_size(rows.max(1));
    Ok((builder, rows))
}

/// The batches a [`whole_row_group`] read decoded, as one batch of `rows`
/// rows with the `projection` columns in that order (the reader returns
/// them in file order). The read is set to one batch, so this joins
/// nothing unless the decoder split it.
pub(crate) fn in_projection_order(
    read_schema: &SchemaRef,
    batches: Vec<RecordBatch>,
    projection: &[&str],
    rows: usize,
) -> Result<RecordBatch, ReadError> {
    let mut fields: Vec<Field> = Vec::with_capacity(projection.len());
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(projection.len());
    if !projection.is_empty() {
        let whole = match <[RecordBatch; 1]>::try_from(batches) {
            Ok([batch]) => batch,
            Err(batches) => concat_batches(read_schema, &batches)
                .map_err(|e| ReadError::Columnar(e.to_string()))?,
        };
        for &name in projection {
            let idx = whole
                .schema()
                .index_of(name)
                .map_err(|_| ReadError::UnknownColumn(name.to_string()))?;
            fields.push(whole.schema().field(idx).clone());
            columns.push(Arc::clone(whole.column(idx)));
        }
    }
    RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .map_err(|e| ReadError::Columnar(e.to_string()))
}

/// Sorted, deduplicated copy of `ids` plus the parquet [`RowSelection`]
/// selecting exactly those rows, as strictly monotonic alternating
/// skip/select runs. Decodes only the rows the ids land on — for k=10
/// hits over a 100k-doc body, ~10 page-fragments instead of the whole
/// column (with page indexes present parquet skips whole pages).
///
/// Shared by [`SuperfileReader::take_by_local_doc_ids`] (sync decode
/// over resident bytes) and the supertable's cold object-store row
/// resolution — the row-selection contract is identical even though
/// the I/O models differ. Pair with [`rank_back_indices`] to restore
/// the caller's order afterwards.
pub(crate) fn row_selection_for_ids(ids: &[u32]) -> (Vec<u32>, RowSelection) {
    let mut sorted: Vec<u32> = ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut selectors: Vec<RowSelector> = Vec::with_capacity(sorted.len() * 2 + 1);
    let mut cursor: u32 = 0;
    for &id in &sorted {
        if id > cursor {
            selectors.push(RowSelector::skip((id - cursor) as usize));
        }
        selectors.push(RowSelector::select(1));
        cursor = id + 1;
    }
    (sorted, RowSelection::from(selectors))
}

/// The rows at `ids` (row numbers in the file, in the caller's order,
/// repeats honored) of `batches`, which hold the file's rows whole and in
/// order, as one batch of `schema`. Each column is gathered straight from
/// the decoded batches, so no copy of the whole column is made on the way.
fn gather_rows(
    schema: &SchemaRef,
    batches: &[RecordBatch],
    ids: &[u32],
) -> Result<RecordBatch, ReadError> {
    // Each batch's first row, to find the batch an id falls in.
    let mut starts = Vec::with_capacity(batches.len());
    let mut rows = 0usize;
    for batch in batches {
        starts.push(rows);
        rows += batch.num_rows();
    }
    let at = ids
        .iter()
        .map(|&id| {
            let row = id as usize;
            if row >= rows {
                return Err(ReadError::DocIdOutOfRange {
                    doc_id: id,
                    n_docs: rows as u64,
                });
            }
            // `starts[0]` is 0, so some batch starts at or before `row`.
            let batch = starts.partition_point(|&start| start <= row) - 1;
            Ok((batch, row - starts[batch]))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let columns = (0..schema.fields().len())
        .map(|column| {
            let arrays: Vec<&dyn Array> = batches
                .iter()
                .map(|batch| batch.column(column).as_ref())
                .collect();
            interleave(&arrays, &at).map_err(|e| ReadError::Columnar(e.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new_with_options(
        Arc::clone(schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(ids.len())),
    )
    .map_err(|e| ReadError::Columnar(e.to_string()))
}

/// Rank-back `take` indices restoring the caller's id order (duplicates
/// honored) over rows selected via [`row_selection_for_ids`]: for each
/// requested id, its row position in `sorted` is its row position in
/// the selected batch. Cheap — typical k is 10..1000.
pub(crate) fn rank_back_indices(ids: &[u32], sorted: &[u32]) -> UInt32Array {
    let mut builder = UInt32Array::builder(ids.len());
    for &id in ids {
        let row = sorted
            .binary_search(&id)
            .expect("requested id present in the sorted selection");
        builder.append_value(row as u32);
    }
    builder.finish()
}

/// Whole-file byte attribution; see [`SuperfileReader::size_breakdown`].
#[derive(Debug, Clone, PartialEq)]
pub struct SuperfileSizeBreakdown {
    pub total_bytes: u64,
    pub n_docs: u64,
    pub row_groups: u64,
    /// Compressed bytes of every Parquet column chunk, and per column.
    pub row_group_bytes: u64,
    pub columns: Vec<(String, u64)>,
    pub id_sidecar_bytes: u64,
    pub vec_bytes: u64,
    pub fts_bytes: u64,
    pub fts: Option<fts_reader::FtsSizeBreakdown>,
}

impl SuperfileSizeBreakdown {
    /// Bytes not covered by any region above: the Parquet footer, page
    /// indexes and region trailers.
    pub fn other_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(
            self.row_group_bytes + self.id_sidecar_bytes + self.vec_bytes + self.fts_bytes,
        )
    }
}

/// Bytes per mebibyte, for the report.
const SIZE_REPORT_BYTES_PER_MIB: f64 = 1024.0 * 1024.0;

impl fmt::Display for SuperfileSizeBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mib = |b: u64| b as f64 / SIZE_REPORT_BYTES_PER_MIB;
        writeln!(
            f,
            "superfile: {} docs, {} B ({:.2} MiB)",
            self.n_docs,
            self.total_bytes,
            mib(self.total_bytes)
        )?;
        writeln!(
            f,
            "  parquet row groups ({})  {:>12} B  {:>9.2} MiB",
            self.row_groups,
            self.row_group_bytes,
            mib(self.row_group_bytes)
        )?;
        for (name, b) in &self.columns {
            writeln!(
                f,
                "    column {:<18} {:>12} B  {:>9.2} MiB",
                name,
                b,
                mib(*b)
            )?;
        }
        writeln!(
            f,
            "  id sidecar              {:>12} B  {:>9.2} MiB",
            self.id_sidecar_bytes,
            mib(self.id_sidecar_bytes)
        )?;
        writeln!(
            f,
            "  vector index            {:>12} B  {:>9.2} MiB",
            self.vec_bytes,
            mib(self.vec_bytes)
        )?;
        writeln!(
            f,
            "  fts blob                {:>12} B  {:>9.2} MiB",
            self.fts_bytes,
            mib(self.fts_bytes)
        )?;
        writeln!(
            f,
            "  footer + trailers       {:>12} B  {:>9.2} MiB",
            self.other_bytes(),
            mib(self.other_bytes())
        )?;
        if let Some(fts) = &self.fts {
            write!(f, "{fts}")?;
        }
        Ok(())
    }
}

/// An FTS read error surfaced by the size report is a read error of the
/// file it was read from; any other FTS error stays typed under it.
fn fts_reader_error_to_read(e: FtsError) -> ReadError {
    match e {
        FtsError::Read(r) => r,
        other => ReadError::Fts(Box::new(other)),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arrow::compute::cast;
    use arrow_array::{Decimal128Array, Int64Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field};
    use async_trait::async_trait;
    use tracing::Level;

    use super::*;
    use crate::{
        superfile::{
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
            format::footer::with_forged_footer_kv,
            ids,
            vector::distance::normalize,
        },
        test_helpers::{capture_logs, decimal128_ids, default_vector_config},
    };

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
            vec![FtsConfig::new("title")],
            vec![],
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
    fn id_lookup_returns_only_matching_ids() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        assert_eq!(r.id_lookup_many(&[10]).expect("lookup"), vec![(10, 0)]);
        assert_eq!(r.id_lookup_many(&[12]).expect("lookup"), vec![(12, 2)]);
        assert!(
            r.id_lookup_many(&[20]).expect("lookup").is_empty(),
            "an id the superfile does not hold resolves to nothing"
        );
    }

    #[test]
    fn id_lookup_many_resolves_present_targets_and_skips_absent() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        // 9 and 20 bracket the stored range (10..=13) on both sides;
        // 11 and 13 are present. One scan, three requested.
        let hits = r
            .id_lookup_many(&[9, 11, 13, 20])
            .expect("should return result");
        assert_eq!(hits, vec![(11, 1), (13, 3)]);
    }

    /// Rows in the shuffled-id fixture. Comfortably past the parquet
    /// reader's 1024-row batch size, so the scan crosses several batches
    /// and its `row_offset` accumulation is exercised.
    const SHUFFLED_ID_ROWS: usize = 5_000;
    /// First `_id` in the shuffled fixture.
    const SHUFFLED_ID_BASE: i128 = 1_000;
    /// Stride of the fixture's id permutation. Coprime with
    /// [`SHUFFLED_ID_ROWS`], so `i -> i * STRIDE % ROWS` is a bijection
    /// and every id appears exactly once, in non-monotonic order.
    const SHUFFLED_ID_STRIDE: usize = 37;

    /// A superfile whose `_id` column is deliberately *not* sorted, so the
    /// scan can't accidentally depend on column order (merged and rewritten
    /// superfiles don't guarantee it). Returns the bytes alongside the ids
    /// in row order, which is the oracle's ground truth.
    fn build_shuffled_id_superfile() -> (Bytes, Vec<i128>) {
        let opts = BuilderOptions::new(schema_with_text(), "doc_id", vec![], vec![]);
        build_shuffled_id_superfile_with(opts)
    }

    /// [`build_shuffled_id_superfile`] in row groups of `rows` rows.
    fn build_shuffled_id_superfile_in_row_groups(rows: usize) -> (Bytes, Vec<i128>) {
        let opts = BuilderOptions::new(schema_with_text(), "doc_id", vec![], vec![]);
        build_shuffled_id_superfile_with(BuilderOptions {
            row_group_size: rows,
            ..opts
        })
    }

    fn build_shuffled_id_superfile_with(opts: BuilderOptions) -> (Bytes, Vec<i128>) {
        let schema = schema_with_text();
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

        let ids: Vec<i128> = (0..SHUFFLED_ID_ROWS)
            .map(|i| SHUFFLED_ID_BASE + (i * SHUFFLED_ID_STRIDE % SHUFFLED_ID_ROWS) as i128)
            .collect();
        let id_arr = Decimal128Array::from_iter_values(ids.iter().copied())
            .with_precision_and_scale(38, 0)
            .expect("Decimal128(38,0)");
        let titles: Vec<String> = ids.iter().map(|id| format!("row {id}")).collect();
        let title = LargeStringArray::from(titles.iter().map(|t| t.as_str()).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(schema, vec![Arc::new(id_arr), Arc::new(title)])
            .expect("build RecordBatch");
        b.add_batch(&batch, &[]).expect("add_batch");
        (Bytes::from(b.finish().expect("finish builder")), ids)
    }

    /// Independent oracle: the first row offset at which each target
    /// appears, computed by a naive linear scan of the known id order.
    /// Deliberately shares no code with `id_lookup_many` — the point is to
    /// disagree with it if the batched scan is wrong.
    fn id_lookup_oracle(ids: &[i128], sorted_targets: &[i128]) -> Vec<(i128, u32)> {
        sorted_targets
            .iter()
            .filter_map(|&target| {
                ids.iter()
                    .position(|&id| id == target)
                    .map(|row| (target, row as u32))
            })
            .collect()
    }

    /// The batched scan agrees with a brute-force scan over a multi-batch,
    /// non-monotonic id column, for a mix of present and absent targets.
    #[test]
    fn id_lookup_many_matches_a_brute_force_scan() {
        let (bytes, ids) = build_shuffled_id_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");

        // Present ids spread across the whole column, interleaved with ids
        // inside the stored range that were never written and ids outside
        // it on both sides.
        let mut targets: Vec<i128> = (0..SHUFFLED_ID_ROWS)
            .step_by(311)
            .map(|i| SHUFFLED_ID_BASE + i as i128)
            .collect();
        targets.push(SHUFFLED_ID_BASE - 1);
        targets.push(SHUFFLED_ID_BASE + SHUFFLED_ID_ROWS as i128);
        targets.sort_unstable();
        targets.dedup();

        assert_eq!(
            r.id_lookup_many(&targets).expect("batched lookup"),
            id_lookup_oracle(&ids, &targets),
            "batched scan disagreed with the brute-force scan"
        );
    }

    /// A repeated `_id` resolves to its *first* row, which is the
    /// semantics the single-target scan had and which merged or rewritten
    /// superfiles can actually produce. Without this the "first occurrence
    /// wins" branch is unreachable from the unique-id fixtures.
    #[test]
    fn id_lookup_many_takes_the_first_occurrence_of_a_repeated_id() {
        let schema = schema_with_text();
        let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![]);
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        // Row 1 and row 3 carry the same `_id`.
        let ids = Decimal128Array::from_iter_values([70i128, 71, 72, 71, 73])
            .with_precision_and_scale(38, 0)
            .expect("Decimal128(38,0)");
        let title = LargeStringArray::from(vec!["a", "b", "c", "b again", "d"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(title)])
            .expect("build RecordBatch");
        b.add_batch(&batch, &[]).expect("add_batch");
        let r = SuperfileReader::open(Bytes::from(b.finish().expect("finish"))).expect("open");

        assert_eq!(
            r.id_lookup_many(&[71, 73]).expect("lookup"),
            vec![(71, 1), (73, 4)],
            "the duplicate resolves to row 1, not row 3"
        );
    }

    /// Every single-target lookup over the same fixture agrees with the
    /// oracle too, so a bug that only shows up at batch size one — or only
    /// past the early-exit — still gets caught.
    #[test]
    fn id_lookup_many_matches_the_oracle_one_target_at_a_time() {
        let (bytes, ids) = build_shuffled_id_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");

        for i in (0..SHUFFLED_ID_ROWS).step_by(499) {
            let target = SHUFFLED_ID_BASE + i as i128;
            assert_eq!(
                r.id_lookup_many(&[target]).expect("single-target lookup"),
                id_lookup_oracle(&ids, &[target]),
                "target {target} disagreed with the brute-force scan"
            );
        }
    }

    #[test]
    fn id_lookup_many_on_empty_batch_is_empty() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        assert!(
            r.id_lookup_many(&[]).expect("empty batch").is_empty(),
            "empty target batch resolves to no hits"
        );
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
    fn take_by_local_doc_ids_resolves_id_and_scalar_columns() {
        // ids 10,11,12,13 / titles per row; local offsets index rows.
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        // Offsets 2 then 0 — order preserved (ids 12 then 10).
        let batch = r
            .take_by_local_doc_ids(&[2, 0], &["doc_id", "title"])
            .expect("take");
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.schema().field(0).name(), "doc_id");
        assert_eq!(batch.schema().field(1).name(), "title");
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal ids");
        assert_eq!(ids.value(0), 12_i128);
        assert_eq!(ids.value(1), 10_i128);
        let titles = batch
            .column(1)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("titles");
        assert_eq!(titles.value(0), "rust embedded system");
        assert_eq!(titles.value(1), "rust async runtime");
    }

    #[test]
    fn take_by_local_doc_ids_respects_projection_order() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        // Reverse of file order: title before doc_id.
        let batch = r
            .take_by_local_doc_ids(&[1], &["title", "doc_id"])
            .expect("take");
        assert_eq!(batch.schema().field(0).name(), "title");
        assert_eq!(batch.schema().field(1).name(), "doc_id");
        let titles = batch
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("titles");
        assert_eq!(titles.value(0), "python data pipeline");
    }

    #[test]
    fn take_by_local_doc_ids_id_only_projection() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let batch = r
            .take_by_local_doc_ids(&[0, 1, 2, 3], &["doc_id"])
            .expect("take");
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.num_rows(), 4);
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal ids");
        assert_eq!(ids.value(0), 10_i128);
        assert_eq!(ids.value(3), 13_i128);
    }

    /// A fresh build writes the stable-id sidecar, and resolving `_id`
    /// through it yields exactly what the Parquet id column would — over
    /// non-monotonic ids (so span arithmetic can't apply) with duplicates
    /// and out-of-order local ids.
    #[test]
    fn stable_id_sidecar_serves_id_and_matches_parquet_fallback() {
        let (bytes, ids) = build_shuffled_id_superfile();

        // The sidecar KVs are present and name the packed layout, which
        // never costs more than one raw i128 per row plus its headers.
        let kvs = footer::read_kv_metadata(&bytes).expect("kv metadata");
        let len: usize = kvs
            .get(kv::IDS_LENGTH)
            .expect("sidecar length present")
            .parse()
            .expect("length is a usize");
        assert_eq!(
            kvs.get(kv::IDS_LAYOUT).map(String::as_str),
            Some(kv::IDS_LAYOUT_PACKED)
        );
        assert!(len <= ids.len() * format::ID_SIDECAR_ENTRY_BYTES + 4096);

        let locals: Vec<u32> = vec![0, 4999, 1, 4999, 37, 100, 2500];
        let expected: Vec<i128> = locals.iter().map(|&l| ids[l as usize]).collect();

        let mut r = SuperfileReader::open(bytes).expect("open superfile");
        assert!(
            r.id_sidecar.is_some(),
            "eager open records the sidecar range"
        );

        let via_sidecar = r
            .take_by_local_doc_ids(&locals, &["doc_id"])
            .expect("take via sidecar");

        // Forcing the fallback (as a pre-sidecar or merged superfile would)
        // must produce byte-identical output.
        r.id_sidecar = None;
        let via_parquet = r
            .take_by_local_doc_ids(&locals, &["doc_id"])
            .expect("take via parquet");
        assert_eq!(via_sidecar, via_parquet, "sidecar and Parquet _id agree");

        let got = via_sidecar
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal ids");
        let got: Vec<i128> = (0..got.len()).map(|i| got.value(i)).collect();
        assert_eq!(got, expected, "resolved ids match the row-order oracle");
    }

    /// A lazily opened superfile resolves a few rows' `_id` from the packed
    /// sidecar's blocks alone — the directory and the blocks the rows fall
    /// in — and gets what the row-order oracle says; a damaged block is an
    /// error, and marks the sidecar broken so later reads go to Parquet.
    #[tokio::test]
    async fn stable_id_sidecar_blocks_serve_a_few_rows_of_a_lazy_reader() {
        let (bytes, ids) = build_shuffled_id_superfile();
        let locals: Vec<u32> = vec![0, 4999, 1, 4999, 37, 100, 2500];
        let expected: Vec<i128> = locals.iter().map(|&l| ids[l as usize]).collect();
        let lazy = lazy_shuffled_id_reader(bytes.clone()).await;
        let got = lazy
            .ids_from_sidecar_blocks(&locals)
            .await
            .expect("read")
            .expect("a packed sidecar serves");
        let got = got
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal ids");
        let got: Vec<i128> = (0..got.len()).map(|i| got.value(i)).collect();
        assert_eq!(got, expected, "ids from the blocks match the oracle");

        // Blocks next to each other share a read and blocks apart do not;
        // either way each block's ids come out of its own slice.
        for locals in [vec![5u32, 1030], vec![5, 4100], vec![1030, 5, 4100, 2050]] {
            let expected: Vec<i128> = locals.iter().map(|&l| ids[l as usize]).collect();
            let got = lazy
                .ids_from_sidecar_blocks(&locals)
                .await
                .expect("read")
                .expect("a packed sidecar serves");
            let got = got
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("decimal ids");
            let got: Vec<i128> = (0..got.len()).map(|i| got.value(i)).collect();
            assert_eq!(got, expected, "rows {locals:?}");
        }

        // A directory that starts block 2 where block 0 starts is malformed,
        // not a panic.
        let reordered = Bytes::from(ids::with_out_of_order_id_block(&bytes, 2, 0));
        let lazy = lazy_shuffled_id_reader(reordered).await;
        assert!(
            lazy.ids_from_sidecar_blocks(&[5, 2050]).await.is_err(),
            "an out-of-order block is an error, for the Parquet fallback"
        );

        let last_block = SHUFFLED_ID_ROWS.div_ceil(ids::BLOCK_DOCS) - 1;
        let damaged = Bytes::from(ids::with_damaged_id_block(&bytes, last_block));
        let lazy = lazy_shuffled_id_reader(damaged).await;
        assert!(
            lazy.ids_from_sidecar_blocks(&[4999]).await.is_err(),
            "a row in the damaged block is an error, for the Parquet fallback"
        );
        assert!(
            lazy.ids_from_sidecar_blocks(&[0, 37])
                .await
                .expect("read")
                .is_none(),
            "a sidecar found broken is not read again"
        );
    }

    /// With `_id` alongside a scalar column, the sidecar serves `_id` and the
    /// Parquet read serves the scalar; the result equals the all-Parquet path
    /// and preserves projection order.
    #[test]
    fn stable_id_sidecar_with_scalar_projection_matches_fallback() {
        let (bytes, ids) = build_shuffled_id_superfile();
        let locals: Vec<u32> = vec![10, 3, 3, 4998];
        let mut r = SuperfileReader::open(bytes).expect("open superfile");

        let with = r
            .take_by_local_doc_ids(&locals, &["title", "doc_id"])
            .expect("take with sidecar");
        r.id_sidecar = None;
        let without = r
            .take_by_local_doc_ids(&locals, &["title", "doc_id"])
            .expect("take without sidecar");
        assert_eq!(with, without, "mixed projection agrees across paths");

        assert_eq!(with.schema().field(0).name(), "title");
        assert_eq!(with.schema().field(1).name(), "doc_id");
        let id_col = with
            .column(1)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal ids");
        assert_eq!(id_col.value(0), ids[10]);
    }

    #[test]
    fn a_bad_id_sidecar_block_opens_and_reads_id_from_parquet() {
        // A 5,000-row superfile with the last block of its id sidecar broken.
        //  - `open` succeeds, since it checks only the sidecar's header.
        //  - `_id` for a doc in a good block comes from the sidecar.
        //  - `_id` for a doc in the broken block comes from Parquet instead, and
        //    marks the sidecar broken.
        //  - Once marked, every read takes `_id` from Parquet, still right, and
        //    the warning is logged only once.
        // This test pins that a bad block costs neither the open nor the answer.
        let (bytes, ids) = build_shuffled_id_superfile();
        let last_block = SHUFFLED_ID_ROWS.div_ceil(ids::BLOCK_DOCS) - 1;
        let damaged = ids::with_damaged_id_block(&bytes, last_block);
        let r = SuperfileReader::open(Bytes::from(damaged)).expect("open reads only the header");
        let read_ids = |docs: &[u32], projection: &[&str]| -> Vec<i128> {
            let batch = r.take_by_local_doc_ids(docs, projection).expect("take");
            let at = projection
                .iter()
                .position(|&n| n == "doc_id")
                .expect("doc_id projected");
            let col = batch
                .column(at)
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("decimal ids");
            col.values().to_vec()
        };
        let bad = SHUFFLED_ID_ROWS - 1;

        let ((), logs) = capture_logs(Level::WARN, || {
            assert_eq!(read_ids(&[0], &["doc_id"]), [ids[0]]);
            assert!(!r.id_sidecar_broken.load(Ordering::Relaxed));
            assert_eq!(
                read_ids(&[0, bad as u32], &["title", "doc_id"]),
                [ids[0], ids[bad]]
            );
            assert!(r.id_sidecar_broken.load(Ordering::Relaxed));
            assert_eq!(read_ids(&[bad as u32], &["doc_id"]), [ids[bad]]);
            assert_eq!(read_ids(&[0], &["doc_id"]), [ids[0]]);
        });
        assert_eq!(
            logs.matches("the id sidecar is unreadable").count(),
            1,
            "{logs}"
        );
    }

    #[test]
    fn row_selection_and_rank_back_honor_duplicates_and_gaps() {
        // Caller order with a duplicate and out-of-order ids.
        let ids = [7u32, 2, 7, 0];
        let (sorted, selection) = row_selection_for_ids(&ids);
        assert_eq!(sorted, vec![0, 2, 7]);
        // select(0), skip(1), select(2), skip(4..7), select(7)
        assert_eq!(selection.row_count(), 3, "one selected row per distinct id");

        let indices = rank_back_indices(&ids, &sorted);
        // Positions in `sorted`: 7→2, 2→1, 7→2, 0→0.
        let got: Vec<u32> = (0..indices.len()).map(|i| indices.value(i)).collect();
        assert_eq!(got, vec![2, 1, 2, 0]);
    }

    /// The whole-column gather takes rows from across the decoded batches,
    /// in the caller's order, repeats and the rows either side of a batch
    /// boundary included, and refuses a row past the file.
    #[test]
    fn gather_rows_crosses_batches_in_the_callers_order() {
        let schema: SchemaRef =
            Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]));
        // Rows 0..9 as three batches of uneven length; each value is its row.
        let batches: Vec<RecordBatch> = [0..3i64, 3..5, 5..9]
            .into_iter()
            .map(|rows| {
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Int64Array::from_iter_values(rows)) as ArrayRef],
                )
                .expect("batch")
            })
            .collect();
        let ids = [8u32, 0, 3, 2, 8, 5, 4];
        let gathered = gather_rows(&schema, &batches, &ids).expect("gather");
        let got = gathered
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64")
            .values()
            .to_vec();
        assert_eq!(got, ids.map(i64::from).to_vec());

        let err = gather_rows(&schema, &batches, &[9]).expect_err("past the file");
        assert!(matches!(
            err,
            ReadError::DocIdOutOfRange {
                doc_id: 9,
                n_docs: 9
            }
        ));
    }

    #[test]
    fn take_by_local_doc_ids_empty_returns_empty_batch() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let batch = r
            .take_by_local_doc_ids(&[], &["doc_id", "title"])
            .expect("take");
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 2);
    }

    #[test]
    fn take_by_local_doc_ids_unknown_column_errors() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let err = r
            .take_by_local_doc_ids(&[0], &["nope"])
            .expect_err("expected error");
        assert!(matches!(err, ReadError::UnknownColumn(_)));
    }

    #[test]
    fn take_by_local_doc_ids_out_of_range_errors() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let err = r
            .take_by_local_doc_ids(&[4], &["doc_id"])
            .expect_err("expected error");
        assert!(matches!(
            err,
            ReadError::DocIdOutOfRange {
                doc_id: 4,
                n_docs: 4
            }
        ));
    }

    /// The page index is tens of MiB per superfile and only the take /
    /// SQL-scan paths read it, so an eager open must not parse it. It must
    /// still be available, and identical, on first demand — parsed from the
    /// resident bytes, not fetched.
    #[tokio::test]
    async fn eager_open_defers_the_page_index_until_something_asks_for_it() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open_with(bytes, OpenOptions { verify_crc: false })
            .expect("open eager");
        assert!(
            r.parquet_meta.offset_index().is_none(),
            "an eager open must not parse the page index"
        );
        let indexed = r
            .parquet_metadata_with_page_index()
            .await
            .expect("page index on demand");
        assert_eq!(
            indexed.num_row_groups(),
            r.parquet_meta.num_row_groups(),
            "the on-demand parse must describe the same file"
        );
        // Cached: a second call returns the same allocation.
        let again = r
            .parquet_metadata_with_page_index()
            .await
            .expect("page index cached");
        assert!(Arc::ptr_eq(&indexed, &again), "the parse must happen once");
    }

    #[tokio::test]
    async fn bm25_search_finds_matching_docs() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let hits = r
            .bm25_hits_async("title", "rust", 5, BoolMode::Or)
            .await
            .expect("BM25 search");
        // docs 0 and 2 contain "rust"; both should appear.
        let doc_ids: HashSet<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        assert!(doc_ids.contains(&0));
        assert!(doc_ids.contains(&2));
    }

    #[tokio::test]
    async fn bm25_search_errors_when_no_fts() {
        // Build a superfile with no FTS, no vec.
        let schema = schema_with_text();
        let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![]);
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let ids = decimal128_ids(vec![1u64]);
        let title = LargeStringArray::from(vec!["x"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(title)])
            .expect("build RecordBatch");
        b.add_batch(&batch, &[]).expect("add_batch");
        let bytes = Bytes::from(b.finish().expect("finish builder"));
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let err = r
            .bm25_hits_async("nope", "x", 1, BoolMode::Or)
            .await
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
        use parquet::basic::Compression;

        use crate::superfile::format::footer::{encode_parquet_body, splice_index_blobs};
        let schema = schema_with_text();
        let ids = decimal128_ids(vec![1u64]);
        let title = LargeStringArray::from(vec!["x"]);
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(title)])
            .expect("build RecordBatch");
        let body = encode_parquet_body(&schema, &[batch], Compression::SNAPPY, 1024, &[])
            .expect("encode parquet body");
        let parts = splice_index_blobs(body, &[], &[], &[], &[]).expect("splice index blobs");
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
        assert_eq!(opts.nprobe, None);
        assert_eq!(opts.rerank_mult(), None);
        assert_eq!(opts.resolve(false), (6, 256));
        assert_eq!(opts.resolve(true), (8, 256));
    }

    #[test]
    fn vector_search_options_builder_chains() {
        let opts = VectorSearchOptions::new()
            .with_nprobe(2)
            .with_rerank_mult(32);
        assert_eq!(opts.nprobe, Some(2));
        assert_eq!(opts.rerank_mult(), Some(32));
        assert_eq!(opts.resolve(false), (2, 32));
        assert_eq!(opts.resolve(true), (2, 32));
    }

    #[tokio::test]
    async fn vector_search_with_default_options_succeeds() {
        // Confirms the default options path actually executes without
        // panicking; the recall is exercised in tests/recall.rs.
        let bytes = build_vector_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let mut q = vec![0.0f32; 16];
        q[2] = 1.0;
        q[5] = 0.5;
        normalize(&mut q);
        let hits = r
            .vector_hits_async("emb", &q, 1, VectorSearchOptions::default())
            .await
            .expect("vector search");
        assert!(!hits.is_empty());
    }

    #[tokio::test]
    async fn vector_search_finds_self() {
        let bytes = build_vector_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        // Query equal to doc 2's vector → top hit must be doc 2.
        let mut q = vec![0.0f32; 16];
        q[2] = 1.0;
        q[5] = 0.5;
        normalize(&mut q);
        let hits = r
            .vector_hits_async("emb", &q, 1, VectorSearchOptions::new().with_nprobe(4))
            .await
            .expect("vector search");
        assert_eq!(hits[0].0, 2);
    }

    #[test]
    fn parquet_bytes_round_trips_as_parquet() {
        // The whole buffer should still be a valid Parquet file.
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let p = r
            .parquet_bytes()
            .expect("eager open should retain parquet bytes")
            .clone();
        let builder = ParquetRecordBatchReaderBuilder::try_new(p)
            .expect("try_new ParquetRecordBatchReaderBuilder");
        let mut reader = builder.build().expect("build parquet reader");
        let batch = reader.next().expect("one batch").expect("decode batch");
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 2);
    }

    #[tokio::test]
    async fn unknown_column_in_search_propagates_fts_error() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let err = r
            .bm25_hits_async("nonexistent", "rust", 5, BoolMode::Or)
            .await
            .expect_err("expected error");
        assert!(matches!(err, ReadError::Fts(_)));
    }

    #[tokio::test]
    async fn bm25_search_multi_combines_columns() {
        // Build a 2-FTS-column file.
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("body", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(
            schema.clone(),
            "doc_id",
            vec![FtsConfig::new("title"), FtsConfig::new("body")],
            vec![],
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
            .await
            .expect("BM25 multi-column search");
        // Both doc 0 (title:rust) and doc 1 (body:rust) hit.
        let doc_ids: HashSet<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        assert!(doc_ids.contains(&0));
        assert!(doc_ids.contains(&1));
    }

    // ── Additional coverage ───────────────────────────────────────────

    #[test]
    fn a_scored_view_is_memoized_per_parameter_pair() {
        // No override borrows the baked reader; an override derives a view
        // once and hands the same one back while the pair is unchanged,
        // and a different pair replaces it — the memo is last-writer-wins
        // and never keeps a stale pair.
        let r = SuperfileReader::open(build_simple_fts_only_superfile()).expect("open");
        assert!(matches!(
            r.fts_scored(&ScoringView::default()).expect("baked"),
            ScoredFts::Baked(_)
        ));
        let a = Bm25Params::new(1.4, 0.6);
        let b = Bm25Params::new(0.9, 0.4);
        let at = |params| ScoringView {
            params: Some(params),
            avgdl: None,
        };
        let first = match r.fts_scored(&at(a)).expect("derive") {
            ScoredFts::Derived(v) => v,
            ScoredFts::Baked(_) => panic!("an override must derive a view"),
        };
        let again = match r.fts_scored(&at(a)).expect("reuse") {
            ScoredFts::Derived(v) => v,
            ScoredFts::Baked(_) => panic!("an override must derive a view"),
        };
        assert!(
            Arc::ptr_eq(&first, &again),
            "same pair: the memoized view is reused"
        );
        let other = match r.fts_scored(&at(b)).expect("derive another") {
            ScoredFts::Derived(v) => v,
            ScoredFts::Baked(_) => panic!("an override must derive a view"),
        };
        assert!(
            !Arc::ptr_eq(&first, &other),
            "a different pair derives a different view"
        );
        assert_eq!(other.fts_columns_config().next().expect("column").params, b);
        assert!(matches!(
            r.fts_scored(&ScoringView::default()).expect("baked"),
            ScoredFts::Baked(_)
        ));
    }

    /// A view at the average a column was built at changes nothing, so the
    /// baked reader serves it; another average derives a view that rescores
    /// that column alone and widens its stored bounds.
    #[test]
    fn a_scored_view_moves_only_the_named_column_average() {
        let r = SuperfileReader::open(build_simple_fts_only_superfile()).expect("open");
        let fts = r.fts().expect("fts");
        let column = fts.fts_columns_config().next().expect("column");
        let (name, baked) = (column.name.clone(), column.avgdl());
        let at = |avgdl| ScoringView {
            params: None,
            avgdl: Some((name.as_str(), avgdl)),
        };
        assert!(matches!(
            r.fts_scored(&at(baked)).expect("baked"),
            ScoredFts::Baked(_)
        ));
        let wider = 2.0 * baked;
        let view = match r.fts_scored(&at(wider)).expect("derive") {
            ScoredFts::Derived(v) => v,
            ScoredFts::Baked(_) => panic!("another average must derive a view"),
        };
        let col = view
            .fts_columns_config()
            .find(|c| c.name == name)
            .expect("column");
        assert_eq!(col.avgdl(), wider);
        assert!(col.bound_scale() > 1.0, "a longer average raises scores");
    }

    #[test]
    fn open_with_verify_crc_off_succeeds() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open_with(bytes, OpenOptions { verify_crc: false })
            .expect("open with crc off");
        assert_eq!(r.n_docs(), 4);
        assert!(r.fts().is_some());
    }

    #[test]
    fn default_open_options_verifies_crc() {
        assert!(OpenOptions::default().verify_crc);
    }

    #[test]
    fn parquet_metadata_reports_row_count() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        let meta = r.parquet_metadata();
        let total: i64 = meta.row_groups().iter().map(|rg| rg.num_rows()).sum();
        assert_eq!(total, 4);
    }

    #[tokio::test]
    async fn lazy_parquet_metadata_loads_offset_index_once() {
        let bytes = build_simple_fts_only_superfile();
        let source: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(bytes));
        let reader = SuperfileReader::open_lazy(source).await.expect("lazy open");
        assert!(
            reader.parquet_metadata().offset_index().is_none(),
            "footer-only lazy open should not preload page indexes"
        );

        let first = reader
            .parquet_metadata_with_page_index()
            .await
            .expect("load offset index");
        assert!(
            first.offset_index().is_some(),
            "targeted row reads require the Parquet offset index"
        );
        let second = reader
            .parquet_metadata_with_page_index()
            .await
            .expect("reuse offset index");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn debug_impl_reports_index_presence() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        let s = format!("{r:?}");
        assert!(s.contains("SuperfileReader"));
        assert!(s.contains("has_fts: true"));
        assert!(s.contains("has_vec: false"));
    }

    #[test]
    fn byte_source_over_eager_bytes_serves_ranges() {
        let bytes = build_simple_fts_only_superfile();
        let total_len = bytes.len() as u64;
        let r = SuperfileReader::open(bytes).expect("open");
        let src = r.byte_source();
        assert_eq!(src.size(), total_len);
    }

    #[test]
    fn get_record_batch_returns_all_rows() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        let batch = r.get_record_batch(None).expect("get_record_batch");
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 2);
    }

    #[test]
    fn get_record_batch_honors_tombstones() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        let mut deleted = RoaringBitmap::new();
        deleted.insert(0);
        deleted.insert(2);
        let batch = r
            .get_record_batch(Some(Arc::new(deleted)))
            .expect("get_record_batch with tombstones");
        // 4 docs, 2 deleted ⇒ 2 live rows survive.
        assert_eq!(batch.num_rows(), 2);
    }

    #[tokio::test]
    async fn bm25_search_pretokenized_matches_hits_path() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        let hits = r
            .bm25_search_pretokenized("title", &["rust"], 5, BoolMode::Or)
            .await
            .expect("pretokenized search");
        let ids: HashSet<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
    }

    #[tokio::test]
    async fn bm25_search_pretokenized_with_floor_prunes() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        let hits = r
            .bm25_search_pretokenized_with_floor("title", &["rust"], 5, BoolMode::Or, 1e9)
            .await
            .expect("floored search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn bm25_hits_async_honors_negation() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        // doc 0 "rust async runtime", doc 2 "rust embedded system":
        // "rust -embedded" keeps doc 0, drops doc 2.
        let hits = r
            .bm25_hits_async("title", "rust -embedded", 5, BoolMode::Or)
            .await
            .expect("negated search");
        let ids: Vec<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        assert!(ids.contains(&0));
        assert!(!ids.contains(&2));
    }

    #[tokio::test]
    async fn token_match_intersects_and_unions() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        // "rust" in docs 0,2; "runtime" in doc 0 only.
        let and = r
            .token_match("title", &["rust", "runtime"], BoolMode::And)
            .await
            .expect("and")
            .0;
        assert_eq!(and, vec![0]);
        let or = r
            .token_match("title", &["rust", "runtime"], BoolMode::Or)
            .await
            .expect("or")
            .0;
        assert_eq!(or, vec![0, 2]);
    }

    #[tokio::test]
    async fn term_df_counts_documents() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        assert_eq!(r.term_df("title", "rust").await.expect("df").0, 2);
        assert_eq!(r.term_df("title", "absent").await.expect("df").0, 0);
    }

    #[tokio::test]
    async fn exact_match_verifies_raw_string_equality() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        // A token subset ("rust") prunes candidates but only the exact
        // stored string matches.
        let hits = r
            .exact_match("title", "rust async runtime")
            .await
            .expect("exact_match")
            .0;
        assert_eq!(hits, vec![0]);
        // No row stores just "rust", so the exact comparison yields none.
        let none = r.exact_match("title", "rust").await.expect("exact_match").0;
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn bm25_search_or_range_restricts_window() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        // docs 0,2 contain "rust"/"embedded"; restrict to [0,2) ⇒ doc 0 only.
        let hits = r
            .bm25_search_or_range_pretokenized("title", &["rust", "embedded"], 10, 0, 2)
            .await
            .expect("ranged search");
        let ids: HashSet<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        assert!(ids.contains(&0));
        assert!(!ids.contains(&2));
    }

    #[tokio::test]
    async fn bm25_search_or_range_with_floor_prunes() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        let hits = r
            .bm25_search_or_range_pretokenized_with_floor(
                "title",
                &["rust", "embedded"],
                10,
                0,
                4,
                1e9,
                fts_reader::QueryIdf::ThisFile,
            )
            .await
            .expect("floored ranged search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn vector_search_clusters_probes_explicit_clusters() {
        let bytes = build_vector_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        let mut q = vec![0.0f32; 16];
        q[2] = 1.0;
        q[5] = 0.5;
        normalize(&mut q);
        // Probe all 4 clusters explicitly; doc 2 (the query's own vector)
        // must be reachable.
        let hits = r
            .vector_search_clusters("emb", &q, 4, &[0, 1, 2, 3], VectorSearchOptions::default())
            .await
            .expect("cluster search");
        let ids: HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(ids.contains(&2));
    }

    #[tokio::test]
    async fn vector_search_errors_when_no_vector_index() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        let err = r
            .vector_hits_async("emb", &[0.0f32; 16], 1, VectorSearchOptions::default())
            .await
            .expect_err("expected error");
        assert!(matches!(err, ReadError::MissingKv(_)));
    }

    /// The lazy reader of [`build_shuffled_id_superfile`], whose ids are
    /// out of row order, so a decode that loses the order cannot pass.
    async fn lazy_shuffled_id_reader(bytes: Bytes) -> SuperfileReader {
        let src: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(bytes));
        SuperfileReader::open_lazy(src).await.expect("open_lazy")
    }

    /// A lazily opened reader keeps the sidecar's location and fetches it
    /// whole: every id it decodes is the id the superfile was built with, in
    /// row order, for any request order and with repeats.
    #[tokio::test]
    async fn lazy_reader_fetches_the_id_sidecar_in_row_order() {
        let (bytes, ids) = build_shuffled_id_superfile();
        let lazy = lazy_shuffled_id_reader(bytes).await;
        assert!(lazy.parquet_bytes().is_none());

        let sidecar = lazy
            .fetch_id_sidecar()
            .await
            .expect("fetch")
            .expect("the superfile carries a sidecar");
        let every: Vec<u32> = (0..SHUFFLED_ID_ROWS as u32).collect();
        assert_eq!(sidecar.ids(&every).expect("decode"), ids);

        let last = SHUFFLED_ID_ROWS as u32 - 1;
        let locals = [last, 0, 37, last, last / 2];
        let expected: Vec<i128> = locals.iter().map(|&l| ids[l as usize]).collect();
        assert_eq!(sidecar.ids(&locals).expect("decode"), expected);
        assert!(matches!(
            sidecar.ids(&[last + 1]),
            Err(ReadError::DocIdOutOfRange { .. })
        ));
    }

    /// A sidecar that does not decode on the cold path is marked broken, and
    /// a marked sidecar is not fetched again: the caller reads `_id` from
    /// another source once, not once per query.
    #[tokio::test]
    async fn a_lazy_reader_does_not_refetch_a_sidecar_marked_broken() {
        let (bytes, _) = build_shuffled_id_superfile();
        let last_block = SHUFFLED_ID_ROWS.div_ceil(ids::BLOCK_DOCS) - 1;
        let damaged = Bytes::from(ids::with_damaged_id_block(&bytes, last_block));
        let lazy = lazy_shuffled_id_reader(damaged).await;
        let sidecar = lazy
            .fetch_id_sidecar()
            .await
            .expect("fetch")
            .expect("the header is fine, so the fetch is");
        let every: Vec<u32> = (0..SHUFFLED_ID_ROWS as u32).collect();
        assert!(matches!(
            sidecar.ids(&every),
            Err(ReadError::MalformedKv(_))
        ));
        lazy.mark_id_sidecar_broken();
        assert!(lazy.fetch_id_sidecar().await.expect("fetch").is_none());
    }

    /// A promoted reader (opened lazily, then given the disk cache's fill
    /// file) slices the sidecar from its local copy when the copy's hole
    /// leaves the sidecar out, as the promotion's vector-blob hole does, and
    /// not when the hole reaches into it; `_id` is right either way.
    #[tokio::test]
    async fn a_promoted_reader_slices_its_sidecar_unless_the_hole_covers_it() {
        let (bytes, ids) = build_shuffled_id_superfile();
        let every: Vec<u32> = (0..SHUFFLED_ID_ROWS as u32).collect();
        let loc = lazy_shuffled_id_reader(bytes.clone())
            .await
            .id_sidecar
            .expect("the superfile carries a sidecar");

        let mut beside = lazy_shuffled_id_reader(bytes.clone()).await;
        beside
            .install_resident_parquet(bytes.clone(), (loc.offset - 1, 1))
            .expect("promote with a hole ending where the sidecar starts");
        let sliced = beside
            .resident_id_sidecar()
            .expect("the local copy holds the whole sidecar");
        assert_eq!(sliced.ids(&every).expect("decode"), ids);

        let mut over = lazy_shuffled_id_reader(bytes.clone()).await;
        over.install_resident_parquet(bytes, (loc.offset + loc.len - 1, 1))
            .expect("promote with a hole over the sidecar's last byte");
        assert!(over.resident_id_sidecar().is_none());
        let last = SHUFFLED_ID_ROWS as u32 - 1;
        let batch = over
            .take_by_local_doc_ids(&[last, 0], &["doc_id"])
            .expect("take _id from resident Parquet");
        let got = batch
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal _id");
        assert_eq!(got.values().to_vec(), vec![ids[last as usize], ids[0]]);
        assert!(over.fetch_id_sidecar().await.expect("fetch").is_some());
    }

    /// The sidecar serves a cold read of the id column that names at least
    /// half the file's rows, counted once each, and only a packed sidecar:
    /// a resident reader, a smaller subset, another column or a raw sidecar
    /// keep their own source.
    #[tokio::test]
    async fn the_sidecar_serves_cold_reads_of_half_the_rows_or_more() {
        let (bytes, _) = build_shuffled_id_superfile();
        let rows = SHUFFLED_ID_ROWS as u32;
        let every: Vec<u32> = (0..rows).collect();
        let half = (SHUFFLED_ID_ROWS as u64).div_ceil(SIDECAR_MIN_ROW_SHARE) as usize;
        let lazy = lazy_shuffled_id_reader(bytes.clone()).await;
        assert!(lazy.id_sidecar_serves(&every, "doc_id"));
        assert!(
            lazy.id_sidecar_serves(&every[1..], "doc_id"),
            "one row short of every row, as an allow-set missing a deleted row"
        );
        assert!(lazy.id_sidecar_serves(&every[..half], "doc_id"));
        assert!(!lazy.id_sidecar_serves(&every[..half - 1], "doc_id"));
        assert!(
            !lazy.id_sidecar_serves(&vec![last_row(rows); rows as usize], "doc_id"),
            "as many entries as rows, all one row, is one row"
        );
        assert!(!lazy.id_sidecar_serves(&every, "title"));

        let mut raw = lazy_shuffled_id_reader(bytes.clone()).await;
        raw.id_sidecar = raw.id_sidecar.map(|loc| IdSidecarLocation {
            packed: false,
            ..loc
        });
        assert!(
            !raw.id_sidecar_serves(&every, "doc_id"),
            "a raw sidecar is sixteen bytes a row, more than the column"
        );

        let eager = SuperfileReader::open(bytes.clone()).expect("open");
        assert!(!eager.id_sidecar_serves(&every, "doc_id"));
    }

    /// The last row of a file of `rows` rows.
    fn last_row(rows: u32) -> u32 {
        rows - 1
    }

    /// A source that hands back one byte short of any read starting at `at`,
    /// as a storage backend that clamps a range without an error would.
    struct ShortAt {
        inner: BytesLazyByteSource,
        at: u64,
    }

    #[async_trait]
    impl LazyByteSource for ShortAt {
        fn size(&self) -> u64 {
            self.inner.size()
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            let bytes = self.inner.range(start, len).await?;
            Ok(match start == self.at {
                true => bytes.slice(..bytes.len() - 1),
                false => bytes,
            })
        }
    }

    /// A sidecar read that comes back short is a failed read, not a damaged
    /// sidecar: it errors as a fetch, leaves the sidecar unmarked, and the
    /// next read tries the sidecar again.
    #[tokio::test]
    async fn a_short_sidecar_read_is_a_failed_fetch_not_a_damaged_sidecar() {
        let (bytes, _) = build_shuffled_id_superfile();
        let loc = lazy_shuffled_id_reader(bytes.clone())
            .await
            .id_sidecar
            .expect("the superfile carries a sidecar");
        let src: Arc<dyn LazyByteSource> = Arc::new(ShortAt {
            inner: BytesLazyByteSource::new(bytes),
            at: loc.offset,
        });
        let lazy = SuperfileReader::open_lazy(src).await.expect("open_lazy");
        for attempt in 0..2 {
            assert!(
                matches!(
                    lazy.fetch_id_sidecar().await,
                    Err(ReadError::RangeFetch {
                        source: LazyByteSourceError::ShortRead { .. },
                        ..
                    })
                ),
                "attempt {attempt}: a short read is a fetch error"
            );
            assert!(!lazy.id_sidecar_broken.load(Ordering::Relaxed));
        }
    }

    /// Rows per row group in the whole-row-group read test: not a divisor
    /// of [`SHUFFLED_ID_ROWS`], so the last row group is short.
    const READ_ROW_GROUP_ROWS: usize = 1_500;

    /// A whole row group reads the same rows, in the same order, as a take
    /// of every row it holds, with the columns in the order asked; a row
    /// group past the last, or a column the file lacks, is an error; and
    /// no column still carries the row count. A column read as views holds
    /// the same strings as `Utf8View`.
    #[test]
    fn a_whole_row_group_reads_what_a_take_of_its_rows_does() {
        let (bytes, _) = build_shuffled_id_superfile_in_row_groups(READ_ROW_GROUP_ROWS);
        let reader = SuperfileReader::open(bytes).expect("open");
        let groups = reader.parquet_metadata().num_row_groups();
        assert_eq!(groups, SHUFFLED_ID_ROWS.div_ceil(READ_ROW_GROUP_ROWS));
        let projection = ["title", "doc_id"];
        let mut start = 0u32;
        for row_group in 0..groups {
            let read = reader
                .read_row_group(row_group, &projection, &[])
                .expect("whole row group");
            let rows: Vec<u32> = (start..start + read.num_rows() as u32).collect();
            let taken = reader
                .take_by_local_doc_ids(&rows, &projection)
                .expect("take");
            assert_eq!(read, taken, "row group {row_group}");

            let viewed = reader
                .read_row_group(row_group, &projection, &["title"])
                .expect("titles as views");
            assert_eq!(viewed.schema().field(0).data_type(), &DataType::Utf8View);
            let titles = cast(viewed.column(0), &DataType::LargeUtf8).expect("cast back");
            assert_eq!(&titles, taken.column(0), "row group {row_group}");
            assert_eq!(viewed.column(1), taken.column(1), "row group {row_group}");
            start += read.num_rows() as u32;
        }
        assert_eq!(start as usize, SHUFFLED_ID_ROWS);

        assert!(matches!(
            reader.read_row_group(groups, &projection, &[]),
            Err(ReadError::Columnar(_))
        ));
        assert!(matches!(
            reader.read_row_group(0, &["nope"], &[]),
            Err(ReadError::UnknownColumn(_))
        ));
        let bare = reader.read_row_group(0, &[], &[]).expect("no column");
        assert_eq!(bare.num_columns(), 0);
        assert_eq!(bare.num_rows(), READ_ROW_GROUP_ROWS);
    }

    /// Rows per row group in the footer-check test, so the file has several.
    const VERIFY_ROW_GROUP_ROWS: usize = 1_250;

    /// The footer holds the sidecar to account: ids that still span the
    /// file's id range but sit in the wrong row group fail the row groups'
    /// `_id` statistics, and ids outside the range fail the range.
    #[test]
    fn verified_ids_hold_the_sidecar_to_the_footer() {
        let (bytes, ids) = build_shuffled_id_superfile_in_row_groups(VERIFY_ROW_GROUP_ROWS);
        let reader = SuperfileReader::open(bytes.clone()).expect("open");
        assert!(reader.parquet_metadata().num_row_groups() > 1);
        let loc = reader.id_sidecar.expect("the superfile carries a sidecar");
        let id_range = (
            *ids.iter().min().expect("rows"),
            *ids.iter().max().expect("rows"),
        );
        let verify = |ids: &[i128]| {
            let raw: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            IdSidecar {
                bytes: Bytes::from(ids::encode_packed(&raw)),
                packed: true,
                n_docs: ids.len() as u64,
            }
            .verified_ids(reader.parquet_metadata(), "doc_id", id_range)
        };
        assert_eq!(
            loc.slice(&bytes, SHUFFLED_ID_ROWS as u64)
                .verified_ids(reader.parquet_metadata(), "doc_id", id_range)
                .expect("the written sidecar passes"),
            ids
        );

        // Move the file's highest id into another row group: the span is
        // unchanged, but its own row group's maximum is not.
        let top = ids
            .iter()
            .position(|&id| id == id_range.1)
            .expect("max row");
        let other = match top / VERIFY_ROW_GROUP_ROWS {
            0 => ids.len() - 1,
            _ => 0,
        };
        let mut moved = ids.clone();
        moved.swap(top, other);
        assert!(matches!(verify(&moved), Err(ReadError::MalformedKv(_))));

        let mut outside = ids.clone();
        outside[0] = id_range.1 + 1;
        assert!(matches!(verify(&outside), Err(ReadError::MalformedKv(_))));

        // Without the footer's `_id` statistics, the range alone still
        // catches ids outside it.
        let raw: Vec<u8> = outside.iter().flat_map(|id| id.to_le_bytes()).collect();
        let no_stats = IdSidecar {
            bytes: Bytes::from(ids::encode_packed(&raw)),
            packed: true,
            n_docs: outside.len() as u64,
        }
        .verified_ids(
            reader.parquet_metadata(),
            "a column with no statistics",
            id_range,
        );
        assert!(matches!(no_stats, Err(ReadError::MalformedKv(_))));
    }

    /// A malformed sidecar location fails an eager open, as it always has, but
    /// only switches the sidecar off on a lazy open: the lazy path uses it as
    /// a shortcut with the Parquet column behind it. A superfile written before
    /// the sidecar existed has none either way.
    #[tokio::test]
    async fn a_malformed_sidecar_location_fails_eager_and_is_ignored_lazily() {
        let lazy =
            |bytes: Bytes| -> Arc<dyn LazyByteSource> { Arc::new(BytesLazyByteSource::new(bytes)) };
        let mut partial = required_kv();
        partial.push((kv::IDS_OFFSET.into(), "0".into()));
        let mut out_of_bounds = required_kv();
        out_of_bounds.push((kv::IDS_OFFSET.into(), u64::MAX.to_string()));
        out_of_bounds.push((kv::IDS_LENGTH.into(), "16".into()));
        for kvs in [partial, out_of_bounds] {
            let bytes = superfile_with_kv(&kvs);
            assert!(matches!(
                SuperfileReader::open(bytes.clone()),
                Err(ReadError::MalformedKv(_))
            ));
            let r = SuperfileReader::open_lazy(lazy(bytes))
                .await
                .expect("a lazy open ignores the location");
            assert!(r.fetch_id_sidecar().await.expect("fetch").is_none());
        }

        let pre_sidecar = superfile_with_kv(&required_kv());
        let r = SuperfileReader::open_lazy(lazy(pre_sidecar))
            .await
            .expect("open_lazy");
        assert!(r.fetch_id_sidecar().await.expect("fetch").is_none());
        assert!(!r.id_sidecar_is_packed());
    }

    /// The raw layout (one little-endian `i128` per doc) decodes by offset; a
    /// doc past the end is the caller's error, and a region shorter than its
    /// doc count is malformed, not misread.
    #[test]
    fn raw_id_sidecar_decodes_by_offset_and_rejects_a_short_region() {
        let ids: Vec<i128> = vec![7, -3, i128::MAX, 0];
        let n_docs = ids.len() as u64;
        let raw: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let sidecar = IdSidecar {
            bytes: Bytes::from(raw.clone()),
            packed: false,
            n_docs,
        };
        assert_eq!(
            sidecar.ids(&[2, 0, 2]).expect("decode"),
            vec![i128::MAX, 7, i128::MAX]
        );
        assert!(matches!(
            sidecar.ids(&[4]),
            Err(ReadError::DocIdOutOfRange { .. })
        ));

        let short = IdSidecar {
            bytes: Bytes::from(raw[..raw.len() - 1].to_vec()),
            packed: false,
            n_docs,
        };
        assert!(matches!(short.ids(&[0]), Err(ReadError::MalformedKv(_))));
    }

    #[tokio::test]
    async fn open_lazy_round_trips_search_and_blocks_eager_only_paths() {
        // Wrap the whole superfile in a lazy source so the lazy open path
        // runs; eager-only methods must then report LazyReaderUnsupported.
        let bytes = build_simple_fts_only_superfile();
        let src: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(bytes));
        let r = SuperfileReader::open_lazy(src).await.expect("open_lazy");
        assert_eq!(r.n_docs(), 4);
        // parquet_bytes is None on the lazy path.
        assert!(r.parquet_bytes().is_none());
        // FTS search still works (its source travels with the inner reader).
        let hits = r
            .bm25_hits_async("title", "rust", 5, BoolMode::Or)
            .await
            .expect("lazy search");
        assert!(!hits.is_empty());
        // Eager-only row materialization is rejected.
        let err = r
            .take_by_local_doc_ids(&[0], &["doc_id"])
            .expect_err("lazy take rejected");
        assert!(matches!(err, ReadError::LazyReaderUnsupported));
        let err = r.get_record_batch(None).expect_err("lazy batch rejected");
        assert!(matches!(err, ReadError::LazyReaderUnsupported));
        // byte_source returns the underlying lazy source (non-empty).
        assert!(r.byte_source().size() > 0);
    }

    #[test]
    fn slice_or_err_rejects_out_of_range() {
        let bytes = Bytes::from(vec![0u8; 32]);
        let err = slice_or_err(&bytes, 16, 64, "vector").expect_err("out of range");
        assert!(matches!(err, ReadError::MalformedKv(_)));
        // In-range slice succeeds and is zero-copy length-correct.
        let ok = slice_or_err(&bytes, 8, 8, "vector").expect("in range");
        assert_eq!(ok.len(), 8);
    }

    #[test]
    fn helper_present_predicates() {
        use crate::superfile::format::footer::KvMap;
        let mut map = KvMap::new();
        map.insert("a".to_string(), "1".to_string());
        assert!(all_present(&map, &["a"]));
        assert!(!all_present(&map, &["a", "b"]));
        assert!(any_present(&map, &["a", "b"]));
        assert!(!any_present(&map, &["x", "y"]));
        // parse_u64 round-trips a numeric value and errors on garbage.
        map.insert("n".to_string(), "42".to_string());
        assert_eq!(parse_u64(&map, "n").expect("parse"), 42);
        map.insert("bad".to_string(), "notnum".to_string());
        assert!(matches!(
            parse_u64(&map, "bad").expect_err("not a u64"),
            ReadError::MalformedKv(_)
        ));
        assert!(matches!(
            parse_u64(&map, "absent").expect_err("missing"),
            ReadError::MissingKv(_)
        ));
    }

    #[test]
    fn fts_bool_mode_reexport_is_bool_mode() {
        // The crate re-exports BoolMode as FtsBoolMode for convenience.
        let m: FtsBoolMode = FtsBoolMode::And;
        assert_eq!(m, BoolMode::And);
    }

    // ── KV-validation error branches (eager + lazy open paths) ─────────

    /// A complete superfile's worth of bytes whose footer carries only
    /// the caller-supplied `extra_kv`. Reuses the production parquet
    /// encode + splice path so the bytes are a real parquet file; the
    /// missing/garbled `inf.*` keys then drive the open-time validation
    /// branches. No FTS/vector blobs are spliced — every key under test
    /// is hand-supplied through `extra_kv`.
    fn superfile_with_kv(extra_kv: &[(String, String)]) -> Bytes {
        use parquet::basic::Compression;

        use crate::superfile::format::footer::{encode_parquet_body, splice_index_blobs};
        /// One row group; the body is tiny, so a single group is enough.
        const ROW_GROUP_SIZE: usize = 1024;
        let schema = schema_with_text();
        let ids = decimal128_ids(vec![1u64]);
        let title = LargeStringArray::from(vec!["x"]);
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(title)])
            .expect("build RecordBatch");
        let body = encode_parquet_body(&schema, &[batch], Compression::SNAPPY, ROW_GROUP_SIZE, &[])
            .expect("encode parquet body");
        // The splice writes region keys itself and drops a caller's, so the
        // malformed ones these tests need are forged onto the footer after.
        let (region, rest): (Vec<_>, Vec<_>) = extra_kv
            .iter()
            .cloned()
            .partition(|(k, _)| kv::REGION_KEYS.contains(&k.as_str()));
        let parts = splice_index_blobs(body, &[], &[], &[], &rest).expect("splice index blobs");
        let region: Vec<(&str, &str)> = region
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        with_forged_footer_kv(&parts.bytes, &region)
    }

    /// The five always-required KV entries, with a correct format value
    /// and a current-major version, so a superfile carrying exactly
    /// these passes required-key + version validation and exposes the
    /// remaining (FTS/vector) branches under test.
    fn required_kv() -> Vec<(String, String)> {
        vec![
            (kv::FORMAT.into(), kv::FORMAT_VALUE.into()),
            (kv::FORMAT_VERSION.into(), format::FORMAT_VERSION.into()),
            (kv::ID_COLUMN.into(), "doc_id".into()),
            (kv::N_DOCS.into(), "1".into()),
            (kv::BUILDER.into(), "test".into()),
        ]
    }

    #[test]
    fn open_with_rejects_format_value_mismatch() {
        // Required keys present, but inf.format carries the wrong sentinel.
        let mut kvs = required_kv();
        kvs[0] = (kv::FORMAT.into(), "not-a-superfile".into());
        let bytes = superfile_with_kv(&kvs);
        let err = SuperfileReader::open(bytes).expect_err("expected error");
        assert!(matches!(err, ReadError::MalformedKv(_)));
    }

    #[test]
    fn open_with_rejects_incompatible_version() {
        // A valid semver that parses but whose major differs from ours.
        let mut kvs = required_kv();
        kvs[1] = (kv::FORMAT_VERSION.into(), "9999.0.0".into());
        let bytes = superfile_with_kv(&kvs);
        let err = SuperfileReader::open(bytes).expect_err("expected error");
        assert!(matches!(err, ReadError::UnsupportedVersion(_)));
    }

    #[test]
    fn open_with_rejects_a_format_version_that_does_not_parse() {
        // Not a version at all: a malformed file, not one from another version.
        let mut kvs = required_kv();
        kvs[1] = (kv::FORMAT_VERSION.into(), "not-a-version".into());
        let bytes = superfile_with_kv(&kvs);
        let err = SuperfileReader::open(bytes).expect_err("expected error");
        assert!(
            matches!(&err, ReadError::Malformed(m) if m.contains("not-a-version")),
            "got {err:?}"
        );
    }

    #[test]
    fn open_with_rejects_partial_fts_keys() {
        // Only one of the three all-or-none inf.fts.* keys present.
        let mut kvs = required_kv();
        kvs.push((kv::FTS_OFFSET.into(), "0".into()));
        let bytes = superfile_with_kv(&kvs);
        let err = SuperfileReader::open(bytes).expect_err("expected error");
        assert!(matches!(err, ReadError::MalformedKv(_)));
    }

    #[test]
    fn open_with_rejects_partial_vec_keys() {
        // Only one of the three all-or-none inf.vec.* keys present.
        let mut kvs = required_kv();
        kvs.push((kv::VEC_OFFSET.into(), "0".into()));
        let bytes = superfile_with_kv(&kvs);
        let err = SuperfileReader::open(bytes).expect_err("expected error");
        assert!(matches!(err, ReadError::MalformedKv(_)));
    }

    #[test]
    fn open_rejects_partial_ids_keys() {
        // Only one of the all-or-none inf.ids.* sidecar keys present.
        let mut kvs = required_kv();
        kvs.push((kv::IDS_OFFSET.into(), "0".into()));
        let bytes = superfile_with_kv(&kvs);
        let err = SuperfileReader::open(bytes).expect_err("expected error");
        assert!(matches!(err, ReadError::MalformedKv(_)));
    }

    #[test]
    fn open_rejects_ids_sidecar_length_mismatch() {
        // Sidecar length that isn't 16 x n_docs (n_docs = 1 here) is rejected
        // rather than trusted into an out-of-range slice later.
        let mut kvs = required_kv();
        kvs.push((kv::IDS_OFFSET.into(), "0".into()));
        kvs.push((kv::IDS_LENGTH.into(), "32".into()));
        let bytes = superfile_with_kv(&kvs);
        let err = SuperfileReader::open(bytes).expect_err("expected error");
        assert!(matches!(err, ReadError::MalformedKv(_)));
    }

    #[test]
    fn open_rejects_ids_sidecar_out_of_bounds() {
        // Correct length (16 x 1) but an offset past EOF must be caught at open,
        // not panic on a later slice.
        let mut kvs = required_kv();
        kvs.push((kv::IDS_OFFSET.into(), u64::MAX.to_string()));
        kvs.push((kv::IDS_LENGTH.into(), "16".into()));
        let bytes = superfile_with_kv(&kvs);
        let err = SuperfileReader::open(bytes).expect_err("expected error");
        assert!(matches!(err, ReadError::MalformedKv(_)));
    }

    #[tokio::test]
    async fn open_lazy_rejects_malformed_kvs() {
        // Same crafted-KV bytes, wrapped in a lazy source, must exercise
        // the lazy open path's mirror validation branches.
        let lazy =
            |bytes: Bytes| -> Arc<dyn LazyByteSource> { Arc::new(BytesLazyByteSource::new(bytes)) };

        // Missing a required key (drop inf.n_docs) → MissingKv.
        let mut kvs = required_kv();
        kvs.retain(|(k, _)| k != kv::N_DOCS);
        let err = SuperfileReader::open_lazy(lazy(superfile_with_kv(&kvs)))
            .await
            .expect_err("missing required");
        assert!(matches!(err, ReadError::MissingKv(_)));

        // Wrong format sentinel → MalformedKv.
        let mut kvs = required_kv();
        kvs[0] = (kv::FORMAT.into(), "not-a-superfile".into());
        let err = SuperfileReader::open_lazy(lazy(superfile_with_kv(&kvs)))
            .await
            .expect_err("bad format");
        assert!(matches!(err, ReadError::MalformedKv(_)));

        // Incompatible version → UnsupportedVersion.
        let mut kvs = required_kv();
        kvs[1] = (kv::FORMAT_VERSION.into(), "9999.0.0".into());
        let err = SuperfileReader::open_lazy(lazy(superfile_with_kv(&kvs)))
            .await
            .expect_err("bad version");
        assert!(matches!(err, ReadError::UnsupportedVersion(_)));

        // Partial vec keys → MalformedKv (the lazy partial-vec branch).
        let mut kvs = required_kv();
        kvs.push((kv::VEC_OFFSET.into(), "0".into()));
        let err = SuperfileReader::open_lazy(lazy(superfile_with_kv(&kvs)))
            .await
            .expect_err("partial vec");
        assert!(matches!(err, ReadError::MalformedKv(_)));

        // Partial fts keys → MalformedKv (the lazy partial-fts branch).
        let mut kvs = required_kv();
        kvs.push((kv::FTS_OFFSET.into(), "0".into()));
        let err = SuperfileReader::open_lazy(lazy(superfile_with_kv(&kvs)))
            .await
            .expect_err("partial fts");
        assert!(matches!(err, ReadError::MalformedKv(_)));
    }

    #[tokio::test]
    async fn id_lookup_on_lazy_reader_is_unsupported() {
        // The id scan needs resident bytes; a lazy-opened reader must
        // surface an Io error rather than silently scanning nothing.
        let bytes = build_simple_fts_only_superfile();
        let src: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(bytes));
        let r = SuperfileReader::open_lazy(src).await.expect("open_lazy");
        let err = r.id_lookup_many(&[10]).expect_err("lazy id scan rejected");
        assert!(matches!(err, ReadError::Io(_)));
    }

    #[tokio::test]
    async fn exact_match_with_no_token_candidates_returns_empty() {
        // A value whose tokens are present in the dictionary but never
        // co-occur in one row drives the empty-candidates short-circuit:
        // "rust" (docs 0,2) AND "javascript" (doc 3) ⇒ no candidate row.
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        let hits = r
            .exact_match("title", "rust javascript")
            .await
            .expect("exact_match")
            .0;
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn exact_match_on_a_column_without_a_full_text_index_errors() {
        // `exact_match` prunes through the column's own term dictionary,
        // so a column with no full-text index — here the Decimal128
        // `doc_id` — is rejected by name rather than failing later on a
        // text downcast that says nothing about the real problem.
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        let err = r
            .exact_match("doc_id", "!!!")
            .await
            .expect_err("expected error");
        match err {
            ReadError::Fts(inner) => {
                let rendered = inner.to_string();
                assert!(
                    rendered.contains("doc_id"),
                    "error must name the column: {rendered}"
                );
            }
            other => panic!("expected ReadError::Fts, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn bm25_hits_on_a_column_without_a_full_text_index_errors() {
        // Same rule for the scored path: without an index there is no
        // analyzer to tokenize the query with, so it fails up front
        // rather than after a pass with some other column's analyzer.
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        let err = r
            .bm25_hits_async("doc_id", "anything", 10, BoolMode::Or)
            .await
            .expect_err("expected error");
        match err {
            ReadError::Fts(inner) => {
                let rendered = inner.to_string();
                assert!(
                    rendered.contains("doc_id"),
                    "error must name the column: {rendered}"
                );
            }
            other => panic!("expected ReadError::Fts, got {other:?}"),
        }
    }
}
