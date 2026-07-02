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

use std::{fmt, io, str, sync::Arc};

use arrow::compute::{concat_batches, take};
use arrow_array::{
    Array, ArrayRef, Decimal128Array, LargeStringArray, RecordBatch, RecordBatchReader, UInt32Array,
};
use arrow_schema::{Field, Schema};
use bytes::Bytes;
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{
            ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection,
            RowSelector,
        },
        parquet_to_arrow_schema,
    },
    file::metadata::{PageIndexPolicy, ParquetMetaData},
};
use rayon::ThreadPool;
use roaring::RoaringBitmap;

use crate::{
    superfile::{
        BytesLazyByteSource, LazyByteSource, LazySubSource, ReadError,
        format::{self, footer, kv},
        fts::{
            reader::{self as fts_reader, BoolMode, FtsReader},
            tokenize::{AsciiLowerTokenizer, Tokenizer},
        },
        vector::reader::{self as vector_reader, VectorReader},
    },
    supertable::query::provider::tombstone_access_plan,
};

/// Speculative Parquet-footer tail length for a lazy open. 64 KiB
/// covers a typical superfile footer (its `inf.*` KVs plus a single
/// row group's column metadata — a few KiB to a few tens of KiB) in
/// one range GET, so the cold open usually costs a single round-trip.
const DEFAULT_TAIL_SPECULATIVE_BYTES: u64 = 64 * 1024;

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
    vec: Option<VectorReader>,
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
    /// 3. **3 GETs** for the embedded FTS subsection, via
    ///    `FtsReader::open_lazy` (header, FST dictionary, doc-length
    ///    tail; postings stay lazy until search).
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

        let vec_fut = async {
            if !vec_present {
                return Ok::<_, ReadError>(None);
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

        Ok(Self {
            bytes: None,
            arrow_meta: None,
            parquet_meta,
            source: Some(source),
            schema,
            id_column,
            n_docs,
            fts,
            vec,
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

        // 3. Parse the Parquet metadata once, with the page index, and
        //    cache it on the reader. `Bytes` implements `ChunkReader`
        //    directly so this is zero-copy, and every later
        //    `take_by_local_doc_ids` reuses this `ArrowReaderMetadata`
        //    instead of re-parsing the footer per call. The page index
        //    lets `RowSelection` skip whole pages on targeted reads.
        let arrow_meta = ArrowReaderMetadata::load(
            &bytes,
            ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Optional),
        )
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

        // 5. Vector path mirrors FTS.
        let vec = if all_present(&kv_map, kv::VEC_KEYS) {
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
        } else if any_present(&kv_map, kv::VEC_KEYS) {
            return Err(ReadError::MalformedKv(
                "partial inf.vec.* keys present".into(),
            ));
        } else {
            None
        };

        Ok(Self {
            bytes: Some(bytes),
            parquet_meta: Arc::clone(arrow_meta.metadata()),
            arrow_meta: Some(arrow_meta),
            source: None,
            schema,
            id_column,
            n_docs,
            fts,
            vec,
        })
    }

    /// The Parquet footer metadata parsed at open — page index included
    /// on the eager path. One parse per reader lifetime; the SQL scan
    /// path serves row-group counts and DataFusion's parquet opener
    /// from this instead of re-reading the footer per query.
    pub fn parquet_metadata(&self) -> &Arc<ParquetMetaData> {
        &self.parquet_meta
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
    ///
    /// Returns `None` for readers opened via [`open_lazy`] — the
    /// lazy path does not materialize the full superfile,
    /// so external-Parquet pass-throughs need either the eager
    /// [`open`] path or an explicit `LazyByteSource::range(0, size)`
    /// against the source.
    ///
    /// [`open`]: SuperfileReader::open
    /// [`open_lazy`]: SuperfileReader::open_lazy
    pub fn parquet_bytes(&self) -> Option<&Bytes> {
        self.bytes.as_ref()
    }
    /// Returns a record batch containing all documents with all columns
    pub fn get_record_batch(
        &self,
        deleted_docs_bitmap: Option<Arc<RoaringBitmap>>,
    ) -> Result<RecordBatch, ReadError> {
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
        let reader = builder
            .build()
            .map_err(|e| ReadError::Columnar(e.to_string()))?;
        let read_schema = reader.schema();
        let batches = reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ReadError::Columnar(e.to_string()))?;
        let record_batch = concat_batches(&read_schema, &batches)
            .map_err(|e| ReadError::Columnar(e.to_string()))?;

        Ok(record_batch)
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
            (Some(bytes), _) => Arc::new(BytesLazyByteSource::new(bytes.clone())),
            (None, Some(src)) => Arc::clone(src),
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

        // 1. Resolve projected names → column indices (file order
        //    for the ProjectionMask, caller order for the output
        //    RecordBatch).
        let mut col_indices = Vec::with_capacity(projection.len());
        let mut out_fields: Vec<Field> = Vec::with_capacity(projection.len());
        for &name in projection {
            let idx = self
                .schema
                .index_of(name)
                .map_err(|_| ReadError::UnknownColumn(name.to_string()))?;
            col_indices.push(idx);
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

        // 4+5. Sorted/dedup'd ids → monotonic skip/select runs.
        //    local_doc_ids is dense parquet-row index (one parquet row
        //    per doc, in id order — invariant of the superfile body),
        //    so the selection lines up directly with parquet row
        //    offsets. Caller's original order (including duplicates)
        //    is restored below via the rank-back step.
        let (sorted_ids, selection) = row_selection_for_ids(local_doc_ids);

        // Metadata-cached read: reuse the `ArrowReaderMetadata` parsed
        // at open (no per-call footer parse), so this targeted read
        // only pays the projected-column page decode. The page index
        // (when present) lets `RowSelection` seek to the relevant
        // pages. CPU-bound over in-memory bytes — callers fan these
        // across `options.reader_pool` for cross-superfile parallelism.
        let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(bytes, arrow_meta);
        let mask = ProjectionMask::roots(builder.parquet_schema(), col_indices.iter().copied());
        let reader = builder
            .with_projection(mask)
            .with_row_selection(selection)
            .build()
            .map_err(|e| ReadError::Columnar(e.to_string()))?;
        let read_schema = reader.schema();
        let batches = reader
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ReadError::Columnar(e.to_string()))?;
        // `selected` has exactly sorted_ids.len() rows in
        // sorted_ids order (parquet honors the selection).
        let selected = concat_batches(&read_schema, &batches)
            .map_err(|e| ReadError::Columnar(e.to_string()))?;
        debug_assert_eq!(
            selected.num_rows(),
            sorted_ids.len(),
            "RowSelection rows ≠ requested distinct doc ids"
        );

        // 6. Rank back into the caller's order via take. Cheap:
        //    typical k is 10..1000.
        let indices = rank_back_indices(local_doc_ids, &sorted_ids);

        // 7. Gather columns in caller's projection order (the
        //    reader returns columns in file order, which may
        //    differ).
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(projection.len());
        for &name in projection {
            let idx = selected
                .schema()
                .index_of(name)
                .map_err(|_| ReadError::UnknownColumn(name.to_string()))?;
            let taken = take(selected.column(idx), &indices, None)
                .map_err(|e| ReadError::Columnar(e.to_string()))?;
            columns.push(taken);
        }
        RecordBatch::try_new(out_schema, columns).map_err(|e| ReadError::Columnar(e.to_string()))
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
    /// its bit. Rebuilds a `ParquetRecordBatchReader` each call
    /// (cold recovery path), rather than reusing the cached
    /// `arrow_meta` like `take_by_local_doc_ids` does.
    pub fn id_lookup(&self, target: i128) -> Result<Option<u32>, ReadError> {
        let bytes = self.bytes.clone().ok_or_else(|| {
            ReadError::Io(io::Error::other(
                "id_lookup requires an eager-opened superfile; this reader was opened via \
                 the lazy path and does not hold the full superfile bytes",
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
    /// hits ordered by descending score — this is the hit kernel, not a
    /// row-returning search; row materialization is `take_by_local_doc_ids`.
    ///
    /// ## Negation (`-term`)
    ///
    /// A `-`-prefixed term excludes every doc containing it, regardless
    /// of score; `mode` applies to the positive terms only. Example:
    /// `"rust -python"` scores docs with `rust`, dropping any that also
    /// contain `python`. A query with only negated terms is rejected
    /// (`FtsError::NegationOnly`).
    pub async fn bm25_hits_async(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
        let tok = AsciiLowerTokenizer;

        // Split the query into positive and negated terms. The parsed
        // tokens borrow `query`, so nothing is copied here.
        let parsed = tok.parse(query);
        let positives: Vec<&str> = parsed.positives.iter().map(|t| &**t).collect();
        let negatives: Vec<&str> = parsed.negatives.iter().map(|t| &**t).collect();
        self.bm25_search_pretokenized_excluding(column, &positives, &negatives, k, mode)
            .await
    }

    /// Pre-tokenized variant of [`Self::bm25_hits_async`] — the caller
    /// supplies the already-tokenized term slice and we skip the
    /// `AsciiLowerTokenizer` pass.
    ///
    /// Used by the supertable layer's fan-out: the cross-superfile
    /// search tokenizes the query once at the orchestrator (to
    /// compute the bloom-skip mask) and then passes the same
    /// `terms` slice to every per-superfile search, avoiding
    /// `(N+1)·T` redundant tokenizations across N superfiles and
    /// a T-token query.
    ///
    /// Terms must be already lower-cased ASCII alphanumeric tokens
    /// — the FST keys are stored in that form. Callers using the
    /// v1 tokenizer can produce them via
    /// `AsciiLowerTokenizer.tokenize(query)`.
    pub async fn bm25_search_pretokenized(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
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
    ) -> Result<Vec<(u32, f32)>, ReadError> {
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
    ) -> Result<Vec<u32>, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.token_match(column, tokens, mode).await?)
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
    ) -> Result<u64, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.token_match_count(column, tokens, mode).await?)
    }

    /// Document frequency of `token` in `column` (0 if absent) — a cheap
    /// header-only read used to estimate a predicate's match count
    /// before running `token_match`. Delegates to
    /// [`FtsReader::term_df`].
    pub async fn term_df(&self, column: &str, token: &str) -> Result<u64, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts.term_df(column, token).await?)
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
    pub async fn exact_match(&self, column: &str, value: &str) -> Result<Vec<u32>, ReadError> {
        // Pass 1 — candidate rows via the index: the term-AND of the
        // string's tokens (a superset of the exact matches).
        let tokens: Vec<String> = AsciiLowerTokenizer.tokenize(value).collect();
        let candidates: Vec<u32> = if tokens.is_empty() {
            // No tokens to prune with: every row is a candidate.
            (0..self.n_docs() as u32).collect()
        } else {
            let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
            self.token_match(column, &refs, BoolMode::And).await?
        };
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Pass 2 — verify raw-string equality on the decoded text.
        let batch = self.take_by_local_doc_ids(&candidates, &[column])?;
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
        Ok(out)
    }

    /// Pre-tokenized BM25 search with negated terms excluded — the
    /// negation sibling of [`Self::bm25_search_pretokenized`]. The
    /// supertable fan-out parses the `-` sigil once at the orchestrator
    /// and hands every superfile the split lists through here.
    pub(crate) async fn bm25_search_pretokenized_excluding(
        &self,
        column: &str,
        positives: &[&str],
        negatives: &[&str],
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
        let fts = self
            .fts()
            .ok_or_else(|| ReadError::MissingKv(kv::FTS_OFFSET))?;
        Ok(fts
            .search_excluding(column, positives, negatives, k, mode)
            .await?)
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
    pub async fn bm25_search_prefix(
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
        let term_bytes = fts.iter_terms_with_prefix(column, lowered.as_bytes())?;
        if term_bytes.is_empty() {
            return Ok(Vec::new());
        }
        // FST keys are valid UTF-8 by construction (AsciiLower
        // tokenizer only emits ASCII bytes); the from_utf8 below
        // is a typed pass-through, not a re-validation cost.
        let term_strings: Vec<&str> = term_bytes
            .iter()
            .filter_map(|b| str::from_utf8(b).ok())
            .collect();
        Ok(fts.search(column, &term_strings, k, BoolMode::Or).await?)
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
    ) -> Result<Vec<(u32, f32)>, ReadError> {
        self.bm25_search_or_range_pretokenized_with_floor(
            column,
            terms,
            k,
            doc_id_start,
            doc_id_end,
            f32::NEG_INFINITY,
        )
        .await
    }

    /// [`Self::bm25_search_or_range_pretokenized`] with a score floor —
    /// same contract as [`Self::bm25_search_pretokenized_with_floor`].
    pub async fn bm25_search_or_range_pretokenized_with_floor(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
        floor: f32,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
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
            )
            .await?)
    }

    /// Prefix-expanded BM25 search restricted to a doc_id sub-range.
    ///
    /// Same expansion logic as [`Self::bm25_search_prefix`] —
    /// AsciiLower the prefix, walk the FST for matching terms, run
    /// BM25 OR over the term set — but only docs in
    /// `[doc_id_start, doc_id_end)` are eligible. Used by the
    /// supertable layer's intra-superfile parallel fan-out on prefix
    /// queries; the per-sub-range expansion is identical (same FST,
    /// same column) so each sub-range expands locally rather than
    /// passing pre-expanded terms across the task boundary.
    pub async fn bm25_search_prefix_range(
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
        let term_bytes = fts.iter_terms_with_prefix(column, lowered.as_bytes())?;
        if term_bytes.is_empty() {
            return Ok(Vec::new());
        }
        let term_strings: Vec<&str> = term_bytes
            .iter()
            .filter_map(|b| str::from_utf8(b).ok())
            .collect();
        Ok(fts
            .search_or_range_pretokenized(column, &term_strings, k, doc_id_start, doc_id_end)
            .await?)
    }

    /// Multi-column BM25 search with per-column weights ("most
    /// fields" semantics: per-column scores summed by weight).
    pub async fn bm25_search_multi(
        &self,
        columns: &[(&str, f32)],
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
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
        self.vector_hits_filtered_async(column, query, k, options, None, None, None)
            .await
    }

    /// As [`Self::vector_hits_async`], but restricts the kNN ranking to
    /// the `local_doc_id`s in `allow` (a per-superfile predicate
    /// allow-set). The allow-set is applied *inside* the coarse 1-bit
    /// shortlist, so the returned top-k is the true k-nearest among
    /// matching rows — pushdown, not post-filter, with no underflow.
    /// `allow == None` is identical to [`Self::vector_hits_async`].
    ///
    /// `deny` (when `Some`) is a per-superfile tombstone set excluded
    /// before ranking on the unfiltered path, so a delete never shrinks
    /// the top-k. It is independent of `allow`; the filtered path passes
    /// `None` because its `allow` already has tombstones subtracted.
    pub async fn vector_hits_filtered_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        allow: Option<Arc<RoaringBitmap>>,
        deny: Option<Arc<RoaringBitmap>>,
        pool: Option<Arc<ThreadPool>>,
    ) -> Result<Vec<(u32, f32)>, ReadError> {
        let filtered = allow.is_some();
        let (nprobe, rerank_mult) = options.resolve(filtered);
        let v = self
            .vec()
            .ok_or_else(|| ReadError::MissingKv(kv::VEC_OFFSET))?;
        let rerank_mult = v.public_rerank_mult(column, rerank_mult);
        Ok(
            v.search_async(column, query, k, nprobe, rerank_mult, allow, deny, pool)
                .await?,
        )
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
        self.vector_search_clusters_filtered(column, query, k, clusters, options, None, None, None)
            .await
    }

    /// As [`Self::vector_search_clusters`], but restricts the kNN
    /// ranking to the `local_doc_id`s in `allow` (a per-superfile
    /// predicate allow-set), applied inside the coarse shortlist.
    /// `allow == None` is identical to [`Self::vector_search_clusters`].
    ///
    /// `deny` (when `Some`) is a per-superfile tombstone set excluded
    /// before ranking on the unfiltered path; see
    /// [`Self::vector_hits_filtered_async`].
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
    ) -> Result<Vec<(u32, f32)>, ReadError> {
        let filtered = allow.is_some();
        let (_, rerank_mult) = options.resolve(filtered);
        let v = self
            .vec()
            .ok_or_else(|| ReadError::MissingKv(kv::VEC_OFFSET))?;
        let rerank_mult = v.public_rerank_mult(column, rerank_mult);
        Ok(
            v.search_clusters_async(column, query, k, clusters, rerank_mult, allow, deny, pool)
                .await?,
        )
    }
}

/// Tuning knobs for vector search (`Supertable::vector_search`).
///
/// Each field is an optional override. `None` means “use the engine default”,
/// which depends on whether the query is filtered:
///
/// - unfiltered: `nprobe=6`, `rerank_mult=256`
/// - filtered (`filter: Some(...)` or an internal allow-set): `nprobe=8`, `rerank_mult=256`
///
/// Set a field with [`Self::with_nprobe`] / [`Self::with_rerank_mult`].
#[derive(Debug, Clone, Copy, Default)]
pub struct VectorSearchOptions {
    /// IVF probe override. `None` → engine default for the query path.
    pub nprobe: Option<usize>,
    rerank_mult: Option<usize>,
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arrow_array::{Decimal128Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field};

    use super::*;
    use crate::{
        superfile::{
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
            vector::distance::normalize,
        },
        test_helpers::{decimal128_ids, default_tokenizer, default_vector_config},
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
    fn id_lookup_returns_only_matching_ids() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let s = r.id_lookup(10).expect("should return result");
        assert_eq!(s.expect("should find id"), 0);
        let s = r.id_lookup(12).expect("should return result");
        assert_eq!(s.expect("should find id"), 2);

        let s = r.id_lookup(20).expect("should return result");
        assert!(s.is_none());
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

    #[tokio::test]
    async fn bm25_search_finds_matching_docs() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open superfile");
        let hits = r
            .bm25_hits_async("title", "rust", 5, BoolMode::Or)
            .await
            .expect("BM25 search");
        // docs 0 and 2 contain "rust"; both should appear.
        let doc_ids: HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(doc_ids.contains(&0));
        assert!(doc_ids.contains(&2));
    }

    #[tokio::test]
    async fn bm25_search_errors_when_no_fts() {
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
        let parts = splice_index_blobs(body, &[], &[], &[]).expect("splice index blobs");
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
            .await
            .expect("BM25 multi-column search");
        // Both doc 0 (title:rust) and doc 1 (body:rust) hit.
        let doc_ids: HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(doc_ids.contains(&0));
        assert!(doc_ids.contains(&1));
    }

    // ── Additional coverage ───────────────────────────────────────────

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
        let ids: HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
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
        let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
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
            .expect("and");
        assert_eq!(and, vec![0]);
        let or = r
            .token_match("title", &["rust", "runtime"], BoolMode::Or)
            .await
            .expect("or");
        assert_eq!(or, vec![0, 2]);
    }

    #[tokio::test]
    async fn term_df_counts_documents() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        assert_eq!(r.term_df("title", "rust").await.expect("df"), 2);
        assert_eq!(r.term_df("title", "absent").await.expect("df"), 0);
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
            .expect("exact_match");
        assert_eq!(hits, vec![0]);
        // No row stores just "rust", so the exact comparison yields none.
        let none = r.exact_match("title", "rust").await.expect("exact_match");
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn bm25_search_prefix_expands_terms() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        // "rust" is a prefix of "rust" (docs 0,2); "ru" expands to it too.
        let hits = r
            .bm25_search_prefix("title", "ru", 5)
            .await
            .expect("prefix search");
        let ids: HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
        // No indexed term begins with "zz".
        assert!(
            r.bm25_search_prefix("title", "zz", 5)
                .await
                .expect("prefix")
                .is_empty()
        );
        // k == 0 short-circuits.
        assert!(
            r.bm25_search_prefix("title", "ru", 0)
                .await
                .expect("zero k")
                .is_empty()
        );
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
        let ids: HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
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
            )
            .await
            .expect("floored ranged search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn bm25_search_prefix_range_restricts_window() {
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        // "ru" expands to "rust" (docs 0,2); restrict to [0,2) ⇒ doc 0.
        let hits = r
            .bm25_search_prefix_range("title", "ru", 10, 0, 2)
            .await
            .expect("prefix range");
        let ids: HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(ids.contains(&0));
        assert!(!ids.contains(&2));
        // Degenerate range / k short-circuits.
        assert!(
            r.bm25_search_prefix_range("title", "ru", 0, 0, 2)
                .await
                .expect("zero k")
                .is_empty()
        );
        assert!(
            r.bm25_search_prefix_range("title", "ru", 10, 2, 2)
                .await
                .expect("empty range")
                .is_empty()
        );
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
        let parts = splice_index_blobs(body, &[], &[], extra_kv).expect("splice index blobs");
        Bytes::from(parts.bytes)
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
        // id_lookup needs resident bytes; a lazy-opened reader must
        // surface an Io error rather than silently scanning nothing.
        let bytes = build_simple_fts_only_superfile();
        let src: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(bytes));
        let r = SuperfileReader::open_lazy(src).await.expect("open_lazy");
        let err = r.id_lookup(10).expect_err("lazy id_lookup rejected");
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
            .expect("exact_match");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn bm25_search_prefix_range_unmatched_prefix_is_empty() {
        // A valid (non-degenerate) range with a prefix no term begins
        // with drives the empty-term-expansion short-circuit.
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        let hits = r
            .bm25_search_prefix_range("title", "zz", 10, 0, 4)
            .await
            .expect("prefix range");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn exact_match_on_non_text_column_errors() {
        // `exact_match` only supports LargeUtf8 columns. Querying the
        // Decimal128 `doc_id` column drives the non-LargeUtf8 downcast
        // failure. The value tokenizes to nothing (punctuation only),
        // so every row becomes a candidate and the verify pass reaches
        // the downcast on the decimal column.
        let bytes = build_simple_fts_only_superfile();
        let r = SuperfileReader::open(bytes).expect("open");
        let err = r
            .exact_match("doc_id", "!!!")
            .await
            .expect_err("expected error");
        match err {
            ReadError::Io(e) => {
                assert!(e.to_string().contains("not LargeUtf8"));
            }
            other => panic!("expected ReadError::Io, got {:?}", other),
        }
    }
}
