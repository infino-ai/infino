// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Parquet footer surgery: write the user's row groups via `parquet-rs`,
//! splice the FTS + vector blobs between the last row group and a
//! rewritten footer that carries `inf.*` KV metadata pointing at them.
//!
//! The splice is safe because Parquet column chunks are addressed by
//! absolute offsets inside the magic-bracketed row-group region (which
//! we don't touch), and Parquet readers ignore unknown
//! `key_value_metadata` keys.
//!
//! `WriterPropertiesBuilder::set_key_value_metadata` can't attach the
//! `inf.*` KVs up-front because their values (`inf.fts.offset`,
//! `inf.fts.length`, `inf.vec.offset`, `inf.vec.length`) are absolute
//! byte offsets, only known after row-group writes complete — hence
//! the post-hoc rewrite.
//!
//! Read + re-encode go through
//! `parquet::file::metadata::{ParquetMetaDataReader,
//! ParquetMetaDataBuilder, ParquetMetaDataWriter, FileMetaData,
//! KeyValue}`. `FileMetaData` is immutable, so patching the KV list
//! means constructing a new `FileMetaData` with the patched KVs and
//! rebuilding the `ParquetMetaData` around it.

use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufReader, Cursor, Read, Seek, SeekFrom, Write},
    sync::Arc,
};

use arrow::record_batch::RecordBatch;
use arrow_schema::Schema;
use bytes::Bytes;
use parquet::{
    arrow::ArrowWriter,
    basic::Compression,
    file::{
        metadata::{
            FileMetaData, KeyValue, ParquetMetaData, ParquetMetaDataBuilder, ParquetMetaDataReader,
            ParquetMetaDataWriter,
        },
        properties::WriterProperties,
    },
    schema::types::ColumnPath,
};
use tempfile::NamedTempFile;

use crate::superfile::{LazyByteSource, LazyByteSourceError, format::kv};

/// Length of Parquet's trailing `PAR1` magic, in bytes.
const PARQUET_MAGIC_LEN: usize = 4;

/// Width of the little-endian `u32` footer-length field that precedes
/// the trailing magic.
const PARQUET_FOOTER_LEN_FIELD_BYTES: usize = 4;

/// Bytes occupied by the Parquet footer suffix: the footer-length
/// field plus the trailing `PAR1` magic. The Thrift footer metadata
/// ends exactly this many bytes before end-of-file.
const PARQUET_FOOTER_SUFFIX_BYTES: usize = PARQUET_FOOTER_LEN_FIELD_BYTES + PARQUET_MAGIC_LEN;

/// Minimum plausible Parquet file size: a leading + trailing `PAR1`
/// magic with a footer-length field between them. Anything shorter
/// cannot carry a footer, so footer parsing bails early.
const PARQUET_MIN_FILE_BYTES: usize = PARQUET_MAGIC_LEN + PARQUET_FOOTER_SUFFIX_BYTES;

/// Output of a successful build.
pub struct ParquetParts {
    /// The complete superfile bytes (valid Parquet + embedded blobs).
    pub bytes: Vec<u8>,
    /// Absolute byte offset of the FTS blob within `bytes` (0 if absent).
    pub fts_offset: u64,
    /// Byte length of the FTS blob (0 if absent).
    pub fts_length: u64,
    /// Absolute byte offset of the vector blob within `bytes` (0 if absent).
    pub vec_offset: u64,
    /// Byte length of the vector blob (0 if absent).
    pub vec_length: u64,
    /// Absolute byte offset of the stable-id sidecar within `bytes` (0 if absent).
    pub ids_offset: u64,
    /// Byte length of the stable-id sidecar (0 if absent).
    pub ids_length: u64,
}

/// Absolute layout of a superfile written through
/// [`splice_index_streams_to`].
pub struct ParquetLayout {
    pub total_size: u64,
    pub fts_offset: u64,
    pub fts_length: u64,
    pub vec_offset: u64,
    pub vec_length: u64,
    pub ids_offset: u64,
    pub ids_length: u64,
}

struct CountingWriter<'a, W> {
    output: &'a mut W,
    written: u64,
}

impl<W: Write> Write for CountingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.output.write(buf)?;
        self.written = self.written.saturating_add(written as u64);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}

/// Map of `inf.*` KV-metadata entries extracted from the Parquet footer.
pub type KvMap = HashMap<String, String>;

/// Errors produced by footer surgery + KV reads.
#[derive(thiserror::Error, Debug)]
pub enum FooterError {
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("malformed parquet: {0}")]
    Malformed(&'static str),
    /// surfaces a `LazyByteSource` failure during
    /// async-tail footer reading. The string carries the upstream
    /// `LazyByteSourceError`'s `Display` so the chain is visible
    /// even after the layer translation.
    #[error("lazy source: {0}")]
    LazySource(String),
}

/// The encoded Parquet body — column chunks + row groups, with the
/// parquet-rs footer stripped — plus its decoded metadata. Output of
/// [`encode_parquet_body`], consumed by [`splice_index_blobs`].
///
/// This is the heavy, blob-independent half of a superfile build: it
/// holds everything that the CPU-bound column encode produced, ready for
/// the cheap blob splice + footer rewrite.
pub struct EncodedBody {
    /// Parquet body written to a scratch temp file and truncated to the end of
    /// the last row group (footer removed). Streamed into the superfile at
    /// splice time so the body — multi-GB at corpus scale — never materializes
    /// as a whole `Vec`. Blobs and the rewritten footer get appended after it.
    body_file: NamedTempFile,
    /// Byte length of the footer-stripped body in `body_file`.
    body_len: u64,
    /// Footer metadata decoded from the original write, carried through
    /// to the rewrite so row groups + column/offset indexes survive.
    metadata: ParquetMetaData,
}

/// Encode `batches` (matching `schema`) into the Parquet body and strip
/// the footer, returning an [`EncodedBody`] ready for blob splicing.
///
/// This is the CPU-heavy half of a superfile write (Arrow → Parquet
/// column encode + compression) and is **independent of the FTS/vector
/// blobs** — the blobs are only appended afterward by
/// [`splice_index_blobs`]. So a caller can run this as a rayon sibling
/// of index finalization (see `SuperfileBuilder::finish`).
///
/// Per-column data-page size limits (e.g. a small limit on the id
/// column) keep point lookups cheap: `RowSelection` + the offset index
/// seek to a tiny page and decompress just that, instead of a whole
/// row-group-sized page.
pub fn encode_parquet_body(
    schema: &Arc<Schema>,
    batches: &[RecordBatch],
    compression: Compression,
    row_group_size: usize,
    column_page_size_limits: &[(&str, usize)],
) -> Result<EncodedBody, FooterError> {
    let mut encoder =
        ParquetBodyEncoder::new(schema, compression, row_group_size, column_page_size_limits)?;
    for batch in batches {
        encoder.write_batch(batch)?;
    }
    encoder.finish()
}

/// Incremental Parquet-body encoder: feeds row batches to an `ArrowWriter`
/// one at a time so the caller can drop each batch after writing it, instead
/// of holding every batch until a single [`encode_parquet_body`] call. A
/// compaction merge that streams many inputs then never holds more than the
/// in-flight input batch plus the writer's current row-group buffer.
///
/// Because `ArrowWriter` cuts row groups purely on accumulated row count (not
/// on `write` boundaries), streaming batches yields byte-identical output to
/// one `encode_parquet_body` over the same batches in the same order.
pub(crate) struct ParquetBodyEncoder {
    body_file: NamedTempFile,
    writer: ArrowWriter<File>,
}

impl ParquetBodyEncoder {
    pub(crate) fn new(
        schema: &Arc<Schema>,
        compression: Compression,
        row_group_size: usize,
        column_page_size_limits: &[(&str, usize)],
    ) -> Result<Self, FooterError> {
        let mut props_builder = WriterProperties::builder()
            .set_compression(compression)
            .set_max_row_group_row_count(Some(row_group_size));
        for (col, limit) in column_page_size_limits {
            props_builder = props_builder
                .set_column_data_page_size_limit(ColumnPath::from((*col).to_string()), *limit);
        }
        let props = props_builder.build();

        // Encode to a scratch temp file rather than a `Vec`, so the body —
        // multi-GB at corpus scale — never materializes whole in RAM. The
        // writer owns an independent handle to the same file (`reopen`); the
        // struct's `body_file` handle reads the footer back after close, and
        // their cursors don't alias.
        let body_file = NamedTempFile::new()?;
        let write_handle = body_file.reopen()?;
        let writer = ArrowWriter::try_new(write_handle, schema.clone(), Some(props))?;
        Ok(Self { body_file, writer })
    }

    pub(crate) fn write_batch(&mut self, batch: &RecordBatch) -> Result<(), FooterError> {
        self.writer.write(batch)?;
        Ok(())
    }

    /// Close the writer and strip the Parquet footer, yielding the same
    /// [`EncodedBody`] as [`encode_parquet_body`].
    pub(crate) fn finish(self) -> Result<EncodedBody, FooterError> {
        let Self { body_file, writer } = self;
        writer.close()?;
        strip_parquet_footer(body_file)
    }
}

/// Read back the just-written Parquet footer from `body_file`, decode its
/// metadata, and truncate the file to the end of the last row group. The
/// splice later appends the blobs + a rewritten footer.
fn strip_parquet_footer(mut body_file: NamedTempFile) -> Result<EncodedBody, FooterError> {
    let file = body_file.as_file_mut();
    let n = file.seek(SeekFrom::End(0))?;
    if n < PARQUET_MIN_FILE_BYTES as u64 {
        return Err(FooterError::Malformed("parquet buffer too short"));
    }
    let mut suffix = [0u8; PARQUET_FOOTER_SUFFIX_BYTES];
    file.seek(SeekFrom::Start(n - PARQUET_FOOTER_SUFFIX_BYTES as u64))?;
    file.read_exact(&mut suffix)?;
    if &suffix[PARQUET_FOOTER_LEN_FIELD_BYTES..] != b"PAR1" {
        return Err(FooterError::Malformed("missing trailing PAR1 magic"));
    }
    let footer_len_bytes: [u8; PARQUET_FOOTER_LEN_FIELD_BYTES] = suffix
        [..PARQUET_FOOTER_LEN_FIELD_BYTES]
        .try_into()
        .map_err(|_| FooterError::Malformed("footer length not 4 bytes"))?;
    let footer_len = u32::from_le_bytes(footer_len_bytes) as u64;
    if n < PARQUET_FOOTER_SUFFIX_BYTES as u64 + footer_len {
        return Err(FooterError::Malformed("footer length out of range"));
    }
    let footer_start = n - PARQUET_FOOTER_SUFFIX_BYTES as u64 - footer_len;
    let mut footer_bytes = vec![0u8; footer_len as usize];
    file.seek(SeekFrom::Start(footer_start))?;
    file.read_exact(&mut footer_bytes)?;
    let metadata = ParquetMetaDataReader::decode_metadata(&footer_bytes)?;

    // Truncate to end of last row group; blobs + the rewritten footer
    // get appended by the splice.
    file.set_len(footer_start)?;

    Ok(EncodedBody {
        body_file,
        body_len: footer_start,
        metadata,
    })
}

/// Splice the `fts_blob` and `vec_blob` between an [`EncodedBody`]'s last
/// row group and a rewritten footer carrying `extra_kv` plus the blob
/// offset/length keys. Either blob may be empty (length 0) — in that case
/// the corresponding offsets in the returned `ParquetParts` are 0 and no
/// `inf.fts.*` / `inf.vec.*` KV keys are written.
///
/// Cheap relative to [`encode_parquet_body`]: byte appends + a footer
/// re-encode, no column work.
pub fn splice_index_blobs(
    body: EncodedBody,
    fts_blob: &[u8],
    vec_blob: &[u8],
    ids_blob: &[u8],
    extra_kv: &[(String, String)],
) -> Result<ParquetParts, FooterError> {
    let mut bytes = Vec::with_capacity(
        (body.body_len as usize)
            .saturating_add(fts_blob.len())
            .saturating_add(vec_blob.len())
            .saturating_add(ids_blob.len()),
    );
    let layout = splice_index_streams_to(
        body,
        Cursor::new(fts_blob),
        fts_blob.len() as u64,
        Cursor::new(vec_blob),
        vec_blob.len() as u64,
        Cursor::new(ids_blob),
        ids_blob.len() as u64,
        extra_kv,
        &mut bytes,
    )?;
    Ok(ParquetParts {
        bytes,
        fts_offset: layout.fts_offset,
        fts_length: layout.fts_length,
        vec_offset: layout.vec_offset,
        vec_length: layout.vec_length,
        ids_offset: layout.ids_offset,
        ids_length: layout.ids_length,
    })
}

/// Stream the stripped Parquet body, optional FTS/vector blobs, and rewritten
/// footer to `output`. This is the single superfile splice implementation:
/// the in-memory [`splice_index_blobs`] wrapper and drain's disk-backed shard
/// assembly both call here.
pub(crate) fn splice_index_streams_to<W, F, V, I>(
    body: EncodedBody,
    fts_blob: F,
    fts_length: u64,
    vec_blob: V,
    vec_length: u64,
    ids_blob: I,
    ids_length: u64,
    extra_kv: &[(String, String)],
    output: W,
) -> Result<ParquetLayout, FooterError>
where
    W: Write,
    F: Read,
    V: Read,
    I: Read,
{
    let EncodedBody {
        body_file,
        body_len,
        metadata,
    } = body;
    splice_carried_body_to(
        BufReader::new(body_file.reopen()?),
        body_len,
        metadata,
        fts_blob,
        fts_length,
        vec_blob,
        vec_length,
        ids_blob,
        ids_length,
        extra_kv,
        output,
    )
}

/// As [`splice_index_streams_to`], for a body that is already encoded —
/// carried verbatim from the superfile being replaced rather than produced
/// by this build.
///
/// The body lands at offset 0 in both files, so the row-group and column
/// offsets in `metadata` stay valid without adjustment.
pub(crate) fn splice_carried_body_to<W, B, F, V, I>(
    mut body: B,
    body_len: u64,
    metadata: ParquetMetaData,
    mut fts_blob: F,
    fts_length: u64,
    mut vec_blob: V,
    vec_length: u64,
    mut ids_blob: I,
    ids_length: u64,
    extra_kv: &[(String, String)],
    mut output: W,
) -> Result<ParquetLayout, FooterError>
where
    W: Write,
    B: Read,
    F: Read,
    V: Read,
    I: Read,
{
    let mut output = CountingWriter {
        output: &mut output,
        written: 0,
    };
    let body_copied = io::copy(&mut body, &mut output)?;
    if body_copied != body_len {
        return Err(FooterError::Malformed("body stream length mismatch"));
    }

    let fts_offset = if fts_length > 0 {
        let offset = output.written;
        let copied = io::copy(&mut fts_blob, &mut output)?;
        if copied != fts_length {
            return Err(FooterError::Malformed("FTS stream length mismatch"));
        }
        offset
    } else {
        0
    };
    let vec_offset = if vec_length > 0 {
        let offset = output.written;
        let copied = io::copy(&mut vec_blob, &mut output)?;
        if copied != vec_length {
            return Err(FooterError::Malformed("vector stream length mismatch"));
        }
        offset
    } else {
        0
    };
    let ids_offset = if ids_length > 0 {
        let offset = output.written;
        let copied = io::copy(&mut ids_blob, &mut output)?;
        if copied != ids_length {
            return Err(FooterError::Malformed("stable-id stream length mismatch"));
        }
        offset
    } else {
        0
    };

    // Patch KV metadata. Preserve everything parquet-rs put in the
    // footer (Arrow schema KV among them); append `extra_kv` plus the
    // `inf.fts.*` / `inf.vec.*` offsets. `FileMetaData` is immutable, so
    // build a new one via the public ctor with the patched KV list and
    // rebuild the `ParquetMetaData` around it, carrying row groups and
    // column / offset indexes through unchanged.
    let old_fm = metadata.file_metadata();
    let mut kvs = old_fm.key_value_metadata().cloned().unwrap_or_default();
    // `metadata` is the carried file's footer, so it already describes
    // that file's regions. The caller supplies a complete replacement set,
    // and keeping both would store each key twice: our reader takes the
    // last value, but a first-match reader resolves the stale offsets, and
    // the footer grows by a key set on every rewrite.
    kvs.retain(|entry| !entry.key.starts_with(kv::PREFIX));
    // Region keys are this splice's to write: a caller's copy describes
    // some other file's layout.
    for (k, v) in extra_kv
        .iter()
        .filter(|(k, _)| !kv::REGION_KEYS.contains(&k.as_str()))
    {
        kvs.push(KeyValue::new(k.clone(), Some(v.clone())));
    }
    if fts_length > 0 {
        kvs.push(KeyValue::new(
            kv::FTS_OFFSET.to_string(),
            Some(fts_offset.to_string()),
        ));
        kvs.push(KeyValue::new(
            kv::FTS_LENGTH.to_string(),
            Some(fts_length.to_string()),
        ));
    }
    if vec_length > 0 {
        kvs.push(KeyValue::new(
            kv::VEC_OFFSET.to_string(),
            Some(vec_offset.to_string()),
        ));
        kvs.push(KeyValue::new(
            kv::VEC_LENGTH.to_string(),
            Some(vec_length.to_string()),
        ));
    }
    if ids_length > 0 {
        kvs.push(KeyValue::new(
            kv::IDS_OFFSET.to_string(),
            Some(ids_offset.to_string()),
        ));
        kvs.push(KeyValue::new(
            kv::IDS_LENGTH.to_string(),
            Some(ids_length.to_string()),
        ));
    }
    let new_fm = FileMetaData::new(
        old_fm.version(),
        old_fm.num_rows(),
        old_fm.created_by().map(String::from),
        Some(kvs),
        old_fm.schema_descr_ptr(),
        old_fm.column_orders().cloned(),
    );
    let new_meta = ParquetMetaDataBuilder::new(new_fm)
        .set_row_groups(metadata.row_groups().to_vec())
        .set_column_index(metadata.column_index().cloned())
        .set_offset_index(metadata.offset_index().cloned())
        .build();

    // Re-encode footer. `ParquetMetaDataWriter::finish` appends the
    // thrift-encoded metadata + the u32 length + the PAR1 magic in one
    // call, leaving `output` ready to read as a complete Parquet file.
    ParquetMetaDataWriter::new(&mut output, &new_meta).finish()?;
    output.flush()?;

    Ok(ParquetLayout {
        total_size: output.written,
        fts_offset,
        fts_length,
        vec_offset,
        vec_length,
        ids_offset,
        ids_length,
    })
}

/// Read all `inf.*` (and any other) KV metadata entries from a
/// superfile's Parquet footer.
pub fn read_kv_metadata(bytes: &[u8]) -> Result<KvMap, FooterError> {
    let n = bytes.len();
    if n < PARQUET_MIN_FILE_BYTES || &bytes[n - PARQUET_MAGIC_LEN..n] != b"PAR1" {
        return Err(FooterError::Malformed("not a Parquet file (missing PAR1)"));
    }
    let footer_len_bytes: [u8; PARQUET_FOOTER_LEN_FIELD_BYTES] = bytes
        [n - PARQUET_FOOTER_SUFFIX_BYTES..n - PARQUET_MAGIC_LEN]
        .try_into()
        .map_err(|_| FooterError::Malformed("footer length not 4 bytes"))?;
    let footer_len = u32::from_le_bytes(footer_len_bytes) as usize;
    if n < PARQUET_FOOTER_SUFFIX_BYTES + footer_len {
        return Err(FooterError::Malformed("footer length out of range"));
    }
    let footer_start = n - PARQUET_FOOTER_SUFFIX_BYTES - footer_len;
    let metadata = ParquetMetaDataReader::decode_metadata(
        &bytes[footer_start..n - PARQUET_FOOTER_SUFFIX_BYTES],
    )?;
    extract_kv_map(&metadata)
}

/// read the decoded Parquet metadata from a
/// superfile footer via a [`LazyByteSource`], bounded to
/// **≤ 2 range GETs**:
///
/// 1. Speculative tail GET of `tail_speculative_bytes` (default
///    64 KiB). Covers the typical superfile footer in one
///    round-trip (Parquet footer is usually a few KiB to a few
///    tens of KiB).
/// 2. Follow-up GET only if the speculative tail didn't reach
///    the footer's start.
///
/// Returns the decoded `ParquetMetaData`; callers extract the
/// KV map via [`extract_kv_map`] or the Arrow schema via
/// `parquet::arrow::schema::parquet_to_arrow_schema`.
pub async fn read_parquet_metadata_lazy(
    source: &dyn LazyByteSource,
    tail_speculative_bytes: u64,
) -> Result<ParquetMetaData, FooterError> {
    // route the parquet tail fetch through
    // `LazyByteSource::tail`. On a `StorageRangeSource` with
    // unknown size this is a single suffix-range GET that
    // returns both the bytes AND the total object size, so
    // we skip the upfront HEAD round-trip cold-open used to
    // pay. On sources that already know their size (in-memory,
    // mmap, pre-HEAD'd `StorageRangeSource`), the default
    // `tail` impl reduces to `size() + range(size - n, n)`,
    // which is exactly what this function used to do
    // explicitly — so no extra work on the warm path.
    let (tail, total) = source
        .tail(tail_speculative_bytes)
        .await
        .map_err(footer_lazy_err)?;
    if total < PARQUET_MIN_FILE_BYTES as u64 {
        return Err(FooterError::Malformed("not a Parquet file (too short)"));
    }
    let spec_len = tail.len() as u64;
    let spec_start = total - spec_len;

    let n = tail.len();
    if &tail[n - PARQUET_MAGIC_LEN..n] != b"PAR1" {
        return Err(FooterError::Malformed("not a Parquet file (missing PAR1)"));
    }
    let footer_len_bytes: [u8; PARQUET_FOOTER_LEN_FIELD_BYTES] = tail
        [n - PARQUET_FOOTER_SUFFIX_BYTES..n - PARQUET_MAGIC_LEN]
        .try_into()
        .map_err(|_| FooterError::Malformed("footer length not 4 bytes"))?;
    let footer_len = u32::from_le_bytes(footer_len_bytes) as usize;
    let footer_end_abs = total - PARQUET_FOOTER_SUFFIX_BYTES as u64;
    let footer_start_abs = (total as usize)
        .checked_sub(PARQUET_FOOTER_SUFFIX_BYTES + footer_len)
        .ok_or(FooterError::Malformed("footer length out of range"))?;

    let footer_bytes: Bytes = if (footer_start_abs as u64) >= spec_start {
        let off_in_tail = footer_start_abs - (spec_start as usize);
        tail.slice(off_in_tail..off_in_tail + footer_len)
    } else {
        source
            .range(footer_start_abs as u64, footer_len as u64)
            .await
            .map_err(footer_lazy_err)?
    };
    debug_assert_eq!(footer_start_abs + footer_len, footer_end_abs as usize);

    ParquetMetaDataReader::decode_metadata(&footer_bytes).map_err(FooterError::from)
}

/// convenience wrapper around
/// [`read_parquet_metadata_lazy`] for the common case where
/// callers only need the `inf.*` KV map (e.g. tests + the eager
/// open path, mirroring the eager [`read_kv_metadata`]).
pub async fn read_kv_metadata_lazy(
    source: &dyn LazyByteSource,
    tail_speculative_bytes: u64,
) -> Result<KvMap, FooterError> {
    let metadata = read_parquet_metadata_lazy(source, tail_speculative_bytes).await?;
    extract_kv_map(&metadata)
}

/// Shared extractor — pulls every KV (key, value) pair out of
/// a decoded `ParquetMetaData` into a HashMap. Used by both the
/// eager and lazy footer-readers above.
pub fn extract_kv_map(metadata: &ParquetMetaData) -> Result<KvMap, FooterError> {
    let mut out: KvMap = HashMap::new();
    if let Some(kvs) = metadata.file_metadata().key_value_metadata() {
        for kv in kvs {
            if let Some(v) = &kv.value {
                out.insert(kv.key.clone(), v.clone());
            }
        }
    }
    Ok(out)
}

/// A blob's `(offset, length)` within its superfile.
pub(crate) type BlobRegion = (u64, u64);

/// Where a superfile's footer places each of its blobs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BlobRegions {
    pub(crate) fts: Option<BlobRegion>,
    pub(crate) vec: Option<BlobRegion>,
    pub(crate) ids: Option<BlobRegion>,
}

impl BlobRegions {
    /// Whether every region ends inside a `file_len`-byte file.
    pub(crate) fn fit_within(&self, file_len: u64) -> bool {
        [self.fts, self.vec, self.ids]
            .into_iter()
            .flatten()
            .all(|(at, len)| at.checked_add(len).is_some_and(|end| end <= file_len))
    }
}

/// Whether `metadata` stores any region key more than once.
///
/// Counts the key list as stored rather than through [`extract_kv_map`],
/// which keeps a key's last copy and so cannot see a stale duplicate.
pub(crate) fn has_duplicated_region_key(metadata: &ParquetMetaData) -> bool {
    let kvs = metadata
        .file_metadata()
        .key_value_metadata()
        .map_or(&[][..], Vec::as_slice);
    kv::REGION_KEYS
        .iter()
        .any(|key| kvs.iter().filter(|entry| entry.key == *key).count() > 1)
}

/// The blob regions this engine's reader resolves from `metadata`, which
/// keeps the last copy of a repeated key; `None` when a region key is not
/// a `u64` or has no partner.
pub(crate) fn resolved_regions(metadata: &ParquetMetaData) -> Option<BlobRegions> {
    let kv_map = extract_kv_map(metadata).ok()?;
    // Outer `None` is a malformed key; inner `None` an absent one.
    let value = |key: &str| -> Option<Option<u64>> {
        kv_map
            .get(key)
            .map_or(Some(None), |v| v.parse().ok().map(Some))
    };
    let region = |offset: &str, length: &str| -> Option<Option<BlobRegion>> {
        match (value(offset)?, value(length)?) {
            (Some(at), Some(len)) => Some(Some((at, len))),
            (None, None) => Some(None),
            _ => None,
        }
    };
    Some(BlobRegions {
        fts: region(kv::FTS_OFFSET, kv::FTS_LENGTH)?,
        vec: region(kv::VEC_OFFSET, kv::VEC_LENGTH)?,
        ids: region(kv::IDS_OFFSET, kv::IDS_LENGTH)?,
    })
}

/// Translate a `LazyByteSourceError` to a `FooterError` for the
/// async-tail readers. Storage failures become `Parquet`-shaped
/// errors via the existing `Malformed` channel — the variant
/// shape exists for both signal and source-chain preservation.
fn footer_lazy_err(e: LazyByteSourceError) -> FooterError {
    FooterError::LazySource(e.to_string())
}

/// Test-only: where `bytes`' footer starts, and the footer decoded.
#[cfg(test)]
fn split_footer(bytes: &[u8]) -> (usize, ParquetMetaData) {
    let n = bytes.len();
    let len_bytes: [u8; PARQUET_FOOTER_LEN_FIELD_BYTES] = bytes
        [n - PARQUET_FOOTER_SUFFIX_BYTES..n - PARQUET_MAGIC_LEN]
        .try_into()
        .expect("footer length field");
    let footer_len = u32::from_le_bytes(len_bytes) as usize;
    let start = n - PARQUET_FOOTER_SUFFIX_BYTES - footer_len;
    let metadata =
        ParquetMetaDataReader::decode_metadata(&bytes[start..n - PARQUET_FOOTER_SUFFIX_BYTES])
            .expect("decode footer");
    (start, metadata)
}

/// Test-only: `metadata` with `extra` appended to its key-value list.
#[cfg(test)]
pub(crate) fn with_appended_kv(
    metadata: &ParquetMetaData,
    extra: &[(&str, &str)],
) -> ParquetMetaData {
    let mut kvs = stored_kvs(metadata);
    kvs.extend(key_values(extra));
    with_kv_list(metadata, kvs)
}

/// Test-only: `bytes` with `extra` written ahead of its footer's own keys,
/// verbatim — where a carried rewrite left its stale copies, and the
/// malformed footers [`splice_carried_body_to`] refuses to write.
#[cfg(test)]
pub(crate) fn with_forged_footer_kv(bytes: &[u8], extra: &[(&str, &str)]) -> Bytes {
    let (start, metadata) = split_footer(bytes);
    let mut kvs = key_values(extra);
    kvs.extend(stored_kvs(&metadata));
    let mut out = bytes[..start].to_vec();
    ParquetMetaDataWriter::new(&mut out, &with_kv_list(&metadata, kvs))
        .finish()
        .expect("write forged footer");
    Bytes::from(out)
}

/// Test-only: the footer's key-value list as stored.
#[cfg(test)]
fn stored_kvs(metadata: &ParquetMetaData) -> Vec<KeyValue> {
    metadata
        .file_metadata()
        .key_value_metadata()
        .cloned()
        .unwrap_or_default()
}

/// Test-only: `pairs` as footer key-value entries.
#[cfg(test)]
fn key_values(pairs: &[(&str, &str)]) -> Vec<KeyValue> {
    pairs
        .iter()
        .map(|(k, v)| KeyValue::new(k.to_string(), Some(v.to_string())))
        .collect()
}

/// Test-only: `metadata` with its key-value list replaced by `kvs`.
#[cfg(test)]
fn with_kv_list(metadata: &ParquetMetaData, kvs: Vec<KeyValue>) -> ParquetMetaData {
    let fm = metadata.file_metadata();
    let fm = FileMetaData::new(
        fm.version(),
        fm.num_rows(),
        fm.created_by().map(String::from),
        Some(kvs),
        fm.schema_descr_ptr(),
        fm.column_orders().cloned(),
    );
    ParquetMetaDataBuilder::new(fm)
        .set_row_groups(metadata.row_groups().to_vec())
        .build()
}

#[cfg(test)]
mod tests {
    use arrow_array::{Float64Array, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field};

    use super::*;

    fn small_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("score", DataType::Float64, false),
            Field::new("category", DataType::Utf8, false),
        ]))
    }

    fn small_batch(schema: &Arc<Schema>) -> RecordBatch {
        let ids = UInt64Array::from(vec![0u64, 1, 2]);
        let scores = Float64Array::from(vec![1.1, 2.2, 3.3]);
        let cats = StringArray::from(vec!["a", "b", "a"]);
        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ids), Arc::new(scores), Arc::new(cats)],
        )
        .expect("build RecordBatch")
    }

    /// Test-only composition of the two production phases — encode the
    /// body then splice the blobs — exercising the exact path
    /// `SuperfileBuilder::finish` runs, minus the rayon parallelism.
    #[allow(clippy::too_many_arguments)]
    fn write_with_blobs(
        schema: &Arc<Schema>,
        batches: &[RecordBatch],
        fts_blob: &[u8],
        vec_blob: &[u8],
        extra_kv: &[(String, String)],
        compression: Compression,
        row_group_size: usize,
        column_page_size_limits: &[(&str, usize)],
    ) -> Result<ParquetParts, FooterError> {
        let body = encode_parquet_body(
            schema,
            batches,
            compression,
            row_group_size,
            column_page_size_limits,
        )?;
        // Sidecar exercised by builder/reader round-trip tests, not this
        // body+splice composition helper.
        splice_index_blobs(body, fts_blob, vec_blob, &[], extra_kv)
    }

    /// The footer's KV entries in the order they are stored, duplicates
    /// included — unlike [`read_kv_metadata`], which folds a repeated key
    /// onto its last value and so cannot see a duplicate at all.
    fn raw_footer_kv(bytes: &[u8]) -> Vec<(String, Option<String>)> {
        footer_metadata(bytes)
            .file_metadata()
            .key_value_metadata()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|kv| (kv.key, kv.value))
            .collect()
    }

    /// A carried rewrite must leave one `inf.*` set in the footer.
    ///
    /// The carried body brings the source's whole footer with it, so
    /// appending this build's keys without dropping the old ones stores
    /// each twice. Our reader takes the last value and is unaffected, but
    /// a first-match reader — pyarrow, DuckDB — resolves the *old*
    /// offsets and reads the previous file's regions, and the footer grows
    /// by a full key set on every rewrite.
    #[test]
    fn a_carried_rewrite_leaves_one_inf_key_set_in_the_footer() {
        const OLD_FTS: &[u8] = b"old fts blob";
        const NEW_FTS: &[u8] = b"a longer new fts blob";

        let schema = small_schema();
        let batch = small_batch(&schema);
        let original = write_with_blobs(
            &schema,
            &[batch],
            OLD_FTS,
            &[],
            &[(kv::N_DOCS.to_string(), "3".to_string())],
            Compression::SNAPPY,
            1024,
            &[],
        )
        .expect("write original");

        // Everything before the FTS blob is the Parquet body, which the
        // carried path reuses verbatim.
        let body = &original.bytes[..original.fts_offset as usize];
        let carried_meta = {
            let n = original.bytes.len();
            let len_bytes: [u8; PARQUET_FOOTER_LEN_FIELD_BYTES] = original.bytes
                [n - PARQUET_FOOTER_SUFFIX_BYTES..n - PARQUET_MAGIC_LEN]
                .try_into()
                .expect("footer length field");
            let footer_len = u32::from_le_bytes(len_bytes) as usize;
            let start = n - PARQUET_FOOTER_SUFFIX_BYTES - footer_len;
            ParquetMetaDataReader::decode_metadata(
                &original.bytes[start..n - PARQUET_FOOTER_SUFFIX_BYTES],
            )
            .expect("decode original footer")
        };

        let mut out = Vec::new();
        let layout = splice_carried_body_to(
            Cursor::new(body),
            body.len() as u64,
            carried_meta,
            Cursor::new(NEW_FTS),
            NEW_FTS.len() as u64,
            Cursor::new(&[][..]),
            0,
            Cursor::new(&[][..]),
            0,
            &[(kv::N_DOCS.to_string(), "3".to_string())],
            &mut out,
        )
        .expect("carried splice");

        let entries = raw_footer_kv(&out);
        for key in kv::ALL {
            let seen = entries.iter().filter(|(k, _)| k == key).count();
            assert!(
                seen <= 1,
                "{key} appears {seen} times; a first-match reader would take \
                 the carried file's value"
            );
        }
        let first_offset = entries
            .iter()
            .find(|(k, _)| k == kv::FTS_OFFSET)
            .and_then(|(_, v)| v.clone())
            .expect("fts offset present");
        assert_eq!(
            first_offset,
            layout.fts_offset.to_string(),
            "the first `inf.fts.offset` must be this rewrite's"
        );
    }

    /// The footer of `bytes`, decoded.
    fn footer_metadata(bytes: &[u8]) -> ParquetMetaData {
        split_footer(bytes).1
    }

    /// A superfile with an FTS and a vector blob, built with `extra_kv`.
    fn write_with_both_blobs(extra_kv: &[(String, String)]) -> ParquetParts {
        const FTS: &[u8] = b"fts blob";
        const VEC: &[u8] = b"a vector blob";
        let schema = small_schema();
        let batch = small_batch(&schema);
        write_with_blobs(
            &schema,
            &[batch],
            FTS,
            VEC,
            extra_kv,
            Compression::SNAPPY,
            1024,
            &[],
        )
        .expect("write superfile")
    }

    /// Where a blob sits is the splice's to say. A caller that hands in
    /// region keys — a rewrite carrying another file's `inf.vec.*` set —
    /// must not get them into the footer, where a first-match reader would
    /// take them over the splice's own.
    #[test]
    fn a_callers_region_keys_never_reach_the_footer() {
        const STALE: &str = "999999";
        let stale: Vec<(String, String)> = kv::REGION_KEYS
            .iter()
            .map(|k| (k.to_string(), STALE.to_string()))
            .collect();
        let parts = write_with_both_blobs(&stale);

        let entries = raw_footer_kv(&parts.bytes);
        for key in kv::REGION_KEYS {
            let values: Vec<_> = entries.iter().filter(|(k, _)| k == key).collect();
            assert!(values.len() <= 1, "{key} stored {} times", values.len());
            assert!(
                values.iter().all(|(_, v)| v.as_deref() != Some(STALE)),
                "{key} kept the caller's value"
            );
        }
        let metadata = footer_metadata(&parts.bytes);
        assert!(!has_duplicated_region_key(&metadata));
        let resolved = resolved_regions(&metadata).expect("well-formed");
        assert_eq!(resolved.fts, Some((parts.fts_offset, parts.fts_length)));
        assert_eq!(resolved.vec, Some((parts.vec_offset, parts.vec_length)));
        assert!(resolved.fit_within(parts.bytes.len() as u64));
    }

    /// A region key stored twice is found by counting the stored list:
    /// which copy a reader resolves depends on the reader.
    #[test]
    fn a_duplicated_region_key_is_counted_as_stored() {
        let parts = write_with_both_blobs(&[]);
        let metadata = footer_metadata(&parts.bytes);
        assert!(!has_duplicated_region_key(&metadata));

        let stale_offset = (parts.vec_offset + 1).to_string();
        let duplicated = with_appended_kv(&metadata, &[(kv::VEC_OFFSET, &stale_offset)]);
        assert!(has_duplicated_region_key(&duplicated));

        // This engine's reader still resolves one: the last copy.
        let resolved = resolved_regions(&duplicated).expect("last copy resolves");
        assert_eq!(resolved.vec, Some((parts.vec_offset + 1, parts.vec_length)));
    }

    /// An offset without its length, or a value that is not a number,
    /// resolves to no region a reader could use.
    #[test]
    fn a_partial_or_unparsable_region_does_not_resolve() {
        // Written without an ids sidecar, so `inf.ids.*` is free to plant.
        let metadata = footer_metadata(&write_with_both_blobs(&[]).bytes);
        let no_length = with_appended_kv(&metadata, &[(kv::IDS_OFFSET, "0")]);
        assert_eq!(resolved_regions(&no_length), None);

        let unparsable = with_appended_kv(
            &metadata,
            &[(kv::IDS_OFFSET, "0"), (kv::IDS_LENGTH, "not a number")],
        );
        assert_eq!(resolved_regions(&unparsable), None);
    }

    /// A region ending past the file does not fit it; one ending exactly at
    /// its end does.
    #[test]
    fn a_region_past_the_end_of_the_file_does_not_fit() {
        const FILE_LEN: u64 = 100;
        let at_end = BlobRegions {
            vec: Some((90, 10)),
            ..Default::default()
        };
        assert!(at_end.fit_within(FILE_LEN));
        let past_end = BlobRegions {
            vec: Some((90, 11)),
            ..Default::default()
        };
        assert!(!past_end.fit_within(FILE_LEN));
        let overflowing = BlobRegions {
            fts: Some((u64::MAX, 1)),
            ..Default::default()
        };
        assert!(!overflowing.fit_within(FILE_LEN));
    }

    #[test]
    fn write_with_no_blobs_produces_valid_parquet() {
        let schema = small_schema();
        let batch = small_batch(&schema);
        let parts = write_with_blobs(
            &schema,
            &[batch],
            &[],
            &[],
            &[],
            Compression::SNAPPY,
            1024,
            &[],
        )
        .expect("write should succeed");
        // PAR1 at start and end.
        assert_eq!(&parts.bytes[..4], b"PAR1");
        assert_eq!(&parts.bytes[parts.bytes.len() - 4..], b"PAR1");
        assert_eq!(parts.fts_length, 0);
        assert_eq!(parts.vec_length, 0);
    }

    #[test]
    fn write_with_fts_blob_records_offset_and_length() {
        let schema = small_schema();
        let batch = small_batch(&schema);
        let fts_blob = b"FTS_BLOB_BYTES_HERE".to_vec();
        let parts = write_with_blobs(
            &schema,
            &[batch],
            &fts_blob,
            &[],
            &[],
            Compression::SNAPPY,
            1024,
            &[],
        )
        .expect("write parquet with blobs");
        assert_eq!(parts.fts_length as usize, fts_blob.len());
        assert!(parts.fts_offset > 4); // after PAR1 magic
        // Verify the blob bytes at the recorded offset.
        let read_back = &parts.bytes
            [parts.fts_offset as usize..parts.fts_offset as usize + parts.fts_length as usize];
        assert_eq!(read_back, fts_blob.as_slice());
    }

    #[test]
    fn write_with_vec_blob_only_records_offset() {
        let schema = small_schema();
        let batch = small_batch(&schema);
        let vec_blob = vec![0xAAu8; 256];
        let parts = write_with_blobs(
            &schema,
            &[batch],
            &[],
            &vec_blob,
            &[],
            Compression::SNAPPY,
            1024,
            &[],
        )
        .expect("write parquet with blobs");
        assert_eq!(parts.fts_length, 0);
        assert_eq!(parts.vec_length as usize, vec_blob.len());
        assert!(parts.vec_offset > 0);
        let read_back = &parts.bytes
            [parts.vec_offset as usize..parts.vec_offset as usize + parts.vec_length as usize];
        assert_eq!(read_back, vec_blob.as_slice());
    }

    #[test]
    fn write_with_both_blobs_produces_distinct_offsets() {
        let schema = small_schema();
        let batch = small_batch(&schema);
        let fts_blob = vec![0x01u8; 100];
        let vec_blob = vec![0x02u8; 200];
        let parts = write_with_blobs(
            &schema,
            &[batch],
            &fts_blob,
            &vec_blob,
            &[],
            Compression::SNAPPY,
            1024,
            &[],
        )
        .expect("write parquet with blobs");
        assert!(parts.vec_offset > parts.fts_offset);
        assert_eq!(parts.fts_offset + parts.fts_length, parts.vec_offset);
    }

    #[test]
    fn read_kv_metadata_finds_extra_kv_entries() {
        let schema = small_schema();
        let batch = small_batch(&schema);
        let extra = vec![
            ("inf.format".to_string(), "infino-superfile".to_string()),
            ("inf.format_version".to_string(), "1.0.0".to_string()),
            ("inf.id_column".to_string(), "id".to_string()),
        ];
        let parts = write_with_blobs(
            &schema,
            &[batch],
            &[],
            &[],
            &extra,
            Compression::SNAPPY,
            1024,
            &[],
        )
        .expect("write parquet with blobs");
        let kv = read_kv_metadata(&parts.bytes).expect("read kv metadata");
        assert_eq!(
            kv.get("inf.format").map(String::as_str),
            Some("infino-superfile")
        );
        assert_eq!(
            kv.get("inf.format_version").map(String::as_str),
            Some("1.0.0")
        );
        assert_eq!(kv.get("inf.id_column").map(String::as_str), Some("id"));
    }

    #[test]
    fn read_kv_metadata_includes_fts_offsets_when_blob_present() {
        let schema = small_schema();
        let batch = small_batch(&schema);
        let fts = vec![0xCCu8; 64];
        let parts = write_with_blobs(
            &schema,
            &[batch],
            &fts,
            &[],
            &[],
            Compression::SNAPPY,
            1024,
            &[],
        )
        .expect("write parquet with blobs");
        let kv = read_kv_metadata(&parts.bytes).expect("read kv metadata");
        assert!(kv.contains_key("inf.fts.offset"));
        assert!(kv.contains_key("inf.fts.length"));
        assert!(!kv.contains_key("inf.vec.offset"));
    }

    #[test]
    fn read_kv_metadata_rejects_non_parquet_input() {
        let err = read_kv_metadata(&[0u8; 16]).expect_err("expected error");
        assert!(matches!(err, FooterError::Malformed(_)));
    }

    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };

    /// counting lazy source used by the
    /// `read_kv_metadata_lazy` tests below. Records every
    /// `range()` invocation so the test can assert the
    /// speculative-tail vs. follow-up GET budget.
    use crate::superfile::lazy_source::{BytesLazyByteSource, LazyByteSource, LazyByteSourceError};

    #[derive(Debug)]
    struct CountingFooterSource {
        inner: BytesLazyByteSource,
        async_calls: Arc<AtomicU64>,
    }

    impl CountingFooterSource {
        fn new(bytes: Bytes) -> Self {
            Self {
                inner: BytesLazyByteSource::new(bytes),
                async_calls: Arc::new(AtomicU64::new(0)),
            }
        }
        fn counter(&self) -> Arc<AtomicU64> {
            Arc::clone(&self.async_calls)
        }
    }

    #[async_trait::async_trait]
    impl LazyByteSource for CountingFooterSource {
        fn size(&self) -> u64 {
            self.inner.size()
        }
        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            self.async_calls.fetch_add(1, Ordering::Relaxed);
            self.inner.range(start, len).await
        }
        fn try_get_range_sync(&self, _start: u64, _len: u64) -> Option<Bytes> {
            None
        }
    }

    /// `read_kv_metadata_lazy` recovers the same
    /// KV map as the eager [`read_kv_metadata`] when the
    /// speculative tail (default 64 KiB) fully contains the
    /// Parquet footer. Range budget: **exactly 1 GET**.
    #[tokio::test]
    async fn lazy_kv_metadata_one_range_when_tail_covers_footer() {
        let schema = small_schema();
        let batch = small_batch(&schema);
        let extra = vec![
            ("inf.format".to_string(), "infino-superfile".to_string()),
            ("inf.format_version".to_string(), "1.0.0".to_string()),
            ("inf.id_column".to_string(), "id".to_string()),
        ];
        let parts = write_with_blobs(
            &schema,
            &[batch],
            &[],
            &[],
            &extra,
            Compression::SNAPPY,
            1024,
            &[],
        )
        .expect("write parquet with blobs");

        let eager = read_kv_metadata(&parts.bytes).expect("eager kv");
        let source = CountingFooterSource::new(Bytes::from(parts.bytes));
        let counter = source.counter();
        let lazy = read_kv_metadata_lazy(&source, 64 * 1024)
            .await
            .expect("lazy kv");

        assert_eq!(
            counter.load(Ordering::Relaxed),
            1,
            "tail wholly contains footer ⇒ exactly 1 range GET",
        );
        assert_eq!(eager, lazy, "lazy + eager KV maps must agree");
    }

    /// when the speculative tail is too small to
    /// cover the footer, `read_kv_metadata_lazy` issues a
    /// **second** GET for the footer body. Total range budget:
    /// **exactly 2 GETs**. KV map still matches the eager path.
    #[tokio::test]
    async fn lazy_kv_metadata_two_ranges_when_followup_needed() {
        let schema = small_schema();
        let batch = small_batch(&schema);
        let extra = vec![
            ("inf.format".to_string(), "infino-superfile".to_string()),
            ("inf.format_version".to_string(), "1.0.0".to_string()),
            ("inf.id_column".to_string(), "id".to_string()),
        ];
        let parts = write_with_blobs(
            &schema,
            &[batch],
            &[],
            &[],
            &extra,
            Compression::SNAPPY,
            1024,
            &[],
        )
        .expect("write parquet with blobs");
        let eager = read_kv_metadata(&parts.bytes).expect("eager kv");

        let source = CountingFooterSource::new(Bytes::from(parts.bytes));
        let counter = source.counter();
        // Force the speculative tail to be ridiculously small
        // (just the trailing 16 bytes — enough for `PAR1` +
        // footer-length, but never for the footer body).
        let lazy = read_kv_metadata_lazy(&source, 16)
            .await
            .expect("lazy kv (followup)");

        assert_eq!(
            counter.load(Ordering::Relaxed),
            2,
            "tail < footer ⇒ 1 tail GET + 1 follow-up GET",
        );
        assert_eq!(eager, lazy, "lazy + eager KV maps must agree");
    }
}
