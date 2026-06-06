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

use crate::superfile::format::kv;
use arrow::record_batch::RecordBatch;
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::metadata::{
    FileMetaData, KeyValue, ParquetMetaData, ParquetMetaDataBuilder, ParquetMetaDataReader,
    ParquetMetaDataWriter,
};
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;
use std::collections::HashMap;
use std::sync::Arc;

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
    /// Parquet bytes truncated to the end of the last row group (footer
    /// removed) — blobs and the rewritten footer get appended here.
    buf: Vec<u8>,
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
    let mut props_builder = WriterProperties::builder()
        .set_compression(compression)
        .set_max_row_group_row_count(Some(row_group_size));
    for (col, limit) in column_page_size_limits {
        props_builder = props_builder
            .set_column_data_page_size_limit(ColumnPath::from((*col).to_string()), *limit);
    }
    let props = props_builder.build();
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buf, schema.clone(), Some(props))?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.close()?;
    }

    // Locate the footer and decode it.
    let n = buf.len();
    if n < 12 {
        return Err(FooterError::Malformed("parquet buffer too short"));
    }
    if &buf[n - 4..n] != b"PAR1" {
        return Err(FooterError::Malformed("missing trailing PAR1 magic"));
    }
    let footer_len_bytes: [u8; 4] = buf[n - 8..n - 4]
        .try_into()
        .map_err(|_| FooterError::Malformed("footer length not 4 bytes"))?;
    let footer_len = u32::from_le_bytes(footer_len_bytes) as usize;
    if n < 8 + footer_len {
        return Err(FooterError::Malformed("footer length out of range"));
    }
    let footer_start = n - 8 - footer_len;
    let footer_bytes = buf[footer_start..n - 8].to_vec();
    let metadata = ParquetMetaDataReader::decode_metadata(&footer_bytes)?;

    // Truncate to end of last row group; blobs + the rewritten footer
    // get appended in `splice_index_blobs`.
    buf.truncate(footer_start);

    Ok(EncodedBody { buf, metadata })
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
    extra_kv: &[(String, String)],
) -> Result<ParquetParts, FooterError> {
    let EncodedBody { mut buf, metadata } = body;

    let fts_offset = if !fts_blob.is_empty() {
        let off = buf.len() as u64;
        buf.extend_from_slice(fts_blob);
        off
    } else {
        0
    };
    let vec_offset = if !vec_blob.is_empty() {
        let off = buf.len() as u64;
        buf.extend_from_slice(vec_blob);
        off
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
    for (k, v) in extra_kv {
        kvs.push(KeyValue::new(k.clone(), Some(v.clone())));
    }
    if !fts_blob.is_empty() {
        kvs.push(KeyValue::new(
            kv::FTS_OFFSET.to_string(),
            Some(fts_offset.to_string()),
        ));
        kvs.push(KeyValue::new(
            kv::FTS_LENGTH.to_string(),
            Some((fts_blob.len() as u64).to_string()),
        ));
    }
    if !vec_blob.is_empty() {
        kvs.push(KeyValue::new(
            kv::VEC_OFFSET.to_string(),
            Some(vec_offset.to_string()),
        ));
        kvs.push(KeyValue::new(
            kv::VEC_LENGTH.to_string(),
            Some((vec_blob.len() as u64).to_string()),
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
    // call, leaving `buf` ready to read as a complete Parquet file.
    ParquetMetaDataWriter::new(&mut buf, &new_meta).finish()?;

    Ok(ParquetParts {
        bytes: buf,
        fts_offset,
        fts_length: fts_blob.len() as u64,
        vec_offset,
        vec_length: vec_blob.len() as u64,
    })
}

/// Read all `inf.*` (and any other) KV metadata entries from a
/// superfile's Parquet footer.
pub fn read_kv_metadata(bytes: &[u8]) -> Result<KvMap, FooterError> {
    let n = bytes.len();
    if n < 12 || &bytes[n - 4..n] != b"PAR1" {
        return Err(FooterError::Malformed("not a Parquet file (missing PAR1)"));
    }
    let footer_len_bytes: [u8; 4] = bytes[n - 8..n - 4]
        .try_into()
        .map_err(|_| FooterError::Malformed("footer length not 4 bytes"))?;
    let footer_len = u32::from_le_bytes(footer_len_bytes) as usize;
    if n < 8 + footer_len {
        return Err(FooterError::Malformed("footer length out of range"));
    }
    let footer_start = n - 8 - footer_len;
    let metadata = ParquetMetaDataReader::decode_metadata(&bytes[footer_start..n - 8])?;
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
    source: &dyn crate::superfile::LazyByteSource,
    tail_speculative_bytes: u64,
) -> Result<parquet::file::metadata::ParquetMetaData, FooterError> {
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
    if total < 12 {
        return Err(FooterError::Malformed("not a Parquet file (too short)"));
    }
    let spec_len = tail.len() as u64;
    let spec_start = total - spec_len;

    let n = tail.len();
    if &tail[n - 4..n] != b"PAR1" {
        return Err(FooterError::Malformed("not a Parquet file (missing PAR1)"));
    }
    let footer_len_bytes: [u8; 4] = tail[n - 8..n - 4]
        .try_into()
        .map_err(|_| FooterError::Malformed("footer length not 4 bytes"))?;
    let footer_len = u32::from_le_bytes(footer_len_bytes) as usize;
    let footer_end_abs = total - 8;
    let footer_start_abs = (total as usize)
        .checked_sub(8 + footer_len)
        .ok_or(FooterError::Malformed("footer length out of range"))?;

    let footer_bytes: bytes::Bytes = if (footer_start_abs as u64) >= spec_start {
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
/// open path mirroring the legacy [`read_kv_metadata`]).
pub async fn read_kv_metadata_lazy(
    source: &dyn crate::superfile::LazyByteSource,
    tail_speculative_bytes: u64,
) -> Result<KvMap, FooterError> {
    let metadata = read_parquet_metadata_lazy(source, tail_speculative_bytes).await?;
    extract_kv_map(&metadata)
}

/// Shared extractor — pulls every KV (key, value) pair out of
/// a decoded `ParquetMetaData` into a HashMap. Used by both the
/// eager and lazy footer-readers above.
pub fn extract_kv_map(
    metadata: &parquet::file::metadata::ParquetMetaData,
) -> Result<KvMap, FooterError> {
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

/// Translate a `LazyByteSourceError` to a `FooterError` for the
/// async-tail readers. Storage failures become `Parquet`-shaped
/// errors via the existing `Malformed` channel — the variant
/// shape exists for both signal and source-chain preservation.
fn footer_lazy_err(e: crate::superfile::LazyByteSourceError) -> FooterError {
    FooterError::LazySource(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field};

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
        splice_index_blobs(body, fts_blob, vec_blob, extra_kv)
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

    /// counting lazy source used by the
    /// `read_kv_metadata_lazy` tests below. Records every
    /// `range()` invocation so the test can assert the
    /// speculative-tail vs. follow-up GET budget.
    use crate::superfile::lazy_source::{BytesLazyByteSource, LazyByteSource, LazyByteSourceError};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Debug)]
    struct CountingFooterSource {
        inner: BytesLazyByteSource,
        async_calls: Arc<AtomicU64>,
    }

    impl CountingFooterSource {
        fn new(bytes: bytes::Bytes) -> Self {
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
        async fn range(&self, start: u64, len: u64) -> Result<bytes::Bytes, LazyByteSourceError> {
            self.async_calls.fetch_add(1, Ordering::Relaxed);
            self.inner.range(start, len).await
        }
        fn try_get_range_sync(&self, _start: u64, _len: u64) -> Option<bytes::Bytes> {
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
        let source = CountingFooterSource::new(bytes::Bytes::from(parts.bytes));
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

        let source = CountingFooterSource::new(bytes::Bytes::from(parts.bytes));
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
