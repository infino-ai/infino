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
use crate::superfile::lazy_source::{LazyByteSource, LazyByteSourceError};
use arrow::record_batch::RecordBatch;
use arrow_schema::Schema;
use parquet::arrow::{ArrowWriter, parquet_to_arrow_schema};
use parquet::basic::Compression;
use parquet::file::metadata::{
    FileMetaData, KeyValue, ParquetMetaDataBuilder, ParquetMetaDataReader, ParquetMetaDataWriter,
};
use parquet::file::properties::WriterProperties;
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
    /// Underlying [`LazyByteSource`] failure (network /
    /// storage / out-of-bounds) on the lazy-open footer
    /// read path. Distinct from `Parquet` so the lazy-open
    /// flow can surface storage failures without
    /// stringifying.
    #[error("lazy source: {0}")]
    Lazy(#[from] LazyByteSourceError),
}

/// Write `batches` (matching `schema`) as Parquet, then splice the
/// `fts_blob` and `vec_blob` between the last row group and a rewritten
/// footer carrying `extra_kv`. Either blob may be empty (length 0) — in
/// that case the corresponding offsets in the returned `ParquetParts`
/// are 0 and no `inf.fts.*` / `inf.vec.*` KV keys are written.
pub fn write_parquet_with_blobs(
    schema: &Arc<Schema>,
    batches: &[RecordBatch],
    fts_blob: &[u8],
    vec_blob: &[u8],
    extra_kv: &[(String, String)],
    compression: Compression,
    row_group_size: usize,
) -> Result<ParquetParts, FooterError> {
    // 1. Write Parquet to an in-memory buffer.
    let props = WriterProperties::builder()
        .set_compression(compression)
        .set_max_row_group_row_count(Some(row_group_size))
        .build();
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buf, schema.clone(), Some(props))?;
        for batch in batches {
            writer.write(batch)?;
        }
        writer.close()?;
    }

    // 2. Locate the footer and decode it.
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

    // 3. Truncate to end of last row group, append blobs.
    buf.truncate(footer_start);

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

    // 4. Patch KV metadata. Preserve everything parquet-rs put in the
    //    footer (Arrow schema KV among them); append `extra_kv` plus
    //    the `inf.fts.*` / `inf.vec.*` offsets. `FileMetaData` is
    //    immutable, so build a new one via the public ctor with the
    //    patched KV list and rebuild the `ParquetMetaData` around it,
    //    carrying row groups and column / offset indexes through
    //    unchanged.
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

    // 5. Re-encode footer. `ParquetMetaDataWriter::finish` appends
    //    the thrift-encoded metadata + the u32 length + the PAR1
    //    magic in one call, leaving `buf` ready to read as a
    //    complete Parquet file.
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

/// Lazy-open variant of [`read_kv_metadata`] + Arrow-schema
/// extraction. Issues two range GETs against `source`:
///
/// 1. Trailing 8 bytes (`[footer_len:u32 LE][PAR1]`) to learn
///    the footer length.
/// 2. The footer body itself.
///
/// Both the `inf.*` KV map *and* the Arrow schema come from
/// the same decoded `ParquetMetaData`, so returning them in
/// one call avoids re-decoding (or re-fetching) the footer
/// on the caller's behalf.
///
/// This is the polymorphism point for the lazy
/// [`crate::superfile::reader::SuperfileReader::open_lazy_with`]
/// flow: callers hand a [`LazyByteSource`] (e.g. an object-
/// store range-fetcher, a broadcast coalescer, or a mmap-
/// backed [`crate::superfile::lazy_source::BytesLazyByteSource`])
/// and pay only the footer cost up-front — not the full
/// segment.
///
/// Total bytes pulled: `8 + footer_len`, typically a few KB
/// at default Parquet writer settings.
pub async fn read_footer_lazy(
    source: &dyn LazyByteSource,
) -> Result<(KvMap, Arc<Schema>), FooterError> {
    let size = source.size();
    if size < 12 {
        return Err(FooterError::Malformed(
            "file smaller than minimum Parquet trailer",
        ));
    }

    // Step 1: trailing 8 bytes carry the footer length + PAR1 magic.
    let trailer = source.range(size - 8, 8).await?;
    if &trailer[4..8] != b"PAR1" {
        return Err(FooterError::Malformed("not a Parquet file (missing PAR1)"));
    }
    let footer_len_bytes: [u8; 4] = trailer[0..4]
        .try_into()
        .map_err(|_| FooterError::Malformed("footer length not 4 bytes"))?;
    let footer_len = u32::from_le_bytes(footer_len_bytes) as u64;
    if size < 8 + footer_len {
        return Err(FooterError::Malformed("footer length out of range"));
    }
    let footer_start = size - 8 - footer_len;

    // Step 2: pull the footer body (no PAR1, no length suffix —
    // `decode_metadata` wants just the thrift-encoded metadata).
    let footer_bytes = source.range(footer_start, footer_len).await?;
    let metadata = ParquetMetaDataReader::decode_metadata(&footer_bytes)?;

    // Extract KVs.
    let mut kvs: KvMap = HashMap::new();
    if let Some(entries) = metadata.file_metadata().key_value_metadata() {
        for entry in entries {
            if let Some(v) = &entry.value {
                kvs.insert(entry.key.clone(), v.clone());
            }
        }
    }

    // Extract Arrow schema. parquet-rs honours the
    // arrow-encoded schema KV (`ARROW:schema`) when present
    // and falls back to inference from the Parquet schema
    // descriptor — same semantics as
    // `ParquetRecordBatchReaderBuilder::try_new`'s schema(),
    // without needing the full file as a ChunkReader.
    let arrow_schema = parquet_to_arrow_schema(
        metadata.file_metadata().schema_descr(),
        metadata.file_metadata().key_value_metadata(),
    )?;
    Ok((kvs, Arc::new(arrow_schema)))
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

    #[test]
    fn write_with_no_blobs_produces_valid_parquet() {
        let schema = small_schema();
        let batch = small_batch(&schema);
        let parts =
            write_parquet_with_blobs(&schema, &[batch], &[], &[], &[], Compression::SNAPPY, 1024)
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
        let parts = write_parquet_with_blobs(
            &schema,
            &[batch],
            &fts_blob,
            &[],
            &[],
            Compression::SNAPPY,
            1024,
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
        let parts = write_parquet_with_blobs(
            &schema,
            &[batch],
            &[],
            &vec_blob,
            &[],
            Compression::SNAPPY,
            1024,
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
        let parts = write_parquet_with_blobs(
            &schema,
            &[batch],
            &fts_blob,
            &vec_blob,
            &[],
            Compression::SNAPPY,
            1024,
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
        let parts = write_parquet_with_blobs(
            &schema,
            &[batch],
            &[],
            &[],
            &extra,
            Compression::SNAPPY,
            1024,
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
        let parts =
            write_parquet_with_blobs(&schema, &[batch], &fts, &[], &[], Compression::SNAPPY, 1024)
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

    /// `read_footer_lazy` must agree with `read_kv_metadata` +
    /// `ParquetRecordBatchReaderBuilder` on both the KV map
    /// and the Arrow schema, while only ever touching the
    /// footer suffix of the file.
    #[tokio::test]
    async fn read_footer_lazy_matches_eager_kv_and_schema() {
        use crate::superfile::lazy_source::BytesLazyByteSource;
        use bytes::Bytes;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let schema = small_schema();
        let batch = small_batch(&schema);
        let extra = vec![
            ("inf.format".to_string(), "infino-superfile".to_string()),
            ("inf.format_version".to_string(), "1.0.0".to_string()),
            ("inf.id_column".to_string(), "id".to_string()),
            ("inf.fts.columns".to_string(), "[\"category\"]".to_string()),
        ];
        let parts = write_parquet_with_blobs(
            &schema,
            &[batch],
            &[],
            &[],
            &extra,
            Compression::SNAPPY,
            1024,
        )
        .expect("write parquet with blobs");

        let eager_kv = read_kv_metadata(&parts.bytes).expect("eager kv");
        let eager_schema =
            ParquetRecordBatchReaderBuilder::try_new(Bytes::from(parts.bytes.clone()))
                .expect("eager builder")
                .schema()
                .clone();

        let src = BytesLazyByteSource::new(Bytes::from(parts.bytes.clone()));
        let (lazy_kv, lazy_schema) = read_footer_lazy(&src).await.expect("lazy footer");

        assert_eq!(lazy_kv, eager_kv);
        assert_eq!(lazy_schema.fields(), eager_schema.fields());
    }

    /// `read_footer_lazy` must only fetch the footer suffix —
    /// not the whole file — even when the file is large and
    /// the bulk of bytes are in row groups. Asserted with a
    /// counting `LazyByteSource` wrapper.
    #[tokio::test]
    async fn read_footer_lazy_fetches_only_footer_suffix() {
        use crate::superfile::lazy_source::{
            BytesLazyByteSource, LazyByteSource, LazyByteSourceError,
        };
        use async_trait::async_trait;
        use bytes::Bytes;
        use std::sync::Mutex;

        struct Counting {
            inner: BytesLazyByteSource,
            ranges: Mutex<Vec<(u64, u64)>>,
        }
        #[async_trait]
        impl LazyByteSource for Counting {
            fn size(&self) -> u64 {
                self.inner.size()
            }
            async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
                self.ranges
                    .lock()
                    .expect("ranges mutex poisoned")
                    .push((start, len));
                self.inner.range(start, len).await
            }
        }

        let schema = small_schema();
        // Eight repeats of the small batch so the row-group region is
        // an order of magnitude larger than the footer.
        let batches: Vec<_> = (0..8).map(|_| small_batch(&schema)).collect();
        let parts = write_parquet_with_blobs(
            &schema,
            &batches,
            &[],
            &[],
            &[("inf.format".to_string(), "infino-superfile".to_string())],
            Compression::SNAPPY,
            1024,
        )
        .expect("write parquet");
        let total = parts.bytes.len() as u64;

        let src = Counting {
            inner: BytesLazyByteSource::new(Bytes::from(parts.bytes)),
            ranges: Mutex::new(Vec::new()),
        };
        let (_kv, _schema) = read_footer_lazy(&src).await.expect("lazy footer");

        let ranges = src.ranges.lock().expect("ranges mutex poisoned").clone();
        // Exactly two range GETs: the 8-byte trailer + the footer body.
        assert_eq!(ranges.len(), 2, "ranges: {ranges:?}");
        // First is the trailing 8 bytes.
        assert_eq!(ranges[0], (total - 8, 8));
        // Second is the footer body, which is a tail-aligned region.
        let (start, len) = ranges[1];
        assert_eq!(start + len, total - 8);
        // Total bytes pulled must be strictly less than the file
        // (proves we're not materializing the row groups).
        let pulled: u64 = ranges.iter().map(|(_, l)| *l).sum();
        assert!(pulled < total, "pulled {pulled} of {total}");
    }
}
