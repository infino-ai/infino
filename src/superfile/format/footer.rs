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
    mem,
    sync::Arc,
};

use arrow::{
    array::{Array, ArrayRef, AsArray, OffsetSizeTrait},
    compute::concat_batches,
    record_batch::RecordBatch,
};
use arrow_schema::{DataType, Schema, SchemaRef};
use bytes::Bytes;
use parquet::{
    arrow::{
        ArrowWriter,
        arrow_writer::{ArrowLeafColumn, ArrowRowGroupWriterFactory, compute_leaves},
    },
    basic::Compression,
    file::{
        metadata::{
            FileMetaData, KeyValue, ParquetMetaData, ParquetMetaDataBuilder, ParquetMetaDataReader,
            ParquetMetaDataWriter,
        },
        properties::{DEFAULT_WRITE_BATCH_SIZE, WriterProperties},
        writer::SerializedFileWriter,
    },
    schema::types::ColumnPath,
};
use rayon::prelude::*;
use tempfile::NamedTempFile;

use crate::{
    config::scratch_root,
    runtime_metrics::ingest::offset_window,
    superfile::{LazyByteSource, LazyByteSourceError, format::kv},
    utils::nulls::concat_sharing_nulls,
};

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

/// Most rows a data page holds. The page index prunes a selective read
/// down to whole pages, and every row of a kept page is decompressed and
/// decoded, so a lookup that matches one row pays for its whole page.
/// The writer's default (20 000 rows) makes a one-row read decode that
/// many; 4096 cuts it fivefold while keeping the page count, and with it
/// the header and page-index overhead a full scan pays, low.
const DATA_PAGE_ROW_LIMIT: usize = 4096;

/// Most string and binary bytes one column gets in a single write to the
/// Parquet writer. The writer checks a page's size only after each run of
/// [`DEFAULT_WRITE_BATCH_SIZE`] values (for a list column, a run ends at
/// the next row), so a page holds a whole run however large its values
/// are, and a page over 2 GiB can't be written (its header stores the
/// size as an `i32`). Writing a batch in slices of at most this many
/// bytes per column keeps a page near this size. A single row is never
/// split, so one row over 2 GiB still can't be written.
///
/// The cap only binds when values average over ~64 KB, so normal tables
/// write the same files as before. A smaller cap would also cut normal
/// text columns into many more, smaller pages.
const MAX_WRITE_BYTES: usize = 64 << 20;

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
    pub range_offset: u64,
    pub range_length: u64,
}

/// The range index region a splice writes after the id sidecar: its bytes
/// as a stream, the stream's length and the length of the directory at its
/// head. A zero `length` writes no region.
pub(crate) struct RangeStream<R> {
    pub(crate) region: R,
    pub(crate) length: u64,
    pub(crate) directory_length: u64,
}

impl RangeStream<io::Empty> {
    /// No region.
    pub(crate) fn none() -> Self {
        Self {
            region: io::empty(),
            length: 0,
            directory_length: 0,
        }
    }
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
    /// A `LazyByteSource` failure while reading the footer (its tail, or a
    /// range inside it), kept typed so a storage error under it stays
    /// reachable.
    #[error("lazy source: {0}")]
    LazySource(#[from] LazyByteSourceError),
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
    /// Footer metadata decoded from the original write (footer only, no page
    /// index), carried through to the rewrite so the row groups survive. The
    /// page index itself stays in the body, where the column chunks point.
    metadata: ParquetMetaData,
}

impl EncodedBody {
    /// The body's row groups to read back, before the splice: a fresh
    /// handle on the scratch file, which holds every column chunk at the
    /// offset the metadata records, and that metadata.
    pub(crate) fn rows_source(&self) -> io::Result<(File, Arc<ParquetMetaData>)> {
        Ok((self.body_file.reopen()?, Arc::new(self.metadata.clone())))
    }
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
/// Because row groups are cut purely on accumulated row count (not on
/// `write` boundaries), streaming batches yields byte-identical output to
/// one `encode_parquet_body` over the same batches in the same order.
///
/// It cuts and feeds row groups exactly as `ArrowWriter` would, but holds
/// each row group's slices until it is full, and full row groups until
/// [`ROW_GROUPS_ENCODED_TOGETHER`] of them are held, then encodes every
/// column of every held group in parallel. Every column writer sees the
/// same slices in the same order it would under `ArrowWriter`, and the
/// chunks are appended group by group in column order, so the bytes are
/// the same; only the encodes overlap. One group at a time, a body of few
/// wide columns — a title and its id — encoded on about one thread.
pub(crate) struct ParquetBodyEncoder {
    body_file: NamedTempFile,
    writer: SerializedFileWriter<File>,
    columns: ArrowRowGroupWriterFactory,
    schema: SchemaRef,
    row_group_size: usize,
    /// The current row group's slices, in arrival order.
    held: Vec<RecordBatch>,
    held_rows: usize,
    /// Full row groups not yet encoded, oldest first.
    full: Vec<Vec<RecordBatch>>,
    /// Whether held groups are encoded side by side on the pool, or one at
    /// a time on this thread (see [`BodyPlan::encode_groups_together`]).
    together: bool,
}

/// Full row groups the encoder holds before encoding them side by side,
/// one task per leaf column per group. The held groups are slices of the
/// caller's batches, so holding them costs no copy; the encoded chunks of
/// this many groups are what the encode holds beyond one group's.
const ROW_GROUPS_ENCODED_TOGETHER: usize = 4;

impl ParquetBodyEncoder {
    pub(crate) fn new(
        schema: &Arc<Schema>,
        compression: Compression,
        row_group_size: usize,
        column_page_size_limits: &[(&str, usize)],
    ) -> Result<Self, FooterError> {
        Self::with_plain_columns(
            schema,
            compression,
            row_group_size,
            column_page_size_limits,
            &[],
            true,
        )
    }

    /// [`Self::new`] with the dictionary turned off for `plain_columns`.
    fn with_plain_columns(
        schema: &Arc<Schema>,
        compression: Compression,
        row_group_size: usize,
        column_page_size_limits: &[(&str, usize)],
        plain_columns: &[String],
        together: bool,
    ) -> Result<Self, FooterError> {
        let props = writer_properties(
            compression,
            row_group_size,
            column_page_size_limits,
            plain_columns,
        );

        // Encode to a scratch temp file rather than a `Vec`, so the body —
        // multi-GB at corpus scale — never materializes whole in RAM. The
        // writer owns an independent handle to the same file (`reopen`); the
        // struct's `body_file` handle reads the footer back after close, and
        // their cursors don't alias.
        let scratch_root = scratch_root();
        let body_file = NamedTempFile::new_in(&scratch_root)?;
        let write_handle = body_file.reopen()?;
        // Built through `ArrowWriter` so the file starts, and its footer
        // carries the Arrow schema, exactly as before.
        let (writer, columns) = ArrowWriter::try_new(write_handle, schema.clone(), Some(props))?
            .into_serialized_writer()?;
        Ok(Self {
            body_file,
            writer,
            columns,
            schema: Arc::clone(schema),
            row_group_size: row_group_size.max(1),
            held: Vec::new(),
            held_rows: 0,
            full: Vec::new(),
            together,
        })
    }

    /// Add `batch`, cutting it where a row group fills as `ArrowWriter` does.
    pub(crate) fn write_batch(&mut self, batch: &RecordBatch) -> Result<(), FooterError> {
        let mut rest = batch.clone();
        while rest.num_rows() > 0 {
            let room = self.row_group_size - self.held_rows;
            let take = room.min(rest.num_rows());
            self.held.push(rest.slice(0, take));
            self.held_rows += take;
            rest = rest.slice(take, rest.num_rows() - take);
            if self.held_rows >= self.row_group_size {
                self.full.push(mem::take(&mut self.held));
                self.held_rows = 0;
                let held = match self.together {
                    true => ROW_GROUPS_ENCODED_TOGETHER,
                    false => 1,
                };
                if self.full.len() >= held {
                    self.flush_row_groups()?;
                }
            }
        }
        Ok(())
    }

    /// Encode the held row groups, one task per leaf column per group, and
    /// append them in order.
    fn flush_row_groups(&mut self) -> Result<(), FooterError> {
        let groups = mem::take(&mut self.full);
        if groups.is_empty() {
            return Ok(());
        }
        let first_index = self.writer.flushed_row_groups().len();
        // Each group's leaf-column writers and, per leaf column, its slices
        // in the order the writer would get them.
        let mut writers = Vec::with_capacity(groups.len());
        let mut leaves: Vec<Vec<Vec<ArrowLeafColumn>>> = Vec::with_capacity(groups.len());
        for (g, group) in groups.iter().enumerate() {
            let group_writers = self.columns.create_column_writers(first_index + g)?;
            let mut group_leaves: Vec<Vec<ArrowLeafColumn>> =
                group_writers.iter().map(|_| Vec::new()).collect();
            for batch in group {
                for part in byte_bounded_slices(batch, MAX_WRITE_BYTES) {
                    let mut leaf_index = 0;
                    for (field, column) in self.schema.fields().iter().zip(part.columns()) {
                        for leaf in compute_leaves(field.as_ref(), column)? {
                            group_leaves[leaf_index].push(leaf);
                            leaf_index += 1;
                        }
                    }
                }
            }
            writers.push(group_writers);
            leaves.push(group_leaves);
        }
        drop(groups);
        let mut tasks: Vec<_> = writers
            .iter_mut()
            .flatten()
            .zip(leaves.iter().flatten())
            .collect();
        match self.together {
            true => tasks.par_iter_mut().try_for_each(|(writer, leaves)| {
                leaves.iter().try_for_each(|leaf| writer.write(leaf))
            })?,
            false => tasks.iter_mut().try_for_each(|(writer, leaves)| {
                leaves.iter().try_for_each(|leaf| writer.write(leaf))
            })?,
        }
        for group_writers in writers {
            let mut row_group = self.writer.next_row_group()?;
            for writer in group_writers {
                writer.close()?.append_to_row_group(&mut row_group)?;
            }
            row_group.close()?;
        }
        Ok(())
    }

    /// Close the writer and strip the Parquet footer, yielding the same
    /// [`EncodedBody`] as [`encode_parquet_body`].
    pub(crate) fn finish(mut self) -> Result<EncodedBody, FooterError> {
        if !self.held.is_empty() {
            self.full.push(mem::take(&mut self.held));
            self.held_rows = 0;
        }
        self.flush_row_groups()?;
        let Self {
            body_file, writer, ..
        } = self;
        writer.close()?;
        strip_parquet_footer(body_file)
    }
}

/// Writer properties for a body: `compression` everywhere, cut every
/// `row_group_size` rows, pages of at most [`DATA_PAGE_ROW_LIMIT`] rows
/// within the given per-column page limits, and no dictionary for
/// `plain_columns`.
fn writer_properties(
    compression: Compression,
    row_group_size: usize,
    column_page_size_limits: &[(&str, usize)],
    plain_columns: &[String],
) -> WriterProperties {
    let mut props_builder = WriterProperties::builder()
        .set_compression(compression)
        .set_max_row_group_row_count(Some(row_group_size))
        .set_data_page_row_count_limit(DATA_PAGE_ROW_LIMIT);
    for (col, limit) in column_page_size_limits {
        props_builder = props_builder
            .set_column_data_page_size_limit(ColumnPath::from((*col).to_string()), *limit);
    }
    for col in plain_columns {
        props_builder =
            props_builder.set_column_dictionary_enabled(ColumnPath::from(col.clone()), false);
    }
    props_builder.build()
}

/// `batch` cut into row slices that each hold at most `max_bytes` of
/// string and binary values in every column, or a single row. A batch
/// under the cap comes back whole, so most writes are unchanged.
///
/// A slice of at least [`DEFAULT_WRITE_BATCH_SIZE`] rows is cut to a
/// multiple of it. Each write starts a new run of values in the Parquet
/// writer, and a page is cut only at the end of a run, so slices of whole
/// runs keep every column's pages within [`DATA_PAGE_ROW_LIMIT`] rows.
fn byte_bounded_slices(batch: &RecordBatch, max_bytes: usize) -> Vec<RecordBatch> {
    // A column under the cap is under it in every slice, so only the
    // others decide where to cut.
    let big: Vec<&ArrayRef> = batch
        .columns()
        .iter()
        .filter(|column| value_bytes(column.as_ref()) > max_bytes)
        .collect();
    if big.is_empty() {
        return vec![batch.clone()];
    }
    let fits = |start: usize, len: usize| {
        big.iter()
            .all(|column| value_bytes(column.slice(start, len).as_ref()) <= max_bytes)
    };
    let mut slices = Vec::new();
    let mut start = 0;
    while start < batch.num_rows() {
        // Most rows from `start` that fit, found by binary search; at least one.
        let (mut lo, mut hi) = (1, batch.num_rows() - start);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if fits(start, mid) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if lo < batch.num_rows() - start && lo >= DEFAULT_WRITE_BATCH_SIZE {
            lo -= lo % DEFAULT_WRITE_BATCH_SIZE;
        }
        slices.push(batch.slice(start, lo));
        start += lo;
    }
    slices
}

/// Bytes of the string and binary values in `array`'s rows, nested ones
/// included, plus fixed-size binary values. Fixed-width primitives count as
/// zero: a run of them makes a small page. Dictionary columns count as zero
/// too: keys share their values, so a slice has no byte range of its own,
/// and a dictionary of very large values is not capped.
fn value_bytes(array: &dyn Array) -> usize {
    match array.data_type() {
        DataType::Utf8 => window_len(array.as_string::<i32>().value_offsets()),
        DataType::LargeUtf8 => window_len(array.as_string::<i64>().value_offsets()),
        DataType::Binary => window_len(array.as_binary::<i32>().value_offsets()),
        DataType::LargeBinary => window_len(array.as_binary::<i64>().value_offsets()),
        DataType::Utf8View => view_bytes(array.as_string_view().lengths()),
        DataType::BinaryView => view_bytes(array.as_binary_view().lengths()),
        DataType::FixedSizeBinary(_) => {
            array.len() * array.as_fixed_size_binary().value_length() as usize
        }
        DataType::List(_) => {
            let list = array.as_list::<i32>();
            window_value_bytes(list.value_offsets(), list.values().as_ref())
        }
        DataType::LargeList(_) => {
            let list = array.as_list::<i64>();
            window_value_bytes(list.value_offsets(), list.values().as_ref())
        }
        DataType::Map(..) => {
            let map = array.as_map();
            window_value_bytes(map.value_offsets(), map.entries())
        }
        // A sliced fixed-size list already holds only its own values.
        DataType::FixedSizeList(..) => value_bytes(array.as_fixed_size_list().values().as_ref()),
        // Each field (and a map's key and value) is its own Parquet column
        // with its own pages, so only the largest one counts.
        DataType::Struct(_) => array
            .as_struct()
            .columns()
            .iter()
            .map(|child| value_bytes(child.as_ref()))
            .max()
            .unwrap_or(0),
        _ => 0,
    }
}

/// Bytes the range `offsets` covers.
fn window_len<O: OffsetSizeTrait>(offsets: &[O]) -> usize {
    let (first, last) = offset_window(offsets);
    last - first
}

/// [`value_bytes`] of the child `values` that `offsets` point at.
fn window_value_bytes<O: OffsetSizeTrait>(offsets: &[O], values: &dyn Array) -> usize {
    let (first, last) = offset_window(offsets);
    value_bytes(values.slice(first, last - first).as_ref())
}

/// Bytes of a view column's values, from their `lengths`.
fn view_bytes(lengths: impl Iterator<Item = u32>) -> usize {
    lengths.map(|len| len as usize).sum()
}

/// Compressed bytes of each top-level column of a body.
type ColumnSizes = HashMap<String, u64>;

/// How a body is written, and which of its columns may drop their
/// dictionary. [`BodyPlan::choose`] settles that from the first row group.
///
/// A dictionary pays off on a column that repeats values: each row costs a
/// few bits instead of a value. On a column of mostly distinct values the
/// dictionary page holds nearly every value once more and zstd compresses
/// the plain values about as well, so the column grows, and every read that
/// touches the group decodes that whole page first, point reads included.
/// The writer's default is a dictionary on every column, which falls back to
/// plain values only once a chunk's dictionary page fills: a column of
/// distinct values then still stores that full page, and still makes every
/// read of the group decode it.
///
/// No fixed rule tells the two apart (a timestamp column can go either way,
/// a long URL column usually wants its dictionary), so the choice is
/// measured on the body's own rows:
///
/// ```text
///   first row group ─┬─ encode as configured ──────┐
///                    └─ encode candidates plain ───┴─▶ per column: plain if smaller
/// ```
///
/// A column only ever moves to whichever encoding is smaller on the
/// sample, so the choice does not grow the body it was measured on.
#[derive(Debug, Clone)]
pub(crate) struct BodyPlan {
    /// The body's codec.
    pub(crate) compression: Compression,
    /// Rows per row group; also the rows the choice is measured on.
    pub(crate) row_group_size: usize,
    /// Flat top-level columns whose dictionary may be dropped.
    pub(crate) dictionary_candidates: Vec<String>,
    /// Per-column data-page size limits, as for [`encode_parquet_body`].
    pub(crate) column_page_size_limits: Vec<(String, usize)>,
    /// Encode several full row groups side by side on the pool, one task
    /// per leaf column per group, rather than one group at a time on the
    /// calling thread. Right for a build that has the pool to itself, such
    /// as a compaction merge; wrong for one of many builds already running
    /// side by side as pool tasks, where a wait inside the pool runs
    /// whatever else is queued — one such piece carried another's whole
    /// fifty-second build.
    pub(crate) encode_groups_together: bool,
}

impl BodyPlan {
    /// The candidates written without a dictionary, for a body whose first
    /// rows are `sample`.
    fn choose(
        &self,
        schema: &Arc<Schema>,
        sample: &RecordBatch,
    ) -> Result<Vec<String>, FooterError> {
        if self.dictionary_candidates.is_empty() {
            return Ok(Vec::new());
        }
        let dictionary = self.measure(schema, sample, &[])?;
        let plain = self.measure(schema, sample, &self.dictionary_candidates)?;
        Ok(plain_where_smaller(
            &self.dictionary_candidates,
            &dictionary,
            &plain,
        ))
    }

    fn page_size_limits(&self) -> Vec<(&str, usize)> {
        self.column_page_size_limits
            .iter()
            .map(|(c, l)| (c.as_str(), *l))
            .collect()
    }

    /// Compressed bytes of each top-level column when `sample` is encoded
    /// with no dictionary for `plain_columns`.
    fn measure(
        &self,
        schema: &Arc<Schema>,
        sample: &RecordBatch,
        plain_columns: &[String],
    ) -> Result<ColumnSizes, FooterError> {
        let props = writer_properties(
            self.compression,
            self.row_group_size,
            &self.page_size_limits(),
            plain_columns,
        );
        let mut writer = ArrowWriter::try_new(io::sink(), schema.clone(), Some(props))?;
        for part in byte_bounded_slices(sample, MAX_WRITE_BYTES) {
            writer.write(&part)?;
        }
        let meta = writer.close()?;
        let mut sizes = ColumnSizes::new();
        for rg in meta.row_groups() {
            for chunk in rg.columns() {
                let column = chunk
                    .column_path()
                    .parts()
                    .first()
                    .cloned()
                    .unwrap_or_default();
                *sizes.entry(column).or_default() +=
                    u64::try_from(chunk.compressed_size()).unwrap_or(0);
            }
        }
        Ok(sizes)
    }
}

/// The `candidates`, sorted, whose plain encoding (`plain`) is smaller than
/// with a dictionary (`dictionary`). A candidate either side did not
/// measure keeps its dictionary.
fn plain_where_smaller(
    candidates: &[String],
    dictionary: &ColumnSizes,
    plain: &ColumnSizes,
) -> Vec<String> {
    let mut plain_columns: Vec<String> = candidates
        .iter()
        .filter(|col| match (dictionary.get(*col), plain.get(*col)) {
            (Some(with), Some(without)) => without < with,
            _ => false,
        })
        .cloned()
        .collect();
    plain_columns.sort();
    plain_columns
}

/// [`ParquetBodyEncoder`] whose dictionaries are chosen by a [`BodyPlan`]
/// from the first rows written: it holds rows until it has one row group
/// (or the body ends), chooses, then writes them and streams the rest
/// straight through. The held rows are the row group the writer would
/// buffer anyway.
pub(crate) struct PlannedBodyEncoder {
    schema: Arc<Schema>,
    plan: BodyPlan,
    held: Vec<RecordBatch>,
    held_rows: usize,
    encoder: Option<ParquetBodyEncoder>,
}

impl PlannedBodyEncoder {
    pub(crate) fn new(schema: &Arc<Schema>, plan: BodyPlan) -> Self {
        Self {
            schema: Arc::clone(schema),
            plan,
            held: Vec::new(),
            held_rows: 0,
            encoder: None,
        }
    }

    pub(crate) fn write_batch(&mut self, batch: &RecordBatch) -> Result<(), FooterError> {
        if let Some(encoder) = self.encoder.as_mut() {
            return encoder.write_batch(batch);
        }
        self.held_rows += batch.num_rows();
        self.held.push(batch.clone());
        if self.held_rows >= self.plan.row_group_size {
            self.start()?;
        }
        Ok(())
    }

    /// Choose the dictionaries from the held rows, open the encoder, and
    /// write them.
    ///
    /// Only the sample (at most one row group) is copied; the held batches
    /// are written as they came, so a merge that feeds one large input
    /// batch never holds a second copy of it.
    fn start(&mut self) -> Result<(), FooterError> {
        let mut sample_parts = Vec::new();
        let mut wanted = self.plan.row_group_size;
        for batch in &self.held {
            if wanted == 0 {
                break;
            }
            let take = batch.num_rows().min(wanted);
            sample_parts.push(batch.slice(0, take));
            wanted -= take;
        }
        // Held batches of a wide table share one null array per type for
        // the columns they lack; joined as Arrow joins them, the sample
        // would hold a null array of a row group's rows for each of those.
        let sample = concat_sharing_nulls(&self.schema, &sample_parts)?;
        let plain_columns = self.plan.choose(&self.schema, &sample)?;
        let mut encoder = ParquetBodyEncoder::with_plain_columns(
            &self.schema,
            self.plan.compression,
            self.plan.row_group_size,
            &self.plan.page_size_limits(),
            &plain_columns,
            self.plan.encode_groups_together,
        )?;
        for batch in mem::take(&mut self.held) {
            encoder.write_batch(&batch)?;
        }
        self.encoder = Some(encoder);
        Ok(())
    }

    /// Close the body, yielding the same [`EncodedBody`] as
    /// [`ParquetBodyEncoder::finish`].
    pub(crate) fn finish(mut self) -> Result<EncodedBody, FooterError> {
        if self.encoder.is_none() {
            self.start()?;
        }
        self.encoder.expect("start opened the encoder").finish()
    }
}

/// [`encode_parquet_body`] with the dictionaries chosen by `plan`.
pub(crate) fn encode_parquet_body_planned(
    schema: &Arc<Schema>,
    batches: &[RecordBatch],
    plan: BodyPlan,
) -> Result<EncodedBody, FooterError> {
    let mut encoder = PlannedBodyEncoder::new(schema, plan);
    for batch in batches {
        encoder.write_batch(batch)?;
    }
    encoder.finish()
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
        RangeStream::none(),
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
pub(crate) fn splice_index_streams_to<W, F, V, I, R>(
    body: EncodedBody,
    fts_blob: F,
    fts_length: u64,
    vec_blob: V,
    vec_length: u64,
    ids_blob: I,
    ids_length: u64,
    range: RangeStream<R>,
    extra_kv: &[(String, String)],
    output: W,
) -> Result<ParquetLayout, FooterError>
where
    W: Write,
    F: Read,
    V: Read,
    I: Read,
    R: Read,
{
    let EncodedBody {
        body_file,
        body_len,
        metadata,
    } = body;
    splice_carried_body_to(
        BufReader::new(body_file.reopen()?),
        body_len,
        &metadata,
        fts_blob,
        fts_length,
        vec_blob,
        vec_length,
        ids_blob,
        ids_length,
        range,
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
///
/// The body already holds the page index (it ends where the old footer
/// began), and the column chunks in `metadata` point at it. So the new
/// footer is written from the row groups alone, whether or not `metadata`
/// had the page index loaded: writing a loaded one again would append a
/// second copy and repoint the footer at it.
pub(crate) fn splice_carried_body_to<W, B, F, V, I, R>(
    mut body: B,
    body_len: u64,
    metadata: &ParquetMetaData,
    mut fts_blob: F,
    fts_length: u64,
    mut vec_blob: V,
    vec_length: u64,
    mut ids_blob: I,
    ids_length: u64,
    mut range: RangeStream<R>,
    extra_kv: &[(String, String)],
    mut output: W,
) -> Result<ParquetLayout, FooterError>
where
    W: Write,
    B: Read,
    F: Read,
    V: Read,
    I: Read,
    R: Read,
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
    let range_offset = if range.length > 0 {
        let offset = output.written;
        let copied = io::copy(&mut range.region, &mut output)?;
        if copied != range.length {
            return Err(FooterError::Malformed("range index stream length mismatch"));
        }
        offset
    } else {
        0
    };

    // Patch KV metadata. Preserve everything parquet-rs put in the
    // footer (Arrow schema KV among them); append `extra_kv` plus the
    // `inf.fts.*` / `inf.vec.*` offsets. `FileMetaData` is immutable, so
    // build a new one via the public ctor with the patched KV list and
    // rebuild the `ParquetMetaData` around it with the carried row groups.
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
    if range.length > 0 {
        for (key, value) in [
            (kv::RANGE_OFFSET, range_offset),
            (kv::RANGE_LENGTH, range.length),
            (kv::RANGE_DIRECTORY_LENGTH, range.directory_length),
        ] {
            kvs.push(KeyValue::new(key.to_string(), Some(value.to_string())));
        }
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
        range_offset,
        range_length: range.length,
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
    let (tail, total) = source.tail(tail_speculative_bytes).await?;
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
            .await?
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
    pub(crate) range: Option<BlobRegion>,
}

impl BlobRegions {
    /// Whether every region ends inside a `file_len`-byte file.
    pub(crate) fn fit_within(&self, file_len: u64) -> bool {
        [self.fts, self.vec, self.ids, self.range]
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
        range: region(kv::RANGE_OFFSET, kv::RANGE_LENGTH)?,
    })
}

/// Test-only: where `bytes`' footer starts, and the footer decoded.
#[cfg(any(test, feature = "test-helpers"))]
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

#[cfg(any(test, feature = "test-helpers"))]
test_visible! {
    /// Test-only: `bytes` with `extra` written ahead of its footer's own keys,
    /// verbatim — where a carried rewrite left its stale copies, and the
    /// malformed footers [`splice_carried_body_to`] refuses to write.
    fn with_forged_footer_kv(bytes: &[u8], extra: &[(&str, &str)]) -> Bytes {
        let (start, metadata) = split_footer(bytes);
        let mut kvs = key_values(extra);
        kvs.extend(stored_kvs(&metadata));
        let mut out = bytes[..start].to_vec();
        ParquetMetaDataWriter::new(&mut out, &with_kv_list(&metadata, kvs))
            .finish()
            .expect("write forged footer");
        Bytes::from(out)
    }
}

/// Test-only: the footer's key-value list as stored.
#[cfg(any(test, feature = "test-helpers"))]
fn stored_kvs(metadata: &ParquetMetaData) -> Vec<KeyValue> {
    metadata
        .file_metadata()
        .key_value_metadata()
        .cloned()
        .unwrap_or_default()
}

/// Test-only: `pairs` as footer key-value entries.
#[cfg(any(test, feature = "test-helpers"))]
fn key_values(pairs: &[(&str, &str)]) -> Vec<KeyValue> {
    pairs
        .iter()
        .map(|(k, v)| KeyValue::new(k.to_string(), Some(v.to_string())))
        .collect()
}

/// Test-only: `metadata` with its key-value list replaced by `kvs`.
#[cfg(any(test, feature = "test-helpers"))]
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
    use std::slice;

    use arrow::compute::concat_batches;
    use arrow_array::{
        BinaryViewArray, FixedSizeBinaryArray, FixedSizeListArray, Float64Array, LargeStringArray,
        StringArray, StringViewArray, StructArray, UInt64Array,
        builder::{ListBuilder, MapBuilder, StringBuilder},
    };
    use arrow_schema::{DataType, Field};
    use parquet::{
        arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
        basic::ZstdLevel,
        file::{
            metadata::PageIndexPolicy,
            reader::{FileReader, SerializedFileReader},
        },
    };

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
            &carried_meta,
            Cursor::new(NEW_FTS),
            NEW_FTS.len() as u64,
            Cursor::new(&[][..]),
            0,
            Cursor::new(&[][..]),
            0,
            RangeStream::none(),
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

    // ── Dictionary choice ───────────────────────────────────────────────

    /// Rows per row group in the dictionary tests.
    const TEST_GROUP_ROWS: usize = 4_096;

    /// Odd 64-bit multiplier that scatters `0..n` into distinct,
    /// unordered values.
    const SCATTER: u64 = 0x9E37_79B9_7F4A_7C15;

    /// A second odd multiplier, so the strings below share no structure
    /// with the scattered integers.
    const SCATTER_2: u64 = 0xD6E8_FEB8_6659_FD93;

    fn plan(schema: &Schema) -> BodyPlan {
        BodyPlan {
            compression: Compression::ZSTD(ZstdLevel::try_new(3).expect("valid zstd level")),
            row_group_size: TEST_GROUP_ROWS,
            dictionary_candidates: schema.fields().iter().map(|f| f.name().clone()).collect(),
            column_page_size_limits: Vec::new(),
            encode_groups_together: true,
        }
    }

    fn sizes(entries: &[(&str, u64)]) -> ColumnSizes {
        entries.iter().map(|(c, b)| (c.to_string(), *b)).collect()
    }

    /// A column of distinct values next to one of four long strings in
    /// random order: the first is smaller plain, the second with its
    /// dictionary (zstd finds the repeats in plain values too, but pays a
    /// back-reference per row where the dictionary pays two bits).
    fn unique_and_status(rows: usize) -> (Arc<Schema>, RecordBatch) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("unique", DataType::UInt64, false),
            Field::new("status", DataType::Utf8, false),
        ]));
        let rows = rows as u64;
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(
                    (0..rows)
                        .map(|i| i.wrapping_mul(SCATTER))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    (0..rows)
                        .map(|i| {
                            let pick = i.wrapping_mul(SCATTER) >> 62;
                            let value = (pick + 1).wrapping_mul(SCATTER_2);
                            format!("{value:016x}{:016x}{value:016x}", !value)
                        })
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .expect("build RecordBatch");
        (schema, batch)
    }

    /// Whether each column chunk of each row group carries a dictionary page.
    fn dictionaries(body: &EncodedBody) -> Vec<Vec<(String, bool)>> {
        body.metadata
            .row_groups()
            .iter()
            .map(|rg| {
                rg.columns()
                    .iter()
                    .map(|c| {
                        (
                            c.column_path().string(),
                            c.dictionary_page_offset().is_some(),
                        )
                    })
                    .collect()
            })
            .collect()
    }

    /// A column loses its dictionary only when plain values are smaller; a
    /// candidate the plain encode did not measure keeps it.
    #[test]
    fn only_dictionaries_that_cost_bytes_are_dropped() {
        let candidates = ["a", "b", "c"].map(String::from);
        let with = sizes(&[("a", 100), ("b", 100), ("c", 100)]);
        let without = sizes(&[("a", 80), ("b", 120)]);
        assert_eq!(
            plain_where_smaller(&candidates, &with, &without),
            vec!["a".to_string()]
        );
    }

    /// The unique column drops its dictionary in every row group, the
    /// strings keep theirs, and the body is smaller for it.
    #[test]
    fn a_unique_column_drops_its_dictionary_and_four_strings_keep_theirs() {
        let (schema, batch) = unique_and_status(2 * TEST_GROUP_ROWS);
        let plan = plan(&schema);
        assert_eq!(
            plan.choose(&schema, &batch.slice(0, TEST_GROUP_ROWS))
                .expect("choose"),
            vec!["unique".to_string()]
        );

        let planned = encode_parquet_body_planned(&schema, slice::from_ref(&batch), plan.clone())
            .expect("planned body");
        let default = encode_parquet_body(
            &schema,
            slice::from_ref(&batch),
            plan.compression,
            TEST_GROUP_ROWS,
            &[],
        )
        .expect("default body");
        let group = vec![("unique".to_string(), false), ("status".to_string(), true)];
        assert_eq!(dictionaries(&planned), vec![group.clone(), group]);
        assert!(
            planned.body_len < default.body_len,
            "planned {} >= default {}",
            planned.body_len,
            default.body_len
        );
    }

    /// No data page holds more than `DATA_PAGE_ROW_LIMIT` rows, so a
    /// selective read that keeps one page decodes at most that many.
    #[test]
    fn data_pages_hold_at_most_the_row_limit() {
        const GROUPS: usize = 3;
        let rows = GROUPS * TEST_GROUP_ROWS;
        let (schema, batch) = unique_and_status(rows);
        let body = encode_parquet_body_planned(
            &schema,
            slice::from_ref(&batch),
            BodyPlan {
                row_group_size: rows,
                ..plan(&schema)
            },
        )
        .expect("planned body");
        for pages in page_rows(body) {
            assert!(pages.len() >= GROUPS, "pages {pages:?}");
            assert!(
                pages.iter().all(|&rows| rows <= DATA_PAGE_ROW_LIMIT as i64),
                "pages {pages:?}"
            );
        }
    }

    /// Rows of each data page in `body`, one list per column chunk.
    fn page_rows(body: EncodedBody) -> Vec<Vec<i64>> {
        let parts = splice_index_blobs(body, &[], &[], &[], &[]).expect("splice");
        let metadata = ParquetMetaDataReader::new()
            .with_page_index_policy(PageIndexPolicy::Required)
            .parse_and_finish(&Bytes::from(parts.bytes))
            .expect("metadata with page index");
        let offsets = metadata.offset_index().expect("offset index");
        let mut chunks = Vec::new();
        for (group, columns) in metadata.row_groups().iter().zip(offsets) {
            for column in columns {
                let firsts: Vec<i64> = column
                    .page_locations()
                    .iter()
                    .map(|page| page.first_row_index)
                    .chain([group.num_rows()])
                    .collect();
                chunks.push(firsts.windows(2).map(|pair| pair[1] - pair[0]).collect());
            }
        }
        chunks
    }

    /// A body that dropped some dictionaries is ordinary Parquet: a standard
    /// reader returns exactly the rows written, whichever way each column
    /// was encoded.
    #[test]
    fn a_body_without_some_dictionaries_reads_back_as_written() {
        let (schema, batch) = unique_and_status(2 * TEST_GROUP_ROWS);
        let body = encode_parquet_body_planned(&schema, slice::from_ref(&batch), plan(&schema))
            .expect("planned body");
        assert!(
            dictionaries(&body).iter().flatten().any(|(_, has)| !has),
            "the fixture must drop a dictionary"
        );
        let parts = splice_index_blobs(body, &[], &[], &[], &[]).expect("splice");
        let read: Vec<RecordBatch> =
            ParquetRecordBatchReaderBuilder::try_new(Bytes::from(parts.bytes))
                .expect("open")
                .build()
                .expect("reader")
                .collect::<Result<_, _>>()
                .expect("read");
        let read = concat_batches(&schema, &read).expect("concat");
        assert_eq!(read.columns(), batch.columns());
    }

    /// With no candidates the plan changes nothing: the body is the one the
    /// writer's defaults produce, byte for byte.
    #[test]
    fn a_plan_with_no_candidates_writes_the_default_body() {
        let (schema, batch) = unique_and_status(TEST_GROUP_ROWS);
        let plan = BodyPlan {
            dictionary_candidates: Vec::new(),
            ..plan(&schema)
        };
        let planned = encode_parquet_body_planned(&schema, slice::from_ref(&batch), plan.clone())
            .expect("planned body");
        let default = encode_parquet_body(
            &schema,
            slice::from_ref(&batch),
            plan.compression,
            TEST_GROUP_ROWS,
            &[],
        )
        .expect("default body");
        assert_eq!(planned.body_len, default.body_len);
        assert_eq!(dictionaries(&planned), dictionaries(&default));
    }

    /// A body shorter than one row group is measured on all of its rows.
    #[test]
    fn a_body_shorter_than_one_group_still_chooses() {
        let (schema, batch) = unique_and_status(TEST_GROUP_ROWS / 2);
        let planned = encode_parquet_body_planned(&schema, slice::from_ref(&batch), plan(&schema))
            .expect("planned body");
        assert_eq!(
            dictionaries(&planned),
            vec![vec![
                ("unique".to_string(), false),
                ("status".to_string(), true)
            ]]
        );
    }

    /// Streaming rows in through the planned encoder chooses the same
    /// dictionaries as handing them over at once, with the same row groups.
    /// (Page boundaries follow the write batches, so the bytes after the
    /// held rows may differ.)
    #[test]
    fn the_planned_encoder_chooses_the_same_however_rows_arrive() {
        let (schema, batch) = unique_and_status(3 * TEST_GROUP_ROWS / 2);
        let whole = encode_parquet_body_planned(&schema, slice::from_ref(&batch), plan(&schema))
            .expect("whole");
        let mut streamed = PlannedBodyEncoder::new(&schema, plan(&schema));
        let piece = TEST_GROUP_ROWS / 3 + 1;
        let mut start = 0;
        while start < batch.num_rows() {
            let len = piece.min(batch.num_rows() - start);
            streamed
                .write_batch(&batch.slice(start, len))
                .expect("write piece");
            start += len;
        }
        let streamed = streamed.finish().expect("streamed");
        let rows = |body: &EncodedBody| -> Vec<i64> {
            body.metadata
                .row_groups()
                .iter()
                .map(|rg| rg.num_rows())
                .collect()
        };
        assert_eq!(rows(&whole), rows(&streamed));
        assert_eq!(dictionaries(&whole), dictionaries(&streamed));
    }

    // ---- byte-bounded write slices ----

    /// Cap the slicing tests use, small so they stay fast.
    const TEST_MAX_WRITE_BYTES: usize = 1000;

    /// Bytes in an ordinary test value; ten of them fill the cap.
    const TEST_VALUE_BYTES: usize = 100;

    /// Bytes in an oversized test value, larger than the cap alone.
    const TEST_BIG_VALUE_BYTES: usize = 5000;

    /// Bytes of the row number each test value starts with, so values differ.
    const TEST_ID_BYTES: usize = 8;

    /// An id column and a text column whose row `i` holds `lens[i]` bytes.
    fn text_batch(lens: &[usize]) -> RecordBatch {
        let text: LargeStringArray = lens
            .iter()
            .enumerate()
            .map(|(i, len)| Some(format!("{i:08}{}", "x".repeat(len - TEST_ID_BYTES))))
            .collect();
        let ids = UInt64Array::from_iter_values(0..lens.len() as u64);
        let schema = Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("text", DataType::LargeUtf8, true),
        ]);
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(ids), Arc::new(text)])
            .expect("text batch")
    }

    /// A batch under the cap is written whole.
    #[test]
    fn small_batch_is_one_slice() {
        let batch = text_batch(&[TEST_VALUE_BYTES; 5]);
        let slices = byte_bounded_slices(&batch, TEST_MAX_WRITE_BYTES);
        assert_eq!(slices, vec![batch]);
    }

    /// Each slice is as long as the cap allows, a value over the cap gets
    /// a slice of its own, and the slices keep every row in order.
    #[test]
    fn slices_hold_at_most_the_cap() {
        let mut lens = vec![TEST_VALUE_BYTES; 50];
        lens.insert(20, TEST_BIG_VALUE_BYTES);
        let batch = text_batch(&lens);
        let slices = byte_bounded_slices(&batch, TEST_MAX_WRITE_BYTES);
        let rows: Vec<usize> = slices.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(rows, vec![10, 10, 1, 10, 10, 10]);
        let joined = concat_batches(&batch.schema(), &slices).expect("concat slices");
        assert_eq!(joined, batch);
    }

    /// Values inside a list count, and a sliced list counts only the
    /// values its own rows point at.
    #[test]
    fn list_values_count_toward_the_cap() {
        const ROWS: usize = 20;
        const ITEMS_PER_ROW: usize = 2;
        let mut builder = ListBuilder::new(StringBuilder::new());
        for _ in 0..ROWS {
            for _ in 0..ITEMS_PER_ROW {
                builder.values().append_value("x".repeat(TEST_VALUE_BYTES));
            }
            builder.append(true);
        }
        let list: ArrayRef = Arc::new(builder.finish());
        let row_bytes = ITEMS_PER_ROW * TEST_VALUE_BYTES;
        assert_eq!(value_bytes(list.slice(3, 4).as_ref()), 4 * row_bytes);

        let batch = RecordBatch::try_from_iter([("tags", list)]).expect("list batch");
        let slices = byte_bounded_slices(&batch, TEST_MAX_WRITE_BYTES);
        let rows: Vec<usize> = slices.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(
            rows,
            vec![TEST_MAX_WRITE_BYTES / row_bytes; ROWS * row_bytes / TEST_MAX_WRITE_BYTES]
        );
    }

    /// Rows in the page-size test: one run of the Parquet writer's values.
    const BIG_PAGE_ROWS: usize = 1024;

    /// Bytes per value in the page-size test: 1024 of them pass the cap.
    const BIG_PAGE_VALUE_BYTES: usize = 80 * 1024;

    /// Largest uncompressed page of the `text` column in `body`.
    fn largest_text_page(body: EncodedBody, schema: &Schema) -> usize {
        let parts = splice_index_blobs(body, &[], &[], &[], &[]).expect("splice");
        let reader = SerializedFileReader::new(Bytes::from(parts.bytes)).expect("reader");
        let text_column = schema.index_of("text").expect("text column");
        let mut pages = reader
            .get_row_group(0)
            .expect("row group")
            .get_column_page_reader(text_column)
            .expect("page reader");
        let mut largest = 0;
        while let Some(page) = pages.get_next_page().expect("page") {
            largest = largest.max(page.buffer().len());
        }
        largest
    }

    /// Values large enough that 1024 of them pass the cap are written in
    /// pages near the cap, not in one page per 1024 values.
    #[test]
    fn large_values_keep_pages_near_the_write_cap() {
        let batch = text_batch(&[BIG_PAGE_VALUE_BYTES; BIG_PAGE_ROWS]);
        let body = encode_parquet_body(
            &batch.schema(),
            slice::from_ref(&batch),
            Compression::ZSTD(ZstdLevel::default()),
            BIG_PAGE_ROWS,
            &[],
        )
        .expect("body");
        let largest = largest_text_page(body, &batch.schema());
        assert!(
            largest <= MAX_WRITE_BYTES + BIG_PAGE_VALUE_BYTES,
            "largest page {largest} bytes"
        );
    }

    /// The planned encoder, which first measures a sample with and without
    /// dictionaries, keeps pages near the cap too, for a view column as
    /// well as an offset one.
    #[test]
    fn planned_body_keeps_pages_near_the_write_cap() {
        let batch = text_batch(&[BIG_PAGE_VALUE_BYTES; BIG_PAGE_ROWS]);
        let text = batch.column_by_name("text").expect("text column");
        let view: ArrayRef = Arc::new(StringViewArray::from_iter(text.as_string::<i64>().iter()));
        let view_batch = RecordBatch::try_from_iter([("text", view)]).expect("view batch");
        for batch in [batch, view_batch] {
            let schema = batch.schema();
            let body = encode_parquet_body_planned(
                &schema,
                slice::from_ref(&batch),
                BodyPlan {
                    row_group_size: BIG_PAGE_ROWS,
                    ..plan(&schema)
                },
            )
            .expect("planned body");
            let largest = largest_text_page(body, &schema);
            assert!(
                largest <= MAX_WRITE_BYTES + BIG_PAGE_VALUE_BYTES,
                "{} largest page {largest} bytes",
                schema.field(0).data_type()
            );
        }
    }

    /// A slice long enough to hold whole runs of the writer's values ends
    /// on a run, so neighbouring columns keep their page row limit.
    #[test]
    fn long_slices_end_on_a_write_run() {
        const ROWS: usize = 3000;
        const CAP_ROWS: usize = 1500;
        let batch = text_batch(&[TEST_VALUE_BYTES; ROWS]);
        let slices = byte_bounded_slices(&batch, CAP_ROWS * TEST_VALUE_BYTES);
        let rows: Vec<usize> = slices.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(
            rows,
            vec![
                DEFAULT_WRITE_BATCH_SIZE,
                DEFAULT_WRITE_BATCH_SIZE,
                ROWS - 2 * DEFAULT_WRITE_BATCH_SIZE
            ]
        );
    }

    /// Bytes per value that make a slice just under two runs of the
    /// writer's values, so slices not cut to a whole run would end mid-run.
    const MID_RUN_VALUE_BYTES: usize = MAX_WRITE_BYTES / (2 * DEFAULT_WRITE_BATCH_SIZE - 1);

    /// With a large text column, the `id` column's pages still hold at
    /// most `DATA_PAGE_ROW_LIMIT` rows.
    #[test]
    fn large_values_keep_the_page_row_limit() {
        let rows = 5 * DEFAULT_WRITE_BATCH_SIZE;
        let batch = text_batch(&vec![MID_RUN_VALUE_BYTES; rows]);
        let schema = batch.schema();
        let body = encode_parquet_body(
            &schema,
            slice::from_ref(&batch),
            Compression::UNCOMPRESSED,
            rows,
            &[],
        )
        .expect("body");
        let id_pages = &page_rows(body)[schema.index_of("id").expect("id column")];
        assert!(
            id_pages
                .iter()
                .all(|&rows| rows <= DATA_PAGE_ROW_LIMIT as i64),
            "id pages {id_pages:?}"
        );
    }

    /// Every type that holds variable-size or wide values counts its bytes,
    /// for the whole array and for a slice of it. A struct or map counts
    /// only its largest field.
    #[test]
    fn every_value_type_counts_its_bytes() {
        const ROWS: usize = 10;
        const SLICE_START: usize = 2;
        const SLICE_ROWS: usize = 3;
        let value = "x".repeat(TEST_VALUE_BYTES);
        let large_text = || -> ArrayRef {
            Arc::new(LargeStringArray::from_iter_values(vec![
                value.as_str();
                ROWS
            ]))
        };

        let mut map = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
        for _ in 0..ROWS {
            map.keys().append_value(&value);
            map.values().append_value(&value);
            map.append(true).expect("map row");
        }
        let item = Arc::new(Field::new("item", DataType::LargeUtf8, true));
        let struct_fields = vec![
            Arc::new(Field::new("a", DataType::LargeUtf8, true)),
            Arc::new(Field::new("b", DataType::LargeUtf8, true)),
        ];
        let cases: Vec<(ArrayRef, usize)> = vec![
            (
                Arc::new(StringViewArray::from_iter_values(vec![
                    value.as_str();
                    ROWS
                ])),
                TEST_VALUE_BYTES,
            ),
            (
                Arc::new(BinaryViewArray::from_iter_values(vec![
                    value.as_bytes();
                    ROWS
                ])),
                TEST_VALUE_BYTES,
            ),
            (
                Arc::new(
                    FixedSizeBinaryArray::try_from_iter(vec![value.as_bytes(); ROWS].into_iter())
                        .expect("fixed-size binary"),
                ),
                TEST_VALUE_BYTES,
            ),
            (
                Arc::new(FixedSizeListArray::new(
                    item,
                    2,
                    Arc::new(LargeStringArray::from_iter_values(vec![
                        value.as_str();
                        2 * ROWS
                    ])),
                    None,
                )),
                2 * TEST_VALUE_BYTES,
            ),
            (Arc::new(map.finish()), TEST_VALUE_BYTES),
            (
                Arc::new(StructArray::new(
                    struct_fields.into(),
                    vec![large_text(), large_text()],
                    None,
                )),
                TEST_VALUE_BYTES,
            ),
        ];
        for (array, row_bytes) in cases {
            let name = array.data_type().to_string();
            assert_eq!(value_bytes(array.as_ref()), ROWS * row_bytes, "{name}");
            assert_eq!(
                value_bytes(array.slice(SLICE_START, SLICE_ROWS).as_ref()),
                SLICE_ROWS * row_bytes,
                "{name} slice"
            );
        }
    }
}
