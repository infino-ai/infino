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
    ops::Range,
    path::{Path, PathBuf},
    slice,
    sync::Arc,
};

use arrow::{
    array::{Array, ArrayRef, AsArray, OffsetSizeTrait, new_empty_array},
    record_batch::RecordBatch,
};
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use bytes::Bytes;
use parquet::{
    arrow::{
        ArrowSchemaConverter, ArrowWriter,
        arrow_writer::{
            ArrowColumnChunk, ArrowColumnWriter, ArrowRowGroupWriterFactory, compute_leaves,
        },
    },
    basic::Compression,
    file::{
        metadata::{
            FileMetaData, KeyValue, ParquetMetaData, ParquetMetaDataBuilder, ParquetMetaDataReader,
            ParquetMetaDataWriter,
        },
        properties::{DEFAULT_WRITE_BATCH_SIZE, WriterProperties, WriterPropertiesPtr},
        writer::SerializedFileWriter,
    },
    schema::types::{ColumnPath, SchemaDescriptor, Type, TypePtr},
};
use rayon::prelude::*;
use tempfile::NamedTempFile;

use crate::{
    config::scratch_root,
    runtime_metrics::ingest::offset_window,
    superfile::{LazyByteSource, LazyByteSourceError, format::kv},
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
///
/// A held group's column writers are made per column, as its encode
/// starts, and closed as it ends ([`FieldWriters`]), so only the columns
/// being encoded hold a writer: a few per thread, not one per column of
/// the body. What a closed column keeps until its group is appended is its
/// encoded pages.
///
/// What it holds is bounded by bytes as well as by groups: once the held
/// slices decode to [`HELD_GROUP_BYTES`], the held full groups are encoded
/// and the row group being filled is opened, its slices so far written to
/// its column writers, and every later slice of that group written as it
/// arrives, a task per column. A wide body — thousands of columns, each
/// row costing a few bytes in each even where it is null — then holds its
/// writers' encoded pages and one slice, not four row groups of its rows.
/// Each writer still sees the same slices in the same order, so the bytes
/// do not change. Where the caller keeps every row it writes for the whole
/// encode ([`Self::with_rows_held_by_caller`]), holding their slices costs
/// nothing, while an open group keeps a writer for every column of the
/// body: the bound then only encodes the full groups held so far, and the
/// group being filled is held until it is full.
pub(crate) struct ParquetBodyEncoder {
    body_file: NamedTempFile,
    writer: SerializedFileWriter<File>,
    columns: ArrowRowGroupWriterFactory,
    /// One field's column writers at a time, for the held groups.
    field_writers: FieldWriters,
    schema: SchemaRef,
    row_group_size: usize,
    /// The current row group's slices, in arrival order, while it is held.
    held: Vec<RecordBatch>,
    /// Rows of the current row group, held or already written.
    held_rows: usize,
    /// Full row groups not yet encoded, oldest first.
    full: Vec<Vec<RecordBatch>>,
    /// What the rows of `full` and `held` decode to, by [`decoded_bytes`].
    held_bytes: usize,
    /// What `held_bytes` may reach before the current row group is written
    /// as it arrives: [`HELD_GROUP_BYTES`].
    held_group_bytes: usize,
    /// The current row group's column writers, once it is written as its
    /// slices arrive rather than held.
    open: Option<Vec<ArrowColumnWriter>>,
    /// Whether a group whose held slices pass the bound is opened and
    /// written as it arrives; otherwise it is held until it is full.
    opens_groups: bool,
    /// Parquet leaf columns under each top-level field, in field order:
    /// the run of column writers a field's task writes to.
    leaves_per_field: Vec<usize>,
    /// Whether held groups are encoded side by side on the pool, or one at
    /// a time on this thread (see [`BodyPlan::encode_groups_together`]).
    together: bool,
}

/// Full row groups the encoder holds before encoding them side by side,
/// one task per column per group. The held groups are slices of the
/// caller's batches, so holding them costs no copy; the encoded chunks of
/// this many groups are what the encode holds beyond one group's.
const ROW_GROUPS_ENCODED_TOGETHER: usize = 4;

/// What the slices the encoder holds may decode to before it writes them
/// (see [`ParquetBodyEncoder`]). A narrow body of ordinary rows holds its
/// four row groups well under this; a body thousands of columns wide
/// reaches it within one.
pub(crate) const HELD_GROUP_BYTES: usize = 256 * 1024 * 1024;

/// Bytes the rows of `batch` decode to: each column's share of its
/// buffers, a buffer shared between columns counted once for each. That
/// is what holding the batch is charged: a column's levels and pages grow
/// with its rows the same way whether its buffer is shared or not.
pub(crate) fn decoded_bytes(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|column| column.to_data().get_slice_memory_size().unwrap_or(0))
        .sum()
}

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
            &scratch_root(),
        )
    }

    /// [`Self::new`] with the dictionary turned off for `plain_columns` and
    /// the body's scratch file under `scratch`.
    fn with_plain_columns(
        schema: &Arc<Schema>,
        compression: Compression,
        row_group_size: usize,
        column_page_size_limits: &[(&str, usize)],
        plain_columns: &[String],
        together: bool,
        scratch: &Path,
    ) -> Result<Self, FooterError> {
        let props = writer_properties(
            compression,
            row_group_size,
            column_page_size_limits,
            plain_columns,
        );

        let (body_file, writer, columns) = scratch_body(schema, props, scratch)?;
        let field_writers = FieldWriters::new(
            writer.schema_descr(),
            schema,
            Arc::clone(writer.properties()),
        );
        let leaves_per_field = schema
            .fields()
            .iter()
            .map(|field| {
                compute_leaves(field.as_ref(), &new_empty_array(field.data_type()))
                    .map(|leaves| leaves.len())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            body_file,
            writer,
            columns,
            field_writers,
            schema: Arc::clone(schema),
            row_group_size: row_group_size.max(1),
            held: Vec::new(),
            held_rows: 0,
            full: Vec::new(),
            held_bytes: 0,
            held_group_bytes: HELD_GROUP_BYTES,
            open: None,
            opens_groups: true,
            leaves_per_field,
            together,
        })
    }

    /// The encoder with `bytes` as its bound on what it holds.
    fn holding_at_most(mut self, bytes: usize) -> Self {
        self.held_group_bytes = bytes;
        self
    }

    /// The encoder for rows its caller keeps until the encode finishes:
    /// past the bound, it encodes the full groups it holds but never opens
    /// a group (see [`ParquetBodyEncoder`]).
    fn with_rows_held_by_caller(mut self) -> Self {
        self.opens_groups = false;
        self
    }

    /// Add `batch`, cutting it where a row group fills as `ArrowWriter` does.
    pub(crate) fn write_batch(&mut self, batch: &RecordBatch) -> Result<(), FooterError> {
        let mut rest = batch.clone();
        while rest.num_rows() > 0 {
            let room = self.row_group_size - self.held_rows;
            let take = room.min(rest.num_rows());
            let part = rest.slice(0, take);
            rest = rest.slice(take, rest.num_rows() - take);
            self.held_rows += take;
            match self.open.as_mut() {
                Some(writers) => encode_into(
                    &self.schema,
                    &self.leaves_per_field,
                    writers,
                    slice::from_ref(&part),
                    self.together,
                )?,
                None => {
                    self.held_bytes += decoded_bytes(&part);
                    self.held.push(part);
                    if self.held_bytes >= self.held_group_bytes {
                        match self.opens_groups {
                            true => self.open_current_group()?,
                            false if !self.full.is_empty() => self.flush_row_groups()?,
                            false => {}
                        }
                    }
                }
            }
            if self.held_rows >= self.row_group_size {
                self.held_rows = 0;
                match self.open.take() {
                    Some(writers) => self.append_row_group(writers)?,
                    None => {
                        self.full.push(mem::take(&mut self.held));
                        let held = match self.together {
                            true => ROW_GROUPS_ENCODED_TOGETHER,
                            false => 1,
                        };
                        if self.full.len() >= held {
                            self.flush_row_groups()?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Stop holding the current row group: encode the full groups held
    /// before it, then open its column writers and write its slices so far.
    fn open_current_group(&mut self) -> Result<(), FooterError> {
        self.flush_row_groups()?;
        let index = self.writer.flushed_row_groups().len();
        let mut writers = self.columns.create_column_writers(index)?;
        let held = mem::take(&mut self.held);
        self.held_bytes = 0;
        encode_into(
            &self.schema,
            &self.leaves_per_field,
            &mut writers,
            &held,
            self.together,
        )?;
        self.open = Some(writers);
        Ok(())
    }

    /// Encode the held full row groups, one task per column per group, and
    /// append them in order.
    ///
    /// A task makes its column's writers, writes the group's slices and
    /// closes them. Side by side, the closed chunks of every held group are
    /// gathered and then appended; one at a time, each column's chunks are
    /// appended as it closes, so the body holds one column's pages at most.
    fn flush_row_groups(&mut self) -> Result<(), FooterError> {
        let groups = mem::take(&mut self.full);
        self.held_bytes = self.held.iter().map(decoded_bytes).sum();
        if groups.is_empty() {
            return Ok(());
        }
        let first_index = self.writer.flushed_row_groups().len();
        let groups: Vec<Vec<RecordBatch>> = groups.iter().map(|g| write_slices(g)).collect();
        let fields = self.schema.fields().len();
        let Self {
            writer,
            field_writers,
            together,
            ..
        } = self;
        if !*together {
            for (g, group) in groups.iter().enumerate() {
                let mut row_group = writer.next_row_group()?;
                for field in 0..fields {
                    let columns = group.iter().map(|batch| batch.column(field));
                    for chunk in field_writers.encode(field, first_index + g, columns)? {
                        chunk.append_to_row_group(&mut row_group)?;
                    }
                }
                row_group.close()?;
            }
            return Ok(());
        }
        let tasks: Vec<(usize, usize)> = (0..groups.len())
            .flat_map(|g| (0..fields).map(move |field| (g, field)))
            .collect();
        let chunks: Vec<Vec<ArrowColumnChunk>> = tasks
            .par_iter()
            .map(|&(g, field)| {
                let columns = groups[g].iter().map(|batch| batch.column(field));
                field_writers.encode(field, first_index + g, columns)
            })
            .collect::<Result<_, _>>()?;
        let n_groups = groups.len();
        drop(groups);
        let mut chunks = chunks.into_iter();
        for _ in 0..n_groups {
            let mut row_group = writer.next_row_group()?;
            for column in chunks.by_ref().take(fields).flatten() {
                column.append_to_row_group(&mut row_group)?;
            }
            row_group.close()?;
        }
        Ok(())
    }

    /// Close `writers`, one row group's, and append it to the body.
    fn append_row_group(&mut self, writers: Vec<ArrowColumnWriter>) -> Result<(), FooterError> {
        let mut row_group = self.writer.next_row_group()?;
        for writer in writers {
            writer.close()?.append_to_row_group(&mut row_group)?;
        }
        row_group.close()?;
        Ok(())
    }

    /// Close the writer and strip the Parquet footer, yielding the same
    /// [`EncodedBody`] as [`encode_parquet_body`].
    pub(crate) fn finish(mut self) -> Result<EncodedBody, FooterError> {
        match self.open.take() {
            Some(writers) => self.append_row_group(writers)?,
            None => {
                if !self.held.is_empty() {
                    self.full.push(mem::take(&mut self.held));
                }
                self.flush_row_groups()?;
            }
        }
        let Self {
            body_file, writer, ..
        } = self;
        writer.close()?;
        strip_parquet_footer(body_file)
    }
}

/// Makes the column writers of one top-level field of a body at a time.
///
/// [`ArrowRowGroupWriterFactory`] makes a row group's writers for every
/// column at once, and each writer holds a codec and encoders of its own
/// for as long as it is open: a zstd codec keeps a decompression context
/// (about 100 KiB) beside its compression context (from a few KiB to over a
/// MiB, by the largest page it has compressed), and a dictionary encoder a
/// hash table of 4,096 slots (about 72 KiB for 64-bit values): on a body
/// thousands of columns wide, a few hundred KiB a column whatever the
/// column holds, while every one is open. A field's writers made here come from
/// a schema of that field alone, whose leaves have the same paths, levels
/// and types as in the whole body's schema, so the chunks they close append
/// to the body's row group exactly as chunks of the whole schema's writers
/// would; the body's writer checks each chunk's column against its own.
struct FieldWriters {
    /// The name of the body's schema root.
    root: String,
    /// The Parquet type of each top-level field, in field order.
    fields: Vec<TypePtr>,
    /// Each top-level Arrow field, in field order.
    arrow: Vec<FieldRef>,
    props: WriterPropertiesPtr,
}

impl FieldWriters {
    /// Writers for the fields of `arrow`, whose Parquet schema is `parquet`,
    /// with `props`.
    fn new(parquet: &SchemaDescriptor, arrow: &Schema, props: WriterPropertiesPtr) -> Self {
        Self {
            root: parquet.root_schema().name().to_string(),
            fields: parquet.root_schema().get_fields().to_vec(),
            arrow: arrow.fields().iter().cloned().collect(),
            props,
        }
    }

    /// The column writers of field `field`, for row group `row_group`.
    fn create(
        &self,
        field: usize,
        row_group: usize,
    ) -> Result<Vec<ArrowColumnWriter>, FooterError> {
        let root = Type::group_type_builder(&self.root)
            .with_fields(vec![Arc::clone(&self.fields[field])])
            .build()?;
        let file = SerializedFileWriter::new(io::sink(), Arc::new(root), Arc::clone(&self.props))?;
        let arrow = Arc::new(Schema::new(vec![Arc::clone(&self.arrow[field])]));
        Ok(ArrowRowGroupWriterFactory::new(&file, arrow).create_column_writers(row_group)?)
    }

    /// `columns`, the values of field `field` in order, written by fresh
    /// writers for row group `row_group` and closed: the field's column
    /// chunks.
    fn encode<'c>(
        &self,
        field: usize,
        row_group: usize,
        columns: impl IntoIterator<Item = &'c ArrayRef>,
    ) -> Result<Vec<ArrowColumnChunk>, FooterError> {
        let mut writers = self.create(field, row_group)?;
        for column in columns {
            self.write(field, &mut writers, column)?;
        }
        writers
            .into_iter()
            .map(|writer| writer.close().map_err(FooterError::from))
            .collect()
    }

    /// Write `column`, values of field `field`, to `writers`, that field's
    /// open column writers.
    fn write(
        &self,
        field: usize,
        writers: &mut [ArrowColumnWriter],
        column: &ArrayRef,
    ) -> Result<(), FooterError> {
        write_leaves(&self.arrow[field], column, writers)
    }
}

/// Write `column`, the values of `field`, to `writers`, the column writers
/// of that field's leaves in order.
fn write_leaves(
    field: &Field,
    column: &ArrayRef,
    writers: &mut [ArrowColumnWriter],
) -> Result<(), FooterError> {
    let leaves = compute_leaves(field, column)?;
    for (writer, leaf) in writers.iter_mut().zip(leaves) {
        writer.write(&leaf)?;
    }
    Ok(())
}

/// One encode task: a top-level field's index, the slices to write, and
/// the run of leaf column writers under that field.
type ColumnTask<'a, 'w> = (usize, &'a [RecordBatch], &'w mut [ArrowColumnWriter]);

/// `writers`, one row group's leaf column writers in schema order, cut
/// into the run under each top-level field, by field index.
fn column_tasks<'w>(
    leaves_per_field: &[usize],
    writers: &'w mut [ArrowColumnWriter],
) -> Result<Vec<(usize, &'w mut [ArrowColumnWriter])>, FooterError> {
    let mut rest = writers;
    let mut tasks = Vec::with_capacity(leaves_per_field.len());
    for (field, &leaves) in leaves_per_field.iter().enumerate() {
        if rest.len() < leaves {
            return Err(FooterError::Malformed(
                "fewer column writers than the schema has leaves",
            ));
        }
        let (own, after) = mem::take(&mut rest).split_at_mut(leaves);
        rest = after;
        tasks.push((field, own));
    }
    if !rest.is_empty() {
        return Err(FooterError::Malformed(
            "more column writers than the schema has leaves",
        ));
    }
    Ok(tasks)
}

/// Run `tasks`, side by side on the pool when `together`.
///
/// A task computes its column's leaves (the levels and values the writer
/// takes) from its slices as it writes them, so a column's levels exist
/// only while its task runs: computed up front, every column's levels of
/// every held group would be alive at once, two bytes a row per column for
/// a nullable one.
fn run_column_tasks(
    schema: &Schema,
    tasks: &mut [ColumnTask<'_, '_>],
    together: bool,
) -> Result<(), FooterError> {
    let encode = |(field, batches, writers): &mut ColumnTask<'_, '_>| -> Result<(), FooterError> {
        let field_ref = schema.field(*field);
        batches
            .iter()
            .try_for_each(|batch| write_leaves(field_ref, batch.column(*field), writers))
    };
    match together {
        true => tasks.par_iter_mut().try_for_each(encode),
        false => tasks.iter_mut().try_for_each(encode),
    }
}

/// Write `batches` to `writers`, one row group's, a task per column.
fn encode_into(
    schema: &Schema,
    leaves_per_field: &[usize],
    writers: &mut [ArrowColumnWriter],
    batches: &[RecordBatch],
    together: bool,
) -> Result<(), FooterError> {
    let parts = write_slices(batches);
    let mut tasks: Vec<ColumnTask<'_, '_>> = column_tasks(leaves_per_field, writers)?
        .into_iter()
        .map(|(field, own)| (field, parts.as_slice(), own))
        .collect();
    run_column_tasks(schema, &mut tasks, together)
}

/// `batches`, in order, each cut by [`byte_bounded_slices`] at
/// [`MAX_WRITE_BYTES`]: the slices every column writer is handed, whether
/// a row group is held and encoded whole or written as it arrives. Cut
/// once per batch rather than in each column's task, since where a batch
/// is cut depends on all of its columns.
fn write_slices(batches: &[RecordBatch]) -> Vec<RecordBatch> {
    batches
        .iter()
        .flat_map(|batch| byte_bounded_slices(batch, MAX_WRITE_BYTES))
        .collect()
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
///   sample rows ─────┬─ encode as configured ──────┐
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
    /// What the rows the encoder holds, to choose the dictionaries and
    /// before it encodes them, may decode to: [`HELD_GROUP_BYTES`].
    pub(crate) held_group_bytes: usize,
    /// The directory the body's scratch file is created in.
    pub(crate) scratch: PathBuf,
}

impl BodyPlan {
    /// The candidates written without a dictionary, for a body whose first
    /// rows are the batches of `sample`, in order.
    fn choose(
        &self,
        schema: &Arc<Schema>,
        sample: &[RecordBatch],
    ) -> Result<Vec<String>, FooterError> {
        if self.dictionary_candidates.is_empty() {
            return Ok(Vec::new());
        }
        let (dictionary, plain) = self.measure(schema, sample)?;
        Ok(plain_where_smaller(
            &self.dictionary_candidates,
            &dictionary,
            &plain,
        ))
    }

    /// Compressed bytes of each candidate column when the batches of
    /// `sample` are encoded in order, with its dictionary and without.
    ///
    /// Each candidate is encoded on its own, once each way, by writers made
    /// for it alone ([`FieldWriters`]) and dropped as soon as its sizes are
    /// read: the measure holds the writers of the columns being measured,
    /// not two encodes of every column of the body, and columns that are not
    /// candidates are not encoded at all. A column's encode depends only on
    /// its own values, the writer settings and the slices it is written in,
    /// which are the body's, so its sizes are those of an encode of the
    /// whole sample. The sample's batches are encoded as they are, slices
    /// of the rows written, never joined into one: a joined sample would
    /// copy a slot per cell of the body, null or not, in proportion to its
    /// width, where encoding the slices copies nothing.
    fn measure(
        &self,
        schema: &Arc<Schema>,
        sample: &[RecordBatch],
    ) -> Result<(ColumnSizes, ColumnSizes), FooterError> {
        let slices = write_slices(sample);
        let candidates: Vec<(usize, Vec<ArrayRef>)> = self
            .dictionary_candidates
            .iter()
            .filter_map(|name| schema.index_of(name).ok())
            .map(|field| (field, sample_columns(&slices, field)))
            .collect();
        let (with, without) = self.field_writers(schema)?;
        self.measure_columns(schema, &with, &without, &candidates)
    }

    /// Writers for the fields of `schema` under this plan: every field with
    /// its dictionary, and every candidate without.
    fn field_writers(
        &self,
        schema: &Arc<Schema>,
    ) -> Result<(FieldWriters, FieldWriters), FooterError> {
        let limits = self.page_size_limits();
        let props = |plain: &[String]| {
            Arc::new(writer_properties(
                self.compression,
                self.row_group_size,
                &limits,
                plain,
            ))
        };
        let (with, without) = (props(&[]), props(&self.dictionary_candidates));
        let parquet = ArrowSchemaConverter::new()
            .with_coerce_types(with.coerce_types())
            .convert(schema)?;
        Ok((
            FieldWriters::new(&parquet, schema, with),
            FieldWriters::new(&parquet, schema, without),
        ))
    }

    /// The names of `candidates`, each a field of `schema` and its sample's
    /// values ([`sample_columns`]), written without a dictionary: those
    /// [`plain_where_smaller`] picks from their sizes measured by `with`
    /// and `without`.
    fn plain_among(
        &self,
        schema: &Schema,
        with: &FieldWriters,
        without: &FieldWriters,
        candidates: &[(usize, Vec<ArrayRef>)],
    ) -> Result<Vec<String>, FooterError> {
        let names: Vec<String> = candidates
            .iter()
            .map(|(field, _)| schema.field(*field).name().clone())
            .collect();
        let (dictionary, plain) = self.measure_columns(schema, with, without, candidates)?;
        Ok(plain_where_smaller(&names, &dictionary, &plain))
    }

    /// Compressed bytes of each of `candidates`, a field of `schema` and its
    /// sample's values in the slices the writer is handed, encoded by `with`
    /// (its dictionary) and by `without`.
    ///
    /// Each candidate is encoded on its own, once each way, by writers made
    /// for it alone and dropped as soon as its sizes are read (see
    /// [`Self::measure`]).
    fn measure_columns(
        &self,
        schema: &Schema,
        with: &FieldWriters,
        without: &FieldWriters,
        candidates: &[(usize, Vec<ArrayRef>)],
    ) -> Result<(ColumnSizes, ColumnSizes), FooterError> {
        let measure = |(field, columns): &(usize, Vec<ArrayRef>)| {
            let size = |writers: &FieldWriters| -> Result<u64, FooterError> {
                Ok(writers
                    .encode(*field, 0, columns)?
                    .iter()
                    .map(|chunk| {
                        u64::try_from(chunk.close().metadata.compressed_size()).unwrap_or(0)
                    })
                    .sum())
            };
            Ok::<_, FooterError>((*field, size(with)?, size(without)?))
        };
        let sizes: Vec<(usize, u64, u64)> = match self.encode_groups_together {
            true => candidates
                .par_iter()
                .map(measure)
                .collect::<Result<_, _>>()?,
            false => candidates.iter().map(measure).collect::<Result<_, _>>()?,
        };
        let (mut dictionary, mut plain) = (ColumnSizes::new(), ColumnSizes::new());
        for (field, with, without) in sizes {
            let name = schema.field(field).name();
            dictionary.insert(name.clone(), with);
            plain.insert(name.clone(), without);
        }
        Ok((dictionary, plain))
    }

    fn page_size_limits(&self) -> Vec<(&str, usize)> {
        self.column_page_size_limits
            .iter()
            .map(|(c, l)| (c.as_str(), *l))
            .collect()
    }
}

/// Column `at` of each of `slices`: one field's values in a sample.
fn sample_columns(slices: &[RecordBatch], at: usize) -> Vec<ArrayRef> {
    slices.iter().map(|s| Arc::clone(s.column(at))).collect()
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
/// from the first rows written: it holds rows until it has a row group's,
/// the rows the choice is measured on, or rows that decode to the plan's
/// [`BodyPlan::held_group_bytes`], or the body ends; chooses; then writes
/// them and streams the rest straight through.
pub(crate) struct PlannedBodyEncoder {
    schema: Arc<Schema>,
    plan: BodyPlan,
    held: Vec<RecordBatch>,
    held_rows: usize,
    /// What `held` decodes to, by [`decoded_bytes`].
    held_bytes: usize,
    /// Whether the caller keeps every row it writes until the encode
    /// finishes, handed on to the encoder (see
    /// [`ParquetBodyEncoder::with_rows_held_by_caller`]).
    rows_held_by_caller: bool,
    encoder: Option<ParquetBodyEncoder>,
}

impl PlannedBodyEncoder {
    pub(crate) fn new(schema: &Arc<Schema>, plan: BodyPlan) -> Self {
        Self {
            schema: Arc::clone(schema),
            plan,
            held: Vec::new(),
            held_rows: 0,
            held_bytes: 0,
            rows_held_by_caller: false,
            encoder: None,
        }
    }

    /// The encoder for rows its caller keeps until the encode finishes.
    pub(crate) fn with_rows_held_by_caller(mut self) -> Self {
        self.rows_held_by_caller = true;
        self
    }

    pub(crate) fn write_batch(&mut self, batch: &RecordBatch) -> Result<(), FooterError> {
        if let Some(encoder) = self.encoder.as_mut() {
            return encoder.write_batch(batch);
        }
        self.held_rows += batch.num_rows();
        self.held_bytes += decoded_bytes(batch);
        self.held.push(batch.clone());
        // The bytes bound holds whoever keeps the rows.
        if self.held_rows >= self.plan.row_group_size
            || self.held_bytes >= self.plan.held_group_bytes
        {
            self.start()?;
        }
        Ok(())
    }

    /// Choose the dictionaries from the held rows, open the encoder, and
    /// write them.
    ///
    /// The sample, the first row group's rows, is slices of the held
    /// batches, and nothing is copied: the held batches are written as they
    /// came, so a merge that feeds one large input batch never holds a
    /// second copy of it, and a wide body's sample costs nothing per column.
    fn start(&mut self) -> Result<(), FooterError> {
        let mut sample = Vec::new();
        let mut wanted = self.plan.row_group_size;
        for batch in &self.held {
            if wanted == 0 {
                break;
            }
            let take = batch.num_rows().min(wanted);
            sample.push(batch.slice(0, take));
            wanted -= take;
        }
        let plain_columns = self.plan.choose(&self.schema, &sample)?;
        let mut encoder = ParquetBodyEncoder::with_plain_columns(
            &self.schema,
            self.plan.compression,
            self.plan.row_group_size,
            &self.plan.page_size_limits(),
            &plain_columns,
            self.plan.encode_groups_together,
            &self.plan.scratch,
        )?
        .holding_at_most(self.plan.held_group_bytes);
        if self.rows_held_by_caller {
            encoder = encoder.with_rows_held_by_caller();
        }
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
    // `batches` outlive the encode, so the slices it holds of them cost
    // nothing.
    let mut encoder = PlannedBodyEncoder::new(schema, plan).with_rows_held_by_caller();
    for batch in batches {
        encoder.write_batch(batch)?;
    }
    encoder.finish()
}

/// A body's scratch file and the Parquet writer over it, with the factory
/// for a whole row group's column writers.
///
/// The body is encoded to a scratch temp file rather than a `Vec`, so it —
/// multi-GB at corpus scale — never materializes whole in RAM. The writer
/// owns an independent handle to the same file (`reopen`); the returned
/// file handle reads the footer back after close, and their cursors don't
/// alias. Built through `ArrowWriter` so the file starts, and its footer
/// carries the Arrow schema, exactly as an `ArrowWriter`'s does. The file is
/// created under `scratch`.
fn scratch_body(
    schema: &Arc<Schema>,
    props: WriterProperties,
    scratch: &Path,
) -> Result<
    (
        NamedTempFile,
        SerializedFileWriter<File>,
        ArrowRowGroupWriterFactory,
    ),
    FooterError,
> {
    let body_file = NamedTempFile::new_in(scratch)?;
    let write_handle = body_file.reopen()?;
    let (writer, columns) = ArrowWriter::try_new(write_handle, Arc::clone(schema), Some(props))?
        .into_serialized_writer()?;
    Ok((body_file, writer, columns))
}

/// A Parquet body written a row group at a time, each row group a run of
/// top-level fields at a time: a pass.
///
/// [`ParquetBodyEncoder`] is handed whole rows and holds them, every column,
/// until a row group's are encoded, or opens a writer for every column of
/// the body. A body thousands of columns wide whose rows decode to many
/// times what they store can afford neither: a row group of its rows can
/// decode to many gigabytes, and its writers, a few hundred kilobytes to a
/// few megabytes each, to gigabytes more. Here the caller hands a row
/// group's rows over once per pass, only that pass's columns of them, and
/// only that pass's writers are open: what the encode holds is the writers
/// of one pass and the rows the caller hands it at a time, however wide the
/// body and however large its rows. The row groups keep their row count, so
/// the body is laid out as one written whole: the same row groups, the
/// same columns, the same pages but where a write ends.
///
/// The dictionaries are chosen as [`PlannedBodyEncoder`] chooses them,
/// from the first row group: the first rows of each pass's columns, up to
/// a row group's or [`BodyPlan::held_group_bytes`] of them, are held and
/// measured with and without each candidate's dictionary before that
/// pass's writers are made.
pub(crate) struct PassBodyEncoder {
    body_file: NamedTempFile,
    writer: SerializedFileWriter<File>,
    schema: SchemaRef,
    plan: BodyPlan,
    /// Writers for every field with its dictionary.
    dictionary: FieldWriters,
    /// Writers for the candidate fields without one.
    plain: FieldWriters,
    /// Per field, whether it is written without its dictionary, once the
    /// first row group's pass over it has chosen.
    plain_fields: Vec<Option<bool>>,
}

impl PassBodyEncoder {
    pub(crate) fn new(schema: &Arc<Schema>, plan: BodyPlan) -> Result<Self, FooterError> {
        let (dictionary, plain) = plan.field_writers(schema)?;
        let props = writer_properties(
            plan.compression,
            plan.row_group_size,
            &plan.page_size_limits(),
            &[],
        );
        let (body_file, writer, _) = scratch_body(schema, props, &plan.scratch)?;
        Ok(Self {
            body_file,
            writer,
            schema: Arc::clone(schema),
            // A field that is no candidate keeps its dictionary: nothing to
            // choose, so its pass need not hold rows to measure.
            plain_fields: schema
                .fields()
                .iter()
                .map(|field| (!plan.dictionary_candidates.contains(field.name())).then_some(false))
                .collect(),
            plan,
            dictionary,
            plain,
        })
    }

    /// Write the body's next row group, its fields in the runs `passes`, in
    /// order: contiguous, together every field of the body once. For each
    /// pass `fill` is handed the pass's fields and a [`PassWriter`], and
    /// writes it the row group's rows, those fields of them in field order,
    /// in row order; every pass must be handed the same rows.
    pub(crate) fn write_row_group<E: From<FooterError>>(
        &mut self,
        passes: &[Range<usize>],
        mut fill: impl FnMut(Range<usize>, &mut PassWriter<'_>) -> Result<(), E>,
    ) -> Result<(), E> {
        let fields = self.schema.fields().len();
        let covered = passes.iter().try_fold(0, |next, pass| {
            (pass.start == next && pass.end > next).then_some(pass.end)
        });
        if covered != Some(fields) {
            return Err(
                FooterError::Malformed("passes do not cover the fields once, in order").into(),
            );
        }
        let Self {
            writer,
            schema,
            plan,
            dictionary,
            plain,
            plain_fields,
            ..
        } = self;
        let index = writer.flushed_row_groups().len();
        let mut row_group = writer.next_row_group().map_err(FooterError::from)?;
        let mut rows: Option<usize> = None;
        for pass in passes {
            let mut pass_writer = PassWriter {
                fields: pass.clone(),
                row_group: index,
                schema,
                plan,
                dictionary,
                plain,
                plain_fields: &mut plain_fields[pass.clone()],
                held: Vec::new(),
                held_bytes: 0,
                open: None,
                rows: 0,
            };
            fill(pass.clone(), &mut pass_writer)?;
            let (pass_rows, chunks) = pass_writer.close()?;
            if *rows.get_or_insert(pass_rows) != pass_rows {
                return Err(
                    FooterError::Malformed("passes of a row group hold different rows").into(),
                );
            }
            for chunk in chunks {
                chunk
                    .append_to_row_group(&mut row_group)
                    .map_err(FooterError::from)?;
            }
        }
        row_group.close().map_err(FooterError::from)?;
        Ok(())
    }

    /// Close the writer and strip the Parquet footer, yielding the same
    /// [`EncodedBody`] as [`ParquetBodyEncoder::finish`].
    pub(crate) fn finish(self) -> Result<EncodedBody, FooterError> {
        let Self {
            body_file, writer, ..
        } = self;
        writer.close()?;
        strip_parquet_footer(body_file)
    }
}

/// One pass of a [`PassBodyEncoder`] row group: the writers of its fields,
/// fed the row group's rows of those fields a batch at a time.
pub(crate) struct PassWriter<'a> {
    fields: Range<usize>,
    row_group: usize,
    schema: &'a Schema,
    plan: &'a BodyPlan,
    dictionary: &'a FieldWriters,
    plain: &'a FieldWriters,
    /// The pass's fields' dictionary choices, `None` until made.
    plain_fields: &'a mut [Option<bool>],
    /// Rows held to choose the pass's dictionaries before its writers are
    /// made; only ever in the body's first row group.
    held: Vec<RecordBatch>,
    held_bytes: usize,
    /// Each field's column writers, in field order, once made.
    open: Option<Vec<Vec<ArrowColumnWriter>>>,
    rows: usize,
}

impl PassWriter<'_> {
    /// Write `batch`, the pass's fields of the row group's next rows, in
    /// field order.
    pub(crate) fn write(&mut self, batch: &RecordBatch) -> Result<(), FooterError> {
        if batch.num_columns() != self.fields.len() {
            return Err(FooterError::Malformed(
                "a pass batch holds other fields than the pass",
            ));
        }
        self.rows += batch.num_rows();
        if self.open.is_none() && self.plain_fields.iter().any(Option::is_none) {
            self.held_bytes += decoded_bytes(batch);
            self.held.push(batch.clone());
            if self.held_bytes >= self.plan.held_group_bytes {
                self.open_writers()?;
            }
            return Ok(());
        }
        if self.open.is_none() {
            self.open_writers()?;
        }
        self.write_open(batch)
    }

    /// Choose the dictionaries of the fields that have none chosen from the
    /// rows held, make every field's writers, and write the held rows.
    fn open_writers(&mut self) -> Result<(), FooterError> {
        if self.plain_fields.iter().any(Option::is_none) {
            let slices = write_slices(&self.held);
            let candidates: Vec<(usize, Vec<ArrayRef>)> = self
                .fields
                .clone()
                .filter(|&field| {
                    self.plan
                        .dictionary_candidates
                        .contains(self.schema.field(field).name())
                })
                .map(|field| {
                    let at = field - self.fields.start;
                    (field, sample_columns(&slices, at))
                })
                .collect();
            let plain =
                self.plan
                    .plain_among(self.schema, self.dictionary, self.plain, &candidates)?;
            for (at, field) in self.fields.clone().enumerate() {
                let written_plain = plain.contains(self.schema.field(field).name());
                self.plain_fields[at].get_or_insert(written_plain);
            }
        }
        let writers = self
            .fields
            .clone()
            .zip(self.plain_fields.iter())
            .map(|(field, plain)| match plain {
                Some(true) => self.plain.create(field, self.row_group),
                _ => self.dictionary.create(field, self.row_group),
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.open = Some(writers);
        self.held_bytes = 0;
        for batch in mem::take(&mut self.held) {
            self.write_open(&batch)?;
        }
        Ok(())
    }

    /// Write `batch` to the open writers, a task per field, side by side on
    /// the pool when the plan encodes that way.
    fn write_open(&mut self, batch: &RecordBatch) -> Result<(), FooterError> {
        let Some(open) = self.open.as_mut() else {
            return Err(FooterError::Malformed(
                "a pass written before its writers were made",
            ));
        };
        let start = self.fields.start;
        let writers = self.dictionary;
        for slice in byte_bounded_slices(batch, MAX_WRITE_BYTES) {
            let write = |(at, field_writers): (usize, &mut Vec<ArrowColumnWriter>)| {
                writers.write(start + at, field_writers, slice.column(at))
            };
            match self.plan.encode_groups_together {
                true => open.par_iter_mut().enumerate().try_for_each(write)?,
                false => open.iter_mut().enumerate().try_for_each(write)?,
            }
        }
        Ok(())
    }

    /// Close the pass's writers: the rows written and the pass's column
    /// chunks, in field order.
    fn close(mut self) -> Result<(usize, Vec<ArrowColumnChunk>), FooterError> {
        if self.open.is_none() {
            self.open_writers()?;
        }
        let chunks = self
            .open
            .take()
            .into_iter()
            .flatten()
            .flatten()
            .map(|writer| writer.close().map_err(FooterError::from))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((self.rows, chunks))
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
    use std::{fs, slice};

    use arrow::{buffer::NullBuffer, compute::concat_batches};
    use arrow_array::{
        ArrayRef, BinaryViewArray, FixedSizeBinaryArray, FixedSizeListArray, Float64Array,
        Int64Array, LargeStringArray, StringArray, StringViewArray, StructArray, UInt64Array,
        builder::{ListBuilder, MapBuilder, StringBuilder},
    };
    use arrow_schema::{DataType, Field, Fields};
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
            held_group_bytes: HELD_GROUP_BYTES,
            scratch: scratch_root(),
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
            plan.choose(&schema, &[batch.slice(0, TEST_GROUP_ROWS)])
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

    /// On a body thousands of columns wide the dictionary choice is still
    /// measured on a whole row group: the sample is slices of the rows
    /// written, so it costs nothing per column. A column whose first
    /// thousand values are distinct and whose others repeat four strings
    /// is larger plain on those first rows alone, and smaller with its
    /// dictionary over the group, so it keeps the dictionary.
    #[test]
    fn a_wide_body_chooses_its_dictionaries_on_a_whole_row_group() {
        const COLUMNS: usize = 9_000;
        const ROWS: usize = 16_384;
        const DISTINCT_FIRST: usize = 1_024;
        let mut fields = vec![Field::new("status", DataType::Utf8, false)];
        fields.extend((1..COLUMNS).map(|c| Field::new(format!("c{c}"), DataType::Int64, true)));
        let schema = Arc::new(Schema::new(fields));
        let status = StringArray::from(
            (0..ROWS as u64)
                .map(|i| {
                    let pick = match i < DISTINCT_FIRST as u64 {
                        true => i.wrapping_mul(SCATTER),
                        false => i.wrapping_mul(SCATTER) >> 62,
                    };
                    let value = (pick + 1).wrapping_mul(SCATTER_2);
                    format!("{value:016x}{:016x}{value:016x}", !value)
                })
                .collect::<Vec<_>>(),
        );
        let nulls = arrow_array::new_null_array(&DataType::Int64, ROWS);
        let mut columns: Vec<ArrayRef> = vec![Arc::new(status)];
        columns.extend((1..COLUMNS).map(|_| Arc::clone(&nulls)));
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns).expect("batch");
        let plan = BodyPlan {
            row_group_size: ROWS,
            dictionary_candidates: vec!["status".to_string()],
            ..plan(&schema)
        };

        assert_eq!(
            plan.choose(&schema, &[batch.slice(0, DISTINCT_FIRST)])
                .expect("choose on the first rows"),
            vec!["status".to_string()],
            "the first rows alone favour plain values"
        );
        let body = encode_parquet_body_planned(&schema, slice::from_ref(&batch), plan)
            .expect("planned body");
        assert_eq!(
            dictionaries(&body)[0][0],
            ("status".to_string(), true),
            "measured over the row group, the column keeps its dictionary"
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

    /// A row group written as it arrives, rather than held and encoded
    /// whole, is cut at the cap the same way: its pages stay near the cap.
    #[test]
    fn a_streamed_row_group_keeps_pages_near_the_write_cap() {
        let batch = text_batch(&[BIG_PAGE_VALUE_BYTES; BIG_PAGE_ROWS]);
        let schema = batch.schema();
        let mut encoder = ParquetBodyEncoder::new(
            &schema,
            Compression::ZSTD(ZstdLevel::default()),
            // Longer than the batch, so its row group is still open after it.
            2 * BIG_PAGE_ROWS,
            &[],
        )
        .expect("encoder")
        .holding_at_most(1);
        encoder.write_batch(&batch).expect("write");
        assert!(
            encoder.open.is_some(),
            "past the bound the row group is written as it arrives"
        );
        let body = encoder.finish().expect("finish");
        let largest = largest_text_page(body, &schema);
        assert!(
            largest <= MAX_WRITE_BYTES + BIG_PAGE_VALUE_BYTES,
            "largest page {largest} bytes"
        );
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

    /// Columns of the wide test body.
    const WIDE_COLUMNS: usize = 300;

    /// One row in this many holds a value in a given wide column.
    const WIDE_FILL_EVERY: usize = 7;

    /// Rows per write to the wide encoder: not a divisor of the row group,
    /// so groups are cut mid-write.
    const WIDE_PIECE_ROWS: usize = 1_000;

    /// A body of [`WIDE_COLUMNS`] nullable string and list columns, each
    /// holding a value in one row in [`WIDE_FILL_EVERY`]; every row costs
    /// a few bytes in every column, null or not.
    fn wide_sparse(rows: usize) -> (Arc<Schema>, RecordBatch) {
        let item = Arc::new(Field::new("item", DataType::Utf8, true));
        let mut fields = Vec::with_capacity(WIDE_COLUMNS);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(WIDE_COLUMNS);
        for c in 0..WIDE_COLUMNS {
            let filled = |r: usize| (r + c).is_multiple_of(WIDE_FILL_EVERY);
            if c.is_multiple_of(2) {
                fields.push(Field::new(format!("s{c}"), DataType::LargeUtf8, true));
                columns.push(Arc::new(LargeStringArray::from_iter(
                    (0..rows).map(|r| filled(r).then(|| format!("{c}:{r}"))),
                )));
            } else {
                fields.push(Field::new(
                    format!("l{c}"),
                    DataType::List(Arc::clone(&item)),
                    true,
                ));
                let mut list = ListBuilder::new(StringBuilder::new()).with_field(Arc::clone(&item));
                for r in 0..rows {
                    if filled(r) {
                        list.values().append_value(format!("{c}"));
                        list.values().append_value(format!("{r}"));
                    }
                    list.append(filled(r));
                }
                columns.push(Arc::new(list.finish()));
            }
        }
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns).expect("wide batch");
        (schema, batch)
    }

    /// The wide body's footer-stripped bytes, encoded holding at most
    /// `bound` bytes, its rows kept by the caller or not; the most it held
    /// after any write; and whether it wrote a row group as it arrived.
    fn encode_wide(
        schema: &Arc<Schema>,
        batch: &RecordBatch,
        bound: usize,
        held_by_caller: bool,
    ) -> (Vec<u8>, usize, bool) {
        let mut encoder = ParquetBodyEncoder::new(
            schema,
            Compression::ZSTD(ZstdLevel::try_new(3).expect("valid zstd level")),
            TEST_GROUP_ROWS,
            &[],
        )
        .expect("encoder")
        .holding_at_most(bound);
        if held_by_caller {
            encoder = encoder.with_rows_held_by_caller();
        }
        let mut most_held = 0;
        let mut streamed = false;
        let mut start = 0;
        while start < batch.num_rows() {
            let len = WIDE_PIECE_ROWS.min(batch.num_rows() - start);
            encoder
                .write_batch(&batch.slice(start, len))
                .expect("write piece");
            most_held = most_held.max(encoder.held_bytes);
            streamed |= encoder.open.is_some();
            start += len;
        }
        let body = encoder.finish().expect("finish");
        let mut bytes = fs::read(body.body_file.path()).expect("read body");
        bytes.truncate(usize::try_from(body.body_len).expect("body length"));
        (bytes, most_held, streamed)
    }

    /// Rows of each write of the mixed body: a page is cut only where a
    /// write ends, so both encodes are fed the same writes.
    const MIXED_WRITE_ROWS: usize = 1_000;

    /// A mixed body: a sparse integer, a string, a list of strings, a
    /// struct of an integer and a string, and a dense id, `rows` rows.
    fn mixed(rows: usize) -> (Arc<Schema>, RecordBatch) {
        let item = Arc::new(Field::new("item", DataType::Utf8, true));
        let inner = Fields::from(vec![
            Field::new("n", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
        ]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("sparse", DataType::Int64, true),
            Field::new("text", DataType::LargeUtf8, true),
            Field::new("tags", DataType::List(Arc::clone(&item)), true),
            Field::new("pair", DataType::Struct(inner.clone()), true),
        ]));
        let mut tags = ListBuilder::new(StringBuilder::new()).with_field(Arc::clone(&item));
        for r in 0..rows {
            if r.is_multiple_of(3) {
                tags.values().append_value(format!("t{}", r % 11));
                tags.values().append_value(format!("u{}", r % 5));
            }
            tags.append(r.is_multiple_of(3));
        }
        let pair = StructArray::new(
            inner,
            vec![
                Arc::new(Int64Array::from_iter(
                    (0..rows).map(|r| r.is_multiple_of(4).then_some(r as i64)),
                )),
                Arc::new(StringArray::from_iter(
                    (0..rows).map(|r| r.is_multiple_of(2).then(|| format!("s{}", r % 9))),
                )),
            ],
            Some(NullBuffer::from_iter(
                (0..rows).map(|r| !r.is_multiple_of(5)),
            )),
        );
        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt64Array::from_iter_values(
                (0..rows as u64).map(|r| r.wrapping_mul(SCATTER)),
            )),
            Arc::new(Int64Array::from_iter(
                (0..rows).map(|r| r.is_multiple_of(13).then_some(r as i64)),
            )),
            Arc::new(LargeStringArray::from_iter(
                (0..rows).map(|r| (r % 2 == 1).then(|| format!("row {r} of the body"))),
            )),
            Arc::new(tags.finish()),
            Arc::new(pair),
        ];
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns).expect("mixed batch");
        (schema, batch)
    }

    /// The bytes `ArrowWriter` writes for `batch`, written in slices of
    /// [`MIXED_WRITE_ROWS`] rows, up to the end of its last row group, under
    /// the body's writer properties.
    fn arrow_writer_body(
        schema: &Arc<Schema>,
        batch: &RecordBatch,
        plain_columns: &[String],
    ) -> Vec<u8> {
        let props = writer_properties(
            Compression::ZSTD(ZstdLevel::try_new(3).expect("valid zstd level")),
            TEST_GROUP_ROWS,
            &[],
            plain_columns,
        );
        let mut bytes = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut bytes, Arc::clone(schema), Some(props)).expect("writer");
        for start in (0..batch.num_rows()).step_by(MIXED_WRITE_ROWS) {
            let len = MIXED_WRITE_ROWS.min(batch.num_rows() - start);
            writer.write(&batch.slice(start, len)).expect("write");
        }
        let meta = writer.close().expect("close");
        let end = meta
            .row_groups()
            .iter()
            .flat_map(|rg| rg.columns())
            .map(|c| {
                let (start, len) = c.byte_range();
                start + len
            })
            .max()
            .expect("a column chunk");
        bytes.truncate(usize::try_from(end).expect("body end"));
        bytes
    }

    /// A body whose columns are encoded one at a time by writers made for
    /// each alone, side by side or in turn, is the bytes `ArrowWriter`
    /// writes with every column's writer open at once: nested and struct
    /// columns included, with and without dictionaries.
    #[test]
    fn a_body_encoded_a_column_at_a_time_is_the_bytes_arrow_writer_writes() {
        let (schema, batch) = mixed(3 * TEST_GROUP_ROWS + 123);
        for plain in [Vec::new(), vec!["id".to_string(), "text".to_string()]] {
            let want = arrow_writer_body(&schema, &batch, &plain);
            for together in [true, false] {
                let mut encoder = ParquetBodyEncoder::with_plain_columns(
                    &schema,
                    Compression::ZSTD(ZstdLevel::try_new(3).expect("valid zstd level")),
                    TEST_GROUP_ROWS,
                    &[],
                    &plain,
                    together,
                    &scratch_root(),
                )
                .expect("encoder");
                for start in (0..batch.num_rows()).step_by(MIXED_WRITE_ROWS) {
                    let len = MIXED_WRITE_ROWS.min(batch.num_rows() - start);
                    encoder
                        .write_batch(&batch.slice(start, len))
                        .expect("write");
                }
                let body = encoder.finish().expect("finish");
                let mut got = fs::read(body.body_file.path()).expect("read body");
                // The body keeps the page index after its row groups.
                got.truncate(want.len());
                assert_eq!(body.metadata.num_row_groups(), 4);
                assert!(
                    got == want,
                    "together={together} plain={plain:?}: {} bytes against {}",
                    got.len(),
                    want.len()
                );
            }
        }
    }

    /// `group`, rows `group_start..` of a body written in pieces of
    /// [`MIXED_WRITE_ROWS`] rows from its first, handed to `write` cut where
    /// those pieces are: the writes the body's row group gets when the whole
    /// body is written in those pieces.
    fn write_group_in_pieces(
        group: &RecordBatch,
        group_start: usize,
        mut write: impl FnMut(&RecordBatch) -> Result<(), FooterError>,
    ) -> Result<(), FooterError> {
        let end = group_start + group.num_rows();
        let mut at = group_start;
        while at < end {
            let next = end.min((at / MIXED_WRITE_ROWS + 1) * MIXED_WRITE_ROWS);
            write(&group.slice(at - group_start, next - at))?;
            at = next;
        }
        Ok(())
    }

    /// `batch` written by a [`PassBodyEncoder`] under `plan`, `width` fields
    /// at a time, a row group of [`TEST_GROUP_ROWS`] rows at a time, each
    /// pass handed its fields in pieces of [`MIXED_WRITE_ROWS`] rows.
    fn write_in_passes(
        schema: &Arc<Schema>,
        batch: &RecordBatch,
        plan: BodyPlan,
        width: usize,
    ) -> EncodedBody {
        let fields = schema.fields().len();
        let passes: Vec<Range<usize>> = (0..fields)
            .step_by(width)
            .map(|start| start..(start + width).min(fields))
            .collect();
        let mut encoder = PassBodyEncoder::new(schema, plan).expect("encoder");
        for start in (0..batch.num_rows()).step_by(TEST_GROUP_ROWS) {
            let group = batch.slice(start, TEST_GROUP_ROWS.min(batch.num_rows() - start));
            encoder
                .write_row_group(&passes, |pass, writer| -> Result<(), FooterError> {
                    let part = group
                        .project(&pass.collect::<Vec<_>>())
                        .expect("project the pass");
                    write_group_in_pieces(&part, start, |piece| writer.write(piece))
                })
                .expect("row group");
        }
        encoder.finish().expect("finish")
    }

    /// A body written a pass of fields at a time is the bytes `ArrowWriter`
    /// writes for the same rows written in the same pieces with every
    /// column's writer open at once, nested and struct columns included,
    /// whatever the pass width.
    #[test]
    fn a_body_written_in_passes_is_the_bytes_arrow_writer_writes() {
        let (schema, batch) = mixed(3 * TEST_GROUP_ROWS + 123);
        let want = arrow_writer_body(&schema, &batch, &[]);
        let plan = BodyPlan {
            dictionary_candidates: Vec::new(),
            ..plan(&schema)
        };
        for width in 1..=schema.fields().len() {
            let body = write_in_passes(&schema, &batch, plan.clone(), width);
            assert_eq!(body.metadata.num_row_groups(), 4);
            let mut got = fs::read(body.body_file.path()).expect("read body");
            got.truncate(want.len());
            assert!(got == want, "{width} fields at a time");
        }
    }

    /// A body written in passes chooses each column's dictionary as one
    /// written by rows chooses it, from the first row group, so the two
    /// bodies are the same bytes whatever the pass width.
    #[test]
    fn a_body_written_in_passes_chooses_the_dictionaries_rows_choose() {
        let (schema, batch) = mixed(3 * TEST_GROUP_ROWS + 123);
        let mut rows = PlannedBodyEncoder::new(&schema, plan(&schema));
        for start in (0..batch.num_rows()).step_by(MIXED_WRITE_ROWS) {
            let len = MIXED_WRITE_ROWS.min(batch.num_rows() - start);
            rows.write_batch(&batch.slice(start, len)).expect("write");
        }
        let rows = rows.finish().expect("finish");
        let want = fs::read(rows.body_file.path()).expect("read body");
        for width in 1..=schema.fields().len() {
            let body = write_in_passes(&schema, &batch, plan(&schema), width);
            let got = fs::read(body.body_file.path()).expect("read body");
            assert!(got == want, "{width} fields at a time");
        }
    }

    /// A pass's rows must be a row group's: a pass handed fewer or more
    /// rows than the passes before it, or other columns than its fields,
    /// fails the row group rather than writing a malformed body.
    #[test]
    fn a_pass_of_other_rows_or_columns_fails_its_row_group() {
        let (schema, batch) = mixed(TEST_GROUP_ROWS);
        let passes = [0..2, 2..schema.fields().len()];
        let mut encoder = PassBodyEncoder::new(&schema, plan(&schema)).expect("encoder");
        let short = encoder.write_row_group(&passes, |pass, writer| {
            let rows = match pass.start {
                0 => batch.num_rows(),
                _ => batch.num_rows() - 1,
            };
            writer.write(
                &batch
                    .slice(0, rows)
                    .project(&pass.collect::<Vec<_>>())
                    .expect("project"),
            )
        });
        assert!(short.is_err(), "a pass of fewer rows fails");
        let mut encoder = PassBodyEncoder::new(&schema, plan(&schema)).expect("encoder");
        let wide = encoder.write_row_group(&passes, |_, writer| writer.write(&batch));
        assert!(wide.is_err(), "a pass of other columns fails");
        let mut encoder = PassBodyEncoder::new(&schema, plan(&schema)).expect("encoder");
        let gap = encoder.write_row_group(&[0..1, 2..schema.fields().len()], |_, _| {
            Ok::<(), FooterError>(())
        });
        assert!(gap.is_err(), "passes that skip a field fail");
    }

    /// The sizes the dictionary choice is made on, each candidate encoded on
    /// its own, are the column chunk sizes of an `ArrowWriter` encode of the
    /// whole sample, with and without the dictionaries.
    #[test]
    fn a_column_measured_alone_is_its_size_in_an_encode_of_the_whole_sample() {
        let (schema, batch) = mixed(TEST_GROUP_ROWS);
        let mut plan = plan(&schema);
        plan.dictionary_candidates = ["id", "sparse", "text"].map(String::from).to_vec();
        let whole = |plain: &[String]| -> ColumnSizes {
            let props = writer_properties(plan.compression, plan.row_group_size, &[], plain);
            let mut writer =
                ArrowWriter::try_new(io::sink(), Arc::clone(&schema), Some(props)).expect("writer");
            writer.write(&batch).expect("write");
            let meta = writer.close().expect("close");
            let mut sizes = ColumnSizes::new();
            for chunk in meta.row_groups().iter().flat_map(|rg| rg.columns()) {
                let column = chunk.column_path().parts()[0].clone();
                if plan.dictionary_candidates.contains(&column) {
                    *sizes.entry(column).or_default() += chunk.compressed_size() as u64;
                }
            }
            sizes
        };
        for together in [true, false] {
            plan.encode_groups_together = together;
            let (with, without) = plan
                .measure(&schema, slice::from_ref(&batch))
                .expect("measure");
            assert_eq!(with, whole(&[]), "together={together}");
            assert_eq!(
                without,
                whole(&plan.dictionary_candidates),
                "together={together}"
            );
        }
    }

    /// A body whose held rows would decode past the bound writes its row
    /// group as the rows arrive instead of holding them, holds no more than
    /// the bound and one write's rows, and writes exactly the bytes it
    /// writes when it holds every group.
    #[test]
    fn a_wide_body_is_written_as_it_arrives_and_its_bytes_do_not_change() {
        let (schema, batch) = wide_sparse(3 * TEST_GROUP_ROWS + TEST_GROUP_ROWS / 2);
        let piece = decoded_bytes(&batch.slice(0, WIDE_PIECE_ROWS));
        let bound = 2 * piece;

        let (held, held_most, held_streamed) = encode_wide(&schema, &batch, usize::MAX, false);
        let (bounded, bounded_most, bounded_streamed) = encode_wide(&schema, &batch, bound, false);
        let (kept, kept_most, kept_streamed) = encode_wide(&schema, &batch, bound, true);

        assert!(!held_streamed, "without a bound every group is held");
        assert!(
            bounded_streamed,
            "past the bound the group is written as it arrives"
        );
        assert!(
            bounded_most < bound + piece,
            "held {bounded_most} bytes against a bound of {bound}"
        );
        assert!(
            held_most > 4 * bounded_most,
            "the unbounded encoder held {held_most}"
        );
        assert_eq!(held, bounded, "the bytes do not depend on what was held");
        assert!(
            !kept_streamed,
            "rows the caller keeps are held, the group being filled included"
        );
        assert!(
            kept_most < held_most,
            "past the bound the full groups are encoded: held {kept_most} of {held_most}"
        );
        assert_eq!(held, kept, "nor on whether the caller keeps the rows");
    }
}
