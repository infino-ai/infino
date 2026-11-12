// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Loading a superfile's page index for the columns a query reads.
//!
//! The page index holds two things per column chunk: the offset index (where
//! each page is, which a read of the column needs to skip pages) and the column
//! index (each page's min and max, which a filter uses to skip pages). Decoded
//! for every column, it costs memory that grows with files x columns, whatever
//! a query reads: tens of MB a file on a table of thousands of columns. So a
//! reader decodes it only for the columns asked for, and adds columns when a
//! later caller asks for more. A column not loaded gets:
//!
//! - column index `NONE`, which skips no page, so pruning never changes a
//!   result because of it;
//! - an offset index that fails any read of the column
//!   ([`unloaded_offset_index`]). An empty one would not do: arrow-rs reads a
//!   column with no page locations as zero rows. Every path that reads pages
//!   asks for its columns first, so this only turns a missed ask into an
//!   error instead of wrong rows.
//!
//! Columns are matched by Parquet field id, then by name. A file written
//! with ids finds a column by id after a rename. A file written before ids
//! holds a column under the name it was written with, so a caller passes
//! every name the column may be stored under (the SQL scan passes the
//! current name and the one the table was created with, as it resolves
//! such a file). A name now used by another column can match that one too,
//! which loads an index nothing reads, never a wrong one.
//!
//! ```text
//!   [data pages][column index][offset index][fts][vec][ids][footer]
//!               |<--- first load reads ---->|   kept as bytes
//!                 ^^   ^^      ^^      ^^       decoded: asked columns only
//! ```
//!
//! The first load reads the whole page index, the same read as when every
//! column was decoded, and the reader keeps its bytes, so a later load decodes
//! from memory and never reads storage again.

use std::{collections::HashSet, mem::size_of, ops::Range, sync::Arc};

use bytes::Bytes;
use parquet::{
    DecodeResult,
    basic::Repetition,
    errors::{ParquetError, Result as ParquetResult},
    file::{
        metadata::{
            ColumnChunkMetaData, FileMetaData, PageIndexPolicy, ParquetColumnIndex,
            ParquetMetaData, ParquetMetaDataBuilder, ParquetMetaDataPushDecoder,
            ParquetOffsetIndex, RowGroupMetaData,
        },
        page_index::{
            column_index::ColumnIndexMetaData,
            offset_index::{OffsetIndexMetaData, PageLocation},
        },
    },
    schema::types::{SchemaDescriptor, Type},
};
use roaring::RoaringBitmap;

/// Name of the group holding the leaves a load decodes (see [`leaves_only`]).
const DECODE_ROOT: &str = "page_index";

/// The offset index a column not loaded gets ([`unloaded_offset_index`]):
/// one page of this size at the chunk's own start. No real page is empty
/// (each has a header), so it also marks the entry as not loaded
/// ([`is_loaded`]).
const UNLOADED_PAGE_SIZE: i32 = 0;

/// The memory one column chunk not loaded takes in a footer, as arrow-rs's
/// `memory_size()` counts it: its `NONE` column index, its offset index and
/// that index's one page location ([`unloaded_offset_index`]). A reader
/// subtracts it to tell the page index it decoded from the placeholders
/// around it.
pub(crate) const UNLOADED_CHUNK_BYTES: u64 = (size_of::<ColumnIndexMetaData>()
    + size_of::<OffsetIndexMetaData>()
    + size_of::<PageLocation>()) as u64;

/// Which top-level columns' page index a load brings in.
#[derive(Clone, Debug)]
pub(crate) enum IndexColumns {
    /// Every column.
    All,
    /// The columns of a set: a scan's filter and read columns, a take's
    /// projection, or none for a load that only reads the index into memory.
    Only(Arc<IndexColumnSet>),
}

/// Top-level columns by Parquet field id or by name (see the module docs).
#[derive(Debug, Default)]
pub(crate) struct IndexColumnSet {
    ids: HashSet<i32>,
    names: HashSet<String>,
}

impl IndexColumns {
    /// No column.
    pub(crate) fn none() -> Self {
        Self::Only(Arc::default())
    }

    /// The columns named, each with its field id when it has one.
    pub(crate) fn of<'a>(columns: impl IntoIterator<Item = (Option<i32>, &'a str)>) -> Self {
        let mut set = IndexColumnSet::default();
        for (id, name) in columns {
            set.ids.extend(id);
            set.names.insert(name.to_owned());
        }
        Self::Only(Arc::new(set))
    }

    /// The columns named, by name alone: for a caller reading a file under
    /// the file's own names.
    pub(crate) fn named<S: AsRef<str>>(names: &[S]) -> Self {
        Self::of(names.iter().map(|name| (None, name.as_ref())))
    }

    /// The leaves of `schema` under these columns (a struct or list column
    /// has several).
    pub(crate) fn leaves(&self, schema: &SchemaDescriptor) -> RoaringBitmap {
        let leaves = (0..schema.num_columns()).map(|leaf| leaf as u32);
        match self {
            Self::All => leaves.collect(),
            Self::Only(set) => leaves
                .filter(|&leaf| {
                    let root = schema.get_column_root(leaf as usize);
                    let info = root.get_basic_info();
                    (info.has_id() && set.ids.contains(&info.id()))
                        || set.names.contains(root.name())
                })
                .collect(),
        }
    }

    /// Whether these columns include every column in `names`, by name.
    pub(crate) fn covers_names(&self, names: &[&str]) -> bool {
        match self {
            Self::All => true,
            Self::Only(set) => names.iter().all(|name| set.names.contains(*name)),
        }
    }

    /// Whether a load for `self` brings in everything one for `other` does.
    pub(crate) fn covers(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::All, _) => true,
            (Self::Only(_), Self::All) => false,
            (Self::Only(mine), Self::Only(theirs)) => {
                theirs.ids.is_subset(&mine.ids) && theirs.names.is_subset(&mine.names)
            }
        }
    }

    /// The columns of both.
    pub(crate) fn union(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Only(mine), Self::Only(theirs)) => Self::Only(Arc::new(IndexColumnSet {
                ids: &mine.ids | &theirs.ids,
                names: &mine.names | &theirs.names,
            })),
            _ => Self::All,
        }
    }
}

/// A file's whole page index, as raw bytes: what a first load reads, kept so
/// later loads decode from memory. Cloning shares the bytes.
#[derive(Clone)]
pub(crate) struct PageIndexBytes {
    /// File offset of the first byte.
    start: u64,
    bytes: Bytes,
    /// Whether these bytes are a copy of their own ([`Self::copied`]) rather
    /// than a slice of bytes the reader holds anyway ([`Self::slice`]).
    copy: bool,
}

impl PageIndexBytes {
    /// The bytes of `parquet`'s whole page index, every column chunk's column
    /// and offset index; `None` when the file has none.
    pub(crate) fn span(parquet: &ParquetMetaData) -> Option<Range<u64>> {
        span_of(chunks(parquet).flat_map(|(_, chunk)| index_ranges(chunk)))
    }

    /// How many bytes of page index these are.
    pub(crate) fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// The memory keeping these bytes takes: their length when they are a
    /// copy, nothing when they are a slice of a resident file.
    pub(crate) fn held_len(&self) -> u64 {
        if self.copy { self.len() } else { 0 }
    }

    /// `span`'s bytes, out of `read`, which starts at file offset `start`;
    /// `None` when `read` doesn't hold all of `span`. A slice, so it shares
    /// `read`'s buffer: right for a resident file, whose bytes stay anyway.
    pub(crate) fn slice(span: &Range<u64>, start: u64, read: &Bytes) -> Option<Self> {
        Some(Self {
            start: span.start,
            bytes: slice_at(read, start, span)?,
            copy: false,
        })
    }

    /// [`Self::slice`], copied, so the buffer `read` came in (a cache block
    /// around it) is freed.
    pub(crate) fn copied(span: &Range<u64>, start: u64, read: &Bytes) -> Option<Self> {
        let kept = Self::slice(span, start, read)?;
        Some(Self {
            start: kept.start,
            bytes: Bytes::copy_from_slice(&kept.bytes),
            copy: true,
        })
    }

    /// `parquet` with the page index of the leaves in `added` decoded and the
    /// rest as `parquet` holds it: loaded where an earlier load decoded it,
    /// [`unloaded_offset_index`] and `NONE` elsewhere. arrow-rs owns the
    /// footer by value, so this is a new copy of it.
    ///
    /// When a chunk of `added` has no offset index, the result has no page
    /// index at all: arrow-rs drops both parts then too, since without page
    /// locations there is nothing to skip with, and its readers read whole
    /// chunks.
    pub(crate) fn apply(
        &self,
        parquet: &ParquetMetaData,
        added: &RoaringBitmap,
    ) -> ParquetResult<ParquetMetaData> {
        let (fresh_columns, fresh_offsets) = self.decode(parquet, added)?;
        let mut indexed = parquet.clone().into_builder();
        let held_columns = indexed.take_column_index();
        let held_offsets = indexed.take_offset_index();
        let Some(fresh_offsets) = fresh_offsets else {
            let indexed = indexed
                .set_column_index(None)
                .set_offset_index(None)
                .build();
            return Ok(indexed);
        };
        let offsets = merge(
            parquet,
            held_offsets,
            Some(fresh_offsets),
            added,
            unloaded_offset_index,
        );
        let columns = merge(parquet, held_columns, fresh_columns, added, |_| {
            ColumnIndexMetaData::NONE
        });
        let indexed = indexed
            .set_column_index(Some(columns))
            .set_offset_index(Some(offsets))
            .build();
        Ok(indexed)
    }

    /// The column and offset index of the leaves in `leaves`, decoded from
    /// these bytes; each row group's entries are in leaf order. arrow-rs
    /// decodes the page index of every chunk of a footer, and drops the whole
    /// offset index when any chunk lacks one, so it is handed a footer of
    /// just these leaves ([`leaves_only`]).
    fn decode(
        &self,
        parquet: &ParquetMetaData,
        leaves: &RoaringBitmap,
    ) -> ParquetResult<(Option<ParquetColumnIndex>, Option<ParquetOffsetIndex>)> {
        if leaves.is_empty() {
            let row_groups = parquet.num_row_groups();
            return Ok((
                Some(vec![Vec::new(); row_groups]),
                Some(vec![Vec::new(); row_groups]),
            ));
        }
        // The file's size only matters for reading its footer, done already.
        let mut decoder = ParquetMetaDataPushDecoder::try_new_with_metadata(
            u64::MAX,
            leaves_only(parquet, leaves)?,
        )?
        .with_column_index_policy(PageIndexPolicy::Optional)
        .with_offset_index_policy(PageIndexPolicy::Optional);
        let end = self.start + self.bytes.len() as u64;
        decoder.push_range(self.start..end, self.bytes.clone())?;
        match decoder.try_decode()? {
            DecodeResult::Data(decoded) => {
                let mut decoded = decoded.into_builder();
                Ok((decoded.take_column_index(), decoded.take_offset_index()))
            }
            other => Err(ParquetError::General(format!(
                "page index decode over {}..{end} did not finish: {other:?}",
                self.start
            ))),
        }
    }
}

/// The bytes of page index (column and offset index) the leaves in `leaves`
/// hold in `parquet`, which a decode of them reads.
pub(crate) fn index_len(parquet: &ParquetMetaData, leaves: &RoaringBitmap) -> u64 {
    chunks(parquet)
        .filter(|(leaf, _)| leaves.contains(*leaf))
        .flat_map(|(_, chunk)| index_ranges(chunk))
        .map(|range| range.end - range.start)
        .sum()
}

/// The bytes of a column chunk's page index: its column index, then its
/// offset index, each when the chunk has one.
fn index_ranges(chunk: &ColumnChunkMetaData) -> impl Iterator<Item = Range<u64>> {
    column_index_range(chunk)
        .into_iter()
        .chain(offset_index_range(chunk))
}

/// The offset index a column chunk not loaded gets: one empty page at the
/// chunk's start. Any read of the column fails, all of it or a selection:
/// arrow-rs reads that page's header out of zero bytes. An empty offset index
/// would not do, since arrow-rs reads a column with no pages as zero rows.
/// The page starts where the chunk does, so arrow-rs infers no dictionary
/// page before it, whatever the file's size.
fn unloaded_offset_index(chunk: &ColumnChunkMetaData) -> OffsetIndexMetaData {
    // Where arrow-rs starts the chunk (`ColumnChunkMetaData::byte_range`).
    let start = chunk
        .dictionary_page_offset()
        .unwrap_or_else(|| chunk.data_page_offset());
    OffsetIndexMetaData {
        page_locations: vec![PageLocation {
            offset: start,
            compressed_page_size: UNLOADED_PAGE_SIZE,
            first_row_index: 0,
        }],
        unencoded_byte_array_data_bytes: None,
    }
}

/// Whether `index` is a column's real offset index, not
/// [`unloaded_offset_index`]. Callers outside this module take page
/// locations through [`loaded_pages`], which checks it.
pub(in crate::superfile) fn is_loaded(index: &OffsetIndexMetaData) -> bool {
    !matches!(index.page_locations().as_slice(), [page] if page.compressed_page_size == UNLOADED_PAGE_SIZE)
}

/// The page locations of leaf `leaf` in row group `row_group` of `parquet`,
/// or `None` when the footer holds none for it: no page index, or a
/// placeholder for a column not loaded ([`unloaded_offset_index`]), whose
/// single page would count wrong and fails a read. Every caller that
/// counts or reads pages from a footer takes them here.
pub(crate) fn loaded_pages(
    parquet: &ParquetMetaData,
    row_group: usize,
    leaf: usize,
) -> Option<&[PageLocation]> {
    let index = parquet.offset_index()?.get(row_group)?.get(leaf)?;
    is_loaded(index).then(|| index.page_locations().as_slice())
}

/// `parquet` cut down to the leaves in `leaves`, for the page-index decoder:
/// each leaf a flat column of the same physical type, which is all the decode
/// of an index needs, with its chunks' index locations and nothing else
/// (statistics, key-value metadata), so the copy stays small on a wide table.
fn leaves_only(
    parquet: &ParquetMetaData,
    leaves: &RoaringBitmap,
) -> ParquetResult<ParquetMetaData> {
    let schema = parquet.file_metadata().schema_descr();
    let fields = leaves
        .iter()
        .map(|leaf| {
            let column = schema.column(leaf as usize);
            Type::primitive_type_builder(&leaf.to_string(), column.physical_type())
                .with_repetition(Repetition::OPTIONAL)
                .with_length(column.type_length())
                .build()
                .map(Arc::new)
        })
        .collect::<ParquetResult<Vec<_>>>()?;
    let root = Type::group_type_builder(DECODE_ROOT)
        .with_fields(fields)
        .build()?;
    let schema = Arc::new(SchemaDescriptor::new(Arc::new(root)));
    let file = parquet.file_metadata();
    let file = FileMetaData::new(
        file.version(),
        file.num_rows(),
        None,
        None,
        Arc::clone(&schema),
        None,
    );
    let row_groups = parquet
        .row_groups()
        .iter()
        .map(|row_group| {
            let columns = leaves
                .iter()
                .enumerate()
                .map(|(at, leaf)| {
                    let chunk = row_group.column(leaf as usize);
                    ColumnChunkMetaData::builder(schema.column(at))
                        .set_column_index_offset(chunk.column_index_offset())
                        .set_column_index_length(chunk.column_index_length())
                        .set_offset_index_offset(chunk.offset_index_offset())
                        .set_offset_index_length(chunk.offset_index_length())
                        .build()
                })
                .collect::<ParquetResult<Vec<_>>>()?;
            RowGroupMetaData::builder(Arc::clone(&schema))
                .set_num_rows(row_group.num_rows())
                .set_column_metadata(columns)
                .build()
        })
        .collect::<ParquetResult<Vec<_>>>()?;
    Ok(ParquetMetaDataBuilder::new(file)
        .set_row_groups(row_groups)
        .build())
}

/// Every leaf's entry of every row group of `parquet`: `fresh`'s, in leaf
/// order, for the leaves in `added`, `held`'s for the rest, and `unloaded`
/// of the chunk where neither has one. `held` is stepped for every leaf,
/// added or not, so its entries stay with their leaves. Entries are moved,
/// not copied.
fn merge<T>(
    parquet: &ParquetMetaData,
    held: Option<Vec<Vec<T>>>,
    fresh: Option<Vec<Vec<T>>>,
    added: &RoaringBitmap,
    unloaded: impl Fn(&ColumnChunkMetaData) -> T,
) -> Vec<Vec<T>> {
    let mut held = held.unwrap_or_default().into_iter();
    let mut fresh = fresh.unwrap_or_default().into_iter();
    parquet
        .row_groups()
        .iter()
        .map(|row_group| {
            let mut held = held.next().unwrap_or_default().into_iter();
            let mut fresh = fresh.next().unwrap_or_default().into_iter();
            (0u32..)
                .zip(row_group.columns())
                .map(|(leaf, chunk)| {
                    let held = held.next();
                    let entry = if added.contains(leaf) {
                        fresh.next()
                    } else {
                        held
                    };
                    entry.unwrap_or_else(|| unloaded(chunk))
                })
                .collect()
        })
        .collect()
}

/// The bytes of file `span` out of `bytes`, which start at file offset
/// `start`; `None` when `bytes` doesn't hold all of `span`.
fn slice_at(bytes: &Bytes, start: u64, span: &Range<u64>) -> Option<Bytes> {
    let from = usize::try_from(span.start.checked_sub(start)?).ok()?;
    let to = usize::try_from(span.end.checked_sub(start)?).ok()?;
    (from <= to && to <= bytes.len()).then(|| bytes.slice(from..to))
}

/// Every column chunk of `parquet`, with its leaf.
fn chunks(parquet: &ParquetMetaData) -> impl Iterator<Item = (u32, &ColumnChunkMetaData)> {
    parquet
        .row_groups()
        .iter()
        .flat_map(|row_group| (0u32..).zip(row_group.columns()))
}

/// The smallest span holding every one of `ranges`.
fn span_of(ranges: impl Iterator<Item = Range<u64>>) -> Option<Range<u64>> {
    ranges.reduce(|span, range| span.start.min(range.start)..span.end.max(range.end))
}

/// The bytes of a column chunk's column index.
fn column_index_range(chunk: &ColumnChunkMetaData) -> Option<Range<u64>> {
    index_range(chunk.column_index_offset(), chunk.column_index_length())
}

/// The bytes of a column chunk's offset index.
fn offset_index_range(chunk: &ColumnChunkMetaData) -> Option<Range<u64>> {
    index_range(chunk.offset_index_offset(), chunk.offset_index_length())
}

/// The bytes of a chunk's index, from its recorded offset and length.
fn index_range(offset: Option<i64>, length: Option<i32>) -> Option<Range<u64>> {
    let offset = u64::try_from(offset?).ok()?;
    let length = u64::try_from(length?).ok()?;
    Some(offset..offset + length)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use arrow_array::{
        ArrayRef, Int64Array, ListArray, RecordBatch, StructArray, types::Int64Type,
    };
    use arrow_schema::{DataType, Field, Fields, Schema};
    use futures::TryStreamExt;
    use parquet::{
        arrow::{
            ArrowWriter, PARQUET_FIELD_ID_META_KEY, ParquetRecordBatchStreamBuilder,
            ProjectionMask,
            arrow_reader::{
                ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
                RowSelection, RowSelector,
            },
        },
        file::{
            metadata::ParquetMetaDataReader,
            properties::{EnabledStatistics, WriterProperties, WriterPropertiesBuilder},
        },
        schema::types::ColumnPath,
    };

    use super::*;

    /// Rows per row group in the fixture.
    const ROWS_PER_GROUP: usize = 64;
    /// Rows per data page, so each column chunk has several pages.
    const ROWS_PER_PAGE: usize = 16;
    /// Row groups in the fixture.
    const GROUPS: usize = 2;
    /// Rows a selected read of the fixture skips first.
    const SKIPPED: usize = 20;
    /// Rows a selected read of the fixture returns.
    const SELECTED: usize = 10;
    /// The field id of the column renamed in the id-matching test.
    const RENAMED_FIELD_ID: i32 = 7;

    /// A Parquet file of three `Int64` columns `a`, `b`, `c` (leaves 0, 1,
    /// 2), several pages a chunk, with its page index; and its footer
    /// parsed without one.
    fn fixture() -> (Bytes, ParquetMetaData) {
        written(abc(), paged().build())
    }

    /// The fixture's rows: `a`, `b`, `c` hold each row number times 1, 2, 3.
    fn abc() -> RecordBatch {
        let columns = ["a", "b", "c"];
        let schema = Schema::new(
            columns
                .map(|name| Field::new(name, DataType::Int64, false))
                .to_vec(),
        );
        let column = |scale: i64| {
            Arc::new(Int64Array::from_iter_values(rows().map(|r| r * scale))) as ArrayRef
        };
        RecordBatch::try_new(Arc::new(schema), vec![column(1), column(2), column(3)])
            .expect("batch")
    }

    /// The fixture's row numbers.
    fn rows() -> Range<i64> {
        0..(ROWS_PER_GROUP * GROUPS) as i64
    }

    /// Writer settings giving each column chunk several pages.
    fn paged() -> WriterPropertiesBuilder {
        WriterProperties::builder()
            .set_max_row_group_row_count(Some(ROWS_PER_GROUP))
            .set_data_page_row_count_limit(ROWS_PER_PAGE)
            .set_write_batch_size(ROWS_PER_PAGE)
    }

    /// `batch` written as a Parquet file with `props`, and its footer parsed
    /// without the page index.
    fn written(batch: RecordBatch, props: WriterProperties) -> (Bytes, ParquetMetaData) {
        let mut file = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut file, batch.schema(), Some(props)).expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close");
        let bytes = Bytes::from(file);
        let footer = ParquetMetaDataReader::new()
            .parse_and_finish(&bytes)
            .expect("footer");
        (bytes, footer)
    }

    /// `footer` with the whole page index, as arrow-rs loads it.
    fn full(bytes: &Bytes, footer: &ParquetMetaData) -> ParquetMetaData {
        let mut loader = ParquetMetaDataReader::new_with_metadata(footer.clone())
            .with_page_index_policy(PageIndexPolicy::Required);
        loader.read_page_indexes(bytes).expect("page index");
        loader.finish().expect("finish")
    }

    /// The fixture's whole page index, as a first load keeps it.
    fn kept(bytes: &Bytes, footer: &ParquetMetaData) -> PageIndexBytes {
        let span = PageIndexBytes::span(footer).expect("a page index");
        PageIndexBytes::copied(&span, 0, bytes).expect("in the file")
    }

    /// `parquet` with the page index of `leaves` added.
    fn load(index: &PageIndexBytes, parquet: &ParquetMetaData, leaves: &[u32]) -> ParquetMetaData {
        index
            .apply(parquet, &leaves.iter().copied().collect())
            .expect("apply")
    }

    /// Which leaves of each row group carry a column index.
    fn loaded_leaves(parquet: &ParquetMetaData) -> Vec<Vec<usize>> {
        parquet
            .column_index()
            .expect("column index")
            .iter()
            .map(|row_group| {
                row_group
                    .iter()
                    .enumerate()
                    .filter(|(_, index)| !matches!(index, ColumnIndexMetaData::NONE))
                    .map(|(leaf, _)| leaf)
                    .collect()
            })
            .collect()
    }

    /// Which leaves of each row group carry their page locations.
    fn located_leaves(parquet: &ParquetMetaData) -> Vec<Vec<usize>> {
        parquet
            .offset_index()
            .expect("offset index")
            .iter()
            .map(|row_group| {
                row_group
                    .iter()
                    .enumerate()
                    .filter(|(_, index)| is_loaded(index))
                    .map(|(leaf, _)| leaf)
                    .collect()
            })
            .collect()
    }

    /// Asserts `parquet` holds `full`'s page index for `leaf` in every row
    /// group.
    fn assert_leaf_matches(parquet: &ParquetMetaData, full: &ParquetMetaData, leaf: usize) {
        for row_group in 0..GROUPS {
            assert_eq!(
                parquet.column_index().expect("column index")[row_group][leaf],
                full.column_index().expect("column index")[row_group][leaf],
                "column index, row group {row_group} leaf {leaf}"
            );
            assert_eq!(
                parquet.offset_index().expect("offset index")[row_group][leaf],
                full.offset_index().expect("offset index")[row_group][leaf],
                "offset index, row group {row_group} leaf {leaf}"
            );
        }
    }

    /// Rows a read of `leaf` returns through `parquet`, all of them or
    /// `SELECTED` of them past the first `SKIPPED`.
    fn read_leaf(
        bytes: &Bytes,
        parquet: ParquetMetaData,
        leaf: usize,
        selected: bool,
    ) -> Result<usize, String> {
        let (arrow, mask) = leaf_read(parquet, leaf)?;
        let mut builder = ParquetRecordBatchReaderBuilder::new_with_metadata(bytes.clone(), arrow)
            .with_projection(mask);
        if selected {
            builder = builder.with_row_selection(selection());
        }
        let mut read = 0;
        for batch in builder.build().map_err(|e| e.to_string())? {
            read += batch.map_err(|e| e.to_string())?.num_rows();
        }
        Ok(read)
    }

    /// [`read_leaf`] through the async reader DataFusion and the byte-source
    /// paths read with.
    async fn read_leaf_async(
        bytes: &Bytes,
        parquet: ParquetMetaData,
        leaf: usize,
        selected: bool,
    ) -> Result<usize, String> {
        let (arrow, mask) = leaf_read(parquet, leaf)?;
        let mut builder =
            ParquetRecordBatchStreamBuilder::new_with_metadata(Cursor::new(bytes.to_vec()), arrow)
                .with_projection(mask);
        if selected {
            builder = builder.with_row_selection(selection());
        }
        let batches: Vec<RecordBatch> = builder
            .build()
            .map_err(|e| e.to_string())?
            .try_collect()
            .await
            .map_err(|e| e.to_string())?;
        Ok(batches.iter().map(RecordBatch::num_rows).sum())
    }

    /// The Arrow metadata of `parquet` and the projection of leaf `leaf`.
    fn leaf_read(
        parquet: ParquetMetaData,
        leaf: usize,
    ) -> Result<(ArrowReaderMetadata, ProjectionMask), String> {
        let arrow = ArrowReaderMetadata::try_new(Arc::new(parquet), ArrowReaderOptions::new())
            .map_err(|e| e.to_string())?;
        let mask = ProjectionMask::leaves(arrow.metadata().file_metadata().schema_descr(), [leaf]);
        Ok((arrow, mask))
    }

    /// The fixture's selected read: [`SELECTED`] rows after [`SKIPPED`].
    fn selection() -> RowSelection {
        RowSelection::from(vec![
            RowSelector::skip(SKIPPED),
            RowSelector::select(SELECTED),
            RowSelector::skip(rows().count() - SKIPPED - SELECTED),
        ])
    }

    #[test]
    fn a_load_decodes_the_page_index_of_the_asked_columns_only() {
        let (bytes, footer) = fixture();
        let full = full(&bytes, &footer);
        let indexed = load(&kept(&bytes, &footer), &footer, &[1]);
        assert_eq!(loaded_leaves(&indexed), vec![vec![1]; GROUPS]);
        assert_eq!(located_leaves(&indexed), vec![vec![1]; GROUPS]);
        assert_leaf_matches(&indexed, &full, 1);
        assert_eq!(
            indexed.row_groups(),
            footer.row_groups(),
            "the footer itself is unchanged"
        );
    }

    #[test]
    fn a_later_load_adds_columns_and_keeps_what_was_loaded() {
        let (bytes, footer) = fixture();
        let full = full(&bytes, &footer);
        let index = kept(&bytes, &footer);
        let first = load(&index, &footer, &[1]);
        let second = load(&index, &first, &[0]);
        assert_eq!(loaded_leaves(&second), vec![vec![0, 1]; GROUPS]);
        assert_eq!(located_leaves(&second), vec![vec![0, 1]; GROUPS]);
        assert_leaf_matches(&second, &full, 0);
        assert_leaf_matches(&second, &full, 1);
    }

    /// Loaded one column at a time or all at once, the page index is
    /// arrow-rs's own, entry for entry.
    #[test]
    fn loading_every_column_gives_the_page_index_arrow_loads() {
        let (bytes, footer) = fixture();
        let full = full(&bytes, &footer);
        let index = kept(&bytes, &footer);
        let at_once = load(&index, &footer, &[0, 1, 2]);
        let one_by_one = [2, 0, 1].into_iter().fold(footer.clone(), |parquet, leaf| {
            load(&index, &parquet, &[leaf])
        });
        for indexed in [at_once, one_by_one] {
            assert_eq!(indexed.column_index(), full.column_index());
            assert_eq!(indexed.offset_index(), full.offset_index());
        }
    }

    /// A column read through metadata that holds its page index returns its
    /// rows; one read through metadata that doesn't fails, all of it or a
    /// selection, rather than returning no rows.
    /// The fixture's columns are dictionary encoded, so a read of one starts
    /// with a dictionary page before its data pages.
    #[test]
    fn a_read_of_a_column_not_loaded_fails_rather_than_returning_no_rows() {
        let (bytes, footer) = fixture();
        assert!(
            footer
                .row_group(0)
                .column(0)
                .dictionary_page_offset()
                .is_some(),
            "the fixture must be dictionary encoded, or the dictionary case is not tested"
        );
        let indexed = load(&kept(&bytes, &footer), &footer, &[1]);
        let placeholder = &indexed.offset_index().expect("offset index")[0][0];
        assert!(!is_loaded(placeholder));
        assert_eq!(
            placeholder.page_locations()[0].offset as u64,
            footer.row_group(0).column(0).byte_range().0,
            "the placeholder sits at the chunk's start, so arrow-rs infers no dictionary page"
        );
        assert_eq!(
            read_leaf(&bytes, indexed.clone(), 1, false),
            Ok(rows().count())
        );
        assert_eq!(read_leaf(&bytes, indexed.clone(), 1, true), Ok(SELECTED));
        for selected in [false, true] {
            assert!(
                read_leaf(&bytes, indexed.clone(), 0, selected).is_err(),
                "selected: {selected}"
            );
        }
    }

    /// The same through the async reader: a column not loaded fails there
    /// too, all of it or a selection, while a loaded one returns its rows.
    #[tokio::test]
    async fn an_async_read_of_a_column_not_loaded_fails_rather_than_returning_no_rows() {
        let (bytes, footer) = fixture();
        let indexed = load(&kept(&bytes, &footer), &footer, &[1]);
        assert_eq!(
            read_leaf_async(&bytes, indexed.clone(), 1, false).await,
            Ok(rows().count())
        );
        assert_eq!(
            read_leaf_async(&bytes, indexed.clone(), 1, true).await,
            Ok(SELECTED)
        );
        for selected in [false, true] {
            assert!(
                read_leaf_async(&bytes, indexed.clone(), 0, selected)
                    .await
                    .is_err(),
                "selected: {selected}"
            );
        }
    }

    /// [`UNLOADED_CHUNK_BYTES`] is what arrow-rs counts for a chunk not
    /// loaded: a footer of placeholders only takes exactly that per chunk
    /// over the bare footer, plus each row group's two entry lists. A
    /// dependency bump that changes the layout fails here instead of
    /// skewing a reader's memory rule.
    #[test]
    fn a_chunk_not_loaded_takes_the_placeholder_size_arrow_counts() {
        let (bytes, footer) = fixture();
        let placeholders = load(&kept(&bytes, &footer), &footer, &[]);
        let chunks =
            (footer.num_row_groups() * footer.file_metadata().schema_descr().num_columns()) as u64;
        let lists = footer.num_row_groups() as u64
            * (size_of::<Vec<ColumnIndexMetaData>>() + size_of::<Vec<OffsetIndexMetaData>>())
                as u64;
        assert_eq!(
            (placeholders.memory_size() - footer.memory_size()) as u64,
            chunks * UNLOADED_CHUNK_BYTES + lists
        );
    }

    #[test]
    fn a_load_of_no_columns_holds_no_page_index_but_every_entry() {
        // So DataFusion's opener, which loads the page index itself when
        // either part is missing, never does.
        let (bytes, footer) = fixture();
        let indexed = load(&kept(&bytes, &footer), &footer, &[]);
        assert_eq!(loaded_leaves(&indexed), vec![Vec::<usize>::new(); GROUPS]);
        assert_eq!(located_leaves(&indexed), vec![Vec::<usize>::new(); GROUPS]);
        assert!(
            indexed
                .offset_index()
                .expect("offset index")
                .iter()
                .all(|row_group| row_group.len() == 3)
        );
    }

    /// The page index is one span: the column indexes, then the offset
    /// indexes, of every chunk.
    #[test]
    fn the_span_is_the_whole_page_index() {
        let (_, footer) = fixture();
        let span = |range: fn(&ColumnChunkMetaData) -> Option<Range<u64>>| {
            span_of(chunks(&footer).filter_map(|(_, chunk)| range(chunk))).expect("ranges")
        };
        assert_eq!(
            PageIndexBytes::span(&footer),
            Some(span(column_index_range).start..span(offset_index_range).end)
        );
    }

    #[test]
    fn kept_bytes_come_from_a_wider_read_and_hold_only_the_index() {
        let (bytes, footer) = fixture();
        let span = PageIndexBytes::span(&footer).expect("span");
        let around = (span.start - 1)..(span.end + 1);
        let read = slice_at(&bytes, 0, &around).expect("in the file");
        let kept = PageIndexBytes::copied(&span, around.start, &read).expect("kept");
        assert_eq!(kept.bytes.len() as u64, span.end - span.start);
        assert!(
            !read.as_ptr_range().contains(&kept.bytes.as_ptr()),
            "a copy, so the buffer the read came in is freed"
        );
        assert_eq!(
            load(&kept, &footer, &[0, 1, 2]).offset_index(),
            full(&bytes, &footer).offset_index()
        );
        assert!(PageIndexBytes::copied(&around, span.start, &kept.bytes).is_none());
    }

    /// A filter on a struct column prunes by its fields, each a leaf of
    /// its own, so the load brings every leaf under the column.
    #[test]
    fn a_struct_column_brings_the_page_index_of_every_leaf_under_it() {
        let values = || Arc::new(Int64Array::from_iter_values(rows())) as ArrayRef;
        let fields = Fields::from(
            ["x", "y"]
                .map(|name| Field::new(name, DataType::Int64, false))
                .to_vec(),
        );
        let point = StructArray::new(fields.clone(), vec![values(), values()], None);
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("point", DataType::Struct(fields), false),
        ]);
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![values(), Arc::new(point)]).expect("batch");
        let (bytes, footer) = written(batch, paged().build());
        let leaves = IndexColumns::named(&["point"]).leaves(footer.file_metadata().schema_descr());
        assert_eq!(leaves, [1, 2].into_iter().collect());
        let indexed = load(
            &kept(&bytes, &footer),
            &footer,
            &leaves.iter().collect::<Vec<_>>(),
        );
        assert_eq!(loaded_leaves(&indexed), vec![vec![1, 2]; GROUPS]);
        assert_eq!(located_leaves(&indexed), vec![vec![1, 2]; GROUPS]);
    }

    /// A list column's page index carries what a flat one does not: a
    /// repetition-level histogram per page, and nulls inside its lists. The
    /// load decodes each leaf through a footer that rebuilds it as a flat
    /// optional leaf, and must still give arrow-rs's own entries, leaf by
    /// leaf or all at once.
    #[test]
    fn a_list_column_loads_the_page_index_arrow_loads() {
        let item = Arc::new(Field::new("item", DataType::Int64, true));
        let lists = ListArray::from_iter_primitive::<Int64Type, _, _>(rows().map(|r| {
            (r % 7 != 0).then(|| {
                (0..r % 4)
                    .map(|i| (i != 2).then_some(r * 10 + i))
                    .collect::<Vec<_>>()
            })
        }));
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("tags", DataType::List(item), true),
            Field::new("c", DataType::Int64, false),
        ]);
        let column = || Arc::new(Int64Array::from_iter_values(rows())) as ArrayRef;
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![column(), Arc::new(lists), column()])
                .expect("batch");
        let (bytes, footer) = written(batch, paged().build());
        let full = full(&bytes, &footer);
        let tags = &full.column_index().expect("column index")[0][1];
        assert!(
            tags.repetition_level_histogram(0).is_some(),
            "guard: the list leaf's column index carries level histograms"
        );
        let index = kept(&bytes, &footer);
        let at_once = load(&index, &footer, &[0, 1, 2]);
        let one_by_one = [1, 2, 0].into_iter().fold(footer.clone(), |parquet, leaf| {
            load(&index, &parquet, &[leaf])
        });
        for indexed in [at_once, one_by_one] {
            assert_eq!(indexed.column_index(), full.column_index());
            assert_eq!(indexed.offset_index(), full.offset_index());
        }
    }

    /// A file written with no page index has nothing to load, so its scans
    /// read pages without one, as before.
    #[test]
    fn a_file_without_a_page_index_has_nothing_to_load() {
        let props = paged()
            .set_statistics_enabled(EnabledStatistics::None)
            .set_offset_index_disabled(true)
            .build();
        let (_, footer) = written(abc(), props);
        assert!(PageIndexBytes::span(&footer).is_none());
    }

    /// A column written without page statistics has page locations but no
    /// column index: asking for it loads its locations and leaves its column
    /// index `NONE`.
    #[test]
    fn a_column_without_page_statistics_loads_its_locations_only() {
        let props = paged()
            .set_column_statistics_enabled(ColumnPath::from("b"), EnabledStatistics::Chunk)
            .build();
        let (bytes, footer) = written(abc(), props);
        let full = full(&bytes, &footer);
        let indexed = load(&kept(&bytes, &footer), &footer, &[0, 1]);
        assert_eq!(loaded_leaves(&indexed), vec![vec![0]; GROUPS]);
        assert_eq!(located_leaves(&indexed), vec![vec![0, 1]; GROUPS]);
        assert_leaf_matches(&indexed, &full, 1);
    }

    /// A file written with field ids finds a column by id after the table
    /// renamed it; one written without ids, by the name it was written with.
    #[test]
    fn index_columns_match_by_field_id_then_by_name() {
        let renamed = Field::new("old_name", DataType::Int64, false).with_metadata(
            [(
                PARQUET_FIELD_ID_META_KEY.to_owned(),
                RENAMED_FIELD_ID.to_string(),
            )]
            .into(),
        );
        let schema = Arc::new(Schema::new(vec![renamed]));
        let values = Arc::new(Int64Array::from_iter_values(rows())) as ArrayRef;
        let batch = RecordBatch::try_new(schema, vec![values]).expect("batch");
        let (_, with_ids) = written(batch, paged().build());
        let schema = with_ids.file_metadata().schema_descr();
        let by_id = IndexColumns::of([(Some(RENAMED_FIELD_ID), "new_name")]);
        assert_eq!(by_id.leaves(schema), [0].into_iter().collect());
        assert!(IndexColumns::named(&["new_name"]).leaves(schema).is_empty());

        // A file written before ids, whose `b` the table has since renamed:
        // found by the name it was written with, passed beside the new one.
        let (_, without_ids) = fixture();
        let schema = without_ids.file_metadata().schema_descr();
        let current_only = IndexColumns::of([(Some(RENAMED_FIELD_ID), "b_renamed")]);
        assert!(current_only.leaves(schema).is_empty());
        let both_names = IndexColumns::of([
            (Some(RENAMED_FIELD_ID), "b_renamed"),
            (Some(RENAMED_FIELD_ID), "b"),
        ]);
        assert_eq!(both_names.leaves(schema), [1].into_iter().collect());
    }

    #[test]
    fn index_columns_match_by_name_and_compare_as_sets() {
        let (_, footer) = fixture();
        let schema = footer.file_metadata().schema_descr();
        assert_eq!(
            IndexColumns::named(&["b", "zz"]).leaves(schema),
            [1].into_iter().collect()
        );
        assert_eq!(IndexColumns::All.leaves(schema), (0..3).collect());
        assert!(IndexColumns::none().leaves(schema).is_empty());
        let a = IndexColumns::named(&["a"]);
        let ab = a.union(&IndexColumns::named(&["b"]));
        assert!(ab.covers(&a) && !a.covers(&ab));
        assert!(IndexColumns::All.covers(&ab) && !ab.covers(&IndexColumns::All));
        assert!(ab.covers(&IndexColumns::none()));
    }
}
