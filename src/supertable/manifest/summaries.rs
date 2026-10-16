// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A part's summaries section: every superfile's per-column scalar
//! statistics in one content-addressed object beside the part, read by
//! column on demand instead of decoded into every entry at open.
//!
//! A part holds its superfiles' identities, sizes, id ranges, partition
//! tags and offsets, which a handle must have resident. The per-column
//! statistics are the part's weight at thousands of columns, and a query
//! reads a handful of columns of them. So they live here, grouped by
//! Arrow type into a few record batches whose rows are (column,
//! superfile) pairs, with a directory from column to its rows. A handle
//! keeps the section's bytes and decodes a type group the first time a
//! query asks for a column in it; a bound read out of a group is a slice
//! of the group's arrays, never a buffer of its own.
//!
//! Layout, little-endian:
//!
//! ```text
//! "ISUM" | u32 version | 16 bytes part id
//! u32 n_superfiles | u32 n_entries | u32 n_groups
//! n_superfiles x 16 bytes   superfile ids, in row order
//! n_entries x 16 bytes      the directory: u32 field id, u32 group,
//!                           u32 row start, u32 row count; by field id
//! n_groups x 16 bytes       each group's IPC stream: u64 offset, u64 length
//! the IPC streams
//! ```
//!
//! The part id is the part's own random id, so a rebuild of the same
//! superfiles never writes a section byte-identical to an earlier one: a
//! section's name is its content hash, a PUT of a name that exists is
//! skipped, and the earlier section may already be queued for deletion.
//! Version 1 sections, written before the id was carried, have no id
//! bytes and are read without the check.
//!
//! A group's batch has `column: UInt32`, `superfile: UInt32`, `min` and
//! `max` of the group's type, `null_count: UInt64`, and, where the type
//! has them, `sum` (the SUM result type), `hll: Binary` and
//! `value_counts: List<Struct<value, count: UInt64>>`. Every column but
//! the first two is nullable: a null bound is an all-null column, a null
//! count or sketch or table is one the superfile does not carry. A
//! superfile that lacks a column has no row for it. A column's rows sit
//! in one run per group, sorted by superfile; a column retyped between
//! superfiles has a run in each type's group.

use std::{
    collections::HashMap,
    fmt,
    io::Cursor,
    sync::{Arc, OnceLock},
};

use arrow::{
    buffer::{NullBuffer, OffsetBuffer},
    compute::concat,
    ipc::{reader::StreamReader, writer::StreamWriter},
};
use arrow_array::{
    Array, ArrayRef, BinaryArray, ListArray, RecordBatch, StructArray, UInt32Array, UInt64Array,
    new_null_array,
};
use arrow_schema::{DataType, Field, Fields, Schema};
use bytes::Bytes;
use datafusion::scalar::ScalarValue;
use thiserror::Error;
use tracing::warn;
use uuid::Uuid;

use crate::supertable::{
    manifest::{
        SuperfileEntry, column_hll,
        encoding::stats_ipc_options,
        hll::HllSketch,
        list::{ScalarStatsAgg, ScalarValueCounts},
        part::{ContentHash, PartId},
    },
    schema::FieldId,
};

/// Storage directory of the sections, beside the parts.
pub const MANIFEST_SUMMARIES_DIR: &str = "manifest-summaries";

/// Where a section with content hash `hash` lives.
pub fn summaries_uri(hash: &ContentHash) -> String {
    format!("{MANIFEST_SUMMARIES_DIR}/summ-{}.bin", hash.to_hex())
}

/// The first four bytes of a section.
const MAGIC: &[u8; 4] = b"ISUM";
/// The layout version this engine writes: the part id follows the version.
const VERSION: u32 = 2;
/// The first layout, without the part id; still read.
const VERSION_WITHOUT_PART_ID: u32 = 1;
/// Magic, the version, the part id and three `u32` counts.
const HEADER_BYTES: usize = MAGIC.len() + U32_BYTES + ID_BYTES + 3 * U32_BYTES;
/// Width of a little-endian `u32`.
const U32_BYTES: usize = 4;
/// Width of a little-endian `u64`.
const U64_BYTES: usize = 8;
/// A superfile id, or the part id, on the wire.
const ID_BYTES: usize = 16;
/// A directory entry: four `u32`s.
const DIRECTORY_ENTRY_BYTES: usize = 4 * U32_BYTES;
/// A group table entry: two `u64`s.
const GROUP_ENTRY_BYTES: usize = 2 * U64_BYTES;
/// The most bytes a group's variable-width array may address: Arrow's
/// 32-bit offsets. A group past it is refused as an error rather than
/// left to the array builders, which panic on the overflow.
const MAX_GROUP_OFFSET: usize = i32::MAX as usize;

/// The group batch's column names.
const COLUMN_FIELD: &str = "column";
const SUPERFILE_FIELD: &str = "superfile";
const MIN_FIELD: &str = "min";
const MAX_FIELD: &str = "max";
const NULL_COUNT_FIELD: &str = "null_count";
const SUM_FIELD: &str = "sum";
const HLL_FIELD: &str = "hll";
const VALUE_COUNTS_FIELD: &str = "value_counts";
/// The value-count struct's members and the list item's name.
const VALUE_FIELD: &str = "value";
const COUNT_FIELD: &str = "count";
const ITEM_FIELD: &str = "item";

/// A section's address: the content-addressed object a list entry names,
/// with its size, which counts toward the part's split cap beside the
/// part's own bytes (a commit re-encodes the latest part's section whole).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummariesRef {
    pub uri: String,
    pub content_hash: ContentHash,
    pub size_bytes: u64,
}

/// Why a section could not be encoded or read.
#[derive(Debug, Clone, Error)]
pub enum SummariesError {
    #[error("summaries section truncated: needed {needed} bytes for {what}, had {had}")]
    Truncated {
        what: &'static str,
        needed: usize,
        had: usize,
    },
    #[error("summaries section does not start with its magic")]
    BadMagic,
    #[error("summaries section version {0} is newer than this engine reads")]
    Version(u32),
    #[error("summaries section is malformed: {0}")]
    Malformed(String),
    #[error("summaries section arrow: {0}")]
    Arrow(String),
}

impl SummariesError {
    fn arrow(error: impl fmt::Display) -> Self {
        Self::Arrow(error.to_string())
    }
}

/// One run of a column's rows inside a group.
#[derive(Debug, Clone, Copy)]
struct DirectoryEntry {
    field: FieldId,
    group: u32,
    row_start: u32,
    row_count: u32,
}

/// One type group, decoded. The batch's `column` array is not kept: the
/// directory says which rows are a column's, and the array is on the
/// wire for a reader of the bytes alone.
struct Group {
    superfile: UInt32Array,
    min: ArrayRef,
    max: ArrayRef,
    null_count: UInt64Array,
    sum: Option<ArrayRef>,
    hll: Option<BinaryArray>,
    value_counts: Option<ListArray>,
}

/// One column's statistics for one superfile, read in place from the
/// decoded group: each accessor reads the one array it needs. The value
/// table alone is built on request, since building it converts every
/// value, and the pruner and the planner's statistics never need it.
#[derive(Clone, Copy)]
pub struct ColumnRow<'a> {
    group: &'a Group,
    index: usize,
}

impl ColumnRow<'_> {
    /// The lower bound as a one-row array; null when the column is null
    /// throughout the superfile.
    pub fn min(&self) -> ArrayRef {
        self.group.min.slice(self.index, 1)
    }

    /// The upper bound as a one-row array.
    pub fn max(&self) -> ArrayRef {
        self.group.max.slice(self.index, 1)
    }

    /// Both bounds as values, when they convert.
    pub fn min_max_values(&self) -> Option<(ScalarValue, ScalarValue)> {
        let min = ScalarValue::try_from_array(&self.group.min, self.index).ok()?;
        let max = ScalarValue::try_from_array(&self.group.max, self.index).ok()?;
        Some((min, max))
    }

    /// Whether the column had no non-null value in the superfile.
    pub fn all_null(&self) -> bool {
        self.group.min.is_null(self.index)
    }

    /// The null count, when the superfile carries it.
    pub fn null_count(&self) -> Option<u64> {
        self.group
            .null_count
            .is_valid(self.index)
            .then(|| self.group.null_count.value(self.index))
    }

    /// The exact sum as a one-row array, when the superfile carries it.
    pub fn sum(&self) -> Option<ArrayRef> {
        self.group
            .sum
            .as_ref()
            .filter(|sum| sum.is_valid(self.index))
            .map(|sum| sum.slice(self.index, 1))
    }

    /// The distinct-count sketch's bytes, borrowed, when the superfile
    /// carries one.
    pub fn sketch(&self) -> Option<&[u8]> {
        self.group
            .hll
            .as_ref()
            .filter(|hll| hll.is_valid(self.index))
            .map(|hll| hll.value(self.index))
    }

    /// The sketch of the exact value table's values, hashed as the column
    /// sketch hashes them, when the superfile carries a table: the
    /// distinct estimate of a column counted exactly, taken from the
    /// table's value array without converting a value.
    pub fn values_sketch(&self) -> Option<HllSketch> {
        let list = self.group.value_counts.as_ref()?;
        if !list.is_valid(self.index) {
            return None;
        }
        let item = list.value(self.index);
        let item = item.as_any().downcast_ref::<StructArray>()?;
        column_hll(&[Arc::clone(item.column(0))])
    }

    /// The exact value table as it sits in the section: one list item, a
    /// struct of values and counts, when the superfile carries a table.
    /// What a re-encode of this row carries over, bytes for bytes, in
    /// place of a rebuild.
    fn value_table_array(&self) -> Option<ArrayRef> {
        self.group
            .value_counts
            .as_ref()
            .filter(|list| list.is_valid(self.index))
            .map(|list| list.value(self.index))
    }

    /// The exact value table, built from its row, when the superfile
    /// carries one.
    pub(crate) fn value_counts(&self) -> Option<ScalarValueCounts> {
        self.value_table_array()
            .and_then(|item| value_counts_of(&item))
    }

    /// Every statistic, in the decoded shape.
    pub fn stats(&self) -> ScalarStatsAgg {
        ScalarStatsAgg {
            min: self.min(),
            max: self.max(),
            null_count: self.null_count(),
            sum: self.sum(),
            hll: self.sketch().map(<[u8]>::to_vec),
            value_counts: self.value_counts(),
        }
    }
}

/// One column's statistics for one superfile, wherever they live, read
/// one statistic at a time. Decoded statistics are borrowed; a section's
/// are read from the decoded group's arrays. Only [`Self::value_counts`]
/// builds anything, so the pruner, the planner's statistics and the
/// rollup fold, which read bounds, counts, sums and sketches, cost one
/// array read each per superfile and column.
#[derive(Clone, Copy)]
pub enum ColumnStats<'a> {
    Inline(&'a ScalarStatsAgg),
    Section(ColumnRow<'a>),
}

impl ColumnStats<'_> {
    /// The lower bound as a one-row array; null when the column is null
    /// throughout the superfile.
    pub fn min(&self) -> ArrayRef {
        match self {
            Self::Inline(agg) => Arc::clone(&agg.min),
            Self::Section(row) => row.min(),
        }
    }

    /// The upper bound as a one-row array.
    pub fn max(&self) -> ArrayRef {
        match self {
            Self::Inline(agg) => Arc::clone(&agg.max),
            Self::Section(row) => row.max(),
        }
    }

    /// Both bounds as values, when they convert.
    pub fn min_max_values(&self) -> Option<(ScalarValue, ScalarValue)> {
        match self {
            Self::Inline(agg) => {
                let min = ScalarValue::try_from_array(agg.min.as_ref(), 0).ok()?;
                let max = ScalarValue::try_from_array(agg.max.as_ref(), 0).ok()?;
                Some((min, max))
            }
            Self::Section(row) => row.min_max_values(),
        }
    }

    /// Whether the column had no non-null value in the superfile.
    pub fn all_null(&self) -> bool {
        match self {
            Self::Inline(agg) => agg.all_null(),
            Self::Section(row) => row.all_null(),
        }
    }

    /// The null count, when the superfile carries it.
    pub fn null_count(&self) -> Option<u64> {
        match self {
            Self::Inline(agg) => agg.null_count,
            Self::Section(row) => row.null_count(),
        }
    }

    /// The exact sum as a one-row array, when the superfile carries it.
    pub fn sum(&self) -> Option<ArrayRef> {
        match self {
            Self::Inline(agg) => agg.sum.clone(),
            Self::Section(row) => row.sum(),
        }
    }

    /// The distinct-count sketch's bytes, when the superfile carries one.
    pub fn sketch(&self) -> Option<&[u8]> {
        match self {
            Self::Inline(agg) => agg.hll.as_deref(),
            Self::Section(row) => row.sketch(),
        }
    }

    /// The sketch that stands in for the exact value table of a column
    /// counted exactly, when the superfile carries a table: the table's
    /// values hashed as the column sketch hashes them, so a table's
    /// estimate and a sketch's merge into one.
    pub fn values_sketch(&self) -> Option<HllSketch> {
        match self {
            Self::Inline(agg) => agg.value_counts.as_ref().and_then(table_sketch),
            Self::Section(row) => row.values_sketch(),
        }
    }

    /// Whether the superfile carries an exact value table for the column.
    fn has_value_table(&self) -> bool {
        match self {
            Self::Inline(agg) => agg.value_counts.is_some(),
            Self::Section(row) => row.value_table_array().is_some(),
        }
    }

    /// The exact value table as a struct array of values and counts, for
    /// the encoder: a section's row is carried over as it is, a decoded
    /// table is built into one. `None` when the superfile carries no
    /// table, or the table's values are not of `data_type`.
    fn value_table_array(&self, data_type: &DataType) -> Option<ArrayRef> {
        let item = match self {
            Self::Inline(agg) => {
                let table = agg.value_counts.as_ref()?;
                let values = ScalarValue::iter_to_array(
                    table.entries().iter().map(|(value, _)| value.clone()),
                )
                .ok()?;
                let counts: ArrayRef = Arc::new(UInt64Array::from_iter_values(
                    table.entries().iter().map(|(_, n)| *n),
                ));
                let item =
                    StructArray::try_new(value_table_fields(data_type), vec![values, counts], None)
                        .ok()?;
                Arc::new(item) as ArrayRef
            }
            Self::Section(row) => row.value_table_array()?,
        };
        let fields = match item.data_type() {
            DataType::Struct(fields) => fields,
            _ => return None,
        };
        (fields.len() == 2 && fields[0].data_type() == data_type).then_some(item)
    }

    /// The exact value table, when the superfile carries one.
    pub(crate) fn value_counts(&self) -> Option<ScalarValueCounts> {
        match self {
            Self::Inline(agg) => agg.value_counts.clone(),
            Self::Section(row) => row.value_counts(),
        }
    }

    /// Every statistic, in the decoded shape.
    pub fn stats(&self) -> ScalarStatsAgg {
        match self {
            Self::Inline(agg) => (*agg).clone(),
            Self::Section(row) => row.stats(),
        }
    }
}

/// The sketch of an exact value table's values, hashed as the column
/// sketch hashes them: `None` for a value type the sketch does not cover.
pub(crate) fn table_sketch(table: &ScalarValueCounts) -> Option<HllSketch> {
    let values =
        ScalarValue::iter_to_array(table.entries().iter().map(|(value, _)| value.clone())).ok()?;
    column_hll(&[values])
}

/// The struct fields of a value table item: the values, of the group's
/// type, and their counts.
fn value_table_fields(data_type: &DataType) -> Fields {
    Fields::from(vec![
        Field::new(VALUE_FIELD, data_type.clone(), false),
        Field::new(COUNT_FIELD, DataType::UInt64, false),
    ])
}

/// The exact value table stored as one list item: a struct of values and
/// counts. `None` when the item does not decode as one.
fn value_counts_of(item: &ArrayRef) -> Option<ScalarValueCounts> {
    let item = item.as_any().downcast_ref::<StructArray>()?;
    let values = item.column(0);
    let counts = item.column(1).as_any().downcast_ref::<UInt64Array>()?;
    let mut entries = Vec::with_capacity(values.len());
    for row in 0..values.len() {
        if values.is_null(row) || counts.is_null(row) {
            return None;
        }
        let value = ScalarValue::try_from_array(values, row).ok()?;
        entries.push((value, counts.value(row)));
    }
    ScalarValueCounts::from_entries(entries)
}

/// A section, parsed to its directory; groups decode on first use.
pub struct SummariesSection {
    bytes: Bytes,
    /// The part the section was written for; `None` on a version 1
    /// section, from before the id was carried.
    part_id: Option<PartId>,
    superfile_ids: Vec<Uuid>,
    /// Sorted by field id, then group.
    directory: Vec<DirectoryEntry>,
    /// Each group's `(offset, length)` in `bytes`.
    groups: Vec<(usize, usize)>,
    decoded: Vec<OnceLock<Result<Group, SummariesError>>>,
}

impl fmt::Debug for SummariesSection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SummariesSection")
            .field("bytes", &self.bytes.len())
            .field("part_id", &self.part_id)
            .field("superfiles", &self.superfile_ids.len())
            .field("directory_entries", &self.directory.len())
            .field("groups", &self.groups.len())
            .finish()
    }
}

/// One row of the section as it is given to the encoder: a column's
/// statistics for one superfile, read from wherever they live. A row
/// read from an earlier section is carried into the new one array for
/// array; only a decoded row is built.
pub struct SummaryRow<'a> {
    pub column: FieldId,
    /// Index into the superfile ids the section is written for.
    pub superfile: u32,
    pub stats: ColumnStats<'a>,
}

impl SummariesSection {
    /// Encode the section of part `part_id` for the superfiles `ids`, in
    /// row order, from `rows`.
    ///
    /// A statistic that cannot be written is left out rather than failing
    /// the commit that writes the section: a row whose bounds are not one
    /// row of one type, or whose superfile is not among `ids`, is dropped
    /// with a warning, and the superfile then reads as having no
    /// statistics for that column, which keeps it in every prune. The
    /// errors left are the ones that mean the encoder itself is broken.
    pub fn encode(
        part_id: PartId,
        ids: &[Uuid],
        rows: Vec<SummaryRow<'_>>,
    ) -> Result<Vec<u8>, SummariesError> {
        // Group by the bounds' type, in a fixed order so equal input gives
        // equal bytes, which content addressing relies on.
        let mut by_type: Vec<(DataType, Vec<SummaryRow<'_>>)> = Vec::new();
        for row in rows {
            let (min, max) = (row.stats.min(), row.stats.max());
            if min.len() != 1 || max.len() != 1 {
                warn!(column = ?row.column, superfile = row.superfile, "a bound is not one row; the statistic is left out of the section");
                continue;
            }
            if min.data_type() != max.data_type() {
                warn!(column = ?row.column, superfile = row.superfile, "min and max differ in type; the statistic is left out of the section");
                continue;
            }
            if row.superfile as usize >= ids.len() {
                warn!(column = ?row.column, superfile = row.superfile, superfiles = ids.len(), "a statistic names a superfile the section has no row for; left out");
                continue;
            }
            let data_type = min.data_type().clone();
            match by_type.iter_mut().find(|(t, _)| *t == data_type) {
                Some((_, rows)) => rows.push(row),
                None => by_type.push((data_type, vec![row])),
            }
        }
        by_type.sort_by_key(|(data_type, _)| data_type.to_string());

        let mut directory: Vec<DirectoryEntry> = Vec::new();
        let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(by_type.len());
        for (group_index, (data_type, mut rows)) in by_type.into_iter().enumerate() {
            rows.sort_by_key(|row| (row.column, row.superfile));
            let mut start = 0usize;
            while start < rows.len() {
                let field = rows[start].column;
                let mut end = start + 1;
                while end < rows.len() && rows[end].column == field {
                    end += 1;
                }
                directory.push(DirectoryEntry {
                    field,
                    group: group_index as u32,
                    row_start: start as u32,
                    row_count: (end - start) as u32,
                });
                start = end;
            }
            payloads.push(encode_group(&data_type, &rows)?);
        }
        directory.sort_by_key(|entry| (entry.field, entry.group));

        let tables_end = HEADER_BYTES
            + ids.len() * ID_BYTES
            + directory.len() * DIRECTORY_ENTRY_BYTES
            + payloads.len() * GROUP_ENTRY_BYTES;
        let mut out = Vec::with_capacity(tables_end + payloads.iter().map(Vec::len).sum::<usize>());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(part_id.0.as_bytes());
        out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
        out.extend_from_slice(&(directory.len() as u32).to_le_bytes());
        out.extend_from_slice(&(payloads.len() as u32).to_le_bytes());
        for id in ids {
            out.extend_from_slice(id.as_bytes());
        }
        for entry in &directory {
            out.extend_from_slice(&entry.field.0.to_le_bytes());
            out.extend_from_slice(&entry.group.to_le_bytes());
            out.extend_from_slice(&entry.row_start.to_le_bytes());
            out.extend_from_slice(&entry.row_count.to_le_bytes());
        }
        let mut offset = tables_end as u64;
        for payload in &payloads {
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            offset += payload.len() as u64;
        }
        for payload in &payloads {
            out.extend_from_slice(payload);
        }
        Ok(out)
    }

    /// Parse a section's header, ids, directory and group table. The
    /// groups decode on first use.
    pub fn parse(bytes: Bytes) -> Result<Self, SummariesError> {
        let mut reader = Reader {
            bytes: &bytes,
            at: 0,
        };
        if reader.take(MAGIC.len(), "magic")? != MAGIC {
            return Err(SummariesError::BadMagic);
        }
        let version = reader.u32("version")?;
        if version > VERSION {
            return Err(SummariesError::Version(version));
        }
        let part_id = if version == VERSION_WITHOUT_PART_ID {
            None
        } else {
            let id = reader.take(ID_BYTES, "part id")?;
            Some(PartId(Uuid::from_slice(id).map_err(|e| {
                SummariesError::Malformed(format!("part id: {e}"))
            })?))
        };
        let n_superfiles = reader.u32("superfile count")? as usize;
        let n_entries = reader.u32("directory size")? as usize;
        let n_groups = reader.u32("group count")? as usize;
        let mut superfile_ids = Vec::with_capacity(n_superfiles);
        for _ in 0..n_superfiles {
            let id = reader.take(ID_BYTES, "superfile id")?;
            superfile_ids.push(
                Uuid::from_slice(id)
                    .map_err(|e| SummariesError::Malformed(format!("superfile id: {e}")))?,
            );
        }
        let mut directory = Vec::with_capacity(n_entries);
        for _ in 0..n_entries {
            let entry = DirectoryEntry {
                field: FieldId(reader.u32("directory field")?),
                group: reader.u32("directory group")?,
                row_start: reader.u32("directory row start")?,
                row_count: reader.u32("directory row count")?,
            };
            if entry.group as usize >= n_groups {
                return Err(SummariesError::Malformed(format!(
                    "directory names group {} of {n_groups}",
                    entry.group
                )));
            }
            directory.push(entry);
        }
        let mut groups = Vec::with_capacity(n_groups);
        for _ in 0..n_groups {
            let offset = reader.u64("group offset")? as usize;
            let length = reader.u64("group length")? as usize;
            let end = offset
                .checked_add(length)
                .ok_or_else(|| SummariesError::Malformed("group range overflows".to_string()))?;
            if end > bytes.len() {
                return Err(SummariesError::Truncated {
                    what: "group payload",
                    needed: end,
                    had: bytes.len(),
                });
            }
            groups.push((offset, length));
        }
        let decoded = (0..n_groups).map(|_| OnceLock::new()).collect();
        Ok(Self {
            bytes,
            part_id,
            superfile_ids,
            directory,
            groups,
            decoded,
        })
    }

    /// The part this section was written for; `None` on a version 1
    /// section.
    pub fn part_id(&self) -> Option<PartId> {
        self.part_id
    }

    /// The superfiles this section holds rows for, in row order.
    pub fn superfile_ids(&self) -> &[Uuid] {
        &self.superfile_ids
    }

    /// The row of superfile `id`, if the section holds it.
    pub fn row_of(&self, id: Uuid) -> Option<u32> {
        self.superfile_ids
            .iter()
            .position(|known| *known == id)
            .map(|row| row as u32)
    }

    /// The statistics of `column` for the superfile at `row`, read in
    /// place, or `None` when that superfile carries no row for the
    /// column. A group that does not decode reads as no statistics, after
    /// a warning: the bytes were hash-verified when loaded, so that is a
    /// defect, and reading no statistics keeps every superfile rather
    /// than dropping one.
    pub fn column(&self, row: u32, column: FieldId) -> Option<ColumnRow<'_>> {
        let first = self.directory.partition_point(|entry| entry.field < column);
        self.directory[first..]
            .iter()
            .take_while(|entry| entry.field == column)
            .find_map(|entry| self.row_in_run(entry, row))
    }

    /// The row of superfile `row` inside the directory run `entry`, when
    /// the run's group decodes and holds it. The run was checked against
    /// the group's length when the group decoded, so the slice is in
    /// bounds; `get` keeps a defect from reaching a panic regardless.
    fn row_in_run(&self, entry: &DirectoryEntry, row: u32) -> Option<ColumnRow<'_>> {
        let group = self.group(entry.group as usize)?;
        let start = entry.row_start as usize;
        let end = start.checked_add(entry.row_count as usize)?;
        let i = group
            .superfile
            .values()
            .get(start..end)?
            .binary_search(&row)
            .ok()?;
        Some(ColumnRow {
            group,
            index: start + i,
        })
    }

    /// [`Self::column`] in the decoded shape, every statistic built.
    pub fn stat(&self, row: u32, column: FieldId) -> Option<ScalarStatsAgg> {
        self.column(row, column).map(|found| found.stats())
    }

    /// Every column the superfile at `row` carries, each read in place.
    pub fn row_columns(&self, row: u32) -> Vec<(FieldId, ColumnRow<'_>)> {
        self.directory
            .iter()
            .filter_map(|entry| Some((entry.field, self.row_in_run(entry, row)?)))
            .collect()
    }

    /// Whether the superfile at `row` carries statistics for any column:
    /// the directory is walked until a run holds the row, decoding only
    /// the groups walked.
    pub fn row_has_statistics(&self, row: u32) -> bool {
        self.directory
            .iter()
            .any(|entry| self.row_in_run(entry, row).is_some())
    }

    /// Every column's statistics for the superfile at `row`, every
    /// statistic built.
    pub fn row_map(&self, row: u32) -> HashMap<FieldId, ScalarStatsAgg> {
        self.row_columns(row)
            .into_iter()
            .map(|(column, found)| (column, found.stats()))
            .collect()
    }

    /// The columns the section holds, each once.
    pub fn columns(&self) -> impl Iterator<Item = FieldId> + '_ {
        let mut last: Option<FieldId> = None;
        self.directory.iter().filter_map(move |entry| {
            if last == Some(entry.field) {
                None
            } else {
                last = Some(entry.field);
                Some(entry.field)
            }
        })
    }

    /// Bytes resident for this section: its bytes, and each decoded group
    /// once more, since the IPC reader copies a group's body into its
    /// arrays. (The arrays themselves report that shared body once per
    /// array, so they are not what is summed here.)
    pub fn resident_bytes(&self) -> usize {
        self.bytes.len()
            + self
                .decoded
                .iter()
                .zip(&self.groups)
                .filter(|(cell, _)| cell.get().is_some_and(|group| group.is_ok()))
                .map(|(_, (_, length))| *length)
                .sum::<usize>()
    }

    /// Group `index`, decoded on first use and checked against the
    /// directory: a run that reaches past the group's rows makes the whole
    /// group read as no statistics, since the directory and the group
    /// disagree and neither can be trusted.
    fn group(&self, index: usize) -> Option<&Group> {
        let cell = self.decoded.get(index)?;
        let decoded = cell.get_or_init(|| {
            let (offset, length) = self.groups[index];
            let group = decode_group(&self.bytes[offset..offset + length])?;
            let rows = group.superfile.len();
            let run_past_the_group = self
                .directory
                .iter()
                .filter(|entry| entry.group as usize == index)
                .any(|entry| {
                    (entry.row_start as usize)
                        .checked_add(entry.row_count as usize)
                        .is_none_or(|end| end > rows)
                });
            if run_past_the_group {
                return Err(SummariesError::Malformed(format!(
                    "a directory run reaches past group {index}'s {rows} rows"
                )));
            }
            Ok(group)
        });
        match decoded {
            Ok(group) => Some(group),
            Err(error) => {
                warn!(%error, group = index, "summaries section group does not decode");
                None
            }
        }
    }
}

/// A cursor over the fixed-width head of a section.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], SummariesError> {
        let end = self.at + n;
        if end > self.bytes.len() {
            return Err(SummariesError::Truncated {
                what,
                needed: end,
                had: self.bytes.len(),
            });
        }
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, SummariesError> {
        let bytes = self.take(U32_BYTES, what)?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("four bytes were taken"),
        ))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, SummariesError> {
        let bytes = self.take(U64_BYTES, what)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("eight bytes were taken"),
        ))
    }
}

/// One group's batch as an IPC stream. `rows` are sorted by (column,
/// superfile) and share `data_type` as their bounds' type.
///
/// The bounds and the null counts are the group; the sums, the sketches
/// and the value tables are carried when they fit the group's 32-bit
/// offsets and left out, with a warning, when they do not: a group that
/// big has lost those statistics, not its bounds, and the commit goes on.
fn encode_group(data_type: &DataType, rows: &[SummaryRow<'_>]) -> Result<Vec<u8>, SummariesError> {
    let column = UInt32Array::from(rows.iter().map(|row| row.column.0).collect::<Vec<_>>());
    let superfile = UInt32Array::from(rows.iter().map(|row| row.superfile).collect::<Vec<_>>());
    let mins: Vec<ArrayRef> = rows.iter().map(|row| row.stats.min()).collect();
    let maxes: Vec<ArrayRef> = rows.iter().map(|row| row.stats.max()).collect();
    let min = concat(&as_dyn(&mins)).map_err(SummariesError::arrow)?;
    let max = concat(&as_dyn(&maxes)).map_err(SummariesError::arrow)?;
    let null_count = UInt64Array::from(
        rows.iter()
            .map(|row| row.stats.null_count())
            .collect::<Vec<_>>(),
    );

    let mut fields = vec![
        Field::new(COLUMN_FIELD, DataType::UInt32, false),
        Field::new(SUPERFILE_FIELD, DataType::UInt32, false),
        Field::new(MIN_FIELD, data_type.clone(), true),
        Field::new(MAX_FIELD, data_type.clone(), true),
        Field::new(NULL_COUNT_FIELD, DataType::UInt64, true),
    ];
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(column),
        Arc::new(superfile),
        min,
        max,
        Arc::new(null_count),
    ];

    let sums: Vec<Option<ArrayRef>> = rows.iter().map(|row| row.stats.sum()).collect();
    if let Some(sum_type) = sums
        .iter()
        .flatten()
        .map(|sum| sum.data_type().clone())
        .next()
    {
        let null_sum = new_null_array(&sum_type, 1);
        let sums: Vec<ArrayRef> = sums
            .iter()
            .map(|sum| match sum {
                Some(sum) if sum.data_type() == &sum_type && sum.len() == 1 => Arc::clone(sum),
                _ => Arc::clone(&null_sum),
            })
            .collect();
        match concat(&as_dyn(&sums)) {
            Ok(array) => {
                fields.push(Field::new(SUM_FIELD, sum_type, true));
                arrays.push(array);
            }
            Err(error) => {
                warn!(%error, group = %data_type, "the group's sums do not encode; left out of the section");
            }
        }
    }

    let sketches: Vec<Option<&[u8]>> = rows.iter().map(|row| row.stats.sketch()).collect();
    if sketches.iter().any(Option::is_some) {
        let total: usize = sketches.iter().flatten().map(|bytes| bytes.len()).sum();
        if total > MAX_GROUP_OFFSET {
            warn!(group = %data_type, bytes = total, "the group's sketches pass the 32-bit offsets; left out of the section");
        } else {
            fields.push(Field::new(HLL_FIELD, DataType::Binary, true));
            arrays.push(Arc::new(BinaryArray::from(sketches)));
        }
    }

    if rows.iter().any(|row| row.stats.has_value_table()) {
        let mut items: Vec<ArrayRef> = Vec::new();
        let mut lengths: Vec<usize> = Vec::with_capacity(rows.len());
        let mut present: Vec<bool> = Vec::with_capacity(rows.len());
        let mut entries = 0usize;
        for row in rows {
            match row.stats.value_table_array(data_type) {
                Some(item) => {
                    entries = entries.saturating_add(item.len());
                    lengths.push(item.len());
                    present.push(true);
                    items.push(item);
                }
                None => {
                    lengths.push(0);
                    present.push(false);
                }
            }
        }
        let list = if entries > MAX_GROUP_OFFSET {
            warn!(group = %data_type, entries, "the group's value tables pass the 32-bit offsets; left out of the section");
            None
        } else if items.is_empty() {
            None
        } else {
            value_table_list(data_type, &items, lengths, present)
                .map_err(|error| {
                    warn!(%error, group = %data_type, "the group's value tables do not encode; left out of the section");
                })
                .ok()
        };
        if let Some((field, list)) = list {
            fields.push(field);
            arrays.push(list);
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).map_err(SummariesError::arrow)?;
    let mut out = Vec::new();
    {
        let mut writer = StreamWriter::try_new_with_options(&mut out, &schema, stats_ipc_options())
            .map_err(SummariesError::arrow)?;
        writer.write(&batch).map_err(SummariesError::arrow)?;
        writer.finish().map_err(SummariesError::arrow)?;
    }
    Ok(out)
}

/// The value-table column of a group: one list item per row, each a
/// struct of values and counts, from the per-row items.
fn value_table_list(
    data_type: &DataType,
    items: &[ArrayRef],
    lengths: Vec<usize>,
    present: Vec<bool>,
) -> Result<(Field, ArrayRef), SummariesError> {
    let items = concat(&as_dyn(items)).map_err(SummariesError::arrow)?;
    let item_field = Arc::new(Field::new(
        ITEM_FIELD,
        DataType::Struct(value_table_fields(data_type)),
        false,
    ));
    let list = ListArray::try_new(
        Arc::clone(&item_field),
        OffsetBuffer::<i32>::from_lengths(lengths),
        items,
        Some(NullBuffer::from(present)),
    )
    .map_err(SummariesError::arrow)?;
    Ok((
        Field::new(VALUE_COUNTS_FIELD, DataType::List(item_field), true),
        Arc::new(list),
    ))
}

/// The arrays as `concat` takes them.
fn as_dyn(arrays: &[ArrayRef]) -> Vec<&dyn Array> {
    arrays.iter().map(AsRef::as_ref).collect()
}

/// A group's IPC stream back to its arrays.
fn decode_group(bytes: &[u8]) -> Result<Group, SummariesError> {
    let reader = StreamReader::try_new(Cursor::new(bytes), None).map_err(SummariesError::arrow)?;
    let batches: Vec<RecordBatch> = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(SummariesError::arrow)?;
    if batches.len() != 1 {
        return Err(SummariesError::Malformed(format!(
            "a group holds one batch, found {}",
            batches.len()
        )));
    }
    let batch = &batches[0];
    let column_by_name = |name: &str| -> Option<ArrayRef> {
        batch
            .schema()
            .index_of(name)
            .ok()
            .map(|i| Arc::clone(batch.column(i)))
    };
    let required = |name: &'static str| -> Result<ArrayRef, SummariesError> {
        column_by_name(name)
            .ok_or_else(|| SummariesError::Malformed(format!("a group without {name}")))
    };
    if required(COLUMN_FIELD)?.data_type() != &DataType::UInt32 {
        return Err(SummariesError::Malformed(
            "column ids are not UInt32".into(),
        ));
    }
    let superfile = required(SUPERFILE_FIELD)?
        .as_any()
        .downcast_ref::<UInt32Array>()
        .cloned()
        .ok_or_else(|| SummariesError::Malformed("superfile rows are not UInt32".into()))?;
    let null_count = required(NULL_COUNT_FIELD)?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .cloned()
        .ok_or_else(|| SummariesError::Malformed("null counts are not UInt64".into()))?;
    let hll = match column_by_name(HLL_FIELD) {
        Some(array) => Some(
            array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .cloned()
                .ok_or_else(|| SummariesError::Malformed("sketches are not Binary".into()))?,
        ),
        None => None,
    };
    let value_counts = match column_by_name(VALUE_COUNTS_FIELD) {
        Some(array) => Some(
            array
                .as_any()
                .downcast_ref::<ListArray>()
                .cloned()
                .ok_or_else(|| SummariesError::Malformed("value tables are not a List".into()))?,
        ),
        None => None,
    };
    Ok(Group {
        superfile,
        min: required(MIN_FIELD)?,
        max: required(MAX_FIELD)?,
        null_count,
        sum: column_by_name(SUM_FIELD),
        hll,
        value_counts,
    })
}

/// Rows for the encoder from entries' statistics, read in place: every
/// column of every entry, with the entry's position as the superfile
/// row. An entry that reads its statistics from an earlier section hands
/// the encoder that section's rows, which it carries over array for
/// array; nothing is decoded to be encoded again.
pub fn rows_from_entries(entries: &[Arc<SuperfileEntry>]) -> Vec<SummaryRow<'_>> {
    let mut rows = Vec::new();
    for (superfile, entry) in entries.iter().enumerate() {
        for (column, stats) in entry.scalar_stats.columns() {
            rows.push(SummaryRow {
                column,
                superfile: superfile as u32,
                stats,
            });
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use arrow_array::{BooleanArray, Float64Array, Int64Array, StringArray};

    use super::*;

    /// Build the statistics of one column from an array, as the writer
    /// does.
    fn stats_of(array: ArrayRef) -> ScalarStatsAgg {
        ScalarStatsAgg::from_column(&array).expect("a type with statistics")
    }

    fn ids(n: usize) -> Vec<Uuid> {
        (0..n).map(|i| Uuid::from_u128(i as u128 + 1)).collect()
    }

    /// The part the test sections are written for.
    fn part() -> PartId {
        PartId(Uuid::from_u128(0xabc))
    }

    const AMOUNT: FieldId = FieldId(3);
    const KIND: FieldId = FieldId(7);
    const FLAG: FieldId = FieldId(9);
    const RATIO: FieldId = FieldId(11);
    const EMPTY: FieldId = FieldId(12);
    const ONLY_SECOND: FieldId = FieldId(20);

    /// Statistics the test rows borrow from: `(column, superfile, stats)`.
    #[derive(Clone)]
    struct Owned(Vec<(FieldId, u32, ScalarStatsAgg)>);

    impl Owned {
        /// The rows for the encoder, each borrowing its statistics here.
        fn rows(&self) -> Vec<SummaryRow<'_>> {
            self.0
                .iter()
                .map(|(column, superfile, stats)| SummaryRow {
                    column: *column,
                    superfile: *superfile,
                    stats: ColumnStats::Inline(stats),
                })
                .collect()
        }
    }

    /// Two superfiles of mixed columns, one column only in the second.
    fn two_superfiles() -> (Vec<Uuid>, Owned) {
        let first: Vec<(FieldId, ScalarStatsAgg)> = vec![
            (
                AMOUNT,
                stats_of(Arc::new(Int64Array::from(vec![Some(3), None, Some(9)]))),
            ),
            (
                KIND,
                stats_of(Arc::new(StringArray::from(vec!["a", "b", "a"]))),
            ),
            (
                FLAG,
                stats_of(Arc::new(BooleanArray::from(vec![true, false, true]))),
            ),
            (
                RATIO,
                stats_of(Arc::new(Float64Array::from(vec![0.5, 1.5]))),
            ),
            (
                EMPTY,
                stats_of(Arc::new(Int64Array::from(vec![None::<i64>, None]))),
            ),
        ];
        let second: Vec<(FieldId, ScalarStatsAgg)> = vec![
            (AMOUNT, stats_of(Arc::new(Int64Array::from(vec![10, 20])))),
            (KIND, stats_of(Arc::new(StringArray::from(vec!["z"])))),
            (FLAG, stats_of(Arc::new(BooleanArray::from(vec![false])))),
            (
                ONLY_SECOND,
                stats_of(Arc::new(Int64Array::from(vec![1, 2, 3]))),
            ),
        ];
        let mut owned = Vec::new();
        for (superfile, columns) in [first, second].into_iter().enumerate() {
            for (column, stats) in columns {
                owned.push((column, superfile as u32, stats));
            }
        }
        (ids(2), Owned(owned))
    }

    #[test]
    fn a_section_round_trips_every_statistic_by_column() {
        let (ids, owned) = two_superfiles();
        let expected = owned.0.clone();
        let bytes = SummariesSection::encode(part(), &ids, owned.rows()).expect("encode");
        let section = SummariesSection::parse(Bytes::from(bytes)).expect("parse");
        assert_eq!(section.superfile_ids(), ids.as_slice());
        assert_eq!(section.part_id(), Some(part()));
        for (column, superfile, stats) in &expected {
            let back = section
                .stat(*superfile, *column)
                .unwrap_or_else(|| panic!("column {column:?} of superfile {superfile}"));
            assert_eq!(&back, stats, "column {column:?} of superfile {superfile}");
        }
        // Columns a superfile does not hold, and columns nobody holds.
        assert!(section.stat(0, ONLY_SECOND).is_none());
        assert!(section.stat(1, RATIO).is_none());
        assert!(section.stat(1, EMPTY).is_none());
        assert!(section.stat(0, FieldId(99)).is_none());
        assert!(section.stat(5, AMOUNT).is_none(), "no such row");
        let columns: Vec<FieldId> = section.columns().collect();
        assert_eq!(columns, vec![AMOUNT, KIND, FLAG, RATIO, EMPTY, ONLY_SECOND]);
    }

    #[test]
    fn a_column_row_reads_each_statistic_in_place() {
        let (ids, owned) = two_superfiles();
        let bytes = SummariesSection::encode(part(), &ids, owned.rows()).expect("encode");
        let section = SummariesSection::parse(Bytes::from(bytes)).expect("parse");
        let amount = section.column(0, AMOUNT).expect("amount of the first");
        assert_eq!(
            amount.min_max_values(),
            Some((ScalarValue::Int64(Some(3)), ScalarValue::Int64(Some(9))))
        );
        assert_eq!(amount.null_count(), Some(1));
        assert!(!amount.all_null());
        let sum = amount.sum().expect("integers sum");
        assert_eq!(
            ScalarValue::try_from_array(&sum, 0).expect("one row"),
            ScalarValue::Int64(Some(12))
        );
        // Two distinct values: counted exactly, so no sketch, and the
        // table's own sketch is the column's register for register.
        assert!(amount.sketch().is_none());
        let table = amount.value_counts().expect("a value table");
        assert_eq!(table.entries().len(), 2);
        let from_table = amount.values_sketch().expect("the table sketches");
        let from_values: ArrayRef = Arc::new(Int64Array::from(vec![3, 9]));
        let from_values = column_hll(&[from_values]).expect("values sketch");
        assert_eq!(from_table.as_bytes(), from_values.as_bytes());
        assert_eq!(amount.stats(), section.row_map(0)[&AMOUNT]);
        let columns: Vec<FieldId> = section
            .row_columns(0)
            .into_iter()
            .map(|(column, _)| column)
            .collect();
        assert_eq!(columns, vec![AMOUNT, KIND, FLAG, RATIO, EMPTY]);

        let empty = section.column(0, EMPTY).expect("the null column");
        assert!(empty.all_null());
        assert_eq!(empty.null_count(), Some(2));
        assert!(empty.sketch().is_none() && empty.values_sketch().is_none());
        assert!(empty.value_counts().is_none());
        assert!(section.column(0, ONLY_SECOND).is_none());
    }

    #[test]
    fn a_section_names_its_part_and_a_rebuild_differs_by_it() {
        let (ids, owned) = two_superfiles();
        let first = SummariesSection::encode(part(), &ids, owned.rows()).expect("encode");
        let rebuilt = PartId(Uuid::from_u128(0xdef));
        let second = SummariesSection::encode(rebuilt, &ids, owned.rows()).expect("encode");
        assert_ne!(ContentHash::of(&first), ContentHash::of(&second));
        assert_eq!(
            SummariesSection::parse(Bytes::from(second))
                .expect("parse")
                .part_id(),
            Some(rebuilt)
        );
    }

    /// A version 1 section, from before the part id was carried, reads
    /// as one with no part id: the same bytes less the id, with the
    /// group table's offsets moved up by its width.
    #[test]
    fn a_section_from_before_the_part_id_still_reads() {
        let (ids, owned) = two_superfiles();
        let expected = owned.0.clone();
        let current = SummariesSection::encode(part(), &ids, owned.rows()).expect("encode");
        let counts_at = MAGIC.len() + U32_BYTES + ID_BYTES;
        let n_superfiles = u32::from_le_bytes(
            current[counts_at..counts_at + U32_BYTES]
                .try_into()
                .expect("four bytes"),
        ) as usize;
        let n_entries = u32::from_le_bytes(
            current[counts_at + U32_BYTES..counts_at + 2 * U32_BYTES]
                .try_into()
                .expect("four bytes"),
        ) as usize;
        let n_groups = u32::from_le_bytes(
            current[counts_at + 2 * U32_BYTES..counts_at + 3 * U32_BYTES]
                .try_into()
                .expect("four bytes"),
        ) as usize;
        let mut legacy = Vec::with_capacity(current.len() - ID_BYTES);
        legacy.extend_from_slice(MAGIC);
        legacy.extend_from_slice(&VERSION_WITHOUT_PART_ID.to_le_bytes());
        legacy.extend_from_slice(&current[counts_at..]);
        let group_table_at = MAGIC.len()
            + 4 * U32_BYTES
            + n_superfiles * ID_BYTES
            + n_entries * DIRECTORY_ENTRY_BYTES;
        for group in 0..n_groups {
            let at = group_table_at + group * GROUP_ENTRY_BYTES;
            let offset =
                u64::from_le_bytes(legacy[at..at + U64_BYTES].try_into().expect("eight bytes"))
                    - ID_BYTES as u64;
            legacy[at..at + U64_BYTES].copy_from_slice(&offset.to_le_bytes());
        }
        let section = SummariesSection::parse(Bytes::from(legacy)).expect("parse");
        assert_eq!(section.part_id(), None);
        assert_eq!(section.superfile_ids(), ids.as_slice());
        for (column, superfile, stats) in &expected {
            assert_eq!(section.stat(*superfile, *column).as_ref(), Some(stats));
        }
    }

    #[test]
    fn a_row_map_holds_every_column_of_its_superfile() {
        let (ids, owned) = two_superfiles();
        let bytes = SummariesSection::encode(part(), &ids, owned.rows()).expect("encode");
        let section = SummariesSection::parse(Bytes::from(bytes)).expect("parse");
        let first = section.row_map(0);
        assert_eq!(first.len(), 5);
        assert!(first[&EMPTY].all_null());
        assert_eq!(first[&AMOUNT].null_count, Some(1));
        let second = section.row_map(1);
        assert_eq!(second.len(), 4);
        assert!(second.contains_key(&ONLY_SECOND));
        assert_eq!(section.row_of(ids[1]), Some(1));
        assert_eq!(section.row_of(Uuid::nil()), None);
        assert!(section.row_has_statistics(0) && section.row_has_statistics(1));
        assert!(!section.row_has_statistics(2), "no such row");
    }

    #[test]
    fn a_column_retyped_between_superfiles_reads_from_both_groups() {
        let ids = ids(2);
        let owned = Owned(vec![
            (AMOUNT, 0, stats_of(Arc::new(Int64Array::from(vec![1, 2])))),
            (
                AMOUNT,
                1,
                stats_of(Arc::new(StringArray::from(vec!["1", "2"]))),
            ),
        ]);
        let bytes = SummariesSection::encode(part(), &ids, owned.rows()).expect("encode");
        let section = SummariesSection::parse(Bytes::from(bytes)).expect("parse");
        assert_eq!(
            section.stat(0, AMOUNT).expect("int").min.data_type(),
            &DataType::Int64
        );
        assert_eq!(
            section.stat(1, AMOUNT).expect("text").min.data_type(),
            &DataType::Utf8
        );
    }

    #[test]
    fn an_empty_section_holds_nothing() {
        let bytes = SummariesSection::encode(part(), &ids(1), Vec::new()).expect("encode");
        let section = SummariesSection::parse(Bytes::from(bytes)).expect("parse");
        assert!(section.stat(0, AMOUNT).is_none());
        assert!(section.row_map(0).is_empty());
    }

    #[test]
    fn a_section_that_is_not_one_is_refused() {
        assert!(matches!(
            SummariesSection::parse(Bytes::from_static(b"nop")),
            Err(SummariesError::Truncated { .. })
        ));
        assert!(matches!(
            SummariesSection::parse(Bytes::from_static(b"nope")),
            Err(SummariesError::BadMagic)
        ));
        assert!(matches!(
            SummariesSection::parse(Bytes::from_static(b"XXXX\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0")),
            Err(SummariesError::BadMagic)
        ));
        let (ids, owned) = two_superfiles();
        let mut bytes = SummariesSection::encode(part(), &ids, owned.rows()).expect("encode");
        bytes.truncate(bytes.len() / 2);
        assert!(matches!(
            SummariesSection::parse(Bytes::from(bytes)),
            Err(SummariesError::Truncated { .. })
        ));
    }

    /// A section written by a later layout than this engine reads is
    /// refused by its version, which the loader turns into no statistics.
    #[test]
    fn a_section_from_a_newer_layout_is_refused_by_its_version() {
        let (ids, owned) = two_superfiles();
        let mut bytes = SummariesSection::encode(part(), &ids, owned.rows()).expect("encode");
        let newer = VERSION + 1;
        bytes[MAGIC.len()..MAGIC.len() + U32_BYTES].copy_from_slice(&newer.to_le_bytes());
        assert!(matches!(
            SummariesSection::parse(Bytes::from(bytes)),
            Err(SummariesError::Version(version)) if version == newer
        ));
    }

    /// A directory run that reaches past its group's rows is bad bytes;
    /// the group reads as no statistics, and nothing panics.
    #[test]
    fn a_directory_run_past_its_group_reads_as_no_statistics() {
        let (ids, owned) = two_superfiles();
        let mut bytes = SummariesSection::encode(part(), &ids, owned.rows()).expect("encode");
        // The first directory entry's row count, past the end of any group.
        let row_count_at = HEADER_BYTES + ids.len() * ID_BYTES + 3 * U32_BYTES;
        bytes[row_count_at..row_count_at + U32_BYTES].copy_from_slice(&u32::MAX.to_le_bytes());
        let section = SummariesSection::parse(Bytes::from(bytes)).expect("the head parses");
        let first_column = section.columns().next().expect("a column");
        assert!(section.column(0, first_column).is_none());
        assert!(section.stat(1, first_column).is_none());
        // Columns of other groups are untouched.
        assert!(section.column(0, KIND).is_some());
    }

    /// A row the encoder cannot write is left out, not fatal: the
    /// superfile reads as having no statistics for that column.
    #[test]
    fn a_row_that_cannot_be_written_is_left_out() {
        let (ids, mut owned) = two_superfiles();
        let first = owned.0[0].2.clone();
        let mismatched = ScalarStatsAgg {
            max: Arc::new(StringArray::from(vec!["z"])) as ArrayRef,
            ..first.clone()
        };
        owned.0.push((FieldId(40), 0, mismatched));
        owned.0.push((FieldId(41), 7, first));
        let bytes = SummariesSection::encode(part(), &ids, owned.rows()).expect("encode");
        let section = SummariesSection::parse(Bytes::from(bytes)).expect("parse");
        assert!(
            section.column(0, FieldId(40)).is_none(),
            "min and max differ in type"
        );
        assert!(
            section.column(0, FieldId(41)).is_none(),
            "no such superfile row"
        );
        assert!(
            section.column(0, AMOUNT).is_some(),
            "the other rows are written"
        );
    }

    /// Rows read from an earlier section are carried into a new one as
    /// they are, beside freshly decoded rows, and read back the same.
    #[test]
    fn rows_from_a_section_carry_into_the_next_one() {
        let (superfile_ids, owned) = two_superfiles();
        let earlier =
            SummariesSection::encode(part(), &superfile_ids, owned.rows()).expect("encode");
        let earlier = SummariesSection::parse(Bytes::from(earlier)).expect("parse");
        let third = stats_of(Arc::new(Int64Array::from(vec![Some(100), None])));
        let mut rows: Vec<SummaryRow<'_>> = Vec::new();
        for (column, superfile, _) in &owned.0 {
            rows.push(SummaryRow {
                column: *column,
                superfile: *superfile,
                stats: ColumnStats::Section(
                    earlier
                        .column(*superfile, *column)
                        .expect("the earlier section holds the row"),
                ),
            });
        }
        rows.push(SummaryRow {
            column: AMOUNT,
            superfile: 2,
            stats: ColumnStats::Inline(&third),
        });
        let next = SummariesSection::encode(PartId(Uuid::from_u128(0xdef)), &ids(3), rows)
            .expect("encode");
        let next = SummariesSection::parse(Bytes::from(next)).expect("parse");
        for (column, superfile, stats) in &owned.0 {
            assert_eq!(next.stat(*superfile, *column).as_ref(), Some(stats));
        }
        assert_eq!(next.stat(2, AMOUNT).as_ref(), Some(&third));
    }

    #[test]
    fn equal_rows_encode_to_equal_bytes() {
        let (ids, owned) = two_superfiles();
        let once = SummariesSection::encode(part(), &ids, owned.rows()).expect("encode");
        let mut reversed = owned.clone();
        reversed.0.reverse();
        let again = SummariesSection::encode(part(), &ids, reversed.rows()).expect("encode");
        assert_eq!(once, again, "row order does not change the bytes");
    }
}
