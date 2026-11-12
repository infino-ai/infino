// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Range index: for each range-indexed column of a superfile, its rows in
//! value order, so a range predicate (`price > 500`, `ts BETWEEN a AND b`)
//! reads only the rows inside the range instead of scanning the column.
//!
//! Zone maps (the per-superfile, per-row-group and per-page min/max) prune
//! a range well only when the data is ordered on the ranged column. A
//! column whose values are spread over every file and every page — a price,
//! a score, a duration — prunes nothing, and a range over it scans the
//! column. This region is the sorted order that column lacks: every
//! non-null value with the row it sits in, sorted by value and cut into
//! blocks behind a directory of each block's first and last value. A range
//! finds its blocks in the directory and reads those alone.
//!
//! Each value is stored as a `u64` key that sorts as the value does
//! ([`KeyKind`]): a signed integer, date or timestamp with its sign bit
//! flipped, an unsigned integer as itself, a float through the IEEE total
//! order (negative values bit-inverted, positive ones with the sign bit
//! set). Arrow compares floats in that same total order, so a range of keys
//! is exactly a range of values.
//!
//! ```text
//!   header   8 B magic | u32 version | u32 n_columns
//!   column   n_columns × 24 B: u32 field_id | u8 key_kind | u8 row_width |
//!            u16 flags | u64 n_values | u32 first_block | u32 n_blocks
//!   block    n_blocks × 32 B: u64 first_key | u64 last_key |
//!            u64 offset (from the region's start) | u16 n_in_block |
//!            u8 key_width | u8 reserved | u32 crc32c of the block's bytes
//!   crc      u32 crc32c of everything above
//!   blocks   per block: n_in_block keys as (key - first_key) at key_width
//!            bits, then n_in_block row ids at the column's row_width bits
//! ```
//!
//! The header, column entries, block entries and their CRC are the
//! **directory**: its length is recorded in the footer, so a cold reader
//! fetches it in one read and then only the blocks a range needs. Blocks of
//! one column are contiguous and in key order, and the columns' blocks
//! follow one another in column order, so the blocks of a range are one
//! contiguous span. Every multi-byte field is little-endian; a payload is a
//! little-endian bit stream (value `i` occupies bits `i·w .. (i+1)·w`).
//!
//! Building sorts each column's `(key, row)` pairs. A superfile's rows can
//! outnumber what is reasonable to sort in memory, so the pairs are sorted
//! in runs of at most [`RUN_ENTRIES`], each full run spilled to scratch,
//! and the runs merged as the blocks are written.
//!
//! Rows sharing a key are sorted by their `_id` when the build is given the
//! id column, and a column's flags say so ([`TIES_IN_ID_ORDER`]); otherwise
//! by row. A sort that breaks ties by `_id` then takes the first rows of a
//! tied key and needs no other, whatever order the superfile's rows are in
//! (a merge concatenates its inputs in an order of its own); without the
//! flag it must read every row holding that key.

use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    fs::File,
    io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    ops::Range,
    path::Path,
};

use arrow::compute::cast;
use arrow_array::{
    Array, RecordBatch,
    cast::AsArray,
    types::{Decimal128Type, Float64Type, Int64Type, UInt64Type},
};
use arrow_schema::{ArrowError, DataType};
use roaring::RoaringBitmap;

use crate::superfile::{
    bits::{for_each_lane, payload_bytes, put_bits, unpack_lanes_u32, width_of},
    format::{checksum::crc32c, u32_le_at, u64_le_at},
};

/// 8-byte magic at the start of the region: `INF` + `RNG` + `01`. The
/// trailing `01` is part of the region's identity, not its version, which
/// is the `u32` after it.
pub(crate) const MAGIC: &[u8; 8] = b"INFRNG01";

/// The region version this engine writes and reads.
pub(crate) const VERSION: u32 = 1;

/// Values per block. Large enough that a block entry (32 bytes) is noise
/// per value, small enough that the two boundary blocks a range decodes in
/// full are a few kilobytes.
pub(crate) const BLOCK_VALUES: usize = 1024;

/// Entries one in-memory sort run holds before it spills: 1.5 Mi entries
/// of 32 bytes (a `u64` key, the 128-bit order of the row among its key's
/// rows, a `u32` row), 48 MiB, whatever the superfile's size.
const RUN_ENTRIES: usize = 3 << 19;

/// Narrowest row id a block packs. A one-row superfile needs no bits for
/// its row, and a block of one value needs none for its key, so without a
/// floor such a block would take no bytes: it would start where the next
/// one does (or at the region's end), which the directory refuses as a
/// damaged layout. One bit per row keeps every block at least a byte.
const MIN_ROW_WIDTH: u8 = 1;

/// Column flag: the rows sharing a key are stored in `_id` order. Unset on a
/// region built without the ids, and on one built before the flag existed,
/// which wrote zero here; such a region keeps them in row order.
const TIES_IN_ID_ORDER: u16 = 1;

/// The sign bit of a 128-bit id: flipped, the ids order as unsigned values.
const ID_SIGN_BIT: u128 = 1 << 127;

/// Bytes of one spilled entry: the key, the order among its key's rows, the
/// row.
const RUN_RECORD_BYTES: usize = 28;
/// Bytes of a spilled entry's key.
const RUN_KEY_BYTES: usize = 8;
/// Bytes of a spilled entry's order among its key's rows.
const RUN_TIE_BYTES: usize = 16;

/// Header bytes: magic, version, column count.
const HEADER_BYTES: usize = 16;
/// Where the header keeps the version.
const VERSION_AT: usize = 8;
/// Where the header keeps the column count.
const N_COLUMNS_AT: usize = 12;

/// Bytes of one column entry.
const COLUMN_ENTRY_BYTES: usize = 24;
/// Column-entry field offsets.
const COL_KIND_AT: usize = 4;
const COL_ROW_WIDTH_AT: usize = 5;
const COL_FLAGS_AT: usize = 6;
const COL_N_VALUES_AT: usize = 8;
const COL_FIRST_BLOCK_AT: usize = 16;
const COL_N_BLOCKS_AT: usize = 20;

/// Bytes of one block entry.
const BLOCK_ENTRY_BYTES: usize = 32;
/// Block-entry field offsets.
const BLK_LAST_KEY_AT: usize = 8;
const BLK_OFFSET_AT: usize = 16;
const BLK_N_AT: usize = 24;
const BLK_KEY_WIDTH_AT: usize = 26;
const BLK_CRC_AT: usize = 28;

/// Bytes of the directory's CRC.
const CRC_BYTES: usize = 4;

/// The sign bit of a 64-bit key.
const SIGN_BIT: u64 = 1 << 63;

/// How a column's values map onto order-preserving `u64` keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeyKind {
    /// Signed integers, dates and timestamps: the `i64` with its sign bit
    /// flipped.
    Signed,
    /// Unsigned integers: the `u64` itself.
    Unsigned,
    /// Floats: the IEEE total order of the `f64`.
    Float,
}

impl KeyKind {
    /// The kind a column of `data_type` is keyed by, or `None` for a type
    /// the range index does not cover.
    pub(crate) fn of(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _) => Some(Self::Signed),
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                Some(Self::Unsigned)
            }
            DataType::Float16 | DataType::Float32 | DataType::Float64 => Some(Self::Float),
            _ => None,
        }
    }

    fn code(self) -> u8 {
        match self {
            Self::Signed => 0,
            Self::Unsigned => 1,
            Self::Float => 2,
        }
    }

    fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Signed),
            1 => Some(Self::Unsigned),
            2 => Some(Self::Float),
            _ => None,
        }
    }

    /// The type a column of this kind is widened to before keying.
    fn widened(self) -> DataType {
        match self {
            Self::Signed => DataType::Int64,
            Self::Unsigned => DataType::UInt64,
            Self::Float => DataType::Float64,
        }
    }
}

/// Whether keys built from values of type `a` mean the same as keys of
/// type `b`: the same kind and, for dates and timestamps, the same unit. A
/// wider integer or float keeps its values, so its keys too; a day count
/// read as nanoseconds does not. A time zone does not change a timestamp's
/// stored value.
pub(crate) fn same_keys(a: &DataType, b: &DataType) -> bool {
    let is_time = |t: &DataType| {
        matches!(
            t,
            DataType::Date32 | DataType::Date64 | DataType::Timestamp(_, _)
        )
    };
    match (a, b) {
        (DataType::Timestamp(x, _), DataType::Timestamp(y, _)) => x == y,
        _ if is_time(a) || is_time(b) => a == b,
        _ => KeyKind::of(a).is_some() && KeyKind::of(a) == KeyKind::of(b),
    }
}

/// The key of a signed value.
pub(crate) fn signed_key(v: i64) -> u64 {
    (v as u64) ^ SIGN_BIT
}

/// The key of an unsigned value.
pub(crate) fn unsigned_key(v: u64) -> u64 {
    v
}

/// The key of a float: its place in the IEEE total order, which is the
/// order Arrow compares floats in (`-NaN < -inf < … < -0.0 < 0.0 < … <
/// inf < NaN`).
pub(crate) fn float_key(v: f64) -> u64 {
    let bits = v.to_bits();
    match bits & SIGN_BIT {
        0 => bits | SIGN_BIT,
        _ => !bits,
    }
}

/// One column the region indexes: its stable field id, the name it has in
/// the batches the region is built from, and how its values are keyed.
#[derive(Debug, Clone)]
pub(crate) struct RangeColumnSpec {
    pub(crate) name: String,
    pub(crate) field_id: u32,
    pub(crate) kind: KeyKind,
}

/// A finished region in two pieces, as the splice streams it: the
/// directory, then the blocks from wherever they were written. Empty when
/// there was no column to index.
pub(crate) struct RangeRegionOutput<R> {
    pub(crate) directory: Vec<u8>,
    pub(crate) blocks: R,
    pub(crate) blocks_len: u64,
}

impl<R> RangeRegionOutput<R> {
    /// Bytes of the whole region.
    pub(crate) fn len(&self) -> u64 {
        self.directory.len() as u64 + self.blocks_len
    }
}

/// Build the region over `batches`, whose rows are the superfile's rows in
/// order (row `i` of the concatenation is local row `i`). Each batch must
/// carry every column of `columns`, by name. With `id_column` named, rows
/// sharing a key are kept in the order of that column's ids, and the region
/// says so ([`TIES_IN_ID_ORDER`]) when every row has one. Spilled runs and
/// the blocks go to scratch files under `scratch`.
pub(crate) fn build_region<I>(
    batches: I,
    columns: &[RangeColumnSpec],
    id_column: Option<&str>,
    scratch: &Path,
) -> io::Result<RangeRegionOutput<BufReader<File>>>
where
    I: IntoIterator<Item = Result<RecordBatch, ArrowError>>,
{
    let mut sorters: Vec<RunSorter> = columns.iter().map(|_| RunSorter::default()).collect();
    let mut n_rows: u64 = 0;
    let mut ids_complete = id_column.is_some();
    for batch in batches {
        let batch = batch.map_err(io::Error::other)?;
        // Every row of the batch takes a `u32` id, its last row included.
        let end_row = n_rows + batch.num_rows() as u64;
        let (Ok(first_row), Ok(_)) = (u32::try_from(n_rows), u32::try_from(end_row)) else {
            return Err(io::Error::other("a superfile holds at most u32::MAX rows"));
        };
        let ids = id_column
            .and_then(|name| batch.column_by_name(name))
            .and_then(|c| c.as_primitive_opt::<Decimal128Type>());
        ids_complete &= ids.is_some_and(|ids| ids.null_count() == 0);
        // A row's order among the rows sharing its key: its id when it has
        // one, else its row.
        let tie = |row: u32| match ids {
            Some(ids) if ids.is_valid((row - first_row) as usize) => {
                (ids.value((row - first_row) as usize) as u128) ^ ID_SIGN_BIT
            }
            _ => u128::from(row),
        };
        for (spec, sorter) in columns.iter().zip(&mut sorters) {
            let array = batch.column_by_name(&spec.name).ok_or_else(|| {
                io::Error::other(format!(
                    "range index: the rows carry no column {:?}",
                    spec.name
                ))
            })?;
            push_keys(array.as_ref(), spec.kind, first_row, |key, row| {
                sorter.push(key, tie(row), row, scratch)
            })?;
        }
        n_rows += batch.num_rows() as u64;
    }
    let row_width = width_of(n_rows.saturating_sub(1)).max(MIN_ROW_WIDTH);
    let flags = if ids_complete { TIES_IN_ID_ORDER } else { 0 };

    let mut blocks_file = BufWriter::new(tempfile::tempfile_in(scratch)?);
    let mut writer = BlockWriter::new(&mut blocks_file, row_width);
    let mut column_entries = Vec::with_capacity(columns.len());
    for (spec, sorter) in columns.iter().zip(sorters) {
        let first_block = writer.entries.len() as u32;
        let n_values = sorter.drain(scratch, |key, row| writer.push(key, row))?;
        writer.flush_block()?;
        column_entries.push(ColumnEntry {
            field_id: spec.field_id,
            kind: spec.kind,
            row_width,
            flags,
            n_values,
            first_block,
            n_blocks: writer.entries.len() as u32 - first_block,
        });
    }
    let BlockWriter {
        entries,
        written: blocks_len,
        ..
    } = writer;
    blocks_file.flush()?;
    let mut blocks = blocks_file.into_inner().map_err(|e| e.into_error())?;
    blocks.seek(SeekFrom::Start(0))?;

    let directory = encode_directory(&column_entries, &entries);
    Ok(RangeRegionOutput {
        directory,
        blocks: BufReader::new(blocks),
        blocks_len,
    })
}

/// Bytes of the directory over `n_columns` columns and `n_blocks` blocks.
fn directory_len(n_columns: usize, n_blocks: usize) -> usize {
    HEADER_BYTES + n_columns * COLUMN_ENTRY_BYTES + n_blocks * BLOCK_ENTRY_BYTES + CRC_BYTES
}

/// The directory over `columns` and `blocks`, block offsets made relative
/// to the region's start (the blocks follow the directory).
fn encode_directory(columns: &[ColumnEntry], blocks: &[BlockEntry]) -> Vec<u8> {
    let len = directory_len(columns.len(), blocks.len());
    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(columns.len() as u32).to_le_bytes());
    for c in columns {
        out.extend_from_slice(&c.field_id.to_le_bytes());
        out.push(c.kind.code());
        out.push(c.row_width);
        out.extend_from_slice(&c.flags.to_le_bytes());
        out.extend_from_slice(&c.n_values.to_le_bytes());
        out.extend_from_slice(&c.first_block.to_le_bytes());
        out.extend_from_slice(&c.n_blocks.to_le_bytes());
    }
    for b in blocks {
        out.extend_from_slice(&b.first_key.to_le_bytes());
        out.extend_from_slice(&b.last_key.to_le_bytes());
        out.extend_from_slice(&(len as u64 + b.offset).to_le_bytes());
        out.extend_from_slice(&b.n.to_le_bytes());
        out.push(b.key_width);
        out.push(0);
        out.extend_from_slice(&b.crc.to_le_bytes());
    }
    let crc = crc32c(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    debug_assert_eq!(out.len(), len);
    out
}

/// Feed `array`'s non-null values to `sink` as `(key, row)`, row `i` of the
/// array being `first_row + i`.
fn push_keys(
    array: &dyn Array,
    kind: KeyKind,
    first_row: u32,
    mut sink: impl FnMut(u64, u32) -> io::Result<()>,
) -> io::Result<()> {
    let widened = cast(array, &kind.widened()).map_err(io::Error::other)?;
    let row_of = |i: usize| first_row + i as u32;
    match kind {
        KeyKind::Signed => {
            for (i, v) in widened.as_primitive::<Int64Type>().iter().enumerate() {
                if let Some(v) = v {
                    sink(signed_key(v), row_of(i))?;
                }
            }
        }
        KeyKind::Unsigned => {
            for (i, v) in widened.as_primitive::<UInt64Type>().iter().enumerate() {
                if let Some(v) = v {
                    sink(unsigned_key(v), row_of(i))?;
                }
            }
        }
        KeyKind::Float => {
            for (i, v) in widened.as_primitive::<Float64Type>().iter().enumerate() {
                if let Some(v) = v {
                    sink(float_key(v), row_of(i))?;
                }
            }
        }
    }
    Ok(())
}

/// One sort entry: a key, the row's order among the rows sharing it (high
/// then low 64 bits), and the row.
type RunEntry = (u64, u64, u64, u32);

/// One column's `(key, row)` pairs, sorted by key and then the order among
/// the key's rows, in runs: the open run in memory, every full one spilled
/// to scratch sorted.
#[derive(Default)]
struct RunSorter {
    open: Vec<RunEntry>,
    spilled: Vec<File>,
}

impl RunSorter {
    fn push(&mut self, key: u64, tie: u128, row: u32, scratch: &Path) -> io::Result<()> {
        self.open
            .push((key, (tie >> u64::BITS) as u64, tie as u64, row));
        if self.open.len() == RUN_ENTRIES {
            self.spill(scratch)?;
        }
        Ok(())
    }

    /// Sort the open run and write it to a scratch file.
    fn spill(&mut self, scratch: &Path) -> io::Result<()> {
        self.open.sort_unstable();
        let mut out = BufWriter::new(tempfile::tempfile_in(scratch)?);
        for &(key, tie_hi, tie_lo, row) in &self.open {
            out.write_all(&key.to_le_bytes())?;
            let tie = (u128::from(tie_hi) << u64::BITS) | u128::from(tie_lo);
            out.write_all(&tie.to_le_bytes())?;
            out.write_all(&row.to_le_bytes())?;
        }
        out.flush()?;
        let mut file = out.into_inner().map_err(|e| e.into_error())?;
        file.seek(SeekFrom::Start(0))?;
        self.spilled.push(file);
        self.open.clear();
        Ok(())
    }

    /// Every pair, by key and then the order among the key's rows, to
    /// `sink` as `(key, row)`; returns how many.
    fn drain(
        mut self,
        scratch: &Path,
        mut sink: impl FnMut(u64, u32) -> io::Result<()>,
    ) -> io::Result<u64> {
        if self.spilled.is_empty() {
            self.open.sort_unstable();
            for &(key, _, _, row) in &self.open {
                sink(key, row)?;
            }
            return Ok(self.open.len() as u64);
        }
        if !self.open.is_empty() {
            self.spill(scratch)?;
        }
        let mut runs: Vec<BufReader<File>> = self.spilled.into_iter().map(BufReader::new).collect();
        let mut heap = BinaryHeap::with_capacity(runs.len());
        for (i, run) in runs.iter_mut().enumerate() {
            if let Some(entry) = read_record(run)? {
                heap.push(Reverse((entry, i)));
            }
        }
        let mut n = 0u64;
        while let Some(Reverse(((key, _, _, row), i))) = heap.pop() {
            sink(key, row)?;
            n += 1;
            if let Some(entry) = read_record(&mut runs[i])? {
                heap.push(Reverse((entry, i)));
            }
        }
        Ok(n)
    }
}

/// The next entry of a spilled run, `None` at its end.
fn read_record(run: &mut impl Read) -> io::Result<Option<RunEntry>> {
    let mut record = [0u8; RUN_RECORD_BYTES];
    match run.read_exact(&mut record) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let (key, rest) = record.split_at(RUN_KEY_BYTES);
    let (tie, row) = rest.split_at(RUN_TIE_BYTES);
    let key = u64::from_le_bytes(key.try_into().expect("8 bytes"));
    let tie = u128::from_le_bytes(tie.try_into().expect("16 bytes"));
    let row = u32::from_le_bytes(row.try_into().expect("4 bytes"));
    Ok(Some((key, (tie >> u64::BITS) as u64, tie as u64, row)))
}

/// One column's entry, before encoding.
struct ColumnEntry {
    field_id: u32,
    kind: KeyKind,
    row_width: u8,
    flags: u16,
    n_values: u64,
    first_block: u32,
    n_blocks: u32,
}

/// One block's entry, before encoding; `offset` is from the blocks' start.
struct BlockEntry {
    first_key: u64,
    last_key: u64,
    offset: u64,
    n: u16,
    key_width: u8,
    crc: u32,
}

/// Packs sorted pairs into blocks of [`BLOCK_VALUES`] and writes them out.
struct BlockWriter<W: Write> {
    out: W,
    written: u64,
    row_width: u8,
    entries: Vec<BlockEntry>,
    keys: Vec<u64>,
    rows: Vec<u32>,
    buf: Vec<u8>,
}

impl<W: Write> BlockWriter<W> {
    fn new(out: W, row_width: u8) -> Self {
        Self {
            out,
            written: 0,
            row_width,
            entries: Vec::new(),
            keys: Vec::with_capacity(BLOCK_VALUES),
            rows: Vec::with_capacity(BLOCK_VALUES),
            buf: Vec::new(),
        }
    }

    fn push(&mut self, key: u64, row: u32) -> io::Result<()> {
        self.keys.push(key);
        self.rows.push(row);
        if self.keys.len() == BLOCK_VALUES {
            self.flush_block()?;
        }
        Ok(())
    }

    /// Encode and write the open block, if it holds anything.
    fn flush_block(&mut self) -> io::Result<()> {
        let n = self.keys.len();
        if n == 0 {
            return Ok(());
        }
        let first_key = self.keys[0];
        let last_key = self.keys[n - 1];
        let key_width = width_of(last_key - first_key);
        let key_bytes = payload_bytes(n, key_width);
        let buf = &mut self.buf;
        buf.clear();
        buf.resize(key_bytes + payload_bytes(n, self.row_width), 0);
        let (key_payload, row_payload) = buf.split_at_mut(key_bytes);
        for (i, &key) in self.keys.iter().enumerate() {
            put_bits(
                key_payload,
                i * key_width as usize,
                key - first_key,
                key_width,
            );
        }
        for (i, &row) in self.rows.iter().enumerate() {
            put_bits(
                row_payload,
                i * self.row_width as usize,
                u64::from(row),
                self.row_width,
            );
        }
        self.out.write_all(buf)?;
        self.entries.push(BlockEntry {
            first_key,
            last_key,
            offset: self.written,
            n: n as u16,
            key_width,
            crc: crc32c(buf),
        });
        self.written += buf.len() as u64;
        self.keys.clear();
        self.rows.clear();
        Ok(())
    }
}

/// A parsed directory: where every column's blocks are and what they hold.
#[derive(Debug)]
pub(crate) struct RangeDirectory {
    columns: Vec<ColumnDir>,
    blocks: Vec<BlockDir>,
    region_len: u64,
}

#[derive(Debug, Clone, Copy)]
struct ColumnDir {
    field_id: u32,
    kind: KeyKind,
    row_width: u8,
    flags: u16,
    n_values: u64,
    first_block: u32,
    n_blocks: u32,
}

#[derive(Debug, Clone, Copy)]
struct BlockDir {
    first_key: u64,
    last_key: u64,
    offset: u64,
    n: u16,
    key_width: u8,
    crc: u32,
}

impl RangeDirectory {
    /// Parse the directory from `bytes`, which must start at the region's
    /// first byte and hold at least the directory; `region_len` is the
    /// whole region's length, the bound every block must fit in.
    pub(crate) fn parse(bytes: &[u8], region_len: u64) -> Result<Self, String> {
        let truncated = || "range index: truncated directory".to_string();
        if bytes.get(..MAGIC.len()) != Some(MAGIC.as_slice()) {
            return Err("range index: bad magic".into());
        }
        let version = u32_le_at(bytes, VERSION_AT).ok_or_else(truncated)?;
        if version != VERSION {
            return Err(format!(
                "range index: version {version}, this engine reads {VERSION}"
            ));
        }
        let n_columns = u32_le_at(bytes, N_COLUMNS_AT).ok_or_else(truncated)? as usize;
        // The CRC that refuses a damaged count is checked only after the
        // column entries are read, so the count is bounded by the bytes
        // before anything is sized by it: one flipped bit in it would
        // otherwise size an allocation of gigabytes and abort the process.
        if directory_len(n_columns, 0) > bytes.len() {
            return Err(truncated());
        }
        let mut columns = Vec::with_capacity(n_columns);
        let mut n_blocks_total = 0usize;
        for c in 0..n_columns {
            let at = HEADER_BYTES + c * COLUMN_ENTRY_BYTES;
            let entry = bytes
                .get(at..at + COLUMN_ENTRY_BYTES)
                .ok_or_else(truncated)?;
            let column = ColumnDir {
                field_id: u32_le_at(entry, 0).ok_or_else(truncated)?,
                kind: KeyKind::from_code(entry[COL_KIND_AT])
                    .ok_or("range index: unknown key kind")?,
                row_width: entry[COL_ROW_WIDTH_AT],
                flags: u16::from_le_bytes([entry[COL_FLAGS_AT], entry[COL_FLAGS_AT + 1]]),
                n_values: u64_le_at(entry, COL_N_VALUES_AT).ok_or_else(truncated)?,
                first_block: u32_le_at(entry, COL_FIRST_BLOCK_AT).ok_or_else(truncated)?,
                n_blocks: u32_le_at(entry, COL_N_BLOCKS_AT).ok_or_else(truncated)?,
            };
            if column.row_width > u32::BITS as u8 {
                return Err("range index: row width past 32 bits".into());
            }
            if column.first_block as usize != n_blocks_total {
                return Err("range index: column blocks are not contiguous".into());
            }
            n_blocks_total += column.n_blocks as usize;
            columns.push(column);
        }
        let len = directory_len(n_columns, n_blocks_total);
        let directory = bytes.get(..len).ok_or_else(truncated)?;
        let (body, crc) = directory.split_at(len - CRC_BYTES);
        if Some(crc32c(body)) != u32_le_at(crc, 0) {
            return Err("range index: directory CRC mismatch".into());
        }
        let blocks_at = HEADER_BYTES + n_columns * COLUMN_ENTRY_BYTES;
        let mut blocks = Vec::with_capacity(n_blocks_total);
        for b in 0..n_blocks_total {
            let entry = &body[blocks_at + b * BLOCK_ENTRY_BYTES..][..BLOCK_ENTRY_BYTES];
            let block = BlockDir {
                first_key: u64_le_at(entry, 0).ok_or_else(truncated)?,
                last_key: u64_le_at(entry, BLK_LAST_KEY_AT).ok_or_else(truncated)?,
                offset: u64_le_at(entry, BLK_OFFSET_AT).ok_or_else(truncated)?,
                n: u16::from_le_bytes([entry[BLK_N_AT], entry[BLK_N_AT + 1]]),
                key_width: entry[BLK_KEY_WIDTH_AT],
                crc: u32_le_at(entry, BLK_CRC_AT).ok_or_else(truncated)?,
            };
            if block.first_key > block.last_key
                || block.n == 0
                || usize::from(block.n) > BLOCK_VALUES
            {
                return Err("range index: malformed block entry".into());
            }
            blocks.push(block);
        }
        // The blocks follow the directory, each starting past the one
        // before, and the last ends inside the region: every block's span
        // is the gap to the next one (or to the region's end), never empty.
        let mut end = len as u64;
        for (b, block) in blocks.iter().enumerate() {
            let ascends = match b {
                0 => block.offset == end,
                _ => block.offset > end,
            };
            if !ascends {
                return Err("range index: block offsets do not ascend".into());
            }
            end = block.offset;
        }
        if (!blocks.is_empty() && end >= region_len) || region_len < len as u64 {
            return Err("range index: a block lies past the region".into());
        }
        // A count takes whole blocks' values from their entries, so each
        // column's entries must add up to the values it holds.
        for column in &columns {
            let first = column.first_block as usize;
            let held: u64 = blocks[first..first + column.n_blocks as usize]
                .iter()
                .map(|b| u64::from(b.n))
                .sum();
            if held != column.n_values {
                return Err("range index: block counts do not add up to the column's".into());
            }
        }
        Ok(Self {
            columns,
            blocks,
            region_len,
        })
    }

    /// The column with stable id `field_id`, `None` when it is not indexed.
    pub(crate) fn column(&self, field_id: u32) -> Option<RangeColumn<'_>> {
        let column = self.columns.iter().find(|c| c.field_id == field_id)?;
        Some(RangeColumn {
            directory: self,
            column: *column,
        })
    }

    /// Where block `b` lies in the region.
    fn block_span(&self, b: usize) -> Range<u64> {
        let start = self.blocks[b].offset;
        let end = self
            .blocks
            .get(b + 1)
            .map_or(self.region_len, |next| next.offset);
        start..end
    }
}

/// One indexed column of a [`RangeDirectory`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct RangeColumn<'a> {
    directory: &'a RangeDirectory,
    column: ColumnDir,
}

impl RangeColumn<'_> {
    /// How the column's values are keyed.
    pub(crate) fn kind(&self) -> KeyKind {
        self.column.kind
    }

    /// Whether the rows sharing a key are stored in `_id` order
    /// ([`TIES_IN_ID_ORDER`]).
    pub(crate) fn ties_in_id_order(&self) -> bool {
        self.column.flags & TIES_IN_ID_ORDER != 0
    }

    /// The column's non-null values in this superfile.
    pub(crate) fn n_values(&self) -> u64 {
        self.column.n_values
    }

    /// The column's blocks, as indices into the directory.
    fn blocks(&self) -> Range<usize> {
        let first = self.column.first_block as usize;
        first..first + self.column.n_blocks as usize
    }

    /// The blocks that may hold a key in `lo..=hi`: from the first whose
    /// last key reaches `lo` to the last whose first key is within `hi`.
    /// Empty when `lo > hi`.
    pub(crate) fn blocks_in(&self, lo: u64, hi: u64) -> Range<usize> {
        let all = self.blocks();
        if lo > hi {
            return all.start..all.start;
        }
        let blocks = &self.directory.blocks[all.clone()];
        let start = blocks.partition_point(|b| b.last_key < lo);
        let end = blocks.partition_point(|b| b.first_key <= hi).max(start);
        all.start + start..all.start + end
    }

    /// The values in `lo..=hi`, estimated from the directory alone: every
    /// value of a block wholly inside the range, and of a block the range
    /// cuts, the share of the block's key span it covers. Off by at most
    /// the two cut blocks' values, and no block is read.
    pub(crate) fn estimate(&self, lo: u64, hi: u64) -> u64 {
        self.blocks_in(lo, hi)
            .map(|b| {
                let block = &self.directory.blocks[b];
                let n = u64::from(block.n);
                if lo <= block.first_key && block.last_key <= hi {
                    return n;
                }
                let span = (block.last_key - block.first_key) as f64 + 1.0;
                let covered = (hi.min(block.last_key) - lo.max(block.first_key)) as f64 + 1.0;
                (n as f64 * covered / span).ceil() as u64
            })
            .sum()
    }

    /// Where `blocks` (a run [`Self::blocks_in`] returned) lie in the
    /// region, as one contiguous span; empty for no block.
    pub(crate) fn span_of(&self, blocks: Range<usize>) -> Range<u64> {
        if blocks.is_empty() {
            return 0..0;
        }
        let first = self.directory.block_span(blocks.start);
        let last = self.directory.block_span(blocks.end - 1);
        first.start..last.end
    }

    /// Where block `b` lies in the region.
    pub(crate) fn block_span(&self, b: usize) -> Range<u64> {
        self.directory.block_span(b)
    }

    /// Whether block `b` lies wholly inside `lo..=hi`, so every row in it
    /// matches without decoding a key.
    fn covered(&self, b: usize, lo: u64, hi: u64) -> bool {
        let block = &self.directory.blocks[b];
        lo <= block.first_key && block.last_key <= hi
    }

    /// The blocks of `lo..=hi` a count must decode: the ones the range
    /// cuts, at most the first and the last.
    pub(crate) fn boundary_blocks(&self, lo: u64, hi: u64) -> Vec<usize> {
        self.blocks_in(lo, hi)
            .filter(|&b| !self.covered(b, lo, hi))
            .collect()
    }

    /// How many values lie in `lo..=hi`. Blocks wholly inside the range
    /// count from the directory; `boundary` holds the bytes of every block
    /// [`Self::boundary_blocks`] names, keyed by its index.
    pub(crate) fn count(
        &self,
        lo: u64,
        hi: u64,
        boundary: &[(usize, &[u8])],
    ) -> Result<u64, String> {
        let mut n = 0u64;
        for b in self.blocks_in(lo, hi) {
            if self.covered(b, lo, hi) {
                n += u64::from(self.directory.blocks[b].n);
                continue;
            }
            let bytes = boundary
                .iter()
                .find(|(at, _)| *at == b)
                .map(|(_, bytes)| *bytes)
                .ok_or("range index: a boundary block was not fetched")?;
            self.for_each_in_block(b, bytes, lo, hi, |_| n += 1)?;
        }
        Ok(n)
    }

    /// Every row holding a value in `lo..=hi`, from `span`, the bytes of
    /// [`Self::span_of`]`(`[`Self::blocks_in`]`(lo, hi))`.
    pub(crate) fn rows(&self, lo: u64, hi: u64, span: &[u8]) -> Result<RoaringBitmap, String> {
        let blocks = self.blocks_in(lo, hi);
        let base = self.span_of(blocks.clone()).start;
        let mut rows: Vec<u32> = Vec::new();
        for b in blocks {
            let at = self.directory.block_span(b);
            let bytes = span
                .get((at.start - base) as usize..(at.end - base) as usize)
                .ok_or("range index: the fetched span is short")?;
            self.for_each_in_block(b, bytes, lo, hi, |row| rows.push(row))?;
        }
        rows.sort_unstable();
        RoaringBitmap::from_sorted_iter(rows).map_err(|e| format!("range index: {e}"))
    }

    /// Every `(key, row)` of block `b` (its `bytes`) whose key lies in
    /// `lo..=hi`, in key order.
    pub(crate) fn pairs_in_block(
        &self,
        b: usize,
        bytes: &[u8],
        lo: u64,
        hi: u64,
    ) -> Result<Vec<(u64, u32)>, String> {
        let block = &self.directory.blocks[b];
        let mut keys = Vec::with_capacity(usize::from(block.n));
        let mut rows = Vec::with_capacity(usize::from(block.n));
        self.for_each_pair_in_block(b, bytes, |key, row| {
            if lo <= key && key <= hi {
                keys.push(key);
                rows.push(row);
            }
        })?;
        Ok(keys.into_iter().zip(rows).collect())
    }

    /// Each `(key, row)` of block `b` (its `bytes`), in key order.
    fn for_each_pair_in_block(
        &self,
        b: usize,
        bytes: &[u8],
        mut sink: impl FnMut(u64, u32),
    ) -> Result<(), String> {
        let block = &self.directory.blocks[b];
        if crc32c(bytes) != block.crc {
            return Err("range index: block CRC mismatch".into());
        }
        let n = usize::from(block.n);
        let short = || "range index: block shorter than its payload".to_string();
        let row_payload = bytes
            .get(payload_bytes(n, block.key_width)..)
            .ok_or_else(short)?;
        let mut rows = vec![0u32; n];
        unpack_lanes_u32(row_payload, 0, self.column.row_width, &mut rows).ok_or_else(short)?;
        let mut i = 0usize;
        for_each_lane(bytes, 0, n, block.key_width, |delta| {
            sink(block.first_key + delta, rows[i]);
            i += 1;
        })
        .ok_or_else(short)
    }

    /// Each row of block `b` (its `bytes`) whose key lies in `lo..=hi`.
    fn for_each_in_block(
        &self,
        b: usize,
        bytes: &[u8],
        lo: u64,
        hi: u64,
        mut sink: impl FnMut(u32),
    ) -> Result<(), String> {
        let block = &self.directory.blocks[b];
        if crc32c(bytes) != block.crc {
            return Err("range index: block CRC mismatch".into());
        }
        let n = usize::from(block.n);
        let key_bytes = payload_bytes(n, block.key_width);
        let short = || "range index: block shorter than its payload".to_string();
        let row_payload = bytes.get(key_bytes..).ok_or_else(short)?;
        let mut rows = vec![0u32; n];
        unpack_lanes_u32(row_payload, 0, self.column.row_width, &mut rows).ok_or_else(short)?;
        if self.covered(b, lo, hi) {
            rows.into_iter().for_each(sink);
            return Ok(());
        }
        let mut i = 0usize;
        for_each_lane(bytes, 0, n, block.key_width, |delta| {
            let key = block.first_key + delta;
            if lo <= key && key <= hi {
                sink(rows[i]);
            }
            i += 1;
        })
        .ok_or_else(short)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::compute::kernels::cmp::{eq, gt};
    use arrow_array::{
        BooleanArray, Date32Array, Decimal128Array, Float64Array, Int32Array, Int64Array,
        LargeStringArray, UInt16Array,
    };
    use arrow_schema::{Field, Schema, TimeUnit};
    use bytes::Bytes;

    use super::*;
    use crate::{
        superfile::{
            BytesLazyByteSource, LazyByteSource, SuperfileReader,
            builder::{BuilderOptions, CarryScope, FtsConfig, SuperfileBuilder},
            page_index::IndexColumns,
        },
        supertable::schema::{FieldId, with_field_id},
    };

    /// Rows of the superfile-level fixture: past three blocks, so ranges
    /// cut blocks and cover whole ones.
    const SUPERFILE_ROWS: usize = 3 * BLOCK_VALUES + 123;
    /// Rows of the fixture's first batch; the rest arrive in a second.
    const FIRST_BATCH_ROWS: usize = 1000;
    /// Every this-many rows the fixture's `price` is null.
    const NULL_EVERY: usize = 13;

    /// Read a built region back whole.
    fn whole(out: RangeRegionOutput<BufReader<File>>) -> Vec<u8> {
        let RangeRegionOutput {
            mut directory,
            mut blocks,
            ..
        } = out;
        blocks.read_to_end(&mut directory).expect("read blocks");
        directory
    }

    fn spec(name: &str, field_id: u32, kind: KeyKind) -> RangeColumnSpec {
        RangeColumnSpec {
            name: name.into(),
            field_id,
            kind,
        }
    }

    /// The rows `pred` holds for: the brute-force oracle.
    fn oracle<T: Copy>(values: &[Option<T>], pred: impl Fn(T) -> bool) -> RoaringBitmap {
        values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_some_and(&pred))
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Rows and count of `lo..=hi` over a built region, through the same
    /// span and boundary reads a reader makes.
    fn query(region: &[u8], field_id: u32, lo: u64, hi: u64) -> (RoaringBitmap, u64) {
        let directory = RangeDirectory::parse(region, region.len() as u64).expect("parse");
        let column = directory.column(field_id).expect("indexed");
        let span = column.span_of(column.blocks_in(lo, hi));
        let rows = column
            .rows(lo, hi, &region[span.start as usize..span.end as usize])
            .expect("rows");
        let boundary: Vec<(usize, &[u8])> = column
            .boundary_blocks(lo, hi)
            .into_iter()
            .map(|b| {
                let at = column.block_span(b);
                (b, &region[at.start as usize..at.end as usize])
            })
            .collect();
        let count = column.count(lo, hi, &boundary).expect("count");
        assert!(
            column.estimate(lo, hi).abs_diff(count) <= 2 * BLOCK_VALUES as u64,
            "the estimate is off by at most the two cut blocks"
        );
        (rows, count)
    }

    /// Keys carry over a change only when they still count the same unit.
    #[test]
    fn keys_carry_over_a_widening_but_not_a_change_of_unit() {
        let ns = DataType::Timestamp(TimeUnit::Nanosecond, None);
        let ns_utc = DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()));
        let ms = DataType::Timestamp(TimeUnit::Millisecond, None);
        assert!(same_keys(&DataType::Int32, &DataType::Int64));
        assert!(same_keys(&DataType::UInt8, &DataType::UInt64));
        assert!(same_keys(&DataType::Float32, &DataType::Float64));
        assert!(same_keys(&DataType::Date32, &DataType::Date32));
        assert!(same_keys(&ns, &ns_utc));
        assert!(!same_keys(&DataType::Date32, &ns));
        assert!(!same_keys(&DataType::Date32, &DataType::Date64));
        assert!(!same_keys(&ms, &ns));
        assert!(!same_keys(&DataType::Int64, &ns));
        assert!(!same_keys(&DataType::Int64, &DataType::Float64));
        assert!(!same_keys(&DataType::LargeUtf8, &DataType::LargeUtf8));
    }

    /// A one-row superfile's region, with two indexed columns, parses and
    /// answers: each column's one-value block still takes bytes of its own.
    #[test]
    fn a_one_row_region_parses_and_answers() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![7])),
                Arc::new(Float64Array::from(vec![2.5])),
            ],
        );
        let scratch = tempfile::tempdir().expect("scratch");
        let region = whole(
            build_region(
                [batch],
                &[spec("a", 1, KeyKind::Signed), spec("b", 2, KeyKind::Float)],
                None,
                scratch.path(),
            )
            .expect("build"),
        );
        let one: RoaringBitmap = [0].into_iter().collect();
        assert_eq!(
            query(&region, 1, signed_key(0), signed_key(10)),
            (one.clone(), 1)
        );
        assert_eq!(
            query(&region, 1, signed_key(8), signed_key(10)),
            (RoaringBitmap::new(), 0)
        );
        assert_eq!(query(&region, 2, float_key(2.5), float_key(2.5)), (one, 1));
    }

    #[test]
    fn keys_sort_as_their_values_do() {
        let ints = [i64::MIN, -5, -1, 0, 1, 7, i64::MAX];
        for pair in ints.windows(2) {
            assert!(signed_key(pair[0]) < signed_key(pair[1]), "{pair:?}");
        }
        let floats = [
            -f64::NAN,
            f64::NEG_INFINITY,
            -1.5,
            -0.0,
            0.0,
            1e-300,
            2.5,
            f64::INFINITY,
            f64::NAN,
        ];
        for pair in floats.windows(2) {
            assert_eq!(
                pair[0].total_cmp(&pair[1]),
                float_key(pair[0]).cmp(&float_key(pair[1])),
                "{pair:?}"
            );
        }
        assert!(unsigned_key(3) < unsigned_key(u64::MAX));
    }

    /// The float keys follow the IEEE total order; this pins that Arrow's
    /// comparison kernels, which evaluate a SQL predicate, use the same one:
    /// `-0.0` below `0.0`, NaN above every number and equal to itself.
    #[test]
    fn arrow_compares_floats_in_the_order_the_keys_follow() {
        let values = Float64Array::from(vec![-0.0, 0.0, f64::NAN, -f64::NAN, 1.0]);
        let zero = Float64Array::new_scalar(0.0);
        let nan = Float64Array::new_scalar(f64::NAN);
        let as_vec = |a: BooleanArray| -> Vec<bool> { a.iter().map(|v| v == Some(true)).collect() };
        assert_eq!(
            as_vec(gt(&values, &zero).expect("gt")),
            vec![false, false, true, false, true]
        );
        assert_eq!(
            as_vec(eq(&values, &zero).expect("eq")),
            vec![false, true, false, false, false]
        );
        assert_eq!(
            as_vec(eq(&values, &nan).expect("eq")),
            vec![false, false, true, false, false]
        );
    }

    #[test]
    fn a_range_reads_exactly_the_rows_inside_it_across_many_blocks() {
        // More values than five blocks, with nulls and repeats, over
        // several batches.
        const ROWS: usize = 5 * BLOCK_VALUES + 37;
        let values: Vec<Option<f64>> = (0..ROWS)
            .map(|i| match i % 11 {
                0 => None,
                _ => Some(((i * 7919) % 2003) as f64 - 1000.0 + (i % 3) as f64 * 0.25),
            })
            .collect();
        let schema = Arc::new(Schema::new(vec![Field::new("p", DataType::Float64, true)]));
        let batches: Vec<Result<RecordBatch, ArrowError>> = values
            .chunks(1000)
            .map(|chunk| {
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Float64Array::from(chunk.to_vec()))],
                )
            })
            .collect();
        let scratch = tempfile::tempdir().expect("scratch");
        let region = whole(
            build_region(
                batches,
                &[spec("p", 9, KeyKind::Float)],
                None,
                scratch.path(),
            )
            .expect("build"),
        );
        let directory = RangeDirectory::parse(&region, region.len() as u64).expect("parse");
        let column = directory.column(9).expect("indexed");
        assert_eq!(column.kind(), KeyKind::Float);
        assert_eq!(
            column.n_values(),
            values.iter().filter(|v| v.is_some()).count() as u64
        );
        assert!(directory.column(8).is_none());

        for (lo, hi) in [
            (-1000.0, 1002.5),
            (-3.0, 4.0),
            (0.0, 0.0),
            (500.25, 500.25),
            (999.0, 5000.0),
            (-5000.0, -999.5),
            (10.0, 5.0),
        ] {
            let (rows, count) = query(&region, 9, float_key(lo), float_key(hi));
            let want = oracle(&values, |v: f64| lo <= v && v <= hi);
            assert_eq!(rows, want, "rows of [{lo}, {hi}]");
            assert_eq!(count, want.len(), "count of [{lo}, {hi}]");
        }
    }

    #[test]
    fn spilled_runs_merge_into_one_sorted_order() {
        let scratch = tempfile::tempdir().expect("scratch");
        let mut sorter = RunSorter::default();
        // Three runs: two spilled by hand, one left open.
        for (i, key) in [5u64, 1, 9, 3, 7, 3, 0, 8].iter().enumerate() {
            sorter
                .push(*key, i as u128, i as u32, scratch.path())
                .expect("push");
            if i == 2 || i == 5 {
                sorter.spill(scratch.path()).expect("spill");
            }
        }
        let mut out = Vec::new();
        let n = sorter
            .drain(scratch.path(), |k, r| {
                out.push((k, r));
                Ok(())
            })
            .expect("drain");
        assert_eq!(n, 8);
        let mut want = out.clone();
        want.sort_unstable();
        assert_eq!(out, want);
        assert_eq!(out.first(), Some(&(0, 6)));
    }

    #[test]
    fn several_columns_of_every_kind_index_side_by_side() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("u", DataType::UInt16, true),
            Field::new("d", DataType::Date32, true),
        ]));
        let a: Vec<Option<i32>> = (0..3000).map(|i| Some(1500 - i)).collect();
        let u: Vec<Option<u16>> = (0..3000)
            .map(|i| (i % 5 != 0).then_some((i % 97) as u16))
            .collect();
        let d: Vec<Option<i32>> = (0..3000).map(|i| Some(19_000 + i % 400)).collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(a)),
                Arc::new(UInt16Array::from(u.clone())),
                Arc::new(Date32Array::from(d.clone())),
            ],
        );
        let scratch = tempfile::tempdir().expect("scratch");
        let region = whole(
            build_region(
                [batch],
                &[
                    spec("a", 1, KeyKind::Signed),
                    spec("u", 2, KeyKind::Unsigned),
                    spec("d", 3, KeyKind::Signed),
                ],
                None,
                scratch.path(),
            )
            .expect("build"),
        );
        let (rows, count) = query(&region, 1, signed_key(-10), signed_key(10));
        let want: RoaringBitmap = (1490..=1510).collect();
        assert_eq!((rows, count), (want.clone(), want.len()));
        let (rows, _) = query(&region, 2, unsigned_key(96), unsigned_key(96));
        assert_eq!(rows, oracle(&u, |v: u16| v == 96));
        let (rows, _) = query(&region, 3, signed_key(19_100), signed_key(19_120));
        assert_eq!(rows, oracle(&d, |v: i32| (19_100..=19_120).contains(&v)));
    }

    #[test]
    fn a_damaged_directory_or_block_is_refused() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from_iter_values(0..4000))],
        );
        let scratch = tempfile::tempdir().expect("scratch");
        let region = whole(
            build_region(
                [batch],
                &[spec("a", 1, KeyKind::Signed)],
                None,
                scratch.path(),
            )
            .expect("build"),
        );
        let mut bad_dir = region.clone();
        bad_dir[HEADER_BYTES] ^= 1;
        assert!(RangeDirectory::parse(&bad_dir, bad_dir.len() as u64).is_err());
        assert!(RangeDirectory::parse(&region[..HEADER_BYTES], region.len() as u64).is_err());
        // A damaged column count, magic and version intact, is refused
        // before it sizes anything.
        let mut bad_count = region.clone();
        bad_count[N_COLUMNS_AT..N_COLUMNS_AT + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(RangeDirectory::parse(&bad_count, bad_count.len() as u64).is_err());

        let directory = RangeDirectory::parse(&region, region.len() as u64).expect("parse");
        let column = directory.column(1).expect("indexed");
        let (lo, hi) = (signed_key(0), signed_key(4000));
        let span = column.span_of(column.blocks_in(lo, hi));
        let mut bytes = region[span.start as usize..span.end as usize].to_vec();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x80;
        assert!(column.rows(lo, hi, &bytes).is_err());
    }

    /// The fixture's `price` and `qty` columns, by row.
    fn fixture_values() -> (Vec<Option<f64>>, Vec<i32>) {
        let price = (0..SUPERFILE_ROWS)
            .map(|i| (i % NULL_EVERY != 0).then(|| ((i * 7919) % 1000) as f64 / 4.0 - 50.0))
            .collect();
        let qty = (0..SUPERFILE_ROWS)
            .map(|i| ((i * 31) % 200) as i32 - 100)
            .collect();
        (price, qty)
    }

    /// A superfile of the fixture's rows with `price` and `qty` range
    /// indexed beside a full-text `title`, built in two batches.
    fn fixture_superfile() -> (BuilderOptions, Bytes) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("_id", DataType::Decimal128(38, 0), false),
            Field::new("price", DataType::Float64, true),
            Field::new("qty", DataType::Int32, false),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let (price, qty) = fixture_values();
        let batch = |rows: Range<usize>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(
                        Decimal128Array::from_iter_values(rows.clone().map(|i| i as i128))
                            .with_precision_and_scale(38, 0)
                            .expect("decimal"),
                    ),
                    Arc::new(Float64Array::from(price[rows.clone()].to_vec())),
                    Arc::new(Int32Array::from(qty[rows.clone()].to_vec())),
                    Arc::new(LargeStringArray::from_iter_values(
                        rows.map(|i| format!("row {i}")),
                    )),
                ],
            )
            .expect("batch")
        };
        let opts = BuilderOptions::new(
            Arc::clone(&schema),
            "_id",
            vec![FtsConfig::new("title")],
            vec![],
        )
        .with_range_columns(vec!["price".into(), "qty".into()]);
        let mut builder = SuperfileBuilder::new(opts.clone()).expect("builder");
        builder
            .add_batch(&batch(0..FIRST_BATCH_ROWS), &[])
            .expect("first batch");
        builder
            .add_batch(&batch(FIRST_BATCH_ROWS..SUPERFILE_ROWS), &[])
            .expect("second batch");
        (opts, Bytes::from(builder.finish().expect("finish")))
    }

    /// Check `reader`'s range answers for the fixture, built with `opts`,
    /// against the oracle.
    async fn assert_ranges(reader: &SuperfileReader, opts: &BuilderOptions, what: &str) {
        let id = |column: &str| opts.field_id_of_column(column).expect("field id");
        let (price, qty) = fixture_values();
        let qty: Vec<Option<i32>> = qty.into_iter().map(Some).collect();
        for (lo, hi) in [(-50.0, 199.75), (0.0, 0.0), (10.25, 30.5), (150.0, 900.0)] {
            let (lo_key, hi_key) = (float_key(lo), float_key(hi));
            let want = oracle(&price, |v: f64| lo <= v && v <= hi);
            let rows = reader
                .range_rows(id("price"), &DataType::Float64, lo_key, hi_key, None)
                .await
                .expect("rows")
                .expect("price is indexed");
            assert_eq!(rows, want, "{what}: price rows in [{lo}, {hi}]");
            let count = reader
                .range_count(id("price"), &DataType::Float64, lo_key, hi_key)
                .await
                .expect("count")
                .expect("price is indexed");
            assert_eq!(count, want.len(), "{what}: price count in [{lo}, {hi}]");
        }
        let rows = reader
            .range_rows(
                id("qty"),
                &DataType::Int32,
                signed_key(-3),
                signed_key(3),
                None,
            )
            .await
            .expect("rows")
            .expect("qty is indexed");
        assert_eq!(
            rows,
            oracle(&qty, |v: i32| (-3..=3).contains(&v)),
            "{what}: qty"
        );
        // A column the region does not hold, or one keyed another way, is
        // left to a scan.
        for (column, data_type) in [("title", DataType::LargeUtf8), ("price", DataType::Int64)] {
            let none = reader
                .range_rows(id(column), &data_type, 0, u64::MAX, None)
                .await
                .expect("rows");
            assert!(none.is_none(), "{what}: {column} as {data_type:?}");
        }
    }

    #[tokio::test]
    async fn a_built_superfile_answers_ranges_eagerly_lazily_and_after_a_carry() {
        let (opts, bytes) = fixture_superfile();
        let eager = SuperfileReader::open(bytes.clone()).expect("eager open");
        assert_ranges(&eager, &opts, "eager").await;
        let source: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(bytes));
        let lazy = SuperfileReader::open_lazy(source).await.expect("lazy open");
        assert_ranges(&lazy, &opts, "lazy").await;

        // A rewrite that carries the body builds the region again from it.
        let mut carry = SuperfileBuilder::new(opts.clone()).expect("carry builder");
        carry.reanalyze_fts_from_reader(&eager).expect("reanalyze");
        carry.set_carried_doc_count(eager.n_docs());
        let mut out = Vec::new();
        carry
            .finish_carrying_body_to(&eager, &mut out)
            .expect("carry");
        let carried = SuperfileReader::open(Bytes::from(out)).expect("open carried");
        assert_ranges(&carried, &opts, "carried").await;
    }

    /// A layout repair carries the body without decoding it, then reads the
    /// range columns and `_id` back to rebuild the region. A scan may have
    /// loaded the source's page index for other columns only, and the footer
    /// fails a read of a column it holds no page index for, so the rebuild
    /// must ask for its own columns.
    #[tokio::test]
    async fn a_carry_reads_the_range_columns_a_scan_did_not_load() {
        let (opts, bytes) = fixture_superfile();
        let source = SuperfileReader::open(bytes).expect("open");
        source
            .parquet_metadata_with_page_index(&IndexColumns::named(&["title"]))
            .await
            .expect("a scan's load");
        let mut carry = SuperfileBuilder::new(opts.clone()).expect("carry builder");
        carry
            .carry_fts_from_reader_scoped(&source, None, CarryScope::AllColumns)
            .expect("carry the FTS blob");
        carry.set_carried_doc_count(source.n_docs());
        let mut out = Vec::new();
        carry
            .finish_carrying_body_to(&source, &mut out)
            .expect("carry");
        let carried = SuperfileReader::open(Bytes::from(out)).expect("open carried");
        assert_ranges(&carried, &opts, "carried after a scan's load").await;
    }

    /// The fixture's table after it renamed `qty` to `quantity` and added a
    /// range-indexed `added` (id `added_id`): what a rewrite of a file the
    /// fixture wrote is built with.
    fn renamed_table(opts: &BuilderOptions, added_id: FieldId) -> BuilderOptions {
        let mut fields: Vec<Field> = opts
            .schema
            .fields()
            .iter()
            .map(|f| match f.name().as_str() {
                "qty" => f.as_ref().clone().with_name("quantity"),
                _ => f.as_ref().clone(),
            })
            .collect();
        fields.push(with_field_id(
            &Field::new("added", DataType::Int64, true),
            added_id,
        ));
        BuilderOptions::new(
            Arc::new(Schema::new(fields)),
            "_id",
            vec![FtsConfig::new("title")],
            vec![],
        )
        .with_range_columns(vec!["price".into(), "quantity".into(), "added".into()])
    }

    #[tokio::test]
    async fn a_carried_body_finds_its_range_columns_by_id() {
        // The table renamed `qty` to `quantity` and added a range-indexed
        // `added` after the fixture was written. The carried body holds
        // `qty` under its old name and no `added` at all: the rewrite indexes
        // `qty` under its id and leaves `added` out, instead of failing.
        let (opts, bytes) = fixture_superfile();
        let source = SuperfileReader::open(bytes).expect("open");
        let id = |column: &str| opts.field_id_of_column(column).expect("field id");
        let added_id = FieldId(id("title").0 + 1);
        let table = renamed_table(&opts, added_id);

        let mut carry = SuperfileBuilder::new(table).expect("carry builder");
        carry.reanalyze_fts_from_reader(&source).expect("reanalyze");
        carry.set_carried_doc_count(source.n_docs());
        let mut out = Vec::new();
        carry
            .finish_carrying_body_to(&source, &mut out)
            .expect("a range column the body lacks is left out");
        let carried = SuperfileReader::open(Bytes::from(out)).expect("open carried");
        assert_ranges(&carried, &opts, "carried under the new schema").await;
        let added = carried
            .range_rows(added_id, &DataType::Int64, 0, u64::MAX, None)
            .await
            .expect("rows");
        assert!(added.is_none(), "a column the body lacks is not indexed");
    }

    /// The same rename, after a scan loaded the source's page index for
    /// another column, carried without decoding the body: the region rebuild
    /// reads `qty` through the footer, so it must ask for it by id, since its
    /// table name is now `quantity`.
    #[tokio::test]
    async fn a_carry_reads_a_renamed_range_column_by_id_after_a_scan_load() {
        let (opts, bytes) = fixture_superfile();
        let source = SuperfileReader::open(bytes).expect("open");
        source
            .parquet_metadata_with_page_index(&IndexColumns::named(&["title"]))
            .await
            .expect("a scan's load");
        let id = |column: &str| opts.field_id_of_column(column).expect("field id");
        let table = renamed_table(&opts, FieldId(id("title").0 + 1));
        let mut carry = SuperfileBuilder::new(table).expect("carry builder");
        carry
            .carry_fts_from_reader_scoped(&source, None, CarryScope::AllColumns)
            .expect("carry the FTS blob");
        carry.set_carried_doc_count(source.n_docs());
        let mut out = Vec::new();
        carry
            .finish_carrying_body_to(&source, &mut out)
            .expect("carry");
        let carried = SuperfileReader::open(Bytes::from(out)).expect("open carried");
        assert_ranges(
            &carried,
            &opts,
            "carried under the new schema after a scan's load",
        )
        .await;
    }

    #[tokio::test]
    async fn a_superfile_without_range_columns_has_no_region() {
        let (opts, _) = fixture_superfile();
        let schema = Arc::clone(&opts.schema);
        let plain = BuilderOptions::new(schema, "_id", vec![], vec![]);
        let mut builder = SuperfileBuilder::new(plain).expect("builder");
        let batch = RecordBatch::try_new(
            Arc::clone(&opts.schema),
            vec![
                Arc::new(
                    Decimal128Array::from_iter_values([1i128])
                        .with_precision_and_scale(38, 0)
                        .expect("decimal"),
                ),
                Arc::new(Float64Array::from(vec![Some(1.0)])),
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(LargeStringArray::from(vec!["a"])),
            ],
        )
        .expect("batch");
        builder.add_batch(&batch, &[]).expect("batch");
        let reader =
            SuperfileReader::open(Bytes::from(builder.finish().expect("finish"))).expect("open");
        let price = opts.field_id_of_column("price").expect("field id");
        let rows = reader
            .range_rows(price, &DataType::Float64, 0, u64::MAX, None)
            .await
            .expect("rows");
        assert!(rows.is_none());
    }

    /// Rows of the tie fixture: one `level` value is held by nearly all of
    /// them, across several blocks.
    const TIE_ROWS: usize = 3 * BLOCK_VALUES + 500;
    /// Every this-many rows (offset [`HIGH_AT`]) `level` is 2.
    const HIGH_EVERY: usize = 500;
    /// The offset of the rows where `level` is 2.
    const HIGH_AT: usize = 7;
    /// Every this-many rows (offset [`LOW_AT`]) `level` is 0; 1 otherwise.
    const LOW_EVERY: usize = 700;
    /// The offset of the rows where `level` is 0.
    const LOW_AT: usize = 3;

    fn level(row: usize) -> i64 {
        if row % HIGH_EVERY == HIGH_AT {
            2
        } else if row % LOW_EVERY == LOW_AT {
            0
        } else {
            1
        }
    }

    /// A superfile of [`TIE_ROWS`] rows with `level` range indexed, whose
    /// row `i` has `_id` `id_of(i)`.
    fn tie_superfile(id_of: impl Fn(usize) -> i128) -> (BuilderOptions, SuperfileReader) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("_id", DataType::Decimal128(38, 0), false),
            Field::new("level", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(
                    Decimal128Array::from_iter_values((0..TIE_ROWS).map(id_of))
                        .with_precision_and_scale(38, 0)
                        .expect("decimal"),
                ),
                Arc::new(Int64Array::from_iter_values((0..TIE_ROWS).map(level))),
            ],
        )
        .expect("batch");
        let opts = BuilderOptions::new(schema, "_id", vec![], vec![])
            .with_range_columns(vec!["level".into()]);
        let mut builder = SuperfileBuilder::new(opts.clone()).expect("builder");
        builder.add_batch(&batch, &[]).expect("batch");
        let bytes = Bytes::from(builder.finish().expect("finish"));
        (opts, SuperfileReader::open(bytes).expect("open"))
    }

    /// A step coprime to [`TIE_ROWS`]: row `i` of the scattered fixture has
    /// `_id` `i * ID_STEP mod TIE_ROWS`, every id once, in no row order.
    const ID_STEP: usize = 7919;

    /// The rows an `ORDER BY level, _id LIMIT k` keeps (`level` descending
    /// with `descending`, the id ascending either way), past `skip`, when
    /// row `i` has id `id_of(i)`.
    fn first_by_level_then_id(
        k: usize,
        descending: bool,
        skip: &RoaringBitmap,
        id_of: &dyn Fn(usize) -> i128,
    ) -> RoaringBitmap {
        let mut rows: Vec<(i64, i128, u32)> = (0..TIE_ROWS)
            .filter(|row| !skip.contains(*row as u32))
            .map(|row| (level(row), id_of(row), row as u32))
            .collect();
        rows.sort_by(|a, b| {
            let by_level = match descending {
                false => a.0.cmp(&b.0),
                true => b.0.cmp(&a.0),
            };
            by_level.then(a.1.cmp(&b.1))
        });
        rows.into_iter().take(k).map(|(_, _, row)| row).collect()
    }

    /// The first `k` rows by a key, either way, keep the lowest `_id`s of
    /// the `k`-th key, even when that key fills several blocks and a
    /// descending walk meets it from its far end, and whatever order the
    /// superfile's rows hold their ids in: ascending, descending, scattered.
    #[tokio::test]
    async fn range_ends_keep_the_lowest_ids_of_the_last_key_either_way() {
        let id_orders: [&dyn Fn(usize) -> i128; 3] = [
            &|row| row as i128 + 1,
            &|row| (TIE_ROWS - row) as i128,
            &|row| (row * ID_STEP % TIE_ROWS) as i128,
        ];
        let all = (signed_key(0), signed_key(2));
        let deleted: RoaringBitmap = [0, 1, 7, 4, 3, 2001].into_iter().collect();
        for (order, id_of) in id_orders.into_iter().enumerate() {
            let (opts, reader) = tie_superfile(id_of);
            let field = opts.field_id_of_column("level").expect("field id");
            for skip in [RoaringBitmap::new(), deleted.clone()] {
                for descending in [false, true] {
                    // `usize::MAX`: a `k` past every row returns them all,
                    // sizing nothing by `k`.
                    for k in [1, 5, 10, 40, TIE_ROWS, usize::MAX] {
                        let ends = reader
                            .range_ends(field, &DataType::Int64, all, k, descending, &skip)
                            .await
                            .expect("ends")
                            .expect("level is indexed");
                        assert!(ends.ties_in_id_order);
                        let rows: RoaringBitmap = ends.pairs.iter().map(|(_, row)| *row).collect();
                        assert_eq!(ends.pairs.len(), rows.len() as usize, "no row twice");
                        assert_eq!(
                            rows,
                            first_by_level_then_id(k, descending, &skip, id_of),
                            "id order {order}, k {k}, descending {descending}, {} deleted",
                            skip.len()
                        );
                    }
                }
            }
        }
    }

    /// Entries spilled in several runs come back merged by key and then by
    /// their order among the key's rows, the full 128 bits of it.
    #[test]
    fn spilled_runs_merge_by_key_then_tie() {
        let scratch = tempfile::tempdir().expect("scratch");
        let mut sorter = RunSorter::default();
        let entries: [(u64, u128, u32); 6] = [
            (7, 5 << 70, 0),
            (3, 9, 1),
            (7, 2, 2),
            (3, 1 << 100, 3),
            (7, (5 << 70) - 1, 4),
            (3, 10, 5),
        ];
        for chunk in entries.chunks(2) {
            for &(key, tie, row) in chunk {
                sorter.push(key, tie, row, scratch.path()).expect("push");
            }
            sorter.spill(scratch.path()).expect("spill");
        }
        let mut out = Vec::new();
        let n = sorter
            .drain(scratch.path(), |key, row| {
                out.push((key, row));
                Ok(())
            })
            .expect("drain");
        assert_eq!(n, entries.len() as u64);
        assert_eq!(out, vec![(3, 1), (3, 5), (3, 3), (7, 2), (7, 4), (7, 0)]);
    }

    /// The build marks the region when it is given an id for every row,
    /// whatever order the ids are in, and not when a batch lacks the id
    /// column, an id is missing, or no id column is named.
    #[test]
    fn the_build_marks_ties_in_id_order_when_every_row_has_an_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("_id", DataType::Decimal128(38, 0), false),
            Field::new("a", DataType::Int64, false),
        ]));
        let batch = |ids: &[i128]| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(
                        Decimal128Array::from_iter_values(ids.iter().copied())
                            .with_precision_and_scale(38, 0)
                            .expect("decimal"),
                    ),
                    Arc::new(Int64Array::from_iter_values(ids.iter().map(|_| 5))),
                ],
            )
        };
        let marked = |batches: Vec<&[i128]>, id_column: Option<&str>| {
            let scratch = tempfile::tempdir().expect("scratch");
            let region = whole(
                build_region(
                    batches.into_iter().map(batch),
                    &[spec("a", 1, KeyKind::Signed)],
                    id_column,
                    scratch.path(),
                )
                .expect("build"),
            );
            let directory = RangeDirectory::parse(&region, region.len() as u64).expect("parse");
            directory.column(1).expect("indexed").ties_in_id_order()
        };
        assert!(marked(vec![&[1, 2, 3], &[9, 10]], Some("_id")));
        assert!(marked(vec![&[9, 5, 3], &[1, 10]], Some("_id")));
        assert!(!marked(vec![&[1, 2, 3]], None));
        assert!(!marked(vec![&[1, 2, 3]], Some("missing")));
        // A null id leaves its row ordered by row, so the region is not marked.
        let nullable = Arc::new(Schema::new(vec![
            Field::new("_id", DataType::Decimal128(38, 0), true),
            Field::new("a", DataType::Int64, false),
        ]));
        let with_null = RecordBatch::try_new(
            nullable,
            vec![
                Arc::new(
                    Decimal128Array::from(vec![Some(1), None])
                        .with_precision_and_scale(38, 0)
                        .expect("decimal"),
                ),
                Arc::new(Int64Array::from(vec![5, 5])),
            ],
        );
        let scratch = tempfile::tempdir().expect("scratch");
        let region = whole(
            build_region(
                [with_null],
                &[spec("a", 1, KeyKind::Signed)],
                Some("_id"),
                scratch.path(),
            )
            .expect("build"),
        );
        let directory = RangeDirectory::parse(&region, region.len() as u64).expect("parse");
        assert!(!directory.column(1).expect("indexed").ties_in_id_order());
    }

    #[test]
    fn a_column_of_nulls_indexes_no_value() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![None, None]))]);
        let scratch = tempfile::tempdir().expect("scratch");
        let region = whole(
            build_region(
                [batch],
                &[spec("a", 1, KeyKind::Signed)],
                None,
                scratch.path(),
            )
            .expect("build"),
        );
        let (rows, count) = query(&region, 1, 0, u64::MAX);
        assert!(rows.is_empty());
        assert_eq!(count, 0);
    }
}
