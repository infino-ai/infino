// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Streaming spill primitives for the bounded-memory build path.
//!
//! Cooperating abstractions:
//!
//! - [`SpillWriter`] — append-only writer that buffers raw f32
//!   vector bytes into a temp file on disk. Used during
//!   `VectorBuilder::add()` once the in-RAM `pre_spill_buffer`
//!   crosses the configured `spill_threshold_bytes`. Wraps a
//!   `BufWriter<File>` so callers don't pay one syscall per
//!   `write_vec`.
//! - [`ChunkedVectorSource`] — read-only iterator over the full
//!   corpus as zero-copy `&[f32]` chunks of fixed row count.
//!   Two implementations: [`InMemoryVectorSource`] (wraps an
//!   `Arc<Vec<f32>>`, no spill needed) and [`MmapVectorSource`]
//!   (wraps a memory-mapped spill file).
//! - [`MaterializedRowSpillWriter`] / [`SpilledCellRows`] — per-cell
//!   Sq8 row scratch accumulated across drain batches, plus bounded
//!   readers used by the streamed materialized-IVF builder.
//!
//! Both `ChunkedVectorSource` implementations own their backing
//! storage so the trait isn't tied to an external lifetime; the
//! chunk slice returned per iteration is valid for the duration
//! of the `&mut self` borrow, which is the scope the pass-2
//! per-chunk loop runs inside.

use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Error, ErrorKind, Read, Write},
    mem::size_of,
    path::{Path, PathBuf},
    sync::Arc,
};

use bytemuck::{cast_slice, try_cast_slice};
use memmap2::Mmap;

use crate::superfile::{
    BuildError,
    vector::{
        cell_posting::{EncodedCellRow, MaterializedIvfRow},
        rerank_codec::RerankCodec,
    },
};

/// Append-only spill writer for f32 vectors. Backed by a
/// `BufWriter<File>` so the hot path (per-vector `write_vec` in
/// `VectorBuilder::add()`) doesn't pay one syscall per call —
/// kernel-bound writes batch up to the buffer size (1 MiB by
/// default) before flushing.
///
/// The on-disk format is **raw little-endian f32**, no header,
/// no checksum, no record framing: a build's pass 1 records the
/// `(dim, n_docs)` separately on `ColumnState` and the spill
/// file is exactly `n_docs * dim * 4` bytes at `finish()` time.
/// This matches what [`MmapVectorSource::open`] expects on the
/// read side.
pub(crate) struct SpillWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    bytes_written: u64,
}

impl SpillWriter {
    /// BufWriter capacity. One 1 MiB write per syscall on
    /// typical Linux kernels; balances per-call amortization
    /// against the temporary memory footprint of the spill
    /// path itself.
    const BUF_CAPACITY: usize = 1 << 20;

    /// Create a fresh spill file at `path`. Truncates if it
    /// exists. Errors if the file can't be created (e.g.
    /// scratch dir is read-only or out of space).
    pub fn create(path: PathBuf) -> Result<Self, BuildError> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        let writer = BufWriter::with_capacity(Self::BUF_CAPACITY, file);
        Ok(Self {
            path,
            writer,
            bytes_written: 0,
        })
    }

    /// Append a raw byte slice. Length must be a multiple of 4
    /// (i.e. well-formed f32 little-endian payload). Used by
    /// the pre-spill drain in `VectorBuilder::add` to flush the
    /// `pre_spill_buffer` in one batched call once the
    /// threshold is crossed.
    pub fn write_all(&mut self, bytes: &[u8]) -> Result<(), BuildError> {
        debug_assert!(
            bytes.len().is_multiple_of(4),
            "spill write_all: byte length {} not a multiple of 4",
            bytes.len()
        );
        self.writer.write_all(bytes)?;
        self.bytes_written += bytes.len() as u64;
        Ok(())
    }

    /// Append one vector. Equivalent to
    /// `write_all(cast_slice(vec))` but spelled out
    /// so the hot path doesn't have to re-derive the cast on
    /// every call.
    pub fn write_vec(&mut self, vec: &[f32]) -> Result<(), BuildError> {
        let bytes: &[u8] = cast_slice(vec);
        self.write_all(bytes)
    }

    /// Total bytes appended through this writer. Counts bytes
    /// at the caller boundary, before the kernel flush. Used
    /// by tests to confirm the spill file grew as expected.
    #[cfg(test)]
    pub(crate) fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Flush the buffer to the kernel, fsync the file, and
    /// return the path. The file is closed but not deleted —
    /// the caller is responsible for handing the path to
    /// `MmapVectorSource::open` (typical) or for cleanup
    /// (test path).
    pub fn finish(mut self) -> Result<PathBuf, BuildError> {
        self.writer.flush()?;
        let file = self
            .writer
            .into_inner()
            .map_err(|e| BuildError::Io(e.into_error()))?;
        file.sync_all()?;
        Ok(self.path)
    }
}

/// Read-only chunked iterator over the full input corpus.
///
/// Each `next_chunk` call yields up to [`Self::chunk_rows`] rows
/// as a contiguous `&[f32]` slice of length
/// `chunk_size_actual * dim`. The slice is valid for the
/// duration of the returned reference (`&mut self` borrow); the
/// underlying owner (`Arc<Vec<f32>>` for in-memory, `Mmap` for
/// spilled) outlives every yielded slice.
///
/// Implementations:
///
/// - [`InMemoryVectorSource`] for builds whose
///   `pre_spill_buffer` never crossed `spill_threshold_bytes`.
/// - [`MmapVectorSource`] for builds that did, opening the
///   spill file `mmap`-style.
///
/// The pass-2 builder loop iterates `while let Some(chunk) =
/// src.next_chunk() { rotate / assign / encode / route to
/// bucket files }` exactly once. `reset` exists for tests +
/// debug paths that want to walk the source twice.
pub trait ChunkedVectorSource {
    /// Total number of rows (vectors) in the source.
    fn n_rows(&self) -> usize;

    /// Dimension of each row. The same value across all chunks.
    #[cfg(test)]
    fn dim(&self) -> usize;

    /// Maximum number of rows the next `next_chunk` returns.
    /// The trailing chunk may return fewer if `n_rows` isn't
    /// a multiple of `chunk_rows`.
    fn chunk_rows(&self) -> usize;

    /// Yield the next chunk of up to `chunk_rows` rows, or
    /// `None` if the source is exhausted. The slice length is
    /// always a multiple of `dim`.
    fn next_chunk(&mut self) -> Option<&[f32]>;

    /// Reset the iterator to row 0. Used by tests; the
    /// pass-2 build loop walks the source exactly once and
    /// doesn't need this.
    #[cfg(test)]
    fn reset(&mut self);
}

/// In-RAM source: wraps an `Arc<Vec<f32>>` holding the full
/// (un-rotated) input corpus. Used when the build never
/// crossed the spill threshold; `VectorBuilder::ColumnState`
/// moves its `pre_spill_buffer` into an `Arc<Vec<f32>>` at the
/// pass-1 → pass-2 boundary.
///
/// Zero-copy slicing on each `next_chunk` call — the chunk
/// `&[f32]` points directly into the `Arc<Vec<f32>>` buffer.
pub struct InMemoryVectorSource {
    buf: Arc<Vec<f32>>,
    dim: usize,
    chunk_rows: usize,
    cursor: usize, // next row to emit
}

impl InMemoryVectorSource {
    /// Construct from an owned buffer. `buf.len()` must be a
    /// multiple of `dim`; the row count is derived from that.
    /// `chunk_rows` must be ≥ 1; values larger than `n_rows`
    /// are silently capped on the trailing chunk by the trait
    /// contract.
    pub fn new(buf: Arc<Vec<f32>>, dim: usize, chunk_rows: usize) -> Self {
        debug_assert!(dim > 0, "InMemoryVectorSource: dim must be > 0");
        debug_assert!(
            chunk_rows > 0,
            "InMemoryVectorSource: chunk_rows must be > 0"
        );
        debug_assert!(
            buf.len().is_multiple_of(dim),
            "InMemoryVectorSource: buf.len() {} not a multiple of dim {}",
            buf.len(),
            dim
        );
        Self {
            buf,
            dim,
            chunk_rows,
            cursor: 0,
        }
    }
}

impl ChunkedVectorSource for InMemoryVectorSource {
    fn n_rows(&self) -> usize {
        self.buf.len() / self.dim
    }

    #[cfg(test)]
    fn dim(&self) -> usize {
        self.dim
    }

    fn chunk_rows(&self) -> usize {
        self.chunk_rows
    }

    fn next_chunk(&mut self) -> Option<&[f32]> {
        let n_rows = self.n_rows();
        if self.cursor >= n_rows {
            return None;
        }
        let take = (n_rows - self.cursor).min(self.chunk_rows);
        let start = self.cursor * self.dim;
        let end = start + take * self.dim;
        self.cursor += take;
        Some(&self.buf[start..end])
    }

    #[cfg(test)]
    fn reset(&mut self) {
        self.cursor = 0;
    }
}

/// Mmap-backed source: opens a spill file written by
/// [`SpillWriter`] and exposes it as zero-copy `&[f32]` chunks.
///
/// The map stays resident for the lifetime of the source; the
/// page cache handles paging. Linear-scan access (which is what
/// pass 2 does) is the kernel's happy case — typical throughput
/// matches raw disk read bandwidth on NVMe.
pub struct MmapVectorSource {
    map: Mmap,
    dim: usize,
    chunk_rows: usize,
    cursor: usize, // next row to emit
}

impl MmapVectorSource {
    /// Open `path` as a memory-mapped spill source. The file
    /// must contain exactly `n_rows * dim * 4` bytes of raw
    /// little-endian f32 (the on-disk format
    /// [`SpillWriter`] produces); `open` validates that
    /// `file_len % (dim * 4) == 0` and derives `n_rows`.
    ///
    /// # Safety
    ///
    /// `Mmap::map` is `unsafe` in `memmap2` because the
    /// process can no longer detect external truncation of
    /// the backing file. Callers must ensure the spill file
    /// is not modified by another process for the lifetime
    /// of the returned source. The build path satisfies this
    /// by holding the `tempfile::TempDir` for the duration —
    /// only the build process owns the file.
    pub fn open(path: &Path, dim: usize, chunk_rows: usize) -> Result<Self, BuildError> {
        debug_assert!(dim > 0, "MmapVectorSource: dim must be > 0");
        debug_assert!(chunk_rows > 0, "MmapVectorSource: chunk_rows must be > 0");
        let file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;
        let row_bytes = dim
            .checked_mul(4)
            .expect("dim * 4 overflows usize — dim > 2^29 is nonsense");
        if !file_len.is_multiple_of(row_bytes) {
            return Err(BuildError::Io(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "spill file length {file_len} is not a multiple of \
                     row size {row_bytes} (dim={dim})"
                ),
            )));
        }
        // SAFETY: see method-level safety comment.
        let map = unsafe { Mmap::map(&file)? };
        Ok(Self {
            map,
            dim,
            chunk_rows,
            cursor: 0,
        })
    }
}

impl ChunkedVectorSource for MmapVectorSource {
    fn n_rows(&self) -> usize {
        self.map.len() / (self.dim * 4)
    }

    #[cfg(test)]
    fn dim(&self) -> usize {
        self.dim
    }

    fn chunk_rows(&self) -> usize {
        self.chunk_rows
    }

    fn next_chunk(&mut self) -> Option<&[f32]> {
        let n_rows = self.n_rows();
        if self.cursor >= n_rows {
            return None;
        }
        let take = (n_rows - self.cursor).min(self.chunk_rows);
        let row_bytes = self.dim * 4;
        let start_b = self.cursor * row_bytes;
        let end_b = start_b + take * row_bytes;
        self.cursor += take;
        // Mmap is page-aligned (≥ 4-aligned) and the slice
        // length is a multiple of 4, so the cast is sound.
        // `try_cast_slice` returns `Err` on any
        // misalignment / length mismatch; we panic via expect
        // because both invariants are upheld by construction
        // (validated in `open`).
        let bytes: &[u8] = &self.map[start_b..end_b];
        let floats: &[f32] =
            try_cast_slice(bytes).expect("mmap slice is page-aligned and length is row-aligned");
        Some(floats)
    }

    #[cfg(test)]
    fn reset(&mut self) {
        self.cursor = 0;
    }
}

/// Bytes of the fixed per-row prefix in a [`MaterializedRowSpillWriter`]
/// record: `stable_id` (i128) + `cluster` (u32) + quantizer-table index (u32)
/// + `norm_sq` presence flag (u8) + `norm_sq` payload (f32, zero when absent).
const ROW_SPILL_PREFIX_BYTES: usize =
    size_of::<i128>() + size_of::<u32>() + size_of::<u32>() + size_of::<u8>() + size_of::<f32>();

/// `norm_sq` presence flag values in the spilled row prefix.
const NORM_ABSENT: u8 = 0;
const NORM_PRESENT: u8 = 1;

/// One finished per-cell spill: the row-record file, its quantizer-table
/// sidecar, and the counts needed to read them back.
#[derive(Debug)]
pub(crate) struct SpilledCellRows {
    rows_path: PathBuf,
    quants_path: PathBuf,
    n_rows: u32,
    n_quants: u32,
    dim: usize,
    rabitq_len: usize,
    rerank_codec: RerankCodec,
}

impl SpilledCellRows {
    pub(crate) fn n_rows(&self) -> usize {
        self.n_rows as usize
    }

    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) fn rerank_codec(&self) -> RerankCodec {
        self.rerank_codec
    }

    pub(crate) fn reader(&self) -> Result<MaterializedRowSpillReader, BuildError> {
        MaterializedRowSpillReader::open(self)
    }

    /// Delete both backing files. Called after the cell superfile is built
    /// and uploaded so drain scratch shrinks as cells complete; the owning
    /// tempdir still sweeps anything left behind on early exit.
    pub(crate) fn remove_files(&self) {
        let _ = fs::remove_file(&self.rows_path);
        let _ = fs::remove_file(&self.quants_path);
    }
}

/// Byte length of one spilled row record for a `(dim, rabitq_len)` shape:
/// the fixed prefix plus the RaBitQ code and the Sq8+epsilon `codes`/`residuals`
/// legs (each `dim` bytes).
fn record_bytes(dim: usize, rabitq_len: usize) -> usize {
    ROW_SPILL_PREFIX_BYTES + rabitq_len + 2 * dim
}

/// Append-only spill for [`MaterializedIvfRow`]s of ONE cell, accumulated
/// across the drain's memory-bounded batches so the drain can build each
/// cell's IVF once per run (then pack many cell IVFs into shard objects)
/// regardless of batch size.
///
/// Rows are written as their already-encoded Sq8+epsilon bytes — no decode, no
/// re-quantization. The per-cluster dequant params (`scale`/`offset`, shared
/// `Arc<[f32]>`s) are NOT duplicated per row: each distinct quantizer is
/// appended once to a sidecar table and rows reference it by index.
pub(crate) struct MaterializedRowSpillWriter {
    rows: BufWriter<File>,
    quants: BufWriter<File>,
    rows_path: PathBuf,
    quants_path: PathBuf,
    dim: usize,
    rabitq_len: usize,
    n_rows: u32,
    n_quants: u32,
    rerank_codec: Option<RerankCodec>,
    quant_idx_by_ptr: HashMap<usize, u32>,
}

/// Durable counters needed to reopen one per-cell row spill at a checkpointed
/// batch boundary. File lengths are derived from these values and truncated on
/// resume, discarding any partial next-batch tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MaterializedRowSpillState {
    pub(crate) n_rows: u32,
    pub(crate) n_quants: u32,
    pub(crate) dim: usize,
    pub(crate) rabitq_len: usize,
    pub(crate) rerank_codec: RerankCodec,
}

impl MaterializedRowSpillWriter {
    /// Create the row + quantizer spill files for `cell` under `dir`.
    pub(crate) fn create(
        dir: &Path,
        cell: u32,
        dim: usize,
        rabitq_len: usize,
    ) -> Result<Self, BuildError> {
        let rows_path = dir.join(format!("cell-{cell}.rows"));
        let quants_path = dir.join(format!("cell-{cell}.quants"));
        let rows_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&rows_path)?;
        let quants_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&quants_path)?;
        Ok(Self {
            rows: BufWriter::with_capacity(SpillWriter::BUF_CAPACITY, rows_file),
            quants: BufWriter::with_capacity(SpillWriter::BUF_CAPACITY, quants_file),
            rows_path,
            quants_path,
            dim,
            rabitq_len,
            n_rows: 0,
            n_quants: 0,
            rerank_codec: None,
            quant_idx_by_ptr: HashMap::new(),
        })
    }

    /// Reset the pointer-identity dedup at a batch boundary.
    pub(crate) fn begin_batch(&mut self) {
        self.quant_idx_by_ptr.clear();
    }

    /// Reopen one checkpointed cell spill for append. Both files are
    /// truncated to the exact durable lengths implied by `state` before the
    /// writer seeks to the end, so a crash during the next batch is replayable.
    pub(crate) fn resume(
        dir: &Path,
        cell: u32,
        state: MaterializedRowSpillState,
    ) -> Result<Self, BuildError> {
        let rows_path = dir.join(format!("cell-{cell}.rows"));
        let quants_path = dir.join(format!("cell-{cell}.quants"));
        let rows_len = u64::from(state.n_rows) * record_bytes(state.dim, state.rabitq_len) as u64;
        let quants_len = u64::from(state.n_quants) * (2 * state.dim * size_of::<f32>()) as u64;

        let rows_file = OpenOptions::new().append(true).open(&rows_path)?;
        rows_file.set_len(rows_len)?;
        let quants_file = OpenOptions::new().append(true).open(&quants_path)?;
        quants_file.set_len(quants_len)?;

        Ok(Self {
            rows: BufWriter::with_capacity(SpillWriter::BUF_CAPACITY, rows_file),
            quants: BufWriter::with_capacity(SpillWriter::BUF_CAPACITY, quants_file),
            rows_path,
            quants_path,
            dim: state.dim,
            rabitq_len: state.rabitq_len,
            n_rows: state.n_rows,
            n_quants: state.n_quants,
            rerank_codec: Some(state.rerank_codec),
            quant_idx_by_ptr: HashMap::new(),
        })
    }

    /// Append one row's encoded bytes (and its quantizer when first seen in
    /// this batch).
    pub(crate) fn append(&mut self, row: &MaterializedIvfRow) -> Result<(), BuildError> {
        let enc = &row.encoded;
        if let Some(codec) = self.rerank_codec {
            if codec != enc.rerank_codec {
                return Err(BuildError::VectorSchemaMismatch(
                    "drain spill cannot mix rerank codecs".into(),
                ));
            }
        } else {
            self.rerank_codec = Some(enc.rerank_codec);
        }
        if enc.codes.len() != self.dim
            || enc.residuals.len() != self.dim
            || row.rabitq_code.len() != self.rabitq_len
            || enc.scale.len() != self.dim
            || enc.offset.len() != self.dim
        {
            return Err(BuildError::Io(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "drain spill: row shape mismatch (codes {}, residuals {}, rabitq {}, \
                     scale {}, offset {}) vs expected dim {} / rabitq {}",
                    enc.codes.len(),
                    enc.residuals.len(),
                    row.rabitq_code.len(),
                    enc.scale.len(),
                    enc.offset.len(),
                    self.dim,
                    self.rabitq_len,
                ),
            )));
        }
        let ptr = Arc::as_ptr(&enc.scale) as *const () as usize;
        let quant_idx = match self.quant_idx_by_ptr.get(&ptr) {
            Some(&idx) => idx,
            None => {
                let idx = self.n_quants;
                self.quants.write_all(cast_slice(enc.scale.as_ref()))?;
                self.quants.write_all(cast_slice(enc.offset.as_ref()))?;
                self.n_quants += 1;
                self.quant_idx_by_ptr.insert(ptr, idx);
                idx
            }
        };
        self.rows.write_all(&row.stable_id.to_le_bytes())?;
        self.rows.write_all(&row.cluster.to_le_bytes())?;
        self.rows.write_all(&quant_idx.to_le_bytes())?;
        match enc.norm_sq {
            Some(n) => {
                self.rows.write_all(&[NORM_PRESENT])?;
                self.rows.write_all(&n.to_le_bytes())?;
            }
            None => {
                self.rows.write_all(&[NORM_ABSENT])?;
                self.rows.write_all(&0f32.to_le_bytes())?;
            }
        }
        self.rows.write_all(&row.rabitq_code)?;
        self.rows.write_all(&enc.codes)?;
        self.rows.write_all(&enc.residuals)?;
        self.n_rows += 1;
        Ok(())
    }

    /// Flush one durable batch boundary and return the counters needed to
    /// reopen it after a process restart.
    pub(crate) fn checkpoint(&mut self) -> Result<MaterializedRowSpillState, BuildError> {
        self.rows.flush()?;
        self.quants.flush()?;
        Ok(MaterializedRowSpillState {
            n_rows: self.n_rows,
            n_quants: self.n_quants,
            dim: self.dim,
            rabitq_len: self.rabitq_len,
            rerank_codec: self
                .rerank_codec
                .expect("checkpointed spill contains at least one row"),
        })
    }

    /// Flush both files and return the readable spill handle.
    pub(crate) fn finish(mut self) -> Result<SpilledCellRows, BuildError> {
        self.rows.flush()?;
        self.quants.flush()?;
        Ok(SpilledCellRows {
            rows_path: self.rows_path,
            quants_path: self.quants_path,
            n_rows: self.n_rows,
            n_quants: self.n_quants,
            dim: self.dim,
            rabitq_len: self.rabitq_len,
            rerank_codec: self
                .rerank_codec
                .expect("finished spill contains at least one row"),
        })
    }
}

/// Bounded reader over one finished per-cell encoded-row spill.
pub(crate) struct MaterializedRowSpillReader {
    rows: BufReader<File>,
    quants: Vec<(Arc<[f32]>, Arc<[f32]>)>,
    remaining: u32,
    dim: usize,
    rabitq_len: usize,
    rerank_codec: RerankCodec,
}

impl MaterializedRowSpillReader {
    fn open(spill: &SpilledCellRows) -> Result<Self, BuildError> {
        let dim = spill.dim;
        let quants = read_spilled_quantizers(spill)?;
        Ok(Self {
            rows: BufReader::new(File::open(&spill.rows_path)?),
            quants,
            remaining: spill.n_rows,
            dim,
            rabitq_len: spill.rabitq_len,
            rerank_codec: spill.rerank_codec,
        })
    }

    pub(crate) fn next_chunk(
        &mut self,
        max_rows: usize,
    ) -> Result<Vec<MaterializedIvfRow>, BuildError> {
        let take = (self.remaining as usize).min(max_rows.max(1));
        let mut rows = Vec::with_capacity(take);
        for _ in 0..take {
            rows.push(read_spilled_row(
                &mut self.rows,
                &self.quants,
                self.dim,
                self.rabitq_len,
                self.rerank_codec,
            )?);
        }
        self.remaining -= take as u32;
        Ok(rows)
    }
}

fn read_spilled_quantizers(
    spill: &SpilledCellRows,
) -> Result<Vec<(Arc<[f32]>, Arc<[f32]>)>, BuildError> {
    let dim = spill.dim;
    let mut quants: Vec<(Arc<[f32]>, Arc<[f32]>)> = Vec::with_capacity(spill.n_quants as usize);
    {
        let mut reader = BufReader::new(File::open(&spill.quants_path)?);
        let mut f32_buf = vec![0u8; dim * size_of::<f32>()];
        for _ in 0..spill.n_quants {
            reader.read_exact(&mut f32_buf)?;
            let scale: Arc<[f32]> = Arc::from(cast_f32_vec(&f32_buf));
            reader.read_exact(&mut f32_buf)?;
            let offset: Arc<[f32]> = Arc::from(cast_f32_vec(&f32_buf));
            quants.push((scale, offset));
        }
    }
    Ok(quants)
}

fn read_spilled_row(
    reader: &mut BufReader<File>,
    quants: &[(Arc<[f32]>, Arc<[f32]>)],
    dim: usize,
    rabitq_len: usize,
    rerank_codec: RerankCodec,
) -> Result<MaterializedIvfRow, BuildError> {
    let mut prefix = [0u8; ROW_SPILL_PREFIX_BYTES];
    reader.read_exact(&mut prefix)?;
    let stable_id = i128::from_le_bytes(prefix[0..16].try_into().expect("16-byte i128 slice"));
    let cluster = u32::from_le_bytes(prefix[16..20].try_into().expect("4-byte u32 slice"));
    let quant_idx =
        u32::from_le_bytes(prefix[20..24].try_into().expect("4-byte u32 slice")) as usize;
    let norm_flag = prefix[24];
    let norm = f32::from_le_bytes(prefix[25..29].try_into().expect("4-byte f32 slice"));
    let (scale, offset) = quants.get(quant_idx).cloned().ok_or_else(|| {
        BuildError::Io(Error::new(
            ErrorKind::InvalidData,
            format!(
                "drain spill: quantizer index {quant_idx} out of range ({} entries)",
                quants.len()
            ),
        ))
    })?;
    let mut rabitq_code = vec![0u8; rabitq_len];
    reader.read_exact(&mut rabitq_code)?;
    let mut codes = vec![0u8; dim];
    reader.read_exact(&mut codes)?;
    let mut residuals = vec![0u8; dim];
    reader.read_exact(&mut residuals)?;
    let norm_sq = (norm_flag == NORM_PRESENT).then_some(norm);
    Ok(MaterializedIvfRow {
        local_doc_id: 0,
        stable_id,
        cluster,
        rabitq_code,
        encoded: EncodedCellRow {
            stable_id,
            rerank_codec,
            scale,
            offset,
            codes,
            residuals,
            norm_sq,
        },
    })
}

/// Read one cell's spilled rows back into [`MaterializedIvfRow`]s.
///
/// Retained for small in-memory callers and equivalence tests. Large drain
/// packs use [`MaterializedRowSpillReader`] directly.
#[cfg(test)]
pub(crate) fn read_spilled_cell_rows(
    spill: &SpilledCellRows,
) -> Result<Vec<MaterializedIvfRow>, BuildError> {
    let mut reader = spill.reader()?;
    let mut rows = Vec::with_capacity(spill.n_rows());
    while rows.len() < spill.n_rows() {
        rows.extend(reader.next_chunk(spill.n_rows() - rows.len())?);
    }
    Ok(rows)
}

/// Decode a raw little-endian f32 byte buffer into an owned vec.
fn cast_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|c| f32::from_le_bytes(c.try_into().expect("4-byte f32 chunk")))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{
        fs::write,
        io::{ErrorKind, Read},
        iter::from_fn,
    };

    use tempfile::tempdir;

    use super::*;
    use crate::superfile::vector::rerank_codec::{SQ8_FIXED_OFFSET, SQ8_FIXED_SCALE};

    /// Build a deterministic f32 corpus of `n_rows × dim`.
    /// Row `r` column `c` = `r * 1000.0 + c as f32` so any
    /// misordering, chunking-boundary, or LE-encoding bug
    /// surfaces as a recognizable mismatch.
    fn synth(n_rows: usize, dim: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(n_rows * dim);
        for r in 0..n_rows {
            for c in 0..dim {
                v.push(r as f32 * 1000.0 + c as f32);
            }
        }
        v
    }

    #[test]
    fn spill_write_then_mmap_read_round_trip() {
        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("spill.bin");
        let n_rows = 17;
        let dim = 8;
        let corpus = synth(n_rows, dim);

        // Write the whole thing in one batch via write_all.
        {
            let mut w = SpillWriter::create(path.clone()).expect("create");
            let bytes: &[u8] = cast_slice(&corpus);
            w.write_all(bytes).expect("write_all");
            assert_eq!(w.bytes_written(), bytes.len() as u64);
            let finished_path = w.finish().expect("finish");
            assert_eq!(finished_path, path);
        }

        // Read back as raw bytes; verify byte-identical.
        {
            let mut f = File::open(&path).expect("open spill");
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).expect("read");
            let expected: &[u8] = cast_slice(&corpus);
            assert_eq!(buf, expected, "raw byte round-trip mismatch");
        }

        // Read back via MmapVectorSource; verify f32-identical.
        let mut src = MmapVectorSource::open(&path, dim, /*chunk_rows=*/ 5).expect("mmap open");
        assert_eq!(src.n_rows(), n_rows);
        assert_eq!(src.dim(), dim);
        assert_eq!(src.chunk_rows(), 5);

        let mut emitted = Vec::with_capacity(n_rows * dim);
        while let Some(chunk) = src.next_chunk() {
            emitted.extend_from_slice(chunk);
        }
        assert_eq!(emitted, corpus, "f32 round-trip via mmap mismatch");
    }

    #[test]
    fn spill_write_vec_per_row_matches_write_all() {
        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("spill_per_row.bin");
        let n_rows = 13;
        let dim = 4;
        let corpus = synth(n_rows, dim);

        let mut w = SpillWriter::create(path.clone()).expect("create");
        for r in 0..n_rows {
            let row = &corpus[r * dim..(r + 1) * dim];
            w.write_vec(row).expect("write_vec");
        }
        w.finish().expect("finish");

        let mut src = MmapVectorSource::open(&path, dim, dim).expect("mmap open");
        let mut emitted = Vec::with_capacity(n_rows * dim);
        while let Some(chunk) = src.next_chunk() {
            emitted.extend_from_slice(chunk);
        }
        assert_eq!(emitted, corpus, "per-row write_vec round-trip mismatch");
    }

    /// Row-spill round trip: rows written across two "batches" (dedup reset
    /// between them) read back bit-identical, with quantizers shared per
    /// source cluster and the batch-boundary reset forcing a re-append.
    #[test]
    fn materialized_row_spill_round_trips() {
        const DIM: usize = 8;
        const RABITQ_LEN: usize = 2;
        let tmp = tempdir().expect("tempdir");

        let quant_a: (Arc<[f32]>, Arc<[f32]>) =
            (Arc::from(vec![1.0f32; DIM]), Arc::from(vec![0.5f32; DIM]));
        let quant_b: (Arc<[f32]>, Arc<[f32]>) =
            (Arc::from(vec![2.0f32; DIM]), Arc::from(vec![0.25f32; DIM]));
        let row = |id: i128, cluster: u32, q: &(Arc<[f32]>, Arc<[f32]>), norm: Option<f32>| {
            MaterializedIvfRow {
                local_doc_id: 0,
                stable_id: id,
                cluster,
                rabitq_code: vec![id as u8, cluster as u8],
                encoded: EncodedCellRow {
                    stable_id: id,
                    rerank_codec: RerankCodec::Sq8Residual,
                    scale: Arc::clone(&q.0),
                    offset: Arc::clone(&q.1),
                    codes: vec![id as u8; DIM],
                    residuals: vec![cluster as u8; DIM],
                    norm_sq: norm,
                },
            }
        };

        let mut w =
            MaterializedRowSpillWriter::create(tmp.path(), 7, DIM, RABITQ_LEN).expect("create");
        w.begin_batch();
        w.append(&row(1, 10, &quant_a, Some(3.5))).expect("append");
        w.append(&row(2, 10, &quant_a, None)).expect("append");
        w.append(&row(3, 11, &quant_b, Some(1.25))).expect("append");
        // Second batch: same quantizer content as quant_a but the dedup map
        // was reset, so it re-appends (correctness over dedup ratio).
        w.begin_batch();
        w.append(&row(4, 10, &quant_a, Some(9.0))).expect("append");
        let spill = w.finish().expect("finish");

        assert_eq!(spill.n_rows, 4);

        let rows = read_spilled_cell_rows(&spill).expect("read back");
        assert_eq!(rows.len(), 4);
        for (got, want_id) in rows.iter().zip([1i128, 2, 3, 4]) {
            assert_eq!(got.stable_id, want_id);
            assert_eq!(got.encoded.stable_id, want_id);
            assert_eq!(got.encoded.rerank_codec, RerankCodec::Sq8Residual);
            assert_eq!(got.rabitq_code.len(), RABITQ_LEN);
            assert_eq!(got.encoded.codes, vec![want_id as u8; DIM]);
        }
        assert_eq!(rows[0].cluster, 10);
        assert_eq!(rows[2].cluster, 11);
        assert_eq!(rows[0].encoded.norm_sq, Some(3.5));
        assert_eq!(rows[1].encoded.norm_sq, None);
        assert_eq!(rows[2].encoded.norm_sq, Some(1.25));
        assert_eq!(rows[2].encoded.scale.as_ref(), vec![2.0f32; DIM].as_slice());
        assert_eq!(
            rows[2].encoded.offset.as_ref(),
            vec![0.25f32; DIM].as_slice()
        );
        // Rows 0 and 1 share one quantizer Arc (same batch, same cluster).
        assert!(Arc::ptr_eq(&rows[0].encoded.scale, &rows[1].encoded.scale));

        let fixed_quant: (Arc<[f32]>, Arc<[f32]>) = (
            Arc::from(vec![SQ8_FIXED_SCALE; DIM]),
            Arc::from(vec![SQ8_FIXED_OFFSET; DIM]),
        );
        let mut fixed = row(10, 13, &fixed_quant, None);
        fixed.encoded.rerank_codec = RerankCodec::Sq8FixedResidual;
        let mut fixed_writer =
            MaterializedRowSpillWriter::create(tmp.path(), 9, DIM, RABITQ_LEN).expect("create");
        fixed_writer.append(&fixed).expect("append fixed");
        let fixed_spill = fixed_writer.finish().expect("finish fixed");
        let fixed_rows = read_spilled_cell_rows(&fixed_spill).expect("read fixed");
        assert_eq!(
            fixed_rows[0].encoded.rerank_codec,
            RerankCodec::Sq8FixedResidual
        );
        assert_eq!(fixed_rows[0].encoded.codes, fixed.encoded.codes);
        assert_eq!(fixed_rows[0].encoded.residuals, fixed.encoded.residuals);

        // Shape mismatches are rejected.
        let mut w2 =
            MaterializedRowSpillWriter::create(tmp.path(), 8, DIM, RABITQ_LEN).expect("create");
        let mut bad = row(9, 12, &quant_a, None);
        bad.encoded.codes = vec![0u8; DIM - 1];
        assert!(w2.append(&bad).is_err(), "shape mismatch must be rejected");

        spill.remove_files();
    }

    #[test]
    fn materialized_row_spill_resume_truncates_partial_batch() {
        const DIM: usize = 8;
        const RABITQ_LEN: usize = 2;
        const CELL: u32 = 3;
        let tmp = tempdir().expect("tempdir");
        let scale: Arc<[f32]> = Arc::from(vec![1.0f32; DIM]);
        let offset: Arc<[f32]> = Arc::from(vec![0.0f32; DIM]);
        let row = |stable_id: i128| MaterializedIvfRow {
            local_doc_id: 0,
            stable_id,
            cluster: CELL,
            rabitq_code: vec![stable_id as u8; RABITQ_LEN],
            encoded: EncodedCellRow {
                stable_id,
                rerank_codec: RerankCodec::Sq8FixedResidual,
                scale: Arc::clone(&scale),
                offset: Arc::clone(&offset),
                codes: vec![stable_id as u8; DIM],
                residuals: vec![0; DIM],
                norm_sq: Some(1.0),
            },
        };

        let mut writer =
            MaterializedRowSpillWriter::create(tmp.path(), CELL, DIM, RABITQ_LEN).expect("create");
        writer.append(&row(1)).expect("append 1");
        writer.append(&row(2)).expect("append 2");
        let state = writer.checkpoint().expect("checkpoint");
        writer.begin_batch();
        writer.append(&row(3)).expect("uncommitted append");
        drop(writer);

        let mut resumed =
            MaterializedRowSpillWriter::resume(tmp.path(), CELL, state).expect("resume");
        resumed.begin_batch();
        resumed.append(&row(4)).expect("replayed batch");
        let spill = resumed.finish().expect("finish");
        let rows = read_spilled_cell_rows(&spill).expect("read");
        assert_eq!(
            rows.iter().map(|row| row.stable_id).collect::<Vec<_>>(),
            vec![1, 2, 4],
            "partial post-checkpoint row must be truncated before replay"
        );
        spill.remove_files();
    }

    #[test]
    fn in_memory_source_yields_full_corpus_in_chunk_size_steps() {
        let n_rows = 25;
        let dim = 3;
        let corpus = synth(n_rows, dim);

        let mut src =
            InMemoryVectorSource::new(Arc::new(corpus.clone()), dim, /*chunk_rows=*/ 7);
        assert_eq!(src.n_rows(), n_rows);
        assert_eq!(src.dim(), dim);
        assert_eq!(src.chunk_rows(), 7);

        // Expect chunks of 7, 7, 7, 4 rows.
        let chunk = src.next_chunk().expect("chunk 0");
        assert_eq!(chunk.len(), 7 * dim);
        assert_eq!(chunk, &corpus[0..7 * dim]);

        let chunk = src.next_chunk().expect("chunk 1");
        assert_eq!(chunk.len(), 7 * dim);
        assert_eq!(chunk, &corpus[7 * dim..14 * dim]);

        let chunk = src.next_chunk().expect("chunk 2");
        assert_eq!(chunk.len(), 7 * dim);
        assert_eq!(chunk, &corpus[14 * dim..21 * dim]);

        let chunk = src.next_chunk().expect("chunk 3 (partial)");
        assert_eq!(chunk.len(), 4 * dim);
        assert_eq!(chunk, &corpus[21 * dim..25 * dim]);

        assert!(src.next_chunk().is_none(), "expected exhausted");
        assert!(src.next_chunk().is_none(), "still exhausted on re-poll");
    }

    #[test]
    fn in_memory_source_reset_replays_from_zero() {
        let n_rows = 10;
        let dim = 4;
        let corpus = synth(n_rows, dim);
        let mut src = InMemoryVectorSource::new(Arc::new(corpus.clone()), dim, 3);

        let first_pass: Vec<f32> = from_fn(|| src.next_chunk().map(|c| c.to_vec()))
            .flatten()
            .collect();
        assert_eq!(first_pass, corpus);

        src.reset();

        let second_pass: Vec<f32> = from_fn(|| src.next_chunk().map(|c| c.to_vec()))
            .flatten()
            .collect();
        assert_eq!(second_pass, corpus, "reset didn't replay full corpus");
    }

    #[test]
    fn mmap_source_chunk_boundary_matches_in_memory() {
        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("xcheck.bin");
        let n_rows = 50;
        let dim = 5;
        let corpus = synth(n_rows, dim);

        let mut w = SpillWriter::create(path.clone()).expect("create");
        w.write_all(cast_slice(&corpus)).expect("write");
        w.finish().expect("finish");

        let chunk_rows = 11;
        let mut mem = InMemoryVectorSource::new(Arc::new(corpus.clone()), dim, chunk_rows);
        let mut mm = MmapVectorSource::open(&path, dim, chunk_rows).expect("mmap");

        loop {
            let a = mem.next_chunk();
            let b = mm.next_chunk();
            match (a, b) {
                (Some(x), Some(y)) => assert_eq!(x, y, "chunk-boundary divergence"),
                (None, None) => break,
                _ => panic!("source exhaustion disagreement"),
            }
        }
    }

    #[test]
    fn mmap_source_rejects_misaligned_file_length() {
        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("bad.bin");
        // Construct a file with 17 bytes — not a multiple of
        // (dim=4) * 4 = 16. Bypasses SpillWriter, which would
        // refuse a non-4-aligned write in debug.
        write(&path, [0u8; 17]).expect("write 17 bytes");
        match MmapVectorSource::open(&path, /*dim=*/ 4, /*chunk_rows=*/ 1) {
            Ok(_) => panic!("expected length-mismatch error, got Ok"),
            Err(BuildError::Io(e)) => {
                assert_eq!(e.kind(), ErrorKind::InvalidData)
            }
            Err(other) => panic!("expected Io InvalidData, got {other:?}"),
        }
    }

    #[test]
    fn empty_corpus_yields_no_chunks() {
        let mem_src = InMemoryVectorSource::new(Arc::new(Vec::<f32>::new()), 4, 8);
        // The trait's `next_chunk` is `&mut self`, so we
        // bind it.
        let mut s = mem_src;
        assert_eq!(s.n_rows(), 0);
        assert!(s.next_chunk().is_none());

        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("empty.bin");
        let w = SpillWriter::create(path.clone()).expect("create");
        w.finish().expect("finish empty");
        let mut s = MmapVectorSource::open(&path, 4, 8).expect("open empty");
        assert_eq!(s.n_rows(), 0);
        assert!(s.next_chunk().is_none());
    }
}
