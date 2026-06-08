// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Streaming spill primitives for the bounded-memory build path.
//!
//! Two cooperating abstractions:
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
//!
//! Both `ChunkedVectorSource` implementations own their backing
//! storage so the trait isn't tied to an external lifetime; the
//! chunk slice returned per iteration is valid for the duration
//! of the `&mut self` borrow, which is the scope the pass-2
//! per-chunk loop runs inside.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;

use crate::superfile::BuildError;

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
    /// `write_all(bytemuck::cast_slice(vec))` but spelled out
    /// so the hot path doesn't have to re-derive the cast on
    /// every call.
    pub fn write_vec(&mut self, vec: &[f32]) -> Result<(), BuildError> {
        let bytes: &[u8] = bytemuck::cast_slice(vec);
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
            return Err(BuildError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
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
        let floats: &[f32] = bytemuck::try_cast_slice(bytes)
            .expect("mmap slice is page-aligned and length is row-aligned");
        Some(floats)
    }

    #[cfg(test)]
    fn reset(&mut self) {
        self.cursor = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

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
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("spill.bin");
        let n_rows = 17;
        let dim = 8;
        let corpus = synth(n_rows, dim);

        // Write the whole thing in one batch via write_all.
        {
            let mut w = SpillWriter::create(path.clone()).expect("create");
            let bytes: &[u8] = bytemuck::cast_slice(&corpus);
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
            let expected: &[u8] = bytemuck::cast_slice(&corpus);
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
        let tmp = tempfile::tempdir().expect("tempdir");
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

        let first_pass: Vec<f32> = std::iter::from_fn(|| src.next_chunk().map(|c| c.to_vec()))
            .flatten()
            .collect();
        assert_eq!(first_pass, corpus);

        src.reset();

        let second_pass: Vec<f32> = std::iter::from_fn(|| src.next_chunk().map(|c| c.to_vec()))
            .flatten()
            .collect();
        assert_eq!(second_pass, corpus, "reset didn't replay full corpus");
    }

    #[test]
    fn mmap_source_chunk_boundary_matches_in_memory() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("xcheck.bin");
        let n_rows = 50;
        let dim = 5;
        let corpus = synth(n_rows, dim);

        let mut w = SpillWriter::create(path.clone()).expect("create");
        w.write_all(bytemuck::cast_slice(&corpus)).expect("write");
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
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("bad.bin");
        // Construct a file with 17 bytes — not a multiple of
        // (dim=4) * 4 = 16. Bypasses SpillWriter, which would
        // refuse a non-4-aligned write in debug.
        std::fs::write(&path, [0u8; 17]).expect("write 17 bytes");
        match MmapVectorSource::open(&path, /*dim=*/ 4, /*chunk_rows=*/ 1) {
            Ok(_) => panic!("expected length-mismatch error, got Ok"),
            Err(BuildError::Io(e)) => {
                assert_eq!(e.kind(), std::io::ErrorKind::InvalidData)
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

        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("empty.bin");
        let w = SpillWriter::create(path.clone()).expect("create");
        w.finish().expect("finish empty");
        let mut s = MmapVectorSource::open(&path, 4, 8).expect("open empty");
        assert_eq!(s.n_rows(), 0);
        assert!(s.next_chunk().is_none());
    }
}
