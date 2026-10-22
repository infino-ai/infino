// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`BlockCachedSource`] — block-granular NVMe retention for lazy
//! (range-GET-backed) superfile reads.
//!
//! The disk cache historically had exactly two states per superfile: a lazy
//! reader whose every `range()` was a fresh object-store GET (nothing
//! retained), or a fully-promoted mmap of the whole object. Between "first
//! touch" and "full promotion" the same byte ranges were re-fetched on every
//! query — and full promotion of everything cannot work once the table
//! outgrows local disk (a 1B-row index is TBs).
//!
//! This source is the missing middle state: reads through it land in a
//! sparse local file at fixed block granularity. A miss fetches one
//! block-aligned GET per run of missing blocks and returns those bytes. Every
//! later read of them, from any query on the shared reader, is local, with no
//! GET. Disk (and budget) use grows with the bytes queries touch, not the object
//! size, which is what lets the cache serve tables far larger than local disk.
//!
//! The cache tracks 64 KiB blocks ([`CACHE_BLOCK_BYTES`]). Each run of a
//! read's missing blocks costs one GET, sized by where the run lies:
//!
//! ```text
//!   where the missing run lies         what its GET covers
//!   ──────────────────────────         ──────────────────────────────────────────────
//!   wholly inside the FTS hole         the read's exact bytes; no blocks are filled
//!   in a 512 KiB group touching the    that whole group, aligned to file offset 0,
//!   aligned region (vector             less the group's leading blocks already
//!   subsection, or a file of           cached
//!   unknown layout)
//!   elsewhere, 512 KiB or longer       its own blocks
//!   elsewhere, shorter, at or past     a window of 512 KiB starting at it: the reads
//!   where one of the last 16 reads     are moving forwards
//!   ended (or the file's first read)
//!   elsewhere, shorter, before all     a window of 512 KiB around it: up to half the
//!   of them                            spare room behind it (more when the end of the
//!                                      file or the aligned region leaves less ahead)
//!                                      over blocks not cached yet, none just before
//!                                      the FTS hole; the rest ahead
//! ```
//!
//! A window fetches through cached stretches of up to 128 KiB and never passes the
//! end of the file. It stops before the aligned region's first block, so it can
//! fill the head of a group touching the region; that group's GET later skips
//! those blocks.
//!
//! Blocks stay small so a large read fetches at most a partial block more at each
//! end; a short read still asks for enough that the reads around it are local,
//! whichever order they come in. A read in the FTS hole is served from blocks that
//! are already there.
//!
//! # A read never waits on the cache disk
//!
//! A query gets its bytes as soon as the GET returns. Writing them to the block
//! file happens afterwards on a blocking thread, because a background download
//! can keep the same disk busy for minutes:
//!
//! ```text
//!   read ─► missing blocks ─► GET ─► bytes ──────────────────► back to the query
//!                                     │
//!                                     ├─► pending: later reads get them from memory
//!                                     ▼
//!          wait for a block writer (a task, no thread)
//!                                     ▼
//!          blocking thread: write ─► mark filled ─► out of pending ─► fsync + index
//! ```
//!
//! - A read looks in pending first, then in the filled blocks; anything in
//!   neither is fetched.
//! - The index on disk lists only blocks that were fsynced, so a crash loses at
//!   most the blocks not synced yet, and later reads fetch them again.
//! - Each store runs at most [`BLOCK_WRITERS`] writes at once, so a slow disk
//!   can't take the blocking threads a query needs.
//! - Each store holds up to [`WRITE_BEHIND_CAP_BYTES`] of fetched runs not
//!   written yet. Past that, a read still gets its bytes, but they are not
//!   cached. A run bigger than the cap is still let in when nothing else is
//!   waiting, and a prefetch, which waits for its own write, is never refused.
//!
//! [`BLOCK_WRITERS`]: super::disk::BLOCK_WRITERS
//! [`WRITE_BEHIND_CAP_BYTES`]: super::disk::WRITE_BEHIND_CAP_BYTES
//!
//! Budget integration: each newly filled run reserves its bytes against the
//! owning [`DiskCacheStore`]'s budget (with LRU eviction pressure) *before*
//! fetching, and the entry's shared `size_bytes` counter grows as blocks
//! land — so eviction sees a lazy entry's true footprint. On budget
//! exhaustion, or once this source's cache entry has been replaced (eviction
//! / mmap promotion), reads stop filling blocks instead of failing: they come
//! from a whole-file local copy when the cache holds one, else uncached from
//! object storage. The source gives its reserved bytes back on `Drop`, when the
//! last reader or the last queued write lets go of it; the sparse file and its
//! index stay on disk so a later generation can adopt them.

#[cfg(test)]
use std::sync::Condvar;
use std::{
    array,
    collections::{BTreeMap, btree_map::Entry},
    fs,
    os::unix::fs::FileExt,
    path::PathBuf,
    process,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::{StreamExt, stream};
use memmap2::Mmap;
use roaring::RoaringBitmap;
use tokio::task::{JoinHandle, spawn_blocking};

#[cfg(test)]
use super::disk::BlockWriteTicket;
use super::disk::{ArcMmapOwner, DiskCacheStore, Reservation};
use crate::{
    runtime_bridge::shared_io_runtime,
    superfile::{LazyByteSource, LazyByteSourceError},
    supertable::manifest::SuperfileUri,
};

/// Cache block size: the unit the cache tracks, fills and indexes. Every GET
/// covers whole blocks, so a read of at least [`MIN_GET_BYTES`] fetches at most
/// one partial block more at each end. Kept small because Parquet column-chunk
/// reads are tens to hundreds of KiB and rarely block-aligned. A 10 GiB file is
/// 160K blocks, so bitmaps stay small.
pub(super) const CACHE_BLOCK_BYTES: u64 = 64 * 1024;

/// What a GET for a short missing run grows to. Blocks are what the cache
/// tracks; this is what one GET asks for, so the two differ on purpose. A run
/// shorter than this grows to a window of this length around it (see
/// [`BlockCachedSource::get_run`]): small reads come in sequences (a column's
/// pages one after another, point lookups near each other in any order), and the
/// next ones are then local. A run at least this long takes only its own blocks,
/// so a large read pays nothing extra.
const MIN_GET_BYTES: u64 = 512 * 1024;

/// [`MIN_GET_BYTES`] in blocks.
const MIN_GET_BLOCKS: u32 = (MIN_GET_BYTES / CACHE_BLOCK_BYTES) as u32;

/// The longest cached stretch, in blocks, a short miss's GET fetches through to
/// reach missing blocks after it inside its window: downloading up to 128 KiB
/// again is cheaper than another GET, while a longer stretch is mostly bytes
/// the cache already holds.
const FETCH_THROUGH_BLOCKS: u32 = 2;

/// How many recent read ends a source keeps to tell a forward scan from a jump
/// back. Each reader of a file (a query's partitions scan their own byte ranges
/// of it, and queries run at once) needs one. With more readers than this, the
/// oldest ends are overwritten, and a reader whose end is gone has its next
/// forward miss taken for a jump back: its window then also covers blocks behind
/// it.
const READ_ENDS: usize = 16;

/// Near the aligned region (see [`SourceLayout::aligned_region`]) a GET covers
/// whole groups of this many bytes, aligned to file offset 0. Vector reads are
/// already 0.25 to 16 MiB merged ranges, so smaller blocks would not cut their
/// over-read, and whole groups let neighbouring cluster prefixes and open-time
/// headers share one GET.
const ALIGNED_GET_BYTES: u64 = 512 * 1024;

/// [`ALIGNED_GET_BYTES`] in blocks.
const ALIGNED_GET_BLOCKS: u32 = (ALIGNED_GET_BYTES / CACHE_BLOCK_BYTES) as u32;

/// A read smaller than this gets a copy of its bytes rather than a slice of the
/// run it came from: a caller may keep it as long as its reader lives (a vector
/// reader keeps its headers), and a slice would keep the whole run in memory.
const COPY_BELOW_BYTES: u64 = 512 * 1024;

/// Leads every block index: the block size it was written with, so a later
/// build reads its block numbers in the right units (see
/// [`indexed_filled_bytes`]).
const IDX_HEADER: [u8; 8] = CACHE_BLOCK_BYTES.to_le_bytes();

/// The block size older builds wrote block indexes with. Such an index
/// converts exactly to ours; one with any other size is not adopted.
const LEGACY_BLOCK_BYTES: u64 = 512 * 1024;

// Every GET size, and the legacy block size, is a whole number of blocks.
const _: () = assert!(
    MIN_GET_BLOCKS > 0
        && MIN_GET_BYTES.is_multiple_of(CACHE_BLOCK_BYTES)
        && ALIGNED_GET_BYTES.is_multiple_of(CACHE_BLOCK_BYTES)
        && LEGACY_BLOCK_BYTES.is_multiple_of(CACHE_BLOCK_BYTES)
);

/// Where a superfile's sections lie, as far as sizing GETs goes: byte ranges
/// `(offset, len)`, `None` when absent or unknown.
#[derive(Clone, Copy, Default)]
pub(crate) struct SourceLayout {
    /// The FTS subsection. A miss wholly inside it is an exact-range GET and
    /// fills no blocks: posting reads are small and scattered, so rounding each
    /// to a block over-fetches many times on a wide OR. Blocks already there
    /// still serve it: those [`BlockCachedSource::prefetch`] filled, or the
    /// leading ones a short GET just before the hole covered. A read only partly
    /// inside behaves as anywhere else.
    pub(crate) fts_hole: Option<(u64, u64)>,
    /// The vector subsection. A GET for a run in a group of [`ALIGNED_GET_BYTES`]
    /// that touches it covers that whole group, so a group at either edge also
    /// covers the bytes just outside the region.
    pub(crate) aligned_region: Option<(u64, u64)>,
}

impl SourceLayout {
    /// Every GET aligned: for a source that serves only vector bytes, and for a
    /// file whose layout is unknown, where vector bytes could lie anywhere.
    pub(crate) const ALL_ALIGNED: Self = Self {
        fts_hole: None,
        aligned_region: Some((0, u64::MAX)),
    };
}

static PERSIST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Lazily-initialized sparse backing file. Created on the first cached read
/// (the object size may only be known after the open-time `tail()` on
/// unknown-size sources). `None` means creation failed once — the source
/// then serves plain passthrough reads forever (cache disabled, not broken).
struct BlockFile {
    file: fs::File,
    size: u64,
    /// Lazily-created read-only mapping for zero-copy hit service.
    /// Inner `None` = mapping failed once; keep serving via pread.
    mmap: OnceLock<Option<Arc<Mmap>>>,
}

/// Block-caching wrapper around a network-backed [`LazyByteSource`].
/// See the module docs for semantics.
pub(crate) struct BlockCachedSource {
    inner: Arc<dyn LazyByteSource>,
    store: Weak<DiskCacheStore>,
    uri: SuperfileUri,
    path: PathBuf,
    /// Distinguishes this source from a replacement entry for the same URI.
    entry_token: Arc<()>,
    /// Whether this source reserves and releases touched-block bytes itself.
    owns_accounting: bool,
    /// Where this file's FTS hole and aligned region lie.
    layout: SourceLayout,
    /// Where the last [`READ_ENDS`] reads ended, `u64::MAX` in unused slots,
    /// written round robin from `next_read_end`. Several readers share one
    /// source (a query's partitions each scan a byte range of the file, and
    /// queries run at once), so a miss continues a scan when it starts at or
    /// past any of them.
    read_ends: [AtomicU64; READ_ENDS],
    next_read_end: AtomicUsize,
    state: OnceLock<Option<BlockFile>>,
    /// Filled-block set. Guarded by a sync mutex; never held across await.
    filled: Mutex<RoaringBitmap>,
    /// Bytes of filled blocks — shared with the owning `CachedEntry`'s
    /// `size_bytes`, so eviction candidates report a lazy entry's real
    /// footprint as it grows.
    filled_bytes: Arc<AtomicU64>,
    /// A handle to this source, so a queued write can keep it alive until the
    /// write is done.
    me: Weak<BlockCachedSource>,
    /// Fetched runs not written to the block file yet, keyed by first block;
    /// reads get their bytes from here meanwhile. Never held across await.
    pending: Mutex<BTreeMap<u32, PendingRun>>,
    /// Set when blocks are marked filled after the last index write. A failed
    /// fsync sets it again, so the next flush retries.
    dirty: AtomicBool,
    /// Set while an index flush runs; see [`Self::flush_index`].
    flushing: AtomicBool,
    /// Tests set this to hold block-file writes; see [`Self::stall_writes`].
    #[cfg(test)]
    write_stall: (Mutex<WriteStallState>, Condvar),
}

/// How a missing run is fetched and cached.
#[derive(Clone, Copy)]
enum Fill {
    /// For a read: may evict colder entries to make room, is not cached once
    /// the cap on unwritten runs is full, and doesn't wait for its write.
    Read,
    /// Ahead of reads: uses only free budget and waits for its own write, so
    /// the cap never refuses it; it holds the bytes until then either way.
    Prefetch,
}

/// A fetched run of blocks waiting to be written to the block file, made by
/// [`BlockCachedSource::admit_run`]. While it exists it holds:
/// - the disk budget reserved for its blocks,
/// - its share of the store's cap on unwritten runs,
/// - its entry in `pending`, so reads get the bytes from memory.
///
/// [`Self::spawn_write`] writes it. Dropping it gives back whatever it still holds,
/// whether the write worked, failed or never ran, and writes the index if the
/// blocks were written.
struct FetchedRun {
    source: Arc<BlockCachedSource>,
    store: Arc<DiskCacheStore>,
    first: u32,
    last: u32,
    bytes: Bytes,
    /// Reserved bytes to give back on drop: all of them until the write is
    /// done, then only those of blocks another fill wrote first.
    unlanded: u64,
    landed: bool,
    /// Declared after `source` so it drops after it: a test that waits for the
    /// writes to finish also sees the source dropped.
    #[cfg(test)]
    _ticket: BlockWriteTicket,
}

impl FetchedRun {
    /// Writes the run once one of the store's block writers is free; `false` if
    /// the write failed. It runs on the shared I/O runtime, never the caller's: a
    /// sync read from a rayon thread runs on a throwaway runtime, and dropping it
    /// would wait for the write or cancel it.
    fn spawn_write(self) -> JoinHandle<bool> {
        shared_io_runtime().spawn(async move {
            let writer = self.store.block_writer().await;
            spawn_blocking(move || {
                let _writer = writer;
                self.land()
            })
            .await
            .unwrap_or(false)
        })
    }

    /// Writes the run to the block file and marks its blocks filled; `false` if
    /// the write failed. Blocking.
    fn land(mut self) -> bool {
        #[cfg(test)]
        self.source.wait_while_stalled();

        let Some(bf) = self.source.created_block_file() else {
            return false;
        };
        let at = u64::from(self.first) * CACHE_BLOCK_BYTES;
        if bf.file.write_all_at(&self.bytes, at).is_err() {
            return false;
        }

        let newly = self.source.mark_filled(bf.size, self.first, self.last);
        self.source.filled_bytes.fetch_add(newly, Ordering::AcqRel);
        self.unlanded = self.unlanded.saturating_sub(newly);
        self.landed = true;
        true
    }
}

impl Drop for FetchedRun {
    fn drop(&mut self) {
        // Out of pending only now, after `land` marked the blocks filled, so a
        // read never finds them in neither place.
        self.source
            .pending
            .lock()
            .expect("pending runs mutex poisoned")
            .remove(&self.first);
        self.store.release_write_behind(self.bytes.len() as u64);

        if self.unlanded > 0 {
            self.store.release_block_bytes(self.unlanded);
        }

        if self.landed {
            self.source.flush_index();
        }
    }
}

/// A fetched run waiting for its block-file write.
struct PendingRun {
    /// Last block of the run (inclusive).
    last: u32,
    /// The run's bytes, starting at its first block.
    bytes: Bytes,
}

/// Where one stretch `[first, last]` of a read's blocks is right now.
enum Segment {
    /// On the block file.
    Filled { first: u32, last: u32 },
    /// In memory: fetched, not yet written. `bytes` start at block `run_first`.
    Pending {
        first: u32,
        last: u32,
        run_first: u32,
        bytes: Bytes,
    },
    /// Neither: the read has to fetch it.
    Missing { first: u32, last: u32 },
}

impl Segment {
    fn blocks(&self) -> (u32, u32) {
        match *self {
            Self::Filled { first, last }
            | Self::Pending { first, last, .. }
            | Self::Missing { first, last } => (first, last),
        }
    }
}

/// Where one block is, while [`BlockCachedSource::segments`] groups blocks.
#[derive(Clone, Copy, PartialEq)]
enum BlockAt {
    Filled,
    Missing,
    /// In the pending run at this index of the snapshot.
    Pending(usize),
}

impl BlockCachedSource {
    #[cfg(test)]
    pub(crate) fn new(
        inner: Arc<dyn LazyByteSource>,
        store: Weak<DiskCacheStore>,
        uri: SuperfileUri,
        path: PathBuf,
    ) -> Arc<Self> {
        Self::new_with_accounting(inner, store, uri, path, true, SourceLayout::default())
    }

    /// Construct a sparse source whose owning entry has already reserved the
    /// complete object size.
    pub(crate) fn new_pre_reserved(
        inner: Arc<dyn LazyByteSource>,
        store: Weak<DiskCacheStore>,
        uri: SuperfileUri,
        path: PathBuf,
        layout: SourceLayout,
    ) -> Arc<Self> {
        Self::new_with_accounting(inner, store, uri, path, false, layout)
    }

    pub(crate) fn new_with_accounting(
        inner: Arc<dyn LazyByteSource>,
        store: Weak<DiskCacheStore>,
        uri: SuperfileUri,
        path: PathBuf,
        owns_accounting: bool,
        layout: SourceLayout,
    ) -> Arc<Self> {
        Arc::new_cyclic(|me| Self {
            inner,
            store,
            uri,
            path,
            entry_token: Arc::new(()),
            owns_accounting,
            // An empty section is no section: nothing can be read in it.
            layout: SourceLayout {
                fts_hole: layout.fts_hole.filter(|&(_, len)| len > 0),
                aligned_region: layout.aligned_region.filter(|&(_, len)| len > 0),
            },
            read_ends: array::from_fn(|_| AtomicU64::new(u64::MAX)),
            next_read_end: AtomicUsize::new(0),
            state: OnceLock::new(),
            filled: Mutex::new(RoaringBitmap::new()),
            filled_bytes: Arc::new(AtomicU64::new(0)),
            me: me.clone(),
            pending: Mutex::new(BTreeMap::new()),
            dirty: AtomicBool::new(false),
            flushing: AtomicBool::new(false),
            #[cfg(test)]
            write_stall: (Mutex::new(WriteStallState::default()), Condvar::new()),
        })
    }

    /// Whether `[start, start + len)` lies fully inside the FTS hole.
    fn in_fts_hole(&self, start: u64, len: u64) -> bool {
        match self.layout.fts_hole {
            Some((off, hole_len)) => start >= off && start + len <= off + hole_len,
            None => false,
        }
    }

    /// The blocks file for a read. A hole read only uses one that exists, since
    /// reads never fill the hole.
    fn block_file_for(&self, in_hole: bool) -> Option<&BlockFile> {
        if in_hole {
            self.created_block_file()
        } else {
            self.block_file()
        }
    }

    /// Fill every block of `[start, start + len)` ahead of reads, with up to
    /// `streams` GETs of about `chunk_bytes` each in flight. Unlike a read, this
    /// also fills the FTS hole, so a scan of the whole hole is served from disk.
    /// Where the range touches the aligned region its ends round out to whole
    /// groups, as a read's would. Uses only free space, never evicting, and stops quietly when
    /// it runs out; later reads then pass through as before.
    pub(crate) async fn prefetch(
        &self,
        start: u64,
        len: u64,
        chunk_bytes: u64,
        streams: usize,
    ) -> Result<(), LazyByteSourceError> {
        let Some(bf) = self.block_file() else {
            return Ok(());
        };
        let end = start.saturating_add(len).min(bf.size);
        if start >= end {
            return Ok(());
        }
        let (b0, b1) = Self::block_span(start, end - start);
        let (b0, b1) = self.aligned_run(bf.size, b0, b1);
        let per_chunk = (chunk_bytes / CACHE_BLOCK_BYTES).max(1) as u32;
        let near_region = self
            .aligned_blocks()
            .is_some_and(|(region_first, region_last)| b0 <= region_last && b1 >= region_first);
        // Near the aligned region, chunks of whole groups from the group holding
        // the range's first block, so no two chunks share a group, a prefetch that
        // stops early leaves none half filled, and a range takes as many chunks as
        // it did with 512 KiB blocks. Elsewhere, chunks from the range's start, so
        // a range no bigger than a chunk is one GET.
        let chunks: Vec<(u32, u32)> = if near_region {
            let per_chunk = per_chunk.div_ceil(ALIGNED_GET_BLOCKS) * ALIGNED_GET_BLOCKS;
            (b0 - b0 % ALIGNED_GET_BLOCKS..=b1)
                .step_by(per_chunk as usize)
                .map(|c0| (c0.max(b0), c0.saturating_add(per_chunk - 1).min(b1)))
                .collect()
        } else {
            (b0..=b1)
                .step_by(per_chunk as usize)
                .map(|c0| (c0, c0.saturating_add(per_chunk - 1).min(b1)))
                .collect()
        };
        // On a stop, chunks not started yet are skipped. Chunks already running
        // finish, so the prefetch returns only after every write it started.
        let stop = AtomicBool::new(false);
        let mut fills = stream::iter(chunks)
            .map(|(c0, c1)| {
                let stop = &stop;
                async move {
                    if stop.load(Ordering::Acquire) {
                        return Ok(true);
                    }
                    self.prefetch_chunk(bf, c0, c1).await
                }
            })
            .buffer_unordered(streams.max(1));
        let mut result = Ok(());
        while let Some(filled) = fills.next().await {
            match filled {
                Ok(true) => {}
                Ok(false) => stop.store(true, Ordering::Release),
                Err(e) => {
                    stop.store(true, Ordering::Release);
                    if result.is_ok() {
                        result = Err(e);
                    }
                }
            }
        }
        result
    }

    /// Whether `token` is this source's own identity token, compared by pointer. Lets the owning
    /// cache entry check "is this source still current" without cloning the `Arc`.
    pub(crate) fn owns_token(&self, token: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.entry_token, token)
    }

    /// Whether this source charges the budget for the blocks it fills, rather than riding on its
    /// owning entry's reservation.
    pub(crate) fn owns_accounting(&self) -> bool {
        self.owns_accounting
    }

    /// Shared filled-bytes counter, installed as the cache entry's
    /// `size_bytes` so accounting and eviction see live growth.
    pub(crate) fn filled_bytes_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.filled_bytes)
    }

    /// The sparse file, created on first use once the object size is known.
    fn block_file(&self) -> Option<&BlockFile> {
        let size = self.inner.size();
        if size == 0 {
            // Size not discovered yet (pre-`tail()` on an unknown-size
            // source) — don't latch the OnceLock; try again next read.
            return None;
        }
        self.state
            .get_or_init(|| {
                if let Some(adopted) = self.try_adopt(size) {
                    return Some(adopted);
                }
                let _ = fs::remove_file(&self.path);
                let _ = fs::remove_file(self.idx_path());
                let file = fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(&self.path)
                    .ok()?;
                file.set_len(size).ok()?;
                Some(BlockFile {
                    file,
                    size,
                    mmap: OnceLock::new(),
                })
            })
            .as_ref()
    }

    /// The sparse file, if it was created. Unlike [`Self::block_file`], never
    /// creates it.
    fn created_block_file(&self) -> Option<&BlockFile> {
        self.state.get()?.as_ref()
    }

    fn idx_path(&self) -> PathBuf {
        let mut p = self.path.clone().into_os_string();
        p.push(".idx");
        PathBuf::from(p)
    }

    /// Adopt a prior generation's sparse file when its persisted index proves
    /// which ranges are valid (immutable superfile, so size-match is enough).
    fn try_adopt(&self, size: u64) -> Option<BlockFile> {
        let meta = fs::metadata(&self.path).ok()?;
        if meta.len() != size {
            return None;
        }
        let idx_bytes = fs::read(self.idx_path()).ok()?;
        let (bitmap, adopted_bytes) = indexed_filled_bytes(size, &idx_bytes)?;
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .ok()?;
        if let Some(store) = self.store.upgrade() {
            if self.owns_accounting {
                store.account_adopted_bytes(&self.uri, adopted_bytes);
            } else {
                store.release_scanned_block_file(&self.uri);
            }
        }
        self.filled_bytes.store(adopted_bytes, Ordering::Release);
        *self.filled.lock().expect("filled bitmap mutex poisoned") = bitmap;
        Some(BlockFile {
            file,
            size,
            mmap: OnceLock::new(),
        })
    }

    /// Serialize the filled-block bitmap for persistence. Taken before the
    /// `sync_data()` barrier so the persisted index only ever names blocks whose
    /// writes the fsync already flushed — a concurrent fill that marks a new
    /// block after this snapshot is simply absent (a safe undercount), never a
    /// durably-claimed hole.
    fn snapshot_index(&self) -> Vec<u8> {
        let filled = self.filled.lock().expect("filled bitmap mutex poisoned");
        serialize_index(&filled)
    }

    /// Write a pre-serialized index snapshot, only after its blocks are durable
    /// AND only while `self.path` still names THIS source's own data inode.
    ///
    /// The shared, generation-less `.blocks` / `.blocks.idx` paths mean a later
    /// generation can unlink our data file and publish a fresh inode (all holes)
    /// at the same path while we are still alive. A plain `path.exists()` guard
    /// then lets a dropped or superseded generation rename its own filled-block
    /// bitmap onto the successor's inode — over-claiming that inode's holes. A
    /// subsequent `try_adopt` trusts the stale bitmap and preads zeros where a
    /// Parquet page-index (ColumnIndex) tag belongs, so the decode fails with
    /// `Required field null_pages is missing`. Comparing inodes keeps the
    /// persisted index faithful to whatever data file the path currently names.
    fn persist_idx(&self, bf: &BlockFile, snapshot: &[u8]) {
        use std::os::unix::fs::MetadataExt;
        let own = match bf.file.metadata() {
            Ok(meta) => meta.ino(),
            Err(_) => return,
        };
        match fs::metadata(&self.path) {
            Ok(meta) if meta.ino() == own => {}
            // Path is gone or now names another generation's inode — never
            // stamp our bitmap onto it.
            _ => return,
        }
        let idx = self.idx_path();
        let seq = PERSIST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut tmp = idx.clone().into_os_string();
        tmp.push(format!(".tmp.{}.{seq}", process::id()));
        let tmp = PathBuf::from(tmp);
        if fs::write(&tmp, snapshot).is_ok() {
            let _ = fs::rename(&tmp, &idx);
        }
    }

    /// Byte length of block `b` (the trailing block may be partial).
    fn block_len(size: u64, b: u32) -> u64 {
        let start = u64::from(b) * CACHE_BLOCK_BYTES;
        (size - start).min(CACHE_BLOCK_BYTES)
    }

    /// Inclusive block index range covering `[start, start + len)`.
    fn block_span(start: u64, len: u64) -> (u32, u32) {
        let b0 = start / CACHE_BLOCK_BYTES;
        let b1 = (start + len - 1) / CACHE_BLOCK_BYTES;
        (b0 as u32, b1 as u32)
    }

    /// Index of the last block of a file of `size` bytes.
    fn last_block(size: u64) -> u32 {
        Self::block_span(0, size.max(1)).1
    }

    /// The blocks one GET for a read's missing run `[first, last]` covers, never
    /// starting before `floor` (blocks this read already fetched). The run after
    /// [`Self::aligned_run`], less the group's leading blocks already cached, is
    /// kept as is when it is at least [`MIN_GET_BYTES`] long or lies in the
    /// aligned region; a shorter one grows to a window of that length:
    ///
    /// ```text
    ///   at or past the block where one of the last READ_ENDS reads ended (or the
    ///   file's first read):
    ///           all of the window ahead. The reads are moving forwards, so blocks
    ///           behind are either cached or skipped (a column the scan does not
    ///           read). Each of a query's partitions scans its own byte range, so
    ///           one end per file would make one partition's forward read look
    ///           like a jump back
    ///   before all of them:
    ///     spare = MIN_GET_BLOCKS - run length
    ///     behind: up to spare / 2 blocks (more when less room is left ahead), over
    ///             blocks not cached yet, stopping at the first cached one, so reads
    ///             arriving in reverse or random order still find their neighbours
    ///             local; none when the FTS hole starts within the window ahead,
    ///             whose term dictionary is read forwards
    ///     ahead:  the rest of the window
    /// ```
    ///
    /// Ahead, a cached stretch of up to FETCH_THROUGH_BLOCKS is fetched through, a
    /// longer one ends the window, and the window ends on the last block not cached
    /// yet. It stops before the aligned region's first block, not before that
    /// block's group, so a short read just before the FTS hole still covers the
    /// postings that the group's head holds. It never passes the end of the file.
    /// A GET still in flight is not pending yet, so two reads at once can cover
    /// the same blocks; the later write finds them filled and gives their budget
    /// back.
    fn get_run(&self, size: u64, first: u32, last: u32, floor: u32) -> (u32, u32) {
        let (grown, last) = self.aligned_run(size, first, last);
        // A group's leading blocks a window before the region already fetched
        // are not fetched again.
        let first = if grown < first {
            self.segments(grown, first - 1)
                .iter()
                .find_map(|segment| match segment {
                    Segment::Missing { first, .. } => Some(*first),
                    _ => None,
                })
                .unwrap_or(first)
        } else {
            grown
        };
        let first = first.max(floor);
        let run = last - first + 1;
        if run >= MIN_GET_BLOCKS {
            return (first, last);
        }
        let (mut lo, mut hi) = (floor, Self::last_block(size));
        if let Some((region_first, region_last)) = self.aligned_blocks() {
            let region_block = self.first_region_block().unwrap_or(region_first);
            if last < region_block {
                // A run before the region: its window stops before the region's
                // first block, which can be inside a group touching the region.
                hi = hi.min(region_block - 1);
            } else if first > region_last {
                lo = lo.max(region_last + 1);
            } else {
                // In the aligned region: whole groups already.
                return (first, last);
            }
        }
        let behind = if self.reads_forward(first) {
            0
        } else {
            // When the FTS hole starts inside the window, after the start of the
            // read's first block (the read ends before it or runs into it), all of the window
            // goes ahead: the term dictionary at the hole's start is read
            // forwards, one lookup after another, so blocks behind the read would
            // only take its place.
            let hole_ahead = self.layout.fts_hole.is_some_and(|(off, _)| {
                let window_end = u64::from(first) + u64::from(MIN_GET_BLOCKS);
                u64::from(first) * CACHE_BLOCK_BYTES < off && off < window_end * CACHE_BLOCK_BYTES
            });
            // Half the spare room behind, more when the end of the file or the
            // aligned region leaves less than that ahead.
            let spare = MIN_GET_BLOCKS - run;
            if hole_ahead {
                0
            } else {
                (spare / 2).max(spare.saturating_sub(hi.saturating_sub(last)))
            }
        };
        let start = match first.checked_sub(1) {
            Some(before) if before >= lo && behind > 0 => {
                match self
                    .segments(first.saturating_sub(behind).max(lo), before)
                    .last()
                {
                    Some(&Segment::Missing { first: from, .. }) => from,
                    _ => first,
                }
            }
            _ => first,
        };
        let window_last = start.saturating_add(MIN_GET_BLOCKS - 1).min(hi);
        let mut end = last;
        if last < window_last {
            for segment in self.segments(last + 1, window_last) {
                let (from, to) = segment.blocks();
                match segment {
                    Segment::Missing { .. } => end = to,
                    _ if to - from < FETCH_THROUGH_BLOCKS => {}
                    _ => break,
                }
            }
        }
        (start, end)
    }

    /// Records where a read ended, over the oldest of the recent ends.
    fn note_read_end(&self, end: u64) {
        let slot = self.next_read_end.fetch_add(1, Ordering::Relaxed) % READ_ENDS;
        self.read_ends[slot].store(end, Ordering::Relaxed);
    }

    /// Whether a miss starting at block `first` continues the reads before it:
    /// it starts at or past the block where one of the recent reads ended, or
    /// it is the file's first read.
    fn reads_forward(&self, first: u32) -> bool {
        let mut seen = false;
        for end in &self.read_ends {
            let end = end.load(Ordering::Relaxed);
            if end == u64::MAX {
                continue;
            }
            seen = true;
            if u64::from(first) >= end.saturating_sub(1) / CACHE_BLOCK_BYTES {
                return true;
            }
        }
        !seen
    }

    /// Run `[first, last]` with each end whose group of [`ALIGNED_GET_BYTES`]
    /// touches the aligned region rounded out to that whole group, clipped to
    /// the file.
    fn aligned_run(&self, size: u64, first: u32, last: u32) -> (u32, u32) {
        let Some((off, len)) = self.layout.aligned_region else {
            return (first, last);
        };
        let group_first = |b: u32| b - b % ALIGNED_GET_BLOCKS;
        let near_region = |b: u32| {
            let start = u64::from(group_first(b)) * CACHE_BLOCK_BYTES;
            start < off.saturating_add(len) && start + ALIGNED_GET_BYTES > off
        };
        let first = if near_region(first) {
            group_first(first)
        } else {
            first
        };
        let last = if near_region(last) {
            (group_first(last) + ALIGNED_GET_BLOCKS - 1).min(Self::last_block(size))
        } else {
            last
        };
        (first, last)
    }

    /// The first and last block of the groups touching the aligned region. A
    /// short run's window enters them only before the region's first block.
    fn aligned_blocks(&self) -> Option<(u32, u32)> {
        let (off, len) = self.layout.aligned_region?;
        let group_blocks = u64::from(ALIGNED_GET_BLOCKS);
        // First block of the group holding `byte`, saturating at the largest
        // block index for a region that runs to the end of any file.
        let group_first =
            |byte: u64| u32::try_from(byte / CACHE_BLOCK_BYTES / group_blocks * group_blocks);
        let first = group_first(off).unwrap_or(u32::MAX);
        let last = group_first(off.saturating_add(len.max(1) - 1))
            .map_or(u32::MAX, |b| b.saturating_add(ALIGNED_GET_BLOCKS - 1));
        Some((first, last))
    }

    /// The block holding the aligned region's first byte.
    fn first_region_block(&self) -> Option<u32> {
        let (off, _) = self.layout.aligned_region?;
        u32::try_from(off / CACHE_BLOCK_BYTES).ok()
    }

    /// Byte start and length of blocks `[first, last]` in a file of `size` bytes.
    fn run_bytes(size: u64, first: u32, last: u32) -> (u64, u64) {
        let start = u64::from(first) * CACHE_BLOCK_BYTES;
        let end = ((u64::from(last) + 1) * CACHE_BLOCK_BYTES).min(size);
        (start, end - start)
    }

    /// The block file that can serve `start..start + len`, or `None` when the read bypasses the
    /// blocks: no block file (in the hole, none created yet), or a range past the end (which the
    /// inner source reports as an error).
    fn blocks_for(&self, start: u64, len: u64, in_hole: bool) -> Option<&BlockFile> {
        let bf = self.block_file_for(in_hole)?;
        (start.saturating_add(len) <= bf.size).then_some(bf)
    }

    /// The cache's fully resident copy of this superfile, when it holds one. Every async read the
    /// blocks cannot serve goes through it before object storage, so a reader a query still holds
    /// after a promotion stays off object storage. The copy is plain bytes, never another block
    /// source, so the read cannot come back here.
    fn whole_file(&self) -> Option<Arc<dyn LazyByteSource>> {
        self.store.upgrade()?.whole_file_source(&self.uri)
    }

    fn all_filled(&self, b0: u32, b1: u32) -> bool {
        self.filled
            .lock()
            .expect("filled bitmap mutex poisoned")
            .contains_range(b0..=b1)
    }

    /// The store, if this source is still the live cache entry. Once eviction or
    /// promotion replaced the entry, this source still serves the blocks it has
    /// but fills no new ones.
    fn store_if_current(&self) -> Option<Arc<DiskCacheStore>> {
        let store = self.store.upgrade()?;
        store
            .lazy_block_entry_is_current(&self.uri, &self.entry_token)
            .then_some(store)
    }

    /// Whether a fetched run is still waiting to be written. The idle sweep keeps
    /// such an entry, because `filled_bytes` does not count the run until then.
    pub(crate) fn has_pending_writes(&self) -> bool {
        !self
            .pending
            .lock()
            .expect("pending runs mutex poisoned")
            .is_empty()
    }

    /// Marks `[b0, b1]` filled and flags the index for the next
    /// [`Self::flush_index`]. Returns the bytes of blocks newly marked (another
    /// fill may have marked some first). Their bytes must already be written.
    fn mark_filled(&self, size: u64, b0: u32, b1: u32) -> u64 {
        let mut filled = self.filled.lock().expect("filled bitmap mutex poisoned");
        let last = Self::last_block(size);
        let short_tail_newly = b1 == last && !filled.contains(last);
        let mut newly = filled.insert_range(b0..=b1) * CACHE_BLOCK_BYTES;
        if short_tail_newly {
            newly -= CACHE_BLOCK_BYTES - Self::block_len(size, last);
        }
        if newly > 0 {
            self.dirty.store(true, Ordering::SeqCst);
        }
        newly
    }

    /// Fsyncs the block file, then writes the index of the blocks marked filled.
    /// Blocking: runs at the end of a block write, or in `Drop`.
    /// - One flush runs per source. A call that finds one running returns at
    ///   once; the running one covers its blocks.
    /// - If blocks were marked during the fsync, the running flush goes again,
    ///   so a burst of writes costs one or two fsyncs, not one each.
    /// - A failed fsync leaves the blocks for the next flush, or for Drop.
    fn flush_index(&self) {
        let Some(bf) = self.created_block_file() else {
            return;
        };

        while self.dirty.load(Ordering::SeqCst) {
            if self.flushing.swap(true, Ordering::SeqCst) {
                return;
            }

            // Clear the flag before reading the bitmap: a block marked after this
            // sets it again, and the loop runs once more.
            self.dirty.store(false, Ordering::SeqCst);
            let snapshot = self.snapshot_index();
            let synced = bf.file.sync_data().is_ok();

            if synced {
                self.persist_idx(bf, &snapshot);
            } else {
                // Keep the flag set so the next flush retries, instead of looping
                // on a failing disk.
                self.dirty.store(true, Ordering::SeqCst);
            }

            self.flushing.store(false, Ordering::SeqCst);

            if !synced {
                return;
            }
        }
    }

    /// Splits blocks `[b0, b1]` into stretches by where their bytes are: pending,
    /// filled or missing. Pending is checked first, and a write marks its blocks
    /// filled before leaving pending, so a block in neither really is missing.
    fn segments(&self, b0: u32, b1: u32) -> Vec<Segment> {
        let pending: Vec<(u32, u32, Bytes)> = self
            .pending
            .lock()
            .expect("pending runs mutex poisoned")
            .range(..=b1)
            .filter(|(_, run)| run.last >= b0)
            .map(|(&first, run)| (first, run.last, run.bytes.clone()))
            .collect();

        let filled = self.filled.lock().expect("filled bitmap mutex poisoned");

        let at = |b: u32| match pending
            .iter()
            .position(|&(first, last, _)| first <= b && b <= last)
        {
            Some(i) => BlockAt::Pending(i),
            None if filled.contains(b) => BlockAt::Filled,
            None => BlockAt::Missing,
        };

        let segment = |first: u32, last: u32, at: BlockAt| match at {
            BlockAt::Filled => Segment::Filled { first, last },
            BlockAt::Missing => Segment::Missing { first, last },
            BlockAt::Pending(i) => Segment::Pending {
                first,
                last,
                run_first: pending[i].0,
                bytes: pending[i].2.clone(),
            },
        };

        let mut segments = Vec::new();

        let (mut first, mut current) = (b0, at(b0));

        for b in b0.saturating_add(1)..=b1 {
            let here = at(b);
            if here != current {
                segments.push(segment(first, b - 1, current));
                (first, current) = (b, here);
            }
        }

        segments.push(segment(first, b1, current));

        segments
    }

    /// Builds `[start, start + len)` from `segments`. `None` if a segment is
    /// missing or a local read fails.
    fn assemble(
        &self,
        bf: &BlockFile,
        start: u64,
        len: u64,
        segments: &[Segment],
    ) -> Option<Bytes> {
        let end = start + len;
        let mut pieces = Vec::with_capacity(segments.len());

        for segment in segments {
            let (first, last) = segment.blocks();
            let s = (u64::from(first) * CACHE_BLOCK_BYTES).max(start);
            let e = ((u64::from(last) + 1) * CACHE_BLOCK_BYTES).min(end);
            pieces.push(match segment {
                Segment::Filled { .. } => self.read_local(bf, s, e - s)?,
                Segment::Pending {
                    run_first, bytes, ..
                } => {
                    let base = u64::from(*run_first) * CACHE_BLOCK_BYTES;
                    bytes.slice((s - base) as usize..(e - base) as usize)
                }
                Segment::Missing { .. } => return None,
            });
        }

        if let [piece] = pieces.as_slice() {
            // See `COPY_BELOW_BYTES`.
            return Some(if len < COPY_BELOW_BYTES {
                Bytes::copy_from_slice(piece)
            } else {
                piece.clone()
            });
        }

        let mut out = BytesMut::with_capacity(len as usize);

        for piece in &pieces {
            out.extend_from_slice(piece);
        }

        Some(out.freeze())
    }

    /// Serves a read that is not fully on the block file: filled blocks from the
    /// file, pending ones from memory, and each missing run with one GET, then
    /// queued for writing. `None` sends the read to the whole-file copy or to
    /// object storage, when:
    /// - the store is gone, or this source is no longer the live entry,
    /// - the budget is full,
    /// - the GET came back short,
    /// - a local read failed.
    async fn read_through(
        &self,
        bf: &BlockFile,
        start: u64,
        len: u64,
    ) -> Result<Option<Bytes>, LazyByteSourceError> {
        let (b0, b1) = Self::block_span(start, len);

        let segments = self.segments(b0, b1);
        let mut served = Vec::with_capacity(segments.len());
        // The run this read fetched last. A GET can cover a later missing stretch
        // of the same read, wholly or in part (an aligned group with a filled
        // block in the middle, or a window fetched through a cached stretch):
        // that part is served from its bytes, and the next GET starts after it.
        let mut fetched: Option<(u32, u32, Bytes)> = None;

        for segment in segments {
            let Segment::Missing { mut first, last } = segment else {
                served.push(segment);
                continue;
            };

            if let Some((run_first, run_last, bytes)) = &fetched
                && *run_first <= first
                && first <= *run_last
            {
                let covered = last.min(*run_last);
                served.push(Segment::Pending {
                    first,
                    last: covered,
                    run_first: *run_first,
                    bytes: bytes.clone(),
                });
                if covered == last {
                    continue;
                }
                first = covered + 1;
            }

            let floor = fetched.as_ref().map_or(0, |(_, run_last, _)| run_last + 1);
            let (run_first, run_last) = self.get_run(bf.size, first, last, floor);
            let Some((bytes, run)) = self
                .fetch_missing(bf, run_first, run_last, Fill::Read)
                .await?
            else {
                return Ok(None);
            };

            if let Some(run) = run {
                // Don't wait for the write: the query already has its bytes.
                drop(run.spawn_write());
            }

            served.push(Segment::Pending {
                first,
                last,
                run_first,
                bytes: bytes.clone(),
            });
            fetched = Some((run_first, run_last, bytes));
        }

        Ok(self.assemble(bf, start, len, &served))
    }

    /// Fetches blocks `[first, last]` with one GET, reserving their budget
    /// first, and queues them for writing. Returns:
    /// - `None` when the read has to go elsewhere: the store is gone or this
    ///   source is no longer the live entry, no budget is left, or the GET
    ///   returned another length (the object is not the one this reader opened);
    /// - the bytes and no run when they are fetched but not queued: the cap on
    ///   unwritten runs is full, or a run starting at the same block is queued.
    ///
    /// If the caller is dropped mid-GET, the reservation gives its bytes back.
    async fn fetch_missing(
        &self,
        bf: &BlockFile,
        first: u32,
        last: u32,
        fill: Fill,
    ) -> Result<Option<(Bytes, Option<FetchedRun>)>, LazyByteSourceError> {
        let Some(store) = self.store_if_current() else {
            return Ok(None);
        };
        let (run_start, run_len) = Self::run_bytes(bf.size, first, last);

        let reservation = if self.owns_accounting {
            let reserved = match fill {
                Fill::Read => store.reserve(run_len).await.ok(),
                Fill::Prefetch => store.try_reserve(run_len),
            };
            let Some(reserved) = reserved else {
                return Ok(None);
            };
            Some(reserved)
        } else {
            None
        };

        let bytes = self.inner.range(run_start, run_len).await?;
        if bytes.len() as u64 != run_len {
            return Ok(None);
        }

        let run = self.admit_run(&store, first, last, &bytes, reservation, fill);
        Ok(Some((bytes, run)))
    }

    /// Queues fetched run `[first, last]` for writing: takes its share of the
    /// store's cap on unwritten runs, puts it in `pending`, and hands it the
    /// reserved budget. `None` (the reservation is given back) if a read finds
    /// the cap full, or a run starting at the same block is already waiting.
    fn admit_run(
        &self,
        store: &Arc<DiskCacheStore>,
        first: u32,
        last: u32,
        bytes: &Bytes,
        reservation: Option<Reservation<'_>>,
        fill: Fill,
    ) -> Option<FetchedRun> {
        let source = self.me.upgrade()?;

        let run_len = bytes.len() as u64;
        match fill {
            Fill::Read => {
                if !store.try_admit_write_behind(run_len) {
                    return None;
                }
            }
            Fill::Prefetch => store.admit_write_behind(run_len),
        }

        match self
            .pending
            .lock()
            .expect("pending runs mutex poisoned")
            .entry(first)
        {
            // That run owns this key until it is written, so this one is served
            // but not cached.
            Entry::Occupied(_) => {
                store.release_write_behind(run_len);
                return None;
            }
            Entry::Vacant(slot) => {
                slot.insert(PendingRun {
                    last,
                    bytes: bytes.clone(),
                });
            }
        }

        let unlanded = reservation.map_or(0, |reserved| {
            reserved.commit();
            run_len
        });

        Some(FetchedRun {
            source,
            #[cfg(test)]
            _ticket: store.block_write_started(),
            store: Arc::clone(store),
            first,
            last,
            bytes: bytes.clone(),
            unlanded,
            landed: false,
        })
    }

    /// Holds this source's block-file writes until the returned guard drops, so a
    /// test can look at a run before it is written. The guard lets them go even
    /// if an assert fails, so a failing test can't hang.
    #[cfg(test)]
    pub(crate) fn stall_writes(&self) -> WriteStall<'_> {
        self.write_stall.0.lock().expect("write stall poisoned").on = true;
        WriteStall(self)
    }

    /// How many block writes are held by [`Self::stall_writes`] right now.
    #[cfg(test)]
    fn stalled_writes(&self) -> usize {
        self.write_stall
            .0
            .lock()
            .expect("write stall poisoned")
            .waiting
    }

    #[cfg(test)]
    fn wait_while_stalled(&self) {
        let (state, resume) = &self.write_stall;
        let mut state = state.lock().expect("write stall poisoned");
        state.waiting += 1;
        while state.on {
            state = resume.wait(state).expect("write stall poisoned");
        }
        state.waiting -= 1;
    }

    /// Serve `[start, start+len)` from the sparse file. `None` on a read
    /// error (caller degrades to passthrough).
    fn read_local(&self, bf: &BlockFile, start: u64, len: u64) -> Option<Bytes> {
        // Zero-copy hit service: filled ranges are handed out as slices of
        // one shared read-only mapping instead of alloc+pread per range. At
        // law width a warm vector query reads ~20 MB through here; the
        // per-range alloc+zero+pread copies were the measured ~6-7 ms
        // prefix-service floor of phase A (and the residual copy cost in
        // the deferred rerank's exact-range gather). Slices keep the
        // mapping alive after eviction, exactly like the promoted
        // parquet/FTS mmaps.
        let mapped = bf.mmap.get_or_init(|| {
            // SAFETY: the blocks file is pre-sized at creation
            // (`file.set_len(size)`) and only ever written through
            // `write_all_at` within that size, so a mapping of the full
            // file never outruns it (no SIGBUS); the read-only shared
            // mapping stays page-cache-coherent with those writes, and
            // only blocks found filled (`all_filled` or a `Filled`
            // segment) are read through it.
            unsafe { Mmap::map(&bf.file) }.ok().map(Arc::new)
        });
        if let Some(m) = mapped.as_ref() {
            let s = usize::try_from(start).ok()?;
            let e = s.checked_add(usize::try_from(len).ok()?)?;
            if e <= m.len() {
                return Some(Bytes::from_owner(ArcMmapOwner(Arc::clone(m))).slice(s..e));
            }
        }
        let mut out = vec![0u8; len as usize];
        bf.file.read_exact_at(&mut out, start).ok()?;
        Some(Bytes::from(out))
    }

    /// One chunk of [`Self::prefetch`]: fetches every block of `[b0, b1]` that is
    /// neither filled nor pending, and waits for it to be written. It reserves
    /// only free budget, so a prefetch never evicts what queries use. A run
    /// another fill queued first is left to it. `false` stops the prefetch, when:
    /// - the budget is full,
    /// - this source is no longer the live entry, or the store is gone,
    /// - the GET came back short, or the write failed.
    async fn prefetch_chunk(
        &self,
        bf: &BlockFile,
        b0: u32,
        b1: u32,
    ) -> Result<bool, LazyByteSourceError> {
        for segment in self.segments(b0, b1) {
            let Segment::Missing { first, last } = segment else {
                continue;
            };

            let Some((_, run)) = self.fetch_missing(bf, first, last, Fill::Prefetch).await? else {
                return Ok(false);
            };

            if let Some(run) = run
                && !run.spawn_write().await.unwrap_or(false)
            {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

/// What [`BlockCachedSource::stall_writes`] holds: whether writes wait, and how
/// many are waiting.
#[cfg(test)]
#[derive(Default)]
struct WriteStallState {
    on: bool,
    waiting: usize,
}

/// Holds a source's block-file writes; see [`BlockCachedSource::stall_writes`].
#[cfg(test)]
pub(crate) struct WriteStall<'a>(&'a BlockCachedSource);

#[cfg(test)]
impl Drop for WriteStall<'_> {
    fn drop(&mut self) {
        let (state, resume) = &self.0.write_stall;
        state.lock().expect("write stall poisoned").on = false;
        resume.notify_all();
    }
}

pub(super) fn serialize_index(bitmap: &RoaringBitmap) -> Vec<u8> {
    let mut buf = Vec::with_capacity(IDX_HEADER.len() + bitmap.serialized_size());
    buf.extend_from_slice(&IDX_HEADER);
    let _ = bitmap.serialize_into(&mut buf);
    buf
}

/// The blocks an index marks filled, in [`CACHE_BLOCK_BYTES`] blocks, and their
/// bytes. An index written with [`LEGACY_BLOCK_BYTES`] converts exactly: each of
/// its blocks is a run of ours at the same offsets. `None` for any other block
/// size, so a corrupt header never claims bytes, or for a block past the end of
/// a `size`-byte file.
pub(super) fn indexed_filled_bytes(size: u64, idx_bytes: &[u8]) -> Option<(RoaringBitmap, u64)> {
    let (header, body) = idx_bytes.split_first_chunk::<8>()?;
    let written_block = u64::from_le_bytes(*header);
    if written_block != CACHE_BLOCK_BYTES && written_block != LEGACY_BLOCK_BYTES {
        return None;
    }
    let written = RoaringBitmap::deserialize_from(body).ok()?;
    if written
        .max()
        .is_some_and(|m| u64::from(m) >= size.div_ceil(written_block))
    {
        return None;
    }
    let per = written_block / CACHE_BLOCK_BYTES;
    let last = BlockCachedSource::last_block(size);
    let bitmap = if per == 1 {
        written
    } else {
        let mut ours = RoaringBitmap::new();
        for b in &written {
            let first = u64::from(b) * per;
            ours.insert_range(first as u32..=((first + per - 1) as u32).min(last));
        }
        ours
    };
    let short_tail = if bitmap.contains(last) {
        CACHE_BLOCK_BYTES - BlockCachedSource::block_len(size, last)
    } else {
        0
    };
    let filled = bitmap.len() * CACHE_BLOCK_BYTES - short_tail;
    Some((bitmap, filled))
}

impl Drop for BlockCachedSource {
    fn drop(&mut self) {
        // Every queued write holds the source, so none is running now. This flush
        // only retries one that failed.
        self.flush_index();
        // Release accounted bytes only; the file and index persist for adoption.
        if self.owns_accounting
            && let Some(store) = self.store.upgrade()
        {
            let filled = self.filled_bytes.load(Ordering::Acquire);
            if filled > 0 {
                store.release_block_bytes(filled);
            }
        }
    }
}

#[async_trait]
impl LazyByteSource for BlockCachedSource {
    fn size(&self) -> u64 {
        self.inner.size()
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        let bytes = self.read(start, len).await;
        if len > 0 {
            self.note_read_end(start.saturating_add(len));
        }
        bytes
    }

    fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
        let bytes = self.read_sync(start, len)?;
        if len > 0 {
            self.note_read_end(start.saturating_add(len));
        }
        Some(bytes)
    }

    async fn tail(&self, len: u64) -> Result<(Bytes, u64), LazyByteSourceError> {
        // Pass through: `tail` both fetches and (on unknown-size sources)
        // discovers the object size, which the inner source caches. Tail
        // bytes are open-time metadata already retained by the open-blob /
        // prefetch overlay above this source, so caching them here as
        // (mostly partial) blocks buys nothing.
        self.inner.tail(len).await
    }
}

impl BlockCachedSource {
    async fn read(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        if len == 0 {
            return Ok(Bytes::new());
        }
        let in_hole = self.in_fts_hole(start, len);
        if let Some(bf) = self.blocks_for(start, len, in_hole) {
            let (b0, b1) = Self::block_span(start, len);
            if self.all_filled(b0, b1) {
                if let Some(bytes) = self.read_local(bf, start, len) {
                    return Ok(bytes);
                }
            } else if !in_hole
                // Missing blocks are fetched and kept, except in the hole,
                // which reads never fill.
                && let Some(bytes) = self.read_through(bf, start, len).await?
            {
                return Ok(bytes);
            }
        }

        match self.whole_file() {
            Some(file) => file.range(start, len).await,
            None => self.inner.range(start, len).await,
        }
    }

    // No whole-file fallback here: a caller that gets `None` takes the async `range`, which has it.
    fn read_sync(&self, start: u64, len: u64) -> Option<Bytes> {
        if len == 0 {
            return Some(Bytes::new());
        }
        let Some(bf) = self.block_file_for(self.in_fts_hole(start, len)) else {
            return self.inner.try_get_range_sync(start, len);
        };
        if start.saturating_add(len) > bf.size {
            return None;
        }
        let (b0, b1) = Self::block_span(start, len);
        if self.all_filled(b0, b1) {
            return self.read_local(bf, start, len);
        }
        // Pending blocks come from memory. A missing block, or a failed local
        // read, sends the read to the inner source.
        self.assemble(bf, start, len, &self.segments(b0, b1))
            .or_else(|| self.inner.try_get_range_sync(start, len))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeSet,
        future::pending,
        path::Path,
        sync::{atomic::AtomicUsize, mpsc},
        thread,
        time::Duration,
    };

    use futures::FutureExt;
    use tempfile::tempdir;
    use tokio::{
        sync::{Barrier, Notify},
        time::{sleep, timeout},
    };

    use super::*;
    use crate::{
        runtime_bridge::bridge_sync_to_async_send,
        supertable::reader_cache::{
            ColdFetchMode, DiskCacheConfig, LruPolicy, disk::BLOCK_WRITERS,
        },
    };

    /// How long a test waits for something that should happen right away, before
    /// failing instead of hanging.
    const NO_WAIT_DEADLINE: Duration = Duration::from_secs(10);

    /// How often [`wait_until`] checks its condition.
    const POLL_INTERVAL: Duration = Duration::from_millis(1);

    /// How long a test waits to see that something does not happen.
    const STAYS_PUT: Duration = Duration::from_millis(50);

    /// Waits until `cond` holds, failing with `what` after [`NO_WAIT_DEADLINE`].
    async fn wait_until(mut cond: impl FnMut() -> bool, what: &str) {
        timeout(NO_WAIT_DEADLINE, async {
            while !cond() {
                sleep(POLL_INTERVAL).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting until {what}"));
    }

    /// In-memory fake source that counts `range` calls and the bytes they ask for.
    struct CountingSource {
        blob: Bytes,
        calls: AtomicUsize,
        bytes: AtomicU64,
    }

    impl CountingSource {
        fn new(n: usize) -> Self {
            let blob: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
            Self {
                blob: Bytes::from(blob),
                calls: AtomicUsize::new(0),
                bytes: AtomicU64::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Acquire)
        }

        fn fetched_bytes(&self) -> u64 {
            self.bytes.load(Ordering::Acquire)
        }
    }

    #[async_trait]
    impl LazyByteSource for CountingSource {
        fn size(&self) -> u64 {
            self.blob.len() as u64
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            self.bytes.fetch_add(len, Ordering::AcqRel);
            let (s, e) = (start as usize, (start + len) as usize);
            if e > self.blob.len() {
                return Err(LazyByteSourceError::OutOfBounds {
                    start,
                    len,
                    size: self.blob.len() as u64,
                });
            }
            Ok(self.blob.slice(s..e))
        }
    }

    /// A store whose budget admits everything; `noop_storage` is never hit
    /// because the block source's inner fake serves all reads.
    fn test_store(dir: &Path, budget: u64) -> Arc<DiskCacheStore> {
        use std::{ops::Range, time::SystemTime};

        use object_store::MultipartUpload;

        use crate::storage::{ObjectMeta, StorageError, StorageProvider};

        #[derive(Debug)]
        struct NoopStorage;

        fn unimplemented_err(uri: &str) -> StorageError {
            StorageError::Permanent {
                uri: uri.into(),
                source: "noop storage".into(),
            }
        }

        #[async_trait]
        impl StorageProvider for NoopStorage {
            async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
                let _ = uri;
                Ok(ObjectMeta {
                    size: 0,
                    etag: None,
                    last_modified: SystemTime::UNIX_EPOCH,
                })
            }
            async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn get_range(&self, uri: &str, _r: Range<u64>) -> Result<Bytes, StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn put_atomic(
                &self,
                uri: &str,
                _b: Bytes,
            ) -> Result<Option<String>, StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn put_overwrite(&self, uri: &str, _b: Bytes) -> Result<(), StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn put_if_match(
                &self,
                uri: &str,
                _b: Bytes,
                _etag: Option<&str>,
            ) -> Result<Option<String>, StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn put_multipart(
                &self,
                uri: &str,
            ) -> Result<Box<dyn MultipartUpload>, StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn delete(&self, _uri: &str) -> Result<(), StorageError> {
                Ok(())
            }
        }

        let cfg = DiskCacheConfig {
            cache_root: dir.to_path_buf(),
            disk_budget_bytes: budget,
            cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
            eviction: Box::new(LruPolicy::new()),
            ..DiskCacheConfig::default()
        };
        DiskCacheStore::new_unpinned(Arc::new(NoopStorage), cfg).expect("test store")
    }

    /// Second read of the same bytes never touches the inner source, the
    /// returned bytes are identical, and accounting matches the touched
    /// block footprint (not the object size).
    #[tokio::test]
    async fn repeat_reads_are_served_locally_and_accounted_by_blocks() {
        const OBJ: usize = 3 * UNIT as usize + 1000;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let inner = Arc::new(CountingSource::new(OBJ));
        let src = BlockCachedSource::new(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("t.blocks"),
        );
        // Install the source as current for its (synthetic) entry.
        store.install_block_entry_for_test(uri, Arc::clone(&src));

        // Read spanning units 0..=2 (one contiguous missing run → 1 GET).
        let start = 100u64;
        let len = 3 * UNIT - 200;
        let first = src.range(start, len).await.expect("first read");
        assert_eq!(first, inner.blob.slice(100..(start + len) as usize));
        assert_eq!(inner.calls(), 1, "one block-run GET for the miss");

        // Identical read → zero inner calls.
        let second = src.range(start, len).await.expect("second read");
        assert_eq!(second, first);
        assert_eq!(inner.calls(), 1, "repeat read must not touch the source");

        // Sub-range and sync reads also come from local blocks.
        let sub = src.range(start + 10, 100).await.expect("sub read");
        assert_eq!(sub, inner.blob.slice(110..210));
        let sync = src
            .try_get_range_sync(start + 10, 100)
            .expect("sync read of filled blocks");
        assert_eq!(sync, sub);
        assert_eq!(inner.calls(), 1);

        // Accounting = 3 whole units (0..=2), not the object size.
        store.block_writes_settled().await;
        let expected = 3 * UNIT;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), expected);
        assert_eq!(store.stats().current_bytes, expected);

        // Touch the trailing partial block: its length is size - 3 units.
        let tail_start = 3 * UNIT + 10;
        let t = src.range(tail_start, 50).await.expect("tail block read");
        store.block_writes_settled().await;
        assert_eq!(
            t,
            inner
                .blob
                .slice(tail_start as usize..tail_start as usize + 50)
        );
        assert_eq!(inner.calls(), 2);
        assert_eq!(
            src.filled_bytes_handle().load(Ordering::Acquire),
            expected + 1000,
            "trailing partial block accounts its real length"
        );

        // Drop the source: accounting released, but the sparse file and its
        // index persist for a later generation to adopt.
        let path = dir.path().join("t.blocks");
        assert!(path.exists());
        store.remove_block_entry_for_test(&uri);
        drop(src);
        assert_eq!(store.stats().current_bytes, 0);
        assert!(path.exists(), "blocks file survives drop for adoption");
        let mut idx = path.clone().into_os_string();
        idx.push(".idx");
        assert!(
            std::path::PathBuf::from(idx).exists(),
            "index sidecar persisted alongside the blocks file"
        );
    }

    /// Reopen/reconnect warm survival: a fresh source adopts a prior
    /// generation's persisted blocks and serves filled ranges with zero refetch.
    #[tokio::test]
    async fn adopt_reuses_blocks_across_source_generations() {
        const OBJ: usize = 3 * UNIT as usize + 1000;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let path = dir.path().join("t.blocks");

        let inner1 = Arc::new(CountingSource::new(OBJ));
        let src1 = BlockCachedSource::new(
            Arc::clone(&inner1) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src1));
        let start = 100u64;
        let len = 2 * UNIT + 500;
        let first = src1.range(start, len).await.expect("gen1 read");
        store.block_writes_settled().await;
        assert_eq!(inner1.calls(), 1);
        let filled = src1.filled_bytes_handle().load(Ordering::Acquire);

        store.remove_block_entry_for_test(&uri);
        drop(src1);
        assert_eq!(store.stats().current_bytes, 0);
        assert!(path.exists());

        let inner2 = Arc::new(CountingSource::new(OBJ));
        let src2 = BlockCachedSource::new(
            Arc::clone(&inner2) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src2));

        let again = src2.range(start, len).await.expect("gen2 read");
        assert_eq!(again, first);
        assert_eq!(inner2.calls(), 0, "adopted blocks serve without refetch");
        assert_eq!(
            src2.filled_bytes_handle().load(Ordering::Acquire),
            filled,
            "adopted footprint re-accounted"
        );
        assert_eq!(store.stats().current_bytes, filled);

        let tail_start = 3 * UNIT + 10;
        let _ = src2.range(tail_start, 50).await.expect("gen2 tail");
        assert_eq!(inner2.calls(), 1, "an unfilled range still fetches");
    }

    /// An absent index sidecar makes adoption fall back to a fresh file rather
    /// than trust the sparse bytes blindly.
    #[tokio::test]
    async fn adopt_falls_back_to_fresh_when_index_is_missing() {
        const OBJ: usize = 3 * CACHE_BLOCK_BYTES as usize + 1000;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let path = dir.path().join("t.blocks");

        let inner1 = Arc::new(CountingSource::new(OBJ));
        let src1 = BlockCachedSource::new(
            Arc::clone(&inner1) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src1));
        let _ = src1
            .range(100, 2 * CACHE_BLOCK_BYTES + 500)
            .await
            .expect("gen1");
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
        drop(src1);

        let mut idx = path.clone().into_os_string();
        idx.push(".idx");
        std::fs::remove_file(PathBuf::from(idx)).expect("drop the index");

        let inner2 = Arc::new(CountingSource::new(OBJ));
        let src2 = BlockCachedSource::new(
            Arc::clone(&inner2) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src2));
        let _ = src2
            .range(100, 2 * CACHE_BLOCK_BYTES + 500)
            .await
            .expect("gen2");
        assert_eq!(
            inner2.calls(),
            1,
            "no index means fresh fetch, not stale reuse"
        );
    }

    #[test]
    fn indexed_filled_bytes_rejects_a_block_id_past_the_file() {
        let size = 2 * CACHE_BLOCK_BYTES;
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(2);
        assert!(indexed_filled_bytes(size, &serialize_index(&bitmap)).is_none());
    }

    #[test]
    fn indexed_filled_bytes_sums_in_range_blocks() {
        let size = 2 * CACHE_BLOCK_BYTES + 100;
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(0);
        bitmap.insert(2);
        let (_, filled) =
            indexed_filled_bytes(size, &serialize_index(&bitmap)).expect("valid index");
        assert_eq!(filled, CACHE_BLOCK_BYTES + 100);
    }

    #[test]
    fn indexed_filled_bytes_rejects_a_mismatched_header() {
        let size = 2 * CACHE_BLOCK_BYTES;
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(0);
        let mut bytes = serialize_index(&bitmap);
        bytes[0] ^= 0xFF;
        assert!(indexed_filled_bytes(size, &bytes).is_none());
    }

    /// Regression for the transient `Required field type_ is missing` decode
    /// crash: two sources for the same superfile share a deterministic
    /// `.blocks` path. The second source's `block_file()` opens with
    /// `truncate(true)`, zeroing the sparse file the first still holds open —
    /// so the first, still trusting its "filled" bitmap, reads zeros where it
    /// had cached real bytes. (In production the second source is a
    /// post-eviction cold refetch of a superfile a long scan still holds; the
    /// zeros land where a Parquet Thrift tag belongs and the decode panics.)
    ///
    /// The second source reads a DISJOINT range so its truncate does not
    /// coincidentally refill the first source's blocks. Fails today (the first
    /// reader reads zeros); passes once eviction unlinks the `.blocks` file so
    /// the refetch gets a fresh inode and the live reader keeps its own.
    #[tokio::test]
    async fn stale_shared_blocks_file_must_not_corrupt_live_reader() {
        const OBJ: usize = 4 * CACHE_BLOCK_BYTES as usize + 1000;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        // Both sources map to the same per-URI scratch path (the bug).
        let path = dir.path().join("shared.blocks");

        // Reader A becomes the current entry, fills blocks 0..=2 onto the
        // shared file, and stays alive (a long scan holding it).
        let inner_a = Arc::new(CountingSource::new(OBJ));
        let a = BlockCachedSource::new(
            Arc::clone(&inner_a) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&a));
        let start = 100u64;
        let len = 2 * CACHE_BLOCK_BYTES + 500;
        let want = inner_a.blob.slice(start as usize..(start + len) as usize);
        let a_first = a.range(start, len).await.expect("A first read");
        store.block_writes_settled().await;
        assert_eq!(a_first, want, "A reads correct bytes before the collision");
        assert_eq!(inner_a.calls(), 1, "A's blocks are on the shared file");

        // Eviction removes the catalog entry but leaves the `.blocks` file on
        // disk (today's bug); reader A is still held by its long scan.
        store.remove_block_entry_for_test(&uri);

        // Reader B is the post-eviction refetch: same uri, same path, becomes
        // current, and fills only the trailing block — but its `block_file()`
        // truncates the shared inode A still holds open first.
        let inner_b = Arc::new(CountingSource::new(OBJ));
        let b = BlockCachedSource::new(
            Arc::clone(&inner_b) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&b));
        let tail = 3 * CACHE_BLOCK_BYTES + 10;
        let _ = b
            .range(tail, 50)
            .await
            .expect("B read (truncates shared file)");
        store.block_writes_settled().await;

        // A re-reads the range it already cached. Its bitmap still says filled,
        // so it serves from the now-truncated shared file.
        let a_again = a.range(start, len).await.expect("A re-read");
        assert_eq!(
            a_again, want,
            "live reader A must still serve its cached bytes, not zeros"
        );
    }

    /// A superseded generation's `.blocks.idx` must never over-claim a
    /// SUCCESSOR generation's data inode. This is the sibling of the
    /// truncate race above: `#489` stopped a refetch from truncating a held
    /// inode by giving each generation a fresh inode; but a dropped
    /// generation still stamps its own filled-block bitmap onto the shared
    /// `.blocks.idx` path — which, after eviction + refetch, now names a
    /// DIFFERENT inode whose corresponding blocks are holes. A later
    /// generation then `try_adopt`s that data file on a size-match alone,
    /// trusts the stale bitmap, and preads zeros where a Parquet page-index
    /// (ColumnIndex) tag belongs → `Required field null_pages is missing`.
    ///
    /// Interleaving: A (gen1) fetches block b onto inode X, its write queued.
    /// Eviction unlinks the files (X stays alive under A). B (gen2)
    /// cold-opens → fresh inode Y (holes), fills a DISJOINT block, persists
    /// idx {other}. A's write runs and persists {b}, over Y's path. D (gen3)
    /// adopts Y (size matches) with bitmap {b} and reads block b → zeros.
    /// Fails today; passes once `persist_idx` fences its write to its own
    /// data inode.
    #[tokio::test]
    async fn dropped_generation_idx_must_not_overclaim_successor_inode() {
        const OBJ: usize = 4 * CACHE_BLOCK_BYTES as usize + 1000;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let path = dir.path().join("shared.blocks");

        // Gen1 A fetches block 0 (covering `start`) onto inode X. Its write is
        // held until gen2 owns the path.
        let inner_a = Arc::new(CountingSource::new(OBJ));
        let a = BlockCachedSource::new(
            Arc::clone(&inner_a) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&a));
        let start = 100u64;
        let len = CACHE_BLOCK_BYTES; // block 0 (and a sliver of 1)
        let want = inner_a.blob.slice(start as usize..(start + len) as usize);
        let a_stall = a.stall_writes();
        assert_eq!(a.range(start, len).await.expect("A fill"), want);

        // Real eviction: drop the catalog entry AND unlink the on-disk files,
        // so the next generation cannot adopt them and gets a FRESH inode. A
        // survives (held by its queued write) with inode X open.
        store.remove_block_entry_for_test(&uri);
        let idx_path = a.idx_path();
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&idx_path);

        // Gen2 B cold-opens: try_adopt fails (files gone) → fresh inode Y, all
        // holes. It fills a DISJOINT trailing block and persists idx {3} for Y.
        let inner_b = Arc::new(CountingSource::new(OBJ));
        let b = BlockCachedSource::new(
            Arc::clone(&inner_b) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&b));
        let tail = 3 * CACHE_BLOCK_BYTES + 10;
        let _ = b.range(tail, 50).await.expect("B fill disjoint block");
        wait_until(|| idx_path.exists(), "B writes its index").await;

        // A's write runs now and writes its {0} bitmap. WITHOUT the inode fence
        // this renames {0} onto `.blocks.idx`, which now names Y (block 0 = hole).
        drop(a_stall);
        store.block_writes_settled().await;
        drop(a);

        // Gen3 D cold-opens → try_adopt(size) accepts Y (len matches) with the
        // over-claimed bitmap {0}, then serves block 0 from a hole = zeros.
        let inner_d = Arc::new(CountingSource::new(OBJ));
        let d = BlockCachedSource::new(
            Arc::clone(&inner_d) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&d));
        let d_read = d.range(start, len).await.expect("D read");
        assert_eq!(
            d_read, want,
            "D must serve real bytes, not zeros from a successor inode's hole \
             claimed by a dropped generation's stale bitmap"
        );
    }

    /// Two disjoint missing runs in one request → one GET per run.
    #[tokio::test]
    async fn disjoint_missing_runs_fetch_separately() {
        const OBJ: usize = 6 * UNIT as usize;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let inner = Arc::new(CountingSource::new(OBJ));
        let src = BlockCachedSource::new(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("runs.blocks"),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src));

        // Fill unit 2 first.
        let b = UNIT;
        let _ = src.range(2 * b, 10).await.expect("fill middle unit");
        assert_eq!(inner.calls(), 1);

        // Read units 1..=3: units 1 and 3 are missing → two run GETs.
        let got = src.range(b, 3 * b).await.expect("spanning read");
        assert_eq!(got, inner.blob.slice(b as usize..(4 * b) as usize));
        assert_eq!(inner.calls(), 3, "two missing runs around the filled unit");

        store.remove_block_entry_for_test(&uri);
    }

    /// When the entry is no longer current (evicted/promoted), reads still
    /// succeed as passthrough and accounting stops growing.
    #[tokio::test]
    async fn stale_entry_degrades_to_passthrough() {
        const OBJ: usize = 2 * CACHE_BLOCK_BYTES as usize;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let inner = Arc::new(CountingSource::new(OBJ));
        let src = BlockCachedSource::new(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("stale.blocks"),
        );
        // Never installed as current → every miss is passthrough.
        let a = src.range(0, 64).await.expect("passthrough read");
        let bb = src.range(0, 64).await.expect("passthrough read again");
        assert_eq!(a, bb);
        assert_eq!(inner.calls(), 2, "uncached passthrough on both reads");
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 0);
        assert_eq!(store.stats().current_bytes, 0);
    }

    /// A source whose FTS hole is `hole`, over an `obj`-byte blob,
    /// installed as current for a fresh uri.
    fn holed_source(
        dir: &Path,
        store: &Arc<DiskCacheStore>,
        obj: usize,
        hole: (u64, u64),
    ) -> (SuperfileUri, Arc<CountingSource>, Arc<BlockCachedSource>) {
        let uri = SuperfileUri::new_v4();
        let inner = Arc::new(CountingSource::new(obj));
        let src = BlockCachedSource::new_with_accounting(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(store),
            uri,
            dir.join("hole.blocks"),
            true,
            SourceLayout {
                fts_hole: Some(hole),
                ..SourceLayout::default()
            },
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src));
        (uri, inner, src)
    }

    /// Hole reads stay exact passthrough until a prefetch fills the hole; then
    /// they, and sync reads, come from disk with no more GETs.
    #[tokio::test]
    async fn prefetched_hole_is_served_from_disk() {
        let b = CACHE_BLOCK_BYTES;
        let obj = 16 * b as usize;
        let hole = (2 * b + 100, 12 * b);
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = holed_source(dir.path(), &store, obj, hole);

        // Before: every hole read is its own exact GET.
        let read_start = 3 * b + 5;
        let _ = src.range(read_start, 1000).await.expect("hole read");
        let _ = src.range(read_start, 1000).await.expect("hole read again");
        assert_eq!(inner.calls(), 2, "hole reads pass through");
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 0);

        // Blocks 2..=14 in 4-block chunks: 4 GETs.
        src.prefetch(hole.0, hole.1, 4 * b, 2)
            .await
            .expect("prefetch");
        assert_eq!(inner.calls(), 6, "one GET per prefetch chunk");
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 13 * b);

        let got = src.range(read_start, 1000).await.expect("hole read");
        assert_eq!(
            got,
            inner
                .blob
                .slice(read_start as usize..read_start as usize + 1000)
        );
        let sync = src
            .try_get_range_sync(read_start, 1000)
            .expect("sync hole read from disk");
        assert_eq!(sync, got);
        assert_eq!(inner.calls(), 6, "prefetched hole reads need no GETs");

        store.remove_block_entry_for_test(&uri);
    }

    /// `prefetch_range` fills a lazy entry's hole through the store.
    #[tokio::test]
    async fn prefetch_range_fills_a_lazy_entry() {
        let b = CACHE_BLOCK_BYTES;
        let hole = (b, 4 * b);
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = holed_source(dir.path(), &store, 8 * b as usize, hole);

        store
            .prefetch_range(&uri, hole.0, hole.1)
            .await
            .expect("prefetch");
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 4 * b);
        let calls = inner.calls();
        let _ = src.range(2 * b, 100).await.expect("hole read");
        assert_eq!(inner.calls(), calls, "served from the prefetched blocks");

        store.remove_block_entry_for_test(&uri);
    }

    /// A prefetch that runs out of budget stops without an error, leaves no
    /// budget reserved for blocks it did not fill, and reads it did not cover
    /// still pass through correctly.
    #[tokio::test]
    async fn prefetch_stops_when_the_budget_runs_out() {
        let b = CACHE_BLOCK_BYTES;
        let obj = 10 * b as usize;
        let hole = (0, 10 * b);
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), 2 * b);
        let (uri, inner, src) = holed_source(dir.path(), &store, obj, hole);

        src.prefetch(hole.0, hole.1, b, 4)
            .await
            .expect("a budget stop is not an error");
        let filled = src.filled_bytes_handle().load(Ordering::Acquire);
        assert!(filled <= 2 * b);
        assert_eq!(
            store.stats().current_bytes,
            filled,
            "only filled blocks stay charged"
        );

        let late = 8 * b + 7;
        let got = src.range(late, 100).await.expect("uncovered hole read");
        assert_eq!(got, inner.blob.slice(late as usize..late as usize + 100));

        store.remove_block_entry_for_test(&uri);
    }

    /// A prefetch that fills nothing does not rewrite the block index.
    #[tokio::test]
    async fn prefetch_of_filled_blocks_skips_the_index_write() {
        let b = CACHE_BLOCK_BYTES;
        let hole = (0, 4 * b);
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = holed_source(dir.path(), &store, 4 * b as usize, hole);

        src.prefetch(hole.0, hole.1, b, 2)
            .await
            .expect("first prefetch");
        let idx = src.idx_path();
        assert!(idx.exists(), "first prefetch writes the index");
        fs::remove_file(&idx).expect("remove index");

        let calls = inner.calls();
        src.prefetch(hole.0, hole.1, b, 2)
            .await
            .expect("second prefetch");
        assert_eq!(inner.calls(), calls, "nothing left to fetch");
        assert!(
            !idx.exists(),
            "nothing filled, so the index is not rewritten"
        );

        store.remove_block_entry_for_test(&uri);
    }

    /// Prefetching a uri with no lazy entry does nothing.
    #[tokio::test]
    async fn prefetch_range_skips_a_uri_that_is_not_open_lazily() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        store
            .prefetch_range(&SuperfileUri::new_v4(), 0, CACHE_BLOCK_BYTES)
            .await
            .expect("no entry is not an error");
    }

    /// A source whose GETs never finish.
    struct HangingSource {
        size: u64,
    }

    #[async_trait]
    impl LazyByteSource for HangingSource {
        fn size(&self) -> u64 {
            self.size
        }

        async fn range(&self, _start: u64, _len: u64) -> Result<Bytes, LazyByteSourceError> {
            pending().await
        }
    }

    /// A read dropped while its GET is in flight gives back the budget it
    /// reserved for the fill.
    #[tokio::test]
    async fn dropped_fill_releases_its_reservation() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let src = BlockCachedSource::new(
            Arc::new(HangingSource {
                size: 4 * CACHE_BLOCK_BYTES,
            }) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("hang.blocks"),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src));

        // Polled once: the fill reserves, then waits on the GET and is dropped.
        assert!(src.range(0, 100).now_or_never().is_none(), "the GET hangs");
        assert_eq!(
            store.stats().current_bytes,
            0,
            "the reservation is given back"
        );

        store.remove_block_entry_for_test(&uri);
    }

    /// A prefetch on a full cache fills nothing rather than evict blocks a
    /// query filled.
    #[tokio::test]
    async fn prefetch_never_evicts() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), 2 * b);
        let uri = SuperfileUri::new_v4();
        let inner = Arc::new(CountingSource::new(2 * b as usize));
        let queried = BlockCachedSource::new(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("queried.blocks"),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&queried));
        let _ = queried.range(0, 2 * b).await.expect("query read");
        assert_eq!(store.stats().current_bytes, 2 * b, "the cache is full");

        let (other_uri, _, other) = holed_source(dir.path(), &store, 4 * b as usize, (0, 4 * b));
        other.prefetch(0, 4 * b, b, 2).await.expect("prefetch");
        assert_eq!(other.filled_bytes_handle().load(Ordering::Acquire), 0);
        assert!(store.is_cached(&uri), "the query's entry stays");
        assert_eq!(store.stats().current_bytes, 2 * b);
        let calls = inner.calls();
        let _ = queried.range(0, 2 * b).await.expect("query read again");
        assert_eq!(inner.calls(), calls, "still served from its blocks");

        store.remove_block_entry_for_test(&other_uri);
        store.remove_block_entry_for_test(&uri);
    }

    /// A [`CountingSource`] over an `obj`-byte blob, behind an installed source
    /// whose aligned region is `aligned`.
    fn layout_source(
        dir: &Path,
        store: &Arc<DiskCacheStore>,
        obj: usize,
        aligned: Option<(u64, u64)>,
    ) -> (SuperfileUri, Arc<CountingSource>, Arc<BlockCachedSource>) {
        let inner = Arc::new(CountingSource::new(obj));
        let uri = SuperfileUri::new_v4();
        let src = BlockCachedSource::new_with_accounting(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(store),
            uri,
            dir.join("layout.blocks"),
            true,
            SourceLayout {
                aligned_region: aligned,
                ..SourceLayout::default()
            },
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src));
        (uri, inner, src)
    }

    /// The bytes one GET for a short read covers: tests that count fills count
    /// in this unit.
    const UNIT: u64 = MIN_GET_BYTES;

    /// Bytes in one aligned group.
    const GROUP: u64 = ALIGNED_GET_BYTES;

    /// Outside the aligned region a GET for a missing run at least as long as
    /// `MIN_GET_BYTES` covers only the blocks the run touches.
    #[tokio::test]
    async fn a_long_miss_gets_only_its_blocks() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(dir.path(), &store, 4 * GROUP as usize, None);

        let (start, len) = (10, MIN_GET_BYTES + 100);
        let got = src.range(start, len).await.expect("read");
        assert_eq!(
            got,
            inner.blob.slice(start as usize..(start + len) as usize)
        );
        assert_eq!(inner.calls(), 1);
        assert_eq!(
            inner.fetched_bytes(),
            MIN_GET_BYTES + b,
            "the blocks it touches, no more"
        );

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A GET for a short miss before where the recent reads ended covers a window
    /// of `MIN_GET_BYTES` around it, so the reads near it cost no GET, before it
    /// or after it.
    #[tokio::test]
    async fn a_short_miss_behind_the_recent_reads_gets_the_blocks_around_it() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(dir.path(), &store, 4 * GROUP as usize, None);

        let _ = src.range(20 * b + 10, 100).await.expect("read block 20");
        let before = inner.fetched_bytes();
        let _ = src.range(4 * b + 10, 100).await.expect("read block 4");
        assert_eq!(inner.fetched_bytes() - before, MIN_GET_BYTES, "one window");
        store.block_writes_settled().await;
        assert!(src.all_filled(1, 8), "three blocks behind it, four ahead");
        for block in 1..=8 {
            let start = block * b + 10;
            let got = src.range(start, 100).await.expect("a nearby page");
            assert_eq!(
                got,
                inner.blob.slice(start as usize..(start + 100) as usize)
            );
        }
        assert_eq!(inner.calls(), 2, "the GET covered every nearby page");
        let _ = src.range(10, 100).await.expect("past the window");
        assert_eq!(inner.calls(), 3, "and nothing past it");

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// The first read on a file, and a miss at or past where a recent read
    /// ended, put the whole window ahead: the reads are moving forwards.
    #[tokio::test]
    async fn a_forward_miss_puts_the_window_ahead() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(dir.path(), &store, 4 * GROUP as usize, None);

        let _ = src.range(4 * b + 10, 100).await.expect("first read");
        store.block_writes_settled().await;
        assert!(
            src.all_filled(4, 11),
            "the first read's window starts at it"
        );
        assert!(!src.all_filled(3, 3), "nothing behind it");
        let _ = src
            .range(14 * b + 10, 100)
            .await
            .expect("a read further on");
        store.block_writes_settled().await;
        assert!(src.all_filled(14, 21), "a later miss ahead, the same");
        assert!(!src.all_filled(12, 13), "nothing behind it either");
        assert_eq!(inner.fetched_bytes(), 2 * MIN_GET_BYTES);

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A GET stops at a filled stretch longer than `FETCH_THROUGH_BLOCKS`,
    /// rather than downloading it again.
    #[tokio::test]
    async fn a_get_stops_at_a_long_filled_stretch() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(dir.path(), &store, 4 * GROUP as usize, None);

        let _ = src
            .range(5 * b + 10, 100)
            .await
            .expect("fetch blocks 5 to 12");
        store.block_writes_settled().await;
        let before = inner.fetched_bytes();
        let _ = src.range(10, 100).await.expect("read block 0");
        assert_eq!(inner.fetched_bytes() - before, 5 * b, "blocks 0 to 4 only");

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A GET also stops at a long pending stretch, blocks fetched but not
    /// written yet.
    #[tokio::test]
    async fn a_get_stops_at_a_long_pending_stretch() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(dir.path(), &store, 4 * GROUP as usize, None);

        let stall = src.stall_writes();
        let _ = src
            .range(5 * b + 10, 100)
            .await
            .expect("fetch blocks 5 to 12");
        let before = inner.fetched_bytes();
        let _ = src.range(10, 100).await.expect("read block 0");
        assert_eq!(inner.fetched_bytes() - before, 5 * b, "blocks 0 to 4 only");

        drop(stall);
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A forward read near the end of the file stops at its last, partial block,
    /// and is charged for the bytes it holds. A miss behind it then takes the
    /// room it cannot use ahead from behind.
    #[tokio::test]
    async fn a_short_run_extends_to_the_partial_last_block() {
        let b = CACHE_BLOCK_BYTES;
        let size = 3 * b + 100;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(dir.path(), &store, size as usize, None);

        let _ = src.range(2 * b + 10, 100).await.expect("read block 2");
        assert_eq!(
            inner.fetched_bytes(),
            b + 100,
            "block 2 to the end, nothing behind"
        );
        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), b + 100);
        assert_eq!(store.stats().current_bytes, b + 100);
        let _ = src.range(b + 10, 100).await.expect("read block 1");
        assert_eq!(inner.fetched_bytes(), size, "then blocks 0 and 1");
        store.block_writes_settled().await;
        assert_eq!(store.stats().current_bytes, size);

        store.remove_block_entry_for_test(&uri);
    }

    /// Two short reads at once whose GETs overlap are both served, and once
    /// written the blocks they share are charged once.
    #[tokio::test]
    async fn overlapping_gets_at_once_charge_their_blocks_once() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = Arc::new(GatedSource {
            inner: CountingSource::new(4 * GROUP as usize),
            gate: Barrier::new(2),
        });
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let (one, two) = tokio::join!(src.range(6 * b + 10, 64), src.range(9 * b + 10, 64));
        let at = |start: u64| {
            inner
                .inner
                .blob
                .slice(start as usize..(start + 64) as usize)
        };
        assert_eq!(one.expect("first read"), at(6 * b + 10));
        assert_eq!(two.expect("second read"), at(9 * b + 10));
        assert_eq!(inner.inner.calls(), 2, "both missed, so both fetched");

        store.block_writes_settled().await;
        let union = (6..=16).count() as u64 * b;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), union);
        assert_eq!(
            store.stats().current_bytes,
            union,
            "the overlap charged once"
        );
        assert_eq!(store.write_behind_in_use(), 0, "the cap came back");
        store.remove_block_entry_for_test(&uri);
    }

    /// Inside the aligned region a GET covers whole aligned groups, so a later
    /// read anywhere in the group costs no GET.
    #[tokio::test]
    async fn a_miss_in_the_aligned_region_gets_whole_groups() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(
            dir.path(),
            &store,
            4 * GROUP as usize,
            SourceLayout::ALL_ALIGNED.aligned_region,
        );

        let mid = GROUP + 3 * CACHE_BLOCK_BYTES + 10;
        let got = src.range(mid, 100).await.expect("read");
        assert_eq!(got, inner.blob.slice(mid as usize..(mid + 100) as usize));
        assert_eq!(inner.fetched_bytes(), GROUP, "the whole second group");

        for start in [GROUP + 10, 2 * GROUP - 200] {
            let got = src.range(start, 100).await.expect("later read");
            assert_eq!(
                got,
                inner.blob.slice(start as usize..(start + 100) as usize)
            );
        }
        assert_eq!(
            inner.calls(),
            1,
            "served from the group, before and after the first read"
        );

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// Only the aligned region rounds to aligned groups: a small read at the end
    /// of a group gets that whole group inside the region, and outside it gets
    /// a 512 KiB window from its own block.
    #[tokio::test]
    async fn only_the_aligned_region_gets_whole_groups() {
        let per_group = (GROUP / CACHE_BLOCK_BYTES) as u32;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, _inner, src) = layout_source(
            dir.path(),
            &store,
            4 * GROUP as usize,
            Some((2 * GROUP, GROUP)),
        );

        let _ = src.range(GROUP - 100, 50).await.expect("read outside");
        let _ = src.range(3 * GROUP - 100, 50).await.expect("read inside");
        store.block_writes_settled().await;
        let last_of = |group: u32| (group + 1) * per_group - 1;
        assert!(
            src.all_filled(last_of(0), last_of(0) + MIN_GET_BLOCKS - 1),
            "a window from its block, not its aligned group"
        );
        assert!(!src.all_filled(0, 0), "not the rest of its group");
        assert!(
            src.all_filled(2 * per_group, last_of(2)),
            "the whole third group"
        );
        assert!(
            !src.all_filled(last_of(2) + 1, last_of(2) + 1),
            "and no further"
        );
        store.remove_block_entry_for_test(&uri);
    }

    /// A group straddling the aligned region's start is fetched whole, even for
    /// a run that ends right before the region, so the region's part of it is
    /// then local.
    #[tokio::test]
    async fn a_group_straddling_the_region_start_is_fetched_whole() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(
            dir.path(),
            &store,
            4 * GROUP as usize,
            Some((GROUP + 4 * b, GROUP)),
        );

        let _ = src
            .range(0, GROUP + 4 * b)
            .await
            .expect("read up to the region");
        assert_eq!(inner.fetched_bytes(), 2 * GROUP, "up to the group's end");
        let _ = src
            .range(GROUP + 5 * b, 100)
            .await
            .expect("read the region");
        assert_eq!(inner.calls(), 1, "the region's part of the group is local");

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A short run's window stops before the aligned region, whose groups are
    /// only ever fetched whole.
    #[tokio::test]
    async fn a_short_run_stops_before_the_aligned_region() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(
            dir.path(),
            &store,
            4 * GROUP as usize,
            Some((2 * GROUP, GROUP)),
        );

        let _ = src
            .range(2 * GROUP - 2 * b + 10, 100)
            .await
            .expect("read before");
        assert_eq!(inner.fetched_bytes(), 2 * b, "up to the region");
        store.block_writes_settled().await;
        let region_first = (2 * GROUP / b) as u32;
        assert!(
            !src.all_filled(region_first, region_first),
            "nothing in the region"
        );
        let _ = src
            .range(2 * GROUP + 10, 100)
            .await
            .expect("read the region");
        assert_eq!(inner.fetched_bytes(), 2 * b + GROUP, "then a whole group");

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// When the aligned region starts inside a group, a short read before it
    /// fetches up to the region's own first block, the head of that group, so
    /// the FTS postings that sit there are local. A read in the region later
    /// fetches the rest of the group, not those blocks again.
    #[tokio::test]
    async fn a_window_reaches_the_head_of_the_regions_first_group() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = Arc::new(CountingSource::new(4 * GROUP as usize));
        let uri = SuperfileUri::new_v4();
        let region = (2 * GROUP + 3 * b + 100, GROUP);
        let src = BlockCachedSource::new_with_accounting(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("head.blocks"),
            true,
            SourceLayout {
                fts_hole: Some((13 * b + 500, region.0 - (13 * b + 500))),
                aligned_region: Some(region),
            },
        );
        store.install_block_entry_for_test(uri, Arc::clone(&src));

        let _ = src
            .range(13 * b + 10, 100)
            .await
            .expect("read before the hole");
        assert_eq!(inner.fetched_bytes(), 6 * b, "blocks 13 to 18");
        store.block_writes_settled().await;
        let _ = src
            .range(17 * b + 10, 600)
            .await
            .expect("a posting read in the group's head");
        assert_eq!(inner.calls(), 1, "local");
        let _ = src
            .range(region.0 + 10, 100)
            .await
            .expect("read the region");
        assert_eq!(
            inner.fetched_bytes(),
            6 * b + 5 * b,
            "the group's other five blocks"
        );

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A short read just before the FTS hole covers the hole's first blocks, and
    /// later reads there are served from them; past them a hole read is an exact
    /// GET.
    #[tokio::test]
    async fn a_short_read_before_the_hole_serves_its_first_bytes() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) =
            holed_source(dir.path(), &store, 4 * GROUP as usize, (GROUP, 2 * GROUP));

        let _ = src
            .range(GROUP - b + 10, 100)
            .await
            .expect("read before the hole");
        store.block_writes_settled().await;
        let read_block = (GROUP / b) as u32 - 1;
        assert!(
            src.all_filled(read_block, read_block + MIN_GET_BLOCKS - 1),
            "all of the window ahead, into the hole"
        );
        assert!(
            !src.all_filled(read_block - 1, read_block - 1),
            "nothing behind"
        );
        let calls = inner.calls();
        let start = GROUP + 10;
        let got = src.range(start, 100).await.expect("read the hole's start");
        assert_eq!(
            got,
            inner.blob.slice(start as usize..(start + 100) as usize)
        );
        assert_eq!(inner.calls(), calls, "served from its blocks");

        let before = inner.fetched_bytes();
        let _ = src
            .range(2 * GROUP + 10, 100)
            .await
            .expect("read past them");
        assert_eq!(inner.calls(), calls + 1);
        assert_eq!(inner.fetched_bytes() - before, 100, "an exact GET");

        store.remove_block_entry_for_test(&uri);
    }

    /// A read over an aligned group with a filled block in the middle fetches the
    /// group once, not once per missing stretch.
    #[tokio::test]
    async fn a_group_with_a_filled_block_inside_is_fetched_once() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(
            dir.path(),
            &store,
            4 * GROUP as usize,
            SourceLayout::ALL_ALIGNED.aligned_region,
        );
        let bf = src.block_file().expect("block file");
        bf.file
            .write_all_at(&inner.blob[3 * b as usize..4 * b as usize], 3 * b)
            .expect("write block 3");
        src.mark_filled(bf.size, 3, 3);

        let got = src.range(0, GROUP).await.expect("read the group");
        assert_eq!(got, inner.blob.slice(0..GROUP as usize));
        assert_eq!(inner.calls(), 1, "one GET for the group");
        assert_eq!(inner.fetched_bytes(), GROUP);

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A prefetch cuts its chunks on a grid from file offset 0, so one that runs
    /// out of budget leaves no aligned group half filled.
    #[tokio::test]
    async fn a_prefetch_stopping_early_leaves_no_group_half_filled() {
        let b = CACHE_BLOCK_BYTES;
        let per_group = ALIGNED_GET_BLOCKS;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), 2 * GROUP);
        let (uri, _inner, src) = layout_source(
            dir.path(),
            &store,
            4 * GROUP as usize,
            Some((2 * GROUP, GROUP)),
        );

        src.prefetch(3 * b, 3 * GROUP - 3 * b, GROUP, 1)
            .await
            .expect("prefetch");
        let region = 2 * per_group..=3 * per_group - 1;
        let filled = region.clone().filter(|&b| src.all_filled(b, b)).count();
        assert!(
            filled == 0 || filled == region.count(),
            "the region's group is filled whole or not at all ({filled} blocks)"
        );

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// Random reads, the shape of point lookups landing on one superfile over
    /// time, cost at most 1.75x one GET per 512 KiB window of the file they
    /// touch. A random read past any recent one looks like a forward scan, so its
    /// window goes ahead only; that is the price of scans, partitioned ones
    /// included, not fetching the columns they skip.
    #[tokio::test]
    async fn random_reads_cost_at_most_1_75x_whole_windows() {
        const READS: u64 = 1000;
        const READ_LEN: u64 = 1000;
        let size = 64 * GROUP;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(dir.path(), &store, size as usize, None);

        // A fixed xorshift sequence, so the test reads the same offsets every run.
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut windows = BTreeSet::new();
        for _ in 0..READS {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let start = x % (size - READ_LEN);
            windows.insert(start / MIN_GET_BYTES);
            windows.insert((start + READ_LEN - 1) / MIN_GET_BYTES);
            let _ = src.range(start, READ_LEN).await.expect("read");
        }
        let whole_windows = windows.len();
        assert!(
            inner.calls() * 4 <= whole_windows * 7,
            "{} GETs for reads touching {whole_windows} windows",
            inner.calls()
        );

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// Reads walking backwards find their neighbours local: after the first
    /// read, whose window goes ahead, each miss is behind every read before it,
    /// so its window reaches behind it, up to the blocks already cached ahead.
    #[tokio::test]
    async fn reads_walking_backwards_cost_a_get_per_half_window() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(dir.path(), &store, 4 * GROUP as usize, None);

        for block in (8..16).rev() {
            let start = block * b + 10;
            let got = src.range(start, 1000).await.expect("read");
            assert_eq!(
                got,
                inner.blob.slice(start as usize..(start + 1000) as usize)
            );
        }
        assert_eq!(
            inner.calls(),
            3,
            "eight reads in reverse: the first read's GET, then two"
        );

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// One sparse read, then a forward scan over the same area from before it:
    /// the scan's first read reaches back over the blocks before it, and the
    /// scan then costs one GET per window.
    #[tokio::test]
    async fn a_sparse_read_then_a_forward_scan_costs_one_get_per_window() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(dir.path(), &store, 4 * GROUP as usize, None);

        let _ = src.range(GROUP + 3 * b, 100).await.expect("sparse read");
        store.block_writes_settled().await;
        for block in 8..24 {
            let _ = src.range(block * b + 10, 1000).await.expect("scan");
        }
        assert_eq!(
            inner.calls(),
            3,
            "the sparse read's window, the blocks before it, then one more"
        );

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A forward scan reading two columns of each row group and skipping the one
    /// between them never fetches the skipped column behind a read: each GET goes
    /// ahead, over the next row group's first column.
    #[tokio::test]
    async fn a_scan_skipping_a_column_fetches_ahead_only() {
        let b = CACHE_BLOCK_BYTES;
        // Each row group: column a (3 blocks), b (16 blocks, never read), c (3).
        let (a, skipped, c) = (3, 16, 3);
        let row_group = a + skipped + c;
        let row_groups = 4;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let size = (row_groups * row_group + 8) * b;
        let (uri, inner, src) = layout_source(dir.path(), &store, size as usize, None);

        for k in 0..row_groups {
            let base = k * row_group * b;
            let _ = src.range(base, a * b).await.expect("column a");
            let _ = src
                .range(base + (a + skipped) * b, c * b)
                .await
                .expect("column c");
        }
        assert_eq!(
            inner.calls() as u64,
            1 + row_groups,
            "the first column a, then one per column c"
        );
        assert_eq!(inner.fetched_bytes(), (1 + row_groups) * MIN_GET_BYTES);

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A short cached stretch inside the window is fetched through, so the
    /// missing blocks after it come in the same GET.
    #[tokio::test]
    async fn a_window_is_fetched_through_a_short_cached_stretch() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(dir.path(), &store, 4 * GROUP as usize, None);
        let bf = src.block_file().expect("block file");
        bf.file
            .write_all_at(&inner.blob[3 * b as usize..5 * b as usize], 3 * b)
            .expect("write blocks 3 and 4");
        src.mark_filled(bf.size, 3, 4);

        let _ = src.range(10, 100).await.expect("read block 0");
        assert_eq!(
            inner.fetched_bytes(),
            MIN_GET_BYTES,
            "blocks 0 to 7 in one GET"
        );
        let _ = src
            .range(6 * b + 10, 100)
            .await
            .expect("read past the stretch");
        assert_eq!(inner.calls(), 1, "already local");

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A cached stretch longer than `FETCH_THROUGH_BLOCKS` ends the window
    /// instead of being fetched through, so a GET never re-downloads much of what
    /// is already local.
    #[tokio::test]
    async fn a_long_cached_stretch_ends_the_window() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(dir.path(), &store, 4 * GROUP as usize, None);
        let bf = src.block_file().expect("block file");
        bf.file
            .write_all_at(&inner.blob[b as usize..7 * b as usize], b)
            .expect("write blocks 1 to 6");
        src.mark_filled(bf.size, 1, 6);

        let _ = src.range(10, 100).await.expect("read block 0");
        assert_eq!(
            inner.fetched_bytes(),
            b,
            "block 0 only, not through six cached"
        );

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// Two partitions scanning their own halves of one file, their reads
    /// interleaved, each still fetch ahead only: a read past where its own
    /// partition last read is forward, wherever the other partition is.
    #[tokio::test]
    async fn interleaved_partitions_skipping_a_column_fetch_ahead_only() {
        let b = CACHE_BLOCK_BYTES;
        // Each row group: column a (3 blocks), b (16 blocks, never read), c (3).
        let (a, skipped, c) = (3, 16, 3);
        let row_group = a + skipped + c;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let size = (4 * row_group + 8) * b;
        let (uri, inner, src) = layout_source(dir.path(), &store, size as usize, None);

        // Partition 1 scans row groups 0 and 1, partition 2 row groups 2 and 3.
        let reads = |k: u64| {
            [
                (k * row_group * b, a * b),
                ((k * row_group + a + skipped) * b, c * b),
            ]
        };
        for (first, second) in [(0, 2), (1, 3)] {
            for (one, two) in reads(first).into_iter().zip(reads(second)) {
                let _ = src.range(one.0, one.1).await.expect("partition 1");
                let _ = src.range(two.0, two.1).await.expect("partition 2");
            }
        }
        assert_eq!(
            inner.calls(),
            6,
            "each partition's first column a, then one per column c"
        );
        assert_eq!(
            inner.fetched_bytes(),
            5 * MIN_GET_BYTES + 3 * b,
            "nothing behind any read: partition 1's second column c stops at partition 2's window"
        );

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// Near the end of the file a miss behind the reads before it shifts its window
    /// back by the room it cannot use ahead, so its GET still covers
    /// `MIN_GET_BYTES` less what is cached.
    #[tokio::test]
    async fn at_the_end_of_the_file_the_window_shifts_back() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(dir.path(), &store, 2 * GROUP as usize, None);

        let _ = src
            .range(2 * GROUP - b + 10, 100)
            .await
            .expect("read the last block");
        assert_eq!(
            inner.fetched_bytes(),
            b,
            "a forward read: just the last block"
        );
        store.block_writes_settled().await;
        let _ = src
            .range(2 * GROUP - 3 * b + 10, 100)
            .await
            .expect("read two blocks back");
        assert_eq!(
            inner.fetched_bytes(),
            b + 7 * b,
            "a window behind, ending at the cached last block"
        );
        let _ = src.range(GROUP + 10, 100).await.expect("read its start");
        assert_eq!(inner.calls(), 2, "already local");

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A missing stretch that starts inside the run this read just fetched and
    /// ends past it gets only its own remainder: nothing is fetched twice, and
    /// both runs are cached.
    #[tokio::test]
    async fn a_stretch_partly_inside_the_last_run_fetches_only_the_rest() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(
            dir.path(),
            &store,
            4 * GROUP as usize,
            SourceLayout::ALL_ALIGNED.aligned_region,
        );
        let bf = src.block_file().expect("block file");
        bf.file
            .write_all_at(&inner.blob[3 * b as usize..4 * b as usize], 3 * b)
            .expect("write block 3");
        src.mark_filled(bf.size, 3, 3);

        let got = src.range(0, 2 * GROUP).await.expect("read two groups");
        assert_eq!(got, inner.blob.slice(0..2 * GROUP as usize));
        assert_eq!(inner.calls(), 2, "one GET per group");
        assert_eq!(inner.fetched_bytes(), 2 * GROUP, "no group fetched twice");
        store.block_writes_settled().await;
        assert!(
            src.all_filled(0, 2 * ALIGNED_GET_BLOCKS - 1),
            "both groups cached"
        );

        store.remove_block_entry_for_test(&uri);
    }

    /// Away from the aligned region a prefetch cuts chunks from the range's
    /// start, so a range no bigger than a chunk is one GET wherever it lies.
    #[tokio::test]
    async fn a_prefetch_no_bigger_than_a_chunk_is_one_get() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = holed_source(
            dir.path(),
            &store,
            8 * GROUP as usize,
            (3 * GROUP, 2 * GROUP),
        );

        src.prefetch(3 * GROUP, 2 * GROUP, 4 * GROUP, 2)
            .await
            .expect("prefetch");
        assert_eq!(
            inner.calls(),
            1,
            "a range across a chunk-size line, one GET"
        );

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// Near the aligned region a prefetch's chunks start at the group holding the
    /// range's first block, so an FTS range ending in the region's first group
    /// takes as many GETs as with 512 KiB blocks.
    #[tokio::test]
    async fn a_prefetch_near_the_region_takes_the_chunks_512_kib_blocks_did() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(
            dir.path(),
            &store,
            12 * GROUP as usize,
            Some((66 * b + 100, 2 * GROUP)),
        );

        // Blocks 60 to 66: across the 64-block line a grid from offset 0 would
        // cut at, inside one 4 MiB chunk from block 56.
        src.prefetch(60 * b, 6 * b + 200, 8 * GROUP, 2)
            .await
            .expect("prefetch");
        assert_eq!(inner.calls(), 1, "one chunk, as with 512 KiB blocks");

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// Just before the aligned region a miss behind the reads before it shifts
    /// its window back by the room the region leaves ahead.
    #[tokio::test]
    async fn before_the_aligned_region_the_window_shifts_back() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(
            dir.path(),
            &store,
            4 * GROUP as usize,
            Some((2 * GROUP, GROUP)),
        );

        let _ = src
            .range(2 * GROUP + 10, 100)
            .await
            .expect("a read in the region");
        let before = inner.fetched_bytes();
        let _ = src
            .range(2 * GROUP - 2 * b + 10, 100)
            .await
            .expect("a read two blocks before the region");
        assert_eq!(
            inner.fetched_bytes() - before,
            MIN_GET_BYTES,
            "a whole window, shifted back from the region"
        );
        store.block_writes_settled().await;
        let region_first = (2 * GROUP / b) as u32;
        assert!(src.all_filled(region_first - MIN_GET_BLOCKS, region_first - 1));

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A read that runs into the FTS hole puts its whole window ahead, over the
    /// term dictionary, even when it is behind the reads before it.
    #[tokio::test]
    async fn a_read_running_into_the_hole_puts_its_window_ahead() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) =
            holed_source(dir.path(), &store, 4 * GROUP as usize, (GROUP, 2 * GROUP));

        let _ = src
            .range(3 * GROUP + 10, 100)
            .await
            .expect("a read after the hole, so the next is behind it");
        let _ = src
            .range(GROUP - b + 10, 2 * b)
            .await
            .expect("read across the hole's start");
        store.block_writes_settled().await;
        let read_block = (GROUP / b) as u32 - 1;
        assert!(
            src.all_filled(read_block, read_block + MIN_GET_BLOCKS - 1),
            "all of the window ahead, into the hole"
        );
        assert!(
            !src.all_filled(read_block - 1, read_block - 1),
            "nothing behind"
        );
        assert_eq!(inner.calls(), 2);

        store.remove_block_entry_for_test(&uri);
    }

    /// An aligned group at the end of the file stops at the file's end.
    #[tokio::test]
    async fn an_aligned_group_stops_at_the_end_of_the_file() {
        let size = GROUP + 3 * CACHE_BLOCK_BYTES + 100;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(
            dir.path(),
            &store,
            size as usize,
            SourceLayout::ALL_ALIGNED.aligned_region,
        );

        let got = src.range(size - 50, 50).await.expect("read the tail");
        assert_eq!(got, inner.blob.slice((size - 50) as usize..));
        assert_eq!(
            inner.fetched_bytes(),
            size - GROUP,
            "the last, partial group"
        );

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// An index written with `block`-byte blocks, marking `blocks`.
    fn index_with_block_size(block: u64, blocks: &[u32]) -> Vec<u8> {
        let mut idx = block.to_le_bytes().to_vec();
        RoaringBitmap::from_iter(blocks.iter().copied())
            .serialize_into(&mut idx)
            .expect("serialize");
        idx
    }

    /// An index written with larger blocks converts to ours: each of its blocks
    /// is the run of ours at the same offsets, the last one clipped to the file.
    #[test]
    fn an_index_written_with_larger_blocks_converts() {
        let per = (LEGACY_BLOCK_BYTES / CACHE_BLOCK_BYTES) as u32;
        let large = LEGACY_BLOCK_BYTES;
        let size = 2 * large + 100;
        let (bitmap, filled) =
            indexed_filled_bytes(size, &index_with_block_size(large, &[1, 2])).expect("converts");
        let last = 2 * per;
        assert_eq!(bitmap, RoaringBitmap::from_iter(per..=last));
        assert_eq!(filled, size - large);
        assert!(
            indexed_filled_bytes(size, &index_with_block_size(large, &[3])).is_none(),
            "a block past the file"
        );
    }

    /// An index written with any block size but ours or the legacy one is not
    /// adopted, even a multiple of ours: a corrupt header must not claim bytes.
    #[test]
    fn an_index_written_with_another_block_size_is_rejected() {
        let size = 4 * GROUP;
        for block in [
            CACHE_BLOCK_BYTES / 2,
            CACHE_BLOCK_BYTES + 1,
            2 * CACHE_BLOCK_BYTES,
            2 * LEGACY_BLOCK_BYTES,
            0,
        ] {
            assert!(indexed_filled_bytes(size, &index_with_block_size(block, &[0])).is_none());
        }
    }

    /// A block file indexed with larger blocks by an older build is adopted with
    /// no refetch. Its index stays as the older build wrote it until this build
    /// fills a block, so an older process sharing the cache can still adopt it;
    /// the next fill rewrites it in our block size.
    #[tokio::test]
    async fn a_block_file_indexed_with_larger_blocks_is_adopted() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let path = dir.path().join("old.blocks");
        let size = 2 * GROUP;

        let inner = Arc::new(CountingSource::new(size as usize));
        let gen1 = BlockCachedSource::new(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path.clone(),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&gen1));
        let _ = gen1.range(0, size).await.expect("fill the file");
        store.block_writes_settled().await;
        let idx = gen1.idx_path();
        store.remove_block_entry_for_test(&uri);
        drop(gen1);
        fs::write(&idx, index_with_block_size(LEGACY_BLOCK_BYTES, &[1])).expect("an older index");

        let inner2 = Arc::new(CountingSource::new(size as usize));
        let gen2 = BlockCachedSource::new(
            Arc::clone(&inner2) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            path,
        );
        store.install_block_entry_for_test(uri, Arc::clone(&gen2));
        let start = LEGACY_BLOCK_BYTES + 10;
        let got = gen2.range(start, 100).await.expect("read the old block");
        assert_eq!(
            got,
            inner2.blob.slice(start as usize..(start + 100) as usize)
        );
        assert_eq!(inner2.calls(), 0, "adopted, no refetch");
        assert_eq!(store.stats().current_bytes, LEGACY_BLOCK_BYTES);

        store.remove_block_entry_for_test(&uri);
        drop(gen2);
        let kept = fs::read(&idx).expect("index");
        assert!(
            !kept.starts_with(&IDX_HEADER),
            "an open that fills nothing leaves the older index as it was"
        );

        let inner3 = Arc::new(CountingSource::new(size as usize));
        let gen3 = BlockCachedSource::new(
            Arc::clone(&inner3) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("old.blocks"),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&gen3));
        let _ = gen3.range(10, 100).await.expect("fill a block");
        assert_eq!(inner3.calls(), 1, "a block the older index did not hold");
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
        drop(gen3);
        let rewritten = fs::read(&idx).expect("index");
        assert!(
            rewritten.starts_with(&IDX_HEADER),
            "rewritten in our block size"
        );
        let (bitmap, _) = indexed_filled_bytes(size, &rewritten).expect("valid");
        assert_eq!(bitmap, RoaringBitmap::from_iter(0..2 * ALIGNED_GET_BLOCKS));
    }

    /// A source over `inner`, installed as current for a fresh uri.
    fn installed_source(
        dir: &Path,
        store: &Arc<DiskCacheStore>,
        inner: Arc<dyn LazyByteSource>,
    ) -> (SuperfileUri, Arc<BlockCachedSource>) {
        let uri = SuperfileUri::new_v4();
        let src = BlockCachedSource::new(inner, Arc::downgrade(store), uri, dir.join("wb.blocks"));
        store.install_block_entry_for_test(uri, Arc::clone(&src));
        (uri, src)
    }

    /// A [`CountingSource`] over an `obj`-byte blob, behind an installed source.
    fn counting_source(
        dir: &Path,
        store: &Arc<DiskCacheStore>,
        obj: usize,
    ) -> (SuperfileUri, Arc<CountingSource>, Arc<BlockCachedSource>) {
        let inner = Arc::new(CountingSource::new(obj));
        let (uri, src) =
            installed_source(dir, store, Arc::clone(&inner) as Arc<dyn LazyByteSource>);
        (uri, inner, src)
    }

    /// A read whose block write is held:
    ///  - returns right away, with the right bytes,
    ///  - repeat reads, async and sync, come from memory with no GET,
    ///  - no index lists the blocks yet.
    /// Once the write runs, the blocks are filled, reserved once, and indexed.
    #[tokio::test]
    async fn a_read_never_waits_for_its_block_write() {
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 4 * b as usize);

        let stall = src.stall_writes();
        let start = 100u64;
        let len = 3 * b - 200;
        let want = inner.blob.slice(start as usize..(start + len) as usize);
        let first = timeout(NO_WAIT_DEADLINE, src.range(start, len))
            .await
            .expect("the read does not wait for its write");
        assert_eq!(first.expect("first read"), want);
        assert_eq!(inner.calls(), 1);

        assert_eq!(src.range(start, len).await.expect("repeat read"), want);
        let sync = src
            .try_get_range_sync(start + 10, 100)
            .expect("sync read of a pending run");
        assert_eq!(sync, inner.blob.slice(110..210));
        assert_eq!(inner.calls(), 1, "pending blocks are served from memory");
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 0);
        assert!(
            !src.idx_path().exists(),
            "no index names an unwritten block"
        );
        assert_eq!(
            store.stats().current_bytes,
            3 * b,
            "the run is charged while it waits"
        );

        drop(stall);
        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 3 * b);
        assert_eq!(store.stats().current_bytes, 3 * b);
        assert_eq!(store.write_behind_in_use(), 0);
        let idx = fs::read(src.idx_path()).expect("index written after the write");
        let (bitmap, _) = indexed_filled_bytes(4 * b, &idx).expect("valid index");
        let per_unit = (b / CACHE_BLOCK_BYTES) as u32;
        assert_eq!(
            bitmap.iter().collect::<Vec<_>>(),
            (0..3 * per_unit).collect::<Vec<_>>()
        );
        assert_eq!(src.range(start, len).await.expect("read from disk"), want);
        assert_eq!(inner.calls(), 1);

        store.remove_block_entry_for_test(&uri);
    }

    /// A run that is never written, like one a shutting-down runtime drops,
    /// gives everything back: its budget, its share of the cap, its pending
    /// entry. No index lists its blocks.
    #[tokio::test]
    async fn a_run_dropped_unrun_gives_everything_back() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, _inner, src) = counting_source(dir.path(), &store, 2 * b as usize);
        let bf = src.block_file().expect("block file");

        let (_, run) = src
            .fetch_missing(bf, 0, 0, Fill::Read)
            .await
            .expect("GET")
            .expect("fetched");
        let run = run.expect("queued");
        assert_eq!(store.stats().current_bytes, b, "the run holds its charge");
        assert!(src.has_pending_writes());

        drop(run);
        assert_eq!(store.stats().current_bytes, 0);
        assert_eq!(store.write_behind_in_use(), 0);
        assert!(!src.has_pending_writes());
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 0);
        assert!(!src.idx_path().exists());
        store.remove_block_entry_for_test(&uri);
    }

    /// A sync read from a thread with no runtime (a rayon reader thread) runs on
    /// a throwaway runtime.
    ///  - the read does not wait for its write,
    ///  - the write still happens after that runtime is gone.
    #[tokio::test]
    async fn a_bridged_read_neither_waits_for_nor_loses_its_write() {
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 2 * b as usize);

        let stall = src.stall_writes();
        let (tx, rx) = mpsc::channel();
        let reader = Arc::clone(&src);
        thread::spawn(move || {
            let got = bridge_sync_to_async_send(async move { reader.range(100, 64).await });
            let _ = tx.send(got);
        });
        let got = rx
            .recv_timeout(NO_WAIT_DEADLINE)
            .expect("the bridged read does not wait for its write");
        assert_eq!(got.expect("bridged read"), inner.blob.slice(100..164));

        drop(stall);
        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), b);
        assert_eq!(store.stats().current_bytes, b);
        assert_eq!(
            store.write_behind_in_use(),
            0,
            "its share of the cap came back"
        );
        store.remove_block_entry_for_test(&uri);
    }

    /// A read that starts inside a pending run gets the right bytes from it,
    /// async and sync, with no GET.
    #[tokio::test]
    async fn a_read_inside_a_pending_run_slices_it_at_its_offset() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 3 * b as usize);

        let stall = src.stall_writes();
        let _ = src.range(0, 3 * b).await.expect("pend blocks 0..=2");
        let at = b + 5;
        let got = src.range(at, 100).await.expect("read inside the run");
        assert_eq!(got, inner.blob.slice(at as usize..at as usize + 100));
        let sync = src
            .try_get_range_sync(at, b)
            .expect("sync read inside the run");
        assert_eq!(sync, inner.blob.slice(at as usize..(at + b) as usize));
        assert_eq!(inner.calls(), 1);

        drop(stall);
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A read that spans filled, pending and missing blocks returns them in
    /// order, and fetches only the missing ones.
    #[tokio::test]
    async fn a_read_stitches_filled_pending_and_missing_blocks() {
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 6 * b as usize);

        // Unit 1 filled, then unit 3 pending behind a stalled writer.
        let _ = src.range(b, 10).await.expect("fill unit 1");
        store.block_writes_settled().await;
        let stall = src.stall_writes();
        let _ = src.range(3 * b, 10).await.expect("pend unit 3");
        assert_eq!(inner.calls(), 2);

        // Units 0..=4: three missing stretches between the filled and the
        // pending window, so three GETs.
        let start = 100;
        let end = 4 * b + 200;
        let got = src.range(start, end - start).await.expect("mixed read");
        assert_eq!(got, inner.blob.slice(start as usize..end as usize));
        assert_eq!(inner.calls(), 5);

        drop(stall);
        store.block_writes_settled().await;
        let (b0, b1) = BlockCachedSource::block_span(start, end - start);
        assert!(src.all_filled(b0, b1), "every block of the read is cached");
        assert_eq!(
            src.filled_bytes_handle().load(Ordering::Acquire),
            5 * b,
            "and charged once"
        );
        store.remove_block_entry_for_test(&uri);
    }

    /// Two reads miss the same run at once and both GET it. It is still cached
    /// once: reserved once, written once, and the cap is given back in full.
    #[tokio::test]
    async fn racing_reads_of_one_run_cache_it_once() {
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = Arc::new(GatedSource {
            inner: CountingSource::new(2 * b as usize),
            gate: Barrier::new(2),
        });
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let stall = src.stall_writes();
        let (one, two) = tokio::join!(src.range(0, 64), src.range(10, 64));
        assert_eq!(one.expect("first read"), inner.inner.blob.slice(0..64));
        assert_eq!(two.expect("second read"), inner.inner.blob.slice(10..74));
        assert_eq!(inner.inner.calls(), 2, "both missed, so both fetched");
        assert_eq!(store.write_behind_in_use(), b, "only one run is queued");
        assert_eq!(store.stats().current_bytes, b, "only one is charged");

        drop(stall);
        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), b);
        assert_eq!(store.stats().current_bytes, b, "charged once");
        assert_eq!(
            store.write_behind_in_use(),
            0,
            "its share of the cap came back"
        );
        store.remove_block_entry_for_test(&uri);
    }

    /// A GET that returns fewer bytes than asked (the object changed) is passed
    /// through but never cached, by a read or by a prefetch.
    #[tokio::test]
    async fn a_short_get_is_never_cached() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = Arc::new(ShortSource(CountingSource::new(2 * b as usize)));
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let got = src.range(0, 64).await.expect("short read passes through");
        assert_eq!(got, inner.0.blob.slice(0..63));
        src.prefetch(0, 2 * b, b, 1)
            .await
            .expect("a short prefetch stops quietly");

        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 0);
        assert_eq!(store.stats().current_bytes, 0, "nothing stays charged");
        assert_eq!(store.write_behind_in_use(), 0);
        assert!(!src.has_pending_writes());
        store.remove_block_entry_for_test(&uri);
    }

    /// A prefetch skips a run a read already fetched and has not written yet:
    /// no second GET, and its blocks are reserved once.
    #[tokio::test]
    async fn prefetch_skips_a_run_waiting_for_its_write() {
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 3 * b as usize);

        let stall = src.stall_writes();
        let _ = src.range(0, 10).await.expect("pend unit 0");
        // One unit per chunk, all three at once, so a prefetch that ignored
        // pending runs would fetch unit 0 again. Its own writes are held too,
        // so let them go once units 1 and 2 are fetched.
        let release = async {
            wait_until(|| inner.calls() >= 3, "the prefetch fetches units 1 and 2").await;
            drop(stall);
        };
        let (prefetched, ()) = tokio::join!(src.prefetch(0, 3 * b, b, 3), release);
        prefetched.expect("prefetch around the pending run");
        assert_eq!(inner.calls(), 3, "only units 1 and 2 are fetched again");

        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 3 * b);
        assert_eq!(store.stats().current_bytes, 3 * b);
        store.remove_block_entry_for_test(&uri);
    }

    /// With the store's cap on unwritten runs full, a miss still gets its bytes
    /// but is not cached, and its reserved budget is given back.
    #[tokio::test]
    async fn a_full_write_behind_cap_serves_uncached() {
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 2 * b as usize);
        let mut taken = 0;
        while store.try_admit_write_behind(b) {
            taken += b;
        }

        let want = inner.blob.slice(0..64);
        assert_eq!(src.range(0, 64).await.expect("read"), want);
        assert_eq!(src.range(0, 64).await.expect("read again"), want);
        assert_eq!(inner.calls(), 2, "nothing was cached, so both reads GET");
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 0);
        assert_eq!(
            store.stats().current_bytes,
            0,
            "the reservation is given back"
        );

        store.release_write_behind(taken);
        assert_eq!(src.range(0, 64).await.expect("read with room"), want);
        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), b);
        store.remove_block_entry_for_test(&uri);
    }

    /// However many runs wait, at most `BLOCK_WRITERS` writes run at once, so a
    /// slow disk can't take every blocking thread. The rest wait as tasks.
    #[tokio::test]
    async fn block_writes_run_at_most_block_writers_at_once() {
        let b = UNIT;
        let runs = BLOCK_WRITERS + 2;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, runs * b as usize);

        let stall = src.stall_writes();
        for run in 0..runs as u64 {
            let _ = src.range(run * b, 10).await.expect("pend one block");
        }
        assert_eq!(inner.calls(), runs);
        wait_until(
            || src.stalled_writes() == BLOCK_WRITERS,
            "every block writer is busy",
        )
        .await;
        sleep(STAYS_PUT).await;
        assert_eq!(src.stalled_writes(), BLOCK_WRITERS, "the others wait");
        assert_eq!(store.write_behind_in_use(), runs as u64 * b);

        drop(stall);
        store.block_writes_settled().await;
        assert_eq!(
            src.filled_bytes_handle().load(Ordering::Acquire),
            runs as u64 * b
        );
        store.remove_block_entry_for_test(&uri);
    }

    /// A full cap on unwritten runs doesn't stop a prefetch: it waits for its own
    /// writes, so every chunk is still filled.
    #[tokio::test]
    async fn prefetch_fills_every_chunk_with_the_cap_full() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 3 * b as usize);
        let mut taken = 0;
        while store.try_admit_write_behind(b) {
            taken += b;
        }

        src.prefetch(0, 3 * b, b, 1).await.expect("prefetch");
        assert_eq!(inner.calls(), 3);
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 3 * b);
        assert_eq!(store.write_behind_in_use(), taken, "its share came back");

        store.release_write_behind(taken);
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A prefetch that finds its run queued by a read while its own GET was in
    /// flight leaves the run to that read and goes on.
    #[tokio::test]
    async fn prefetch_goes_on_past_a_run_queued_meanwhile() {
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = Arc::new(HeldSource {
            inner: CountingSource::new(2 * b as usize),
            first: AtomicBool::new(true),
            held: AtomicBool::new(false),
            release: Notify::new(),
        });
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );
        let bf = src.block_file().expect("block file");

        let stall = src.stall_writes();
        let read_meanwhile = async {
            wait_until(|| inner.held.load(Ordering::Acquire), "the prefetch GETs").await;
            let _ = src.range(0, 10).await.expect("the read queues block 0");
            inner.release.notify_one();
        };
        let (chunk, ()) = timeout(NO_WAIT_DEADLINE, async {
            tokio::join!(src.prefetch_chunk(bf, 0, 0), read_meanwhile)
        })
        .await
        .expect("the prefetch doesn't wait for the read's write");
        assert!(chunk.expect("prefetch chunk"), "the prefetch goes on");
        assert_eq!(
            store.write_behind_in_use(),
            b,
            "only the read's run is queued"
        );

        drop(stall);
        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), b);
        assert_eq!(store.stats().current_bytes, b, "charged once");
        store.remove_block_entry_for_test(&uri);
    }

    /// Holds every `range` until two are in flight, so two reads both miss.
    struct GatedSource {
        inner: CountingSource,
        gate: Barrier,
    }

    #[async_trait]
    impl LazyByteSource for GatedSource {
        fn size(&self) -> u64 {
            self.inner.size()
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            timeout(NO_WAIT_DEADLINE, self.gate.wait())
                .await
                .expect("both reads reach the GET");
            self.inner.range(start, len).await
        }
    }

    /// Holds the first `range` until `release`, so another fill can queue the
    /// same run meanwhile. `held` is set while it waits.
    struct HeldSource {
        inner: CountingSource,
        first: AtomicBool,
        held: AtomicBool,
        release: Notify,
    }

    #[async_trait]
    impl LazyByteSource for HeldSource {
        fn size(&self) -> u64 {
            self.inner.size()
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            if self.first.swap(false, Ordering::AcqRel) {
                self.held.store(true, Ordering::Release);
                timeout(NO_WAIT_DEADLINE, self.release.notified())
                    .await
                    .expect("the held GET is let go");
            }
            self.inner.range(start, len).await
        }
    }

    /// Returns one byte less than asked, like an object replaced by a shorter one.
    struct ShortSource(CountingSource);

    #[async_trait]
    impl LazyByteSource for ShortSource {
        fn size(&self) -> u64 {
            self.0.size()
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            let bytes = self.0.range(start, len).await?;
            Ok(bytes.slice(..bytes.len().saturating_sub(1)))
        }
    }
}
