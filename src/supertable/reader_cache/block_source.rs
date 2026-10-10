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
//!          blocking thread: write ─► mark filled ─► out of pending ─► fsync + index
//! ```
//!
//! - A read looks in pending first, then in the filled blocks; anything in
//!   neither is fetched.
//! - The index on disk lists only blocks that were fsynced, so a crash loses at
//!   most the blocks not synced yet, and later reads fetch them again.
//! - Each store holds at most 256 MiB of fetched runs not written yet. Past
//!   that, a read still gets its bytes, but they are not cached.
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
    collections::{BTreeMap, btree_map::Entry},
    fs,
    os::unix::fs::FileExt,
    path::PathBuf,
    process,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::{StreamExt, stream};
use memmap2::Mmap;
use roaring::RoaringBitmap;
use tokio::task::JoinHandle;

#[cfg(test)]
use super::disk::BlockWriteTicket;
use super::disk::{ArcMmapOwner, DiskCacheStore, Reservation};
use crate::{
    runtime_bridge::shared_io_runtime,
    superfile::{LazyByteSource, LazyByteSourceError},
    supertable::manifest::SuperfileUri,
};

/// Cache block size. Misses fetch block-aligned runs, so this bounds both
/// the read amplification of a small scan (a request pays at most one
/// leading + one trailing partial block of overhead) and the bitmap size
/// (a 32 MiB cell superfile is 64 blocks; a 27 GiB one at 1B scale is
/// ~55K). Post-drain vector queries read ~0.25–2 MiB scan ranges, so
/// 512 KiB keeps first-touch overshoot well under 2× while still
/// coalescing a multi-MiB scan into a handful of blocks.
pub(super) const CACHE_BLOCK_BYTES: u64 = 512 * 1024;

const IDX_HEADER: [u8; 8] = CACHE_BLOCK_BYTES.to_le_bytes();

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
    /// Virtual hole: `(offset, len)` of a subsection whose reads bypass the
    /// block cache and fetch exact ranges from the inner source. Set to the
    /// FTS subsection: posting reads are ~KiB-sized and scattered, so
    /// rounding each to a 512 KiB block over-fetches ~200× per read on a
    /// wide OR (the block size is tuned for vector's 0.25–2 MiB scans).
    /// Their warm locality comes from the background-fill mmap promotion,
    /// not from this cache. So a read inside the hole never fills blocks on a
    /// miss, but is served from disk when its blocks are there; only
    /// [`Self::prefetch`] fills them. Reads only partially overlapping the
    /// hole keep block semantics.
    passthrough: Option<(u64, u64)>,
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
    write_stall: (Mutex<bool>, Condvar),
}

/// Runs a block-file write or fsync on the shared I/O runtime's blocking
/// threads. Never on the caller's runtime: a sync read from a rayon thread runs
/// on a throwaway runtime, and dropping it would wait for the write or cancel it.
fn spawn_disk_io<R: Send + 'static>(io: impl FnOnce() -> R + Send + 'static) -> JoinHandle<R> {
    shared_io_runtime().spawn_blocking(io)
}

/// A fetched run of blocks waiting to be written to the block file, made by
/// [`BlockCachedSource::admit_run`]. While it exists it holds:
/// - the disk budget reserved for its blocks,
/// - its share of the store's cap on unwritten runs,
/// - its entry in `pending`, so reads get the bytes from memory.
///
/// [`Self::land`] writes it. Dropping it gives back whatever it still holds,
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
    /// Writes the run to the block file and marks its blocks filled; `false` if
    /// the write failed. Blocking: run it with [`spawn_disk_io`].
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
        Self::new_with_accounting(inner, store, uri, path, true, None)
    }

    /// Construct a sparse source whose owning entry has already reserved the
    /// complete object size. `passthrough` is the optional exact-read hole
    /// (see the field docs) — the FTS subsection on the cold-open path.
    pub(crate) fn new_pre_reserved(
        inner: Arc<dyn LazyByteSource>,
        store: Weak<DiskCacheStore>,
        uri: SuperfileUri,
        path: PathBuf,
        passthrough: Option<(u64, u64)>,
    ) -> Arc<Self> {
        Self::new_with_accounting(inner, store, uri, path, false, passthrough)
    }

    pub(crate) fn new_with_accounting(
        inner: Arc<dyn LazyByteSource>,
        store: Weak<DiskCacheStore>,
        uri: SuperfileUri,
        path: PathBuf,
        owns_accounting: bool,
        passthrough: Option<(u64, u64)>,
    ) -> Arc<Self> {
        Arc::new_cyclic(|me| Self {
            inner,
            store,
            uri,
            path,
            entry_token: Arc::new(()),
            owns_accounting,
            passthrough,
            state: OnceLock::new(),
            filled: Mutex::new(RoaringBitmap::new()),
            filled_bytes: Arc::new(AtomicU64::new(0)),
            me: me.clone(),
            pending: Mutex::new(BTreeMap::new()),
            dirty: AtomicBool::new(false),
            flushing: AtomicBool::new(false),
            #[cfg(test)]
            write_stall: (Mutex::new(false), Condvar::new()),
        })
    }

    /// Whether `[start, start + len)` lies fully inside the passthrough hole.
    fn in_passthrough(&self, start: u64, len: u64) -> bool {
        match self.passthrough {
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
    /// also fills the passthrough hole, so a scan of the whole hole is served
    /// from disk. Uses only free space, never evicting, and stops quietly when
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
        let per_chunk = (chunk_bytes / CACHE_BLOCK_BYTES).max(1) as u32;
        let chunks = (b0..=b1)
            .step_by(per_chunk as usize)
            .map(|c0| (c0, c0.saturating_add(per_chunk - 1).min(b1)));
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
        let filled = self.filled.lock().expect("filled bitmap mutex poisoned");
        (b0..=b1).all(|b| filled.contains(b))
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
        let mut newly = 0u64;
        for b in b0..=b1 {
            if filled.insert(b) {
                newly += Self::block_len(size, b);
            }
        }
        if newly > 0 {
            self.dirty.store(true, Ordering::SeqCst);
        }
        newly
    }

    /// Fsyncs the block file, then writes the index of the blocks marked filled.
    /// Blocking: run it with [`spawn_disk_io`] or outside async code.
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
            // A small read gets a copy: a caller can keep it as long as its reader
            // lives (a vector reader keeps its headers), and a slice would keep
            // the whole fetched run in memory.
            return Some(if len < CACHE_BLOCK_BYTES {
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

        let mut segments = self.segments(b0, b1);

        for segment in &mut segments {
            let Segment::Missing { first, last } = *segment else {
                continue;
            };

            let Some(store) = self.store_if_current() else {
                return Ok(None);
            };

            let Some((bytes, reservation)) =
                self.fetch_run(&store, bf.size, first, last, true).await?
            else {
                return Ok(None);
            };

            if let Some(run) = self.admit_run(&store, first, last, &bytes, reservation) {
                // Don't wait for the write: the query already has its bytes.
                drop(spawn_disk_io(move || run.land()));
            }

            *segment = Segment::Pending {
                first,
                last,
                run_first: first,
                bytes,
            };
        }

        Ok(self.assemble(bf, start, len, &segments))
    }

    /// Reserves budget for blocks `[first, last]` and fetches them with one GET.
    /// `None` if they can't be cached:
    /// - no budget left (`may_evict` lets the reservation evict colder entries),
    /// - the GET returned another length, so the object is not the one this
    ///   reader opened.
    /// If the caller is dropped mid-GET, the reservation gives its bytes back.
    async fn fetch_run<'s>(
        &self,
        store: &'s DiskCacheStore,
        size: u64,
        first: u32,
        last: u32,
        may_evict: bool,
    ) -> Result<Option<(Bytes, Option<Reservation<'s>>)>, LazyByteSourceError> {
        let (run_start, run_len) = Self::run_bytes(size, first, last);

        let reservation = if !self.owns_accounting {
            None
        } else if may_evict {
            let Ok(reserved) = store.reserve(run_len).await else {
                return Ok(None);
            };
            Some(reserved)
        } else {
            let Some(reserved) = store.try_reserve(run_len) else {
                return Ok(None);
            };
            Some(reserved)
        };

        let bytes = self.inner.range(run_start, run_len).await?;

        Ok((bytes.len() as u64 == run_len).then_some((bytes, reservation)))
    }

    /// Queues fetched run `[first, last]` for writing: takes its share of the
    /// store's cap on unwritten runs, puts it in `pending`, and hands it the
    /// reserved budget. `None` (the reservation is given back) if the cap is
    /// full or a run starting at the same block is already waiting.
    fn admit_run(
        &self,
        store: &Arc<DiskCacheStore>,
        first: u32,
        last: u32,
        bytes: &Bytes,
        reservation: Option<Reservation<'_>>,
    ) -> Option<FetchedRun> {
        let source = self.me.upgrade()?;

        let run_len = bytes.len() as u64;
        if !store.try_admit_write_behind(run_len) {
            return None;
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
        *self.write_stall.0.lock().expect("write stall poisoned") = true;
        WriteStall(self)
    }

    #[cfg(test)]
    fn wait_while_stalled(&self) {
        let (stalled, resume) = &self.write_stall;
        let mut stalled = stalled.lock().expect("write stall poisoned");
        while *stalled {
            stalled = resume.wait(stalled).expect("write stall poisoned");
        }
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
    /// only free budget, so a prefetch never evicts what queries use. `false`
    /// stops the prefetch, when:
    /// - the budget or the cap on unwritten runs is full,
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

            let Some(store) = self.store_if_current() else {
                return Ok(false);
            };

            let Some((bytes, reservation)) =
                self.fetch_run(&store, bf.size, first, last, false).await?
            else {
                return Ok(false);
            };

            let Some(run) = self.admit_run(&store, first, last, &bytes, reservation) else {
                return Ok(false);
            };

            if !spawn_disk_io(move || run.land()).await.unwrap_or(false) {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

/// Holds a source's block-file writes; see [`BlockCachedSource::stall_writes`].
#[cfg(test)]
pub(crate) struct WriteStall<'a>(&'a BlockCachedSource);

#[cfg(test)]
impl Drop for WriteStall<'_> {
    fn drop(&mut self) {
        let (stalled, resume) = &self.0.write_stall;
        *stalled.lock().expect("write stall poisoned") = false;
        resume.notify_all();
    }
}

pub(super) fn serialize_index(bitmap: &RoaringBitmap) -> Vec<u8> {
    let mut buf = Vec::with_capacity(IDX_HEADER.len() + bitmap.serialized_size());
    buf.extend_from_slice(&IDX_HEADER);
    let _ = bitmap.serialize_into(&mut buf);
    buf
}

pub(super) fn indexed_filled_bytes(size: u64, idx_bytes: &[u8]) -> Option<(RoaringBitmap, u64)> {
    let body = idx_bytes.strip_prefix(&IDX_HEADER)?;
    let bitmap = RoaringBitmap::deserialize_from(body).ok()?;
    let n_blocks = size.div_ceil(CACHE_BLOCK_BYTES);
    if bitmap.max().is_some_and(|m| u64::from(m) >= n_blocks) {
        return None;
    }
    let filled = bitmap
        .iter()
        .map(|b| BlockCachedSource::block_len(size, b))
        .sum();
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
        if len == 0 {
            return Ok(Bytes::new());
        }
        let in_hole = self.in_passthrough(start, len);
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
    fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
        if len == 0 {
            return Some(Bytes::new());
        }
        let Some(bf) = self.block_file_for(self.in_passthrough(start, len)) else {
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

    async fn tail(&self, len: u64) -> Result<(Bytes, u64), LazyByteSourceError> {
        // Pass through: `tail` both fetches and (on unknown-size sources)
        // discovers the object size, which the inner source caches. Tail
        // bytes are open-time metadata already retained by the open-blob /
        // prefetch overlay above this source, so caching them here as
        // (mostly partial) blocks buys nothing.
        self.inner.tail(len).await
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::pending,
        path::Path,
        sync::{atomic::AtomicUsize, mpsc},
        thread,
        time::Duration,
    };

    use futures::FutureExt;
    use tempfile::tempdir;
    use tokio::sync::Barrier;

    use super::*;
    use crate::{
        runtime_bridge::bridge_sync_to_async_send,
        supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, LruPolicy},
    };

    /// How long a test waits for something that should happen right away, before
    /// failing instead of hanging.
    const NO_WAIT_DEADLINE: Duration = Duration::from_secs(10);

    /// In-memory fake source that counts `range` calls.
    struct CountingSource {
        blob: Bytes,
        calls: AtomicUsize,
    }

    impl CountingSource {
        fn new(n: usize) -> Self {
            let blob: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
            Self {
                blob: Bytes::from(blob),
                calls: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Acquire)
        }
    }

    #[async_trait]
    impl LazyByteSource for CountingSource {
        fn size(&self) -> u64 {
            self.blob.len() as u64
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            self.calls.fetch_add(1, Ordering::AcqRel);
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
        const OBJ: usize = 3 * CACHE_BLOCK_BYTES as usize + 1000;
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

        // Read spanning blocks 0..=2 (one contiguous missing run → 1 GET).
        let start = 100u64;
        let len = 2 * CACHE_BLOCK_BYTES + 500;
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

        // Accounting = 3 whole blocks (0..=2), not the object size.
        store.block_writes_settled().await;
        let expected = 3 * CACHE_BLOCK_BYTES;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), expected);
        assert_eq!(store.stats().current_bytes, expected);

        // Touch the trailing partial block: its length is size - 3*B.
        let tail_start = 3 * CACHE_BLOCK_BYTES + 10;
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
        let start = 100u64;
        let len = 2 * CACHE_BLOCK_BYTES + 500;
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

        let tail_start = 3 * CACHE_BLOCK_BYTES + 10;
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
        wait_for_file(&idx_path).await;

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

    /// Waits for `path` to appear, written by another source's flush.
    async fn wait_for_file(path: &Path) {
        tokio::time::timeout(NO_WAIT_DEADLINE, async {
            while !path.exists() {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("the file is written in time");
    }

    /// Two disjoint missing runs in one request → one GET per run.
    #[tokio::test]
    async fn disjoint_missing_runs_fetch_separately() {
        const OBJ: usize = 6 * CACHE_BLOCK_BYTES as usize;
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

        // Fill block 2 first.
        let b = CACHE_BLOCK_BYTES;
        let _ = src.range(2 * b, 10).await.expect("fill middle block");
        assert_eq!(inner.calls(), 1);

        // Read blocks 1..=3: blocks 1 and 3 are missing → two run GETs.
        let got = src.range(b, 3 * b).await.expect("spanning read");
        assert_eq!(got, inner.blob.slice(b as usize..(4 * b) as usize));
        assert_eq!(inner.calls(), 3, "two missing runs around the filled block");

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

    /// A source with a passthrough hole of `hole` over an `obj`-byte blob,
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
            Some(hole),
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
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 4 * b as usize);

        let stall = src.stall_writes();
        let start = 100u64;
        let len = 2 * b;
        let want = inner.blob.slice(start as usize..(start + len) as usize);
        let first = tokio::time::timeout(NO_WAIT_DEADLINE, src.range(start, len))
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
        assert_eq!(bitmap.iter().collect::<Vec<_>>(), vec![0, 1, 2]);
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
        let size = src.block_file().expect("block file").size;

        let (bytes, reservation) = src
            .fetch_run(&store, size, 0, 0, true)
            .await
            .expect("GET")
            .expect("fetched");
        let run = src
            .admit_run(&store, 0, 0, &bytes, reservation)
            .expect("admitted");
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
        let b = CACHE_BLOCK_BYTES;
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
        assert_eq!(store.write_behind_in_use(), 0, "the allowance came back");
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
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 6 * b as usize);

        // Block 1 filled, then block 3 pending behind a stalled writer.
        let _ = src.range(b, 10).await.expect("fill block 1");
        store.block_writes_settled().await;
        let stall = src.stall_writes();
        let _ = src.range(3 * b, 10).await.expect("pend block 3");
        assert_eq!(inner.calls(), 2);

        // Blocks 0..=4: 0, 2 and 4 are missing, so three GETs.
        let start = 100;
        let end = 4 * b + 200;
        let got = src.range(start, end - start).await.expect("mixed read");
        assert_eq!(got, inner.blob.slice(start as usize..end as usize));
        assert_eq!(inner.calls(), 5);

        drop(stall);
        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 5 * b);
        store.remove_block_entry_for_test(&uri);
    }

    /// Two reads miss the same run at once and both GET it. It is still cached
    /// once: reserved once, written once, and the cap is given back in full.
    #[tokio::test]
    async fn racing_reads_of_one_run_cache_it_once() {
        let b = CACHE_BLOCK_BYTES;
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

        let (one, two) = tokio::join!(src.range(0, 64), src.range(10, 64));
        assert_eq!(one.expect("first read"), inner.inner.blob.slice(0..64));
        assert_eq!(two.expect("second read"), inner.inner.blob.slice(10..74));
        assert_eq!(inner.inner.calls(), 2, "both missed, so both fetched");

        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), b);
        assert_eq!(store.stats().current_bytes, b, "charged once");
        assert_eq!(store.write_behind_in_use(), 0, "the allowance came back");
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
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 3 * b as usize);

        let stall = src.stall_writes();
        let _ = src.range(0, 10).await.expect("pend block 0");
        // One block per chunk, all three at once, so a prefetch that ignored
        // pending runs would fetch block 0 again. Its own writes are held too,
        // so let them go once blocks 1 and 2 are fetched.
        let release = async {
            tokio::time::timeout(NO_WAIT_DEADLINE, async {
                while inner.calls() < 3 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await
            .expect("the prefetch fetches blocks 1 and 2");
            drop(stall);
        };
        let (prefetched, ()) = tokio::join!(src.prefetch(0, 3 * b, b, 3), release);
        prefetched.expect("prefetch around the pending run");
        assert_eq!(inner.calls(), 3, "only blocks 1 and 2 are fetched again");

        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 3 * b);
        assert_eq!(store.stats().current_bytes, 3 * b);
        store.remove_block_entry_for_test(&uri);
    }

    /// With the store's cap on unwritten runs full, a miss still gets its bytes
    /// but is not cached, and its reserved budget is given back.
    #[tokio::test]
    async fn a_full_write_behind_allowance_serves_uncached() {
        let b = CACHE_BLOCK_BYTES;
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
            self.gate.wait().await;
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
