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
//! # A read never waits on the cache disk, and reads missing the same blocks share a GET
//!
//! A query gets its bytes as soon as the GET returns. Writing them to the block
//! file happens afterwards on a blocking thread, because a background download
//! can keep the same disk busy for minutes. Reads that miss the same blocks at
//! the same time share one GET:
//!
//! ```text
//!   read A ─► missing blocks ─► GET window registered in `runs` ─► GET ─► bytes ─► back to A
//!   read B ─► same blocks ─► finds A's window in `runs` ─► awaits the same GET ─┤
//!                                                                              ├─► in memory: later reads get them from here
//!                                                                              ▼
//!          wait for a block writer (a task, no thread)
//!                                                                              ▼
//!          blocking thread: write ─► mark filled ─► out of memory ─► fsync + index
//! ```
//!
//! - A run is in `runs` from the moment a read registers it, before its GET is
//!   sent, until its blocks are on the block file. A read takes each block from
//!   a fetched run, else the block file, else the run fetching it, and
//!   registers and fetches only the rest. A prefetch and the sync read path do
//!   not register their runs.
//! - The GET runs as a task on the shared I/O runtime, not on any read's: a
//!   sync read's runtime is a throwaway one, gone once that read returns. The
//!   read that registered it and every read that needs its blocks meanwhile
//!   await the same task, which is cancelled only when all of them are
//!   dropped.
//! - A read that has waited [`HEDGE_AFTER`] on another read's GET, longer for a
//!   big one, sends its own and keeps whichever lands first, so one stuck GET
//!   can't hold every read sharing it. A GET that ends with no bytes for a
//!   waiting read leaves it to fetch its blocks itself; one that fails fails
//!   every read sharing it.
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
    any::Any,
    array,
    collections::BTreeMap,
    fs,
    os::unix::fs::FileExt,
    panic::resume_unwind,
    path::PathBuf,
    pin::Pin,
    process,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::{
    StreamExt,
    future::{FutureExt, Shared, WeakShared, join_all},
    stream,
};
use memmap2::Mmap;
use roaring::RoaringBitmap;
use tokio::{
    select,
    task::{JoinHandle, spawn_blocking},
    time::timeout,
};
use tracing::Instrument;

#[cfg(test)]
use super::disk::BlockWriteTicket;
use super::disk::{ArcMmapOwner, DiskCacheStore, Reservation};
use crate::{
    runtime_bridge::shared_io_runtime,
    storage::StorageError,
    superfile::{CacheBlocks, CacheRoom, LazyByteSource, LazyByteSourceError},
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

/// How long a read waits on another read's GET before it GETs the same blocks
/// itself and keeps whichever lands first. Far above a normal GET (on GCS about
/// 45 ms at p50 and under 140 ms at p90), far below a stalled one (one attempt
/// can run to the store's 60 s request timeout): it costs nothing normally, and
/// bounds how long one stuck GET holds the reads sharing it. A big GET gets
/// longer: see [`STUCK_BYTES_PER_SEC`].
const HEDGE_AFTER: Duration = Duration::from_secs(2);

/// A GET slower than this is taken for stuck: a read waiting on another read's
/// GET hedges once it has waited [`HEDGE_AFTER`] plus the time the GET's bytes
/// take at this rate, so a big GET that is only slow is not fetched twice. One
/// GET on GCS moves tens of MB/s even with twenty others in flight.
const STUCK_BYTES_PER_SEC: u64 = 16 << 20;

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
    /// Runs in memory, keyed by first block; see [`MemoryRun`]. Never held
    /// across await. Dropping a fetch under it is fine: the GET's task, and
    /// with it the [`Claim`] that takes this lock, ends on the I/O runtime.
    runs: Mutex<BTreeMap<u32, MemoryRun>>,
    /// Source of [`MemoryRun::id`].
    next_run_id: AtomicU64,
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
/// - its entry in `runs`, so reads get the bytes from memory.
///
/// [`Self::spawn_write`] writes it. Dropping it gives back whatever it still holds,
/// whether the write worked, failed or never ran, and writes the index if the
/// blocks were written.
struct FetchedRun {
    source: Arc<BlockCachedSource>,
    store: Arc<DiskCacheStore>,
    /// Its entry in `runs`.
    id: u64,
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
        // Out of memory only now, after `land` marked the blocks filled, so a
        // read never finds them in neither place.
        self.source.remove_run(self.first, self.id);
        self.store.release_write_behind(self.bytes.len() as u64);

        if self.unlanded > 0 {
            self.store.release_block_bytes(self.unlanded);
        }

        if self.landed {
            self.source.flush_index();
        }
    }
}

/// A run's fetch, shared by every read that needs its blocks: the read that
/// registered it and every read that finds it in `runs` await clones of one
/// [`FetchTask`]. Dropping the last clone cancels the GET.
type RunFetch = Shared<FetchTask>;

/// A run's GET, spawned on the shared I/O runtime when a read registers the
/// run. Reads await it from their own runtimes, and a sync read's is a
/// throwaway one: a GET driven there would die with it while other reads still
/// wait on it. Dropping this cancels the GET.
struct FetchTask(JoinHandle<Fetched>);

impl Future for FetchTask {
    type Output = Fetched;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Fetched> {
        Pin::new(&mut self.0).poll(cx).map(|ended| match ended {
            Ok(fetched) => fetched,
            // A fetch that panicked panics its reads, as it would have in the
            // read that ran it, but each read raises it outside the `Shared`: a
            // panic inside it poisons the `Shared` and wakes nobody, so a read
            // that polled it meanwhile would wait forever. One cancelled (the
            // runtime shutting down) has nothing for them.
            Err(ended) => match ended.try_into_panic() {
                Ok(panic) => Fetched::Panicked(FetchPanic::new(panic)),
                Err(_) => Fetched::Elsewhere,
            },
        })
    }
}

impl Drop for FetchTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// How a shared run fetch ends.
#[derive(Clone)]
enum Fetched {
    /// The run's bytes, starting at its first block.
    Bytes(Bytes),
    /// Read elsewhere (the whole-file copy or object storage): the store is
    /// gone, the block file could not be created, this source is no longer the
    /// live entry, the budget is full, or the GET came back short.
    Elsewhere,
    /// The GET failed.
    Failed(FetchError),
    /// The fetch panicked.
    Panicked(FetchPanic),
}

/// A failed GET, shared by every read of the run: the first read to take it
/// gets the error itself and the others a copy (see [`copy_error`]), so each
/// fails at once rather than trying the GET again.
#[derive(Clone)]
struct FetchError(Arc<SharedError>);

struct SharedError {
    error: Mutex<Option<LazyByteSourceError>>,
    copy: LazyByteSourceError,
}

impl FetchError {
    fn new(error: LazyByteSourceError) -> Self {
        Self(Arc::new(SharedError {
            copy: copy_error(&error),
            error: Mutex::new(Some(error)),
        }))
    }

    fn take(&self) -> LazyByteSourceError {
        self.0
            .error
            .lock()
            .expect("fetch error mutex poisoned")
            .take()
            .unwrap_or_else(|| copy_error(&self.0.copy))
    }
}

/// A panicked fetch, shared by every read of the run: the first read to take it
/// raises the panic itself, the others a panic saying so.
#[derive(Clone)]
struct FetchPanic(Arc<Mutex<Option<Box<dyn Any + Send>>>>);

impl FetchPanic {
    fn new(panic: Box<dyn Any + Send>) -> Self {
        Self(Arc::new(Mutex::new(Some(panic))))
    }

    fn resume(&self) -> ! {
        let panic = self.0.lock().expect("fetch panic mutex poisoned").take();
        match panic {
            Some(panic) => resume_unwind(panic),
            None => panic!("a block fetch shared with another read panicked"),
        }
    }
}

/// A copy of `error` for the reads sharing a failed GET past the first: the
/// same variant and message, with a storage error's source kept as its text.
fn copy_error(error: &LazyByteSourceError) -> LazyByteSourceError {
    let storage = match error {
        LazyByteSourceError::Storage(storage) => storage,
        &LazyByteSourceError::OutOfBounds { start, len, size } => {
            return LazyByteSourceError::OutOfBounds { start, len, size };
        }
        &LazyByteSourceError::ShortRead {
            start,
            requested,
            got,
        } => {
            return LazyByteSourceError::ShortRead {
                start,
                requested,
                got,
            };
        }
    };
    LazyByteSourceError::Storage(match storage {
        StorageError::NotFound { uri } => StorageError::NotFound { uri: uri.clone() },
        StorageError::PreconditionFailed { uri } => {
            StorageError::PreconditionFailed { uri: uri.clone() }
        }
        StorageError::TransientExhausted { uri, source } => StorageError::TransientExhausted {
            uri: uri.clone(),
            source: source.to_string().into(),
        },
        StorageError::PermissionDenied { uri } => {
            StorageError::PermissionDenied { uri: uri.clone() }
        }
        StorageError::Permanent { uri, source } => StorageError::Permanent {
            uri: uri.clone(),
            source: source.to_string().into(),
        },
    })
}

/// A run of blocks in memory, in `runs` under its first block: from its
/// registration until its blocks are on the block file.
struct MemoryRun {
    /// Tells this run from a later one under the same first block, so a guard
    /// only ever removes its own.
    id: u64,
    /// Last block of the run (inclusive).
    last: u32,
    bytes: RunBytes,
}

impl MemoryRun {
    /// Whether this is a fetch every read dropped, its entry not taken out yet.
    fn dropped(&self) -> bool {
        matches!(&self.bytes, RunBytes::Fetching(fetch) if fetch.upgrade().is_none())
    }
}

/// Where a run's bytes are. `runs` holds a weak handle to a fetch, so it never
/// keeps a GET alive: once every read awaiting it is dropped, the GET is
/// cancelled and its [`Claim`] takes the entry out. A look at `runs` takes
/// strong ones, as `RunBytes<RunFetch>`.
enum RunBytes<F = WeakShared<FetchTask>> {
    /// Its GET is in flight.
    Fetching(F),
    /// Fetched, waiting for its block-file write: the run's bytes, starting at
    /// its first block.
    Fetched(Bytes),
}

/// The entry a read registered in `runs`, owned by the run's fetch task. Unless
/// [`BlockCachedSource::admit_run`] hands it on to the run's [`FetchedRun`],
/// the entry leaves `runs` when the claim drops: when the fetch ends without
/// the run being cached, or is cancelled.
struct Claim {
    source: Arc<BlockCachedSource>,
    first: u32,
    id: u64,
    handed_on: bool,
}

impl Claim {
    /// Leaves the entry to the run's [`FetchedRun`], and returns its id.
    fn hand_on(mut self) -> u64 {
        self.handed_on = true;
        self.id
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        if !self.handed_on {
            self.source.remove_run(self.first, self.id);
        }
    }
}

/// Where one stretch `[first, last]` of a read's blocks is right now.
enum Segment {
    /// On the block file.
    Filled { first: u32, last: u32 },
    /// In memory, fetched. `bytes` start at block `run_first`.
    InMemory {
        first: u32,
        last: u32,
        run_first: u32,
        bytes: Bytes,
    },
    /// Being fetched in run `[run_first, run_last]`; `fetch` ends with its
    /// bytes. `own` when this read registered the run, rather than found
    /// another read's.
    Fetching {
        first: u32,
        last: u32,
        run_first: u32,
        run_last: u32,
        fetch: RunFetch,
        own: bool,
    },
    /// In none of the above. A read registers a fetch for it (see
    /// [`BlockCachedSource::claim_segments`]); other callers see it as is.
    Missing { first: u32, last: u32 },
}

impl Segment {
    fn blocks(&self) -> (u32, u32) {
        match *self {
            Self::Filled { first, last }
            | Self::InMemory { first, last, .. }
            | Self::Fetching { first, last, .. }
            | Self::Missing { first, last } => (first, last),
        }
    }

    /// The same, over blocks `[first, last]` of it.
    fn part(&self, first: u32, last: u32) -> Self {
        match self {
            Self::Filled { .. } => Self::Filled { first, last },
            Self::InMemory {
                run_first, bytes, ..
            } => Self::InMemory {
                first,
                last,
                run_first: *run_first,
                bytes: bytes.clone(),
            },
            Self::Fetching {
                run_first,
                run_last,
                fetch,
                own,
                ..
            } => Self::Fetching {
                first,
                last,
                run_first: *run_first,
                run_last: *run_last,
                fetch: fetch.clone(),
                own: *own,
            },
            Self::Missing { .. } => Self::Missing { first, last },
        }
    }
}

/// How [`BlockCachedSource::segments_in`] shows blocks another read is
/// fetching.
#[derive(Clone, Copy, PartialEq)]
enum InFlight {
    /// As [`Segment::Fetching`]: where the read gets them.
    Fetching,
    /// As [`Segment::Missing`], as if nothing were in flight: how GETs are
    /// sized, so a read never makes more of them than it would alone.
    Missing,
}

/// Where one block is, while [`BlockCachedSource::segments_in`] groups blocks.
#[derive(Clone, Copy, PartialEq)]
enum BlockAt {
    Filled,
    Missing,
    /// In the fetched run at this index of the snapshot.
    InMemory(usize),
    /// In the run being fetched at this index of the snapshot.
    Fetching(usize),
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
            runs: Mutex::new(BTreeMap::new()),
            next_run_id: AtomicU64::new(0),
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
        // Path is gone or now names another generation's inode — never
        // stamp our bitmap onto it.
        if !self.path_names(bf) {
            return;
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

    /// Whether `self.path` still names `bf`'s inode, rather than being gone or
    /// naming a later generation's file.
    fn path_names(&self, bf: &BlockFile) -> bool {
        use std::os::unix::fs::MetadataExt;
        match (bf.file.metadata(), fs::metadata(&self.path)) {
            (Ok(own), Ok(named)) => own.ino() == named.ino(),
            _ => false,
        }
    }

    /// Whether this source has a block file and its path still names it.
    pub(crate) fn path_names_its_file(&self) -> bool {
        self.created_block_file()
            .is_some_and(|bf| self.path_names(bf))
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
    /// Blocks another read is fetching count as not cached here, as if nothing
    /// were in flight, and so do the blocks of the read's own earlier windows,
    /// which `floor` stops at: the window is the one the read would GET alone.
    /// [`Self::claim_segments`] decides what of it to send. `runs` is locked by
    /// the caller.
    fn get_run(
        &self,
        runs: &BTreeMap<u32, MemoryRun>,
        size: u64,
        first: u32,
        last: u32,
        floor: u32,
    ) -> (u32, u32) {
        let (grown, last) = self.aligned_run(size, first, last);
        // A group's leading blocks that an earlier window, this read's or a
        // past one's, already fetched are not fetched again.
        let from = grown.max(floor);
        let first = if from < first {
            self.segments_in(runs, from, first - 1, InFlight::Missing)
                .iter()
                .find_map(|segment| match segment {
                    Segment::Missing { first, .. } => Some(*first),
                    _ => None,
                })
                .unwrap_or(first)
        } else {
            first
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
                    .segments_in(
                        runs,
                        first.saturating_sub(behind).max(lo),
                        before,
                        InFlight::Missing,
                    )
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
            for segment in self.segments_in(runs, last + 1, window_last, InFlight::Missing) {
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

    /// Whether a run is in memory: being fetched, or waiting to be written. The
    /// idle sweep keeps such an entry, because `filled_bytes` does not count the
    /// run until it is written.
    pub(crate) fn has_runs_in_memory(&self) -> bool {
        !self.runs.lock().expect("runs mutex poisoned").is_empty()
    }

    /// Takes run `id` out of `runs`, if it is still there under block `first`.
    fn remove_run(&self, first: u32, id: u64) {
        let mut runs = self.runs.lock().expect("runs mutex poisoned");
        if runs.get(&first).is_some_and(|run| run.id == id) {
            runs.remove(&first);
        }
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

    /// Splits blocks `[b0, b1]` into stretches by where their bytes are. Each
    /// block is, in this order: in a fetched run, on the block file, in a run
    /// being fetched, or missing. A write marks its blocks filled before its run
    /// leaves `runs`, so a block found in none of these really is missing.
    fn segments(&self, b0: u32, b1: u32) -> Vec<Segment> {
        let runs = self.runs.lock().expect("runs mutex poisoned");
        self.segments_in(&runs, b0, b1, InFlight::Fetching)
    }

    /// [`Self::segments`] over `runs`, locked by the caller, with the blocks
    /// other reads are fetching shown as `in_flight` says.
    fn segments_in(
        &self,
        runs: &BTreeMap<u32, MemoryRun>,
        b0: u32,
        b1: u32,
        in_flight: InFlight,
    ) -> Vec<Segment> {
        // A fetch every read dropped is no run.
        let snapshot: Vec<(u32, u32, RunBytes<RunFetch>)> = runs
            .range(..=b1)
            .filter(|(_, run)| run.last >= b0)
            .filter_map(|(&first, run)| {
                let bytes = match &run.bytes {
                    RunBytes::Fetched(bytes) => RunBytes::Fetched(bytes.clone()),
                    RunBytes::Fetching(_) if in_flight == InFlight::Missing => return None,
                    RunBytes::Fetching(fetch) => RunBytes::Fetching(fetch.upgrade()?),
                };
                Some((first, run.last, bytes))
            })
            .collect();

        // Where each block is, painted from the lowest precedence up: runs being
        // fetched, then the block file, then fetched runs, so a block also in a
        // run being fetched is read from where it already is. Of two runs of one
        // kind over a block, the one starting first is painted last and wins.
        let mut at = vec![BlockAt::Missing; (b1 - b0) as usize + 1];
        let paint = |at: &mut [BlockAt], fetched: bool| {
            for (i, (first, last, bytes)) in snapshot.iter().enumerate().rev() {
                if matches!(bytes, RunBytes::Fetched(_)) == fetched {
                    let here = if fetched {
                        BlockAt::InMemory(i)
                    } else {
                        BlockAt::Fetching(i)
                    };
                    at[(first.max(&b0) - b0) as usize..=(last.min(&b1) - b0) as usize].fill(here);
                }
            }
        };
        paint(&mut at, false);
        for b in self
            .filled
            .lock()
            .expect("filled bitmap mutex poisoned")
            .range(b0..=b1)
        {
            at[(b - b0) as usize] = BlockAt::Filled;
        }
        paint(&mut at, true);

        at.chunk_by(|a, b| a == b)
            .scan(b0, |next, stretch| {
                let (first, last) = (*next, *next + stretch.len() as u32 - 1);
                *next = last + 1;
                Some((first, last, stretch[0]))
            })
            .map(|(first, last, at)| match at {
                BlockAt::Filled => Segment::Filled { first, last },
                BlockAt::Missing => Segment::Missing { first, last },
                BlockAt::InMemory(i) | BlockAt::Fetching(i) => {
                    let (run_first, run_last, bytes) = &snapshot[i];
                    match bytes {
                        RunBytes::Fetched(bytes) => Segment::InMemory {
                            first,
                            last,
                            run_first: *run_first,
                            bytes: bytes.clone(),
                        },
                        RunBytes::Fetching(fetch) => Segment::Fetching {
                            first,
                            last,
                            run_first: *run_first,
                            run_last: *run_last,
                            fetch: fetch.clone(),
                            own: false,
                        },
                    }
                }
            })
            .collect()
    }

    /// Registers missing run `[first, last]` in `runs`, and spawns its GET.
    fn register(&self, runs: &mut BTreeMap<u32, MemoryRun>, first: u32, last: u32) -> RunFetch {
        let source = self.me.upgrade().expect("a read holds its source");
        let id = self.next_run_id.fetch_add(1, Ordering::Relaxed);
        let claim = Claim {
            source: Arc::clone(&source),
            first,
            id,
            handed_on: false,
        };
        let fetch = async move {
            match source.block_file() {
                Some(bf) => source.fetch_now(bf, first, last, Some(claim)).await,
                None => Fetched::Elsewhere,
            }
        };
        let fetch = FetchTask(shared_io_runtime().spawn(fetch.in_current_span())).shared();
        // An entry already under `first` can only be a run every read dropped
        // (its blocks looked missing). Its claim, when it drops, finds another
        // id here and leaves this run alone.
        runs.insert(
            first,
            MemoryRun {
                id,
                last,
                bytes: RunBytes::Fetching(fetch.downgrade().expect("a fetch not polled yet")),
            },
        );
        fetch
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
                Segment::InMemory {
                    run_first, bytes, ..
                } => {
                    let base = u64::from(*run_first) * CACHE_BLOCK_BYTES;
                    bytes.slice((s - base) as usize..(e - base) as usize)
                }
                Segment::Fetching { .. } | Segment::Missing { .. } => return None,
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
    /// file, fetched runs from memory, and every other run through its shared
    /// fetch, its own (see [`Self::claim_segments`]) or another read's (hedged
    /// past [`HEDGE_AFTER`]). It awaits all of them at once, so it takes as long
    /// as the slowest, not their sum. `None` sends the read to the whole-file
    /// copy or to object storage: one of its own fetches ended without bytes
    /// ([`Fetched::Elsewhere`]), or a local read failed.
    async fn read_through(
        &self,
        bf: &BlockFile,
        start: u64,
        len: u64,
        window: Window,
    ) -> Result<Option<Bytes>, LazyByteSourceError> {
        let (b0, b1) = Self::block_span(start, len);
        let segments = self.claim_segments(b0, b1, bf.size, window);

        // Every fetch is awaited to its end, even once one has ended without
        // bytes: the others queue their bytes for writing as they land, so those
        // are cached for the next read rather than cancelled mid-GET.
        let ended = join_all(segments.into_iter().map(|segment| async move {
            let Segment::Fetching {
                first,
                last,
                run_first,
                run_last,
                fetch,
                own,
            } = segment
            else {
                return Ok(segment);
            };
            let (run_first, fetched) = if own {
                (run_first, fetch.await)
            } else {
                self.wait_or_hedge(bf, first, last, (run_first, run_last), fetch)
                    .await
            };
            match fetched {
                Fetched::Bytes(bytes) => Ok(Segment::InMemory {
                    first,
                    last,
                    run_first,
                    bytes,
                }),
                stopped => Err(stopped),
            }
        }))
        .await;

        let mut segments = Vec::with_capacity(ended.len());
        for segment in ended {
            match segment {
                Ok(segment) => segments.push(segment),
                Err(Fetched::Failed(error)) => return Err(error.take()),
                Err(Fetched::Panicked(panic)) => panic.resume(),
                Err(_) => return Ok(None),
            }
        }
        Ok(self.assemble(bf, start, len, &segments))
    }

    /// Waits for another read's fetch of blocks `[first, last]`, in run `run`
    /// (its first and last block). Past [`Self::hedge_after`] it also GETs those
    /// blocks itself and keeps whichever lands first; a hedge that ends without
    /// bytes is dropped and the wait goes on. A fetch that ends elsewhere for
    /// its own reasons (its budget, a short GET) leaves this read to fetch its
    /// blocks itself, as it would have without waiting. Returns the block the
    /// bytes start at, and how the fetch ended.
    async fn wait_or_hedge(
        &self,
        bf: &BlockFile,
        first: u32,
        last: u32,
        run: (u32, u32),
        mut fetch: RunFetch,
    ) -> (u32, Fetched) {
        let (run_first, run_last) = run;
        let after = Self::hedge_after(Self::run_bytes(bf.size, run_first, run_last).1);
        let waited = match timeout(after, &mut fetch).await {
            Ok(fetched) => fetched,
            Err(_) => select! {
                fetched = &mut fetch => fetched,
                hedged = self.fetch_now(bf, first, last, None) => match hedged {
                    Fetched::Bytes(_) => return (first, hedged),
                    _ => fetch.await,
                },
            },
        };
        match waited {
            Fetched::Elsewhere => (first, self.fetch_now(bf, first, last, None).await),
            fetched => (run_first, fetched),
        }
    }

    /// How long a read waits on another read's GET of `bytes` before it GETs the
    /// blocks it needs itself: [`HEDGE_AFTER`], plus the time those bytes take at
    /// [`STUCK_BYTES_PER_SEC`].
    fn hedge_after(bytes: u64) -> Duration {
        HEDGE_AFTER + Duration::from_millis(bytes.saturating_mul(1000) / STUCK_BYTES_PER_SEC)
    }

    /// Fetches blocks `[first, last]` and queues them for writing without
    /// waiting for the write; see [`Self::fetch_missing`].
    async fn fetch_now(
        &self,
        bf: &BlockFile,
        first: u32,
        last: u32,
        claim: Option<Claim>,
    ) -> Fetched {
        match self.fetch_missing(bf, first, last, Fill::Read, claim).await {
            Ok(Some((bytes, run))) => {
                if let Some(run) = run {
                    // Don't wait for the write: the reads already have their bytes.
                    drop(run.spawn_write());
                }
                Fetched::Bytes(bytes)
            }
            Ok(None) => Fetched::Elsewhere,
            Err(error) => Fetched::Failed(FetchError::new(error)),
        }
    }

    /// Splits `[b0, b1]` as [`Self::segments`] does, and registers a fetch for
    /// each GET this read needs, so a read racing it finds those blocks in
    /// flight. It looks, plans and registers under one hold of `runs`, so no
    /// other read registers any of those blocks meanwhile.
    ///
    /// The GETs are the windows the read would send if nothing else were
    /// reading ([`Self::plan_windows`]), so waiting never costs it more GETs
    /// than fetching. A window that holds none of the read's missing blocks is
    /// not sent, and the others are sent from their first missing block: what
    /// is in flight before that block is waited for, not fetched again.
    ///
    /// ```text
    ///   block:    5   6   7..9   10 ....... 17   18   19   20
    ///   now:      M   C   M      [ in flight  ]   M    C    M
    ///   windows:  [ 5 ...... 12 ][ 13 .................... 20 ]   as if alone
    ///   GETs:     [ 5 ...... 12 ]  wait for 13..17 [ 18 .... 20 ]
    /// ```
    ///
    /// Window 1 fetches through blocks 10..12 (it would have alone, and the read
    /// takes them from it); window 2 starts on blocks another read is fetching,
    /// so the read waits for those and GETs it from block 18.
    fn claim_segments(&self, b0: u32, b1: u32, size: u64, window: Window) -> Vec<Segment> {
        let mut runs = self.runs.lock().expect("runs mutex poisoned");
        let segments = self.segments_in(&runs, b0, b1, InFlight::Fetching);
        let missing: Vec<(u32, u32)> = segments
            .iter()
            .filter_map(|segment| match *segment {
                Segment::Missing { first, last } => Some((first, last)),
                _ => None,
            })
            .collect();
        if missing.is_empty() {
            return segments;
        }

        let mut fetches = Vec::new();
        for (start, last) in self.plan_windows(&runs, size, b0, b1, window) {
            // A window that holds none of the read's missing blocks is not sent:
            // nothing waits on it, so dropping it would cancel it mid-GET. The
            // read that wants its other blocks sizes a window of its own.
            if !missing
                .iter()
                .any(|&(from, to)| from <= last && start <= to)
            {
                continue;
            }
            let first = self
                .segments_in(&runs, start, last, InFlight::Fetching)
                .iter()
                .find_map(|segment| match *segment {
                    Segment::Missing { first, .. } => Some(first),
                    _ => None,
                });
            if let Some(first) = first {
                fetches.push((first, last, self.register(&mut runs, first, last)));
            }
        }
        drop(runs);

        // The read takes every block its own GETs cover from them, those another
        // read is fetching too: it pays for those GETs either way, and waits on
        // nothing else. Every missing block lies in one of them.
        let mut claimed = Vec::with_capacity(segments.len() + 2 * fetches.len());
        for segment in segments {
            let (first, last) = segment.blocks();
            if !matches!(segment, Segment::Missing { .. } | Segment::Fetching { .. }) {
                claimed.push(segment);
                continue;
            }
            let mut next = first;
            for (run_first, run_last, fetch) in &fetches {
                let (from, to) = (next.max(*run_first), last.min(*run_last));
                if from > to {
                    continue;
                }
                if next < from {
                    claimed.push(segment.part(next, from - 1));
                }
                claimed.push(Segment::Fetching {
                    first: from,
                    last: to,
                    run_first: *run_first,
                    run_last: *run_last,
                    fetch: fetch.clone(),
                    own: true,
                });
                next = to + 1;
            }
            if next <= last {
                claimed.push(segment.part(next, last));
            }
        }
        debug_assert!(
            !claimed
                .iter()
                .any(|segment| matches!(segment, Segment::Missing { .. })),
            "a missing block outside the read's windows"
        );
        claimed
    }

    /// The GET windows a read of blocks `[b0, b1]` would send if nothing else
    /// were reading, in order: one per stretch not cached yet, counting blocks
    /// other reads are fetching as not cached, sized by [`Self::get_run`]. A
    /// stretch, or the part of it, that the previous window already covers gets
    /// no window of its own, and each window starts after the previous one.
    fn plan_windows(
        &self,
        runs: &BTreeMap<u32, MemoryRun>,
        size: u64,
        b0: u32,
        b1: u32,
        window: Window,
    ) -> Vec<(u32, u32)> {
        let mut windows: Vec<(u32, u32)> = Vec::new();
        for segment in self.segments_in(runs, b0, b1, InFlight::Missing) {
            let Segment::Missing { mut first, last } = segment else {
                continue;
            };
            if let Some(&(_, covered)) = windows.last()
                && first <= covered
            {
                if last <= covered {
                    continue;
                }
                first = covered + 1;
            }
            let floor = windows.last().map_or(0, |&(_, covered)| covered + 1);
            windows.push(match window {
                Window::Grown => self.get_run(runs, size, first, last, floor),
                Window::Exact => (first, last),
            });
        }
        windows
    }

    /// Fetches blocks `[first, last]` with one GET, reserving their budget
    /// first, and queues them for writing. Returns:
    /// - `None` when the read has to go elsewhere: the store is gone or this
    ///   source is no longer the live entry, no budget is left, or the GET
    ///   returned another length (the object is not the one this reader opened);
    /// - the bytes and no run when they are fetched but not queued: the cap on
    ///   unwritten runs is full, or another run starting at the same block is in
    ///   memory.
    ///
    /// If the caller is dropped mid-GET, the reservation gives its bytes back.
    /// `claim` is the run's entry in `runs` when a read registered it; see
    /// [`Self::admit_run`].
    async fn fetch_missing(
        &self,
        bf: &BlockFile,
        first: u32,
        last: u32,
        fill: Fill,
        claim: Option<Claim>,
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

        let run = self.admit_run(&store, first, last, &bytes, reservation, fill, claim);
        Ok(Some((bytes, run)))
    }

    /// Queues fetched run `[first, last]` for writing: takes its share of the
    /// store's cap on unwritten runs, keeps its bytes in `runs`, and hands it the
    /// reserved budget. A registered run keeps the `claim`'s entry; an
    /// unregistered one (a prefetch's or a hedge's) gets a new entry. `None`
    /// (the reservation is given back, and a registered run leaves `runs`) if a
    /// read finds the cap full, or an unregistered run finds another run
    /// starting at `first`.
    fn admit_run(
        &self,
        store: &Arc<DiskCacheStore>,
        first: u32,
        last: u32,
        bytes: &Bytes,
        reservation: Option<Reservation<'_>>,
        fill: Fill,
        claim: Option<Claim>,
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

        let claimed = claim.map(Claim::hand_on);
        let mut runs = self.runs.lock().expect("runs mutex poisoned");
        let id = match claimed {
            // The entry is gone only when every read dropped this fetch while it
            // finished, and a new read registered the blocks again: the bytes
            // are still written, under an id no entry has.
            Some(id) => {
                if let Some(run) = runs.get_mut(&first).filter(|run| run.id == id) {
                    run.bytes = RunBytes::Fetched(bytes.clone());
                }
                id
            }
            None => {
                // Another run owns this key until it is written, so this one is
                // served but not cached. A run every read dropped is no run: its
                // claim, when it drops, finds another id here and leaves this
                // one alone.
                if runs.get(&first).is_some_and(|run| !run.dropped()) {
                    store.release_write_behind(run_len);
                    return None;
                }
                let id = self.next_run_id.fetch_add(1, Ordering::Relaxed);
                runs.insert(
                    first,
                    MemoryRun {
                        id,
                        last,
                        bytes: RunBytes::Fetched(bytes.clone()),
                    },
                );
                id
            }
        };
        drop(runs);

        let unlanded = reservation.map_or(0, |reserved| {
            reserved.commit();
            run_len
        });

        Some(FetchedRun {
            source,
            #[cfg(test)]
            _ticket: store.block_write_started(),
            store: Arc::clone(store),
            id,
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
    /// neither filled nor in memory, and waits for it to be written. It reserves
    /// only free budget, so a prefetch never evicts what queries use. It first
    /// waits for the runs reads are fetching (driving their GETs) until none is
    /// left, then fetches what is still missing, so a run a read did not cache
    /// is filled too, and looks again until nothing is. A run another fill
    /// queued is left to it. `false` stops the prefetch, when:
    /// - the budget is full,
    /// - this source is no longer the live entry, or the store is gone,
    /// - the GET came back short, or the write failed.
    async fn prefetch_chunk(
        &self,
        bf: &BlockFile,
        b0: u32,
        b1: u32,
    ) -> Result<bool, LazyByteSourceError> {
        // Until no block is missing: a read can register a run at the first
        // block of one of this chunk's GETs while it is out. That run keeps the
        // GET's bytes from being cached, and its own fetch can end without
        // caching them either.
        loop {
            let segments = loop {
                let segments = self.segments(b0, b1);
                let fetching: Vec<RunFetch> = segments
                    .iter()
                    .filter_map(|segment| match segment {
                        Segment::Fetching { fetch, .. } => Some(fetch.clone()),
                        _ => None,
                    })
                    .collect();
                if fetching.is_empty() {
                    break segments;
                }
                for fetched in join_all(fetching).await {
                    if let Fetched::Panicked(panic) = fetched {
                        panic.resume();
                    }
                }
            };

            let missing: Vec<(u32, u32)> = segments
                .iter()
                .filter_map(|segment| match *segment {
                    Segment::Missing { first, last } => Some((first, last)),
                    _ => None,
                })
                .collect();
            if missing.is_empty() {
                return Ok(true);
            }

            for (first, last) in missing {
                let Some((_, run)) = self
                    .fetch_missing(bf, first, last, Fill::Prefetch, None)
                    .await?
                else {
                    return Ok(false);
                };

                if let Some(run) = run
                    && !run.spawn_write().await.unwrap_or(false)
                {
                    return Ok(false);
                }
            }
        }
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

/// How far a read that misses the cache fetches.
#[derive(Clone, Copy)]
enum Window {
    /// Grown to a window around the read ([`BlockCachedSource::get_run`]), so
    /// its neighbours are local for the reads after it.
    Grown,
    /// Just the blocks the read touches: a caller that knows exactly which
    /// bytes it needs, and fetches them all at once.
    Exact,
}

#[async_trait]
impl LazyByteSource for BlockCachedSource {
    fn size(&self) -> u64 {
        self.inner.size()
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        let bytes = self.read(start, len, Window::Grown).await;
        self.note_read(start, len);
        bytes
    }

    fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
        let bytes = self.read_sync(start, len)?;
        self.note_read(start, len);
        Some(bytes)
    }

    fn note_read(&self, start: u64, len: u64) {
        if len > 0 {
            self.note_read_end(start.saturating_add(len));
        }
    }

    fn is_local(&self, start: u64, len: u64) -> bool {
        self.holds(start, len) || self.whole_file().is_some()
    }

    async fn range_ahead(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        self.read(start, len, Window::Grown).await
    }

    /// [`Self::range`] without the window: a miss fetches just the blocks the
    /// read touches and keeps them, since the caller knows the bytes around
    /// them are not what it reads next, and a repeat of the read finds them
    /// local.
    async fn range_exact(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        let bytes = self.read(start, len, Window::Exact).await;
        self.note_read(start, len);
        bytes
    }

    async fn tail(&self, len: u64) -> Result<(Bytes, u64), LazyByteSourceError> {
        // Pass through: `tail` both fetches and (on unknown-size sources)
        // discovers the object size, which the inner source caches. Tail
        // bytes are open-time metadata already retained by the open-blob /
        // prefetch overlay above this source, so caching them here as
        // (mostly partial) blocks buys nothing.
        self.inner.tail(len).await
    }

    fn cache_blocks(&self) -> Option<CacheBlocks> {
        // A file whose layout was unknown at open is treated as all vector
        // section, so every miss there downloads whole groups.
        let grouped = self.aligned_blocks().map_or((0, 0), |(first, last)| {
            (
                u64::from(first) * CACHE_BLOCK_BYTES,
                (u64::from(last) + 1) * CACHE_BLOCK_BYTES,
            )
        });
        Some(CacheBlocks {
            block_bytes: CACHE_BLOCK_BYTES,
            min_get_bytes: MIN_GET_BYTES,
            group_bytes: ALIGNED_GET_BYTES,
            grouped,
        })
    }

    fn cache_room_ahead(&self, bytes: u64) -> Option<CacheRoom> {
        match self.store.upgrade() {
            Some(store) => Some(Box::new(store.try_reserve_ahead(bytes)?)),
            None => Some(Box::new(())),
        }
    }
}

impl BlockCachedSource {
    async fn read(
        &self,
        start: u64,
        len: u64,
        window: Window,
    ) -> Result<Bytes, LazyByteSourceError> {
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
                && let Some(bytes) = self.read_through(bf, start, len, window).await?
            {
                return Ok(bytes);
            }
        }

        match self.whole_file() {
            Some(file) => file.range(start, len).await,
            None => self.inner.range(start, len).await,
        }
    }

    /// Whether every block of the range is on disk or being written, without
    /// reading it or noting a read end. Without a block file, the inner
    /// source answers.
    fn holds(&self, start: u64, len: u64) -> bool {
        if len == 0 {
            return true;
        }
        let Some(bf) = self.block_file_for(self.in_fts_hole(start, len)) else {
            return self.inner.is_local(start, len);
        };
        if start.saturating_add(len) > bf.size {
            return false;
        }
        let (b0, b1) = Self::block_span(start, len);
        self.all_filled(b0, b1)
            || self
                .segments(b0, b1)
                .iter()
                .all(|segment| matches!(segment, Segment::Filled { .. } | Segment::InMemory { .. }))
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
        // Blocks in memory come from there. A block missing or still being
        // fetched (this read can't wait), or a failed local read, sends the read
        // to the inner source.
        self.assemble(bf, start, len, &self.segments(b0, b1))
            .or_else(|| self.inner.try_get_range_sync(start, len))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeSet,
        future::pending,
        ops::Range,
        panic::{AssertUnwindSafe, panic_any},
        path::Path,
        pin::pin,
        sync::{atomic::AtomicUsize, mpsc},
        task::Poll,
        thread,
        time::{Duration, Instant as StdInstant},
    };

    use futures::{FutureExt, poll};
    use tempfile::tempdir;
    use tokio::{
        sync::Notify,
        time::{Instant, advance, sleep, timeout},
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

    /// Blocks until `cond` holds, failing with `what` after [`NO_WAIT_DEADLINE`]
    /// of real time. For tests on a paused clock waiting on work on the I/O
    /// runtime: [`wait_until`]'s sleeps would let the paused clock run ahead.
    fn wait_real(cond: impl Fn() -> bool, what: &str) {
        let deadline = StdInstant::now() + NO_WAIT_DEADLINE;
        while !cond() {
            assert!(
                StdInstant::now() < deadline,
                "timed out waiting until {what}"
            );
            thread::sleep(POLL_INTERVAL);
        }
    }

    /// In-memory fake source that counts `range` calls and the bytes they ask for.
    struct CountingSource {
        blob: Bytes,
        calls: AtomicUsize,
        bytes: AtomicU64,
        ranges: Mutex<Vec<Range<u64>>>,
    }

    impl CountingSource {
        fn new(n: usize) -> Self {
            let blob: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
            Self {
                blob: Bytes::from(blob),
                calls: AtomicUsize::new(0),
                bytes: AtomicU64::new(0),
                ranges: Mutex::new(Vec::new()),
            }
        }

        /// Every range asked for, in order.
        fn ranges(&self) -> Vec<Range<u64>> {
            self.ranges.lock().expect("ranges").clone()
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
            self.ranges.lock().expect("ranges").push(start..start + len);
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

        // The fill reserves, then waits on the GET, and the read is dropped.
        let mut read = Box::pin(src.range(0, 100));
        assert!(poll!(read.as_mut()).is_pending(), "the GET hangs");
        wait_until(|| store.stats().current_bytes > 0, "the fill reserves").await;
        drop(read);
        // The GET's task is cancelled on the I/O runtime, which then drops it.
        wait_until(
            || store.stats().current_bytes == 0,
            "the reservation is given back",
        )
        .await;

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

    /// Asking whether bytes are local is not a read: more probes far ahead than
    /// there are read-end slots still leave a forward reader's next window
    /// ahead of it. And the whole range is checked, not just its ends.
    #[tokio::test]
    async fn asking_whether_bytes_are_local_is_not_a_read() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(dir.path(), &store, 4 * GROUP as usize, None);

        let _ = src.range(10, 100).await.expect("read blocks 0 to 7");
        store.block_writes_settled().await;
        let bf = src.block_file().expect("block file");
        bf.file
            .write_all_at(&inner.blob[24 * b as usize..32 * b as usize], 24 * b)
            .expect("write blocks 24 to 31");
        src.mark_filled(bf.size, 24, 31);
        let calls = inner.calls();
        for block in 24..32 {
            for _ in 0..2 {
                assert!(src.is_local(block * b, b), "block {block} is filled");
            }
        }
        assert_eq!(inner.calls(), calls, "a probe downloads nothing");
        assert!(
            !src.is_local(7 * b, 18 * b),
            "both ends filled, blocks between them missing"
        );

        let _ = src
            .range(10 * b + 10, 100)
            .await
            .expect("a read further on");
        store.block_writes_settled().await;
        assert!(src.all_filled(10, 17), "its window still goes ahead");
        assert!(!src.all_filled(8, 9), "nothing behind it");

        store.remove_block_entry_for_test(&uri);
    }

    /// A reader served from what it downloaded ahead never asks the cache, so
    /// it notes the read instead. However many other reads end past it in the
    /// meantime, its next miss still puts the window ahead.
    #[tokio::test]
    async fn a_read_served_from_bytes_downloaded_ahead_keeps_its_direction() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, _inner, src) = layout_source(dir.path(), &store, 4 * GROUP as usize, None);

        let _ = src.range(10, 100).await.expect("read blocks 0 to 7");
        store.block_writes_settled().await;
        // Other readers of the file end far ahead, more of them than there
        // are read-end slots.
        for _ in 0..READ_ENDS {
            src.note_read(60 * b, b);
        }
        // This reader reads blocks 8 and 9 from memory.
        src.note_read(8 * b, 2 * b);
        let _ = src.range(10 * b + 10, 100).await.expect("its next read");
        store.block_writes_settled().await;
        assert!(src.all_filled(10, 17), "its window goes ahead");
        assert!(!src.all_filled(8, 9), "nothing behind it");

        store.remove_block_entry_for_test(&uri);
    }

    /// Blocks downloaded but not written yet are served from memory, so they
    /// count as local.
    #[tokio::test]
    async fn blocks_being_written_are_local() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, _inner, src) = layout_source(dir.path(), &store, 4 * GROUP as usize, None);

        let stall = src.stall_writes();
        let _ = src.range(10, 100).await.expect("read blocks 0 to 7");
        assert!(!src.all_filled(0, 7), "not written yet");
        assert!(src.is_local(0, 8 * b), "but local");
        assert!(!src.is_local(8 * b, b), "block 8 was never read");

        drop(stall);
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

    /// A GET also stops at a long stretch in memory, blocks fetched but not
    /// written yet.
    #[tokio::test]
    async fn a_get_stops_at_a_long_stretch_in_memory() {
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

    /// A short read whose block lies in another read's window, still being
    /// fetched, waits for that GET instead of making its own: one GET serves both,
    /// and only that window is filled and charged.
    #[tokio::test]
    async fn a_read_inside_another_reads_window_waits_for_its_get() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(4 * GROUP as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut one = pin!(src.range(6 * b + 10, 64));
        let mut two = pin!(src.range(9 * b + 10, 64));
        assert!(poll!(one.as_mut()).is_pending(), "the first read GETs");
        assert!(poll!(two.as_mut()).is_pending(), "the second read waits");
        inner.release();
        let (one, two) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(one, two) })
            .await
            .expect("both reads finish");
        let at = |start: u64| {
            inner
                .inner
                .blob
                .slice(start as usize..(start + 64) as usize)
        };
        assert_eq!(one.expect("first read"), at(6 * b + 10));
        assert_eq!(two.expect("second read"), at(9 * b + 10));
        assert_eq!(inner.inner.calls(), 1, "one GET for both reads");

        store.block_writes_settled().await;
        let window = u64::from(MIN_GET_BLOCKS) * b;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), window);
        assert_eq!(
            store.stats().current_bytes,
            window,
            "the window, charged once"
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

    /// A file of unknown layout downloads whole 512 KiB groups, and says so to
    /// read-ahead; any other file reports blocks plus a window for a short
    /// miss.
    #[tokio::test]
    async fn a_file_of_unknown_layout_reports_whole_group_downloads() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = layout_source(
            dir.path(),
            &store,
            4 * GROUP as usize,
            SourceLayout::ALL_ALIGNED.aligned_region,
        );
        let blocks = src.cache_blocks().expect("cache blocks");
        let start = GROUP + 3 * CACHE_BLOCK_BYTES + 10;
        let read = start..start + 100;
        assert_eq!(blocks.reach(&read), GROUP..2 * GROUP);
        let _ = src.range(read.start, 100).await.expect("read");
        assert_eq!(inner.fetched_bytes(), GROUP, "the group its reach names");
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);

        let (uri, _, src) = layout_source(dir.path(), &store, 4 * GROUP as usize, None);
        assert_eq!(
            src.cache_blocks(),
            Some(CacheBlocks {
                block_bytes: CACHE_BLOCK_BYTES,
                min_get_bytes: MIN_GET_BYTES,
                group_bytes: GROUP,
                grouped: (0, 0),
            })
        );
        store.remove_block_entry_for_test(&uri);
    }

    /// What [`CacheBlocks`] says a cold miss downloads holds for the cache
    /// itself, around a vector section that covers only part of the file: a
    /// short and a long miss starting at every block, each on a cold file.
    /// - every byte downloaded is inside the miss's `reach`;
    /// - the bytes downloaded are at most its `download_len`.
    #[tokio::test]
    async fn cache_blocks_bound_every_miss_around_a_vector_section() {
        let b = CACHE_BLOCK_BYTES;
        let size = 8 * GROUP;
        // Starts and ends mid-group, so both edges round out.
        let region = (2 * GROUP + 3 * b, 2 * GROUP);
        for len in [100, 3 * b] {
            for block in 0..(size - len) / b {
                let dir = tempdir().expect("tempdir");
                let store = test_store(dir.path(), u64::MAX);
                let (uri, inner, src) =
                    layout_source(dir.path(), &store, size as usize, Some(region));
                let read = block * b + 10..block * b + 10 + len;
                let blocks = src.cache_blocks().expect("cache blocks");
                let _ = src.range(read.start, len).await.expect("read");
                let reach = blocks.reach(&read);
                for got in inner.ranges() {
                    assert!(
                        reach.start <= got.start && got.end <= reach.end.min(size),
                        "{read:?}: downloaded {got:?}, reach {reach:?}"
                    );
                }
                assert!(
                    inner.fetched_bytes() <= blocks.download_len(&read),
                    "{read:?}: downloaded {} bytes, download_len {}",
                    inner.fetched_bytes(),
                    blocks.download_len(&read)
                );
                store.block_writes_settled().await;
                store.remove_block_entry_for_test(&uri);
            }
        }
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

    /// An exact read that misses fetches just the blocks it touches, not a
    /// window, and keeps them: the same read again makes no GET, and the
    /// bytes are the ones a windowed read returns.
    #[tokio::test]
    async fn an_exact_read_fetches_its_blocks_once_and_keeps_them() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (_uri, inner, src) = counting_source(dir.path(), &store, 64 * UNIT as usize);
        let (start, len) = (10 * UNIT + 100, 4096);
        let want = inner.blob.slice(start as usize..(start + len) as usize);

        assert_eq!(src.range_exact(start, len).await.expect("exact read"), want);
        assert_eq!(inner.calls(), 1);
        assert!(
            inner.fetched_bytes() <= 2 * CACHE_BLOCK_BYTES,
            "the blocks the read touches, not a window: {} bytes",
            inner.fetched_bytes()
        );
        for _ in 0..2 {
            assert_eq!(src.range_exact(start, len).await.expect("repeat"), want);
        }
        assert_eq!(src.range(start, len).await.expect("windowed"), want);
        assert_eq!(inner.calls(), 1, "repeats come from the kept blocks");
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
            .expect("sync read of a run in memory");
        assert_eq!(sync, inner.blob.slice(110..210));
        assert_eq!(inner.calls(), 1, "blocks in memory are served from there");
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
    /// gives everything back: its budget, its share of the cap, its entry in
    /// `runs`. No index lists its blocks.
    #[tokio::test]
    async fn a_run_dropped_unrun_gives_everything_back() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, _inner, src) = counting_source(dir.path(), &store, 2 * b as usize);
        let bf = src.block_file().expect("block file");

        let (_, run) = src
            .fetch_missing(bf, 0, 0, Fill::Read, None)
            .await
            .expect("GET")
            .expect("fetched");
        let run = run.expect("queued");
        assert_eq!(store.stats().current_bytes, b, "the run holds its charge");
        assert!(src.has_runs_in_memory());

        drop(run);
        assert_eq!(store.stats().current_bytes, 0);
        assert_eq!(store.write_behind_in_use(), 0);
        assert!(!src.has_runs_in_memory());
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

    /// A read that starts inside a run in memory gets the right bytes from it,
    /// async and sync, with no GET.
    #[tokio::test]
    async fn a_read_inside_a_run_in_memory_slices_it_at_its_offset() {
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

    /// A read that spans filled, in-memory and missing blocks returns them in
    /// order, and fetches only the missing ones.
    #[tokio::test]
    async fn a_read_stitches_filled_in_memory_and_missing_blocks() {
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) = counting_source(dir.path(), &store, 6 * b as usize);

        // Unit 1 filled, then unit 3 in memory behind a stalled writer.
        let _ = src.range(b, 10).await.expect("fill unit 1");
        store.block_writes_settled().await;
        let stall = src.stall_writes();
        let _ = src.range(3 * b, 10).await.expect("unit 3 into memory");
        assert_eq!(inner.calls(), 2);

        // Units 0..=4: three missing stretches between the filled and the
        // in-memory window, so three GETs.
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

    /// Two reads that miss the same run at once fetch it once: the second waits
    /// for the first's GET. The run is reserved once, written once, and the cap
    /// is given back in full.
    #[tokio::test]
    async fn racing_reads_of_one_run_fetch_it_once() {
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let stall = src.stall_writes();
        let mut one = pin!(src.range(0, 64));
        let mut two = pin!(src.range(10, 64));
        assert!(poll!(one.as_mut()).is_pending(), "the first read GETs");
        assert!(poll!(two.as_mut()).is_pending(), "the second read waits");
        inner.release();
        let (one, two) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(one, two) })
            .await
            .expect("both reads finish");
        assert_eq!(one.expect("first read"), inner.inner.blob.slice(0..64));
        assert_eq!(two.expect("second read"), inner.inner.blob.slice(10..74));
        assert_eq!(inner.inner.calls(), 1, "one GET for both reads");
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

    /// Many small reads of one block, issued together, cost one GET.
    #[tokio::test]
    async fn parallel_small_reads_of_one_block_fetch_it_once() {
        const READS: u64 = 8;
        const READ_LEN: u64 = 50;
        const READ_STRIDE: u64 = 100;
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut reads: Vec<_> = (0..READS)
            .map(|i| Box::pin(src.range(i * READ_STRIDE, READ_LEN)))
            .collect();
        for read in &mut reads {
            assert!(poll!(read.as_mut()).is_pending(), "one GET, the rest wait");
        }
        inner.release();
        let got = timeout(NO_WAIT_DEADLINE, join_all(reads))
            .await
            .expect("every read finishes");
        for (i, bytes) in (0..READS).zip(got) {
            let s = (i * READ_STRIDE) as usize;
            assert_eq!(
                bytes.expect("read"),
                inner.inner.blob.slice(s..s + READ_LEN as usize)
            );
        }
        assert_eq!(inner.inner.calls(), 1, "one GET for all of them");

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A read dropped mid-GET leaves the GET to the read still waiting on it,
    /// which drives it to the end: one GET, and the run is cached.
    #[tokio::test]
    async fn a_dropped_fetcher_leaves_its_get_to_a_waiting_read() {
        // One unit is one short read's GET window.
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut one = Box::pin(src.range(0, 64));
        let mut two = pin!(src.range(10, 64));
        assert!(poll!(one.as_mut()).is_pending(), "the first read GETs");
        assert!(poll!(two.as_mut()).is_pending(), "the second read waits");
        drop(one);
        inner.release();
        let two = timeout(NO_WAIT_DEADLINE, two)
            .await
            .expect("the second read finishes");
        assert_eq!(two.expect("second read"), inner.inner.blob.slice(10..74));
        assert_eq!(inner.inner.calls(), 1, "the first read's GET, finished");

        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), b);
        assert_eq!(store.stats().current_bytes, b, "cached once");
        assert_eq!(store.write_behind_in_use(), 0);
        assert!(!src.has_runs_in_memory());
        store.remove_block_entry_for_test(&uri);
    }

    /// A read nobody polls any more can't hold up a read waiting on its GET:
    /// the waiting read drives it.
    #[tokio::test]
    async fn a_waiting_read_does_not_need_its_fetcher_polled() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut parked = pin!(src.range(0, 64));
        let mut waiting = pin!(src.range(10, 64));
        assert!(poll!(parked.as_mut()).is_pending(), "the first read GETs");
        assert!(
            poll!(waiting.as_mut()).is_pending(),
            "the second read waits"
        );
        inner.release();
        let waiting = timeout(NO_WAIT_DEADLINE, waiting)
            .await
            .expect("finishes with the first read never polled again");
        assert_eq!(
            waiting.expect("second read"),
            inner.inner.blob.slice(10..74)
        );
        assert_eq!(inner.inner.calls(), 1);
        assert_eq!(
            poll!(parked.as_mut()).map(|bytes| bytes.expect("first read")),
            Poll::Ready(inner.inner.blob.slice(0..64)),
            "the parked read finds its bytes when polled",
        );

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// Every read of a run dropped mid-GET cancels it: nothing stays charged or
    /// in memory.
    #[tokio::test]
    async fn dropping_every_read_of_a_run_cancels_its_get() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::new(HangingSource {
                size: 2 * CACHE_BLOCK_BYTES,
            }) as Arc<dyn LazyByteSource>,
        );

        let mut one = Box::pin(src.range(0, 64));
        let mut two = Box::pin(src.range(10, 64));
        assert!(poll!(one.as_mut()).is_pending());
        assert!(poll!(two.as_mut()).is_pending());
        assert!(src.has_runs_in_memory(), "the run is registered");
        drop((one, two));
        // The GET's task is cancelled on the I/O runtime, which then drops it.
        wait_until(
            || !src.has_runs_in_memory(),
            "the run is gone with its last read",
        )
        .await;
        wait_until(
            || store.stats().current_bytes == 0,
            "its budget is given back",
        )
        .await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A failed GET reaches every read waiting on it as its error, at once: none
    /// tries the GET again.
    #[tokio::test]
    async fn a_failed_get_errors_every_read_sharing_it() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_ending(2 * CACHE_BLOCK_BYTES as usize, 1, HeldEnd::Fails);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut one = pin!(src.range(0, 64));
        let mut two = pin!(src.range(10, 64));
        assert!(poll!(one.as_mut()).is_pending(), "the first read GETs");
        assert!(poll!(two.as_mut()).is_pending(), "the second read waits");
        inner.release();
        let (one, two) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(one, two) })
            .await
            .expect("both reads finish");
        for read in [one, two] {
            assert!(
                matches!(read, Err(LazyByteSourceError::OutOfBounds { .. })),
                "each read gets the GET's error, got {read:?}"
            );
        }
        assert_eq!(inner.inner.calls(), 0, "and no read GETs again");

        store.block_writes_settled().await;
        assert_eq!(store.stats().current_bytes, 0, "nothing was cached");
        assert_eq!(store.write_behind_in_use(), 0);
        assert!(!src.has_runs_in_memory());
        store.remove_block_entry_for_test(&uri);
    }

    /// Waiting never costs more GETs than fetching: a read whose missing blocks
    /// sit on both sides of a run in flight GETs them with the run, in one GET,
    /// as it would if nothing were in flight.
    #[tokio::test]
    async fn a_run_in_flight_never_splits_a_read_into_more_gets() {
        // One unit is one short read's GET window.
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(3 * b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut middle = pin!(src.range(b + 10, 10));
        assert!(
            poll!(middle.as_mut()).is_pending(),
            "the middle window is in flight"
        );
        wait_until(|| inner.held.load(Ordering::Acquire) == 1, "its GET holds").await;
        let all = timeout(NO_WAIT_DEADLINE, src.range(0, 3 * b))
            .await
            .expect("the wide read doesn't wait")
            .expect("wide read");
        assert_eq!(all, inner.inner.blob.slice(0..(3 * b) as usize));
        assert_eq!(inner.inner.calls(), 1, "one GET for all three windows");

        inner.release();
        let middle = timeout(NO_WAIT_DEADLINE, middle)
            .await
            .expect("the held read finishes")
            .expect("middle read");
        assert_eq!(
            middle,
            inner.inner.blob.slice((b + 10) as usize..(b + 20) as usize)
        );
        assert_eq!(inner.inner.calls(), 2, "as many GETs as without waiting");

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A read whose missing blocks start inside a window in flight waits for
    /// that window and GETs only the rest.
    #[tokio::test]
    async fn a_read_fetches_only_what_is_not_in_flight() {
        // One unit is one short read's GET window.
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut first = pin!(src.range(10, 10));
        let mut both = pin!(src.range(0, 2 * b));
        assert!(
            poll!(first.as_mut()).is_pending(),
            "the first window is in flight"
        );
        wait_until(|| inner.held.load(Ordering::Acquire) == 1, "its GET holds").await;
        assert!(
            poll!(both.as_mut()).is_pending(),
            "the wide read waits for it"
        );
        wait_until(|| inner.inner.calls() == 1, "the wide read GETs").await;
        assert_eq!(
            inner.inner.ranges(),
            vec![b..2 * b],
            "the second window alone"
        );
        inner.release();
        let (first, both) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(first, both) })
            .await
            .expect("both reads finish");
        assert_eq!(first.expect("first read"), inner.inner.blob.slice(10..20));
        assert_eq!(
            both.expect("wide read"),
            inner.inner.blob.slice(0..(2 * b) as usize)
        );
        assert_eq!(inner.inner.calls(), 2, "one GET per window");

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A prefetch waits for a run a read is fetching, then fetches only the rest.
    #[tokio::test]
    async fn prefetch_waits_for_a_run_a_read_is_fetching() {
        // One unit is one short read's GET window.
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );
        let bf = src.block_file().expect("block file");

        let mut read = pin!(src.range(0, 10));
        assert!(
            poll!(read.as_mut()).is_pending(),
            "the first window is being fetched"
        );
        let mut chunk = pin!(src.prefetch_chunk(bf, 0, 2 * MIN_GET_BLOCKS - 1));
        assert!(
            poll!(chunk.as_mut()).is_pending(),
            "the prefetch waits for it"
        );
        assert_eq!(inner.inner.calls(), 0, "without a GET of its own yet");

        inner.release();
        let chunk = timeout(NO_WAIT_DEADLINE, chunk)
            .await
            .expect("the prefetch finishes")
            .expect("prefetch chunk");
        assert!(chunk, "the prefetch goes on");
        assert_eq!(
            inner.inner.calls(),
            2,
            "the first window by the read, the rest by it"
        );
        let read = timeout(NO_WAIT_DEADLINE, read)
            .await
            .expect("the read finishes")
            .expect("read");
        assert_eq!(read, inner.inner.blob.slice(0..10));

        store.block_writes_settled().await;
        assert!(src.all_filled(0, 2 * MIN_GET_BLOCKS - 1));
        assert_eq!(store.stats().current_bytes, 2 * b);
        store.remove_block_entry_for_test(&uri);
    }

    /// A run a read fetched but could not cache (the cap was full) is filled by
    /// a prefetch waiting on it.
    #[tokio::test]
    async fn prefetch_fills_a_run_a_read_left_uncached() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );
        let bf = src.block_file().expect("block file");
        let mut taken = 0;
        while store.try_admit_write_behind(b) {
            taken += b;
        }

        let mut read = pin!(src.range(0, 10));
        assert!(
            poll!(read.as_mut()).is_pending(),
            "block 0 is being fetched"
        );
        let mut chunk = pin!(src.prefetch_chunk(bf, 0, 0));
        assert!(
            poll!(chunk.as_mut()).is_pending(),
            "the prefetch waits for it"
        );
        inner.release();
        let (read, chunk) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(read, chunk) })
            .await
            .expect("both finish");
        assert_eq!(read.expect("read"), inner.inner.blob.slice(0..10));
        assert!(chunk.expect("prefetch chunk"));
        assert_eq!(
            inner.inner.calls(),
            2,
            "the read's GET, then the prefetch's"
        );

        store.block_writes_settled().await;
        assert!(src.all_filled(0, 0), "the prefetch filled it");
        store.release_write_behind(taken);
        store.remove_block_entry_for_test(&uri);
    }

    /// A GET that returns fewer bytes than asked (the object changed) is passed
    /// through but never cached, by a read or by a prefetch.
    #[tokio::test]
    async fn a_short_get_is_never_cached() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = Arc::new(ShortAt {
            inner: CountingSource::new(2 * b as usize),
            start: None,
        });
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let got = src.range(0, 64).await.expect("short read passes through");
        assert_eq!(got, inner.inner.blob.slice(0..63));
        src.prefetch(0, 2 * b, b, 1)
            .await
            .expect("a short prefetch stops quietly");

        store.block_writes_settled().await;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 0);
        assert_eq!(store.stats().current_bytes, 0, "nothing stays charged");
        assert_eq!(store.write_behind_in_use(), 0);
        assert!(!src.has_runs_in_memory());
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
        let _ = src.range(0, 10).await.expect("unit 0 into memory");
        // One unit per chunk, all three at once, so a prefetch that ignored
        // runs in memory would fetch unit 0 again. Its own writes are held too,
        // so let them go once units 1 and 2 are fetched.
        let release = async {
            wait_until(|| inner.calls() >= 3, "the prefetch fetches units 1 and 2").await;
            drop(stall);
        };
        let (prefetched, ()) = tokio::join!(src.prefetch(0, 3 * b, b, 3), release);
        prefetched.expect("prefetch around the run in memory");
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
        let inner = held_source(2 * b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );
        let bf = src.block_file().expect("block file");

        let stall = src.stall_writes();
        let read_meanwhile = async {
            wait_until(
                || inner.held.load(Ordering::Acquire) > 0,
                "the prefetch GETs",
            )
            .await;
            let _ = src.range(0, 10).await.expect("the read queues block 0");
            inner.release();
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

    /// A read whose window runs through another read's run in flight GETs it as
    /// it would alone, in one GET from its first missing block, waits only for
    /// what lies before that block, and its run is cached with every block
    /// charged once.
    #[tokio::test]
    async fn a_window_through_a_run_in_flight_is_one_get_and_cached() {
        // One unit is one short read's GET window.
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_sources(4 * b as usize, 2);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut zero = pin!(src.range(10, 10));
        let mut two = pin!(src.range(2 * b + 10, 10));
        assert!(
            poll!(zero.as_mut()).is_pending(),
            "window 0 is being fetched"
        );
        assert!(
            poll!(two.as_mut()).is_pending(),
            "window 2 is being fetched"
        );
        wait_until(|| inner.held.load(Ordering::Acquire) == 2, "both GETs hold").await;
        let mut all = pin!(src.range(0, 4 * b));
        assert!(
            poll!(all.as_mut()).is_pending(),
            "the wide read waits for window 0"
        );
        wait_until(|| inner.inner.calls() == 1, "the wide read GETs").await;
        assert_eq!(
            inner.inner.ranges(),
            vec![b..4 * b],
            "windows 1 to 3, in one go"
        );

        inner.release();
        let (zero, two, all) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(zero, two, all) })
            .await
            .expect("every read finishes");
        assert_eq!(zero.expect("window 0"), inner.inner.blob.slice(10..20));
        let two_at = (2 * b + 10) as usize;
        assert_eq!(
            two.expect("window 2"),
            inner.inner.blob.slice(two_at..two_at + 10)
        );
        assert_eq!(all.expect("wide read"), inner.inner.blob.slice(..));
        assert_eq!(inner.inner.calls(), 3, "one GET per run");

        store.block_writes_settled().await;
        assert!(
            src.all_filled(0, 4 * MIN_GET_BLOCKS - 1),
            "every block is cached"
        );
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 4 * b);
        assert_eq!(
            store.stats().current_bytes,
            4 * b,
            "each block charged once"
        );
        assert_eq!(store.write_behind_in_use(), 0);
        assert!(!src.has_runs_in_memory());
        store.remove_block_entry_for_test(&uri);
    }

    /// A read waiting on a run that starts before its first block slices its
    /// bytes at the right offset.
    #[tokio::test]
    async fn a_waiting_read_slices_a_longer_run_at_its_offset() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(3 * b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut wide = pin!(src.range(10, 2 * b));
        let (start, len) = (b + 5, b + 100);
        let mut inside = pin!(src.range(start, len));
        assert!(
            poll!(wide.as_mut()).is_pending(),
            "blocks 0 to 2 are being fetched"
        );
        assert!(poll!(inside.as_mut()).is_pending(), "the read inside waits");
        inner.release();
        let inside = timeout(NO_WAIT_DEADLINE, inside)
            .await
            .expect("the read inside finishes")
            .expect("read inside");
        assert_eq!(
            inside,
            inner
                .inner
                .blob
                .slice(start as usize..(start + len) as usize)
        );
        assert_eq!(inner.inner.calls(), 1);

        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A run the cap on unwritten runs refuses still reaches the reads waiting
    /// on it; only later reads GET it again.
    #[tokio::test]
    async fn a_run_the_cap_refuses_still_reaches_its_waiting_reads() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );
        let mut taken = 0;
        while store.try_admit_write_behind(b) {
            taken += b;
        }

        let mut one = pin!(src.range(0, 64));
        let mut two = pin!(src.range(10, 64));
        assert!(poll!(one.as_mut()).is_pending());
        assert!(poll!(two.as_mut()).is_pending());
        inner.release();
        let (one, two) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(one, two) })
            .await
            .expect("both reads finish");
        assert_eq!(one.expect("first read"), inner.inner.blob.slice(0..64));
        assert_eq!(two.expect("second read"), inner.inner.blob.slice(10..74));
        assert_eq!(inner.inner.calls(), 1, "one GET for both");
        assert!(!src.has_runs_in_memory(), "the run was not kept");
        assert_eq!(store.stats().current_bytes, 0);
        assert_eq!(store.write_behind_in_use(), taken);

        let _ = src.range(0, 64).await.expect("a later read");
        assert_eq!(inner.inner.calls(), 2, "GETs it again");
        store.release_write_behind(taken);
        store.remove_block_entry_for_test(&uri);
    }

    /// The sync path never waits for a run being fetched: it passes the read on.
    #[tokio::test]
    async fn the_sync_path_passes_on_a_run_being_fetched() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * CACHE_BLOCK_BYTES as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut read = pin!(src.range(0, 64));
        assert!(
            poll!(read.as_mut()).is_pending(),
            "block 0 is being fetched"
        );
        assert!(src.try_get_range_sync(10, 64).is_none(), "passed on");
        assert!(src.has_runs_in_memory(), "the run is still being fetched");
        inner.release();
        let read = timeout(NO_WAIT_DEADLINE, read)
            .await
            .expect("the read finishes")
            .expect("read");
        assert_eq!(read, inner.inner.blob.slice(0..64));
        assert_eq!(
            src.try_get_range_sync(10, 64),
            Some(inner.inner.blob.slice(10..74)),
            "now from memory or the block file",
        );
        assert_eq!(inner.inner.calls(), 1);
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A read one of whose runs comes back short goes to the source, and its runs
    /// that did land are still cached for the next read: window 0 is short,
    /// window 2 lands, then the read goes to the source.
    #[tokio::test]
    async fn a_short_run_sends_the_read_elsewhere_and_keeps_its_other_runs() {
        // One unit is one short read's GET window.
        let b = UNIT;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = Arc::new(ShortAt {
            inner: CountingSource::new(3 * b as usize),
            start: Some(0),
        });
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );
        let _ = src.range(b + 10, 10).await.expect("fill window 1");
        store.block_writes_settled().await;

        let calls = inner.inner.calls();
        let _ = src.range(0, 3 * b).await;
        assert_eq!(
            inner.inner.calls() - calls,
            3,
            "windows 0 and 2, then the source"
        );
        store.block_writes_settled().await;
        assert!(!src.has_runs_in_memory());
        assert_eq!(
            store.stats().current_bytes,
            2 * b,
            "windows 1 and 2 are cached"
        );
        store.remove_block_entry_for_test(&uri);
    }

    /// Returns one byte less than asked, for every range or only for the one
    /// starting at `start`, as if the object were replaced by a shorter one.
    struct ShortAt {
        inner: CountingSource,
        start: Option<u64>,
    }

    #[async_trait]
    impl LazyByteSource for ShortAt {
        fn size(&self) -> u64 {
            self.inner.size()
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            let bytes = self.inner.range(start, len).await?;
            Ok(if self.start.is_none_or(|short| short == start) {
                bytes.slice(..bytes.len().saturating_sub(1))
            } else {
                bytes
            })
        }
    }

    /// A read stuck behind another read's GET for longer than `HEDGE_AFTER` GETs
    /// the blocks itself, so one stuck GET can't hold it.
    #[tokio::test(start_paused = true)]
    async fn a_waiting_read_hedges_a_stuck_get() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * CACHE_BLOCK_BYTES as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut stuck = pin!(src.range(0, 64));
        assert!(
            poll!(stuck.as_mut()).is_pending(),
            "the first read's GET is stuck"
        );
        wait_real(|| inner.held.load(Ordering::Acquire) == 1, "the GET holds");
        let started = Instant::now();
        let hedged = src.range(10, 64).await.expect("the waiting read");
        assert_eq!(hedged, inner.inner.blob.slice(10..74));
        assert!(started.elapsed() >= HEDGE_AFTER, "it waited first");
        assert_eq!(inner.inner.calls(), 1, "then GET the block itself");

        inner.release();
        assert_eq!(
            stuck.await.expect("the first read"),
            inner.inner.blob.slice(0..64)
        );
        store.remove_block_entry_for_test(&uri);
    }

    /// A read whose wait ends under `HEDGE_AFTER` sends no GET of its own.
    #[tokio::test(start_paused = true)]
    async fn a_waiting_read_does_not_hedge_a_get_that_lands_in_time() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * CACHE_BLOCK_BYTES as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut one = pin!(src.range(0, 64));
        let mut two = pin!(src.range(10, 64));
        assert!(poll!(one.as_mut()).is_pending());
        assert!(poll!(two.as_mut()).is_pending());
        advance(HEDGE_AFTER - Duration::from_millis(1)).await;
        assert!(poll!(two.as_mut()).is_pending(), "still waiting");
        inner.release();
        // The GET runs on the I/O runtime, so this runtime looks idle meanwhile
        // and its paused clock jumps to the next timer: `two`'s hedge. Awaiting
        // `one` alone first keeps `two` from being polled until its fetch is done.
        assert_eq!(
            one.await.expect("first read"),
            inner.inner.blob.slice(0..64)
        );
        assert_eq!(
            two.await.expect("second read"),
            inner.inner.blob.slice(10..74)
        );
        assert_eq!(inner.inner.calls(), 1, "one GET for both");
        store.remove_block_entry_for_test(&uri);
    }

    /// Holds its first `hold` ranges until [`Self::release`]; `held` counts the
    /// ones holding. A held range then does what `ends` says.
    struct HeldSource {
        inner: CountingSource,
        hold: AtomicUsize,
        held: AtomicUsize,
        released: AtomicBool,
        let_go: Notify,
        ends: HeldEnd,
    }

    /// What a [`HeldSource`]'s held range does once let go.
    #[derive(Clone, Copy)]
    enum HeldEnd {
        /// Reads.
        Reads,
        /// Fails.
        Fails,
        /// Comes back one byte short, as if the object were replaced by a
        /// shorter one.
        Short,
        /// Panics with [`GET_PANIC`], as a bug in the GET would.
        Panics,
    }

    /// What a [`HeldEnd::Panics`] range panics with.
    const GET_PANIC: &str = "the held GET panicked";

    impl HeldSource {
        fn release(&self) {
            self.released.store(true, Ordering::Release);
            self.let_go.notify_waiters();
        }

        /// Lets go only the range that has held longest; the others hold on.
        fn release_first(&self) {
            self.let_go.notify_one();
        }
    }

    /// A [`HeldSource`] over an `obj`-byte blob that holds its first range.
    fn held_source(obj: usize) -> Arc<HeldSource> {
        held_sources(obj, 1)
    }

    /// A [`HeldSource`] over an `obj`-byte blob that holds its first `hold` ranges.
    fn held_sources(obj: usize, hold: usize) -> Arc<HeldSource> {
        held_ending(obj, hold, HeldEnd::Reads)
    }

    /// A [`HeldSource`] over an `obj`-byte blob that holds its first `hold`
    /// ranges, which then do what `ends` says.
    fn held_ending(obj: usize, hold: usize, ends: HeldEnd) -> Arc<HeldSource> {
        Arc::new(HeldSource {
            inner: CountingSource::new(obj),
            hold: AtomicUsize::new(hold),
            held: AtomicUsize::new(0),
            released: AtomicBool::new(false),
            let_go: Notify::new(),
            ends,
        })
    }

    #[async_trait]
    impl LazyByteSource for HeldSource {
        fn size(&self) -> u64 {
            self.inner.size()
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            let held = self
                .hold
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
                .is_ok();
            if held {
                self.held.fetch_add(1, Ordering::AcqRel);
                let mut let_go = pin!(self.let_go.notified());
                let_go.as_mut().enable();
                if !self.released.load(Ordering::Acquire) {
                    timeout(NO_WAIT_DEADLINE, let_go)
                        .await
                        .expect("the held GET is let go");
                }
                match self.ends {
                    HeldEnd::Reads => {}
                    HeldEnd::Fails => {
                        return Err(LazyByteSourceError::OutOfBounds {
                            start,
                            len,
                            size: 0,
                        });
                    }
                    HeldEnd::Short => {
                        let bytes = self.inner.range(start, len).await?;
                        return Ok(bytes.slice(..bytes.len().saturating_sub(1)));
                    }
                    HeldEnd::Panics => panic_any(GET_PANIC),
                }
            }
            self.inner.range(start, len).await
        }
    }

    /// Writes the object's bytes for blocks `[b0, b1]` to `src`'s block file and
    /// marks them filled.
    fn fill_blocks(src: &BlockCachedSource, blob: &Bytes, b0: u64, b1: u64) {
        let b = CACHE_BLOCK_BYTES;
        let bf = src.block_file().expect("block file");
        bf.file
            .write_all_at(&blob[(b0 * b) as usize..((b1 + 1) * b) as usize], b0 * b)
            .expect("write blocks");
        src.mark_filled(bf.size, b0 as u32, b1 as u32);
    }

    /// The blocks each GET asked for, first and last, in block order.
    fn get_blocks(ranges: &[Range<u64>]) -> Vec<(u64, u64)> {
        let b = CACHE_BLOCK_BYTES;
        let mut blocks: Vec<(u64, u64)> = ranges
            .iter()
            .map(|range| (range.start / b, (range.end - 1) / b))
            .collect();
        // GETs run as tasks, in no set order.
        blocks.sort_unstable();
        blocks
    }

    /// A read next to another read's window in flight makes the GETs it would
    /// make alone, from their first missing block: here 2, where giving up on
    /// its windows would cost 3.
    ///
    /// ```text
    ///   block:  5  6  7  8  9  10 .. 17  18  19  20
    ///           M  F  M  M  M  [ in flight ]  M   F   M
    ///   GETs:   [ 5 .. 12 ]          [ 18 .. 20 ]   (13..17 waited for)
    /// ```
    #[tokio::test]
    async fn a_read_beside_a_window_in_flight_makes_the_gets_it_would_alone() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(32 * b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );
        fill_blocks(&src, &inner.inner.blob, 6, 6);
        fill_blocks(&src, &inner.inner.blob, 19, 19);

        let mut held = pin!(src.range(10 * b, 10));
        assert!(
            poll!(held.as_mut()).is_pending(),
            "window [10, 17] is in flight"
        );
        wait_until(|| inner.held.load(Ordering::Acquire) == 1, "its GET holds").await;
        let mut wide = pin!(src.range(5 * b, 16 * b));
        assert!(
            poll!(wide.as_mut()).is_pending(),
            "the wide read waits for it"
        );
        wait_until(|| inner.inner.calls() == 2, "the wide read GETs").await;
        assert_eq!(get_blocks(&inner.inner.ranges()), vec![(5, 12), (18, 20)]);

        inner.release();
        let (held, wide) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(held, wide) })
            .await
            .expect("both reads finish");
        held.expect("held read");
        assert_eq!(
            wide.expect("wide read"),
            inner.inner.blob.slice((5 * b) as usize..(21 * b) as usize)
        );
        assert_eq!(inner.inner.calls(), 3, "and the held window's own GET");
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A read's later window does not fetch again what its earlier one covers,
    /// in front of the aligned region: 19..23, not 17..23.
    #[tokio::test]
    async fn a_later_window_starts_after_what_the_read_has() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        // Group [16, 23] holds the region's first block, 19.
        let (uri, inner, src) =
            layout_source(dir.path(), &store, 32 * b as usize, Some((19 * b, 4 * b)));
        fill_blocks(&src, &inner.blob, 10, 10);
        fill_blocks(&src, &inner.blob, 17, 18);

        let got = src.range(9 * b, 14 * b).await.expect("read");
        assert_eq!(got, inner.blob.slice((9 * b) as usize..(23 * b) as usize));
        assert_eq!(get_blocks(&inner.ranges()), vec![(9, 16), (19, 23)]);
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// As above, with blocks 17..18 fetched but not written yet: the read still
    /// makes its two windowed GETs.
    #[tokio::test]
    async fn a_run_waiting_for_its_write_does_not_split_a_later_window() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let (uri, inner, src) =
            layout_source(dir.path(), &store, 32 * b as usize, Some((19 * b, 4 * b)));
        fill_blocks(&src, &inner.blob, 10, 10);
        let stall = src.stall_writes();
        let bf = src.block_file().expect("block file");
        let mut chunk = pin!(src.prefetch_chunk(bf, 17, 18));
        wait_until(
            || {
                let _ = (&mut chunk).now_or_never();
                src.stalled_writes() == 1
            },
            "blocks 17..18 wait for their write",
        )
        .await;

        let got = src.range(9 * b, 14 * b).await.expect("read");
        assert_eq!(got, inner.blob.slice((9 * b) as usize..(23 * b) as usize));
        assert_eq!(get_blocks(&inner.ranges()[1..]), vec![(9, 16), (19, 23)]);

        drop(stall);
        assert!(chunk.await.expect("prefetch"));
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A read that waits on another read's window hedges the blocks it needs,
    /// from where they start, not from the window's start.
    #[tokio::test(start_paused = true)]
    async fn a_hedge_inside_a_stuck_window_gets_the_right_bytes() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(8 * b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut stuck = pin!(src.range(0, 64));
        assert!(poll!(stuck.as_mut()).is_pending(), "window [0, 7] is stuck");
        wait_real(|| inner.held.load(Ordering::Acquire) == 1, "the GET holds");
        let hedged = src.range(3 * b + 10, 64).await.expect("the waiting read");
        assert_eq!(
            hedged,
            inner
                .inner
                .blob
                .slice((3 * b + 10) as usize..(3 * b + 74) as usize)
        );
        assert_eq!(get_blocks(&inner.inner.ranges()), vec![(3, 3)]);

        inner.release();
        stuck.await.expect("the stuck read");
        store.remove_block_entry_for_test(&uri);
    }

    /// The read that sent a GET waits for it however long it takes: hedging is
    /// for reads that joined someone else's.
    #[tokio::test(start_paused = true)]
    async fn a_read_never_hedges_its_own_get() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * CACHE_BLOCK_BYTES as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut stuck = pin!(src.range(0, 64));
        assert!(poll!(stuck.as_mut()).is_pending());
        advance(3 * HEDGE_AFTER).await;
        assert!(poll!(stuck.as_mut()).is_pending(), "still waiting");
        assert_eq!(inner.inner.calls(), 0, "with no GET of its own");

        inner.release();
        stuck.await.expect("the read");
        assert_eq!(inner.inner.calls(), 1);
        store.remove_block_entry_for_test(&uri);
    }

    /// Blocks being fetched are not local: a caller that asks gets them through
    /// an async read, which waits for the GET.
    #[tokio::test]
    async fn a_run_being_fetched_is_not_local() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * CACHE_BLOCK_BYTES as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut read = pin!(src.range(0, 64));
        assert!(poll!(read.as_mut()).is_pending());
        assert!(!src.is_local(10, 10), "in flight");
        inner.release();
        read.await.expect("the read");
        assert!(src.is_local(10, 10), "in memory once fetched");
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// Ranges sleep `delay` on the timer of the runtime that polls them, like an
    /// HTTP request's timeout. A range is counted when it starts, so one cancelled
    /// mid-sleep still counts.
    struct SlowSource {
        inner: CountingSource,
        delay: Duration,
    }

    #[async_trait]
    impl LazyByteSource for SlowSource {
        fn size(&self) -> u64 {
            self.inner.size()
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            let bytes = self.inner.range(start, len).await;
            sleep(self.delay).await;
            bytes
        }
    }

    /// A sync read on a throwaway runtime sends a GET and gives up on it, and
    /// its runtime is gone; another read waiting on that GET still gets it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_shared_get_outlives_the_runtime_of_the_read_that_sent_it() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = Arc::new(SlowSource {
            inner: CountingSource::new(2 * CACHE_BLOCK_BYTES as usize),
            delay: Duration::from_millis(300),
        });
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let sender = Arc::clone(&src);
        let gave_up = thread::spawn(move || {
            // No runtime on this thread: the bridge builds a throwaway one.
            bridge_sync_to_async_send(async move {
                timeout(Duration::from_millis(100), sender.range(0, 64))
                    .await
                    .is_err()
            })
        });
        wait_until(|| src.has_runs_in_memory(), "the sync read sends its GET").await;
        let waiter = Arc::clone(&src);
        let waiting = tokio::spawn(async move { waiter.range(10, 64).await });
        assert!(gave_up.join().expect("sync read"), "the sync read gave up");

        let got = timeout(NO_WAIT_DEADLINE, waiting)
            .await
            .expect("the waiting read finishes")
            .expect("and does not panic")
            .expect("the waiting read");
        assert_eq!(got, inner.inner.blob.slice(10..74));
        assert_eq!(inner.inner.calls(), 1, "from the one GET");
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A GET that panics panics every read on it, and still ends their shared
    /// fetch: a panic inside the `Shared` would poison it without waking the
    /// reads that polled it meanwhile, and they would wait forever.
    #[tokio::test]
    async fn a_get_that_panics_panics_every_read_on_it() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_ending(2 * CACHE_BLOCK_BYTES as usize, 1, HeldEnd::Panics);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut sent = pin!(src.range(0, 64));
        assert!(poll!(sent.as_mut()).is_pending(), "its GET holds");
        let fetch = match src.segments(0, 0).as_slice() {
            [Segment::Fetching { fetch, .. }] => fetch.clone(),
            _ => panic!("the read's run is in flight"),
        };
        let mut waiting = pin!(src.range(10, 64));
        assert!(
            poll!(waiting.as_mut()).is_pending(),
            "a read waits on that GET"
        );

        inner.release();
        let panic = AssertUnwindSafe(sent)
            .catch_unwind()
            .await
            .expect_err("the read that sent the GET panics");
        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&GET_PANIC),
            "with the GET's panic"
        );
        assert!(
            AssertUnwindSafe(waiting).catch_unwind().await.is_err(),
            "so does the read waiting on it"
        );
        assert!(fetch.peek().is_some(), "and their fetch ended");
        store.remove_block_entry_for_test(&uri);
    }

    /// The claim of a run every read dropped, dropping only once a new run took
    /// its first block, leaves the new run alone: reads still find it in flight
    /// and share its GET.
    #[tokio::test]
    async fn a_stale_claim_leaves_a_new_run_at_its_block_alone() {
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(2 * CACHE_BLOCK_BYTES as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );
        let stale = Claim {
            source: Arc::clone(&src),
            first: 0,
            id: src.next_run_id.fetch_add(1, Ordering::Relaxed),
            handed_on: false,
        };

        let mut sent = pin!(src.range(0, 64));
        assert!(
            poll!(sent.as_mut()).is_pending(),
            "a new run at block 0 is in flight"
        );
        drop(stale);
        let mut waiting = pin!(src.range(10, 64));
        assert!(
            poll!(waiting.as_mut()).is_pending(),
            "a read still finds it in flight"
        );

        inner.release();
        let (sent, waiting) = timeout(NO_WAIT_DEADLINE, async { tokio::join!(sent, waiting) })
            .await
            .expect("both reads finish");
        sent.expect("the read that sent the GET");
        assert_eq!(
            waiting.expect("the read waiting on it"),
            inner.inner.blob.slice(10..74)
        );
        assert_eq!(inner.inner.calls(), 1, "one GET for both");
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A block on the block file inside another read's window in flight is read
    /// from the file: nothing waits on that GET for it.
    #[tokio::test]
    async fn a_filled_block_inside_a_window_in_flight_is_not_waited_for() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(16 * b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );
        fill_blocks(&src, &inner.inner.blob, 1, 1);

        let mut held = pin!(src.range(10, 10));
        assert!(
            poll!(held.as_mut()).is_pending(),
            "window [0, 7] is in flight"
        );
        wait_until(|| inner.held.load(Ordering::Acquire) == 1, "its GET holds").await;

        assert!(
            matches!(
                src.segments(1, 2).as_slice(),
                [
                    Segment::Filled { first: 1, last: 1 },
                    Segment::Fetching {
                        first: 2,
                        last: 2,
                        ..
                    }
                ]
            ),
            "block 1 is on the file, block 2 in flight"
        );
        let bf = src.block_file().expect("block file");
        let prefetched = timeout(STAYS_PUT, src.prefetch_chunk(bf, 1, 1))
            .await
            .expect("a prefetch of block 1 does not wait for the GET");
        assert!(prefetched.expect("the prefetch"), "and finds it filled");

        inner.release();
        held.await.expect("the held read");
        assert_eq!(
            get_blocks(&inner.inner.ranges()),
            vec![(0, 7)],
            "the window ran through filled block 1"
        );
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A prefetch whose GET lands while a read's run at the GET's first block is
    /// still in flight does not take its chunk as done: it waits for that run,
    /// then fetches what is still missing.
    #[tokio::test]
    async fn a_prefetch_beaten_to_its_first_block_still_fills_its_chunk() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_sources(32 * b as usize, 2);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let prefetching = Arc::clone(&src);
        let chunk = tokio::spawn(async move {
            let bf = prefetching.block_file().expect("block file");
            prefetching.prefetch_chunk(bf, 0, 15).await
        });
        wait_until(
            || inner.held.load(Ordering::Acquire) == 1,
            "the prefetch's GET of [0, 15] holds",
        )
        .await;
        let mut read = pin!(src.range(0, 64));
        assert!(poll!(read.as_mut()).is_pending(), "a read registers [0, 7]");
        wait_until(
            || inner.held.load(Ordering::Acquire) == 2,
            "the read's GET holds",
        )
        .await;

        // The prefetch's GET lands first; the read's run still holds block 0.
        inner.release_first();
        wait_until(|| inner.inner.calls() == 1, "the prefetch's GET lands").await;
        assert_eq!(get_blocks(&inner.inner.ranges()), vec![(0, 15)]);
        sleep(STAYS_PUT).await;
        assert!(
            !chunk.is_finished(),
            "the prefetch waits for the read's run"
        );

        inner.release();
        read.await.expect("the read");
        let chunk = timeout(NO_WAIT_DEADLINE, chunk)
            .await
            .expect("the prefetch finishes")
            .expect("and does not panic");
        assert!(chunk.expect("the prefetch"), "the prefetch goes on");
        assert_eq!(
            get_blocks(&inner.inner.ranges()),
            vec![(0, 7), (0, 15), (8, 15)],
            "and fetches what the read's run left missing"
        );
        store.block_writes_settled().await;
        assert!(src.all_filled(0, 15), "every block of the chunk is cached");
        store.remove_block_entry_for_test(&uri);
    }

    /// A read whose blocks all lie in another read's window in flight registers
    /// nothing: its own window would hold none of its missing blocks, and
    /// dropping it would cancel a GET already sent.
    #[tokio::test]
    async fn a_read_inside_a_window_in_flight_sends_no_window_of_its_own() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(16 * b as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut held = pin!(src.range(0, 64));
        assert!(
            poll!(held.as_mut()).is_pending(),
            "window [0, 7] is in flight"
        );
        wait_until(|| inner.held.load(Ordering::Acquire) == 1, "its GET holds").await;
        let mut inside = pin!(src.range(3 * b + 10, 64));
        assert!(poll!(inside.as_mut()).is_pending(), "the read inside waits");
        assert_eq!(
            src.next_run_id.load(Ordering::Acquire),
            1,
            "and registers no window"
        );

        inner.release();
        held.await.expect("held read");
        assert_eq!(
            inside.await.expect("read inside"),
            inner
                .inner
                .blob
                .slice((3 * b + 10) as usize..(3 * b + 74) as usize)
        );
        assert_eq!(inner.inner.calls(), 1, "one GET for both");
        store.block_writes_settled().await;
        store.remove_block_entry_for_test(&uri);
    }

    /// A read waits longer on a big GET before it hedges: a GET that is only
    /// slow for its size is not fetched twice.
    #[tokio::test(start_paused = true)]
    async fn a_big_get_is_given_longer_before_a_hedge() {
        let big = 8 << 20;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_source(big as usize);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut stuck = pin!(src.range(0, big));
        assert!(poll!(stuck.as_mut()).is_pending(), "the big GET is stuck");
        wait_real(|| inner.held.load(Ordering::Acquire) == 1, "the GET holds");
        let started = Instant::now();
        let hedged = src.range(10, 64).await.expect("the waiting read");
        assert_eq!(hedged, inner.inner.blob.slice(10..74));
        let after = BlockCachedSource::hedge_after(big);
        assert!(after > HEDGE_AFTER, "a big GET gets longer");
        assert!(
            started.elapsed() >= after,
            "the read waited that long first"
        );
        assert_eq!(inner.inner.calls(), 1, "then GET its block itself");

        inner.release();
        stuck.await.expect("the big read");
        store.remove_block_entry_for_test(&uri);
    }

    /// A read whose shared GET comes back short (another read's window) GETs
    /// and caches its own blocks, as it would have without waiting, instead of
    /// giving up on them.
    #[tokio::test]
    async fn a_read_whose_shared_get_ends_elsewhere_fetches_its_own_blocks() {
        let b = CACHE_BLOCK_BYTES;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let inner = held_ending(8 * b as usize, 1, HeldEnd::Short);
        let (uri, src) = installed_source(
            dir.path(),
            &store,
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
        );

        let mut owner = pin!(src.range(0, 64));
        assert!(
            poll!(owner.as_mut()).is_pending(),
            "window [0, 7] is in flight"
        );
        wait_until(|| inner.held.load(Ordering::Acquire) == 1, "its GET holds").await;
        let mut waiting = pin!(src.range(3 * b + 10, 64));
        assert!(poll!(waiting.as_mut()).is_pending(), "the other read waits");

        inner.release();
        owner.await.expect("the owner reads elsewhere");
        assert_eq!(
            waiting.await.expect("the waiting read"),
            inner
                .inner
                .blob
                .slice((3 * b + 10) as usize..(3 * b + 74) as usize)
        );
        assert!(
            inner.inner.ranges().contains(&(3 * b..4 * b)),
            "the waiting read GETs its block"
        );
        store.block_writes_settled().await;
        assert!(src.all_filled(3, 3), "and caches it");
        store.remove_block_entry_for_test(&uri);
    }
}
