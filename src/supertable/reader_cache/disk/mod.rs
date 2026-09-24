// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`DiskCacheStore`] — Tier 1 cache wrapping a [`StorageProvider`] with parallel cold-fetch and LRU
//! eviction. Split across sibling files: `read` (open a superfile), `fetch` (cold-fetch it in), `budget`
//! (reserve and evict), `store_files` (the on-disk directory), and `sources` (mmap).

use std::{
    collections::HashSet,
    fmt, fs,
    path::PathBuf,
    sync::{
        Arc, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant, UNIX_EPOCH},
};

use dashmap::DashMap;
use memmap2::Mmap;
use thiserror::Error;
use tokio::sync::{Notify, OnceCell, Semaphore};

use crate::{
    storage::{StorageError, StorageProvider},
    superfile::reader::SuperfileReader,
    supertable::{
        manifest::SuperfileUri,
        reader_cache::{
            block_source::BlockCachedSource,
            config::{ColdFetchMode, DiskCacheConfig},
        },
    },
};

/// Parquet footer tail-speculation length for cold opens. Must match
/// `SuperfileReader::open_lazy_with` so the cold-fetch overlay covers
/// the entire upcoming `source.tail()` read.
const PARQUET_TAIL_SPEC_BYTES: u64 = 64 * 1024;

/// Fallback vector-subsection open-range length when the manifest
/// carries only a `(offset, len)` hint without explicit open ranges.
/// Enough bytes to parse the vector outer header; the reader then
/// discovers the rest.
const VECTOR_OPEN_HEADER_FALLBACK_BYTES: u64 = 32;

/// Fallback FTS open-range length under the same conditions as
/// [`VECTOR_OPEN_HEADER_FALLBACK_BYTES`]. Enough to parse the FTS
/// blob header.
const FTS_OPEN_HEADER_FALLBACK_BYTES: u64 = 48;

/// Poll cadence while waiting for another task to mmap-promote a
/// superfile. Short so the waiter picks up the promotion promptly
/// without busy-spinning.
const MMAP_PROMOTION_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Yield cadence while a background fill waits for its foreground reader.
const STORE_UPGRADE_RETRY_INTERVAL: Duration = Duration::from_millis(10);

/// Filename suffix for per-superfile sparse block-cache files.
const BLOCKS_FILE_SUFFIX: &str = ".blocks";

/// Filename suffix (appended after `.blocks`) for the persisted filled-block
/// index sidecar.
const BLOCKS_IDX_SUFFIX: &str = ".idx";

const BLOCKS_IDX_TMP_INFIX: &str = ".idx.tmp.";

/// How old an in-flight tempfile must be before the open-time scan reclaims it. A live cold
/// fetch's tempfile is seconds old, so one this stale can only be a crashed fetch's leftover;
/// deleting a live one would fail the owner's rename and cost it a retried fetch.
const TMP_RECLAIM_AGE: Duration = Duration::from_secs(15 * 60);

/// Process-global count of in-flight foreground queries. Used with
/// [`foreground_notify`] so a fill's `select!` wakes promptly when a query
/// begins and can re-check its per-URI pause condition; it is **not** a
/// process-wide pause signal (unrelated URI fills keep running).
static FOREGROUND_QUERIES: AtomicU64 = AtomicU64::new(0);
/// Wakes background fills so they re-check per-URI quiescence when a
/// foreground query arrives.
static FOREGROUND_NOTIFY: OnceLock<Notify> = OnceLock::new();

fn foreground_notify() -> &'static Notify {
    FOREGROUND_NOTIFY.get_or_init(Notify::new)
}

/// RAII guard marking a foreground query in flight for its lifetime.
///
/// Entering the guard notifies waiting fills so a same-URI fill can yield
/// to lazy query reads. Unrelated URI fills are not paused by this guard —
/// only by that URI's own reader hold ([`reader_blocks_background_fill`]).
pub struct ForegroundQueryGuard(());

impl ForegroundQueryGuard {
    pub fn enter() -> Self {
        FOREGROUND_QUERIES.fetch_add(1, Ordering::AcqRel);
        foreground_notify().notify_waiters();
        ForegroundQueryGuard(())
    }
}

impl Drop for ForegroundQueryGuard {
    fn drop(&mut self) {
        FOREGROUND_QUERIES.fetch_sub(1, Ordering::AcqRel);
        // Wake fills waiting on the notify so they can resume after the
        // query releases same-URI readers.
        foreground_notify().notify_waiters();
    }
}

/// File mtime in microseconds since the unix epoch, 0 if the filesystem has none.
fn file_mtime_us(meta: &fs::Metadata) -> u64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// Pause this URI's background fill while a caller besides the cache entry holds its lazy reader
/// (`strong_count > 1`). Other URIs are unaffected.
fn reader_blocks_background_fill(reader: &Weak<SuperfileReader>) -> bool {
    reader.strong_count() > 1
}

/// Errors surfaced by [`DiskCacheStore::open_for_query`] and its sibling reader entry points.
#[derive(Debug, Error)]
pub enum DiskCacheError {
    #[error("storage error during cold fetch")]
    Storage(#[from] StorageError),
    #[error("local filesystem error: {0}")]
    Io(#[from] std::io::Error),
    #[error("superfile reader failed to open mmap'd bytes: {0}")]
    SuperfileOpen(String),
    /// The cached / freshly-fetched superfile bytes failed to
    /// parse. The source [`crate::superfile::ReadError`] chain is
    /// preserved so callers that want variant-level detail can
    /// match on it instead of a stringified message.
    #[error("superfile reader failed to open bytes")]
    SuperfileOpenRead(#[from] crate::superfile::ReadError),
    /// Eviction could not free enough space: every cached entry is pinned, or the incoming
    /// superfile alone exceeds the budget. [`DiskCacheStore::open_for_query`] degrades to an
    /// uncached range-only reader on this; whole-file reads surface it.
    #[error("disk cache budget exceeded with no eligible victims")]
    BudgetExceeded,
    /// An invalid or conflicting configuration was supplied.
    #[error("config: {0}")]
    Config(String),
}

/// Live cache entry: the cached [`SuperfileReader`] plus the [`Residency`] that says where its
/// bytes currently live. Dropping the last `Arc<SuperfileReader>` (evicted, no in-flight query)
/// releases whatever backs it, whether that is a heap buffer, an mmap, or the block source.
///
/// In-flight queries pin the reader independently, so the cache can evict the entry and unlink the
/// file while a query still reads the mmapped bytes. POSIX keeps the mapping valid until the last
/// reference drops.
pub(crate) struct CachedEntry {
    reader: Arc<SuperfileReader>,
    residency: Residency,
    /// Bytes charged to the budget. Fixed at insertion for [`EntryAccounting::Eager`]; for
    /// [`EntryAccounting::SourceOwned`] it tracks the block source's live filled-bytes counter.
    size_bytes: Arc<AtomicU64>,
    /// Who releases `size_bytes` when the entry drops.
    accounting: EntryAccounting,
    last_access_us: AtomicU64,
}

/// What a read needs locally. It decides two things: whether a cached entry is a hit (a whole
/// file serves every intent, a lazy [`Residency::Paged`] entry serves `Stream` and `Warm` but not
/// `Load`), and how a miss is fetched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadIntent {
    /// Read ranges on demand through the block cache and never download the whole file. A whole
    /// file already on local disk is still used, since it costs no GETs. Vector search.
    Stream,
    /// Serve now from ranges, and download the whole file in the background toward a local mmap.
    /// FTS and SQL queries, and other non-vector reads.
    Warm,
    /// Download the whole file and mmap it before serving; a lazy entry is never enough. Every
    /// compaction input, user table or hidden index, since the merge reads all of it.
    Load,
}

/// Where a cached entry's bytes live. `Mapped` and `Buffered` hold the whole file; `Paged` holds
/// only the ranges read so far.
enum Residency {
    /// Whole superfile in an anonymous heap buffer, and the only copy. A background task writes it
    /// to disk and promotes it to [`Residency::Mapped`].
    Buffered,
    /// Superfile mmapped from the local cache file. `vector_source` is `Some` when the vector blob
    /// was left out of the file (vector search reads only a few clusters of it) and is served from
    /// the block cache instead. Parquet and FTS always come from the mmap.
    Mapped {
        mmap: Arc<Mmap>,
        vector_source: Option<Arc<BlockCachedSource>>,
    },
    /// Byte ranges fetched on demand from object storage and cached locally in blocks.
    /// `fill_spawned` latches the single background download so it starts at most once.
    Paged {
        block_source: Arc<BlockCachedSource>,
        fill_spawned: AtomicBool,
    },
}

impl CachedEntry {
    /// The mmap when this entry is [`Residency::Mapped`], for the idle-page sweep's `madvise`.
    fn mmap(&self) -> Option<&Arc<Mmap>> {
        match &self.residency {
            Residency::Mapped { mmap, .. } => Some(mmap),
            _ => None,
        }
    }

    /// Whether the whole file is mmapped locally, so the hot path never reads object storage.
    fn is_mapped(&self) -> bool {
        matches!(self.residency, Residency::Mapped { .. })
    }

    /// Whether the whole file is local, [`Residency::Mapped`] or [`Residency::Buffered`], so reads
    /// need no GETs.
    fn has_whole_file(&self) -> bool {
        matches!(
            self.residency,
            Residency::Mapped { .. } | Residency::Buffered
        )
    }

    /// The live block source: a [`Residency::Paged`] entry's source, or the retained vector hole of
    /// a [`Residency::Mapped`] entry.
    fn block_source(&self) -> Option<&Arc<BlockCachedSource>> {
        match &self.residency {
            Residency::Paged { block_source, .. } => Some(block_source),
            Residency::Mapped { vector_source, .. } => vector_source.as_ref(),
            Residency::Buffered => None,
        }
    }

    /// The background-fill latch when this entry is [`Residency::Paged`].
    fn fill_spawned(&self) -> Option<&AtomicBool> {
        match &self.residency {
            Residency::Paged { fill_spawned, .. } => Some(fill_spawned),
            _ => None,
        }
    }

    /// Every byte this entry keeps charged: its own `size_bytes`, plus the blocks a retained vector
    /// source charges for itself.
    #[cfg(test)]
    fn charged_bytes(&self) -> u64 {
        let own = self.size_bytes.load(Ordering::Acquire);
        match &self.residency {
            Residency::Mapped {
                vector_source: Some(source),
                ..
            } if source.owns_accounting() => {
                own + source.filled_bytes_handle().load(Ordering::Acquire)
            }
            _ => own,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryAccounting {
    /// Store-reserved entry; removal releases `size_bytes`.
    Eager,
    /// Block-source-reserved entry; source drop releases bytes.
    SourceOwned,
}

/// One in-flight walk for a URI: concurrent readers share the `OnceCell` and its result.
type Coordinator = Arc<OnceCell<Result<Arc<CachedEntry>, DiskCacheError>>>;

/// Snapshot of the disk cache's load. Surfaced via
/// [`DiskCacheStore::stats`] for the supertable's
/// observability hook and for tests that need to assert on
/// cache state.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub n_entries: u64,
    pub current_bytes: u64,
    pub budget_bytes: u64,
    pub n_cold_fetches: u64,
    /// Reads served by adopting a cache file that was on disk but not yet indexed.
    pub n_disk_reuses: u64,
    pub n_evictions: u64,
    /// Cumulative count of entries `madvise(MADV_DONTNEED)`'d
    /// by the idle-threshold sweep thread. Includes individual
    /// `sweep_once()` invocations.
    pub n_madvise_calls: u64,
    /// Total count of entries dropped because GC deleted from objectstore.
    pub n_gc_drops: u64,
}

/// A cache file found on disk at open but not opened yet. Its bytes already count against the budget.
#[derive(Debug, Clone, Copy)]
struct UnindexedFile {
    size_bytes: u64,
    /// File mtime in microseconds since the unix epoch; eviction drops the oldest first. mtime and
    /// not atime: relatime/noatime mounts make atime undependable, and for a file no read has
    /// opened the two match anyway, so oldest-mtime is oldest-copy, the likeliest dead one.
    mtime_us: u64,
}

/// Pulls superfile bytes through a [`StorageProvider`] and caches them locally as mmap-backed
/// `SuperfileReader`s. Construction is sync; the reads ([`Self::open_for_query`] and its siblings)
/// are async, since a miss fetches through the provider's async interface.
pub struct DiskCacheStore {
    storage: Arc<dyn StorageProvider>,
    config: DiskCacheConfig,
    started_at: Instant,
    cached: DashMap<SuperfileUri, Arc<CachedEntry>>,
    /// One in-flight walk per URI (tiers 2 to 4). The first caller to miss creates the cell and
    /// runs the walk; later callers await it. Removed once the walk settles.
    coordinators: DashMap<SuperfileUri, Coordinator>,
    /// Files on disk that no read has opened yet. Filled by `scan_cache_root`, drained by reuse or eviction.
    unindexed: DashMap<SuperfileUri, UnindexedFile>,
    block_files: DashMap<SuperfileUri, UnindexedFile>,
    current_bytes: AtomicU64,
    /// Live disk budget in bytes, seeded from `config.disk_budget_bytes`.
    /// An engine-managed (auto-sized) budget is raised — never lowered —
    /// by [`Self::reconcile_budget_floor`] as the table's on-storage
    /// footprint grows (the hidden vector index roughly doubles a vector
    /// table's working set after the drain). An explicitly configured
    /// budget never changes.
    budget_bytes: AtomicU64,
    /// Whether the budget is engine-managed (the user configured a cache
    /// directory but no byte budget). Set via
    /// [`Self::mark_budget_auto_sized`] at construction time.
    budget_auto_sized: AtomicBool,
    /// One-shot latch so an explicit budget smaller than the table
    /// footprint warns once, not on every reconcile.
    budget_warned: AtomicBool,
    n_cold_fetches: AtomicU64,
    n_disk_reuses: AtomicU64,
    n_evictions: AtomicU64,
    n_gc_drops: AtomicU64,
    n_madvise_calls: AtomicU64,
    /// Number of callers explicitly waiting for lazy background
    /// promotion. A waiter means promotion is now latency-critical,
    /// so the background task may start even if a lazy reader Arc is
    /// still held by the waiter.
    n_promotion_waiters: AtomicU64,
    /// Callback for "which URIs are currently pinned" — feeds
    /// the eviction policy.
    ///
    /// Interior mutability lets the supertable install a
    /// `Weak<SupertableInner>`-based closure after the cache
    /// is constructed and stashed in `SupertableOptions`.
    /// The closure can be swapped at any
    /// time via [`Self::set_pinned_fn`]; eviction loops
    /// clone the current `Arc<dyn Fn>` out from under the
    /// mutex and invoke it lock-free, so the mutex is held
    /// only for the Arc bump.
    pinned_fn: std::sync::Mutex<Arc<dyn Fn() -> HashSet<SuperfileUri> + Send + Sync>>,
    /// Global cap on concurrent background full-superfile fills.
    prefetch_semaphore: Arc<Semaphore>,
    /// Numbers each download's tempfile, see [`Self::tmp_path`].
    tmp_seq: AtomicU64,
}

impl fmt::Debug for DiskCacheStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiskCacheStore")
            .field("cache_root", &self.config.cache_root)
            .field("budget_bytes", &self.disk_budget_bytes())
            .field("current_bytes", &self.current_bytes.load(Ordering::Acquire))
            .field("n_entries", &self.cached.len())
            .field(
                "n_cold_fetches",
                &self.n_cold_fetches.load(Ordering::Acquire),
            )
            .finish()
    }
}

mod budget;
mod fetch;
mod read;
mod sources;
mod store_files;

pub(crate) use fetch::skip_background_fill;
pub(crate) use sources::{ArcMmapOwner, mmap_readonly_bytes};

impl DiskCacheStore {
    /// Construct a new disk cache rooted at `config.cache_root`
    /// (created if absent) backed by `storage`. `pinned_fn`
    /// returns the currently-pinned URI set on each eviction
    /// invocation — pass a `HashSet::new`-returning closure
    /// for the "nothing pinned" case (tests / standalone).
    pub fn new(
        storage: Arc<dyn StorageProvider>,
        config: DiskCacheConfig,
        pinned_fn: Arc<dyn Fn() -> HashSet<SuperfileUri> + Send + Sync>,
    ) -> Result<Arc<Self>, DiskCacheError> {
        if config.cold_fetch_mode == ColdFetchMode::RangeOnly {
            return Err(DiskCacheError::Config(
                "range_only does not currently use a disk cache; \
                 omit cache_dir or choose a different cold_fetch_mode"
                    .into(),
            ));
        }
        fs::create_dir_all(&config.cache_root)?;
        let threshold_secs = config.mmap_cold_threshold_secs;
        let interval_secs = config.mmap_sweep_interval_secs.max(1);
        let configured_budget = config.disk_budget_bytes;
        let prefetch_semaphore = Arc::new(Semaphore::new(config.prefetch_concurrency.max(1)));
        let store = Arc::new(Self {
            storage,
            config,
            started_at: Instant::now(),
            cached: DashMap::new(),
            coordinators: DashMap::new(),
            unindexed: DashMap::new(),
            block_files: DashMap::new(),
            current_bytes: AtomicU64::new(0),
            budget_bytes: AtomicU64::new(configured_budget),
            budget_auto_sized: AtomicBool::new(false),
            budget_warned: AtomicBool::new(false),
            n_cold_fetches: AtomicU64::new(0),
            n_disk_reuses: AtomicU64::new(0),
            n_evictions: AtomicU64::new(0),
            n_gc_drops: AtomicU64::new(0),
            n_madvise_calls: AtomicU64::new(0),
            n_promotion_waiters: AtomicU64::new(0),
            pinned_fn: std::sync::Mutex::new(pinned_fn),
            prefetch_semaphore,
            tmp_seq: AtomicU64::new(0),
        });

        // Record what is already on disk so the budget is correct from the start. Files are opened
        // lazily, by the reads that need them.
        store.scan_cache_root();

        // Idle-threshold sweep thread. Library-not-service
        // shape: holds a Weak<Self> and exits naturally when the last Arc
        // drops (no explicit shutdown signal needed; `Drop
        // for DiskCacheStore` is the visible exit).
        //
        // `std::thread::spawn` rather than `tokio::spawn` —
        // the sweep is a sync `madvise` syscall over a short
        // list of mmaps, doesn't need an async runtime, and
        // works even for embedders that haven't installed a
        // Tokio runtime on the calling thread.
        if threshold_secs > 0 {
            let weak = Arc::downgrade(&store);
            let _ = thread::Builder::new()
                .name("infino-disk-cache-sweep".into())
                .spawn(move || {
                    loop {
                        thread::sleep(Duration::from_secs(interval_secs));
                        match weak.upgrade() {
                            None => break,
                            Some(strong) => {
                                strong.sweep_once();
                            }
                        }
                    }
                });
            // Drop the JoinHandle — the thread runs to natural
            // exit when the Weak upgrade fails. Tests + drop
            // both finalize cleanly because the OS reclaims
            // the thread on process exit; explicit join isn't
            // required for correctness.
        }

        Ok(store)
    }

    /// Construct with a "nothing pinned" callback. Useful for
    /// tests and standalone-cache use.
    pub fn new_unpinned(
        storage: Arc<dyn StorageProvider>,
        config: DiskCacheConfig,
    ) -> Result<Arc<Self>, DiskCacheError> {
        Self::new(storage, config, Arc::new(HashSet::new))
    }

    /// Storage used for cold fetch when the caller does not override it.
    pub(crate) fn resolve_storage(
        &self,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Arc<dyn StorageProvider> {
        storage
            .map(Arc::clone)
            .unwrap_or_else(|| Arc::clone(&self.storage))
    }

    /// Snapshot of the cache's load. Cheap; reads atomics +
    /// a `DashMap::len` (which itself is `O(shards)`).
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            n_entries: self.cached.len() as u64,
            current_bytes: self.current_bytes.load(Ordering::Acquire),
            budget_bytes: self.disk_budget_bytes(),
            n_cold_fetches: self.n_cold_fetches.load(Ordering::Acquire),
            n_disk_reuses: self.n_disk_reuses.load(Ordering::Acquire),
            n_evictions: self.n_evictions.load(Ordering::Acquire),
            n_madvise_calls: self.n_madvise_calls.load(Ordering::Acquire),
            n_gc_drops: self.n_gc_drops.load(Ordering::Acquire),
        }
    }

    /// Current disk budget in bytes — the live value, not the
    /// construction-time config (see [`Self::reconcile_budget_floor`]).
    pub fn disk_budget_bytes(&self) -> u64 {
        self.budget_bytes.load(Ordering::Acquire)
    }

    /// Mark this cache's budget as engine-managed: the user configured a
    /// cache directory but no explicit byte budget, so the engine may
    /// raise (never lower) the budget as the table's on-storage footprint
    /// grows. Without this, a vector table silently outgrows any fixed
    /// default the moment the drain writes the hidden index — a second
    /// on-storage copy of the vector payload the user cannot be expected
    /// to size for.
    pub fn mark_budget_auto_sized(&self) {
        self.budget_auto_sized.store(true, Ordering::Release);
    }

    /// Reconcile the budget against the table's current on-storage
    /// footprint. `floor_bytes` is the caller-computed budget floor
    /// (footprint + headroom); `footprint_bytes` is the raw footprint,
    /// used for the undersized-budget warning.
    ///
    /// - **Auto-sized budget** ([`Self::mark_budget_auto_sized`]): raised
    ///   to `floor_bytes` when larger. Never lowered — shrinking under
    ///   live readers would force an eviction storm for no benefit.
    /// - **Explicit budget**: respected verbatim. If the footprint
    ///   exceeds it, warn once that steady-state reads will evict and
    ///   re-fetch instead of staying cache-resident.
    pub fn reconcile_budget_floor(&self, floor_bytes: u64, footprint_bytes: u64) {
        if self.budget_auto_sized.load(Ordering::Acquire) {
            let mut current = self.budget_bytes.load(Ordering::Acquire);
            while floor_bytes > current {
                match self.budget_bytes.compare_exchange_weak(
                    current,
                    floor_bytes,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(next) => current = next,
                }
            }
            return;
        }
        let budget = self.disk_budget_bytes();
        if footprint_bytes > budget && !self.budget_warned.swap(true, Ordering::AcqRel) {
            tracing::warn!(
                "disk cache budget ({budget} B) is below the table's on-storage footprint \
                 ({footprint_bytes} B, hidden vector index included): steady-state queries \
                 will evict and re-fetch. Raise ConnectOptions::with_cache_budget_bytes (or \
                 storage.disk_budget_bytes), or omit the budget to let the engine size it."
            );
        }
    }

    /// Replace the pinned-URI callback. Used by
    /// [`Supertable::create`](crate::supertable::Supertable::create)
    /// / [`Supertable::open`](crate::supertable::Supertable::open)
    /// to install a `Weak<SupertableInner>`-based closure
    /// after the cache has been moved into the supertable.
    /// The new closure takes effect on the next
    /// eviction sweep; in-flight evictions complete with the
    /// previous closure (we clone the `Arc` before invoking).
    ///
    /// Multi-supertable scenarios (one cache shared across
    /// supertables — uncommon, plan-allowed): only the most
    /// recent `set_pinned_fn` call wins. The closure can
    /// itself walk multiple `Weak<...>` references if a
    /// caller needs to pin URIs from several supertables.
    pub fn set_pinned_fn(&self, pinned_fn: Arc<dyn Fn() -> HashSet<SuperfileUri> + Send + Sync>) {
        let mut g = self.pinned_fn.lock().expect("pinned_fn mutex poisoned");
        *g = pinned_fn;
    }

    pub(crate) fn now_us(&self) -> u64 {
        self.started_at.elapsed().as_micros() as u64
    }

    /// Build a per-URI cache file path under `cache_root`.
    pub(crate) fn cache_path(&self, uri: &SuperfileUri) -> PathBuf {
        self.config.cache_root.join(uri.cache_filename())
    }

    /// Build a per-URI sparse block-cache path under `cache_root`.
    pub(crate) fn blocks_path(&self, uri: &SuperfileUri) -> PathBuf {
        self.config
            .cache_root
            .join(format!("{}{BLOCKS_FILE_SUFFIX}", uri.cache_filename()))
    }

    /// Path of the persisted filled-block index
    pub(crate) fn blocks_idx_path(&self, uri: &SuperfileUri) -> PathBuf {
        self.config.cache_root.join(format!(
            "{}{BLOCKS_FILE_SUFFIX}{BLOCKS_IDX_SUFFIX}",
            uri.cache_filename()
        ))
    }

    /// A fresh tempfile for one download of `uri`, renamed to `cache_path` once complete. Every
    /// call names a new file, since several downloads of one superfile can run at once.
    pub(crate) fn tmp_path(&self, uri: &SuperfileUri) -> PathBuf {
        let seq = self.tmp_seq.fetch_add(1, Ordering::Relaxed);
        self.config.cache_root.join(uri.cache_tmp_filename(seq))
    }

    // Test and bench helpers. Compiled only for tests and the `test-helpers` feature, never into
    // the shipped library.

    /// Whether `uri` has any cache entry at all, including a lazy [`Residency::Paged`] one. Use
    /// [`Self::is_mmap_promoted`] to test for a finished whole-file mmap.
    #[cfg(test)]
    pub(crate) fn is_cached(&self, uri: &SuperfileUri) -> bool {
        self.cached.contains_key(uri)
    }

    /// Whether `uri` is cached as a [`Residency::Mapped`] entry. False while it is still lazy or
    /// its background fill is in flight.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn is_mmap_promoted(&self, uri: &SuperfileUri) -> bool {
        self.cached.get(uri).map(|e| e.is_mapped()).unwrap_or(false)
    }

    /// The URIs the installed `pinned_fn` protects from eviction right now. The closure is called
    /// outside the mutex.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn current_pinned_uris(&self) -> HashSet<SuperfileUri> {
        let f = {
            let g = self.pinned_fn.lock().expect("pinned_fn mutex poisoned");
            Arc::clone(&g)
        };
        f()
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::{
        ops::Range,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
        time::{Duration, Instant},
    };

    use arrow_array::{LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use async_trait::async_trait;
    use bytes::Bytes;
    use object_store::MultipartUpload;
    use roaring::RoaringBitmap;
    use tempfile::TempDir;
    use tokio::{sync::Notify, time::sleep};

    use crate::{
        storage::{LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider},
        superfile::{
            BytesLazyByteSource, LazyByteSource,
            builder::{BuilderOptions, SuperfileBuilder},
        },
        supertable::{
            manifest::SuperfileUri,
            reader_cache::{
                block_source::{BlockCachedSource, CACHE_BLOCK_BYTES, serialize_index},
                config::DiskCacheConfig,
                disk::*,
            },
        },
        test_helpers::{decimal128_id_field, decimal128_ids, default_vector_config},
    };

    /// Storage that records every ranged GET, so a test can check which bytes left object storage.
    /// [`Self::pause_next_read`] parks the next ranged GET until [`Self::resume`], so a test can act
    /// in the middle of a download.
    #[derive(Debug)]
    pub(crate) struct RecordingStorage {
        inner: Arc<dyn StorageProvider>,
        ranges: Mutex<Vec<Range<u64>>>,
        pause_next: AtomicBool,
        paused: AtomicBool,
        resume: Notify,
    }

    impl RecordingStorage {
        pub(crate) fn over(inner: Arc<dyn StorageProvider>) -> Arc<Self> {
            Arc::new(Self {
                inner,
                ranges: Mutex::new(Vec::new()),
                pause_next: AtomicBool::new(false),
                paused: AtomicBool::new(false),
                resume: Notify::new(),
            })
        }

        pub(crate) fn n_ranges(&self) -> usize {
            self.ranges.lock().expect("ranges").len()
        }

        pub(crate) fn ranges_since(&self, from: usize) -> Vec<Range<u64>> {
            self.ranges.lock().expect("ranges")[from..].to_vec()
        }

        pub(crate) fn pause_next_read(&self) {
            self.pause_next.store(true, Ordering::SeqCst);
        }

        pub(crate) fn is_paused(&self) -> bool {
            self.paused.load(Ordering::SeqCst)
        }

        pub(crate) fn resume(&self) {
            self.resume.notify_one();
        }
    }

    #[async_trait]
    impl StorageProvider for RecordingStorage {
        async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
            self.inner.head(uri).await
        }
        async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
            self.inner.get(uri).await
        }
        async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
            self.ranges.lock().expect("ranges").push(range.clone());
            if self.pause_next.swap(false, Ordering::SeqCst) {
                self.paused.store(true, Ordering::SeqCst);
                self.resume.notified().await;
            }
            self.inner.get_range(uri, range).await
        }
        async fn put_atomic(
            &self,
            uri: &str,
            bytes: Bytes,
        ) -> Result<Option<String>, StorageError> {
            self.inner.put_atomic(uri, bytes).await
        }
        async fn put_if_match(
            &self,
            uri: &str,
            bytes: Bytes,
            expected_etag: Option<&str>,
        ) -> Result<Option<String>, StorageError> {
            self.inner.put_if_match(uri, bytes, expected_etag).await
        }
        async fn put_multipart(&self, uri: &str) -> Result<Box<dyn MultipartUpload>, StorageError> {
            self.inner.put_multipart(uri).await
        }
        async fn delete(&self, uri: &str) -> Result<(), StorageError> {
            self.inner.delete(uri).await
        }
    }

    /// A block source over empty storage, for tests that only need a [`Residency::Paged`] entry to
    /// exist. Does not own budget accounting, so dropping it never touches `current_bytes`.
    pub(crate) fn dummy_block_source(
        store: &Arc<DiskCacheStore>,
        uri: SuperfileUri,
    ) -> Arc<BlockCachedSource> {
        let inner: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(Bytes::new()));
        BlockCachedSource::new_pre_reserved(
            inner,
            Arc::downgrade(store),
            uri,
            store.blocks_path(&uri),
            None,
        )
    }

    pub(crate) fn seed_valid_block_file(
        store: &Arc<DiskCacheStore>,
        uri: &SuperfileUri,
        size: u64,
        blocks: &[u32],
    ) -> u64 {
        let f = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(store.blocks_path(uri))
            .expect("blocks");
        f.set_len(size).expect("set_len");
        let mut bitmap = RoaringBitmap::new();
        for &b in blocks {
            bitmap.insert(b);
        }
        fs::write(store.blocks_idx_path(uri), serialize_index(&bitmap)).expect("idx");
        blocks
            .iter()
            .map(|&b| {
                let start = u64::from(b) * CACHE_BLOCK_BYTES;
                (size - start).min(CACHE_BLOCK_BYTES)
            })
            .sum()
    }

    /// Local-filesystem background promotion should finish well within this.
    pub(crate) const PROMOTE_TIMEOUT: Duration = Duration::from_secs(10);
    /// Long enough to cover several background quiet-interval checks.
    pub(crate) const FOREGROUND_GUARD_HOLD: Duration = Duration::from_millis(50);
    /// Long enough for a promotion to be observed if one were coming, short
    /// enough to keep a "stays deferred" assertion quick.
    pub(crate) const DEFER_OBSERVATION: Duration = Duration::from_millis(500);

    /// Poll [`DiskCacheStore::is_mmap_promoted`] until it turns true or
    /// `within` elapses.
    ///
    /// Deliberately not [`DiskCacheStore::wait_until_mmap_promoted`]: that
    /// registers a promotion waiter, which is itself an override of the
    /// fill's deferral, so a test using it cannot tell whether promotion
    /// happened on its own.
    pub(crate) async fn poll_until_mmap_promoted(
        store: &Arc<DiskCacheStore>,
        uri: &SuperfileUri,
        within: Duration,
    ) -> bool {
        let deadline = Instant::now() + within;
        while Instant::now() < deadline {
            if store.is_mmap_promoted(uri) {
                return true;
            }
            sleep(POLL_INTERVAL).await;
        }
        store.is_mmap_promoted(uri)
    }

    /// Gap between `is_mmap_promoted` polls.
    const POLL_INTERVAL: Duration = Duration::from_millis(5);
    /// Large enough that one-byte sequential range reads cannot finish before
    /// the preemption test enters its foreground guard.
    pub(crate) const PREEMPT_TEST_BYTES: usize = 1 << 20;
    /// Tiny explicit budget used to prove reconciliation raises (or refuses to raise) it; smaller
    /// than any real superfile.
    pub(crate) const TEST_TINY_BUDGET_BYTES: u64 = 4;
    /// A comfortably large budget floor for the raise paths.
    pub(crate) const TEST_RAISED_FLOOR_BYTES: u64 = 1 << 20;
    /// Attempts per interleaving-sensitive test; enough to surface a torn drop-vs-fill without
    /// making the suite slow.
    pub(crate) const RACE_ITERATIONS: usize = 200;
    /// Cache files to seed a directory with when the point of the test is "more files than the read
    /// touches". Small: these tests assert on counters, not on wall time.
    pub(crate) const SEEDED_CACHE_FILES: usize = 8;

    /// Build the raw bytes of a minimal superfile (one scalar batch,
    /// no indexes).
    pub(crate) fn tiny_superfile_bytes() -> Bytes {
        let schema = Arc::new(Schema::new(vec![
            decimal128_id_field("doc_id"),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![]);
        let mut b = SuperfileBuilder::new(opts).expect("builder");
        let ids = decimal128_ids(vec![1u64]);
        let titles = LargeStringArray::from(vec!["alpha"]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)]).expect("batch");
        b.add_batch(&batch, &[]).expect("add_batch");
        Bytes::from(b.finish().expect("finish"))
    }

    /// Like [`tiny_superfile_bytes`] but with a 16-dim vector column, so the reader carries a real
    /// vector blob. Needed to drive the cold-fetch promote down its vector-hole path, where the
    /// blob stays on the block cache instead of the mmap.
    pub(crate) fn tiny_vector_superfile_bytes() -> Bytes {
        let schema = Arc::new(Schema::new(vec![decimal128_id_field("doc_id")]));
        let opts = BuilderOptions::new(
            schema.clone(),
            "doc_id",
            vec![],
            vec![default_vector_config("emb", 7)],
        );
        let mut b = SuperfileBuilder::new(opts).expect("builder");
        let batch = RecordBatch::try_new(schema, vec![Arc::new(decimal128_ids(vec![10u64, 11]))])
            .expect("batch");
        // Two rows on distinct axes so IVF clustering has something to place.
        let mut embeddings = vec![0.0f32; 2 * 16];
        embeddings[0] = 1.0;
        embeddings[16 + 1] = 1.0;
        b.add_batch(&batch, &[embeddings.as_slice()])
            .expect("add_batch");
        Bytes::from(b.finish().expect("finish"))
    }

    pub(crate) fn test_store() -> (TempDir, Arc<DiskCacheStore>) {
        test_store_with(|cfg| {
            cfg.mmap_cold_threshold_secs = 0;
        })
    }

    /// Build a store, applying `mutate` to the default config first.
    /// The storage root is the tempdir; cache files live under
    /// `<tempdir>/cache`. The sweep thread is left disabled by
    /// default (callers that want it enable it through `mutate`).
    pub(crate) fn test_store_with(
        mutate: impl FnOnce(&mut DiskCacheConfig),
    ) -> (TempDir, Arc<DiskCacheStore>) {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let mut cfg = DiskCacheConfig {
            cache_root: dir.path().join("cache"),
            mmap_cold_threshold_secs: 0,
            ..Default::default()
        };
        mutate(&mut cfg);
        let store = DiskCacheStore::new_unpinned(storage, cfg).expect("store");
        (dir, store)
    }

    /// Open a second store over `store`'s cache_root, the way a restart or a sibling process
    /// arrives at a directory it did not write. `mutate` adjusts the config first.
    pub(crate) fn reopen_store(
        store: &Arc<DiskCacheStore>,
        mutate: impl FnOnce(&mut DiskCacheConfig),
    ) -> Arc<DiskCacheStore> {
        let mut cfg = DiskCacheConfig {
            cache_root: store.config.cache_root.clone(),
            mmap_cold_threshold_secs: 0,
            ..Default::default()
        };
        mutate(&mut cfg);
        DiskCacheStore::new_unpinned(Arc::clone(&store.storage), cfg).expect("reopened store")
    }

    /// Put `bytes` at the storage location `store.reader(&uri)` will
    /// cold-fetch from, so the cold path has something to read.
    pub(crate) async fn put_superfile(
        store: &Arc<DiskCacheStore>,
        uri: &SuperfileUri,
        bytes: Bytes,
    ) {
        store
            .storage
            .put_atomic(&uri.storage_path(), bytes)
            .await
            .expect("put superfile");
    }

    // ----- construction / config -----

    /// Write `bytes` straight to the cache path for `uri`, the way a prior process (or a concurrent
    /// writer) leaves a finished file behind.
    pub(crate) fn seed_cache_file(
        store: &Arc<DiskCacheStore>,
        uri: &SuperfileUri,
        bytes: &Bytes,
    ) -> u64 {
        let path = store.cache_path(uri);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("cache root");
        }
        fs::write(&path, bytes.as_ref()).expect("seed cache file");
        bytes.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use std::{io::Error as IoError, sync::Arc};

    use crate::supertable::{
        manifest::SuperfileUri,
        reader_cache::disk::{test_support::*, *},
    };

    #[tokio::test]
    async fn new_creates_cache_root() {
        let (dir, store) = test_store();
        assert!(dir.path().join("cache").is_dir(), "cache_root created");
        // Debug impl exercises the custom formatter.
        let dbg = format!("{store:?}");
        assert!(dbg.contains("DiskCacheStore"));
        assert!(dbg.contains("n_cold_fetches"));
    }

    #[tokio::test]
    async fn new_with_sweep_thread_enabled_spawns_and_drops_cleanly() {
        // threshold > 0 takes the std::thread::spawn branch; interval
        // is clamped to >= 1. The Weak<Self> lets the thread exit when
        // we drop the last Arc.
        let (_dir, store) = test_store_with(|cfg| {
            cfg.mmap_cold_threshold_secs = 1;
            cfg.mmap_sweep_interval_secs = 0; // exercises `.max(1)` clamp
        });
        drop(store); // thread observes the failed Weak upgrade and exits
    }

    #[tokio::test]
    async fn new_unpinned_installs_empty_pinned_set() {
        let (_dir, store) = test_store();
        assert!(store.current_pinned_uris().is_empty());
    }

    // ----- stats / accessors -----

    #[tokio::test]
    async fn stats_reflect_config_and_counters() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.disk_budget_bytes = 12345;
        });
        let s = store.stats();
        assert_eq!(s.budget_bytes, 12345);
        assert_eq!(s.n_entries, 0);
        assert_eq!(s.current_bytes, 0);
        assert_eq!(s.n_cold_fetches, 0);
        assert_eq!(s.n_evictions, 0);
        assert_eq!(s.n_madvise_calls, 0);
        // CacheStats is Clone + Debug + Default.
        let _ = format!("{:?}", s.clone());
        assert_eq!(CacheStats::default().n_entries, 0);
    }

    #[tokio::test]
    async fn set_and_read_pinned_fn() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        store.set_pinned_fn(Arc::new(move || {
            let mut s = HashSet::new();
            s.insert(uri);
            s
        }));
        let pinned = store.current_pinned_uris();
        assert!(pinned.contains(&uri));
        assert_eq!(pinned.len(), 1);
    }

    #[tokio::test]
    async fn disk_cache_error_displays_all_variants() {
        let variants = [
            DiskCacheError::SuperfileOpen("x".into()),
            DiskCacheError::BudgetExceeded,
            DiskCacheError::Io(IoError::other("boom")),
        ];
        for v in variants {
            assert!(!format!("{v}").is_empty());
            assert!(!format!("{v:?}").is_empty());
        }
    }
}
