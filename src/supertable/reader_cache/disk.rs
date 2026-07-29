// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`DiskCacheStore`] — Tier 1 cache wrapping a
//! [`StorageProvider`] with parallel cold-fetch + LRU
//! eviction.

use std::{
    collections::HashSet,
    fmt, fs, io,
    io::SeekFrom,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{
        Arc, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use bytes::Bytes;
use dashmap::{DashMap, mapref::entry::Entry};
use futures::{
    future::try_join_all,
    stream::{FuturesUnordered, StreamExt},
};
use memmap2::{Mmap, UncheckedAdvice};
use thiserror::Error;
use tokio::{
    io::{AsyncSeekExt, AsyncWriteExt},
    sync::{Notify, OnceCell, Semaphore, oneshot},
    task::{JoinHandle, spawn_blocking},
};

use super::{
    block_source::BlockCachedSource,
    config::{ColdFetchMode, DiskCacheConfig, EvictionCandidate},
};
use crate::{
    config::global as global_config,
    runtime_metrics::io::scope_background,
    storage::{StorageError, StorageProvider},
    superfile::{
        BytesLazyByteSource, LazyByteSource, LazyByteSourceError, PrefetchedSource,
        format::{footer, kv},
        reader::{OpenOptions, SuperfileReader},
    },
    supertable::{
        StorageRangeSource,
        manifest::{SubsectionOffsets, SuperfileUri},
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

/// Pause this URI's background full-object fill while a caller besides the
/// cache entry holds its lazy reader (`strong_count > 1`). Unrelated URIs
/// are unaffected — that is the per-URI quiescence contract.
fn reader_blocks_background_fill(reader: &Weak<SuperfileReader>) -> bool {
    reader.strong_count() > 1
}

/// Errors surfaced by [`DiskCacheStore::reader`].
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
    /// Eviction couldn't free enough space because every
    /// cached entry was pinned (or there were no cached
    /// entries and the incoming superfile alone exceeds the
    /// disk budget). The query layer can fall back to a
    /// `RangeOnly` path on this error; the cache itself just
    /// surfaces it as a typed error.
    #[error("disk cache budget exceeded with no eligible victims")]
    BudgetExceeded,
    /// An invalid or conflicting configuration was supplied.
    #[error("config: {0}")]
    Config(String),
}

/// Live cache entry. Holds the cached `Arc<SuperfileReader>`
/// (constructed once on cache fill); the `Bytes` inside the
/// reader is mmap-backed via `Bytes::from_owner(ArcMmapOwner)`,
/// so dropping the last `Arc<SuperfileReader>` (cache evict +
/// no in-flight queries) drops the mmap and unmaps the file.
///
/// In-flight queries pin the reader independently — the
/// cache can evict the entry and unlink the on-disk file
/// while a query still holds an `Arc<SuperfileReader>` over
/// the now-unlinked-but-mmap'd bytes. POSIX semantics
/// (mac/linux): the mmap stays valid until the last
/// reference drops.
///
/// `mmap` is `None` for in-memory-bytes-backed entries
/// produced by the hybrid cold-fetch path (transient, before
/// `finalize_to_mmap` runs); `Some` once the entry is
/// mmap-backed. The idle-threshold sweep thread iterates
/// entries with `Some(mmap)` and calls
/// `madvise(MADV_DONTNEED)` on those that haven't been
/// accessed in `mmap_cold_threshold_secs`.
struct CachedEntry {
    reader: Arc<SuperfileReader>,
    /// Separate handle on the mmap for `MADV_DONTNEED`. Same
    /// `Arc<Mmap>` instance that backs the reader's `Bytes`
    /// — both share the underlying OS mapping, so `madvise`
    /// on either path affects the cached entry's resident
    /// pages.
    mmap: Option<Arc<Mmap>>,
    /// Accounted bytes for this entry. For eager entries this is fixed at
    /// insertion; for block-backed lazy entries this points at the block
    /// source's live filled-bytes counter.
    size_bytes: Arc<AtomicU64>,
    /// Who owns accounting release for this entry.
    accounting: EntryAccounting,
    /// Identity of the sparse source currently allowed to grow this lazy
    /// entry. `None` for eager and fully mmap-backed entries.
    block_token: Option<Arc<()>>,
    /// Live block-cache source for lazy (and hybrid mmap+hole) entries.
    /// Retained across vector-excluding background fill so touched vector
    /// ranges stay local after parquet/FTS promote to mmap.
    block_source: Option<Arc<BlockCachedSource>>,
    /// Whether a background fill task has been spawned for this URI.
    /// Vector opens leave this false (block-cache only); an later FTS/SQL
    /// open may flip it and start fill.
    fill_spawned: AtomicBool,
    last_access_us: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryAccounting {
    /// Store-reserved entry; removal releases `size_bytes`.
    Eager,
    /// Block-source-reserved entry; source drop releases bytes.
    #[cfg(test)]
    SourceOwned,
}

/// Coalescing cell — concurrent cold readers on the same URI
/// share one `OnceCell` and observe the same fetch result.
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
    pub n_evictions: u64,
    /// Cumulative count of entries `madvise(MADV_DONTNEED)`'d
    /// by the idle-threshold sweep thread. Includes individual
    /// `sweep_once()` invocations.
    pub n_madvise_calls: u64,
}

/// Pulls superfile bytes through a [`StorageProvider`] and
/// caches them locally as mmap-backed `SuperfileReader`s.
///
/// Construction is sync; `reader()` is async (cold fetches
/// go through the storage provider's async interface).
pub struct DiskCacheStore {
    storage: Arc<dyn StorageProvider>,
    config: DiskCacheConfig,
    started_at: Instant,
    cached: DashMap<SuperfileUri, Arc<CachedEntry>>,
    /// Per-URI cold-fetch coalescing. Inserted by the first
    /// caller to touch a cold URI; subsequent callers find
    /// the same `OnceCell` and `await` it via
    /// `get_or_try_init`.
    coordinators: DashMap<SuperfileUri, Coordinator>,
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
    n_evictions: AtomicU64,
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
            current_bytes: AtomicU64::new(0),
            budget_bytes: AtomicU64::new(configured_budget),
            budget_auto_sized: AtomicBool::new(false),
            budget_warned: AtomicBool::new(false),
            n_cold_fetches: AtomicU64::new(0),
            n_evictions: AtomicU64::new(0),
            n_madvise_calls: AtomicU64::new(0),
            n_promotion_waiters: AtomicU64::new(0),
            pinned_fn: std::sync::Mutex::new(pinned_fn),
            prefetch_semaphore,
        });

        // Reuse any cache files a prior run (or another handle) left on disk:
        // rebuild the in-memory index so reads hit the NVMe bytes instead of
        // cold-fetching them back from object storage.
        store.restore_from_cache_root();

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

    /// Run one pass of the `MADV_DONTNEED` sweep against
    /// currently-cached entries. Each entry with
    /// `now - last_access_us > mmap_cold_threshold_secs * 1e6`
    /// gets `madvise(MADV_DONTNEED)` on its mmap; pages
    /// re-fault on next read (cheap on SSD-backed page cache).
    ///
    /// Exposed for explicit invocation from tests so they
    /// don't have to sleep for the sweep cadence. The
    /// background thread calls this on each tick.
    ///
    /// Iteration safety: snapshots `(uri, mmap_arc,
    /// last_access)` tuples into a Vec, drops the DashMap
    /// iterator (releasing shard guards), then `madvise`s.
    /// Holding shard guards through `madvise` would block
    /// eviction during the sweep — `madvise` on a multi-GB
    /// mmap can take milliseconds.
    pub fn sweep_once(&self) -> u64 {
        let threshold_us = self
            .config
            .mmap_cold_threshold_secs
            .saturating_mul(1_000_000);
        let now_us = self.now_us();
        // Snapshot: clone the Arc<Mmap> + last-access into an
        // owned Vec, then drop the iterator.
        let snapshot: Vec<(SuperfileUri, Arc<Mmap>, u64)> = self
            .cached
            .iter()
            .filter_map(|e| {
                let mmap = e.value().mmap.clone()?;
                let last = e.value().last_access_us.load(Ordering::Acquire);
                Some((*e.key(), mmap, last))
            })
            .collect();
        let mut n_advised = 0u64;
        for (_uri, mmap, last_access) in snapshot {
            let idle = now_us.saturating_sub(last_access);
            if idle >= threshold_us {
                // `MADV_DONTNEED` lives on `UncheckedAdvice` in
                // memmap2 because it's unsafe for *writable*
                // mappings (pages truly freed → re-reads see
                // zero-filled). For our **read-only** mappings
                // it's safe: dropped pages re-fault from the
                // backing file on next access. The cache files
                // are immutable once written + we never write
                // to the mmap, so the read-back is bit-identical.
                //
                // Errors are non-fatal — typically platform
                // limitations on macOS/BSD; we just skip.
                //
                // SAFETY: the mmap is read-only and the backing
                // file is immutable for the lifetime of this
                // mapping; pages dropped by `MADV_DONTNEED`
                // re-fault from disk on next read.
                let _ = unsafe { mmap.unchecked_advise(UncheckedAdvice::DontNeed) };
                n_advised += 1;
            }
        }
        if n_advised > 0 {
            self.n_madvise_calls.fetch_add(n_advised, Ordering::AcqRel);
        }
        n_advised
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
    fn resolve_storage(
        &self,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Arc<dyn StorageProvider> {
        storage
            .map(Arc::clone)
            .unwrap_or_else(|| Arc::clone(&self.storage))
    }

    /// Hot path. Cached → cloned `Arc<SuperfileReader>`; cold
    /// → coalesced cold-fetch coordinator. Dispatches by
    /// `config.cold_fetch_mode`:
    ///
    /// - [`ColdFetchMode::LazyForegroundWithBackgroundFill`] (default):
    ///   foreground returns a lazy reader over a `StorageRangeSource`
    ///   that pays only the per-query range budget; a background task
    ///   downloads the full superfile to NVMe and swaps in the mmap'd
    ///   entry, so subsequent (warm) queries are resident. Minimizes
    ///   cold-query p50 on object-storage-native deployments.
    /// - [`ColdFetchMode::HybridWithPrefetch`]:
    ///   parallel range-GETs feed the foreground reader (built
    ///   from in-memory bytes) and a fire-and-forget cache fill
    ///   (mmap'd, registered on completion). Foreground returns
    ///   when range-fetches finish; pwrites + mmap + cache
    ///   registration finalize in the background.
    /// - [`ColdFetchMode::RangeOnly`]: callers should construct
    ///   a `StorageRangeSource` + `SuperfileReader::open_lazy`
    ///   directly — `DiskCacheStore::reader` rejects this mode
    ///   because the disk-cache layer isn't the right entry
    ///   point — `RangeOnly` bypasses the cache by design.
    pub async fn reader(
        self: &Arc<Self>,
        uri: &SuperfileUri,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        // Default allows fill — same as FTS/SQL. Vector search must call
        // [`Self::reader_with_hints`] with `allow_background_fill = false`.
        self.reader_with_hints(uri, None, None, true).await
    }

    /// like [`Self::reader`] but takes a precomputed
    /// [`SubsectionOffsets`] hint (sourced from the manifest's
    /// [`crate::supertable::manifest::SuperfileEntry::subsection_offsets`]).
    /// On a cold miss in the
    /// `LazyForegroundWithBackgroundFill` mode the hint lets the
    /// cold-fetch path fire the parquet-footer, vector subsection,
    /// and FTS subsection GETs **in parallel** (1 RTT cold open)
    /// instead of doing the parquet footer first and the
    /// subsection fetches second (2 RTTs).
    ///
    /// `allow_background_fill` is the modality gate: FTS/SQL pass `true`
    /// so parquet/FTS bytes can promote to mmap (vector blob skipped);
    /// vector search passes `false` and retains only the block cache.
    ///
    /// `None` falls back to the 2-RTT shape — same shape,
    /// slower. The other cold-fetch modes (`HybridWithPrefetch`,
    /// `RangeOnly`) ignore the hint today.
    pub async fn reader_with_hints(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
        allow_background_fill: bool,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        match self.config.cold_fetch_mode {
            ColdFetchMode::HybridWithPrefetch => self.reader_hybrid(uri, storage).await,
            ColdFetchMode::RangeOnly => Err(DiskCacheError::SuperfileOpen(
                "ColdFetchMode::RangeOnly bypasses the disk cache; \
                 construct StorageRangeSource + open_lazy directly"
                    .into(),
            )),
            ColdFetchMode::LazyForegroundWithBackgroundFill => {
                self.reader_lazy_with_bg_fill_hinted(uri, offsets, storage, allow_background_fill)
                    .await
            }
        }
    }

    /// Open a streaming, RangeOnly reader directly against object
    /// storage, bypassing the disk cache entirely: no budget
    /// reservation, no background fill, no entry inserted into
    /// `cached`.
    ///
    /// Used as the [`DiskCacheError::BudgetExceeded`] fallback —
    /// e.g. a single superfile larger than the whole cache budget.
    /// The query still succeeds by issuing range GETs for only the
    /// bytes the reader touches; nothing is admitted, so there's
    /// nothing to evict.
    pub async fn open_range_only(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        let fetch_storage = self.resolve_storage(storage);
        let storage_uri = Self::storage_path(uri);
        let range_src: Arc<dyn LazyByteSource> = match offsets {
            Some(o) if o.total_size > 0 => Arc::new(StorageRangeSource::with_known_size(
                fetch_storage,
                storage_uri,
                o.total_size,
            )),
            _ => Arc::new(StorageRangeSource::with_unknown_size(
                fetch_storage,
                storage_uri,
            )),
        };
        // Range-only is also a lazy reader over object storage. A full CRC
        // scan here would turn a fallback path meant to issue targeted
        // ranges into a whole-superfile read.
        let reader =
            SuperfileReader::open_lazy_with(range_src, OpenOptions { verify_crc: false }).await?;
        Ok(Arc::new(reader))
    }

    /// Strictly-cached cold-fetch path — waits for all pwrites
    /// + fsync + mmap before returning. Public for integration
    /// tests that want this deterministic behavior; the
    /// production reader path uses `reader_hybrid`.
    pub async fn reader_synchronous(
        self: &Arc<Self>,
        uri: &SuperfileUri,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        let storage = Arc::clone(&self.storage);
        self.reader_synchronous_with_storage(uri, storage).await
    }

    /// Like [`Self::reader_synchronous`], but fetches a cache miss through
    /// `fetch_storage` instead of the cache's own `self.storage`. Needed for
    /// the hidden vector-index, whose superfiles live behind a prefixed storage
    /// provider that the shared (user-keyed) cache's `self.storage` can't
    /// resolve — without this the cold-fetch reads the wrong path. On a cache
    /// hit it returns the resident mmap-backed reader regardless of storage.
    pub async fn reader_synchronous_with_storage(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        fetch_storage: Arc<dyn StorageProvider>,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        if let Some(entry) = self.cached.get(uri) {
            if entry.mmap.is_some() {
                entry.last_access_us.store(self.now_us(), Ordering::Release);
                return Ok(Arc::clone(&entry.reader));
            }
            drop(entry);
            if let Some((_, removed)) = self.cached.remove(uri) {
                self.release_entry_accounting(&removed);
            }
            self.coordinators.remove(uri);
            let replacement = self.cold_fetch(uri, Arc::clone(&fetch_storage)).await?;
            return Ok(Arc::clone(&replacement.reader));
        }
        let cell = self
            .coordinators
            .entry(*uri)
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();
        let result = cell
            .get_or_init(|| async { self.cold_fetch(uri, Arc::clone(&fetch_storage)).await })
            .await;
        match result {
            Ok(entry) => {
                self.coordinators.remove(uri);
                Ok(Arc::clone(&entry.reader))
            }
            Err(_e) => {
                self.coordinators.remove(uri);
                Err(self
                    .cold_fetch(uri, Arc::clone(&fetch_storage))
                    .await
                    .err()
                    .unwrap_or(DiskCacheError::SuperfileOpen("cold fetch error".into())))
            }
        }
    }

    /// Hybrid cold-fetch. Range-fetches feed the foreground
    /// reader from in-memory bytes; pwrites + mmap + cache
    /// registration run as a background task that outlives
    /// this method's return.
    async fn reader_hybrid(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        if let Some(entry) = self.cached.get(uri) {
            entry.last_access_us.store(self.now_us(), Ordering::Release);
            return Ok(Arc::clone(&entry.reader));
        }
        let cell = self
            .coordinators
            .entry(*uri)
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();
        // OnceCell value: `Result<Arc<CachedEntry>, ...>` but we
        // only need the reader part for the foreground response.
        // The coordinator builds a CachedEntry whose `reader` is
        // the in-memory-backed `Arc<SuperfileReader>`; the
        // background task replaces the entry in `cached` with a
        // mmap-backed reader once the disk file is finalized.
        let result = cell
            .get_or_init(|| async {
                let fetch_storage = self.resolve_storage(storage);
                self.cold_fetch_hybrid(uri, fetch_storage).await
            })
            .await;
        match result {
            Ok(entry) => Ok(Arc::clone(&entry.reader)),
            Err(DiskCacheError::BudgetExceeded) => {
                self.coordinators.remove(uri);
                Err(DiskCacheError::BudgetExceeded)
            }
            Err(_) => {
                // Only the retry path needs the resolved storage handle; the Ok
                // and BudgetExceeded arms skip the clone.
                self.coordinators.remove(uri);
                let fetch_storage = self.resolve_storage(storage);
                self.cold_fetch_hybrid(uri, fetch_storage)
                    .await
                    .map(|entry| Arc::clone(&entry.reader))
            }
        }
    }

    /// Whether `uri` has any cache entry — including a still-lazy
    /// `LazyForegroundWithBackgroundFill` reader whose `mmap` is `None`.
    /// Use [`Self::is_mmap_promoted`] to test for residency.
    pub fn is_cached(&self, uri: &SuperfileUri) -> bool {
        self.cached.contains_key(uri)
    }

    /// Whether `uri` is cached with a finished mmap promotion
    /// (`CachedEntry::mmap == Some`). False while
    /// `LazyForegroundWithBackgroundFill` still holds the lazy
    /// in-memory reader or the background download is in flight.
    pub fn is_mmap_promoted(&self, uri: &SuperfileUri) -> bool {
        self.cached
            .get(uri)
            .map(|e| e.mmap.is_some())
            .unwrap_or(false)
    }

    /// Block until the background fill has swapped in the
    /// mmap-backed reader, or fail after `timeout`.
    pub async fn wait_until_mmap_promoted(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        timeout: Duration,
    ) -> Result<(), DiskCacheError> {
        let _guard = PromotionWaitGuard::new(&self.n_promotion_waiters);
        let start = Instant::now();
        while start.elapsed() < timeout {
            if self.is_mmap_promoted(uri) {
                return Ok(());
            }
            tokio::time::sleep(MMAP_PROMOTION_POLL_INTERVAL).await;
        }
        Err(DiskCacheError::SuperfileOpen(format!(
            "superfile {uri:?} not mmap-promoted within {timeout:?}"
        )))
    }

    /// Block until no cache entry has a background fill still in flight
    /// (fill spawned, not yet mmap-promoted), or fail after `timeout`.
    ///
    /// Scoped to work the caller's own opens actually caused: entries that
    /// never spawned a fill (vector opens) and superfiles never opened at
    /// all are not waited on. Registering as a promotion waiter releases
    /// fills that are politely waiting on a held foreground reader.
    pub async fn wait_until_fills_settled(
        self: &Arc<Self>,
        timeout: Duration,
    ) -> Result<(), DiskCacheError> {
        let _guard = PromotionWaitGuard::new(&self.n_promotion_waiters);
        let start = Instant::now();
        loop {
            let pending = self.cached.iter().any(|entry| {
                entry.value().fill_spawned.load(Ordering::Acquire) && entry.value().mmap.is_none()
            });
            if !pending {
                return Ok(());
            }
            if start.elapsed() >= timeout {
                return Err(DiskCacheError::SuperfileOpen(format!(
                    "background fills not settled within {timeout:?}"
                )));
            }
            tokio::time::sleep(MMAP_PROMOTION_POLL_INTERVAL).await;
        }
    }

    /// Snapshot of the cache's load. Cheap; reads atomics +
    /// a `DashMap::len` (which itself is `O(shards)`).
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            n_entries: self.cached.len() as u64,
            current_bytes: self.current_bytes.load(Ordering::Acquire),
            budget_bytes: self.disk_budget_bytes(),
            n_cold_fetches: self.n_cold_fetches.load(Ordering::Acquire),
            n_evictions: self.n_evictions.load(Ordering::Acquire),
            n_madvise_calls: self.n_madvise_calls.load(Ordering::Acquire),
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

    /// Sum of mmap virtual sizes across all cached entries
    /// with an active mapping. This is the **upper bound**
    /// on the cache's resident memory — actual RSS is some
    /// subset (only pages that have been faulted in and not
    /// yet `madvise(MADV_DONTNEED)`'d by a sweep). Used by
    /// [`crate::supertable::Supertable::stats`] to
    /// report `mmap_resident_bytes` and to drive the
    /// budget-aware sweep in [`Self::sweep_for_budget`].
    pub fn current_mmap_size_bytes(&self) -> u64 {
        self.cached
            .iter()
            .filter_map(|e| e.value().mmap.as_ref().map(|m| m.len() as u64))
            .sum()
    }

    /// drop mmap pages until the cache's working set
    /// is back under `budget_bytes`. No-op if already under
    /// budget. Returns the number of entries that received
    /// `madvise(MADV_DONTNEED)`.
    ///
    /// Policy: iterate entries by ascending `last_access_us`
    /// (oldest first); `madvise` each one until the
    /// projected residency drops below the budget. Entries
    /// stay in the cache map — pages re-fault from the
    /// backing file on next access. The on-disk cache and
    /// `disk_budget_bytes` are unchanged; only the RSS
    /// footprint is affected.
    ///
    /// Pinned URIs are NOT skipped here: pinning protects
    /// against EVICTION (entry removal + file unlink), not
    /// against page reclaim. A pinned entry whose pages
    /// have been madvise'd re-faults on next access and
    /// behaves correctly; the cost is one re-fault per
    /// re-touched page.
    pub fn sweep_for_budget(&self, budget_bytes: u64) -> u64 {
        let mut total = self.current_mmap_size_bytes();
        if total <= budget_bytes {
            return 0;
        }
        // Snapshot candidates: (uri, mmap_arc, last_access,
        // size). Drop the iterator before madvise calls so
        // we don't hold shard guards across the syscall.
        let mut candidates: Vec<(SuperfileUri, Arc<Mmap>, u64, u64)> = self
            .cached
            .iter()
            .filter_map(|e| {
                let mmap = e.value().mmap.clone()?;
                Some((
                    *e.key(),
                    mmap,
                    e.value().last_access_us.load(Ordering::Acquire),
                    e.value().size_bytes.load(Ordering::Acquire),
                ))
            })
            .collect();
        // Oldest-first.
        candidates.sort_by_key(|(_, _, last, _)| *last);

        let mut n_advised = 0u64;
        for (_uri, mmap, _last, size) in candidates {
            if total <= budget_bytes {
                break;
            }
            // SAFETY: the mmap is read-only and the backing
            // file is immutable for the mapping's lifetime;
            // pages dropped by MADV_DONTNEED re-fault from
            // disk on next read. Identical safety argument
            // to the `sweep_once` path; see that fn for the
            // full discussion.
            let _ = unsafe { mmap.unchecked_advise(UncheckedAdvice::DontNeed) };
            self.n_madvise_calls.fetch_add(1, Ordering::AcqRel);
            total = total.saturating_sub(size);
            n_advised += 1;
        }
        n_advised
    }

    /// Observability accessor: invoke the currently-installed
    /// `pinned_fn` and return its result. Useful for tests
    /// that want to assert which URIs are protected from
    /// eviction at the moment of the call; also for
    /// debug-time inspection of long-running caches.
    ///
    /// Cheap: clones the `Arc<dyn Fn>` out of the mutex,
    /// drops the lock, then invokes the closure. The closure
    /// itself is whatever the caller installed — most
    /// commonly the `Weak<SupertableInner>`-based snapshot
    /// installed by [`crate::supertable::Supertable::create`]
    /// / [`crate::supertable::Supertable::open`].
    pub fn current_pinned_uris(&self) -> HashSet<SuperfileUri> {
        let f = {
            let g = self.pinned_fn.lock().expect("pinned_fn mutex poisoned");
            Arc::clone(&g)
        };
        f()
    }

    /// Insert already-in-hand bytes into the cache without
    /// round-tripping through storage. Used by the writer to
    /// pre-populate the cache with the superfiles it just
    /// published, so the producer's next query on its own
    /// superfiles skips the cold-fetch wall-time hit (parallel
    /// range-fetch + pwrite + mmap, ~50-150 ms per superfile on
    /// the laptop bench).
    ///
    /// Idempotent: if `uri` is already in the cache,
    /// returns `Ok(())` without re-writing. Failure modes:
    /// - [`DiskCacheError::BudgetExceeded`] if the byte
    ///   count won't fit even after eviction.
    /// - [`DiskCacheError::Io`] for filesystem failures
    ///   (cache dir not writable, disk full, etc.).
    /// - [`DiskCacheError::SuperfileOpen`] if the bytes
    ///   don't parse as a valid superfile (programmer error
    ///   — the writer must hand over the same bytes it
    ///   wrote to storage).
    ///
    /// Cold-fetch semantics: does **not** increment
    /// `n_cold_fetches` (this is a warm insert, not a
    /// storage round-trip). Increments `n_entries` and
    /// `current_bytes` exactly as the cold-fetch path does.
    pub async fn insert_warm(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        bytes: Bytes,
    ) -> Result<(), DiskCacheError> {
        // Idempotent: already-cached URIs are a no-op. The
        // writer may call this for superfiles a prior commit
        // already published (e.g., an OCC retry where the
        // same UUID superfile got re-inserted into the cache).
        if self.cached.contains_key(uri) {
            return Ok(());
        }

        let size = bytes.len() as u64;

        // Reserve budget (CAS-loop with eviction on miss).
        // Use `reserve_manual` so a panic between this and
        // the DashMap insert doesn't double-decrement on
        // unwind — `reserve_manual` keeps the bytes
        // reserved; we manually roll back on the rare error
        // path below.
        self.reserve_manual(size).await?;

        // Roll back the reservation on any error past this
        // point. Wrap the rest in a closure-shape so `?`
        // works while we still get to undo current_bytes
        // on failure.
        let result: Result<Arc<CachedEntry>, DiskCacheError> = async {
            let tmp = self.tmp_path(uri);
            let final_path = self.cache_path(uri);

            // Write the bytes to a tmp file, then atomically rename into place.
            // No fsync: the disk cache is a reconstructible mirror of bytes that
            // are already durable in object storage, so a crash losing an
            // unflushed cache file just cold-fetches on the next open — and
            // `restore_from_cache_root` CRC-verifies on-disk files at open,
            // dropping any torn one. Skipping the fsync keeps the committer's
            // warm-fill off the synchronous disk-flush path.
            {
                let mut file = tokio::fs::File::create(&tmp).await?;
                file.write_all(&bytes).await?;
                file.flush().await?;
            }
            tokio::fs::rename(&tmp, &final_path).await?;

            // mmap the freshly-written file + open it as a superfile reader.
            // Skip CRC: the committer just built these bytes in memory and they
            // are known-valid (CRC'd at build, already opened as a reader for
            // summary extraction) — re-scanning here is redundant. Files read
            // back from a PRIOR run take the verifying path via
            // `restore_from_cache_root`.
            self.open_cached_entry(&final_path, size, false)
        }
        .await;

        let entry = match result {
            Ok(e) => e,
            Err(e) => {
                // Roll back the reservation; leave any tmp
                // file behind for next-run cleanup (the
                // write may have partially succeeded).
                self.current_bytes.fetch_sub(size, Ordering::Release);
                return Err(e);
            }
        };

        // Final commit: install into the cache map. If a
        // concurrent caller raced us to the same URI (e.g.,
        // a cold-fetch landed first), prefer the
        // already-present entry — release our reservation
        // for the duplicate bytes.
        match self.cached.entry(*uri) {
            Entry::Vacant(v) => {
                v.insert(entry);
            }
            Entry::Occupied(_) => {
                // Lost the race; release our reservation +
                // unlink the just-written file (or leave it
                // — the existing entry mmaps a different
                // file on disk).
                self.current_bytes.fetch_sub(size, Ordering::Release);
                let _ = fs::remove_file(self.cache_path(uri));
            }
        }
        Ok(())
    }

    // ----- internals -----

    fn now_us(&self) -> u64 {
        self.started_at.elapsed().as_micros() as u64
    }

    /// mmap a cache file and open it as a [`SuperfileReader`], building the
    /// `CachedEntry`. Shared by the warm-insert path and the open-time index
    /// rebuild ([`Self::restore_from_cache_root`]); the caller owns budget
    /// accounting and the `cached`-map insert. The reader's bytes and
    /// `CachedEntry.mmap` share one `Arc<Mmap>` so a later `MADV_DONTNEED`
    /// sweep touches the same mapping.
    fn open_cached_entry(
        &self,
        path: &Path,
        size: u64,
        verify_crc: bool,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let mmap = open_readonly_mmap(path).map_err(DiskCacheError::Io)?;
        let mmap_arc = Arc::new(mmap);
        let reader_bytes = Bytes::from_owner(ArcMmapOwner(Arc::clone(&mmap_arc)));
        let reader = SuperfileReader::open_with(reader_bytes, OpenOptions { verify_crc })?;
        Ok(Arc::new(CachedEntry {
            reader: Arc::new(reader),
            mmap: Some(mmap_arc),
            size_bytes: Arc::new(AtomicU64::new(size)),
            accounting: EntryAccounting::Eager,
            block_token: None,
            block_source: None,
            fill_spawned: AtomicBool::new(false),
            last_access_us: AtomicU64::new(self.now_us()),
        }))
    }

    /// Rebuild the in-memory index from cache files a prior run (or another
    /// handle) left under `cache_root`, so a fresh `DiskCacheStore` reuses the
    /// NVMe bytes instead of cold-fetching them back from object storage. Each
    /// complete `seg-<uuid>.sf.parquet` is mmap'd, opened (CRC-verified per
    /// config), and inserted; `.tmp` in-flight files and anything that fails to
    /// open (truncated / incompatible) are skipped and unlinked. Best-effort:
    /// a scan error leaves the index empty (every read just cold-fetches, as
    /// before). The budget is enforced lazily — entries are mmap-lazy (no RSS
    /// until touched) and the first `sweep_for_budget` trims any excess.
    fn restore_from_cache_root(self: &Arc<Self>) {
        let dir = match fs::read_dir(&self.config.cache_root) {
            Ok(d) => d,
            Err(_) => return,
        };
        for entry in dir.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name.ends_with(BLOCKS_FILE_SUFFIX) {
                let _ = fs::remove_file(&path);
                continue;
            }
            let Some(uri) = SuperfileUri::from_cache_filename(name) else {
                continue; // `.tmp` in-flight or foreign file — skip.
            };
            let size = match entry.metadata() {
                Ok(m) if m.len() > 0 => m.len(),
                _ => continue,
            };
            match self.open_cached_entry(&path, size, self.config.verify_crc_on_open) {
                Ok(cached_entry) => {
                    if self.cached.insert(uri, cached_entry).is_none() {
                        self.current_bytes.fetch_add(size, Ordering::Release);
                    }
                }
                Err(_) => {
                    // Truncated / corrupt / incompatible: drop it so the next
                    // read cold-fetches a clean copy.
                    let _ = fs::remove_file(&path);
                }
            }
        }
    }

    /// Build a per-URI cache file path under `cache_root`.
    fn cache_path(&self, uri: &SuperfileUri) -> PathBuf {
        self.config.cache_root.join(uri.cache_filename())
    }

    /// Build a per-URI sparse block-cache path under `cache_root`.
    fn blocks_path(&self, uri: &SuperfileUri) -> PathBuf {
        self.config
            .cache_root
            .join(format!("{}{BLOCKS_FILE_SUFFIX}", uri.cache_filename()))
    }

    /// Build a per-URI tempfile path (sparse destination
    /// during cold fetch; renamed to `cache_path` on success).
    fn tmp_path(&self, uri: &SuperfileUri) -> PathBuf {
        self.config.cache_root.join(uri.cache_tmp_filename())
    }

    /// The storage-side URI for a superfile, mirroring the
    /// writer's persist layout.
    fn storage_path(uri: &SuperfileUri) -> String {
        uri.storage_path()
    }

    /// Hybrid cold-fetch. Returns the foreground reader
    /// (in-memory-bytes-backed) as soon as range-fetches
    /// complete; spawns a background task to fsync + rename +
    /// mmap + register the cache entry. Subsequent callers on
    /// the same URI either see the in-flight OnceCell (same
    /// foreground reader) or, once finalize completes, hit
    /// the mmap-backed cache entry.
    async fn cold_fetch_hybrid(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        fetch_storage: Arc<dyn StorageProvider>,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let storage_uri = Self::storage_path(uri);
        let head = fetch_storage.head(&storage_uri).await?;
        let size = head.size;
        // Don't use the borrow-lifetimed Reservation guard
        // because it would tie the future to `&self` and block
        // the `tokio::spawn` of the background finalizer. We
        // reserve manually here; the background task either
        // commits (cache filled) or rolls back via fetch_sub.
        self.reserve_manual(size).await?;
        let reserved_bytes = size;
        let tmp = self.tmp_path(uri);
        let final_path = self.cache_path(uri);

        // 1. Parallel range-GETs. Each task: get_range →
        //    save Bytes for foreground assembly + spawn a
        //    fire-and-forget pwrite.
        let n_streams = self.config.cold_fetch_streams.max(1) as u64;
        let chunk_size = self
            .config
            .cold_fetch_chunk_bytes
            .max(size.div_ceil(n_streams));
        let n_chunks = if size == 0 {
            0
        } else {
            size.div_ceil(chunk_size)
        };

        let file = tokio::fs::File::create(&tmp).await?;
        file.set_len(size).await?;
        let file = Arc::new(tokio::sync::Mutex::new(file));

        // Per-chunk slot for the foreground buffer assembly.
        let chunks: Arc<tokio::sync::Mutex<Vec<Option<(u64, Bytes)>>>> =
            Arc::new(tokio::sync::Mutex::new(vec![None; n_chunks as usize]));

        let mut fetch_handles = Vec::with_capacity(n_chunks as usize);
        let mut write_handles = Vec::with_capacity(n_chunks as usize);

        for i in 0..n_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(size);
            let storage = Arc::clone(&fetch_storage);
            let file = Arc::clone(&file);
            let chunks = Arc::clone(&chunks);
            let uri_s = storage_uri.clone();

            // Spawn the fetch task. It captures a Sender for
            // its pwrite handle so the outer task can join
            // pwrites separately from fetches.
            let (write_tx, write_rx) = oneshot::channel::<JoinHandle<Result<(), DiskCacheError>>>();
            write_handles.push(write_rx);

            fetch_handles.push(tokio::spawn(async move {
                let bytes = storage.get_range(&uri_s, start..end).await?;
                // Save Bytes for the foreground.
                {
                    let mut guard = chunks.lock().await;
                    guard[i as usize] = Some((start, bytes.clone()));
                }
                // Spawn the pwrite as a fire-and-forget task.
                // Its JoinHandle goes to the background
                // finalizer (via oneshot) so the foreground
                // doesn't wait for it.
                let pwrite_handle = tokio::spawn(async move {
                    let mut guard = file.lock().await;
                    guard.seek(SeekFrom::Start(start)).await?;
                    guard.write_all(&bytes).await?;
                    Ok::<(), DiskCacheError>(())
                });
                let _ = write_tx.send(pwrite_handle);
                Ok::<(), DiskCacheError>(())
            }));
        }

        // 2. Await all fetches (NOT pwrites). Foreground bytes
        //    are now complete.
        for h in fetch_handles {
            h.await
                .map_err(|e| DiskCacheError::SuperfileOpen(format!("fetch join: {e}")))??;
        }

        // 3. Assemble the in-memory buffer for the foreground.
        let buffer = {
            let chunks_guard = chunks.lock().await;
            let mut buf = vec![0u8; size as usize];
            for (start, bytes) in chunks_guard.iter().flatten() {
                let s = *start as usize;
                let e = s + bytes.len();
                buf[s..e].copy_from_slice(bytes);
            }
            buf
        };
        let foreground_bytes = Bytes::from(buffer);
        let foreground_reader = SuperfileReader::open_with(
            foreground_bytes,
            OpenOptions {
                verify_crc: self.config.verify_crc_on_open,
            },
        )?;
        let foreground_reader = Arc::new(foreground_reader);

        // 4. Construct a CachedEntry with the foreground
        //    reader. Multiple foreground callers waiting on
        //    the coordinator's OnceCell each get an Arc clone
        //    of this reader. Once the background finalizer
        //    completes, the same `cached` slot gets replaced
        //    by a mmap-backed reader; from that point on,
        //    cache hits serve the mmap reader instead.
        let entry = Arc::new(CachedEntry {
            reader: Arc::clone(&foreground_reader),
            mmap: None, // hybrid foreground entry is in-memory; finalizer mmaps later
            size_bytes: Arc::new(AtomicU64::new(size)),
            accounting: EntryAccounting::Eager,
            block_token: None,
            block_source: None,
            fill_spawned: AtomicBool::new(false),
            last_access_us: AtomicU64::new(self.now_us()),
        });
        self.n_cold_fetches.fetch_add(1, Ordering::AcqRel);
        // Register entry in the cache so subsequent reader()
        // calls hit cache rather than re-entering the
        // coordinator.
        self.cached.insert(*uri, Arc::clone(&entry));

        // 5. Spawn the background finalizer: wait for pwrites,
        //    fsync, rename, mmap, and atomically replace the
        //    cached entry with a mmap-backed reader. On error,
        //    release the manual reservation back to the pool.
        let store = Arc::clone(self);
        let uri_owned = *uri;
        let tmp_owned = tmp.clone();
        let final_owned = final_path.clone();
        let file_owned = Arc::clone(&file);
        tokio::spawn(async move {
            let _ = finalize_to_mmap(
                store,
                uri_owned,
                tmp_owned,
                final_owned,
                file_owned,
                write_handles,
                size,
                reserved_bytes,
            )
            .await;
        });

        Ok(entry)
    }

    /// lazy-foreground cold-fetch coordinator.
    /// Returns immediately with a
    /// [`SuperfileReader::open_lazy`]-built reader over a
    /// [`crate::supertable::StorageRangeSource`]; spawns a
    /// background task that waits for foreground lazy readers
    /// to release before fetching the full superfile, mmap'ing
    /// it, and replacing the cached entry. Subsequent
    /// `reader(uri)` calls return the mmap-backed reader (zero
    /// S3 GETs for any subsequent search).
    /// lazy cold-fetch coordinator. When `offsets` is `Some`,
    /// the cold open uses manifest-provided size/open-batch hints;
    /// when `None`, it falls back to unknown-size suffix-tail
    /// discovery.
    async fn reader_lazy_with_bg_fill_hinted(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
        allow_background_fill: bool,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        if let Some(entry) = self.cached.get(uri) {
            entry.last_access_us.store(self.now_us(), Ordering::Release);
            if allow_background_fill {
                self.maybe_spawn_background_fill(uri, &entry, storage);
            }
            return Ok(Arc::clone(&entry.reader));
        }
        let cell = self
            .coordinators
            .entry(*uri)
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();
        let result = cell
            .get_or_init(|| async {
                let fetch_storage = self.resolve_storage(storage);
                self.cold_fetch_lazy(uri, offsets, fetch_storage).await
            })
            .await;
        let fetch_storage = self.resolve_storage(storage);
        match result {
            Ok(entry) => {
                if allow_background_fill {
                    self.maybe_spawn_background_fill(uri, entry, storage);
                }
                Ok(Arc::clone(&entry.reader))
            }
            Err(_e) => {
                self.coordinators.remove(uri);
                match self.cold_fetch_lazy(uri, offsets, fetch_storage).await {
                    Ok(entry) => {
                        if allow_background_fill {
                            self.maybe_spawn_background_fill(uri, &entry, storage);
                        }
                        Ok(Arc::clone(&entry.reader))
                    }
                    Err(e) => Err(e),
                }
            }
        }
    }

    /// Start parquet/FTS background fill once per URI when an FTS/SQL open
    /// asks for it. Vector opens never call this — they keep block-cache
    /// retention only. Fill skips the vector blob range.
    fn maybe_spawn_background_fill(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        entry: &CachedEntry,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) {
        if skip_background_fill() || entry.mmap.is_some() {
            return;
        }
        if entry
            .fill_spawned
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let size = entry.size_bytes.load(Ordering::Acquire);
        let skip_vec = vector_blob_range(&entry.reader);
        let store = Arc::downgrade(self);
        let reader = Arc::downgrade(&entry.reader);
        let uri_owned = *uri;
        let storage_uri_owned = Self::storage_path(uri);
        let fetch_storage = self.resolve_storage(storage);
        tokio::spawn(async move {
            let _ = lazy_background_fill(
                store,
                reader,
                uri_owned,
                storage_uri_owned,
                size,
                size,
                fetch_storage,
                skip_vec,
            )
            .await;
        });
    }

    /// Lazy cold-fetch path. Foreground builds a reader via
    /// `SuperfileReader::open_lazy_with(StorageRangeSource)`;
    /// background task waits for foreground lazy readers to release,
    /// then downloads the full superfile to NVMe, mmaps it, and replaces
    /// the cache entry.
    ///
    /// If `offsets` is present, the lazy source starts with a known
    /// superfile size and an optional open-batch overlay:
    ///   - with `open_blob`: zero superfile-object GETs at open time,
    ///     because manifest-part fetch already carried the bytes.
    ///   - without `open_blob`: parquet tail + vector + FTS open ranges
    ///     are fetched in one parallel batch.
    ///
    /// If `offsets` is absent, the source starts with unknown size and
    /// discovers it through the first suffix-tail fetch.
    async fn cold_fetch_lazy(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        offsets: Option<&SubsectionOffsets>,
        fetch_storage: Arc<dyn StorageProvider>,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let storage_uri = Self::storage_path(uri);
        let block_source_arc: Arc<BlockCachedSource>;
        let (lazy_reader, size) = if let Some(offsets) = offsets {
            let total_size = offsets.total_size;

            // Match `SuperfileReader::open_lazy_with`'s parquet tail
            // speculation length so the overlay covers the entire
            // upcoming `source.tail()` call.
            let parquet_tail_len = PARQUET_TAIL_SPEC_BYTES.min(total_size);
            let parquet_tail_start = total_size.saturating_sub(parquet_tail_len);

            // Seed the inner lazy readers with exact open-time metadata
            // when the manifest carries it. Older/incomplete hints fall
            // back to fixed headers; the readers then discover the rest.
            let vec_ranges = if !offsets.vec_open_ranges.is_empty() {
                offsets.vec_open_ranges.clone()
            } else {
                match offsets.vec {
                    Some((off, len)) if len > 0 => {
                        vec![(off, VECTOR_OPEN_HEADER_FALLBACK_BYTES.min(len))]
                    }
                    _ => Vec::new(),
                }
            };
            let fts_ranges = if !offsets.fts_open_ranges.is_empty() {
                offsets.fts_open_ranges.clone()
            } else {
                match offsets.fts {
                    Some((off, len)) if len > 0 => {
                        vec![(off, FTS_OPEN_HEADER_FALLBACK_BYTES.min(len))]
                    }
                    _ => Vec::new(),
                }
            };

            // Build the lazy source with the size baked in (no HEAD or suffix
            // discovery), then overlay the open-time byte ranges.
            let inner: Arc<dyn LazyByteSource> = Arc::new(StorageRangeSource::with_known_size(
                Arc::clone(&fetch_storage),
                storage_uri.clone(),
                total_size,
            ));
            let block_source = BlockCachedSource::new_pre_reserved(
                inner,
                Arc::downgrade(self),
                *uri,
                self.blocks_path(uri),
                // FTS subsection reads bypass block rounding (exact ranges);
                // see the `passthrough` field docs.
                offsets.fts,
            );
            block_source_arc = Arc::clone(&block_source);
            let mut overlay = PrefetchedSource::new(block_source);

            if !offsets.open_blob.is_empty() {
                // The open-batch bytes (parquet tail + vector + FTS open
                // ranges) already rode in with the manifest part GET that
                // `cold_open` performed. Install them straight into the
                // overlay: ZERO open-time GETs against the superfile object.
                for (off, bytes) in &offsets.open_blob {
                    overlay.install(*off, Bytes::copy_from_slice(bytes));
                }
            } else {
                // Fallback when no captured open blob is present:
                // fetch the open batch over the wire
                // (parquet tail + vec + fts ranges in parallel, 1 RTT).
                let storage_for_parquet = Arc::clone(&fetch_storage);
                let storage_for_vec = Arc::clone(&fetch_storage);
                let storage_for_fts = Arc::clone(&fetch_storage);
                let parquet_uri = storage_uri.clone();
                let vec_uri = storage_uri.clone();
                let fts_uri = storage_uri.clone();

                let parquet_fut = async move {
                    let end = total_size;
                    let start = parquet_tail_start;
                    if end == start {
                        return Ok::<_, StorageError>(Bytes::new());
                    }
                    storage_for_parquet
                        .get_range(&parquet_uri, start..end)
                        .await
                };
                let vec_fut =
                    async move { fetch_hint_ranges(storage_for_vec, vec_uri, vec_ranges).await };
                let fts_fut =
                    async move { fetch_hint_ranges(storage_for_fts, fts_uri, fts_ranges).await };

                let (parquet_bytes, vec_pre, fts_pre) =
                    futures::try_join!(parquet_fut, vec_fut, fts_fut)?;
                if !parquet_bytes.is_empty() {
                    overlay.install(parquet_tail_start, parquet_bytes);
                }
                for (off, bytes) in vec_pre {
                    overlay.install(off, bytes);
                }
                for (off, bytes) in fts_pre {
                    overlay.install(off, bytes);
                }
            }
            let source: Arc<dyn LazyByteSource> = Arc::new(overlay);

            // Every internal read inside `open_lazy_with` (parquet tail,
            // vec subsection head, fts subsection) hits the overlay sync
            // when the open batch is present. Lazy opens intentionally
            // skip full CRC scans: verifying every subsection would force
            // whole-superfile range reads, defeating the lazy/open-batch
            // path. Eager cache promotion can still verify when it
            // materializes the full superfile.
            let lazy_reader = SuperfileReader::open_lazy_with(
                Arc::clone(&source),
                OpenOptions { verify_crc: false },
            )
            .await?;
            (lazy_reader, total_size)
        } else {
            // Unknown-size path: avoid the cold-open HEAD round-trip.
            // The first `tail()` inside `open_lazy_with` is a native
            // suffix-range GET that returns both footer bytes and total
            // object size, then patches the source's size atomic.
            let range_src: Arc<dyn LazyByteSource> =
                Arc::new(StorageRangeSource::with_unknown_size(
                    Arc::clone(&fetch_storage),
                    storage_uri.clone(),
                ));
            let block_source = BlockCachedSource::new_pre_reserved(
                range_src,
                Arc::downgrade(self),
                *uri,
                self.blocks_path(uri),
                // No manifest hints here, so the FTS subsection is unknown.
                None,
            );
            block_source_arc = Arc::clone(&block_source);
            let source: Arc<dyn LazyByteSource> = block_source;
            let lazy_reader = SuperfileReader::open_lazy_with(
                Arc::clone(&source),
                OpenOptions { verify_crc: false },
            )
            .await?;
            let size = source.size();
            (lazy_reader, size)
        };

        self.reserve_manual(size).await?;

        let lazy_reader = Arc::new(lazy_reader);
        let block_token = block_source_arc.entry_token();
        let entry = Arc::new(CachedEntry {
            reader: Arc::clone(&lazy_reader),
            mmap: None,
            size_bytes: Arc::new(AtomicU64::new(size)),
            accounting: EntryAccounting::Eager,
            block_token: Some(block_token),
            block_source: Some(block_source_arc),
            // Fill is modality-gated via [`Self::maybe_spawn_background_fill`]
            // after the open returns — vector never starts it.
            fill_spawned: AtomicBool::new(false),
            last_access_us: AtomicU64::new(self.now_us()),
        });
        self.n_cold_fetches.fetch_add(1, Ordering::AcqRel);
        self.cached.insert(*uri, Arc::clone(&entry));

        Ok(entry)
    }

    /// Run the cold-fetch coordinator for `uri`. Reserves
    /// budget, fetches, mmap's, registers in `cached`.
    async fn cold_fetch(
        &self,
        uri: &SuperfileUri,
        fetch_storage: Arc<dyn StorageProvider>,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let storage_uri = Self::storage_path(uri);
        let head = fetch_storage.head(&storage_uri).await?;
        let size = head.size;

        // Reserve budget (CAS-loop with eviction on miss).
        let reservation = self.reserve(size).await?;

        // Pump bytes from storage to a sparse destination.
        let tmp = self.tmp_path(uri);
        let final_path = self.cache_path(uri);
        self.cold_fetch_to_disk(&fetch_storage, &storage_uri, &tmp, size)
            .await?;

        // Promote to final path + open as mmap.
        tokio::fs::rename(&tmp, &final_path).await?;
        let mmap = open_readonly_mmap(&final_path).map_err(DiskCacheError::Io)?;
        // Wrap into Arc<Mmap> so the cache's mmap field and
        // the reader's Bytes::from_owner share one mapping.
        let mmap_arc = Arc::new(mmap);
        let bytes = Bytes::from_owner(ArcMmapOwner(Arc::clone(&mmap_arc)));
        let reader = SuperfileReader::open_with(
            bytes,
            OpenOptions {
                verify_crc: self.config.verify_crc_on_open,
            },
        )?;
        let entry = Arc::new(CachedEntry {
            reader: Arc::new(reader),
            mmap: Some(mmap_arc),
            size_bytes: Arc::new(AtomicU64::new(size)),
            accounting: EntryAccounting::Eager,
            block_token: None,
            block_source: None,
            fill_spawned: AtomicBool::new(false),
            last_access_us: AtomicU64::new(self.now_us()),
        });
        self.cached.insert(*uri, Arc::clone(&entry));
        self.n_cold_fetches.fetch_add(1, Ordering::AcqRel);
        reservation.commit();
        Ok(entry)
    }

    /// Same as [`Self::reserve`] but returns just the
    /// reserved-bytes count instead of a borrow-lifetimed
    /// guard. Caller is responsible for either committing
    /// (no-op — the bytes stay reserved as part of a cached
    /// entry) or rolling back via
    /// `self.current_bytes.fetch_sub(bytes, Release)` on
    /// failure. Used by the hybrid cold-fetch path where the
    /// reservation outlives the borrow on `&self` via a
    /// `tokio::spawn`-ed background finalizer.
    async fn reserve_manual(&self, bytes: u64) -> Result<(), DiskCacheError> {
        loop {
            let budget = self.disk_budget_bytes();
            let cur = self.current_bytes.load(Ordering::Acquire);
            if cur + bytes <= budget {
                if self
                    .current_bytes
                    .compare_exchange_weak(cur, cur + bytes, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return Ok(());
                }
                continue;
            }
            let needed = (cur + bytes).saturating_sub(budget);
            self.evict_at_least(needed).await?;
        }
    }

    /// Reserve bytes for block-cache growth.
    pub(super) async fn reserve_block_bytes(&self, bytes: u64) -> Result<(), DiskCacheError> {
        self.reserve_manual(bytes).await
    }

    /// Release previously reserved block-cache bytes.
    pub(super) fn release_block_bytes(&self, bytes: u64) {
        self.current_bytes.fetch_sub(bytes, Ordering::Release);
    }

    /// True when `token` still identifies the live lazy entry for `uri`.
    pub(super) fn lazy_block_entry_is_current(&self, uri: &SuperfileUri, token: &Arc<()>) -> bool {
        self.cached
            .get(uri)
            .and_then(|entry| {
                entry
                    .block_token
                    .as_ref()
                    .map(|current| Arc::ptr_eq(current, token))
            })
            .unwrap_or(false)
    }

    /// Release accounting for one removed cache entry.
    fn release_entry_accounting(&self, entry: &CachedEntry) {
        if entry.accounting == EntryAccounting::Eager {
            self.current_bytes
                .fetch_sub(entry.size_bytes.load(Ordering::Acquire), Ordering::Release);
        }
    }

    #[cfg(test)]
    pub(super) fn install_block_entry_for_test(
        &self,
        uri: SuperfileUri,
        filled: Arc<AtomicU64>,
        block_token: Arc<()>,
    ) {
        let reader =
            SuperfileReader::open(tests::tiny_superfile_bytes()).expect("tiny superfile opens");
        self.cached.insert(
            uri,
            Arc::new(CachedEntry {
                reader: Arc::new(reader),
                mmap: None,
                size_bytes: filled,
                accounting: EntryAccounting::SourceOwned,
                block_token: Some(block_token),
                block_source: None,
                fill_spawned: AtomicBool::new(false),
                last_access_us: AtomicU64::new(self.now_us()),
            }),
        );
    }

    #[cfg(test)]
    pub(super) fn remove_block_entry_for_test(&self, uri: &SuperfileUri) {
        let _ = self.cached.remove(uri);
    }

    /// Reserve `bytes` of disk budget via CAS-loop on
    /// `current_bytes`. On budget pressure runs eviction;
    /// retries until either reserved or `BudgetExceeded`.
    async fn reserve(&self, bytes: u64) -> Result<Reservation<'_>, DiskCacheError> {
        loop {
            let budget = self.disk_budget_bytes();
            let cur = self.current_bytes.load(Ordering::Acquire);
            if cur + bytes <= budget {
                if self
                    .current_bytes
                    .compare_exchange_weak(cur, cur + bytes, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return Ok(Reservation {
                        store: self,
                        bytes,
                        committed: false,
                    });
                }
                // Lost the race; another reservation slipped
                // in. Re-read and retry — most of the time
                // there's still room.
                continue;
            }
            // Over budget — try eviction. If eviction frees
            // enough, the next loop iteration's CAS will
            // succeed.
            let needed = (cur + bytes).saturating_sub(budget);
            self.evict_at_least(needed).await?;
        }
    }

    /// Drive the eviction policy until either `bytes_needed`
    /// is freed or no eligible victims remain (→
    /// `BudgetExceeded`).
    async fn evict_at_least(&self, bytes_needed: u64) -> Result<(), DiskCacheError> {
        // Clone the current pinned_fn out of the mutex
        // before invoking it — the closure itself may
        // acquire other locks (e.g., the supertable's
        // manifest ArcSwap), and holding the cache's
        // pinned_fn mutex across that call invites
        // deadlocks.
        let pinned_fn = {
            let g = self.pinned_fn.lock().expect("pinned_fn mutex poisoned");
            Arc::clone(&g)
        };
        let pinned = pinned_fn();
        let candidates: Vec<EvictionCandidate> = self
            .cached
            .iter()
            .map(|e| EvictionCandidate {
                uri: *e.key(),
                size_bytes: e.value().size_bytes.load(Ordering::Acquire),
                last_access_us: e.value().last_access_us.load(Ordering::Acquire),
            })
            .collect();
        let victims = self
            .config
            .eviction
            .select_for_eviction(&candidates, &pinned, bytes_needed);
        if victims.is_empty() {
            return Err(DiskCacheError::BudgetExceeded);
        }
        for uri in victims {
            // Atomic gate against concurrent eviction: only
            // the caller that wins `DashMap::remove` runs
            // unlink + decrement. Without this gate, two
            // reservations evicting the same victim could
            // double-decrement current_bytes.
            if let Some((_, entry)) = self.cached.remove(&uri) {
                let path = self.cache_path(&uri);
                let _ = fs::remove_file(&path);
                self.release_entry_accounting(&entry);
                self.n_evictions.fetch_add(1, Ordering::AcqRel);
            }
        }
        Ok(())
    }

    /// Fetch `size` bytes from `storage_uri` into `dest_path`
    /// via parallel range-GETs. Mutex-serialized writes; the
    /// fetches are the slow path so the per-write mutex
    /// contention is negligible.
    async fn cold_fetch_to_disk(
        &self,
        fetch_storage: &Arc<dyn StorageProvider>,
        storage_uri: &str,
        dest_path: &Path,
        size: u64,
    ) -> Result<(), DiskCacheError> {
        let n_streams = self.config.cold_fetch_streams.max(1);
        // Fixed chunk size — do NOT scale with `size`. Peak
        // in-flight memory is `n_streams × chunk_size`
        // regardless of superfile size, because the per-fill
        // semaphore below caps concurrent chunks at `n_streams`.
        let chunk_size = self.config.cold_fetch_chunk_bytes.max(1);

        // Preallocate the destination as a plain `std::fs::File`
        // so chunk writers can use positioned (`pwrite`) writes
        // off the async reactor without a shared file lock.
        let file = {
            let f = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(dest_path)?;
            f.set_len(size)?;
            Arc::new(f)
        };

        let n_chunks = if size == 0 {
            0
        } else {
            size.div_ceil(chunk_size)
        };
        // Per-fill concurrency cap: at most `n_streams` chunks
        // hold their fetched `Bytes` resident at once.
        let stream_sem = Arc::new(tokio::sync::Semaphore::new(n_streams));
        let mut joins = Vec::with_capacity(n_chunks as usize);
        for i in 0..n_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(size);
            let storage = Arc::clone(fetch_storage);
            let file = Arc::clone(&file);
            let uri = storage_uri.to_string();
            let stream_sem = Arc::clone(&stream_sem);
            joins.push(tokio::spawn(async move {
                let _permit = stream_sem.acquire_owned().await.map_err(|e| {
                    DiskCacheError::SuperfileOpen(format!("stream semaphore closed: {e}"))
                })?;
                let bytes = storage.get_range(&uri, start..end).await?;
                spawn_blocking(move || file.write_all_at(&bytes, start))
                    .await
                    .map_err(|e| DiskCacheError::SuperfileOpen(format!("write join: {e}")))??;
                Ok::<(), DiskCacheError>(())
            }));
        }
        for h in joins {
            h.await
                .map_err(|e| DiskCacheError::SuperfileOpen(format!("join error: {e}")))??;
        }
        spawn_blocking(move || file.sync_all())
            .await
            .map_err(|e| DiskCacheError::SuperfileOpen(format!("fsync join: {e}")))??;
        Ok(())
    }
}

/// RAII guard for a disk-budget reservation. Drop without
/// `commit()` releases the reserved bytes back to the pool —
/// the caller's reservation never lands.
struct Reservation<'a> {
    store: &'a DiskCacheStore,
    bytes: u64,
    committed: bool,
}

impl<'a> Reservation<'a> {
    fn commit(mut self) {
        self.committed = true;
    }
}

impl<'a> Drop for Reservation<'a> {
    fn drop(&mut self) {
        if !self.committed {
            self.store
                .current_bytes
                .fetch_sub(self.bytes, Ordering::Release);
        }
    }
}

struct PromotionWaitGuard<'a>(&'a AtomicU64);

impl<'a> PromotionWaitGuard<'a> {
    fn new(counter: &'a AtomicU64) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self(counter)
    }
}

impl Drop for PromotionWaitGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Background finalizer for the hybrid cold-fetch. Awaits
/// all pwrites, fsyncs + renames the destination file, mmaps
/// it, and atomically replaces the cache entry with a
/// mmap-backed reader. On failure, releases the disk
/// reservation back to the pool and removes the entry.
async fn finalize_to_mmap(
    store: Arc<DiskCacheStore>,
    uri: SuperfileUri,
    tmp_path: PathBuf,
    final_path: PathBuf,
    file: Arc<tokio::sync::Mutex<tokio::fs::File>>,
    pwrite_handles: Vec<oneshot::Receiver<JoinHandle<Result<(), DiskCacheError>>>>,
    size: u64,
    reserved_bytes: u64,
) -> Result<(), DiskCacheError> {
    let res: Result<(), DiskCacheError> = async {
        // 1. Resolve every pwrite handle through its oneshot,
        //    then await the underlying join.
        for recv in pwrite_handles {
            let handle = recv
                .await
                .map_err(|e| DiskCacheError::SuperfileOpen(format!("pwrite handle: {e}")))?;
            handle
                .await
                .map_err(|e| DiskCacheError::SuperfileOpen(format!("pwrite join: {e}")))??;
        }
        // 2. fsync + drop the file before rename.
        {
            let mut guard = file.lock().await;
            guard.flush().await?;
            guard.sync_all().await?;
        }
        drop(file);
        tokio::fs::rename(&tmp_path, &final_path).await?;
        let mmap = open_readonly_mmap(&final_path)?;
        let mmap_arc = Arc::new(mmap);
        let bytes = Bytes::from_owner(ArcMmapOwner(Arc::clone(&mmap_arc)));
        let reader = SuperfileReader::open_with(
            bytes,
            OpenOptions {
                verify_crc: store.config.verify_crc_on_open,
            },
        )?;
        // Replace the in-memory-backed entry with the
        // mmap-backed one — but **only if it's still
        // present**. The entry may have been evicted by a
        // racing reservation between when this finalizer
        // started and now; in that case we drop the mmap
        // file (eviction already released the reservation
        // via fetch_sub) and don't re-insert. Without this
        // check, the finalizer would silently violate the
        // budget invariant by reinstating an evicted entry.
        match store.cached.entry(uri) {
            Entry::Occupied(mut occ) => {
                *occ.get_mut() = Arc::new(CachedEntry {
                    reader: Arc::new(reader),
                    mmap: Some(mmap_arc),
                    size_bytes: Arc::new(AtomicU64::new(size)),
                    accounting: EntryAccounting::Eager,
                    block_token: None,
                    block_source: None,
                    fill_spawned: AtomicBool::new(false),
                    last_access_us: AtomicU64::new(store.started_at.elapsed().as_micros() as u64),
                });
            }
            Entry::Vacant(_) => {
                let _ = fs::remove_file(&final_path);
            }
        }
        store.coordinators.remove(&uri);
        Ok::<(), DiskCacheError>(())
    }
    .await;
    if res.is_err() {
        // Rollback. Use the same atomic gate as eviction
        // (`cached.remove(uri).is_some()`) so we don't double-
        // decrement when a racing eviction already removed
        // this entry + released its bytes.
        if let Some((_, entry)) = store.cached.remove(&uri) {
            store.release_entry_accounting(&entry);
        }
        store.coordinators.remove(&uri);
    }
    // `reserved_bytes` parameter is retained for future use
    // (e.g., observability counters); the bytes accounting is
    // entirely driven by `cached.remove` gating now.
    let _ = reserved_bytes;
    res
}

async fn fetch_hint_ranges(
    storage: Arc<dyn StorageProvider>,
    storage_uri: String,
    ranges: Vec<(u64, u64)>,
) -> Result<Vec<(u64, Bytes)>, StorageError> {
    try_join_all(
        ranges
            .into_iter()
            .filter(|&(_, len)| len > 0)
            .map(|(off, len)| {
                let storage = Arc::clone(&storage);
                let storage_uri = storage_uri.clone();
                async move {
                    let bytes = storage.get_range(&storage_uri, off..off + len).await?;
                    Ok::<_, StorageError>((off, bytes))
                }
            }),
    )
    .await
}

fn background_store_abandoned(store: &Arc<DiskCacheStore>) -> bool {
    Arc::strong_count(store) == 1
}

async fn wait_for_lazy_foreground_release(
    store: &Weak<DiskCacheStore>,
    reader: &Weak<SuperfileReader>,
) -> Option<Arc<DiskCacheStore>> {
    loop {
        if store.strong_count() == 0 || reader.strong_count() == 0 {
            return None;
        }
        if let Some(strong) = store.upgrade()
            && strong.n_promotion_waiters.load(Ordering::Acquire) > 0
        {
            return Some(strong);
        }
        if reader.strong_count() <= 1 {
            // `strong_count == 1` also occurs briefly while a caller is
            // acquiring the cache entry, so re-check after one scheduler turn.
            tokio::time::sleep(STORE_UPGRADE_RETRY_INTERVAL).await;
            if reader.strong_count() <= 1 {
                return store.upgrade();
            }
            continue;
        }
        tokio::time::sleep(STORE_UPGRADE_RETRY_INTERVAL).await;
    }
}

/// Wait until this URI's lazy reader is held only by the cache entry.
/// Unrelated table/URI fills are not gated here — only this reader's
/// strong-count. A grace re-check covers the open→query handoff.
async fn wait_for_reader_quiescence(
    store: &Arc<DiskCacheStore>,
    reader: &Weak<SuperfileReader>,
) -> bool {
    loop {
        while reader_blocks_background_fill(reader) {
            if background_store_abandoned(store) {
                return false;
            }
            tokio::time::sleep(STORE_UPGRADE_RETRY_INTERVAL).await;
        }
        if reader.strong_count() == 0 {
            return false;
        }
        tokio::time::sleep(STORE_UPGRADE_RETRY_INTERVAL).await;
        if reader.strong_count() == 0 {
            return false;
        }
        if !reader_blocks_background_fill(reader) {
            return !background_store_abandoned(store);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackgroundFillOutcome {
    Complete,
    Paused,
    Abandoned,
}

async fn cold_fetch_to_disk_cancelable(
    store: &Arc<DiskCacheStore>,
    reader: &Weak<SuperfileReader>,
    fetch_storage: &Arc<dyn StorageProvider>,
    storage_uri: &str,
    dest_path: &Path,
    size: u64,
    filled: &mut Vec<bool>,
    skip_vec: Option<(u64, u64)>,
) -> Result<BackgroundFillOutcome, DiskCacheError> {
    let n_streams = store.config.cold_fetch_streams.max(1);
    let chunk_size = store.config.cold_fetch_chunk_bytes.max(1);
    let n_chunks = if size == 0 {
        0
    } else {
        size.div_ceil(chunk_size)
    };
    // `filled` is the resume cursor, owned by the caller across pause/resume:
    // an entry is `true` once its chunk is durably written. On the first
    // attempt it is empty; size it and truncate the destination. On a resume
    // (a same-URI reader paused the previous attempt) it carries the
    // already-written chunks, so the fetch skips them instead of
    // re-downloading the whole object from byte 0.
    let first_attempt = filled.len() != n_chunks as usize;
    if first_attempt {
        filled.clear();
        filled.resize(n_chunks as usize, false);
    }
    let file = {
        let mut opts = fs::OpenOptions::new();
        opts.write(true).create(true);
        if first_attempt {
            opts.truncate(true);
        }
        let file = opts.open(dest_path)?;
        if first_attempt {
            file.set_len(size)?;
        }
        Arc::new(file)
    };

    let mut next_chunk = 0u64;
    let mut in_flight = FuturesUnordered::new();

    // Bound memory by `n_streams × chunk_size` and stop promptly when the
    // short-lived cache that requested this background fill is dropped.
    loop {
        while next_chunk < n_chunks && in_flight.len() < n_streams {
            // Skip chunks a prior attempt already wrote (resume cursor).
            if filled[next_chunk as usize] {
                next_chunk += 1;
                continue;
            }
            if background_store_abandoned(store) {
                return Ok(BackgroundFillOutcome::Abandoned);
            }
            if reader.strong_count() == 0 {
                return Ok(BackgroundFillOutcome::Abandoned);
            }
            if reader_blocks_background_fill(reader) {
                return Ok(BackgroundFillOutcome::Paused);
            }
            let chunk_idx = next_chunk;
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(size);
            // Vector blob stays on the block cache: leave those bytes sparse
            // in the fill file (no GET). Parquet + FTS ranges still download.
            let fetch_ranges = chunk_fetch_ranges(start, end, skip_vec);
            if fetch_ranges.is_empty() {
                filled[chunk_idx as usize] = true;
                next_chunk += 1;
                continue;
            }
            let storage = Arc::clone(fetch_storage);
            let file = Arc::clone(&file);
            let uri = storage_uri.to_string();
            // Tag fill ranges as background so query-window meters attribute
            // only foreground lazy/probe GETs to the cold query cost.
            in_flight.push(async move {
                for (range_start, range_end) in fetch_ranges {
                    let len = range_end - range_start;
                    let bytes =
                        scope_background(storage.get_range(&uri, range_start..range_end)).await?;
                    let file = Arc::clone(&file);
                    spawn_blocking(move || file.write_all_at(&bytes, range_start))
                        .await
                        .map_err(|error| {
                            DiskCacheError::SuperfileOpen(format!("write join: {error}"))
                        })??;
                    let _ = len;
                }
                Ok::<u64, DiskCacheError>(chunk_idx)
            });
            next_chunk += 1;
        }

        let foreground = foreground_notify().notified();
        tokio::pin!(foreground);
        let _ = foreground.as_mut().enable();
        if reader.strong_count() == 0 {
            return Ok(BackgroundFillOutcome::Abandoned);
        }
        if reader_blocks_background_fill(reader) {
            return Ok(BackgroundFillOutcome::Paused);
        }
        tokio::select! {
            biased;
            _ = &mut foreground => {
                // A query started: re-check same-URI hold. Unrelated fills
                // (strong_count == 1) fall through and keep downloading.
                if reader.strong_count() == 0 {
                    return Ok(BackgroundFillOutcome::Abandoned);
                }
                if reader_blocks_background_fill(reader) {
                    return Ok(BackgroundFillOutcome::Paused);
                }
            }
            result = in_flight.next() => match result {
                // Mark the chunk durable only once its write completes, so a
                // pause mid-flight re-fetches just the unfinished chunks.
                Some(result) => filled[result? as usize] = true,
                None => break,
            }
        }
        if background_store_abandoned(store) {
            return Ok(BackgroundFillOutcome::Abandoned);
        }
    }

    if background_store_abandoned(store) {
        return Ok(BackgroundFillOutcome::Abandoned);
    }
    if reader.strong_count() == 0 {
        return Ok(BackgroundFillOutcome::Abandoned);
    }
    if reader_blocks_background_fill(reader) {
        return Ok(BackgroundFillOutcome::Paused);
    }
    spawn_blocking(move || file.sync_all())
        .await
        .map_err(|error| DiskCacheError::SuperfileOpen(format!("fsync join: {error}")))??;
    Ok(BackgroundFillOutcome::Complete)
}

fn rollback_lazy_background_fill(store: &Arc<DiskCacheStore>, uri: &SuperfileUri, tmp: &Path) {
    if let Some((_, entry)) = store.cached.remove(uri) {
        store.release_entry_accounting(&entry);
    }
    store.coordinators.remove(uri);
    let _ = fs::remove_file(tmp);
}

/// Diagnostic gate for measuring lazy foreground reads without promotion,
/// from `diagnostics.disable_background_fill` (YAML-only; no env override).
pub(crate) fn skip_background_fill() -> bool {
    global_config().diagnostics.disable_background_fill
}

/// Promote one released lazy reader to an mmap-backed cache entry.
///
/// When `skip_vec` is set, the fill file leaves the vector blob sparse and
/// promotion opens a hybrid reader: mmap for parquet/FTS, the preserved
/// block-cache source for vector ranges.
async fn lazy_background_fill(
    store: Weak<DiskCacheStore>,
    reader: Weak<SuperfileReader>,
    uri: SuperfileUri,
    storage_uri: String,
    size: u64,
    reserved_bytes: u64,
    fetch_storage: Arc<dyn StorageProvider>,
    skip_vec: Option<(u64, u64)>,
) -> Result<(), DiskCacheError> {
    let Some(store) = wait_for_lazy_foreground_release(&store, &reader).await else {
        return Ok(());
    };
    let tmp = store.tmp_path(&uri);
    let final_path = store.cache_path(&uri);

    if background_store_abandoned(&store) {
        rollback_lazy_background_fill(&store, &uri, &tmp);
        let _ = reserved_bytes;
        return Ok(());
    }

    let _prefetch_permit = match Arc::clone(&store.prefetch_semaphore).acquire_owned().await {
        Ok(permit) => permit,
        Err(error) => {
            rollback_lazy_background_fill(&store, &uri, &tmp);
            return Err(DiskCacheError::SuperfileOpen(format!(
                "prefetch semaphore closed: {error}"
            )));
        }
    };
    // Resume cursor: chunks durably written so far, preserved across
    // pause/resume so a same-URI reader interrupting the fill costs only
    // the unfinished chunks rather than a re-download of the whole object.
    let mut filled: Vec<bool> = Vec::new();
    loop {
        if !wait_for_reader_quiescence(&store, &reader).await {
            rollback_lazy_background_fill(&store, &uri, &tmp);
            return Ok(());
        }
        match cold_fetch_to_disk_cancelable(
            &store,
            &reader,
            &fetch_storage,
            &storage_uri,
            &tmp,
            size,
            &mut filled,
            skip_vec,
        )
        .await?
        {
            BackgroundFillOutcome::Complete => break,
            // Keep the partial `tmp` and the `filled` cursor: the next attempt
            // resumes from the first unwritten chunk.
            BackgroundFillOutcome::Paused => {}
            BackgroundFillOutcome::Abandoned => {
                rollback_lazy_background_fill(&store, &uri, &tmp);
                return Ok(());
            }
        }
    }

    let result: Result<(), DiskCacheError> = async {
        if background_store_abandoned(&store) {
            return Ok(());
        }

        tokio::fs::rename(&tmp, &final_path).await?;
        let mmap = open_readonly_mmap(&final_path)?;
        let mmap_arc = Arc::new(mmap);
        let bytes = Bytes::from_owner(ArcMmapOwner(Arc::clone(&mmap_arc)));

        // Reuse the live block-cache source when excluding the vector blob so
        // touched vector ranges from the cold query stay local after promote.
        let prior_block = store
            .cached
            .get(&uri)
            .and_then(|entry| entry.block_source.clone());
        let (promoted_reader, block_token, block_source) = match (skip_vec, prior_block) {
            (Some((vec_off, vec_len)), Some(block_source)) => {
                let block_token = block_source.entry_token();
                let local: Arc<dyn LazyByteSource> =
                    Arc::new(BytesLazyByteSource::new(bytes.clone()));
                let source: Arc<dyn LazyByteSource> = Arc::new(HoleFallbackSource {
                    local,
                    hole_start: vec_off,
                    hole_len: vec_len,
                    fallback: Arc::clone(&block_source),
                });
                let mut reader =
                    SuperfileReader::open_lazy_with(source, OpenOptions { verify_crc: false })
                        .await?;
                // Sync parquet decodes (take / id scans) run off the mmap;
                // the sparse vector region stays behind the hole source.
                reader.install_resident_parquet(bytes)?;
                (reader, Some(block_token), Some(block_source))
            }
            (Some((vec_off, vec_len)), None) => {
                // Evicted mid-fill: fresh block cache over storage for the hole.
                let remote: Arc<dyn LazyByteSource> =
                    Arc::new(StorageRangeSource::with_known_size(
                        Arc::clone(&fetch_storage),
                        storage_uri.clone(),
                        size,
                    ));
                let block_source = BlockCachedSource::new_pre_reserved(
                    remote,
                    Arc::downgrade(&store),
                    uri,
                    store.blocks_path(&uri),
                    // Serves only the promoted reader's vector hole; FTS
                    // bytes come from the mmap.
                    None,
                );
                let block_token = block_source.entry_token();
                let local: Arc<dyn LazyByteSource> =
                    Arc::new(BytesLazyByteSource::new(bytes.clone()));
                let source: Arc<dyn LazyByteSource> = Arc::new(HoleFallbackSource {
                    local,
                    hole_start: vec_off,
                    hole_len: vec_len,
                    fallback: Arc::clone(&block_source),
                });
                let mut reader =
                    SuperfileReader::open_lazy_with(source, OpenOptions { verify_crc: false })
                        .await?;
                // Sync parquet decodes (take / id scans) run off the mmap;
                // the sparse vector region stays behind the hole source.
                reader.install_resident_parquet(bytes)?;
                (reader, Some(block_token), Some(block_source))
            }
            (None, _) => {
                let reader = SuperfileReader::open_with(
                    bytes,
                    OpenOptions {
                        verify_crc: store.config.verify_crc_on_open,
                    },
                )?;
                (reader, None, None)
            }
        };

        match store.cached.entry(uri) {
            Entry::Occupied(mut occupied) => {
                *occupied.get_mut() = Arc::new(CachedEntry {
                    reader: Arc::new(promoted_reader),
                    mmap: Some(mmap_arc),
                    size_bytes: Arc::new(AtomicU64::new(size)),
                    accounting: EntryAccounting::Eager,
                    block_token,
                    block_source,
                    fill_spawned: AtomicBool::new(true),
                    last_access_us: AtomicU64::new(store.now_us()),
                });
            }
            Entry::Vacant(_) => {
                let _ = fs::remove_file(&final_path);
            }
        }
        store.coordinators.remove(&uri);
        Ok(())
    }
    .await;

    if result.is_err() || background_store_abandoned(&store) {
        rollback_lazy_background_fill(&store, &uri, &tmp);
        let _ = fs::remove_file(&tmp);
    }
    let _ = reserved_bytes;
    result
}

/// Absolute `(offset, length)` of the vector blob from Parquet KV metadata.
fn vector_blob_range(reader: &SuperfileReader) -> Option<(u64, u64)> {
    let kv_map = footer::extract_kv_map(reader.parquet_metadata()).ok()?;
    let off: u64 = kv_map.get(kv::VEC_OFFSET)?.parse().ok()?;
    let len: u64 = kv_map.get(kv::VEC_LENGTH)?.parse().ok()?;
    (len > 0).then_some((off, len))
}

/// Sub-ranges of `[start, end)` that are outside an optional skip hole.
///
/// Empty means the whole chunk lies inside the hole (no GET).
fn chunk_fetch_ranges(start: u64, end: u64, skip: Option<(u64, u64)>) -> Vec<(u64, u64)> {
    debug_assert!(start <= end);
    let Some((hole_start, hole_len)) = skip else {
        return vec![(start, end)];
    };
    if hole_len == 0 || start == end {
        return vec![(start, end)];
    }
    let hole_end = hole_start.saturating_add(hole_len);
    if end <= hole_start || start >= hole_end {
        return vec![(start, end)];
    }
    let mut out = Vec::with_capacity(2);
    if start < hole_start {
        out.push((start, hole_start.min(end)));
    }
    if end > hole_end {
        out.push((hole_end.max(start), end));
    }
    out
}

/// Local mmap/bytes source with a hole that falls through to another source.
///
/// Used after background fill excludes the vector blob: parquet + FTS come
/// from the filled mmap; vector ranges keep using the block cache.
struct HoleFallbackSource {
    local: Arc<dyn LazyByteSource>,
    hole_start: u64,
    hole_len: u64,
    fallback: Arc<BlockCachedSource>,
}

impl HoleFallbackSource {
    fn hole_end(&self) -> u64 {
        self.hole_start.saturating_add(self.hole_len)
    }

    fn overlaps_hole(&self, start: u64, len: u64) -> bool {
        let end = start.saturating_add(len);
        end > self.hole_start && start < self.hole_end()
    }

    fn fully_in_hole(&self, start: u64, len: u64) -> bool {
        let end = start.saturating_add(len);
        start >= self.hole_start && end <= self.hole_end()
    }
}

#[async_trait]
impl LazyByteSource for HoleFallbackSource {
    fn size(&self) -> u64 {
        self.local.size()
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        if len == 0 {
            return Ok(Bytes::new());
        }
        if !self.overlaps_hole(start, len) {
            return self.local.range(start, len).await;
        }
        if self.fully_in_hole(start, len) {
            return self.fallback.range(start, len).await;
        }
        // Spanning request: stitch local and fallback pieces in order.
        let end = start + len;
        let hole_end = self.hole_end();
        let mut pieces = Vec::with_capacity(3);
        let mut cursor = start;
        if cursor < self.hole_start {
            let piece_end = self.hole_start.min(end);
            pieces.push(self.local.range(cursor, piece_end - cursor).await?);
            cursor = piece_end;
        }
        if cursor < end && cursor < hole_end {
            let piece_end = hole_end.min(end);
            pieces.push(self.fallback.range(cursor, piece_end - cursor).await?);
            cursor = piece_end;
        }
        if cursor < end {
            pieces.push(self.local.range(cursor, end - cursor).await?);
        }
        if pieces.len() == 1 {
            return Ok(pieces.pop().expect("one piece"));
        }
        let mut out = Vec::with_capacity(len as usize);
        for piece in pieces {
            out.extend_from_slice(&piece);
        }
        Ok(Bytes::from(out))
    }

    fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
        if len == 0 {
            return Some(Bytes::new());
        }
        if !self.overlaps_hole(start, len) {
            return self.local.try_get_range_sync(start, len);
        }
        if self.fully_in_hole(start, len) {
            return self.fallback.try_get_range_sync(start, len);
        }
        // Spanning sync reads are rare; force the async path.
        None
    }
}

/// Newtype around `Arc<Mmap>` that delegates `AsRef<[u8]>`
/// to the underlying `Mmap`. Lets the cache's `mmap: Arc<Mmap>`
/// field and the reader's `Bytes::from_owner(...)` share the
/// same `Arc<Mmap>` — both refer to the same OS mapping, so
/// `madvise` on the cache's handle affects the reader's
/// resident pages (the idle-threshold sweep relies on this).
struct ArcMmapOwner(Arc<Mmap>);

impl AsRef<[u8]> for ArcMmapOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

fn open_readonly_mmap(path: &Path) -> io::Result<Mmap> {
    let file = fs::File::open(path)?;
    // SAFETY: the cache file is created + filled + fsync'd
    // before this mmap call. The file is owned by us; no
    // other process modifies it. Once mmap'd we never write
    // to it (eviction unlinks + drops the Arc<Mmap>, which
    // unmaps cleanly under POSIX even if the file's already
    // unlinked).
    unsafe { Mmap::map(&file) }
}

/// Open a completed local superfile as zero-copy mmap-backed [`Bytes`].
///
/// Drain assembles very large packed shards in temporary files and maps the
/// finished file through this helper before handing it to the ordinary
/// `prepare_superfile`/publish path. Keeping the unsafe mmap construction in
/// this module preserves the repository's documented mmap safety boundary.
pub(crate) fn mmap_readonly_bytes(path: &Path) -> io::Result<Bytes> {
    let mmap = Arc::new(open_readonly_mmap(path)?);
    Ok(Bytes::from_owner(ArcMmapOwner(mmap)))
}

#[cfg(test)]
mod tests {
    use std::io::Error as IoError;

    use arrow_array::{LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use tempfile::TempDir;
    use tokio::{spawn, task::yield_now, time::timeout};

    use super::*;
    use crate::{
        storage::LocalFsStorageProvider,
        superfile::builder::{BuilderOptions, SuperfileBuilder},
        test_helpers::{decimal128_id_field, decimal128_ids},
    };

    /// Local-filesystem background promotion should finish well within this.
    const PROMOTE_TIMEOUT: Duration = Duration::from_secs(10);
    /// Long enough to cover several background quiet-interval checks.
    const FOREGROUND_GUARD_HOLD: Duration = Duration::from_millis(50);
    /// Large enough that one-byte sequential range reads cannot finish before
    /// the preemption test enters its foreground guard.
    const PREEMPT_TEST_BYTES: usize = 1 << 20;

    /// Build the raw bytes of a minimal superfile (one scalar batch,
    /// no indexes).
    pub(super) fn tiny_superfile_bytes() -> Bytes {
        let schema = Arc::new(Schema::new(vec![
            decimal128_id_field("doc_id"),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![], None);
        let mut b = SuperfileBuilder::new(opts).expect("builder");
        let ids = decimal128_ids(vec![1u64]);
        let titles = LargeStringArray::from(vec!["alpha"]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)]).expect("batch");
        b.add_batch(&batch, &[]).expect("add_batch");
        Bytes::from(b.finish().expect("finish"))
    }

    fn test_store() -> (TempDir, Arc<DiskCacheStore>) {
        test_store_with(|cfg| {
            cfg.mmap_cold_threshold_secs = 0;
        })
    }

    /// Build a store, applying `mutate` to the default config first.
    /// The storage root is the tempdir; cache files live under
    /// `<tempdir>/cache`. The sweep thread is left disabled by
    /// default (callers that want it enable it through `mutate`).
    fn test_store_with(
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

    /// Put `bytes` at the storage location `store.reader(&uri)` will
    /// cold-fetch from, so the cold path has something to read.
    async fn put_superfile(store: &Arc<DiskCacheStore>, uri: &SuperfileUri, bytes: Bytes) {
        store
            .storage
            .put_atomic(&uri.storage_path(), bytes)
            .await
            .expect("put superfile");
    }

    // ----- construction / config -----

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
    async fn is_mmap_promoted_false_for_unknown_uri() {
        let (_dir, store) = test_store();
        assert!(!store.is_mmap_promoted(&SuperfileUri::new_v4()));
    }

    /// `rollback_lazy_background_fill` undoes an in-flight promotion: it drops
    /// the cache entry, forgets the coordinator, and deletes the tmp scratch
    /// file left by the partial download.
    #[tokio::test]
    async fn rollback_lazy_background_fill_evicts_entry_and_tmp() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();

        // Seed a cache entry the way a lazy fill would, plus a leftover tmp
        // scratch file for the partial download.
        store.install_block_entry_for_test(uri, Arc::new(AtomicU64::new(0)), Arc::new(()));
        assert!(
            store.is_cached(&uri),
            "entry must be cached before rollback"
        );
        let tmp = store.tmp_path(&uri);
        std::fs::write(&tmp, b"partial-download-bytes").expect("seed tmp scratch file");
        assert!(tmp.exists(), "tmp scratch file must exist before rollback");

        rollback_lazy_background_fill(&store, &uri, &tmp);

        assert!(
            !store.is_cached(&uri),
            "cached entry must be gone after rollback"
        );
        assert!(
            !tmp.exists(),
            "tmp scratch file must be deleted after rollback"
        );
    }

    // ----- warm insert path (insert_warm + cold-free path) -----

    #[tokio::test]
    async fn insert_warm_caches_and_serves_reader() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        store.insert_warm(&uri, bytes).await.expect("insert_warm");

        // Entry is mmap-backed, counted, and warm inserts don't bump
        // the cold-fetch counter.
        assert!(store.is_mmap_promoted(&uri));
        let s = store.stats();
        assert_eq!(s.n_entries, 1);
        assert_eq!(s.current_bytes, size);
        assert_eq!(s.n_cold_fetches, 0);
        assert_eq!(store.current_mmap_size_bytes(), size);

        // The cache file landed on disk.
        assert!(store.cache_path(&uri).is_file());

        // reader() hits the cache (still no cold fetch).
        let _r = store.reader(&uri).await.expect("reader");
        assert_eq!(store.stats().n_cold_fetches, 0);
    }

    #[tokio::test]
    async fn insert_warm_is_idempotent() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("first");
        let before = store.stats().current_bytes;
        // Second insert with the same URI is a no-op.
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("second");
        assert_eq!(store.stats().current_bytes, before);
        assert_eq!(store.stats().n_entries, 1);
    }

    #[tokio::test]
    async fn insert_warm_rejects_unparseable_bytes() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let err = store
            .insert_warm(&uri, Bytes::from_static(b"not a superfile"))
            .await
            .expect_err("garbage must fail to open");
        // Reservation rolled back on the error path.
        assert_eq!(store.stats().current_bytes, 0);
        assert_eq!(store.stats().n_entries, 0);
        // Surfaced as a typed open/read error.
        let _ = format!("{err}");
        let _ = format!("{err:?}");
    }

    #[tokio::test]
    async fn insert_warm_budget_exceeded_when_too_big() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.disk_budget_bytes = 4; // smaller than any real superfile
        });
        let uri = SuperfileUri::new_v4();
        let err = store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect_err("must exceed budget");
        assert!(matches!(err, DiskCacheError::BudgetExceeded));
        assert_eq!(store.stats().current_bytes, 0);
    }

    // ----- engine-managed (auto-sized) budget reconciliation -----

    /// Tiny explicit budget used to prove reconciliation raises (or
    /// refuses to raise) it; smaller than any real superfile.
    const TEST_TINY_BUDGET_BYTES: u64 = 4;
    /// A comfortably large budget floor for the raise paths.
    const TEST_RAISED_FLOOR_BYTES: u64 = 1 << 20;

    #[tokio::test]
    async fn auto_budget_is_raised_and_admits_previously_oversized_entry() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.disk_budget_bytes = TEST_TINY_BUDGET_BYTES;
        });
        store.mark_budget_auto_sized();
        // Undersized: the tiny superfile cannot be admitted.
        let uri = SuperfileUri::new_v4();
        let err = store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect_err("undersized budget must reject");
        assert!(matches!(err, DiskCacheError::BudgetExceeded));

        // Reconcile raises the auto-sized budget; the same insert succeeds.
        store.reconcile_budget_floor(TEST_RAISED_FLOOR_BYTES, TEST_RAISED_FLOOR_BYTES);
        assert_eq!(store.disk_budget_bytes(), TEST_RAISED_FLOOR_BYTES);
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("raised budget admits the entry");

        // Raise-only: a smaller floor later never lowers the budget.
        store.reconcile_budget_floor(TEST_TINY_BUDGET_BYTES, TEST_TINY_BUDGET_BYTES);
        assert_eq!(store.disk_budget_bytes(), TEST_RAISED_FLOOR_BYTES);
    }

    #[tokio::test]
    async fn explicit_budget_is_never_changed_by_reconcile() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.disk_budget_bytes = TEST_TINY_BUDGET_BYTES;
        });
        // No mark_budget_auto_sized(): the budget is explicit. Reconcile
        // must warn (once) but leave the budget verbatim.
        store.reconcile_budget_floor(TEST_RAISED_FLOOR_BYTES, TEST_RAISED_FLOOR_BYTES);
        store.reconcile_budget_floor(TEST_RAISED_FLOOR_BYTES, TEST_RAISED_FLOOR_BYTES);
        assert_eq!(store.disk_budget_bytes(), TEST_TINY_BUDGET_BYTES);
        assert_eq!(store.stats().budget_bytes, TEST_TINY_BUDGET_BYTES);
    }

    #[tokio::test]
    async fn rebuild_index_from_cache_root_on_open() {
        // A prior handle's cache files on `cache_root` must be reused by a fresh
        // store: the constructor rebuilds the in-memory index from them, so a
        // restart / second handle serves reads off NVMe with no cold-fetch.
        let dir = TempDir::new().expect("tempdir");
        let cache_root = dir.path().join("cache");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;

        // First handle: warm-insert a superfile, then drop it (files persist).
        {
            let cfg = DiskCacheConfig {
                cache_root: cache_root.clone(),
                mmap_cold_threshold_secs: 0,
                ..Default::default()
            };
            let store = DiskCacheStore::new_unpinned(Arc::clone(&storage), cfg).expect("store1");
            store.insert_warm(&uri, bytes).await.expect("insert_warm");
            assert!(store.cache_path(&uri).is_file());
        }

        // Second handle on the SAME cache_root: constructor rebuilds the index.
        let cfg2 = DiskCacheConfig {
            cache_root: cache_root.clone(),
            mmap_cold_threshold_secs: 0,
            ..Default::default()
        };
        let store2 = DiskCacheStore::new_unpinned(Arc::clone(&storage), cfg2).expect("store2");

        let s = store2.stats();
        assert_eq!(s.n_entries, 1, "rebuilt index has the cached superfile");
        assert_eq!(s.current_bytes, size, "rebuilt byte accounting matches");
        assert_eq!(
            s.n_cold_fetches, 0,
            "rebuild mmaps locally, never cold-fetches"
        );

        // A read is served from the rebuilt entry — still zero cold fetches.
        let _r = store2
            .reader(&uri)
            .await
            .expect("reader from rebuilt index");
        assert_eq!(
            store2.stats().n_cold_fetches,
            0,
            "read served from NVMe via rebuilt index, no object-store GET"
        );
    }

    // ----- cold fetch: synchronous path -----

    #[tokio::test]
    async fn reader_synchronous_cold_then_warm_hit() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        put_superfile(&store, &uri, bytes).await;

        let _r = store.reader_synchronous(&uri).await.expect("cold");
        let s = store.stats();
        assert_eq!(s.n_cold_fetches, 1);
        assert_eq!(s.n_entries, 1);
        assert_eq!(s.current_bytes, size);
        // mmap-backed after the synchronous fetch.
        assert!(store.is_mmap_promoted(&uri));

        // Second call is a warm cache hit (no new cold fetch).
        let _r2 = store.reader_synchronous(&uri).await.expect("warm");
        assert_eq!(store.stats().n_cold_fetches, 1);
    }

    #[tokio::test]
    async fn reader_synchronous_missing_object_errors() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        // Nothing put at the storage path → head() fails.
        let err = store.reader_synchronous(&uri).await.expect_err("no object");
        let _ = format!("{err}");
        // Coordinator removed so a later (successful) put can proceed.
        assert!(store.coordinators.is_empty());
    }

    // ----- cold fetch: hybrid path (default mode) -----

    #[tokio::test]
    async fn reader_hybrid_cold_then_stays_lazy_without_full_promotion() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        // reader() dispatches by config default.
        let r = store.reader(&uri).await.expect("cold hybrid");
        assert_eq!(r.n_docs(), 1);
        assert_eq!(store.stats().n_cold_fetches, 1);
        assert_eq!(store.stats().n_entries, 1);

        // Warm path remains lazy/block-backed by design (no full-file barrier).
        assert!(!store.is_mmap_promoted(&uri));

        // Warm hit reuses cached reader; no extra cold fetch.
        let _r2 = store.reader(&uri).await.expect("warm");
        assert_eq!(store.stats().n_cold_fetches, 1);
    }

    #[tokio::test]
    async fn reader_hybrid_empty_object_zero_chunks() {
        // size == 0 takes the n_chunks == 0 branch in cold_fetch_hybrid;
        // the empty buffer fails to parse as a superfile, surfacing an
        // open error rather than a cache entry.
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, Bytes::new()).await;
        let err = store.reader(&uri).await.expect_err("empty not a superfile");
        let _ = format!("{err}");
    }
    #[tokio::test]
    async fn cold_fetch_uses_caller_storage_not_cache_embedded_storage() {
        use crate::storage::{LocalFsStorageProvider, PrefixedStorageProvider};

        let dir = TempDir::new().expect("tempdir");
        let user_storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("user root"));
        let hidden_root = dir.path().join("hidden_prefix");
        std::fs::create_dir_all(&hidden_root).expect("hidden root");
        let hidden_storage: Arc<dyn StorageProvider> = Arc::new(PrefixedStorageProvider::new(
            Arc::clone(&user_storage),
            "hidden_prefix",
        ));

        let cache = DiskCacheStore::new_unpinned(
            Arc::clone(&user_storage),
            DiskCacheConfig {
                cache_root: dir.path().join("cache"),
                mmap_cold_threshold_secs: 0,
                ..Default::default()
            },
        )
        .expect("cache");

        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        hidden_storage
            .put_atomic(&uri.storage_path(), bytes.clone())
            .await
            .expect("put at hidden prefix");

        let reader = cache
            .reader_with_hints(&uri, None, Some(&hidden_storage), true)
            .await
            .expect("cold fetch via caller storage");
        assert_eq!(reader.n_docs(), 1);
        assert_eq!(cache.stats().n_cold_fetches, 1);
    }

    /// Bench default (`LazyForegroundWithBackgroundFill`): the lazy inner
    /// `StorageRangeSource` must honor the caller's prefixed storage, not
    /// the cache's embedded user-root provider.
    #[tokio::test]
    async fn lazy_cold_fetch_uses_caller_storage_not_cache_embedded_storage() {
        use crate::storage::{LocalFsStorageProvider, PrefixedStorageProvider};

        let dir = TempDir::new().expect("tempdir");
        let user_storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("user root"));
        let hidden_root = dir.path().join("hidden_prefix");
        std::fs::create_dir_all(&hidden_root).expect("hidden root");
        let hidden_storage: Arc<dyn StorageProvider> = Arc::new(PrefixedStorageProvider::new(
            Arc::clone(&user_storage),
            "hidden_prefix",
        ));

        let cache = DiskCacheStore::new_unpinned(
            Arc::clone(&user_storage),
            DiskCacheConfig {
                cache_root: dir.path().join("cache"),
                cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
                mmap_cold_threshold_secs: 0,
                ..Default::default()
            },
        )
        .expect("cache");

        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        hidden_storage
            .put_atomic(&uri.storage_path(), bytes.clone())
            .await
            .expect("put at hidden prefix");

        let reader = cache
            .reader_with_hints(&uri, None, Some(&hidden_storage), true)
            .await
            .expect("lazy cold fetch via caller storage");
        assert_eq!(reader.n_docs(), 1);
        assert_eq!(cache.stats().n_cold_fetches, 1);
    }

    /// Compaction opens hidden superfiles through
    /// `reader_synchronous_with_storage`, which must return an eager reader
    /// even when the cache currently holds a lazy entry from query fan-out.
    #[tokio::test]
    async fn reader_synchronous_with_storage_upgrades_lazy_hidden_entry() {
        use crate::storage::{LocalFsStorageProvider, PrefixedStorageProvider};

        let dir = TempDir::new().expect("tempdir");
        let user_storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("user root"));
        let hidden_root = dir.path().join("hidden_prefix");
        std::fs::create_dir_all(&hidden_root).expect("hidden root");
        let hidden_storage: Arc<dyn StorageProvider> = Arc::new(PrefixedStorageProvider::new(
            Arc::clone(&user_storage),
            "hidden_prefix",
        ));

        let cache = DiskCacheStore::new_unpinned(
            Arc::clone(&user_storage),
            DiskCacheConfig {
                cache_root: dir.path().join("cache"),
                cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
                mmap_cold_threshold_secs: 0,
                ..Default::default()
            },
        )
        .expect("cache");

        let uri = SuperfileUri::new_v4();
        hidden_storage
            .put_atomic(&uri.storage_path(), tiny_superfile_bytes())
            .await
            .expect("put at hidden prefix");

        // Query path admission: lazy reader with no resident parquet bytes.
        let lazy = cache
            .reader_with_hints(&uri, None, Some(&hidden_storage), true)
            .await
            .expect("lazy cold fetch via caller storage");
        assert!(
            lazy.parquet_bytes().is_none(),
            "lazy mode should not materialize full parquet bytes"
        );

        // Compaction path must force an eager reopen via caller storage.
        let eager = cache
            .reader_synchronous_with_storage(&uri, Arc::clone(&hidden_storage))
            .await
            .expect("synchronous compaction open");
        assert!(
            eager.parquet_bytes().is_some(),
            "compaction input must have resident parquet bytes"
        );
        let batch = eager
            .get_record_batch(None)
            .expect("compaction should read full RecordBatch");
        assert_eq!(batch.num_rows(), 1);
    }

    // ----- RangeOnly mode rejects + open_range_only bypass -----

    #[test]
    fn reader_range_only_mode_is_rejected() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let cfg = DiskCacheConfig {
            cache_root: dir.path().join("cache"),
            cold_fetch_mode: ColdFetchMode::RangeOnly,
            ..Default::default()
        };
        let err = DiskCacheStore::new_unpinned(storage, cfg)
            .expect_err("range_only + disk cache must be rejected");
        assert!(matches!(err, DiskCacheError::Config(_)));
    }

    #[tokio::test]
    async fn open_range_only_unknown_size_reads_directly() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;
        // offsets = None → unknown-size StorageRangeSource.
        let r = store
            .open_range_only(&uri, None, None)
            .await
            .expect("range open");
        assert_eq!(r.n_docs(), 1);
        // Bypasses the cache entirely — nothing admitted.
        assert_eq!(store.stats().n_entries, 0);
        assert_eq!(store.stats().current_bytes, 0);
    }

    #[tokio::test]
    async fn open_range_only_known_size_reads_directly() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let total = bytes.len() as u64;
        put_superfile(&store, &uri, bytes).await;
        let offsets = SubsectionOffsets {
            total_size: total,
            vec: None,
            fts: None,
            vec_open_ranges: Vec::new(),
            fts_open_ranges: Vec::new(),
            open_blob: Vec::new(),
        };
        let r = store
            .open_range_only(&uri, Some(&offsets), None)
            .await
            .expect("known-size range open");
        assert_eq!(r.n_docs(), 1);
    }

    // ----- lazy-foreground-with-background-fill mode -----

    #[tokio::test]
    async fn reader_lazy_unknown_size_promotes_after_release() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        // reader_with_hints(None) → unknown-size lazy cold fetch.
        let r = store.reader(&uri).await.expect("lazy cold");
        assert_eq!(r.n_docs(), 1);
        assert_eq!(store.stats().n_cold_fetches, 1);

        // Releasing the foreground reader permits the full-file background
        // fill to replace the lazy entry with an mmap-backed reader.
        drop(r);
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("background promotion");
        let r2 = store.reader(&uri).await.expect("warm mmap");
        assert_eq!(store.stats().n_cold_fetches, 1);
        assert!(store.is_mmap_promoted(&uri));
        assert!(r2.parquet_bytes().is_some());
    }

    #[tokio::test]
    async fn reader_lazy_with_hints_known_size_promotes_after_release() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let total = bytes.len() as u64;
        put_superfile(&store, &uri, bytes).await;

        // Known size, no open_blob → fetches the open batch over the
        // wire (parquet tail + vec + fts ranges) using the fallback
        // header lengths derived from `vec`/`fts` hints.
        let offsets = SubsectionOffsets {
            total_size: total,
            vec: None,
            fts: None,
            vec_open_ranges: Vec::new(),
            fts_open_ranges: Vec::new(),
            open_blob: Vec::new(),
        };
        let r = store
            .reader_with_hints(&uri, Some(&offsets), None, true)
            .await
            .expect("lazy hinted cold");
        assert_eq!(r.n_docs(), 1);
        assert_eq!(store.stats().n_cold_fetches, 1);
        drop(r);
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("background promotion");
        let r2 = store
            .reader_with_hints(&uri, Some(&offsets), None, true)
            .await
            .expect("warm hinted mmap");
        assert_eq!(store.stats().n_cold_fetches, 1);
        assert!(store.is_mmap_promoted(&uri));
        assert!(r2.parquet_bytes().is_some());
    }

    #[tokio::test]
    async fn vector_open_skips_fill_fts_open_starts_it() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        // Vector modality: block-cache only — no background fill.
        let vector_reader = store
            .reader_with_hints(&uri, None, None, false)
            .await
            .expect("vector lazy open");
        drop(vector_reader);
        tokio::time::sleep(FOREGROUND_GUARD_HOLD).await;
        assert!(
            !store.is_mmap_promoted(&uri),
            "vector open must not spawn background fill"
        );

        // FTS/SQL modality on the same URI starts fill after the fact.
        let fts_reader = store
            .reader_with_hints(&uri, None, None, true)
            .await
            .expect("fts lazy open");
        drop(fts_reader);
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("FTS open must start background fill");
        assert!(store.is_mmap_promoted(&uri));
    }

    #[tokio::test]
    async fn background_fill_waits_for_same_uri_reader() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        let reader = store.reader(&uri).await.expect("lazy cold");
        let _foreground = ForegroundQueryGuard::enter();
        tokio::time::sleep(FOREGROUND_GUARD_HOLD).await;
        assert!(
            !store.is_mmap_promoted(&uri),
            "background promotion must yield while this URI's lazy reader is held"
        );

        drop(reader);
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("promotion resumes after the URI reader is released");
    }

    #[tokio::test]
    async fn reader_for_one_uri_does_not_pause_another_uri_fill() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
            cfg.prefetch_concurrency = 2;
        });
        let held_uri = SuperfileUri::new_v4();
        let fill_uri = SuperfileUri::new_v4();
        put_superfile(&store, &held_uri, tiny_superfile_bytes()).await;
        put_superfile(&store, &fill_uri, tiny_superfile_bytes()).await;

        let held_reader = store.reader(&held_uri).await.expect("held lazy reader");
        let _fill_reader = store.reader(&fill_uri).await.expect("fill lazy reader");
        drop(_fill_reader);
        let _foreground = ForegroundQueryGuard::enter();
        store
            .wait_until_mmap_promoted(&fill_uri, PROMOTE_TIMEOUT)
            .await
            .expect("unrelated URI fill must proceed while another URI is held");
        assert!(
            !store.is_mmap_promoted(&held_uri),
            "held URI must still wait for its own reader release"
        );
        drop(held_reader);
    }

    /// `HoleFallbackSource` serves ranges outside the vector hole from the
    /// local (filled) bytes and ranges inside the hole from the fallback block
    /// cache, stitching a spanning read from both halves. Covers the geometry
    /// helpers (`overlaps_hole` / `fully_in_hole` / `hole_end`) alongside
    /// `size`, `range`, and the `try_get_range_sync` fast path.
    #[tokio::test]
    async fn hole_fallback_source_routes_local_and_fallback_by_hole() {
        use crate::superfile::lazy_source::BytesLazyByteSource;

        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        // local = the "filled" bytes (0xAA); the fallback's inner = the block
        // cache side (0xBB) that serves the excluded vector hole.
        let local: Arc<dyn LazyByteSource> =
            Arc::new(BytesLazyByteSource::new(Bytes::from(vec![0xAAu8; 100])));
        let remote: Arc<dyn LazyByteSource> =
            Arc::new(BytesLazyByteSource::new(Bytes::from(vec![0xBBu8; 100])));
        let fallback = BlockCachedSource::new_pre_reserved(
            remote,
            Arc::downgrade(&store),
            uri,
            store.blocks_path(&uri),
            None,
        );
        let hfs = HoleFallbackSource {
            local,
            hole_start: 40,
            hole_len: 20,
            fallback,
        };

        assert_eq!(hfs.size(), 100, "size reflects the local (full) source");

        // Wholly before the hole → local bytes.
        assert_eq!(
            &hfs.range(0, 10).await.expect("pre-hole")[..],
            &[0xAAu8; 10]
        );
        // Wholly inside the hole → fallback bytes.
        assert_eq!(
            &hfs.range(40, 20).await.expect("in-hole")[..],
            &[0xBBu8; 20]
        );
        // Spanning: local[30..40] + fallback[40..60] + local[60..70].
        let mut want = vec![0xAAu8; 10];
        want.extend_from_slice(&[0xBBu8; 20]);
        want.extend_from_slice(&[0xAAu8; 10]);
        assert_eq!(
            &hfs.range(30, 40).await.expect("spanning")[..],
            &want[..],
            "spanning read stitches local + fallback + local in order",
        );

        // Sync fast path: a read outside the hole resolves locally; a spanning
        // read returns None to force the async path.
        assert_eq!(
            hfs.try_get_range_sync(0, 10).as_deref(),
            Some(&[0xAAu8; 10][..]),
            "sync read outside the hole comes from local",
        );
        assert!(
            hfs.try_get_range_sync(30, 40).is_none(),
            "spanning sync read forces the async path",
        );
    }

    #[test]
    fn chunk_fetch_ranges_skips_vector_hole() {
        assert_eq!(
            chunk_fetch_ranges(0, 100, None),
            vec![(0, 100)],
            "no hole ⇒ full chunk"
        );
        assert_eq!(
            chunk_fetch_ranges(0, 100, Some((100, 50))),
            vec![(0, 100)],
            "hole after chunk ⇒ full chunk"
        );
        assert_eq!(
            chunk_fetch_ranges(0, 100, Some((0, 100))),
            Vec::<(u64, u64)>::new(),
            "chunk fully inside hole ⇒ no GET"
        );
        assert_eq!(
            chunk_fetch_ranges(50, 150, Some((0, 200))),
            Vec::<(u64, u64)>::new(),
            "chunk fully inside larger hole ⇒ no GET"
        );
        assert_eq!(
            chunk_fetch_ranges(0, 100, Some((40, 20))),
            vec![(0, 40), (60, 100)],
            "hole splits chunk into two fetch ranges"
        );
        assert_eq!(
            chunk_fetch_ranges(0, 100, Some((80, 40))),
            vec![(0, 80)],
            "hole overlapping chunk end ⇒ leading fetch only"
        );
        assert_eq!(
            chunk_fetch_ranges(0, 100, Some((0, 40))),
            vec![(40, 100)],
            "hole overlapping chunk start ⇒ trailing fetch only"
        );
    }

    #[tokio::test]
    async fn same_uri_reader_pauses_in_flight_background_ranges() {
        let (dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_streams = 1;
            cfg.cold_fetch_chunk_bytes = 1;
        });
        let uri = SuperfileUri::new_v4();
        let storage_uri = uri.storage_path();
        store
            .storage
            .put_atomic(&storage_uri, Bytes::from(vec![7u8; PREEMPT_TEST_BYTES]))
            .await
            .expect("put background-fill payload");
        let destination = dir.path().join("preempt.tmp");
        let fill_store = Arc::clone(&store);
        let fill_storage = Arc::clone(&store.storage);
        let fill_destination = destination.clone();
        let signal_reader = Arc::new(
            SuperfileReader::open(tiny_superfile_bytes()).expect("foreground signal reader"),
        );
        let signal_weak = Arc::downgrade(&signal_reader);
        let fill = spawn(async move {
            let mut filled = Vec::new();
            let outcome = cold_fetch_to_disk_cancelable(
                &fill_store,
                &signal_weak,
                &fill_storage,
                &storage_uri,
                &fill_destination,
                PREEMPT_TEST_BYTES as u64,
                &mut filled,
                None,
            )
            .await;
            (outcome, filled)
        });

        timeout(PROMOTE_TIMEOUT, async {
            while !destination.exists() {
                yield_now().await;
            }
        })
        .await
        .expect("background fill started");
        // Holding the signal reader (strong_count > 1) is the per-URI pause.
        let foreground = Arc::clone(&signal_reader);
        let _ = ForegroundQueryGuard::enter();
        let (outcome, filled) = fill.await.expect("background task joined");
        let outcome = outcome.expect("background fill returned an outcome");
        assert_eq!(outcome, BackgroundFillOutcome::Paused);
        // Resume cursor is sized to the object's chunk count and preserved
        // across the pause so a later attempt resumes rather than restarting.
        assert_eq!(filled.len(), PREEMPT_TEST_BYTES);
        assert!(
            filled.iter().any(|&done| !done),
            "a same-URI pause must leave unfinished chunks for the resume"
        );
        drop(foreground);
    }

    // ----- wait_until_mmap_promoted timeout path -----

    #[tokio::test]
    async fn wait_until_mmap_promoted_times_out_for_unpromoted() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        // Never fetched → never promoted → times out.
        let err = store
            .wait_until_mmap_promoted(&uri, Duration::from_millis(30))
            .await
            .expect_err("must time out");
        assert!(matches!(err, DiskCacheError::SuperfileOpen(_)));
        // Guard restored the waiter counter.
        assert_eq!(store.n_promotion_waiters.load(Ordering::Acquire), 0);
    }

    // ----- eviction + budget -----

    #[tokio::test]
    async fn cold_fetch_evicts_lru_when_over_budget() {
        // Budget fits ~1.5 entries, forcing eviction of the older one
        // when the second cold fetch reserves.
        let one = tiny_superfile_bytes();
        let entry_size = one.len() as u64;
        let (_dir, store) = test_store_with(move |cfg| {
            cfg.disk_budget_bytes = entry_size + entry_size / 2;
        });

        let uri_a = SuperfileUri::new_v4();
        let uri_b = SuperfileUri::new_v4();
        put_superfile(&store, &uri_a, tiny_superfile_bytes()).await;
        put_superfile(&store, &uri_b, tiny_superfile_bytes()).await;

        store.reader_synchronous(&uri_a).await.expect("a");
        store.reader_synchronous(&uri_b).await.expect("b");

        // a was the LRU victim; b is resident.
        assert_eq!(store.stats().n_evictions, 1);
        assert!(store.cached.contains_key(&uri_b));
        assert!(!store.cached.contains_key(&uri_a));
        // a's cache file was unlinked.
        assert!(!store.cache_path(&uri_a).exists());
        assert_eq!(store.stats().current_bytes, entry_size);
    }

    #[tokio::test]
    async fn cold_fetch_budget_exceeded_with_all_pinned() {
        let one = tiny_superfile_bytes();
        let entry_size = one.len() as u64;
        let (_dir, store) = test_store_with(move |cfg| {
            cfg.disk_budget_bytes = entry_size + entry_size / 2;
        });

        let uri_a = SuperfileUri::new_v4();
        let uri_b = SuperfileUri::new_v4();
        put_superfile(&store, &uri_a, tiny_superfile_bytes()).await;
        put_superfile(&store, &uri_b, tiny_superfile_bytes()).await;

        // First fetch lands.
        store.reader_synchronous(&uri_a).await.expect("a");
        // Pin everything so eviction finds no victims.
        store.set_pinned_fn(Arc::new(move || {
            let mut s = HashSet::new();
            s.insert(uri_a);
            s
        }));
        let err = store
            .reader_synchronous(&uri_b)
            .await
            .expect_err("no eligible victims");
        assert!(matches!(err, DiskCacheError::BudgetExceeded));
        // a stays put; budget unchanged.
        assert!(store.cached.contains_key(&uri_a));
    }

    // ----- sweep_once / sweep_for_budget / madvise counters -----

    #[tokio::test]
    async fn sweep_once_advises_idle_mmap_entries() {
        // threshold 0 means every entry is immediately "idle".
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("warm");
        let advised = store.sweep_once();
        assert_eq!(advised, 1);
        assert_eq!(store.stats().n_madvise_calls, 1);
        // A second sweep advises again (counter accumulates).
        assert_eq!(store.sweep_once(), 1);
        assert_eq!(store.stats().n_madvise_calls, 2);
    }

    #[tokio::test]
    async fn sweep_once_skips_when_threshold_not_reached() {
        // Large threshold → nothing is idle, so no madvise.
        let (_dir, store) = test_store_with(|cfg| {
            cfg.mmap_cold_threshold_secs = 1_000_000;
        });
        let uri = SuperfileUri::new_v4();
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("warm");
        assert_eq!(store.sweep_once(), 0);
        assert_eq!(store.stats().n_madvise_calls, 0);
    }

    #[tokio::test]
    async fn sweep_for_budget_noop_under_budget() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("warm");
        // budget far above resident size → no madvise.
        assert_eq!(store.sweep_for_budget(u64::MAX), 0);
        assert_eq!(store.stats().n_madvise_calls, 0);
    }

    #[tokio::test]
    async fn sweep_for_budget_reclaims_oldest_first() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("warm");
        let resident = store.current_mmap_size_bytes();
        assert!(resident > 0);
        // budget 0 forces every entry to be advised.
        let advised = store.sweep_for_budget(0);
        assert_eq!(advised, 1);
        assert_eq!(store.stats().n_madvise_calls, 1);
    }

    #[tokio::test]
    async fn current_mmap_size_bytes_zero_when_empty() {
        let (_dir, store) = test_store();
        assert_eq!(store.current_mmap_size_bytes(), 0);
    }

    // ----- error type conversions / Debug -----

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
