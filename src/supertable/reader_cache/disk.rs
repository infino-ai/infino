// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`DiskCacheStore`] — Tier 1 cache wrapping a
//! [`StorageProvider`] with parallel cold-fetch + LRU
//! eviction.

use std::collections::HashSet;
use std::io::SeekFrom;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use bytes::Bytes;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use futures::stream::{FuturesUnordered, StreamExt};
use thiserror::Error;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::OnceCell;

use super::config::{ColdFetchMode, DiskCacheConfig, EvictionCandidate};
use crate::storage::{StorageError, StorageProvider};
use crate::superfile::PrefetchedSource;
use crate::superfile::reader::{OpenOptions, SuperfileReader};
use crate::supertable::manifest::{SubsectionOffsets, SuperfileUri};

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

/// Yield cadence in the background-fill upgrade loop. Gives a
/// short-lived caller (e.g. a cold benchmark with a fresh cache per
/// iteration) a scheduling turn to drop the cache before a
/// full-superfile fill starts.
const STORE_UPGRADE_RETRY_INTERVAL: Duration = Duration::from_millis(10);

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
    mmap: Option<Arc<memmap2::Mmap>>,
    size_bytes: u64,
    last_access_us: AtomicU64,
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
    /// Global cap on concurrent background superfile fills. Each
    /// background fill acquires one permit for its whole
    /// duration; foreground per-query range reads don't. Sized
    /// `config.prefetch_concurrency` at construction.
    prefetch_semaphore: Arc<tokio::sync::Semaphore>,
}

impl std::fmt::Debug for DiskCacheStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskCacheStore")
            .field("cache_root", &self.config.cache_root)
            .field("budget_bytes", &self.config.disk_budget_bytes)
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
        std::fs::create_dir_all(&config.cache_root)?;
        let threshold_secs = config.mmap_cold_threshold_secs;
        let interval_secs = config.mmap_sweep_interval_secs.max(1);
        let prefetch_semaphore = Arc::new(tokio::sync::Semaphore::new(
            config.prefetch_concurrency.max(1),
        ));
        let store = Arc::new(Self {
            storage,
            config,
            started_at: Instant::now(),
            cached: DashMap::new(),
            coordinators: DashMap::new(),
            current_bytes: AtomicU64::new(0),
            n_cold_fetches: AtomicU64::new(0),
            n_evictions: AtomicU64::new(0),
            n_madvise_calls: AtomicU64::new(0),
            n_promotion_waiters: AtomicU64::new(0),
            pinned_fn: std::sync::Mutex::new(pinned_fn),
            prefetch_semaphore,
        });

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
            let _ = std::thread::Builder::new()
                .name("infino-disk-cache-sweep".into())
                .spawn(move || {
                    loop {
                        std::thread::sleep(std::time::Duration::from_secs(interval_secs));
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
        let snapshot: Vec<(SuperfileUri, Arc<memmap2::Mmap>, u64)> = self
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
                let _ = unsafe { mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
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

    /// Hot path. Cached → cloned `Arc<SuperfileReader>`; cold
    /// → coalesced cold-fetch coordinator. Dispatches by
    /// `config.cold_fetch_mode`:
    ///
    /// - [`ColdFetchMode::HybridWithPrefetch`] (default):
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
        self.reader_with_hints(uri, None).await
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
    /// `None` falls back to the 2-RTT shape — same shape,
    /// slower. The other cold-fetch modes (`HybridWithPrefetch`,
    /// `RangeOnly`) ignore the hint today.
    pub async fn reader_with_hints(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        offsets: Option<&SubsectionOffsets>,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        match self.config.cold_fetch_mode {
            ColdFetchMode::HybridWithPrefetch => self.reader_hybrid(uri).await,
            ColdFetchMode::RangeOnly => Err(DiskCacheError::SuperfileOpen(
                "ColdFetchMode::RangeOnly bypasses the disk cache; \
                 construct StorageRangeSource + open_lazy directly"
                    .into(),
            )),
            ColdFetchMode::LazyForegroundWithBackgroundFill => {
                self.reader_lazy_with_bg_fill_hinted(uri, offsets.cloned())
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
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        let storage_uri = Self::storage_path(uri);
        let range_src: Arc<dyn crate::superfile::LazyByteSource> = match offsets {
            Some(o) if o.total_size > 0 => {
                Arc::new(crate::supertable::StorageRangeSource::with_known_size(
                    Arc::clone(&self.storage),
                    storage_uri,
                    o.total_size,
                ))
            }
            _ => Arc::new(crate::supertable::StorageRangeSource::with_unknown_size(
                Arc::clone(&self.storage),
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
        if let Some(entry) = self.cached.get(uri) {
            entry.last_access_us.store(self.now_us(), Ordering::Release);
            return Ok(Arc::clone(&entry.reader));
        }
        let cell = self
            .coordinators
            .entry(*uri)
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();
        let result = cell
            .get_or_init(|| async { self.cold_fetch(uri).await })
            .await;
        match result {
            Ok(entry) => {
                self.coordinators.remove(uri);
                Ok(Arc::clone(&entry.reader))
            }
            Err(_e) => {
                self.coordinators.remove(uri);
                Err(self
                    .cold_fetch(uri)
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
            .get_or_init(|| async { self.cold_fetch_hybrid(uri).await })
            .await;
        match result {
            Ok(entry) => Ok(Arc::clone(&entry.reader)),
            Err(DiskCacheError::BudgetExceeded) => {
                self.coordinators.remove(uri);
                Err(DiskCacheError::BudgetExceeded)
            }
            Err(_) => {
                self.coordinators.remove(uri);
                self.cold_fetch_hybrid(uri)
                    .await
                    .map(|entry| Arc::clone(&entry.reader))
            }
        }
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

    /// Snapshot of the cache's load. Cheap; reads atomics +
    /// a `DashMap::len` (which itself is `O(shards)`).
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            n_entries: self.cached.len() as u64,
            current_bytes: self.current_bytes.load(Ordering::Acquire),
            budget_bytes: self.config.disk_budget_bytes,
            n_cold_fetches: self.n_cold_fetches.load(Ordering::Acquire),
            n_evictions: self.n_evictions.load(Ordering::Acquire),
            n_madvise_calls: self.n_madvise_calls.load(Ordering::Acquire),
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
        let mut candidates: Vec<(SuperfileUri, Arc<memmap2::Mmap>, u64, u64)> = self
            .cached
            .iter()
            .filter_map(|e| {
                let mmap = e.value().mmap.clone()?;
                Some((
                    *e.key(),
                    mmap,
                    e.value().last_access_us.load(Ordering::Acquire),
                    e.value().size_bytes,
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
            let _ = unsafe { mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
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

            // Write the bytes to a tmp file, fsync, then
            // atomically rename into place. Same shape as
            // the cold-fetch path's tmp→final promote.
            {
                let mut file = tokio::fs::File::create(&tmp).await?;
                file.write_all(&bytes).await?;
                file.flush().await?;
                file.sync_all().await?;
            }
            tokio::fs::rename(&tmp, &final_path).await?;

            // mmap the freshly-written file. Same Arc<Mmap>
            // shared between CachedEntry.mmap and the
            // reader's Bytes::from_owner so a later
            // MADV_DONTNEED sweep touches the same mapping.
            let mmap = open_readonly_mmap(&final_path).map_err(DiskCacheError::Io)?;
            let mmap_arc = Arc::new(mmap);
            let reader_bytes = Bytes::from_owner(ArcMmapOwner(Arc::clone(&mmap_arc)));
            let reader = SuperfileReader::open_with(
                reader_bytes,
                OpenOptions {
                    verify_crc: self.config.verify_crc_on_open,
                },
            )?;

            let entry = Arc::new(CachedEntry {
                reader: Arc::new(reader),
                mmap: Some(mmap_arc),
                size_bytes: size,
                last_access_us: AtomicU64::new(self.now_us()),
            });
            Ok(entry)
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
            dashmap::mapref::entry::Entry::Vacant(v) => {
                v.insert(entry);
            }
            dashmap::mapref::entry::Entry::Occupied(_) => {
                // Lost the race; release our reservation +
                // unlink the just-written file (or leave it
                // — the existing entry mmaps a different
                // file on disk).
                self.current_bytes.fetch_sub(size, Ordering::Release);
                let _ = std::fs::remove_file(self.cache_path(uri));
            }
        }
        Ok(())
    }

    // ----- internals -----

    fn now_us(&self) -> u64 {
        self.started_at.elapsed().as_micros() as u64
    }

    /// Build a per-URI cache file path under `cache_root`.
    fn cache_path(&self, uri: &SuperfileUri) -> PathBuf {
        self.config.cache_root.join(uri.cache_filename())
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
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let storage_uri = Self::storage_path(uri);
        let head = self.storage.head(&storage_uri).await?;
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
            let storage = Arc::clone(&self.storage);
            let file = Arc::clone(&file);
            let chunks = Arc::clone(&chunks);
            let uri_s = storage_uri.clone();

            // Spawn the fetch task. It captures a Sender for
            // its pwrite handle so the outer task can join
            // pwrites separately from fetches.
            let (write_tx, write_rx) = tokio::sync::oneshot::channel::<
                tokio::task::JoinHandle<Result<(), DiskCacheError>>,
            >();
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
            size_bytes: size,
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
        offsets: Option<SubsectionOffsets>,
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
        let result = cell
            .get_or_init(|| async { self.cold_fetch_lazy(uri, offsets.as_ref()).await })
            .await;
        match result {
            Ok(entry) => Ok(Arc::clone(&entry.reader)),
            Err(_e) => {
                self.coordinators.remove(uri);
                Err(self
                    .cold_fetch_lazy(uri, offsets.as_ref())
                    .await
                    .err()
                    .unwrap_or(DiskCacheError::SuperfileOpen(
                        "lazy cold fetch error".into(),
                    )))
            }
        }
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
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let storage_uri = Self::storage_path(uri);
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

            // Build the lazy source: a `StorageRangeSource` with the
            // size baked in (no HEAD, no suffix-range discovery)
            // wrapped in a `PrefetchedSource` overlay carrying the
            // open-time byte ranges at their absolute offsets.
            let inner: Arc<dyn crate::superfile::LazyByteSource> =
                Arc::new(crate::supertable::StorageRangeSource::with_known_size(
                    Arc::clone(&self.storage),
                    storage_uri.clone(),
                    total_size,
                ));
            let mut overlay = PrefetchedSource::new(inner);

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
                let storage_for_parquet = Arc::clone(&self.storage);
                let storage_for_vec = Arc::clone(&self.storage);
                let storage_for_fts = Arc::clone(&self.storage);
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
            let source: Arc<dyn crate::superfile::LazyByteSource> = Arc::new(overlay);

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
            let range_src: Arc<dyn crate::superfile::LazyByteSource> =
                Arc::new(crate::supertable::StorageRangeSource::with_unknown_size(
                    Arc::clone(&self.storage),
                    storage_uri.clone(),
                ));
            let lazy_reader = SuperfileReader::open_lazy_with(
                Arc::clone(&range_src),
                OpenOptions { verify_crc: false },
            )
            .await?;
            let size = range_src.size();
            (lazy_reader, size)
        };

        self.reserve_manual(size).await?;
        let reserved_bytes = size;

        let lazy_reader = Arc::new(lazy_reader);
        let entry = Arc::new(CachedEntry {
            reader: Arc::clone(&lazy_reader),
            mmap: None,
            size_bytes: size,
            last_access_us: AtomicU64::new(self.now_us()),
        });
        self.n_cold_fetches.fetch_add(1, Ordering::AcqRel);
        self.cached.insert(*uri, Arc::clone(&entry));

        // Background promotion waits until foreground lazy readers
        // release before starting full-superfile fetch, so cache fill
        // does not compete with query-critical range GETs.
        if !skip_background_fill() {
            let store = Arc::downgrade(self);
            let reader = Arc::downgrade(&lazy_reader);
            let uri_owned = *uri;
            let storage_uri_owned = storage_uri;
            tokio::spawn(async move {
                let _ = lazy_background_fill(
                    store,
                    reader,
                    uri_owned,
                    storage_uri_owned,
                    size,
                    reserved_bytes,
                )
                .await;
            });
        }

        Ok(entry)
    }

    /// Run the cold-fetch coordinator for `uri`. Reserves
    /// budget, fetches, mmap's, registers in `cached`.
    async fn cold_fetch(&self, uri: &SuperfileUri) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let storage_uri = Self::storage_path(uri);
        let head = self.storage.head(&storage_uri).await?;
        let size = head.size;

        // Reserve budget (CAS-loop with eviction on miss).
        let reservation = self.reserve(size).await?;

        // Pump bytes from storage to a sparse destination.
        let tmp = self.tmp_path(uri);
        let final_path = self.cache_path(uri);
        self.cold_fetch_to_disk(&storage_uri, &tmp, size).await?;

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
            size_bytes: size,
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
            let cur = self.current_bytes.load(Ordering::Acquire);
            if cur + bytes <= self.config.disk_budget_bytes {
                if self
                    .current_bytes
                    .compare_exchange_weak(cur, cur + bytes, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return Ok(());
                }
                continue;
            }
            let needed = (cur + bytes).saturating_sub(self.config.disk_budget_bytes);
            self.evict_at_least(needed).await?;
        }
    }

    /// Reserve `bytes` of disk budget via CAS-loop on
    /// `current_bytes`. On budget pressure runs eviction;
    /// retries until either reserved or `BudgetExceeded`.
    async fn reserve(&self, bytes: u64) -> Result<Reservation<'_>, DiskCacheError> {
        loop {
            let cur = self.current_bytes.load(Ordering::Acquire);
            if cur + bytes <= self.config.disk_budget_bytes {
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
            let needed = (cur + bytes).saturating_sub(self.config.disk_budget_bytes);
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
                size_bytes: e.value().size_bytes,
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
                let _ = std::fs::remove_file(&path);
                self.current_bytes
                    .fetch_sub(entry.size_bytes, Ordering::Release);
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
        storage_uri: &str,
        dest_path: &std::path::Path,
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
            let f = std::fs::OpenOptions::new()
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
            let storage = Arc::clone(&self.storage);
            let file = Arc::clone(&file);
            let uri = storage_uri.to_string();
            let stream_sem = Arc::clone(&stream_sem);
            joins.push(tokio::spawn(async move {
                let _permit = stream_sem.acquire_owned().await.map_err(|e| {
                    DiskCacheError::SuperfileOpen(format!("stream semaphore closed: {e}"))
                })?;
                let bytes = storage.get_range(&uri, start..end).await?;
                tokio::task::spawn_blocking(move || file.write_all_at(&bytes, start))
                    .await
                    .map_err(|e| DiskCacheError::SuperfileOpen(format!("write join: {e}")))??;
                Ok::<(), DiskCacheError>(())
            }));
        }
        for h in joins {
            h.await
                .map_err(|e| DiskCacheError::SuperfileOpen(format!("join error: {e}")))??;
        }
        tokio::task::spawn_blocking(move || file.sync_all())
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
    tmp_path: std::path::PathBuf,
    final_path: std::path::PathBuf,
    file: Arc<tokio::sync::Mutex<tokio::fs::File>>,
    pwrite_handles: Vec<
        tokio::sync::oneshot::Receiver<tokio::task::JoinHandle<Result<(), DiskCacheError>>>,
    >,
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
                    size_bytes: size,
                    last_access_us: AtomicU64::new(store.started_at.elapsed().as_micros() as u64),
                });
            }
            Entry::Vacant(_) => {
                let _ = std::fs::remove_file(&final_path);
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
            store
                .current_bytes
                .fetch_sub(entry.size_bytes, Ordering::Release);
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
    futures::future::try_join_all(ranges.into_iter().filter(|&(_, len)| len > 0).map(
        |(off, len)| {
            let storage = Arc::clone(&storage);
            let storage_uri = storage_uri.clone();
            async move {
                let bytes = storage.get_range(&storage_uri, off..off + len).await?;
                Ok::<_, StorageError>((off, bytes))
            }
        },
    ))
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
            // Give short-lived callers (notably cold benchmarks with a
            // fresh cache per iteration) a scheduling turn to drop the
            // cache before we start a full-superfile background fill.
            tokio::time::sleep(STORE_UPGRADE_RETRY_INTERVAL).await;
            // Re-check after the grace sleep: `strong_count == 1` is
            // also what acquisition looks like from outside — between
            // `cold_fetch_lazy` dropping its local Arc and the caller
            // cloning out of the cache entry, the entry briefly holds
            // the only reference. A poll landing in that window (or a
            // caller re-acquiring during the sleep) must NOT start the
            // fill while the reader is held; keep waiting instead.
            if reader.strong_count() <= 1 {
                return store.upgrade();
            }
            continue;
        }
        tokio::time::sleep(STORE_UPGRADE_RETRY_INTERVAL).await;
    }
}

async fn cold_fetch_to_disk_cancelable(
    store: &Arc<DiskCacheStore>,
    storage_uri: &str,
    dest_path: &std::path::Path,
    size: u64,
) -> Result<bool, DiskCacheError> {
    let n_streams = store.config.cold_fetch_streams.max(1);
    let chunk_size = store.config.cold_fetch_chunk_bytes.max(1);
    let file = {
        let f = std::fs::OpenOptions::new()
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
    let mut next_chunk = 0u64;
    let mut in_flight = FuturesUnordered::new();

    // Fetch the full superfile in bounded chunks instead of a whole-object
    // `get()`: large superfiles can be hundreds of MiB to many GiB, and
    // materializing them as one `Bytes` would reintroduce the RSS spike
    // this disk-cache path is meant to avoid.
    loop {
        while next_chunk < n_chunks && in_flight.len() < n_streams {
            if background_store_abandoned(store) {
                return Ok(false);
            }
            let start = next_chunk * chunk_size;
            let end = (start + chunk_size).min(size);
            let storage = Arc::clone(&store.storage);
            let file = Arc::clone(&file);
            let uri = storage_uri.to_string();
            in_flight.push(async move {
                let bytes = storage.get_range(&uri, start..end).await?;
                tokio::task::spawn_blocking(move || file.write_all_at(&bytes, start))
                    .await
                    .map_err(|e| DiskCacheError::SuperfileOpen(format!("write join: {e}")))??;
                Ok::<(), DiskCacheError>(())
            });
            next_chunk += 1;
        }

        match in_flight.next().await {
            Some(res) => res?,
            None => break,
        }
        if background_store_abandoned(store) {
            return Ok(false);
        }
    }

    if background_store_abandoned(store) {
        return Ok(false);
    }
    // `write_all_at` writes directly through an unbuffered `std::fs::File`;
    // there is no `BufWriter` layer to flush before syncing durability.
    tokio::task::spawn_blocking(move || file.sync_all())
        .await
        .map_err(|e| DiskCacheError::SuperfileOpen(format!("fsync join: {e}")))??;
    Ok(true)
}

fn rollback_lazy_background_fill(
    store: &Arc<DiskCacheStore>,
    uri: &SuperfileUri,
    tmp: &std::path::Path,
) {
    if let Some((_, entry)) = store.cached.remove(uri) {
        store
            .current_bytes
            .fetch_sub(entry.size_bytes, Ordering::Release);
    }
    store.coordinators.remove(uri);
    let _ = std::fs::remove_file(tmp);
}

/// background promotion path for the
/// `LazyForegroundWithBackgroundFill` cold-fetch mode.
/// Waits for foreground lazy readers to release, downloads the
/// full superfile via cancelable parallel range-GETs to NVMe,
/// mmaps the resulting file, and atomically replaces the lazy
/// cache entry with a mmap-backed reader. Subsequent
/// `reader(uri)` calls hit the promoted entry — every query
/// resolves from mmap (zero S3 GETs).
/// Diagnostic gate for the `LazyForegroundWithBackgroundFill`
/// full-superfile promotion. When `INFINO_DISABLE_BG_FILL=1` (or
/// `true`), the cold-fetch path installs the open-blob overlay
/// and serves the foreground query over range GETs, but never
/// spawns the full-superfile background download. Lets us measure
/// the cold fan-out cost in isolation from the competing
/// full-superfile fills.
pub(crate) fn skip_background_fill() -> bool {
    static SKIP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SKIP.get_or_init(|| {
        std::env::var("INFINO_DISABLE_BG_FILL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

async fn lazy_background_fill(
    store: Weak<DiskCacheStore>,
    reader: Weak<SuperfileReader>,
    uri: SuperfileUri,
    storage_uri: String,
    size: u64,
    reserved_bytes: u64,
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

    // Global background-fill concurrency cap. Held for the whole
    // fill so process-wide background memory is bounded by
    // `prefetch_concurrency × (cold_fetch_streams ×
    // cold_fetch_chunk_bytes)`. Acquired before any GET; foreground
    // per-query reads never touch this semaphore.
    let _prefetch_permit = match Arc::clone(&store.prefetch_semaphore).acquire_owned().await {
        Ok(p) => p,
        Err(e) => {
            store.coordinators.remove(&uri);
            if let Some((_, entry)) = store.cached.remove(&uri) {
                store
                    .current_bytes
                    .fetch_sub(entry.size_bytes, Ordering::Release);
            }
            return Err(DiskCacheError::SuperfileOpen(format!(
                "prefetch semaphore closed: {e}"
            )));
        }
    };

    let res: Result<(), DiskCacheError> = async {
        if background_store_abandoned(&store) {
            return Ok(());
        }
        // 1. Parallel range-GETs into the temp file (existing
        //    cold_fetch_to_disk shape, but cancelable for
        //    short-lived caches).
        if !cold_fetch_to_disk_cancelable(&store, &storage_uri, &tmp, size).await? {
            return Ok(());
        }
        if background_store_abandoned(&store) {
            return Ok(());
        }

        // 2. Promote to final path + mmap.
        tokio::fs::rename(&tmp, &final_path).await?;
        let mmap = open_readonly_mmap(&final_path)?;
        let mmap_arc = Arc::new(mmap);
        let bytes = Bytes::from_owner(ArcMmapOwner(Arc::clone(&mmap_arc)));
        let reader = SuperfileReader::open_with(
            bytes,
            OpenOptions {
                verify_crc: store.config.verify_crc_on_open,
            },
        )?;

        // 3. Atomically replace the lazy entry with the
        //    mmap-backed one — but only if it's still
        //    present (a racing eviction may have removed it
        //    + released the reservation in the meantime).
        match store.cached.entry(uri) {
            Entry::Occupied(mut occ) => {
                *occ.get_mut() = Arc::new(CachedEntry {
                    reader: Arc::new(reader),
                    mmap: Some(mmap_arc),
                    size_bytes: size,
                    last_access_us: AtomicU64::new(store.now_us()),
                });
            }
            Entry::Vacant(_) => {
                let _ = std::fs::remove_file(&final_path);
            }
        }
        store.coordinators.remove(&uri);
        Ok::<(), DiskCacheError>(())
    }
    .await;

    if res.is_err() || background_store_abandoned(&store) {
        // Rollback — same atomic gate as eviction so we don't
        // double-decrement when a racing eviction already
        // removed this entry + released its bytes.
        rollback_lazy_background_fill(&store, &uri, &tmp);
        // Clean up the temp file if cold_fetch_to_disk failed
        // mid-way.
        let _ = std::fs::remove_file(&tmp);
    }
    let _ = reserved_bytes; // retained for future observability
    res
}

/// Newtype around `Arc<Mmap>` that delegates `AsRef<[u8]>`
/// to the underlying `Mmap`. Lets the cache's `mmap: Arc<Mmap>`
/// field and the reader's `Bytes::from_owner(...)` share the
/// same `Arc<Mmap>` — both refer to the same OS mapping, so
/// `madvise` on the cache's handle affects the reader's
/// resident pages (the idle-threshold sweep relies on this).
struct ArcMmapOwner(Arc<memmap2::Mmap>);

impl AsRef<[u8]> for ArcMmapOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

fn open_readonly_mmap(path: &std::path::Path) -> std::io::Result<memmap2::Mmap> {
    let file = std::fs::File::open(path)?;
    // SAFETY: the cache file is created + filled + fsync'd
    // before this mmap call. The file is owned by us; no
    // other process modifies it. Once mmap'd we never write
    // to it (eviction unlinks + drops the Arc<Mmap>, which
    // unmaps cleanly under POSIX even if the file's already
    // unlinked).
    unsafe { memmap2::Mmap::map(&file) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LocalFsStorageProvider;
    use crate::superfile::SuperfileReader;
    use crate::superfile::builder::{BuilderOptions, SuperfileBuilder};
    use crate::test_helpers::{decimal128_id_field, decimal128_ids};
    use arrow_array::{LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use tempfile::TempDir;

    /// Poll turns the test lets elapse while the reader is held — the
    /// pre-fix code returned after a single grace sleep, so anything
    /// > 1 distinguishes "kept waiting" from "barreled ahead".
    const HELD_POLL_TURNS: u32 = 5;

    /// Minimal eager superfile reader (one scalar batch, no indexes).
    fn tiny_reader() -> Arc<SuperfileReader> {
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
        let bytes = Bytes::from(b.finish().expect("finish"));
        Arc::new(SuperfileReader::open(bytes).expect("open"))
    }

    fn test_store() -> (TempDir, Arc<DiskCacheStore>) {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let cfg = DiskCacheConfig {
            cache_root: dir.path().join("cache"),
            ..Default::default()
        };
        let store = DiskCacheStore::new_unpinned(storage, cfg).expect("store");
        (dir, store)
    }

    /// Regression: `strong_count == 1` is also what *acquisition*
    /// looks like — between `cold_fetch_lazy` dropping its local Arc
    /// and the caller cloning out of the cache entry, the entry
    /// briefly holds the only reference. The pre-fix wait returned
    /// unconditionally after its grace sleep, so a poll landing in
    /// that window started the full-segment fill while the foreground
    /// reader was held (the CI flake in
    /// `lazy_background_fill_waits_for_foreground_reader_drop`).
    /// The wait must re-check after the grace sleep and keep waiting.
    #[tokio::test(start_paused = true)]
    async fn wait_for_release_rechecks_reader_after_grace_sleep() {
        let (_dir, store) = test_store();
        let reader = tiny_reader();

        let weak_store = Arc::downgrade(&store);
        let weak_reader = Arc::downgrade(&reader);
        // At spawn the only strong ref is `reader` itself — the exact
        // shape of the acquisition window (the entry's ref, caller's
        // clone not yet taken).
        let waiter = tokio::spawn(async move {
            wait_for_lazy_foreground_release(&weak_store, &weak_reader).await
        });

        // Let the waiter observe count == 1 and enter its grace sleep.
        tokio::time::sleep(Duration::from_millis(1)).await;
        // The caller's clone lands while the waiter is mid-grace: the
        // reader is now genuinely held.
        let held = Arc::clone(&reader);

        // Let several poll intervals elapse (paused time auto-advances
        // through every sleep). The buggy wait returned after ONE.
        tokio::time::sleep(STORE_UPGRADE_RETRY_INTERVAL * HELD_POLL_TURNS).await;
        assert!(
            !waiter.is_finished(),
            "background fill must keep waiting while the foreground reader is held"
        );

        // Release the hold (one strong ref — the entry's — remains):
        // the wait must now complete and hand back the store.
        drop(held);
        let got = waiter.await.expect("waiter join");
        assert!(
            got.is_some(),
            "wait must yield the store once the foreground hold releases"
        );
        drop(reader);
    }
}
