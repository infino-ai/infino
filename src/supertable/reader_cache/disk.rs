//! [`DiskCacheStore`] — Tier 1 cache wrapping a
//! [`StorageProvider`] with parallel cold-fetch + LRU
//! eviction.

use std::collections::HashSet;
use std::io::SeekFrom;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use bytes::Bytes;
use dashmap::DashMap;
use thiserror::Error;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::OnceCell;

use super::config::{ColdFetchMode, DiskCacheConfig, EvictionCandidate};
use crate::storage::{StorageError, StorageProvider};
use crate::superfile::reader::{OpenOptions, SuperfileReader};
use crate::supertable::manifest::SuperfileUri;

/// Errors surfaced by [`DiskCacheStore::reader`].
#[derive(Debug, Error)]
pub enum DiskCacheError {
    #[error("storage error during cold fetch")]
    Storage(#[from] StorageError),
    #[error("local filesystem error: {0}")]
    Io(#[from] std::io::Error),
    #[error("superfile reader failed to open mmap'd bytes: {0}")]
    SuperfileOpen(String),
    /// Eviction couldn't free enough space because every
    /// cached entry was pinned (or there were no cached
    /// entries and the incoming segment alone exceeds the
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

/// Pulls segment bytes through a [`StorageProvider`] and
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
            pinned_fn: std::sync::Mutex::new(pinned_fn),
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
        match self.config.cold_fetch_mode {
            ColdFetchMode::HybridWithPrefetch => self.reader_hybrid(uri).await,
            ColdFetchMode::RangeOnly => Err(DiskCacheError::SuperfileOpen(
                "ColdFetchMode::RangeOnly bypasses the disk cache; \
                 construct StorageRangeSource + open_lazy directly"
                    .into(),
            )),
        }
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
            Err(_e) => {
                self.coordinators.remove(uri);
                Err(self.cold_fetch_hybrid(uri).await.err().unwrap_or(
                    DiskCacheError::SuperfileOpen("hybrid cold fetch error".into()),
                ))
            }
        }
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
    /// after the cache has been moved into the supertable
    /// (M14b.1). The new closure takes effect on the next
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
    /// [`crate::supertable::Supertable::stats`] (M14c) to
    /// report `mmap_resident_bytes` and to drive the
    /// budget-aware sweep in [`Self::sweep_for_budget`].
    pub fn current_mmap_size_bytes(&self) -> u64 {
        self.cached
            .iter()
            .filter_map(|e| e.value().mmap.as_ref().map(|m| m.len() as u64))
            .sum()
    }

    /// M14c — drop mmap pages until the cache's working set
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
    /// range-fetch + pwrite + mmap, ~50-150 ms per segment on
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
        // same UUID segment got re-inserted into the cache).
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
            )
            .map_err(|e| DiskCacheError::SuperfileOpen(e.to_string()))?;

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
        self.config.cache_root.join(format!("seg-{}.sf", uri.0))
    }

    /// Build a per-URI tempfile path (sparse destination
    /// during cold fetch; renamed to `cache_path` on success).
    fn tmp_path(&self, uri: &SuperfileUri) -> PathBuf {
        self.config.cache_root.join(format!("seg-{}.sf.tmp", uri.0))
    }

    /// The storage-side URI for a segment, mirroring the
    /// writer's persist layout.
    fn storage_path(uri: &SuperfileUri) -> String {
        format!("data/seg-{}.sf", uri.0)
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
            for slot in chunks_guard.iter() {
                if let Some((start, bytes)) = slot {
                    let s = *start as usize;
                    let e = s + bytes.len();
                    buf[s..e].copy_from_slice(bytes);
                }
            }
            buf
        };
        let foreground_bytes = Bytes::from(buffer);
        let foreground_reader = SuperfileReader::open_with(
            foreground_bytes,
            OpenOptions {
                verify_crc: self.config.verify_crc_on_open,
            },
        )
        .map_err(|e| DiskCacheError::SuperfileOpen(e.to_string()))?;
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
        )
        .map_err(|e| DiskCacheError::SuperfileOpen(e.to_string()))?;
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
        let n_streams = self.config.cold_fetch_streams.max(1) as u64;
        let chunk_size = self
            .config
            .cold_fetch_chunk_bytes
            .max(size.div_ceil(n_streams));
        let file = tokio::fs::File::create(dest_path).await?;
        file.set_len(size).await?;
        let file = Arc::new(tokio::sync::Mutex::new(file));

        let n_chunks = if size == 0 {
            0
        } else {
            size.div_ceil(chunk_size)
        };
        let mut joins = Vec::with_capacity(n_chunks as usize);
        for i in 0..n_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(size);
            let storage = Arc::clone(&self.storage);
            let file = Arc::clone(&file);
            let uri = storage_uri.to_string();
            joins.push(tokio::spawn(async move {
                let bytes = storage.get_range(&uri, start..end).await?;
                let mut guard = file.lock().await;
                guard.seek(SeekFrom::Start(start)).await?;
                guard.write_all(&bytes).await?;
                Ok::<(), DiskCacheError>(())
            }));
        }
        for h in joins {
            h.await
                .map_err(|e| DiskCacheError::SuperfileOpen(format!("join error: {e}")))??;
        }
        let mut guard = file.lock().await;
        guard.flush().await?;
        guard.sync_all().await?;
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
    #[allow(dead_code)]
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
        )
        .map_err(|e| DiskCacheError::SuperfileOpen(e.to_string()))?;
        // Replace the in-memory-backed entry with the
        // mmap-backed one — but **only if it's still
        // present**. The entry may have been evicted by a
        // racing reservation between when this finalizer
        // started and now; in that case we drop the mmap
        // file (eviction already released the reservation
        // via fetch_sub) and don't re-insert. Without this
        // check, the finalizer would silently violate the
        // budget invariant by reinstating an evicted entry.
        use dashmap::mapref::entry::Entry;
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
