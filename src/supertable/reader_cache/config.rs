// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Disk-cache runtime configuration + pluggable eviction
//! policy.

use std::{
    collections::HashSet,
    env::temp_dir,
    fmt::{self, Debug},
    path::PathBuf,
    sync::atomic::AtomicU64,
};

use crate::supertable::manifest::SuperfileUri;

/// How `DiskCacheStore` services a cache miss.
///
/// Set via `DiskCacheConfig::cold_fetch_mode`. Default:
/// [`ColdFetchMode::LazyForegroundWithBackgroundFill`] — object-storage-native
/// deployments make cold-query p50 the primary objective, and the foreground
/// then pays only the per-query range budget while the full-superfile cache
/// fill happens off the latency-critical path (warm stays resident).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColdFetchMode {
    /// Parallel range-GETs fan out over the superfile; each
    /// response is Arc-cloned and tee'd to (a) the foreground
    /// caller (in-memory `SuperfileReader`) and (b) a
    /// fire-and-forget pwrite into the cache file. Foreground
    /// returns when all range-fetches complete; pwrites +
    /// mmap + cache registration finalize in the background.
    ///
    /// **1× bandwidth per cold miss** — same range responses
    /// serve both consumers; no re-fetching between foreground
    /// query and cache fill.
    HybridWithPrefetch,
    /// Foreground query goes straight through `get_range` via
    /// [`StorageRangeSource`] — no disk-cache fill.
    /// Best for stateless / query-once callers where the
    /// cache-fill bandwidth is wasted.
    ///
    /// [`StorageRangeSource`]: crate::supertable::StorageRangeSource
    RangeOnly,
    /// foreground returns immediately with a
    /// [`SuperfileReader::open_lazy`]-built reader over a
    /// [`StorageRangeSource`]; pays only the cold-open
    /// + cold-search byte budget against object storage
    /// (~6 GETs / ~2-3 MiB on a typical 1.5 GiB superfile).
    /// A background task downloads the full superfile to NVMe
    /// after foreground lazy readers release, then mmaps it
    /// + replaces the cache entry;
    /// any subsequent `reader(uri)` call returns the
    /// mmap-backed reader and the corresponding search issues
    /// **zero** S3 GETs.
    ///
    /// **Up to 2× bandwidth per cold miss** — foreground
    /// per-query ranges and the eventual full-superfile cache
    /// fill both read from object storage, but the fill is
    /// deferred until the latency-critical foreground reader
    /// is dropped. The tradeoff: minimal cold-query latency
    /// (one-superfile hot working set fits in a few range-GETs)
    /// at the cost of extra cold-fetch bandwidth vs.
    /// `HybridWithPrefetch`.
    /// Pick this mode for object-storage-native deployments
    /// where cold-query p50 latency is the primary objective.
    ///
    /// [`SuperfileReader::open_lazy`]: crate::superfile::reader::SuperfileReader::open_lazy
    /// [`StorageRangeSource`]: crate::supertable::StorageRangeSource
    #[default]
    LazyForegroundWithBackgroundFill,
}

/// Runtime configuration for [`super::DiskCacheStore`].
///
/// Owns the eviction policy via `Box<dyn CacheEvictionPolicy>`
/// — this is the runtime-side type (not the serde-side
/// `DiskCacheSettings` from `Config`; one converts to the
/// other at supertable construction).
pub struct DiskCacheConfig {
    /// Filesystem root for cached superfile files. Created
    /// (recursively) at `DiskCacheStore::new`.
    pub cache_root: PathBuf,
    /// Tier 1 size cap. Soft cap — exceeded transiently
    /// during a reservation that races with eviction; the
    /// CAS-loop reservation primitive keeps the steady state
    /// bounded.
    pub disk_budget_bytes: u64,
    /// How a cache miss is serviced. See [`ColdFetchMode`].
    pub cold_fetch_mode: ColdFetchMode,
    /// Parallel range-GET streams per cold miss.
    pub cold_fetch_streams: usize,
    /// Range-GET chunk size in bytes. Smaller = more
    /// parallelism, larger = fewer HTTP round-trips. The
    /// product `cold_fetch_streams × cold_fetch_chunk_bytes`
    /// bounds peak in-flight memory per cold miss — the
    /// chunk size is fixed at this value regardless of
    /// superfile size, so a large superfile fans out into more
    /// chunks rather than inflating per-chunk memory.
    pub cold_fetch_chunk_bytes: u64,
    /// Global cap on concurrent **background** superfile fills
    /// (the `LazyForegroundWithBackgroundFill` full-superfile
    /// download). Each in-flight fill is itself bounded to
    /// `cold_fetch_streams × cold_fetch_chunk_bytes`, so the
    /// process-wide background-fill memory ceiling is
    /// `prefetch_concurrency × cold_fetch_streams ×
    /// cold_fetch_chunk_bytes`. Foreground per-query range
    /// reads do not count against this cap. Default 8.
    pub prefetch_concurrency: usize,
    /// Idle threshold (seconds) past which a cached entry's
    /// mmap pages get `MADV_DONTNEED`'d by the background
    /// sweep thread. Default 300 s. Set to `0` to
    /// disable the sweep entirely — useful for tests / for
    /// callers that don't want the background thread.
    pub mmap_cold_threshold_secs: u64,
    /// How often the sweep thread runs (seconds). Default
    /// `mmap_cold_threshold_secs / 4` (~75 s at the 300 s
    /// default threshold). Has no effect when
    /// `mmap_cold_threshold_secs == 0`.
    pub mmap_sweep_interval_secs: u64,
    /// Pluggable eviction policy. Default: [`LruPolicy`].
    pub eviction: Box<dyn CacheEvictionPolicy>,
    /// Whether the cache's `SuperfileReader::open` calls
    /// verify CRC. Default `true`. Plumbed independently
    /// from the supertable's own `verify_crc_on_open` so
    /// callers constructing a `DiskCacheStore` directly can
    /// set the right value for their storage backend; the
    /// supertable typically sets both knobs from the same
    /// `Config::supertable::verify_crc_on_open` source.
    pub verify_crc_on_open: bool,
}

/// Default Tier-1 disk-cache byte budget (10 GiB).
const DEFAULT_DISK_BUDGET_BYTES: u64 = 10 * (1 << 30);
/// Default number of parallel range-GET streams per cold miss.
const DEFAULT_COLD_FETCH_STREAMS: usize = 16;
/// Default range-GET chunk size (16 MiB); peak in-flight bytes is
/// `streams × chunk`.
const DEFAULT_COLD_FETCH_CHUNK_BYTES: u64 = 16 * (1 << 20);
/// Default number of concurrent background full-superfile fills.
const DEFAULT_PREFETCH_CONCURRENCY: usize = 8;
/// Default idle age (seconds) before an mmap is `MADV_DONTNEED`-swept.
const DEFAULT_MMAP_COLD_THRESHOLD_SECS: u64 = 300;
/// Default background mmap-sweep period (seconds, ≈ threshold / 4).
const DEFAULT_MMAP_SWEEP_INTERVAL_SECS: u64 = 75;

impl Default for DiskCacheConfig {
    fn default() -> Self {
        Self {
            cache_root: temp_dir().join("infino-disk-cache"),
            disk_budget_bytes: DEFAULT_DISK_BUDGET_BYTES,
            cold_fetch_mode: ColdFetchMode::default(),
            cold_fetch_streams: DEFAULT_COLD_FETCH_STREAMS,
            cold_fetch_chunk_bytes: DEFAULT_COLD_FETCH_CHUNK_BYTES,
            prefetch_concurrency: DEFAULT_PREFETCH_CONCURRENCY,
            mmap_cold_threshold_secs: DEFAULT_MMAP_COLD_THRESHOLD_SECS,
            mmap_sweep_interval_secs: DEFAULT_MMAP_SWEEP_INTERVAL_SECS,
            eviction: Box::new(LruPolicy::new()),
            verify_crc_on_open: true,
        }
    }
}

impl Debug for DiskCacheConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiskCacheConfig")
            .field("cache_root", &self.cache_root)
            .field("disk_budget_bytes", &self.disk_budget_bytes)
            .field("cold_fetch_mode", &self.cold_fetch_mode)
            .field("cold_fetch_streams", &self.cold_fetch_streams)
            .field("cold_fetch_chunk_bytes", &self.cold_fetch_chunk_bytes)
            .field("prefetch_concurrency", &self.prefetch_concurrency)
            .field("mmap_cold_threshold_secs", &self.mmap_cold_threshold_secs)
            .field("mmap_sweep_interval_secs", &self.mmap_sweep_interval_secs)
            .field("eviction", &"<dyn CacheEvictionPolicy>")
            .finish()
    }
}

/// What an eviction policy needs to know about each cached
/// entry to choose victims.
#[derive(Debug, Clone)]
pub struct EvictionCandidate {
    pub uri: SuperfileUri,
    pub size_bytes: u64,
    /// Microseconds-since-construction at which this entry
    /// was last accessed. Monotonic per `DiskCacheStore`
    /// instance.
    pub last_access_us: u64,
}

/// Pluggable eviction policy. Used by [`super::DiskCacheStore`]
/// when a cold-fetch reservation can't fit in the disk budget
/// and needs to free room.
///
/// Implementations are pure functions — given the current
/// candidate set + the pinned set + the bytes required, return
/// a list of victims to evict. The store does the actual
/// drop + unlink under an atomic gate (DashMap::remove).
pub trait CacheEvictionPolicy: Send + Sync {
    /// Choose victims totaling at least `bytes_needed` from
    /// `candidates`, **excluding** any URI in `pinned`.
    ///
    /// Returns an empty `Vec` if no eligible victims can free
    /// enough — the caller surfaces `CacheBudgetExceeded`,
    /// which the query layer folds into a
    /// `ColdFetchMode::RangeOnly` fallback.
    ///
    /// Order of returned URIs is the eviction order — the
    /// store unlinks them in sequence and stops as soon as
    /// `bytes_needed` is freed.
    fn select_for_eviction(
        &self,
        candidates: &[EvictionCandidate],
        pinned: &HashSet<SuperfileUri>,
        bytes_needed: u64,
    ) -> Vec<SuperfileUri>;
}

/// Least-recently-accessed eviction policy. The default — works
/// well for the typical "recent superfiles are queried more often
/// than old ones" pattern. Workload-specific policies (e.g.,
/// LFU, ARC, S3-FIFO) can swap this out via
/// [`DiskCacheConfig::eviction`].
#[derive(Debug, Default)]
pub struct LruPolicy {
    /// Monotonic counter — used in tests to keep the policy
    /// deterministic. Default impl just reads `last_access_us`
    /// from the candidates so this field is currently unused.
    _seq: AtomicU64,
}

impl LruPolicy {
    pub fn new() -> Self {
        Self::default()
    }
}

impl CacheEvictionPolicy for LruPolicy {
    fn select_for_eviction(
        &self,
        candidates: &[EvictionCandidate],
        pinned: &HashSet<SuperfileUri>,
        bytes_needed: u64,
    ) -> Vec<SuperfileUri> {
        // Filter pinned, sort by ascending last_access_us
        // (oldest first), take until cumulative size ≥
        // bytes_needed.
        let mut eligible: Vec<&EvictionCandidate> = candidates
            .iter()
            .filter(|c| !pinned.contains(&c.uri))
            .collect();
        eligible.sort_by_key(|c| c.last_access_us);
        let mut victims = Vec::new();
        let mut freed = 0u64;
        for c in eligible {
            if freed >= bytes_needed {
                break;
            }
            victims.push(c.uri);
            freed = freed.saturating_add(c.size_bytes);
        }
        if freed < bytes_needed {
            // Couldn't free enough — return empty so the caller
            // surfaces CacheBudgetExceeded.
            Vec::new()
        } else {
            victims
        }
    }
}
