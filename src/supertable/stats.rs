// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Observability snapshot.
//!
//! [`Supertable::stats`] returns a [`SupertableStats`] view
//! over the supertable's current load: which manifest version
//! is pinned, how many superfiles and manifest parts are live,
//! and the OS-reported resident memory for this process.
//!
//! Stats are read-once snapshots — no internal counters reset
//! and no allocator hooks engaged. Repeat calls are cheap
//! (one syscall for RSS; everything else is an `ArcSwap::load`
//! plus length reads on the in-memory manifest cache).
//!
//! ## Scope
//!
//! - Manifest-side counters: pinned version, live segment
//!   count, manifest-part counts (referenced + hydrated).
//! - Process-level memory: OS-reported RSS.
//! - Disk-cache aggregates (when a cache is attached): mmap
//!   virtual size, configured budget, cumulative cold-fetch
//!   and eviction counts, current entry count.
//!
//! Per-coordinator counters (range-GETs issued, pwrites in
//! flight, foreground subscribers satisfied) and a heap-
//! resident split-out from RSS are not exposed; they require
//! query-path integration and an allocator-aware shim
//! respectively.

/// Snapshot of supertable load + process memory. Cheap to
/// produce (one RSS syscall + a manifest snapshot read).
/// Returned by-value; callers can clone or destructure freely.
#[derive(Debug, Clone, Default)]
pub struct SupertableStats {
    // ---- Manifest-side observables ---------------------------------
    /// Current pinned `manifest_id`. Monotonically increasing
    /// across commits; `0` for a freshly `create`'d supertable
    /// with no commits.
    pub manifest_id: u64,

    /// Number of superfiles visible to a new reader captured
    /// right now. Equivalent to
    /// `Supertable::reader().n_superfiles()`.
    pub n_superfiles: usize,

    /// Number of manifest parts referenced in the currently
    /// pinned [`crate::supertable::manifest::list::ManifestList`].
    ///
    /// `None` for supertables whose current `Manifest` has no
    /// persisted list — i.e., freshly `create`'d in-process
    /// supertables, or supertables with `options.storage =
    /// None`. Such supertables operate entirely through the
    /// flat `SuperfileList`; the hierarchical part structure is
    /// only meaningful for persisted supertables.
    pub n_manifest_parts: Option<usize>,

    /// Number of manifest parts that have been hydrated in
    /// the in-memory cache. Always ≤ `n_manifest_parts` when
    /// the latter is `Some`. `0` for in-process-only
    /// supertables (no parts exist to hydrate).
    pub n_manifest_parts_loaded: usize,

    // ---- Process memory --------------------------------------------
    /// Resident set size of this **process**, as reported by
    /// the OS (Mach `task_info` on macOS, `/proc/self/statm`
    /// on Linux, etc.). The whole-process RSS includes
    /// allocations from any other supertables, DataFusion
    /// plans, Arrow buffers, the disk cache's mmaps, and
    /// any non-supertable workload sharing this process.
    ///
    /// `0` if the OS-specific accessor fails (very rare on
    /// supported platforms). Stats consumers should treat
    /// `process_rss_bytes == 0` as "unavailable" rather than
    /// "the process has no resident memory".
    pub process_rss_bytes: u64,

    // ---- Disk cache (when attached) --------------------------------
    /// Disk cache's mmap virtual-size sum.
    /// `None` for supertables without a disk cache attached.
    /// Upper bound on the cache's resident memory — actual
    /// RSS contribution is a subset (only faulted pages
    /// that haven't been `madvise(MADV_DONTNEED)`'d by a
    /// sweep). Compare against [`Self::memory_budget_bytes`]
    /// for budget-pressure signals.
    pub mmap_resident_bytes: Option<u64>,

    /// Configured memory budget for the disk cache.
    /// `None` when [`crate::supertable::SupertableOptions::with_memory_budget`]
    /// hasn't been called; in that case the cache runs the
    /// idle-threshold sweep but isn't proactively bounded.
    pub memory_budget_bytes: Option<u64>,

    /// Cumulative count of cold-fetch completions through
    /// the attached disk cache. Each value is the count of
    /// cache misses that finished a backing-store fetch and
    /// produced a `CachedEntry`. Coalesced fetches count
    /// once per shared coordinator, not once per subscriber
    /// — so a thundering-herd scenario looks like one
    /// increment rather than N.
    ///
    /// `None` when no disk cache is attached. `Some(0)` on a
    /// freshly created cache before any cold fetch lands.
    pub n_cold_fetches: Option<u64>,

    /// Cumulative count of disk-cache evictions. Increments
    /// each time an entry is removed by the cache's
    /// eviction policy (budget pressure) or by an explicit
    /// `evict`. The idle-threshold sweep's `madvise` calls do
    /// NOT increment this counter — they're tracked in
    /// [`Self::n_cache_madvise_calls`] separately.
    ///
    /// `None` when no disk cache is attached.
    pub n_cache_evictions: Option<u64>,

    /// Cumulative count of `madvise(MADV_DONTNEED)` calls
    /// issued by the idle-threshold sweep. Higher than
    /// `n_cache_evictions` typical; the sweep drops resident
    /// pages without unmapping, so a single entry can be
    /// `madvise`'d repeatedly across sweeps.
    ///
    /// `None` when no disk cache is attached.
    pub n_cache_madvise_calls: Option<u64>,

    /// Current count of cached entries (one per resident
    /// `SuperfileReader`). Point-in-time snapshot; reads a
    /// `DashMap::len` under the hood.
    ///
    /// `None` when no disk cache is attached.
    pub n_cache_entries: Option<u64>,
}

/// Read the current process's resident set size (RSS) in
/// bytes. Returns `0` if the OS doesn't expose the metric
/// (very rare — `memory-stats` covers macOS, Linux, Windows).
///
/// Cheap: one syscall on Unix (e.g., `task_info` on macOS,
/// read+parse of `/proc/self/statm` on Linux). Allocation-
/// free apart from the optional `String` for the proc
/// filesystem path on Linux (handled inside `memory-stats`).
pub fn process_rss_bytes() -> u64 {
    memory_stats::memory_stats()
        .map(|s| s.physical_mem as u64)
        .unwrap_or(0)
}
