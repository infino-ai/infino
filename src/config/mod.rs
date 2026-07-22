// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! System-wide configuration for infino.
//!
//! ## Sources
//!
//! [`Config::load`] merges, in increasing precedence:
//!
//!   1. **Embedded defaults.** `config.yaml` in this module is
//!      `include_str!`'d at compile time. Shipping with the binary
//!      means there's always a usable floor.
//!   2. **`/etc/infino/config.yaml`** — system-wide override.
//!   3. **User config.** `$XDG_CONFIG_HOME/infino/config.yaml`
//!      (or `$HOME/.config/infino/config.yaml` if `XDG_CONFIG_HOME`
//!      is unset).
//!   4. **`./infino.yaml`** — per-project / per-cwd override.
//!
//! Each layer is a partial override — keys absent from a higher
//! layer fall through to lower layers.
//!
//! **Environment variables never override config.** Engine behavior
//! is set exclusively in YAML so a run's effective configuration is
//! readable from files, not reconstructed from process env. (Env
//! overrides existed once and produced silent drift between runs;
//! `env_vars_do_not_override_config` pins the removal.)
//!
//! ## Adding a new field
//!
//! 1. Add the field to [`Config`] with a `serde` rename / default
//!    if appropriate.
//! 2. Add the same key to `config.yaml` with its default value.
//! 3. Add a docstring and a unit test exercising the YAML override
//!    path.

use std::{
    collections::HashMap,
    env, fmt,
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Duration,
};

use figment::{
    Figment,
    providers::{Format, Yaml},
};
use serde::{
    Deserialize, Serialize,
    de::{self, Deserializer, Visitor},
    ser::Serializer,
};

use crate::superfile::vector::rerank_codec::RerankCodec;

/// Embedded baseline. Compiled in via `include_str!`.
const EMBEDDED_DEFAULT: &str = include_str!("config.yaml");

/// Engine default connection budget when none is configured; used by both
/// [`MemorySettings`] and the connect path. `0` is the deliberate measure-only
/// (no-ceiling) sentinel that `from_budget_bytes` maps to a measured budget.
///
/// A future non-trivial default (e.g. a fraction of system RAM) changes here.
/// `from_budget_bytes` stays a pure value mapper; the only added work then is
/// letting the config field distinguish "unset" from an explicit `0`.
pub(crate) const DEFAULT_CONNECTION_BUDGET_BYTES: u64 = 0;

/// Errors from config load + validation.
///
/// `figment::Error` is ~200 bytes; boxing keeps the `Result` size
/// small (clippy `result_large_err`) and gives us room to add
/// validation variants later.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("config load failed: {0}")]
    Figment(Box<figment::Error>),
}

impl From<figment::Error> for ConfigError {
    fn from(e: figment::Error) -> Self {
        Self::Figment(Box::new(e))
    }
}

/// System-wide infino settings.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct Config {
    /// Supertable runtime knobs (thread pools, id column,
    /// commit threshold).
    #[serde(default)]
    pub supertable: SupertableSettings,
    /// Storage backend and disk-cache wiring. Defaults to
    /// in-memory-only; object-store deployments set this to
    /// `backend: s3` plus a bucket/prefix.
    #[serde(default)]
    pub storage: StorageSettings,
    /// Compaction settings.
    #[serde(default)]
    pub compaction: CompactionSettings,
    /// Vector-index build / search / drain tuning knobs.
    #[serde(default)]
    pub vector: VectorSettings,
    /// Diagnostic and hardware-capability toggles. These gate
    /// instrumentation (timers / tracing) or force a slower code
    /// path for A/B measurement; none of them change query
    /// results. Default: everything off.
    #[serde(default)]
    pub diagnostics: DiagnosticsSettings,
    /// Per-connection memory budget.
    #[serde(default)]
    pub memory: MemorySettings,
}

/// Memory subsection of [`Config`]. All memory-related settings for Infino here.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct MemorySettings {
    /// Per-connection memory (heap) budget in bytes. `0` (the default) is
    /// measure-only: usage is tracked but never refused. A positive value
    /// enforces a ceiling so one connection can't exhaust process memory.
    ///
    /// Applies to connections built from a config file (`apply_config`). Code
    /// that opens a connection programmatically sets the budget on
    /// [`ConnectOptions::with_connection_memory_budget_bytes`] instead.
    ///
    /// [`ConnectOptions::with_connection_memory_budget_bytes`]: crate::ConnectOptions::with_connection_memory_budget_bytes
    pub connection_budget_bytes: u64,
}

impl Default for MemorySettings {
    fn default() -> Self {
        Self {
            connection_budget_bytes: DEFAULT_CONNECTION_BUDGET_BYTES,
        }
    }
}

/// Process-wide config, loaded once from the standard hierarchy
/// (see [`Config::load`]) on first access and cached for the life
/// of the process.
///
/// This is the source for tuning knobs that are read deep in leaf
/// code paths — SIMD dispatch, the I/O timeline, the vector drain —
/// where threading a per-table [`crate::supertable::SupertableOptions`]
/// down to the read site isn't practical. Such knobs were previously
/// bespoke `std::env::var("INFINO_…")` reads; they now live in
/// [`Config`] and are read from here, so YAML alone controls them.
///
/// Load failure falls back to the embedded defaults so a read site
/// never panics on a malformed host config.
pub fn global() -> &'static Config {
    static GLOBAL: OnceLock<Config> = OnceLock::new();
    GLOBAL.get_or_init(|| Config::load().unwrap_or_default())
}

/// Supertable subsection of [`Config`]. Keeps supertable-
/// specific knobs grouped so they don't crowd the top-level
/// namespace as the layer grows.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct SupertableSettings {
    /// Reader fan-out pool size. `auto` resolves to `num_cpus`.
    pub reader_threads: ThreadCount,
    /// Writer commit-shard pool size. `auto` resolves to
    /// `max(1, num_cpus / 2)`.
    pub writer_threads: ThreadCount,
    /// Name of the system-managed primary-key column the
    /// supertable injects on every `append()`. Type is fixed
    /// at the supertable layer; this knob is only the column
    /// name as it appears in the schema and in SQL queries.
    /// Leading underscore signals a system-owned field —
    /// callers can override (e.g. `row_id`, `uuid`) when
    /// `_id` collides with a business field name, but the
    /// column type and generation semantics don't change.
    pub id_column: String,
    /// Threshold above which the supertable's writer triggers
    /// an internal `commit()` to flush the in-memory buffer.
    /// In mebibytes (1 MiB == 1024 × 1024 bytes). `0`
    /// disables auto-flush — only caller-driven `commit()`
    /// produces superfiles.
    pub commit_threshold_size_mb: u64,
    /// Verify the trailing whole-blob CRC and per-subsection
    /// CRCs on every `SuperfileReader::open`. Defaults to
    /// `true`. Set to `false` only when the underlying
    /// storage already validates checksums (content-
    /// addressed object store, ZFS, etc.) — skipping the
    /// scan trades that storage-layer guarantee for faster
    /// cold opens.
    pub verify_crc_on_open: bool,
}

impl Default for SupertableSettings {
    fn default() -> Self {
        Self {
            reader_threads: ThreadCount::default(),
            writer_threads: ThreadCount::default(),
            id_column: default_id_column(),
            commit_threshold_size_mb: DEFAULT_COMMIT_THRESHOLD_SIZE_MB,
            verify_crc_on_open: DEFAULT_VERIFY_CRC_ON_OPEN,
        }
    }
}

const DEFAULT_COMMIT_THRESHOLD_SIZE_MB: u64 = 1024;
const DEFAULT_VERIFY_CRC_ON_OPEN: bool = true;

// Compaction defaults
const DEFAULT_COMPACTION_TARGET_SUPERFILE_SIZE_MB: u64 = 1024;
const DEFAULT_COMPACTION_MIN_FILL_PERCENT: u8 = 80;
const DEFAULT_COMPACTION_MAX_MEMORY_MB: u64 = DEFAULT_COMPACTION_TARGET_SUPERFILE_SIZE_MB + 2048;

/// How old a tombstone sidecar seal has to be before compaction treats
/// its owner as dead and takes over, instead of backing off.
/// Scale this up if target_superfile_size_mb is raised well past the default
pub const DEFAULT_STALE_SEAL_TIMEOUT_MS: u64 = 2 * 60 * 1000;

/// Compaction settings: target size, fill floor, and memory budget.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct CompactionSettings {
    /// Target size of a compacted superfile, in MiB.
    pub target_superfile_size_mb: u64,
    /// Minimum estimated live bytes to trigger a merge,
    /// as a percentage of `target_superfile_size_mb`.
    pub min_fill_percent: u8,
    /// Maximum memory budget for materializing inputs during a single merge, in MiB.
    pub max_memory_mb: u64,
    /// How old a sealed tombstone sidecar has to be, in milliseconds,
    /// before it's treated as abandoned
    pub stale_seal_timeout_ms: u64,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            target_superfile_size_mb: DEFAULT_COMPACTION_TARGET_SUPERFILE_SIZE_MB,
            min_fill_percent: DEFAULT_COMPACTION_MIN_FILL_PERCENT,
            max_memory_mb: DEFAULT_COMPACTION_MAX_MEMORY_MB,
            stale_seal_timeout_ms: DEFAULT_STALE_SEAL_TIMEOUT_MS,
        }
    }
}

/// Minimum age an unreferenced object must reach before [`crate::Supertable::optimize`] deletes it.
pub const DEFAULT_GC_SAFETY_GAP: Duration = Duration::from_secs(86_400);

// Vector-tuning defaults. Kept equal to the historical inline
// literals so folding these knobs into config preserves behavior.
/// Default overflow threshold before a merged cell superfile is split. STOPGAP:
/// Max docs in a global cell before compaction splits it in two. Set high: a
/// cell this size serves well on its own (the per-cell fine IVF prunes within
/// it), so the grid stays coarse and split-free at <= 10M (cells stay ~156K at
/// 64 cells even at 10M). The split engages only at 100M/1B to bound cell size;
/// lower this if higher-scale recall needs a finer grid.
const DEFAULT_VECTOR_CELL_SPLIT_DOC_CAP: u64 = 500_000;
/// Default k-means training points per centroid for per-cell sub-builds.
const DEFAULT_VECTOR_KMEANS_PTS_PER_CENTROID: usize = 64;
/// Default per-cell fine-probe floor: the minimum fine IVF clusters probed
/// inside each selected cell. Small cells stay at this known-good minimum.
const DEFAULT_VECTOR_FINE_NPROBE_FLOOR: usize = 4;
/// Default proportional fine-probe fraction. `0.0` ⇒ proportional depth off:
/// the probe is the fixed floor. `> 0` probes `floor(pct × cell fine clusters)`
/// so depth tracks cell size (recall lever for large cells).
const DEFAULT_VECTOR_FINE_NPROBE_PCT: f64 = 0.0;
/// Default user superfiles the hidden-index drain materializes per batch.
const DEFAULT_VECTOR_DRAIN_BATCH_SUPERFILES: i64 = 64;
/// Default boundary-replication budget (commit + drain). `<= 1.0` disables
/// replication, which is the default: at 10M it was a measured net loss —
/// the extra boundary copies inflated cell size (159K → 232K rows), crowding
/// the RaBitQ shortlist and displacing true neighbors before rerank (recall
/// 0.997 → 0.975) while adding ~50% storage and ~35% GETs/query. Grid+fine
/// union routing carries boundary coverage instead.
const DEFAULT_VECTOR_DRAIN_REPLICA_TARGET_FACTOR: f32 = 1.0;
/// Default cell count for the **user** table's grid — the grid trained at the
/// first commit, used to cell-pack user superfiles and to route the pre-drain
/// query. Finer cells make the default single-cell pre-drain probe both more
/// precise and cheaper.
const DEFAULT_VECTOR_USER_CELL_COUNT: usize = 256;
/// Default cell count for the **hidden** vector index. The drain trains and
/// reads its grid at this count; post-drain routing runs at this granularity.
/// Equal to the user count by default — one 256-cell grid drives packing,
/// pre-drain routing, the drain, and post-drain routing (`user_grid` is
/// trained only when the counts differ).
const DEFAULT_VECTOR_HIDDEN_CELL_COUNT: usize = 256;
/// Default hidden vector-index compaction target superfile size (MiB). Sized
/// to hold a full packed cell shard plus incremental deltas so the drain's
/// base shard stays a merge candidate and absorbs later deltas rather than
/// being sealed as over-target on the first pass.
const DEFAULT_VECTOR_COMPACTION_TARGET_MB: u64 = 2048;
/// Default hidden vector-index compaction min-fill: `0` disables the size leg,
/// so a cell consolidates on the >= 2 fragment count alone — drain generations
/// collapse and post-compact cold GET stays at the post-drain level.
const DEFAULT_VECTOR_COMPACTION_MIN_FILL_PERCENT: u8 = 0;
/// Default hidden vector-index compaction per-pass memory ceiling (MiB). Must
/// stay >= the target or it caps the packed inputs below a full output.
const DEFAULT_VECTOR_COMPACTION_MAX_MEMORY_MB: u64 = DEFAULT_VECTOR_COMPACTION_TARGET_MB + 2048;

/// How the writer aligns user-superfile vector clusters to the global
/// cell grid. Selected by `vector.user_centroids`.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CentroidAlignment {
    /// Local per-superfile k-means (default). Each superfile trains
    /// its own clusters.
    #[default]
    Local,
    /// Build user superfiles aligned to the global cell grid
    /// (cluster `c` == cell `c`) so the drain routes cluster → cell
    /// doc-correctly without re-scoring.
    Global,
}

/// Per-cell consolidation op the hidden-index drain applies. Selected
/// by `vector.drain_consolidate`.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DrainConsolidate {
    /// Materialize each superfile's rows, assign to the nearest global
    /// cell, and re-cluster per cell (default).
    #[default]
    Kmeans,
    /// Route each superfile's local clusters to their nearest global
    /// cell and keep them verbatim as multi-cluster fragments (no
    /// re-cluster).
    Splice,
}

/// Vector-index build / search / drain tuning knobs. Grouped so the
/// vector-specific levers don't crowd the top-level namespace. All
/// have defaults equal to the engine's built-in behavior; a fresh
/// install never needs to set any of them.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VectorSettings {
    /// Absolute cap on fine IVF centroids probed per vector search.
    /// `None` (the default) derives the budget from `nprobe` and the
    /// number of eligible superfiles at query time; `Some(n)` forces
    /// exactly `n`.
    pub inner_budget: Option<usize>,
    /// Per-cell fine-probe floor — the minimum number of fine IVF
    /// clusters probed inside each selected cell. The user-table routing
    /// default takes this value; the hidden index's per-table stamp in
    /// the manifest overrides it. Pairs with [`Self::fine_nprobe_pct`]
    /// as `max(floor, floor(pct × clusters))`.
    pub fine_nprobe_floor: usize,
    /// Proportional fine-probe fraction for UNFILTERED vector search:
    /// probe `floor(pct × the cell's fine-cluster count)` so depth scales
    /// with cell size. `0.0` (the default) turns the proportional depth
    /// off, leaving the fixed floor. Filtered queries always ignore it.
    pub fine_nprobe_pct: f64,
    /// Default rerank codec for cosine vector columns. Non-cosine
    /// metrics still use locally fitted [`RerankCodec::Sq8Residual`]
    /// at column construction time. Per-column overrides at table
    /// create win over this default.
    pub rerank_codec: RerankCodec,
    /// K-means training points per centroid for the drain's per-cell
    /// sub-builds. Higher trains on more points (slower, tighter
    /// clusters).
    pub kmeans_pts_per_centroid: usize,
    /// Doc count above which a merged cell superfile is split into two
    /// sub-cells during hidden-index maintenance.
    pub cell_split_doc_cap: u64,
    /// How user-superfile clusters align to the global cell grid.
    pub user_centroids: CentroidAlignment,
    /// User superfiles the hidden-index drain materializes per batch
    /// before publishing that batch's cell superfiles. Bounds drain
    /// RAM to O(batch). `-1` = unbounded (one merge, O(corpus) RAM);
    /// `0` = skip the drain entirely.
    pub drain_batch_superfiles: i64,
    /// Target storage amplification for boundary-only drain
    /// replication. `1.2` lets the drain add at most `0.2 × rows`
    /// extra copies of rows near a Voronoi boundary; `<= 1.0` disables
    /// replication.
    pub drain_replica_target_factor: f32,
    /// Per-cell consolidation op the drain applies.
    pub drain_consolidate: DrainConsolidate,
    /// Read fan-out for the drain's superfile opens. `auto` resolves
    /// to one in-flight read per hardware thread, floored at the
    /// background-fill default and capped at 64.
    pub drain_read_concurrency: ThreadCount,
    /// Cell count for the **user** table's grid, trained at the first commit —
    /// controls user-superfile cell packing and pre-drain query routing.
    /// Stamped into the manifest at create; changing it later affects new
    /// tables only.
    pub user_cell_count: usize,
    /// Cell count for the **hidden** vector index grid, trained at the same
    /// first commit. Independent of `user_cell_count` so the pre-drain and
    /// post-drain grids can be tuned separately; the drain reads this grid
    /// verbatim.
    pub hidden_cell_count: usize,
    /// Hidden vector-index compaction target superfile size (MiB). Distinct
    /// from the user table's `compaction.target_superfile_size_mb`; a
    /// packed cell shard stays a merge candidate until it reaches this.
    pub compaction_target_mb: u64,
    /// Hidden vector-index compaction min-fill: a merge fires only once its
    /// combined inputs reach this percentage of `compaction_target_mb`.
    pub compaction_min_fill_percent: u8,
    /// Hidden vector-index compaction per-pass memory ceiling (MiB). Caps
    /// the input bytes packed into one merge, so it must stay >=
    /// `compaction_target_mb` or the target is never reached.
    pub compaction_max_memory_mb: u64,
}

impl Default for VectorSettings {
    fn default() -> Self {
        Self {
            inner_budget: None,
            fine_nprobe_floor: DEFAULT_VECTOR_FINE_NPROBE_FLOOR,
            fine_nprobe_pct: DEFAULT_VECTOR_FINE_NPROBE_PCT,
            rerank_codec: RerankCodec::default(),
            kmeans_pts_per_centroid: DEFAULT_VECTOR_KMEANS_PTS_PER_CENTROID,
            cell_split_doc_cap: DEFAULT_VECTOR_CELL_SPLIT_DOC_CAP,
            user_centroids: CentroidAlignment::Local,
            drain_batch_superfiles: DEFAULT_VECTOR_DRAIN_BATCH_SUPERFILES,
            drain_replica_target_factor: DEFAULT_VECTOR_DRAIN_REPLICA_TARGET_FACTOR,
            drain_consolidate: DrainConsolidate::Kmeans,
            drain_read_concurrency: ThreadCount::Auto,
            user_cell_count: DEFAULT_VECTOR_USER_CELL_COUNT,
            hidden_cell_count: DEFAULT_VECTOR_HIDDEN_CELL_COUNT,
            compaction_target_mb: DEFAULT_VECTOR_COMPACTION_TARGET_MB,
            compaction_min_fill_percent: DEFAULT_VECTOR_COMPACTION_MIN_FILL_PERCENT,
            compaction_max_memory_mb: DEFAULT_VECTOR_COMPACTION_MAX_MEMORY_MB,
        }
    }
}

/// Diagnostic and hardware-capability toggles. Each gates
/// instrumentation or forces a slower path for A/B measurement; none
/// change query results. Default: all `false`.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(default)]
pub struct DiagnosticsSettings {
    /// Accumulate per-phase timers during the vector drain build.
    pub drain_build_timers: bool,
    /// Emit the FTS builder's finish-phase profile.
    pub fts_profile: bool,
    /// Capture the object-store I/O timeline.
    pub io_timeline: bool,
    /// Force the AVX2 vector-distance path even where AVX-512 is
    /// available (A/B measurement).
    pub disable_avx512: bool,
    /// Force the scalar vector-distance path even where AVX2 is
    /// available (A/B measurement).
    pub disable_avx2: bool,
    /// Skip the disk cache's lazy background fill so foreground-only
    /// read behavior can be measured (A/B measurement).
    pub disable_background_fill: bool,
}

/// Gc settings used by `optimize()`'s bundled gc sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcSettings {
    /// Minimum age an unreferenced object must reach before it's deleted.
    pub safety_gap: Duration,
}

impl Default for GcSettings {
    fn default() -> Self {
        Self {
            safety_gap: DEFAULT_GC_SAFETY_GAP,
        }
    }
}

impl GcSettings {
    /// Gc settings with the given safety gap;
    pub fn with_safety_gap(mut self, gap: Duration) -> Self {
        self.safety_gap = gap;
        self
    }
}

/// Options for [`crate::Supertable::optimize`].
///
/// Additional operation kinds (e.g. vector-index maintenance) will be
/// added here without breaking this type.
#[derive(Debug, Clone, Default)]
pub struct OptimizeOptions {
    pub(crate) compaction: CompactionSettings,
    pub(crate) gc: GcSettings,
}

impl OptimizeOptions {
    /// Options for a compaction-only optimize with the given settings.
    pub fn compact(settings: CompactionSettings) -> Self {
        Self {
            compaction: settings,
            gc: GcSettings::default(),
        }
    }

    /// Override the gc settings `optimize()`'s bundled sweep uses.
    pub fn with_gc(mut self, gc: GcSettings) -> Self {
        self.gc = gc;
        self
    }
}

/// Persistent storage backend selected by [`StorageSettings`].
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum StorageBackend {
    /// In-memory-only supertable; no durable storage is
    /// attached by config.
    #[default]
    None,
    /// Local filesystem provider rooted at
    /// [`StorageSettings::local_root`].
    LocalFs,
    /// AWS S3 provider rooted at
    /// `s3://storage.bucket/storage.prefix`.
    S3,
    /// Azure Blob provider; `storage.bucket` names the container,
    /// rooted at `azure://storage.bucket/storage.prefix`.
    Azure,
    /// GCS provider rooted at `gs://storage.bucket/storage.prefix`.
    Gcs,
}

/// Config-side spelling for disk-cache cold-fetch mode. Kept
/// separate from the runtime enum so serde naming stays stable
/// without coupling config format to internal module layout.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum StorageColdFetchMode {
    /// Parallel range GETs serve both the foreground reader and the
    /// disk-cache fill. Foreground returns after the range fetches;
    /// pwrite, mmap, and cache registration finish in the background.
    /// Uses one copy of superfile bandwidth per cold miss.
    HybridWithPrefetch,
    /// Single-range sequential fetches (no background fill). Useful
    /// for constrained environments where parallelism is undesirable.
    RangeOnly,
    /// Foreground returns a lazy reader and a background task fills
    /// the disk cache asynchronously. With manifest open-batch bytes
    /// present, open issues zero superfile-object GETs; otherwise it
    /// fetches the parquet tail plus vector/FTS open ranges. First
    /// query pays per-cluster range GETs; subsequent queries resolve
    /// from mmap once the fill completes.
    #[default]
    LazyForegroundWithBackgroundFill,
}

/// Storage + disk-cache settings applied by
/// [`crate::supertable::SupertableOptions::apply_config`].
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct StorageSettings {
    /// Which backend to attach. `none` preserves the old
    /// in-memory-only behavior.
    pub backend: StorageBackend,
    /// Local filesystem root when `backend: local_fs`.
    pub local_root: Option<PathBuf>,
    /// Object-store bucket name (used by the `s3` backend).
    pub bucket: Option<String>,
    /// Credentials/tuning for the backend, keyed by `object_store`
    /// config strings (`aws_*` / `azure_*`). Empty → ambient identity.
    pub storage_options: HashMap<String, String>,
    /// Logical key prefix inside the bucket. All manifest and
    /// superfile objects are written under
    /// `<bucket>/<prefix>/<manifest|superfiles>/…`. Empty means the
    /// bucket root. Not used by the `local_fs` backend (use
    /// `local_root` instead).
    pub prefix: String,
    /// Disk-cache root. When set with any persistent backend,
    /// `apply_config` attaches a `DiskCacheStore` so reads go
    /// through the object-store lazy/cached path.
    pub disk_cache_root: Option<PathBuf>,
    pub disk_budget_bytes: u64,
    /// Byte budget for the content-addressed manifest-part cache, kept
    /// in a `manifest-parts/` subdirectory of `disk_cache_root`. The
    /// loader reads part bytes from local disk on a hit instead of
    /// fetching from object storage. Independent of `disk_budget_bytes`
    /// (which sizes the superfile-content cache). Default 2 GiB.
    pub manifest_disk_budget_bytes: u64,
    pub cold_fetch_mode: StorageColdFetchMode,
    pub cold_fetch_streams: usize,
    pub cold_fetch_chunk_bytes: u64,
    /// Global cap on concurrent background superfile fills. See
    /// [`crate::supertable::reader_cache::DiskCacheConfig::prefetch_concurrency`].
    pub prefetch_concurrency: usize,
    /// Minimum age (seconds) before an mmap'd superfile is
    /// considered cold and eligible for eviction by the sweep.
    /// Default: 300 s (5 min). Prevents thrashing on superfiles
    /// that just finished their background fill.
    pub mmap_cold_threshold_secs: u64,
    /// Interval (seconds) between mmap eviction sweeps. The sweep
    /// drops pages for superfiles older than
    /// `mmap_cold_threshold_secs` and not accessed since the
    /// previous sweep. Default: 75 s.
    pub mmap_sweep_interval_secs: u64,
}

impl Default for StorageSettings {
    fn default() -> Self {
        Self {
            backend: StorageBackend::None,
            local_root: None,
            bucket: None,
            storage_options: HashMap::new(),
            prefix: String::new(),
            disk_cache_root: None,
            disk_budget_bytes: DEFAULT_DISK_BUDGET_BYTES,
            manifest_disk_budget_bytes: DEFAULT_MANIFEST_DISK_BUDGET_BYTES,
            cold_fetch_mode: StorageColdFetchMode::LazyForegroundWithBackgroundFill,
            cold_fetch_streams: DEFAULT_COLD_FETCH_STREAMS,
            cold_fetch_chunk_bytes: DEFAULT_COLD_FETCH_CHUNK_BYTES,
            prefetch_concurrency: DEFAULT_PREFETCH_CONCURRENCY,
            mmap_cold_threshold_secs: DEFAULT_MMAP_COLD_THRESHOLD_SECS,
            mmap_sweep_interval_secs: DEFAULT_MMAP_SWEEP_INTERVAL_SECS,
        }
    }
}

/// Default disk-cache byte budget exposed in the shipped config (10 GiB).
const DEFAULT_DISK_BUDGET_BYTES: u64 = 10 * (1 << 30);
/// Default manifest-part cache byte budget (2 GiB). Parts are small
/// (KB–few MB each), so this holds a large working set of parts.
const DEFAULT_MANIFEST_DISK_BUDGET_BYTES: u64 = 2 * (1 << 30);
/// Default parallel cold-fetch streams at the config layer.
const DEFAULT_COLD_FETCH_STREAMS: usize = 8;
/// Default cold-fetch range chunk size (4 MiB).
const DEFAULT_COLD_FETCH_CHUNK_BYTES: u64 = 4 * (1 << 20);
/// Default concurrent background full-superfile fills.
pub(crate) const DEFAULT_PREFETCH_CONCURRENCY: usize = 8;
/// Default idle age (seconds) before an mmap is swept.
const DEFAULT_MMAP_COLD_THRESHOLD_SECS: u64 = 300;
/// Default background mmap-sweep period (seconds).
const DEFAULT_MMAP_SWEEP_INTERVAL_SECS: u64 = 75;

fn default_id_column() -> String {
    "_id".to_string()
}

/// Thread count specifier — either `auto` (defer to a runtime
/// default) or an explicit positive integer.
///
/// In YAML / env, the value can be the string `"auto"` (case-
/// insensitive) or a positive integer. The serialized form is
/// `"auto"` for [`ThreadCount::Auto`] and the integer otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThreadCount {
    /// Resolve at runtime to a hardware-aware default supplied by
    /// the consumer (typically a function of `num_cpus`).
    #[default]
    Auto,
    /// Use exactly this many threads. Clamped to `≥ 1` at
    /// resolution time.
    Fixed(usize),
}

impl ThreadCount {
    /// Resolve to a concrete thread count. `Auto` falls back to
    /// `default_for_auto`; both branches clamp the result to
    /// `≥ 1` so we never construct a zero-thread rayon pool.
    pub fn resolve_or_default(self, default_for_auto: usize) -> usize {
        match self {
            Self::Auto => default_for_auto.max(1),
            Self::Fixed(n) => n.max(1),
        }
    }
}

impl<'de> Deserialize<'de> for ThreadCount {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ThreadCount;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("\"auto\" or a positive integer")
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                if v.eq_ignore_ascii_case("auto") {
                    Ok(ThreadCount::Auto)
                } else {
                    v.parse::<usize>().map(ThreadCount::Fixed).map_err(|e| {
                        de::Error::custom(format!(
                            "thread count must be \"auto\" or a positive integer; \
                                 got {v:?} ({e})"
                        ))
                    })
                }
            }
            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
                self.visit_str(&v)
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(ThreadCount::Fixed(v as usize))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                if v < 0 {
                    Err(de::Error::custom("thread count must be ≥ 0"))
                } else {
                    Ok(ThreadCount::Fixed(v as usize))
                }
            }
        }
        d.deserialize_any(V)
    }
}

impl Serialize for ThreadCount {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Auto => s.serialize_str("auto"),
            Self::Fixed(n) => s.serialize_u64(*n as u64),
        }
    }
}

impl Config {
    /// Load from the standard hierarchy. See module docs for the
    /// precedence order.
    pub fn load() -> Result<Self, ConfigError> {
        Self::from_figment(default_figment())
    }

    /// Load from only the embedded defaults — no file or env
    /// overrides. Useful for tests and for documenting what the
    /// shipped default is independent of any host environment.
    pub fn defaults() -> Result<Self, ConfigError> {
        Ok(Figment::new()
            .merge(Yaml::string(EMBEDDED_DEFAULT))
            .extract()?)
    }

    /// Extract from a caller-provided figment. Used by tests so they
    /// don't have to touch the real filesystem or env. Public so
    /// downstream crates can build their own layered config (e.g. a
    /// CLI that adds a `--config-file` source) without duplicating
    /// the embedded-default + extraction machinery.
    pub fn from_figment(fig: Figment) -> Result<Self, ConfigError> {
        Ok(fig.extract()?)
    }
}

/// Build the standard layered figment used by [`Config::load`].
/// YAML files only — process env never participates.
fn default_figment() -> Figment {
    let mut fig = Figment::new().merge(Yaml::string(EMBEDDED_DEFAULT));

    let etc = Path::new("/etc/infino/config.yaml");
    if etc.is_file() {
        fig = fig.merge(Yaml::file(etc));
    }

    if let Some(p) = user_config_path()
        && p.is_file()
    {
        fig = fig.merge(Yaml::file(p));
    }

    let cwd = Path::new("./infino.yaml");
    if cwd.is_file() {
        fig = fig.merge(Yaml::file(cwd));
    }

    fig
}

/// Resolve the user-level config path. Honors `XDG_CONFIG_HOME`
/// first; falls back to `$HOME/.config/infino/config.yaml`.
fn user_config_path() -> Option<PathBuf> {
    if let Ok(xdg) = env::var("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(xdg).join("infino/config.yaml"));
    }
    env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".config/infino/config.yaml"))
}

#[cfg(test)]
mod tests {
    use std::{env, sync::Mutex};

    use figment::providers::Serialized;
    use serde_json::json;

    use super::*;

    /// Serialize tests that mutate process-global env so they don't
    /// race. `unsafe { std::env::set_var }` requires this in the 2024
    /// edition.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn embedded_default_loads_with_expected_value() {
        let cfg = Config::defaults().expect("embedded default must parse");
        assert_eq!(cfg.supertable.commit_threshold_size_mb, 1024);
    }

    #[test]
    fn env_vars_do_not_override_config() {
        let _g = ENV_LOCK.lock().expect("acquire lock");
        // SAFETY: serialized via ENV_LOCK; cleanup at end.
        unsafe {
            env::set_var("INFINO_SUPERTABLE__COMMIT_THRESHOLD_SIZE_MB", "2048");
            env::set_var("INFINO_VECTOR__DRAIN_REPLICA_TARGET_FACTOR", "9.9");
            env::set_var("INFINO_DIAGNOSTICS__IO_TIMELINE", "true");
        }
        let cfg = Config::load().expect("load ignoring env");
        assert_eq!(
            cfg.supertable.commit_threshold_size_mb, 1024,
            "engine config must come from YAML only"
        );
        assert_eq!(cfg.vector.drain_replica_target_factor, 1.0);
        assert!(!cfg.diagnostics.io_timeline);
        unsafe {
            env::remove_var("INFINO_SUPERTABLE__COMMIT_THRESHOLD_SIZE_MB");
            env::remove_var("INFINO_VECTOR__DRAIN_REPLICA_TARGET_FACTOR");
            env::remove_var("INFINO_DIAGNOSTICS__IO_TIMELINE");
        }
    }

    #[test]
    fn from_figment_with_yaml_layer_overrides_default() {
        let yaml = r#"
supertable:
  commit_threshold_size_mb: 512
"#;
        let fig = Figment::new()
            .merge(Yaml::string(EMBEDDED_DEFAULT))
            .merge(Yaml::string(yaml));
        let cfg = Config::from_figment(fig).expect("layered yaml");
        assert_eq!(cfg.supertable.commit_threshold_size_mb, 512);
    }

    #[test]
    fn embedded_default_storage_is_in_memory_only() {
        let cfg = Config::defaults().expect("embedded default must parse");
        assert_eq!(cfg.storage.backend, StorageBackend::None);
        assert_eq!(cfg.storage.bucket, None);
        assert_eq!(cfg.storage.disk_cache_root, None);
    }

    #[test]
    fn storage_s3_config_parses_bucket_prefix_and_cache() {
        let yaml = r#"
storage:
  backend: s3
  bucket: example-bucket
  prefix: infino-real-s3-integration/example
  disk_cache_root: /tmp/infino-cache
  cold_fetch_mode: lazy_foreground_with_background_fill
  cold_fetch_streams: 8
  cold_fetch_chunk_bytes: 4194304
"#;
        let fig = Figment::new()
            .merge(Yaml::string(EMBEDDED_DEFAULT))
            .merge(Yaml::string(yaml));
        let cfg = Config::from_figment(fig).expect("parse config");
        assert_eq!(cfg.storage.backend, StorageBackend::S3);
        assert_eq!(cfg.storage.bucket.as_deref(), Some("example-bucket"));
        assert_eq!(cfg.storage.prefix, "infino-real-s3-integration/example");
        assert_eq!(
            cfg.storage.disk_cache_root.as_deref(),
            Some(Path::new("/tmp/infino-cache"))
        );
        assert_eq!(
            cfg.storage.cold_fetch_mode,
            StorageColdFetchMode::LazyForegroundWithBackgroundFill
        );
    }

    #[test]
    fn storage_azure_config_parses_container_as_bucket() {
        let yaml = r#"
storage:
  backend: azure
  bucket: infino-azure-container
  prefix: infino-real-azure-integration/example
"#;
        let fig = Figment::new()
            .merge(Yaml::string(EMBEDDED_DEFAULT))
            .merge(Yaml::string(yaml));
        let cfg = Config::from_figment(fig).expect("parse config");
        assert_eq!(cfg.storage.backend, StorageBackend::Azure);
        assert_eq!(
            cfg.storage.bucket.as_deref(),
            Some("infino-azure-container")
        );
        assert_eq!(cfg.storage.prefix, "infino-real-azure-integration/example");
    }

    #[test]
    fn storage_gcs_config_parses_bucket() {
        let yaml = r#"
storage:
  backend: gcs
  bucket: infino-gcs-bucket
  prefix: tbl
"#;
        let fig = Figment::new()
            .merge(Yaml::string(EMBEDDED_DEFAULT))
            .merge(Yaml::string(yaml));
        let cfg = Config::from_figment(fig).expect("parse config");
        assert_eq!(cfg.storage.backend, StorageBackend::Gcs);
        assert_eq!(cfg.storage.bucket.as_deref(), Some("infino-gcs-bucket"));
        assert_eq!(cfg.storage.prefix, "tbl");
    }

    #[test]
    fn last_yaml_wins_among_layers() {
        // Layer order: A (default 1024) → B (set 256) → C (set 4096).
        // Final value is 4096; the middle layer is shadowed.
        let fig = Figment::new()
            .merge(Yaml::string(EMBEDDED_DEFAULT))
            .merge(Yaml::string(
                "supertable:\n  commit_threshold_size_mb: 256\n",
            ))
            .merge(Yaml::string(
                "supertable:\n  commit_threshold_size_mb: 4096\n",
            ));
        let cfg = Config::from_figment(fig).expect("parse config");
        assert_eq!(cfg.supertable.commit_threshold_size_mb, 4096);
    }

    #[test]
    fn invalid_value_type_errors_clearly() {
        // String where number expected → figment surfaces a typed
        // deserialization error.
        let fig = Figment::new()
            .merge(Yaml::string(EMBEDDED_DEFAULT))
            .merge(Yaml::string(
                "supertable:\n  commit_threshold_size_mb: \"not-a-number\"\n",
            ));
        let err = Config::from_figment(fig).expect_err("expected error");
        let msg = err.to_string();
        assert!(
            msg.contains("commit_threshold_size_mb")
                || msg.contains("invalid type")
                || msg.contains("expected"),
            "expected a typed-error message; got {msg:?}"
        );
    }

    #[test]
    fn programmatic_override_via_serialized_provider() {
        // Demonstrates that downstream callers can layer a Rust
        // struct override on top of the file/env stack. Used in tests
        // and proves Serialized as a valid override surface.
        #[derive(Serialize)]
        struct SupertableOverride {
            commit_threshold_size_mb: u64,
        }
        #[derive(Serialize)]
        struct Override {
            supertable: SupertableOverride,
        }
        let fig = Figment::new()
            .merge(Yaml::string(EMBEDDED_DEFAULT))
            .merge(Serialized::defaults(Override {
                supertable: SupertableOverride {
                    commit_threshold_size_mb: 16,
                },
            }));
        let cfg = Config::from_figment(fig).expect("parse config");
        assert_eq!(cfg.supertable.commit_threshold_size_mb, 16);
    }

    #[test]
    fn user_config_path_uses_xdg_when_set() {
        let _g = ENV_LOCK.lock().expect("acquire lock");
        // SAFETY: serialized via ENV_LOCK.
        unsafe { env::set_var("XDG_CONFIG_HOME", "/tmp/xdg-test") };
        let p = user_config_path().expect("path");
        assert_eq!(p, PathBuf::from("/tmp/xdg-test/infino/config.yaml"));
        unsafe { env::remove_var("XDG_CONFIG_HOME") };
    }

    #[test]
    fn supertable_defaults_are_auto() {
        let cfg = Config::defaults().expect("embedded default must parse");
        assert_eq!(cfg.supertable.reader_threads, ThreadCount::Auto);
        assert_eq!(cfg.supertable.writer_threads, ThreadCount::Auto);
    }

    #[test]
    fn memory_budget_defaults_to_measure_only() {
        let cfg = Config::defaults().expect("embedded default must parse");
        assert_eq!(cfg.memory.connection_budget_bytes, 0);
    }

    #[test]
    fn thread_count_parses_auto_string() {
        let yaml = r#"
commit_threshold_size_mb: 1024
supertable:
  reader_threads: auto
  writer_threads: AUTO
"#;
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(yaml))).expect("parse config");
        assert_eq!(cfg.supertable.reader_threads, ThreadCount::Auto);
        assert_eq!(cfg.supertable.writer_threads, ThreadCount::Auto);
    }

    #[test]
    fn thread_count_parses_integer() {
        let yaml = r#"
commit_threshold_size_mb: 1024
supertable:
  reader_threads: 8
  writer_threads: 4
"#;
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(yaml))).expect("parse config");
        assert_eq!(cfg.supertable.reader_threads, ThreadCount::Fixed(8));
        assert_eq!(cfg.supertable.writer_threads, ThreadCount::Fixed(4));
    }

    #[test]
    fn thread_count_rejects_garbage_string() {
        let yaml = r#"
commit_threshold_size_mb: 1024
supertable:
  reader_threads: banana
"#;
        let err = Config::from_figment(Figment::new().merge(Yaml::string(yaml)))
            .expect_err("expected error");
        let msg = err.to_string();
        assert!(
            msg.contains("auto") || msg.contains("positive integer") || msg.contains("banana"),
            "expected a typed-error message; got {msg:?}"
        );
    }

    #[test]
    fn thread_count_resolve_clamps_to_one() {
        assert_eq!(ThreadCount::Auto.resolve_or_default(0), 1);
        assert_eq!(ThreadCount::Fixed(0).resolve_or_default(8), 1);
        assert_eq!(ThreadCount::Auto.resolve_or_default(7), 7);
        assert_eq!(ThreadCount::Fixed(3).resolve_or_default(8), 3);
    }

    #[test]
    fn thread_count_yaml_layer_overrides_default() {
        let yaml = r#"
supertable:
  writer_threads: 4
  reader_threads: auto
"#;
        let fig = Figment::new()
            .merge(Yaml::string(EMBEDDED_DEFAULT))
            .merge(Yaml::string(yaml));
        let cfg = Config::from_figment(fig).expect("layered yaml");
        assert_eq!(cfg.supertable.writer_threads, ThreadCount::Fixed(4));
        assert_eq!(cfg.supertable.reader_threads, ThreadCount::Auto);
    }

    #[test]
    fn user_config_path_falls_back_to_home() {
        let _g = ENV_LOCK.lock().expect("acquire lock");
        // SAFETY: serialized via ENV_LOCK.
        unsafe {
            env::remove_var("XDG_CONFIG_HOME");
            env::set_var("HOME", "/tmp/home-test");
        }
        let p = user_config_path().expect("path");
        assert_eq!(
            p,
            PathBuf::from("/tmp/home-test/.config/infino/config.yaml")
        );
        unsafe { env::remove_var("HOME") };
    }

    #[test]
    fn embedded_default_compaction_matches_spec() {
        let cfg = Config::defaults().expect("embedded default must parse");
        let c = &cfg.compaction;
        assert_eq!(
            c.target_superfile_size_mb,
            DEFAULT_COMPACTION_TARGET_SUPERFILE_SIZE_MB
        );
        assert_eq!(c.min_fill_percent, DEFAULT_COMPACTION_MIN_FILL_PERCENT);
        assert_eq!(
            c.max_memory_mb, DEFAULT_COMPACTION_MAX_MEMORY_MB,
            "target + 2048"
        );
    }

    #[test]
    fn compaction_struct_default_equals_embedded_yaml() {
        // The Rust `Default` and the shipped YAML must not drift.
        let cfg = Config::defaults().expect("embedded default must parse");
        assert_eq!(cfg.compaction, CompactionSettings::default());
    }

    #[test]
    fn compaction_yaml_layer_overrides_defaults() {
        let yaml = r#"
               compaction:
                    target_superfile_size_mb: 2048
                    min_fill_percent: 50
           "#;
        let fig = Figment::new()
            .merge(Yaml::string(EMBEDDED_DEFAULT))
            .merge(Yaml::string(yaml));
        let cfg = Config::from_figment(fig).expect("layered yaml");
        assert_eq!(cfg.compaction.target_superfile_size_mb, 2048);
        assert_eq!(cfg.compaction.min_fill_percent, 50);
        assert_eq!(cfg.compaction.max_memory_mb, 3072);
    }

    #[test]
    fn compaction_invalid_value_type_errors_clearly() {
        let fig = Figment::new()
            .merge(Yaml::string(EMBEDDED_DEFAULT))
            .merge(Yaml::string(
                "compaction:\n  target_superfile_size_mb: \"not-a-number\"\n",
            ));
        let err = Config::from_figment(fig).expect_err("expected error");
        let msg = err.to_string();
        assert!(
            msg.contains("target_superfile_size_mb")
                || msg.contains("invalid type")
                || msg.contains("expected"),
            "expected a typed-error message; got {msg:?}"
        );
    }

    #[test]
    fn compaction_min_fill_percent_rejects_out_of_u8_range() {
        // 256 overflows u8 — figment surfaces a typed error rather
        // than silently truncating.
        let fig = Figment::new()
            .merge(Yaml::string(EMBEDDED_DEFAULT))
            .merge(Yaml::string("compaction:\n  min_fill_percent: 256\n"));
        let err = Config::from_figment(fig).expect_err("expected error");
        let msg = err.to_string();
        assert!(
            msg.contains("min_fill_percent")
                || msg.contains("256")
                || msg.contains("u8")
                || msg.contains("out of range")
                || msg.contains("invalid value"),
            "expected an out-of-range message; got {msg:?}"
        );
    }

    /// `ThreadCount` serializes back to its config spelling (`"auto"` /
    /// an integer), deserializes from an owned-string value, and
    /// rejects a negative integer and a wrong-typed value (the latter
    /// surfacing the visitor's `expecting` message).
    #[test]
    fn thread_count_serde_round_trips_and_rejects_bad_types() {
        // Serialize both variants.
        assert_eq!(
            serde_json::to_value(ThreadCount::Auto).expect("serialize auto"),
            json!("auto")
        );
        assert_eq!(
            serde_json::to_value(ThreadCount::Fixed(8)).expect("serialize fixed"),
            json!(8)
        );

        // Deserialize from an owned-string `Value` exercises the
        // `visit_string` arm (vs `visit_str` for borrowed input).
        let tc: ThreadCount =
            serde_json::from_value(json!("auto")).expect("deserialize owned string");
        assert!(matches!(tc, ThreadCount::Auto));

        // A negative integer is rejected by the signed-int visitor.
        assert!(serde_json::from_str::<ThreadCount>("-1").is_err());

        // A wrong-typed value (bool) fails through the default visitor,
        // which formats the `expecting` description.
        assert!(serde_json::from_str::<ThreadCount>("true").is_err());
    }

    #[test]
    fn embedded_default_vector_equals_struct_default() {
        // The shipped YAML and the Rust `Default` must not drift.
        let cfg = Config::defaults().expect("embedded default must parse");
        assert_eq!(cfg.vector, VectorSettings::default());
        assert_eq!(cfg.vector.inner_budget, None);
        assert_eq!(cfg.vector.cell_split_doc_cap, 500_000);
        assert_eq!(cfg.vector.user_centroids, CentroidAlignment::Local);
        assert_eq!(cfg.vector.drain_consolidate, DrainConsolidate::Kmeans);
        assert_eq!(cfg.vector.rerank_codec, RerankCodec::Sq8FixedResidual);
        assert_eq!(cfg.vector.drain_read_concurrency, ThreadCount::Auto);
    }

    #[test]
    fn embedded_default_diagnostics_all_off() {
        let cfg = Config::defaults().expect("embedded default must parse");
        assert_eq!(cfg.diagnostics, DiagnosticsSettings::default());
        assert!(!cfg.diagnostics.io_timeline);
        assert!(!cfg.diagnostics.disable_avx512);
    }

    #[test]
    fn vector_yaml_layer_overrides_defaults() {
        let yaml = r#"
vector:
  inner_budget: 4096
  cell_split_doc_cap: 100000
  user_centroids: global
  drain_consolidate: splice
  rerank_codec: sq8_residual
  drain_replica_target_factor: 1.25
  drain_read_concurrency: 12
"#;
        let fig = Figment::new()
            .merge(Yaml::string(EMBEDDED_DEFAULT))
            .merge(Yaml::string(yaml));
        let cfg = Config::from_figment(fig).expect("layered yaml");
        assert_eq!(cfg.vector.inner_budget, Some(4096));
        assert_eq!(cfg.vector.cell_split_doc_cap, 100_000);
        assert_eq!(cfg.vector.user_centroids, CentroidAlignment::Global);
        assert_eq!(cfg.vector.drain_consolidate, DrainConsolidate::Splice);
        assert_eq!(cfg.vector.rerank_codec, RerankCodec::Sq8Residual);
        assert_eq!(cfg.vector.drain_replica_target_factor, 1.25);
        assert_eq!(cfg.vector.drain_read_concurrency, ThreadCount::Fixed(12));
        // Untouched keys fall through to the embedded default.
        assert_eq!(cfg.vector.drain_batch_superfiles, 64);
    }
}
