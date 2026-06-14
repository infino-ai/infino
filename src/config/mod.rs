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
//!   5. **Environment variables** prefixed `INFINO_`. Field names
//!      are uppercased and nested keys join with `__`;
//!      e.g. `supertable.commit_threshold_size_mb` is set by
//!      `INFINO_SUPERTABLE__COMMIT_THRESHOLD_SIZE_MB`.
//!
//! Each layer is a partial override — keys absent from a higher
//! layer fall through to lower layers. Unknown keys at any layer
//! are accepted (figment's default leniency); typos in env vars
//! therefore silently no-op. We document the published variables
//! here and rely on tests + code review to keep them in sync.
//!
//! ## Adding a new field
//!
//! 1. Add the field to [`Config`] with a `serde` rename / default
//!    if appropriate.
//! 2. Add the same key to `config.yaml` with its default value.
//! 3. Add a docstring and a unit test exercising the override path.

use figment::Figment;
use figment::providers::{Env, Format, Yaml};
use serde::de::{self, Deserializer, Visitor};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

/// Embedded baseline. Compiled in via `include_str!`.
const EMBEDDED_DEFAULT: &str = include_str!("config.yaml");

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
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
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
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            target_superfile_size_mb: DEFAULT_COMPACTION_TARGET_SUPERFILE_SIZE_MB,
            min_fill_percent: DEFAULT_COMPACTION_MIN_FILL_PERCENT,
            max_memory_mb: DEFAULT_COMPACTION_MAX_MEMORY_MB,
        }
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
            prefix: String::new(),
            disk_cache_root: None,
            disk_budget_bytes: DEFAULT_DISK_BUDGET_BYTES,
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
/// Default parallel cold-fetch streams at the config layer.
const DEFAULT_COLD_FETCH_STREAMS: usize = 8;
/// Default cold-fetch range chunk size (4 MiB).
const DEFAULT_COLD_FETCH_CHUNK_BYTES: u64 = 4 * (1 << 20);
/// Default concurrent background full-superfile fills.
const DEFAULT_PREFETCH_CONCURRENCY: usize = 8;
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

    // `split("__")` lets nested fields be addressed in env, e.g.
    // `INFINO_SUPERTABLE__READER_THREADS=8` maps to
    // `supertable.reader_threads`. Single-underscore field names
    // are unaffected.
    fig.merge(Env::prefixed("INFINO_").split("__"))
}

/// Resolve the user-level config path. Honors `XDG_CONFIG_HOME`
/// first; falls back to `$HOME/.config/infino/config.yaml`.
fn user_config_path() -> Option<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(xdg).join("infino/config.yaml"));
    }
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".config/infino/config.yaml"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use figment::providers::Serialized;
    use std::sync::Mutex;

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
    fn env_overrides_default() {
        let _g = ENV_LOCK.lock().expect("acquire lock");
        // SAFETY: serialized via ENV_LOCK; cleanup at end.
        unsafe { std::env::set_var("INFINO_SUPERTABLE__COMMIT_THRESHOLD_SIZE_MB", "2048") };
        let cfg = Config::load().expect("load with env override");
        assert_eq!(cfg.supertable.commit_threshold_size_mb, 2048);
        unsafe { std::env::remove_var("INFINO_SUPERTABLE__COMMIT_THRESHOLD_SIZE_MB") };
    }

    #[test]
    fn missing_env_falls_through_to_default() {
        let _g = ENV_LOCK.lock().expect("acquire lock");
        // SAFETY: serialized via ENV_LOCK; we ensure the var is unset
        // before reading.
        unsafe { std::env::remove_var("INFINO_SUPERTABLE__COMMIT_THRESHOLD_SIZE_MB") };
        let cfg = Config::load().expect("load with no env override");
        assert_eq!(cfg.supertable.commit_threshold_size_mb, 1024);
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
        unsafe { std::env::set_var("XDG_CONFIG_HOME", "/tmp/xdg-test") };
        let p = user_config_path().expect("path");
        assert_eq!(p, PathBuf::from("/tmp/xdg-test/infino/config.yaml"));
        unsafe { std::env::remove_var("XDG_CONFIG_HOME") };
    }

    #[test]
    fn supertable_defaults_are_auto() {
        let cfg = Config::defaults().expect("embedded default must parse");
        assert_eq!(cfg.supertable.reader_threads, ThreadCount::Auto);
        assert_eq!(cfg.supertable.writer_threads, ThreadCount::Auto);
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
    fn nested_env_var_overrides_supertable_field() {
        let _g = ENV_LOCK.lock().expect("acquire lock");
        // SAFETY: serialized via ENV_LOCK; cleanup at end.
        unsafe {
            std::env::set_var("INFINO_SUPERTABLE__WRITER_THREADS", "4");
            std::env::set_var("INFINO_SUPERTABLE__READER_THREADS", "auto");
        }
        let cfg = Config::load().expect("load with nested env override");
        assert_eq!(cfg.supertable.writer_threads, ThreadCount::Fixed(4));
        assert_eq!(cfg.supertable.reader_threads, ThreadCount::Auto);
        unsafe {
            std::env::remove_var("INFINO_SUPERTABLE__WRITER_THREADS");
            std::env::remove_var("INFINO_SUPERTABLE__READER_THREADS");
        }
    }

    #[test]
    fn user_config_path_falls_back_to_home() {
        let _g = ENV_LOCK.lock().expect("acquire lock");
        // SAFETY: serialized via ENV_LOCK.
        unsafe {
            std::env::remove_var("XDG_CONFIG_HOME");
            std::env::set_var("HOME", "/tmp/home-test");
        }
        let p = user_config_path().expect("path");
        assert_eq!(
            p,
            PathBuf::from("/tmp/home-test/.config/infino/config.yaml")
        );
        unsafe { std::env::remove_var("HOME") };
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
    fn compaction_nested_env_var_overrides_field() {
        let _g = ENV_LOCK.lock().expect("acquire lock");
        unsafe {
            std::env::set_var("INFINO_COMPACTION__TARGET_SUPERFILE_SIZE_MB", "4096");
            std::env::set_var("INFINO_COMPACTION__MIN_FILL_PERCENT", "60");
        }
        let cfg = Config::load().expect("load with compaction env override");
        assert_eq!(cfg.compaction.target_superfile_size_mb, 4096);
        assert_eq!(cfg.compaction.min_fill_percent, 60);
        unsafe {
            std::env::remove_var("INFINO_COMPACTION__TARGET_SUPERFILE_SIZE_MB");
            std::env::remove_var("INFINO_COMPACTION__MIN_FILL_PERCENT");
        }
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
}
