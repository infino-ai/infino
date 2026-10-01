// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`ConnectOptions`] — storage + cache configuration the URI scheme
//! can't carry (credentials, region, endpoint, disk cache). Passed to
//! [`connect_with`](crate::connect_with); plain [`connect`](crate::connect)
//! uses the default.

use std::{collections::HashMap, path::PathBuf};

use crate::supertable::{Consistency, reader_cache::ColdFetchMode as InternalColdFetchMode};

/// How a disk-cache miss is serviced when reading cold superfiles from
/// object storage. The cache-servicing modes need a cache
/// ([`ConnectOptions::with_cache_dir`]); `RangeOnly` is the cacheless
/// path and does not currently use a disk cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColdFetchMode {
    /// Parallel range-GETs that tee into both the live query and the
    /// cache fill — 1× object-store bandwidth per cold miss.
    HybridWithPrefetch,
    /// Range-GETs straight through with no cache fill — best for
    /// query-once / stateless callers.
    RangeOnly,
    /// A lazy reader serves the query immediately (a few range-GETs);
    /// the full superfile is downloaded to the cache in the background.
    /// Lowest cold-query latency — the default.
    #[default]
    LazyForegroundWithBackgroundFill,
}

impl ColdFetchMode {
    pub(crate) fn to_internal(self) -> InternalColdFetchMode {
        match self {
            ColdFetchMode::HybridWithPrefetch => InternalColdFetchMode::HybridWithPrefetch,
            ColdFetchMode::RangeOnly => InternalColdFetchMode::RangeOnly,
            ColdFetchMode::LazyForegroundWithBackgroundFill => {
                InternalColdFetchMode::LazyForegroundWithBackgroundFill
            }
        }
    }
}

/// Storage configuration for [`connect_with`](crate::connect_with).
///
/// The storage **backend** is derived from the URI scheme passed to
/// `connect` (`s3://…`, `az://…`, `file://…`, `memory://`, or a bare
/// path), not from these options — `ConnectOptions` carries only what
/// the URI can't express. The common cases need no options:
/// `connect("./data")` and `connect("s3://bucket/prefix")` (ambient
/// cloud identity) both work with the default.
#[derive(Debug, Clone, Default)]
pub struct ConnectOptions {
    /// Credentials/tuning for the URI-selected backend, keyed by
    /// `object_store` config strings. Empty → ambient cloud identity.
    pub(crate) storage_options: HashMap<String, String>,
    /// Disk-cache root. `None` (default) → caching off; cold reads go
    /// straight to object storage. Set → a local NVMe tier under this
    /// directory, per table (`<cache_dir>/<table>`).
    pub(crate) cache_dir: Option<PathBuf>,
    /// Disk-cache byte budget. `None` → the cache's built-in default.
    /// Applies per table.
    pub(crate) cache_budget_bytes: Option<u64>,
    /// Cold-fetch strategy when the disk cache is enabled.
    pub(crate) cold_fetch_mode: ColdFetchMode,
    /// Per-connection memory (heap) budget in bytes. `None` (default) tracks
    /// usage without enforcing; `Some(n)` enforces a ceiling so one connection
    /// can't exhaust process memory. Applies to the whole connection, shared
    /// across supertables.
    pub(crate) connection_memory_budget_bytes: Option<u64>,
    /// Read-consistency policy for every table opened or created on this
    /// connection (see [`Consistency`]). Default:
    /// [`Consistency::BoundedStaleness`] with a 1s window — the engine default,
    /// which amortizes the per-query manifest-pointer re-check. Set
    /// [`Consistency::Strong`] for a pointer re-check on every query.
    pub(crate) read_consistency: Consistency,
    /// Probe the backend at `connect`. Default `false`; opt in for
    /// fail-fast on bad credentials.
    pub(crate) validate: bool,
    /// API key for a hosted (`https://…`) connect target, sent as a bearer
    /// credential on every request. Ignored by local (object-store) backends.
    /// When unset, a hosted connection falls back to the `INFINO_API_KEY`
    /// environment variable.
    pub(crate) api_key: Option<String>,
}

impl ConnectOptions {
    /// Default options — ambient credentials for object-store backends,
    /// disk cache off.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable a local disk cache rooted at `dir` (off by default). Cold
    /// superfile reads are cached to NVMe; per table, under
    /// `<dir>/<table>`. No effect on `memory://` catalogs.
    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// Set the disk-cache byte budget (per table). Defaults to the
    /// cache's built-in budget when unset. Only meaningful with
    /// [`with_cache_dir`](Self::with_cache_dir).
    pub fn with_cache_budget_bytes(mut self, bytes: u64) -> Self {
        self.cache_budget_bytes = Some(bytes);
        self
    }

    /// Choose how cold misses are serviced (see [`ColdFetchMode`]). Only
    /// meaningful with [`with_cache_dir`](Self::with_cache_dir).
    /// `RangeOnly` is the cacheless path and is rejected if a `cache_dir` is set.
    pub fn with_cold_fetch_mode(mut self, mode: ColdFetchMode) -> Self {
        self.cold_fetch_mode = mode;
        self
    }

    /// Set a per-connection memory budget, in bytes. Unset (the default)
    /// tracks usage without enforcing; a positive value enforces a ceiling so
    /// one connection can't exhaust process memory. Shared across all of the
    /// connection's tables.
    pub fn with_connection_memory_budget_bytes(mut self, bytes: u64) -> Self {
        self.connection_memory_budget_bytes = Some(bytes);
        self
    }

    /// Set one storage option (e.g. `aws_access_key_id`,
    /// `azure_storage_account_key`). An unknown or cross-backend key
    /// errors at connect time. Chainable.
    pub fn with_storage_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.storage_options.insert(key.into(), value.into());
        self
    }

    /// Set the read-consistency policy for tables on this connection (see
    /// [`Consistency`]). Unset defaults to [`Consistency::BoundedStaleness`]
    /// with a 1s window (the engine default); pass [`Consistency::Strong`] to
    /// re-check the manifest pointer on every query. Chainable.
    pub fn with_read_consistency(mut self, consistency: Consistency) -> Self {
        self.read_consistency = consistency;
        self
    }

    /// Probe the object store at `connect` (default `false`). `true`
    /// fails fast on bad credentials instead of on first use.
    pub fn with_validate(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }

    /// Set the API key for a hosted (`https://<host>/<db>`) connect target,
    /// sent as a bearer credential. Ignored by local backends. When unset, a
    /// hosted connection falls back to the `INFINO_API_KEY` environment
    /// variable. Chainable.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// The configured API key, if any (used by the hosted transport). Only the
    /// `remote` transport reads this, so it is dead code in a build without it.
    #[cfg_attr(not(feature = "remote"), allow(dead_code))]
    pub(crate) fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn read_consistency_defaults_to_bounded_staleness() {
        assert_eq!(
            ConnectOptions::new().read_consistency,
            Consistency::BoundedStaleness(Duration::from_secs(1)),
            "unset read consistency defaults to the engine's BoundedStaleness(1s)"
        );
    }

    #[test]
    fn with_read_consistency_overrides_the_default() {
        let opts = ConnectOptions::new().with_read_consistency(Consistency::Strong);
        assert_eq!(opts.read_consistency, Consistency::Strong);
    }

    #[test]
    fn with_storage_option_round_trips() {
        let o = ConnectOptions::new().with_storage_option("aws_region", "us-east-1");
        assert_eq!(
            o.storage_options.get("aws_region").map(String::as_str),
            Some("us-east-1")
        );
    }

    #[test]
    fn with_api_key_round_trips() {
        let o = ConnectOptions::new().with_api_key("ik_test");
        assert_eq!(o.api_key(), Some("ik_test"));
        assert_eq!(ConnectOptions::new().api_key(), None);
    }
}
