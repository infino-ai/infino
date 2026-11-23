// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`ConnectOptions`] — storage + cache configuration the URI scheme
//! can't carry (credentials, region, endpoint, disk cache). Passed to
//! [`connect_with`](crate::connect_with); plain [`connect`](crate::connect)
//! uses the default.

use std::{collections::HashMap, path::PathBuf, time::Duration};
#[cfg(any(test, feature = "test-helpers"))]
use std::{
    fmt::{Debug, Formatter, Result as FmtResult},
    sync::Arc,
};

#[cfg(any(test, feature = "test-helpers"))]
use crate::storage::StorageProvider;
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
    /// Disk-cache root. `None` (default) → no disk cache; cold reads go
    /// straight to object storage, by range. Each open table still keeps
    /// the lazy readers it opened for later queries, weighed by what they
    /// hold (the parsed footer and page index, and what later reads loaded
    /// into them: FTS doc maps and norms, vector norms, the range
    /// directory), up to 256 MiB a table, charged to the connection's memory
    /// budget, which takes them back when a query needs the room. Set → a
    /// local NVMe tier under this
    /// directory, per table (`<cache_dir>/<table>`).
    pub(crate) cache_dir: Option<PathBuf>,
    /// Disk-cache byte budget. `None` → a share of the total size of the
    /// disk holding `cache_dir` (see
    /// [`with_cache_budget_bytes`](Self::with_cache_budget_bytes)). Applies
    /// per table.
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
    /// Deadline for each request on a hosted connection, from connecting to
    /// the last byte of the answer. `None` (default) → 620 s, a little past
    /// the longest a hosted request can run (600 s). Ignored by local
    /// backends.
    pub(crate) request_timeout: Option<Duration>,
    /// Test-only: wraps every storage provider the connection builds (the
    /// catalog root and each table's), so a test can slow or count reads.
    #[cfg(any(test, feature = "test-helpers"))]
    pub(crate) storage_wrapper: Option<StorageWrapper>,
}

/// See [`ConnectOptions::storage_wrapper`].
#[cfg(any(test, feature = "test-helpers"))]
#[derive(Clone)]
pub(crate) struct StorageWrapper(
    pub(crate) Arc<dyn Fn(Arc<dyn StorageProvider>) -> Arc<dyn StorageProvider> + Send + Sync>,
);

#[cfg(any(test, feature = "test-helpers"))]
impl Debug for StorageWrapper {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("StorageWrapper")
    }
}

impl ConnectOptions {
    /// Default options — ambient credentials for object-store backends,
    /// disk cache off.
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(any(test, feature = "test-helpers"))]
    test_visible! {
        /// Test-only: pass every storage provider this connection builds through `wrap`, so a test
        /// can slow, count or hold its reads.
        fn with_storage_wrapper(
            mut self,
            wrap: impl Fn(Arc<dyn StorageProvider>) -> Arc<dyn StorageProvider>
            + Send
            + Sync
            + 'static,
        ) -> Self {
            self.storage_wrapper = Some(StorageWrapper(Arc::new(wrap)));
            self
        }
    }

    /// Enable a local disk cache rooted at `dir` (off by default). Cold
    /// superfile reads are cached to NVMe; per table, under
    /// `<dir>/<table>`. SQL that spills to disk spills under `dir` too,
    /// within a tenth of the disk's size, instead of the OS temp directory.
    /// No effect on `memory://` catalogs. Without it, each open table reads
    /// superfiles from object storage by range and keeps the readers it
    /// opened for later queries, within a byte bound charged to the
    /// connection's memory budget (see the `cache_dir` field).
    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// Set the disk-cache byte budget (per table). When unset, each table's
    /// cache may fill 90% of the total size of the disk holding the cache
    /// directory. Only meaningful with
    /// [`with_cache_dir`](Self::with_cache_dir).
    ///
    /// The default assumes the disk is there for one table's cache. It does
    /// not look at what else is on the disk, at a container's storage or
    /// memory limit (a `tmpfs` directory counts against memory), or at
    /// other tables and processes caching on the same disk, each of which
    /// gets the same share. Set the budget in any of those cases.
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
    /// tracks usage without enforcing; a positive value enforces a ceiling on
    /// what this connection's work may hold. Shared across all of the
    /// connection's tables.
    ///
    /// With a budget set, a full-text `LIKE` or `text_match` keeps each file's term dictionary
    /// across its steps while the budget is at most a quarter full, so it reads it once rather than
    /// at each step. Unset, it reads it at each step.
    ///
    /// Covers search and ingest. It does **not** cover compaction: a merge is
    /// not work done on behalf of a connection, and sizing it from a budget set
    /// for queries meant a pass could be refused on a machine with the memory
    /// to spare. What bounds a merge is the host — the compaction runner admits
    /// one at a time against free memory, and `compaction.max_merge_input_mb` caps
    /// the input bytes a single merge packs. So this is not a ceiling on the
    /// process: an optimize running alongside holds memory outside it.
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

    /// Set the deadline for each request on a hosted (`https://<host>/<db>`)
    /// connection, from connecting to the last byte of the answer. A request
    /// that has not finished by then fails with
    /// [`InfinoError::Unavailable`](crate::InfinoError::Unavailable).
    /// Ignored by local backends. Chainable.
    ///
    /// The default, 620 s, is a little past the longest a hosted request can
    /// run (600 s). A shorter deadline returns sooner from a stuck request,
    /// but an append, update or delete commits inside its request: one that
    /// times out may still commit, so check before retrying it. A zero
    /// deadline is rejected at connect.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    /// The configured request deadline, if any (used by the hosted
    /// transport).
    #[cfg_attr(not(feature = "remote"), allow(dead_code))]
    pub(crate) fn request_timeout(&self) -> Option<Duration> {
        self.request_timeout
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

    #[test]
    fn with_request_timeout_round_trips() {
        let o = ConnectOptions::new().with_request_timeout(Duration::from_secs(30));
        assert_eq!(o.request_timeout(), Some(Duration::from_secs(30)));
        assert_eq!(ConnectOptions::new().request_timeout(), None);
    }
}
