// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`ConnectOptions`] — storage + cache configuration the URI scheme
//! can't carry (credentials, region, endpoint, disk cache). Passed to
//! [`connect_with`](crate::connect_with); plain [`connect`](crate::connect)
//! uses the default.

use std::path::PathBuf;

use crate::supertable::reader_cache::ColdFetchMode as InternalColdFetchMode;

/// Explicit S3-compatible endpoint + static credentials — for MinIO,
/// Cloudflare R2, Ceph, or a test S3 server. When unset, S3 uses the
/// ambient AWS default-credential chain and default region.
#[derive(Debug, Clone)]
pub(crate) struct S3Config {
    pub(crate) endpoint: String,
    pub(crate) region: String,
    pub(crate) access_key: String,
    pub(crate) secret_key: String,
}

/// How a disk-cache miss is serviced when reading cold superfiles from
/// object storage. Only meaningful when a disk cache is configured
/// ([`ConnectOptions::with_cache_dir`]).
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
/// `connect("./data")` and `connect("s3://bucket/prefix")` (ambient AWS
/// credentials) both work with the default.
#[derive(Debug, Clone, Default)]
pub struct ConnectOptions {
    pub(crate) s3: Option<S3Config>,
    /// Disk-cache root. `None` (default) → caching off; cold reads go
    /// straight to object storage. Set → a local NVMe tier under this
    /// directory, per table (`<cache_dir>/<table>`).
    pub(crate) cache_dir: Option<PathBuf>,
    /// Disk-cache byte budget. `None` → the cache's built-in default.
    /// Applies per table.
    pub(crate) cache_budget_bytes: Option<u64>,
    /// Cold-fetch strategy when the disk cache is enabled.
    pub(crate) cold_fetch_mode: ColdFetchMode,
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
    pub fn with_cold_fetch_mode(mut self, mode: ColdFetchMode) -> Self {
        self.cold_fetch_mode = mode;
        self
    }

    /// Use an explicit S3-compatible endpoint with static credentials
    /// (MinIO / R2 / Ceph / a test S3 server) instead of the ambient AWS
    /// default-credential chain. Only affects `s3://` catalogs.
    pub fn with_s3_endpoint(
        mut self,
        endpoint: impl Into<String>,
        region: impl Into<String>,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
    ) -> Self {
        self.s3 = Some(S3Config {
            endpoint: endpoint.into(),
            region: region.into(),
            access_key: access_key.into(),
            secret_key: secret_key.into(),
        });
        self
    }
}
