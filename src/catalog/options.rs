// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`ConnectOptions`] — storage configuration the URI scheme can't carry
//! (credentials, region, endpoint). Passed to
//! [`connect_with`](crate::connect_with); plain [`connect`](crate::connect)
//! uses the default.

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
}

impl ConnectOptions {
    /// Default options — ambient credentials for object-store backends,
    /// no overrides.
    pub fn new() -> Self {
        Self::default()
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
