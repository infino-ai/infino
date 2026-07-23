// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! URI → storage backend parsing for the `connect` entry point.
//!
//! The backend is derived from the URI scheme;
//! [`ConnectOptions`](crate::ConnectOptions) carries only what the URI
//! can't (credentials, region/endpoint).

use std::path::PathBuf;

use crate::InfinoError;

/// A parsed catalog-root location. One catalog lives at the root; each
/// table is a child subtree ([`Backend::join`]).
#[derive(Debug, Clone)]
pub(crate) enum Backend {
    /// Local filesystem rooted at `root`.
    LocalFs { root: PathBuf },
    /// S3 (or S3-compatible) bucket with a logical key prefix.
    S3 { bucket: String, prefix: String },
    /// Azure blob container with a logical key prefix.
    Azure { container: String, prefix: String },
    /// GCS bucket with a logical key prefix.
    Gcs { bucket: String, prefix: String },
    /// A hosted-service endpoint reached over HTTP(S). `base_url` is the
    /// scheme + host (e.g. `https://base.example.ai`) and `database` is the
    /// path segment. Unlike the object-store backends this is a dispatch
    /// tag: it selects the remote transport rather than a `StorageProvider`.
    /// The fields are read only by the `remote` transport.
    #[cfg_attr(not(feature = "remote"), allow(dead_code))]
    Remote { base_url: String, database: String },
    /// In-process, non-persistent catalog (`memory://`).
    Memory,
}

impl Backend {
    /// The backend rooted at the `segment` child of this root — used to
    /// locate a single table's subtree under the catalog root.
    pub(crate) fn join(&self, segment: &str) -> Backend {
        match self {
            Backend::LocalFs { root } => Backend::LocalFs {
                root: root.join(segment),
            },
            Backend::S3 { bucket, prefix } => Backend::S3 {
                bucket: bucket.clone(),
                prefix: join_prefix(prefix, segment),
            },
            Backend::Azure { container, prefix } => Backend::Azure {
                container: container.clone(),
                prefix: join_prefix(prefix, segment),
            },
            Backend::Gcs { bucket, prefix } => Backend::Gcs {
                bucket: bucket.clone(),
                prefix: join_prefix(prefix, segment),
            },
            Backend::Memory => Backend::Memory,
            // A remote catalog root has no local storage subtree to descend
            // into; tables are addressed by name over the wire, not by prefix.
            Backend::Remote { .. } => self.clone(),
        }
    }
}

/// Join a logical object-store key prefix with a child segment, with no
/// leading/trailing slash surprises.
fn join_prefix(prefix: &str, segment: &str) -> String {
    let p = prefix.trim_matches('/');
    if p.is_empty() {
        segment.to_string()
    } else {
        format!("{p}/{segment}")
    }
}

/// Parse a catalog URI into its backend. Recognized schemes:
/// `memory://` (in-process), `s3://bucket/prefix`,
/// `az://container/prefix` (also `azure://`), `gs://bucket/prefix`
/// (also `gcs://`), `file://path`, a bare path (`./data`, `/abs/path`) →
/// local filesystem, and `https://host/<database>` (or `http://` for a
/// local endpoint) → a hosted-service connection.
pub(crate) fn parse_uri(uri: &str) -> Result<Backend, InfinoError> {
    if uri == "memory://" || uri == "memory:" || uri == "memory" {
        return Ok(Backend::Memory);
    }
    if let Some(rest) = uri.strip_prefix("s3://") {
        let (bucket, prefix) = split_bucket_prefix(rest);
        if bucket.is_empty() {
            return Err(InfinoError::Backend(format!(
                "s3 URI missing bucket: {uri}"
            )));
        }
        return Ok(Backend::S3 { bucket, prefix });
    }
    if let Some(rest) = uri
        .strip_prefix("az://")
        .or_else(|| uri.strip_prefix("azure://"))
    {
        let (container, prefix) = split_bucket_prefix(rest);
        if container.is_empty() {
            return Err(InfinoError::Backend(format!(
                "azure URI missing container: {uri}"
            )));
        }
        return Ok(Backend::Azure { container, prefix });
    }
    if let Some(rest) = uri
        .strip_prefix("gs://")
        .or_else(|| uri.strip_prefix("gcs://"))
    {
        let (bucket, prefix) = split_bucket_prefix(rest);
        if bucket.is_empty() {
            return Err(InfinoError::Backend(format!(
                "gcs URI missing bucket: {uri}"
            )));
        }
        return Ok(Backend::Gcs { bucket, prefix });
    }
    if let Some(rest) = uri.strip_prefix("file://") {
        return Ok(Backend::LocalFs {
            root: PathBuf::from(rest),
        });
    }
    if let Some(rest) = uri
        .strip_prefix("https://")
        .or_else(|| uri.strip_prefix("http://"))
    {
        let is_http = uri.starts_with("http://");
        let (host, database) = match rest.split_once('/') {
            Some((host, db)) => (host, db.trim_matches('/')),
            None => (rest, ""),
        };
        if host.is_empty() {
            return Err(InfinoError::Backend(format!(
                "remote URI missing host: {uri}"
            )));
        }
        // A bearer credential must never travel in the clear, so plaintext
        // `http://` is accepted only for a local endpoint; any other host
        // must use `https://`.
        if is_http && !is_localhost(host) {
            return Err(InfinoError::Backend(format!(
                "http:// is only allowed for localhost; use https:// for a remote host: {uri}"
            )));
        }
        if database.is_empty() {
            return Err(InfinoError::Backend(format!(
                "remote URI missing database path (expected https://host/<database>): {uri}"
            )));
        }
        let scheme = if is_http { "http://" } else { "https://" };
        return Ok(Backend::Remote {
            base_url: format!("{scheme}{host}"),
            database: database.to_string(),
        });
    }
    // A bare path is a local filesystem root. Any other `scheme://` is
    // unsupported (don't silently treat `gdrive://…` as a directory name).
    if uri.contains("://") {
        return Err(InfinoError::Backend(format!(
            "unsupported catalog URI scheme: {uri}"
        )));
    }
    Ok(Backend::LocalFs {
        root: PathBuf::from(uri),
    })
}

/// Whether `host` (with an optional `:port`) is a local endpoint — the only
/// case where plaintext `http://` is allowed. A remote host must use TLS so
/// the bearer credential is never sent in the clear.
fn is_localhost(host: &str) -> bool {
    let bare = host.split(':').next().unwrap_or(host);
    bare == "localhost" || bare == "127.0.0.1"
}

/// Split `bucket/key/prefix` into `("bucket", "key/prefix")`; a bare
/// `bucket` yields an empty prefix.
fn split_bucket_prefix(rest: &str) -> (String, String) {
    match rest.split_once('/') {
        Some((bucket, prefix)) => (bucket.to_string(), prefix.trim_matches('/').to_string()),
        None => (rest.to_string(), String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_memory() {
        assert!(matches!(parse_uri("memory://"), Ok(Backend::Memory)));
    }

    #[test]
    fn parses_bare_path_as_localfs() {
        match parse_uri("./data").expect("parse") {
            Backend::LocalFs { root } => assert_eq!(root, PathBuf::from("./data")),
            other => panic!("expected LocalFs, got {other:?}"),
        }
    }

    #[test]
    fn parses_s3_bucket_and_prefix() {
        match parse_uri("s3://my-bucket/some/prefix").expect("parse") {
            Backend::S3 { bucket, prefix } => {
                assert_eq!(bucket, "my-bucket");
                assert_eq!(prefix, "some/prefix");
            }
            other => panic!("expected S3, got {other:?}"),
        }
    }

    #[test]
    fn join_appends_table_segment() {
        let b = parse_uri("s3://b/root").expect("parse").join("users");
        match b {
            Backend::S3 { prefix, .. } => assert_eq!(prefix, "root/users"),
            other => panic!("expected S3, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_scheme() {
        assert!(parse_uri("gdrive://bucket/x").is_err());
    }

    #[test]
    fn parses_gcs_bucket_and_prefix() {
        match parse_uri("gs://my-bucket/some/prefix").expect("parse") {
            Backend::Gcs { bucket, prefix } => {
                assert_eq!(bucket, "my-bucket");
                assert_eq!(prefix, "some/prefix");
            }
            other => panic!("expected Gcs, got {other:?}"),
        }
    }

    #[test]
    fn parses_gcs_alias_scheme() {
        assert!(matches!(
            parse_uri("gcs://b/p").expect("parse"),
            Backend::Gcs { .. }
        ));
    }

    #[test]
    fn gcs_join_appends_table_segment() {
        match parse_uri("gs://b/root").expect("parse").join("users") {
            Backend::Gcs { prefix, .. } => assert_eq!(prefix, "root/users"),
            other => panic!("expected Gcs, got {other:?}"),
        }
    }

    #[test]
    fn rejects_gcs_uri_without_bucket() {
        assert!(parse_uri("gs://").is_err());
    }

    #[test]
    fn parses_https_remote() {
        match parse_uri("https://base.example.ai/my-db").expect("parse") {
            Backend::Remote { base_url, database } => {
                assert_eq!(base_url, "https://base.example.ai");
                assert_eq!(database, "my-db");
            }
            other => panic!("expected Remote, got {other:?}"),
        }
    }

    #[test]
    fn http_allowed_only_for_localhost() {
        assert!(matches!(
            parse_uri("http://localhost:8080/db").expect("parse"),
            Backend::Remote { .. }
        ));
        assert!(matches!(
            parse_uri("http://127.0.0.1:9000/db").expect("parse"),
            Backend::Remote { .. }
        ));
        // A plaintext bearer credential to a remote host is refused.
        assert!(parse_uri("http://example.com/db").is_err());
    }

    #[test]
    fn remote_requires_db_path() {
        assert!(parse_uri("https://base.example.ai/").is_err());
        assert!(parse_uri("https://base.example.ai").is_err());
    }

    #[test]
    fn remote_root_join_is_noop() {
        let root = parse_uri("https://base.example.ai/my-db").expect("parse");
        assert!(
            matches!(root.join("users"), Backend::Remote { database, .. } if database == "my-db")
        );
    }
}
