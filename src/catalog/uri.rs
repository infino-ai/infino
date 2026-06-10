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
            Backend::Memory => Backend::Memory,
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
/// `az://container/prefix` (also `azure://`), `file://path`, and a bare
/// path (`./data`, `/abs/path`) → local filesystem.
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
    if let Some(rest) = uri.strip_prefix("file://") {
        return Ok(Backend::LocalFs {
            root: PathBuf::from(rest),
        });
    }
    // A bare path is a local filesystem root. Any other `scheme://` is
    // unsupported (don't silently treat `gs://…` as a directory name).
    if uri.contains("://") {
        return Err(InfinoError::Backend(format!(
            "unsupported catalog URI scheme: {uri}"
        )));
    }
    Ok(Backend::LocalFs {
        root: PathBuf::from(uri),
    })
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
        assert!(parse_uri("gs://bucket/x").is_err());
    }
}
