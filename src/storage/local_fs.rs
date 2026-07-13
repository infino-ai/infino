// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! LocalFS-backed [`StorageProvider`].
//!
//! Wraps `object_store::local::LocalFileSystem` so the same
//! supertable code paths exercise both LocalFS (dev / tests /
//! single-node) and S3 (production / multi-node) without
//! backend-specific branching above the storage trait.
//!
//! The path scoping is: every URI handed to a method is
//! relative to the `root` passed at construction. So
//! `provider.get("data/seg-abc.sf.parquet")` reads
//! `<root>/data/seg-abc.sf.parquet`. No upward traversal — paths with
//! `..` get rejected by `object_store::path::Path`.

use std::{ops::Range, path::PathBuf, sync::Arc};

use async_trait::async_trait;
use bytes::Bytes;
use fs4::tokio::AsyncFileExt;
use futures::TryStreamExt;
use object_store::{
    Error as ObjError, MultipartUpload, ObjectStore, ObjectStoreExt, PutMode, PutOptions,
    PutPayload, local::LocalFileSystem, path::Path as ObjPath,
};

use super::{ObjectMeta, StorageError, StorageProvider};

#[derive(Debug)]
pub struct LocalFsStorageProvider {
    root: PathBuf,
    store: Arc<LocalFileSystem>,
}

impl LocalFsStorageProvider {
    /// Construct a new LocalFS-backed provider rooted at
    /// `root`. The directory is created (recursively) if it
    /// doesn't exist; surfacing
    /// [`StorageError::Permanent`] only if creation fails
    /// (permission denied, parent doesn't exist + we can't
    /// mkdir, etc.).
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, StorageError> {
        let root: PathBuf = root.into();
        std::fs::create_dir_all(&root).map_err(|e| StorageError::Permanent {
            uri: root.display().to_string(),
            source: Box::new(e),
        })?;
        let store =
            LocalFileSystem::new_with_prefix(&root).map_err(|e| StorageError::Permanent {
                uri: root.display().to_string(),
                source: Box::new(e),
            })?;
        Ok(Self {
            root,
            store: Arc::new(store),
        })
    }

    /// Filesystem root this provider is scoped to. Useful for
    /// tests that need to inspect on-disk state directly.
    pub fn root(&self) -> &PathBuf {
        &self.root
    }

    fn path(uri: &str) -> Result<ObjPath, StorageError> {
        ObjPath::parse(uri).map_err(|e| StorageError::Permanent {
            uri: uri.into(),
            source: Box::new(e),
        })
    }
}

/// Translate an `object_store::Error` to our `StorageError`.
///
/// The mapping:
/// - `NotFound` → `NotFound`
/// - `AlreadyExists` / `Precondition` → `PreconditionFailed`
/// - everything else → `Permanent` (object_store has already
///   retried transient failures internally per its
///   `RetryConfig`; by the time we see one here it's
///   exhausted)
fn translate(uri: &str, e: ObjError) -> StorageError {
    match e {
        ObjError::NotFound { .. } => StorageError::NotFound { uri: uri.into() },
        ObjError::AlreadyExists { .. } | ObjError::Precondition { .. } => {
            StorageError::PreconditionFailed { uri: uri.into() }
        }
        ObjError::Generic { source, .. } => StorageError::TransientExhausted {
            uri: uri.into(),
            source,
        },
        other => StorageError::Permanent {
            uri: uri.into(),
            source: Box::new(other),
        },
    }
}

#[async_trait]
impl StorageProvider for LocalFsStorageProvider {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        let path = Self::path(uri)?;
        let meta = self
            .store
            .head(&path)
            .await
            .map_err(|e| translate(uri, e))?;
        Ok(ObjectMeta {
            size: meta.size as u64,
            etag: meta.e_tag,
            last_modified: meta.last_modified.into(),
        })
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        let path = Self::path(uri)?;
        let result = self.store.get(&path).await.map_err(|e| translate(uri, e))?;
        // `GetResult.meta` matches the version we're about to
        // read — no separate HEAD needed to capture the etag.
        let meta = ObjectMeta {
            size: result.meta.size as u64,
            etag: result.meta.e_tag.clone(),
            last_modified: result.meta.last_modified.into(),
        };
        let bytes = result.bytes().await.map_err(|e| translate(uri, e))?;
        Ok((bytes, meta))
    }

    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(uri = uri, len = range.end - range.start))
    )]
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        let path = Self::path(uri)?;
        self.store
            .get_range(&path, range)
            .await
            .map_err(|e| translate(uri, e))
    }

    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        let path = Self::path(uri)?;
        let opts = PutOptions {
            mode: PutMode::Create,
            ..Default::default()
        };
        self.store
            .put_opts(&path, PutPayload::from_bytes(bytes), opts)
            .await
            .map(|r| r.e_tag)
            .map_err(|e| translate(uri, e))
    }

    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        let path = Self::path(uri)?;
        match expected_etag {
            // None == create-only-if-absent. Same as put_atomic.
            None => {
                let opts = PutOptions {
                    mode: PutMode::Create,
                    ..Default::default()
                };
                self.store
                    .put_opts(&path, PutPayload::from_bytes(bytes), opts)
                    .await
                    .map(|r| r.e_tag)
                    .map_err(|e| translate(uri, e))
            }
            // Some(tag) == update-if-etag-matches.
            //
            // `object_store::LocalFileSystem` doesn't implement
            // `PutMode::Update` directly (it surfaces `NotImplemented`).
            // We implement etag-conditional update as
            // read-then-overwrite, bracketed by an advisory
            // `flock` on `<root>/_supertable/.lock` so two
            // processes can't both observe the same prior etag
            // and race the overwrite. POSIX `flock` releases on
            // fd close, so the lock file drops at the end of
            // this branch and the next contender proceeds.
            // S3 / GCS providers use native conditional PUT and
            // don't need this scaffolding — see
            // `S3StorageProvider::put_if_match`.
            Some(expected) => {
                let lock_path = self.root.join("_supertable").join(".lock");
                // The pointer commit path already creates
                // `_supertable/` on the first write; doing it
                // here too is idempotent + makes the lock
                // robust against any other call site that
                // routes through put_if_match before the
                // pointer commits.
                if let Some(parent) = lock_path.parent() {
                    let _ = tokio::fs::create_dir_all(parent).await;
                }
                let lock_file = tokio::fs::OpenOptions::new()
                    .create(true)
                    .read(true)
                    .write(true)
                    .truncate(false)
                    .open(&lock_path)
                    .await
                    .map_err(|e| StorageError::Permanent {
                        uri: uri.into(),
                        source: Box::new(e),
                    })?;
                lock_file
                    .lock_exclusive()
                    .map_err(|e| StorageError::Permanent {
                        uri: uri.into(),
                        source: Box::new(e),
                    })?;
                // Lock held below until `lock_file` drops at
                // end of branch (or early-return). Holding it
                // across `.await` points blocks the
                // tokio worker; head + put on LocalFS are
                // microseconds, so the worst-case stall is
                // bounded.

                let result: Result<Option<String>, StorageError> = async {
                    let current = self
                        .store
                        .head(&path)
                        .await
                        .map_err(|e| translate(uri, e))?;
                    let current_etag = current.e_tag.as_deref().unwrap_or("");
                    if current_etag != expected {
                        return Err(StorageError::PreconditionFailed { uri: uri.into() });
                    }
                    let opts = PutOptions {
                        mode: PutMode::Overwrite,
                        ..Default::default()
                    };
                    self.store
                        .put_opts(&path, PutPayload::from_bytes(bytes), opts)
                        .await
                        .map(|r| r.e_tag)
                        .map_err(|e| translate(uri, e))
                }
                .await;
                // `lock_file` drops here → POSIX flock
                // releases when the fd closes. Best-effort
                // explicit unlock too, ignoring failures (the
                // kernel cleans up regardless).
                let _ = lock_file.unlock_async().await;
                result
            }
        }
    }

    async fn put_multipart(&self, uri: &str) -> Result<Box<dyn MultipartUpload>, StorageError> {
        let path = Self::path(uri)?;
        self.store
            .put_multipart(&path)
            .await
            .map_err(|e| translate(uri, e))
    }

    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        let path = Self::path(uri)?;
        match self.store.delete(&path).await {
            Ok(()) => Ok(()),
            Err(ObjError::NotFound { .. }) => Ok(()),
            Err(e) => Err(translate(uri, e)),
        }
    }

    async fn list_with_prefix_metadata(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
        let path = ObjPath::from(prefix);
        let mut stream = self.store.list(Some(&path));
        let mut out = Vec::new();
        while let Some(meta) = stream.try_next().await.map_err(|e| translate(prefix, e))? {
            out.push((
                meta.location.to_string(),
                ObjectMeta {
                    size: meta.size,
                    etag: meta.e_tag,
                    last_modified: meta.last_modified.into(),
                },
            ));
        }
        Ok(out)
    }

    fn object_store_handle(&self, uri: &str) -> Option<(Arc<dyn ObjectStore>, ObjPath)> {
        // The prefix (root) is baked into the LocalFileSystem store, so
        // the object key is the bare uri.
        let path = Self::path(uri).ok()?;
        Some((Arc::clone(&self.store) as Arc<dyn ObjectStore>, path))
    }
}

#[cfg(test)]
mod tests {
    //! `StorageProvider` trait contract against
    //! `LocalFsStorageProvider`.
    //!
    //! Covers: round-trip put + get; head returns accurate
    //! size + etag presence; range-fetch over a known
    //! object; `put_atomic` rejects an already-existing
    //! target; `put_if_match` honors ETag preconditions
    //! (success + failure paths) — the OCC primitive the
    //! manifest-pointer commit rides on; `delete` is
    //! idempotent on a missing target; `get` / `head` /
    //! `get_range` return `NotFound` on missing; advisory
    //! flock file is created on `put_if_match` (the TOCTOU-
    //! closing path); `put_multipart` returns a handle.
    use std::{
        error::Error,
        time::{Duration, SystemTime},
    };

    use bytes::Bytes;
    use tempfile::TempDir;

    use super::*;

    fn provider() -> (TempDir, LocalFsStorageProvider) {
        let dir = TempDir::new().expect("tempdir");
        let p = LocalFsStorageProvider::new(dir.path()).expect("provider");
        (dir, p)
    }

    #[tokio::test]
    async fn put_then_get_roundtrip() {
        let (_dir, p) = provider();
        let payload = Bytes::from_static(b"hello supertable storage");
        p.put_atomic("data/seg-abc.sf.parquet", payload.clone())
            .await
            .expect("put");
        let (got, _) = p.get("data/seg-abc.sf.parquet").await.expect("get");
        assert_eq!(got, payload);
    }

    #[tokio::test]
    async fn get_if_none_match_skips_unchanged_and_reads_changed() {
        // Exercises the trait's default conditional-get (plain get +
        // local etag compare) — the semantics every backend must
        // honor: matching etag → `None`; changed object → full body.
        let (_dir, p) = provider();
        let uri = "cond/pointer";
        p.put_atomic(uri, Bytes::from_static(b"v1"))
            .await
            .expect("put v1");
        let (_, meta) = p.get(uri).await.expect("get v1");
        let etag_v1 = meta.etag.expect("localfs reports etags");

        let unchanged = p.get_if_none_match(uri, &etag_v1).await.expect("probe");
        assert!(unchanged.is_none(), "matching etag answers not-modified");

        // Overwrite with different content (and length, so the
        // mtime+size etag can't collide within one clock tick).
        p.put_if_match(uri, Bytes::from_static(b"v2-longer"), Some(&etag_v1))
            .await
            .expect("cas overwrite");
        let changed = p
            .get_if_none_match(uri, &etag_v1)
            .await
            .expect("probe changed")
            .expect("changed object returns the body");
        assert_eq!(changed.0, Bytes::from_static(b"v2-longer"));
        assert_ne!(changed.1.etag.expect("etag"), etag_v1);
    }

    #[tokio::test]
    async fn head_returns_accurate_size() {
        let (_dir, p) = provider();
        let payload = Bytes::from_static(&[0xABu8; 1024]);
        p.put_atomic("data/seg-head.sf.parquet", payload)
            .await
            .expect("put");

        let meta = p.head("data/seg-head.sf.parquet").await.expect("head");
        assert_eq!(meta.size, 1024);
        // LocalFS surfaces an mtime-derived etag; other
        // backends may not. Assert presence, not value.
        assert!(meta.etag.is_some(), "LocalFS should surface an etag");
    }

    #[tokio::test]
    async fn get_range_reads_exact_slice() {
        let (_dir, p) = provider();
        let payload: Vec<u8> = (0u8..=255).collect();
        p.put_atomic("data/seg-range.sf.parquet", Bytes::from(payload.clone()))
            .await
            .expect("put");

        let slice = p
            .get_range("data/seg-range.sf.parquet", 32..64)
            .await
            .expect("range");
        assert_eq!(slice.as_ref(), &payload[32..64]);

        let tail = p
            .get_range("data/seg-range.sf.parquet", 255..256)
            .await
            .expect("range tail");
        assert_eq!(tail.as_ref(), &payload[255..256]);
    }

    #[tokio::test]
    async fn put_atomic_rejects_existing() {
        let (_dir, p) = provider();
        let payload = Bytes::from_static(b"first writer wins");
        p.put_atomic("manifest-lists/list-1.json", payload.clone())
            .await
            .expect("first put");

        let err = p
            .put_atomic("manifest-lists/list-1.json", Bytes::from_static(b"second"))
            .await
            .expect_err("second put must fail");
        assert!(
            matches!(err, StorageError::PreconditionFailed { .. }),
            "expected PreconditionFailed, got {err:?}"
        );

        let (got, _) = p
            .get("manifest-lists/list-1.json")
            .await
            .expect("get after losing put");
        assert_eq!(got, payload);
    }

    #[tokio::test]
    async fn put_if_match_with_correct_etag_succeeds() {
        let (_dir, p) = provider();
        p.put_atomic("ptr/current", Bytes::from_static(b"v1"))
            .await
            .expect("initial");
        let meta = p.head("ptr/current").await.expect("head");
        let etag = meta.etag.expect("LocalFS etag");

        p.put_if_match("ptr/current", Bytes::from_static(b"v2"), Some(&etag))
            .await
            .expect("conditional update with correct etag");

        let (got, _) = p.get("ptr/current").await.expect("get v2");
        assert_eq!(got.as_ref(), b"v2");
    }

    #[tokio::test]
    async fn cas_conformance_holds() {
        let (_dir, p) = provider();
        crate::test_helpers::cas_conformance::cas_conformance(&p, "ptr/cas-conf", true).await;
    }

    #[tokio::test]
    async fn put_if_match_with_stale_etag_fails() {
        let (_dir, p) = provider();
        p.put_atomic("ptr/current", Bytes::from_static(b"v1"))
            .await
            .expect("initial");
        let stale_meta = p.head("ptr/current").await.expect("head v1");
        let stale_etag = stale_meta.etag.clone().expect("etag v1");

        // Legitimate writer wins the OCC race.
        p.put_if_match(
            "ptr/current",
            Bytes::from_static(b"v_intermediate"),
            Some(&stale_etag),
        )
        .await
        .expect("legitimate update");

        // Second writer with the now-stale etag must lose.
        let err = p
            .put_if_match(
                "ptr/current",
                Bytes::from_static(b"v_stale_writer"),
                Some(&stale_etag),
            )
            .await
            .expect_err("stale etag must fail");
        assert!(
            matches!(err, StorageError::PreconditionFailed { .. }),
            "expected PreconditionFailed, got {err:?}"
        );

        let (got, _) = p.get("ptr/current").await.expect("get");
        assert_eq!(got.as_ref(), b"v_intermediate");
    }

    #[tokio::test]
    async fn delete_is_idempotent() {
        let (_dir, p) = provider();
        p.put_atomic("data/orphan.sf.parquet", Bytes::from_static(b"x"))
            .await
            .expect("put");

        p.delete("data/orphan.sf.parquet")
            .await
            .expect("first delete");
        p.delete("data/orphan.sf.parquet")
            .await
            .expect("second delete (idempotent)");
        p.delete("data/never-existed.sf.parquet")
            .await
            .expect("delete of never-existing");
    }

    #[tokio::test]
    async fn missing_object_returns_not_found() {
        let (_dir, p) = provider();
        let err = p
            .head("data/no-such.sf.parquet")
            .await
            .expect_err("head missing");
        assert!(matches!(err, StorageError::NotFound { .. }));

        let err = p
            .get("data/no-such.sf.parquet")
            .await
            .expect_err("get missing");
        assert!(matches!(err, StorageError::NotFound { .. }));

        let err = p
            .get_range("data/no-such.sf.parquet", 0..1)
            .await
            .expect_err("get_range missing");
        assert!(matches!(err, StorageError::NotFound { .. }));
    }

    #[tokio::test]
    async fn put_at_nested_path_creates_dirs() {
        // Forward-slash-separated paths are object_store
        // idiom; LocalFileSystem creates intermediate dirs.
        let (_dir, p) = provider();
        p.put_atomic("a/b/c/d/leaf.bin", Bytes::from_static(b"deep"))
            .await
            .expect("nested put");
        let (got, _) = p.get("a/b/c/d/leaf.bin").await.expect("nested get");
        assert_eq!(got.as_ref(), b"deep");
    }

    #[tokio::test]
    async fn put_if_match_creates_supertable_lock_file() {
        // `put_if_match`'s Some(etag) branch acquires an
        // advisory flock on `<root>/_supertable/.lock` to
        // close the read-then-overwrite TOCTOU window. The
        // lock file persists (best-effort cleanup is not
        // attempted), so its presence after a successful
        // conditional update is a direct signal the lock
        // path was exercised.
        let dir = TempDir::new().expect("tempdir");
        let p = LocalFsStorageProvider::new(dir.path()).expect("provider");
        p.put_atomic("ptr/current", Bytes::from_static(b"v1"))
            .await
            .expect("initial");
        let etag = p
            .head("ptr/current")
            .await
            .expect("head")
            .etag
            .expect("etag");
        p.put_if_match("ptr/current", Bytes::from_static(b"v2"), Some(&etag))
            .await
            .expect("conditional update");

        let lock_path = dir.path().join("_supertable").join(".lock");
        assert!(
            lock_path.exists(),
            "expected advisory lock file at {lock_path:?}"
        );
    }

    #[tokio::test]
    async fn put_multipart_returns_handle() {
        // Surface check only — driving real part PUTs
        // happens at the supertable commit layer.
        let (_dir, p) = provider();
        let mut upload = p
            .put_multipart("data/multipart-test.sf.parquet")
            .await
            .expect("multipart handle");
        upload.abort().await.expect("abort");
    }

    #[tokio::test]
    async fn list_with_prefix_returns_matching_keys() {
        let (_dir, p) = provider();
        for key in ["seg/a.parquet", "seg/b.parquet", "other/c.parquet"] {
            p.put_atomic(key, Bytes::from_static(b"x"))
                .await
                .expect("put");
        }
        let mut under_seg = p.list_with_prefix("seg").await.expect("list");
        under_seg.sort();
        assert_eq!(under_seg, vec!["seg/a.parquet", "seg/b.parquet"]);

        let all = p.list_with_prefix("").await.expect("list all");
        assert_eq!(all.len(), 3);

        let none = p
            .list_with_prefix("does-not-exist")
            .await
            .expect("list empty");
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn list_with_prefix_metadata_returns_mtime_and_size() {
        let (_dir, p) = provider();
        let before = SystemTime::now()
            .checked_sub(Duration::from_secs(2))
            .expect("parsing failed");
        p.put_atomic("data/a.parquet", Bytes::from_static(b"hello"))
            .await
            .expect("put");
        let after = SystemTime::now()
            .checked_add(Duration::from_secs(2))
            .expect("parsing failed");

        let mut entries = p
            .list_with_prefix_metadata("data/")
            .await
            .expect("list metadata");
        assert_eq!(entries.len(), 1);
        entries.sort_by_key(|(key, _)| key.clone());
        let (key, meta) = &entries[0];
        assert_eq!(key, "data/a.parquet");
        assert!(meta.last_modified >= before, "mtime too old");
        assert!(meta.last_modified <= after, "mtime in future");
        assert_eq!(meta.size, 5);
    }

    #[tokio::test]
    async fn object_store_handle_exposes_store_and_key() {
        let (_dir, p) = provider();
        let (_store, path) = p
            .object_store_handle("seg/x.parquet")
            .expect("handle for valid uri");
        assert_eq!(path.to_string(), "seg/x.parquet");
    }

    #[test]
    fn new_records_root_and_creates_it() {
        let dir = TempDir::new().expect("tempdir");
        let root = dir.path().join("nested/created/here");
        let p = LocalFsStorageProvider::new(&root).expect("provider creates root");
        assert_eq!(p.root(), &root);
        assert!(root.is_dir());
    }

    #[test]
    fn translate_maps_generic_to_transient_exhausted() {
        // `object_store` retries transient failures internally per its
        // RetryConfig; a `Generic` reaching `translate` is post-retry,
        // so it maps to `TransientExhausted`.
        let boxed: Box<dyn Error + Send + Sync> = "boom".into();
        let e = ObjError::Generic {
            store: "test",
            source: boxed,
        };
        let mapped = translate("data/x.sf.parquet", e);
        assert!(
            matches!(mapped, StorageError::TransientExhausted { .. }),
            "expected TransientExhausted, got {mapped:?}"
        );
    }

    #[test]
    fn translate_maps_unhandled_variant_to_permanent() {
        // A variant with no dedicated arm (e.g. `NotImplemented`)
        // falls through to the catch-all `Permanent`.
        let e = ObjError::NotImplemented {
            operation: "put_opts(Update)".into(),
            implementer: "LocalFileSystem".into(),
        };
        let mapped = translate("data/x.sf.parquet", e);
        match mapped {
            StorageError::Permanent { uri, .. } => assert_eq!(uri, "data/x.sf.parquet"),
            other => panic!("expected Permanent, got {other:?}"),
        }
    }

    #[test]
    fn translate_maps_already_exists_and_precondition_to_precondition_failed() {
        let already = ObjError::AlreadyExists {
            path: "p".into(),
            source: "exists".into(),
        };
        assert!(matches!(
            translate("uri", already),
            StorageError::PreconditionFailed { .. }
        ));
        let precond = ObjError::Precondition {
            path: "p".into(),
            source: "stale".into(),
        };
        assert!(matches!(
            translate("uri", precond),
            StorageError::PreconditionFailed { .. }
        ));
    }

    #[test]
    fn translate_maps_not_found() {
        let nf = ObjError::NotFound {
            path: "p".into(),
            source: "missing".into(),
        };
        assert!(matches!(
            translate("uri", nf),
            StorageError::NotFound { .. }
        ));
    }

    #[tokio::test]
    async fn invalid_path_surfaces_permanent_error() {
        // A NUL byte is illegal in an `object_store::path::Path`, so
        // `Self::path` fails before any I/O — surfacing the `path()`
        // error arm as `Permanent`.
        let (_dir, p) = provider();
        let bad_uri = "data/seg\0bad.sf.parquet";
        let err = p.head(bad_uri).await.expect_err("illegal path must fail");
        match err {
            StorageError::Permanent { uri, .. } => assert_eq!(uri, bad_uri),
            other => panic!("expected Permanent, got {other:?}"),
        }
        // The same rejection happens on the write paths.
        let err = p
            .put_atomic(bad_uri, Bytes::from_static(b"x"))
            .await
            .expect_err("illegal path must fail on put");
        assert!(matches!(err, StorageError::Permanent { .. }));
    }

    #[tokio::test]
    async fn object_store_handle_returns_none_for_invalid_path() {
        // `object_store_handle` swallows the path-parse error and
        // returns `None` (the `?`-via-`.ok()?` arm).
        let (_dir, p) = provider();
        assert!(p.object_store_handle("data/bad\0path").is_none());
    }
}
