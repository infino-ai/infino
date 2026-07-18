// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Tiered superfile-bytes lookup.
//!
//! [`superfile_reader`] is the single accessor the query paths
//! (`bm25_search`, `vector_search`, `query_sql`) use to turn a
//! `SuperfileUri` into an `Arc<SuperfileReader>`. The policy:
//!
//!   1. **In-memory tier first.** If `store.reader(uri)`
//!      succeeds — i.e., this process's writer recently
//!      published the superfile and the bytes are still in
//!      `InMemoryReaderCache` — return that reader. Fast
//!      path; no syscalls.
//!   2. **Disk cache fallback.** Miss in the in-memory tier
//!      AND a `DiskCacheStore` is attached →
//!      `DiskCacheStore::reader(uri)` (`await`ed directly).
//!      The cache itself handles cold-fetch from object
//!      storage, pwrite to the local cache directory, and
//!      mmap.
//!   3. **No cache.** Miss in the in-memory tier and no
//!      cache attached → surface the in-memory tier's
//!      `ReaderCacheError::NotFound`. The in-process-only
//!      path; supports callers without storage attached.
//!
//! The accessor is `async`: the query paths
//! (`SupertableReader::vector_search` / `bm25_search` /
//! `query_sql`) are themselves async and run on the owning
//! tokio runtime, so the cold object-store fetch the cache
//! issues is driven by that runtime's reactor — no sync
//! bridge, no throwaway `current_thread` runtime, and
//! object-store retries fire correctly.

use std::{io, sync::Arc};

use crate::{
    storage::StorageProvider,
    superfile::{ReadError, SuperfileReader},
    supertable::{
        manifest::{SubsectionOffsets, SuperfileUri},
        reader_cache::{
            DiskCacheStore, ReaderCacheError, SuperfileReaderCache, disk::DiskCacheError,
        },
    },
};

/// Look up `uri`'s `SuperfileReader`, preferring the in-
/// memory tier and falling back to the disk cache when
/// configured. See the module-level docs for the precise
/// policy.
///
/// `offsets` is an optional pre-known layout hint
/// pulled from the manifest's [`SubsectionOffsets`]. When `Some`
/// the disk-cache cold-fetch path fires the parquet-footer,
/// vector subsection, and FTS subsection GETs **in parallel**
/// (1 RTT cold open) instead of doing the parquet footer first
/// and the subsection fetches second (2 RTTs). `None` falls back
/// to the 2-RTT path — same shape, slower.
pub async fn superfile_reader(
    store: &Arc<dyn SuperfileReaderCache>,
    disk_cache: Option<&Arc<DiskCacheStore>>,
    storage: Option<&Arc<dyn StorageProvider>>,
    uri: &SuperfileUri,
    offsets: Option<&SubsectionOffsets>,
    allow_background_fill: bool,
) -> Result<Arc<SuperfileReader>, ReaderCacheError> {
    // 1. In-memory tier.
    match store.reader(uri) {
        Ok(r) => return Ok(r),
        Err(ReaderCacheError::NotFound { .. }) => {
            // Fall through to the cache.
        }
        Err(other) => return Err(other),
    }

    // 2. Disk cache fallback (when attached).
    if let Some(cache) = disk_cache {
        match cache
            .reader_with_hints(uri, offsets, storage, allow_background_fill)
            .await
        {
            Ok(reader) => return Ok(reader),
            // Cache can't admit this superfile (e.g. it's larger than the
            // whole budget). Stream it directly via range GETs instead
            // of failing the query.
            Err(DiskCacheError::BudgetExceeded) => {
                return cache
                    .open_range_only(uri, offsets, storage)
                    .await
                    .map_err(cache_open_failed);
            }
            Err(e) => return Err(cache_open_failed(e)),
        }
    }

    // 3. Storage-only fallback. This covers reopened LocalFs/S3
    // handles configured with durable storage but no disk cache.
    // It is intentionally whole-object: callers who need bounded
    // memory attach `DiskCacheStore`, which uses lazy/range opens.
    if let Some(storage) = storage {
        let path = uri.storage_path();
        let (bytes, _) = storage
            .get(&path)
            .await
            .map_err(|e| ReaderCacheError::OpenFailed {
                source: ReadError::Io(io::Error::other(format!("storage fetch {path}: {e}"))),
            })?;
        let reader = SuperfileReader::open(bytes)
            .map_err(|source| ReaderCacheError::OpenFailed { source })?;
        return Ok(Arc::new(reader));
    }

    Err(ReaderCacheError::NotFound { uri: *uri })
}

fn cache_open_failed(e: DiskCacheError) -> ReaderCacheError {
    ReaderCacheError::OpenFailed {
        source: ReadError::Io(io::Error::other(format!("disk cache fetch: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use tempfile::TempDir;

    use super::*;
    use crate::{
        storage::LocalFsStorageProvider,
        superfile::{
            ReadError,
            builder::{BuilderOptions, SuperfileBuilder},
        },
        supertable::reader_cache::{InMemoryReaderCache, config::DiskCacheConfig},
        test_helpers::{decimal128_id_field, decimal128_ids},
    };

    /// `n_docs()` of the superfile every helper below builds — three
    /// scalar rows, no FTS / vector indexes.
    const N_DOCS: u64 = 3;

    /// A `disk_budget_bytes` smaller than any real superfile, so the
    /// disk cache rejects admission with `BudgetExceeded` and the
    /// caller is forced down the `open_range_only` fallback.
    const TINY_BUDGET_BYTES: u64 = 4;

    /// Build minimal valid superfile bytes (parquet body + KV metadata
    /// only — what `SuperfileReader::open` requires, no indexes).
    fn minimal_superfile_bytes() -> Bytes {
        let schema: Arc<Schema> = Arc::new(Schema::new(vec![
            decimal128_id_field("doc_id"),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![], None);
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let ids = decimal128_ids(vec![1u64, 2, 3]);
        let title = LargeStringArray::from(vec!["a", "b", "c"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(title)])
            .expect("build RecordBatch");
        b.add_batch(&batch, &[]).expect("add_batch");
        Bytes::from(b.finish().expect("finish builder"))
    }

    /// An empty in-memory tier — every lookup misses with `NotFound`,
    /// so the accessor falls through to whatever fallback is attached.
    fn empty_store() -> Arc<dyn SuperfileReaderCache> {
        Arc::new(InMemoryReaderCache::new())
    }

    /// A local-FS storage provider rooted at `dir`.
    fn local_storage(dir: &TempDir) -> Arc<dyn StorageProvider> {
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"))
    }

    /// Build a `DiskCacheStore` over `storage`, applying `mutate` to
    /// the default config (cache files live under `<dir>/cache`).
    fn disk_cache(
        dir: &TempDir,
        storage: &Arc<dyn StorageProvider>,
        mutate: impl FnOnce(&mut DiskCacheConfig),
    ) -> Arc<DiskCacheStore> {
        let mut cfg = DiskCacheConfig {
            cache_root: dir.path().join("cache"),
            mmap_cold_threshold_secs: 0,
            ..Default::default()
        };
        mutate(&mut cfg);
        DiskCacheStore::new_unpinned(Arc::clone(storage), cfg).expect("disk cache store")
    }

    /// Put `bytes` at the storage path the accessor cold-fetches from.
    async fn put_at_storage(storage: &Arc<dyn StorageProvider>, uri: &SuperfileUri, bytes: Bytes) {
        storage
            .put_atomic(&uri.storage_path(), bytes)
            .await
            .expect("put superfile bytes");
    }

    // ---- tier 1: in-memory hit -----------------------------------------

    #[tokio::test]
    async fn in_memory_hit_returns_reader_without_touching_fallbacks() {
        let store = empty_store();
        let uri = SuperfileUri::new_v4();
        store
            .insert(uri, minimal_superfile_bytes())
            .expect("insert into in-memory tier");

        // No disk cache, no storage attached: if the in-memory tier is
        // consulted first (it is), neither fallback is needed.
        let reader = superfile_reader(&store, None, None, &uri, None, true)
            .await
            .expect("in-memory hit");
        assert_eq!(reader.n_docs(), N_DOCS);
    }

    // ---- tier 1: non-NotFound error short-circuits ---------------------

    /// In-memory tier that always fails with a non-`NotFound` error.
    /// Used to prove the accessor surfaces such errors immediately
    /// rather than falling through to a fallback.
    #[derive(Debug)]
    struct AlwaysOpenFailedCache;

    impl SuperfileReaderCache for AlwaysOpenFailedCache {
        fn reader(&self, _uri: &SuperfileUri) -> Result<Arc<SuperfileReader>, ReaderCacheError> {
            Err(ReaderCacheError::OpenFailed {
                source: ReadError::Io(io::Error::other("in-memory tier boom")),
            })
        }
        fn insert(&self, _uri: SuperfileUri, _bytes: Bytes) -> Result<(), ReaderCacheError> {
            Ok(())
        }
        fn resident_bytes(&self) -> usize {
            0
        }
    }

    #[tokio::test]
    async fn in_memory_non_not_found_error_short_circuits_before_fallback() {
        let store: Arc<dyn SuperfileReaderCache> = Arc::new(AlwaysOpenFailedCache);
        let uri = SuperfileUri::new_v4();
        let dir = TempDir::new().expect("tempdir");
        let storage = local_storage(&dir);
        // A working fallback is attached; the in-memory error must win.
        put_at_storage(&storage, &uri, minimal_superfile_bytes()).await;

        let err = superfile_reader(&store, None, Some(&storage), &uri, None, true)
            .await
            .expect_err("in-memory error must propagate");
        assert!(
            matches!(err, ReaderCacheError::OpenFailed { .. }),
            "expected the in-memory OpenFailed to surface, got {err:?}",
        );
    }

    // ---- tier 2: disk cache fallback -----------------------------------

    #[tokio::test]
    async fn disk_cache_cold_fetch_on_in_memory_miss() {
        let dir = TempDir::new().expect("tempdir");
        let storage = local_storage(&dir);
        let uri = SuperfileUri::new_v4();
        put_at_storage(&storage, &uri, minimal_superfile_bytes()).await;
        let cache = disk_cache(&dir, &storage, |_| {});

        let reader = superfile_reader(&empty_store(), Some(&cache), None, &uri, None, true)
            .await
            .expect("disk cache cold fetch");
        assert_eq!(reader.n_docs(), N_DOCS);
    }

    #[tokio::test]
    async fn disk_cache_budget_exceeded_falls_back_to_range_only() {
        let dir = TempDir::new().expect("tempdir");
        let storage = local_storage(&dir);
        let uri = SuperfileUri::new_v4();
        put_at_storage(&storage, &uri, minimal_superfile_bytes()).await;
        // Budget too small to admit the superfile → BudgetExceeded →
        // the accessor streams it via range GETs instead of failing.
        let cache = disk_cache(&dir, &storage, |cfg| {
            cfg.disk_budget_bytes = TINY_BUDGET_BYTES;
        });

        let reader = superfile_reader(&empty_store(), Some(&cache), None, &uri, None, true)
            .await
            .expect("range-only fallback on budget exceeded");
        assert_eq!(reader.n_docs(), N_DOCS);
    }

    #[tokio::test]
    async fn disk_cache_open_failure_surfaces_as_open_failed() {
        let dir = TempDir::new().expect("tempdir");
        let storage = local_storage(&dir);
        let cache = disk_cache(&dir, &storage, |_| {});
        // Nothing put at storage: the cold fetch can't find the bytes.
        let uri = SuperfileUri::new_v4();

        let err = superfile_reader(&empty_store(), Some(&cache), None, &uri, None, true)
            .await
            .expect_err("missing storage object must error");
        assert!(
            matches!(err, ReaderCacheError::OpenFailed { .. }),
            "expected OpenFailed, got {err:?}",
        );
    }

    // ---- tier 3: storage-only fallback ---------------------------------

    #[tokio::test]
    async fn storage_only_fallback_opens_whole_object() {
        let dir = TempDir::new().expect("tempdir");
        let storage = local_storage(&dir);
        let uri = SuperfileUri::new_v4();
        put_at_storage(&storage, &uri, minimal_superfile_bytes()).await;

        // No disk cache, but durable storage attached: whole-object open.
        let reader = superfile_reader(&empty_store(), None, Some(&storage), &uri, None, true)
            .await
            .expect("storage-only fallback");
        assert_eq!(reader.n_docs(), N_DOCS);
    }

    #[tokio::test]
    async fn storage_only_fallback_missing_object_is_open_failed() {
        let dir = TempDir::new().expect("tempdir");
        let storage = local_storage(&dir);
        let uri = SuperfileUri::new_v4();
        // Nothing put at storage → the GET fails.

        let err = superfile_reader(&empty_store(), None, Some(&storage), &uri, None, true)
            .await
            .expect_err("missing object must error");
        assert!(
            matches!(err, ReaderCacheError::OpenFailed { .. }),
            "expected OpenFailed, got {err:?}",
        );
    }

    // ---- tier 3: no cache, no storage ----------------------------------

    #[tokio::test]
    async fn no_cache_no_storage_returns_not_found() {
        let uri = SuperfileUri::new_v4();
        let err = superfile_reader(&empty_store(), None, None, &uri, None, true)
            .await
            .expect_err("in-process-only miss must be NotFound");
        match err {
            ReaderCacheError::NotFound { uri: got } => assert_eq!(got, uri),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    // ---- cache_open_failed mapping -------------------------------------

    #[test]
    fn cache_open_failed_maps_to_open_failed_and_preserves_message() {
        let mapped = cache_open_failed(DiskCacheError::BudgetExceeded);
        match mapped {
            ReaderCacheError::OpenFailed { source } => {
                let msg = source.to_string();
                assert!(
                    msg.contains("disk cache fetch"),
                    "wrapped message should name the disk cache path, got {msg:?}",
                );
            }
            other => panic!("expected OpenFailed, got {other:?}"),
        }
    }
}
