// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use std::{path::PathBuf, sync::Arc};

use bytes::Bytes;

use super::{ObjectMeta, PrefixedStorageProvider, StorageError, StorageProvider, UsageMeter};
use crate::supertable::manifest::{SUPERFILE_DATA_DIR, term_index::STORAGE_PREFIX};

/// Routes superfile, slice and tier keys to a shared table's storage and every other key, term-index roots
/// included, to this table's own. Roots stay private because each side retires its own and a content-addressed
/// root can be identical on both. Deletes of shared keys are dropped; never gc a table on this provider.
#[derive(Debug)]
pub struct SplitStorageProvider {
    manifest: Arc<dyn StorageProvider>,
    objects: Arc<dyn StorageProvider>,
}

/// Where a table's attached replicas keep their manifests. While anything is here, the table's gc keeps every
/// superfile and term-index object and compaction skips, since replica objects sit in the table unlisted until
/// grafted.
pub const REPLICAS_DIR: &str = "_replicas/";

/// A table's marker that replicas may be attached, beside [`REPLICAS_DIR`] rather than in it: one
/// HEAD answers what a listing of the replicas would, for paths that must not list.
const ATTACHED_MARKER: &str = "_replicas.attached";

/// Records that a replica of the table on `storage` exists; after the replica's manifest.
pub(crate) async fn mark_attached(storage: &dyn StorageProvider) -> Result<(), StorageError> {
    storage
        .put_overwrite(ATTACHED_MARKER, Bytes::from_static(b"attached"))
        .await
}

/// Whether the table on `storage` may have replicas attached; an unreadable marker counts as yes.
pub(crate) async fn may_be_attached(storage: &dyn StorageProvider) -> bool {
    !matches!(
        storage.head(ATTACHED_MARKER).await,
        Err(StorageError::NotFound { .. })
    )
}

/// Clears the marker once no replica is attached. A replica that attaches meanwhile writes its
/// marker after its manifest, so the listing after the delete sees it and the marker is restored.
pub(crate) async fn settle_attached(storage: &dyn StorageProvider) -> Result<(), StorageError> {
    if !attached_replicas(storage).await?.is_empty() {
        return Ok(());
    }
    match storage.delete(ATTACHED_MARKER).await {
        Ok(()) | Err(StorageError::NotFound { .. }) => {}
        Err(e) => return Err(e),
    }
    if !attached_replicas(storage).await?.is_empty() {
        mark_attached(storage).await?;
    }
    Ok(())
}

/// Names of the replicas attached to the table on `storage`: those with an object left under
/// [`REPLICAS_DIR`]. Judged by objects, since a local directory outlives its last file.
pub(crate) async fn attached_replicas(
    storage: &dyn StorageProvider,
) -> Result<Vec<String>, StorageError> {
    let mut names: Vec<String> = storage
        .list_with_prefix(REPLICAS_DIR)
        .await?
        .iter()
        .filter_map(|key| {
            key.strip_prefix(REPLICAS_DIR)?
                .split('/')
                .next()
                .map(str::to_owned)
        })
        .collect();
    names.sort();
    names.dedup();
    Ok(names)
}

impl SplitStorageProvider {
    pub fn new(manifest: Arc<dyn StorageProvider>, objects: Arc<dyn StorageProvider>) -> Self {
        Self { manifest, objects }
    }

    /// Storage for replica `name` of the table on `table`: its manifest under
    /// [`REPLICAS_DIR`]`<name>/`, its objects in the table.
    pub fn for_replica(table: Arc<dyn StorageProvider>, name: &str) -> Self {
        let manifest =
            PrefixedStorageProvider::new(Arc::clone(&table), format!("{REPLICAS_DIR}{name}"));
        Self::new(Arc::new(manifest), table)
    }

    fn is_shared(uri: &str) -> bool {
        let data = uri
            .strip_prefix(SUPERFILE_DATA_DIR)
            .is_some_and(|rest| rest.starts_with('/'));
        let index = uri
            .strip_prefix(STORAGE_PREFIX)
            .is_some_and(|rest| !rest.starts_with("root-"));
        data || index
    }

    fn route(&self, uri: &str) -> &dyn StorageProvider {
        match Self::is_shared(uri) {
            true => self.objects.as_ref(),
            false => self.manifest.as_ref(),
        }
    }
}

#[async_trait::async_trait]
impl StorageProvider for SplitStorageProvider {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.route(uri).head(uri).await
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        self.route(uri).get(uri).await
    }

    async fn get_if_none_match(
        &self,
        uri: &str,
        etag: &str,
    ) -> Result<Option<(Bytes, ObjectMeta)>, StorageError> {
        self.route(uri).get_if_none_match(uri, etag).await
    }

    async fn get_range(
        &self,
        uri: &str,
        range: std::ops::Range<u64>,
    ) -> Result<Bytes, StorageError> {
        self.route(uri).get_range(uri, range).await
    }

    async fn tail(&self, uri: &str, len: u64) -> Result<(Bytes, u64), StorageError> {
        self.route(uri).tail(uri, len).await
    }

    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        self.route(uri).put_atomic(uri, bytes).await
    }

    async fn put_overwrite(&self, uri: &str, bytes: Bytes) -> Result<(), StorageError> {
        self.route(uri).put_overwrite(uri, bytes).await
    }

    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        self.route(uri)
            .put_if_match(uri, bytes, expected_etag)
            .await
    }

    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        self.route(uri).put_multipart(uri).await
    }

    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        match Self::is_shared(uri) {
            true => Ok(()),
            false => self.manifest.delete(uri).await,
        }
    }

    async fn delete_once(&self, uri: &str) -> Result<(), StorageError> {
        match Self::is_shared(uri) {
            true => Ok(()),
            false => self.manifest.delete_once(uri).await,
        }
    }

    async fn list_with_prefix(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        self.route(prefix).list_with_prefix(prefix).await
    }

    async fn list_with_prefix_metadata(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
        self.route(prefix).list_with_prefix_metadata(prefix).await
    }

    async fn list_child_prefixes(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        self.route(prefix).list_child_prefixes(prefix).await
    }

    fn object_store_handle(
        &self,
        uri: &str,
    ) -> Option<(Arc<dyn object_store::ObjectStore>, object_store::path::Path)> {
        self.route(uri).object_store_handle(uri)
    }

    fn usage_meter(&self) -> Arc<UsageMeter> {
        self.objects.usage_meter()
    }

    fn local_path(&self, uri: &str) -> Option<PathBuf> {
        self.route(uri).local_path(uri)
    }
}

#[cfg(test)]
mod tests {
    use super::SplitStorageProvider;

    #[test]
    fn shares_superfiles_slices_and_tiers_but_not_roots_or_manifests() {
        for key in [
            "data/seg-1.sf.parquet",
            "term-index/slice-ab.bin",
            "term-index/tier-ab.bin",
        ] {
            assert!(SplitStorageProvider::is_shared(key), "{key}");
        }
        for key in [
            "term-index/root-ab.bin",
            "_supertable/current",
            "manifest/1",
            "manifest-parts/x",
            "database/x",
        ] {
            assert!(!SplitStorageProvider::is_shared(key), "{key}");
        }
    }
}
