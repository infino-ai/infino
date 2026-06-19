// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The catalog body — a `name → table-record` map persisted as one
//! JSON object on the catalog root storage, mutated under optimistic
//! concurrency control.
//!
//! The catalog mirrors the supertable manifest's commit discipline
//! (read the current object + its ETag → modify → conditional PUT →
//! retry on conflict), giving atomic, last-writer-*loses* updates and
//! cross-process visibility on shared object storage. It is a single
//! small object rather than the manifest's pointer-plus-immutable-body
//! split: the catalog is tiny list-level metadata, so the body
//! indirection (which exists to cache large immutable manifest lists)
//! buys nothing here.

use std::{collections::BTreeMap, io::Cursor, sync::Arc};

use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::Schema;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::{
    InfinoError,
    storage::{StorageError, StorageProvider},
};

/// Object key (relative to the catalog root storage) holding the catalog.
pub(crate) const CATALOG_PATH: &str = "_catalog/current";

/// Bound on OCC retries before a contended commit gives up.
const MAX_CATALOG_RETRIES: u32 = 16;

/// One vector index's declaration, as recorded in the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VectorEntry {
    pub(crate) column: String,
    pub(crate) dim: usize,
    pub(crate) n_cent: usize,
    /// `"cosine"` / `"l2sq"` / `"negdot"` — the metric's lowercased name,
    /// matching the manifest's encoding so `open`'s options-hash check
    /// stays in lockstep.
    pub(crate) metric: String,
}

/// One table's catalog record: where its data lives plus the schema +
/// index declarations needed to reopen it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TableEntry {
    /// Table subtree relative to the catalog root (the table name).
    pub(crate) location: String,
    /// Arrow-IPC bytes of the user schema (no `_id` column).
    pub(crate) schema_ipc: Vec<u8>,
    /// FTS-indexed column names.
    pub(crate) fts: Vec<String>,
    /// Vector-indexed columns.
    pub(crate) vectors: Vec<VectorEntry>,
    /// Creation time, seconds since the Unix epoch.
    pub(crate) created_at_unix: u64,
}

/// The catalog body: the table map plus a monotonically increasing id
/// (bumped on every successful commit, for observability).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct CatalogBody {
    pub(crate) catalog_id: u64,
    pub(crate) tables: BTreeMap<String, TableEntry>,
}

/// Read the current catalog body + its ETag. A missing catalog object
/// (fresh root) reads as an empty body with no ETag.
pub(crate) async fn read_catalog(
    storage: &dyn StorageProvider,
) -> Result<(CatalogBody, Option<String>), InfinoError> {
    match storage.get(CATALOG_PATH).await {
        Ok((bytes, meta)) => {
            let body: CatalogBody = serde_json::from_slice(&bytes)
                .map_err(|e| InfinoError::Backend(format!("corrupt catalog: {e}")))?;
            Ok((body, meta.etag))
        }
        Err(StorageError::NotFound { .. }) => Ok((CatalogBody::default(), None)),
        Err(e) => Err(InfinoError::from(e)),
    }
}

/// Apply `mutate` to the current catalog and publish it with an OCC
/// conditional PUT, retrying on a concurrent conflict. `mutate` sees the
/// freshest body each attempt; if it rejects the change (e.g. a name
/// collision → `AlreadyExists`), that error is returned without retrying.
pub(crate) async fn commit_catalog<F>(
    storage: &dyn StorageProvider,
    mut mutate: F,
) -> Result<(), InfinoError>
where
    F: FnMut(&mut CatalogBody) -> Result<(), InfinoError>,
{
    for _ in 0..MAX_CATALOG_RETRIES {
        let (mut body, etag) = read_catalog(storage).await?;
        mutate(&mut body)?;
        body.catalog_id += 1;
        let bytes = Bytes::from(
            serde_json::to_vec(&body)
                .map_err(|e| InfinoError::Backend(format!("encode catalog: {e}")))?,
        );
        let put = match etag {
            Some(prev) => storage.put_if_match(CATALOG_PATH, bytes, Some(&prev)).await,
            None => storage.put_atomic(CATALOG_PATH, bytes).await,
        };
        match put {
            Ok(_) => return Ok(()),
            // A concurrent writer published first — re-read and retry.
            Err(StorageError::PreconditionFailed { .. }) => continue,
            Err(e) => return Err(InfinoError::from(e)),
        }
    }
    Err(InfinoError::Backend(
        "catalog commit exceeded its retry budget under contention".into(),
    ))
}

/// Serialize a user schema to Arrow-IPC bytes (schema-only stream).
pub(crate) fn schema_to_ipc(schema: &Schema) -> Result<Vec<u8>, InfinoError> {
    let mut out = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut out, schema)
            .map_err(|e| InfinoError::Backend(format!("schema ipc write: {e}")))?;
        writer
            .finish()
            .map_err(|e| InfinoError::Backend(format!("schema ipc finish: {e}")))?;
    }
    Ok(out)
}

/// Reconstruct a schema from Arrow-IPC bytes written by [`schema_to_ipc`].
pub(crate) fn schema_from_ipc(bytes: &[u8]) -> Result<Arc<Schema>, InfinoError> {
    let reader = StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|e| InfinoError::Backend(format!("schema ipc read: {e}")))?;
    Ok(reader.schema())
}

#[cfg(test)]
mod tests {
    use std::{error::Error, ops::Range, sync::Mutex, time::SystemTime};

    use arrow_schema::{DataType, Field};
    use async_trait::async_trait;
    use object_store::MultipartUpload;
    use tempfile::TempDir;

    use super::*;
    use crate::storage::{LocalFsStorageProvider, ObjectMeta};

    fn local(dir: &TempDir) -> Arc<dyn StorageProvider> {
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"))
    }

    fn sample_schema() -> Schema {
        Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("n", DataType::Int64, true),
        ])
    }

    fn sample_table_entry() -> TableEntry {
        TableEntry {
            location: "docs".into(),
            schema_ipc: schema_to_ipc(&sample_schema()).expect("ipc"),
            fts: vec!["title".into()],
            vectors: vec![VectorEntry {
                column: "emb".into(),
                dim: 8,
                n_cent: 4,
                metric: "cosine".into(),
            }],
            created_at_unix: 0,
        }
    }

    // ---- read_catalog --------------------------------------------------

    #[tokio::test]
    async fn read_catalog_missing_returns_empty_default() {
        let dir = TempDir::new().expect("tempdir");
        let s = local(&dir);
        let (body, etag) = read_catalog(s.as_ref()).await.expect("read");
        assert_eq!(body.catalog_id, 0);
        assert!(body.tables.is_empty());
        assert!(etag.is_none(), "no object yet ⇒ no etag");
    }

    #[tokio::test]
    async fn read_catalog_corrupt_bytes_errors() {
        let dir = TempDir::new().expect("tempdir");
        let s = local(&dir);
        s.put_atomic(CATALOG_PATH, Bytes::from_static(b"not json"))
            .await
            .expect("seed garbage");
        let err = read_catalog(s.as_ref()).await.expect_err("corrupt catalog");
        assert!(
            matches!(err, InfinoError::Backend(_)),
            "corrupt JSON ⇒ Backend error, got {err:?}",
        );
    }

    // ---- commit_catalog (real OCC over LocalFs) ------------------------

    #[tokio::test]
    async fn commit_then_read_round_trips_table_entry() {
        let dir = TempDir::new().expect("tempdir");
        let s = local(&dir);
        commit_catalog(s.as_ref(), |body| {
            body.tables.insert("docs".into(), sample_table_entry());
            Ok(())
        })
        .await
        .expect("commit");

        let (body, etag) = read_catalog(s.as_ref()).await.expect("read");
        assert_eq!(body.catalog_id, 1, "first commit bumps the id to 1");
        assert!(etag.is_some(), "published object carries an etag");
        let entry = body.tables.get("docs").expect("table present");
        assert_eq!(entry.location, "docs");
        assert_eq!(entry.fts, vec!["title".to_string()]);
        assert_eq!(entry.vectors.len(), 1);
        let schema = schema_from_ipc(&entry.schema_ipc).expect("decode schema");
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "title");
    }

    #[tokio::test]
    async fn commit_catalog_increments_catalog_id_per_commit() {
        let dir = TempDir::new().expect("tempdir");
        let s = local(&dir);
        for _ in 0..3 {
            commit_catalog(s.as_ref(), |_| Ok(()))
                .await
                .expect("commit");
        }
        let (body, _) = read_catalog(s.as_ref()).await.expect("read");
        assert_eq!(body.catalog_id, 3);
    }

    #[tokio::test]
    async fn commit_catalog_mutate_rejection_does_not_write() {
        let dir = TempDir::new().expect("tempdir");
        let s = local(&dir);
        let err = commit_catalog(s.as_ref(), |_| {
            Err(InfinoError::AlreadyExists("dup".into()))
        })
        .await
        .expect_err("mutate rejects the change");
        assert!(
            matches!(err, InfinoError::AlreadyExists(_)),
            "mutate's own error surfaces unchanged, got {err:?}",
        );
        // Nothing was published — the catalog object is still absent.
        let (body, etag) = read_catalog(s.as_ref()).await.expect("read");
        assert_eq!(body.catalog_id, 0);
        assert!(etag.is_none());
    }

    // ---- schema IPC round-trip -----------------------------------------

    #[test]
    fn schema_ipc_round_trips() {
        let schema = sample_schema();
        let bytes = schema_to_ipc(&schema).expect("to ipc");
        let back = schema_from_ipc(&bytes).expect("from ipc");
        assert_eq!(back.fields().len(), schema.fields().len());
        assert_eq!(back.field(0).name(), "title");
        assert_eq!(back.field(1).data_type(), &DataType::Int64);
    }

    #[test]
    fn schema_from_ipc_rejects_garbage() {
        let err = schema_from_ipc(b"not arrow ipc").expect_err("garbage bytes");
        assert!(matches!(err, InfinoError::Backend(_)));
    }

    // ---- commit_catalog OCC behaviour (mock injecting conflicts) -------

    /// In-memory single-object store whose conditional writes can be made
    /// to conflict on demand, to drive `commit_catalog`'s OCC retry loop.
    #[derive(Debug, Default)]
    struct MockState {
        object: Option<(Bytes, String)>,
        next_etag: u64,
        /// Number of upcoming writes to reject with `PreconditionFailed`
        /// before letting one land.
        put_fails_remaining: u32,
        /// Reject every write — used to exhaust the retry budget.
        always_fail_put: bool,
        put_calls: u32,
        get_calls: u32,
    }

    #[derive(Debug, Default)]
    struct OccMockStorage {
        state: Mutex<MockState>,
    }

    impl OccMockStorage {
        fn new() -> Self {
            Self::default()
        }

        /// Pre-seed the stored object so reads parse and commits take the
        /// `put_if_match` (etag-present) branch.
        fn seed(&self, bytes: Bytes) {
            let mut st = self.state.lock().expect("lock");
            st.next_etag += 1;
            let etag = format!("etag-{}", st.next_etag);
            st.object = Some((bytes, etag));
        }

        fn put_calls(&self) -> u32 {
            self.state.lock().expect("lock").put_calls
        }

        fn get_calls(&self) -> u32 {
            self.state.lock().expect("lock").get_calls
        }

        /// Shared write path for both `put_atomic` and `put_if_match`:
        /// honours the configured conflict injection, otherwise stores
        /// the bytes under a fresh etag.
        fn try_put(&self, bytes: Bytes) -> Result<Option<String>, StorageError> {
            let mut st = self.state.lock().expect("lock");
            st.put_calls += 1;
            if st.always_fail_put || st.put_fails_remaining > 0 {
                st.put_fails_remaining = st.put_fails_remaining.saturating_sub(1);
                return Err(StorageError::PreconditionFailed {
                    uri: CATALOG_PATH.into(),
                });
            }
            st.next_etag += 1;
            let etag = format!("etag-{}", st.next_etag);
            st.object = Some((bytes, etag.clone()));
            Ok(Some(etag))
        }
    }

    fn mock_unimplemented(uri: &str) -> StorageError {
        let boxed: Box<dyn Error + Send + Sync> = "unimplemented for mock".into();
        StorageError::Permanent {
            uri: uri.into(),
            source: boxed,
        }
    }

    #[async_trait]
    impl StorageProvider for OccMockStorage {
        async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
            let st = self.state.lock().expect("lock");
            match &st.object {
                Some((b, etag)) => Ok(ObjectMeta {
                    size: b.len() as u64,
                    etag: Some(etag.clone()),
                    last_modified: SystemTime::UNIX_EPOCH,
                }),
                None => Err(StorageError::NotFound { uri: uri.into() }),
            }
        }

        async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
            let mut st = self.state.lock().expect("lock");
            st.get_calls += 1;
            match &st.object {
                Some((b, etag)) => Ok((
                    b.clone(),
                    ObjectMeta {
                        size: b.len() as u64,
                        etag: Some(etag.clone()),
                        last_modified: SystemTime::UNIX_EPOCH,
                    },
                )),
                None => Err(StorageError::NotFound { uri: uri.into() }),
            }
        }

        async fn get_range(&self, uri: &str, _range: Range<u64>) -> Result<Bytes, StorageError> {
            Err(mock_unimplemented(uri))
        }

        async fn put_atomic(
            &self,
            _uri: &str,
            bytes: Bytes,
        ) -> Result<Option<String>, StorageError> {
            self.try_put(bytes)
        }

        async fn put_if_match(
            &self,
            _uri: &str,
            bytes: Bytes,
            _expected_etag: Option<&str>,
        ) -> Result<Option<String>, StorageError> {
            self.try_put(bytes)
        }

        async fn put_multipart(&self, uri: &str) -> Result<Box<dyn MultipartUpload>, StorageError> {
            Err(mock_unimplemented(uri))
        }

        async fn delete(&self, _uri: &str) -> Result<(), StorageError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn commit_catalog_retries_on_precondition_then_succeeds() {
        let mock = Arc::new(OccMockStorage::new());
        // Seed a valid empty catalog so reads parse and the etag-present
        // `put_if_match` path is exercised.
        mock.seed(Bytes::from(
            serde_json::to_vec(&CatalogBody::default()).expect("encode"),
        ));
        mock.state.lock().expect("lock").put_fails_remaining = 2;

        let s: Arc<dyn StorageProvider> = mock.clone();
        commit_catalog(s.as_ref(), |body| {
            body.tables.insert("docs".into(), sample_table_entry());
            Ok(())
        })
        .await
        .expect("commit eventually lands");

        assert_eq!(mock.put_calls(), 3, "two conflicts then one success");
        assert!(
            mock.get_calls() >= 3,
            "each attempt re-reads the freshest body",
        );
        let (body, _) = read_catalog(s.as_ref()).await.expect("read");
        assert!(body.tables.contains_key("docs"), "the change landed");
    }

    #[tokio::test]
    async fn commit_catalog_exhausts_retry_budget_under_contention() {
        let mock = Arc::new(OccMockStorage::new());
        mock.seed(Bytes::from(
            serde_json::to_vec(&CatalogBody::default()).expect("encode"),
        ));
        mock.state.lock().expect("lock").always_fail_put = true;

        let s: Arc<dyn StorageProvider> = mock.clone();
        let err = commit_catalog(s.as_ref(), |_| Ok(()))
            .await
            .expect_err("never lands under perpetual contention");
        assert!(
            matches!(err, InfinoError::Backend(_)),
            "budget exhaustion ⇒ Backend error, got {err:?}",
        );
        assert_eq!(
            mock.put_calls(),
            MAX_CATALOG_RETRIES,
            "tries exactly the retry budget",
        );
    }
}
