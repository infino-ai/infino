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

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_schema::Schema;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::InfinoError;
use crate::storage::{StorageError, StorageProvider};

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
        let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut out, schema)
            .map_err(|e| InfinoError::Backend(format!("schema ipc write: {e}")))?;
        writer
            .finish()
            .map_err(|e| InfinoError::Backend(format!("schema ipc finish: {e}")))?;
    }
    Ok(out)
}

/// Reconstruct a schema from Arrow-IPC bytes written by [`schema_to_ipc`].
pub(crate) fn schema_from_ipc(bytes: &[u8]) -> Result<Arc<Schema>, InfinoError> {
    let reader = arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None)
        .map_err(|e| InfinoError::Backend(format!("schema ipc read: {e}")))?;
    Ok(reader.schema())
}
