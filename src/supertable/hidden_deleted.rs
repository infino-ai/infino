// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Consolidated deleted-user-`_id` set for the hidden vector-index table.
//!
//! User deletes tombstone only the user table; hidden cell superfiles keep
//! deleted rows physically present until drain/compaction removes them. Vector
//! search consults this content-addressed blob (loaded from object storage)
//! instead of per-cell tombstone sidecars — which are never populated on the
//! hidden index and would cost a wasted GET wave.

use bytes::Bytes;

use crate::{
    storage::StorageProvider,
    supertable::{
        manifest::{Manifest, part::ContentHash},
    },
};

/// Magic prefix on a packed deleted-user-`_id` blob.
const DELETED_IDS_MAGIC: &[u8; 4] = b"HDEL";

/// Wire-format version for [`DELETED_IDS_MAGIC`] blobs.
const DELETED_IDS_VERSION: u8 = 1;

/// Header: magic (4) + version (1) + count (4).
const DELETED_IDS_HEADER_LEN: usize = 4 + 1 + 4;

/// Bytes per serialized `_id` (a little-endian `i128`).
const DELETED_ID_LEN: usize = 16;

/// Object-storage path for a content-addressed deleted-`_id` blob.
pub(crate) fn storage_path(hash: &ContentHash) -> String {
    format!("hidden-deleted-ids/deleted-{}.bin", hash.to_hex())
}

/// Serialize the consolidated deleted user-`_id` set. The caller passes a
/// sorted, deduplicated slice so the on-disk order is canonical.
pub(crate) fn encode_deleted_ids(sorted_ids: &[i128]) -> Vec<u8> {
    let mut out = Vec::with_capacity(DELETED_IDS_HEADER_LEN + sorted_ids.len() * DELETED_ID_LEN);
    out.extend_from_slice(DELETED_IDS_MAGIC);
    out.push(DELETED_IDS_VERSION);
    out.extend_from_slice(&(sorted_ids.len() as u32).to_le_bytes());
    for id in sorted_ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum HiddenDeletedError {
    #[error("deleted-id blob truncated")]
    Truncated,
    #[error("deleted-id blob bad magic")]
    BadMagic,
    #[error("deleted-id blob unsupported version {0}")]
    UnsupportedVersion(u8),
    #[error("storage: {0}")]
    Storage(String),
    #[error("content hash mismatch")]
    HashMismatch,
}

/// Parse a deleted-`_id` blob written by [`encode_deleted_ids`].
pub(crate) fn decode_deleted_ids(bytes: &[u8]) -> Result<Vec<i128>, HiddenDeletedError> {
    if bytes.len() < DELETED_IDS_HEADER_LEN {
        return Err(HiddenDeletedError::Truncated);
    }
    if &bytes[0..4] != DELETED_IDS_MAGIC {
        return Err(HiddenDeletedError::BadMagic);
    }
    let version = bytes[4];
    if version != DELETED_IDS_VERSION {
        return Err(HiddenDeletedError::UnsupportedVersion(version));
    }
    let count = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) as usize;
    let body = &bytes[DELETED_IDS_HEADER_LEN..];
    if body.len() != count * DELETED_ID_LEN {
        return Err(HiddenDeletedError::Truncated);
    }
    let mut ids = Vec::with_capacity(count);
    for chunk in body.chunks_exact(DELETED_ID_LEN) {
        let mut buf = [0u8; DELETED_ID_LEN];
        buf.copy_from_slice(chunk);
        ids.push(i128::from_le_bytes(buf));
    }
    Ok(ids)
}

/// Load the hidden index's consolidated deleted user-`_id` set from the
/// manifest. Returns an empty vec when no blob is stamped (legacy manifests
/// or no deletes pending).
pub(crate) async fn load_deleted_user_ids(
    manifest: &Manifest,
    storage: &dyn StorageProvider,
) -> Result<Vec<i128>, HiddenDeletedError> {
    let Some((path, expected)) = manifest.deleted_user_ids_blob() else {
        return Ok(Vec::new());
    };
    let bytes = fetch_and_verify(storage, path, &expected).await?;
    decode_deleted_ids(&bytes)
}

async fn fetch_and_verify(
    storage: &dyn StorageProvider,
    path: &str,
    expected: &ContentHash,
) -> Result<Bytes, HiddenDeletedError> {
    let (bytes, _) = storage
        .get(path)
        .await
        .map_err(|e| HiddenDeletedError::Storage(e.to_string()))?;
    if ContentHash::of(bytes.as_ref()) != *expected {
        return Err(HiddenDeletedError::HashMismatch);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deleted_ids_encode_decode_roundtrip() {
        let ids: Vec<i128> = vec![i128::MIN, -1, 0, 1, 42, 1 << 100, i128::MAX];
        let bytes = encode_deleted_ids(&ids);
        assert_eq!(decode_deleted_ids(&bytes).expect("decode"), ids);
        assert!(decode_deleted_ids(&[]).is_err());
        assert!(decode_deleted_ids(&encode_deleted_ids(&[])).expect("empty").is_empty());
    }
}
