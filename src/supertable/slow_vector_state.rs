// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Slow-CAS vector-state blob: a table's superfile entries, verbatim, in one
//! content-addressed object.
//!
//! The manifest's fast CAS (pointer + list) churns on every commit of any
//! sort, and every rewritten part gets a fresh identity — so routing state
//! carried only by the parts is re-fetched whenever anything changes. This
//! blob gives the drain-owned slow state (per-superfile fp32 centroids +
//! offsets inside each [`SuperfileEntry`]) its own identity: drain / hidden
//! compaction publish it after membership settles, `ManifestSnapshot::update` clears
//! the reference on any membership change, and every other manifest
//! transition (deleted-id stamps, user commits) preserves it — so a loaded
//! consumer keeps its decoded entries in memory until the drainer actually
//! replaces them.
//!
//! Format: the blob IS a [`ManifestPart`] encoding (`part::encode` /
//! `part::decode`) with the nil part id — zero new entry serialization.
//! Same logical entries produce byte-identical blobs and therefore the same
//! content-addressed URI, so republishing unchanged state is a no-op PUT.
//! This module owns only the storage format and fetch/verify discipline; the
//! decoded entries live in the hydrated `ManifestSnapshot` (there is deliberately no
//! separate cache).

use std::{mem::size_of, sync::Arc};

use bytes::Bytes;
use uuid::Uuid;

use crate::{
    storage::{StorageError, StorageProvider},
    supertable::manifest::{
        SuperfileEntry,
        part::{self, ContentHash, ManifestPart, PartId},
    },
};

/// Versioned envelope used only while a drain checkpoint is active. Final
/// settled state keeps the legacy manifest-part-only encoding.
const CHECKPOINT_MAGIC: &[u8; 8] = b"INFSVS02";
const CHECKPOINT_HEADER_BYTES: usize = CHECKPOINT_MAGIC.len() + 3 * size_of::<u64>();
const CHECKPOINT_VISIBLE_LEN_OFF: usize = CHECKPOINT_MAGIC.len();
const CHECKPOINT_METADATA_LEN_OFF: usize = CHECKPOINT_VISIBLE_LEN_OFF + size_of::<u64>();
const CHECKPOINT_PENDING_LEN_OFF: usize = CHECKPOINT_METADATA_LEN_OFF + size_of::<u64>();

/// Object-storage prefix for content-addressed slow vector-state blobs,
/// relative to the owning table's storage provider (the hidden table's
/// provider is prefixed, so blobs land under the hidden subtree and request
/// metering attributes them to the hidden index automatically).
pub(crate) const STORAGE_PREFIX: &str = "slow-vector-state/";

/// Object-storage path for a content-addressed slow vector-state blob.
pub(crate) fn storage_path(hash: &ContentHash) -> String {
    format!("{STORAGE_PREFIX}state-{}.bin", hash.to_hex())
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum SlowVectorStateError {
    #[error("storage: {0}")]
    Storage(String),
    #[error("content hash mismatch")]
    HashMismatch,
    #[error("state parse: {0}")]
    Parse(String),
}

#[derive(Debug, Clone)]
pub(crate) struct PendingDrainState {
    /// Opaque drain-epoch metadata owned by `writer.rs`.
    pub metadata: Vec<u8>,
    /// Uploaded, not-yet-visible worker shard entries.
    pub entries: Vec<Arc<SuperfileEntry>>,
}

#[derive(Debug, Clone)]
pub(crate) struct SlowVectorState {
    /// Current manifest-visible hidden membership.
    pub entries: Vec<Arc<SuperfileEntry>>,
    /// In-progress drain completion state, absent after final publication.
    pub pending_drain: Option<PendingDrainState>,
}

/// Serialize `entries` verbatim through the manifest-part codec. The part id
/// is the nil UUID: the blob is not a real part, and a constant id keeps the
/// encoding deterministic (same entries ⇒ same bytes ⇒ same [`ContentHash`]).
pub(crate) fn encode_entries(entries: &[Arc<SuperfileEntry>]) -> Vec<u8> {
    let synthetic = ManifestPart {
        format_version: part::FORMAT_VERSION.into(),
        part_id: PartId(Uuid::nil()),
        superfiles: entries.to_vec(),
    };
    part::encode(&synthetic)
}

/// Decode a blob written by [`encode_entries`].
pub(crate) fn decode_entries(
    bytes: &[u8],
) -> Result<Vec<Arc<SuperfileEntry>>, SlowVectorStateError> {
    let decoded = part::decode(bytes).map_err(|e| SlowVectorStateError::Parse(e.to_string()))?;
    Ok(decoded.superfiles)
}

fn encode_checkpoint_state(
    entries: &[Arc<SuperfileEntry>],
    pending: &PendingDrainState,
) -> Vec<u8> {
    let visible = encode_entries(entries);
    let pending_entries = encode_entries(&pending.entries);
    let mut bytes = Vec::with_capacity(
        CHECKPOINT_HEADER_BYTES + visible.len() + pending.metadata.len() + pending_entries.len(),
    );
    bytes.extend_from_slice(CHECKPOINT_MAGIC);
    bytes.extend_from_slice(&(visible.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(pending.metadata.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(pending_entries.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&visible);
    bytes.extend_from_slice(&pending.metadata);
    bytes.extend_from_slice(&pending_entries);
    bytes
}

pub(crate) fn decode_state(bytes: &[u8]) -> Result<SlowVectorState, SlowVectorStateError> {
    if !bytes.starts_with(CHECKPOINT_MAGIC) {
        return Ok(SlowVectorState {
            entries: decode_entries(bytes)?,
            pending_drain: None,
        });
    }
    if bytes.len() < CHECKPOINT_HEADER_BYTES {
        return Err(SlowVectorStateError::Parse(
            "checkpoint header truncated".into(),
        ));
    }
    let visible_len = u64::from_le_bytes(
        bytes[CHECKPOINT_VISIBLE_LEN_OFF..CHECKPOINT_METADATA_LEN_OFF]
            .try_into()
            .expect("checkpoint visible length slice"),
    ) as usize;
    let metadata_len = u64::from_le_bytes(
        bytes[CHECKPOINT_METADATA_LEN_OFF..CHECKPOINT_PENDING_LEN_OFF]
            .try_into()
            .expect("checkpoint metadata length slice"),
    ) as usize;
    let pending_len = u64::from_le_bytes(
        bytes[CHECKPOINT_PENDING_LEN_OFF..CHECKPOINT_HEADER_BYTES]
            .try_into()
            .expect("checkpoint entries length slice"),
    ) as usize;
    let visible_end = CHECKPOINT_HEADER_BYTES
        .checked_add(visible_len)
        .ok_or_else(|| SlowVectorStateError::Parse("checkpoint length overflow".into()))?;
    let metadata_end = visible_end
        .checked_add(metadata_len)
        .ok_or_else(|| SlowVectorStateError::Parse("checkpoint length overflow".into()))?;
    let pending_end = metadata_end
        .checked_add(pending_len)
        .ok_or_else(|| SlowVectorStateError::Parse("checkpoint length overflow".into()))?;
    if pending_end != bytes.len() {
        return Err(SlowVectorStateError::Parse(
            "checkpoint envelope length mismatch".into(),
        ));
    }
    Ok(SlowVectorState {
        entries: decode_entries(&bytes[CHECKPOINT_HEADER_BYTES..visible_end])?,
        pending_drain: Some(PendingDrainState {
            metadata: bytes[visible_end..metadata_end].to_vec(),
            entries: decode_entries(&bytes[metadata_end..pending_end])?,
        }),
    })
}

async fn write_bytes(
    storage: &dyn StorageProvider,
    bytes: Vec<u8>,
) -> Result<(String, ContentHash), SlowVectorStateError> {
    let content_hash = ContentHash::of(&bytes);
    let uri = storage_path(&content_hash);
    match storage.put_atomic(&uri, Bytes::from(bytes)).await {
        Ok(_) | Err(StorageError::PreconditionFailed { .. }) => {}
        Err(error) => return Err(SlowVectorStateError::Storage(error.to_string())),
    }
    Ok((uri, content_hash))
}

/// Content-address and PUT the blob for `entries`. Idempotent: the URI is
/// hash-derived, so a raced identical PUT surfacing
/// [`StorageError::PreconditionFailed`] means the bytes are already durable.
/// Visibility is decided by the manifest-list ref stamp, not by this PUT.
pub(crate) async fn write_state(
    storage: &dyn StorageProvider,
    entries: &[Arc<SuperfileEntry>],
) -> Result<(String, ContentHash), SlowVectorStateError> {
    write_bytes(storage, encode_entries(entries)).await
}

/// Publish current visible membership plus an in-progress drain checkpoint in
/// the same content-addressed slow-CAS state referenced by the hidden manifest.
pub(crate) async fn write_state_with_pending_drain(
    storage: &dyn StorageProvider,
    entries: &[Arc<SuperfileEntry>],
    pending: &PendingDrainState,
) -> Result<(String, ContentHash), SlowVectorStateError> {
    write_bytes(storage, encode_checkpoint_state(entries, pending)).await
}

/// Fetch the blob at `uri`, verify its bytes hash to `expected`, and decode.
/// Callers fall back to manifest-part loading on any error — a bad blob must
/// never fail a table open or a query.
pub(crate) async fn load_state(
    storage: &dyn StorageProvider,
    uri: &str,
    expected: &ContentHash,
) -> Result<Vec<Arc<SuperfileEntry>>, SlowVectorStateError> {
    Ok(load_full_state(storage, uri, expected).await?.entries)
}

pub(crate) async fn load_full_state(
    storage: &dyn StorageProvider,
    uri: &str,
    expected: &ContentHash,
) -> Result<SlowVectorState, SlowVectorStateError> {
    let (bytes, _) = storage
        .get(uri)
        .await
        .map_err(|e| SlowVectorStateError::Storage(e.to_string()))?;
    if ContentHash::of(bytes.as_ref()) != *expected {
        return Err(SlowVectorStateError::HashMismatch);
    }
    decode_state(bytes.as_ref())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tempfile::tempdir;

    use super::*;
    use crate::{
        storage::LocalFsStorageProvider, superfile::vector::layout::VectorLayout,
        supertable::manifest::SuperfileUri,
    };

    /// Doc count for the first fixture entry; arbitrary but distinct from
    /// `SECOND_N_DOCS` so field mix-ups fail loudly.
    const FIRST_N_DOCS: u64 = 42;
    /// Doc count for the second fixture entry.
    const SECOND_N_DOCS: u64 = 7;

    fn entry(n_docs: u64, cell: u32) -> Arc<SuperfileEntry> {
        let id = Uuid::new_v4();
        Arc::new(SuperfileEntry {
            birth_version: 3,
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs,
            id_min: 10,
            id_max: 10 + n_docs.saturating_sub(1) as i128,
            scalar_stats: HashMap::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: cell.to_le_bytes().to_vec(),
            partition_hint: Some(cell),
            vector_layout: VectorLayout::Ivf,
            subsection_offsets: None,
        })
    }

    fn assert_entries_match(a: &SuperfileEntry, b: &SuperfileEntry) {
        assert_eq!(a.superfile_id, b.superfile_id);
        assert_eq!(a.uri, b.uri);
        assert_eq!(a.n_docs, b.n_docs);
        assert_eq!(a.id_min, b.id_min);
        assert_eq!(a.id_max, b.id_max);
        assert_eq!(a.partition_key, b.partition_key);
        assert_eq!(a.partition_hint, b.partition_hint);
        assert_eq!(a.birth_version, b.birth_version);
    }

    #[test]
    fn entries_roundtrip_and_deterministic() {
        let entries = vec![entry(FIRST_N_DOCS, 0), entry(SECOND_N_DOCS, 5)];
        let bytes = encode_entries(&entries);
        let decoded = decode_entries(&bytes).expect("decode");
        assert_eq!(decoded.len(), entries.len());
        for (d, e) in decoded.iter().zip(entries.iter()) {
            assert_entries_match(d, e);
        }
        // Same logical entries ⇒ same bytes ⇒ same content hash ⇒ same URI —
        // the property the content-addressed republish-is-a-no-op rides on.
        let again = encode_entries(&entries);
        assert_eq!(bytes, again);
        assert_eq!(
            storage_path(&ContentHash::of(&bytes)),
            storage_path(&ContentHash::of(&again))
        );
    }

    #[test]
    fn decode_garbage_is_parse_error() {
        let err = decode_entries(&[0u8; 16]).expect_err("garbage");
        assert!(matches!(err, SlowVectorStateError::Parse(_)), "{err:?}");
    }

    #[test]
    fn pending_drain_envelope_round_trips_visible_and_pending_entries() {
        let visible = vec![entry(FIRST_N_DOCS, 1)];
        let pending_entries = vec![entry(SECOND_N_DOCS, 2)];
        let pending = PendingDrainState {
            metadata: b"checkpoint metadata".to_vec(),
            entries: pending_entries.clone(),
        };
        let bytes = encode_checkpoint_state(&visible, &pending);
        let decoded = decode_state(&bytes).expect("decode checkpoint envelope");
        assert_eq!(decoded.entries.len(), 1);
        assert_entries_match(&decoded.entries[0], &visible[0]);
        let decoded_pending = decoded.pending_drain.expect("pending drain");
        assert_eq!(decoded_pending.metadata, pending.metadata);
        assert_eq!(decoded_pending.entries.len(), 1);
        assert_entries_match(&decoded_pending.entries[0], &pending_entries[0]);
    }

    #[tokio::test]
    async fn write_load_verifies_hash_and_is_idempotent() {
        let dir = tempdir().expect("tempdir");
        let storage = LocalFsStorageProvider::new(dir.path()).expect("provider");
        let entries = vec![entry(FIRST_N_DOCS, 1)];

        let (uri, hash) = write_state(&storage, &entries).await.expect("write");
        // Re-publishing identical content must succeed (PreconditionFailed
        // from the hash-derived URI is benign by construction).
        let (uri2, hash2) = write_state(&storage, &entries).await.expect("rewrite");
        assert_eq!(uri, uri2);
        assert_eq!(hash, hash2);

        let loaded = load_state(&storage, &uri, &hash).await.expect("load");
        assert_eq!(loaded.len(), 1);
        assert_entries_match(&loaded[0], &entries[0]);

        let wrong = ContentHash::of(b"wrong");
        let err = load_state(&storage, &uri, &wrong)
            .await
            .expect_err("hash mismatch");
        assert!(matches!(err, SlowVectorStateError::HashMismatch), "{err:?}");

        let missing = load_state(&storage, "slow-vector-state/absent.bin", &hash)
            .await
            .expect_err("missing object");
        assert!(
            matches!(missing, SlowVectorStateError::Storage(_)),
            "{missing:?}"
        );
    }
}
