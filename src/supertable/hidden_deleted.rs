// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Consolidated deleted-user-`_id` set for the hidden vector-index table.
//!
//! User deletes tombstone only the user table; hidden cell superfiles keep
//! deleted rows physically present until drain/compaction removes them. The
//! hidden manifest fast payload carries this encoded set inline, so vector
//! search consults resident manifest bytes and never performs a deleted-set
//! GET.

use std::sync::Arc;

use crate::supertable::manifest::ManifestSnapshot;

/// Magic prefix on a packed deleted-user-`_id` set.
const DELETED_IDS_MAGIC: &[u8; 4] = b"HDEL";

/// Wire-format version for [`DELETED_IDS_MAGIC`] blobs.
const DELETED_IDS_VERSION: u8 = 1;

/// Header: magic (4) + version (1) + count (4).
const DELETED_IDS_HEADER_LEN: usize = 4 + 1 + 4;

/// Bytes per serialized `_id` (a little-endian `i128`).
const DELETED_ID_LEN: usize = 16;

/// Whether ids are in their canonical set representation: strictly ascending
/// (therefore sorted and deduplicated).
fn is_canonical_deleted_id_set(ids: &[i128]) -> bool {
    ids.windows(2).all(|pair| pair[0] < pair[1])
}

/// Serialize the consolidated deleted user-`_id` set. Strictly ascending
/// on-disk order is a wire invariant — consumers `binary_search` the decoded
/// set, so an unsorted blob silently resurrects deleted rows. The normal
/// caller already passes canonical ids; borrow that slice without copying.
/// An unsorted or duplicate-containing future caller is canonicalized here.
pub(crate) fn encode_deleted_ids(ids: &[i128]) -> Vec<u8> {
    let canonical_ids;
    let ids = if is_canonical_deleted_id_set(ids) {
        ids
    } else {
        canonical_ids = {
            let mut canonical = ids.to_vec();
            canonical.sort_unstable();
            canonical.dedup();
            canonical
        };
        canonical_ids.as_slice()
    };
    let mut out = Vec::with_capacity(DELETED_IDS_HEADER_LEN + ids.len() * DELETED_ID_LEN);
    out.extend_from_slice(DELETED_IDS_MAGIC);
    out.push(DELETED_IDS_VERSION);
    out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum HiddenDeletedError {
    #[error("deleted-id set truncated")]
    Truncated,
    #[error("deleted-id set bad magic")]
    BadMagic,
    #[error("deleted-id set unsupported version {0}")]
    UnsupportedVersion(u8),
    #[error("deleted-id set not strictly ascending on the wire")]
    NonCanonical,
}

/// Parse a deleted-`_id` set written by [`encode_deleted_ids`].
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
    // Consumers `binary_search` this set. Reject unsorted/duplicate wire
    // order in release too — a corrupt blob would otherwise silently
    // resurrect deleted rows via false-negative searches.
    if !is_canonical_deleted_id_set(&ids) {
        return Err(HiddenDeletedError::NonCanonical);
    }
    Ok(ids)
}

/// Decode the hidden index's resident deleted user-`_id` set from the
/// manifest. Returns an empty set when none is stamped. There is deliberately
/// no storage fallback here: the two-blob contract requires this state to ride
/// in the hidden manifest fast payload.
pub(crate) fn deleted_user_ids(
    manifest: &ManifestSnapshot,
) -> Result<Arc<Vec<i128>>, HiddenDeletedError> {
    let Some(bytes) = manifest.deleted_user_ids_inline() else {
        return Ok(Arc::new(Vec::new()));
    };
    Ok(Arc::new(decode_deleted_ids(bytes)?))
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
        assert!(
            decode_deleted_ids(&encode_deleted_ids(&[]))
                .expect("empty")
                .is_empty()
        );
    }

    /// An unsorted caller must still produce a sorted wire blob — consumers
    /// `binary_search` the decoded set, so order is a correctness invariant,
    /// not a convention.
    #[test]
    fn unsorted_input_encodes_sorted() {
        let ids: Vec<i128> = vec![42, -1, i128::MAX, 0, 42];
        let decoded = decode_deleted_ids(&encode_deleted_ids(&ids)).expect("decode");
        assert_eq!(decoded, vec![-1, 0, 42, i128::MAX]);
        assert!(decoded.binary_search(&-1).is_ok());
        assert!(decoded.binary_search(&i128::MAX).is_ok());
    }

    #[test]
    fn decode_rejects_unsorted_wire_bytes() {
        let mut bytes = encode_deleted_ids(&[1, 2, 3]);
        let (left, right) = bytes[DELETED_IDS_HEADER_LEN..].split_at_mut(DELETED_ID_LEN);
        left.swap_with_slice(&mut right[..DELETED_ID_LEN]);
        assert!(matches!(
            decode_deleted_ids(&bytes),
            Err(HiddenDeletedError::NonCanonical)
        ));
    }
}
