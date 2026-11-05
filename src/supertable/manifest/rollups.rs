// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A part's rollups in an object of their own: the per-column bounds and
//! null counts folded over the part's superfiles, which the list-level
//! pruner reads without loading the part.
//!
//! The list is rewritten whole at every commit, and a writer that loses a
//! commit race reads it whole again before it retries. A part's rollups
//! grow with the table's columns, so on a wide table they were nearly all
//! of the list, and both costs grew with parts times columns: a commit
//! rewrote every part's rollups to add one part's superfiles, and a loser
//! re-read them all. Out here, a part's rollups are written once, when the
//! part is, and a handle reads them once and keeps them by content hash:
//! the list carries their address, and a refresh reads only the rollups of
//! parts it has not seen.
//!
//! Rollups no larger than [`INLINE_ROLLUPS_MAX_BYTES`] stay inline in the
//! list, as every list before major 4 carries them: on a narrow table they
//! cost the list less than an object and a request of their own would.
//!
//! Layout, little-endian:
//!
//! ```text
//! "IROL" | u32 version | 16 bytes part id
//! the rollups as one Arrow IPC stream, the list's inline form
//! ```
//!
//! The part id is the part's own random id, so a rebuild never writes an
//! object byte-identical to an earlier one: its name is its content hash,
//! a PUT of a name that exists is skipped, and the earlier object may
//! already be queued for deletion.

use std::collections::HashMap;

use thiserror::Error;
use uuid::Uuid;

use crate::supertable::{
    manifest::{
        encoding::{decode_scalar_stats, encode_scalar_stats},
        list::ScalarStatsAgg,
        part::{ContentHash, PartId},
    },
    schema::FieldId,
};

/// Storage directory of the rollups objects, beside the parts.
pub const MANIFEST_ROLLUPS_DIR: &str = "manifest-rollups";

/// The most bytes a part's encoded rollups may hold and still stay inline in
/// the list. A rollup takes a few hundred bytes a column, so this keeps a
/// table of up to several dozen columns as it was, and moves a wider one's
/// out.
pub const INLINE_ROLLUPS_MAX_BYTES: usize = 16 * 1024;

/// The first four bytes of a rollups object.
const MAGIC: &[u8; 4] = b"IROL";
/// The layout version this engine writes.
const VERSION: u32 = 1;
/// Width of a little-endian `u32`.
const U32_BYTES: usize = 4;
/// The part id on the wire.
const ID_BYTES: usize = 16;
/// Magic, the version and the part id.
const HEADER_BYTES: usize = MAGIC.len() + U32_BYTES + ID_BYTES;

/// Where a rollups object with content hash `hash` lives.
pub fn rollups_uri(hash: &ContentHash) -> String {
    format!("{MANIFEST_ROLLUPS_DIR}/roll-{}.bin", hash.to_hex())
}

/// A rollups object's address, as the list entry names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RollupsRef {
    pub uri: String,
    pub content_hash: ContentHash,
    pub size_bytes: u64,
}

/// Where a part's rollups live.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum RollupsAt {
    /// In the list entry itself: rollups no larger than
    /// [`INLINE_ROLLUPS_MAX_BYTES`], and every part of a list before major 4.
    #[default]
    Inline,
    /// In their own object, which this handle read into the entry.
    Stored(RollupsRef),
    /// In their own object, which this handle could not read. The entry
    /// carries no rollups, so every prune keeps the part; a commit carries
    /// the reference over as it is and never rewrites the part, since the
    /// rollups it would fold from what it sees would leave out the part's
    /// earlier superfiles.
    Unread(RollupsRef),
}

impl RollupsAt {
    /// The object the list names, when the rollups live in one.
    pub fn reference(&self) -> Option<&RollupsRef> {
        match self {
            Self::Inline => None,
            Self::Stored(reference) | Self::Unread(reference) => Some(reference),
        }
    }

    /// Whether this handle could not read the part's rollups.
    pub fn is_unread(&self) -> bool {
        matches!(self, Self::Unread(_))
    }
}

/// Why a rollups object could not be read.
#[derive(Debug, Clone, Error)]
pub enum RollupsError {
    #[error("rollups object truncated: {had} bytes, its header needs {HEADER_BYTES}")]
    Truncated { had: usize },
    #[error("rollups object does not start with its magic")]
    BadMagic,
    #[error("rollups object version {0} is newer than this engine reads")]
    Version(u32),
    #[error("rollups object written for part {written_for}, read for part {part}")]
    NotItsPart { written_for: Uuid, part: Uuid },
    #[error("rollups object: {0}")]
    Decode(String),
}

/// Encode `rollups` for part `part_id`, or `None` when they are small enough
/// to stay inline in the list ([`INLINE_ROLLUPS_MAX_BYTES`]).
pub fn encode(part_id: PartId, rollups: &HashMap<FieldId, ScalarStatsAgg>) -> Option<Vec<u8>> {
    let ipc = encode_scalar_stats(rollups);
    if ipc.len() <= INLINE_ROLLUPS_MAX_BYTES {
        return None;
    }
    let mut bytes = Vec::with_capacity(HEADER_BYTES + ipc.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(part_id.0.as_bytes());
    bytes.extend_from_slice(&ipc);
    Some(bytes)
}

/// The rollups `bytes` hold, checked to be part `part_id`'s.
pub fn parse(
    bytes: &[u8],
    part_id: PartId,
) -> Result<HashMap<FieldId, ScalarStatsAgg>, RollupsError> {
    if bytes.len() < HEADER_BYTES {
        return Err(RollupsError::Truncated { had: bytes.len() });
    }
    let (magic, rest) = bytes.split_at(MAGIC.len());
    if magic != MAGIC {
        return Err(RollupsError::BadMagic);
    }
    let (version, rest) = rest.split_at(U32_BYTES);
    let version = u32::from_le_bytes(version.try_into().map_err(|_| RollupsError::BadMagic)?);
    if version != VERSION {
        return Err(RollupsError::Version(version));
    }
    let (id, ipc) = rest.split_at(ID_BYTES);
    let written_for = Uuid::from_slice(id).map_err(|e| RollupsError::Decode(e.to_string()))?;
    if written_for != part_id.0 {
        return Err(RollupsError::NotItsPart {
            written_for,
            part: part_id.0,
        });
    }
    decode_scalar_stats(ipc, None).map_err(|e| RollupsError::Decode(e.to_string()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array};

    use super::*;

    /// Columns enough that their rollups pass the inline cap.
    const WIDE_COLUMNS: u32 = 400;
    /// A few columns, whose rollups stay inline.
    const NARROW_COLUMNS: u32 = 3;
    /// Where the test's field ids start.
    const FIRST_FIELD: u32 = 1;

    /// One bound as a one-row array.
    fn bound(value: i64) -> ArrayRef {
        Arc::new(Int64Array::from(vec![value]))
    }

    /// Rollups of `columns` integer columns, each its own bounds.
    fn rollups(columns: u32) -> HashMap<FieldId, ScalarStatsAgg> {
        (FIRST_FIELD..FIRST_FIELD + columns)
            .map(|id| {
                (
                    FieldId(id),
                    ScalarStatsAgg {
                        min: bound(i64::from(id)),
                        max: bound(i64::from(id) * 2),
                        null_count: Some(u64::from(id)),
                        sum: None,
                        hll: None,
                        value_counts: None,
                    },
                )
            })
            .collect()
    }

    /// A wide part's rollups round-trip through their object, by column.
    #[test]
    fn a_wide_part_s_rollups_round_trip_through_their_object() {
        let part = PartId::new_v4();
        let written = rollups(WIDE_COLUMNS);
        let bytes = encode(part, &written).expect("wide rollups go to an object");
        let read = parse(&bytes, part).expect("parse");
        assert_eq!(read.len(), written.len());
        for (id, agg) in &written {
            let back = &read[id];
            assert_eq!(back.min.as_ref(), agg.min.as_ref(), "{id:?}");
            assert_eq!(back.max.as_ref(), agg.max.as_ref(), "{id:?}");
            assert_eq!(back.null_count, agg.null_count, "{id:?}");
        }
    }

    /// A narrow part's rollups stay inline in the list.
    #[test]
    fn a_narrow_part_s_rollups_stay_inline() {
        assert!(encode(PartId::new_v4(), &rollups(NARROW_COLUMNS)).is_none());
    }

    /// The same rollups written for two parts are two objects, and each is
    /// refused when read for the other part.
    #[test]
    fn an_object_names_its_part_and_is_refused_for_another() {
        let written = rollups(WIDE_COLUMNS);
        let (first, second) = (PartId::new_v4(), PartId::new_v4());
        let first_bytes = encode(first, &written).expect("object");
        let second_bytes = encode(second, &written).expect("object");
        assert_ne!(
            ContentHash::of(&first_bytes),
            ContentHash::of(&second_bytes)
        );
        assert!(matches!(
            parse(&first_bytes, second),
            Err(RollupsError::NotItsPart { .. })
        ));
    }

    /// A short object, a foreign one and one of a newer layout are refused.
    #[test]
    fn a_malformed_object_is_refused() {
        let part = PartId::new_v4();
        let bytes = encode(part, &rollups(WIDE_COLUMNS)).expect("object");
        assert!(matches!(
            parse(&bytes[..HEADER_BYTES - 1], part),
            Err(RollupsError::Truncated { .. })
        ));
        let mut foreign = bytes.clone();
        foreign[0] = b'X';
        assert!(matches!(parse(&foreign, part), Err(RollupsError::BadMagic)));
        let mut newer = bytes;
        newer[MAGIC.len()..MAGIC.len() + U32_BYTES].copy_from_slice(&(VERSION + 1).to_le_bytes());
        assert!(matches!(parse(&newer, part), Err(RollupsError::Version(_))));
    }
}
