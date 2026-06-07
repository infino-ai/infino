// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! WAL state-document shape.
//!
//! Serialized as JSON, one object per WAL entry, at
//! `wal/mutations/<wal_id_hex>.json`. Every transition through
//! the update / delete state machine writes a fresh version of
//! this document under the same path via `put_if_match` (etag
//! CAS). The shape is intentionally flat and forward-compatible:
//! unknown JSON keys are tolerated on the read path so future
//! additions don't break older readers.
//!
//! ## Why JSON
//!
//! Three considerations:
//!
//! - The state doc is touched many times per WAL (once per
//!   tombstone outcome, plus the INTENT → APPENDED → COMPLETE
//!   transitions). Human-readable bytes make on-disk forensics
//!   from `s3 cp` / `cat *.json` trivial.
//! - Volume is bounded by a per-mutation cap on the target-list
//!   size, so even worst-case state docs sit in the
//!   single-digit-MB range — well within what JSON tolerates.
//! - The payload (Arrow IPC `new_rows`, for UPDATE) goes in a
//!   separate `.arrow` sidecar — the JSON state doc never carries
//!   megabytes of row data through its CAS rewrites.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Schema version for the state-document JSON shape. Bumped
/// when a field is added or semantics changes. Old readers
/// reject anything > the version they were built against.
pub const SCHEMA_VERSION: u32 = 1;

/// Byte width of a 128-bit identifier (`i128`), big-endian on the
/// wire.
const ID_BYTE_LEN: usize = 16;

/// Hex-string length of a 128-bit identifier — two lowercase hex
/// digits per byte. This is the exact JSON-wire form, so changing it
/// is a wire-format change.
const ID_HEX_LEN: usize = ID_BYTE_LEN * 2;

/// Parse failure for the hex form of any 128-bit identifier
/// ([`WalId`], [`RowId`], [`SupertableHandleId`]). Carries enough context
/// to pinpoint the offending byte position.
#[derive(Debug, thiserror::Error)]
pub enum IdParseError {
    #[error("id hex must be exactly 32 chars; got {len}")]
    WrongLength { len: usize },
    #[error("id hex contains non-hex at byte position {pos}: {snippet:?}")]
    InvalidHex { pos: usize, snippet: String },
}

/// Defines a Snowflake-shaped 128-bit newtype with the shared
/// hex encoding, `Serialize` / `Deserialize` (as hex string),
/// and parse helpers. All three id types in this module share
/// the same encoding so they sort time-ordered when listed and
/// can be mixed freely on the JSON wire.
macro_rules! define_id_type {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(pub i128);

        impl $name {
            /// 32-char zero-padded lowercase hex of `self.0.to_be_bytes()`.
            /// Stable across releases — JSON payload format, so any
            /// change here is a wire-format change.
            pub fn to_hex(self) -> String {
                let bytes = self.0.to_be_bytes();
                let mut out = String::with_capacity(ID_HEX_LEN);
                for b in bytes {
                    // `{:02x}` always emits two lowercase hex digits.
                    use std::fmt::Write as _;
                    let _ = write!(out, "{b:02x}");
                }
                out
            }

            /// Inverse of `to_hex`. Rejects strings whose length isn't
            /// exactly 32 or that contain non-hex characters.
            pub fn from_hex(s: &str) -> Result<Self, IdParseError> {
                if s.len() != ID_HEX_LEN {
                    return Err(IdParseError::WrongLength { len: s.len() });
                }
                let mut bytes = [0u8; ID_BYTE_LEN];
                for (i, byte) in bytes.iter_mut().enumerate() {
                    *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| {
                        IdParseError::InvalidHex {
                            pos: 2 * i,
                            snippet: s[2 * i..2 * i + 2].to_string(),
                        }
                    })?;
                }
                Ok(Self(i128::from_be_bytes(bytes)))
            }
        }

        impl Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(&self.to_hex())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let s = String::deserialize(d)?;
                Self::from_hex(&s).map_err(serde::de::Error::custom)
            }
        }
    };
}

define_id_type! {
    /// Snowflake-shaped 128-bit identifier for one WAL entry.
    ///
    /// Same layout as the supertable's `_id` column
    /// (`utils::idgen::IdGenerator`): 64-bit ms timestamp + 40-bit
    /// random worker_id + 24-bit sequence, packed into an `i128`.
    /// Stored as 32-char zero-padded lowercase hex of the value's
    /// big-endian bytes so file listings sort time-ordered with a
    /// trailing tie-break on worker_id + sequence.
    ///
    /// The hex form is the on-disk filename
    /// (`wal/mutations/<hex>.json`) and is also the wire form in
    /// the state doc's `wal_id` field.
    WalId
}

define_id_type! {
    /// Snowflake-shaped 128-bit value of the supertable's `_id`
    /// column — a row's stable global identifier, minted by
    /// `IdGenerator::next_id` at append time. Distinct from
    /// [`WalId`] (which identifies a WAL state doc) and from
    /// the local `u32` doc-id (a row's position within one
    /// superfile, used by tombstones / FTS / vector indices).
    RowId
}

define_id_type! {
    /// Snowflake-shaped 128-bit id of one [`crate::supertable::Supertable`]
    /// handle, minted at handle construction. Stamped onto WAL
    /// leases as `owner` so the recovery sweep can tell whether
    /// a peer or a stale local handle holds a given WAL.
    SupertableHandleId
}

/// Kind of mutation a WAL is driving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum OpKind {
    Update,
    Delete,
}

/// State-machine position of a WAL. UPDATE walks
/// INTENT → APPENDED → COMPLETE; DELETE walks INTENT → COMPLETE
/// directly (no append phase).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum WalState {
    Intent,
    Appended,
    Complete,
}

/// Cooperative ownership of one WAL by the process currently
/// driving it. The lease is **advisory** — the load-bearing
/// safety primitive is the etag CAS on the state doc; the lease
/// only prevents two processes from doing duplicate work
/// (still-correct but wasteful). Any safety argument of the form
/// "this is safe because the lease holds" is wrong by
/// construction — safety has to surface at the CAS layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lease {
    /// Handle id of the process currently driving this WAL.
    /// Set on take; cleared on COMPLETE or preemption.
    pub owner: SupertableHandleId,
    pub acquired_at: DateTime<Utc>,
    /// Heartbeat-extended; recovery treats expiry as the cue to
    /// preempt.
    pub expires_at: DateTime<Utc>,
}

/// One row's tombstone-phase progress. Each `target_id` from the
/// resolved predicate starts at `Pending`, flips to `Tombstoned`
/// (sidecar bit landed) or `NotFound` (no superfile claims this
/// id) once step 2 sees it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TombstoneEntry {
    /// `_id` of the row being tombstoned.
    pub target_id: RowId,
    pub outcome: TombstoneOutcome,
    /// Audit field: which superfile this tombstone landed in.
    /// `None` until outcome flips to `Tombstoned`; stays `None`
    /// for `Pending` and `NotFound`.
    #[serde(default)]
    pub tombstoned_in_superfile: Option<Uuid>,
}

/// Terminal-or-pending state of a single target's tombstone work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum TombstoneOutcome {
    /// The tombstone phase hasn't reached this target yet.
    Pending,
    /// The target's local doc_id is set in the resolved
    /// superfile's tombstone sidecar.
    Tombstoned,
    /// No superfile in the current manifest claims this id —
    /// either a peer beat us to the tombstone, or compaction
    /// removed the row's superfile in a way that lost the id.
    /// Surfaced to the caller in `OperationOutcome.n_not_found`.
    NotFound,
}

/// Identifies the compaction job that sealed a tombstone
/// sidecar. Mirrors the on-disk `SealRecord` in the sidecar
/// codec — kept here too because abandoned-compaction recovery
/// needs to correlate a sealed sidecar back to the compaction
/// intent that owns it (via `compactions/<id>.intent` lookup).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealRecord {
    pub compaction_id: Uuid,
    pub sealed_at: DateTime<Utc>,
}

/// One contiguous span of pre-minted `_id`s reserved at the
/// caller's `update()` entry point, before any work depending on
/// the ids begins. Spans flatten in order; the flattened length
/// equals `new_row_count`.
///
/// Multiple spans appear when `IdGenerator::reserve_range`
/// crossed a ferroid sequence ms boundary mid-call: the
/// generator's 24-bit sequence per ms could be exhausted mid-
/// reservation under contention, so it returns the contiguous
/// prefix and starts a fresh span in the next ms. Single-span
/// is the typical case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdSpan {
    pub first: RowId,
    pub last: RowId,
}

impl IdSpan {
    /// Inclusive length of the span (`last - first + 1`).
    /// Panics if `last < first` — span construction is engine-
    /// internal so an inverted span is a bug, not a user error.
    pub fn len(&self) -> u64 {
        debug_assert!(self.first.0 <= self.last.0, "inverted span");
        (self.last.0 - self.first.0 + 1) as u64
    }

    /// True iff the span carries zero ids. Spans should never
    /// be empty in practice — `reserve_range` only emits
    /// non-empty spans — but the trait bound on collections
    /// wants this defined.
    pub fn is_empty(&self) -> bool {
        self.first.0 > self.last.0
    }
}

/// Top-level on-disk shape of one WAL entry's state document.
/// JSON, written via `put_if_match` (etag CAS) on every
/// transition; bounded in size by a per-mutation cap on the
/// target-list length.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WalStateDoc {
    /// Echoes the filename's `<wal_id_hex>` — included in the
    /// payload so a leaked or copied state doc identifies
    /// itself without depending on its on-disk path.
    pub wal_id: WalId,

    /// Bumped on every wire-format change; readers reject
    /// anything they don't recognize before parsing further.
    pub schema_version: u32,

    pub op_kind: OpKind,
    pub state: WalState,
    pub created_at: DateTime<Utc>,

    /// `None` if no process currently owns this WAL (e.g. fresh
    /// WAL between create and the first heartbeat, or after
    /// preemption clears the field). Recovery treats absent +
    /// expired equivalently.
    #[serde(default)]
    pub lease: Option<Lease>,

    /// Human-readable rendering of the caller's predicate, for
    /// forensics + audit only. Recovery never re-evaluates this
    /// string — `target_ids` below is the durable resolution.
    pub predicate_repr: String,

    /// The resolved set the predicate matched at call-time
    /// (against the supertable's current manifest snapshot).
    /// Recovery replays from this list directly — the predicate
    /// itself is never re-evaluated, so the resolution is
    /// stable across whatever manifest churn happens between
    /// the original call and a later recovery. Length is
    /// bounded by the per-mutation target cap.
    pub target_ids: Vec<RowId>,

    // -------- UPDATE-only fields below --------
    // All `None` / empty for DELETE; OpKind discriminates.
    /// Row count of the IPC sidecar's payload. Same as
    /// `target_ids.len()` for UPDATE (1:1 cardinality at Step
    /// 0a).
    #[serde(default)]
    pub new_row_count: Option<u32>,

    /// blake3 hex of the IPC sidecar's bytes. Append-phase replay
    /// verifies this before consuming the sidecar to detect
    /// corruption.
    #[serde(default)]
    pub new_row_content_hash: Option<String>,

    /// UUID v4 minted at the caller's `update()` entry; the
    /// dedicated superfile the UPDATE's new rows will land in.
    /// Stored before any I/O so recovery produces bit-identical
    /// superfile bytes.
    ///
    /// **Why one dedicated superfile per UPDATE, not merged
    /// with the commit's regular `append()` rows?** Replay
    /// safety. The new superfile must be reconstructable from
    /// durable state alone (this WAL doc + the `.arrow`
    /// sidecar), because a crash anywhere between `update()`
    /// and the manifest CAS leaves recovery responsible for
    /// finishing the work. The pre-minted UUID combined with
    /// `minted_id_spans` and `new_row_content_hash` pins every
    /// byte of the superfile so a recovery process's re-PUT
    /// lands at the same path with identical content. Regular
    /// `append()` rows live in an in-memory buffer that's never
    /// persisted before commit; on crash the caller retries.
    /// Merging the two would force one of: (a) every append
    /// goes through the etag-CAS state machine (latency hit on
    /// the common path), or (b) updates lose crash-replay
    /// determinism. Both are worse than the current cost — one
    /// extra superfile object per UPDATE — at any realistic
    /// update:append ratio.
    #[serde(default)]
    pub preallocated_superfile_id: Option<Uuid>,

    /// Ordered list of contiguous `_id` spans pre-minted at
    /// call-time. Flattens to exactly `new_row_count` ids; row
    /// `i` of the IPC sidecar gets `flatten(spans)[i]`. Pinning
    /// the ids in the WAL state doc before any superfile write
    /// is what makes the append-phase replay-safe — a recovery
    /// process re-running the append writes bit-identical bytes
    /// because the row→id mapping is fixed by this field, not
    /// by what `IdGenerator` happens to produce at replay time.
    #[serde(default)]
    pub minted_id_spans: Vec<IdSpan>,

    // -------- Both UPDATE and DELETE below --------
    /// Per-target tombstone progress. One entry per `target_id`,
    /// in the same order. The tombstone phase walks this list
    /// and flips each entry's outcome through CAS-PUTs to the
    /// state doc — every flip is durable before the next
    /// target's work starts.
    pub tombstone_progress: Vec<TombstoneEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_wal_id() -> WalId {
        // Pick something with non-trivial bytes to catch
        // byte-order regressions in to_hex/from_hex.
        WalId(0x0011_2233_4455_6677_8899_AABB_CCDD_EEFF_i128)
    }

    // ---- WalId hex encoding ---------------------------------------------

    #[test]
    fn wal_id_hex_round_trips() {
        let w = sample_wal_id();
        let h = w.to_hex();
        assert_eq!(h.len(), 32);
        // Strict lowercase — the format is wire-stable and any
        // case change would break peers that compare strings.
        assert_eq!(h, h.to_lowercase());
        assert_eq!(WalId::from_hex(&h).expect("parse"), w);
    }

    #[test]
    fn wal_id_hex_zero_pads_high_zero_bytes() {
        // Small ids must still produce a full 32-char string,
        // otherwise the lexicographic sort guarantee on filenames
        // breaks at version 0.
        let h = WalId(1).to_hex();
        assert_eq!(h, "00000000000000000000000000000001");
    }

    #[test]
    fn wal_id_hex_preserves_be_byte_order() {
        // 0x01..00 as big-endian bytes should produce the
        // matching hex left-to-right.
        let w = WalId(i128::from_be_bytes([
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
            0x0F, 0x00,
        ]));
        assert_eq!(w.to_hex(), "0102030405060708090a0b0c0d0e0f00");
    }

    #[test]
    fn wal_id_from_hex_rejects_wrong_length() {
        let too_short = "0000";
        assert!(matches!(
            WalId::from_hex(too_short),
            Err(IdParseError::WrongLength { len: 4 })
        ));
        let too_long = "0".repeat(33);
        assert!(matches!(
            WalId::from_hex(&too_long),
            Err(IdParseError::WrongLength { len: 33 })
        ));
    }

    #[test]
    fn wal_id_from_hex_rejects_non_hex() {
        let mut s = String::from("0").repeat(30);
        s.push('z');
        s.push('z');
        assert!(matches!(
            WalId::from_hex(&s),
            Err(IdParseError::InvalidHex { pos: 30, .. })
        ));
    }

    // ---- WalStateDoc serde round-trip -----------------------------------

    fn sample_state_doc() -> WalStateDoc {
        WalStateDoc {
            wal_id: sample_wal_id(),
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Update,
            state: WalState::Intent,
            created_at: "2026-05-30T10:00:00Z".parse().expect("ts"),
            lease: Some(Lease {
                owner: SupertableHandleId(42),
                acquired_at: "2026-05-30T10:00:01Z".parse().expect("ts"),
                expires_at: "2026-05-30T10:01:01Z".parse().expect("ts"),
            }),
            predicate_repr: "status = 'pending'".into(),
            target_ids: vec![RowId(1), RowId(2), RowId(3)],
            new_row_count: Some(3),
            new_row_content_hash: Some("deadbeef".into()),
            preallocated_superfile_id: Some(Uuid::nil()),
            minted_id_spans: vec![
                IdSpan {
                    first: RowId(100),
                    last: RowId(101),
                },
                IdSpan {
                    first: RowId(200),
                    last: RowId(200),
                },
            ],
            tombstone_progress: vec![
                TombstoneEntry {
                    target_id: RowId(1),
                    outcome: TombstoneOutcome::Pending,
                    tombstoned_in_superfile: None,
                },
                TombstoneEntry {
                    target_id: RowId(2),
                    outcome: TombstoneOutcome::Pending,
                    tombstoned_in_superfile: None,
                },
                TombstoneEntry {
                    target_id: RowId(3),
                    outcome: TombstoneOutcome::Pending,
                    tombstoned_in_superfile: None,
                },
            ],
        }
    }

    #[test]
    fn state_doc_round_trips_update_through_json() {
        let original = sample_state_doc();
        let json = serde_json::to_string(&original).expect("encode");
        let decoded: WalStateDoc = serde_json::from_str(&json).expect("decode");
        assert_eq!(decoded, original);
    }

    #[test]
    fn state_doc_round_trips_delete_through_json() {
        // DELETE has none of the UPDATE-only fields. Verify the
        // `#[serde(default)]` annotations + `Option` shape keep
        // the JSON minimal AND round-trip cleanly.
        let original = WalStateDoc {
            op_kind: OpKind::Delete,
            new_row_count: None,
            new_row_content_hash: None,
            preallocated_superfile_id: None,
            minted_id_spans: Vec::new(),
            ..sample_state_doc()
        };
        let json = serde_json::to_string(&original).expect("encode");
        let decoded: WalStateDoc = serde_json::from_str(&json).expect("decode");
        assert_eq!(decoded, original);
    }

    #[test]
    fn state_doc_tolerates_unknown_fields_on_read() {
        // Forward-compat: a future code addition with a new
        // field (e.g. `priority`) must still parse with this
        // reader. The default serde derive ignores unknown
        // keys, but lock that behavior in with an explicit
        // test so someone doesn't accidentally add
        // `deny_unknown_fields`.
        let mut json: serde_json::Value = serde_json::to_value(sample_state_doc()).expect("encode");
        json.as_object_mut()
            .expect("object")
            .insert("priority".into(), serde_json::json!("high"));
        let serialized = serde_json::to_string(&json).expect("re-encode");
        let _: WalStateDoc = serde_json::from_str(&serialized).expect("decode w/ extra field");
    }

    #[test]
    fn state_doc_rejects_unknown_op_kind() {
        // Negative test on the OpKind enum — a future addition
        // of a new variant (e.g. "MERGE") must explicitly bump
        // `schema_version`; a current reader sees the new
        // variant and refuses to parse rather than silently
        // treating it as the default. (`#[serde(rename_all =
        // "UPPERCASE")]` produces exact-match arms — unknown
        // strings error.)
        let json = r#"{
            "wal_id": "00000000000000000000000000000001",
            "schema_version": 1,
            "op_kind": "MERGE",
            "state": "INTENT",
            "created_at": "2026-05-30T10:00:00Z",
            "predicate_repr": "x",
            "target_ids": [],
            "tombstone_progress": []
        }"#;
        let err = serde_json::from_str::<WalStateDoc>(json).expect_err("must fail");
        assert!(
            err.to_string().contains("MERGE")
                || err.to_string().contains("variant")
                || err.to_string().contains("op_kind"),
            "expected variant-mismatch error; got: {err}"
        );
    }

    // ---- IdSpan ---------------------------------------------------------

    #[test]
    fn id_span_len_inclusive() {
        let s = IdSpan {
            first: RowId(10),
            last: RowId(14),
        };
        assert_eq!(s.len(), 5);
        let single = IdSpan {
            first: RowId(7),
            last: RowId(7),
        };
        assert_eq!(single.len(), 1);
    }
}
