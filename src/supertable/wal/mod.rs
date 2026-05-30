//! Write-ahead-log primitives for the update / delete pipelines.
//!
//! ## What lives here
//!
//! - [`state_doc`] — the on-disk JSON shape of one WAL entry's
//!   state document, plus the `wal_id` ↔ filename encoding.
//! - [`persistence`] — the storage-level CAS primitives
//!   (`WalStore::create` / `read` / `update_with_etag`) that
//!   every higher-level WAL operation sits on. The whole crate's
//!   storage interaction for WAL entries goes through this one
//!   type so the CAS contract is enforced in exactly one place.
//! - [`tombstones_codec`] — hand-rolled byte framing for the
//!   per-superfile tombstone sidecar object (magic + version +
//!   optional `SealRecord` + `RoaringBitmap`).
//!
//! ## What does NOT live here
//!
//! Pipeline orchestration — append + tombstone state machines,
//! the recovery scan, leases, GC — is intentionally out of scope.
//! This module ships the durability + serialization layer only.
//! Nothing here knows what a `target_id` means or when a state
//! transition is legal; those rules belong to the pipeline layer
//! that sits on top.
//!
//! ## On-disk layout
//!
//! State-document objects live at
//! `wal/mutations/<wal_id_hex>.json`. Sidecar Arrow-IPC payloads
//! live at `wal/mutations/<wal_id_hex>.arrow`. Tombstone bitmaps
//! live one-per-superfile at `superfiles/<superfile_id>.tombstones`
//! (not under `wal/`).

pub mod persistence;
pub mod pipeline;
pub mod state_doc;
pub mod tombstones_codec;

pub use persistence::{Etag, WalStore, WalStoreError};
pub use state_doc::{
    Lease, OpKind, SealRecord, TombstoneEntry, TombstoneOutcome, WalId, WalState, WalStateDoc,
};
pub use tombstones_codec::{SidecarCodecError, TombstonesSidecar, decode_sidecar, encode_sidecar};
