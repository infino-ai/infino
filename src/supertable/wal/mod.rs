// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Write-ahead-log primitives for the update / delete pipelines.
//!
//! ## What lives here
//!
//! Durability + serialization layer:
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
//! Coordination layer (built on the durability layer):
//!
//! - [`lease`] — advisory cooperative ownership: `try_acquire` /
//!   `try_heartbeat` / `try_release` + a `spawn_heartbeat`
//!   background task with stuck-worker detection.
//! - [`pipeline`] — the append-phase + tombstone-phase
//!   orchestrators that drive a WAL through its state machine
//!   (Intent → Appended → Complete for UPDATE; Intent →
//!   Complete for DELETE).
//! - [`recovery`] — on-demand sweep that lists WALs at
//!   `wal/mutations/*.json`, takes the lease, and drives each
//!   non-`Complete` WAL through the rest of its pipeline.
//! - [`gc`] — sweep over `wal/mutations/*` reaping `Complete`
//!   state docs past the wal-grace window + orphan `.arrow`
//!   sidecars past the sidecar-grace window.
//! - [`tombstones_admin`] — compaction-facing `seal` +
//!   `live_rows` helpers built on the sidecar codec; provides
//!   the freeze-the-sources surface a tombstone-aware compactor
//!   needs.
//!
//! ## On-disk layout
//!
//! State-document objects live at
//! `wal/mutations/<wal_id_hex>.json`. Sidecar Arrow-IPC payloads
//! live at `wal/mutations/<wal_id_hex>.arrow`. Tombstone bitmaps
//! live one-per-superfile at `superfiles/<superfile_id>.tombstones`
//! (not under `wal/`).

pub mod gc;
pub mod lease;
pub mod persistence;
pub mod pipeline;
pub mod recovery;
pub mod state_doc;
pub mod tombstones_admin;
pub mod tombstones_codec;

#[cfg(test)]
mod recovery_sweep_tests;

pub use persistence::{Etag, WalStore, WalStoreError};
pub use state_doc::{
    Lease, OpKind, SealRecord, TombstoneEntry, TombstoneOutcome, WalId, WalState, WalStateDoc,
};
pub use tombstones_codec::{SidecarCodecError, TombstonesSidecar, decode_sidecar, encode_sidecar};
