// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Reader-side tombstone bitmap cache.
//!
//! Every superfile-touching query (FTS, vector, SQL) consults this
//! cache before returning per-superfile hits. The cache holds at
//! most one [`RoaringBitmap`] per superfile, keyed by
//! `superfile_id`; the bitmap's set bits are the local doc-ids
//! whose rows have been tombstoned.
//!
//! ## What lives here
//!
//! - [`cache::SidecarCache`] — the DashMap-backed cache, plus the
//!   sync-callable `bitmap_for` accessor the query paths use.
//! - [`cache::TombstoneSeqView`] — the manifest-derived freshness
//!   authority the cache validates entries against.
//! - [`cache::SidecarCacheError`] — typed errors surfaced when a
//!   refresh fails.
//!
//! [`RoaringBitmap`]: roaring::RoaringBitmap

pub mod cache;

pub use cache::{SidecarCache, SidecarCacheError, TombstoneSeqView};
