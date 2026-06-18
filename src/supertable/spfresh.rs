// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! MVCC SPFresh maintenance for the hidden global vector cell index.
//!
//! The user table stays time-ordered and immutable. The hidden index is a
//! derived, cell-ordered acceleration layer maintained with SPFresh/LIRE-style
//! logical updates expressed as append/MVCC physical swaps:
//!
//!   1. Assign incoming vectors to nearest manifest centroids with zero GETs.
//!   2. For each touched cell only: append one delta superfile (no GETs).
//!   3. Compaction merges multiple small superfiles per cell toward one packed
//!      base via the existing `compaction/mod.rs` path.
//!   4. Locally refresh touched cell centroids.
//!
//! This is not in-place mutation of user Parquet. It is generation-based
//! base/delta maintenance over immutable cell superfiles.

use std::collections::HashMap;

use crate::supertable::manifest::ClusterCentroids;


/// Append-only count bookkeeping for touched cells.
///
/// Bumps each touched cell's indexed-doc count so routing
/// (`score_clusters_into`) never skips a populated cell as empty.
/// Sq8 centroid codes are left untouched - they are bootstrapped once
/// and rebalanced by compaction/SPFresh split-merge, never recomputed
/// from an fp32 running mean per commit.
pub fn apply_cell_count_updates(
    base: &ClusterCentroids,
    count_updates: &HashMap<u32, u32>,
) -> ClusterCentroids {
    let mut updated = base.clone();
    for (&cell, &count) in count_updates {
        if let Some(slot) = updated.counts.get_mut(cell as usize) {
            *slot = count;
        }
    }
    updated
}
