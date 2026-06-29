// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Pre-drain **incoming** routing — manifest-resident cluster centroids only.
//!
//! Undrained user superfiles already carry per-cluster Sq8+residual centroids in
//! their manifest `vector_summary`. That *is* the incoming buffer: no separate
//! in-memory copy, no register hook on commit, no tree work until drain/compact.
//!
//! - **Query:** a flat SIMD scan of those centroids
//!   ([`score_undrained_manifest_clusters`]) — no tree build, zero GETs — folded
//!   into the **same** radius-bounded admission as the persisted base tree (one
//!   shared `CoverageBound`), the incoming list being the tree's un-indexed tail.
//! - **Drain / compact:** batch tree build ([`super::insert::rebuild_tree_batch`]).

use std::{cmp::Ordering, sync::Arc};

use crate::{
    superfile::vector::distance::Metric,
    supertable::manifest::{Manifest, SuperfileEntry, list::OpannRouting},
};

use super::page::LeafRef;
use super::paged::ScoredLeaf;

/// User superfiles not yet drained into hidden cells (by arrival ordinal).
pub(crate) fn undrained_user_vector_entries(
    user_manifest: &Manifest,
    hidden_routing: Option<&OpannRouting>,
    column: &str,
) -> Vec<Arc<SuperfileEntry>> {
    let watermark = hidden_routing
        .map(|r| r.drained_max_arrival_ordinal)
        .unwrap_or(0);
    user_manifest
        .superfiles
        .iter()
        .filter(|e| {
            e.arrival_ordinal > watermark
                && e
                    .vector_summary
                    .get(column)
                    .is_some_and(|vs| vs.clusters.n_cent > 0)
        })
        .cloned()
        .collect()
}

/// Flat SIMD scan of the incoming buffer — **no tree build, no object-store GET.**
///
/// Undrained user superfiles carry per-cluster Sq8+residual centroids in their
/// manifest `vector_summary`. Score the query against each non-empty cluster's
/// centroid through the shared `Sq8ResidualKernel` ([`ClusterCentroids::score_one`])
/// and return them as [`ScoredLeaf`]s sorted by near edge `max(0, d − r)` — the
/// un-indexed incoming tail of the logical routing tree.
///
/// Admission is **not** done here. The caller folds this list into the *same*
/// radius-bounded [`CoverageBound`](super::paged::CoverageBound) as the
/// persisted-tree descent via
/// [`radius_bounded_admit`](super::paged::radius_bounded_admit), so the incoming
/// clusters share one bound (and one coverage floor) with the page leaves rather
/// than admitting against a separate floor. `survives` is applied there too,
/// uniformly with the tree leaves.
pub(crate) fn score_undrained_manifest_clusters(
    undrained: &[Arc<SuperfileEntry>],
    column: &str,
    metric: Metric,
    query: &[f32],
) -> Vec<ScoredLeaf> {
    let mut scored: Vec<ScoredLeaf> = Vec::new();
    for entry in undrained {
        let superfile_id = entry.superfile_id.as_u128();
        let Some(vs) = entry.vector_summary.get(column) else {
            continue;
        };
        let cc = &vs.clusters;
        let n_cent = cc.n_cent as usize;
        if n_cent == 0 || vs.cluster_offsets.len() != n_cent {
            continue;
        }
        for c in 0..n_cent {
            let count = cc.counts[c];
            if count == 0 {
                continue;
            }
            // SIMD Sq8+ε kernel — never a per-component scalar decode.
            let d = cc.score_one(metric, c, query);
            let r = cc.radii.get(c).copied().unwrap_or(0.0);
            scored.push(ScoredLeaf {
                near: (d - r).max(0.0),
                d,
                far: d + r,
                count,
                leaf: LeafRef {
                    superfile_id,
                    doc_off: vs.cluster_offsets[c],
                    count,
                    cluster_id: c as u32,
                },
            });
        }
    }
    scored.sort_by(|a, b| a.near.partial_cmp(&b.near).unwrap_or(Ordering::Equal));
    scored
}
