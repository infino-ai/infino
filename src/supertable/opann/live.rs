// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Pre-drain **incoming** routing — manifest-resident cluster centroids only.
//!
//! Undrained user superfiles already carry per-cluster Sq8+residual centroids in
//! their manifest `vector_summary`. That *is* the incoming buffer: no separate
//! in-memory copy, no register hook on commit, no tree work until drain/compact.
//!
//! - **Query:** a flat SIMD scan of those centroids
//!   ([`admit_undrained_manifest_clusters`]) — no tree build, zero GETs — merged
//!   with the persisted base tree post-drain.
//! - **Drain / compact:** batch tree build ([`super::insert::rebuild_tree_batch`]).

use std::{cmp::Ordering, sync::Arc};

use crate::{
    superfile::vector::distance::Metric,
    supertable::manifest::{Manifest, SuperfileEntry, list::OpannRouting},
};

use super::page::LeafRef;

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
/// manifest `vector_summary`. Score the query against each surviving cluster's
/// centroid through the shared [`Sq8ResidualKernel`]
/// ([`ClusterCentroids::score_one`]) and admit **every** non-empty cluster.
/// Pre-drain user superfiles are IVF-fragmented; a doc-count coverage floor would
/// prune sibling clusters in the same file and starve rerank, while fetch still
/// coalesces to one range-GET per superfile. Returns each cluster's [`LeafRef`]
/// + its centroid distance for merge with the persisted-tree descent post-drain.
pub(crate) fn admit_undrained_manifest_clusters(
    undrained: &[Arc<SuperfileEntry>],
    column: &str,
    metric: Metric,
    query: &[f32],
    _floor: usize,
    survives: impl Fn(u128) -> bool,
) -> Vec<(LeafRef, f32)> {
    // One scored candidate per surviving, non-empty cluster.
    struct Cand {
        near: f32,
        d: f32,
        far: f32,
        count: u32,
        leaf: LeafRef,
    }
    let mut cands: Vec<Cand> = Vec::new();
    for entry in undrained {
        let superfile_id = entry.superfile_id.as_u128();
        if !survives(superfile_id) {
            continue;
        }
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
            cands.push(Cand {
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
    cands.sort_by(|a, b| a.near.partial_cmp(&b.near).unwrap_or(Ordering::Equal));
    // User superfiles are IVF-fragmented: semantic neighbors sit in many internal
    // clusters within one file. A doc-count coverage floor (meant for consolidated
    // post-drain cells) prunes sibling clusters and starves rerank — recall@10
    // drops while fetch cost stays one coalesced GET per superfile either way.
    cands.into_iter().map(|c| (c.leaf, c.d)).collect()
}
