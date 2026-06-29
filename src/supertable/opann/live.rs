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
/// ([`ClusterCentroids::score_one`]) and admit clusters by the *same*
/// radius-bounded coverage floor the persisted-tree descent uses
/// ([`super::paged::PagedTree::radius_bounded_descent`]) — process clusters by
/// near edge `(d − r)`, freeze the prune bound to the far edge `(d + r)` of the
/// nearest clusters once they cumulatively cover `floor` vectors, and admit
/// every cluster whose near edge still reaches that bound. Returns each admitted
/// cluster's [`LeafRef`] + its centroid distance, identical in shape to the tree
/// descent's output so the two candidate sets merge directly.
pub(crate) fn admit_undrained_manifest_clusters(
    undrained: &[Arc<SuperfileEntry>],
    column: &str,
    metric: Metric,
    query: &[f32],
    floor: usize,
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
    let mut admitted: Vec<(LeafRef, f32)> = Vec::new();
    let mut covered: u64 = 0;
    let mut admitted_far = 0.0f32;
    let mut bound = f32::INFINITY;
    for cand in cands {
        if cand.near > bound {
            break;
        }
        admitted.push((cand.leaf, cand.d));
        covered += cand.count as u64;
        admitted_far = admitted_far.max(cand.far);
        if bound.is_infinite() && covered >= floor as u64 {
            bound = admitted_far;
        }
    }
    admitted
}
