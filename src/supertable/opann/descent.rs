// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Best-first descent over a routing tree, shared by the in-memory
//! [`super::tree::CentroidTree`], the single on-disk [`super::page::Page`], and
//! the cross-page [`super::paged::PagedTree`].
//!
//! All three descend the identical algorithm — a min-heap keyed on node
//! distance: pop the closest node; a leaf is a probe, an internal node pushes
//! its children. Only the node representation differs (an in-RAM node index, a
//! local page index, or a `(page, local)` handle that may resolve a new page),
//! so the loop lives here once, generic over the node handle `N`, and each
//! caller supplies one closure that both classifies a node and pushes its
//! already-scored children. A single closure (rather than separate score +
//! expand) lets a caller hold mutable state — e.g. a lazily-populated page
//! cache — exclusively across the descent. Keeping one copy means the on-disk
//! descents can never silently diverge from the in-memory one.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Best-first descent yielding up to `n_probe` `(cell_id, distance)` pairs in
/// the order leaves are reached. The search is seeded with `root` at distance
/// `root_dist`. `expand(node, &mut kids)` returns `Some(cell_id)` when `node`
/// is a leaf; otherwise it pushes each child as `(child_handle, child_distance)`
/// into `kids` and returns `None`. Children are scored by the caller inside
/// `expand` (co-located with the data they come from), so the descent itself
/// needs no scoring access. Pure compute — the caller owns all scoring data
/// and any I/O.
pub(super) fn best_first<N: Copy, L: Copy>(
    root: N,
    root_dist: f32,
    n_probe: usize,
    mut expand: impl FnMut(N, &mut Vec<(N, f32)>) -> Option<L>,
) -> Vec<(L, f32)> {
    let mut heap: BinaryHeap<Cand<N>> = BinaryHeap::new();
    heap.push(Cand {
        dist: root_dist,
        node: root,
    });
    let mut out: Vec<(L, f32)> = Vec::new();
    let mut kids: Vec<(N, f32)> = Vec::new();
    while let Some(Cand { dist, node }) = heap.pop() {
        kids.clear();
        if let Some(leaf) = expand(node, &mut kids) {
            out.push((leaf, dist));
            if out.len() >= n_probe {
                break;
            }
        } else {
            for &(child, child_dist) in &kids {
                heap.push(Cand {
                    dist: child_dist,
                    node: child,
                });
            }
        }
    }
    out
}

/// Best-first heap candidate, ordered so `BinaryHeap` pops the *smallest*
/// distance first (a min-heap via reversed compare). Ordering is on `dist`
/// only; the node handle `N` is carried, never compared.
struct Cand<N> {
    dist: f32,
    node: N,
}

impl<N> PartialEq for Cand<N> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl<N> Eq for Cand<N> {}
impl<N> Ord for Cand<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: the "greatest" candidate is the nearest one.
        other.dist.total_cmp(&self.dist)
    }
}
impl<N> PartialOrd for Cand<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
