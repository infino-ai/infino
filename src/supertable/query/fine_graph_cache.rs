// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Process-wide cache of the fine-centroid graphs (the drain's placement graph
//! and the graph queries route with), so a new generation updates the previous
//! generation's graph instead of rebuilding it over every fine centroid.
//!
//! A drain adds fine clusters (its own k-means output) and a merge or split
//! removes some, but most clusters survive from one drain to the next.
//! Superfiles never change, so a fine centroid is fully named by its
//! superfile, cell and position in the cell, and the centroid behind that key
//! never moves. The cached graph is updated by dropping the nodes whose keys
//! are gone ([`Hnsw::remove_nodes`]) and inserting the new keys in key order
//! ([`Hnsw::extend_serial`]), so the work grows with what changed rather than
//! with the table.
//!
//! The cache lives in the process rather than on a table handle because a
//! long-running process may open a new connection for each drain. An entry is
//! matched to a table by sharing a superfile with it: superfile ids are
//! unique, so a shared superfile means the cached nodes describe the same
//! centroids. A process restart costs one full build.

use std::{
    collections::HashSet,
    sync::{LazyLock, Mutex, MutexGuard, PoisonError},
};

use uuid::Uuid;

use crate::{
    superfile::vector::{
        distance::Metric,
        hnsw::{Hnsw, HnswParams, NodeScorer},
    },
    supertable::schema::FieldId,
};

/// One fine centroid. A placement graph keys it by superfile, cell and
/// position within the cell; a router graph by superfile and flat cluster id,
/// with no cell.
pub(crate) type FineNodeKey = (Uuid, Option<u32>, u32);

/// What a cached graph is for. The two cover different node sets (a router
/// graph leaves out empty and superseded cells), so each is cached apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GraphRole {
    /// The drain's graph for placing rows in cells.
    Placement,
    /// The graph queries route with.
    Router,
}

/// Graphs kept at once. Each holds adjacency for every fine centroid of one
/// table column (about 1.2 GB at 8.5M centroids), so the cache stays small.
const MAX_CACHED_GRAPHS: usize = 4;

/// Share of a graph's nodes that may be removed or added, counted over every
/// update since its last full build, before it is rebuilt instead of updated.
/// An updated graph keeps the quality of the build it started from, so once
/// much of it has changed a fresh build costs little more and starts that
/// quality over. Counting additions keeps a table that only grows from
/// patching one graph forever.
const MAX_CHANGED_SHARE: f64 = 0.2;

/// A graph over one table column's fine centroids, with the key of each node.
struct CachedGraph {
    role: GraphRole,
    column: FieldId,
    dim: usize,
    metric: Metric,
    /// `keys[node]` is the centroid behind graph node `node`.
    keys: Vec<FineNodeKey>,
    superfiles: HashSet<Uuid>,
    graph: Hnsw,
    /// Nodes removed plus nodes added over the updates since the last full
    /// build.
    changed_since_build: usize,
}

impl CachedGraph {
    /// Whether this is the `role` graph of the table column that owns any of
    /// `superfiles`.
    fn covers<'a>(
        &self,
        role: GraphRole,
        column: FieldId,
        dim: usize,
        metric: Metric,
        mut superfiles: impl Iterator<Item = &'a Uuid>,
    ) -> bool {
        self.role == role
            && self.column == column
            && self.dim == dim
            && self.metric == metric
            && superfiles.any(|id| self.superfiles.contains(id))
    }
}

/// The cache, usable after a panic elsewhere left the lock poisoned: every
/// entry is replaced whole, so none is ever left half-written.
fn lock_cache() -> MutexGuard<'static, Vec<CachedGraph>> {
    CACHE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Cached graphs, least recently stored first.
static CACHE: LazyLock<Mutex<Vec<CachedGraph>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// How a [`cached_graph`] call produced its graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GraphSource {
    /// No usable cached graph: built over every centroid.
    Built,
    /// A cached graph was updated with this many removed and added nodes.
    Updated { removed: usize, added: usize },
}

/// The `role` graph over `keys`, which must be sorted and distinct. A cached
/// graph of the same role and table column is updated while few enough of its
/// nodes have changed since its last full build; otherwise the graph is built
/// over every key. Returns the
/// graph, its scorer, `order` (`order[node]` is the index into `keys` of graph
/// node `node`) and how the graph was produced. `make_scorer` receives `order`
/// and must score the centroids of `keys` in that order. `parallel` inserts
/// over the rayon pool; serial gives the same graph for the same inputs. The
/// result is cached for the next call.
pub(crate) fn cached_graph<S, F>(
    role: GraphRole,
    column: FieldId,
    dim: usize,
    metric: Metric,
    keys: &[FineNodeKey],
    parallel: bool,
    make_scorer: F,
) -> (Hnsw, S, Vec<usize>, GraphSource)
where
    S: NodeScorer + Sync,
    F: FnOnce(&[usize]) -> S,
{
    debug_assert!(
        keys.windows(2).all(|w| w[0] < w[1]),
        "keys must be sorted and distinct"
    );
    let params = HnswParams::default();
    let (graph, scorer, order, source, changed_since_build) =
        match take_cached(role, column, dim, metric, keys) {
            Some(cached) => match update_plan(&cached.keys, cached.changed_since_build, keys) {
                Some((keep, order, removed, added)) => {
                    let scorer = make_scorer(&order);
                    let changed = cached.changed_since_build + removed + added;
                    let kept = cached.graph.remove_nodes(&keep, &scorer);
                    let graph = if parallel {
                        kept.extend(&scorer, params)
                    } else {
                        kept.extend_serial(&scorer, params)
                    };
                    (
                        graph,
                        scorer,
                        order,
                        GraphSource::Updated { removed, added },
                        changed,
                    )
                }
                None => build(keys, params, parallel, make_scorer),
            },
            None => build(keys, params, parallel, make_scorer),
        };
    store(CachedGraph {
        role,
        column,
        dim,
        metric,
        keys: order.iter().map(|&i| keys[i]).collect(),
        superfiles: keys.iter().map(|k| k.0).collect(),
        graph: graph.clone(),
        changed_since_build,
    });
    (graph, scorer, order, source)
}

fn build<S, F>(
    keys: &[FineNodeKey],
    params: HnswParams,
    parallel: bool,
    make_scorer: F,
) -> (Hnsw, S, Vec<usize>, GraphSource, usize)
where
    S: NodeScorer + Sync,
    F: FnOnce(&[usize]) -> S,
{
    let order: Vec<usize> = (0..keys.len()).collect();
    let scorer = make_scorer(&order);
    let graph = if parallel {
        Hnsw::build(&scorer, params)
    } else {
        Hnsw::build_serial(&scorer, params)
    };
    (graph, scorer, order, GraphSource::Built, 0)
}

/// How to turn a graph over `cached` into one over `keys`: which cached nodes
/// to keep, the new node order (survivors in their cached order, then the new
/// keys in key order), and the removed and added counts. `None` when this
/// update together with the `changed_before` nodes of earlier ones would
/// change too much of the graph to update rather than rebuild.
fn update_plan(
    cached: &[FineNodeKey],
    changed_before: usize,
    keys: &[FineNodeKey],
) -> Option<(Vec<bool>, Vec<usize>, usize, usize)> {
    let mut present = vec![false; keys.len()];
    let mut keep = Vec::with_capacity(cached.len());
    let mut order = Vec::with_capacity(keys.len());
    for key in cached {
        match keys.binary_search(key) {
            Ok(i) => {
                present[i] = true;
                keep.push(true);
                order.push(i);
            }
            Err(_) => keep.push(false),
        }
    }
    let removed = cached.len() - order.len();
    let survivors = order.len();
    order.extend((0..keys.len()).filter(|&i| !present[i]));
    let added = order.len() - survivors;
    let changed = changed_before + removed + added;
    if changed as f64 > MAX_CHANGED_SHARE * cached.len().max(keys.len()) as f64 {
        return None;
    }
    Some((keep, order, removed, added))
}

/// Remove and return the cached `role` graph of this table column, if any.
/// Taking it out keeps two concurrent builds for one table from updating the
/// same graph; the second builds from scratch.
fn take_cached(
    role: GraphRole,
    column: FieldId,
    dim: usize,
    metric: Metric,
    keys: &[FineNodeKey],
) -> Option<CachedGraph> {
    let mut cache = lock_cache();
    let at = cache
        .iter()
        .position(|c| c.covers(role, column, dim, metric, keys.iter().map(|k| &k.0)))?;
    Some(cache.remove(at))
}

/// Cache `entry`, replacing any graph of the same table column a concurrent
/// build stored meanwhile, so the cache holds at most one per table column.
fn store(entry: CachedGraph) {
    let mut cache = lock_cache();
    cache.retain(|c| {
        !c.covers(
            entry.role,
            entry.column,
            entry.dim,
            entry.metric,
            entry.superfiles.iter(),
        )
    });
    cache.push(entry);
    if cache.len() > MAX_CACHED_GRAPHS {
        cache.remove(0);
    }
}

#[cfg(test)]
mod tests {
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    use super::*;
    use crate::superfile::vector::hnsw::Fp32Scorer;

    /// Dimension of the test centroids.
    const DIM: usize = 16;
    /// Centroids per test cell.
    const PER_CELL: u32 = 50;

    /// Odd multipliers that spread each key part over the whole seed, so no
    /// two test keys share a seed (and so a centroid).
    const SEED_MIX: [u64; 3] = [
        0x9E37_79B9_7F4A_7C15,
        0xC2B2_AE3D_27D4_EB4F,
        0x1656_67B1_9E37_79F9,
    ];

    fn centroid(key: &FineNodeKey) -> Vec<f32> {
        let seed = (key.0.as_u128() as u64).wrapping_mul(SEED_MIX[0])
            ^ u64::from(key.1.unwrap_or(0)).wrapping_mul(SEED_MIX[1])
            ^ u64::from(key.2).wrapping_mul(SEED_MIX[2]);
        let mut rng = StdRng::seed_from_u64(seed);
        (0..DIM).map(|_| rng.random::<f32>() - 0.5).collect()
    }

    fn cell_keys(superfile: u128, cell: u32) -> Vec<FineNodeKey> {
        (0..PER_CELL).map(|p| key(superfile, cell, p)).collect()
    }

    fn graph_over(
        column: FieldId,
        keys: &[FineNodeKey],
    ) -> (Hnsw, Fp32Scorer, Vec<usize>, GraphSource) {
        cached_graph(
            GraphRole::Placement,
            column,
            DIM,
            Metric::L2Sq,
            keys,
            false,
            |order| {
                let ordered: Vec<Vec<f32>> = order.iter().map(|&i| centroid(&keys[i])).collect();
                Fp32Scorer::from_vectors(&ordered, DIM, Metric::L2Sq)
            },
        )
    }

    #[test]
    fn a_second_drain_updates_the_cached_graph() {
        // A superfile id range no other test uses, so the shared cache cannot
        // match another test's graph.
        let base = 0xF1_0000_0000u128;
        let column = FieldId(7);
        let mut keys: Vec<FineNodeKey> = (0..20).flat_map(|c| cell_keys(base, c)).collect();
        keys.sort();
        let (_, _, _, source) = graph_over(column, &keys);
        assert_eq!(source, GraphSource::Built);

        // The next drain: one cell's clusters are gone, a new superfile adds two.
        let mut next: Vec<FineNodeKey> = keys.iter().copied().filter(|k| k.1 != Some(3)).collect();
        next.extend(cell_keys(base + 1, 0));
        next.extend(cell_keys(base + 1, 1));
        next.sort();
        let (graph, scorer, order, source) = graph_over(column, &next);
        assert_eq!(
            source,
            GraphSource::Updated {
                removed: PER_CELL as usize,
                added: 2 * PER_CELL as usize
            }
        );
        assert_eq!(graph.len(), next.len());
        // Every node is reachable by its own centroid, the new ones included.
        for node in (0..order.len()).step_by(17) {
            let found = graph.search(&scorer, &centroid(&next[order[node]]), 1, 64);
            assert_eq!(found[0].0, node as u32, "node {node} unreachable");
        }
    }

    fn key(superfile: u128, cell: u32, pos: u32) -> FineNodeKey {
        (Uuid::from_u128(superfile), Some(cell), pos)
    }

    #[test]
    fn an_update_keeps_survivors_in_order_and_appends_new_keys_sorted() {
        let cached = vec![
            key(2, 0, 0),
            key(1, 0, 0),
            key(1, 0, 1),
            key(3, 0, 0),
            key(3, 0, 1),
        ];
        let mut keys = vec![
            key(1, 0, 0),
            key(1, 0, 1),
            key(2, 0, 0),
            key(3, 0, 0),
            key(3, 0, 1),
            key(4, 0, 0),
        ];
        keys.sort();
        let (keep, order, removed, added) = update_plan(&cached, 0, &keys).expect("small change");
        assert_eq!(keep, vec![true; 5]);
        assert_eq!((removed, added), (0, 1));
        let node_keys: Vec<FineNodeKey> = order.iter().map(|&i| keys[i]).collect();
        assert_eq!(&node_keys[..5], &cached[..]);
        assert_eq!(node_keys[5], key(4, 0, 0));
    }

    #[test]
    fn a_graph_that_changed_too_much_since_its_build_is_rebuilt() {
        let cached: Vec<FineNodeKey> = (0..10).map(|p| key(1, 0, p)).collect();
        // Seven of ten survive: 30% removed, over the limit.
        let fewer: Vec<FineNodeKey> = (0..7).map(|p| key(1, 0, p)).collect();
        assert!(update_plan(&cached, 0, &fewer).is_none());
        // Ten survive and three are added: 30% added, over the limit too.
        let more: Vec<FineNodeKey> = (0..13).map(|p| key(1, 0, p)).collect();
        assert!(update_plan(&cached, 0, &more).is_none());
        // One removed: an update...
        let one_gone: Vec<FineNodeKey> = (0..9).map(|p| key(1, 0, p)).collect();
        let (keep, _, removed, added) = update_plan(&cached, 0, &one_gone).expect("update");
        assert_eq!((removed, added), (1, 0));
        assert!(!keep[9]);
        // ...unless earlier updates already changed two nodes since the build.
        assert!(update_plan(&cached, 2, &one_gone).is_none());
    }
}
