// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Follows a table's stored fine-centroid graph from one generation to the
//! next (see [`fine_graph_store`](super::fine_graph_store) for the stored
//! form).
//!
//! The process keeps the last graph it followed, with the base and steps it
//! was put together from. Asked for a later generation of the same graph it
//! fetches and applies only the steps it has not seen, so the work grows
//! with what changed. Asked for a graph it does not hold, or one it cannot
//! reach from what it holds, it fetches the base and every step.
//!
//! The kept graphs live in the process rather than on a table handle, so a
//! table that is reopened still follows from where the process left off.

use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};

use futures::future::try_join_all;
use tokio::{task::spawn_blocking, try_join};
use tracing::warn;

use crate::{
    storage::StorageProvider,
    supertable::{
        manifest::{list::RoutingRef, part::ContentHash},
        query::fine_graph_store::{GraphBase, GraphHead, GraphStep, KeyedGraph},
        slow_vector_state::{fetch_resident_index_blob, storage_path},
    },
};

/// Graphs kept at once. Each holds the adjacency and node keys of every
/// fine centroid of one table column, so the count stays small.
const MAX_FOLLOWED_GRAPHS: usize = 4;

/// A graph this process followed, and what it was put together from.
struct Followed {
    base: ContentHash,
    /// The steps applied on top of the base, oldest first.
    steps: Vec<ContentHash>,
    graph: KeyedGraph,
}

impl Followed {
    /// Whether `head` is this graph or a later generation of it.
    fn is_behind_or_at(&self, head: &GraphHead) -> bool {
        self.base == head.base.0
            && self.steps.len() <= head.steps.len()
            && self
                .steps
                .iter()
                .zip(&head.steps)
                .all(|(applied, (step, _))| applied == step)
    }
}

/// Followed graphs, least recently kept first.
static FOLLOWED: LazyLock<Mutex<Vec<Followed>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// The kept graphs, usable after a panic elsewhere left the lock poisoned:
/// every entry is replaced whole, so none is ever left half-written.
fn lock_followed() -> MutexGuard<'static, Vec<Followed>> {
    FOLLOWED.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Remove and return the kept graph `head` is a later generation of, if any.
/// Taking it out keeps two callers from applying steps to the same graph; the
/// second starts from the base.
fn take_behind(kept: &mut Vec<Followed>, head: &GraphHead) -> Option<Followed> {
    let at = kept.iter().position(|f| f.is_behind_or_at(head))?;
    Some(kept.remove(at))
}

/// Keep `entry` in place of the earlier generations of the same graph. An
/// entry that is behind one already kept is dropped, so a reader on an older
/// generation does not undo a newer one's work.
fn keep(kept: &mut Vec<Followed>, entry: Followed) {
    let ahead = |a: &Followed, b: &Followed| {
        a.base == b.base && a.steps.len() >= b.steps.len() && a.steps.starts_with(&b.steps)
    };
    if kept.iter().any(|k| ahead(k, &entry)) {
        return;
    }
    kept.retain(|k| !ahead(&entry, k));
    kept.push(entry);
    if kept.len() > MAX_FOLLOWED_GRAPHS {
        kept.remove(0);
    }
}

/// Drop every kept graph, as a process that has just started has none.
#[cfg(test)]
pub(crate) fn forget_all() {
    lock_followed().clear();
}

/// The graph `head` describes, or `None` when an object cannot be fetched,
/// does not decode, or the result is not the graph the head names. The
/// caller then builds the graph itself.
pub(crate) async fn follow(storage: &dyn StorageProvider, head: &GraphHead) -> Option<KeyedGraph> {
    let started = take_behind(&mut lock_followed(), head);
    if let Some(started) = started {
        let applied = started.steps.len();
        match put_together(storage, head, Some(started.graph), applied).await {
            Some(graph) => return Some(kept_copy(head, graph)),
            None => {
                warn!("fine-centroid graph: the kept graph did not follow; starting from the base")
            }
        }
    }
    let graph = put_together(storage, head, None, 0).await?;
    Some(kept_copy(head, graph))
}

/// Keep a copy of `graph`, the generation `head` describes, and return it.
fn kept_copy(head: &GraphHead, graph: KeyedGraph) -> KeyedGraph {
    keep(
        &mut lock_followed(),
        Followed {
            base: head.base.0,
            steps: head.steps.iter().map(|(step, _)| *step).collect(),
            graph: graph.clone(),
        },
    );
    graph
}

/// Fetch what `head` lists past its first `applied` steps (and the base when
/// there is no `graph` to start from), apply it, and return the result if it
/// is the generation `head` describes.
async fn put_together(
    storage: &dyn StorageProvider,
    head: &GraphHead,
    graph: Option<KeyedGraph>,
    applied: usize,
) -> Option<KeyedGraph> {
    let fetch = |hash: ContentHash| async move {
        let reference = RoutingRef {
            uri: storage_path(&hash),
            content_hash: hash,
        };
        fetch_resident_index_blob(storage, &reference)
            .await
            .map(|(bytes, _)| bytes)
    };
    let base = match graph {
        Some(_) => None,
        None => Some(fetch(head.base.0)),
    };
    let steps = try_join_all(head.steps[applied..].iter().map(|(step, _)| fetch(*step)));
    let (base, steps) = match base {
        Some(base) => {
            let (base, steps) = try_join!(base, steps)
                .map_err(|error| warn!(%error, "fine-centroid graph: fetch failed"))
                .ok()?;
            (Some(base), steps)
        }
        None => (
            None,
            steps
                .await
                .map_err(|error| warn!(%error, "fine-centroid graph: fetch failed"))
                .ok()?,
        ),
    };
    let head = head.clone();
    // Decoding and applying read every changed node: off the I/O threads.
    spawn_blocking(move || {
        let mut graph = match (graph, base) {
            (Some(graph), _) => graph,
            (None, Some(base)) => GraphBase::from_bytes(&base)?.into_graph(),
            (None, None) => return None,
        };
        for step in &steps {
            graph = GraphStep::from_bytes(step)?
                .apply(graph)
                .map_err(|error| warn!(?error, "fine-centroid graph: a step did not apply"))
                .ok()?;
        }
        head.describes(&graph).then_some(graph)
    })
    .await
    .ok()?
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;
    use crate::{
        superfile::vector::hnsw::{Hnsw, HnswParams, Sq16Scorer},
        supertable::{
            query::fine_graph_cache::{FineNodeKey, GraphRole},
            schema::FieldId,
        },
    };

    /// Dimensions of the test centroids.
    const DIM: usize = 8;
    /// Nodes of the test graph.
    const NODES: usize = 40;

    fn key(i: usize) -> FineNodeKey {
        (Uuid::from_u128(1 + i as u128), None, i as u32)
    }

    fn graph() -> KeyedGraph {
        let vectors: Vec<Vec<f32>> = (0..NODES)
            .map(|i| {
                let mut v = vec![0.0; DIM];
                v[i % DIM] = 1.0;
                v[(i / DIM) % DIM] += 0.5;
                v
            })
            .collect();
        let scorer = Sq16Scorer::from_unit_vectors(&vectors, DIM);
        KeyedGraph::new(
            (0..NODES).map(|i| Some(key(i))).collect(),
            Hnsw::build_serial(&scorer, HnswParams::default()),
        )
    }

    fn hash(n: u8) -> ContentHash {
        ContentHash::of(&[n])
    }

    /// A head over base `base` with `steps` steps, named by their index.
    fn head(base: u8, steps: u8) -> GraphHead {
        let graph = graph();
        let keys = graph.keys.iter().flatten().copied().collect();
        let full = GraphBase::new(keys, graph.graph).expect("base");
        let mut head = GraphHead::of_base(
            GraphRole::Router,
            FieldId(1),
            DIM as u32,
            hash(base),
            1,
            &full,
        );
        head.steps = (0..steps).map(|step| (hash(100 + step), 1)).collect();
        head
    }

    fn followed(base: u8, steps: u8) -> Followed {
        let head = head(base, steps);
        Followed {
            base: head.base.0,
            steps: head.steps.iter().map(|(step, _)| *step).collect(),
            graph: graph(),
        }
    }

    /// A kept graph is taken for its own generation and for later ones, and
    /// left alone for an earlier generation or another graph.
    #[test]
    fn a_kept_graph_is_taken_only_for_its_own_or_a_later_generation() {
        let mut kept = vec![followed(1, 2)];
        assert!(take_behind(&mut kept, &head(1, 1)).is_none(), "earlier");
        assert!(
            take_behind(&mut kept, &head(2, 3)).is_none(),
            "another base"
        );
        assert_eq!(kept.len(), 1);
        let taken = take_behind(&mut kept, &head(1, 3)).expect("later");
        assert_eq!(taken.steps.len(), 2);
        assert!(kept.is_empty());
        let mut kept = vec![followed(1, 2)];
        assert!(take_behind(&mut kept, &head(1, 2)).is_some(), "its own");
    }

    /// Keeping a later generation replaces the earlier one; keeping an
    /// earlier one changes nothing; the oldest graph goes when there are too
    /// many.
    #[test]
    fn the_latest_generation_of_each_graph_is_kept() {
        let mut kept = Vec::new();
        keep(&mut kept, followed(1, 1));
        keep(&mut kept, followed(1, 3));
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].steps.len(), 3);
        keep(&mut kept, followed(1, 2));
        assert_eq!(kept[0].steps.len(), 3, "an earlier generation is dropped");

        for base in 2..=MAX_FOLLOWED_GRAPHS as u8 + 1 {
            keep(&mut kept, followed(base, 0));
        }
        assert_eq!(kept.len(), MAX_FOLLOWED_GRAPHS);
        assert!(kept.iter().all(|k| k.base != hash(1)), "the oldest went");
    }
}
