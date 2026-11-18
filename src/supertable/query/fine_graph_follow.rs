// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Loads a stored fine-centroid graph (see
//! [`fine_graph_store`](super::fine_graph_store)).
//!
//! The process keeps the last graph it loaded. For a newer version of the
//! same graph it fetches and applies only the steps it does not have. With
//! nothing kept it fetches the base and every step.

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

/// How many graphs the process keeps.
const MAX_KEPT_GRAPHS: usize = 4;

/// A loaded graph and the base and steps it was made from.
struct Kept {
    base: ContentHash,
    steps: Vec<ContentHash>,
    graph: KeyedGraph,
}

impl Kept {
    /// Whether `head` is this graph or a newer version of it.
    fn leads_to(&self, head: &GraphHead) -> bool {
        self.base == head.base.0
            && self.steps.len() <= head.steps.len()
            && self
                .steps
                .iter()
                .zip(&head.steps)
                .all(|(applied, (step, _))| applied == step)
    }
}

static KEPT: LazyLock<Mutex<Vec<Kept>>> = LazyLock::new(|| Mutex::new(Vec::new()));

fn lock_kept() -> MutexGuard<'static, Vec<Kept>> {
    KEPT.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Remove and return the kept graph that `head` is a newer version of.
fn take_older(kept: &mut Vec<Kept>, head: &GraphHead) -> Option<Kept> {
    let at = kept.iter().position(|k| k.leads_to(head))?;
    Some(kept.remove(at))
}

/// Keep `entry` unless a newer version of the same graph is already kept.
fn keep(kept: &mut Vec<Kept>, entry: Kept) {
    let newer = |a: &Kept, b: &Kept| a.base == b.base && a.steps.starts_with(&b.steps);
    if kept.iter().any(|k| newer(k, &entry)) {
        return;
    }
    kept.retain(|k| !newer(&entry, k));
    kept.push(entry);
    if kept.len() > MAX_KEPT_GRAPHS {
        kept.remove(0);
    }
}

#[cfg(test)]
pub(crate) fn forget_all() {
    lock_kept().clear();
}

/// The graph `head` describes. `None` if it cannot be fetched, decoded or
/// matched to the head; the caller then builds the graph itself.
pub(crate) async fn load(storage: &dyn StorageProvider, head: &GraphHead) -> Option<KeyedGraph> {
    let older = take_older(&mut lock_kept(), head);
    let mut graph = None;
    if let Some(older) = older {
        let applied = older.steps.len();
        graph = build(storage, head, Some(older.graph), applied).await;
    }
    let graph = match graph {
        Some(graph) => graph,
        None => build(storage, head, None, 0).await?,
    };
    keep(
        &mut lock_kept(),
        Kept {
            base: head.base.0,
            steps: head.steps.iter().map(|(step, _)| *step).collect(),
            graph: graph.clone(),
        },
    );
    Some(graph)
}

/// Apply the steps of `head` after the first `applied` to `graph`, or to the
/// base when there is no `graph`.
async fn build(
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
    let hashes = head.steps[applied..].iter().map(|(step, _)| *step);
    let base = graph.is_none().then_some(head.base.0);
    let fetched = try_join!(
        async {
            match base {
                Some(base) => fetch(base).await.map(Some),
                None => Ok(None),
            }
        },
        try_join_all(hashes.map(fetch)),
    );
    let (base, steps) = fetched
        .map_err(|error| warn!(%error, "fine-centroid graph: fetch failed"))
        .ok()?;
    let head = head.clone();
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

    const DIM: usize = 8;
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

    fn head(base: u8, steps: u8) -> GraphHead {
        let graph = graph();
        let keys = graph.keys.iter().flatten().copied().collect();
        let full = GraphBase::new(keys, Hnsw::clone(&graph.graph)).expect("base");
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

    fn kept_at(base: u8, steps: u8) -> Kept {
        let head = head(base, steps);
        Kept {
            base: head.base.0,
            steps: head.steps.iter().map(|(step, _)| *step).collect(),
            graph: graph(),
        }
    }

    #[test]
    fn a_kept_graph_is_taken_only_for_its_own_or_a_later_generation() {
        let mut kept = vec![kept_at(1, 2)];
        assert!(take_older(&mut kept, &head(1, 1)).is_none(), "earlier");
        assert!(take_older(&mut kept, &head(2, 3)).is_none(), "another base");
        assert_eq!(kept.len(), 1);
        let taken = take_older(&mut kept, &head(1, 3)).expect("later");
        assert_eq!(taken.steps.len(), 2);
        assert!(kept.is_empty());
        let mut kept = vec![kept_at(1, 2)];
        assert!(take_older(&mut kept, &head(1, 2)).is_some(), "its own");
    }

    #[test]
    fn the_latest_generation_of_each_graph_is_kept() {
        let mut kept = Vec::new();
        keep(&mut kept, kept_at(1, 1));
        keep(&mut kept, kept_at(1, 3));
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].steps.len(), 3);
        keep(&mut kept, kept_at(1, 2));
        assert_eq!(kept[0].steps.len(), 3, "an earlier generation is dropped");

        for base in 2..=MAX_KEPT_GRAPHS as u8 + 1 {
            keep(&mut kept, kept_at(base, 0));
        }
        assert_eq!(kept.len(), MAX_KEPT_GRAPHS);
        assert!(kept.iter().all(|k| k.base != hash(1)), "the oldest went");
    }
}
