// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Loads a stored fine-centroid graph (see
//! [`fine_graph_store`](super::fine_graph_store)).
//!
//! A caller that still has the graph it loaded last passes it in. If the
//! stored graph is a newer version of it, only the missing steps are fetched
//! and applied. Otherwise the base and every step are.

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

/// A loaded graph and the base and steps it was made from.
pub(crate) struct Loaded {
    base: ContentHash,
    steps: Vec<ContentHash>,
    pub(crate) graph: KeyedGraph,
}

impl Loaded {
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

/// The graph `head` describes, starting from `last` when it can. `None` if
/// it cannot be fetched, decoded or matched to the head.
pub(crate) async fn load(
    storage: &dyn StorageProvider,
    head: &GraphHead,
    last: Option<&Loaded>,
) -> Option<Loaded> {
    let mut graph = None;
    if let Some(last) = last.filter(|last| last.leads_to(head)) {
        graph = build(storage, head, Some(last.graph.clone()), last.steps.len()).await;
    }
    let graph = match graph {
        Some(graph) => graph,
        None => build(storage, head, None, 0).await?,
    };
    Some(Loaded {
        base: head.base.0,
        steps: head.steps.iter().map(|(step, _)| *step).collect(),
        graph,
    })
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
