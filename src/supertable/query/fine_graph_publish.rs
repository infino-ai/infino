// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Writes a fine-centroid graph in its stored form (see
//! [`fine_graph_store`](super::fine_graph_store)).
//!
//! The process keeps the last graph it wrote for a table. The next one is
//! written as a step from it. With none kept, or when a fresh base is due, a
//! base is written.

use std::{
    collections::HashSet,
    sync::{LazyLock, Mutex, MutexGuard, PoisonError},
};

use tokio::task::spawn_blocking;
use tracing::warn;
use uuid::Uuid;

use crate::{
    storage::StorageProvider,
    superfile::vector::hnsw::Hnsw,
    supertable::{
        manifest::list::RoutingRef,
        query::{
            fine_graph_cache::{FineNodeKey, GraphRole, MAX_CACHED_GRAPHS},
            fine_graph_store::{GraphBase, GraphHead, GraphStep, KeyedGraph},
        },
        schema::FieldId,
        slow_vector_state::write_resident_index_blob,
    },
};

/// The last graph written for one table column.
struct Written {
    head: GraphHead,
    reference: RoutingRef,
    graph: KeyedGraph,
    superfiles: HashSet<Uuid>,
}

static WRITTEN: LazyLock<Mutex<Vec<Written>>> = LazyLock::new(|| Mutex::new(Vec::new()));

fn lock_written() -> MutexGuard<'static, Vec<Written>> {
    WRITTEN.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Whether `next` is `last` with some nodes removed and some added after it.
fn continues(last: &KeyedGraph, next: &KeyedGraph) -> bool {
    next.keys.len() >= last.keys.len()
        && last
            .keys
            .iter()
            .zip(&next.keys)
            .all(|(before, after)| after.is_none() || after == before)
}

/// Whether the next graph of the table that owns any of `superfiles` must be
/// written as a base. The caller then builds that graph from scratch, since
/// a base holds no holes.
pub(crate) fn base_due<'a>(
    role: GraphRole,
    column: FieldId,
    mut superfiles: impl Iterator<Item = &'a Uuid>,
) -> bool {
    let written = lock_written();
    superfiles.any(|id| {
        written.iter().any(|w| {
            w.head.role == role
                && w.head.column == column
                && w.superfiles.contains(id)
                && w.head.fold_due()
        })
    })
}

/// Whether `reference` is the head of a graph this process wrote.
pub(crate) fn wrote(reference: &RoutingRef) -> bool {
    lock_written().iter().any(|w| w.reference == *reference)
}

/// Write `graph` and return the ref of its head. `None` if it could not be
/// written in this form; the caller then writes the graph whole.
pub(crate) async fn write(
    storage: &dyn StorageProvider,
    role: GraphRole,
    column: FieldId,
    dim: u32,
    graph: KeyedGraph,
) -> Option<RoutingRef> {
    let last = {
        let mut written = lock_written();
        written
            .iter()
            .position(|w| {
                w.head.role == role && w.head.column == column && continues(&w.graph, &graph)
            })
            .map(|at| written.remove(at))
    };
    if last
        .as_ref()
        .is_some_and(|last| last.head.describes(&graph))
    {
        let last = last?;
        let reference = last.reference.clone();
        lock_written().push(last);
        return Some(reference);
    }
    let (graph, object) = spawn_blocking(move || {
        let object = match last.filter(|last| !last.head.fold_due()) {
            Some(last) => GraphStep::between(&last.graph, &graph)
                .ok()
                .map(|step| (step.to_bytes(), Object::Step(last.head, step))),
            None => {
                let keys: Option<Vec<FineNodeKey>> = graph.keys.iter().copied().collect();
                keys.and_then(|keys| GraphBase::new(keys, Hnsw::clone(&graph.graph)).ok())
                    .map(|base| (base.to_bytes(), Object::Base(base)))
            }
        };
        (graph, object)
    })
    .await
    .ok()?;
    let (bytes, object) = object?;
    let len = bytes.len() as u64;
    let stored = put(storage, bytes).await?;
    let head = match object {
        Object::Step(head, step) => head.with_step(stored.content_hash, len, &step),
        Object::Base(base) => {
            GraphHead::of_base(role, column, dim, stored.content_hash, len, &base)
        }
    };
    let reference = put(storage, head.to_bytes()).await?;
    let superfiles = graph.keys.iter().flatten().map(|key| key.0).collect();
    let mut written = lock_written();
    written.push(Written {
        head,
        reference: reference.clone(),
        graph,
        superfiles,
    });
    if written.len() > MAX_CACHED_GRAPHS {
        written.remove(0);
    }
    Some(reference)
}

enum Object {
    Step(GraphHead, GraphStep),
    Base(GraphBase),
}

async fn put(storage: &dyn StorageProvider, bytes: Vec<u8>) -> Option<RoutingRef> {
    write_resident_index_blob(storage, bytes)
        .await
        .map_err(|error| warn!(%error, "fine-centroid graph: write failed"))
        .ok()
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::{
        storage::LocalFsStorageProvider,
        superfile::vector::hnsw::{HnswParams, Sq16Scorer},
        supertable::slow_vector_state::fetch_resident_index_blob,
    };

    const DIM: usize = 8;
    const NODES: usize = 60;
    const ADDED: usize = 4;

    fn vectors(count: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|i| {
                let mut v = vec![0.0; DIM];
                v[i % DIM] = 1.0;
                v[(i / DIM) % DIM] += 0.5;
                v
            })
            .collect()
    }

    /// Keys of a table no other test writes.
    fn key(table: u128, i: usize) -> FineNodeKey {
        (Uuid::from_u128(table), None, i as u32)
    }

    fn first(table: u128) -> KeyedGraph {
        let scorer = Sq16Scorer::from_unit_vectors(&vectors(NODES), DIM);
        KeyedGraph::new(
            (0..NODES).map(|i| Some(key(table, i))).collect(),
            Hnsw::build_serial(&scorer, HnswParams::default()),
        )
    }

    /// `graph` with node 3 removed and `ADDED` nodes after the last.
    fn next(table: u128, graph: &KeyedGraph) -> KeyedGraph {
        let slots = graph.keys.len();
        let before = Sq16Scorer::from_unit_vectors(&vectors(slots), DIM);
        let after = Sq16Scorer::from_unit_vectors(&vectors(slots + ADDED), DIM);
        let mut keys = graph.keys.clone();
        keys[3] = None;
        keys.extend((slots..slots + ADDED).map(|i| Some(key(table, i))));
        KeyedGraph::new(
            keys,
            Hnsw::clone(&graph.graph)
                .unlink_nodes(&[3], &before)
                .extend_serial(&after, HnswParams::default()),
        )
    }

    async fn head(storage: &dyn StorageProvider, graph: KeyedGraph) -> Option<GraphHead> {
        let reference = write(storage, GraphRole::Router, FieldId(1), DIM as u32, graph).await?;
        let (bytes, _) = fetch_resident_index_blob(storage, &reference)
            .await
            .expect("fetch");
        GraphHead::from_bytes(&bytes)
    }

    #[tokio::test]
    async fn a_later_graph_is_written_as_a_step() {
        let table = 0xA1_0000;
        let dir = TempDir::new().expect("tempdir");
        let storage = LocalFsStorageProvider::new(dir.path()).expect("storage");
        let one = first(table);
        let two = next(table, &one);

        let base = head(&storage, one.clone()).await.expect("base");
        assert!(base.steps.is_empty());
        assert!(base.describes(&one));

        let stepped = head(&storage, two.clone()).await.expect("step");
        assert_eq!(stepped.base, base.base);
        assert_eq!(stepped.steps.len(), 1);
        assert!(stepped.describes(&two));

        let again = head(&storage, two).await.expect("unchanged");
        assert_eq!(again, stepped);
    }

    #[tokio::test]
    async fn a_graph_with_holes_and_nothing_kept_is_not_written() {
        let table = 0xA2_0000;
        let dir = TempDir::new().expect("tempdir");
        let storage = LocalFsStorageProvider::new(dir.path()).expect("storage");
        assert!(head(&storage, next(table, &first(table))).await.is_none());
    }
}
