// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The stored form of a fine-centroid graph: a full copy, then one small
//! object per generation that says what changed.
//!
//! ```text
//! head ──► base            the whole graph, and the centroid behind each node
//!      ├─► step 1          what the first generation after the base changed
//!      ├─► step 2
//!      └─► ...
//! ```
//!
//! A reader that holds one generation follows the writer by fetching the
//! steps it has not applied. One that holds nothing fetches the base and
//! every step. Either way it ends with the writer's graph exactly, which
//! the head's checksum confirms.
//!
//! Every object is named by the hash of its bytes, so none is ever edited:
//! a new generation writes one step and a new head. When the steps have
//! grown large beside the base ([`GraphHead::fold_due`]) the writer builds a
//! fresh base, which also clears the holes removed nodes leave.

use std::{fmt, sync::Arc};

use uuid::Uuid;

use crate::{
    superfile::vector::hnsw::{Cursor, GraphChange, GraphChangeError, Hnsw},
    supertable::{
        manifest::part::ContentHash,
        query::fine_graph_cache::{FineNodeKey, GraphRole},
        schema::FieldId,
    },
};

/// Magic of an encoded [`GraphHead`].
const HEAD_MAGIC: &[u8; 8] = b"INFCGH01";
/// Magic of an encoded [`GraphBase`].
const BASE_MAGIC: &[u8; 8] = b"INFCGB01";
/// Magic of an encoded [`GraphStep`].
const STEP_MAGIC: &[u8; 8] = b"INFCGS01";

/// Share of a graph's slots that may be holes or changed nodes, counted over
/// the steps since its base, before a fresh base is due, in divisor form
/// (`slots / this`). An updated graph keeps the quality of the build it
/// started from, so once a fifth of it has changed a fresh build costs
/// little more and starts that quality over.
const FOLD_CHANGED_SHARE_DIVISOR: u64 = 5;

/// Share of the base's bytes the steps may add up to before a fresh base is
/// due, in divisor form (`base bytes / this`): past it a reader with nothing
/// fetches half as much again as the graph itself.
const FOLD_STEP_BYTES_SHARE_DIVISOR: u64 = 2;

/// Steps a head may list before a fresh base is due, however small they are:
/// a reader with nothing fetches each one.
const FOLD_MAX_STEPS: usize = 64;

/// Bytes of an encoded hash.
const HASH_BYTES: usize = 32;
/// Bytes of an encoded superfile id.
const UUID_BYTES: usize = 16;

/// Tag of an encoded node key that is a hole.
const KEY_HOLE: u8 = 0;
/// Tag of an encoded node key with no cell.
const KEY_NO_CELL: u8 = 1;
/// Tag of an encoded node key with a cell.
const KEY_WITH_CELL: u8 = 2;

/// Code of [`GraphRole::Placement`] in an encoded head.
const ROLE_PLACEMENT: u8 = 0;
/// Code of [`GraphRole::Router`] in an encoded head.
const ROLE_ROUTER: u8 = 1;

/// Why a stored graph could not be put back together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GraphStoreError {
    /// A step did not apply to the graph before it.
    Change(GraphChangeError),
    /// A step names a different number of keys than nodes it appends, or a
    /// base a different number than its graph has nodes.
    KeyCount,
    /// A base holds a hole. Only a fresh build is written as a base.
    HoleInBase,
    /// The graph put together is not the one the head describes.
    #[cfg(test)]
    HeadMismatch,
}

impl From<GraphChangeError> for GraphStoreError {
    fn from(error: GraphChangeError) -> Self {
        Self::Change(error)
    }
}

/// A graph and the centroid behind each of its nodes.
#[derive(Clone)]
pub(crate) struct KeyedGraph {
    /// `keys[node]` is the centroid behind graph node `node`; `None` for a
    /// hole.
    pub(crate) keys: Vec<Option<FineNodeKey>>,
    /// Shared with the query router, so a loaded graph is held once.
    pub(crate) graph: Arc<Hnsw>,
    /// The graph's [`Hnsw::node_digest`], kept current as steps are applied
    /// so a step is checked without reading the whole graph.
    node_digest: u64,
}

impl KeyedGraph {
    pub(crate) fn new(keys: Vec<Option<FineNodeKey>>, graph: Hnsw) -> Self {
        Self::shared(keys, Arc::new(graph))
    }

    pub(crate) fn shared(keys: Vec<Option<FineNodeKey>>, graph: Arc<Hnsw>) -> Self {
        let node_digest = graph.node_digest();
        Self {
            keys,
            graph,
            node_digest,
        }
    }
}

/// A full copy of a graph: what a reader with nothing starts from. It holds
/// no holes, since it is written from a fresh build.
#[derive(Clone)]
pub(crate) struct GraphBase {
    keys: Vec<FineNodeKey>,
    graph: Hnsw,
}

impl fmt::Debug for KeyedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyedGraph")
            .field("slots", &self.graph.len())
            .field("live", &self.graph.live_len())
            .finish()
    }
}

impl fmt::Debug for GraphBase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphBase")
            .field("nodes", &self.graph.len())
            .finish()
    }
}

impl GraphBase {
    /// The base of `graph`, whose node `i` stands for `keys[i]`.
    pub(crate) fn new(keys: Vec<FineNodeKey>, graph: Hnsw) -> Result<Self, GraphStoreError> {
        if keys.len() != graph.len() {
            return Err(GraphStoreError::KeyCount);
        }
        if graph.live_len() != graph.len() {
            return Err(GraphStoreError::HoleInBase);
        }
        Ok(Self { keys, graph })
    }

    /// The base as bytes: magic, the keys, then the graph. Paired with
    /// [`from_bytes`](Self::from_bytes).
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let graph = self.graph.to_bytes();
        let mut out = Vec::with_capacity(BASE_MAGIC.len() + self.keys.len() * 25 + graph.len());
        out.extend_from_slice(BASE_MAGIC);
        out.extend_from_slice(&(self.keys.len() as u64).to_le_bytes());
        for key in &self.keys {
            put_key(&mut out, Some(key));
        }
        out.extend_from_slice(&graph);
        out
    }

    /// Decode [`to_bytes`](Self::to_bytes). `None` on a bad magic, a
    /// truncation, a hole among the keys, or a graph that does not decode or
    /// does not match the keys.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut c = Cursor::new(bytes);
        if c.take(BASE_MAGIC.len())? != BASE_MAGIC {
            return None;
        }
        let keys = take_keys(&mut c)?
            .into_iter()
            .collect::<Option<Vec<FineNodeKey>>>()?;
        let graph = Hnsw::from_bytes(c.take(c.remaining())?)?;
        Self::new(keys, graph).ok()
    }

    /// The graph a reader holds once it has this base and no step.
    pub(crate) fn into_graph(self) -> KeyedGraph {
        KeyedGraph::new(self.keys.into_iter().map(Some).collect(), self.graph)
    }
}

/// What one generation changed: the graph's own change, and the centroid
/// behind each node it appends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraphStep {
    change: GraphChange,
    /// Keys of the appended nodes, in node order; `None` for one that is
    /// already a hole.
    appended_keys: Vec<Option<FineNodeKey>>,
}

impl GraphStep {
    /// The step that turns `before` into `after`, a later generation of it.
    pub(crate) fn between(
        before: &KeyedGraph,
        after: &KeyedGraph,
    ) -> Result<Self, GraphStoreError> {
        if after.keys.len() != after.graph.len() || before.keys.len() != before.graph.len() {
            return Err(GraphStoreError::KeyCount);
        }
        let change = before.graph.diff(&after.graph)?;
        Ok(Self {
            change,
            appended_keys: after.keys[before.keys.len()..].to_vec(),
        })
    }

    /// Apply this step to `graph`, which must be the generation it was made
    /// from. A step that does not fit is an error, and `graph` is consumed
    /// either way: the caller starts again from the base.
    pub(crate) fn apply(&self, graph: KeyedGraph) -> Result<KeyedGraph, GraphStoreError> {
        let (_, _, appended) = self.change.counts();
        if appended != self.appended_keys.len() || graph.keys.len() != graph.graph.len() {
            return Err(GraphStoreError::KeyCount);
        }
        let KeyedGraph {
            mut keys,
            graph: hnsw,
            node_digest,
        } = graph;
        let (hnsw, node_digest) =
            Arc::unwrap_or_clone(hnsw).apply_change(&self.change, node_digest)?;
        for &node in self.change.removed() {
            keys[node as usize] = None;
        }
        keys.extend_from_slice(&self.appended_keys);
        Ok(KeyedGraph {
            keys,
            graph: Arc::new(hnsw),
            node_digest,
        })
    }

    /// The step as bytes: magic, the appended keys, then the graph's change.
    /// Paired with [`from_bytes`](Self::from_bytes).
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(STEP_MAGIC);
        out.extend_from_slice(&(self.appended_keys.len() as u64).to_le_bytes());
        for key in &self.appended_keys {
            put_key(&mut out, key.as_ref());
        }
        out.extend_from_slice(&self.change.to_bytes());
        out
    }

    /// Decode [`to_bytes`](Self::to_bytes). `None` on a bad magic, a
    /// truncation or a change that does not decode. Whether the step fits a
    /// graph is checked when it is applied.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut c = Cursor::new(bytes);
        if c.take(STEP_MAGIC.len())? != STEP_MAGIC {
            return None;
        }
        let appended_keys = take_keys(&mut c)?;
        let change = GraphChange::from_bytes(c.take(c.remaining())?)?;
        Some(Self {
            change,
            appended_keys,
        })
    }
}

/// What a reader fetches to hold the current generation of one column's
/// graph: its base, then its steps in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraphHead {
    pub(crate) role: GraphRole,
    pub(crate) column: FieldId,
    pub(crate) dim: u32,
    /// The base, and its size in bytes.
    pub(crate) base: (ContentHash, u64),
    /// The steps since the base, oldest first, each with its size in bytes.
    pub(crate) steps: Vec<(ContentHash, u64)>,
    /// Slots in the current graph, holes included.
    slots: u64,
    /// Nodes removed plus nodes appended over the steps since the base.
    changed_since_base: u64,
    /// [`Hnsw::checksum`] of the current graph.
    checksum: u64,
}

impl GraphHead {
    /// How many leading bytes [`is_head`](Self::is_head) needs.
    pub(crate) const MAGIC_BYTES: usize = HEAD_MAGIC.len();

    /// Whether an object starting with `prefix` is a head.
    pub(crate) fn is_head(prefix: &[u8]) -> bool {
        prefix.starts_with(HEAD_MAGIC)
    }

    /// The head of a graph that is its base `base`, stored as `bytes` bytes
    /// under `hash`, with no step yet.
    pub(crate) fn of_base(
        role: GraphRole,
        column: FieldId,
        dim: u32,
        hash: ContentHash,
        bytes: u64,
        base: &GraphBase,
    ) -> Self {
        Self {
            role,
            column,
            dim,
            base: (hash, bytes),
            steps: Vec::new(),
            slots: base.graph.len() as u64,
            changed_since_base: 0,
            checksum: base.graph.checksum(),
        }
    }

    /// The head after `step`, stored as `bytes` bytes under `hash`, is added
    /// to this one's steps.
    pub(crate) fn with_step(&self, hash: ContentHash, bytes: u64, step: &GraphStep) -> Self {
        let (removed, _, appended) = step.change.counts();
        let mut steps = self.steps.clone();
        steps.push((hash, bytes));
        Self {
            role: self.role,
            column: self.column,
            dim: self.dim,
            base: self.base,
            steps,
            slots: self.slots + appended as u64,
            changed_since_base: self.changed_since_base + (removed + appended) as u64,
            checksum: step.change.checksum(),
        }
    }

    /// Whether the writer should build a fresh base instead of adding one
    /// more step: a fifth of the graph has changed since the base, the steps
    /// add up to half the base's bytes, or there are [`FOLD_MAX_STEPS`] of
    /// them.
    pub(crate) fn fold_due(&self) -> bool {
        let step_bytes: u64 = self.steps.iter().map(|(_, bytes)| *bytes).sum();
        self.changed_since_base > self.slots / FOLD_CHANGED_SHARE_DIVISOR
            || step_bytes > self.base.1 / FOLD_STEP_BYTES_SHARE_DIVISOR
            || self.steps.len() >= FOLD_MAX_STEPS
    }

    /// Whether `graph` is the generation this head describes.
    pub(crate) fn describes(&self, graph: &KeyedGraph) -> bool {
        graph.keys.len() == graph.graph.len()
            && graph.graph.len() as u64 == self.slots
            && graph.graph.checksum_of(graph.node_digest) == self.checksum
    }

    /// Put the current generation together from the decoded `base` and
    /// `steps`, which must be this head's, in its order.
    #[cfg(test)]
    pub(crate) fn replay(
        &self,
        base: GraphBase,
        steps: &[GraphStep],
    ) -> Result<KeyedGraph, GraphStoreError> {
        if steps.len() != self.steps.len() {
            return Err(GraphStoreError::HeadMismatch);
        }
        let mut graph = base.into_graph();
        for step in steps {
            graph = step.apply(graph)?;
        }
        if !self.describes(&graph) {
            return Err(GraphStoreError::HeadMismatch);
        }
        Ok(graph)
    }

    /// The head as bytes. Paired with [`from_bytes`](Self::from_bytes).
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(HEAD_MAGIC);
        out.push(match self.role {
            GraphRole::Placement => ROLE_PLACEMENT,
            GraphRole::Router => ROLE_ROUTER,
        });
        out.extend_from_slice(&self.column.0.to_le_bytes());
        out.extend_from_slice(&self.dim.to_le_bytes());
        out.extend_from_slice(&self.slots.to_le_bytes());
        out.extend_from_slice(&self.changed_since_base.to_le_bytes());
        out.extend_from_slice(&self.checksum.to_le_bytes());
        put_object(&mut out, &self.base);
        out.extend_from_slice(&(self.steps.len() as u64).to_le_bytes());
        for step in &self.steps {
            put_object(&mut out, step);
        }
        out
    }

    /// Decode [`to_bytes`](Self::to_bytes). `None` on a bad magic, an
    /// unknown role, a truncation or trailing bytes.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut c = Cursor::new(bytes);
        if c.take(HEAD_MAGIC.len())? != HEAD_MAGIC {
            return None;
        }
        let role = match c.u8()? {
            ROLE_PLACEMENT => GraphRole::Placement,
            ROLE_ROUTER => GraphRole::Router,
            _ => return None,
        };
        let column = FieldId(c.u32()?);
        let dim = c.u32()?;
        let slots = c.u64()?;
        let changed_since_base = c.u64()?;
        let checksum = c.u64()?;
        let base = take_object(&mut c)?;
        let count = usize::try_from(c.u64()?).ok()?;
        if count.checked_mul(HASH_BYTES + 8)? > c.remaining() {
            return None;
        }
        let mut steps = Vec::with_capacity(count);
        for _ in 0..count {
            steps.push(take_object(&mut c)?);
        }
        if c.remaining() != 0 {
            return None;
        }
        Some(Self {
            role,
            column,
            dim,
            base,
            steps,
            slots,
            changed_since_base,
            checksum,
        })
    }
}

/// Append a stored object's hash and size.
fn put_object(out: &mut Vec<u8>, (hash, bytes): &(ContentHash, u64)) {
    out.extend_from_slice(&hash.0);
    out.extend_from_slice(&bytes.to_le_bytes());
}

/// Read what [`put_object`] wrote.
fn take_object(c: &mut Cursor<'_>) -> Option<(ContentHash, u64)> {
    let hash: [u8; HASH_BYTES] = c.take(HASH_BYTES)?.try_into().ok()?;
    Some((ContentHash(hash), c.u64()?))
}

/// Append one node key: a tag, then the superfile, the cell when it has one,
/// and the position.
fn put_key(out: &mut Vec<u8>, key: Option<&FineNodeKey>) {
    let Some((superfile, cell, pos)) = key else {
        out.push(KEY_HOLE);
        return;
    };
    out.push(if cell.is_some() {
        KEY_WITH_CELL
    } else {
        KEY_NO_CELL
    });
    out.extend_from_slice(superfile.as_bytes());
    if let Some(cell) = cell {
        out.extend_from_slice(&cell.to_le_bytes());
    }
    out.extend_from_slice(&pos.to_le_bytes());
}

/// Read a `u64` count, then that many keys written by [`put_key`]. The count
/// is checked against the bytes left before anything is reserved.
fn take_keys(c: &mut Cursor<'_>) -> Option<Vec<Option<FineNodeKey>>> {
    let count = usize::try_from(c.u64()?).ok()?;
    // A key is at least its tag.
    if count > c.remaining() {
        return None;
    }
    let mut keys = Vec::with_capacity(count);
    for _ in 0..count {
        let tag = c.u8()?;
        if tag == KEY_HOLE {
            keys.push(None);
            continue;
        }
        let superfile = Uuid::from_bytes(c.take(UUID_BYTES)?.try_into().ok()?);
        let cell = match tag {
            KEY_NO_CELL => None,
            KEY_WITH_CELL => Some(c.u32()?),
            _ => return None,
        };
        keys.push(Some((superfile, cell, c.u32()?)));
    }
    Some(keys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::vector::hnsw::{HnswParams, Sq16Scorer};

    /// Dimensions of the test centroids.
    const DIM: usize = 16;
    /// Nodes the test graph starts with.
    const START_NODES: usize = 600;
    /// Nodes each generation removes and adds.
    const CHURN: usize = 40;
    /// Generations the reader follows.
    const GENERATIONS: usize = 6;

    /// `count` deterministic unit vectors.
    fn unit_vectors(count: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        (0..count)
            .map(|_| {
                let mut v: Vec<f32> = (0..DIM).map(|_| next()).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect()
    }

    /// The key of the `i`-th test centroid: three superfiles, some with a
    /// cell and some without.
    fn key(i: usize) -> FineNodeKey {
        let superfile = Uuid::from_u128(1 + (i % 3) as u128);
        (
            superfile,
            i.is_multiple_of(2).then_some((i % 7) as u32),
            i as u32,
        )
    }

    /// A writer's graph over the first `nodes` test centroids.
    fn built(vectors: &[Vec<f32>], nodes: usize) -> KeyedGraph {
        let scorer = Sq16Scorer::from_unit_vectors(&vectors[..nodes], DIM);
        KeyedGraph::new(
            (0..nodes).map(|i| Some(key(i))).collect(),
            Hnsw::build_serial(&scorer, HnswParams::default()),
        )
    }

    /// The writer's next generation: `CHURN` live nodes removed and `CHURN`
    /// centroids added after the last slot.
    fn next_generation(graph: &KeyedGraph, vectors: &[Vec<f32>], round: usize) -> KeyedGraph {
        let slots = graph.graph.len();
        let scorer = Sq16Scorer::from_unit_vectors(&vectors[..slots], DIM);
        let removed: Vec<u32> = (0..slots as u32)
            .filter(|&node| !graph.graph.is_hole(node))
            .skip(round)
            .step_by(slots / CHURN)
            .take(CHURN)
            .collect();
        let mut keys = graph.keys.clone();
        for &node in &removed {
            keys[node as usize] = None;
        }
        keys.extend((slots..slots + CHURN).map(|i| Some(key(i))));
        let scorer_after = Sq16Scorer::from_unit_vectors(&vectors[..slots + CHURN], DIM);
        let hnsw = Hnsw::clone(&graph.graph)
            .unlink_nodes(&removed, &scorer)
            .extend_serial(&scorer_after, HnswParams::default());
        KeyedGraph::new(keys, hnsw)
    }

    /// A reader that has only the stored bytes (the head, the base and each
    /// step) ends with the writer's graph and keys exactly, at every
    /// generation.
    #[test]
    fn a_reader_with_only_the_stored_objects_matches_the_writer() {
        let vectors = unit_vectors(START_NODES + GENERATIONS * CHURN, 0x51A7);
        let mut writer = built(&vectors, START_NODES);
        let base = GraphBase::new(
            writer.keys.iter().flatten().copied().collect(),
            Hnsw::clone(&writer.graph),
        )
        .expect("a fresh build has no holes");
        let base_bytes = base.to_bytes();
        let mut head = GraphHead::of_base(
            GraphRole::Router,
            FieldId(7),
            DIM as u32,
            ContentHash::of(&base_bytes),
            base_bytes.len() as u64,
            &base,
        );
        let mut step_bytes: Vec<Vec<u8>> = Vec::new();

        for round in 0..GENERATIONS {
            let after = next_generation(&writer, &vectors, round);
            let step = GraphStep::between(&writer, &after).expect("a later generation");
            let bytes = step.to_bytes();
            head = head.with_step(ContentHash::of(&bytes), bytes.len() as u64, &step);
            step_bytes.push(bytes);
            writer = after;

            let read_head = GraphHead::from_bytes(&head.to_bytes()).expect("head decodes");
            assert_eq!(read_head, head);
            let read_base = GraphBase::from_bytes(&base_bytes).expect("base decodes");
            let read_steps: Vec<GraphStep> = step_bytes
                .iter()
                .map(|bytes| GraphStep::from_bytes(bytes).expect("step decodes"))
                .collect();
            let reader = read_head
                .replay(read_base, &read_steps)
                .expect("the stored objects replay");
            assert_eq!(reader.keys, writer.keys, "round {round}");
            assert_eq!(
                reader.graph.checksum(),
                writer.graph.checksum(),
                "round {round}"
            );
        }
        assert_eq!(head.steps.len(), GENERATIONS);
    }

    /// A step is far smaller than the base it applies to.
    #[test]
    fn a_step_is_small_beside_its_base() {
        let vectors = unit_vectors(START_NODES + CHURN, 0xB16);
        let writer = built(&vectors, START_NODES);
        let base = GraphBase::new(
            writer.keys.iter().flatten().copied().collect(),
            Hnsw::clone(&writer.graph),
        )
        .expect("base");
        let after = next_generation(&writer, &vectors, 0);
        let step = GraphStep::between(&writer, &after).expect("step");
        assert!(step.to_bytes().len() * 2 < base.to_bytes().len());
    }

    /// A head whose steps do not replay to its checksum is refused, as are
    /// steps in the wrong number, and a step applied to another generation.
    #[test]
    fn steps_that_do_not_fit_are_refused() {
        let vectors = unit_vectors(START_NODES + 2 * CHURN, 0xFAB);
        let first = built(&vectors, START_NODES);
        let second = next_generation(&first, &vectors, 0);
        let third = next_generation(&second, &vectors, 1);
        let to_second = GraphStep::between(&first, &second).expect("step");
        let to_third = GraphStep::between(&second, &third).expect("step");
        let base = GraphBase::new(
            first.keys.iter().flatten().copied().collect(),
            Hnsw::clone(&first.graph),
        )
        .expect("base");
        let hash = ContentHash::of(b"x");
        let head = GraphHead::of_base(GraphRole::Placement, FieldId(1), DIM as u32, hash, 1, &base)
            .with_step(hash, 1, &to_second);

        assert_eq!(
            head.replay(base.clone(), &[])
                .expect_err("a step is missing"),
            GraphStoreError::HeadMismatch
        );
        assert!(matches!(
            head.replay(base.clone(), std::slice::from_ref(&to_third))
                .expect_err("a step for another generation"),
            GraphStoreError::Change(GraphChangeError::WrongBase)
        ));
        assert!(head.replay(base, std::slice::from_ref(&to_second)).is_ok());
    }

    /// A base is a fresh build: a graph with a hole, or with the wrong number
    /// of keys, is not one.
    #[test]
    fn a_base_holds_no_holes() {
        let vectors = unit_vectors(START_NODES + CHURN, 0xBA5E);
        let fresh = built(&vectors, START_NODES);
        let holed = next_generation(&fresh, &vectors, 0);
        let keys: Vec<FineNodeKey> = (0..holed.graph.len()).map(key).collect();
        assert_eq!(
            GraphBase::new(keys, Hnsw::clone(&holed.graph)).expect_err("holes"),
            GraphStoreError::HoleInBase
        );
        assert_eq!(
            GraphBase::new(vec![key(0)], Hnsw::clone(&fresh.graph)).expect_err("one key"),
            GraphStoreError::KeyCount
        );
    }

    /// A fresh base is due once a fifth of the graph has changed, the steps
    /// reach half the base's bytes, or there are too many of them.
    #[test]
    fn a_fresh_base_is_due_when_the_steps_have_grown() {
        const BASE_BYTES: u64 = 1_000;
        let vectors = unit_vectors(START_NODES + 4 * CHURN, 0xF01D);
        let first = built(&vectors, START_NODES);
        let base = GraphBase::new(
            first.keys.iter().flatten().copied().collect(),
            Hnsw::clone(&first.graph),
        )
        .expect("base");
        let hash = ContentHash::of(b"x");
        let head = GraphHead::of_base(
            GraphRole::Router,
            FieldId(1),
            DIM as u32,
            hash,
            BASE_BYTES,
            &base,
        );
        assert!(!head.fold_due());

        let second = next_generation(&first, &vectors, 0);
        let step = GraphStep::between(&first, &second).expect("step");
        // One small step: 80 of 640 slots changed, an eighth.
        let small = head.with_step(hash, 1, &step);
        assert!(!small.fold_due());
        // The same step, stored large.
        assert!(
            head.with_step(hash, BASE_BYTES / FOLD_STEP_BYTES_SHARE_DIVISOR + 1, &step)
                .fold_due()
        );
        // A second step takes the changed share past a fifth.
        let third = next_generation(&second, &vectors, 1);
        let again = GraphStep::between(&second, &third).expect("step");
        assert!(small.with_step(hash, 1, &again).fold_due());
        // Many steps, however small.
        let mut many = head.clone();
        many.steps = vec![(hash, 0); FOLD_MAX_STEPS];
        assert!(many.fold_due());
    }

    /// Truncated or altered bytes decode to nothing, never to a panic.
    #[test]
    fn damaged_objects_do_not_decode() {
        let vectors = unit_vectors(START_NODES + CHURN, 0xDEAD);
        let first = built(&vectors, START_NODES);
        let second = next_generation(&first, &vectors, 0);
        let base = GraphBase::new(
            first.keys.iter().flatten().copied().collect(),
            Hnsw::clone(&first.graph),
        )
        .expect("base");
        let step = GraphStep::between(&first, &second).expect("step");
        let hash = ContentHash::of(b"x");
        let head = GraphHead::of_base(GraphRole::Router, FieldId(1), DIM as u32, hash, 1, &base)
            .with_step(hash, 1, &step);

        for bytes in [base.to_bytes(), step.to_bytes(), head.to_bytes()] {
            for cut in [0, 4, 8, 9, bytes.len() / 2, bytes.len() - 1] {
                let cut = &bytes[..cut];
                assert!(GraphBase::from_bytes(cut).is_none());
                assert!(GraphStep::from_bytes(cut).is_none());
                assert!(GraphHead::from_bytes(cut).is_none());
            }
        }
        let mut wrong_magic = head.to_bytes();
        wrong_magic[0] ^= 0xFF;
        assert!(GraphHead::from_bytes(&wrong_magic).is_none());
        let mut trailing = head.to_bytes();
        trailing.push(0);
        assert!(GraphHead::from_bytes(&trailing).is_none());
    }
}
