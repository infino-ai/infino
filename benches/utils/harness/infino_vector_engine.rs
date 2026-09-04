// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Infino reference implementation of [`VectorEngine`].
//!
//! The canonical `write` builds one unified superfile through
//! `SuperfileBuilder`, opens a `SuperfileReader`, and retains both the
//! bytes and the reader. In-tree benches use those retained bytes for
//! cold upload and the retained reader for correctness/warm search.

use std::sync::Arc;

use arrow_array::{Decimal128Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::superfile::{
    SuperfileReader,
    builder::{BuilderOptions, SuperfileBuilder, VectorConfig},
    reader::VectorSearchOptions,
    vector::distance::Metric as InfinoMetric,
};
use rayon::prelude::*;

use super::{Capabilities, VectorEngine, VectorHit, VectorMetric, VectorSearch};
use crate::corpus::{self, block_on_inmem};

const ID_COLUMN: &str = "doc_id";
const WRITE_CHUNK: usize = 65_536;
const ROT_SEED: u64 = 7;

fn map_metric(metric: VectorMetric) -> InfinoMetric {
    match metric {
        VectorMetric::L2Sq => InfinoMetric::L2Sq,
        VectorMetric::Cosine => InfinoMetric::Cosine,
        VectorMetric::NegDot => InfinoMetric::NegDot,
    }
}

fn build_superfile(
    column: &str,
    vectors: &[f32],
    dim: usize,
    metric: VectorMetric,
    id_base: usize,
) -> Vec<u8> {
    let n_docs = vectors.len() / dim;
    let ids: Vec<u64> = (id_base as u64..(id_base + n_docs) as u64).collect();
    build_superfile_with_ids(column, vectors, dim, metric, &ids)
}

/// [`build_superfile`] with explicit per-row `_id`s — the engine's own
/// stable-id mechanism (the `_id` column every superfile carries), used by
/// the insert/remove rebuilds so surviving rows keep the ids the caller
/// already holds instead of being renumbered positionally.
fn build_superfile_with_ids(
    column: &str,
    vectors: &[f32],
    dim: usize,
    metric: VectorMetric,
    row_ids: &[u64],
) -> Vec<u8> {
    let n_docs = vectors.len() / dim;
    assert_eq!(row_ids.len(), n_docs, "one _id per row");
    let metric = map_metric(metric);
    let schema = Arc::new(Schema::new(vec![Field::new(
        ID_COLUMN,
        DataType::Decimal128(38, 0),
        false,
    )]));
    let opts = BuilderOptions::new(
        schema.clone(),
        ID_COLUMN,
        vec![],
        vec![VectorConfig {
            provided_centroids: None,
            column: column.into(),
            dim,
            rot_seed: ROT_SEED,
            metric,
            rerank_codec: corpus::bench_rerank_codec(metric),
        }],
        None,
    );
    let mut builder = SuperfileBuilder::new(opts).expect("SuperfileBuilder::new");
    let mut offset = 0;
    while offset < n_docs {
        let len = WRITE_CHUNK.min(n_docs - offset);
        let ids: Decimal128Array = row_ids[offset..offset + len]
            .iter()
            .map(|&i| Some(i as i128))
            .collect::<Decimal128Array>()
            .with_precision_and_scale(38, 0)
            .expect("decimal128 precision/scale");
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids)]).expect("RecordBatch");
        builder
            .add_batch(&batch, &[&vectors[offset * dim..(offset + len) * dim]])
            .expect("add_batch");
        offset += len;
    }
    builder.finish().expect("SuperfileBuilder::finish")
}

/// Every row's stable `_id`, in local-doc order, read from the artifact's
/// own id column — the same local→stable resolution the supertable layer
/// performs (`take_by_local_doc_ids` on the id column). The superfile IS
/// the id map; no side state to drift.
fn stable_ids_of(reader: &SuperfileReader) -> Vec<u64> {
    let n_docs = reader.n_docs() as u32;
    let locals: Vec<u32> = (0..n_docs).collect();
    let batch = reader
        .take_by_local_doc_ids(&locals, &[reader.id_column()])
        .expect("read the _id column");
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("_id column is Decimal128");
    (0..ids.len()).map(|i| ids.value(i) as u64).collect()
}

pub struct InfinoVectorEngine;

pub struct InfinoVectorIndex {
    column: String,
    dim: usize,
    metric: VectorMetric,
    bytes: Option<Vec<u8>>,
    reader: Option<SuperfileReader>,
    /// Retained fp32 source vectors, kept ONLY so `insert`/`remove` can
    /// rebuild the superfile from an updated corpus — infino superfiles
    /// are immutable once `finish()` is called (see
    /// `src/superfile/builder.rs`), so there is no in-place add/remove at
    /// this tier. Retaining the full fp32 source is a bench-only cost no
    /// shipping caller would pay; `load` deliberately leaves this `None`
    /// since a loaded index has no fp32 source to reconstruct from.
    source_vectors: Option<Vec<f32>>,
    /// Whether insert/remove ever ran. A freshly written artifact has
    /// dense `_id`s equal to local ids, so `read` skips the id resolve on
    /// the unmutated path and its timing is unchanged; after a mutation,
    /// hits resolve through the artifact's id column like any caller's
    /// would.
    mutated: bool,
}

impl InfinoVectorIndex {
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_deref().expect("bytes requested before write")
    }

    pub fn reader(&self) -> &SuperfileReader {
        self.reader.as_ref().expect("reader requested before write")
    }
}

impl VectorEngine for InfinoVectorEngine {
    type Index = InfinoVectorIndex;

    fn name() -> &'static str {
        "infino"
    }

    fn capabilities() -> Capabilities {
        Capabilities {
            fts: true,
            vector: true,
            sql: true,
            hybrid: true,
            // Honest but heavy: infino has no in-place insert/remove at
            // the superfile tier, so both are implemented as a full
            // incremental rebuild — see `insert`/`remove` below.
            vector_insert: true,
            vector_remove: true,
            // Genuinely cheap and native: `finish()` already returns
            // final bytes, `SuperfileReader::open` already reopens them.
            vector_save_load: true,
        }
    }

    fn create(column: &str, dim: usize, metric: VectorMetric) -> Self::Index {
        InfinoVectorIndex {
            column: column.to_string(),
            dim,
            metric,
            bytes: None,
            reader: None,
            source_vectors: None,
            mutated: false,
        }
    }

    fn write(index: &mut Self::Index, vectors: &[f32]) {
        let bytes = build_superfile(&index.column, vectors, index.dim, index.metric, 0);
        index.reader =
            Some(SuperfileReader::open(Bytes::from(bytes.clone())).expect("open SuperfileReader"));
        index.bytes = Some(bytes);
        index.source_vectors = Some(vectors.to_vec());
    }

    fn parallel_write(
        column: &str,
        vectors: &[f32],
        dim: usize,
        metric: VectorMetric,
        writers: usize,
    ) {
        let writers = writers.max(1);
        if writers == 1 {
            std::hint::black_box(build_superfile(column, vectors, dim, metric, 0));
            return;
        }
        let n_docs = vectors.len() / dim;
        let docs_per_shard = n_docs.div_ceil(writers);
        let shards: Vec<Vec<u8>> = (0..writers)
            .into_par_iter()
            .filter_map(|shard| {
                let start_doc = shard * docs_per_shard;
                if start_doc >= n_docs {
                    return None;
                }
                let len_docs = docs_per_shard.min(n_docs - start_doc);
                let start = start_doc * dim;
                let end = (start_doc + len_docs) * dim;
                Some(build_superfile(
                    column,
                    &vectors[start..end],
                    dim,
                    metric,
                    start_doc,
                ))
            })
            .collect();
        std::hint::black_box(shards);
    }

    fn read(index: &Self::Index, query: &[f32], k: usize, search: VectorSearch) -> Vec<VectorHit> {
        let opts = VectorSearchOptions::new()
            .with_nprobe(search.nprobe)
            .with_rerank_mult(search.rerank_mult);
        let hits = block_on_inmem(
            index
                .reader()
                .vector_hits_async(&index.column, query, k, opts),
        )
        .expect("vector_search");
        if !index.mutated {
            // Freshly written artifacts have `_id == local` by
            // construction; skip the resolve so the mainline cells'
            // timing is byte-identical to before mutations existed.
            return hits
                .into_iter()
                .map(|(doc_id, distance)| VectorHit {
                    doc_id: u64::from(doc_id),
                    distance,
                })
                .collect();
        }
        // After a mutation, locals are renumbered by the rebuild; resolve
        // to stable `_id`s through the artifact's own id column, the same
        // resolution any caller's hits go through.
        let locals: Vec<u32> = hits.iter().map(|(doc_id, _)| *doc_id).collect();
        let batch = index
            .reader()
            .take_by_local_doc_ids(&locals, &[index.reader().id_column()])
            .expect("resolve hit _ids");
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id column is Decimal128");
        hits.into_iter()
            .enumerate()
            .map(|(i, (_, distance))| VectorHit {
                doc_id: ids.value(i) as u64,
                distance,
            })
            .collect()
    }

    fn close(index: &mut Self::Index) {
        index.reader = None;
    }

    fn delete(_index: Self::Index) {
        // Dropping the in-memory bytes/reader releases the artifact.
    }

    /// NOT a true append-to-served-index. Infino superfiles are
    /// immutable once `finish()` is called — there is no insert-after
    /// -seal operation anywhere in the crate. What this measures instead
    /// is the cost of growing the corpus by `vectors.len() / dim` rows and
    /// re-sealing from scratch: it appends to the retained fp32 source
    /// (see [`InfinoVectorIndex::source_vectors`]) and re-`finish()`s a
    /// fresh superfile. This is the honest number for "how much does
    /// growing an infino superfile by n rows cost", which is a different
    /// (and for infino, much heavier) question than "insert latency"
    /// implies for a mutable-index engine.
    fn insert(index: &mut Self::Index, vectors: &[f32], next_id: u64) -> bool {
        // A LOADED index retains no fp32 source, so the rebuild that
        // implements mutation here is impossible for it. Static
        // capabilities cannot express state-dependent support; `false` is
        // the trait's channel for it, and callers assert on the return.
        let Some(existing) = index.source_vectors.as_ref() else {
            return false;
        };
        // The artifact's own `_id` column is the id authority: new rows get
        // `next_id..`, survivors keep the ids they already carry, and the
        // trait's uniqueness contract is checked against the real ids
        // rather than assumed from a positional count.
        let mut row_ids = stable_ids_of(index.reader());
        let max_id = row_ids.iter().copied().max().unwrap_or(0);
        assert!(
            row_ids.is_empty() || next_id > max_id,
            "InfinoVectorEngine::insert: next_id {next_id} must exceed the max stored _id {max_id}"
        );
        assert!(
            vectors.len().is_multiple_of(index.dim),
            "insert buffer must be whole rows: {} floats at dim {}",
            vectors.len(),
            index.dim
        );
        let added = vectors.len() / index.dim;
        // A `next_id` near u64::MAX would make this range wrap EMPTY (not
        // duplicate ids), and the builder's one-`_id`-per-row assert then
        // fails loudly on the length mismatch — no silent state exists.
        row_ids.extend(next_id..next_id + added as u64);
        let mut combined = existing.clone();
        combined.extend_from_slice(vectors);
        let rebuilt =
            build_superfile_with_ids(&index.column, &combined, index.dim, index.metric, &row_ids);
        index.reader = Some(
            SuperfileReader::open(Bytes::from(rebuilt.clone())).expect("open SuperfileReader"),
        );
        index.bytes = Some(rebuilt);
        index.source_vectors = Some(combined);
        index.mutated = true;
        true
    }

    /// Same honesty caveat as `insert`: no remove-by-id exists at the
    /// superfile tier. This filters the retained source vectors and
    /// rebuilds — a full rebuild minus the removed rows, not an in-place
    /// tombstone. `ids` name stable `_id`s (the artifact's own id column),
    /// and survivors KEEP their ids across the rebuild, so a later
    /// insert/remove/search still means the rows the caller thinks it
    /// means.
    fn remove(index: &mut Self::Index, ids: &[u64]) -> bool {
        // Same state-dependent unsupport as `insert`: no retained source,
        // no rebuild.
        let Some(existing) = index.source_vectors.as_ref() else {
            return false;
        };
        let dim = index.dim;
        let drop_set: std::collections::HashSet<u64> = ids.iter().copied().collect();
        let row_ids = stable_ids_of(index.reader());
        let n_docs = existing.len() / dim;
        assert_eq!(row_ids.len(), n_docs, "source vectors track the artifact");
        let mut kept = Vec::with_capacity(existing.len());
        let mut kept_ids = Vec::with_capacity(row_ids.len());
        for (doc, &stable) in row_ids.iter().enumerate() {
            if !drop_set.contains(&stable) {
                kept.extend_from_slice(&existing[doc * dim..(doc + 1) * dim]);
                kept_ids.push(stable);
            }
        }
        let rebuilt = build_superfile_with_ids(&index.column, &kept, dim, index.metric, &kept_ids);
        index.reader = Some(
            SuperfileReader::open(Bytes::from(rebuilt.clone())).expect("open SuperfileReader"),
        );
        index.bytes = Some(rebuilt);
        index.source_vectors = Some(kept);
        index.mutated = true;
        true
    }

    /// This one is real: `finish()` already returns final bytes.
    fn save(index: &Self::Index) -> Option<Vec<u8>> {
        Some(index.bytes().to_vec())
    }

    /// This one is real: `SuperfileReader::open` already reopens from
    /// bytes. `source_vectors` is deliberately left `None` — a loaded
    /// index has no fp32 source to reconstruct, so `insert`/`remove`
    /// after `load` will panic until the source is re-supplied by a
    /// caller that tracks it independently.
    fn load(column: &str, dim: usize, metric: VectorMetric, bytes: &[u8]) -> Option<Self::Index> {
        let owned = Bytes::from(bytes.to_vec());
        let reader = SuperfileReader::open(owned.clone()).expect("open SuperfileReader");
        Some(InfinoVectorIndex {
            column: column.to_string(),
            dim,
            metric,
            bytes: Some(owned.to_vec()),
            reader: Some(reader),
            source_vectors: None,
            // Forced on: these bytes may have been saved AFTER an
            // insert/remove, so their `_id`s need not equal locals, and
            // density cannot be checked without an O(n) id-column scan
            // inside the timed load. Resolving hits through the id column
            // is also what production reads do — the identity fast path is
            // provably safe only for artifacts this process built fresh.
            mutated: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows planted on distinct axes so nearest-neighbor identity is exact.
    const TEST_DIM: usize = 16;
    /// Enough rows that removing two from the middle genuinely renumbers
    /// the locals behind them.
    const TEST_ROWS: usize = 8;

    fn axis_rows(n: usize) -> Vec<f32> {
        let mut flat = vec![0.0f32; n * TEST_DIM];
        for row in 0..n {
            flat[row * TEST_DIM + row % TEST_DIM] = 1.0;
        }
        flat
    }

    fn query_for(row: usize) -> Vec<f32> {
        let mut q = vec![0.0f32; TEST_DIM];
        q[row % TEST_DIM] = 1.0;
        q
    }

    fn top1(index: &InfinoVectorIndex, row: usize) -> u64 {
        InfinoVectorEngine::read(
            index,
            &query_for(row),
            1,
            VectorSearch {
                nprobe: usize::MAX,
                rerank_mult: 4,
            },
        )
        .first()
        .expect("one hit")
        .doc_id
    }

    /// The regression the id column exists to prevent: removing middle
    /// rows renumbers locals, but hits and later mutations must keep
    /// speaking stable `_id`s. Before the fix, removing {2, 5} made row 7
    /// answer as 5, and a follow-up remove would have deleted the wrong
    /// rows.
    #[test]
    fn remove_preserves_surviving_ids_and_insert_continues_them() {
        let mut index = InfinoVectorEngine::create("emb", TEST_DIM, VectorMetric::Cosine);
        InfinoVectorEngine::write(&mut index, &axis_rows(TEST_ROWS));
        assert_eq!(top1(&index, 7), 7, "fresh artifact: local == stable");

        assert!(InfinoVectorEngine::remove(&mut index, &[2, 5]));
        assert_eq!(
            top1(&index, 7),
            7,
            "row 7 keeps _id 7 though its local id shrank by two"
        );
        assert_eq!(top1(&index, 3), 3, "row 3 keeps _id 3 behind one removal");

        // The trait's running-counter contract: the caller hands the next
        // unused id, and it lands verbatim.
        let mut extra = vec![0.0f32; TEST_DIM];
        extra[TEST_DIM - 1] = 1.0;
        assert!(InfinoVectorEngine::insert(
            &mut index,
            &extra,
            TEST_ROWS as u64
        ));
        assert_eq!(
            top1(&index, TEST_DIM - 1),
            TEST_ROWS as u64,
            "inserted row answers with the caller-assigned id"
        );

        // Removing by a STABLE id after the renumbering removes the right
        // row: _id 7 (now at a shifted local position) disappears, and the
        // axis-7 query falls to some other row, never a phantom 7.
        assert!(InfinoVectorEngine::remove(&mut index, &[7]));
        assert_ne!(top1(&index, 7), 7, "_id 7 is gone, not renumbered onto");
    }

    /// A save/load round trip after mutations must keep answering with
    /// stable `_id`s (the saved bytes carry non-dense ids), and mutating
    /// the loaded index — which retains no fp32 source — must report
    /// unsupported instead of panicking.
    #[test]
    fn loaded_index_resolves_saved_ids_and_declines_mutation() {
        let mut index = InfinoVectorEngine::create("emb", TEST_DIM, VectorMetric::Cosine);
        InfinoVectorEngine::write(&mut index, &axis_rows(TEST_ROWS));
        assert!(InfinoVectorEngine::remove(&mut index, &[2, 5]));
        let saved = InfinoVectorEngine::save(&index).expect("superfile bytes");

        let mut loaded = InfinoVectorEngine::load("emb", TEST_DIM, VectorMetric::Cosine, &saved)
            .expect("reopen saved bytes");
        assert_eq!(
            top1(&loaded, 7),
            7,
            "the loaded artifact's non-dense _ids resolve, not its locals"
        );
        let extra = vec![0.0f32; TEST_DIM];
        assert!(
            !InfinoVectorEngine::insert(&mut loaded, &extra, TEST_ROWS as u64),
            "no retained source: insert reports unsupported"
        );
        assert!(
            !InfinoVectorEngine::remove(&mut loaded, &[7]),
            "no retained source: remove reports unsupported"
        );
    }
}
