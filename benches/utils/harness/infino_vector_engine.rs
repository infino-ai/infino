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
        let ids: Decimal128Array = ((id_base + offset) as u64..(id_base + offset + len) as u64)
            .map(|i| Some(i as i128))
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
        hits.into_iter()
            .map(|(doc_id, distance)| VectorHit {
                doc_id: u64::from(doc_id),
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
        let existing = index
            .source_vectors
            .as_ref()
            .expect("insert requires InfinoVectorIndex::source_vectors retained from write()");
        // Ids are always 0..n_docs by construction (see `build_superfile`'s
        // `id_base`), so `next_id` is validated rather than threaded
        // through — this reference impl does not support inserting at an
        // arbitrary sparse id.
        let expected_next_id = (existing.len() / index.dim) as u64;
        assert_eq!(
            next_id, expected_next_id,
            "InfinoVectorEngine::insert only supports appending at the current doc count"
        );
        let mut combined = existing.clone();
        combined.extend_from_slice(vectors);
        let rebuilt = build_superfile(&index.column, &combined, index.dim, index.metric, 0);
        index.reader = Some(
            SuperfileReader::open(Bytes::from(rebuilt.clone())).expect("open SuperfileReader"),
        );
        index.bytes = Some(rebuilt);
        index.source_vectors = Some(combined);
        true
    }

    /// Same honesty caveat as `insert`: no remove-by-id exists at the
    /// superfile tier. This filters the retained source vectors by id and
    /// rebuilds — a full rebuild minus the removed rows, not an in-place
    /// tombstone. `ids` are the `0..n_docs` positional ids `write`/`insert`
    /// assign, so this filters by index position directly.
    fn remove(index: &mut Self::Index, ids: &[u64]) -> bool {
        let existing = index
            .source_vectors
            .as_ref()
            .expect("remove requires InfinoVectorIndex::source_vectors retained from write()");
        let dim = index.dim;
        let drop_set: std::collections::HashSet<u64> = ids.iter().copied().collect();
        let n_docs = existing.len() / dim;
        let mut kept = Vec::with_capacity(existing.len());
        for doc in 0..n_docs {
            if !drop_set.contains(&(doc as u64)) {
                kept.extend_from_slice(&existing[doc * dim..(doc + 1) * dim]);
            }
        }
        let rebuilt = build_superfile(&index.column, &kept, dim, index.metric, 0);
        index.reader = Some(
            SuperfileReader::open(Bytes::from(rebuilt.clone())).expect("open SuperfileReader"),
        );
        index.bytes = Some(rebuilt);
        index.source_vectors = Some(kept);
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
        })
    }
}
