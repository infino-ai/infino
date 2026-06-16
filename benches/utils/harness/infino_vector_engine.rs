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
use rayon::prelude::*;

use infino::superfile::SuperfileReader;
use infino::superfile::builder::{BuilderOptions, SuperfileBuilder, VectorConfig};
use infino::superfile::reader::VectorSearchOptions;
use infino::superfile::vector::distance::Metric as InfinoMetric;
use infino::superfile::vector::rerank_codec::RerankCodec;

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
    n_cent: usize,
    id_base: usize,
) -> Vec<u8> {
    let n_docs = vectors.len() / dim;
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
            column: column.into(),
            dim,
            n_cent,
            rot_seed: ROT_SEED,
            metric: map_metric(metric),
            rerank_codec: RerankCodec::Sq8ResidualEpsilon,
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
    n_cent: usize,
    bytes: Option<Vec<u8>>,
    reader: Option<SuperfileReader>,
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
        }
    }

    fn create(column: &str, dim: usize, metric: VectorMetric, n_cent: usize) -> Self::Index {
        InfinoVectorIndex {
            column: column.to_string(),
            dim,
            metric,
            n_cent,
            bytes: None,
            reader: None,
        }
    }

    fn write(index: &mut Self::Index, vectors: &[f32]) {
        let bytes = build_superfile(
            &index.column,
            vectors,
            index.dim,
            index.metric,
            index.n_cent,
            0,
        );
        index.reader =
            Some(SuperfileReader::open(Bytes::from(bytes.clone())).expect("open SuperfileReader"));
        index.bytes = Some(bytes);
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
            let n_docs = vectors.len() / dim;
            std::hint::black_box(build_superfile(
                column,
                vectors,
                dim,
                metric,
                corpus::n_cent(n_docs),
                0,
            ));
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
                    corpus::n_cent(len_docs),
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
}
