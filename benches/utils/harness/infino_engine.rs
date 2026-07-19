// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! infino reference implementation of [`FtsEngine`].
//!
//! Measures infino exactly as an API consumer uses it: build a unified
//! `.parquet` superfile through [`SuperfileBuilder`], then query the
//! embedded BM25 index through [`SuperfileReader`]. No internal hooks —
//! the same public surface any downstream user calls, and the same
//! builder/tokenizer the in-tree `superfile::fts` bench uses.

use std::sync::Arc;

use arrow_array::{Decimal128Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    superfile::{
        SuperfileReader,
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        fts::reader::BoolMode as InfinoBoolMode,
    },
    test_helpers::default_tokenizer,
};
use rayon::prelude::*;

use super::{BoolMode, Capabilities, FtsEngine, Hit};
use crate::corpus::block_on_inmem;

/// Auto-injected primary-key column for the superfile schema.
const ID_COLUMN: &str = "doc_id";

/// Rows per `add_batch` — bounds the transient RecordBatch footprint
/// during ingest, mirroring the production commit path.
const WRITE_CHUNK: usize = 65_536;

/// Build one superfile (`.parquet` + embedded FTS blob) from `docs` with
/// a single builder, returning the finished bytes. Shared by the
/// queryable `write` and the build-throughput probes. `positions`
/// selects the FTS config: the queryable artifact is positional (the
/// warm battery includes phrase queries), while the default-config
/// build rows measure `positions: false` — what a table gets unless it
/// opts in.
fn build_superfile(column: &str, docs: &[(u64, &str)], positions: bool) -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(ID_COLUMN, DataType::Decimal128(38, 0), false),
        Field::new(column, DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        ID_COLUMN,
        vec![FtsConfig {
            column: column.to_string(),
            positions,
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut builder = SuperfileBuilder::new(opts).expect("SuperfileBuilder::new");
    for chunk in docs.chunks(WRITE_CHUNK) {
        let ids: Decimal128Array = chunk
            .iter()
            .map(|(id, _)| Some(*id as i128))
            .collect::<Decimal128Array>()
            .with_precision_and_scale(38, 0)
            .expect("decimal128 precision/scale");
        let texts = LargeStringArray::from(chunk.iter().map(|(_, t)| *t).collect::<Vec<&str>>());
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(texts)])
            .expect("RecordBatch");
        builder.add_batch(&batch, &[]).expect("add_batch");
    }
    builder.finish().expect("SuperfileBuilder::finish")
}

/// Build-only run at `writers` parallelism: shard the corpus across
/// `writers` builders, each emitting its own superfile (the same
/// sharded-ingest shape a partitioned commit produces); bytes
/// discarded. `writers <= 1` is a plain single-builder run.
fn parallel_build(column: &str, docs: &[(u64, &str)], writers: usize, positions: bool) {
    if writers <= 1 {
        std::hint::black_box(build_superfile(column, docs, positions));
        return;
    }
    let shard_len = docs.len().div_ceil(writers);
    let shards: Vec<Vec<u8>> = docs
        .par_chunks(shard_len)
        .map(|shard| build_superfile(column, shard, positions))
        .collect();
    std::hint::black_box(shards);
}

/// Single-writer build at the default FTS config (`positions: false`),
/// returning the finished bytes so the caller can report the
/// default-config stored size alongside the timing.
pub fn build_positionless(column: &str, docs: &[(u64, &str)]) -> Vec<u8> {
    build_superfile(column, docs, false)
}

/// Build-only throughput probe at the default FTS config
/// (`positions: false`) and `writers` parallelism.
pub fn parallel_build_positionless(column: &str, docs: &[(u64, &str)], writers: usize) {
    parallel_build(column, docs, writers, false);
}

/// infino as a comparison engine.
pub struct InfinoFtsEngine;

/// Sealed infino FTS index: the opened `SuperfileReader` over the
/// finished `.parquet` bytes, plus the indexed column name.
pub struct InfinoFtsIndex {
    column: String,
    bytes: Option<Vec<u8>>,
    reader: Option<SuperfileReader>,
}

impl InfinoFtsIndex {
    /// Bytes produced by the measured 1-writer build. Used by infino's
    /// own cold-tier bench to upload the exact artifact that was built
    /// and searched, not a rebuilt copy.
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_deref().expect("bytes requested before write")
    }

    /// Reader opened on the measured 1-writer artifact.
    pub fn reader(&self) -> &SuperfileReader {
        self.reader.as_ref().expect("reader requested before write")
    }
}

impl FtsEngine for InfinoFtsEngine {
    type Index = InfinoFtsIndex;

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

    fn create(column: &str) -> Self::Index {
        InfinoFtsIndex {
            column: column.to_string(),
            bytes: None,
            reader: None,
        }
    }

    fn write(index: &mut Self::Index, docs: &[(u64, &str)]) {
        let bytes = build_superfile(&index.column, docs, true);
        index.reader =
            Some(SuperfileReader::open(Bytes::from(bytes.clone())).expect("open SuperfileReader"));
        index.bytes = Some(bytes);
    }

    fn parallel_write(column: &str, docs: &[(u64, &str)], writers: usize) {
        parallel_build(column, docs, writers, true);
    }

    fn read(index: &Self::Index, terms: &[&str], k: usize, mode: BoolMode) -> Vec<Hit> {
        let reader = index.reader();
        let infino_mode = match mode {
            BoolMode::Or => InfinoBoolMode::Or,
            BoolMode::And => InfinoBoolMode::And,
        };
        let hits = block_on_inmem(reader.bm25_search_pretokenized(
            index.column.as_str(),
            terms,
            k,
            infino_mode,
        ))
        .expect("bm25 search");
        hits.into_iter()
            .map(|(doc_id, score)| Hit {
                doc_id: u64::from(doc_id),
                score,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_write_read_roundtrip() {
        let mut idx = InfinoFtsEngine::create("title");
        let docs: [(u64, &str); 3] = [
            (0, "the quick brown fox"),
            (1, "a lazy sleeping dog"),
            (2, "quick foxes leap"),
        ];
        InfinoFtsEngine::write(&mut idx, &docs);

        let hits = InfinoFtsEngine::read(&idx, &["quick"], 10, BoolMode::Or);
        let ids: Vec<u64> = hits.iter().map(|h| h.doc_id).collect();
        assert!(
            ids.contains(&0) && ids.contains(&2),
            "docs 0 and 2 contain 'quick'; got {ids:?}"
        );
        assert!(!ids.contains(&1), "doc 1 has no 'quick'; got {ids:?}");

        // AND of two terms only matches the doc containing both.
        let and_hits = InfinoFtsEngine::read(&idx, &["quick", "fox"], 10, BoolMode::And);
        let and_ids: Vec<u64> = and_hits.iter().map(|h| h.doc_id).collect();
        assert_eq!(
            and_ids,
            vec![0],
            "only doc 0 has both 'quick' and 'fox': {and_ids:?}"
        );
    }
}
