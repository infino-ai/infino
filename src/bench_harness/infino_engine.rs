// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! infino reference implementation of [`FtsEngine`].
//!
//! Measures infino exactly as an API consumer uses it: build a unified
//! `.parquet` superfile through [`SuperfileBuilder`], then query the
//! embedded BM25 index through [`SuperfileReader`]. No internal hooks —
//! the same public surface any downstream user calls.

use std::sync::Arc;

use arrow_array::{Decimal128Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;

use super::{BoolMode, Capabilities, FtsEngine, Hit};
use crate::superfile::SuperfileReader;
use crate::superfile::builder::{BuilderOptions, FtsConfig, SuperfileBuilder};
use crate::superfile::fts::reader::BoolMode as InfinoBoolMode;
use crate::superfile::fts::tokenize::{AsciiLowerTokenizer, Tokenizer};

/// Auto-injected primary-key column for the superfile schema.
const ID_COLUMN: &str = "doc_id";

/// Rows per `add_batch` — bounds the transient RecordBatch footprint
/// during ingest, mirroring the production commit path.
const WRITE_CHUNK: usize = 65_536;

/// infino as a comparison engine.
pub struct InfinoFtsEngine;

/// Sealed infino FTS index: the opened `SuperfileReader` over the
/// finished `.parquet` bytes, plus the indexed column name.
pub struct InfinoFtsIndex {
    column: String,
    reader: Option<SuperfileReader>,
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

    fn open(column: &str) -> Self::Index {
        InfinoFtsIndex {
            column: column.to_string(),
            reader: None,
        }
    }

    fn write(index: &mut Self::Index, docs: &[(u64, &str)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new(ID_COLUMN, DataType::Decimal128(38, 0), false),
            Field::new(index.column.as_str(), DataType::LargeUtf8, false),
        ]));
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(AsciiLowerTokenizer::new());
        let opts = BuilderOptions::new(
            schema.clone(),
            ID_COLUMN,
            vec![FtsConfig {
                column: index.column.clone(),
            }],
            vec![],
            Some(tokenizer),
        );
        let mut builder = SuperfileBuilder::new(opts).expect("SuperfileBuilder::new");
        for chunk in docs.chunks(WRITE_CHUNK) {
            let ids: Decimal128Array = chunk
                .iter()
                .map(|(id, _)| Some(*id as i128))
                .collect::<Decimal128Array>()
                .with_precision_and_scale(38, 0)
                .expect("decimal128 precision/scale");
            let texts =
                LargeStringArray::from(chunk.iter().map(|(_, t)| *t).collect::<Vec<&str>>());
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(texts)])
                .expect("RecordBatch");
            builder.add_batch(&batch, &[]).expect("add_batch");
        }
        let bytes = builder.finish().expect("SuperfileBuilder::finish");
        index.reader =
            Some(SuperfileReader::open(Bytes::from(bytes)).expect("open SuperfileReader"));
    }

    fn read(index: &Self::Index, terms: &[&str], k: usize, mode: BoolMode) -> Vec<Hit> {
        let reader = index.reader.as_ref().expect("read called before write");
        let infino_mode = match mode {
            BoolMode::Or => InfinoBoolMode::Or,
            BoolMode::And => InfinoBoolMode::And,
        };
        let hits = futures::executor::block_on(reader.bm25_search_pretokenized(
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_write_read_roundtrip() {
        let mut idx = InfinoFtsEngine::open("title");
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
        assert!(
            !ids.contains(&1),
            "doc 1 has no 'quick'; got {ids:?}"
        );

        // AND of two terms only matches the doc containing both.
        let and_hits = InfinoFtsEngine::read(&idx, &["quick", "fox"], 10, BoolMode::And);
        let and_ids: Vec<u64> = and_hits.iter().map(|h| h.doc_id).collect();
        assert_eq!(and_ids, vec![0], "only doc 0 has both 'quick' and 'fox': {and_ids:?}");
    }
}
