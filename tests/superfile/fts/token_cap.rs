// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A token longer than the cap is chopped into pieces at index time; the
//! query side must chop the same way, or a long token is looked up as a
//! term the index never wrote and silently matches nothing. Checked end
//! to end through the superfile for both analyzers.

use std::{collections::HashSet, sync::Arc};

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    superfile::{
        SuperfileReader,
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        fts::{reader::BoolMode, tokenize::MAX_TOKEN_CHARS},
    },
    test_helpers::decimal128_ids,
};

const K_ALL: usize = 100;
/// Longer than one piece, shorter than two.
const LONG_RUN: usize = MAX_TOKEN_CHARS + 45;

fn build(analyzer: &str, docs: &[&str]) -> SuperfileReader {
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig::new("title").analyzer(analyzer)],
        vec![],
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
    let ids = decimal128_ids(0..docs.len() as u64);
    let titles = LargeStringArray::from(docs.to_vec());
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)])
        .expect("build RecordBatch");
    b.add_batch(&batch, &[]).expect("add_batch");
    SuperfileReader::open(Bytes::from(b.finish().expect("finish"))).expect("open")
}

async fn hits(r: &SuperfileReader, query: &str) -> HashSet<u64> {
    r.bm25_hits_async("title", query, K_ALL, BoolMode::Or)
        .await
        .expect("search")
        .iter()
        .map(|(d, _)| *d as u64)
        .collect()
}

#[tokio::test]
async fn a_token_past_the_cap_is_found_by_the_same_token_in_a_query() {
    let run_a = "a".repeat(LONG_RUN);
    let run_b = "b".repeat(LONG_RUN);
    let docs = [
        format!("{run_a} shared"),
        format!("{run_b} shared"),
        "short shared".to_string(),
    ];
    let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
    for analyzer in ["ascii_lower", "standard"] {
        let r = build(analyzer, &refs);
        assert_eq!(
            hits(&r, &run_a).await,
            HashSet::from([0]),
            "{analyzer}: the whole run"
        );
        assert_eq!(
            hits(&r, &run_a.to_ascii_uppercase()).await,
            HashSet::from([0]),
            "{analyzer}: case-folded and chopped alike"
        );
        assert_eq!(
            hits(&r, &run_b).await,
            HashSet::from([1]),
            "{analyzer}: the other run"
        );
        // Each piece is a real term of its own.
        assert_eq!(
            hits(&r, &run_a[..MAX_TOKEN_CHARS]).await,
            HashSet::from([0]),
            "{analyzer}: the first piece"
        );
        assert_eq!(
            hits(&r, &run_a[MAX_TOKEN_CHARS..]).await,
            HashSet::from([0]),
            "{analyzer}: the remainder piece"
        );
        assert_eq!(hits(&r, "shared").await, HashSet::from([0, 1, 2]));
    }
}
