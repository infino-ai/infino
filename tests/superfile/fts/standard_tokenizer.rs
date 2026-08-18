// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! BM25 oracle for the `standard` (UAX#29) analyzer.
//!
//! Every other oracle in this suite builds and queries under the
//! default `ascii_lower` tokenizer. This module builds the superfile
//! *and* indexes the brute-force reference under [`StandardTokenizer`],
//! over a corpus whose text (non-ASCII letters, punctuation, digits)
//! the two analyzers segment differently — so the reader's standard
//! query+doc tokenization and scoring are pinned against a reference
//! that tokenizes the same way, and a bug specific to the standard path
//! (wrong analyzer selected, query tokenized under a different analyzer
//! than the docs) diverges from the oracle.

use std::{collections::HashSet, sync::Arc};

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    superfile::{
        SuperfileReader,
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        fts::{
            reader::BoolMode,
            tokenize::{StandardTokenizer, Tokenizer},
        },
    },
    test_helpers::{brute_force_bm25::BruteForceBm25, decimal128_ids},
};

const SCORE_ABS_TOLERANCE: f32 = 1e-3;
const K_ALL: usize = 64;

/// Build a single-FTS-column superfile under the `standard` analyzer.
/// `doc_id` == row index.
fn build_standard(corpus: &[(u64, &str)]) -> SuperfileReader {
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![],
        Some(Arc::new(StandardTokenizer)),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
    let ids = decimal128_ids(corpus.iter().map(|(i, _)| *i));
    let titles = LargeStringArray::from(corpus.iter().map(|(_, t)| *t).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)])
        .expect("build RecordBatch");
    b.add_batch(&batch, &[]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    SuperfileReader::open(bytes).expect("open superfile")
}

/// Corpus with text the `standard` analyzer segments differently from
/// `ascii_lower`: accented letters, apostrophes, digits, punctuation.
fn corpus() -> Vec<(u64, &'static str)> {
    vec![
        (0, "rust async runtime"),
        (1, "café résumé naïve"), // non-ASCII: dropped entirely by ascii_lower
        (2, "the rust programming café"), // "café" co-occurs with "rust"
        (3, "version 2 point 0 release"),
        (4, "rust systems programming"),
        (5, "async await futures rust"),
        (6, "python data café pipeline"), // "café" without "rust"
        (7, "naïve bayes classifier"),    // "naïve" again
    ]
}

/// Reader vs standard-tokenizer oracle for `query`: identical match set
/// (at k=all) and per-doc scores within tolerance.
async fn assert_matches_oracle(
    reader: &SuperfileReader,
    oracle: &BruteForceBm25,
    query: &str,
    mode: BoolMode,
) {
    let got: Vec<(u64, f32)> = reader
        .bm25_hits_async("title", query, K_ALL, mode)
        .await
        .expect("bm25 search")
        .into_iter()
        .map(|(d, s)| (d as u64, s))
        .collect();
    // Score the same clauses the reader parsed, so the comparison can't
    // diverge on tokenization of the query string itself.
    let tok = StandardTokenizer;
    let clauses = tok.parse(query).into_clauses(mode);
    let own = |v: Vec<std::borrow::Cow<'_, str>>| -> Vec<String> {
        v.into_iter().map(|t| t.into_owned()).collect()
    };
    let want = oracle.top_k_clauses(
        &own(clauses.musts),
        &own(clauses.shoulds),
        &own(clauses.negatives),
        K_ALL,
    );

    let got_ids: HashSet<u64> = got.iter().map(|(d, _)| *d).collect();
    let want_ids: HashSet<u64> = want.iter().map(|(d, _)| *d).collect();
    assert_eq!(
        got_ids, want_ids,
        "standard {query:?} {mode:?}: match sets disagree"
    );
    let want_scores: std::collections::HashMap<u64, f32> = want.into_iter().collect();
    for (d, s) in &got {
        let w = want_scores[d];
        assert!(
            (s - w).abs() <= SCORE_ABS_TOLERANCE,
            "standard {query:?} doc {d}: reader={s} oracle={w}"
        );
    }
}

#[tokio::test]
async fn standard_tokenizer_agrees_with_oracle() {
    let corp = corpus();
    let reader = build_standard(&corp);
    let oracle = BruteForceBm25::index(&corp, &StandardTokenizer as &dyn Tokenizer);
    for (query, mode) in [
        ("rust", BoolMode::Or),
        ("café", BoolMode::Or),
        ("rust café", BoolMode::Or),
        ("rust café", BoolMode::And),
        ("naïve", BoolMode::Or),
        ("+rust café", BoolMode::Or),
        ("rust -café", BoolMode::Or),
    ] {
        assert_matches_oracle(&reader, &oracle, query, mode).await;
    }
}

#[tokio::test]
async fn standard_tokenizer_indexes_non_ascii_terms() {
    // Discriminating check that the standard analyzer is actually in
    // force: "café" and "naïve" contain non-ASCII bytes that the default
    // `ascii_lower` tokenizer drops entirely. Under `standard` they are
    // real, searchable terms — so a non-empty result here proves the
    // build/query path used the standard analyzer, not a silent
    // ascii_lower fallback.
    let corp = corpus();
    let reader = build_standard(&corp);

    let cafe = reader
        .bm25_hits_async("title", "café", K_ALL, BoolMode::Or)
        .await
        .expect("search café");
    let cafe_ids: HashSet<u64> = cafe.iter().map(|(d, _)| *d as u64).collect();
    assert_eq!(
        cafe_ids,
        HashSet::from([1, 2, 6]),
        "café is indexed as a term under the standard analyzer"
    );

    let naive = reader
        .bm25_hits_async("title", "naïve", K_ALL, BoolMode::Or)
        .await
        .expect("search naïve");
    let naive_ids: HashSet<u64> = naive.iter().map(|(d, _)| *d as u64).collect();
    assert_eq!(
        naive_ids,
        HashSet::from([1, 7]),
        "naïve indexed under standard"
    );
}
