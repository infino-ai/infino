// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! BM25 correctness oracle for a column carrying an **analysis chain**
//! — a stopword set, a stemmer, or both, on top of a base tokenizer.
//!
//! Same discipline as `brute_force_oracle`: the optimized walks are
//! graded against the textbook scorer in
//! [`infino::test_helpers::brute_force_bm25`], which shares no code
//! with them. The chain adds two things that could disagree silently,
//! so both are graded here rather than only asserted against planted
//! truth:
//!
//! * **Doc length.** Removing a stopword shortens the document, which
//!   moves *every* score in the column through the BM25 length
//!   normalizer. If the engine and the oracle counted length
//!   differently — one counting the tokens the chain emitted, the other
//!   the tokens it saw — the ranking would still look plausible and be
//!   wrong. Both count emitted tokens, which is what Lucene's norms
//!   count and what a merge's carried postings preserve.
//! * **Phrase spacing.** A removed token leaves a position hole on both
//!   sides: the index skips its ordinal, and a quoted query carries the
//!   matching offsets. The phrase oracle walks positions for the same
//!   reason, so a hole the engine left and one the query asked for have
//!   to line up exactly.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    superfile::{
        SuperfileReader,
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        fts::{
            reader::BoolMode,
            tokenize::{Phrase, Tokenizer, tokenizer_for_name},
        },
    },
    test_helpers::{brute_force_bm25::BruteForceBm25, decimal128_ids},
};

/// k large enough to capture every match on the corpus below.
const K_ALL: usize = 64;
/// Score-equality tolerance between the two BM25 implementations.
const SCORE_ABS_TOLERANCE: f32 = 1e-3;

/// Prose corpus: stopwords in every position (leading, interior,
/// trailing, consecutive), inflections that stem together, and
/// documents that differ *only* in the stopwords between two content
/// words — which is what makes the phrase holes observable.
fn corpus() -> Vec<(u64, &'static str)> {
    vec![
        (0, "the end of the world is nigh"),
        (1, "end world"),
        (2, "new york city"),
        (3, "new the york city"),
        (4, "running the studies of the mind"),
        (5, "she runs and he ran"),
        (6, "a runner runs the race"),
        (7, "the studies are not conclusive"),
        (8, "study the world"),
        (9, "of the and to be it is"),
        (10, "walking walked walks walk"),
        (11, "the quick brown fox jumps over the lazy dog"),
        (12, "fox and hound"),
        (13, "cities and their studies"),
        (14, "the city studies running"),
        (15, "nothing here at all"),
    ]
}

fn build(analyzer: &str) -> (SuperfileReader, BruteForceBm25, Arc<dyn Tokenizer>) {
    let corp = corpus();
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig::new("title").analyzer(analyzer).positions(true)],
        vec![],
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
    let ids = decimal128_ids(corp.iter().map(|(i, _)| *i));
    let titles = LargeStringArray::from(corp.iter().map(|(_, t)| *t).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)])
        .expect("build RecordBatch");
    b.add_batch(&batch, &[]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let reader = SuperfileReader::open(bytes).expect("open superfile");
    // The oracle indexes the same corpus through the same chain, which
    // is the whole point: grading a chained column against the base
    // tokenizer would compare two different formulas and pass nothing.
    let tok = tokenizer_for_name(analyzer).expect("known analyzer");
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    (reader, oracle, tok)
}

/// Doc-ids a query matches, as a set.
async fn matching(reader: &SuperfileReader, query: &str) -> HashSet<u64> {
    reader
        .bm25_hits_async("title", query, K_ALL, BoolMode::Or)
        .await
        .expect("query")
        .into_iter()
        .map(|(d, _)| d as u64)
        .collect()
}

/// Grade one query's match set *and* per-doc scores against the
/// oracle, through the clause model so phrases in every polarity are
/// covered.
async fn assert_matches_oracle(
    reader: &SuperfileReader,
    oracle: &BruteForceBm25,
    tok: &dyn Tokenizer,
    query: &str,
    mode: BoolMode,
) {
    let clauses = tok.parse(query).into_clauses(mode);
    let own = |v: Vec<std::borrow::Cow<'_, str>>| -> Vec<String> {
        v.into_iter().map(|t| t.into_owned()).collect()
    };
    let own_ph = |v: Vec<Phrase<std::borrow::Cow<'_, str>>>| -> Vec<Phrase<String>> {
        v.iter().map(|p| p.map(|t| t.to_string())).collect()
    };
    let want = oracle.top_k_atoms(
        &own(clauses.musts),
        &own_ph(clauses.must_phrases),
        &own(clauses.shoulds),
        &own_ph(clauses.should_phrases),
        &own(clauses.negatives),
        &own_ph(clauses.negative_phrases),
        K_ALL,
    );
    let got: Vec<(u64, f32)> = reader
        .bm25_hits_async("title", query, K_ALL, mode)
        .await
        .expect("chained-column query")
        .into_iter()
        .map(|(d, s)| (d as u64, s))
        .collect();

    let got_ids: HashSet<u64> = got.iter().map(|(d, _)| *d).collect();
    let want_ids: HashSet<u64> = want.iter().map(|(d, _)| *d).collect();
    assert_eq!(got_ids, want_ids, "query {query:?}: match sets disagree");

    let want_scores: HashMap<u64, f32> = want.into_iter().collect();
    for (doc, score) in got {
        let expected = want_scores[&doc];
        assert!(
            (score - expected).abs() <= SCORE_ABS_TOLERANCE,
            "query {query:?} doc {doc}: score {score} vs oracle {expected}"
        );
    }
}

/// Every chain, every clause shape, graded on scores as well as sets.
/// The scores are the part that pins doc length: a length disagreement
/// keeps the match sets identical and moves every number.
#[tokio::test]
async fn chained_columns_match_the_textbook_scorer() {
    let queries = [
        // Single terms, common and rare.
        "world",
        "studies",
        "running",
        "fox",
        // Unions and conjunctions.
        "world studies",
        "+running +studies",
        "city studies running",
        // Negation.
        "studies -running",
        // Phrases, adjacent and holed.
        "\"new york\"",
        "\"end of the world\"",
        "\"the studies\"",
        "+\"new york\" city",
        "world -\"end of the world\"",
        // Queries made entirely of words a stopword set removes: no
        // clause survives, which must be an empty result and not a
        // disagreement.
        "the and of",
    ];
    for analyzer in [
        "standard+stop=english",
        "standard+stem=english",
        "standard+stop=english+stem=english",
        "ascii_lower+stop=english+stem=english",
    ] {
        let (reader, oracle, tok) = build(analyzer);
        for query in queries {
            for mode in [BoolMode::Or, BoolMode::And] {
                assert_matches_oracle(&reader, &oracle, tok.as_ref(), query, mode).await;
            }
        }
    }
}

/// The hole is load-bearing, not incidental: on a stopworded column
/// `"new york"` must separate two documents that differ only by the
/// stopword between those two words. Asserted against corpus truth
/// rather than the oracle, so a hole bug that happens to affect both
/// implementations the same way still fails here.
#[tokio::test]
async fn phrase_holes_separate_documents_that_differ_only_by_a_stopword() {
    let (reader, _, _) = build("standard+stop=english");
    // Doc 2 is "new york city", doc 3 is "new the york city". Only the
    // first has the two words adjacent.
    assert_eq!(matching(&reader, "\"new york\"").await, HashSet::from([2]));
    // ...and asking for them one apart selects the other one, which is
    // the same mechanism read in the opposite direction.
    assert_eq!(
        matching(&reader, "\"new the york\"").await,
        HashSet::from([3])
    );
    // Doc 0 is "the end of the world is nigh", doc 1 is "end world".
    // The query's own removed words are the spacing it asks for, so it
    // selects doc 0 — and the adjacent spelling selects doc 1.
    assert_eq!(
        matching(&reader, "\"end of the world\"").await,
        HashSet::from([0])
    );
    assert_eq!(matching(&reader, "\"end world\"").await, HashSet::from([1]));
}

/// A stemmed column folds inflections onto one term, and the effect is
/// symmetric: any spelling of a word finds every document holding any
/// other spelling of it.
#[tokio::test]
async fn stemming_is_symmetric_across_inflections() {
    let (reader, _, _) = build("standard+stem=english");
    // "walking walked walks walk" (doc 10) is reachable by all four.
    for spelling in ["walking", "walked", "walks", "walk"] {
        let hits = matching(&reader, spelling).await;
        assert!(
            hits.contains(&10),
            "{spelling:?} must reach the document holding its inflections"
        );
    }
    // Every spelling of "run" reaches the same set, because they are one
    // term in the index.
    let running = matching(&reader, "running").await;
    assert_eq!(running, matching(&reader, "runs").await);
    assert_eq!(running, matching(&reader, "run").await);
    // Porter2 has no rule for the irregular past, so "ran" stays its own
    // term — a stemmer folds what it has rules for, not everything.
    assert_ne!(running, matching(&reader, "ran").await);
}
