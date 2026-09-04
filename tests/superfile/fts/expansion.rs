// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Ranked BM25 over query-time term groups, against the brute-force
//! oracle's group semantics (Σ member tf, idf of the commonest member).
//!
//! The expansion is applied on the reader side by the engine and on the
//! oracle side by hand (or by a tiny independent rewrite in the fuzz
//! cell), so the two never share the code under test. Every cell also
//! pins block-max soundness: the top-k at a small `k`, where the walk's
//! pruning bar is live, must equal the head of the same query at `k` ≥
//! the match count, where nothing is ever pruned.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use infino::{
    superfile::{
        SuperfileReader,
        fts::reader::{BoolMode, QueryExpansion},
    },
    test_helpers::{
        brute_force_bm25::{BruteForceBm25, OracleQuery},
        default_tokenizer,
    },
};
use proptest::prelude::*;

use crate::fts::brute_force_oracle::build_infino_superfile;

/// Score-equality tolerance: the two scorers share the BM25 formula and,
/// on the short docs planted here (every length inside the norm table's
/// exact region), identical inputs; only f64-vs-f32 idf rounding and sum
/// order remain.
const SCORE_ABS_TOLERANCE: f32 = 1e-3;
/// A `k` that covers every match on the planted corpora.
const K_ALL: usize = 4096;
/// A `k` far below the match counts, so the top-k heap fills and the
/// walks' pruning bars engage.
const K_SMALL: usize = 3;

/// The planted families and stop terms.
fn vocabulary() -> Arc<QueryExpansion> {
    Arc::new(
        QueryExpansion::new()
            .stop(["the", "and", "of"])
            .group("run", ["runs", "running", "ran"])
            .group("fail", ["fails", "failing", "failed"]),
    )
}

fn run_group() -> Vec<String> {
    ["run", "runs", "running", "ran"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn fail_group() -> Vec<String> {
    ["fail", "fails", "failing", "failed"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn owned(terms: &[&str]) -> Vec<String> {
    terms.iter().map(|s| s.to_string()).collect()
}

/// A corpus where the surface forms have very different document
/// frequencies (so the max-df idf visibly differs from each member's
/// own), several docs carry two forms (so Σ tf differs from any single
/// member's tf), and every doc is short enough for exact length norms.
fn corpus() -> Vec<(u64, &'static str)> {
    vec![
        (0, "run run runs login"),
        (1, "running fails on the login page"),
        (2, "the job ran and failed"),
        (3, "runs of the suite pass"),
        (4, "run failing tests first"),
        (5, "login page redesign"),
        (6, "nothing to see here"),
        (7, "runner runs runs run"),
        (8, "fail fast fail often"),
        (9, "the who"),
        (10, "run run run run"),
        (11, "ran"),
        (12, "fails fails failed"),
        (13, "pass the tests"),
        (14, "login login run"),
        (15, "the run of the mill"),
    ]
}

async fn reader_hits(
    r: &SuperfileReader,
    query: &str,
    k: usize,
    mode: BoolMode,
    expansion: Option<&QueryExpansion>,
) -> Vec<(u64, f32)> {
    r.bm25_hits_expanded_async("title", query, k, mode, expansion)
        .await
        .expect("bm25 with expansion")
        .into_iter()
        .map(|(d, s)| (d as u64, s))
        .collect()
}

/// Full-result agreement: same doc set, same per-doc score.
fn assert_same_scores(got: &[(u64, f32)], want: &[(u64, f32)], what: &str) {
    let got_ids: HashSet<u64> = got.iter().map(|(d, _)| *d).collect();
    let want_ids: HashSet<u64> = want.iter().map(|(d, _)| *d).collect();
    assert_eq!(got_ids, want_ids, "{what}: match set");
    let want_scores: HashMap<u64, f32> = want.iter().copied().collect();
    for (d, s) in got {
        let w = want_scores[d];
        assert!(
            (s - w).abs() < SCORE_ABS_TOLERANCE,
            "{what}: score on doc {d}: reader={s} oracle={w}"
        );
    }
}

/// Block-max soundness: the pruned small-`k` result equals the head of
/// the unpruned full result, compared as score multisets so a tie at the
/// `k`-th place cannot make the check flaky.
fn assert_pruned_head_matches(small: &[(u64, f32)], full: &[(u64, f32)], what: &str) {
    assert_eq!(
        small.len(),
        K_SMALL.min(full.len()),
        "{what}: pruned result size"
    );
    let full_ids: HashSet<u64> = full.iter().map(|(d, _)| *d).collect();
    for (d, _) in small {
        assert!(
            full_ids.contains(d),
            "{what}: pruned hit {d} is not a match"
        );
    }
    let mut small_scores: Vec<f32> = small.iter().map(|(_, s)| *s).collect();
    let mut head_scores: Vec<f32> = full.iter().take(K_SMALL).map(|(_, s)| *s).collect();
    small_scores.sort_by(f32::total_cmp);
    head_scores.sort_by(f32::total_cmp);
    for (a, b) in small_scores.iter().zip(&head_scores) {
        assert!(
            (a - b).abs() < SCORE_ABS_TOLERANCE,
            "{what}: pruned top-{K_SMALL} scores {small_scores:?} != unpruned head {head_scores:?}"
        );
    }
}

/// One planted case: the query string the reader sees and the clause
/// lists the oracle scores, written out by hand.
struct Case {
    query: &'static str,
    mode: BoolMode,
    oracle: OracleQuery,
}

fn cases() -> Vec<Case> {
    vec![
        // A lone group in Or: the union scored as one term.
        Case {
            query: "run",
            mode: BoolMode::Or,
            oracle: OracleQuery {
                should_groups: vec![run_group()],
                ..OracleQuery::default()
            },
        },
        // Two groups Or'd: two atoms, each Σ tf / max-df idf.
        Case {
            query: "run fail",
            mode: BoolMode::Or,
            oracle: OracleQuery {
                should_groups: vec![run_group(), fail_group()],
                ..OracleQuery::default()
            },
        },
        // Group + plain term Or'd; the stop term vanishes.
        Case {
            query: "the run login",
            mode: BoolMode::Or,
            oracle: OracleQuery {
                shoulds: owned(&["login"]),
                should_groups: vec![run_group()],
                ..OracleQuery::default()
            },
        },
        // And: a must group satisfied by any member, plus a must term.
        Case {
            query: "run login",
            mode: BoolMode::And,
            oracle: OracleQuery {
                musts: owned(&["login"]),
                must_groups: vec![run_group()],
                ..OracleQuery::default()
            },
        },
        // Two must groups.
        Case {
            query: "run fail",
            mode: BoolMode::And,
            oracle: OracleQuery {
                must_groups: vec![run_group(), fail_group()],
                ..OracleQuery::default()
            },
        },
        // Sigils: +group is a must, bare group a should, -group excludes
        // any member.
        Case {
            query: "+run fail -login",
            mode: BoolMode::Or,
            oracle: OracleQuery {
                must_groups: vec![run_group()],
                should_groups: vec![fail_group()],
                negatives: owned(&["login"]),
                ..OracleQuery::default()
            },
        },
        Case {
            query: "login -run",
            mode: BoolMode::Or,
            oracle: OracleQuery {
                shoulds: owned(&["login"]),
                negative_groups: vec![run_group()],
                ..OracleQuery::default()
            },
        },
        // A member used literally stays a plain term (no chasing).
        Case {
            query: "runs run",
            mode: BoolMode::Or,
            oracle: OracleQuery {
                shoulds: owned(&["runs"]),
                should_groups: vec![run_group()],
                ..OracleQuery::default()
            },
        },
        // A group with only a rare member present in the corpus of the
        // chosen docs still scores with the commonest member's idf.
        Case {
            query: "+run +pass",
            mode: BoolMode::Or,
            oracle: OracleQuery {
                musts: owned(&["pass"]),
                must_groups: vec![run_group()],
                ..OracleQuery::default()
            },
        },
    ]
}

#[tokio::test]
async fn grouped_queries_match_the_oracle_and_prune_soundly() {
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    let vocab = vocabulary();

    for case in cases() {
        let what = format!("{:?} ({:?})", case.query, case.mode);
        let full = reader_hits(&r, case.query, K_ALL, case.mode, Some(&vocab)).await;
        let want = oracle.top_k_expanded(&case.oracle, K_ALL);
        assert!(!want.is_empty(), "{what}: the oracle must match something");
        assert_same_scores(&full, &want, &what);
        let small = reader_hits(&r, case.query, K_SMALL, case.mode, Some(&vocab)).await;
        assert_pruned_head_matches(&small, &full, &what);
    }
}

#[tokio::test]
async fn a_group_scores_with_the_commonest_members_idf_and_summed_tf() {
    // Doc 11 holds only `ran` (the rarest form, df 2); doc 10 holds `run`
    // four times; doc 0 holds `run` twice and `runs` once. Under the
    // group all three score as if the corpus had one stem: doc 11's
    // score is the commonest form's idf, not `ran`'s inflated one, and
    // doc 0's tf is 3, so it outranks doc 11 and, by saturation, sits
    // below doc 10.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    let vocab = vocabulary();

    let hits = reader_hits(&r, "run", K_ALL, BoolMode::Or, Some(&vocab)).await;
    let score: HashMap<u64, f32> = hits.iter().copied().collect();
    assert!(score[&10] > score[&0], "tf 4 outranks tf 3 at equal length");
    assert!(score[&0] > score[&11], "tf 3 outranks tf 1");

    // The literal `ran` alone scores higher than the same doc under the
    // group: its own idf is larger than the commonest member's.
    let literal = reader_hits(&r, "ran", K_ALL, BoolMode::Or, None).await;
    let literal_score: HashMap<u64, f32> = literal.iter().copied().collect();
    assert!(
        literal_score[&11] > score[&11],
        "the rare form's own idf ({}) must exceed the group idf ({})",
        literal_score[&11],
        score[&11]
    );
    // And the oracle agrees on every value.
    let want = oracle.top_k_expanded(
        &OracleQuery {
            should_groups: vec![run_group()],
            ..OracleQuery::default()
        },
        K_ALL,
    );
    assert_same_scores(&hits, &want, "run group");
}

#[tokio::test]
async fn no_expansion_and_an_empty_expansion_are_the_same_search() {
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let empty = QueryExpansion::new();
    for (query, mode) in [
        ("run login", BoolMode::Or),
        ("+run -login the", BoolMode::And),
        ("fails runs", BoolMode::And),
    ] {
        let none = reader_hits(&r, query, K_ALL, mode, None).await;
        let some = reader_hits(&r, query, K_ALL, mode, Some(&empty)).await;
        assert_eq!(
            none.iter()
                .map(|&(d, s)| (d, s.to_bits()))
                .collect::<Vec<_>>(),
            some.iter()
                .map(|&(d, s)| (d, s.to_bits()))
                .collect::<Vec<_>>(),
            "{query:?} {mode:?}: an empty expansion must be bit-identical to none"
        );
    }
}

#[tokio::test]
async fn an_unmatchable_expansion_entry_is_a_typed_error() {
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let bad = QueryExpansion::new().group("run", ["new york"]);
    let err = r
        .bm25_hits_expanded_async("title", "run", K_ALL, BoolMode::Or, Some(&bad))
        .await
        .expect_err("two-word member");
    assert!(
        err.to_string().contains("new york"),
        "the error names the entry: {err}"
    );
}

// ── fuzz cell: random corpora, random families ──────────────────────────

/// Vocabulary for the fuzz cell: three families of four forms plus four
/// loose words, so families co-occur and a doc often holds two forms.
const FAMILIES: &[[&str; 4]] = &[
    ["run", "runs", "running", "ran"],
    ["fail", "fails", "failing", "failed"],
    ["test", "tests", "testing", "tested"],
];
const LOOSE: &[&str] = &["login", "page", "suite", "pass"];

fn fuzz_vocab() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = FAMILIES.iter().flatten().copied().collect();
    v.extend_from_slice(LOOSE);
    v
}

/// Family membership for the test-side rewrite: `head → members`.
fn family_of(word: &str) -> Option<Vec<String>> {
    FAMILIES
        .iter()
        .find(|f| f[0] == word)
        .map(|f| f.iter().map(|s| s.to_string()).collect())
}

fn fuzz_expansion() -> QueryExpansion {
    let mut exp = QueryExpansion::new();
    for f in FAMILIES {
        exp = exp.group(f[0], f[1..].iter().copied());
    }
    exp
}

/// One query atom: polarity (0 must, 1 bare, 2 negative) and a vocab index.
#[derive(Clone, Debug)]
struct Atom {
    polarity: u8,
    word: usize,
}

/// Docs of 1..=12 vocab words (every length inside the exact norm region).
fn corpus_strategy() -> impl Strategy<Value = Vec<Vec<usize>>> {
    let words = fuzz_vocab().len();
    prop::collection::vec(prop::collection::vec(0..words, 1..=12), 1..=300)
}

fn atoms_strategy() -> impl Strategy<Value = Vec<Atom>> {
    let words = fuzz_vocab().len();
    prop::collection::vec(
        (0u8..3u8, 0..words).prop_map(|(polarity, word)| Atom { polarity, word }),
        1..=4,
    )
}

/// Render the atoms as a query string (deduplicated, at least one
/// positive so the query is never negation-only), and build the oracle
/// clause lists by the same rule the engine follows: a head expands to
/// its family, everything else stays a term. Independent of the engine's
/// own rewrite.
fn render(atoms: &[Atom], and_mode: bool) -> (String, OracleQuery) {
    let vocab = fuzz_vocab();
    let mut atoms = atoms.to_vec();
    if atoms.iter().all(|a| a.polarity == 2) {
        atoms[0].polarity = 1;
    }
    let mut seen = HashSet::new();
    let mut rendered: Vec<String> = Vec::new();
    let mut q = OracleQuery::default();
    for a in &atoms {
        let word = vocab[a.word];
        let text = match a.polarity {
            0 => format!("+{word}"),
            2 => format!("-{word}"),
            _ => word.to_string(),
        };
        if !seen.insert(text.clone()) {
            continue;
        }
        rendered.push(text);
        let family = family_of(word);
        // Bare atoms take the mode's polarity.
        let is_must = a.polarity == 0 || (a.polarity == 1 && and_mode);
        match (a.polarity, family) {
            (2, Some(f)) => q.negative_groups.push(f),
            (2, None) => q.negatives.push(word.to_string()),
            (_, Some(f)) if is_must => q.must_groups.push(f),
            (_, Some(f)) => q.should_groups.push(f),
            (_, None) if is_must => q.musts.push(word.to_string()),
            (_, None) => q.shoulds.push(word.to_string()),
        }
    }
    (rendered.join(" "), q)
}

fn rt() -> &'static tokio::runtime::Runtime {
    use std::sync::OnceLock;
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("build fuzz runtime")
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn fuzz_grouped_queries_match_the_oracle(
        corpus_idx in corpus_strategy(),
        atoms in atoms_strategy(),
        and_mode in any::<bool>(),
        k in 1usize..=8,
    ) {
        let vocab = fuzz_vocab();
        let owned: Vec<(u64, String)> = corpus_idx
            .iter()
            .enumerate()
            .map(|(i, ws)| (i as u64, ws.iter().map(|&w| vocab[w]).collect::<Vec<_>>().join(" ")))
            .collect();
        let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
        let r = build_infino_superfile(&refs);
        let tok = default_tokenizer();
        let oracle = BruteForceBm25::index(&refs, tok.as_ref());
        let exp = fuzz_expansion();
        let mode = if and_mode { BoolMode::And } else { BoolMode::Or };
        let (query, oq) = render(&atoms, and_mode);

        let want = oracle.top_k_expanded(&oq, refs.len());
        let full = rt().block_on(async {
            reader_hits(&r, &query, refs.len(), mode, Some(&exp)).await
        });
        let want_ids: HashSet<u64> = want.iter().map(|(d, _)| *d).collect();
        let got_ids: HashSet<u64> = full.iter().map(|(d, _)| *d).collect();
        prop_assert_eq!(&got_ids, &want_ids, "match set for {:?} {:?}", query, mode);
        let want_scores: HashMap<u64, f32> = want.iter().copied().collect();
        for (d, s) in &full {
            let w = want_scores[d];
            prop_assert!(
                (s - w).abs() < SCORE_ABS_TOLERANCE,
                "score on doc {} for {:?} {:?}: reader={} oracle={}", d, query, mode, s, w
            );
        }
        // Pruned top-k: exactly min(k, matches) hits, every one a match,
        // and its score multiset equals the unpruned head's.
        let small = rt().block_on(async { reader_hits(&r, &query, k, mode, Some(&exp)).await });
        prop_assert_eq!(small.len(), k.min(want.len()), "pruned size for {:?} {:?}", query, mode);
        let mut small_scores: Vec<f32> = small.iter().map(|(_, s)| *s).collect();
        let mut head_scores: Vec<f32> = want.iter().take(k).map(|(_, s)| *s).collect();
        small_scores.sort_by(f32::total_cmp);
        head_scores.sort_by(f32::total_cmp);
        for ((d, _), (a, b)) in small.iter().zip(small_scores.iter().zip(&head_scores)) {
            prop_assert!(want_ids.contains(d), "pruned hit {} is not a match", d);
            prop_assert!(
                (a - b).abs() < SCORE_ABS_TOLERANCE,
                "pruned scores {:?} vs head {:?} for {:?} {:?}", small_scores, head_scores, query, mode
            );
        }
    }
}
