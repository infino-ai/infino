// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! End-to-end tests for exact phrase queries (`"a b"`), pinning the
//! semantics against corpus truth and the brute-force phrase oracle:
//!
//! * a phrase matches only contiguous, in-order token sequences;
//! * its per-doc tf is the number of occurrence starts (overlaps
//!   like "a a a" for "a a" count each start);
//! * it composes with the clause model in every polarity;
//! * scores agree with the oracle's sum-idf phrase atom.

use std::collections::{HashMap, HashSet};

use infino::{
    superfile::{SuperfileReader, fts::reader::BoolMode},
    test_helpers::{brute_force_bm25::BruteForceBm25, default_tokenizer},
};

use crate::fts::brute_force_oracle::{
    build_infino_superfile_positional, build_multi_block_corpus, corpus,
};

/// k large enough to capture every match on the 60-doc corpus.
const K_ALL: usize = 64;
/// k covering every match in the 1000-doc multi-block corpus.
const K_ALL_MULTI_BLOCK: usize = 1024;
/// Score-equality tolerance between the two BM25 implementations.
const SCORE_ABS_TOLERANCE: f32 = 1e-3;

async fn search_hits(
    reader: &SuperfileReader,
    query: &str,
    k: usize,
    mode: BoolMode,
) -> Vec<(u64, f32)> {
    reader
        .bm25_hits_async("title", query, k, mode)
        .await
        .expect("phrase query")
        .into_iter()
        .map(|(d, s)| (d as u64, s))
        .collect()
}

/// Compare reader results against the phrase oracle for `query`:
/// identical match sets, per-doc scores within tolerance.
async fn assert_matches_oracle(
    reader: &SuperfileReader,
    oracle: &BruteForceBm25,
    query: &str,
    mode: BoolMode,
    k: usize,
) {
    let tok = default_tokenizer();
    let clauses = tok.parse(query).into_clauses(mode);
    let own = |v: Vec<std::borrow::Cow<'_, str>>| -> Vec<String> {
        v.into_iter().map(|t| t.into_owned()).collect()
    };
    let own_ph = |v: Vec<Vec<std::borrow::Cow<'_, str>>>| -> Vec<Vec<String>> {
        v.into_iter()
            .map(|p| p.into_iter().map(|t| t.into_owned()).collect())
            .collect()
    };
    let want = oracle.top_k_atoms(
        &own(clauses.musts),
        &own_ph(clauses.must_phrases),
        &own(clauses.shoulds),
        &own_ph(clauses.should_phrases),
        &own(clauses.negatives),
        &own_ph(clauses.negative_phrases),
        k,
    );
    let got = search_hits(reader, query, k, mode).await;

    let got_ids: HashSet<u64> = got.iter().map(|(d, _)| *d).collect();
    let want_ids: HashSet<u64> = want.iter().map(|(d, _)| *d).collect();
    assert_eq!(got_ids, want_ids, "query {query:?}: match sets disagree");

    let want_scores: HashMap<u64, f32> = want.into_iter().collect();
    for (d, s) in &got {
        let w = want_scores[d];
        assert!(
            (s - w).abs() <= SCORE_ABS_TOLERANCE,
            "query {query:?} doc {d}: reader score {s} vs oracle {w}"
        );
    }
}

// ── planted semantics ─────────────────────────────────────────────────

/// Small corpus with deliberate phrase shapes: boundaries, repeats,
/// overlapping self-phrases, and near-misses.
fn phrase_corpus() -> Vec<(u64, &'static str)> {
    vec![
        (0, "new york city"),               // match at doc start
        (1, "i love new york"),             // match at doc end
        (2, "york new haven"),              // both words, wrong order
        (3, "new deal in york county"),     // both words, not adjacent
        (4, "new york new york"),           // phrase twice
        (5, "buzz buzz buzz"),              // overlapping "buzz buzz"
        (6, "the new york times daily"),    // interior match
        (7, "brand new yorkshire pudding"), // prefix-ish near miss
    ]
}

#[tokio::test]
async fn phrase_matches_only_contiguous_in_order() {
    let corp = phrase_corpus();
    let r = build_infino_superfile_positional(&corp);
    let hits = search_hits(&r, r#""new york""#, K_ALL, BoolMode::Or).await;
    let mut ids: Vec<u64> = hits.iter().map(|(d, _)| *d).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 1, 4, 6], "boundaries + interior + repeats");
}

#[tokio::test]
async fn overlapping_occurrences_each_count() {
    // "buzz buzz" in "buzz buzz buzz": starts at 0 and 1 → tf 2,
    // which must outscore a doc where... there is only one such doc;
    // pin via the oracle instead.
    let corp = phrase_corpus();
    let r = build_infino_superfile_positional(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    assert_matches_oracle(&r, &oracle, r#""buzz buzz""#, BoolMode::Or, K_ALL).await;
}

#[tokio::test]
async fn phrase_oracle_agreement_small_corpus() {
    let corp = phrase_corpus();
    let r = build_infino_superfile_positional(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    for query in [
        r#""new york""#,
        r#""new york" city"#,
        r#"+"new york" +times"#,
        r#"haven -"new york""#,
        r#""new york city""#,
        r#""york new""#,
        r#"+"new york" -times"#,
    ] {
        assert_matches_oracle(&r, &oracle, query, BoolMode::Or, K_ALL).await;
        assert_matches_oracle(&r, &oracle, query, BoolMode::And, K_ALL).await;
    }
}

#[tokio::test]
async fn three_token_phrase_and_absent_member() {
    let corp = phrase_corpus();
    let r = build_infino_superfile_positional(&corp);
    let hits = search_hits(&r, r#""new york city""#, K_ALL, BoolMode::Or).await;
    assert_eq!(
        hits.iter().map(|(d, _)| *d).collect::<Vec<_>>(),
        vec![0],
        "three-token phrase"
    );
    let hits = search_hits(&r, r#""new zealand""#, K_ALL, BoolMode::Or).await;
    assert!(hits.is_empty(), "absent member matches nothing");
}

#[tokio::test]
async fn dropped_token_leaves_a_phrase_gap() {
    // The default tokenizer drops runs containing non-ASCII bytes.
    // A dropped run must still leave a position gap, or the tokens on
    // either side of it would look adjacent and a phrase would match
    // text that isn't contiguous.
    let corp = vec![
        (0u64, "new york"),          // genuinely adjacent → must match
        (1, "new café york"),        // dropped word between → must NOT match
        (2, "café new york"),        // dropped word before the phrase → matches
        (3, "new york café"),        // dropped word after the phrase → matches
        (4, "new naïve fresh york"), // dropped word + real word between → no match
    ];
    let r = build_infino_superfile_positional(&corp);
    let hits = search_hits(&r, r#""new york""#, K_ALL, BoolMode::Or).await;
    let mut ids: Vec<u64> = hits.iter().map(|(d, _)| *d).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![0, 2, 3],
        "phrase must span a dropped token as a gap, not treat its neighbours as adjacent"
    );

    // The single dropped-word doc is not merely absent because `york`
    // is missing — `york` alone still matches doc 1.
    let york = search_hits(&r, "york", K_ALL, BoolMode::Or).await;
    assert!(
        york.iter().any(|(d, _)| *d == 1),
        "doc 1 still contains the term `york`"
    );
}

#[tokio::test]
async fn single_and_repeated_words_on_sixty_doc_corpus() {
    // The negation-suite corpus, positional: phrase behavior must
    // agree with the oracle on organic text too.
    let corp = corpus();
    let r = build_infino_superfile_positional(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    for query in [
        r#""rust async""#,
        r#""web framework""#,
        r#""search engine" rust"#,
        r#"+"inverted index""#,
    ] {
        assert_matches_oracle(&r, &oracle, query, BoolMode::Or, K_ALL).await;
    }
}

// ── multi-block corpus (positions cross skip boundaries) ─────────────

#[tokio::test]
async fn phrase_oracle_agreement_multi_block() {
    // "alpha beta" is adjacent in every doc divisible by 12 (~83
    // docs), with alpha spanning 3 PFOR blocks — the per-block run
    // offsets and the block-crossing position decode are all
    // exercised, as is a phrase should over a large union.
    let owned = build_multi_block_corpus();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let r = build_infino_superfile_positional(&refs);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&refs, tok.as_ref());
    for query in [
        r#""alpha beta""#,
        r#""alpha beta gamma""#,
        r#"+"alpha beta" epsilon"#,
        r#"delta -"alpha beta""#,
        r#""beta gamma" "gamma delta""#,
    ] {
        assert_matches_oracle(&r, &oracle, query, BoolMode::Or, K_ALL_MULTI_BLOCK).await;
    }
    // Truncated top-k exercises the atom walk's pruning bar.
    assert_matches_oracle(&r, &oracle, r#""alpha beta" gamma"#, BoolMode::Or, 10).await;
}

#[tokio::test]
async fn phrase_block_crossing_rejection() {
    // The block-crossing offset logic must reject as well as accept.
    // `epsilon` fires every 20th doc, and 20 is a multiple of both 4
    // and 5, so `beta` and `gamma` are ALWAYS planted between `alpha`
    // and `epsilon`. The pair therefore co-occurs at every 60th doc —
    // 17 docs spanning all three of `alpha`'s posting blocks — yet is
    // never adjacent, so `"alpha epsilon"` must match nothing. This
    // drives the per-block run-offset rebuild across block boundaries
    // on a genuine rejection, not just on docs that match.
    let owned = build_multi_block_corpus();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let r = build_infino_superfile_positional(&refs);

    let phrase = search_hits(&r, r#""alpha epsilon""#, K_ALL_MULTI_BLOCK, BoolMode::Or).await;
    assert!(
        phrase.is_empty(),
        "alpha and epsilon co-occur but are never adjacent — phrase must not match: {phrase:?}"
    );
    // Guard the guard: the members really do co-occur (across blocks),
    // so the empty phrase result is a rejection, not an empty
    // intersection. Term-AND matches the 17 multiples of 60.
    let both = search_hits(&r, "+alpha +epsilon", K_ALL_MULTI_BLOCK, BoolMode::Or).await;
    assert_eq!(
        both.len(),
        17,
        "alpha ∧ epsilon co-occur at every 60th doc: {}",
        both.len()
    );

    // `"alpha gamma"` is the mixed case across blocks: adjacent (and
    // matching) only where `beta` is absent between them — docs
    // divisible by 15 but not 4 — and rejected elsewhere. Pinning it
    // against the oracle checks the offset logic accepts exactly the
    // true-adjacent occurrences and rejects the rest.
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&refs, tok.as_ref());
    assert_matches_oracle(
        &r,
        &oracle,
        r#""alpha gamma""#,
        BoolMode::Or,
        K_ALL_MULTI_BLOCK,
    )
    .await;
}

#[tokio::test]
async fn must_phrase_skips_cooccurring_nonadjacent() {
    // The must-driven walk is two-phase: Phase 1 aligns each phrase must by
    // member *co-occurrence* (no positions), Phase 2 verifies adjacency and
    // scores only the survivors. This pins that Phase 2 actually rejects the
    // docs where the members co-occur but are not adjacent — a verify bug would
    // leak every co-occurrence into the must result.
    //
    // In the multi-block corpus alpha (every 3rd doc) and gamma (every 5th)
    // co-occur at every 15th doc, but beta (every 4th) is planted between them
    // whenever the doc is also divisible by 4 (i.e. divisible by 60), so
    // `"alpha gamma"` is adjacent exactly at the docs divisible by 15 but not 4.
    // Co-occurrence: 67 docs (multiples of 15); adjacent: 50 (those 67 minus the
    // 17 multiples of 60, where beta splits the pair).
    let owned = build_multi_block_corpus();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let r = build_infino_superfile_positional(&refs);

    // Guard the guard: the members really do co-occur at 67 docs (term-AND), so
    // the smaller phrase result below is a Phase-2 rejection, not a thin
    // intersection.
    let cooccur = search_hits(&r, "+alpha +gamma", K_ALL_MULTI_BLOCK, BoolMode::Or).await;
    assert_eq!(
        cooccur.len(),
        67,
        "alpha and gamma co-occur at every 15th doc: {}",
        cooccur.len()
    );

    // The must phrase keeps only the adjacent subset: the 17 docs divisible by
    // 60, where beta sits between alpha and gamma, are aligned in Phase 1 and
    // rejected in Phase 2.
    let must_phrase = search_hits(&r, r#"+"alpha gamma""#, K_ALL_MULTI_BLOCK, BoolMode::Or).await;
    assert_eq!(
        must_phrase.len(),
        50,
        "must phrase matches only the adjacent docs (co-occurring-but-not-adjacent skipped): {}",
        must_phrase.len()
    );

    // Pin the ranked walk against the oracle: the phrase must alone, the phrase
    // must beside a should term (the should drags scoring onto verified docs),
    // and the phrase as one of two musts (And). Small k keeps the pruning bar
    // live through the two-phase walk, so a bug that skips or mis-scores a
    // verified doc diverges from ground truth.
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&refs, tok.as_ref());
    for (query, mode) in [
        (r#"+"alpha gamma""#, BoolMode::Or),
        (r#"+"alpha gamma" delta"#, BoolMode::Or),
        (r#""alpha gamma" delta"#, BoolMode::And),
    ] {
        for k in [3usize, 10, K_ALL_MULTI_BLOCK] {
            assert_matches_oracle(&r, &oracle, query, mode, k).await;
        }
    }
}

#[tokio::test]
async fn truncated_top_k_pruning_agrees_with_oracle() {
    // Every query shape whose ranked walk can skip phrase
    // verification once the heap fills — a small k keeps the bar live
    // for most of the walk, so a pruning bug (skipping a doc that
    // belongs in the top k, or scoring a skipped doc) diverges from
    // the oracle. Shapes cover the three prune sites: the union
    // advance (bare phrase, alone and beside other atoms), the
    // must-driven leapfrog (phrase musts), and the should drag
    // (phrase should landing on must-matched docs).
    let owned = build_multi_block_corpus();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let r = build_infino_superfile_positional(&refs);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&refs, tok.as_ref());
    for query in [
        r#""alpha beta""#,
        r#"+"alpha beta""#,
        r#"+"alpha beta" +delta"#,
        r#"+delta "alpha beta""#,
        r#""alpha beta" "gamma delta""#,
        r#""alpha beta" epsilon -"gamma delta""#,
    ] {
        for k in [3, 10] {
            assert_matches_oracle(&r, &oracle, query, BoolMode::Or, k).await;
        }
    }
}

#[tokio::test]
async fn unranked_ids_and_count_agree_with_oracle() {
    // The unranked walk (`token_match` / `count`) is a separate spine
    // from ranked search and must never prune: every verified match
    // counts. Pin its ids and count against the oracle's full match
    // set for both combination modes.
    let owned = build_multi_block_corpus();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let r = build_infino_superfile_positional(&refs);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&refs, tok.as_ref());

    let alpha_beta = || vec![vec!["alpha".to_string(), "beta".to_string()]];
    let shapes: Vec<(Vec<&str>, Vec<Vec<String>>)> = vec![
        (vec![], alpha_beta()),
        (vec!["gamma"], alpha_beta()),
        (
            vec![],
            vec![
                vec!["alpha".to_string(), "beta".to_string()],
                vec!["gamma".to_string(), "delta".to_string()],
            ],
        ),
    ];
    for (terms, phrases) in shapes {
        for mode in [BoolMode::And, BoolMode::Or] {
            let owned_terms: Vec<String> = terms.iter().map(|t| t.to_string()).collect();
            // Under And the atoms are all musts; under Or all shoulds.
            let (musts, must_ph, shoulds, should_ph) = match mode {
                BoolMode::And => (owned_terms.clone(), phrases.clone(), vec![], vec![]),
                BoolMode::Or => (vec![], vec![], owned_terms.clone(), phrases.clone()),
            };
            let want: HashSet<u64> = oracle
                .top_k_atoms(&musts, &must_ph, &shoulds, &should_ph, &[], &[], refs.len())
                .into_iter()
                .map(|(d, _)| d)
                .collect();

            let ids = r
                .atoms_match_ids("title", &terms, &phrases, mode)
                .await
                .expect("atoms_match_ids")
                .0;
            let got: HashSet<u64> = ids.into_iter().map(u64::from).collect();
            assert_eq!(
                got, want,
                "unranked ids disagree ({terms:?} + {phrases:?}, {mode:?})"
            );

            let count = r
                .atoms_match_count("title", &terms, &phrases, mode, &[], &[])
                .await
                .expect("atoms_match_count")
                .0;
            assert_eq!(
                count as usize,
                want.len(),
                "unranked count disagrees ({terms:?} + {phrases:?}, {mode:?})"
            );
        }
    }
}

/// A phrase of common words AND'd with a rare term: the AND walk aligns on the
/// phrase's cheap member-approximation and verifies adjacency only where the
/// rare term also lands. This pins that the two-phase split stays exact — a doc
/// matches iff it holds the rare term AND the contiguous phrase — across every
/// tricky case: phrase-adjacent-but-no-rare-term (pruned), rare-term-with-
/// members-present-but-not-adjacent (rejected at verify), and rare-term-only.
#[tokio::test]
async fn phrase_and_rare_term_two_phase_agrees() {
    // "the"/"who" are in most docs; "uk" in a few. Only docs with BOTH the
    // contiguous phrase "the who" and "uk" may match.
    let refs: Vec<(u64, &str)> = vec![
        (0, "the who played in the uk last year"), // "the who" adjacent + uk  -> MATCH
        (1, "the who reunion tour"),               // "the who" adjacent, no uk -> prune
        (2, "who owns the uk media"),              // members + uk, not adjacent -> reject
        (3, "the band who toured the uk"),         // members + uk, not adjacent -> reject
        (4, "uk weather today"),                   // uk only, no phrase       -> no match
        (5, "the who the who in the uk"),          // two "the who" starts + uk -> MATCH
        (6, "nothing relevant here at all"),       // neither                  -> no match
    ];
    let r = build_infino_superfile_positional(&refs);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&refs, tok.as_ref());

    let terms = vec!["uk"];
    let phrases = vec![vec!["the".to_string(), "who".to_string()]];
    let owned_terms: Vec<String> = terms.iter().map(|t| t.to_string()).collect();

    let want: HashSet<u64> = oracle
        .top_k_atoms(&owned_terms, &phrases, &[], &[], &[], &[], refs.len())
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    assert_eq!(
        want,
        HashSet::from([0, 5]),
        "oracle sanity for +\"the who\" +uk"
    );

    let ids = r
        .atoms_match_ids("title", &terms, &phrases, BoolMode::And)
        .await
        .expect("atoms_match_ids")
        .0;
    let got: HashSet<u64> = ids.into_iter().map(u64::from).collect();
    assert_eq!(got, want, "two-phase AND ids for phrase + rare term");

    let count = r
        .atoms_match_count("title", &terms, &phrases, BoolMode::And, &[], &[])
        .await
        .expect("atoms_match_count")
        .0;
    assert_eq!(
        count as usize,
        want.len(),
        "two-phase AND count for phrase + rare term"
    );
}

/// The two-phase phrase AND at scale, where the phrase members are **dense**
/// and therefore BITSET-encoded (the shape the optimization exists for). The
/// small corpus above keeps members in single PACKED/inline blocks; here `the`
/// and `who` appear in every one of a few thousand docs, so their posting lists
/// are stored as bitset blocks. That exercises the path the small test can't:
/// `approx_seek` confirming a member by `contains` bit-test on a bitset block
/// (no decode), then `verify_at_aligned` → `materialize_at` decoding that same
/// bitset block only for the aligned survivors. Cross-checked against the
/// brute-force oracle for ids and count.
#[tokio::test]
async fn two_phase_phrase_with_dense_bitset_members_agrees() {
    // Every doc holds `who` and `the` (dense ⇒ bitset members) but not adjacent
    // as "the who"; the phrase is planted only on every 13th doc, and the rare
    // co-term `uk` on every 40th. A doc matches `+"the who" +uk` iff both land,
    // i.e. every lcm(13,40)=520th doc.
    const N_DOCS: u64 = 2600;
    const PHRASE_STRIDE: u64 = 13;
    const UK_STRIDE: u64 = 40;
    let owned: Vec<(u64, String)> = (0..N_DOCS)
        .map(|d| {
            // Base: `who` and `the` present but separated ⇒ no "the who".
            let mut s = format!("who a{d} the b{d}");
            if d.is_multiple_of(PHRASE_STRIDE) {
                // Plant an adjacent "the who" at the front.
                s = format!("the who {s}");
            }
            if d.is_multiple_of(UK_STRIDE) {
                s.push_str(" uk");
            }
            (d, s)
        })
        .collect();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let r = build_infino_superfile_positional(&refs);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&refs, tok.as_ref());

    let terms = vec!["uk"];
    let phrases = vec![vec!["the".to_string(), "who".to_string()]];
    let owned_terms: Vec<String> = terms.iter().map(|t| t.to_string()).collect();

    let want: HashSet<u64> = oracle
        .top_k_atoms(&owned_terms, &phrases, &[], &[], &[], &[], refs.len())
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    // Independent structural expectation: multiples of lcm(13, 40) = 520.
    let expected: HashSet<u64> = (0..N_DOCS).filter(|d| d.is_multiple_of(520)).collect();
    assert_eq!(want, expected, "oracle sanity for dense +\"the who\" +uk");
    assert!(!want.is_empty(), "corpus must plant some matches");

    let ids = r
        .atoms_match_ids("title", &terms, &phrases, BoolMode::And)
        .await
        .expect("atoms_match_ids")
        .0;
    let got: HashSet<u64> = ids.into_iter().map(u64::from).collect();
    assert_eq!(got, want, "dense two-phase AND ids");

    let count = r
        .atoms_match_count("title", &terms, &phrases, BoolMode::And, &[], &[])
        .await
        .expect("atoms_match_count")
        .0;
    assert_eq!(count as usize, want.len(), "dense two-phase AND count");
}
