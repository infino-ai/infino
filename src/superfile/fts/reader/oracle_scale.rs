// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! BM25 oracle at the scale where the pruned kernels act. The other
//! oracles grade small corpora, where every list is a block or two and a
//! union fits one window, so a ranked kernel never runs with a full heap
//! across many blocks and windows. Here each corpus spans several union
//! windows and its common terms several dozen blocks, and every query is
//! graded three ways against textbook brute-force BM25: the ranked search,
//! the exact-count search (its hits and its count), and the unranked count.
//!
//! Corpora and queries are generated from fixed seeds, so a failure is
//! reproducible from its message alone. Documents are at most
//! [`MAX_DOC_LEN`] tokens long, which keeps every stored length norm exact,
//! so the two scorers agree to rounding.

use std::collections::HashSet;

use bytes::Bytes;

use super::{FtsReader, core::ClauseLists};
use crate::{
    superfile::fts::{builder::FtsBuilder, reader::BoolMode, tokenize::Phrase},
    test_helpers::{brute_force_bm25::BruteForceBm25, default_tokenizer},
};

/// Vocabulary, most frequent first.
const VOCAB: [&str; 16] = [
    "the", "of", "and", "to", "new", "york", "city", "music", "band", "album", "river", "stone",
    "quartz", "zephyr", "onyx", "lynx",
];

/// Relative draw weight of each [`VOCAB`] word: a Zipf-like fall from a
/// word in most documents to words in a handful, so a corpus holds bitset,
/// packed and short-form lists at once.
const WEIGHTS: [u32; 16] = [400, 200, 120, 80, 60, 45, 35, 25, 18, 12, 8, 5, 3, 2, 1, 1];

/// Longest document. Under 16 tokens every length norm is stored exactly.
const MAX_DOC_LEN: u64 = 15;

/// Union window width the kernels use, mirrored so the corpus sizes below
/// read against it.
const WINDOW: u32 = 4096;

/// Corpus sizes: a window and a half, under three windows, four and more.
const CORPUS_SIZES: [u32; 3] = [WINDOW + WINDOW / 2, 11_000, 17_000];

/// Queries per corpus.
const QUERIES: usize = 96;

/// Score tolerance: the scorers share the formula and the inputs, so this
/// absorbs only summation order and f32 rounding.
const SCORE_TOLERANCE: f32 = 1e-3;

/// Docs that carry the planted phrase, one in this many.
const PLANTED_PHRASE_EVERY: u64 = 40;

/// Planted in every document but one in [`EVERY_GAP`]: its blocks are runs
/// of consecutive docs, the run encoding.
const EVERY: &str = "every";
const EVERY_GAP: u32 = 1000;

/// Planted at a fixed stride, so their deltas pack narrower than presence
/// bits: the packed encoding, which random placement never produces.
const STRIDED: [(&str, u32, u32); 2] = [("triad", 3, 0), ("penta", 5, 1)];

/// How far under the rare word's document frequency the explicit pairs'
/// deepest `k` sits, and the largest `k` the rare-led walk is routed for.
const RARE_PAIR_K_SLACK: usize = 4;
const RARE_PAIR_MAX_K: usize = 128;

/// Words a query may name: the drawn vocabulary and the planted terms.
const QUERY_WORDS: [&str; 19] = [
    "the", "of", "and", "to", "new", "york", "city", "music", "band", "album", "river", "stone",
    "quartz", "zephyr", "onyx", "lynx", "every", "triad", "penta",
];

/// A small deterministic generator (xorshift64).
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }

    fn pick(&mut self, from: &[&'static str]) -> &'static str {
        from[self.below(from.len() as u64) as usize]
    }

    /// `n` distinct words from `from` (which must hold at least `n`).
    fn distinct(&mut self, from: &[&'static str], n: u64) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        while (out.len() as u64) < n {
            let w = self.pick(from).to_string();
            if !out.contains(&w) {
                out.push(w);
            }
        }
        out
    }

    fn word(&mut self) -> &'static str {
        let total: u32 = WEIGHTS.iter().sum();
        let mut x = self.below(u64::from(total)) as u32;
        for (w, &weight) in VOCAB.iter().zip(&WEIGHTS) {
            if x < weight {
                return w;
            }
            x -= weight;
        }
        VOCAB[VOCAB.len() - 1]
    }
}

/// A generated corpus. Windows alternate between short and long documents,
/// so block and window maxima vary across the doc-id space and the bounds
/// have something to prune; one doc in [`PLANTED_PHRASE_EVERY`] carries
/// `new york`, so a phrase over it has thousands of matches.
fn corpus(n_docs: u32, seed: u64) -> Vec<String> {
    let mut rng = Rng(seed);
    (0..n_docs)
        .map(|i| {
            let mut words: Vec<&str> = Vec::new();
            if i % EVERY_GAP != EVERY_GAP - 1 {
                words.push(EVERY);
            }
            for &(word, stride, offset) in &STRIDED {
                if i % stride == offset {
                    words.push(word);
                }
            }
            if rng.below(PLANTED_PHRASE_EVERY) == 0 {
                words.extend(["new", "york"]);
            }
            let room = MAX_DOC_LEN - words.len() as u64;
            let short = (i / WINDOW).is_multiple_of(2);
            let max_len = if short { room / 2 } else { room };
            let len = 1 + rng.below(max_len.max(1));
            words.extend((0..len).map(|_| rng.word()));
            words.join(" ")
        })
        .collect()
}

/// One generated query, in clause-list form.
#[derive(Debug, Default)]
struct Query {
    musts: Vec<String>,
    shoulds: Vec<String>,
    negatives: Vec<String>,
    must_phrases: Vec<Phrase<String>>,
    should_phrases: Vec<Phrase<String>>,
    negative_phrases: Vec<Phrase<String>>,
    k: usize,
}

impl Query {
    /// A query from one of several templates. Random clause mixes cover the
    /// general space; the others force the shapes a random mix rarely
    /// produces, each of which some kernel exists for: a rare-plus-common
    /// pair at small `k` (the rare-led walk), several overlapping shoulds
    /// (the windowed union with more than one essential term), several
    /// musts over packed lists (the batched intersection), and a dominant
    /// term in a union of rarer ones (the anchored count).
    fn generate(rng: &mut Rng) -> Self {
        let mut q = Query::default();
        match rng.below(10) {
            0 | 1 => {
                q.shoulds = vec![
                    rng.pick(&["music", "band", "album", "city", "of", "to"])
                        .to_string(),
                    rng.pick(&["stone", "quartz", "zephyr", "onyx", "lynx"])
                        .to_string(),
                ];
                q.k = [1, 10, 50, 128][rng.below(4) as usize];
                return q;
            }
            2 | 3 => {
                let mid = [
                    "and", "to", "new", "york", "city", "music", "band", "album", "river",
                ];
                let n = 3 + rng.below(2);
                q.shoulds = rng.distinct(&mid, n);
                q.k = [10, 50, 100, 1000][rng.below(4) as usize];
                return q;
            }
            4 | 5 => {
                let pool = [
                    "the", "of", "and", "every", "triad", "penta", "city", "music",
                ];
                let n = 2 + rng.below(3);
                q.musts = rng.distinct(&pool, n);
                q.k = [1, 10, 100, 1000][rng.below(4) as usize];
                return q;
            }
            6 => {
                q.shoulds = vec![EVERY.to_string()];
                // Rarer than an eighth of `every` together, so the count
                // anchors on it.
                q.shoulds
                    .extend(rng.distinct(&["river", "stone", "quartz", "zephyr"], 2));
                q.k = [10, 100][rng.below(2) as usize];
                return q;
            }
            _ => {}
        }
        let atoms = 1 + rng.below(5);
        let mut seen = HashSet::new();
        for _ in 0..atoms {
            let phrase = rng.below(4) == 0;
            let tokens: Vec<String> = match phrase {
                true if rng.below(2) == 0 => vec!["new".into(), "york".into()],
                true => (0..2 + rng.below(2))
                    .map(|_| rng.pick(&QUERY_WORDS).to_string())
                    .collect(),
                false => vec![rng.pick(&QUERY_WORDS).to_string()],
            };
            if !seen.insert(tokens.clone()) {
                continue;
            }
            let polarity = rng.below(20);
            match (tokens.len(), polarity) {
                (1, 0..=4) => q.musts.push(tokens[0].clone()),
                (1, 5..=15) => q.shoulds.push(tokens[0].clone()),
                (1, _) => q.negatives.push(tokens[0].clone()),
                (_, 0..=4) => q.must_phrases.push(Phrase::adjacent(tokens)),
                (_, 5..=15) => q.should_phrases.push(Phrase::adjacent(tokens)),
                (_, _) => q.negative_phrases.push(Phrase::adjacent(tokens)),
            }
        }
        if q.musts.is_empty()
            && q.shoulds.is_empty()
            && q.must_phrases.is_empty()
            && q.should_phrases.is_empty()
        {
            q.shoulds.push(rng.word().to_string());
        }
        q.k = match rng.below(5) {
            0 => 1,
            1 => 10,
            2 => 100,
            3 => 1000,
            _ => 1 + rng.below(2000) as usize,
        };
        q
    }
}

/// Assert `got` is a valid top-`k` of the full match list `want` (ranked
/// descending): the right length, only matches, the same scores (ties may
/// pick different docs), and, when every match fits, the same docs.
fn assert_top_k(got: &[(u32, f32)], want: &[(u64, f32)], k: usize, label: &str) {
    assert_eq!(got.len(), k.min(want.len()), "{label}: hit count");
    let want_ids: HashSet<u64> = want.iter().map(|&(d, _)| d).collect();
    for &(d, _) in got {
        assert!(
            want_ids.contains(&u64::from(d)),
            "{label}: doc {d} does not match"
        );
    }
    let mut got_scores: Vec<f32> = got.iter().map(|&(_, s)| s).collect();
    got_scores.sort_by(|a, b| b.total_cmp(a));
    for (i, (g, &(_, w))) in got_scores.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= SCORE_TOLERANCE,
            "{label}: rank {i} score {g} vs brute force {w}"
        );
    }
    if k >= want.len() {
        let got_ids: HashSet<u64> = got.iter().map(|&(d, _)| u64::from(d)).collect();
        assert_eq!(got_ids, want_ids, "{label}: full result set");
    }
}

/// Borrow a list of owned strings as `&str`s.
fn as_strs(v: &[String]) -> Vec<&str> {
    v.iter().map(String::as_str).collect()
}

#[tokio::test]
async fn pruned_kernels_match_brute_force_bm25_across_windows() {
    let tokenizer = default_tokenizer();
    for (ci, &n_docs) in CORPUS_SIZES.iter().enumerate() {
        let seed = 0x9E37_79B9_7F4A_7C15 ^ (ci as u64 + 1);
        let texts = corpus(n_docs, seed);
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), true).expect("register");
        for (i, t) in texts.iter().enumerate() {
            b.add_doc(0, i as u32, t).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let refs: Vec<(u64, &str)> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, t.as_str()))
            .collect();
        let oracle = BruteForceBm25::index(&refs, tokenizer.as_ref());

        let mut rng = Rng(seed.rotate_left(17));
        let mut queries: Vec<Query> = (0..QUERIES).map(|_| Query::generate(&mut rng)).collect();
        // Rare-plus-common pairs at the small `k` the rare-led walk serves,
        // listed rather than drawn, so that walk's completion rule is graded
        // on every corpus whatever the generator happens to draw. A `k` just
        // under the rare word's document frequency puts the walk's k-th best
        // among the rare word's weakest documents, below the common word's
        // best: the case where accepting the walk's answer would be wrong.
        let df = |w: &str| {
            texts
                .iter()
                .filter(|t| t.split(' ').any(|x| x == w))
                .count()
        };
        // The pair must be at least sixteen times apart in document frequency
        // for the router to send it there.
        for common in ["music", "band", "city"] {
            for rare in ["onyx", "lynx"] {
                let near_df = df(rare)
                    .saturating_sub(RARE_PAIR_K_SLACK)
                    .min(RARE_PAIR_MAX_K);
                // Past the rare word's documents too: the walk then holds every
                // one of them and the common word is ranked alone and merged.
                let past_df = (df(rare) + RARE_PAIR_K_SLACK).min(RARE_PAIR_MAX_K);
                for k in [10, 50, 128, near_df, past_df] {
                    queries.push(Query {
                        shoulds: vec![common.to_string(), rare.to_string()],
                        k: k.max(1),
                        ..Query::default()
                    });
                }
            }
        }
        // The same with two or three common words: the rare-led walk for wider
        // unions, which must hand back to the general walk when the k-th best
        // does not clear the common words' maxima together.
        for commons in [&["music", "city"][..], &["band", "city", "music"]] {
            for rare in ["onyx", "lynx"] {
                let near_df = df(rare)
                    .saturating_sub(RARE_PAIR_K_SLACK)
                    .min(RARE_PAIR_MAX_K);
                let past_df = (df(rare) + RARE_PAIR_K_SLACK).min(RARE_PAIR_MAX_K);
                for k in [10, 50, near_df, past_df] {
                    let mut shoulds: Vec<String> = commons.iter().map(|w| w.to_string()).collect();
                    shoulds.insert(1, rare.to_string());
                    queries.push(Query {
                        shoulds,
                        k: k.max(1),
                        ..Query::default()
                    });
                }
            }
        }
        // A required term with several optional ones at a large `k`: the
        // general must+should walk, its collector at `k` past the heap's
        // range, and the batch pre-filter by the optional terms' range maxima;
        // a common required word with rarer optional ones also drives the
        // must-led walk by its optional terms. With and without an exclusion.
        for (musts, shoulds) in [
            (&["every"][..], &["music", "city", "band"][..]),
            (&["the"], &["stone", "river"]),
            (&["every"], &["of", "onyx"]),
        ] {
            for k in [200, 600, 1000] {
                for negatives in [&[][..], &["quartz"][..]] {
                    queries.push(Query {
                        musts: musts.iter().map(|w| w.to_string()).collect(),
                        shoulds: shoulds.iter().map(|w| w.to_string()).collect(),
                        negatives: negatives.iter().map(|w| w.to_string()).collect(),
                        k,
                        ..Query::default()
                    });
                }
            }
        }
        // Unions with an excluded term at small `k`: an exclusion keeps the
        // single-leader walk on its per-candidate path, which the batched
        // leader block otherwise takes over once the heap is full.
        // A strong term with weak ones, so the walk has a single leader.
        for shoulds in [
            &["stone", "the"][..],
            &["quartz", "of", "and"],
            &["river", "the"],
        ] {
            for k in [10, 50] {
                queries.push(Query {
                    shoulds: shoulds.iter().map(|w| w.to_string()).collect(),
                    negatives: vec!["onyx".to_string()],
                    k,
                    ..Query::default()
                });
            }
        }
        for (qi, q) in queries.into_iter().enumerate() {
            let label = format!("corpus {ci} ({n_docs} docs) query {qi}: {q:?}");
            let want = oracle.top_k_atoms(
                &q.musts,
                &q.must_phrases,
                &q.shoulds,
                &q.should_phrases,
                &q.negatives,
                &q.negative_phrases,
                n_docs as usize,
            );
            let (musts, shoulds, negatives) = (
                as_strs(&q.musts),
                as_strs(&q.shoulds),
                as_strs(&q.negatives),
            );
            let lists = || ClauseLists {
                musts: &musts,
                shoulds: &shoulds,
                negatives: &negatives,
                must_phrases: &q.must_phrases,
                should_phrases: &q.should_phrases,
                negative_phrases: &q.negative_phrases,
                ..ClauseLists::default()
            };

            let ranked: Vec<(u32, f32)> = r
                .search_excluding("body", lists(), q.k, f32::NEG_INFINITY)
                .await
                .expect("ranked search")
                .into_iter()
                .map(|(d, s)| (d.get(), s))
                .collect();
            assert_top_k(&ranked, &want, q.k, &format!("{label} ranked"));

            let (counted, count) = r
                .search_excluding_with_count("body", lists(), q.k)
                .await
                .expect("exact-count search");
            let counted: Vec<(u32, f32)> = counted.into_iter().map(|(d, s)| (d.get(), s)).collect();
            assert_top_k(&counted, &want, q.k, &format!("{label} exact-count"));
            assert_eq!(count, want.len() as u64, "{label}: exact count");

            // The unranked count: musts decide the match set when there are
            // any (shoulds only score), otherwise the shoulds' union does.
            let has_musts = !musts.is_empty() || !q.must_phrases.is_empty();
            let (terms, phrases, mode) = match has_musts {
                true => (&musts, &q.must_phrases, BoolMode::And),
                false => (&shoulds, &q.should_phrases, BoolMode::Or),
            };
            let (unranked, _) = r
                .atoms_match_count(
                    "body",
                    terms,
                    phrases,
                    mode,
                    &negatives,
                    &q.negative_phrases,
                )
                .await
                .expect("unranked count");
            assert_eq!(unranked, want.len() as u64, "{label}: unranked count");

            // A plain-term query also counts through the term kernels (the
            // union and intersection count plans), which is the path a
            // term-only count takes; the atom walk above is the general one.
            let plain = q.must_phrases.is_empty()
                && q.should_phrases.is_empty()
                && negatives.is_empty()
                && q.negative_phrases.is_empty();
            if plain {
                let (by_terms, _) = r
                    .token_match_count("body", terms, mode)
                    .await
                    .expect("term count");
                assert_eq!(by_terms, want.len() as u64, "{label}: term count");
            }
        }
    }
}
