// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Expected match sets read straight off a planted corpus, by splitting
//! its text on whitespace.
//!
//! Every FTS suite that plants a corpus needs these, and each one grew
//! its own copy — four identical `docs_with` bodies differing only in
//! whether the corpus held `&str` or `String`. One copy, generic over
//! both, is the whole point: these are the *independent* side of every
//! assertion in those suites, so two of them drifting apart would let a
//! suite grade the engine against a slightly different notion of what a
//! token is than its neighbour does.
//!
//! Deliberately naive: whitespace splitting, exact string equality, no
//! analyzer. That is what makes them independent of the reader and of
//! the brute-force scorer alike. A suite that needs the production
//! tokenization asserts against [`infino::test_helpers::brute_force_bm25`]
//! instead.

use std::collections::HashSet;

/// Documents whose text contains `term` as a whitespace-delimited token.
pub fn docs_with<S: AsRef<str>>(corpus: &[(u64, S)], term: &str) -> HashSet<u64> {
    corpus
        .iter()
        .filter(|(_, text)| text.as_ref().split_whitespace().any(|w| w == term))
        .map(|(doc, _)| *doc)
        .collect()
}

/// Documents whose text contains `phrase` as consecutive
/// whitespace-delimited tokens, in order.
pub fn docs_with_phrase<S: AsRef<str>>(corpus: &[(u64, S)], phrase: &[&str]) -> HashSet<u64> {
    corpus
        .iter()
        .filter(|(_, text)| {
            let tokens: Vec<&str> = text.as_ref().split_whitespace().collect();
            tokens.windows(phrase.len()).any(|w| w == phrase)
        })
        .map(|(doc, _)| *doc)
        .collect()
}
