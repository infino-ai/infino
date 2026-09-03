// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Term-dictionary expansion on [`FtsReader`]: widen the tokens of one
//! `LIKE` leaf to the indexed terms they cover, so the table layer's SQL
//! `WHERE` pushdown can answer a substring predicate from posting lists
//! instead of a column scan. Its own `impl FtsReader` block, split from
//! the reader `core`.

use std::str;

use super::{core::*, work::MatchWork};
use crate::superfile::{error::FtsError, fts::dict::make_key};

/// How one `LIKE` fragment token constrains an indexed term. The text is
/// already the column tokenizer's output (lowercased, split), so it
/// compares byte-for-byte against dictionary keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TermPattern<'a> {
    /// The term itself — a token the fragment closes on both sides.
    Exact(&'a str),
    /// Terms beginning with the text: the token's end may sit mid-term
    /// (`'abc%'`).
    Prefix(&'a str),
    /// Terms ending with the text: the token's start may sit mid-term
    /// (`'%abc'`).
    Suffix(&'a str),
    /// Terms containing the text: both ends may sit mid-term (`'%abc%'`).
    Contains(&'a str),
}

impl TermPattern<'_> {
    /// A pattern whose start may sit mid-term. It cannot be bounded to a
    /// dictionary subtree, so it is tested against every key of the
    /// column's range — one shared walk serves all such patterns.
    fn is_open_left(&self) -> bool {
        matches!(self, TermPattern::Suffix(_) | TermPattern::Contains(_))
    }

    /// Whether an indexed `term` is covered by this pattern.
    fn covers(&self, term: &str) -> bool {
        match self {
            TermPattern::Exact(text) => term == *text,
            TermPattern::Prefix(text) => term.starts_with(text),
            TermPattern::Suffix(text) => term.ends_with(text),
            TermPattern::Contains(text) => term.contains(text),
        }
    }
}

/// One pattern's terms as a walk collects them, with the cap it may not
/// exceed. Past the cap the pattern is too broad for a posting-list
/// answer and its collection stops.
struct Collected {
    terms: Vec<String>,
    too_many: bool,
}

impl Collected {
    fn new() -> Self {
        Self {
            terms: Vec::new(),
            too_many: false,
        }
    }

    /// Record `term`; `false` once the cap is hit (the caller stops
    /// feeding this pattern).
    fn admit(&mut self, term: &str, max_terms: usize) -> bool {
        if self.terms.len() == max_terms {
            self.too_many = true;
            return false;
        }
        self.terms.push(term.to_owned());
        true
    }

    fn finish(self) -> Option<Vec<String>> {
        (!self.too_many).then_some(self.terms)
    }
}

impl FtsReader {
    /// Expand each of `patterns` into the indexed terms of `column` it
    /// covers, in lex order, in one pass over the dictionary. A slot is
    /// `None` when more than `max_terms` terms qualify for its pattern —
    /// too broad for a posting-list answer; the caller falls back to
    /// scanning for that token.
    ///
    /// An [`TermPattern::Exact`] token is its own expansion. A prefix
    /// token walks only its own subtree. Every suffix or infix token is
    /// tested against each key of one shared walk over the column's whole
    /// range, so a multi-fragment `LIKE` pays for the vocabulary once, not
    /// once per token. The FST is fetched once when any pattern needs it
    /// (one planned range, like a match's build).
    ///
    /// Errors with `FtsError::UnknownColumn` when `column` is not
    /// FTS-indexed in this superfile, like [`Self::token_match`].
    pub(crate) async fn expand_terms(
        &self,
        column: &str,
        patterns: &[TermPattern<'_>],
        max_terms: usize,
    ) -> Result<(Vec<Option<Vec<String>>>, MatchWork), FtsError> {
        self.resolve_column_id(column)?;
        let mut collected: Vec<Collected> = patterns.iter().map(|_| Collected::new()).collect();
        let mut work = MatchWork::default();
        if patterns.iter().any(|p| !matches!(p, TermPattern::Exact(_))) {
            let fst_bytes = self.dict_bytes_async().await?;
            let dict = Self::open_dict(&fst_bytes)?;
            work.planned_ranges += 1;
            // Every key in the column's range starts with `<column>\x1F`;
            // the term is what follows. Keys are the tokenizer's UTF-8
            // output, so the conversion holds by construction; a key that
            // fails it is skipped rather than trusted.
            let term_start = make_key(column, "").len();
            for (slot, pattern) in collected.iter_mut().zip(patterns) {
                if let TermPattern::Prefix(text) = pattern {
                    dict.for_each_prefix(&make_key(column, text), |key, _| {
                        match str::from_utf8(&key[term_start..]) {
                            Ok(term) => slot.admit(term, max_terms),
                            Err(_) => true,
                        }
                    });
                }
            }
            let mut open_left: Vec<usize> = (0..patterns.len())
                .filter(|&i| patterns[i].is_open_left())
                .collect();
            if !open_left.is_empty() {
                dict.for_each_prefix(&make_key(column, ""), |key, _| {
                    if let Ok(term) = str::from_utf8(&key[term_start..]) {
                        open_left.retain(|&i| {
                            !patterns[i].covers(term) || collected[i].admit(term, max_terms)
                        });
                    }
                    // Stop once every open-left pattern has hit its cap.
                    !open_left.is_empty()
                });
            }
        }
        let mut out = Vec::with_capacity(patterns.len());
        for (slot, pattern) in collected.into_iter().zip(patterns) {
            out.push(match pattern {
                TermPattern::Exact(term) => Some(vec![(*term).to_owned()]),
                _ => slot.finish(),
            });
        }
        Ok((out, work))
    }
}

#[cfg(test)]
mod tests {
    use super::{super::test_util::build_blob, *};

    /// Generous cap so a test never trips the too-many fallback by accident.
    const MAX_TERMS: usize = 64;

    fn expand_all(
        r: &FtsReader,
        patterns: &[TermPattern<'_>],
        max_terms: usize,
    ) -> (Vec<Option<Vec<String>>>, MatchWork) {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        rt.block_on(r.expand_terms("body", patterns, max_terms))
            .expect("expand_terms")
    }

    fn expand(r: &FtsReader, pattern: TermPattern<'_>, max_terms: usize) -> Option<Vec<String>> {
        expand_all(r, &[pattern], max_terms).0.remove(0)
    }

    fn owned(terms: &[&str]) -> Option<Vec<String>> {
        Some(terms.iter().map(|t| (*t).to_owned()).collect())
    }

    #[test]
    fn each_pattern_shape_covers_the_right_terms() {
        // Vocabulary: a, async, boot, is, java, runtime, rust, spring, tokio.
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        assert_eq!(
            expand(&r, TermPattern::Prefix("ru"), MAX_TERMS),
            owned(&["runtime", "rust"]),
            "a prefix walks only its subtree, in lex order"
        );
        assert_eq!(
            expand(&r, TermPattern::Suffix("me"), MAX_TERMS),
            owned(&["runtime"])
        );
        assert_eq!(
            expand(&r, TermPattern::Contains("o"), MAX_TERMS),
            owned(&["boot", "tokio"])
        );
        assert_eq!(
            expand(&r, TermPattern::Exact("rust"), MAX_TERMS),
            owned(&["rust"]),
            "an exact token is its own expansion"
        );
        assert_eq!(
            expand(&r, TermPattern::Contains("zzz"), MAX_TERMS),
            Some(Vec::new()),
            "nothing qualifies ⇒ an empty (not absent) expansion"
        );
    }

    #[test]
    fn several_patterns_expand_in_one_dictionary_pass() {
        // Two open-left tokens, a prefix and an exact token: one FST fetch,
        // each slot exactly what the single-pattern calls return.
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let (out, work) = expand_all(
            &r,
            &[
                TermPattern::Contains("o"),
                TermPattern::Suffix("me"),
                TermPattern::Exact("rust"),
                TermPattern::Prefix("ja"),
            ],
            MAX_TERMS,
        );
        assert_eq!(
            out,
            vec![
                owned(&["boot", "tokio"]),
                owned(&["runtime"]),
                owned(&["rust"]),
                owned(&["java"]),
            ]
        );
        assert_eq!(work.planned_ranges, 1, "one FST fetch for the whole leaf");
    }

    #[test]
    fn expansion_past_the_cap_is_reported_as_too_many() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        // Every term contains the empty string; a cap of two cannot hold
        // nine of them.
        assert_eq!(expand(&r, TermPattern::Contains(""), 2), None);
        // Exactly at the cap still fits.
        assert_eq!(
            expand(&r, TermPattern::Prefix("ru"), 2),
            owned(&["runtime", "rust"])
        );
        // In a shared walk one pattern hitting its cap leaves the others
        // collecting.
        let (out, _) = expand_all(
            &r,
            &[TermPattern::Contains(""), TermPattern::Contains("zz")],
            2,
        );
        assert_eq!(out, vec![None, Some(Vec::new())]);
    }

    #[test]
    fn a_dictionary_walk_reports_its_fetch_but_an_exact_token_does_not() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let (_, walked) = expand_all(&r, &[TermPattern::Prefix("ru")], MAX_TERMS);
        assert_eq!(walked.planned_ranges, 1, "one FST fetch per walk");
        let (_, exact) = expand_all(&r, &[TermPattern::Exact("rust")], MAX_TERMS);
        assert_eq!(exact.planned_ranges, 0, "no dictionary needed");
    }

    #[test]
    fn unknown_column_errors_like_a_match_would() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let err = rt
            .block_on(r.expand_terms("nope", &[TermPattern::Prefix("ru")], MAX_TERMS))
            .expect_err("unknown column");
        assert!(matches!(err, FtsError::UnknownColumn(_)));
    }
}
