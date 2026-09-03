// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Term-dictionary expansion on [`FtsReader`]: widen the tokens of one
//! `LIKE` leaf to the indexed terms they cover, so the table layer's SQL
//! `WHERE` pushdown can answer a substring predicate from posting lists
//! instead of a column scan. Its own `impl FtsReader` block, split from
//! the reader `core`.

use std::{borrow::Cow, str};

use super::{core::*, work::MatchWork};
use crate::superfile::{error::FtsError, fts::dict::make_key};

/// The ASCII letter whose Unicode case-folding class holds a character
/// `to_lowercase` keeps as itself: `s`, folded together with the long s
/// [`LONG_S`]. An `ILIKE` token holding it may be spelled with `ſ` in a
/// matching row's indexed term, so only a dictionary walk can find it.
pub(crate) const LONG_S_ASCII: char = 's';

/// Long s (U+017F). Simple case folding puts it in `s`'s class;
/// `to_lowercase` leaves it, so an indexed term can carry it.
const LONG_S: char = 'ſ';

/// Kelvin sign (U+212A), folded with `k`. `to_lowercase` maps it to `k`
/// before indexing, so a term never carries it; folded here anyway so the
/// comparison does not depend on that.
const KELVIN_SIGN: char = 'K';

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

/// How much of the dictionary a pattern has to see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Walk {
    /// The token is its own expansion.
    None,
    /// Only the keys sharing the token's prefix.
    Subtree,
    /// Every key of the column: the walk shared by all such patterns.
    Full,
}

impl TermPattern<'_> {
    fn text(&self) -> &str {
        match self {
            TermPattern::Exact(text)
            | TermPattern::Prefix(text)
            | TermPattern::Suffix(text)
            | TermPattern::Contains(text) => text,
        }
    }

    /// Which walk resolves this pattern. Under `fold` (`ILIKE`) an exact
    /// or prefix token holding `s` may be spelled with `ſ`, which sorts
    /// elsewhere in the dictionary, so its subtree bound is lost and it
    /// joins the full walk.
    fn walk(&self, fold: bool) -> Walk {
        let long_s = fold && self.text().contains(LONG_S_ASCII);
        match self {
            TermPattern::Exact(_) if long_s => Walk::Full,
            TermPattern::Exact(_) => Walk::None,
            TermPattern::Prefix(_) if long_s => Walk::Full,
            TermPattern::Prefix(_) => Walk::Subtree,
            TermPattern::Suffix(_) | TermPattern::Contains(_) => Walk::Full,
        }
    }

    /// Whether `term` (already folded when the leaf is an `ILIKE`) is
    /// covered by this pattern.
    fn covers(&self, term: &str) -> bool {
        match self {
            TermPattern::Exact(text) => term == *text,
            TermPattern::Prefix(text) => term.starts_with(text),
            TermPattern::Suffix(text) => term.ends_with(text),
            TermPattern::Contains(text) => term.contains(text),
        }
    }
}

/// A dictionary term with `ſ` and `K` folded to the ASCII members of
/// their case-folding classes — the view an `ILIKE` token is compared
/// against. Borrowed when nothing folds.
fn fold_term(term: &str) -> Cow<'_, str> {
    if term.contains([LONG_S, KELVIN_SIGN]) {
        Cow::Owned(
            term.chars()
                .map(|c| match c {
                    LONG_S => 's',
                    KELVIN_SIGN => 'k',
                    other => other,
                })
                .collect(),
        )
    } else {
        Cow::Borrowed(term)
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
    /// once per token. Under `fold` (`ILIKE`) terms are compared with `ſ`
    /// and `K` folded to `s` and `k`, and an exact or prefix token holding
    /// an `s` joins the shared walk (its `ſ` spelling sorts elsewhere).
    /// The FST is fetched once when any pattern needs it (one planned
    /// range, like a match's build).
    ///
    /// Errors with `FtsError::UnknownColumn` when `column` is not
    /// FTS-indexed in this superfile, like [`Self::token_match`].
    pub(crate) async fn expand_terms(
        &self,
        column: &str,
        patterns: &[TermPattern<'_>],
        fold: bool,
        max_terms: usize,
    ) -> Result<(Vec<Option<Vec<String>>>, MatchWork), FtsError> {
        self.resolve_column_id(column)?;
        let walks: Vec<Walk> = patterns.iter().map(|p| p.walk(fold)).collect();
        let mut collected: Vec<Collected> = patterns.iter().map(|_| Collected::new()).collect();
        let mut work = MatchWork::default();
        if walks.iter().any(|w| *w != Walk::None) {
            let fst_bytes = self.dict_bytes_async().await?;
            let dict = Self::open_dict(&fst_bytes)?;
            work.planned_ranges += 1;
            // Every key in the column's range starts with `<column>\x1F`;
            // the term is what follows. Keys are the tokenizer's UTF-8
            // output, so the conversion holds by construction; a key that
            // fails it is skipped rather than trusted.
            let term_start = make_key(column, "").len();
            for ((slot, pattern), walk) in collected.iter_mut().zip(patterns).zip(&walks) {
                if *walk == Walk::Subtree {
                    dict.for_each_prefix(&make_key(column, pattern.text()), |key, _| {
                        match str::from_utf8(&key[term_start..]) {
                            Ok(term) => slot.admit(term, max_terms),
                            Err(_) => true,
                        }
                    });
                }
            }
            let mut full: Vec<usize> = (0..patterns.len())
                .filter(|&i| walks[i] == Walk::Full)
                .collect();
            if !full.is_empty() {
                dict.for_each_prefix(&make_key(column, ""), |key, _| {
                    if let Ok(term) = str::from_utf8(&key[term_start..]) {
                        let view = if fold {
                            fold_term(term)
                        } else {
                            Cow::Borrowed(term)
                        };
                        full.retain(|&i| {
                            !patterns[i].covers(&view) || collected[i].admit(term, max_terms)
                        });
                    }
                    // Stop once every full-walk pattern has hit its cap.
                    !full.is_empty()
                });
            }
        }
        let mut out = Vec::with_capacity(patterns.len());
        for ((slot, pattern), walk) in collected.into_iter().zip(patterns).zip(&walks) {
            out.push(match walk {
                Walk::None => Some(vec![pattern.text().to_owned()]),
                Walk::Subtree | Walk::Full => slot.finish(),
            });
        }
        Ok((out, work))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::test_util::{build_blob, build_standard_fold_blob},
        *,
    };

    /// Generous cap so a test never trips the too-many fallback by accident.
    const MAX_TERMS: usize = 64;

    fn expand_all(
        r: &FtsReader,
        patterns: &[TermPattern<'_>],
        fold: bool,
        max_terms: usize,
    ) -> (Vec<Option<Vec<String>>>, MatchWork) {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        rt.block_on(r.expand_terms("body", patterns, fold, max_terms))
            .expect("expand_terms")
    }

    fn expand(r: &FtsReader, pattern: TermPattern<'_>, max_terms: usize) -> Option<Vec<String>> {
        expand_all(r, &[pattern], false, max_terms).0.remove(0)
    }

    fn expand_fold(r: &FtsReader, pattern: TermPattern<'_>) -> Option<Vec<String>> {
        expand_all(r, &[pattern], true, MAX_TERMS).0.remove(0)
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
            false,
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
    fn folding_finds_the_long_s_spellings_a_case_insensitive_match_admits() {
        // Terms: k, kelvin, riſe, rise, set, sun, sunset, ſun (lex order
        // puts `ſ…` after every ASCII-initial term).
        let (blob, json) = build_standard_fold_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        // Case-sensitive: byte-exact, the long-s spellings are not covered
        // (`ſun` shares the bytes `un`, not `su`).
        assert_eq!(
            expand(&r, TermPattern::Prefix("su"), MAX_TERMS),
            owned(&["sun", "sunset"])
        );
        assert_eq!(
            expand(&r, TermPattern::Contains("su"), MAX_TERMS),
            owned(&["sun", "sunset"])
        );
        assert_eq!(
            expand(&r, TermPattern::Exact("sun"), MAX_TERMS),
            owned(&["sun"])
        );
        // Folded: every shape sees `ſ` as `s`.
        assert_eq!(
            expand_fold(&r, TermPattern::Exact("sun")),
            owned(&["sun", "ſun"]),
            "an exact token holding `s` walks for its `ſ` spelling"
        );
        assert_eq!(
            expand_fold(&r, TermPattern::Prefix("su")),
            owned(&["sun", "sunset", "ſun"])
        );
        assert_eq!(
            expand_fold(&r, TermPattern::Suffix("se")),
            owned(&["rise", "riſe"])
        );
        assert_eq!(
            expand_fold(&r, TermPattern::Contains("su")),
            owned(&["sun", "sunset", "ſun"])
        );
        // The Kelvin sign was lowercased to `k` at index time, so a folded
        // token without an `s` needs no walk at all.
        let (out, work) = expand_all(&r, &[TermPattern::Exact("kelvin")], true, MAX_TERMS);
        assert_eq!(out, vec![owned(&["kelvin"])]);
        assert_eq!(work.planned_ranges, 0);
        assert_eq!(
            expand_fold(&r, TermPattern::Suffix("k")),
            owned(&["k"]),
            "the indexed term is already `k`"
        );
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
            false,
            2,
        );
        assert_eq!(out, vec![None, Some(Vec::new())]);
    }

    #[test]
    fn a_dictionary_walk_reports_its_fetch_but_an_exact_token_does_not() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let (_, walked) = expand_all(&r, &[TermPattern::Prefix("ru")], false, MAX_TERMS);
        assert_eq!(walked.planned_ranges, 1, "one FST fetch per walk");
        let (_, exact) = expand_all(&r, &[TermPattern::Exact("rust")], false, MAX_TERMS);
        assert_eq!(exact.planned_ranges, 0, "no dictionary needed");
    }

    #[test]
    fn unknown_column_errors_like_a_match_would() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let err = rt
            .block_on(r.expand_terms("nope", &[TermPattern::Prefix("ru")], false, MAX_TERMS))
            .expect_err("unknown column");
        assert!(matches!(err, FtsError::UnknownColumn(_)));
    }
}
