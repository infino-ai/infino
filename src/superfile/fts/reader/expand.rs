// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Term-dictionary expansion on [`FtsReader`]: widen one `LIKE` fragment
//! token to the indexed terms it covers, so the table layer's SQL `WHERE`
//! pushdown can answer a substring predicate from posting lists instead of
//! a column scan. Its own `impl FtsReader` block, split from the reader
//! `core`.

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
    /// The dictionary range the walk is bounded to. A prefix pattern walks
    /// only its own subtree; a suffix or infix pattern has to see the
    /// column's whole vocabulary, since any term may end with or contain
    /// the text.
    fn walk_prefix(&self) -> &str {
        match self {
            TermPattern::Prefix(text) => text,
            TermPattern::Exact(_) | TermPattern::Suffix(_) | TermPattern::Contains(_) => "",
        }
    }

    /// Whether an indexed `term` is covered by this pattern.
    fn covers(&self, term: &str) -> bool {
        match self {
            TermPattern::Exact(text) => term == *text,
            // The walk already stayed inside the prefix subtree.
            TermPattern::Prefix(_) => true,
            TermPattern::Suffix(text) => term.ends_with(text),
            TermPattern::Contains(text) => term.contains(text),
        }
    }
}

impl FtsReader {
    /// Expand `pattern` into the indexed terms of `column` it covers, in
    /// lex order. `Ok((None, work))` says more than `max_terms` terms
    /// qualify — the pattern is too broad for a posting-list answer and
    /// the caller falls back to scanning. An [`TermPattern::Exact`] token
    /// is its own expansion and costs no dictionary fetch; every other
    /// shape fetches the FST once (one planned range, like a match's
    /// build) and walks the column's key range.
    ///
    /// Errors with `FtsError::UnknownColumn` when `column` is not
    /// FTS-indexed in this superfile, like [`Self::token_match`].
    pub(crate) async fn expand_terms(
        &self,
        column: &str,
        pattern: TermPattern<'_>,
        max_terms: usize,
    ) -> Result<(Option<Vec<String>>, MatchWork), FtsError> {
        self.resolve_column_id(column)?;
        if let TermPattern::Exact(term) = pattern {
            return Ok((Some(vec![term.to_owned()]), MatchWork::default()));
        }
        let fst_bytes = self.dict_bytes_async().await?;
        let dict = Self::open_dict(&fst_bytes)?;
        let key_prefix = make_key(column, pattern.walk_prefix());
        // Every key in the walk starts with `<column>\x1F`; the term is
        // what follows.
        let term_start = make_key(column, "").len();
        let mut terms = Vec::new();
        let mut too_many = false;
        dict.for_each_prefix(&key_prefix, |key, _| {
            // Keys are the tokenizer's UTF-8 output, so the conversion
            // holds by construction; a key that fails it is skipped rather
            // than trusted.
            if let Ok(term) = str::from_utf8(&key[term_start..])
                && pattern.covers(term)
            {
                if terms.len() == max_terms {
                    too_many = true;
                    return false;
                }
                terms.push(term.to_owned());
            }
            true
        });
        let work = MatchWork {
            planned_ranges: 1,
            ..MatchWork::default()
        };
        Ok(((!too_many).then_some(terms), work))
    }
}

#[cfg(test)]
mod tests {
    use super::{super::test_util::build_blob, *};

    /// Generous cap so a test never trips the too-many fallback by accident.
    const MAX_TERMS: usize = 64;

    fn expand(r: &FtsReader, pattern: TermPattern<'_>, max_terms: usize) -> Option<Vec<String>> {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        rt.block_on(r.expand_terms("body", pattern, max_terms))
            .expect("expand_terms")
            .0
    }

    #[test]
    fn each_pattern_shape_covers_the_right_terms() {
        // Vocabulary: a, async, boot, is, java, runtime, rust, spring, tokio.
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        assert_eq!(
            expand(&r, TermPattern::Prefix("ru"), MAX_TERMS),
            Some(vec!["runtime".to_owned(), "rust".to_owned()]),
            "a prefix walks only its subtree, in lex order"
        );
        assert_eq!(
            expand(&r, TermPattern::Suffix("me"), MAX_TERMS),
            Some(vec!["runtime".to_owned()])
        );
        assert_eq!(
            expand(&r, TermPattern::Contains("o"), MAX_TERMS),
            Some(vec!["boot".to_owned(), "tokio".to_owned()])
        );
        assert_eq!(
            expand(&r, TermPattern::Exact("rust"), MAX_TERMS),
            Some(vec!["rust".to_owned()]),
            "an exact token is its own expansion"
        );
        assert_eq!(
            expand(&r, TermPattern::Contains("zzz"), MAX_TERMS),
            Some(Vec::new()),
            "nothing qualifies ⇒ an empty (not absent) expansion"
        );
    }

    #[test]
    fn expansion_past_the_cap_is_reported_as_too_many() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        // Nine terms contain a vowel-or-consonant anywhere; a cap of two
        // cannot hold them.
        assert_eq!(expand(&r, TermPattern::Contains(""), 2), None);
        // Exactly at the cap still fits.
        assert_eq!(
            expand(&r, TermPattern::Prefix("ru"), 2),
            Some(vec!["runtime".to_owned(), "rust".to_owned()])
        );
    }

    #[test]
    fn a_dictionary_walk_reports_its_fetch_but_an_exact_token_does_not() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, walked) = rt
            .block_on(r.expand_terms("body", TermPattern::Prefix("ru"), MAX_TERMS))
            .expect("expand");
        assert_eq!(walked.planned_ranges, 1, "one FST fetch per walk");
        let (_, exact) = rt
            .block_on(r.expand_terms("body", TermPattern::Exact("rust"), MAX_TERMS))
            .expect("expand");
        assert_eq!(exact.planned_ranges, 0, "no dictionary needed");
    }

    #[test]
    fn unknown_column_errors_like_a_match_would() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let err = rt
            .block_on(r.expand_terms("nope", TermPattern::Prefix("ru"), MAX_TERMS))
            .expect_err("unknown column");
        assert!(matches!(err, FtsError::UnknownColumn(_)));
    }
}
