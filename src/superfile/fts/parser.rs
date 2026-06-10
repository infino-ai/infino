// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! BM25 query-string parser.
//!
//! Splits a raw BM25 query into positive and negative clauses on a
//! leading `-` sigil.
//!
//! This must run before tokenizing: the tokenizer treats `-` as
//! punctuation, so `"rust -python"` would tokenize to `["rust",
//! "python"]` and lose the marker. So we split on whitespace, decide
//! each run's polarity, then tokenize each side.
//!
//! Polarity only — no validation. A query with no positives is returned
//! as-is; the caller decides that's an error.

use crate::superfile::fts::tokenize::Tokenizer;

/// Positive (scored) and negative (excluding) token lists parsed from
/// a query string.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ParsedQuery {
    pub positives: Vec<String>,
    pub negatives: Vec<String>,
}

/// Split `query` into positive and negative tokens.
///
/// A whitespace-delimited run with a leading `-` and a non-empty
/// remainder is negative; everything else is positive. Only a *leading*
/// `-` negates (`"a-b"` is positive). Both sides go through `tok`, so
/// negated terms get the same normalization as indexed terms.
pub fn parse<T: Tokenizer + ?Sized>(query: &str, tok: &T) -> ParsedQuery {
    let mut parsed = ParsedQuery::default();
    for run in query.split_whitespace() {
        match run.strip_prefix('-') {
            // Leading `-` with a non-empty remainder → negative clause.
            Some(rest) if !rest.is_empty() => {
                tok.tokenize_each(rest, &mut |t| parsed.negatives.push(t.to_owned()));
            }
            // Everything else → positive (a bare `-` tokenizes to nothing).
            _ => {
                tok.tokenize_each(run, &mut |t| parsed.positives.push(t.to_owned()));
            }
        }
    }
    parsed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::fts::tokenize::AsciiLowerTokenizer;

    fn parse(query: &str) -> ParsedQuery {
        super::parse(query, &AsciiLowerTokenizer)
    }

    #[test]
    fn positives_only() {
        let p = parse("rust async");
        assert_eq!(p.positives, vec!["rust", "async"]);
        assert!(p.negatives.is_empty());
    }

    #[test]
    fn single_negative() {
        let p = parse("rust -python");
        assert_eq!(p.positives, vec!["rust"]);
        assert_eq!(p.negatives, vec!["python"]);
    }

    #[test]
    fn multiple_negatives() {
        let p = parse("rust async -python -php");
        assert_eq!(p.positives, vec!["rust", "async"]);
        assert_eq!(p.negatives, vec!["python", "php"]);
    }

    #[test]
    fn negation_only() {
        // No positive clause — the parser reports it faithfully; the
        // caller turns this into an error.
        let p = parse("-python");
        assert!(p.positives.is_empty());
        assert_eq!(p.negatives, vec!["python"]);
    }

    #[test]
    fn interior_hyphen_is_not_negation() {
        // `a-b` is one run with an interior `-`; the tokenizer splits it
        // into two positive tokens. Nothing is negated.
        let p = parse("a-b");
        assert_eq!(p.positives, vec!["a", "b"]);
        assert!(p.negatives.is_empty());
    }

    #[test]
    fn bare_dash_contributes_nothing() {
        let p = parse("rust - python");
        assert_eq!(p.positives, vec!["rust", "python"]);
        assert!(p.negatives.is_empty());
    }

    #[test]
    fn double_dash_strips_one_then_tokenizes() {
        // `--py`: strip the one leading `-`, leaving `-py`; the
        // tokenizer drops the remaining `-` and yields `py`.
        let p = parse("--py");
        assert!(p.positives.is_empty());
        assert_eq!(p.negatives, vec!["py"]);
    }

    #[test]
    fn negated_term_is_normalized() {
        // The negated side is lower-cased like the index.
        let p = parse("rust -PYTHON");
        assert_eq!(p.negatives, vec!["python"]);
    }

    #[test]
    fn empty_query() {
        let p = parse("");
        assert!(p.positives.is_empty());
        assert!(p.negatives.is_empty());
    }
}
