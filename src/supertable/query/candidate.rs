// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Two-pass candidate planning for SQL `WHERE` predicates over
//! FTS-indexed columns.
//!
//! ## Why
//!
//! Without an index, `SELECT title FROM supertable WHERE title = 'rust
//! async runtime'` decodes the whole `title` column and drops the
//! non-matching rows. The inverted index already knows which rows
//! contain a term, so we resolve a small **candidate row set** from the
//! postings and decode only those rows — the row-level analog of the
//! term-bloom *superfile* prune.
//!
//! ## How (two passes)
//!
//!   1. **Candidate generation (this module).** The `WHERE` `Expr` tree
//!      is lowered to a [`CandidatePlan`] — a boolean tree whose leaves
//!      retrieve rows via [`SuperfileReader::token_match`]. Evaluated
//!      against one superfile it yields a `RoaringBitmap` of candidate
//!      `local_doc_id`s, or `None` ("no usable bound — scan the
//!      superfile").
//!   2. **Verification (DataFusion).** The provider turns the candidate
//!      set into a Parquet row selection so only those rows decode, and
//!      DataFusion's `FilterExec` (filters are reported `Inexact`)
//!      re-applies the **exact** predicate. The candidate set only has
//!      to be a *superset* of the true matches.
//!
//! ## Soundness
//!
//! A row equal to `'a b'` tokenizes to a set containing both `a` and
//! `b`, so it is in the term-AND `token_match(col, [a, b], And)`.
//! Requiring the literal's tokens can only keep a non-matching row
//! (wrong order, extra words, different spacing), never drop a matching
//! one — the exact equality is verified in pass 2. `AND` with an
//! un-boundable child drops that child (keeps more rows — still a
//! superset); `OR` with any un-boundable child is itself `Unbounded`;
//! `NOT`, non-FTS columns, range ops, and `LIKE` are `Unbounded` (a
//! word-token index can't soundly bound substring / negation).

use std::collections::HashSet;
use std::sync::Arc;

use datafusion::logical_expr::{Expr, Operator};
use datafusion::scalar::ScalarValue;
use futures::future::BoxFuture;
use roaring::RoaringBitmap;

use crate::superfile::ReadError;
use crate::superfile::SuperfileReader;
use crate::superfile::fts::reader::BoolMode;
use crate::superfile::fts::tokenize::Tokenizer;

/// A superfile-independent boolean plan over FTS term retrievals, lowered
/// once from a SQL `WHERE` clause and [`evaluate`](CandidatePlan::evaluate)d
/// per superfile to a superset of the rows satisfying the FTS-resolvable
/// part of the predicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CandidatePlan {
    /// Rows whose `column` contains every one of `tokens` (term-AND).
    /// The candidate superset of `column = '<text tokenizing to tokens>'`;
    /// the exact predicate is re-verified by the `FilterExec` above the
    /// scan (filters are reported `Inexact`). Resolved per superfile by a
    /// single `token_match(.., And)` — postings only, no column decode —
    /// so verification + projection happen together in DataFusion's one
    /// scan pass. (An `exact_match`-per-leaf alternative was measured and
    /// rejected: it decodes the predicate column in its own pass 2, once
    /// per `OR`/`IN` branch, on top of the scan — multi-decode.)
    TermsAll { column: String, tokens: Vec<String> },
    /// Intersection of children (logical `AND`).
    And(Vec<CandidatePlan>),
    /// Union of children (logical `OR`).
    Or(Vec<CandidatePlan>),
    /// No usable bound: scan the superfile and let `FilterExec` verify.
    Unbounded,
}

impl CandidatePlan {
    /// Lower the conjunction of top-level `filters` (DataFusion ANDs the
    /// provider's filters together) into one plan. `fts_cols` is the set
    /// of FTS-indexed column names; `tokenizer` is the index tokenizer
    /// (absent ⇒ no FTS columns ⇒ always [`Unbounded`]).
    pub(crate) fn from_filters(
        filters: &[Expr],
        fts_cols: &HashSet<&str>,
        tokenizer: Option<&Arc<dyn Tokenizer>>,
    ) -> CandidatePlan {
        let Some(tok) = tokenizer else {
            return CandidatePlan::Unbounded;
        };
        if fts_cols.is_empty() {
            return CandidatePlan::Unbounded;
        }
        and_combine(
            filters
                .iter()
                .map(|f| lower(f, fts_cols, tok.as_ref()))
                .collect(),
        )
    }

    /// Evaluate against one superfile's reader. `Ok(None)` means "no bound
    /// — scan all rows"; `Ok(Some(bitmap))` is the candidate
    /// `local_doc_id` superset (possibly empty). `TermsAll` is one
    /// `token_match(.., And)`; `And`/`Or` intersect/union children.
    pub(crate) fn evaluate<'a>(
        &'a self,
        reader: &'a SuperfileReader,
    ) -> BoxFuture<'a, Result<Option<RoaringBitmap>, ReadError>> {
        Box::pin(async move {
            match self {
                CandidatePlan::Unbounded => Ok(None),
                CandidatePlan::TermsAll { column, tokens } => {
                    let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
                    let docs = reader.token_match(column, &refs, BoolMode::And).await?;
                    Ok(Some(docs.into_iter().collect()))
                }
                CandidatePlan::And(children) => {
                    let mut acc: Option<RoaringBitmap> = None;
                    for c in children {
                        if let Some(bm) = c.evaluate(reader).await? {
                            acc = Some(match acc {
                                Some(a) => a & bm,
                                None => bm,
                            });
                            if acc.as_ref().is_some_and(RoaringBitmap::is_empty) {
                                return Ok(Some(RoaringBitmap::new()));
                            }
                        }
                        // A `None` (unbounded) child adds no constraint.
                    }
                    Ok(acc)
                }
                CandidatePlan::Or(children) => {
                    let mut acc = RoaringBitmap::new();
                    for c in children {
                        match c.evaluate(reader).await? {
                            Some(bm) => acc |= bm,
                            // An unbounded branch makes the union unbounded.
                            None => return Ok(None),
                        }
                    }
                    Ok(Some(acc))
                }
            }
        })
    }
}

impl CandidatePlan {
    /// Cheap upper-bound estimate of how many rows this plan would match
    /// in `reader`'s superfile, computed from per-term `df` only (no
    /// `token_match`, no posting decode). The bound follows the boolean
    /// tree: a term-`AND` can't exceed the **smallest** term's `df`
    /// (`min`); an `OR`/`IN` union can't exceed the **sum** of branch
    /// estimates (capped at `n_docs`); `Unbounded` is `n_docs` (no
    /// bound). The provider uses this to skip the index pushdown when a
    /// predicate would match a large fraction of the superfile — there the
    /// matches saturate the data pages so an index `RowSelection` can't
    /// skip any, and a plain scan is cheaper.
    pub(crate) fn estimate<'a>(
        &'a self,
        reader: &'a SuperfileReader,
    ) -> BoxFuture<'a, Result<u64, ReadError>> {
        Box::pin(async move {
            let n_docs = reader.n_docs();
            match self {
                CandidatePlan::Unbounded => Ok(n_docs),
                CandidatePlan::TermsAll { column, tokens } => {
                    if tokens.is_empty() {
                        return Ok(n_docs);
                    }
                    // Intersection ≤ the rarest token's df.
                    let mut min_df = u64::MAX;
                    for t in tokens {
                        min_df = min_df.min(reader.term_df(column, t).await?);
                    }
                    Ok(min_df.min(n_docs))
                }
                CandidatePlan::And(children) => {
                    let mut m = n_docs;
                    for c in children {
                        m = m.min(c.estimate(reader).await?);
                    }
                    Ok(m)
                }
                CandidatePlan::Or(children) => {
                    let mut sum: u64 = 0;
                    for c in children {
                        sum = sum.saturating_add(c.estimate(reader).await?);
                    }
                    Ok(sum.min(n_docs))
                }
            }
        })
    }
}

/// Lower one `Expr` node.
fn lower(expr: &Expr, fts_cols: &HashSet<&str>, tok: &dyn Tokenizer) -> CandidatePlan {
    match expr {
        Expr::BinaryExpr(be) => match be.op {
            Operator::And => and_combine(vec![
                lower(&be.left, fts_cols, tok),
                lower(&be.right, fts_cols, tok),
            ]),
            Operator::Or => or_combine(vec![
                lower(&be.left, fts_cols, tok),
                lower(&be.right, fts_cols, tok),
            ]),
            Operator::Eq => eq_leaf(&be.left, &be.right, fts_cols, tok),
            // Range / inequality / arithmetic ops aren't term-bounded.
            _ => CandidatePlan::Unbounded,
        },
        // `IN (a, b, …)` on an FTS column is an OR of equalities.
        Expr::InList(il) if !il.negated => in_list_leaf(il, fts_cols, tok),
        // NOT, LIKE, IS NULL, functions, etc. — not soundly term-bounded.
        _ => CandidatePlan::Unbounded,
    }
}

/// Lower `col = 'literal'` (either operand order) on an FTS column.
fn eq_leaf(
    left: &Expr,
    right: &Expr,
    fts_cols: &HashSet<&str>,
    tok: &dyn Tokenizer,
) -> CandidatePlan {
    let (column, value) = match (left, right) {
        (Expr::Column(c), Expr::Literal(v, _)) => (&c.name, v),
        (Expr::Literal(v, _), Expr::Column(c)) => (&c.name, v),
        _ => return CandidatePlan::Unbounded,
    };
    terms_all(column, value, fts_cols, tok)
}

/// Lower `col IN ('a', 'b', …)` on an FTS column to an OR of term-ANDs.
fn in_list_leaf(
    il: &datafusion::logical_expr::expr::InList,
    fts_cols: &HashSet<&str>,
    tok: &dyn Tokenizer,
) -> CandidatePlan {
    let Expr::Column(c) = il.expr.as_ref() else {
        return CandidatePlan::Unbounded;
    };
    let mut branches = Vec::with_capacity(il.list.len());
    for item in &il.list {
        let Expr::Literal(v, _) = item else {
            return CandidatePlan::Unbounded;
        };
        branches.push(terms_all(&c.name, v, fts_cols, tok));
    }
    or_combine(branches)
}

/// Build a `TermsAll` leaf for `column = value`, or `Unbounded` if the
/// column isn't FTS-indexed, the value isn't a string, or it tokenizes
/// to nothing (e.g. the empty string — no tokens to bound with).
fn terms_all(
    column: &str,
    value: &ScalarValue,
    fts_cols: &HashSet<&str>,
    tok: &dyn Tokenizer,
) -> CandidatePlan {
    if !fts_cols.contains(column) {
        return CandidatePlan::Unbounded;
    }
    let Some(s) = scalar_str(value) else {
        return CandidatePlan::Unbounded;
    };
    let tokens: Vec<String> = tok.tokenize(s).collect();
    if tokens.is_empty() {
        return CandidatePlan::Unbounded;
    }
    CandidatePlan::TermsAll {
        column: column.to_owned(),
        tokens,
    }
}

/// Extract a UTF-8 string from a scalar literal, if it is one.
fn scalar_str(v: &ScalarValue) -> Option<&str> {
    match v {
        ScalarValue::Utf8(Some(s))
        | ScalarValue::LargeUtf8(Some(s))
        | ScalarValue::Utf8View(Some(s)) => Some(s.as_str()),
        _ => None,
    }
}

/// Combine children under `AND`: an `Unbounded` child drops out (adds
/// no constraint), nested `And`s flatten, all-unbounded → `Unbounded`.
fn and_combine(children: Vec<CandidatePlan>) -> CandidatePlan {
    let mut flat = Vec::with_capacity(children.len());
    for c in children {
        match c {
            CandidatePlan::Unbounded => {}
            CandidatePlan::And(inner) => flat.extend(inner),
            other => flat.push(other),
        }
    }
    collapse(flat, true)
}

/// Combine children under `OR`: any `Unbounded` child makes the whole
/// union `Unbounded`; nested `Or`s flatten.
fn or_combine(children: Vec<CandidatePlan>) -> CandidatePlan {
    let mut flat = Vec::with_capacity(children.len());
    for c in children {
        match c {
            CandidatePlan::Unbounded => return CandidatePlan::Unbounded,
            CandidatePlan::Or(inner) => flat.extend(inner),
            other => flat.push(other),
        }
    }
    collapse(flat, false)
}

/// Wrap a flattened child list back into `And`/`Or`, collapsing the
/// 0- and 1-child degenerate cases.
fn collapse(mut flat: Vec<CandidatePlan>, is_and: bool) -> CandidatePlan {
    match flat.len() {
        0 => CandidatePlan::Unbounded,
        1 => flat.pop().expect("len checked == 1"),
        _ if is_and => CandidatePlan::And(flat),
        _ => CandidatePlan::Or(flat),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::logical_expr::expr::InList;
    use datafusion::prelude::{col, lit};

    use crate::superfile::fts::tokenize::AsciiLowerTokenizer;

    fn fts_cols() -> HashSet<&'static str> {
        let mut s = HashSet::new();
        s.insert("title");
        s
    }

    fn tok() -> Arc<dyn Tokenizer> {
        Arc::new(AsciiLowerTokenizer)
    }

    fn plan(expr: Expr) -> CandidatePlan {
        CandidatePlan::from_filters(&[expr], &fts_cols(), Some(&tok()))
    }

    #[test]
    fn eq_on_fts_column_lowers_to_terms_all() {
        let p = plan(col("title").eq(lit("rust async")));
        assert_eq!(
            p,
            CandidatePlan::TermsAll {
                column: "title".into(),
                tokens: vec!["rust".into(), "async".into()],
            }
        );
    }

    #[test]
    fn eq_operands_reversed_still_lowers() {
        let p = plan(lit("rust").eq(col("title")));
        assert_eq!(
            p,
            CandidatePlan::TermsAll {
                column: "title".into(),
                tokens: vec!["rust".into()],
            }
        );
    }

    #[test]
    fn eq_on_non_fts_column_is_unbounded() {
        assert_eq!(
            plan(col("category").eq(lit("rust"))),
            CandidatePlan::Unbounded
        );
    }

    #[test]
    fn empty_literal_is_unbounded() {
        assert_eq!(plan(col("title").eq(lit(""))), CandidatePlan::Unbounded);
    }

    #[test]
    fn range_op_is_unbounded() {
        assert_eq!(plan(col("title").gt(lit("m"))), CandidatePlan::Unbounded);
    }

    #[test]
    fn and_of_fts_and_non_fts_keeps_only_fts_branch() {
        let p = plan(
            col("title")
                .eq(lit("rust"))
                .and(col("category").eq(lit("lang"))),
        );
        assert_eq!(
            p,
            CandidatePlan::TermsAll {
                column: "title".into(),
                tokens: vec!["rust".into()],
            }
        );
    }

    #[test]
    fn and_of_two_fts_equalities_intersects() {
        let p = plan(
            col("title")
                .eq(lit("rust"))
                .and(col("title").eq(lit("async"))),
        );
        match p {
            CandidatePlan::And(children) => assert_eq!(children.len(), 2),
            other => panic!("expected And, got {other:?}"),
        }
    }

    #[test]
    fn or_of_two_fts_equalities_unions() {
        let p = plan(
            col("title")
                .eq(lit("rust"))
                .or(col("title").eq(lit("python"))),
        );
        match p {
            CandidatePlan::Or(children) => assert_eq!(children.len(), 2),
            other => panic!("expected Or, got {other:?}"),
        }
    }

    #[test]
    fn or_with_non_fts_branch_is_unbounded() {
        let p = plan(
            col("title")
                .eq(lit("rust"))
                .or(col("category").eq(lit("lang"))),
        );
        assert_eq!(p, CandidatePlan::Unbounded);
    }

    #[test]
    fn not_is_unbounded() {
        assert_eq!(
            plan(!col("title").eq(lit("rust"))),
            CandidatePlan::Unbounded
        );
    }

    #[test]
    fn not_eq_is_unbounded() {
        // `title != 'rust'` (Operator::NotEq) can't be term-bounded.
        assert_eq!(
            plan(col("title").not_eq(lit("rust"))),
            CandidatePlan::Unbounded
        );
    }

    #[test]
    fn and_with_not_child_keeps_fts_branch() {
        // `title = 'rust' AND NOT (title = 'compiler')` — the NOT branch
        // is un-boundable and drops out of candidate generation (verified
        // in pass 2), so candidates still come from the FTS branch.
        let p = plan(
            col("title")
                .eq(lit("rust"))
                .and(!col("title").eq(lit("compiler"))),
        );
        assert_eq!(
            p,
            CandidatePlan::TermsAll {
                column: "title".into(),
                tokens: vec!["rust".into()],
            }
        );
    }

    #[test]
    fn like_is_unbounded() {
        assert_eq!(
            plan(col("title").like(lit("rust%"))),
            CandidatePlan::Unbounded
        );
    }

    #[test]
    fn in_list_on_fts_column_is_or_of_terms_all() {
        let expr = Expr::InList(InList::new(
            Box::new(col("title")),
            vec![lit("rust"), lit("python")],
            false,
        ));
        match plan(expr) {
            CandidatePlan::Or(children) => {
                assert_eq!(children.len(), 2);
                assert!(matches!(children[0], CandidatePlan::TermsAll { .. }));
            }
            other => panic!("expected Or, got {other:?}"),
        }
    }

    #[test]
    fn negated_in_list_is_unbounded() {
        let expr = Expr::InList(InList::new(Box::new(col("title")), vec![lit("rust")], true));
        assert_eq!(plan(expr), CandidatePlan::Unbounded);
    }

    #[test]
    fn no_tokenizer_is_unbounded() {
        let p = CandidatePlan::from_filters(&[col("title").eq(lit("rust"))], &fts_cols(), None);
        assert_eq!(p, CandidatePlan::Unbounded);
    }
}
