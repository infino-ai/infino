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
//! `NOT`, non-FTS columns, and range ops are `Unbounded` (a word-token
//! index can't soundly bound negation or ordering).
//!
//! `LIKE` is bounded through the term dictionary. The pattern is split
//! at its wildcards into literal fragments and each fragment is tokenized
//! with the column's analyzer. A token the fragment closes on both sides
//! (a separator inside the fragment, or the pattern's own start / end)
//! must be indexed as itself; a token bordering a wildcard may be the
//! head or tail of a longer indexed term and is widened, per superfile,
//! to every term that starts with / ends with / contains it. Which edges
//! count as closed, and which open tokens can be used at all, depends on
//! the analyzer — see `Analyzer`. `NOT LIKE` and `ILIKE` stay
//! `Unbounded`.

use std::{collections::HashSet, mem, sync::Arc};

use datafusion::{
    logical_expr::{
        Expr, Operator,
        expr::{InList, Like},
    },
    scalar::ScalarValue,
};
use futures::future::BoxFuture;
use roaring::RoaringBitmap;

use crate::{
    superfile::{
        ReadError, SuperfileReader,
        fts::{
            reader::{BoolMode, MatchWork, TermPattern},
            tokenize::{ASCII_LOWER_TOKENIZER, STANDARD_TOKENIZER, Tokenizer},
        },
    },
    supertable::{
        error::QueryError,
        manifest::ManifestSnapshot,
        query::prune::{PruneLeaf, select_superfiles},
    },
};

/// Most indexed terms one `LIKE` fragment token may widen to before the
/// index gives up on it. Each expanded term costs a df probe and a
/// posting walk, and a token this broad matches enough rows that the
/// scan wins anyway — the provider's selectivity gate would send it there
/// after paying for the probes. Sibling of the provider's `PUSHDOWN_*`
/// gates.
pub(crate) const LIKE_MAX_TERMS: usize = 1024;

/// `LIKE` wildcard matching any run of characters, including none.
const LIKE_ANY: char = '%';

/// `LIKE` wildcard matching exactly one character.
const LIKE_ONE: char = '_';

/// The escape Arrow's `LIKE` kernel reads: the character after it is
/// literal, whatever it is.
const LIKE_ESCAPE: char = '\\';

/// ASCII characters UAX #29 lets join two words — apostrophe, quote,
/// full stop, colon, comma, semicolon, underscore (`don't`, `3.5`,
/// `a:b`, `1,000`, `x_y`, a Hebrew `"`). Any other ASCII non-alphanumeric
/// is a hard word break under the `standard` analyzer.
const WORD_JOINERS: &[char] = &['\'', '"', '.', ':', ',', ';', '_'];

/// Lowercase final sigma: `to_lowercase` spells a word-final `Σ` this way
/// and a medial one `σ` — the one context-sensitive mapping in Unicode
/// lowercasing.
const FINAL_SIGMA: char = 'ς';

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
    /// Rows whose `column` contains any one of `terms` (term-OR): one
    /// `LIKE` fragment token bound to a superfile's vocabulary by
    /// [`expand`](Self::expand). Resolved by a single `token_match(.., Or)`;
    /// empty `terms` (nothing in that superfile's dictionary qualifies)
    /// matches no row, so the superfile is skipped without a scan.
    TermsAny { column: String, terms: Vec<String> },
    /// Rows whose `column` satisfies every `LIKE` fragment token, before
    /// the superfile is known: a complete token must be indexed as itself,
    /// an open-edged one as some term it is the head / tail / infix of.
    /// [`expand`](Self::expand) turns it into an `And` of `TermsAny` per
    /// superfile; `evaluate` / `estimate` do the same inline when handed
    /// the unexpanded leaf.
    TermsLike {
        column: String,
        tokens: Vec<LikeToken>,
    },
    /// Intersection of children (logical `AND`).
    And(Vec<CandidatePlan>),
    /// Union of children (logical `OR`).
    Or(Vec<CandidatePlan>),
    /// No usable bound: scan the superfile and let `FilterExec` verify.
    Unbounded,
}

/// One token of a `LIKE` fragment, as the column's analyzer produced it,
/// with which of its ends may sit mid-term in a matching row's text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LikeToken {
    /// The analyzer's token text (already split and lowercased).
    pub(crate) text: String,
    /// A wildcard precedes the token with no separator in between, so a
    /// matching row may hold it as the tail of a longer term.
    pub(crate) open_left: bool,
    /// A wildcard follows the token with no separator in between, so a
    /// matching row may hold it as the head of a longer term.
    pub(crate) open_right: bool,
}

impl LikeToken {
    /// Closed on both sides: the token must be indexed exactly as itself.
    fn is_complete(&self) -> bool {
        !self.open_left && !self.open_right
    }

    /// The dictionary shape the open edges call for.
    fn pattern(&self) -> TermPattern<'_> {
        match (self.open_left, self.open_right) {
            (false, false) => TermPattern::Exact(&self.text),
            (false, true) => TermPattern::Prefix(&self.text),
            (true, false) => TermPattern::Suffix(&self.text),
            (true, true) => TermPattern::Contains(&self.text),
        }
    }
}

impl CandidatePlan {
    /// Lower the conjunction of top-level `filters` (DataFusion ANDs the
    /// provider's filters together) into one plan. `fts_cols` is the set
    /// of FTS-indexed column names; `resolve` maps an FTS column to the
    /// tokenizer it was indexed with, so per-column analyzers lower query
    /// text the same way the column was tokenized at ingest. Empty
    /// `fts_cols` ⇒ no FTS columns ⇒ always [`Unbounded`].
    pub(crate) fn from_filters(
        filters: &[Expr],
        fts_cols: &HashSet<&str>,
        resolve: &dyn Fn(&str) -> Arc<dyn Tokenizer>,
    ) -> CandidatePlan {
        if fts_cols.is_empty() {
            return CandidatePlan::Unbounded;
        }
        and_combine(
            filters
                .iter()
                .map(|f| lower(f, fts_cols, resolve))
                .collect(),
        )
    }

    /// Evaluate against one superfile's reader. `Ok(None)` means "no bound
    /// — scan all rows"; `Ok(Some(bitmap))` is the candidate
    /// `local_doc_id` superset (possibly empty). `TermsAll` is one
    /// `token_match(.., And)`; `And`/`Or` intersect/union children.
    /// The second element sums the posting-walk work of every `TermsAll`
    /// leaf the evaluation touched (an early-out keeps the work already
    /// done), so the caller can flush it per superfile.
    pub(crate) fn evaluate<'a>(
        &'a self,
        reader: &'a SuperfileReader,
    ) -> BoxFuture<'a, Result<(Option<RoaringBitmap>, MatchWork), ReadError>> {
        Box::pin(async move {
            match self {
                CandidatePlan::Unbounded => Ok((None, MatchWork::default())),
                CandidatePlan::TermsAll { column, tokens } => {
                    let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
                    let (docs, work) = reader.token_match(column, &refs, BoolMode::And).await?;
                    Ok((Some(docs.into_iter().collect()), work))
                }
                CandidatePlan::TermsAny { column, terms } => {
                    // No qualifying term in this superfile ⇒ no row; the
                    // match returns the empty set for an empty term list.
                    let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
                    let (docs, work) = reader.token_match(column, &refs, BoolMode::Or).await?;
                    Ok((Some(docs.into_iter().collect()), work))
                }
                CandidatePlan::TermsLike { column, tokens } => {
                    let (expanded, mut work) = expand_like(reader, column, tokens).await?;
                    let (docs, eval_work) = expanded.evaluate(reader).await?;
                    work.merge(eval_work);
                    Ok((docs, work))
                }
                CandidatePlan::And(children) => {
                    let mut acc: Option<RoaringBitmap> = None;
                    let mut work = MatchWork::default();
                    for c in children {
                        let (child, child_work) = c.evaluate(reader).await?;
                        work.merge(child_work);
                        if let Some(bm) = child {
                            acc = Some(match acc {
                                Some(a) => a & bm,
                                None => bm,
                            });
                            if acc.as_ref().is_some_and(RoaringBitmap::is_empty) {
                                return Ok((Some(RoaringBitmap::new()), work));
                            }
                        }
                        // A `None` (unbounded) child adds no constraint.
                    }
                    Ok((acc, work))
                }
                CandidatePlan::Or(children) => {
                    let mut acc = RoaringBitmap::new();
                    let mut work = MatchWork::default();
                    for c in children {
                        let (child, child_work) = c.evaluate(reader).await?;
                        work.merge(child_work);
                        match child {
                            Some(bm) => acc |= bm,
                            // An unbounded branch makes the union unbounded.
                            None => return Ok((None, work)),
                        }
                    }
                    Ok((Some(acc), work))
                }
            }
        })
    }
}

impl CandidatePlan {
    /// ManifestSnapshot-only superfile survival gate for this plan: the superfile
    /// ids that *could* match according to term blooms. `None` means no
    /// gate — the plan is [`Unbounded`], contains an `OR`, or otherwise
    /// cannot be expressed as a conjunction of bloom leaves.
    pub(crate) async fn surviving_superfile_ids(
        &self,
        manifest: &ManifestSnapshot,
    ) -> Result<Option<HashSet<u128>>, QueryError> {
        let mut groups = Vec::new();
        if !self.collect_survival_or_groups(&mut groups) {
            return Ok(None);
        }
        if groups.is_empty() {
            return Ok(Some(HashSet::new()));
        }
        let mut acc = HashSet::new();
        for leaves in groups {
            if leaves.is_empty() {
                continue;
            }
            acc.extend(
                select_superfiles(manifest, &leaves)
                    .await?
                    .iter()
                    .map(|e| e.superfile_id.as_u128()),
            );
        }
        Ok(Some(acc))
    }

    /// Flatten this plan into one bloom-conjunction group per `OR` branch.
    fn collect_survival_or_groups(&self, groups: &mut Vec<Vec<PruneLeaf>>) -> bool {
        match self {
            CandidatePlan::Or(children) => children
                .iter()
                .all(|child| child.collect_survival_or_branch(groups)),
            other => other.collect_survival_or_branch(groups),
        }
    }

    /// Append one `OR` branch as a single conjunctive leaf group. A branch
    /// that yields no leaf at all (a `LIKE` whose tokens are all open on
    /// the left — no summary bounds a suffix or infix) constrains no
    /// superfile, so the whole disjunction has no gate: `false`, not an
    /// empty group, which the caller would read as "nothing survives".
    fn collect_survival_or_branch(&self, groups: &mut Vec<Vec<PruneLeaf>>) -> bool {
        match self {
            CandidatePlan::Unbounded => false,
            CandidatePlan::Or(children) => children
                .iter()
                .all(|child| child.collect_survival_or_branch(groups)),
            other => {
                let mut leaves = Vec::new();
                if !other.append_prune_leaves(&mut leaves) || leaves.is_empty() {
                    return false;
                }
                groups.push(leaves);
                true
            }
        }
    }

    /// Append conjunctive [`PruneLeaf`]s for this subtree. Returns `false`
    /// when the subtree contains an `OR` (not expressible as one bloom
    /// conjunction).
    fn append_prune_leaves(&self, leaves: &mut Vec<PruneLeaf>) -> bool {
        match self {
            CandidatePlan::Unbounded => true,
            CandidatePlan::TermsAll { column, tokens } => {
                if tokens.is_empty() {
                    return true;
                }
                leaves.push(PruneLeaf::TermPresence {
                    column: column.clone(),
                    terms: tokens.clone(),
                    mode: BoolMode::And,
                });
                true
            }
            CandidatePlan::TermsAny { column, terms } => {
                if !terms.is_empty() {
                    leaves.push(PruneLeaf::TermPresence {
                        column: column.clone(),
                        terms: terms.clone(),
                        mode: BoolMode::Or,
                    });
                }
                true
            }
            CandidatePlan::TermsLike { column, tokens } => {
                // A complete token must be present as itself → term bloom;
                // a token open only on the right must head some term → lex
                // term-range overlap. A token open on the left bounds no
                // manifest summary.
                let complete: Vec<String> = tokens
                    .iter()
                    .filter(|t| t.is_complete())
                    .map(|t| t.text.clone())
                    .collect();
                if !complete.is_empty() {
                    leaves.push(PruneLeaf::TermPresence {
                        column: column.clone(),
                        terms: complete,
                        mode: BoolMode::And,
                    });
                }
                for token in tokens.iter().filter(|t| !t.open_left && t.open_right) {
                    leaves.push(PruneLeaf::Prefix {
                        column: column.clone(),
                        prefix: token.text.as_bytes().to_vec(),
                    });
                }
                true
            }
            CandidatePlan::And(children) => children
                .iter()
                .all(|child| child.append_prune_leaves(leaves)),
            CandidatePlan::Or(_) => false,
        }
    }

    /// Cheap upper-bound estimate of how many rows this plan would match
    /// in `reader`'s superfile, computed from per-term `df` only (no
    /// `token_match`, no posting decode). The bound follows the boolean
    /// tree: a term-`AND` can't exceed the **smallest** term's `df`
    /// (`min`); a term-`OR` (an expanded `LIKE` token) and an `OR`/`IN`
    /// union can't exceed the **sum** of their parts (capped at `n_docs`);
    /// `Unbounded` is `n_docs` (no bound). The provider uses this to skip
    /// the index pushdown when a predicate would match a large fraction of
    /// the superfile — there the matches saturate the data pages so an
    /// index `RowSelection` can't skip any, and a plain scan is cheaper.
    /// The second element sums the header-fetch work of every leaf df
    /// probe (and the dictionary walk of an unexpanded `LIKE` leaf), so
    /// the caller can flush it per superfile.
    pub(crate) fn estimate<'a>(
        &'a self,
        reader: &'a SuperfileReader,
    ) -> BoxFuture<'a, Result<(u64, MatchWork), ReadError>> {
        Box::pin(async move {
            let n_docs = reader.n_docs();
            match self {
                CandidatePlan::Unbounded => Ok((n_docs, MatchWork::default())),
                CandidatePlan::TermsAll { column, tokens } => {
                    if tokens.is_empty() {
                        return Ok((n_docs, MatchWork::default()));
                    }
                    // Intersection ≤ the rarest token's df — resolved with
                    // one batched df lookup (single FST parse + coalesced
                    // header fetch) rather than one parse + fetch per token.
                    let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
                    let (dfs, work) = reader.term_dfs(column, &refs).await?;
                    let min_df = dfs.into_iter().min().unwrap_or(u64::MAX);
                    Ok((min_df.min(n_docs), work))
                }
                CandidatePlan::TermsAny { column, terms } => {
                    if terms.is_empty() {
                        return Ok((0, MatchWork::default()));
                    }
                    // Union ≤ the sum of the terms' dfs.
                    let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
                    let (dfs, work) = reader.term_dfs(column, &refs).await?;
                    let sum = dfs.into_iter().fold(0u64, u64::saturating_add);
                    Ok((sum.min(n_docs), work))
                }
                CandidatePlan::TermsLike { column, tokens } => {
                    let (expanded, mut work) = expand_like(reader, column, tokens).await?;
                    let (rows, est_work) = expanded.estimate(reader).await?;
                    work.merge(est_work);
                    Ok((rows, work))
                }
                CandidatePlan::And(children) => {
                    let mut m = n_docs;
                    let mut work = MatchWork::default();
                    for c in children {
                        let (child, child_work) = c.estimate(reader).await?;
                        work.merge(child_work);
                        m = m.min(child);
                    }
                    Ok((m, work))
                }
                CandidatePlan::Or(children) => {
                    let mut sum: u64 = 0;
                    let mut work = MatchWork::default();
                    for c in children {
                        let (child, child_work) = c.estimate(reader).await?;
                        work.merge(child_work);
                        sum = sum.saturating_add(child);
                    }
                    Ok((sum.min(n_docs), work))
                }
            }
        })
    }
}

impl CandidatePlan {
    /// Whether any leaf still needs a superfile's dictionary
    /// ([`TermsLike`](Self::TermsLike)). The provider expands such a plan
    /// once per superfile and estimates / evaluates the result, instead of
    /// paying the dictionary walk in both steps.
    pub(crate) fn has_like(&self) -> bool {
        match self {
            CandidatePlan::TermsLike { .. } => true,
            CandidatePlan::And(children) | CandidatePlan::Or(children) => {
                children.iter().any(CandidatePlan::has_like)
            }
            CandidatePlan::TermsAll { .. }
            | CandidatePlan::TermsAny { .. }
            | CandidatePlan::Unbounded => false,
        }
    }

    /// Bind every [`TermsLike`](Self::TermsLike) leaf to `reader`'s
    /// superfile: each fragment token becomes the
    /// [`TermsAny`](Self::TermsAny) of the indexed terms it covers, or
    /// `Unbounded` (dropping out of its `AND`) when more than
    /// [`LIKE_MAX_TERMS`] qualify. Every other node is copied. The work is
    /// the dictionary fetches performed.
    pub(crate) fn expand<'a>(
        &'a self,
        reader: &'a SuperfileReader,
    ) -> BoxFuture<'a, Result<(CandidatePlan, MatchWork), ReadError>> {
        Box::pin(async move {
            match self {
                CandidatePlan::TermsLike { column, tokens } => {
                    expand_like(reader, column, tokens).await
                }
                CandidatePlan::And(children) => {
                    let (expanded, work) = expand_children(reader, children).await?;
                    Ok((and_combine(expanded), work))
                }
                CandidatePlan::Or(children) => {
                    let (expanded, work) = expand_children(reader, children).await?;
                    Ok((or_combine(expanded), work))
                }
                leaf => Ok((leaf.clone(), MatchWork::default())),
            }
        })
    }
}

/// Expand each child in order, summing the dictionary work.
async fn expand_children(
    reader: &SuperfileReader,
    children: &[CandidatePlan],
) -> Result<(Vec<CandidatePlan>, MatchWork), ReadError> {
    let mut expanded = Vec::with_capacity(children.len());
    let mut work = MatchWork::default();
    for child in children {
        let (plan, child_work) = child.expand(reader).await?;
        work.merge(child_work);
        expanded.push(plan);
    }
    Ok((expanded, work))
}

/// Bind one `TermsLike` leaf to a superfile: the `AND` of each token's
/// expansion. A token widening past [`LIKE_MAX_TERMS`] contributes no
/// constraint (`Unbounded`, which `and_combine` drops); every token too
/// wide ⇒ the leaf is `Unbounded` and the superfile scans.
async fn expand_like(
    reader: &SuperfileReader,
    column: &str,
    tokens: &[LikeToken],
) -> Result<(CandidatePlan, MatchWork), ReadError> {
    let mut parts = Vec::with_capacity(tokens.len());
    let mut work = MatchWork::default();
    for token in tokens {
        let (terms, expand_work) = reader
            .expand_terms(column, token.pattern(), LIKE_MAX_TERMS)
            .await?;
        work.merge(expand_work);
        parts.push(match terms {
            Some(terms) => CandidatePlan::TermsAny {
                column: column.to_owned(),
                terms,
            },
            None => CandidatePlan::Unbounded,
        });
    }
    Ok((and_combine(parts), work))
}

/// Lower one `Expr` node.
fn lower(
    expr: &Expr,
    fts_cols: &HashSet<&str>,
    resolve: &dyn Fn(&str) -> Arc<dyn Tokenizer>,
) -> CandidatePlan {
    match expr {
        Expr::BinaryExpr(be) => match be.op {
            Operator::And => and_combine(vec![
                lower(&be.left, fts_cols, resolve),
                lower(&be.right, fts_cols, resolve),
            ]),
            Operator::Or => or_combine(vec![
                lower(&be.left, fts_cols, resolve),
                lower(&be.right, fts_cols, resolve),
            ]),
            Operator::Eq => eq_leaf(&be.left, &be.right, fts_cols, resolve),
            // Range / inequality / arithmetic ops aren't term-bounded.
            _ => CandidatePlan::Unbounded,
        },
        // `IN (a, b, …)` on an FTS column is an OR of equalities.
        Expr::InList(il) if !il.negated => in_list_leaf(il, fts_cols, resolve),
        // `LIKE` on an FTS column is bounded through the term dictionary.
        Expr::Like(like) => like_leaf(like, fts_cols, resolve),
        // NOT, IS NULL, functions, etc. — not soundly term-bounded.
        _ => CandidatePlan::Unbounded,
    }
}

/// Lower `col = 'literal'` (either operand order) on an FTS column.
fn eq_leaf(
    left: &Expr,
    right: &Expr,
    fts_cols: &HashSet<&str>,
    resolve: &dyn Fn(&str) -> Arc<dyn Tokenizer>,
) -> CandidatePlan {
    let (column, value) = match (left, right) {
        (Expr::Column(c), Expr::Literal(v, _)) => (&c.name, v),
        (Expr::Literal(v, _), Expr::Column(c)) => (&c.name, v),
        _ => return CandidatePlan::Unbounded,
    };
    terms_all(column, value, fts_cols, resolve)
}

/// Lower `col IN ('a', 'b', …)` on an FTS column to an OR of term-ANDs.
fn in_list_leaf(
    il: &InList,
    fts_cols: &HashSet<&str>,
    resolve: &dyn Fn(&str) -> Arc<dyn Tokenizer>,
) -> CandidatePlan {
    let Expr::Column(c) = il.expr.as_ref() else {
        return CandidatePlan::Unbounded;
    };
    let mut branches = Vec::with_capacity(il.list.len());
    for item in &il.list {
        let Expr::Literal(v, _) = item else {
            return CandidatePlan::Unbounded;
        };
        branches.push(terms_all(&c.name, v, fts_cols, resolve));
    }
    or_combine(branches)
}

/// Lower `col LIKE 'pattern'` on an FTS column. The pattern's literal
/// fragments are tokenized with the column's analyzer; every token the
/// analyzer can bound soundly becomes a constraint (see `Analyzer::admits`).
/// All-complete tokens are the same term-AND an equality lowers to;
/// otherwise the leaf waits for a superfile's dictionary. `Unbounded` for
/// `NOT LIKE`, `ILIKE`, a non-column or non-literal operand, a non-FTS
/// column, an escape other than `\`, or a pattern with no usable token.
fn like_leaf(
    like: &Like,
    fts_cols: &HashSet<&str>,
    resolve: &dyn Fn(&str) -> Arc<dyn Tokenizer>,
) -> CandidatePlan {
    // `NOT LIKE` excludes rows — no term set bounds an exclusion. `ILIKE`
    // matches under Unicode simple case folding, which is wider than the
    // analyzers' lowercasing (`ſ` folds to `s`; `to_lowercase` keeps it),
    // so a folded match could hide behind a term the pattern's lowercased
    // token never reaches.
    if like.negated || like.case_insensitive {
        return CandidatePlan::Unbounded;
    }
    // Arrow's kernel reads `\` as the escape; the executor rejects others.
    if like.escape_char.is_some_and(|c| c != LIKE_ESCAPE) {
        return CandidatePlan::Unbounded;
    }
    let (Expr::Column(c), Expr::Literal(v, _)) = (like.expr.as_ref(), like.pattern.as_ref()) else {
        return CandidatePlan::Unbounded;
    };
    if !fts_cols.contains(c.name.as_str()) {
        return CandidatePlan::Unbounded;
    }
    let Some(pattern) = scalar_str(v) else {
        return CandidatePlan::Unbounded;
    };
    let tok = resolve(&c.name);
    let Some(analyzer) = Analyzer::of(tok.as_ref()) else {
        return CandidatePlan::Unbounded;
    };
    let Some(fragments) = like_fragments(pattern) else {
        return CandidatePlan::Unbounded;
    };
    let tokens: Vec<LikeToken> = fragments
        .iter()
        .flat_map(|fragment| fragment.tokens(tok.as_ref(), analyzer))
        .collect();
    if tokens.is_empty() {
        return CandidatePlan::Unbounded;
    }
    if tokens.iter().all(LikeToken::is_complete) {
        return CandidatePlan::TermsAll {
            column: c.name.clone(),
            tokens: tokens.into_iter().map(|t| t.text).collect(),
        };
    }
    CandidatePlan::TermsLike {
        column: c.name.clone(),
        tokens,
    }
}

/// One maximal run of literal (non-wildcard) pattern characters, and
/// whether the pattern's own start / end bounds it (no wildcard before /
/// after it).
#[derive(Debug, PartialEq, Eq)]
struct Fragment {
    text: String,
    at_start: bool,
    at_end: bool,
}

impl Fragment {
    /// Tokenize the fragment with the column's analyzer and mark each
    /// token's open edges. An interior token is bordered by separators
    /// inside the fragment on both sides. The first token is closed on the
    /// left when the pattern starts here or the fragment's first character
    /// is a hard separator (the token began after it); the last token
    /// likewise on the right. Any other edge may continue into the text a
    /// wildcard stands for. Tokens the analyzer cannot bound soundly are
    /// dropped — a dropped constraint keeps a superset.
    fn tokens(&self, tok: &dyn Tokenizer, analyzer: Analyzer) -> Vec<LikeToken> {
        let texts: Vec<String> = tok.tokenize(&self.text).collect();
        let n = texts.len();
        let left_closed = self.at_start
            || self
                .text
                .chars()
                .next()
                .is_some_and(|c| analyzer.hard_separator(c));
        let right_closed = self.at_end
            || self
                .text
                .chars()
                .next_back()
                .is_some_and(|c| analyzer.hard_separator(c));
        texts
            .into_iter()
            .enumerate()
            .filter_map(|(i, text)| {
                analyzer.admits(LikeToken {
                    text,
                    open_left: i == 0 && !left_closed,
                    open_right: i + 1 == n && !right_closed,
                })
            })
            .collect()
    }
}

/// Split a `LIKE` pattern at its wildcards into literal fragments,
/// unescaping `\x` to a literal `x`. `None` for a trailing `\`, which the
/// executor rejects — nothing to plan.
fn like_fragments(pattern: &str) -> Option<Vec<Fragment>> {
    let mut fragments = Vec::new();
    let mut text = String::new();
    // No wildcard seen yet: the next fragment starts where the pattern does.
    let mut at_start = true;
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            LIKE_ESCAPE => text.push(chars.next()?),
            LIKE_ANY | LIKE_ONE => {
                if !text.is_empty() {
                    fragments.push(Fragment {
                        text: mem::take(&mut text),
                        at_start,
                        at_end: false,
                    });
                }
                at_start = false;
            }
            other => text.push(other),
        }
    }
    if !text.is_empty() {
        fragments.push(Fragment {
            text,
            at_start,
            at_end: true,
        });
    }
    Some(fragments)
}

/// Which shipped analyzer indexed a column. It decides which fragment
/// edges are closed and which open tokens the index can bound at all; an
/// analyzer this module does not know keeps `LIKE` `Unbounded`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Analyzer {
    /// `ascii_lower`: `[A-Za-z0-9]` runs, every other ASCII byte a
    /// separator, a run holding any non-ASCII byte dropped whole.
    AsciiLower,
    /// `standard`: UAX #29 words, lowercased with `to_lowercase`.
    Standard,
}

impl Analyzer {
    fn of(tok: &dyn Tokenizer) -> Option<Analyzer> {
        match tok.name() {
            ASCII_LOWER_TOKENIZER => Some(Analyzer::AsciiLower),
            STANDARD_TOKENIZER => Some(Analyzer::Standard),
            _ => None,
        }
    }

    /// A character that ends a token wherever it appears, so a token next
    /// to it inside a fragment has the same boundary in any row's text.
    fn hard_separator(self, c: char) -> bool {
        match self {
            // Every ASCII byte outside `[A-Za-z0-9]` splits a run; a
            // non-ASCII byte extends (and poisons) one instead.
            Analyzer::AsciiLower => c.is_ascii() && !c.is_ascii_alphanumeric(),
            // UAX #29 may join a word across a `WORD_JOINERS` character
            // (`don't`); every other ASCII non-alphanumeric always breaks.
            // Non-ASCII punctuation is left open rather than classified.
            Analyzer::Standard => {
                c.is_ascii() && !c.is_ascii_alphanumeric() && !WORD_JOINERS.contains(&c)
            }
        }
    }

    /// Whether the index can soundly require `token` of a matching row.
    fn admits(self, token: LikeToken) -> Option<LikeToken> {
        match self {
            // A run holding any non-ASCII byte is dropped whole, so a term
            // that merely *contains* a fragment token may not exist
            // (`Firefox—the` indexes nothing). Only a token the fragment
            // closes on both sides is guaranteed indexed as itself.
            Analyzer::AsciiLower if !token.is_complete() => None,
            // `to_lowercase` spells a word-final `Σ` as `ς` and a medial
            // one as `σ`, so a token whose end may sit mid-word has two
            // possible spellings in the dictionary.
            Analyzer::Standard if token.open_right && token.text.ends_with(FINAL_SIGMA) => None,
            _ => Some(token),
        }
    }
}

/// Build a `TermsAll` leaf for `column = value`, or `Unbounded` if the
/// column isn't FTS-indexed, the value isn't a string, or it tokenizes
/// to nothing (e.g. the empty string — no tokens to bound with).
fn terms_all(
    column: &str,
    value: &ScalarValue,
    fts_cols: &HashSet<&str>,
    resolve: &dyn Fn(&str) -> Arc<dyn Tokenizer>,
) -> CandidatePlan {
    if !fts_cols.contains(column) {
        return CandidatePlan::Unbounded;
    }
    let Some(s) = scalar_str(value) else {
        return CandidatePlan::Unbounded;
    };
    let tok = resolve(column);
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

/// Manifest prune leaves for the `LIKE` predicates in `filters`: the same
/// lowering as the candidate plan, reduced to what a superfile summary can
/// answer — a term bloom for complete tokens and a lex-range check for a
/// prefix token. Descends `AND` and aliases; anything else contributes
/// nothing (the superfile is kept).
pub(crate) fn like_prune_leaves(
    filters: &[Expr],
    fts_cols: &HashSet<&str>,
    resolve: &dyn Fn(&str) -> Arc<dyn Tokenizer>,
) -> Vec<PruneLeaf> {
    let mut out = Vec::new();
    for filter in filters {
        collect_like_leaves(filter, fts_cols, resolve, &mut out);
    }
    out
}

/// Walk one filter expression for `LIKE` nodes, lowering each to its
/// prune leaves.
fn collect_like_leaves(
    expr: &Expr,
    fts_cols: &HashSet<&str>,
    resolve: &dyn Fn(&str) -> Arc<dyn Tokenizer>,
    out: &mut Vec<PruneLeaf>,
) {
    match expr {
        Expr::Alias(a) => collect_like_leaves(&a.expr, fts_cols, resolve, out),
        Expr::BinaryExpr(be) if be.op == Operator::And => {
            collect_like_leaves(&be.left, fts_cols, resolve, out);
            collect_like_leaves(&be.right, fts_cols, resolve, out);
        }
        Expr::Like(like) => {
            // A single leaf is always one conjunctive group.
            like_leaf(like, fts_cols, resolve).append_prune_leaves(out);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use datafusion::{
        logical_expr::expr::InList,
        prelude::{col, lit},
    };

    use super::*;
    use crate::superfile::fts::tokenize::{AsciiLowerTokenizer, StandardTokenizer};

    fn fts_cols() -> HashSet<&'static str> {
        let mut s = HashSet::new();
        s.insert("title");
        s
    }

    /// Resolver for the lowering tests: every column tokenizes with the
    /// ASCII-lower analyzer.
    fn ascii_resolver(_col: &str) -> Arc<dyn Tokenizer> {
        Arc::new(AsciiLowerTokenizer)
    }

    fn plan(expr: Expr) -> CandidatePlan {
        CandidatePlan::from_filters(&[expr], &fts_cols(), &ascii_resolver)
    }

    /// Bloom-survival flattening: an `AND` of term-alls collapses to one
    /// conjunctive group; a top-level `OR` yields one group per branch; a plan
    /// that isn't a pure conjunction (`Unbounded`, or an `OR` nested under an
    /// `AND`) is inexpressible and returns `false`.
    #[test]
    fn candidate_plan_flattens_into_bloom_survival_groups() {
        let terms = |c: &str, t: &str| CandidatePlan::TermsAll {
            column: c.to_string(),
            tokens: vec![t.to_string()],
        };

        let mut groups = Vec::new();
        assert!(
            CandidatePlan::And(vec![terms("a", "x"), terms("b", "y")])
                .collect_survival_or_groups(&mut groups)
        );
        assert_eq!(groups.len(), 1, "AND is one conjunctive group");
        assert_eq!(groups[0].len(), 2, "with a leaf per term-all");

        let mut groups = Vec::new();
        assert!(
            CandidatePlan::Or(vec![terms("a", "x"), terms("b", "y")])
                .collect_survival_or_groups(&mut groups)
        );
        assert_eq!(groups.len(), 2, "top-level OR is one group per branch");

        let mut groups = Vec::new();
        assert!(!CandidatePlan::Unbounded.collect_survival_or_groups(&mut groups));

        let mut groups = Vec::new();
        assert!(
            !CandidatePlan::And(vec![
                terms("a", "x"),
                CandidatePlan::Or(vec![terms("b", "y"), terms("c", "z")]),
            ])
            .collect_survival_or_groups(&mut groups),
            "OR nested under AND is not a single bloom conjunction"
        );
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

    /// Resolver for the Unicode-aware analyzer.
    fn standard_resolver(_col: &str) -> Arc<dyn Tokenizer> {
        Arc::new(StandardTokenizer)
    }

    fn standard_plan(expr: Expr) -> CandidatePlan {
        CandidatePlan::from_filters(&[expr], &fts_cols(), &standard_resolver)
    }

    fn like_token(text: &str, open_left: bool, open_right: bool) -> LikeToken {
        LikeToken {
            text: text.into(),
            open_left,
            open_right,
        }
    }

    fn terms_like(tokens: Vec<LikeToken>) -> CandidatePlan {
        CandidatePlan::TermsLike {
            column: "title".into(),
            tokens,
        }
    }

    fn terms_all(tokens: &[&str]) -> CandidatePlan {
        CandidatePlan::TermsAll {
            column: "title".into(),
            tokens: tokens.iter().map(|t| (*t).into()).collect(),
        }
    }

    #[test]
    fn like_without_wildcards_is_the_equality_term_and() {
        // `title LIKE 'rust async'` admits only the exact value, whose
        // tokens are the literal's — the same superset equality lowers to.
        assert_eq!(
            plan(col("title").like(lit("rust async"))),
            terms_all(&["rust", "async"])
        );
    }

    #[test]
    fn like_open_edges_lower_to_dictionary_shapes_under_standard() {
        assert_eq!(
            standard_plan(col("title").like(lit("rust%"))),
            terms_like(vec![like_token("rust", false, true)])
        );
        assert_eq!(
            standard_plan(col("title").like(lit("%rust"))),
            terms_like(vec![like_token("rust", true, false)])
        );
        assert_eq!(
            standard_plan(col("title").like(lit("%rust%"))),
            terms_like(vec![like_token("rust", true, true)])
        );
        // `_` is a wildcard too: `ab` closed on the left by the pattern
        // start and open on the right; `cd` open on both sides.
        assert_eq!(
            standard_plan(col("title").like(lit("ab_cd%"))),
            terms_like(vec![
                like_token("ab", false, true),
                like_token("cd", true, true)
            ])
        );
    }

    #[test]
    fn like_separators_inside_the_fragment_close_a_token() {
        // Spaces are hard breaks: both tokens are complete, and an
        // all-complete pattern is the plain term-AND.
        assert_eq!(
            standard_plan(col("title").like(lit("% quick fox %"))),
            terms_all(&["quick", "fox"])
        );
        // The hyphen closes `fox` on the left; the wildcard leaves the
        // right open.
        assert_eq!(
            standard_plan(col("title").like(lit("%-fox%"))),
            terms_like(vec![like_token("fox", false, true)])
        );
    }

    #[test]
    fn like_word_joiners_leave_an_edge_open_under_standard() {
        // `'` can join `don` to what follows (`don't`), so the token may be
        // the head of a longer term; `.` likewise keeps `fox` open on the
        // left (`a.fox`), while the trailing space closes its right.
        assert_eq!(
            standard_plan(col("title").like(lit("%don'%"))),
            terms_like(vec![like_token("don", true, true)])
        );
        assert_eq!(
            standard_plan(col("title").like(lit("%.fox %"))),
            terms_like(vec![like_token("fox", true, false)])
        );
    }

    #[test]
    fn like_final_sigma_open_right_is_dropped_under_standard() {
        // `ΟΔΟΣ` lowercases to `οδος` on its own but to `οδοσ…` mid-word,
        // so an open-right token has two spellings and cannot be required.
        assert_eq!(
            standard_plan(col("title").like(lit("%ΟΔΟΣ%"))),
            CandidatePlan::Unbounded
        );
        // Closed on the right the word really ends there: kept as a suffix.
        assert_eq!(
            standard_plan(col("title").like(lit("%ΟΔΟΣ"))),
            terms_like(vec![like_token("οδος", true, false)])
        );
    }

    #[test]
    fn like_under_ascii_lower_keeps_only_complete_tokens() {
        // The default analyzer drops any run holding a non-ASCII byte, so
        // a token that may be the head or tail of a longer run is not
        // guaranteed indexed: open-edged tokens drop out, and a pattern
        // made only of them is Unbounded.
        assert_eq!(
            plan(col("title").like(lit("%rust%"))),
            CandidatePlan::Unbounded
        );
        assert_eq!(
            plan(col("title").like(lit("rust%"))),
            CandidatePlan::Unbounded
        );
        // A token closed by separators inside the fragment is exact.
        assert_eq!(
            plan(col("title").like(lit("%(rust)%"))),
            terms_all(&["rust"])
        );
        // Mixed: the complete token stays, the open one drops.
        assert_eq!(
            plan(col("title").like(lit("rust async%"))),
            terms_all(&["rust"])
        );
    }

    #[test]
    fn like_escape_makes_a_wildcard_literal() {
        // `\%` is a literal percent sign: no wildcard, so the value must be
        // exactly `100% sure` and its tokens bound it.
        assert_eq!(
            plan(col("title").like(lit("100\\% sure"))),
            terms_all(&["100", "sure"])
        );
        // The literal `%` is itself a separator, so `100` is complete even
        // though a real wildcard follows the fragment.
        assert_eq!(
            standard_plan(col("title").like(lit("100\\%%"))),
            terms_all(&["100"])
        );
        // A trailing backslash is a pattern the executor rejects.
        assert_eq!(
            plan(col("title").like(lit("rust\\"))),
            CandidatePlan::Unbounded
        );
    }

    #[test]
    fn negated_and_case_insensitive_like_are_unbounded() {
        assert_eq!(
            standard_plan(col("title").not_like(lit("rust%"))),
            CandidatePlan::Unbounded
        );
        assert_eq!(
            standard_plan(col("title").ilike(lit("rust%"))),
            CandidatePlan::Unbounded
        );
    }

    #[test]
    fn like_needs_an_fts_column_and_a_literal_pattern() {
        assert_eq!(
            standard_plan(col("category").like(lit("rust%"))),
            CandidatePlan::Unbounded
        );
        assert_eq!(
            standard_plan(col("title").like(col("category"))),
            CandidatePlan::Unbounded
        );
        // Wildcards only: no fragment, nothing to bound.
        assert_eq!(
            standard_plan(col("title").like(lit("%_%"))),
            CandidatePlan::Unbounded
        );
    }

    #[test]
    fn like_prunes_with_a_bloom_for_complete_tokens_and_a_range_for_a_prefix() {
        // `rust` heads a term → lex range; `quick`, `fox` are complete →
        // one bloom AND; `tail` is open on the left → no summary bounds it.
        let leaves = like_prune_leaves(
            &[col("title").like(lit("rust% quick fox %tail"))],
            &fts_cols(),
            &standard_resolver,
        );
        assert_eq!(leaves.len(), 2);
        assert!(matches!(
            &leaves[0],
            PruneLeaf::TermPresence { column, terms, mode: BoolMode::And }
                if column == "title" && *terms == ["quick".to_owned(), "fox".to_owned()]
        ));
        assert!(matches!(
            &leaves[1],
            PruneLeaf::Prefix { column, prefix } if column == "title" && prefix == b"rust"
        ));
        // Under the default analyzer only the complete tokens survive.
        let leaves = like_prune_leaves(
            &[col("title").like(lit("rust% quick fox %tail"))],
            &fts_cols(),
            &ascii_resolver,
        );
        assert_eq!(leaves.len(), 1);
        assert!(matches!(
            &leaves[0],
            PruneLeaf::TermPresence { terms, mode: BoolMode::And, .. }
                if *terms == ["quick".to_owned(), "fox".to_owned()]
        ));
    }

    #[test]
    fn a_like_with_only_open_left_tokens_has_no_survival_gate() {
        // `%fox%` bounds no manifest summary (no bloom term, no prefix), so
        // the plan must report "no gate" — an empty leaf group would read as
        // "no superfile survives" and silently drop every match.
        let contains = terms_like(vec![like_token("fox", true, true)]);
        let mut groups = Vec::new();
        assert!(!contains.collect_survival_or_groups(&mut groups));
        assert!(groups.is_empty());
        // The same leaf under an OR disables the whole disjunction's gate.
        let mut groups = Vec::new();
        assert!(
            !CandidatePlan::Or(vec![terms_all(&["alpha"]), contains.clone()])
                .collect_survival_or_groups(&mut groups)
        );
        // Under an AND, the sibling's leaf still gates (a superset stays
        // sound: every match has `alpha`).
        let mut groups = Vec::new();
        assert!(
            CandidatePlan::And(vec![terms_all(&["alpha"]), contains])
                .collect_survival_or_groups(&mut groups)
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 1);
    }

    #[test]
    fn has_like_finds_a_dictionary_leaf_anywhere_in_the_tree() {
        assert!(standard_plan(col("title").like(lit("rust%"))).has_like());
        assert!(
            standard_plan(
                col("title")
                    .eq(lit("alpha"))
                    .or(col("title").like(lit("%beta")))
            )
            .has_like()
        );
        assert!(!plan(col("title").eq(lit("alpha"))).has_like());
        assert!(!CandidatePlan::Unbounded.has_like());
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
    fn no_fts_columns_is_unbounded() {
        // With no FTS columns there is nothing to term-bound. (The old
        // "FTS columns but no tokenizer" state is unrepresentable now
        // that a per-column tokenizer always exists when FTS columns do.)
        let p = CandidatePlan::from_filters(
            &[col("title").eq(lit("rust"))],
            &HashSet::new(),
            &ascii_resolver,
        );
        assert_eq!(p, CandidatePlan::Unbounded);
    }

    #[test]
    fn lowering_uses_the_per_column_tokenizer() {
        // `title` is analyzed with the Unicode-aware standard tokenizer,
        // which keeps non-ASCII letters; ascii_lower drops the whole
        // token. The lowering must pick the column's own analyzer.
        let resolve = |col: &str| -> Arc<dyn Tokenizer> {
            if col == "title" {
                Arc::new(StandardTokenizer)
            } else {
                Arc::new(AsciiLowerTokenizer)
            }
        };
        let bounded =
            CandidatePlan::from_filters(&[col("title").eq(lit("Süd"))], &fts_cols(), &resolve);
        assert_eq!(
            bounded,
            CandidatePlan::TermsAll {
                column: "title".to_owned(),
                tokens: vec!["süd".to_owned()],
            }
        );

        // The same literal under ascii_lower drops the non-ASCII token,
        // leaving nothing to bound with ⇒ Unbounded. Proves the result
        // above came from the standard tokenizer, not a table-wide default.
        let unbounded = CandidatePlan::from_filters(
            &[col("title").eq(lit("Süd"))],
            &fts_cols(),
            &ascii_resolver,
        );
        assert_eq!(unbounded, CandidatePlan::Unbounded);
    }
}
