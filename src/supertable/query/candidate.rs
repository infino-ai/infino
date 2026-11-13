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
//! count as closed, and which open tokens can be used at all, follows the
//! `standard` word rules — see `hard_separator` and `admits`. `ILIKE` is bounded the same way for
//! ASCII tokens: Arrow folds case under Unicode simple case folding, so
//! a matching row may spell `s` as `ſ` and `k` as `K`, and the expansion
//! compares dictionary terms with those folded back (`fold_term`). Tokens
//! with non-ASCII characters are not bounded under `ILIKE`. `NOT LIKE`
//! stays `Unbounded`.
//!
//! ## The exact shape
//!
//! One `LIKE` shape skips pass 2 entirely: `col ILIKE '%word%'` on a
//! `standard` column, where `word` is one whole ASCII token, and any `AND`
//! / `OR` of such filters. The dictionary decides their rows (see
//! [`exact_contains`] and [`ExactFilter`]), the provider reports them
//! `Exact`, and the scan selects exactly those rows — every other shape
//! above stays a verified superset.
//!
//! Why one leaf is exact. Arrow matches `%f%` case-insensitively under
//! simple case folding. The one-token condition means every character of
//! `f` is a word character or a joiner the word rules keep inside a word
//! (`.` between letters, `'`, `_`, …), so an occurrence of `f` in any text
//! lies inside one word, and that word, lowercased, contains `f`
//! lowercased; conversely a word containing the lowercased `f` spells it
//! out in its row. Case folding and lowercasing agree on ASCII but for the
//! characters the dictionary walk folds back (`ſ`, the Kelvin sign) and the
//! dotted capital I, which the walk marks doubtful. A word longer than the
//! tokenizer's cut is indexed in pieces, and a match across a cut is marked
//! doubtful too. Doubtful rows are checked against their text; every other
//! row is decided by the dictionary. What the rule leaves out needs the
//! text: an anchored `f%` or `%f` (the dictionary does not know which word
//! starts or ends a value, nor the punctuation beside it), case-sensitive
//! `LIKE` (terms are lowercased), `_` and several literals (order and
//! gaps), a literal with a separator in it (a phrase), a non-ASCII literal,
//! and a column with a stopword set or stemmer.
//!
//! Why a tree of exact leaves is exact. Each leaf splits a superfile's rows
//! into proven, doubtful and false; `AND` and `OR` combine those splits row
//! by row, and a row the combination leaves doubtful is checked against the
//! whole conjunct's text. With only `AND` and `OR` above the leaves, an
//! unknown (NULL) leaf makes the conjunct true exactly when a false one
//! would, and `WHERE` keeps only the true rows; `NOT` breaks that (`NOT
//! NULL` is unknown, `NOT false` is true).

use std::{
    collections::{HashMap, HashSet},
    fmt, mem,
    sync::Arc,
};

use arrow_schema::{DataType, TimeUnit};
use datafusion::{
    logical_expr::{
        Expr, Operator,
        expr::{Between, InList, Like},
    },
    scalar::ScalarValue,
};
use futures::{FutureExt, future::BoxFuture};
use rayon::ThreadPool;
use roaring::RoaringBitmap;
use tokio::sync::Semaphore;
use tracing::debug;
use uuid::Uuid;

use crate::{
    runtime_bridge::run_on_pool_while_awaited,
    superfile::{
        ReadError, SuperfileReader,
        fts::{
            reader::{
                BoolMode, ContainsRows, FetchedTermMemo, LONG_S_ASCII, MatchWork, TermPattern,
            },
            tokenize::{MAX_TOKEN_CHARS, STANDARD_TOKENIZER, Tokenizer},
        },
        id_space::RowId,
        range::{KeyKind, float_key, signed_key, unsigned_key},
    },
    supertable::{
        error::QueryError,
        manifest::{ManifestSnapshot, SuperfileEntry, SuperfileUri},
        query::{
            fts::IndexedSuperfile,
            prune::{PruneLeaf, select_superfiles},
        },
        schema::FieldId,
    },
};

/// Deepest plan (`AND`/`OR` levels) matched on the reader pool. A match
/// recurses once per level, and a pool thread has rayon's default stack
/// (or what a caller's pool sets), far smaller than the 16 MiB query runtime
/// threads that match deeper plans. A debug build overflows a 2 MiB pool
/// thread between 150 and 200 levels; real filters sit far below 64.
const MAX_POOL_PLAN_DEPTH: usize = 64;

/// What a SQL `WHERE` pushed into a search TVF admits, resolved once per
/// query against the pinned snapshot and shared by every retriever the
/// query runs (`hybrid_search` runs two): the superfiles that may hold a
/// match, and — when the index can bound rows — the candidate rows in each.
///
/// Built by `SupertableReader::candidate_scope`. The superfile set is the
/// intersection of the manifest's scalar / null / `IN` statistics pruning
/// (the same leaves the SQL scan uses, so `path = 'x'` on a table whose
/// superfiles are stats-disjoint on `path` opens only the file that can
/// hold it) with the [`CandidatePlan`]'s term-bloom survival. The rows are
/// the plan's per-superfile evaluation, tombstones already subtracted.
// No `Default`: the derived one would pair an empty `superfiles` with
// `allow: None`, which reads as "no files, but every row of every file is
// admitted" — the opposite polarity from [`CandidateScope::empty`], and a
// silent way for a future `..Default::default()` to drop the row bound.
// Construct one through `empty()` or `SupertableReader::candidate_scope`.
#[derive(Debug, Clone)]
pub(crate) struct CandidateScope {
    /// Superfiles a match may live in, in manifest order.
    pub(crate) superfiles: Vec<Arc<SuperfileEntry>>,
    /// Candidate `local_doc_id`s per superfile — a superset of the rows
    /// satisfying the predicate, keyed by the superfile's URI; a superfile
    /// absent from the map has no candidate row. `None` when the plan is
    /// [`Unbounded`](CandidatePlan::Unbounded): the index bounds no row,
    /// every row of every superfile above is a candidate, and the caller
    /// fills its top-k by over-fetching under the exact predicate.
    pub(crate) allow: Option<HashMap<SuperfileUri, Arc<RoaringBitmap>>>,
    /// A predicate checked on each superfile when it is opened, when set:
    /// its kernel then admits only the rows of [`Self::allow`] (every row
    /// when unbounded) the gate also keeps, so a ranked search's `k` hits are
    /// the `k` best rows the predicate keeps, in one pass, and a superfile
    /// the search never opens is never checked.
    pub(crate) gate: Option<Arc<dyn RowGate>>,
}

/// The rows of an opened superfile a predicate keeps, checked when the
/// superfile is opened rather than for every superfile before a search.
pub(crate) trait RowGate: Send + Sync + fmt::Debug {
    /// Of superfile `suid`, read through `r`, the rows the search can
    /// return that the predicate keeps.
    fn rows<'a>(
        &'a self,
        r: &'a Arc<SuperfileReader>,
        suid: Uuid,
    ) -> BoxFuture<'a, Result<RoaringBitmap, QueryError>>;
}

impl CandidateScope {
    /// A scope that admits nothing: no superfile can hold a match.
    pub(crate) fn empty() -> Self {
        Self {
            superfiles: Vec::new(),
            allow: Some(HashMap::new()),
            gate: None,
        }
    }

    /// The scope's superfiles that still hold a candidate row: every one
    /// under an unbounded plan, and under a bounded one those with an entry
    /// in `allow` (a superfile whose candidate set came back empty has
    /// none). The one place the two gates - statistics survival and row
    /// survival - are read together, so a caller cannot ask about a file
    /// outside the scope and be told yes.
    pub(crate) fn admitted_superfiles(&self) -> impl Iterator<Item = &Arc<SuperfileEntry>> {
        self.superfiles.iter().filter(move |entry| {
            self.allow
                .as_ref()
                .is_none_or(|allow| allow.contains_key(&entry.uri))
        })
    }

    /// Whether this scope narrows the *rows* of `kept` at all: false under
    /// an unbounded plan, and false when every kept superfile's candidate
    /// set is its whole row set - a bitmap that admits every row is no gate,
    /// and a caller with a faster ungated kernel may use it. True as soon as
    /// one kept superfile has an absent or partial set. Tombstones are
    /// already subtracted from the sets, so a superfile with deleted rows
    /// reads as bounded; that is the conservative side.
    pub(crate) fn bounds_rows(&self, kept: &[Arc<SuperfileEntry>]) -> bool {
        let Some(allow) = self.allow.as_ref() else {
            return false;
        };
        kept.iter().any(|entry| {
            allow
                .get(&entry.uri)
                .is_none_or(|rows| rows.len() < entry.n_docs)
        })
    }
}

/// Most indexed terms one `LIKE` fragment token may widen to before the
/// index gives up on it. Each expanded term costs a df probe and a
/// posting walk, and a token this broad matches enough rows that the
/// scan wins anyway — the provider's selectivity gate would send it there
/// after paying for the probes. Sibling of the provider's `PUSHDOWN_*`
/// gates. Bounds the verified path only: the exact shape
/// ([`exact_contains`]) has no scan to fall back to and takes every term.
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

/// Each range-indexed column's id and type, by name: what
/// [`CandidatePlan::from_filters`] lowers range predicates against.
pub(crate) type RangeColumns = HashMap<String, (FieldId, DataType)>;

/// The columns a predicate can be bounded on: the full-text indexed and
/// the range indexed.
struct IndexedColumns<'a, 'b> {
    fts: &'a HashSet<&'b str>,
    range: &'a RangeColumns,
}

/// Per column, the prefetched-term memo a superfile's plan evaluation
/// serves its exact-match terms from — built from the table-level term
/// index's postings locations, so the superfile's dictionary is not read
/// for a term the memo covers. Empty when the table has no index.
pub(crate) type TermMemos = HashMap<String, Arc<FetchedTermMemo>>;

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
    /// the unexpanded leaf. `fold` marks an `ILIKE`: dictionary terms are
    /// compared with `ſ` / `K` folded back to `s` / `k`, the two ASCII
    /// letters whose case-folding class Arrow widens past lowercasing.
    TermsLike {
        column: String,
        tokens: Vec<LikeToken>,
        fold: bool,
    },
    /// Rows whose column with table id `field` holds a value keyed
    /// ([`KeyKind`]) within `lo..=hi`: a range or equality predicate on a
    /// range-indexed column,
    /// every conjunct on the column folded into one interval. Resolved per
    /// superfile from its range index; a superfile without one for the
    /// column is scanned. Bounds are inclusive and hold exactly the values
    /// the predicate does (`col > 5` reads from the key after 5's); the
    /// predicate is still verified above the scan as for every other leaf.
    /// `lo > hi` holds no row.
    Range {
        field: FieldId,
        kind: KeyKind,
        lo: u64,
        hi: u64,
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
    /// text the same way the column was tokenized at ingest. A column
    /// `resolve` cannot answer for has no index to bound against, so its
    /// predicate lowers to [`Unbounded`] — no analyzer is assumed on its
    /// behalf. Empty `fts_cols` ⇒ no FTS columns ⇒ always [`Unbounded`].
    ///
    /// `range_cols` maps each range-indexed column to its type: a range or
    /// equality on one of them lowers to a [`Range`](Self::Range) leaf.
    pub(crate) fn from_filters(
        filters: &[Expr],
        fts_cols: &HashSet<&str>,
        range_cols: &RangeColumns,
        resolve: &dyn Fn(&str) -> Option<Arc<dyn Tokenizer>>,
    ) -> CandidatePlan {
        if fts_cols.is_empty() && range_cols.is_empty() {
            return CandidatePlan::Unbounded;
        }
        let cols = IndexedColumns {
            fts: fts_cols,
            range: range_cols,
        };
        and_combine(filters.iter().map(|f| lower(f, &cols, resolve)).collect())
    }

    /// Every `(column, terms)` this plan resolves through exact term match,
    /// deduplicated per column — what a caller asks the table-level term
    /// index for, so each superfile can resolve them from its postings
    /// alone. `LIKE` tokens are left out: they expand per superfile.
    pub(crate) fn term_requests(&self) -> HashMap<String, Vec<String>> {
        fn walk(plan: &CandidatePlan, out: &mut HashMap<String, Vec<String>>) {
            match plan {
                CandidatePlan::TermsAll { column, tokens } => {
                    out.entry(column.clone())
                        .or_default()
                        .extend(tokens.iter().cloned());
                }
                CandidatePlan::TermsAny { column, terms } => {
                    out.entry(column.clone())
                        .or_default()
                        .extend(terms.iter().cloned());
                }
                CandidatePlan::And(children) | CandidatePlan::Or(children) => {
                    for c in children {
                        walk(c, out);
                    }
                }
                CandidatePlan::TermsLike { .. }
                | CandidatePlan::Range { .. }
                | CandidatePlan::Unbounded => {}
            }
        }
        let mut out = HashMap::new();
        walk(self, &mut out);
        for terms in out.values_mut() {
            terms.sort_unstable();
            terms.dedup();
        }
        out
    }

    /// Evaluate against one superfile's reader. `Ok(None)` means "no bound,
    /// scan all rows"; `Ok(Some(bitmap))` is the candidate `local_doc_id`
    /// superset (possibly empty). `TermsAll` is one `token_match(.., And)`;
    /// `And`/`Or` intersect/union children. The second element sums the
    /// posting-walk work of every `TermsAll` leaf the evaluation touched (an
    /// early-out keeps the work already done), so the caller can flush it per
    /// superfile.
    ///
    /// `pool` is where a `LIKE` leaf's dictionary walk and a range decode
    /// run. With `None`, a range decode runs inline when not in a tokio task
    /// (as on a pool thread), but a `LIKE` walk goes to rayon's global pool.
    pub(crate) fn evaluate<'a>(
        &'a self,
        reader: &'a SuperfileReader,
        pool: Option<&'a ThreadPool>,
        memos: &'a TermMemos,
    ) -> BoxFuture<'a, Result<(Option<RoaringBitmap>, MatchWork), ReadError>> {
        Box::pin(async move {
            match self {
                CandidatePlan::Unbounded => Ok((None, MatchWork::default())),
                CandidatePlan::TermsAll { column, tokens } => {
                    let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
                    let memo = memos.get(column).map(Arc::as_ref);
                    let (docs, work) = reader
                        .token_match_prefetched(column, &refs, BoolMode::And, memo)
                        .await?;
                    Ok((Some(docs.into_iter().map(RowId::get).collect()), work))
                }
                CandidatePlan::TermsAny { column, terms } => {
                    // No qualifying term in this superfile ⇒ no row, decided
                    // here rather than left to the match's empty-input rule.
                    if terms.is_empty() {
                        return Ok((Some(RoaringBitmap::new()), MatchWork::default()));
                    }
                    let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
                    let memo = memos.get(column).map(Arc::as_ref);
                    let (docs, work) = reader
                        .token_match_prefetched(column, &refs, BoolMode::Or, memo)
                        .await?;
                    Ok((Some(docs.into_iter().map(RowId::get).collect()), work))
                }
                CandidatePlan::TermsLike {
                    column,
                    tokens,
                    fold,
                } => {
                    let (expanded, mut work) =
                        expand_like(reader, column, tokens, *fold, true, pool).await?;
                    let (docs, eval_work) = expanded.evaluate(reader, pool, memos).await?;
                    work.merge(eval_work);
                    Ok((docs, work))
                }
                CandidatePlan::Range {
                    field,
                    kind,
                    lo,
                    hi,
                } => {
                    let rows = reader.range_rows(*field, *kind, *lo, *hi, pool).await?;
                    Ok((rows, MatchWork::default()))
                }
                CandidatePlan::And(children) => {
                    let mut acc: Option<RoaringBitmap> = None;
                    let mut work = MatchWork::default();
                    for c in children {
                        let (child, child_work) = c.evaluate(reader, pool, memos).await?;
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
                        let (child, child_work) = c.evaluate(reader, pool, memos).await?;
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
    /// [`Self::evaluate`] on a `pool` thread when `reader` holds its indexes
    /// in memory (`parquet_bytes` is set; a hybrid reader lacks only its
    /// vector blob, which a match never reads), so planning matches many
    /// superfiles at once.
    ///
    /// A local read never yields: it hits memory or blocks the thread on a
    /// page fault. So matches polled together on one task run one after
    /// another; on the pool each gets its own thread. A lazy reader's reads
    /// wait on the network and already overlap, so its match stays here.
    ///
    /// A plan deeper than [`MAX_POOL_PLAN_DEPTH`] stays here too: its match
    /// would overflow a pool thread's stack, which aborts the process.
    ///
    /// A match that would wait on the pool (another query filling a shared
    /// cell) gives up there, releases its slot and reruns here: wasted work,
    /// never a wrong answer. A match holds one of `slots` while on the pool,
    /// which bounds how many one query puts there.
    pub(crate) async fn evaluate_on_pool(
        self: Arc<Self>,
        reader: Arc<SuperfileReader>,
        superfile: Uuid,
        pool: &ThreadPool,
        slots: &Semaphore,
        mut memos: TermMemos,
    ) -> Result<(Option<RoaringBitmap>, MatchWork), ReadError> {
        // A `LIKE` leaf hands its dictionary walk to a pool itself, which
        // would leave that walk running on the global pool when the poll
        // below gives up. Callers expand `LIKE` first (see `Self::expand`).
        if reader.parquet_bytes().is_some()
            && !self.has_like()
            && self.depth() <= MAX_POOL_PLAN_DEPTH
        {
            let _slot = slots
                .acquire()
                .await
                .map_err(|_| ReadError::TaskDropped("planning match slot"))?;
            let plan = Arc::clone(&self);
            let local = Arc::clone(&reader);
            // `memos` rides along and comes back, so the fallback below can
            // reuse it without a copy.
            let (matched, returned) =
                run_on_pool_while_awaited(pool, "planning match", move || {
                    let matched = plan.evaluate(&local, None, &memos).now_or_never();
                    (matched, memos)
                })
                .await
                .map_err(|e| ReadError::TaskDropped(e.0))?;
            if let Some(matched) = matched {
                return matched;
            }
            // Rare and harmless once, but if resident matches fell back
            // routinely they would each do their work twice, and a bench
            // would still read flat against main.
            debug!(
                %superfile,
                "a resident match could not finish on the pool; rerunning it on the planning task"
            );
            memos = returned;
        }
        self.evaluate(&reader, Some(pool), &memos).await
    }

    /// Manifest-only superfile survival gate for this plan: the superfile
    /// ids that *could* match by term blooms and the term index. `None`
    /// means no gate: the plan, or one of its `OR` branches, bounds no
    /// superfile.
    pub(crate) async fn surviving_superfile_ids(
        &self,
        manifest: &ManifestSnapshot,
    ) -> Result<Option<HashSet<u128>>, QueryError> {
        let mut groups = Vec::new();
        if !self.collect_survival_or_groups(&mut groups) {
            return Ok(None);
        }
        // One group is a plain conjunction; several are an `OR` of them.
        let leaves: Vec<PruneLeaf> = match groups.len() {
            0 => return Ok(Some(HashSet::new())),
            1 => groups.swap_remove(0),
            _ => PruneLeaf::any_of(groups).into_iter().collect(),
        };
        Ok(Some(
            select_superfiles(manifest, &leaves)
                .await?
                .iter()
                .map(|e| e.superfile_id.as_u128())
                .collect(),
        ))
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
            CandidatePlan::TermsLike {
                column,
                tokens,
                fold,
            } => {
                // A complete token must be present as itself → term bloom;
                // a token open only on the right must head some term → lex
                // term-range overlap. A token open on the left bounds no
                // manifest summary. Under `ILIKE` a token holding `s` may
                // be spelled with `ſ` in the dictionary, so neither summary
                // can be asked about it (`k` is safe: `K` lowercases to
                // `k` before indexing).
                let summarizable = |t: &&LikeToken| !*fold || !t.text.contains(LONG_S_ASCII);
                let complete: Vec<String> = tokens
                    .iter()
                    .filter(|t| t.is_complete())
                    .filter(summarizable)
                    .map(|t| t.text.clone())
                    .collect();
                if !complete.is_empty() {
                    leaves.push(PruneLeaf::TermPresence {
                        column: column.clone(),
                        terms: complete,
                        mode: BoolMode::And,
                    });
                }
                for token in tokens
                    .iter()
                    .filter(|t| !t.open_left && t.open_right)
                    .filter(summarizable)
                {
                    leaves.push(PruneLeaf::Prefix {
                        column: column.clone(),
                        prefix: token.text.as_bytes().to_vec(),
                    });
                }
                true
            }
            // The superfiles a range cannot reach are skipped by the
            // manifest's min / max, from the predicate itself; there is no
            // term summary to add.
            CandidatePlan::Range { .. } => true,
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
    ///
    /// `located` is the term index's view of this superfile. A leaf whose
    /// words it lists takes their `df` from it and reads nothing; other
    /// leaves read the superfile as before.
    pub(crate) fn estimate<'a>(
        &'a self,
        reader: &'a SuperfileReader,
        pool: Option<&'a ThreadPool>,
        located: Option<IndexedSuperfile<'a>>,
    ) -> BoxFuture<'a, Result<(u64, MatchWork), ReadError>> {
        Box::pin(async move {
            let n_docs = reader.n_docs();
            match self {
                CandidatePlan::Unbounded => Ok((n_docs, MatchWork::default())),
                CandidatePlan::TermsAll { column, tokens } => {
                    if tokens.is_empty() {
                        return Ok((n_docs, MatchWork::default()));
                    }
                    // Intersection ≤ the rarest token's df.
                    let (min_df, work) =
                        fold_dfs(reader, located, column, tokens, u64::MAX, u64::min).await?;
                    Ok((min_df.min(n_docs), work))
                }
                CandidatePlan::TermsAny { column, terms } => {
                    if terms.is_empty() {
                        return Ok((0, MatchWork::default()));
                    }
                    // Union ≤ the sum of the terms' dfs.
                    let (sum, work) =
                        fold_dfs(reader, located, column, terms, 0, u64::saturating_add).await?;
                    Ok((sum.min(n_docs), work))
                }
                CandidatePlan::TermsLike {
                    column,
                    tokens,
                    fold,
                } => {
                    // The index was never asked about the expanded words.
                    let (expanded, mut work) =
                        expand_like(reader, column, tokens, *fold, true, pool).await?;
                    let (rows, est_work) = expanded.estimate(reader, pool, None).await?;
                    work.merge(est_work);
                    Ok((rows, work))
                }
                // The range's rows, estimated from the directory alone, the
                // blocks it cuts by the share of their key span it covers:
                // no block is read to decide whether to read blocks. A
                // superfile with no index for the column is unbounded.
                CandidatePlan::Range {
                    field,
                    kind,
                    lo,
                    hi,
                } => {
                    let rows = reader.range_estimate(*field, *kind, *lo, *hi).await?;
                    Ok((rows.map_or(n_docs, |n| n.min(n_docs)), MatchWork::default()))
                }
                CandidatePlan::And(children) => {
                    let mut m = n_docs;
                    let mut work = MatchWork::default();
                    for c in children {
                        let (child, child_work) = c.estimate(reader, pool, located).await?;
                        work.merge(child_work);
                        m = m.min(child);
                    }
                    Ok((m, work))
                }
                CandidatePlan::Or(children) => {
                    let mut sum: u64 = 0;
                    let mut work = MatchWork::default();
                    for c in children {
                        let (child, child_work) = c.estimate(reader, pool, located).await?;
                        work.merge(child_work);
                        sum = sum.saturating_add(child);
                    }
                    Ok((sum.min(n_docs), work))
                }
            }
        })
    }
}

/// Fold the `df` of each of `terms` with `fold`, starting at `init`. Taken
/// from the term index when it answers every term (no read), else from one
/// batched lookup in the superfile's dictionary.
async fn fold_dfs(
    reader: &SuperfileReader,
    located: Option<IndexedSuperfile<'_>>,
    column: &str,
    terms: &[String],
    init: u64,
    fold: fn(u64, u64) -> u64,
) -> Result<(u64, MatchWork), ReadError> {
    let indexed = located.and_then(|l| l.held(column)).and_then(|held| {
        terms
            .iter()
            .try_fold(init, |acc, term| Some(fold(acc, held.df(term)?)))
    });
    if let Some(folded) = indexed {
        return Ok((folded, MatchWork::default()));
    }
    let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
    let (dfs, work) = reader.term_dfs(column, &refs).await?;
    Ok((dfs.into_iter().fold(init, fold), work))
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
            | CandidatePlan::Range { .. }
            | CandidatePlan::Unbounded => false,
        }
    }

    /// How many `AND`/`OR` levels deep this plan nests; a leaf is 1.
    fn depth(&self) -> usize {
        match self {
            CandidatePlan::And(children) | CandidatePlan::Or(children) => {
                1 + children.iter().map(CandidatePlan::depth).max().unwrap_or(0)
            }
            CandidatePlan::TermsAll { .. }
            | CandidatePlan::TermsAny { .. }
            | CandidatePlan::TermsLike { .. }
            | CandidatePlan::Range { .. }
            | CandidatePlan::Unbounded => 1,
        }
    }

    /// Whether every bounding leaf is a [`Range`](Self::Range): the rows it
    /// selects are numbers, which cost little to decode and check.
    pub(crate) fn only_ranges(&self) -> bool {
        match self {
            CandidatePlan::Range { .. } => true,
            CandidatePlan::And(children) | CandidatePlan::Or(children) => {
                children.iter().all(CandidatePlan::only_ranges)
            }
            CandidatePlan::TermsAll { .. }
            | CandidatePlan::TermsAny { .. }
            | CandidatePlan::TermsLike { .. }
            | CandidatePlan::Unbounded => false,
        }
    }

    /// Bind every [`TermsLike`](Self::TermsLike) leaf to `reader`'s
    /// superfile: each fragment token becomes the
    /// [`TermsAny`](Self::TermsAny) of the indexed terms it covers, or
    /// `Unbounded` (dropping out of its `AND`) when more than
    /// [`LIKE_MAX_TERMS`] qualify or when the token needs the column's
    /// whole dictionary walked and `full_walk_pays(column)` says that walk
    /// is dearer than the scan it would replace. Every other node is
    /// copied. The work is the dictionary fetches performed.
    pub(crate) fn expand<'a>(
        &'a self,
        reader: &'a SuperfileReader,
        full_walk_pays: &'a (dyn Fn(&str) -> bool + Sync),
        pool: Option<&'a ThreadPool>,
    ) -> BoxFuture<'a, Result<(CandidatePlan, MatchWork), ReadError>> {
        Box::pin(async move {
            match self {
                CandidatePlan::TermsLike {
                    column,
                    tokens,
                    fold,
                } => expand_like(reader, column, tokens, *fold, full_walk_pays(column), pool).await,
                CandidatePlan::And(children) => {
                    let (expanded, work) =
                        expand_children(reader, children, full_walk_pays, pool).await?;
                    Ok((and_combine(expanded), work))
                }
                CandidatePlan::Or(children) => {
                    let (expanded, work) =
                        expand_children(reader, children, full_walk_pays, pool).await?;
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
    full_walk_pays: &(dyn Fn(&str) -> bool + Sync),
    pool: Option<&ThreadPool>,
) -> Result<(Vec<CandidatePlan>, MatchWork), ReadError> {
    let mut expanded = Vec::with_capacity(children.len());
    let mut work = MatchWork::default();
    for child in children {
        let (plan, child_work) = child.expand(reader, full_walk_pays, pool).await?;
        work.merge(child_work);
        expanded.push(plan);
    }
    Ok((expanded, work))
}

/// Bind one `TermsLike` leaf to a superfile: the `AND` of each token's
/// expansion. A token widening past [`LIKE_MAX_TERMS`], or one that needs
/// the whole dictionary walked while `allow_full_walk` is off, contributes
/// no constraint (`Unbounded`, which `and_combine` drops); every token
/// unanswered ⇒ the leaf is `Unbounded` and the superfile scans. `fold`
/// is the leaf's `ILIKE` flag; the dictionary walk runs on `pool`.
async fn expand_like(
    reader: &SuperfileReader,
    column: &str,
    tokens: &[LikeToken],
    fold: bool,
    allow_full_walk: bool,
    pool: Option<&ThreadPool>,
) -> Result<(CandidatePlan, MatchWork), ReadError> {
    // One dictionary pass widens every token of the leaf.
    let patterns: Vec<TermPattern<'_>> = tokens.iter().map(LikeToken::pattern).collect();
    let (expansions, work) = reader
        .expand_terms(
            column,
            &patterns,
            fold,
            LIKE_MAX_TERMS,
            allow_full_walk,
            pool,
        )
        .await?;
    let parts = expansions
        .into_iter()
        .map(|terms| match terms {
            Some(terms) => CandidatePlan::TermsAny {
                column: column.to_owned(),
                terms,
            },
            None => CandidatePlan::Unbounded,
        })
        .collect();
    Ok((and_combine(parts), work))
}

/// Lower one `Expr` node.
fn lower(
    expr: &Expr,
    cols: &IndexedColumns<'_, '_>,
    resolve: &dyn Fn(&str) -> Option<Arc<dyn Tokenizer>>,
) -> CandidatePlan {
    let fts_cols = cols.fts;
    match expr {
        Expr::BinaryExpr(be) => match be.op {
            Operator::And => and_combine(vec![
                lower(&be.left, cols, resolve),
                lower(&be.right, cols, resolve),
            ]),
            Operator::Or => or_combine(vec![
                lower(&be.left, cols, resolve),
                lower(&be.right, cols, resolve),
            ]),
            Operator::Eq | Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
                if range_operand(&be.left, &be.right, cols.range).is_some() =>
            {
                comparison_range(&be.left, be.op, &be.right, cols.range)
            }
            Operator::Eq => eq_leaf(&be.left, &be.right, fts_cols, resolve),
            // Inequality and arithmetic ops aren't bounded.
            _ => CandidatePlan::Unbounded,
        },
        // `BETWEEN a AND b` on a range-indexed column is the interval.
        Expr::Between(between) if !between.negated => between_range(between, cols.range),
        // `IN (a, b, …)` on an FTS column is an OR of equalities.
        Expr::InList(il) if !il.negated => in_list_leaf(il, fts_cols, resolve),
        // `LIKE` on an FTS column is bounded through the term dictionary.
        Expr::Like(like) => like_leaf(like, fts_cols, resolve),
        // NOT, IS NULL, functions, etc. — not soundly bounded.
        _ => CandidatePlan::Unbounded,
    }
}

/// `filters` read as one interval of keys on the range-indexed `column`,
/// with the column's id: every filter a comparison or `BETWEEN` on `column`
/// against literals, folded together, and the whole key space when there is
/// no filter.
/// `None` when any filter is something else — a filter on another column,
/// say, which the interval alone cannot answer. The interval holds exactly
/// the rows the filters do, but for a NULL, which no comparison holds and
/// no interval reaches.
pub(crate) fn single_range(
    filters: &[Expr],
    column: &str,
    range_cols: &RangeColumns,
) -> Option<(FieldId, KeyKind, u64, u64)> {
    let (field, data_type) = range_cols.get(column)?;
    let kind = KeyKind::of(data_type)?;
    let mut interval = (0, u64::MAX);
    for filter in filters {
        let plan = match filter {
            Expr::BinaryExpr(be) => comparison_range(&be.left, be.op, &be.right, range_cols),
            Expr::Between(between) if !between.negated => between_range(between, range_cols),
            _ => return None,
        };
        let CandidatePlan::Range {
            field: f, lo, hi, ..
        } = plan
        else {
            return None;
        };
        if f != *field {
            return None;
        }
        interval = (interval.0.max(lo), interval.1.min(hi));
    }
    Some((*field, kind, interval.0, interval.1))
}

/// The range-indexed column (its id and type) and the literal of a
/// comparison between them, in either order, with whether the column is on
/// the left.
fn range_operand<'a>(
    left: &'a Expr,
    right: &'a Expr,
    range_cols: &'a RangeColumns,
) -> Option<(FieldId, &'a DataType, &'a ScalarValue, bool)> {
    let (column, value, column_left) = match (left, right) {
        (Expr::Column(c), Expr::Literal(v, _)) => (&c.name, v, true),
        (Expr::Literal(v, _), Expr::Column(c)) => (&c.name, v, false),
        _ => return None,
    };
    let (field, data_type) = range_cols.get(column)?;
    Some((*field, data_type, value, column_left))
}

/// Lower `column op literal` (either order) on a range-indexed column to
/// the interval of keys it can hold, or `Unbounded` for a literal the
/// column's keys do not order (a null, a NaN, another type).
fn comparison_range(
    left: &Expr,
    op: Operator,
    right: &Expr,
    range_cols: &RangeColumns,
) -> CandidatePlan {
    let Some((field, data_type, value, column_left)) = range_operand(left, right, range_cols)
    else {
        return CandidatePlan::Unbounded;
    };
    let Some((kind, key)) = literal_key(value, data_type) else {
        return CandidatePlan::Unbounded;
    };
    // `literal < column` is `column > literal`.
    let op = match (column_left, op) {
        (true, op) => op,
        (false, Operator::Lt) => Operator::Gt,
        (false, Operator::LtEq) => Operator::GtEq,
        (false, Operator::Gt) => Operator::Lt,
        (false, Operator::GtEq) => Operator::LtEq,
        (false, op) => op,
    };
    // Keys order as Arrow compares the values, floats included (IEEE total
    // order), so a strict bound is the next key over and the interval is
    // the predicate's exactly. An interval past either end of the keys
    // holds nothing.
    const EMPTY: (u64, u64) = (u64::MAX, 0);
    let (lo, hi) = match op {
        Operator::Eq => (key, key),
        Operator::GtEq => (key, u64::MAX),
        Operator::LtEq => (0, key),
        Operator::Gt => key.checked_add(1).map_or(EMPTY, |next| (next, u64::MAX)),
        Operator::Lt => key.checked_sub(1).map_or(EMPTY, |prev| (0, prev)),
        _ => return CandidatePlan::Unbounded,
    };
    CandidatePlan::Range {
        field,
        kind,
        lo,
        hi,
    }
}

/// Lower `column BETWEEN low AND high` on a range-indexed column.
fn between_range(between: &Between, range_cols: &RangeColumns) -> CandidatePlan {
    let (Expr::Column(c), Expr::Literal(low, _), Expr::Literal(high, _)) = (
        between.expr.as_ref(),
        between.low.as_ref(),
        between.high.as_ref(),
    ) else {
        return CandidatePlan::Unbounded;
    };
    let Some((field, data_type)) = range_cols.get(&c.name) else {
        return CandidatePlan::Unbounded;
    };
    match (literal_key(low, data_type), literal_key(high, data_type)) {
        (Some((kind, lo)), Some((_, hi))) => CandidatePlan::Range {
            field: *field,
            kind,
            lo,
            hi,
        },
        _ => CandidatePlan::Unbounded,
    }
}

/// The key a column of `data_type` stores `value` under. `None` for a
/// null, a NaN, or a literal the column's keys do not share — a timestamp
/// of another unit, a negative value for an unsigned column, an integer a
/// float cannot hold exactly.
fn literal_key(value: &ScalarValue, data_type: &DataType) -> Option<(KeyKind, u64)> {
    let kind = KeyKind::of(data_type)?;
    let key = |k| Some((kind, k));
    match kind {
        KeyKind::Signed => {
            let v = match (value, data_type) {
                (ScalarValue::Date32(Some(v)), DataType::Date32) => i64::from(*v),
                (ScalarValue::Date64(Some(v)), DataType::Date64) => *v,
                (
                    ScalarValue::TimestampSecond(Some(v), _),
                    DataType::Timestamp(TimeUnit::Second, _),
                )
                | (
                    ScalarValue::TimestampMillisecond(Some(v), _),
                    DataType::Timestamp(TimeUnit::Millisecond, _),
                )
                | (
                    ScalarValue::TimestampMicrosecond(Some(v), _),
                    DataType::Timestamp(TimeUnit::Microsecond, _),
                )
                | (
                    ScalarValue::TimestampNanosecond(Some(v), _),
                    DataType::Timestamp(TimeUnit::Nanosecond, _),
                ) => *v,
                (_, DataType::Date32 | DataType::Date64 | DataType::Timestamp(_, _)) => {
                    return None;
                }
                _ => integer_literal(value)?.try_into().ok()?,
            };
            key(signed_key(v))
        }
        KeyKind::Unsigned => key(unsigned_key(integer_literal(value)?.try_into().ok()?)),
        KeyKind::Float => {
            let v = match value {
                ScalarValue::Float64(Some(v)) => *v,
                ScalarValue::Float32(Some(v)) => f64::from(*v),
                ScalarValue::Float16(Some(v)) => f64::from(*v),
                _ => {
                    let integer = integer_literal(value)?;
                    let v = integer as f64;
                    (v as i128 == integer).then_some(v)?
                }
            };
            match v.is_nan() {
                true => None,
                false => key(float_key(v)),
            }
        }
    }
}

/// An integer literal's value, whatever its width and sign.
fn integer_literal(value: &ScalarValue) -> Option<i128> {
    match value {
        ScalarValue::Int8(Some(v)) => Some(i128::from(*v)),
        ScalarValue::Int16(Some(v)) => Some(i128::from(*v)),
        ScalarValue::Int32(Some(v)) => Some(i128::from(*v)),
        ScalarValue::Int64(Some(v)) => Some(i128::from(*v)),
        ScalarValue::UInt8(Some(v)) => Some(i128::from(*v)),
        ScalarValue::UInt16(Some(v)) => Some(i128::from(*v)),
        ScalarValue::UInt32(Some(v)) => Some(i128::from(*v)),
        ScalarValue::UInt64(Some(v)) => Some(i128::from(*v)),
        _ => None,
    }
}

/// Lower `col = 'literal'` (either operand order) on an FTS column.
fn eq_leaf(
    left: &Expr,
    right: &Expr,
    fts_cols: &HashSet<&str>,
    resolve: &dyn Fn(&str) -> Option<Arc<dyn Tokenizer>>,
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
    resolve: &dyn Fn(&str) -> Option<Arc<dyn Tokenizer>>,
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

/// Lower `col LIKE 'pattern'` / `col ILIKE 'pattern'` on an FTS column.
/// The pattern's literal fragments are tokenized with the column's
/// analyzer; every token the analyzer can bound soundly becomes a
/// constraint (see `admits`). All-complete tokens are the same
/// term-AND an equality lowers to, unless an `ILIKE` token holds an `s`
/// (its `ſ` spelling needs the dictionary); otherwise the leaf waits for
/// a superfile's dictionary. `Unbounded` for `NOT LIKE`, a non-column or
/// non-literal operand, a non-FTS column, an escape other than `\`, or a
/// pattern with no usable token.
fn like_leaf(
    like: &Like,
    fts_cols: &HashSet<&str>,
    resolve: &dyn Fn(&str) -> Option<Arc<dyn Tokenizer>>,
) -> CandidatePlan {
    let Some(parts) = like_parts(like, fts_cols, resolve) else {
        return CandidatePlan::Unbounded;
    };
    let LikeParts {
        column,
        tok,
        pattern,
        fold,
    } = parts;
    let tokens: Vec<LikeToken> = pattern
        .fragments
        .iter()
        .flat_map(|fragment| fragment.tokens(tok.as_ref(), fold))
        .collect();
    if tokens.is_empty() {
        return CandidatePlan::Unbounded;
    }
    // Complete tokens are exact terms — unless `ILIKE` may have spelled
    // an `s` as `ſ`, which only the dictionary walk can find.
    let exact_terms = tokens.iter().all(LikeToken::is_complete)
        && !(fold && tokens.iter().any(|t| t.text.contains(LONG_S_ASCII)));
    if exact_terms {
        return CandidatePlan::TermsAll {
            column: column.to_owned(),
            tokens: tokens.into_iter().map(|t| t.text).collect(),
        };
    }
    CandidatePlan::TermsLike {
        column: column.to_owned(),
        tokens,
        fold,
    }
}

/// A `LIKE` / `ILIKE` leaf the index can reason about, parsed once for
/// both lowerings: the bounding one ([`like_leaf`]) and the exact one
/// ([`exact_contains`]).
struct LikeParts<'a> {
    /// The FTS column the pattern is matched against.
    column: &'a str,
    /// The analyzer the table indexes that column with.
    tok: Arc<dyn Tokenizer>,
    pattern: LikePattern,
    /// `ILIKE`: Arrow compares under Unicode simple case folding (ASCII
    /// fast paths aside), so the analyzers' lowercasing does not reproduce
    /// every match — this carries that to the token rules and the
    /// expansion.
    fold: bool,
}

/// Parse `like` for the index, or `None` for a leaf it cannot bound:
/// `NOT LIKE` (no term set bounds an exclusion), a non-column or
/// non-literal operand, a non-FTS column, an escape other than `\`, a
/// column whose analysis is not plain `standard` (see
/// [`is_plain_standard`]), or a pattern the executor rejects.
fn like_parts<'a>(
    like: &'a Like,
    fts_cols: &HashSet<&str>,
    resolve: &dyn Fn(&str) -> Option<Arc<dyn Tokenizer>>,
) -> Option<LikeParts<'a>> {
    // Arrow's kernel reads `\` as the escape; the executor rejects others.
    if like.negated || like.escape_char.is_some_and(|c| c != LIKE_ESCAPE) {
        return None;
    }
    let (Expr::Column(c), Expr::Literal(v, _)) = (like.expr.as_ref(), like.pattern.as_ref()) else {
        return None;
    };
    if !fts_cols.contains(c.name.as_str()) {
        return None;
    }
    let pattern = like_fragments(scalar_str(v)?)?;
    let tok = resolve(&c.name)?;
    if !is_plain_standard(tok.as_ref()) {
        return None;
    }
    Some(LikeParts {
        column: &c.name,
        tok,
        pattern,
        fold: like.case_insensitive,
    })
}

/// A `WHERE` conjunct the term dictionary answers exactly: `column ILIKE
/// '%needle%'`, with the rows it matches read from the postings rather
/// than the text (see [`exact_contains`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExactContains {
    /// The FTS column.
    pub(crate) column: String,
    /// The pattern's literal, lowercased: one whole `standard` token.
    pub(crate) needle: String,
}

/// `filter` as an [`ExactContains`], or `None` when it stays a bounded,
/// verified filter.
///
/// Exact only when all of these hold:
///  - `ILIKE`, not `NOT ILIKE` or case-sensitive `LIKE`;
///  - a bare FTS column indexed by `standard`;
///  - the pattern is `%f%` with one literal `f` and no `_`;
///  - `f` is ASCII and tokenizes to exactly one token, `f` lowercased.
///
/// Rows the dictionary can't decide (a dotted capital I, a match across a
/// 255-char cut) come back doubtful and are checked against their text.
/// Why the rule is exact is in the module's "The exact shape" section.
pub(crate) fn exact_contains(
    filter: &Expr,
    fts_cols: &HashSet<&str>,
    resolve: &dyn Fn(&str) -> Option<Arc<dyn Tokenizer>>,
) -> Option<ExactContains> {
    let Expr::Like(like) = filter else {
        return None;
    };
    let parts = like_parts(like, fts_cols, resolve)?;
    if !parts.fold || parts.pattern.any_one {
        return None;
    }
    let [fragment] = parts.pattern.fragments.as_slice() else {
        return None;
    };
    if fragment.at_start || fragment.at_end || !fragment.text.is_ascii() {
        return None;
    }
    let needle = fragment.text.to_ascii_lowercase();
    let mut tokens = parts.tok.tokenize(&fragment.text);
    let one_token = tokens.next().is_some_and(|token| token == needle) && tokens.next().is_none();
    one_token.then(|| ExactContains {
        column: parts.column.to_owned(),
        needle,
    })
}

/// An `AND` / `OR` tree of [`ExactContains`] leaves, e.g. `title ILIKE
/// '%japan%' OR title ILIKE '%japanese%'`.
///
/// Leaves combine row by row ([`Self::rows`]). A NULL holds no term, so
/// every leaf reads it as false, which matches SQL only while there is no
/// `NOT` above it, so `NOT` is never exact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExactFilter {
    Contains(ExactContains),
    And(Vec<ExactFilter>),
    Or(Vec<ExactFilter>),
}

impl ExactFilter {
    /// Every leaf of the tree, in order, repeats included.
    pub(crate) fn leaves(&self) -> Vec<&ExactContains> {
        fn walk<'a>(filter: &'a ExactFilter, out: &mut Vec<&'a ExactContains>) {
            match filter {
                ExactFilter::Contains(leaf) => out.push(leaf),
                ExactFilter::And(children) | ExactFilter::Or(children) => {
                    for child in children {
                        walk(child, out);
                    }
                }
            }
        }
        let mut out = Vec::new();
        walk(self, &mut out);
        out
    }

    /// The tree's rows in one superfile, from each leaf's
    /// (`leaf_rows`). A row is proven when the tree holds with every
    /// doubtful leaf read as false, and admitted when it holds with every
    /// doubtful leaf read as true; the admitted rows that are not proven
    /// are the doubtful ones. Under `AND` that is the intersection of the
    /// children's proven rows, and of their admitted rows; under `OR` the
    /// union of each.
    pub(crate) fn rows(&self, leaf_rows: &dyn Fn(&ExactContains) -> ContainsRows) -> ContainsRows {
        match self {
            ExactFilter::Contains(leaf) => leaf_rows(leaf),
            ExactFilter::And(children) => combine_rows(children, leaf_rows, true),
            ExactFilter::Or(children) => combine_rows(children, leaf_rows, false),
        }
    }
}

/// [`ExactFilter::rows`] of an `AND` (`and`) or an `OR` of `children`.
fn combine_rows(
    children: &[ExactFilter],
    leaf_rows: &dyn Fn(&ExactContains) -> ContainsRows,
    and: bool,
) -> ContainsRows {
    let mut proven: Option<RoaringBitmap> = None;
    let mut admitted: Option<RoaringBitmap> = None;
    for child in children {
        let rows = child.rows(leaf_rows);
        let child_admitted = &rows.proven | &rows.doubtful;
        let merge = |acc: Option<RoaringBitmap>, next: RoaringBitmap| match acc {
            Some(acc) if and => acc & next,
            Some(acc) => acc | next,
            None => next,
        };
        proven = Some(merge(proven, rows.proven));
        admitted = Some(merge(admitted, child_admitted));
    }
    let proven = proven.unwrap_or_default();
    let mut doubtful = admitted.unwrap_or_default();
    doubtful -= &proven;
    ContainsRows { proven, doubtful }
}

/// `filter` as an [`ExactFilter`], or `None` when some part of it is not
/// exact — then the whole conjunct stays a bounded, verified filter.
pub(crate) fn exact_filter(
    filter: &Expr,
    fts_cols: &HashSet<&str>,
    resolve: &dyn Fn(&str) -> Option<Arc<dyn Tokenizer>>,
) -> Option<ExactFilter> {
    match filter {
        Expr::BinaryExpr(be) if matches!(be.op, Operator::And | Operator::Or) => {
            let children = [&be.left, &be.right]
                .into_iter()
                .map(|side| exact_filter(side, fts_cols, resolve))
                .collect::<Option<Vec<ExactFilter>>>()?;
            // Flatten a chain of the same operator into one node, as the
            // SQL planner nests `a OR b OR c` two at a time.
            let and = be.op == Operator::And;
            let mut flat = Vec::with_capacity(children.len());
            for child in children {
                match child {
                    ExactFilter::And(inner) if and => flat.extend(inner),
                    ExactFilter::Or(inner) if !and => flat.extend(inner),
                    other => flat.push(other),
                }
            }
            Some(match and {
                true => ExactFilter::And(flat),
                false => ExactFilter::Or(flat),
            })
        }
        _ => exact_contains(filter, fts_cols, resolve).map(ExactFilter::Contains),
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
    /// dropped — a dropped constraint keeps a superset. `fold` is the
    /// `ILIKE` flag.
    ///
    /// A fragment word longer than [`MAX_TOKEN_CHARS`] comes back cut into
    /// pieces, and the edges between two pieces are not word boundaries: a
    /// matching row's copy of the word may start anywhere relative to the
    /// fragment, so its own cuts fall elsewhere. A token of exactly the cut
    /// length, and the token after it, are therefore dropped (a genuine
    /// word of that length goes too, which only loosens the bound).
    fn tokens(&self, tok: &dyn Tokenizer, fold: bool) -> Vec<LikeToken> {
        let texts: Vec<String> = tok.tokenize(&self.text).collect();
        let n = texts.len();
        let cut_piece: Vec<bool> = texts
            .iter()
            .map(|t| t.chars().count() == MAX_TOKEN_CHARS)
            .collect();
        // A cut piece, or the piece its cut edge runs into.
        let beside_cut = |i: usize| cut_piece[i] || (i > 0 && cut_piece[i - 1]);
        let left_closed = self.at_start || self.text.chars().next().is_some_and(hard_separator);
        let right_closed = self.at_end || self.text.chars().next_back().is_some_and(hard_separator);
        texts
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| !beside_cut(i))
            .filter_map(|(i, text)| {
                admits(
                    LikeToken {
                        text,
                        open_left: i == 0 && !left_closed,
                        open_right: i + 1 == n && !right_closed,
                    },
                    fold,
                )
            })
            .collect()
    }
}

/// A `LIKE` pattern split at its wildcards.
#[derive(Debug, PartialEq, Eq)]
struct LikePattern {
    /// The literal runs between the wildcards, in order.
    fragments: Vec<Fragment>,
    /// The pattern holds an unescaped `_`: a gap of exactly one character
    /// somewhere, which a fragment list alone does not show.
    any_one: bool,
}

/// Split a `LIKE` pattern at its wildcards into literal fragments,
/// unescaping `\x` to a literal `x` for any `x`, as Arrow's kernel reads
/// it. `None` for a trailing `\`, which the executor rejects — nothing to
/// plan.
fn like_fragments(pattern: &str) -> Option<LikePattern> {
    let mut fragments = Vec::new();
    let mut text = String::new();
    let mut any_one = false;
    // No wildcard seen yet: the next fragment starts where the pattern does.
    let mut at_start = true;
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            LIKE_ESCAPE => text.push(chars.next()?),
            LIKE_ANY | LIKE_ONE => {
                any_one |= c == LIKE_ONE;
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
    Some(LikePattern { fragments, any_one })
}

/// Whether a column's analysis is plain `standard`, the only one whose
/// token rules this lowering reasons about; any other keeps `LIKE`
/// `Unbounded`.
///
/// A column carrying an **analysis chain** is not plain, and must not be.
/// The lowering bounds a `LIKE` fragment by the terms it tokenizes to,
/// which is only sound while a term in the index is a
/// substring-preserving image of the text: with a stemmer it is not.
/// `LIKE '%runni%'` matches the text `running`, whose indexed term is
/// `run` — so a prefix walk for `runni` finds nothing and would drop a
/// superfile that really matches. A chain reports a composite name, so
/// the name comparison excludes it.
fn is_plain_standard(tok: &dyn Tokenizer) -> bool {
    tok.name() == STANDARD_TOKENIZER
}

/// A character that ends a token wherever it appears, so a token next
/// to it inside a fragment has the same boundary in any row's text.
///
/// UAX #29 may join a word across a `WORD_JOINERS` character (`don't`);
/// every other ASCII non-alphanumeric always breaks. Non-ASCII
/// punctuation is left open rather than classified.
fn hard_separator(c: char) -> bool {
    c.is_ascii() && !c.is_ascii_alphanumeric() && !WORD_JOINERS.contains(&c)
}

/// Whether the index can soundly require `token` of a matching row;
/// `fold` is the `ILIKE` flag.
fn admits(token: LikeToken, fold: bool) -> Option<LikeToken> {
    // Under `ILIKE`, Arrow folds case with Unicode simple case folding
    // (through a regex whenever the column or the pattern is not pure
    // ASCII). A non-ASCII token then matches spellings `to_lowercase`
    // never produces (`ς` for a medial `σ`, `ϐ` for `β`), so only an
    // ASCII token can be required.
    if fold && !token.text.is_ascii() {
        return None;
    }
    // `to_lowercase` spells a word-final `Σ` as `ς` and a medial one as
    // `σ`, so a token whose end may sit mid-word has two possible
    // spellings in the dictionary.
    if token.open_right && token.text.ends_with(FINAL_SIGMA) {
        return None;
    }
    Some(token)
}

/// Build a `TermsAll` leaf for `column = value`, or `Unbounded` if the
/// column isn't FTS-indexed, the value isn't a string, or it tokenizes
/// to nothing (e.g. the empty string — no tokens to bound with).
fn terms_all(
    column: &str,
    value: &ScalarValue,
    fts_cols: &HashSet<&str>,
    resolve: &dyn Fn(&str) -> Option<Arc<dyn Tokenizer>>,
) -> CandidatePlan {
    if !fts_cols.contains(column) {
        return CandidatePlan::Unbounded;
    }
    let Some(s) = scalar_str(value) else {
        return CandidatePlan::Unbounded;
    };
    let Some(tok) = resolve(column) else {
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
/// [`Range`](CandidatePlan::Range) children on one column fold into one
/// interval, their intersection: `price >= 10 AND price < 20` arrives as
/// two conjuncts, and each alone reaches most of the column, where the
/// pair reaches only the rows between.
fn and_combine(children: Vec<CandidatePlan>) -> CandidatePlan {
    let mut flat: Vec<CandidatePlan> = Vec::with_capacity(children.len());
    for c in children {
        match c {
            CandidatePlan::Unbounded => {}
            CandidatePlan::And(inner) => inner.into_iter().for_each(|c| fold_into(&mut flat, c)),
            other => fold_into(&mut flat, other),
        }
    }
    collapse(flat, true)
}

/// Push `child` onto the conjunction `flat`, intersecting it with a
/// [`Range`](CandidatePlan::Range) already there on the same column.
fn fold_into(flat: &mut Vec<CandidatePlan>, child: CandidatePlan) {
    if let CandidatePlan::Range {
        field,
        kind,
        lo,
        hi,
    } = &child && let Some(CandidatePlan::Range {
        lo: lo_acc,
        hi: hi_acc,
        ..
    }) = flat.iter_mut().find(
        |p| matches!(p, CandidatePlan::Range { field: f, kind: k, .. } if f == field && k == kind),
    ) {
        *lo_acc = (*lo_acc).max(*lo);
        *hi_acc = (*hi_acc).min(*hi);
        return;
    }
    flat.push(child);
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
    resolve: &dyn Fn(&str) -> Option<Arc<dyn Tokenizer>>,
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
    resolve: &dyn Fn(&str) -> Option<Arc<dyn Tokenizer>>,
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
    use std::{sync::Mutex, time::Duration};

    use arrow_array::{Int32Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use async_trait::async_trait;
    use bytes::Bytes;
    use datafusion::{
        logical_expr::expr::InList,
        prelude::{cast, col, lit},
    };
    use rayon::ThreadPoolBuilder;
    use tokio::{task::yield_now, time::timeout};
    use uuid::Uuid;

    use super::*;
    use crate::{
        runtime_bridge::held_pool,
        superfile::{
            BytesLazyByteSource, LazyByteSource, LazyByteSourceError, OpenOptions,
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
            fts::{
                analysis::{Stemmer, Stopwords, chain_tokenizer},
                tokenize::{STANDARD_TOKENIZER, StandardTokenizer},
            },
            vector::layout::VectorLayout,
        },
        test_helpers::decimal128_ids,
    };

    /// How long a test waits for a match that must finish.
    const MUST_FINISH: Duration = Duration::from_secs(10);

    fn fts_cols() -> HashSet<&'static str> {
        let mut s = HashSet::new();
        s.insert("title");
        s
    }

    /// A manifest entry of `n_docs` rows with nothing else filled in: the
    /// scope reads only `uri` and `n_docs` of it.
    fn entry(n_docs: u64) -> Arc<SuperfileEntry> {
        Arc::new(SuperfileEntry {
            physical_schema: None,
            superfile_id: Uuid::new_v4(),
            uri: SuperfileUri::new_v4(),
            stem: None,
            n_docs,
            id_min: 0,
            id_max: n_docs as i128 - 1,
            scalar_stats: Default::default(),
            fts_summary: Default::default(),
            vector_summary: Default::default(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
            vector_layout: VectorLayout::Ivf,
            birth_version: 0,
        })
    }

    /// The candidate rows `0..n` of a superfile.
    fn rows(n: u32) -> Arc<RoaringBitmap> {
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert_range(0..n);
        Arc::new(bitmap)
    }

    /// The admitted superfiles are the scope's own, less those a bounded
    /// plan left without a candidate row; an unbounded plan admits every
    /// one of the scope's superfiles, and an empty scope none.
    #[test]
    fn admitted_superfiles_reads_the_file_gate_and_the_row_gate_together() {
        let (a, b, c) = (entry(4), entry(4), entry(4));
        let uris = |admitted: Vec<&Arc<SuperfileEntry>>| -> Vec<SuperfileUri> {
            admitted.into_iter().map(|e| e.uri).collect()
        };
        let bounded = CandidateScope {
            superfiles: vec![Arc::clone(&a), Arc::clone(&b), Arc::clone(&c)],
            // `b` evaluated to no candidate row and was dropped from the map.
            allow: Some(HashMap::from([(a.uri, rows(2)), (c.uri, rows(4))])),
            gate: None,
        };
        assert_eq!(
            uris(bounded.admitted_superfiles().collect()),
            vec![a.uri, c.uri],
            "in scope order, without the file whose set is empty"
        );
        let unbounded = CandidateScope {
            superfiles: vec![Arc::clone(&a), Arc::clone(&b)],
            allow: None,
            gate: None,
        };
        assert_eq!(
            uris(unbounded.admitted_superfiles().collect()),
            vec![a.uri, b.uri]
        );
        assert_eq!(CandidateScope::empty().admitted_superfiles().count(), 0);
    }

    /// A scope narrows the rows of the kept superfiles only when some kept
    /// set is absent or smaller than the file: whole sets gate nothing, and
    /// an unbounded plan has no sets at all.
    #[test]
    fn bounds_rows_only_when_a_kept_set_is_less_than_the_whole_file() {
        let (whole, partial, missing) = (entry(4), entry(4), entry(4));
        let scope = CandidateScope {
            superfiles: vec![
                Arc::clone(&whole),
                Arc::clone(&partial),
                Arc::clone(&missing),
            ],
            allow: Some(HashMap::from([
                (whole.uri, rows(4)),
                (partial.uri, rows(3)),
            ])),
            gate: None,
        };
        assert!(
            !scope.bounds_rows(&[Arc::clone(&whole)]),
            "a set holding every row of every kept file is no gate"
        );
        assert!(scope.bounds_rows(&[Arc::clone(&whole), Arc::clone(&partial)]));
        assert!(scope.bounds_rows(&[Arc::clone(&missing)]));
        assert!(!scope.bounds_rows(&[]), "nothing kept, nothing bounded");
        let unbounded = CandidateScope {
            superfiles: vec![Arc::clone(&partial)],
            allow: None,
            gate: None,
        };
        assert!(!unbounded.bounds_rows(&[partial]));
    }

    /// Resolver whose column carries an analysis chain — the case the
    /// `LIKE` lowering must refuse to bound.
    fn stemming_resolver(_col: &str) -> Option<Arc<dyn Tokenizer>> {
        Some(chain_tokenizer(Stopwords::None, Stemmer::English))
    }

    /// Resolver for a stopworded column.
    fn stopping_resolver(_col: &str) -> Option<Arc<dyn Tokenizer>> {
        Some(chain_tokenizer(Stopwords::English, Stemmer::None))
    }

    /// The guard is that a chain's *derived name* is not plain
    /// `standard`. Asserted directly, because every `LIKE` test below
    /// would also pass if the resolver simply returned no tokenizer at
    /// all — a false pass that would hide the guard disappearing.
    #[test]
    fn a_chains_name_is_not_plain_standard() {
        let plain = chain_tokenizer(Stopwords::None, Stemmer::None);
        assert_eq!(plain.name(), STANDARD_TOKENIZER);
        assert!(
            is_plain_standard(plain.as_ref()),
            "a plain column must still be bounded"
        );
        for chained in [
            chain_tokenizer(Stopwords::English, Stemmer::None),
            chain_tokenizer(Stopwords::None, Stemmer::English),
            chain_tokenizer(Stopwords::English, Stemmer::English),
        ] {
            assert!(
                !is_plain_standard(chained.as_ref()),
                "{:?} must not be recognized as a bare analyzer",
                chained.name()
            );
        }
    }

    /// A stemmed column gets **no** `LIKE` constraint, in any fragment
    /// shape. The lowering bounds a fragment by the terms it tokenizes
    /// to, and stemming breaks the substring relationship that makes
    /// that sound: `%runni%` matches the text `running`, which is
    /// indexed as `run`, so a term or prefix bound built from `runni`
    /// would drop a superfile that genuinely matches. Dropping the
    /// constraint keeps a superset, which is always allowed; keeping a
    /// wrong one is not.
    #[test]
    fn like_on_a_chained_column_is_unbounded() {
        for pattern in ["running", "%runni%", "runn%", "%running", "run_ing"] {
            let expr = col("title").like(lit(pattern));
            assert_eq!(
                CandidatePlan::from_filters(
                    &[expr],
                    &fts_cols(),
                    &RangeColumns::new(),
                    &stemming_resolver
                ),
                CandidatePlan::Unbounded,
                "LIKE {pattern:?} on a stemmed column must not be bounded"
            );
        }
    }

    /// Equality still lowers on a chained column, and soundly: the
    /// literal runs through the *same* chain the postings did, so the
    /// terms it yields are terms the index really holds, and requiring
    /// them keeps a superset of the rows that actually compare equal.
    /// A literal the chain reduces to nothing bounds nothing.
    #[test]
    fn equality_on_a_chained_column_lowers_to_the_chain_s_terms() {
        let plan = CandidatePlan::from_filters(
            &[col("title").eq(lit("running studies"))],
            &fts_cols(),
            &RangeColumns::new(),
            &stemming_resolver,
        );
        assert_eq!(
            plan,
            CandidatePlan::TermsAll {
                column: "title".to_string(),
                tokens: vec!["run".to_string(), "studi".to_string()],
            }
        );
        // Every token removed by the chain's stopword set: there is no
        // term left to require, so the predicate bounds nothing rather
        // than bounding it with an empty conjunction.
        assert_eq!(
            CandidatePlan::from_filters(
                &[col("title").eq(lit("of the"))],
                &fts_cols(),
                &RangeColumns::new(),
                &stopping_resolver
            ),
            CandidatePlan::Unbounded
        );
    }

    fn plan(expr: Expr) -> CandidatePlan {
        CandidatePlan::from_filters(
            &[expr],
            &fts_cols(),
            &RangeColumns::new(),
            &standard_resolver,
        )
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

    /// The range-indexed columns the range lowering tests use.
    fn range_cols() -> RangeColumns {
        RangeColumns::from([
            ("price".to_string(), (FieldId(1), DataType::Float64)),
            ("qty".to_string(), (FieldId(2), DataType::Int32)),
            (
                "ts".to_string(),
                (FieldId(3), DataType::Timestamp(TimeUnit::Millisecond, None)),
            ),
        ])
    }

    fn range_plan(filters: &[Expr]) -> CandidatePlan {
        CandidatePlan::from_filters(filters, &HashSet::new(), &range_cols(), &standard_resolver)
    }

    fn range(column: &str, kind: KeyKind, lo: u64, hi: u64) -> CandidatePlan {
        CandidatePlan::Range {
            field: range_cols()[column].0,
            kind,
            lo,
            hi,
        }
    }

    #[test]
    fn a_comparison_on_a_range_column_lowers_to_its_interval_either_way_round() {
        let signed = |v: i64| signed_key(v);
        assert_eq!(
            range_plan(&[col("qty").gt(lit(5i32))]),
            range("qty", KeyKind::Signed, signed(6), u64::MAX),
            "a strict integer bound is the next key"
        );
        assert_eq!(
            range_plan(&[lit(5i32).gt(col("qty"))]),
            range("qty", KeyKind::Signed, 0, signed(4)),
            "`5 > qty` is `qty < 5`"
        );
        assert_eq!(
            range_plan(&[col("qty").gt_eq(lit(5i32))]),
            range("qty", KeyKind::Signed, signed(5), u64::MAX)
        );
        assert_eq!(
            range_plan(&[col("price").gt(lit(2.5f64))]),
            range("price", KeyKind::Float, float_key(2.5) + 1, u64::MAX),
            "a strict float bound is the next key in the total order"
        );
        assert_eq!(
            range_plan(&[col("qty").eq(lit(7i64))]),
            range("qty", KeyKind::Signed, signed(7), signed(7)),
            "an integer literal of another width"
        );
        assert_eq!(
            range_plan(&[col("price").lt_eq(lit(2.5f64))]),
            range("price", KeyKind::Float, 0, float_key(2.5))
        );
    }

    #[test]
    fn conjuncts_on_one_column_fold_into_one_interval() {
        // DataFusion hands `BETWEEN` and `a <= x AND x < b` over as separate
        // filters; each alone would reach most of the column.
        let filters = [col("qty").gt_eq(lit(-3i32)), col("qty").lt(lit(9i32))];
        assert_eq!(
            range_plan(&filters),
            range("qty", KeyKind::Signed, signed_key(-3), signed_key(8))
        );
        assert_eq!(
            range_plan(&[col("qty").between(lit(-3i32), lit(9i32))]),
            range("qty", KeyKind::Signed, signed_key(-3), signed_key(9))
        );
        // Disjoint conjuncts hold no row.
        let empty = range_plan(&[col("qty").gt(lit(9i32)), col("qty").lt(lit(-3i32))]);
        let CandidatePlan::Range { lo, hi, .. } = empty else {
            panic!("expected one range, got {empty:?}");
        };
        assert!(lo > hi);
        // Two columns stay two leaves.
        assert!(matches!(
            range_plan(&[col("qty").gt(lit(1i32)), col("price").gt(lit(1.0f64))]),
            CandidatePlan::And(children) if children.len() == 2
        ));
    }

    #[test]
    fn a_float_zero_is_its_own_key_and_a_nan_or_an_inexact_integer_bounds_nothing() {
        // Arrow orders `-0.0` below `0.0`, so each is one key.
        assert_eq!(
            range_plan(&[col("price").eq(lit(0.0f64))]),
            range("price", KeyKind::Float, float_key(0.0), float_key(0.0))
        );
        assert_eq!(
            range_plan(&[col("price").gt(lit(f64::NAN))]),
            CandidatePlan::Unbounded
        );
        assert_eq!(
            range_plan(&[col("price").gt(lit(3i64))]),
            range("price", KeyKind::Float, float_key(3.0) + 1, u64::MAX)
        );
        assert_eq!(
            range_plan(&[col("price").gt(lit(i64::MAX))]),
            CandidatePlan::Unbounded,
            "an integer no float holds exactly"
        );
    }

    #[test]
    fn a_literal_the_keys_do_not_share_bounds_nothing() {
        // A timestamp of another unit, a string, a null, a cast column.
        let seconds = Expr::Literal(ScalarValue::TimestampSecond(Some(1), None), None);
        assert_eq!(
            range_plan(&[col("ts").gt(seconds)]),
            CandidatePlan::Unbounded
        );
        let millis = Expr::Literal(ScalarValue::TimestampMillisecond(Some(1), None), None);
        assert_eq!(
            range_plan(&[col("ts").gt_eq(millis)]),
            range("ts", KeyKind::Signed, signed_key(1), u64::MAX)
        );
        assert_eq!(
            range_plan(&[col("qty").gt(lit("5"))]),
            CandidatePlan::Unbounded
        );
        assert_eq!(
            range_plan(&[col("qty").gt(Expr::Literal(ScalarValue::Int32(None), None))]),
            CandidatePlan::Unbounded
        );
        assert_eq!(
            range_plan(&[cast(col("qty"), DataType::Int64).gt(lit(5i64))]),
            CandidatePlan::Unbounded
        );
        // A column with no range index.
        assert_eq!(
            range_plan(&[col("other").gt(lit(5i32))]),
            CandidatePlan::Unbounded
        );
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
    fn standard_resolver(_col: &str) -> Option<Arc<dyn Tokenizer>> {
        Some(Arc::new(StandardTokenizer))
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
            fold: false,
        }
    }

    fn terms_ilike(tokens: Vec<LikeToken>) -> CandidatePlan {
        CandidatePlan::TermsLike {
            column: "title".into(),
            tokens,
            fold: true,
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
            plan(col("title").like(lit("rust%"))),
            terms_like(vec![like_token("rust", false, true)])
        );
        assert_eq!(
            plan(col("title").like(lit("%rust"))),
            terms_like(vec![like_token("rust", true, false)])
        );
        assert_eq!(
            plan(col("title").like(lit("%rust%"))),
            terms_like(vec![like_token("rust", true, true)])
        );
        // `_` is a wildcard too: `ab` closed on the left by the pattern
        // start and open on the right; `cd` open on both sides.
        assert_eq!(
            plan(col("title").like(lit("ab_cd%"))),
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
            plan(col("title").like(lit("% quick fox %"))),
            terms_all(&["quick", "fox"])
        );
        // The hyphen closes `fox` on the left; the wildcard leaves the
        // right open.
        assert_eq!(
            plan(col("title").like(lit("%-fox%"))),
            terms_like(vec![like_token("fox", false, true)])
        );
    }

    #[test]
    fn like_word_joiners_leave_an_edge_open_under_standard() {
        // `'` can join `don` to what follows (`don't`), so the token may be
        // the head of a longer term; `.` likewise keeps `fox` open on the
        // left (`a.fox`), while the trailing space closes its right.
        assert_eq!(
            plan(col("title").like(lit("%don'%"))),
            terms_like(vec![like_token("don", true, true)])
        );
        assert_eq!(
            plan(col("title").like(lit("%.fox %"))),
            terms_like(vec![like_token("fox", true, false)])
        );
    }

    #[test]
    fn like_final_sigma_open_right_is_dropped_under_standard() {
        // `ΟΔΟΣ` lowercases to `οδος` on its own but to `οδοσ…` mid-word,
        // so an open-right token has two spellings and cannot be required.
        assert_eq!(
            plan(col("title").like(lit("%ΟΔΟΣ%"))),
            CandidatePlan::Unbounded
        );
        // Closed on the right the word really ends there: kept as a suffix.
        assert_eq!(
            plan(col("title").like(lit("%ΟΔΟΣ"))),
            terms_like(vec![like_token("οδος", true, false)])
        );
    }

    /// A fragment word past the tokenizer's cut, so the query side cuts
    /// it into a piece of the cut length and a remainder.
    const LONG_FRAGMENT_WORD: usize = MAX_TOKEN_CHARS + 45;

    #[test]
    fn like_drops_the_pieces_a_long_fragment_word_is_cut_into() {
        // The pieces' shared edge is no word boundary: a matching row's
        // copy of the word starts wherever it starts, so its own cuts fall
        // elsewhere. `%a…a%` (300 a's) against `y` + 300 a's, indexed as
        // `ya…a` (the cut length) and 46 a's, must not require a term
        // ending in 255 a's.
        let word = "a".repeat(LONG_FRAGMENT_WORD);
        assert_eq!(
            plan(col("title").like(lit(format!("%{word}%")))),
            CandidatePlan::Unbounded
        );
        assert_eq!(
            plan(col("title").ilike(lit(format!("%{word}")))),
            CandidatePlan::Unbounded
        );
        // A separate word beside the long one keeps its own bound.
        assert_eq!(
            plan(col("title").like(lit(format!("%{word} fox%")))),
            terms_like(vec![like_token("fox", false, true)])
        );
        // A word of exactly the cut length cannot be told from a piece.
        let exact_cut = "a".repeat(MAX_TOKEN_CHARS);
        assert_eq!(
            plan(col("title").like(lit(format!("% {exact_cut} fox %")))),
            CandidatePlan::Unbounded
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
        assert_eq!(plan(col("title").like(lit("100\\%%"))), terms_all(&["100"]));
        // A trailing backslash is a pattern the executor rejects.
        assert_eq!(
            plan(col("title").like(lit("rust\\"))),
            CandidatePlan::Unbounded
        );
    }

    #[test]
    fn negated_like_is_unbounded() {
        assert_eq!(
            plan(col("title").not_like(lit("rust%"))),
            CandidatePlan::Unbounded
        );
        assert_eq!(
            plan(col("title").not_ilike(lit("rust%"))),
            CandidatePlan::Unbounded
        );
    }

    #[test]
    fn ilike_lowers_like_like_with_the_fold_flag() {
        // The pattern's own case is irrelevant (the analyzer lowercases);
        // the leaf carries `fold` so the dictionary walk folds `ſ`.
        assert_eq!(
            plan(col("title").ilike(lit("%FoX%"))),
            terms_ilike(vec![like_token("fox", true, true)])
        );
        // Complete tokens without an `s` are exact terms, as under LIKE.
        assert_eq!(
            plan(col("title").ilike(lit("% Quick Fox %"))),
            terms_all(&["quick", "fox"])
        );
        // A complete token holding an `s` may be spelled `ſ` in a matching
        // row's term, so it needs the dictionary even though it is complete.
        assert_eq!(
            plan(col("title").ilike(lit("% rust %"))),
            terms_ilike(vec![like_token("rust", false, false)])
        );
    }

    #[test]
    fn ilike_bounds_only_ascii_tokens() {
        // Arrow's fold widens non-ASCII letters past `to_lowercase`
        // (medial `σ` matches `ς`), so the token cannot be required.
        assert_eq!(
            plan(col("title").ilike(lit("%ΟΔΟΣ%"))),
            CandidatePlan::Unbounded
        );
        // A mixed pattern keeps the ASCII token and drops the other.
        assert_eq!(
            plan(col("title").ilike(lit("%fox%süd%"))),
            terms_ilike(vec![like_token("fox", true, true)])
        );
    }

    #[test]
    fn ilike_prune_leaves_skip_tokens_that_may_be_spelled_with_a_long_s() {
        // `fox` is a safe bloom term; `rust`'s dictionary spelling may be
        // `ruſt`, so no summary is asked about it; `quic` prefix is safe.
        let leaves = like_prune_leaves(
            &[col("title").ilike(lit("quic% rust fox %"))],
            &fts_cols(),
            &standard_resolver,
        );
        assert_eq!(leaves.len(), 2);
        assert!(matches!(
            &leaves[0],
            PruneLeaf::TermPresence { terms, mode: BoolMode::And, .. }
                if *terms == ["fox".to_owned()]
        ));
        assert!(matches!(
            &leaves[1],
            PruneLeaf::Prefix { prefix, .. } if prefix == b"quic"
        ));
        // An open-right token holding `s` loses its term-range leaf too:
        // its `ſ` spelling sorts nowhere near the prefix. Only the safe
        // complete token is left to ask about.
        let leaves = like_prune_leaves(
            &[col("title").ilike(lit("sun% fox %"))],
            &fts_cols(),
            &standard_resolver,
        );
        assert_eq!(leaves.len(), 1);
        assert!(matches!(
            &leaves[0],
            PruneLeaf::TermPresence { terms, mode: BoolMode::And, .. }
                if *terms == ["fox".to_owned()]
        ));
    }

    #[test]
    fn like_needs_an_fts_column_and_a_literal_pattern() {
        assert_eq!(
            plan(col("category").like(lit("rust%"))),
            CandidatePlan::Unbounded
        );
        assert_eq!(
            plan(col("title").like(col("category"))),
            CandidatePlan::Unbounded
        );
        // Wildcards only: no fragment, nothing to bound.
        assert_eq!(
            plan(col("title").like(lit("%_%"))),
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
        assert!(plan(col("title").like(lit("rust%"))).has_like());
        assert!(
            plan(
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
            &RangeColumns::new(),
            &standard_resolver,
        );
        assert_eq!(p, CandidatePlan::Unbounded);
    }

    #[test]
    fn lowering_uses_the_per_column_tokenizer() {
        // The same literal lowers to different terms under a plain and a
        // stemming column: the lowering must pick the column's own chain.
        let filters = [col("title").eq(lit("Running"))];
        let terms = |tokens: &[&str]| CandidatePlan::TermsAll {
            column: "title".to_owned(),
            tokens: tokens.iter().map(|t| t.to_string()).collect(),
        };
        assert_eq!(
            CandidatePlan::from_filters(
                &filters,
                &fts_cols(),
                &RangeColumns::new(),
                &standard_resolver
            ),
            terms(&["running"])
        );
        assert_eq!(
            CandidatePlan::from_filters(
                &filters,
                &fts_cols(),
                &RangeColumns::new(),
                &stemming_resolver
            ),
            terms(&["run"])
        );
    }

    /// `exact_contains` of `expr` over `title` under the standard analyzer.
    fn exact(expr: Expr) -> Option<ExactContains> {
        exact_contains(&expr, &fts_cols(), &standard_resolver)
    }

    fn needle(text: &str) -> Option<ExactContains> {
        Some(ExactContains {
            column: "title".into(),
            needle: text.into(),
        })
    }

    #[test]
    fn exact_contains_accepts_one_whole_standard_token_between_wildcards() {
        let accepted = [
            ("%bbc%", "bbc"),
            ("%BBC%", "bbc"),
            ("%%bbc%%", "bbc"),
            // Escapes read as Arrow reads them: `\x` is `x` for any `x`.
            (r"%b\_c%", "b_c"),
            (r"%\b\b\c%", "bbc"),
            // Joiners the word rules keep inside a word.
            ("%don't%", "don't"),
            ("%3.5%", "3.5"),
            ("%u.s%", "u.s"),
        ];
        for (pattern, want) in accepted {
            assert_eq!(
                exact(col("title").ilike(lit(pattern))),
                needle(want),
                "{pattern}"
            );
        }
        // The longest word the tokenizer keeps whole.
        let longest = "a".repeat(MAX_TOKEN_CHARS);
        assert_eq!(
            exact(col("title").ilike(lit(format!("%{longest}%")))),
            needle(&longest)
        );
        // A qualified column is the same column.
        assert_eq!(exact(col("t.title").ilike(lit("%bbc%"))), needle("bbc"));
    }

    #[test]
    fn exact_contains_leaves_everything_the_text_must_decide_to_the_verified_path() {
        let too_long = "a".repeat(MAX_TOKEN_CHARS + 1);
        let rejected = [
            // Case-sensitive, negated.
            col("title").like(lit("%bbc%")),
            col("title").not_ilike(lit("%bbc%")),
            // Anchored at either end, or both.
            col("title").ilike(lit("bbc%")),
            col("title").ilike(lit("%bbc")),
            col("title").ilike(lit("bbc")),
            // A one-character gap, and several literals.
            col("title").ilike(lit("%_bbc%")),
            col("title").ilike(lit("%b_c%")),
            col("title").ilike(lit("%bbc%news%")),
            // A literal that is not one whole token.
            col("title").ilike(lit("%bbc.%")),
            col("title").ilike(lit("%.net%")),
            col("title").ilike(lit("%a.1%")),
            col("title").ilike(lit("%bbc news%")),
            col("title").ilike(lit(format!("%{too_long}%"))),
            // Escaped wildcards and backslashes are literal text.
            col("title").ilike(lit(r"%\\bbc%")),
            col("title").ilike(lit(r"%bbc\%")),
            col("title").ilike(lit(r"%\%bbc%")),
            col("title").ilike(lit(r"%bbc\")),
            // No literal at all.
            col("title").ilike(lit("%%")),
            // A non-ASCII literal.
            col("title").ilike(lit("%riſe%")),
            // Not a bare FTS column against a literal.
            col("body").ilike(lit("%bbc%")),
            cast(col("title"), DataType::Utf8).ilike(lit("%bbc%")),
            col("title").ilike(col("title")),
        ];
        for expr in rejected {
            assert_eq!(exact(expr.clone()), None, "{expr}");
        }
        // Analysis chains: the rule is plain `standard`'s.
        let expr = col("title").ilike(lit("%bbc%"));
        for resolve in [stemming_resolver, stopping_resolver] {
            assert_eq!(exact_contains(&expr, &fts_cols(), &resolve), None);
        }
    }

    /// `exact_filter` over `title` and `body`, both standard.
    fn exact_tree(expr: Expr) -> Option<ExactFilter> {
        let cols: HashSet<&str> = HashSet::from(["title", "body"]);
        exact_filter(&expr, &cols, &standard_resolver)
    }

    fn leaf(column: &str, needle: &str) -> ExactFilter {
        ExactFilter::Contains(ExactContains {
            column: column.into(),
            needle: needle.into(),
        })
    }

    #[test]
    fn exact_filter_takes_and_or_trees_of_exact_leaves_flattening_chains() {
        let japan = || col("title").ilike(lit("%Japan%"));
        let japanese = || col("title").ilike(lit("%Japanese%"));
        let bbc = || col("body").ilike(lit("%bbc%"));
        assert_eq!(exact_tree(japan()), Some(leaf("title", "japan")));
        assert_eq!(
            exact_tree(japan().or(japanese()).or(bbc())),
            Some(ExactFilter::Or(vec![
                leaf("title", "japan"),
                leaf("title", "japanese"),
                leaf("body", "bbc"),
            ])),
            "a chain of ORs is one node, across columns"
        );
        assert_eq!(
            exact_tree(japan().and(bbc()).or(japanese())),
            Some(ExactFilter::Or(vec![
                ExactFilter::And(vec![leaf("title", "japan"), leaf("body", "bbc")]),
                leaf("title", "japanese"),
            ]))
        );
        // One part that is not exact leaves the whole tree verified.
        for expr in [
            japan().or(col("category").eq(lit("y"))),
            japan().or(col("title").ilike(lit("japan%"))),
            japan().or(col("title").like(lit("%japan%"))),
            japan().and(japanese().or(col("title").ilike(lit("%_x%")))),
            Expr::Not(Box::new(japan().or(japanese()))),
            Expr::Not(Box::new(japan())),
        ] {
            assert_eq!(exact_tree(expr.clone()), None, "{expr}");
        }
    }

    #[test]
    fn exact_rows_combine_proven_and_doubtful_through_and_and_or() {
        let rows = |proven: &[u32], doubtful: &[u32]| ContainsRows {
            proven: proven.iter().copied().collect(),
            doubtful: doubtful.iter().copied().collect(),
        };
        // a: proves 1 2, doubts 3; b: proves 2 4, doubts 1 5.
        let leaf_rows = |leaf: &ExactContains| match leaf.needle.as_str() {
            "a" => rows(&[1, 2], &[3]),
            "b" => rows(&[2, 4], &[1, 5]),
            other => panic!("no leaf {other}"),
        };
        let (a, b) = (leaf("title", "a"), leaf("title", "b"));
        // AND: proven in both; admitted in both, less the proven.
        assert_eq!(
            ExactFilter::And(vec![a.clone(), b.clone()]).rows(&leaf_rows),
            rows(&[2], &[1])
        );
        // OR: proven in either; admitted in either, less the proven.
        assert_eq!(
            ExactFilter::Or(vec![a.clone(), b.clone()]).rows(&leaf_rows),
            rows(&[1, 2, 4], &[3, 5])
        );
        // Nested: (a AND b) OR a.
        assert_eq!(
            ExactFilter::Or(vec![ExactFilter::And(vec![a.clone(), b]), a]).rows(&leaf_rows),
            rows(&[1, 2], &[3])
        );
    }

    /// The range-indexed column in [`titled_superfile`].
    const HITS: &str = "hits";

    /// Four titles with a range-indexed `hits` column, and the table id of
    /// `hits` that a range plan names:
    ///
    /// | row | title                   | hits |
    /// |-----|-------------------------|------|
    /// | 0   | rust async runtime      | 5    |
    /// | 1   | python data pipeline    | 50   |
    /// | 2   | rust embedded system    | 7    |
    /// | 3   | javascript web frontend | 70   |
    fn titled_superfile() -> (Bytes, FieldId) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(HITS, DataType::Int32, false),
        ]));
        let opts = BuilderOptions::new(
            Arc::clone(&schema),
            "doc_id",
            vec![FtsConfig::new("title")],
            vec![],
        )
        .with_range_columns(vec![HITS.into()]);
        let hits = opts.field_id_of_column(HITS).expect("hits field id");
        let mut builder = SuperfileBuilder::new(opts).expect("builder");
        let titles = LargeStringArray::from(vec![
            "rust async runtime",
            "python data pipeline",
            "rust embedded system",
            "javascript web frontend",
        ]);
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(decimal128_ids([10, 11, 12, 13])),
                Arc::new(titles),
                Arc::new(Int32Array::from(vec![5, 50, 7, 70])),
            ],
        )
        .expect("batch");
        builder.add_batch(&batch, &[]).expect("add batch");
        (Bytes::from(builder.finish().expect("finish")), hits)
    }

    /// [`titled_superfile`] with all its bytes in memory.
    fn resident_reader() -> Arc<SuperfileReader> {
        Arc::new(SuperfileReader::open(titled_superfile().0).expect("open"))
    }

    fn hits_between(field: FieldId, lo: i64, hi: i64) -> CandidatePlan {
        CandidatePlan::Range {
            field,
            kind: KeyKind::Signed,
            lo: signed_key(lo),
            hi: signed_key(hi),
        }
    }

    fn one_thread_pool() -> ThreadPool {
        ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool")
    }

    fn row_list(rows: Option<RoaringBitmap>) -> Option<Vec<u32>> {
        rows.map(|r| r.iter().collect())
    }

    #[tokio::test]
    async fn resident_plans_finish_in_one_poll_on_a_pool_thread() {
        // Both resident shapes of one superfile: all its bytes in memory, and
        // the disk cache's local copy (opened lazily over a source, then
        // given its bytes), and each plan shape the provider sends to the pool.
        // - Polled once on a pool thread, every match finishes.
        // - Each gives the rows the async match gives.
        // This test pins that a resident match never has to rerun after its
        // pool hop, which would do its work twice.
        let (bytes, hits) = titled_superfile();
        let in_memory = SuperfileReader::open(bytes.clone()).expect("open");
        let source: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(bytes.clone()));
        let mut disk_copy = SuperfileReader::open_lazy(source).await.expect("lazy open");
        disk_copy
            .install_resident_parquet(bytes, (0, 0))
            .expect("install the local copy");
        let pool = one_thread_pool();
        let any = |terms: &[&str]| CandidatePlan::TermsAny {
            column: "title".into(),
            terms: terms.iter().map(|t| t.to_string()).collect(),
        };
        let cases = [
            ("one word", terms_all(&["rust"]), Some(vec![0, 2])),
            (
                "either of two words",
                any(&["python", "web"]),
                Some(vec![1, 3]),
            ),
            ("any of no words", any(&[]), Some(vec![])),
            ("a range", hits_between(hits, 0, 10), Some(vec![0, 2])),
            (
                "a word and a range",
                CandidatePlan::And(vec![terms_all(&["rust"]), hits_between(hits, 0, 6)]),
                Some(vec![0]),
            ),
            (
                "two words no row holds",
                CandidatePlan::And(vec![terms_all(&["python"]), terms_all(&["rust"])]),
                Some(vec![]),
            ),
            (
                "one title or another",
                CandidatePlan::Or(vec![terms_all(&["rust"]), terms_all(&["python"])]),
                Some(vec![0, 1, 2]),
            ),
            (
                "a word or anything",
                CandidatePlan::Or(vec![terms_all(&["rust"]), CandidatePlan::Unbounded]),
                None,
            ),
        ];
        for (shape, reader) in [("in memory", &in_memory), ("disk copy", &disk_copy)] {
            // Fresh memos per shape, so the second one reads its own indexes.
            let memos = TermMemos::new();
            for (what, plan, want) in &cases {
                let (rows, _) = pool
                    .install(|| plan.evaluate(reader, None, &memos).now_or_never())
                    .unwrap_or_else(|| panic!("{shape}, {what}: the match waited on the pool"))
                    .expect(what);
                let (async_rows, _) = plan
                    .evaluate(reader, Some(&pool), &memos)
                    .await
                    .expect(what);
                assert_eq!(row_list(rows.clone()), *want, "{shape}, {what}");
                assert_eq!(
                    rows, async_rows,
                    "{shape}, {what}: differs from the async match"
                );
            }
        }
    }

    #[tokio::test]
    async fn a_resident_superfile_is_matched_on_the_pool() {
        // A resident superfile, and a pool whose only thread is busy.
        // - The first poll queues the match on the pool and does not finish.
        // - Once the pool is free, it finishes with rows 0 and 2.
        // This test pins that planning hands a resident superfile's match to
        // the pool instead of running it on its own task.
        let (pool, release) = held_pool();
        let slots = Semaphore::new(1);
        let mut matched = Box::pin(Arc::new(terms_all(&["rust"])).evaluate_on_pool(
            resident_reader(),
            Uuid::nil(),
            &pool,
            &slots,
            TermMemos::new(),
        ));
        assert!(
            (&mut matched).now_or_never().is_none(),
            "the match waits for the pool"
        );
        drop(release);
        let (rows, _) = timeout(MUST_FINISH, matched)
            .await
            .expect("finishes once the pool is free")
            .expect("match");
        assert_eq!(row_list(rows), Some(vec![0, 2]));
    }

    #[tokio::test]
    async fn a_deep_plan_is_matched_off_the_pool() {
        // A resident superfile, a busy pool, and a plan nesting `AND` and `OR`
        // one level deeper than the pool takes.
        // - The match finishes anyway, with rows 0 and 2.
        // This test pins that a deep plan stays on the planning task, whose
        // stack holds it; on a pool thread it overflows and aborts.
        let mut deep = terms_all(&["rust"]);
        for level in 1..=MAX_POOL_PLAN_DEPTH {
            let other = terms_all(&["rust"]);
            deep = if level % 2 == 0 {
                CandidatePlan::And(vec![deep, other])
            } else {
                CandidatePlan::Or(vec![deep, other])
            };
        }
        assert_eq!(deep.depth(), MAX_POOL_PLAN_DEPTH + 1);
        let (pool, _release) = held_pool();
        let slots = Semaphore::new(1);
        let matched = Arc::new(deep).evaluate_on_pool(
            resident_reader(),
            Uuid::nil(),
            &pool,
            &slots,
            TermMemos::new(),
        );
        let (rows, _) = timeout(MUST_FINISH, matched)
            .await
            .expect("finishes without the busy pool")
            .expect("match");
        assert_eq!(row_list(rows), Some(vec![0, 2]));
    }

    #[tokio::test]
    async fn a_lazy_superfile_is_matched_off_the_pool() {
        // The same superfile opened lazily, a busy pool, and no free slot.
        // - The match finishes anyway, with rows 0 and 2.
        // This test pins that a lazy superfile keeps the async path and takes
        // no slot, so a match waiting on a download holds neither a pool
        // thread nor a slot.
        let source: Arc<dyn LazyByteSource> =
            Arc::new(BytesLazyByteSource::new(titled_superfile().0));
        let reader = Arc::new(SuperfileReader::open_lazy(source).await.expect("lazy open"));
        let (pool, _release) = held_pool();
        let no_slots = Semaphore::new(0);
        let matched = Arc::new(terms_all(&["rust"])).evaluate_on_pool(
            reader,
            Uuid::nil(),
            &pool,
            &no_slots,
            TermMemos::new(),
        );
        let (rows, _) = timeout(MUST_FINISH, matched)
            .await
            .expect("finishes while the pool is busy")
            .expect("match");
        assert_eq!(row_list(rows), Some(vec![0, 2]));
    }

    #[tokio::test]
    async fn a_like_plan_is_matched_off_the_pool() {
        // A resident superfile, a `LIKE` plan not yet bound to its words, a
        // free pool, and no free slot.
        // - The match finishes, with rows 0 and 2.
        // This test pins that a plan whose dictionary walk hops to a pool on
        // its own never takes the pool path, where that walk would be left
        // running on the global pool.
        let like = Arc::new(plan(col("title").like(lit("rust%"))));
        let pool = one_thread_pool();
        let no_slots = Semaphore::new(0);
        let matched = like.evaluate_on_pool(
            resident_reader(),
            Uuid::nil(),
            &pool,
            &no_slots,
            TermMemos::new(),
        );
        let (rows, _) = timeout(MUST_FINISH, matched)
            .await
            .expect("finishes without a slot")
            .expect("match");
        assert_eq!(row_list(rows), Some(vec![0, 2]));
    }

    /// A source whose every read waits one turn first, as a shared cell
    /// another query is filling does, and notes how many `slots` were free
    /// when it was asked.
    struct SlotWatchingSource {
        bytes: BytesLazyByteSource,
        slots: Arc<Semaphore>,
        free_at_read: Mutex<Vec<usize>>,
    }

    #[async_trait]
    impl LazyByteSource for SlotWatchingSource {
        fn size(&self) -> u64 {
            self.bytes.size()
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            self.free_at_read
                .lock()
                .expect("free_at_read lock")
                .push(self.slots.available_permits());
            yield_now().await;
            self.bytes.range(start, len).await
        }
    }

    #[tokio::test]
    async fn a_match_that_waits_on_the_pool_reruns_here_without_its_slot() {
        // A resident superfile whose index reads wait one turn, and one slot.
        // - On the pool the match reads while holding the slot, and gives up.
        // - It reruns here with the slot free, and finds rows 0 and 2.
        // This test pins the fallback: a match the pool cannot finish in one
        // poll still answers, and does not hold a slot while it reruns.
        let bytes = titled_superfile().0;
        let slots = Arc::new(Semaphore::new(1));
        let source = Arc::new(SlotWatchingSource {
            bytes: BytesLazyByteSource::new(bytes.clone()),
            slots: Arc::clone(&slots),
            free_at_read: Mutex::new(Vec::new()),
        });
        let as_source: Arc<dyn LazyByteSource> = Arc::clone(&source) as Arc<dyn LazyByteSource>;
        let mut reader = SuperfileReader::open_lazy_with(as_source, OpenOptions::default())
            .await
            .expect("lazy open");
        reader
            .install_resident_parquet(bytes, (0, 0))
            .expect("promote");
        source
            .free_at_read
            .lock()
            .expect("free_at_read lock")
            .clear();

        let pool = one_thread_pool();
        let (rows, _) = timeout(
            MUST_FINISH,
            Arc::new(terms_all(&["rust"])).evaluate_on_pool(
                Arc::new(reader),
                Uuid::nil(),
                &pool,
                &slots,
                TermMemos::new(),
            ),
        )
        .await
        .expect("finishes")
        .expect("match");
        assert_eq!(row_list(rows), Some(vec![0, 2]));
        let free_at_read = source.free_at_read.lock().expect("free_at_read lock");
        assert_eq!(
            free_at_read.first(),
            Some(&0),
            "the first read is on the pool, holding the slot"
        );
        assert_eq!(
            free_at_read.last(),
            Some(&1),
            "the rerun reads with the slot free"
        );
    }

    #[tokio::test]
    async fn a_match_holds_its_slot_only_while_on_the_pool() {
        // A resident superfile and one slot.
        // - Queued on a busy pool, the match holds the slot.
        // - Dropped, as a cancelled query drops it, it gives the slot back.
        // - Run to the end on a free pool, it gives the slot back too.
        // This test pins the bound on how many matches one query puts on the
        // pool.
        let reader = resident_reader();
        let rust = Arc::new(terms_all(&["rust"]));
        let slots = Semaphore::new(1);
        let (busy, _release) = held_pool();
        let mut queued = Box::pin(Arc::clone(&rust).evaluate_on_pool(
            Arc::clone(&reader),
            Uuid::nil(),
            &busy,
            &slots,
            TermMemos::new(),
        ));
        assert!((&mut queued).now_or_never().is_none());
        assert_eq!(slots.available_permits(), 0, "the queued match holds it");
        drop(queued);
        assert_eq!(
            slots.available_permits(),
            1,
            "a dropped match gives it back"
        );

        let free = one_thread_pool();
        timeout(
            MUST_FINISH,
            rust.evaluate_on_pool(reader, Uuid::nil(), &free, &slots, TermMemos::new()),
        )
        .await
        .expect("finishes on a free pool")
        .expect("match");
        assert_eq!(
            slots.available_permits(),
            1,
            "a finished match gives it back"
        );
    }
}
