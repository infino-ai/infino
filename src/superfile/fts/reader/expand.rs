// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Term-dictionary expansion on [`FtsReader`]: widen the tokens of one
//! `LIKE` leaf to the indexed terms they cover, so the table layer's SQL
//! `WHERE` pushdown can answer a substring predicate from posting lists
//! instead of a column scan, and answer an `ILIKE '%needle%'` exactly
//! ([`FtsReader::contains_rows`]). Its own `impl FtsReader` block, split
//! from the reader `core`.

use std::{borrow::Cow, mem::size_of, str, sync::Arc};

use rayon::ThreadPool;
use roaring::RoaringBitmap;

use super::{core::*, cursor::TermCursor, metadata::ColumnMeta, work::MatchWork};
use crate::{
    memory::{ConnectionMemoryBudget, OverBudget, Reservation},
    runtime_bridge::run_on_pool,
    runtime_metrics::op_stats::timed_section,
    superfile::{
        ReadError,
        error::FtsError,
        fts::{
            posting::BLOCK_LEN,
            tokenize::{MAX_TOKEN_CHARS, STANDARD_TOKENIZER},
        },
        id_space::{DocMap, FtsDocId},
    },
    utils::terms::{make_key, value::DictEntry},
};

/// Long s (U+017F). Simple case folding puts it in `s`'s class;
/// `to_lowercase` leaves it, so an indexed term can carry it.
const LONG_S: char = 'ſ';

/// Kelvin sign (U+212A), folded with `k`. `to_lowercase` maps it to `k`
/// before indexing, so a term never carries it; folded here anyway so the
/// comparison does not depend on that. Written as an escape: the glyph is
/// indistinguishable from an ASCII `K` in most fonts, and an ASCII `K`
/// here would make the fold a no-op without any test noticing.
const KELVIN_SIGN: char = '\u{212A}';

/// Every non-ASCII character Unicode simple case folding puts in an ASCII
/// letter's class, paired with that letter. These are the only spellings
/// an `ILIKE` on an ASCII token can match that `to_lowercase` of ASCII
/// text never produces; every fold rule in the crate derives from this
/// table so the pairs are written once.
pub(crate) const FOLD_PAIRS: &[(char, char)] = &[(LONG_S, 's'), (KELVIN_SIGN, 'k')];

/// The ASCII letter whose fold partner an indexed term can actually
/// carry: `s` (the long s survives `to_lowercase`; the Kelvin sign does
/// not). An `ILIKE` token holding it may be spelled with `ſ` in a
/// matching row's term, so only a dictionary walk can find it.
pub(crate) const LONG_S_ASCII: char = FOLD_PAIRS[0].1;

/// Combining dot above (U+0307). `to_lowercase` turns `İ` (U+0130) into an
/// `i` followed by this mark, so a term can hold an `i` its row spelled
/// `İ` — a letter Arrow's `ILIKE` does not match against `i`. Written as
/// an escape: the mark is invisible on its own.
const COMBINING_DOT_ABOVE: char = '\u{307}';

/// The letter `İ` lowercases onto, ahead of [`COMBINING_DOT_ABOVE`].
const DOTTED_I_BASE: char = 'i';

/// Postings bytes one fetch wave of [`FtsReader::contains_rows`] may pull
/// before its terms are unioned and the bytes released. The exact path
/// has no term cap, so a needle covering much of a large vocabulary would
/// otherwise hold every covered term's postings at once. 8 MiB still
/// carries thousands of typical terms per round trip. Each wave is charged
/// to the connection memory budget while it is held, and the table layer
/// caps how many superfiles fetch at once, so the waves of a scan over many
/// superfiles are bounded by both rather than by this size alone.
const CONTAINS_FETCH_BATCH_BYTES: usize = 8 << 20;

/// Term values a walk's budget charge grows by at a time (see
/// `ValueCharge`): small next to any budget a connection sets, and
/// enough that the budget's shared counter stays off the per-key path.
const VALUE_CHARGE_STEP: usize = 4096;

/// The query-term weight a match-only cursor is built with: nothing is
/// scored, so nothing is weighted.
const UNWEIGHTED: u32 = 1;

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

    /// Whether a match of this pattern may begin in `term` and run on
    /// past a tokenizer cut (see [`straddles_cut`]); `view` is `term` as
    /// [`Self::covers`] compares it. Only a token whose start may sit
    /// mid-term can: a token closed on the left starts a word, a word's
    /// first piece is the full cut length, and a pattern token is shorter
    /// than that, so it ends before the first cut.
    fn crosses_cut(&self, term: &str, view: &str) -> bool {
        match self {
            TermPattern::Suffix(text) | TermPattern::Contains(text) => {
                straddles_cut(term, view, text)
            }
            TermPattern::Exact(_) | TermPattern::Prefix(_) => false,
        }
    }

    /// Whether a term this pattern covers under `ILIKE` may still come
    /// from a row the `ILIKE` does not match. The one way is a dotted
    /// capital I: its row spells `İ`, which Arrow does not fold to `i`, and
    /// the term holds `i` + [`COMBINING_DOT_ABOVE`] — or only the `i`,
    /// when a tokenizer cut fell between the two and the term is a piece
    /// of the cut length. Only an `i` ending the text can meet the mark
    /// (inside the text the mark would split the match), so this asks for
    /// that first. It flags a term holding the mark anywhere, and every
    /// piece of the cut length: a caller checks those rows' text, which
    /// costs work, never a row.
    fn doubtful_cover(&self, term: &str) -> bool {
        self.text().ends_with(DOTTED_I_BASE)
            && (term.contains(COMBINING_DOT_ABOVE) || is_cut_length(term))
    }
}

/// Whether `term` is exactly the tokenizer's cut length, as every piece of
/// a cut word but the last is. The byte length is tested first: a term
/// under the cut length in bytes is under it in characters, so the common
/// term costs one compare.
fn is_cut_length(term: &str) -> bool {
    term.len() >= MAX_TOKEN_CHARS && term.chars().count() == MAX_TOKEN_CHARS
}

/// Whether `term` may be the piece before a tokenizer cut that an
/// occurrence of `text` runs across.
///
/// The tokenizer indexes a word longer than [`MAX_TOKEN_CHARS`] as
/// consecutive pieces, each of exactly that many characters but the last,
/// so a match crossing a cut is split between two terms and neither holds
/// it. The piece before the cut is the one that shows it: it is exactly the
/// cut length and its folded `view` ends with a proper, non-empty head of
/// `text`. A match of text no longer than the cut length crosses at most
/// one cut, so testing that piece finds every such match. A genuine word
/// of exactly the cut length passes too; that only admits a row the
/// caller then checks, never loses one.
fn straddles_cut(term: &str, view: &str, text: &str) -> bool {
    is_cut_length(term)
        && text
            .char_indices()
            .skip(1)
            .any(|(at, _)| view.ends_with(&text[..at]))
}

/// A [`TermPattern`] with its text owned, so a leaf's patterns can move
/// onto the reader pool with the dictionary walk.
#[derive(Debug, Clone)]
enum OwnedPattern {
    Exact(String),
    Prefix(String),
    Suffix(String),
    Contains(String),
}

impl OwnedPattern {
    fn borrow(&self) -> TermPattern<'_> {
        match self {
            OwnedPattern::Exact(text) => TermPattern::Exact(text),
            OwnedPattern::Prefix(text) => TermPattern::Prefix(text),
            OwnedPattern::Suffix(text) => TermPattern::Suffix(text),
            OwnedPattern::Contains(text) => TermPattern::Contains(text),
        }
    }
}

impl TermPattern<'_> {
    fn into_owned(self) -> OwnedPattern {
        match self {
            TermPattern::Exact(text) => OwnedPattern::Exact(text.to_owned()),
            TermPattern::Prefix(text) => OwnedPattern::Prefix(text.to_owned()),
            TermPattern::Suffix(text) => OwnedPattern::Suffix(text.to_owned()),
            TermPattern::Contains(text) => OwnedPattern::Contains(text.to_owned()),
        }
    }
}

/// A dictionary term with `ſ` and `K` folded to the ASCII members of
/// their case-folding classes — the view an `ILIKE` token is compared
/// against. Borrowed when nothing folds.
fn fold_term(term: &str) -> Cow<'_, str> {
    let fold = |c: char| {
        FOLD_PAIRS
            .iter()
            .find(|&&(from, _)| from == c)
            .map(|&(_, to)| to)
    };
    if term.chars().any(|c| fold(c).is_some()) {
        Cow::Owned(term.chars().map(|c| fold(c).unwrap_or(c)).collect())
    } else {
        Cow::Borrowed(term)
    }
}

/// What a dictionary walk keeps of each term it admits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Keep {
    /// The term's text, for a match to resolve again: the verified path,
    /// where every admitted term is only a candidate.
    Terms,
    /// The term's dictionary value, sorted into terms whose rows are
    /// proven to match and terms whose rows need their text checked: the
    /// exact path, which reads the postings straight from the values.
    Values,
}

/// One pattern's terms as a walk collects them, with the cap it may not
/// exceed. Past the cap the pattern is too broad for a posting-list
/// answer and its collection stops.
struct Collected {
    /// Admitted terms' text, in lex order ([`Keep::Terms`]).
    terms: Vec<String>,
    /// Values of admitted terms whose rows match ([`Keep::Values`]).
    proven: Vec<DictEntry>,
    /// Values of admitted terms whose rows may not match
    /// ([`Keep::Values`]).
    doubtful: Vec<DictEntry>,
    too_many: bool,
}

impl Collected {
    fn new() -> Self {
        Self {
            terms: Vec::new(),
            proven: Vec::new(),
            doubtful: Vec::new(),
            too_many: false,
        }
    }

    /// Record `term` (`doubtful` when its rows may not match); `false`
    /// once the cap is hit (the caller stops feeding this pattern).
    fn admit(
        &mut self,
        term: &str,
        value: DictEntry,
        doubtful: bool,
        keep: Keep,
        max_terms: usize,
    ) -> bool {
        if self.terms.len() + self.proven.len() + self.doubtful.len() == max_terms {
            self.too_many = true;
            return false;
        }
        match keep {
            Keep::Terms => self.terms.push(term.to_owned()),
            Keep::Values if doubtful => self.doubtful.push(value),
            Keep::Values => self.proven.push(value),
        }
        true
    }

    fn finish(self) -> Option<Vec<String>> {
        (!self.too_many).then_some(self.terms)
    }
}

/// A dictionary key that is not UTF-8. Keys are the tokenizer's UTF-8
/// output, so one that fails is a damaged dictionary; skipping it could
/// drop the rows it indexes, so a walk fails instead.
fn non_utf8_key() -> FtsError {
    FtsError::Read(ReadError::Malformed(
        "fts dictionary key is not UTF-8".into(),
    ))
}

/// `bytes` of `budget` for the exact path's `what`, held until the guard
/// drops; `None` when no budget is attached. A refusal is
/// [`FtsError::OverBudget`], raised before the bytes are fetched or
/// allocated.
fn reserve_exact(
    budget: Option<&Arc<ConnectionMemoryBudget>>,
    bytes: usize,
    what: &str,
) -> Result<Option<Reservation>, FtsError> {
    budget
        .map(|budget| {
            budget
                .try_reserve(bytes)
                .map_err(|refusal| exact_over_budget(what, refusal))
        })
        .transpose()
}

/// The budget's refusal of the exact path's `what`, as
/// [`FtsError::OverBudget`].
fn exact_over_budget(what: &str, refusal: OverBudget) -> FtsError {
    FtsError::OverBudget(format!("exact ILIKE {what}, {refusal}"))
}

/// Term values a walk keeps, charged to a budget while it walks: the
/// reservation runs up to [`VALUE_CHARGE_STEP`] values ahead of what has
/// been admitted, so a broad needle over a large vocabulary is refused
/// part-way through the walk rather than after it has collected every
/// value.
struct ValueCharge {
    held: Reservation,
    /// Values the reservation covers.
    covered: usize,
}

impl ValueCharge {
    fn new(budget: &Arc<ConnectionMemoryBudget>) -> Result<Self, FtsError> {
        let held = budget
            .try_reserve(0)
            .map_err(|refusal| exact_over_budget("term values", refusal))?;
        Ok(Self { held, covered: 0 })
    }

    /// Grow the reservation, a step at a time, until it covers `values`.
    fn cover(&mut self, values: usize) -> Result<(), FtsError> {
        while self.covered < values {
            self.held
                .try_grow(VALUE_CHARGE_STEP * size_of::<DictEntry>())
                .map_err(|refusal| exact_over_budget("term values", refusal))?;
            self.covered += VALUE_CHARGE_STEP;
        }
        Ok(())
    }
}

/// One `Pfor` term's postings reference, as the exact path's fetch waves
/// read it.
#[derive(Debug, Clone, Copy)]
struct TermBody {
    metadata_offset: usize,
    /// The body's length in bytes.
    length: usize,
    short: bool,
}

/// How many of `bodies`, from the front, one fetch wave takes: bodies
/// until their bytes would pass [`CONTAINS_FETCH_BATCH_BYTES`], and
/// always at least one.
fn wave_len(bodies: &[TermBody]) -> usize {
    let mut bytes = 0usize;
    bodies
        .iter()
        .position(|body| {
            bytes += body.length;
            bytes > CONTAINS_FETCH_BATCH_BYTES
        })
        .map_or(bodies.len(), |past| past.max(1))
}

/// Rows of one column an `ILIKE '%needle%'` matches, as the dictionary
/// decides them ([`FtsReader::contains_rows`]).
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct ContainsRows {
    /// Parquet rows the dictionary proves match.
    pub(crate) proven: RoaringBitmap,
    /// Parquet rows that may match, which only their text can decide: a
    /// word cut by the tokenizer, or a dotted capital I (see
    /// [`TermPattern::doubtful_cover`]). Disjoint from `proven`; every row
    /// outside both provably does not match.
    pub(crate) doubtful: RoaringBitmap,
}

impl FtsReader {
    /// Expand each of `patterns` into the indexed terms of `column` it
    /// covers, in lex order, in one pass over the dictionary. A slot is
    /// `None` when its pattern cannot be answered from the dictionary:
    /// more than `max_terms` terms qualify (too broad for a posting-list
    /// answer), or it needs the whole column walked and `allow_full_walk`
    /// is off (the caller judged that walk dearer than the scan it would
    /// replace). The caller falls back to scanning for that token.
    ///
    /// An [`TermPattern::Exact`] token is its own expansion. A prefix token
    /// walks only its own subtree. Every suffix or infix token is tested
    /// against each key of one shared walk over the column's whole range, so a
    /// multi-fragment `LIKE` pays for the vocabulary once, not once per token.
    /// Under `fold` (`ILIKE`) terms are compared with `ſ` and `K` folded to `s`
    /// and `k`, and an exact or prefix token holding an `s` joins the shared
    /// walk (its `ſ` spelling sorts elsewhere). The term dictionary is fetched
    /// once when any pattern needs it (one planned range, like a match's
    /// build).
    ///
    /// The dictionary fetch is I/O and stays on the calling runtime; the walk
    /// itself is CPU — up to the column's whole vocabulary — and runs on
    /// `pool` (the configured reader pool, or rayon's global pool when
    /// `None`) behind a oneshot, so no tokio worker sits under it.
    ///
    /// Errors with `FtsError::UnknownColumn` when `column` is not
    /// FTS-indexed in this superfile, like [`Self::token_match`].
    pub(crate) async fn expand_terms(
        &self,
        column: &str,
        patterns: &[TermPattern<'_>],
        fold: bool,
        max_terms: usize,
        allow_full_walk: bool,
        pool: Option<&ThreadPool>,
    ) -> Result<(Vec<Option<Vec<String>>>, MatchWork), FtsError> {
        self.resolve_column_id(column)?;
        let walks: Vec<Walk> = patterns.iter().map(|p| p.walk(fold)).collect();
        let mut work = MatchWork::default();
        let needs_dict = walks
            .iter()
            .any(|w| *w == Walk::Subtree || (*w == Walk::Full && allow_full_walk));
        let collected = if needs_dict {
            let dict_bytes = self.dict_bytes_async().await?;
            work.planned_ranges += 1;
            let column = column.to_owned();
            let owned: Vec<OwnedPattern> = patterns.iter().map(|p| p.into_owned()).collect();
            let walks = walks.clone();
            run_on_pool(pool, "like expansion", move || {
                walk_dictionary(
                    &dict_bytes,
                    &column,
                    &owned,
                    &walks,
                    fold,
                    max_terms,
                    allow_full_walk,
                    Keep::Terms,
                    None,
                )
                .map(|(collected, _)| collected)
            })
            .await
            .map_err(|_| FtsError::TaskDropped("like expansion"))??
        } else {
            patterns.iter().map(|_| Collected::new()).collect()
        };
        let mut out = Vec::with_capacity(patterns.len());
        for ((slot, pattern), walk) in collected.into_iter().zip(patterns).zip(&walks) {
            out.push(match walk {
                Walk::None => Some(vec![pattern.text().to_owned()]),
                Walk::Full if !allow_full_walk => None,
                Walk::Subtree | Walk::Full => slot.finish(),
            });
        }
        Ok((out, work))
    }

    /// Every indexed term of `column` that begins with `term_prefix` (the
    /// prefix as it appears in the term dictionary — the caller lowercases it
    /// for the column's analyzer), in lex order, without the column key prefix.
    /// The prefix search's expansion. The dictionary fetch stays on the calling
    /// runtime; the subtree walk runs on `pool` behind a oneshot, like
    /// [`Self::expand_terms`]. Empty when `column` is not FTS-indexed here or
    /// no term matches.
    pub(crate) async fn terms_with_prefix(
        &self,
        column: &str,
        term_prefix: &[u8],
        pool: Option<&ThreadPool>,
    ) -> Result<Vec<Vec<u8>>, FtsError> {
        if !self.has_column(column) {
            return Ok(Vec::new());
        }
        let dict_bytes = self.dict_bytes_async().await?;
        let column = column.to_owned();
        let term_prefix = term_prefix.to_vec();
        run_on_pool(pool, "prefix expansion", move || {
            collect_terms_with_prefix(&dict_bytes, &column, &term_prefix)
        })
        .await
        .map_err(|_| FtsError::TaskDropped("prefix expansion"))?
    }

    /// For each of `needles`, the Parquet rows of `column` an `ILIKE
    /// '%needle%'` matches, from the dictionary and the postings alone, but
    /// for the few rows [`ContainsRows::doubtful`] leaves to the caller; one
    /// result per needle, in order. The caller has established that every
    /// needle is lowercase ASCII and the whole of one token under the
    /// `standard` analyzer (see the table layer's `exact_contains`); under
    /// that rule an indexed term's containing the needle decides its rows,
    /// cut words and the dotted capital I aside.
    ///
    /// Every key of the column is visited once for all the needles, with
    /// no term cap: an exact answer needs every covered term, and there is
    /// no scan to fall back to. The dictionary fetch and the posting
    /// fetches are I/O on this runtime; the walk and the unions are CPU on
    /// `pool`, like [`Self::expand_terms`].
    ///
    /// The covered terms' values, each union's document bitset and each
    /// fetch wave's postings are charged to `budget` while they are held
    /// (no charge when `None`); a refusal fails the call with
    /// `FtsError::OverBudget`.
    ///
    /// Errors with `FtsError::UnknownColumn` when `column` is not
    /// FTS-indexed here, and with `FtsError::ExactNeedsStandard` when this
    /// superfile indexed it with another analyzer — the rule the answer
    /// rests on is the standard analyzer's, and a plan that promised an
    /// exact answer has no fallback.
    pub(crate) async fn contains_rows(
        &self,
        column: &str,
        needles: &[&str],
        pool: Option<&ThreadPool>,
        budget: Option<&Arc<ConnectionMemoryBudget>>,
    ) -> Result<(Vec<ContainsRows>, MatchWork), FtsError> {
        let column_id = self.resolve_column_id(column)?;
        let col = &self.columns[column_id as usize];
        let analyzer = col.tokenizer.name();
        if analyzer != STANDARD_TOKENIZER {
            return Err(FtsError::ExactNeedsStandard {
                column: column.to_owned(),
                analyzer: analyzer.to_owned(),
            });
        }
        let mut work = MatchWork::default();
        if needles.is_empty() {
            return Ok((Vec::new(), work));
        }
        let dict_bytes = self.dict_bytes_async().await?;
        work.planned_ranges += 1;
        let owned_column = column.to_owned();
        let patterns: Vec<OwnedPattern> = needles
            .iter()
            .map(|needle| OwnedPattern::Contains((*needle).to_owned()))
            .collect();
        let walks = vec![Walk::Full; patterns.len()];
        // The values the walk keeps are charged as it admits them.
        let charge = budget.map(ValueCharge::new).transpose()?;
        let (walked, walk_ns) = run_on_pool(pool, "contains walk", move || {
            timed_section(|| {
                walk_dictionary(
                    &dict_bytes,
                    &owned_column,
                    &patterns,
                    &walks,
                    true,
                    // No cap: an exact answer needs every covered term.
                    usize::MAX,
                    true,
                    Keep::Values,
                    charge,
                )
            })
        })
        .await
        .map_err(|_| FtsError::TaskDropped("contains walk"))?;
        work.kernel_cpu_ns += walk_ns;
        // The charge is held until every needle's rows are unioned.
        let (walked, _values) = walked?;
        let mut out = Vec::with_capacity(needles.len());
        for walked in walked {
            let proven = self
                .union_rows(col, walked.proven, pool, budget, &mut work)
                .await?;
            let mut doubtful = self
                .union_rows(col, walked.doubtful, pool, budget, &mut work)
                .await?;
            doubtful -= &proven;
            out.push(ContainsRows { proven, doubtful });
        }
        Ok((out, work))
    }

    /// The Parquet rows the postings behind `values` hold, unioned. Inline
    /// (df=1) values carry their document; the rest are fetched in waves
    /// of at most [`CONTAINS_FETCH_BATCH_BYTES`] on this runtime and ORed
    /// into one document bitset on `pool`, each wave's bytes released
    /// before the next is fetched. The bitset becomes rows through the
    /// blob's doc map last: a merged superfile numbers its documents in an
    /// order of its own. The bitset, and each wave while it is held, are
    /// charged to `budget`.
    async fn union_rows(
        &self,
        col: &ColumnMeta,
        values: Vec<DictEntry>,
        pool: Option<&ThreadPool>,
        budget: Option<&Arc<ConnectionMemoryBudget>>,
        work: &mut MatchWork,
    ) -> Result<RoaringBitmap, FtsError> {
        if values.is_empty() {
            return Ok(RoaringBitmap::new());
        }
        let n_docs = self.n_docs;
        let mut inline: Vec<u32> = Vec::new();
        let mut bodies: Vec<TermBody> = Vec::new();
        for value in values {
            match value {
                DictEntry::Inline { doc_id, .. } => inline.push(doc_id),
                DictEntry::Pfor {
                    metadata_offset,
                    postings_length,
                    short,
                } => bodies.push(TermBody {
                    metadata_offset: metadata_offset as usize,
                    length: postings_length as usize,
                    short,
                }),
            }
        }
        let words = n_docs as usize / u64::BITS as usize + 1;
        let _bits = reserve_exact(budget, words * size_of::<u64>(), "row bitset")?;
        let mut bits = vec![0u64; words];
        let mut rest = bodies.as_slice();
        while !rest.is_empty() {
            let (wave, tail) = rest.split_at(wave_len(rest));
            rest = tail;
            // Released once this wave is unioned, before the next is fetched.
            let _held = reserve_exact(
                budget,
                wave.iter().map(|body| body.length).sum(),
                "postings",
            )?;
            let refs: Vec<(usize, usize)> = wave
                .iter()
                .map(|body| (body.metadata_offset, body.length))
                .collect();
            let fetched = self.fetch_term_postings(&refs).await?;
            let fetched_bytes: usize = fetched.iter().map(|b| b.len()).sum();
            work.postings_bytes += fetched_bytes as u64;
            // One range per body, as a match's build counts.
            work.planned_ranges += wave.len() as u64;
            let forms: Vec<bool> = wave.iter().map(|body| body.short).collect();
            let col = col.clone();
            let (ored, ns) = run_on_pool(pool, "contains union", move || {
                timed_section(|| {
                    let mut scratch = [0u32; BLOCK_LEN];
                    for (bytes, short) in fetched.into_iter().zip(forms) {
                        let cursor =
                            TermCursor::for_body(bytes, short, &col, None, UNWEIGHTED, true)?;
                        // The bitset spans this blob's documents; a list
                        // reaching past them is a damaged blob, refused
                        // before it can index out of the bitset.
                        if cursor
                            .blocks
                            .last()
                            .is_some_and(|b| b.last_doc_id >= n_docs)
                        {
                            return Err(posting_past_documents());
                        }
                        or_cursor_into_bitset(&mut bits, &cursor, &mut scratch);
                    }
                    Ok(bits)
                })
            })
            .await
            .map_err(|_| FtsError::TaskDropped("contains union"))?;
            work.kernel_cpu_ns += ns;
            bits = ored?;
        }
        let doc_map = self.doc_map.clone();
        let (rows, ns) = run_on_pool(pool, "contains rows", move || {
            timed_section(|| rows_of(bits, &inline, n_docs, &doc_map))
        })
        .await
        .map_err(|_| FtsError::TaskDropped("contains rows"))?;
        work.kernel_cpu_ns += ns;
        rows
    }
}

/// A posting that names a document past the end of its blob.
fn posting_past_documents() -> FtsError {
    FtsError::Read(ReadError::Malformed(
        "fts posting names a document past the blob's end".into(),
    ))
}

/// The Parquet rows of the documents set in `bits` or listed in `inline`,
/// through `doc_map`.
fn rows_of(
    mut bits: Vec<u64>,
    inline: &[u32],
    n_docs: u32,
    doc_map: &DocMap,
) -> Result<RoaringBitmap, FtsError> {
    for &doc in inline {
        if doc >= n_docs {
            return Err(posting_past_documents());
        }
        bits[(doc / u64::BITS) as usize] |= 1u64 << (doc % u64::BITS);
    }
    let mut rows = RoaringBitmap::new();
    for (index, &word) in bits.iter().enumerate() {
        let mut rest = word;
        while rest != 0 {
            let doc = index as u32 * u64::BITS + rest.trailing_zeros();
            // The bitset's last word runs past the blob's last document; a
            // bit set there came from a damaged posting body.
            if doc >= n_docs {
                return Err(posting_past_documents());
            }
            rest &= rest - 1;
            rows.insert(doc_map.row_of(FtsDocId::new(doc)).get());
        }
    }
    Ok(rows)
}

/// Charge `admitted` values to `charge`, when the walk carries one. `false`,
/// with the refusal left in `refused`, once the budget says no: the walk
/// stops there.
fn charge_admitted(
    charge: &mut Option<ValueCharge>,
    admitted: usize,
    refused: &mut Option<FtsError>,
) -> bool {
    let Some(charge) = charge.as_mut() else {
        return true;
    };
    match charge.cover(admitted) {
        Ok(()) => true,
        Err(refusal) => {
            *refused = Some(refusal);
            false
        }
    }
}

/// The CPU half of [`FtsReader::expand_terms`] and
/// [`FtsReader::contains_rows`]: one pass over the fetched dictionary that
/// fills every pattern's collector, keeping what `keep` names. Runs on the
/// reader pool, so it takes owned inputs. With a `charge`, the values it
/// admits are charged to its budget as it goes (a refusal ends the walk
/// with `FtsError::OverBudget`), and the charge comes back with the
/// collectors for the caller to hold while it uses them.
fn walk_dictionary(
    dict_bytes: &[u8],
    column: &str,
    patterns: &[OwnedPattern],
    walks: &[Walk],
    fold: bool,
    max_terms: usize,
    allow_full_walk: bool,
    keep: Keep,
    mut charge: Option<ValueCharge>,
) -> Result<(Vec<Collected>, Option<ValueCharge>), FtsError> {
    let dict = FtsReader::open_dict(dict_bytes)?;
    let mut collected: Vec<Collected> = patterns.iter().map(|_| Collected::new()).collect();
    // Every key in the column's range starts with `<column>\x1F`; the
    // term is what follows. `for_each_prefix` only visits keys carrying
    // the prefix it was given, and every prefix below begins with that
    // column key, so the slice never runs past a key. A key that is not
    // UTF-8 ends the walk with an error (see `non_utf8_key`), as does a
    // refused charge.
    let term_start = make_key(column, "").len();
    let mut bad_key = false;
    let mut refused: Option<FtsError> = None;
    let mut admitted = 0usize;
    for ((slot, pattern), walk) in collected.iter_mut().zip(patterns).zip(walks) {
        if bad_key || refused.is_some() {
            break;
        }
        if *walk == Walk::Subtree {
            dict.for_each_prefix(&make_key(column, pattern.borrow().text()), |key, value| {
                let Ok(term) = str::from_utf8(&key[term_start..]) else {
                    bad_key = true;
                    return false;
                };
                if !slot.admit(term, value, false, keep, max_terms) {
                    return false;
                }
                admitted += 1;
                charge_admitted(&mut charge, admitted, &mut refused)
            });
        }
    }
    let mut full: Vec<usize> = (0..patterns.len())
        .filter(|&i| walks[i] == Walk::Full && allow_full_walk)
        .collect();
    if !full.is_empty() && !bad_key && refused.is_none() {
        dict.for_each_prefix(&make_key(column, ""), |key, value| {
            let Ok(term) = str::from_utf8(&key[term_start..]) else {
                bad_key = true;
                return false;
            };
            let view = if fold {
                fold_term(term)
            } else {
                Cow::Borrowed(term)
            };
            // A covered term proves its rows unless `ILIKE`'s dotted-I
            // exception applies. A term holding only the head of a match
            // that crosses a tokenizer cut is admitted too, as doubtful:
            // its row may match, and the caller checks the text.
            full.retain(|&i| {
                let pattern = patterns[i].borrow();
                let doubtful = if pattern.covers(&view) {
                    fold && pattern.doubtful_cover(term)
                } else if pattern.crosses_cut(term, &view) {
                    true
                } else {
                    return true;
                };
                let kept = collected[i].admit(term, value, doubtful, keep, max_terms);
                admitted += usize::from(kept);
                kept
            });
            // Stop once the budget refuses, or every full-walk pattern has
            // hit its cap.
            charge_admitted(&mut charge, admitted, &mut refused) && !full.is_empty()
        });
    }
    if let Some(refusal) = refused {
        return Err(refusal);
    }
    if bad_key {
        return Err(non_utf8_key());
    }
    Ok((collected, charge))
}

#[cfg(test)]
mod tests {
    use arrow::{
        array::{BooleanArray, Scalar, StringArray},
        compute::kernels::comparison::ilike,
    };
    use tokio::runtime::Runtime;

    use super::{
        super::test_util::{
            build_blob, build_standard_blob, build_standard_blob_with, build_standard_fold_blob,
        },
        *,
    };
    use crate::utils::terms::TermDictBuilder;

    /// Generous cap so a test never trips the too-many fallback by accident.
    const MAX_TERMS: usize = 64;

    /// A `body` column analyzed by `standard` plus the English stemmer.
    const STEMMED_BODY_JSON: &str =
        r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75,"stemmer":"english"}]"#;

    fn expand_all(
        r: &FtsReader,
        patterns: &[TermPattern<'_>],
        fold: bool,
        max_terms: usize,
    ) -> (Vec<Option<Vec<String>>>, MatchWork) {
        let rt = Runtime::new().expect("runtime");
        rt.block_on(r.expand_terms("body", patterns, fold, max_terms, true, None))
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
        // Two open-left tokens, a prefix and an exact token: one dictionary
        // fetch, each slot exactly what the single-pattern calls return.
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
        assert_eq!(
            work.planned_ranges, 1,
            "one dictionary fetch for the whole leaf"
        );
    }

    #[test]
    fn a_disallowed_full_walk_leaves_the_pattern_unanswered() {
        // The caller judged the whole-column walk dearer than the scan: the
        // suffix and infix tokens come back `None`, the prefix token still
        // walks its subtree, the exact token is untouched, and the term
        // dictionary is still fetched once for the subtree walk.
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let rt = Runtime::new().expect("runtime");
        let (out, work) = rt
            .block_on(r.expand_terms(
                "body",
                &[
                    TermPattern::Contains("o"),
                    TermPattern::Suffix("me"),
                    TermPattern::Prefix("ja"),
                    TermPattern::Exact("rust"),
                ],
                false,
                MAX_TERMS,
                false,
                None,
            ))
            .expect("expand_terms");
        assert_eq!(out, vec![None, None, owned(&["java"]), owned(&["rust"])]);
        assert_eq!(work.planned_ranges, 1);
        // Nothing but full-walk patterns: no dictionary fetch at all.
        let (out, work) = rt
            .block_on(r.expand_terms(
                "body",
                &[TermPattern::Contains("o")],
                false,
                MAX_TERMS,
                false,
                None,
            ))
            .expect("expand_terms");
        assert_eq!(out, vec![None]);
        assert_eq!(work.planned_ranges, 0);
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

    /// Characters planted ahead of the word in the cut fixtures: one short
    /// of the tokenizer's cut, so the word's first character ends the
    /// first piece.
    const HEAD_RUN: usize = MAX_TOKEN_CHARS - 1;

    #[test]
    fn a_match_crossing_a_tokenizer_cut_is_found_through_the_piece_before_it() {
        // `x…xBBC` (254 x's) is indexed as `x…xb` (the cut length) and
        // `bc`, so no term holds `bbc`. The piece before the cut ends with
        // `b`, a head of `bbc`, which is how an open-left token still
        // finds the row. `x…xbb cz` (253 x's) is a near miss the dictionary
        // cannot tell apart — its 255-character word ends with `bb` — and
        // is admitted too; a caller verifies what it keeps.
        let cut = format!("{}BBC", "x".repeat(HEAD_RUN));
        let near = format!("{}bb cz", "x".repeat(HEAD_RUN - 1));
        let (blob, json) = build_standard_blob(&[&cut, &near, "abbcd", "zzz"]);
        let r = FtsReader::open(blob, &json).expect("open");
        let head = format!("{}b", "x".repeat(HEAD_RUN));
        let near_head = format!("{}bb", "x".repeat(HEAD_RUN - 1));
        // Lex order: `abbcd`, then the 253-x head (`b` sorts before `x`).
        assert_eq!(
            expand(&r, TermPattern::Contains("bbc"), MAX_TERMS),
            Some(vec!["abbcd".to_owned(), near_head.clone(), head.clone()])
        );
        assert_eq!(
            expand(&r, TermPattern::Suffix("bbc"), MAX_TERMS),
            Some(vec![near_head.clone(), head.clone()])
        );
        assert_eq!(
            expand_fold(&r, TermPattern::Contains("bbc")),
            Some(vec!["abbcd".to_owned(), near_head, head])
        );
        // A token closed on the left starts a word, which the first cut
        // never splits: no piece is admitted for it.
        assert_eq!(
            expand(&r, TermPattern::Prefix("bbc"), MAX_TERMS),
            Some(Vec::new())
        );
    }

    #[test]
    fn straddles_cut_needs_the_cut_length_and_a_proper_head_of_the_text() {
        let head = format!("{}b", "x".repeat(HEAD_RUN));
        assert!(straddles_cut(&head, &head, "bbc"));
        // The whole text at the end is a plain match, not a crossing one.
        let whole = format!("{}bbc", "x".repeat(MAX_TOKEN_CHARS - "bbc".len()));
        assert!(!straddles_cut(&whole, &whole, "bbc"));
        // Too short to be a cut piece.
        assert!(!straddles_cut("xb", "xb", "bbc"));
        // One character past the cut length is not a piece either (a
        // dictionary may still hold such a term).
        let long = format!("{}b", "x".repeat(MAX_TOKEN_CHARS));
        assert!(!straddles_cut(&long, &long, "bbc"));
        // Counted in characters: 254 two-byte letters and a `b`.
        let wide = format!("{}b", "é".repeat(HEAD_RUN));
        assert!(straddles_cut(&wide, &wide, "bbc"));
        // A one-character text has no proper head.
        assert!(!straddles_cut(&head, &head, "b"));
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
        assert_eq!(walked.planned_ranges, 1, "one dictionary fetch per walk");
        let (_, exact) = expand_all(&r, &[TermPattern::Exact("rust")], false, MAX_TERMS);
        assert_eq!(exact.planned_ranges, 0, "no dictionary needed");
    }

    /// Rows of the contains fixture carrying the dense term, enough for
    /// several long-form posting blocks.
    const DENSE_ROWS: usize = 3 * BLOCK_LEN + 7;

    /// Every third fixture row carries `BBC`, so its postings are long
    /// form too.
    const BBC_EVERY: usize = 3;

    /// The contains fixture: many rows of a dense term, `BBC` in a
    /// multi-block term and in df=1 terms, and the planted spellings the
    /// exact rule turns on — a match across a tokenizer cut and a near
    /// miss beside one, a dotted capital I whole and at a cut, the long s
    /// and the Kelvin sign.
    fn contains_docs() -> Vec<String> {
        let head = "x".repeat(HEAD_RUN);
        let mut docs: Vec<String> = (0..DENSE_ROWS)
            .map(|i| match i % BBC_EVERY {
                0 => format!("common BBC News {i}"),
                _ => format!("common filler {i}"),
            })
            .collect();
        docs.extend(
            [
                format!("{head}BBC"),
                format!("{}bb cz", "x".repeat(HEAD_RUN - 1)),
                format!("{head}\u{130}"),
                "TAX\u{130} rank".to_owned(),
                "TAXI rank".to_owned(),
                "the bbc's abbcd".to_owned(),
                "\u{17F}un \u{212A}elvin".to_owned(),
                String::new(),
            ]
            .iter()
            .cloned(),
        );
        docs
    }

    /// Needles the fixture is asked for.
    const NEEDLES: &[&str] = &[
        "bbc", "xi", "taxi", "sun", "kelvin", "common", "news", "zzq", "b",
    ];

    /// The rows Arrow's own `ILIKE '%needle%'` matches in `docs`.
    fn arrow_ilike(docs: &[String], needle: &str) -> RoaringBitmap {
        let haystack = StringArray::from_iter_values(docs.iter());
        let pattern = Scalar::new(StringArray::from(vec![format!("%{needle}%")]));
        let matched: BooleanArray = ilike(&haystack, &pattern).expect("ilike");
        (0..docs.len() as u32)
            .filter(|&row| matched.value(row as usize))
            .collect()
    }

    fn contains_all(r: &FtsReader, needles: &[&str]) -> (Vec<ContainsRows>, MatchWork) {
        let rt = Runtime::new().expect("runtime");
        rt.block_on(r.contains_rows("body", needles, None, None))
            .expect("contains_rows")
    }

    fn contains(r: &FtsReader, needle: &str) -> (ContainsRows, MatchWork) {
        let (mut rows, work) = contains_all(r, &[needle]);
        (rows.remove(0), work)
    }

    /// The kernel's contract, held to Arrow: a proven row matches, a
    /// matching row is proven or doubtful, and the two never overlap.
    fn assert_bracketed(rows: &ContainsRows, oracle: &RoaringBitmap, context: &str) {
        assert!(
            rows.proven.is_subset(oracle),
            "{context}: proven rows {:?} that do not match",
            &rows.proven - oracle
        );
        let admitted = &rows.proven | &rows.doubtful;
        assert!(
            oracle.is_subset(&admitted),
            "{context}: matching rows {:?} neither proven nor doubtful",
            oracle - &admitted
        );
        assert!(rows.proven.is_disjoint(&rows.doubtful), "{context}");
    }

    #[test]
    fn contains_rows_brackets_arrows_ilike() {
        let docs = contains_docs();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let planted = DENSE_ROWS as u32;
        let (cut_row, near_row, dotted_cut_row, taxi_dotted, taxi, _, fold_row) = (
            planted,
            planted + 1,
            planted + 2,
            planted + 3,
            planted + 4,
            planted + 5,
            planted + 6,
        );
        let (blob, json) = build_standard_blob(&refs);
        let r = FtsReader::open(blob, &json).expect("open");
        for needle in NEEDLES {
            let (rows, work) = contains(&r, needle);
            let oracle = arrow_ilike(&docs, needle);
            assert_bracketed(&rows, &oracle, needle);
            assert!(work.planned_ranges >= 1, "the dictionary fetch is counted");
        }
        // The planted spellings land where the rule says.
        let (bbc, _) = contains(&r, "bbc");
        assert_eq!(
            bbc.doubtful,
            RoaringBitmap::from_iter([near_row, cut_row]),
            "the cut match and the near miss beside a cut, nothing else"
        );
        assert!(arrow_ilike(&docs, "bbc").contains(cut_row));
        assert!(!arrow_ilike(&docs, "bbc").contains(near_row));
        let (xi, _) = contains(&r, "xi");
        assert!(xi.doubtful.contains(dotted_cut_row), "`İ` at a cut");
        assert!(!arrow_ilike(&docs, "xi").contains(dotted_cut_row));
        let (taxi_rows, _) = contains(&r, "taxi");
        assert!(taxi_rows.proven.contains(taxi));
        assert!(taxi_rows.doubtful.contains(taxi_dotted), "`İ` whole");
        assert!(!arrow_ilike(&docs, "taxi").contains(taxi_dotted));
        for needle in ["sun", "kelvin"] {
            let (folded, _) = contains(&r, needle);
            assert!(folded.proven.contains(fold_row), "{needle}");
        }
        // A needle no cut or dotted I can touch is decided outright.
        for needle in ["common", "news", "zzq"] {
            let (clean, _) = contains(&r, needle);
            assert!(clean.doubtful.is_empty(), "{needle}");
            assert_eq!(clean.proven, arrow_ilike(&docs, needle), "{needle}");
        }
    }

    #[test]
    fn contains_rows_names_parquet_rows_on_a_blob_that_reorders_its_documents() {
        // Blob position `i` holds row `order[i]`, as a reordering merge
        // writes it. The postings are in blob positions; the answer must
        // be in rows.
        let docs = contains_docs();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let n = docs.len() as u32;
        let order: Vec<u32> = (0..n).rev().collect();
        let (blob, json) = build_standard_blob_with(&refs, Some(&order));
        let r = FtsReader::open(blob, &json).expect("open");
        assert!(r.has_doc_map(), "the fixture must reorder to mean anything");
        for needle in NEEDLES {
            let (rows, _) = contains(&r, needle);
            let oracle = arrow_ilike(&docs, needle);
            assert_bracketed(&rows, &oracle, needle);
        }
        let (common, _) = contains(&r, "common");
        assert_eq!(common.proven, arrow_ilike(&docs, "common"));
    }

    #[test]
    fn several_needles_share_one_walk_and_answer_as_each_alone() {
        // The needles of an `OR` on one column: one dictionary fetch for
        // all of them, and each needle's rows exactly what it gets on its
        // own — a repeated needle included.
        let docs = contains_docs();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let (blob, json) = build_standard_blob(&refs);
        let r = FtsReader::open(blob, &json).expect("open");
        let needles = ["bbc", "xi", "taxi", "bbc", "zzq"];
        let (together, work) = contains_all(&r, &needles);
        assert_eq!(together.len(), needles.len());
        // Alone, each needle pays one dictionary range plus its postings'.
        let mut posting_ranges = 0;
        for (needle, rows) in needles.iter().zip(&together) {
            let (alone, alone_work) = contains(&r, needle);
            assert_eq!(rows, &alone, "{needle}");
            posting_ranges += alone_work.planned_ranges - 1;
        }
        assert_eq!(
            work.planned_ranges,
            1 + posting_ranges,
            "one dictionary range for the whole set"
        );
        let (none, work) = contains_all(&r, &[]);
        assert!(none.is_empty());
        assert_eq!(work.planned_ranges, 0, "no needle, no fetch");
    }

    #[test]
    fn contains_rows_refuses_a_column_indexed_by_another_analyzer() {
        let (blob, _) = build_blob();
        let r = FtsReader::open(blob, STEMMED_BODY_JSON).expect("open");
        let rt = Runtime::new().expect("runtime");
        let err = rt
            .block_on(r.contains_rows("body", &["rust"], None, None))
            .expect_err("stemmed column");
        assert!(matches!(err, FtsError::ExactNeedsStandard { .. }), "{err}");
        let err = rt
            .block_on(r.contains_rows("nope", &["rust"], None, None))
            .expect_err("unknown column");
        assert!(matches!(err, FtsError::UnknownColumn(_)));
    }

    /// A byte UTF-8 never contains, to end a dictionary key with.
    const NOT_UTF8_BYTE: u8 = 0xFF;

    #[test]
    fn a_non_utf8_dictionary_key_fails_every_walk_rather_than_skip_its_rows() {
        // `body`'s terms: `rust`, and one that is `r` plus a byte no UTF-8
        // holds. Skipping the second could drop the rows it indexes.
        let mut dict = TermDictBuilder::new();
        dict.insert(
            &make_key("body", "rust"),
            DictEntry::Inline { doc_id: 0, tf: 1 },
        );
        let mut bad = make_key("body", "r");
        bad.push(NOT_UTF8_BYTE);
        dict.insert(&bad, DictEntry::Inline { doc_id: 1, tf: 1 });
        let dict_bytes = dict.finish();
        let walks = [
            // The LIKE expansion's shared walk and a prefix's subtree walk.
            (OwnedPattern::Contains("us".into()), Walk::Full, Keep::Terms),
            (OwnedPattern::Prefix("r".into()), Walk::Subtree, Keep::Terms),
            // The exact path's walk.
            (
                OwnedPattern::Contains("us".into()),
                Walk::Full,
                Keep::Values,
            ),
        ];
        for (pattern, walk, keep) in walks {
            let err = walk_dictionary(
                &dict_bytes,
                "body",
                &[pattern],
                &[walk],
                true,
                usize::MAX,
                true,
                keep,
                None,
            )
            .err()
            .unwrap_or_else(|| panic!("{walk:?} / {keep:?} skipped the key"));
            assert!(
                matches!(err, FtsError::Read(ReadError::Malformed(_))),
                "{walk:?}: {err}"
            );
        }
    }

    #[test]
    fn a_value_charge_grows_a_step_ahead_and_refuses_past_the_budget() {
        let step_bytes = VALUE_CHARGE_STEP * size_of::<DictEntry>();
        let measured = ConnectionMemoryBudget::measured();
        let mut charge = ValueCharge::new(&measured).expect("measured");
        charge.cover(1).expect("measured");
        assert_eq!(
            measured.used_bytes(),
            step_bytes,
            "one step covers one value"
        );
        charge.cover(VALUE_CHARGE_STEP).expect("measured");
        assert_eq!(measured.used_bytes(), step_bytes, "and a whole step");
        charge.cover(VALUE_CHARGE_STEP + 1).expect("measured");
        assert_eq!(measured.used_bytes(), 2 * step_bytes);
        drop(charge);
        assert_eq!(measured.used_bytes(), 0, "released with the charge");
        // A budget below one step refuses the first value.
        let bounded = ConnectionMemoryBudget::with_limit(step_bytes as u64);
        let mut charge = ValueCharge::new(&bounded).expect("an empty charge fits");
        let err = charge.cover(1).expect_err("a step is past 90% of one step");
        assert!(matches!(err, FtsError::OverBudget(_)), "{err}");
    }

    #[test]
    fn a_charged_walk_holds_its_values_and_stops_when_refused() {
        let docs = contains_docs();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let (blob, json) = build_standard_blob(&refs);
        let r = FtsReader::open(blob, &json).expect("open");
        let rt = Runtime::new().expect("runtime");
        let dict_bytes = rt.block_on(r.dict_bytes_async()).expect("dictionary");
        let walk = |charge: Option<ValueCharge>| {
            walk_dictionary(
                &dict_bytes,
                "body",
                &[OwnedPattern::Contains("common".into())],
                &[Walk::Full],
                true,
                usize::MAX,
                true,
                Keep::Values,
                charge,
            )
        };
        let measured = ConnectionMemoryBudget::measured();
        let (collected, charge) = walk(Some(ValueCharge::new(&measured).expect("measured")))
            .expect("a measured budget never refuses");
        let kept = collected[0].proven.len() + collected[0].doubtful.len();
        let charge = charge.expect("the charge comes back");
        assert!(
            kept > 0 && charge.covered >= kept,
            "{} < {kept}",
            charge.covered
        );
        assert!(measured.used_bytes() > 0, "held while the caller unions");
        drop(charge);
        assert_eq!(measured.used_bytes(), 0);
        // Below one step, the first value admitted ends the walk.
        let step_bytes = (VALUE_CHARGE_STEP * size_of::<DictEntry>()) as u64;
        let bounded = ConnectionMemoryBudget::with_limit(step_bytes);
        let err = walk(Some(ValueCharge::new(&bounded).expect("empty")))
            .err()
            .expect("refused part-way through the walk");
        assert!(matches!(err, FtsError::OverBudget(_)), "{err}");
    }

    /// A budget smaller than one document bitset of the contains fixture.
    const TOO_SMALL_FOR_A_BITSET: u64 = 1;

    #[test]
    fn contains_rows_charges_the_budget_and_refuses_past_it() {
        let docs = contains_docs();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let (blob, json) = build_standard_blob(&refs);
        let r = FtsReader::open(blob, &json).expect("open");
        let rt = Runtime::new().expect("runtime");
        // Measured: charged while held, all of it released on return.
        let measured = ConnectionMemoryBudget::measured();
        rt.block_on(r.contains_rows("body", &["common"], None, Some(&measured)))
            .expect("a measured budget never refuses");
        let bitset = (docs.len() / u64::BITS as usize + 1) * size_of::<u64>();
        assert!(
            measured.peak() >= bitset,
            "the document bitset is charged: peak {} < {bitset}",
            measured.peak()
        );
        assert_eq!(measured.used_bytes(), 0, "and released on return");
        // Bounded below one bitset: refused, as a budget refusal.
        let bounded = ConnectionMemoryBudget::with_limit(TOO_SMALL_FOR_A_BITSET);
        let err = rt
            .block_on(r.contains_rows("body", &["common"], None, Some(&bounded)))
            .expect_err("over the budget");
        assert!(matches!(err, FtsError::OverBudget(_)), "{err}");
        assert!(
            ReadError::from(err).over_budget().is_some(),
            "reads as a budget refusal through the read error"
        );
        assert!(bounded.denials() > 0);
        assert_eq!(bounded.used_bytes(), 0, "a refusal holds nothing");
    }

    #[test]
    fn a_fetch_wave_stops_at_the_byte_budget_and_always_takes_one_body() {
        let body = |length: usize| TermBody {
            metadata_offset: 0,
            length,
            short: false,
        };
        let half = CONTAINS_FETCH_BATCH_BYTES / 2;
        let bodies = [body(half), body(half), body(1)];
        assert_eq!(wave_len(&bodies), 2, "exactly the budget fits");
        assert_eq!(wave_len(&bodies[2..]), 1);
        let oversized = [body(CONTAINS_FETCH_BATCH_BYTES + 1), body(1)];
        assert_eq!(
            wave_len(&oversized),
            1,
            "one body over the budget goes alone"
        );
    }

    #[test]
    fn unknown_column_errors_like_a_match_would() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let rt = Runtime::new().expect("runtime");
        let err = rt
            .block_on(r.expand_terms(
                "nope",
                &[TermPattern::Prefix("ru")],
                false,
                MAX_TERMS,
                true,
                None,
            ))
            .expect_err("unknown column");
        assert!(matches!(err, FtsError::UnknownColumn(_)));
    }
}
