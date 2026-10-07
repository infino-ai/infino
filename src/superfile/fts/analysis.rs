// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The per-column analysis chain: the `standard` tokenizer plus optional
//! stopword removal and stemming, in that order.
//!
//! ## Shape
//!
//! Here *tokenizer* means the thing that splits text and *analyzer*
//! means the whole chain: that tokenizer plus its filters.
//!
//! The base is always [`StandardTokenizer`], followed by up to two token
//! filters:
//! stopwords are removed from the *unstemmed* token, then what
//! survives is stemmed. The order is not configurable and is baked
//! into the chain's name, so there is exactly one canonical spelling
//! per chain.
//!
//! Stopwords have to precede stemming because a stopword list is
//! written in surface forms — `are`, `their`, `these` — and stemming
//! first would leave the list matching neither those nor their stems
//! reliably. It is also the order every filter-chain analyzer applies
//! them in.
//!
//! ## Every filter here is opt-in, and the default is untouched
//!
//! A column that declares no filter is not wrapped at all
//! ([`chain_tokenizer`] hands back the bare base tokenizer), keeps its
//! plain tokenizer name, and takes the same monomorphized ingest scan
//! it always did. So the default column is byte-identical and
//! code-path-identical to one declared before chains existed.
//!
//! That matters because the default is the parity surface, and it is
//! not this module's to move. `standard` — tokenize and lowercase,
//! no stopwords, no stemming — is what Lucene's default analyzer does:
//! `IndexWriterConfig` defaults to `StandardAnalyzer`, whose no-arg
//! constructor is documented as "builds an analyzer with no stop
//! words" and which never stemmed. Anything that closes a remaining
//! gap in `standard` itself belongs to the tokenizer and the scorer,
//! not here; this module only adds filters a caller asks for by name.
//!
//! ## Persistence: two additive fields, and a derived identity
//!
//! A column persists, only when set, a `stopwords` and a `stemmer`
//! field; the base is not recorded because there is only one. A
//! `tokenizer` name recorded by an earlier writer is still checked on
//! read ([`check_recorded_tokenizer`]): absent or `standard` opens, any
//! other name refuses the table. Both are ordinary additive
//! fields: absent means the filter is off, which is the one thing a
//! file written before the filter existed can mean, so a current
//! reader infers the right analysis from an old file with no special
//! handling.
//!
//! The chain also has a **name** — `"standard+stop=english+stem=english"`
//! from [`chain_name`] — but it is derived in memory and never written.
//! It exists because three places have to reason about a column's
//! analysis as one value, and reading three fields at each of them
//! would be three chances to forget one:
//!
//!   - the `LIKE` prune lowering, which recognizes analyzers by
//!     [`Tokenizer::name`] and must decline to bound a column whose
//!     terms are not substring-preserving images of its text;
//!   - the merge carry check, which compares these identities to refuse
//!     merging differently-analyzed superfiles;
//!   - the table's options-hash, which needs analysis in its identity.
//!
//! Keeping the name out of the format is what makes a *rollback* a
//! degradation rather than an outage. An engine predating a filter
//! ignores its field and analyzes the column as unfiltered — wrong
//! answers on that column until it rolls forward, recoverable — where a
//! composite name it could not parse would make the table refuse to
//! open. The precedent is deliberate: the same trade decided against
//! marking these files with a new FTS section version, on the grounds
//! that a lost-recall regression is recoverable and an unopenable table
//! is an outage.
//!
//! What is *not* ignorable is an unrecognized **value** of a field the
//! reader does know: `"stopwords":"german"` on an engine that ships no
//! German list is refused, because there is no sound way to proceed —
//! the analysis cannot be reproduced. Unknown field, degrade; unknown
//! value, error.
//!
//! ## Positions
//!
//! Stopword removal leaves a **hole**: the position ordinal a removed
//! token would have occupied is skipped, so the tokens on either side
//! of it are not adjacent. Without that, `"new york"` would phrase-match
//! `"new the york"`. Holes reach the index through
//! [`Tokenizer::tokenize_each_positioned`] and the query through
//! [`ParsedQuery`]'s per-phrase offsets.
//!
//! Doc length counts the tokens the chain *emits*, which is what Lucene
//! counts (`IndexingChain.PerField.invert` bumps `invertState.length`
//! inside the `incrementToken` loop, so a token `StopFilter` dropped
//! never reaches it) and what the carried postings of a merge preserve.

use std::{any::Any, borrow::Cow, sync::Arc};

use rust_stemmers::{Algorithm, Stemmer as Snowball};

use super::tokenize::{STANDARD_TOKENIZER, StandardTokenizer, Tokenizer};
use crate::superfile::error::ReadError;

/// Stopword set applied to a column, after the base tokenizer and
/// before the stemmer.
///
/// Named built-ins only. A user-supplied word list would have to
/// persist beside the filter name, which is exactly the
/// additive-and-ignorable shape the composite name exists to avoid; the
/// extension path stays open as a future `+stop=custom` name whose
/// sibling word list an older engine rejects on the unknown name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
// A second language is the obvious next addition, and adding an enum
// variant is a breaking change unless the enum says otherwise. Callers
// only ever *construct* these, never match on them, so the attribute
// costs nothing and buys every later set a patch release.
#[non_exhaustive]
pub enum Stopwords {
    /// Keep every token (the default).
    #[default]
    None,
    /// Lucene's `EnglishAnalyzer.ENGLISH_STOP_WORDS_SET` — the 33-word
    /// Snowball English list: `a an and are as at be but by for if in
    /// into is it no not of on or such that the their then there these
    /// they this to was will with`.
    English,
}

/// Stemmer applied to a column, after stopword removal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// `#[non_exhaustive]` for the same reason as [`Stopwords`].
#[non_exhaustive]
pub enum Stemmer {
    /// Index tokens as written (the default).
    #[default]
    None,
    /// Snowball English (Porter2), via `rust-stemmers`.
    ///
    /// Named `english` rather than `porter` in both the API and the
    /// persisted name because the two are different algorithms:
    /// Snowball's `english` *is* Porter2, whereas Lucene's
    /// `PorterStemFilter` is the original Porter. Spelling this
    /// `porter` would name the wrong one.
    English,
}

/// Lucene's English stopword set (`EnglishAnalyzer.ENGLISH_STOP_WORDS_SET`),
/// **sorted** so [`Stopwords::contains`] can binary-search it.
///
/// ## Frozen: this list is on-disk format state, not a tunable
///
/// A column persists the *name* `stop=english`, never the words — so
/// this constant is what a reader reconstructs the column's analysis
/// from, and editing it re-analyzes every column already built with it.
/// Add a word and a query for it starts missing documents that contain
/// it; remove one and queries tokenize differently than the postings
/// were built. Both are wrong answers with no error, because the *name*
/// still matches — exactly the failure the composite name exists to
/// prevent, reached through the back door.
///
/// So it does not change. `english_stopword_set_is_frozen` pins every
/// word and fails loudly if one moves. If a different list is ever
/// genuinely wanted, it arrives as a **new name** (`stop=english2`)
/// resolved by a new arm in [`Stopwords::from_name`] beside this one,
/// leaving files built under the old name reconstructible from the old
/// list — never as an edit here.
///
/// Sorted-slice + binary search rather than a hash set: 33 short words
/// are searched in ~5 comparisons with no hashing and no lazy-init, and
/// the length pre-filter below rejects most corpus tokens before the
/// search runs at all.
pub(crate) const ENGLISH_STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// Longest word in [`ENGLISH_STOPWORDS`]. A token longer than this
/// cannot be a stopword, so one integer compare rejects the great
/// majority of corpus tokens before any string work.
const ENGLISH_STOPWORD_MAX_LEN: usize = 5;

impl Stopwords {
    /// This filter's analysis revision (see [`chain_revision`]).
    ///
    /// The shipped list has not changed since it was introduced, so a
    /// column filtered by it holds the same terms it always did.
    pub(crate) fn revision(self) -> u32 {
        match self {
            Self::None | Self::English => 0,
        }
    }

    /// Whether `token` — already lowercased by the base tokenizer, and
    /// not yet stemmed — is in this set.
    #[inline]
    fn contains(self, token: &str) -> bool {
        match self {
            Stopwords::None => false,
            Stopwords::English => {
                token.len() <= ENGLISH_STOPWORD_MAX_LEN
                    && ENGLISH_STOPWORDS.binary_search(&token).is_ok()
            }
        }
    }
}

/// The `standard` tokenizer's analysis revision (see [`chain_revision`]).
///
/// At 1: the token-length cap applies, every entry point folds through
/// one emitter, and emoji are emitted as tokens. A revision numbers a
/// chain's output, not the changes that produced it.
const STANDARD_REVISION: u32 = 1;

/// Accept the base-tokenizer name a column recorded, if any.
///
/// Only `standard` is reproducible. Aliasing another name to it would
/// query an index split one way with terms split another, so any other
/// name refuses the table instead.
pub(crate) fn check_recorded_tokenizer(
    column: &str,
    recorded: Option<&str>,
) -> Result<(), ReadError> {
    match recorded {
        None | Some(STANDARD_TOKENIZER) => Ok(()),
        Some(name) => Err(ReadError::RemovedAnalyzer {
            column: column.to_string(),
            analyzer: name.to_string(),
        }),
    }
}

/// Scan `text` with the base tokenizer into `(token, position)` pairs.
///
/// `standard` drops nothing — every segment carrying an alphanumeric is
/// emitted — so its emission ordinal is already the gap-inclusive
/// position. Monomorphized over `F` so the ingest path's interning
/// closure inlines into the scan loop.
#[inline]
fn scan_positioned<F: FnMut(&str, u64)>(text: &str, mut f: F) {
    let mut position = 0u64;
    StandardTokenizer.tokenize_each_inline(text, |tok| {
        f(tok, position);
        position += 1;
    });
}

/// The base tokenizer plus its stopword set and stemmer.
///
/// Constructed only through [`chain_tokenizer`],
/// so a `ChainTokenizer` always carries at least one active filter — a
/// chain with neither is the base tokenizer itself, under the base's own
/// plain name, and must not be wrapped (wrapping it would change the
/// reported name and give up the base's downcast fast path for nothing).
pub struct ChainTokenizer {
    stopwords: Stopwords,
    /// This chain's canonical composite name, from the static table in
    /// [`chain_name`] — which is what keeps [`Tokenizer::name`] able to
    /// return `&'static str`.
    name: &'static str,
    /// The Snowball stemmer, built once at construction.
    /// `None` for [`Stemmer::None`].
    snowball: Option<Snowball>,
}

// `rust_stemmers::Stemmer` holds a function pointer and no interior
// mutability, so the chain is as shareable as the trait requires. Assert
// it here rather than discovering it as an `impl Tokenizer` error.
static_assertions::assert_impl_all!(Snowball: Send, Sync);

impl std::fmt::Debug for ChainTokenizer {
    /// Hand-written because `rust_stemmers::Stemmer` is not `Debug`.
    /// Prints the canonical name, which determines every field.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainTokenizer")
            .field("name", &self.name)
            .finish()
    }
}

impl ChainTokenizer {
    /// Apply the filter chain to one base token: `None` when the
    /// stopword set removes it, otherwise the token to index (stemmed,
    /// if a stemmer is configured).
    ///
    /// Borrowed when no stemmer runs or the stemmer returns its input
    /// unchanged, which is the common case for already-stemmed forms.
    #[inline]
    fn filter<'t>(&self, token: &'t str) -> Option<Cow<'t, str>> {
        if self.stopwords.contains(token) {
            return None;
        }
        match &self.snowball {
            None => Some(Cow::Borrowed(token)),
            Some(s) => Some(s.stem(token)),
        }
    }

    /// Monomorphized positional scan: `(token, position)` per emitted
    /// token, where `position` is the gap-inclusive ordinal. A token the
    /// stopword filter removes advances the ordinal without emitting —
    /// the phrase hole.
    ///
    /// Same role as [`StandardTokenizer::tokenize_each_inline`], and
    /// reached the same way: the FTS build path downcasts through
    /// [`Tokenizer::as_any`] so the interning closure inlines into the
    /// scan instead of paying an indirect call per token.
    #[inline]
    pub fn tokenize_each_inline_positioned<F: FnMut(&str, u64)>(&self, text: &str, mut f: F) {
        scan_positioned(text, |tok, position| {
            if let Some(t) = self.filter(tok) {
                f(&t, position);
            }
        });
    }

    /// Monomorphized positionless scan — the same tokens
    /// [`Self::tokenize_each_inline_positioned`] emits, with the
    /// ordinal discarded, so the two can never disagree about which
    /// tokens exist.
    #[inline]
    pub fn tokenize_each_inline<F: FnMut(&str)>(&self, text: &str, mut f: F) {
        self.tokenize_each_inline_positioned(text, |tok, _position| f(tok));
    }
}

impl Tokenizer for ChainTokenizer {
    fn name(&self) -> &'static str {
        self.name
    }

    fn tokenize<'a>(&'a self, text: &'a str) -> Box<dyn Iterator<Item = String> + 'a> {
        let mut out = Vec::new();
        self.tokenize_each_inline(text, |t| out.push(t.to_owned()));
        Box::new(out.into_iter())
    }

    fn tokenize_each(&self, text: &str, f: &mut dyn FnMut(&str)) {
        self.tokenize_each_inline(text, |t| f(t));
    }

    fn tokenize_each_positioned(&self, text: &str, f: &mut dyn FnMut(&str, u64)) {
        self.tokenize_each_inline_positioned(text, |t, p| f(t, p));
    }

    /// Returns the chain itself, never the base it wraps. Two callers
    /// depend on that: the FTS build path's downcast, which must reach
    /// [`Self::tokenize_each_inline_positioned`] and not the base's
    /// unfiltered scan; and the `LIKE` lowering, which recognizes a
    /// column's analyzer by [`Tokenizer::name`] and must not mistake a
    /// stemming column for a plain one (see `is_plain_standard`).
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// The query side runs the same chain, so a term is looked up in the
    /// form it was indexed under. Always owned: a stemmed token is a new
    /// string, and the stopword filter makes the borrowed/owned split
    /// per token rather than per query, which is not worth a second scan
    /// implementation on query-length input.
    fn tokenize_each_query<'q>(&self, text: &'q str, f: &mut dyn FnMut(Cow<'q, str>)) {
        self.tokenize_each_inline(text, |t| f(Cow::Owned(t.to_owned())));
    }

    /// Overridden because the stopword filter drops tokens: a quoted
    /// phrase must carry the holes it left, or the phrase asks for
    /// words closer together than the index put them and matches
    /// nothing.
    fn tokenize_each_query_positioned<'q>(
        &self,
        text: &'q str,
        f: &mut dyn FnMut(Cow<'q, str>, u64),
    ) {
        self.tokenize_each_inline_positioned(text, |t, position| {
            f(Cow::Owned(t.to_owned()), position)
        });
    }
}

/// A chain's identity as one string — `standard` when no filter is
/// active, otherwise `standard` followed by its filters.
///
/// **Derived, never persisted.** It is what [`Tokenizer::name`] reports,
/// so the three places that reason about a column's analysis as a single
/// value get it from one place (see the module docs). The format stores
/// the components as separate fields instead, and nothing parses this
/// back — a chain is only ever built from its components.
///
/// Every reachable string is a `&'static str` here because the built-in
/// sets are a closed set: two stopword settings × two stemmer settings. Component order is fixed (`stop` before `stem`,
/// matching the order the filters run), so one chain has exactly one
/// spelling and the options-hash is stable.
pub(crate) fn chain_name(stopwords: Stopwords, stemmer: Stemmer) -> &'static str {
    match (stopwords, stemmer) {
        (Stopwords::None, Stemmer::None) => STANDARD_TOKENIZER,
        (Stopwords::English, Stemmer::None) => "standard+stop=english",
        (Stopwords::None, Stemmer::English) => "standard+stem=english",
        (Stopwords::English, Stemmer::English) => "standard+stop=english+stem=english",
    }
}

/// The revision of the terms a chain emits.
///
/// A column's analysis name says *which* analysis produced its postings;
/// it cannot say which **version** of that analysis, because a name does
/// not change when the tokens behind it do. Two files can name
/// `standard` and hold different terms for the same text, and nothing in
/// the file distinguishes them — so a query analyzed by today's chain can
/// look up a term an older index never wrote, and match nothing.
///
/// This number closes that gap: it is stamped per column and compared
/// rather than inferred. Bump the component that actually moved whenever a
/// change can alter the tokens a chain emits for any input — a new
/// boundary rule, a different fold, a cap, a filter's word list. A change
/// that cannot alter output (a faster path over identical tokens) leaves it
/// alone.
///
/// The chain's revision is the sum of its parts', so each part moves it
/// independently — see [`combine_revisions`].
pub(crate) fn chain_revision(stopwords: Stopwords, stemmer: Stemmer) -> u32 {
    combine_revisions(STANDARD_REVISION, stopwords.revision(), stemmer.revision())
}

/// The revision to credit terms whose own is unknown.
///
/// The oldest, so a column recording no revision can never read as
/// current on the strength of an assumption.
pub(crate) const UNKNOWN_ANALYSIS_REVISION: u32 = 0;

/// Fold a chain's part revisions into the one number the format stores.
///
/// Summing rather than taking the maximum is what makes a filter bump
/// visible: under a maximum, a stemmer moving from 0 to 1 is swallowed by
/// a base already at 1, so the chain keeps its old revision and every file
/// analyzed by the old stemmer reads as current.
fn combine_revisions(base: u32, stopwords: u32, stemmer: u32) -> u32 {
    base + stopwords + stemmer
}

/// Build the tokenizer for a chain: the bare [`StandardTokenizer`] when
/// no filter is active, otherwise a [`ChainTokenizer`].
///
/// A filterless chain must *not* be wrapped — the wrapper would report a
/// composite name no plain column has and would hide the base from the
/// build path's downcast, costing ingest throughput for no behaviour
/// change.
pub(crate) fn chain_tokenizer(stopwords: Stopwords, stemmer: Stemmer) -> Arc<dyn Tokenizer> {
    if stopwords == Stopwords::None && stemmer == Stemmer::None {
        return Arc::new(StandardTokenizer);
    }
    Arc::new(ChainTokenizer {
        stopwords,
        name: chain_name(stopwords, stemmer),
        snowball: match stemmer {
            Stemmer::None => None,
            Stemmer::English => Some(Snowball::create(Algorithm::English)),
        },
    })
}

impl Stopwords {
    /// The name this set persists under, or `None` for
    /// [`Stopwords::None`] — which is written by omitting the field.
    pub(crate) fn as_str(self) -> Option<&'static str> {
        match self {
            Stopwords::None => None,
            Stopwords::English => Some("english"),
        }
    }

    /// Resolve a set by name — `"english"` — or `None` for a name this
    /// engine cannot reproduce.
    ///
    /// **Exact match, and the single source of truth for the spelling.**
    /// Every caller turns `None` into an error rather than a silent
    /// fallback: analyzing with a set we do not have is not a degraded
    /// index, it is a different one. Public because the language
    /// bindings take this name as a string from their callers and must
    /// resolve it the same way the format does — three private copies of
    /// the same match is how they drift, and one of them case-folded
    /// while the others did not.
    ///
    /// Case-folding is deliberately absent: accepting more spellings
    /// later is additive, tightening later is not.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "english" => Some(Stopwords::English),
            _ => None,
        }
    }
}

impl Stemmer {
    /// This filter's analysis revision (see [`chain_revision`]).
    ///
    /// The algorithm is Snowball English via `rust-stemmers`, unchanged
    /// since it was introduced.
    pub(crate) fn revision(self) -> u32 {
        match self {
            Self::None | Self::English => 0,
        }
    }

    /// The name this stemmer persists under; see
    /// [`Stopwords::as_str`].
    pub(crate) fn as_str(self) -> Option<&'static str> {
        match self {
            Stemmer::None => None,
            Stemmer::English => Some("english"),
        }
    }

    /// Resolve a stemmer by name — `"english"`; same exact-match rule and
    /// same rationale as [`Stopwords::from_name`].
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "english" => Some(Stemmer::English),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the chain whose derived name is `name`. A test lookup, not
    /// a parser — nothing in the engine turns a name back into
    /// components, because the format stores the components.
    fn chain(name: &str) -> Arc<dyn Tokenizer> {
        for stop in [Stopwords::None, Stopwords::English] {
            for stem in [Stemmer::None, Stemmer::English] {
                if chain_name(stop, stem) == name {
                    return chain_tokenizer(stop, stem);
                }
            }
        }
        panic!("no chain has the name {name:?}");
    }

    fn tokens(name: &str, text: &str) -> Vec<String> {
        chain(name).tokenize(text).collect()
    }

    fn positioned(name: &str, text: &str) -> Vec<(String, u64)> {
        let tok = chain(name);
        let mut out = Vec::new();
        tok.tokenize_each_positioned(text, &mut |t, p| out.push((t.to_owned(), p)));
        out
    }

    /// The exact set a column declaring `stop=english` is analyzed
    /// with, spelled out so a change to [`ENGLISH_STOPWORDS`] has to
    /// come here and be argued for.
    ///
    /// This is not a restatement of the constant for its own sake. The
    /// name `stop=english` is what reaches disk, so the words behind it
    /// are format state: change them and every column already built
    /// under that name is analyzed one way and queried another, with no
    /// error, because the name still matches.
    const FROZEN_ENGLISH_STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is",
        "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there",
        "these", "they", "this", "to", "was", "will", "with",
    ];

    /// If this fails, do **not** update the expectation to match the
    /// code. The words behind `stop=english` are on-disk format state
    /// (see [`ENGLISH_STOPWORDS`]): every column already built under
    /// that name would be silently analyzed one way and queried
    /// another. A different list arrives as a new name —
    /// `stop=english2`, with its own `from_name` arm and its own frozen
    /// list — so files under the old name stay reconstructible.
    #[test]
    fn english_stopword_set_is_frozen() {
        assert_eq!(
            ENGLISH_STOPWORDS, FROZEN_ENGLISH_STOPWORDS,
            "the `stop=english` word list changed. This is a format \
             change, not a tuning change: columns already built under \
             that name would be analyzed with the old list and queried \
             with the new one, silently. Ship a new name instead."
        );
    }

    /// The stemmer has the same problem one layer out, and worse: the
    /// rules behind `stem=english` live in `rust-stemmers`, whose
    /// version range admits any `1.x`, so a routine dependency bump
    /// could change how an existing column's postings *would* be
    /// tokenized while the postings themselves keep the old forms.
    ///
    /// These pairs are the observable contract. If a bump breaks this,
    /// the bump is a format change — reject it or ship `stem=english2`;
    /// do not re-record the expectations.
    #[test]
    fn english_stemmer_output_is_frozen() {
        // Chosen to cover the Porter2 steps that actually differ
        // between implementations: -ing/-ed removal with and without
        // stem doubling, -ies/-y, -ational/-ate, -ness/-ful/-ment
        // suffixes, short-word protection, and the irregulars it has no
        // rule for.
        let cases = [
            ("running", "run"),
            ("runs", "run"),
            ("run", "run"),
            ("runner", "runner"),
            ("ran", "ran"),
            ("studies", "studi"),
            ("study", "studi"),
            ("cities", "citi"),
            ("relational", "relat"),
            ("hopping", "hop"),
            ("hoping", "hope"),
            ("happiness", "happi"),
            ("hopeful", "hope"),
            ("argument", "argument"),
            ("agreed", "agre"),
            ("news", "news"),
            ("sky", "sky"),
            ("is", "is"),
        ];
        let tok = chain("standard+stem=english");
        for (word, want) in cases {
            let got: Vec<String> = tok.tokenize(word).collect();
            assert_eq!(
                got,
                vec![want.to_string()],
                "`stem=english` changed for {word:?}. This is a format \
                 change: columns already built under that name hold the \
                 old stems and would now be queried with new ones. \
                 Reject the bump or ship a new name."
            );
        }
    }

    #[test]
    fn english_stopwords_are_sorted_and_within_the_length_bound() {
        // `contains` binary-searches the slice behind a length
        // pre-filter, so both invariants are load-bearing: an unsorted
        // slice silently misses words, and a word longer than the bound
        // is unreachable.
        let mut sorted = ENGLISH_STOPWORDS.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, ENGLISH_STOPWORDS, "list must be sorted");
        sorted.dedup();
        assert_eq!(sorted.len(), ENGLISH_STOPWORDS.len(), "no duplicates");
        for w in ENGLISH_STOPWORDS {
            assert!(
                w.len() <= ENGLISH_STOPWORD_MAX_LEN,
                "{w:?} exceeds the length pre-filter bound"
            );
            assert!(Stopwords::English.contains(w), "{w:?} must be a stopword");
        }
        assert!(
            !Stopwords::English.contains("theory"),
            "prefix is not a hit"
        );
        assert!(!Stopwords::English.contains("th"));
    }

    #[test]
    fn a_filterless_chain_is_the_bare_base_tokenizer() {
        // Not merely an optimization: the wrapper would persist under a
        // name no plain column carries, so the round-trip below is the
        // format guarantee that a default column is unchanged.
        let tok = chain_tokenizer(Stopwords::None, Stemmer::None);
        assert_eq!(tok.name(), chain_name(Stopwords::None, Stemmer::None));
        assert!(
            tok.as_any().downcast_ref::<ChainTokenizer>().is_none(),
            "a filterless chain must not wrap"
        );
    }

    /// Every chain has a distinct derived name, and the tokenizer built
    /// from a set of components reports it. Distinctness is what the
    /// three consumers of the name depend on: the `LIKE` lowering, the
    /// merge carry check, and the options-hash all treat two columns as
    /// differently analyzed exactly when their names differ.
    #[test]
    fn every_chain_has_a_distinct_derived_name() {
        let mut seen: Vec<&str> = Vec::new();
        for stop in [Stopwords::None, Stopwords::English] {
            for stem in [Stemmer::None, Stemmer::English] {
                let name = chain_name(stop, stem);
                assert!(!seen.contains(&name), "{name:?} is not unique");
                seen.push(name);
                assert_eq!(
                    chain_tokenizer(stop, stem).name(),
                    name,
                    "{name:?}: the built tokenizer must report its own name"
                );
            }
        }
        assert_eq!(seen.len(), 4, "two stopword sets x two stemmers");
    }

    /// Every shipped chain emits exactly these tokens. A change here is a
    /// format change: columns already built hold the old terms and would
    /// be queried with the new ones. Bump the part that moved
    /// ([`STANDARD_REVISION`] and friends) so a reindex can repair them —
    /// do not re-record the expectations.
    ///
    /// The text covers what separates the chains: case, a hyphen split, a
    /// digit run, a stopword, an inflected word, and non-ASCII including
    /// an emoji, which `standard` emits as a token.
    #[test]
    fn every_chain_emits_its_recorded_tokens() {
        const TEXT: &str = "The Quick brown-foxes JUMPED over 42 lazy dogs! Café ☕ naïve";
        let expected: [(&str, &[&str]); 4] = [
            (
                "standard",
                &[
                    "the", "quick", "brown", "foxes", "jumped", "over", "42", "lazy", "dogs",
                    "café", "☕", "naïve",
                ],
            ),
            (
                "standard+stop=english",
                &[
                    "quick", "brown", "foxes", "jumped", "over", "42", "lazy", "dogs", "café",
                    "☕", "naïve",
                ],
            ),
            (
                "standard+stem=english",
                &[
                    "the", "quick", "brown", "fox", "jump", "over", "42", "lazi", "dog", "café",
                    "☕", "naïv",
                ],
            ),
            (
                "standard+stop=english+stem=english",
                &[
                    "quick", "brown", "fox", "jump", "over", "42", "lazi", "dog", "café", "☕",
                    "naïv",
                ],
            ),
        ];
        for (name, want) in expected {
            assert_eq!(
                tokens(name, TEXT),
                want,
                "{name:?} changed what it emits. This is a format change: \
                 bump the part that moved rather than re-recording this."
            );
        }
    }

    /// Every chain is at revision 1 today, which is what makes summing the
    /// parts a no-op for existing files: no column's recorded revision
    /// changes meaning under the new rule.
    #[test]
    fn every_shipped_chain_is_at_revision_one() {
        for stop in [Stopwords::None, Stopwords::English] {
            for stem in [Stemmer::None, Stemmer::English] {
                let name = chain_name(stop, stem);
                assert_eq!(
                    chain_revision(stop, stem),
                    1,
                    "{name:?} moved off revision 1 — every file it wrote \
                     reads as stale, so this needs to be deliberate"
                );
            }
        }
    }

    /// A bump in any one part must move the chain's revision. Taking the
    /// maximum instead hides a filter bump behind a higher base, so a
    /// column whose stemmer moved keeps reporting the revision it had and
    /// a reindex leaves its terms stale.
    #[test]
    fn a_chain_revision_moves_when_any_single_part_moves() {
        let flat = combine_revisions(1, 0, 0);
        assert_eq!(combine_revisions(2, 0, 0), flat + 1, "base moved");
        assert_eq!(combine_revisions(1, 1, 0), flat + 1, "stopwords moved");
        assert_eq!(combine_revisions(1, 0, 1), flat + 1, "stemmer moved");
        assert_eq!(combine_revisions(2, 1, 1), flat + 3, "every part moved");
    }

    /// The persisted field values round-trip, and a value this engine
    /// cannot reproduce resolves to `None` so the caller can refuse the
    /// file. An unknown *field* is ignorable — an unknown *value* of a
    /// known field is not, because there is no sound way to proceed.
    #[test]
    fn filter_names_round_trip_and_unknown_values_are_refused() {
        assert_eq!(Stopwords::English.as_str(), Some("english"));
        assert_eq!(Stemmer::English.as_str(), Some("english"));
        // `None` is written by omitting the field, so it has no name.
        assert_eq!(Stopwords::None.as_str(), None);
        assert_eq!(Stemmer::None.as_str(), None);
        assert_eq!(Stopwords::from_name("english"), Some(Stopwords::English));
        assert_eq!(Stemmer::from_name("english"), Some(Stemmer::English));
        for unknown in ["german", "porter", "English", "", "none"] {
            assert_eq!(Stopwords::from_name(unknown), None, "{unknown:?}");
            assert_eq!(Stemmer::from_name(unknown), None, "{unknown:?}");
        }
    }

    #[test]
    fn stopwords_are_removed_and_leave_a_position_hole() {
        // The hole is what keeps `"new york"` from phrase-matching
        // `"new the york"`: without it both index at positions 0, 1.
        assert_eq!(
            positioned("standard+stop=english", "new the york"),
            vec![("new".to_string(), 0), ("york".to_string(), 2)]
        );
        // A leading and a trailing stopword each consume their ordinal
        // too, so the kept tokens' *relative* spacing is preserved
        // wherever they sit.
        assert_eq!(
            positioned("standard+stop=english", "the new york of"),
            vec![("new".to_string(), 1), ("york".to_string(), 2)]
        );
        // Nothing but stopwords: no tokens, and no phantom position.
        assert!(positioned("standard+stop=english", "the and of").is_empty());
    }

    #[test]
    fn stemming_folds_inflections_onto_one_term() {
        assert_eq!(
            tokens("standard+stem=english", "running runs runner ran"),
            vec!["run", "run", "runner", "ran"],
            "Porter2 folds the regular inflections, not the irregular ones"
        );
        // Both filters, in the fixed order: the stopword set is matched
        // against the *unstemmed* token, so `their` is removed as a
        // stopword rather than stemmed first.
        assert_eq!(
            tokens(
                "standard+stop=english+stem=english",
                "their studies are running"
            ),
            vec!["studi", "run"]
        );
    }

    #[test]
    fn the_base_tokenizer_still_decides_the_token_alphabet() {
        // `standard` keeps non-ASCII, and the English stemmer leaves a
        // word it has no rule for alone.
        assert_eq!(
            tokens("standard+stem=english", "Café Studies"),
            vec!["café", "studi"]
        );
    }

    #[test]
    fn every_entry_point_emits_the_same_tokens() {
        // `tokenize` / `tokenize_each` / `tokenize_each_query` /
        // `tokenize_each_positioned` all route through one scan; a
        // disagreement between them would index text under one form and
        // query it under another.
        let text = "The Running Studies of New York are here";
        for name in [
            "standard+stop=english",
            "standard+stem=english",
            "standard+stop=english+stem=english",
        ] {
            let tok = chain(name);
            let via_tokenize: Vec<String> = tok.tokenize(text).collect();
            let mut via_each = Vec::new();
            tok.tokenize_each(text, &mut |t| via_each.push(t.to_owned()));
            let mut via_query = Vec::new();
            tok.tokenize_each_query(text, &mut |t| via_query.push(t.into_owned()));
            let via_positioned: Vec<String> =
                positioned(name, text).into_iter().map(|(t, _)| t).collect();
            assert_eq!(via_tokenize, via_each, "{name}: tokenize vs tokenize_each");
            assert_eq!(via_tokenize, via_query, "{name}: tokenize vs query");
            assert_eq!(
                via_tokenize, via_positioned,
                "{name}: tokenize vs positioned"
            );
        }
    }

    #[test]
    fn only_an_absent_or_standard_recorded_tokenizer_is_accepted() {
        check_recorded_tokenizer("body", None).expect("absent opens");
        check_recorded_tokenizer("body", Some(STANDARD_TOKENIZER)).expect("standard opens");
        let msg = check_recorded_tokenizer("body", Some("ascii_lower"))
            .expect_err("a removed analyzer is refused")
            .to_string();
        assert!(
            msg.contains("\"ascii_lower\"") && msg.contains("re-create"),
            "the error names the analyzer and the fix: {msg}"
        );
    }
}
