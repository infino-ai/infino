// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The per-column analysis chain: a base tokenizer plus optional
//! stopword removal and stemming, in that order.
//!
//! ## Shape
//!
//! One base tokenizer ([`AsciiLowerTokenizer`] or
//! [`StandardTokenizer`]) followed by up to two token filters:
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
//! plain analyzer name, and takes the same monomorphized ingest scan
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
//! ## The chain is its name
//!
//! A chain persists as one composite analyzer name —
//! `"standard+stop=english+stem=english"` — and nothing else. That is a
//! deliberate choice over a sibling `"stemmer"` field in
//! `inf.fts.columns`: an older reader *ignores* an unknown JSON field,
//! and an ignored analysis filter means the reader tokenizes queries
//! differently than the index was built, which is wrong answers rather
//! than an error. The analyzer name is the one channel already shipped
//! readers fail loud on — [`tokenizer_for_name`] returning `None` is an
//! open error at every call site — so routing the whole chain through
//! it makes an engine that predates a filter refuse the file instead of
//! mis-ranking it.
//!
//! Carrying the chain in the name also means nothing else has to learn
//! about it. The name is already what the catalog record
//! (`TableEntry::fts_analyzers`), the remote create-table wire and the
//! options-hash's `fts_analyzers` block carry, so a chained column
//! rides all three unchanged, and a plain column's bytes stay
//! byte-identical to today's in every one of them.
//!
//! Because the built-in sets are a closed set (no user-supplied word
//! lists — those would need a sibling field and would give the name
//! back its ambiguity), the reachable names are a finite table, which
//! is what lets [`Tokenizer::name`] keep returning `&'static str`.
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

use super::tokenize::{
    ASCII_LOWER_TOKENIZER, AsciiLowerTokenizer, STANDARD_TOKENIZER, StandardTokenizer, Tokenizer,
};

/// Stopword set applied to a column, after the base tokenizer and
/// before the stemmer.
///
/// Named built-ins only. A user-supplied word list would have to
/// persist beside the analyzer name, which is exactly the
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
/// whose `parse_chain_name` arm is added beside this one, leaving files
/// built under the old name reconstructible from the old list — never
/// as an edit here.
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

/// The base tokenizer a chain is built on — the two shipped analyzers,
/// as a closed enum so the chain can call each one's monomorphized
/// inherent scan instead of dispatching through `dyn Tokenizer` per
/// token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Base {
    /// [`AsciiLowerTokenizer`].
    AsciiLower,
    /// [`StandardTokenizer`].
    Standard,
}

impl Base {
    /// This base's own plain analyzer name — the chain name with no
    /// filters.
    pub(crate) fn name(self) -> &'static str {
        chain_name(self, Stopwords::None, Stemmer::None)
    }

    /// Resolve a base analyzer name. Rejects a composite name; use
    /// [`parse_chain_name`] for one of those.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            ASCII_LOWER_TOKENIZER => Some(Base::AsciiLower),
            STANDARD_TOKENIZER => Some(Base::Standard),
            _ => None,
        }
    }

    /// Scan `text` into `(token, position)` pairs where `position` is
    /// the **gap-inclusive** ordinal: a run the base tokenizer itself
    /// drops (`ascii_lower`'s non-ASCII rule) still consumes one.
    ///
    /// Monomorphized over `F` for the same reason the base tokenizers'
    /// own `tokenize_each_inline` methods are: the ingest path's
    /// interning closure has to inline into the scan loop.
    #[inline]
    fn scan_positioned<F: FnMut(&str, u64)>(self, text: &str, mut f: F) {
        match self {
            Base::AsciiLower => AsciiLowerTokenizer.tokenize_each_inline_positioned(text, f),
            // `standard` drops nothing — every segment carrying an
            // alphanumeric is emitted — so its emission ordinal is
            // already the gap-inclusive position.
            Base::Standard => {
                let mut position = 0u64;
                StandardTokenizer.tokenize_each_inline(text, |tok| {
                    f(tok, position);
                    position += 1;
                });
            }
        }
    }
}

/// A base tokenizer plus its stopword set and stemmer.
///
/// Constructed only through [`chain_tokenizer`] / [`tokenizer_for_name`],
/// so a `ChainTokenizer` always carries at least one active filter — a
/// chain with neither is the base tokenizer itself, under the base's own
/// plain name, and must not be wrapped (wrapping it would change the
/// persisted name and give up the base's downcast fast path for nothing).
pub struct ChainTokenizer {
    base: Base,
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
    /// Same role as [`AsciiLowerTokenizer::tokenize_each_inline_positioned`],
    /// and reached the same way: the FTS build path downcasts through
    /// [`Tokenizer::as_any`] so the interning closure inlines into the
    /// scan instead of paying an indirect call per token.
    #[inline]
    pub fn tokenize_each_inline_positioned<F: FnMut(&str, u64)>(&self, text: &str, mut f: F) {
        self.base.scan_positioned(text, |tok, position| {
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
    /// stemming column for a plain one (see `Analyzer::of`).
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

/// The canonical composite name for a chain, or the base's own plain
/// name when no filter is active.
///
/// The whole reachable set, spelled out: with named built-ins only, the
/// chain has two bases × two stopword settings × two stemmer settings,
/// so every name a column can carry is a `&'static str` here. That is
/// what lets [`Tokenizer::name`] stay `&'static str` and the persisted
/// name stay a closed vocabulary a reader either knows or rejects.
///
/// Component order is fixed (`stop` before `stem`, matching the order
/// the filters run), so one chain has exactly one spelling.
pub(crate) fn chain_name(base: Base, stopwords: Stopwords, stemmer: Stemmer) -> &'static str {
    match (base, stopwords, stemmer) {
        (Base::Standard, Stopwords::None, Stemmer::None) => STANDARD_TOKENIZER,
        (Base::Standard, Stopwords::English, Stemmer::None) => "standard+stop=english",
        (Base::Standard, Stopwords::None, Stemmer::English) => "standard+stem=english",
        (Base::Standard, Stopwords::English, Stemmer::English) => {
            "standard+stop=english+stem=english"
        }
        (Base::AsciiLower, Stopwords::None, Stemmer::None) => ASCII_LOWER_TOKENIZER,
        (Base::AsciiLower, Stopwords::English, Stemmer::None) => "ascii_lower+stop=english",
        (Base::AsciiLower, Stopwords::None, Stemmer::English) => "ascii_lower+stem=english",
        (Base::AsciiLower, Stopwords::English, Stemmer::English) => {
            "ascii_lower+stop=english+stem=english"
        }
    }
}

/// Build the tokenizer for a chain: the bare base tokenizer when no
/// filter is active, otherwise a [`ChainTokenizer`].
///
/// A filterless chain must *not* be wrapped — the wrapper would persist
/// under a composite name no plain column has and would hide the base
/// from the build path's downcast, costing ingest throughput for no
/// behaviour change.
pub(crate) fn chain_tokenizer(
    base: Base,
    stopwords: Stopwords,
    stemmer: Stemmer,
) -> Arc<dyn Tokenizer> {
    if stopwords == Stopwords::None && stemmer == Stemmer::None {
        return match base {
            Base::AsciiLower => Arc::new(AsciiLowerTokenizer),
            Base::Standard => Arc::new(StandardTokenizer),
        };
    }
    Arc::new(ChainTokenizer {
        base,
        stopwords,
        name: chain_name(base, stopwords, stemmer),
        snowball: match stemmer {
            Stemmer::None => None,
            Stemmer::English => Some(Snowball::create(Algorithm::English)),
        },
    })
}

/// Parse a composite analyzer name into its components, or `None` for
/// anything this engine does not implement — an unknown base, an
/// unknown filter, a filter named twice, or components out of canonical
/// order.
///
/// Strict on purpose. This is the fail-loud channel: a name an engine
/// cannot reproduce exactly must be refused, because tokenizing a query
/// with an approximation of the chain the postings were built with
/// returns wrong answers instead of an error.
pub(crate) fn parse_chain_name(name: &str) -> Option<(Base, Stopwords, Stemmer)> {
    let mut parts = name.split('+');
    let base = Base::from_name(parts.next()?)?;
    let mut stopwords = Stopwords::None;
    let mut stemmer = Stemmer::None;
    for part in parts {
        match part {
            // `stop` must precede `stem`, so a `stop` arriving after one
            // was set — or after `stem` was — is not canonical.
            "stop=english" if stopwords == Stopwords::None && stemmer == Stemmer::None => {
                stopwords = Stopwords::English;
            }
            "stem=english" if stemmer == Stemmer::None => stemmer = Stemmer::English,
            _ => return None,
        }
    }
    // A filterless name reaches here only as the bare base name, which
    // `chain_name` agrees on — so the round-trip below holds for it too.
    match chain_name(base, stopwords, stemmer) == name {
        true => Some((base, stopwords, stemmer)),
        false => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(name: &str) -> Arc<dyn Tokenizer> {
        let (base, stop, stem) = parse_chain_name(name).expect("known chain");
        chain_tokenizer(base, stop, stem)
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
    /// `stop=english2`, with its own `parse_chain_name` arm and its own
    /// frozen list — so files under the old name stay reconstructible.
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
        for base in [Base::AsciiLower, Base::Standard] {
            let tok = chain_tokenizer(base, Stopwords::None, Stemmer::None);
            assert_eq!(tok.name(), chain_name(base, Stopwords::None, Stemmer::None));
            assert!(
                tok.as_any().downcast_ref::<ChainTokenizer>().is_none(),
                "a filterless chain must not wrap"
            );
        }
    }

    #[test]
    fn chain_names_round_trip_and_reject_non_canonical_spellings() {
        for base in [Base::AsciiLower, Base::Standard] {
            for stop in [Stopwords::None, Stopwords::English] {
                for stem in [Stemmer::None, Stemmer::English] {
                    let name = chain_name(base, stop, stem);
                    assert_eq!(
                        parse_chain_name(name),
                        Some((base, stop, stem)),
                        "{name:?} must round-trip"
                    );
                    assert_eq!(
                        chain_tokenizer(base, stop, stem).name(),
                        name,
                        "{name:?}: the built tokenizer must report its own name"
                    );
                }
            }
        }
        // Every rejection below is a name that would otherwise be
        // tokenized by an approximation of the chain it asks for.
        for bad in [
            "nonesuch",
            "standard+stem=english+stop=english", // components out of order
            "standard+stop=english+stop=english", // repeated
            "standard+stop=german",               // set we do not ship
            "standard+stem=porter",               // a different algorithm
            "standard+",
            "+standard",
            "standard+stem=english+",
            "STANDARD",
        ] {
            assert_eq!(parse_chain_name(bad), None, "{bad:?} must be rejected");
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
        // `ascii_lower` drops a non-ASCII run whole and leaves its
        // ordinal as a hole; the filters run on what survives that.
        assert_eq!(
            positioned("ascii_lower+stem=english", "Running café STUDIES"),
            vec![("run".to_string(), 0), ("studi".to_string(), 2)]
        );
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
            "ascii_lower+stop=english+stem=english",
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
}
