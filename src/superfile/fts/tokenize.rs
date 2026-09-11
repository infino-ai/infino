// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Tokenization, plus BM25 query parsing ([`Tokenizer::parse`]).
//! The parser lives here because the `+` / `-` clause sigils must be
//! handled before tokenizing — the tokenizer splits on both.
//!
//! Ships two tokenizers: [`StandardTokenizer`] (the default) and
//! [`AsciiLowerTokenizer`]. The [`Tokenizer`] trait is the extension
//! point for ICU / language-aware stemmers / custom char filters
//! under the same trait without touching FTS code.
//!
//! [`AsciiLowerTokenizer`] semantics:
//!   - Split on any byte that isn't `[A-Za-z0-9]`.
//!   - Lowercase each ASCII letter (bytes `b'A'..=b'Z'` → `b'a'..=b'z'`).
//!   - Drop any token that contains a non-ASCII byte (high-bit set).
//!     Non-ASCII tokens are silently dropped (not an error) — the
//!     ASCII-only design is intentional; richer tokenizers can opt
//!     into the trait without changing the FTS pipeline.
//!   - Empty tokens are never emitted.
//!
//! [`StandardTokenizer`] semantics (Unicode-aware):
//!   - Segment on Unicode text boundaries (UAX #29 word boundaries),
//!     keeping runs that contain alphanumerics and discarding
//!     whitespace/punctuation-only segments.
//!   - Lowercase each token via full Unicode case folding.
//!   - Non-ASCII letters and digits are preserved (not dropped), so
//!     accented and non-Latin scripts remain searchable.
//!   - **No Unicode normalization** (NFC/NFD): a token is emitted in its
//!     input code-point encoding. This matches the `standard` analyzer in
//!     Lucene / Elasticsearch, which applies normalization only through a
//!     separate, opt-in ICU filter — never in the standard pipeline.
//!     Canonicalizing equivalent encodings (e.g. precomposed `é` vs. the
//!     base `e` + combining acute) is therefore a distinct analyzer's job:
//!     a normalizing analyzer plugs in through the [`Tokenizer`] trait
//!     rather than altering `standard`'s semantics.

use std::{
    any::Any,
    borrow::Cow,
    collections::BTreeSet,
    ops::Deref,
    str::{from_utf8, from_utf8_unchecked},
    sync::Arc,
};

use unicode_segmentation::UnicodeSegmentation;
use wide::u8x16;

use super::{
    analysis::{Base, Stemmer, Stopwords, chain_tokenizer},
    reader::BoolMode,
};

/// Smallest byte value that is non-ASCII (has the high bit set). The
/// v1 ASCII-only rule drops any token containing a byte `>= this`.
const NON_ASCII_BYTE_MIN: u8 = 0x80;

/// Low-16-bit mask applied to a `u8x16` comparison bitmask, keeping
/// one bit per SIMD lane (the scan processes 16 bytes per chunk).
const LANE_BITMASK: u32 = 0xFFFF;

/// Initial capacity of the lowercase-token scratch buffer. Sized to
/// the common case of short tokens so the hot path rarely reallocs.
const TOKEN_SCRATCH_INITIAL_CAP: usize = 32;

/// Trait every tokenizer impl must satisfy.
///
/// Three entry points:
///
///   - [`Tokenizer::tokenize`] — iterator-shaped, yields owned
///     `String`s. Convenient for query-side / one-off use, but
///     allocates one heap `String` per token.
///
///   - [`Tokenizer::tokenize_each`] — callback-shaped, hands the
///     callback a `&str` borrowed from an internal scratch buffer
///     (valid only for the duration of the call). Zero-alloc on the
///     hot ingest path. The default impl wraps `tokenize`; impls
///     that can do better (like [`AsciiLowerTokenizer`]) override.
///     The callback is `&mut dyn FnMut`, so each per-token call
///     pays one indirect dispatch and LLVM cannot inline the
///     callback body into the tokenizer scan loop.
///
///   - [`Tokenizer::as_any`] — downcast hatch so the FTS build path
///     can take a monomorphic fast path for either shipped tokenizer.
///     It bypasses the `&mut dyn FnMut(&str)` indirection by calling
///     the inherent `tokenize_each_inline` method
///     ([`AsciiLowerTokenizer::tokenize_each_inline`],
///     [`StandardTokenizer::tokenize_each_inline`]), whose
///     `F: FnMut(&str)` parameter lets LLVM inline the callback
///     body straight into the tokenizer's scan. Custom
///     tokenizers don't need to opt in — they just return `self`
///     and never get downcast.
pub trait Tokenizer: Send + Sync + std::fmt::Debug + 'static {
    /// Stable registry name recorded in a column's stored FTS config
    /// (the `tokenizer` field of `inf.fts.columns`) and used to
    /// reconstruct the matching tokenizer at read time via
    /// [`tokenizer_for_name`]. Must round-trip:
    /// `tokenizer_for_name(t.name())` yields a tokenizer of the same
    /// kind as `t`.
    fn name(&self) -> &'static str;

    /// Yield each token as an owned `String` lower-cased per the
    /// implementation's rules.
    ///
    /// ## Phrase positions and dropped tokens
    ///
    /// The positional index numbers tokens by the position this trait
    /// reports, and exact-phrase matching checks those numbers for
    /// adjacency. A tokenizer that *drops* an input token (a stopword
    /// filter, a non-ASCII skip) must leave the dropped token's
    /// ordinal behind as a hole, or the tokens on either side of it
    /// look adjacent and a phrase matches text that is not contiguous.
    /// [`Tokenizer::tokenize_each_positioned`] is that channel; a
    /// tokenizer that drops tokens overrides it.
    fn tokenize<'a>(&'a self, text: &'a str) -> Box<dyn Iterator<Item = String> + 'a>;

    /// Call `f(&token)` for each token. The `&str` passed to `f` is
    /// valid only for that call — copy it (e.g. into a bump arena) if
    /// you need to keep it.
    ///
    /// Default impl iterates `self.tokenize(...)` and calls `f` on
    /// each `String` (one heap alloc per token). Impls that can be
    /// zero-alloc should override.
    fn tokenize_each(&self, text: &str, f: &mut dyn FnMut(&str)) {
        for s in self.tokenize(text) {
            f(&s);
        }
    }

    /// Call `f(&token, position)` for each token, where `position` is
    /// the token's **gap-inclusive** ordinal: every input token the
    /// scan considers consumes one, including the ones this tokenizer
    /// drops. The positional index build reads positions from here, so
    /// a dropped token's skipped ordinal is the phrase hole that keeps
    /// its neighbours non-adjacent (see the note on
    /// [`Tokenizer::tokenize`]).
    ///
    /// The `&str` lifetime rule is [`Tokenizer::tokenize_each`]'s.
    ///
    /// Default impl numbers the emitted tokens consecutively, which is
    /// correct for any tokenizer that drops nothing. A tokenizer that
    /// drops tokens **must** override this, or it silently reports no
    /// holes.
    fn tokenize_each_positioned(&self, text: &str, f: &mut dyn FnMut(&str, u64)) {
        let mut position = 0u64;
        self.tokenize_each(text, &mut |t| {
            f(t, position);
            position += 1;
        });
    }

    /// Downcast hatch for the FTS build hot path. Default impl
    /// returns `self` cast to `&dyn Any`; concrete impls should
    /// not override unless they wrap another tokenizer.
    fn as_any(&self) -> &dyn Any;

    /// Used to tokenize a query. The tokens handed to `f` stay alive
    /// as long as `text` (the query) is alive.
    fn tokenize_each_query<'q>(&self, text: &'q str, f: &mut dyn FnMut(Cow<'q, str>)) {
        self.tokenize_each(text, &mut |t| f(Cow::Owned(t.to_owned())));
    }

    /// [`Tokenizer::tokenize_each_query`] plus each token's
    /// gap-inclusive position — the query-side counterpart of
    /// [`Tokenizer::tokenize_each_positioned`], and what lets a quoted
    /// phrase keep the holes this tokenizer left in it.
    ///
    /// Default impl numbers the emitted tokens consecutively, keeping
    /// the borrowing `tokenize_each_query` override a tokenizer may
    /// have. A tokenizer that drops tokens **must** override this, for
    /// the same reason it must override the indexing counterpart: a
    /// phrase whose holes are not reported matches text where the words
    /// are not that far apart.
    fn tokenize_each_query_positioned<'q>(
        &self,
        text: &'q str,
        f: &mut dyn FnMut(Cow<'q, str>, u64),
    ) {
        let mut position = 0u64;
        self.tokenize_each_query(text, &mut |t| {
            f(t, position);
            position += 1;
        });
    }

    /// Used to parse a query into its clauses by leading sigil:
    /// `"+rust async -python"` → musts `["rust"]`, positives
    /// `["async"]`, negatives `["python"]`. A `+`-prefixed run is a
    /// **must** clause (the doc must contain it), a `-`-prefixed run
    /// a **must-not** clause (hard exclusion), and a bare run lands
    /// in `positives`, whose polarity the query layer resolves from
    /// the default operator (`BoolMode`). A query with no must or
    /// positive clause is not an error here; the caller checks.
    fn parse<'q>(&self, query: &'q str) -> ParsedQuery<'q> {
        let mut parsed = ParsedQuery::default();
        let bytes = query.as_bytes();
        let mut i = 0usize;
        let mut seg_start = 0usize;
        while i < bytes.len() {
            if bytes[i] != b'"' {
                i += 1;
                continue;
            }
            let Some(close_rel) = query[i + 1..].find('"') else {
                // Unbalanced quote: treat the dangling `"` as
                // whitespace (lenient, like lucene's parser) — close
                // the unquoted segment here and keep scanning after it.
                self.parse_unquoted_segment(&query[seg_start..i], &mut parsed);
                i += 1;
                seg_start = i;
                continue;
            };
            let close = i + 1 + close_rel;
            // A `+` / `-` glued to the opening quote — and itself at a
            // token boundary — sets the phrase's polarity; the sigil
            // byte is excluded from the unquoted segment.
            let sigil = match i > seg_start {
                true => {
                    let boundary = i - 1 == seg_start || bytes[i - 2].is_ascii_whitespace();
                    match (boundary, bytes[i - 1]) {
                        (true, b'+') => Some(b'+'),
                        (true, b'-') => Some(b'-'),
                        _ => None,
                    }
                }
                false => None,
            };
            let unquoted_end = match sigil {
                Some(_) => i - 1,
                None => i,
            };
            self.parse_unquoted_segment(&query[seg_start..unquoted_end], &mut parsed);
            let mut phrase: Phrase<Cow<'q, str>> = Phrase::default();
            // Offsets are relative to the first *surviving* term. A
            // leading token the analysis chain removed is unobservable
            // — there is nothing before it to space it from — and
            // normalizing here is what lets the verifier subtract an
            // offset from a position without underflowing near the
            // start of a document.
            let mut first_position: Option<u64> = None;
            self.tokenize_each_query_positioned(&query[i + 1..close], &mut |t, position| {
                let first = *first_position.get_or_insert(position);
                phrase.push(t, position.saturating_sub(first));
            });
            match (phrase.len(), sigil) {
                // Empty quotes contribute nothing.
                (0, _) => {}
                // A single-token phrase is just that term — degrade to
                // the term list of the same polarity. Its offset is
                // necessarily 0 and carries no information: a lone
                // token has nothing to be spaced from.
                (1, Some(b'-')) => parsed.negatives.push(phrase.pop_term()),
                (1, Some(b'+')) => parsed.musts.push(phrase.pop_term()),
                (1, _) => parsed.positives.push(phrase.pop_term()),
                (_, Some(b'-')) => parsed.negative_phrases.push(phrase),
                (_, Some(b'+')) => parsed.must_phrases.push(phrase),
                (_, _) => parsed.positive_phrases.push(phrase),
            }
            i = close + 1;
            seg_start = i;
        }
        self.parse_unquoted_segment(&query[seg_start..], &mut parsed);
        parsed
    }

    /// Parse one stretch of query text containing no quotes — the
    /// pre-phrase grammar: whitespace runs with optional `+`/`-`
    /// clause sigils.
    fn parse_unquoted_segment<'q>(&self, segment: &'q str, parsed: &mut ParsedQuery<'q>) {
        for run in segment.split_whitespace() {
            match (run.strip_prefix('-'), run.strip_prefix('+')) {
                (Some(rest), _) if !rest.is_empty() => {
                    self.tokenize_each_query(rest, &mut |t| parsed.negatives.push(t));
                }
                (_, Some(rest)) if !rest.is_empty() => {
                    self.tokenize_each_query(rest, &mut |t| parsed.musts.push(t));
                }
                _ => self.tokenize_each_query(run, &mut |t| parsed.positives.push(t)),
            }
        }
    }
}

/// Tokenize several `texts` into one sorted, de-duplicated term list.
/// For building a single term set from many values (e.g. an `IN` list)
/// where a word shared across values must be probed only once.
pub(crate) fn unique_tokens<'a>(
    tok: &dyn Tokenizer,
    texts: impl IntoIterator<Item = &'a str>,
) -> Vec<String> {
    texts
        .into_iter()
        .flat_map(|t| tok.tokenize(t))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// ASCII whitespace + punctuation split, ASCII lowercase, no stemming,
/// no stopwords. The simplest tokenizer that's still useful.
#[derive(Debug, Clone, Copy, Default)]
pub struct AsciiLowerTokenizer;

impl AsciiLowerTokenizer {
    pub fn new() -> Self {
        Self
    }

    /// Zero-alloc emission with a borrowed fast path, generic over
    /// the callback type so LLVM can inline the callback body into
    /// the per-byte scan loop. Same shape as the trait
    /// [`Tokenizer::tokenize_each`] but takes `mut f: F` instead of
    /// `f: &mut dyn FnMut(&str)`, which:
    ///
    ///   * eliminates the per-token indirect dispatch on the
    ///     callback (~150M call sites on the 1M-doc bench);
    ///   * lets LLVM CSE common subexpressions between the
    ///     callback body (intern hash, dense_doc_tf load) and
    ///     the scan loop, and hoist invariants like the
    ///     interner / dense-array base pointers across iterations.
    ///
    /// The two byte-scan passes (skip non-token bytes; extend a
    /// token run) are SIMD-accelerated via [`simd_skip_non_token`]
    /// and [`simd_scan_token_run`] respectively — both process 16
    /// bytes per `u8x16` chunk on AVX2 hosts, replacing the
    /// previous one-byte-per-iteration scalar `while` loops. The
    /// bench corpus has ~2 KB / doc of token bytes, so the
    /// 16×-wide scan trades ~9.4M scalar branches per doc for
    /// ~590K `u8x16` chunk ops + a scalar tail.
    ///
    /// Scans the input once. For each token-byte run:
    ///   * If the run is **already lowercase ASCII** (the common case
    ///     for log lines, telemetry tokens, "term00042"-shaped Zipfian
    ///     bench corpora, and lower-cased ingestion pipelines) the
    ///     callback gets a borrowed `&str` slicing directly into the
    ///     input — zero copy, zero scratch-buf write.
    ///   * If the run contains uppercase ASCII bytes, the run is
    ///     copied into a reusable scratch `buf` while lower-casing in
    ///     place. The callback then gets `&buf`.
    ///   * If the run contains any non-ASCII byte (≥ 0x80), the whole
    ///     run is dropped per the v1 ASCII-only rule.
    ///
    /// The borrowed/copied `&str` is only valid for that one callback
    /// call. The next callback invocation may overwrite `buf` or hand
    /// out a different slice; copy via bumpalo/Box if you need to
    /// keep it.
    #[inline]
    pub fn tokenize_each_inline<F: FnMut(&str)>(&self, text: &str, mut f: F) {
        // One scan implementation — delegate and discard the position
        // ordinal so the positionless / query paths and the positional
        // index build can never disagree on which tokens are emitted.
        self.tokenize_each_inline_positioned(text, |tok, _position| f(tok));
    }

    /// Like [`Self::tokenize_each_inline`], but also hands each emitted
    /// token its **position ordinal**: the count of token runs scanned
    /// before it, *including runs this tokenizer drops*.
    ///
    /// A run that contains a non-ASCII byte is dropped (no token), but
    /// it still consumes one ordinal — so a phrase never treats the
    /// tokens on either side of a dropped word as adjacent. Without
    /// this, `"new café york"` would tokenize to `new`, `york` at
    /// positions 0, 1 and the phrase `"new york"` would wrongly match
    /// it; with the gap they sit at 0 and 2. Used by the positional
    /// index build. (A run consisting *only* of non-ASCII bytes carries
    /// no ASCII token byte to anchor the scan and is treated as a
    /// separator, not a gap — the ASCII tokenizer cannot represent such
    /// text at all.)
    ///
    /// The borrowed/copied `&str` is valid only for that one callback
    /// call, exactly as in [`Self::tokenize_each_inline`].
    #[inline]
    pub fn tokenize_each_inline_positioned<F: FnMut(&str, u64)>(&self, text: &str, mut f: F) {
        let bytes = text.as_bytes();
        let mut buf: Vec<u8> = Vec::new();
        let mut pos = 0;
        let mut position: u64 = 0;
        while pos < bytes.len() {
            pos = simd_skip_non_token(bytes, pos);
            if pos >= bytes.len() {
                return;
            }
            let start = pos;
            let (end, had_upper, had_non_ascii) = simd_scan_token_run(bytes, pos);
            pos = end;
            if start == pos {
                continue;
            }
            // Every scanned run occupies one ordinal, dropped or not.
            let this_position = position;
            position += 1;
            if had_non_ascii {
                // Dropped per the v1 ASCII-only rule — the ordinal it
                // just consumed is the phrase gap it leaves behind.
                continue;
            }
            if !had_upper {
                // Fast path: borrow directly from `text`.
                //
                // SAFETY: `is_token_byte` only accepts ASCII
                // alphanumerics, so every byte in `bytes[start..end]`
                // is a single-byte ASCII codepoint. The slice is
                // therefore valid UTF-8 and the original `text`
                // outlives the callback call.
                let s = unsafe { from_utf8_unchecked(&bytes[start..end]) };
                f(s, this_position);
            } else {
                // Slow path: copy + lowercase into the reusable buf.
                buf.clear();
                buf.reserve(end - start);
                for &b in &bytes[start..end] {
                    buf.push(b.to_ascii_lowercase());
                }
                // SAFETY: same reasoning — every byte pushed is an
                // ASCII alphanumeric (or its lowercased form, which
                // is also ASCII).
                let s = unsafe { from_utf8_unchecked(&buf) };
                f(s, this_position);
            }
        }
    }
}

/// SIMD scan: advance `pos` past non-token bytes, returning the
/// index of the first ASCII alphanumeric byte (or `bytes.len()`).
///
/// Replaces a per-byte `while !is_ascii_alphanumeric(bytes[pos])`
/// loop with a 16-byte `u8x16` chunked scan. Within each chunk we
/// build an "is token byte" mask (`'0'..='9' | 'A'..='Z' |
/// 'a'..='z'`) via three range comparisons + two ORs, then jump to
/// the first set bit via `trailing_zeros`. Falls back to a scalar
/// tail for `bytes.len() % 16` trailing bytes.
#[inline(always)]
fn simd_skip_non_token(bytes: &[u8], mut pos: usize) -> usize {
    const LANES: usize = 16;
    while pos + LANES <= bytes.len() {
        // SAFETY: `pos + LANES <= bytes.len()` was checked above,
        // so reading 16 bytes from `bytes.as_ptr().add(pos)` stays
        // in-bounds. The cast to `*const [u8; LANES]` and deref
        // produces a copy (the array is loaded by value into the
        // SIMD register on the next line).
        let arr: [u8; LANES] = unsafe { *(bytes.as_ptr().add(pos) as *const [u8; LANES]) };
        let chunk = u8x16::from(arr);
        let is_digit = chunk.simd_ge(u8x16::splat(b'0')) & chunk.simd_le(u8x16::splat(b'9'));
        let is_upper = chunk.simd_ge(u8x16::splat(b'A')) & chunk.simd_le(u8x16::splat(b'Z'));
        let is_lower = chunk.simd_ge(u8x16::splat(b'a')) & chunk.simd_le(u8x16::splat(b'z'));
        let is_token = is_digit | is_upper | is_lower;
        let mask = is_token.to_bitmask() & LANE_BITMASK;
        if mask == 0 {
            pos += LANES;
        } else {
            return pos + mask.trailing_zeros() as usize;
        }
    }
    // Scalar tail.
    while pos < bytes.len() && !bytes[pos].is_ascii_alphanumeric() {
        pos += 1;
    }
    pos
}

/// SIMD scan: extend a token-byte run starting at `pos`, returning
/// `(end, had_upper, had_non_ascii)`. The run extends as long as
/// each byte is either an ASCII alphanumeric **or** a non-ASCII
/// (high-bit) byte — the latter just sets the `had_non_ascii`
/// drop-marker per the v1 ASCII-only rule. The run stops on the
/// first ASCII separator (any non-alphanumeric byte `< 0x80`).
///
/// Equivalent to the scalar version above but with 16-byte
/// chunked compares. Within each chunk:
///   * build the "extend" mask (`is_token | is_high`);
///   * if all 16 lanes extend, OR-in the `had_upper` /
///     `had_non_ascii` flags from the full chunk and advance 16;
///   * otherwise find the first separator lane via
///     `trailing_zeros(!extend & 0xFFFF)`, mask the flag bitmasks
///     to the consumed prefix only (so flags from bytes past the
///     separator don't leak into this token), and return.
#[inline(always)]
fn simd_scan_token_run(bytes: &[u8], mut pos: usize) -> (usize, bool, bool) {
    const LANES: usize = 16;
    let mut had_upper = false;
    let mut had_non_ascii = false;
    while pos + LANES <= bytes.len() {
        // SAFETY: bounds-checked at the loop guard above.
        let arr: [u8; LANES] = unsafe { *(bytes.as_ptr().add(pos) as *const [u8; LANES]) };
        let chunk = u8x16::from(arr);
        let is_digit = chunk.simd_ge(u8x16::splat(b'0')) & chunk.simd_le(u8x16::splat(b'9'));
        let is_upper = chunk.simd_ge(u8x16::splat(b'A')) & chunk.simd_le(u8x16::splat(b'Z'));
        let is_lower = chunk.simd_ge(u8x16::splat(b'a')) & chunk.simd_le(u8x16::splat(b'z'));
        // High-bit detect: mask high bit, compare equal to 0x80.
        // `simd_eq` is bit-equality on signed/unsigned-agnostic
        // `cmp_eq_mask_i8_m128i`, so this works for high bytes
        // even though `simd_gt`/`simd_ge` against `0x80` would
        // need an XOR-flip trick to handle signed-i8 wrap.
        let is_high =
            (chunk & u8x16::splat(NON_ASCII_BYTE_MIN)).simd_eq(u8x16::splat(NON_ASCII_BYTE_MIN));
        let is_token = is_digit | is_upper | is_lower;
        let is_extend = is_token | is_high;
        let extend_mask = is_extend.to_bitmask() & LANE_BITMASK;
        let upper_mask = is_upper.to_bitmask() & LANE_BITMASK;
        let high_mask = is_high.to_bitmask() & LANE_BITMASK;
        let non_extend = !extend_mask & LANE_BITMASK;
        if non_extend == 0 {
            had_upper |= upper_mask != 0;
            had_non_ascii |= high_mask != 0;
            pos += LANES;
        } else {
            let sep_idx = non_extend.trailing_zeros() as usize;
            let prefix_mask: u32 = (1u32 << sep_idx).wrapping_sub(1);
            had_upper |= (upper_mask & prefix_mask) != 0;
            had_non_ascii |= (high_mask & prefix_mask) != 0;
            pos += sep_idx;
            return (pos, had_upper, had_non_ascii);
        }
    }
    // Scalar tail (same logic as the scalar version).
    while pos < bytes.len() {
        let b = bytes[pos];
        if is_token_byte(b) {
            had_upper |= b.is_ascii_uppercase();
            pos += 1;
        } else if b >= NON_ASCII_BYTE_MIN {
            had_non_ascii = true;
            pos += 1;
        } else {
            break;
        }
    }
    (pos, had_upper, had_non_ascii)
}

/// One quoted phrase: its terms in query order, plus each term's
/// **offset** — how far that term sits from the phrase's first term in
/// token positions.
///
/// Offsets exist because an analysis chain can remove a token from the
/// middle of a phrase. `"end of the world"` on a column whose stopword
/// set drops `of` and `the` keeps two terms that were *four* positions
/// apart in the query, so requiring them adjacent would match nothing
/// and requiring them merely co-present would match `end world`.
/// Carrying the offsets keeps the phrase meaning what it says: the
/// index must hold `end` and `world` three positions apart, which is
/// exactly where the same chain put them at index time.
///
/// Offsets are relative to the first *kept* term, so they always start
/// at 0. A leading token the chain removed is unobservable — there is
/// nothing before it to space it from — and normalizing here is what
/// lets the verifier subtract an offset from a position without
/// underflowing near the start of a document.
///
/// Without a chain every offset is its term's index, so a phrase
/// behaves exactly as it did before offsets existed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phrase<T> {
    /// The phrase's terms, in query order.
    pub terms: Vec<T>,
    /// `offsets[i]` is `terms[i]`'s distance from `terms[0]`. Strictly
    /// ascending, and `offsets[0] == 0`.
    pub offsets: Vec<u32>,
}

impl<T> Default for Phrase<T> {
    fn default() -> Self {
        Self {
            terms: Vec::new(),
            offsets: Vec::new(),
        }
    }
}

impl<T> Phrase<T> {
    /// A phrase whose terms are adjacent — offsets `0..n`. The shape
    /// every phrase has on a column with no analysis chain.
    pub fn adjacent(terms: Vec<T>) -> Self {
        let offsets = (0..terms.len() as u32).collect();
        Self { terms, offsets }
    }

    /// Append `term` at `offset` from the phrase's first term. The
    /// caller normalizes, so nothing about how the phrase was built
    /// survives into the value — two phrases with the same terms and
    /// the same spacing are the same phrase, however each was
    /// assembled.
    ///
    /// `offset` saturates rather than panicking on a position a
    /// tokenizer reported out of order: a tokenizer is an extension
    /// point, and a misbehaving one should cost recall on its own
    /// column, not abort a query.
    fn push(&mut self, term: T, offset: u64) {
        self.offsets.push(offset.min(u32::MAX as u64) as u32);
        self.terms.push(term);
    }

    /// Take the sole term of a one-term phrase, which degrades to a
    /// plain term clause.
    fn pop_term(&mut self) -> T {
        self.terms.pop().expect("one term")
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// The offsets, for the positional verification.
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Rebuild the phrase over owned terms, preserving the offsets.
    pub fn map<U>(&self, f: impl FnMut(&T) -> U) -> Phrase<U> {
        Phrase {
            terms: self.terms.iter().map(f).collect(),
            offsets: self.offsets.clone(),
        }
    }
}

impl<T> Deref for Phrase<T> {
    type Target = [T];

    /// Derefs to the term slice so a phrase reads as its terms
    /// wherever the offsets are not the point.
    fn deref(&self) -> &[T] {
        &self.terms
    }
}

/// A parsed BM25 query, split into its clause lists by leading sigil:
/// `+term` → `musts`, bare `term` → `positives`, `-term` →
/// `negatives`. Tokens may borrow the query string, so this can't
/// outlive the query.
#[derive(Debug, Default)]
pub struct ParsedQuery<'q> {
    /// `+`-sigiled tokens: the doc must contain every one.
    pub musts: Vec<Cow<'q, str>>,
    /// Bare (sigil-less) tokens. Their polarity comes from the
    /// default operator: [`BoolMode::And`] treats them as musts,
    /// [`BoolMode::Or`] as shoulds (scoring-only once any must
    /// exists; a plain union when none does).
    pub positives: Vec<Cow<'q, str>>,
    /// `-`-sigiled tokens: any doc containing one is excluded.
    pub negatives: Vec<Cow<'q, str>>,
    /// `+"…"`-quoted runs of two or more tokens: the doc must contain
    /// the exact token sequence. (Single-token phrases degrade into
    /// `musts`.)
    pub must_phrases: Vec<Phrase<Cow<'q, str>>>,
    /// Bare-quoted multi-token runs; polarity resolved from the
    /// default operator like bare terms.
    pub positive_phrases: Vec<Phrase<Cow<'q, str>>>,
    /// `-"…"`-quoted multi-token runs: any doc containing the exact
    /// sequence is excluded.
    pub negative_phrases: Vec<Phrase<Cow<'q, str>>>,
}

/// A query's clause lists with the default operator already applied —
/// what [`ParsedQuery::into_clauses`] produces and the search kernels
/// consume. `shoulds` is non-empty only under [`BoolMode::Or`].
#[derive(Debug, Default)]
pub struct QueryClauses<'q> {
    /// Every doc in the result must contain all of these.
    pub musts: Vec<Cow<'q, str>>,
    /// Scoring-only when `musts` is non-empty; otherwise the match is
    /// their union.
    pub shoulds: Vec<Cow<'q, str>>,
    /// Docs containing any of these are excluded.
    pub negatives: Vec<Cow<'q, str>>,
    /// Multi-token phrases every doc in the result must contain.
    pub must_phrases: Vec<Phrase<Cow<'q, str>>>,
    /// Scoring-only phrases when `musts`/`must_phrases` is non-empty;
    /// otherwise part of the union match.
    pub should_phrases: Vec<Phrase<Cow<'q, str>>>,
    /// Docs containing any of these exact sequences are excluded.
    pub negative_phrases: Vec<Phrase<Cow<'q, str>>>,
}

impl<'q> ParsedQuery<'q> {
    /// Resolve the bare tokens' polarity from the default operator
    /// `mode`: `And` folds them into `musts`, `Or` makes them
    /// `shoulds`. Sigiled tokens keep their explicit polarity.
    pub fn into_clauses(self, mode: BoolMode) -> QueryClauses<'q> {
        let ParsedQuery {
            mut musts,
            positives,
            negatives,
            mut must_phrases,
            positive_phrases,
            negative_phrases,
        } = self;
        let shoulds = match mode {
            BoolMode::And => {
                musts.extend(positives);
                Vec::new()
            }
            BoolMode::Or => positives,
        };
        let should_phrases = match mode {
            BoolMode::And => {
                must_phrases.extend(positive_phrases);
                Vec::new()
            }
            BoolMode::Or => positive_phrases,
        };
        QueryClauses {
            musts,
            shoulds,
            negatives,
            must_phrases,
            should_phrases,
            negative_phrases,
        }
    }
}

impl Tokenizer for AsciiLowerTokenizer {
    fn name(&self) -> &'static str {
        ASCII_LOWER_TOKENIZER
    }

    fn tokenize<'a>(&'a self, text: &'a str) -> Box<dyn Iterator<Item = String> + 'a> {
        Box::new(AsciiLowerIter::new(text.as_bytes()))
    }

    /// Trait-object dispatch path: delegates to the inherent
    /// [`tokenize_each_inline`](Self::tokenize_each_inline) so the
    /// body lives in one place. Callers that hold a concrete
    /// [`AsciiLowerTokenizer`] (or successfully downcast a `&dyn
    /// Tokenizer` via [`Tokenizer::as_any`]) should call the
    /// inherent method directly to skip the per-token
    /// `&mut dyn FnMut(&str)` indirection.
    fn tokenize_each(&self, text: &str, f: &mut dyn FnMut(&str)) {
        self.tokenize_each_inline(text, |s| f(s));
    }

    /// Overridden because this tokenizer drops non-ASCII runs: the
    /// default consecutive numbering would report no hole where one
    /// was left.
    fn tokenize_each_positioned(&self, text: &str, f: &mut dyn FnMut(&str, u64)) {
        self.tokenize_each_inline_positioned(text, |s, position| f(s, position));
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Zero-copy override: an already-lowercase token borrows from
    /// `text`; only a token that needs lowercasing is copied.
    fn tokenize_each_query<'q>(&self, text: &'q str, f: &mut dyn FnMut(Cow<'q, str>)) {
        // One scan implementation — delegate and discard the position,
        // so the two query entry points cannot disagree about which
        // tokens a query has.
        self.tokenize_each_query_positioned(text, &mut |t, _position| f(t));
    }

    /// Zero-copy *and* gap-aware: the same borrowed-where-possible
    /// tokens as [`Self::tokenize_each_query`], each with the
    /// gap-inclusive ordinal a dropped non-ASCII run leaves behind.
    ///
    /// The gap is why this is overridden rather than left to the
    /// trait's consecutive default. A phrase query is verified against
    /// the positions the *index* recorded, and the index has always
    /// left a hole for a dropped run — so numbering the query's
    /// surviving tokens consecutively asks for them closer together
    /// than they were indexed, and `"new café york"` could never match
    /// the text it was copied from.
    fn tokenize_each_query_positioned<'q>(
        &self,
        text: &'q str,
        f: &mut dyn FnMut(Cow<'q, str>, u64),
    ) {
        let bytes = text.as_bytes();
        let mut pos = 0;
        let mut position: u64 = 0;
        while pos < bytes.len() {
            pos = simd_skip_non_token(bytes, pos);
            if pos >= bytes.len() {
                return;
            }
            let start = pos;
            let (end, had_upper, had_non_ascii) = simd_scan_token_run(bytes, pos);
            pos = end;
            if start == pos {
                continue;
            }
            // Every scanned run occupies one ordinal, dropped or not —
            // the same rule `tokenize_each_inline_positioned` follows,
            // so query and index agree on the spacing.
            let this_position = position;
            position += 1;
            if had_non_ascii {
                continue;
            }
            let s = from_utf8(&bytes[start..end]).expect("ASCII-only by construction");
            if had_upper {
                f(Cow::Owned(s.to_ascii_lowercase()), this_position);
            } else {
                f(Cow::Borrowed(s), this_position);
            }
        }
    }
}

/// Internal iterator that walks the input byte slice once, emitting
/// lowercased tokens. Skips tokens containing non-ASCII bytes per the
/// v1 ASCII-only rule.
struct AsciiLowerIter<'a> {
    src: &'a [u8],
    pos: usize,
    buf: Vec<u8>,
}

impl<'a> AsciiLowerIter<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            pos: 0,
            buf: Vec::with_capacity(TOKEN_SCRATCH_INITIAL_CAP),
        }
    }
}

impl Iterator for AsciiLowerIter<'_> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        loop {
            // Skip non-token bytes.
            while self.pos < self.src.len() && !is_token_byte(self.src[self.pos]) {
                self.pos += 1;
            }
            if self.pos >= self.src.len() {
                return None;
            }

            // Accumulate one token.
            self.buf.clear();
            let mut had_non_ascii = false;
            while self.pos < self.src.len() {
                let b = self.src[self.pos];
                if is_token_byte(b) {
                    self.buf.push(b.to_ascii_lowercase());
                    self.pos += 1;
                } else if b >= NON_ASCII_BYTE_MIN {
                    // Non-ASCII byte inside a contiguous "word-ish" run —
                    // mark this run as non-ASCII and consume until a true
                    // separator. Drop the whole token.
                    had_non_ascii = true;
                    self.pos += 1;
                } else {
                    break;
                }
            }

            if had_non_ascii || self.buf.is_empty() {
                continue;
            }

            // SAFETY: we only push ASCII letters and digits via
            // is_token_byte + to_ascii_lowercase, so the buffer is
            // guaranteed valid UTF-8.
            let s = from_utf8(&self.buf)
                .expect("ASCII-only by construction")
                .to_owned();
            return Some(s);
        }
    }
}

/// `[A-Za-z0-9]` — the v1 token alphabet.
#[inline]
fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric()
}

/// Name of the ASCII-only tokenizer in a column's FTS config.
pub const ASCII_LOWER_TOKENIZER: &str = "ascii_lower";

/// Name of the Unicode-aware standard tokenizer in a column's FTS
/// config, and the analyzer a column gets when none is named.
pub const STANDARD_TOKENIZER: &str = "standard";

/// Resolve an analyzer name to a tokenizer instance, or `None` for a
/// name this engine cannot reproduce. The single routing point shared
/// by the build path (mapping a chosen analyzer to the tokenizer used
/// at index time) and the read path (reconstructing a column's
/// tokenizer from the name recorded in its stored config). Callers
/// translate `None` into their own error — a malformed-superfile read
/// error, or an invalid-argument error at table-create time.
///
/// Resolves a **base tokenizer** name only. A column's stopword set and
/// stemmer are separate persisted fields, so a chained column's
/// tokenizer is built by [`chain_tokenizer`] from all three rather than
/// resolved from a single string here — see
/// [`crate::superfile::fts::analysis`].
pub fn tokenizer_for_name(name: &str) -> Option<Arc<dyn Tokenizer>> {
    let base = Base::from_name(name)?;
    Some(chain_tokenizer(base, Stopwords::None, Stemmer::None))
}

// ── UAX #29 word breaks, restricted to ASCII ─────────────────────────
//
// Over ASCII the word-break property takes only seven values, so the
// general segmenter's per-character property lookups collapse into one
// 256-entry table and a handful of bit tests. `standard` is the default
// analyzer and most corpora are predominantly ASCII, so this is the path
// that decides ingest throughput. `ascii_word_segments` is verified
// against `unicode_words` exhaustively in the tests below — the two must
// agree exactly, or text would be indexed under one tokenization and
// queried under another.

/// Word_Break classes reachable from an ASCII byte, as disjoint bits so
/// a rule can test a set membership with one `&`.
const WB_OTHER: u8 = 0;
/// `[A-Za-z]`.
const WB_ALETTER: u8 = 1 << 0;
/// `[0-9]`.
const WB_NUMERIC: u8 = 1 << 1;
/// `_` — joins to alphanumerics on either side and is kept *inside* the
/// token (`_id` and `a_b` are each one word), unlike every other joiner.
const WB_EXTEND_NUM_LET: u8 = 1 << 2;
/// `:` — joins letters only.
const WB_MID_LETTER: u8 = 1 << 3;
/// `,` `;` — join digits only.
const WB_MID_NUM: u8 = 1 << 4;
/// `.` `'` — join letters *or* digits (MidNumLet plus Single_Quote,
/// which are indistinguishable within ASCII).
const WB_MID_NUM_LET: u8 = 1 << 5;

/// `WB_ALETTER | WB_NUMERIC` — the classes that alone can carry a token.
const WB_ALNUM: u8 = WB_ALETTER | WB_NUMERIC;

/// Word_Break class per byte value. Non-ASCII bytes map to
/// [`WB_OTHER`]; the ASCII scan is only entered for all-ASCII input, so
/// those entries are never consulted.
static ASCII_WORD_BREAK_CLASS: [u8; 256] = {
    let mut table = [WB_OTHER; 256];
    let mut b = 0usize;
    while b < 128 {
        table[b] = match b as u8 {
            b'A'..=b'Z' | b'a'..=b'z' => WB_ALETTER,
            b'0'..=b'9' => WB_NUMERIC,
            b'_' => WB_EXTEND_NUM_LET,
            b':' => WB_MID_LETTER,
            b',' | b';' => WB_MID_NUM,
            b'.' | b'\'' => WB_MID_NUM_LET,
            _ => WB_OTHER,
        };
        b += 1;
    }
    table
};

#[inline(always)]
fn wb_class(b: u8) -> u8 {
    ASCII_WORD_BREAK_CLASS[b as usize]
}

/// Whether UAX #29 joins the bytes classed `a` and `b` (adjacent, `a`
/// first) into one word. `prev` is the class before `a` and `next` the
/// class after `b`, both [`WB_OTHER`] past the ends of the input — the
/// mid-joiner rules are the only context-sensitive ones and need
/// exactly one byte of lookaround on each side.
#[inline(always)]
fn wb_joined(prev: u8, a: u8, b: u8, next: u8) -> bool {
    // WB5 / WB8 / WB9 / WB10 — letters and digits run together.
    if a & WB_ALNUM != 0 && b & WB_ALNUM != 0 {
        return true;
    }
    // WB13a / WB13b — `_` binds to alphanumerics and to itself.
    if a & (WB_ALNUM | WB_EXTEND_NUM_LET) != 0 && b & WB_EXTEND_NUM_LET != 0 {
        return true;
    }
    if a & WB_EXTEND_NUM_LET != 0 && b & WB_ALNUM != 0 {
        return true;
    }
    // WB6 / WB7 — a single `:`/`.`/`'` between two letters (`don't`,
    // `a.b`). Doubling it breaks, because the lookaround then sees the
    // joiner rather than a letter.
    if a & WB_ALETTER != 0 && b & (WB_MID_LETTER | WB_MID_NUM_LET) != 0 && next & WB_ALETTER != 0 {
        return true;
    }
    if a & (WB_MID_LETTER | WB_MID_NUM_LET) != 0 && b & WB_ALETTER != 0 && prev & WB_ALETTER != 0 {
        return true;
    }
    // WB11 / WB12 — the digit twin (`3.14`, `1,000`).
    if a & WB_NUMERIC != 0 && b & (WB_MID_NUM | WB_MID_NUM_LET) != 0 && next & WB_NUMERIC != 0 {
        return true;
    }
    if a & (WB_MID_NUM | WB_MID_NUM_LET) != 0 && b & WB_NUMERIC != 0 && prev & WB_NUMERIC != 0 {
        return true;
    }
    // WB999 — break everywhere else.
    false
}

/// The joiner bytes — the classes whose effect depends on what sits
/// either side of them. A 16-byte window holding none of these and no
/// non-ASCII byte contains only `WB_ALETTER`, `WB_NUMERIC` and
/// `WB_OTHER`, and over just those classes UAX #29 collapses to
/// WB5/WB8/WB9/WB10: maximal runs of `[A-Za-z0-9]`, which is exactly
/// [`AsciiLowerTokenizer`]'s rule. That equivalence is what lets the
/// vector lane below drive such windows from bitmasks and leave only
/// the neighbourhood of a joiner to the byte-at-a-time path.
const WB_JOINER_BYTES: [u8; 6] = [b'_', b':', b',', b';', b'.', b'\''];

/// Low-16 mask with the bottom `k` bits set (`k <= 16`).
#[inline(always)]
fn low_mask(k: usize) -> u32 {
    ((1u32 << k) - 1) & LANE_BITMASK
}

/// Count of consecutive set bits starting at bit 0, capped at 16.
#[inline(always)]
fn leading_run(m: u32) -> usize {
    ((!m) & (LANE_BITMASK | 0x1_0000)).trailing_zeros() as usize
}

/// Segment all-ASCII `bytes` on UAX #29 word boundaries, calling
/// `f(start, end, has_upper)` for each segment that carries at least one
/// alphanumeric. Segments of pure punctuation are dropped, matching
/// `unicode_words`. `has_upper` reports whether the segment holds an
/// upper-case byte, so a caller can borrow instead of case-folding.
///
/// Byte ranges rather than `&str` slices: the caller owns the input and
/// can reslice it at whatever lifetime it needs, which keeps the
/// zero-copy query path free of any lifetime juggling.
///
/// Runs a 16-byte vector lane over windows free of joiners and
/// non-ASCII bytes (see [`WB_JOINER_BYTES`]) and the byte-at-a-time
/// [`ascii_word_segments_scalar`] elsewhere. The two are held
/// equivalent by an exhaustive differential test.
///
/// Callers must have established that the input is ASCII, so every index
/// is a codepoint boundary.
#[inline]
fn ascii_word_segments<F: FnMut(usize, usize, bool)>(bytes: &[u8], mut f: F) {
    const LANES: usize = 16;
    let n = bytes.len();
    // The segment under construction. `pure_alnum` tracks whether every
    // byte of it so far is `[A-Za-z0-9]`, which is the precondition for
    // handing the segment across into the vector lane: only then does
    // "continues iff the next byte is alphanumeric" describe it.
    let mut seg_start = 0usize;
    let mut has_alnum = false;
    let mut has_upper = false;
    let mut pure_alnum = true;
    let mut i = 0usize;

    while i < n {
        // The vector lane needs a full window, and needs the open
        // segment to be one it can reason about: either nothing is open,
        // or what is open is a plain alphanumeric run.
        let lane_ok = i + LANES <= n && (i == seg_start || (pure_alnum && has_alnum));
        if lane_ok {
            // SAFETY: `i + LANES <= n` checked above, so 16 bytes from
            // `bytes.as_ptr().add(i)` stay in bounds. The cast and deref
            // copy the array by value into the SIMD register.
            let arr: [u8; LANES] = unsafe { *(bytes.as_ptr().add(i) as *const [u8; LANES]) };
            let chunk = u8x16::from(arr);
            let is_digit = chunk.simd_ge(u8x16::splat(b'0')) & chunk.simd_le(u8x16::splat(b'9'));
            let is_upper = chunk.simd_ge(u8x16::splat(b'A')) & chunk.simd_le(u8x16::splat(b'Z'));
            let is_lower = chunk.simd_ge(u8x16::splat(b'a')) & chunk.simd_le(u8x16::splat(b'z'));
            let mut is_complex = (chunk & u8x16::splat(NON_ASCII_BYTE_MIN))
                .simd_eq(u8x16::splat(NON_ASCII_BYTE_MIN));
            for joiner in WB_JOINER_BYTES {
                is_complex |= chunk.simd_eq(u8x16::splat(joiner));
            }
            if (is_complex.to_bitmask() & LANE_BITMASK) == 0 {
                let alnum = (is_digit | is_upper | is_lower).to_bitmask() & LANE_BITMASK;
                let upper = is_upper.to_bitmask() & LANE_BITMASK;
                let mut consumed = 0usize;

                // Resolve the run handed in from the previous window.
                if i > seg_start {
                    if alnum & 1 == 0 {
                        // Byte `i` is neither alphanumeric nor a joiner,
                        // so it cannot extend the run: the segment ends.
                        f(seg_start, i, has_upper);
                        seg_start = i;
                        has_alnum = false;
                        has_upper = false;
                    } else {
                        let run = leading_run(alnum);
                        has_upper |= (upper & low_mask(run)) != 0;
                        if run == LANES {
                            // Still open at the window's end; carry on.
                            i += LANES;
                            continue;
                        }
                        f(seg_start, i + run, has_upper);
                        seg_start = i + run;
                        has_alnum = false;
                        has_upper = false;
                        consumed = run;
                    }
                }

                // Every further run lies wholly inside the window, so
                // the byte after it is a plain separator and no
                // lookahead past the window is needed — except for a run
                // touching the last byte, which is carried instead.
                let mut rest = alnum & !low_mask(consumed);
                let mut carried = false;
                while rest != 0 {
                    let start = rest.trailing_zeros() as usize;
                    let len = leading_run(rest >> start);
                    let end = start + len;
                    let run_upper = (upper >> start) & low_mask(len) != 0;
                    if end >= LANES {
                        seg_start = i + start;
                        has_alnum = true;
                        has_upper = run_upper;
                        pure_alnum = true;
                        carried = true;
                        break;
                    }
                    f(i + start, i + end, run_upper);
                    rest &= !low_mask(end);
                }
                if !carried {
                    seg_start = i + LANES;
                    has_alnum = false;
                    has_upper = false;
                    pure_alnum = true;
                }
                i += LANES;
                continue;
            }
        }

        // Byte-at-a-time step, identical in effect to the scalar
        // segmenter: decide whether byte `i` joins the previous one and
        // close the segment when it does not.
        let cur = wb_class(bytes[i]);
        if i > seg_start {
            let a = wb_class(bytes[i - 1]);
            let prev = if i >= 2 {
                wb_class(bytes[i - 2])
            } else {
                WB_OTHER
            };
            let next = if i + 1 < n {
                wb_class(bytes[i + 1])
            } else {
                WB_OTHER
            };
            if !wb_joined(prev, a, cur, next) {
                if has_alnum {
                    f(seg_start, i, has_upper);
                }
                seg_start = i;
                has_alnum = false;
                has_upper = false;
                pure_alnum = true;
            }
        }
        has_alnum |= cur & WB_ALNUM != 0;
        has_upper |= bytes[i].is_ascii_uppercase();
        pure_alnum &= bytes[i].is_ascii_alphanumeric();
        i += 1;
    }

    if has_alnum && seg_start < n {
        f(seg_start, n, has_upper);
    }
}

/// Byte-at-a-time reference for [`ascii_word_segments`], kept as the
/// implementation the vector lane is tested against. Same contract.
#[cfg(test)]
fn ascii_word_segments_scalar<F: FnMut(usize, usize, bool)>(bytes: &[u8], mut f: F) {
    let n = bytes.len();
    let mut seg_start = 0usize;
    let mut has_alnum = false;
    let mut has_upper = false;
    let mut prev = WB_OTHER;
    let mut a = WB_OTHER;
    for i in 0..n {
        let cur = wb_class(bytes[i]);
        if i > seg_start {
            let next = if i + 1 < n {
                wb_class(bytes[i + 1])
            } else {
                WB_OTHER
            };
            if !wb_joined(prev, a, cur, next) {
                if has_alnum {
                    f(seg_start, i, has_upper);
                }
                seg_start = i;
                has_alnum = false;
                has_upper = false;
            }
        }
        has_alnum |= cur & WB_ALNUM != 0;
        has_upper |= bytes[i].is_ascii_uppercase();
        prev = a;
        a = cur;
    }
    if has_alnum && seg_start < n {
        f(seg_start, n, has_upper);
    }
}

/// Unicode-aware tokenizer: UAX #29 word segmentation followed by full
/// Unicode lowercasing, preserving non-ASCII text. See the module-level
/// docs for the exact semantics and how it differs from
/// [`AsciiLowerTokenizer`].
#[derive(Debug, Clone, Copy, Default)]
pub struct StandardTokenizer;

impl StandardTokenizer {
    pub fn new() -> Self {
        Self
    }

    /// Monomorphized token scan — the same tokens
    /// [`Tokenizer::tokenize_each`] emits, but with a concrete `F` so
    /// LLVM can inline the callback into the scan loop instead of
    /// paying an indirect call per token. Every other entry point on
    /// this tokenizer routes through here, so no two of them can
    /// disagree about which tokens exist.
    ///
    /// All-ASCII input takes the table-driven ASCII word-break scan;
    /// anything else falls back to the general UAX #29 segmenter. The
    /// check is whole-input rather than per-chunk because a non-ASCII
    /// character changes how its ASCII neighbours segment (`aéb` is one
    /// word), so a chunk boundary could not be placed soundly.
    #[inline]
    pub fn tokenize_each_inline<F: FnMut(&str)>(&self, text: &str, mut f: F) {
        if text.is_ascii() {
            ascii_word_segments(text.as_bytes(), |start, end, has_upper| {
                let seg = &text[start..end];
                if has_upper {
                    // Cased ASCII only, so `to_ascii_lowercase` agrees
                    // with the Unicode fold the non-ASCII path applies.
                    f(&seg.to_ascii_lowercase());
                } else {
                    f(seg);
                }
            });
            return;
        }
        let mut buf = String::new();
        for word in text.unicode_words() {
            // Borrow directly when every cased character is already
            // lowercase (the common case for lowercased corpora); only
            // allocate to case-fold a word carrying an upper/title-case
            // letter. Non-alphabetic characters (digits, apostrophes)
            // are unaffected by lowercasing, so they never force a copy.
            if word.chars().all(|c| !c.is_alphabetic() || c.is_lowercase()) {
                f(word);
            } else {
                buf.clear();
                // Context-aware full-string lowercasing. `str::to_lowercase`
                // applies Unicode special-casing such as Final_Sigma (a
                // word-final `Σ` lowercases to `ς`, but to `σ` elsewhere);
                // a char-by-char fold has no word context and would emit
                // `σ` in both spots.
                buf.push_str(&word.to_lowercase());
                f(&buf);
            }
        }
    }
}

impl Tokenizer for StandardTokenizer {
    fn name(&self) -> &'static str {
        STANDARD_TOKENIZER
    }

    /// Collected rather than lazy: sharing one scan with the ingest
    /// path is what guarantees a term is indexed and queried under the
    /// same form, and the query strings this runs on are short.
    fn tokenize<'a>(&'a self, text: &'a str) -> Box<dyn Iterator<Item = String> + 'a> {
        let mut out = Vec::new();
        self.tokenize_each_inline(text, |t| out.push(t.to_owned()));
        Box::new(out.into_iter())
    }

    /// Trait-object dispatch path: delegates to the inherent
    /// [`tokenize_each_inline`](Self::tokenize_each_inline) so the body
    /// lives in one place. Callers holding a concrete
    /// [`StandardTokenizer`] (or downcasting a `&dyn Tokenizer` via
    /// [`Tokenizer::as_any`]) should call the inherent method directly
    /// to skip the per-token `&mut dyn FnMut(&str)` indirection.
    fn tokenize_each(&self, text: &str, f: &mut dyn FnMut(&str)) {
        self.tokenize_each_inline(text, |s| f(s));
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Zero-copy override for the query side: an already-lowercase
    /// all-ASCII token borrows from `text`; only a token needing a case
    /// fold is copied. Non-ASCII input keeps the general segmenter and
    /// its owned folds.
    fn tokenize_each_query<'q>(&self, text: &'q str, f: &mut dyn FnMut(Cow<'q, str>)) {
        if text.is_ascii() {
            ascii_word_segments(text.as_bytes(), |start, end, has_upper| {
                // `text` outlives the callback, so the reslice carries
                // the caller's `'q` with no lifetime gymnastics.
                let seg: &'q str = &text[start..end];
                if has_upper {
                    f(Cow::Owned(seg.to_ascii_lowercase()));
                } else {
                    f(Cow::Borrowed(seg));
                }
            });
            return;
        }
        self.tokenize_each_inline(text, |t| f(Cow::Owned(t.to_owned())));
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// A parsed query's phrases as plain term lists. `Phrase`'s own
    /// equality includes the offsets; these assertions are about which
    /// terms a phrase parsed to, and the offsets of a phrase with no
    /// analysis chain are always `0..n` (covered on its own below).
    fn phrase_terms<'q>(phrases: &'q [Phrase<Cow<'q, str>>]) -> Vec<Vec<&'q str>> {
        phrases
            .iter()
            .map(|p| p.iter().map(|t| &**t).collect())
            .collect()
    }

    fn tokens(text: &str) -> Vec<String> {
        AsciiLowerTokenizer.tokenize(text).collect()
    }

    /// Collect `(token, position)` pairs from the positional scan.
    fn positioned(text: &str) -> Vec<(String, u64)> {
        let mut out = Vec::new();
        AsciiLowerTokenizer
            .tokenize_each_inline_positioned(text, |tok, pos| out.push((tok.to_owned(), pos)));
        out
    }

    // ---- ASCII word-break fast path vs the general segmenter ----
    //
    // `standard` indexes through the ASCII scan and may be queried
    // through it too, so a single disagreement with `unicode_words`
    // would index a term under one tokenization and search it under
    // another. These verify the two exhaustively rather than by
    // sampling: the class table is small enough that full coverage of
    // short inputs is cheap, and every joiner rule is context-sensitive
    // over at most one byte on each side, so short inputs are where any
    // divergence has to show up.

    /// Segments from the byte-at-a-time reference, as owned strings.
    fn ascii_scalar_segments(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        ascii_word_segments_scalar(text.as_bytes(), |start, end, _| {
            out.push(text[start..end].to_owned());
        });
        out
    }

    /// Segments plus their `has_upper` flag, from both engines, so the
    /// borrow-versus-case-fold decision is compared too and not just the
    /// boundaries.
    fn ascii_fast_flags(text: &str) -> Vec<(usize, usize, bool)> {
        let mut out = Vec::new();
        ascii_word_segments(text.as_bytes(), |s, e, u| out.push((s, e, u)));
        out
    }
    fn ascii_scalar_flags(text: &str) -> Vec<(usize, usize, bool)> {
        let mut out = Vec::new();
        ascii_word_segments_scalar(text.as_bytes(), |s, e, u| out.push((s, e, u)));
        out
    }

    /// The vector lane runs 16 bytes at a time and hands partial runs
    /// across window boundaries, so a divergence from the reference
    /// would most likely appear only at a particular length or offset.
    /// Sweep every length across two full strides against a byte
    /// alphabet that includes each joiner, so windows land clean,
    /// dirty, and split across the boundary.
    #[test]
    fn simd_lane_matches_the_scalar_reference_across_window_boundaries() {
        const STRIDE: usize = 16;
        let alphabet = b"aB1_:,.' z9";
        // A deterministic pseudo-random walk over the alphabet gives
        // mixed windows; the fixed seed keeps a failure reproducible.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            alphabet[(state % alphabet.len() as u64) as usize]
        };
        for len in 0..=(3 * STRIDE + 3) {
            for _trial in 0..300 {
                let buf: Vec<u8> = (0..len).map(|_| next()).collect();
                let text = std::str::from_utf8(&buf).expect("ascii is utf8");
                assert_eq!(
                    ascii_fast_flags(text),
                    ascii_scalar_flags(text),
                    "vector lane diverged from the reference on {text:?}"
                );
            }
        }
    }

    /// Exhaustive where it matters: the five bytes straddling the
    /// 16-byte window boundary, over every arrangement of the class
    /// alphabet, against three different fillers and a range of
    /// lengths. Varying the whole string instead would be astronomically
    /// many cases for no extra coverage — the lane's only
    /// position-dependent behaviour is how it hands a run across that
    /// boundary, and these are the bytes that decide it.
    #[test]
    fn simd_lane_matches_the_scalar_reference_around_the_window_boundary() {
        const BOUNDARY: usize = 16;
        const VARY: usize = 5;
        const VARY_AT: usize = BOUNDARY - 2;
        let alphabet = b"a1_. ";
        for filler in [b'a', b' ', b'B'] {
            for len in (BOUNDARY + VARY)..=(BOUNDARY + VARY + 4) {
                let mut counter = [0usize; VARY];
                loop {
                    let mut buf = vec![filler; len];
                    for (k, &idx) in counter.iter().enumerate() {
                        buf[VARY_AT + k] = alphabet[idx];
                    }
                    let text = std::str::from_utf8(&buf).expect("ascii is utf8");
                    assert_eq!(
                        ascii_fast_flags(text),
                        ascii_scalar_flags(text),
                        "vector lane diverged from the reference on {text:?}"
                    );
                    let mut k = 0;
                    loop {
                        if k == VARY {
                            break;
                        }
                        counter[k] += 1;
                        if counter[k] < alphabet.len() {
                            break;
                        }
                        counter[k] = 0;
                        k += 1;
                    }
                    if k == VARY {
                        break;
                    }
                }
            }
        }
    }

    /// And the reference itself still agrees with the general
    /// segmenter, so the chain vector lane == reference == `unicode_words`
    /// is closed at both links.
    #[test]
    fn scalar_reference_matches_unicode_words_on_joiner_shapes() {
        for text in [
            "don't 3.14 a_b A:B 1,000 x''y a..b",
            "_lead trail_ a1_2b 3.14.15 1,000,000",
            "the quick brown fox jumps over the lazy dog again and again",
        ] {
            assert_eq!(
                ascii_scalar_segments(text),
                unicode_segments(text),
                "on {text:?}"
            );
        }
    }

    /// Segments the fast path yields, as owned strings.
    fn ascii_fast_segments(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        ascii_word_segments(text.as_bytes(), |start, end, _| {
            out.push(text[start..end].to_owned());
        });
        out
    }

    /// What the general UAX #29 segmenter yields for the same input.
    fn unicode_segments(text: &str) -> Vec<String> {
        text.unicode_words().map(str::to_owned).collect()
    }

    /// One byte per Word_Break class the ASCII table can produce, plus
    /// the shapes that stress the lookaround: letters, a digit, `_`,
    /// `:`, `,`, `.`, `'`, a plain separator, and a space.
    const WB_ALPHABET: &[u8] = b"aB1_:,.' ";

    #[test]
    fn ascii_fast_path_matches_unicode_words_for_every_short_input() {
        // All 128 ASCII bytes at lengths 1..=3. The mid-joiner rules
        // look at one byte either side of a boundary, so a three-byte
        // window covers every rule's full context.
        let mut buf = [0u8; 3];
        for len in 1..=3usize {
            let mut counter = vec![0u8; len];
            loop {
                buf[..len].copy_from_slice(&counter[..len]);
                let text = std::str::from_utf8(&buf[..len]).expect("ascii is utf8");
                assert_eq!(
                    ascii_fast_segments(text),
                    unicode_segments(text),
                    "segmentation diverged on {text:?} (bytes {:?})",
                    &buf[..len]
                );
                // Odometer over 0..128 in each position.
                let mut i = 0;
                loop {
                    if i == len {
                        break;
                    }
                    counter[i] += 1;
                    if counter[i] < 128 {
                        break;
                    }
                    counter[i] = 0;
                    i += 1;
                }
                if i == len {
                    break;
                }
            }
        }
    }

    #[test]
    fn ascii_fast_path_matches_unicode_words_for_longer_class_words() {
        // Longer inputs over one representative byte per class, up to
        // length 6 — chained joiners (`a.b.c`, `1,000,000`, `a__b`) and
        // the doubling that must break.
        let alpha = WB_ALPHABET;
        for len in 4..=6usize {
            let mut counter = vec![0usize; len];
            let mut buf = vec![0u8; len];
            loop {
                for (slot, &idx) in buf.iter_mut().zip(counter.iter()) {
                    *slot = alpha[idx];
                }
                let text = std::str::from_utf8(&buf).expect("ascii is utf8");
                assert_eq!(
                    ascii_fast_segments(text),
                    unicode_segments(text),
                    "segmentation diverged on {text:?}"
                );
                let mut i = 0;
                loop {
                    if i == len {
                        break;
                    }
                    counter[i] += 1;
                    if counter[i] < alpha.len() {
                        break;
                    }
                    counter[i] = 0;
                    i += 1;
                }
                if i == len {
                    break;
                }
            }
        }
    }

    #[test]
    fn standard_tokens_match_the_general_segmenter_on_mixed_text() {
        // The whole-input ASCII check must not change what `standard`
        // emits: an input with any non-ASCII byte takes the general
        // segmenter, and an all-ASCII one must agree with it.
        for text in [
            "the quick brown fox",
            "Don't stop 3.14 or 1,000,000 things",
            "snake_case and _leading and trailing_",
            "a.b.c A:B 1;2 x''y z--w",
            "café résumé naïve Straße",
            "mixed café and plain ascii 42",
            "中文 text with ascii",
            "",
            "   ",
            "___",
            "!!!",
        ] {
            let mut fast = Vec::new();
            StandardTokenizer.tokenize_each_inline(text, |t| fast.push(t.to_owned()));
            let expected: Vec<String> = text.unicode_words().map(|w| w.to_lowercase()).collect();
            assert_eq!(fast, expected, "tokens diverged on {text:?}");
        }
    }

    #[test]
    fn standard_entry_points_agree_with_each_other() {
        // `tokenize`, `tokenize_each` and `tokenize_each_query` all
        // route through one scan; this pins that they stay in lockstep,
        // since a divergence between the index and query paths is
        // silent recall loss rather than a visible failure.
        for text in [
            "Don't PANIC 3.14",
            "snake_case Mixed CASE",
            "café AU lait",
            "a.b,c 1.2,3",
        ] {
            let via_tokenize: Vec<String> = StandardTokenizer.tokenize(text).collect();
            let mut via_each = Vec::new();
            StandardTokenizer.tokenize_each(text, &mut |t| via_each.push(t.to_owned()));
            let mut via_query = Vec::new();
            StandardTokenizer.tokenize_each_query(text, &mut |t| via_query.push(t.into_owned()));
            assert_eq!(
                via_tokenize, via_each,
                "tokenize vs tokenize_each on {text:?}"
            );
            assert_eq!(
                via_tokenize, via_query,
                "tokenize vs query path on {text:?}"
            );
        }
    }

    /// Whether the analyzer picks the ASCII scan is decided for the
    /// *whole input*, so the same word takes a different code path
    /// depending on what else the text contains: `"don't"` alone goes
    /// through the ASCII scan, `"don't café"` through the general
    /// segmenter. Documents are indexed one way and queries tokenized
    /// the other, so if the two paths disagreed on a shared word the
    /// query would silently miss — no error, just lost recall. Pin that
    /// a word's tokens do not depend on its neighbours' encoding.
    #[test]
    fn a_words_tokens_do_not_depend_on_non_ascii_elsewhere_in_the_text() {
        for ascii_text in [
            "don't stop",
            "3.14 and 1,000",
            "snake_case _leading trailing_",
            "a.b.c A:B x''y",
            "plain words here",
            "MiXeD CaSe WORDS",
            "hello",
        ] {
            // Fast path: the text is entirely ASCII.
            let fast: Vec<String> = StandardTokenizer.tokenize(ascii_text).collect();
            assert!(ascii_text.is_ascii(), "fixture must be ASCII");

            // Slow path: the identical text plus one non-ASCII word, so
            // the whole input falls back to the general segmenter. The
            // ASCII words' tokens must come out the same.
            for suffix in [" café", " Straße", " 中文"] {
                let mixed = format!("{ascii_text}{suffix}");
                assert!(!mixed.is_ascii(), "fixture must leave the ASCII path");
                let slow: Vec<String> = StandardTokenizer.tokenize(&mixed).collect();
                assert_eq!(
                    slow[..fast.len()],
                    fast[..],
                    "{ascii_text:?} tokenized differently once {suffix:?} joined the text"
                );
            }
        }
    }

    /// The ASCII scan runs 16 bytes at a time with a scalar tail for the
    /// remainder, so a token's bytes can be split across the vector and
    /// scalar paths depending only on where it sits in the buffer. Pad
    /// the front through a full stride so every token visits both, and
    /// require the tokens to be identical each time.
    #[test]
    fn ascii_scan_is_independent_of_alignment_across_the_simd_stride() {
        const STRIDE: usize = 16;
        for text in [
            "don't 3.14 snake_case a.b.c MiXeD",
            "the quick brown fox jumps over the lazy dog",
            "a:b 1,000 x_y Z",
        ] {
            let baseline: Vec<String> = StandardTokenizer.tokenize(text).collect();
            let ascii_baseline: Vec<String> = AsciiLowerTokenizer.tokenize(text).collect();
            // One extra stride so the tail length cycles through every
            // residue class mod 16.
            for pad in 1..=STRIDE + 1 {
                let padded = format!("{}{}", " ".repeat(pad), text);
                assert_eq!(
                    StandardTokenizer.tokenize(&padded).collect::<Vec<_>>(),
                    baseline,
                    "standard tokens shifted at pad {pad} for {text:?}"
                );
                assert_eq!(
                    AsciiLowerTokenizer.tokenize(&padded).collect::<Vec<_>>(),
                    ascii_baseline,
                    "ascii_lower tokens shifted at pad {pad} for {text:?}"
                );
            }
        }
    }

    /// Characters spanning every ASCII word-break class plus non-ASCII
    /// letters from three scripts, so a generated string lands on either
    /// code path and on the boundaries between them.
    const MIXED_ALPHABET: &[char] = &[
        'a', 'B', 'z', '1', '9', '_', ':', ',', ';', '.', '\'', ' ', '-', '"', 'é', 'Ä', 'ß', 'Σ',
        '中', '数',
    ];

    proptest! {
        /// Both code paths, against one oracle, for arbitrary mixed
        /// input: whatever the analyzer emits must equal the general
        /// UAX #29 segmenter's words, lowercased. Holding for every
        /// input means the ASCII scan and the fallback also agree with
        /// each other, which is the property the index and query sides
        /// depend on.
        #[test]
        fn standard_tokens_match_the_general_segmenter_for_any_mixed_input(
            indices in proptest::collection::vec(0..MIXED_ALPHABET.len(), 0..52)
        ) {
            let text: String = indices.iter().map(|&i| MIXED_ALPHABET[i]).collect();
            let mut got = Vec::new();
            StandardTokenizer.tokenize_each_inline(&text, |t| got.push(t.to_owned()));
            let expected: Vec<String> =
                text.unicode_words().map(|w| w.to_lowercase()).collect();
            prop_assert_eq!(&got, &expected, "tokens diverged on {:?}", text);
        }

        /// Every entry point on the analyzer must agree on arbitrary
        /// mixed input. Documents are indexed through `tokenize_each`
        /// and queries run through `tokenize` / `tokenize_each_query`,
        /// so a divergence here is lost recall rather than a failure.
        #[test]
        fn standard_entry_points_agree_for_any_mixed_input(
            indices in proptest::collection::vec(0..MIXED_ALPHABET.len(), 0..52)
        ) {
            let text: String = indices.iter().map(|&i| MIXED_ALPHABET[i]).collect();
            let via_tokenize: Vec<String> = StandardTokenizer.tokenize(&text).collect();
            let mut via_each = Vec::new();
            StandardTokenizer.tokenize_each(&text, &mut |t| via_each.push(t.to_owned()));
            let mut via_query = Vec::new();
            StandardTokenizer.tokenize_each_query(&text, &mut |t| via_query.push(t.into_owned()));
            prop_assert_eq!(&via_tokenize, &via_each, "tokenize vs tokenize_each on {:?}", text);
            prop_assert_eq!(&via_tokenize, &via_query, "tokenize vs query path on {:?}", text);
        }
    }

    // ---- StandardTokenizer (Unicode-aware) ----

    /// Tokens via the trait `tokenize` path.
    fn std_tokens(text: &str) -> Vec<String> {
        StandardTokenizer.tokenize(text).collect()
    }

    /// Tokens via the `tokenize_each` borrowing path — must agree with
    /// `tokenize` on which tokens are emitted.
    fn std_tokens_each(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        StandardTokenizer.tokenize_each(text, &mut |t| out.push(t.to_owned()));
        out
    }

    #[test]
    fn standard_lowercases_and_splits_ascii() {
        assert_eq!(
            std_tokens("Rust Async Runtime"),
            vec!["rust", "async", "runtime"]
        );
        assert_eq!(
            std_tokens_each("Rust Async Runtime"),
            vec!["rust", "async", "runtime"]
        );
    }

    #[test]
    fn standard_keeps_non_ascii_lowercased() {
        // The key divergence from AsciiLowerTokenizer, which drops these.
        assert_eq!(std_tokens("Café RÉSUMÉ"), vec!["café", "résumé"]);
        assert_eq!(std_tokens_each("Café RÉSUMÉ"), vec!["café", "résumé"]);
        // AsciiLowerTokenizer drops the same tokens entirely.
        assert_eq!(
            AsciiLowerTokenizer
                .tokenize("Café RÉSUMÉ")
                .collect::<Vec<_>>(),
            Vec::<String>::new()
        );
    }

    #[test]
    fn standard_splits_cjk_per_ideograph() {
        // UAX #29 treats each CJK ideograph as its own word.
        assert_eq!(std_tokens("日本語"), vec!["日", "本", "語"]);
    }

    #[test]
    fn standard_keeps_intra_word_numeric_and_apostrophe() {
        // UAX #29 keeps a decimal number and a mid-word apostrophe together.
        assert_eq!(std_tokens("pi is 3.14"), vec!["pi", "is", "3.14"]);
        assert_eq!(std_tokens("don't stop"), vec!["don't", "stop"]);
    }

    #[test]
    fn standard_splits_on_hyphen_and_drops_punctuation() {
        assert_eq!(std_tokens("wi-fi, hello!"), vec!["wi", "fi", "hello"]);
        assert_eq!(std_tokens("...   ???"), Vec::<String>::new());
    }

    #[test]
    fn standard_borrow_and_copy_paths_agree() {
        // Mixed already-lower, needs-fold, digit, and non-ASCII tokens:
        // the borrow fast path and the copy path must emit the same set.
        let text = "alpha Beta 42 gamma2 Δelta";
        assert_eq!(std_tokens(text), std_tokens_each(text));
    }

    #[test]
    fn standard_copy_path_lowercases_final_sigma_like_tokenize() {
        // Regression: the copy path must lowercase with context-aware
        // `str::to_lowercase`, not char-by-char. A word-final capital `Σ`
        // folds to `ς` (final sigma) but to `σ` elsewhere; a char-by-char
        // fold has no word context and would emit `σ` in both places. Text
        // is indexed through `tokenize_each` and queried through
        // `tokenize`, so any divergence stores a Greek term under one form
        // and searches it under another — the document becomes unfindable.
        let text = "ΟΔΟΣ"; // one Greek word; the final Σ must fold to ς
        // Both tokenizer paths agree, and both equal Rust's context-aware
        // folding (which the char-by-char version would not).
        assert_eq!(std_tokens(text), std_tokens_each(text));
        assert_eq!(std_tokens_each(text), vec![text.to_lowercase()]);
        assert!(
            std_tokens_each(text)[0].ends_with('ς'),
            "word-final Σ must fold to final sigma ς, not σ"
        );
    }

    #[test]
    fn standard_empty_and_whitespace_yield_nothing() {
        assert_eq!(std_tokens(""), Vec::<String>::new());
        assert_eq!(std_tokens("   \t\n"), Vec::<String>::new());
    }

    #[test]
    fn standard_query_parse_keeps_non_ascii_and_sigils() {
        // Query-side tokenization (via the default `tokenize_each_query`
        // → `parse`) must keep non-ASCII and honor +/- clause sigils.
        let p = StandardTokenizer.parse("Café -Résumé +Ötzi");
        assert_eq!(p.positives, vec!["café"]);
        assert_eq!(p.negatives, vec!["résumé"]);
        assert_eq!(p.musts, vec!["ötzi"]);
    }

    #[test]
    fn tokenizer_for_name_resolves_known_and_rejects_unknown() {
        assert!(tokenizer_for_name(ASCII_LOWER_TOKENIZER).is_some());
        assert!(tokenizer_for_name(STANDARD_TOKENIZER).is_some());
        assert!(tokenizer_for_name("nonesuch").is_none());
        // The resolved standard tokenizer keeps non-ASCII through the
        // trait object, confirming the right impl is wired.
        let tok = tokenizer_for_name(STANDARD_TOKENIZER).expect("standard");
        assert_eq!(tok.tokenize("Café").collect::<Vec<_>>(), vec!["café"]);
    }

    /// A phrase's offsets on a chainless tokenizer are its terms'
    /// indices, and the parsed value equals the `adjacent` fixture the
    /// tests build — nothing about *how* a phrase was assembled
    /// survives into it, so two phrases with the same terms and the
    /// same spacing are the same phrase.
    #[test]
    fn a_chainless_phrase_is_adjacent_and_compares_equal_to_the_fixture() {
        let p = StandardTokenizer.parse("\"new york city\"");
        let want = Phrase::adjacent(vec![
            Cow::Borrowed("new"),
            Cow::Borrowed("york"),
            Cow::Borrowed("city"),
        ]);
        assert_eq!(p.positive_phrases, vec![want]);
        assert_eq!(p.positive_phrases[0].offsets(), &[0, 1, 2]);
    }

    /// A tokenizer that drops tokens reports holes, and the parser
    /// turns them into offsets normalized to the first surviving term.
    /// `ascii_lower` drops non-ASCII runs, which is a hole with no
    /// analysis chain involved — so the offset plumbing is exercised
    /// by the tokenizer that has always left gaps.
    #[test]
    fn a_phrase_carries_the_holes_its_tokenizer_left() {
        // `café` is dropped whole and consumes one ordinal, so `york`
        // sits two positions after `new`.
        let p = AsciiLowerTokenizer.parse("\"new café york\"");
        assert_eq!(phrase_terms(&p.positive_phrases), vec![vec!["new", "york"]]);
        assert_eq!(p.positive_phrases[0].offsets(), &[0, 2]);
        // A *leading* dropped run is unobservable: there is nothing
        // before it to space the phrase from, so the offsets still
        // start at 0.
        let p = AsciiLowerTokenizer.parse("\"café new york\"");
        assert_eq!(p.positive_phrases[0].offsets(), &[0, 1]);
    }

    #[test]
    fn positioned_leaves_a_gap_for_dropped_runs() {
        // No drops: positions are dense emission ordinals, and the
        // positioned scan emits exactly the same tokens as the plain
        // one (delegation guarantees this).
        assert_eq!(
            positioned("the quick brown fox"),
            vec![
                ("the".into(), 0),
                ("quick".into(), 1),
                ("brown".into(), 2),
                ("fox".into(), 3),
            ],
        );

        // A dropped (non-ASCII) run consumes an ordinal but emits no
        // token, so the neighbours are NOT adjacent: `york` is at 2,
        // not 1 — this is what stops `"new york"` matching this text.
        assert_eq!(
            positioned("new café york"),
            vec![("new".into(), 0), ("york".into(), 2)],
        );
        // Gap whether the dropped word leads, trails, or sits between.
        assert_eq!(
            positioned("café new york"),
            vec![("new".into(), 1), ("york".into(), 2)],
        );
        assert_eq!(
            positioned("new york café"),
            vec![("new".into(), 0), ("york".into(), 1)],
        );

        // The plain scan still emits the same tokens (positions dropped).
        let mut plain = Vec::new();
        AsciiLowerTokenizer.tokenize_each_inline("new café york", |t| plain.push(t.to_owned()));
        assert_eq!(plain, vec!["new".to_string(), "york".to_string()]);
    }

    #[test]
    fn empty_input_yields_nothing() {
        assert_eq!(tokens(""), Vec::<String>::new());
    }

    #[test]
    fn whitespace_only_yields_nothing() {
        assert_eq!(tokens("   \t\n\r"), Vec::<String>::new());
    }

    #[test]
    fn single_token_lowercased() {
        assert_eq!(tokens("Hello"), vec!["hello"]);
    }

    #[test]
    fn unique_tokens_dedups_and_sorts_across_values() {
        let tok = AsciiLowerTokenizer;
        // values share 'juice'; result is one sorted set, no repeat
        let got = unique_tokens(&tok, ["Orange Juice", "Apple Juice"]);
        assert_eq!(got, vec!["apple", "juice", "orange"]);
    }

    #[test]
    fn multiple_tokens_split_on_whitespace() {
        assert_eq!(
            tokens("Rust async runtime"),
            vec!["rust", "async", "runtime"]
        );
    }

    #[test]
    fn punctuation_splits_tokens() {
        assert_eq!(
            tokens("hello,world!foo;bar.baz?"),
            vec!["hello", "world", "foo", "bar", "baz"]
        );
    }

    #[test]
    fn case_folding_applies_to_uppercase_only() {
        assert_eq!(tokens("ABC abc XyZ"), vec!["abc", "abc", "xyz"]);
    }

    #[test]
    fn alphanumerics_kept_together() {
        assert_eq!(tokens("foo123 bar456"), vec!["foo123", "bar456"]);
    }

    #[test]
    fn pure_numeric_tokens_kept() {
        assert_eq!(tokens("404 200 500"), vec!["404", "200", "500"]);
    }

    #[test]
    fn underscore_is_a_separator_in_v1() {
        // `_` is not in `[A-Za-z0-9]` — it splits tokens. v2 may revisit.
        assert_eq!(tokens("foo_bar"), vec!["foo", "bar"]);
    }

    #[test]
    fn dash_is_a_separator() {
        assert_eq!(tokens("rust-async"), vec!["rust", "async"]);
    }

    #[test]
    fn non_ascii_token_is_dropped() {
        // ASCII-only tokenizer: "café" has a non-ASCII byte, so the
        // entire token is dropped.
        assert_eq!(tokens("café"), Vec::<String>::new());
    }

    #[test]
    fn non_ascii_token_drops_only_that_token() {
        // Surrounding ASCII tokens still come through.
        assert_eq!(tokens("hello café world"), vec!["hello", "world"]);
    }

    #[test]
    fn cjk_input_yields_nothing() {
        assert_eq!(tokens("日本語"), Vec::<String>::new());
    }

    #[test]
    fn emoji_input_yields_nothing() {
        assert_eq!(tokens("hello 🚀 world"), vec!["hello", "world"]);
    }

    #[test]
    fn multiple_consecutive_separators_are_collapsed() {
        assert_eq!(tokens("foo,,,bar"), vec!["foo", "bar"]);
        assert_eq!(tokens("foo   bar"), vec!["foo", "bar"]);
    }

    #[test]
    fn leading_and_trailing_separators_are_skipped() {
        assert_eq!(tokens("  foo bar  "), vec!["foo", "bar"]);
        assert_eq!(tokens("...foo..."), vec!["foo"]);
    }

    #[test]
    fn tokenizer_is_send_and_sync() {
        // Compile-time assertion via the Tokenizer trait bound.
        fn is_send_sync<T: Send + Sync>() {}
        is_send_sync::<AsciiLowerTokenizer>();
    }

    #[test]
    fn tokenizer_used_via_dyn_trait() {
        // The trait object form is what the FtsBuilder will hold.
        let tok: Box<dyn Tokenizer> = Box::new(AsciiLowerTokenizer);
        let v: Vec<String> = tok.tokenize("Hello WORLD").collect();
        assert_eq!(v, vec!["hello", "world"]);
    }

    #[test]
    fn stress_long_input_does_not_panic() {
        // Rough scale-test: 1 MB of pseudo-text.
        let chunk = "lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
        let big = chunk.repeat(20_000);
        let count = AsciiLowerTokenizer.tokenize(&big).count();
        // 8 tokens per chunk × 20_000 = 160_000.
        assert_eq!(count, 8 * 20_000);
    }

    // ---- parse (the `-` negation sigil) ----

    fn parse(query: &str) -> ParsedQuery<'_> {
        AsciiLowerTokenizer.parse(query)
    }

    #[test]
    fn parse_default_trait_impl_matches_override() {
        // A tokenizer that overrides nothing gets the same split via
        // the default `parse` impl (owned tokens).
        #[derive(Debug)]
        struct PlainTok;
        impl Tokenizer for PlainTok {
            fn name(&self) -> &'static str {
                "plain_test"
            }
            fn tokenize<'a>(&'a self, text: &'a str) -> Box<dyn Iterator<Item = String> + 'a> {
                AsciiLowerTokenizer.tokenize(text)
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
        let p = PlainTok.parse("Rust -PYTHON");
        assert_eq!(p.positives, vec!["rust"]);
        assert_eq!(p.negatives, vec!["python"]);
        assert!(matches!(p.positives[0], Cow::Owned(_)));
    }

    #[test]
    fn parse_positives_only() {
        let p = parse("rust async");
        assert_eq!(p.positives, vec!["rust", "async"]);
        assert!(p.negatives.is_empty());
    }

    #[test]
    fn parse_single_negative() {
        let p = parse("rust -python");
        assert_eq!(p.positives, vec!["rust"]);
        assert_eq!(p.negatives, vec!["python"]);
    }

    #[test]
    fn parse_multiple_negatives() {
        let p = parse("rust async -python -php");
        assert_eq!(p.positives, vec!["rust", "async"]);
        assert_eq!(p.negatives, vec!["python", "php"]);
    }

    #[test]
    fn parse_negation_only() {
        // No positive clause — the parser reports it faithfully; the
        // caller turns this into an error.
        let p = parse("-python");
        assert!(p.positives.is_empty());
        assert_eq!(p.negatives, vec!["python"]);
    }

    #[test]
    fn parse_interior_hyphen_is_not_negation() {
        // `a-b` is one run with an interior `-`; the scan splits it
        // into two positive tokens. Nothing is negated.
        let p = parse("a-b");
        assert_eq!(p.positives, vec!["a", "b"]);
        assert!(p.negatives.is_empty());
    }

    #[test]
    fn parse_bare_dash_contributes_nothing() {
        let p = parse("rust - python");
        assert_eq!(p.positives, vec!["rust", "python"]);
        assert!(p.negatives.is_empty());
    }

    #[test]
    fn parse_double_dash_strips_one_then_tokenizes() {
        // `--py`: strip the one leading `-`, leaving `-py`; the scan
        // drops the remaining `-` and yields `py`.
        let p = parse("--py");
        assert!(p.positives.is_empty());
        assert_eq!(p.negatives, vec!["py"]);
    }

    #[test]
    fn parse_negated_term_is_normalized() {
        // The negated side is lower-cased like the index.
        let p = parse("rust -PYTHON");
        assert_eq!(p.negatives, vec!["python"]);
    }

    #[test]
    fn parse_empty_query() {
        let p = parse("");
        assert!(p.musts.is_empty());
        assert!(p.positives.is_empty());
        assert!(p.negatives.is_empty());
    }

    // ---- parse (the `+` must sigil) ----

    #[test]
    fn parse_must_sigil() {
        let p = parse("+climate policy");
        assert_eq!(p.musts, vec!["climate"]);
        assert_eq!(p.positives, vec!["policy"]);
        assert!(p.negatives.is_empty());
    }

    #[test]
    fn parse_all_must() {
        let p = parse("+griffith +observatory");
        assert_eq!(p.musts, vec!["griffith", "observatory"]);
        assert!(p.positives.is_empty());
    }

    #[test]
    fn parse_must_with_negation() {
        let p = parse("+python -snake -monty");
        assert_eq!(p.musts, vec!["python"]);
        assert!(p.positives.is_empty());
        assert_eq!(p.negatives, vec!["snake", "monty"]);
    }

    #[test]
    fn parse_interior_plus_is_not_must() {
        // `a+b` is one run with an interior `+`; the scan splits it
        // into two bare tokens. Nothing is a must clause.
        let p = parse("a+b");
        assert!(p.musts.is_empty());
        assert_eq!(p.positives, vec!["a", "b"]);
    }

    #[test]
    fn parse_bare_plus_contributes_nothing() {
        let p = parse("rust + python");
        assert!(p.musts.is_empty());
        assert_eq!(p.positives, vec!["rust", "python"]);
    }

    #[test]
    fn parse_must_term_is_normalized() {
        // The must side is lower-cased like the index.
        let p = parse("+RUST async");
        assert_eq!(p.musts, vec!["rust"]);
        assert_eq!(p.positives, vec!["async"]);
    }

    #[test]
    fn parse_minus_wins_over_plus_ordering() {
        // `-` is checked first, so `-+x` negates (strip `-`, the scan
        // drops the `+`); `+-x` is a must (strip `+`, scan drops `-`).
        let p = parse("-+x");
        assert_eq!(p.negatives, vec!["x"]);
        let p = parse("+-x");
        assert_eq!(p.musts, vec!["x"]);
    }

    // ---- parse (quoted phrase atoms) ----

    #[test]
    fn parse_pure_phrase() {
        let p = parse(r#""griffith observatory""#);
        assert_eq!(
            phrase_terms(&p.positive_phrases),
            vec![vec!["griffith", "observatory"]]
        );
        assert!(p.positives.is_empty());
        assert!(p.musts.is_empty());
    }

    #[test]
    fn parse_phrase_polarities() {
        let p = parse(r#"+"the who" -"memory unsafe" "new york""#);
        assert_eq!(phrase_terms(&p.must_phrases), vec![vec!["the", "who"]]);
        assert_eq!(
            phrase_terms(&p.negative_phrases),
            vec![vec!["memory", "unsafe"]]
        );
        assert_eq!(phrase_terms(&p.positive_phrases), vec![vec!["new", "york"]]);
    }

    #[test]
    fn parse_phrase_mixes_with_terms() {
        let p = parse(r#"+"the who" +uk rust -python"#);
        assert_eq!(phrase_terms(&p.must_phrases), vec![vec!["the", "who"]]);
        assert_eq!(p.musts, vec!["uk"]);
        assert_eq!(p.positives, vec!["rust"]);
        assert_eq!(p.negatives, vec!["python"]);
    }

    #[test]
    fn parse_single_token_phrase_degrades_to_term() {
        let p = parse(r#""york" +"london" -"paris""#);
        assert!(p.positive_phrases.is_empty());
        assert!(p.must_phrases.is_empty());
        assert!(p.negative_phrases.is_empty());
        assert_eq!(p.positives, vec!["york"]);
        assert_eq!(p.musts, vec!["london"]);
        assert_eq!(p.negatives, vec!["paris"]);
    }

    #[test]
    fn parse_empty_quotes_contribute_nothing() {
        let p = parse(r#"rust "" async"#);
        assert_eq!(p.positives, vec!["rust", "async"]);
        assert!(p.positive_phrases.is_empty());
    }

    #[test]
    fn parse_unbalanced_quote_is_whitespace() {
        // The dangling quote splits the text; everything parses as
        // bare terms (lenient, never an error).
        let p = parse(r#"rust "new york"#);
        assert_eq!(p.positives, vec!["rust", "new", "york"]);
        assert!(p.positive_phrases.is_empty());
    }

    #[test]
    fn parse_phrase_tokens_are_normalized() {
        // Phrase innards run through the same tokenizer: lowercased,
        // punctuation split.
        let p = parse(r#""New-York City""#);
        assert_eq!(
            phrase_terms(&p.positive_phrases),
            vec![vec!["new", "york", "city"]]
        );
    }

    #[test]
    fn parse_interior_sigil_before_quote_is_not_polarity() {
        // `abc+"x y"`: the `+` is interior to the run, not a phrase
        // sigil — the phrase is bare and `abc` parses from the
        // unquoted segment (its trailing `+` strips as punctuation).
        let p = parse(r#"abc+"x y""#);
        assert_eq!(p.positives, vec!["abc"]);
        assert_eq!(phrase_terms(&p.positive_phrases), vec![vec!["x", "y"]]);
        assert!(p.must_phrases.is_empty());
    }

    #[test]
    fn parse_adjacent_phrases() {
        let p = parse(r#""a b""c d""#);
        assert_eq!(
            phrase_terms(&p.positive_phrases),
            vec![vec!["a", "b"], vec!["c", "d"]]
        );
    }

    #[test]
    fn into_clauses_resolves_phrase_polarity_by_mode() {
        let c = parse(r#""new york" +"the who" -"bad seq" rust"#).into_clauses(BoolMode::Or);
        assert_eq!(phrase_terms(&c.should_phrases), vec![vec!["new", "york"]]);
        assert_eq!(phrase_terms(&c.must_phrases), vec![vec!["the", "who"]]);
        assert_eq!(phrase_terms(&c.negative_phrases), vec![vec!["bad", "seq"]]);
        assert_eq!(c.shoulds, vec!["rust"]);

        let c = parse(r#""new york" rust"#).into_clauses(BoolMode::And);
        assert_eq!(phrase_terms(&c.must_phrases), vec![vec!["new", "york"]]);
        assert!(c.should_phrases.is_empty());
        assert_eq!(c.musts, vec!["rust"]);
    }

    // ---- into_clauses (default-operator resolution) ----

    #[test]
    fn into_clauses_or_maps_bare_to_should() {
        let c = parse("+climate policy -spam").into_clauses(BoolMode::Or);
        assert_eq!(c.musts, vec!["climate"]);
        assert_eq!(c.shoulds, vec!["policy"]);
        assert_eq!(c.negatives, vec!["spam"]);
    }

    #[test]
    fn into_clauses_and_folds_bare_into_musts() {
        let c = parse("+climate policy -spam").into_clauses(BoolMode::And);
        assert_eq!(c.musts, vec!["climate", "policy"]);
        assert!(c.shoulds.is_empty());
        assert_eq!(c.negatives, vec!["spam"]);
    }

    #[test]
    fn into_clauses_legacy_shapes_unchanged() {
        // Sigil-less queries resolve exactly as the pre-clause model:
        // Or ⇒ all shoulds (union), And ⇒ all musts (intersection).
        let c = parse("rust async").into_clauses(BoolMode::Or);
        assert!(c.musts.is_empty());
        assert_eq!(c.shoulds, vec!["rust", "async"]);

        let c = parse("rust async").into_clauses(BoolMode::And);
        assert_eq!(c.musts, vec!["rust", "async"]);
        assert!(c.shoulds.is_empty());
    }

    #[test]
    fn parse_lowercase_tokens_borrow_the_query() {
        // Zero-copy contract: already-lowercase runs must not allocate.
        let p = parse("rust -python");
        assert!(matches!(p.positives[0], Cow::Borrowed(_)));
        assert!(matches!(p.negatives[0], Cow::Borrowed(_)));
    }

    #[test]
    fn parse_uppercase_token_is_the_only_copy() {
        let p = parse("rust -PYTHON");
        assert!(matches!(p.positives[0], Cow::Borrowed(_)));
        assert!(matches!(p.negatives[0], Cow::Owned(_)));
    }

    /// The `new` constructor plus the trait-object `tokenize_each`
    /// dispatch path (distinct from the inherent `tokenize_each_inline`
    /// the hot path uses). Mixed case and punctuation confirm the
    /// lowercasing + separator splitting.
    #[test]
    fn dyn_tokenize_each_lowercases_and_splits() {
        let tok = AsciiLowerTokenizer::new();
        let dynt: &dyn Tokenizer = &tok;
        let mut out = Vec::new();
        dynt.tokenize_each("Hello, World rust", &mut |s| out.push(s.to_string()));
        assert_eq!(out, vec!["hello", "world", "rust"]);
    }
}
