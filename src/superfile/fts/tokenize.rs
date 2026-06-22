// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Tokenization, plus BM25 query parsing ([`Tokenizer::parse`]).
//! The parser lives here because the `-` negation sigil must be
//! handled before tokenizing — the tokenizer splits on `-`.
//!
//! Ships one tokenizer: [`AsciiLowerTokenizer`]. The [`Tokenizer`]
//! trait is the extension point for ICU / language-aware stemmers /
//! custom char filters under the same trait without touching FTS
//! code.
//!
//! Semantics:
//!   - Split on any byte that isn't `[A-Za-z0-9]`.
//!   - Lowercase each ASCII letter (bytes `b'A'..=b'Z'` → `b'a'..=b'z'`).
//!   - Drop any token that contains a non-ASCII byte (high-bit set).
//!     Non-ASCII tokens are silently dropped (not an error) — the
//!     ASCII-only design is intentional; richer tokenizers can opt
//!     into the trait without changing the FTS pipeline.
//!   - Empty tokens are never emitted.

use std::{
    any::Any,
    borrow::Cow,
    collections::BTreeSet,
    str::{from_utf8, from_utf8_unchecked},
};

use wide::u8x16;

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
///     can take a monomorphic fast path when the tokenizer is the
///     default [`AsciiLowerTokenizer`]. The fast path bypasses the
///     `&mut dyn FnMut(&str)` indirection by calling the inherent
///     [`AsciiLowerTokenizer::tokenize_each_inline`] method, whose
///     `F: FnMut(&str)` parameter lets LLVM inline the callback
///     body straight into the tokenizer's per-byte scan. Custom
///     tokenizers don't need to opt in — they just return `self`
///     and never get downcast.
pub trait Tokenizer: Send + Sync + 'static {
    /// Yield each token as an owned `String` lower-cased per the
    /// implementation's rules.
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

    /// Downcast hatch for the FTS build hot path. Default impl
    /// returns `self` cast to `&dyn Any`; concrete impls should
    /// not override unless they wrap another tokenizer.
    fn as_any(&self) -> &dyn Any;

    /// Used to tokenize a query. The tokens handed to `f` stay alive
    /// as long as `text` (the query) is alive.
    fn tokenize_each_query<'q>(&self, text: &'q str, f: &mut dyn FnMut(Cow<'q, str>)) {
        self.tokenize_each(text, &mut |t| f(Cow::Owned(t.to_owned())));
    }

    /// Used to parse a query: `"rust -python"` → positives `["rust"]`,
    /// negatives `["python"]`. A query with no positives is not an
    /// error here; the caller checks.
    fn parse<'q>(&self, query: &'q str) -> ParsedQuery<'q> {
        let mut parsed = ParsedQuery::default();
        for run in query.split_whitespace() {
            match run.strip_prefix('-') {
                Some(rest) if !rest.is_empty() => {
                    self.tokenize_each_query(rest, &mut |t| parsed.negatives.push(t));
                }
                _ => self.tokenize_each_query(run, &mut |t| parsed.positives.push(t)),
            }
        }
        parsed
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
        let bytes = text.as_bytes();
        let mut buf: Vec<u8> = Vec::new();
        let mut pos = 0;
        while pos < bytes.len() {
            pos = simd_skip_non_token(bytes, pos);
            if pos >= bytes.len() {
                return;
            }
            let start = pos;
            let (end, had_upper, had_non_ascii) = simd_scan_token_run(bytes, pos);
            pos = end;
            if had_non_ascii || start == pos {
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
                f(s);
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
                f(s);
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

/// A parsed BM25 query: positive (scored) and negative (excluding)
/// tokens. Tokens may borrow the query string, so this can't outlive
/// the query.
#[derive(Debug, Default)]
pub struct ParsedQuery<'q> {
    pub positives: Vec<Cow<'q, str>>,
    pub negatives: Vec<Cow<'q, str>>,
}

impl Tokenizer for AsciiLowerTokenizer {
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

    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Zero-copy override: an already-lowercase token borrows from
    /// `text`; only a token that needs lowercasing is copied.
    fn tokenize_each_query<'q>(&self, text: &'q str, f: &mut dyn FnMut(Cow<'q, str>)) {
        let bytes = text.as_bytes();
        let mut pos = 0;
        while pos < bytes.len() {
            pos = simd_skip_non_token(bytes, pos);
            if pos >= bytes.len() {
                return;
            }
            let start = pos;
            let (end, had_upper, had_non_ascii) = simd_scan_token_run(bytes, pos);
            pos = end;
            if had_non_ascii || start == pos {
                continue;
            }
            let s = from_utf8(&bytes[start..end]).expect("ASCII-only by construction");
            if had_upper {
                f(Cow::Owned(s.to_ascii_lowercase()));
            } else {
                f(Cow::Borrowed(s));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(text: &str) -> Vec<String> {
        AsciiLowerTokenizer.tokenize(text).collect()
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
        struct PlainTok;
        impl Tokenizer for PlainTok {
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
        assert!(p.positives.is_empty());
        assert!(p.negatives.is_empty());
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
