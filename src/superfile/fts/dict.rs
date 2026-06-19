// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FST-backed term dictionary.
//!
//! The unified FTS blob stores every `(column, term)` pair as a single FST
//! keyed by `<column_name>\x1F<term>` (ASCII Unit Separator). Values are
//! `u64` — the per-`(column, term)` metadata offset directly. At the
//! dict level it's just an opaque integer keyed by bytes.
//!
//! The FST is **prefix-compressed** (FST tail-merging makes shared
//! suffixes near-free) and **prefix-iterable** — a range scan over keys
//! starting with `<column>\x1F` returns exactly the terms in that
//! column, in sorted order, in O(matching keys).
//!
//! # Build / read split
//!
//! `DictBuilder` stages inserts in a `BTreeMap` to absorb the FST's
//! "must insert in sorted order" requirement; `finish()` drains the
//! map in order into `fst::MapBuilder` and returns the serialized FST
//! bytes. `DictReader` opens those bytes (zero-copy via `Bytes`) and
//! exposes exact lookup + prefix iteration.

use std::{collections::BTreeMap, io::Write};

use fst::{IntoStreamer, Map, MapBuilder, Streamer};

use crate::superfile::format::FST_SEPARATOR;

/// Build a canonical FST key from `(column_name, term)`.
///
/// Encoding: `<column_name_utf8> | 0x1F | <term_utf8>`. The separator
/// byte (`FST_SEPARATOR`, ASCII Unit Separator) is below every printable
/// ASCII byte, so prefix iteration `column_name\x1F` cleanly captures
/// every term in that column.
///
/// Callers must ensure `column_name` does not itself contain the
/// separator byte — see [`validate_column_name`]. `term` may be any
/// bytes the tokenizer produced; v1's `AsciiLowerTokenizer` produces
/// only `[a-z0-9]+`, so the separator can never appear in a term.
pub fn make_key(column_name: &str, term: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(column_name.len() + 1 + term.len());
    k.extend_from_slice(column_name.as_bytes());
    k.push(FST_SEPARATOR);
    k.extend_from_slice(term.as_bytes());
    k
}

/// Returns `true` if `column_name` is safe to use as the column part of
/// an FST key: it must not contain the FST separator byte (otherwise
/// prefix iteration could return cross-column matches).
///
/// All other bytes are allowed; format-level naming rules (no `inf.`
/// prefix, etc.) are enforced elsewhere.
#[inline]
pub fn validate_column_name(column_name: &str) -> bool {
    !column_name.as_bytes().contains(&FST_SEPARATOR)
}

/// Stages keys for FST construction.
///
/// `fst::MapBuilder` requires keys to be inserted in sorted order;
/// `BTreeMap` absorbs that constraint while also deduplicating on
/// repeated key inserts (last write wins, matching the FST invariant
/// of one value per key).
#[derive(Debug, Default)]
pub struct DictBuilder {
    sorted_buffer: BTreeMap<Vec<u8>, u64>,
}

impl DictBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stage a `(key, value)` pair. Inserts can arrive in any order;
    /// repeated inserts of the same key keep the most-recent value.
    pub fn insert(&mut self, key: &[u8], value: u64) {
        self.sorted_buffer.insert(key.to_vec(), value);
    }

    /// Number of distinct keys staged so far.
    pub fn len(&self) -> usize {
        self.sorted_buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sorted_buffer.is_empty()
    }

    /// Finalize the FST and return its serialized bytes.
    ///
    /// Panics on internal FST errors — these can only occur if the
    /// `BTreeMap` invariant (sorted, unique keys) is broken, which is
    /// impossible without `unsafe`.
    pub fn finish(self) -> Vec<u8> {
        let mut builder = MapBuilder::memory();
        for (k, v) in self.sorted_buffer {
            builder
                .insert(&k, v)
                .expect("BTreeMap guarantees sorted, unique keys");
        }
        builder
            .into_inner()
            .expect("in-memory FST writer cannot fail at finalize")
    }
}

/// Streaming FST builder: wraps `fst::MapBuilder<W>` for callers that
/// can supply keys in already-sorted order and want the FST bytes
/// written progressively to a `Write` sink (e.g. a scratch file) so
/// the dictionary never has to be materialised in RAM.
///
/// Use this when build-time memory matters and the producer already
/// emits keys in lex order — for example after a k-way merge over
/// pre-sorted partition spills. Use [`DictBuilder`] when the caller
/// inserts in arbitrary order and the dictionary fits in RAM.
///
/// Mirrors the `fst::MapBuilder<W>` contract: keys MUST be strictly
/// greater than the previous key. Out-of-order inserts return an
/// `fst::Error`.
pub struct StreamingDictBuilder<W: Write> {
    inner: MapBuilder<W>,
    n_keys: u64,
}

impl<W: Write> StreamingDictBuilder<W> {
    /// Wrap a `Write` sink. The FST format header is written
    /// immediately so the sink position advances on construction.
    ///
    /// **Buffering**: `StreamingDictBuilder` does not interpose its
    /// own buffer; callers writing to a `File` (or any sink with
    /// per-call syscall cost) should wrap in a
    /// `std::io::BufWriter` first. The internal `fst::MapBuilder`
    /// writes in small chunks per key, so passing a raw `File`
    /// makes each `insert_sorted` call trigger multiple `write`
    /// syscalls. The FTS builder follows this contract — see
    /// `FtsBuilder::finish_to`, which wraps the scratch FST file in
    /// a `BufWriter` before constructing the streaming builder.
    pub fn new(w: W) -> Result<Self, fst::Error> {
        Ok(Self {
            inner: MapBuilder::new(w)?,
            n_keys: 0,
        })
    }

    /// Insert a `(key, value)` pair. `key` must be strictly greater
    /// than the previously inserted key. Returns an error otherwise.
    pub fn insert_sorted(&mut self, key: &[u8], value: u64) -> Result<(), fst::Error> {
        self.inner.insert(key, value)?;
        self.n_keys += 1;
        Ok(())
    }

    /// Number of keys inserted so far.
    #[inline]
    pub fn n_keys(&self) -> u64 {
        self.n_keys
    }

    /// Finalise the FST (writes the trailer) and return the inner
    /// writer. The writer is left at the byte just past the FST.
    pub fn finish(self) -> Result<W, fst::Error> {
        self.inner.into_inner()
    }
}

/// Reads a serialized FST. Borrows its bytes — zero-copy when the
/// caller has the blob mmap'd or held in a `Bytes`.
pub struct DictReader<'a> {
    fst: Map<&'a [u8]>,
}

impl<'a> DictReader<'a> {
    /// Open from already-serialized FST bytes (the output of
    /// `DictBuilder::finish`).
    pub fn open(bytes: &'a [u8]) -> Result<Self, fst::Error> {
        Ok(Self {
            fst: Map::new(bytes)?,
        })
    }

    /// Exact lookup. Returns `None` if `key` is not in the FST.
    #[inline]
    pub fn lookup(&self, key: &[u8]) -> Option<u64> {
        self.fst.get(key)
    }

    /// Number of `(key, value)` pairs in the FST.
    #[inline]
    pub fn len(&self) -> usize {
        self.fst.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.fst.is_empty()
    }

    /// Collect every `(key, value)` pair whose key starts with `prefix`,
    /// in lexicographic order. Returns owned data so the caller doesn't
    /// have to manage stream lifetimes.
    ///
    /// Uses an FST range scan that starts at `prefix` and stops as soon
    /// as a key without `prefix` appears — O(matching keys), not O(N).
    /// Suitable for "list every term in column X" diagnostics; for hot
    /// query paths use [`Self::lookup`].
    pub fn iter_prefix(&self, prefix: &[u8]) -> Vec<(Vec<u8>, u64)> {
        let mut out = Vec::new();
        let mut stream = self.fst.range().ge(prefix).into_stream();
        while let Some((key, value)) = stream.next() {
            if !key.starts_with(prefix) {
                break;
            }
            out.push((key.to_vec(), value));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- make_key -------------------------------------------------------

    #[test]
    fn make_key_encodes_with_unit_separator() {
        assert_eq!(make_key("title", "rust"), b"title\x1Frust");
    }

    #[test]
    fn make_key_handles_empty_term() {
        // The bare prefix used by iter_prefix to "list all terms in column".
        assert_eq!(make_key("title", ""), b"title\x1F");
    }

    #[test]
    fn make_key_handles_empty_column() {
        // Edge case; not produced in practice but must not panic.
        assert_eq!(make_key("", "rust"), b"\x1Frust");
    }

    #[test]
    fn make_key_handles_both_empty() {
        assert_eq!(make_key("", ""), b"\x1F");
    }

    #[test]
    fn make_key_preserves_term_bytes() {
        // The tokenizer drops non-ASCII tokens, but make_key itself is
        // byte-transparent. Multi-byte UTF-8 in the term comes through
        // exactly.
        let key = make_key("body", "café");
        assert_eq!(&key[0..4], b"body");
        assert_eq!(key[4], FST_SEPARATOR);
        assert_eq!(&key[5..], "café".as_bytes());
    }

    #[test]
    fn make_key_capacity_avoids_realloc() {
        // Sanity: the Vec is sized exactly. Not a behaviour requirement
        // but a perf one — protects the hot-path call against
        // accidental realloc.
        let key = make_key("title", "rust");
        assert_eq!(key.len(), key.capacity());
    }

    // --- validate_column_name -------------------------------------------

    #[test]
    fn validate_column_name_rejects_separator_byte() {
        // A column name containing 0x1F would let prefix iteration leak
        // across columns. Format-level validation rejects this at the
        // builder boundary.
        let bad = "ti\x1Ftle";
        assert!(!validate_column_name(bad));
    }

    #[test]
    fn validate_column_name_accepts_normal_names() {
        for name in [
            "title",
            "body",
            "headline",
            "field_1",
            "field-2",
            "MyField",
            "snake_case",
            "camelCase",
            "PascalCase",
            "with spaces",
            "with.dots",
        ] {
            assert!(validate_column_name(name), "expected {name:?} to be valid");
        }
    }

    #[test]
    fn validate_column_name_accepts_empty_string() {
        // Length restrictions are enforced at the builder level, not here.
        assert!(validate_column_name(""));
    }

    #[test]
    fn validate_column_name_accepts_high_unicode() {
        // No restriction on non-ASCII bytes. Format-level rules can add
        // restrictions if needed; the dict layer is byte-transparent.
        assert!(validate_column_name("café"));
        assert!(validate_column_name("日本語"));
    }

    // --- DictBuilder + DictReader roundtrip -----------------------------

    #[test]
    fn lookup_roundtrip_basic() {
        let mut b = DictBuilder::new();
        b.insert(&make_key("title", "rust"), 100);
        b.insert(&make_key("title", "async"), 200);
        b.insert(&make_key("body", "tokio"), 300);
        let bytes = b.finish();

        let r = DictReader::open(&bytes).expect("open DictReader");
        assert_eq!(r.lookup(&make_key("title", "rust")), Some(100));
        assert_eq!(r.lookup(&make_key("title", "async")), Some(200));
        assert_eq!(r.lookup(&make_key("body", "tokio")), Some(300));
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn lookup_missing_returns_none() {
        let mut b = DictBuilder::new();
        b.insert(&make_key("title", "rust"), 1);
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        assert_eq!(r.lookup(&make_key("title", "java")), None);
        assert_eq!(r.lookup(&make_key("body", "rust")), None);
        assert_eq!(r.lookup(b""), None);
    }

    #[test]
    fn out_of_order_inserts_work() {
        // Direct fst::MapBuilder rejects out-of-order keys; our
        // DictBuilder absorbs the sort via BTreeMap.
        let mut b = DictBuilder::new();
        let keys = ["z", "a", "m", "b", "y", "c"];
        for (i, k) in keys.iter().enumerate() {
            b.insert(&make_key("col", k), i as u64);
        }
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        for (i, k) in keys.iter().enumerate() {
            assert_eq!(r.lookup(&make_key("col", k)), Some(i as u64));
        }
    }

    #[test]
    fn duplicate_inserts_keep_last_value() {
        // BTreeMap dedups; the FST gets the latest value. Documents
        // the "last write wins" semantic for DictBuilder::insert.
        let mut b = DictBuilder::new();
        b.insert(&make_key("col", "key"), 1);
        b.insert(&make_key("col", "key"), 2);
        b.insert(&make_key("col", "key"), 999);
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        assert_eq!(r.lookup(&make_key("col", "key")), Some(999));
        assert_eq!(r.len(), 1);
    }

    // --- Prefix iteration ----------------------------------------------

    #[test]
    fn iter_prefix_returns_matching_keys_only() {
        let mut b = DictBuilder::new();
        // Three columns, distinct vocabularies.
        b.insert(&make_key("title", "alpha"), 1);
        b.insert(&make_key("title", "beta"), 2);
        b.insert(&make_key("title", "gamma"), 3);
        b.insert(&make_key("body", "alpha"), 10);
        b.insert(&make_key("body", "delta"), 20);
        b.insert(&make_key("tag", "epsilon"), 100);
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        // Listing all terms in `title`.
        let title_terms = r.iter_prefix(b"title\x1F");
        let title_only: Vec<&[u8]> = title_terms.iter().map(|(k, _)| k.as_slice()).collect();
        assert_eq!(
            title_only,
            vec![
                b"title\x1Falpha".as_slice(),
                b"title\x1Fbeta".as_slice(),
                b"title\x1Fgamma".as_slice(),
            ],
            "iter_prefix on title\\x1F returns only title-prefixed keys"
        );
        assert_eq!(
            title_terms.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );

        // body has different vocabulary.
        let body_terms = r.iter_prefix(b"body\x1F");
        assert_eq!(body_terms.len(), 2);

        // tag has just one.
        let tag_terms = r.iter_prefix(b"tag\x1F");
        assert_eq!(tag_terms.len(), 1);
    }

    #[test]
    fn iter_prefix_lexicographic_order() {
        // Insert in arbitrary order, expect lex order out.
        let mut b = DictBuilder::new();
        let terms = ["zebra", "ant", "monkey", "apple", "banana", "aardvark"];
        for (i, t) in terms.iter().enumerate() {
            b.insert(&make_key("col", t), i as u64);
        }
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        let listed: Vec<Vec<u8>> = r
            .iter_prefix(b"col\x1F")
            .into_iter()
            .map(|(k, _)| k[4..].to_vec()) // strip "col\x1F"
            .collect();
        let mut expected: Vec<&str> = terms.to_vec();
        expected.sort();
        let expected: Vec<Vec<u8>> = expected.iter().map(|s| s.as_bytes().to_vec()).collect();
        assert_eq!(listed, expected);
    }

    #[test]
    fn iter_prefix_no_match_returns_empty() {
        let mut b = DictBuilder::new();
        b.insert(&make_key("title", "rust"), 1);
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        // Column that doesn't exist in the FST.
        assert_eq!(r.iter_prefix(b"missing\x1F"), Vec::new());
        // Empty prefix on non-empty FST returns everything (defining
        // behavior — useful for a "dump dict" diagnostic).
        assert_eq!(r.iter_prefix(b"").len(), 1);
    }

    #[test]
    fn iter_prefix_stops_at_first_non_match() {
        // Implementation correctness: iter_prefix must stop scanning
        // as soon as a non-matching key appears. We can't observe the
        // stopping directly, but we can confirm the result is correct
        // when many post-prefix keys exist.
        let mut b = DictBuilder::new();
        // Lots of `body\x1F*` keys after the (alphabetically earlier)
        // `title\x1F*` block.
        for i in 0..1000 {
            b.insert(&make_key("body", &format!("term{i:04}")), i as u64);
        }
        b.insert(&make_key("title", "rust"), 9999);
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        // Title scan must yield exactly one entry, regardless of how
        // many body entries follow (`body` < `title` alphabetically,
        // so by the time the scan reaches `title\x1F` it has already
        // skipped past every `body\x1F*` key).
        let title = r.iter_prefix(b"title\x1F");
        assert_eq!(title.len(), 1);
        assert_eq!(title[0].1, 9999);
    }

    // --- Empty-FST handling --------------------------------------------

    #[test]
    fn empty_builder_produces_valid_empty_fst() {
        let bytes = DictBuilder::new().finish();
        let r = DictReader::open(&bytes).expect("open DictReader");
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert_eq!(r.lookup(b"anything"), None);
        assert_eq!(r.iter_prefix(b""), Vec::new());
        assert_eq!(r.iter_prefix(b"prefix"), Vec::new());
    }

    // --- Stress / scale -------------------------------------------------

    #[test]
    fn handles_thousand_keys_roundtrip() {
        let mut b = DictBuilder::new();
        for i in 0..1000 {
            b.insert(&make_key("body", &format!("term{i:04}")), i as u64);
        }
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");
        assert_eq!(r.len(), 1000);

        for i in 0..1000 {
            let k = make_key("body", &format!("term{i:04}"));
            assert_eq!(r.lookup(&k), Some(i as u64));
        }

        // Prefix iter returns all 1000 in lex order.
        let listed = r.iter_prefix(b"body\x1F");
        assert_eq!(listed.len(), 1000);
        for w in listed.windows(2) {
            assert!(w[0].0 < w[1].0, "prefix iter not in lex order");
        }
    }

    #[test]
    fn serialization_is_deterministic() {
        // Same inputs (regardless of insertion order) produce
        // byte-identical FST output. Important for reproducible builds
        // and for content-addressed superfile hashing.
        let mut b1 = DictBuilder::new();
        b1.insert(&make_key("a", "x"), 1);
        b1.insert(&make_key("b", "y"), 2);
        b1.insert(&make_key("c", "z"), 3);
        let bytes1 = b1.finish();

        let mut b2 = DictBuilder::new();
        b2.insert(&make_key("c", "z"), 3);
        b2.insert(&make_key("a", "x"), 1);
        b2.insert(&make_key("b", "y"), 2);
        let bytes2 = b2.finish();

        assert_eq!(bytes1, bytes2);
    }

    #[test]
    fn keys_with_same_prefix_share_state() {
        // FST tail-merging makes shared prefixes effectively free.
        // We don't probe FST internals; just verify lookups are still
        // correct when many keys share a long common prefix.
        let mut b = DictBuilder::new();
        for i in 0..100 {
            b.insert(
                &make_key("very_long_column_name", &format!("term{i}")),
                i as u64,
            );
        }
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");
        assert_eq!(r.len(), 100);
        for i in 0..100 {
            let k = make_key("very_long_column_name", &format!("term{i}"));
            assert_eq!(r.lookup(&k), Some(i as u64));
        }
    }

    #[test]
    fn lookup_distinguishes_columns_with_same_term() {
        // The whole point of `<col>\x1F<term>` keying: same term in
        // different columns must look up to different values.
        let mut b = DictBuilder::new();
        b.insert(&make_key("title", "rust"), 1);
        b.insert(&make_key("body", "rust"), 2);
        b.insert(&make_key("tag", "rust"), 3);
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        assert_eq!(r.lookup(&make_key("title", "rust")), Some(1));
        assert_eq!(r.lookup(&make_key("body", "rust")), Some(2));
        assert_eq!(r.lookup(&make_key("tag", "rust")), Some(3));
    }

    /// `DictBuilder` tracks key counts + emptiness, which round-trip
    /// through `DictReader::{len, is_empty}`; the streaming builder
    /// tracks `n_keys` as sorted keys arrive.
    #[test]
    fn builder_and_reader_track_key_counts() {
        let mut b = DictBuilder::new();
        assert!(b.is_empty(), "fresh builder is empty");
        assert_eq!(b.len(), 0);
        b.insert(b"alpha", 1);
        b.insert(b"beta", 2);
        assert!(!b.is_empty());
        assert_eq!(b.len(), 2);
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open dict");
        assert_eq!(r.len(), 2);
        assert!(!r.is_empty());

        // Streaming builder counts keys fed in strictly-sorted order.
        let mut sb = StreamingDictBuilder::new(Vec::new()).expect("streaming builder");
        assert_eq!(sb.n_keys(), 0);
        sb.insert_sorted(b"a", 1).expect("sorted insert");
        sb.insert_sorted(b"b", 2).expect("sorted insert");
        assert_eq!(sb.n_keys(), 2);
        let _ = sb.finish().expect("finish streaming builder");
    }
}
