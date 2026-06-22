// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Lex term range — the second skip-pruning summary per FTS column.
//!
//! Where the bloom drives **exact-term** skip ("does this superfile
//! contain term X?"), the term range drives **prefix-query** skip
//! ("could this superfile contain any term starting with prefix P?").
//! Stored on `FtsSummaryAgg` as `term_range` — the lex-smallest and
//! lex-largest terms in the superfile's FST for that column.
//!
//! A prefix `p` matches some term in the superfile iff the half-open
//! lex interval `[p, prefix_upper_bound(p))` overlaps the
//! superfile's `[min_term, max_term]`. The two helpers below — one
//! to compute `prefix_upper_bound` and one to check overlap — are
//! the entire skip operation; the manifest's superfile list scan
//! does this comparison once per superfile, before any byte of
//! payload is touched.
//!
//! # Edge cases
//!
//! - **Empty prefix** matches every term, so there's no upper
//!   bound and the range overlaps every non-empty superfile.
//! - **All-`0xFF` prefix** has no successor in the byte ordering
//!   (you can't increment past `0xFF`); only terms that are
//!   exactly `prefix` or are lex-greater starting with all-`0xFF`
//!   bytes can match. The helper returns `None` to signal "no
//!   upper bound exists" and the overlap predicate falls back to
//!   `min_term <= prefix <= max_term`.
//! - **Trailing `0xFF` bytes** in the prefix: stripped before
//!   incrementing the last non-`0xFF` byte. Example:
//!   `prefix_upper_bound([0xFE, 0xFF, 0xFF]) == Some([0xFF])`.

/// Compute the lex-successor of a prefix — the smallest byte
/// sequence that is strictly greater than every string starting
/// with `prefix`.
///
/// Returns `None` if no such successor exists in the byte
/// ordering: an empty prefix or an all-`0xFF` prefix has no upper
/// bound.
///
/// # Examples
///
/// - `prefix_upper_bound(b"err") == Some(b"ers")`
/// - `prefix_upper_bound(b"erz") == Some(b"es")` — wraps through
///   the trailing `z` (`0x7A` → `0x7B` is `'{'`, but here `z` is
///   not `0xFF`, so we just increment in place: `b"er{"`. So
///   actually: `prefix_upper_bound(b"erz") == Some(b"er{")`. The
///   "wrap" only happens when the trailing byte is `0xFF`.
/// - `prefix_upper_bound(b"\xFF") == None`
/// - `prefix_upper_bound(b"\xFE\xFF\xFF") == Some(b"\xFF")`
/// - `prefix_upper_bound(b"") == None`
pub fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut up = prefix.to_vec();
    while let Some(&0xFF) = up.last() {
        up.pop();
    }
    if up.is_empty() {
        return None;
    }
    let last = up
        .last_mut()
        .expect("invariant: non-empty after stripping 0xFF tail");
    *last = last
        .checked_add(1)
        .expect("invariant: last byte is not 0xFF (stripped above)");
    Some(up)
}

/// Whether some term in `[min_term, max_term]` could start with
/// `prefix`.
///
/// Concretely: the half-open interval `[prefix,
/// prefix_upper_bound(prefix))` overlaps the superfile's closed
/// interval `[min_term, max_term]`.
///
/// Used by the prefix-query skip helper to decide whether to fan
/// out into a superfile. False positives over-fetch (superfile is
/// scanned but no term matches) — acceptable. False negatives
/// under-fetch — never allowed; would silently drop matching
/// terms.
pub fn prefix_overlaps_range(prefix: &[u8], min_term: &[u8], max_term: &[u8]) -> bool {
    // Empty prefix matches every term.
    if prefix.is_empty() {
        return true;
    }
    // The superfile's max term must be ≥ the prefix itself, OR
    // start with the prefix. Equivalently: max_term must be ≥ prefix
    // when compared lex.
    if max_term < prefix {
        return false;
    }
    // Now check the upper-bound side.
    match prefix_upper_bound(prefix) {
        // No upper bound exists (all-0xFF prefix). Then the only
        // terms that could match are ones lex-≥ prefix and made up
        // of bytes ≤ 0xFF — every byte is ≤ 0xFF, so any term ≥
        // prefix could match. Combined with the max ≥ prefix check
        // above, this means: overlap iff max_term ≥ prefix, which
        // we just confirmed.
        None => true,
        // Standard case: overlap iff min_term < upper.
        Some(upper) => min_term < upper.as_slice(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- prefix_upper_bound ---------------------------------------

    #[test]
    fn upper_bound_of_simple_ascii_prefix() {
        assert_eq!(prefix_upper_bound(b"err"), Some(b"ers".to_vec()));
        assert_eq!(prefix_upper_bound(b"a"), Some(b"b".to_vec()));
        assert_eq!(prefix_upper_bound(b"abc"), Some(b"abd".to_vec()));
    }

    #[test]
    fn upper_bound_increments_last_byte_only() {
        // Verify it doesn't carry into earlier bytes for non-0xFF.
        assert_eq!(prefix_upper_bound(b"erz"), Some(b"er{".to_vec()));
        // 'z' = 0x7A → 0x7B = '{'. No carry.
    }

    #[test]
    fn upper_bound_of_empty_prefix_is_none() {
        assert_eq!(prefix_upper_bound(b""), None);
    }

    #[test]
    fn upper_bound_of_all_ff_prefix_is_none() {
        assert_eq!(prefix_upper_bound(&[0xFF]), None);
        assert_eq!(prefix_upper_bound(&[0xFF, 0xFF, 0xFF]), None);
    }

    #[test]
    fn upper_bound_strips_trailing_ff_then_increments() {
        // [0xFE, 0xFF, 0xFF] → strip trailing 0xFFs → [0xFE] →
        // increment → [0xFF].
        assert_eq!(prefix_upper_bound(&[0xFE, 0xFF, 0xFF]), Some(vec![0xFF]),);
        // [0x10, 0xFF] → [0x10] → [0x11].
        assert_eq!(prefix_upper_bound(&[0x10, 0xFF]), Some(vec![0x11]));
        // ['a', 0xFF, 0xFF] → ['a'] → ['b'].
        assert_eq!(prefix_upper_bound(&[b'a', 0xFF, 0xFF]), Some(b"b".to_vec()),);
    }

    #[test]
    fn upper_bound_at_byte_boundary() {
        // 0x7E + 1 = 0x7F (still single byte; no overflow).
        assert_eq!(prefix_upper_bound(&[0x7E]), Some(vec![0x7F]));
        // 0xFE + 1 = 0xFF (max single byte).
        assert_eq!(prefix_upper_bound(&[0xFE]), Some(vec![0xFF]));
    }

    #[test]
    fn upper_bound_is_strictly_greater_than_prefix() {
        // Property: for any non-(empty / all-0xFF) prefix, the
        // returned upper bound is strictly lex-greater than the
        // prefix itself.
        let cases: &[&[u8]] = &[
            b"a",
            b"ab",
            b"abc",
            b"err",
            &[0x00],
            &[0x10, 0x20, 0x30],
            &[0xFE],
            &[0xFE, 0xFF],
            &[0x10, 0xFF],
        ];
        for p in cases {
            let up = prefix_upper_bound(p)
                .unwrap_or_else(|| panic!("{:?} should have an upper bound", p));
            assert!(
                up.as_slice() > *p,
                "upper bound {:?} not > prefix {:?}",
                up,
                p,
            );
        }
    }

    #[test]
    fn upper_bound_excludes_strings_starting_with_prefix() {
        // Property: every string starting with `prefix` is lex-LESS
        // than `prefix_upper_bound(prefix)`.
        let p = b"err";
        let up = prefix_upper_bound(p).expect("has upper");

        let with_prefix: &[&[u8]] = &[
            b"err",
            b"err\0",
            b"errand",
            b"erratic",
            b"errz",
            b"err\xFF",
            b"err\xFF\xFF\xFF",
        ];
        for s in with_prefix {
            let s_bytes: &[u8] = s;
            assert!(
                s_bytes < up.as_slice(),
                "{:?} (starting with prefix) should be < upper {:?}",
                s,
                up,
            );
        }
    }

    // ---- prefix_overlaps_range ------------------------------------

    #[test]
    fn empty_prefix_overlaps_any_range() {
        assert!(prefix_overlaps_range(b"", b"alpha", b"omega"));
        assert!(prefix_overlaps_range(b"", b"\x00", b"\xFF"));
        assert!(prefix_overlaps_range(b"", b"x", b"x"));
    }

    #[test]
    fn prefix_inside_range_overlaps() {
        // superfile's terms are in [b"checkin", b"checkout"], prefix
        // "check" should overlap.
        assert!(prefix_overlaps_range(b"check", b"checkin", b"checkout"));
        // Single-term superfile matching the prefix.
        assert!(prefix_overlaps_range(b"err", b"errand", b"errand"));
    }

    #[test]
    fn prefix_above_max_term_does_not_overlap() {
        // superfile terms ≤ "alpha", prefix "beta" can't appear.
        assert!(!prefix_overlaps_range(b"beta", b"a", b"alpha"));
        assert!(!prefix_overlaps_range(b"zzz", b"a", b"y"));
    }

    #[test]
    fn prefix_below_min_term_does_not_overlap() {
        // superfile terms ≥ "g", prefix "a" can't appear (since
        // every "a..." is < "g").
        assert!(!prefix_overlaps_range(b"a", b"g", b"z"));
        assert!(!prefix_overlaps_range(b"abc", b"def", b"xyz"));
    }

    #[test]
    fn prefix_equals_min_term_overlaps() {
        // Superfile min is exactly the prefix.
        assert!(prefix_overlaps_range(b"err", b"err", b"erz"));
    }

    #[test]
    fn prefix_equals_max_term_overlaps() {
        // Superfile max is exactly the prefix.
        assert!(prefix_overlaps_range(b"err", b"a", b"err"));
    }

    #[test]
    fn all_ff_prefix_overlaps_when_max_is_at_least_prefix() {
        // No upper bound exists; overlap iff max ≥ prefix.
        assert!(prefix_overlaps_range(&[0xFF], &[0xFF], &[0xFF]));
        assert!(prefix_overlaps_range(&[0xFF], &[0x80], &[0xFF, 0xFF]));
        assert!(!prefix_overlaps_range(&[0xFF], &[0x00], &[0xFE]));
    }

    #[test]
    fn prefix_overlap_property_no_false_negatives() {
        // For any superfile range and any prefix, if `prefix` is a
        // strict prefix of any term in [min, max], the helper must
        // return true (false negatives would silently drop matching
        // terms — unsound).
        //
        // We construct synthetic superfiles where the term list
        // straddles the prefix boundary and verify the helper says
        // "overlap" in every case.
        let prefix = b"err";

        // (min, max, planted_term_starting_with_prefix)
        let cases: &[(&[u8], &[u8], &[u8])] = &[
            (b"err", b"err", b"err"),
            (b"err", b"errand", b"err"),
            (b"err", b"errand", b"errand"),
            (b"epsilon", b"errand", b"errand"),
            (b"epsilon", b"zeta", b"errand"),
            (b"era", b"errz", b"errand"),
            (b"era", b"err", b"err"),
            (b"e", b"f", b"errand"),
        ];
        for (min, max, planted) in cases {
            // sanity: planted starts with prefix and falls in [min, max].
            assert!(planted.starts_with(prefix), "test setup broken");
            assert!(
                min <= planted && planted <= max,
                "test setup broken: planted {:?} not in [{:?}, {:?}]",
                planted,
                min,
                max,
            );
            assert!(
                prefix_overlaps_range(prefix, min, max),
                "false negative for [{:?}, {:?}], planted {:?}",
                min,
                max,
                planted,
            );
        }
    }

    #[test]
    fn prefix_overlap_handles_trailing_ff_in_prefix() {
        // prefix [0xFE, 0xFF, 0xFF] → upper [0xFF]. Superfiles whose
        // min is < [0xFF] and max ≥ [0xFE, 0xFF, 0xFF] overlap.
        let p = &[0xFE, 0xFF, 0xFF];
        assert!(prefix_overlaps_range(
            p,
            &[0xFE, 0xFF, 0xFF],
            &[0xFE, 0xFF, 0xFF],
        ));
        assert!(prefix_overlaps_range(p, &[0xFE, 0x00], &[0xFE, 0xFF, 0xFF]));
        // max below prefix → no overlap.
        assert!(!prefix_overlaps_range(
            p,
            &[0xFE, 0x00],
            &[0xFE, 0xFE, 0xFF]
        ));
        // min ≥ upper bound [0xFF] → no overlap.
        assert!(!prefix_overlaps_range(p, &[0xFF], &[0xFF, 0xFF]));
    }
}
