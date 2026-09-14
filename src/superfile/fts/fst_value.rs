// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FST-value bit layout — the inline-encoding short-circuit for
//! `df = 1` terms, and the short/long flag for everything else.
//!
//! Every term's FST value is a `u64`. Bit 0 selects form:
//!
//! - `value & 1 == 0` → **postings form**. The payload stores both
//!   `metadata_offset` and `postings_length`, so the reader can fetch
//!   the complete term range in one GET instead of probing the 20 B
//!   metadata header first. `postings_length` is a hint — the header
//!   carries the same value and is the authority; see
//!   [`PFOR_LENGTH_UNKNOWN`].
//! - `value & 1 == 1` → **inline form**. The payload `(doc_id, tf)`
//!   lives entirely in bits 1..63; there is no postings-region entry
//!   for this term (no metadata header, no skip table, no PFOR block).
//!
//! Inline layout (when bit 0 = 1):
//!
//! ```text
//!   bits  1..33 : doc_id (u32)
//!   bits 33..63 : tf (30 bits — covers any realistic per-doc tf)
//! ```
//!
//! Postings-form layout depends on the blob version ([`ValueLayout`]):
//!
//! ```text
//!   Legacy (V1–V6):  bits  1..43 : metadata_offset (42 bits)
//!                    bits 43..64 : postings_length (21 bits)
//!   Flagged (V7+):   bits  1..22 : postings_length (21 bits)
//!                    bit  22     : 1 = short form, 0 = long (PFOR) form
//!                    bits 23..64 : metadata_offset (41 bits, 2 TiB)
//! ```
//!
//! The flag tells the reader how to interpret the fetched range before
//! it touches a byte of it: a long-form range starts with the metadata
//! header, skip table and blocks; a short-form range is a
//! `fts::short` body with none of those. A legacy value spends bit 1 on
//! its offset, so the layout is selected by version, never inferred.
//!
//! Why the flagged layout puts the offset in the **high** bits: terms are
//! emitted in dictionary order, so consecutive keys have ascending,
//! nearby offsets. The FST shares each transition's common output
//! prefix along the path, so two neighbouring values that differ only in
//! their low bits leave a small residual at the leaf — a length field in
//! the high bits (the legacy layout) scrambled exactly the bits that
//! would otherwise be shared, and cost every term a full-width value.
//!
//! Why low-bit flag (not high-bit): the `fst` crate VLQ-encodes
//! values, so encoded length grows with magnitude. Putting the flag
//! in the low bit keeps postings-form values small (~5–6 bytes VLQ at
//! 16 GB superfile scale); only inline values pay the larger encoding
//! (~7–8 bytes VLQ for the composite). High-bit flag would force
//! *every* value to a full ~9-byte encoding.

const DOC_ID_SHIFT: u32 = 1;
const TF_SHIFT: u32 = 33;
const LEGACY_OFFSET_SHIFT: u32 = 1;
const LEGACY_OFFSET_BITS: u32 = 42;
const PFOR_LENGTH_BITS: u32 = 21;
const LEGACY_LENGTH_SHIFT: u32 = LEGACY_OFFSET_SHIFT + LEGACY_OFFSET_BITS;
const FLAGGED_LENGTH_SHIFT: u32 = 1;
const SHORT_FLAG_SHIFT: u32 = FLAGGED_LENGTH_SHIFT + PFOR_LENGTH_BITS;
const FLAGGED_OFFSET_SHIFT: u32 = SHORT_FLAG_SHIFT + 1;
const FLAGGED_OFFSET_BITS: u32 = 64 - FLAGGED_OFFSET_SHIFT;
const LEGACY_OFFSET_MAX: u64 = (1u64 << LEGACY_OFFSET_BITS) - 1;
const FLAGGED_OFFSET_MAX: u64 = (1u64 << FLAGGED_OFFSET_BITS) - 1;
pub(crate) const PFOR_LENGTH_MAX: u32 = (1u32 << PFOR_LENGTH_BITS) - 1;
/// Length-slot sentinel: a term whose postings don't fit the 21-bit
/// slot stores this instead, and the reader gets the real length from
/// the term's metadata header. All-ones is safe as a legacy real
/// length too — the header still resolves it to the same value.
pub(crate) const PFOR_LENGTH_UNKNOWN: u32 = PFOR_LENGTH_MAX;
/// Maximum `tf` representable in the inline form's 30-bit slot.
/// Real-world per-doc tf is bounded by document length (in tokens),
/// which fits a u16; this limit only exists to guarantee the
/// pack/unpack round-trip in debug.
pub(crate) const INLINE_TF_MAX: u32 = (1 << 30) - 1;

/// How a postings-form value packs its offset — selected by the blob
/// version the value was read from (or is being written into).
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum ValueLayout {
    /// `V1`–`V6`: 42-bit offset from bit 1; every postings-form term is
    /// long form.
    Legacy,
    /// `V7` and later: bit 1 is the short/long flag, 41-bit offset from
    /// bit 2.
    Flagged,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum FstValue {
    /// df ≥ 2 (or a df = 1 posting that could not inline) — fetch
    /// `postings_length` bytes from `metadata_offset`. `short` says
    /// which body those bytes hold: the long form (metadata header, skip
    /// table, PFOR blocks) or the short form (`fts::short`).
    Pfor {
        metadata_offset: u64,
        /// `None` when the slot held [`PFOR_LENGTH_UNKNOWN`]: the term
        /// is too large for the slot and its length must be read from
        /// the metadata header at `metadata_offset`. Never `None` for
        /// a short-form term, whose body is at most a few hundred bytes.
        postings_length_hint: Option<u32>,
        short: bool,
    },
    /// df = 1 — the entire posting is right here. No postings-region
    /// read required.
    Inline { doc_id: u32, tf: u32 },
}

impl FstValue {
    #[inline]
    pub(crate) fn unpack(packed: u64, layout: ValueLayout) -> Self {
        if packed & 1 == 0 {
            let (metadata_offset, slot, short) = match layout {
                ValueLayout::Legacy => (
                    (packed >> LEGACY_OFFSET_SHIFT) & LEGACY_OFFSET_MAX,
                    ((packed >> LEGACY_LENGTH_SHIFT) as u32) & PFOR_LENGTH_MAX,
                    false,
                ),
                ValueLayout::Flagged => (
                    (packed >> FLAGGED_OFFSET_SHIFT) & FLAGGED_OFFSET_MAX,
                    ((packed >> FLAGGED_LENGTH_SHIFT) as u32) & PFOR_LENGTH_MAX,
                    (packed >> SHORT_FLAG_SHIFT) & 1 == 1,
                ),
            };
            Self::Pfor {
                metadata_offset,
                postings_length_hint: match slot {
                    PFOR_LENGTH_UNKNOWN => None,
                    len => Some(len),
                },
                short,
            }
        } else {
            let doc_id = (packed >> DOC_ID_SHIFT) as u32;
            let tf = ((packed >> TF_SHIFT) as u32) & INLINE_TF_MAX;
            Self::Inline { doc_id, tf }
        }
    }

    /// Pack `(metadata_offset, postings_length)` into the postings-form
    /// FST value. The low bit is always 0.
    ///
    /// A `postings_length` at or past [`PFOR_LENGTH_UNKNOWN`] is stored
    /// as that sentinel rather than rejected — the reader recovers the
    /// true length from the term's metadata header. `short` is only
    /// representable under [`ValueLayout::Flagged`]; a legacy layout
    /// asserts it is `false`.
    #[inline]
    pub(crate) fn pack_pfor(
        metadata_offset: u64,
        postings_length: u32,
        layout: ValueLayout,
        short: bool,
    ) -> u64 {
        let slot = u64::from(postings_length.min(PFOR_LENGTH_UNKNOWN));
        match layout {
            ValueLayout::Legacy => {
                assert!(!short, "the legacy value layout has no short-form flag");
                assert!(
                    metadata_offset <= LEGACY_OFFSET_MAX,
                    "metadata_offset {metadata_offset} overflows the {LEGACY_OFFSET_BITS}-bit slot"
                );
                (metadata_offset << LEGACY_OFFSET_SHIFT) | (slot << LEGACY_LENGTH_SHIFT)
            }
            ValueLayout::Flagged => {
                assert!(
                    metadata_offset <= FLAGGED_OFFSET_MAX,
                    "metadata_offset {metadata_offset} overflows the {FLAGGED_OFFSET_BITS}-bit slot"
                );
                assert!(
                    !short || postings_length < PFOR_LENGTH_UNKNOWN,
                    "a short-form body must fit the length slot"
                );
                (metadata_offset << FLAGGED_OFFSET_SHIFT)
                    | ((short as u64) << SHORT_FLAG_SHIFT)
                    | (slot << FLAGGED_LENGTH_SHIFT)
            }
        }
    }

    /// Pack a `(doc_id, tf)` pair into the inline-form FST value. The
    /// low bit is always 1. Layout-independent.
    #[inline]
    pub(crate) fn pack_inline(doc_id: u32, tf: u32) -> u64 {
        assert!(
            tf <= INLINE_TF_MAX,
            "tf {tf} overflows the inline 30-bit slot (max {INLINE_TF_MAX})"
        );
        1 | ((doc_id as u64) << DOC_ID_SHIFT) | ((tf as u64) << TF_SHIFT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pfor_round_trip_both_layouts() {
        for &(offset, len) in &[
            (0u64, 20u32),
            (1, 128),
            (20, 4096),
            (1 << 20, 1 << 16),
            ((1u64 << 34) - 1, (1 << 20) - 1),
            (1u64 << 34, PFOR_LENGTH_UNKNOWN - 1),
            (FLAGGED_OFFSET_MAX, 1 << 16),
        ] {
            for layout in [ValueLayout::Legacy, ValueLayout::Flagged] {
                let packed = FstValue::pack_pfor(offset, len, layout, false);
                assert_eq!(packed & 1, 0, "postings form must have low bit clear");
                assert_eq!(
                    FstValue::unpack(packed, layout),
                    FstValue::Pfor {
                        metadata_offset: offset,
                        postings_length_hint: Some(len),
                        short: false,
                    }
                );
            }
            let packed = FstValue::pack_pfor(
                offset,
                len.min(PFOR_LENGTH_UNKNOWN - 1),
                ValueLayout::Flagged,
                true,
            );
            assert_eq!(
                FstValue::unpack(packed, ValueLayout::Flagged),
                FstValue::Pfor {
                    metadata_offset: offset,
                    postings_length_hint: Some(len.min(PFOR_LENGTH_UNKNOWN - 1)),
                    short: true,
                }
            );
        }
        // The legacy slot reaches one bit further.
        let packed = FstValue::pack_pfor(LEGACY_OFFSET_MAX, 7, ValueLayout::Legacy, false);
        assert_eq!(
            FstValue::unpack(packed, ValueLayout::Legacy),
            FstValue::Pfor {
                metadata_offset: LEGACY_OFFSET_MAX,
                postings_length_hint: Some(7),
                short: false,
            }
        );
    }

    #[test]
    fn a_legacy_odd_offset_is_not_a_short_flag_under_its_own_layout() {
        // Under the legacy layout bit 1 is the offset's low bit. Reading
        // such a value with the legacy layout must give the odd offset
        // back, never a short-form term — the layout is chosen by blob
        // version precisely so this cannot happen.
        let packed = FstValue::pack_pfor(4097, 40, ValueLayout::Legacy, false);
        assert_eq!(
            FstValue::unpack(packed, ValueLayout::Legacy),
            FstValue::Pfor {
                metadata_offset: 4097,
                postings_length_hint: Some(40),
                short: false,
            }
        );
    }

    #[test]
    #[should_panic(expected = "no short-form flag")]
    fn legacy_layout_refuses_short() {
        let _ = FstValue::pack_pfor(0, 8, ValueLayout::Legacy, true);
    }

    #[test]
    fn oversize_length_becomes_unknown() {
        const SIXTY_FOUR_MIB: u32 = 64 << 20;
        for &len in &[PFOR_LENGTH_UNKNOWN, PFOR_LENGTH_UNKNOWN + 1, SIXTY_FOUR_MIB] {
            for layout in [ValueLayout::Legacy, ValueLayout::Flagged] {
                let packed = FstValue::pack_pfor(4096, len, layout, false);
                assert_eq!(packed & 1, 0, "postings form must have low bit clear");
                assert_eq!(
                    FstValue::unpack(packed, layout),
                    FstValue::Pfor {
                        metadata_offset: 4096,
                        postings_length_hint: None,
                        short: false,
                    },
                    "length {len} must degrade to the header-probe sentinel"
                );
            }
        }
    }

    #[test]
    fn largest_expressible_length_is_not_the_sentinel() {
        let packed =
            FstValue::pack_pfor(4096, PFOR_LENGTH_UNKNOWN - 1, ValueLayout::Flagged, false);
        assert_eq!(
            FstValue::unpack(packed, ValueLayout::Flagged),
            FstValue::Pfor {
                metadata_offset: 4096,
                postings_length_hint: Some(PFOR_LENGTH_UNKNOWN - 1),
                short: false,
            }
        );
    }

    #[test]
    fn inline_round_trip() {
        let cases = [
            (0u32, 0u32),
            (1, 1),
            (500_000, 7),
            (u32::MAX, INLINE_TF_MAX),
        ];
        for &(doc_id, tf) in &cases {
            let packed = FstValue::pack_inline(doc_id, tf);
            assert_eq!(packed & 1, 1, "inline form must have low bit set");
            for layout in [ValueLayout::Legacy, ValueLayout::Flagged] {
                assert_eq!(
                    FstValue::unpack(packed, layout),
                    FstValue::Inline { doc_id, tf }
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "overflows the inline 30-bit slot")]
    fn inline_tf_overflow_panics() {
        let _ = FstValue::pack_inline(0, INLINE_TF_MAX + 1);
    }

    #[test]
    fn flag_bit_distinguishes_forms() {
        let pfor = FstValue::pack_pfor(42, 128, ValueLayout::Flagged, false);
        let inline = FstValue::pack_inline(42, 7);
        assert_ne!(pfor & 1, inline & 1);
    }

    #[test]
    fn flagged_neighbours_differ_only_in_their_low_bits() {
        // Two terms 40 bytes apart in the postings region: the values
        // share every bit above the length field's, which is what lets
        // the FST fold the common part into the shared prefix.
        let a = FstValue::pack_pfor(1 << 30, 40, ValueLayout::Flagged, false);
        let b = FstValue::pack_pfor((1 << 30) + 40, 24, ValueLayout::Flagged, true);
        assert!(b > a, "ascending offsets pack to ascending values");
        assert!(b - a < 1u64 << FLAGGED_OFFSET_SHIFT << 6, "residual stays small");
    }
}
