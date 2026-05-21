//! FST-value bit layout — the inline-encoding short-circuit for
//! `df = 1` terms.
//!
//! Every term's FST value is a `u64`. Bit 0 selects form:
//!
//! - `value & 1 == 0` → **PFOR form**. `value >> 1` is the
//!   metadata_offset into the postings region. The reader proceeds
//!   with the existing 20 B-metadata-header / skip-table / PFOR-block
//!   path.
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
//! Why low-bit flag (not high-bit): the `fst` crate VLQ-encodes
//! values, so encoded length grows with magnitude. Putting the flag
//! in the low bit keeps PFOR-form values small (~5–6 bytes VLQ at
//! 16 GB segment scale); only inline values pay the larger encoding
//! (~7–8 bytes VLQ for the composite). High-bit flag would force
//! *every* value to a full ~9-byte encoding.

const DOC_ID_SHIFT: u32 = 1;
const TF_SHIFT: u32 = 33;
/// Maximum `tf` representable in the inline form's 30-bit slot.
/// Real-world per-doc tf is bounded by document length (in tokens),
/// which fits a u16; this limit only exists to guarantee the
/// pack/unpack round-trip in debug.
pub(crate) const INLINE_TF_MAX: u32 = (1 << 30) - 1;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum FstValue {
    /// df ≥ 2 — read the metadata header at `metadata_offset` to
    /// recover df, postings_length, num_blocks, then walk the skip
    /// table and PFOR blocks as before.
    Pfor { metadata_offset: u64 },
    /// df = 1 — the entire posting is right here. No postings-region
    /// read required.
    Inline { doc_id: u32, tf: u32 },
}

impl FstValue {
    #[inline]
    pub(crate) fn unpack(packed: u64) -> Self {
        if packed & 1 == 0 {
            Self::Pfor {
                metadata_offset: packed >> DOC_ID_SHIFT,
            }
        } else {
            let doc_id = (packed >> DOC_ID_SHIFT) as u32;
            let tf = ((packed >> TF_SHIFT) as u32) & INLINE_TF_MAX;
            Self::Inline { doc_id, tf }
        }
    }

    /// Pack a metadata_offset into the PFOR-form FST value. The low
    /// bit is always 0; the offset occupies bits 1..N.
    #[inline]
    pub(crate) fn pack_pfor(metadata_offset: u64) -> u64 {
        debug_assert!(
            metadata_offset < (1u64 << 63),
            "metadata_offset {metadata_offset} overflows the 63-bit PFOR slot"
        );
        metadata_offset << DOC_ID_SHIFT
    }

    /// Pack a `(doc_id, tf)` pair into the inline-form FST value. The
    /// low bit is always 1.
    #[inline]
    pub(crate) fn pack_inline(doc_id: u32, tf: u32) -> u64 {
        debug_assert!(
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
    fn pfor_round_trip() {
        for &offset in &[0u64, 1, 20, 1 << 20, (1 << 34) - 1, 1 << 34] {
            let packed = FstValue::pack_pfor(offset);
            assert_eq!(packed & 1, 0, "PFOR form must have low bit clear");
            assert_eq!(
                FstValue::unpack(packed),
                FstValue::Pfor {
                    metadata_offset: offset
                }
            );
        }
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
            assert_eq!(FstValue::unpack(packed), FstValue::Inline { doc_id, tf });
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "overflows the inline 30-bit slot")]
    fn inline_tf_overflow_panics_in_debug() {
        let _ = FstValue::pack_inline(0, INLINE_TF_MAX + 1);
    }

    #[test]
    fn flag_bit_distinguishes_forms() {
        let pfor = FstValue::pack_pfor(42);
        let inline = FstValue::pack_inline(42, 7);
        assert_ne!(pfor & 1, inline & 1);
    }
}
