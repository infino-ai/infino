// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Turning a term's stored 4-byte block-max slots into score upper bounds
//! at the statistics a query actually scores with.
//!
//! Every ranked kernel prunes on `bound <= threshold`, so a bound has to
//! sit at or above every score in its block and, for pruning to keep its
//! power, as close to the true maximum as the format allows. How a slot
//! is read differs by blob version ([`StoredBound`]); [`BoundDecoder`] is
//! the single place that difference is interpreted, so the single-term
//! walk and the multi-term cursors cannot drift apart.

use crate::superfile::{format, fts::reader::metadata::ColumnMeta};

/// How a blob's 4-byte block-max slot (skip entry and coarse entry
/// alike) encodes the block's maximum score. Decided once per blob from
/// its version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StoredBound {
    /// `V6`: exact `f32` bits of the maximum, in the scorer's own scale,
    /// at the average document length the file declares — which is what
    /// the reader scores the file at, so nothing needs correcting.
    Exact,
    /// `V5`: exact `f32` bits carrying the `(k1 + 1)` factor the scorer
    /// has since dropped, at the file's row-count average.
    Legacy,
    /// `V1`–`V4`: `ceil(max × scale)` fixed point, otherwise as
    /// [`Self::Legacy`]. These blobs also predate the coarse table.
    LegacyFixedPoint,
}

impl StoredBound {
    pub(super) fn for_version(version: u32) -> Self {
        match version {
            format::fts::VERSION_V6 => Self::Exact,
            format::fts::VERSION_V5 => Self::Legacy,
            _ => Self::LegacyFixedPoint,
        }
    }

    /// Whether each PFOR term's region ends with a coarse block-max table
    /// (one slot per [`format::fts::COARSE_BLOCK_MAX_SPAN`] blocks).
    pub(super) fn has_coarse(self) -> bool {
        self != Self::LegacyFixedPoint
    }

    /// Whether the file's declared average document length is the one to
    /// score it at, or a row-count average the reader must correct.
    pub(super) fn declares_scoring_average(self) -> bool {
        self == Self::Exact
    }
}

/// Decodes one term's slots into bounds at the query's scoring.
///
/// A stored maximum is a score at the statistics the build had, so every
/// factor that moves the scored value away from them is owed as a
/// multiplier, and they compose into one: the term's `idf / local_idf`
/// (exact — the score is linear in idf — and the only factor on the
/// default table-wide path, where a repeated-term weight also lands) and
/// the column's [`ColumnMeta::bound_scale`] (a supremum, so loosening
/// rather than exact; `1.0` unless the query overrides `k1`/`b` or the
/// file predates the current version). A decoded score is nudged up one
/// `f32` ULP first, so the multiply's rounding cannot dip below a
/// score-tied document and let a rising floor skip it.
#[derive(Debug, Clone, Copy)]
pub(super) struct BoundDecoder {
    stored: StoredBound,
    scale: f32,
}

impl BoundDecoder {
    /// `idf_weight` is the idf the term's scores use (any global override
    /// and repeated-term weight folded in); `local_idf` is what this
    /// superfile's own statistics give, which is what the bounds were
    /// baked with.
    pub(super) fn new(
        stored: StoredBound,
        col: &ColumnMeta,
        idf_weight: f32,
        local_idf: f32,
    ) -> Self {
        let idf_ratio = match local_idf > 0.0 && idf_weight != local_idf {
            true => idf_weight / local_idf,
            false => 1.0,
        };
        Self {
            stored,
            scale: idf_ratio * col.bound_scale,
        }
    }

    /// The upper bound a raw slot stands for.
    #[inline]
    pub(super) fn bound(&self, raw: u32) -> f32 {
        let stored = match self.stored {
            StoredBound::Exact | StoredBound::Legacy => f32::from_bits(raw).next_up(),
            StoredBound::LegacyFixedPoint => {
                raw.saturating_add(1) as f32 / format::fts::BLOCK_MAX_BM25_FIXED_POINT_SCALE
            }
        };
        stored * self.scale
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;

    use super::*;
    use crate::superfile::fts::{
        builder::FtsBuilder, reader::FtsReader, tokenize::AsciiLowerTokenizer,
    };

    #[test]
    fn versions_map_to_their_slot_meaning() {
        assert_eq!(
            StoredBound::for_version(format::fts::VERSION_V6),
            StoredBound::Exact
        );
        assert_eq!(
            StoredBound::for_version(format::fts::VERSION_V5),
            StoredBound::Legacy
        );
        for v in [
            format::fts::VERSION_V4,
            format::fts::VERSION_V3,
            format::fts::VERSION_V2,
        ] {
            assert_eq!(StoredBound::for_version(v), StoredBound::LegacyFixedPoint);
        }
        assert!(StoredBound::Exact.has_coarse() && StoredBound::Legacy.has_coarse());
        assert!(!StoredBound::LegacyFixedPoint.has_coarse());
        assert!(StoredBound::Exact.declares_scoring_average());
        assert!(!StoredBound::Legacy.declares_scoring_average());
    }

    #[test]
    fn a_decoded_bound_takes_the_idf_ratio_and_the_column_scale() {
        let mut b = FtsBuilder::new(Arc::new(AsciiLowerTokenizer));
        b.register_column("body".into(), false).expect("register");
        b.add_doc(0, 0, "a b").expect("doc");
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let mut col = r.columns[0].clone();
        col.bound_scale = 0.5;

        // idf 2.0 against a local 1.0 doubles; the column halves: net 1.0.
        let unit = BoundDecoder::new(StoredBound::Exact, &col, 2.0, 1.0);
        assert_eq!(unit.bound(0.25f32.to_bits()), 0.25f32.next_up());
        // Legacy scores decode identically to exact ones; only the column
        // scale (set on open) tells them apart.
        let legacy = BoundDecoder::new(StoredBound::Legacy, &col, 1.0, 1.0);
        assert_eq!(legacy.bound(0.25f32.to_bits()), 0.25f32.next_up() * 0.5);
        // Fixed point: one step of slack, then the same multiplier.
        let fixed = BoundDecoder::new(StoredBound::LegacyFixedPoint, &col, 3.0, 1.0);
        assert_eq!(
            fixed.bound(250),
            251.0 / format::fts::BLOCK_MAX_BM25_FIXED_POINT_SCALE * 3.0 * 0.5
        );
        // A zero local idf (a term in every document) leaves the ratio at 1.
        let none = BoundDecoder::new(StoredBound::Exact, &col, 2.0, 0.0);
        assert_eq!(none.bound(1.0f32.to_bits()), 1.0f32.next_up() * 0.5);
    }
}
