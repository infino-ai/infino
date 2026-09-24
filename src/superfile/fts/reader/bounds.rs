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

use crate::superfile::{
    format::{
        self,
        fts::{BlobLayout, BlockLayout, SkipLayout},
    },
    fts::reader::metadata::ColumnMeta,
};

/// The FTS blob version, as far as reading a block-max slot is
/// concerned: what the 4-byte slot (skip entry and coarse entry alike)
/// encodes and which average document length it was baked at. Named by
/// version rather than by property so a reader of the match arms sees
/// the same numbers the file header carries.
///
/// [`Self::for_version`] is also the reader's accept list: a version
/// with no arm here cannot be opened. So a new blob version has to be
/// added here deliberately, and can never fall through to an older
/// arm's decode by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StoredBound {
    /// [`format::fts::VERSION_V7`]: as [`Self::V6`] for every slot —
    /// the version changes the rare-term layout and the dictionary
    /// value, not the bounds or the average they are baked at.
    V7,
    /// [`format::fts::VERSION_V6`]: exact `f32` bits of the maximum, in
    /// the scorer's own scale, at the table-wide average the file
    /// declares — which is what the reader scores the file at, so
    /// nothing needs correcting.
    V6,
    /// [`format::fts::VERSION_V5`]: exact `f32` bits carrying the
    /// `(k1 + 1)` factor the scorer has since dropped, at the file's
    /// row-count average.
    V5,
    /// [`format::fts::VERSION_V1_LEGACY`] through
    /// [`format::fts::VERSION_V4`]: `ceil(max × scale)` fixed point,
    /// otherwise as [`Self::V5`]. These blobs also predate the coarse
    /// table.
    V1ToV4,
}

impl StoredBound {
    /// The layout table of a version this variant stands for. `V1ToV4`
    /// reads as `V4`: the three fields the bound decoder's callers take
    /// from it (coarse table, block header, skip entries) are the same
    /// across `V1`–`V4`.
    fn layout(self) -> BlobLayout {
        let version = match self {
            Self::V7 => format::fts::VERSION_V7,
            Self::V6 => format::fts::VERSION_V6,
            Self::V5 => format::fts::VERSION_V5,
            Self::V1ToV4 => format::fts::VERSION_V4,
        };
        BlobLayout::for_version(version).expect("every variant names a known version")
    }

    /// Which block header the version's posting blocks carry.
    pub(super) fn block_layout(self) -> BlockLayout {
        self.layout().block
    }

    /// How the version's skip tables locate their blocks.
    pub(super) fn skip_layout(self) -> SkipLayout {
        self.layout().skip
    }

    /// The variant for a blob version, or `None` for a version this
    /// reader does not know — which the open path turns into an
    /// unsupported-version error.
    pub(super) fn for_version(version: u32) -> Option<Self> {
        match version {
            format::fts::VERSION_V7 => Some(Self::V7),
            format::fts::VERSION_V6 => Some(Self::V6),
            format::fts::VERSION_V5 => Some(Self::V5),
            format::fts::VERSION_V1_LEGACY
            | format::fts::VERSION_V2
            | format::fts::VERSION_V3
            | format::fts::VERSION_V4 => Some(Self::V1ToV4),
            _ => None,
        }
    }

    /// Whether each PFOR term's region ends with a coarse block-max table
    /// (one slot per [`format::fts::COARSE_BLOCK_MAX_SPAN`] blocks).
    pub(super) fn has_coarse(self) -> bool {
        self.layout().coarse
    }

    /// Whether the file's declared average document length is the one to
    /// score it at, or a row-count average the reader must correct.
    pub(super) fn declares_scoring_average(self) -> bool {
        matches!(self, Self::V6 | Self::V7)
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
            scale: idf_ratio * col.bound_scale(),
        }
    }

    /// The upper bound a raw slot stands for.
    #[inline]
    pub(super) fn bound(&self, raw: u32) -> f32 {
        let stored = match self.stored {
            StoredBound::V7 | StoredBound::V6 | StoredBound::V5 => f32::from_bits(raw).next_up(),
            StoredBound::V1ToV4 => {
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
    fn every_accepted_version_has_an_arm_and_the_next_one_is_refused() {
        // The mapping doubles as the accept list, so a version added to
        // the format without an arm here fails to open rather than
        // decoding under an older version's rules. The builder's
        // current version must map to the variant scored at the
        // declared average.
        let accepted = [
            (format::fts::VERSION_V1_LEGACY, StoredBound::V1ToV4),
            (format::fts::VERSION_V2, StoredBound::V1ToV4),
            (format::fts::VERSION_V3, StoredBound::V1ToV4),
            (format::fts::VERSION_V4, StoredBound::V1ToV4),
            (format::fts::VERSION_V5, StoredBound::V5),
            (format::fts::VERSION_V6, StoredBound::V6),
            (format::fts::VERSION_V7, StoredBound::V7),
        ];
        for (version, want) in accepted {
            assert_eq!(
                StoredBound::for_version(version),
                Some(want),
                "version {version}"
            );
        }
        assert_eq!(StoredBound::for_version(format::fts::VERSION_V7 + 1), None);
        assert_eq!(StoredBound::for_version(0), None);

        assert!(
            StoredBound::V7.has_coarse()
                && StoredBound::V6.has_coarse()
                && StoredBound::V5.has_coarse()
        );
        assert!(!StoredBound::V1ToV4.has_coarse());
        assert!(StoredBound::V7.declares_scoring_average());
        assert!(StoredBound::V6.declares_scoring_average());
        assert!(!StoredBound::V5.declares_scoring_average());
        assert!(!StoredBound::V1ToV4.declares_scoring_average());
    }

    #[test]
    fn a_decoded_bound_takes_the_idf_ratio_and_the_column_scale() {
        let mut b = FtsBuilder::new(Arc::new(AsciiLowerTokenizer));
        b.register_column("body".into(), false).expect("register");
        b.add_doc(0, 0, "a b").expect("doc");
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let col = r.columns[0].clone().with_bound_scale_for_test(0.5);

        // idf 2.0 against a local 1.0 doubles; the column halves: net 1.0.
        let unit = BoundDecoder::new(StoredBound::V6, &col, 2.0, 1.0);
        assert_eq!(unit.bound(0.25f32.to_bits()), 0.25f32.next_up());
        // V5 scores decode as V6 ones do; only the column scale (set on
        // open) tells them apart.
        let legacy = BoundDecoder::new(StoredBound::V5, &col, 1.0, 1.0);
        assert_eq!(legacy.bound(0.25f32.to_bits()), 0.25f32.next_up() * 0.5);
        // Fixed point: one step of slack, then the same multiplier.
        let fixed = BoundDecoder::new(StoredBound::V1ToV4, &col, 3.0, 1.0);
        assert_eq!(
            fixed.bound(250),
            251.0 / format::fts::BLOCK_MAX_BM25_FIXED_POINT_SCALE * 3.0 * 0.5
        );
        // A zero local idf (a term in every document) leaves the ratio at 1.
        let none = BoundDecoder::new(StoredBound::V6, &col, 2.0, 0.0);
        assert_eq!(none.bound(1.0f32.to_bits()), 1.0f32.next_up() * 0.5);
    }
}
