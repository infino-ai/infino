// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Turning a term's stored 4-byte block-max slots into score upper bounds
//! at the statistics a query actually scores with.
//!
//! Every ranked kernel prunes on `bound <= threshold`, so a bound has to
//! sit at or above every score in its block and, for pruning to keep its
//! power, as close to the true maximum as the format allows. A slot holds
//! the exact `f32` bits of the block's maximum at the statistics the build
//! had; [`BoundDecoder`] is the single place it is brought to the query's,
//! so the single-term walk and the multi-term cursors cannot drift apart.

use crate::superfile::fts::reader::metadata::ColumnMeta;

/// Decodes one term's slots into bounds at the query's scoring.
///
/// A stored maximum is a score at the statistics the build had, so every
/// factor that moves the scored value away from them is owed as a
/// multiplier, and they compose into one: the term's `idf / local_idf`
/// (exact — the score is linear in idf — and the only factor on the
/// default table-wide path, where a repeated-term weight also lands) and
/// the column's [`ColumnMeta::bound_scale`] (a supremum, so loosening
/// rather than exact; `1.0` unless the query overrides `k1`/`b`). A decoded
/// score is nudged up one `f32` ULP first, so the multiply's rounding
/// cannot dip below a score-tied document and let a rising floor skip it.
#[derive(Debug, Clone, Copy)]
pub(super) struct BoundDecoder {
    scale: f32,
}

impl BoundDecoder {
    /// `idf_weight` is the idf the term's scores use (any global override
    /// and repeated-term weight folded in); `local_idf` is what this
    /// superfile's own statistics give, which is what the bounds were
    /// baked with.
    pub(super) fn new(col: &ColumnMeta, idf_weight: f32, local_idf: f32) -> Self {
        let idf_ratio = match local_idf > 0.0 && idf_weight != local_idf {
            true => idf_weight / local_idf,
            false => 1.0,
        };
        Self {
            scale: idf_ratio * col.bound_scale(),
        }
    }

    /// A decoder for a cursor that only counts matches and never scores: its
    /// bounds are never compared, so it needs neither the idf nor the
    /// column's bound scale, which under a parameter override is computed
    /// from norms the count-only build must not read.
    pub(super) fn unscored() -> Self {
        Self { scale: 1.0 }
    }

    /// The upper bound a raw slot stands for.
    #[inline]
    pub(super) fn bound(&self, raw: u32) -> f32 {
        f32::from_bits(raw).next_up() * self.scale
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;

    use super::*;
    use crate::superfile::fts::{
        bm25, builder::FtsBuilder, reader::FtsReader, tokenize::StandardTokenizer,
    };

    #[test]
    fn a_decoded_bound_takes_the_idf_ratio_and_the_column_scale() {
        let mut b = FtsBuilder::new(Arc::new(StandardTokenizer));
        b.register_column("body".into(), false).expect("register");
        b.add_doc(0, 0, "a b").expect("doc");
        b.add_doc(0, 1, "a b c d e f").expect("doc");
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        // At the declared parameters the stored bounds are exact.
        let declared = &r.columns[0];
        assert_eq!(declared.bound_scale(), 1.0);
        let same = BoundDecoder::new(declared, 1.0, 1.0);
        assert_eq!(same.bound(0.25f32.to_bits()), 0.25f32.next_up());

        // An override owes the column's supremum factor on top of the idf
        // ratio: idf 2.0 against a local 1.0 doubles.
        let view = r.with_bm25_override(bm25::Bm25Params::new(0.9, 0.4));
        let col = &view.columns[0];
        let scale = col.bound_scale();
        let doubled = BoundDecoder::new(col, 2.0, 1.0);
        assert_eq!(
            doubled.bound(0.25f32.to_bits()),
            0.25f32.next_up() * (2.0 * scale)
        );
        // A zero local idf (a term in every document) leaves the ratio at 1.
        let none = BoundDecoder::new(col, 2.0, 0.0);
        assert_eq!(none.bound(1.0f32.to_bits()), 1.0f32.next_up() * scale);
    }
}
