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

/// The relative slack a bound decoded at a different scale carries, folded
/// into the scale. The kernel scores `fl(fl(idf·tf) / d)` and the stored
/// maximum is the same expression at the term's local idf, with the same
/// `d = fl(tf + norm)`, so the two differ by the idf alone, exactly in real
/// arithmetic. In `f32` each side's multiply and divide round by up to one
/// unit `u = 2^-24` relative, the one-ULP nudge gives back at least `u`, and
/// the ratio, its product with the column scale, this factor and the final
/// multiply round by up to `u` each. The bound can therefore fall short of
/// the score by about `5u` before slack, and `16u` leaves it above after
/// every rounding. Cheap for pruning: a millionth of the score.
const RESCALE_SLACK: f32 = 8.0 * f32::EPSILON;

/// Decodes one term's slots into bounds at the query's scoring.
///
/// A stored maximum is a score at the statistics the build had, so every
/// factor that moves the scored value away from them is owed as a
/// multiplier, and they compose into one: the term's `idf / local_idf`
/// (exact — the score is linear in idf — and the only factor on the
/// default table-wide path, where a repeated-term weight also lands) and
/// the column's [`ColumnMeta::bound_scale`] (a supremum, so loosening
/// rather than exact; `1.0` unless the query overrides `k1`/`b`). A decoded
/// score is nudged up one `f32` ULP first. At the statistics the build had
/// that alone keeps the bound above every score in the block, since the
/// kernel computes the same score bit for bit. Rescaled, it does not: the
/// stored maximum and the kernel's score round independently, so the
/// scale carries [`RESCALE_SLACK`] as well. Without it a bound can land a
/// step or two below a document's score, and a floor equal to that score,
/// which ties must survive, skips the document's block.
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
        let scale = idf_ratio * col.bound_scale();
        let scale = match scale == 1.0 {
            true => scale,
            false => scale * (1.0 + RESCALE_SLACK),
        };
        Self { scale }
    }

    /// A decoder for a cursor that only counts matches and never scores: its
    /// bounds are never compared, so it needs neither the idf nor the
    /// column's bound scale, which under a parameter override is computed
    /// from norms the count-only build must not read.
    pub(super) fn unscored() -> Self {
        Self { scale: 1.0 }
    }

    /// The decoder's scale as bits: two decoders with the same bits turn
    /// every slot into the same bound.
    pub(super) fn scale_bits(&self) -> u32 {
        self.scale.to_bits()
    }

    /// The upper bound a raw slot stands for.
    #[inline]
    pub(super) fn bound(&self, raw: u32) -> f32 {
        f32::from_bits(raw).next_up() * self.scale
    }
}

#[cfg(test)]
mod tests {

    use bytes::Bytes;

    use super::*;
    use crate::superfile::fts::{bm25, builder::FtsBuilder, reader::FtsReader};

    /// The term's own idfs the sweep rescales from: below one, one, the
    /// value a table-wide search lost a tied document at, and a rare
    /// term's.
    const SWEEP_LOCAL_IDFS: [f32; 4] = [0.37, 1.0, 3.632_968_4, 9.81];
    /// Table-wide idfs tried on each side of each local one.
    const SWEEP_IDF_STEPS: i32 = 256;
    /// The relative step between them: about one `f32` step of the ratio.
    const SWEEP_IDF_STEP: f32 = 1.0 / 131_072.0;
    /// Term frequencies swept.
    const SWEEP_MAX_TF: u32 = 6;
    /// Length norms swept: `SWEEP_NORMS` of them, from `SWEEP_NORM_MIN` up
    /// in steps of `SWEEP_NORM_STEP`.
    const SWEEP_NORMS: u32 = 64;
    const SWEEP_NORM_MIN: f32 = 0.3;
    const SWEEP_NORM_STEP: f32 = 0.047;

    /// A bound rescaled to a table-wide idf is never below the score the
    /// kernel computes at that idf for the document it was baked from.
    /// The stored maximum is the kernel's own score at the term's local
    /// idf; the kernel then scores at the table's. Each side rounds on its
    /// own, and the ratio between them rounds too, so the decoded bound
    /// must carry slack for all of it: a bound a step below a document's
    /// score lets a floor equal to that score skip the document's block.
    #[test]
    fn a_rescaled_bound_is_never_below_the_score_it_bounds() {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        b.add_doc(0, 0, "a b").expect("doc");
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let col = &r.columns[0];
        assert_eq!(col.bound_scale(), 1.0);
        for local in SWEEP_LOCAL_IDFS {
            for step in -SWEEP_IDF_STEPS..=SWEEP_IDF_STEPS {
                let idf = local * (1.0 + step as f32 * SWEEP_IDF_STEP);
                let decoder = BoundDecoder::new(col, idf, local);
                for tf in 1..=SWEEP_MAX_TF {
                    for n in 0..SWEEP_NORMS {
                        let norm = SWEEP_NORM_MIN + n as f32 * SWEEP_NORM_STEP;
                        let stored = bm25::score_with_dl_norm_k1(local, tf, norm);
                        let score = bm25::score_with_dl_norm_k1(idf, tf, norm);
                        let bound = decoder.bound(stored.to_bits());
                        assert!(
                            bound >= score,
                            "local idf {local}, idf {idf}, tf {tf}, norm {norm}: \
                             bound {bound} below score {score}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_decoded_bound_takes_the_idf_ratio_and_the_column_scale() {
        let mut b = FtsBuilder::new();
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
            0.25f32.next_up() * (2.0 * scale * (1.0 + RESCALE_SLACK))
        );
        // A zero local idf (a term in every document) leaves the ratio at 1.
        let none = BoundDecoder::new(col, 2.0, 0.0);
        assert_eq!(
            none.bound(1.0f32.to_bits()),
            1.0f32.next_up() * (scale * (1.0 + RESCALE_SLACK))
        );
    }
}
