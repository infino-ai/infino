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

use crate::superfile::fts::{bm25::BOUND_ROUNDING_SLACK, reader::metadata::ColumnMeta};

/// Decodes one term's slots into bounds at the query's scoring.
///
/// A stored maximum is a score at the statistics the build had, so every
/// factor that moves the scored value away from them is owed as a
/// multiplier, and they compose into one: the term's `idf / local_idf`
/// (exact — the score is linear in idf — and the only factor on the
/// default table-wide path, where a repeated-term weight also lands) and
/// the column's [`ColumnMeta::bound_scale`] (a supremum, so loosening
/// rather than exact; `1.0` unless the query moves `k1`, `b` or the
/// average length the norms are computed at, which a multi-superfile
/// table does for every superfile built at another average).
///
/// A decoded score is nudged up one `f32` ULP first. At the statistics the
/// build had that alone keeps the bound above every score in the block,
/// since the kernel computes the same score bit for bit. Rescaled, it does
/// not: the stored maximum, the kernel's score and the scale each round on
/// their own, so the scale also carries [`BOUND_ROUNDING_SLACK`]. Without
/// it a bound can land a step or two below a document's score, and a floor
/// equal to that score, which ties must survive, skips the document's
/// block.
///
/// How short the bound can fall, in units of `u = 2^-24` relative. The
/// kernel scores `fl(fl(idf·tf) / fl(tf + norm))` and the stored maximum is
/// the same expression at the local idf and the build's norm:
///
/// - The kernel's multiply, add and divide can each raise its score by
///   `u`, and the stored maximum's can each lower it by `u`: `6u`.
/// - The one-ULP nudge gives back at least `u`.
/// - The idf ratio, the column scale (its two adds and its divide), their
///   product, the slack's multiply and the final multiply can each lower
///   the bound by `u`: `7u`.
///
/// That is about `12u` when the query rescores the norms. When it keeps
/// them, both sides share `fl(tf + norm)` and the column scale is exactly
/// `1.0`, which drops the two adds and the scale's four roundings: about
/// `6u`. The slack is `16u`, above both, and costs pruning a millionth of
/// the score.
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
        let bound_scale = col.bound_scale();
        // Both factors, not their product: two factors away from one can
        // multiply to exactly one and still owe the slack.
        let scale = match idf_ratio == 1.0 && bound_scale == 1.0 {
            true => 1.0,
            false => idf_ratio * bound_scale * (1.0 + BOUND_ROUNDING_SLACK),
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

    use std::ops::RangeInclusive;

    use bytes::Bytes;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    use super::*;
    use crate::superfile::fts::{
        bm25,
        builder::FtsBuilder,
        reader::{FtsReader, ScoringView},
    };

    /// Seeds for the random sweeps, so a failure reproduces.
    const SWEEP_SEED: u64 = 0x5eed_b0d5;
    /// Draws in the sweep at the build's norms.
    const SWEEP_SAMPLES: usize = 2_000_000;
    /// Idfs drawn, from a term in most documents to a very rare one.
    const SWEEP_IDFS: RangeInclusive<f32> = 0.05..=15.0;
    /// The relative distance from the local idf of a table-wide idf drawn
    /// near it: within a few hundred `f32` steps of the ratio being one,
    /// where a table barely moves the term's statistics.
    const SWEEP_NEAR_IDF: RangeInclusive<f32> = -2e-5..=2e-5;
    /// Term frequencies drawn.
    const SWEEP_TFS: RangeInclusive<u32> = 1..=300;
    /// Length norms drawn: a short title to a document many times the
    /// average.
    const SWEEP_NORMS: RangeInclusive<f32> = 0.3..=300.0;
    /// A case the sweep must keep covering, found by a wider random search:
    /// local idf, table idf, tf and norm. With a slack of one `f32` epsilon
    /// its bound lands below its score.
    const SWEEP_KNOWN_CASE: (f32, f32, u32, f32) = (3.229_955, 4.713_608_7, 82, 218.124_62);

    /// Documents in the rescored corpus: one per length, so many norm
    /// buckets are occupied.
    const RESCORE_DOCS: u32 = 400;
    /// Averages the corpus is rescored at, as multiples of the one it was
    /// built at: smaller, just off it either side, and larger.
    const RESCORE_AVGDL_FACTORS: [f32; 9] =
        [0.25, 0.6, 0.999, 1.000_001, 1.001, 1.3, 2.0, 4.0, 9.0];
    /// Draws per rescored average.
    const RESCORE_SAMPLES: usize = 250_000;

    /// A reader over one column `body`, documents `texts` in order.
    fn reader(texts: impl IntoIterator<Item = String>) -> FtsReader {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for (i, text) in texts.into_iter().enumerate() {
            b.add_doc(0, i as u32, &text).expect("doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open")
    }

    /// A table-wide idf for a term whose own is `local`: half the time
    /// anywhere, half the time within a few steps of `local`.
    fn draw_idf(rng: &mut StdRng, local: f32) -> f32 {
        match rng.random_bool(0.5) {
            true => rng.random_range(SWEEP_IDFS),
            false => local * (1.0 + rng.random_range(SWEEP_NEAR_IDF)),
        }
    }

    /// Asserts the bound `decoder` gives the score stored at `local` idf
    /// and `stored_norm` is at or above the score the kernel computes at
    /// `idf` and `norm`.
    fn assert_bounds(
        decoder: BoundDecoder,
        (local, idf, tf): (f32, f32, u32),
        stored_norm: f32,
        norm: f32,
    ) {
        let stored = bm25::score_with_dl_norm_k1(local, tf, stored_norm);
        let score = bm25::score_with_dl_norm_k1(idf, tf, norm);
        let bound = decoder.bound(stored.to_bits());
        assert!(
            bound >= score,
            "local idf {local}, idf {idf}, tf {tf}, stored norm {stored_norm}, norm {norm}: \
             bound {bound} below score {score}"
        );
    }

    /// A bound rescaled to a table-wide idf is never below the score the
    /// kernel computes at that idf for the document it was baked from.
    /// The stored maximum is the kernel's own score at the term's local
    /// idf; the kernel then scores at the table's. Each side rounds on its
    /// own, and the ratio between them rounds too, so the decoded bound
    /// must carry slack for all of it: a bound a step below a document's
    /// score lets a floor equal to that score skip the document's block.
    /// The ranges are wide enough that a slack of one `f32` epsilon fails.
    #[test]
    fn a_rescaled_bound_is_never_below_the_score_it_bounds() {
        let r = reader(["a b".to_string()]);
        let col = &r.columns[0];
        assert_eq!(col.bound_scale(), 1.0);
        let (local, idf, tf, norm) = SWEEP_KNOWN_CASE;
        assert_bounds(
            BoundDecoder::new(col, idf, local),
            (local, idf, tf),
            norm,
            norm,
        );
        let mut rng = StdRng::seed_from_u64(SWEEP_SEED);
        for _ in 0..SWEEP_SAMPLES {
            let local = rng.random_range(SWEEP_IDFS);
            let idf = draw_idf(&mut rng, local);
            let tf = rng.random_range(SWEEP_TFS);
            let norm = rng.random_range(SWEEP_NORMS);
            assert_bounds(
                BoundDecoder::new(col, idf, local),
                (local, idf, tf),
                norm,
                norm,
            );
        }
    }

    /// The same at another average length, the path every superfile of a
    /// multi-superfile table built at an average other than the table's
    /// takes: the kernel's norm is no longer the stored one, and the
    /// column's bound scale rounds on its own. The idf may also stay the
    /// file's own, so the column scale is the only factor.
    #[test]
    fn a_bound_rescored_at_another_average_is_never_below_the_score_it_bounds() {
        let r = reader((0..RESCORE_DOCS).map(|len| "w".to_string() + &" p".repeat(len as usize)));
        let base = &r.columns[0];
        let baked = base.avgdl();
        let mut rng = StdRng::seed_from_u64(SWEEP_SEED);
        for factor in RESCORE_AVGDL_FACTORS {
            let scoring = ScoringView {
                norms: None,
                params: None,
                avgdl: Some(("body", baked * factor)),
            };
            let view = r.with_scoring(&scoring).expect("the average moves");
            let col = &view.columns[0];
            if factor > 1.0 {
                assert!(col.bound_scale() > 1.0, "factor {factor}: no column scale");
            }
            for _ in 0..RESCORE_SAMPLES {
                let doc = rng.random_range(0..RESCORE_DOCS);
                let local = rng.random_range(SWEEP_IDFS);
                let idf = match rng.random_bool(0.25) {
                    true => local,
                    false => draw_idf(&mut rng, local),
                };
                let tf = rng.random_range(SWEEP_TFS);
                let stored_norm = base.dl_norm_k1().get(doc);
                let norm = col.dl_norm_k1().get(doc);
                let decoder = BoundDecoder::new(col, idf, local);
                assert_bounds(decoder, (local, idf, tf), stored_norm, norm);
            }
        }
    }

    #[test]
    fn a_decoded_bound_takes_the_idf_ratio_and_the_column_scale() {
        let r = reader(["a b".to_string(), "a b c d e f".to_string()]);
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
            0.25f32.next_up() * (2.0 * scale * (1.0 + BOUND_ROUNDING_SLACK))
        );
        // A zero local idf (a term in every document) leaves the ratio at 1.
        let none = BoundDecoder::new(col, 2.0, 0.0);
        assert_eq!(
            none.bound(1.0f32.to_bits()),
            1.0f32.next_up() * (scale * (1.0 + BOUND_ROUNDING_SLACK))
        );
    }

    /// Two factors away from one whose product rounds to exactly one still
    /// owe the slack: the decoder tests each factor, not the product.
    #[test]
    fn factors_that_multiply_to_one_still_carry_the_slack() {
        let r = reader(["a b".to_string(), "a b c d e f".to_string()]);
        let view = r.with_bm25_override(bm25::Bm25Params::new(0.9, 0.4));
        let col = &view.columns[0];
        let scale = col.bound_scale();
        assert!(scale > 1.0);
        // An idf ratio that cancels the column scale.
        let local = 1.0;
        let idf = 1.0 / scale;
        assert_eq!(idf / local * scale, 1.0, "the factors must cancel exactly");
        let decoder = BoundDecoder::new(col, idf, local);
        assert_eq!(
            decoder.bound(1.0f32.to_bits()),
            1.0f32.next_up() * (1.0 + BOUND_ROUNDING_SLACK)
        );
    }
}
