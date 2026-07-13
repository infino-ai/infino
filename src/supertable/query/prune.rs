// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Unified superfile-selection (pruning) for the boolean-predicate
//! query paths.
//!
//! FTS (exact + prefix) and SQL scalar filtering ask the *same*
//! question before they touch any superfile bytes: "which superfiles could
//! possibly contain a row this predicate matches?" Each answers it by
//! conservatively evaluating a per-column test against the manifest's
//! summaries — term bloom, term range, scalar min/max — first over the
//! list's part-level aggregates, then over the surviving superfiles'
//! per-superfile summaries.
//!
//! This module owns that two-tier walk so the three call sites
//! (`bm25_search`, `bm25_search_prefix`, the SQL `SupertableProvider`)
//! share one selection path instead of each re-deriving it. The
//! per-leaf math is **not** reimplemented here: each [`PruneLeaf`]
//! delegates to the existing helpers in [`super::skip`] (superfile tier)
//! and [`crate::supertable::manifest::list_prune`] (part tier), so edge
//! behavior — empty-term handling, missing-column "always keep",
//! conservatism — is inherited verbatim.
//!
//! Vector kNN is intentionally *not* a leaf here: its prune signal is a
//! centroid/cutoff test whose cutoff only exists during fan-out, a
//! different shape from these static boolean tests. It keeps its own
//! path.

use std::sync::Arc;

use datafusion::scalar::ScalarValue;

use crate::{
    superfile::fts::reader::BoolMode,
    supertable::{
        error::QueryError,
        manifest::{
            ManifestSnapshot, ScalarStatsAgg, SuperfileEntry,
            list::Manifest,
            list_prune::{prune_parts_for_fts_prefix, prune_parts_for_fts_terms},
            part::PartId,
        },
        query::skip::{
            ScalarOp, ScalarPredicate, fts_bloom_skip, fts_prefix_skip, null_check_may_match,
            null_check_skip, scalar_skip, scalar_value_may_match, scalar_value_set_skip,
        },
    },
};

/// One conjunct of a prune predicate: a per-column test backed by a
/// manifest summary. The full predicate is the **conjunction** of its
/// leaves — a superfile survives only if every leaf keeps it. (A
/// `TermPresence` leaf carries its own intra-leaf OR/AND over terms via
/// [`BoolMode`]; cross-column OR isn't expressible yet and isn't needed
/// — an unprunable predicate simply contributes no leaf and the superfile
/// is kept.)
pub(crate) enum PruneLeaf {
    /// Exact-term presence on an FTS column → term bloom.
    TermPresence {
        column: String,
        terms: Vec<String>,
        mode: BoolMode,
    },
    /// Prefix on an FTS column → term range overlap.
    Prefix { column: String, prefix: Vec<u8> },
    /// Scalar comparison on a scalar column → per-column min/max.
    Scalar(ScalarPredicate),
    /// `column IN (values)` → keep if the min/max could hold any value.
    ScalarValueSet {
        column: String,
        values: Vec<ScalarValue>,
    },
    /// `column IS NULL` (`want_null`) / `IS NOT NULL` → keep via the
    /// per-column null count and all-null check.
    NullCheck { column: String, want_null: bool },
}

impl PruneLeaf {
    /// Identified Which manifest parts this leaf keeps, from the part-level
    /// aggregates (`ManifestPartEntry`). `None` = no part constraint →
    /// keep all parts. The per-superfile tier runs separately.
    pub(crate) fn keep_parts(&self, list: &Manifest) -> Option<Vec<PartId>> {
        match self {
            PruneLeaf::TermPresence {
                column,
                terms,
                mode,
            } => {
                let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
                Some(prune_parts_for_fts_terms(list, column, &refs, *mode))
            }
            PruneLeaf::Prefix { column, prefix } => {
                Some(prune_parts_for_fts_prefix(list, column, prefix))
            }
            PruneLeaf::Scalar(pred) => Some(scalar_keep_parts(list, pred)),
            PruneLeaf::ScalarValueSet { column, values } => {
                Some(scalar_value_set_keep_parts(list, column, values))
            }
            PruneLeaf::NullCheck { column, want_null } => {
                Some(null_check_keep_parts(list, column, *want_null))
            }
        }
    }
}

/// Keep each part whose `column` aggregate satisfies `keep`. A missing
/// aggregate keeps the part (conservative — never a false prune). The
/// stats are length-1 [`ArrayRef`]s decoded when the list loaded, so
/// reading them here is free of per-query Arrow decode.
fn keep_parts_where_agg(
    list: &Manifest,
    column: &str,
    keep: impl Fn(&ScalarStatsAgg) -> bool,
) -> Vec<PartId> {
    list.parts
        .iter()
        .filter_map(|entry| {
            let k = entry.scalar_stats_agg.get(column).is_none_or(&keep);
            k.then_some(entry.part_id)
        })
        .collect()
}

// Keep each part whose aggregate min/max for `column` satisfies `may_match`;
// undecodable bounds keep the part.
fn keep_parts_where(
    list: &Manifest,
    column: &str,
    may_match: impl Fn(&ScalarValue, &ScalarValue) -> bool,
) -> Vec<PartId> {
    keep_parts_where_agg(list, column, |agg| {
        agg_minmax(agg).is_none_or(|(min, max)| may_match(&min, &max))
    })
}

// An aggregate's decoded min/max, or `None` when the bounds don't decode.
fn agg_minmax(agg: &ScalarStatsAgg) -> Option<(ScalarValue, ScalarValue)> {
    match (
        ScalarValue::try_from_array(agg.min.as_ref(), 0),
        ScalarValue::try_from_array(agg.max.as_ref(), 0),
    ) {
        (Ok(min), Ok(max)) => Some((min, max)),
        _ => None,
    }
}

// Part-tier `IS [NOT] NULL` prune; the superfile-tier sibling lives in `skip`.
fn null_check_keep_parts(list: &Manifest, column: &str, want_null: bool) -> Vec<PartId> {
    keep_parts_where_agg(list, column, |agg| null_check_may_match(agg, want_null))
}

// Part-tier scalar prune: keep parts whose min/max could satisfy `pred`.
fn scalar_keep_parts(list: &Manifest, pred: &ScalarPredicate) -> Vec<PartId> {
    keep_parts_where(list, &pred.column, |min, max| {
        scalar_value_may_match(min, max, pred.op, &pred.value)
    })
}

// Part-tier `IN` prune: keep parts whose min/max could hold *any* listed
// value (an `IN` is a disjunction of equalities).
fn scalar_value_set_keep_parts(
    list: &Manifest,
    column: &str,
    values: &[ScalarValue],
) -> Vec<PartId> {
    keep_parts_where(list, column, |min, max| {
        values
            .iter()
            .any(|v| scalar_value_may_match(min, max, ScalarOp::Eq, v))
    })
}

/// Select the superfiles a predicate could match, newest-first in
/// manifest order, applying the two prune tiers (part aggregates →
/// per-superfile summaries). Returns the surviving superfile entries; the
/// caller drives execution over them (search fan-out or DataFusion
/// scan).
///
/// An empty `leaves` slice keeps every superfile (the no-`WHERE` scan).
pub(crate) async fn select_superfiles(
    manifest: &ManifestSnapshot,
    leaves: &[PruneLeaf],
) -> Result<Vec<Arc<SuperfileEntry>>, QueryError> {
    // ---- Tier A: part-level prune (only when a hierarchical list
    // exists; otherwise the flat superfile view is the whole table).
    let superfiles = manifest
        .get_pruned_superfiles(leaves)
        .await
        .map_err(QueryError::ManifestLoad)?;

    if superfiles.is_empty() {
        return Ok(Vec::new());
    }

    // ---- Tier B: per-superfile prune. Start all-keep, AND each leaf's
    // mask. Scalar leaves are evaluated together (one `scalar_skip`
    // conjunction call) to match the pre-unification semantics.
    let mut mask = vec![true; superfiles.len()];

    let scalar_preds: Vec<ScalarPredicate> = leaves
        .iter()
        .filter_map(|l| match l {
            PruneLeaf::Scalar(p) => Some(p.clone()),
            _ => None,
        })
        .collect();
    if !scalar_preds.is_empty() {
        and_into(&mut mask, &scalar_skip(&superfiles, &scalar_preds));
    }

    for leaf in leaves {
        match leaf {
            PruneLeaf::TermPresence {
                column,
                terms,
                mode,
            } => {
                let refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();
                and_into(
                    &mut mask,
                    &fts_bloom_skip(&superfiles, column, &refs, *mode),
                );
            }
            PruneLeaf::Prefix { column, prefix } => {
                and_into(&mut mask, &fts_prefix_skip(&superfiles, column, prefix));
            }
            PruneLeaf::ScalarValueSet { column, values } => {
                and_into(
                    &mut mask,
                    &scalar_value_set_skip(&superfiles, column, values),
                );
            }
            PruneLeaf::NullCheck { column, want_null } => {
                and_into(&mut mask, &null_check_skip(&superfiles, column, *want_null));
            }
            // Scalar leaves handled above as one conjunction.
            PruneLeaf::Scalar(_) => {}
        }
    }

    Ok(superfiles
        .into_iter()
        .zip(mask)
        .filter_map(|(entry, keep)| keep.then_some(entry))
        .collect())
}

/// Element-wise `dst &= src`. Both slices are one bool per surviving
/// superfile, in the same order, so the index alignment holds.
fn and_into(dst: &mut [bool], src: &[bool]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d &= *s;
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        slice::from_ref,
    };

    use arrow_array::{Int64Array, LargeStringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{col, lit};
    use uuid::Uuid;

    use super::*;
    use crate::{
        superfile::builder::FtsConfig,
        supertable::{
            SupertableOptions,
            manifest::{
                FtsSummaryAgg, ManifestSnapshot, ScalarStatsAgg, SuperfileEntry, SuperfileUri,
                aggregates,
                bloom::BloomBuilder,
                list::{FORMAT_VERSION, Manifest, ManifestPartEntry, PartitionStrategy},
                part::{ContentHash, PartId},
            },
            query::{provider::exprs_to_value_set_leaves, skip::ScalarOp},
        },
        test_helpers::default_tokenizer,
    };

    fn seg_int(col: &str, min: i64, max: i64) -> Arc<SuperfileEntry> {
        let id = Uuid::new_v4();
        let mut cols: HashMap<String, ScalarStatsAgg> = HashMap::new();
        cols.insert(
            col.to_string(),
            ScalarStatsAgg::from_min_max(
                Arc::new(Int64Array::from(vec![min])),
                Arc::new(Int64Array::from(vec![max])),
            ),
        );
        Arc::new(SuperfileEntry {
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs: 1,
            id_min: 0,
            id_max: 0,
            scalar_stats: cols,
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    fn part_from(segs: &[Arc<SuperfileEntry>], seed: u8) -> ManifestPartEntry {
        let aggs = aggregates::compute(segs, None);
        ManifestPartEntry {
            part_id: PartId(Uuid::from_bytes([seed; 16])),
            uri: format!("manifests/part-{seed:02x}.avro.zst"),
            n_superfiles: segs.len() as u64,
            size_bytes_compressed: 1,
            size_bytes_uncompressed: 1,
            content_hash: ContentHash([seed; 32]),
            id_range: aggs.id_range,
            scalar_stats_agg: aggs.scalar_stats_agg,
            fts_summary_agg: aggs.fts_summary_agg,
            vector_summary_agg: aggs.vector_summary_agg,
        }
    }

    fn list_with(parts: Vec<ManifestPartEntry>) -> Manifest {
        Manifest {
            tombstone_seqs: Default::default(),
            format_version: FORMAT_VERSION.into(),
            manifest_id: 1,
            options_hash: ContentHash([0u8; 32]),
            schema: Vec::new(),
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 64,
            },
            parts,
        }
    }

    fn pred(col: &str, op: ScalarOp, v: i64) -> ScalarPredicate {
        ScalarPredicate {
            column: col.to_string(),
            op,
            value: ScalarValue::Int64(Some(v)),
        }
    }

    #[test]
    fn scalar_keep_parts_prunes_non_overlapping_part() {
        let p0 = part_from(&[seg_int("x", 0, 10)], 0);
        let p1 = part_from(&[seg_int("x", 100, 110)], 1);
        let list = list_with(vec![p0.clone(), p1.clone()]);

        // x = 5 → only p0's [0,10] aggregate can contain it.
        assert_eq!(
            scalar_keep_parts(&list, &pred("x", ScalarOp::Eq, 5)),
            vec![p0.part_id]
        );
        // x = 105 → only p1's [100,110].
        assert_eq!(
            scalar_keep_parts(&list, &pred("x", ScalarOp::Eq, 105)),
            vec![p1.part_id]
        );
        // x > 50 → p0.max=10 can't; p1 kept.
        assert_eq!(
            scalar_keep_parts(&list, &pred("x", ScalarOp::Gt, 50)),
            vec![p1.part_id]
        );
    }

    #[test]
    fn scalar_value_set_keep_parts_keeps_every_part_holding_a_listed_value() {
        let p0 = part_from(&[seg_int("x", 0, 10)], 0);
        let p1 = part_from(&[seg_int("x", 100, 110)], 1);
        let p2 = part_from(&[seg_int("x", 200, 210)], 2);
        let list = list_with(vec![p0.clone(), p1.clone(), p2.clone()]);
        let i = |n| ScalarValue::Int64(Some(n));

        // IN (5, 205) → p0 ([0,10]) and p2 ([200,210]); not p1.
        assert_eq!(
            scalar_value_set_keep_parts(&list, "x", &[i(5), i(205)]),
            vec![p0.part_id, p2.part_id]
        );
        // IN (50) → in no part's range.
        assert!(scalar_value_set_keep_parts(&list, "x", &[i(50)]).is_empty());
        // Unknown column → conservative keep-all.
        assert_eq!(
            scalar_value_set_keep_parts(&list, "missing", &[i(5)]),
            vec![p0.part_id, p1.part_id, p2.part_id]
        );
    }

    #[test]
    fn or_of_equalities_prunes_at_both_tiers() {
        // `x = 5 OR x = 205` lowers to the same ScalarValueSet the IN path
        // builds; measure the drop each tier makes for that leaf.
        let s = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
        let expr = col("x").eq(lit(5_i64)).or(col("x").eq(lit(205_i64)));
        let leaves = exprs_to_value_set_leaves(&[expr], &s, &HashSet::new(), None);
        let (column, values) = match leaves.as_slice() {
            [PruneLeaf::ScalarValueSet { column, values }] => (column.as_str(), values.clone()),
            _ => panic!("expected one ScalarValueSet leaf from the OR"),
        };

        // Tier A — part aggregates: 3 parts → 2; p1's [100,110] holds
        // neither value, so the part-level prune drops it.
        let p0 = part_from(&[seg_int("x", 0, 10)], 0);
        let p1 = part_from(&[seg_int("x", 100, 110)], 1);
        let p2 = part_from(&[seg_int("x", 200, 210)], 2);
        let list = list_with(vec![p0.clone(), p1, p2.clone()]);
        assert_eq!(
            scalar_value_set_keep_parts(&list, column, &values),
            vec![p0.part_id, p2.part_id],
            "part tier prunes 1 of 3"
        );

        // Tier B — per-superfile stats: within a surviving part, 2
        // superfiles → 1; [50,60] holds neither value, dropped here.
        let segs = vec![seg_int("x", 0, 10), seg_int("x", 50, 60)];
        assert_eq!(
            scalar_value_set_skip(&segs, column, &values),
            vec![true, false],
            "superfile tier prunes 1 of 2"
        );
    }

    #[test]
    fn scalar_keep_parts_keeps_on_missing_column_aggregate() {
        // No aggregate for the queried column → conservative keep.
        let p0 = part_from(&[seg_int("x", 0, 10)], 0);
        let list = list_with(vec![p0.clone()]);
        assert_eq!(
            scalar_keep_parts(&list, &pred("other", ScalarOp::Eq, 5)),
            vec![p0.part_id]
        );
    }

    // ---- FTS prunes where plain min/max (DataFusion) cannot --------

    fn opts_title_fts() -> Arc<SupertableOptions> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]));
        let tk = default_tokenizer();
        Arc::new(
            SupertableOptions::new(
                schema,
                vec![FtsConfig {
                    column: "title".into(),
                }],
                vec![],
                Some(tk),
            )
            .expect("opts"),
        )
    }

    /// A single superfile whose `title` column carries both the scalar
    /// min/max (what a plain Parquet reader exposes) and an FTS term
    /// bloom + range (what infino adds). Each title is treated as one
    /// token, so the bloom is exact membership over the title values.
    fn seg_title(titles: &[&str]) -> Arc<SuperfileEntry> {
        let mut sorted = titles.to_vec();
        sorted.sort();
        let (mn, mx) = (sorted[0], sorted[sorted.len() - 1]);

        let mut cols: HashMap<String, ScalarStatsAgg> = HashMap::new();
        cols.insert(
            "title".to_string(),
            ScalarStatsAgg::from_min_max(
                Arc::new(LargeStringArray::from(vec![mn])),
                Arc::new(LargeStringArray::from(vec![mx])),
            ),
        );

        let mut bb = BloomBuilder::new();
        for t in titles {
            bb.insert(t.as_bytes());
        }
        let mut fts = HashMap::new();
        fts.insert(
            "title".to_string(),
            FtsSummaryAgg::new_with_params(
                bb.finish(),
                titles.len() as u32,
                (mn.as_bytes().to_vec(), mx.as_bytes().to_vec()),
            ),
        );

        let id = Uuid::new_v4();
        Arc::new(SuperfileEntry {
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs: titles.len() as u64,
            id_min: 0,
            id_max: 0,
            scalar_stats: cols,
            fts_summary: fts,
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    #[tokio::test]
    async fn fts_bloom_prunes_superfile_that_scalar_minmax_cannot() {
        // Pathology: a superfile whose title values straddle the queried
        // value lexicographically, so the scalar [min,max] (the only
        // signal a plain Parquet+DataFusion scan has) can't rule it out
        // — yet the superfile provably does not contain the value.
        //
        // Superfile A: {apple, zebra} → range ["apple","zebra"] spans
        //   "mango", so scalar min/max keeps it; its term bloom lacks
        //   "mango".
        // Superfile B: {kiwi, mango} → actually contains "mango".
        let a = seg_title(&["apple", "zebra"]);
        let b = seg_title(&["kiwi", "mango"]);
        let manifest =
            ManifestSnapshot::empty(opts_title_fts()).with_appended(vec![a.clone(), b.clone()]);

        let scalar_leaf = PruneLeaf::Scalar(ScalarPredicate {
            column: "title".into(),
            op: ScalarOp::Eq,
            value: ScalarValue::Utf8(Some("mango".into())),
        });
        let term_leaf = PruneLeaf::TermPresence {
            column: "title".into(),
            terms: vec!["mango".into()],
            mode: BoolMode::And,
        };

        // DataFusion-equivalent: scalar min/max only. "mango" is within
        // both superfiles' lexicographic ranges, so neither is pruned.
        let scalar_only = select_superfiles(&manifest, from_ref(&scalar_leaf))
            .await
            .expect("select");
        assert_eq!(
            scalar_only.len(),
            2,
            "scalar min/max alone cannot prune either superfile"
        );

        // infino's term bloom proves "mango" absent from superfile A and
        // prunes it; only the superfile that can actually match remains.
        let with_fts = select_superfiles(&manifest, &[scalar_leaf, term_leaf])
            .await
            .expect("select");
        let kept: Vec<_> = with_fts.iter().map(|e| e.superfile_id).collect();
        assert_eq!(
            kept,
            vec![b.superfile_id],
            "FTS bloom prunes the superfile plain min/max could not, keeping only the real match"
        );
    }

    // ---- Unified-substrate coverage across leaf modes -------------
    //
    // These drive `select_superfiles` (the path SQL + FTS share) on a
    // flat (no-list) manifest, so they exercise the superfile tier + the
    // AND-combination of leaves. Part-tier behavior is covered directly
    // by the `scalar_keep_parts` tests above and the `list_prune` suite.

    /// Superfile carrying both a scalar min/max (what a plain Parquet
    /// reader exposes) and an FTS term bloom + range (what infino adds)
    /// for the `title` column. `bloom_tokens` are inserted as exact
    /// terms; the term range is their lex span.
    fn seg(scalar_min: &str, scalar_max: &str, bloom_tokens: &[&str]) -> Arc<SuperfileEntry> {
        let mut cols: HashMap<String, ScalarStatsAgg> = HashMap::new();
        cols.insert(
            "title".to_string(),
            ScalarStatsAgg::from_min_max(
                Arc::new(LargeStringArray::from(vec![scalar_min])),
                Arc::new(LargeStringArray::from(vec![scalar_max])),
            ),
        );
        let mut bb = BloomBuilder::new();
        for t in bloom_tokens {
            bb.insert(t.as_bytes());
        }
        let term_range = if bloom_tokens.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let mut sorted = bloom_tokens.to_vec();
            sorted.sort();
            (
                sorted[0].as_bytes().to_vec(),
                sorted[sorted.len() - 1].as_bytes().to_vec(),
            )
        };
        let mut fts = HashMap::new();
        fts.insert(
            "title".to_string(),
            FtsSummaryAgg::new_with_params(bb.finish(), bloom_tokens.len() as u32, term_range),
        );
        let id = Uuid::new_v4();
        Arc::new(SuperfileEntry {
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs: bloom_tokens.len().max(1) as u64,
            id_min: 0,
            id_max: 0,
            scalar_stats: cols,
            fts_summary: fts,
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    fn manifest(segs: Vec<Arc<SuperfileEntry>>) -> ManifestSnapshot {
        ManifestSnapshot::empty(opts_title_fts()).with_appended(segs)
    }

    async fn ids(m: &ManifestSnapshot, leaves: &[PruneLeaf]) -> Vec<Uuid> {
        select_superfiles(m, leaves)
            .await
            .expect("select")
            .iter()
            .map(|e| e.superfile_id)
            .collect()
    }

    fn scalar(col: &str, op: ScalarOp, v: &str) -> PruneLeaf {
        PruneLeaf::Scalar(ScalarPredicate {
            column: col.into(),
            op,
            value: ScalarValue::Utf8(Some(v.into())),
        })
    }
    fn eq(col: &str, v: &str) -> PruneLeaf {
        scalar(col, ScalarOp::Eq, v)
    }
    fn term(col: &str, terms: &[&str], mode: BoolMode) -> PruneLeaf {
        PruneLeaf::TermPresence {
            column: col.into(),
            terms: terms.iter().map(|s| s.to_string()).collect(),
            mode,
        }
    }
    fn pfx(col: &str, p: &str) -> PruneLeaf {
        PruneLeaf::Prefix {
            column: col.into(),
            prefix: p.as_bytes().to_vec(),
        }
    }

    // --- "Better than DataFusion": the FTS index prunes where the
    //     scalar min/max a plain Parquet scan relies on cannot.

    #[tokio::test]
    async fn multi_token_equality_prunes_when_any_token_absent() {
        // `title = 'rust async'`: the literal tokenizes to {rust,async}.
        // Superfile has 'rust' but not 'async', and its lex range spans
        // the literal — so min/max keeps it, the AND-bloom prunes it.
        let a = seg("a", "z", &["rust", "tokio"]);
        let m = manifest(vec![a.clone()]);
        assert_eq!(
            ids(&m, &[eq("title", "rust async")]).await,
            vec![a.superfile_id]
        );
        assert!(
            ids(
                &m,
                &[
                    eq("title", "rust async"),
                    term("title", &["rust", "async"], BoolMode::And),
                ],
            )
            .await
            .is_empty(),
            "AND-bloom prunes a superfile missing one of the literal's tokens"
        );
    }

    #[tokio::test]
    async fn bloom_keeps_only_the_token_holder_across_many_wide_superfiles() {
        // Four superfiles whose scalar ranges all span "mango", plus the
        // one superfile that holds it. Plain min/max prunes none; the
        // bloom isolates the single holder.
        let s1 = seg("a", "z", &["alpha", "omega"]);
        let s2 = seg("a", "z", &["beta", "gamma"]);
        let hit = seg("a", "z", &["mango", "kiwi"]);
        let s3 = seg("a", "z", &["delta", "sigma"]);
        let m = manifest(vec![s1, s2, hit.clone(), s3]);
        assert_eq!(
            ids(&m, &[eq("title", "mango")]).await.len(),
            4,
            "min/max cannot prune any wide-range superfile"
        );
        assert_eq!(
            ids(
                &m,
                &[
                    eq("title", "mango"),
                    term("title", &["mango"], BoolMode::And)
                ],
            )
            .await,
            vec![hit.superfile_id],
            "bloom keeps exactly the superfile that holds the token"
        );
    }

    #[tokio::test]
    async fn prefix_leaf_prunes_by_term_range() {
        // Prefix mode (the bm25_search_prefix path) routes through the
        // same substrate: a non-overlapping term range is pruned.
        let outside = seg("a", "z", &["apple", "banana"]);
        let inside = seg("a", "z", &["rustic", "rusty"]);
        let m = manifest(vec![outside, inside.clone()]);
        assert_eq!(
            ids(&m, &[pfx("title", "rust")]).await,
            vec![inside.superfile_id]
        );
    }

    // --- Substrate correctness across the remaining leaf modes.

    #[tokio::test]
    async fn term_presence_or_keeps_any_match() {
        let a = seg("a", "z", &["alpha", "beta"]);
        let b = seg("a", "z", &["gamma", "delta"]);
        let m = manifest(vec![a.clone(), b]);
        assert_eq!(
            ids(&m, &[term("title", &["alpha", "missing"], BoolMode::Or)]).await,
            vec![a.superfile_id],
            "OR keeps a superfile with any matching term, prunes one with none"
        );
    }

    #[tokio::test]
    async fn scalar_conjunction_prunes_outside_range() {
        // Two scalar leaves AND together — parity with DataFusion's
        // min/max, but verifies the substrate combines leaves correctly.
        let lo = seg("a", "c", &[]);
        let mid = seg("m", "o", &[]);
        let hi = seg("x", "z", &[]);
        let m = manifest(vec![lo, mid.clone(), hi]);
        assert_eq!(
            ids(
                &m,
                &[
                    scalar("title", ScalarOp::GtEq, "m"),
                    scalar("title", ScalarOp::LtEq, "p"),
                ],
            )
            .await,
            vec![mid.superfile_id]
        );
    }

    #[tokio::test]
    async fn empty_predicate_keeps_all_superfiles() {
        let m = manifest(vec![seg("a", "b", &[]), seg("c", "d", &[])]);
        assert_eq!(ids(&m, &[]).await.len(), 2, "no leaves → full scan");
    }

    // --- Conservativeness: never a false prune.

    #[tokio::test]
    async fn unknown_column_leaves_never_prune() {
        let a = seg("a", "z", &["x"]);
        let m = manifest(vec![a.clone()]);
        assert_eq!(
            ids(&m, &[eq("missing", "v")]).await,
            vec![a.superfile_id],
            "scalar on a column with no stats keeps the superfile"
        );
        assert_eq!(
            ids(&m, &[term("missing", &["v"], BoolMode::And)]).await,
            vec![a.superfile_id],
            "term presence on a column with no FTS summary keeps the superfile"
        );
    }

    #[tokio::test]
    async fn superfile_holding_all_tokens_is_never_dropped() {
        let a = seg("a", "z", &["rust", "async", "tokio"]);
        let m = manifest(vec![a.clone()]);
        assert_eq!(
            ids(
                &m,
                &[
                    eq("title", "rust async"),
                    term("title", &["rust", "async"], BoolMode::And),
                ],
            )
            .await,
            vec![a.superfile_id],
            "a superfile whose terms cover the literal is always kept"
        );
    }
}
