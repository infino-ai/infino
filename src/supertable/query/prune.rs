//! Unified segment-selection (pruning) for the boolean-predicate
//! query paths.
//!
//! FTS (exact + prefix) and SQL scalar filtering ask the *same*
//! question before they touch any segment bytes: "which segments could
//! possibly contain a row this predicate matches?" Each answers it by
//! conservatively evaluating a per-column test against the manifest's
//! summaries — term bloom, term range, scalar min/max — first over the
//! list's part-level aggregates, then over the surviving segments'
//! per-segment summaries.
//!
//! This module owns that two-tier walk so the three call sites
//! (`bm25_search`, `bm25_search_prefix`, the SQL `SupertableProvider`)
//! share one selection path instead of each re-deriving it. The
//! per-leaf math is **not** reimplemented here: each [`PruneLeaf`]
//! delegates to the existing helpers in [`super::skip`] (segment tier)
//! and [`crate::supertable::manifest::list_prune`] (part tier), so edge
//! behavior — empty-term handling, missing-column "always keep",
//! conservatism — is inherited verbatim.
//!
//! Vector kNN is intentionally *not* a leaf here: its prune signal is a
//! centroid/cutoff test whose cutoff only exists during fan-out, a
//! different shape from these static boolean tests. It keeps its own
//! path.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::ipc::reader::StreamReader;
use datafusion::scalar::ScalarValue;

use crate::superfile::fts::reader::BoolMode;
use crate::supertable::error::QueryError;
use crate::supertable::manifest::list::ManifestList;
use crate::supertable::manifest::list_prune::{
    prune_parts_for_fts_prefix, prune_parts_for_fts_terms,
};
use crate::supertable::manifest::part::PartId;
use crate::supertable::manifest::{Manifest, SuperfileEntry};

use super::hierarchical_iter;
use super::skip::{
    ScalarPredicate, fts_bloom_skip, fts_prefix_skip, scalar_skip, scalar_value_may_match,
};

/// One conjunct of a prune predicate: a per-column test backed by a
/// manifest summary. The full predicate is the **conjunction** of its
/// leaves — a segment survives only if every leaf keeps it. (A
/// `TermPresence` leaf carries its own intra-leaf OR/AND over terms via
/// [`BoolMode`]; cross-column OR isn't expressible yet and isn't needed
/// — an unprunable predicate simply contributes no leaf and the segment
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
}

impl PruneLeaf {
    /// Part-tier keep set for this leaf, or `None` when the leaf has no
    /// part-level pruner (it imposes no part constraint → keep all).
    fn keep_parts(&self, list: &ManifestList) -> Option<Vec<PartId>> {
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
        }
    }
}

/// Part-tier scalar prune: keep each part whose aggregate min/max for
/// the predicate's column could satisfy it. A missing aggregate or
/// undecodable bounds → keep (conservative — never a false prune).
///
/// The aggregate min/max live as length-1 Arrow IPC batches
/// (`ScalarStatsAgg.{min,max}`); we decode them and reuse the same
/// comparison core the segment tier uses ([`scalar_value_may_match`]).
fn scalar_keep_parts(list: &ManifestList, pred: &ScalarPredicate) -> Vec<PartId> {
    list.parts
        .iter()
        .filter_map(|entry| {
            let keep = match entry.scalar_stats_agg.get(&pred.column) {
                None => true,
                Some(agg) => {
                    match (
                        decode_length1_scalar(&agg.min),
                        decode_length1_scalar(&agg.max),
                    ) {
                        (Some(min), Some(max)) => {
                            scalar_value_may_match(&min, &max, pred.op, &pred.value)
                        }
                        _ => true,
                    }
                }
            };
            keep.then_some(entry.part_id)
        })
        .collect()
}

/// Decode a length-1 Arrow IPC stream (the `ScalarStatsAgg.{min,max}`
/// wire shape — one batch, one column, one row) into its single
/// `ScalarValue`. `None` on any decode failure, which callers treat as
/// "keep".
fn decode_length1_scalar(bytes: &[u8]) -> Option<ScalarValue> {
    let reader = StreamReader::try_new(bytes, None).ok()?;
    for batch in reader {
        let batch = batch.ok()?;
        if batch.num_columns() >= 1 && batch.num_rows() >= 1 {
            return ScalarValue::try_from_array(batch.column(0).as_ref(), 0).ok();
        }
    }
    None
}

/// Select the segments a predicate could match, newest-first in
/// manifest order, applying the two prune tiers (part aggregates →
/// per-segment summaries). Returns the surviving segment entries; the
/// caller drives execution over them (search fan-out or DataFusion
/// scan).
///
/// An empty `leaves` slice keeps every segment (the no-`WHERE` scan).
pub(crate) async fn select_segments(
    manifest: &Manifest,
    leaves: &[PruneLeaf],
) -> Result<Vec<Arc<SuperfileEntry>>, QueryError> {
    // ---- Tier A: part-level prune (only when a hierarchical list
    // exists; otherwise the flat segment view is the whole table).
    let superfiles: Vec<Arc<SuperfileEntry>> = match manifest.list.as_ref() {
        Some(list) => {
            // Intersect each constraining leaf's kept-part set. A leaf
            // with no part pruner (`None`) imposes no constraint.
            let mut kept: Option<HashSet<PartId>> = None;
            for leaf in leaves {
                if let Some(part_ids) = leaf.keep_parts(list) {
                    let set: HashSet<PartId> = part_ids.into_iter().collect();
                    kept = Some(match kept {
                        None => set,
                        Some(existing) => existing.intersection(&set).copied().collect(),
                    });
                }
            }
            // Preserve manifest (time) order of the surviving parts.
            let ordered: Vec<PartId> = match kept {
                Some(set) => list
                    .parts
                    .iter()
                    .map(|p| p.part_id)
                    .filter(|id| set.contains(id))
                    .collect(),
                None => list.parts.iter().map(|p| p.part_id).collect(),
            };
            hierarchical_iter::load_and_flatten(manifest, &ordered).await?
        }
        None => hierarchical_iter::fallback_to_flat_segments(manifest),
    };

    if superfiles.is_empty() {
        return Ok(Vec::new());
    }

    // ---- Tier B: per-segment prune. Start all-keep, AND each leaf's
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
/// segment, in the same order, so the index alignment holds.
fn and_into(dst: &mut [bool], src: &[bool]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d &= *s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::builder::FtsConfig;
    use crate::supertable::SupertableOptions;
    use crate::supertable::manifest::aggregates;
    use crate::supertable::manifest::list::{
        FORMAT_VERSION, ManifestList, ManifestListEntry, PartitionStrategy,
    };
    use crate::supertable::manifest::part::{ContentHash, PartId};
    use crate::supertable::manifest::{
        FtsSummary, Manifest, ScalarStatsTable, SuperfileEntry, SuperfileUri, bloom::BloomBuilder,
    };
    use crate::supertable::query::skip::ScalarOp;
    use arrow_array::{ArrayRef, Int64Array, LargeStringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::collections::HashMap;
    use uuid::Uuid;

    fn seg_int(col: &str, min: i64, max: i64) -> Arc<SuperfileEntry> {
        let id = Uuid::new_v4();
        let mut cols: HashMap<String, (ArrayRef, ArrayRef)> = HashMap::new();
        cols.insert(
            col.to_string(),
            (
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
            scalar_stats: ScalarStatsTable { cols },
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    fn part_from(segs: &[Arc<SuperfileEntry>], seed: u8) -> ManifestListEntry {
        let aggs = aggregates::compute(segs);
        ManifestListEntry {
            part_id: PartId(Uuid::from_bytes([seed; 16])),
            uri: format!("manifests/part-{seed:02x}.avro.zst"),
            n_superfiles: segs.len() as u64,
            size_bytes_compressed: 1,
            size_bytes_uncompressed: 1,
            content_hash: ContentHash([seed; 32]),
            partition_key: Vec::new(),
            id_range: aggs.id_range,
            scalar_stats_agg: aggs.scalar_stats_agg,
            fts_summary_agg: aggs.fts_summary_agg,
            vector_summary_agg: aggs.vector_summary_agg,
        }
    }

    fn list_with(parts: Vec<ManifestListEntry>) -> ManifestList {
        ManifestList {
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
        let tk = crate::test_helpers::default_tokenizer();
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

    /// A single segment whose `title` column carries both the scalar
    /// min/max (what a plain Parquet reader exposes) and an FTS term
    /// bloom + range (what infino adds). Each title is treated as one
    /// token, so the bloom is exact membership over the title values.
    fn seg_title(titles: &[&str]) -> Arc<SuperfileEntry> {
        let mut sorted = titles.to_vec();
        sorted.sort();
        let (mn, mx) = (sorted[0], sorted[sorted.len() - 1]);

        let mut cols: HashMap<String, (ArrayRef, ArrayRef)> = HashMap::new();
        cols.insert(
            "title".to_string(),
            (
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
            FtsSummary {
                term_bloom: bb.finish(),
                n_terms_distinct: titles.len() as u32,
                term_range: (mn.as_bytes().to_vec(), mx.as_bytes().to_vec()),
            },
        );

        let id = Uuid::new_v4();
        Arc::new(SuperfileEntry {
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs: titles.len() as u64,
            id_min: 0,
            id_max: 0,
            scalar_stats: ScalarStatsTable { cols },
            fts_summary: fts,
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    #[tokio::test]
    async fn fts_bloom_prunes_segment_that_scalar_minmax_cannot() {
        // Pathology: a segment whose title values straddle the queried
        // value lexicographically, so the scalar [min,max] (the only
        // signal a plain Parquet+DataFusion scan has) can't rule it out
        // — yet the segment provably does not contain the value.
        //
        // Segment A: {apple, zebra} → range ["apple","zebra"] spans
        //   "mango", so scalar min/max keeps it; its term bloom lacks
        //   "mango".
        // Segment B: {kiwi, mango} → actually contains "mango".
        let a = seg_title(&["apple", "zebra"]);
        let b = seg_title(&["kiwi", "mango"]);
        let manifest = Manifest::empty(opts_title_fts()).with_appended(vec![a.clone(), b.clone()]);

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
        // both segments' lexicographic ranges, so neither is pruned.
        let scalar_only = select_segments(&manifest, std::slice::from_ref(&scalar_leaf))
            .await
            .expect("select");
        assert_eq!(
            scalar_only.len(),
            2,
            "scalar min/max alone cannot prune either segment"
        );

        // infino's term bloom proves "mango" absent from segment A and
        // prunes it; only the segment that can actually match remains.
        let with_fts = select_segments(&manifest, &[scalar_leaf, term_leaf])
            .await
            .expect("select");
        let kept: Vec<_> = with_fts.iter().map(|e| e.superfile_id).collect();
        assert_eq!(
            kept,
            vec![b.superfile_id],
            "FTS bloom prunes the segment plain min/max could not, keeping only the real match"
        );
    }

    // ---- Unified-substrate coverage across leaf modes -------------
    //
    // These drive `select_segments` (the path SQL + FTS share) on a
    // flat (no-list) manifest, so they exercise the segment tier + the
    // AND-combination of leaves. Part-tier behavior is covered directly
    // by the `scalar_keep_parts` tests above and the `list_prune` suite.

    /// Segment carrying both a scalar min/max (what a plain Parquet
    /// reader exposes) and an FTS term bloom + range (what infino adds)
    /// for the `title` column. `bloom_tokens` are inserted as exact
    /// terms; the term range is their lex span.
    fn seg(scalar_min: &str, scalar_max: &str, bloom_tokens: &[&str]) -> Arc<SuperfileEntry> {
        let mut cols: HashMap<String, (ArrayRef, ArrayRef)> = HashMap::new();
        cols.insert(
            "title".to_string(),
            (
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
            FtsSummary {
                term_bloom: bb.finish(),
                n_terms_distinct: bloom_tokens.len() as u32,
                term_range,
            },
        );
        let id = Uuid::new_v4();
        Arc::new(SuperfileEntry {
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs: bloom_tokens.len().max(1) as u64,
            id_min: 0,
            id_max: 0,
            scalar_stats: ScalarStatsTable { cols },
            fts_summary: fts,
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    fn manifest(segs: Vec<Arc<SuperfileEntry>>) -> Manifest {
        Manifest::empty(opts_title_fts()).with_appended(segs)
    }

    async fn ids(m: &Manifest, leaves: &[PruneLeaf]) -> Vec<Uuid> {
        select_segments(m, leaves)
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
        // Segment has 'rust' but not 'async', and its lex range spans
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
            "AND-bloom prunes a segment missing one of the literal's tokens"
        );
    }

    #[tokio::test]
    async fn bloom_keeps_only_the_token_holder_across_many_wide_segments() {
        // Four segments whose scalar ranges all span "mango", plus the
        // one segment that holds it. Plain min/max prunes none; the
        // bloom isolates the single holder.
        let s1 = seg("a", "z", &["alpha", "omega"]);
        let s2 = seg("a", "z", &["beta", "gamma"]);
        let hit = seg("a", "z", &["mango", "kiwi"]);
        let s3 = seg("a", "z", &["delta", "sigma"]);
        let m = manifest(vec![s1, s2, hit.clone(), s3]);
        assert_eq!(
            ids(&m, &[eq("title", "mango")]).await.len(),
            4,
            "min/max cannot prune any wide-range segment"
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
            "bloom keeps exactly the segment that holds the token"
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
            "OR keeps a segment with any matching term, prunes one with none"
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
    async fn empty_predicate_keeps_all_segments() {
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
            "scalar on a column with no stats keeps the segment"
        );
        assert_eq!(
            ids(&m, &[term("missing", &["v"], BoolMode::And)]).await,
            vec![a.superfile_id],
            "term presence on a column with no FTS summary keeps the segment"
        );
    }

    #[tokio::test]
    async fn segment_holding_all_tokens_is_never_dropped() {
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
            "a segment whose terms cover the literal is always kept"
        );
    }
}
