// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Per-`ManifestPart` aggregate skip summaries — writer-side.
//!
//! When the writer commits a manifest part containing N
//! superfiles, [`compute`] walks the superfiles and produces the
//! list-level aggregate values that drive list-level prune
//! pruning:
//!
//! - `id_range`: `(min(seg.id_min), max(seg.id_max))`.
//! - `scalar_stats_agg`: per scalar column, column-wise min /
//!   max across all superfiles. Encoded as length-1 Arrow IPC
//!   bytes (same encoding `ManifestPartEntry` uses).
//! - `fts_summary_agg`: per FTS column, the union of the superfiles'
//!   [`FtsSummaryAgg`]s — bloom bit-OR union + `(min(min_term),
//!   max(max_term))` term-range union — folded via
//!   [`FtsSummaryAgg::merge`].
//! - `vector_summary_agg`: per vector column, mean-of-
//!   centroids + max(distance + superfile_radius). Bounds every
//!   superfile's vector ball with one outer ball, so the
//!   list-level vector skip is correct by construction (no
//!   false negatives).
//!
//! The bloom union is exact for "any superfile contained this term"
//! (the block-and-mask scheme is positional). `n_terms_distinct` is a
//! deferred planner hint; a true HLL-based distinct-term union lands
//! when measured.

use std::{
    cmp::{max, min},
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use crate::supertable::manifest::{
    SuperfileEntry,
    list::{FtsSummaryAgg, ManifestPartEntry, ScalarStatsAgg, VectorSummaryAgg},
};

/// All four aggregate buckets for one [`ManifestPartEntry`].
/// Built by [`compute`] and inserted verbatim into the entry.
#[derive(Debug, Default)]
pub struct AggregateSet {
    pub id_range: (i128, i128),
    pub scalar_stats_agg: HashMap<String, ScalarStatsAgg>,
    pub fts_summary_agg: BTreeMap<String, FtsSummaryAgg>,
    pub vector_summary_agg: BTreeMap<String, VectorSummaryAgg>,
}

/// Build the aggregate set for one manifest part from its
/// superfile list.
///
/// Empty `superfiles` → all-default `AggregateSet` (id_range
/// `(0, 0)`, empty maps). The list-level pruner treats empty
/// maps as "no info on these columns" and defaults to
/// "always-keep" — correctness is preserved.
pub fn compute(
    superfiles: &[Arc<SuperfileEntry>],
    base_part: Option<&ManifestPartEntry>,
) -> AggregateSet {
    if superfiles.is_empty() {
        return base_part
            .map(|b| AggregateSet {
                id_range: (b.id_range.0, b.id_range.1),
                scalar_stats_agg: b.scalar_stats_agg.clone(),
                fts_summary_agg: b.fts_summary_agg.clone(),
                vector_summary_agg: b.vector_summary_agg.clone(),
            })
            .unwrap_or_default();
    }

    let mut id_min = superfiles.iter().map(|s| s.id_min).min().unwrap_or(0);
    let mut id_max = superfiles.iter().map(|s| s.id_max).max().unwrap_or(0);
    let mut scalar_stats_agg = scalar_stats_agg(superfiles);
    let mut fts_summary_agg = fts_summary_agg(superfiles);
    let mut vector_summary_agg = vector_summary_agg(superfiles);

    if let Some(base_part) = base_part {
        id_min = min(id_min, base_part.id_range.0);
        id_max = max(id_max, base_part.id_range.1);
        ScalarStatsAgg::merge(&mut scalar_stats_agg, &base_part.scalar_stats_agg);
        FtsSummaryAgg::merge(&mut fts_summary_agg, &base_part.fts_summary_agg);
        VectorSummaryAgg::merge(&mut vector_summary_agg, &base_part.vector_summary_agg);
    }

    AggregateSet {
        id_range: (id_min, id_max),
        scalar_stats_agg,
        fts_summary_agg,
        vector_summary_agg,
    }
}

// ---------------------------------------------------------
// Scalar stats: per column, fold the superfiles' aggregates.
// ---------------------------------------------------------

fn scalar_stats_agg(superfiles: &[Arc<SuperfileEntry>]) -> HashMap<String, ScalarStatsAgg> {
    // Fold each superfile's per-column aggregate table together. `merge_tables`
    // keeps the min/max extremes (column union) and combines the additive
    // stats (null count / sum / HLL) only when every contributor carries them
    // — the same part-level semantics, sharing one code path with the
    // per-column merge instead of a hand-rolled fold + a duplicate min/max
    // helper.
    let mut out: HashMap<String, ScalarStatsAgg> = HashMap::new();
    for seg in superfiles {
        ScalarStatsAgg::merge(&mut out, &seg.scalar_stats);
    }
    out
}

// ---------------------------------------------------------
// FTS summary aggregate: fold the superfiles' summaries.
// ---------------------------------------------------------

fn fts_summary_agg(superfiles: &[Arc<SuperfileEntry>]) -> BTreeMap<String, FtsSummaryAgg> {
    // Fold each superfile's per-column summary together via
    // `FtsSummaryAgg::merge` — bloom bit-OR union (exact for the "any
    // superfile contained this term" semantic, since the block-and-mask
    // scheme is positional) + term-range union — sharing one code path with
    // the per-column merge instead of a hand-rolled fold.
    let mut out: BTreeMap<String, FtsSummaryAgg> = BTreeMap::new();
    for seg in superfiles {
        for (col, summary) in &seg.fts_summary {
            out.entry(col.clone())
                .and_modify(|acc| acc.merge_with(summary))
                .or_insert_with(|| summary.clone());
        }
    }
    out
}

// ---------------------------------------------------------
// Vector summary aggregate: mean centroid + envelope radius.
// ---------------------------------------------------------

fn vector_summary_agg(superfiles: &[Arc<SuperfileEntry>]) -> BTreeMap<String, VectorSummaryAgg> {
    let mut per_column: HashMap<String, Vec<(&[f32], f32)>> = HashMap::new();
    for seg in superfiles {
        for (col, summary) in &seg.vector_summary {
            per_column
                .entry(col.clone())
                .or_default()
                .push((summary.centroid.as_slice(), summary.radius));
        }
    }
    let mut out = BTreeMap::new();
    for (col, entries) in per_column {
        let Some(first_dim) = entries.first().map(|(c, _)| c.len()) else {
            continue;
        };
        if entries.iter().any(|(c, _)| c.len() != first_dim) {
            // Skip columns with inconsistent dim (shouldn't
            // happen — schema enforces a single dim per column).
            continue;
        }
        let mut mean = vec![0.0_f64; first_dim];
        for (centroid, _) in &entries {
            for (i, v) in centroid.iter().enumerate() {
                mean[i] += *v as f64;
            }
        }
        let n = entries.len() as f64;
        let mean_f32: Vec<f32> = mean.into_iter().map(|x| (x / n) as f32).collect();

        // envelope_radius = max(distance(seg_centroid, mean) +
        // seg_radius) over all superfiles. Distance = L2 — works
        // for the L2sq/cosine/negdot metrics (cosine over
        // normalized centroids is equivalent to L2 distance).
        // Conservative: a metric-specific tightening is a
        // follow-up optimization.
        let mut envelope_radius: f32 = 0.0;
        for (centroid, radius) in &entries {
            let dist = l2_distance(centroid, &mean_f32);
            envelope_radius = envelope_radius.max(dist + radius);
        }

        let centroid_envelope = mean_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        out.insert(
            col,
            VectorSummaryAgg {
                centroid_envelope,
                // Per-column count of folded superfile centroids, so an
                // incremental `merge` and this batch build agree on the mean.
                n_vectors: entries.len() as u32,
                envelope_radius,
            },
        );
    }
    out
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "l2_distance: dim mismatch");
    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow_array::{ArrayRef, Int64Array, LargeStringArray, StringArray};

    use super::*;
    use crate::supertable::manifest::{
        FtsSummaryAgg, ScalarStatsAgg, SuperfileEntry, SuperfileUri,
        part::{ContentHash, PartId},
    };

    /// A `ManifestPartEntry` standing in for an existing part, carrying the
    /// given id range + per-column scalar aggregates (empty fts/vector aggs).
    fn base_entry(
        id_range: (i128, i128),
        scalar_stats_agg: HashMap<String, ScalarStatsAgg>,
    ) -> ManifestPartEntry {
        ManifestPartEntry {
            part_id: PartId(uuid::Uuid::from_bytes([0xb; 16])),
            uri: "manifests/part-base.avro.zst".into(),
            n_superfiles: 1,
            size_bytes_compressed: 1,
            size_bytes_uncompressed: 1,
            content_hash: ContentHash([0u8; 32]),
            id_range,
            scalar_stats_agg,
            fts_summary_agg: BTreeMap::new(),
            vector_summary_agg: BTreeMap::new(),
        }
    }

    /// A single-column i64 scalar-stats table, for building `base_entry`s.
    fn scalar_i64(col: &str, vals: Vec<i64>) -> HashMap<String, ScalarStatsAgg> {
        let arr: ArrayRef = Arc::new(Int64Array::from(vals));
        let mut m = HashMap::new();
        m.insert(
            col.to_string(),
            ScalarStatsAgg::from_column(&arr).expect("i64 is orderable"),
        );
        m
    }

    fn seg_with_string_minmax(col: &str, min: &str, max: &str, large: bool) -> Arc<SuperfileEntry> {
        let (mn, mx): (ArrayRef, ArrayRef) = if large {
            (
                Arc::new(LargeStringArray::from(vec![Some(min)])),
                Arc::new(LargeStringArray::from(vec![Some(max)])),
            )
        } else {
            (
                Arc::new(StringArray::from(vec![Some(min)])),
                Arc::new(StringArray::from(vec![Some(max)])),
            )
        };
        let mut cols = HashMap::new();
        cols.insert(col.to_string(), ScalarStatsAgg::from_min_max(mn, mx));
        Arc::new(SuperfileEntry {
            superfile_id: uuid::Uuid::new_v4(),
            uri: SuperfileUri::new_v4(),
            n_docs: 1,
            id_min: 0,
            id_max: 0,
            scalar_stats: cols,
            fts_summary: HashMap::<String, FtsSummaryAgg>::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    fn string_val(arr: &ArrayRef) -> String {
        if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
            return a.value(0).to_string();
        }
        if let Some(a) = arr.as_any().downcast_ref::<LargeStringArray>() {
            return a.value(0).to_string();
        }
        panic!(
            "expected Utf8 or LargeUtf8 column; got {:?}",
            arr.data_type()
        );
    }

    #[test]
    fn scalar_stats_agg_unions_utf8_min_max_across_superfiles() {
        let segs = vec![
            seg_with_string_minmax("title", "alpha", "delta", false),
            seg_with_string_minmax("title", "bravo", "echo", false),
        ];
        let aggs = scalar_stats_agg(&segs);
        let agg = aggs.get("title").expect("title agg present");
        assert_eq!(string_val(&agg.min), "alpha");
        assert_eq!(string_val(&agg.max), "echo");
    }

    #[test]
    fn scalar_stats_agg_unions_large_utf8_min_max_across_superfiles() {
        let segs = vec![
            seg_with_string_minmax("body", "mango", "papaya", true),
            seg_with_string_minmax("body", "apple", "orange", true),
        ];
        let aggs = scalar_stats_agg(&segs);
        let agg = aggs.get("body").expect("body agg present");
        assert_eq!(string_val(&agg.min), "apple");
        assert_eq!(string_val(&agg.max), "papaya");
    }

    /// Build a superfile carrying full per-column stats for one Int64 column
    /// (min/max + null count + exact sum + HLL), via `from_column`.
    fn seg_with_i64(col: &str, vals: Vec<i64>) -> Arc<SuperfileEntry> {
        let arr: ArrayRef = Arc::new(Int64Array::from(vals));
        let mut cols = HashMap::new();
        cols.insert(
            col.to_string(),
            ScalarStatsAgg::from_column(&arr).expect("i64 is orderable"),
        );
        Arc::new(SuperfileEntry {
            superfile_id: uuid::Uuid::new_v4(),
            uri: SuperfileUri::new_v4(),
            n_docs: 1,
            id_min: 0,
            id_max: 0,
            scalar_stats: cols,
            fts_summary: HashMap::<String, FtsSummaryAgg>::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    fn i64_val(arr: &ArrayRef) -> i64 {
        arr.as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 array")
            .value(0)
    }

    #[test]
    fn scalar_stats_agg_folds_additive_stats_across_superfiles() {
        // Two superfiles, same column: min/max take the extremes; null_count
        // and sum fold additively; HLL merges.
        let segs = vec![
            seg_with_i64("n", vec![10, 50]), // sum 60
            seg_with_i64("n", vec![5, 30]),  // sum 35
        ];
        let aggs = scalar_stats_agg(&segs);
        let agg = aggs.get("n").expect("n agg present");
        assert_eq!(i64_val(&agg.min), 5);
        assert_eq!(i64_val(&agg.max), 50);
        assert_eq!(agg.null_count, Some(0)); // no nulls in either
        assert_eq!(i64_val(agg.sum.as_ref().expect("summed")), 95); // 60 + 35
        assert!(agg.hll.is_some(), "HLL sketches fold across superfiles");
    }

    #[test]
    fn scalar_stats_agg_drops_additive_when_a_superfile_lacks_the_stat() {
        // One superfile carries full stats; the other carries only min/max
        // (e.g. an older segment). The additive stats become unknowable at
        // the part level, but min/max still union.
        let bounds_only = {
            let mn: ArrayRef = Arc::new(Int64Array::from(vec![3]));
            let mx: ArrayRef = Arc::new(Int64Array::from(vec![4]));
            let mut cols = HashMap::new();
            cols.insert("n".to_string(), ScalarStatsAgg::from_min_max(mn, mx));
            Arc::new(SuperfileEntry {
                superfile_id: uuid::Uuid::new_v4(),
                uri: SuperfileUri::new_v4(),
                n_docs: 1,
                id_min: 0,
                id_max: 0,
                scalar_stats: cols,
                fts_summary: HashMap::<String, FtsSummaryAgg>::new(),
                vector_summary: HashMap::new(),
                partition_key: Vec::new(),
                partition_hint: None,
                subsection_offsets: None,
            })
        };
        let segs = vec![seg_with_i64("n", vec![1, 100]), bounds_only];
        let aggs = scalar_stats_agg(&segs);
        let agg = aggs.get("n").expect("n agg present");
        // min/max union across both.
        assert_eq!(i64_val(&agg.min), 1);
        assert_eq!(i64_val(&agg.max), 100);
        // Additive stats drop to None because one contributor lacks them.
        assert!(agg.sum.is_none(), "sum unknowable when a segment lacks it");
        assert!(agg.null_count.is_none());
        assert!(agg.hll.is_none());
    }

    #[test]
    fn compute_empty_superfiles_with_base_part_returns_base_aggregates() {
        // No new superfiles, but an existing part is supplied → `compute`
        // carries the base part's aggregates forward verbatim (the rebalance
        // "nothing new for this partition" case).
        let base = base_entry((100, 200), scalar_i64("n", vec![5, 9]));
        let aggs = compute(&[], Some(&base));
        assert_eq!(aggs.id_range, (100, 200));
        let n = aggs.scalar_stats_agg.get("n").expect("n carried forward");
        assert_eq!(i64_val(&n.min), 5);
        assert_eq!(i64_val(&n.max), 9);
        // fts/vector aggregates are carried over as-is (empty here).
        assert!(aggs.fts_summary_agg.is_empty());
        assert!(aggs.vector_summary_agg.is_empty());
    }

    #[test]
    fn compute_empty_superfiles_without_base_part_is_default() {
        // The `None` arm of the empty-superfiles branch → all-default.
        let aggs = compute(&[], None);
        assert_eq!(aggs.id_range, (0, 0));
        assert!(aggs.scalar_stats_agg.is_empty());
    }

    #[test]
    fn compute_nonempty_superfiles_folds_base_part() {
        // New superfiles for column "n"; the base part spans a wider id range
        // on both ends and also carries a column "m" the new superfiles lack.
        // The fold must: extend id_range to cover both, merge "n", and carry
        // "m" forward.
        let new_segs = vec![seg_with_i64("n", vec![20, 80])]; // seg id_range is (0, 0)
        let mut base_scalar = scalar_i64("n", vec![10, 60]);
        base_scalar.extend(scalar_i64("m", vec![1, 2]));
        // Base id_range (-5, 200) straddles the segs' (0, 0) on both ends.
        let base = base_entry((-5, 200), base_scalar);

        let aggs = compute(&new_segs, Some(&base));

        // id_range = (min(0, -5), max(0, 200)).
        assert_eq!(aggs.id_range, (-5, 200));
        // "n" merged across new superfiles + base: min(20, 10), max(80, 60).
        let n = aggs.scalar_stats_agg.get("n").expect("n");
        assert_eq!(i64_val(&n.min), 10);
        assert_eq!(i64_val(&n.max), 80);
        // "m" exists only in the base part → carried forward by the merge.
        let m = aggs.scalar_stats_agg.get("m").expect("m from base");
        assert_eq!(i64_val(&m.min), 1);
        assert_eq!(i64_val(&m.max), 2);
    }
}
