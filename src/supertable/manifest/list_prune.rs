//! List-level skip pruning — reader-side.
//!
//! Walks a [`ManifestList`]'s `parts` and applies the
//! aggregate skip tests in [`ManifestListEntry`] to identify
//! candidate parts for a given query shape. Survivors are
//! the parts the query layer should load (via
//! [`Manifest::part`]) for per-segment pruning.
//!
//! These functions are standalone — they don't depend on
//! the in-memory `Manifest` or its `ManifestPartLoader`.
//! That keeps them testable in isolation and lets the
//! query-layer integration choose its own loading shape.
//!
//! ## Correctness invariants
//!
//! - **Monotonic**: every part the flat (segment-level) prune
//!   would visit is also a survivor here. Aggregate
//!   summaries are constructed to over-approximate the union
//!   of segment-level skip data, so a query that matches any
//!   segment in a part necessarily matches the part's
//!   aggregate.
//! - **"Always-keep" defaults**: parts with empty `*_agg`
//!   entries for the queried column trivially survive (e.g.
//!   pre-aggregate manifests, or entries where a particular
//!   column has no info).
//!
//! [`Manifest`]: super::Manifest
//! [`Manifest::part`]: super::Manifest::part
//! [`ManifestList`]: super::list::ManifestList
//! [`ManifestListEntry`]: super::list::ManifestListEntry

use crate::superfile::fts::reader::BoolMode;
use crate::supertable::manifest::list::{ManifestList, ManifestListEntry};
use crate::supertable::manifest::part::PartId;

/// Filter the list's parts to those whose
/// `term_range_union[column]` overlaps the prefix
/// `[prefix, prefix_upper_bound)`.
///
/// Parts without an `fts_summary_agg` entry for this column
/// (no info) survive — same "always-keep" treatment the
/// list-level pruner gives to missing aggregates.
pub fn prune_parts_for_fts_prefix(list: &ManifestList, column: &str, prefix: &[u8]) -> Vec<PartId> {
    let upper = prefix_upper_bound(prefix);
    list.parts
        .iter()
        .filter_map(|entry| {
            if part_overlaps_prefix(entry, column, prefix, upper.as_deref()) {
                Some(entry.part_id)
            } else {
                None
            }
        })
        .collect()
}

fn part_overlaps_prefix(
    entry: &ManifestListEntry,
    column: &str,
    prefix: &[u8],
    upper: Option<&[u8]>,
) -> bool {
    let Some(agg) = entry.fts_summary_agg.get(column) else {
        // No info → always-keep.
        return true;
    };
    let Some((min_term, max_term)) = agg.term_range_union.as_ref() else {
        // Every segment had an empty FST for this column;
        // nothing to match. Skip.
        return false;
    };
    // Overlap check: [prefix, upper) intersects [min_term, max_term]
    // iff prefix <= max_term && (upper is None || min_term < upper).
    if prefix > max_term.as_slice() {
        return false;
    }
    !matches!(upper, Some(u) if min_term.as_slice() >= u)
}

/// Compute the lex-upper-bound for a prefix: the smallest
/// byte string that doesn't start with `prefix`. `None`
/// signals "no upper bound" (e.g., a prefix of all 0xFF
/// bytes — every byte string starts with that or has no
/// successor in lex order).
///
/// `[prefix, prefix_upper_bound())` is the set of all byte
/// strings starting with `prefix`.
fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut out = prefix.to_vec();
    while let Some(&b) = out.last() {
        if b == 0xff {
            out.pop();
        } else {
            *out.last_mut().expect("non-empty") = b + 1;
            return Some(out);
        }
    }
    None
}

/// Filter the list's parts to those whose
/// `term_bloom_union[column]` allows at least one query
/// term (mode = Or) or all of them (mode = And) — i.e. the
/// list-level analogue of segment-level `fts_bloom_skip`.
///
/// Parts without a bloom union entry for this column (e.g.,
/// pre-aggregate manifests or aggregates that fell back
/// to "no info" due to a shape mismatch) survive — same
/// always-keep treatment as the rest of `list_prune`. An
/// empty `query_terms` slice yields an empty mask; callers
/// should special-case that upstream.
///
/// Used by `bm25_search` (exact-term) to prune entire parts
/// before lazy-loading. Complements
/// `prune_parts_for_fts_prefix` (which uses term-range
/// overlap on prefix queries) and segment-level
/// `fts_bloom_skip` (applied after a part is loaded).
pub fn prune_parts_for_fts_terms(
    list: &ManifestList,
    column: &str,
    query_terms: &[&str],
    mode: BoolMode,
) -> Vec<PartId> {
    if query_terms.is_empty() {
        return Vec::new();
    }
    list.parts
        .iter()
        .filter_map(|entry| {
            if part_matches_terms(entry, column, query_terms, mode) {
                Some(entry.part_id)
            } else {
                None
            }
        })
        .collect()
}

fn part_matches_terms(
    entry: &ManifestListEntry,
    column: &str,
    query_terms: &[&str],
    mode: BoolMode,
) -> bool {
    let Some(agg) = entry.fts_summary_agg.get(column) else {
        return true; // no info → always-keep
    };
    if agg.term_bloom_union.is_empty() || agg.term_bloom_n_blocks == 0 {
        return true; // empty union → always-keep
    }
    let Some(bloom) = crate::supertable::manifest::bloom::Bloom::from_bytes(&agg.term_bloom_union)
    else {
        // Corrupt / unexpected shape → fall back to
        // always-keep (correctness over selectivity).
        return true;
    };
    match mode {
        BoolMode::Or => query_terms.iter().any(|t| bloom.contains(t.as_bytes())),
        BoolMode::And => query_terms.iter().all(|t| bloom.contains(t.as_bytes())),
    }
}

/// Filter the list's parts to those whose `id_range`
/// overlaps the inclusive range `[query_min, query_max]`.
///
/// The id column is `Decimal128(38, 0)` (the supertable-
/// injected `_id` column), so this is the type-specialized
/// hot path for `WHERE _id BETWEEN ? AND ?`. For other
/// scalar columns, use [`prune_parts_for_scalar_min_max_bytes`].
pub fn prune_parts_for_id_range(
    list: &ManifestList,
    query_min: i128,
    query_max: i128,
) -> Vec<PartId> {
    list.parts
        .iter()
        .filter_map(|entry| {
            let (lo, hi) = entry.id_range;
            // `(query_min, query_max)` overlaps `(lo, hi)` iff
            // query_min <= hi && query_max >= lo.
            if query_min <= hi && query_max >= lo {
                Some(entry.part_id)
            } else {
                None
            }
        })
        .collect()
}

/// Filter parts whose `vector_summary_agg[column]` envelope
/// can possibly contain a vector within `query_cutoff` of
/// `query`. Conservative: a part survives iff
/// `distance(query, envelope_center) ≤ envelope_radius +
/// query_cutoff`. Parts with no vector summary for this
/// column survive (no info).
///
/// Distance is L2; for cosine workloads, the query vector + centroids
/// should be normalized at the caller layer (matching the convention the
/// segment-level vector skip already uses).
pub fn prune_parts_for_vector(
    list: &ManifestList,
    column: &str,
    query: &[f32],
    query_cutoff: f32,
) -> Vec<PartId> {
    list.parts
        .iter()
        .filter_map(|entry| {
            let Some(agg) = entry.vector_summary_agg.get(column) else {
                return Some(entry.part_id);
            };
            if agg.centroid_envelope.is_empty() {
                // Empty envelope — no info; keep.
                return Some(entry.part_id);
            }
            let envelope = decode_centroid_envelope(&agg.centroid_envelope);
            if envelope.len() != query.len() {
                // Dim mismatch — keep (the per-segment prune
                // will reject correctly).
                return Some(entry.part_id);
            }
            let dist = l2_distance(query, &envelope);
            if dist <= agg.envelope_radius + query_cutoff {
                Some(entry.part_id)
            } else {
                None
            }
        })
        .collect()
}

fn decode_centroid_envelope(bytes: &[u8]) -> Vec<f32> {
    let dim = bytes.len() / 4;
    let mut out = Vec::with_capacity(dim);
    for i in 0..dim {
        let s = i * 4;
        out.push(f32::from_le_bytes([
            bytes[s],
            bytes[s + 1],
            bytes[s + 2],
            bytes[s + 3],
        ]));
    }
    out
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..a.len().min(b.len()) {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supertable::manifest::aggregates;
    use crate::supertable::manifest::bloom::BloomBuilder;
    use crate::supertable::manifest::list::{
        FORMAT_VERSION, ManifestList, ManifestListEntry, PartitionStrategy,
    };
    use crate::supertable::manifest::part::{ContentHash, PartId};
    use crate::supertable::manifest::{FtsSummary, ScalarStatsTable, VectorSummary};
    use crate::supertable::{SuperfileEntry, SuperfileUri};
    use arrow_array::Int64Array;
    use std::collections::HashMap;
    use std::sync::Arc;
    use uuid::Uuid;

    #[test]
    fn prefix_upper_bound_basic() {
        assert_eq!(prefix_upper_bound(b"abc"), Some(b"abd".to_vec()));
        assert_eq!(prefix_upper_bound(b"ab\xff"), Some(b"ac".to_vec()));
        assert_eq!(prefix_upper_bound(b"\xff\xff"), None);
        assert_eq!(prefix_upper_bound(b""), None);
    }

    // ---- Helpers for the aggregates::compute and
    //      prune_parts_for_* tests below.

    fn seg(
        id_min: i128,
        id_max: i128,
        title_terms: &[&str],
        vec_centroid: Option<Vec<f32>>,
        vec_radius: f32,
    ) -> Arc<SuperfileEntry> {
        let id = Uuid::new_v4();
        let mut fts = HashMap::new();
        if !title_terms.is_empty() {
            let mut bloom = BloomBuilder::with_n_blocks(16);
            for t in title_terms {
                bloom.insert(t.as_bytes());
            }
            let term_range = {
                let mut sorted = title_terms
                    .iter()
                    .map(|t| t.as_bytes().to_vec())
                    .collect::<Vec<_>>();
                sorted.sort();
                (
                    sorted.first().cloned().unwrap_or_default(),
                    sorted.last().cloned().unwrap_or_default(),
                )
            };
            fts.insert(
                "title".into(),
                FtsSummary {
                    term_bloom: bloom.finish(),
                    n_terms_distinct: title_terms.len() as u32,
                    term_range,
                },
            );
        }
        let mut vec_summary = HashMap::new();
        if let Some(c) = vec_centroid {
            vec_summary.insert(
                "emb".into(),
                VectorSummary {
                    centroid: c,
                    radius: vec_radius,
                    clusters: crate::supertable::manifest::ClusterCentroids::empty(),
                },
            );
        }
        Arc::new(SuperfileEntry {
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs: ((id_max - id_min) + 1) as u64,
            id_min,
            id_max,
            scalar_stats: ScalarStatsTable::new(),
            fts_summary: fts,
            vector_summary: vec_summary,
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    fn entry_from_segments(superfiles: &[Arc<SuperfileEntry>], seed: u8) -> ManifestListEntry {
        let aggs = aggregates::compute(superfiles);
        ManifestListEntry {
            part_id: PartId(Uuid::from_bytes([seed; 16])),
            uri: format!("manifests/part-{seed:02x}.avro.zst"),
            n_superfiles: superfiles.len() as u64,
            size_bytes_compressed: 1024,
            size_bytes_uncompressed: 4096,
            content_hash: ContentHash([seed; 32]),
            partition_key: Vec::new(),
            id_range: aggs.id_range,
            scalar_stats_agg: aggs.scalar_stats_agg,
            fts_summary_agg: aggs.fts_summary_agg,
            vector_summary_agg: aggs.vector_summary_agg,
        }
    }

    fn list_with(entries: Vec<ManifestListEntry>) -> ManifestList {
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
            parts: entries,
        }
    }

    // ---- aggregates::compute — value correctness.

    #[test]
    fn aggregates_compute_empty_returns_default() {
        let aggs = aggregates::compute(&[]);
        assert_eq!(aggs.id_range, (0, 0));
        assert!(aggs.scalar_stats_agg.is_empty());
        assert!(aggs.fts_summary_agg.is_empty());
        assert!(aggs.vector_summary_agg.is_empty());
    }

    #[test]
    fn aggregates_compute_id_range_is_min_max_across_segments() {
        let s_a = seg(100, 199, &["alpha"], None, 0.0);
        let s_b = seg(0, 99, &["beta"], None, 0.0);
        let s_c = seg(500, 599, &["gamma"], None, 0.0);
        let aggs = aggregates::compute(&[s_a, s_b, s_c]);
        assert_eq!(aggs.id_range, (0, 599));
    }

    #[test]
    fn aggregates_compute_fts_term_range_union() {
        // Three superfiles with different term ranges; the
        // empty-FST one contributes nothing to the union.
        let s_a = seg(0, 10, &["alpha", "bravo", "charlie"], None, 0.0);
        let s_b = seg(11, 20, &["bravo", "charlie", "delta"], None, 0.0);
        let id = Uuid::new_v4();
        let mut empty_fts = HashMap::new();
        empty_fts.insert(
            "title".into(),
            FtsSummary {
                term_bloom: BloomBuilder::with_n_blocks(16).finish(),
                n_terms_distinct: 0,
                term_range: (Vec::new(), Vec::new()),
            },
        );
        let s_c = Arc::new(SuperfileEntry {
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs: 5,
            id_min: 21,
            id_max: 25,
            scalar_stats: ScalarStatsTable::new(),
            fts_summary: empty_fts,
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        });

        let aggs = aggregates::compute(&[s_a, s_b, s_c]);
        let fts_agg = aggs.fts_summary_agg.get("title").expect("title agg");
        let (mn, mx) = fts_agg.term_range_union.as_ref().expect("range");
        assert_eq!(mn, b"alpha", "min of mins across non-empty FSTs");
        assert_eq!(mx, b"delta", "max of maxes across non-empty FSTs");
    }

    #[test]
    fn aggregates_compute_fts_all_empty_yields_none_range() {
        let id = Uuid::new_v4();
        let mut empty_fts = HashMap::new();
        empty_fts.insert(
            "title".into(),
            FtsSummary {
                term_bloom: BloomBuilder::with_n_blocks(16).finish(),
                n_terms_distinct: 0,
                term_range: (Vec::new(), Vec::new()),
            },
        );
        let s = Arc::new(SuperfileEntry {
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs: 0,
            id_min: 0,
            id_max: 0,
            scalar_stats: ScalarStatsTable::new(),
            fts_summary: empty_fts,
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        });

        let aggs = aggregates::compute(&[s]);
        // Column not in the map (skipped entirely) — list-
        // level pruner treats this as "no info, always-keep".
        assert!(
            !aggs.fts_summary_agg.contains_key("title")
                || aggs
                    .fts_summary_agg
                    .get("title")
                    .expect("agg")
                    .term_range_union
                    .is_none()
        );
    }

    #[test]
    fn aggregates_compute_vector_envelope_bounds_all_segment_balls() {
        let s_a = seg(0, 10, &[], Some(vec![1.0, 0.0, 0.0]), 0.5);
        let s_b = seg(11, 20, &[], Some(vec![0.0, 1.0, 0.0]), 0.5);
        let aggs = aggregates::compute(&[s_a.clone(), s_b.clone()]);
        let v = aggs.vector_summary_agg.get("emb").expect("vec agg");
        let mean = [0.5, 0.5, 0.0];
        // Each segment's centroid is ~0.707 from the mean; +
        // radius 0.5 → envelope_radius >= 1.207.
        assert!(
            v.envelope_radius >= 1.207 - 0.01,
            "envelope radius must dominate each seg ball; got {}",
            v.envelope_radius
        );
        let decoded: Vec<f32> = v
            .centroid_envelope
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(decoded.len(), 3);
        for (i, x) in decoded.iter().enumerate() {
            assert!(
                (x - mean[i]).abs() < 1e-5,
                "envelope[{}]={} expected {}",
                i,
                x,
                mean[i]
            );
        }
    }

    #[test]
    fn aggregates_compute_scalar_min_max_per_column() {
        use std::collections::HashMap as Map;
        fn make(id_min: i128, ts_lo: i64, ts_hi: i64) -> Arc<SuperfileEntry> {
            let id = Uuid::new_v4();
            let mut cols: Map<String, (arrow_array::ArrayRef, arrow_array::ArrayRef)> = Map::new();
            let mn: arrow_array::ArrayRef = Arc::new(Int64Array::from(vec![ts_lo]));
            let mx: arrow_array::ArrayRef = Arc::new(Int64Array::from(vec![ts_hi]));
            cols.insert("ts".into(), (mn, mx));
            Arc::new(SuperfileEntry {
                superfile_id: id,
                uri: SuperfileUri(id),
                n_docs: 1,
                id_min,
                id_max: id_min,
                scalar_stats: ScalarStatsTable { cols },
                fts_summary: HashMap::new(),
                vector_summary: HashMap::new(),
                partition_key: Vec::new(),
                partition_hint: None,
                subsection_offsets: None,
            })
        }
        let segs = vec![make(0, 100, 200), make(1, 50, 150), make(2, 300, 400)];
        let aggs = aggregates::compute(&segs);
        let s = aggs
            .scalar_stats_agg
            .get("ts")
            .expect("ts scalar agg present");
        // IPC byte introspection is a separate concern; here we just
        // confirm presence + non-empty encoding.
        assert!(!s.min.is_empty(), "ts min IPC bytes must be non-empty");
        assert!(!s.max.is_empty(), "ts max IPC bytes must be non-empty");
    }

    #[test]
    fn aggregates_compute_id_range_for_uint64_column_via_stats_table() {
        // The id column's min/max as Arrow stats survive the
        // aggregate path even though id_min/id_max are
        // tracked separately.
        use std::collections::HashMap as Map;
        fn make(id_lo: i128, id_hi: i128) -> Arc<SuperfileEntry> {
            let id = Uuid::new_v4();
            let mut cols: Map<String, (arrow_array::ArrayRef, arrow_array::ArrayRef)> = Map::new();
            let mn: arrow_array::ArrayRef = Arc::new(
                arrow_array::Decimal128Array::from(vec![id_lo])
                    .with_precision_and_scale(38, 0)
                    .expect("decimal128"),
            );
            let mx: arrow_array::ArrayRef = Arc::new(
                arrow_array::Decimal128Array::from(vec![id_hi])
                    .with_precision_and_scale(38, 0)
                    .expect("decimal128"),
            );
            cols.insert("_id".into(), (mn, mx));
            Arc::new(SuperfileEntry {
                superfile_id: id,
                uri: SuperfileUri(id),
                n_docs: 1,
                id_min: id_lo,
                id_max: id_hi,
                scalar_stats: ScalarStatsTable { cols },
                fts_summary: HashMap::new(),
                vector_summary: HashMap::new(),
                partition_key: Vec::new(),
                partition_hint: None,
                subsection_offsets: None,
            })
        }
        let segs = vec![make(0, 99), make(100, 199), make(200, 299)];
        let aggs = aggregates::compute(&segs);
        assert_eq!(aggs.id_range, (0, 299));
        assert!(aggs.scalar_stats_agg.contains_key("_id"));
    }

    // ---- list_prune — query-shape correctness.

    #[test]
    fn prune_parts_for_id_range_filters_non_overlapping_parts() {
        let part0 = entry_from_segments(&[seg(0, 99, &[], None, 0.0)], 0);
        let part1 = entry_from_segments(&[seg(100, 199, &[], None, 0.0)], 1);
        let part2 = entry_from_segments(&[seg(200, 299, &[], None, 0.0)], 2);
        let part3 = entry_from_segments(&[seg(300, 399, &[], None, 0.0)], 3);
        let list = list_with(vec![part0, part1.clone(), part2.clone(), part3]);

        let survivors = prune_parts_for_id_range(&list, 150, 250);
        let ids: Vec<_> = survivors.into_iter().collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&part1.part_id));
        assert!(ids.contains(&part2.part_id));
    }

    #[test]
    fn prune_parts_for_fts_prefix_filters_disjoint_term_ranges() {
        let part0 =
            entry_from_segments(&[seg(0, 10, &["alpha", "bravo", "charlie"], None, 0.0)], 0);
        let part1 =
            entry_from_segments(&[seg(11, 20, &["delta", "echo", "foxtrot"], None, 0.0)], 1);
        let part2 = entry_from_segments(&[seg(21, 30, &["hotel", "kilo", "lima"], None, 0.0)], 2);
        let list = list_with(vec![part0, part1.clone(), part2]);

        let survivors = prune_parts_for_fts_prefix(&list, "title", b"echo");
        assert_eq!(survivors.len(), 1);
        assert_eq!(survivors[0], part1.part_id);
    }

    #[test]
    fn prune_parts_for_fts_prefix_keeps_part_with_no_aggregate() {
        // Part has no FTS aggregate for the queried column —
        // always-keep.
        let part = entry_from_segments(&[seg(0, 10, &[], None, 0.0)], 0);
        let list = list_with(vec![part.clone()]);
        let survivors = prune_parts_for_fts_prefix(&list, "missing", b"any");
        assert_eq!(survivors, vec![part.part_id]);
    }

    #[test]
    fn prune_parts_for_vector_filters_far_parts() {
        let part_a = entry_from_segments(&[seg(0, 10, &[], Some(vec![10.0, 0.0, 0.0]), 0.5)], 0);
        let part_b = entry_from_segments(&[seg(11, 20, &[], Some(vec![-10.0, 0.0, 0.0]), 0.5)], 1);
        let list = list_with(vec![part_a.clone(), part_b]);
        let survivors = prune_parts_for_vector(&list, "emb", &[10.0, 0.0, 0.0], 1.0);
        assert_eq!(survivors.len(), 1);
        assert_eq!(survivors[0], part_a.part_id);
    }

    #[test]
    fn prune_parts_for_vector_keeps_overlapping_envelope() {
        let part_a = entry_from_segments(&[seg(0, 10, &[], Some(vec![1.0, 0.0, 0.0]), 1.0)], 0);
        let part_b = entry_from_segments(&[seg(11, 20, &[], Some(vec![-1.0, 0.0, 0.0]), 1.0)], 1);
        let list = list_with(vec![part_a, part_b]);
        let survivors = prune_parts_for_vector(&list, "emb", &[0.0, 0.0, 0.0], 1.0);
        assert_eq!(
            survivors.len(),
            2,
            "both envelopes contain origin within cutoff"
        );
    }

    #[test]
    fn pruning_is_monotonic_no_false_negatives() {
        // Property: any segment the flat (segment-level)
        // pruner would visit is necessarily in a part the
        // list-level pruner keeps. Aggregates over-
        // approximate the segment-level skip data.
        let segs_part0 = vec![
            seg(0, 10, &["apple"], None, 0.0),
            seg(11, 20, &["banana", "cherry"], None, 0.0),
        ];
        let segs_part1 = vec![
            seg(21, 30, &["alpha"], None, 0.0),
            seg(31, 40, &["echo", "foxtrot"], None, 0.0),
        ];
        let part0 = entry_from_segments(&segs_part0, 0);
        let part1 = entry_from_segments(&segs_part1, 1);
        let list = list_with(vec![part0.clone(), part1.clone()]);

        let survivors = prune_parts_for_fts_prefix(&list, "title", b"ban");
        assert!(
            survivors.contains(&part0.part_id),
            "must keep matching part"
        );

        let survivors2 = prune_parts_for_fts_prefix(&list, "title", b"ec");
        assert!(survivors2.contains(&part1.part_id));
    }
}
