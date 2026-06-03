//! Manifest-level skip pruning helpers.
//!
//! Each helper takes a pinned [`Manifest`] snapshot plus a query
//! shape and returns a `Vec<bool>` mask — one slot per segment, in
//! manifest order — where `true` means "keep" and `false` means
//! "prune".  The masks are pure functions of manifest metadata
//! ([`SuperfileEntry::scalar_stats`], [`SuperfileEntry::fts_summary`],
//! [`SuperfileEntry::vector_summary`]) — **no store calls**.
//! Pruned superfiles are dropped before the query layer issues any
//! per-segment work, so an irrelevant segment never causes a
//! `SuperfileReaderCache::reader` call (the load-bearing perf claim of
//! the skip layer).
//!
//! Helpers are independent and idempotent. In v1, the BM25
//! query paths consume `fts_bloom_skip` (exact-term) and
//! `fts_prefix_skip` (prefix); vector and SQL paths do not yet
//! consume their helpers (see those modules' headers).
//!
//! ## Conservatism
//!
//! All helpers err on the side of keeping a segment when in
//! doubt:
//!
//! - Unknown column → keep all (per-segment search will surface
//!   the column-missing error to the caller).
//! - All-zero or absent summary → keep (treat as "may match").
//! - Empty query (no terms / `prefix == ""`) → keep all.
//!
//! False-positive keeps cost a per-segment search call but never
//! a wrong answer. False-negative prunes would silently drop
//! relevant docs and are forbidden.
//!
//! ## Vector centroid skip
//!
//! Conservative pre-cutoff pruning is hard for IVF vectors
//! because we don't know the global top-k cutoff distance until
//! at least one segment has been searched. v1
//! [`vector_centroid_skip`] returns all-keep and exposes
//! [`superfiles_sorted_by_centroid_distance`] so a future
//! incremental top-k pruning layer has the ordering it needs
//! without yet committing to a specific early-termination
//! algorithm.

use std::sync::Arc;

use crate::superfile::fts::reader::BoolMode;
use crate::superfile::vector::distance::{Metric, distance};

use crate::supertable::manifest::{Manifest, SuperfileEntry, term_range::prefix_overlaps_range};

/// Bloom-skip mask for an exact-term BM25 search.
///
/// For each segment, look up every tokenized query term in the
/// segment's per-column term-presence bloom:
///
/// - `BoolMode::Or`  — keep if **any** term is possibly-present
///   (a doc containing any term contributes a positive score).
/// - `BoolMode::And` — keep if **all** terms are possibly-present
///   (a relevant doc must contain every term, so a single
///   definitely-absent term prunes the whole segment).
///
/// `query_terms` are the terms after the same tokenizer used at
/// index time. Per the v1 tokenizer (`AsciiLowerTokenizer`) that
/// means already-lowercased ASCII tokens — no whitespace splits
/// inside individual entries.
///
/// An empty `query_terms` slice short-circuits to all-keep (the
/// BM25 search itself returns an empty result, but pruning
/// superfiles preemptively would mask that signal).
pub fn fts_bloom_skip(
    superfiles: &[Arc<SuperfileEntry>],
    column: &str,
    query_terms: &[&str],
    mode: BoolMode,
) -> Vec<bool> {
    if query_terms.is_empty() {
        return vec![true; superfiles.len()];
    }
    superfiles
        .iter()
        .map(|entry| match entry.fts_summary.get(column) {
            None => true,
            Some(summary) => match mode {
                BoolMode::Or => query_terms
                    .iter()
                    .any(|t| summary.term_bloom.contains(t.as_bytes())),
                BoolMode::And => query_terms
                    .iter()
                    .all(|t| summary.term_bloom.contains(t.as_bytes())),
            },
        })
        .collect()
}

/// Term-range skip mask for a prefix BM25 search.
///
/// For each segment, check whether `[prefix, prefix_upper_bound)`
/// overlaps the segment's lex term range
/// `[fts_summary.term_range.0, fts_summary.term_range.1]`. A
/// non-overlapping segment cannot contain any term beginning with
/// `prefix` and is pruned.
///
/// `prefix` is the same lowercased byte sequence the prefix
/// search uses against the FST — see [`prefix_overlaps_range`]
/// for the exact comparison semantics.
///
/// An empty `prefix` (every term matches) short-circuits to
/// all-keep.
pub fn fts_prefix_skip(
    superfiles: &[Arc<SuperfileEntry>],
    column: &str,
    prefix: &[u8],
) -> Vec<bool> {
    if prefix.is_empty() {
        return vec![true; superfiles.len()];
    }
    superfiles
        .iter()
        .map(|entry| match entry.fts_summary.get(column) {
            None => true,
            Some(summary) => {
                let (min_term, max_term) = &summary.term_range;
                if min_term.is_empty() && max_term.is_empty() {
                    // 0-term segment — bloom build also flags
                    // this; nothing matches. Prune.
                    return false;
                }
                prefix_overlaps_range(prefix, min_term, max_term)
            }
        })
        .collect()
}

/// Vector centroid skip mask for a kNN search.
///
/// **v1 returns all-keep.** Cluster-aware skip in IVF with
/// 1-bit RaBitQ shortlist + full-precision rerank requires a
/// running top-k cutoff distance to drive triangle-inequality
/// pruning, which only becomes available *during* fan-out. The
/// machinery for incremental cutoff-driven termination lands
/// once the bench harness has the per-stage latency numbers to
/// motivate the right shape.
///
/// Until then, callers can use
/// [`superfiles_sorted_by_centroid_distance`] to bias fan-out
/// order toward likely-close superfiles — that alone gives a
/// near-cutoff result fast for cache-aware top-k merging.
pub fn vector_centroid_skip(manifest: &Manifest, _column: &str, _query: &[f32]) -> Vec<bool> {
    vec![true; manifest.superfiles.len()]
}

/// Indices into `manifest.superfiles` sorted ascending by the
/// per-segment centroid's distance to `query` under `metric`.
///
/// Segments without a vector summary for `column` are sorted to
/// the end (treated as worst-case). Used as a fan-out hint for
/// vector search: searching closer-centroid superfiles first means
/// later superfiles are likelier to be skippable once the running
/// top-k has converged.
///
/// Returns indices, not entries, to keep the caller in control
/// of how to materialize the ordered fan-out (rayon `par_iter`
/// over indices is the typical shape).
pub fn superfiles_sorted_by_centroid_distance(
    manifest: &Manifest,
    column: &str,
    query: &[f32],
    metric: Metric,
) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = manifest
        .superfiles
        .iter()
        .enumerate()
        .map(|(i, entry)| match entry.vector_summary.get(column) {
            Some(vs) if vs.centroid.len() == query.len() => {
                (i, distance(metric, query, &vs.centroid))
            }
            _ => (i, f32::INFINITY),
        })
        .collect();
    // pdqsort: per-query segment skip ordering. (segment_idx, dist)
    // tuples are unique by segment_idx, so any tie-break is fine.
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use uuid::Uuid;

    use crate::superfile::builder::{FtsConfig, VectorConfig};

    use crate::superfile::vector::distance::Metric;
    use crate::supertable::SupertableOptions;
    use crate::supertable::manifest::{
        FtsSummary, Manifest, ScalarStatsTable, SuperfileEntry, SuperfileUri, VectorSummary,
        bloom::BloomBuilder,
    };
    use arrow_schema::{DataType, Field, Schema};

    use super::*;

    fn opts_simple() -> Arc<SupertableOptions> {
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

    fn opts_with_vector() -> Arc<SupertableOptions> {
        // dim ≥ 16 per SupertableOptions invariant.
        let dim = 16;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "emb",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        )]));
        Arc::new(
            SupertableOptions::new(
                schema,
                vec![],
                vec![VectorConfig {
                    column: "emb".into(),
                    dim,
                    n_cent: 4,
                    rot_seed: 0,
                    metric: Metric::Cosine,
                    rerank_codec: crate::superfile::vector::rerank_codec::RerankCodec::Fp32,
                }],
                None,
            )
            .expect("opts"),
        )
    }

    fn empty_segment() -> SuperfileEntry {
        let uri = SuperfileUri::new_v4();
        SuperfileEntry {
            superfile_id: Uuid::new_v4(),
            uri,
            n_docs: 0,
            id_min: 0,
            id_max: 0,
            scalar_stats: ScalarStatsTable::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        }
    }

    /// Build a one-column FTS summary with the given indexed terms.
    fn fts_summary_with(column: &str, terms: &[&str]) -> (String, FtsSummary) {
        let mut bb = BloomBuilder::new();
        for t in terms {
            bb.insert(t.as_bytes());
        }
        let term_range = match (terms.first(), terms.last()) {
            (Some(min), Some(max)) => (min.as_bytes().to_vec(), max.as_bytes().to_vec()),
            _ => (Vec::new(), Vec::new()),
        };
        let summary = FtsSummary {
            term_bloom: bb.finish(),
            n_terms_distinct: terms.len() as u32,
            term_range,
        };
        (column.to_string(), summary)
    }

    fn segment_with_terms(column: &str, terms: &[&str]) -> Arc<SuperfileEntry> {
        let mut e = empty_segment();
        let (k, v) = fts_summary_with(column, terms);
        e.fts_summary.insert(k, v);
        Arc::new(e)
    }

    fn segment_with_centroid(column: &str, centroid: Vec<f32>, radius: f32) -> Arc<SuperfileEntry> {
        let mut e = empty_segment();
        e.vector_summary
            .insert(column.to_string(), VectorSummary { centroid, radius });
        Arc::new(e)
    }

    fn manifest_with(
        opts: Arc<SupertableOptions>,
        superfiles: Vec<Arc<SuperfileEntry>>,
    ) -> Manifest {
        // build via `with_appended` so the new outer
        // Manifest's metadata fields (list, parts, loader) get
        // initialized correctly. Equivalent to the old direct-
        // field-assignment helper for the test's purposes.
        Manifest::empty(opts).with_appended(superfiles)
    }

    // ---- fts_bloom_skip ----------------------------------------------

    #[test]
    fn bloom_skip_keeps_segments_with_any_query_term_in_or_mode() {
        let s_a = segment_with_terms("title", &["alpha", "beta"]);
        let s_b = segment_with_terms("title", &["gamma", "delta"]);
        let m = manifest_with(opts_simple(), vec![s_a, s_b]);
        let mask = fts_bloom_skip(&m.superfiles, "title", &["alpha", "missing"], BoolMode::Or);
        // Segment A has alpha → keep. Segment B has neither → prune.
        assert_eq!(mask, vec![true, false]);
    }

    #[test]
    fn bloom_skip_requires_all_terms_present_in_and_mode() {
        let s_a = segment_with_terms("title", &["alpha", "beta"]);
        let s_b = segment_with_terms("title", &["alpha", "gamma"]);
        let m = manifest_with(opts_simple(), vec![s_a, s_b]);
        let mask = fts_bloom_skip(&m.superfiles, "title", &["alpha", "beta"], BoolMode::And);
        // Segment A has both. Segment B is missing 'beta' → prune.
        assert_eq!(mask, vec![true, false]);
    }

    #[test]
    fn bloom_skip_unknown_column_keeps_all() {
        let s = segment_with_terms("title", &["alpha"]);
        let m = manifest_with(opts_simple(), vec![s]);
        let mask = fts_bloom_skip(&m.superfiles, "no_such_column", &["alpha"], BoolMode::Or);
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn bloom_skip_empty_terms_keeps_all() {
        let s = segment_with_terms("title", &["alpha"]);
        let m = manifest_with(opts_simple(), vec![s]);
        let mask = fts_bloom_skip(&m.superfiles, "title", &[], BoolMode::Or);
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn bloom_skip_with_no_segments_returns_empty_vec() {
        let m = manifest_with(opts_simple(), vec![]);
        let mask = fts_bloom_skip(&m.superfiles, "title", &["alpha"], BoolMode::Or);
        assert!(mask.is_empty());
    }

    // ---- fts_prefix_skip ---------------------------------------------

    #[test]
    fn prefix_skip_prunes_segments_outside_prefix_range() {
        // Segment A: terms in ['apple', 'banana'] → prefix "rust"
        //            doesn't overlap.
        // Segment B: terms in ['python', 'rust']  → prefix "rust"
        //            overlaps the upper end.
        let s_a = segment_with_terms("title", &["apple", "banana"]);
        let s_b = segment_with_terms("title", &["python", "rust"]);
        let m = manifest_with(opts_simple(), vec![s_a, s_b]);
        let mask = fts_prefix_skip(&m.superfiles, "title", b"rust");
        assert_eq!(mask, vec![false, true]);
    }

    #[test]
    fn prefix_skip_keeps_segments_with_matching_prefix_inside_range() {
        // Terms ['rusting', 'rusty'] → prefix "rust" overlaps.
        let s = segment_with_terms("title", &["rusting", "rusty"]);
        let m = manifest_with(opts_simple(), vec![s]);
        let mask = fts_prefix_skip(&m.superfiles, "title", b"rust");
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn prefix_skip_empty_prefix_keeps_all() {
        let s = segment_with_terms("title", &["alpha"]);
        let m = manifest_with(opts_simple(), vec![s]);
        let mask = fts_prefix_skip(&m.superfiles, "title", b"");
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn prefix_skip_unknown_column_keeps_all() {
        let s = segment_with_terms("title", &["alpha"]);
        let m = manifest_with(opts_simple(), vec![s]);
        let mask = fts_prefix_skip(&m.superfiles, "no_such_column", b"alp");
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn prefix_skip_zero_term_segment_pruned() {
        // Empty term_range = no terms indexed. Prefix can't match.
        let s = Arc::new(empty_segment());
        let m = manifest_with(opts_simple(), vec![s]);
        let mask = fts_prefix_skip(&m.superfiles, "title", b"rust");
        // No FTS summary on the segment → keep (column-missing
        // path). Sanity: this is the "unknown column" path, not
        // the "0-term FTS column" path.
        assert_eq!(mask, vec![true]);
    }

    // ---- vector_centroid_skip + ordering ------------------------------

    #[test]
    fn vector_centroid_skip_v1_keeps_all_segments() {
        let s_a = segment_with_centroid("emb", vec![0.0; 16], 0.5);
        let s_b = segment_with_centroid("emb", vec![10.0; 16], 0.5);
        let m = manifest_with(opts_with_vector(), vec![s_a, s_b]);
        let q = vec![0.0f32; 16];
        let mask = vector_centroid_skip(&m, "emb", &q);
        assert_eq!(mask, vec![true, true]);
    }

    #[test]
    fn superfiles_sorted_by_centroid_distance_orders_by_metric() {
        // L2-sq metric on simple 1-hot centroids.
        let opts = opts_with_vector();
        let near = segment_with_centroid(
            "emb",
            {
                let mut v = vec![0.0f32; 16];
                v[0] = 1.0;
                v
            },
            0.0,
        );
        let far = segment_with_centroid(
            "emb",
            {
                let mut v = vec![0.0f32; 16];
                v[7] = 1.0;
                v
            },
            0.0,
        );
        let m = manifest_with(opts, vec![far.clone(), near.clone()]);
        let q = {
            let mut v = vec![0.0f32; 16];
            v[0] = 1.0;
            v
        };
        let order = superfiles_sorted_by_centroid_distance(&m, "emb", &q, Metric::L2Sq);
        // `near` (idx 1) should come before `far` (idx 0).
        assert_eq!(order, vec![1, 0]);
    }

    #[test]
    fn superfiles_sorted_by_centroid_distance_pushes_missing_summary_to_end() {
        let with_v = segment_with_centroid("emb", vec![1.0f32; 16], 0.0);
        let without_v = Arc::new(empty_segment());
        let m = manifest_with(opts_with_vector(), vec![without_v, with_v]);
        let q = vec![1.0f32; 16];
        let order = superfiles_sorted_by_centroid_distance(&m, "emb", &q, Metric::L2Sq);
        // Index 1 (has summary) sorted before index 0 (missing).
        assert_eq!(order, vec![1, 0]);
    }
}
