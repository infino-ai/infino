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
//!   bytes (same encoding `ManifestListEntry` uses).
//! - `fts_summary_agg.term_range_union`: per FTS column,
//!   `(min(min_term), max(max_term))` across superfiles with
//!   non-empty FSTs. Absent (`None`) if every superfile's FST
//!   for that column is empty.
//! - `vector_summary_agg`: per vector column, mean-of-
//!   centroids + max(distance + superfile_radius). Bounds every
//!   superfile's vector ball with one outer ball, so the
//!   list-level vector skip is correct by construction (no
//!   false negatives).
//!
//! `fts_summary_agg.term_bloom_union` is currently emitted
//! empty (interpreted as "always-keep" by the list-level
//! pruner) and `n_terms_distinct` as 0; HLL-sized blooms over
//! the union of per-superfile FSTs would require a per-commit
//! cardinality estimate over each superfile's terms.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::ipc::writer::StreamWriter;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Field, Schema};

use crate::supertable::manifest::SuperfileEntry;
use crate::supertable::manifest::list::{FtsSummaryAgg, ScalarStatsAgg, VectorSummaryAgg};

/// All four aggregate buckets for one [`ManifestListEntry`].
/// Built by [`compute`] and inserted verbatim into the entry.
#[derive(Debug, Default)]
pub struct AggregateSet {
    pub id_range: (i128, i128),
    pub scalar_stats_agg: BTreeMap<String, ScalarStatsAgg>,
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
pub fn compute(superfiles: &[Arc<SuperfileEntry>]) -> AggregateSet {
    if superfiles.is_empty() {
        return AggregateSet::default();
    }

    let id_min = superfiles.iter().map(|s| s.id_min).min().unwrap_or(0);
    let id_max = superfiles.iter().map(|s| s.id_max).max().unwrap_or(0);

    AggregateSet {
        id_range: (id_min, id_max),
        scalar_stats_agg: scalar_stats_agg(superfiles),
        fts_summary_agg: fts_summary_agg(superfiles),
        vector_summary_agg: vector_summary_agg(superfiles),
    }
}

// ---------------------------------------------------------
// Scalar stats: per column, min-of-mins / max-of-maxes.
// ---------------------------------------------------------

fn scalar_stats_agg(superfiles: &[Arc<SuperfileEntry>]) -> BTreeMap<String, ScalarStatsAgg> {
    // Gather, per column, all per-superfile (min, max) ArrayRefs.
    let mut per_column: HashMap<String, (Vec<ArrayRef>, Vec<ArrayRef>)> = HashMap::new();
    for seg in superfiles {
        for (col, (mn, mx)) in &seg.scalar_stats.cols {
            let entry = per_column.entry(col.clone()).or_default();
            entry.0.push(mn.clone());
            entry.1.push(mx.clone());
        }
    }

    let mut out = BTreeMap::new();
    for (col, (mins, maxes)) in per_column {
        if mins.is_empty() {
            continue;
        }
        // Concat each side into a single Array, then take its
        // (min, max). Encode as length-1 Arrow IPC bytes (the
        // ScalarStatsAgg shape).
        let combined_min = match concat_arrays(&mins) {
            Some(a) => a,
            None => continue,
        };
        let combined_max = match concat_arrays(&maxes) {
            Some(a) => a,
            None => continue,
        };
        let Some((agg_min, _)) = column_min_max(&combined_min) else {
            continue;
        };
        let Some((_, agg_max)) = column_min_max(&combined_max) else {
            continue;
        };
        let min_bytes = ipc_encode_length1(&col, &agg_min);
        let max_bytes = ipc_encode_length1(&col, &agg_max);

        // Additive rollups (null count / sum / HLL) combine with
        // intersection semantics: every segment must carry the stat or
        // the part-level value is unknowable (`None`, never zero).
        let null_count = superfiles.iter().try_fold(0u64, |acc, seg| {
            acc.checked_add(*seg.scalar_stats.null_counts.get(&col)?)
        });
        let sum = superfiles
            .iter()
            .try_fold(None::<ArrayRef>, |acc, seg| {
                let part = seg.scalar_stats.sums.get(&col)?;
                Some(Some(match acc {
                    None => part.clone(),
                    Some(total) => crate::supertable::manifest::add_sum_arrays(&total, part)?,
                }))
            })
            .flatten()
            .map(|total| ipc_encode_length1(&col, &total));
        let hll = superfiles
            .iter()
            .try_fold(
                None::<crate::supertable::manifest::hll::HllSketch>,
                |acc, seg| {
                    let sketch = crate::supertable::manifest::hll::HllSketch::from_bytes(
                        seg.scalar_stats.hll.get(&col)?,
                    )?;
                    Some(Some(match acc {
                        None => sketch,
                        Some(mut merged) => {
                            merged.merge(&sketch);
                            merged
                        }
                    }))
                },
            )
            .flatten()
            .map(|sketch| sketch.as_bytes().to_vec());

        out.insert(
            col,
            ScalarStatsAgg {
                min: min_bytes,
                max: max_bytes,
                null_count,
                sum,
                hll,
            },
        );
    }
    out
}

fn concat_arrays(arrays: &[ArrayRef]) -> Option<ArrayRef> {
    let refs: Vec<&dyn arrow_array::Array> = arrays.iter().map(|a| a.as_ref()).collect();
    arrow::compute::concat(&refs).ok()
}

/// Compute (min, max) of a single combined Array as length-1
/// `ArrayRef`s. Mirrors the helper in `manifest/mod.rs`
/// (private there); duplicated to avoid pub-ifying that one
/// since their callers will diverge over time.
fn column_min_max(col: &ArrayRef) -> Option<(ArrayRef, ArrayRef)> {
    use arrow::compute::kernels::aggregate as agg;
    use arrow_array::*;
    use arrow_schema::DataType;
    macro_rules! prim {
        ($ty:ty) => {{
            let a = col.as_any().downcast_ref::<$ty>()?;
            let mn = agg::min(a)?;
            let mx = agg::max(a)?;
            let mn_arr: ArrayRef = Arc::new(<$ty>::from(vec![mn]));
            let mx_arr: ArrayRef = Arc::new(<$ty>::from(vec![mx]));
            Some((mn_arr, mx_arr))
        }};
    }
    match col.data_type() {
        DataType::Int8 => prim!(Int8Array),
        DataType::Int16 => prim!(Int16Array),
        DataType::Int32 => prim!(Int32Array),
        DataType::Int64 => prim!(Int64Array),
        DataType::UInt8 => prim!(UInt8Array),
        DataType::UInt16 => prim!(UInt16Array),
        DataType::UInt32 => prim!(UInt32Array),
        DataType::UInt64 => prim!(UInt64Array),
        DataType::Float32 => prim!(Float32Array),
        DataType::Float64 => prim!(Float64Array),
        DataType::Boolean => {
            let a = col.as_any().downcast_ref::<BooleanArray>()?;
            let mn = agg::min_boolean(a)?;
            let mx = agg::max_boolean(a)?;
            let mn_arr: ArrayRef = Arc::new(BooleanArray::from(vec![mn]));
            let mx_arr: ArrayRef = Arc::new(BooleanArray::from(vec![mx]));
            Some((mn_arr, mx_arr))
        }
        DataType::Utf8 => {
            let a = col.as_any().downcast_ref::<StringArray>()?;
            let mn = agg::min_string(a)?;
            let mx = agg::max_string(a)?;
            let mn_arr: ArrayRef = Arc::new(StringArray::from(vec![mn]));
            let mx_arr: ArrayRef = Arc::new(StringArray::from(vec![mx]));
            Some((mn_arr, mx_arr))
        }
        DataType::LargeUtf8 => {
            let a = col.as_any().downcast_ref::<LargeStringArray>()?;
            let mn = agg::min_string(a)?;
            let mx = agg::max_string(a)?;
            let mn_arr: ArrayRef = Arc::new(LargeStringArray::from(vec![mn]));
            let mx_arr: ArrayRef = Arc::new(LargeStringArray::from(vec![mx]));
            Some((mn_arr, mx_arr))
        }
        DataType::Decimal128(precision, scale) => {
            // The id column (`_id`) is the canonical
            // Decimal128 column. Min/max via Arrow's
            // generic aggregate kernel; reconstruct the
            // length-1 array with the same precision +
            // scale so downstream IPC encoding preserves
            // the type identity.
            let a = col.as_any().downcast_ref::<Decimal128Array>()?;
            let mn = agg::min(a)?;
            let mx = agg::max(a)?;
            let mn_arr: ArrayRef = Arc::new(
                Decimal128Array::from(vec![mn])
                    .with_precision_and_scale(*precision, *scale)
                    .ok()?,
            );
            let mx_arr: ArrayRef = Arc::new(
                Decimal128Array::from(vec![mx])
                    .with_precision_and_scale(*precision, *scale)
                    .ok()?,
            );
            Some((mn_arr, mx_arr))
        }
        _ => None,
    }
}

/// Serialize a single length-1 ArrayRef as Arrow IPC bytes —
/// matches the `ScalarStatsAgg.{min,max}` wire shape
/// (the per-summary encoding the manifest part carries; we
/// mirror it at the list level so decoders are uniform).
fn ipc_encode_length1(col_name: &str, arr: &ArrayRef) -> Vec<u8> {
    let field = Field::new(col_name, arr.data_type().clone(), true);
    let schema = Arc::new(Schema::new(vec![field]));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![arr.clone()]).expect("schema/array match");
    let mut out = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut out, &schema).expect("ipc writer init");
        writer.write(&batch).expect("ipc write");
        writer.finish().expect("ipc finish");
    }
    out
}

// ---------------------------------------------------------
// FTS summary aggregate: term_range_union + term_bloom_union.
// ---------------------------------------------------------
//
// Bloom union is the bit-OR of each superfile's bloom for the
// column. Same bloom-shape across all superfiles (the writer
// builds with `BloomBuilder::new()` — fixed `DEFAULT_N_BLOCKS`
// + xxh3_64), so bit-OR preserves "any superfile contained
// this term" with the same FPR. The block-and-mask scheme
// `Bloom::contains` uses is positional, not arithmetic,
// so bit-OR is exact for the union semantic. The
// `n_terms_distinct` field stays 0 — HLL-based estimation
// would let the planner pick prune ordering across columns,
// but isn't required for correctness; lands when measured.

type FtsColumnAccumulator = (Option<Vec<u8>>, Option<Vec<u8>>, Option<Vec<u8>>, u32);

fn fts_summary_agg(superfiles: &[Arc<SuperfileEntry>]) -> BTreeMap<String, FtsSummaryAgg> {
    // Per-column accumulators: (min, max, union-bloom-bytes,
    // n_blocks). The union bloom starts at zero-init the
    // first time we see a superfile with a populated bloom for
    // that column; subsequent superfiles bit-OR into it.
    let mut per_column: HashMap<String, FtsColumnAccumulator> = HashMap::new();
    for seg in superfiles {
        for (col, summary) in &seg.fts_summary {
            let entry = per_column.entry(col.clone()).or_default();
            // Range update (skipping superfiles with empty FST
            // ranges; they contribute neither min nor max).
            let has_range = !(summary.term_range.0.is_empty() && summary.term_range.1.is_empty());
            if has_range {
                match &entry.0 {
                    Some(curr_min) if curr_min.as_slice() <= summary.term_range.0.as_slice() => {}
                    _ => entry.0 = Some(summary.term_range.0.clone()),
                }
                match &entry.1 {
                    Some(curr_max) if curr_max.as_slice() >= summary.term_range.1.as_slice() => {}
                    _ => entry.1 = Some(summary.term_range.1.clone()),
                }
            }
            // Union the superfile's bloom into the part-level
            // bloom. Skip zero-length blooms (superfiles with
            // no FST entries for this column).
            let seg_bytes = summary.term_bloom.to_bytes();
            if seg_bytes.is_empty() {
                continue;
            }
            match &mut entry.2 {
                None => {
                    entry.2 = Some(seg_bytes);
                    entry.3 = summary.term_bloom.n_blocks() as u32;
                }
                Some(acc) => {
                    // Bloom-union invariant: all superfiles
                    // share the same shape. If a length
                    // mismatch ever shows up, drop the
                    // union to "no info" — correctness is
                    // preserved (list-level prune treats
                    // empty union as always-keep), only
                    // selectivity suffers. This shouldn't
                    // happen in practice: the writer's bloom
                    // size is fixed at construction.
                    if acc.len() == seg_bytes.len() {
                        for (a, b) in acc.iter_mut().zip(seg_bytes.iter()) {
                            *a |= *b;
                        }
                    } else {
                        // Mismatched shapes; fall back.
                        entry.2 = None;
                        entry.3 = 0;
                    }
                }
            }
        }
    }
    let mut out = BTreeMap::new();
    for (col, (mn, mx, bloom_bytes, n_blocks)) in per_column {
        let term_range_union = match (mn, mx) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        };
        let (term_bloom_union, term_bloom_n_blocks) = match bloom_bytes {
            Some(b) => (b, n_blocks),
            None => (Vec::new(), 0),
        };
        out.insert(
            col,
            FtsSummaryAgg {
                term_bloom_union,
                term_bloom_n_blocks,
                // HLL-estimated distinct term count stays
                // deferred — it's a planner hint, not a
                // correctness requirement.
                n_terms_distinct: 0,
                term_range_union,
            },
        );
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
    use super::*;
    use crate::supertable::manifest::{FtsSummary, ScalarStatsTable, SuperfileEntry, SuperfileUri};
    use arrow_array::{ArrayRef, LargeStringArray, StringArray};
    use std::collections::HashMap;

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
        cols.insert(col.to_string(), (mn, mx));
        Arc::new(SuperfileEntry {
            superfile_id: uuid::Uuid::new_v4(),
            uri: SuperfileUri::new_v4(),
            n_docs: 1,
            id_min: 0,
            id_max: 0,
            scalar_stats: ScalarStatsTable {
                cols,
                ..Default::default()
            },
            fts_summary: HashMap::<String, FtsSummary>::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    fn decode_ipc_string(bytes: &[u8]) -> String {
        use arrow::ipc::reader::StreamReader;
        let mut reader = StreamReader::try_new(bytes, None).expect("ipc reader");
        let batch = reader.next().expect("at least one batch").expect("ok");
        // The encoder produces a length-1 batch with the column under its name.
        let col = batch.column(0);
        if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
            return a.value(0).to_string();
        }
        if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
            return a.value(0).to_string();
        }
        panic!(
            "expected Utf8 or LargeUtf8 column; got {:?}",
            col.data_type()
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
        assert_eq!(decode_ipc_string(&agg.min), "alpha");
        assert_eq!(decode_ipc_string(&agg.max), "echo");
    }

    #[test]
    fn scalar_stats_agg_unions_large_utf8_min_max_across_superfiles() {
        let segs = vec![
            seg_with_string_minmax("body", "mango", "papaya", true),
            seg_with_string_minmax("body", "apple", "orange", true),
        ];
        let aggs = scalar_stats_agg(&segs);
        let agg = aggs.get("body").expect("body agg present");
        assert_eq!(decode_ipc_string(&agg.min), "apple");
        assert_eq!(decode_ipc_string(&agg.max), "papaya");
    }
}
