// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! ManifestSnapshot-level skip pruning helpers.
//!
//! Each helper takes a pinned [`ManifestSnapshot`] snapshot plus a query
//! shape and returns a `Vec<bool>` mask — one slot per superfile, in
//! manifest order — where `true` means "keep" and `false` means
//! "prune".  The masks are pure functions of manifest metadata
//! ([`SuperfileEntry::scalar_stats`], [`SuperfileEntry::fts_summary`],
//! [`SuperfileEntry::vector_summary`]) — **no store calls**.
//! Pruned superfiles are dropped before the query layer issues any
//! per-superfile work, so an irrelevant superfile never causes a
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
//! All helpers err on the side of keeping a superfile when in
//! doubt:
//!
//! - Unknown column → keep all (per-superfile search will surface
//!   the column-missing error to the caller).
//! - All-zero or absent summary → keep (treat as "may match").
//! - Empty query (no terms / `prefix == ""`) → keep all.
//!
//! False-positive keeps cost a per-superfile search call but never
//! a wrong answer. False-negative prunes would silently drop
//! relevant docs and are forbidden.
//!
//! ## Vector centroid skip
//!
//! Conservative pre-cutoff pruning is hard for IVF vectors
//! because we don't know the global top-k cutoff distance until
//! at least one superfile has been searched. v1
//! [`vector_centroid_skip`] returns all-keep and exposes
//! [`superfiles_sorted_by_centroid_distance`] so a future
//! incremental top-k pruning layer has the ordering it needs
//! without yet committing to a specific early-termination
//! algorithm.

use std::{cmp::Ordering, sync::Arc};

use arrow_schema::DataType;
use datafusion::scalar::ScalarValue;

use crate::{
    superfile::{
        fts::reader::BoolMode,
        vector::distance::{Metric, distance},
    },
    supertable::{
        manifest::{ManifestSnapshot, ScalarStatsAgg, SuperfileEntry},
        schema::{FieldId, map::Resolution},
    },
};

/// Bloom-skip mask for an exact-term BM25 search.
///
/// For each superfile, look up every tokenized query term in the
/// superfile's per-column term-presence bloom:
///
/// - `BoolMode::Or`  — keep if **any** term is possibly-present
///   (a doc containing any term contributes a positive score).
/// - `BoolMode::And` — keep if **all** terms are possibly-present
///   (a relevant doc must contain every term, so a single
///   definitely-absent term prunes the whole superfile).
///
/// `query_terms` are the terms after the same tokenizer used at
/// index time (the column's analyzer). For an `ascii_lower` column
/// that means already-lowercased ASCII tokens; there are no
/// whitespace splits inside individual entries.
///
/// An empty `query_terms` slice short-circuits to all-keep (the
/// BM25 search itself returns an empty result, but pruning
/// superfiles preemptively would mask that signal).
pub fn fts_bloom_skip(
    superfiles: &[Arc<SuperfileEntry>],
    column: FieldId,
    query_terms: &[&str],
    mode: BoolMode,
) -> Vec<bool> {
    if query_terms.is_empty() {
        return vec![true; superfiles.len()];
    }
    superfiles
        .iter()
        .map(|entry| match entry.fts_summary.get(&column) {
            None => true,
            Some(summary) => match mode {
                BoolMode::Or => query_terms
                    .iter()
                    .any(|t| summary.may_contain(t.as_bytes())),
                BoolMode::And => query_terms
                    .iter()
                    .all(|t| summary.may_contain(t.as_bytes())),
            },
        })
        .collect()
}

/// Term-range skip mask for a prefix BM25 search.
///
/// For each superfile, check whether `[prefix, prefix_upper_bound)`
/// overlaps the superfile's lex term range (via
/// [`FtsSummaryAgg::may_match_prefix`]). A non-overlapping superfile
/// cannot contain any term beginning with `prefix` and is pruned.
///
/// `prefix` is the same lowercased byte sequence the prefix search uses
/// against the FST.
///
/// An empty `prefix` (every term matches) short-circuits to
/// all-keep.
///
/// [`FtsSummaryAgg::may_match_prefix`]: crate::supertable::manifest::FtsSummaryAgg::may_match_prefix
pub fn fts_prefix_skip(
    superfiles: &[Arc<SuperfileEntry>],
    column: FieldId,
    prefix: &[u8],
) -> Vec<bool> {
    if prefix.is_empty() {
        return vec![true; superfiles.len()];
    }
    superfiles
        .iter()
        .map(|entry| match entry.fts_summary.get(&column) {
            None => true,
            // `may_match_prefix` returns false for a `None` range (0-term
            // superfile — nothing matches, prune).
            Some(summary) => summary.may_match_prefix(prefix),
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
pub fn vector_centroid_skip(
    manifest: &ManifestSnapshot,
    _column: &str,
    _query: &[f32],
) -> Vec<bool> {
    vec![true; manifest.superfiles.len()]
}

/// Indices into `manifest.superfiles` sorted ascending by the
/// per-superfile centroid's distance to `query` under `metric`.
///
/// Superfiles without a vector summary for `column` are sorted to
/// the end (treated as worst-case). Used as a fan-out hint for
/// vector search: searching closer-centroid superfiles first means
/// later superfiles are likelier to be skippable once the running
/// top-k has converged.
///
/// Returns indices, not entries, to keep the caller in control
/// of how to materialize the ordered fan-out (rayon `par_iter`
/// over indices is the typical shape).
pub fn superfiles_sorted_by_centroid_distance(
    manifest: &ManifestSnapshot,
    column: FieldId,
    query: &[f32],
    metric: Metric,
) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = manifest
        .superfiles
        .iter()
        .enumerate()
        .map(|(i, entry)| match entry.vector_summary.get(&column) {
            Some(vs) if vs.centroid.len() == query.len() => {
                (i, distance(metric, query, &vs.centroid))
            }
            _ => (i, f32::INFINITY),
        })
        .collect();
    // pdqsort: per-query superfile skip ordering. (superfile_idx, dist)
    // tuples are unique by superfile_idx, so any tie-break is fine.
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    scored.into_iter().map(|(i, _)| i).collect()
}

/// Comparison operator in a normalized scalar-skip predicate.
///
/// These mirror the SQL comparison operators the
/// `SupertableProvider` lowers from a DataFusion `Expr` into
/// infino's own predicate form. Any operator we can't normalize is
/// simply never handed to [`scalar_skip`] — the superfile is kept and
/// DataFusion's `FilterExec` still applies the predicate to rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

/// One conjunct of a SQL `WHERE` clause, normalized to
/// `column <op> literal`. The literal is a DataFusion
/// [`ScalarValue`]; [`scalar_skip`] coerces it to the column's
/// stored stat type at compare time.
#[derive(Debug, Clone)]
pub struct ScalarPredicate {
    /// Scalar column name; must match a key in
    /// `SuperfileEntry::scalar_stats` to contribute any pruning.
    pub column: String,
    /// Comparison operator.
    pub op: ScalarOp,
    /// Right-hand-side literal from the query.
    pub value: ScalarValue,
}

/// Scalar-skip mask for a conjunction of `column <op> literal`
/// predicates (a SQL `WHERE` of `AND`-ed simple comparisons).
///
/// For each superfile, consult the per-column min/max persisted in
/// [`SuperfileEntry::scalar_stats`] and keep the superfile unless
/// some predicate *proves* no row in the superfile can satisfy it.
/// Because the predicates are conjunctive, a single
/// definitely-false predicate prunes the whole superfile.
///
/// Conservatism (never a false prune): a superfile is kept when
///
/// - the column has no persisted stats (the writer skips types
///   whose ordering isn't well-defined, and all-null columns),
/// - either bound is NULL,
/// - the literal can't be coerced to the column's stat type, or
/// - the values are otherwise incomparable.
///
/// An empty predicate slice keeps every superfile.
///
/// This is the SQL-side sibling of [`fts_bloom_skip`] /
/// [`fts_prefix_skip`]: **infino owns superfile selection.**
/// DataFusion only executes over the surviving superfiles (and does
/// its own row-group/page pruning inside each Parquet superfile).
/// `predicates` paired with the id of the column each names; a predicate
/// on a column the table does not have keeps every superfile.
pub fn scalar_skip(
    manifest: &ManifestSnapshot,
    superfiles: &[Arc<SuperfileEntry>],
    predicates: &[(Option<FieldId>, &ScalarPredicate)],
) -> Vec<bool> {
    if predicates.is_empty() {
        return vec![true; superfiles.len()];
    }
    let guarded: Vec<(Option<ColumnTypeGuard>, &ScalarPredicate)> = predicates
        .iter()
        .map(|(id, p)| (id.map(|id| ColumnTypeGuard::new(manifest, id)), *p))
        .collect();
    superfiles
        .iter()
        .map(|entry| {
            guarded.iter().all(|(guard, p)| match guard {
                None => true,
                Some(guard) => {
                    guard.stats_are_stale(entry) || superfile_may_match(entry, guard.id(), p)
                }
            })
        })
        .collect()
}

/// Keep each superfile whose `column` min/max could hold *any* of
/// `values` (an `IN` list is a disjunction). Empty `values` keeps all.
/// The SQL-side sibling of [`scalar_skip`] for the `IN` shape.
pub fn scalar_value_set_skip(
    manifest: &ManifestSnapshot,
    superfiles: &[Arc<SuperfileEntry>],
    column: FieldId,
    values: &[ScalarValue],
) -> Vec<bool> {
    if values.is_empty() {
        return vec![true; superfiles.len()];
    }

    let guard = ColumnTypeGuard::new(manifest, column);
    superfiles
        .iter()
        .map(|entry| {
            if guard.stats_are_stale(entry) {
                return true;
            }
            match superfile_minmax(entry, column) {
                None => true,
                Some((min, max)) => values
                    .iter()
                    .any(|v| scalar_value_may_match(&min, &max, ScalarOp::Eq, v)),
            }
        })
        .collect()
}

/// Keep each superfile whose `column` stats could still satisfy
/// `IS [NOT] NULL`. A missing stat keeps the superfile.
pub fn null_check_skip(
    manifest: &ManifestSnapshot,
    superfiles: &[Arc<SuperfileEntry>],
    column: FieldId,
    want_null: bool,
) -> Vec<bool> {
    let guard = ColumnTypeGuard::new(manifest, column);
    superfiles
        .iter()
        .map(|entry| {
            guard.stats_are_stale(entry)
                || entry
                    .scalar_stats
                    .get(&column)
                    .is_none_or(|agg| null_check_may_match(agg, want_null))
        })
        .collect()
}

/// Whether one column's persisted statistics still describe the values a
/// file's rows read as.
///
/// Min, max and null count are recorded in the type the file was written
/// in, and a retype flips the table's type at once: the read path casts
/// every file that has not been rewritten yet. For those files the
/// recorded statistics describe values the column no longer holds — a
/// string column read as integers is all nulls whatever its null count
/// said, and a truncating cast moves every value off its old bounds.
/// Pruning on them would drop a file that does hold matches, so a file
/// holding the column in another type keeps its place and is read.
pub(crate) struct ColumnTypeGuard {
    /// The column this guard covers.
    id: FieldId,
    /// The table's current view of the column, with no file bound to it
    /// yet. `None` for the injected id column and for a column the table
    /// does not have — neither can go stale.
    current: Option<Resolution>,
    /// Whether the table is converting this column from an older type.
    converting: bool,
}

impl ColumnTypeGuard {
    /// The guard for the column with id `column` in `manifest`'s table.
    pub(crate) fn new(manifest: &ManifestSnapshot, column: FieldId) -> Self {
        let schema = manifest.table_schema();
        let field = schema.fields().iter().find(|f| f.id == column);
        Self {
            id: column,
            current: field.map(|f| Resolution {
                id: f.id,
                name: f.name.clone(),
                data_type: f.data_type.clone(),
                physical: None,
            }),
            converting: field.is_some_and(|f| f.converting_from.is_some()),
        }
    }

    /// The column this guard covers.
    pub(crate) fn id(&self) -> FieldId {
        self.id
    }

    /// Whether the aggregate statistics a manifest part folds over its
    /// files can mix types, which they do for the whole span of a
    /// conversion: some of the part's files hold the old type and some the
    /// new, and one pair of bounds cannot describe both.
    pub(crate) fn aggregates_mix_types(&self) -> bool {
        self.converting
    }

    /// Whether `entry`'s statistics for the column are in a type the table
    /// has since left.
    pub(crate) fn stats_are_stale(&self, entry: &SuperfileEntry) -> bool {
        let Some(current) = self.current.as_ref() else {
            return false;
        };
        match entry.physical_schema.as_ref() {
            Some(physical) => Resolution {
                physical: physical.column_by_id(current.id).cloned(),
                ..current.clone()
            }
            .needs_cast(),
            // A file that records no physical schema was written before
            // field ids: it holds every column in the type the table had
            // then, which for a column under conversion is the old one.
            None => self.converting,
        }
    }
}

/// Whether a column's stats could still match `IS [NOT] NULL`, shared by
/// both prune tiers:
///  - `IS NULL` (`want_null`): keep unless the stats prove zero nulls.
///  - `IS NOT NULL`: keep unless the column is entirely null.
pub(crate) fn null_check_may_match(agg: &ScalarStatsAgg, want_null: bool) -> bool {
    if want_null {
        agg.null_count != Some(0)
    } else {
        !agg_all_null(agg)
    }
}

/// All values are null iff the min stat is null — no non-null value fed it.
fn agg_all_null(agg: &ScalarStatsAgg) -> bool {
    ScalarValue::try_from_array(agg.min.as_ref(), 0)
        .map(|v| v.is_null())
        .unwrap_or(false)
}

/// Whether `entry` *could* contain a row satisfying `pred`, judged
/// only from the superfile's persisted min/max. Conservative: any
/// uncertainty returns `true` (keep).
fn superfile_may_match(entry: &SuperfileEntry, column: FieldId, pred: &ScalarPredicate) -> bool {
    match superfile_minmax(entry, column) {
        None => true,
        Some((min, max)) => scalar_value_may_match(&min, &max, pred.op, &pred.value),
    }
}

/// The superfile's persisted min/max for `column`, or `None` when the
/// column has no stats or the bounds don't decode (caller keeps).
fn superfile_minmax(entry: &SuperfileEntry, column: FieldId) -> Option<(ScalarValue, ScalarValue)> {
    let agg = entry.scalar_stats.get(&column)?;
    match (
        ScalarValue::try_from_array(agg.min.as_ref(), 0),
        ScalarValue::try_from_array(agg.max.as_ref(), 0),
    ) {
        (Ok(min), Ok(max)) => Some((min, max)),
        _ => None,
    }
}

/// Conservative `min`/`max`-vs-`value` comparison core, shared by the
/// superfile tier ([`superfile_may_match`]) and the part tier (the scalar
/// part prune in [`crate::supertable::query::prune`]). Returns `true`
/// (keep) on any uncertainty: null bounds, an un-coercible literal, or
/// otherwise-incomparable values. Never a false prune.
pub(crate) fn scalar_value_may_match(
    min: &ScalarValue,
    max: &ScalarValue,
    op: ScalarOp,
    value: &ScalarValue,
) -> bool {
    if min.is_null() || max.is_null() {
        return true;
    }
    let Some((v, min, max)) = comparable(value, min, max) else {
        return true;
    };
    let (min, max) = (&min, &max);
    let cmp_v_min = v.partial_cmp(min);
    let cmp_v_max = v.partial_cmp(max);
    match op {
        // keep iff min <= v <= max
        ScalarOp::Eq => match (cmp_v_min, cmp_v_max) {
            (Some(lo), Some(hi)) => lo != Ordering::Less && hi != Ordering::Greater,
            _ => true,
        },
        // prune only when the superfile is a single constant == v
        ScalarOp::NotEq => {
            let constant = min.partial_cmp(max) == Some(Ordering::Equal);
            let equals_v = cmp_v_min == Some(Ordering::Equal);
            !(constant && equals_v)
        }
        // keep iff some row could be < v, i.e. min < v
        ScalarOp::Lt => matches!(cmp_v_min, Some(Ordering::Greater) | None),
        // keep iff min <= v
        ScalarOp::LtEq => !matches!(cmp_v_min, Some(Ordering::Less)),
        // keep iff max > v, i.e. v < max
        ScalarOp::Gt => matches!(cmp_v_max, Some(Ordering::Less) | None),
        // keep iff max >= v
        ScalarOp::GtEq => !matches!(cmp_v_max, Some(Ordering::Greater)),
    }
}

/// The literal and the bounds brought into one type for comparison, or
/// `None` when they cannot be compared exactly, in which case the file is
/// kept. Same type: as they are. Same family (the integer, float, decimal,
/// string or binary types): all three promoted to the family's widest
/// type, and only when every promotion is exact — an `Int64` bound above
/// 2^53 does not survive a trip through `Float64`, and comparing it there
/// could prune a file that holds the value. Any other pair — the
/// lexicographic bounds of a string column against a number, say — says
/// nothing about the column's range in the literal's type, so it is never
/// compared.
fn comparable(
    value: &ScalarValue,
    min: &ScalarValue,
    max: &ScalarValue,
) -> Option<(ScalarValue, ScalarValue, ScalarValue)> {
    if value.is_null() {
        return None;
    }
    let (stat_type, literal_type) = (min.data_type(), value.data_type());
    if stat_type == literal_type {
        return Some((value.clone(), min.clone(), max.clone()));
    }
    let target = promotion_target(&stat_type, &literal_type)?;
    Some((
        exact_cast(value, &target)?,
        exact_cast(min, &target)?,
        exact_cast(max, &target)?,
    ))
}

/// Which family a type belongs to, for promotion.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TypeFamily {
    Integer,
    Float,
    Decimal,
    String,
    Binary,
}

fn type_family(data_type: &DataType) -> Option<TypeFamily> {
    Some(match data_type {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => TypeFamily::Integer,
        DataType::Float16 | DataType::Float32 | DataType::Float64 => TypeFamily::Float,
        DataType::Decimal128(..) | DataType::Decimal256(..) => TypeFamily::Decimal,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => TypeFamily::String,
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => TypeFamily::Binary,
        _ => return None,
    })
}

/// The type two values of related types are compared in: the widest of
/// their families. `None` when the families differ or either has none.
fn promotion_target(a: &DataType, b: &DataType) -> Option<DataType> {
    let (fa, fb) = (type_family(a)?, type_family(b)?);
    Some(match (fa, fb) {
        (TypeFamily::String, TypeFamily::String) => DataType::LargeUtf8,
        (TypeFamily::Binary, TypeFamily::Binary) => DataType::LargeBinary,
        (TypeFamily::Float, TypeFamily::Float | TypeFamily::Integer | TypeFamily::Decimal)
        | (TypeFamily::Integer | TypeFamily::Decimal, TypeFamily::Float) => DataType::Float64,
        (TypeFamily::Decimal, TypeFamily::Decimal | TypeFamily::Integer)
        | (TypeFamily::Integer, TypeFamily::Decimal) => {
            let scale = |dt: &DataType| match dt {
                DataType::Decimal128(_, s) | DataType::Decimal256(_, s) => *s,
                _ => 0,
            };
            DataType::Decimal256(DECIMAL256_MAX_PRECISION, scale(a).max(scale(b)))
        }
        (TypeFamily::Integer, TypeFamily::Integer) => DataType::Int64,
        _ => return None,
    })
}

/// `value` in `target`, only when the round trip back to its own type
/// returns it unchanged.
fn exact_cast(value: &ScalarValue, target: &DataType) -> Option<ScalarValue> {
    let promoted = value.cast_to(target).ok().filter(|v| !v.is_null())?;
    let back = promoted.cast_to(&value.data_type()).ok()?;
    (&back == value).then_some(promoted)
}

/// The widest decimal precision, so a promoted decimal comparison loses
/// no digits.
const DECIMAL256_MAX_PRECISION: u8 = 76;

#[cfg(test)]
mod tests {
    use crate::{supertable::schema::FieldId, test_helpers::fid};

    /// Pair each predicate with the id of its column, as the pruner does.
    fn pairs(preds: &[ScalarPredicate]) -> Vec<(Option<FieldId>, &ScalarPredicate)> {
        preds.iter().map(|p| (Some(fid(&p.column)), p)).collect()
    }

    /// A manifest whose table has no column under conversion and no file
    /// holding one in an older type, so every mask below is the pure
    /// statistics answer.
    fn current_types() -> ManifestSnapshot {
        ManifestSnapshot::empty(opts_simple())
    }
    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{ArrayRef, Date32Array, Int64Array, LargeStringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::scalar::ScalarValue;
    use uuid::Uuid;

    use super::*;
    use crate::{
        superfile::{
            builder::{FtsConfig, VectorConfig},
            fts::reader::ColumnLengthStats,
            vector::{distance::Metric, layout::VectorLayout, rerank_codec::RerankCodec},
        },
        supertable::{
            SupertableOptions,
            manifest::{
                FtsSummaryAgg, ManifestSnapshot, ScalarStatsAgg, SuperfileEntry, SuperfileUri,
                VectorSummary, bloom::BloomBuilder,
            },
        },
        test_helpers::default_tokenizer,
    };

    fn opts_simple() -> Arc<SupertableOptions> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]));
        let _tk = default_tokenizer();
        Arc::new(
            SupertableOptions::new(schema, vec![FtsConfig::new("title")], vec![]).expect("opts"),
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
                    rot_seed: 0,
                    metric: Metric::Cosine,
                    rerank_codec: RerankCodec::Fp32,
                    provided_centroids: None,
                }],
            )
            .expect("opts"),
        )
    }

    fn empty_superfile() -> SuperfileEntry {
        let uri = SuperfileUri::new_v4();
        SuperfileEntry {
            physical_schema: None,
            stem: None,
            birth_version: 0,
            superfile_id: Uuid::new_v4(),
            uri,
            n_docs: 0,
            id_min: 0,
            id_max: 0,
            scalar_stats: HashMap::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            vector_layout: VectorLayout::Ivf,
            subsection_offsets: None,
        }
    }

    /// Build a one-column FTS summary with the given indexed terms.
    fn fts_summary_with(column: &str, terms: &[&str]) -> (FieldId, FtsSummaryAgg) {
        let mut bb = BloomBuilder::new();
        for t in terms {
            bb.insert(t.as_bytes());
        }
        let term_range = match (terms.first(), terms.last()) {
            (Some(min), Some(max)) => (min.as_bytes().to_vec(), max.as_bytes().to_vec()),
            _ => (Vec::new(), Vec::new()),
        };
        let summary = FtsSummaryAgg::new_with_params(
            Some(bb.finish()),
            terms.len() as u32,
            term_range,
            ColumnLengthStats::default(),
        );
        (fid(column), summary)
    }

    fn superfile_with_terms(column: &str, terms: &[&str]) -> Arc<SuperfileEntry> {
        let mut e = empty_superfile();
        let (k, v) = fts_summary_with(column, terms);
        e.fts_summary.insert(k, v);
        Arc::new(e)
    }

    fn superfile_with_centroid(column: &str, centroid: Vec<f32>) -> Arc<SuperfileEntry> {
        let mut e = empty_superfile();
        e.vector_summary.insert(
            fid(column),
            VectorSummary {
                centroid,
                cells: Vec::new(),
            },
        );
        Arc::new(e)
    }

    // ---- fts_bloom_skip ----------------------------------------------

    #[test]
    fn bloom_skip_keeps_superfiles_with_any_query_term_in_or_mode() {
        let s_a = superfile_with_terms("title", &["alpha", "beta"]);
        let s_b = superfile_with_terms("title", &["gamma", "delta"]);
        let m = ManifestSnapshot::new_from_superfiles(opts_simple(), vec![s_a, s_b]);
        let mask = fts_bloom_skip(
            &m.superfiles,
            fid("title"),
            &["alpha", "missing"],
            BoolMode::Or,
        );
        // Superfile A has alpha → keep. Superfile B has neither → prune.
        assert_eq!(mask, vec![true, false]);
    }

    #[test]
    fn bloom_skip_requires_all_terms_present_in_and_mode() {
        let s_a = superfile_with_terms("title", &["alpha", "beta"]);
        let s_b = superfile_with_terms("title", &["alpha", "gamma"]);
        let m = ManifestSnapshot::new_from_superfiles(opts_simple(), vec![s_a, s_b]);
        let mask = fts_bloom_skip(
            &m.superfiles,
            fid("title"),
            &["alpha", "beta"],
            BoolMode::And,
        );
        // Superfile A has both. Superfile B is missing 'beta' → prune.
        assert_eq!(mask, vec![true, false]);
    }

    #[test]
    fn bloom_skip_unknown_column_keeps_all() {
        let s = superfile_with_terms("title", &["alpha"]);
        let m = ManifestSnapshot::new_from_superfiles(opts_simple(), vec![s]);
        let mask = fts_bloom_skip(
            &m.superfiles,
            fid("no_such_column"),
            &["alpha"],
            BoolMode::Or,
        );
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn bloom_skip_empty_terms_keeps_all() {
        let s = superfile_with_terms("title", &["alpha"]);
        let m = ManifestSnapshot::new_from_superfiles(opts_simple(), vec![s]);
        let mask = fts_bloom_skip(&m.superfiles, fid("title"), &[], BoolMode::Or);
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn bloom_skip_with_no_superfiles_returns_empty_vec() {
        let m = ManifestSnapshot::new_from_superfiles(opts_simple(), vec![]);
        let mask = fts_bloom_skip(&m.superfiles, fid("title"), &["alpha"], BoolMode::Or);
        assert!(mask.is_empty());
    }

    // ---- fts_prefix_skip ---------------------------------------------

    #[test]
    fn prefix_skip_prunes_superfiles_outside_prefix_range() {
        // Superfile A: terms in ['apple', 'banana'] → prefix "rust"
        //            doesn't overlap.
        // Superfile B: terms in ['python', 'rust']  → prefix "rust"
        //            overlaps the upper end.
        let s_a = superfile_with_terms("title", &["apple", "banana"]);
        let s_b = superfile_with_terms("title", &["python", "rust"]);
        let m = ManifestSnapshot::new_from_superfiles(opts_simple(), vec![s_a, s_b]);
        let mask = fts_prefix_skip(&m.superfiles, fid("title"), b"rust");
        assert_eq!(mask, vec![false, true]);
    }

    #[test]
    fn prefix_skip_keeps_superfiles_with_matching_prefix_inside_range() {
        // Terms ['rusting', 'rusty'] → prefix "rust" overlaps.
        let s = superfile_with_terms("title", &["rusting", "rusty"]);
        let m = ManifestSnapshot::new_from_superfiles(opts_simple(), vec![s]);
        let mask = fts_prefix_skip(&m.superfiles, fid("title"), b"rust");
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn prefix_skip_empty_prefix_keeps_all() {
        let s = superfile_with_terms("title", &["alpha"]);
        let m = ManifestSnapshot::new_from_superfiles(opts_simple(), vec![s]);
        let mask = fts_prefix_skip(&m.superfiles, fid("title"), b"");
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn prefix_skip_unknown_column_keeps_all() {
        let s = superfile_with_terms("title", &["alpha"]);
        let m = ManifestSnapshot::new_from_superfiles(opts_simple(), vec![s]);
        let mask = fts_prefix_skip(&m.superfiles, fid("no_such_column"), b"alp");
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn prefix_skip_zero_term_superfile_pruned() {
        // Empty term_range = no terms indexed. Prefix can't match.
        let s = Arc::new(empty_superfile());
        let m = ManifestSnapshot::new_from_superfiles(opts_simple(), vec![s]);
        let mask = fts_prefix_skip(&m.superfiles, fid("title"), b"rust");
        // No FTS summary on the superfile → keep (column-missing
        // path). Sanity: this is the "unknown column" path, not
        // the "0-term FTS column" path.
        assert_eq!(mask, vec![true]);
    }

    // ---- vector_centroid_skip + ordering ------------------------------

    #[test]
    fn vector_centroid_skip_v1_keeps_all_superfiles() {
        let s_a = superfile_with_centroid("emb", vec![0.0; 16]);
        let s_b = superfile_with_centroid("emb", vec![10.0; 16]);
        let m = ManifestSnapshot::new_from_superfiles(opts_with_vector(), vec![s_a, s_b]);
        let q = vec![0.0f32; 16];
        let mask = vector_centroid_skip(&m, "emb", &q);
        assert_eq!(mask, vec![true, true]);
    }

    #[test]
    fn superfiles_sorted_by_centroid_distance_orders_by_metric() {
        // L2-sq metric on simple 1-hot centroids.
        let opts = opts_with_vector();
        let near = superfile_with_centroid("emb", {
            let mut v = vec![0.0f32; 16];
            v[0] = 1.0;
            v
        });
        let far = superfile_with_centroid("emb", {
            let mut v = vec![0.0f32; 16];
            v[7] = 1.0;
            v
        });
        let m = ManifestSnapshot::new_from_superfiles(opts, vec![far.clone(), near.clone()]);
        let q = {
            let mut v = vec![0.0f32; 16];
            v[0] = 1.0;
            v
        };
        let order = superfiles_sorted_by_centroid_distance(&m, fid("emb"), &q, Metric::L2Sq);
        // `near` (idx 1) should come before `far` (idx 0).
        assert_eq!(order, vec![1, 0]);
    }

    #[test]
    fn superfiles_sorted_by_centroid_distance_pushes_missing_summary_to_end() {
        let with_v = superfile_with_centroid("emb", vec![1.0f32; 16]);
        let without_v = Arc::new(empty_superfile());
        let m = ManifestSnapshot::new_from_superfiles(opts_with_vector(), vec![without_v, with_v]);
        let q = vec![1.0f32; 16];
        let order = superfiles_sorted_by_centroid_distance(&m, fid("emb"), &q, Metric::L2Sq);
        // Index 1 (has summary) sorted before index 0 (missing).
        assert_eq!(order, vec![1, 0]);
    }

    // ---- scalar_skip -------------------------------------------------

    fn seg_with_int_stats(col: &str, min: i64, max: i64) -> Arc<SuperfileEntry> {
        let mut e = empty_superfile();
        let mn: ArrayRef = Arc::new(Int64Array::from(vec![min]));
        let mx: ArrayRef = Arc::new(Int64Array::from(vec![max]));
        e.scalar_stats
            .insert(fid(col), ScalarStatsAgg::from_min_max(mn, mx));
        Arc::new(e)
    }

    fn seg_with_str_stats(col: &str, min: &str, max: &str) -> Arc<SuperfileEntry> {
        let mut e = empty_superfile();
        let mn: ArrayRef = Arc::new(LargeStringArray::from(vec![min]));
        let mx: ArrayRef = Arc::new(LargeStringArray::from(vec![max]));
        e.scalar_stats
            .insert(fid(col), ScalarStatsAgg::from_min_max(mn, mx));
        Arc::new(e)
    }

    // Date32 bounds stored as days-since-epoch, the ClickBench `EventDate`
    // shape now that temporal columns carry manifest min/max.
    fn seg_with_date_stats(col: &str, min: i32, max: i32) -> Arc<SuperfileEntry> {
        let mut e = empty_superfile();
        let mn: ArrayRef = Arc::new(Date32Array::from(vec![min]));
        let mx: ArrayRef = Arc::new(Date32Array::from(vec![max]));
        e.scalar_stats
            .insert(fid(col), ScalarStatsAgg::from_min_max(mn, mx));
        Arc::new(e)
    }

    fn pred(column: &str, op: ScalarOp, value: ScalarValue) -> ScalarPredicate {
        ScalarPredicate {
            column: column.to_string(),
            op,
            value,
        }
    }

    #[test]
    fn scalar_skip_empty_predicates_keeps_all() {
        let segs = vec![
            seg_with_int_stats("x", 0, 10),
            seg_with_int_stats("x", 100, 110),
        ];
        assert_eq!(
            scalar_skip(&current_types(), &segs, &pairs(&[])),
            vec![true, true]
        );
    }

    #[test]
    fn scalar_value_set_skip_keeps_superfiles_holding_any_listed_value() {
        let segs = vec![
            seg_with_int_stats("x", 0, 10),
            seg_with_int_stats("x", 100, 110),
            seg_with_int_stats("x", 200, 210),
        ];
        let i = |n| ScalarValue::Int64(Some(n));
        // IN (5, 205) → A's [0,10] and C's [200,210], not B.
        assert_eq!(
            scalar_value_set_skip(&current_types(), &segs, fid("x"), &[i(5), i(205)]),
            vec![true, false, true]
        );
        // IN (50) → matches no range.
        assert_eq!(
            scalar_value_set_skip(&current_types(), &segs, fid("x"), &[i(50)]),
            vec![false, false, false]
        );
        // Empty list and unknown column both keep all (conservative).
        assert_eq!(
            scalar_value_set_skip(&current_types(), &segs, fid("x"), &[]),
            vec![true, true, true]
        );
        assert_eq!(
            scalar_value_set_skip(&current_types(), &segs, fid("missing"), &[i(5)]),
            vec![true, true, true]
        );
    }

    #[test]
    fn null_check_may_match_covers_both_predicates() {
        let arr = |v: Option<i64>| Arc::new(Int64Array::from(vec![v])) as ArrayRef;
        let agg = |min: Option<i64>, null_count: Option<u64>| ScalarStatsAgg {
            min: arr(min),
            max: arr(min),
            null_count,
            sum: None,
            hll: None,
            value_counts: None,
        };

        // No nulls: IS NULL drops, IS NOT NULL keeps.
        let no_null = agg(Some(5), Some(0));
        assert!(!null_check_may_match(&no_null, true));
        assert!(null_check_may_match(&no_null, false));

        // All null (min is null): IS NULL keeps, IS NOT NULL drops.
        let all_null = agg(None, Some(10));
        assert!(null_check_may_match(&all_null, true));
        assert!(!null_check_may_match(&all_null, false));

        // Some nulls (min present): both keep.
        let mixed = agg(Some(5), Some(2));
        assert!(null_check_may_match(&mixed, true));
        assert!(null_check_may_match(&mixed, false));

        // Unknown null count (None): can't prove zero nulls, both keep.
        let unknown = agg(Some(5), None);
        assert!(null_check_may_match(&unknown, true));
        assert!(null_check_may_match(&unknown, false));
    }

    #[test]
    fn null_check_skip_keeps_on_missing_stat() {
        let segs = vec![seg_with_int_stats("x", 0, 10)];
        // Column not in stats → conservative keep for either predicate.
        assert_eq!(
            null_check_skip(&current_types(), &segs, fid("missing"), true),
            vec![true]
        );
        assert_eq!(
            null_check_skip(&current_types(), &segs, fid("missing"), false),
            vec![true]
        );
    }

    #[test]
    fn scalar_skip_eq_prunes_superfiles_whose_range_excludes_value() {
        let segs = vec![
            seg_with_int_stats("x", 0, 10),
            seg_with_int_stats("x", 100, 110),
        ];
        // x = 5 → only A's [0,10] can contain it.
        let mask = scalar_skip(
            &current_types(),
            &segs,
            &pairs(&[pred("x", ScalarOp::Eq, ScalarValue::Int64(Some(5)))]),
        );
        assert_eq!(mask, vec![true, false]);
        // x = 105 → only B's [100,110].
        let mask = scalar_skip(
            &current_types(),
            &segs,
            &pairs(&[pred("x", ScalarOp::Eq, ScalarValue::Int64(Some(105)))]),
        );
        assert_eq!(mask, vec![false, true]);
        // Range boundary is inclusive.
        let mask = scalar_skip(
            &current_types(),
            &segs,
            &pairs(&[pred("x", ScalarOp::Eq, ScalarValue::Int64(Some(10)))]),
        );
        assert_eq!(mask, vec![true, false]);
    }

    #[test]
    fn scalar_skip_range_ops_prune_by_min_or_max() {
        let segs = vec![
            seg_with_int_stats("x", 0, 10),
            seg_with_int_stats("x", 100, 110),
        ];
        // x > 50 → A.max=10 can't; B kept.
        assert_eq!(
            scalar_skip(
                &current_types(),
                &segs,
                &pairs(&[pred("x", ScalarOp::Gt, ScalarValue::Int64(Some(50)))])
            ),
            vec![false, true]
        );
        // x < 50 → A.min=0 ok; B.min=100 can't.
        assert_eq!(
            scalar_skip(
                &current_types(),
                &segs,
                &pairs(&[pred("x", ScalarOp::Lt, ScalarValue::Int64(Some(50)))])
            ),
            vec![true, false]
        );
        // x >= 110 → A can't (max 10); B can (max 110).
        assert_eq!(
            scalar_skip(
                &current_types(),
                &segs,
                &pairs(&[pred("x", ScalarOp::GtEq, ScalarValue::Int64(Some(110)))])
            ),
            vec![false, true]
        );
        // x <= 0 → A can (min 0); B can't (min 100).
        assert_eq!(
            scalar_skip(
                &current_types(),
                &segs,
                &pairs(&[pred("x", ScalarOp::LtEq, ScalarValue::Int64(Some(0)))])
            ),
            vec![true, false]
        );
    }

    #[test]
    fn scalar_skip_range_ops_prune_temporal_columns() {
        // Two date superfiles with disjoint ranges (as ClickBench's
        // time-ordered `hits` produces). Before temporal min/max landed these
        // carried no bounds and neither could ever be pruned.
        let d = |day| ScalarValue::Date32(Some(day));
        let segs = vec![
            seg_with_date_stats("EventDate", 100, 200),
            seg_with_date_stats("EventDate", 500, 600),
        ];
        // EventDate > 300 → A.max=200 can't; B kept.
        assert_eq!(
            scalar_skip(
                &current_types(),
                &segs,
                &pairs(&[pred("EventDate", ScalarOp::Gt, d(300))])
            ),
            vec![false, true]
        );
        // EventDate < 300 → A.min=100 ok; B.min=500 can't.
        assert_eq!(
            scalar_skip(
                &current_types(),
                &segs,
                &pairs(&[pred("EventDate", ScalarOp::Lt, d(300))])
            ),
            vec![true, false]
        );
        // BETWEEN 250 AND 450 (>=250 AND <=450) → both disjoint ranges pruned.
        assert_eq!(
            scalar_skip(
                &current_types(),
                &segs,
                &pairs(&[
                    pred("EventDate", ScalarOp::GtEq, d(250)),
                    pred("EventDate", ScalarOp::LtEq, d(450)),
                ])
            ),
            vec![false, false]
        );
    }

    #[test]
    fn scalar_skip_conjunction_prunes_when_any_predicate_excludes() {
        // A=[0,3], B=[6,7]; WHERE x >= 5 AND x <= 8.
        let segs = vec![seg_with_int_stats("x", 0, 3), seg_with_int_stats("x", 6, 7)];
        let preds = [
            pred("x", ScalarOp::GtEq, ScalarValue::Int64(Some(5))),
            pred("x", ScalarOp::LtEq, ScalarValue::Int64(Some(8))),
        ];
        // A: max=3 < 5 → the >=5 conjunct prunes it. B kept.
        assert_eq!(
            scalar_skip(&current_types(), &segs, &pairs(&preds)),
            vec![false, true]
        );
    }

    #[test]
    fn scalar_skip_unknown_column_keeps_all() {
        let segs = vec![seg_with_int_stats("x", 0, 10)];
        let mask = scalar_skip(
            &current_types(),
            &segs,
            &pairs(&[pred("not_a_col", ScalarOp::Eq, ScalarValue::Int64(Some(5)))]),
        );
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn scalar_skip_coerces_utf8_literal_against_largeutf8_stats() {
        // Stats stored as LargeUtf8; predicate literal is Utf8.
        let segs = vec![
            seg_with_str_stats("name", "apple", "mango"),
            seg_with_str_stats("name", "tango", "zulu"),
        ];
        // name = 'banana' → within A's [apple, mango], outside B's.
        let mask = scalar_skip(
            &current_types(),
            &segs,
            &pairs(&[pred(
                "name",
                ScalarOp::Eq,
                ScalarValue::Utf8(Some("banana".into())),
            )]),
        );
        assert_eq!(mask, vec![true, false]);
    }

    #[test]
    fn scalar_skip_null_stats_keeps_superfile() {
        let mut e = empty_superfile();
        let mn: ArrayRef = Arc::new(Int64Array::from(vec![None::<i64>]));
        let mx: ArrayRef = Arc::new(Int64Array::from(vec![None::<i64>]));
        e.scalar_stats
            .insert(fid("x"), ScalarStatsAgg::from_min_max(mn, mx));
        let segs = vec![Arc::new(e)];
        let mask = scalar_skip(
            &current_types(),
            &segs,
            &pairs(&[pred("x", ScalarOp::Eq, ScalarValue::Int64(Some(5)))]),
        );
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn scalar_skip_not_eq_prunes_only_constant_superfile() {
        let segs = vec![seg_with_int_stats("x", 5, 5), seg_with_int_stats("x", 5, 9)];
        // x != 5 → constant all-5 superfile matches nothing → prune;
        // the ranged superfile is kept.
        let mask = scalar_skip(
            &current_types(),
            &segs,
            &pairs(&[pred("x", ScalarOp::NotEq, ScalarValue::Int64(Some(5)))]),
        );
        assert_eq!(mask, vec![false, true]);
    }
    /// Bounds and literal in the same numeric family compare in the
    /// family's widest type, exactly: a fractional literal against integer
    /// bounds prunes when no integer in the bounds can satisfy it and keeps
    /// the file when one can, which a cast of the literal toward the
    /// integer type got wrong by truncating.
    #[test]
    fn a_float_literal_against_integer_bounds_compares_exactly() {
        let (min, max) = (ScalarValue::Int64(Some(5)), ScalarValue::Int64(Some(5)));
        let lit = ScalarValue::Float64(Some(5.5));
        assert!(!scalar_value_may_match(&min, &max, ScalarOp::Eq, &lit));
        assert!(scalar_value_may_match(&min, &max, ScalarOp::NotEq, &lit));
        assert!(
            scalar_value_may_match(&min, &max, ScalarOp::Lt, &lit),
            "5 < 5.5"
        );
        assert!(scalar_value_may_match(&min, &max, ScalarOp::LtEq, &lit));
        assert!(
            !scalar_value_may_match(&min, &max, ScalarOp::Gt, &lit),
            "5 > 5.5 is false"
        );
        assert!(!scalar_value_may_match(&min, &max, ScalarOp::GtEq, &lit));

        // A narrower integer type on the file's side promotes the same way.
        let (min32, max32) = (ScalarValue::Int32(Some(1)), ScalarValue::Int32(Some(9)));
        assert!(scalar_value_may_match(
            &min32,
            &max32,
            ScalarOp::Eq,
            &ScalarValue::Int64(Some(4))
        ));
        assert!(!scalar_value_may_match(
            &min32,
            &max32,
            ScalarOp::Eq,
            &ScalarValue::Int64(Some(40))
        ));
    }

    /// A promotion that is not exact keeps the file: an `Int64` bound above
    /// 2^53 does not survive `Float64`, so it is never compared there.
    #[test]
    fn an_inexact_promotion_keeps_the_file() {
        let big = (1i64 << 53) + 1;
        let (min, max) = (ScalarValue::Int64(Some(big)), ScalarValue::Int64(Some(big)));
        let lit = ScalarValue::Float64(Some(1.0));
        assert!(scalar_value_may_match(&min, &max, ScalarOp::Eq, &lit));
        assert!(scalar_value_may_match(&min, &max, ScalarOp::Lt, &lit));
    }

    /// Bounds of another type family say nothing about the literal's range:
    /// the file is kept whatever the operator. Within the string family a
    /// narrow literal still compares against wide bounds.
    #[test]
    fn a_type_family_mismatch_keeps_the_file_and_string_widths_still_compare() {
        let (min, max) = (
            ScalarValue::LargeUtf8(Some("apple".into())),
            ScalarValue::LargeUtf8(Some("pear".into())),
        );
        for op in [ScalarOp::Eq, ScalarOp::Lt, ScalarOp::Gt, ScalarOp::NotEq] {
            assert!(scalar_value_may_match(
                &min,
                &max,
                op,
                &ScalarValue::Int64(Some(7))
            ));
        }
        assert!(scalar_value_may_match(
            &min,
            &max,
            ScalarOp::Eq,
            &ScalarValue::Utf8(Some("kiwi".into()))
        ));
        assert!(!scalar_value_may_match(
            &min,
            &max,
            ScalarOp::Eq,
            &ScalarValue::Utf8(Some("zebra".into()))
        ));
        let (imin, imax) = (ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(2)));
        assert!(scalar_value_may_match(
            &imin,
            &imax,
            ScalarOp::Eq,
            &ScalarValue::Utf8(Some("1".into()))
        ));
        assert!(!scalar_value_may_match(
            &imin,
            &imax,
            ScalarOp::Eq,
            &ScalarValue::Int64(Some(3))
        ));
    }
}
