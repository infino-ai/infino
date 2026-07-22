// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `Manifest` — the top-tier of the two-tier hierarchical manifest.
//! A small JSON document (~MB even at 1M superfiles) that references one or
//! more [`ManifestPart`] files by URI + content hash, carries the
//! table-level metadata (schema, column configs, partition strategy), and
//! surfaces per-part aggregate skip summaries that drive list-level pruning.
//!
//! Format: JSON, **pretty-printed and deterministically
//! ordered** so byte-equal logical content produces byte-equal
//! files — the property the content-addressing optimization
//! rides on (a list whose contents match a prior version's
//! gets the same URI and isn't re-PUT).
//!
//! [`ManifestPart`]: super::part::ManifestPart

use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
};

use arrow::compute::concat;
use arrow_array::{
    Array, ArrayRef, Int64Array, LargeStringArray, RecordBatch, StringArray, UInt64Array,
};
use arrow_schema::{DataType, Schema};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use datafusion::scalar::ScalarValue;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::supertable::manifest::{
    add_sum_arrays,
    bloom::Bloom,
    column_hll, column_min_max, column_sum,
    encoding::{
        DecodeError, EncodeError, decode_cluster_centroids, decode_length1_array,
        decode_value_counts, encode_cluster_centroids, encode_length1_array, encode_value_counts,
    },
    hll::HllSketch,
    merge_min_max_arrays,
    part::{BLAKE3_DIGEST_BYTES, BLAKE3_HEX_LEN, ContentHash, PartId},
    term_range::prefix_overlaps_range,
};

/// Wire format version for the manifest list.
///
/// Major must match at decode time; minor differences are
/// accepted (deny-unknown-major / allow-unknown-minor — same
/// shape as the [`super::part::FORMAT_VERSION`] policy for
/// manifest parts).
pub const FORMAT_VERSION: &str = "1.0";
/// Reserved list-aggregate key carrying each part's superfile birth range.
pub(crate) const BIRTH_VERSION_AGGREGATE_COLUMN: &str = "__infino_birth_version";
/// Maximum distinct values retained as an exact per-column frequency table.
/// Columns crossing the cap carry no table, so high-cardinality data has
/// bounded manifest cost.
const MAX_EXACT_VALUE_COUNTS: usize = 256;

/// Cells probed per query — exactly one, the grid-nearest. The p=1 read
/// shape is the design point (one cell fetch, ~fine_nprobe × 2 MiB runs);
/// recall past its coverage ceiling is bought with boundary replication at
/// drain time, never by widening the probe set.
const DEFAULT_CELL_NPROBE_MIN: usize = 1;
/// Hard cap on cells probed per query. Equal to the floor: adaptive
/// widening is off — coverage is a drain-side (replication) concern.
const DEFAULT_CELL_NPROBE_MAX: usize = 1;
/// Margin on the distance-ratio probe threshold (`τ = d*·(1+slack)`).
/// Inert while `nprobe_max == nprobe_min`; acts only when a table
/// explicitly widens its persisted routing.
const DEFAULT_CELL_SLACK: f32 = 1.0;

// ---------- Public in-memory shapes ----------

/// Top-level manifest list. The wire format is the JSON
/// produced by [`encode`]; this struct is the in-memory
/// shape callers (the supertable's load/refresh path and
/// the writer's commit path) consume.
#[derive(Debug, Clone)]
pub struct Manifest {
    /// `[FORMAT_VERSION]` constant at encode time; rejected
    /// at decode time if major mismatch.
    pub format_version: String,
    /// Monotonically-increasing version of this supertable.
    /// `0` is the initial empty manifest; each successful
    /// commit increments by 1.
    pub manifest_id: u64,
    /// Content hash of the canonicalized `SupertableOptions`
    /// — guards against schema/column-config drift across
    /// process restarts.
    pub options_hash: ContentHash,
    /// Arrow-IPC bytes of the supertable's user schema.
    /// Stored as bytes so we don't depend on Arrow's
    /// JSON-schema serializer (which doesn't round-trip
    /// `FixedSizeList<Float32>` correctly in 0.x).
    pub schema: Vec<u8>,
    /// Name of the user-supplied id column.
    pub id_column: String,
    /// Per-FTS-column configuration. Stable across the
    /// supertable's lifetime — schema change requires
    /// external compaction.
    pub fts_columns: Vec<FtsColumnInfo>,
    /// Per-vector-column configuration.
    pub vector_columns: Vec<VectorColumnInfo>,
    /// How superfiles are grouped into manifest parts. Locked
    /// at supertable creation; see
    /// [`crate::supertable::options::SupertableOptions::effective_partition_strategy`]
    /// for how the field is resolved.
    pub partition_strategy: PartitionStrategy,
    /// Object-storage prefix for the hidden vector-index sibling
    /// supertable (e.g. `_infino_<uuid>_vector_index/`). Set at
    /// create when vector columns are configured; immutable.
    pub vector_index_storage_prefix: Option<String>,
    /// Global vector cell-index coordination state, owned by THIS (the user)
    /// table. Source of truth for the immutable cell grids trained from the
    /// first committed batch: `grid` at `hidden_cell_count` cells (read
    /// verbatim by the drain, which writes it as the hidden sibling's
    /// `PartitionStrategy::VectorCell` with live per-cell counts on top) and
    /// the optional finer `user_grid` at `user_cell_count` cells (user-side
    /// packing + pre-drain routing only). `None` until the first commit with
    /// vectors, and on tables without vector columns.
    pub global_vector_index: Option<GlobalVectorIndex>,
    /// Drained user commit-versions — **hidden manifest only** (empty on the
    /// user manifest). The hidden-index drain advances this as it consumes user
    /// commits into cells; see [`DrainedVersionRanges`].
    pub drained_ranges: DrainedVersionRanges,
    /// The deleted-`_id` set's encoded bytes carried INLINE in the list —
    /// so the set rides the pointer payload and consulting it never costs
    /// a GET.
    pub deleted_user_ids_inline: Option<Vec<u8>>,
    /// Slow-CAS section: content-addressed blob holding this table's
    /// superfile entries verbatim (the drain-owned routing/centroid state).
    /// Present only on the hidden vector-index table, stamped exclusively by
    /// drain / hidden compaction; the user table's slow section stays `None`
    /// (the inverse of the hidden manifest — same format, no special paths).
    /// `ManifestSnapshot::update` deliberately does NOT carry these into its
    /// successor list: any membership change invalidates the blob, and only
    /// maintenance republishes it. Every consumer keys on ref presence,
    /// never on table kind.
    pub slow_vector_state_uri: Option<String>,
    pub slow_vector_state_content_hash: Option<ContentHash>,
    /// Centroid-section sibling: every visible entry's fp32 fine centroids
    /// concatenated contiguously in `(entry, column, cell)` order. The
    /// stripped-summary admit rescore hydrates it in one fetch on the
    /// first cold query (NVMe-spilled, page-cache-evictable) instead of
    /// fanning one block GET per shortlisted cell per query — measured 53
    /// then 43 GETs on the first two post-drain cold queries at 1M/256.
    /// Stamped and cleared together with the full ref; absent on older
    /// manifests (consumers fall back to per-superfile centroid reads).
    pub slow_vector_state_centroids: Option<RoutingRef>,
    /// Entries — one per manifest part referenced by this
    /// list. Ordered by insertion order (commit order); the
    /// list-level pruner walks them in order.
    pub parts: Vec<ManifestPartEntry>,
    /// Per-superfile tombstone-sidecar version: `superfile_id →` the
    /// `manifest_id` of the commit whose tombstone phase last changed
    /// that superfile's sidecar. The mutation pipeline stamps this map
    /// (a fresh list + pointer CAS) right after its sidecar CAS-PUTs
    /// land, so a reader that refreshes the manifest learns *which*
    /// sidecars changed without polling each one — cross-process
    /// delete visibility is bounded by the read-consistency window,
    /// not a sidecar cache TTL. The map is authoritative for sidecar
    /// *existence* too: a superfile absent from it has no sidecar, so
    /// readers skip the storage GET entirely. Entries are dropped when
    /// their superfile leaves the manifest (compaction/removal).
    pub tombstone_seqs: BTreeMap<Uuid, u64>,
}

/// Content-addressed reference to a sibling object of a manifest
/// artifact: a manifest part's routing form ([`ManifestPartEntry::routing`],
/// same entries with each vector summary's cluster blocks encoded as
/// counts + 1-bit admit slab, no fp32 payload) or the slow-CAS centroid
/// section ([`Manifest::slow_vector_state_centroids`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingRef {
    pub uri: String,
    pub content_hash: ContentHash,
}

/// Global vector cell-index state owned by the user-table manifest (see
/// [`Manifest::global_vector_index`]). Distinct from
/// [`PartitionStrategy`]: it describes the *global grid the table's vectors are
/// indexed into*, not how this table partitions its own superfiles (user
/// superfiles are append-only and span all cells).
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalVectorIndex {
    /// Vector column this grid indexes.
    pub column: String,
    /// Immutable cell centroids trained at first commit at
    /// `hidden_cell_count` cells. This is the grid the drain reads verbatim
    /// to build the hidden cell index. When `user_grid` is absent it also
    /// serves the user side (packing + pre-drain routing).
    pub grid: super::ClusterCentroids,
    /// Optional finer grid trained at the same first commit at
    /// `user_cell_count` cells. Used only on the user side: cell-packing
    /// user superfiles at commit and routing the pre-drain query. The drain
    /// never reads it. `None` on tables created before it existed (and when
    /// `user_cell_count == hidden_cell_count`) — consumers fall back to
    /// `grid` via [`Self::into_user_grid`].
    pub user_grid: Option<super::ClusterCentroids>,
}

impl GlobalVectorIndex {
    /// The grid the user side (commit packing + pre-drain routing) uses:
    /// `user_grid` when trained, else `grid`. Cell tags stamped into user
    /// superfiles and the query's cell ranking must come from this one grid.
    pub fn into_user_grid(self) -> super::ClusterCentroids {
        self.user_grid.unwrap_or(self.grid)
    }

    /// Borrowing form of [`Self::into_user_grid`] for the query path.
    ///
    /// Callers that only need to score/rank must use this (or
    /// [`crate::supertable::manifest::ManifestSnapshot::global_vector_index`])
    /// instead of cloning the index: [`super::ClusterCentroids`]'s transposed
    /// SIMD cache is per-instance, and a clone that drops it forces a full
    /// scalar transpose rebuild on the next scan.
    pub fn user_grid(&self) -> &super::ClusterCentroids {
        self.user_grid.as_ref().unwrap_or(&self.grid)
    }
}

/// Normalized set of drained user commit-versions, stored **only on the hidden
/// manifest**. Tracks which user `manifest_id`s the hidden-index drain has
/// consumed into cells, so repeated/incremental drains skip already-drained
/// commits. Intervals are inclusive `[lo, hi]`, sorted, disjoint, and
/// adjacent-merged.
///
/// The maximal contiguous drained prefix from genesis is just the first
/// interval; out-of-order completions from **parallel drainers** appear as
/// extra intervals that merge as the gaps between them fill. Interval count is
/// bounded by in-flight drain concurrency — not corpus size — so it stays tiny.
/// Empty on the user manifest (which never drains).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DrainedVersionRanges {
    ranges: Vec<(u64, u64)>,
}

impl DrainedVersionRanges {
    /// Rebuild from persisted intervals (normalizes defensively).
    ///
    /// Rejects inverted intervals (`lo > hi`) instead of admitting them:
    /// this decodes remote manifest state, and a malformed interval would
    /// make `contains`/`covers` lie in release builds — versions could be
    /// skipped or re-drained. Corruption fails the manifest load loudly.
    pub fn from_intervals(intervals: Vec<(u64, u64)>) -> Result<Self, ListParseError> {
        let mut s = Self { ranges: Vec::new() };
        for (lo, hi) in intervals {
            if lo > hi {
                return Err(ListParseError::BadFieldValue(
                    "drained_ranges",
                    format!("inverted interval [{lo}, {hi}]"),
                ));
            }
            s.insert_range(lo, hi);
        }
        Ok(s)
    }

    /// The normalized intervals, for serialization.
    pub fn intervals(&self) -> &[(u64, u64)] {
        &self.ranges
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Is `version` covered by a drained interval?
    pub fn contains(&self, version: u64) -> bool {
        self.ranges
            .iter()
            .any(|&(lo, hi)| lo <= version && version <= hi)
    }

    /// Is every version in inclusive `[lo, hi]` covered by one drained run?
    pub fn covers(&self, lo: u64, hi: u64) -> bool {
        self.ranges
            .iter()
            .any(|&(drained_lo, drained_hi)| drained_lo <= lo && hi <= drained_hi)
    }

    /// End of the maximal contiguous drained run anchored at genesis (0), or
    /// `None` if version 0 isn't covered (no genesis-anchored prefix yet). The
    /// drain uses this to extend the prefix over **vacuous version gaps** —
    /// commit-versions that added no superfile (deletes, compaction outputs that
    /// carry an older birth_version, empty commits). Those versions are never
    /// reused, so folding them into the prefix keeps the set from fragmenting
    /// permanently around every superfile-less commit.
    pub fn prefix_end(&self) -> Option<u64> {
        match self.ranges.first() {
            Some(&(0, hi)) => Some(hi),
            _ => None,
        }
    }

    /// Mark a single version drained (merging into the normalized set).
    pub fn insert(&mut self, version: u64) {
        self.insert_range(version, version);
    }

    /// Mark an inclusive `[lo, hi]` interval drained, keeping the set
    /// normalized (sorted, disjoint, adjacent-merged). Adjacent means
    /// gap-free: `[1,3]` + `[4,6]` → `[1,6]` (version 4 = 3+1, no hole),
    /// whereas `[1,3]` + `[5,6]` stays split (version 4 is undrained).
    pub fn insert_range(&mut self, lo: u64, hi: u64) {
        debug_assert!(lo <= hi);
        self.ranges.push((lo, hi));
        self.ranges.sort_unstable_by_key(|&(l, _)| l);
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(self.ranges.len());
        for &(l, h) in &self.ranges {
            match merged.last_mut() {
                Some(last) if l <= last.1.saturating_add(1) => last.1 = last.1.max(h),
                _ => merged.push((l, h)),
            }
        }
        self.ranges = merged;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FtsColumnInfo {
    pub column: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorColumnInfo {
    pub column: String,
    pub dim: usize,
    pub n_cent: usize,
    pub rot_seed: u64,
    /// `"cosine"`, `"l2sq"`, or `"negdot"` — matches the
    /// `VectorConfig::metric` shape.
    pub metric: String,
}

/// Adaptive cell-probe tuning for the hidden VectorCell index.
/// Calibrated per table; persisted in the manifest list.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellRoutingParams {
    /// Nearest cells always probed (recall floor).
    pub nprobe_min: usize,
    /// Hard cap on cells probed per query (GET budget).
    pub nprobe_max: usize,
    /// Margin on the distance-ratio probe threshold (`τ = d*·(1+slack)`).
    pub slack: f32,
    /// Floor on fine IVF centroids probed per selected hidden cell. The
    /// actual probe is `max(this, floor(pct × the cell's fine-cluster count))`.
    pub fine_nprobe: usize,
}

impl Default for CellRoutingParams {
    fn default() -> Self {
        Self {
            nprobe_min: DEFAULT_CELL_NPROBE_MIN,
            nprobe_max: DEFAULT_CELL_NPROBE_MAX,
            slack: DEFAULT_CELL_SLACK,
            fine_nprobe: crate::config::global().vector.fine_nprobe_floor,
        }
    }
}

/// How superfiles are routed into manifest parts. Stamped into
/// the list on first commit; immutable thereafter (changes
/// require external compaction).
#[derive(Debug, Clone, PartialEq)]
pub enum PartitionStrategy {
    TimeRange {
        column: String,
        granularity_secs: i64,
    },
    Hash {
        column: String,
        n_buckets: u32,
    },
    /// Boundaries are Arrow-IPC bytes of length-1 RecordBatch
    /// values (one bytes per boundary), keeping
    /// `ManifestSettings` DataFusion-free.
    ColumnRange {
        column: String,
        boundaries: Vec<Vec<u8>>,
    },
    /// Partition by vector distance to fixed global centroids.
    /// Each row is assigned to the cell whose centroid is nearest.
    VectorCell {
        column: String,
        clusters: super::ClusterCentroids,
        routing: CellRoutingParams,
    },
    IngestionTime {
        granularity_secs: i64,
    },
}

#[derive(Debug, Clone)]
pub struct ManifestPartEntry {
    pub part_id: PartId,
    /// Storage URI for the part. Typically
    /// `manifest-parts/part-<hash>.avro.zst`. Content-addressed so
    /// two identical lists share part files.
    pub uri: String,
    pub n_superfiles: u64,
    pub size_bytes_compressed: u64,
    pub size_bytes_uncompressed: u64,
    pub content_hash: ContentHash,
    /// Routing-only sibling of this part (same entries, vector summaries
    /// as counts + 1-bit admit slab, no fp32). Consumer opens with
    /// `summary_centroids_from_superfiles` load it instead of `uri` —
    /// the user table's fp32 fine centroids stay in storage. Absent on
    /// parts written before the sibling existed (loads fall back to the
    /// full part).
    pub routing: Option<RoutingRef>,
    /// Aggregate id range across this part's superfiles. `i128`
    /// matches the supertable-injected `_id` column type
    /// (`Decimal128(38, 0)`); signed-int comparison gives
    /// time-ordered skip-pruning because the high bit stays
    /// 0 for any plausible current-era timestamp.
    pub id_range: (i128, i128),
    /// Per-scalar-column aggregate min/max across all
    /// superfiles in this part. An empty map is interpreted as
    /// "always-keep" by the list-level pruner.
    pub scalar_stats_agg: HashMap<String, ScalarStatsAgg>,
    /// Per-FTS-column aggregate bloom-union + range-union.
    /// Empty → always-keep.
    pub fts_summary_agg: BTreeMap<String, FtsSummaryAgg>,
}

impl ManifestPartEntry {
    /// Inclusive birth-version range for list-level drain pruning.
    pub(crate) fn birth_version_range(&self) -> Option<(u64, u64)> {
        let aggregate = self.scalar_stats_agg.get(BIRTH_VERSION_AGGREGATE_COLUMN)?;
        let min = aggregate.min.as_any().downcast_ref::<UInt64Array>()?;
        let max = aggregate.max.as_any().downcast_ref::<UInt64Array>()?;
        Some((min.value(0), max.value(0)))
    }
}

/// Aggregate scalar stats across a part's superfiles. Min/max and exact sums
/// use Arrow arrays; low-cardinality columns may additionally carry a capped
/// exact value-frequency table. This is the same in-memory shape the
/// per-superfile `SuperfileEntry.scalar_stats` map uses. Stats are decoded once
/// when the manifest list is loaded, so the list-level scalar prune
/// ([`crate::supertable::query::prune`]) compares against them
/// without a per-query Arrow-IPC decode. The JSON wire form still
/// stores them as base64 Arrow-IPC bytes (see [`ScalarStatsAggDto`]).
#[derive(Debug, Clone)]
pub struct ScalarStatsAgg {
    pub min: ArrayRef,
    pub max: ArrayRef,
    /// Σ null_count across the part's segments; `None` when any
    /// segment lacks the stat (total unknowable, never zero).
    pub null_count: Option<u64>,
    /// Part-wide exact sum as a length-1 [`ArrayRef`] (same typing
    /// as the per-segment scalar-stats `sum`); `None` when any
    /// segment lacks it or the fold overflowed.
    pub sum: Option<ArrayRef>,
    /// Merged HLL distinct sketch (raw registers); `None` when any
    /// segment lacks one.
    pub hll: Option<Vec<u8>>,
    /// Exact non-null value frequencies when the column has at most
    /// [`MAX_EXACT_VALUE_COUNTS`] distinct values.
    pub(crate) value_counts: Option<ScalarValueCounts>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ScalarValueCounts {
    entries: Vec<(ScalarValue, u64)>,
}

impl ScalarValueCounts {
    fn from_column(column: &ArrayRef) -> Option<Self> {
        if !exact_value_counts_type(column.data_type()) {
            return None;
        }
        match column.data_type() {
            DataType::Int64 => {
                let array = column.as_any().downcast_ref::<Int64Array>()?;
                return Self::from_int64(array);
            }
            DataType::Utf8 => {
                let array = column.as_any().downcast_ref::<StringArray>()?;
                return Self::from_strings(
                    (0..array.len()).map(|row| (!array.is_null(row)).then(|| array.value(row))),
                    ScalarValue::Utf8,
                );
            }
            DataType::LargeUtf8 => {
                let array = column.as_any().downcast_ref::<LargeStringArray>()?;
                return Self::from_strings(
                    (0..array.len()).map(|row| (!array.is_null(row)).then(|| array.value(row))),
                    ScalarValue::LargeUtf8,
                );
            }
            _ => {}
        }
        let mut counts: HashMap<ScalarValue, u64> = HashMap::new();
        for row in 0..column.len() {
            if column.is_null(row) {
                continue;
            }
            let value = ScalarValue::try_from_array(column, row).ok()?;
            if !counts.contains_key(&value) && counts.len() >= MAX_EXACT_VALUE_COUNTS {
                return None;
            }
            let count = counts.entry(value).or_default();
            *count = count.checked_add(1)?;
        }
        Self::from_entries(counts.into_iter().collect())
    }

    fn from_int64(column: &Int64Array) -> Option<Self> {
        let mut counts: HashMap<i64, u64> = HashMap::new();
        for row in 0..column.len() {
            if column.is_null(row) {
                continue;
            }
            let value = column.value(row);
            if !counts.contains_key(&value) && counts.len() >= MAX_EXACT_VALUE_COUNTS {
                return None;
            }
            let count = counts.entry(value).or_default();
            *count = count.checked_add(1)?;
        }
        Self::from_entries(
            counts
                .into_iter()
                .map(|(value, count)| (ScalarValue::Int64(Some(value)), count))
                .collect(),
        )
    }

    fn from_strings<'a>(
        values: impl IntoIterator<Item = Option<&'a str>>,
        wrap: fn(Option<String>) -> ScalarValue,
    ) -> Option<Self> {
        let mut counts: HashMap<String, u64> = HashMap::new();
        for value in values.into_iter().flatten() {
            if let Some(count) = counts.get_mut(value) {
                *count = count.checked_add(1)?;
                continue;
            }
            if counts.len() >= MAX_EXACT_VALUE_COUNTS {
                return None;
            }
            counts.insert(value.to_string(), 1);
        }
        Self::from_entries(
            counts
                .into_iter()
                .map(|(value, count)| (wrap(Some(value)), count))
                .collect(),
        )
    }

    pub(crate) fn from_entries(entries: Vec<(ScalarValue, u64)>) -> Option<Self> {
        if entries.is_empty() || entries.len() > MAX_EXACT_VALUE_COUNTS {
            return None;
        }
        if !exact_value_counts_type(&entries[0].0.data_type()) {
            return None;
        }
        let mut merged: HashMap<ScalarValue, u64> = HashMap::with_capacity(entries.len());
        for (value, count) in entries {
            if value.is_null() || count == 0 {
                return None;
            }
            let total = merged.entry(value).or_default();
            *total = total.checked_add(count)?;
            if merged.len() > MAX_EXACT_VALUE_COUNTS {
                return None;
            }
        }
        let mut entries: Vec<(ScalarValue, u64)> = merged.into_iter().collect();
        if entries
            .iter()
            .any(|(value, _)| value.data_type() != entries[0].0.data_type())
        {
            return None;
        }
        entries.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
        if entries
            .windows(2)
            .any(|pair| pair[0].0.partial_cmp(&pair[1].0).is_none())
        {
            return None;
        }
        Some(Self { entries })
    }

    pub(crate) fn merged_with(&self, other: &Self) -> Option<Self> {
        Self::from_entries(self.entries.iter().chain(&other.entries).cloned().collect())
    }

    pub(crate) fn entries(&self) -> &[(ScalarValue, u64)] {
        &self.entries
    }
}

fn exact_value_counts_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Decimal128(_, _)
    )
}

impl PartialEq for ScalarStatsAgg {
    /// Equality compares the Arrow array contents (via [`ArrayRef::to_data`])
    /// rather than pointer identity, so two independently-built aggregates
    /// carrying the same values compare equal — the behaviour the round-trip
    /// tests rely on. `ArrayRef` is not `Eq` (floats), so this type only
    /// implements `PartialEq`.
    fn eq(&self, other: &Self) -> bool {
        let sum_eq = match (&self.sum, &other.sum) {
            (Some(a), Some(b)) => a.to_data() == b.to_data(),
            (None, None) => true,
            _ => false,
        };
        self.min.to_data() == other.min.to_data()
            && self.max.to_data() == other.max.to_data()
            && self.null_count == other.null_count
            && sum_eq
            && self.hll == other.hll
            && self.value_counts == other.value_counts
    }
}

impl ScalarStatsAgg {
    /// Build the aggregate for one column from a resident Arrow array.
    ///
    /// Returns `None` for types without a well-defined ordering (anything
    /// other than integer / float / boolean / utf8 / decimal) — those carry
    /// no min/max, so there's nothing to prune on. An all-null column of a
    /// supported type still returns an aggregate: null min/max plus the
    /// null count, so `IS [NOT] NULL` can prune on it. When present, every
    /// companion stat (null count, exact sum, HLL sketch) is computed in the
    /// same pass; sum/HLL/value counts stay `None` for unsupported types or
    /// when the exact-value cardinality cap is exceeded.
    pub fn from_column(column: &ArrayRef) -> Option<ScalarStatsAgg> {
        let (min, max) = column_min_max(column)?;
        let null_count = u64::try_from(column.null_count()).ok();

        Some(ScalarStatsAgg {
            min,
            max,
            null_count,
            sum: column_sum(column),
            hll: column_hll(column).map(|s| s.as_bytes().to_vec()),
            value_counts: ScalarValueCounts::from_column(column),
        })
    }

    /// Build a per-column aggregate table from one `RecordBatch`, keyed by
    /// column name. Columns whose type isn't orderable are skipped (no
    /// entry), mirroring [`ScalarStatsAgg::from_column`]. Thin wrapper over
    /// [`ScalarStatsAgg::from_batches`] (a single-array concat is a cheap
    /// clone).
    pub fn from_batch(
        scalar_schema: &Schema,
        batch: &RecordBatch,
    ) -> HashMap<String, ScalarStatsAgg> {
        ScalarStatsAgg::from_batches(scalar_schema, &[batch])
    }

    /// Build a per-column aggregate table across several `RecordBatch`es.
    ///
    /// Each column is concatenated across the batches before its stats are
    /// computed. A column whose concat fails (shape mismatch) is skipped —
    /// the prune planner treats missing stats as "can't prune", the safe
    /// default. An empty `batches` slice yields an empty table.
    pub fn from_batches(
        scalar_schema: &Schema,
        batches: &[&RecordBatch],
    ) -> HashMap<String, ScalarStatsAgg> {
        let mut out = HashMap::new();
        if batches.is_empty() {
            return out;
        }
        for (idx, field) in scalar_schema.fields().iter().enumerate() {
            // A batch shorter than the schema (malformed input) doesn't carry
            // this column. Use a checked lookup and skip the column rather
            // than panicking via `RecordBatch::column` — missing stats are the
            // safe default (the prune planner treats them as "can't prune").
            let Some(arrays) = batches
                .iter()
                .map(|b| b.columns().get(idx).map(|c| c.as_ref()))
                .collect::<Option<Vec<&dyn Array>>>()
            else {
                continue;
            };
            let combined = match concat(&arrays) {
                Ok(a) => a,
                Err(_) => continue,
            };
            if let Some(agg) = ScalarStatsAgg::from_column(&combined) {
                out.insert(field.name().to_string(), agg);
            }
        }
        out
    }

    /// Merge `other` into `self` for the same column.
    ///
    /// On success, min/max keep the extremes across both sides and the
    /// additive stats (null count, sum, HLL) combine **only when both sides
    /// carry them** — a side missing the stat makes the total unknowable, so
    /// the merged entry drops to `None` (consumers treat missing as "no
    /// statistics", never as zero).
    ///
    /// Returns [`ScalarStatsMergeError`] (leaving `self` **untouched**) when
    /// the two min/max arrays have incompatible Arrow types. The bounds can't
    /// be combined soundly, and silently keeping `self`'s bounds would
    /// under-cover `other`'s values — making the pruner drop matching rows
    /// (a false prune). In a well-formed table a column has a single type, so
    /// this signals corruption or a logic bug; the caller decides how to
    /// degrade (see [`ScalarStatsAgg::merge_tables`]).
    pub fn merge_with(&mut self, other: &ScalarStatsAgg) -> Result<(), ScalarStatsMergeError> {
        // Resolve the bounds first; bail before mutating anything so a failed
        // merge can't leave half-updated, internally-inconsistent stats.
        let Some((min, max)) = merge_min_max_arrays(&self.min, &other.min, &self.max, &other.max)
        else {
            return Err(ScalarStatsMergeError {
                left: self.min.data_type().clone(),
                right: other.min.data_type().clone(),
            });
        };
        self.min = min;
        self.max = max;
        self.null_count = match (self.null_count, other.null_count) {
            (Some(a), Some(b)) => a.checked_add(b),
            _ => None,
        };
        self.sum = match (&self.sum, &other.sum) {
            (Some(a), Some(b)) => add_sum_arrays(a, b),
            _ => None,
        };
        self.hll = match (&self.hll, &other.hll) {
            (Some(a), Some(b)) => match (HllSketch::from_bytes(a), HllSketch::from_bytes(b)) {
                (Some(mut merged), Some(other_sketch)) => {
                    merged.merge(&other_sketch);
                    Some(merged.as_bytes().to_vec())
                }
                _ => None,
            },
            _ => None,
        };
        self.value_counts = match (&self.value_counts, &other.value_counts) {
            (Some(left), Some(right)) => left.merged_with(right),
            _ => None,
        };
        Ok(())
    }

    /// Merge two per-column scalar-stats tables
    /// (`HashMap<String, ScalarStatsAgg>`), folding `other` into `into`.
    ///
    /// Column **union**: a column present only in `other` is inserted; a
    /// column present in both is merged per-column via
    /// [`ScalarStatsAgg::merge`]. Folding this over a set of per-superfile
    /// tables yields the part-level aggregate.
    ///
    /// If a shared column's min/max types are incompatible (merge errors), the
    /// column is **dropped** from `into` rather than kept with stale bounds —
    /// an absent column is "no info" to the pruner (always keep), which is
    /// conservative; keeping unsound bounds could drop matching rows.
    pub fn merge(
        into: &mut HashMap<String, ScalarStatsAgg>,
        other: &HashMap<String, ScalarStatsAgg>,
    ) {
        for (col, other_agg) in other {
            if let Some(existing) = into.get_mut(col) {
                if existing.merge_with(other_agg).is_err() {
                    into.remove(col);
                }
            } else {
                into.insert(col.clone(), other_agg.clone());
            }
        }
    }

    /// Test-only constructor for an aggregate carrying only min/max bounds
    /// (no null count, sum, or HLL) — the shape many skip-pruning tests
    /// build directly.
    #[cfg(test)]
    pub(crate) fn from_min_max(min: ArrayRef, max: ArrayRef) -> Self {
        Self {
            min,
            max,
            null_count: None,
            sum: None,
            hll: None,
            value_counts: None,
        }
    }
}

/// Two aggregates for the same column carry incompatible min/max Arrow types,
/// so their bounds can't be combined into a sound merged bound. Returned by
/// [`ScalarStatsAgg::merge`] instead of silently keeping stale bounds (which
/// could make pruning drop matching rows). In a well-formed table a column
/// has one type, so this signals corruption or a logic bug.
#[derive(Debug, Error)]
#[error("incompatible scalar-stats min/max types: {left:?} vs {right:?}")]
pub struct ScalarStatsMergeError {
    left: DataType,
    right: DataType,
}

/// FTS skip summary for one column. Used both per-superfile
/// (`SuperfileEntry.fts_summary`) and as the per-part aggregate
/// (`ManifestPartEntry.fts_summary_agg`) — the per-part value is the
/// bloom-union + range-union across the part's superfiles.
///
/// The bloom is held as a decoded [`Bloom`] (cheap `Arc<[u64]>` clone) so
/// the prune hot path can call `term_bloom.contains(..)` without a
/// per-query `Bloom::from_bytes` copy; the JSON/byte wire form stores the
/// bloom bytes (see [`FtsSummaryAggDto`] / [`super::encoding`]). The
/// `Default` shape — `term_bloom: None`, no range — is treated as
/// "always-keep" by the list-level pruner (correctness preserved;
/// selectivity 0).
#[derive(Debug, Clone, Default)]
pub struct FtsSummaryAgg {
    /// Term-presence bloom. `None` means "no bloom info" — the list-level
    /// pruner treats it as always-keep. `Bloom` carries its own block
    /// count, so no separate `n_blocks` field is needed.
    pub term_bloom: Option<Bloom>,
    /// HyperLogLog-estimated distinct term count. `0` for the `Default`
    /// shape and currently for the part-level rollup (deferred).
    pub n_terms_distinct: u64,
    /// `(min, max)` lex term range. `None` if the FST was empty for this
    /// column (per-superfile) or every superfile's FST was empty (part).
    pub term_range: Option<(Vec<u8>, Vec<u8>)>,
}

impl PartialEq for FtsSummaryAgg {
    /// `Bloom` is not `PartialEq`, so compare it by its serialized bytes
    /// (the round-trip tests rely on value equality). Mirrors the manual
    /// `PartialEq` on [`ScalarStatsAgg`].
    fn eq(&self, other: &Self) -> bool {
        let bloom_eq = match (&self.term_bloom, &other.term_bloom) {
            (Some(a), Some(b)) => a.to_bytes() == b.to_bytes(),
            (None, None) => true,
            _ => false,
        };
        bloom_eq
            && self.n_terms_distinct == other.n_terms_distinct
            && self.term_range == other.term_range
    }
}

impl FtsSummaryAgg {
    /// Merge `other` into `self` — the union the part-level aggregate forms
    /// across a part's superfiles:
    ///
    /// - **bloom**: bit-OR of the two filters (a term in either is in the
    ///   union). Both must share a shape; a mismatch can't be unioned soundly,
    ///   so the merged bloom drops to `None`.
    /// - **term range**: widened to span both — `(min(mins), max(maxes))` lex.
    /// - **distinct count**: a deferred planner hint; takes the larger side.
    ///
    /// **`None` is the identity here** (an empty contributor that leaves the
    /// other side intact) — what a fold from [`Default::default`] over
    /// per-superfile summaries needs, since every superfile carries a bloom
    /// ([`new_with_params`] always yields `Some`). This is deliberately
    /// *distinct* from the prune-time reading of `term_bloom: None` as "no
    /// info / always-keep": a sound union of a known bloom with a genuinely
    /// unknown one is unknown (`None`), so `merge` must only be folded over
    /// summaries that carry real blooms — never over a true no-info summary.
    ///
    /// Folding `merge` over a part's superfiles yields the same bloom-union and
    /// range-union as [`crate::supertable::manifest::aggregates`]'s rollup; the
    /// `n_terms_distinct` hint differs (here the larger side, vs. the rollup's
    /// current placeholder `0`).
    pub fn merge_with(&mut self, other: &FtsSummaryAgg) {
        self.term_bloom = match (self.term_bloom.take(), other.term_bloom.as_ref()) {
            (Some(a), Some(b)) => union_blooms(&a, b),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b.clone()),
            (None, None) => None,
        };
        self.term_range = match (self.term_range.take(), other.term_range.as_ref()) {
            (Some((amin, amax)), Some((bmin, bmax))) => {
                let min = if &amin <= bmin { amin } else { bmin.clone() };
                let max = if &amax >= bmax { amax } else { bmax.clone() };
                Some((min, max))
            }
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b.clone()),
            (None, None) => None,
        };
        self.n_terms_distinct = self.n_terms_distinct.max(other.n_terms_distinct);
    }

    /// Merge two per-FTS-column summary tables
    /// (`BTreeMap<String, FtsSummaryAgg>`), folding `other` into `into`.
    ///
    /// Column **union**: a column present only in `other` is inserted; a
    /// column present in both is merged per-column via
    /// [`FtsSummaryAgg::merge`]. Folding this over a set of per-part
    /// tables yields the table-level aggregate.
    ///
    /// If a shared column's bloom shapes are incompatible (merge yields `None`),
    /// the column is **dropped** from `into` rather than kept with a `None`
    /// bloom — an absent column is "no info" to the pruner (always keep), which
    /// is conservative and equivalent.
    pub fn merge(
        into: &mut BTreeMap<String, FtsSummaryAgg>,
        other: &BTreeMap<String, FtsSummaryAgg>,
    ) {
        for (col, other_agg) in other {
            if let Some(existing) = into.get_mut(col) {
                existing.merge_with(other_agg);
                if existing.term_bloom.is_none() {
                    into.remove(col);
                }
            } else {
                into.insert(col.clone(), other_agg.clone());
            }
        }
    }

    /// Build the per-superfile summary for one column from its freshly-built
    /// bloom, distinct-term count, and `(min, max)` lex term range.
    ///
    /// Adapts the per-superfile shape to this type: the bloom is always
    /// present (`Some`); the count widens `u32` → `u64`; and an empty
    /// `(min, max)` range (a 0-term column) becomes `None` — the same
    /// "no range" signal the pruner already understands.
    pub fn new_with_params(
        term_bloom: Bloom,
        n_terms_distinct: u32,
        term_range: (Vec<u8>, Vec<u8>),
    ) -> Self {
        let term_range = if term_range.0.is_empty() && term_range.1.is_empty() {
            None
        } else {
            Some(term_range)
        };
        Self {
            term_bloom: Some(term_bloom),
            n_terms_distinct: u64::from(n_terms_distinct),
            term_range,
        }
    }

    /// Whether this summary's bloom *may* contain `term` (a `false` is
    /// definitive: the term is absent). A `None` bloom is "no info", so it
    /// conservatively returns `true` (keep). This is the per-term primitive
    /// both the superfile-level (`fts_bloom_skip`) and list-level
    /// (`part_matches_terms`) bloom skips build on.
    pub fn may_contain(&self, term: &[u8]) -> bool {
        self.term_bloom.as_ref().is_none_or(|b| b.contains(term))
    }

    /// Whether this summary's lex term range *could* contain a term starting
    /// with `prefix` (i.e. `[prefix, prefix_upper_bound)` overlaps the range).
    /// A `None` range means the FST was empty for this column — nothing
    /// matches, so this returns `false` (prune). The per-term-range primitive
    /// both the superfile-level (`fts_prefix_skip`) and list-level
    /// (`part_overlaps_prefix`) prefix skips build on.
    pub fn may_match_prefix(&self, prefix: &[u8]) -> bool {
        match self.term_range.as_ref() {
            Some((min, max)) => prefix_overlaps_range(prefix, min, max),
            None => false,
        }
    }
}

/// Bit-OR two same-shape blooms into their union. Different shapes can't be
/// unioned (the block layout differs), so this returns `None` — "no bloom
/// info", which the list-level pruner treats as always-keep.
fn union_blooms(a: &Bloom, b: &Bloom) -> Option<Bloom> {
    let mut ab = a.to_bytes();
    let bb = b.to_bytes();
    if ab.len() != bb.len() {
        return None;
    }
    for (x, y) in ab.iter_mut().zip(bb.iter()) {
        *x |= *y;
    }
    Bloom::from_bytes(&ab)
}

// ---------- Errors ----------

#[derive(Debug, Error)]
pub enum ListParseError {
    #[error("json parse failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("base64 decode failed for {field}: {source}")]
    Base64 {
        field: &'static str,
        source: base64::DecodeError,
    },
    #[error("bad content_hash: {0}")]
    BadContentHash(String),
    #[error("bad part_id: {0}")]
    BadPartId(String),
    #[error("bad value for field {0}: {1:?}")]
    BadFieldValue(&'static str, String),
    #[error("scalar stats decode failed for {field}: {source}")]
    ScalarStats {
        field: &'static str,
        source: DecodeError,
    },
    /// A non-empty `term_bloom_union` payload didn't decode to a valid
    /// `Bloom` layout (`n_blocks × BLOCK_BYTES`, power-of-two). Surfaced
    /// rather than silently dropped to "no bloom", so on-disk corruption
    /// isn't masked as a valid always-keep summary.
    #[error("invalid term bloom layout: {0} bytes")]
    InvalidBloom(usize),
    #[error("incompatible major version: got {got}, supported {supported}")]
    IncompatibleMajorVersion { got: String, supported: String },
}

#[derive(Debug, Error)]
pub enum ListEncodeError {
    #[error("json encode failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("scalar stats encode failed for {field}: {source}")]
    ScalarStats {
        field: &'static str,
        source: EncodeError,
    },
}

// ---------- Wire (serde) types ----------

/// JSON wire shape — DTO that owns the base64 transformations.
///
/// Field ordering here is the JSON-output order; serde_json's
/// `to_writer_pretty` preserves declaration order, so output
/// is deterministic for content-addressing.
#[derive(Serialize, Deserialize)]
struct ManifestDto {
    format_version: String,
    manifest_id: u64,
    options_hash: String, // "blake3:<64hex>"
    schema: String,       // base64
    id_column: String,
    fts_columns: Vec<FtsColumnInfo>,
    vector_columns: Vec<VectorColumnInfoDto>,
    #[serde(default)]
    vector_index_storage_prefix: Option<String>,
    #[serde(default)]
    deleted_user_ids_inline_b64: Option<String>, // base64 of encoded id set
    #[serde(default)]
    slow_vector_state_uri: Option<String>,
    #[serde(default)]
    slow_vector_state_content_hash: Option<String>, // "blake3:<64hex>"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    slow_vector_state_centroids_uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    slow_vector_state_centroids_content_hash: Option<String>, // "blake3:<64hex>"
    partition_strategy: PartitionStrategyDto,
    #[serde(default)]
    global_vector_index: Option<GlobalVectorIndexDto>,
    #[serde(default)]
    drained_ranges: Vec<(u64, u64)>,
    parts: Vec<ManifestPartEntryDto>,
    tombstone_seqs: BTreeMap<String, u64>, // UUID keys
}

#[derive(Serialize, Deserialize)]
struct GlobalVectorIndexDto {
    column: String,
    grid_b64: String,
    /// Absent on manifests written before the user-side grid existed.
    #[serde(default)]
    user_grid_b64: Option<String>,
}

// VectorColumnInfo's `dim`/`n_cent` are `usize` in memory but
// JSON should canonicalize as `u64` so round-trip on 32-bit
// hosts isn't a footgun.
#[derive(Serialize, Deserialize)]
struct VectorColumnInfoDto {
    column: String,
    dim: u64,
    n_cent: u64,
    rot_seed: u64,
    metric: String,
}

impl Serialize for FtsColumnInfo {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut st = s.serialize_struct("FtsColumnInfo", 1)?;
        use serde::ser::SerializeStruct;
        st.serialize_field("column", &self.column)?;
        st.end()
    }
}

impl<'de> Deserialize<'de> for FtsColumnInfo {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Inner {
            column: String,
        }
        Inner::deserialize(d).map(|i| Self { column: i.column })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PartitionStrategyDto {
    TimeRange {
        column: String,
        granularity_secs: i64,
    },
    Hash {
        column: String,
        n_buckets: u32,
    },
    ColumnRange {
        column: String,
        boundaries: Vec<String>, // base64 per boundary
    },
    VectorCell {
        column: String,
        clusters_b64: String,
        #[serde(default)]
        routing: Option<CellRoutingParamsDto>,
    },
    IngestionTime {
        granularity_secs: i64,
    },
}

#[derive(Serialize, Deserialize, Default)]
struct CellRoutingParamsDto {
    #[serde(default)]
    nprobe_min: usize,
    #[serde(default)]
    nprobe_max: usize,
    /// `Option` rather than a zero sentinel: `slack = 0.0` is a meaningful
    /// persisted value (no near-tie widening — τ = d* exactly), so absence
    /// must be distinguishable from explicit zero. The `usize` knobs keep
    /// the zero sentinel — zero probes/fine-runs is not a valid config.
    #[serde(default)]
    slack: Option<f32>,
    #[serde(default)]
    fine_nprobe: usize,
}

impl From<CellRoutingParams> for CellRoutingParamsDto {
    fn from(r: CellRoutingParams) -> Self {
        Self {
            nprobe_min: r.nprobe_min,
            nprobe_max: r.nprobe_max,
            slack: Some(r.slack),
            fine_nprobe: r.fine_nprobe,
        }
    }
}

impl From<CellRoutingParamsDto> for CellRoutingParams {
    fn from(d: CellRoutingParamsDto) -> Self {
        let mut r = CellRoutingParams::default();
        if d.nprobe_min > 0 {
            r.nprobe_min = d.nprobe_min;
        }
        if d.nprobe_max > 0 {
            r.nprobe_max = d.nprobe_max;
        }
        if let Some(slack) = d.slack {
            r.slack = slack;
        }
        if d.fine_nprobe > 0 {
            r.fine_nprobe = d.fine_nprobe;
        }
        r.nprobe_max = r.nprobe_max.max(r.nprobe_min);
        r
    }
}

#[derive(Serialize, Deserialize)]
struct ManifestPartEntryDto {
    part_id: String, // UUID
    uri: String,
    n_superfiles: u64,
    size_bytes_compressed: u64,
    size_bytes_uncompressed: u64,
    content_hash: String, // "blake3:<hex>"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    routing_uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    routing_content_hash: Option<String>, // "blake3:<hex>"
    // i128 stringified as decimal — JSON numbers are bounded
    // to f64 precision (~53 bits) so we can't round-trip a
    // 128-bit value as a JSON number without loss. Decimal
    // strings keep the manifest list debuggable in `jq`
    // (`echo '...' | jq '.parts[0].id_range'` shows real
    // values) and avoid base64 ambiguity.
    id_range: (String, String),
    scalar_stats_agg: BTreeMap<String, ScalarStatsAggDto>,
    fts_summary_agg: BTreeMap<String, FtsSummaryAggDto>,
}

#[derive(Serialize, Deserialize)]
struct ScalarStatsAggDto {
    min: String, // base64
    max: String, // base64
    /// `None` ↔ field absent in JSON (parts written before the stat
    /// existed decode cleanly).
    #[serde(default)]
    null_count: Option<u64>,
    #[serde(default)]
    sum: Option<String>, // base64
    #[serde(default)]
    hll: Option<String>, // base64
    #[serde(default, skip_serializing_if = "Option::is_none")]
    value_counts: Option<String>, // base64 Arrow IPC
}

#[derive(Serialize, Deserialize)]
struct FtsSummaryAggDto {
    /// base64 of `Bloom::to_bytes()`; empty string ↔ no bloom (`None`).
    /// The block count is inferred from the byte length at decode, so no
    /// separate `n_blocks` field is carried. (Older manifests that still
    /// carry `term_bloom_n_blocks` decode cleanly — serde ignores it.)
    term_bloom_union: String,
    n_terms_distinct: u64,
    /// `None` ↔ field absent in JSON, not a `null`. Cleaner
    /// `jq` shape and avoids the
    /// `null`-vs-`{"min":"","max":""}` ambiguity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    term_range_union: Option<TermRangeUnionDto>,
}

#[derive(Serialize, Deserialize)]
struct TermRangeUnionDto {
    min: String, // base64
    max: String, // base64
}

// ---------- DTO conversions ----------

fn encode_hash(h: &ContentHash) -> String {
    format!("blake3:{}", h.to_hex())
}

fn decode_hash(s: &str) -> Result<ContentHash, ListParseError> {
    let hex = s
        .strip_prefix("blake3:")
        .ok_or_else(|| ListParseError::BadContentHash(s.into()))?;
    if hex.len() != BLAKE3_HEX_LEN {
        return Err(ListParseError::BadContentHash(s.into()));
    }
    let mut out = [0u8; BLAKE3_DIGEST_BYTES];
    for i in 0..BLAKE3_DIGEST_BYTES {
        let byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
            .map_err(|_| ListParseError::BadContentHash(s.into()))?;
        out[i] = byte;
    }
    Ok(ContentHash(out))
}

fn encode_b64(b: &[u8]) -> String {
    BASE64.encode(b)
}

fn decode_b64(s: &str, field: &'static str) -> Result<Vec<u8>, ListParseError> {
    BASE64
        .decode(s)
        .map_err(|source| ListParseError::Base64 { field, source })
}

fn encode_scalar_array(
    column: &str,
    field: &'static str,
    arr: &ArrayRef,
) -> Result<String, ListEncodeError> {
    let bytes = encode_length1_array(column, arr)
        .map_err(|source| ListEncodeError::ScalarStats { field, source })?;
    Ok(encode_b64(&bytes))
}

fn entry_to_dto(e: &ManifestPartEntry) -> Result<ManifestPartEntryDto, ListEncodeError> {
    let mut scalar_stats_agg = BTreeMap::new();
    for (k, v) in &e.scalar_stats_agg {
        let sum = match &v.sum {
            None => None,
            Some(s) => Some(encode_scalar_array(k, "scalar_stats_agg.sum", s)?),
        };
        scalar_stats_agg.insert(
            k.clone(),
            ScalarStatsAggDto {
                min: encode_scalar_array(k, "scalar_stats_agg.min", &v.min)?,
                max: encode_scalar_array(k, "scalar_stats_agg.max", &v.max)?,
                null_count: v.null_count,
                sum,
                hll: v.hll.as_deref().map(encode_b64),
                value_counts: v
                    .value_counts
                    .as_ref()
                    .map(encode_value_counts)
                    .transpose()
                    .map_err(|source| ListEncodeError::ScalarStats {
                        field: "scalar_stats_agg.value_counts",
                        source,
                    })?
                    .as_deref()
                    .map(encode_b64),
            },
        );
    }
    Ok(ManifestPartEntryDto {
        part_id: e.part_id.0.to_string(),
        uri: e.uri.clone(),
        n_superfiles: e.n_superfiles,
        size_bytes_compressed: e.size_bytes_compressed,
        size_bytes_uncompressed: e.size_bytes_uncompressed,
        content_hash: encode_hash(&e.content_hash),
        routing_uri: e.routing.as_ref().map(|r| r.uri.clone()),
        routing_content_hash: e.routing.as_ref().map(|r| encode_hash(&r.content_hash)),
        id_range: (e.id_range.0.to_string(), e.id_range.1.to_string()),
        scalar_stats_agg,
        fts_summary_agg: e
            .fts_summary_agg
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    FtsSummaryAggDto {
                        term_bloom_union: v
                            .term_bloom
                            .as_ref()
                            .map(|b| encode_b64(&b.to_bytes()))
                            .unwrap_or_default(),
                        n_terms_distinct: v.n_terms_distinct,
                        term_range_union: v.term_range.as_ref().map(|(mn, mx)| TermRangeUnionDto {
                            min: encode_b64(mn),
                            max: encode_b64(mx),
                        }),
                    },
                )
            })
            .collect(),
    })
}

fn entry_from_dto(d: ManifestPartEntryDto) -> Result<ManifestPartEntry, ListParseError> {
    let part_id =
        PartId(Uuid::parse_str(&d.part_id).map_err(|e| ListParseError::BadPartId(e.to_string()))?);
    let content_hash = decode_hash(&d.content_hash)?;
    let mut scalar_stats_agg = HashMap::new();
    for (k, v) in d.scalar_stats_agg {
        let min = decode_length1_array(&decode_b64(&v.min, "scalar_stats_agg.min")?).map_err(
            |source| ListParseError::ScalarStats {
                field: "scalar_stats_agg.min",
                source,
            },
        )?;
        let max = decode_length1_array(&decode_b64(&v.max, "scalar_stats_agg.max")?).map_err(
            |source| ListParseError::ScalarStats {
                field: "scalar_stats_agg.max",
                source,
            },
        )?;
        let sum = match v.sum.as_deref() {
            None => None,
            Some(s) => Some(
                decode_length1_array(&decode_b64(s, "scalar_stats_agg.sum")?).map_err(
                    |source| ListParseError::ScalarStats {
                        field: "scalar_stats_agg.sum",
                        source,
                    },
                )?,
            ),
        };
        scalar_stats_agg.insert(
            k,
            ScalarStatsAgg {
                min,
                max,
                null_count: v.null_count,
                sum,
                hll: v
                    .hll
                    .as_deref()
                    .map(|s| decode_b64(s, "scalar_stats_agg.hll"))
                    .transpose()?,
                value_counts: v
                    .value_counts
                    .as_deref()
                    .map(|encoded| {
                        decode_value_counts(&decode_b64(encoded, "scalar_stats_agg.value_counts")?)
                            .map_err(|source| ListParseError::ScalarStats {
                                field: "scalar_stats_agg.value_counts",
                                source,
                            })
                    })
                    .transpose()?,
            },
        );
    }
    let mut fts_summary_agg = BTreeMap::new();
    for (k, v) in d.fts_summary_agg {
        fts_summary_agg.insert(
            k,
            FtsSummaryAgg {
                term_bloom: {
                    let bytes = decode_b64(&v.term_bloom_union, "term_bloom_union")?;
                    // Empty ⇒ no bloom (the pruner conservatively keeps the
                    // part). Non-empty but malformed ⇒ on-disk corruption,
                    // surfaced as a parse error rather than masked as a valid
                    // "always-keep" summary.
                    if bytes.is_empty() {
                        None
                    } else {
                        Some(
                            Bloom::from_bytes(&bytes)
                                .ok_or(ListParseError::InvalidBloom(bytes.len()))?,
                        )
                    }
                },
                n_terms_distinct: v.n_terms_distinct,
                term_range: match v.term_range_union {
                    None => None,
                    Some(tr) => Some((
                        decode_b64(&tr.min, "term_range_union.min")?,
                        decode_b64(&tr.max, "term_range_union.max")?,
                    )),
                },
            },
        );
    }
    Ok(ManifestPartEntry {
        part_id,
        uri: d.uri,
        n_superfiles: d.n_superfiles,
        size_bytes_compressed: d.size_bytes_compressed,
        size_bytes_uncompressed: d.size_bytes_uncompressed,
        content_hash,
        // Require both halves, mirroring the slow-state routing ref: a
        // part carrying only one is treated as having no sibling.
        routing: match (d.routing_uri, d.routing_content_hash.as_deref()) {
            (Some(uri), Some(hash)) => Some(RoutingRef {
                uri,
                content_hash: decode_hash(hash)?,
            }),
            _ => None,
        },
        id_range: {
            let lo =
                d.id_range.0.parse::<i128>().map_err(|_| {
                    ListParseError::BadFieldValue("id_range[0]", d.id_range.0.clone())
                })?;
            let hi =
                d.id_range.1.parse::<i128>().map_err(|_| {
                    ListParseError::BadFieldValue("id_range[1]", d.id_range.1.clone())
                })?;
            (lo, hi)
        },
        scalar_stats_agg,
        fts_summary_agg,
    })
}

fn strategy_to_dto(s: &PartitionStrategy) -> PartitionStrategyDto {
    match s {
        PartitionStrategy::TimeRange {
            column,
            granularity_secs,
        } => PartitionStrategyDto::TimeRange {
            column: column.clone(),
            granularity_secs: *granularity_secs,
        },
        PartitionStrategy::Hash { column, n_buckets } => PartitionStrategyDto::Hash {
            column: column.clone(),
            n_buckets: *n_buckets,
        },
        PartitionStrategy::ColumnRange { column, boundaries } => {
            PartitionStrategyDto::ColumnRange {
                column: column.clone(),
                boundaries: boundaries.iter().map(|b| encode_b64(b)).collect(),
            }
        }
        PartitionStrategy::VectorCell {
            column,
            clusters,
            routing,
        } => {
            let enc = encode_cluster_centroids(clusters);
            PartitionStrategyDto::VectorCell {
                column: column.clone(),
                clusters_b64: encode_b64(&enc),
                routing: Some(CellRoutingParamsDto::from(*routing)),
            }
        }
        PartitionStrategy::IngestionTime { granularity_secs } => {
            PartitionStrategyDto::IngestionTime {
                granularity_secs: *granularity_secs,
            }
        }
    }
}

fn strategy_from_dto(d: PartitionStrategyDto) -> Result<PartitionStrategy, ListParseError> {
    Ok(match d {
        PartitionStrategyDto::TimeRange {
            column,
            granularity_secs,
        } => PartitionStrategy::TimeRange {
            column,
            granularity_secs,
        },
        PartitionStrategyDto::Hash { column, n_buckets } => {
            PartitionStrategy::Hash { column, n_buckets }
        }
        PartitionStrategyDto::ColumnRange { column, boundaries } => {
            let mut bs = Vec::with_capacity(boundaries.len());
            for b in boundaries {
                bs.push(decode_b64(&b, "partition_strategy.boundaries")?);
            }
            PartitionStrategy::ColumnRange {
                column,
                boundaries: bs,
            }
        }
        PartitionStrategyDto::VectorCell {
            column,
            clusters_b64,
            routing,
        } => {
            let bytes = decode_b64(&clusters_b64, "partition_strategy.clusters")?;
            let clusters = decode_cluster_centroids(&bytes).map_err(|e| {
                ListParseError::BadFieldValue("partition_strategy.clusters", e.to_string())
            })?;
            PartitionStrategy::VectorCell {
                column,
                clusters,
                routing: routing.map(CellRoutingParams::from).unwrap_or_default(),
            }
        }
        PartitionStrategyDto::IngestionTime { granularity_secs } => {
            PartitionStrategy::IngestionTime { granularity_secs }
        }
    })
}

fn list_to_dto(l: &Manifest) -> Result<ManifestDto, ListEncodeError> {
    let parts = l
        .parts
        .iter()
        .map(entry_to_dto)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ManifestDto {
        format_version: l.format_version.clone(),
        manifest_id: l.manifest_id,
        options_hash: encode_hash(&l.options_hash),
        schema: encode_b64(&l.schema),
        id_column: l.id_column.clone(),
        fts_columns: l.fts_columns.clone(),
        vector_columns: l
            .vector_columns
            .iter()
            .map(|c| VectorColumnInfoDto {
                column: c.column.clone(),
                dim: c.dim as u64,
                n_cent: c.n_cent as u64,
                rot_seed: c.rot_seed,
                metric: c.metric.clone(),
            })
            .collect(),
        partition_strategy: strategy_to_dto(&l.partition_strategy),
        vector_index_storage_prefix: l.vector_index_storage_prefix.clone(),
        global_vector_index: l
            .global_vector_index
            .as_ref()
            .map(|g| GlobalVectorIndexDto {
                column: g.column.clone(),
                grid_b64: encode_b64(&encode_cluster_centroids(&g.grid)),
                user_grid_b64: g
                    .user_grid
                    .as_ref()
                    .map(|grid| encode_b64(&encode_cluster_centroids(grid))),
            }),
        drained_ranges: l.drained_ranges.intervals().to_vec(),
        deleted_user_ids_inline_b64: l.deleted_user_ids_inline.as_deref().map(encode_b64),
        slow_vector_state_uri: l.slow_vector_state_uri.clone(),
        slow_vector_state_content_hash: l.slow_vector_state_content_hash.as_ref().map(encode_hash),
        slow_vector_state_centroids_uri: l
            .slow_vector_state_centroids
            .as_ref()
            .map(|r| r.uri.clone()),
        slow_vector_state_centroids_content_hash: l
            .slow_vector_state_centroids
            .as_ref()
            .map(|r| encode_hash(&r.content_hash)),
        parts,
        tombstone_seqs: l
            .tombstone_seqs
            .iter()
            .map(|(id, seq)| (id.to_string(), *seq))
            .collect(),
    })
}

fn list_from_dto(d: ManifestDto) -> Result<Manifest, ListParseError> {
    check_major(&d.format_version)?;
    let options_hash = decode_hash(&d.options_hash)?;
    let schema = decode_b64(&d.schema, "schema")?;
    let mut parts = Vec::with_capacity(d.parts.len());
    for entry in d.parts {
        parts.push(entry_from_dto(entry)?);
    }
    Ok(Manifest {
        format_version: d.format_version,
        manifest_id: d.manifest_id,
        options_hash,
        schema,
        id_column: d.id_column,
        fts_columns: d.fts_columns,
        vector_columns: d
            .vector_columns
            .into_iter()
            .map(|c| VectorColumnInfo {
                column: c.column,
                dim: c.dim as usize,
                n_cent: c.n_cent as usize,
                rot_seed: c.rot_seed,
                metric: c.metric,
            })
            .collect(),
        partition_strategy: strategy_from_dto(d.partition_strategy)?,
        vector_index_storage_prefix: d.vector_index_storage_prefix,
        global_vector_index: d
            .global_vector_index
            .map(|g| -> Result<GlobalVectorIndex, ListParseError> {
                let bytes = decode_b64(&g.grid_b64, "global_vector_index.grid")?;
                let grid = decode_cluster_centroids(&bytes).map_err(|e| {
                    ListParseError::BadFieldValue("global_vector_index.grid", e.to_string())
                })?;
                let user_grid = g
                    .user_grid_b64
                    .as_deref()
                    .map(|b64| -> Result<_, ListParseError> {
                        let bytes = decode_b64(b64, "global_vector_index.user_grid")?;
                        decode_cluster_centroids(&bytes).map_err(|e| {
                            ListParseError::BadFieldValue(
                                "global_vector_index.user_grid",
                                e.to_string(),
                            )
                        })
                    })
                    .transpose()?;
                Ok(GlobalVectorIndex {
                    column: g.column,
                    grid,
                    user_grid,
                })
            })
            .transpose()?,
        drained_ranges: DrainedVersionRanges::from_intervals(d.drained_ranges)?,
        deleted_user_ids_inline: d
            .deleted_user_ids_inline_b64
            .as_deref()
            .map(|b64| decode_b64(b64, "deleted_user_ids_inline"))
            .transpose()?,
        // Atomic pair: unlike the centroids section below (whose contract is
        // fall-back-to-None), the slow-state ref is required where present —
        // a half-written ref is corruption, not an absent section.
        slow_vector_state_uri: match (&d.slow_vector_state_uri, &d.slow_vector_state_content_hash) {
            (Some(_), None) | (None, Some(_)) => {
                return Err(ListParseError::BadFieldValue(
                    "slow_vector_state",
                    "uri and content_hash must be present together".into(),
                ));
            }
            _ => d.slow_vector_state_uri,
        },
        slow_vector_state_content_hash: d
            .slow_vector_state_content_hash
            .as_deref()
            .map(decode_hash)
            .transpose()?,
        // Require both halves: a manifest carrying only one is treated as
        // having no section (consumers fall back to per-superfile reads)
        // rather than failing the whole list decode.
        slow_vector_state_centroids: match (
            d.slow_vector_state_centroids_uri,
            d.slow_vector_state_centroids_content_hash.as_deref(),
        ) {
            (Some(uri), Some(hash)) => Some(RoutingRef {
                uri,
                content_hash: decode_hash(hash)?,
            }),
            _ => None,
        },
        parts,
        tombstone_seqs: d
            .tombstone_seqs
            .into_iter()
            .map(|(id, seq)| {
                Uuid::parse_str(&id)
                    .map(|u| (u, seq))
                    .map_err(|_| ListParseError::BadFieldValue("tombstone_seqs", id))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?,
    })
}

// ---------- Encode / decode ----------

/// JSON-encode a manifest list. Pretty-printed; field order
/// is the declaration order in `ManifestDto` and child
/// types, so byte-output is deterministic for content-equal
/// inputs.
pub fn encode(list: &Manifest) -> Result<Vec<u8>, ListEncodeError> {
    let dto = list_to_dto(list)?;
    Ok(serde_json::to_vec_pretty(&dto)?)
}

/// JSON-decode a manifest list. Verifies major-version
/// compatibility; allows unknown minor versions.
pub fn decode(bytes: &[u8]) -> Result<Manifest, ListParseError> {
    let dto: ManifestDto = serde_json::from_slice(bytes)?;
    list_from_dto(dto)
}

fn check_major(fv: &str) -> Result<(), ListParseError> {
    let supported_major = FORMAT_VERSION
        .split('.')
        .next()
        .expect("constant has a dot");
    let got_major = fv.split('.').next().unwrap_or("");
    if got_major != supported_major {
        return Err(ListParseError::IncompatibleMajorVersion {
            got: fv.to_string(),
            supported: FORMAT_VERSION.to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! JSON round-trip tests for `Manifest`.
    //!
    //! Covers: empty / N-entry round-trip; every
    //! `PartitionStrategy` variant; aggregate skip summaries
    //! survive round-trip including the `term_range_union:
    //! None` "field absent in JSON" shape; schema bytes
    //! round-trip via base64 (binary-safe); same logical
    //! content → byte-equal JSON (the property
    //! content-addressing rides on); format_version
    //! major/minor compat; part reuse across versions
    //! decodes to bit-equal entries; top-level JSON keys are
    //! jq-friendly.
    use std::{
        collections::{BTreeMap, HashMap},
        str::from_utf8,
        sync::Arc,
    };

    use arrow_array::{BinaryArray, BooleanArray, Int64Array, StringArray};
    use arrow_schema::{DataType, Field};
    use uuid::Uuid;

    use super::{
        super::{
            ClusterCentroids,
            bloom::BloomBuilder,
            part::{ContentHash, PartId},
        },
        *,
    };

    /// Build a per-column aggregate from a plain `i64` array (no nulls).
    fn agg_i64(vals: Vec<i64>) -> ScalarStatsAgg {
        let arr: ArrayRef = Arc::new(Int64Array::from(vals));
        ScalarStatsAgg::from_column(&arr).expect("i64 is orderable")
    }

    /// Read the single value out of a length-1 `Int64` aggregate array.
    fn i64_at0(arr: &ArrayRef) -> i64 {
        arr.as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 array")
            .value(0)
    }

    /// `birth_version_range` reads the UInt64 min/max of the birth-version
    /// aggregate; an entry lacking that aggregate has no range.
    #[test]
    fn birth_version_range_reads_uint64_bounds() {
        use arrow_array::UInt64Array;
        let mut entry = rich_entry(1);
        let mut stats = HashMap::new();
        stats.insert(
            BIRTH_VERSION_AGGREGATE_COLUMN.to_string(),
            ScalarStatsAgg {
                min: Arc::new(UInt64Array::from(vec![10u64])) as ArrayRef,
                max: Arc::new(UInt64Array::from(vec![20u64])) as ArrayRef,
                null_count: None,
                sum: None,
                hll: None,
                value_counts: None,
            },
        );
        entry.scalar_stats_agg = stats;
        assert_eq!(entry.birth_version_range(), Some((10, 20)));

        let mut bare = rich_entry(2);
        bare.scalar_stats_agg = HashMap::new();
        assert_eq!(bare.birth_version_range(), None);
    }

    #[test]
    fn scalar_agg_from_column_computes_min_max_sum_nullcount() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![Some(3), None, Some(7), Some(1)]));
        let agg = ScalarStatsAgg::from_column(&arr).expect("orderable");
        assert_eq!(i64_at0(&agg.min), 1);
        assert_eq!(i64_at0(&agg.max), 7);
        assert_eq!(agg.null_count, Some(1));
        assert_eq!(i64_at0(agg.sum.as_ref().expect("sum")), 11); // 3 + 7 + 1
        assert!(agg.hll.is_some());
        assert_eq!(
            agg.value_counts
                .as_ref()
                .expect("low-cardinality counts")
                .entries(),
            &[
                (ScalarValue::Int64(Some(1)), 1),
                (ScalarValue::Int64(Some(3)), 1),
                (ScalarValue::Int64(Some(7)), 1),
            ]
        );
    }

    #[test]
    fn scalar_value_counts_are_exact_and_capped() {
        let repeated: ArrayRef = Arc::new(StringArray::from(vec![
            Some("rust"),
            Some("go"),
            Some("rust"),
            None,
        ]));
        let counts = ScalarStatsAgg::from_column(&repeated)
            .expect("utf8 stats")
            .value_counts
            .expect("under cap");
        assert_eq!(
            counts.entries(),
            &[
                (ScalarValue::Utf8(Some("go".into())), 1),
                (ScalarValue::Utf8(Some("rust".into())), 2),
            ]
        );

        let high_cardinality: ArrayRef = Arc::new(Int64Array::from_iter_values(
            0..=MAX_EXACT_VALUE_COUNTS as i64,
        ));
        assert!(
            ScalarStatsAgg::from_column(&high_cardinality)
                .expect("int stats")
                .value_counts
                .is_none(),
            "the 257th distinct value must drop the exact table"
        );
    }

    #[test]
    fn scalar_agg_from_batch_builds_each_column() {
        let schema = Schema::new(vec![
            Field::new("x", DataType::Int64, true),
            Field::new("y", DataType::Int64, true),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(vec![3, 7, 1])) as ArrayRef,
                Arc::new(Int64Array::from(vec![20, 5, 9])) as ArrayRef,
            ],
        )
        .expect("batch");
        let table = ScalarStatsAgg::from_batch(&schema, &batch);
        assert_eq!(table.len(), 2);
        assert_eq!(i64_at0(&table["x"].min), 1);
        assert_eq!(i64_at0(&table["x"].max), 7);
        assert_eq!(i64_at0(&table["y"].min), 5);
        assert_eq!(i64_at0(&table["y"].max), 20);
    }

    #[test]
    fn scalar_agg_from_batches_concats_then_aggregates() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int64, true)]);
        let b1 = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![10, 50])) as ArrayRef],
        )
        .expect("b1");
        let b2 = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![5, 200])) as ArrayRef],
        )
        .expect("b2");
        let table = ScalarStatsAgg::from_batches(&schema, &[&b1, &b2]);
        assert_eq!(i64_at0(&table["x"].min), 5);
        assert_eq!(i64_at0(&table["x"].max), 200);
        assert_eq!(i64_at0(table["x"].sum.as_ref().expect("sum")), 265); // 10+50+5+200

        // Empty input yields an empty table.
        assert!(ScalarStatsAgg::from_batches(&schema, &[]).is_empty());
    }

    #[test]
    fn scalar_agg_merge_keeps_extremes_and_adds_additive() {
        let mut a = agg_i64(vec![10, 50]); // min 10, max 50, sum 60, nulls 0
        let b = agg_i64(vec![5, 30]); // min 5,  max 30, sum 35, nulls 0
        a.merge_with(&b).expect("same type merges");
        assert_eq!(i64_at0(&a.min), 5);
        assert_eq!(i64_at0(&a.max), 50);
        assert_eq!(i64_at0(a.sum.as_ref().expect("sum")), 95); // 60 + 35
        assert_eq!(a.null_count, Some(0));
        assert!(a.hll.is_some());
        assert_eq!(
            a.value_counts
                .as_ref()
                .expect("merged counts")
                .entries()
                .len(),
            4
        );
    }

    #[test]
    fn scalar_agg_merge_drops_value_counts_above_cap() {
        let left: ArrayRef = Arc::new(Int64Array::from_iter_values(
            0..=(MAX_EXACT_VALUE_COUNTS / 2) as i64,
        ));
        let right: ArrayRef = Arc::new(Int64Array::from_iter_values(
            (MAX_EXACT_VALUE_COUNTS / 2 + 1) as i64..=MAX_EXACT_VALUE_COUNTS as i64,
        ));
        let mut left = ScalarStatsAgg::from_column(&left).expect("left stats");
        let right = ScalarStatsAgg::from_column(&right).expect("right stats");
        left.merge_with(&right).expect("same type");
        assert!(left.value_counts.is_none());
    }

    #[test]
    fn scalar_agg_merge_drops_additive_when_one_side_missing() {
        let mut a = agg_i64(vec![1, 2]);
        let mut b = agg_i64(vec![3, 4]);
        // Simulate a contributor that never computed the additive stats.
        b.sum = None;
        b.null_count = None;
        b.hll = None;
        b.value_counts = None;
        a.merge_with(&b).expect("same type merges");
        // min/max still merge (union semantics over the bounds).
        assert_eq!(i64_at0(&a.min), 1);
        assert_eq!(i64_at0(&a.max), 4);
        // Additive stats become unknowable when any contributor lacks them.
        assert!(a.sum.is_none());
        assert!(a.null_count.is_none());
        assert!(a.hll.is_none());
        assert!(a.value_counts.is_none());
    }

    #[test]
    fn merge_tables_unions_columns_and_merges_shared() {
        let mut t1: HashMap<String, ScalarStatsAgg> = HashMap::new();
        t1.insert("a".into(), agg_i64(vec![10, 50]));

        let mut t2: HashMap<String, ScalarStatsAgg> = HashMap::new();
        t2.insert("a".into(), agg_i64(vec![5, 30]));
        t2.insert("b".into(), agg_i64(vec![100, 200]));

        ScalarStatsAgg::merge(&mut t1, &t2);
        assert_eq!(t1.len(), 2);
        // Shared column "a" is merged per-column (extremes kept).
        assert_eq!(i64_at0(&t1["a"].min), 5);
        assert_eq!(i64_at0(&t1["a"].max), 50);
        // Column "b", present only in t2, is inserted.
        assert_eq!(i64_at0(&t1["b"].min), 100);
        assert_eq!(i64_at0(&t1["b"].max), 200);
    }

    // ---- from_column: per-type branch coverage ----

    #[test]
    fn scalar_agg_from_column_utf8_has_minmax_and_hll_but_no_sum() {
        // Utf8 is orderable (min/max) and hashable (HLL) but not summable.
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["alpha", "delta", "bravo"]));
        let agg = ScalarStatsAgg::from_column(&arr).expect("utf8 is orderable");
        assert_eq!(agg.min.len(), 1);
        assert_eq!(agg.max.len(), 1);
        assert!(agg.sum.is_none(), "utf8 is not summable");
        assert!(agg.hll.is_some(), "utf8 supports HLL");
        assert_eq!(agg.null_count, Some(0));
    }

    #[test]
    fn scalar_agg_from_column_boolean_has_no_sum_no_hll() {
        // Boolean has min/max but neither a sum nor an HLL sketch.
        let arr: ArrayRef = Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)]));
        let agg = ScalarStatsAgg::from_column(&arr).expect("bool is orderable");
        assert!(agg.sum.is_none(), "bool not summable");
        assert!(agg.hll.is_none(), "bool not in the HLL type set");
        assert_eq!(agg.null_count, Some(1));
    }

    #[test]
    fn scalar_agg_from_column_unorderable_type_is_none() {
        // Binary has no meaningful ordering, so it isn't in `column_min_max`'s
        // supported set → no stats at all. (Temporal types now *are* supported;
        // see `min_max_stats_cover_temporal_columns` in the parent module.)
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![&b"a"[..], b"b", b"c"]));
        assert!(ScalarStatsAgg::from_column(&arr).is_none());
    }

    // ---- from_batches: skip branches ----

    #[test]
    fn scalar_agg_from_batches_skips_unorderable_column() {
        let schema = Schema::new(vec![Field::new("d", DataType::Binary, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(BinaryArray::from(vec![&b"a"[..], b"b"])) as ArrayRef],
        )
        .expect("batch");
        let table = ScalarStatsAgg::from_batches(&schema, &[&batch]);
        assert!(table.is_empty(), "unorderable column yields no entry");
    }

    #[test]
    fn scalar_agg_from_batches_skips_column_on_concat_type_mismatch() {
        // Two batches whose column 0 differ in type → concat fails → the
        // column is skipped (the `Err(_) => continue` branch).
        let schema = Schema::new(vec![Field::new("x", DataType::Int64, true)]);
        let b1 = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)])),
            vec![Arc::new(Int64Array::from(vec![1])) as ArrayRef],
        )
        .expect("b1");
        let b2 = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec!["a"])) as ArrayRef],
        )
        .expect("b2");
        let table = ScalarStatsAgg::from_batches(&schema, &[&b1, &b2]);
        assert!(table.is_empty(), "concat type mismatch skips the column");
    }

    #[test]
    fn scalar_agg_from_batches_skips_column_missing_from_a_batch() {
        // The schema names two columns, but the second batch carries only
        // the first. The lookup for column index 1 must skip, not panic via
        // `RecordBatch::column`.
        let schema = Schema::new(vec![
            Field::new("x", DataType::Int64, true),
            Field::new("y", DataType::Int64, true),
        ]);
        let b1 = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
                Arc::new(Int64Array::from(vec![3, 4])) as ArrayRef,
            ],
        )
        .expect("b1");
        let b2 = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)])),
            vec![Arc::new(Int64Array::from(vec![5])) as ArrayRef],
        )
        .expect("b2");
        let table = ScalarStatsAgg::from_batches(&schema, &[&b1, &b2]);
        // "x" is in both batches → aggregated; "y" is absent from b2 → skipped.
        assert!(table.contains_key("x"));
        assert!(
            !table.contains_key("y"),
            "a column missing from a batch is skipped, not panicked"
        );
    }

    // ---- merge: per-field branch coverage ----

    #[test]
    fn scalar_agg_merge_type_mismatch_errors_and_leaves_self_unchanged() {
        let mut a = agg_i64(vec![10, 50]); // Int64 bounds + sum
        let sum_present_before = a.sum.is_some();
        let b = {
            let arr: ArrayRef = Arc::new(StringArray::from(vec!["m", "z"]));
            ScalarStatsAgg::from_column(&arr).expect("utf8")
        };
        // Incompatible min/max types: merge must error rather than silently
        // keep stale bounds (which could cause a false prune).
        assert!(a.merge_with(&b).is_err(), "incompatible types must error");
        // `self` is left fully untouched — no half-merged state.
        assert_eq!(i64_at0(&a.min), 10);
        assert_eq!(i64_at0(&a.max), 50);
        assert_eq!(
            a.sum.is_some(),
            sum_present_before,
            "additive stats untouched on error"
        );
    }

    #[test]
    fn scalar_agg_merge_sum_overflow_drops_sum() {
        let mut a = agg_i64(vec![i64::MAX]); // sum = i64::MAX
        let b = agg_i64(vec![1]); // sum = 1
        a.merge_with(&b).expect("same type merges");
        assert!(a.sum.is_none(), "i64 sum overflow → None");
        // min/max still merge correctly.
        assert_eq!(i64_at0(&a.min), 1);
        assert_eq!(i64_at0(&a.max), i64::MAX);
    }

    #[test]
    fn scalar_agg_merge_invalid_hll_bytes_drops_hll() {
        let mut a = agg_i64(vec![1, 2]); // valid HLL
        let mut b = agg_i64(vec![3, 4]);
        b.hll = Some(vec![1, 2, 3]); // not a valid sketch
        a.merge_with(&b).expect("same type merges");
        assert!(a.hll.is_none(), "unparseable HLL bytes → None");
    }

    #[test]
    fn scalar_agg_merge_null_count_overflow_drops() {
        let mut a = agg_i64(vec![1]);
        a.null_count = Some(u64::MAX);
        let mut b = agg_i64(vec![2]);
        b.null_count = Some(1);
        a.merge_with(&b).expect("same type merges");
        assert!(a.null_count.is_none(), "null_count overflow → None");
    }

    #[test]
    fn merge_tables_keeps_columns_only_in_self() {
        let mut t1: HashMap<String, ScalarStatsAgg> = HashMap::new();
        t1.insert("a".into(), agg_i64(vec![1, 5]));
        t1.insert("c".into(), agg_i64(vec![7, 9]));
        let mut t2: HashMap<String, ScalarStatsAgg> = HashMap::new();
        t2.insert("a".into(), agg_i64(vec![0, 3]));

        ScalarStatsAgg::merge(&mut t1, &t2);
        // "c" exists only in self → untouched.
        assert_eq!(i64_at0(&t1["c"].min), 7);
        assert_eq!(i64_at0(&t1["c"].max), 9);
        // "a" merged.
        assert_eq!(i64_at0(&t1["a"].min), 0);
        assert_eq!(i64_at0(&t1["a"].max), 5);
    }

    #[test]
    fn merge_tables_drops_shared_column_on_type_mismatch() {
        // Same column name, incompatible Arrow types across the two tables.
        // The column must be dropped (→ "no info", always keep) rather than
        // kept with stale, under-covering bounds.
        let mut t1: HashMap<String, ScalarStatsAgg> = HashMap::new();
        t1.insert("x".into(), agg_i64(vec![1, 10]));
        let mut t2: HashMap<String, ScalarStatsAgg> = HashMap::new();
        let utf8: ArrayRef = Arc::new(StringArray::from(vec!["a", "z"]));
        t2.insert(
            "x".into(),
            ScalarStatsAgg::from_column(&utf8).expect("utf8"),
        );

        ScalarStatsAgg::merge(&mut t1, &t2);
        assert!(
            !t1.contains_key("x"),
            "type-mismatched column is dropped, not kept with stale bounds"
        );
    }

    // ---- PartialEq: every inequality branch ----

    #[test]
    fn scalar_agg_partial_eq_detects_each_field() {
        let base = agg_i64(vec![1, 10]);
        assert_eq!(base, base.clone());

        let mut d = base.clone();
        d.min = Arc::new(Int64Array::from(vec![0])) as ArrayRef;
        assert_ne!(base, d, "min differs");

        let mut d = base.clone();
        d.max = Arc::new(Int64Array::from(vec![999])) as ArrayRef;
        assert_ne!(base, d, "max differs");

        let mut d = base.clone();
        d.null_count = Some(42);
        assert_ne!(base, d, "null_count differs");

        let mut d = base.clone();
        d.sum = None;
        assert_ne!(base, d, "sum Some vs None");

        let mut d = base.clone();
        d.sum = Some(Arc::new(Int64Array::from(vec![123])) as ArrayRef);
        assert_ne!(base, d, "sum Some vs different Some");

        let mut d = base.clone();
        d.hll = Some(vec![9, 9, 9, 9]);
        assert_ne!(base, d, "hll differs");

        // Both sides with sum == None compare equal on that field.
        let mut a = base.clone();
        a.sum = None;
        let mut b = base.clone();
        b.sum = None;
        assert_eq!(a, b, "both sum None → equal");
    }

    // ---- encode / decode error propagation through the list ----

    #[test]
    fn encode_rejects_non_single_row_scalar_agg() {
        // A min array with more than one row violates the length-1 contract;
        // encode must surface ListEncodeError::ScalarStats, not panic.
        let mut list = empty_list();
        let mut entry = rich_entry(1);
        let bad: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        entry
            .scalar_stats_agg
            .get_mut("ts")
            .expect("ts present")
            .min = bad;
        list.parts = vec![entry];
        let err = encode(&list).expect_err("non-length-1 min must fail encode");
        assert!(
            matches!(
                err,
                ListEncodeError::ScalarStats {
                    field: "scalar_stats_agg.min",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn decode_surfaces_scalar_stats_error_on_corrupt_min() {
        // Encode a valid list, then replace one column's `min` base64 with
        // valid base64 of non-IPC bytes; decode must surface a typed error.
        let list = rich_list(1);
        let bytes = encode(&list).expect("encode");
        let mut v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        let cols: Vec<String> = v["parts"][0]["scalar_stats_agg"]
            .as_object()
            .expect("scalar_stats_agg object")
            .keys()
            .cloned()
            .collect();
        let key = cols.first().expect("at least one scalar column");
        let garbage_b64 = BASE64.encode(b"not arrow ipc");
        v["parts"][0]["scalar_stats_agg"][key.as_str()]["min"] =
            serde_json::Value::String(garbage_b64);
        let tampered = serde_json::to_vec(&v).expect("reserialize");
        let err = decode(&tampered).expect_err("corrupt min must fail decode");
        assert!(
            matches!(
                err,
                ListParseError::ScalarStats {
                    field: "scalar_stats_agg.min",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    fn empty_list() -> Manifest {
        Manifest {
            drained_ranges: Default::default(),
            global_vector_index: None,
            tombstone_seqs: Default::default(),
            format_version: FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: Vec::new(),
            id_column: "doc_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "doc_id".into(),
                n_buckets: 64,
            },
            vector_index_storage_prefix: None,
            deleted_user_ids_inline: None,
            slow_vector_state_uri: None,
            slow_vector_state_content_hash: None,
            slow_vector_state_centroids: None,
            parts: vec![],
        }
    }

    fn rich_entry(seed: u8) -> ManifestPartEntry {
        // Several columns (inserted out of sorted order) so the JSON
        // round-trip and byte-equality tests actually exercise the
        // HashMap → BTreeMap re-sort that keeps the wire form
        // deterministic for content-addressing.
        let mut scalar = HashMap::new();
        for col in ["ts", "amount", "_id"] {
            scalar.insert(
                col.to_string(),
                ScalarStatsAgg {
                    min: Arc::new(Int64Array::from(vec![i64::from(seed)])) as ArrayRef,
                    max: Arc::new(Int64Array::from(vec![i64::from(seed) + 1_000])) as ArrayRef,
                    null_count: Some(u64::from(seed)),
                    sum: Some(Arc::new(Int64Array::from(vec![i64::from(seed) * 7])) as ArrayRef),
                    hll: Some(vec![seed; 4]),
                    value_counts: ScalarValueCounts::from_entries(vec![(
                        ScalarValue::Int64(Some(i64::from(seed))),
                        1,
                    )]),
                },
            );
        }

        let mut fts = BTreeMap::new();
        let mut title_bloom = BloomBuilder::with_n_blocks(16);
        title_bloom.insert(format!("title_{seed}").as_bytes());
        fts.insert(
            "title".into(),
            FtsSummaryAgg {
                term_bloom: Some(title_bloom.finish()),
                n_terms_distinct: 1_048_576,
                term_range: Some((b"alpha".to_vec(), b"zulu".to_vec())),
            },
        );
        // "body": no bloom info, no range (the all-None / always-keep shape).
        fts.insert(
            "body".into(),
            FtsSummaryAgg {
                term_bloom: None,
                n_terms_distinct: 0,
                term_range: None,
            },
        );

        ManifestPartEntry {
            part_id: PartId(Uuid::from_bytes([seed; 16])),
            uri: format!("manifests/part-{seed:02x}.avro.zst"),
            n_superfiles: 9_847,
            size_bytes_compressed: 10_485_760,
            size_bytes_uncompressed: 26_214_400,
            content_hash: ContentHash([seed; 32]),
            // Alternate present/absent so the round-trip covers both the
            // stamped sibling and the pre-sibling (legacy) shape.
            routing: (seed % 2 == 1).then(|| RoutingRef {
                uri: format!("manifests/part-{seed:02x}-routing.avro.zst"),
                content_hash: ContentHash([seed ^ 0xff; 32]),
            }),
            id_range: (0, 245_678_901),
            scalar_stats_agg: scalar,
            fts_summary_agg: fts,
        }
    }

    fn rich_list(n_parts: u8) -> Manifest {
        let mut list = empty_list();
        list.manifest_id = 42;
        list.options_hash = ContentHash([0xab; 32]);
        list.schema = vec![0x01, 0x02, 0x03, 0xff, 0xfe];
        list.fts_columns = vec![
            FtsColumnInfo {
                column: "title".into(),
            },
            FtsColumnInfo {
                column: "body".into(),
            },
        ];
        list.vector_columns = vec![VectorColumnInfo {
            column: "emb".into(),
            dim: 384,
            n_cent: 64,
            rot_seed: 7,
            metric: "cosine".into(),
        }];
        list.partition_strategy = PartitionStrategy::TimeRange {
            column: "ts".into(),
            granularity_secs: 86_400,
        };
        list.parts = (0..n_parts).map(rich_entry).collect();
        list
    }

    fn assert_entries_equal(a: &ManifestPartEntry, b: &ManifestPartEntry) {
        assert_eq!(a.part_id, b.part_id, "part_id");
        assert_eq!(a.uri, b.uri, "uri");
        assert_eq!(a.n_superfiles, b.n_superfiles, "n_superfiles");
        assert_eq!(
            a.size_bytes_compressed, b.size_bytes_compressed,
            "size_bytes_compressed"
        );
        assert_eq!(
            a.size_bytes_uncompressed, b.size_bytes_uncompressed,
            "size_bytes_uncompressed"
        );
        assert_eq!(a.content_hash, b.content_hash, "content_hash");
        assert_eq!(a.routing, b.routing, "routing");
        assert_eq!(a.id_range, b.id_range, "id_range");
        assert_eq!(a.scalar_stats_agg, b.scalar_stats_agg, "scalar_stats_agg");
        assert_eq!(a.fts_summary_agg, b.fts_summary_agg, "fts_summary_agg");
    }

    fn assert_lists_equal(a: &Manifest, b: &Manifest) {
        assert_eq!(a.format_version, b.format_version);
        assert_eq!(a.manifest_id, b.manifest_id);
        assert_eq!(a.options_hash, b.options_hash);
        assert_eq!(a.schema, b.schema);
        assert_eq!(a.id_column, b.id_column);
        assert_eq!(a.fts_columns, b.fts_columns);
        assert_eq!(a.vector_columns, b.vector_columns);
        assert_eq!(a.vector_index_storage_prefix, b.vector_index_storage_prefix);
        assert_eq!(a.partition_strategy, b.partition_strategy);
        assert_eq!(a.parts.len(), b.parts.len());
        for (a_e, b_e) in a.parts.iter().zip(b.parts.iter()) {
            assert_entries_equal(a_e, b_e);
        }
    }

    #[test]
    fn empty_list_roundtrip() {
        let list = empty_list();
        let bytes = encode(&list).expect("encode");
        let decoded = decode(&bytes).expect("decode");
        assert_lists_equal(&decoded, &list);
    }

    #[test]
    fn rich_list_roundtrip_multiple_parts() {
        let list = rich_list(5);
        let bytes = encode(&list).expect("encode");
        let decoded = decode(&bytes).expect("decode");
        assert_lists_equal(&decoded, &list);
    }

    #[test]
    fn tombstone_seqs_roundtrip() {
        let mut list = empty_list();
        list.tombstone_seqs.insert(Uuid::from_u128(0x42), 7);
        list.tombstone_seqs.insert(Uuid::from_u128(0x43), 9);
        let bytes = encode(&list).expect("encode");
        let decoded = decode(&bytes).expect("decode");
        assert_eq!(decoded.tombstone_seqs, list.tombstone_seqs);
    }

    #[test]
    fn partition_strategy_time_range_roundtrip() {
        let mut list = empty_list();
        list.partition_strategy = PartitionStrategy::TimeRange {
            column: "event_ts".into(),
            granularity_secs: 3600,
        };
        let bytes = encode(&list).expect("encode");
        let decoded = decode(&bytes).expect("decode");
        assert_eq!(decoded.partition_strategy, list.partition_strategy);
    }

    #[test]
    fn partition_strategy_hash_roundtrip() {
        let mut list = empty_list();
        list.partition_strategy = PartitionStrategy::Hash {
            column: "user_id".into(),
            n_buckets: 1024,
        };
        let bytes = encode(&list).expect("encode");
        let decoded = decode(&bytes).expect("decode");
        assert_eq!(decoded.partition_strategy, list.partition_strategy);
    }

    /// An explicitly-persisted `slack = 0.0` must survive the round-trip:
    /// the DTO once used zero as its "field absent" sentinel, so reopening
    /// a manifest silently widened the probe threshold back to the default
    /// slack — a routing behavior change on tables that intentionally
    /// persist no near-tie widening. Absent stays default.
    #[test]
    fn cell_routing_params_zero_slack_roundtrip() {
        let mut list = empty_list();
        list.partition_strategy = PartitionStrategy::VectorCell {
            column: "emb".into(),
            clusters: ClusterCentroids::from_fp32(1, 4, &[0.5, 0.5, 0.5, 0.5], vec![1]),
            routing: CellRoutingParams {
                slack: 0.0,
                ..CellRoutingParams::default()
            },
        };
        let bytes = encode(&list).expect("encode");
        let decoded = decode(&bytes).expect("decode");
        let PartitionStrategy::VectorCell { routing, .. } = &decoded.partition_strategy else {
            panic!("VectorCell strategy must survive the round-trip");
        };
        assert_eq!(
            routing.slack, 0.0,
            "explicit zero slack must not decode back to the default"
        );
        // A JSON body without the field still lands on the default.
        let s = from_utf8(&bytes).expect("utf8");
        let stripped = s.replace("\"slack\": 0.0,", "");
        assert_ne!(stripped, s, "fixture must actually strip the field");
        let legacy = decode(stripped.as_bytes()).expect("decode without slack");
        let PartitionStrategy::VectorCell { routing, .. } = &legacy.partition_strategy else {
            panic!("VectorCell strategy must survive the stripped decode");
        };
        assert_eq!(
            routing.slack,
            CellRoutingParams::default().slack,
            "absent slack keeps the default"
        );
    }

    #[test]
    fn partition_strategy_column_range_roundtrip() {
        let mut list = empty_list();
        list.partition_strategy = PartitionStrategy::ColumnRange {
            column: "category".into(),
            boundaries: vec![
                vec![0x01, 0x02, 0x03],
                vec![0xff, 0xfe, 0xfd, 0xfc],
                vec![0x00, 0x80, 0xff],
            ],
        };
        let bytes = encode(&list).expect("encode");
        let decoded = decode(&bytes).expect("decode");
        assert_eq!(decoded.partition_strategy, list.partition_strategy);
    }

    #[test]
    fn partition_strategy_vector_cell_roundtrip() {
        use super::super::ClusterCentroids;
        let mut list = empty_list();
        list.partition_strategy = PartitionStrategy::VectorCell {
            column: "emb".into(),
            clusters: ClusterCentroids::from_fp32(
                2,
                4,
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                vec![1, 1],
            ),
            routing: CellRoutingParams::default(),
        };
        let bytes = encode(&list).expect("encode");
        let decoded = decode(&bytes).expect("decode");
        assert_eq!(decoded.partition_strategy, list.partition_strategy);
    }

    #[test]
    fn global_vector_index_roundtrip() {
        use super::super::ClusterCentroids;
        let mut list = empty_list();
        list.global_vector_index = Some(GlobalVectorIndex {
            column: "emb".into(),
            grid: ClusterCentroids::from_fp32(
                2,
                4,
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                vec![3, 5],
            ),
            user_grid: None,
        });
        let bytes = encode(&list).expect("encode");
        let decoded = decode(&bytes).expect("decode");
        assert_eq!(decoded.global_vector_index, list.global_vector_index);
        // With no user grid, the user side falls back to the drain grid.
        let fallback = decoded
            .global_vector_index
            .clone()
            .expect("index present")
            .into_user_grid();
        assert_eq!(
            Some(&fallback),
            list.global_vector_index.as_ref().map(|g| &g.grid)
        );

        // Finer user-side grid rides along and decodes distinct from `grid`.
        let user_grid = ClusterCentroids::from_fp32(1, 4, &[8.0, 9.0, 10.0, 11.0], vec![7]);
        if let Some(g) = list.global_vector_index.as_mut() {
            g.user_grid = Some(user_grid.clone());
        }
        let bytes = encode(&list).expect("encode with user grid");
        let decoded = decode(&bytes).expect("decode with user grid");
        assert_eq!(decoded.global_vector_index, list.global_vector_index);
        assert_eq!(
            *decoded
                .global_vector_index
                .as_ref()
                .expect("index present")
                .user_grid(),
            user_grid
        );
        // Absent by default (back-compat: old manifests without the field).
        assert!(empty_list().global_vector_index.is_none());
    }

    #[test]
    fn slow_vector_state_ref_roundtrip() {
        let mut list = empty_list();
        list.slow_vector_state_uri = Some("slow-vector-state/state-abc.bin".into());
        list.slow_vector_state_content_hash = Some(ContentHash([7u8; 32]));
        list.slow_vector_state_centroids = Some(RoutingRef {
            uri: "slow-vector-state/state-def.centroids.bin".into(),
            content_hash: ContentHash([8u8; 32]),
        });
        let bytes = encode(&list).expect("encode");
        let decoded = decode(&bytes).expect("decode");
        assert_eq!(decoded.slow_vector_state_uri, list.slow_vector_state_uri);
        assert_eq!(
            decoded.slow_vector_state_content_hash,
            list.slow_vector_state_content_hash
        );
        assert_eq!(
            decoded.slow_vector_state_centroids,
            list.slow_vector_state_centroids
        );
        // Absent by default (user tables and old manifests without the field).
        let plain = empty_list();
        let bytes = encode(&plain).expect("encode");
        assert!(
            !from_utf8(&bytes)
                .expect("utf8")
                .contains("slow_vector_state_centroids"),
            "absent section ref must not serialize null fields"
        );
        let decoded = decode(&bytes).expect("decode");
        assert!(decoded.slow_vector_state_uri.is_none());
        assert!(decoded.slow_vector_state_content_hash.is_none());
        assert!(decoded.slow_vector_state_centroids.is_none());
    }

    /// A slow-state hash that isn't `blake3:<64hex>` is rejected with
    /// `BadContentHash` (same `decode_hash` used by every other hash field).
    #[test]
    fn slow_vector_state_bad_hash_rejected() {
        let mut list = empty_list();
        list.slow_vector_state_uri = Some("slow-vector-state/state-abc.bin".into());
        list.slow_vector_state_content_hash = Some(ContentHash([9u8; 32]));
        let bytes = encode(&list).expect("encode");
        let s = from_utf8(&bytes).expect("utf8");
        let full = "09".repeat(BLAKE3_DIGEST_BYTES);
        let tampered = s.replacen(&format!("blake3:{full}"), "blake3:xyz", 1);
        assert_ne!(tampered, s, "tamper must change the bytes");
        let err = decode(tampered.as_bytes()).expect_err("bad slow-state hash");
        assert!(
            matches!(err, ListParseError::BadContentHash(_)),
            "expected BadContentHash, got {err:?}"
        );
    }

    #[test]
    fn drained_version_ranges_merge_and_contains() {
        let mut d = DrainedVersionRanges::default();
        assert!(!d.contains(1));
        // Contiguous inserts merge into one interval.
        d.insert(1);
        d.insert(2);
        d.insert(3);
        assert_eq!(d.intervals(), &[(1, 3)]);
        // A gap stays split.
        d.insert(5);
        assert_eq!(d.intervals(), &[(1, 3), (5, 5)]);
        assert!(d.contains(2) && d.contains(5) && !d.contains(4));
        // Filling the gap merges everything.
        d.insert(4);
        assert_eq!(d.intervals(), &[(1, 5)]);
        // Out-of-order inserts (the parallel-drainer case) coalesce as holes fill.
        d.insert_range(10, 12);
        d.insert(8);
        assert_eq!(d.intervals(), &[(1, 5), (8, 8), (10, 12)]);
        d.insert(9);
        assert_eq!(d.intervals(), &[(1, 5), (8, 12)]);
        // from_intervals normalizes unsorted/overlapping input.
        let d2 = DrainedVersionRanges::from_intervals(vec![(5, 7), (1, 3), (4, 4)])
            .expect("valid intervals");
        assert_eq!(d2.intervals(), &[(1, 7)]);
    }

    #[test]
    fn drained_ranges_prefix_absorbs_vacuous_gaps() {
        // The drain's lo logic: extend the genesis-anchored prefix over
        // commit-versions that have no superfile (deletes, etc.).
        let mut p = DrainedVersionRanges::default();
        assert_eq!(p.prefix_end(), None);
        p.insert_range(0, 3);
        assert_eq!(p.prefix_end(), Some(3));
        // Versions 4..6 are vacuous (no superfiles); next drained superfile is v7.
        // The drain inserts [prefix_end+1 ..= 7] = [4..=7], folding the gap.
        let lo = p.prefix_end().map(|e| e + 1).unwrap_or(0);
        p.insert_range(lo, 7);
        assert_eq!(
            p.intervals(),
            &[(0, 7)],
            "vacuous gap [4..6] must fold into the prefix, not fragment"
        );
        assert_eq!(p.prefix_end(), Some(7));
        // A set not anchored at genesis (e.g. a parallel high-slice) has no prefix.
        assert_eq!(
            DrainedVersionRanges::from_intervals(vec![(5, 7)])
                .expect("valid intervals")
                .prefix_end(),
            None
        );
    }

    #[test]
    fn drained_ranges_roundtrip() {
        let mut list = empty_list();
        list.drained_ranges =
            DrainedVersionRanges::from_intervals(vec![(1, 4), (7, 9)]).expect("valid intervals");
        let bytes = encode(&list).expect("encode");
        let decoded = decode(&bytes).expect("decode");
        assert_eq!(decoded.drained_ranges, list.drained_ranges);
        assert!(empty_list().drained_ranges.is_empty());
    }

    #[test]
    fn drained_ranges_cover_only_complete_intervals() {
        let ranges =
            DrainedVersionRanges::from_intervals(vec![(1, 4), (7, 9)]).expect("valid intervals");
        assert!(ranges.covers(2, 4));
        assert!(!ranges.covers(3, 7));
        assert!(!ranges.covers(5, 6));
    }

    #[test]
    fn term_range_union_none_is_absent_from_json() {
        // term_range_union is #[serde(skip_serializing_if =
        // "Option::is_none")], so None must produce field
        // absence, not `"term_range_union": null`. This is the
        // property cross-version content-addressing rides on.
        let list = rich_list(1);
        let bytes = encode(&list).expect("encode");
        let s = from_utf8(&bytes).expect("utf8");
        let body_fts = serde_json::from_slice::<serde_json::Value>(&bytes).expect("json");
        let fts_agg = &body_fts["parts"][0]["fts_summary_agg"]["body"];
        assert!(
            fts_agg.get("term_range_union").is_none(),
            "term_range_union must be absent in json when None; got body fts_agg = {body_fts:#}"
        );
        let title_agg = &body_fts["parts"][0]["fts_summary_agg"]["title"];
        assert!(title_agg.get("term_range_union").is_some());
        assert!(s.contains("\"term_bloom_union\""));
        let _ = decode(&bytes).expect("decode still works");
    }

    fn fts_agg(terms: &[&[u8]], n_blocks: usize, range: Option<(&[u8], &[u8])>) -> FtsSummaryAgg {
        let mut b = BloomBuilder::with_n_blocks(n_blocks);
        for t in terms {
            b.insert(t);
        }
        FtsSummaryAgg {
            term_bloom: Some(b.finish()),
            n_terms_distinct: terms.len() as u64,
            term_range: range.map(|(mn, mx)| (mn.to_vec(), mx.to_vec())),
        }
    }

    #[test]
    fn fts_agg_merge_unions_blooms_widens_range_and_takes_max_distinct() {
        let mut a = fts_agg(&[b"alpha"], 16, Some((b"alpha", b"mango")));
        a.n_terms_distinct = 3;
        let b = fts_agg(&[b"omega"], 16, Some((b"beta", b"zulu")));
        // b.n_terms_distinct == 1 (one term)
        a.merge_with(&b);

        let bloom = a.term_bloom.as_ref().expect("union bloom");
        assert!(
            bloom.contains(b"alpha"),
            "term from self survives the union"
        );
        assert!(bloom.contains(b"omega"), "term from other joins the union");
        // Range widened to span both: (min(alpha,beta), max(mango,zulu)).
        assert_eq!(a.term_range, Some((b"alpha".to_vec(), b"zulu".to_vec())));
        assert_eq!(a.n_terms_distinct, 3, "distinct hint takes the larger side");
    }

    #[test]
    fn fts_agg_merge_none_side_contributes_nothing() {
        // Some.merge_with(None) keeps self untouched.
        let mut a = fts_agg(&[b"x"], 16, Some((b"a", b"m")));
        a.merge_with(&FtsSummaryAgg::default());
        assert!(a.term_bloom.as_ref().expect("kept").contains(b"x"));
        assert_eq!(a.term_range, Some((b"a".to_vec(), b"m".to_vec())));

        // None.merge_with(Some) adopts the other side.
        let mut none_side = FtsSummaryAgg::default();
        none_side.merge_with(&fts_agg(&[b"y"], 16, Some((b"n", b"z"))));
        assert!(none_side.term_bloom.as_ref().expect("taken").contains(b"y"));
        assert_eq!(none_side.term_range, Some((b"n".to_vec(), b"z".to_vec())));
    }

    #[test]
    fn fts_agg_merge_bloom_shape_mismatch_drops_to_none() {
        // Different block counts can't be unioned → conservative "no info".
        let mut a = fts_agg(&[b"a"], 16, None);
        let b = fts_agg(&[b"b"], 8, None);
        a.merge_with(&b);
        assert!(
            a.term_bloom.is_none(),
            "shape mismatch → no bloom info (always-keep)"
        );
    }

    #[test]
    fn fts_agg_from_superfile_adapts_per_superfile_shape() {
        let mut b = BloomBuilder::with_n_blocks(16);
        b.insert(b"alpha");
        let agg = FtsSummaryAgg::new_with_params(b.finish(), 7, (b"a".to_vec(), b"z".to_vec()));
        assert!(
            agg.term_bloom
                .as_ref()
                .expect("bloom present")
                .contains(b"alpha")
        );
        assert_eq!(agg.n_terms_distinct, 7u64); // u32 → u64
        assert_eq!(agg.term_range, Some((b"a".to_vec(), b"z".to_vec())));

        // A 0-term column: empty (min, max) → `None` range, but a built bloom
        // is still present.
        let empty = FtsSummaryAgg::new_with_params(
            BloomBuilder::with_n_blocks(16).finish(),
            0,
            (Vec::new(), Vec::new()),
        );
        assert_eq!(empty.term_range, None);
        assert!(empty.term_bloom.is_some());
    }

    #[test]
    fn fts_agg_may_contain() {
        let mut b = BloomBuilder::with_n_blocks(16);
        b.insert(b"present");
        let agg = FtsSummaryAgg {
            term_bloom: Some(b.finish()),
            n_terms_distinct: 1,
            term_range: None,
        };
        assert!(agg.may_contain(b"present"));
        assert!(!agg.may_contain(b"definitely-absent-term"));
        // No bloom info → conservatively keep.
        assert!(FtsSummaryAgg::default().may_contain(b"anything"));
    }

    #[test]
    fn fts_agg_may_match_prefix() {
        let agg = FtsSummaryAgg {
            term_bloom: None,
            n_terms_distinct: 0,
            term_range: Some((b"bravo".to_vec(), b"mango".to_vec())),
        };
        assert!(
            agg.may_match_prefix(b"echo"),
            "prefix inside [bravo, mango]"
        );
        assert!(!agg.may_match_prefix(b"zulu"), "above max → no overlap");
        assert!(!agg.may_match_prefix(b"alpha"), "below min → no overlap");
        // No range (empty FST) → nothing matches → prune.
        assert!(!FtsSummaryAgg::default().may_match_prefix(b"echo"));
    }

    #[test]
    fn same_logical_content_produces_byte_equal_json() {
        // Two lists built from identical inputs must produce
        // bit-identical JSON output — the property cross-
        // version content-addressing rides on.
        let list_a = rich_list(3);
        let list_b = rich_list(3);
        let bytes_a = encode(&list_a).expect("encode a");
        let bytes_b = encode(&list_b).expect("encode b");
        assert_eq!(bytes_a, bytes_b, "byte-equal JSON for byte-equal input");
    }

    #[test]
    fn incompatible_major_version_rejected() {
        let mut list = empty_list();
        list.format_version = "2.0".into();
        let bytes = encode(&list).expect("encode");
        let err = decode(&bytes).expect_err("major 2 must reject");
        assert!(
            matches!(err, ListParseError::IncompatibleMajorVersion { .. }),
            "expected IncompatibleMajorVersion, got {err:?}"
        );
    }

    #[test]
    fn minor_version_compatible() {
        let mut list = empty_list();
        list.format_version = "1.99".into();
        let bytes = encode(&list).expect("encode");
        let decoded = decode(&bytes).expect("minor 99 must accept");
        assert_eq!(decoded.format_version, "1.99");
    }

    #[test]
    fn part_reuse_across_versions() {
        // Two manifest lists at different manifest_ids that
        // both reference the same entry must round-trip into
        // bit-equal entries — the property that lets readers
        // Arc::clone the part from the prior in-memory
        // ManifestSnapshot into the new one.
        let entry = rich_entry(99);

        let mut list_v42 = empty_list();
        list_v42.manifest_id = 42;
        list_v42.parts = vec![entry.clone()];

        let mut list_v43 = empty_list();
        list_v43.manifest_id = 43;
        list_v43.parts = vec![entry.clone()];

        let b_v42 = encode(&list_v42).expect("encode v42");
        let b_v43 = encode(&list_v43).expect("encode v43");
        let d_v42 = decode(&b_v42).expect("decode v42");
        let d_v43 = decode(&b_v43).expect("decode v43");

        assert_eq!(d_v42.parts.len(), 1);
        assert_eq!(d_v43.parts.len(), 1);
        assert_entries_equal(&d_v42.parts[0], &d_v43.parts[0]);
        assert_ne!(d_v42.manifest_id, d_v43.manifest_id);
    }

    #[test]
    fn json_top_level_keys_are_jq_friendly() {
        // ManifestSnapshot list is the operator's debugging surface;
        // the top-level keys are the contract.
        let list = rich_list(1);
        let bytes = encode(&list).expect("encode");
        let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        let obj = v.as_object().expect("object");
        let expected = [
            "format_version",
            "manifest_id",
            "options_hash",
            "schema",
            "id_column",
            "fts_columns",
            "vector_columns",
            "vector_index_storage_prefix",
            "partition_strategy",
            "parts",
        ];
        for key in expected {
            assert!(obj.contains_key(key), "missing top-level key {key}");
        }
        assert!(
            obj["options_hash"]
                .as_str()
                .unwrap_or("")
                .starts_with("blake3:"),
            "options_hash should be 'blake3:<hex>' for jq-debuggability"
        );
    }

    #[test]
    fn vector_index_storage_prefix_roundtrip() {
        let mut list = empty_list();
        list.vector_index_storage_prefix = Some("_infino_deadbeef_vector_index".into());
        let got = decode(&encode(&list).expect("encode")).expect("decode");
        assert_eq!(
            got.vector_index_storage_prefix,
            list.vector_index_storage_prefix
        );
    }

    #[test]
    fn binary_safe_schema_roundtrip() {
        // Arrow-IPC bytes contain arbitrary u8 — base64 must
        // preserve the full byte range in both directions.
        let mut list = empty_list();
        list.schema = (0u8..=255).collect();
        let bytes = encode(&list).expect("encode");
        let decoded = decode(&bytes).expect("decode");
        assert_eq!(decoded.schema, list.schema);
    }

    #[test]
    fn malformed_base64_surfaces_typed_error() {
        let list = rich_list(1);
        let bytes = encode(&list).expect("encode");
        let s = from_utf8(&bytes).expect("utf8");
        let tampered = s.replacen("\"schema\": \"", "\"schema\": \"!!!!", 1);
        let err = decode(tampered.as_bytes()).expect_err("must fail");
        assert!(
            matches!(err, ListParseError::Base64 { .. }),
            "expected Base64 error, got {err:?}"
        );
    }

    /// A non-empty `term_bloom_union` that doesn't decode to a valid
    /// `Bloom` layout is on-disk corruption: surface it as
    /// `InvalidBloom`, not a silent `None` that the pruner would read as
    /// a valid "always-keep" summary. (An empty string stays `None` —
    /// that's the legitimate no-bloom encoding, covered by round-trip.)
    #[test]
    fn malformed_term_bloom_surfaces_typed_error() {
        let list = rich_list(1);
        let bytes = encode(&list).expect("encode");
        let s = from_utf8(&bytes).expect("utf8");
        // rich_entry's "body" column has term_bloom = None ⇒ "". Swap it
        // for base64 of 3 bytes ("abc") — non-empty, but not a valid
        // `n_blocks × BLOCK_BYTES` bloom layout.
        let tampered = s.replacen(
            "\"term_bloom_union\": \"\"",
            "\"term_bloom_union\": \"YWJj\"",
            1,
        );
        assert_ne!(
            tampered, s,
            "test fixture must contain an empty bloom union"
        );
        let err = decode(tampered.as_bytes()).expect_err("malformed bloom");
        assert!(
            matches!(err, ListParseError::InvalidBloom(3)),
            "expected InvalidBloom(3), got {err:?}"
        );
    }

    /// A `content_hash` lacking the `blake3:` prefix is rejected with a
    /// `BadContentHash` error (`decode_hash`'s prefix-strip branch).
    #[test]
    fn options_hash_without_blake3_prefix_rejected() {
        let list = rich_list(1);
        let bytes = encode(&list).expect("encode");
        let s = from_utf8(&bytes).expect("utf8");
        // rich_list stamps options_hash = blake3:abab...; drop the prefix.
        let tampered = s.replacen("\"blake3:", "\"nothex:", 1);
        let err = decode(tampered.as_bytes()).expect_err("missing prefix");
        assert!(
            matches!(err, ListParseError::BadContentHash(_)),
            "expected BadContentHash, got {err:?}"
        );
    }

    /// A `content_hash` whose hex payload is the wrong length is rejected
    /// (`decode_hash`'s length-check branch).
    #[test]
    fn content_hash_wrong_hex_length_rejected() {
        let list = rich_list(1);
        let bytes = encode(&list).expect("encode");
        let s = from_utf8(&bytes).expect("utf8");
        // The first per-part content_hash is 64 hex chars of 'c' (seed 0
        // ⇒ ContentHash([0;32]) ⇒ all "00"). Shorten it to 2 chars.
        let full = "0".repeat(BLAKE3_HEX_LEN);
        let tampered = s.replacen(&format!("blake3:{full}"), "blake3:00", 1);
        assert_ne!(tampered, s, "tamper must change the bytes");
        let err = decode(tampered.as_bytes()).expect_err("short hash");
        assert!(
            matches!(err, ListParseError::BadContentHash(_)),
            "expected BadContentHash, got {err:?}"
        );
    }

    /// A non-numeric `id_range` value surfaces a `BadFieldValue` error
    /// (`entry_from_dto`'s `i128::parse` branch).
    #[test]
    fn non_numeric_id_range_rejected() {
        let list = rich_list(1);
        let bytes = encode(&list).expect("encode");
        let mut v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        v["parts"][0]["id_range"][0] = serde_json::Value::String("not-an-int".into());
        let tampered = serde_json::to_vec(&v).expect("reencode");
        let err = decode(&tampered).expect_err("bad id_range");
        assert!(
            matches!(err, ListParseError::BadFieldValue("id_range[0]", _)),
            "expected BadFieldValue, got {err:?}"
        );
    }

    /// The upper id_range bound is validated independently of the lower
    /// one (`entry_from_dto`'s second `i128::parse` branch).
    #[test]
    fn non_numeric_id_range_upper_bound_rejected() {
        let list = rich_list(1);
        let bytes = encode(&list).expect("encode");
        let mut v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        v["parts"][0]["id_range"][1] = serde_json::Value::String("xyz".into());
        let tampered = serde_json::to_vec(&v).expect("reencode");
        let err = decode(&tampered).expect_err("bad id_range upper");
        assert!(
            matches!(err, ListParseError::BadFieldValue("id_range[1]", _)),
            "expected BadFieldValue, got {err:?}"
        );
    }

    // ----- Tests for FtsSummaryAgg::merge_tables -----

    #[test]
    fn fts_merge_empty_into_and_empty_other() {
        let mut into = BTreeMap::new();
        let other = BTreeMap::new();
        FtsSummaryAgg::merge(&mut into, &other);
        assert!(into.is_empty());
    }

    #[test]
    fn fts_merge_empty_into_adopts_other() {
        let mut into = BTreeMap::new();
        let mut other = BTreeMap::new();
        let mut b = super::super::bloom::BloomBuilder::with_n_blocks(16);
        b.insert(b"test");
        let bloom = b.finish();
        let summary = FtsSummaryAgg {
            term_bloom: Some(bloom),
            n_terms_distinct: 42,
            term_range: Some((b"apple".to_vec(), b"zebra".to_vec())),
        };
        other.insert("col1".to_string(), summary.clone());
        FtsSummaryAgg::merge(&mut into, &other);
        assert_eq!(into.len(), 1);
        assert_eq!(into["col1"], summary);
    }

    #[test]
    fn fts_merge_preserves_columns_only_in_into() {
        let mut into = BTreeMap::new();
        let other = BTreeMap::new();
        let mut b = super::super::bloom::BloomBuilder::with_n_blocks(16);
        b.insert(b"test");
        let bloom = b.finish();
        let summary = FtsSummaryAgg {
            term_bloom: Some(bloom),
            n_terms_distinct: 10,
            term_range: Some((b"a".to_vec(), b"z".to_vec())),
        };
        into.insert("only_in_into".to_string(), summary.clone());
        FtsSummaryAgg::merge(&mut into, &other);
        assert_eq!(into.len(), 1);
        assert_eq!(into["only_in_into"], summary);
    }

    #[test]
    fn fts_merge_tables_merges_shared_columns() {
        let mut into = BTreeMap::new();
        let mut other = BTreeMap::new();
        let mut b1 = super::super::bloom::BloomBuilder::with_n_blocks(16);
        b1.insert(b"test1");
        let bloom1 = b1.finish();
        let mut b2 = super::super::bloom::BloomBuilder::with_n_blocks(16);
        b2.insert(b"test2");
        let bloom2 = b2.finish();
        let summary1 = FtsSummaryAgg {
            term_bloom: Some(bloom1),
            n_terms_distinct: 10,
            term_range: Some((b"apple".to_vec(), b"mango".to_vec())),
        };
        let summary2 = FtsSummaryAgg {
            term_bloom: Some(bloom2),
            n_terms_distinct: 15,
            term_range: Some((b"banana".to_vec(), b"zebra".to_vec())),
        };
        into.insert("shared".to_string(), summary1);
        other.insert("shared".to_string(), summary2);
        FtsSummaryAgg::merge(&mut into, &other);
        assert_eq!(into.len(), 1);
        // After merge: ranges should widen, distinct count should be max
        let merged = &into["shared"];
        assert_eq!(merged.n_terms_distinct, 15);
        assert_eq!(
            merged.term_range.as_ref().expect("should be present").0,
            b"apple"
        ); // min
        assert_eq!(
            merged.term_range.as_ref().expect("should be present").1,
            b"zebra"
        ); // max
    }

    #[test]
    fn fts_merge_drops_column_on_bloom_shape_mismatch() {
        let mut into = BTreeMap::new();
        let mut other = BTreeMap::new();
        let mut b1 = super::super::bloom::BloomBuilder::with_n_blocks(16);
        b1.insert(b"test1");
        let bloom1 = b1.finish();
        let mut b2 = super::super::bloom::BloomBuilder::with_n_blocks(8);
        b2.insert(b"test2");
        let bloom2 = b2.finish();
        let summary1 = FtsSummaryAgg {
            term_bloom: Some(bloom1),
            n_terms_distinct: 10,
            term_range: Some((b"a".to_vec(), b"z".to_vec())),
        };
        let summary2 = FtsSummaryAgg {
            term_bloom: Some(bloom2),
            n_terms_distinct: 15,
            term_range: Some((b"a".to_vec(), b"z".to_vec())),
        };
        into.insert("col".to_string(), summary1);
        other.insert("col".to_string(), summary2);
        FtsSummaryAgg::merge(&mut into, &other);
        // Column should be dropped because bloom shapes don't match
        assert!(into.is_empty());
    }

    #[test]
    fn fts_merge_union_of_columns() {
        let mut into = BTreeMap::new();
        let mut other = BTreeMap::new();
        let mut b = super::super::bloom::BloomBuilder::with_n_blocks(16);
        b.insert(b"test");
        let bloom = b.finish();
        into.insert(
            "col1".to_string(),
            FtsSummaryAgg {
                term_bloom: Some(bloom.clone()),
                n_terms_distinct: 10,
                term_range: Some((b"a".to_vec(), b"z".to_vec())),
            },
        );
        other.insert(
            "col2".to_string(),
            FtsSummaryAgg {
                term_bloom: Some(bloom),
                n_terms_distinct: 20,
                term_range: Some((b"a".to_vec(), b"z".to_vec())),
            },
        );
        FtsSummaryAgg::merge(&mut into, &other);
        assert_eq!(into.len(), 2);
        assert!(into.contains_key("col1"));
        assert!(into.contains_key("col2"));
    }

    #[test]
    fn fts_merge_with_none_blooms() {
        let mut into = BTreeMap::new();
        let mut other = BTreeMap::new();
        let summary1 = FtsSummaryAgg {
            term_bloom: None,
            n_terms_distinct: 0,
            term_range: None,
        };
        let summary2 = FtsSummaryAgg {
            term_bloom: None,
            n_terms_distinct: 0,
            term_range: None,
        };
        into.insert("col".to_string(), summary1);
        other.insert("col".to_string(), summary2);
        FtsSummaryAgg::merge(&mut into, &other);
        // Both had None blooms, result should be None (dropped)
        assert!(into.is_empty());
    }
}
