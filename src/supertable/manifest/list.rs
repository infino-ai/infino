// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `ManifestList` — the top-tier of the two-tier hierarchical manifest.
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

use std::collections::BTreeMap;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::part::{BLAKE3_DIGEST_BYTES, BLAKE3_HEX_LEN, ContentHash, PartId};

/// Wire format version for the manifest list.
///
/// Major must match at decode time; minor differences are
/// accepted (deny-unknown-major / allow-unknown-minor — same
/// shape as the [`super::part::FORMAT_VERSION`] policy for
/// manifest parts).
pub const FORMAT_VERSION: &str = "1.0";

// ---------- Public in-memory shapes ----------

/// Top-level manifest list. The wire format is the JSON
/// produced by [`encode`]; this struct is the in-memory
/// shape callers (the supertable's load/refresh path and
/// the writer's commit path) consume.
#[derive(Debug, Clone)]
pub struct ManifestList {
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
    /// Entries — one per manifest part referenced by this
    /// list. Ordered by insertion order (commit order); the
    /// list-level pruner walks them in order.
    pub parts: Vec<ManifestListEntry>,
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

/// How superfiles are routed into manifest parts. Stamped into
/// the list on first commit; immutable thereafter (changes
/// require external compaction).
#[derive(Debug, Clone, PartialEq, Eq)]
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
}

#[derive(Debug, Clone)]
pub struct ManifestListEntry {
    pub part_id: PartId,
    /// Storage URI for the part. Typically
    /// `manifests/part-<hash>.avro.zst`. Content-addressed so
    /// two identical lists share part files.
    pub uri: String,
    pub n_superfiles: u64,
    pub size_bytes_compressed: u64,
    pub size_bytes_uncompressed: u64,
    pub content_hash: ContentHash,
    /// The partition's encoded key (8-byte LE u64 for
    /// `TimeRange`, etc.). Filled by the writer from
    /// `PartitionStrategy::assign`; empty when no real
    /// partition strategy is configured (single-bucket Hash).
    pub partition_key: Vec<u8>,
    /// Aggregate id range across this part's superfiles. `i128`
    /// matches the supertable-injected `_id` column type
    /// (`Decimal128(38, 0)`); signed-int comparison gives
    /// time-ordered skip-pruning because the high bit stays
    /// 0 for any plausible current-era timestamp.
    pub id_range: (i128, i128),
    /// Per-scalar-column aggregate min/max across all
    /// superfiles in this part. An empty map is interpreted as
    /// "always-keep" by the list-level pruner.
    pub scalar_stats_agg: BTreeMap<String, ScalarStatsAgg>,
    /// Per-FTS-column aggregate bloom-union + range-union.
    /// Empty → always-keep.
    pub fts_summary_agg: BTreeMap<String, FtsSummaryAgg>,
    /// Per-vector-column aggregate centroid envelope.
    /// Empty → always-keep.
    pub vector_summary_agg: BTreeMap<String, VectorSummaryAgg>,
}

/// Aggregate scalar stats across a part's superfiles. Min/max
/// are Arrow-IPC bytes of length-1 arrays (matches
/// `ScalarStatsTable.cols` per-column shape).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScalarStatsAgg {
    pub min: Vec<u8>,
    pub max: Vec<u8>,
    /// Σ null_count across the part's segments; `None` when any
    /// segment lacks the stat (total unknowable, never zero).
    pub null_count: Option<u64>,
    /// Part-wide exact sum as Arrow-IPC length-1 bytes (same typing
    /// as the per-segment `ScalarStatsTable.sums`); `None` when any
    /// segment lacks it or the fold overflowed.
    pub sum: Option<Vec<u8>>,
    /// Merged HLL distinct sketch (raw registers); `None` when any
    /// segment lacks one.
    pub hll: Option<Vec<u8>>,
}

/// Aggregate FTS summary across a part's superfiles.
///
/// When populated, built via streaming HLL + a power-of-two-
/// rounded blocked bloom sized to
/// `manifest.list_bloom_target_fpr` (default 0.10) at the
/// part's measured distinct-term cardinality. The `Default`
/// shape — empty bloom, no range — is treated as "always-
/// keep" by the list-level pruner (correctness preserved;
/// selectivity 0).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FtsSummaryAgg {
    pub term_bloom_union: Vec<u8>,
    /// Power-of-two block count for the union bloom. The
    /// existing `BloomBuilder::with_n_blocks` asserts pow2;
    /// emitting this here means decoders can reconstruct the
    /// bloom shape without inferring from byte length.
    pub term_bloom_n_blocks: u32,
    /// HyperLogLog-estimated distinct term count across the
    /// part's superfiles. `0` for the `Default` shape.
    pub n_terms_distinct: u64,
    /// `(min, max)` term range across the part. `None` if
    /// every superfile's FST was empty for this column.
    pub term_range_union: Option<(Vec<u8>, Vec<u8>)>,
}

/// Aggregate vector summary across a part's superfiles —
/// mean-of-centroids + max-distance-with-superfile-radius (one
/// outer ball bounding every superfile's vector ball). The
/// `Default` shape is treated as "always-keep" by the list-
/// level pruner.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VectorSummaryAgg {
    /// Packed LE f32 — same encoding as `VectorSummary.centroid`.
    pub centroid_envelope: Vec<u8>,
    pub envelope_radius: f32,
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
    #[error("incompatible major version: got {got}, supported {supported}")]
    IncompatibleMajorVersion { got: String, supported: String },
}

#[derive(Debug, Error)]
pub enum ListEncodeError {
    #[error("json encode failed: {0}")]
    Json(#[from] serde_json::Error),
}

// ---------- Wire (serde) types ----------

/// JSON wire shape — DTO that owns the base64 transformations.
///
/// Field ordering here is the JSON-output order; serde_json's
/// `to_writer_pretty` preserves declaration order, so output
/// is deterministic for content-addressing.
#[derive(Serialize, Deserialize)]
struct ManifestListDto {
    format_version: String,
    manifest_id: u64,
    options_hash: String, // "blake3:<64hex>"
    schema: String,       // base64
    id_column: String,
    fts_columns: Vec<FtsColumnInfo>,
    vector_columns: Vec<VectorColumnInfoDto>,
    partition_strategy: PartitionStrategyDto,
    parts: Vec<ManifestListEntryDto>,
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
}

#[derive(Serialize, Deserialize)]
struct ManifestListEntryDto {
    part_id: String, // UUID
    uri: String,
    n_superfiles: u64,
    size_bytes_compressed: u64,
    size_bytes_uncompressed: u64,
    content_hash: String,  // "blake3:<hex>"
    partition_key: String, // base64
    // i128 stringified as decimal — JSON numbers are bounded
    // to f64 precision (~53 bits) so we can't round-trip a
    // 128-bit value as a JSON number without loss. Decimal
    // strings keep the manifest list debuggable in `jq`
    // (`echo '...' | jq '.parts[0].id_range'` shows real
    // values) and avoid base64 ambiguity.
    id_range: (String, String),
    scalar_stats_agg: BTreeMap<String, ScalarStatsAggDto>,
    fts_summary_agg: BTreeMap<String, FtsSummaryAggDto>,
    vector_summary_agg: BTreeMap<String, VectorSummaryAggDto>,
}

#[derive(Serialize, Deserialize)]
struct ScalarStatsAggDto {
    min: String, // base64
    max: String, // base64
    /// `None` ↔ field absent in JSON (parts written before the stat
    /// existed decode cleanly).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    null_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sum: Option<String>, // base64
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hll: Option<String>, // base64
}

#[derive(Serialize, Deserialize)]
struct FtsSummaryAggDto {
    term_bloom_union: String, // base64
    term_bloom_n_blocks: u32,
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

#[derive(Serialize, Deserialize)]
struct VectorSummaryAggDto {
    centroid_envelope: String, // base64
    envelope_radius: f32,
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

fn entry_to_dto(e: &ManifestListEntry) -> ManifestListEntryDto {
    ManifestListEntryDto {
        part_id: e.part_id.0.to_string(),
        uri: e.uri.clone(),
        n_superfiles: e.n_superfiles,
        size_bytes_compressed: e.size_bytes_compressed,
        size_bytes_uncompressed: e.size_bytes_uncompressed,
        content_hash: encode_hash(&e.content_hash),
        partition_key: encode_b64(&e.partition_key),
        id_range: (e.id_range.0.to_string(), e.id_range.1.to_string()),
        scalar_stats_agg: e
            .scalar_stats_agg
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    ScalarStatsAggDto {
                        min: encode_b64(&v.min),
                        max: encode_b64(&v.max),
                        null_count: v.null_count,
                        sum: v.sum.as_deref().map(encode_b64),
                        hll: v.hll.as_deref().map(encode_b64),
                    },
                )
            })
            .collect(),
        fts_summary_agg: e
            .fts_summary_agg
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    FtsSummaryAggDto {
                        term_bloom_union: encode_b64(&v.term_bloom_union),
                        term_bloom_n_blocks: v.term_bloom_n_blocks,
                        n_terms_distinct: v.n_terms_distinct,
                        term_range_union: v.term_range_union.as_ref().map(|(mn, mx)| {
                            TermRangeUnionDto {
                                min: encode_b64(mn),
                                max: encode_b64(mx),
                            }
                        }),
                    },
                )
            })
            .collect(),
        vector_summary_agg: e
            .vector_summary_agg
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    VectorSummaryAggDto {
                        centroid_envelope: encode_b64(&v.centroid_envelope),
                        envelope_radius: v.envelope_radius,
                    },
                )
            })
            .collect(),
    }
}

fn entry_from_dto(d: ManifestListEntryDto) -> Result<ManifestListEntry, ListParseError> {
    let part_id = PartId(
        uuid::Uuid::parse_str(&d.part_id).map_err(|e| ListParseError::BadPartId(e.to_string()))?,
    );
    let content_hash = decode_hash(&d.content_hash)?;
    let partition_key = decode_b64(&d.partition_key, "partition_key")?;
    let mut scalar_stats_agg = BTreeMap::new();
    for (k, v) in d.scalar_stats_agg {
        scalar_stats_agg.insert(
            k,
            ScalarStatsAgg {
                min: decode_b64(&v.min, "scalar_stats_agg.min")?,
                max: decode_b64(&v.max, "scalar_stats_agg.max")?,
                null_count: v.null_count,
                sum: v
                    .sum
                    .as_deref()
                    .map(|s| decode_b64(s, "scalar_stats_agg.sum"))
                    .transpose()?,
                hll: v
                    .hll
                    .as_deref()
                    .map(|s| decode_b64(s, "scalar_stats_agg.hll"))
                    .transpose()?,
            },
        );
    }
    let mut fts_summary_agg = BTreeMap::new();
    for (k, v) in d.fts_summary_agg {
        fts_summary_agg.insert(
            k,
            FtsSummaryAgg {
                term_bloom_union: decode_b64(&v.term_bloom_union, "term_bloom_union")?,
                term_bloom_n_blocks: v.term_bloom_n_blocks,
                n_terms_distinct: v.n_terms_distinct,
                term_range_union: match v.term_range_union {
                    None => None,
                    Some(tr) => Some((
                        decode_b64(&tr.min, "term_range_union.min")?,
                        decode_b64(&tr.max, "term_range_union.max")?,
                    )),
                },
            },
        );
    }
    let mut vector_summary_agg = BTreeMap::new();
    for (k, v) in d.vector_summary_agg {
        vector_summary_agg.insert(
            k,
            VectorSummaryAgg {
                centroid_envelope: decode_b64(&v.centroid_envelope, "centroid_envelope")?,
                envelope_radius: v.envelope_radius,
            },
        );
    }
    Ok(ManifestListEntry {
        part_id,
        uri: d.uri,
        n_superfiles: d.n_superfiles,
        size_bytes_compressed: d.size_bytes_compressed,
        size_bytes_uncompressed: d.size_bytes_uncompressed,
        content_hash,
        partition_key,
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
        vector_summary_agg,
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
    })
}

fn list_to_dto(l: &ManifestList) -> ManifestListDto {
    ManifestListDto {
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
        parts: l.parts.iter().map(entry_to_dto).collect(),
    }
}

fn list_from_dto(d: ManifestListDto) -> Result<ManifestList, ListParseError> {
    check_major(&d.format_version)?;
    let options_hash = decode_hash(&d.options_hash)?;
    let schema = decode_b64(&d.schema, "schema")?;
    let mut parts = Vec::with_capacity(d.parts.len());
    for entry in d.parts {
        parts.push(entry_from_dto(entry)?);
    }
    Ok(ManifestList {
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
        parts,
    })
}

// ---------- Encode / decode ----------

/// JSON-encode a manifest list. Pretty-printed; field order
/// is the declaration order in `ManifestListDto` and child
/// types, so byte-output is deterministic for content-equal
/// inputs.
pub fn encode(list: &ManifestList) -> Result<Vec<u8>, ListEncodeError> {
    let dto = list_to_dto(list);
    Ok(serde_json::to_vec_pretty(&dto)?)
}

/// JSON-decode a manifest list. Verifies major-version
/// compatibility; allows unknown minor versions.
pub fn decode(bytes: &[u8]) -> Result<ManifestList, ListParseError> {
    let dto: ManifestListDto = serde_json::from_slice(bytes)?;
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
    //! JSON round-trip tests for `ManifestList`.
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
    use super::super::part::{ContentHash, PartId};
    use super::*;
    use std::collections::BTreeMap;
    use uuid::Uuid;

    fn empty_list() -> ManifestList {
        ManifestList {
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
            parts: vec![],
        }
    }

    fn rich_entry(seed: u8) -> ManifestListEntry {
        let mut scalar = BTreeMap::new();
        scalar.insert(
            "ts".into(),
            ScalarStatsAgg {
                min: vec![seed, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06],
                max: vec![seed, 0xff, 0xfe, 0xfd, 0xfc, 0xfb, 0xfa, 0xf9],
                null_count: Some(u64::from(seed)),
                sum: Some(vec![seed, 0x10, 0x20]),
                hll: Some(vec![seed; 4]),
            },
        );

        let mut fts = BTreeMap::new();
        fts.insert(
            "title".into(),
            FtsSummaryAgg {
                term_bloom_union: vec![seed; 64],
                term_bloom_n_blocks: 16,
                n_terms_distinct: 1_048_576,
                term_range_union: Some((b"alpha".to_vec(), b"zulu".to_vec())),
            },
        );
        fts.insert(
            "body".into(),
            FtsSummaryAgg {
                term_bloom_union: vec![0u8; 32],
                term_bloom_n_blocks: 8,
                n_terms_distinct: 0,
                term_range_union: None,
            },
        );

        let mut vec_agg = BTreeMap::new();
        vec_agg.insert(
            "emb".into(),
            VectorSummaryAgg {
                centroid_envelope: 0.5_f32.to_le_bytes().repeat(8),
                envelope_radius: 0.71_f32,
            },
        );

        ManifestListEntry {
            part_id: PartId(Uuid::from_bytes([seed; 16])),
            uri: format!("manifests/part-{seed:02x}.avro.zst"),
            n_superfiles: 9_847,
            size_bytes_compressed: 10_485_760,
            size_bytes_uncompressed: 26_214_400,
            content_hash: ContentHash([seed; 32]),
            partition_key: vec![seed, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            id_range: (0, 245_678_901),
            scalar_stats_agg: scalar,
            fts_summary_agg: fts,
            vector_summary_agg: vec_agg,
        }
    }

    fn rich_list(n_parts: u8) -> ManifestList {
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

    fn assert_entries_equal(a: &ManifestListEntry, b: &ManifestListEntry) {
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
        assert_eq!(a.partition_key, b.partition_key, "partition_key");
        assert_eq!(a.id_range, b.id_range, "id_range");
        assert_eq!(a.scalar_stats_agg, b.scalar_stats_agg, "scalar_stats_agg");
        assert_eq!(a.fts_summary_agg, b.fts_summary_agg, "fts_summary_agg");
        assert_eq!(
            a.vector_summary_agg.len(),
            b.vector_summary_agg.len(),
            "vector_summary_agg count"
        );
        for (k, av) in &a.vector_summary_agg {
            let bv = b
                .vector_summary_agg
                .get(k)
                .unwrap_or_else(|| panic!("missing vec {k}"));
            assert_eq!(av.centroid_envelope, bv.centroid_envelope, "vec {k} env");
            assert_eq!(
                av.envelope_radius.to_bits(),
                bv.envelope_radius.to_bits(),
                "vec {k} radius bits"
            );
        }
    }

    fn assert_lists_equal(a: &ManifestList, b: &ManifestList) {
        assert_eq!(a.format_version, b.format_version);
        assert_eq!(a.manifest_id, b.manifest_id);
        assert_eq!(a.options_hash, b.options_hash);
        assert_eq!(a.schema, b.schema);
        assert_eq!(a.id_column, b.id_column);
        assert_eq!(a.fts_columns, b.fts_columns);
        assert_eq!(a.vector_columns, b.vector_columns);
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
    fn term_range_union_none_is_absent_from_json() {
        // term_range_union is #[serde(skip_serializing_if =
        // "Option::is_none")], so None must produce field
        // absence, not `"term_range_union": null`. This is the
        // property cross-version content-addressing rides on.
        let list = rich_list(1);
        let bytes = encode(&list).expect("encode");
        let s = std::str::from_utf8(&bytes).expect("utf8");
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
        // Manifest into the new one.
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
        // Manifest list is the operator's debugging surface;
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
        let s = std::str::from_utf8(&bytes).expect("utf8");
        let tampered = s.replacen("\"schema\": \"", "\"schema\": \"!!!!", 1);
        let err = decode(tampered.as_bytes()).expect_err("must fail");
        assert!(
            matches!(err, ListParseError::Base64 { .. }),
            "expected Base64 error, got {err:?}"
        );
    }
}
