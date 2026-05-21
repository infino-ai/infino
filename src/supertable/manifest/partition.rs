//! Partition assignment.
//!
//! Given a `SuperfileEntry`'s per-column min/max summaries +
//! the supertable's configured `PartitionStrategy`, decide
//! which partition the segment belongs to. Drives the
//! writer's "rewrite latest part" policy: superfiles in the
//! same partition share a `ManifestPart`; superfiles in
//! different partitions go into separate parts so a
//! single-partition commit rewrites exactly one part.

use crate::supertable::error::CommitError;
use crate::supertable::manifest::SuperfileEntry;
use crate::supertable::manifest::list::PartitionStrategy;

/// Opaque partition identifier. Encoded into
/// `SuperfileEntry.partition_key` + `ManifestListEntry.partition_key`
/// for the manifest layer; the writer uses this typed shape
/// in-memory to group superfiles before encoding.
///
/// The on-disk encoding (LE u64 / u32 / u16) is the
/// responsibility of [`encode_partition_key`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartitionKey {
    /// `time_range` bucket = `value / granularity_secs`.
    TimeRange(u64),
    /// `hash` bucket = `hash(column_value) % n_buckets`.
    /// For `n_buckets == 1` the writer short-circuits to
    /// `Hash(0)` without requiring the `partition_hint`
    /// field — see [`assign_partition`].
    Hash(u32),
    /// `column_range` bucket = boundary index.
    ColumnRange(u16),
}

/// Encode a `PartitionKey` to its on-disk bytes — the shape
/// `SuperfileEntry.partition_key` and `ManifestListEntry.partition_key`
/// carry: 8-byte LE u64 for TimeRange, 4-byte LE u32 for
/// Hash, 2-byte LE u16 for ColumnRange.
pub fn encode_partition_key(key: &PartitionKey) -> Vec<u8> {
    match key {
        PartitionKey::TimeRange(b) => b.to_le_bytes().to_vec(),
        PartitionKey::Hash(b) => b.to_le_bytes().to_vec(),
        PartitionKey::ColumnRange(b) => b.to_le_bytes().to_vec(),
    }
}

/// Decode a `partition_key: Vec<u8>` back to its typed
/// shape, given the strategy. Used by the writer when
/// reading existing list entries to group surviving parts
/// by partition.
pub fn decode_partition_key(
    bytes: &[u8],
    strategy: &PartitionStrategy,
) -> Result<PartitionKey, CommitError> {
    match strategy {
        PartitionStrategy::TimeRange { .. } => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| {
                CommitError::PointerParse(format!(
                    "TimeRange partition_key must be 8 bytes; got {}",
                    bytes.len()
                ))
            })?;
            Ok(PartitionKey::TimeRange(u64::from_le_bytes(arr)))
        }
        PartitionStrategy::Hash { .. } => {
            let arr: [u8; 4] = bytes.try_into().map_err(|_| {
                CommitError::PointerParse(format!(
                    "Hash partition_key must be 4 bytes; got {}",
                    bytes.len()
                ))
            })?;
            Ok(PartitionKey::Hash(u32::from_le_bytes(arr)))
        }
        PartitionStrategy::ColumnRange { .. } => {
            let arr: [u8; 2] = bytes.try_into().map_err(|_| {
                CommitError::PointerParse(format!(
                    "ColumnRange partition_key must be 2 bytes; got {}",
                    bytes.len()
                ))
            })?;
            Ok(PartitionKey::ColumnRange(u16::from_le_bytes(arr)))
        }
    }
}

/// Decide which partition `seg` belongs to under `strategy`.
///
/// - **TimeRange**: segment's `(min, max)` on the partition
///   column must fall within a single bucket
///   (`min / granularity_secs == max / granularity_secs`).
///   Spans → `SuperfileSpansPartition`.
/// - **Hash**: requires `seg.partition_hint = Some(bucket)`
///   from the writer's pre-shard step — except the
///   `n_buckets == 1` special case (every row hashes to
///   bucket 0; pre-shard is trivial). Without the hint,
///   surfaces `SuperfileSpansPartition` with a "hash strategy
///   requires pre-sharded superfiles" message.
/// - **ColumnRange**: segment's `(min, max)` must fall within
///   one boundary interval. Spans → `SuperfileSpansPartition`.
///
/// The `n_buckets == 1` Hash short-circuit is critical for
/// backward compatibility: the M15a default partition
/// strategy (when nothing's configured) is
/// `Hash { id_column, n_buckets: 1 }`, and the existing
/// writer path doesn't yet pre-shard — so the hint is None.
/// Special-casing 1-bucket lets every existing test keep
/// running on the new partition-aware writer without
/// any test changes.
pub fn assign_partition(
    seg: &SuperfileEntry,
    strategy: &PartitionStrategy,
) -> Result<PartitionKey, CommitError> {
    match strategy {
        PartitionStrategy::TimeRange {
            column,
            granularity_secs,
        } => {
            if *granularity_secs <= 0 {
                return Err(CommitError::SuperfileSpansPartition {
                    detail: format!(
                        "TimeRange granularity_secs must be > 0; got {granularity_secs}"
                    ),
                });
            }
            let (min, max) = scalar_i64_minmax(seg, column)?;
            let g = *granularity_secs;
            let min_bucket = min.div_euclid(g);
            let max_bucket = max.div_euclid(g);
            if min_bucket != max_bucket {
                return Err(CommitError::SuperfileSpansPartition {
                    detail: format!(
                        "segment {} column {column:?} [{min}, {max}] spans buckets \
                         {min_bucket}..={max_bucket}; reduce commit_threshold_size_mb \
                         or flush at granularity boundaries",
                        seg.uri.0
                    ),
                });
            }
            Ok(PartitionKey::TimeRange(min_bucket as u64))
        }

        PartitionStrategy::Hash {
            column: _,
            n_buckets,
        } => {
            // Single-bucket short-circuit: every row hashes
            // to bucket 0 trivially. No pre-shard required.
            if *n_buckets <= 1 {
                return Ok(PartitionKey::Hash(0));
            }
            // Multi-bucket: writer must have stamped
            // partition_hint at pre-shard time.
            let bucket =
                seg.partition_hint
                    .ok_or_else(|| CommitError::SuperfileSpansPartition {
                        detail: format!(
                            "Hash{{n_buckets:{n_buckets}}} strategy requires pre-sharded \
                         superfiles; SuperfileEntry.partition_hint must be Some(bucket) \
                         (segment {})",
                            seg.uri.0
                        ),
                    })?;
            if bucket >= *n_buckets {
                return Err(CommitError::SuperfileSpansPartition {
                    detail: format!(
                        "Hash{{n_buckets:{n_buckets}}} got partition_hint={bucket} \
                         (out of range)"
                    ),
                });
            }
            Ok(PartitionKey::Hash(bucket))
        }

        PartitionStrategy::ColumnRange {
            column: _,
            boundaries: _,
        } => Err(CommitError::SuperfileSpansPartition {
            detail: "ColumnRange partition assignment lands in M15a follow-up; \
                     no writer currently emits ColumnRange-partitioned commits"
                .into(),
        }),
    }
}

/// Extract the segment's `(min, max)` for `column` as `i64`.
/// `ScalarStatsTable.cols[column]` carries Arrow length-1
/// `ArrayRef`s; this helper downcasts against the column's
/// actual Arrow type and returns the value at index 0.
///
/// Supported types: `Int64` (epoch seconds-style integer
/// columns) and the three timestamp widths
/// (`TimestampSecond` / `TimestampMillisecond` /
/// `TimestampMicrosecond` / `TimestampNanosecond`). All
/// timestamp values downcast to i64 directly; the
/// granularity-bucket math in `assign_partition` treats
/// them as opaque i64 — callers configuring
/// `granularity_secs` are responsible for matching it to
/// the column's actual unit (seconds for `Int64`,
/// microseconds for `TimestampMicrosecond`, etc.).
fn scalar_i64_minmax(seg: &SuperfileEntry, column: &str) -> Result<(i64, i64), CommitError> {
    let (mn_arr, mx_arr) =
        seg.scalar_stats
            .cols
            .get(column)
            .ok_or_else(|| CommitError::SuperfileSpansPartition {
                detail: format!(
                    "TimeRange strategy: segment {} has no scalar_stats \
                     for column {column:?}",
                    seg.uri.0
                ),
            })?;
    let min = downcast_i64(mn_arr.as_ref(), column, seg)?;
    let max = downcast_i64(mx_arr.as_ref(), column, seg)?;
    Ok((min, max))
}

fn downcast_i64(
    arr: &dyn arrow_array::Array,
    column: &str,
    seg: &SuperfileEntry,
) -> Result<i64, CommitError> {
    use arrow_array::*;
    use arrow_schema::DataType;
    if arr.is_empty() || arr.is_null(0) {
        return Err(CommitError::SuperfileSpansPartition {
            detail: format!(
                "TimeRange strategy: segment {} column {column:?} stats array \
                 is empty or null at index 0",
                seg.uri.0
            ),
        });
    }
    let v = match arr.data_type() {
        DataType::Int64 => arr
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| a.value(0)),
        DataType::Timestamp(arrow_schema::TimeUnit::Second, _) => arr
            .as_any()
            .downcast_ref::<TimestampSecondArray>()
            .map(|a| a.value(0)),
        DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, _) => arr
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .map(|a| a.value(0)),
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => arr
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .map(|a| a.value(0)),
        DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => arr
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .map(|a| a.value(0)),
        other => {
            return Err(CommitError::SuperfileSpansPartition {
                detail: format!(
                    "TimeRange strategy: segment {} column {column:?} has \
                     unsupported type {other:?}; expected Int64 or Timestamp*",
                    seg.uri.0
                ),
            });
        }
    };
    v.ok_or_else(|| CommitError::SuperfileSpansPartition {
        detail: format!(
            "TimeRange strategy: segment {} column {column:?} downcast failed",
            seg.uri.0
        ),
    })
}
