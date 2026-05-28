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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supertable::manifest::{ScalarStatsTable, SuperfileEntry, SuperfileUri};
    use arrow_array::{
        ArrayRef, Int32Array, Int64Array, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray,
    };
    use std::collections::HashMap;
    use std::sync::Arc;

    // ---- Helpers --------------------------------------------------------

    fn empty_seg() -> SuperfileEntry {
        SuperfileEntry {
            superfile_id: uuid::Uuid::nil(),
            uri: SuperfileUri(uuid::Uuid::nil()),
            n_docs: 0,
            id_min: 0,
            id_max: 0,
            scalar_stats: ScalarStatsTable::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
        }
    }

    fn seg_with_i64(column: &str, min: i64, max: i64) -> SuperfileEntry {
        let mut s = empty_seg();
        let mn: ArrayRef = Arc::new(Int64Array::from(vec![min]));
        let mx: ArrayRef = Arc::new(Int64Array::from(vec![max]));
        s.scalar_stats.cols.insert(column.to_string(), (mn, mx));
        s
    }

    fn assert_spans_partition(err: CommitError, needle: &str) {
        match err {
            CommitError::SuperfileSpansPartition { detail } => assert!(
                detail.contains(needle),
                "expected `{needle}` in detail; got: {detail}"
            ),
            other => panic!("expected SuperfileSpansPartition; got {other:?}"),
        }
    }

    // ---- encode_partition_key ------------------------------------------

    #[test]
    fn encode_partition_key_time_range_emits_le_u64() {
        let bytes = encode_partition_key(&PartitionKey::TimeRange(0x01_02_03_04_05_06_07_08));
        assert_eq!(bytes.len(), 8);
        assert_eq!(bytes, 0x01_02_03_04_05_06_07_08u64.to_le_bytes().to_vec());
    }

    #[test]
    fn encode_partition_key_hash_emits_le_u32() {
        let bytes = encode_partition_key(&PartitionKey::Hash(0xCAFEBABE));
        assert_eq!(bytes.len(), 4);
        assert_eq!(bytes, 0xCAFEBABEu32.to_le_bytes().to_vec());
    }

    #[test]
    fn encode_partition_key_column_range_emits_le_u16() {
        let bytes = encode_partition_key(&PartitionKey::ColumnRange(0xDEAD));
        assert_eq!(bytes.len(), 2);
        assert_eq!(bytes, 0xDEADu16.to_le_bytes().to_vec());
    }

    // ---- decode_partition_key — success roundtrip ----------------------

    #[test]
    fn decode_partition_key_round_trips_time_range() {
        let original = PartitionKey::TimeRange(42);
        let bytes = encode_partition_key(&original);
        let strategy = PartitionStrategy::TimeRange {
            column: "_id".into(),
            granularity_secs: 86_400,
        };
        let decoded = decode_partition_key(&bytes, &strategy).expect("decode");
        assert_eq!(decoded, original);
    }

    #[test]
    fn decode_partition_key_round_trips_hash() {
        let original = PartitionKey::Hash(5);
        let bytes = encode_partition_key(&original);
        let strategy = PartitionStrategy::Hash {
            column: "_id".into(),
            n_buckets: 16,
        };
        let decoded = decode_partition_key(&bytes, &strategy).expect("decode");
        assert_eq!(decoded, original);
    }

    #[test]
    fn decode_partition_key_round_trips_column_range() {
        let original = PartitionKey::ColumnRange(3);
        let bytes = encode_partition_key(&original);
        let strategy = PartitionStrategy::ColumnRange {
            column: "_id".into(),
            boundaries: vec![vec![]],
        };
        let decoded = decode_partition_key(&bytes, &strategy).expect("decode");
        assert_eq!(decoded, original);
    }

    // ---- decode_partition_key — size mismatch errors -------------------

    #[test]
    fn decode_partition_key_rejects_wrong_size_time_range() {
        let strategy = PartitionStrategy::TimeRange {
            column: "_id".into(),
            granularity_secs: 1,
        };
        let err = decode_partition_key(&[1, 2, 3], &strategy).expect_err("must error");
        match err {
            CommitError::PointerParse(msg) => {
                assert!(msg.contains("TimeRange"), "{msg}");
                assert!(msg.contains("8 bytes"), "{msg}");
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn decode_partition_key_rejects_wrong_size_hash() {
        let strategy = PartitionStrategy::Hash {
            column: "_id".into(),
            n_buckets: 4,
        };
        let err = decode_partition_key(&[1, 2, 3], &strategy).expect_err("must error");
        match err {
            CommitError::PointerParse(msg) => {
                assert!(msg.contains("Hash"), "{msg}");
                assert!(msg.contains("4 bytes"), "{msg}");
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn decode_partition_key_rejects_wrong_size_column_range() {
        let strategy = PartitionStrategy::ColumnRange {
            column: "_id".into(),
            boundaries: vec![vec![]],
        };
        let err = decode_partition_key(&[1, 2, 3], &strategy).expect_err("must error");
        match err {
            CommitError::PointerParse(msg) => {
                assert!(msg.contains("ColumnRange"), "{msg}");
                assert!(msg.contains("2 bytes"), "{msg}");
            }
            other => panic!("got {other:?}"),
        }
    }

    // ---- assign_partition: TimeRange -----------------------------------

    #[test]
    fn assign_partition_time_range_single_bucket() {
        // min and max sit in the same daily bucket → success.
        let strategy = PartitionStrategy::TimeRange {
            column: "ts".into(),
            granularity_secs: 86_400,
        };
        // 100 .. 90_000 → both buckets are 0..1 → same bucket 0.
        let seg = seg_with_i64("ts", 100, 80_000);
        let key = assign_partition(&seg, &strategy).expect("assign");
        assert_eq!(key, PartitionKey::TimeRange(0));
    }

    #[test]
    fn assign_partition_time_range_rejects_segment_spanning_buckets() {
        // min in bucket 0, max in bucket 1 → SuperfileSpansPartition.
        let strategy = PartitionStrategy::TimeRange {
            column: "ts".into(),
            granularity_secs: 86_400,
        };
        let seg = seg_with_i64("ts", 100, 100_000);
        let err = assign_partition(&seg, &strategy).expect_err("must span");
        assert_spans_partition(err, "spans buckets");
    }

    #[test]
    fn assign_partition_time_range_rejects_zero_granularity() {
        let strategy = PartitionStrategy::TimeRange {
            column: "ts".into(),
            granularity_secs: 0,
        };
        let seg = seg_with_i64("ts", 0, 0);
        let err = assign_partition(&seg, &strategy).expect_err("must reject");
        assert_spans_partition(err, "granularity_secs must be > 0");
    }

    #[test]
    fn assign_partition_time_range_rejects_negative_granularity() {
        let strategy = PartitionStrategy::TimeRange {
            column: "ts".into(),
            granularity_secs: -1,
        };
        let seg = seg_with_i64("ts", 0, 0);
        let err = assign_partition(&seg, &strategy).expect_err("must reject");
        assert_spans_partition(err, "granularity_secs must be > 0");
    }

    #[test]
    fn assign_partition_time_range_rejects_missing_column_stats() {
        let strategy = PartitionStrategy::TimeRange {
            column: "ts".into(),
            granularity_secs: 86_400,
        };
        let seg = empty_seg(); // no scalar_stats at all
        let err = assign_partition(&seg, &strategy).expect_err("missing stats");
        assert_spans_partition(err, "no scalar_stats");
    }

    #[test]
    fn assign_partition_time_range_supports_timestamp_columns() {
        // Each timestamp width must downcast cleanly so users can
        // configure granularity_secs against the column's actual
        // unit. Covers Second/Milli/Micro/Nano arms in downcast_i64.
        let strategy = PartitionStrategy::TimeRange {
            column: "ts".into(),
            granularity_secs: 86_400,
        };
        let cases: Vec<(ArrayRef, ArrayRef)> = vec![
            (
                Arc::new(TimestampSecondArray::from(vec![100])),
                Arc::new(TimestampSecondArray::from(vec![200])),
            ),
            (
                Arc::new(TimestampMillisecondArray::from(vec![100])),
                Arc::new(TimestampMillisecondArray::from(vec![200])),
            ),
            (
                Arc::new(TimestampMicrosecondArray::from(vec![100])),
                Arc::new(TimestampMicrosecondArray::from(vec![200])),
            ),
            (
                Arc::new(TimestampNanosecondArray::from(vec![100])),
                Arc::new(TimestampNanosecondArray::from(vec![200])),
            ),
        ];
        for (mn, mx) in cases {
            let mut seg = empty_seg();
            seg.scalar_stats.cols.insert("ts".into(), (mn, mx));
            let key = assign_partition(&seg, &strategy).expect("assign");
            assert_eq!(key, PartitionKey::TimeRange(0));
        }
    }

    #[test]
    fn assign_partition_time_range_rejects_unsupported_column_type() {
        // Int32 isn't supported (only Int64 + Timestamp widths).
        // Surfaces as SuperfileSpansPartition with a type-name hint.
        let strategy = PartitionStrategy::TimeRange {
            column: "ts".into(),
            granularity_secs: 86_400,
        };
        let mut seg = empty_seg();
        let mn: ArrayRef = Arc::new(Int32Array::from(vec![100]));
        let mx: ArrayRef = Arc::new(Int32Array::from(vec![200]));
        seg.scalar_stats.cols.insert("ts".into(), (mn, mx));
        let err = assign_partition(&seg, &strategy).expect_err("unsupported");
        assert_spans_partition(err, "unsupported type");
    }

    #[test]
    fn assign_partition_time_range_rejects_null_stats_array() {
        let strategy = PartitionStrategy::TimeRange {
            column: "ts".into(),
            granularity_secs: 86_400,
        };
        let mut seg = empty_seg();
        // Length-1 array with a single null value.
        let nulls: Vec<Option<i64>> = vec![None];
        let mn: ArrayRef = Arc::new(Int64Array::from(nulls.clone()));
        let mx: ArrayRef = Arc::new(Int64Array::from(nulls));
        seg.scalar_stats.cols.insert("ts".into(), (mn, mx));
        let err = assign_partition(&seg, &strategy).expect_err("null stats");
        assert_spans_partition(err, "empty or null at index 0");
    }

    #[test]
    fn assign_partition_time_range_handles_negative_values_with_div_euclid() {
        // `div_euclid` (not bare `/`) ensures negative timestamps
        // bucket consistently — same-bucket pairs across the
        // negative range must succeed. Catches an accidental
        // `min / g != max / g` regression that breaks at the
        // sign flip.
        let strategy = PartitionStrategy::TimeRange {
            column: "ts".into(),
            granularity_secs: 10,
        };
        // -25 div_euclid 10 = -3; -21 div_euclid 10 = -3.
        let seg = seg_with_i64("ts", -25, -21);
        let key = assign_partition(&seg, &strategy).expect("assign");
        assert_eq!(key, PartitionKey::TimeRange(-3i64 as u64));
    }

    // ---- assign_partition: Hash ----------------------------------------

    #[test]
    fn assign_partition_hash_single_bucket_short_circuits() {
        // n_buckets=1 → always bucket 0, no partition_hint needed.
        // This is the M15a default; tests rely on it staying
        // hint-less.
        let strategy = PartitionStrategy::Hash {
            column: "_id".into(),
            n_buckets: 1,
        };
        let seg = empty_seg();
        let key = assign_partition(&seg, &strategy).expect("assign");
        assert_eq!(key, PartitionKey::Hash(0));
    }

    #[test]
    fn assign_partition_hash_zero_buckets_treats_as_one() {
        // Defensive: n_buckets=0 falls into the <=1 short-circuit
        // rather than panicking on a later modulo.
        let strategy = PartitionStrategy::Hash {
            column: "_id".into(),
            n_buckets: 0,
        };
        let seg = empty_seg();
        let key = assign_partition(&seg, &strategy).expect("assign");
        assert_eq!(key, PartitionKey::Hash(0));
    }

    #[test]
    fn assign_partition_hash_uses_partition_hint() {
        let strategy = PartitionStrategy::Hash {
            column: "_id".into(),
            n_buckets: 4,
        };
        let mut seg = empty_seg();
        seg.partition_hint = Some(2);
        let key = assign_partition(&seg, &strategy).expect("assign");
        assert_eq!(key, PartitionKey::Hash(2));
    }

    #[test]
    fn assign_partition_hash_requires_hint_when_multi_bucket() {
        let strategy = PartitionStrategy::Hash {
            column: "_id".into(),
            n_buckets: 4,
        };
        let seg = empty_seg(); // hint = None
        let err = assign_partition(&seg, &strategy).expect_err("must reject");
        assert_spans_partition(err, "requires pre-sharded");
    }

    #[test]
    fn assign_partition_hash_rejects_out_of_range_hint() {
        let strategy = PartitionStrategy::Hash {
            column: "_id".into(),
            n_buckets: 4,
        };
        let mut seg = empty_seg();
        seg.partition_hint = Some(4); // == n_buckets, off-by-one
        let err = assign_partition(&seg, &strategy).expect_err("out of range");
        assert_spans_partition(err, "out of range");
    }

    // ---- assign_partition: ColumnRange (currently unimplemented) -------

    #[test]
    fn assign_partition_column_range_is_not_yet_supported() {
        // The implementation explicitly bails on ColumnRange until
        // M15a follow-up. Locking the message keeps it visible
        // when the follow-up lands.
        let strategy = PartitionStrategy::ColumnRange {
            column: "_id".into(),
            boundaries: vec![vec![]],
        };
        let seg = empty_seg();
        let err = assign_partition(&seg, &strategy).expect_err("not impl");
        assert_spans_partition(err, "ColumnRange partition assignment lands");
    }
}
