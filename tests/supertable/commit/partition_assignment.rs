// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Partition-aware writer + part-reuse.
//!
//! Covers the load-bearing invariants:
//!
//!   - **Default strategy = `Hash{n_buckets: 1}`.** The
//!     single-bucket Hash strategy is observationally
//!     equivalent to today's "one part per commit" path,
//!     so existing tests stay green AND the partition-split
//!     code path is exercised on every commit. Multi-commit
//!     scenarios exercise part-reuse: each commit's
//!     `ManifestPart` rebuilds the prior part's superfiles +
//!     the commit's new ones.
//!   - **Latest-part rewrite under default strategy.** After
//!     three commits, the manifest list has exactly one
//!     entry (one partition), and that entry's
//!     `n_superfiles` equals the cumulative superfile count.
//!     The `part_id` differs from commit to commit (each
//!     rewrite produces a fresh part with a new
//!     content-hash).
//!   - **Part-split at the target-superfiles threshold.**
//!     With `with_target_superfiles_per_partition(N)`, when a
//!     commit would push a partition's part above N
//!     superfiles, the writer emits a fresh part for that
//!     partition's new superfiles instead of rewriting the
//!     existing one. The list grows to two entries for
//!     that partition.
//!   - **Hash{n_buckets > 1} without partition_hint
//!     errors.** The writer can't pre-shard input batches
//!     yet (deferred), so a Hash strategy with n_buckets >
//!     1 fails the partition-assignment contract.
//!   - **TimeRange decoder wired up.** Int64 / Timestamp*
//!     columns drive bucket assignment from per-superfile
//!     min/max stats; superfiles spanning a granularity
//!     boundary surface `SuperfileSpansPartition` at commit
//!     time. Unsupported column types (e.g. UInt64) also
//!     fail with a typed error, not a silent miscount.
//!   - **ColumnRange is reserved.** Its partition_assignment
//!     path still surfaces a typed error today; existing
//!     config + storage paths accept the strategy — the
//!     failure is at commit time, not options-validation
//!     time.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use infino::superfile::builder::FtsConfig;
use infino::superfile::fts::tokenize::Tokenizer;
use infino::supertable::Supertable;
use infino::supertable::manifest::list::PartitionStrategy;
use infino::supertable::storage::{LocalFsStorageProvider, StorageProvider};
use infino::test_helpers::{build_title_batch, default_supertable_options, default_tokenizer};

/// Commits driven per partition-assignment scenario.
const COMMITS_PER_TEST: usize = 3;
/// Hash-partition bucket count for the multi-bucket fixture.
const HASH_N_BUCKETS: u32 = 4;
/// One-day partition granularity (seconds).
const DAY_GRANULARITY_SECS: i64 = 86_400;
/// Single-thread rayon pool for deterministic assignment.
const RAYON_POOL_THREADS: usize = 1;
/// A partition key is an 8-byte big-endian bucket id.
const PARTITION_KEY_BYTES: usize = 8;
use tempfile::TempDir;

#[test]
fn default_strategy_is_single_bucket_hash_observationally_equivalent_to_pre_m15a() {
    // Default = Hash{id_column, n_buckets: 1}. Three
    // commits → manifest list has exactly one entry with
    // accumulated superfiles.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    for _i in 0..COMMITS_PER_TEST {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["x"])).expect("append");
        w.commit().expect("commit");
    }

    let r = st.reader();
    let m = r.manifest();
    let list_entries = m.get_all_list_entries();
    assert_eq!(
        list_entries.len(),
        1,
        "single-bucket default → one list entry; got {} entries",
        list_entries.len()
    );
    assert_eq!(
        list_entries[0].n_superfiles, 3,
        "after 3 single-superfile commits the part should hold 3 superfiles"
    );
    // partition_key is the 4-byte LE encoding of bucket 0.
    assert_eq!(list_entries[0].partition_key, [0u8, 0, 0, 0]);
}

#[test]
fn rewrite_path_produces_fresh_part_id_per_commit() {
    // The "rewrite latest" path always emits a new
    // `part_id` because each rewrite is a content-
    // addressed new part. The PRIOR part becomes orphan
    // (GC'd by compaction); the new part replaces it in the list.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    let mut part_ids = Vec::new();
    for _i in 0..COMMITS_PER_TEST {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["x"])).expect("append");
        w.commit().expect("commit");
        let m_id = {
            let r = st.reader();
            let m = r.manifest();
            let list_entries = m.get_all_list_entries();
            list_entries[0].part_id
        };
        part_ids.push(m_id);
    }

    assert_ne!(part_ids[0], part_ids[1], "rewrite must mint a new part_id");
    assert_ne!(part_ids[1], part_ids[2]);
    assert_ne!(part_ids[0], part_ids[2]);
}

#[test]
fn target_superfiles_per_partition_triggers_part_split() {
    // With target_superfiles_per_partition = 2 and
    // single-superfile commits, the third commit pushes the
    // partition over the cap and emits a fresh part. The
    // list grows from 1 entry to 2 entries (both for the
    // same partition_key — the old entry preserved, the
    // new entry for fresh superfiles).
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let opts = default_supertable_options()
        .with_storage(Arc::clone(&storage))
        .with_target_superfiles_per_part(2);
    let st = Supertable::create(opts).expect("create");

    for _i in 0..COMMITS_PER_TEST {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["x"])).expect("append");
        w.commit().expect("commit");
    }

    let r = st.reader();
    let m = r.manifest();
    let list_entries = m.get_all_list_entries();
    assert_eq!(
        list_entries.len(),
        2,
        "after 3 commits with target=2, the partition should split into 2 entries; \
         got {} entries",
        list_entries.len()
    );
    assert_eq!(
        list_entries[0].partition_key, list_entries[1].partition_key,
        "both entries should share the same partition_key (same partition, split into 2 parts)"
    );
    let total_superfiles: u64 = list_entries.iter().map(|p| p.n_superfiles).sum();
    assert_eq!(total_superfiles, 3);
}

#[test]
fn hash_strategy_with_multiple_buckets_errors_without_partition_hint() {
    // The writer doesn't pre-shard yet; superfiles come out
    // with `partition_hint = None`. Hash{n_buckets > 1}
    // requires the hint, so assign_partition surfaces
    // SuperfileSpansPartition. Writer.commit propagates as a
    // BuildError::Store wrapping the CommitError.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let opts = default_supertable_options()
        .with_storage(Arc::clone(&storage))
        .with_partition_strategy(PartitionStrategy::Hash {
            column: "doc_id".into(),
            n_buckets: HASH_N_BUCKETS,
        });
    let st = Supertable::create(opts).expect("create");

    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["alpha"])).expect("append");
    let err = w.commit().expect_err("commit must fail without pre-shard");
    let s = format!("{err}");
    assert!(
        s.contains("pre-sharded") || s.contains("partition_hint"),
        "expected partition-assignment error, got: {s}"
    );
}

#[test]
fn time_range_strategy_on_unsupported_column_type_errors_cleanly() {
    // The supertable-injected `_id` column is
    // `Decimal128(38, 0)`, which is NOT in TimeRange's
    // supported type set (Int64 + Timestamp{Second,
    // Millisecond, Microsecond, Nanosecond}). TimeRange's
    // bucket math operates on signed 64-bit values;
    // surfacing a typed error here keeps users from
    // accidentally configuring TimeRange on an
    // unsupported column and getting silently wrong
    // partition assignments.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let opts = default_supertable_options()
        .with_storage(Arc::clone(&storage))
        .with_partition_strategy(PartitionStrategy::TimeRange {
            column: "_id".into(),
            granularity_secs: DAY_GRANULARITY_SECS,
        });
    let st = Supertable::create(opts).expect("create");

    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["alpha"])).expect("append");
    let err = w
        .commit()
        .expect_err("commit must fail on unsupported column type");
    let s = format!("{err}");
    assert!(
        s.contains("unsupported type") || s.contains("expected Int64 or Timestamp"),
        "expected unsupported-type TimeRange error; got: {s}"
    );
}

#[test]
fn time_range_assigns_int64_superfiles_to_bucket_zero() {
    // Happy path: an Int64-keyed schema with TimeRange
    // partition_strategy, single-bucket-spanning batch →
    // commit succeeds + the manifest list's entry carries
    // a TimeRange partition_key (8 bytes LE bucket index).
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    // Build a schema where the timestamp-style column
    // (`ts_secs`) is Int64.
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("doc_id", arrow_schema::DataType::UInt64, false),
        arrow_schema::Field::new("ts_secs", arrow_schema::DataType::Int64, false),
        arrow_schema::Field::new("title", arrow_schema::DataType::LargeUtf8, false),
    ]));
    let tk: Arc<dyn Tokenizer> = default_tokenizer();
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("pool"),
    );
    let opts = infino::supertable::SupertableOptions::new(
        schema.clone(),
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![],
        Some(tk),
    )
    .expect("opts")
    .with_writer_pool(pool)
    .with_storage(Arc::clone(&storage))
    .with_partition_strategy(PartitionStrategy::TimeRange {
        column: "ts_secs".into(),
        granularity_secs: DAY_GRANULARITY_SECS,
    });

    let st = Supertable::create(opts).expect("create");
    // All ts values land within day-0 (epoch seconds 0..86400).
    let batch = arrow_array::RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::UInt64Array::from(vec![0u64, 1])),
            Arc::new(arrow_array::Int64Array::from(vec![10_i64, 20])),
            Arc::new(arrow_array::LargeStringArray::from(vec!["a", "b"])),
        ],
    )
    .expect("batch");
    {
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        w.commit()
            .expect("TimeRange commit must succeed for a single-bucket batch");
    }
    let r = st.reader();
    let m = r.manifest();
    let list_entries = m.get_all_list_entries();
    assert_eq!(
        list_entries.len(),
        1,
        "single-bucket commit produces one part"
    );
    // TimeRange partition_key is 8 bytes LE bucket index.
    assert_eq!(list_entries[0].partition_key.len(), PARTITION_KEY_BYTES);
    let bucket = u64::from_le_bytes(
        list_entries[0]
            .partition_key
            .as_slice()
            .try_into()
            .expect("8-byte le"),
    );
    assert_eq!(bucket, 0, "ts in [10, 20] @ granularity 86400 → bucket 0");
}

#[test]
fn time_range_superfile_spanning_two_buckets_errors() {
    // Bucket-spanning batch (ts crosses a day boundary)
    // surfaces `SuperfileSpansPartition` so the writer
    // doesn't silently group two days' rows under one
    // partition key.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("doc_id", arrow_schema::DataType::UInt64, false),
        arrow_schema::Field::new("ts_secs", arrow_schema::DataType::Int64, false),
        arrow_schema::Field::new("title", arrow_schema::DataType::LargeUtf8, false),
    ]));
    let tk: Arc<dyn Tokenizer> = default_tokenizer();
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("pool"),
    );
    let opts = infino::supertable::SupertableOptions::new(
        schema.clone(),
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![],
        Some(tk),
    )
    .expect("opts")
    .with_writer_pool(pool)
    .with_storage(Arc::clone(&storage))
    .with_partition_strategy(PartitionStrategy::TimeRange {
        column: "ts_secs".into(),
        granularity_secs: DAY_GRANULARITY_SECS,
    });

    let st = Supertable::create(opts).expect("create");
    // ts values in [10, 86_500] → spans day 0 and day 1.
    let batch = arrow_array::RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::UInt64Array::from(vec![0u64, 1])),
            Arc::new(arrow_array::Int64Array::from(vec![10_i64, 86_500])),
            Arc::new(arrow_array::LargeStringArray::from(vec!["a", "b"])),
        ],
    )
    .expect("batch");
    let mut w = st.writer().expect("writer");
    w.append(&batch).expect("append");
    let err = w.commit().expect_err("spanning two buckets must error");
    let s = format!("{err}");
    assert!(
        s.contains("spans buckets"),
        "expected SuperfileSpansPartition with spans-buckets detail; got: {s}"
    );
}
