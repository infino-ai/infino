// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Per-op work stats over the write surface: a write wrapped in
//! [`with_op_stats`] reports the rows and bytes it indexed,
//! deterministically — the same batch into the same table state reports
//! the same priced counters whether the writer pool is one thread or
//! eight, and a writer minted outside a scope records nothing. The
//! width-dependent output-shape counters (superfile count/bytes,
//! distinct terms) are recorded-only and exempted from the invariance
//! contract, exactly as the module documents.

#![deny(clippy::unwrap_used)]

use std::{mem, sync::Arc};

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{col, lit};
use infino::{
    config::{CompactionSettings, OptimizeOptions},
    runtime_metrics::op_stats::{OpStats, with_op_stats},
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
    test_helpers::{
        build_title_batch, default_supertable_options, default_tokenizer, default_vector_config,
        fault_storage::{FaultKind, FaultOp, FaultStorage},
        schema_id_title,
    },
};
use rayon::ThreadPoolBuilder;
use tempfile::TempDir;

/// Rows in the width-invariance fixture — comfortably above the widest
/// pool below, so `n_shards = min(width, rows)` is genuinely
/// width-bound and the shard split really differs between widths.
const WIDTH_TEST_ROWS: usize = 40;

/// Superfile split size for the cross-width fixtures, in MiB.
///
/// Shard count is `min(ceil(buffered_bytes / split), pool_threads, rows)`
/// — bytes first, pool width only as a cap. At the shipped split a test
/// batch would always be one shard, so the widest pool would produce the
/// same single superfile as the narrowest and the invariance assertions
/// would pass vacuously. Squeezing the split makes the pool width the
/// binding term, which is the thing under test.
const WIDTH_TEST_SPLIT_MB: u64 = 1;

/// Rows for the fixtures that must span several shards: enough Arrow
/// footprint to clear `WIDTH_TEST_SPLIT_MB` several times over.
const SHARDING_TEST_ROWS: usize = 40_000;
/// Writer-pool widths the invariance test compares.
const POOL_WIDTHS: [usize; 3] = [1, 2, 8];
/// Vector fixture dimensionality (engine minimum is 16).
const DIM: usize = 16;
/// Rows in the vector-bytes pin fixture.
const VECTOR_ROWS: usize = 24;
/// Random-rotation seed for the vector fixture.
const VECTOR_ROT_SEED: u64 = 13;

/// Zero the counters the write contract exempts: superfile count and
/// the per-superfile overhead ride the writer pool's width, distinct
/// terms double-count across shards, and kernel CPU is measured time.
/// Serves the cross-width comparisons; everything left must be equal.
fn deterministic_write(mut stats: OpStats) -> OpStats {
    stats.superfiles_written = 0;
    stats.superfile_bytes_written = 0;
    stats.fts_terms_indexed = 0;
    stats.kernel_cpu_ns = 0;
    stats
}

/// The `title`-only options with an explicit writer-pool width.
fn options_with_pool_width(n_threads: usize) -> SupertableOptions {
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .expect("writer pool"),
    );
    SupertableOptions::new(
        schema_id_title(),
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        Vec::new(),
        Some(default_tokenizer()),
    )
    .expect("valid options")
    .with_writer_pool(pool)
    .with_superfile_buffer_split_mb(WIDTH_TEST_SPLIT_MB)
}

/// The width-test corpus: same titles every call, so every width builds
/// from an identical batch.
fn width_test_batch() -> RecordBatch {
    let titles: Vec<String> = (0..WIDTH_TEST_ROWS)
        .map(|i| format!("row{i:03} common tokens"))
        .collect();
    let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
    build_title_batch(&refs)
}

/// A batch big enough to split across the widest pool: each row carries a
/// kibibyte of title text, so 40k rows clear the 1 MiB split many times.
fn sharding_test_batch() -> RecordBatch {
    let filler = "lorem ipsum dolor sit amet ".repeat(38);
    let titles: Vec<String> = (0..SHARDING_TEST_ROWS)
        .map(|i| format!("row{i:06} common tokens {filler}"))
        .collect();
    let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
    build_title_batch(&refs)
}

/// One scoped append+commit of `batch` into a fresh table built from
/// `options`, returning the write's work stats.
fn scoped_append(options: SupertableOptions, batch: &RecordBatch) -> OpStats {
    let st = Supertable::create(options).expect("create");
    let ((), stats) = with_op_stats(|| {
        st.append(batch).expect("append");
    });
    stats
}

#[test]
fn write_stats_are_invariant_to_writer_pool_width() {
    // The load-bearing determinism test: the priced counters must not
    // depend on how many rayon threads happened to shard the build.
    let batch = sharding_test_batch();
    let snapshots: Vec<OpStats> = POOL_WIDTHS
        .iter()
        .map(|&width| scoped_append(options_with_pool_width(width), &batch))
        .collect();

    let baseline = deterministic_write(snapshots[0]);
    for (width, stats) in POOL_WIDTHS.iter().zip(&snapshots).skip(1) {
        assert_eq!(
            deterministic_write(*stats),
            baseline,
            "priced write counters must match width 1 at width {width}"
        );
    }
    // The mask is load-bearing, not vacuous: the widest pool really did
    // shard more than the single thread.
    assert!(
        snapshots[2].superfiles_written > snapshots[0].superfiles_written,
        "width {} must produce more superfiles than width 1 ({} vs {})",
        POOL_WIDTHS[2],
        snapshots[2].superfiles_written,
        snapshots[0].superfiles_written
    );
}

#[test]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "per-thread CPU clock is Linux procfs (schedstat); off Linux kernel_cpu_ns is always 0"
)]
fn an_append_reports_its_build_cpu() {
    // The write meter's CPU leg. Before the metered fan-out existed, every
    // append reported zero kernel CPU and a consumer pricing that field
    // billed the per-write floor regardless of size. The sharding fixture
    // guarantees a real multi-shard build, whose k-means/encode work is far
    // above the schedstat tick.
    let batch = sharding_test_batch();
    let stats = scoped_append(options_with_pool_width(POOL_WIDTHS[2]), &batch);
    assert!(
        stats.kernel_cpu_ns > 0,
        "a {SHARDING_TEST_ROWS}-row append must report its build CPU; got 0"
    );
}

#[test]
fn planned_write_requests_follow_the_data_not_the_shard_count() {
    // The write-side twin of `planned_read_ranges`, and priceable for the
    // same reason: a PUT is 12.5x a GET in the cost model and never
    // resolves from residency, so requests are a real share of write
    // COGS — but the ACTUAL PUTs are ours, rising with how wide the pool
    // sharded and how often a contended publish retried. What is billed
    // is what the data requires: the objects it occupies at the target
    // size, plus the manifest json and pointer every commit publishes.
    let batch = sharding_test_batch();
    let narrow = scoped_append(options_with_pool_width(POOL_WIDTHS[0]), &batch);
    let wide = scoped_append(options_with_pool_width(POOL_WIDTHS[2]), &batch);

    assert!(
        wide.superfiles_written > narrow.superfiles_written,
        "the fixture must really shard wider ({} vs {}), or the invariance \
         below proves nothing",
        wide.superfiles_written,
        narrow.superfiles_written
    );
    assert_eq!(
        narrow.planned_write_requests, wide.planned_write_requests,
        "more shards must not mean more planned requests"
    );
    assert!(
        narrow.planned_write_requests >= 2,
        "every commit publishes at least the manifest json and the pointer, \
         got {}",
        narrow.planned_write_requests
    );
}

#[test]
fn repeated_identical_appends_report_identical_counters() {
    // Same batch, same (fresh) table state, same pool width: the whole
    // masked snapshot matches, and the recorded superfile count does
    // too (it is width-dependent, not run-dependent).
    let batch = width_test_batch();
    let a = scoped_append(options_with_pool_width(2), &batch);
    let b = scoped_append(options_with_pool_width(2), &batch);
    assert_eq!(
        deterministic_write(a),
        deterministic_write(b),
        "same batch, same table state, same work"
    );
    assert_eq!(
        a.superfiles_written, b.superfiles_written,
        "the shard split is a function of width and rows, not of the run"
    );
}

#[test]
fn an_append_pins_exact_rows_and_subset_fts_bytes() {
    // Exact equality, not >0: a double flush would report 2x and pass a
    // positivity check.
    let batch = width_test_batch();
    let stats = scoped_append(options_with_pool_width(1), &batch);
    assert_eq!(
        stats.rows_written, WIDTH_TEST_ROWS as u64,
        "one committed row per input row"
    );
    assert_eq!(stats.rows_tombstoned, 0, "appends tombstone nothing");
    assert!(
        stats.scalar_bytes_written > 0,
        "the scalar leg counts the buffered Arrow payload"
    );
    // The FTS leg is a subset of the scalar leg, not additional payload.
    assert!(
        stats.fts_text_bytes_written > 0
            && stats.fts_text_bytes_written <= stats.scalar_bytes_written,
        "fts bytes ({}) are a subset of scalar bytes ({})",
        stats.fts_text_bytes_written,
        stats.scalar_bytes_written
    );
    assert_eq!(
        stats.vector_bytes_written, 0,
        "no vector column in this fixture"
    );
    assert!(
        stats.superfiles_written > 0 && stats.superfile_bytes_written > 0,
        "a committed append published at least one superfile"
    );
}

#[test]
fn a_vector_append_pins_exact_payload_bytes() {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(Arc::clone(&item), DIM as i32),
            false,
        ),
    ]));
    let options = SupertableOptions::new(
        Arc::clone(&schema),
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![default_vector_config("emb", VECTOR_ROT_SEED)],
        Some(default_tokenizer()),
    )
    .expect("valid options");

    let flat: Vec<f32> = (0..VECTOR_ROWS * DIM).map(|i| i as f32 * 0.25).collect();
    let titles: Vec<String> = (0..VECTOR_ROWS).map(|i| format!("vec row{i:03}")).collect();
    let fsl = FixedSizeListArray::try_new(
        item,
        DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )
    .expect("FSL");
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(LargeStringArray::from(titles)) as ArrayRef,
            Arc::new(fsl),
        ],
    )
    .expect("batch");

    let st = Supertable::create(options).expect("create");
    let ((), stats) = with_op_stats(|| {
        st.append(&batch).expect("append");
    });
    assert_eq!(stats.rows_written, VECTOR_ROWS as u64);
    assert_eq!(
        stats.vector_bytes_written,
        (VECTOR_ROWS * DIM * mem::size_of::<f32>()) as u64,
        "the vector leg is exactly rows x dim x 4"
    );
}

#[test]
fn a_writer_minted_outside_the_scope_records_nothing() {
    // Pickup happens at writer mint, not at commit time — the same
    // contract the reader has.
    let st = Supertable::create(options_with_pool_width(1)).expect("create");
    let mut w = st.writer().expect("writer");
    let batch = width_test_batch();
    let ((), stats) = with_op_stats(|| {
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    });
    assert_eq!(
        stats.rows_written, 0,
        "pickup happens at writer mint, not inside the scope"
    );
    assert_eq!(stats.superfiles_written, 0);
}

/// A writer holds its ingested work until the commit that publishes it
/// returns Ok. Dropping the writer discards the buffer without committing
/// — silently and by design — so it must discard the counters with it.
/// Every call here returns Ok, which is what makes this the one path a
/// "failed ops return Err" argument does not cover.
#[test]
fn a_dropped_writer_reports_nothing_it_never_committed() {
    let st = Supertable::create(options_with_pool_width(1)).expect("create");
    let batch = width_test_batch();
    let ((), stats) = with_op_stats(|| {
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        drop(w);
    });
    assert_eq!(
        stats.rows_written, 0,
        "the buffer was dropped, so no row was durably indexed"
    );
    assert_eq!(stats.scalar_bytes_written, 0);
    assert_eq!(stats.vector_bytes_written, 0);
    assert_eq!(stats.fts_text_bytes_written, 0);
    assert_eq!(stats.superfiles_written, 0);
}

/// The manifest pointer's URI fragment — the object whose conditional
/// write is the commit's CAS, and so the one to fail to force a retry.
const POINTER_URI_FRAGMENT: &str = "_supertable/current";
/// Rows in the two-column fixture that separates the FTS leg from the
/// scalar leg.
const MIXED_SCHEMA_ROWS: usize = 32;

/// Rows in the parent batch the slice fixture carves from — large enough
/// that billing its whole allocation is unmistakable.
const SLICE_TEST_PARENT_ROWS: usize = 4_000;
/// Rows the slice fixture actually appends.
const SLICE_TEST_ROWS: usize = 10;
/// Mid-array offset for the second slice arm — far enough in that a
/// sizing blind to offsets would visibly bill the wrong window.
const SLICE_TEST_OFFSET: usize = 1_700;

/// Priced bytes measure the rows an append carries, not the allocation
/// they happen to live in. Arrow slices share their parent's buffers, and
/// `get_array_memory_size` sums buffer *capacity* while ignoring offset
/// and length — so a 10-row slice of a 4,000-row batch reported the whole
/// parent. Chunked ingest (read one large batch, append it in zero-copy
/// slices) is exactly that pattern, and it billed every chunk for the
/// entire batch.
#[test]
fn a_sliced_append_is_billed_for_its_own_rows() {
    let titles: Vec<String> = (0..SLICE_TEST_PARENT_ROWS)
        .map(|i| format!("row {i} carrying enough text to allocate a real buffer"))
        .collect();
    let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
    // Two windows: one at offset 0 and one from the middle. Chunked
    // ingest slices at 0, then len, then 2*len — every chunk after the
    // first carries a non-zero offset, and a sizing that honoured length
    // while ignoring offset would pass the first arm and still over-bill
    // every later chunk.
    for offset in [0, SLICE_TEST_OFFSET] {
        let sliced = build_title_batch(&refs).slice(offset, SLICE_TEST_ROWS);
        let standalone = build_title_batch(&refs[offset..offset + SLICE_TEST_ROWS]);

        let st_sliced = Supertable::create(options_with_pool_width(1)).expect("create");
        let ((), from_slice) = with_op_stats(|| {
            st_sliced.append(&sliced).expect("append the slice");
        });
        let st_standalone = Supertable::create(options_with_pool_width(1)).expect("create");
        let ((), from_standalone) = with_op_stats(|| {
            st_standalone
                .append(&standalone)
                .expect("append the standalone batch");
        });

        assert_eq!(
            from_slice.rows_written, from_standalone.rows_written,
            "offset {offset}: both appends carry the same rows"
        );
        assert_eq!(
            from_slice.scalar_bytes_written, from_standalone.scalar_bytes_written,
            "offset {offset}: the same rows must price the same whether \
             they arrive as a slice of a larger batch or as a batch of \
             their own"
        );
        assert_eq!(
            from_slice.fts_text_bytes_written, from_standalone.fts_text_bytes_written,
            "offset {offset}: and so must the indexed-text leg"
        );
    }
}

/// The determinism contract covers commit retries as well as pool width,
/// and the collector promises the manifest pair is counted once per
/// successful commit rather than once per OCC attempt. Only the width
/// half had a test. A lost pointer CAS drives the retry deterministically,
/// so this does not depend on winning a race.
#[test]
fn write_stats_are_invariant_to_a_commit_retry() {
    let uncontended_dir = TempDir::new().expect("tempdir");
    let uncontended: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(uncontended_dir.path()).expect("provider"));
    let baseline_table =
        Supertable::create(default_supertable_options().with_storage(uncontended)).expect("create");
    let ((), baseline) = with_op_stats(|| {
        baseline_table
            .append(&width_test_batch())
            .expect("uncontended append");
    });

    let contended_dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(contended_dir.path()).expect("provider"));
    let faults = FaultStorage::wrap(local);
    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);
    let retried_table =
        Supertable::create(default_supertable_options().with_storage(storage)).expect("create");
    // Lose the pointer CAS exactly once: the commit must reissue and still
    // report its work a single time.
    faults.fail_with(
        FaultKind::Precondition,
        FaultOp::PutIfMatch,
        POINTER_URI_FRAGMENT,
        1,
    );
    let ((), retried) = with_op_stats(|| {
        retried_table
            .append(&width_test_batch())
            .expect("append survives one lost CAS");
    });

    assert!(
        faults.fired() > 0,
        "the fixture must actually lose a CAS, or the retry half is untested"
    );
    assert_eq!(
        retried.rows_written, baseline.rows_written,
        "a retried commit indexes the same rows once"
    );
    assert_eq!(retried.scalar_bytes_written, baseline.scalar_bytes_written);
    assert_eq!(
        retried.fts_text_bytes_written,
        baseline.fts_text_bytes_written
    );
    assert_eq!(
        retried.planned_write_requests, baseline.planned_write_requests,
        "the manifest pair counts once per successful commit, not once per \
         attempt"
    );
}

/// The FTS leg is documented as a subset of the scalar leg rather than
/// additional payload. Every other fixture here has exactly one user
/// column and it *is* the FTS column, so the two counters come out equal
/// by construction and the subset relation is never actually exercised.
/// This one carries an unindexed column too, so the inequality is strict.
#[test]
fn the_fts_leg_is_a_strict_subset_when_a_column_is_unindexed() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("rating", DataType::Int64, false),
    ]));
    let titles: Vec<String> = (0..MIXED_SCHEMA_ROWS)
        .map(|i| format!("document {i} about rust and search"))
        .collect();
    let title_col: ArrayRef = Arc::new(LargeStringArray::from(
        titles.iter().map(String::as_str).collect::<Vec<_>>(),
    ));
    let rating_col: ArrayRef = Arc::new(arrow_array::Int64Array::from(
        (0..MIXED_SCHEMA_ROWS as i64).collect::<Vec<_>>(),
    ));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![title_col, rating_col]).expect("batch");

    let options = SupertableOptions::new(
        Arc::clone(&schema),
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        Vec::new(),
        Some(default_tokenizer()),
    )
    .expect("valid options");
    let st = Supertable::create(options).expect("create");
    let ((), stats) = with_op_stats(|| {
        st.append(&batch).expect("append");
    });

    assert!(
        stats.fts_text_bytes_written > 0,
        "the indexed column carries text"
    );
    assert!(
        stats.fts_text_bytes_written < stats.scalar_bytes_written,
        "the FTS leg counts only the indexed column, so an unindexed \
         column must make it a strict subset: fts={} scalar={}",
        stats.fts_text_bytes_written,
        stats.scalar_bytes_written
    );
}

/// A storage-backed table (mutations require storage for the WAL
/// pipeline) with three committed rows.
fn seeded_storage_table(dir: &TempDir) -> Supertable {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st =
        Supertable::create(default_supertable_options().with_storage(storage)).expect("create");
    st.append(&build_title_batch(&["alpha", "bravo", "charlie"]))
        .expect("seed append");
    st
}

#[test]
fn a_delete_reports_its_tombstoned_rows_and_its_scan() {
    let dir = TempDir::new().expect("tempdir");
    let st = seeded_storage_table(&dir);
    let (outcome, stats) =
        with_op_stats(|| st.delete(col("title").eq(lit("bravo"))).expect("delete"));
    assert_eq!(outcome.n_tombstoned(), 1);
    assert_eq!(
        stats.rows_tombstoned, 1,
        "the delete's write leg reports its retired rows"
    );
    assert_eq!(stats.rows_written, 0, "deletes index no new rows");
    // A pure delete commits no manifest; its tombstone CAS-writes are
    // recorded-only actuals, so the planned counter stays silent.
    assert_eq!(
        stats.planned_write_requests, 0,
        "a delete plans no commit requests"
    );
    // The predicate resolve is real read work the caller asked for, so
    // it reports through the same scope as the write leg. It runs on a
    // fresh, uncached context that carries the collector; routing it
    // through the shared cached one would report zero for an
    // arbitrarily large table scan, and would leave a SELECT-then-
    // delete-by-id path costing more than the equivalent DELETE WHERE.
    assert!(
        stats.planned_read_ranges > 0,
        "the mutation's predicate scan must report its read work"
    );
    // The resolve decodes rows, and `rows_materialized` is the counter
    // that says so. It sat at zero while the CPU, page-byte and range
    // counters beside it all reported, because the resolve collected its
    // DataFrame without harvesting DataFusion's leaf metrics.
    assert!(
        stats.rows_materialized > 0,
        "the predicate scan's decoded rows must reach the counter"
    );
}

#[test]
fn an_update_reports_replacement_rows_and_tombstones() {
    let dir = TempDir::new().expect("tempdir");
    let st = seeded_storage_table(&dir);
    let (outcome, stats) = with_op_stats(|| {
        st.update(
            col("title").eq(lit("bravo")),
            &build_title_batch(&["bravo-replacement"]),
        )
        .expect("update")
    });
    assert_eq!(outcome.matched(), 1);
    assert_eq!(
        stats.rows_written, 1,
        "the update's replacement rows count as written"
    );
    assert_eq!(
        stats.rows_tombstoned, 1,
        "the update's retired rows count as tombstoned"
    );
    // An update's plan: its one WAL-preallocated replacement superfile
    // plus the manifest json + pointer of its commit.
    assert_eq!(
        stats.planned_write_requests, 3,
        "an update plans exactly one data object + the manifest pair"
    );
    // The replacement payload is ingested work and must be measured the
    // same way an append's is. This reported zero until the byte legs
    // were captured at call time — the batch is dropped once IPC-encoded,
    // so there is no second chance at commit.
    assert!(
        stats.scalar_bytes_written > 0,
        "an update's replacement rows must report their ingested bytes"
    );
    assert!(
        stats.fts_text_bytes_written > 0,
        "the replacement's indexed text is part of that payload"
    );
    assert!(
        stats.fts_text_bytes_written <= stats.scalar_bytes_written,
        "the FTS leg is a subset of the scalar leg, not additional payload"
    );
}

/// An update's replacement payload and an append of the very same batch
/// must price identically. Both are one row of the caller's own columns;
/// the `_id` the engine mints belongs to neither. Append measured the
/// id-bearing batch until this was pinned, so every update looked
/// cheaper than the append that wrote the same data.
#[test]
fn an_update_and_an_equivalent_append_price_the_same_payload() {
    let batch = build_title_batch(&["bravo-replacement"]);

    let update_dir = TempDir::new().expect("tempdir");
    let update_table = seeded_storage_table(&update_dir);
    let (outcome, update_stats) = with_op_stats(|| {
        update_table
            .update(col("title").eq(lit("bravo")), &batch)
            .expect("update")
    });
    assert_eq!(outcome.matched(), 1, "the fixture must match one row");

    let append_dir = TempDir::new().expect("tempdir");
    let append_table = seeded_storage_table(&append_dir);
    let ((), append_stats) = with_op_stats(|| {
        append_table.append(&batch).expect("append");
    });

    assert_eq!(
        update_stats.rows_written, append_stats.rows_written,
        "one replacement row is one written row, same as an append"
    );
    assert_eq!(
        update_stats.scalar_bytes_written, append_stats.scalar_bytes_written,
        "the same batch must price the same whether it arrives as an \
         update's replacement or as a plain append"
    );
    assert_eq!(
        update_stats.fts_text_bytes_written, append_stats.fts_text_bytes_written,
        "and so must its indexed-text leg"
    );
}

#[test]
fn optimize_does_not_re_bill_ingested_rows() {
    // Compaction rewrites rows the caller already paid to ingest; if
    // optimize() ever lands in rows_written, every optimize re-counts
    // the whole table. Recorded output-shape counters may move; the
    // priced input-shape counters must stay zero.
    let dir = TempDir::new().expect("tempdir");
    let st = seeded_storage_table(&dir);
    // A second commit gives compaction something to merge.
    st.append(&build_title_batch(&["delta", "echo"]))
        .expect("second append");
    let before = st.reader().expect("reader").n_superfiles();
    let ((), stats) = with_op_stats(|| {
        st.optimize(&OptimizeOptions::compact(CompactionSettings {
            // Without this the default is 50, so two superfiles never
            // select a merge job and the zero assertions below hold
            // because nothing ran.
            min_superfiles_for_merge: 2,
            target_superfile_size_mb: 1,
            min_fill_percent: 1,
            ..CompactionSettings::default()
        }))
        .expect("optimize");
    });
    // The zeros below only mean something if a merge actually ran. With
    // the shipped `min_superfiles_for_merge` this fixture selected no job
    // at all, so the assertions passed because nothing happened.
    let after = st.reader().expect("reader").n_superfiles();
    assert!(
        after < before,
        "the fixture must really merge ({before} -> {after}), or the \
         zero assertions below prove nothing"
    );
    assert_eq!(
        stats.rows_written, 0,
        "optimize must never re-count ingested rows"
    );
    assert_eq!(stats.scalar_bytes_written, 0);
    assert_eq!(stats.vector_bytes_written, 0);
    assert_eq!(stats.fts_text_bytes_written, 0);
}
