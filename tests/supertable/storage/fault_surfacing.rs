// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Storage faults must surface as clean errors — never a panic, a wrong
//! result, or a mislabeled contention retry — and the paths that promise
//! recovery must actually recover once the fault clears.
//!
//! Each test wraps LocalFS in [`FaultStorage`], arms one fault at the
//! exact operation a subsystem depends on (superfile PUT, manifest-list
//! PUT, pointer CAS, pointer GET, list GET, cold range-GET, GC delete),
//! asserts the operation's caller reports the failure without corrupting
//! table state, and — where the API contract says so — retries the same
//! call with the fault cleared and asserts it completes.

#![deny(clippy::unwrap_used)]

use std::{sync::Arc, time::Duration};

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use datafusion::prelude::{col, lit};
use infino::{
    InfinoError,
    config::OptimizeOptions,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::{
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        fts::reader::BoolMode,
    },
    supertable::{
        SuperfileUri, Supertable, SupertableOptions,
        manifest::commit::{MANIFEST_DIR, POINTER_PATH, manifest_uri},
    },
    test_helpers::{
        build_title_batch, decimal128_id_field, decimal128_ids, default_supertable_options,
        default_tokenizer, default_vector_config,
        fault_storage::{FaultKind, FaultOp, FaultStorage},
        lazy_foreground_disk_cache,
    },
};
use rayon::ThreadPoolBuilder;
use tempfile::TempDir;

/// Dimension of the vector fixture's column; matches
/// `default_vector_config`.
const VECTOR_DIM: usize = 16;
/// Rows in the vector fixture, planted in two directions so the commit
/// assigns them across cells rather than into one.
const VECTOR_ROWS: usize = 24;
/// Rotation seed for the vector fixture's column.
const VECTOR_ROT_SEED: u64 = 21;
/// Writer threads for the vector fixture, and so the commit's packed-shard
/// count. Two is the smallest width that puts one shard on the wire while
/// another is still packing — the shape the pipelined publish exists for.
const VECTOR_WRITERS: usize = 2;

/// Commits made before the faulted optimize, so there is genuinely more
/// than one superfile for maintenance to consider.
const OPTIMIZE_FAULT_COMMITS: usize = 3;

/// Top-k for the recovery searches; above corpus size.
const FTS_TOP_K: usize = 8;
/// Generous rule budget for fanout paths (cold search issues many range
/// GETs); large enough that every fetch of the failing phase hits it.
const FANOUT_FAULTS: usize = 1024;
/// Rule budget for the sidecar-CAS test: comfortably above the engine's own
/// per-sidecar retry budget, so every attempt in that loop loses its race.
const SIDECAR_CAS_FAULTS: usize = 64;
/// Suffix of the per-superfile tombstone sidecars the delete path CAS-writes.
/// Targeting it leaves superfile, manifest, and WAL writes untouched.
const TOMBSTONES_SUFFIX: &str = ".tombstones";

/// A LocalFS-backed table wrapped in `FaultStorage`, with one committed
/// batch, plus the tempdir guard.
fn faulted_table() -> (Supertable, Arc<FaultStorage>, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let faults = FaultStorage::wrap(local);
    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);
    let st =
        Supertable::create(default_supertable_options().with_storage(storage)).expect("create");
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["first commit alpha"]))
        .expect("append");
    w.commit().expect("commit");
    assert_eq!(st.manifest_id(), 1);
    (st, faults, dir)
}

/// A LocalFS-backed VECTOR table wrapped in `FaultStorage`, empty, with a
/// batch ready to append. A vector table is what puts a commit on the
/// pipelined publish — a table with no vector columns never enters the
/// drain-commit path at all — so this is the fixture for faulting a PUT
/// the uploader issues while other shards are still packing.
fn faulted_vector_table() -> (Supertable, RecordBatch, Arc<FaultStorage>, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let faults = FaultStorage::wrap(local);
    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);

    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(Arc::clone(&item), VECTOR_DIM as i32),
            false,
        ),
    ]));
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(VECTOR_WRITERS)
            .build()
            .expect("rayon pool"),
    );
    let options = SupertableOptions::new(
        Arc::clone(&schema),
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![default_vector_config("emb", VECTOR_ROT_SEED)],
        Some(default_tokenizer()),
    )
    .expect("valid options")
    .with_writer_pool(pool)
    .with_storage(storage);

    let titles: Vec<String> = (0..VECTOR_ROWS).map(|i| format!("vec row{i:03}")).collect();
    let mut flat = vec![0.0f32; VECTOR_ROWS * VECTOR_DIM];
    for row in 0..VECTOR_ROWS {
        flat[row * VECTOR_DIM + usize::from(row >= VECTOR_ROWS / 2)] = 1.0;
    }
    let fsl = FixedSizeListArray::try_new(
        item,
        VECTOR_DIM as i32,
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
    (st, batch, faults, dir)
}

#[test]
fn pipelined_commit_surfaces_shard_put_fault_and_recovers() {
    let (st, batch, faults, _dir) = faulted_vector_table();

    // The uploader starts PUTting as soon as the first shard seals, so this
    // fault lands mid-commit rather than in a publish wave after the pack.
    faults.fail(FaultOp::PutAtomic, "data/", 1);
    let mut w = st.writer().expect("writer");
    w.append(&batch).expect("append");
    let err = w
        .commit()
        .expect_err("a failed shard upload must fail the commit");
    assert!(
        format!("{err:?}").contains("injected"),
        "the injected fault must reach the caller, got: {err:?}"
    );
    assert_eq!(faults.fired(), 1, "exactly the armed fault fired");
    assert_eq!(
        st.manifest_id(),
        0,
        "a commit that lost a shard upload publishes nothing"
    );
    assert_eq!(
        st.reader().expect("reader").n_superfiles(),
        0,
        "shards uploaded before the failure are orphans, not table state"
    );

    // Same contract the text path keeps: the buffered rows survive the
    // failure, so the retry publishes once the fault is spent.
    w.commit().expect("retry after the fault clears");
    assert_eq!(st.manifest_id(), 1, "the retry publishes");
    let reader = st.reader().expect("reader");
    assert!(
        reader.n_superfiles() >= 1,
        "the retried commit published its shards"
    );
    let mut query = vec![0.0f32; VECTOR_DIM];
    query[0] = 1.0;
    let hits = reader
        .vector_search("emb", &query, VECTOR_ROWS, Default::default(), None, None)
        .expect("vector search after recovery");
    let found: usize = hits.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        found, VECTOR_ROWS,
        "the retry must restore every row, not merely some of them"
    );
}

/// Maintenance is not allowed to damage the table it is maintaining. A
/// storage failure anywhere in `optimize` must leave every committed row
/// readable and the next maintenance pass able to complete — whether the
/// faulted pass reports the error or absorbs it as a skipped step.
#[test]
fn optimize_survives_a_storage_fault_without_losing_rows() {
    let (st, batch, faults, _dir) = faulted_vector_table();
    for _ in 0..OPTIMIZE_FAULT_COMMITS {
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    let rows_before = st.reader().expect("reader").n_superfiles();
    assert!(
        rows_before >= OPTIMIZE_FAULT_COMMITS,
        "the sources committed"
    );

    faults.fail(FaultOp::PutAtomic, "data/", 1);
    let outcome = st.optimize(&OptimizeOptions::default());
    if let Err(error) = &outcome {
        assert!(
            format!("{error:?}").contains("injected"),
            "a maintenance failure must name the injected fault: {error:?}"
        );
    }
    assert_eq!(faults.fired(), 1, "the armed fault fired");

    // Whatever the pass decided, the table is intact: every committed row
    // is still searchable, and maintenance completes on a second pass.
    let reader = st.reader().expect("reader");
    let mut query = vec![0.0f32; VECTOR_DIM];
    query[0] = 1.0;
    let expected = VECTOR_ROWS * OPTIMIZE_FAULT_COMMITS;
    let hits = reader
        .vector_search("emb", &query, expected, Default::default(), None, None)
        .expect("search after a faulted optimize");
    let found: usize = hits.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        found, expected,
        "every committed row survives a faulted maintenance pass"
    );
    st.optimize(&OptimizeOptions::default())
        .expect("maintenance completes once the fault clears");
}

/// The drain publishes the hidden vector index from durable user
/// superfiles. A storage failure there must surface, leave the user table
/// exactly as it was (it is the durable source, never rewritten by the
/// drain), and leave the drain re-runnable — the next attempt rebuilds
/// from the same user superfiles and serves.
#[test]
fn drain_surfaces_a_hidden_index_put_fault_and_re_runs() {
    let (st, batch, faults, _dir) = faulted_vector_table();
    {
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    let committed_id = st.manifest_id();
    assert!(committed_id >= 1, "the source commit landed");

    // Fail the hidden index's own superfile PUT. The token is the hidden
    // table's prefix, so the user table's writes are untouched.
    faults.fail(FaultOp::PutAtomic, "_vector_index/data/", 1);
    let failed = st.drain_vectors_to_cells_sync();
    assert!(
        failed.is_err(),
        "a failed hidden-index PUT must surface, got {failed:?}"
    );
    assert_eq!(faults.fired(), 1, "exactly the armed fault fired");
    assert_eq!(
        st.manifest_id(),
        committed_id,
        "the user table is the durable source and the drain never rewrites it"
    );
    // A wrapped provider is not a local directory, so the mmap fast path
    // must decline it: `local_path` returning `Some` here would hand an
    // open-time caller a path that does not name the object it wants.
    let wrapped: &dyn StorageProvider = faults.as_ref();
    assert!(
        wrapped.local_path("data/anything").is_none(),
        "a wrapper provider has no local directory to mmap from"
    );

    // Fault spent: the drain re-runs from the same user superfiles.
    st.drain_vectors_to_cells_sync()
        .expect("drain re-runs once the fault clears");
    let reader = st.reader().expect("reader");
    let mut query = vec![0.0f32; VECTOR_DIM];
    query[0] = 1.0;
    let hits = reader
        .vector_search("emb", &query, VECTOR_ROWS, Default::default(), None, None)
        .expect("post-drain vector search");
    let found: usize = hits.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        found, VECTOR_ROWS,
        "the re-run drain serves every row, not a partial rebuild"
    );
}

#[test]
fn commit_surfaces_superfile_put_fault_and_recovers() {
    let (st, faults, _dir) = faulted_table();

    faults.fail(FaultOp::PutAtomic, "data/", 1);
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["second commit beta"]))
        .expect("append");
    let err = w
        .commit()
        .expect_err("superfile PUT fault must fail the commit");
    assert!(
        format!("{err}").contains("injected fault") || format!("{err:?}").contains("injected"),
        "the injected fault must be visible in the surfaced error, got: {err:?}"
    );
    assert_eq!(faults.fired(), 1, "exactly the armed fault fired");
    assert_eq!(st.manifest_id(), 1, "nothing published");

    // The commit contract keeps buffered rows on failure so the caller
    // can retry; with the fault cleared the same writer must publish.
    w.commit().expect("retry after the fault clears");
    assert_eq!(st.manifest_id(), 2);
    assert_eq!(st.reader().expect("reader").n_superfiles(), 2);
}

#[test]
fn commit_surfaces_manifest_list_put_fault_as_storage_error() {
    let (st, faults, _dir) = faulted_table();

    faults.fail(FaultOp::PutAtomic, &format!("{MANIFEST_DIR}/manifest-"), 1);
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["second commit beta"]))
        .expect("append");
    let err = w.commit().expect_err("list PUT fault must fail the commit");
    // A transient list-PUT failure is a storage error. Only
    // PreconditionFailed (a lost race) may be folded into the
    // write-contention retry story; masking real faults as contention
    // would burn the whole retry budget re-hitting a broken store. Tie
    // the failure to the injection so a fault that never fired (or a
    // different storage error) can't false-pass this assert.
    assert_eq!(faults.fired(), 1, "exactly the armed fault fired");
    let rendered = format!("{err:?}");
    assert!(
        rendered.contains("injected") && !format!("{err}").contains("contention"),
        "the injected fault must surface as a storage error, not contention: {rendered}"
    );
    assert_eq!(st.manifest_id(), 1, "nothing published");

    // No orphan either: the list PUT itself failed, so the id was never
    // occupied and the retry republishes densely at id 2.
    w.commit().expect("retry after the fault clears");
    assert_eq!(st.manifest_id(), 2);
}

#[test]
fn commit_surfaces_pointer_cas_fault_without_publishing() {
    let (st, faults, _dir) = faulted_table();

    faults.fail(FaultOp::PutIfMatch, POINTER_PATH, 1);
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["second commit beta"]))
        .expect("append");
    let err = w
        .commit()
        .expect_err("pointer CAS fault must fail the commit");
    assert_eq!(faults.fired(), 1, "exactly the armed fault fired");
    let rendered = format!("{err:?}");
    assert!(
        rendered.contains("injected") && !format!("{err}").contains("contention"),
        "the injected CAS fault must surface as a storage error, not contention: {rendered}"
    );

    // The visibility barrier never moved: the handle still sees only the
    // first commit. (The already-PUT list at id 2 is crash residue — the
    // recovery story for it is the orphan-skip fix, not this test.)
    assert_eq!(st.manifest_id(), 1, "commit did not publish");
    assert_eq!(st.reader().expect("reader").n_superfiles(), 1);
}

/// A refused credential must reach the caller as
/// [`InfinoError::PermissionDenied`], not as the generic backend fault a
/// transient failure produces: the two call for different responses — retry
/// the transient one, supply fresh credentials for this one — so collapsing
/// them leaves a caller with no way to react.
///
/// The commit path is the one that erased it hardest: the condition crosses
/// four wrappers (storage -> commit -> build -> append flush) on its way out.
#[test]
fn a_refused_credential_surfaces_as_permission_denied_not_a_generic_fault() {
    // The superfile PUT is refused mid-commit.
    let (st, faults, _dir) = faulted_table();
    faults.fail_with(FaultKind::PermissionDenied, FaultOp::PutAtomic, "data/", 1);
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["second commit beta"]))
        .expect("append");
    let err = InfinoError::from(w.commit().expect_err("a refused PUT must fail the commit"));
    assert!(
        matches!(err, InfinoError::PermissionDenied(_)),
        "commit reported {err:?}"
    );
    assert_eq!(st.manifest_id(), 1, "nothing published");
}

/// Open reads the pointer before anything else, so a refused credential
/// there is the first thing a caller sees.
#[test]
fn open_surfaces_a_refused_credential_as_permission_denied() {
    let (st, faults, _dir) = faulted_table();
    drop(st);

    faults.fail_with(FaultKind::PermissionDenied, FaultOp::Get, POINTER_PATH, 1);
    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);
    let err = InfinoError::from(
        Supertable::open(default_supertable_options().with_storage(storage))
            .expect_err("a refused pointer GET must fail the open"),
    );
    assert!(
        matches!(err, InfinoError::PermissionDenied(_)),
        "open reported {err:?}"
    );
}

#[test]
fn open_surfaces_pointer_get_fault_and_recovers() {
    let (st, faults, _dir) = faulted_table();
    drop(st);

    faults.fail(FaultOp::Get, POINTER_PATH, 1);
    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);
    let err = Supertable::open(default_supertable_options().with_storage(storage))
        .expect_err("pointer GET fault must fail the open");
    assert!(
        format!("{err:?}").to_lowercase().contains("storage")
            || format!("{err}").contains("injected"),
        "open must surface the storage failure, got: {err:?}"
    );

    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);
    let st = Supertable::open(default_supertable_options().with_storage(storage))
        .expect("open recovers once the fault clears");
    assert_eq!(st.manifest_id(), 1);
}

#[test]
fn open_surfaces_manifest_list_get_fault_and_recovers() {
    let (st, faults, _dir) = faulted_table();
    drop(st);

    faults.fail(FaultOp::Get, &manifest_uri(1), 1);
    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);
    Supertable::open(default_supertable_options().with_storage(storage))
        .expect_err("manifest-list GET fault must fail the open");
    assert!(faults.fired() >= 1, "the armed list fault fired");

    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);
    let st = Supertable::open(default_supertable_options().with_storage(storage))
        .expect("open recovers once the fault clears");
    assert_eq!(st.reader().expect("reader").n_superfiles(), 1);
}

/// Superfile bytes for the disk-cache cold-read test: four docs, two of
/// which contain the probe term "special".
fn fts_superfile_bytes() -> Bytes {
    let schema = Arc::new(Schema::new(vec![
        decimal128_id_field("doc_id"),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut builder = SuperfileBuilder::new(opts).expect("builder");
    let ids = decimal128_ids(vec![1u64, 2, 3, 4]);
    let titles = LargeStringArray::from(vec![
        "alpha bravo special",
        "charlie delta",
        "echo special foxtrot",
        "gamma hotel",
    ]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)]).expect("batch");
    builder.add_batch(&batch, &[]).expect("add_batch");
    Bytes::from(builder.finish().expect("finish"))
}

/// Cold reads live at the disk-cache layer — at supertable scale a small
/// table is served outright from its manifest part (zero superfile
/// reads), so the fault goes where the bytes actually flow: the lazy
/// cold fetch. It must surface the failure, and the same URI must serve
/// a correct BM25 answer once the fault clears.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_reader_surfaces_range_fault_and_recovers() {
    let store_dir = TempDir::new().expect("storage");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));
    let faults = FaultStorage::wrap(local);
    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);

    let uri = SuperfileUri::new_v4();
    storage
        .put_atomic(&uri.storage_path(), fts_superfile_bytes())
        .await
        .expect("seed superfile");

    let cache_dir = TempDir::new().expect("cache tempdir");
    let cache = lazy_foreground_disk_cache(Arc::clone(&storage), cache_dir.path());

    faults.fail(FaultOp::GetRange, "data/", FANOUT_FAULTS);
    faults.fail(FaultOp::Get, "data/", FANOUT_FAULTS);
    faults.fail(FaultOp::Head, "data/", FANOUT_FAULTS);
    assert!(
        cache.reader(&uri).await.is_err(),
        "a cold fetch whose reads all fail must error, not hand out a reader"
    );
    assert!(faults.fired() >= 1, "the cold fetch actually hit the fault");

    faults.clear();
    let reader = cache
        .reader(&uri)
        .await
        .expect("the same cold fetch recovers once the fault clears");
    let fts = reader.fts().expect("fts");
    let hits = fts
        .search("title", &["special"], FTS_TOP_K, BoolMode::Or)
        .await
        .expect("bm25 over the recovered reader");
    assert_eq!(hits.len(), 2, "two docs contain the probe term");
}

#[test]
fn gc_counts_delete_faults_without_failing_the_sweep() {
    let (st, faults, _dir) = faulted_table();

    // An unreferenced object under manifest/ — plant it through the
    // faulted provider (no rules armed yet, passes through).
    futures::executor::block_on(async {
        Arc::<FaultStorage>::clone(&faults)
            .put_atomic(&manifest_uri(7), Bytes::from_static(b"orphan"))
            .await
    })
    .expect("plant orphan");

    faults.fail(FaultOp::Delete, &manifest_uri(7), FANOUT_FAULTS);
    let report = st.gc(Duration::ZERO).expect("gc tolerates delete faults");
    assert!(
        report.delete_errors >= 1,
        "the failed delete is reported, not swallowed: {report:?}"
    );

    faults.clear();
    let report = st.gc(Duration::ZERO).expect("gc");
    assert_eq!(report.delete_errors, 0);
    let gone = futures::executor::block_on(async {
        Arc::<FaultStorage>::clone(&faults)
            .get(&manifest_uri(7))
            .await
    });
    assert!(gone.is_err(), "the orphan is reclaimed once deletes heal");
}

/// The mirror image of the tests above. Where a transient fault must never
/// be dressed up as contention, a genuine lost CAS must be *reported* as
/// contention: `delete` drives per-superfile tombstone sidecars through a
/// CAS loop, and losing every attempt is the one commit failure worth
/// retrying — nothing partial is visible and a reissue re-resolves the
/// predicate against fresh state. It has to reach the caller as the
/// retryable [`InfinoError::Conflict`], not as an opaque backend fault a
/// serving layer can only give up on.
#[test]
fn delete_losing_the_sidecar_cas_surfaces_a_retryable_conflict() {
    let (st, faults, _dir) = faulted_table();

    faults.fail_with(
        FaultKind::Precondition,
        FaultOp::PutIfMatch,
        TOMBSTONES_SUFFIX,
        SIDECAR_CAS_FAULTS,
    );
    let err = st
        .delete(col("title").eq(lit("first commit alpha")))
        .expect_err("a sidecar CAS that never lands must fail the delete");

    assert!(
        matches!(err, InfinoError::Conflict(_)),
        "a lost sidecar CAS must surface as a retryable Conflict, got {err:?}"
    );
    assert!(
        faults.fired() > 1,
        "the engine must spend its own CAS retries before surfacing, fired {}",
        faults.fired()
    );

    // Retryable means retryable: with the peer writer gone, the same delete
    // lands and the row is tombstoned.
    faults.clear();
    let stats = st
        .delete(col("title").eq(lit("first commit alpha")))
        .expect("the reissued delete succeeds once contention clears");
    assert_eq!(stats.n_tombstoned(), 1);
}
