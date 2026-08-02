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

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::{
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        fts::reader::BoolMode,
    },
    supertable::{
        SuperfileUri, Supertable,
        manifest::commit::{MANIFEST_DIR, POINTER_PATH, manifest_uri},
    },
    test_helpers::{
        build_title_batch, decimal128_id_field, decimal128_ids, default_supertable_options,
        default_tokenizer,
        fault_storage::{FaultOp, FaultStorage},
        lazy_foreground_disk_cache,
    },
};
use tempfile::TempDir;

/// Top-k for the recovery searches; above corpus size.
const FTS_TOP_K: usize = 8;
/// Generous rule budget for fanout paths (cold search issues many range
/// GETs); large enough that every fetch of the failing phase hits it.
const FANOUT_FAULTS: usize = 1024;

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
    // would burn the whole retry budget re-hitting a broken store.
    let rendered = format!("{err}");
    assert!(
        !rendered.contains("contention"),
        "a transient fault must not be mislabeled as contention: {rendered}"
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
    let rendered = format!("{err}");
    assert!(
        !rendered.contains("contention"),
        "a transient CAS fault must not be mislabeled as contention: {rendered}"
    );

    // The visibility barrier never moved: the handle still sees only the
    // first commit. (The already-PUT list at id 2 is crash residue — the
    // recovery story for it is the orphan-skip fix, not this test.)
    assert_eq!(st.manifest_id(), 1, "commit did not publish");
    assert_eq!(st.reader().expect("reader").n_superfiles(), 1);
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
