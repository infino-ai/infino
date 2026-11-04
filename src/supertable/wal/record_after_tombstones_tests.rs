// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A mutation's hidden deleted-set record, under a concurrent writer.
//!
//! The tombstone phase records the rows it retires in the hidden vector
//! index's deleted set after the tombstones are durable. These tests pin
//! that the mutation reports its real outcome however that record fares
//! against other writers committing to the hidden index: a record that
//! loses races is retried, one that never lands within its budget is left
//! to WAL recovery, and in neither case does the mutation report a
//! conflict that would invite reissuing it. They read WAL state documents
//! and the hidden manifest directly, so they live in-crate.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

use arrow_array::{FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{Expr, col, lit};
use tempfile::TempDir;

use crate::{
    storage::{LocalFsStorageProvider, StorageProvider},
    supertable::{
        Supertable, SupertableOptions, hidden_deleted,
        mutations::{CommitError, MutationStats},
        wal::{
            WalStore,
            lease::DEFAULT_LEASE_DURATION,
            recovery::scan_and_recover,
            state_doc::{OpKind, SupertableHandleId, TombstoneOutcome, WalState},
        },
        writer::{RecordRetryBudget, SupertableWriter, record_hidden_deleted_ids},
    },
    test_helpers::{
        default_vector_config, distinct_unit_vectors,
        fault_storage::{FaultKind, FaultOp, FaultStorage},
    },
};

/// Matches `default_vector_config`'s dimension.
const DIM: usize = 16;
/// Random-rotation seed for the fixture's vector index.
const VECTOR_ROT_SEED: u64 = 11;
/// Rows each mutation retires.
const RETIRED_ROWS: usize = 10;
/// Rows no mutation touches, so a mutation that over-reaches shows.
const KEPT_ROWS: usize = 3;
/// Title of the rows the mutations target.
const TARGET_TITLE: &str = "retire-me";
/// Title of the rows no mutation targets.
const KEPT_TITLE: &str = "keep-me";
/// Title an update gives its replacement rows.
const REPLACEMENT_TITLE: &str = "replacement";
/// The user table's commit retry budget. Kept small so a record that never
/// lands spends its whole budget quickly.
const COMMIT_RETRIES: u32 = 2;
/// Commit attempts the deleted-set record gets with [`COMMIT_RETRIES`]: four
/// times the table's commit retry budget.
const RECORD_ATTEMPTS: usize = 8;
/// Lost races the record rides out in the "lands late" tests: fewer than
/// [`RECORD_ATTEMPTS`], more than [`COMMIT_RETRIES`].
const LOST_RACES: usize = 5;
/// More lost races than any budget spends.
const UNLIMITED_LOST_RACES: usize = 10_000;
/// A URI fragment only the hidden vector index's pointer contains. Its
/// storage sits under a prefix ending `_vector_index`; the user table's
/// pointer has no such prefix.
const HIDDEN_POINTER: &str = "vector_index/_supertable/current";
/// Owner id of the recovery sweep these tests run, distinct from the
/// writer's handle.
const SWEEP_OWNER: i128 = 0x5EED;
/// Delete rounds in the concurrent-writer test.
const CONCURRENT_ROUNDS: usize = 4;
/// The table's commit retry budget in the concurrent-writer test: the
/// engine default, so the record rides out the race as it would in use.
const CONCURRENT_COMMIT_RETRIES: u32 = 10;
/// Pause between the concurrent writer's commits to the hidden index, so it
/// commits as often as a busy background writer rather than in a spin.
const PEER_COMMIT_PAUSE: Duration = Duration::from_millis(20);

fn vector_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]))
}

fn vector_options(storage: &Arc<dyn StorageProvider>) -> SupertableOptions {
    vector_options_with_retries(storage, COMMIT_RETRIES)
}

fn vector_options_with_retries(
    storage: &Arc<dyn StorageProvider>,
    commit_retries: u32,
) -> SupertableOptions {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("writer pool"),
    );
    SupertableOptions::new(
        vector_schema(),
        vec![],
        vec![default_vector_config("emb", VECTOR_ROT_SEED)],
    )
    .expect("valid options")
    .with_writer_pool(pool)
    .with_storage(Arc::clone(storage))
    .with_max_commit_retries(commit_retries)
}

/// `titles.len()` rows, one distinct unit vector each, seeded by `seed`.
fn batch(titles: &[&str], seed: u64) -> RecordBatch {
    let flat = distinct_unit_vectors(titles.len(), DIM, seed);
    let embeddings = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM as i32,
        Arc::new(Float32Array::from(flat)),
        None,
    )
    .expect("embedding column");
    RecordBatch::try_new(
        vector_schema(),
        vec![
            Arc::new(LargeStringArray::from(titles.to_vec())),
            Arc::new(embeddings),
        ],
    )
    .expect("batch matches schema")
}

/// A vector table over fault-injecting storage holding [`RETIRED_ROWS`] rows
/// titled [`TARGET_TITLE`] and [`KEPT_ROWS`] titled [`KEPT_TITLE`].
fn seeded_table() -> (TempDir, Arc<FaultStorage>, Supertable) {
    seeded_table_with_retries(COMMIT_RETRIES)
}

/// [`seeded_table`] with the given commit retry budget.
fn seeded_table_with_retries(commit_retries: u32) -> (TempDir, Arc<FaultStorage>, Supertable) {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let faults = FaultStorage::wrap(local);
    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);
    let table =
        Supertable::create(vector_options_with_retries(&storage, commit_retries)).expect("create");
    assert!(
        table.inner().vector_index_table.is_some(),
        "a vector table carries a hidden vector index"
    );
    let mut titles = vec![TARGET_TITLE; RETIRED_ROWS];
    titles.extend([KEPT_TITLE; KEPT_ROWS]);
    let mut writer = table.writer().expect("writer");
    writer.append(&batch(&titles, 1)).expect("append");
    writer.commit().expect("commit appends");
    (dir, faults, table)
}

fn titled(title: &str) -> Expr {
    col("title").eq(lit(title.to_string()))
}

/// The `_id`s of the rows `predicate` matches now.
fn ids_matching(table: &Supertable, predicate: Expr) -> Vec<i128> {
    let mut ids = table
        .reader_strong()
        .expect("reader")
        .scan_ids_matching(predicate)
        .expect("scan ids");
    ids.sort_unstable();
    ids
}

/// The hidden index's deleted set as this handle sees it.
fn hidden_deleted_set(table: &Supertable) -> Vec<i128> {
    let hidden = table
        .inner()
        .vector_index_table
        .as_ref()
        .expect("hidden index");
    hidden_deleted::deleted_user_ids(&hidden.inner().manifest.load_full())
        .expect("decode deleted set")
        .as_ref()
        .clone()
}

fn contains_all(set: &[i128], ids: &[i128]) -> bool {
    ids.iter().all(|id| set.binary_search(id).is_ok())
}

fn contains_none(set: &[i128], ids: &[i128]) -> bool {
    ids.iter().all(|id| set.binary_search(id).is_err())
}

/// The one outcome of a commit that drove exactly one mutation.
fn sole_outcome(
    table: &Supertable,
    mutate: impl FnOnce(&mut SupertableWriter),
) -> Result<MutationStats, CommitError> {
    let mut writer = table.writer().expect("writer");
    mutate(&mut writer);
    let result = writer.commit()?;
    assert_eq!(result.outcomes.len(), 1, "one mutation was buffered");
    Ok(result.outcomes[0].clone())
}

fn delete_targets(table: &Supertable) -> Result<MutationStats, CommitError> {
    sole_outcome(table, |writer| {
        let pending = writer.delete(titled(TARGET_TITLE)).expect("buffer delete");
        assert_eq!(pending.matched, RETIRED_ROWS);
    })
}

fn update_targets(table: &Supertable) -> Result<MutationStats, CommitError> {
    sole_outcome(table, |writer| {
        let pending = writer
            .update(
                titled(TARGET_TITLE),
                batch(&[REPLACEMENT_TITLE; RETIRED_ROWS], 2),
            )
            .expect("buffer update");
        assert_eq!(pending.matched, RETIRED_ROWS);
    })
}

fn assert_true_counts(stats: &MutationStats) {
    assert_eq!(
        stats.matched, RETIRED_ROWS,
        "matched is the real match count"
    );
    assert_eq!(
        stats.n_tombstoned, RETIRED_ROWS,
        "every matched row was retired"
    );
    assert_eq!(stats.n_not_found, 0);
}

fn wal_store(faults: &Arc<FaultStorage>) -> WalStore {
    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(faults);
    WalStore::new(storage)
}

/// The table's WAL state docs, as `(op_kind, state, lease held, every target
/// settled)`.
async fn wal_docs(faults: &Arc<FaultStorage>) -> Vec<(OpKind, WalState, bool, bool)> {
    let store = wal_store(faults);
    let mut docs = Vec::new();
    for wal_id in store.list_wal_ids().await.expect("list wal ids") {
        let (doc, _) = store.read(wal_id).await.expect("read wal doc");
        let settled = doc
            .tombstone_progress
            .iter()
            .all(|entry| entry.outcome != TombstoneOutcome::Pending);
        docs.push((doc.op_kind, doc.state, doc.lease.is_some(), settled));
    }
    docs
}

/// A delete whose deleted-set record loses the race a few times — more than
/// the table's commit retry budget — then lands: one call, the real counts,
/// the ids in the hidden deleted set, and no WAL left behind.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_delete_whose_record_loses_some_races_lands_it_and_reports_once() {
    let (_dir, faults, table) = seeded_table();
    let targets = ids_matching(&table, titled(TARGET_TITLE));
    assert_eq!(targets.len(), RETIRED_ROWS);

    faults.fail_with(
        FaultKind::Precondition,
        FaultOp::PutIfMatch,
        HIDDEN_POINTER,
        LOST_RACES,
    );
    let stats = delete_targets(&table).expect("a record that lands late is a success");
    assert_eq!(
        faults.fired(),
        LOST_RACES,
        "the record rode out every injected lost race"
    );
    assert_true_counts(&stats);

    assert!(
        contains_all(&hidden_deleted_set(&table), &targets),
        "the retired ids must be in the hidden deleted set"
    );
    assert!(
        wal_docs(&faults).await.is_empty(),
        "a completed delete cleans its WAL up"
    );
    assert!(ids_matching(&table, titled(TARGET_TITLE)).is_empty());
    assert_eq!(ids_matching(&table, titled(KEPT_TITLE)).len(), KEPT_ROWS);
}

/// A delete whose deleted-set record never lands within its budget still
/// reports the rows it retired, not a conflict and not zero. The WAL is
/// left for recovery, unowned and with every target settled; the sweep
/// records the ids and completes it. A client that reissued the delete
/// anyway would match nothing — correctly, since the rows are gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_delete_whose_record_never_lands_reports_its_rows_and_recovery_records_them() {
    let (_dir, faults, table) = seeded_table();
    let targets = ids_matching(&table, titled(TARGET_TITLE));

    faults.fail_with(
        FaultKind::Precondition,
        FaultOp::PutIfMatch,
        HIDDEN_POINTER,
        UNLIMITED_LOST_RACES,
    );
    let stats = delete_targets(&table)
        .expect("the rows are retired; a deferred record must not fail the delete");
    assert_eq!(
        faults.fired(),
        RECORD_ATTEMPTS,
        "the record spends its own budget, four times the commit retry budget"
    );
    faults.clear();
    assert_true_counts(&stats);

    assert!(
        contains_none(&hidden_deleted_set(&table), &targets),
        "the record never landed"
    );
    assert_eq!(
        wal_docs(&faults).await,
        vec![(OpKind::Delete, WalState::Intent, false, true)],
        "the WAL is left short of Complete, unowned, with every target settled"
    );
    assert!(
        ids_matching(&table, titled(TARGET_TITLE)).is_empty(),
        "the rows are retired"
    );

    let report = scan_and_recover(
        &table,
        SupertableHandleId(SWEEP_OWNER),
        DEFAULT_LEASE_DURATION,
    )
    .await
    .expect("sweep");
    assert_eq!(report.n_tombstone_only_completed, 1);
    assert_eq!(report.total_targets_tombstoned, RETIRED_ROWS);
    assert!(
        contains_all(&hidden_deleted_set(&table), &targets),
        "recovery records the retired ids"
    );
    let docs = wal_docs(&faults).await;
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].1, WalState::Complete, "recovery completes the WAL");

    // A reissued delete finds nothing to delete: the first one happened.
    let mut writer = table.writer().expect("writer");
    let reissued = writer
        .delete(titled(TARGET_TITLE))
        .expect("buffer reissued delete");
    assert_eq!(
        reissued.matched, 0,
        "the first delete already retired the rows"
    );
    assert_eq!(ids_matching(&table, titled(KEPT_TITLE)).len(), KEPT_ROWS);
}

/// An update whose deleted-set record loses a few races replaces its rows
/// exactly once and reports the real counts.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_update_whose_record_loses_some_races_replaces_its_rows_once() {
    let (_dir, faults, table) = seeded_table();
    let targets = ids_matching(&table, titled(TARGET_TITLE));

    faults.fail_with(
        FaultKind::Precondition,
        FaultOp::PutIfMatch,
        HIDDEN_POINTER,
        LOST_RACES,
    );
    let stats = update_targets(&table).expect("a record that lands late is a success");
    assert_eq!(faults.fired(), LOST_RACES);
    assert_true_counts(&stats);

    assert!(contains_all(&hidden_deleted_set(&table), &targets));
    assert!(wal_docs(&faults).await.is_empty());
    assert!(ids_matching(&table, titled(TARGET_TITLE)).is_empty());
    assert_eq!(
        ids_matching(&table, titled(REPLACEMENT_TITLE)).len(),
        RETIRED_ROWS,
        "each row was replaced exactly once"
    );
    assert_eq!(ids_matching(&table, titled(KEPT_TITLE)).len(), KEPT_ROWS);
}

/// An update whose deleted-set record never lands reports its real counts;
/// recovery records the replaced ids without replacing anything again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_update_whose_record_never_lands_reports_its_rows_and_recovery_records_them() {
    let (_dir, faults, table) = seeded_table();
    let targets = ids_matching(&table, titled(TARGET_TITLE));

    faults.fail_with(
        FaultKind::Precondition,
        FaultOp::PutIfMatch,
        HIDDEN_POINTER,
        UNLIMITED_LOST_RACES,
    );
    let stats = update_targets(&table)
        .expect("the rows are replaced; a deferred record must not fail the update");
    assert_eq!(faults.fired(), RECORD_ATTEMPTS);
    faults.clear();
    assert_true_counts(&stats);

    assert!(contains_none(&hidden_deleted_set(&table), &targets));
    assert_eq!(
        wal_docs(&faults).await,
        vec![(OpKind::Update, WalState::Appended, false, true)],
        "the WAL is left at Appended, unowned, with every target settled"
    );
    assert_eq!(
        ids_matching(&table, titled(REPLACEMENT_TITLE)).len(),
        RETIRED_ROWS
    );

    let report = scan_and_recover(
        &table,
        SupertableHandleId(SWEEP_OWNER),
        DEFAULT_LEASE_DURATION,
    )
    .await
    .expect("sweep");
    assert_eq!(report.n_tombstone_only_completed, 1);
    assert!(contains_all(&hidden_deleted_set(&table), &targets));
    let docs = wal_docs(&faults).await;
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].1, WalState::Complete);

    assert!(ids_matching(&table, titled(TARGET_TITLE)).is_empty());
    assert_eq!(
        ids_matching(&table, titled(REPLACEMENT_TITLE)).len(),
        RETIRED_ROWS,
        "recovery replaced nothing a second time"
    );
    assert_eq!(ids_matching(&table, titled(KEPT_TITLE)).len(), KEPT_ROWS);
}

/// A genuine storage fault on the record is still a fault: the commit
/// fails, and not as a conflict, so nothing invites a reissue.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_storage_fault_on_the_record_surfaces_as_a_fault() {
    let (_dir, faults, table) = seeded_table();

    faults.fail_with(FaultKind::Transient, FaultOp::PutIfMatch, HIDDEN_POINTER, 1);
    let err = delete_targets(&table).expect_err("a broken store must surface");
    assert_eq!(faults.fired(), 1, "the fault came from the injection");
    assert!(
        matches!(err, CommitError::PartialCommit { .. }),
        "expected a partial commit, got {err:?}"
    );
    assert!(
        !err.is_conflict(),
        "a storage fault must not read as a retryable conflict: {err}"
    );
    assert!(
        err.to_string().contains("deleted-set record failed"),
        "the record is what failed: {err}"
    );
}

/// The handle's view of the hidden index falls behind whenever another
/// writer commits there — here a second handle on the same table. The
/// record must catch up and land rather than report a race it never ran.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_delete_lands_its_record_when_another_writer_moved_the_hidden_index() {
    let (_dir, faults, table) = seeded_table();
    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);
    let peer = Supertable::open(vector_options(&storage)).expect("open a second handle");
    let peer_hidden = Arc::clone(
        peer.inner()
            .vector_index_table
            .as_ref()
            .expect("peer's hidden index"),
    );
    let budget = RecordRetryBudget {
        max_attempts: RECORD_ATTEMPTS as u32,
        deadline: DEFAULT_LEASE_DURATION,
    };
    record_hidden_deleted_ids(peer_hidden.inner(), &[-1], budget)
        .await
        .expect("the peer's commit to the hidden index");

    let targets = ids_matching(&table, titled(TARGET_TITLE));
    let stats = delete_targets(&table).expect("a stale view of the hidden index is not a failure");
    assert_true_counts(&stats);
    let deleted = hidden_deleted_set(&table);
    assert!(contains_all(&deleted, &targets));
    assert!(
        contains_all(&deleted, &[-1]),
        "the peer's record survives ours"
    );
    assert!(wal_docs(&faults).await.is_empty());
}

/// Deletes under a second writer committing to the hidden index throughout:
/// every delete reports its rows, and every record lands — directly, or,
/// for one that lost every race in its budget, through recovery.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn deletes_under_a_concurrent_hidden_index_writer_report_their_rows() {
    let (_dir, faults, table) = seeded_table_with_retries(CONCURRENT_COMMIT_RETRIES);
    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);
    let peer = Supertable::open(vector_options_with_retries(
        &storage,
        CONCURRENT_COMMIT_RETRIES,
    ))
    .expect("open a second handle");
    let peer_hidden = Arc::clone(
        peer.inner()
            .vector_index_table
            .as_ref()
            .expect("peer's hidden index"),
    );

    let stop = Arc::new(AtomicBool::new(false));
    let peer_commits = Arc::new(AtomicUsize::new(0));
    let hammer = {
        let stop = Arc::clone(&stop);
        let peer_commits = Arc::clone(&peer_commits);
        thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("peer runtime");
            let budget = RecordRetryBudget {
                max_attempts: u32::MAX,
                deadline: DEFAULT_LEASE_DURATION,
            };
            let mut next_id: i128 = -1;
            while !stop.load(Ordering::SeqCst) {
                runtime
                    .block_on(record_hidden_deleted_ids(
                        peer_hidden.inner(),
                        &[next_id],
                        budget,
                    ))
                    .expect("the peer's commit to the hidden index");
                peer_commits.fetch_add(1, Ordering::SeqCst);
                next_id -= 1;
                thread::sleep(PEER_COMMIT_PAUSE);
            }
        })
    };

    let mut all_targets = Vec::new();
    for round in 0..CONCURRENT_ROUNDS {
        let title = format!("round-{round}");
        let mut writer = table.writer().expect("writer");
        writer
            .append(&batch(&[title.as_str(); RETIRED_ROWS], 100 + round as u64))
            .expect("append");
        writer.commit().expect("commit appends");
        drop(writer);
        let targets = ids_matching(&table, titled(&title));
        assert_eq!(targets.len(), RETIRED_ROWS);

        let stats = sole_outcome(&table, |writer| {
            writer.delete(titled(&title)).expect("buffer delete");
        })
        .unwrap_or_else(|e| panic!("round {round}: the delete must not fail: {e}"));
        assert_eq!(stats.matched, RETIRED_ROWS, "round {round}");
        assert_eq!(stats.n_tombstoned, RETIRED_ROWS, "round {round}");
        assert!(ids_matching(&table, titled(&title)).is_empty());
        all_targets.extend(targets);
    }
    stop.store(true, Ordering::SeqCst);
    hammer.join().expect("peer thread");
    assert!(
        peer_commits.load(Ordering::SeqCst) > 0,
        "the peer committed to the hidden index during the run"
    );

    // A delete whose record lost every race in its budget left its WAL,
    // unowned and settled, for recovery; nothing else may be left.
    let deferred = wal_docs(&faults).await;
    assert!(
        deferred
            .iter()
            .all(|doc| *doc == (OpKind::Delete, WalState::Intent, false, true)),
        "only deferred records may be left behind: {deferred:?}"
    );
    let report = scan_and_recover(
        &table,
        SupertableHandleId(SWEEP_OWNER),
        DEFAULT_LEASE_DURATION,
    )
    .await
    .expect("sweep");
    assert_eq!(report.n_tombstone_only_completed, deferred.len());

    all_targets.sort_unstable();
    assert!(
        contains_all(&hidden_deleted_set(&table), &all_targets),
        "every delete's record landed"
    );
    assert!(
        wal_docs(&faults)
            .await
            .iter()
            .all(|doc| doc.1 == WalState::Complete),
        "every delete's WAL is complete"
    );
}
