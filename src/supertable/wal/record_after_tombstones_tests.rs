// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A mutation that has retired its rows, under a concurrent writer.
//!
//! Once the tombstone phase has settled every target and landed its bits,
//! the rows are retired. The steps that finish the mutation follow: persist
//! the settled targets, stamp the tombstone seqs, record the rows in the
//! hidden vector index's deleted set, and flip the WAL to `Complete`. These
//! tests pin that the mutation reports its real outcome however those steps
//! fare — a record that loses races is retried, and any step that still
//! fails leaves the WAL for recovery — and that nothing it reports invites
//! reissuing a mutation that already happened. The handle that left a WAL
//! unfinished finishes it itself. They read WAL state documents and the
//! hidden manifest directly, so they live in-crate.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use arrow_array::{FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{Expr, col, lit};
use rayon::ThreadPoolBuilder;
use tempfile::TempDir;
use tokio::{runtime::Builder as RuntimeBuilder, time::sleep};

use crate::{
    storage::{LocalFsStorageProvider, StorageProvider},
    supertable::{
        BuildError, Supertable, SupertableOptions, hidden_deleted,
        mutations::{CommitError, MutationStats},
        wal::{
            WalStore,
            lease::DEFAULT_LEASE_DURATION,
            recovery::scan_and_recover,
            state_doc::{OpKind, SupertableHandleId, TombstoneOutcome, WalState},
        },
        writer::{CommitRetryBudget, SupertableWriter, record_hidden_deleted_ids},
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
/// Embedding seed for the rows every table is seeded with.
const SEEDED_ROWS_SEED: u64 = 1;
/// Embedding seed for an update's replacement rows.
const REPLACEMENT_SEED: u64 = 2;
/// Embedding seed of the concurrent-writer test's first round; each round
/// adds its index.
const ROUND_SEED_BASE: u64 = 100;
/// Embedding seed for the rows an append commits to finish a deferred WAL.
const FOLLOW_UP_SEED: u64 = 3;
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
/// Title of the rows an append commits to finish a deferred WAL.
const FOLLOW_UP_TITLE: &str = "follow-up";
/// The user table's commit retry budget. Kept small so a step that never
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
/// A URI fragment both tables' pointers contain. The tombstone-seq stamp
/// writes the user table's before the record writes the hidden one's.
const ANY_POINTER: &str = "_supertable/current";
/// A URI fragment every WAL state document contains.
const WAL_STATE_DOCS: &str = "wal/mutations/";
/// Owner id of the recovery sweep these tests run, distinct from the
/// writer's handle.
const SWEEP_OWNER: i128 = 0x5EED;
/// Id a second handle records in the hidden deleted set; no row has it.
const PEER_RECORDED_ID: i128 = -1;
/// Delete rounds in the concurrent-writer test.
const CONCURRENT_ROUNDS: usize = 4;
/// The table's commit retry budget in the concurrent-writer test: the
/// engine default, so the record rides out the race as it would in use.
const CONCURRENT_COMMIT_RETRIES: u32 = 10;
/// Pause between the concurrent writer's commits to the hidden index, so it
/// commits as often as a busy background writer rather than in a spin.
const PEER_COMMIT_PAUSE: Duration = Duration::from_millis(20);
/// Window a deadline-bound record is given in the deadline test.
const RECORD_WINDOW: Duration = Duration::from_millis(300);
/// Scheduling jitter the deadline test tolerates past the window: the record
/// estimates an attempt's length from the ones before it, and a loaded
/// runner can stretch the last one a little.
const DEADLINE_JITTER: Duration = Duration::from_millis(50);
/// How long the background-retry test waits for the handle to finish its
/// WAL: comfortably past the retry's first attempt.
const BACKGROUND_FINISH_WAIT: Duration = Duration::from_secs(30);
/// Poll interval while waiting on the background retry.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

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

fn vector_options(storage: &Arc<dyn StorageProvider>, commit_retries: u32) -> SupertableOptions {
    let pool = Arc::new(
        ThreadPoolBuilder::new()
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
/// titled [`TARGET_TITLE`] and [`KEPT_ROWS`] titled [`KEPT_TITLE`], with the
/// handle's background retry held so a WAL it leaves unfinished stays so
/// until the test finishes it.
fn seeded_table() -> (TempDir, Arc<FaultStorage>, Supertable) {
    let seeded = seeded_table_with_retries(COMMIT_RETRIES);
    seeded.2.inner().unfinished_wals.hold_background_retry();
    seeded
}

/// [`seeded_table`] with the given commit retry budget and the background
/// retry free to run.
fn seeded_table_with_retries(commit_retries: u32) -> (TempDir, Arc<FaultStorage>, Supertable) {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let faults = FaultStorage::wrap(local);
    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);
    let table = Supertable::create(vector_options(&storage, commit_retries)).expect("create");
    assert!(
        table.inner().vector_index_table.is_some(),
        "a vector table carries a hidden vector index"
    );
    let mut titles = vec![TARGET_TITLE; RETIRED_ROWS];
    titles.extend([KEPT_TITLE; KEPT_ROWS]);
    let mut writer = table.writer().expect("writer");
    writer
        .append(&batch(&titles, SEEDED_ROWS_SEED))
        .expect("append");
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

fn hidden_index(table: &Supertable) -> Arc<Supertable> {
    Arc::clone(
        table
            .inner()
            .vector_index_table
            .as_ref()
            .expect("hidden index"),
    )
}

/// The hidden index's deleted set as this handle sees it.
fn hidden_deleted_set(table: &Supertable) -> Vec<i128> {
    hidden_deleted::deleted_user_ids(&hidden_index(table).inner().manifest.load_full())
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
                batch(&[REPLACEMENT_TITLE; RETIRED_ROWS], REPLACEMENT_SEED),
            )
            .expect("buffer update");
        assert_eq!(pending.matched, RETIRED_ROWS);
    })
}

/// Commit an append through the same handle — the handle's next commit.
fn commit_follow_up_append(table: &Supertable) {
    let mut writer = table.writer().expect("writer");
    writer
        .append(&batch(&[FOLLOW_UP_TITLE], FOLLOW_UP_SEED))
        .expect("append");
    writer.commit().expect("commit the follow-up append");
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

/// The one WAL a deferred mutation of `op_kind` leaves: short of
/// `Complete` at `state`, unowned, every target settled — and remembered by
/// the handle as its own to finish.
async fn assert_left_unfinished(
    table: &Supertable,
    faults: &Arc<FaultStorage>,
    op_kind: OpKind,
    state: WalState,
) {
    assert_eq!(
        wal_docs(faults).await,
        vec![(op_kind, state, false, true)],
        "the WAL is left short of Complete, unowned, with every target settled"
    );
    assert_eq!(
        table.inner().unfinished_wals.pending().len(),
        1,
        "the handle remembers the WAL as its own to finish"
    );
}

/// Every WAL left on storage is `Complete`, and the handle has none left.
async fn assert_all_finished(table: &Supertable, faults: &Arc<FaultStorage>) {
    let docs = wal_docs(faults).await;
    assert!(
        docs.iter().all(|doc| doc.1 == WalState::Complete),
        "every WAL is complete: {docs:?}"
    );
    assert!(table.inner().unfinished_wals.pending().is_empty());
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
    assert!(table.inner().unfinished_wals.pending().is_empty());
    assert!(ids_matching(&table, titled(TARGET_TITLE)).is_empty());
    assert_eq!(ids_matching(&table, titled(KEPT_TITLE)).len(), KEPT_ROWS);
}

/// A delete whose deleted-set record never lands within its budget still
/// reports the rows it retired, not a conflict and not zero. The WAL is
/// left for recovery; the sweep records the ids and completes it. A client
/// that reissued the delete anyway would match nothing — correctly, since
/// the rows are gone.
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
    assert_left_unfinished(&table, &faults, OpKind::Delete, WalState::Intent).await;
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
    assert_left_unfinished(&table, &faults, OpKind::Update, WalState::Appended).await;
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

/// A storage fault on the record — not a lost race — after the rows are
/// retired defers too: the delete reports its rows, and the same handle's
/// next commit, with no reopen, records the ids and completes the WAL.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_storage_fault_on_the_record_defers_and_the_next_commit_finishes_it() {
    let (_dir, faults, table) = seeded_table();
    let targets = ids_matching(&table, titled(TARGET_TITLE));

    faults.fail_with(FaultKind::Transient, FaultOp::PutIfMatch, HIDDEN_POINTER, 1);
    let stats = delete_targets(&table).expect("the rows are retired; a record fault defers");
    assert_eq!(faults.fired(), 1, "the fault came from the injection");
    assert_true_counts(&stats);
    assert!(contains_none(&hidden_deleted_set(&table), &targets));
    assert_left_unfinished(&table, &faults, OpKind::Delete, WalState::Intent).await;

    commit_follow_up_append(&table);
    assert!(
        contains_all(&hidden_deleted_set(&table), &targets),
        "the handle's next commit records the retired ids"
    );
    assert_all_finished(&table, &faults).await;
}

/// Refused credentials on the record are the one finishing failure that
/// surfaces — but never as a conflict, and with a message that says the
/// rows are gone, so nothing reads it as a reason to reissue. The WAL is
/// left unfinished, and the handle finishes it once storage accepts it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_credentials_on_the_record_surface_without_inviting_a_reissue() {
    let (_dir, faults, table) = seeded_table();
    let targets = ids_matching(&table, titled(TARGET_TITLE));

    faults.fail_with(
        FaultKind::PermissionDenied,
        FaultOp::PutIfMatch,
        HIDDEN_POINTER,
        1,
    );
    let err = delete_targets(&table).expect_err("refused credentials must surface");
    assert_eq!(faults.fired(), 1, "the fault came from the injection");
    assert!(
        matches!(err, CommitError::PartialCommit { .. }),
        "expected a partial commit, got {err:?}"
    );
    assert!(err.is_permission_denied(), "reported as refused: {err}");
    assert!(!err.is_conflict(), "never a conflict: {err}");
    assert!(
        err.to_string().contains("must not be reissued"),
        "the message says the rows are retired: {err}"
    );
    assert!(
        ids_matching(&table, titled(TARGET_TITLE)).is_empty(),
        "the rows are retired all the same"
    );
    assert_left_unfinished(&table, &faults, OpKind::Delete, WalState::Intent).await;

    commit_follow_up_append(&table);
    assert!(contains_all(&hidden_deleted_set(&table), &targets));
    assert_all_finished(&table, &faults).await;
}

/// A tombstone-seq stamp that keeps losing its race runs after the rows are
/// retired, so it defers the same way rather than reading as a conflict.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stamp_that_never_lands_defers_instead_of_a_conflict() {
    let (_dir, faults, table) = seeded_table();
    let targets = ids_matching(&table, titled(TARGET_TITLE));

    // The stamp is the first pointer write after the targets settle; it
    // spends its own budget and the record never starts.
    faults.fail_with(
        FaultKind::Precondition,
        FaultOp::PutIfMatch,
        ANY_POINTER,
        UNLIMITED_LOST_RACES,
    );
    let stats = delete_targets(&table).expect("a stamp lost to contention defers");
    assert_eq!(
        faults.fired(),
        COMMIT_RETRIES as usize,
        "the stamp spent its budget"
    );
    faults.clear();
    assert_true_counts(&stats);
    assert_left_unfinished(&table, &faults, OpKind::Delete, WalState::Intent).await;

    commit_follow_up_append(&table);
    assert!(contains_all(&hidden_deleted_set(&table), &targets));
    assert!(ids_matching(&table, titled(TARGET_TITLE)).is_empty());
    assert_all_finished(&table, &faults).await;
}

/// The `Complete` write losing its race — a peer preempting a lapsed lease,
/// say — comes after the rows are retired and recorded, so it defers too.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lost_complete_write_defers_instead_of_a_conflict() {
    let (_dir, faults, table) = seeded_table();
    let targets = ids_matching(&table, titled(TARGET_TITLE));

    // Arm the WAL-doc fault as the record writes the hidden pointer, so it
    // hits the next state-doc write: the `Complete` one.
    let arming = Arc::clone(&faults);
    faults.before(FaultOp::PutIfMatch, HIDDEN_POINTER, move || {
        arming.fail_with(
            FaultKind::Precondition,
            FaultOp::PutIfMatch,
            WAL_STATE_DOCS,
            1,
        );
    });
    let stats = delete_targets(&table).expect("a lost Complete write defers");
    assert_eq!(faults.fired(), 1, "the Complete write lost its race");
    assert_true_counts(&stats);
    assert!(
        contains_all(&hidden_deleted_set(&table), &targets),
        "the record landed before the Complete write"
    );
    assert_left_unfinished(&table, &faults, OpKind::Delete, WalState::Intent).await;

    commit_follow_up_append(&table);
    assert_all_finished(&table, &faults).await;
}

/// A handle that runs no further commit still finishes the WAL it left:
/// its background retry does, with no reopen.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_background_retry_finishes_a_deferred_wal() {
    let (_dir, faults, table) = seeded_table_with_retries(COMMIT_RETRIES);
    let targets = ids_matching(&table, titled(TARGET_TITLE));

    faults.fail_with(FaultKind::Transient, FaultOp::PutIfMatch, HIDDEN_POINTER, 1);
    let stats = delete_targets(&table).expect("a record fault defers");
    assert_true_counts(&stats);

    let waited_from = Instant::now();
    // Settled, not merely an empty list: the retry takes the WAL off the
    // list before it records the deleted set.
    while !table.inner().unfinished_wals.settled() {
        assert!(
            waited_from.elapsed() < BACKGROUND_FINISH_WAIT,
            "the background retry never finished the WAL"
        );
        sleep(POLL_INTERVAL).await;
    }
    assert!(contains_all(&hidden_deleted_set(&table), &targets));
    assert_all_finished(&table, &faults).await;
}

/// The handle's view of the hidden index falls behind whenever another
/// writer commits there — here a second handle on the same table. Catching
/// up is a reload, not a lost race: it costs no attempt, so even a
/// one-attempt record lands, and so does the delete's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_record_catches_up_when_another_writer_moved_the_hidden_index() {
    let (_dir, faults, table) = seeded_table();
    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);
    let peer =
        Supertable::open(vector_options(&storage, COMMIT_RETRIES)).expect("open a second handle");
    let one_attempt = CommitRetryBudget {
        max_attempts: 1,
        deadline: None,
    };
    record_hidden_deleted_ids(
        hidden_index(&peer).inner(),
        &[PEER_RECORDED_ID],
        one_attempt,
    )
    .await
    .expect("the peer's commit to the hidden index");
    record_hidden_deleted_ids(
        hidden_index(&peer).inner(),
        &[PEER_RECORDED_ID - 1],
        one_attempt,
    )
    .await
    .expect("a second peer commit");

    // This handle's view is now behind by two commits.
    let ours = PEER_RECORDED_ID - 2;
    record_hidden_deleted_ids(hidden_index(&table).inner(), &[ours], one_attempt)
        .await
        .expect("a stale view reloads without spending the one attempt");

    let targets = ids_matching(&table, titled(TARGET_TITLE));
    let stats = delete_targets(&table).expect("a stale view of the hidden index is not a failure");
    assert_true_counts(&stats);
    let deleted = hidden_deleted_set(&table);
    assert!(contains_all(&deleted, &targets));
    assert!(
        contains_all(&deleted, &[PEER_RECORDED_ID, PEER_RECORDED_ID - 1, ours]),
        "every writer's record survives the others"
    );
    assert!(wal_docs(&faults).await.is_empty());
}

/// A record bounded by a deadline gives up before it, not one attempt past
/// it: it starts no attempt it could not expect to finish in time.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_record_starts_no_attempt_past_its_deadline() {
    let (_dir, faults, table) = seeded_table();
    faults.fail_with(
        FaultKind::Precondition,
        FaultOp::PutIfMatch,
        HIDDEN_POINTER,
        UNLIMITED_LOST_RACES,
    );
    let started = Instant::now();
    let deadline = started + RECORD_WINDOW;
    let budget = CommitRetryBudget {
        max_attempts: u32::MAX,
        deadline: Some(deadline),
    };
    let err = record_hidden_deleted_ids(hidden_index(&table).inner(), &[PEER_RECORDED_ID], budget)
        .await
        .expect_err("every attempt loses");
    assert!(
        matches!(err, BuildError::WriteContention),
        "running out of budget is typed contention, got {err:?}"
    );
    assert!(faults.fired() > 1, "it retried within the window");
    assert!(
        Instant::now() <= deadline + DEADLINE_JITTER,
        "it gave up inside its window, {:?} after starting",
        started.elapsed()
    );
}

/// Deletes under a second writer committing to the hidden index throughout:
/// every delete reports its rows, and every record lands — directly, or,
/// for one that lost every race in its budget, through recovery.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn deletes_under_a_concurrent_hidden_index_writer_report_their_rows() {
    let (_dir, faults, table) = seeded_table_with_retries(CONCURRENT_COMMIT_RETRIES);
    table.inner().unfinished_wals.hold_background_retry();
    let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);
    let peer = Supertable::open(vector_options(&storage, CONCURRENT_COMMIT_RETRIES))
        .expect("open a second handle");
    let peer_hidden = hidden_index(&peer);

    let stop = Arc::new(AtomicBool::new(false));
    let peer_commits = Arc::new(AtomicUsize::new(0));
    let hammer = {
        let stop = Arc::clone(&stop);
        let peer_commits = Arc::clone(&peer_commits);
        thread::spawn(move || {
            let runtime = RuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("peer runtime");
            let budget = CommitRetryBudget {
                max_attempts: u32::MAX,
                deadline: None,
            };
            let mut next_id = PEER_RECORDED_ID;
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
            .append(&batch(
                &[title.as_str(); RETIRED_ROWS],
                ROUND_SEED_BASE + round as u64,
            ))
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

    // A delete whose finish lost every race in its budget left its WAL,
    // unowned and settled, for recovery; nothing else may be left.
    let deferred = wal_docs(&faults).await;
    assert!(
        deferred
            .iter()
            .all(|doc| *doc == (OpKind::Delete, WalState::Intent, false, true)),
        "only deferred WALs may be left behind: {deferred:?}"
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
