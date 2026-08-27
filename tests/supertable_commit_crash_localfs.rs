// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Crash safety for the persisted supertable commit path on
//! LocalFS.
//!
//! Parent spawns a child copy of this test binary with an
//! env var pointing at a temp directory + the kill point to
//! hit. The child drives one or two commits through a `CrashStorage`
//! wrapper that calls `std::process::abort()` immediately
//! after the underlying PUT lands (raising SIGABRT, which
//! drops the process without running any Drop impls — the
//! semantic equivalent of `kill -9` for our durability
//! claim).
//!
//! The parent then `Supertable::open`s the temp directory
//! and asserts the recovered state is one of two coherent
//! outcomes:
//!
//!   - The pointer file is missing or still references the
//!     prior committed `manifest_id` → open returns the prior
//!     state (or `PointerUnreadable` on a fresh supertable).
//!     Any superfile / manifest-part / manifest-list bytes
//!     written before the crash but never referenced by a
//!     committed pointer are **orphans**: tolerated by
//!     readers and GC'd by compaction.
//!   - The pointer file has been atomically replaced with
//!     the new version → open returns the new state. The
//!     crash happened AFTER the visibility barrier; the
//!     commit is durable.
//!
//! This is the load-bearing property of the
//! atomic-rename pointer commit: the pointer is the *only*
//! object that ever gets renamed, so the question "did the
//! commit succeed?" reduces to "did the pointer's rename
//! complete?" — a single atomic operation on LocalFS.
//!
//! `Supertable::create` publishes an initial empty manifest
//! (id 0) before any commit, so a freshly created table is
//! already durably openable. The kill points below account for
//! create's initial list + pointer PUTs (see `kill_point_config`).
//!
//! Kill points exercised (one test function each):
//!
//! | Test fn                                                      | Crash point                                | Expected post-crash open state                    |
//! |--------------------------------------------------------------|---------------------------------------------|----------------------------------------------------|
//! | `crash_post_superfile_no_prior_commit_yields_empty_table`      | After 1st commit's superfile PUT, before its list/pointer | `manifest_id == 0` (create's empty manifest), orphan superfile |
//! | `crash_post_list_no_prior_commit_yields_pointer_unreadable`    | After create's list PUT, before its pointer | `OpenError::PointerUnreadable`                     |
//! | `crash_post_superfile_on_second_commit_yields_v1`                | First commit succeeds; 2nd commit's superfile PUT triggers | `manifest_id == 1` (v_prev), orphan v2 superfile    |
//! | `crash_post_list_on_second_commit_yields_v1`                   | First commit succeeds; 2nd commit's list PUT triggers   | `manifest_id == 1`, orphan v2 list + part         |
//! | `crash_post_list_on_second_commit_recovers_next_commit`        | Same crash point as above                    | v1 recovered; the FIRST post-crash commit publishes at id 3, skipping the orphaned id 2 |
//! | `crash_post_pointer_on_second_commit_yields_v2`                | First commit succeeds; 2nd commit's pointer PUT triggers AFTER it lands | `manifest_id == 2` (commit was durable)           |
//! | `crash_post_hidden_child_superfile_yields_pre_split_index`     | Batched cell split: first child superfile PUT (pre-commit) | Hidden index at drained state; orphan children GC'd |
//! | `crash_post_hidden_list_yields_pre_split_index`                | Batched cell split: hidden list PUT, before its pointer | Hidden index at drained state; orphans GC'd |
//! | `crash_post_hidden_pointer_yields_split_index`                 | Batched cell split: hidden pointer CAS AFTER it lands | Split durable + immediately queryable (regression tripwire for the pre-#498 window); post-crash `optimize` + `gc` run clean |
//! | `crash_post_hidden_repack_shard_yields_pre_split_index`        | Bulk repack: first packed-shard PUT (pre-pin, pre-commit) | Hidden index at drained state; orphan shard GC'd |
//! | `crash_between_split_batches_yields_first_batch`               | Two batches: crash in batch 2's pin stamp, after batch 1's commit | Batch 1 durable + queryable; batch 2 orphans GC'd |
//!
//! LocalFS-only. The atomic-rename semantics hinge on local
//! filesystem behavior; RustFS's crash story is its own
//! concern.

#![deny(clippy::unwrap_used)]

use std::{
    env,
    ops::Range,
    path::PathBuf,
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use arrow_array::{Array, Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use bytes::Bytes;
use infino::{
    VectorSearchOptions,
    config::OptimizeOptions,
    superfile::{
        builder::{FtsConfig, VectorConfig},
        vector::{distance::Metric, rerank_codec::RerankCodec},
    },
    supertable::{
        OpenError, Supertable, SupertableOptions,
        storage::{LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options, default_tokenizer},
};

const ENV_DIR: &str = "INFINO_M12_CRASH_DIR";
const ENV_KILL_POINT: &str = "INFINO_M12_CRASH_KILL_POINT";

/// One named kill point. The child reads the env var and
/// configures the `CrashStorage` to match.
const KP_SEG_FIRST: &str = "seg-1";
const KP_LIST_FIRST: &str = "list-1";
const KP_SEG_SECOND: &str = "seg-2";
const KP_LIST_SECOND: &str = "list-2";
const KP_POINTER_SECOND: &str = "pointer-2";

/// Hidden vector-index kill points: crash inside the batched cell-split's
/// window (child superfile upload → hidden list PUT → hidden pointer CAS).
/// The hidden table's storage prefix is `_infino_<uuid>_vector_index/` with
/// a per-table random uuid, so these match by CONTAINED token (the same
/// `_vector_index` token `runtime_metrics::io` classifies hidden URIs by)
/// rather than a static prefix, and the child ARMS the counter only once
/// the split starts — create/commit/drain traffic through the same wrapper
/// is not counted, keeping the nth-match config independent of how many
/// PUTs the drain issues.
const KP_HIDDEN_SPLIT_SEG: &str = "hidden-split-seg";
const KP_HIDDEN_SPLIT_LIST: &str = "hidden-split-list";
const KP_HIDDEN_SPLIT_POINTER: &str = "hidden-split-pointer";
/// Bulk-repack variant: crash on the repack's first packed-shard PUT,
/// before its slow-CAS pin and commit.
const KP_HIDDEN_REPACK_SEG: &str = "hidden-repack-seg";
/// Multi-batch variant: two sequential single-cell batches; crash inside
/// batch 2's window, after batch 1's commit is durable.
const KP_HIDDEN_SPLIT_SECOND_LIST: &str = "hidden-split-second-list";

/// Vector-commit kill point: crash on a USER-table superfile PUT issued by
/// the pipelined publish, i.e. while the remaining shards of the same
/// commit are still packing. The text kill points above cannot reach this
/// window — a table with no vector columns never enters the drain-commit
/// path — so this is the one that covers uploads that start before the
/// batch is complete.
const KP_VECTOR_COMMIT_SEG: &str = "vector-commit-seg";
/// Writer threads (and so packed shards) for the pipelined-commit child.
/// Two is the smallest count that puts one shard on the wire while another
/// is still being packed, which is the property under test.
const VECTOR_COMMIT_SHARDS: usize = 2;

/// Exit code used when the crash child finishes WITHOUT aborting —
/// signals a misconfigured kill point (distinct from a clean exit).
const MISCONFIGURED_KILL_POINT_EXIT_CODE: i32 = 2;

/// Storage wrapper that aborts the process after the N-th
/// PUT whose URI matches the trigger returns success (prefix
/// match for the user-table kill points; contained-token match
/// for the hidden-index ones, whose per-table uuid prefix cannot
/// be known statically). Everything else is forwarded verbatim
/// to the inner `LocalFsStorageProvider`. Matches count only
/// while `armed` — the hidden kill points arm right before the
/// split so earlier create/commit/drain PUTs don't shift the
/// nth-match configuration.
#[derive(Debug)]
struct CrashStorage {
    inner: LocalFsStorageProvider,
    trigger_path_prefix: String,
    trigger_is_contains: bool,
    trigger_after_nth_match: usize,
    matches_seen: AtomicUsize,
    armed: AtomicBool,
    abort_label: String,
}

impl CrashStorage {
    fn new(
        inner: LocalFsStorageProvider,
        trigger_path_prefix: impl Into<String>,
        trigger_after_nth_match: usize,
        abort_label: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            trigger_path_prefix: trigger_path_prefix.into(),
            trigger_is_contains: false,
            trigger_after_nth_match,
            matches_seen: AtomicUsize::new(0),
            armed: AtomicBool::new(true),
            abort_label: abort_label.into(),
        }
    }

    /// Contained-token matcher, starting DISARMED; the child arms it at the
    /// step it wants counted (`arm`).
    fn new_contains_disarmed(
        inner: LocalFsStorageProvider,
        trigger_token: impl Into<String>,
        trigger_after_nth_match: usize,
        abort_label: impl Into<String>,
    ) -> Self {
        Self {
            trigger_is_contains: true,
            armed: AtomicBool::new(false),
            ..Self::new(inner, trigger_token, trigger_after_nth_match, abort_label)
        }
    }

    fn arm(&self) {
        self.armed.store(true, Ordering::SeqCst);
    }

    fn uri_matches(&self, uri: &str) -> bool {
        if self.trigger_is_contains {
            uri.contains(&self.trigger_path_prefix)
        } else {
            uri.starts_with(&self.trigger_path_prefix)
        }
    }

    /// Called from put_atomic / put_if_match after the
    /// inner provider returns, with the arm state CAPTURED AT REQUEST
    /// START — a request already in flight when `arm()` fires must not
    /// count toward the post-arm matches. Aborts the process iff that
    /// snapshot was armed AND `is_match` AND `ok` AND this is the Nth
    /// such match.
    fn maybe_abort(&self, uri: &str, armed_at_start: bool, is_match: bool, ok: bool) {
        if !(armed_at_start && is_match && ok) {
            return;
        }
        let n = self.matches_seen.fetch_add(1, Ordering::SeqCst) + 1;
        if n == self.trigger_after_nth_match {
            eprintln!(
                "CRASH-CHILD: aborting ({label}) after PUT uri={uri} match#={n}",
                label = self.abort_label
            );
            std::process::abort();
        }
    }
}

#[async_trait]
impl StorageProvider for CrashStorage {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.inner.head(uri).await
    }
    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        self.inner.get(uri).await
    }
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        self.inner.get_range(uri, range).await
    }
    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        let armed_at_start = self.armed.load(Ordering::SeqCst);
        let is_match = self.uri_matches(uri);
        let result = self.inner.put_atomic(uri, bytes).await;
        self.maybe_abort(uri, armed_at_start, is_match, result.is_ok());
        result
    }
    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        let armed_at_start = self.armed.load(Ordering::SeqCst);
        let is_match = self.uri_matches(uri);
        let result = self.inner.put_if_match(uri, bytes, expected_etag).await;
        self.maybe_abort(uri, armed_at_start, is_match, result.is_ok());
        result
    }
    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        self.inner.put_multipart(uri).await
    }
    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        self.inner.delete(uri).await
    }
}

/// Translate a kill point name into (trigger_path_prefix,
/// trigger_after_nth_match, n_commits). The child uses this
/// to configure `CrashStorage` and decide how many successful
/// commits to land before the crashing one.
fn kill_point_config(kp: &str) -> (&'static str, usize, usize) {
    // `Supertable::create` publishes an initial empty manifest before any
    // commit: one PUT to `manifest/` (the id-0 list) and one to
    // `_supertable/current` (the id-0 pointer), and zero to `data/`. The
    // nth-match counts below account for that initial write:
    //   - `KP_LIST_FIRST` (nth=1) now fires on create's own list write —
    //     still "no pointer written yet", so still yields PointerUnreadable.
    //   - the second-commit list/pointer kill points are +1 (nth=3): create's
    //     initial write occupies the first match, the first commit the second,
    //     the crashing second commit the third.
    //   - `data/` counts are unaffected (create writes no superfile).
    match kp {
        KP_SEG_FIRST => ("data/", 1, 1),
        KP_LIST_FIRST => ("manifest/", 1, 1),
        KP_SEG_SECOND => ("data/", 2, 2),
        KP_LIST_SECOND => ("manifest/", 3, 2),
        KP_POINTER_SECOND => ("_supertable/current", 3, 2),
        other => panic!("unknown kill point {other}"),
    }
}

/// Vector dimension for the hidden-split crash fixture.
const CRASH_EMB_DIM: usize = 16;
/// Rows per planted direction; two directions → two populated hidden cells,
/// and the busiest cell has enough rows to split k-ways.
const CRASH_ROWS_PER_DIRECTION: usize = 8;
/// Rotation seed for the crash fixture's vector column.
const CRASH_ROT_SEED: u64 = 7;
/// Probe width covering every cell of the tiny fixture grid, pre- and
/// post-split.
const CRASH_NPROBE: usize = 64;
/// Marker file (in the crash dir, outside every swept prefix) through which
/// the child reports the drained hidden manifest id to the parent.
const DRAINED_HIDDEN_ID_MARKER: &str = "drained-hidden-manifest-id";

/// Options + one committed batch for the hidden-split crash tests: `title`
/// FTS plus a 16-dim `emb` vector column with the Sq8Residual rerank codec
/// the split path requires (`test_helpers::default_vector_config` is Fp32 —
/// unusable here). The 1-thread writer pool keeps the drain to one packed
/// shard so the hidden PUT sequence is deterministic. Rows: 8 at `e_0` and
/// 8 at `e_1` → two populated hidden cells after the drain.
fn vector_crash_fixture() -> (SupertableOptions, arrow_array::RecordBatch) {
    vector_crash_fixture_with_writers(1)
}

/// [`vector_crash_fixture`] with an explicit writer-pool size. The pool
/// size is the commit's packed-shard count, so this is how a test asks for
/// more than one shard per commit.
fn vector_crash_fixture_with_writers(
    writers: usize,
) -> (SupertableOptions, arrow_array::RecordBatch) {
    let dim = CRASH_EMB_DIM;
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(item_field.clone(), dim as i32),
            false,
        ),
    ]));
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(writers)
            .build()
            .expect("rayon pool"),
    );
    let options = SupertableOptions::new(
        schema.clone(),
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![VectorConfig {
            column: "emb".into(),
            dim,
            rot_seed: CRASH_ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8Residual,
            provided_centroids: None,
        }],
        Some(default_tokenizer()),
    )
    .expect("valid options")
    .with_writer_pool(pool);

    let n = CRASH_ROWS_PER_DIRECTION * 2;
    let titles = LargeStringArray::from((0..n).map(|i| format!("doc-{i}")).collect::<Vec<_>>());
    let mut flat = vec![0.0f32; n * dim];
    for r in 0..n {
        flat[r * dim + usize::from(r >= n / 2)] = 1.0;
    }
    let fsl = FixedSizeListArray::new(
        item_field,
        dim as i32,
        Arc::new(Float32Array::from(flat)),
        None,
    );
    let batch = arrow_array::RecordBatch::try_new(
        schema,
        vec![
            Arc::new(titles) as Arc<dyn Array>,
            Arc::new(fsl) as Arc<dyn Array>,
        ],
    )
    .expect("batch");
    (options, batch)
}

/// Translate a hidden kill point into (contained URI token, nth armed
/// match). All fire on the FIRST armed match of their step — the child arms
/// the counter right before the split, so create/commit/drain PUTs through
/// the same wrapper don't shift the count.
fn hidden_kill_point_config(kp: &str) -> (&'static str, usize) {
    match kp {
        KP_HIDDEN_SPLIT_SEG | KP_HIDDEN_REPACK_SEG => ("_vector_index/data/", 1),
        // nth 1 on list/pointer is the batch's upload-PIN stamp; the
        // membership commit's list/pointer are nth 2.
        KP_HIDDEN_SPLIT_LIST => ("_vector_index/manifest/", 1),
        KP_HIDDEN_SPLIT_POINTER => ("_vector_index/_supertable/current", 2),
        // Second batch's pin-list PUT: batch 1 pin list (1) + batch 1
        // commit list (2) + batch 2 pin list (3) — a crash BETWEEN batch
        // commits, after batch 1 is durable.
        KP_HIDDEN_SPLIT_SECOND_LIST => ("_vector_index/manifest/", 3),
        other => panic!("unknown hidden kill point {other}"),
    }
}

/// Child path for the hidden-split kill points: create a vector table,
/// commit one batch, drain it into the hidden per-cell index, ARM the crash
/// storage, then split the busiest hidden cell. The batched split's PUT
/// sequence (child superfiles → hidden list → hidden pointer CAS) crosses
/// the armed kill point and aborts; reaching the end means the kill point
/// never fired.
fn run_vector_crash_child(dir: PathBuf, kill_point: &str) -> ! {
    let (token, nth) = hidden_kill_point_config(kill_point);

    let local = LocalFsStorageProvider::new(&dir).expect("local fs provider");
    let wrapped = Arc::new(CrashStorage::new_contains_disarmed(
        local, token, nth, kill_point,
    ));
    let storage: Arc<dyn StorageProvider> = Arc::clone(&wrapped) as Arc<dyn StorageProvider>;

    let (options, batch) = vector_crash_fixture();
    let st = Supertable::create(options.with_storage(storage)).expect("create");
    let mut w = st.writer().expect("writer");
    w.append(&batch).expect("append");
    w.commit().expect("commit");
    st.drain_vectors_to_cells_sync().expect("drain to cells");

    // Record the drained hidden generation for the parent's assertions —
    // stamp/checkpoint publishes don't bump ids deterministically enough to
    // hardcode, but "pre-split vs pre-split + 1" is exact.
    let drained_id = st.vector_index_table().expect("hidden index").manifest_id();
    std::fs::write(dir.join(DRAINED_HIDDEN_ID_MARKER), drained_id.to_string())
        .expect("write drained-id marker");

    wrapped.arm();
    let split = if kill_point == KP_HIDDEN_REPACK_SEG {
        st.repack_all_hidden_cells_sync().expect("repack") > 0
    } else if kill_point == KP_HIDDEN_SPLIT_SECOND_LIST {
        // Two sequential single-cell batches: the kill point sits inside
        // batch 2's window, so batch 1 must land durably first.
        let first = st.split_busiest_hidden_cell_sync().expect("first split");
        let second = st.split_busiest_hidden_cell_sync().expect("second split");
        first && second
    } else {
        st.split_busiest_hidden_cell_sync().expect("split")
    };

    eprintln!(
        "CRASH-CHILD: completed split (committed={split}) without aborting \
         (kill_point={kill_point}) — test configuration is wrong"
    );
    std::process::exit(MISCONFIGURED_KILL_POINT_EXIT_CODE);
}

/// Child path for [`KP_VECTOR_COMMIT_SEG`]: create a vector table and
/// commit one batch wide enough to pack [`VECTOR_COMMIT_SHARDS`] shards.
/// The commit takes the pipelined publish, so its first user superfile PUT
/// lands while the other shard is still packing; the wrapper aborts there,
/// mid-commit and before the manifest list or pointer is written.
fn run_vector_commit_crash_child(dir: PathBuf) -> ! {
    let local = LocalFsStorageProvider::new(&dir).expect("local fs provider");
    // `data/` is the user table's superfile prefix; the hidden index writes
    // under `_infino_<uuid>_vector_index/data/`, which this prefix match
    // does not accept — so the first match is the commit's own shard PUT.
    let wrapped = Arc::new(CrashStorage::new(local, "data/", 1, KP_VECTOR_COMMIT_SEG));
    let storage: Arc<dyn StorageProvider> = wrapped;

    let (options, batch) = vector_crash_fixture_with_writers(VECTOR_COMMIT_SHARDS);
    let st = Supertable::create(options.with_storage(storage)).expect("create");
    let mut w = st.writer().expect("writer");
    w.append(&batch).expect("append");
    w.commit().expect("commit");

    eprintln!(
        "CRASH-CHILD: completed the vector commit without aborting \
         (kill_point={KP_VECTOR_COMMIT_SEG}) — test configuration is wrong"
    );
    std::process::exit(MISCONFIGURED_KILL_POINT_EXIT_CODE);
}

/// Child path: build a Supertable on `CrashStorage` and run
/// up to `n_commits` commits. The wrapper triggers
/// `std::process::abort()` mid-flight in the last commit
/// once the Nth matching PUT lands. The function never
/// returns normally — either it aborts (expected) or, if
/// the test configuration is wrong (Nth match doesn't fire),
/// the commit completes and the function exits cleanly.
/// The parent treats either as failure of expectations.
fn run_crash_child(dir: PathBuf, kill_point: &str) -> ! {
    let (prefix, nth, n_commits) = kill_point_config(kill_point);

    let local = LocalFsStorageProvider::new(&dir).expect("local fs provider");
    let wrapped = Arc::new(CrashStorage::new(local, prefix, nth, kill_point));
    let storage: Arc<dyn StorageProvider> = wrapped;

    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    for c in 1..=n_commits {
        let mut w = st.writer().expect("writer");
        let titles = match c {
            1 => vec!["first commit alpha"],
            2 => vec!["second commit beta"],
            _ => vec!["nth commit gamma"],
        };
        let batch = build_title_batch(&titles);
        w.append(&batch).expect("append");
        // commit may abort mid-flight; if it returns
        // we either misconfigured the kill point or
        // we're on a successful commit before the
        // crashing one.
        w.commit().expect("commit");
    }

    // If we reach here, the crash never fired. Print + exit
    // with a recognizable non-zero code so the parent can
    // distinguish "no crash fired" from "child aborted as
    // expected".
    eprintln!(
        "CRASH-CHILD: completed {n_commits} commits without aborting (kill_point={kill_point}) — \
         test configuration is wrong"
    );
    std::process::exit(MISCONFIGURED_KILL_POINT_EXIT_CODE);
}

/// Spawn a child copy of this test binary, filtered to a
/// single named test, with the kill-point env var set.
fn spawn_crash_child(test_name: &str, kill_point: &str) -> PathBuf {
    let tmp = tempfile::tempdir().expect("tempdir");
    // `into_path` lets the parent inspect the directory after
    // the child aborts (otherwise the TempDir guard would drop
    // it before our verification runs). It leaks the dir, but
    // that's fine for a single test invocation.
    let dir = tmp.keep();

    let exe = env::current_exe().expect("current_exe");
    let status = Command::new(&exe)
        .args(["--exact", "--test-threads=1", test_name])
        .env(ENV_DIR, &dir)
        .env(ENV_KILL_POINT, kill_point)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("spawn child");

    assert!(
        !status.success(),
        "child should have aborted (SIGABRT); got clean exit {status:?}"
    );

    dir
}

/// Parent-side dispatch: if the env var is set, become the
/// child. Otherwise return so the caller runs as parent.
fn dispatch_child_if_set() -> Option<()> {
    if let Ok(dir) = env::var(ENV_DIR) {
        let kp = env::var(ENV_KILL_POINT).expect("ENV_KILL_POINT must be set with ENV_DIR");
        if kp.starts_with("hidden-") {
            run_vector_crash_child(PathBuf::from(dir), &kp);
        }
        if kp == KP_VECTOR_COMMIT_SEG {
            run_vector_commit_crash_child(PathBuf::from(dir));
        }
        run_crash_child(PathBuf::from(dir), &kp);
    }
    None
}

/// Parent-side verification shared by the hidden-split kill points: reopen
/// the user table with a plain provider (the hidden table reopens
/// automatically off the manifest's `vector_index_storage_prefix`), assert
/// the hidden manifest generation, assert no docs were lost through the
/// crash (both planted directions fully retrievable), then run an explicit
/// hidden-table `gc` (the background sweep never fires inside a test's
/// lifetime) and return its deleted-object count.
fn verify_hidden_split_crash(dir: &PathBuf, expected_id_delta: u64) -> u64 {
    let drained_id: u64 = std::fs::read_to_string(dir.join(DRAINED_HIDDEN_ID_MARKER))
        .expect("child wrote the drained-id marker before arming")
        .trim()
        .parse()
        .expect("marker holds a manifest id");
    // Per-kill-point id delta: the pre-upload pin stamp advances the id by
    // one BEFORE any byte moves, the membership commit by one more. A kill
    // inside the pin stamp (pre-pointer) recovers at +0; after the pin's
    // pointer at +1; after the commit's pointer at +2.
    let expect_hidden_manifest_id = drained_id + expected_id_delta;
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir).expect("provider"));
    let (options, _) = vector_crash_fixture();
    let recovered =
        Supertable::open(options.with_storage(storage)).expect("open recovers the user table");
    let hidden = recovered
        .vector_index_table()
        .expect("vector table reopens its hidden index")
        .clone();
    assert_eq!(
        hidden.manifest_id(),
        expect_hidden_manifest_id,
        "hidden index generation after the crash (drained at {drained_id})"
    );

    // Sweep FIRST — production's recovery order. A crash between a list
    // PUT and its pointer CAS (e.g. inside a pin stamp) leaves an orphaned
    // next-id manifest list, and the NEXT commit at that id fails with
    // write contention until the orphan is reclaimed — a pre-existing
    // property of the one-writer-per-manifest-id list PUT, surfaced by the
    // between-batches kill point. Children pinned by a pre-upload pin
    // survive this sweep by design.
    let first_sweep = hidden.gc(Duration::ZERO).expect("hidden gc");

    // Doc conservation FIRST — before any optimize that could repair the
    // state under test: one exhaustive-width query per planted direction
    // must retrieve its full half, whichever side of the crash the hidden
    // generation landed on. For the durable-split generations this is the
    // regression tripwire for the pre-#498 window (a split commit whose
    // slow-state restamp is dropped under-serves exactly here).
    let reader = recovered.reader().expect("reader");
    let mut seen_ids: Vec<i128> = Vec::new();
    for direction in 0..2usize {
        let mut query = vec![0.0f32; CRASH_EMB_DIM];
        query[direction] = 1.0;
        let batches = reader
            .vector_search(
                "emb",
                &query,
                CRASH_ROWS_PER_DIRECTION,
                VectorSearchOptions::new().with_nprobe(CRASH_NPROBE),
                None,
                None,
            )
            .expect("vector search");
        let mut direction_ids: Vec<i128> = Vec::new();
        for batch in &batches {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("_id column");
            for i in 0..ids.len() {
                direction_ids.push(ids.value(i));
            }
        }
        direction_ids.sort_unstable();
        direction_ids.dedup();
        assert_eq!(
            direction_ids.len(),
            CRASH_ROWS_PER_DIRECTION,
            "direction {direction} must retrieve its full planted half after the crash"
        );
        seen_ids.extend(direction_ids);
    }
    seen_ids.sort_unstable();
    seen_ids.dedup();
    assert_eq!(
        seen_ids.len(),
        CRASH_ROWS_PER_DIRECTION * 2,
        "every planted doc is retrievable after the crash"
    );

    // The next maintenance cycle must complete cleanly on EVERY recovered
    // state. Its pass-final slow-state stamp also releases any stale
    // upload pin the crash left behind (abandon-based recovery), so the
    // second sweep below reclaims what the pin was protecting.
    recovered
        .optimize(&OptimizeOptions::default())
        .expect("post-crash optimize completes the interrupted maintenance");
    let second_sweep = hidden.gc(Duration::ZERO).expect("hidden gc after optimize");

    first_sweep.objects_deleted + second_sweep.objects_deleted
}

#[test]
fn crash_mid_pipelined_vector_commit_yields_empty_table() {
    if dispatch_child_if_set().is_some() {
        return; // unreachable; child never returns
    }
    let dir = spawn_crash_child(
        "crash_mid_pipelined_vector_commit_yields_empty_table",
        KP_VECTOR_COMMIT_SEG,
    );

    // The abort fired on a shard PUT the pipelined publish issued while the
    // commit's other shard was still packing — the window this path opens
    // that the batch-wave publish did not. The durability boundary must be
    // where it always was: no manifest list, no pointer, so recovery lands
    // on create's empty id-0 manifest and every uploaded shard is an orphan.
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
    let (options, _) = vector_crash_fixture_with_writers(VECTOR_COMMIT_SHARDS);
    let recovered =
        Supertable::open(options.with_storage(storage)).expect("open recovers the id-0 manifest");
    assert_eq!(
        recovered.manifest_id(),
        0,
        "the crashing commit never stamped a manifest → recover create's empty id-0"
    );
    let reader = recovered.reader().expect("reader");
    assert_eq!(
        reader.n_superfiles(),
        0,
        "shards uploaded before the crash are orphans, invisible without a committed list"
    );

    // A query on the recovered table must answer from the empty manifest
    // rather than fault on the orphaned bytes.
    let mut query = vec![0.0f32; CRASH_EMB_DIM];
    query[0] = 1.0;
    let hits = reader
        .vector_search(
            "emb",
            &query,
            CRASH_ROWS_PER_DIRECTION,
            VectorSearchOptions::new().with_nprobe(CRASH_NPROBE),
            None,
            None,
        )
        .expect("vector search on the recovered table");
    assert!(
        hits.iter().all(|batch| batch.num_rows() == 0),
        "an uncommitted shard must not be readable after recovery"
    );

    let n_orphans = std::fs::read_dir(dir.join("data"))
        .map(|rd| rd.count())
        .unwrap_or(0);
    assert!(
        n_orphans >= 1,
        "the pipelined PUT that preceded the abort must be on disk as an orphan; found {n_orphans}"
    );
}

#[test]
fn crash_post_superfile_no_prior_commit_yields_empty_table() {
    if dispatch_child_if_set().is_some() {
        return; // unreachable; child never returns
    }
    let dir = spawn_crash_child(
        "crash_post_superfile_no_prior_commit_yields_empty_table",
        KP_SEG_FIRST,
    );

    // Parent verifies. The crash fired after the first commit's superfile PUT,
    // before that commit's manifest list/pointer. `create` already published
    // the initial empty manifest (id 0), so open recovers that durable empty
    // state rather than failing — the uncommitted superfile is an orphan the
    // reader ignores.
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
    let recovered = Supertable::open(default_supertable_options().with_storage(storage))
        .expect("open recovers create's empty id-0 manifest");
    assert_eq!(
        recovered.manifest_id(),
        0,
        "no commit landed → recover create's empty id-0 manifest"
    );
    assert_eq!(
        recovered.reader().expect("reader").n_superfiles(),
        0,
        "the orphan superfile is invisible without a committed manifest list"
    );

    // The orphan superfile file is present and ignored — the
    // superfile is just bytes under data/; readers don't
    // discover it without a committed manifest list.
    let data_dir = dir.join("data");
    let n_orphans = std::fs::read_dir(&data_dir)
        .map(|rd| rd.count())
        .unwrap_or(0);
    assert!(
        n_orphans >= 1,
        "orphan superfile must be present on disk; found {n_orphans} in {data_dir:?}"
    );
}

#[test]
fn crash_post_list_no_prior_commit_yields_pointer_unreadable() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let dir = spawn_crash_child(
        "crash_post_list_no_prior_commit_yields_pointer_unreadable",
        KP_LIST_FIRST,
    );

    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
    let err = Supertable::open(default_supertable_options().with_storage(storage))
        .expect_err("must reject post-crash state with no pointer");
    assert!(
        matches!(err, OpenError::ManifestLoadError(_)),
        "expected PointerUnreadable, got {err:?}"
    );

    // The orphan manifest is on disk but unreferenced.
    let manifest_dir = dir.join("manifest");
    let n_orphan_manifests = std::fs::read_dir(&manifest_dir)
        .map(|rd| rd.count())
        .unwrap_or(0);
    assert!(
        n_orphan_manifests >= 1,
        "orphan manifest must be present; found {n_orphan_manifests} in {manifest_dir:?}"
    );
}

#[test]
fn crash_post_superfile_on_second_commit_yields_v1() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let dir = spawn_crash_child(
        "crash_post_superfile_on_second_commit_yields_v1",
        KP_SEG_SECOND,
    );

    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
    let consumer =
        Supertable::open(default_supertable_options().with_storage(storage)).expect("open at v1");
    assert_eq!(consumer.manifest_id(), 1, "must recover at v1");
    assert_eq!(
        consumer.reader().expect("reader").n_superfiles(),
        1,
        "v1 has exactly the first commit's superfile; v2's orphan superfile is invisible"
    );
}

#[test]
fn crash_post_list_on_second_commit_yields_v1() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let dir = spawn_crash_child("crash_post_list_on_second_commit_yields_v1", KP_LIST_SECOND);

    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
    let consumer =
        Supertable::open(default_supertable_options().with_storage(storage)).expect("open at v1");
    assert_eq!(consumer.manifest_id(), 1);
    assert_eq!(consumer.reader().expect("reader").n_superfiles(), 1);

    // Orphan v2 manifest list and v2 part are on disk —
    // tolerated here; compaction GCs them later.
    let manifest_dir = dir.join("manifest");
    let n_lists = std::fs::read_dir(&manifest_dir)
        .map(|rd| rd.count())
        .unwrap_or(0);
    assert!(
        n_lists >= 2,
        "v1 list + orphan v2 list both on disk; found {n_lists}"
    );
}

/// Crash after the FIRST split-child superfile PUT, before the batch's
/// hidden list/pointer: the previous (drained) hidden generation stays
/// intact and fully queryable, and an explicit `gc` reclaims the orphaned
/// child bytes (they are younger than any real reclaim grace, but a
/// zero-gap sweep proves they are unreferenced).
#[test]
fn crash_post_hidden_child_superfile_yields_pre_split_index() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let dir = spawn_crash_child(
        "crash_post_hidden_child_superfile_yields_pre_split_index",
        KP_HIDDEN_SPLIT_SEG,
    );
    let deleted = verify_hidden_split_crash(&dir, 1);
    assert!(
        deleted >= 1,
        "the pinned child is released by the recovery stamp and reclaimed; deleted {deleted}"
    );
}

/// Crash after the batch's hidden manifest LIST PUT, before its pointer
/// CAS: still the drained generation (the pointer is the visibility
/// barrier), with the children + orphan list reclaimable.
#[test]
fn crash_post_hidden_list_yields_pre_split_index() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let dir = spawn_crash_child(
        "crash_post_hidden_list_yields_pre_split_index",
        KP_HIDDEN_SPLIT_LIST,
    );
    let deleted = verify_hidden_split_crash(&dir, 0);
    assert!(
        deleted >= 1,
        "the half-stamped pin list is a gc-reclaimable orphan; deleted {deleted}"
    );
}

/// Crash on the bulk repack's FIRST packed-shard PUT, before its slow-CAS
/// pin and commit: the drained (pre-split) hidden generation stays intact
/// and fully queryable, and an explicit `gc` reclaims the unpinned orphan
/// shard.
#[test]
fn crash_post_hidden_repack_shard_yields_pre_split_index() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let dir = spawn_crash_child(
        "crash_post_hidden_repack_shard_yields_pre_split_index",
        KP_HIDDEN_REPACK_SEG,
    );
    let deleted = verify_hidden_split_crash(&dir, 1);
    assert!(
        deleted >= 1,
        "the pinned shard is released by the recovery stamp and reclaimed; deleted {deleted}"
    );
}

/// Crash BETWEEN two batch commits (inside batch 2's pre-upload pin stamp,
/// after batch 1's pointer CAS): the pass's mid-loop contract — each batch
/// is all-or-nothing and a partial pass leaves a valid, partially-split
/// grid the next optimize finishes. Batch 1's split is durable and
/// queryable; batch 2's leftovers (at minimum the half-stamped pin list)
/// are reclaimable; `optimize` completes the pass.
#[test]
fn crash_between_split_batches_yields_first_batch() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let dir = spawn_crash_child(
        "crash_between_split_batches_yields_first_batch",
        KP_HIDDEN_SPLIT_SECOND_LIST,
    );
    let deleted = verify_hidden_split_crash(&dir, 2);
    assert!(
        deleted >= 1,
        "batch 2's crash leftovers are reclaimed across the recovery sweeps; deleted {deleted}"
    );
}

/// Crash immediately AFTER the batch's hidden pointer CAS lands: the split
/// is durable (the reopened hidden index is the post-split generation),
/// immediately queryable with no docs lost (the pre-#498 unrecoverable
/// window's regression tripwire), and the next `optimize` + `gc` complete
/// cleanly.
#[test]
fn crash_post_hidden_pointer_yields_split_index() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let dir = spawn_crash_child(
        "crash_post_hidden_pointer_yields_split_index",
        KP_HIDDEN_SPLIT_POINTER,
    );
    // Split durable; gc must simply succeed (superseded parents remain
    // referenced — reclaiming their dead blocks is the merge phase's job).
    verify_hidden_split_crash(&dir, 2);
}

#[test]
fn crash_post_list_on_second_commit_recovers_next_commit() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let dir = spawn_crash_child(
        "crash_post_list_on_second_commit_recovers_next_commit",
        KP_LIST_SECOND,
    );

    // Recovery opens at v1: the crashed second commit never published, but
    // its manifest list survives on disk, occupying id 2 (lists are
    // conditional-create and never overwritten).
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
    let recovered =
        Supertable::open(default_supertable_options().with_storage(storage)).expect("open at v1");
    assert_eq!(recovered.manifest_id(), 1);

    // The FIRST post-crash commit must publish. Regression: the OCC retry
    // loop used to refresh from the (unmoved) pointer, re-derive the same
    // occupied id 2, and exhaust as WriteContentionExhausted until a GC
    // sweep past the safety gap reclaimed the orphan. It now skips to id 3
    // and leaves the orphan for GC.
    let mut w = recovered.writer().expect("writer");
    w.append(&build_title_batch(&["post crash gamma"]))
        .expect("append");
    w.commit()
        .expect("first post-crash commit must publish past the orphaned list");
    assert_eq!(
        recovered.manifest_id(),
        3,
        "commit skips the orphaned id 2 and publishes at id 3"
    );
    assert_eq!(
        recovered.reader().expect("reader").n_superfiles(),
        2,
        "first commit's superfile + the post-crash commit's superfile"
    );
}

#[test]
fn crash_post_pointer_on_second_commit_yields_v2() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let dir = spawn_crash_child(
        "crash_post_pointer_on_second_commit_yields_v2",
        KP_POINTER_SECOND,
    );

    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
    let consumer =
        Supertable::open(default_supertable_options().with_storage(storage)).expect("open at v2");
    assert_eq!(
        consumer.manifest_id(),
        2,
        "pointer rename completed before crash → commit is durable"
    );
    assert_eq!(
        consumer.reader().expect("reader").n_superfiles(),
        2,
        "v2 sees both commits' superfiles"
    );
}
