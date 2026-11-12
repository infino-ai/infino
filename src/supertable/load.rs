// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A load: batches that several replicas write into one table at once, grafted into it as they
//! commit (see [`merge`]), and finished once with an exact account.
//!
//! The load's state lives in the table's storage, so it survives restarts of every process
//! involved: `_loads/<load>.json` says whether the load is open or finished (and what it
//! delivered), and a replica's `sealed` marker sits beside its manifest. A replica belongs to one
//! load and is named `<load>.<node>`.

use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, Ordering},
};

use arrow_array::RecordBatch;
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt, stream};
use serde::{Deserialize, Serialize};

use crate::{
    InfinoError,
    storage::{REPLICAS_DIR, StorageError, StorageProvider},
    supertable::{handle::Supertable, merge},
};

/// Where a table keeps its loads' records; outside [`REPLICAS_DIR`], so a record never counts as
/// an attached replica.
const LOADS_DIR: &str = "_loads/";

/// A replica's marker, beside its manifest, that it takes no more batches.
const SEALED_KEY: &str = "sealed";

/// Replica manifest objects a finish deletes at once.
const DELETES_IN_FLIGHT: usize = 64;

/// Longest load or node id.
const ID_MAX_LEN: usize = 128;

/// Most graft rounds a finish runs waiting for one that adds nothing: one picks up the last
/// commits, another the final folds.
const FINISH_ROUNDS: usize = 4;

/// What a finished load delivered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadOutcome {
    /// Rows the load added to the table, each batch once.
    pub rows: u64,
    /// Copies of batches left out: resends a replica or another replica had delivered already.
    pub duplicates: usize,
    /// Rows the copies held.
    pub duplicate_rows: u64,
}

/// Where a load is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum LoadState {
    /// No record: never begun on this table.
    Unknown,
    /// Begun and taking batches.
    Open,
    /// Finished, with what it delivered.
    Finished(LoadOutcome),
}

/// One replica's account after a graft round.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplicaAccount {
    /// The replica's name, `<load>.<node>`.
    pub replica: String,
    /// Rows the replica holds, copies included.
    pub rows: u64,
    /// Superfiles the replica holds.
    pub superfiles: usize,
    /// Superfiles this round added to the table.
    pub merged: usize,
    /// Superfiles the table held already.
    pub already_merged: usize,
    /// Superfiles left out as copies.
    pub duplicates: usize,
    /// Rows those copies hold.
    pub duplicate_rows: u64,
    /// Whether the replica takes no more batches.
    pub sealed: bool,
}

impl ReplicaAccount {
    /// Whether every superfile of the replica is in the table or left out as a copy.
    pub fn is_accounted(&self) -> bool {
        self.merged == 0 && self.already_merged + self.duplicates == self.superfiles
    }
}

fn check_id(what: &str, id: &str) -> Result<(), InfinoError> {
    let ok = !id.is_empty()
        && id.len() <= ID_MAX_LEN
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    match ok {
        true => Ok(()),
        false => Err(InfinoError::Config(format!(
            "{what} id {id:?}: 1 to {ID_MAX_LEN} of [A-Za-z0-9_-]"
        ))),
    }
}

fn storage_of(table: &Supertable) -> Result<Arc<dyn StorageProvider>, InfinoError> {
    table
        .inner()
        .options
        .storage
        .clone()
        .ok_or_else(|| InfinoError::Unsupported("a load needs a table on storage".into()))
}

fn backend(e: impl std::fmt::Display) -> InfinoError {
    InfinoError::Backend(e.to_string())
}

fn record_key(load: &str) -> String {
    format!("{LOADS_DIR}{load}.json")
}

fn replica_name(load: &str, node: &str) -> String {
    format!("{load}.{node}")
}

fn replica_prefix(load: &str) -> String {
    format!("{load}.")
}

/// `load`'s record on `table` and the version a conditional write replaces it at.
fn read_record(table: &Supertable, load: &str) -> Result<(LoadState, Option<String>), InfinoError> {
    check_id("load", load)?;
    let storage = storage_of(table)?;
    table.block_on_query(async {
        match storage.get(&record_key(load)).await {
            Ok((bytes, meta)) => Ok((serde_json::from_slice(&bytes).map_err(backend)?, meta.etag)),
            Err(StorageError::NotFound { .. }) => Ok((LoadState::Unknown, None)),
            Err(e) => Err(backend(e)),
        }
    })
}

/// The state `load` is in on `table`.
pub fn load_state(table: &Supertable, load: &str) -> Result<LoadState, InfinoError> {
    read_record(table, load).map(|(state, _)| state)
}

/// Begins `load` on `table`, or reports the state it is already in: beginning an open load again is
/// harmless, and a finished one stays finished (a new load needs a new id).
pub fn begin_load(table: &Supertable, load: &str) -> Result<LoadState, InfinoError> {
    check_id("load", load)?;
    let storage = storage_of(table)?;
    let open = serde_json::to_vec(&LoadState::Open).map_err(backend)?;
    let created = table.block_on_query(async {
        match storage
            .put_atomic(&record_key(load), Bytes::from(open))
            .await
        {
            Ok(_) => Ok(true),
            Err(e) if e.is_conflict() => Ok(false),
            Err(e) => Err(backend(e)),
        }
    })?;
    match created {
        true => Ok(LoadState::Open),
        false => load_state(table, load),
    }
}

/// This node's replica of `load` on `table`, which must be open. A replica sealed before a restart
/// opens sealed.
pub fn open_load_replica(
    table: &Supertable,
    load: &str,
    node: &str,
) -> Result<LoadReplica, InfinoError> {
    check_id("node", node)?;
    match load_state(table, load)? {
        LoadState::Open => {}
        state => {
            return Err(InfinoError::Conflict(format!(
                "load {load} is not open: {state:?}"
            )));
        }
    }
    let replica = merge::open_replica(table, &replica_name(load, node)).map_err(backend)?;
    // A finish between the check above and this replica's creation would never graft it.
    if load_state(table, load)? != LoadState::Open {
        detach_load(table, load)?;
        return Err(InfinoError::Conflict(format!(
            "load {load} finished while this replica opened"
        )));
    }
    let sealed = is_sealed(&replica)?;
    Ok(LoadReplica {
        table: replica,
        load: load.to_owned(),
        closed: AtomicBool::new(sealed),
        in_flight: Mutex::new(0),
        settled: Condvar::new(),
    })
}

fn is_sealed(replica: &Supertable) -> Result<bool, InfinoError> {
    let storage = storage_of(replica)?;
    replica.block_on_query(async {
        match storage.head(SEALED_KEY).await {
            Ok(_) => Ok(true),
            Err(StorageError::NotFound { .. }) => Ok(false),
            Err(e) => Err(backend(e)),
        }
    })
}

fn mark_sealed(replica: &Supertable) -> Result<(), InfinoError> {
    let storage = storage_of(replica)?;
    replica
        .block_on_query(storage.put_overwrite(SEALED_KEY, Bytes::from_static(b"sealed")))
        .map_err(backend)
}

/// An append in flight on a replica, counted until it returns or panics, so a seal waits for it.
struct InFlight<'a>(&'a LoadReplica);

impl<'a> InFlight<'a> {
    fn enter(replica: &'a LoadReplica) -> Result<Self, InfinoError> {
        let mut n = replica.in_flight.lock().expect("in-flight count");
        if replica.closed.load(Ordering::Acquire) {
            return Err(replica.sealed_error());
        }
        *n += 1;
        Ok(InFlight(replica))
    }
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        let mut n = self.0.in_flight.lock().expect("in-flight count");
        *n -= 1;
        self.0.settled.notify_all();
    }
}

/// One node's replica of a load. Share one per node: it tracks the batches in flight.
pub struct LoadReplica {
    table: Supertable,
    load: String,
    /// Set by [`Self::seal`]: no batch is taken after it.
    closed: AtomicBool,
    in_flight: Mutex<usize>,
    settled: Condvar,
}

impl LoadReplica {
    /// Appends `batch` as batch `id` of the load and commits it; `false` when this replica had
    /// committed that id already. Errors once the replica is sealing or sealed, and when a seal
    /// lands during the append (send the batch again; a copy that did land is dropped).
    pub fn append_batch(&self, batch: &RecordBatch, id: &str) -> Result<bool, InfinoError> {
        let _in_flight = InFlight::enter(self)?;
        // Another node sealed this replica (`seal_absent_replica`): a batch committed after the
        // finish read it would never reach the table, so it is not acknowledged.
        if is_sealed(&self.table)? {
            return Err(self.sealed_error());
        }
        // Keyed by load, so loads reusing a batch id never meet.
        let appended = merge::append_batch(&self.table, batch, &format!("{}/{id}", self.load))?;
        if is_sealed(&self.table)? {
            return Err(InfinoError::Conflict(format!(
                "load {}: this replica was sealed during the append; send the batch again",
                self.load
            )));
        }
        Ok(appended)
    }

    fn sealed_error(&self) -> InfinoError {
        InfinoError::Conflict(format!("load {}: this replica is sealed", self.load))
    }

    /// Folds the next run of this replica's index layers, as a background thread calls it while
    /// the load runs; `false` when there was nothing to fold.
    pub fn fold(&self) -> Result<bool, InfinoError> {
        merge::fold_replica(&self.table).map_err(backend)
    }

    /// Takes no more batches, waits for those in flight, folds the whole index into one layer and
    /// marks the replica sealed in storage. Safe to repeat.
    pub fn seal(&self) -> Result<(), InfinoError> {
        {
            let mut n = self.in_flight.lock().expect("in-flight count");
            self.closed.store(true, Ordering::Release);
            while *n > 0 {
                n = self.settled.wait(n).expect("in-flight count");
            }
        }
        if is_sealed(&self.table)? {
            return Ok(());
        }
        merge::fold_replica_all(&self.table).map_err(backend)?;
        mark_sealed(&self.table)
    }

    /// Whether the replica takes no more batches.
    pub fn is_sealed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

/// Seals `node`'s replica of open load `load` on its behalf, when that node is gone: what it
/// committed is grafted as it stands, without its final fold. Errors when there is no such replica.
pub fn seal_absent_replica(table: &Supertable, load: &str, node: &str) -> Result<(), InfinoError> {
    check_id("node", node)?;
    if load_state(table, load)? != LoadState::Open {
        return Err(InfinoError::Conflict(format!("load {load} is not open")));
    }
    match merge::open_existing_replica(table, &replica_name(load, node)).map_err(backend)? {
        Some(replica) => mark_sealed(&replica),
        None => Err(InfinoError::NotFound(format!(
            "load {load} has no replica on node {node}"
        ))),
    }
}

/// Grafts into `table` what every replica of open load `load` committed since the last round, and
/// accounts for each; nothing for a finished load.
pub fn graft_load(table: &Supertable, load: &str) -> Result<Vec<ReplicaAccount>, InfinoError> {
    match load_state(table, load)? {
        LoadState::Open => {}
        LoadState::Finished(_) => return Ok(vec![]),
        LoadState::Unknown => {
            return Err(InfinoError::NotFound(format!(
                "load {load} was never begun"
            )));
        }
    }
    let prefix = replica_prefix(load);
    let names: Vec<String> = merge::attached_replicas(table)
        .map_err(backend)?
        .into_iter()
        .filter(|n| n.starts_with(&prefix))
        .collect();
    let mut accounts = vec![];
    for name in names {
        let Some(replica) = cached_replica(table, &name)? else {
            continue;
        };
        // Read before the snapshot, so a replica already sealed has every batch it acknowledged
        // in the snapshot.
        let sealed = is_sealed(&replica)?;
        let reader = replica.reader().map_err(backend)?;
        let report = merge::graft(table, &reader, true).map_err(backend)?;
        let live = &reader.manifest().superfiles;
        accounts.push(ReplicaAccount {
            rows: live.iter().map(|e| e.n_docs).sum(),
            superfiles: live.len(),
            merged: report.merged,
            already_merged: report.already_merged,
            duplicates: report.duplicates,
            duplicate_rows: report.duplicate_rows,
            sealed,
            replica: name,
        });
    }
    Ok(accounts)
}

/// Replica `name` of `table`, opened once per handle and kept for later rounds; `None` once it is
/// gone (detached by a finish).
fn cached_replica(table: &Supertable, name: &str) -> Result<Option<Supertable>, InfinoError> {
    let mut handles = table
        .inner()
        .replica_handles
        .lock()
        .expect("replica handles");
    if let Some(replica) = handles.get(name) {
        return Ok(Some(replica.clone()));
    }
    let replica = merge::open_existing_replica(table, name).map_err(backend)?;
    if let Some(replica) = &replica {
        handles.insert(name.to_owned(), replica.clone());
    }
    Ok(replica)
}

/// Finishes `load`: once every replica is sealed, grafts until a round adds nothing, checks that
/// every replica is accounted for (and the rows, given `expected_rows`), records the outcome and
/// detaches the load's replicas. A finished load reports its outcome again, finishing any detach
/// a crash cut short.
///
/// Errors, leaving the load open and its replicas attached, when a replica is not sealed, a
/// replica is not accounted for, or the rows differ from `expected_rows`.
pub fn finish_load(
    table: &Supertable,
    load: &str,
    expected_rows: Option<u64>,
) -> Result<LoadOutcome, InfinoError> {
    let (state, version) = read_record(table, load)?;
    match state {
        LoadState::Finished(outcome) => {
            detach_load(table, load)?;
            return Ok(outcome);
        }
        LoadState::Unknown => {
            return Err(InfinoError::NotFound(format!(
                "load {load} was never begun"
            )));
        }
        LoadState::Open => {}
    }
    let mut accounts = graft_load(table, load)?;
    let unsealed: Vec<&str> = accounts
        .iter()
        .filter(|a| !a.sealed)
        .map(|a| a.replica.as_str())
        .collect();
    if !unsealed.is_empty() {
        return Err(InfinoError::Conflict(format!(
            "load {load}: replicas not sealed: {unsealed:?}"
        )));
    }
    for _ in 1..FINISH_ROUNDS {
        if accounts.iter().all(|a| a.merged == 0) {
            break;
        }
        accounts = graft_load(table, load)?;
    }
    let unaccounted: Vec<&str> = accounts
        .iter()
        .filter(|a| !a.is_accounted())
        .map(|a| a.replica.as_str())
        .collect();
    if !unaccounted.is_empty() {
        return Err(InfinoError::Conflict(format!(
            "load {load}: replicas not accounted for: {unaccounted:?}"
        )));
    }
    let outcome = LoadOutcome {
        rows: accounts.iter().map(|a| a.rows - a.duplicate_rows).sum(),
        duplicates: accounts.iter().map(|a| a.duplicates).sum(),
        duplicate_rows: accounts.iter().map(|a| a.duplicate_rows).sum(),
    };
    if let Some(want) = expected_rows.filter(|&w| w != outcome.rows) {
        return Err(InfinoError::Conflict(format!(
            "load {load} delivered {} rows, expected {want}",
            outcome.rows
        )));
    }
    let storage = storage_of(table)?;
    let record = serde_json::to_vec(&LoadState::Finished(outcome.clone())).map_err(backend)?;
    // Written only over the open record this finish read; a finish that lost the race reports
    // the winner's outcome.
    let won = table.block_on_query(async {
        match storage
            .put_if_match(&record_key(load), Bytes::from(record), version.as_deref())
            .await
        {
            Ok(_) => Ok(true),
            Err(e) if e.is_conflict() => Ok(false),
            Err(e) => Err(backend(e)),
        }
    })?;
    if !won {
        return match load_state(table, load)? {
            LoadState::Finished(outcome) => {
                detach_load(table, load)?;
                Ok(outcome)
            }
            state => Err(InfinoError::Conflict(format!(
                "load {load} changed during its finish: {state:?}"
            ))),
        };
    }
    detach_load(table, load)?;
    Ok(outcome)
}

/// Deletes the manifests of `load`'s replicas under [`REPLICAS_DIR`].
fn detach_load(table: &Supertable, load: &str) -> Result<usize, InfinoError> {
    let storage = storage_of(table)?;
    table
        .inner()
        .replica_handles
        .lock()
        .expect("replica handles")
        .retain(|name, _| !name.starts_with(&replica_prefix(load)));
    let prefix = format!("{REPLICAS_DIR}{}", replica_prefix(load));
    table.block_on_query(async {
        let keys = storage
            .list_with_prefix(REPLICAS_DIR)
            .await
            .map_err(backend)?;
        let mine: Vec<&String> = keys.iter().filter(|k| k.starts_with(&prefix)).collect();
        stream::iter(&mine)
            .map(|key| {
                let storage = &storage;
                async move { storage.delete(key).await.map_err(backend) }
            })
            .buffer_unordered(DELETES_IN_FLIGHT)
            .try_collect::<Vec<()>>()
            .await?;
        crate::storage::settle_attached(storage.as_ref())
            .await
            .map_err(backend)?;
        Ok(mine.len())
    })
}

#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, LargeStringArray};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};

    use super::*;
    use crate::{
        storage::LocalFsStorageProvider, superfile::builder::FtsConfig,
        supertable::SupertableOptions,
    };

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]))
    }

    fn table() -> (tempfile::TempDir, Supertable) {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs"));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool"),
        );
        let options = SupertableOptions::new(schema(), vec![FtsConfig::new("title")], vec![])
            .expect("options")
            .with_writer_pool(pool)
            .with_read_consistency(crate::Consistency::Strong)
            .with_storage(storage);
        (dir, Supertable::create(options).expect("create"))
    }

    fn batch(rows: usize) -> RecordBatch {
        let arr: ArrayRef = Arc::new(LargeStringArray::from(vec!["shared x"; rows]));
        RecordBatch::try_new(schema(), vec![arr]).expect("batch")
    }

    fn rows(t: &Supertable) -> u64 {
        t.reader()
            .expect("reader")
            .manifest()
            .superfiles
            .iter()
            .map(|e| e.n_docs)
            .sum()
    }

    #[test]
    fn two_replicas_load_finish_and_report_once() {
        let (_dir, t) = table();
        assert_eq!(begin_load(&t, "l1").expect("begin"), LoadState::Open);
        let (a, b) = (
            open_load_replica(&t, "l1", "n1").expect("a"),
            open_load_replica(&t, "l1", "n2").expect("b"),
        );
        assert!(a.append_batch(&batch(10), "b1").expect("append"));
        assert!(b.append_batch(&batch(10), "b2").expect("append"));
        assert!(
            b.append_batch(&batch(10), "b1")
                .expect("a copy on another replica")
        );
        graft_load(&t, "l1").expect("graft");
        assert_eq!(rows(&t), 20, "queryable mid-load, b1 once");
        a.seal().expect("seal");
        b.seal().expect("seal");
        let outcome = finish_load(&t, "l1", Some(20)).expect("finish");
        assert_eq!(
            outcome,
            LoadOutcome {
                rows: 20,
                duplicates: 1,
                duplicate_rows: 10
            }
        );
        assert!(!t.has_attached_replicas().expect("attached"));
        assert_eq!(
            load_state(&t, "l1").expect("state"),
            LoadState::Finished(outcome.clone())
        );
        assert_eq!(
            finish_load(&t, "l1", Some(20)).expect("again"),
            outcome,
            "a repeat reports it"
        );
        assert!(
            b.append_batch(&batch(1), "late").is_err(),
            "a sealed replica takes nothing"
        );
        assert!(
            open_load_replica(&t, "l1", "n3").is_err(),
            "a finished load takes no replica"
        );
        assert_eq!(rows(&t), 20);
    }

    #[test]
    fn loads_reusing_batch_ids_both_deliver() {
        let (_dir, t) = table();
        for load in ["la", "lb"] {
            begin_load(&t, load).expect("begin");
            let r = open_load_replica(&t, load, "n1").expect("replica");
            assert!(r.append_batch(&batch(10), "b").expect("append"));
            r.seal().expect("seal");
            assert_eq!(finish_load(&t, load, Some(10)).expect("finish").rows, 10);
        }
        assert_eq!(rows(&t), 20);
    }

    #[test]
    fn finishing_one_load_leaves_another_attached() {
        let (_dir, t) = table();
        begin_load(&t, "la").expect("begin");
        begin_load(&t, "lb").expect("begin");
        let a = open_load_replica(&t, "la", "n1").expect("replica");
        a.append_batch(&batch(10), "b1").expect("append");
        assert_eq!(finish_load(&t, "lb", None).expect("finish").rows, 0);
        assert!(
            t.has_attached_replicas().expect("attached"),
            "la is still loading"
        );
        assert!(a.append_batch(&batch(10), "b2").expect("la goes on"));
        a.seal().expect("seal");
        assert_eq!(finish_load(&t, "la", None).expect("finish").rows, 20);
        assert!(!t.has_attached_replicas().expect("attached"));
    }

    #[test]
    fn an_unsealed_replica_holds_the_finish_until_sealed_for_it() {
        let (_dir, t) = table();
        begin_load(&t, "l1").expect("begin");
        let gone = open_load_replica(&t, "l1", "n1").expect("replica");
        gone.append_batch(&batch(10), "b1").expect("append");
        assert!(finish_load(&t, "l1", None).is_err(), "n1 is not sealed");
        assert!(
            t.has_attached_replicas().expect("attached"),
            "a failed finish detaches nothing"
        );
        seal_absent_replica(&t, "l1", "n1").expect("seal for the gone node");
        assert_eq!(finish_load(&t, "l1", None).expect("finish").rows, 10);
    }

    #[test]
    fn a_sealed_replica_reopens_sealed() {
        let (_dir, t) = table();
        begin_load(&t, "l1").expect("begin");
        open_load_replica(&t, "l1", "n1")
            .expect("replica")
            .seal()
            .expect("seal");
        let reopened = open_load_replica(&t, "l1", "n1").expect("reopen");
        assert!(reopened.is_sealed());
        assert!(reopened.append_batch(&batch(1), "b").is_err());
        assert!(finish_load(&t, "nope", None).is_err(), "never begun");
        assert!(begin_load(&t, "bad/id").is_err());
    }

    #[test]
    fn a_row_count_mismatch_fails_the_finish_and_keeps_the_load_open() {
        let (_dir, t) = table();
        begin_load(&t, "l1").expect("begin");
        let r = open_load_replica(&t, "l1", "n1").expect("replica");
        r.append_batch(&batch(10), "b1").expect("append");
        r.seal().expect("seal");
        assert!(finish_load(&t, "l1", Some(11)).is_err());
        assert_eq!(load_state(&t, "l1").expect("state"), LoadState::Open);
        assert!(t.has_attached_replicas().expect("attached"));
        assert_eq!(finish_load(&t, "l1", Some(10)).expect("finish").rows, 10);
    }

    #[test]
    fn a_finished_load_takes_no_seal_or_graft_and_leaves_nothing_attached() {
        let (_dir, t) = table();
        begin_load(&t, "l1").expect("begin");
        let r = open_load_replica(&t, "l1", "n1").expect("replica");
        r.append_batch(&batch(10), "b1").expect("append");
        assert!(matches!(
            seal_absent_replica(&t, "l1", "n2"),
            Err(InfinoError::NotFound(_))
        ));
        r.seal().expect("seal");
        finish_load(&t, "l1", Some(10)).expect("finish");
        assert!(seal_absent_replica(&t, "l1", "n1").is_err());
        assert!(graft_load(&t, "l1").expect("graft").is_empty());
        assert!(!t.has_attached_replicas().expect("attached"));
        let storage = storage_of(&t).expect("storage");
        assert!(
            !t.block_on_query(crate::storage::may_be_attached(storage.as_ref())),
            "the marker goes with the last replica"
        );
        assert!(matches!(
            graft_load(&t, "never"),
            Err(InfinoError::NotFound(_))
        ));
    }

    #[test]
    fn a_replica_sealed_by_another_refuses_appends() {
        let (_dir, t) = table();
        begin_load(&t, "l1").expect("begin");
        let r = open_load_replica(&t, "l1", "n1").expect("replica");
        r.append_batch(&batch(10), "b1").expect("append");
        seal_absent_replica(&t, "l1", "n1").expect("sealed for it");
        assert!(matches!(
            r.append_batch(&batch(10), "b2"),
            Err(InfinoError::Conflict(_))
        ));
        assert_eq!(finish_load(&t, "l1", None).expect("finish").rows, 10);
    }
}
