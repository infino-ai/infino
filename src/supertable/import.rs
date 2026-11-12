// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! An import: batches that several replicas write into one table at once, grafted into it as they
//! commit (see [`merge`]), and finished once with an exact account.
//!
//! The import's state lives in the table's storage, so it survives restarts of every process
//! involved: `_imports/<import>.json` says whether the import is open or finished (and what it
//! delivered), and a replica's `sealed` marker sits beside its manifest. A replica belongs to one
//! import and is named `<import>.<node>`.

use std::{
    num::NonZeroU32,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::SystemTime,
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

/// Where a table keeps its imports' records; outside [`REPLICAS_DIR`], so a record never counts as
/// an attached replica.
const IMPORTS_DIR: &str = "_imports/";

/// A replica's marker, beside its manifest, that it takes no more batches.
const SEALED_KEY: &str = "sealed";

/// Replica manifest objects a finish deletes at once.
const DELETES_IN_FLIGHT: usize = 64;

/// Longest import or node id.
const ID_MAX_LEN: usize = 128;

/// Most imports a table holds open at once; each runs its own graft rounds against the table.
const MAX_OPEN_IMPORTS: usize = 8;

/// Record writes an abort tries before giving up to a record that keeps changing.
const ABORT_TRIES: usize = 3;

/// Most graft rounds a finish runs waiting for one that adds nothing: one picks up the last
/// commits, another the final folds.
const FINISH_ROUNDS: usize = 4;

/// What a finished import delivered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportOutcome {
    /// Rows the import added to the table, each batch once.
    pub rows: u64,
    /// Copies of batches left out: resends a replica or another replica had delivered already.
    pub duplicates: usize,
    /// Rows the copies held.
    pub duplicate_rows: u64,
}

/// Where an import is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ImportState {
    /// No record: never begun on this table.
    Unknown,
    /// Begun and taking batches.
    Open,
    /// A finish is running: existing replicas keep taking batches, and no new one opens.
    Finishing,
    /// Finished, with what it delivered.
    Finished(ImportOutcome),
    /// Ended without a finish: the batches grafted before stay in the table, the rest are dropped.
    Aborted,
}

/// One replica's account after a graft round.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplicaAccount {
    /// The replica's name, `<import>.<node>`.
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
        .ok_or_else(|| InfinoError::Unsupported("an import needs a table on storage".into()))
}

fn backend(e: impl std::fmt::Display) -> InfinoError {
    InfinoError::Backend(e.to_string())
}

fn record_key(import: &str) -> String {
    format!("{IMPORTS_DIR}{import}.json")
}

fn replica_name(import: &str, node: &str) -> String {
    format!("{import}.{node}")
}

fn replica_prefix(import: &str) -> String {
    format!("{import}.")
}

/// How many replicas write an import's batches. The engine records it; whoever routes the batches
/// keeps to it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Writers {
    /// Every replica that is up.
    #[default]
    Auto,
    /// At most this many replicas; one is a single-writer build.
    Count(NonZeroU32),
}

/// An import's stored record.
#[derive(Serialize, Deserialize)]
struct Record {
    #[serde(flatten)]
    state: ImportState,
    #[serde(default)]
    writers: Writers,
    /// Unique to each finish's claim, so two claims never store the same bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    claim: Option<String>,
}

/// `import`'s record on `table` and the version a conditional write replaces it at.
fn read_record(table: &Supertable, import: &str) -> Result<(Record, Option<String>), InfinoError> {
    check_id("import", import)?;
    let storage = storage_of(table)?;
    table.block_on_query(async {
        match storage.get(&record_key(import)).await {
            Ok((bytes, meta)) => Ok((serde_json::from_slice(&bytes).map_err(backend)?, meta.etag)),
            Err(StorageError::NotFound { .. }) => Ok((
                Record {
                    state: ImportState::Unknown,
                    writers: Writers::Auto,
                    claim: None,
                },
                None,
            )),
            Err(e) => Err(backend(e)),
        }
    })
}

/// Replaces `import`'s record at `version` with `state`; the new version, or `None` when another
/// writer replaced it first.
fn write_record(
    table: &Supertable,
    import: &str,
    state: ImportState,
    writers: Writers,
    version: Option<&str>,
) -> Result<Option<String>, InfinoError> {
    let claim = matches!(state, ImportState::Finishing).then(|| uuid::Uuid::new_v4().to_string());
    let bytes = serde_json::to_vec(&Record {
        state,
        writers,
        claim,
    })
    .map_err(backend)?;
    let storage = storage_of(table)?;
    let written = table.block_on_query(async {
        match storage
            .put_if_match(&record_key(import), Bytes::from(bytes), version)
            .await
        {
            Ok(etag) => Ok(Some(etag)),
            Err(e) if e.is_conflict() => Ok(None),
            Err(e) => Err(backend(e)),
        }
    })?;
    match written {
        Some(Some(etag)) => Ok(Some(etag)),
        Some(None) => Err(InfinoError::Unsupported(
            "an import needs storage that versions its objects".into(),
        )),
        None => Ok(None),
    }
}

/// The error for an import that is not open: final once it is finished or was never begun, worth
/// retrying while a finish runs.
fn not_open(import: &str, state: &ImportState) -> InfinoError {
    match state {
        ImportState::Finishing => InfinoError::Conflict(format!(
            "import {import} is finishing and takes no new replica"
        )),
        ImportState::Finished(_) => InfinoError::NotFound(format!("import {import} is finished")),
        ImportState::Aborted => InfinoError::NotFound(format!("import {import} was aborted")),
        _ => InfinoError::NotFound(format!("import {import} was never begun")),
    }
}

/// The state `import` is in on `table`.
pub fn import_state(table: &Supertable, import: &str) -> Result<ImportState, InfinoError> {
    read_record(table, import).map(|(record, _)| record.state)
}

/// How many replicas write `import`; errors with [`InfinoError::NotFound`] when it was never begun.
pub fn import_writers(table: &Supertable, import: &str) -> Result<Writers, InfinoError> {
    let (record, _) = read_record(table, import)?;
    match record.state {
        ImportState::Unknown => Err(not_open(import, &record.state)),
        _ => Ok(record.writers),
    }
}

/// The imports on `table` that are open or finishing, so a coordinator that restarted can take
/// them up again.
pub fn open_imports(table: &Supertable) -> Result<Vec<String>, InfinoError> {
    let storage = storage_of(table)?;
    let keys = table
        .block_on_query(storage.list_with_prefix(IMPORTS_DIR))
        .map_err(backend)?;
    let mut open = vec![];
    for key in keys {
        let Some(import) = key
            .strip_prefix(IMPORTS_DIR)
            .and_then(|k| k.strip_suffix(".json"))
        else {
            continue;
        };
        if check_id("import", import).is_ok()
            && matches!(
                import_state(table, import)?,
                ImportState::Open | ImportState::Finishing
            )
        {
            open.push(import.to_owned());
        }
    }
    Ok(open)
}

/// Begins `import` on `table` with `writers` (every replica when `None`), or reports the state it
/// is already in: beginning an open import again is harmless, and a finished one stays finished (a
/// new import needs a new id). Errors with [`InfinoError::Conflict`] when an unfinished import
/// began with other `writers`.
pub fn begin_import(
    table: &Supertable,
    import: &str,
    writers: Option<Writers>,
) -> Result<ImportState, InfinoError> {
    check_id("import", import)?;
    let storage = storage_of(table)?;
    let snapshot = table.inner().manifest.load();
    if !snapshot.vector_configs().is_empty()
        || !snapshot.companions().is_empty()
        || snapshot.slow_vector_state_blob().is_some()
    {
        return Err(InfinoError::Unsupported(
            "an import can't fill a table with vector columns or companion tables".into(),
        ));
    }
    if read_record(table, import)?.0.state == ImportState::Unknown
        && open_imports(table)?.len() >= MAX_OPEN_IMPORTS
    {
        return Err(InfinoError::Conflict(format!(
            "the table has {MAX_OPEN_IMPORTS} imports open; finish one first"
        )));
    }
    let open = serde_json::to_vec(&Record {
        state: ImportState::Open,
        writers: writers.unwrap_or_default(),
        claim: None,
    })
    .map_err(backend)?;
    let created = table.block_on_query(async {
        match storage
            .put_atomic(&record_key(import), Bytes::from(open))
            .await
        {
            Ok(_) => Ok(true),
            Err(e) if e.is_conflict() => Ok(false),
            Err(e) => Err(backend(e)),
        }
    })?;
    if created {
        return Ok(ImportState::Open);
    }
    let (record, _) = read_record(table, import)?;
    match (&record.state, writers) {
        (ImportState::Open | ImportState::Finishing, Some(w)) if w != record.writers => {
            Err(InfinoError::Conflict(format!(
                "import {import} began with writers {:?}, not {w:?}",
                record.writers
            )))
        }
        _ => Ok(record.state),
    }
}

/// This node's replica of `import` on `table`, which must be open. A replica sealed before a restart
/// opens sealed.
pub fn open_import_replica(
    table: &Supertable,
    import: &str,
    node: &str,
) -> Result<ImportReplica, InfinoError> {
    check_id("node", node)?;
    let (record, _) = read_record(table, import)?;
    let state = record.state;
    if !matches!(state, ImportState::Open | ImportState::Finishing) {
        return Err(not_open(import, &state));
    }
    let name = replica_name(import, node);
    let replica = match merge::open_existing_replica(table, &name).map_err(backend)? {
        Some(replica) => replica,
        None if state != ImportState::Open => return Err(not_open(import, &state)),
        None => {
            check_writers(table, import, record.writers, 0)?;
            let replica = merge::open_replica(table, &name).map_err(backend)?;
            // A finish lists replicas after it records `Finishing`, so a replica created after
            // that may be missing from its listing and would never be grafted.
            let state = import_state(table, import)?;
            if state != ImportState::Open {
                detach(table, &format!("{REPLICAS_DIR}{name}/"))?;
                return Err(not_open(import, &state));
            }
            // Two replicas opening at once can both pass the first check.
            if let Err(e) = check_writers(table, import, record.writers, 1) {
                detach(table, &format!("{REPLICAS_DIR}{name}/"))?;
                return Err(e);
            }
            replica
        }
    };
    let sealed = is_sealed(&replica)?;
    Ok(ImportReplica {
        table: replica,
        owner: table.clone(),
        import: import.to_owned(),
        closed: AtomicBool::new(sealed),
        in_flight: Mutex::new(0),
        settled: Condvar::new(),
    })
}

/// Refuses another writer when `import`'s unsealed replicas, besides the `opened` just created,
/// already reach its `writers`.
fn check_writers(
    table: &Supertable,
    import: &str,
    writers: Writers,
    opened: usize,
) -> Result<(), InfinoError> {
    let Writers::Count(most) = writers else {
        return Ok(());
    };
    let storage = storage_of(table)?;
    let prefix = replica_prefix(import);
    let names: Vec<String> = merge::attached_replicas(table)
        .map_err(backend)?
        .into_iter()
        .filter(|n| n.starts_with(&prefix))
        .collect();
    let mut unsealed = 0;
    for name in names {
        let marker = format!("{REPLICAS_DIR}{name}/{SEALED_KEY}");
        let sealed = table.block_on_query(async {
            match storage.head(&marker).await {
                Ok(_) => Ok(true),
                Err(StorageError::NotFound { .. }) => Ok(false),
                Err(e) => Err(backend(e)),
            }
        })?;
        unsealed += usize::from(!sealed);
    }
    match unsealed.saturating_sub(opened) < most.get() as usize {
        true => Ok(()),
        false => Err(InfinoError::Conflict(format!(
            "import {import} takes at most {most} writers; send the batch to one of them"
        ))),
    }
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
struct InFlight<'a>(&'a ImportReplica);

impl<'a> InFlight<'a> {
    fn enter(replica: &'a ImportReplica) -> Result<Self, InfinoError> {
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

/// One node's replica of an import. Share one per node: it tracks the batches in flight.
pub struct ImportReplica {
    table: Supertable,
    /// The table the import fills, which holds the import's record.
    owner: Supertable,
    import: String,
    /// Set by [`Self::seal`]: no batch is taken after it.
    closed: AtomicBool,
    in_flight: Mutex<usize>,
    settled: Condvar,
}

impl ImportReplica {
    /// Appends `batch` as batch `id` of the import and commits it; `false` when this replica had
    /// committed that id already. Errors once the replica is sealing or sealed, and when a seal
    /// lands during the append (send the batch again; a copy that did land is dropped).
    pub fn append_batch(&self, batch: &RecordBatch, id: &str) -> Result<bool, InfinoError> {
        let _in_flight = InFlight::enter(self)?;
        // Another node sealed this replica (`seal_absent_replica`): a batch committed after the
        // finish read it would never reach the table, so it is not acknowledged.
        if self.check_sealed()? {
            return Err(self.sealed_error());
        }
        // The graft needs the replica's schema to stay the table's.
        let schema = self.owner.table_schema();
        let offered = batch.schema();
        let added: Vec<&str> = offered
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .filter(|name| schema.field_named(name).is_none())
            .collect();
        if !added.is_empty() {
            return Err(InfinoError::Unsupported(format!(
                "import {}: a batch can't add columns {added:?}; add them to the table before the import begins",
                self.import
            )));
        }
        // Keyed by import, so imports reusing a batch id never meet.
        let appended = merge::append_batch(&self.table, batch, &format!("{}/{id}", self.import))?;
        if self.check_sealed()? {
            return Err(InfinoError::Conflict(format!(
                "import {}: this replica was sealed during the append; send the batch again",
                self.import
            )));
        }
        // A finish deletes the seal marker when it detaches, so its record is checked too.
        match import_state(&self.owner, &self.import)? {
            ImportState::Open | ImportState::Finishing => Ok(appended),
            state => Err(not_open(&self.import, &state)),
        }
    }

    /// Whether the replica is sealed in storage; once it is, [`Self::is_sealed`] says so too.
    fn check_sealed(&self) -> Result<bool, InfinoError> {
        let sealed = is_sealed(&self.table)?;
        if sealed {
            self.closed.store(true, Ordering::Release);
        }
        Ok(sealed)
    }

    fn sealed_error(&self) -> InfinoError {
        InfinoError::Conflict(format!("import {}: this replica is sealed", self.import))
    }

    /// Folds the next run of this replica's index layers, as a background thread calls it while
    /// the import runs; `false` when there was nothing to fold.
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

    /// Whether the replica takes no more batches: sealed here, or found sealed in storage by an
    /// append (a [`InfinoError::Conflict`] from [`Self::append_batch`] with this `false` is plain
    /// write contention).
    pub fn is_sealed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

/// Seals `node`'s replica of the open `import` on its behalf, when that node is gone: what it
/// committed is grafted as it stands, without its final fold. Errors when there is no such replica.
pub fn seal_absent_replica(
    table: &Supertable,
    import: &str,
    node: &str,
) -> Result<(), InfinoError> {
    check_id("node", node)?;
    let state = import_state(table, import)?;
    if !matches!(state, ImportState::Open | ImportState::Finishing) {
        return Err(not_open(import, &state));
    }
    let name = replica_name(import, node);
    let Some(replica) = merge::open_existing_replica(table, &name).map_err(backend)? else {
        return Err(InfinoError::NotFound(format!(
            "import {import} has no replica on node {node}"
        )));
    };
    mark_sealed(&replica)?;
    // If a finish detached in between, this marker would count as an attached replica.
    let state = import_state(table, import)?;
    if matches!(state, ImportState::Finished(_) | ImportState::Aborted) {
        detach(table, &format!("{REPLICAS_DIR}{name}/"))?;
        return Err(not_open(import, &state));
    }
    Ok(())
}

/// Grafts into `table` what every replica of the open `import` committed since the last round, and
/// accounts for each; nothing for a finished import.
pub fn graft_import(table: &Supertable, import: &str) -> Result<Vec<ReplicaAccount>, InfinoError> {
    match import_state(table, import)? {
        ImportState::Open | ImportState::Finishing => {}
        ImportState::Finished(_) | ImportState::Aborted => {
            forget_handles(table, &format!("{REPLICAS_DIR}{}", replica_prefix(import)));
            return Ok(vec![]);
        }
        ImportState::Unknown => {
            return Err(InfinoError::NotFound(format!(
                "import {import} was never begun"
            )));
        }
    }
    let prefix = replica_prefix(import);
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

/// Aborts `import`, also while a finish runs: the batches grafted so far stay in the table, the
/// rest are dropped with the replicas. Errors with [`InfinoError::NotFound`] once it is finished
/// or was never begun.
pub fn abort_import(table: &Supertable, import: &str) -> Result<(), InfinoError> {
    for _ in 0..ABORT_TRIES {
        let (record, version) = read_record(table, import)?;
        match record.state {
            ImportState::Aborted => {}
            ImportState::Open | ImportState::Finishing => {
                let written = write_record(
                    table,
                    import,
                    ImportState::Aborted,
                    record.writers,
                    version.as_deref(),
                )?;
                if written.is_none() {
                    continue;
                }
            }
            state => return Err(not_open(import, &state)),
        }
        detach_import(table, import)?;
        return Ok(());
    }
    Err(InfinoError::Conflict(format!(
        "import {import} kept changing during its abort"
    )))
}

/// When a replica of `import` last committed, or `None` when it has no replica; the same answer on
/// every node, so an idle import can be judged from storage.
pub fn import_activity(
    table: &Supertable,
    import: &str,
) -> Result<Option<SystemTime>, InfinoError> {
    check_id("import", import)?;
    let storage = storage_of(table)?;
    let prefix = format!("{REPLICAS_DIR}{}", replica_prefix(import));
    let objects = table
        .block_on_query(storage.list_with_prefix_metadata(REPLICAS_DIR))
        .map_err(backend)?;
    Ok(objects
        .into_iter()
        .filter(|(key, _)| key.starts_with(&prefix))
        .map(|(_, meta)| meta.last_modified)
        .max())
}

/// Refuses `op` on a replica, and on a table while an import has replicas attached: rows not yet
/// grafted would miss the change, and a schema change would stop every graft.
pub(crate) fn refuse_during_import(table: &Supertable, op: &str) -> Result<(), InfinoError> {
    if table.inner().options.term_index_replica {
        return Err(InfinoError::Unsupported(format!(
            "{op} on an import's replica"
        )));
    }
    let Some(storage) = table.inner().options.storage.clone() else {
        return Ok(());
    };
    match table.block_on_query(crate::storage::is_attached(storage.as_ref())) {
        Ok(false) => Ok(()),
        Ok(true) => Err(InfinoError::Conflict(format!(
            "{op} waits until the table's open imports finish or are aborted"
        ))),
        Err(e) => Err(backend(e)),
    }
}

/// Finishes `import`: once every replica is sealed, grafts until a round adds nothing, checks that
/// every replica is accounted for (and the rows, given `expected_rows`), records the outcome and
/// detaches the import's replicas. A finished import reports its outcome again, finishing any detach
/// a crash cut short.
///
/// While it runs the import is [`ImportState::Finishing`] and takes no new replica; a later finish takes
/// over one that crashed. Errors with [`InfinoError::Conflict`], leaving the import open and its
/// replicas attached, when a replica is not sealed, a replica is not accounted for, the rows differ
/// from `expected_rows`, or another finish is running; [`InfinoError::NotFound`] for an import never
/// begun.
pub fn finish_import(
    table: &Supertable,
    import: &str,
    expected_rows: Option<u64>,
) -> Result<ImportOutcome, InfinoError> {
    let (record, version) = read_record(table, import)?;
    let (state, writers) = (record.state, record.writers);
    match state {
        ImportState::Finished(outcome) => {
            detach_import(table, import)?;
            if let Some(want) = expected_rows.filter(|&w| w != outcome.rows) {
                return Err(InfinoError::Cardinality(format!(
                    "import {import} finished with {} rows, not {want}",
                    outcome.rows
                )));
            }
            return Ok(outcome);
        }
        ImportState::Unknown | ImportState::Aborted => return Err(not_open(import, &state)),
        ImportState::Open | ImportState::Finishing => {}
    }
    // The claim is the record's version: a finish that crashed is taken over, and one that was
    // taken over can neither finish nor reopen the import.
    let Some(claim) = write_record(
        table,
        import,
        ImportState::Finishing,
        writers,
        version.as_deref(),
    )?
    else {
        return finished_by_another(table, import);
    };
    let finished = finish_claimed(table, import, expected_rows, writers, &claim);
    if finished.is_err() {
        write_record(table, import, ImportState::Open, writers, Some(&claim))?;
    }
    finished
}

/// What a finish that lost its claim reports: the outcome once the other finish recorded it.
fn finished_by_another(table: &Supertable, import: &str) -> Result<ImportOutcome, InfinoError> {
    match import_state(table, import)? {
        ImportState::Finished(outcome) => {
            detach_import(table, import)?;
            Ok(outcome)
        }
        ImportState::Aborted => Err(not_open(import, &ImportState::Aborted)),
        _ => Err(InfinoError::Conflict(format!(
            "another finish of import {import} is running"
        ))),
    }
}

fn check_sealed(import: &str, accounts: &[ReplicaAccount]) -> Result<(), InfinoError> {
    let unsealed: Vec<&str> = accounts
        .iter()
        .filter(|a| !a.sealed)
        .map(|a| a.replica.as_str())
        .collect();
    match unsealed.is_empty() {
        true => Ok(()),
        false => Err(InfinoError::Conflict(format!(
            "import {import}: replicas not sealed: {unsealed:?}"
        ))),
    }
}

/// The finish of `import` once it holds `claim`.
fn finish_claimed(
    table: &Supertable,
    import: &str,
    expected_rows: Option<u64>,
    writers: Writers,
    claim: &str,
) -> Result<ImportOutcome, InfinoError> {
    let mut accounts = graft_import(table, import)?;
    check_sealed(import, &accounts)?;
    for _ in 1..FINISH_ROUNDS {
        if accounts.iter().all(|a| a.merged == 0) {
            break;
        }
        accounts = graft_import(table, import)?;
    }
    check_sealed(import, &accounts)?;
    let unaccounted: Vec<&str> = accounts
        .iter()
        .filter(|a| !a.is_accounted())
        .map(|a| a.replica.as_str())
        .collect();
    if !unaccounted.is_empty() {
        return Err(InfinoError::Conflict(format!(
            "import {import}: replicas not accounted for: {unaccounted:?}"
        )));
    }
    let outcome = ImportOutcome {
        rows: accounts.iter().map(|a| a.rows - a.duplicate_rows).sum(),
        duplicates: accounts.iter().map(|a| a.duplicates).sum(),
        duplicate_rows: accounts.iter().map(|a| a.duplicate_rows).sum(),
    };
    if let Some(want) = expected_rows.filter(|&w| w != outcome.rows) {
        return Err(InfinoError::Conflict(format!(
            "import {import} delivered {} rows, expected {want}",
            outcome.rows
        )));
    }
    if write_record(
        table,
        import,
        ImportState::Finished(outcome.clone()),
        writers,
        Some(claim),
    )?
    .is_none()
    {
        return finished_by_another(table, import);
    }
    detach_import(table, import)?;
    Ok(outcome)
}

/// Deletes `import`'s replicas under [`REPLICAS_DIR`].
fn detach_import(table: &Supertable, import: &str) -> Result<usize, InfinoError> {
    detach(table, &format!("{REPLICAS_DIR}{}", replica_prefix(import)))
}

/// Drops the cached handles of replicas whose objects start with `key_prefix`.
fn forget_handles(table: &Supertable, key_prefix: &str) {
    table
        .inner()
        .replica_handles
        .lock()
        .expect("replica handles")
        .retain(|name, _| !format!("{REPLICAS_DIR}{name}/").starts_with(key_prefix));
}

/// Deletes the replica objects under `key_prefix`, then settles the attached marker.
fn detach(table: &Supertable, key_prefix: &str) -> Result<usize, InfinoError> {
    let storage = storage_of(table)?;
    forget_handles(table, key_prefix);
    table.block_on_query(async {
        // Listed by directory: a store may match a prefix only at whole path segments.
        let keys = storage
            .list_with_prefix(REPLICAS_DIR)
            .await
            .map_err(backend)?;
        let mine: Vec<&String> = keys.iter().filter(|k| k.starts_with(key_prefix)).collect();
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
        assert_eq!(
            begin_import(&t, "l1", None).expect("begin"),
            ImportState::Open
        );
        let (a, b) = (
            open_import_replica(&t, "l1", "n1").expect("a"),
            open_import_replica(&t, "l1", "n2").expect("b"),
        );
        assert!(a.append_batch(&batch(10), "b1").expect("append"));
        assert!(b.append_batch(&batch(10), "b2").expect("append"));
        assert!(
            b.append_batch(&batch(10), "b1")
                .expect("a copy on another replica")
        );
        graft_import(&t, "l1").expect("graft");
        assert_eq!(rows(&t), 20, "queryable mid-import, b1 once");
        a.seal().expect("seal");
        b.seal().expect("seal");
        let outcome = finish_import(&t, "l1", Some(20)).expect("finish");
        assert_eq!(
            outcome,
            ImportOutcome {
                rows: 20,
                duplicates: 1,
                duplicate_rows: 10
            }
        );
        assert!(!t.has_attached_replicas().expect("attached"));
        assert_eq!(
            import_state(&t, "l1").expect("state"),
            ImportState::Finished(outcome.clone())
        );
        assert_eq!(
            finish_import(&t, "l1", Some(20)).expect("again"),
            outcome,
            "a repeat reports it"
        );
        assert!(
            b.append_batch(&batch(1), "late").is_err(),
            "a sealed replica takes nothing"
        );
        assert!(
            open_import_replica(&t, "l1", "n3").is_err(),
            "a finished import takes no replica"
        );
        assert_eq!(rows(&t), 20);
    }

    #[test]
    fn loads_reusing_batch_ids_both_deliver() {
        let (_dir, t) = table();
        for import in ["la", "lb"] {
            begin_import(&t, import, None).expect("begin");
            let r = open_import_replica(&t, import, "n1").expect("replica");
            assert!(r.append_batch(&batch(10), "b").expect("append"));
            r.seal().expect("seal");
            assert_eq!(
                finish_import(&t, import, Some(10)).expect("finish").rows,
                10
            );
        }
        assert_eq!(rows(&t), 20);
    }

    #[test]
    fn finishing_one_load_leaves_another_attached() {
        let (_dir, t) = table();
        begin_import(&t, "la", None).expect("begin");
        begin_import(&t, "lb", None).expect("begin");
        let a = open_import_replica(&t, "la", "n1").expect("replica");
        a.append_batch(&batch(10), "b1").expect("append");
        assert_eq!(finish_import(&t, "lb", None).expect("finish").rows, 0);
        assert!(
            t.has_attached_replicas().expect("attached"),
            "la is still loading"
        );
        assert!(a.append_batch(&batch(10), "b2").expect("la goes on"));
        a.seal().expect("seal");
        assert_eq!(finish_import(&t, "la", None).expect("finish").rows, 20);
        assert!(!t.has_attached_replicas().expect("attached"));
    }

    #[test]
    fn an_unsealed_replica_holds_the_finish_until_sealed_for_it() {
        let (_dir, t) = table();
        begin_import(&t, "l1", None).expect("begin");
        let gone = open_import_replica(&t, "l1", "n1").expect("replica");
        gone.append_batch(&batch(10), "b1").expect("append");
        assert!(finish_import(&t, "l1", None).is_err(), "n1 is not sealed");
        assert!(
            t.has_attached_replicas().expect("attached"),
            "a failed finish detaches nothing"
        );
        seal_absent_replica(&t, "l1", "n1").expect("seal for the gone node");
        assert_eq!(finish_import(&t, "l1", None).expect("finish").rows, 10);
    }

    #[test]
    fn a_sealed_replica_reopens_sealed() {
        let (_dir, t) = table();
        begin_import(&t, "l1", None).expect("begin");
        open_import_replica(&t, "l1", "n1")
            .expect("replica")
            .seal()
            .expect("seal");
        let reopened = open_import_replica(&t, "l1", "n1").expect("reopen");
        assert!(reopened.is_sealed());
        assert!(reopened.append_batch(&batch(1), "b").is_err());
        assert!(finish_import(&t, "nope", None).is_err(), "never begun");
        assert!(begin_import(&t, "bad/id", None).is_err());
    }

    #[test]
    fn a_row_count_mismatch_fails_the_finish_and_keeps_the_load_open() {
        let (_dir, t) = table();
        begin_import(&t, "l1", None).expect("begin");
        let r = open_import_replica(&t, "l1", "n1").expect("replica");
        r.append_batch(&batch(10), "b1").expect("append");
        r.seal().expect("seal");
        assert!(finish_import(&t, "l1", Some(11)).is_err());
        assert_eq!(import_state(&t, "l1").expect("state"), ImportState::Open);
        assert!(t.has_attached_replicas().expect("attached"));
        assert_eq!(finish_import(&t, "l1", Some(10)).expect("finish").rows, 10);
    }

    #[test]
    fn a_finished_load_takes_no_seal_or_graft_and_leaves_nothing_attached() {
        let (_dir, t) = table();
        begin_import(&t, "l1", None).expect("begin");
        let r = open_import_replica(&t, "l1", "n1").expect("replica");
        r.append_batch(&batch(10), "b1").expect("append");
        assert!(matches!(
            seal_absent_replica(&t, "l1", "n2"),
            Err(InfinoError::NotFound(_))
        ));
        r.seal().expect("seal");
        finish_import(&t, "l1", Some(10)).expect("finish");
        assert!(seal_absent_replica(&t, "l1", "n1").is_err());
        assert!(graft_import(&t, "l1").expect("graft").is_empty());
        assert!(!t.has_attached_replicas().expect("attached"));
        let storage = storage_of(&t).expect("storage");
        assert!(
            !t.block_on_query(crate::storage::may_be_attached(storage.as_ref())),
            "the marker goes with the last replica"
        );
        assert!(matches!(
            graft_import(&t, "never"),
            Err(InfinoError::NotFound(_))
        ));
    }

    #[test]
    fn a_replica_sealed_by_another_refuses_appends() {
        let (_dir, t) = table();
        begin_import(&t, "l1", None).expect("begin");
        let r = open_import_replica(&t, "l1", "n1").expect("replica");
        r.append_batch(&batch(10), "b1").expect("append");
        seal_absent_replica(&t, "l1", "n1").expect("sealed for it");
        assert!(matches!(
            r.append_batch(&batch(10), "b2"),
            Err(InfinoError::Conflict(_))
        ));
        assert!(r.is_sealed(), "the append saw the seal");
        assert_eq!(finish_import(&t, "l1", None).expect("finish").rows, 10);
    }

    #[test]
    fn a_replica_created_while_a_finish_runs_is_refused() {
        let (_dir, t) = table();
        begin_import(&t, "l1", None).expect("begin");
        let early = open_import_replica(&t, "l1", "n1").expect("replica");
        // A finish records its claim before it lists the replicas.
        let (_, version) = read_record(&t, "l1").expect("record");
        let claim = write_record(
            &t,
            "l1",
            ImportState::Finishing,
            Writers::Auto,
            version.as_deref(),
        )
        .expect("claim")
        .expect("won");
        assert!(matches!(
            open_import_replica(&t, "l1", "n2"),
            Err(InfinoError::Conflict(_))
        ));
        assert_eq!(
            merge::attached_replicas(&t).expect("attached"),
            vec!["l1.n1".to_owned()],
            "the refused replica left nothing behind"
        );
        assert!(early.append_batch(&batch(10), "b1").expect("append"));
        let reopened = open_import_replica(&t, "l1", "n1").expect("an existing replica reopens");
        reopened.seal().expect("seal");
        // A second finish takes the claim over; the first can no longer write.
        assert_eq!(finish_import(&t, "l1", Some(10)).expect("finish").rows, 10);
        assert!(
            write_record(&t, "l1", ImportState::Open, Writers::Auto, Some(&claim))
                .expect("write")
                .is_none()
        );
        assert!(matches!(
            open_import_replica(&t, "l1", "n2"),
            Err(InfinoError::NotFound(_))
        ));
    }

    #[test]
    fn a_failed_finish_reopens_the_load_for_new_replicas() {
        let (_dir, t) = table();
        begin_import(&t, "l1", None).expect("begin");
        let r = open_import_replica(&t, "l1", "n1").expect("replica");
        r.append_batch(&batch(10), "b1").expect("append");
        r.seal().expect("seal");
        assert!(matches!(
            finish_import(&t, "l1", Some(20)),
            Err(InfinoError::Conflict(_))
        ));
        assert_eq!(import_state(&t, "l1").expect("state"), ImportState::Open);
        let late = open_import_replica(&t, "l1", "n2").expect("a new replica opens");
        late.append_batch(&batch(10), "b2")
            .expect("the missing batch");
        late.seal().expect("seal");
        assert_eq!(finish_import(&t, "l1", Some(20)).expect("finish").rows, 20);
    }

    #[test]
    fn an_append_after_the_finish_detached_is_refused() {
        let (_dir, t) = table();
        begin_import(&t, "l1", None).expect("begin");
        let r = open_import_replica(&t, "l1", "n1").expect("replica");
        r.append_batch(&batch(10), "b1").expect("append");
        seal_absent_replica(&t, "l1", "n1").expect("sealed for it");
        assert_eq!(finish_import(&t, "l1", None).expect("finish").rows, 10);
        assert!(r.append_batch(&batch(10), "b2").is_err());
        assert!(!t.has_attached_replicas().expect("attached"));
        assert_eq!(rows(&t), 10);
    }

    #[test]
    fn open_imports_lists_the_unfinished_ones() {
        let (_dir, t) = table();
        assert!(open_imports(&t).expect("none").is_empty());
        for import in ["la", "lb"] {
            begin_import(&t, import, None).expect("begin");
        }
        finish_import(&t, "la", None).expect("finish");
        assert_eq!(open_imports(&t).expect("open"), vec!["lb".to_owned()]);
    }

    #[test]
    fn writers_are_kept_through_the_import() {
        let (_dir, t) = table();
        let one = Writers::Count(NonZeroU32::MIN);
        begin_import(&t, "l1", Some(one)).expect("begin");
        assert_eq!(import_writers(&t, "l1").expect("writers"), one);
        assert!(matches!(
            begin_import(&t, "l1", Some(Writers::Auto)),
            Err(InfinoError::Conflict(_))
        ));
        assert_eq!(
            begin_import(&t, "l1", None).expect("a batch starting it again"),
            ImportState::Open
        );
        let r = open_import_replica(&t, "l1", "n1").expect("replica");
        r.append_batch(&batch(10), "b1").expect("append");
        r.seal().expect("seal");
        assert!(finish_import(&t, "l1", Some(11)).is_err());
        assert_eq!(
            import_writers(&t, "l1").expect("after a failed finish"),
            one
        );
        finish_import(&t, "l1", Some(10)).expect("finish");
        assert_eq!(import_writers(&t, "l1").expect("after the finish"), one);
        assert!(matches!(
            import_writers(&t, "never"),
            Err(InfinoError::NotFound(_))
        ));
    }

    #[test]
    fn updates_and_deletes_wait_for_open_imports() {
        use datafusion::prelude::{col, lit};
        let (_dir, t) = table();
        t.append(&batch(5)).expect("plain rows");
        begin_import(&t, "l1", None).expect("begin");
        let r = open_import_replica(&t, "l1", "n1").expect("replica");
        r.append_batch(&batch(10), "b1").expect("append");
        let refused = |e: Result<crate::MutationStats, InfinoError>| {
            matches!(e, Err(InfinoError::Conflict(_)))
        };
        assert!(refused(t.delete(col("title").eq(lit("shared x")))));
        assert!(refused(
            t.update(col("title").eq(lit("shared x")), &batch(15))
        ));
        r.seal().expect("seal");
        finish_import(&t, "l1", Some(10)).expect("finish");
        let deleted = t
            .delete(col("title").eq(lit("shared x")))
            .expect("after the finish");
        assert_eq!(deleted.n_tombstoned(), 15, "plain and imported rows alike");
    }

    #[test]
    fn an_abort_keeps_grafted_rows_and_ends_the_import() {
        let (_dir, t) = table();
        begin_import(&t, "l1", None).expect("begin");
        let r = open_import_replica(&t, "l1", "n1").expect("replica");
        r.append_batch(&batch(10), "b1").expect("append");
        graft_import(&t, "l1").expect("graft");
        r.append_batch(&batch(10), "b2").expect("not grafted yet");
        abort_import(&t, "l1").expect("abort");
        abort_import(&t, "l1").expect("again is harmless");
        assert_eq!(import_state(&t, "l1").expect("state"), ImportState::Aborted);
        assert_eq!(rows(&t), 10);
        assert!(!t.has_attached_replicas().expect("attached"));
        assert!(matches!(
            r.append_batch(&batch(1), "b3"),
            Err(InfinoError::NotFound(_))
        ));
        assert!(matches!(
            finish_import(&t, "l1", None),
            Err(InfinoError::NotFound(_))
        ));
        assert!(open_imports(&t).expect("open").is_empty());
    }

    #[test]
    fn writers_cap_the_unsealed_replicas() {
        let (_dir, t) = table();
        begin_import(&t, "l1", Some(Writers::Count(NonZeroU32::MIN))).expect("begin");
        let first = open_import_replica(&t, "l1", "n1").expect("the one writer");
        assert!(matches!(
            open_import_replica(&t, "l1", "n2"),
            Err(InfinoError::Conflict(_))
        ));
        assert_eq!(
            merge::attached_replicas(&t).expect("attached"),
            vec!["l1.n1".to_owned()]
        );
        first.append_batch(&batch(10), "b1").expect("append");
        first.seal().expect("seal");
        let next = open_import_replica(&t, "l1", "n2").expect("a sealed writer frees its place");
        next.append_batch(&batch(10), "b2").expect("append");
        next.seal().expect("seal");
        assert_eq!(finish_import(&t, "l1", Some(20)).expect("finish").rows, 20);
    }

    #[test]
    fn activity_comes_from_the_replicas_commits() {
        let (_dir, t) = table();
        begin_import(&t, "l1", None).expect("begin");
        assert_eq!(import_activity(&t, "l1").expect("none yet"), None);
        let r = open_import_replica(&t, "l1", "n1").expect("replica");
        r.append_batch(&batch(10), "b1").expect("append");
        assert!(import_activity(&t, "l1").expect("activity").is_some());
    }

    #[test]
    fn a_table_holds_a_bounded_number_of_open_imports() {
        let (_dir, t) = table();
        for i in 0..MAX_OPEN_IMPORTS {
            begin_import(&t, &format!("l{i}"), None).expect("begin");
        }
        assert!(matches!(
            begin_import(&t, "one-more", None),
            Err(InfinoError::Conflict(_))
        ));
        begin_import(&t, "l0", None).expect("an open one begins again");
        finish_import(&t, "l0", None).expect("finish");
        begin_import(&t, "one-more", None).expect("room again");
    }

    #[test]
    fn a_finished_import_refuses_another_row_count() {
        let (_dir, t) = table();
        begin_import(&t, "l1", None).expect("begin");
        finish_import(&t, "l1", Some(0)).expect("finish");
        assert!(matches!(
            finish_import(&t, "l1", Some(5)),
            Err(InfinoError::Cardinality(_))
        ));
        assert_eq!(finish_import(&t, "l1", None).expect("again").rows, 0);
    }

    #[test]
    fn a_batch_with_a_new_column_is_refused_on_a_replica() {
        let (_dir, t) = table();
        begin_import(&t, "l1", None).expect("begin");
        let r = open_import_replica(&t, "l1", "n1").expect("replica");
        let wide = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("extra", DataType::LargeUtf8, true),
        ]));
        let cols: Vec<ArrayRef> = vec![
            Arc::new(LargeStringArray::from(vec!["a"; 3])),
            Arc::new(LargeStringArray::from(vec!["b"; 3])),
        ];
        let wide = RecordBatch::try_new(wide, cols).expect("batch");
        assert!(matches!(
            r.append_batch(&wide, "b1"),
            Err(InfinoError::Unsupported(_))
        ));
        r.append_batch(&batch(10), "b2").expect("append");
        graft_import(&t, "l1").expect("grafts go on");
        r.seal().expect("seal");
        assert_eq!(finish_import(&t, "l1", Some(10)).expect("finish").rows, 10);
    }
}
