// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `SupertableWriter` — the single-writer append + commit path.
//!
//! **Naming convention.** `SupertableWriter` is a long-lived
//! append handle — `append×N → commit`, repeated across many
//! commits over its lifetime. Contrast
//! [`crate::superfile::SuperfileBuilder`], which is a single-shot
//! factory consuming `self` to produce one immutable artifact.
//! Each `commit` here internally spawns many superfile builders,
//! one per shard.
//!
//! Acquired via [`Supertable::writer`](super::Supertable::writer);
//! at most one writer is outstanding per supertable at a time
//! (enforced by the inner state's `writer_outstanding` flag, with
//! release on `Drop`). Holds an in-memory buffer of
//! `(scalar_batch, vectors_per_column)` payloads that
//! [`SupertableWriter::commit`] partitions across the writer
//! pool's rayon workers — each worker constructs its own
//! [`SuperfileBuilder`], feeds its slice, and emits one
//! self-contained superfile. All resulting superfiles are published
//! in a single `ArcSwap` of the manifest at the end.
//!
//! ## Flow
//!
//! - `append(batch)` runs schema + null validation via
//!   `vector_split`, pushes a `BufferedBatch` onto the writer's
//!   buffer, and triggers an internal `commit()` if the running
//!   buffer-byte estimate crosses the configured threshold.
//! - `commit()` drains the buffer, partitions across the writer
//!   pool, runs each shard build in parallel, and publishes all
//!   shards as new superfiles in one manifest swap. Idempotent on
//!   an empty buffer (no-op return Ok). The writer slot is
//!   released on `Drop`; callers don't need a separate `finish()`
//!   call.
//!
//! ## Buffer ownership
//!
//! Vectors arrive from the input `RecordBatch` as
//! `FixedSizeListArray` columns; `vector_split` views them as
//! `&[f32]` slices. To keep the buffer ownership clean across
//! `append` calls (each input batch can be dropped by the caller
//! once `append` returns), we Arc-clone the underlying
//! `Float32Array` payloads into the buffer. At commit time we
//! re-derive `&[f32]` slices from the Arc'd arrays for the
//! per-shard `SuperfileBuilder::add_batch` call. No bytes copied;
//! just Arc reference counts.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use arrow::ipc::writer::StreamWriter;
use arrow_array::{
    Array, ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, RecordBatch,
};
use bytes::Bytes;
use chrono::Utc;
use object_store::PutPayload;
use rayon::prelude::*;

use crate::storage::{StorageError, StorageProvider};
use crate::superfile::builder::SuperfileBuilder;
use crate::superfile::format::CRC_BYTES;
use crate::superfile::format::fts::{HEADER_SIZE as FTS_HEADER_SIZE, U64_BYTES, hdr};
use crate::superfile::format::vec::{
    CLUSTER_IDX_ENTRY_BYTES, DIR_ENTRY_SIZE, OUTER_HEADER_SIZE, SUB_HEADER_SIZE, U32_BYTES,
    dir_entry, outer_hdr, sub_hdr,
};
use crate::superfile::format::{footer::read_kv_metadata, kv};
use crate::supertable::manifest::Manifest;
use crate::supertable::manifest::ManifestPartLoader;
use crate::supertable::manifest::commit::get_current_manifest_etag;
use crate::supertable::manifest::commit::{self as commit_mod};
use crate::supertable::manifest::list::{
    FORMAT_VERSION as LIST_FORMAT_VERSION, ManifestList, PartitionStrategy,
};
use crate::supertable::manifest::part::{self as part_mod, PartId};
use crate::supertable::{CommitError as SupertableCommitError, ManifestLoadError};

use super::build::fanout_shards;
use super::error::BuildError;
use super::handle::{Supertable, SupertableInner};
use super::manifest::bloom::BloomBuilder;
use super::manifest::{
    FtsSummary, ScalarStatsTable, SubsectionOffsets, SuperfileEntry, SuperfileUri, VectorSummary,
};
use super::mutations::{
    CommitError, CommitResult, MAX_TARGETS_PER_MUTATION, MutationError, MutationStats,
    PendingDelete, PendingUpdate,
};
use super::options::SupertableOptions;
use super::utils::vector_split::split_vectors;
use super::wal::WalStore;
use super::wal::pipeline::{self, TombstonePhaseOutcome};
use super::wal::state_doc::{
    IdSpan, OpKind, RowId, SCHEMA_VERSION, TombstoneEntry, TombstoneOutcome, WalId, WalState,
    WalStateDoc,
};

/// Single-writer append + commit handle.
///
/// At most one outstanding per supertable. Acquire via
/// [`Supertable::writer`]; uncommitted buffer data is **lost on
/// drop** (no implicit flush) — callers must invoke `commit()`
/// to publish.
pub struct SupertableWriter {
    inner: Arc<SupertableInner>,
    /// Accumulated input from append() calls. The writer (not the
    /// SuperfileBuilder) owns the buffer so commit() can rayon-
    /// shard it across workers, each running its own builder.
    buffer: Vec<BufferedBatch>,
    /// Estimated byte cost of `buffer` so append() can auto-flush
    /// when the buffer crosses the configured threshold.
    buffer_bytes: usize,
    /// Pending update entries, in buffer order. Each is
    /// fully-resolved at `update()` call time (predicate
    /// captured, `_id` range minted, IPC sidecar bytes encoded);
    /// `commit()` drives them through the WAL pipeline in order.
    pending_updates: Vec<PendingUpdateEntry>,
    /// Pending delete entries, in buffer order. Each carries
    /// the call-time resolved `target_ids` + a pre-minted
    /// `wal_id`; `commit()` builds the WAL state doc and drives
    /// the tombstone phase.
    pending_deletes: Vec<PendingDeleteEntry>,
}

/// One buffered update. Resources here are all reserved at the
/// `update()` call so the writer can drop the `RecordBatch`
/// after IPC-encoding it (the `ipc_bytes` are what the WAL
/// sidecar carries).
struct PendingUpdateEntry {
    wal_id: crate::supertable::wal::state_doc::WalId,
    target_ids: Vec<i128>,
    preallocated_superfile_id: uuid::Uuid,
    minted_id_spans: Vec<crate::supertable::wal::state_doc::IdSpan>,
    new_row_count: u32,
    new_row_content_hash: String,
    ipc_bytes: Bytes,
}

/// One buffered delete. Just the call-time resolved target_ids
/// + a pre-minted `wal_id`.
struct PendingDeleteEntry {
    wal_id: crate::supertable::wal::state_doc::WalId,
    target_ids: Vec<i128>,
}

impl std::fmt::Debug for SupertableWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SupertableWriter")
            .field("buffered_batches", &self.buffer.len())
            .field("buffered_bytes", &self.buffer_bytes)
            .field("manifest_id", &self.inner.manifest.load().manifest_id)
            .finish()
    }
}

/// One buffered append-call payload. Vectors stored as
/// `Arc<Float32Array>` so the buffer owns its data outright;
/// per-shard builders re-derive `&[f32]` slices via
/// [`Float32Array::values`] without copying.
struct BufferedBatch {
    scalar: RecordBatch,
    vectors: Vec<Arc<Float32Array>>,
}

/// Row-balanced split of the writer's buffered batches into
/// `n_shards` shard inputs, each shaped as a `Vec<BufferedBatch>`
/// that [`build_one_shard`] can consume directly. The split walks
/// rows across the original buffer in order and emits zero-copy
/// Arrow slices (`RecordBatch::slice` + `Float32Array::slice` —
/// adjust buffer offsets only; underlying memory stays Arc-counted),
/// so no payload bytes are copied even when a shard boundary falls
/// in the middle of a `BufferedBatch`.
///
/// Row imbalance across shards is ≤ 1: with `total_rows = q·n + r`,
/// the first `r` shards get `q+1` rows and the rest get `q`.
///
/// Trailing empty shards (only possible when `total_rows < n_shards`)
/// are dropped before return; callers see exactly the shards that
/// will produce a non-empty superfile.
fn split_buffer_into_row_shards(
    buffer: Vec<BufferedBatch>,
    n_shards: usize,
    vector_dims: &[usize],
) -> Vec<Vec<BufferedBatch>> {
    debug_assert!(n_shards > 0);
    let total_rows: usize = buffer.iter().map(|b| b.scalar.num_rows()).sum();
    if total_rows == 0 {
        return Vec::new();
    }
    let base = total_rows / n_shards;
    let remainder = total_rows % n_shards;
    let target = |i: usize| if i < remainder { base + 1 } else { base };

    let mut shards: Vec<Vec<BufferedBatch>> = (0..n_shards).map(|_| Vec::new()).collect();
    let mut shard_idx = 0usize;
    let mut shard_remaining = target(0);

    for batch in buffer {
        let n_rows = batch.scalar.num_rows();
        if n_rows == 0 {
            continue;
        }
        let mut row_cursor = 0;
        while row_cursor < n_rows {
            // Skip ahead over any zero-target shards (only happens
            // when total_rows < n_shards, leaving trailing shards
            // with target == 0).
            while shard_remaining == 0 && shard_idx + 1 < n_shards {
                shard_idx += 1;
                shard_remaining = target(shard_idx);
            }
            let take = std::cmp::min(shard_remaining, n_rows - row_cursor);
            let scalar = batch.scalar.slice(row_cursor, take);
            let vectors: Vec<Arc<Float32Array>> = batch
                .vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dim = vector_dims[i];
                    Arc::new(v.slice(row_cursor * dim, take * dim))
                })
                .collect();
            shards[shard_idx].push(BufferedBatch { scalar, vectors });
            row_cursor += take;
            shard_remaining -= take;
        }
    }
    shards.retain(|s| !s.is_empty());
    shards
}

/// The public folded `update` / `delete` buffer exactly one mutation
/// before committing, so `CommitResult.outcomes` carries exactly one
/// entry; surface it (or a backend error if, impossibly, none landed).
fn single_outcome(res: CommitResult) -> Result<MutationStats, crate::InfinoError> {
    res.outcomes.into_iter().next().ok_or_else(|| {
        crate::InfinoError::Backend("commit produced no mutation outcome".to_string())
    })
}

impl Supertable {
    /// Append one batch of rows and commit — durable when this returns.
    ///
    /// Folds the buffered writer + commit into a single call: one
    /// `append` == one commit == one sealed superfile, so callers batch
    /// rows per call rather than calling once per row.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_array::{LargeStringArray, RecordBatch};
    /// # use arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, IndexSpec};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # let posts = db.create_table("posts", schema.clone(), IndexSpec::new().fts("body"))?;
    /// let batch = RecordBatch::try_new(
    ///     schema,
    ///     vec![Arc::new(LargeStringArray::from(vec!["hello world"]))],
    /// )?;
    /// posts.append(&batch)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn append(&self, batch: &RecordBatch) -> Result<(), crate::InfinoError> {
        let mut w = self.writer()?;
        w.append(batch)?;
        w.commit()?;
        Ok(())
    }

    /// Replace every row matching `predicate` with `new_rows`, then
    /// commit. `new_rows.num_rows()` must equal the match count.
    /// Durable when this returns.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_array::{LargeStringArray, RecordBatch};
    /// # use arrow_schema::{DataType, Field, Schema};
    /// # use datafusion::prelude::{col, lit};
    /// # use infino::{connect, IndexSpec};
    /// # let dir = tempfile::tempdir()?; // update/delete need durable storage
    /// # let db = connect(dir.path().to_str().expect("utf8 path"))?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # let posts = db.create_table("posts", schema.clone(), IndexSpec::new().fts("body"))?;
    /// # let row = |s: &str| RecordBatch::try_new(
    /// #     schema.clone(), vec![Arc::new(LargeStringArray::from(vec![s]))]).expect("batch");
    /// # posts.append(&row("draft"))?;
    /// let stats = posts.update(col("body").eq(lit("draft")), &row("published"))?;
    /// assert_eq!(stats.matched(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn update(
        &self,
        predicate: datafusion::prelude::Expr,
        new_rows: &RecordBatch,
    ) -> Result<MutationStats, crate::InfinoError> {
        let mut w = self.writer()?;
        w.update(predicate, new_rows.clone())?;
        single_outcome(w.commit()?)
    }

    /// Tombstone every row matching `predicate`, then commit. Durable
    /// when this returns.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_array::{LargeStringArray, RecordBatch};
    /// # use arrow_schema::{DataType, Field, Schema};
    /// # use datafusion::prelude::{col, lit};
    /// # use infino::{connect, IndexSpec};
    /// # let dir = tempfile::tempdir()?; // update/delete need durable storage
    /// # let db = connect(dir.path().to_str().expect("utf8 path"))?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # let posts = db.create_table("posts", schema.clone(), IndexSpec::new().fts("body"))?;
    /// # posts.append(&RecordBatch::try_new(
    /// #     schema, vec![Arc::new(LargeStringArray::from(vec!["spam"]))])?)?;
    /// let stats = posts.delete(col("body").eq(lit("spam")))?;
    /// assert_eq!(stats.n_tombstoned(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn delete(
        &self,
        predicate: datafusion::prelude::Expr,
    ) -> Result<MutationStats, crate::InfinoError> {
        let mut w = self.writer()?;
        w.delete(predicate)?;
        single_outcome(w.commit()?)
    }

    test_visible! {
    /// Acquire the single writer for this supertable.
    ///
    /// Returns [`BuildError::SupertableInUse`] if another
    /// `SupertableWriter` is already outstanding (drop it before
    /// acquiring a new one). Each `Supertable` has exactly one
    /// active writer slot at a time, enforced atomically; when
    /// the writer is dropped, the slot is released and a
    /// subsequent `writer()` call succeeds.
    fn writer(&self) -> Result<SupertableWriter, BuildError> {
        match self.inner().writer_outstanding.compare_exchange(
            false,
            true,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => Ok(SupertableWriter {
                inner: Arc::clone(self.inner()),
                buffer: Vec::new(),
                buffer_bytes: 0,
                pending_updates: Vec::new(),
                pending_deletes: Vec::new(),
            }),
            Err(_) => Err(BuildError::SupertableInUse),
        }
    }
    }
}

impl SupertableWriter {
    /// Number of buffered batches not yet committed. Useful for
    /// tests + diagnostics; not part of the production hot path.
    pub fn buffered_batches(&self) -> usize {
        self.buffer.len()
    }

    /// Estimated bytes of buffered (un-committed) data.
    pub fn buffered_bytes(&self) -> usize {
        self.buffer_bytes
    }

    /// Add one batch to the in-memory buffer. Triggers an
    /// internal `commit()` if the running buffer-byte estimate
    /// crosses the configured threshold (or returns immediately
    /// if `commit_threshold_size_mb == 0`).
    ///
    /// The supplied batch's schema must match
    /// [`SupertableOptions::user_schema`] — i.e., it must NOT
    /// contain the id column. This method injects the id column
    /// unconditionally; the buffered batch's schema therefore
    /// matches [`SupertableOptions::scalar_schema`] with the
    /// id column at position 0.
    pub fn append(&mut self, batch: &RecordBatch) -> Result<(), BuildError> {
        let options = &self.inner.options;

        // Validate + split. Batch schema is user_schema (no id col).
        let (scalar_no_id, _vector_slices) = split_vectors(batch, options)?;

        // Re-derive owned Arc<Float32Array> handles for each
        // vector column. We can't keep the &[f32] slices from
        // split_vectors in the buffer (their lifetime is tied to
        // `batch`, which the caller reclaims after this returns).
        // The Arc<Float32Array> shares the same underlying buffer
        // — no bytes copied.
        let mut vectors = Vec::with_capacity(options.vector_columns.len());
        for vc in &options.vector_columns {
            let col_idx = batch
                .schema()
                .index_of(&vc.column)
                .map_err(|_| BuildError::BatchSchemaMismatch)?;
            let fsl = batch
                .column(col_idx)
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or(BuildError::BatchSchemaMismatch)?;
            let values = fsl.values();
            let f32_arr = values
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or(BuildError::BatchSchemaMismatch)?
                .clone();
            vectors.push(Arc::new(f32_arr));
        }

        // Mint one id per row and prepend the id column. Lock
        // is uncontended in practice (writer-slot exclusivity
        // serializes append per supertable handle); held only
        // long enough to drain N ids into the Vec.
        let n_rows = scalar_no_id.num_rows();
        let mut ids: Vec<i128> = Vec::with_capacity(n_rows);
        {
            let generator = self
                .inner
                .id_generator
                .lock()
                .expect("id_generator mutex poisoned");
            for _ in 0..n_rows {
                ids.push(generator.next_id());
            }
        }
        let id_array = Decimal128Array::from(ids)
            .with_precision_and_scale(
                crate::supertable::options::DECIMAL128_PRECISION,
                crate::supertable::options::DECIMAL128_SCALE,
            )
            .expect(
                "invariant: precision 38 + scale 0 always valid \
                 for any i128 payload",
            );
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(scalar_no_id.num_columns() + 1);
        columns.push(Arc::new(id_array));
        columns.extend(scalar_no_id.columns().iter().cloned());
        let scalar = RecordBatch::try_new(options.scalar_schema(), columns)
            .map_err(|_| BuildError::BatchSchemaMismatch)?;

        // Estimate byte cost: Arrow scalar columns + f32 vector
        // payload. RecordBatch::get_array_memory_size accounts
        // for buffer allocations (rough but good enough for
        // threshold gating).
        let bytes = scalar.get_array_memory_size()
            + vectors
                .iter()
                .map(|v| v.len() * std::mem::size_of::<f32>())
                .sum::<usize>();

        self.buffer.push(BufferedBatch { scalar, vectors });
        self.buffer_bytes += bytes;

        // Auto-flush if over threshold.
        let threshold = (options.commit_threshold_size_mb as usize)
            .saturating_mul(1024)
            .saturating_mul(1024);
        if threshold > 0 && self.buffer_bytes >= threshold {
            self.commit_appends_internal()?;
        }

        Ok(())
    }

    /// Buffer a delete operation. Every row whose `_id`
    /// matches `predicate` at call time will be tombstoned by
    /// the next [`commit`] call.
    ///
    /// `predicate` is evaluated **immediately** against the
    /// current manifest snapshot (the same ArcSwap-backed view
    /// queries use). The resolved `_id` set is captured on the
    /// writer's pending-deletes buffer; rows that newly match
    /// `predicate` between this call and `commit()` (because of
    /// an interleaving append on this or another writer) are
    /// NOT tombstoned — only the captured `_id` list is.
    ///
    /// **Does NOT make the change durable.** Buffered deletes
    /// are lost on writer drop until the next successful
    /// `commit()`. Symmetric with buffered `append()`s.
    ///
    /// [`commit`]: SupertableWriter::commit
    pub fn delete(
        &mut self,
        predicate: datafusion::prelude::Expr,
    ) -> Result<PendingDelete, MutationError> {
        // Pre-flight: storage must be attached for the WAL
        // pipeline to drive this op at commit time.
        let _ = self
            .inner
            .options
            .storage
            .as_ref()
            .ok_or(MutationError::NoStorageAttached)?;

        // Resolve the predicate against the current manifest
        // snapshot. NOTE: the writer's pending-appends buffer
        // is NOT flushed here. Captured-at-call semantics mean
        // the delete sees the manifest as it stood at this
        // call's instant; rows the caller appended in the same
        // writer session are not yet in the manifest.
        let supertable = Supertable::from_inner(Arc::clone(&self.inner));
        let target_ids = supertable
            .reader()
            .scan_ids_matching(predicate)
            .map_err(MutationError::PredicateEval)?;
        let matched = target_ids.len();
        if matched > MAX_TARGETS_PER_MUTATION {
            return Err(MutationError::MatchCountExceedsCap {
                matched,
                cap: MAX_TARGETS_PER_MUTATION,
            });
        }

        // Pre-mint the wal_id so we can surface it at commit
        // time even on a partial-failure path (the recovery
        // sweep on a fresh open completes any WAL whose id
        // already landed in storage).
        let wal_id_value = self
            .inner
            .id_generator
            .lock()
            .expect("id_generator mutex poisoned")
            .next_id();

        self.pending_deletes.push(PendingDeleteEntry {
            wal_id: WalId(wal_id_value),
            target_ids,
        });
        Ok(PendingDelete { matched })
    }

    /// Buffer a 1:1-cardinality update: at the next [`commit`],
    /// `new_rows` is appended as the replacement payload AND
    /// every row whose `_id` matched `predicate` at call entry
    /// is tombstoned.
    ///
    /// `predicate` is evaluated **immediately** against the
    /// current manifest snapshot; the resolved `_id` set + the
    /// IPC-encoded payload + a pre-reserved `_id` range + a
    /// preallocated superfile UUID are captured on the writer's
    /// pending-updates buffer. `commit()` drives each entry
    /// through its WAL pipeline (append → tombstone).
    ///
    /// **Cardinality:** `new_rows.num_rows()` MUST equal the
    /// predicate's resolved match count. Mismatch returns
    /// `CardinalityMismatch` and nothing is buffered.
    ///
    /// **Does NOT make the change durable.** Symmetric with
    /// buffered `append()` / `delete()`s.
    ///
    /// [`commit`]: SupertableWriter::commit
    pub fn update(
        &mut self,
        predicate: datafusion::prelude::Expr,
        new_rows: RecordBatch,
    ) -> Result<PendingUpdate, MutationError> {
        // Pre-flight: storage attached.
        let _ = self
            .inner
            .options
            .storage
            .as_ref()
            .ok_or(MutationError::NoStorageAttached)?;

        // Schema check (no _id column on the user-facing path).
        if new_rows.schema().as_ref() != self.inner.options.schema.as_ref() {
            return Err(MutationError::SchemaMismatch(format!(
                "expected {:?}, got {:?}",
                self.inner.options.schema.fields(),
                new_rows.schema().fields()
            )));
        }

        // Resolve predicate against the manifest snapshot.
        // Captured-at-call semantics: appends still in this
        // writer's buffer don't count toward the match set.
        let supertable = Supertable::from_inner(Arc::clone(&self.inner));
        let target_ids = supertable
            .reader()
            .scan_ids_matching(predicate)
            .map_err(MutationError::PredicateEval)?;
        let matched = target_ids.len();
        if matched > MAX_TARGETS_PER_MUTATION {
            return Err(MutationError::MatchCountExceedsCap {
                matched,
                cap: MAX_TARGETS_PER_MUTATION,
            });
        }
        let new_row_count = new_rows.num_rows();
        if matched != new_row_count {
            return Err(MutationError::CardinalityMismatch {
                matched,
                new_rows: new_row_count,
            });
        }

        // Cardinality 0 is a structurally-impossible update —
        // the WAL pipeline needs `preallocated_superfile_id`
        // and at least one minted id span. We mint a wal_id so
        // the caller's `PendingUpdate` is comparable to the
        // non-zero shape, but skip buffering. The commit's
        // `CommitResult.outcomes` will reflect `matched: 0` if
        // the caller routes through the buffer instead.
        if matched == 0 {
            return Ok(PendingUpdate { matched: 0 });
        }

        // Reserve _id range + preallocate superfile id + mint
        // wal_id under one lock so the relative ordering is
        // deterministic and visible to any recovery replay.
        let (wal_id_value, minted_id_spans, preallocated_superfile_id) = {
            let idgen = self.inner.id_generator.lock().expect("idgen mutex");
            let spans = idgen
                .reserve_range(matched as u32)
                .into_iter()
                .map(|(first, last)| IdSpan {
                    first: RowId(first),
                    last: RowId(last),
                })
                .collect::<Vec<_>>();
            let wal_id_value = idgen.next_id();
            let preallocated = uuid::Uuid::new_v4();
            (wal_id_value, spans, preallocated)
        };

        // IPC-encode the new_rows batch + blake3. Doing this at
        // call time (rather than commit time) means the caller
        // can drop the `RecordBatch` immediately — the buffer
        // owns the bytes from here on.
        let ipc_bytes = encode_record_batch_ipc(&new_rows).map_err(|e| {
            MutationError::Storage(crate::storage::StorageError::Permanent {
                uri: "ipc encode".into(),
                source: Box::new(std::io::Error::other(e)),
            })
        })?;
        let content_hash = blake3::hash(&ipc_bytes).to_hex().to_string();

        self.pending_updates.push(PendingUpdateEntry {
            wal_id: WalId(wal_id_value),
            target_ids,
            preallocated_superfile_id,
            minted_id_spans,
            new_row_count: matched as u32,
            new_row_content_hash: content_hash,
            ipc_bytes,
        });
        Ok(PendingUpdate { matched })
    }

    /// Flush every buffered operation atomically (from the
    /// caller's perspective):
    ///
    /// 1. Pending appends → built into superfiles, manifest
    ///    swap committed.
    /// 2. Pending updates, in buffer order → per-op WAL
    ///    pipeline (append phase + tombstone phase).
    /// 3. Pending deletes, in buffer order → per-op WAL
    ///    pipeline (tombstone phase only).
    ///
    /// On success returns a [`CommitResult`] with one
    /// [`MutationStats`] per buffered mutation (in buffer
    /// order). On a mid-flush mutation failure surfaces
    /// [`CommitError::PartialCommit`] listing the WALs that DID
    /// land durably; the remaining buffered ops stay on the
    /// writer for retry, and the recovery sweep on the next
    /// supertable open completes the listed WALs if this
    /// process dies before retrying.
    ///
    /// [`CommitResult`]: crate::supertable::mutations::CommitResult
    /// [`MutationStats`]: crate::supertable::mutations::MutationStats
    /// [`CommitError::PartialCommit`]: crate::supertable::mutations::CommitError::PartialCommit
    pub fn commit(&mut self) -> Result<CommitResult, CommitError> {
        // Step 1: flush appends. A failure here is atomic —
        // the buffer is preserved and no mutation WAL has
        // landed yet.
        if !self.buffer.is_empty() {
            self.commit_appends_internal()
                .map_err(CommitError::AppendFlush)?;
        }

        let total_mutations = self.pending_updates.len() + self.pending_deletes.len();
        let mut committed_wal_ids: Vec<crate::supertable::wal::state_doc::WalId> =
            Vec::with_capacity(total_mutations);
        let mut outcomes: Vec<MutationStats> = Vec::with_capacity(total_mutations);

        // Step 2: drive pending updates in buffer order. On
        // mid-loop failure, the failed entry is dropped (its
        // WAL may already be on storage; recovery sweep
        // completes it on the next open) and the unattempted
        // entries stay on `self.pending_updates` for retry.
        let mut updates_to_run = std::mem::take(&mut self.pending_updates);
        let mut update_cursor = 0usize;
        while update_cursor < updates_to_run.len() {
            let entry = &updates_to_run[update_cursor];
            match self.drive_one_update(entry) {
                Ok(outcome) => {
                    committed_wal_ids.push(outcome.wal_id);
                    outcomes.push(outcome);
                    update_cursor += 1;
                }
                Err(cause) => {
                    // Drop the failed entry + put the rest
                    // back on the buffer.
                    let remaining: Vec<PendingUpdateEntry> =
                        updates_to_run.split_off(update_cursor + 1);
                    self.pending_updates = remaining;
                    // Don't lose the not-yet-attempted deletes
                    // either — they stay where they were on
                    // self.pending_deletes (we hadn't taken
                    // them yet).
                    return Err(CommitError::PartialCommit {
                        committed_wal_ids,
                        committed: outcomes.len(),
                        total: total_mutations,
                        cause: Box::new(cause),
                    });
                }
            }
        }

        // Step 3: drive pending deletes in buffer order.
        let mut deletes_to_run = std::mem::take(&mut self.pending_deletes);
        let mut delete_cursor = 0usize;
        while delete_cursor < deletes_to_run.len() {
            let entry = &deletes_to_run[delete_cursor];
            match self.drive_one_delete(entry) {
                Ok(outcome) => {
                    committed_wal_ids.push(outcome.wal_id);
                    outcomes.push(outcome);
                    delete_cursor += 1;
                }
                Err(cause) => {
                    let remaining: Vec<PendingDeleteEntry> =
                        deletes_to_run.split_off(delete_cursor + 1);
                    self.pending_deletes = remaining;
                    return Err(CommitError::PartialCommit {
                        committed_wal_ids,
                        committed: outcomes.len(),
                        total: total_mutations,
                        cause: Box::new(cause),
                    });
                }
            }
        }

        Ok(CommitResult {
            wal_ids: committed_wal_ids,
            outcomes,
        })
    }

    /// Drive one pending update entry through its full WAL
    /// pipeline. Returns the per-op outcome on success.
    fn drive_one_update(&self, entry: &PendingUpdateEntry) -> Result<MutationStats, MutationError> {
        let storage = self
            .inner
            .options
            .storage
            .as_ref()
            .ok_or(MutationError::NoStorageAttached)?
            .clone();

        let wal_doc = WalStateDoc {
            wal_id: entry.wal_id,
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Update,
            state: WalState::Intent,
            created_at: Utc::now(),
            lease: None,
            predicate_repr: "writer.update()".into(),
            target_ids: entry.target_ids.iter().map(|&v| RowId(v)).collect(),
            new_row_count: Some(entry.new_row_count),
            new_row_content_hash: Some(entry.new_row_content_hash.clone()),
            preallocated_superfile_id: Some(entry.preallocated_superfile_id),
            minted_id_spans: entry.minted_id_spans.clone(),
            tombstone_progress: entry
                .target_ids
                .iter()
                .map(|&v| TombstoneEntry {
                    target_id: RowId(v),
                    outcome: TombstoneOutcome::Pending,
                    tombstoned_in_superfile: None,
                })
                .collect(),
        };

        let wal_store = WalStore::new(Arc::clone(&storage));
        let supertable = Supertable::from_inner(Arc::clone(&self.inner));
        let wal_id = entry.wal_id;
        let ipc_bytes = entry.ipc_bytes.clone();
        let drive = async move {
            wal_store
                .put_arrow(wal_id, ipc_bytes)
                .await
                .map_err(MutationError::WalStore)?;
            let etag = wal_store
                .create(&wal_doc)
                .await
                .map_err(MutationError::WalStore)?;
            let (_outcome, doc_after_append, etag_after_append) =
                pipeline::run_append_phase(&supertable, &wal_store, &wal_doc, &etag).await?;
            let (outcome, _post, _post_etag) = pipeline::run_tombstone_phase(
                &supertable,
                &wal_store,
                &doc_after_append,
                &etag_after_append,
            )
            .await?;
            let (n_t, n_nf) = match outcome {
                TombstonePhaseOutcome::Applied {
                    n_tombstoned,
                    n_not_found,
                }
                | TombstonePhaseOutcome::AlreadyComplete {
                    n_tombstoned,
                    n_not_found,
                } => (n_tombstoned, n_not_found),
            };
            // Best-effort cleanup of the WAL artifacts.
            let _ = wal_store.delete_arrow(wal_id).await;
            let _ = wal_store.delete_state(wal_id).await;
            Ok::<_, MutationError>((n_t, n_nf))
        };
        let (n_tombstoned, n_not_found) = match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(drive))?,
            Err(_) => self.inner.query_runtime().block_on(drive)?,
        };
        Ok(MutationStats {
            wal_id: entry.wal_id,
            matched: entry.target_ids.len(),
            n_tombstoned,
            n_not_found,
        })
    }

    /// Drive one pending delete entry through its tombstone
    /// phase. Returns the per-op outcome on success.
    fn drive_one_delete(&self, entry: &PendingDeleteEntry) -> Result<MutationStats, MutationError> {
        let storage = self
            .inner
            .options
            .storage
            .as_ref()
            .ok_or(MutationError::NoStorageAttached)?
            .clone();

        let wal_doc = WalStateDoc {
            wal_id: entry.wal_id,
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Delete,
            state: WalState::Intent,
            created_at: Utc::now(),
            lease: None,
            predicate_repr: "writer.delete()".into(),
            target_ids: entry.target_ids.iter().map(|&v| RowId(v)).collect(),
            new_row_count: None,
            new_row_content_hash: None,
            preallocated_superfile_id: None,
            minted_id_spans: Vec::new(),
            tombstone_progress: entry
                .target_ids
                .iter()
                .map(|&v| TombstoneEntry {
                    target_id: RowId(v),
                    outcome: TombstoneOutcome::Pending,
                    tombstoned_in_superfile: None,
                })
                .collect(),
        };

        let wal_store = WalStore::new(Arc::clone(&storage));
        let supertable = Supertable::from_inner(Arc::clone(&self.inner));
        let wal_id = entry.wal_id;
        let drive = async move {
            let etag = wal_store
                .create(&wal_doc)
                .await
                .map_err(MutationError::WalStore)?;
            let (outcome, _post, _post_etag) =
                pipeline::run_tombstone_phase(&supertable, &wal_store, &wal_doc, &etag).await?;
            let (n_t, n_nf) = match outcome {
                TombstonePhaseOutcome::Applied {
                    n_tombstoned,
                    n_not_found,
                }
                | TombstonePhaseOutcome::AlreadyComplete {
                    n_tombstoned,
                    n_not_found,
                } => (n_tombstoned, n_not_found),
            };
            let _ = wal_store.delete_state(wal_id).await;
            Ok::<_, MutationError>((n_t, n_nf))
        };
        let (n_tombstoned, n_not_found) = match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(drive))?,
            Err(_) => self.inner.query_runtime().block_on(drive)?,
        };
        Ok(MutationStats {
            wal_id: entry.wal_id,
            matched: entry.target_ids.len(),
            n_tombstoned,
            n_not_found,
        })
    }

    /// Drain the pending-appends buffer and publish all shard
    /// outputs in one manifest swap. Internal-only; the public
    /// [`SupertableWriter::commit`] calls this first before
    /// driving pending mutations.
    ///
    /// Rows are balanced evenly across shards regardless of the
    /// caller's `append()` cadence — many small appends followed by
    /// one `commit` produce the same shard layout as one large append.
    fn commit_appends_internal(&mut self) -> Result<(), BuildError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let buffer = std::mem::take(&mut self.buffer);
        self.buffer_bytes = 0;

        let total_rows: usize = buffer.iter().map(|b| b.scalar.num_rows()).sum();
        if total_rows == 0 {
            return Ok(());
        }

        let writer_pool = Arc::clone(&self.inner.options.writer_pool);
        let n_threads = writer_pool.current_num_threads().max(1);
        let n_shards = n_threads.min(total_rows);

        let vector_dims: Vec<usize> = self
            .inner
            .options
            .vector_columns
            .iter()
            .map(|vc| vc.dim)
            .collect();
        let shards = split_buffer_into_row_shards(buffer, n_shards, &vector_dims);

        // One shared fan-out for every modality — FTS, vector, combined.
        // No per-modality concurrency cap: rayon's work-stealing balances
        // the inter-shard fan-out against each shard's intra-shard
        // parallelism on the writer pool (see `build::fanout_shards`).
        let outputs: Vec<ShardOutput> = fanout_shards(&writer_pool, &shards, |slice| {
            build_one_shard(slice.as_slice(), &self.inner.options)
        })?;

        publish_superfiles(&self.inner, outputs)?;
        Ok(())
    }
}

impl Drop for SupertableWriter {
    fn drop(&mut self) {
        // Release the writer slot. Uncommitted buffer is
        // intentionally lost — callers must invoke commit()
        // explicitly to publish.
        self.inner
            .writer_outstanding
            .store(false, Ordering::Release);
    }
}

/// Output of one rayon shard worker.
///
/// FTS + vector summaries are derived in [`publish_superfiles`] from
/// the cached `SuperfileReader` (cheaper than re-walking buffered
/// batches). `scalar_stats` is computed here, before the buffer is
/// dropped, since the post-store `SuperfileReader` only exposes
/// parquet row groups — Arrow batch min/max would require a full
/// re-decode through DataFusion or parquet-rs's stats reader.
pub struct ShardOutput {
    bytes: Bytes,
    n_docs: u64,
    /// `id_min` / `id_max`: only meaningful when `n_docs > 0`.
    /// For a 0-doc shard (empty slice — shouldn't happen given
    /// chunk sizing, but defensive), both are 0. Stored as
    /// `i128` to carry the 128-bit Snowflake-shaped ids
    /// produced by [`crate::supertable::utils::idgen::IdGenerator`].
    id_min: i128,
    id_max: i128,
    /// Per-scalar-column min/max for skip pruning. Computed from
    /// the shard's `BufferedBatch` slice via Arrow per-type
    /// aggregate kernels; types whose ordering isn't well-defined
    /// (FixedSizeList, struct, etc.) are absent and treated as
    /// "can't prune" by the skip planner.
    scalar_stats: ScalarStatsTable,
}

impl ShardOutput {
    pub fn new_with_params(
        bytes: Bytes,
        n_docs: u64,
        id_min: i128,
        id_max: i128,
        scalar_stats: ScalarStatsTable,
    ) -> Self {
        Self {
            bytes,
            n_docs,
            id_min,
            id_max,
            scalar_stats,
        }
    }
}

/// Build one superfile from one slice of buffered batches. Runs on
/// a rayon worker thread inside the writer pool's `install`.
fn build_one_shard(
    slice: &[BufferedBatch],
    options: &SupertableOptions,
) -> Result<ShardOutput, BuildError> {
    let mut builder = SuperfileBuilder::new(options.builder_options())?;

    let scalar_schema = options.scalar_schema();
    // The supertable always prepends the id column at index 0
    // via `SupertableOptions::scalar_schema`, so we can skip
    // the schema lookup here.
    let id_idx = 0;

    let mut id_min = i128::MAX;
    let mut id_max = i128::MIN;
    let mut n_docs: u64 = 0;

    for buffered in slice {
        let id_col = buffered
            .scalar
            .column(id_idx)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .ok_or_else(|| {
                BuildError::IdColumnWrongType(
                    options.id_column.clone(),
                    "<id column not Decimal128 at runtime>".to_string(),
                )
            })?;
        for i in 0..id_col.len() {
            let v = id_col.value(i);
            id_min = id_min.min(v);
            id_max = id_max.max(v);
        }
        n_docs += id_col.len() as u64;

        // Float32Array::values() returns &ScalarBuffer<f32>;
        // ScalarBuffer derefs to &[f32], so AsRef does the slice
        // view without a copy.
        let vector_slices: Vec<&[f32]> = buffered
            .vectors
            .iter()
            .map(|fa| fa.values().as_ref())
            .collect();
        builder.add_batch(&buffered.scalar, &vector_slices)?;
    }

    // Compute per-scalar-column min/max BEFORE moving `slice`'s
    // batches into the builder via `finish`. We pass references —
    // `from_batches` doesn't take ownership.
    let scalar_batches: Vec<&RecordBatch> = slice.iter().map(|b| &b.scalar).collect();
    let scalar_stats = ScalarStatsTable::from_batches(&scalar_schema, &scalar_batches);

    let bytes = Bytes::from(builder.finish()?);

    let (id_min, id_max) = if n_docs == 0 {
        (0, 0)
    } else {
        (id_min, id_max)
    };

    Ok(ShardOutput {
        bytes,
        n_docs,
        id_min,
        id_max,
        scalar_stats,
    })
}

/// Pull the superfile's `(total_size, vec_off/len, fts_off/len)`
/// out of the freshly-written parquet KV metadata so the manifest
/// can carry it forward as a [`SubsectionOffsets`]. Returns `None`
/// if the bytes don't parse — that path falls back to the
/// 2-RTT cold open shape rather than failing the publish.
pub(crate) fn build_subsection_offsets(bytes: &Bytes) -> Option<SubsectionOffsets> {
    let kvs = read_kv_metadata(bytes).ok()?;
    let get = |k: &str| -> Option<u64> { kvs.get(k).and_then(|s| s.parse::<u64>().ok()) };
    let vec = match (get(kv::VEC_OFFSET), get(kv::VEC_LENGTH)) {
        (Some(o), Some(l)) if l > 0 => Some((o, l)),
        _ => None,
    };
    let fts = match (get(kv::FTS_OFFSET), get(kv::FTS_LENGTH)) {
        (Some(o), Some(l)) if l > 0 => Some((o, l)),
        _ => None,
    };
    let total_size = bytes.len() as u64;
    let vec_open_ranges = vec
        .and_then(|(off, len)| vector_open_ranges(bytes, off, len))
        .unwrap_or_default();
    let fts_open_ranges = fts
        .and_then(|(off, len)| fts_open_ranges(bytes, off, len))
        .unwrap_or_default();

    // capture the open-time batch bytes (parquet
    // footer tail + vector open ranges + FTS open ranges) so the
    // reader can resolve a superfile's open metadata straight from
    // the manifest part, issuing zero per-superfile open GETs.
    let open_blob = build_open_blob(bytes, total_size, &vec_open_ranges, &fts_open_ranges);

    Some(SubsectionOffsets {
        total_size,
        vec,
        fts,
        vec_open_ranges,
        fts_open_ranges,
        open_blob,
    })
}

/// Slice the bytes for the superfile's open-time batch out of the
/// freshly-written superfile so the manifest can carry them
/// inline. Mirrors the cold-fetch open batch in
/// `DiskCacheStore::cold_fetch_lazy_with_hints`: the parquet
/// footer tail (matching the 64 KiB speculation length) plus each
/// vector / FTS open range. Returns `(absolute_offset, bytes)`
/// tuples; an empty `Vec` disables the inline-open fast path for
/// this superfile.
fn build_open_blob(
    bytes: &Bytes,
    total_size: u64,
    vec_open_ranges: &[(u64, u64)],
    fts_open_ranges: &[(u64, u64)],
) -> Vec<(u64, Vec<u8>)> {
    // Must match `cold_fetch_lazy_with_hints`'s parquet tail
    // speculation length so the overlay covers `source.tail()`.
    const PARQUET_TAIL_SPEC: u64 = 64 * 1024;
    let mut blob: Vec<(u64, Vec<u8>)> =
        Vec::with_capacity(1 + vec_open_ranges.len() + fts_open_ranges.len());

    let parquet_tail_len = PARQUET_TAIL_SPEC.min(total_size);
    let parquet_tail_start = total_size.saturating_sub(parquet_tail_len);
    let slice = |off: u64, len: u64| -> Option<Vec<u8>> {
        let start = off as usize;
        let end = start.checked_add(len as usize)?;
        bytes.get(start..end).map(|s| s.to_vec())
    };
    if parquet_tail_len > 0 {
        match slice(parquet_tail_start, parquet_tail_len) {
            Some(b) => blob.push((parquet_tail_start, b)),
            None => return Vec::new(),
        }
    }
    for &(off, len) in vec_open_ranges.iter().chain(fts_open_ranges.iter()) {
        match slice(off, len) {
            Some(b) => blob.push((off, b)),
            // A range we can't satisfy means the capture is
            // inconsistent; disable the fast path rather than ship
            // a partial overlay.
            None => return Vec::new(),
        }
    }
    blob
}

fn vector_open_ranges(bytes: &Bytes, off: u64, len: u64) -> Option<Vec<(u64, u64)>> {
    let start = off as usize;
    let end = start.checked_add(len as usize)?;
    let blob = bytes.get(start..end)?;
    if blob.len() < OUTER_HEADER_SIZE + CRC_BYTES {
        return None;
    }
    let n_columns =
        read_u32_le(blob.get(outer_hdr::N_COLUMNS_OFF..outer_hdr::N_COLUMNS_OFF + U32_BYTES)?)
            as usize;
    let dir_offset =
        read_u64_le(blob.get(outer_hdr::DIR_OFFSET_OFF..outer_hdr::DIR_OFFSET_OFF + U64_BYTES)?)
            as usize;
    let dir_size = n_columns.checked_mul(DIR_ENTRY_SIZE)?;
    let dir_end = dir_offset.checked_add(dir_size)?.checked_add(CRC_BYTES)?;
    let dir = blob.get(dir_offset..dir_offset + dir_size)?;

    let mut ranges = vec![(off + dir_offset as u64, (dir_size + CRC_BYTES) as u64)];
    ranges.push((off, OUTER_HEADER_SIZE as u64));
    for i in 0..n_columns {
        let entry = i * DIR_ENTRY_SIZE;
        let subsection_off = read_u64_le(dir.get(
            entry + dir_entry::SUBSECTION_OFF_OFF
                ..entry + dir_entry::SUBSECTION_OFF_OFF + U64_BYTES,
        )?) as usize;
        let subsection_len = read_u64_le(dir.get(
            entry + dir_entry::SUBSECTION_LEN_OFF
                ..entry + dir_entry::SUBSECTION_LEN_OFF + U64_BYTES,
        )?) as usize;
        let codec_meta_off = read_u32_le(dir.get(
            entry + dir_entry::CODEC_META_OFF_OFF
                ..entry + dir_entry::CODEC_META_OFF_OFF + U32_BYTES,
        )?) as usize;
        let codec_meta_size = read_u32_le(dir.get(
            entry + dir_entry::CODEC_META_SIZE_OFF
                ..entry + dir_entry::CODEC_META_SIZE_OFF + U32_BYTES,
        )?) as usize;
        if subsection_off.checked_add(SUB_HEADER_SIZE)? > blob.len()
            || subsection_off.checked_add(subsection_len)? > blob.len()
        {
            return None;
        }
        ranges.push((off + subsection_off as u64, SUB_HEADER_SIZE as u64));
        let sub = blob.get(subsection_off..subsection_off + subsection_len)?;
        let centroids_off = read_u64_le(
            sub.get(sub_hdr::CENTROIDS_OFF_OFF..sub_hdr::CENTROIDS_OFF_OFF + U64_BYTES)?,
        ) as usize;
        let cluster_idx_off = read_u64_le(
            sub.get(sub_hdr::CLUSTER_IDX_OFF_OFF..sub_hdr::CLUSTER_IDX_OFF_OFF + U64_BYTES)?,
        ) as usize;
        let cluster_idx_end = cluster_idx_off.checked_add(
            CLUSTER_IDX_ENTRY_BYTES
                * read_u32_le(dir.get(
                    entry + dir_entry::N_CENT_OFF..entry + dir_entry::N_CENT_OFF + U32_BYTES,
                )?) as usize,
        )?;
        if centroids_off < SUB_HEADER_SIZE || cluster_idx_end > subsection_len {
            return None;
        }
        // Stage only [cluster_idx .. cluster_idx_end]. The fp32 centroids that
        // precede it are read solely by the rare fallback per-segment `nprobe`
        // path (segments lacking a manifest cluster summary), which range-GETs
        // them from the superfile on demand — they remain on disk. The hot
        // cluster-probe path reads only `cluster_idx`, so keeping centroids out
        // of the open_blob makes the manifest-inline open footprint independent
        // of `n_cent` (centroids are ~99% of it at high `n_cent`).
        ranges.push((
            off + subsection_off as u64 + cluster_idx_off as u64,
            (cluster_idx_end - cluster_idx_off) as u64,
        ));
        if codec_meta_size > 0 {
            let meta_end = codec_meta_off.checked_add(codec_meta_size)?;
            if meta_end > subsection_len {
                return None;
            }
        }
    }
    if dir_end > blob.len() {
        return None;
    }
    Some(merge_ranges(ranges))
}

fn fts_open_ranges(bytes: &Bytes, off: u64, len: u64) -> Option<Vec<(u64, u64)>> {
    let start = off as usize;
    let end = start.checked_add(len as usize)?;
    let blob = bytes.get(start..end)?;
    if blob.len() < FTS_HEADER_SIZE {
        return None;
    }
    let postings_offset =
        read_u64_le(blob.get(hdr::POSTINGS_OFFSET_OFF..hdr::POSTINGS_OFFSET_OFF + U64_BYTES)?)
            as usize;
    let doc_lengths_offset =
        read_u64_le(blob.get(hdr::DOC_LENGTHS_DIR_OFF..hdr::DOC_LENGTHS_DIR_OFF + U64_BYTES)?)
            as usize;
    if postings_offset > blob.len()
        || doc_lengths_offset > blob.len()
        || postings_offset > doc_lengths_offset
    {
        return None;
    }
    Some(merge_ranges(vec![
        (off, postings_offset as u64),
        (
            off + doc_lengths_offset as u64,
            (blob.len() - doc_lengths_offset) as u64,
        ),
    ]))
}

fn merge_ranges(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.retain(|&(_, len)| len > 0);
    ranges.sort_unstable_by_key(|&(off, _)| off);
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (off, len) in ranges {
        let end = off + len;
        if let Some((last_off, last_len)) = merged.last_mut() {
            let last_end = *last_off + *last_len;
            if off <= last_end {
                *last_len = (*last_len).max(end - *last_off);
                continue;
            }
        }
        merged.push((off, len));
    }
    merged
}

fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("u32 slice length"))
}

fn read_u64_le(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("u64 slice length"))
}

/// Per-shard publish artifacts produced in parallel before the
/// serial manifest swap. One entry per non-empty shard.
pub(crate) struct PreparedSuperfile {
    pub(crate) entry: Arc<SuperfileEntry>,
    /// Bytes destined for the in-memory superfile store. `Some` on
    /// the in-memory-only path and the storage-without-cache
    /// path; `None` on the cache-attached path (the disk cache
    /// hydrates lazily from storage).
    pub(crate) bytes_for_store: Option<(SuperfileUri, Bytes)>,
    pub(crate) bytes_for_storage: Option<(SuperfileUri, Bytes)>,
    pub(crate) bytes_for_cache: Option<(SuperfileUri, Bytes)>,
}

impl PreparedSuperfile {
    /// Open a `SuperfileReader` directly on this superfile's bytes.
    /// Returns `None` if no bytes are held (cache-attached path with
    /// no prepopulation — bytes went to storage only).
    #[cfg(test)]
    pub(crate) fn open_reader(
        &self,
    ) -> Option<Result<crate::superfile::SuperfileReader, crate::superfile::ReadError>> {
        let bytes = self
            .bytes_for_store
            .as_ref()
            .or(self.bytes_for_storage.as_ref())
            .or(self.bytes_for_cache.as_ref())
            .map(|(_, b)| b.clone())?;
        Some(crate::superfile::SuperfileReader::open(bytes))
    }
}

/// Build the per-shard publish artifacts: open a `SuperfileReader`
/// on the shard bytes, derive FTS + vector summaries, and decide
/// the bytes-disposition triplet. Pure per-shard work — no shared
/// mutable state, safe to run in parallel across shards.
pub(super) fn prepare_superfile(
    inner: &SupertableInner,
    shard: ShardOutput,
) -> Result<Option<PreparedSuperfile>, BuildError> {
    if shard.n_docs == 0 {
        return Ok(None);
    }

    let uri = SuperfileUri::new_v4();

    let bytes_for_storage = inner.options.storage.is_some().then(|| shard.bytes.clone());
    let cache_attached = inner.options.disk_cache.is_some() && inner.options.storage.is_some();
    // `bytes_for_store` (in-memory tier) is gated only on cache attachment —
    // a cache-attached producer keeps superfile bytes out of the unbounded
    // in-memory store regardless of whether we pre-populate the disk cache.
    let bytes_for_store = (!cache_attached).then(|| shard.bytes.clone());
    // Pre-populating the disk cache is opt-out: a write-only producer that
    // drops the cache right after ingest skips this wasted warm-fill.
    let bytes_for_cache =
        (cache_attached && inner.options.prepopulate_cache_on_commit).then(|| shard.bytes.clone());

    // Open the reader directly on shard bytes (not via the
    // in-memory `SuperfileReaderCache`). This lets the cache-attached
    // path skip the in-memory tier entirely — the bytes can go
    // straight to object storage without a RAM detour, which is
    // what removes the 100GB OOM trap (the in-memory cache doesn't
    // evict, so a long-running writer with cache + storage would
    // otherwise accumulate every superfile's bytes in RAM forever).
    let reader = crate::superfile::SuperfileReader::open_with(
        shard.bytes.clone(),
        inner.options.superfile_open_options(),
    )
    .map_err(|e| BuildError::Store(format!("opening superfile for summary: {e}")))?;

    let mut fts_summary: HashMap<String, FtsSummary> = HashMap::new();
    if let Some(fts_reader) = reader.fts() {
        for fc in &inner.options.fts_columns {
            let terms = fts_reader
                .iter_column_terms(&fc.column)
                .expect("FST bytes valid: superfile just built");
            let n_terms_distinct = terms.len() as u32;
            let (min_term, max_term) = match (terms.first(), terms.last()) {
                (Some(min), Some(max)) => (min.clone(), max.clone()),
                _ => (Vec::new(), Vec::new()),
            };
            let mut bloom_builder = BloomBuilder::new();
            for term in &terms {
                bloom_builder.insert(term);
            }
            fts_summary.insert(
                fc.column.clone(),
                FtsSummary {
                    term_bloom: bloom_builder.finish(),
                    n_terms_distinct,
                    term_range: (min_term, max_term),
                },
            );
        }
    }

    let mut vector_summary: HashMap<String, VectorSummary> = HashMap::new();
    if let Some(vec_reader) = reader.vec() {
        for vc in &inner.options.vector_columns {
            if let Some((centroid, radius)) = vec_reader.summary(&vc.column) {
                // Stage the per-cluster centroids (Sq8) into the
                // manifest so a query can rank this superfile's clusters
                // globally without opening the superfile.
                let clusters = vec_reader
                    .cluster_centroids(&vc.column)
                    .map(|(n_cent, dim, fp32, counts)| {
                        crate::supertable::manifest::ClusterCentroids::from_fp32(
                            n_cent, dim, &fp32, counts,
                        )
                    })
                    .unwrap_or_default();
                vector_summary.insert(
                    vc.column.clone(),
                    VectorSummary {
                        centroid,
                        radius,
                        clusters,
                    },
                );
            }
        }
    }

    // capture `(total_size, vec_off/len, fts_off/len)`
    // from the freshly-written bytes' parquet KV metadata. Caching
    // these on the manifest lets `DiskCacheStore::reader_with_hints`
    // fire the parquet-footer, vector, and FTS subsection GETs in
    // parallel on cold open (1 RTT instead of 2 sequential).
    let subsection_offsets = build_subsection_offsets(&shard.bytes);

    let entry = Arc::new(SuperfileEntry {
        superfile_id: uuid::Uuid::new_v4(),
        uri,
        n_docs: shard.n_docs,
        id_min: shard.id_min,
        id_max: shard.id_max,
        scalar_stats: shard.scalar_stats,
        fts_summary,
        vector_summary,
        // Partition assignment populated by the per-shard
        // `PartitionStrategy` wiring elsewhere; superfiles
        // emitted here remain unpartitioned (default).
        partition_key: Vec::new(),
        partition_hint: None,
        subsection_offsets,
    });

    Ok(Some(PreparedSuperfile {
        entry,
        bytes_for_store: bytes_for_store.map(|b| (uri, b)),
        bytes_for_storage: bytes_for_storage.map(|b| (uri, b)),
        bytes_for_cache: bytes_for_cache.map(|b| (uri, b)),
    }))
}

/// Insert each shard's bytes into the superfile store, derive
/// per-superfile summaries from the stored `SuperfileReader`, and
/// publish all entries in one `ArcSwap` of the manifest.
///
/// Per-shard work (reader open, FTS bloom build, vector summary,
/// `SuperfileEntry` construction) runs in parallel across the
/// writer pool — for an FTS supertable the bloom build alone is
/// O(n_terms_distinct) per FTS column per shard, which at 10M
/// docs × 4 superfiles is the dominant cost. Manifest swap +
/// storage write-through stay serial after the join.
fn publish_superfiles(
    inner: &SupertableInner,
    outputs: Vec<ShardOutput>,
) -> Result<(), BuildError> {
    let prepared: Vec<PreparedSuperfile> = inner.options.writer_pool.install(|| {
        outputs
            .into_par_iter()
            .filter_map(|shard| prepare_superfile(inner, shard).transpose())
            .collect::<Result<Vec<_>, _>>()
    })?;

    let mut new_entries: Vec<Arc<SuperfileEntry>> = Vec::with_capacity(prepared.len());
    let mut pending_storage_writes: Vec<(SuperfileUri, Bytes)> = Vec::new();
    let mut pending_cache_inserts: Vec<(SuperfileUri, Bytes)> = Vec::new();

    for p in prepared {
        if let Some((uri, b)) = p.bytes_for_store {
            inner
                .options
                .store
                .insert(uri, b)
                .map_err(|e| BuildError::Store(e.to_string()))?;
        }
        if let Some(t) = p.bytes_for_storage {
            pending_storage_writes.push(t);
        }
        if let Some(t) = p.bytes_for_cache {
            pending_cache_inserts.push(t);
        }
        new_entries.push(p.entry);
    }

    if new_entries.is_empty() {
        return Ok(());
    }

    let old = inner.manifest.load();

    // Storage write-through: when storage is attached, persist
    // each superfile's bytes + the new manifest (parts + list +
    // pointer) before swapping the in-memory state. If any
    // storage operation fails the commit fails as a whole —
    // the in-memory manifest is **not** updated, so callers
    // see a clean rollback to the prior state.
    if let Some(storage) = inner.options.storage.as_ref().cloned() {
        // Drop the read-locked snapshot before entering
        // persist_commit — the OCC retry loop will reload
        // inner.manifest each iteration to incorporate any
        // commits from other writers that won the race.
        drop(old);
        let new_manifest = persist_commit(inner, storage, new_entries, pending_storage_writes)
            .map_err(|e| BuildError::Store(e.to_string()))?;
        inner.manifest.store(Arc::new(new_manifest));

        // Warm the cache with the superfiles we just persisted.
        // Skips the cold-fetch round-trip on the producer's
        // next query against its own superfiles (each superfile
        // otherwise costs one storage HEAD + parallel
        // range-GETs to refetch what we already have in
        // hand). Best-effort: a cache insert failure (e.g.,
        // budget exhausted) is logged via the error path but
        // doesn't fail the commit — the superfile is durably
        // in storage, and a subsequent query will cold-fetch
        // it as if pre-population hadn't been attempted.
        if !pending_cache_inserts.is_empty()
            && let Some(cache) = inner.options.disk_cache.as_ref().cloned()
        {
            warm_cache_after_commit(inner, &cache, pending_cache_inserts);
        }

        // Best-effort memory-budget enforcement. When commits
        // pre-populate the cache (above), sustained writers grow
        // the working set linearly, so a post-commit check +
        // sweep keeps the working set under the configured
        // budget. Pages re-fault from disk on next access if
        // needed; the cache entries themselves are unaffected.
        // Runs regardless of pre-population so an externally
        // warmed cache is still bounded.
        if let (Some(cache), Some(budget)) = (
            inner.options.disk_cache.as_ref(),
            inner.options.memory_budget_bytes,
        ) {
            cache.sweep_for_budget(budget);
        }
        return Ok(());
    }

    let new = old.with_appended(new_entries);
    inner.manifest.store(Arc::new(new));

    Ok(())
}

// OCC retry budget — read from
// `SupertableOptions::max_commit_retries` (default 10) so
// callers with high contention can raise it. The
// `attempt + 1 < retries` check + the final
// `WriteContentionExhausted` return keep the loop bounded
// regardless of the configured value.

/// Jittered exponential backoff between OCC retries.
///
/// Base 10 ms, doubling per attempt, capped at 1 s, with ±30%
/// jitter to break up lockstep retries from racing writers.
/// Jitter source is the low bits of the system's nanosecond
/// clock — no `rand` dep needed.
fn backoff_delay(attempt: u32) -> std::time::Duration {
    const BASE_MS: u64 = 10;
    const CAP_MS: u64 = 1000;
    // Cap the doubling exponent so the pre-cap delay plateaus instead
    // of overflowing the shift on a high attempt count.
    const MAX_SHIFT: u32 = 6;
    // Jitter is a uniform percentage in `-JITTER_RANGE_PCT..=+JITTER_RANGE_PCT`,
    // drawn from the clock's low nanosecond bits. `JITTER_MODULUS`
    // is `2 × JITTER_RANGE_PCT + 1` so the modulo spans the full range.
    const JITTER_RANGE_PCT: i64 = 30;
    const JITTER_MODULUS: u64 = 61;
    const PERCENT_DIVISOR: i64 = 100;
    let exp = BASE_MS.saturating_mul(1u64 << attempt.min(MAX_SHIFT));
    let capped = exp.min(CAP_MS);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    let jitter_pct = (nanos % JITTER_MODULUS) as i64 - JITTER_RANGE_PCT;
    let adjusted = ((capped as i64) + (capped as i64 * jitter_pct / PERCENT_DIVISOR)).max(1) as u64;
    std::time::Duration::from_millis(adjusted)
}

/// Storage write-through with OCC retry. Persist the new
/// superfiles + manifest to storage, returning the new
/// in-memory `Manifest` with the fresh `ManifestList` +
/// `ManifestPartLoader` installed.
///
/// **OCC retry semantics.** On each iteration:
///  1. Reload `inner.manifest` to incorporate any commit a
///     racing writer published since our last attempt.
///  2. Derive `new_superfile_list = old.superfile_list.with_appended(new_entries.clone())`.
///  3. Try `try_commit_attempt` (write superfiles → write part +
///     list → conditional pointer PUT).
///  4. On `WriteContentionExhausted` with retries left: refresh
///     `inner.manifest` from storage (inheriting unchanged
///     parts via content-addressed Arc::clone), sleep with
///     jittered backoff, loop.
///  5. After `opts.max_commit_retries` exhausted: surface
///     `CommitError::WriteContentionExhausted` to the caller.
///
/// **Idempotency across retries.** Superfile URIs are UUID v4 —
/// statically random, so a retry uses the same URIs as the
/// prior attempt. The superfile-bytes PUT swallows
/// `PreconditionFailed` (URI already exists with bit-identical
/// content from our prior attempt). Manifest parts are
/// content-addressed; identical content yields identical URIs
/// and the part-write path already swallows
/// `PreconditionFailed`. Only the pointer PUT must win the
/// CAS; everything below it is idempotent.
///
/// When no real partitioning is configured, all post-commit
/// superfiles go into one `ManifestPart` with a fresh `PartId`.
/// With a real `PartitionStrategy`, `try_commit_attempt` runs
/// the per-partition part-reuse path described on that fn.
pub(in crate::supertable) fn persist_commit(
    inner: &SupertableInner,
    storage: Arc<dyn crate::storage::StorageProvider>,
    new_entries: Vec<Arc<SuperfileEntry>>,
    mut pending_storage_writes: Vec<(SuperfileUri, Bytes)>,
) -> Result<crate::supertable::Manifest, crate::supertable::CommitError> {
    let storage_async = Arc::clone(&storage);
    let opts = Arc::clone(&inner.options);

    // The writer's commit path is sync but the persistence
    // layer is async. Two cases:
    //
    // - **Sync caller** (no ambient tokio runtime): drive
    //   the future on the supertable's owned `query_runtime`
    //   (lazy-init the first time we hit this branch).
    // - **Async caller** (inside a tokio runtime): use the
    //   ambient runtime via `Handle::current().block_on`
    //   wrapped in `block_in_place`. This avoids creating
    //   (and later dropping) a second owned runtime — which
    //   would otherwise panic at Drop with "cannot drop a
    //   runtime in a context where blocking is not allowed".
    //   Requires the ambient runtime to be `multi_thread`.
    let max_retries = opts.max_commit_retries.max(1);
    let drive = async move {
        let mut last_err: Option<crate::supertable::CommitError> = None;
        for attempt in 0..max_retries {
            // Reload `inner.manifest` each iteration so a
            // racing writer's commit (visible via
            // `refresh_inner_state_async` below from a prior
            // iteration) feeds into our successor's
            // `new_superfile_list`.
            let old = inner.manifest.load_full();
            let new_superfile_list = old.superfile_list.with_appended(new_entries.clone());
            let pending_writes = &mut pending_storage_writes;

            match try_commit_attempt(
                Arc::clone(&storage_async),
                Arc::clone(&opts),
                Arc::clone(&old),
                &new_entries,
                &[],
                new_superfile_list.manifest_id,
                pending_writes,
            )
            .await
            {
                Ok(new_list) => {
                    return Ok::<_, crate::supertable::CommitError>((new_list, new_superfile_list));
                }
                Err(crate::supertable::CommitError::WriteContentionExhausted)
                    if attempt + 1 < max_retries =>
                {
                    // Lost the pointer CAS (or a sub-write
                    // CAS, translated to the same variant).
                    // Refresh local state to see the winner's
                    // commit, sleep with jittered backoff,
                    // retry.
                    refresh_inner_state_async(inner, &storage_async).await?;
                    last_err = Some(crate::supertable::CommitError::WriteContentionExhausted);
                    tokio::time::sleep(backoff_delay(attempt)).await;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap_or(crate::supertable::CommitError::WriteContentionExhausted))
    };

    let (new_list, new_superfile_list) = match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            // Ambient tokio runtime present — use it. Don't
            // touch `inner.query_runtime()` so we don't risk
            // dropping our owned runtime from within
            // another's worker context.
            tokio::task::block_in_place(|| handle.block_on(drive))?
        }
        Err(_) => {
            // Sync caller; lazy-init the supertable's
            // owned runtime.
            inner.query_runtime().block_on(drive)?
        }
    };

    // Build the new in-memory Manifest with the persisted
    // list + a fresh ManifestPartLoader installed.
    let loader = Arc::new(ManifestPartLoader::new(Arc::clone(&storage), &new_list));
    Ok(Manifest {
        superfile_list: new_superfile_list,
        list: Some(new_list),
        parts: dashmap::DashMap::new(),
        loader: Some(loader),
    })
}

// Writes the superfile list to storage. Performs the side-effect of modifying pending_storage_writes
// to remove successfully written entries.
// Swallow `PreconditionFailed` per-PUT: on a retry after a
// lost pointer-CAS, the same URI was already written by
// our prior attempt with bit-identical bytes (superfile URIs
// are UUID v4 — collision rate 2^-122). A "URI exists"
// hit here means our own prior attempt; treat as success
// so the retry path is fully idempotent.
//
// Size-gated dispatch: superfiles ≥
// `put_multipart_threshold_bytes` route through
// `put_multipart` (S3 multipart upload, in-place
// streaming on LocalFS) instead of a single `put_atomic`
// PUT. Smaller superfiles stay on the single-PUT path —
// multipart has per-request overhead that isn't worth
// the parallelism below the threshold. The default
// threshold (100 MiB) matches the S3 SDK's standard
// cutoff.
pub async fn write_superfile_list(
    storage: &Arc<dyn StorageProvider>,
    opts: &Arc<SupertableOptions>,
    pending_storage_writes: &mut Vec<(SuperfileUri, Bytes)>,
) -> Result<(), SupertableCommitError> {
    let multipart_threshold = opts.put_multipart_threshold_bytes;
    let put_futs = pending_storage_writes
        .iter_mut()
        .enumerate()
        .map(|(i, (uri, bytes))| {
            let storage = Arc::clone(storage);
            async move {
                let path = superfile_storage_path(uri);
                let result = if (bytes.len() as u64) >= multipart_threshold {
                    put_superfile_multipart(storage.as_ref(), &path, bytes.clone()).await
                } else {
                    // Superfile writes don't chain CAS, so the
                    // returned etag isn't needed here.
                    storage.put_atomic(&path, bytes.clone()).await.map(|_| ())
                };
                match result {
                    Ok(()) => Ok(i),
                    Err(StorageError::PreconditionFailed { .. }) => Ok(i),
                    Err(e) => Err(SupertableCommitError::from(e)),
                }
            }
        });

    let mut err = None;
    let mut successful_writes_idx = Vec::with_capacity(put_futs.len());

    for r in futures::future::join_all(put_futs).await.into_iter().rev() {
        match r {
            Ok(i) => successful_writes_idx.push(i),
            Err(e) => err = Some(e),
        }
    }

    for idx in successful_writes_idx {
        pending_storage_writes.remove(idx);
    }

    if let Some(e) = err {
        return Err(e);
    }

    Ok(())
}

/// One attempt at the commit sequence: write superfile bytes
/// → group new entries by partition → rewrite the latest part
/// per touched partition (preserving untouched parts' URIs)
/// → conditional pointer PUT. The retry loop in
/// `persist_commit` wraps this to handle contention.
///
/// **Partition-aware path.** Each commit's new superfiles are
/// routed by `assign_partition` into per-partition groups.
/// For each touched partition, the writer finds the latest
/// existing part (if any), rebuilds it with the union of its
/// existing superfiles + the new ones, and emits a new
/// `ManifestListEntry` that replaces the prior one (same
/// `partition_key`, new `part_id` + content hash). Untouched
/// partitions' list entries carry over verbatim — no
/// re-encode, no PUT. A cold partition (no prior entry) gets
/// a fresh part with just the new superfiles. The result: a
/// single-partition commit rewrites exactly one part
/// regardless of how many other partitions exist — the
/// load-bearing property the part-reuse optimization relies
/// on.
pub(crate) async fn try_commit_attempt(
    storage: Arc<dyn crate::storage::StorageProvider>,
    opts: Arc<SupertableOptions>,
    old: Arc<crate::supertable::Manifest>,
    new_entries: &[Arc<SuperfileEntry>],
    entries_to_remove: &[Arc<SuperfileEntry>],
    new_manifest_id: u64,
    pending_storage_writes: &mut Vec<(SuperfileUri, Bytes)>,
) -> Result<ManifestList, SupertableCommitError> {
    // 1. Write each new superfile's bytes to storage in parallel.
    write_superfile_list(&storage, &opts, pending_storage_writes).await?;

    // 2. Rebalance the manifest for the commit.
    let (out_list_entries, parts_to_write) =
        commit_mod::rebalance_for_commit(&opts, &old, new_entries, entries_to_remove).await?;

    // 3. Build the new manifest list. The options_hash
    //    digest covers (schema, id_column, fts/vector
    //    column declarations, partition strategy);
    //    Supertable::open validates the caller's options
    //    against this so a schema mismatch surfaces as a
    //    typed error rather than a downstream decode
    //    failure.
    let strategy: PartitionStrategy = old
        .list
        .as_ref()
        .map(|l| l.partition_strategy.clone())
        .unwrap_or_else(|| opts.effective_partition_strategy());
    let opts_hash =
        crate::supertable::manifest::options_hash::compute_options_hash(opts.as_ref(), &strategy);
    let new_list = ManifestList {
        format_version: LIST_FORMAT_VERSION.into(),
        manifest_id: new_manifest_id,
        options_hash: opts_hash,
        schema: Vec::new(),
        id_column: opts.id_column.clone(),
        fts_columns: opts
            .fts_columns
            .iter()
            .map(|f| crate::supertable::manifest::list::FtsColumnInfo {
                column: f.column.clone(),
            })
            .collect(),
        vector_columns: opts
            .vector_columns
            .iter()
            .map(|v| crate::supertable::manifest::list::VectorColumnInfo {
                column: v.column.clone(),
                dim: v.dim,
                n_cent: v.n_cent,
                rot_seed: v.rot_seed,
                metric: format!("{:?}", v.metric).to_lowercase(),
            })
            .collect(),
        partition_strategy: strategy,
        parts: out_list_entries,
    };

    // 4. Read the prior pointer's etag for the CAS. Fresh
    //    supertable → no pointer yet → None etag (initial
    //    commit).
    let prev_etag = get_current_manifest_etag(&storage, old).await?;

    // 5. Parallel-issue (touched parts) + list PUTs, then
    //    conditional pointer PUT (the visibility barrier).
    //    Untouched parts are NOT re-PUT — their URIs (and
    //    content-hashes) are unchanged in the new list.
    let encoded_refs: Vec<&[u8]> = parts_to_write
        .iter()
        .map(|ep| ep.encoded.as_slice())
        .collect();
    commit_mod::commit_manifest(
        storage.as_ref(),
        prev_etag.as_deref(),
        &new_list,
        &encoded_refs,
    )
    .await?;
    // Silence the unused-import warning when no path uses
    // `PartId` / `part_mod` directly (helpers consume them
    // from inside `build_part_and_entry`).
    let _ = std::marker::PhantomData::<(PartId, part_mod::ContentHash)>;

    Ok(new_list)
}

/// Re-read the manifest pointer from storage, load any newer
/// manifest list, inherit unchanged parts from the current
/// in-memory `Manifest` via content-addressed `Arc::clone`,
/// eager-fetch newly-referenced parts, and `ArcSwap` the
/// refreshed `Manifest` into `inner.manifest`.
///
/// Called from the OCC retry loop between attempts so the next
/// iteration's `inner.manifest.load_full()` sees the winning
/// writer's state — `with_appended` then chains our pending
/// superfiles onto theirs at the new monotonic `manifest_id`.
///
/// Mirrors the logic in [`Supertable::refresh`] but operates
/// on `&SupertableInner` so it can be called from inside the
/// writer's commit path without holding a `Supertable` handle.
async fn refresh_inner_state_async(
    inner: &SupertableInner,
    storage: &Arc<dyn crate::storage::StorageProvider>,
) -> Result<(), crate::supertable::CommitError> {
    let current = inner.manifest.load_full();
    let manifest = match Manifest::load(Some(current), storage.clone(), None).await {
        Ok(manifest) => manifest,
        Err(ManifestLoadError::PointerNotFound) => return Ok(()),
        Err(ManifestLoadError::AlreadyLoaded) => return Ok(()),
        Err(err) => return Err(crate::supertable::CommitError::ManifestLoadError(err)),
    };
    inner.manifest.store(manifest);
    Ok(())
}

/// Storage path for a superfile's bytes. Lives under `data/`
/// alongside the `_supertable/` manifest hierarchy.
/// IPC-encode a `RecordBatch` to a byte buffer. Mirrors the
/// shape the WAL's arrow sidecar carries: an
/// `arrow_ipc::writer::StreamWriter` writes one batch followed
/// by a finish marker. The recovery / append-phase reader
/// decodes the same way.
fn encode_record_batch_ipc(batch: &arrow_array::RecordBatch) -> Result<Bytes, String> {
    let mut out: Vec<u8> = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut out, &batch.schema())
            .map_err(|e| format!("ipc writer init: {e}"))?;
        writer.write(batch).map_err(|e| format!("ipc write: {e}"))?;
        writer.finish().map_err(|e| format!("ipc finish: {e}"))?;
    }
    Ok(Bytes::from(out))
}

fn superfile_storage_path(uri: &SuperfileUri) -> String {
    uri.storage_path()
}

/// Multipart-upload variant of the writer's per-superfile put.
/// Routes through [`crate::storage::StorageProvider::put_multipart`]
/// for superfiles large enough that a single PUT is wasteful
/// (slow on a backend stall, high RSS during the put).
///
/// Idempotency: superfile URIs are UUID v4, so the only "URI
/// exists" hit on retry comes from our own prior attempt
/// with bit-identical bytes. Head-first lets us short-circuit
/// that case before re-running the multipart dance. The
/// single-PUT path achieves the same effect by returning
/// `PreconditionFailed`, which the call-site swallows;
/// multipart's `complete()` doesn't carry a precondition, so
/// we need to detect "already there" explicitly.
///
/// Part size: 8 MiB — comfortably above S3's 5-MiB minimum
/// and a clean fit for the cold-fetch coordinator's default
/// 16-MiB chunk reads on the way back out. Parts are pushed
/// in declaration order; the parts run concurrently inside
/// `object_store` after their futures are polled.
async fn put_superfile_multipart(
    storage: &dyn crate::storage::StorageProvider,
    path: &str,
    bytes: Bytes,
) -> Result<(), crate::storage::StorageError> {
    const PART_BYTES: usize = 8 * (1 << 20);

    // Same-bytes retry skip. Failures other than NotFound
    // propagate so we don't paper over a degraded backend.
    match storage.head(path).await {
        Ok(_) => return Err(StorageError::PreconditionFailed { uri: path.into() }),
        Err(StorageError::NotFound { .. }) => {}
        Err(e) => return Err(e),
    }

    let mut upload = storage.put_multipart(path).await?;
    let total = bytes.len();
    let mut parts: Vec<object_store::UploadPart> = Vec::with_capacity(total / PART_BYTES + 1);
    let mut offset = 0;
    while offset < total {
        let end = std::cmp::min(offset + PART_BYTES, total);
        let chunk = bytes.slice(offset..end);
        parts.push(upload.put_part(PutPayload::from_bytes(chunk)));
        offset = end;
    }
    // Drive part-uploads concurrently. `try_join_all` cancels
    // remaining parts if one fails — semantically equivalent to
    // abandoning the upload, with `abort()` below as cleanup.
    if let Err(e) = futures::future::try_join_all(parts).await {
        // Best-effort abort; ignore failure (the upload may
        // already be in a terminal state, or the backend may
        // have lost the multipart-upload ID).
        let _ = upload.abort().await;
        return Err(StorageError::Permanent {
            uri: path.into(),
            source: Box::new(e),
        });
    }
    if let Err(e) = upload.complete().await {
        let _ = upload.abort().await;
        return Err(StorageError::Permanent {
            uri: path.into(),
            source: Box::new(e),
        });
    }
    Ok(())
}

/// Drive `DiskCacheStore::insert_warm` for each
/// just-published superfile via the same sync→async bridge
/// the rest of the writer uses (`block_in_place +
/// Handle::block_on` when an ambient runtime is present;
/// `inner.query_runtime()` otherwise).
///
/// Failures are swallowed with an `eprintln!` log line —
/// the superfiles are already durable in storage and the
/// manifest commit has succeeded, so the cache miss
/// becomes a "warm load fails → next query cold-fetches"
/// degradation, not a correctness break.
fn warm_cache_after_commit(
    inner: &SupertableInner,
    cache: &Arc<crate::supertable::reader_cache::DiskCacheStore>,
    pending: Vec<(SuperfileUri, Bytes)>,
) {
    let cache = Arc::clone(cache);
    let drive = async move {
        for (uri, bytes) in pending {
            if let Err(e) = cache.insert_warm(&uri, bytes).await {
                tracing::warn!(
                    "supertable: warm cache pre-population failed for {}: {} \
                     (superfile is durable in storage; first query will cold-fetch)",
                    uri.0,
                    e
                );
            }
        }
    };
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            tokio::task::block_in_place(|| handle.block_on(drive));
        }
        Err(_) => {
            inner.query_runtime().block_on(drive);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::{FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use figment::Figment;
    use figment::providers::{Format, Yaml};
    use rayon::ThreadPoolBuilder;

    use crate::superfile::builder::FtsConfig;
    use crate::superfile::builder::VectorConfig;
    use crate::superfile::fts::reader::BoolMode;
    use crate::superfile::vector::distance::Metric;
    use crate::supertable::SupertableOptions;
    use crate::supertable::handle::Supertable;
    use crate::supertable::storage::LocalFsStorageProvider;
    use crate::test_helpers::default_tokenizer as tok;
    use tempfile::TempDir;

    fn schema_id_title() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]))
    }

    fn fixed_list_f32(dim: usize) -> DataType {
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        )
    }

    fn schema_id_title_emb(dim: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", fixed_list_f32(dim), false),
        ]))
    }

    fn options_id_title() -> SupertableOptions {
        SupertableOptions::new(
            schema_id_title(),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(tok()),
        )
        .expect("valid options")
    }

    /// Force a single-threaded writer pool for deterministic
    /// shard counts in tests.
    fn options_id_title_serial() -> SupertableOptions {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("build pool"),
        );
        options_id_title().with_writer_pool(pool)
    }

    /// Build a writer pool with N threads.
    fn writer_pool_with(n: usize) -> Arc<rayon::ThreadPool> {
        Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("build pool"),
        )
    }

    fn build_simple_batch(_start: u64, n: usize) -> RecordBatch {
        // The supertable injects `_id` at append time; the
        // user-facing batch carries only the user columns.
        let titles =
            LargeStringArray::from((0..n).map(|i| format!("doc {i} alpha")).collect::<Vec<_>>());
        RecordBatch::try_new(schema_id_title(), vec![Arc::new(titles)]).expect("build batch")
    }

    // ---- writer slot exclusion ---------------------------------------

    #[test]
    fn writer_slot_is_exclusive() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let _w = st.writer().expect("first writer");
        let err = st.writer().expect_err("second writer should fail");
        assert!(matches!(err, BuildError::SupertableInUse));
    }

    #[test]
    fn writer_slot_releases_on_drop() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        {
            let _w = st.writer().expect("first writer");
            // dropped at scope end
        }
        // Slot now free.
        let _w2 = st.writer().expect("second writer after drop");
    }

    // ---- single-writer end-to-end (serial pool) ----------------------

    #[test]
    fn append_then_commit_publishes_one_superfile() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 4)).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        assert_eq!(r.manifest_id(), 1);
        assert_eq!(r.n_superfiles(), 1);
        assert_eq!(r.n_docs_total(), 4);
    }

    #[test]
    fn commit_with_empty_buffer_is_noop() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        w.commit().expect("commit-empty");
        assert_eq!(st.manifest_id(), 0, "no manifest swap on empty commit");
        assert_eq!(st.reader().n_superfiles(), 0);
    }

    #[tokio::test]
    async fn superfile_is_queryable_via_store() {
        // The published superfile's bytes are in the store; we
        // can fetch a SuperfileReader and run bm25_search on it.

        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 4)).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let superfile = &r.manifest().superfiles[0];
        let store = &st.options().store;
        let sf_reader = store.reader(&superfile.uri).expect("reader");
        let hits = sf_reader
            .bm25_hits_async("title", "alpha", 10, BoolMode::Or)
            .await
            .expect("bm25");
        // All 4 docs contain "alpha"; should all be returned.
        assert_eq!(hits.len(), 4);
    }

    // ---- id_min / id_max + n_docs ------------------------------------

    #[test]
    fn superfile_entry_records_id_range_and_n_docs() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(100, 3)).expect("a");
        w.append(&build_simple_batch(50, 2)).expect("b");
        w.commit().expect("commit");

        let r = st.reader();
        let seg = &r.manifest().superfiles[0];
        assert_eq!(seg.n_docs, 5);
        // _id values are auto-injected via the supertable's
        // monotonic generator. We don't know the exact values
        // (timestamp-prefixed); we just assert that min < max
        // and both are positive (high bit 0).
        assert!(seg.id_min > 0);
        assert!(seg.id_max > seg.id_min, "id_max should exceed id_min");
    }

    // ---- FTS summary --------------------------------------------------

    #[test]
    fn superfile_entry_carries_fts_summary() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 4)).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let seg = &r.manifest().superfiles[0];
        let fts = seg
            .fts_summary
            .get("title")
            .expect("title FTS summary present");

        // Each doc's title is "doc <i> alpha"; tokenized with
        // ASCII-lower, distinct terms include "doc", "alpha",
        // and digits 0-3. The FST will dedupe; n_terms_distinct
        // is at least 3 (doc, alpha, plus some digit tokens).
        assert!(
            fts.n_terms_distinct >= 3,
            "expected ≥ 3 distinct terms, got {}",
            fts.n_terms_distinct,
        );
        // Bloom should report present for inserted terms.
        assert!(fts.term_bloom.contains(b"alpha"));
        assert!(fts.term_bloom.contains(b"doc"));
        // Lex range should be non-empty and consistent.
        assert!(!fts.term_range.0.is_empty());
        assert!(!fts.term_range.1.is_empty());
        assert!(
            fts.term_range.0 <= fts.term_range.1,
            "min_term <= max_term invariant",
        );
    }

    // ---- vector summary ----------------------------------------------

    fn build_vector_batch(_start: u64, n: usize, dim: usize) -> RecordBatch {
        let titles = LargeStringArray::from((0..n).map(|i| format!("doc {i}")).collect::<Vec<_>>());
        let mut flat = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                flat.push(((i + j) as f32) / 100.0);
            }
        }
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Float32Array::from(flat);
        let fsl = FixedSizeListArray::try_new(item_field, dim as i32, Arc::new(values), None)
            .expect("FSL");
        RecordBatch::try_new(
            schema_id_title_emb(dim),
            vec![Arc::new(titles), Arc::new(fsl)],
        )
        .expect("batch")
    }

    fn options_with_vector(dim: usize) -> SupertableOptions {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("build pool"),
        );
        SupertableOptions::new(
            schema_id_title_emb(dim),
            vec![],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: crate::superfile::vector::rerank_codec::RerankCodec::Fp32,
            }],
            None,
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    #[test]
    fn superfile_entry_carries_vector_summary() {
        let dim = 16;
        let st = Supertable::create(options_with_vector(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        // Need at least n_cent docs so kmeans has data to cluster.
        w.append(&build_vector_batch(0, 8, dim)).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let seg = &r.manifest().superfiles[0];
        let vs = seg
            .vector_summary
            .get("emb")
            .expect("emb vector summary present");
        assert_eq!(vs.centroid.len(), dim);
        assert!(vs.radius >= 0.0);
        // Per-cluster centroids are staged into the manifest for
        // cross-superfile global cluster selection.
        assert!(
            !vs.clusters.is_empty(),
            "cluster centroids must be populated"
        );
        assert_eq!(vs.clusters.dim as usize, dim);
        assert!(vs.clusters.n_cent >= 1);
        assert_eq!(vs.clusters.counts.len(), vs.clusters.n_cent as usize);
        assert_eq!(vs.clusters.mins.len(), vs.clusters.n_cent as usize);
        assert_eq!(vs.clusters.scales.len(), vs.clusters.n_cent as usize);
        assert_eq!(vs.clusters.codes.len(), vs.clusters.n_cent as usize * dim);
        // Every indexed doc lands in exactly one cluster, so the
        // per-cluster counts sum to the superfile's doc count.
        let total: u64 = vs.clusters.counts.iter().map(|&c| c as u64).sum();
        assert_eq!(total, seg.n_docs);
    }

    #[test]
    fn open_blob_omits_fp32_centroids_keeps_cluster_idx() {
        // `dim` is chosen so the fp32 centroid block (`n_cent * dim * 4`) is
        // far larger than any structural open range (outer header, directory,
        // sub-header, cluster_idx), making the exclusion unambiguous.
        let dim = 64;
        let st = Supertable::create(options_with_vector(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 8, dim)).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let seg = &r.manifest().superfiles[0];
        let vs = seg.vector_summary.get("emb").expect("emb summary");
        let n_cent = vs.clusters.n_cent as usize;
        assert!(n_cent >= 1 && vs.clusters.dim as usize == dim);

        let offsets = seg
            .subsection_offsets
            .as_ref()
            .expect("subsection offsets captured at commit");
        let centroids_bytes = (n_cent * dim * 4) as u64;
        let cluster_idx_bytes =
            (n_cent * crate::superfile::format::vec::CLUSTER_IDX_ENTRY_BYTES) as u64;

        // No captured open range is centroid-sized: the fp32 centroids are not
        // staged into the manifest open_blob (the cluster-probe hot path never
        // reads them; the fallback nprobe path range-GETs them on demand).
        assert!(
            offsets
                .vec_open_ranges
                .iter()
                .all(|&(_, len)| len < centroids_bytes),
            "open_blob must not carry fp32 centroids; ranges={:?}, centroids={centroids_bytes} B",
            offsets.vec_open_ranges,
        );
        // ...but it must still carry the small cluster_idx that the
        // cluster-probe path reads zero-GET on cold open.
        assert!(
            offsets
                .vec_open_ranges
                .iter()
                .any(|&(_, len)| len == cluster_idx_bytes),
            "open_blob must carry cluster_idx ({cluster_idx_bytes} B); ranges={:?}",
            offsets.vec_open_ranges,
        );
    }

    // ---- rayon-shard parallelism -------------------------------------

    #[test]
    fn commit_produces_one_superfile_per_writer_pool_thread() {
        // With N writer-pool threads and a buffer of M >= N
        // batches, commit should emit N superfiles (one per
        // shard).
        for n_threads in [1usize, 2, 4] {
            let opts = options_id_title().with_writer_pool(writer_pool_with(n_threads));
            let st = Supertable::create(opts).expect("create");
            let mut w = st.writer().expect("writer");
            // Push enough batches to fill every shard.
            for i in 0..n_threads * 2 {
                w.append(&build_simple_batch(i as u64 * 10, 3))
                    .expect("append");
            }
            w.commit().expect("commit");

            let r = st.reader();
            assert_eq!(
                r.n_superfiles(),
                n_threads,
                "expected {n_threads} superfiles for {n_threads}-thread pool",
            );
            assert_eq!(r.n_docs_total(), (n_threads * 2 * 3) as u64);
        }
    }

    #[test]
    fn commit_with_fewer_batches_than_threads_skips_empty_shards() {
        // 4 threads, only 2 batches — chunk_size = 1, two chunks
        // get one batch each, the other two get nothing.
        // Should produce 2 superfiles, not 4.
        let opts = options_id_title().with_writer_pool(writer_pool_with(4));
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 1)).expect("a");
        w.append(&build_simple_batch(1, 1)).expect("b");
        w.commit().expect("commit");

        let r = st.reader();
        assert_eq!(r.n_superfiles(), 2);
        assert_eq!(r.n_docs_total(), 2);
    }

    #[test]
    fn apply_config_with_fixed_writer_threads_emits_that_many_superfiles() {
        let yaml = r#"
commit_threshold_size_mb: 1024
supertable:
  reader_threads: 1
  writer_threads: 4
"#;
        let cfg = crate::config::Config::from_figment(Figment::new().merge(Yaml::string(yaml)))
            .expect("parse config");

        // End-to-end: build options, route them through apply_config,
        // and verify the writer pool actually sized to the config's
        // 4 threads (one superfile per shard).
        let opts = options_id_title().apply_config(&cfg).expect("apply_config");
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");
        for i in 0..8u64 {
            w.append(&build_simple_batch(i * 10, 3)).expect("append");
        }
        w.commit().expect("commit");

        let r = st.reader();
        assert_eq!(
            r.n_superfiles(),
            4,
            "writer_threads=4 should yield 4 shards"
        );
        assert_eq!(r.n_docs_total(), 24);
    }

    // ---- threshold auto-flush ----------------------------------------

    #[test]
    fn append_auto_flushes_when_buffer_crosses_threshold() {
        // 1 MiB threshold; one append > 1 MiB should auto-commit.
        let opts = options_id_title_serial().with_commit_threshold_size_mb(1);
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");

        // Build a large batch: 50K docs × ~50-byte titles ≈ 2.5 MiB.
        let batch = build_simple_batch(0, 50_000);
        w.append(&batch).expect("append");

        // Threshold should have tripped; manifest_id has advanced.
        assert_eq!(st.manifest_id(), 1, "auto-flush should fire");
        assert_eq!(w.buffered_batches(), 0, "buffer drained on auto-flush");

        // No further commit should land an empty superfile.
        w.commit().expect("commit-empty");
        assert_eq!(st.manifest_id(), 1);
    }

    #[test]
    fn append_does_not_auto_flush_when_threshold_zero() {
        let opts = options_id_title_serial().with_commit_threshold_size_mb(0);
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 50_000)).expect("append");
        assert_eq!(st.manifest_id(), 0, "no auto-flush at threshold=0");
        assert!(w.buffered_batches() > 0);
    }

    // commit latency O(n) regression with localfs storage provider

    /// Each `Supertable::append` call rewrites the entire manifest part
    /// (Avro-encode + zstd-compress all N accumulated superfile entries,
    /// then PUT to storage). Commit K is O(K), so 100 sequential commits
    /// are O(n²) total and latency grows linearly with superfile count.
    #[ignore = "known O(n) regression: manifest part rewrite on every commit"]
    #[test]
    fn commit_latency_is_constant_with_localfs() {
        const N: usize = 100;
        const DOCS_PER_COMMIT: usize = 64;
        const MAX_GROWTH_FACTOR: f64 = 2.0;

        let dir = TempDir::new().expect("tempdir");
        let storage = Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let opts = options_id_title_serial().with_storage(storage);
        let st = Supertable::create(opts).expect("create");

        let mut latencies_ms: Vec<u128> = Vec::with_capacity(N);
        for i in 0..N {
            let batch = build_simple_batch(i as u64, DOCS_PER_COMMIT);
            let t0 = std::time::Instant::now();
            st.append(&batch).expect("append");
            latencies_ms.push(t0.elapsed().as_millis());
        }

        let avg = |slice: &[u128]| slice.iter().sum::<u128>() as f64 / slice.len() as f64;
        let first5_avg = avg(&latencies_ms[..5]);
        let last5_avg = avg(&latencies_ms[N - 5..]);
        let ratio = last5_avg / first5_avg.max(1.0);

        println!(
            "first-5 avg: {first5_avg:.1}ms  last-5 avg: {last5_avg:.1}ms  ratio: {ratio:.1}x"
        );
        assert!(
            ratio <= MAX_GROWTH_FACTOR,
            "commit latency grew {ratio:.1}x from first-5 ({first5_avg:.1}ms) to \
             last-5 ({last5_avg:.1}ms) — O(n) growth in manifest rewrite path"
        );
    }

    // ---- manifest copy-on-write across multiple commits -------------

    #[test]
    fn each_commit_appends_to_existing_superfiles() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 2)).expect("a1");
        w.commit().expect("c1");
        w.append(&build_simple_batch(10, 3)).expect("a2");
        w.commit().expect("c2");
        w.append(&build_simple_batch(20, 1)).expect("a3");
        w.commit().expect("commit");

        let r = st.reader();
        assert_eq!(r.manifest_id(), 3);
        assert_eq!(r.n_superfiles(), 3);
        assert_eq!(r.n_docs_total(), 6);
    }
}
