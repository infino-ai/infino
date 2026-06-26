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

use std::{
    cmp,
    collections::HashMap,
    fmt, io,
    marker::PhantomData,
    mem, slice,
    sync::{Arc, atomic::Ordering},
    time,
};

use arrow::ipc::writer::StreamWriter;
use arrow_array::{
    Array, ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array,
};
use bytes::Bytes;
use chrono::Utc;
use datafusion::prelude::Expr;
use futures::{
    future::try_join_all,
    stream::{self, StreamExt},
};
use object_store::{PutPayload, UploadPart};
use rayon::prelude::*;
use tokio::time::sleep;

use super::{
    build::{fanout_shards, fanout_shards_in_pool_scope},
    error::BuildError,
    handle::{GLOBAL_VECTOR_KMEANS_ITERS, GLOBAL_VECTOR_KMEANS_SEED, Supertable, SupertableInner},
    manifest::{
        FtsSummaryAgg, ScalarStatsAgg, SubsectionOffsets, SuperfileEntry, SuperfileUri,
        VectorSummary, bloom::BloomBuilder,
    },
    mutations::{
        CommitError, CommitResult, MAX_TARGETS_PER_MUTATION, MutationError, MutationStats,
        PendingDelete, PendingUpdate,
    },
    options::{DECIMAL128_PRECISION, DECIMAL128_SCALE, SupertableOptions},
    spfresh,
    utils::vector_split::split_vectors,
    wal::{
        WalStore,
        pipeline::{self, TombstonePhaseOutcome},
        state_doc::{
            IdSpan, OpKind, RowId, SCHEMA_VERSION, TombstoneEntry, TombstoneOutcome, WalId,
            WalState, WalStateDoc,
        },
    },
};
#[cfg(test)]
use crate::superfile::ReadError;
use crate::{
    InfinoError,
    runtime_bridge::bridge_on_runtime,
    storage::{StorageError, StorageProvider},
    superfile::{
        SuperfileReader,
        builder::SuperfileBuilder,
        format::{
            CRC_BYTES,
            footer::read_kv_metadata,
            fts::{HEADER_SIZE as FTS_HEADER_SIZE, U64_BYTES, hdr},
            kv,
            vec::{
                CLUSTER_IDX_ENTRY_BYTES, DIR_ENTRY_SIZE, OUTER_HEADER_SIZE, SUB_HEADER_SIZE,
                U32_BYTES, dir_entry, outer_hdr, sub_hdr,
            },
        },
        reader::vector_layout_from_kv,
        vector::{
            cell_posting::{
                EncodedCellRow, MaterializedIvfRow, manifest_centroid_components_from_row,
            },
            distance::Metric,
            ivf_merge::{RoutedCellSubsection, encode_encoded_rows},
            kmeans::kmeans_with_assignments,
            layout::VectorLayout,
            reader::VectorReader,
        },
    },
    supertable::{
        CommitError as SupertableCommitError, ManifestLoadError,
        error::ManifestError,
        manifest::{
            ClusterCentroids, Manifest,
            commit::{get_current_manifest_etag, put_immutable_blob},
            list::{CellRoutingParams, OpannRouting, PartitionStrategy},
            part::{self as part_mod, ContentHash, PartId},
            partition::{assign_partition, encode_partition_key},
        },
        opann::{
            insert::{LeafInsert, update_tree},
            paged::{OverlayPageSource, ResidentPageSource, SplitPages},
            store,
        },
        query::{
            dispatch::{open_compaction_input, open_reader},
            vector::stable_ids_by_local_for_routing,
        },
        reader_cache::DiskCacheStore,
    },
};

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
    wal_id: WalId,
    target_ids: Vec<i128>,
    preallocated_superfile_id: uuid::Uuid,
    minted_id_spans: Vec<IdSpan>,
    new_row_count: u32,
    new_row_content_hash: String,
    ipc_bytes: Bytes,
}

/// One buffered delete. Just the call-time resolved target_ids
/// + a pre-minted `wal_id`.
struct PendingDeleteEntry {
    wal_id: WalId,
    target_ids: Vec<i128>,
}

impl fmt::Debug for SupertableWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
            let take = cmp::min(shard_remaining, n_rows - row_cursor);
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

/// After a manifest swap that drops superfile references, schedule a deferred
/// GC sweep instead of inline `storage.delete`. Inline delete races snapshot-
/// pinned readers that may still cold-fetch superseded bytes.
fn schedule_background_storage_reclaim(inner: Arc<SupertableInner>) {
    if inner.options.storage.is_none() {
        return;
    }
    // Integration tests that need reclaim call `Supertable::gc()` explicitly
    // (see `tests/supertable/compact_gc.rs`). Spawning here from a
    // `current_thread` tokio test runtime panics in `block_in_place`.
    #[cfg(not(test))]
    {
        let rt = inner.query_runtime();
        rt.spawn(async move {
            sleep(super::gc::DEFAULT_SUPERFILE_RECLAIM_GRACE).await;
            if let Err(e) = super::gc::gc_storage_sweep_for_inner(
                &inner,
                super::gc::DEFAULT_SUPERFILE_RECLAIM_GRACE,
            )
            .await
            {
                tracing::debug!("supertable: deferred storage reclaim: {e}");
            }
        });
    }
    #[cfg(test)]
    {
        let _ = inner;
    }
}

/// Sq8+ε IVF rows aligned to scalar `_id` row order. Optional tombstone bitmap
/// skips deleted locals (cell maintenance); incoming routing passes `None`.
fn align_materialized_rows_in_doc_order(
    mut rows: Vec<MaterializedIvfRow>,
    stable_ids_by_local: &[i128],
    tombstones: Option<&roaring::RoaringBitmap>,
) -> Vec<MaterializedIvfRow> {
    if let Some(bm) = tombstones {
        rows.retain(|r| !bm.contains(r.local_doc_id));
    }
    let n_rows = stable_ids_by_local.len();
    // Dense local ids: sort into doc order without per-slot clones.
    if rows.len() == n_rows {
        rows.sort_by_key(|r| r.local_doc_id);
        for (local, row) in rows.iter_mut().enumerate() {
            if row.stable_id == 0 {
                row.stable_id = stable_ids_by_local[local];
                row.encoded.stable_id = row.stable_id;
            }
            row.local_doc_id = local as u32;
        }
        return rows;
    }
    let mut by_local: Vec<Option<MaterializedIvfRow>> = vec![None; n_rows];
    for mut row in rows {
        if row.stable_id == 0 {
            let slot = row.local_doc_id as usize;
            if slot < n_rows {
                row.stable_id = stable_ids_by_local[slot];
                row.encoded.stable_id = row.stable_id;
            }
        }
        let slot = row.local_doc_id as usize;
        if slot < n_rows {
            by_local[slot] = Some(row);
        }
    }
    by_local
        .into_iter()
        .enumerate()
        .filter_map(|(i, r)| {
            r.map(|mut row| {
                row.local_doc_id = i as u32;
                row
            })
        })
        .collect()
}

async fn materialized_ivf_rows_in_doc_order(
    vec_reader: &VectorReader,
    column: &str,
    stable_ids_by_local: &[i128],
    tombstones: Option<&roaring::RoaringBitmap>,
) -> Result<Vec<MaterializedIvfRow>, BuildError> {
    let rows = vec_reader
        .materialized_index_rows_async(column)
        .await
        .ok_or_else(|| {
            BuildError::Store(format!(
                "IVF maintenance: column '{column}' missing Sq8Residual index"
            ))
        })?;
    Ok(align_materialized_rows_in_doc_order(
        rows,
        stable_ids_by_local,
        tombstones,
    ))
}

async fn open_incoming_superfile_for_drain(
    store: &Arc<dyn crate::supertable::reader_cache::SuperfileReaderCache>,
    disk_cache: Option<&Arc<DiskCacheStore>>,
    storage: Option<&Arc<dyn StorageProvider>>,
    entry: &Arc<SuperfileEntry>,
) -> Result<Arc<SuperfileReader>, BuildError> {
    if let Ok(reader) = store.reader(&entry.uri) {
        return Ok(reader);
    }
    if let Ok(reader) = open_compaction_input(store, disk_cache, storage, entry).await {
        return Ok(reader);
    }
    let Some(storage) = storage else {
        return Err(BuildError::Store(
            "incoming superfile not resident in reader cache".into(),
        ));
    };
    let path = entry.uri.storage_path();
    let (bytes, _) = storage
        .get(&path)
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;
    SuperfileReader::open(bytes)
        .map(Arc::new)
        .map_err(|e| BuildError::Store(e.to_string()))
}

/// One commit's hidden-index batch, ready to build into a single
/// "incoming" IVF superfile. No per-cell split happens here — callers
/// invoke [`Supertable::drain`] (via [`Supertable::optimize`]) to route
/// INCOMING rows into per-cell superfiles.
struct HiddenIncomingPlan {
    buffer: Vec<BufferedBatch>,
    clusters: ClusterCentroids,
    column: String,
}

/// Build ONE "incoming" IVF superfile from a commit's hidden batch,
/// tagged with the reserved incoming partition. This reuses the standard IVF
/// superfile writer (`build_one_shard`) verbatim — no per-cell work. Queries
/// always scan incoming superfiles until [`drain_incoming_to_cells`] routes
/// them into per-cell IVF superfiles and removes them.
fn execute_hidden_incoming_plan_in_scope(
    inner: &SupertableInner,
    plan: HiddenIncomingPlan,
) -> Result<HiddenIncomingPrepare, BuildError> {
    let HiddenIncomingPlan {
        buffer,
        clusters,
        column,
    } = plan;
    let empty_batch = SuperfilePublishBatch {
        new_entries: Vec::new(),
        to_remove: Vec::new(),
        pending_storage_writes: Vec::new(),
        pending_cache_inserts: Vec::new(),
    };
    if buffer.is_empty() {
        return Ok(HiddenIncomingPrepare {
            batch: empty_batch,
            cell_updates: HashMap::new(),
            radii_updates: HashMap::new(),
            clusters,
            column,
        });
    }
    // Normal IVF layout for the incoming append region (same as user-table vectors).
    let shard = build_one_shard_with_layout(&buffer, &inner.options, VectorLayout::Ivf)?;
    let prepared = prepare_superfile(inner, shard)?
        .ok_or_else(|| BuildError::Store("hidden incoming superfile unexpectedly empty".into()))?;
    let entry = finish_superfile_entry(
        inner,
        prepared.entry,
        Some(super::handle::INCOMING_VECTOR_CELL),
    )?;
    let prepared = PreparedSuperfile {
        entry,
        bytes_for_store: prepared.bytes_for_store,
        bytes_for_storage: prepared.bytes_for_storage,
        bytes_for_cache: prepared.bytes_for_cache,
    };
    let batch = collect_prepared_superfiles(inner, vec![prepared])?;
    Ok(HiddenIncomingPrepare {
        batch,
        // Counts are bumped by maintenance when rows actually land in cells.
        cell_updates: HashMap::new(),
        radii_updates: HashMap::new(),
        clusters,
        column,
    })
}

/// Split buffered rows into per-cell shards based on nearest centroid.
/// Each shard carries all rows assigned to one cell; the caller stamps
/// `partition_hint` on the resulting superfile entries.
fn split_buffer_by_vector_cell(
    buffer: Vec<BufferedBatch>,
    cells: &ClusterCentroids,
    metric: Metric,
    vec_col_idx: usize,
) -> Vec<(u32, Vec<BufferedBatch>)> {
    let k = cells.n_cent as usize;
    let mut cell_batches: Vec<Vec<BufferedBatch>> = (0..k).map(|_| Vec::new()).collect();
    for batch in buffer {
        let n_rows = batch.scalar.num_rows();
        if n_rows == 0 {
            continue;
        }
        let vecs = batch.vectors[vec_col_idx].values();
        let mut assignments = vec![0u32; n_rows];
        cells.assign_rows(metric, vecs, &mut assignments);
        let mut per_cell_rows: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
        for (row, &cell) in assignments.iter().enumerate() {
            per_cell_rows[cell as usize].push(row);
        }
        for (cell_id, rows) in per_cell_rows.into_iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let indices = UInt32Array::from(rows.iter().map(|&r| r as u32).collect::<Vec<_>>());
            let scalar_cols: Vec<ArrayRef> = (0..batch.scalar.num_columns())
                .map(|col_idx| {
                    arrow::compute::take(batch.scalar.column(col_idx), &indices, None)
                        .expect("take column")
                })
                .collect();
            let scalar_batch =
                RecordBatch::try_new(batch.scalar.schema(), scalar_cols).expect("rebuild batch");
            let vectors: Vec<Arc<Float32Array>> = batch
                .vectors
                .iter()
                .map(|v| {
                    let vdim = v.len() / n_rows;
                    let mut out = Vec::with_capacity(rows.len() * vdim);
                    for &r in &rows {
                        out.extend_from_slice(&v.values()[r * vdim..(r + 1) * vdim]);
                    }
                    std::sync::Arc::new(Float32Array::from(out))
                })
                .collect();
            cell_batches[cell_id].push(BufferedBatch {
                scalar: scalar_batch,
                vectors,
            });
        }
    }
    cell_batches
        .into_iter()
        .enumerate()
        .filter(|(_, batches)| !batches.is_empty())
        .map(|(cell_id, batches)| (cell_id as u32, batches))
        .collect()
}

/// The public folded `update` / `delete` buffer exactly one mutation
/// before committing, so `CommitResult.outcomes` carries exactly one
/// entry; surface it (or a backend error if, impossibly, none landed).
fn single_outcome(res: CommitResult) -> Result<MutationStats, InfinoError> {
    res.outcomes
        .into_iter()
        .next()
        .ok_or_else(|| InfinoError::Backend("commit produced no mutation outcome".to_string()))
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
    pub fn append(&self, batch: &RecordBatch) -> Result<(), InfinoError> {
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
        predicate: Expr,
        new_rows: &RecordBatch,
    ) -> Result<MutationStats, InfinoError> {
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
    pub fn delete(&self, predicate: Expr) -> Result<MutationStats, InfinoError> {
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

fn bootstrap_centroids_from_batch(
    batches: &[BufferedBatch],
    vec_dim: usize,
    n_cells: usize,
    metric: Metric,
) -> Option<ClusterCentroids> {
    let mut vectors = Vec::new();
    for batch in batches {
        if batch.vectors.is_empty() {
            continue;
        }
        let vecs = batch.vectors[0].values();
        let n_rows = batch.scalar.num_rows();
        for row in 0..n_rows {
            vectors.extend_from_slice(&vecs[row * vec_dim..(row + 1) * vec_dim]);
        }
    }
    let n_docs = vectors.len() / vec_dim;
    if n_docs == 0 {
        return None;
    }
    let k = n_cells.min(n_docs).max(1);
    let (centroids, assignments) = kmeans_with_assignments(
        &vectors,
        vec_dim,
        k,
        GLOBAL_VECTOR_KMEANS_ITERS,
        GLOBAL_VECTOR_KMEANS_SEED,
    );
    let mut counts = vec![0u32; k];
    for &a in &assignments {
        counts[a as usize] += 1;
    }
    let clusters =
        ClusterCentroids::from_fp32(metric, k as u32, vec_dim as u32, &centroids, counts);
    Some(clusters.clone().with_radii(per_cell_radii(
        &clusters,
        &vectors,
        &assignments,
        vec_dim,
        metric,
    )))
}

fn per_cell_radii(
    clusters: &ClusterCentroids,
    vectors: &[f32],
    assignments: &[u32],
    vec_dim: usize,
    metric: Metric,
) -> Vec<f32> {
    let n_cent = clusters.n_cent as usize;
    let mut radii = vec![0.0f32; n_cent];
    for (doc_idx, &cell) in assignments.iter().enumerate() {
        let c = cell as usize;
        if c >= n_cent {
            continue;
        }
        let member = &vectors[doc_idx * vec_dim..(doc_idx + 1) * vec_dim];
        let dist = clusters.score_one(metric, c, member);
        if dist > radii[c] {
            radii[c] = dist;
        }
    }
    radii
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
            .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
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
                .map(|v| v.len() * mem::size_of::<f32>())
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

    /// Dual-write helper for the hidden vector-index supertable: buffer one
    /// shard batch using the user table's stable row ids instead of minting
    /// new ones so global-index hits can map back to user superfiles.
    pub(crate) fn append_dual_write_batch(
        &mut self,
        ids: &Decimal128Array,
        vectors: &[Arc<Float32Array>],
    ) -> Result<(), BuildError> {
        let options = &self.inner.options;
        let n_rows = ids.len();
        if n_rows == 0 {
            return Ok::<(), BuildError>(());
        }
        if vectors.len() != options.vector_columns.len() {
            return Err(BuildError::BatchSchemaMismatch);
        }
        let id_array = ids
            .clone()
            .with_precision_and_scale(
                crate::supertable::options::DECIMAL128_PRECISION,
                crate::supertable::options::DECIMAL128_SCALE,
            )
            .expect(
                "invariant: precision 38 + scale 0 always valid \
                 for any i128 payload",
            );
        let scalar = RecordBatch::try_new(
            options.scalar_schema(),
            vec![Arc::new(id_array) as ArrayRef],
        )
        .map_err(|_| BuildError::BatchSchemaMismatch)?;
        let bytes = scalar.get_array_memory_size()
            + vectors
                .iter()
                .map(|v| v.len() * std::mem::size_of::<f32>())
                .sum::<usize>();
        self.buffer.push(BufferedBatch {
            scalar,
            vectors: vectors.to_vec(),
        });
        self.buffer_bytes += bytes;
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
    pub fn delete(&mut self, predicate: Expr) -> Result<PendingDelete, MutationError> {
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
        predicate: Expr,
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
            MutationError::Storage(StorageError::Permanent {
                uri: "ipc encode".into(),
                source: Box::new(io::Error::other(e)),
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
        let mut committed_wal_ids: Vec<WalId> = Vec::with_capacity(total_mutations);
        let mut outcomes: Vec<MutationStats> = Vec::with_capacity(total_mutations);

        // Step 2: drive pending updates in buffer order. On
        // mid-loop failure, the failed entry is dropped (its
        // WAL may already be on storage; recovery sweep
        // completes it on the next open) and the unattempted
        // entries stay on `self.pending_updates` for retry.
        let mut updates_to_run = mem::take(&mut self.pending_updates);
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
        let mut deletes_to_run = mem::take(&mut self.pending_deletes);
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
        let (n_tombstoned, n_not_found) = bridge_on_runtime(drive, &self.inner.query_runtime())?;
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
        // The hidden vector-index cells are not rewritten on a user delete, so
        // the deleted rows stay physically present in them. Record the resolved
        // user `_id`s into the hidden index's resident deleted-set so vector
        // search drops them in memory (zero per-cell tombstone GETs).
        let hidden_inner = self
            .inner
            .vector_index_table
            .as_ref()
            .map(|vit| Arc::clone(vit.inner()));
        let deleted_ids: Vec<i128> = entry.target_ids.clone();
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
            // Best-effort, mirroring the dual-write append path: a failure here
            // leaves the durable user-table delete intact; vector search may
            // transiently surface a deleted row until the next successful
            // record (or a tombstone-aware hidden compaction prunes it).
            if let Some(hi) = hidden_inner {
                if let Err(e) = record_hidden_deleted_ids(&hi, &deleted_ids).await {
                    tracing::warn!(
                        "supertable: hidden vector-index deleted-set record failed: {e} \
                         (user-table delete is durable; vector search may transiently \
                         return deleted rows until the next successful record)"
                    );
                }
            }
            Ok::<_, MutationError>((n_t, n_nf))
        };
        let (n_tombstoned, n_not_found) = bridge_on_runtime(drive, &self.inner.query_runtime())?;
        Ok(MutationStats {
            wal_id: entry.wal_id,
            matched: entry.target_ids.len(),
            n_tombstoned,
            n_not_found,
        })
    }

    /// Plan one incoming IVF superfile from the hidden writer's buffered vectors.
    /// Bootstraps the global cell centroids from the first batch if they don't
    /// exist yet (routing and drain both need them), but does NOT split
    /// by cell and does NOT touch per-cell counts — that happens in
    /// [`drain_incoming_to_cells`], when rows land in cells.
    fn plan_hidden_incoming_shard(&mut self) -> Result<HiddenIncomingPlan, BuildError> {
        let empty = HiddenIncomingPlan {
            buffer: Vec::new(),
            clusters: ClusterCentroids::default(),
            column: String::new(),
        };
        if self.buffer.is_empty() {
            return Ok(empty);
        }
        super::handle::apply_pending_partition_strategy(&self.inner);
        let vec_dim = self
            .inner
            .options
            .vector_columns
            .first()
            .map(|vc| vc.dim)
            .unwrap_or(0);
        if vec_dim == 0 {
            return Ok(empty);
        }
        let column = self
            .inner
            .options
            .vector_columns
            .first()
            .map(|vc| vc.column.clone())
            .unwrap_or_default();

        let metric = self
            .inner
            .options
            .vector_columns
            .first()
            .map(|vc| vc.metric)
            .unwrap_or(Metric::L2Sq);

        // Ensure the global cell grid exists (bootstrap once from the first
        // batch). Counts are left untouched — maintenance bumps them when it
        // moves incoming rows into cells.
        let mut strategy = self.inner.manifest.load().get_partition_strategy();
        let clusters = match &strategy {
            PartitionStrategy::VectorCell { clusters, .. }
                if clusters.n_cent > 0 && clusters.dim > 0 =>
            {
                clusters.clone()
            }
            _ => {
                let boot = bootstrap_centroids_from_batch(
                    &self.buffer,
                    vec_dim,
                    super::handle::GLOBAL_VECTOR_CELL_COUNT,
                    metric,
                )
                .ok_or_else(|| {
                    BuildError::Store("hidden index: bootstrap centroids from batch failed".into())
                })?;
                strategy = PartitionStrategy::VectorCell {
                    column: column.clone(),
                    clusters: boot.clone(),
                    routing: Default::default(),
                };
                self.inner.manifest.store(Arc::new(
                    self.inner
                        .manifest
                        .load()
                        .with_partition_strategy(strategy.clone()),
                ));
                boot
            }
        };

        let buffer = std::mem::take(&mut self.buffer);
        self.buffer_bytes = 0;
        Ok(HiddenIncomingPlan {
            buffer,
            clusters,
            column,
        })
    }

    fn prepare_hidden_incoming_build(&mut self) -> Result<HiddenIncomingPrepare, BuildError> {
        let plan = self.plan_hidden_incoming_shard()?;
        let inner = Arc::clone(&self.inner);
        self.inner
            .options
            .writer_pool
            .install(|| execute_hidden_incoming_plan_in_scope(&inner, plan))
    }

    fn commit_hidden_incoming_internal(&mut self) -> Result<(), BuildError> {
        let prep = self.prepare_hidden_incoming_build()?;
        if prep.batch.new_entries.is_empty() {
            return Ok(());
        }
        let storage = self
            .inner
            .options
            .storage
            .as_ref()
            .cloned()
            .ok_or_else(|| BuildError::Store("hidden incoming commit requires storage".into()))?;
        let inner = Arc::clone(&self.inner);
        bridge_on_runtime(
            publish_hidden_incoming_async(inner, storage, prep),
            &self.inner.query_runtime(),
        )?;
        Ok(())
    }

    /// [`SupertableWriter::commit`] calls this first before
    /// driving pending mutations.
    ///
    /// Rows are balanced evenly across shards regardless of the
    /// caller's `append()` cadence — many small appends followed by
    /// one `commit` produce the same shard layout as one large append.
    fn commit_appends_internal(&mut self) -> Result<(), BuildError> {
        if self.buffer.is_empty() {
            return Ok::<(), BuildError>(());
        }
        if crate::supertable::handle::is_hidden_vector_index_table(&self.inner.options)
            && self.inner.options.storage.is_some()
        {
            return self.commit_hidden_incoming_internal();
        }
        let buffer = mem::take(&mut self.buffer);
        self.buffer_bytes = 0;

        // Dual-write vectors into the hidden index writer, then build/publish
        // user and hidden superfiles in parallel (create via rayon::join,
        // publish via tokio::join).
        let mut hidden_writer = None;
        if let Some(vit) = self.inner.vector_index_table.as_ref() {
            match vit.writer() {
                Ok(mut vw) => {
                    for batch in &buffer {
                        let Some(ids) = batch
                            .scalar
                            .column(0)
                            .as_any()
                            .downcast_ref::<Decimal128Array>()
                        else {
                            tracing::warn!(
                                "supertable: hidden vector-index dual-write missing _id column"
                            );
                            continue;
                        };
                        if let Err(e) = vw.append_dual_write_batch(ids, &batch.vectors) {
                            tracing::warn!(
                                "supertable: hidden vector-index append failed: {e} (user-table commit continues; vector search may be stale)"
                            );
                            break;
                        }
                    }
                    hidden_writer = Some(vw);
                }
                Err(e) => {
                    tracing::warn!(
                        "supertable: hidden vector-index writer unavailable: {e} (vector search may be stale)"
                    );
                }
            }
        }

        let total_rows: usize = buffer.iter().map(|b| b.scalar.num_rows()).sum();
        if total_rows == 0 {
            return Ok::<(), BuildError>(());
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
        // VectorCell strategy: pre-shard by nearest centroid instead of
        // round-robin. Each shard becomes one superfile in its cell-partition.
        let (shards, cell_hints): (Vec<Vec<BufferedBatch>>, Vec<Option<u32>>) =
            if let Some(PartitionStrategy::VectorCell { ref clusters, .. }) =
                self.inner.options.partition_strategy
            {
                let metric = self
                    .inner
                    .options
                    .vector_columns
                    .first()
                    .map(|vc| vc.metric)
                    .unwrap_or(Metric::L2Sq);
                if clusters.n_cent > 0 && clusters.dim > 0 {
                    // Run on the build pool: `split_buffer_by_vector_cell` →
                    // `assign_rows` is a CPU wave (per-row nearest-cell scoring)
                    // and must dispatch to `writer_pool`, not the global rayon
                    // pool, per the rayon-owns-CPU concurrency contract.
                    let cell_shards = writer_pool
                        .install(|| split_buffer_by_vector_cell(buffer, clusters, metric, 0));
                    let hints: Vec<Option<u32>> = cell_shards
                        .iter()
                        .map(|(cell_id, _)| Some(*cell_id))
                        .collect();
                    let shards: Vec<Vec<BufferedBatch>> = cell_shards
                        .into_iter()
                        .map(|(_, batches)| batches)
                        .collect();
                    (shards, hints)
                } else {
                    let shards = split_buffer_into_row_shards(buffer, n_shards, &vector_dims);
                    let hints = vec![None; shards.len()];
                    (shards, hints)
                }
            } else {
                let shards = split_buffer_into_row_shards(buffer, n_shards, &vector_dims);
                let hints = vec![None; shards.len()];
                (shards, hints)
            };

        // Parallel create: user superfile build + hidden incoming build
        // share one writer-pool install (rayon::join, no nested install).
        // Parallel publish: user + hidden manifest/storage commits overlap.
        let user_inner = Arc::clone(&self.inner);
        let user_options = Arc::clone(&self.inner.options);
        let hidden_plan = if let Some(vw) = hidden_writer.as_mut() {
            Some(vw.plan_hidden_incoming_shard()?)
        } else {
            None
        };
        let hidden_inner = hidden_writer.as_ref().map(|vw| Arc::clone(&vw.inner));
        let (user_batch, hidden_side) =
            if let (Some(plan), Some(hidden_inner)) = (hidden_plan, hidden_inner) {
                writer_pool.install(
                || -> Result<
                    (
                        SuperfilePublishBatch,
                        Option<(Arc<SupertableInner>, HiddenIncomingPrepare)>,
                    ),
                    BuildError,
                > {
                    let shards_ref = &shards;
                    let hints = cell_hints.clone();
                    let hidden_inner = Arc::clone(&hidden_inner);
                    let (user_batch, hidden_prep) = rayon::join(
                        || -> Result<SuperfilePublishBatch, BuildError> {
                            let outputs = fanout_shards_in_pool_scope(shards_ref, |slice| {
                                build_one_shard(slice.as_slice(), &user_options)
                            })?;
                            prepare_user_superfile_batch_in_scope(&user_inner, outputs, hints)
                        },
                        || execute_hidden_incoming_plan_in_scope(&hidden_inner, plan),
                    );
                    Ok((
                        user_batch?,
                        Some((hidden_inner, hidden_prep?)),
                    ))
                },
            )?
            } else {
                let outputs = fanout_shards(&writer_pool, &shards, |slice| {
                    build_one_shard(slice.as_slice(), &self.inner.options)
                })?;
                let batch = prepare_user_superfile_batch(&self.inner, outputs, cell_hints)?;
                (batch, None)
            };

        let drive = async {
            match hidden_side {
                Some((hidden_inner, prep)) => {
                    let hidden_storage = hidden_inner
                        .options
                        .storage
                        .as_ref()
                        .cloned()
                        .ok_or_else(|| {
                            BuildError::Store("hidden incoming commit requires storage".into())
                        })?;
                    let (user_res, hidden_res) = tokio::join!(
                        persist_superfile_publish_batch_async(&user_inner, user_batch),
                        publish_hidden_incoming_async(
                            Arc::clone(&hidden_inner),
                            hidden_storage,
                            prep
                        ),
                    );
                    user_res?;
                    if let Err(e) = hidden_res {
                        tracing::warn!(
                            "supertable: hidden vector-index commit failed: {e} (vector search may be stale)"
                        );
                    }
                    Ok::<(), BuildError>(())
                }
                None => persist_superfile_publish_batch_async(&user_inner, user_batch).await,
            }
        };
        bridge_on_runtime(drive, &self.inner.query_runtime())?;
        if self.inner.options.storage.is_some() {
            schedule_background_storage_reclaim(Arc::clone(&self.inner));
        }
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
/// FTS + vector summaries are derived in `prepare_user_superfile_batch` from
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
    scalar_stats: HashMap<String, ScalarStatsAgg>,
}

impl ShardOutput {
    pub fn new_with_params(
        bytes: Bytes,
        n_docs: u64,
        id_min: i128,
        id_max: i128,
        scalar_stats: HashMap<String, ScalarStatsAgg>,
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
    build_one_shard_with_layout(slice, options, options.vector_layout)
}

/// Same as [`build_one_shard`] but with an explicit vector layout override.
/// The hidden index's "incoming" append superfile is always built in IVF layout.
fn build_one_shard_with_layout(
    slice: &[BufferedBatch],
    options: &SupertableOptions,
    vector_layout: crate::superfile::vector::layout::VectorLayout,
) -> Result<ShardOutput, BuildError> {
    let mut builder =
        SuperfileBuilder::new(options.builder_options().with_vector_layout(vector_layout))?;

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
    let scalar_stats = ScalarStatsAgg::from_batches(&scalar_schema, &scalar_batches);

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
    // Derive the layout from the `kvs` already parsed above rather than
    // re-reading the footer via `read_vector_layout_from_bytes`.
    let layout = vector_layout_from_kv(&kvs);
    if layout == VectorLayout::CellPosting {
        // Cell-posting hidden superfiles are read in bulk (a full-cell scan of
        // the contiguous vec blob) and served resident from the disk cache.
        // Staging their bytes into the manifest `open_blob` would replicate the
        // entire vector index into the manifest — its size would grow with the
        // whole dataset (memory + cold-load GET cost), since the open overlay
        // captures each superfile's vec blob *and* parquet tail. Skip the
        // inline overlay entirely; the vec subsection is fetched on demand
        // (and cached) via `fetch_cell_posting_blob`. Offsets are still carried
        // so that fetch knows where to read.
        return Some(SubsectionOffsets {
            total_size,
            vec,
            fts,
            vec_open_ranges: Vec::new(),
            fts_open_ranges: Vec::new(),
            open_blob: Vec::new(),
        });
    }
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
    pub(crate) fn open_reader(&self) -> Option<Result<SuperfileReader, ReadError>> {
        let bytes = self
            .bytes_for_store
            .as_ref()
            .or(self.bytes_for_storage.as_ref())
            .or(self.bytes_for_cache.as_ref())
            .map(|(_, b)| b.clone())?;
        Some(SuperfileReader::open(bytes))
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
    prepare_superfile_with_uri(inner, shard, None, &[])
}

pub(super) fn prepare_superfile_with_uri(
    inner: &SupertableInner,
    shard: ShardOutput,
    reuse_uri: Option<SuperfileUri>,
    // Per-output-cluster covering radii for the vector column, aligned with the
    // cell's clusters by ordinal. Empty for every path except the drain's
    // per-cell splice; populates `VectorSummary.clusters.radii` so the
    // within-cell admission is radius-aware.
    cluster_radii: &[f32],
) -> Result<Option<PreparedSuperfile>, BuildError> {
    if shard.n_docs == 0 {
        return Ok(None);
    }

    let uri = reuse_uri.unwrap_or_else(SuperfileUri::new_v4);

    let bytes_for_storage = inner.options.storage.is_some().then(|| shard.bytes.clone());
    let cache_attached = inner.options.disk_cache.is_some() && inner.options.storage.is_some();
    // `bytes_for_store` (in-memory tier) is gated only on cache attachment —
    // a cache-attached producer keeps superfile bytes out of the unbounded
    // in-memory store regardless of whether we pre-populate the disk cache.
    let bytes_for_store = (!cache_attached).then(|| shard.bytes.clone());
    // Always warm-fill the disk cache when attached: commits are durable in
    // object storage first, then mirrored locally so maintenance/compaction
    // can merge from mmap-resident bytes without re-fetching whole objects.
    let bytes_for_cache = cache_attached.then(|| shard.bytes.clone());

    // Open the reader directly on shard bytes (not via the
    // in-memory `SuperfileReaderCache`). This lets the cache-attached
    // path skip the in-memory tier entirely — the bytes can go
    // straight to object storage without a RAM detour, which is
    // what removes the 100GB OOM trap (the in-memory cache doesn't
    // evict, so a long-running writer with cache + storage would
    // otherwise accumulate every superfile's bytes in RAM forever).
    let reader =
        SuperfileReader::open_with(shard.bytes.clone(), inner.options.superfile_open_options())
            .map_err(|e| BuildError::Store(format!("opening superfile for summary: {e}")))?;

    let mut fts_summary: HashMap<String, FtsSummaryAgg> = HashMap::new();
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
                FtsSummaryAgg::new_with_params(
                    bloom_builder.finish(),
                    n_terms_distinct,
                    (min_term, max_term),
                ),
            );
        }
    }

    let mut vector_summary: HashMap<String, VectorSummary> = HashMap::new();
    if let Some(vec_reader) = reader.vec() {
        for vc in &inner.options.vector_columns {
            if let Some((c_dim, c_scale, c_offset, c_rows, c_norm, radius)) =
                vec_reader.summary(&vc.column)
            {
                let centroid = ClusterCentroids {
                    n_cent: 1,
                    dim: c_dim,
                    scale: c_scale,
                    offset: c_offset,
                    rows: c_rows,
                    norms: c_norm.map(|n| vec![n]),
                    counts: vec![1],
                    radii: Vec::new(),
                };
                let mut clusters = vec_reader
                    .cluster_centroids_encoded(&vc.column)
                    .map(
                        |(n_cent, dim, scale, offset, rows, norms, counts)| ClusterCentroids {
                            n_cent,
                            dim,
                            scale,
                            offset,
                            rows,
                            norms,
                            counts,
                            radii: Vec::new(),
                        },
                    )
                    .unwrap_or_default();
                // Per-cluster covering radii (drain splice only) so the
                // within-cell admission scores by region overlap, not nearest
                // centroid. Aligned with `clusters` by ordinal.
                if !cluster_radii.is_empty() && clusters.n_cent as usize == cluster_radii.len() {
                    clusters.radii = cluster_radii.to_vec();
                }
                // Per-cluster byte offsets, read from the cell's cluster index
                // (the bytes the splice just wrote), aligned with `clusters` by
                // cluster ordinal — so a query can range-GET an individual
                // cluster without re-reading the index.
                let cluster_offsets = vec_reader
                    .cluster_doc_offsets(&vc.column)
                    .unwrap_or_default();
                vector_summary.insert(
                    vc.column.clone(),
                    VectorSummary {
                        centroid,
                        radius,
                        clusters,
                        cluster_offsets,
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
    let vector_layout = read_vector_layout_from_bytes(&shard.bytes);
    if vector_layout == VectorLayout::CellPosting
        && subsection_offsets.as_ref().and_then(|o| o.vec).is_none()
    {
        let kvs = crate::superfile::format::footer::read_kv_metadata(shard.bytes.as_ref())
            .map(|kvs| kvs.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        return Err(BuildError::Store(format!(
            "cell-posting superfile missing inf.vec offset/length; kv_keys={kvs:?}"
        )));
    }

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
        vector_layout,
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
fn finish_superfile_entry(
    inner: &SupertableInner,
    entry: Arc<SuperfileEntry>,
    hint: Option<u32>,
) -> Result<Arc<SuperfileEntry>, BuildError> {
    let old = entry.as_ref();
    let mut staged = SuperfileEntry {
        superfile_id: old.superfile_id,
        uri: old.uri,
        n_docs: old.n_docs,
        id_min: old.id_min,
        id_max: old.id_max,
        scalar_stats: old.scalar_stats.clone(),
        fts_summary: old.fts_summary.clone(),
        vector_summary: old.vector_summary.clone(),
        partition_key: old.partition_key.clone(),
        partition_hint: hint.or(old.partition_hint),
        subsection_offsets: old.subsection_offsets.clone(),
        vector_layout: old.vector_layout,
    };
    let strategy = inner.manifest.load().get_partition_strategy();
    let pk = assign_partition(&staged, &strategy)
        .map_err(|e| BuildError::Store(format!("partition assign: {e}")))?;
    staged.partition_key = encode_partition_key(&pk);
    Ok(Arc::new(staged))
}

/// Collected superfile entries + pending storage/cache writes for one publish.
struct SuperfilePublishBatch {
    new_entries: Vec<Arc<SuperfileEntry>>,
    to_remove: Vec<Arc<SuperfileEntry>>,
    pending_storage_writes: Vec<(SuperfileUri, Bytes)>,
    pending_cache_inserts: Vec<(SuperfileUri, Bytes)>,
}

/// Hidden incoming build artifacts produced before manifest swap.
struct HiddenIncomingPrepare {
    batch: SuperfilePublishBatch,
    cell_updates: HashMap<u32, u32>,
    radii_updates: HashMap<u32, f32>,
    clusters: ClusterCentroids,
    column: String,
}

fn collect_prepared_superfiles(
    inner: &SupertableInner,
    prepared: Vec<PreparedSuperfile>,
) -> Result<SuperfilePublishBatch, BuildError> {
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
    Ok(SuperfilePublishBatch {
        new_entries,
        to_remove: Vec::new(),
        pending_storage_writes,
        pending_cache_inserts,
    })
}

fn prepare_user_superfile_batch_in_scope(
    inner: &SupertableInner,
    outputs: Vec<ShardOutput>,
    hints: Vec<Option<u32>>,
) -> Result<SuperfilePublishBatch, BuildError> {
    let prepared: Vec<PreparedSuperfile> =
        outputs
            .into_par_iter()
            .zip(hints.into_par_iter())
            .filter_map(|(shard, hint)| match prepare_superfile(inner, shard) {
                Ok(Some(p)) => Some(finish_superfile_entry(inner, p.entry, hint).map(|entry| {
                    PreparedSuperfile {
                        entry,
                        bytes_for_store: p.bytes_for_store,
                        bytes_for_storage: p.bytes_for_storage,
                        bytes_for_cache: p.bytes_for_cache,
                    }
                })),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            })
            .collect::<Result<Vec<_>, _>>()?;
    collect_prepared_superfiles(inner, prepared)
}

fn prepare_user_superfile_batch(
    inner: &SupertableInner,
    outputs: Vec<ShardOutput>,
    hints: Vec<Option<u32>>,
) -> Result<SuperfilePublishBatch, BuildError> {
    inner
        .options
        .writer_pool
        .install(|| prepare_user_superfile_batch_in_scope(inner, outputs, hints))
}

async fn persist_superfile_publish_batch_async(
    inner: &SupertableInner,
    batch: SuperfilePublishBatch,
) -> Result<(), BuildError> {
    if batch.new_entries.is_empty() {
        return Ok(());
    }
    if let Some(storage) = inner.options.storage.as_ref().cloned() {
        let new_manifest = persist_commit_async(
            inner,
            storage,
            batch.new_entries,
            &batch.to_remove,
            batch.pending_storage_writes,
            Vec::new(),
            OpannRoutingCommit::Inherit,
        )
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;
        inner.manifest.store(Arc::new(new_manifest));
        // Already async — await the warm-cache fill directly. Do NOT call
        // `warm_cache_after_commit` here: its sync `block_in_place` + nested
        // `block_on` inside the `tokio::join!` commit future deadlocks the
        // runtime (main thread parked, all workers idle).
        if let Some(cache) = inner.options.disk_cache.as_ref() {
            warm_cache_inserts(cache, batch.pending_cache_inserts).await;
        }
        if let (Some(cache), Some(budget)) = (
            inner.options.disk_cache.as_ref(),
            inner.options.memory_budget_bytes,
        ) {
            cache.sweep_for_budget(budget);
        }
        return Ok(());
    }
    let old = inner.manifest.load();
    let new = old.with_appended(batch.new_entries);
    inner.manifest.store(Arc::new(new));
    Ok(())
}

/// Single-thread rayon pool for incoming-routing CPU work (cell assignment + per-cell
/// superfile encode). Installing the build under this pool pins all its nested
/// `par_iter`/`join` to one thread instead of fanning out across every core, so
/// drain can't starve foreground ingest CPU.
static MAINT_POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();

fn maint_pool() -> &'static rayon::ThreadPool {
    MAINT_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|_| "hidden-maint-cpu".into())
            .build()
            .expect("hidden maintenance rayon pool")
    })
}

/// Page-node budget per OPANN routing page (a connected subtree of ≤ this many
/// nodes; descent crosses a page boundary at the cut edges).
const OPANN_PAGE_MAX_NODES: usize = 4096;

/// One new partition's routing copy for the OPANN tree: the cluster's owning
/// superfile id, its `(doc_off, count)` range within that superfile's IVF
/// (`(0, 0)` = the whole partition — one object fetch then scan inside), its
/// fp32 centroid (the ingestion-surface k-means center / a transient decode of
/// the cell's stored centroid — never persisted), and its covering radius.
pub(in crate::supertable) struct PartitionRoutingCopy {
    pub(in crate::supertable) superfile_id: u128,
    pub(in crate::supertable) doc_off: u32,
    pub(in crate::supertable) count: u32,
    /// Internal IVF cluster ordinal within `superfile_id` (selects the cluster's
    /// Sq8 scale/offset at probe time). 0 for the whole-cell `(0,0)` leaf.
    pub(in crate::supertable) cluster_id: u32,
    pub(in crate::supertable) centroid_fp32: Vec<f32>,
    pub(in crate::supertable) radius: f32,
}

/// How a commit treats the manifest's OPANN routing root.
pub(crate) enum OpannRoutingCommit {
    /// Carry the prior routing forward unchanged (no tree change).
    Inherit,
    /// Stamp this routing into the committed manifest. `routing` is `Some` for a
    /// commit that copy-on-write-updated the tree, `None` to clear routing when
    /// the tree is emptied. `pages` are the changed routing-tree pages
    /// `(uri, bytes)` — content-addressed immutable blobs written in the commit's
    /// parallel pre-pointer wave, exactly like manifest parts.
    Replace {
        routing: Option<OpannRouting>,
        pages: Vec<(String, Bytes)>,
    },
}

/// Returns [`OpannRoutingCommit::Replace`] with the new root (`None` only when
/// the tree is emptied), or [`OpannRoutingCommit::Inherit`] when the root is
/// unchanged and no new pages need writing.
pub(in crate::supertable) fn opann_routing_commit_from_split(
    split: Option<SplitPages>,
    prior_root: Option<ContentHash>,
    params: CellRoutingParams,
    bundle: Option<(ContentHash, Vec<u8>)>,
    prior_deleted: Option<(String, ContentHash)>,
) -> OpannRoutingCommit {
    // Carry the prior deleted-set blob ref onto any freshly-built routing so a
    // tree-changing commit (drain / split / compaction) does not wipe it — the
    // deleted rows are still physically present in the (untouched) cells.
    let attach_bundle = |routing: &mut OpannRouting,
                         pages: &mut Vec<(String, Bytes)>,
                         bundle: Option<(ContentHash, Vec<u8>)>| {
        if let Some((uri, hash)) = prior_deleted.as_ref() {
            routing.deleted_ids_uri = Some(uri.clone());
            routing.deleted_ids_content_hash = Some(hash.clone());
        }
        if let Some((hash, bytes)) = bundle {
            let uri = store::resident_uri(&hash);
            pages.push((uri.clone(), Bytes::from(bytes)));
            routing.resident_uri = Some(uri);
            routing.resident_content_hash = Some(hash);
        }
    };
    match split {
        Some(split) if !split.pages.is_empty() => {
            let mut pages: Vec<(String, Bytes)> = split
                .pages
                .into_iter()
                .map(|(hash, bytes)| (store::page_uri(&hash), Bytes::from(bytes)))
                .collect();
            let mut routing = OpannRouting {
                root_page: split.root,
                routing: params,
                resident_uri: None,
                resident_content_hash: None,
                deleted_ids_uri: None,
                deleted_ids_content_hash: None,
            };
            attach_bundle(&mut routing, &mut pages, bundle);
            OpannRoutingCommit::Replace {
                routing: Some(routing),
                pages,
            }
        }
        // Root moved but every rewritten page was already on object storage
        // (content-addressed dedup at a prior commit). Still stamp the new root.
        Some(split) if Some(split.root) != prior_root => {
            let mut pages = Vec::new();
            let mut routing = OpannRouting {
                root_page: split.root,
                routing: params,
                resident_uri: None,
                resident_content_hash: None,
                deleted_ids_uri: None,
                deleted_ids_content_hash: None,
            };
            attach_bundle(&mut routing, &mut pages, bundle);
            OpannRoutingCommit::Replace {
                routing: Some(routing),
                pages,
            }
        }
        Some(_) => OpannRoutingCommit::Inherit,
        None => OpannRoutingCommit::Replace {
            routing: None,
            pages: Vec::new(),
        },
    }
}

/// Copy-on-write-update the OPANN routing tree: drop every leaf whose cell id is
/// in `removed`, splice in `added`, and return the resulting [`OpannRoutingCommit`]
/// (the changed pages + new root, or `Inherit` when nothing changed). `root ==
/// None` (genesis) builds the first tree from `added`.
pub(in crate::supertable) async fn opann_routing_update(
    inner: &SupertableInner,
    current: &Manifest,
    removed: &[u128],
    added: &[PartitionRoutingCopy],
) -> Result<OpannRoutingCommit, BuildError> {
    let Some(vec_col) = inner.options.vector_columns.first() else {
        return Ok(OpannRoutingCommit::Inherit);
    };
    if removed.is_empty() && added.is_empty() {
        return Ok(OpannRoutingCommit::Inherit);
    }
    let prior = current.opann_routing();
    let prior_root = prior.map(|r| r.root_page);
    let params = prior.map(|r| r.routing).unwrap_or_default();
    let leaves: Vec<LeafInsert> = added
        .iter()
        .map(|c| LeafInsert {
            superfile_id: c.superfile_id,
            doc_off: c.doc_off,
            count: c.count,
            cluster_id: c.cluster_id,
            centroid_fp32: c.centroid_fp32.clone(),
            radius: c.radius,
        })
        .collect();
    // Descend the current tree's resident pages (warm); empty source covers the
    // genesis case (no prior root).
    let source = match prior_root {
        Some(_) => current
            .opann_resident_tree()
            .await
            .map_err(|e| BuildError::Store(e.to_string()))?
            .unwrap_or_else(|| Arc::new(ResidentPageSource::from_pages(HashMap::new()))),
        None => Arc::new(ResidentPageSource::from_pages(HashMap::new())),
    };
    let split = update_tree(
        source.as_ref(),
        prior_root,
        vec_col.metric,
        vec_col.dim,
        removed,
        &leaves,
        OPANN_PAGE_MAX_NODES,
    )
    .map_err(|e| BuildError::Store(e.to_string()))?;
    let bundle = split.as_ref().and_then(|s| {
        if Some(s.root) == prior_root && s.pages.is_empty() {
            return None;
        }
        let overlay = OverlayPageSource::new(source.as_ref(), &s.pages);
        store::pack_resident_bundle(&overlay, s.root)
            .ok()
            .map(|bytes| (ContentHash::of(bytes.as_ref()), bytes))
    });
    let prior_deleted = prior.and_then(|r| {
        r.deleted_ids_uri
            .clone()
            .zip(r.deleted_ids_content_hash.clone())
    });
    Ok(opann_routing_commit_from_split(
        split, prior_root, params, bundle, prior_deleted,
    ))
}

/// Route every accumulated INCOMING IVF superfile into per-cell delta superfiles.
/// Entry point for [`Supertable::drain`] and for split/reassign redrive (step 9).
///
/// Async I/O (object-store + disk-cache reads) must run on the table's
/// `query_runtime` — the disk cache's coordination primitives are bound to
/// that runtime.
pub(in crate::supertable) async fn drain_incoming_to_cells(
    inner: Arc<SupertableInner>,
) -> Result<(), BuildError> {
    route_incoming_to_manifest_cells_if_ready(inner, 1, None).await
}

/// Read each accumulated incoming IVF superfile's Sq8+ε rows, assign every row
/// to its nearest global cell, build one Sq8 IVF superfile per touched cell,
/// then publish those cell superfiles and remove the routed incoming
/// superfiles in one OCC commit.
async fn route_incoming_to_manifest_cells_if_ready(
    inner: Arc<SupertableInner>,
    min_incoming: usize,
    assign_among: Option<&[u32]>,
) -> Result<(), BuildError> {
    // Single-flight: skip if another routing/compaction pass is already running.
    if inner
        .compaction_outstanding
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return Ok(());
    }
    struct Slot<'a>(&'a std::sync::atomic::AtomicBool);
    impl Drop for Slot<'_> {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Release);
        }
    }
    let _slot = Slot(&inner.compaction_outstanding);

    let manifest = inner.manifest.load_full();
    let incoming_key = crate::supertable::manifest::partition::encode_partition_key(
        &crate::supertable::manifest::partition::PartitionKey::VectorCell(
            super::handle::INCOMING_VECTOR_CELL,
        ),
    );
    let incoming: Vec<Arc<SuperfileEntry>> = manifest
        .get_all_superfiles()
        .iter()
        .filter(|e| e.partition_key == incoming_key)
        .cloned()
        .collect();
    if incoming.len() < min_incoming {
        return Ok(());
    }

    let (clusters, column, routing) = match manifest.get_partition_strategy() {
        PartitionStrategy::VectorCell {
            clusters,
            column,
            routing,
        } => (clusters, column, routing),
        _ => return Ok(()),
    };
    if clusters.n_cent == 0 || clusters.dim == 0 {
        return Ok(());
    }
    let Some(vec_col) = inner.options.vector_columns.first().cloned() else {
        return Ok(());
    };
    let metric = vec_col.metric;

    let store = inner.options.store.clone();
    let disk_cache = inner.options.disk_cache.clone();
    let storage_opt = inner.options.storage.clone();
    let storage = storage_opt
        .clone()
        .ok_or_else(|| BuildError::Store("incoming routing requires storage".into()))?;

    // Open every INCOMING superfile first so bytes are mmap-resident before the
    // byte-splice drain (sync fast path); avoids cold subsection GETs on drain.
    // Keep the readers alive: `encode_encoded_rows` borrows their `VectorReader`s.
    let readers: Vec<Arc<SuperfileReader>> = stream::iter(incoming.iter().map(|entry| {
        let entry = Arc::clone(entry);
        let store = Arc::clone(&store);
        let disk_cache = disk_cache.clone();
        let storage_opt = storage_opt.clone();
        async move {
            open_incoming_superfile_for_drain(
                &store,
                disk_cache.as_ref(),
                storage_opt.as_ref(),
                &entry,
            )
            .await
        }
    }))
    .buffered(commit_write_concurrency())
    .collect::<Vec<_>>()
    .await
    .into_iter()
    .collect::<Result<Vec<_>, BuildError>>()?;

    // Incoming superfiles are streaming-built and carry NO inline `_id` region,
    // so the byte-splice cannot read stable ids from the subsection bytes.
    // Resolve each reader's per-local stable `_id` here (scalar `_id` column /
    // span arithmetic — the same path the query remap uses), in the SAME order
    // as `readers`, and hand them to the splice below. This await must happen
    // before the sync `writer_pool.install` scope, which cannot await.
    let mut stable_ids_per_input: Vec<Vec<i128>> = Vec::with_capacity(readers.len());
    for (entry, reader) in incoming.iter().zip(readers.iter()) {
        let ids = stable_ids_by_local_for_routing(&manifest, entry.as_ref(), reader.as_ref())
            .await
            .map_err(|e| BuildError::Store(e.to_string()))?;
        stable_ids_per_input.push(ids);
    }

    // Assign rows to cells by byte-splicing each routed Sq8+ε row verbatim into
    // its destination cell's subsection — no re-quantization, so recall is
    // preserved exactly. The materialized rebuild path is bypassed entirely.
    let inner = Arc::clone(&inner);
    let column_name = column.clone();
    let (prepared, cell_updates, radii_updates): (
        Vec<PreparedSuperfile>,
        HashMap<u32, u32>,
        HashMap<u32, f32>,
    ) = inner
        .options
        .writer_pool
        .install(|| -> Result<_, BuildError> {
            let mut merge_inputs: Vec<(&VectorReader, &str)> = Vec::with_capacity(readers.len());
            for reader in &readers {
                let v = reader.vec().ok_or_else(|| {
                    BuildError::Store("incoming superfile missing vector index".into())
                })?;
                merge_inputs.push((v, column_name.as_str()));
            }

            // Route each row to its cell and report the member's distance to that
            // cell's centroid. Pure (no shared state) so `encode_encoded_rows` can
            // run the routing pass in parallel on the ambient pool (this whole block
            // is under `writer_pool.install`, the half-cores pool); it reduces these
            // distances into a per-cell max radius carried on each returned
            // `RoutedCellSubsection`.
            // SPANN closure replication on the main drain path: a boundary row
            // lands in a few near cells (RNG-pruned, capped) so a capped query
            // probe still finds it — no uncapped read. The reassign path
            // (`assign_among`, split rebalance) stays single-winner. fp32
            // centroids precomputed once for the RNG prune.
            let cent_fp32: Vec<Vec<f32>> = clusters
                .to_encoded_rows()
                .iter()
                .map(|r| manifest_centroid_components_from_row(r, clusters.dim as usize))
                .collect();
            let route = |row: &EncodedCellRow| -> Vec<(u32, f32)> {
                let cells = if let Some(cands) = assign_among {
                    vec![spfresh::nearest_among_cells_encoded(&clusters, metric, cands, row)]
                } else {
                    spfresh::spann_replica_cells_encoded(
                        &clusters,
                        metric,
                        row,
                        &cent_fp32,
                        spfresh::DRAIN_REPLICA_CLOSURE_RATIO,
                        spfresh::DRAIN_MAX_REPLICAS,
                    )
                };
                cells
                    .into_iter()
                    .map(|cell| {
                        let dist = spfresh::encoded_shard_radius(
                            &clusters,
                            metric,
                            cell,
                            slice::from_ref(row),
                        );
                        (cell, dist)
                    })
                    .collect()
            };
            let routed = encode_encoded_rows(&merge_inputs, &stable_ids_per_input, route)?;

            let mut prepared = Vec::with_capacity(routed.len());
            let mut cell_updates = HashMap::new();
            let mut radii_updates = HashMap::new();
            for (cell_id, routed_cell) in routed {
                let added = routed_cell.subsection.n_docs;
                // Read before the value is consumed by the splice-publish below.
                let shard_radius = routed_cell.shard_radius;
                let p = build_prepared_ivf_from_spliced(&inner, cell_id, routed_cell)?;
                let base = clusters.counts.get(cell_id as usize).copied().unwrap_or(0);
                if shard_radius > 0.0 {
                    radii_updates.insert(cell_id, shard_radius);
                }
                cell_updates.insert(cell_id, base.saturating_add(added));
                prepared.push(p);
            }
            Ok((prepared, cell_updates, radii_updates))
        })?;
    if prepared.is_empty() {
        return Ok(());
    }
    let batch = collect_prepared_superfiles(&inner, prepared)?;

    // Bump per-cell counts so routing sees the now-populated cells.
    let updated_clusters = spfresh::apply_cell_updates(&clusters, &cell_updates, &radii_updates);
    // Each new cell superfile becomes one whole-partition leaf in the OPANN
    // routing tree: `(doc_off, count) = (0, 0)` (fetch the whole partition, scan
    // inside), keyed to the committed superfile UUID, with the cell's Sq8
    // centroid decoded transiently to fp32.
    let dim = updated_clusters.dim as usize;
    let cell_rows = updated_clusters.to_encoded_rows();
    let added: Vec<PartitionRoutingCopy> = batch
        .new_entries
        .iter()
        .filter_map(|e| {
            let cell = e.partition_hint? as usize;
            cell_rows.get(cell).map(|row| PartitionRoutingCopy {
                superfile_id: e.superfile_id.as_u128(),
                doc_off: 0,
                count: 0,
                cluster_id: 0,
                centroid_fp32: manifest_centroid_components_from_row(row, dim),
                radius: updated_clusters.radii.get(cell).copied().unwrap_or(0.0),
            })
        })
        .collect();
    inner
        .manifest
        .store(Arc::new(inner.manifest.load().with_partition_strategy(
            PartitionStrategy::VectorCell {
                column,
                clusters: updated_clusters,
                routing,
            },
        )));

    // Copy-on-write-update the OPANN routing tree with the new cell leaves (or
    // build the genesis tree when there is no prior root). The changed pages +
    // new root ride the commit's blob wave + pointer flip below.
    let current = inner.manifest.load_full();
    let opann_commit = opann_routing_update(&inner, &current, &[], &added).await?;

    // Publish: add the cell superfiles, remove the routed incoming superfiles,
    // and stamp the new routing root.
    let new_manifest = persist_commit_async(
        &inner,
        Arc::clone(&storage),
        batch.new_entries,
        &incoming,
        batch.pending_storage_writes,
        Vec::new(),
        opann_commit,
    )
    .await
    .map_err(|e| BuildError::Store(e.to_string()))?;
    inner.manifest.store(Arc::new(new_manifest));

    if let Some(cache) = inner.options.disk_cache.as_ref() {
        warm_cache_inserts(cache, batch.pending_cache_inserts).await;
    }
    if let (Some(cache), Some(budget)) = (
        inner.options.disk_cache.as_ref(),
        inner.options.memory_budget_bytes,
    ) {
        cache.sweep_for_budget(budget);
    }

    schedule_background_storage_reclaim(Arc::clone(&inner));
    Ok(())
}

/// Load Sq8+ε IVF rows from one cell superfile (no fp32 reconstruction).
async fn load_materialized_rows_from_ivf_superfile(
    inner: &SupertableInner,
    entry: &Arc<SuperfileEntry>,
    column: &str,
    now: time::Instant,
) -> Result<Vec<MaterializedIvfRow>, BuildError> {
    let storage = inner
        .options
        .storage
        .as_ref()
        .ok_or_else(|| BuildError::Store("cell maintenance requires storage".into()))?;
    let store = inner.options.store.clone();
    let disk_cache = inner.options.disk_cache.as_ref();

    let bitmap = inner
        .tombstone_cache
        .as_ref()
        .map(|t| t.bitmap_for(entry.superfile_id, now))
        .transpose()
        .map_err(|e| BuildError::Store(e.to_string()))?;

    let reader = open_reader(&store, disk_cache, Some(storage), entry)
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;

    let manifest = inner.manifest.load_full();
    let stable_ids = stable_ids_by_local_for_routing(&manifest, entry, &reader)
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;
    let vec_reader = reader
        .vec()
        .ok_or_else(|| BuildError::Store("IVF cell superfile missing vector index".into()))?;
    materialized_ivf_rows_in_doc_order(vec_reader, column, &stable_ids, bitmap.as_deref()).await
}

/// Build one Sq8 IVF superfile via the normal superfile/vector builder.
fn build_prepared_ivf_from_materialized(
    inner: &SupertableInner,
    partition_hint: u32,
    rows: Vec<MaterializedIvfRow>,
) -> Result<PreparedSuperfile, BuildError> {
    if rows.is_empty() {
        return Err(BuildError::NoDocsToBuild);
    }
    let shard = build_one_shard_from_materialized(&rows, &inner.options, VectorLayout::Ivf)?;
    let prepared = prepare_superfile(inner, shard)?.ok_or(BuildError::NoDocsToBuild)?;
    let entry = finish_superfile_entry(inner, prepared.entry, Some(partition_hint))?;
    Ok(PreparedSuperfile {
        entry,
        bytes_for_store: prepared.bytes_for_store,
        bytes_for_storage: prepared.bytes_for_storage,
        bytes_for_cache: prepared.bytes_for_cache,
    })
}

/// Build one cell's Sq8 IVF superfile from a byte-spliced subsection — no
/// re-quantization. The IVF subsection (codes + rerank bytes) is injected
/// verbatim via [`SuperfileBuilder::set_prebuilt_ivf_subsection`]; the scalar
/// `_id` batch is written in the same `local_doc_id` order the subsection used,
/// so the superfile's id pages line up with the IVF rows.
fn build_one_shard_from_spliced(
    routed: RoutedCellSubsection,
    options: &SupertableOptions,
) -> Result<ShardOutput, BuildError> {
    let RoutedCellSubsection {
        subsection,
        stable_ids,
        // The drain reads `shard_radius` / `cluster_radii` off the value before
        // this point; the per-shard build only needs the subsection + id order.
        shard_radius: _,
        cluster_radii: _,
    } = routed;
    if stable_ids.is_empty() {
        return Err(BuildError::NoDocsToBuild);
    }
    let id_array = Decimal128Array::from_iter_values(stable_ids.iter().copied())
        .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
        .expect("invariant: precision 38 + scale 0 always valid for any i128 payload");
    let scalar = RecordBatch::try_new(
        options.scalar_schema(),
        vec![Arc::new(id_array) as ArrayRef],
    )
    .map_err(|_| BuildError::BatchSchemaMismatch)?;

    let mut builder = SuperfileBuilder::new(
        options
            .builder_options()
            .with_vector_layout(VectorLayout::Ivf),
    )?;
    builder.add_batch_ids_only(&scalar)?;
    builder.set_prebuilt_ivf_subsection(0, subsection)?;

    let id_min = stable_ids.iter().copied().min().unwrap_or(0);
    let id_max = stable_ids.iter().copied().max().unwrap_or(0);
    let n_docs = stable_ids.len() as u64;
    let scalar_stats = ScalarStatsAgg::from_batches(&options.scalar_schema(), &[&scalar]);
    let bytes = Bytes::from(builder.finish()?);

    Ok(ShardOutput {
        bytes,
        n_docs,
        id_min,
        id_max,
        scalar_stats,
    })
}

/// Build one cell's Sq8 IVF superfile from a byte-spliced subsection and finish
/// its manifest entry (carrying the cell partition hint). Drain fast path:
/// preserves rerank bytes verbatim, so recall is unchanged.
fn build_prepared_ivf_from_spliced(
    inner: &SupertableInner,
    partition_hint: u32,
    mut routed: RoutedCellSubsection,
) -> Result<PreparedSuperfile, BuildError> {
    // Per-output-cluster covering radii (taken before the splice consumes
    // `routed`). Carried onto the cell's VectorSummary so the within-cell
    // `select_cells_adaptive` admission is radius-aware; round-trips through the
    // manifest encoding when populated.
    let cluster_radii = std::mem::take(&mut routed.cluster_radii);
    let shard = build_one_shard_from_spliced(routed, &inner.options)?;
    let prepared = prepare_superfile_with_uri(inner, shard, None, &cluster_radii)?
        .ok_or(BuildError::NoDocsToBuild)?;
    let entry = finish_superfile_entry(inner, prepared.entry, Some(partition_hint))?;
    Ok(PreparedSuperfile {
        entry,
        bytes_for_store: prepared.bytes_for_store,
        bytes_for_storage: prepared.bytes_for_storage,
        bytes_for_cache: prepared.bytes_for_cache,
    })
}

/// Same as [`build_one_shard_with_layout`] but feeds Sq8+ε materialized IVF rows
/// into the normal vector builder — no fp32 corpus decode.
fn build_one_shard_from_materialized(
    rows: &[MaterializedIvfRow],
    options: &SupertableOptions,
    vector_layout: crate::superfile::vector::layout::VectorLayout,
) -> Result<ShardOutput, BuildError> {
    let id_array = Decimal128Array::from_iter_values(rows.iter().map(|r| r.stable_id))
        .with_precision_and_scale(
            crate::supertable::options::DECIMAL128_PRECISION,
            crate::supertable::options::DECIMAL128_SCALE,
        )
        .expect("invariant: precision 38 + scale 0 always valid for any i128 payload");
    let scalar = RecordBatch::try_new(
        options.scalar_schema(),
        vec![Arc::new(id_array) as ArrayRef],
    )
    .map_err(|_| BuildError::BatchSchemaMismatch)?;

    let mut builder =
        SuperfileBuilder::new(options.builder_options().with_vector_layout(vector_layout))?;
    builder.add_batch_ids_only(&scalar)?;
    builder.load_materialized_ivf_rows(rows.to_vec())?;

    let id_min = rows.iter().map(|r| r.stable_id).min().unwrap_or(0);
    let id_max = rows.iter().map(|r| r.stable_id).max().unwrap_or(0);
    let n_docs = rows.len() as u64;
    let scalar_stats = ScalarStatsAgg::from_batches(&options.scalar_schema(), &[&scalar]);
    let bytes = Bytes::from(builder.finish()?);

    Ok(ShardOutput {
        bytes,
        n_docs,
        id_min,
        id_max,
        scalar_stats,
    })
}

/// Minimum overflow rows required to split a cell into two sub-cells — a split
/// needs at least one row per side, so fewer than this is a no-op.
const MIN_ROWS_TO_SPLIT_CELL: usize = 2;

/// SPFresh steps 7–9: Sq8-native split, centroid extension, neighborhood
/// reassign, then redrive rows through incoming staging (not direct cell publish).
pub(in crate::supertable) async fn split_overflow_cell_after_compaction(
    inner: Arc<SupertableInner>,
    merged_entry: &Arc<SuperfileEntry>,
    split_cell: u32,
) -> Result<(), BuildError> {
    if !spfresh::split_overflow_needed(merged_entry.n_docs) {
        return Ok(());
    }

    let manifest = inner.manifest.load_full();
    let (clusters, column, routing, metric, _vec_dim) = match manifest.get_partition_strategy() {
        PartitionStrategy::VectorCell {
            clusters,
            column,
            routing,
        } => {
            let Some(vec_col) = inner.options.vector_columns.first() else {
                return Ok(());
            };
            (clusters, column, routing, vec_col.metric, vec_col.dim)
        }
        _ => return Ok(()),
    };
    if clusters.n_cent == 0 || clusters.dim == 0 {
        return Ok(());
    }

    let storage = inner
        .options
        .storage
        .clone()
        .ok_or_else(|| BuildError::Store("cell split requires storage".into()))?;

    let now = time::Instant::now();
    let overflow_materialized =
        load_materialized_rows_from_ivf_superfile(&inner, merged_entry, &column, now).await?;
    if overflow_materialized.len() < MIN_ROWS_TO_SPLIT_CELL {
        return Ok(());
    }
    let overflow_encoded: Vec<EncodedCellRow> = overflow_materialized
        .iter()
        .map(|r| r.encoded.clone())
        .collect();

    let (sub0, sub1) = maint_pool()
        .install(|| spfresh::plan_sq8_split(&overflow_encoded, &clusters, split_cell, metric));
    let mut sub_centroids = sub0;
    sub_centroids.extend_from_slice(&sub1);

    let old_n_cent = clusters.n_cent;
    let (mut updated_clusters, new_cell_id) =
        spfresh::insert_split_centroid(&clusters, metric, split_cell, &sub_centroids);
    let neighborhood = spfresh::reassign_neighborhood(split_cell, old_n_cent, new_cell_id);

    let mut to_remove: Vec<Arc<SuperfileEntry>> = Vec::new();
    for entry in manifest.superfiles.iter() {
        if entry
            .partition_hint
            .is_some_and(|hint| neighborhood.contains(&hint))
        {
            to_remove.push(Arc::clone(entry));
        }
    }

    let mut all_materialized: Vec<MaterializedIvfRow> = Vec::new();
    for entry in &to_remove {
        let mut rows =
            load_materialized_rows_from_ivf_superfile(&inner, entry, &column, now).await?;
        all_materialized.append(&mut rows);
    }
    if all_materialized.is_empty() {
        return Ok(());
    }

    // Rows leave the neighborhood cells; counts reset until routing lands them.
    spfresh::zero_cell_counts(&mut updated_clusters, &neighborhood);

    let incoming_prepared = maint_pool().install(|| -> Result<PreparedSuperfile, BuildError> {
        let mut rows = all_materialized;
        rows.sort_by_key(|r| r.stable_id);
        for (local, row) in rows.iter_mut().enumerate() {
            row.local_doc_id = local as u32;
        }
        build_prepared_ivf_from_materialized(&inner, super::handle::INCOMING_VECTOR_CELL, rows)
    })?;

    let batch = collect_prepared_superfiles(&inner, vec![incoming_prepared])?;

    inner
        .manifest
        .store(Arc::new(manifest.with_partition_strategy(
            PartitionStrategy::VectorCell {
                column: column.clone(),
                clusters: updated_clusters.clone(),
                routing,
            },
        )));

    // Drop the stale neighborhood leaves from the OPANN routing tree: their
    // superfiles are removed by this commit (rows staged to INCOMING), and the
    // new per-cell superfiles are minted — and re-added as leaves keyed by
    // their committed superfile UUID — by the step-9 redrive below. This is the
    // grow-on-split path: each overflow split CoW-edits the tree (N→N+1 cells)
    // instead of inheriting a stale root, so a table can start at a few coarse
    // cells and let `optimize` grow the tree as cells overflow.
    let removed: Vec<u128> = to_remove.iter().map(|e| e.superfile_id.as_u128()).collect();
    let current = inner.manifest.load_full();
    let opann_commit = opann_routing_update(&inner, &current, &removed, &[])
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;

    let new_manifest = persist_commit_async(
        &inner,
        Arc::clone(&storage),
        batch.new_entries,
        &to_remove,
        batch.pending_storage_writes,
        Vec::new(),
        opann_commit,
    )
    .await
    .map_err(|e| BuildError::Store(e.to_string()))?;
    inner.manifest.store(Arc::new(new_manifest));

    schedule_background_storage_reclaim(Arc::clone(&inner));

    // Step 9: redrive through incoming → route into per-cell IVF superfiles.
    // Step 6 neighborhood reassignment: restrict assignment to P−1/P/P₂/P+1.
    route_incoming_to_manifest_cells_if_ready(Arc::clone(&inner), 1, Some(&neighborhood)).await
}

async fn publish_hidden_incoming_async(
    inner: Arc<SupertableInner>,
    storage: Arc<dyn crate::storage::StorageProvider>,
    prep: HiddenIncomingPrepare,
) -> Result<(), BuildError> {
    if prep.batch.new_entries.is_empty() {
        return Ok(());
    }
    let manifest_before = inner.manifest.load();
    let routing = match manifest_before.get_partition_strategy() {
        PartitionStrategy::VectorCell { routing, .. } => routing,
        _ => Default::default(),
    };
    let updated_clusters =
        spfresh::apply_cell_updates(&prep.clusters, &prep.cell_updates, &prep.radii_updates);
    let updated_strategy = PartitionStrategy::VectorCell {
        column: prep.column,
        clusters: updated_clusters,
        routing,
    };
    inner.manifest.store(Arc::new(
        manifest_before.with_partition_strategy(updated_strategy),
    ));
    // We already hold the incoming superfile's bytes here. Warm them onto local
    // disk so a later [`drain_incoming_to_cells`] pass reads them via mmap
    // instead of cold-fetching back from object storage.
    let incoming_writes: Vec<(SuperfileUri, Bytes)> = prep.batch.pending_storage_writes.clone();
    let new_manifest = persist_commit_async(
        &inner,
        Arc::clone(&storage),
        prep.batch.new_entries,
        &prep.batch.to_remove,
        prep.batch.pending_storage_writes,
        Vec::new(),
        OpannRoutingCommit::Inherit,
    )
    .await
    .map_err(|e| BuildError::Store(e.to_string()))?;
    inner.manifest.store(Arc::new(new_manifest));
    // We're already in async context here — warm the disk cache by awaiting
    // directly. (Do NOT use `warm_cache_after_commit`, which does a nested
    // sync `block_in_place` + `block_on`; calling that from inside this commit
    // future deadlocks the runtime.)
    if let Some(cache) = inner.options.disk_cache.as_ref() {
        warm_cache_inserts(cache, incoming_writes).await;
    }
    schedule_background_storage_reclaim(Arc::clone(&inner));
    if let Some(cache) = inner.options.disk_cache.as_ref() {
        warm_cache_inserts(cache, prep.batch.pending_cache_inserts).await;
    }
    if let (Some(cache), Some(budget)) = (
        inner.options.disk_cache.as_ref(),
        inner.options.memory_budget_bytes,
    ) {
        cache.sweep_for_budget(budget);
    }
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
pub(super) fn backoff_delay(attempt: u32) -> time::Duration {
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
    let nanos = time::SystemTime::now()
        .duration_since(time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    let jitter_pct = (nanos % JITTER_MODULUS) as i64 - JITTER_RANGE_PCT;
    let adjusted = ((capped as i64) + (capped as i64 * jitter_pct / PERCENT_DIVISOR)).max(1) as u64;
    time::Duration::from_millis(adjusted)
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
pub(in crate::supertable) async fn persist_commit_async(
    inner: &SupertableInner,
    storage: Arc<dyn StorageProvider>,
    new_entries: Vec<Arc<SuperfileEntry>>,
    entries_to_remove: &[Arc<SuperfileEntry>],
    mut pending_storage_writes: Vec<(SuperfileUri, Bytes)>,
    mut pending_storage_replaces: Vec<(SuperfileUri, Bytes)>,
    opann_routing: OpannRoutingCommit,
) -> Result<Manifest, SupertableCommitError> {
    let storage_async = Arc::clone(&storage);
    let opts = Arc::clone(&inner.options);
    let max_retries = opts.max_commit_retries.max(1);
    let drive = async move {
        let mut last_err: Option<SupertableCommitError> = None;
        for attempt in 0..max_retries {
            let old = inner.manifest.load_full();
            let pending_writes = &mut pending_storage_writes;
            let pending_replaces = &mut pending_storage_replaces;
            match try_commit_attempt(
                Arc::clone(&storage_async),
                Arc::clone(&opts),
                Arc::clone(&old),
                &new_entries,
                entries_to_remove,
                pending_writes,
                pending_replaces,
                &opann_routing,
            )
            .await
            {
                Ok(new_manifest) => return Ok(new_manifest),
                Err(SupertableCommitError::WriteContentionExhausted)
                    if attempt + 1 < max_retries =>
                {
                    refresh_inner_state_async(inner, &storage_async).await?;
                    last_err = Some(SupertableCommitError::WriteContentionExhausted);
                    sleep(backoff_delay(attempt)).await;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap_or(SupertableCommitError::WriteContentionExhausted))
    };
    // Genuinely async: callers `.await` this from async contexts already driven
    // on `query_runtime`. Driving it to completion here with a nested `block_on`
    // would serialize the `tokio::join!` in `commit` (the user + hidden publishes
    // are meant to overlap) and risk a nested-block_on panic. The sync→async
    // bridge lives only in the `persist_commit` wrapper below.
    drive.await
}

pub(in crate::supertable) fn persist_commit(
    inner: &SupertableInner,
    storage: Arc<dyn StorageProvider>,
    new_entries: Vec<Arc<SuperfileEntry>>,
    entries_to_remove: &[Arc<SuperfileEntry>],
    pending_storage_writes: Vec<(SuperfileUri, Bytes)>,
    pending_storage_replaces: Vec<(SuperfileUri, Bytes)>,
) -> Result<(), SupertableCommitError> {
    let drive = persist_commit_async(
        inner,
        storage,
        new_entries,
        entries_to_remove,
        pending_storage_writes,
        pending_storage_replaces,
        OpannRoutingCommit::Inherit,
    );
    let new_manifest = bridge_on_runtime(drive, &inner.query_runtime())?;
    inner.manifest.store(Arc::new(new_manifest));
    Ok(())
}

/// Record `new_deleted` user `_id`s into the hidden index's resident deleted
/// set and persist the change. Loads the prior set, unions `new_deleted` in,
/// writes a new content-addressed blob, and swaps `OpannRouting` — inheriting
/// the routing tree (root + resident bundle), changing only the deleted-set
/// blob reference.
///
/// The hidden cells are NOT rewritten on a user delete (the drain byte-splices
/// every incoming row, tombstoned or not — `encode_encoded_rows` is passed
/// `None` for incoming routing), so deleted rows stay physically present in the
/// cells. This resident set is what the vector read path consults — in memory,
/// zero per-cell tombstone GETs — to drop them. It is monotonic until a
/// tombstone-aware hidden compaction physically removes the rows and prunes
/// their ids.
///
/// No-op pre-drain (no routing tree to attach the set to yet) and when no id is
/// actually new. The hidden index is single-writer, so the load→union→swap is
/// not racing another hidden commit; `persist_commit_async`'s OCC loop still
/// guards the pointer CAS against other processes.
async fn record_hidden_deleted_ids(
    inner: &SupertableInner,
    new_deleted: &[i128],
) -> Result<(), BuildError> {
    if new_deleted.is_empty() {
        return Ok(());
    }
    let Some(storage) = inner.options.storage.clone() else {
        return Ok(());
    };
    let current = inner.manifest.load_full();
    let Some(prior) = current.opann_routing() else {
        return Ok(());
    };
    let mut ids = store::load_deleted_ids(
        prior,
        storage.as_ref(),
        inner.options.disk_cache.as_ref(),
    )
    .await
    .map_err(|e| BuildError::Store(e.to_string()))?;
    let before = ids.len();
    ids.extend_from_slice(new_deleted);
    ids.sort_unstable();
    ids.dedup();
    if ids.len() == before {
        // Every id was already recorded — nothing to persist.
        return Ok(());
    }
    let bytes = store::encode_deleted_ids(&ids);
    let hash = ContentHash::of(&bytes);
    let uri = store::deleted_ids_uri(&hash);
    let mut routing = prior.clone();
    routing.deleted_ids_uri = Some(uri.clone());
    routing.deleted_ids_content_hash = Some(hash);
    // Replace carrying the unchanged tree fields + the new deleted-set blob;
    // the blob travels as a content-addressed page write in the pre-pointer
    // wave, exactly like a routing page.
    let commit = OpannRoutingCommit::Replace {
        routing: Some(routing),
        pages: vec![(uri, Bytes::from(bytes))],
    };
    let new_manifest =
        persist_commit_async(inner, storage, Vec::new(), &[], Vec::new(), Vec::new(), commit)
            .await
            .map_err(|e| BuildError::Store(e.to_string()))?;
    inner.manifest.store(Arc::new(new_manifest));
    Ok(())
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
async fn put_superfile_replace(
    storage: &Arc<dyn StorageProvider>,
    path: &str,
    bytes: Bytes,
) -> Result<(), StorageError> {
    match storage.head(path).await {
        Ok(meta) => storage
            .put_if_match(path, bytes, meta.etag.as_deref())
            .await
            .map(|_| ()),
        Err(StorageError::NotFound { .. }) => storage.put_atomic(path, bytes).await.map(|_| ()),
        Err(e) => Err(e),
    }
}

/// Commit-time object-store write fanout width: half the machine's CPU
/// parallelism, floored at 1. A single commit and a concurrent background
/// maintenance compaction each fan out their PUTs at this width, so keeping
/// each at ~50% of cores bounds the combined in-flight PUTs to roughly the
/// core count rather than a multiple of it.
fn commit_write_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get() / 2)
        .unwrap_or(1)
        .max(1)
}

pub async fn write_superfile_list(
    storage: &Arc<dyn StorageProvider>,
    opts: &Arc<SupertableOptions>,
    pending_storage_writes: &mut Vec<(SuperfileUri, Bytes)>,
    pending_storage_replaces: &mut Vec<(SuperfileUri, Bytes)>,
) -> Result<(), SupertableCommitError> {
    // Bound object-store fanout to half the machine's CPU parallelism. A vector
    // commit can stage one hidden delta per touched cell plus user shards;
    // driving all PUTs at once opens dozens of sockets and can stall the commit
    // path. Crucially, bulk ingest commits overlap background hidden-index
    // SPFresh maintenance (its own compaction PUT/GET waves), so a full-width
    // fanout from each stacks and starves the connection pool until requests
    // hit the per-request timeout. Capping each operation at ~50% of cores
    // leaves headroom for a concurrent maintenance pass without saturation.
    let write_concurrency = commit_write_concurrency();

    let replace_futs = pending_storage_replaces
        .iter()
        .enumerate()
        .map(|(i, (uri, bytes))| {
            let storage = Arc::clone(storage);
            let uri = *uri;
            let bytes = bytes.clone();
            async move {
                let path = superfile_storage_path(&uri);
                put_superfile_replace(&storage, &path, bytes)
                    .await
                    .map(|()| i)
                    .map_err(SupertableCommitError::from)
            }
        });
    let mut err = None;
    let mut successful_replace_idx = Vec::with_capacity(pending_storage_replaces.len());
    for r in stream::iter(replace_futs)
        .buffer_unordered(write_concurrency)
        .collect::<Vec<_>>()
        .await
    {
        match r {
            Ok(i) => successful_replace_idx.push(i),
            Err(e) => err = Some(e),
        }
    }
    successful_replace_idx.sort_unstable_by(|a, b| b.cmp(a));
    for idx in successful_replace_idx {
        pending_storage_replaces.remove(idx);
    }
    if let Some(e) = err {
        return Err(e);
    }

    let multipart_threshold = opts.put_multipart_threshold_bytes;
    let put_futs = pending_storage_writes
        .iter()
        .enumerate()
        .map(|(i, (uri, bytes))| {
            let storage = Arc::clone(storage);
            let uri = *uri;
            let bytes = bytes.clone();
            async move {
                let path = superfile_storage_path(&uri);
                let result = if (bytes.len() as u64) >= multipart_threshold {
                    put_superfile_multipart(storage.as_ref(), &path, bytes.clone()).await
                } else {
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
    let mut successful_writes_idx = Vec::with_capacity(pending_storage_writes.len());

    for r in stream::iter(put_futs)
        .buffer_unordered(write_concurrency)
        .collect::<Vec<_>>()
        .await
    {
        match r {
            Ok(i) => successful_writes_idx.push(i),
            Err(e) => err = Some(e),
        }
    }

    successful_writes_idx.sort_unstable_by(|a, b| b.cmp(a));
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
    storage: Arc<dyn StorageProvider>,
    opts: Arc<SupertableOptions>,
    current_manifest: Arc<Manifest>,
    new_entries: &[Arc<SuperfileEntry>],
    entries_to_remove: &[Arc<SuperfileEntry>],
    pending_storage_writes: &mut Vec<(SuperfileUri, Bytes)>,
    pending_storage_replaces: &mut Vec<(SuperfileUri, Bytes)>,
    opann_routing: &OpannRoutingCommit,
) -> Result<Manifest, SupertableCommitError> {
    // 1. Write each new superfile's bytes to storage in parallel.
    write_superfile_list(
        &storage,
        &opts,
        pending_storage_writes,
        pending_storage_replaces,
    )
    .await?;

    // 2. update the manifest for the commit.
    let (new_manifest, parts_to_write) = current_manifest
        .update(new_entries, entries_to_remove)
        .await?;

    // 2a. Apply the OPANN routing update: stamp the new routing root into the
    //     committed manifest and collect the changed routing-tree pages, written
    //     below as content-addressed immutable blobs before the pointer flip.
    let (new_manifest, page_blobs): (Manifest, &[(String, Bytes)]) = match opann_routing {
        OpannRoutingCommit::Inherit => (new_manifest, &[]),
        OpannRoutingCommit::Replace { routing, pages } => {
            (new_manifest.with_opann_routing(routing.clone()), pages.as_slice())
        }
    };

    // 3. Read the prior pointer's etag for the CAS. Fresh
    //    supertable → no pointer yet → None etag (initial
    //    commit).
    let prev_etag = get_current_manifest_etag(&storage, current_manifest).await?;

    // 3a. Write the changed routing-tree pages — durable before the pointer that
    //     references the new root. Content-addressed; benign collision on retry.
    for (uri, bytes) in page_blobs {
        put_immutable_blob(storage.as_ref(), uri, bytes.clone())
            .await
            .map_err(SupertableCommitError::Storage)?;
        // Pre-warm the disk cache with the bytes we just wrote, so the first
        // post-commit query (and the next incremental tree update) reads the
        // bundle/pages from the warm mmap instead of re-GETting them. Under
        // heavy write traffic every commit mints a new root/bundle hash, so
        // without this each post-commit read degenerates to an object-store
        // GET. Best-effort: the blob is already durable on storage, so a seed
        // failure costs only a later cold GET, never correctness.
        if let Some(cache) = opts.disk_cache.as_ref()
            && let Err(e) = cache.seed_blob(ContentHash::of(bytes.as_ref()), bytes.clone()).await
        {
            tracing::debug!("opann routing-tree blob cache pre-warm failed for {uri}: {e}");
        }
    }

    // 4. Parallel-issue (touched parts) + list PUTs, then
    //    conditional pointer PUT (the visibility barrier).
    //    Untouched parts are NOT re-PUT — their URIs (and
    //    content-hashes) are unchanged in the new list.
    let encoded_refs: Vec<&[u8]> = parts_to_write
        .iter()
        .map(|ep| ep.encoded.as_slice())
        .collect();
    new_manifest
        .write(storage.as_ref(), prev_etag.as_deref(), &encoded_refs)
        .await?;
    // Silence the unused-import warning when no path uses
    // `PartId` / `part_mod` directly (helpers consume them
    // from inside `build_part_and_entry`).
    let _ = PhantomData::<(PartId, part_mod::ContentHash)>;

    Ok(new_manifest)
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
    storage: &Arc<dyn StorageProvider>,
) -> Result<(), SupertableCommitError> {
    let current = inner.manifest.load_full();
    let manifest = match Manifest::load(Some(current), storage.clone(), None).await {
        Ok(manifest) => manifest,
        Err(ManifestLoadError::PointerNotFound) => return Ok(()),
        Err(ManifestLoadError::AlreadyLoaded) => return Ok(()),
        Err(err) => {
            return Err(SupertableCommitError::ManifestError(
                ManifestError::ManifestLoadError(err),
            ));
        }
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
fn encode_record_batch_ipc(batch: &RecordBatch) -> Result<Bytes, String> {
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
    storage: &dyn StorageProvider,
    path: &str,
    bytes: Bytes,
) -> Result<(), StorageError> {
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
    let mut parts: Vec<UploadPart> = Vec::with_capacity(total / PART_BYTES + 1);
    let mut offset = 0;
    while offset < total {
        let end = cmp::min(offset + PART_BYTES, total);
        let chunk = bytes.slice(offset..end);
        parts.push(upload.put_part(PutPayload::from_bytes(chunk)));
        offset = end;
    }
    // Drive part-uploads concurrently. `try_join_all` cancels
    // remaining parts if one fails — semantically equivalent to
    // abandoning the upload, with `abort()` below as cleanup.
    if let Err(e) = try_join_all(parts).await {
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

/// After a successful compaction manifest commit: warm-insert the merged
/// output into the disk cache and schedule deferred reclaim of superseded
/// superfiles. Superseded cache entries are left to the LRU — they are no
/// longer manifest-visible and will age out.
pub(in crate::supertable) async fn finalize_compaction_commit(
    inner: Arc<SupertableInner>,
    _storage: &Arc<dyn crate::storage::StorageProvider>,
    _new_entries: &[Arc<SuperfileEntry>],
    _entries_to_remove: &[Arc<SuperfileEntry>],
    pending_cache_inserts: Vec<(SuperfileUri, Bytes)>,
) {
    schedule_background_storage_reclaim(Arc::clone(&inner));
    if !pending_cache_inserts.is_empty()
        && let Some(cache) = inner.options.disk_cache.as_ref().cloned()
    {
        warm_cache_after_commit(&inner, &cache, pending_cache_inserts);
    }
    if let (Some(cache), Some(budget)) = (
        inner.options.disk_cache.as_ref(),
        inner.options.memory_budget_bytes,
    ) {
        cache.sweep_for_budget(budget);
    }
}

/// Pre-populate the warm cache with each just-published superfile's bytes.
///
/// Best-effort: each failure is swallowed with a tracing warning — the
/// superfiles are already durable in storage and the manifest commit has
/// succeeded, so a cache miss becomes a cold-fetch on first read, not a
/// correctness break. Shared by every commit/route finalize path so the
/// loop + warning text live in one place.
async fn warm_cache_inserts(cache: &Arc<DiskCacheStore>, inserts: Vec<(SuperfileUri, Bytes)>) {
    for (uri, bytes) in inserts {
        if let Err(e) = cache.insert_warm(&uri, bytes).await {
            tracing::warn!(
                "supertable: warm cache pre-population failed for {}: {} \
                 (superfile is durable in storage; first query will cold-fetch)",
                uri.0,
                e
            );
        }
    }
}

/// Sync entry point for [`warm_cache_inserts`]: drives it on `query_runtime`
/// via the shared [`bridge_on_runtime`] bridge (the disk cache's async
/// coordination is bound to that runtime).
fn warm_cache_after_commit(
    inner: &SupertableInner,
    cache: &Arc<DiskCacheStore>,
    pending: Vec<(SuperfileUri, Bytes)>,
) {
    let cache = Arc::clone(cache);
    bridge_on_runtime(warm_cache_inserts(&cache, pending), &inner.query_runtime());
}

pub(crate) fn read_vector_layout_from_bytes(bytes: &Bytes) -> VectorLayout {
    match read_kv_metadata(bytes.as_ref()) {
        Ok(kvs) => vector_layout_from_kv(&kvs),
        Err(_) => VectorLayout::Ivf,
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Instant};

    use arrow_array::{FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use figment::{
        Figment,
        providers::{Format, Yaml},
    };
    use rayon::ThreadPoolBuilder;
    use tempfile::TempDir;

    use super::*;
    use crate::{
        config::Config,
        superfile::{
            builder::{FtsConfig, VectorConfig},
            fts::reader::BoolMode,
            vector::{distance::Metric, rerank_codec::RerankCodec},
        },
        supertable::{SupertableOptions, handle::Supertable, storage::LocalFsStorageProvider},
        test_helpers::default_tokenizer as tok,
    };

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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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
        assert!(fts.may_contain(b"alpha"));
        assert!(fts.may_contain(b"doc"));
        // Lex range should be present and consistent.
        let (min_term, max_term) = fts.term_range.as_ref().expect("non-empty FST has a range");
        assert!(!min_term.is_empty());
        assert!(!max_term.is_empty());
        assert!(min_term <= max_term, "min_term <= max_term invariant");
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
                rerank_codec: RerankCodec::Fp32,
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
        assert_eq!(vs.centroid.dim as usize, dim);
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
        assert_eq!(vs.clusters.scale.len(), dim);
        assert_eq!(vs.clusters.offset.len(), dim);
        assert!(!vs.clusters.rows.is_empty());
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
        let cluster_idx_bytes = (n_cent * CLUSTER_IDX_ENTRY_BYTES) as u64;

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
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(yaml))).expect("parse config");

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
            let t0 = Instant::now();
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

    // ---- merge_ranges helper -----------------------------------------

    #[test]
    fn merge_ranges_coalesces_overlapping_and_adjacent_drops_empty() {
        // (off, len) inputs: an empty range (dropped), two
        // overlapping ranges (coalesced), one adjacent range
        // (coalesced, since `off <= last_end`), and one disjoint
        // range (kept separate). Unsorted on input.
        let input = vec![
            (100u64, 10u64), // disjoint, far away
            (0, 0),          // empty — dropped
            (10, 10),        // [10,20)
            (15, 10),        // [15,25) overlaps prior → [10,25)
            (25, 5),         // [25,30) adjacent → [10,30)
        ];
        let merged = merge_ranges(input);
        assert_eq!(merged, vec![(10, 20), (100, 10)]);
    }

    #[test]
    fn merge_ranges_empty_input_is_empty() {
        assert!(merge_ranges(Vec::new()).is_empty());
    }

    // ---- build_subsection_offsets on real superfile bytes ------------

    #[test]
    fn build_subsection_offsets_captures_total_size_and_fts_range() {
        // A freshly-built FTS superfile should produce subsection
        // offsets: total_size matches the byte length and the FTS
        // open ranges are non-empty (there's an FTS index).
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 8)).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let seg = &r.manifest().superfiles[0];
        let store = &st.options().store;
        // Fetch the bytes back from the in-memory store.
        let reader = store.reader(&seg.uri).expect("reader");
        // Confirm the manifest already carries subsection offsets and
        // that total_size is plausible (> 0).
        let offsets = seg
            .subsection_offsets
            .as_ref()
            .expect("offsets captured at commit");
        assert!(offsets.total_size > 0);
        assert!(
            offsets.fts.is_some(),
            "an FTS superfile must record an FTS subsection"
        );
        assert!(
            !offsets.fts_open_ranges.is_empty(),
            "FTS open ranges should be populated for the cold-open fast path"
        );
        // n_docs sanity via the reader, ensuring the bytes parse.
        assert_eq!(reader.n_docs(), 8);
    }

    #[test]
    fn build_subsection_offsets_on_garbage_returns_none() {
        // Bytes that aren't a valid superfile (no parquet footer)
        // must fall back to None rather than panic.
        let garbage = Bytes::from_static(b"not a parquet file at all");
        assert!(build_subsection_offsets(&garbage).is_none());
    }

    // ---- vector append path ------------------------------------------

    #[test]
    fn append_with_vector_column_publishes_superfile() {
        // Drive the vector branch of `append` (the FixedSizeList
        // downcast + Arc<Float32Array> buffering).
        let dim = 16;
        let st = Supertable::create(options_with_vector(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 8, dim)).expect("append");
        assert!(
            w.buffered_bytes() > 0,
            "buffered_bytes must account for the vector payload"
        );
        w.commit().expect("commit");

        let r = st.reader();
        assert_eq!(r.n_superfiles(), 1);
        assert_eq!(r.n_docs_total(), 8);
    }

    // ---- end-to-end update / delete through Supertable ----------------

    /// A storage-backed supertable, required for the WAL-driven
    /// update/delete pipeline.
    fn storage_backed_st(dir: &TempDir) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        Supertable::create(options_id_title_serial().with_storage(storage)).expect("create")
    }

    fn row(title: &str) -> RecordBatch {
        RecordBatch::try_new(
            schema_id_title(),
            vec![Arc::new(LargeStringArray::from(vec![title]))],
        )
        .expect("row batch")
    }

    #[test]
    fn delete_tombstones_matching_row() {
        use datafusion::prelude::{col, lit};
        let dir = TempDir::new().expect("tempdir");
        let st = storage_backed_st(&dir);
        st.append(&build_simple_batch(0, 1)).expect("append");
        // build_simple_batch titles are "doc 0 alpha".
        let stats = st
            .delete(col("title").eq(lit("doc 0 alpha")))
            .expect("delete");
        assert_eq!(stats.matched(), 1);
        assert_eq!(stats.n_tombstoned(), 1);
    }

    #[test]
    fn delete_unmatched_predicate_is_noop() {
        use datafusion::prelude::{col, lit};
        let dir = TempDir::new().expect("tempdir");
        let st = storage_backed_st(&dir);
        st.append(&build_simple_batch(0, 1)).expect("append");
        let stats = st
            .delete(col("title").eq(lit("no such title")))
            .expect("delete");
        assert_eq!(stats.matched(), 0);
        assert_eq!(stats.n_tombstoned(), 0);
    }

    #[test]
    fn update_replaces_matching_row() {
        use datafusion::prelude::{col, lit};
        let dir = TempDir::new().expect("tempdir");
        let st = storage_backed_st(&dir);
        st.append(&row("draft")).expect("append");
        let stats = st
            .update(col("title").eq(lit("draft")), &row("published"))
            .expect("update");
        assert_eq!(stats.matched(), 1);
        assert_eq!(stats.n_tombstoned(), 1);
    }

    #[test]
    fn update_cardinality_mismatch_is_rejected() {
        use datafusion::prelude::{col, lit};
        let dir = TempDir::new().expect("tempdir");
        let st = storage_backed_st(&dir);
        st.append(&row("draft")).expect("append");
        // Predicate matches one row but new_rows has two — cardinality
        // mismatch surfaces as a typed writer error.
        let two = RecordBatch::try_new(
            schema_id_title(),
            vec![Arc::new(LargeStringArray::from(vec!["a", "b"]))],
        )
        .expect("two-row batch");
        let mut w = st.writer().expect("writer");
        let err = w
            .update(col("title").eq(lit("draft")), two)
            .expect_err("cardinality mismatch");
        assert!(
            matches!(
                err,
                MutationError::CardinalityMismatch {
                    matched: 1,
                    new_rows: 2
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn update_without_storage_is_rejected() {
        use datafusion::prelude::{col, lit};
        // No storage attached → the update pre-flight rejects.
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        let err = w
            .update(col("title").eq(lit("x")), row("y"))
            .expect_err("no storage");
        assert!(matches!(err, MutationError::NoStorageAttached), "{err:?}");
    }

    #[test]
    fn delete_without_storage_is_rejected() {
        use datafusion::prelude::{col, lit};
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        let err = w.delete(col("title").eq(lit("x"))).expect_err("no storage");
        assert!(matches!(err, MutationError::NoStorageAttached), "{err:?}");
    }

    #[test]
    fn buffered_bytes_grows_then_resets_on_commit() {
        let st = Supertable::create(options_id_title_serial()).expect("create");
        let mut w = st.writer().expect("writer");
        assert_eq!(w.buffered_bytes(), 0);
        w.append(&build_simple_batch(0, 4)).expect("append");
        assert!(w.buffered_bytes() > 0, "buffer cost recorded");
        assert_eq!(w.buffered_batches(), 1);
        w.commit().expect("commit");
        assert_eq!(w.buffered_bytes(), 0, "buffer drained on commit");
        assert_eq!(w.buffered_batches(), 0);
    }
}
