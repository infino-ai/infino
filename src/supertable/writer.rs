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

#[cfg(test)]
use std::sync::Mutex as StdMutex;
use std::{
    cmp,
    collections::{HashMap, HashSet, hash_map::Entry},
    env, fmt, fs,
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    marker::PhantomData,
    mem,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, atomic::Ordering},
    time,
};

use arrow::{
    compute::{concat_batches, take},
    ipc::writer::StreamWriter,
};
use arrow_array::{
    Array, ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array,
};
use blake3::Hasher as Blake3Hasher;
use bytes::Bytes;
use chrono::Utc;
use datafusion::prelude::Expr;
use futures::{
    future::try_join_all,
    stream::{self, StreamExt},
};
use object_store::{MultipartUpload, PutPayload, UploadPart};
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;
use tokio::time::sleep;
use tracing::{debug, error};
use uuid::Uuid;

use super::{
    build::fanout_shards,
    error::BuildError,
    handle::{GLOBAL_VECTOR_KMEANS_ITERS, GLOBAL_VECTOR_KMEANS_SEED, Supertable, SupertableInner},
    manifest::{
        CellVectorSummary, FtsSummaryAgg, ManifestSnapshot, ScalarStatsAgg, SubsectionOffsets,
        SuperfileEntry, SuperfileUri, VectorSummary, bloom::BloomBuilder,
    },
    mutations::{
        CommitError, CommitResult, MAX_TARGETS_PER_MUTATION, MutationError, MutationStats,
        PendingDelete, PendingUpdate,
    },
    opann,
    options::{DECIMAL128_PRECISION, DECIMAL128_SCALE, SupertableOptions},
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
    config::{self, CentroidAlignment, DrainConsolidate, ThreadCount},
    memory::{ConnectionMemoryBudget, Reservation},
    runtime_bridge::bridge_on_runtime,
    storage::{StorageError, StorageProvider},
    superfile::{
        BuildError as SuperfileBuildError, SuperfileReader,
        builder::{SuperfileBuilder, VectorConfig},
        format::{
            CRC_BYTES,
            footer::read_kv_metadata,
            fts::{HEADER_SIZE_V1_LEGACY as FTS_HEADER_SIZE, U64_BYTES, hdr},
            kv,
            vec::{
                CELL_DIR_ENTRY_SIZE, CLUSTER_IDX_ENTRY_BYTES, DIR_ENTRY_SIZE, DOC_ID_BYTES,
                OUTER_HEADER_SIZE, STABLE_ID_BYTES, SUB_HEADER_SIZE, U32_BYTES, cell_dir_entry,
                dir_entry, outer_hdr, sub_hdr,
            },
        },
        reader::vector_layout_from_kv,
        vector::{
            builder::{
                MultiCellSubsectionSource, build_merged_subsection_from_fp32,
                build_merged_subsection_from_materialized,
                build_merged_subsection_from_spilled_materialized,
            },
            cell_posting::{EncodedCellRow, MaterializedIvfRow},
            distance::Metric,
            ivf_merge::{
                MergedIvfSubsection, merge_fragment_subsections, route_clusters_into_cells,
            },
            kmeans::kmeans_with_assignments,
            layout::VectorLayout,
            quant::BitQuantizer,
            reader::{VectorColumnConfig, VectorReader},
            rerank_codec::RerankCodec,
            rotation::RandomRotation,
            spill::{MaterializedRowSpillState, MaterializedRowSpillWriter, SpilledCellRows},
        },
    },
    supertable::{
        CommitError as SupertableCommitError, ManifestLoadError,
        error::ManifestError,
        hidden_deleted::{self, encode_deleted_ids},
        manifest::{
            ClusterCentroids, RabitqAdmitContext,
            commit::get_current_manifest_etag,
            list::{CellRoutingParams, DrainedVersionRanges, GlobalVectorIndex, PartitionStrategy},
            options_hash,
            part::{self as part_mod, PartId},
        },
        query::{dispatch::open_reader, vector::stable_ids_by_local_for_routing},
        reader_cache::{DiskCacheStore, disk::mmap_readonly_bytes},
        slow_vector_state,
        slow_vector_state::{CentroidSection, fetch_centroid_section},
    },
};

/// Target bytes per fine IVF run inside one global cell. Fine-centroid count
/// is derived from this target; it is not copied from the outer/global grid or
/// repeated as a fixed count for every small commit delta.
const DRAIN_FINE_RUN_TARGET_BYTES: usize = 2 * 1024 * 1024;
/// Multipart chunk size for large superfile uploads.
const SUPERFILE_MULTIPART_PART_BYTES: usize = 8 * (1 << 20);
/// Stable IDs fed to the streamed shard Parquet builder per Arrow batch.
const DRAIN_ID_BATCH_ROWS: usize = 64 * 1024;
const DRAIN_CHECKPOINT_SCHEMA: u32 = 1;
/// Local checkpoint filename inside one epoch scratch directory.
const DRAIN_LOCAL_CHECKPOINT_FILE: &str = "checkpoint.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DrainCheckpointSource {
    superfile_id: String,
    uri: String,
    birth_version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DrainRemoteShard {
    shard_id: u32,
    superfile_id: String,
    cell_counts: Vec<(u32, u32)>,
}

/// Object-storage state: intentionally small. It preserves completed output
/// shards across node replacement, while unfinished shards are recomputed from
/// immutable user superfiles instead of uploading corpus-sized scratch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DrainRemoteCheckpoint {
    schema: u32,
    epoch_id: String,
    options_hash: String,
    sources: Vec<DrainCheckpointSource>,
    batch_layout: Vec<Vec<u64>>,
    shard_count: usize,
    completed_shards: Vec<DrainRemoteShard>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DrainLocalSpill {
    n_rows: u32,
    n_quants: u32,
    dim: usize,
    rabitq_len: usize,
    rerank_codec_id: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DrainLocalCell {
    n_docs: u32,
    subsection_len: u64,
    rerank_codec_id: u8,
}

/// Same-node state: exact spill offsets at the last completed source batch and
/// completed cell-IVF files. Every update is fsync + atomic rename.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DrainLocalCheckpoint {
    schema: u32,
    epoch_id: String,
    batches_done: usize,
    spills: HashMap<u32, DrainLocalSpill>,
    built_cells: HashMap<u32, DrainLocalCell>,
    added_per_cell: HashMap<u32, u32>,
}

impl DrainLocalCheckpoint {
    fn new(epoch_id: String) -> Self {
        Self {
            schema: DRAIN_CHECKPOINT_SCHEMA,
            epoch_id,
            batches_done: 0,
            spills: HashMap::new(),
            built_cells: HashMap::new(),
            added_per_cell: HashMap::new(),
        }
    }
}

struct DrainRemoteState {
    checkpoint: DrainRemoteCheckpoint,
    entries: Vec<Arc<SuperfileEntry>>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrainTestFailurePhase {
    AfterBatch,
    AfterShard,
}

#[cfg(test)]
struct DrainTestFailure {
    phase: DrainTestFailurePhase,
    completed: usize,
}

#[cfg(test)]
static DRAIN_TEST_FAILURES: StdMutex<Option<HashMap<String, DrainTestFailure>>> =
    StdMutex::new(None);

#[cfg(test)]
fn inject_drain_test_failure(epoch_id: String, phase: DrainTestFailurePhase, completed: usize) {
    let mut guard = DRAIN_TEST_FAILURES.lock().expect("drain test failure lock");
    let map = guard.get_or_insert_with(HashMap::new);
    map.insert(epoch_id, DrainTestFailure { phase, completed });
}

#[cfg(test)]
fn maybe_fail_drain_for_test(
    epoch_id: &str,
    phase: DrainTestFailurePhase,
    completed: usize,
) -> Result<(), BuildError> {
    let mut guard = DRAIN_TEST_FAILURES.lock().expect("drain test failure lock");
    let Some(map) = guard.as_mut() else {
        return Ok(());
    };
    let should_fail = map
        .get(epoch_id)
        .is_some_and(|failure| failure.phase == phase && completed >= failure.completed);
    if should_fail {
        map.remove(epoch_id);
        return Err(BuildError::Store(format!(
            "injected drain failure after {phase:?} {completed}"
        )));
    }
    Ok(())
}

// Approximate multiples for the memory the build will use, reserved up front rather than
// accounted for exactly. Building the superfile holds the FTS and vector blobs plus the
// serialized file in memory at once, so the real peak is a few times the raw ingested bytes.
//
// Each kind of blob has a separate factor, so the estimate tracks the schema & ingestion data
// closely: memory for building vector blobs >> memory for the FTS blob >> memory for plain
// scalar columns.
//
// Stored as numerator over DENOM so the estimate stays integer-only (halves).
const BUILD_SCRATCH_DENOM: usize = 2;

// Scalar columns, held then serialized into the Parquet body: ~2.5x.
const BUILD_SCALAR_NUM: usize = 5;

// f32 vector payload, rebuilt as quantized + rerank codecs alongside the raw input: ~6.5x.
const BUILD_VECTOR_NUM: usize = 13;

// FTS text, ~1.5x for the FST + postings structures. Added on top of the scalar factor, not
// instead of it: the same text bytes are held as a column and drive the index build at once.
const BUILD_FTS_NUM: usize = 3;

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
    /// Held Arrow scalar bytes across `buffer` (id + user columns,
    /// including the FTS text columns).
    buffer_scalar_bytes: usize,
    /// Held f32 vector payload bytes across `buffer`.
    buffer_vector_bytes: usize,
    /// Byte size of the FTS-indexed text columns within `buffer`. A
    /// subset of `buffer_scalar_bytes`, not extra held memory; tracked
    /// only to weight the build-scratch reserve, since the FTS index
    /// structures built at commit scale with the text input.
    buffer_fts_bytes: usize,
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
            .field("buffered_bytes", &self.buffered_bytes())
            .field("manifest_id", &self.inner.manifest.load().manifest_id)
            .finish()
    }
}

/// One buffered append-call payload. Vectors stored as
/// `Arc<Float32Array>` so the buffer owns its data outright;
/// per-shard builders re-derive `&[f32]` slices via
/// [`Float32Array::values`] without copying.
#[derive(Clone)]
struct BufferedBatch {
    scalar: RecordBatch,
    vectors: Vec<Arc<Float32Array>>,
}

/// Zero-copy view of one vector column across the buffered batches:
/// `row(local)` resolves a commit-wide row ordinal to its `&[f32]` slice
/// inside the owning batch's Arrow buffer. Replaces the commit-time
/// flatten, which materialized a full copy of every vector column
/// (12.8 GiB at a 3.125M-row × dim-1024 commit) just to hand out row
/// slices — a peak-RSS driver on top of the buffered batches themselves.
struct VectorColumnView<'a> {
    dim: usize,
    /// Per-batch contiguous values, in buffer order.
    batches: Vec<&'a [f32]>,
    /// `offsets[i]` = first commit-wide row of batch `i`, plus a trailing
    /// total-row sentinel.
    offsets: Vec<usize>,
}

impl<'a> VectorColumnView<'a> {
    fn over(buffer: &'a [BufferedBatch], col_idx: usize, dim: usize) -> Self {
        let mut batches = Vec::with_capacity(buffer.len());
        let mut offsets = Vec::with_capacity(buffer.len() + 1);
        let mut total = 0usize;
        for buffered in buffer {
            offsets.push(total);
            let values: &[f32] = buffered.vectors[col_idx].values();
            total += values.len() / dim.max(1);
            batches.push(values);
        }
        offsets.push(total);
        Self {
            dim,
            batches,
            offsets,
        }
    }

    fn n_rows(&self) -> usize {
        self.offsets.last().copied().unwrap_or(0)
    }

    /// The commit-wide row `local` as a `&[f32]` of length `dim`.
    fn row(&self, local: usize) -> Result<&'a [f32], BuildError> {
        // partition_point returns the first offset > local; its
        // predecessor is the owning batch.
        let batch = self
            .offsets
            .partition_point(|&first_row| first_row <= local)
            .saturating_sub(1);
        let in_batch = local
            .checked_sub(self.offsets[batch])
            .ok_or_else(|| BuildError::Store(format!("vector row {local} before batch start")))?;
        let start = in_batch * self.dim;
        self.batches
            .get(batch)
            .and_then(|values| values.get(start..start + self.dim))
            .ok_or_else(|| BuildError::Store(format!("vector row {local} out of buffered range")))
    }
}

/// Row-balanced split of the writer's buffered batches into
/// `n_shards` shard inputs, each shaped as a `Vec<BufferedBatch>`
/// that [`build_one_shard_with_layout`] can consume directly. The split walks
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
async fn materialized_ivf_rows_in_doc_order(
    vec_reader: &VectorReader,
    column: &str,
    stable_ids_by_local: &[i128],
    tombstones: Option<&roaring::RoaringBitmap>,
) -> Result<Vec<MaterializedIvfRow>, BuildError> {
    let mut rows = vec_reader
        .materialized_index_rows_async(column)
        .await
        .ok_or_else(|| {
            BuildError::Store(format!(
                "IVF maintenance: column '{column}' missing Sq8Residual index"
            ))
        })?;
    let n_rows = stable_ids_by_local.len();
    let mut by_local = vec![None; n_rows];
    for row in &mut rows {
        if tombstones.is_some_and(|bm| bm.contains(row.local_doc_id)) {
            continue;
        }
        let slot = row.local_doc_id as usize;
        if slot < n_rows {
            // Cell superfiles inline the stable `_id` in the IVF blob, so the
            // read-back already carries it (nonzero). Region-less incoming
            // superfiles return 0 here and fall back to the scalar `_id` column
            // resolved into `stable_ids_by_local`.
            if row.stable_id == 0 {
                row.stable_id = stable_ids_by_local[slot];
                row.encoded.stable_id = row.stable_id;
            }
            by_local[slot] = Some(row.clone());
        }
    }
    Ok(by_local
        .into_iter()
        .enumerate()
        .filter_map(|(i, r)| {
            r.map(|mut row| {
                row.local_doc_id = i as u32;
                row
            })
        })
        .collect())
}

/// Split buffered rows into per-cell shards based on nearest centroid.
/// Each shard carries all rows assigned to one cell; the caller stamps
/// `partition_hint` on the resulting superfile entries.
fn split_buffer_by_vector_cell(
    buffer: Vec<BufferedBatch>,
    cells: &ClusterCentroids,
    metric: Metric,
    vec_col_idx: usize,
) -> Result<Vec<(u32, Vec<BufferedBatch>)>, BuildError> {
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
            // Checked: an out-of-range assignment must roll the commit back,
            // not abort the writer.
            per_cell_rows
                .get_mut(cell as usize)
                .ok_or_else(|| {
                    BuildError::Store(format!(
                        "vector-cell split: row {row} assigned to out-of-range cell {cell} (k={k})"
                    ))
                })?
                .push(row);
        }
        for (cell_id, rows) in per_cell_rows.into_iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let indices = UInt32Array::from(rows.iter().map(|&r| r as u32).collect::<Vec<_>>());
            // Propagate instead of panicking: a take/rebuild failure mid-commit
            // must roll the append back cleanly, not abort the process.
            let scalar_cols: Vec<ArrayRef> = (0..batch.scalar.num_columns())
                .map(|col_idx| {
                    take(batch.scalar.column(col_idx), &indices, None).map_err(|e| {
                        BuildError::Store(format!(
                            "vector-cell split: take column {col_idx} for cell {cell_id}: {e}"
                        ))
                    })
                })
                .collect::<Result<_, _>>()?;
            let scalar_batch =
                RecordBatch::try_new(batch.scalar.schema(), scalar_cols).map_err(|e| {
                    BuildError::Store(format!(
                        "vector-cell split: rebuild batch for cell {cell_id}: {e}"
                    ))
                })?;
            let vectors: Vec<Arc<Float32Array>> = batch
                .vectors
                .iter()
                .map(|v| -> Result<Arc<Float32Array>, BuildError> {
                    // One divisibility check bounds the whole loop: rows come
                    // from this batch (r < n_rows), so r*vdim + vdim <= len.
                    if v.len() % n_rows != 0 {
                        return Err(BuildError::Store(format!(
                            "vector-cell split: {} values do not divide across {n_rows} rows",
                            v.len()
                        )));
                    }
                    let vdim = v.len() / n_rows;
                    let mut out = Vec::with_capacity(rows.len() * vdim);
                    for &r in &rows {
                        out.extend_from_slice(&v.values()[r * vdim..(r + 1) * vdim]);
                    }
                    Ok(Arc::new(Float32Array::from(out)))
                })
                .collect::<Result<_, _>>()?;
            cell_batches[cell_id].push(BufferedBatch {
                scalar: scalar_batch,
                vectors,
            });
        }
    }
    Ok(cell_batches
        .into_iter()
        .enumerate()
        .filter(|(_, batches)| !batches.is_empty())
        .map(|(cell_id, batches)| (cell_id as u32, batches))
        .collect())
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
    /// # use infino::arrow_array::{LargeStringArray, RecordBatch};
    /// # use infino::arrow_schema::{DataType, Field, Schema};
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
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(rows = batch.num_rows()))
    )]
    pub fn append(&self, batch: &RecordBatch) -> Result<(), InfinoError> {
        let mut w = self
            .writer()
            .map_err(|e| InfinoError::from(e).with_context("append", None))?;
        w.append(batch)
            .map_err(|e| InfinoError::from(e).with_context("append", None))?;
        w.commit()
            .map_err(|e| InfinoError::from(e).with_context("append", None))?;
        Ok(())
    }

    /// Replace every row matching `predicate` with `new_rows`, then
    /// commit. `new_rows.num_rows()` must equal the match count.
    /// Durable when this returns.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use infino::arrow_array::{LargeStringArray, RecordBatch};
    /// # use infino::arrow_schema::{DataType, Field, Schema};
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
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(new_rows = new_rows.num_rows()))
    )]
    pub fn update(
        &self,
        predicate: Expr,
        new_rows: &RecordBatch,
    ) -> Result<MutationStats, InfinoError> {
        let mut w = self
            .writer()
            .map_err(|e| InfinoError::from(e).with_context("update", None))?;
        w.update(predicate, new_rows.clone())
            .map_err(|e| InfinoError::from(e).with_context("update", None))?;
        single_outcome(
            w.commit()
                .map_err(|e| InfinoError::from(e).with_context("update", None))?,
        )
        .map_err(|e| e.with_context("update", None))
    }

    /// Tombstone every row matching `predicate`, then commit. Durable
    /// when this returns.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use infino::arrow_array::{LargeStringArray, RecordBatch};
    /// # use infino::arrow_schema::{DataType, Field, Schema};
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
    #[cfg_attr(feature = "detailed-tracing", tracing::instrument(skip_all))]
    pub fn delete(&self, predicate: Expr) -> Result<MutationStats, InfinoError> {
        let mut w = self
            .writer()
            .map_err(|e| InfinoError::from(e).with_context("delete", None))?;
        w.delete(predicate)
            .map_err(|e| InfinoError::from(e).with_context("delete", None))?;
        single_outcome(
            w.commit()
                .map_err(|e| InfinoError::from(e).with_context("delete", None))?,
        )
        .map_err(|e| e.with_context("delete", None))
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
    ///
    /// Consumer-memory-mode handles
    /// (`summary_centroids_from_superfiles`) are read-only by
    /// construction: they hydrate routing-form manifest parts (no
    /// summary fp32), and a commit from that state would re-encode
    /// stripped summaries into the durable full wire form. Refused
    /// here — at acquisition, not deep inside a commit.
    fn writer(&self) -> Result<SupertableWriter, BuildError> {
        if self.inner().options.summary_centroids_from_superfiles {
            return Err(BuildError::Store(
                "this handle opened in consumer memory mode \
                 (summary_centroids_from_superfiles): summaries hydrate without fp32, so it \
                 cannot write — open a writer handle with the mode off"
                    .into(),
            ));
        }
        match self.inner().writer_outstanding.compare_exchange(
            false,
            true,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => Ok(SupertableWriter {
                inner: Arc::clone(self.inner()),
                buffer: Vec::new(),
                buffer_scalar_bytes: 0,
                buffer_vector_bytes: 0,
                buffer_fts_bytes: 0,
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
) -> Option<ClusterCentroids> {
    let mut vectors = Vec::new();
    for batch in batches {
        let Some(first) = batch.vectors.first() else {
            continue;
        };
        let vecs = first.values();
        let n_rows = batch.scalar.num_rows();
        // Checked: a malformed buffered batch (vector column shorter than
        // rows × dim) must fail the bootstrap, not panic the commit.
        let expected = n_rows.checked_mul(vec_dim)?;
        if vecs.len() < expected {
            return None;
        }
        vectors.extend_from_slice(&vecs[..expected]);
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
    Some(ClusterCentroids::from_fp32(
        k as u32,
        vec_dim as u32,
        &centroids,
        counts,
    ))
}

impl SupertableWriter {
    /// Number of buffered batches not yet committed. Useful for
    /// tests + diagnostics; not part of the production hot path.
    pub fn buffered_batches(&self) -> usize {
        self.buffer.len()
    }

    /// Bytes of buffered (un-committed) data actually held in memory:
    /// the scalar columns plus the f32 vector payload. This is the
    /// figure the auto-flush threshold is compared against (the FTS
    /// weighting only affects the build-scratch reserve, not held size).
    pub fn buffered_bytes(&self) -> usize {
        self.buffer_scalar_bytes + self.buffer_vector_bytes
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
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(rows = batch.num_rows(), buffered = self.buffer.len()))
    )]
    pub fn append(&mut self, batch: &RecordBatch) -> Result<(), BuildError> {
        let options = &self.inner.options;

        // Validate + split. Batch schema is user_schema (no id col).
        let (scalar_no_id, _vector_slices) = split_vectors(batch, options)?;

        // Re-derive owned Arc<Float32Array> handles for each vector column. We can't keep the &[f32] slices from
        // split_vectors in the buffer (their lifetime is tied to `batch`, which the caller reclaims after this returns).
        // The Arc<Float32Array> shares the same underlying buffer — no bytes copied.
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

        // Estimate byte cost per input class. get_array_memory_size accounts for
        // Arrow buffer allocations (rough but good enough); the vector payload is
        // its exact f32 size. The FTS text columns are a subset of the scalar
        // columns, summed separately only to weight the build-scratch reserve.
        let scalar_bytes = scalar.get_array_memory_size();
        let vector_bytes = vectors
            .iter()
            .map(|v| v.len() * mem::size_of::<f32>())
            .sum::<usize>();
        let fts_bytes = options
            .fts_columns
            .iter()
            .filter_map(|fc| scalar.schema().index_of(&fc.column).ok())
            .map(|idx| scalar.column(idx).get_array_memory_size())
            .sum::<usize>();

        self.buffer.push(BufferedBatch { scalar, vectors });
        self.buffer_scalar_bytes += scalar_bytes;
        self.buffer_vector_bytes += vector_bytes;
        self.buffer_fts_bytes += fts_bytes;

        // Auto-flush on held bytes (scalar + vector); the FTS weighting is a
        // reserve-time concern, not held memory.
        let threshold = (options.commit_threshold_size_mb as usize)
            .saturating_mul(1024)
            .saturating_mul(1024);
        if threshold > 0 && self.buffered_bytes() >= threshold {
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
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(
            buffered = self.buffer.len(),
            updates = self.pending_updates.len(),
            deletes = self.pending_deletes.len(),
        ))
    )]
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
                    error!(
                        committed = outcomes.len(),
                        total = total_mutations,
                        error = %cause,
                        "partial commit: update failed mid-flush"
                    );
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
                    error!(
                        committed = outcomes.len(),
                        total = total_mutations,
                        error = %cause,
                        "partial commit: delete failed mid-flush"
                    );
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
            if let Some(hi) = hidden_inner
                && let Err(e) = record_hidden_deleted_ids(&hi, &deleted_ids).await
            {
                tracing::warn!(
                    "supertable: hidden vector-index deleted-set record failed: {e} \
                     (user-table delete is durable; vector search may transiently \
                     return deleted rows until the next successful record)"
                );
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

    /// [`SupertableWriter::commit`] calls this first before
    /// driving pending mutations.
    ///
    /// Rows are balanced evenly across shards regardless of the
    /// caller's `append()` cadence — many small appends followed by
    /// one `commit` produce the same shard layout as one large append.
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(buffered = self.buffer.len()))
    )]
    fn commit_appends_internal(&mut self) -> Result<(), BuildError> {
        if self.buffer.is_empty() {
            return Ok::<(), BuildError>(());
        }

        // Try reserving the transient heap from the ConnectionMemoryBudget before draining the buffer.
        // For now, if memory reservation is refused, the buffer is left untouched, but this behaviour can be changed.
        //
        // Held until this function returns, i.e. past `publish_superfiles` below.
        let _build_guard = reserve_build_scratch(
            &self.inner.options.connection_memory_budget,
            self.buffer_scalar_bytes,
            self.buffer_vector_bytes,
            self.buffer_fts_bytes,
        )?;

        // Take the buffer so a concurrent append can't observe a half-drained
        // state, but keep the batches for restore on any later failure (S9).
        let saved_scalar = self.buffer_scalar_bytes;
        let saved_vector = self.buffer_vector_bytes;
        let saved_fts = self.buffer_fts_bytes;
        let buffer = mem::take(&mut self.buffer);
        self.buffer_scalar_bytes = 0;
        self.buffer_vector_bytes = 0;
        self.buffer_fts_bytes = 0;

        match self.commit_appends_with_taken_buffer(&buffer) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.buffer = buffer;
                self.buffer_scalar_bytes = saved_scalar;
                self.buffer_vector_bytes = saved_vector;
                self.buffer_fts_bytes = saved_fts;
                Err(e)
            }
        }
    }

    /// Body of [`Self::commit_appends_internal`] after the buffer has been
    /// taken. On `Err`, the caller restores `buffer` onto the writer.
    fn commit_appends_with_taken_buffer(&self, buffer: &[BufferedBatch]) -> Result<(), BuildError> {
        // Phase A — train the global cell grid from the FIRST committed batch
        // into pending OCC metadata (not a bare ArcSwap.store). The pack path
        // below reads the same local `pending_gvi` / existing manifest grid;
        // the stamp lands with the membership commit (S10).
        let pending_gvi: Option<GlobalVectorIndex> = if self
            .inner
            .manifest
            .load()
            .get_global_vector_index()
            .is_none()
            && !buffer.is_empty()
            && let Some(vc) = self.inner.options.vector_columns.first()
            && let Some(grid) = bootstrap_centroids_from_batch(
                buffer,
                vc.dim,
                super::handle::hidden_vector_cell_count(&self.inner.options),
            ) {
            let hidden_cells = super::handle::hidden_vector_cell_count(&self.inner.options);
            let user_cells = super::handle::user_vector_cell_count(&self.inner.options);
            let user_grid = (user_cells != hidden_cells)
                .then(|| bootstrap_centroids_from_batch(buffer, vc.dim, user_cells))
                .flatten();
            Some(GlobalVectorIndex {
                column: vc.column.clone(),
                grid,
                user_grid,
            })
        } else {
            None
        };

        let total_rows: usize = buffer.iter().map(|b| b.scalar.num_rows()).sum();
        if total_rows == 0 {
            return Ok(());
        }

        let list_metadata = CommitListMetadata {
            partition_strategy: None,
            global_vector_index: pending_gvi.clone(),
            drained_ranges: None,
        };

        // Vector commit: same row-shard fanout as the legacy path. Each writer
        // assigns its rows to cells, calls drain's pack
        // (`build_merged_subsection_from_fp32` → materialized pack: sampled
        // fine k-means + Sq8), overlapped with Parquet+FTS, then splices IVF
        // blobs into the superfile and publishes. Drain does not write/S3 on
        // this path. No slow CAS.
        if !self.inner.options.vector_columns.is_empty() {
            let commit_t0 = time::Instant::now();
            let pack_grid = pending_gvi
                .as_ref()
                .cloned()
                .or_else(|| self.inner.manifest.load().get_global_vector_index())
                .ok_or_else(|| {
                    BuildError::Store(
                        "vector columns present but global cell grid missing after Phase A".into(),
                    )
                })?
                .into_user_grid();
            let metric = self
                .inner
                .options
                .vector_columns
                .first()
                .map(|vc| vc.metric)
                .unwrap_or(Metric::L2Sq);
            let (outputs, cell_hints) =
                commit_shards_via_drain(buffer, &self.inner, &pack_grid, metric)?;
            let build_elapsed = commit_t0.elapsed();
            let output_bytes: usize = outputs.iter().map(|output| output.bytes.len()).sum();
            let user_batch = prepare_user_superfile_batch(&self.inner, outputs, cell_hints)?;
            let prepare_elapsed = commit_t0.elapsed().saturating_sub(build_elapsed);
            let data_put_bytes: usize = user_batch
                .pending_storage_writes
                .iter()
                .map(|(_, bytes)| bytes.len())
                .sum();
            let publish_t0 = time::Instant::now();
            bridge_on_runtime(
                persist_superfile_publish_batch_async(&self.inner, user_batch, list_metadata),
                &self.inner.query_runtime(),
            )?;
            if crate::storage::io_counters::timeline_enabled() {
                eprintln!(
                    "[supertable commit] build {:.1}ms ({:.1} MiB output) + prepare {:.1}ms + \
                     publish {:.1}ms ({:.1} MiB data PUT)",
                    build_elapsed.as_secs_f64() * 1e3,
                    output_bytes as f64 / (1u64 << 20) as f64,
                    prepare_elapsed.as_secs_f64() * 1e3,
                    publish_t0.elapsed().as_secs_f64() * 1e3,
                    data_put_bytes as f64 / (1u64 << 20) as f64,
                );
            }
            if self.inner.options.storage.is_some() {
                schedule_background_storage_reclaim(Arc::clone(&self.inner));
            }
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
        // Clone into shard builders so `buffer` stays intact for S9 restore.
        // Arrow batches are Arc-backed — this is a shallow clone of handles.
        let owned = buffer.to_vec();
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
                    let cell_shards = writer_pool
                        .install(|| split_buffer_by_vector_cell(owned, clusters, metric, 0))?;
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
                    let shards = split_buffer_into_row_shards(owned, n_shards, &vector_dims);
                    let hints = vec![None; shards.len()];
                    (shards, hints)
                }
            } else {
                let shards = split_buffer_into_row_shards(owned, n_shards, &vector_dims);
                let hints = vec![None; shards.len()];
                (shards, hints)
            };

        let user_inner = Arc::clone(&self.inner);
        let user_options = Arc::clone(&self.inner.options);
        // A/B knob (`vector.user_centroids: global`): build user superfiles
        // aligned to the GLOBAL cell grid (cluster c == cell c) instead of local
        // k-means. Prefer the pending bootstrap stamp when this is the first
        // vector commit; otherwise read the durable/manifest grid.
        let user_global_centroids: Option<std::sync::Arc<[f32]>> =
            if config::global().vector.user_centroids == CentroidAlignment::Global {
                pending_gvi
                    .as_ref()
                    .cloned()
                    .or_else(|| self.inner.manifest.load().get_global_vector_index())
                    .filter(|g| g.grid.n_cent > 0 && g.grid.dim > 0)
                    .map(|g| g.grid.to_fp32().into())
            } else {
                None
            };

        // Phase B: user-only build + publish. No hidden incoming build/publish;
        // the hidden cell index is drained later straight from these user
        // superfiles, and pre-drain queries fall back to them.
        let outputs = fanout_shards(&writer_pool, &shards, |slice| {
            build_one_shard_with_layout(
                slice.as_slice(),
                &user_options,
                user_options.vector_layout,
                user_global_centroids.clone(),
            )
        })?;
        let superfiles = outputs.len();
        let user_batch = prepare_user_superfile_batch(&self.inner, outputs, cell_hints)?;
        bridge_on_runtime(
            persist_superfile_publish_batch_async(&user_inner, user_batch, list_metadata),
            &self.inner.query_runtime(),
        )?;
        if self.inner.options.storage.is_some() {
            schedule_background_storage_reclaim(Arc::clone(&self.inner));
        }
        debug!(superfiles, "published appended superfiles");

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

/// Reserve the build's estimated transient heap:
///
/// estimate = (2.5*scalar_raw_bytes + 6.5*vector_raw_bytes + 1.5*fts_text_raw_bytes)
///
/// returns `OverBudget` when a bounded budget can't fit it.
fn reserve_build_scratch(
    budget: &Arc<ConnectionMemoryBudget>,
    scalar_bytes: usize,
    vector_bytes: usize,
    fts_bytes: usize,
) -> Result<Reservation, BuildError> {
    //  The constants are kept integer rather than float, so the estimate is calculated as such:
    //
    //     (BUILD_SCALAR_NUM * scalar_bytes) + (BUILD_VECTOR_NUM * vector_bytes) + (BUILD_FTS_NUM * fts_bytes)
    //   ------------------------------------------------------------------------------------------------------
    //                                            BUILD_SCRATCH_DENOM
    //
    let estimate = scalar_bytes
        .saturating_mul(BUILD_SCALAR_NUM)
        .saturating_add(vector_bytes.saturating_mul(BUILD_VECTOR_NUM))
        .saturating_add(fts_bytes.saturating_mul(BUILD_FTS_NUM))
        / BUILD_SCRATCH_DENOM;

    budget
        .try_reserve(estimate)
        // Label the message "during ingest" so it can be told apart from a query
        // or SQL over-budget error once it reaches the public InfinoError.
        .map_err(|e| BuildError::OverBudget(format!("during ingest, {e}")))
}

/// Build one superfile from one slice of buffered batches with an explicit
/// vector layout override. Runs on a rayon worker thread inside the writer
/// pool's `install`. The commit path always passes an explicit layout +
/// optional global centroids.
fn build_one_shard_with_layout(
    slice: &[BufferedBatch],
    options: &SupertableOptions,
    vector_layout: crate::superfile::vector::layout::VectorLayout,
    provided_centroids: Option<std::sync::Arc<[f32]>>,
) -> Result<ShardOutput, BuildError> {
    let mut builder = SuperfileBuilder::new(
        options
            .builder_options()
            .with_vector_layout(vector_layout)
            .with_vector_centroids(provided_centroids),
    )?;

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
    // Multi-cell open ranges need the column dim to bound the cluster index
    // (the cell directory carries no n_cent); a single logical column is the
    // multi-cell contract.
    let vec_dim = kvs
        .get(kv::VEC_COLUMNS)
        .and_then(|json| serde_json::from_str::<Vec<VectorColumnConfig>>(json).ok())
        .and_then(|cols| match cols.as_slice() {
            [only] => Some(only.dim),
            _ => None,
        });
    let vec_open_ranges = vec
        .and_then(|(off, len)| vector_open_ranges(bytes, off, len, vec_dim))
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

fn vector_open_ranges(
    bytes: &Bytes,
    off: u64,
    len: u64,
    dim: Option<usize>,
) -> Option<Vec<(u64, u64)>> {
    let start = off as usize;
    let end = start.checked_add(len as usize)?;
    let blob = bytes.get(start..end)?;
    if blob.len() < OUTER_HEADER_SIZE + CRC_BYTES {
        return None;
    }
    let version =
        read_u32_le(blob.get(outer_hdr::VERSION_OFF..outer_hdr::VERSION_OFF + U32_BYTES)?);
    if version == crate::superfile::format::vec::VERSION_MULTI_CELL {
        return vector_open_ranges_multi_cell(blob, off, dim?);
    }
    // Reject any version we don't recognize instead of falling through to the
    // v1 layout (a future/corrupt version would otherwise be mis-parsed).
    if version != crate::superfile::format::vec::VERSION {
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

/// Open-time ranges for a v2 multi-cell vector blob: outer header, cell
/// directory, and each cell's sub-header + cluster index — the same v1
/// discipline as the single-cell path above. The fp32 centroids, Sq8
/// scale/offset meta, per-row norms, and the inline stable-id region all
/// stay on disk: they are read per probed cell through the block cache
/// (deferred rescore, the lazy Sq8-meta arm, and the probe wave's
/// stable-id piggyback). Staging them here made the open footprint —
/// manifest-inline open blobs *and* the cold-open hint fetch — grow with
/// per-row data: measured 318 MiB of hidden-data open fetch at 10M and
/// 3.62 GiB / 12.3 s at 100M, with user manifest parts at 3.28 GiB from
/// the embedded copies.
fn vector_open_ranges_multi_cell(blob: &[u8], off: u64, dim: usize) -> Option<Vec<(u64, u64)>> {
    use crate::superfile::format::vec::U64_BYTES;
    if dim == 0 {
        return None;
    }
    let n_cells =
        read_u32_le(blob.get(outer_hdr::N_CELLS_OFF..outer_hdr::N_CELLS_OFF + U32_BYTES)?) as usize;
    let dir_offset =
        read_u64_le(blob.get(outer_hdr::DIR_OFFSET_OFF..outer_hdr::DIR_OFFSET_OFF + U64_BYTES)?)
            as usize;
    let dir_size = n_cells.checked_mul(CELL_DIR_ENTRY_SIZE)?;
    let dir_end = dir_offset.checked_add(dir_size)?.checked_add(CRC_BYTES)?;
    if dir_end > blob.len() {
        return None;
    }
    let dir = blob.get(dir_offset..dir_offset + dir_size)?;
    let mut ranges = vec![
        (off, OUTER_HEADER_SIZE as u64),
        (off + dir_offset as u64, (dir_size + CRC_BYTES) as u64),
    ];
    for i in 0..n_cells {
        let entry = i * CELL_DIR_ENTRY_SIZE;
        let subsection_off = read_u64_le(dir.get(
            entry + cell_dir_entry::SUBSECTION_OFF_OFF
                ..entry + cell_dir_entry::SUBSECTION_OFF_OFF + U64_BYTES,
        )?) as usize;
        let subsection_len = read_u64_le(dir.get(
            entry + cell_dir_entry::SUBSECTION_LEN_OFF
                ..entry + cell_dir_entry::SUBSECTION_LEN_OFF + U64_BYTES,
        )?) as usize;
        if subsection_off.checked_add(SUB_HEADER_SIZE)? > blob.len()
            || subsection_off.checked_add(subsection_len)? > blob.len()
        {
            return None;
        }
        let sub = blob.get(subsection_off..subsection_off + subsection_len)?;
        let centroids_off = read_u64_le(
            sub.get(sub_hdr::CENTROIDS_OFF_OFF..sub_hdr::CENTROIDS_OFF_OFF + U64_BYTES)?,
        ) as usize;
        let cluster_idx_off = read_u64_le(
            sub.get(sub_hdr::CLUSTER_IDX_OFF_OFF..sub_hdr::CLUSTER_IDX_OFF_OFF + U64_BYTES)?,
        ) as usize;
        let centroids_span = cluster_idx_off.checked_sub(centroids_off)?;
        if centroids_off < SUB_HEADER_SIZE || !centroids_span.is_multiple_of(dim * 4) {
            return None;
        }
        let n_cent = centroids_span / (dim * 4);
        let cluster_idx_end =
            cluster_idx_off.checked_add(n_cent.checked_mul(CLUSTER_IDX_ENTRY_BYTES)?)?;
        if cluster_idx_end > subsection_len {
            return None;
        }
        ranges.push((off + subsection_off as u64, SUB_HEADER_SIZE as u64));
        ranges.push((
            off + (subsection_off + cluster_idx_off) as u64,
            (cluster_idx_end - cluster_idx_off) as u64,
        ));
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

/// One vector column's per-cell manifest summary from a freshly written
/// superfile: the per-cluster fp32 centroids (so a query ranks this
/// superfile's clusters globally without opening it) plus the 1-bit
/// admit slab computed alongside them — the summary wire blob persists
/// both, and consumers decode the slab at hydration instead of
/// re-deriving one rotation per centroid. Shared by the commit staging
/// path and the WAL update pipeline.
pub(crate) fn build_column_vector_summary(
    vec_reader: &VectorReader,
    vc: &VectorConfig,
) -> Option<VectorSummary> {
    let centroid = vec_reader.summary(&vc.column)?;
    let cells: Vec<CellVectorSummary> = vec_reader
        .cluster_centroids_by_cell(&vc.column)
        .unwrap_or_default()
        .into_iter()
        .map(|(cell_id, n_cent, dim, fp32, counts)| CellVectorSummary {
            cell_id,
            clusters: ClusterCentroids::from_fp32(n_cent, dim, &fp32, counts),
        })
        .collect();
    let rotation = RandomRotation::new(vc.dim, vc.rot_seed);
    let quant = BitQuantizer::new(vc.dim);
    for cell in &cells {
        if cell.clusters.dim as usize == vc.dim {
            cell.clusters
                .prewarm_admit_codes(&rotation, &quant, vc.rot_seed);
        }
    }
    Some(VectorSummary { centroid, cells })
}

/// Build the per-shard publish artifacts: open a `SuperfileReader`
/// on the shard bytes, derive FTS + vector summaries, and decide
/// the bytes-disposition triplet. Pure per-shard work — no shared
/// mutable state, safe to run in parallel across shards.
pub(super) fn prepare_superfile(
    inner: &SupertableInner,
    shard: ShardOutput,
) -> Result<Option<PreparedSuperfile>, BuildError> {
    prepare_superfile_with_uri(inner, shard, None)
}

pub(super) fn prepare_superfile_with_uri(
    inner: &SupertableInner,
    shard: ShardOutput,
    reuse_uri: Option<SuperfileUri>,
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
    // Warm-fill the disk cache when attached AND the producer opts in
    // (`prepopulate_cache_on_commit`, default true): commits are durable in
    // object storage first, then mirrored locally so maintenance/compaction
    // can merge from mmap-resident bytes without re-fetching whole objects.
    // Ingest-only producers that drop the writer immediately (e.g. the bench)
    // set this false — mirroring would be a pure second fsync'd write + CRC
    // re-scan of every superfile, ~doubling per-commit write I/O for no reader.
    let bytes_for_cache =
        (cache_attached && inner.options.prepopulate_cache_on_commit).then(|| shard.bytes.clone());

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
            if let Some(summary) = build_column_vector_summary(vec_reader, vc) {
                vector_summary.insert(vc.column.clone(), summary);
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
        // Hidden cell superfile; stamped by the hidden manifest's own
        // `update`. Irrelevant to the user-side drain watermark.
        birth_version: 0,
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
/// docs × 4 superfiles is the dominant cost. ManifestSnapshot swap +
/// storage write-through stay serial after the join.
fn finish_superfile_entry(
    entry: Arc<SuperfileEntry>,
    hint: Option<u32>,
) -> Result<Arc<SuperfileEntry>, BuildError> {
    let old = entry.as_ref();
    let staged = SuperfileEntry {
        birth_version: old.birth_version,
        superfile_id: old.superfile_id,
        uri: old.uri,
        n_docs: old.n_docs,
        id_min: old.id_min,
        id_max: old.id_max,
        scalar_stats: old.scalar_stats.clone(),
        fts_summary: old.fts_summary.clone(),
        vector_summary: old.vector_summary.clone(),
        // Partition key is now stamped by manifest update at commit time.
        partition_key: Vec::new(),
        partition_hint: hint.or(old.partition_hint),
        subsection_offsets: old.subsection_offsets.clone(),
        vector_layout: old.vector_layout,
    };
    Ok(Arc::new(staged))
}

/// Collected superfile entries + pending storage/cache writes for one publish.
struct SuperfilePublishBatch {
    new_entries: Vec<Arc<SuperfileEntry>>,
    to_remove: Vec<Arc<SuperfileEntry>>,
    pending_storage_writes: Vec<(SuperfileUri, Bytes)>,
    pending_cache_inserts: Vec<(SuperfileUri, Bytes)>,
    /// In-memory reader-cache inserts deferred until after durable (or
    /// local) membership publish succeeds — inserting earlier leaves
    /// orphaned cache entries when the CAS fails (S12).
    pending_store_inserts: Vec<(SuperfileUri, Bytes)>,
}

fn collect_prepared_superfiles(
    _inner: &SupertableInner,
    prepared: Vec<PreparedSuperfile>,
) -> Result<SuperfilePublishBatch, BuildError> {
    let mut new_entries: Vec<Arc<SuperfileEntry>> = Vec::with_capacity(prepared.len());
    let mut pending_storage_writes: Vec<(SuperfileUri, Bytes)> = Vec::new();
    let mut pending_cache_inserts: Vec<(SuperfileUri, Bytes)> = Vec::new();
    let mut pending_store_inserts: Vec<(SuperfileUri, Bytes)> = Vec::new();
    for p in prepared {
        if let Some(t) = p.bytes_for_store {
            pending_store_inserts.push(t);
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
        pending_store_inserts,
    })
}

fn apply_pending_store_inserts(inner: &SupertableInner, inserts: Vec<(SuperfileUri, Bytes)>) {
    for (uri, bytes) in inserts {
        // Non-fatal: bytes are durable (or local-appended) and a later
        // open can refetch. Mirrors the WAL append path.
        let _ = inner.options.store.insert(uri, bytes);
    }
}

fn prepare_user_superfile_batch_in_scope(
    inner: &SupertableInner,
    outputs: Vec<ShardOutput>,
    hints: Vec<Option<u32>>,
) -> Result<SuperfilePublishBatch, BuildError> {
    // `zip` silently truncates to the shorter side; a length mismatch here
    // would drop shard outputs or hints and publish an incomplete commit.
    if outputs.len() != hints.len() {
        return Err(BuildError::Store(format!(
            "superfile publish inputs out of sync: {} shard outputs for {} partition hints",
            outputs.len(),
            hints.len()
        )));
    }
    let prepared: Vec<PreparedSuperfile> = outputs
        .into_par_iter()
        .zip(hints.into_par_iter())
        .filter_map(|(shard, hint)| match prepare_superfile(inner, shard) {
            Ok(Some(p)) => {
                Some(
                    finish_superfile_entry(p.entry, hint).map(|entry| PreparedSuperfile {
                        entry,
                        bytes_for_store: p.bytes_for_store,
                        bytes_for_storage: p.bytes_for_storage,
                        bytes_for_cache: p.bytes_for_cache,
                    }),
                )
            }
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
    list_metadata: CommitListMetadata,
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
            list_metadata,
        )
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;
        inner.manifest.store(Arc::new(new_manifest));
        apply_pending_store_inserts(inner, batch.pending_store_inserts);
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
    // Local (no-storage) path: stamp list metadata onto the OCC base, then
    // append — `with_appended` preserves the stamped fields.
    let new = if list_metadata.is_empty() {
        old.with_appended(batch.new_entries)
    } else {
        list_metadata.apply(&old).with_appended(batch.new_entries)
    };
    // Insert the bytes BEFORE publishing the manifest, and fail on error:
    // with no storage attached the in-memory store is the ONLY copy, so
    // publishing first would expose entries whose bytes a reader can't
    // fetch, and a failed insert would leave a "successful" commit with
    // lost data. (The storage-backed path above keeps insert-after-store
    // non-fatal — its bytes are already durable and refetchable.)
    for (uri, bytes) in batch.pending_store_inserts {
        inner
            .options
            .store
            .insert(uri, bytes)
            .map_err(|e| BuildError::Store(format!("store insert for {uri:?}: {e}")))?;
    }
    inner.manifest.store(Arc::new(new));
    Ok(())
}

/// Single-thread rayon pool for incoming-routing CPU work (cell assignment + per-cell
/// superfile encode). Installing the build under this pool pins all its nested
/// `par_iter`/`join` to one thread instead of fanning out across every core, so
/// routing can't starve foreground ingest CPU.
static MAINT_POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();

fn maint_pool() -> Result<&'static ThreadPool, BuildError> {
    if let Some(pool) = MAINT_POOL.get() {
        return Ok(pool);
    }
    // Build outside `get_or_init` so a spawn failure propagates instead of
    // panicking the maintenance path (`OnceLock::get_or_try_init` is not
    // stable). A racing initializer wins harmlessly; ours is dropped.
    let pool = ThreadPoolBuilder::new()
        .num_threads(1)
        .thread_name(|_| "hidden-maint-cpu".into())
        .build()
        .map_err(|e| BuildError::Store(format!("hidden maintenance rayon pool: {e}")))?;
    Ok(MAINT_POOL.get_or_init(|| pool))
}

/// No-staging drain: read committed user superfiles, assign their encoded rows
/// to the global cells, and publish the hidden index as packed multi-cell
/// superfiles (`cell_id % N`, `N = writer_pool`). Reads from `user_inner`,
/// writes to `hidden_inner`; user superfiles remain the durable source.
///
/// Processes user superfiles in BOUNDED BATCHES (`drain_batch_superfiles`) so
/// working-set RAM stays O(batch). Kmeans mode accumulates encoded rows in one
/// disk spill per global cell and trains that cell's fine IVF once over the
/// complete cross-batch population. Splice mode accumulates source clusters
/// verbatim. Both modes finally stream complete cell IVFs into at most one
/// MultiCellIvf per writer worker. **Incremental**: skips user commits whose
/// `birth_version` is already in the hidden manifest's `drained_ranges`.
/// Pre-drain queries see an empty hidden index (0 results) until this runs.
///
/// Batch size comes from `vector.drain_batch_superfiles`, which
/// [`SupertableOptions::apply_config`] copies into the option below; per-table
/// callers can still override it via `with_drain_batch_superfiles`.
fn drain_batch_superfiles(opts: &SupertableOptions) -> i64 {
    opts.drain_batch_superfiles
}

fn spill_row_to_cell(
    spills: &mut HashMap<u32, MaterializedRowSpillWriter>,
    added: &mut HashMap<u32, u32>,
    scratch: &Path,
    cell: u32,
    row: &MaterializedIvfRow,
) -> Result<(), BuildError> {
    let writer = match spills.entry(cell) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => entry.insert(MaterializedRowSpillWriter::create(
            scratch,
            cell,
            row.encoded.codes.len(),
            row.rabitq_code.len(),
        )?),
    };
    writer.append(row)?;
    let count = added.entry(cell).or_insert(0);
    *count = count.saturating_add(1);
    Ok(())
}

fn spill_unfinished_shard_row(
    spills: &mut HashMap<u32, MaterializedRowSpillWriter>,
    added: &mut HashMap<u32, u32>,
    completed_shards: &HashSet<u32>,
    shard_count: usize,
    scratch: &Path,
    cell: u32,
    row: &MaterializedIvfRow,
) -> Result<(), BuildError> {
    if completed_shards.contains(&(packed_cell_shard(cell, shard_count) as u32)) {
        return Ok(());
    }
    spill_row_to_cell(spills, added, scratch, cell, row)
}

fn drain_checkpoint_source(entry: &SuperfileEntry) -> DrainCheckpointSource {
    DrainCheckpointSource {
        superfile_id: entry.superfile_id.to_string(),
        uri: entry.uri.0.to_string(),
        birth_version: entry.birth_version,
    }
}

fn drain_epoch_id(
    options_hash: &str,
    sources: &[DrainCheckpointSource],
    batch_layout: &[Vec<u64>],
    shard_count: usize,
    consolidate: DrainConsolidate,
) -> String {
    let mut hasher = Blake3Hasher::new();
    hasher.update(&DRAIN_CHECKPOINT_SCHEMA.to_le_bytes());
    hasher.update(&(shard_count as u64).to_le_bytes());
    hasher.update(options_hash.as_bytes());
    // Consolidate mode changes the packed-cell bytes; pin it in the epoch.
    hasher.update(match consolidate {
        DrainConsolidate::Kmeans => b"kmeans",
        DrainConsolidate::Splice => b"splice",
    });
    for source in sources {
        hasher.update(source.superfile_id.as_bytes());
        hasher.update(source.uri.as_bytes());
        hasher.update(&source.birth_version.to_le_bytes());
    }
    for batch in batch_layout {
        hasher.update(&(batch.len() as u64).to_le_bytes());
        for version in batch {
            hasher.update(&version.to_le_bytes());
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn drain_scratch_dir(epoch_id: &str) -> PathBuf {
    env::temp_dir().join("infino-drain").join(epoch_id)
}

fn drain_local_checkpoint_path(scratch: &Path) -> PathBuf {
    scratch.join(DRAIN_LOCAL_CHECKPOINT_FILE)
}

fn load_drain_local_checkpoint(
    scratch: &Path,
    epoch_id: &str,
) -> Result<Option<DrainLocalCheckpoint>, BuildError> {
    let path = drain_local_checkpoint_path(scratch);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(BuildError::Store(format!(
                "drain local checkpoint read {}: {error}",
                path.display()
            )));
        }
    };
    let checkpoint: DrainLocalCheckpoint = serde_json::from_slice(&bytes)
        .map_err(|error| BuildError::Store(format!("drain local checkpoint decode: {error}")))?;
    if checkpoint.schema != DRAIN_CHECKPOINT_SCHEMA || checkpoint.epoch_id != epoch_id {
        return Err(BuildError::Store(format!(
            "drain local checkpoint at {} is incompatible (schema {}, epoch {})",
            path.display(),
            checkpoint.schema,
            checkpoint.epoch_id
        )));
    }
    Ok(Some(checkpoint))
}

fn save_drain_local_checkpoint(
    scratch: &Path,
    checkpoint: &DrainLocalCheckpoint,
) -> Result<(), BuildError> {
    fs::create_dir_all(scratch)
        .map_err(|error| BuildError::Store(format!("drain scratch create: {error}")))?;
    let bytes = serde_json::to_vec(checkpoint)
        .map_err(|error| BuildError::Store(format!("drain local checkpoint encode: {error}")))?;
    let final_path = drain_local_checkpoint_path(scratch);
    let temp_path = scratch.join(format!("{DRAIN_LOCAL_CHECKPOINT_FILE}.tmp"));
    {
        let mut file = File::create(&temp_path)
            .map_err(|error| BuildError::Store(format!("drain checkpoint create: {error}")))?;
        file.write_all(&bytes)
            .map_err(|error| BuildError::Store(format!("drain checkpoint write: {error}")))?;
        file.sync_all()
            .map_err(|error| BuildError::Store(format!("drain checkpoint fsync: {error}")))?;
    }
    fs::rename(&temp_path, &final_path)
        .map_err(|error| BuildError::Store(format!("drain checkpoint rename: {error}")))?;
    File::open(scratch)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| BuildError::Store(format!("drain checkpoint dir fsync: {error}")))?;
    Ok(())
}

async fn load_drain_remote_checkpoint(
    inner: &SupertableInner,
) -> Result<Option<DrainRemoteState>, BuildError> {
    let manifest = inner.manifest.load_full();
    let Some((uri, hash)) = manifest.slow_vector_state_blob() else {
        return Ok(None);
    };
    let storage = inner
        .options
        .storage
        .as_ref()
        .ok_or_else(|| BuildError::Store("drain checkpoint requires storage".into()))?;
    let state = slow_vector_state::load_full_state(storage.as_ref(), uri, &hash)
        .await
        .map_err(|error| BuildError::Store(format!("drain slow-CAS load: {error}")))?;
    let Some(pending) = state.pending_drain else {
        return Ok(None);
    };
    let checkpoint: DrainRemoteCheckpoint = serde_json::from_slice(&pending.metadata)
        .map_err(|error| BuildError::Store(format!("drain remote checkpoint decode: {error}")))?;
    if checkpoint.schema != DRAIN_CHECKPOINT_SCHEMA {
        return Err(BuildError::Store(format!(
            "drain remote checkpoint schema {} != supported {}",
            checkpoint.schema, DRAIN_CHECKPOINT_SCHEMA
        )));
    }
    if pending.entries.len() != checkpoint.completed_shards.len() {
        return Err(BuildError::Store(format!(
            "drain slow-CAS has {} pending entries for {} completed shards",
            pending.entries.len(),
            checkpoint.completed_shards.len()
        )));
    }
    let entry_ids: HashSet<String> = pending
        .entries
        .iter()
        .map(|entry| entry.superfile_id.to_string())
        .collect();
    if checkpoint
        .completed_shards
        .iter()
        .any(|shard| !entry_ids.contains(&shard.superfile_id))
    {
        return Err(BuildError::Store(
            "drain slow-CAS checkpoint references a missing pending entry".into(),
        ));
    }
    Ok(Some(DrainRemoteState {
        checkpoint,
        entries: pending.entries,
    }))
}

async fn save_drain_remote_checkpoint(
    inner: &SupertableInner,
    state: &mut DrainRemoteState,
) -> Result<(), BuildError> {
    let metadata = serde_json::to_vec(&state.checkpoint)
        .map_err(|error| BuildError::Store(format!("drain checkpoint encode: {error}")))?;
    stamp_slow_vector_state(
        inner,
        Some(slow_vector_state::PendingDrainState {
            metadata,
            entries: state.entries.clone(),
        }),
    )
    .await
}

async fn create_drain_remote_checkpoint(
    inner: &SupertableInner,
    checkpoint: DrainRemoteCheckpoint,
) -> Result<DrainRemoteState, BuildError> {
    let mut state = DrainRemoteState {
        checkpoint,
        entries: Vec::new(),
    };
    save_drain_remote_checkpoint(inner, &mut state).await?;
    Ok(state)
}

fn make_drain_batches(
    sources: Vec<Arc<SuperfileEntry>>,
    budget: usize,
) -> Vec<(Vec<u64>, Vec<Arc<SuperfileEntry>>)> {
    let mut by_version = std::collections::BTreeMap::<u64, Vec<Arc<SuperfileEntry>>>::new();
    for source in sources {
        by_version
            .entry(source.birth_version)
            .or_default()
            .push(source);
    }
    let mut batches = Vec::new();
    let mut versions = Vec::new();
    let mut superfiles = Vec::new();
    for (version, mut version_superfiles) in by_version {
        if !superfiles.is_empty()
            && superfiles.len().saturating_add(version_superfiles.len()) > budget
        {
            batches.push((mem::take(&mut versions), mem::take(&mut superfiles)));
        }
        versions.push(version);
        superfiles.append(&mut version_superfiles);
        if superfiles.len() >= budget {
            batches.push((mem::take(&mut versions), mem::take(&mut superfiles)));
        }
    }
    if !superfiles.is_empty() {
        batches.push((versions, superfiles));
    }
    batches
}

fn drain_batch_layout(batches: &[(Vec<u64>, Vec<Arc<SuperfileEntry>>)]) -> Vec<Vec<u64>> {
    batches
        .iter()
        .map(|(versions, _)| versions.clone())
        .collect()
}

/// Drain replica factor at or below which no boundary replicas are added.
const DEFAULT_DRAIN_REPLICA_TARGET_FACTOR: f32 = 1.0;

/// Target storage amplification for boundary-only drain replication. For
/// example, `1.2` means the drain may add at most `0.2 * rows` extra row copies,
/// selected from rows closest to a Voronoi boundary. Values `<= 1.0` disable
/// replication; the default drain path is unchanged. Sourced from
/// `vector.drain_replica_target_factor`.
fn drain_replica_target_factor() -> f32 {
    let factor = config::global().vector.drain_replica_target_factor;
    if factor.is_finite() && factor > DEFAULT_DRAIN_REPLICA_TARGET_FACTOR {
        factor
    } else {
        DEFAULT_DRAIN_REPLICA_TARGET_FACTOR
    }
}

fn drain_replica_extra_budget(n_rows: usize, target_factor: f32) -> usize {
    if n_rows == 0 || target_factor <= DEFAULT_DRAIN_REPLICA_TARGET_FACTOR {
        return 0;
    }
    let target_rows = (n_rows as f64 * target_factor as f64).ceil() as usize;
    // The closure emits up to REPLICA_CLOSURE_MAX_REPLICAS candidates per
    // row, so factors up to 1 + that many are meaningful (a row can be
    // materialized in every cell of its closure).
    target_rows
        .saturating_sub(n_rows)
        .min(n_rows.saturating_mul(opann::REPLICA_CLOSURE_MAX_REPLICAS))
}

async fn materialized_user_rows_for_drain(
    reader: &SuperfileReader,
    column: &str,
    stable_ids: &[i128],
    tombstones: Option<&roaring::RoaringBitmap>,
) -> Result<Vec<MaterializedIvfRow>, BuildError> {
    let vec_reader = reader
        .vec()
        .ok_or_else(|| BuildError::Store("user superfile missing vector index".into()))?;
    if vec_reader.is_multi_cell() {
        let cells = vec_reader
            .materialized_cells_rows_async(None)
            .await
            .ok_or_else(|| {
                BuildError::Store(format!(
                    "drain materialize: multi-cell column '{column}' missing Sq8Residual index"
                ))
            })?;
        // One physical row per `_id`: boundary stubs share the primary's
        // stable_id, so `or_insert` keeps the first posting seen.
        let mut by_id: HashMap<i128, MaterializedIvfRow> = HashMap::new();
        for (_, rows) in cells {
            for row in rows {
                by_id.entry(row.stable_id).or_insert(row);
            }
        }
        // Tombstones address Parquet primary locals (not IVF file-locals /
        // stubs). Resolve deleted locals → `_id`, then drop by stable_id.
        if let Some(bm) = tombstones
            && !bm.is_empty()
        {
            let locals: Vec<u32> = bm.iter().collect();
            let id_column = reader.id_column();
            let batch = reader
                .take_by_local_doc_ids(&locals, &[id_column])
                .map_err(|e| BuildError::Store(e.to_string()))?;
            let array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| BuildError::Store("_id column missing".into()))?;
            let deleted: HashSet<i128> = array.values().iter().copied().collect();
            by_id.retain(|stable_id, _| !deleted.contains(stable_id));
        }
        let mut rows: Vec<MaterializedIvfRow> = by_id.into_values().collect();
        rows.sort_by_key(|row| row.stable_id);
        for (local, row) in rows.iter_mut().enumerate() {
            row.local_doc_id = local as u32;
        }
        return Ok(rows);
    }
    materialized_ivf_rows_in_doc_order(vec_reader, column, stable_ids, tombstones).await
}

pub(in crate::supertable) async fn drain_user_superfiles_to_hidden_cells(
    user_inner: Arc<SupertableInner>,
    hidden_inner: Arc<SupertableInner>,
) -> Result<(), BuildError> {
    // Single-flight on the hidden side.
    if hidden_inner
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
    let _slot = Slot(&hidden_inner.compaction_outstanding);

    // The global cell grid is owned by the USER manifest (bootstrapped at the
    // first commit). The hidden cell index is the derived copy this drain writes.
    let Some(gvi) = user_inner.manifest.load_full().get_global_vector_index() else {
        return Ok(());
    };
    let clusters = gvi.grid;
    let column = gvi.column;
    if clusters.n_cent == 0 || clusters.dim == 0 {
        return Ok(());
    }
    // Preserve any existing hidden-side query tuning (`routing`) across drains.
    let routing = match hidden_inner.manifest.load_full().get_partition_strategy() {
        PartitionStrategy::VectorCell { routing, .. } => routing,
        _ => CellRoutingParams::default(),
    };

    // Source: every user-table vector superfile, processed in BOUNDED BATCHES so
    // drain working-set RAM stays O(batch) instead of O(corpus) (the >3M memory
    // wall). Each batch opens its readers, builds its cell superfiles, publishes
    // them (append — one file per touched cell), then frees its working set.
    // Batch size is `vector.drain_batch_superfiles`: `0` = skip, `-1` =
    // unbounded single merge.
    let user_manifest = user_inner.manifest.load_full();
    // A cold-open user manifest is parts-backed and may have an empty flat
    // view. Drain must hydrate the authoritative user parts; reading only
    // `get_all_superfiles()` silently turns drain into a no-op after reopen.
    let sources = user_manifest
        .get_all_superfiles_loaded()
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;
    if sources.is_empty() {
        return Ok(());
    }
    let batch_cfg = drain_batch_superfiles(&user_inner.options);
    if batch_cfg == 0 {
        eprintln!("[supertable drain] skipped (drain_batch_superfiles = 0)");
        return Ok(());
    }

    let storage = hidden_inner
        .options
        .storage
        .clone()
        .ok_or_else(|| BuildError::Store("hidden drain requires storage".into()))?;
    let shard_count = packed_cell_shard_count(&hidden_inner.options);
    let consolidate = user_inner.options.drain_consolidate;
    let budget = if batch_cfg < 0 {
        usize::MAX
    } else {
        (batch_cfg as usize).max(1)
    };

    // A remote checkpoint pins the exact source epoch. New user commits can
    // land while it is in progress; they are intentionally left for the next
    // drain instead of invalidating or silently replacing this epoch.
    let drained = hidden_inner.manifest.load_full().get_drained_ranges();
    let user_strategy = user_manifest.get_partition_strategy();
    let current_options_hash =
        options_hash::compute_options_hash(user_inner.options.as_ref(), &user_strategy).to_hex();
    let (batches, mut remote_state) = if let Some(remote_state) =
        load_drain_remote_checkpoint(&hidden_inner).await?
    {
        if remote_state.checkpoint.shard_count != shard_count {
            return Err(BuildError::Store(format!(
                "drain checkpoint shard count {} != configured writer width {shard_count}",
                remote_state.checkpoint.shard_count
            )));
        }
        if remote_state.checkpoint.options_hash != current_options_hash {
            return Err(BuildError::Store(format!(
                "drain checkpoint options hash {} != current {}",
                remote_state.checkpoint.options_hash, current_options_hash
            )));
        }
        let n_drained = remote_state
            .checkpoint
            .sources
            .iter()
            .filter(|source| drained.contains(source.birth_version))
            .count();
        if n_drained == remote_state.checkpoint.sources.len() {
            let scratch = drain_scratch_dir(&remote_state.checkpoint.epoch_id);
            if let Err(error) = fs::remove_dir_all(&scratch)
                && error.kind() != io::ErrorKind::NotFound
            {
                tracing::warn!("drain local checkpoint cleanup failed: {error}");
            }
            refresh_slow_vector_state(&hidden_inner).await?;
            schedule_background_storage_reclaim(Arc::clone(&hidden_inner));
            return Ok(());
        }
        if n_drained != 0 {
            return Err(BuildError::Store(
                "drain checkpoint source versions are only partially committed".into(),
            ));
        }

        let source_by_id: HashMap<String, Arc<SuperfileEntry>> = sources
            .iter()
            .map(|entry| (entry.superfile_id.to_string(), Arc::clone(entry)))
            .collect();
        let mut selected = Vec::with_capacity(remote_state.checkpoint.sources.len());
        for source in &remote_state.checkpoint.sources {
            let entry = source_by_id.get(&source.superfile_id).ok_or_else(|| {
                BuildError::Store(format!(
                    "drain checkpoint source {} is missing from the user manifest",
                    source.superfile_id
                ))
            })?;
            if entry.uri.0.to_string() != source.uri || entry.birth_version != source.birth_version
            {
                return Err(BuildError::Store(format!(
                    "drain checkpoint source {} no longer matches the user manifest",
                    source.superfile_id
                )));
            }
            selected.push(Arc::clone(entry));
        }
        let batches = make_drain_batches(selected, budget);
        let batch_layout = drain_batch_layout(&batches);
        if batch_layout != remote_state.checkpoint.batch_layout {
            return Err(BuildError::Store(
                "drain checkpoint batch layout differs from current configuration".into(),
            ));
        }
        let epoch_id = drain_epoch_id(
            &current_options_hash,
            &remote_state.checkpoint.sources,
            &batch_layout,
            shard_count,
            consolidate,
        );
        if epoch_id != remote_state.checkpoint.epoch_id {
            return Err(BuildError::Store(
                "drain checkpoint epoch hash is invalid".into(),
            ));
        }
        (batches, remote_state)
    } else {
        let mut selected: Vec<Arc<SuperfileEntry>> = sources
            .iter()
            .filter(|entry| !drained.contains(entry.birth_version))
            .cloned()
            .collect();
        if selected.is_empty() {
            eprintln!(
                "[supertable drain] nothing to drain: all {} user superfile(s) already drained",
                sources.len()
            );
            return Ok(());
        }
        selected.sort_unstable_by(|left, right| {
            left.birth_version
                .cmp(&right.birth_version)
                .then_with(|| left.superfile_id.cmp(&right.superfile_id))
        });
        let source_refs: Vec<DrainCheckpointSource> = selected
            .iter()
            .map(|entry| drain_checkpoint_source(entry))
            .collect();
        let batches = make_drain_batches(selected, budget);
        let batch_layout = drain_batch_layout(&batches);
        let epoch_id = drain_epoch_id(
            &current_options_hash,
            &source_refs,
            &batch_layout,
            shard_count,
            consolidate,
        );
        let checkpoint = DrainRemoteCheckpoint {
            schema: DRAIN_CHECKPOINT_SCHEMA,
            epoch_id,
            options_hash: current_options_hash,
            sources: source_refs,
            batch_layout,
            shard_count,
            completed_shards: Vec::new(),
        };
        let remote_state = create_drain_remote_checkpoint(&hidden_inner, checkpoint).await?;
        (batches, remote_state)
    };

    let store = user_inner.options.store.clone();
    let storage_opt = user_inner.options.storage.clone();
    let (metric, drain_rot_seed) = hidden_inner
        .options
        .vector_columns
        .first()
        .map(|c| (c.metric, c.rot_seed))
        .unwrap_or((Metric::L2Sq, 0));
    // assign-skip: with global-aligned user superfiles (`vector.user_centroids:
    // global`) cluster c == cell c, so group by the row's own cluster ordinal
    // instead of the O(n·n_cent) per-row nearest-cell scoring.
    let assign_skip = config::global().vector.user_centroids == CentroidAlignment::Global;
    let column_name = column.clone();

    let drain_t0 = std::time::Instant::now();
    let drain_rss0 = proc_rss_mib();
    let n_batches = batches.len();
    // Carries per-cell counts cumulatively across batches; the centroids
    // are immutable (owned by the user manifest), so each batch's
    // `apply_cell_updates` builds on the prior batches' running totals.
    let mut running_clusters = clusters;
    // The batch budget bounds source materialization. Kmeans rows accumulate
    // in per-cell disk spills; complete cell IVFs and final worker shards are
    // built only after every source batch is durable.
    let drain_scratch = drain_scratch_dir(&remote_state.checkpoint.epoch_id);
    fs::create_dir_all(&drain_scratch)
        .map_err(|error| BuildError::Store(format!("drain scratch create: {error}")))?;
    let mut local_checkpoint =
        load_drain_local_checkpoint(&drain_scratch, &remote_state.checkpoint.epoch_id)?
            .unwrap_or_else(|| DrainLocalCheckpoint::new(remote_state.checkpoint.epoch_id.clone()));
    if local_checkpoint.batches_done > n_batches {
        return Err(BuildError::Store(format!(
            "drain local checkpoint completed {} of only {n_batches} batches",
            local_checkpoint.batches_done
        )));
    }

    let mut completed_shards = HashSet::new();
    let mut new_entries = Vec::new();
    let mut added_per_cell = local_checkpoint.added_per_cell.clone();
    let pending_entry_by_id: HashMap<String, Arc<SuperfileEntry>> = remote_state
        .entries
        .iter()
        .map(|entry| (entry.superfile_id.to_string(), Arc::clone(entry)))
        .collect();
    for remote_shard in &remote_state.checkpoint.completed_shards {
        if !completed_shards.insert(remote_shard.shard_id) {
            return Err(BuildError::Store(format!(
                "drain checkpoint repeats shard {}",
                remote_shard.shard_id
            )));
        }
        let entry = pending_entry_by_id
            .get(&remote_shard.superfile_id)
            .cloned()
            .ok_or_else(|| {
                BuildError::Store(format!(
                    "drain checkpoint shard {} entry {} is missing",
                    remote_shard.shard_id, remote_shard.superfile_id
                ))
            })?;
        if entry.partition_hint != Some(remote_shard.shard_id) {
            return Err(BuildError::Store(format!(
                "drain checkpoint shard {} entry has partition hint {:?}",
                remote_shard.shard_id, entry.partition_hint
            )));
        }
        storage
            .head(&superfile_storage_path(&entry.uri))
            .await
            .map_err(|error| {
                BuildError::Store(format!(
                    "drain checkpoint shard {} object is unavailable: {error}",
                    remote_shard.shard_id
                ))
            })?;
        for &(cell, count) in &remote_shard.cell_counts {
            match added_per_cell.insert(cell, count) {
                Some(existing) if existing != count => {
                    return Err(BuildError::Store(format!(
                        "drain checkpoint cell {cell} count {count} != local count {existing}"
                    )));
                }
                _ => {}
            }
        }
        new_entries.push(entry);
    }

    let mut cell_spills = HashMap::new();
    for (&cell, spill) in &local_checkpoint.spills {
        if completed_shards.contains(&(packed_cell_shard(cell, shard_count) as u32)) {
            continue;
        }
        let rerank_codec = RerankCodec::from_codec_id(spill.rerank_codec_id).ok_or_else(|| {
            BuildError::Store(format!(
                "cell {cell}: checkpoint has unknown codec id {}",
                spill.rerank_codec_id
            ))
        })?;
        cell_spills.insert(
            cell,
            MaterializedRowSpillWriter::resume(
                &drain_scratch,
                cell,
                MaterializedRowSpillState {
                    n_rows: spill.n_rows,
                    n_quants: spill.n_quants,
                    dim: spill.dim,
                    rabitq_len: spill.rabitq_len,
                    rerank_codec,
                },
            )?,
        );
    }
    let mut packed_cells = Vec::new();
    for (&cell, state) in &local_checkpoint.built_cells {
        let cell_shard = packed_cell_shard(cell, shard_count) as u32;
        if completed_shards.contains(&cell_shard) {
            continue;
        }
        packed_cells.push(restore_spilled_packed_cell(&drain_scratch, cell, state)?);
    }

    for (batch_idx, (_, batch_sources)) in batches.iter().enumerate() {
        if batch_idx < local_checkpoint.batches_done {
            continue;
        }
        let batch_t0 = std::time::Instant::now();
        // Timeline diagnostic only: snapshot GETs for this batch without
        // clearing the shared usage meter (a no-op when the env gate is off).
        let gets_before = if crate::storage::io_counters::timeline_enabled() {
            let snap = storage_opt.as_ref().map(|s| s.usage_meter().snapshot());
            crate::storage::io_counters::timeline_reset();
            snap
        } else {
            None
        };
        let read_concurrency = drain_read_concurrency();
        // Open this batch's user superfiles FULLY RESIDENT: the splice/materialize
        // read via `try_get_range_sync` on rayon workers, which needs the whole
        // superfile in memory — a lazy reader yields VectorReadError. Reuse a
        // resident cached reader if present, else fetch the full bytes + open.
        // `buffer_unordered` yields each open as it completes, so one straggler
        // read can't stall the fan-out window (order is irrelevant — rows are
        // bucketed by cell downstream). Routing-id resolution is resident (no
        // object-store I/O), so it rides each open's future and overlaps the
        // other reads' in-flight bytes.
        let readers: Vec<(Arc<SuperfileReader>, Vec<i128>)> =
            stream::iter(batch_sources.iter().map(|entry| {
                let entry = Arc::clone(entry);
                let store = Arc::clone(&store);
                let storage_opt = storage_opt.clone();
                let manifest = Arc::clone(&user_manifest);
                async move {
                    // Fully-resident only: the splice reads real vector bytes
                    // synchronously, which a promoted hybrid reader (sparse
                    // vector region) cannot serve.
                    let reader = match store.reader(&entry.uri) {
                        Ok(r) if r.is_fully_resident() => r,
                        _ => {
                            let storage = storage_opt.as_ref().ok_or_else(|| {
                                BuildError::Store(
                                    "drain requires storage to load user superfiles".into(),
                                )
                            })?;
                            let (bytes, _) = storage
                                .get(&entry.uri.storage_path())
                                .await
                                .map_err(|e| BuildError::Store(e.to_string()))?;
                            Arc::new(
                                SuperfileReader::open(bytes)
                                    .map_err(|e| BuildError::Store(e.to_string()))?,
                            )
                        }
                    };
                    let stable_ids = stable_ids_by_local_for_routing(&manifest, &entry, &reader)
                        .await
                        .map_err(|e| BuildError::Store(e.to_string()))?;
                    Ok::<_, BuildError>((reader, stable_ids))
                }
            }))
            .buffer_unordered(read_concurrency)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, BuildError>>()?;

        // The batch's superfile reads land here (opens are fully resident). The
        // timeline distinguishes a serial dependent chain (concurrency ~1x) from
        // overlapped reads (concurrency ~ buffered fan-out) — the lever for the
        // materialize phase. Gated on INFINO_IO_TIMELINE.
        if crate::storage::io_counters::timeline_enabled() {
            let spans = crate::storage::io_counters::timeline_take();
            let range_gets = match (storage_opt.as_ref(), gets_before.as_ref()) {
                (Some(s), Some(before)) => s.usage_meter().snapshot().since(before).get_count,
                _ => 0,
            };
            let min_start = spans.iter().map(|s| s.start_us).min().unwrap_or(0);
            let max_end = spans.iter().map(|s| s.end_us).max().unwrap_or(0);
            let wall_us = max_end.saturating_sub(min_start);
            let sum_us: u64 = spans
                .iter()
                .map(|s| s.end_us.saturating_sub(s.start_us))
                .sum();
            let bytes: u64 = spans.iter().map(|s| s.len).sum();
            let concurrency = if wall_us > 0 {
                sum_us as f64 / wall_us as f64
            } else {
                0.0
            };
            eprintln!(
                "[supertable drain] batch {}/{} materialize I/O: {} object reads, {:.1} MiB, wall {:.1}ms, Σdur {:.1}ms, implied concurrency {:.1}x ({} range-gets)",
                batch_idx + 1,
                n_batches,
                spans.len(),
                bytes as f64 / (1u64 << 20) as f64,
                wall_us as f64 / 1e3,
                sum_us as f64 / 1e3,
                concurrency,
                range_gets,
            );
        }

        // One consolidate mode, one shared checkpoint + pack/upload tail.
        // `drain_consolidate` selects how cells are produced — never a fallback.
        let batch_log = match consolidate {
            DrainConsolidate::Splice => {
                let column_name_ref = column_name.as_str();
                let stable_ids_per_input: Vec<Vec<i128>> =
                    readers.iter().map(|(_, ids)| ids.clone()).collect();
                let routed: HashMap<u32, (MergedIvfSubsection, Vec<i128>)> =
                    hidden_inner.options.writer_pool.install(
                        || -> Result<HashMap<u32, (MergedIvfSubsection, Vec<i128>)>, BuildError> {
                            let inputs: Vec<(&VectorReader, &str)> = readers
                                .iter()
                                .map(|(r, _)| {
                                    r.vec()
                                        .ok_or_else(|| {
                                            BuildError::Store(
                                                "user superfile missing vector index".into(),
                                            )
                                        })
                                        .map(|vr| (vr, column_name_ref))
                                })
                                .collect::<Result<_, _>>()?;
                            let clusters_ref = &running_clusters;
                            route_clusters_into_cells(
                                &inputs,
                                &stable_ids_per_input,
                                |centroid: &[f32]| {
                                    let mut assign = [0u32];
                                    clusters_ref.assign_rows(metric, centroid, &mut assign);
                                    vec![assign[0]]
                                },
                            )
                            .map_err(|e| e.into())
                        },
                    )?;
                let n_cells = routed.len();
                let dim = running_clusters.dim as usize;
                for (cell_id, (subsection, stable_ids)) in routed {
                    accumulate_splice_cell(
                        &mut packed_cells,
                        &mut local_checkpoint,
                        &mut added_per_cell,
                        &completed_shards,
                        shard_count,
                        drain_scratch.as_path(),
                        cell_id,
                        subsection,
                        stable_ids,
                        dim,
                        metric,
                    )?;
                }
                format!(
                    "splice: route+accumulate {:.1}ms, {n_cells} cell(s)",
                    batch_t0.elapsed().as_secs_f64() * 1e3,
                )
            }
            DrainConsolidate::Kmeans => {
                let column_for_mat = column_name.clone();
                let tombstone_cache = user_inner.tombstone_cache.clone();
                let now = time::Instant::now();
                let row_sets: Vec<Vec<MaterializedIvfRow>> =
                    stream::iter(readers.iter().zip(batch_sources.iter()).map(
                        |((reader, stable_ids), entry)| {
                            let column_for_mat = column_for_mat.clone();
                            let tombstone_cache = tombstone_cache.clone();
                            let entry = Arc::clone(entry);
                            async move {
                                let bitmap = tombstone_cache
                                    .as_ref()
                                    .map(|t| t.bitmap_for(entry.superfile_id, now))
                                    .transpose()
                                    .map_err(|e| BuildError::Store(e.to_string()))?;
                                materialized_user_rows_for_drain(
                                    reader,
                                    &column_for_mat,
                                    stable_ids,
                                    bitmap.as_deref(),
                                )
                                .await
                            }
                        },
                    ))
                    .buffered(commit_write_concurrency())
                    .collect::<Vec<_>>()
                    .await
                    .into_iter()
                    .collect::<Result<Vec<_>, BuildError>>()?;
                let t_mat = batch_t0.elapsed().as_secs_f64() * 1e3;

                let all_rows: Vec<MaterializedIvfRow> = row_sets.into_iter().flatten().collect();
                let n_batch_rows = all_rows.len();
                for writer in cell_spills.values_mut() {
                    writer.begin_batch();
                }
                let replica_target = drain_replica_target_factor();
                // Distinct corpus only: user superfiles already carry
                // commit-time boundary replicas (user-space recall rides on
                // them); without this dedup every ingest copy assigns beside
                // its primary and lands as a same-cell duplicate that wastes
                // top-k slots — measured at 100K/factor 1.5: 211,009 stored
                // rows for 100,000 distinct, 88,961 same-cell duplicate
                // pairs, post-drain recall 0.950 → 0.870. The zero-budget
                // fast path must dedup too (S11); it only skips re-assign.
                let mut seen_stable_ids: HashSet<i128> = HashSet::with_capacity(n_batch_rows);
                let distinct_rows: Vec<&MaterializedIvfRow> = all_rows
                    .iter()
                    .filter(|row| seen_stable_ids.insert(row.stable_id))
                    .collect();
                if assign_skip
                    && drain_replica_extra_budget(distinct_rows.len(), replica_target) == 0
                {
                    // Globally-aligned superfiles with no drain-side budget:
                    // trust ingest placement on the distinct set (replicas
                    // included as already stamped at commit).
                    for row in &distinct_rows {
                        spill_unfinished_shard_row(
                            &mut cell_spills,
                            &mut added_per_cell,
                            &completed_shards,
                            shard_count,
                            drain_scratch.as_path(),
                            row.cluster,
                            row,
                        )?;
                    }
                } else {
                    let replica_extra_budget =
                        drain_replica_extra_budget(distinct_rows.len(), replica_target);
                    let clusters_ref = &running_clusters;
                    // Shared admit context + 20% shortlist window: the same
                    // 1-bit prefilter the commit assign uses, so drain
                    // assignment compute scales with the window too.
                    let admit_ctx =
                        RabitqAdmitContext::new(clusters_ref.dim as usize, drain_rot_seed);
                    let window = opann::assignment_shortlist_window(clusters_ref.n_cent as usize);
                    let assignments: Vec<opann::BoundaryAssignment> =
                        hidden_inner.options.writer_pool.install(|| {
                            distinct_rows
                                .par_iter()
                                .map(|row| {
                                    opann::boundary_assignment_encoded(
                                        clusters_ref,
                                        metric,
                                        &row.encoded,
                                        &admit_ctx,
                                        window,
                                    )
                                })
                                .collect()
                        });
                    let mut replica_candidates: Vec<(usize, u32, f32)> = assignments
                        .iter()
                        .enumerate()
                        .flat_map(|(row_idx, assignment)| {
                            assignment
                                .replicas
                                .iter()
                                .flatten()
                                .map(move |&(cell, margin)| (row_idx, cell, margin))
                        })
                        .collect();
                    replica_candidates.sort_by(|a, b| a.2.total_cmp(&b.2));
                    for (row_idx, cell, _) in
                        replica_candidates.into_iter().take(replica_extra_budget)
                    {
                        spill_unfinished_shard_row(
                            &mut cell_spills,
                            &mut added_per_cell,
                            &completed_shards,
                            shard_count,
                            drain_scratch.as_path(),
                            cell,
                            distinct_rows[row_idx],
                        )?;
                    }
                    for (row, assignment) in distinct_rows.iter().zip(&assignments) {
                        spill_unfinished_shard_row(
                            &mut cell_spills,
                            &mut added_per_cell,
                            &completed_shards,
                            shard_count,
                            drain_scratch.as_path(),
                            assignment.primary,
                            row,
                        )?;
                    }
                }
                let mut checkpointed_spills = HashMap::with_capacity(cell_spills.len());
                for (&cell, writer) in &mut cell_spills {
                    let state = writer.checkpoint().map_err(BuildError::from)?;
                    checkpointed_spills.insert(
                        cell,
                        DrainLocalSpill {
                            n_rows: state.n_rows,
                            n_quants: state.n_quants,
                            dim: state.dim,
                            rabitq_len: state.rabitq_len,
                            rerank_codec_id: state.rerank_codec.codec_id(),
                        },
                    );
                }
                local_checkpoint.spills = checkpointed_spills;
                let t_spill = batch_t0.elapsed().as_secs_f64() * 1e3;
                format!(
                    "kmeans: materialize {:.1}ms + {} {:.1}ms, {} batch row(s) -> {} cell spill(s)",
                    t_mat,
                    if assign_skip {
                        "group(assign-skip)+spill"
                    } else {
                        "assign+spill"
                    },
                    t_spill - t_mat,
                    n_batch_rows,
                    cell_spills.len(),
                )
            }
        };

        local_checkpoint.batches_done = batch_idx + 1;
        local_checkpoint.added_per_cell = added_per_cell.clone();
        save_drain_local_checkpoint(&drain_scratch, &local_checkpoint)?;
        #[cfg(test)]
        maybe_fail_drain_for_test(
            &remote_state.checkpoint.epoch_id,
            DrainTestFailurePhase::AfterBatch,
            local_checkpoint.batches_done,
        )?;
        eprintln!(
            "[supertable drain] batch {}/{} ({} sf, {batch_log})",
            batch_idx + 1,
            n_batches,
            batch_sources.len(),
        );
    }

    // One task per final worker shard. Splice cells are already packed; each
    // kmeans worker streams its row-spilled cells one at a time, checkpoints
    // their completed IVF files, then assembles one MultiCellIvf.
    {
        let build_t0 = time::Instant::now();
        let scratch = drain_scratch.as_path();
        let n_cells_total = added_per_cell.len();
        let total_rows: u64 = added_per_cell.values().map(|count| u64::from(*count)).sum();
        let n_shards = shard_count;

        let mut cell_counts_by_shard: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
        for (&cell, &count) in &added_per_cell {
            let shard = packed_cell_shard(cell, n_shards) as u32;
            cell_counts_by_shard
                .entry(shard)
                .or_default()
                .push((cell, count));
        }
        for counts in cell_counts_by_shard.values_mut() {
            counts.sort_unstable_by_key(|(cell, _)| *cell);
        }
        let expected_shards = cell_counts_by_shard.len();

        crate::superfile::vector::builder::build_phase_timers::reset();
        let mut sources: Vec<(u32, DrainCellSource)> = packed_cells
            .into_iter()
            .map(|cell| (cell.cell_id, DrainCellSource::Packed(cell)))
            .collect();
        match consolidate {
            DrainConsolidate::Splice => {
                if !cell_spills.is_empty() {
                    return Err(BuildError::Store(
                        "splice drain must not leave materialized row spills".into(),
                    ));
                }
            }
            DrainConsolidate::Kmeans => {
                sources.extend(
                    cell_spills
                        .into_iter()
                        .map(|(cell, writer)| {
                            writer
                                .finish()
                                .map(|spill| (cell, DrainCellSource::Rows(spill)))
                                .map_err(BuildError::from)
                        })
                        .collect::<Result<Vec<_>, BuildError>>()?,
                );
            }
        }
        if sources.is_empty() && !added_per_cell.is_empty() {
            return Err(BuildError::Store(
                "drain has cell counts but no cell build sources".into(),
            ));
        }
        let mut shard_sources = group_cells_by_packed_shard(sources, n_shards);
        shard_sources.retain(|(shard_id, _)| !completed_shards.contains(shard_id));
        let checkpoint = Arc::new(Mutex::new(local_checkpoint));
        let vector_config = hidden_inner
            .options
            .vector_columns
            .first()
            .cloned()
            .ok_or_else(|| BuildError::Store("drain pack requires a vector column".into()))?;
        let prepared_shards: Vec<PreparedSuperfile> = fanout_shards(
            &hidden_inner.options.writer_pool,
            &shard_sources,
            |(shard_id, cells)| {
                let mut packed = Vec::with_capacity(cells.len());
                for (cell_id, source) in cells {
                    let cell = match source {
                        DrainCellSource::Packed(cell) => cell.clone(),
                        DrainCellSource::Rows(spill) => {
                            let cell = build_spilled_packed_cell_from_rows(
                                scratch,
                                *cell_id,
                                spill,
                                &vector_config,
                            )?;
                            {
                                let mut state = checkpoint.lock().map_err(|_| {
                                    BuildError::Store("drain checkpoint lock poisoned".into())
                                })?;
                                state.spills.remove(cell_id);
                                state.built_cells.insert(
                                    *cell_id,
                                    DrainLocalCell {
                                        n_docs: cell.n_docs,
                                        subsection_len: cell.subsection_len,
                                        rerank_codec_id: cell.rerank_codec.codec_id(),
                                    },
                                );
                                save_drain_local_checkpoint(&drain_scratch, &state)?;
                            }
                            spill.remove_files();
                            cell
                        }
                    };
                    packed.push((*cell_id, cell));
                }
                build_prepared_from_spilled_cells(&hidden_inner, scratch, *shard_id, &packed)
            },
        )?;
        local_checkpoint = checkpoint
            .lock()
            .map_err(|_| BuildError::Store("drain checkpoint lock poisoned".into()))?
            .clone();

        if prepared_shards.len() + completed_shards.len() > n_shards {
            return Err(BuildError::Store(format!(
                "drain produced {} packed shards for {n_shards} workers",
                prepared_shards.len() + completed_shards.len()
            )));
        }
        let publish = collect_prepared_superfiles(&hidden_inner, prepared_shards)?;
        if !publish.to_remove.is_empty() {
            return Err(BuildError::Store(
                "drain prepared removals while publishing new worker shards".into(),
            ));
        }
        let entry_by_uri: HashMap<SuperfileUri, Arc<SuperfileEntry>> = publish
            .new_entries
            .iter()
            .map(|entry| (entry.uri, Arc::clone(entry)))
            .collect();
        let pending_cache_inserts = publish.pending_cache_inserts;
        let pending_store_inserts = publish.pending_store_inserts;
        let multipart_threshold = hidden_inner.options.put_multipart_threshold_bytes;
        let put_futures = publish
            .pending_storage_writes
            .into_iter()
            .map(|(uri, bytes)| {
                let storage = Arc::clone(&storage);
                async move {
                    put_new_superfile_bytes(&storage, multipart_threshold, uri, bytes)
                        .await
                        .map(|()| uri)
                        .map_err(|error| BuildError::Store(error.to_string()))
                }
            });
        let mut uploads = stream::iter(put_futures).buffer_unordered(commit_write_concurrency());
        while let Some(uploaded) = uploads.next().await {
            let uri = uploaded?;
            let entry = entry_by_uri.get(&uri).cloned().ok_or_else(|| {
                BuildError::Store(format!("uploaded drain shard {} has no entry", uri.0))
            })?;
            let shard_id = entry.partition_hint.ok_or_else(|| {
                BuildError::Store(format!(
                    "uploaded drain shard {} has no partition hint",
                    uri.0
                ))
            })?;
            let cell_counts = cell_counts_by_shard
                .get(&shard_id)
                .cloned()
                .ok_or_else(|| {
                    BuildError::Store(format!(
                        "uploaded drain shard {shard_id} has no cell counts"
                    ))
                })?;
            remote_state.entries.push(Arc::clone(&entry));
            remote_state
                .checkpoint
                .completed_shards
                .push(DrainRemoteShard {
                    shard_id,
                    superfile_id: entry.superfile_id.to_string(),
                    cell_counts: cell_counts.clone(),
                });
            remote_state
                .checkpoint
                .completed_shards
                .sort_unstable_by_key(|shard| shard.shard_id);
            save_drain_remote_checkpoint(&hidden_inner, &mut remote_state).await?;
            #[cfg(test)]
            maybe_fail_drain_for_test(
                &remote_state.checkpoint.epoch_id,
                DrainTestFailurePhase::AfterShard,
                remote_state.checkpoint.completed_shards.len(),
            )?;
            completed_shards.insert(shard_id);
            new_entries.push(entry);

            for (cell, _) in cell_counts {
                local_checkpoint.spills.remove(&cell);
                if let Some(state) = local_checkpoint.built_cells.remove(&cell)
                    && let Ok(packed) = restore_spilled_packed_cell(&drain_scratch, cell, &state)
                {
                    remove_spilled_packed_cell(&packed);
                }
            }
            save_drain_local_checkpoint(&drain_scratch, &local_checkpoint)?;
        }
        if new_entries.len() != expected_shards {
            return Err(BuildError::Store(format!(
                "drain has {} completed shards but expected {expected_shards}",
                new_entries.len()
            )));
        }
        let n_shard_files = new_entries.len();

        // Grid cell counts are read only as a populated/empty marker (`== 0`),
        // never for their magnitude — the precise live-doc total is derived
        // from the files when it matters (e.g. split eligibility reads
        // tombstone-aware per-cell counts). This running sum is therefore an
        // approximate population signal, not an exact cumulative doc count.
        let mut cell_updates: HashMap<u32, u32> = HashMap::new();
        for (cell, added) in &added_per_cell {
            let base = running_clusters
                .counts
                .get(*cell as usize)
                .copied()
                .unwrap_or(0);
            cell_updates.insert(*cell, base.saturating_add(*added));
        }
        running_clusters = opann::apply_cell_updates(&running_clusters, &cell_updates);
        let mut new_drained = hidden_inner.manifest.load_full().get_drained_ranges();
        let drained_max = batches
            .iter()
            .flat_map(|(versions, _)| versions.iter().copied())
            .max()
            .unwrap_or(0);
        let lo = new_drained.prefix_end().map(|end| end + 1).unwrap_or(0);
        new_drained.insert_range(lo.min(drained_max), drained_max);
        // Grid + drained watermark must land in the same OCC attempt as the
        // shard membership append — never ArcSwap.store them beforehand
        // (contention refresh would drop the stamps; readers would also see
        // an advanced watermark without the new shards).
        let list_metadata = CommitListMetadata {
            partition_strategy: Some(PartitionStrategy::VectorCell {
                column: column.clone(),
                clusters: running_clusters.clone(),
                routing,
            }),
            drained_ranges: Some(new_drained),
            global_vector_index: None,
        };
        let no_removals: Vec<Arc<SuperfileEntry>> = Vec::new();
        let new_manifest = persist_commit_async(
            &hidden_inner,
            Arc::clone(&storage),
            new_entries,
            &no_removals,
            Vec::new(),
            Vec::new(),
            list_metadata,
        )
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;
        hidden_inner.manifest.store(Arc::new(new_manifest));
        apply_pending_store_inserts(&hidden_inner, pending_store_inserts);
        if !pending_cache_inserts.is_empty()
            && let Some(cache) = hidden_inner.options.disk_cache.as_ref()
        {
            warm_cache_after_commit(&hidden_inner, cache, pending_cache_inserts);
        }
        if let Err(error) = fs::remove_dir_all(&drain_scratch)
            && error.kind() != io::ErrorKind::NotFound
        {
            tracing::warn!("drain local checkpoint cleanup failed: {error}");
        }
        eprintln!(
            "[supertable drain] cell build: {} row(s), {} cell(s) -> {} packed shard superfile(s) for {} worker(s), {:.1}ms",
            total_rows,
            n_cells_total,
            n_shard_files,
            n_shards,
            build_t0.elapsed().as_secs_f64() * 1e3,
        );
        if crate::superfile::vector::builder::build_phase_timers::enabled() {
            let (train_ms, assign_ms, calib_ms) =
                crate::superfile::vector::builder::build_phase_timers::snapshot_ms();
            eprintln!(
                "[supertable drain] cell build phases (summed CPU, {n_cells_total} cells): train {train_ms:.1}ms + assign {assign_ms:.1}ms + calibrate {calib_ms:.1}ms",
            );
        }
    }

    eprintln!(
        "[supertable drain] done ({}, {} batch(es), budget {} sf): total {:.1}ms; RSS {} -> {} MiB",
        match consolidate {
            DrainConsolidate::Kmeans => "kmeans",
            DrainConsolidate::Splice => "splice",
        },
        n_batches,
        batch_cfg,
        drain_t0.elapsed().as_secs_f64() * 1e3,
        drain_rss0
            .map(|v| format!("{v:.0}"))
            .unwrap_or_else(|| "?".into()),
        proc_rss_mib()
            .map(|v| format!("{v:.0}"))
            .unwrap_or_else(|| "?".into()),
    );
    // Membership has settled: publish the slow-CAS entry blob and stamp its
    // ref (the per-batch `update`s cleared it). Hidden tables have no manifest
    // parts, so publication is required for reopen and cannot degrade to a
    // warning.
    refresh_slow_vector_state(&hidden_inner).await?;
    schedule_background_storage_reclaim(Arc::clone(&hidden_inner));
    Ok(())
}

/// Load Sq8+ε IVF rows from one hidden superfile.
///
/// - Legacy one-cell-per-file (`Ivf`): all rows (file == cell).
/// - Packed multi-cell (`MultiCellIvf`): only cells in `only_cells` when
///   provided; otherwise every cell in the directory. Rows keep cell-local
///   `local_doc_id`s; stable ids come from the inline region.
async fn load_materialized_rows_from_ivf_superfile(
    inner: &SupertableInner,
    entry: &Arc<SuperfileEntry>,
    column: &str,
    now: time::Instant,
    only_cells: Option<&[u32]>,
) -> Result<Vec<MaterializedIvfRow>, BuildError> {
    let (reader, bitmap) = open_ivf_reader_with_tombstones(inner, entry, now).await?;
    let vec_reader = reader
        .vec()
        .ok_or_else(|| BuildError::Store("IVF cell superfile missing vector index".into()))?;

    if vec_reader.is_multi_cell() {
        let groups =
            group_multicell_rows(vec_reader, column, only_cells, bitmap.as_deref()).await?;
        return Ok(groups.into_iter().flat_map(|(_, rows)| rows).collect());
    }

    let manifest = inner.manifest.load_full();
    let stable_ids = stable_ids_by_local_for_routing(&manifest, entry, &reader)
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;
    materialized_ivf_rows_in_doc_order(vec_reader, column, &stable_ids, bitmap.as_deref()).await
}

/// Per-cell live rows from a packed (`MultiCellIvf`) entry, opening it exactly
/// once (a single multi-cell decode) and returning `(cell_id, rows)` grouped by
/// cell with this entry's tombstones applied. The cell-split republish needs
/// each kept cell's rows separately; calling the flattening loader per cell
/// re-opened and re-decoded the same entry once per cell (S decodes where one
/// does).
async fn load_materialized_rows_by_cell_from_ivf_superfile(
    inner: &SupertableInner,
    entry: &Arc<SuperfileEntry>,
    column: &str,
    now: time::Instant,
    only_cells: &[u32],
) -> Result<Vec<(u32, Vec<MaterializedIvfRow>)>, BuildError> {
    let (reader, bitmap) = open_ivf_reader_with_tombstones(inner, entry, now).await?;
    let vec_reader = reader
        .vec()
        .ok_or_else(|| BuildError::Store("IVF cell superfile missing vector index".into()))?;
    if !vec_reader.is_multi_cell() {
        return Err(BuildError::Store(
            "per-cell row load requires a packed multi-cell entry".into(),
        ));
    }
    group_multicell_rows(vec_reader, column, Some(only_cells), bitmap.as_deref()).await
}

/// Open a maintenance reader for `entry` plus its tombstone bitmap (if a
/// tombstone cache is attached). Shared by the flattening and per-cell loaders.
async fn open_ivf_reader_with_tombstones(
    inner: &SupertableInner,
    entry: &Arc<SuperfileEntry>,
    now: time::Instant,
) -> Result<(Arc<SuperfileReader>, Option<Arc<roaring::RoaringBitmap>>), BuildError> {
    let storage = inner
        .options
        .storage
        .as_ref()
        .ok_or_else(|| BuildError::Store("cell maintenance requires storage".into()))?;
    let disk_cache = inner.options.disk_cache.as_ref();
    let bitmap = inner
        .tombstone_cache
        .as_ref()
        .map(|t| t.bitmap_for(entry.superfile_id, now))
        .transpose()
        .map_err(|e| BuildError::Store(e.to_string()))?;
    let reader = open_reader(&inner.options.store, disk_cache, Some(storage), entry, true)
        .await
        .map_err(|e| BuildError::Store(e.to_string()))?;
    Ok((reader, bitmap))
}

/// Decode a packed multi-cell entry into `(cell_id, rows)` groups, applying
/// `tombstones` per cell against the file-local doc base. `vec_reader` must be
/// multi-cell (callers check).
async fn group_multicell_rows(
    vec_reader: &VectorReader,
    column: &str,
    only_cells: Option<&[u32]>,
    tombstones: Option<&roaring::RoaringBitmap>,
) -> Result<Vec<(u32, Vec<MaterializedIvfRow>)>, BuildError> {
    let cells = vec_reader
        .materialized_cells_rows_async(only_cells)
        .await
        .ok_or_else(|| {
            BuildError::Store(format!(
                "IVF maintenance: multi-cell column '{column}' missing Sq8Residual index"
            ))
        })?;
    // File-local doc bases follow cell-directory order (same as parquet).
    let mut file_doc_base_by_cell: HashMap<u32, u32> = HashMap::new();
    let mut running = 0u32;
    for (ci, &cell_id) in vec_reader.packed_cell_ids().iter().enumerate() {
        file_doc_base_by_cell.insert(cell_id, running);
        let n = vec_reader
            .vector_columns_config()
            .nth(ci)
            .map(|c| c.n_docs)
            .unwrap_or(0);
        running = running.saturating_add(n);
    }
    let mut out = Vec::with_capacity(cells.len());
    for (cell_id, mut rows) in cells {
        let base = file_doc_base_by_cell.get(&cell_id).copied().unwrap_or(0);
        if let Some(bm) = tombstones {
            rows.retain(|r| !bm.contains(base + r.local_doc_id));
        }
        out.push((cell_id, rows));
    }
    Ok(out)
}

/// Per-cell doc counts from a packed (or legacy) entry. Legacy returns one
/// `(partition_hint_or_0, n_docs)` pair.
async fn cell_doc_counts_for_entry(
    inner: &SupertableInner,
    entry: &Arc<SuperfileEntry>,
) -> Result<Vec<(u32, u32)>, BuildError> {
    let storage = inner
        .options
        .storage
        .as_ref()
        .ok_or_else(|| BuildError::Store("cell maintenance requires storage".into()))?;
    let reader = open_reader(
        &inner.options.store,
        inner.options.disk_cache.as_ref(),
        Some(storage),
        entry,
        true,
    )
    .await
    .map_err(|e| BuildError::Store(e.to_string()))?;
    let v = reader
        .vec()
        .ok_or_else(|| BuildError::Store("IVF entry missing vector index".into()))?;
    if v.is_multi_cell() {
        Ok(v.packed_cell_ids()
            .iter()
            .filter_map(|&cell| {
                let n = v.packed_cell_n_docs(cell)?;
                Some((cell, n))
            })
            .collect())
    } else {
        let cell = entry.partition_hint.unwrap_or(0);
        Ok(vec![(cell, entry.n_docs as u32)])
    }
}

/// Coarse current RSS in MiB from `/proc/self/status` (Linux); `None` elsewhere
/// or on parse failure. Drain instrumentation only — not a hot path.
fn proc_rss_mib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: f64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb / 1024.0);
        }
    }
    None
}

/// One completed cell-IVF and its stable-id column, both spilled to local
/// scratch. The vector blob writer streams the subsection file directly into
/// its final packed shard; no shard ever rehydrates all of its cell bytes.
#[derive(Clone)]
struct SpilledPackedCell {
    cell_id: u32,
    n_docs: u32,
    rerank_codec: RerankCodec,
    subsection_len: u64,
    subsection_path: PathBuf,
    stable_ids_path: PathBuf,
}

enum DrainCellSource {
    Packed(SpilledPackedCell),
    Rows(SpilledCellRows),
}

impl MultiCellSubsectionSource for SpilledPackedCell {
    fn cell_id(&self) -> u32 {
        self.cell_id
    }

    fn n_docs(&self) -> u32 {
        self.n_docs
    }

    fn len(&self) -> u64 {
        self.subsection_len
    }

    fn rerank_codec(&self) -> RerankCodec {
        self.rerank_codec
    }

    fn write_to(&self, output: &mut dyn Write) -> Result<(), SuperfileBuildError> {
        let file = File::open(&self.subsection_path)?;
        let copied = io::copy(&mut BufReader::new(file), output)?;
        if copied != self.subsection_len {
            return Err(SuperfileBuildError::VectorSchemaMismatch(format!(
                "cell {} subsection spill length {copied} != expected {}",
                self.cell_id, self.subsection_len
            )));
        }
        Ok(())
    }
}

impl MultiCellSubsectionSource for &SpilledPackedCell {
    fn cell_id(&self) -> u32 {
        (*self).cell_id()
    }

    fn n_docs(&self) -> u32 {
        (*self).n_docs()
    }

    fn len(&self) -> u64 {
        (*self).len()
    }

    fn rerank_codec(&self) -> RerankCodec {
        (*self).rerank_codec()
    }

    fn write_to(&self, output: &mut dyn Write) -> Result<(), SuperfileBuildError> {
        (*self).write_to(output)
    }
}

fn spill_packed_cell(
    scratch: &Path,
    cell_id: u32,
    subsection: MergedIvfSubsection,
    stable_ids: &[i128],
) -> Result<SpilledPackedCell, BuildError> {
    if stable_ids.len() != subsection.n_docs as usize {
        return Err(BuildError::Store(format!(
            "cell {cell_id}: stable_ids len {} != subsection n_docs {}",
            stable_ids.len(),
            subsection.n_docs
        )));
    }

    let subsection_path = scratch.join(format!("cell-{cell_id}.ivf"));
    let subsection_temp = scratch.join(format!("cell-{cell_id}.ivf.tmp"));
    {
        let mut subsection_file = File::create(&subsection_temp)
            .map_err(|error| BuildError::Store(format!("cell subsection create: {error}")))?;
        subsection_file
            .write_all(&subsection.bytes)
            .map_err(|error| BuildError::Store(format!("cell subsection write: {error}")))?;
        subsection_file
            .sync_all()
            .map_err(|error| BuildError::Store(format!("cell subsection fsync: {error}")))?;
    }
    fs::rename(&subsection_temp, &subsection_path)
        .map_err(|error| BuildError::Store(format!("cell subsection rename: {error}")))?;
    let subsection_len = subsection.bytes.len() as u64;

    let stable_ids_path = scratch.join(format!("cell-{cell_id}.ids"));
    let stable_ids_temp = scratch.join(format!("cell-{cell_id}.ids.tmp"));
    {
        let ids_file = File::create(&stable_ids_temp)
            .map_err(|error| BuildError::Store(format!("cell ids create: {error}")))?;
        let mut writer = BufWriter::new(ids_file);
        for stable_id in stable_ids {
            writer
                .write_all(&stable_id.to_le_bytes())
                .map_err(|error| BuildError::Store(format!("cell ids write: {error}")))?;
        }
        writer
            .flush()
            .map_err(|error| BuildError::Store(format!("cell ids flush: {error}")))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| BuildError::Store(format!("cell ids fsync: {error}")))?;
    }
    fs::rename(&stable_ids_temp, &stable_ids_path)
        .map_err(|error| BuildError::Store(format!("cell ids rename: {error}")))?;

    Ok(SpilledPackedCell {
        cell_id,
        n_docs: subsection.n_docs,
        rerank_codec: subsection.rerank_codec,
        subsection_len,
        subsection_path,
        stable_ids_path,
    })
}

fn build_spilled_packed_cell_from_rows(
    scratch: &Path,
    cell_id: u32,
    spill: &SpilledCellRows,
    vector_config: &VectorConfig,
) -> Result<SpilledPackedCell, BuildError> {
    let subsection_path = scratch.join(format!("cell-{cell_id}.ivf"));
    let subsection_temp = scratch.join(format!("cell-{cell_id}.ivf.tmp"));
    let stable_ids_path = scratch.join(format!("cell-{cell_id}.ids"));
    let stable_ids_temp = scratch.join(format!("cell-{cell_id}.ids.tmp"));
    let cell_config = drain_cell_vector_config(vector_config, spill.n_rows());
    let built = build_merged_subsection_from_spilled_materialized(
        cell_config,
        spill,
        &subsection_temp,
        &stable_ids_temp,
        scratch,
    )?;
    fs::rename(&subsection_temp, &subsection_path)
        .map_err(|error| BuildError::Store(format!("cell subsection rename: {error}")))?;
    fs::rename(&stable_ids_temp, &stable_ids_path)
        .map_err(|error| BuildError::Store(format!("cell ids rename: {error}")))?;
    Ok(SpilledPackedCell {
        cell_id,
        n_docs: built.n_docs,
        rerank_codec: built.rerank_codec,
        subsection_len: built.subsection_len,
        subsection_path,
        stable_ids_path,
    })
}

fn restore_spilled_packed_cell(
    scratch: &Path,
    cell_id: u32,
    state: &DrainLocalCell,
) -> Result<SpilledPackedCell, BuildError> {
    let rerank_codec = RerankCodec::from_codec_id(state.rerank_codec_id).ok_or_else(|| {
        BuildError::Store(format!(
            "cell {cell_id}: checkpoint has unknown codec id {}",
            state.rerank_codec_id
        ))
    })?;
    let subsection_path = scratch.join(format!("cell-{cell_id}.ivf"));
    let stable_ids_path = scratch.join(format!("cell-{cell_id}.ids"));
    let subsection_size = fs::metadata(&subsection_path)
        .map_err(|error| BuildError::Store(format!("cell subsection metadata: {error}")))?
        .len();
    if subsection_size != state.subsection_len {
        return Err(BuildError::Store(format!(
            "cell {cell_id}: checkpointed subsection length {} != file length {subsection_size}",
            state.subsection_len
        )));
    }
    let ids_size = fs::metadata(&stable_ids_path)
        .map_err(|error| BuildError::Store(format!("cell ids metadata: {error}")))?
        .len();
    let expected_ids_size = u64::from(state.n_docs) * STABLE_ID_BYTES as u64;
    if ids_size != expected_ids_size {
        return Err(BuildError::Store(format!(
            "cell {cell_id}: checkpointed ids length {expected_ids_size} != file length {ids_size}"
        )));
    }
    Ok(SpilledPackedCell {
        cell_id,
        n_docs: state.n_docs,
        rerank_codec,
        subsection_len: state.subsection_len,
        subsection_path,
        stable_ids_path,
    })
}

fn remove_spilled_packed_cell(cell: &SpilledPackedCell) {
    let _ = fs::remove_file(&cell.subsection_path);
    let _ = fs::remove_file(&cell.stable_ids_path);
}

fn load_merged_from_spilled(
    cell: &SpilledPackedCell,
    dim: usize,
) -> Result<(MergedIvfSubsection, Vec<i128>), BuildError> {
    let bytes = fs::read(&cell.subsection_path)
        .map_err(|error| BuildError::Store(format!("cell subsection spill read: {error}")))?;
    if bytes.len() as u64 != cell.subsection_len {
        return Err(BuildError::Store(format!(
            "cell {}: spill length {} != expected {}",
            cell.cell_id,
            bytes.len(),
            cell.subsection_len
        )));
    }
    if bytes.len() < SUB_HEADER_SIZE + CRC_BYTES {
        return Err(BuildError::Store(format!(
            "cell {}: spilled subsection too short",
            cell.cell_id
        )));
    }
    let centroids_off = u64::from_le_bytes(
        bytes[sub_hdr::CENTROIDS_OFF_OFF..sub_hdr::CENTROIDS_OFF_OFF + 8]
            .try_into()
            .expect("8-byte centroids off"),
    ) as usize;
    let cluster_idx_off = u64::from_le_bytes(
        bytes[sub_hdr::CLUSTER_IDX_OFF_OFF..sub_hdr::CLUSTER_IDX_OFF_OFF + 8]
            .try_into()
            .expect("8-byte cluster idx off"),
    ) as usize;
    let summary_off = u64::from_le_bytes(
        bytes[sub_hdr::SUMMARY_OFF_OFF..sub_hdr::SUMMARY_OFF_OFF + 8]
            .try_into()
            .expect("8-byte summary off"),
    ) as usize;
    let codec_meta_size = u32::from_le_bytes(
        bytes[sub_hdr::CODEC_META_SIZE_OFF..sub_hdr::CODEC_META_SIZE_OFF + U32_BYTES]
            .try_into()
            .expect("4-byte codec meta size"),
    ) as usize;
    if cluster_idx_off < centroids_off || !(cluster_idx_off - centroids_off).is_multiple_of(dim * 4)
    {
        return Err(BuildError::Store(format!(
            "cell {}: invalid centroid region for dim {dim}",
            cell.cell_id
        )));
    }
    let n_cent = (cluster_idx_off - centroids_off) / (dim * 4);
    let codec_meta_off = cluster_idx_off + n_cent * CLUSTER_IDX_ENTRY_BYTES;
    let ids = read_spilled_stable_ids(cell)?;
    Ok((
        MergedIvfSubsection {
            bytes,
            n_cent,
            n_docs: cell.n_docs,
            rerank_codec: cell.rerank_codec,
            summary_offset_in_sub: summary_off,
            codec_meta_offset_in_sub: if codec_meta_size == 0 {
                0
            } else {
                codec_meta_off
            },
            codec_meta_size,
        },
        ids,
    ))
}

/// Accumulate one splice-routed cell into the shared packed-cell scratch.
///
/// First touch spills the routed subsection. A later batch that routes into
/// the same cell concatenates clusters with [`merge_fragment_subsections`]
/// (the same verbatim `splice_fragments_into_cell` primitive) — not a second
/// consolidate mode and not a silent fallback.
fn accumulate_splice_cell(
    packed_cells: &mut Vec<SpilledPackedCell>,
    local_checkpoint: &mut DrainLocalCheckpoint,
    added_per_cell: &mut HashMap<u32, u32>,
    completed_shards: &HashSet<u32>,
    shard_count: usize,
    scratch: &Path,
    cell_id: u32,
    subsection: MergedIvfSubsection,
    stable_ids: Vec<i128>,
    dim: usize,
    metric: Metric,
) -> Result<(), BuildError> {
    let shard = packed_cell_shard(cell_id, shard_count) as u32;
    if completed_shards.contains(&shard) {
        return Ok(());
    }

    let (subsection, stable_ids) =
        match packed_cells.iter().position(|cell| cell.cell_id == cell_id) {
            Some(idx) => {
                let existing = packed_cells.swap_remove(idx);
                let (left, left_ids) = load_merged_from_spilled(&existing, dim)?;
                remove_spilled_packed_cell(&existing);
                local_checkpoint.built_cells.remove(&cell_id);
                merge_fragment_subsections(&left, &left_ids, &subsection, &stable_ids, dim, metric)?
            }
            None => (subsection, stable_ids),
        };

    let n_docs = subsection.n_docs;
    let packed = spill_packed_cell(scratch, cell_id, subsection, &stable_ids)?;
    local_checkpoint.built_cells.insert(
        cell_id,
        DrainLocalCell {
            n_docs: packed.n_docs,
            subsection_len: packed.subsection_len,
            rerank_codec_id: packed.rerank_codec.codec_id(),
        },
    );
    packed_cells.push(packed);
    added_per_cell.insert(cell_id, n_docs);
    Ok(())
}

fn read_spilled_stable_ids(cell: &SpilledPackedCell) -> Result<Vec<i128>, BuildError> {
    let mut reader = BufReader::new(
        File::open(&cell.stable_ids_path)
            .map_err(|error| BuildError::Store(format!("cell ids spill open: {error}")))?,
    );
    let mut ids = Vec::with_capacity(cell.n_docs as usize);
    let mut encoded = [0u8; STABLE_ID_BYTES];
    for _ in 0..cell.n_docs {
        reader
            .read_exact(&mut encoded)
            .map_err(|error| BuildError::Store(format!("cell ids spill read: {error}")))?;
        ids.push(i128::from_le_bytes(encoded));
    }
    Ok(ids)
}

/// Drain packed-layout shard count: align with the writer pool width
/// (same rule as ingest's per-commit shard count).
fn packed_cell_shard_count(options: &SupertableOptions) -> usize {
    options.writer_pool.current_num_threads().max(1)
}

/// Shared cell → packed-shard mapping: `cell_id % shard_count`.
fn packed_cell_shard(cell: u32, shard_count: usize) -> usize {
    debug_assert!(shard_count > 0);
    (cell as usize) % shard_count
}

/// Group `(cell_id, payload)` into `shard_count` buckets by `cell % N`.
fn group_cells_by_packed_shard<T>(
    cells: Vec<(u32, T)>,
    shard_count: usize,
) -> Vec<(u32, Vec<(u32, T)>)> {
    debug_assert!(shard_count > 0);
    let mut buckets: Vec<Vec<(u32, T)>> = (0..shard_count).map(|_| Vec::new()).collect();
    for (cell, payload) in cells {
        buckets[packed_cell_shard(cell, shard_count)].push((cell, payload));
    }
    buckets
        .into_iter()
        .enumerate()
        .filter(|(_, cells)| !cells.is_empty())
        .map(|(shard, mut cells)| {
            cells.sort_unstable_by_key(|(cell, _)| *cell);
            (shard as u32, cells)
        })
        .collect()
}

/// One commit-buffer row for the shared assign+pack core.
#[derive(Clone, Copy)]
enum PackRow<'a> {
    Fp32 { stable_id: i128, vector: &'a [f32] },
}

/// One cell after boundary assignment, before IVF subsection build.
/// Packing (k-means + encode) belongs in the parallel shard stage — not here.
struct AssignedCellGroup<'a> {
    cell_id: u32,
    /// `(stable_id, is_primary, row)` sorted primary-first then by id.
    members: Vec<(i128, bool, PackRow<'a>)>,
}

/// One packed cell IVF. Primary-vs-stub markers live on
/// [`AssignedCellGroup::members`]; the commit writer consumes them **before**
/// pack (Parquet keeps primaries only), and the hidden drain indexes every
/// posting, so the packed group carries no separate marker copy.
struct PackedCellGroup {
    cell_id: u32,
    subsection: MergedIvfSubsection,
    #[cfg(test)]
    stable_ids: Vec<i128>,
}

fn pack_row_stable_id(row: PackRow<'_>) -> i128 {
    match row {
        PackRow::Fp32 { stable_id, .. } => stable_id,
    }
}

/// Commit assignment core: fp32 rows in, boundary assignment and replica
/// budget applied once, cell buckets out. Does **not** build IVF subsections —
/// that runs in the shard-stage pack (parallel). Boundary replicas are vector
/// postings only; callers decide which primaries become Parquet rows.
fn assign_cells<'a>(
    rows: &[PackRow<'a>],
    clusters: &ClusterCentroids,
    metric: Metric,
    rot_seed: u64,
    replica_target_factor: f32,
) -> Result<Vec<AssignedCellGroup<'a>>, BuildError> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let replica_extra_budget = drain_replica_extra_budget(rows.len(), replica_target_factor);
    // Per-row nearest-cell scoring is the commit CPU wave: run it on the
    // ambient rayon pool (callers wrap this in `writer_pool.install`).
    // One shared admit context per batch (rotation / quantizer / cosine
    // table); each row is 1-bit shortlisted over the grid and exact-scored
    // only inside the 20% window, so assignment compute scales with the
    // window instead of the full cell count.
    let admit_ctx = RabitqAdmitContext::new(clusters.dim as usize, rot_seed);
    let window = opann::assignment_shortlist_window(clusters.n_cent as usize);
    let assignments: Vec<opann::BoundaryAssignment> = rows
        .par_iter()
        .map(|row| match *row {
            PackRow::Fp32 { vector, .. } => {
                opann::boundary_assignment_fp32(clusters, metric, vector, &admit_ctx, window)
            }
        })
        .collect();

    let mut replica_candidates: Vec<(usize, u32, f32)> = assignments
        .iter()
        .enumerate()
        .flat_map(|(row_idx, assignment)| {
            assignment
                .replicas
                .iter()
                .flatten()
                .map(move |&(cell, margin)| (row_idx, cell, margin))
        })
        .collect();
    replica_candidates.sort_by(|a, b| a.2.total_cmp(&b.2));

    let mut buckets: HashMap<u32, Vec<(i128, bool, PackRow<'a>)>> = HashMap::new();
    for (row_idx, cell, _) in replica_candidates.into_iter().take(replica_extra_budget) {
        let row = rows[row_idx];
        buckets
            .entry(cell)
            .or_default()
            .push((pack_row_stable_id(row), false, row));
    }
    for (row, assignment) in rows.iter().zip(&assignments) {
        buckets
            .entry(assignment.primary)
            .or_default()
            .push((pack_row_stable_id(*row), true, *row));
    }

    let mut out = Vec::with_capacity(buckets.len());
    for (cell_id, mut members) in buckets {
        members.sort_by_key(|(stable_id, is_primary, _)| (!*is_primary, *stable_id));
        out.push(AssignedCellGroup { cell_id, members });
    }
    out.sort_unstable_by_key(|group| group.cell_id);
    Ok(out)
}

/// Size one cell's fine IVF so one run is approximately
/// [`DRAIN_FINE_RUN_TARGET_BYTES`]. The stride counts every per-row byte in
/// the packed IVF: RaBitQ estimate code, local id, Sq8+epsilon rerank bytes,
/// inline stable id, and the conservative norm word.
fn drain_cell_vector_config(cfg: &VectorConfig, n_rows: usize) -> VectorConfig {
    debug_assert!(n_rows > 0);
    let dim = cfg.dim;
    let rerank_codec = if cfg.rerank_codec.is_sq8_residual_family() {
        cfg.rerank_codec
    } else {
        RerankCodec::Sq8Residual
    };
    let rabitq_bytes = dim.div_ceil(u8::BITS as usize);
    let rerank_bytes = rerank_codec.per_vector_bytes(dim);
    let row_stride =
        rabitq_bytes + DOC_ID_BYTES + rerank_bytes + STABLE_ID_BYTES + mem::size_of::<f32>();
    let rows_per_run = (DRAIN_FINE_RUN_TARGET_BYTES / row_stride.max(1)).max(1);
    let n_cent = n_rows.div_ceil(rows_per_run).clamp(1, n_rows);
    VectorConfig {
        n_cent,
        rerank_codec,
        provided_centroids: None,
        ..cfg.clone()
    }
}

fn drain_pack_assigned_cell(
    group: AssignedCellGroup<'_>,
    cfg: &VectorConfig,
) -> Result<PackedCellGroup, BuildError> {
    let AssignedCellGroup { cell_id, members } = group;
    if members.is_empty() {
        return Err(BuildError::Store(format!(
            "cell {cell_id}: assign produced an empty bucket"
        )));
    }
    let dim = cfg.dim;
    let cell_cfg = drain_cell_vector_config(cfg, members.len());
    let stable_ids: Vec<i128> = members.iter().map(|(stable_id, _, _)| *stable_id).collect();
    let mut corpus = Vec::with_capacity(members.len() * dim);
    for (_, _, row) in &members {
        match *row {
            PackRow::Fp32 { vector, .. } => corpus.extend_from_slice(vector),
        }
    }
    // Drain's fp32 in-memory stream pack (why fp32 support exists).
    let subsection = build_merged_subsection_from_fp32(cell_cfg, Arc::new(corpus), &stable_ids)?;
    Ok(PackedCellGroup {
        cell_id,
        subsection,
        #[cfg(test)]
        stable_ids,
    })
}

/// Build one multi-cell packed superfile: many complete cell-IVFs in one
/// Parquet object, `partition_hint = shard_id`.
fn build_one_shard_from_packed_cells(
    cells: Vec<(u32, MergedIvfSubsection, Vec<i128>)>,
    options: &SupertableOptions,
) -> Result<ShardOutput, BuildError> {
    if cells.is_empty() {
        return Err(BuildError::NoDocsToBuild);
    }
    // Sort by cell_id up front so the concatenated `_id` column order matches
    // the subsection order the builder re-sorts into — a caller passing cells
    // out of cell_id order would otherwise diverge parquet `_id` from the
    // vector rows (parity with the sibling shard builder).
    let mut cells = cells;
    cells.sort_by_key(|(cell_id, _, _)| *cell_id);
    let mut stable_ids: Vec<i128> = Vec::new();
    let mut subsections: Vec<(u32, MergedIvfSubsection)> = Vec::with_capacity(cells.len());
    for (cell_id, sub, ids) in cells {
        if ids.len() != sub.n_docs as usize {
            return Err(BuildError::Store(format!(
                "cell {cell_id}: stable_ids len {} != subsection n_docs {}",
                ids.len(),
                sub.n_docs
            )));
        }
        stable_ids.extend_from_slice(&ids);
        subsections.push((cell_id, sub));
    }
    let id_array = Decimal128Array::from_iter_values(stable_ids.iter().copied())
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

    let mut builder = SuperfileBuilder::new(
        options
            .builder_options()
            .with_vector_layout(VectorLayout::MultiCellIvf),
    )?;
    builder.add_batch_ids_only(&scalar)?;
    builder.set_prebuilt_multi_cell_ivfs(subsections)?;

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

/// Prepare a packed multi-cell shard for publish (`partition_hint = shard_id`).
fn build_prepared_from_packed_cells(
    inner: &SupertableInner,
    shard_id: u32,
    cells: Vec<(u32, MergedIvfSubsection, Vec<i128>)>,
) -> Result<PreparedSuperfile, BuildError> {
    let shard = build_one_shard_from_packed_cells(cells, &inner.options)?;
    let prepared = prepare_superfile(inner, shard)?.ok_or(BuildError::NoDocsToBuild)?;
    let entry = finish_superfile_entry(prepared.entry, Some(shard_id))?;
    Ok(PreparedSuperfile {
        entry,
        bytes_for_store: prepared.bytes_for_store,
        bytes_for_storage: prepared.bytes_for_storage,
        bytes_for_cache: prepared.bytes_for_cache,
    })
}

/// Build exactly one packed drain superfile for one writer-pool shard.
///
/// Cell-IVFs stay disk-backed while the shared vector/superfile streamers
/// assemble the output. The completed file is mmap-backed into `Bytes`, then
/// handed to the ordinary `prepare_superfile` path so summaries, layout hints,
/// cache disposition, and manifest entry construction are not duplicated.
fn build_prepared_from_spilled_cells(
    inner: &SupertableInner,
    scratch: &Path,
    shard_id: u32,
    cells: &[(u32, SpilledPackedCell)],
) -> Result<PreparedSuperfile, BuildError> {
    if cells.is_empty() {
        return Err(BuildError::NoDocsToBuild);
    }
    let mut ordered: Vec<&SpilledPackedCell> = cells.iter().map(|(_, cell)| cell).collect();
    ordered.sort_unstable_by_key(|cell| cell.cell_id);

    let n_docs = ordered
        .iter()
        .map(|cell| cell.n_docs as usize)
        .sum::<usize>();
    let scalar_schema = inner.options.scalar_schema();
    let mut scalar_stats = HashMap::new();
    let mut builder = SuperfileBuilder::new(
        inner
            .options
            .builder_options()
            .with_vector_layout(VectorLayout::MultiCellIvf),
    )?;
    let mut id_min = i128::MAX;
    let mut id_max = i128::MIN;
    let mut ids_seen = 0usize;
    for cell in &ordered {
        let mut reader = BufReader::new(
            File::open(&cell.stable_ids_path)
                .map_err(|error| BuildError::Store(format!("cell ids spill open: {error}")))?,
        );
        let mut remaining = cell.n_docs as usize;
        while remaining > 0 {
            let take = remaining.min(DRAIN_ID_BATCH_ROWS);
            let mut ids = Vec::with_capacity(take);
            let mut encoded = [0u8; STABLE_ID_BYTES];
            for _ in 0..take {
                reader
                    .read_exact(&mut encoded)
                    .map_err(|error| BuildError::Store(format!("cell ids spill read: {error}")))?;
                let id = i128::from_le_bytes(encoded);
                id_min = id_min.min(id);
                id_max = id_max.max(id);
                ids.push(id);
            }
            let id_array = Decimal128Array::from_iter_values(ids)
                .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
                .expect("invariant: precision 38 + scale 0 always valid for any i128 payload");
            let scalar =
                RecordBatch::try_new(scalar_schema.clone(), vec![Arc::new(id_array) as ArrayRef])
                    .map_err(|_| BuildError::BatchSchemaMismatch)?;
            ScalarStatsAgg::merge(
                &mut scalar_stats,
                &ScalarStatsAgg::from_batch(&scalar_schema, &scalar),
            );
            builder.add_batch_ids_only(&scalar)?;
            ids_seen += take;
            remaining -= take;
        }
    }
    if ids_seen != n_docs {
        return Err(BuildError::Store(format!(
            "shard {shard_id}: stable id count {ids_seen} != expected {n_docs}"
        )));
    }

    let mut output = NamedTempFile::new_in(scratch)
        .map_err(|error| BuildError::Store(format!("packed shard temp create: {error}")))?;
    builder.finish_multi_cell_sources_to(&ordered, BufWriter::new(output.as_file_mut()))?;
    output
        .as_file_mut()
        .flush()
        .map_err(|error| BuildError::Store(format!("packed shard temp flush: {error}")))?;
    let bytes = mmap_readonly_bytes(output.path())
        .map_err(|error| BuildError::Store(format!("packed shard mmap: {error}")))?;

    let (id_min, id_max) = if n_docs == 0 {
        (0, 0)
    } else {
        (id_min, id_max)
    };
    let shard = ShardOutput {
        bytes,
        n_docs: n_docs as u64,
        id_min,
        id_max,
        scalar_stats,
    };
    let prepared = prepare_superfile(inner, shard)?.ok_or(BuildError::NoDocsToBuild)?;
    let entry = finish_superfile_entry(prepared.entry, Some(shard_id))?;
    Ok(PreparedSuperfile {
        entry,
        bytes_for_store: prepared.bytes_for_store,
        bytes_for_storage: prepared.bytes_for_storage,
        bytes_for_cache: prepared.bytes_for_cache,
    })
}

/// Commit vector path — drain's flow with a Parquet+FTS finish:
///
/// 1. assign the **whole buffer** to global cells in one pass (drain's core;
///    the boundary-replica budget is batch-global, exactly like drain),
/// 2. group whole cells into ≤ `n_writers` shard files (`cell % N` — drain's
///    [`group_cells_by_packed_shard`]),
/// 3. each writer: `rayon::join` — drain pack (fp32→Sq8→materialized fine
///    IVF) ‖ Parquet+FTS for that shard's primary rows — then splice + finish.
///
/// Rows are resharded by centroid distance instead of arrival time; drain
/// never writes superfiles or touches S3 here — the writer publishes through
/// the normal batch path.
fn commit_shards_via_drain(
    buffer: &[BufferedBatch],
    inner: &SupertableInner,
    clusters: &ClusterCentroids,
    metric: Metric,
) -> Result<(Vec<ShardOutput>, Vec<Option<u32>>), BuildError> {
    let stage_t0 = time::Instant::now();
    let vc = inner
        .options
        .vector_columns
        .first()
        .cloned()
        .ok_or_else(|| BuildError::Store("drain-commit requires a vector column".into()))?;
    let dim = vc.dim;
    if dim != clusters.dim as usize {
        return Err(BuildError::Store(format!(
            "commit vector dim {dim} does not match global grid dim {}",
            clusters.dim
        )));
    }

    // Collect ids + scalar batches; vectors stay in their Arrow buffers
    // behind zero-copy views (no flatten — see `VectorColumnView`).
    let mut stable_ids: Vec<i128> = Vec::new();
    let mut scalar_batches: Vec<&RecordBatch> = Vec::with_capacity(buffer.len());
    for buffered in buffer {
        let id_col = buffered
            .scalar
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .ok_or_else(|| {
                BuildError::IdColumnWrongType(
                    inner.options.id_column.clone(),
                    "<id column not Decimal128 at runtime>".to_string(),
                )
            })?;
        for i in 0..id_col.len() {
            stable_ids.push(id_col.value(i));
        }
        scalar_batches.push(&buffered.scalar);
    }
    if stable_ids.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let vector_views: Vec<VectorColumnView<'_>> = inner
        .options
        .vector_columns
        .iter()
        .enumerate()
        .map(|(col_idx, col)| VectorColumnView::over(buffer, col_idx, col.dim))
        .collect();
    let primary_view = vector_views
        .first()
        .ok_or_else(|| BuildError::Store("drain-commit missing vector values".into()))?;
    if primary_view.n_rows() != stable_ids.len() {
        return Err(BuildError::Store(format!(
            "commit vector rows {} != id rows {}",
            primary_view.n_rows(),
            stable_ids.len()
        )));
    }

    let scalar_schema = inner.options.scalar_schema();
    let source_scalar = concat_batches(&scalar_schema, scalar_batches.iter().copied())
        .map_err(|err| BuildError::Store(err.to_string()))?;
    let local_by_id: HashMap<i128, u32> = stable_ids
        .iter()
        .enumerate()
        .map(|(local, &id)| (id, local as u32))
        .collect();
    let flatten_elapsed = stage_t0.elapsed();

    // One global assign over the batch (drain's core; runs on the writer pool).
    let rows: Vec<PackRow<'_>> = stable_ids
        .iter()
        .enumerate()
        .map(|(local, &stable_id)| {
            Ok(PackRow::Fp32 {
                stable_id,
                vector: primary_view.row(local)?,
            })
        })
        .collect::<Result<_, BuildError>>()?;
    let replica_target = drain_replica_target_factor();
    let assigned = inner
        .options
        .writer_pool
        .install(|| assign_cells(&rows, clusters, metric, vc.rot_seed, replica_target))?;
    let assign_elapsed = stage_t0.elapsed().saturating_sub(flatten_elapsed);
    let assigned_cells: Vec<(u32, AssignedCellGroup<'_>)> = assigned
        .into_iter()
        .map(|group| (group.cell_id, group))
        .collect();
    let packed_shards =
        group_cells_by_packed_shard(assigned_cells, packed_cell_shard_count(&inner.options));

    let options = &inner.options;
    let shard_outputs = fanout_shards(&inner.options.writer_pool, &packed_shards, |task| {
        let (shard_id, cells) = task;
        build_one_packed_shard_via_drain(
            cells,
            &source_scalar,
            &vector_views,
            &local_by_id,
            options,
            &vc,
        )
        .map(|output| output.map(|output| (*shard_id, output)))
    })?;
    let fanout_elapsed = stage_t0
        .elapsed()
        .saturating_sub(flatten_elapsed)
        .saturating_sub(assign_elapsed);
    if crate::storage::io_counters::timeline_enabled() {
        eprintln!(
            "[supertable commit] flatten {:.1}ms + assign {:.1}ms + shard pack/finish {:.1}ms",
            flatten_elapsed.as_secs_f64() * 1e3,
            assign_elapsed.as_secs_f64() * 1e3,
            fanout_elapsed.as_secs_f64() * 1e3,
        );
    }

    let mut outputs = Vec::with_capacity(shard_outputs.len());
    let mut cell_hints = Vec::with_capacity(shard_outputs.len());
    for entry in shard_outputs.into_iter().flatten() {
        cell_hints.push(Some(entry.0));
        outputs.push(entry.1);
    }
    Ok((outputs, cell_hints))
}

/// One writer, one packed shard (a group of whole cells): drain pack of the
/// cells' IVF blobs ‖ Parquet+FTS of the cells' primary rows, then splice.
///
/// Parquet row order = IVF primary order (cells ascending, primaries in
/// member order within each cell) so Parquet local `l` and vector file-local
/// `l` carry the same `_id`; boundary stubs stay vector-only postings.
/// Returns `None` for a stub-only shard (no primary rows — the primaries live
/// in their home cells' files; dropping the replica copies loses nothing).
fn build_one_packed_shard_via_drain(
    cells: &[(u32, AssignedCellGroup<'_>)],
    source_scalar: &RecordBatch,
    vector_views: &[VectorColumnView<'_>],
    local_by_id: &HashMap<i128, u32>,
    options: &SupertableOptions,
    vc: &VectorConfig,
) -> Result<Option<ShardOutput>, BuildError> {
    let mut ordered_locals: Vec<u32> = Vec::new();
    for (_, group) in cells {
        for (member_id, is_primary, _) in &group.members {
            if !*is_primary {
                continue;
            }
            let local = local_by_id.get(member_id).copied().ok_or_else(|| {
                BuildError::Store(format!(
                    "primary stable_id {member_id} missing from commit rows"
                ))
            })?;
            ordered_locals.push(local);
        }
    }
    if ordered_locals.is_empty() {
        return Ok(None);
    }

    // Drain packs this shard's cell IVFs; Parquet+FTS build overlaps it.
    let (packed_groups, body_and_fts) = rayon::join(
        || {
            cells
                .iter()
                .map(|(cell_id, group)| {
                    let owned = AssignedCellGroup {
                        cell_id: *cell_id,
                        members: group.members.clone(),
                    };
                    drain_pack_assigned_cell(owned, vc)
                })
                .collect::<Result<Vec<_>, BuildError>>()
        },
        || build_shard_parquet_and_fts(source_scalar, vector_views, &ordered_locals, options),
    );
    let packed_groups = packed_groups?;
    let (mut builder, id_min, id_max, n_docs, scalar_stats) = body_and_fts?;

    let subsections: Vec<(u32, MergedIvfSubsection)> = packed_groups
        .into_iter()
        .map(|g| (g.cell_id, g.subsection))
        .collect();
    builder.set_prebuilt_multi_cell_ivfs(subsections)?;
    let bytes = Bytes::from(builder.finish()?);

    Ok(Some(ShardOutput {
        bytes,
        n_docs,
        id_min,
        id_max,
        scalar_stats,
    }))
}

/// Parquet body + FTS for one shard, rows reordered to `ordered_locals`
/// (primaries in IVF order). MultiCell has no streaming VectorBuilder, so
/// `add_batch` builds scalars + FTS and only validates the vector slices;
/// IVF subsections arrive from drain via `set_prebuilt_multi_cell_ivfs`.
#[allow(clippy::type_complexity)]
fn build_shard_parquet_and_fts(
    source_scalar: &RecordBatch,
    vector_views: &[VectorColumnView<'_>],
    ordered_locals: &[u32],
    options: &SupertableOptions,
) -> Result<
    (
        SuperfileBuilder,
        i128,
        i128,
        u64,
        HashMap<String, ScalarStatsAgg>,
    ),
    BuildError,
> {
    let take_indices = UInt32Array::from(ordered_locals.to_vec());
    let columns: Vec<ArrayRef> = source_scalar
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &take_indices, None))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| BuildError::Store(err.to_string()))?;
    let scalar = RecordBatch::try_new(source_scalar.schema(), columns)
        .map_err(|_| BuildError::BatchSchemaMismatch)?;

    // This shard's rows in IVF order — the one remaining vector copy on
    // the commit path, shard-sized and transient (the commit-wide flatten
    // it replaced held every column for the whole commit).
    let mut ordered_vectors: Vec<Vec<f32>> = Vec::with_capacity(vector_views.len());
    for view in vector_views {
        let mut ordered = Vec::with_capacity(ordered_locals.len() * view.dim);
        for &local in ordered_locals {
            ordered.extend_from_slice(view.row(local as usize)?);
        }
        ordered_vectors.push(ordered);
    }
    let vector_slices: Vec<&[f32]> = ordered_vectors.iter().map(Vec::as_slice).collect();

    let mut builder = SuperfileBuilder::new(
        options
            .builder_options()
            .with_vector_layout(VectorLayout::MultiCellIvf),
    )?;
    builder.add_batch(&scalar, &vector_slices)?;

    let scalar_schema = options.scalar_schema();
    let scalar_stats = ScalarStatsAgg::from_batches(&scalar_schema, &[&scalar]);

    let id_col = scalar
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| {
            BuildError::IdColumnWrongType(
                options.id_column.clone(),
                "<id column not Decimal128 at runtime>".to_string(),
            )
        })?;
    let mut id_min = i128::MAX;
    let mut id_max = i128::MIN;
    for i in 0..id_col.len() {
        let v = id_col.value(i);
        id_min = id_min.min(v);
        id_max = id_max.max(v);
    }
    let n_docs = id_col.len() as u64;
    let (id_min, id_max) = if n_docs == 0 {
        (0, 0)
    } else {
        (id_min, id_max)
    };
    Ok((builder, id_min, id_max, n_docs, scalar_stats))
}

/// Minimum overflow rows required to split a cell into two sub-cells — a split
/// needs at least one row per side, so fewer than this is a no-op.
const MIN_ROWS_TO_SPLIT_CELL: usize = 2;

/// Physical count changes from one committed cell split. The split pass keeps
/// this small delta in memory so it can choose the next overflow without
/// reopening every superfile to rebuild the complete count table.
pub(in crate::supertable) struct CellSplitOutcome {
    new_cell_id: u32,
    retained_docs: u64,
    new_cell_docs: u64,
}

/// Split one over-cap **global cell** into two balanced sub-cells. Extracts the
/// cell's live rows (dropping tombstones) from every superfile that holds it,
/// median-splits them into two centroids, rebuilds both sub-cells (and
/// republishes any other cells that shared a packed shard) as fresh packed
/// superfiles, and atomically swaps the grid `{..,P,..}` → `{..,P,new..}` in one
/// commit. `split_cell` stays live and queryable until the swap lands. The
/// caller ([`split_overflow_cells`]) picks the cell from physical file counts.
///
/// Returns the committed physical count delta. `None` is a defensive no-op
/// result; the caller remembers it for this pass so unchanged physical counts
/// cannot select the same cell repeatedly. User deletes are represented by the
/// hidden resident deleted-id set rather than hidden tombstones, so a
/// delete-heavy user table does not normally reach this branch.
pub(in crate::supertable) async fn split_overflow_cell(
    inner: Arc<SupertableInner>,
    split_cell: u32,
) -> Result<Option<CellSplitOutcome>, BuildError> {
    let manifest = inner.manifest.load_full();
    let (clusters, column, routing, metric, _vec_dim) = match manifest.get_partition_strategy() {
        PartitionStrategy::VectorCell {
            clusters,
            column,
            routing,
        } => {
            let Some(vec_col) = inner.options.vector_columns.first() else {
                return Ok(None);
            };
            (clusters, column, routing, vec_col.metric, vec_col.dim)
        }
        _ => return Ok(None),
    };
    if clusters.n_cent == 0 || clusters.dim == 0 || split_cell >= clusters.n_cent {
        return Ok(None);
    }

    let now = time::Instant::now();
    let storage = inner
        .options
        .storage
        .clone()
        .ok_or_else(|| BuildError::Store("cell split requires storage".into()))?;

    // Shadow split scoped to `split_cell`: extract its live rows, route them
    // into two sub-cells, and swap atomically. No neighbor rebalance (deferred;
    // nprobe >= 2 covers the shifted boundary), no count zeroing, no INCOMING
    // staging. `split_cell` stays live and queryable until the swap commit.
    let neighborhood_slice = [split_cell];

    // Select files that actually contain a neighborhood cell (cell directory
    // for packed; partition_hint == cell_id for legacy).
    let mut to_remove: Vec<Arc<SuperfileEntry>> = Vec::new();
    let mut keep_cells_by_entry: Vec<(Arc<SuperfileEntry>, Vec<u32>)> = Vec::new();
    for entry in manifest.superfiles.iter() {
        if entry.vector_layout == VectorLayout::MultiCellIvf {
            let counts = cell_doc_counts_for_entry(&inner, entry).await?;
            let has_neighborhood = counts
                .iter()
                .any(|(cell, _)| neighborhood_slice.contains(cell));
            if !has_neighborhood {
                continue;
            }
            let keep: Vec<u32> = counts
                .into_iter()
                .map(|(cell, _)| cell)
                .filter(|cell| !neighborhood_slice.contains(cell))
                .collect();
            to_remove.push(Arc::clone(entry));
            if !keep.is_empty() {
                keep_cells_by_entry.push((Arc::clone(entry), keep));
            }
        } else {
            let Some(hint) = entry.partition_hint else {
                continue;
            };
            if neighborhood_slice.contains(&hint) {
                to_remove.push(Arc::clone(entry));
            }
        }
    }

    let mut all_materialized: Vec<MaterializedIvfRow> = Vec::new();
    for entry in &to_remove {
        let mut rows = load_materialized_rows_from_ivf_superfile(
            &inner,
            entry,
            &column,
            now,
            Some(&neighborhood_slice),
        )
        .await?;
        all_materialized.append(&mut rows);
    }
    if all_materialized.len() < MIN_ROWS_TO_SPLIT_CELL {
        return Ok(None);
    }

    // Plan the binary split over exactly the extracted (live) rows: `assign[i]`
    // routes all_materialized[i] to sub-cell 0 (keeps split_cell's id) or 1
    // (new_cell_id). Insert the second sub-centroid into the grid.
    // Borrow the encoded rows into the planner instead of cloning the whole
    // (largest) cell's Sq8+ε payload — a clone here doubled the biggest cell's
    // resident bytes at split time (a RAM cliff at 100M/1B).
    let split_refs: Vec<&EncodedCellRow> = all_materialized.iter().map(|r| &r.encoded).collect();
    let (sub0, sub1, assign) =
        maint_pool()?.install(|| opann::plan_sq8_split(&split_refs, &clusters, split_cell, metric));
    let mut sub_centroids = sub0;
    sub_centroids.extend_from_slice(&sub1);
    let (updated_clusters, new_cell_id) =
        opann::insert_split_centroid(&clusters, split_cell, &sub_centroids);

    // Republish non-neighborhood cells that shared a packed shard so they are
    // not deleted with the neighborhood extract. Use the tombstone-aware
    // loader so deleted locals are not resurrected.
    let mut prepared_keep: Vec<PreparedSuperfile> = Vec::new();
    for (entry, keep_ids) in &keep_cells_by_entry {
        // One decode of the entry for all kept cells, grouped by cell.
        let groups = load_materialized_rows_by_cell_from_ivf_superfile(
            &inner, entry, &column, now, keep_ids,
        )
        .await?;
        let mut packed: Vec<(u32, MergedIvfSubsection, Vec<i128>)> = Vec::new();
        for (cell_id, mut rows) in groups {
            if rows.is_empty() {
                continue;
            }
            for (i, row) in rows.iter_mut().enumerate() {
                row.local_doc_id = i as u32;
            }
            let stable_ids: Vec<i128> = rows.iter().map(|r| r.stable_id).collect();
            let mut cfg = inner
                .options
                .vector_columns
                .first()
                .cloned()
                .ok_or_else(|| BuildError::Store("missing vector column".into()))?;
            let n_cent = rows
                .iter()
                .map(|r| r.cluster as usize + 1)
                .max()
                .unwrap_or(1)
                .max(1);
            cfg.n_cent = n_cent;
            let sub = build_merged_subsection_from_materialized(cfg, rows)?;
            packed.push((cell_id, sub, stable_ids));
        }
        if packed.is_empty() {
            continue;
        }
        let shard_id = entry.partition_hint.unwrap_or_else(|| {
            packed_cell_shard(packed[0].0, packed_cell_shard_count(&inner.options)) as u32
        });
        prepared_keep.push(build_prepared_from_packed_cells(&inner, shard_id, packed)?);
    }

    // Route the extracted rows into the two sub-cells and build each as a packed
    // cell with the same builder the republish above uses (no new packing).
    // Rows keep their inherited L2 fine-cluster ordinal (a per-sub-cell
    // re-cluster is a later refinement).
    let mut group0: Vec<MaterializedIvfRow> = Vec::new();
    let mut group1: Vec<MaterializedIvfRow> = Vec::new();
    for (row, &side) in all_materialized.into_iter().zip(assign.iter()) {
        if side == 0 {
            group0.push(row);
        } else {
            group1.push(row);
        }
    }
    let n0 = group0.len() as u32;
    let n1 = group1.len() as u32;
    let build_subcell = |cell_id: u32,
                         mut rows: Vec<MaterializedIvfRow>|
     -> Result<Option<PreparedSuperfile>, BuildError> {
        if rows.is_empty() {
            return Ok(None);
        }
        for (i, row) in rows.iter_mut().enumerate() {
            row.local_doc_id = i as u32;
        }
        let stable_ids: Vec<i128> = rows.iter().map(|r| r.stable_id).collect();
        let mut cfg = inner
            .options
            .vector_columns
            .first()
            .cloned()
            .ok_or_else(|| BuildError::Store("missing vector column".into()))?;
        cfg.n_cent = rows
            .iter()
            .map(|r| r.cluster as usize + 1)
            .max()
            .unwrap_or(1)
            .max(1);
        let sub = build_merged_subsection_from_materialized(cfg, rows)?;
        let shard_id = packed_cell_shard(cell_id, packed_cell_shard_count(&inner.options)) as u32;
        build_prepared_from_packed_cells(&inner, shard_id, vec![(cell_id, sub, stable_ids)])
            .map(Some)
    };
    let mut all_prepared = prepared_keep;
    all_prepared.extend(build_subcell(split_cell, group0)?);
    all_prepared.extend(build_subcell(new_cell_id, group1)?);
    if all_prepared.is_empty() {
        return Ok(None);
    }

    // Set the two sub-cell counts from the routing; every other cell unchanged.
    let updated_clusters = opann::apply_cell_count_updates(
        &updated_clusters,
        &std::collections::HashMap::from([(split_cell, n0), (new_cell_id, n1)]),
    );

    let batch = collect_prepared_superfiles(&inner, all_prepared)?;

    // Publish the new grid in the same OCC attempt as the replacement
    // membership. Pre-storing the strategy is not atomic with the CAS and
    // is lost on contention refresh. `drained_ranges` and every other
    // manifest field ride through `update` unchanged — a hidden-space reorg
    // consumes no user commit and must not disturb coverage.
    let list_metadata = CommitListMetadata {
        partition_strategy: Some(PartitionStrategy::VectorCell {
            column: column.clone(),
            clusters: updated_clusters.clone(),
            routing,
        }),
        drained_ranges: None,
        global_vector_index: None,
    };

    let new_manifest = persist_commit_async(
        &inner,
        Arc::clone(&storage),
        batch.new_entries,
        &to_remove,
        batch.pending_storage_writes,
        Vec::new(),
        list_metadata,
    )
    .await
    .map_err(|e| BuildError::Store(e.to_string()))?;
    inner.manifest.store(Arc::new(new_manifest));
    apply_pending_store_inserts(&inner, batch.pending_store_inserts);

    schedule_background_storage_reclaim(Arc::clone(&inner));

    Ok(Some(CellSplitOutcome {
        new_cell_id,
        retained_docs: u64::from(n0),
        new_cell_docs: u64::from(n1),
    }))
}

/// Split-then-merge phase 1: repeatedly split the largest over-cap global cell
/// until every cell is within `cell_split_doc_cap`. Eligibility is read from the
/// live grid counts (not a just-merged shard), which keeps the split its own
/// snapshot-consistent phase — it never removes a superfile a later merge job
/// planned to use — and lets an over-cap cell converge within one `optimize`
/// rather than one split per pass. Each split commits atomically, so a mid-loop
/// failure leaves a valid, partially-split grid that the next `optimize`
/// finishes. Splitting first also avoids merging a cell that is about to be
/// re-split (the merge output would be discarded immediately).
pub(in crate::supertable) async fn split_overflow_cells(
    inner: Arc<SupertableInner>,
) -> Result<(), BuildError> {
    // Safety bound only: a balanced (median) cut halves a cell each split, so a
    // cell converges in ~log2(size / cap) splits — far below this. It just stops
    // a pathological non-shrinking split from looping forever.
    const MAX_SPLITS_PER_OPTIMIZE: usize = 4096;
    let manifest = inner.manifest.load_full();
    if !matches!(
        manifest.get_partition_strategy(),
        PartitionStrategy::VectorCell { .. }
    ) {
        return Ok(());
    }
    // Compute physical counts once. Each successful split returns the two
    // replacement counts, so later iterations update this table in O(1)
    // instead of reopening every superfile for another full recount.
    let mut cell_counts: HashMap<u32, u64> = HashMap::new();
    for entry in manifest.superfiles.iter() {
        for (cell, n) in cell_doc_counts_for_entry(&inner, entry).await? {
            *cell_counts.entry(cell).or_default() += u64::from(n);
        }
    }

    // Defensive progress guard for cells whose split no-op'd this pass.
    // Selection uses physical counts, so without this set any unchanged
    // over-cap cell would be selected repeatedly up to the split bound.
    // Hidden user deletes use the resident deleted-id set, not hidden
    // tombstones; this is not the normal delete-heavy-table path.
    let mut unsplittable: HashSet<u32> = HashSet::new();
    for iteration in 0..MAX_SPLITS_PER_OPTIMIZE {
        let mut best: Option<(u32, u64)> = None;
        for (cell, n) in &cell_counts {
            let n = *n;
            if opann::split_overflow_needed(n)
                && !unsplittable.contains(cell)
                && best.is_none_or(|(_, b)| n > b)
            {
                best = Some((*cell, n));
            }
        }
        let Some((split_cell, n)) = best else {
            return Ok(());
        };
        if (n as usize) < MIN_ROWS_TO_SPLIT_CELL {
            return Ok(());
        }
        match split_overflow_cell(Arc::clone(&inner), split_cell).await? {
            Some(outcome) => {
                cell_counts.insert(split_cell, outcome.retained_docs);
                cell_counts.insert(outcome.new_cell_id, outcome.new_cell_docs);
            }
            None => {
                unsplittable.insert(split_cell);
            }
        }
        if iteration + 1 == MAX_SPLITS_PER_OPTIMIZE {
            tracing::warn!(
                "cell split: hit per-optimize split bound ({MAX_SPLITS_PER_OPTIMIZE}); \
                 over-cap cells remain and will converge on the next optimize"
            );
        }
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
/// in-memory `ManifestSnapshot` with the fresh persisted Manifest +
/// loader installed.
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
/// content from our prior attempt). ManifestSnapshot parts are
/// content-addressed; identical content yields identical URIs
/// and the part-write path already swallows
/// `PreconditionFailed`. Only the pointer PUT must win the
/// CAS; everything below it is idempotent.
///
/// When no real partitioning is configured, all post-commit
/// superfiles go into one `ManifestPart` with a fresh `PartId`.
/// With a real `PartitionStrategy`, `try_commit_attempt` runs
/// the per-partition part-reuse path described on that fn.
/// Publish the slow-CAS vector-state blob for `inner`'s CURRENT membership
/// and stamp its ref on the manifest list. Called after a maintenance
/// sequence settles hidden vector membership (end of drain; end of the
/// hidden compaction pass, after merges + finalize + any cell splits) —
/// scoped by call site, never by a table-kind test. `ManifestSnapshot::update`
/// cleared the ref when membership changed; this restamps it so consumers'
/// resident centroid state is invalidated exactly once, by maintenance.
///
/// Writes the content-addressed blob idempotently (`PreconditionFailed` =
/// already durable), then a list+pointer etag-CAS stamp with refresh-and-retry
/// on contention — so a lost race rebuilds the blob from the winning
/// membership, never stamping stale state.
pub(in crate::supertable) async fn refresh_slow_vector_state(
    inner: &SupertableInner,
) -> Result<(), BuildError> {
    stamp_slow_vector_state(inner, None).await
}

/// The PREVIOUS generation's centroid section for `manifest`, through the
/// table's single-slot cache (fetch on miss, reuse on URI match). `None`
/// when no section is stamped (fresh table) or the fetch fails — the
/// composer then requires every entry's fp32 to be resident and errors
/// loudly otherwise.
async fn previous_centroid_section(
    options: &SupertableOptions,
    storage: &dyn StorageProvider,
    manifest: &ManifestSnapshot,
) -> Option<Arc<CentroidSection>> {
    let reference = manifest.slow_vector_state_centroids_blob()?.clone();
    let slot = Arc::clone(&options.centroid_section_cache);
    let mut guard = slot.lock().await;
    if let Some(section) = guard.as_ref()
        && section.uri() == reference.uri
    {
        return Some(Arc::clone(section));
    }
    match fetch_centroid_section(storage, &reference, manifest.get_all_superfiles()).await {
        Ok(section) => {
            let section = Arc::new(section);
            *guard = Some(Arc::clone(&section));
            Some(section)
        }
        Err(error) => {
            tracing::warn!(
                "previous centroid section {} unavailable ({error}); republish must compose \
                 from resident fp32 only",
                reference.uri
            );
            None
        }
    }
}

async fn stamp_slow_vector_state(
    inner: &SupertableInner,
    pending_drain: Option<slow_vector_state::PendingDrainState>,
) -> Result<(), BuildError> {
    let Some(storage) = inner.options.storage.clone() else {
        return Ok(());
    };
    let max_retries = inner.options.max_commit_retries.max(1);
    for attempt in 0..max_retries {
        let old = inner.manifest.load_full();
        let entries = old.get_all_superfiles();
        if entries.is_empty() && pending_drain.is_none() {
            // Nothing to describe (pre-drain / empty table); the ref is
            // already absent because `update` never carries it forward.
            return Ok(());
        }
        // Carried-forward entries are stripped (routing-shaped hydration);
        // their fp32 composes from the previous generation's section.
        let previous_section =
            previous_centroid_section(&inner.options, storage.as_ref(), &old).await;
        let published = match pending_drain.as_ref() {
            Some(pending) => {
                slow_vector_state::write_state_with_pending_drain(
                    storage.as_ref(),
                    entries,
                    pending,
                    previous_section.as_deref(),
                )
                .await
            }
            None => {
                slow_vector_state::write_state(
                    storage.as_ref(),
                    entries,
                    previous_section.as_deref(),
                )
                .await
            }
        }
        .map_err(|e| BuildError::Store(e.to_string()))?;
        if let Some((cur_uri, cur_hash)) = old.slow_vector_state_blob()
            && cur_uri == published.uri
            && cur_hash == published.content_hash
            && old.slow_vector_state_centroids_blob() == Some(&published.centroids)
        {
            // Same membership already stamped — republish is a no-op.
            return Ok(());
        }
        let new_manifest =
            old.with_slow_vector_state(published.uri, published.content_hash, published.centroids);
        let prev_etag = get_current_manifest_etag(&storage, Arc::clone(&old))
            .await
            .map_err(|e| BuildError::Store(e.to_string()))?;
        match new_manifest
            .write(storage.as_ref(), prev_etag.as_deref(), &[])
            .await
        {
            Ok(()) => {
                inner.manifest.store(Arc::new(new_manifest));
                return Ok(());
            }
            Err(SupertableCommitError::WriteContentionExhausted) if attempt + 1 < max_retries => {
                refresh_inner_state_async(inner, &storage)
                    .await
                    .map_err(|e| BuildError::Store(e.to_string()))?;
                sleep(backoff_delay(attempt)).await;
            }
            Err(e) => return Err(BuildError::Store(e.to_string())),
        }
    }
    Err(BuildError::Store(
        "slow vector-state refresh: write contention exhausted".into(),
    ))
}

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
    let max_retries = inner.options.max_commit_retries.max(1);
    for attempt in 0..max_retries {
        let old = inner.manifest.load_full();
        let mut ids = hidden_deleted::deleted_user_ids(&old)
            .map_err(|e| BuildError::Store(e.to_string()))?
            .as_ref()
            .clone();
        let before = ids.len();
        ids.extend_from_slice(new_deleted);
        ids.sort_unstable();
        ids.dedup();
        if ids.len() == before {
            return Ok(());
        }
        let bytes = encode_deleted_ids(&ids);
        let new_manifest = old.with_deleted_user_ids(bytes);
        let prev_etag = get_current_manifest_etag(&storage, Arc::clone(&old))
            .await
            .map_err(|e| BuildError::Store(e.to_string()))?;
        match new_manifest
            .write(storage.as_ref(), prev_etag.as_deref(), &[])
            .await
        {
            Ok(()) => {
                inner.manifest.store(Arc::new(new_manifest));
                return Ok(());
            }
            Err(SupertableCommitError::WriteContentionExhausted) if attempt + 1 < max_retries => {
                refresh_inner_state_async(inner, &storage)
                    .await
                    .map_err(|e| BuildError::Store(e.to_string()))?;
                sleep(backoff_delay(attempt)).await;
            }
            Err(e) => return Err(BuildError::Store(e.to_string())),
        }
    }
    Err(BuildError::Store(
        "deleted-set record: write contention exhausted".into(),
    ))
}

/// List-level metadata stamped onto the OCC base snapshot for one durable
/// commit attempt. Applied inside every retry so contention refresh cannot
/// drop grid / watermark / bootstrap stamps that must land with membership.
#[derive(Debug, Default, Clone)]
pub(crate) struct CommitListMetadata {
    pub(crate) partition_strategy: Option<PartitionStrategy>,
    pub(crate) global_vector_index: Option<GlobalVectorIndex>,
    pub(crate) drained_ranges: Option<DrainedVersionRanges>,
}

impl CommitListMetadata {
    pub(crate) fn empty() -> Self {
        Self::default()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.partition_strategy.is_none()
            && self.global_vector_index.is_none()
            && self.drained_ranges.is_none()
    }

    /// Overlay stamped fields onto `base`. `ManifestSnapshot` is not
    /// `Clone`; start from an identity stamp (`with_drained_ranges` of the
    /// current ranges) then layer call-site fields.
    pub(crate) fn apply(&self, base: &ManifestSnapshot) -> ManifestSnapshot {
        let mut out = base.with_drained_ranges(base.get_drained_ranges());
        if let Some(strategy) = self.partition_strategy.clone() {
            out = out.with_partition_strategy(strategy);
        }
        if let Some(index) = self.global_vector_index.clone() {
            out = out.with_global_vector_index(index);
        }
        if let Some(ranges) = self.drained_ranges.clone() {
            out = out.with_drained_ranges(ranges);
        }
        out
    }
}

pub(in crate::supertable) async fn persist_commit_async(
    inner: &SupertableInner,
    storage: Arc<dyn StorageProvider>,
    new_entries: Vec<Arc<SuperfileEntry>>,
    entries_to_remove: &[Arc<SuperfileEntry>],
    mut pending_storage_writes: Vec<(SuperfileUri, Bytes)>,
    mut pending_storage_replaces: Vec<(SuperfileUri, Bytes)>,
    list_metadata: CommitListMetadata,
) -> Result<ManifestSnapshot, SupertableCommitError> {
    let storage_async = Arc::clone(&storage);
    let opts = Arc::clone(&inner.options);
    let max_retries = opts.max_commit_retries.max(1);
    let drive = async move {
        let mut last_err: Option<SupertableCommitError> = None;
        for attempt in 0..max_retries {
            let old = inner.manifest.load_full();
            // Re-apply call-site stamps on every attempt. A pre-store of these
            // fields is not OCC-safe: contention refresh reloads from storage
            // and would drop them before a successful CAS.
            let base = if list_metadata.is_empty() {
                old
            } else {
                Arc::new(list_metadata.apply(&old))
            };
            let pending_writes = &mut pending_storage_writes;
            let pending_replaces = &mut pending_storage_replaces;
            match try_commit_attempt(
                Arc::clone(&storage_async),
                Arc::clone(&opts),
                base,
                &new_entries,
                entries_to_remove,
                NewEntryBirthVersions::StampCommit,
                pending_writes,
                pending_replaces,
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
    list_metadata: CommitListMetadata,
) -> Result<(), SupertableCommitError> {
    let drive = persist_commit_async(
        inner,
        storage,
        new_entries,
        entries_to_remove,
        pending_storage_writes,
        pending_storage_replaces,
        list_metadata,
    );
    let new_manifest = bridge_on_runtime(drive, &inner.query_runtime())?;
    inner.manifest.store(Arc::new(new_manifest));
    inner.reconcile_tombstone_seqs();
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

/// Upper bound on the drain's auto-sized read fan-out — keeps a very large box
/// from stampeding a single S3 prefix. An explicit env override is not clamped.
const DRAIN_READ_CONCURRENCY_CAP: usize = 64;

/// Read fan-out for the drain's superfile opens — bulk S3 reads off the
/// query-critical path. Ideal sizing tracks network bandwidth; vCPU count is the
/// portable runtime proxy for it (a cloud instance's NIC scales with its size).
/// The auto default is one in-flight read per hardware thread, floored at the
/// read layer's background-fill default (`prefetch_concurrency`) so small boxes
/// still fan out, and capped at [`DRAIN_READ_CONCURRENCY_CAP`]. Sourced from
/// `vector.drain_read_concurrency`; an explicit integer there is used verbatim
/// (unclamped), while `auto` applies the vCPU-derived default.
fn drain_read_concurrency() -> usize {
    if let ThreadCount::Fixed(n) = config::global().vector.drain_read_concurrency
        && n > 0
    {
        return n;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(
            crate::config::DEFAULT_PREFETCH_CONCURRENCY,
            DRAIN_READ_CONCURRENCY_CAP,
        )
}

#[cfg_attr(
    feature = "detailed-tracing",
    tracing::instrument(skip_all, fields(superfiles = pending_storage_writes.len()))
)]
pub async fn write_superfile_list(
    storage: &Arc<dyn StorageProvider>,
    opts: &Arc<SupertableOptions>,
    pending_storage_writes: &mut Vec<(SuperfileUri, Bytes)>,
    pending_storage_replaces: &mut Vec<(SuperfileUri, Bytes)>,
) -> Result<(), SupertableCommitError> {
    write_superfile_list_with_threshold(
        storage,
        opts,
        opts.put_multipart_threshold_bytes,
        pending_storage_writes,
        pending_storage_replaces,
    )
    .await
}

async fn put_new_superfile_bytes(
    storage: &Arc<dyn StorageProvider>,
    multipart_threshold: u64,
    uri: SuperfileUri,
    bytes: Bytes,
) -> Result<(), SupertableCommitError> {
    let path = superfile_storage_path(&uri);
    let result = if (bytes.len() as u64) >= multipart_threshold {
        put_superfile_multipart(storage.as_ref(), &path, bytes).await
    } else {
        storage.put_atomic(&path, bytes).await.map(|_| ())
    };
    match result {
        Ok(()) | Err(StorageError::PreconditionFailed { .. }) => Ok(()),
        Err(error) => Err(SupertableCommitError::from(error)),
    }
}

async fn write_superfile_list_with_threshold(
    storage: &Arc<dyn StorageProvider>,
    _opts: &Arc<SupertableOptions>,
    put_multipart_threshold_bytes: u64,
    pending_storage_writes: &mut Vec<(SuperfileUri, Bytes)>,
    pending_storage_replaces: &mut Vec<(SuperfileUri, Bytes)>,
) -> Result<(), SupertableCommitError> {
    // Bound object-store fanout to half the machine's CPU parallelism. A vector
    // commit can stage one hidden delta per touched cell plus user shards;
    // driving all PUTs at once opens dozens of sockets and can stall the commit
    // path. Crucially, bulk ingest commits overlap background hidden-index
    // OPANN maintenance (its own compaction PUT/GET waves), so a full-width
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

    let multipart_threshold = put_multipart_threshold_bytes;
    let put_futs = pending_storage_writes
        .iter()
        .enumerate()
        .map(|(i, (uri, bytes))| {
            let storage = Arc::clone(storage);
            let uri = *uri;
            let bytes = bytes.clone();
            async move {
                put_new_superfile_bytes(&storage, multipart_threshold, uri, bytes)
                    .await
                    .map(|()| i)
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
/// `ManifestPartEntry` that replaces the prior one (same
/// `partition_key`, new `part_id` + content hash). Untouched
/// partitions' list entries carry over verbatim — no
/// re-encode, no PUT. A cold partition (no prior entry) gets
/// a fresh part with just the new superfiles. The result: a
/// single-partition commit rewrites exactly one part
/// regardless of how many other partitions exist — the
/// load-bearing property the part-reuse optimization relies
/// on.
#[derive(Clone, Copy)]
pub(crate) enum NewEntryBirthVersions {
    /// User append/update data is born in the manifest commit publishing it.
    StampCommit,
    /// Compaction changes physical residency but preserves logical lineage.
    Preserve,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn try_commit_attempt(
    storage: Arc<dyn StorageProvider>,
    opts: Arc<SupertableOptions>,
    current_manifest: Arc<ManifestSnapshot>,
    new_entries: &[Arc<SuperfileEntry>],
    entries_to_remove: &[Arc<SuperfileEntry>],
    birth_versions: NewEntryBirthVersions,
    pending_storage_writes: &mut Vec<(SuperfileUri, Bytes)>,
    pending_storage_replaces: &mut Vec<(SuperfileUri, Bytes)>,
) -> Result<ManifestSnapshot, SupertableCommitError> {
    // 1. Write each new superfile's bytes to storage in parallel.
    write_superfile_list(
        &storage,
        &opts,
        pending_storage_writes,
        pending_storage_replaces,
    )
    .await?;

    // 2. update the manifest for the commit.
    let (mut new_manifest, parts_to_write) = match birth_versions {
        NewEntryBirthVersions::StampCommit => {
            current_manifest
                .update(new_entries, entries_to_remove)
                .await?
        }
        NewEntryBirthVersions::Preserve => {
            current_manifest
                .update_preserving_birth_versions(new_entries, entries_to_remove)
                .await?
        }
    };

    // 2b. Hidden VectorCell membership lives in the slow-state blob.
    //     `update` clears the ref; restamp it onto this same successor
    //     before the list/pointer CAS so a crash cannot leave durable
    //     membership with a missing slow-state ref (S17).
    if super::handle::is_hidden_vector_index_table(&opts) {
        let entries = new_manifest.get_all_superfiles();
        if !entries.is_empty() {
            // Carried-forward entries are stripped; the PREVIOUS manifest
            // still holds the section ref `update` cleared — compose the
            // new generation's section from it plus this commit's fresh
            // (fp32-resident) entries.
            let previous_section =
                previous_centroid_section(&opts, storage.as_ref(), current_manifest.as_ref()).await;
            let published = slow_vector_state::write_state(
                storage.as_ref(),
                entries,
                previous_section.as_deref(),
            )
            .await
            .map_err(|e| {
                SupertableCommitError::ManifestError(ManifestError::ManifestLoadError(
                    ManifestLoadError::SlowStateHydration(e.to_string()),
                ))
            })?;
            new_manifest = new_manifest.with_slow_vector_state_ref(
                published.uri,
                published.content_hash,
                published.centroids,
            );
        }
    }

    // 3. Read the prior pointer's etag for the CAS. Fresh
    //    supertable → no pointer yet → None etag (initial
    //    commit).
    let prev_etag = get_current_manifest_etag(&storage, current_manifest).await?;

    // 4. Parallel-issue (touched parts) + list PUTs, then
    //    conditional pointer PUT (the visibility barrier).
    //    Untouched parts are NOT re-PUT — their URIs (and
    //    content-hashes) are unchanged in the new list. Each touched
    //    part ships both wire forms: full and the routing sibling the
    //    list entry references.
    let encoded_refs: Vec<&[u8]> = parts_to_write
        .iter()
        .flat_map(|ep| [Some(ep.encoded.as_slice()), ep.routing_encoded.as_deref()])
        .flatten()
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
/// in-memory `ManifestSnapshot` via content-addressed `Arc::clone`,
/// eager-fetch newly-referenced parts, and `ArcSwap` the
/// refreshed `ManifestSnapshot` into `inner.manifest`.
///
/// Called from the OCC retry loop between attempts so the next
/// iteration's `inner.manifest.load_full()` sees the winning
/// writer's state — `with_appended` then chains our pending
/// superfiles onto theirs at the new monotonic `manifest_id`.
///
/// Mirrors the logic in [`Supertable::refresh`] but operates
/// on `&SupertableInner` so it can be called from inside the
/// writer's commit path without holding a `Supertable` handle.
pub(in crate::supertable) async fn refresh_inner_state_async(
    inner: &SupertableInner,
    storage: &Arc<dyn StorageProvider>,
) -> Result<(), SupertableCommitError> {
    let current = inner.manifest.load_full();
    let manifest = match ManifestSnapshot::load(Some(current), storage.clone(), None).await {
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
    inner.reconcile_tombstone_seqs();
    Ok(())
}

/// CAS-publish a successor manifest whose tombstone seq for every
/// superfile in `touched` is bumped to the successor's `manifest_id`.
///
/// This is the mutation pipeline's post-sidecar stamp: it runs after
/// the tombstone phase's sidecar CAS-PUTs and *before* the WAL flips
/// to `Complete`, so a crash in between is completed by the recovery
/// sweep and "WAL complete ⇒ manifest stamped" holds. Readers on
/// other processes pick the bump up on their next manifest refresh
/// and refetch exactly the named sidecars — this is what bounds
/// cross-process delete visibility by the read-consistency window.
///
/// No superfile entries or parts change, so each attempt writes only
/// the list + pointer. OCC discipline matches [`persist_commit`]:
/// reload on contention, jittered backoff, bounded by
/// `max_commit_retries`.
pub(in crate::supertable) async fn stamp_tombstone_seqs(
    inner: &SupertableInner,
    touched: &[Uuid],
) -> Result<(), SupertableCommitError> {
    let Some(storage) = inner.options.storage.clone() else {
        return Ok(());
    };
    let max_retries = inner.options.max_commit_retries.max(1);
    for attempt in 0..max_retries {
        let old = inner.manifest.load_full();
        let Some(new_manifest) = old.with_tombstone_seqs_bumped(touched) else {
            // No persisted list ⇒ in-process-only ⇒ nothing to stamp.
            return Ok(());
        };
        let prev_etag = match get_current_manifest_etag(&storage, Arc::clone(&old)).await {
            Ok(etag) => etag,
            // Pointer moved past our snapshot — reload and retry.
            Err(SupertableCommitError::WriteContentionExhausted) if attempt + 1 < max_retries => {
                refresh_inner_state_async(inner, &storage).await?;
                sleep(backoff_delay(attempt)).await;
                continue;
            }
            Err(e) => return Err(e),
        };
        match new_manifest
            .write(storage.as_ref(), prev_etag.as_deref(), &[])
            .await
        {
            Ok(()) => {
                inner.manifest.store(Arc::new(new_manifest));
                inner.reconcile_tombstone_seqs();
                return Ok(());
            }
            Err(SupertableCommitError::WriteContentionExhausted) if attempt + 1 < max_retries => {
                refresh_inner_state_async(inner, &storage).await?;
                sleep(backoff_delay(attempt)).await;
            }
            Err(e) => return Err(e),
        }
    }
    Err(SupertableCommitError::WriteContentionExhausted)
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
/// 16-MiB chunk reads on the way back out. Parts are pushed in declaration
/// order and driven in bounded concurrent groups so mmap-backed shards remain
/// memory-bounded during upload.
/// Write `bytes` to `path`, routing through multipart (staged blocks) at or
/// above `multipart_threshold` and a single `put_atomic` below it. A single
/// `Put Blob`/`PutObject` is capped at ~5 GiB by Azure/S3, so any blob that can
/// grow past that (e.g. the slow-vector-state centroid blob at 100M+ docs) must
/// take the multipart path. Content-addressed callers treat `PreconditionFailed`
/// as "identical bytes already durable" and swallow it.
pub(in crate::supertable) async fn put_bytes_multipart_or_atomic(
    storage: &dyn StorageProvider,
    path: &str,
    bytes: Bytes,
    multipart_threshold: u64,
) -> Result<(), StorageError> {
    if (bytes.len() as u64) >= multipart_threshold {
        put_superfile_multipart(storage, path, bytes).await
    } else {
        storage.put_atomic(path, bytes).await.map(|_| ())
    }
}

async fn put_superfile_multipart(
    storage: &dyn StorageProvider,
    path: &str,
    bytes: Bytes,
) -> Result<(), StorageError> {
    // Same-bytes retry skip. Failures other than NotFound
    // propagate so we don't paper over a degraded backend.
    match storage.head(path).await {
        Ok(_) => return Err(StorageError::PreconditionFailed { uri: path.into() }),
        Err(StorageError::NotFound { .. }) => {}
        Err(e) => return Err(e),
    }

    let mut upload = storage.put_multipart(path).await?;
    let total = bytes.len();
    let part_concurrency = commit_write_concurrency().max(1);
    let mut parts: Vec<UploadPart> = Vec::with_capacity(part_concurrency);
    let mut offset = 0;
    while offset < total {
        let end = cmp::min(offset + SUPERFILE_MULTIPART_PART_BYTES, total);
        let chunk = bytes.slice(offset..end);
        parts.push(upload.put_part(PutPayload::from_bytes(chunk)));
        offset = end;
        if parts.len() == part_concurrency {
            flush_superfile_multipart_parts(&mut upload, path, &mut parts).await?;
        }
    }
    flush_superfile_multipart_parts(&mut upload, path, &mut parts).await?;
    if let Err(e) = upload.complete().await {
        let _ = upload.abort().await;
        return Err(StorageError::Permanent {
            uri: path.into(),
            source: Box::new(e),
        });
    }
    Ok(())
}

/// Upload one bounded group of multipart chunks. Keeping only
/// `commit_write_concurrency()` chunks in flight prevents a multi-GB mmap-backed
/// shard from faulting every part into memory at once.
async fn flush_superfile_multipart_parts(
    upload: &mut Box<dyn MultipartUpload>,
    path: &str,
    parts: &mut Vec<UploadPart>,
) -> Result<(), StorageError> {
    if parts.is_empty() {
        return Ok(());
    }
    if let Err(error) = try_join_all(mem::take(parts)).await {
        // Best-effort abort; ignore failure (the upload may already be in a
        // terminal state, or the backend may have lost the upload id).
        let _ = upload.abort().await;
        return Err(StorageError::Permanent {
            uri: path.into(),
            source: Box::new(error),
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
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

    use arrow_array::{
        Array, Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch,
    };
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

    /// Small fixed vector dimension accepted by the vector builder.
    const COMMIT_AS_DRAIN_TEST_DIM: usize = 16;
    /// Small row count that still exercises multiple global cells.
    const COMMIT_AS_DRAIN_TEST_ROWS: usize = 8;
    /// Rotation seed for assignment admit contexts in these tests.
    const COMMIT_AS_DRAIN_TEST_ROT_SEED: u64 = 7;
    /// Boundary test target that permits one extra posting per input row.
    const BOUNDARY_STUB_TARGET_FACTOR: f32 = 2.0;

    /// `SupertableWriter`'s `Debug` impl renders its buffered-batch summary.
    #[test]
    fn supertable_writer_debug_renders() {
        let directory = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(directory.path()).expect("provider"));
        let table = Supertable::create(
            options_title_emb_serial(COMMIT_AS_DRAIN_TEST_DIM, COMMIT_AS_DRAIN_TEST_ROWS)
                .with_storage(storage),
        )
        .expect("create");
        let writer = table.writer().expect("writer");
        let rendered = format!("{writer:?}");
        assert!(
            rendered.contains("SupertableWriter"),
            "debug must render the writer, got {rendered}"
        );
    }

    /// `split_buffer_by_vector_cell` routes each buffered row to its
    /// nearest-centroid shard: rows near e_0 land in cell 0, rows near e_1 in
    /// cell 1, and empty cells are dropped.
    #[test]
    fn split_buffer_by_vector_cell_routes_rows_to_nearest_cell() {
        use std::collections::HashMap;

        use arrow_array::{Float32Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};

        let dim = 4usize;
        // Two centroids: e_0 and e_1.
        let centroids = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let cells = ClusterCentroids::from_fp32(2, dim as u32, &centroids, vec![1u32; 2]);

        // Four rows: 0,1 point at e_0; 2,3 point at e_1.
        let scalar = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("t", DataType::Utf8, false)])),
            vec![Arc::new(StringArray::from(vec!["a", "b", "c", "d"]))],
        )
        .expect("scalar batch");
        let vectors = Float32Array::from(vec![
            0.9, 0.1, 0.0, 0.0, // near e_0
            1.0, 0.0, 0.0, 0.0, // e_0
            0.0, 0.9, 0.1, 0.0, // near e_1
            0.0, 1.0, 0.0, 0.0, // e_1
        ]);
        let batch = BufferedBatch {
            scalar,
            vectors: vec![Arc::new(vectors)],
        };

        let out = split_buffer_by_vector_cell(vec![batch], &cells, Metric::Cosine, 0)
            .expect("split buffer by vector cell");
        let mut rows_by_cell: HashMap<u32, usize> = HashMap::new();
        for (cell, batches) in &out {
            rows_by_cell.insert(*cell, batches.iter().map(|b| b.scalar.num_rows()).sum());
        }
        assert_eq!(
            rows_by_cell.get(&0).copied(),
            Some(2),
            "two rows must route to the e_0 cell"
        );
        assert_eq!(
            rows_by_cell.get(&1).copied(),
            Some(2),
            "two rows must route to the e_1 cell"
        );
    }

    #[test]
    fn drain_local_checkpoint_round_trips_and_rejects_other_epoch() {
        let directory = TempDir::new().expect("tempdir");
        let mut checkpoint = DrainLocalCheckpoint::new("epoch-a".into());
        checkpoint.batches_done = 2;
        checkpoint.spills.insert(
            7,
            DrainLocalSpill {
                n_rows: 11,
                n_quants: 3,
                dim: 16,
                rabitq_len: 2,
                rerank_codec_id: RerankCodec::Sq8FixedResidual.codec_id(),
            },
        );
        checkpoint.built_cells.insert(
            2,
            DrainLocalCell {
                n_docs: 9,
                subsection_len: 1_024,
                rerank_codec_id: RerankCodec::Sq8FixedResidual.codec_id(),
            },
        );
        checkpoint.added_per_cell.insert(2, 9);
        checkpoint.added_per_cell.insert(7, 11);
        save_drain_local_checkpoint(directory.path(), &checkpoint).expect("save");

        let loaded = load_drain_local_checkpoint(directory.path(), "epoch-a")
            .expect("load")
            .expect("checkpoint");
        assert_eq!(loaded, checkpoint);
        assert!(
            load_drain_local_checkpoint(directory.path(), "epoch-b").is_err(),
            "an incompatible local epoch must fail loud"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drain_remote_checkpoint_lives_in_slow_cas_state() {
        let directory = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(directory.path()).expect("provider"));
        let table =
            Supertable::create(options_id_title_serial().with_storage(Arc::clone(&storage)))
                .expect("create table");
        let mut writer = table.writer().expect("writer");
        writer
            .append(&build_simple_batch(0, 2))
            .expect("append visible entry");
        writer.commit().expect("commit visible entry");
        drop(writer);
        let pending_entry = Arc::clone(&table.reader().manifest().superfiles[0]);
        let sources = vec![DrainCheckpointSource {
            superfile_id: "source-id".into(),
            uri: "source-uri".into(),
            birth_version: 4,
        }];
        let batch_layout = vec![vec![4]];
        let options_hash = "options".to_string();
        let checkpoint = DrainRemoteCheckpoint {
            schema: DRAIN_CHECKPOINT_SCHEMA,
            epoch_id: drain_epoch_id(
                &options_hash,
                &sources,
                &batch_layout,
                2,
                DrainConsolidate::Kmeans,
            ),
            options_hash,
            sources,
            batch_layout,
            shard_count: 2,
            completed_shards: Vec::new(),
        };
        let mut state = create_drain_remote_checkpoint(table.inner(), checkpoint.clone())
            .await
            .expect("create");
        let loaded = load_drain_remote_checkpoint(table.inner())
            .await
            .expect("load")
            .expect("checkpoint");
        assert_eq!(loaded.checkpoint, checkpoint);

        state.entries.push(Arc::clone(&pending_entry));
        state.checkpoint.completed_shards.push(DrainRemoteShard {
            shard_id: 1,
            superfile_id: pending_entry.superfile_id.to_string(),
            cell_counts: vec![(3, 10)],
        });
        save_drain_remote_checkpoint(table.inner(), &mut state)
            .await
            .expect("CAS update");
        let updated = load_drain_remote_checkpoint(table.inner())
            .await
            .expect("reload")
            .expect("checkpoint");
        assert_eq!(updated.checkpoint.completed_shards.len(), 1);
        assert_eq!(updated.entries.len(), 1);
        assert_eq!(updated.entries[0].superfile_id, pending_entry.superfile_id);

        refresh_slow_vector_state(table.inner())
            .await
            .expect("replace checkpoint with settled slow state");
        assert!(
            load_drain_remote_checkpoint(table.inner())
                .await
                .expect("load settled state")
                .is_none(),
            "settled slow-CAS state must not retain a drain checkpoint"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drain_resumes_from_last_local_batch_checkpoint() {
        let directory = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(directory.path()).expect("provider"));
        let options = options_title_emb_serial(COMMIT_AS_DRAIN_TEST_DIM, COMMIT_AS_DRAIN_TEST_ROWS)
            .with_storage(storage)
            .with_drain_batch_superfiles(1);
        let table = Supertable::create(options).expect("create");
        for _ in 0..2 {
            let mut writer = table.writer().expect("writer");
            writer
                .append(&build_axis_vector_batch(
                    COMMIT_AS_DRAIN_TEST_ROWS,
                    COMMIT_AS_DRAIN_TEST_DIM,
                ))
                .expect("append");
            writer.commit().expect("commit");
        }
        let (hidden, epoch_id) = current_drain_epoch(&table).await;
        inject_drain_test_failure(epoch_id.clone(), DrainTestFailurePhase::AfterBatch, 1);
        let first = drain_user_superfiles_to_hidden_cells(
            Arc::clone(table.inner()),
            Arc::clone(hidden.inner()),
        )
        .await;
        assert!(first.is_err(), "first drain must stop at the failpoint");
        let local = load_drain_local_checkpoint(&drain_scratch_dir(&epoch_id), &epoch_id)
            .expect("load local checkpoint")
            .expect("local checkpoint");
        assert_eq!(local.batches_done, 1);

        drain_user_superfiles_to_hidden_cells(
            Arc::clone(table.inner()),
            Arc::clone(hidden.inner()),
        )
        .await
        .expect("resume drain");
        assert!(
            !drain_scratch_dir(&epoch_id).exists(),
            "successful final CAS removes local checkpoint scratch"
        );
        assert!(
            load_drain_remote_checkpoint(hidden.inner())
                .await
                .expect("load settled slow state")
                .is_none(),
            "settled slow-CAS state contains no pending drain"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drain_preserves_uploaded_shard_across_node_replacement() {
        let directory = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(directory.path()).expect("provider"));
        let options = options_title_emb_serial(COMMIT_AS_DRAIN_TEST_DIM, COMMIT_AS_DRAIN_TEST_ROWS)
            .with_storage(storage)
            .with_writer_pool(writer_pool_with(2))
            .with_drain_batch_superfiles(1);
        let table = Supertable::create(options).expect("create");
        let mut writer = table.writer().expect("writer");
        writer
            .append(&build_axis_vector_batch(
                4 * COMMIT_AS_DRAIN_TEST_ROWS,
                COMMIT_AS_DRAIN_TEST_DIM,
            ))
            .expect("append");
        writer.commit().expect("commit");
        drop(writer);

        let (hidden, epoch_id) = current_drain_epoch(&table).await;
        inject_drain_test_failure(epoch_id.clone(), DrainTestFailurePhase::AfterShard, 1);
        let first = drain_user_superfiles_to_hidden_cells(
            Arc::clone(table.inner()),
            Arc::clone(hidden.inner()),
        )
        .await;
        assert!(first.is_err(), "first drain must stop after one shard");
        let checkpoint = load_drain_remote_checkpoint(hidden.inner())
            .await
            .expect("load pending slow state")
            .expect("pending drain");
        assert_eq!(checkpoint.checkpoint.completed_shards.len(), 1);
        let preserved_id = checkpoint.entries[0].superfile_id;
        let preserved_path = checkpoint.entries[0].uri.storage_path();
        hidden
            .gc_async(Duration::ZERO)
            .await
            .expect("GC with active checkpoint");
        hidden
            .options()
            .storage
            .as_ref()
            .expect("hidden storage")
            .head(&preserved_path)
            .await
            .expect("checkpointed shard remains live through GC");

        // Simulate replacement on a node without the local spill/cell files.
        fs::remove_dir_all(drain_scratch_dir(&epoch_id)).expect("drop local scratch");
        drain_user_superfiles_to_hidden_cells(
            Arc::clone(table.inner()),
            Arc::clone(hidden.inner()),
        )
        .await
        .expect("replacement-node resume");
        assert!(
            hidden
                .reader()
                .manifest()
                .superfiles
                .iter()
                .any(|entry| entry.superfile_id == preserved_id),
            "final manifest must reuse the shard recorded in slow-CAS"
        );
        assert!(
            load_drain_remote_checkpoint(hidden.inner())
                .await
                .expect("load settled state")
                .is_none()
        );
    }

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
                positions: false,
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

    /// Splice-mode multi-batch drain where both single-superfile batches route
    /// the same directions into the same cells: batch 1 spills each cell, batch
    /// 2 concatenates onto it via the fragment-merge path
    /// (`spill_packed_cell` → `merge_fragment_subsections` → reload). Every
    /// ingested doc must survive into the hidden cell index.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn splice_drain_concatenates_same_cell_across_batches() {
        use crate::superfile::reader::VectorSearchOptions;

        let directory = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(directory.path()).expect("provider"));
        let options = options_title_emb_serial(COMMIT_AS_DRAIN_TEST_DIM, COMMIT_AS_DRAIN_TEST_ROWS)
            .with_storage(storage)
            .with_drain_consolidate(DrainConsolidate::Splice)
            .with_drain_batch_superfiles(1);
        let table = Supertable::create(options).expect("create");
        for _ in 0..2 {
            let mut writer = table.writer().expect("writer");
            writer
                .append(&build_axis_vector_batch(
                    COMMIT_AS_DRAIN_TEST_ROWS,
                    COMMIT_AS_DRAIN_TEST_DIM,
                ))
                .expect("append");
            writer.commit().expect("commit");
        }
        let (hidden, _epoch) = current_drain_epoch(&table).await;
        drain_user_superfiles_to_hidden_cells(
            Arc::clone(table.inner()),
            Arc::clone(hidden.inner()),
        )
        .await
        .expect("splice drain across batches");
        assert!(
            hidden.reader().n_superfiles() > 0,
            "splice drain must populate the hidden cell index"
        );
        // The e_0 direction (one doc per batch) still resolves through the
        // concatenated cell.
        let mut q = vec![0.0f32; COMMIT_AS_DRAIN_TEST_DIM];
        q[0] = 1.0;
        let hits = table
            .reader()
            .vector_hits(
                "emb",
                &q,
                COMMIT_AS_DRAIN_TEST_ROWS * 2,
                VectorSearchOptions::new().with_nprobe(32),
                None,
            )
            .expect("search");
        assert!(
            !hits.is_empty(),
            "docs survive the cross-batch splice concatenate"
        );
    }

    /// Row count for the open-footprint regression fixture: large enough
    /// that row-proportional staging (stable ids at 16 B/row + norms at
    /// 4 B/row ≈ 100 KB here) is unmistakable against the v1-discipline
    /// footprint (headers + cluster index, a few KB).
    const OPEN_RANGES_FIXTURE_ROWS: usize = 5_000;
    /// Ceiling on the staged vector open bytes for the fixture — generous
    /// against headers + cluster index, far below any per-row region.
    const OPEN_RANGES_FIXTURE_CEILING_BYTES: u64 = 16 * 1024;

    /// Multi-cell superfiles stage only sub-headers + cluster indexes in
    /// their open ranges (the v1 discipline): the open footprint must not
    /// scale with row count. Staging the full open-time region embedded
    /// per-row stable ids / norms / Sq8 meta into manifest open blobs and
    /// the cold-open hint fetch — measured 3.62 GiB of hidden-data open
    /// fetch and 3.28 GiB of manifest parts at 100M.
    #[test]
    fn multi_cell_open_ranges_exclude_row_proportional_regions() {
        let dim = COMMIT_AS_DRAIN_TEST_DIM;
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(
            options_title_emb_serial(dim, COMMIT_AS_DRAIN_TEST_ROWS).with_storage(storage),
        )
        .expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_axis_vector_batch(OPEN_RANGES_FIXTURE_ROWS, dim))
            .expect("append");
        w.commit().expect("commit");
        drop(w);

        let mut checked = 0usize;
        for entry in walkdir(dir.path()) {
            let bytes = Bytes::from(fs::read(&entry).expect("read superfile"));
            let Some(offsets) = build_subsection_offsets(&bytes) else {
                continue;
            };
            if offsets.vec_open_ranges.is_empty() {
                continue;
            }
            let staged: u64 = offsets.vec_open_ranges.iter().map(|&(_, len)| len).sum();
            assert!(
                staged <= OPEN_RANGES_FIXTURE_CEILING_BYTES,
                "{entry:?}: staged vector open bytes {staged} scale with rows \
                 (ceiling {OPEN_RANGES_FIXTURE_CEILING_BYTES})"
            );
            checked += 1;
        }
        assert!(checked > 0, "fixture must produce vector superfiles");
    }

    /// Recursively collect the `.sf.parquet` superfiles under a temp root.
    fn walkdir(root: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.to_string_lossy().ends_with(".sf.parquet") {
                    out.push(path);
                }
            }
        }
        out
    }

    fn options_title_emb_serial(dim: usize, n_cent: usize) -> SupertableOptions {
        SupertableOptions::new(
            schema_id_title_emb(dim),
            vec![],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent,
                rot_seed: 7,
                metric: Metric::L2Sq,
                rerank_codec: RerankCodec::Fp32,
                provided_centroids: None,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(writer_pool_with(1))
    }

    fn build_axis_vector_batch(n: usize, dim: usize) -> RecordBatch {
        let titles =
            LargeStringArray::from((0..n).map(|i| format!("doc {i} beta")).collect::<Vec<_>>());
        let mut flat = Vec::with_capacity(n * dim);
        for row in 0..n {
            for d in 0..dim {
                flat.push(if d == row % dim { 1.0 } else { 0.0 });
            }
        }
        let values = Arc::new(Float32Array::from(flat));
        let list = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            values,
            None,
        )
        .expect("fixed-size list");
        RecordBatch::try_new(
            schema_id_title_emb(dim),
            vec![Arc::new(titles), Arc::new(list)],
        )
        .expect("vector batch")
    }

    async fn current_drain_epoch(table: &Supertable) -> (Arc<Supertable>, String) {
        let hidden = table
            .inner()
            .vector_index_table
            .as_ref()
            .expect("hidden table")
            .clone();
        let user_manifest = table.inner().manifest.load_full();
        let drained = hidden.inner().manifest.load_full().get_drained_ranges();
        let mut sources: Vec<Arc<SuperfileEntry>> = user_manifest
            .get_all_superfiles_loaded()
            .await
            .expect("load user sources")
            .into_iter()
            .filter(|entry| !drained.contains(entry.birth_version))
            .collect();
        sources.sort_unstable_by(|left, right| {
            left.birth_version
                .cmp(&right.birth_version)
                .then_with(|| left.superfile_id.cmp(&right.superfile_id))
        });
        let batch_cfg = drain_batch_superfiles(&table.inner().options);
        let budget = if batch_cfg < 0 {
            usize::MAX
        } else {
            (batch_cfg as usize).max(1)
        };
        let source_refs: Vec<DrainCheckpointSource> = sources
            .iter()
            .map(|entry| drain_checkpoint_source(entry))
            .collect();
        let batches = make_drain_batches(sources, budget);
        let batch_layout = drain_batch_layout(&batches);
        let strategy = user_manifest.get_partition_strategy();
        let options_hash =
            options_hash::compute_options_hash(table.inner().options.as_ref(), &strategy).to_hex();
        let shard_count = packed_cell_shard_count(&hidden.inner().options);
        (
            hidden,
            drain_epoch_id(
                &options_hash,
                &source_refs,
                &batch_layout,
                shard_count,
                table.inner().options.drain_consolidate,
            ),
        )
    }

    fn committed_reader(st: &Supertable) -> (Arc<SuperfileEntry>, Arc<SuperfileReader>) {
        let entry = Arc::clone(&st.reader().manifest().superfiles[0]);
        let reader = st
            .options()
            .store
            .reader(&entry.uri)
            .expect("committed superfile reader");
        (entry, reader)
    }

    // ---- ingest memory budget ----------------------------------------

    #[test]
    fn reserve_build_scratch_weights_each_input_class() {
        // Pins the per-class estimate against the constants: a measured budget
        // never denies, so `used()` is exactly the reserved amount. The point of
        // the split (vs one blanket factor) is that a byte of vector payload
        // reserves more than a byte of scalar, and the FTS term is additive on
        // top of the scalar hold, not a replacement.
        let budget = ConnectionMemoryBudget::measured();
        let (scalar, vector, fts) = (1000usize, 2000usize, 400usize);

        let guard = reserve_build_scratch(&budget, scalar, vector, fts)
            .expect("measured budget never denies");
        let expected =
            (BUILD_SCALAR_NUM * scalar + BUILD_VECTOR_NUM * vector + BUILD_FTS_NUM * fts)
                / BUILD_SCRATCH_DENOM;
        assert_eq!(budget.used(), expected);
        drop(guard);
        assert_eq!(budget.used(), 0, "reservation released on drop");

        // Same byte count, different class: vector costs strictly more than
        // scalar, and adding FTS text raises the reserve further. Read `used()`
        // while the guard is alive, then release it before the next call.
        let reserved = |s, v, f| {
            let _guard = reserve_build_scratch(&budget, s, v, f).expect("measured");
            budget.used()
        };
        let scalar_only = reserved(1000, 0, 0);
        let vector_only = reserved(0, 1000, 0);
        let scalar_plus_fts = reserved(1000, 0, 1000);
        assert!(
            vector_only > scalar_only,
            "a vector byte must reserve more than a scalar byte ({vector_only} vs {scalar_only})"
        );
        assert!(
            scalar_plus_fts > scalar_only,
            "the FTS term is additive on top of scalar ({scalar_plus_fts} vs {scalar_only})"
        );
    }

    #[test]
    fn append_over_budget_is_refused() {
        // A 1-byte bounded budget floors the enforced gate to 0, so building
        // any non-empty batch (whose weighted reserve is > 0) is refused. The
        // public folded append surfaces it as InfinoError::OverBudget.
        let mut opts = options_id_title_serial();
        opts.connection_memory_budget = ConnectionMemoryBudget::with_limit(1);
        let st = Supertable::create(opts).expect("create");

        let err = st
            .append(&build_simple_batch(0, 8))
            .expect_err("build over a 0-byte gate is refused");
        let InfinoError::OverBudget(msg) = err else {
            panic!("expected InfinoError::OverBudget, got {err:?}");
        };
        // The message names the ingest path, so a caller can tell it apart from
        // a query or SQL over-budget error (which share the OverBudget variant).
        assert!(
            msg.contains("ingest"),
            "over-budget message should identify ingest: {msg}"
        );

        // Nothing was published, and a refused reservation commits nothing.
        assert_eq!(st.reader().n_docs_total(), 0);
        assert!(st.options().connection_memory_budget.denials() >= 1);
        assert_eq!(st.options().connection_memory_budget.peak(), 0);
    }

    #[test]
    fn append_under_measured_budget_runs_and_tracks_peak() {
        // Measured budget never refuses; the build still reserves, so peak > 0
        // proves the reservation ran on the ingest path.
        let mut opts = options_id_title_serial();
        opts.connection_memory_budget = ConnectionMemoryBudget::measured();
        let st = Supertable::create(opts).expect("create");

        st.append(&build_simple_batch(0, 8))
            .expect("measured budget never refuses");
        assert_eq!(st.reader().n_docs_total(), 8);

        let budget = &st.options().connection_memory_budget;
        assert_eq!(budget.denials(), 0);
        assert!(
            budget.peak() > 0,
            "the build must reserve against the budget"
        );
    }

    #[test]
    fn append_under_ample_bounded_budget_runs() {
        // A bounded (enforcing) budget well above the build must admit the
        // ingest, not refuse on principle.
        const AMPLE_BUDGET_BYTES: u64 = 1 << 30; // 1 GiB, far above an 8-row batch.
        let mut opts = options_id_title_serial();
        opts.connection_memory_budget = ConnectionMemoryBudget::with_limit(AMPLE_BUDGET_BYTES);
        let st = Supertable::create(opts).expect("create");

        st.append(&build_simple_batch(0, 8))
            .expect("under-budget append runs under a bounded budget");
        assert_eq!(st.reader().n_docs_total(), 8);

        let budget = &st.options().connection_memory_budget;
        assert_eq!(budget.denials(), 0);
        assert!(budget.limit().is_some(), "bounded, not measured");
    }

    #[test]
    fn over_budget_commit_preserves_the_buffer() {
        // Reserving before draining the buffer means a refused commit leaves the
        // buffered rows intact, so the caller can retry or back off.
        let mut opts = options_id_title_serial();
        opts.connection_memory_budget = ConnectionMemoryBudget::with_limit(1);
        let st = Supertable::create(opts).expect("create");

        let mut w = st.writer().expect("writer");
        w.append(&build_simple_batch(0, 8)).expect("append buffers");
        assert_eq!(w.buffered_batches(), 1);

        let err = w
            .commit()
            .expect_err("commit over a 0-byte gate is refused");
        assert!(
            matches!(err, CommitError::AppendFlush(BuildError::OverBudget(_))),
            "got {err:?}"
        );
        // The buffer was not drained.
        assert_eq!(w.buffered_batches(), 1);
    }

    #[test]
    fn auto_flush_over_budget_is_refused_from_append() {
        // With a commit threshold set, `append` auto-flushes once the buffer
        // crosses it, so the refusal surfaces out of `append` itself (the
        // auto-flush exit) rather than an explicit `commit`. A batch large
        // enough to exceed the 1 MiB threshold in one call trips it.
        const AUTO_FLUSH_TRIP_ROWS: usize = 40_000;
        let mut opts = options_id_title_serial().with_commit_threshold_size_mb(1);
        opts.connection_memory_budget = ConnectionMemoryBudget::with_limit(1);
        let st = Supertable::create(opts).expect("create");

        let mut w = st.writer().expect("writer");
        let err = w
            .append(&build_simple_batch(0, AUTO_FLUSH_TRIP_ROWS))
            .expect_err("auto-flush over a 0-byte gate is refused");
        assert!(matches!(err, BuildError::OverBudget(_)), "got {err:?}");
    }

    #[test]
    fn vector_ingest_over_budget_is_refused() {
        // The gate covers the vector build path too: a vector-schema ingest over
        // a 0-byte gate is refused as the public OverBudget, nothing published.
        let dim = 16;
        let mut opts = options_with_vector(dim);
        opts.connection_memory_budget = ConnectionMemoryBudget::with_limit(1);
        let st = Supertable::create(opts).expect("create");

        let err = st
            .append(&build_vector_batch(0, 8, dim))
            .expect_err("vector build over a 0-byte gate is refused");
        assert!(matches!(err, InfinoError::OverBudget(_)), "got {err:?}");
        assert_eq!(st.reader().n_docs_total(), 0);
    }

    #[test]
    fn vector_ingest_reserves_and_runs_under_measured() {
        // Measured never refuses; peak > 0 proves the vector build (kmeans +
        // quantization + serialized blob) actually reserved against the budget.
        let dim = 16;
        let mut opts = options_with_vector(dim);
        opts.connection_memory_budget = ConnectionMemoryBudget::measured();
        let st = Supertable::create(opts).expect("create");

        st.append(&build_vector_batch(0, 8, dim))
            .expect("measured vector ingest runs");
        assert_eq!(st.reader().n_docs_total(), 8);

        let budget = &st.options().connection_memory_budget;
        assert_eq!(budget.denials(), 0);
        assert!(
            budget.peak() > 0,
            "the vector build must reserve against the budget"
        );
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

    /// A consumer-memory-mode handle (`summary_centroids_from_superfiles`)
    /// hydrates summaries without fp32, so committing from it would hit
    /// the wire encoder's stripped-summary panic deep inside the commit.
    /// The writer slot refuses up front instead.
    #[test]
    fn consumer_memory_mode_handle_refuses_writer() {
        let opts = options_id_title_serial().with_summary_centroids_from_superfiles(true);
        let st = Supertable::create(opts).expect("create");
        let err = st
            .writer()
            .expect_err("consumer-mode handle must not write");
        assert!(
            err.to_string().contains("consumer memory mode"),
            "unexpected refusal: {err}"
        );
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
                provided_centroids: None,
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
        // Per-cluster centroids are staged into the manifest for
        // cross-superfile global cluster selection.
        assert!(
            vs.cells.iter().any(|cell| !cell.clusters.is_empty()),
            "cluster centroids must be populated"
        );
        assert!(vs.cells.iter().all(|cell| {
            cell.clusters.dim as usize == dim
                && cell.clusters.n_cent >= 1
                && cell.clusters.counts.len() == cell.clusters.n_cent as usize
                && cell.clusters.centroids.len() == cell.clusters.n_cent as usize * dim
        }));
        // Every Parquet row lands in at least one cluster; boundary
        // replication (on by default) may add stub copies up to the
        // configured storage-amplification budget on top.
        let total: u64 = vs
            .cells
            .iter()
            .flat_map(|cell| cell.clusters.counts.iter())
            .map(|&count| count as u64)
            .sum();
        assert!(total >= seg.n_docs, "counts {total} < rows {}", seg.n_docs);
        let budget_cap = (seg.n_docs as f64
            * f64::from(config::global().vector.drain_replica_target_factor.max(1.0)))
        .ceil() as u64;
        assert!(
            total <= budget_cap,
            "counts {total} exceed replica budget cap {budget_cap}"
        );
    }

    #[test]
    fn grid_commit_writes_multicell_parquet_in_vector_order() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(
            options_title_emb_serial(COMMIT_AS_DRAIN_TEST_DIM, COMMIT_AS_DRAIN_TEST_ROWS)
                .with_storage(storage),
        )
        .expect("create");
        assert!(
            !st.reader().options().vector_columns.is_empty(),
            "fixture must declare vector columns so commit takes the assign-pack path"
        );
        let mut w = st.writer().expect("writer");
        w.append(&build_axis_vector_batch(
            COMMIT_AS_DRAIN_TEST_ROWS,
            COMMIT_AS_DRAIN_TEST_DIM,
        ))
        .expect("append");
        w.commit().expect("commit");

        let (entry, reader) = committed_reader(&st);
        assert_eq!(entry.vector_layout, VectorLayout::MultiCellIvf);
        assert_eq!(entry.n_docs, COMMIT_AS_DRAIN_TEST_ROWS as u64);

        let vec_reader = reader.vec().expect("vector reader");
        assert!(vec_reader.is_multi_cell());
        let vector_locals: Vec<u32> = (0..vec_reader.n_docs() as u32).collect();
        let vector_ids = vec_reader
            .inline_stable_ids_for_locals(&vector_locals)
            .expect("inline stable ids");

        let parquet_locals: Vec<u32> = (0..entry.n_docs as u32).collect();
        let parquet_batch = reader
            .take_by_local_doc_ids(&parquet_locals, &["_id"])
            .expect("read parquet ids");
        let parquet_ids = parquet_batch
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal ids")
            .values()
            .to_vec();
        // Parquet stores each row once, in vector (cell) order. The IVF
        // additionally carries boundary-replica stubs (replication is on by
        // default), so the inline id stream is the parquet order plus stub
        // duplicates: first occurrences must line up 1:1 with parquet, and
        // every remaining inline id must duplicate some parquet row.
        let mut seen = HashSet::new();
        let first_occurrence: Vec<i128> = vector_ids
            .iter()
            .copied()
            .filter(|id| seen.insert(*id))
            .collect();
        assert_eq!(parquet_ids, first_occurrence);
        let parquet_set: HashSet<i128> = parquet_ids.iter().copied().collect();
        assert!(
            vector_ids.iter().all(|id| parquet_set.contains(id)),
            "every stub must duplicate a parquet row"
        );
    }

    #[test]
    fn assign_pack_boundary_replicas_are_vector_only_stubs() {
        let dim = COMMIT_AS_DRAIN_TEST_DIM;
        let mut centroids = vec![0.0f32; dim * 2];
        centroids[dim] = 1.0;
        let clusters = ClusterCentroids::from_fp32(2, dim as u32, &centroids, vec![0, 0]);
        let vectors = [
            vec![0.49; dim],
            vec![0.51; dim],
            vec![0.48; dim],
            vec![0.52; dim],
        ];
        let stable_ids = [10_i128, 11, 12, 13];
        let rows: Vec<PackRow<'_>> = vectors
            .iter()
            .zip(stable_ids)
            .map(|(vector, stable_id)| PackRow::Fp32 { stable_id, vector })
            .collect();
        let assigned = assign_cells(
            &rows,
            &clusters,
            Metric::L2Sq,
            COMMIT_AS_DRAIN_TEST_ROT_SEED,
            BOUNDARY_STUB_TARGET_FACTOR,
        )
        .expect("assign");

        let postings: usize = assigned.iter().map(|group| group.members.len()).sum();
        let primaries: usize = assigned
            .iter()
            .flat_map(|group| group.members.iter())
            .filter(|(_, is_primary, _)| *is_primary)
            .count();
        assert_eq!(primaries, rows.len());
        assert!(
            postings > primaries,
            "boundary replicas add vector postings, not primary rows"
        );

        let cfg = VectorConfig {
            column: "emb".into(),
            dim,
            n_cent: 2,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
            provided_centroids: None,
        };
        for group in assigned {
            let n_members = group.members.len();
            let packed = drain_pack_assigned_cell(group, &cfg).expect("drain pack");
            assert_eq!(packed.stable_ids.len(), n_members);
            assert_eq!(packed.subsection.n_docs as usize, n_members);
        }
    }

    #[test]
    fn drain_fine_centroids_follow_two_mib_run_target() {
        const DIM: usize = 1024;
        const COMMIT_CELL_ROWS: usize = 98;
        const DRAINED_CELL_ROWS: usize = 1_562;

        let cfg = VectorConfig {
            column: "emb".into(),
            dim: DIM,
            n_cent: 64,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
            provided_centroids: None,
        };
        assert_eq!(
            drain_cell_vector_config(&cfg, COMMIT_CELL_ROWS).n_cent,
            1,
            "a small commit delta fits one ~2 MiB fine run"
        );
        assert_eq!(
            drain_cell_vector_config(&cfg, DRAINED_CELL_ROWS).n_cent,
            2,
            "a fully drained cell needs two ~2 MiB fine runs"
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

    /// `put_superfile_replace` creates on first write (NotFound → put_atomic)
    /// and overwrites on the second (head → put_if_match), leaving the object
    /// content equal to the most recent bytes written.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn put_superfile_replace_creates_then_overwrites() {
        let directory = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(directory.path()).expect("provider"));
        let path = "superfiles/replace-me.sf";

        // First write to a fresh path routes through the NotFound → put_atomic
        // arm and creates the object.
        let first = Bytes::from_static(b"first-body-contents");
        put_superfile_replace(&storage, path, first.clone())
            .await
            .expect("first put creates");
        let (read_first, _) = storage.get(path).await.expect("read after create");
        assert_eq!(read_first, first, "created object holds the first bytes");

        // Second write to the same path routes through head → put_if_match and
        // replaces the content.
        let second = Bytes::from_static(b"second-body-different-length");
        put_superfile_replace(&storage, path, second.clone())
            .await
            .expect("second put overwrites");
        let (read_second, _) = storage.get(path).await.expect("read after overwrite");
        assert_eq!(read_second, second, "overwrite installs the new bytes");
        assert_ne!(
            read_second, read_first,
            "object content actually changed between writes"
        );
    }
}
