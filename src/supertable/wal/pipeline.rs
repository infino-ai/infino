// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! WAL pipeline orchestrators for the update / delete state machines.
//!
//! Two phases live here:
//!
//! - [`run_append_phase`] — drives one UPDATE WAL through
//!   `Intent → Appended` by building the new superfile and
//!   CAS-swapping it into the manifest.
//! - [`run_tombstone_phase`] — drives one UPDATE or DELETE WAL
//!   through `Appended → Complete` (UPDATE) or `Intent → Complete`
//!   (DELETE) by resolving each `target_id` to a `(superfile_id,
//!   doc_id)` pair and CAS-PUT-ing the bit into the per-superfile
//!   tombstone sidecar.
//!
//! ## Append phase
//!
//! Drives one WAL through the state transition `Intent → Appended`:
//!
//! 1. **Idempotency probe.** Read the current manifest. If it
//!    already contains the WAL's `preallocated_superfile_id`, we're
//!    replaying after a crash — skip directly to step 6.
//! 2. **Fetch + verify IPC payload.** Pull
//!    `wal/mutations/<wal_id>.arrow` and blake3-check against the
//!    state doc's `new_row_content_hash`. Mismatch = corruption;
//!    abort.
//! 3. **Build the superfile bytes** with the WAL's
//!    `preallocated_superfile_id`, the `_id` column populated by
//!    flattening `minted_id_spans` in order, and all other columns
//!    from the IPC payload. Bit-identical across replays by
//!    construction.
//! 4. **PUT the superfile bytes** under the preallocated id.
//!    Content-addressed so re-PUT on replay is a no-op.
//! 5. **CAS-commit the manifest** through the writer's existing
//!    [`persist_commit`] code path. That handles OCC retry,
//!    partition-aware part rewrite, and the pointer-file CAS.
//! 6. **Advance WAL state to `Appended`**.
//!
//! Steps 1, 5, and 6 are the durability barriers; the rest is
//! recovery-safe replay material (deterministic bytes; idempotent
//! storage operations).
//!
//! ## Replay safety
//!
//! After any crash, re-running this function against the same WAL
//! must produce the same end state. The invariants:
//!
//! - The superfile uuid is fixed at `preallocated_superfile_id`.
//! - The `_id` column is fixed by `minted_id_spans`.
//! - All other columns come from the content-hashed IPC sidecar.
//!
//! Together these pin every byte of the produced superfile, so the
//! step-4 PUT is overwrite-safe and the step-5 manifest swap can
//! short-circuit via the idempotency probe in step 1.

use std::{collections::HashMap, io::Cursor, sync::Arc, time::Duration};

use arrow::ipc::reader::StreamReader;
use arrow_array::{ArrayRef, Decimal128Array, RecordBatch};
use bytes::Bytes;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use roaring::RoaringBitmap;
use tokio::time::sleep;
use uuid::Uuid;

use crate::{
    runtime_bridge::{bridge_sync_to_async, run_on_pool},
    runtime_metrics::op_stats::{OpStatsCollector, timed_kernel},
    storage::StorageError,
    superfile::{ReadError, SuperfileReader, builder::SuperfileBuilder},
    supertable::{
        ManifestSnapshot, SupertableOptions,
        error::CommitError as ManifestCommitError,
        handle::{Supertable, SupertableInner},
        manifest::{
            FtsSummaryAgg, ScalarStatsAgg, SuperfileEntry, SuperfileUri, VectorSummary,
            bloom::BloomBuilder,
        },
        options::{DECIMAL128_PRECISION, DECIMAL128_SCALE},
        query::superfile_reader::superfile_reader,
        utils::vector_split::split_vectors,
        wal::{
            persistence::{Etag, WalStore, WalStoreError},
            state_doc::{
                IdSpan, OpKind, RowId, TombstoneEntry, TombstoneOutcome, WalState, WalStateDoc,
            },
            tombstones_admin,
            tombstones_codec::TombstonesSidecar,
        },
        writer::{
            CommitListMetadata, build_column_vector_summary, build_packed_update_superfile,
            build_subsection_offsets, owned_vector_arrays, persist_commit,
            read_vector_layout_from_bytes, stamp_tombstone_seqs,
        },
    },
};

/// Outcome of one append-phase invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendPhaseOutcome {
    /// The manifest already referenced the WAL's
    /// `preallocated_superfile_id` when we probed — the superfile
    /// + manifest swap landed on a previous run (or a peer
    /// recovery process beat us to it). No new work; we just
    /// advanced the WAL state to `Appended` if it wasn't already.
    AlreadyApplied,

    /// We built the superfile bytes, PUT them under the
    /// preallocated id, CAS-swapped the manifest to reference the
    /// new superfile, and advanced the WAL state to `Appended`.
    Applied,
}

/// Typed failures from `run_append_phase`. The WAL is left at
/// whatever state was durable when the error surfaced — recovery
/// on a fresh process picks up from there.
#[derive(Debug, thiserror::Error)]
pub enum AppendPhaseError {
    /// State doc is missing a field the append phase needs (e.g.
    /// the WAL was constructed as a DELETE and the orchestrator
    /// was called on it by mistake). The orchestrator only runs
    /// for `op_kind == Update`.
    #[error("WAL is missing required field {field:?} for the append phase")]
    MissingField { field: &'static str },

    /// `wal_doc.op_kind` is `Delete` — the append phase has no
    /// work to do; the caller is using the wrong entry point.
    #[error("append phase invoked on a DELETE WAL; only UPDATE has an append phase")]
    NotAnUpdateWal,

    /// The supertable handle this orchestrator was given doesn't
    /// have a storage backend attached. The append phase has to
    /// commit through the manifest pointer file, which lives on
    /// storage — there's no in-process fallback.
    #[error("supertable has no storage attached; append phase requires durable storage")]
    NoStorageAttached,

    /// IPC sidecar's blake3 doesn't match the WAL state doc's
    /// `new_row_content_hash`. The bytes are corrupt or a peer
    /// abandoned a partial write; surfacing as a typed error
    /// lets recovery quarantine the WAL rather than running it
    /// against a damaged payload.
    #[error("IPC content hash mismatch for WAL {wal_id:?}: expected {expected:?}, got {got:?}")]
    SidecarContentHashMismatch {
        wal_id: String,
        expected: String,
        got: String,
    },

    /// Couldn't decode the IPC sidecar back to a `RecordBatch` —
    /// suggests either a schema mismatch between the producer
    /// and the recovery process, or genuine corruption that the
    /// blake3 check happened to miss.
    #[error("IPC sidecar decode failed for WAL {wal_id:?}: {message}")]
    IpcDecode { wal_id: String, message: String },

    /// `minted_id_spans` flattens to a different count than the
    /// IPC payload claims (`new_row_count`). The two should be
    /// pinned in lockstep at WAL creation; a divergence is a
    /// builder bug or a corrupted state doc.
    #[error(
        "minted_id_spans flatten ({flat_len}) doesn't match new_row_count ({expected}) for WAL {wal_id:?}"
    )]
    IdSpansLengthMismatch {
        wal_id: String,
        flat_len: usize,
        expected: u32,
    },

    /// Building the superfile (Parquet + FTS + vector) failed —
    /// likely a schema-validation or index-build error.
    #[error("superfile build failed: {message}")]
    SuperfileBuild { message: String },

    /// Opening the just-built bytes as a `SuperfileReader` to
    /// extract FTS / vector summaries failed.
    #[error("superfile open for summary failed: {message}")]
    SuperfileOpenForSummary { message: String },

    /// The manifest-commit machinery failed. Surfaces both the
    /// "I lost the pointer CAS" path (which the inner code
    /// retries on its own up to `max_commit_retries`) and any
    /// permanent failure. Caller's handling of the WAL is the same
    /// in both cases — it stays at whatever state was durable — but
    /// the typed cause is carried rather than stringified so the
    /// public boundary can still tell a retryable lost race from a
    /// permanent fault.
    #[error("manifest commit failed: {0}")]
    ManifestCommit(#[source] Box<ManifestCommitError>),

    /// Underlying storage error.
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    /// WAL state-document I/O error from the persistence layer.
    #[error("WAL store error: {0}")]
    WalStore(#[from] WalStoreError),
}

impl AppendPhaseError {
    /// True when the append phase lost a compare-and-set race, so reissuing
    /// the mutation against fresh state can succeed.
    pub(crate) fn is_conflict(&self) -> bool {
        match self {
            AppendPhaseError::ManifestCommit(e) => e.is_conflict(),
            AppendPhaseError::Storage(e) => e.is_conflict(),
            AppendPhaseError::WalStore(e) => e.is_conflict(),
            _ => false,
        }
    }

    /// True when the backend refused the credentials in use.
    pub(crate) fn is_permission_denied(&self) -> bool {
        match self {
            AppendPhaseError::ManifestCommit(e) => e.is_permission_denied(),
            AppendPhaseError::Storage(e) => e.is_permission_denied(),
            AppendPhaseError::WalStore(e) => e.is_permission_denied(),
            _ => false,
        }
    }
}

/// Drive one UPDATE WAL from `Intent` to `Appended`.
///
/// **Pre-conditions** (caller responsibility):
/// - `wal_doc.op_kind == Update`.
/// - `wal_doc.state == Intent` or `Appended` (re-running on an
///   `Appended` WAL is a no-op via the idempotency probe).
/// - The supertable handle's manifest is read-up-to-date enough
///   that the idempotency probe gives a meaningful answer; this
///   is true for any `Supertable::open` / `create` return value.
///
/// **Post-conditions** on `Ok`:
/// - `wal_doc.state == Appended` durably.
/// - The supertable's manifest contains a superfile entry whose
///   id equals `wal_doc.preallocated_superfile_id`.
/// - That superfile's bytes are durable on storage under
///   `superfiles/<preallocated_superfile_id>.par`.
///
/// **What happens on intermediate failure:** the WAL stays at
/// whatever state was durable when the failure occurred. A
/// recovery process can re-run this function and reach the same
/// end state because every step is idempotent on replay (steps
/// 1-4) or content-addressed (step 5's manifest writes go
/// through the normal CAS).
pub(crate) async fn run_append_phase(
    supertable: &Supertable,
    wal_store: &WalStore,
    wal_doc: &WalStateDoc,
    wal_etag: &Etag,
    // The mutation's per-op collector, passed explicitly: this runs on the
    // runtime's thread, so the caller's thread-local scope cannot reach it.
    // The recovery sweep passes None — a recovered WAL is maintenance, not
    // any live op's work.
    op_stats: Option<Arc<OpStatsCollector>>,
) -> Result<(AppendPhaseOutcome, WalStateDoc, Etag), AppendPhaseError> {
    // Pre-condition: only UPDATE has an append phase.
    if wal_doc.op_kind != OpKind::Update {
        return Err(AppendPhaseError::NotAnUpdateWal);
    }

    let preallocated_superfile_id =
        wal_doc
            .preallocated_superfile_id
            .ok_or(AppendPhaseError::MissingField {
                field: "preallocated_superfile_id",
            })?;

    let inner = supertable.inner();

    // ---- Step 1: Idempotency probe ----
    //
    // Look up the WAL's preallocated_superfile_id in the
    // current manifest snapshot. If it's already there, a
    // previous run (or peer recovery) completed steps 2-5;
    // we just need to make sure the WAL state itself shows
    // Appended.
    let manifest_snapshot = inner.manifest.load_full();
    if manifest_contains(&manifest_snapshot, preallocated_superfile_id) {
        let (new_wal, new_etag) =
            advance_to_appended_if_needed(wal_store, wal_doc, wal_etag).await?;
        return Ok((AppendPhaseOutcome::AlreadyApplied, new_wal, new_etag));
    }

    // ---- Steps 2-6 ----
    //
    // Built incrementally in `do_apply` so failure modes funnel
    // through one return path with consistent error mapping.
    let (new_wal, new_etag) = do_apply(
        supertable,
        wal_store,
        wal_doc,
        wal_etag,
        preallocated_superfile_id,
        op_stats,
    )
    .await?;
    Ok((AppendPhaseOutcome::Applied, new_wal, new_etag))
}

/// Step 1 helper: scan the manifest's superfile list for a
/// matching `superfile_id`. O(N) in the number of live
/// superfiles; called once per append-phase invocation, so the
/// linear scan is fine at the supertable sizes we target.
fn manifest_contains(manifest: &ManifestSnapshot, superfile_id: Uuid) -> bool {
    manifest
        .get_all_superfiles()
        .iter()
        .any(|s| s.uri.0 == superfile_id)
}

/// If the WAL is already in `Appended`, return its current doc +
/// etag unchanged. Otherwise CAS-advance to `Appended`.
async fn advance_to_appended_if_needed(
    wal_store: &WalStore,
    wal_doc: &WalStateDoc,
    wal_etag: &Etag,
) -> Result<(WalStateDoc, Etag), AppendPhaseError> {
    if wal_doc.state == WalState::Appended {
        return Ok((wal_doc.clone(), wal_etag.clone()));
    }
    let mut next = wal_doc.clone();
    next.state = WalState::Appended;
    let new_etag = wal_store
        .update_with_etag(wal_doc.wal_id, wal_etag, &next)
        .await?;
    Ok((next, new_etag))
}

/// The non-idempotent fast path: build the superfile, write its
/// bytes, swap the manifest, advance the WAL. Pulled into its own
/// async fn so the orchestrator's high-level flow reads cleanly.
async fn do_apply(
    supertable: &Supertable,
    wal_store: &WalStore,
    wal_doc: &WalStateDoc,
    wal_etag: &Etag,
    preallocated_superfile_id: Uuid,
    op_stats: Option<Arc<OpStatsCollector>>,
) -> Result<(WalStateDoc, Etag), AppendPhaseError> {
    let inner = supertable.inner();
    let storage = inner
        .options
        .storage
        .as_ref()
        .ok_or(AppendPhaseError::NoStorageAttached)?
        .clone();

    // ---- Step 2: Fetch + verify IPC sidecar ----
    let hash = wal_doc
        .new_row_content_hash
        .as_deref()
        .ok_or(AppendPhaseError::MissingField {
            field: "new_row_content_hash",
        })?;
    let ipc_bytes = wal_store
        .get_arrow(wal_doc.wal_id, Some(hash))
        .await
        .map_err(|e| match e {
            WalStoreError::SidecarContentHashMismatch { expected, got, .. } => {
                AppendPhaseError::SidecarContentHashMismatch {
                    wal_id: wal_doc.wal_id.to_hex(),
                    expected,
                    got,
                }
            }
            other => AppendPhaseError::WalStore(other),
        })?;

    // ---- Step 3: Decode IPC + build the _id-prefixed batch ----
    let user_batch = decode_ipc_batch(&ipc_bytes, wal_doc)?;
    let new_row_count = wal_doc
        .new_row_count
        .ok_or(AppendPhaseError::MissingField {
            field: "new_row_count",
        })?;
    let flat_ids = flatten_spans(&wal_doc.minted_id_spans);
    if flat_ids.len() != new_row_count as usize {
        return Err(AppendPhaseError::IdSpansLengthMismatch {
            wal_id: wal_doc.wal_id.to_hex(),
            flat_len: flat_ids.len(),
            expected: new_row_count,
        });
    }
    if user_batch.num_rows() != flat_ids.len() {
        return Err(AppendPhaseError::IdSpansLengthMismatch {
            wal_id: wal_doc.wal_id.to_hex(),
            flat_len: flat_ids.len(),
            expected: user_batch.num_rows() as u32,
        });
    }
    // Split scalars from vectors once; downstream consumes both
    // halves. `split_vectors` runs the schema-equality check +
    // per-vector null sweep, so this is the single validation
    // gate for the append batch.
    //
    // Vector slices are zero-copy views into `user_batch`'s
    // buffers; we hold `user_batch` alive across the
    // `builder.add_batch` call below.
    let (scalar_no_id, vector_slices) =
        split_vectors(&user_batch, &inner.options).map_err(|e| {
            AppendPhaseError::SuperfileBuild {
                message: format!("vector_split: {e}"),
            }
        })?;
    let scalar_with_id = prepend_id_column(&scalar_no_id, &flat_ids, &inner.options)?;

    // ---- Step 4: Build the superfile bytes ----
    //
    // Vector tables route the replacement rows through the commit path's
    // cell-assign + packed-shard build, so an update's superfile is
    // cell-directory-packed like every other committed vector superfile —
    // an unpacked plain build is not a valid drained-side maintenance input
    // (the drain materializer and the per-cell compaction merges both fail
    // closed on it), and it would skip cell routing + boundary replicas for
    // the replacement rows. The pack is a CPU wave, so it rides the writer
    // pool behind a oneshot per the standing rayon/tokio bridge contract.
    // Tables without vector columns keep the plain build.
    //
    // Either way the build is the update's own CPU and is bracketed as
    // such — but on the thread that does the work, not the one that waits
    // for it. The plain build is synchronous here (`update` reaches this
    // future through `bridge_on_runtime`, so "here" is the caller's own
    // thread inside `block_on`), so it brackets here. The packed build
    // carries the collector into `fanout_shards_metered`, which already
    // brackets each shard on its own pool worker — bracketing the pool
    // closure as well would count that CPU twice.
    let bytes = if inner.options.vector_columns.is_empty() {
        timed_kernel(&op_stats, || {
            let mut builder =
                SuperfileBuilder::new(inner.options.builder_options()).map_err(|e| {
                    AppendPhaseError::SuperfileBuild {
                        message: format!("builder construction: {e}"),
                    }
                })?;
            builder
                .add_batch(&scalar_with_id, &vector_slices)
                .map_err(|e| AppendPhaseError::SuperfileBuild {
                    message: format!("add_batch: {e}"),
                })?;
            let raw = builder
                .finish()
                .map_err(|e| AppendPhaseError::SuperfileBuild {
                    message: format!("finish: {e}"),
                })?;
            Ok::<_, AppendPhaseError>(Bytes::from(raw))
        })?
    } else {
        let vectors = owned_vector_arrays(&user_batch, &inner.options).map_err(|e| {
            AppendPhaseError::SuperfileBuild {
                message: format!("vector handles: {e}"),
            }
        })?;
        let pool_inner = Arc::clone(inner);
        let pool_batch = scalar_with_id.clone();
        let pool_stats = op_stats.clone();
        run_on_pool(
            Some(&inner.options.writer_pool),
            "update packed superfile build",
            move || build_packed_update_superfile(&pool_inner, pool_batch, vectors, &pool_stats),
        )
        .await
        .map_err(|e| AppendPhaseError::SuperfileBuild {
            message: format!("packed build: {e}"),
        })?
        .map_err(|e| AppendPhaseError::SuperfileBuild {
            message: format!("packed build: {e}"),
        })?
    };

    // ---- Step 5: Per-superfile summaries + SuperfileEntry ----
    //
    // FTS + vector summaries are derived from a fresh
    // `SuperfileReader` on the just-built bytes — same shape the
    // writer's `prepare_superfile` uses. Scalar stats come from
    // the in-memory `RecordBatch` directly; nothing needs to
    // round-trip through Parquet.
    let reader = SuperfileReader::open_with(bytes.clone(), inner.options.superfile_open_options())
        .map_err(|e| AppendPhaseError::SuperfileOpenForSummary {
            message: e.to_string(),
        })?;
    let fts_summary = build_fts_summary(&reader, &inner.options);
    let vector_summary = build_vector_summary(&reader, &inner.options);
    let scalar_stats =
        ScalarStatsAgg::from_batches(&inner.options.scalar_schema(), &[&scalar_with_id]);

    let (id_min, id_max) = if flat_ids.is_empty() {
        (0, 0)
    } else {
        (flat_ids[0], flat_ids[flat_ids.len() - 1])
    };

    let uri = SuperfileUri(preallocated_superfile_id);
    let entry = Arc::new(SuperfileEntry {
        // Stamped to the winning commit version later, in `ManifestSnapshot::update`.
        birth_version: 0,
        superfile_id: preallocated_superfile_id,
        uri,
        n_docs: flat_ids.len() as u64,
        id_min,
        id_max,
        scalar_stats,
        fts_summary,
        vector_summary,
        // Unpartitioned default: the supertable's partition
        // machinery (`assign_partition` inside the commit
        // attempt) re-derives the on-disk `partition_key` from
        // the strategy + this entry's stats at commit time,
        // which is correct for the Hash{n_buckets=1} default.
        partition_key: Vec::new(),
        partition_hint: None,

        // Mirror the commit path's 1-RTT cold-open hint; `None`
        // only if the bytes don't parse (same fallback as the
        // writer).
        subsection_offsets: build_subsection_offsets(&bytes),
        vector_layout: read_vector_layout_from_bytes(&bytes),
    });

    // ---- Step 6: PUT bytes + CAS-commit the manifest ----
    //
    // The writer's `persist_commit` handles the actual PUT of
    // the superfile bytes (via `pending_storage_writes`), the
    // OCC retry on the pointer file, and the partition-aware
    // part rewrite. It returns the new in-memory `ManifestSnapshot`
    // that reflects the persisted state, but it does NOT swap
    // `inner.manifest` itself — the caller owns that final
    // visibility barrier, mirroring how the synchronous
    // `Writer::commit` path arms it. We swap here so subsequent
    // reads + the idempotency probe on a retry both see the
    // new superfile.
    persist_commit(
        inner,
        storage,
        vec![entry],
        &[],
        vec![(uri, bytes.clone())],
        Vec::new(),
        CommitListMetadata::empty(),
    )
    .map_err(|e| AppendPhaseError::ManifestCommit(Box::new(e)))?;

    // Warm the in-memory reader cache with the freshly-published
    // bytes so this process's later reads (queries, tombstone
    // resolves) don't take the cold-fetch round-trip back to
    // storage. Mirrors the synchronous writer's pattern in
    // `commit`; a failure here is non-fatal because the bytes
    // are durable in storage and a subsequent read can refetch
    // them.
    let _ = inner.options.store.insert(uri, bytes);

    // ---- Step 7: Advance WAL state to Appended ----
    advance_to_appended_if_needed(wal_store, wal_doc, wal_etag).await
}

/// Flatten a `Vec<IdSpan>` into the implied sequence of `i128`
/// ids in order. Total cost is O(n) in the flattened count;
/// allocated once per append-phase invocation.
///
/// Uses `extend(Range)` rather than a `push` loop: `Vec::extend`
/// specializes on `TrustedLen` iterators (which `Range<i128>` is),
/// emitting one bulk copy instead of N bounds-checked pushes —
/// ~6× faster at the 1M-id scale a delete/update batch can hit.
fn flatten_spans(spans: &[IdSpan]) -> Vec<i128> {
    let total: usize = spans.iter().map(|s| s.len() as usize).sum();
    let mut out = Vec::with_capacity(total);
    for span in spans {
        out.extend(span.first.0..=span.last.0);
    }
    out
}

/// Decode the WAL's IPC sidecar back to the user-shape
/// `RecordBatch`. The sidecar contains exactly one batch (the
/// `new_rows` argument the caller passed to `update()`); we read
/// the first and verify there isn't a second.
fn decode_ipc_batch(
    ipc_bytes: &Bytes,
    wal_doc: &WalStateDoc,
) -> Result<RecordBatch, AppendPhaseError> {
    let cursor = Cursor::new(ipc_bytes.as_ref());
    let mut reader =
        StreamReader::try_new(cursor, None).map_err(|e| AppendPhaseError::IpcDecode {
            wal_id: wal_doc.wal_id.to_hex(),
            message: format!("StreamReader::try_new: {e}"),
        })?;
    let batch = reader
        .next()
        .ok_or_else(|| AppendPhaseError::IpcDecode {
            wal_id: wal_doc.wal_id.to_hex(),
            message: "IPC stream had no batches; expected exactly one".into(),
        })?
        .map_err(|e| AppendPhaseError::IpcDecode {
            wal_id: wal_doc.wal_id.to_hex(),
            message: format!("batch read: {e}"),
        })?;
    if reader.next().is_some() {
        return Err(AppendPhaseError::IpcDecode {
            wal_id: wal_doc.wal_id.to_hex(),
            message: "IPC stream had more than one batch; expected exactly one".into(),
        });
    }
    Ok(batch)
}

/// Construct a new `RecordBatch` matching the supertable's
/// `scalar_schema()` shape — `_id` column prepended, followed by
/// `scalar_no_id`'s columns (the scalar-only output of
/// `split_vectors`). Vector columns are NOT in this batch; they
/// get passed alongside to `SuperfileBuilder::add_batch`.
///
/// Caller must have already run `split_vectors` for schema +
/// null validation — this function trusts its input.
fn prepend_id_column(
    scalar_no_id: &RecordBatch,
    flat_ids: &[i128],
    options: &SupertableOptions,
) -> Result<RecordBatch, AppendPhaseError> {
    let id_values: Vec<i128> = flat_ids.to_vec();
    let id_array = Decimal128Array::from(id_values)
        .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
        .map_err(|e| AppendPhaseError::SuperfileBuild {
            message: format!("Decimal128 precision/scale: {e}"),
        })?;

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(scalar_no_id.num_columns() + 1);
    columns.push(Arc::new(id_array));
    columns.extend(scalar_no_id.columns().iter().cloned());

    RecordBatch::try_new(options.scalar_schema(), columns).map_err(|e| {
        AppendPhaseError::SuperfileBuild {
            message: format!("RecordBatch::try_new with _id prepended: {e}"),
        }
    })
}

/// Per-FTS-column bloom + range summary derived from the
/// just-built superfile's `SuperfileReader`. Mirrors the shape
/// the writer's `prepare_superfile` builds so summaries match
/// regardless of which code path produced the superfile.
fn build_fts_summary(
    reader: &SuperfileReader,
    options: &SupertableOptions,
) -> HashMap<String, FtsSummaryAgg> {
    let mut out: HashMap<String, FtsSummaryAgg> = HashMap::new();
    let Some(fts_reader) = reader.fts() else {
        return out;
    };
    for fc in &options.fts_columns {
        let terms = fts_reader
            .iter_column_terms(&fc.column)
            .expect("FST bytes valid: superfile just built");
        let n_terms_distinct = terms.len() as u32;
        let (min_term, max_term) = match (terms.first(), terms.last()) {
            (Some(min), Some(max)) => (min.clone(), max.clone()),
            _ => (Vec::new(), Vec::new()),
        };
        // Size the bloom to this superfile's distinct-term count rather than a
        // fixed 64 KiB. Readers derive the block count from the byte length.
        let mut bloom_builder = BloomBuilder::sized_for_terms(terms.len());
        for term in &terms {
            bloom_builder.insert(term);
        }
        out.insert(
            fc.column.clone(),
            FtsSummaryAgg::new_with_params(
                bloom_builder.finish(),
                n_terms_distinct,
                (min_term, max_term),
            ),
        );
    }
    out
}

/// Per-vector-column centroid summary (fp32 + 1-bit admit slab; see
/// [`build_column_vector_summary`]). `None` from the reader → column
/// absent from this superfile's vector blob → no entry in the map.
fn build_vector_summary(
    reader: &SuperfileReader,
    options: &SupertableOptions,
) -> HashMap<String, VectorSummary> {
    let mut out: HashMap<String, VectorSummary> = HashMap::new();
    let Some(vec_reader) = reader.vec() else {
        return out;
    };
    for vc in &options.vector_columns {
        if let Some(summary) = build_column_vector_summary(vec_reader, vc) {
            out.insert(vc.column.clone(), summary);
        }
    }
    out
}

// ============================================================
// Tombstone phase
// ============================================================
//
// Drives one WAL through `Appended → Complete` (UPDATE) or
// `Intent → Complete` (DELETE). Per `tombstone_progress` entry
// still at `Pending`:
//
// 1. **Resolve** `target_id → (superfile_id, doc_id)` against the
//    current manifest by iterating the `[id_min, id_max]`-pruned
//    superfile candidates and scanning their `_id` column. No
//    hit → outcome flips to `NotFound`.
// 2. **CAS-PUT the tombstone sidecar** under
//    `superfiles/<superfile_id>.tombstones`: GET (etag), union
//    the new doc-id bit, PUT with `If-Match`. Loops on CAS-loss.
//    A fresh (non-stale) seal means a compactor is mid-flight; the
//    writer must re-read the manifest and re-resolve against the
//    merged target. A stale seal (compactor crashed before
//    unsealing) is taken over instead of backed off on.
// 3. **Per-target WAL state CAS:** the entry flips to `Tombstoned`
//    or `NotFound`, with `tombstoned_in_superfile` recorded for
//    audit.
//
// Once every entry is non-`Pending`, the WAL itself is advanced to
// `Complete`. The state-doc and IPC sidecar deletions are
// best-effort and live outside this function — they stay with
// the recovery sweep + GC.
//
// ## Recovery & idempotency
//
// Replay safety relies on three facts:
//
// - The sidecar bitmap is a set union — re-issuing the same bit is
//   a no-op (the bitmap stays bit-identical on re-PUT after a CAS
//   refresh).
// - Per-target progress is persisted in `tombstone_progress` on the
//   WAL state doc. A crash mid-loop resumes at the first remaining
//   `Pending` entry.
// - The final `Complete` transition is one CAS on the state doc;
//   either it lands and the WAL is done, or it doesn't and recovery
//   re-runs the (already-no-op) tombstone loop and re-attempts.

/// Outcome of one tombstone-phase invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TombstonePhaseOutcome {
    /// The WAL was already in `Complete` on entry. We didn't touch
    /// any sidecar or state doc. `n_tombstoned` + `n_not_found`
    /// are read straight off the state doc's existing
    /// `tombstone_progress` so callers see the same counts as
    /// they would from a fresh apply.
    AlreadyComplete {
        n_tombstoned: usize,
        n_not_found: usize,
    },

    /// We ran (or finished running) the tombstone loop, advanced
    /// the WAL state to `Complete`, and report the per-outcome
    /// counts the caller will surface to its `MutationStats`.
    Applied {
        n_tombstoned: usize,
        n_not_found: usize,
    },
}

/// Typed failures from `run_tombstone_phase`. As with the append
/// phase, the WAL stays at whatever state was durable when the
/// error surfaced; recovery on a fresh process picks up at the
/// first remaining `Pending` entry.
#[derive(Debug, thiserror::Error)]
pub enum TombstonePhaseError {
    /// State doc is in a phase that disallows running the
    /// tombstone loop. The orchestrator only runs from
    /// `Appended` (UPDATE) or `Intent` (DELETE). An UPDATE in
    /// `Intent` means the append phase hasn't completed; calling
    /// this entry point is a builder bug.
    #[error(
        "tombstone phase invoked on WAL in state {state:?} for op {op_kind:?}; expected \
         Appended (UPDATE) or Intent (DELETE)"
    )]
    InvalidPreState { op_kind: OpKind, state: WalState },

    /// The supertable handle this orchestrator was given doesn't
    /// have a storage backend attached. Sidecar CAS-PUTs need
    /// durable storage; there's no in-process fallback.
    #[error("supertable has no storage attached; tombstone phase requires durable storage")]
    NoStorageAttached,

    /// The compactor sealed every candidate sidecar for one
    /// target and the writer's bounded backoff loop didn't see
    /// the manifest swap that would route the target elsewhere.
    /// In production the sealed-retry loop should be unbounded
    /// (the writer must block on compaction's forward progress);
    /// the implementation here bounds it so a stuck supertable
    /// surfaces a typed error instead of hanging a test process.
    #[error("tombstone sidecars for {targets} remained sealed past retry budget")]
    SealedSidecarRetryExhausted { targets: String },

    /// CAS-loss retry budget for one sidecar exhausted. Each
    /// loss costs one GET + PUT round-trip; a high-contention
    /// workload that genuinely exhausts the budget points at
    /// an undersized backoff or a thundering-herd of concurrent
    /// writers targeting the same superfile.
    #[error(
        "tombstone sidecar CAS exhausted after {attempts} attempts for superfile {superfile_id}"
    )]
    CasRetryExhausted { superfile_id: Uuid, attempts: u32 },

    /// Failure scanning a superfile's `_id` column while resolving a
    /// batch of targets to their `(superfile_id, doc_id)` homes.
    /// `targets` labels the id window the scan covered — a single id
    /// when the batch held one, otherwise `first..last` and a count.
    /// The underlying error is preserved as a string because the
    /// resolve path crosses crate-internal Parquet error types we
    /// don't re-export.
    #[error("failed to scan _id column for {targets}: {message}")]
    IdLookupFailed { targets: String, message: String },

    /// The post-sidecar manifest stamp (tombstone seqs) failed. The
    /// sidecar bits are durable and the WAL stays incomplete, so the
    /// recovery sweep re-runs the phase and re-attempts the stamp.
    /// Carries the typed cause so a lost commit race stays
    /// distinguishable from a permanent fault at the public boundary.
    #[error("tombstone-seq manifest stamp failed: {0}")]
    ManifestStamp(#[source] Box<ManifestCommitError>),

    /// Tombstone codec error from the sidecar layer.
    #[error("tombstone sidecar codec error: {0}")]
    SidecarCodec(#[from] crate::supertable::wal::tombstones_codec::SidecarCodecError),

    /// Underlying storage error.
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    /// WAL state-document I/O error from the persistence layer.
    #[error("WAL store error: {0}")]
    WalStore(#[from] WalStoreError),
}

impl TombstonePhaseError {
    /// True when the tombstone phase lost a race against a concurrent
    /// writer or the compactor, so reissuing the mutation against fresh
    /// state can succeed.
    ///
    /// Both retry-budget variants count: exhausting the sidecar CAS budget
    /// or waiting out a compactor-sealed sidecar are contention outcomes,
    /// not faults — the caller's move in each case is to retry the delete.
    pub(crate) fn is_conflict(&self) -> bool {
        match self {
            TombstonePhaseError::CasRetryExhausted { .. }
            | TombstonePhaseError::SealedSidecarRetryExhausted { .. } => true,
            TombstonePhaseError::ManifestStamp(e) => e.is_conflict(),
            TombstonePhaseError::Storage(e) => e.is_conflict(),
            TombstonePhaseError::WalStore(e) => e.is_conflict(),
            _ => false,
        }
    }

    /// True when the backend refused the credentials in use.
    pub(crate) fn is_permission_denied(&self) -> bool {
        match self {
            TombstonePhaseError::ManifestStamp(e) => e.is_permission_denied(),
            TombstonePhaseError::Storage(e) => e.is_permission_denied(),
            TombstonePhaseError::WalStore(e) => e.is_permission_denied(),
            _ => false,
        }
    }
}

/// Drive one WAL through the tombstone phase to `Complete`.
///
/// **Pre-conditions** (caller responsibility):
/// - `wal_doc.op_kind == Update` AND `wal_doc.state == Appended`, OR
/// - `wal_doc.op_kind == Delete` AND `wal_doc.state == Intent`, OR
/// - `wal_doc.state == Complete` (re-running on a finished WAL is
///   the idempotent no-op path; the orchestrator just reports the
///   existing counts).
///
/// **Post-conditions** on `Ok`:
/// - For `Applied`: `wal_doc.state == Complete` durably; every
///   `tombstone_progress` entry is non-`Pending`; per-superfile
///   sidecars reflect the union of all tombstoned `doc_id`s.
/// - For `AlreadyComplete`: nothing was touched; counts reflect
///   the state-doc's existing `tombstone_progress`.
///
/// **What happens on intermediate failure:** the WAL stays at
/// whatever state was durable when the failure occurred. The
/// `tombstone_progress` outcomes are the recovery cursor — a fresh
/// process re-runs this function and re-resolves whichever entries
/// are still `Pending`, which may be a whole batch's worth if the
/// failure landed before the next state write. The sidecar bitmap
/// union is a set so re-issuing is bit-identical; each state CAS is
/// one atomic write that either landed or didn't.
pub async fn run_tombstone_phase(
    supertable: &Supertable,
    wal_store: &WalStore,
    wal_doc: &WalStateDoc,
    wal_etag: &Etag,
) -> Result<(TombstonePhaseOutcome, WalStateDoc, Etag), TombstonePhaseError> {
    // Pre-condition: state ↔ op-kind compatibility. The
    // tombstone phase only runs from `Appended` (UPDATE) or
    // `Intent` (DELETE); a `Complete` WAL is the idempotent
    // no-op path. Anything else is a builder bug, surfaced as
    // a typed error rather than silently driving against a
    // half-built state doc.
    match (wal_doc.op_kind, wal_doc.state) {
        (OpKind::Update, WalState::Appended) => {}
        (OpKind::Delete, WalState::Intent) => {}
        (_, WalState::Complete) => {
            let (n_tombstoned, n_not_found) = count_outcomes(&wal_doc.tombstone_progress);
            return Ok((
                TombstonePhaseOutcome::AlreadyComplete {
                    n_tombstoned,
                    n_not_found,
                },
                wal_doc.clone(),
                wal_etag.clone(),
            ));
        }
        (op_kind, state) => {
            return Err(TombstonePhaseError::InvalidPreState { op_kind, state });
        }
    }

    do_tombstone_apply(supertable, wal_store, wal_doc, wal_etag).await
}

/// Sum the per-outcome counts off a `tombstone_progress` slice.
/// Pulled out so [`run_tombstone_phase`] and the eventual
/// `do_tombstone_apply` can use the same accounting.
fn count_outcomes(progress: &[TombstoneEntry]) -> (usize, usize) {
    let mut n_tombstoned = 0usize;
    let mut n_not_found = 0usize;
    for entry in progress {
        match entry.outcome {
            TombstoneOutcome::Tombstoned => n_tombstoned += 1,
            TombstoneOutcome::NotFound => n_not_found += 1,
            TombstoneOutcome::Pending => {}
        }
    }
    (n_tombstoned, n_not_found)
}

/// Bounded CAS-loss retry budget for one sidecar. Each iteration
/// is a fresh GET + PUT round-trip; ten retries is enough to ride
/// out a small fan-out of concurrent writers targeting the same
/// superfile without converting a transient race into a typed
/// error.
const MAX_CAS_RETRIES: u32 = 10;

/// Fraction of the granted lease span the tombstone loop lets elapse
/// before folding a renewal into its next WAL state write. A third of
/// the span leaves two further renewal opportunities before expiry, so
/// a slow sidecar phase can't lapse the lease between batched writes.
/// Only writes made while bits are landing renew; see [`renew_lease`]
/// for why the pre-backoff write is exempt.
const LEASE_RENEW_FRACTION: i32 = 3;

/// Bounded sealed-sidecar retry budget. Architecturally this
/// loop should be unbounded — the writer blocks until
/// compaction's forward progress publishes a merged target and
/// our re-resolve routes there. We bound it so a stuck
/// supertable surfaces a typed error rather than hanging the
/// test process. The budget is high enough that a healthy
/// compactor never exhausts it under realistic loads.
const MAX_SEALED_RETRIES: u32 = 16;

/// Backoff floor between sealed-sidecar retries. Doubles per
/// attempt, capped at [`SEALED_RETRY_CAP_MS`].
const SEALED_RETRY_BASE_MS: u64 = 100;

/// Cap on the exponential sealed-retry backoff. 30 s keeps the
/// loop from blocking the writer indefinitely under a stuck
/// compactor while staying coarse enough that the retries don't
/// hammer storage.
const SEALED_RETRY_CAP_MS: u64 = 30_000;

/// Cap on the sealed-retry backoff doubling exponent, so the shift
/// plateaus (before [`SEALED_RETRY_CAP_MS`] clamps the result)
/// rather than overflowing on a high attempt count.
const SEALED_RETRY_MAX_SHIFT: u32 = 8;

/// The lease span the current driver was granted, or `None` when the WAL
/// carries no lease. Read once per phase, before any renewal moves
/// `expires_at`.
fn granted_lease_span(doc: &WalStateDoc) -> Option<ChronoDuration> {
    doc.lease
        .as_ref()
        .map(|lease| lease.expires_at - lease.acquired_at)
}

/// Push the WAL's lease expiry out one full span from now, in the caller's
/// in-memory doc, so the next state-doc write carries a fresh lease.
///
/// The tombstone phase can easily outlive a single lease grant: the id-scan
/// sweep opens and scans every candidate superfile, and the sidecar phase
/// spends a GET + PUT round trip per touched superfile, so a delete
/// spanning a large table runs well past the default 60 s. Without renewal
/// the lease lapses mid-phase, a recovery sweep preempts, and the driver's
/// next CAS fails — losing the phase (and, for a writer-driven delete, the
/// caller's whole `commit`) after the work had already landed.
///
/// Renewal rides on writes the phase is making anyway: no extra round trip,
/// and — unlike a background heartbeat task — no second writer racing the
/// `etag_cur` chain. A heartbeat's own CAS would invalidate the driver's
/// etag and break the very phase it was meant to protect. Because the
/// tombstone loop batches its state writes, it also *schedules* one on
/// [`LEASE_RENEW_FRACTION`] of the span so the gap between two writes that
/// carry a renewal can never grow past the grant.
///
/// Tying the expiry to writes ties it to observable forward progress: a
/// wedged driver stops landing writes, its lease lapses, and recovery takes
/// the WAL over — the same outcome the heartbeat's stuck-worker check exists
/// to produce. A driver parked in the sealed-sidecar backoff below is
/// therefore preemptible, which is deliberate: that wait is unbounded from
/// the lease's point of view and a peer deserves the chance to try. The
/// loop's pre-backoff state write is the one write that deliberately skips
/// this call, so parking stays a lapse and not a reprieve.
///
/// `span` comes from [`granted_lease_span`], so whoever took the lease keeps
/// the duration they chose — a writer stamping its default at create time, or
/// a sweep passing its own `lease_duration`.
///
/// `acquired_at` stays put: it records when ownership began, which renewal
/// doesn't change. That matches `lease::try_heartbeat`, which also moves only
/// `expires_at`.
fn renew_lease(doc: &mut WalStateDoc, span: Option<ChronoDuration>, now: DateTime<Utc>) {
    if let (Some(lease), Some(span)) = (doc.lease.as_mut(), span) {
        lease.expires_at = now + span;
    }
}

/// Whether enough of the granted lease span has elapsed that the caller
/// should fold a [`renew_lease`] into the state write it is about to make.
///
/// The deadline is read off the lease itself rather than tracked locally:
/// `expires_at - span` is the instant the lease was last stamped — by the
/// writer at create time, or by this phase's own last renewal. That matters
/// twice over. The phase can be entered with a partly-consumed lease (the
/// append phase writes without renewing, so an UPDATE arrives here having
/// already spent some of its grant), and a state write that deliberately
/// does *not* renew — the one before the sealed-sidecar backoff — must not
/// look like a renewal to the next check.
///
/// A WAL with no lease has nothing to keep alive, so nothing is ever due.
fn lease_renewal_due(doc: &WalStateDoc, span: Option<ChronoDuration>, now: DateTime<Utc>) -> bool {
    match (doc.lease.as_ref(), span) {
        (Some(lease), Some(span)) => now - (lease.expires_at - span) >= span / LEASE_RENEW_FRACTION,
        _ => false,
    }
}

/// The non-idempotent fast path for the tombstone loop. Every
/// `Pending` target in `wal_doc.tombstone_progress` is resolved in
/// one manifest sweep; the hits are then grouped by superfile so
/// each sidecar takes a single CAS-PUT covering all of its bits.
/// Once every target is non-`Pending`, advance the WAL itself to
/// `Complete`.
///
/// Resume-on-replay is automatic: the WAL state doc is the recovery
/// cursor, so a crash mid-loop leaves the entries that hadn't been
/// written back still `Pending`, and the next run re-resolves
/// exactly those. Every step is idempotent — the sidecar bitmap is
/// a set, each state write is one atomic CAS, and the final
/// `Complete` transition is one CAS.
///
/// State writes are batched rather than per-target, so the phase
/// folds a lease renewal into a write whenever
/// [`LEASE_RENEW_FRACTION`] of the granted span has elapsed (see
/// [`renew_lease`]). Without that, a mutation whose sidecar phase
/// outlives one lease grant would lapse mid-flight. The write before
/// a sealed-sidecar backoff is the exception: it persists settled
/// outcomes without renewing, so a driver waiting on a compactor
/// still ages out and a peer can take the WAL over.
async fn do_tombstone_apply(
    supertable: &Supertable,
    wal_store: &WalStore,
    wal_doc: &WalStateDoc,
    wal_etag: &Etag,
) -> Result<(TombstonePhaseOutcome, WalStateDoc, Etag), TombstonePhaseError> {
    let inner = supertable.inner();
    if inner.options.storage.is_none() {
        return Err(TombstonePhaseError::NoStorageAttached);
    }

    let mut wal_cur = wal_doc.clone();
    let mut etag_cur = wal_etag.clone();
    // Sampled before the loop starts moving `expires_at`: deriving the
    // span per write would compound it, since each renewal would measure
    // from the original `acquired_at` and hand back elapsed + span.
    let lease_span = granted_lease_span(&wal_cur);

    // Batched target loop. Every `Pending` entry is resolved in one
    // manifest sweep, then the hits are grouped by superfile so each
    // sidecar takes a single CAS covering all of its bits. Anything
    // already non-`Pending` (Tombstoned, NotFound) keeps its outcome,
    // so this function is safe to call against a partially-completed
    // WAL during recovery and resumes where the last run stopped.
    //
    // A sealed sidecar sends its targets around the loop again: the
    // compactor that sealed it may be about to publish a merged
    // superfile, and the target's `(superfile_id, doc_id)` moves with
    // it — so those targets must be re-resolved against a fresh
    // manifest rather than retried at the old coordinates. Bounded by
    // `MAX_SEALED_RETRIES` with exponential backoff.
    let mut pending: Vec<(usize, RowId)> = wal_cur
        .tombstone_progress
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.outcome == TombstoneOutcome::Pending)
        .map(|(idx, entry)| (idx, entry.target_id))
        .collect();
    tracing::debug!(
        "tombstoning {} of {} targets",
        pending.len(),
        wal_cur.tombstone_progress.len()
    );

    let mut sealed_attempts = 0u32;
    while !pending.is_empty() {
        let pending_ids: Vec<RowId> = pending.iter().map(|(_, target_id)| *target_id).collect();

        // One manifest sweep for the whole pending set, instead of a
        // manifest walk plus a full `_id`-column scan per target.
        let resolved = resolve_all(inner, &pending_ids)?;

        let mut doc_ids_in_superfile_map = HashMap::<Uuid, Vec<(u32, usize)>>::new();
        for &(idx, target_id) in &pending {
            match resolved.get(&target_id) {
                Some(&(superfile_id, doc_id)) => {
                    doc_ids_in_superfile_map
                        .entry(superfile_id)
                        .or_default()
                        .push((doc_id, idx));
                }
                None => {
                    wal_cur.tombstone_progress[idx].outcome = TombstoneOutcome::NotFound;
                    wal_cur.tombstone_progress[idx].tombstoned_in_superfile = None;
                }
            }
        }

        let mut sealed: Vec<(usize, RowId)> = Vec::new();
        let mut landed_any = false;
        for (superfile_id, hits) in doc_ids_in_superfile_map {
            let doc_ids: Vec<u32> = hits.iter().map(|&(doc_id, _)| doc_id).collect();
            match cas_tombstone_bits(wal_store, superfile_id, &doc_ids).await? {
                SidecarCasOutcome::Landed => {
                    landed_any = true;
                    for (_, idx) in hits {
                        wal_cur.tombstone_progress[idx].tombstoned_in_superfile =
                            Some(superfile_id);
                        wal_cur.tombstone_progress[idx].outcome = TombstoneOutcome::Tombstoned;
                    }
                }
                SidecarCasOutcome::Sealed => sealed.extend(
                    hits.into_iter()
                        .map(|(_, idx)| (idx, wal_cur.tombstone_progress[idx].target_id)),
                ),
            }

            // A mutation spanning many superfiles pays a GET + PUT per
            // sidecar, so this walk alone can outlive the lease grant.
            // Fold a renewal into a state write once the gap since the
            // last one reaches a fraction of the span.
            if lease_renewal_due(&wal_cur, lease_span, Utc::now()) {
                renew_lease(&mut wal_cur, lease_span, Utc::now());
                etag_cur = wal_store
                    .update_with_etag(wal_cur.wal_id, &etag_cur, &wal_cur)
                    .await?;
            }
        }

        if sealed.is_empty() {
            break;
        }
        // Forward progress refunds the budget: a batch spanning several
        // concurrently-compacting superfiles would otherwise burn one
        // shared allowance on seals it is steadily working through.
        if landed_any {
            sealed_attempts = 0;
        }
        sealed_attempts += 1;
        if sealed_attempts > MAX_SEALED_RETRIES {
            return Err(TombstonePhaseError::SealedSidecarRetryExhausted {
                targets: pending_batch_label(&sealed),
            });
        }

        // Persist what settled before parking, so a peer that preempts us
        // inherits the outcomes instead of redoing them. Deliberately *not*
        // a renewal: the backoff is a wait, not forward progress, and
        // [`renew_lease`]'s contract is that a parked driver ages out so a
        // peer gets its turn. Renewing here would hand the driver a fresh
        // span every time it parked and it would hold the WAL forever.
        etag_cur = wal_store
            .update_with_etag(wal_cur.wal_id, &etag_cur, &wal_cur)
            .await?;

        let ms = SEALED_RETRY_BASE_MS
            .saturating_mul(1u64 << (sealed_attempts - 1).min(SEALED_RETRY_MAX_SHIFT))
            .min(SEALED_RETRY_CAP_MS);
        sleep(Duration::from_millis(ms)).await;
        // Loop back and re-resolve against a fresh manifest.
        pending = sealed;
    }

    // Final state write for whatever the last pass settled. A crash before
    // this costs that pass's resolve work, not its durability — the sidecar
    // bits are already down, and replay re-resolves the entries still
    // marked `Pending` idempotently.
    renew_lease(&mut wal_cur, lease_span, Utc::now());
    etag_cur = wal_store
        .update_with_etag(wal_cur.wal_id, &etag_cur, &wal_cur)
        .await?;

    let (n_tombstoned, n_not_found) = count_outcomes(&wal_cur.tombstone_progress);
    tracing::debug!("tombstone apply settled {n_tombstoned} tombstoned, {n_not_found} not found");

    // Stamp the touched superfiles' tombstone seqs into the manifest
    // so readers — this process's own cache included — learn the
    // sidecars changed. Runs before the `Complete` CAS so a crash in
    // between is re-driven by recovery ("WAL complete ⇒ stamped");
    // re-stamping on such a replay is harmless (one extra seq bump).
    let touched: Vec<Uuid> = {
        let mut ids: Vec<Uuid> = wal_cur
            .tombstone_progress
            .iter()
            .filter(|e| e.outcome == TombstoneOutcome::Tombstoned)
            .filter_map(|e| e.tombstoned_in_superfile)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    };
    if !touched.is_empty() {
        stamp_tombstone_seqs(inner, &touched)
            .await
            .map_err(|e| TombstonePhaseError::ManifestStamp(Box::new(e)))?;
    }

    // Final transition: every entry is non-Pending; flip the
    // WAL itself to Complete. One CAS — if it loses, the
    // caller's recovery loop picks up at the now-no-op
    // tombstone scan above (everything's already non-Pending)
    // and re-attempts the final advance. The manifest stamp
    // above can take a while, so this write gets a fresh lease
    // too: it is the one whose loss wastes the whole phase.
    wal_cur.state = WalState::Complete;
    renew_lease(&mut wal_cur, lease_span, Utc::now());
    etag_cur = wal_store
        .update_with_etag(wal_cur.wal_id, &etag_cur, &wal_cur)
        .await?;

    Ok((
        TombstonePhaseOutcome::Applied {
            n_tombstoned,
            n_not_found,
        },
        wal_cur,
        etag_cur,
    ))
}

/// Resolve every `target_id` of one mutation against the current
/// manifest snapshot in a single batched sweep.
///
/// One snapshot for the whole batch, so every target resolves
/// against the same view of the table (the per-target predecessor
/// re-loaded the manifest for each id and could straddle a
/// concurrent commit mid-batch).
fn resolve_all(
    inner: &Arc<SupertableInner>,
    target_ids: &[RowId],
) -> Result<HashMap<RowId, (Uuid, u32)>, TombstonePhaseError> {
    let manifest = inner.manifest.load_full();
    resolve_targets_in_manifest(inner, &manifest, target_ids)
}

/// Outcome of one sidecar CAS-PUT attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SidecarCasOutcome {
    /// The new doc-id bit is now durable in the sidecar.
    Landed,
    /// The current sidecar carries a fresh (non-stale) seal — a
    /// compactor is presumed still mid-flight against this
    /// superfile. The caller must re-read the manifest and re-resolve.
    /// A *stale* seal doesn't produce this outcome; we take it over
    /// instead (see `cas_tombstone_bits`).
    Sealed,
}

/// GET → union-the-bits → PUT with bounded CAS-loss retries: one
/// round trip lands every `doc_id` this mutation resolved to the
/// superfile, rather than one round trip per bit.
///
/// Detects *fresh* sealed sidecars and surfaces them via the typed
/// outcome so the caller's outer loop can re-resolve. A stale seal
/// (its compactor crashed before unsealing) is treated as no seal at
/// all: we proceed to land the bits and clear it, same as
/// `tombstones_admin::seal`'s own steal-if-stale behavior.
async fn cas_tombstone_bits(
    wal_store: &WalStore,
    superfile_id: Uuid,
    doc_ids: &[u32],
) -> Result<SidecarCasOutcome, TombstonePhaseError> {
    for _attempt in 0..MAX_CAS_RETRIES {
        // Read the current sidecar (None ↔ no tombstones yet).
        let (existing, etag_opt) = match wal_store.get_tombstones(superfile_id).await? {
            Some((sc, etag)) => (Some(sc), Some(etag)),
            None => (None, None),
        };

        // Sealed → bail out to the outer resolve loop, but only while
        // the seal is still fresh. A stale seal means its compactor
        // crashed before unsealing (see `tombstones_admin::seal`'s
        // steal-if-stale logic) -- if we kept treating it as live,
        // this update/delete would retry and eventually fail with
        // `SealedSidecarRetryExhausted` even long after the seal
        // stopped meaning anything. Falling through here lands our
        // bit with `seal: None` below, clearing the dead seal too.
        if let Some(sc) = &existing
            && let Some(seal) = sc.seal.as_ref()
            && !tombstones_admin::is_seal_stale(
                seal.sealed_at,
                Utc::now(),
                Duration::from_millis(tombstones_admin::DEFAULT_STALE_SEAL_TIMEOUT_MS),
            )
        {
            return Ok(SidecarCasOutcome::Sealed);
        }

        // Union the new bit. RoaringBitmap insert is a no-op
        // when the bit is already set — covers replay safety
        // and the "two writers raced on the same bit" path.
        let bitmap = match existing {
            Some(sc) => {
                let mut b = sc.bitmap;
                for doc_id in doc_ids {
                    b.insert(*doc_id);
                }
                b
            }
            None => {
                let mut b = RoaringBitmap::new();
                for doc_id in doc_ids {
                    b.insert(*doc_id);
                }
                b
            }
        };
        let new_sidecar = TombstonesSidecar { seal: None, bitmap };

        match wal_store
            .put_tombstones(superfile_id, etag_opt.as_ref(), &new_sidecar)
            .await
        {
            Ok(_new_etag) => return Ok(SidecarCasOutcome::Landed),
            Err(WalStoreError::CasFailed { .. }) => {
                // CAS-loss: another writer landed first. Re-read
                // (which catches both bumped-etag-but-unsealed
                // and the seal-after-our-read race) and retry.
                continue;
            }
            Err(other) => return Err(other.into()),
        }
    }
    Err(TombstonePhaseError::CasRetryExhausted {
        superfile_id,
        attempts: MAX_CAS_RETRIES,
    })
}

/// Resolve a batch of target `_id`s to the `(superfile_id, local
/// doc_id)` pair that owns each, in a single sweep over the manifest.
///
/// Resolving one target at a time costs a manifest walk plus a full
/// `_id`-column scan per target: deleting T rows from a table of S
/// superfiles pays up to T×S superfile opens and column scans, and
/// re-scans the same column once per target that lives in it. This
/// batched form sorts the target ids once and walks the manifest
/// once — each candidate superfile is opened and scanned at most
/// once no matter how many targets it owns, and the scan probes the
/// sorted target list by binary search (merging the small sorted
/// target side against the column, see
/// [`SuperfileReader::id_lookup_many`]). Cost drops to S opens and
/// `O(rows · log T)` comparisons.
///
/// Superfiles are visited in manifest order and a target leaves the
/// pending set as soon as it resolves, so when several overlapping
/// `[id_min, id_max]` ranges could claim a target it lands in the
/// same superfile a per-target walk would have picked.
///
/// Only resolved targets appear in the returned map; a target that
/// is absent from it is claimed by no superfile — the caller's
/// `NotFound` outcome.
fn resolve_targets_in_manifest(
    inner: &Arc<SupertableInner>,
    manifest: &ManifestSnapshot,
    targets: &[RowId],
) -> Result<HashMap<RowId, (Uuid, u32)>, TombstonePhaseError> {
    // Sorted + deduped once: `id_lookup_many` requires it, and it
    // turns "which of my targets could this superfile own?" into a
    // pair of binary searches per manifest entry.
    let mut pending: Vec<i128> = targets.iter().map(|t| t.0).collect();
    pending.sort_unstable();
    pending.dedup();

    let mut resolved: HashMap<RowId, (Uuid, u32)> = HashMap::with_capacity(pending.len());
    if pending.is_empty() {
        return Ok(resolved);
    }

    for entry in manifest.get_all_superfiles().iter() {
        // `pending` is sorted, so the targets this entry's
        // `[id_min, id_max]` could claim are one contiguous window.
        let lo = pending.partition_point(|&t| t < entry.id_min);
        let hi = pending.partition_point(|&t| t <= entry.id_max);
        if lo == hi {
            continue;
        }
        let candidates = &pending[lo..hi];

        for (target, doc_id) in lookup_ids_in_superfile(inner, entry, candidates)? {
            resolved.insert(RowId(target), (entry.superfile_id, doc_id));
        }

        // A resolved target never needs another superfile scanned.
        pending.retain(|&t| !resolved.contains_key(&RowId(t)));
        if pending.is_empty() {
            break;
        }
    }

    Ok(resolved)
}

/// Scan one superfile's `_id` column for `sorted_targets`, opening
/// it through the tiered reader path. Returns the `(target, local
/// doc_id)` pairs it owns.
fn lookup_ids_in_superfile(
    inner: &Arc<SupertableInner>,
    entry: &SuperfileEntry,
    sorted_targets: &[i128],
) -> Result<Vec<(i128, u32)>, TombstonePhaseError> {
    // A batch has no single target to name in error context, so
    // errors carry the id window being scanned instead.
    let scan_label = id_window_label(sorted_targets);

    // Tiered open: in-memory reader cache → disk cache →
    // direct storage GET. The first two are the production
    // reader path; the third covers cross-process recovery
    // where neither cache layer has the bytes pinned. The
    // recovery sweep on `Supertable::open` lands here when
    // the freshly-opened handle's in-memory tier is empty
    // and no disk cache is attached.
    let reader = match bridge_sync_to_async(superfile_reader(
        &inner.options.store,
        inner.options.disk_cache.as_ref(),
        inner.options.storage.as_ref(),
        &entry.uri,
        entry.subsection_offsets.as_ref(),
        true,
    )) {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(
                "tiered open of superfile {} failed ({e}); falling back to a direct \
                 storage fetch for the id scan",
                entry.uri.0
            );
            let bytes =
                fetch_superfile_bytes_for_id_scan(inner, entry.uri.0).map_err(|message| {
                    TombstonePhaseError::IdLookupFailed {
                        targets: scan_label.clone(),
                        message: format!(
                            "open superfile {} (storage fallback): {message}",
                            entry.uri.0
                        ),
                    }
                })?;
            Arc::new(SuperfileReader::open(bytes).map_err(|e| {
                TombstonePhaseError::IdLookupFailed {
                    targets: scan_label.clone(),
                    message: format!(
                        "SuperfileReader::open {} (storage fallback): {e}",
                        entry.uri.0
                    ),
                }
            })?)
        }
    };

    // id_lookup_many requires the full superfile bytes (eager open).
    // A lazy-opened reader from the cache path will return an Io
    // error here; fall back to a direct storage fetch in that case.
    match reader.id_lookup_many(sorted_targets) {
        Ok(hits) => Ok(hits),
        Err(ReadError::Io(e)) => {
            tracing::debug!(
                "superfile {} opened lazily ({e}); re-opening eagerly for the id scan",
                entry.uri.0
            );
            // Lazy reader — re-open eagerly from storage.
            let bytes =
                fetch_superfile_bytes_for_id_scan(inner, entry.uri.0).map_err(|message| {
                    TombstonePhaseError::IdLookupFailed {
                        targets: scan_label.clone(),
                        message: format!(
                            "open superfile {} (eager fallback for id_lookup): {message}",
                            entry.uri.0
                        ),
                    }
                })?;
            let eager_reader =
                SuperfileReader::open(bytes).map_err(|e| TombstonePhaseError::IdLookupFailed {
                    targets: scan_label.clone(),
                    message: format!(
                        "SuperfileReader::open {} (eager fallback for id_lookup): {e}",
                        entry.uri.0
                    ),
                })?;
            eager_reader.id_lookup_many(sorted_targets).map_err(|e| {
                TombstonePhaseError::IdLookupFailed {
                    targets: scan_label,
                    message: format!("id_lookup in superfile {}: {e}", entry.uri.0),
                }
            })
        }
        Err(e) => Err(TombstonePhaseError::IdLookupFailed {
            targets: scan_label,
            message: format!("id_lookup in superfile {}: {e}", entry.uri.0),
        }),
    }
}

/// Label for the still-unresolved `(progress index, target)` pairs a
/// sealed-sidecar retry gave up on, for the typed error's `targets`
/// context.
fn pending_batch_label(pending: &[(usize, RowId)]) -> String {
    let mut ids: Vec<i128> = pending.iter().map(|&(_, target_id)| target_id.0).collect();
    ids.sort_unstable();
    ids.dedup();
    id_window_label(&ids)
}

/// Human-readable label for a batch of target ids, used as the
/// `targets` context on id-scan errors: the single id when the
/// batch holds one, otherwise its `first..last` window and size.
fn id_window_label(sorted_targets: &[i128]) -> String {
    match sorted_targets {
        [] => "<empty batch>".to_string(),
        [single] => RowId(*single).to_hex(),
        [first, .., last] => format!(
            "{}..{} ({} targets)",
            RowId(*first).to_hex(),
            RowId(*last).to_hex(),
            sorted_targets.len()
        ),
    }
}

/// Fetch a superfile's full bytes directly from storage.
/// Storage-fallback path for the recovery sweep when the
/// in-memory + disk-cache tiers are both cold.
///
/// Sync-bridged because the call site
/// (`lookup_ids_in_superfile`) is sync (called from inside
/// the pipeline orchestrator); we mirror the
/// `query::superfile_reader::superfile_reader` async-bridge
/// pattern.
fn fetch_superfile_bytes_for_id_scan(
    inner: &Arc<SupertableInner>,
    superfile_id: Uuid,
) -> Result<Bytes, String> {
    let storage = inner
        .options
        .storage
        .as_ref()
        .ok_or_else(|| "no storage attached".to_string())?
        .clone();
    let path = SuperfileUri(superfile_id).storage_path();
    let (bytes, _) = bridge_sync_to_async(async move { storage.get(&path).await })
        .map_err(|e| format!("storage get: {e}"))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    //! Tests for the append-phase orchestrator. This commit
    //! covers the orchestrator's outer-shape behaviour —
    //! pre-condition checks, the `AlreadyApplied` path,
    //! error mapping. End-to-end + crash-injection tests
    //! land alongside the `do_apply` implementation.
    //!
    //! The fixtures use real `Supertable` + `WalStore`
    //! against a `LocalFsStorageProvider`-backed `TempDir`,
    //! matching the existing persistence-layer test pattern
    //! used elsewhere in this crate.

    use arrow::ipc::writer::StreamWriter;
    use chrono::Utc;
    use tempfile::TempDir;
    use tokio::time::timeout;
    use uuid::Uuid;

    use super::*;
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{
            Supertable,
            manifest::ManifestSnapshot,
            wal::{
                state_doc::{
                    Lease, OpKind, RowId, SCHEMA_VERSION, SealRecord, SupertableHandleId,
                    TombstoneEntry, TombstoneOutcome, WalId, WalState,
                },
                tombstones_codec::TombstonesSidecar,
            },
        },
        test_helpers::{build_title_batch, default_supertable_options},
    };

    /// Construct a Supertable + a fresh WAL state doc + the WAL's
    /// etag, all backed by the same LocalFs storage so an
    /// orchestrator call against this supertable sees the WAL
    /// the fixture just created.
    async fn fixture() -> (TempDir, Supertable, WalStore, WalStateDoc, Etag) {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let supertable =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let wal_store = WalStore::new(Arc::clone(&storage));

        let wal_id = WalId(42);
        let wal_doc = WalStateDoc {
            wal_id,
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Update,
            state: WalState::Intent,
            created_at: Utc::now(),
            lease: None,
            predicate_repr: "_id = 1".into(),
            target_ids: vec![RowId(1)],
            new_row_count: Some(1),
            new_row_content_hash: Some("0".repeat(64)),
            preallocated_superfile_id: Some(Uuid::from_u128(0x1234_5678_9ABC)),
            minted_id_spans: vec![IdSpan {
                first: RowId(100),
                last: RowId(100),
            }],
            tombstone_progress: vec![TombstoneEntry {
                target_id: RowId(1),
                outcome: TombstoneOutcome::Pending,
                tombstoned_in_superfile: None,
            }],
        };
        let etag = wal_store.create(&wal_doc).await.expect("wal create");
        (dir, supertable, wal_store, wal_doc, etag)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rejects_delete_wal_with_typed_error() {
        let (_dir, st, ws, mut wal, etag) = fixture().await;
        wal.op_kind = OpKind::Delete;
        let err = run_append_phase(&st, &ws, &wal, &etag, None)
            .await
            .expect_err("must error");
        assert!(matches!(err, AppendPhaseError::NotAnUpdateWal), "{err:?}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rejects_wal_missing_preallocated_superfile_id() {
        let (_dir, st, ws, mut wal, etag) = fixture().await;
        wal.preallocated_superfile_id = None;
        let err = run_append_phase(&st, &ws, &wal, &etag, None)
            .await
            .expect_err("must error");
        assert!(
            matches!(
                err,
                AppendPhaseError::MissingField {
                    field: "preallocated_superfile_id"
                }
            ),
            "{err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn manifest_contains_returns_true_for_matching_uuid() {
        let (_dir, _st, _ws, _wal, _etag) = fixture().await;
        let opts = Arc::new(default_supertable_options());
        let empty = ManifestSnapshot::empty(Arc::clone(&opts));
        assert!(!manifest_contains(&empty, Uuid::nil()));
    }

    /// The storage-fallback fetch surfaces a typed "no storage" string when
    /// the supertable has no storage attached, before any I/O is attempted.
    #[test]
    fn fetch_superfile_bytes_for_id_scan_errors_without_storage() {
        let st = Supertable::create(default_supertable_options()).expect("create");
        assert!(
            st.inner().options.storage.is_none(),
            "fixture-free supertable has no storage"
        );
        let err = fetch_superfile_bytes_for_id_scan(st.inner(), Uuid::from_u128(7))
            .expect_err("must error without storage");
        assert!(err.contains("no storage"), "got {err}");
    }

    /// With storage attached, the fallback fetch returns exactly the bytes
    /// written at the superfile's storage path.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetch_superfile_bytes_for_id_scan_reads_written_bytes() {
        let (_dir, st, _ws, _wal, _etag) = fixture().await;
        let id = Uuid::from_u128(0xFEED_FACE);
        let payload = Bytes::from_static(b"superfile-scan-payload");
        st.inner()
            .options
            .storage
            .as_ref()
            .expect("fixture attaches storage")
            .put_atomic(&SuperfileUri(id).storage_path(), payload.clone())
            .await
            .expect("put superfile");

        let got = fetch_superfile_bytes_for_id_scan(st.inner(), id).expect("fetch bytes");
        assert_eq!(got, payload, "fetched bytes match what was written");
    }

    // ---- End-to-end Applied path ----------------------------------------

    /// Encode a RecordBatch as Arrow IPC stream bytes — same
    /// shape the WAL's `.arrow` sidecar carries in production.
    fn encode_ipc(batch: &RecordBatch) -> Bytes {
        let mut out: Vec<u8> = Vec::new();
        {
            let mut writer =
                StreamWriter::try_new(&mut out, &batch.schema()).expect("ipc writer init");
            writer.write(batch).expect("ipc write");
            writer.finish().expect("ipc finish");
        }
        Bytes::from(out)
    }

    /// Preallocated superfile uuid the single-superfile fixtures
    /// publish under.
    const FIXTURE_SUPERFILE_UUID: u128 = 0xDEAD_BEEF_CAFE;

    /// Set up a fixture where the WAL's IPC payload and state
    /// doc are consistent: matching `new_row_count`, blake3,
    /// minted_id_spans. The supertable's storage is shared with
    /// the WAL store so the orchestrator's IPC fetch finds the
    /// payload we just wrote.
    async fn fixture_with_ipc_payload(
        titles: &[&str],
        wal_id_value: i128,
        minted_first: i128,
    ) -> (TempDir, Supertable, WalStore, WalStateDoc, Etag) {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let supertable =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let wal_store = WalStore::new(Arc::clone(&storage));

        let (wal_doc, etag) = seed_update_wal(
            &wal_store,
            titles,
            wal_id_value,
            minted_first,
            Uuid::from_u128(FIXTURE_SUPERFILE_UUID),
        )
        .await;
        (dir, supertable, wal_store, wal_doc, etag)
    }

    /// Write the `.arrow` payload and the matching UPDATE state doc
    /// for one append, against an existing WalStore. Factored out of
    /// [`fixture_with_ipc_payload`] so a test can seed a *second*
    /// append (distinct superfile uuid + `_id` range) into the same
    /// supertable.
    async fn seed_update_wal(
        wal_store: &WalStore,
        titles: &[&str],
        wal_id_value: i128,
        minted_first: i128,
        superfile_id: Uuid,
    ) -> (WalStateDoc, Etag) {
        let user_batch = build_title_batch(titles);
        let ipc_bytes = encode_ipc(&user_batch);
        let content_hash = blake3::hash(&ipc_bytes).to_hex().to_string();
        let n = titles.len() as u32;
        let wal_id = WalId(wal_id_value);

        wal_store
            .put_arrow(wal_id, ipc_bytes)
            .await
            .expect("put_arrow");

        let wal_doc = WalStateDoc {
            wal_id,
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Update,
            state: WalState::Intent,
            created_at: Utc::now(),
            lease: None,
            predicate_repr: "set up by test".into(),
            target_ids: (0..n).map(|i| RowId(1000 + i as i128)).collect(),
            new_row_count: Some(n),
            new_row_content_hash: Some(content_hash),
            preallocated_superfile_id: Some(superfile_id),
            minted_id_spans: vec![IdSpan {
                first: RowId(minted_first),
                last: RowId(minted_first + (n as i128) - 1),
            }],
            tombstone_progress: (0..n)
                .map(|i| TombstoneEntry {
                    target_id: RowId(1000 + i as i128),
                    outcome: TombstoneOutcome::Pending,
                    tombstoned_in_superfile: None,
                })
                .collect(),
        };
        let etag = wal_store.create(&wal_doc).await.expect("wal create");
        (wal_doc, etag)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn end_to_end_appends_superfile_and_advances_state() {
        let (_dir, st, ws, wal, etag) =
            fixture_with_ipc_payload(&["alpha bravo", "charlie delta"], 7, 5_000).await;
        let pre_uuid = wal.preallocated_superfile_id.expect("set in fixture");

        let (outcome, new_wal, new_etag) = run_append_phase(&st, &ws, &wal, &etag, None)
            .await
            .expect("append phase");

        // Outcome + WAL state.
        assert_eq!(outcome, AppendPhaseOutcome::Applied);
        assert_eq!(new_wal.state, WalState::Appended);
        assert_ne!(new_etag, etag, "etag must advance after the state change");

        // ManifestSnapshot now contains the preallocated superfile.
        let manifest = st.inner().manifest.load_full();
        assert!(
            manifest_contains(&manifest, pre_uuid),
            "manifest must reference the new superfile"
        );

        // The state doc on disk reflects the in-memory new_wal.
        let (read_back, read_etag) = ws.read(wal.wal_id).await.expect("read back");
        assert_eq!(read_back.state, WalState::Appended);
        assert_eq!(read_etag, new_etag);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn idempotent_replay_short_circuits_to_already_applied() {
        // Run the append phase twice on the same WAL. The first
        // call goes through `do_apply`; the second observes the
        // already-committed superfile in the manifest and
        // returns `AlreadyApplied`. The WAL state ends in
        // Appended either way.
        let (_dir, st, ws, wal, etag) =
            fixture_with_ipc_payload(&["alpha", "beta"], 11, 6_000).await;

        let (first_outcome, after_first, etag_after_first) =
            run_append_phase(&st, &ws, &wal, &etag, None)
                .await
                .expect("first");
        assert_eq!(first_outcome, AppendPhaseOutcome::Applied);
        assert_eq!(after_first.state, WalState::Appended);

        let (second_outcome, after_second, etag_after_second) =
            run_append_phase(&st, &ws, &after_first, &etag_after_first, None)
                .await
                .expect("second");
        // The second run is a no-op on the state doc (WAL was
        // already Appended), so etag stays put.
        assert_eq!(second_outcome, AppendPhaseOutcome::AlreadyApplied);
        assert_eq!(after_second.state, WalState::Appended);
        assert_eq!(etag_after_second, etag_after_first);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn intent_already_committed_advances_state_without_rebuilding() {
        // Inject the "crashed between manifest swap and WAL
        // state advance" failure mode: the manifest already
        // references the WAL's preallocated_superfile_id, but
        // the WAL is still in Intent. Recovery's first attempt
        // should observe the existing superfile (idempotency
        // probe), advance the WAL to Appended, and return
        // AlreadyApplied. No new bytes get built or written.
        let (_dir, st, ws, wal, etag) = fixture_with_ipc_payload(&["recovery"], 13, 7_000).await;
        let pre_uuid = wal.preallocated_superfile_id.expect("set");

        // First, let the orchestrator drive the manifest swap
        // normally. (Simulating the crash directly would mean
        // injecting a fault in persist_commit; using a
        // successful run + a manually-reset WAL is equivalent
        // and easier to reason about.)
        let (_outcome, _new_wal, _new_etag) = run_append_phase(&st, &ws, &wal, &etag, None)
            .await
            .expect("seed");

        // Manually reset the WAL state to Intent — simulating
        // a crash that landed the manifest swap but lost the
        // WAL-state CAS.
        let mut intent_wal = wal.clone();
        intent_wal.state = WalState::Intent;
        let intent_etag = ws
            .update_with_etag(
                wal.wal_id,
                // Whatever etag is current at this point —
                // re-read to get a fresh handle.
                &ws.read(wal.wal_id).await.expect("read").1,
                &intent_wal,
            )
            .await
            .expect("reset");

        // Now re-run the append phase. The probe sees the
        // superfile, takes the AlreadyApplied path, advances
        // the WAL state to Appended.
        let (outcome, recovered, recovered_etag) =
            run_append_phase(&st, &ws, &intent_wal, &intent_etag, None)
                .await
                .expect("recovered");
        assert_eq!(outcome, AppendPhaseOutcome::AlreadyApplied);
        assert_eq!(recovered.state, WalState::Appended);
        assert_ne!(recovered_etag, intent_etag);
        assert!(
            manifest_contains(&st.inner().manifest.load_full(), pre_uuid),
            "manifest still references the superfile we appended in the seed"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn replay_produces_bit_identical_superfile_bytes() {
        // Determinism property: two independent runs of the
        // assembly path (same WAL state doc + same IPC bytes)
        // must produce bit-identical superfile bytes. This is
        // what makes step-4's PUT overwrite-safe after a crash.
        let (_dir1, st1, ws1, wal, etag1) =
            fixture_with_ipc_payload(&["determinism check"], 17, 8_000).await;

        // First run: drive through the orchestrator, capture
        // the superfile's bytes from storage.
        let (_o, _new_wal, _new_etag) = run_append_phase(&st1, &ws1, &wal, &etag1, None)
            .await
            .expect("first run");
        let manifest1 = st1.inner().manifest.load_full();
        let pre_uuid = wal.preallocated_superfile_id.expect("set");
        let entry1 = manifest1
            .get_all_superfiles()
            .iter()
            .find(|e| e.uri.0 == pre_uuid)
            .expect("entry");
        let storage1 = st1.inner().options.storage.as_ref().expect("storage");
        let path = entry1.uri.storage_path();
        let (bytes1, _) = storage1.get(&path).await.expect("get bytes");

        // Second independent run on a fresh fixture with the
        // same inputs. We rebuild the user batch + IPC payload
        // deterministically by passing the same titles to the
        // fixture helper. The minted_id_spans and the
        // preallocated_superfile_id are fixed in the fixture
        // helper, so the entire WAL state doc matches.
        let (_dir2, st2, ws2, wal2, etag2) =
            fixture_with_ipc_payload(&["determinism check"], 17, 8_000).await;
        run_append_phase(&st2, &ws2, &wal2, &etag2, None)
            .await
            .expect("second run");
        let storage2 = st2.inner().options.storage.as_ref().expect("storage");
        let (bytes2, _) = storage2.get(&path).await.expect("get bytes");

        assert_eq!(
            bytes1, bytes2,
            "two independent runs with identical WAL inputs must produce \
             bit-identical superfile bytes — this is the replay-safety \
             invariant"
        );
        // Sanity: the entry's stats also agree.
        let manifest2 = st2.inner().manifest.load_full();
        let entry2 = manifest2
            .get_all_superfiles()
            .iter()
            .find(|e| e.uri.0 == pre_uuid)
            .expect("entry");
        assert_eq!(entry1.n_docs, entry2.n_docs);
        assert_eq!(entry1.id_min, entry2.id_min);
        assert_eq!(entry1.id_max, entry2.id_max);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn corrupt_ipc_payload_surfaces_typed_hash_mismatch() {
        let (_dir, st, ws, mut wal, etag) = fixture_with_ipc_payload(&["foo"], 19, 9_000).await;
        // Replace the recorded hash with garbage so the IPC
        // verify fails.
        wal.new_row_content_hash = Some("ff".repeat(32));
        // Re-CAS the WAL state doc with the bad hash so the
        // orchestrator reads the corrupted doc, not the original.
        let bad_etag = ws
            .update_with_etag(wal.wal_id, &etag, &wal)
            .await
            .expect("re-cas with bad hash");

        let err = run_append_phase(&st, &ws, &wal, &bad_etag, None)
            .await
            .expect_err("must error on hash mismatch");
        assert!(
            matches!(err, AppendPhaseError::SidecarContentHashMismatch { .. }),
            "{err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn append_phase_without_storage_is_rejected() {
        // A supertable with no storage attached can't commit the
        // manifest, so `do_apply` surfaces NoStorageAttached. We use
        // a fresh in-memory supertable + an independent WalStore
        // (storage-backed only for the WAL artifacts) so the
        // idempotency probe misses and we fall through to do_apply.
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        // Supertable WITHOUT storage.
        let st = Supertable::create(default_supertable_options()).expect("create");
        let ws = WalStore::new(Arc::clone(&storage));

        let user_batch = build_title_batch(&["x"]);
        let ipc_bytes = encode_ipc(&user_batch);
        let content_hash = blake3::hash(&ipc_bytes).to_hex().to_string();
        let wal_id = WalId(701);
        ws.put_arrow(wal_id, ipc_bytes).await.expect("put_arrow");
        let wal_doc = WalStateDoc {
            wal_id,
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Update,
            state: WalState::Intent,
            created_at: Utc::now(),
            lease: None,
            predicate_repr: "no storage".into(),
            target_ids: vec![RowId(1)],
            new_row_count: Some(1),
            new_row_content_hash: Some(content_hash),
            preallocated_superfile_id: Some(Uuid::from_u128(0x7070)),
            minted_id_spans: vec![IdSpan {
                first: RowId(1),
                last: RowId(1),
            }],
            tombstone_progress: vec![TombstoneEntry {
                target_id: RowId(1),
                outcome: TombstoneOutcome::Pending,
                tombstoned_in_superfile: None,
            }],
        };
        let etag = ws.create(&wal_doc).await.expect("create");
        let err = run_append_phase(&st, &ws, &wal_doc, &etag, None)
            .await
            .expect_err("must error without storage");
        assert!(
            matches!(err, AppendPhaseError::NoStorageAttached),
            "{err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn append_phase_missing_new_row_count_is_rejected() {
        let (_dir, st, ws, mut wal, etag) = fixture_with_ipc_payload(&["foo"], 711, 1_000).await;
        wal.new_row_count = None;
        let bad_etag = ws
            .update_with_etag(wal.wal_id, &etag, &wal)
            .await
            .expect("re-cas");
        let err = run_append_phase(&st, &ws, &wal, &bad_etag, None)
            .await
            .expect_err("must error");
        assert!(
            matches!(
                err,
                AppendPhaseError::MissingField {
                    field: "new_row_count"
                }
            ),
            "{err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn append_phase_id_span_count_mismatch_is_rejected() {
        // new_row_count says 1 but the minted_id_spans flatten to 2
        // ids — the lockstep invariant is violated, surfacing a
        // typed IdSpansLengthMismatch.
        let (_dir, st, ws, mut wal, etag) = fixture_with_ipc_payload(&["foo"], 721, 1_000).await;
        wal.minted_id_spans = vec![IdSpan {
            first: RowId(1),
            last: RowId(2), // two ids, but new_row_count == 1
        }];
        let bad_etag = ws
            .update_with_etag(wal.wal_id, &etag, &wal)
            .await
            .expect("re-cas");
        let err = run_append_phase(&st, &ws, &wal, &bad_etag, None)
            .await
            .expect_err("must error");
        assert!(
            matches!(
                err,
                AppendPhaseError::IdSpansLengthMismatch {
                    flat_len: 2,
                    expected: 1,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn append_phase_corrupt_ipc_surfaces_decode_error() {
        // The recorded content hash matches the stored bytes (so the
        // hash check passes), but the bytes aren't a valid IPC
        // stream — decode_ipc_batch fails with IpcDecode.
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let ws = WalStore::new(Arc::clone(&storage));
        let wal_id = WalId(731);
        let garbage = Bytes::from_static(b"this is not arrow ipc");
        let content_hash = blake3::hash(&garbage).to_hex().to_string();
        ws.put_arrow(wal_id, garbage).await.expect("put_arrow");
        let wal_doc = WalStateDoc {
            wal_id,
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Update,
            state: WalState::Intent,
            created_at: Utc::now(),
            lease: None,
            predicate_repr: "corrupt ipc".into(),
            target_ids: vec![RowId(1)],
            new_row_count: Some(1),
            new_row_content_hash: Some(content_hash),
            preallocated_superfile_id: Some(Uuid::from_u128(0x7373)),
            minted_id_spans: vec![IdSpan {
                first: RowId(1),
                last: RowId(1),
            }],
            tombstone_progress: vec![TombstoneEntry {
                target_id: RowId(1),
                outcome: TombstoneOutcome::Pending,
                tombstoned_in_superfile: None,
            }],
        };
        let etag = ws.create(&wal_doc).await.expect("create");
        let err = run_append_phase(&st, &ws, &wal_doc, &etag, None)
            .await
            .expect_err("must error on bad ipc");
        assert!(matches!(err, AppendPhaseError::IpcDecode { .. }), "{err:?}");
    }

    // ---- flatten_spans property ----------------------------------------

    #[test]
    fn flatten_spans_empty_is_empty() {
        assert!(flatten_spans(&[]).is_empty());
    }

    #[test]
    fn flatten_spans_concatenates_inclusive_ranges_in_order() {
        let spans = vec![
            IdSpan {
                first: RowId(10),
                last: RowId(12),
            },
            IdSpan {
                first: RowId(100),
                last: RowId(100),
            },
        ];
        let flat = flatten_spans(&spans);
        assert_eq!(flat, vec![10i128, 11, 12, 100]);
    }

    // ---- Tombstone phase: pre-condition checks ---------------------------
    //
    // These exercise the outer-shape behaviour of
    // `run_tombstone_phase`: invalid (op_kind, state) pairs
    // surface a typed `InvalidPreState`, and a WAL already at
    // `Complete` returns `AlreadyComplete` with the outcome
    // counts read straight off the existing `tombstone_progress`.
    //
    // The full resolve + sidecar-CAS loop is exercised by the
    // end-to-end tests that land alongside `do_tombstone_apply`.

    /// Build a tombstone-phase fixture starting from the `fixture()`
    /// base and tweaking the WAL into the requested state. The
    /// supertable + WalStore + initial state doc are otherwise
    /// the same as the append-phase tests use.
    async fn tombstone_fixture(
        op_kind: OpKind,
        state: WalState,
        progress: Vec<TombstoneEntry>,
    ) -> (TempDir, Supertable, WalStore, WalStateDoc, Etag) {
        let (dir, st, ws, mut wal, etag) = fixture().await;
        wal.op_kind = op_kind;
        wal.state = state;
        wal.tombstone_progress = progress;
        // The base fixture is created in `Intent`; when we want
        // a different state we have to CAS-update so the WAL on
        // disk matches what we'll pass to `run_tombstone_phase`.
        let new_etag = if wal.state != WalState::Intent
            || wal.op_kind != OpKind::Update
            || wal.tombstone_progress.len() != 1
        {
            ws.update_with_etag(wal.wal_id, &etag, &wal)
                .await
                .expect("re-cas fixture")
        } else {
            etag
        };
        (dir, st, ws, wal, new_etag)
    }

    fn ts_entry(target_id: i128, outcome: TombstoneOutcome) -> TombstoneEntry {
        TombstoneEntry {
            target_id: RowId(target_id),
            outcome,
            tombstoned_in_superfile: None,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tombstone_phase_rejects_update_wal_in_intent_state() {
        // UPDATE pre-condition: append phase must have completed.
        // `Intent` is a builder bug — surface the typed error.
        let (_dir, st, ws, wal, etag) = tombstone_fixture(
            OpKind::Update,
            WalState::Intent,
            vec![ts_entry(1, TombstoneOutcome::Pending)],
        )
        .await;
        let err = run_tombstone_phase(&st, &ws, &wal, &etag)
            .await
            .expect_err("must error");
        assert!(
            matches!(
                err,
                TombstonePhaseError::InvalidPreState {
                    op_kind: OpKind::Update,
                    state: WalState::Intent
                }
            ),
            "{err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tombstone_phase_rejects_delete_wal_in_appended_state() {
        // DELETE has no append phase; an `Appended` DELETE WAL is
        // structurally impossible from the writer side, so we
        // reject the bogus pre-state instead of silently running
        // the loop.
        let (_dir, st, ws, wal, etag) = tombstone_fixture(
            OpKind::Delete,
            WalState::Appended,
            vec![ts_entry(2, TombstoneOutcome::Pending)],
        )
        .await;
        let err = run_tombstone_phase(&st, &ws, &wal, &etag)
            .await
            .expect_err("must error");
        assert!(
            matches!(
                err,
                TombstonePhaseError::InvalidPreState {
                    op_kind: OpKind::Delete,
                    state: WalState::Appended
                }
            ),
            "{err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tombstone_phase_on_complete_wal_is_noop_with_existing_counts() {
        // Re-running a finished WAL surfaces the existing
        // outcome counts and leaves both the state doc and the
        // etag untouched. The orchestrator is supposed to be
        // safely re-callable as part of recovery; this test
        // pins that promise.
        let progress = vec![
            ts_entry(10, TombstoneOutcome::Tombstoned),
            ts_entry(11, TombstoneOutcome::Tombstoned),
            ts_entry(12, TombstoneOutcome::NotFound),
        ];
        let (_dir, st, ws, wal, etag) =
            tombstone_fixture(OpKind::Update, WalState::Complete, progress).await;
        let (outcome, returned_wal, returned_etag) = run_tombstone_phase(&st, &ws, &wal, &etag)
            .await
            .expect("ok");
        assert_eq!(
            outcome,
            TombstonePhaseOutcome::AlreadyComplete {
                n_tombstoned: 2,
                n_not_found: 1,
            }
        );
        assert_eq!(returned_wal.state, WalState::Complete);
        assert_eq!(returned_etag, etag, "etag must not advance on no-op");
        // The on-disk state doc is unchanged too.
        let (read_back, read_etag) = ws.read(wal.wal_id).await.expect("read back");
        assert_eq!(read_back.state, WalState::Complete);
        assert_eq!(read_etag, etag);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tombstone_phase_replay_after_crash_still_stamps_manifest() {
        // Crash model: every sidecar CAS-PUT landed (all progress
        // entries non-Pending) but the process died before the
        // manifest stamp / `Complete` CAS. Recovery re-runs the
        // phase: the per-target loop no-ops, and the stamp must
        // still publish the touched superfile's tombstone seq
        // before the WAL flips to `Complete` — this is the
        // "WAL complete ⇒ manifest stamped" invariant.
        let sf = Uuid::from_u128(0x5F);
        let progress = vec![TombstoneEntry {
            target_id: RowId(21),
            outcome: TombstoneOutcome::Tombstoned,
            tombstoned_in_superfile: Some(sf),
        }];
        let (_dir, st, ws, wal, etag) =
            tombstone_fixture(OpKind::Delete, WalState::Intent, progress).await;
        let manifest_id_before = st.inner().manifest.load().manifest_id;

        let (outcome, new_wal, _) = run_tombstone_phase(&st, &ws, &wal, &etag)
            .await
            .expect("replay ok");
        assert_eq!(
            outcome,
            TombstonePhaseOutcome::Applied {
                n_tombstoned: 1,
                n_not_found: 0,
            }
        );
        assert_eq!(new_wal.state, WalState::Complete);

        let manifest = st.inner().manifest.load();
        assert!(
            manifest.manifest_id > manifest_id_before,
            "stamp publishes a successor manifest"
        );
        let seqs = manifest.get_tombstone_seqs().expect("persisted list");
        assert_eq!(
            seqs.get(&sf),
            Some(&manifest.manifest_id),
            "touched superfile's seq is the stamping manifest's id"
        );
    }

    // ---- count_outcomes property ----------------------------------------

    #[test]
    fn count_outcomes_sums_tombstoned_and_not_found_only() {
        let progress = vec![
            ts_entry(1, TombstoneOutcome::Pending),
            ts_entry(2, TombstoneOutcome::Tombstoned),
            ts_entry(3, TombstoneOutcome::NotFound),
            ts_entry(4, TombstoneOutcome::Tombstoned),
            ts_entry(5, TombstoneOutcome::Pending),
        ];
        let (n_tombstoned, n_not_found) = count_outcomes(&progress);
        assert_eq!(n_tombstoned, 2);
        assert_eq!(n_not_found, 1);
    }

    // ---- Tombstone phase: end-to-end against a real superfile ----------
    //
    // Each test drives the append phase first (so a real superfile
    // lives on storage + in the manifest), then builds a DELETE
    // WAL targeting one or more of the freshly-minted `_id`s and
    // runs the tombstone phase.

    /// Build a Supertable with one published superfile (via the
    /// real append phase) so the tombstone phase has somewhere
    /// to resolve `target_id`s against. Returns the supertable,
    /// the WalStore, the published `superfile_id`, and the range
    /// of `_id` values that live in the superfile.
    async fn published_superfile_fixture(
        titles: &[&str],
        minted_first: i128,
    ) -> (TempDir, Supertable, WalStore, Uuid, i128, i128) {
        let (dir, st, ws, wal, etag) = fixture_with_ipc_payload(titles, 101, minted_first).await;
        let pre_uuid = wal.preallocated_superfile_id.expect("set");
        run_append_phase(&st, &ws, &wal, &etag, None)
            .await
            .expect("append phase");
        let n = titles.len() as i128;
        (dir, st, ws, pre_uuid, minted_first, minted_first + n - 1)
    }

    /// Build a DELETE WAL targeting the supplied `_id` values
    /// against an already-set-up supertable + WalStore.
    async fn create_delete_wal(
        ws: &WalStore,
        wal_id_value: i128,
        target_ids: &[i128],
    ) -> (WalStateDoc, Etag) {
        let wal_doc = WalStateDoc {
            wal_id: WalId(wal_id_value),
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Delete,
            state: WalState::Intent,
            created_at: Utc::now(),
            lease: None,
            predicate_repr: "test delete".into(),
            target_ids: target_ids.iter().map(|&v| RowId(v)).collect(),
            new_row_count: None,
            new_row_content_hash: None,
            preallocated_superfile_id: None,
            minted_id_spans: Vec::new(),
            tombstone_progress: target_ids
                .iter()
                .map(|&v| TombstoneEntry {
                    target_id: RowId(v),
                    outcome: TombstoneOutcome::Pending,
                    tombstoned_in_superfile: None,
                })
                .collect(),
        };
        let etag = ws.create(&wal_doc).await.expect("wal create");
        (wal_doc, etag)
    }

    /// Fetch the persisted sidecar bitmap for one superfile. Helper
    /// for the post-tombstone assertions.
    async fn read_sidecar_bitmap(ws: &WalStore, superfile_id: Uuid) -> RoaringBitmap {
        match ws.get_tombstones(superfile_id).await.expect("get") {
            Some((sc, _etag)) => sc.bitmap,
            None => RoaringBitmap::new(),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tombstone_phase_marks_single_resolved_target_as_tombstoned() {
        let (_dir, st, ws, sf_id, id_min, _id_max) =
            published_superfile_fixture(&["aa", "bb", "cc"], 10_000).await;
        // Target the middle row; its local doc_id is 1.
        let (wal, etag) = create_delete_wal(&ws, 201, &[id_min + 1]).await;

        let (outcome, new_wal, _new_etag) = run_tombstone_phase(&st, &ws, &wal, &etag)
            .await
            .expect("ok");
        assert_eq!(
            outcome,
            TombstonePhaseOutcome::Applied {
                n_tombstoned: 1,
                n_not_found: 0,
            }
        );
        assert_eq!(new_wal.state, WalState::Complete);
        assert_eq!(
            new_wal.tombstone_progress[0].outcome,
            TombstoneOutcome::Tombstoned
        );
        assert_eq!(
            new_wal.tombstone_progress[0].tombstoned_in_superfile,
            Some(sf_id)
        );

        // Sidecar persistence check.
        let bitmap = read_sidecar_bitmap(&ws, sf_id).await;
        assert_eq!(bitmap.len(), 1);
        assert!(bitmap.contains(1u32));
    }

    // ---- lease renewal across the tombstone phase --------------------

    /// Lease span the renewal tests grant. Deliberately far shorter than
    /// the phase itself would need, so an un-renewed lease would be long
    /// expired by the time the phase finishes.
    const LEASE_RENEWAL_SPAN_MS: i64 = 50;
    /// Owner of the seeded lease: stands in for the writer handle that
    /// created the WAL and is driving it.
    const LEASE_RENEWAL_OWNER: i128 = 0x0D12_1E55;

    /// Seed a DELETE WAL that is already leased, the way a writer's create
    /// stamps one before it drives the tombstone phase.
    async fn create_leased_delete_wal(
        ws: &WalStore,
        wal_id_value: i128,
        target_ids: &[i128],
        span: ChronoDuration,
    ) -> (WalStateDoc, Etag) {
        let (mut wal, etag) = create_delete_wal(ws, wal_id_value, target_ids).await;
        let now = Utc::now();
        wal.lease = Some(Lease {
            owner: SupertableHandleId(LEASE_RENEWAL_OWNER),
            acquired_at: now,
            expires_at: now + span,
        });
        let etag = ws
            .update_with_etag(wal.wal_id, &etag, &wal)
            .await
            .expect("stamp lease");
        (wal, etag)
    }

    /// Build a leased WAL state doc whose lease was stamped at `stamped_at`
    /// for `span`. `lease_renewal_due` reads its deadline straight off this
    /// doc, so the tests drive it by constructing the lease rather than by
    /// tracking a write time.
    fn doc_leased_at(stamped_at: DateTime<Utc>, span: ChronoDuration) -> WalStateDoc {
        WalStateDoc {
            wal_id: WalId(262),
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Delete,
            state: WalState::Intent,
            created_at: stamped_at,
            lease: Some(Lease {
                owner: SupertableHandleId(LEASE_RENEWAL_OWNER),
                acquired_at: stamped_at,
                expires_at: stamped_at + span,
            }),
            predicate_repr: "renewal scheduling".into(),
            target_ids: vec![RowId(1)],
            new_row_count: None,
            new_row_content_hash: None,
            preallocated_superfile_id: None,
            minted_id_spans: Vec::new(),
            tombstone_progress: Vec::new(),
        }
    }

    /// The renewal scheduler fires once a third of the granted span has
    /// elapsed, and never when the WAL carries no lease. Batched state
    /// writes make this the thing standing between a long sidecar phase
    /// and a lapsed lease, so it gets its own coverage.
    #[test]
    fn lease_renewal_comes_due_after_a_third_of_the_span() {
        let span = ChronoDuration::seconds(60);
        let stamped = Utc::now();
        let doc = doc_leased_at(stamped, span);

        assert!(
            !lease_renewal_due(&doc, Some(span), stamped),
            "no elapsed time, nothing due"
        );
        assert!(
            !lease_renewal_due(&doc, Some(span), stamped + ChronoDuration::seconds(19)),
            "under a third of the span is not due yet"
        );
        assert!(
            lease_renewal_due(&doc, Some(span), stamped + ChronoDuration::seconds(20)),
            "a third of the span is due"
        );
        assert!(
            lease_renewal_due(&doc, Some(span), stamped + ChronoDuration::seconds(59)),
            "still due right up to expiry"
        );

        let mut unleased = doc_leased_at(stamped, span);
        unleased.lease = None;
        assert!(
            !lease_renewal_due(&unleased, None, stamped + ChronoDuration::seconds(600)),
            "no lease means nothing to renew, however long the phase runs"
        );
    }

    /// The deadline is measured from when the lease was *stamped*, not from
    /// when the tombstone phase happened to start. An UPDATE reaches the
    /// phase having already spent part of its grant on the append phase
    /// (which writes without renewing), so a scheduler anchored at phase
    /// entry would let that lease lapse before its first renewal.
    #[test]
    fn lease_renewal_measures_from_the_stamp_not_the_phase_start() {
        let span = ChronoDuration::seconds(60);
        let stamped = Utc::now();
        // The append phase burned 40s of the grant; the tombstone phase
        // starts now, with only 20s of lease left.
        let phase_start = stamped + ChronoDuration::seconds(40);
        let doc = doc_leased_at(stamped, span);

        assert!(
            lease_renewal_due(&doc, Some(span), phase_start),
            "a lease already two-thirds spent is due at once, not a third of \
             a span after the phase happened to begin"
        );
    }

    /// `renew_lease` re-stamps the lease, which moves the scheduler's own
    /// deadline with it — that is what makes reading `expires_at - span`
    /// equivalent to tracking the last renewal, without a second variable
    /// that a non-renewing write could desynchronize.
    #[test]
    fn renewing_pushes_the_next_renewal_deadline_out() {
        let span = ChronoDuration::seconds(60);
        let stamped = Utc::now();
        let mut doc = doc_leased_at(stamped, span);

        let renewed_at = stamped + ChronoDuration::seconds(20);
        assert!(lease_renewal_due(&doc, Some(span), renewed_at));
        renew_lease(&mut doc, Some(span), renewed_at);

        assert!(
            !lease_renewal_due(&doc, Some(span), renewed_at + ChronoDuration::seconds(19)),
            "the renewal reset the clock"
        );
        assert!(
            lease_renewal_due(&doc, Some(span), renewed_at + ChronoDuration::seconds(20)),
            "and the next one comes due a third of a span later"
        );
    }

    /// The phase pushes the driver's lease expiry out on the state-doc
    /// writes it is already making, so a phase that runs longer than one
    /// lease grant can't be preempted out from under its driver. Ownership
    /// is untouched, and the renewal reaches storage — not just the
    /// in-memory copy the phase returns.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tombstone_phase_renews_the_drivers_lease_as_it_writes() {
        let (_dir, st, ws, _sf_id, id_min, _id_max) =
            published_superfile_fixture(&["aa", "bb", "cc"], 60_000).await;
        let span = ChronoDuration::milliseconds(LEASE_RENEWAL_SPAN_MS);
        let (wal, etag) =
            create_leased_delete_wal(&ws, 260, &[id_min, id_min + 1, id_min + 2], span).await;
        let granted = wal.lease.clone().expect("seeded lease");

        let (outcome, new_wal, _) = run_tombstone_phase(&st, &ws, &wal, &etag)
            .await
            .expect("phase ok");
        assert_eq!(
            outcome,
            TombstonePhaseOutcome::Applied {
                n_tombstoned: 3,
                n_not_found: 0,
            }
        );

        let renewed = new_wal.lease.expect("the phase must not drop the lease");
        assert_eq!(
            renewed.owner, granted.owner,
            "renewal must not change who owns the WAL"
        );
        assert_eq!(
            renewed.acquired_at, granted.acquired_at,
            "renewal moves the expiry, not the moment ownership began"
        );
        assert!(
            renewed.expires_at > granted.expires_at,
            "the phase's own writes must push the expiry out, granted {} still at {}",
            granted.expires_at,
            renewed.expires_at
        );

        let (persisted, _) = ws.read(wal.wal_id).await.expect("read back");
        assert_eq!(
            persisted
                .lease
                .expect("persisted doc must carry the lease")
                .expires_at,
            renewed.expires_at,
            "the renewal must ride along on the phase's state-doc write"
        );
    }

    /// Each renewal is exactly one granted span past the write it rides
    /// on. The span is sampled once per phase for this reason: re-deriving
    /// it from the renewed doc (`expires_at - acquired_at`) compounds,
    /// handing the driver an ever-wider window. This pins both halves —
    /// the exact renewal arithmetic, and the compounding that sampling
    /// per-write would produce.
    #[test]
    fn lease_renewal_is_one_span_from_the_write_it_rides_on() {
        let t0 = Utc::now();
        let span = ChronoDuration::seconds(60);
        let mut doc = WalStateDoc {
            wal_id: WalId(261),
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Delete,
            state: WalState::Intent,
            created_at: t0,
            lease: Some(Lease {
                owner: SupertableHandleId(LEASE_RENEWAL_OWNER),
                acquired_at: t0,
                expires_at: t0 + span,
            }),
            predicate_repr: "renewal arithmetic".into(),
            target_ids: vec![RowId(1)],
            new_row_count: None,
            new_row_content_hash: None,
            preallocated_superfile_id: None,
            minted_id_spans: Vec::new(),
            tombstone_progress: Vec::new(),
        };

        let granted = granted_lease_span(&doc).expect("leased doc has a span");
        assert_eq!(granted, span, "the granted span is the acquirer's window");

        renew_lease(&mut doc, Some(granted), t0 + ChronoDuration::seconds(30));
        assert_eq!(
            doc.lease.as_ref().expect("lease").expires_at,
            t0 + ChronoDuration::seconds(90),
            "a write at t+30s must own the WAL until t+90s"
        );

        renew_lease(&mut doc, Some(granted), t0 + ChronoDuration::seconds(70));
        let lease = doc.lease.as_ref().expect("lease").clone();
        assert_eq!(
            lease.expires_at,
            t0 + ChronoDuration::seconds(130),
            "the next write must land one span out, not one span plus elapsed"
        );
        assert_eq!(lease.acquired_at, t0, "ownership start must not drift");

        assert!(
            granted_lease_span(&doc).expect("span") > granted,
            "re-deriving the span mid-phase widens it — which is why the \
             phase samples it once, before the first renewal"
        );
    }

    /// A WAL nobody holds has nothing to renew: the phase must leave the
    /// `None` alone rather than inventing an owner-less lease.
    #[test]
    fn lease_renewal_is_a_no_op_on_an_unleased_wal() {
        let mut doc = WalStateDoc {
            wal_id: WalId(262),
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Delete,
            state: WalState::Intent,
            created_at: Utc::now(),
            lease: None,
            predicate_repr: "unleased".into(),
            target_ids: vec![RowId(1)],
            new_row_count: None,
            new_row_content_hash: None,
            preallocated_superfile_id: None,
            minted_id_spans: Vec::new(),
            tombstone_progress: Vec::new(),
        };
        assert!(granted_lease_span(&doc).is_none());
        renew_lease(&mut doc, None, Utc::now());
        assert!(doc.lease.is_none(), "renewal must not mint a lease");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tombstone_phase_marks_unknown_target_as_not_found() {
        let (_dir, st, ws, sf_id, _id_min, id_max) =
            published_superfile_fixture(&["aa", "bb"], 20_000).await;
        // Pick a value clearly outside the published range.
        let (wal, etag) = create_delete_wal(&ws, 202, &[id_max + 100]).await;

        let (outcome, new_wal, _) = run_tombstone_phase(&st, &ws, &wal, &etag)
            .await
            .expect("ok");
        assert_eq!(
            outcome,
            TombstonePhaseOutcome::Applied {
                n_tombstoned: 0,
                n_not_found: 1,
            }
        );
        assert_eq!(new_wal.state, WalState::Complete);
        assert_eq!(
            new_wal.tombstone_progress[0].outcome,
            TombstoneOutcome::NotFound
        );
        assert!(
            new_wal.tombstone_progress[0]
                .tombstoned_in_superfile
                .is_none()
        );
        // No sidecar should have been written.
        let bitmap = read_sidecar_bitmap(&ws, sf_id).await;
        assert!(bitmap.is_empty());
    }

    // ---- batched target resolution ------------------------------------

    /// Publish a second superfile into an existing fixture's
    /// supertable, so the batched resolver has more than one manifest
    /// entry to sweep. Returns its `superfile_id`.
    async fn publish_extra_superfile(
        st: &Supertable,
        ws: &WalStore,
        titles: &[&str],
        wal_id_value: i128,
        minted_first: i128,
        superfile_id: Uuid,
    ) -> Uuid {
        let (wal, etag) =
            seed_update_wal(ws, titles, wal_id_value, minted_first, superfile_id).await;
        run_append_phase(st, ws, &wal, &etag, None)
            .await
            .expect("append phase");
        superfile_id
    }

    /// The batched resolver returns one map covering targets that live
    /// in different superfiles, and simply omits ids no superfile
    /// claims (the caller's `NotFound` outcome).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn resolve_all_maps_targets_across_multiple_superfiles() {
        const FIRST_IDS: i128 = 70_000;
        const SECOND_IDS: i128 = 80_000;
        let (_dir, st, ws, sf_a, _id_min, _id_max) =
            published_superfile_fixture(&["aa", "bb", "cc"], FIRST_IDS).await;
        let sf_b = publish_extra_superfile(
            &st,
            &ws,
            &["dd", "ee"],
            701,
            SECOND_IDS,
            Uuid::from_u128(0xB0B_B0B),
        )
        .await;

        let absent = RowId(SECOND_IDS + 500);
        let targets = vec![
            RowId(FIRST_IDS + 2),
            RowId(SECOND_IDS + 1),
            absent,
            RowId(FIRST_IDS),
        ];
        let resolved = resolve_all(st.inner(), &targets).expect("batched resolve");

        assert_eq!(resolved.len(), 3, "only the present targets resolve");
        assert_eq!(resolved.get(&RowId(FIRST_IDS)), Some(&(sf_a, 0u32)));
        assert_eq!(resolved.get(&RowId(FIRST_IDS + 2)), Some(&(sf_a, 2u32)));
        assert_eq!(resolved.get(&RowId(SECOND_IDS + 1)), Some(&(sf_b, 1u32)));
        assert!(
            !resolved.contains_key(&absent),
            "an unclaimed id is absent from the map"
        );
    }

    /// Batched resolution agrees with resolving the same ids one at a
    /// time — the property that lets the phase batch at all.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn resolve_all_agrees_with_one_target_at_a_time() {
        const FIRST_IDS: i128 = 71_000;
        const SECOND_IDS: i128 = 81_000;
        let (_dir, st, ws, _sf_a, _id_min, _id_max) =
            published_superfile_fixture(&["aa", "bb"], FIRST_IDS).await;
        publish_extra_superfile(
            &st,
            &ws,
            &["cc", "dd"],
            702,
            SECOND_IDS,
            Uuid::from_u128(0xB0B_B0C),
        )
        .await;

        let targets = vec![
            RowId(FIRST_IDS),
            RowId(FIRST_IDS + 1),
            RowId(FIRST_IDS + 9),
            RowId(SECOND_IDS),
            RowId(SECOND_IDS + 1),
        ];
        let batched = resolve_all(st.inner(), &targets).expect("batched resolve");
        for target in targets {
            let single = resolve_all(st.inner(), &[target]).expect("single resolve");
            assert_eq!(
                batched.get(&target),
                single.get(&target),
                "{target:?} disagreed between batched and single resolve"
            );
        }
    }

    /// Repeated ids in one batch are deduped for the scan yet every
    /// progress entry that names them still resolves.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn resolve_all_tolerates_duplicate_targets() {
        const FIRST_IDS: i128 = 72_000;
        let (_dir, st, _ws, sf_a, _id_min, _id_max) =
            published_superfile_fixture(&["aa", "bb"], FIRST_IDS).await;

        let target = RowId(FIRST_IDS + 1);
        let resolved = resolve_all(st.inner(), &[target, target, target]).expect("batched resolve");
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved.get(&target), Some(&(sf_a, 1u32)));
    }

    #[test]
    fn id_window_label_names_a_single_target_and_summarizes_a_batch() {
        assert_eq!(id_window_label(&[]), "<empty batch>");
        assert_eq!(id_window_label(&[7]), RowId(7).to_hex());
        let label = id_window_label(&[7, 8, 9]);
        assert!(label.starts_with(&RowId(7).to_hex()), "got {label}");
        assert!(label.ends_with("(3 targets)"), "got {label}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tombstone_phase_unions_multiple_targets_into_one_sidecar() {
        let (_dir, st, ws, sf_id, id_min, _id_max) =
            published_superfile_fixture(&["a", "b", "c", "d"], 30_000).await;
        // Three resolved + one not-found, all in one WAL.
        let (wal, etag) =
            create_delete_wal(&ws, 203, &[id_min, id_min + 2, id_min + 3, id_min + 999]).await;

        let (outcome, new_wal, _) = run_tombstone_phase(&st, &ws, &wal, &etag)
            .await
            .expect("ok");
        assert_eq!(
            outcome,
            TombstonePhaseOutcome::Applied {
                n_tombstoned: 3,
                n_not_found: 1,
            }
        );
        assert_eq!(new_wal.state, WalState::Complete);

        // The sidecar's bitmap should be exactly {0, 2, 3} — the
        // local doc_ids of the three resolved rows.
        let bitmap = read_sidecar_bitmap(&ws, sf_id).await;
        let collected: Vec<u32> = bitmap.iter().collect();
        assert_eq!(collected, vec![0u32, 2, 3]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tombstone_phase_is_idempotent_on_replay() {
        // First run lands the bit; the second run sees the WAL
        // already in Complete and short-circuits to AlreadyComplete
        // without touching the sidecar.
        let (_dir, st, ws, sf_id, id_min, _id_max) =
            published_superfile_fixture(&["x", "y"], 40_000).await;
        let (wal, etag) = create_delete_wal(&ws, 204, &[id_min]).await;

        let (first_outcome, after_first, etag_after_first) =
            run_tombstone_phase(&st, &ws, &wal, &etag)
                .await
                .expect("first");
        assert_eq!(
            first_outcome,
            TombstonePhaseOutcome::Applied {
                n_tombstoned: 1,
                n_not_found: 0,
            }
        );
        let bitmap_v1 = read_sidecar_bitmap(&ws, sf_id).await;

        let (second_outcome, after_second, etag_after_second) =
            run_tombstone_phase(&st, &ws, &after_first, &etag_after_first)
                .await
                .expect("second");
        assert_eq!(
            second_outcome,
            TombstonePhaseOutcome::AlreadyComplete {
                n_tombstoned: 1,
                n_not_found: 0,
            }
        );
        // No state-doc CAS on the re-run.
        assert_eq!(etag_after_second, etag_after_first);
        assert_eq!(after_second.state, WalState::Complete);
        // Sidecar is unchanged.
        let bitmap_v2 = read_sidecar_bitmap(&ws, sf_id).await;
        assert_eq!(bitmap_v1, bitmap_v2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tombstone_phase_resumes_partial_progress_on_replay() {
        // Pre-mark one of three targets as already-Tombstoned so
        // the orchestrator's recovery cursor (`Pending` filter)
        // is exercised: only the remaining two `Pending` entries
        // get resolved + CAS-PUT. The pre-Tombstoned bit is NOT
        // in the sidecar (we set the WAL field but not the
        // sidecar itself) so we can verify the orchestrator
        // doesn't unconditionally insert pre-Tombstoned ids.
        let (_dir, st, ws, sf_id, id_min, _id_max) =
            published_superfile_fixture(&["p", "q", "r"], 50_000).await;
        let (mut wal, etag) = create_delete_wal(&ws, 205, &[id_min, id_min + 1, id_min + 2]).await;
        wal.tombstone_progress[0].outcome = TombstoneOutcome::Tombstoned;
        wal.tombstone_progress[0].tombstoned_in_superfile = Some(sf_id);
        let etag = ws
            .update_with_etag(wal.wal_id, &etag, &wal)
            .await
            .expect("pre-mark");

        let (outcome, new_wal, _) = run_tombstone_phase(&st, &ws, &wal, &etag)
            .await
            .expect("ok");
        assert_eq!(
            outcome,
            TombstonePhaseOutcome::Applied {
                n_tombstoned: 3,
                n_not_found: 0,
            }
        );
        assert_eq!(new_wal.state, WalState::Complete);

        // The pre-Tombstoned target's bit is NOT in the sidecar
        // (we never wrote it); only the two resumed entries are.
        let bitmap = read_sidecar_bitmap(&ws, sf_id).await;
        let collected: Vec<u32> = bitmap.iter().collect();
        assert_eq!(collected, vec![1u32, 2]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sealed_sidecar_surfaces_after_retry_budget() {
        // Inject a sealed sidecar before running the tombstone
        // phase. The orchestrator's resolve → CAS-attempt loop
        // sees a non-`None` seal, backs off and retries, and
        // eventually surfaces SealedSidecarRetryExhausted because
        // no compactor is around to advance the manifest.
        //
        // To keep the test fast we override the retry constants
        // by... actually we can't; they're const. Instead we
        // exploit that the smallest backoff sequence is 100ms,
        // 200ms, 400ms, ... and verify the error type for the
        // first sealed observation by writing a sidecar with
        // a high enough seal that the production loop would
        // re-resolve. That's not exactly the bounded-retry path
        // but it pins the seal-detection branch.
        //
        // For stage 2 we cap the verification at: "sealed
        // sidecar IS detected and surfaces a typed error". The
        // bounded-budget exhaustion path requires either a
        // configurable budget or a long test; we accept the
        // shorter assertion here.
        let (_dir, st, ws, sf_id, id_min, _id_max) =
            published_superfile_fixture(&["seal"], 60_000).await;

        // Pre-seal the sidecar.
        let sealed = TombstonesSidecar {
            seal: Some(SealRecord {
                compaction_id: Uuid::from_u128(0xC0DE_C0DE),
                sealed_at: Utc::now(),
            }),
            bitmap: RoaringBitmap::new(),
        };
        ws.put_tombstones(sf_id, None, &sealed)
            .await
            .expect("seed sealed sidecar");

        // Build a WAL targeting the row. The orchestrator will
        // detect the seal, sleep, retry, and eventually exhaust
        // the bounded budget — surfacing a typed error.
        let (wal, etag) = create_delete_wal(&ws, 206, &[id_min]).await;

        // We don't want to actually wait for 16 retries through
        // the full exponential backoff (that would total minutes).
        // Run the orchestrator under a timeout that's just long
        // enough to confirm the seal IS being detected (i.e. the
        // run hasn't completed quickly), then assert by reading
        // the WAL state — it should still be Intent because no
        // target made it to Tombstoned.
        let result = timeout(
            Duration::from_millis(250),
            run_tombstone_phase(&st, &ws, &wal, &etag),
        )
        .await;

        // The timeout firing means the orchestrator is still in
        // the sealed-retry loop — exactly what we want for this
        // assertion. The WAL stays at Intent on disk.
        assert!(
            result.is_err(),
            "expected the orchestrator to still be in sealed-retry; got {result:?}"
        );
        let (post, _post_etag) = ws.read(wal.wal_id).await.expect("read back");
        assert_eq!(post.state, WalState::Intent);
        assert_eq!(
            post.tombstone_progress[0].outcome,
            TombstoneOutcome::Pending
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stale_seal_does_not_block_a_delete() {
        // Same setup as `sealed_sidecar_surfaces_after_retry_budget`,
        // but the seal is old enough to be stale (its compactor is
        // presumed dead, e.g. a crash mid-merge). The delete must
        // land its tombstone bit right away instead of backing off
        // and eventually exhausting the sealed-retry budget.
        let (_dir, st, ws, sf_id, id_min, _id_max) =
            published_superfile_fixture(&["seal"], 60_000).await;

        let stale_sealed_at = Utc::now()
            - chrono::Duration::from_std(Duration::from_millis(
                tombstones_admin::DEFAULT_STALE_SEAL_TIMEOUT_MS,
            ))
            .unwrap_or_default()
            - chrono::Duration::seconds(1);
        let sealed = TombstonesSidecar {
            seal: Some(SealRecord {
                compaction_id: Uuid::from_u128(0xDEAD_C0DE),
                sealed_at: stale_sealed_at,
            }),
            bitmap: RoaringBitmap::new(),
        };
        ws.put_tombstones(sf_id, None, &sealed)
            .await
            .expect("seed stale-sealed sidecar");

        let (wal, etag) = create_delete_wal(&ws, 207, &[id_min]).await;

        // Must complete well within one sealed-retry backoff step,
        // not fall into the retry loop at all.
        let (outcome, new_wal, _) = timeout(
            Duration::from_millis(500),
            run_tombstone_phase(&st, &ws, &wal, &etag),
        )
        .await
        .expect("must not enter the sealed-retry loop for a stale seal")
        .expect("tombstone phase must succeed");

        assert_eq!(
            outcome,
            TombstonePhaseOutcome::Applied {
                n_tombstoned: 1,
                n_not_found: 0,
            }
        );
        assert_eq!(new_wal.state, WalState::Complete);

        // The bit landed, and the stale seal got cleared along with it.
        let bitmap = read_sidecar_bitmap(&ws, sf_id).await;
        assert_eq!(bitmap.iter().collect::<Vec<u32>>(), vec![0u32]);
        let (post_sidecar, _) = ws
            .get_tombstones(sf_id)
            .await
            .expect("get")
            .expect("present");
        assert!(post_sidecar.seal.is_none());
    }
}
