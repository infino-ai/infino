// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared fan-out/dispatch for the superfile-parallel query paths.
//!
//! Vector kNN and BM25/prefix FTS both face the identical shape: a
//! pinned manifest snapshot, a kept set of superfiles (after manifest
//! pruning), and a per-superfile search kernel whose result is a list of
//! `(local_doc_id, score)` pairs. The plumbing around that kernel —
//! open every superfile reader concurrently, warm the tombstone sidecar
//! cache in one batch, run each superfile's kernel, tag the hits with
//! their superfile URI, and drop tombstoned rows — is the same for both.
//!
//! This module owns that plumbing so the two query paths share one
//! orchestrator instead of each re-implementing the fan-out. The
//! division of labor is the project-wide model:
//!
//!   * **tokio bounds I/O concurrency — stays wide.** One spawned tokio
//!     task per work unit, aborted if its caller goes away: each opens its
//!     superfile reader and runs the kernel, so superfile opens and cold
//!     object-store range GETs across hundreds of superfiles are all in
//!     flight at once on the shared multi-thread query runtime. The one
//!     ceiling is `SupertableOptions::superfiles_in_flight` (256 by
//!     default), so a table of thousands of superfiles does not start
//!     thousands of opens together. Never cap the fan-out's task count at
//!     the reader pool's width — `tokio::spawn` dispatches a task, not an
//!     OS thread, and object-store GETs are latency-bound, so narrowing
//!     their concurrency only hurts cold latency.
//!   * **The reader pool bounds CPU concurrency — the configured knob.**
//!     Both vector and FTS scoring run their scan/scoring kernel on the
//!     configured reader pool, bridged back to the awaiting tokio task
//!     via a oneshot (`runtime_bridge::run_on_pool`) so no tokio worker
//!     blocks under the compute. The vector kernel parallelizes its own
//!     scoring + rerank with `par_iter` (see `superfile/vector/reader.rs`);
//!     FTS kernels are scored on one reader-pool thread per work unit,
//!     with intra-superfile parallelism expressed as additional tokio
//!     work units (doc-id sub-ranges; see `query/fts.rs`). Cheap
//!     shapes stay inline where the pool round trip would cost more
//!     than the scan, and so does an FTS unit a caller blocks on
//!     directly (`runtime_bridge::run_on_pool_unless_blocking`): its
//!     thread is already parked on the query, so running the kernel there
//!     adds no concurrency beyond the callers' own threads.
//!
//! The per-superfile merge (top-k ascending for vector distance,
//! descending for BM25 relevance) stays with each caller; this layer
//! returns the per-unit tagged+filtered hit lists.

use std::{
    collections::{HashSet, VecDeque},
    future::Future,
    pin::Pin,
    sync::Arc,
    time::Instant,
};

use arrow_array::Decimal128Array;
use datafusion::common::runtime::SpawnedTask;
use futures::stream::{FuturesUnordered, StreamExt};
use roaring::RoaringBitmap;
use tracing::{Instrument, Span, trace};
use uuid::Uuid;

use crate::{
    runtime_metrics::op_stats::{self, OpStatsCollector},
    storage::StorageProvider,
    superfile::{SuperfileReader, builder::VectorConfig, vector::layout::VectorLayout},
    supertable::{
        error::QueryError,
        handle::SupertableReader,
        manifest::SuperfileEntry,
        query::{
            SuperfileHit,
            exec::common::{stamp_stable_ids, take_rows_byte_source, take_rows_object_store},
            superfile_reader::{OpenTierCounts, superfile_reader_tiered},
            vector::row_id_from_manifest_entry,
        },
        reader_cache::{DiskCacheStore, OpenTier, ReadIntent, SuperfileReaderCache},
        tombstones::SidecarCache,
    },
    utils::join::try_join_bounded,
};

/// Open one superfile's `SuperfileReader` through the reader cache.
/// Warm opens are in-memory cache hits (microseconds); cold opens
/// fetch the superfile header/footer from object storage. Always
/// `await`ed so the open I/O overlaps across the fan-out.
pub(crate) async fn open_reader(
    store: &Arc<dyn SuperfileReaderCache>,
    disk_cache: Option<&Arc<DiskCacheStore>>,
    storage: Option<&Arc<dyn StorageProvider>>,
    entry: &SuperfileEntry,
    intent: ReadIntent,
) -> Result<Arc<SuperfileReader>, QueryError> {
    open_reader_tiered(store, disk_cache, storage, entry, intent)
        .await
        .map(|(reader, _)| reader)
}

/// [`open_reader`], also saying which cache tier served the file, for a
/// fan-out that counts its opens. The span is `debug`: one per superfile is
/// too many to export with every query, so the fan-out's phase span carries
/// the counts instead.
#[cfg_attr(
    feature = "detailed-tracing",
    tracing::instrument(level = "debug", skip_all, fields(uri = ?entry.uri))
)]
async fn open_reader_tiered(
    store: &Arc<dyn SuperfileReaderCache>,
    disk_cache: Option<&Arc<DiskCacheStore>>,
    storage: Option<&Arc<dyn StorageProvider>>,
    entry: &SuperfileEntry,
    intent: ReadIntent,
) -> Result<(Arc<SuperfileReader>, OpenTier), QueryError> {
    superfile_reader_tiered(
        store,
        disk_cache,
        storage,
        &entry.uri,
        &entry.storage_path(),
        entry.subsection_offsets.as_ref(),
        intent,
    )
    .await
    .map_err(QueryError::store)
}

/// Verify that each configured vector column is present in this superfile and
/// that its stored rerank codec can actually serve the table.
///
/// The stored superfile codec is **authoritative**: scoring dispatches on each
/// superfile's own codec id (per unit), so any codec that was written is scored
/// correctly regardless of the table's *current* default. The default is not
/// persisted — reopen re-derives it from the metric — so it can differ from what
/// a table was created with (e.g. across a release that changes the default for
/// a metric), and a table may legitimately hold a mix of codecs across
/// generations. Comparing the stored codec against the re-derived default would
/// therefore reject valid data. Instead we reject only a stored codec that
/// genuinely cannot serve this table: one that does not support the table's
/// metric, or a non-IVF-mergeable codec inside a multi-cell (hidden-index) unit,
/// which requires mergeable codecs.
pub(crate) fn verify_superfile_vector_codecs(
    reader: &SuperfileReader,
    expected: &[VectorConfig],
) -> Result<(), QueryError> {
    // A file written before a vector column existed holds no index for it
    // and contributes nothing to a search on it; only the columns a file
    // does hold are checked.
    let Some(vector) = reader.vec() else {
        return Ok(());
    };
    for config in expected {
        for column in vector
            .vector_columns_config()
            .filter(|column| column.name == config.column)
        {
            let stored = column.rerank_codec;
            let usable = stored.supports_metric(config.metric)
                && (!vector.is_multi_cell() || stored.is_ivf_mergeable());
            if !usable {
                return Err(QueryError::Internal(format!(
                    "vector codec {} stored for {:?} cannot serve this table (metric {:?}{})",
                    stored.name(),
                    config.column,
                    config.metric,
                    if vector.is_multi_cell() {
                        "; a multi-cell unit requires an IVF-mergeable codec"
                    } else {
                        ""
                    },
                )));
            }
        }
    }
    Ok(())
}

/// Open one superfile for **compaction**, with its bytes locally available for
/// *synchronous* reads.
///
/// Compaction's Sq8 IVF merge reads each input's centroid/code subsection via
/// `VectorReader::try_get_range_sync` and its id column via
/// `SuperfileReader::record_batches` — both resolve straight off
/// locally-present bytes, never async I/O. The lazy query reader returned by
/// [`open_reader`] only exposes its bytes synchronously after a *background*
/// mmap promotion, so a compaction that races that promotion sees a reader
/// with no resident bytes (`record_batches` → `LazyReaderUnsupported`, and
/// `try_get_range_sync` → `None`) and fails. Force the disk cache to
/// mmap-promote the input first via [`DiskCacheStore::reader_synchronous_with_storage`]:
/// the bytes are NVMe-backed and OS-paged — bounded by the cache budget and the
/// `MADV_DONTNEED` sweep — so this does **not** pull whole superfiles into the
/// heap (the whole point of the streamed/mmap design). Falls back to the query
/// opener when no disk cache is configured.
pub(crate) async fn open_compaction_input(
    store: &Arc<dyn SuperfileReaderCache>,
    disk_cache: Option<&Arc<DiskCacheStore>>,
    storage: Option<&Arc<dyn StorageProvider>>,
    entry: &SuperfileEntry,
) -> Result<Arc<SuperfileReader>, QueryError> {
    if let Some(storage) = storage {
        if let Some(cache) = disk_cache {
            let reader = cache
                .reader_synchronous_with_storage(
                    &entry.uri,
                    &entry.storage_path(),
                    Arc::clone(storage),
                )
                .await
                .map_err(QueryError::store);
            // Fully-resident only: a promoted hybrid reader exposes parquet
            // bytes but leaves the vector blob sparse, and the Sq8 merge
            // below reads real vector bytes synchronously.
            if let Ok(reader) = reader
                && reader.is_fully_resident()
            {
                return Ok(reader);
            }
        }
        // Compaction needs synchronous Parquet/id-column access; if the hidden
        // table was opened without a disk cache, force an eager open here.
        let path = entry.storage_path();
        let (bytes, _) = storage.get(&path).await.map_err(QueryError::store)?;
        let reader = SuperfileReader::open(bytes)?;
        return Ok(Arc::new(reader));
    }
    // Compaction is not a query modality; allow fill so inputs can promote.
    open_reader(store, disk_cache, storage, entry, ReadIntent::Warm).await
}

/// Tag a kernel's results with their source and stamp stable ids immediately
/// when the manifest's contiguous span makes that translation arithmetic.
pub(crate) fn tag_hits(entry: &SuperfileEntry, hits: Vec<(u32, f32)>) -> Vec<SuperfileHit> {
    // Hoist the span check: `row_id_from_manifest_entry(entry, local)` is
    // `id_min + local` behind local-independent validity checks, so one
    // base lookup per unit stamps every hit with a single add. Stamping at
    // tag time (rather than post-selection) also spares the resolver a
    // per-URI manifest-entry lookup — the fan-out already holds the entry.
    let base = row_id_from_manifest_entry(entry, 0);
    hits.into_iter()
        .map(|(local_doc_id, score)| SuperfileHit {
            superfile: entry.uri,
            local_doc_id,
            score,
            stable_id: base.map(|b| b + i128::from(local_doc_id)),
        })
        .collect()
}

/// Resolve a superfile's tombstones to a non-empty deny bitmap, or `None`
/// when it has none. After the orchestrator's batched
/// [`SidecarCache::prefetch`] this is an in-memory cache hit. The single
/// source of the "look up the bitmap, treat empty as absent" step shared
/// by the post-rank filter here, the allow-set subtraction, and the
/// unfiltered deny-set pushdown.
pub(crate) fn tombstone_deny_set(
    cache: &SidecarCache,
    superfile_id: Uuid,
    now: Instant,
) -> Result<Option<Arc<RoaringBitmap>>, QueryError> {
    let bitmap = cache
        .bitmap_for(superfile_id, now)
        .map_err(QueryError::tombstone_cache)?;
    Ok((!bitmap.is_empty()).then_some(bitmap))
}

/// Drop tombstoned `local_doc_id`s from one superfile's hits — the
/// post-rank filter for query paths that rank without a deny set (FTS).
pub(crate) fn apply_tombstone_filter(
    cache: Option<&Arc<SidecarCache>>,
    entry: &SuperfileEntry,
    hits: &mut Vec<SuperfileHit>,
    now: Instant,
) -> Result<(), QueryError> {
    let Some(cache) = cache else {
        return Ok(());
    };
    let Some(bitmap) = tombstone_deny_set(cache, entry.superfile_id, now)? else {
        return Ok(());
    };
    hits.retain(|h| !bitmap.contains(h.local_doc_id));
    Ok(())
}

/// Attach stable `_id`s to tagged hits without paying a Parquet `_id` decode
/// when span arithmetic already answers.
///
/// Id-ordered user superfiles (FTS-only, and any non-MultiCell layout) map
/// `local → id_min + local` from the manifest. Cell-packed MultiCell files and
/// gapped spans fall through to [`stable_ids_for_tagged_hits`] (inline IVF
/// region, then resident `_id` pages). Callers can defer a lazy FTS `_id` read
/// until after global top-k selection, avoiding one decode per candidate
/// superfile. Skipping the Parquet take on the arithmetic path is what keeps
/// warm FTS at microseconds instead of tens of milliseconds — the same class
/// of bug as the SQL vector id-only fast path.
pub(crate) async fn attach_stable_ids(
    reader: &SuperfileReader,
    entry: &SuperfileEntry,
    hits: &mut [SuperfileHit],
    fetch_lazy_id_page: bool,
    op_stats: &Option<Arc<OpStatsCollector>>,
) -> Result<(), QueryError> {
    if hits.is_empty() {
        return Ok(());
    }
    if let Some(base) = row_id_from_manifest_entry(entry, 0) {
        for hit in hits.iter_mut() {
            hit.stable_id = Some(base + i128::from(hit.local_doc_id));
        }
        return Ok(());
    }
    let locals: Vec<u32> = hits.iter().map(|h| h.local_doc_id).collect();
    if let Some(ids) = stable_ids_for_tagged_hits(reader, &locals).await? {
        // The inline stable-id region is one planned range per file —
        // whether it resolved from residency, the cold ride-along stash,
        // or a fetch (the plan requests it identically either way).
        if let Some(stats) = op_stats {
            stats.add_planned_read_ranges(1);
        }
        for (hit, id) in hits.iter_mut().zip(ids) {
            hit.stable_id = Some(id);
        }
        return Ok(());
    }
    if !fetch_lazy_id_page {
        return Ok(());
    }
    let id_column = reader.id_column();
    // Sync decode when the reader holds resident parquet bytes (eager or
    // promoted hybrid) — the async stream take pays per-call setup that
    // dominates targeted id reads; only genuinely lazy readers await it.
    let batch = if reader.can_take_by_local_doc_ids() {
        let (batch, decode_ns) = op_stats::timed_section(|| {
            reader
                .take_by_local_doc_ids(&locals, &[id_column])
                .map_err(|error| QueryError::Internal(error.to_string()))
        });
        if let Some(stats) = op_stats {
            stats.add_kernel_cpu_ns(decode_ns);
        }
        batch?
    } else {
        // No caller fetches the lazy id page today; one that does decodes on
        // the global pool.
        take_rows_byte_source(reader, &locals, &[id_column], None)
            .await
            .map_err(|error| QueryError::Internal(error.to_string()))?
    };
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| QueryError::Internal("_id column missing".into()))?;
    for (hit, id) in hits.iter_mut().zip(ids.values()) {
        hit.stable_id = Some(*id);
    }
    Ok(())
}

/// Stamp a final hit set before row resolution.
///
/// Fan-out intentionally avoids fetching a lazy `_id` page for every candidate
/// superfile. Once the caller has reduced those candidates to its final result
/// set, this helper groups unresolved hits by URI and performs at most one
/// targeted `_id` read per hit-bearing file.
pub(crate) async fn attach_stable_ids_to_hits(
    table_reader: &SupertableReader,
    hits: &mut [SuperfileHit],
) -> Result<(), QueryError> {
    // Arithmetic-capable files were stamped at tag time ([`tag_hits`]);
    // only hits from cell-packed / gapped-span files arrive unresolved. Fill
    // them through the shared, cache-backed id resolution: inline id / span
    // arithmetic where possible, else a `_id`-column read via the decoded
    // scalar cache (so a warm reader decodes each superfile's `_id` column
    // once, not once per query). This replaces a per-superfile
    // `take_by_local_doc_ids`, whose scattered per-query Parquet page reads
    // dominated large-k scored latency on real corpora.
    stamp_stable_ids(table_reader, hits)
        .await
        .map_err(QueryError::DataFusion)?;
    if let Some(missing) = hits.iter().find(|h| h.stable_id.is_none()) {
        return Err(QueryError::Internal(format!(
            "hit {:?}/{} missing stable _id after search-wave stamping",
            missing.superfile, missing.local_doc_id
        )));
    }
    Ok(())
}

/// MultiCell IVF locals include boundary stubs and do not address Parquet
/// rows. Resolve the tombstone bitmap's Parquet locals to stable `_id`s, then
/// filter tagged IVF hits by identity. Non-MultiCell files keep the trusted
/// local-id fast path above.
pub(crate) async fn apply_resolved_tombstone_filter(
    reader: &SuperfileReader,
    storage: Option<&Arc<dyn StorageProvider>>,
    cache: Option<&Arc<SidecarCache>>,
    entry: &SuperfileEntry,
    hits: &mut Vec<SuperfileHit>,
    now: Instant,
    op_stats: &Option<Arc<OpStatsCollector>>,
) -> Result<(), QueryError> {
    if entry.vector_layout != VectorLayout::MultiCellIvf {
        return apply_tombstone_filter(cache, entry, hits, now);
    }
    let Some(cache) = cache else {
        return Ok(());
    };
    let bitmap = cache
        .bitmap_for(entry.superfile_id, now)
        .map_err(QueryError::tombstone_cache)?;
    if bitmap.is_empty() {
        return Ok(());
    }
    let locals: Vec<u32> = bitmap.iter().collect();
    let id_column = reader.id_column();
    let batch = if reader.parquet_bytes().is_some() {
        let (batch, decode_ns) = op_stats::timed_section(|| {
            reader
                .take_by_local_doc_ids(&locals, &[id_column])
                .map_err(QueryError::from)
        });
        if let Some(stats) = op_stats {
            stats.add_kernel_cpu_ns(decode_ns);
        }
        batch?
    } else {
        let storage = storage.ok_or_else(|| {
            QueryError::Internal(
                "MultiCell tombstone resolve needs resident bytes or storage".into(),
            )
        })?;
        let (object_store, path) = storage
            .object_store_handle(&entry.storage_path())
            .ok_or_else(|| QueryError::Internal("no object_store handle for superfile".into()))?;
        let file_size = entry
            .subsection_offsets
            .as_ref()
            .map(|offsets| offsets.total_size);
        take_rows_object_store(
            object_store,
            path,
            file_size,
            reader.schema(),
            reader.n_docs(),
            &locals,
            &[id_column],
        )
        .await
        .map_err(QueryError::DataFusion)?
    };
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| QueryError::Internal("_id column missing".into()))?;
    let deleted: HashSet<i128> = ids.values().iter().copied().collect();
    hits.retain(|hit| hit.stable_id.is_none_or(|id| !deleted.contains(&id)));
    Ok(())
}

/// Resolve stable user `_id`s for tagged hits from bytes already resident
/// on this superfile reader — inline IVF region first (materialized hidden
/// cells), then the scalar `_id` column (INCOMING staging / MultiCell).
/// `None` when the bytes are not yet mmap'd (cold lazy); [`attach_stable_ids`]
/// then performs the targeted object-store read before returning the hit set.
async fn stable_ids_for_tagged_hits(
    reader: &SuperfileReader,
    locals: &[u32],
) -> Result<Option<Vec<i128>>, QueryError> {
    if locals.is_empty() {
        return Ok(Some(Vec::new()));
    }
    if let Some(v) = reader.vec()
        && let Some(ids) = v.inline_stable_ids_for_locals(locals)
    {
        return Ok(Some(ids));
    }
    if let Some(v) = reader.vec()
        && let Some(ids) = v
            .inline_stable_ids_for_locals_async(locals)
            .await
            .map_err(QueryError::from)?
    {
        return Ok(Some(ids));
    }
    if locals
        .iter()
        .any(|&local| u64::from(local) >= reader.n_docs())
    {
        return Ok(None);
    }
    if reader.parquet_bytes().is_none() {
        return Ok(None);
    }
    let id_column = reader.id_column();
    let batch = reader
        .take_by_local_doc_ids(locals, &[id_column])
        .map_err(QueryError::from)?;
    let array = batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| QueryError::Internal("_id column missing".into()))?;
    Ok(Some(array.values().to_vec()))
}

/// Record one superfile open on the operation's stats, when metered.
fn note_superfile_opened(op_stats: Option<&Arc<OpStatsCollector>>) {
    if let Some(stats) = op_stats {
        stats.add_score_pruning_survived(1);
    }
}

/// What every unit of one fan-out shares: where readers are opened from,
/// the tombstone cache (warmed for the batch) and the instant it was warmed
/// at, and the operation's stats. Built once per fan-out; each unit then
/// opens its reader, checks its vector codecs, counts the open and runs the
/// body — inline on the current task, or on a task of its own.
#[derive(Clone)]
struct FanoutContext {
    store: Arc<dyn SuperfileReaderCache>,
    disk_cache: Option<Arc<DiskCacheStore>>,
    storage: Option<Arc<dyn StorageProvider>>,
    vector_columns: Arc<Vec<VectorConfig>>,
    tombstone_cache: Option<Arc<SidecarCache>>,
    op_stats: Option<Arc<OpStatsCollector>>,
    now: Instant,
    intent: ReadIntent,
    /// This fan-out's opens by cache tier, for the phase span that awaits
    /// it. `None` unless that span declares the tier fields.
    tiers: Option<Arc<OpenTierCounts>>,
}

impl FanoutContext {
    async fn new<P>(
        reader: &SupertableReader,
        units: &[(Arc<SuperfileEntry>, P)],
        prefetch_tombstones: bool,
        intent: ReadIntent,
        tiers: Option<Arc<OpenTierCounts>>,
    ) -> Self {
        let manifest = reader.manifest();
        let tombstone_cache = reader.tombstone_cache.clone();
        let now = Instant::now();
        // Warm the tombstone sidecars for every distinct superfile in one
        // concurrent batch before the per-superfile fan-out. Skipped by
        // callers whose tombstones are resolved elsewhere (the hidden path
        // filters via the resident deleted-set, so its per-cell sidecars are
        // always empty and prefetching them is a wasted wave of GETs on the
        // cold critical path).
        if prefetch_tombstones && let Some(cache) = tombstone_cache.as_ref() {
            let mut ids: Vec<Uuid> = units.iter().map(|(e, _)| e.superfile_id).collect();
            ids.sort_unstable();
            ids.dedup();
            cache.prefetch(&ids, now).await;
        }
        Self {
            store: Arc::clone(&manifest.options.store),
            disk_cache: manifest.options.disk_cache.as_ref().map(Arc::clone),
            storage: manifest.options.storage.as_ref().map(Arc::clone),
            vector_columns: Arc::new(manifest.vector_configs()),
            tombstone_cache,
            op_stats: reader.op_stats.clone(),
            now,
            intent,
            tiers,
        }
    }

    /// Open `entry`'s reader and run `body` on it, on the current task.
    async fn run<P, R, B, Fut>(
        &self,
        body: B,
        entry: Arc<SuperfileEntry>,
        params: P,
    ) -> Result<R, QueryError>
    where
        B: Fn(
            Arc<SuperfileReader>,
            Arc<SuperfileEntry>,
            Option<Arc<SidecarCache>>,
            Instant,
            P,
        ) -> Fut,
        Fut: Future<Output = Result<R, QueryError>>,
    {
        let (r, tier) = open_reader_tiered(
            &self.store,
            self.disk_cache.as_ref(),
            self.storage.as_ref(),
            &entry,
            self.intent,
        )
        .await?;
        if let Some(tiers) = &self.tiers {
            tiers.add(tier);
        }
        verify_superfile_vector_codecs(&r, &self.vector_columns)?;
        note_superfile_opened(self.op_stats.as_ref());
        body(r, entry, self.tombstone_cache.clone(), self.now, params).await
    }

    /// [`Self::run`] on its own task on the shared query runtime, so the
    /// units' cold opens overlap (see [`spawn_unit`]).
    fn spawn<P, R, B, Fut>(
        &self,
        body: &B,
        entry: Arc<SuperfileEntry>,
        params: P,
    ) -> impl Future<Output = Result<R, QueryError>>
    where
        P: Send + 'static,
        R: Send + 'static,
        B: Fn(
                Arc<SuperfileReader>,
                Arc<SuperfileEntry>,
                Option<Arc<SidecarCache>>,
                Instant,
                P,
            ) -> Fut
            + Clone
            + Send
            + 'static,
        Fut: Future<Output = Result<R, QueryError>> + Send + 'static,
    {
        let ctx = self.clone();
        let body = body.clone();
        spawn_unit(async move { ctx.run(body, entry, params).await }.in_current_span())
    }
}

/// `unit` on its own task on the shared query runtime, aborted if the returned
/// future is dropped first: when a collecting caller short-circuits on another
/// unit's error, or the statement driving the fan-out is cancelled (a SQL
/// statement refused for memory). A bare `tokio::spawn` handle would leave the
/// task fetching and decoding after its caller had gone. The join error is
/// flattened into a `QueryError`, so a collecting caller short-circuits on
/// the first failing superfile.
fn spawn_unit<R: Send + 'static>(
    unit: impl Future<Output = Result<R, QueryError>> + Send + 'static,
) -> impl Future<Output = Result<R, QueryError>> {
    let task = SpawnedTask::spawn(unit);
    async move {
        task.await
            .map_err(|e| QueryError::Store(format!("fan-out task join: {e}")))?
    }
}

/// Lower-level fan-out primitive: the shared orchestration behind
/// [`fanout`] and the count path, generic over the per-superfile result
/// `R`.
///
/// It warms the tombstone sidecar cache for every distinct superfile in
/// one batch, then spawns one task per unit on the shared query runtime,
/// each opening its reader and running `body`, at most
/// `superfiles_in_flight` at a time. Results come back in unit order, and
/// the **first** per-superfile error (in time, not spawn order)
/// short-circuits the whole fan-out: it returns early and aborts the units
/// still running (see [`spawn_unit`]).
///
/// `body` runs inside each task with the opened reader, the superfile
/// entry, the (warmed) tombstone cache + the batch `now` instant, and
/// the unit's params. Resolving the per-superfile tombstone bitmap and
/// applying it is the body's job, since callers differ: [`fanout`]
/// tags + retains hits, while the count path either takes the O(1)
/// `term_df` fast path (no tombstones) or counts the matching ids minus
/// tombstones.
///
/// Awaited under a `tiered_span!`, it records on that span how many of
/// its opens each cache tier served. Under any other span it counts
/// nothing.
pub(crate) async fn fanout_with<P, R, B, Fut>(
    reader: &SupertableReader,
    units: Vec<(Arc<SuperfileEntry>, P)>,
    prefetch_tombstones: bool,
    intent: ReadIntent,
    body: B,
) -> Result<Vec<R>, QueryError>
where
    P: Send + 'static,
    R: Send + 'static,
    B: Fn(Arc<SuperfileReader>, Arc<SuperfileEntry>, Option<Arc<SidecarCache>>, Instant, P) -> Fut
        + Clone
        + Send
        + 'static,
    Fut: Future<Output = Result<R, QueryError>> + Send + 'static,
{
    let tiers = OpenTierCounts::for_current_span();
    let out = fanout_with_counted(
        reader,
        units,
        prefetch_tombstones,
        intent,
        body,
        tiers.clone(),
    )
    .await?;
    if let Some(tiers) = &tiers {
        tiers.record_on(&Span::current());
    }
    Ok(out)
}

/// [`fanout_with`], counting its opens into `tiers` without recording them,
/// for a phase that fans out in waves and records the total once.
pub(crate) async fn fanout_with_counted<P, R, B, Fut>(
    reader: &SupertableReader,
    units: Vec<(Arc<SuperfileEntry>, P)>,
    prefetch_tombstones: bool,
    intent: ReadIntent,
    body: B,
    tiers: Option<Arc<OpenTierCounts>>,
) -> Result<Vec<R>, QueryError>
where
    P: Send + 'static,
    R: Send + 'static,
    B: Fn(Arc<SuperfileReader>, Arc<SuperfileEntry>, Option<Arc<SidecarCache>>, Instant, P) -> Fut
        + Clone
        + Send
        + 'static,
    Fut: Future<Output = Result<R, QueryError>> + Send + 'static,
{
    if units.is_empty() {
        return Ok(Vec::new());
    }
    trace!(units = units.len(), "fanning query out across superfiles");
    let ctx = FanoutContext::new(reader, &units, prefetch_tombstones, intent, tiers).await;

    // Single unit (the common case for a compacted, single-superfile
    // table): run the body inline on the current task. `tokio::spawn`
    // here would only add a thread handoff and a join with nothing to
    // overlap against — the spawn path's win is concurrency across units,
    // which doesn't exist at one unit. Semantically identical to the
    // fan-out below with a one-element result.
    if units.len() == 1 {
        let (entry, params) = units.into_iter().next().expect("len == 1");
        return Ok(vec![ctx.run(body, entry, params).await?]);
    }
    try_join_bounded(
        units
            .into_iter()
            .map(|(entry, params)| ctx.spawn(&body, entry, params)),
        reader.options().superfiles_in_flight,
    )
    .await
}

/// [`fanout_with`] for units the caller has ordered by how much they can
/// still contribute: at most `window` units are in flight, units start in
/// the given order, and `skip_before_open` is asked about each unit just
/// before its superfile would be opened — so a unit that can no longer
/// change the outcome (its score ceiling is below the running k-th score)
/// is never opened at all. Results come back in completion order, which
/// the callers here merge order-independently.
///
/// The window is the trade: strictly sequential opens maximise skips but
/// serialise I/O; a wider window keeps the CPU busy while the first results
/// raise the floor for the rest.
///
/// Records its opens by tier as [`fanout_with`] does. A skipped unit is
/// never opened, so it is not counted.
/// The units of a fan-out: each superfile (or a slice of one) with the
/// parameters its kernel runs under.
/// Opens issued in all ahead of a plan that is still resolving. Enough to
/// overlap the plan's round trip with real work; few enough that a plan
/// which keeps a dozen superfiles out of a thousand has wasted little. The
/// full window applies once the plan is in, or where there is none.
/// Measured at a billion documents against object storage: a full window
/// issued ahead of the plan opened sixty-four superfiles a merge then ruled
/// out, more bytes than the whole pruned query; a window of eight that
/// refilled as opens finished issued fifty.
const PENDING_PLAN_WINDOW: usize = 8;

/// Windows of units a fan-out must have for the cap above to apply. A plan
/// is worth waiting for where it can rule out more than a window or two of
/// opens; on a table of a few windows the opens that would run ahead of it
/// are most of the query, and holding them costs the plan's round trip for
/// nothing. Measured at a hundred million documents, 112 superfiles: the
/// cap cost 13% at k=100, where the merge rarely prunes.
const PENDING_PLAN_MIN_WINDOWS: usize = 4;

/// Opens in flight until the first of them have answered. The running floor
/// is what lets a kernel skip posting blocks and the fan-out skip whole
/// superfiles, and it forms from the first answers; a window opened wide
/// before any answer scores every superfile against no floor at all. A
/// billion documents measured: the full window from the start read three
/// to four times the postings of a query the merge does not serve, for no
/// gain in its wall. The configured window applies once this many have
/// answered.
const FLOOR_WINDOW: usize = 64;

pub(crate) type Units<P> = Vec<(Arc<SuperfileEntry>, P)>;
/// What a plan that arrives while a fan-out runs does to the units not yet
/// issued: given them, it returns the ones to go on with in the order to
/// issue them, and results settled without an open — already folded into
/// whatever the caller folds results into, so the driver only collects them.
pub(crate) type Replan<'a, P, R> = Box<dyn FnOnce(Units<P>) -> (Units<P>, Vec<R>) + Send + 'a>;
/// A plan in the making, resolving beside the opens it may change; `None`
/// when it has nothing to apply.
pub(crate) type ReplanFuture<'a, P, R> =
    Pin<Box<dyn Future<Output = Option<Replan<'a, P, R>>> + Send + 'a>>;

/// The ordered fan-out, with an optional plan that arrives while it runs:
/// the units are issued in the order given, `window` at a time, and when
/// `replan` resolves the units not yet issued are handed to it and replaced
/// by what it returns. A plan still unresolved when the last open finishes
/// is dropped unawaited.
pub(crate) async fn fanout_with_ordered<'a, P, R, B, Fut, S>(
    reader: &SupertableReader,
    units: Vec<(Arc<SuperfileEntry>, P)>,
    window: usize,
    skip_before_open: S,
    body: B,
    mut on_result: impl FnMut(&R),
    replan: Option<ReplanFuture<'a, P, R>>,
) -> Result<Vec<R>, QueryError>
where
    P: Send + 'static,
    R: Send + 'static,
    S: Fn(&P) -> bool,
    B: Fn(Arc<SuperfileReader>, Arc<SuperfileEntry>, Option<Arc<SidecarCache>>, Instant, P) -> Fut
        + Clone
        + Send
        + 'static,
    Fut: Future<Output = Result<R, QueryError>> + Send + 'static,
{
    let tiers = OpenTierCounts::for_current_span();
    // Nothing to open — the index answered for every superfile, or the
    // floor ruled them all out — is still an outcome the phase reports.
    if units.is_empty() {
        if let Some(tiers) = &tiers {
            tiers.record_on(&Span::current());
        }
        return Ok(Vec::new());
    }
    trace!(
        units = units.len(),
        window, "fanning query out across superfiles in ceiling order"
    );
    let ctx = FanoutContext::new(reader, &units, true, reader.open_intent(), tiers.clone()).await;
    let started = Instant::now();
    // One unit and no plan to wait for: inline, as in [`fanout_with`];
    // nothing to overlap against.
    if units.len() == 1 && replan.is_none() {
        let (entry, params) = units.into_iter().next().expect("len == 1");
        let out = match skip_before_open(&params) {
            true => Vec::new(),
            false => vec![ctx.run(body, entry, params).await?],
        };
        if let Some(r) = out.first() {
            on_result(r);
            reader.note_open_cost(started.elapsed().as_nanos() as u64);
        }
        if let Some(tiers) = &tiers {
            tiers.record_on(&Span::current());
        }
        return Ok(out);
    }
    let mut in_flight: FuturesUnordered<_> = FuturesUnordered::new();
    let mut out = Vec::with_capacity(units.len());
    let mut opened = 0u64;
    let mut pending: VecDeque<(Arc<SuperfileEntry>, P)> = units.into();
    let mut replan = replan;
    let window = window.max(1);
    let cap_ahead_of_plan = pending.len() > PENDING_PLAN_MIN_WINDOWS * window;
    loop {
        let issue = match (
            replan.is_some() && cap_ahead_of_plan,
            out.len() < FLOOR_WINDOW,
        ) {
            (true, _) => PENDING_PLAN_WINDOW
                .saturating_sub(opened as usize)
                .min(window),
            (false, true) => FLOOR_WINDOW.min(window),
            (false, false) => window,
        };
        while in_flight.len() < issue {
            let Some((entry, params)) = pending.pop_front() else {
                break;
            };
            if skip_before_open(&params) {
                continue;
            }
            in_flight.push(ctx.spawn(&body, entry, params));
            opened += 1;
        }
        // The plan is polled beside the opens for as long as any run: one
        // that reports once every unit is issued changes no open, but what
        // it read is in memory for the next query of its terms. Once the
        // opens are done it is dropped unawaited — unless units still wait
        // on it, which the select below then awaits alone.
        if in_flight.is_empty() && (pending.is_empty() || replan.is_none()) {
            break;
        }
        // A finished unit is seen here, on the one task that drives the
        // fan-out, before the next unit starts: what the caller folds from
        // it (a floor from its scores) needs no lock the units would queue
        // on. The plan, when it resolves, is applied here too, before the
        // next unit is issued; it comes first when both are ready.
        tokio::select! {
            biased;
            plan = async {
                match replan.as_mut() {
                    Some(plan) => plan.await,
                    None => None,
                }
            }, if replan.is_some() => {
                replan = None;
                if let Some(plan) = plan {
                    let (keep, settled) = plan(pending.drain(..).collect());
                    pending = keep.into();
                    out.extend(settled);
                }
            }
            next = in_flight.next(), if !in_flight.is_empty() => {
                if let Some(result) = next {
                    let r = result?;
                    on_result(&r);
                    out.push(r);
                }
            }
        }
    }
    // What an open cost this query: the fan-out's wall over the opens it
    // made, the saving a tier merge can weigh a round's exclusions at.
    if let Some(per_open) = (started.elapsed().as_nanos() as u64).checked_div(opened) {
        reader.note_open_cost(per_open);
    }
    if let Some(tiers) = &tiers {
        tiers.record_on(&Span::current());
    }
    Ok(out)
}

/// Fan out a kernel whose local ids are Parquet-local (FTS and exact-match).
///
/// These hits can apply ordinary tombstones directly and defer stable-id
/// stamping until after global top-k selection. Vector MultiCell hits use
/// [`fanout`] instead because their local ids include cell-ordering and
/// boundary stubs.
pub(crate) async fn fanout_local_hits<P, K, Fut>(
    reader: &SupertableReader,
    units: Vec<(Arc<SuperfileEntry>, P)>,
    kernel: K,
) -> Result<Vec<Vec<SuperfileHit>>, QueryError>
where
    P: Send + 'static,
    K: Fn(Arc<SuperfileReader>, P) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<Vec<(u32, f32)>, QueryError>> + Send + 'static,
{
    fanout_with(
        reader,
        units,
        true,
        reader.open_intent(),
        local_hits_body(kernel),
    )
    .await
}

/// [`fanout_local_hits`] in ceiling order — see [`fanout_with_ordered`].
pub(crate) async fn fanout_local_hits_ordered<'a, P, K, Fut, S>(
    reader: &SupertableReader,
    units: Vec<(Arc<SuperfileEntry>, P)>,
    window: usize,
    skip_before_open: S,
    kernel: K,
    on_result: impl FnMut(&Vec<SuperfileHit>),
    replan: Option<ReplanFuture<'a, P, Vec<SuperfileHit>>>,
) -> Result<Vec<Vec<SuperfileHit>>, QueryError>
where
    P: Send + 'static,
    S: Fn(&P) -> bool,
    K: Fn(Arc<SuperfileReader>, P) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<Vec<(u32, f32)>, QueryError>> + Send + 'static,
{
    fanout_with_ordered(
        reader,
        units,
        window,
        skip_before_open,
        local_hits_body(kernel),
        on_result,
        replan,
    )
    .await
}

/// The per-unit body of the local-hits fan-outs: run the kernel, tag its
/// hits with the superfile, and drop the tombstoned ones.
fn local_hits_body<P, K, Fut>(
    kernel: K,
) -> impl Fn(
    Arc<SuperfileReader>,
    Arc<SuperfileEntry>,
    Option<Arc<SidecarCache>>,
    Instant,
    P,
) -> std::pin::Pin<Box<dyn Future<Output = Result<Vec<SuperfileHit>, QueryError>> + Send>>
+ Clone
where
    P: Send + 'static,
    K: Fn(Arc<SuperfileReader>, P) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<Vec<(u32, f32)>, QueryError>> + Send + 'static,
{
    move |r, entry, tombstone_cache, now, params| {
        let kernel = kernel.clone();
        Box::pin(async move {
            let hits = kernel(r, params).await?;
            let mut tagged = tag_hits(&entry, hits);
            apply_tombstone_filter(tombstone_cache.as_ref(), &entry, &mut tagged, now)?;
            Ok::<Vec<SuperfileHit>, QueryError>(tagged)
        })
    }
}

#[cfg(test)]
mod codec_verify_tests {
    use std::sync::Arc;

    use arrow_array::RecordBatch;
    use arrow_schema::Schema;
    use bytes::Bytes;

    use super::verify_superfile_vector_codecs;
    use crate::{
        superfile::{
            SuperfileReader,
            builder::{BuilderOptions, SuperfileBuilder, VectorConfig},
            vector::{distance::Metric, rerank_codec::RerankCodec},
        },
        test_helpers::{decimal128_id_field, decimal128_ids, default_vector_config},
    };

    const DIM: usize = 16;

    fn cfg(metric: Metric, codec: RerankCodec) -> VectorConfig {
        VectorConfig {
            metric,
            ..default_vector_config("emb", 7).with_rerank_codec(codec)
        }
    }

    /// Build one tiny single superfile whose "emb" column is written with
    /// `codec` under `metric`, and open it as a reader.
    fn build_reader(metric: Metric, codec: RerankCodec) -> SuperfileReader {
        let schema = Arc::new(Schema::new(vec![decimal128_id_field("doc_id")]));
        let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![cfg(metric, codec)]);
        let mut b = SuperfileBuilder::new(opts).expect("new builder");
        let batch = RecordBatch::try_new(schema, vec![Arc::new(decimal128_ids(vec![10u64, 11]))])
            .expect("batch");
        let mut v = vec![0.0f32; 2 * DIM]; // two rows, each on a distinct axis
        v[0] = 1.0;
        v[DIM + 1] = 1.0;
        b.add_batch(&batch, &[v.as_slice()]).expect("add_batch");
        SuperfileReader::open(Bytes::from(b.finish().expect("finish"))).expect("open")
    }

    /// Regression for the backward-compat break: a table written with
    /// `Sq8Residual` (the pre-flip L2/NegDot default) must still verify — and so
    /// serve — when reopened under code whose re-derived default is now
    /// `Sq16Adaptive`. The catalog does not persist the rerank codec, so reopen
    /// rebuilds the config from the metric with the *current* default; the stored
    /// superfile codec is authoritative, so the verifier must accept it.
    #[test]
    fn reopen_after_default_flip_accepts_stored_codec() {
        let reader = build_reader(Metric::L2Sq, RerankCodec::Sq8Residual);
        let expected = [cfg(Metric::L2Sq, RerankCodec::Sq16Adaptive)];
        verify_superfile_vector_codecs(&reader, &expected)
            .expect("stored Sq8Residual must be accepted under the Sq16Adaptive default");
    }

    /// The relaxation trusts *valid* stored codecs, not invalid ones: a stored
    /// codec that cannot serve the table's metric (a fixed-grid cosine-only
    /// `Sq16` under an L2Sq config) is still rejected.
    #[test]
    fn verifier_rejects_metric_incompatible_stored_codec() {
        let reader = build_reader(Metric::Cosine, RerankCodec::Sq16);
        let expected = [cfg(Metric::L2Sq, RerankCodec::Sq16Adaptive)];
        assert!(
            verify_superfile_vector_codecs(&reader, &expected).is_err(),
            "a cosine-only Sq16 codec must not be accepted for an L2Sq table"
        );
    }
}

#[cfg(test)]
mod fanout_tier_tests {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    use tempfile::TempDir;
    use tokio::{
        sync::Semaphore,
        time::{Instant, sleep},
    };

    use super::*;
    use crate::{
        storage::LocalFsStorageProvider,
        supertable::Supertable,
        test_helpers::{build_title_batch, default_supertable_options},
    };

    /// Commits in the fixture, one superfile each.
    const COMMITS: u64 = 3;

    /// Fan out over every superfile `table` lists with a body that does
    /// nothing, and return the opens by tier.
    async fn fanout_tiers(table: &Supertable) -> [u64; 6] {
        let reader = table.reader().expect("reader");
        let units: Vec<(Arc<SuperfileEntry>, ())> = reader
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|entry| (Arc::clone(entry), ()))
            .collect();
        let tiers = Arc::new(OpenTierCounts::default());
        let out = fanout_with_counted(
            &reader,
            units,
            false,
            ReadIntent::Warm,
            |_, _, _, _, ()| async { Ok::<(), QueryError>(()) },
            Some(Arc::clone(&tiers)),
        )
        .await
        .expect("fan-out");
        assert_eq!(out.len() as u64, COMMITS, "every unit ran");
        tiers.snapshot()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_fanout_counts_each_open_in_the_tier_that_served_it() {
        // Three superfiles, one unit each, so each is counted once, in the
        // tier that served it:
        //  - the writer leaves what it commits in the in-memory reader
        //    cache, so the table that wrote them opens all three from memory.
        //  - a table reopened on the same storage has nothing cached and no
        //    disk cache, so all three are opened from storage by range: the
        //    lazy tier.
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = || default_supertable_options().with_storage(Arc::clone(&storage));
        let table = Supertable::create(options()).expect("create");
        for title in ["alpha", "bravo", "charlie"] {
            let mut writer = table.writer().expect("writer");
            writer.append(&build_title_batch(&[title])).expect("append");
            writer.commit().expect("commit");
        }
        assert_eq!(fanout_tiers(&table).await, [COMMITS, 0, 0, 0, 0, 0]);

        let reopened = Supertable::open(options()).expect("reopen");
        assert_eq!(fanout_tiers(&reopened).await, [0, 0, COMMITS, 0, 0, 0]);
    }

    /// Superfiles in the cap fixture, one unit each: more than [`FANOUT_CAP`].
    const CAPPED_COMMITS: usize = 12;
    /// The cap on superfiles in flight the cap fixture is opened with.
    const FANOUT_CAP: usize = 3;
    /// The least time each unit's body runs, so a fan-out that ignored its
    /// cap would start more units than the cap while the first still run.
    const UNIT_TIME: Duration = Duration::from_millis(20);
    /// How long a unit waits for the units it expects beside it. Only a cap
    /// tighter than expected makes it wait this long.
    const OVERLAP_WAIT: Duration = Duration::from_secs(5);
    /// How often a waiting unit looks again.
    const OVERLAP_POLL: Duration = Duration::from_millis(1);

    /// The most units of a fan-out over every superfile on `storage` running
    /// at once, on a table opened with `cap` superfiles in flight (the
    /// default when `None`). Each unit holds its slot until `expected` have
    /// run at once, so the peak does not depend on how quickly the opens
    /// before the units finish.
    async fn peak_units(
        storage: &Arc<dyn StorageProvider>,
        cap: Option<usize>,
        expected: usize,
    ) -> usize {
        let mut options = default_supertable_options().with_storage(Arc::clone(storage));
        if let Some(cap) = cap {
            options = options.with_superfiles_in_flight(cap);
        }
        let table = Supertable::open(options).expect("open");
        let reader = table.reader().expect("reader");
        let units: Vec<(Arc<SuperfileEntry>, ())> = reader
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|entry| (Arc::clone(entry), ()))
            .collect();
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let body = {
            let running = Arc::clone(&running);
            let peak = Arc::clone(&peak);
            move |_, _, _, _, ()| {
                let running = Arc::clone(&running);
                let peak = Arc::clone(&peak);
                async move {
                    let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    sleep(UNIT_TIME).await;
                    let deadline = Instant::now() + OVERLAP_WAIT;
                    while peak.load(Ordering::SeqCst) < expected && Instant::now() < deadline {
                        sleep(OVERLAP_POLL).await;
                    }
                    running.fetch_sub(1, Ordering::SeqCst);
                    Ok::<(), QueryError>(())
                }
            }
        };
        let out = fanout_with(&reader, units, false, ReadIntent::Warm, body)
            .await
            .expect("fan-out");
        assert_eq!(out.len(), CAPPED_COMMITS, "every unit ran");
        peak.load(Ordering::SeqCst)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_fanout_runs_at_most_the_cap_of_superfiles_at_once() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let table =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        for i in 0..CAPPED_COMMITS {
            let mut writer = table.writer().expect("writer");
            writer
                .append(&build_title_batch(&[&format!("title {i}")]))
                .expect("append");
            writer.commit().expect("commit");
        }
        assert_eq!(
            peak_units(&storage, None, CAPPED_COMMITS).await,
            CAPPED_COMMITS,
            "under the default cap every unit runs at once, so the cap has work to do"
        );
        assert_eq!(
            peak_units(&storage, Some(FANOUT_CAP), FANOUT_CAP).await,
            FANOUT_CAP,
            "no more units than the cap run at once, and the fan-out reaches it"
        );
    }

    /// Superfiles in the plan fixture: more than four windows of them, so the
    /// cap ahead of a plan applies, and more than the cap lets through.
    const PLANNED_COMMITS: usize = 40;
    /// How long the fixture's plan takes to arrive: long after the opens
    /// issued ahead of it, which do nothing, have finished.
    const PLAN_DELAY: Duration = Duration::from_millis(100);
    /// A window the fixture exceeds four times over, no narrower than the
    /// cap, so the cap is what holds.
    const WIDE_WINDOW: usize = PENDING_PLAN_WINDOW;

    /// While a plan is still to come only a few opens run ahead of it, and
    /// the fan-out waits for the plan rather than ending when those finish:
    /// every unit the plan keeps is answered. The opens hold until the plan
    /// is in, as a cold open outlasts the plan's round trip, so what ran
    /// ahead of the plan is exactly what the window let through.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_pending_plan_holds_the_fanout_to_a_few_opens_until_it_arrives() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let table =
            Supertable::create(default_supertable_options().with_storage(storage)).expect("create");
        for i in 0..PLANNED_COMMITS {
            let mut writer = table.writer().expect("writer");
            writer
                .append(&build_title_batch(&[&format!("title {i}")]))
                .expect("append");
            writer.commit().expect("commit");
        }
        let reader = table.reader().expect("reader");
        let units: Vec<(Arc<SuperfileEntry>, ())> = reader
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|entry| (Arc::clone(entry), ()))
            .collect();
        assert_eq!(units.len(), PLANNED_COMMITS);
        let opened = Arc::new(AtomicUsize::new(0));
        let opened_before_plan = Arc::new(AtomicUsize::new(0));
        // Opens wait on this until the plan is in; the plan then lets every
        // unit through.
        let gate = Arc::new(Semaphore::new(0));
        let plan: ReplanFuture<'_, (), ()> = {
            let opened = Arc::clone(&opened);
            let opened_before_plan = Arc::clone(&opened_before_plan);
            let gate = Arc::clone(&gate);
            Box::pin(async move {
                tokio::time::sleep(PLAN_DELAY).await;
                opened_before_plan.store(opened.load(Ordering::Acquire), Ordering::Release);
                gate.add_permits(PLANNED_COMMITS);
                let keep_all: Replan<'_, (), ()> = Box::new(|units| (units, Vec::new()));
                Some(keep_all)
            })
        };
        let body = {
            let opened = Arc::clone(&opened);
            let gate = Arc::clone(&gate);
            move |_, _, _, _, ()| {
                opened.fetch_add(1, Ordering::AcqRel);
                let gate = Arc::clone(&gate);
                async move {
                    gate.acquire().await.expect("gate open").forget();
                    Ok::<(), QueryError>(())
                }
            }
        };
        let out = fanout_with_ordered(
            &reader,
            units,
            WIDE_WINDOW,
            |_: &()| false,
            body,
            |_: &()| {},
            Some(plan),
        )
        .await
        .expect("fan-out");
        assert_eq!(out.len(), PLANNED_COMMITS, "every unit the plan kept ran");
        assert_eq!(
            opened_before_plan.load(Ordering::Acquire),
            PENDING_PLAN_WINDOW,
            "only the pending-plan window ran ahead of the plan"
        );
    }
}

#[cfg(test)]
mod spawn_unit_tests {
    use std::{future::pending, time::Duration};

    use tokio::{sync::oneshot, time::timeout};

    use super::*;

    /// Ample time for an aborted task to be dropped by the runtime.
    const ABORT_WAIT: Duration = Duration::from_secs(5);

    /// Sends on its channel when dropped: the unit's future was dropped, which
    /// for a task that never finishes means it was aborted.
    struct SignalOnDrop(Option<oneshot::Sender<()>>);

    impl Drop for SignalOnDrop {
        fn drop(&mut self) {
            if let Some(signal) = self.0.take() {
                let _ = signal.send(());
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dropping_a_unit_aborts_its_task() {
        let (started_tx, started) = oneshot::channel();
        let (dropped_tx, dropped) = oneshot::channel();
        let unit = spawn_unit(async move {
            let _signal = SignalOnDrop(Some(dropped_tx));
            let _ = started_tx.send(());
            pending::<Result<(), QueryError>>().await
        });
        started.await.expect("the unit's task started");
        drop(unit);
        timeout(ABORT_WAIT, dropped)
            .await
            .expect("dropping the unit aborts its task instead of leaving it running")
            .expect("the task's future was dropped");
    }
}
