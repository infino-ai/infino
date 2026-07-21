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
//!   * **tokio owns the fan-out and I/O.** One `tokio::spawn` task per
//!     work unit: each opens its superfile reader and runs the kernel,
//!     so superfile opens and cold object-store range GETs across
//!     hundreds of superfiles are all in flight at once on the shared
//!     multi-thread query runtime.
//!   * **CPU model is per-kernel, not uniform.** The vector kernel
//!     parallelizes its own scoring + rerank with `par_iter` (see
//!     `superfile/vector/reader.rs`). The FTS BMW/MaxScore kernel
//!     scores **serially inside its tokio task** — there is no rayon in
//!     the FTS scoring path. Intra-superfile FTS parallelism is instead
//!     expressed as additional tokio work units (doc-id sub-ranges; see
//!     `query/fts.rs`).
//!
//! The per-superfile merge (top-k ascending for vector distance,
//! descending for BM25 relevance) stays with each caller; this layer
//! returns the per-unit tagged+filtered hit lists.

use std::{collections::HashSet, future::Future, sync::Arc, time::Instant};

use arrow_array::Decimal128Array;
use futures::future::try_join_all;
use roaring::RoaringBitmap;
use tracing::trace;
use uuid::Uuid;

use super::SuperfileHit;
use crate::{
    storage::StorageProvider,
    superfile::{
        SuperfileReader,
        builder::VectorConfig,
        vector::{layout::VectorLayout, rerank_codec::RerankCodec},
    },
    supertable::{
        error::QueryError,
        handle::SupertableReader,
        manifest::SuperfileEntry,
        query::{
            exec::common::{take_rows_byte_source, take_rows_object_store},
            superfile_reader::superfile_reader,
            vector::row_id_from_manifest_entry,
        },
        reader_cache::{DiskCacheStore, SuperfileReaderCache},
        tombstones::SidecarCache,
    },
};

/// Open one superfile's `SuperfileReader` through the reader cache.
/// Warm opens are in-memory cache hits (microseconds); cold opens
/// fetch the superfile header/footer from object storage. Always
/// `await`ed so the open I/O overlaps across the fan-out.
#[cfg_attr(
    feature = "detailed-tracing",
    tracing::instrument(skip_all, fields(uri = ?entry.uri))
)]
pub(crate) async fn open_reader(
    store: &Arc<dyn SuperfileReaderCache>,
    disk_cache: Option<&Arc<DiskCacheStore>>,
    storage: Option<&Arc<dyn StorageProvider>>,
    entry: &SuperfileEntry,
    allow_background_fill: bool,
) -> Result<Arc<SuperfileReader>, QueryError> {
    superfile_reader(
        store,
        disk_cache,
        storage,
        &entry.uri,
        entry.subsection_offsets.as_ref(),
        allow_background_fill,
    )
    .await
    .map_err(|e| QueryError::Store(e.to_string()))
}

/// Verify that parsed on-disk vector codecs match this table's write options.
pub(crate) fn verify_superfile_vector_codecs(
    reader: &SuperfileReader,
    expected: &[VectorConfig],
) -> Result<(), QueryError> {
    if expected.is_empty() {
        return Ok(());
    }
    let vector = reader.vec().ok_or_else(|| {
        QueryError::Execute("superfile is missing configured vector index".into())
    })?;
    for config in expected {
        let expected_codec =
            if vector.is_multi_cell() && !config.rerank_codec.is_sq8_residual_family() {
                RerankCodec::Sq8Residual
            } else {
                config.rerank_codec
            };
        let mut matched = false;
        for column in vector
            .vector_columns_config()
            .filter(|column| column.name == config.column)
        {
            matched = true;
            if column.rerank_codec != expected_codec {
                return Err(QueryError::Execute(format!(
                    "vector codec mismatch for {:?}: table expects {}, superfile stores {}",
                    config.column,
                    expected_codec.name(),
                    column.rerank_codec.name()
                )));
            }
        }
        if !matched {
            return Err(QueryError::Execute(format!(
                "superfile is missing configured vector column {:?}",
                config.column
            )));
        }
    }
    Ok(())
}

/// Open one superfile for **compaction**, with its bytes locally available for
/// *synchronous* reads.
///
/// Compaction's Sq8 IVF merge reads each input's centroid/code subsection via
/// `VectorReader::try_get_range_sync` and its id column via
/// `SuperfileReader::get_record_batch` — both resolve straight off
/// locally-present bytes, never async I/O. The lazy query reader returned by
/// [`open_reader`] only exposes its bytes synchronously after a *background*
/// mmap promotion, so a compaction that races that promotion sees a reader
/// with no resident bytes (`get_record_batch` → `LazyReaderUnsupported`, and
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
                .reader_synchronous_with_storage(&entry.uri, Arc::clone(storage))
                .await
                .map_err(|e| QueryError::Store(e.to_string()));
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
        let path = entry.uri.storage_path();
        let (bytes, _) = storage
            .get(&path)
            .await
            .map_err(|e| QueryError::Store(e.to_string()))?;
        let reader = SuperfileReader::open(bytes).map_err(|e| QueryError::Store(e.to_string()))?;
        return Ok(Arc::new(reader));
    }
    // Compaction is not a query modality; allow fill so inputs can promote.
    open_reader(store, disk_cache, storage, entry, true).await
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
        .map_err(|e| QueryError::Store(format!("tombstone cache: {e}")))?;
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
        reader
            .take_by_local_doc_ids(&locals, &[id_column])
            .map_err(|error| QueryError::Execute(error.to_string()))?
    } else {
        take_rows_byte_source(reader, &locals, &[id_column])
            .await
            .map_err(|error| QueryError::Execute(error.to_string()))?
    };
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| QueryError::Execute("_id column missing".into()))?;
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
    // only hits from cell-packed / gapped-span files arrive unresolved. On
    // such tables that is EVERY hit, so this must not copy hits: process
    // contiguous same-file runs in place (unranked hits arrive file-grouped
    // from the fan-out; ranked top-k interleaves, but is top-k-sized). Per
    // run: one manifest lookup, one reader open, one in-place stamp.
    let manifest = Arc::clone(table_reader.manifest());
    let store = Arc::clone(&manifest.options.store);
    let disk_cache = manifest.options.disk_cache.clone();
    let storage = manifest.options.storage.clone();
    let mut start = 0usize;
    while start < hits.len() {
        if hits[start].stable_id.is_some() {
            start += 1;
            continue;
        }
        let uri = hits[start].superfile;
        let mut end = start + 1;
        while end < hits.len() && hits[end].superfile == uri && hits[end].stable_id.is_none() {
            end += 1;
        }
        let entry = manifest
            .lookup_superfile_entry(uri)
            .await
            .map_err(QueryError::ManifestLoad)?
            .ok_or_else(|| {
                QueryError::Execute(format!("hit superfile {uri:?} missing from manifest"))
            })?;
        // FTS post-topk id stamp — allow fill (same modality as the search).
        let reader =
            open_reader(&store, disk_cache.as_ref(), storage.as_ref(), &entry, true).await?;
        attach_stable_ids(&reader, &entry, &mut hits[start..end], true).await?;
        if let Some(missing) = hits[start..end].iter().find(|h| h.stable_id.is_none()) {
            return Err(QueryError::Execute(format!(
                "hit {uri:?}/{} missing stable _id after search-wave stamping",
                missing.local_doc_id
            )));
        }
        start = end;
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
) -> Result<(), QueryError> {
    if entry.vector_layout != VectorLayout::MultiCellIvf {
        return apply_tombstone_filter(cache, entry, hits, now);
    }
    let Some(cache) = cache else {
        return Ok(());
    };
    let bitmap = cache
        .bitmap_for(entry.superfile_id, now)
        .map_err(|e| QueryError::Store(format!("tombstone cache: {e}")))?;
    if bitmap.is_empty() {
        return Ok(());
    }
    let locals: Vec<u32> = bitmap.iter().collect();
    let id_column = reader.id_column();
    let batch = if reader.parquet_bytes().is_some() {
        reader
            .take_by_local_doc_ids(&locals, &[id_column])
            .map_err(|e| QueryError::Execute(e.to_string()))?
    } else {
        let storage = storage.ok_or_else(|| {
            QueryError::Execute(
                "MultiCell tombstone resolve needs resident bytes or storage".into(),
            )
        })?;
        let (object_store, path) = storage
            .object_store_handle(&entry.uri.storage_path())
            .ok_or_else(|| QueryError::Execute("no object_store handle for superfile".into()))?;
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
        .map_err(|e| QueryError::Execute(e.to_string()))?
    };
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| QueryError::Execute("_id column missing".into()))?;
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
            .map_err(|e| QueryError::Execute(e.to_string()))?
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
        .map_err(|e| QueryError::Execute(e.to_string()))?;
    let array = batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| QueryError::Execute("_id column missing".into()))?;
    Ok(Some(array.values().to_vec()))
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
        true, // FTS/local-hit path — background fill allowed
        move |r, entry, tombstone_cache, now, params| {
            let kernel = kernel.clone();
            async move {
                let hits = kernel(r, params).await?;
                let mut tagged = tag_hits(&entry, hits);
                apply_tombstone_filter(tombstone_cache.as_ref(), &entry, &mut tagged, now)?;
                Ok::<Vec<SuperfileHit>, QueryError>(tagged)
            }
        },
    )
    .await
}

/// Lower-level fan-out primitive: the shared orchestration behind
/// [`fanout`] and the count path, generic over the per-superfile result
/// `R`.
///
/// It warms the tombstone sidecar cache for every distinct superfile in
/// one batch, `tokio::spawn`s one task per unit on the shared query
/// runtime (each opening its reader concurrently), then collects every
/// task with [`futures::future::try_join_all`] — so the **first**
/// per-superfile error (in time, not spawn order) short-circuits the
/// whole fan-out and returns early.
///
/// `body` runs inside each task with the opened reader, the superfile
/// entry, the (warmed) tombstone cache + the batch `now` instant, and
/// the unit's params. Resolving the per-superfile tombstone bitmap and
/// applying it is the body's job, since callers differ: [`fanout`]
/// tags + retains hits, while the count path either takes the O(1)
/// `term_df` fast path (no tombstones) or counts the matching ids minus
/// tombstones.
pub(crate) async fn fanout_with<P, R, B, Fut>(
    reader: &SupertableReader,
    units: Vec<(Arc<SuperfileEntry>, P)>,
    prefetch_tombstones: bool,
    allow_background_fill: bool,
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
    if units.is_empty() {
        return Ok(Vec::new());
    }
    trace!(units = units.len(), "fanning query out across superfiles");
    let manifest = reader.manifest();
    let store = Arc::clone(&manifest.options.store);
    let disk_cache = manifest.options.disk_cache.as_ref().map(Arc::clone);
    let storage = manifest.options.storage.as_ref().map(Arc::clone);
    let vector_columns = Arc::new(manifest.options.vector_columns.clone());
    let tombstone_cache = reader.tombstone_cache.clone();
    let now = Instant::now();

    // Warm the tombstone sidecars for every distinct superfile in one
    // concurrent batch before the per-superfile fan-out. Skipped by callers
    // whose tombstones are resolved elsewhere (the hidden path filters via
    // the resident deleted-set, so its per-cell sidecars are always empty
    // and prefetching them is a wasted wave of GETs on the cold critical path).
    if prefetch_tombstones && let Some(cache) = tombstone_cache.as_ref() {
        let mut ids: Vec<Uuid> = units.iter().map(|(e, _)| e.superfile_id).collect();
        ids.sort_unstable();
        ids.dedup();
        cache.prefetch(&ids, now).await;
    }

    // Single unit (the common case for a compacted, single-superfile
    // table): run the body inline on the current task. `tokio::spawn`
    // here would only add a thread handoff and a join with nothing to
    // overlap against — the spawn path's win is concurrency across units,
    // which doesn't exist at one unit. Semantically identical to the
    // fan-out below with a one-element result.
    if units.len() == 1 {
        let (entry, params) = units.into_iter().next().expect("len == 1");
        let r = open_reader(
            &store,
            disk_cache.as_ref(),
            storage.as_ref(),
            &entry,
            allow_background_fill,
        )
        .await?;
        verify_superfile_vector_codecs(&r, &vector_columns)?;
        let out = body(r, entry, tombstone_cache, now, params).await?;
        return Ok(vec![out]);
    }

    let handles = units.into_iter().map(|(entry, params)| {
        let store = Arc::clone(&store);
        let disk_cache = disk_cache.clone();
        let storage = storage.clone();
        let tombstone_cache = tombstone_cache.clone();
        let body = body.clone();
        let vector_columns = Arc::clone(&vector_columns);
        let handle = tokio::spawn(async move {
            let r = open_reader(
                &store,
                disk_cache.as_ref(),
                storage.as_ref(),
                &entry,
                allow_background_fill,
            )
            .await?;
            verify_superfile_vector_codecs(&r, &vector_columns)?;
            body(r, entry, tombstone_cache, now, params).await
        });
        // Flatten the join error into a QueryError so `try_join_all`
        // short-circuits on the first failing superfile.
        async move {
            handle
                .await
                .map_err(|e| QueryError::Store(format!("fan-out task join: {e}")))?
        }
    });
    try_join_all(handles).await
}
