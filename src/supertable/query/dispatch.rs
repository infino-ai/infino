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

use std::{future::Future, sync::Arc, time::Instant};

use arrow_array::Decimal128Array;
use futures::future::try_join_all;
use uuid::Uuid;

use super::SuperfileHit;
use crate::{
    storage::StorageProvider,
    superfile::SuperfileReader,
    supertable::{
        error::QueryError,
        handle::SupertableReader,
        manifest::SuperfileEntry,
        query::superfile_reader::superfile_reader,
        reader_cache::{DiskCacheStore, SuperfileReaderCache},
        tombstones::SidecarCache,
    },
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
) -> Result<Arc<SuperfileReader>, QueryError> {
    let entry_storage = storage_for_entry(entry, storage);
    superfile_reader(
        store,
        disk_cache,
        entry_storage.as_ref(),
        &entry.uri,
        entry.subsection_offsets.as_ref(),
    )
    .await
    .map_err(|e| QueryError::Store(e.to_string()))
}

/// The storage to fetch `entry`'s bytes through. A hidden vector-index INCOMING
/// entry is a pointer to a user-table superfile whose bytes live one prefix
/// level up — the hidden index's storage is a prefixed view of the user storage
/// (see [`PrefixedStorageProvider`](crate::storage::PrefixedStorageProvider)),
/// so an INCOMING pointer must be read through that inner (user) storage;
/// fetching it through the hidden prefix resolves to a path that does not exist.
/// Every other entry — real hidden cells, ordinary user superfiles — uses
/// `storage` as given.
pub(crate) fn storage_for_entry(
    entry: &SuperfileEntry,
    storage: Option<&Arc<dyn StorageProvider>>,
) -> Option<Arc<dyn StorageProvider>> {
    if entry.is_incoming_pointer() {
        if let Some(inner) = storage.and_then(|s| s.prefix_inner()) {
            return Some(inner);
        }
    }
    storage.map(Arc::clone)
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
    if let (Some(cache), Some(storage)) = (disk_cache, storage) {
        return cache
            .reader_synchronous_with_storage(&entry.uri, Arc::clone(storage))
            .await
            .map_err(|e| QueryError::Store(e.to_string()));
    }
    open_reader(store, disk_cache, storage, entry).await
}

/// Tag a kernel's `(local_doc_id, score)` results with their source
/// superfile URI.
pub(crate) fn tag_hits(entry: &SuperfileEntry, hits: Vec<(u32, f32)>) -> Vec<SuperfileHit> {
    hits.into_iter()
        .map(|(local_doc_id, score)| SuperfileHit {
            superfile: entry.uri,
            local_doc_id,
            score,
            stable_id: None,
        })
        .collect()
}

/// Drop tombstoned `local_doc_id`s from one superfile's hits. After the
/// orchestrator's batched [`SidecarCache::prefetch`] every lookup here
/// is an in-memory cache hit, so this is a cheap retain pass.
pub(crate) fn apply_tombstone_filter(
    cache: Option<&Arc<SidecarCache>>,
    entry: &SuperfileEntry,
    hits: &mut Vec<SuperfileHit>,
    now: Instant,
) -> Result<(), QueryError> {
    let Some(cache) = cache else {
        return Ok(());
    };
    let bitmap = cache
        .bitmap_for(entry.superfile_id, now)
        .map_err(|e| QueryError::Store(format!("tombstone cache: {e}")))?;
    if bitmap.is_empty() {
        return Ok(());
    }
    hits.retain(|h| !bitmap.contains(h.local_doc_id));
    Ok(())
}

/// Resolve stable user `_id`s for tagged hits from bytes already resident
/// on this superfile reader — inline IVF region first (materialized hidden
/// cells), then the scalar `_id` column (INCOMING staging superfiles).
/// `None` when the bytes are not yet mmap'd (cold lazy); the remap step
/// falls back to a manifest-backed read.
fn stable_ids_for_tagged_hits(reader: &SuperfileReader, locals: &[u32]) -> Option<Vec<i128>> {
    if locals.is_empty() {
        return Some(Vec::new());
    }
    if let Some(v) = reader.vec() {
        if let Some(ids) = v.inline_stable_ids_for_locals(locals) {
            return Some(ids);
        }
    }
    if reader.parquet_bytes().is_none() {
        return None;
    }
    let id_column = reader.id_column();
    let batch = reader.take_by_local_doc_ids(locals, &[id_column]).ok()?;
    let array = batch.column(0).as_any().downcast_ref::<Decimal128Array>()?;
    Some(array.values().to_vec())
}

/// Fan a per-superfile async kernel out across `units`, returning each
/// unit's tagged + tombstone-filtered hits in input order.
///
/// Each unit is `(superfile_entry, params)`; `params` carries any
/// per-unit kernel input (e.g. an FTS doc-id sub-range — `()` for
/// vector). The orchestrator:
///
///   1. Warms the tombstone sidecar cache for every distinct superfile
///      in one concurrent batch (so the post-search filter is all
///      cache hits).
///   2. `tokio::spawn`s one task per unit on the shared query runtime;
///      each opens its reader (`await`) and runs `kernel` (`await`) —
///      so opens and the kernel's cold GETs are concurrent across the
///      whole fan-out.
///   3. Tags + tombstone-filters each unit's hits.
///
/// The kernel returns `(local_doc_id, score)` pairs. CPU policy is the
/// kernel's own: the vector kernel parallelizes with `par_iter`, while
/// the FTS kernel scores serially within this task (FTS parallelism is
/// expressed as extra work units, not rayon).
pub(crate) async fn fanout<P, K, Fut>(
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
        move |r, entry, tombstone_cache, now, params| {
            let kernel = kernel.clone();
            async move {
                let reader_for_ids = Arc::clone(&r);
                let hits = kernel(r, params).await?;
                let mut tagged = tag_hits(&entry, hits);
                apply_tombstone_filter(tombstone_cache.as_ref(), &entry, &mut tagged, now)?;
                // Piggyback the hidden→user `_id` resolve onto the search.
                // Materialized hidden cells: inline `_id` region (prefetched
                // on cold, resident on warm). INCOMING staging superfiles:
                // scalar `_id` column via sync `take_by_local_doc_ids` on
                // resident bytes — both skip the trailing remap GET.
                if !tagged.is_empty() {
                    let locals: Vec<u32> = tagged.iter().map(|h| h.local_doc_id).collect();
                    if let Some(ids) = stable_ids_for_tagged_hits(&reader_for_ids, &locals) {
                        for (h, id) in tagged.iter_mut().zip(ids) {
                            h.stable_id = Some(id);
                        }
                    }
                }
                Ok::<Vec<SuperfileHit>, QueryError>(tagged)
            }
        },
    )
    .await
}

/// Fan-out for the OPANN hidden vector-index path. Identical to [`fanout`] —
/// runs the kernel, tags hits, and piggybacks the hidden→user `_id` resolve —
/// but does NOT prefetch or apply per-cell tombstone sidecars. The hidden
/// cells' sidecars are never populated (user deletes are recorded in the
/// resident deleted-set instead), so the prefetch wave fetched only empty
/// objects on the cold critical path and the post-score filter was a no-op.
/// Deletes are dropped by the caller against the resident deleted-set, keyed by
/// the `stable_id` resolved here.
pub(crate) async fn fanout_untombstoned<P, K, Fut>(
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
        false,
        move |r, entry, _tombstone_cache, _now, params| {
            let kernel = kernel.clone();
            async move {
                let reader_for_ids = Arc::clone(&r);
                let hits = kernel(r, params).await?;
                let mut tagged = tag_hits(&entry, hits);
                if !tagged.is_empty() {
                    let locals: Vec<u32> = tagged.iter().map(|h| h.local_doc_id).collect();
                    if let Some(ids) = stable_ids_for_tagged_hits(&reader_for_ids, &locals) {
                        for (h, id) in tagged.iter_mut().zip(ids) {
                            h.stable_id = Some(id);
                        }
                    }
                }
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
    let manifest = reader.manifest();
    let store = Arc::clone(&manifest.options.store);
    let disk_cache = manifest.options.disk_cache.as_ref().map(Arc::clone);
    let storage = manifest.options.storage.as_ref().map(Arc::clone);
    let tombstone_cache = reader.tombstone_cache.clone();
    let now = Instant::now();

    // Warm the tombstone sidecars for every distinct superfile in one
    // concurrent batch before the per-superfile fan-out. Skipped by callers
    // whose tombstones are resolved elsewhere (the OPANN hidden path filters
    // via the resident deleted-set, so its per-cell sidecars are always empty
    // and prefetching them is a wasted wave of GETs on the cold critical path).
    if prefetch_tombstones {
        if let Some(cache) = tombstone_cache.as_ref() {
            let mut ids: Vec<Uuid> = units.iter().map(|(e, _)| e.superfile_id).collect();
            ids.sort_unstable();
            ids.dedup();
            cache.prefetch(&ids, now).await;
        }
    }

    let handles = units.into_iter().map(|(entry, params)| {
        let store = Arc::clone(&store);
        let disk_cache = disk_cache.clone();
        let storage = storage.clone();
        let tombstone_cache = tombstone_cache.clone();
        let body = body.clone();
        let handle = tokio::spawn(async move {
            let r = open_reader(&store, disk_cache.as_ref(), storage.as_ref(), &entry).await?;
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
