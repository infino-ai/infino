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

use futures::future::try_join_all;
use roaring::RoaringBitmap;
use tracing::trace;
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
#[cfg_attr(
    feature = "detailed-tracing",
    tracing::instrument(skip_all, fields(uri = ?entry.uri))
)]
pub(crate) async fn open_reader(
    store: &Arc<dyn SuperfileReaderCache>,
    disk_cache: Option<&Arc<DiskCacheStore>>,
    storage: Option<&Arc<dyn StorageProvider>>,
    entry: &SuperfileEntry,
) -> Result<Arc<SuperfileReader>, QueryError> {
    superfile_reader(
        store,
        disk_cache,
        storage,
        &entry.uri,
        entry.subsection_offsets.as_ref(),
    )
    .await
    .map_err(|e| QueryError::Store(e.to_string()))
}

/// Tag a kernel's `(local_doc_id, score)` results with their source
/// superfile URI.
pub(crate) fn tag_hits(entry: &SuperfileEntry, hits: Vec<(u32, f32)>) -> Vec<SuperfileHit> {
    hits.into_iter()
        .map(|(local_doc_id, score)| SuperfileHit {
            superfile: entry.uri,
            local_doc_id,
            score,
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
    let tombstone_cache = reader.tombstone_cache.clone();
    let now = Instant::now();

    // Warm the tombstone sidecars for every distinct superfile in one
    // concurrent batch before the per-superfile fan-out.
    if let Some(cache) = tombstone_cache.as_ref() {
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
        let r = open_reader(&store, disk_cache.as_ref(), storage.as_ref(), &entry).await?;
        let out = body(r, entry, tombstone_cache, now, params).await?;
        return Ok(vec![out]);
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
