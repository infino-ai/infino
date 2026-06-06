//! Shared fan-out/dispatch for the segment-parallel query paths.
//!
//! Vector kNN and BM25/prefix FTS both face the identical shape: a
//! pinned manifest snapshot, a kept set of segments (after manifest
//! pruning), and a per-segment search kernel whose result is a list of
//! `(local_doc_id, score)` pairs. The plumbing around that kernel —
//! open every segment reader concurrently, warm the tombstone sidecar
//! cache in one batch, run each segment's kernel, tag the hits with
//! their segment URI, and drop tombstoned rows — is the same for both.
//!
//! This module owns that plumbing so the two query paths share one
//! orchestrator instead of each re-implementing the fan-out. The
//! division of labor is the project-wide model:
//!
//!   * **tokio owns the I/O waves.** Segment opens and the kernel's
//!     cold object-store range GETs are `await`ed on the shared
//!     multi-thread query runtime, so reqwest connections pool and
//!     hundreds of segments' fetches are in flight at once.
//!   * **rayon owns the CPU waves.** The per-segment compute (centroid
//!     / 1-bit-code scoring + rerank for vector; BMW/MaxScore scoring
//!     for FTS) runs on the global rayon pool via the kernel's internal
//!     `par_iter` — never on the tokio workers that must stay free to
//!     drive the I/O.
//!
//! The per-segment merge (top-k ascending for vector distance,
//! descending for BM25 relevance) stays with each caller; this layer
//! returns the per-unit tagged+filtered hit lists.

use std::future::Future;
use std::sync::Arc;

use crate::storage::StorageProvider;
use crate::superfile::SuperfileReader;
use crate::supertable::error::QueryError;
use crate::supertable::handle::SupertableReader;
use crate::supertable::manifest::SuperfileEntry;
use crate::supertable::reader_cache::{DiskCacheStore, SuperfileReaderCache};
use crate::supertable::tombstones::SidecarCache;

use super::SuperfileHit;

/// Open one segment's `SuperfileReader` through the reader cache.
/// Warm opens are in-memory cache hits (microseconds); cold opens
/// fetch the segment header/footer from object storage. Always
/// `await`ed so the open I/O overlaps across the fan-out.
pub(crate) async fn open_reader(
    store: &Arc<dyn SuperfileReaderCache>,
    disk_cache: Option<&Arc<DiskCacheStore>>,
    storage: Option<&Arc<dyn StorageProvider>>,
    entry: &SuperfileEntry,
) -> Result<Arc<SuperfileReader>, QueryError> {
    crate::supertable::query::superfile_reader::superfile_reader(
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
/// segment URI.
pub(crate) fn tag_hits(entry: &SuperfileEntry, hits: Vec<(u32, f32)>) -> Vec<SuperfileHit> {
    hits.into_iter()
        .map(|(local_doc_id, score)| SuperfileHit {
            segment: entry.uri,
            local_doc_id,
            score,
        })
        .collect()
}

/// Drop tombstoned `local_doc_id`s from one segment's hits. After the
/// orchestrator's batched [`SidecarCache::prefetch`] every lookup here
/// is an in-memory cache hit, so this is a cheap retain pass.
pub(crate) fn apply_tombstone_filter(
    cache: Option<&Arc<SidecarCache>>,
    entry: &SuperfileEntry,
    hits: &mut Vec<SuperfileHit>,
    now: std::time::Instant,
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

/// Fan a per-segment async kernel out across `units`, returning each
/// unit's tagged + tombstone-filtered hits in input order.
///
/// Each unit is `(segment_entry, params)`; `params` carries any
/// per-unit kernel input (e.g. an FTS doc-id sub-range — `()` for
/// vector). The orchestrator:
///
///   1. Warms the tombstone sidecar cache for every distinct segment
///      in one concurrent batch (so the post-search filter is all
///      cache hits).
///   2. `tokio::spawn`s one task per unit on the shared query runtime;
///      each opens its reader (`await`) and runs `kernel` (`await`) —
///      so opens and the kernel's cold GETs are concurrent across the
///      whole fan-out.
///   3. Tags + tombstone-filters each unit's hits.
///
/// The kernel returns `(local_doc_id, score)` pairs; it is responsible
/// for keeping its own CPU on the global rayon pool (internal
/// `par_iter`).
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
    if units.is_empty() {
        return Ok(Vec::new());
    }
    let manifest = reader.manifest();
    let store = Arc::clone(&manifest.options.store);
    let disk_cache = manifest.options.disk_cache.as_ref().map(Arc::clone);
    let storage = manifest.options.storage.as_ref().map(Arc::clone);
    let tombstone_cache = reader.tombstone_cache.clone();
    let now = std::time::Instant::now();

    // Warm the tombstone sidecars for every distinct segment in one
    // concurrent batch before the per-segment fan-out.
    if let Some(cache) = tombstone_cache.as_ref() {
        let mut ids: Vec<uuid::Uuid> = units.iter().map(|(e, _)| e.superfile_id).collect();
        ids.sort_unstable();
        ids.dedup();
        cache.prefetch(&ids, now).await;
    }

    let handles: Vec<_> = units
        .into_iter()
        .map(|(entry, params)| {
            let store = Arc::clone(&store);
            let disk_cache = disk_cache.clone();
            let storage = storage.clone();
            let tombstone_cache = tombstone_cache.clone();
            let kernel = kernel.clone();
            tokio::spawn(async move {
                let r = open_reader(&store, disk_cache.as_ref(), storage.as_ref(), &entry).await?;
                let hits = kernel(r, params).await?;
                let mut tagged = tag_hits(&entry, hits);
                apply_tombstone_filter(tombstone_cache.as_ref(), &entry, &mut tagged, now)?;
                Ok::<Vec<SuperfileHit>, QueryError>(tagged)
            })
        })
        .collect();

    let mut out: Vec<Vec<SuperfileHit>> = Vec::with_capacity(handles.len());
    for h in handles {
        let tagged = h
            .await
            .map_err(|e| QueryError::Store(format!("fan-out task join: {e}")))??;
        out.push(tagged);
    }
    Ok(out)
}
