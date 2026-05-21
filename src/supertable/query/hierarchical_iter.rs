//! Hierarchical lazy-load helper.
//!
//! The bridge between the list-level prune helpers
//! (`list_prune::prune_parts_for_*`) and the per-part
//! segment iteration the query paths need:
//!
//!   1. Caller computes a `kept_part_ids: Vec<PartId>` via
//!      the appropriate `prune_parts_for_*` for its query
//!      shape.
//!   2. [`load_kept_parts`] lazy-loads each kept part via
//!      `Manifest::part(id).await`, in parallel. Already-
//!      loaded parts (eager mode, or warm OnceCells) cost
//!      nothing.
//!   3. [`flatten_segments`] concatenates the loaded parts'
//!      superfiles into a single `Vec<Arc<SuperfileEntry>>`
//!      that the existing segment-level skip + fan-out
//!      code consumes.
//!
//! Sync bridge via the same ambient-runtime detection the
//! writer's `persist_commit` uses (`Handle::try_current` →
//! `block_in_place + handle.block_on` when in an async
//! context; `sql_runtime.block_on` otherwise). The query
//! paths stay sync end-to-end; callers don't acquire any
//! runtime knowledge.

use std::sync::Arc;

use crate::supertable::error::QueryError;
use crate::supertable::manifest::part::{ManifestPart, PartId};
use crate::supertable::manifest::{Manifest, SuperfileEntry};

/// Lazy-load each part in `kept_part_ids` via
/// `Manifest::part(id).await`, in parallel.
///
/// Cheap when parts are already loaded (eager mode, or a
/// prior query warmed them) — each `Manifest::part` call
/// hits the part's `OnceCell` and returns an `Arc::clone`
/// without I/O. Lazy mode triggers one storage GET per
/// not-yet-loaded part; the `join_all` issues them in
/// parallel so wall-clock is `max(per-part GET latency)`
/// not the serial sum.
///
/// Sync→async bridge via the standard pattern:
/// `Handle::try_current` → `block_in_place + handle.block_on`
/// inside an ambient runtime; build a `new_current_thread`
/// runtime ad-hoc outside one. Mirrors `superfile_reader`'s
/// disk-cache bridge.
pub fn load_kept_parts(
    manifest: &Manifest,
    kept_part_ids: &[PartId],
) -> Result<Vec<Arc<ManifestPart>>, QueryError> {
    if kept_part_ids.is_empty() {
        return Ok(Vec::new());
    }
    let load_futs: Vec<_> = kept_part_ids
        .iter()
        .map(|id| {
            let pid = *id;
            async move { manifest.part(pid).await }
        })
        .collect();

    let drive = async move {
        let loaded = futures::future::join_all(load_futs).await;
        let mut out = Vec::with_capacity(loaded.len());
        for r in loaded {
            out.push(r.map_err(|e| QueryError::Store(format!("part load: {e}")))?);
        }
        Ok::<Vec<Arc<ManifestPart>>, QueryError>(out)
    };

    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(drive)),
        Err(_) => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| {
                    QueryError::Store(format!("tokio runtime build for hierarchical_iter: {e}"))
                })?;
            rt.block_on(drive)
        }
    }
}

/// Concatenate the loaded parts' superfiles into a flat
/// `Vec<Arc<SuperfileEntry>>` for downstream segment-level
/// skip + fan-out. Cheap: every entry is `Arc::clone` of an
/// already-allocated `SuperfileEntry`.
pub fn flatten_segments(parts: &[Arc<ManifestPart>]) -> Vec<Arc<SuperfileEntry>> {
    let total: usize = parts.iter().map(|p| p.superfiles.len()).sum();
    let mut out = Vec::with_capacity(total);
    for p in parts {
        out.extend(p.superfiles.iter().cloned());
    }
    out
}

/// Combined helper: lazy-load + flatten in one call. The
/// common shape across query paths.
pub fn load_and_flatten(
    manifest: &Manifest,
    kept_part_ids: &[PartId],
) -> Result<Vec<Arc<SuperfileEntry>>, QueryError> {
    let parts = load_kept_parts(manifest, kept_part_ids)?;
    Ok(flatten_segments(&parts))
}

/// **Fallback shape** for query callers operating on
/// in-process manifests with no `list` (in-memory-only
/// supertables, or supertables that haven't persisted yet):
/// just return the flat `manifest.superfiles`. The eager-mode
/// + lazy-mode hierarchical path through `load_and_flatten`
/// requires a `ManifestList`; this branch covers the no-list
/// case so the query paths remain uniformly callable.
pub fn fallback_to_flat_segments(manifest: &Manifest) -> Vec<Arc<SuperfileEntry>> {
    manifest.superfiles.iter().cloned().collect()
}
