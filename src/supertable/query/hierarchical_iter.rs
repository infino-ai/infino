// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Hierarchical lazy-load helper.
//!
//! The bridge between the list-level prune helpers
//! (`list_prune::prune_parts_for_*`) and the per-part
//! superfile iteration the query paths need:
//!
//!   1. Caller computes a `kept_part_ids: Vec<PartId>` via
//!      the appropriate `prune_parts_for_*` for its query
//!      shape.
//!   2. [`load_kept_parts`] lazy-loads each kept part via
//!      `ManifestSnapshot::part(id).await`, in parallel. Already-
//!      loaded parts (eager mode, or warm OnceCells) cost
//!      nothing.
//!   3. [`flatten_superfiles`] concatenates the loaded parts'
//!      superfiles into a single `Vec<Arc<SuperfileEntry>>`
//!      that the existing superfile-level skip + fan-out
//!      code consumes.
//!
//! `async` end-to-end: the query paths that call these helpers
//! are themselves async and run on the owning tokio runtime, so
//! part-load GETs are driven by that runtime's reactor (no sync
//! bridge, no throwaway runtime).

use std::sync::Arc;

use crate::supertable::{
    ManifestLoadError,
    manifest::{
        ManifestSnapshot, SuperfileEntry,
        part::{ManifestPart, PartId},
    },
};

/// Lazy-load each part in `kept_part_ids` via
/// `ManifestSnapshot::part(id).await`, in parallel.
///
/// Cheap when parts are already loaded (eager mode, or a
/// prior query warmed them) — each `ManifestSnapshot::part` call
/// hits the part's `OnceCell` and returns an `Arc::clone`
/// without I/O. Lazy mode triggers one storage GET per
/// not-yet-loaded part; the `join_all` issues them in
/// parallel so wall-clock is `max(per-part GET latency)`
/// not the serial sum.
///
/// `await`s each part load on the caller's runtime; cold lazy
/// parts' GETs run on that runtime's reactor.
pub async fn load_kept_parts(
    manifest: &ManifestSnapshot,
    kept_part_ids: &[PartId],
) -> Result<Vec<Arc<ManifestPart>>, ManifestLoadError> {
    if kept_part_ids.is_empty() {
        return Ok(Vec::new());
    }
    let load_futs: Vec<_> = kept_part_ids
        .iter()
        .map(|id| {
            let pid = *id;
            async move { manifest.get_part_by_id(pid).await }
        })
        .collect();

    let loaded = futures::future::join_all(load_futs).await;
    let mut out = Vec::with_capacity(loaded.len());
    for r in loaded {
        out.push(r?);
    }
    Ok(out)
}

/// Concatenate the loaded parts' superfiles into a flat
/// `Vec<Arc<SuperfileEntry>>` for downstream superfile-level
/// skip + fan-out. Cheap: every entry is `Arc::clone` of an
/// already-allocated `SuperfileEntry`.
pub fn flatten_superfiles(parts: &[Arc<ManifestPart>]) -> Vec<Arc<SuperfileEntry>> {
    let total: usize = parts.iter().map(|p| p.superfiles.len()).sum();
    let mut out = Vec::with_capacity(total);
    for p in parts {
        out.extend(p.superfiles.iter().cloned());
    }
    out
}

/// Combined helper: lazy-load + flatten in one call. The
/// common shape across query paths.
pub async fn load_and_flatten(
    manifest: &ManifestSnapshot,
    kept_part_ids: &[PartId],
) -> Result<Vec<Arc<SuperfileEntry>>, ManifestLoadError> {
    let parts = load_kept_parts(manifest, kept_part_ids).await?;
    Ok(flatten_superfiles(&parts))
}

/// **Fallback shape** for query callers operating on
/// in-process manifests with no `list` (in-memory-only
/// supertables, or supertables that haven't persisted yet): just return
/// the flat `manifest.superfiles`. The eager-mode + lazy-mode hierarchical
/// path through `load_and_flatten` requires a persisted Manifest; this branch
/// covers the no-list case so the query paths remain uniformly callable.
pub fn fallback_to_flat_superfiles(manifest: &ManifestSnapshot) -> Vec<Arc<SuperfileEntry>> {
    manifest.superfiles.to_vec()
}
