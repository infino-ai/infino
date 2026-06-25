// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! OPANN path-B vector probe: tree descent selects leaves; each admitted leaf
//! is fetched with a direct range GET on the superfile object (no Parquet
//! footer, no internal IVF centroid scoring).

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use roaring::RoaringBitmap;

use super::{SuperfileHit, dispatch};
use crate::{
    superfile::{SuperfileReader, VectorError},
    supertable::{
        error::QueryError,
        handle::SupertableReader,
        manifest::{Manifest, SuperfileEntry, SuperfileUri},
        opann::{page::LeafRef, paged::PagedTree},
    },
};

use super::vector::VectorSearchOptions;

/// Radius-aware adaptive leaf admission (§7.3): always probe the
/// `nprobe_min` nearest OPANN leaves, then admit farther leaves whose
/// radius-aware lower bound clears τ up to `nprobe_max`.
pub(super) fn adaptive_probe_leaves(
    candidates: Vec<(LeafRef, f32)>,
    radius_of: &HashMap<u128, f32>,
    nprobe_min: usize,
    nprobe_max: usize,
    slack: f32,
) -> Vec<(LeafRef, f32)> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let mut scored = candidates;
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let radius = |leaf: LeafRef| radius_of.get(&leaf.superfile_id).copied().unwrap_or(0.0);
    let lb = |i: usize| {
        let (leaf, d) = scored[i];
        (d - radius(leaf)).max(0.0)
    };
    let (c0, d_star) = scored[0];
    let r_star = radius(c0);
    let tau = if r_star > 0.0 {
        d_star + slack * r_star
    } else {
        d_star * (1.0 + slack)
    };
    let nprobe_min = nprobe_min.max(1);
    let nprobe_max = nprobe_max.max(nprobe_min);
    let n = scored.len();
    let floor = nprobe_min.min(n);
    let mut chosen: HashSet<(u128, u32)> = scored[..floor]
        .iter()
        .map(|(leaf, _)| (leaf.superfile_id, leaf.doc_off))
        .collect();
    let mut out: Vec<(LeafRef, f32)> = scored[..floor].to_vec();
    let mut rest: Vec<usize> = (floor..n).collect();
    rest.sort_by(|&a, &b| lb(a).partial_cmp(&lb(b)).unwrap_or(Ordering::Equal));
    for i in rest {
        if out.len() >= nprobe_max {
            break;
        }
        if lb(i) <= tau {
            let key = (scored[i].0.superfile_id, scored[i].0.doc_off);
            if chosen.insert(key) {
                out.push(scored[i]);
            }
        }
    }
    out
}

/// Descend the resident OPANN tree and return admitted probe leaves.
pub(super) async fn select_opann_probe_leaves(
    _reader: &SupertableReader,
    manifest: &Manifest,
    column: &str,
    query: &[f32],
    options: &VectorSearchOptions,
    survives: impl Fn(u128) -> bool,
) -> Result<Option<Vec<(LeafRef, f32, Arc<SuperfileEntry>)>>, QueryError> {
    let tree = manifest
        .opann_resident_tree()
        .await
        .map_err(|e| QueryError::Store(format!("opann tree load: {e}")))?;
    let Some(source) = tree else {
        return Ok(None);
    };
    let Some((root, routing)) = manifest.opann_routing().map(|r| (r.root_page, r.routing)) else {
        return Ok(None);
    };

    // Collect every surviving leaf (same as [`super::vector::SupertableReader::candidate_superfiles_for_vector`]):
    // radius-aware τ admission runs over the full candidate pool and trims to
    // `[floor, nprobe_max]`. Capping descent at `nprobe_max` first drops cells
    // that τ would have admitted — spread-out queries need those far-but-large-
    // radius partitions, not a smaller fixed centroid-depth budget.
    let candidates = PagedTree::new(source, root)
        .select_probes_where(query, usize::MAX, &survives)
        .map_err(|e| QueryError::Store(format!("opann descent: {e}")))?;

    let entries = super::vector::ordered_manifest_superfiles(manifest).await?;
    let entry_by_id: HashMap<u128, Arc<SuperfileEntry>> = entries
        .iter()
        .map(|e| (e.superfile_id.as_u128(), Arc::clone(e)))
        .collect();

    let radius_of: HashMap<u128, f32> = entries
        .iter()
        .filter_map(|sf| {
            sf.vector_summary
                .get(column)
                .map(|vs| (sf.superfile_id.as_u128(), vs.radius))
        })
        .collect();

    let floor = options.nprobe.unwrap_or(routing.nprobe_min);
    let admitted = adaptive_probe_leaves(
        candidates,
        &radius_of,
        floor,
        routing.nprobe_max,
        routing.slack,
    );

    let mut out = Vec::with_capacity(admitted.len());
    for (leaf, dist) in admitted {
        let Some(entry) = entry_by_id.get(&leaf.superfile_id) else {
            continue;
        };
        if !entry_has_vector_probe_layout(entry) {
            continue;
        }
        out.push((leaf, dist, Arc::clone(entry)));
    }
    if out.is_empty() {
        return Ok(None);
    }
    Ok(Some(out))
}

fn entry_has_vector_probe_layout(entry: &SuperfileEntry) -> bool {
    entry
        .subsection_offsets
        .as_ref()
        .and_then(|o| o.vec)
        .is_some_and(|(_, len)| len > 0)
}

/// Fan whole-cell probes across the OPANN-selected hidden cells, reusing the
/// **normal** superfile read path. [`dispatch::fanout`] opens each cell through
/// the same tiered opener the user-table vector/FTS/SQL fan-outs use (in-memory
/// reader cache → disk-cache mmap → storage fallback), warms and applies the
/// tombstone sidecar, and tags hits with their superfile — so the hidden cell
/// bytes are mmap-backed and shared, never re-fetched into the heap per query.
/// The tree already routed to the cell, so each cell is scanned whole via
/// [`probe_leaf_async`] with `(doc_off, count) = (0, 0)`.
///
/// [`probe_leaf_async`]: crate::superfile::vector::reader::VectorReader::probe_leaf_async
pub(super) async fn fanout_opann_leaf_probes(
    reader: &SupertableReader,
    leaves: Vec<(LeafRef, f32, Arc<SuperfileEntry>)>,
    column: &str,
    query: &[f32],
    k: usize,
    options: VectorSearchOptions,
    allow: Option<HashMap<SuperfileUri, Arc<RoaringBitmap>>>,
) -> Result<Vec<SuperfileHit>, QueryError> {
    let filtered = allow.is_some();
    let rerank_mult = options.resolve(filtered).1;

    // One whole-cell unit per hidden superfile (dedupe legacy per-cluster
    // leaves). For filtered search a cell whose predicate matched no row is
    // absent from `allow` and dropped — it never opens or fetches.
    let mut units: Vec<(Arc<SuperfileEntry>, Option<Arc<RoaringBitmap>>)> = Vec::new();
    let mut seen: HashSet<u128> = HashSet::new();
    for (leaf, _dist, entry) in leaves {
        if !seen.insert(leaf.superfile_id) {
            continue;
        }
        let bitmap = match allow.as_ref() {
            Some(m) => match m.get(&entry.uri) {
                Some(bm) => Some(Arc::clone(bm)),
                None => continue,
            },
            None => None,
        };
        units.push((entry, bitmap));
    }
    if units.is_empty() {
        return Ok(Vec::new());
    }

    let column = Arc::new(column.to_owned());
    let query = Arc::new(query.to_vec());
    let kernel = move |r: Arc<SuperfileReader>, bitmap: Option<Arc<RoaringBitmap>>| {
        let column = Arc::clone(&column);
        let query = Arc::clone(&query);
        async move {
            let v = r.vec().ok_or_else(|| {
                QueryError::Store("hidden cell superfile missing vector subsection".into())
            })?;
            v.probe_leaf_async(&column, &query, k, 0, 0, rerank_mult, bitmap)
                .await
                .map_err(map_vector_err)
        }
    };
    let per_superfile = dispatch::fanout(reader, units, kernel).await?;
    // Our `top_k_ascending` takes the per-superfile nested vec and flattens
    // internally (opann's takes a pre-flattened vec); pass it directly.
    Ok(super::vector::top_k_ascending(per_superfile, k))
}

fn map_vector_err(e: VectorError) -> QueryError {
    QueryError::Store(format!("vector leaf probe: {e}"))
}
