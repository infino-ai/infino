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
        opann::{page::LeafRef, paged::PagedTree, store},
    },
};

use super::vector::{VectorSearchOptions, row_id_from_manifest_entry};

/// Radius-aware adaptive leaf admission (§7.3): always probe the
/// `nprobe_min` nearest OPANN leaves, then admit farther leaves whose
/// radius-aware lower bound clears τ up to `nprobe_max`.
pub(super) fn adaptive_probe_leaves(
    candidates: Vec<(LeafRef, f32)>,
    radius_of: impl Fn(LeafRef) -> f32,
    nprobe_min: usize,
    nprobe_max: usize,
    slack: f32,
) -> Vec<(LeafRef, f32)> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let mut scored = candidates;
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let radius = |leaf: LeafRef| radius_of(leaf);
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
    let entries = super::vector::ordered_manifest_superfiles(manifest).await?;

    let mut out: Vec<(LeafRef, f32, Arc<SuperfileEntry>)> = Vec::new();

    // Descend the routing tree to cell leaves — ONLY when a tree is published.
    // Before the first drain there is no tree yet; the INCOMING-staging probes
    // below still run, so a pre-drain query is the SAME OPANN fan-out with zero
    // cell probes — not a separate scan path. (When there is neither a tree nor
    // any incoming cells — e.g. a user table that isn't OPANN-routed — `out`
    // stays empty and the caller falls back to the per-superfile IVF scan.)
    if let (Some(source), Some(routing_info)) = (tree, manifest.opann_routing()) {
        let root = routing_info.root_page;
        let routing = routing_info.routing;

        // Radius-aware τ admission runs over the full candidate pool and trims
        // to `[floor, nprobe_max]`. Capping descent at `nprobe_max` first drops
        // cells that τ would have admitted — spread-out queries need those
        // far-but-large-radius partitions, not a fixed centroid-depth budget.
        let candidates = PagedTree::new(source, root)
            .select_probes_where(query, usize::MAX, &survives)
            .map_err(|e| QueryError::Store(format!("opann descent: {e}")))?;

        let entry_by_id: HashMap<u128, Arc<SuperfileEntry>> = entries
            .iter()
            .map(|e| (e.superfile_id.as_u128(), Arc::clone(e)))
            .collect();

        // Per-cluster covering radius for the τ admission, read from the
        // resident manifest summary. Every cluster of every superfile is its own
        // tree leaf, so the radius is per-CLUSTER (indexed by `leaf.cluster_id`),
        // not per-superfile. Falls back to 0 (collapse to centroid distance)
        // when a leaf's cluster has no recorded radius.
        let radius_of = |leaf: LeafRef| {
            entry_by_id
                .get(&leaf.superfile_id)
                .and_then(|e| e.vector_summary.get(column))
                .and_then(|vs| vs.clusters.radii.get(leaf.cluster_id as usize).copied())
                .unwrap_or(0.0)
        };

        let floor = options.nprobe.unwrap_or(routing.nprobe_min);
        let admitted = adaptive_probe_leaves(
            candidates,
            radius_of,
            floor,
            routing.nprobe_max,
            routing.slack,
        );

        // Every admitted leaf is already a single internal IVF cluster — the
        // tree routed straight to it (centroid → cluster byte range). Probe that
        // range directly: no within-superfile rescan, no whole-superfile leaf.
        out.reserve(admitted.len());
        for (leaf, dist) in admitted {
            let Some(entry) = entry_by_id.get(&leaf.superfile_id) else {
                continue;
            };
            if !entry_has_vector_probe_layout(entry) {
                continue;
            }
            out.push((leaf, dist, Arc::clone(entry)));
        }
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

/// Fan the OPANN-admitted probes across hidden cells, reusing the **normal**
/// superfile read path. [`dispatch::fanout`] opens each cell through the same
/// tiered opener the user-table fan-outs use (in-memory reader cache →
/// disk-cache mmap → storage fallback), warms + applies the tombstone sidecar,
/// and tags hits with their superfile — so the cell bytes are mmap-backed and
/// shared. Every admitted leaf is a single internal IVF cluster; the clusters of
/// one superfile COALESCE into a single open and one fetch batch of their byte
/// ranges (`probe_clusters_at_async`) — so a query is ~nprobe contiguous cluster
/// range-GETs, independent of superfile size. There is no whole-cell rescan.
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

    // Group admitted leaves by their owning superfile: every leaf is a single
    // internal IVF cluster, so the clusters of one superfile coalesce into one
    // open + one fetch batch of their byte ranges. `order` preserves first-seen
    // order for determinism. Empty clusters (count 0) carry no rows — skip them.
    let mut order: Vec<u128> = Vec::new();
    let mut by_sf: HashMap<u128, (Arc<SuperfileEntry>, Vec<(u32, u32, u32)>)> = HashMap::new();
    for (leaf, _dist, entry) in leaves {
        if leaf.count == 0 {
            continue;
        }
        let slot = by_sf.entry(leaf.superfile_id).or_insert_with(|| {
            order.push(leaf.superfile_id);
            (Arc::clone(&entry), Vec::new())
        });
        slot.1.push((leaf.cluster_id, leaf.doc_off, leaf.count));
    }

    // Resident deleted-user-`_id` set. The hidden cells are NOT rewritten on a
    // user delete, so the deleted rows stay physically present in them. This
    // consolidated set — written on the delete commit, loaded through the disk
    // cache (warm = zero GETs) — is the authoritative vector-search delete
    // filter. It is mapped to a per-superfile deny-set of LOCAL doc-ids and
    // pushed into the per-cell coarse-scan kernel, so each cell's top-k is
    // selected from LIVE rows (this preserves recall@k under deletes). Both
    // mappings land in the kernel's DOC_ID space: a contiguous-span superfile
    // (a registered INCOMING user superfile) maps `_id → _id - id_min` by
    // arithmetic, no read; a gapped cell maps through its inline `_id` region.
    // A post-merge backstop on the resolved `stable_id` (below) covers any cell
    // neither mapping reached.
    let manifest = reader.manifest();
    let deleted: Vec<i128> = match (manifest.opann_routing(), manifest.options.storage.as_ref()) {
        (Some(routing), Some(storage)) if routing.deleted_ids_uri.is_some() => {
            store::load_deleted_ids(routing, storage.as_ref(), manifest.options.disk_cache.as_ref())
                .await
                .map_err(|e| QueryError::Store(format!("deleted-set load: {e}")))?
        }
        _ => Vec::new(),
    };
    let deleted = Arc::new(deleted);

    // For filtered search a cell whose predicate matched no row is absent from
    // `allow` and dropped — it never opens or fetches.
    let mut units: Vec<(
        Arc<SuperfileEntry>,
        (Vec<(u32, u32, u32)>, Option<Arc<RoaringBitmap>>, Option<Arc<RoaringBitmap>>),
    )> = Vec::new();
    for sid in order {
        let (entry, metas) = by_sf.remove(&sid).expect("present in by_sf");
        let bitmap = match allow.as_ref() {
            Some(m) => match m.get(&entry.uri) {
                Some(bm) => Some(Arc::clone(bm)),
                None => continue,
            },
            None => None,
        };
        // Pre-heap deny for a contiguous-span (INCOMING user) superfile, mapped
        // arithmetically here while its entry is in scope. `None` means the
        // kernel maps through the cell's inline `_id` region instead.
        let arith_deny = arith_deny_locals(&entry, &deleted);
        units.push((entry, (metas, bitmap, arith_deny)));
    }
    if units.is_empty() {
        return Ok(Vec::new());
    }

    let column = Arc::new(column.to_owned());
    let query = Arc::new(query.to_vec());
    let deleted_for_kernel = Arc::clone(&deleted);
    let kernel = move |r: Arc<SuperfileReader>,
                       (metas, bitmap, arith_deny): (
        Vec<(u32, u32, u32)>,
        Option<Arc<RoaringBitmap>>,
        Option<Arc<RoaringBitmap>>,
    )| {
        let column = Arc::clone(&column);
        let query = Arc::clone(&query);
        let deleted = Arc::clone(&deleted_for_kernel);
        async move {
            let v = r.vec().ok_or_else(|| {
                QueryError::Store("hidden cell superfile missing vector subsection".into())
            })?;
            // Per-cell deny-set of LOCAL doc-ids (deleted user `_id`s). A
            // contiguous-span INCOMING superfile is mapped arithmetically by the
            // caller (`arith_deny`); a gapped cell maps through its inline `_id`
            // region here. Either way the set is in the kernel's DOC_ID space.
            let deny = match arith_deny {
                Some(d) => Some(d),
                None => v.inline_deleted_locals(&deleted).await.map_err(map_vector_err)?.map(Arc::new),
            };
            // Every leaf is a cluster range; fetch the admitted clusters as
            // contiguous range-GETs and score them. No whole-cell rescan.
            v.probe_clusters_at_async(&column, &query, k, &metas, rerank_mult, bitmap, deny)
                .await
                .map_err(map_vector_err)
        }
    };

    let mut per_superfile = dispatch::fanout_untombstoned(reader, units, kernel).await?;
    if !deleted.is_empty() {
        // Post-merge backstop (covers INCOMING cells with no inline `_id`
        // region). `deleted` is stored ascending, so `binary_search` is the
        // membership test.
        for hits in per_superfile.iter_mut() {
            hits.retain(|h| match h.stable_id {
                Some(id) => deleted.binary_search(&id).is_err(),
                None => true,
            });
        }
    }
    // SPANN replication writes a boundary row into several cells, so the same
    // row can surface from more than one probed cell. Dedup by stable_id (keep
    // the best/min score per row) so replicas don't waste top-k slots. Hits
    // with no stable_id (INCOMING cells, no inline `_id`) pass through.
    let mut best: HashMap<i128, SuperfileHit> = HashMap::new();
    let mut passthrough: Vec<SuperfileHit> = Vec::new();
    for hit in per_superfile.into_iter().flatten() {
        match hit.stable_id {
            Some(id) => {
                let replace = best.get(&id).map(|h| hit.score < h.score).unwrap_or(true);
                if replace {
                    best.insert(id, hit);
                }
            }
            None => passthrough.push(hit),
        }
    }
    let deduped: Vec<SuperfileHit> = best.into_values().chain(passthrough).collect();
    Ok(super::vector::top_k_ascending(vec![deduped], k))
}

fn map_vector_err(e: VectorError) -> QueryError {
    QueryError::Store(format!("vector leaf probe: {e}"))
}

/// Pre-heap deny-set (in DOC_ID space) for a contiguous-span superfile: row `i`
/// carries `_id == id_min + i`, so a deleted `_id` in `[id_min, id_max]` maps to
/// DOC_ID `_id - id_min` by arithmetic — the exact space
/// `score_cluster_codes_into_heap` tests `deny` in, with no inline `_id` region
/// and no read. `None` when the span is gapped (the caller maps through the
/// cell's inline `_id` region instead) or nothing in `deleted` falls in range.
fn arith_deny_locals(entry: &SuperfileEntry, deleted: &[i128]) -> Option<Arc<RoaringBitmap>> {
    if deleted.is_empty() || row_id_from_manifest_entry(entry, 0).is_none() {
        return None;
    }
    let (id_min, id_max) = (entry.id_min, entry.id_max);
    let bm: RoaringBitmap = deleted
        .iter()
        .filter(|&&d| d >= id_min && d <= id_max)
        .map(|&d| (d - id_min) as u32)
        .collect();
    (!bm.is_empty()).then(|| Arc::new(bm))
}
