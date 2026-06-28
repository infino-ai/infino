// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! OPANN vector fetch path: the radius-bounded tree descent admits leaves; each
//! admitted leaf is fetched with a direct range GET on its superfile object (no
//! Parquet footer, no whole-cell IVF centroid scan) and reranked.

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

/// Coverage floor for the radius-bounded descent, as a multiple of `k`. The
/// descent (see [`PagedTree::radius_bounded_descent`]) admits clusters by their
/// optimistic lower bound `(d − radius)` and freezes its prune bound to the far
/// edge of the nearest clusters once they cumulatively cover this multiple of
/// `k` vectors.
///
/// Why a coverage floor and not a pure radius threshold: a cell's radius is its
/// *intra*-cluster spread; it says nothing about how far the sibling IVF cells
/// of the same semantic region sit from each other. When a region is fragmented
/// across superfiles, the top-k spread across those siblings and the nearest
/// cell's radius can't reach them — a `d* + slack·r*` threshold collapses to
/// ~`d*` and starves admission. A doc-count floor instead pulls in as many of
/// the nearest (by optimistic bound) clusters as it takes to cover ~`k`
/// candidates: more small sibling clusters when fragmented (pre-drain), far fewer
/// once consolidated (post-drain). No upper cap — fan-out tracks the data's
/// fragmentation, not a fixed probe budget.
///
/// The floor is generous (32×) on purpose: the descent's prune bound derives
/// from it on **resident** metadata (Sq8 centroid distances + tree radii, no
/// payload fetch), so a wide floor keeps that bound a conservative cover of the
/// true k-th NN despite Sq8 distance error. That conservativeness is what lets
/// the kernel fetch the **whole** candidate set in ONE coalesced wave and rerank
/// — no fetch-time confirmation round-trip — and still hold recall.
const ADMIT_COVERAGE_K_MULT: usize = 32;

/// Descend the resident OPANN tree and return the query's candidate cluster
/// leaves — the radius-bounded admission — each paired with its superfile entry.
/// The descent prunes by covering radius and freezes its bound on resident
/// metadata alone (no payload fetch), and that bound conservatively covers the
/// true k-th NN, so the caller fetches this **whole** set in one coalesced wave
/// and reranks: there is no fetch-time confirmation pass and no second wave.
pub(super) async fn select_opann_leaves(
    _reader: &SupertableReader,
    manifest: &Manifest,
    query: &[f32],
    k: usize,
    _options: &VectorSearchOptions,
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

        // Radius-bounded descent: prune any subtree whose covering ball can't
        // reach the bound (the far edge of the nearest clusters covering `floor`
        // vectors — resident metadata, no payload fetched), returning only the
        // clusters that could hold a top-k vector — P(q), not M. No probe budget.
        let floor = ADMIT_COVERAGE_K_MULT.saturating_mul(k);
        let candidates = PagedTree::new(source, root)
            .radius_bounded_descent(query, floor, &survives)
            .map_err(|e| QueryError::Store(format!("opann descent: {e}")))?;

        let entry_by_id: HashMap<u128, Arc<SuperfileEntry>> = entries
            .iter()
            .map(|e| (e.superfile_id.as_u128(), Arc::clone(e)))
            .collect();

        // Every leaf is a single internal IVF cluster the tree routed straight to
        // (centroid → cluster byte range), fetched directly with no
        // within-superfile rescan. Keep only entries carrying the direct-fetch
        // layout.
        out.reserve(candidates.len());
        for (leaf, dist) in candidates {
            if let Some(entry) = entry_by_id.get(&leaf.superfile_id)
                && entry_has_opann_fetch_layout(entry)
            {
                out.push((leaf, dist, Arc::clone(entry)));
            }
        }
    }

    if out.is_empty() {
        return Ok(None);
    }
    Ok(Some(out))
}

fn entry_has_opann_fetch_layout(entry: &SuperfileEntry) -> bool {
    entry
        .subsection_offsets
        .as_ref()
        .and_then(|o| o.vec)
        .is_some_and(|(_, len)| len > 0)
}

/// Fetch the OPANN-admitted leaves across hidden cells, reusing the **normal**
/// superfile read path. [`dispatch::fanout`] opens each cell through the same
/// tiered opener the user-table fan-outs use (in-memory reader cache →
/// disk-cache mmap → storage fallback), warms + applies the tombstone sidecar,
/// and tags hits with their superfile — so the cell bytes are mmap-backed and
/// shared. Every admitted leaf is a single internal IVF cluster; the clusters of
/// one superfile COALESCE into a single open and one fetch batch of their byte
/// ranges (`probe_clusters_at_async`, the IVF cluster scan) — so a query is P(q)
/// coalesced range-GETs (one per touched superfile), independent of superfile
/// size. There is no whole-cell rescan.
pub(super) async fn fetch_opann_leaves(
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
    Ok(dedup_and_top_k(
        per_superfile.into_iter().flatten().collect(),
        k,
    ))
}

/// Dedup hits by `stable_id` (keep the min-score replica; pass through hits with
/// no stable_id), then return the global top-`k` ascending by score.
///
/// SPANN replication writes a boundary row into several cells, so the same row
/// can surface from more than one probed cell — and, across a confirmation pass,
/// from cells in different fetch batches. Deduping by `stable_id` keeps the best
/// score per row so replicas don't waste top-k slots.
fn dedup_and_top_k(hits: Vec<SuperfileHit>, k: usize) -> Vec<SuperfileHit> {
    let mut best: HashMap<i128, SuperfileHit> = HashMap::new();
    let mut passthrough: Vec<SuperfileHit> = Vec::new();
    for hit in hits {
        match hit.stable_id {
            Some(id) => {
                if best.get(&id).map(|h| hit.score < h.score).unwrap_or(true) {
                    best.insert(id, hit);
                }
            }
            None => passthrough.push(hit),
        }
    }
    let deduped: Vec<SuperfileHit> = best.into_values().chain(passthrough).collect();
    super::vector::top_k_ascending(vec![deduped], k)
}


fn map_vector_err(e: VectorError) -> QueryError {
    QueryError::Store(format!("vector leaf fetch: {e}"))
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
