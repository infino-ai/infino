// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! OPANN vector fetch path: the radius-bounded tree descent admits leaves; each
//! admitted leaf is fetched with a direct range GET on its superfile object (no
//! Parquet footer, no whole-cell IVF centroid scan) and reranked.

use std::{
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
        opann::{
            live::{self, undrained_user_vector_entries},
            page::LeafRef,
            paged::{ResidentPageSource, ScoredLeaf, radius_bounded_admit},
            store,
        },
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
    reader: &SupertableReader,
    manifest: &Manifest,
    query: &[f32],
    k: usize,
    _options: &VectorSearchOptions,
    survives: impl Fn(u128) -> bool + Copy,
) -> Result<Option<Vec<(LeafRef, f32, Arc<SuperfileEntry>)>>, QueryError> {
    let floor = ADMIT_COVERAGE_K_MULT.saturating_mul(k);
    let metric = reader
        .options()
        .vector_columns
        .first()
        .map(|c| c.metric)
        .unwrap_or(crate::superfile::vector::distance::Metric::L2Sq);
    let column = reader
        .options()
        .vector_columns
        .first()
        .map(|c| c.column.as_str());

    // Incoming buffer (undrained user superfiles, both pre- and post-drain): a
    // flat SIMD scan of the manifest-resident cluster centroids — no tree build,
    // no object-store GET — scored and near-sorted. This is the un-indexed tail of
    // the routing tree; the balanced tree is a batch artifact of drain/compact,
    // and the live path never builds or rebuilds a tree on the query/commit path.
    let extra: Vec<ScoredLeaf> = match (column, reader.hidden_parent_user_manifest()) {
        (Some(column), Some(user_manifest)) => {
            let undrained = undrained_user_vector_entries(
                user_manifest.as_ref(),
                manifest.opann_routing(),
                column,
            );
            if undrained.is_empty() {
                Vec::new()
            } else {
                live::score_undrained_manifest_clusters(&undrained, column, metric, query)
            }
        }
        _ => Vec::new(),
    };

    // ONE radius-bounded admission over the logical tree = persisted routing pages
    // (post-drain) ∪ the incoming list, under a single shared `CoverageBound`. The
    // tree pointer advances by node (frontier heap); the incoming-list pointer
    // advances linearly over `extra`. The tree is absent pre-drain (`opann_routing`
    // is `None` until the first drain), so the same bound then governs the incoming
    // list alone — never a separate per-source floor.
    let candidates: Vec<(LeafRef, f32)> = if let Some(routing) = manifest.opann_routing()
        && let Some(source) = reader.ensure_opann_resident_tree().await?
    {
        radius_bounded_admit(Some((source.as_ref(), routing.root_page)), &extra, query, floor, survives)
            .map_err(|e| QueryError::Store(format!("opann admit: {e}")))?
    } else {
        radius_bounded_admit::<ResidentPageSource>(None, &extra, query, floor, survives)
            .map_err(|e| QueryError::Store(format!("opann admit: {e}")))?
    };

    if candidates.is_empty() {
        return Ok(None);
    }

    let admitted_ids: Vec<u128> = candidates
        .iter()
        .map(|(leaf, _)| leaf.superfile_id)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let entry_by_id = resolve_opann_leaf_entries(reader, manifest, &admitted_ids).await?;

    let mut out: Vec<(LeafRef, f32, Arc<SuperfileEntry>)> = Vec::new();
    out.reserve(candidates.len());
    for (leaf, dist) in candidates {
        if let Some(entry) = entry_by_id.get(&leaf.superfile_id)
            && entry_has_opann_fetch_layout(entry)
        {
            out.push((leaf, dist, Arc::clone(entry)));
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

/// Map admitted OPANN leaf superfile ids to entries. Undrained user-table
/// superfiles are resolved from the parent manifest (no GET); hidden cells
/// load only the parts that contain each id — not the full list flatten.
async fn resolve_opann_leaf_entries(
    reader: &SupertableReader,
    manifest: &Manifest,
    superfile_ids: &[u128],
) -> Result<HashMap<u128, Arc<SuperfileEntry>>, QueryError> {
    let mut out = HashMap::with_capacity(superfile_ids.len());
    if let Some(column) = reader
        .options()
        .vector_columns
        .first()
        .map(|c| c.column.as_str())
        && let Some(user_manifest) = reader.hidden_parent_user_manifest()
    {
        for entry in crate::supertable::opann::live::undrained_user_vector_entries(
            user_manifest.as_ref(),
            manifest.opann_routing(),
            column,
        ) {
            let id = entry.superfile_id.as_u128();
            if superfile_ids.contains(&id) {
                out.insert(id, entry);
            }
        }
    }
    for &id in superfile_ids {
        if out.contains_key(&id) {
            continue;
        }
        let Some(entry) = manifest
            .lookup_superfile_entry_by_id(uuid::Uuid::from_u128(id))
            .await
            .map_err(QueryError::ManifestLoad)?
        else {
            continue;
        };
        out.insert(id, entry);
    }
    Ok(out)
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
    // mappings land in the kernel's DOC_ID space: a contiguous-span user-staged
    // superfile maps `_id → _id - id_min` by arithmetic, no read; a hidden cell
    // maps through its inline `_id` region piggybacked on the cluster fetch wave
    // (same stash as remap). A post-merge backstop on the resolved `stable_id`
    // (below) covers any cell neither mapping reached.
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
        (
            Vec<(u32, u32, u32)>,
            Option<Arc<RoaringBitmap>>,
            Option<Arc<RoaringBitmap>>,
            Option<Arc<Vec<i128>>>,
        ),
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
        // Pre-heap deny for a user-staged (contiguous-span) superfile only.
        // Hidden cells resolve deny from the inline `_id` region inside probe.
        let arith_deny = arith_deny_locals(&entry, &deleted);
        let deleted_for_probe = if arith_deny.is_none() && !deleted.is_empty() {
            Some(Arc::clone(&deleted))
        } else {
            None
        };
        units.push((entry, (metas, bitmap, arith_deny, deleted_for_probe)));
    }
    if units.is_empty() {
        return Ok(Vec::new());
    }

    let column = Arc::new(column.to_owned());
    let query = Arc::new(query.to_vec());
    let kernel = move |r: Arc<SuperfileReader>,
                       (metas, bitmap, arith_deny, deleted_for_probe): (
        Vec<(u32, u32, u32)>,
        Option<Arc<RoaringBitmap>>,
        Option<Arc<RoaringBitmap>>,
        Option<Arc<Vec<i128>>>,
    )| {
        let column = Arc::clone(&column);
        let query = Arc::clone(&query);
        async move {
            let v = r.vec().ok_or_else(|| {
                QueryError::Store("hidden cell superfile missing vector subsection".into())
            })?;
            v.probe_clusters_at_async(
                &column,
                &query,
                k,
                &metas,
                rerank_mult,
                bitmap,
                arith_deny,
                deleted_for_probe,
            )
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

/// Pre-heap deny-set (in DOC_ID space) for a user-staged superfile whose rows
/// are a contiguous `_id` span: row `i` carries `_id == id_min + i`, so a
/// deleted `_id` in `[id_min, id_max]` maps to DOC_ID `_id - id_min` by
/// arithmetic — the exact space `score_cluster_codes_into_heap` tests `deny`
/// in, with no inline `_id` region and no read. Hidden per-cell outputs do not
/// qualify (gapped inline region); the probe path maps those instead.
fn arith_deny_locals(entry: &SuperfileEntry, deleted: &[i128]) -> Option<Arc<RoaringBitmap>> {
    if deleted.is_empty()
        || !entry.is_user_staged_for_hidden_index()
        || row_id_from_manifest_entry(entry, 0).is_none()
    {
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
