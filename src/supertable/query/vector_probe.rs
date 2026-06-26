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
    superfile::{SuperfileReader, VectorError, vector::distance::Metric},
    supertable::{
        error::QueryError,
        handle::{INCOMING_VECTOR_CELL, SupertableReader},
        manifest::{Manifest, SuperfileEntry, SuperfileUri},
        opann::{page::LeafRef, paged::PagedTree, store},
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

    // Metric for scoring each admitted cell's resident per-cluster centroids.
    let metric = manifest
        .options
        .vector_columns
        .iter()
        .find(|vc| vc.column == column)
        .map(|vc| vc.metric)
        .unwrap_or(Metric::L2Sq);

    let mut out = Vec::with_capacity(admitted.len());
    for (leaf, dist) in admitted {
        let Some(entry) = entry_by_id.get(&leaf.superfile_id) else {
            continue;
        };
        if !entry_has_vector_probe_layout(entry) {
            continue;
        }
        // Per-cell LOCAL cluster selection. The tree already localized to this
        // cell; now score its OWN resident cluster centroids, take the nearest
        // (adaptive, radius-aware), and emit one offset leaf per chosen cluster —
        // its byte range coming from the resident per-cluster offset + count. A
        // query then range-GETs only those clusters, so fetch cost is independent
        // of cell size. Falls back to a whole-cell `(0, 0)` probe when the cell
        // carries no resident per-cluster offsets (e.g. legacy/incoming cells).
        match entry.vector_summary.get(column) {
            Some(vs)
                if vs.clusters.n_cent > 0
                    && vs.cluster_offsets.len() == vs.clusters.counts.len()
                    && !vs.cluster_offsets.is_empty() =>
            {
                for c in vs.clusters.select_cells_adaptive(metric, query, floor, routing) {
                    let ci = c as usize;
                    out.push((
                        LeafRef {
                            superfile_id: leaf.superfile_id,
                            doc_off: vs.cluster_offsets[ci],
                            count: vs.clusters.counts[ci],
                            cluster_id: c,
                        },
                        dist,
                        Arc::clone(entry),
                    ));
                }
            }
            _ => out.push((leaf, dist, Arc::clone(entry))),
        }
    }

    // Geometric-drain adaptation (no opann counterpart — opann dual-writes per
    // vector, so it has no staging backlog): rows appended since the last drain
    // live in INCOMING staging superfiles the routing tree has not clustered
    // yet. The tree can't route to them, so always probe them — whole-scan
    // `(doc_off, count) = (0, 0)` — in this same parallel wave, keeping
    // un-drained rows searchable before (and between) drains. `survives` gates
    // them on the same predicate as the routed leaves.
    let mut seen: HashSet<u128> = out.iter().map(|(l, _, _)| l.superfile_id).collect();
    for entry in &entries {
        if entry.partition_hint == Some(INCOMING_VECTOR_CELL)
            && survives(entry.superfile_id.as_u128())
            && entry_has_vector_probe_layout(entry)
            && seen.insert(entry.superfile_id.as_u128())
        {
            let leaf = LeafRef {
                superfile_id: entry.superfile_id.as_u128(),
                doc_off: 0,
                count: 0,
                cluster_id: 0,
            };
            out.push((leaf, 0.0, Arc::clone(entry)));
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

/// How to probe one hidden cell superfile.
enum ProbePlan {
    /// Legacy / incoming / compaction-merge `(0,0)` leaf: rescan the whole cell
    /// (cluster index + every non-empty internal cluster).
    WholeCell,
    /// OPANN offset leaves admitted for this cell — `(cluster_id, doc_off,
    /// count)` per internal cluster, fetched as contiguous range-GETs.
    Clusters(Vec<(u32, u32, u32)>),
}

/// Fan the OPANN-admitted probes across hidden cells, reusing the **normal**
/// superfile read path. [`dispatch::fanout`] opens each cell through the same
/// tiered opener the user-table fan-outs use (in-memory reader cache →
/// disk-cache mmap → storage fallback), warms + applies the tombstone sidecar,
/// and tags hits with their superfile — so the cell bytes are mmap-backed and
/// shared. Admitted offset leaves of one cell COALESCE into a single open and
/// one fetch batch of their cluster ranges (`probe_clusters_at_async`) — so a
/// query is ~nprobe contiguous cluster range-GETs, independent of cell size —
/// while a `(0,0)` leaf falls back to the whole-cell IVF rescan.
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

    // Group admitted leaves by their owning superfile: per-cluster offset leaves
    // of one cell coalesce into one open + one fetch batch; a `(0,0)` leaf marks
    // the whole cell. `order` preserves first-seen order for determinism.
    let mut order: Vec<u128> = Vec::new();
    let mut by_sf: HashMap<u128, (Arc<SuperfileEntry>, Vec<(u32, u32, u32)>, bool)> = HashMap::new();
    for (leaf, _dist, entry) in leaves {
        let slot = by_sf.entry(leaf.superfile_id).or_insert_with(|| {
            order.push(leaf.superfile_id);
            (Arc::clone(&entry), Vec::new(), false)
        });
        if leaf.doc_off == 0 && leaf.count == 0 {
            slot.2 = true;
        } else {
            slot.1.push((leaf.cluster_id, leaf.doc_off, leaf.count));
        }
    }

    // For filtered search a cell whose predicate matched no row is absent from
    // `allow` and dropped — it never opens or fetches.
    let mut units: Vec<(Arc<SuperfileEntry>, (ProbePlan, Option<Arc<RoaringBitmap>>))> = Vec::new();
    for sid in order {
        let (entry, metas, whole) = by_sf.remove(&sid).expect("present in by_sf");
        let bitmap = match allow.as_ref() {
            Some(m) => match m.get(&entry.uri) {
                Some(bm) => Some(Arc::clone(bm)),
                None => continue,
            },
            None => None,
        };
        let plan = if whole || metas.is_empty() {
            ProbePlan::WholeCell
        } else {
            ProbePlan::Clusters(metas)
        };
        units.push((entry, (plan, bitmap)));
    }
    if units.is_empty() {
        return Ok(Vec::new());
    }

    let column = Arc::new(column.to_owned());
    let query = Arc::new(query.to_vec());
    let kernel =
        move |r: Arc<SuperfileReader>, (plan, bitmap): (ProbePlan, Option<Arc<RoaringBitmap>>)| {
            let column = Arc::clone(&column);
            let query = Arc::clone(&query);
            async move {
                let v = r.vec().ok_or_else(|| {
                    QueryError::Store("hidden cell superfile missing vector subsection".into())
                })?;
                match plan {
                    ProbePlan::WholeCell => v
                        .probe_leaf_async(&column, &query, k, 0, 0, rerank_mult, bitmap)
                        .await
                        .map_err(map_vector_err),
                    ProbePlan::Clusters(metas) => v
                        .probe_clusters_at_async(&column, &query, k, &metas, rerank_mult, bitmap)
                        .await
                        .map_err(map_vector_err),
                }
            }
        };
    // Resident deleted-user-`_id` set. The hidden cells are NOT rewritten on a
    // user delete, so the deleted rows stay physically present and would
    // otherwise leak into results (on the `_id`-only path especially, which
    // does no per-superfile tombstone remap). This consolidated set — written
    // on the delete commit, loaded through the disk cache (warm = zero GETs) —
    // is the authoritative vector-search delete filter, applied uniformly on
    // every projection path. Each hidden hit already carries its resolved user
    // `_id` in `stable_id` (set during the fan-out), so the drop is in-memory.
    let manifest = reader.manifest();
    let deleted: Vec<i128> = match (manifest.opann_routing(), manifest.options.storage.as_ref()) {
        (Some(routing), Some(storage)) if routing.deleted_ids_uri.is_some() => {
            store::load_deleted_ids(routing, storage.as_ref(), manifest.options.disk_cache.as_ref())
                .await
                .map_err(|e| QueryError::Store(format!("deleted-set load: {e}")))?
        }
        _ => Vec::new(),
    };

    let mut per_superfile = dispatch::fanout_untombstoned(reader, units, kernel).await?;
    if !deleted.is_empty() {
        // `deleted` is stored ascending, so `binary_search` is the membership test.
        for hits in per_superfile.iter_mut() {
            hits.retain(|h| match h.stable_id {
                Some(id) => deleted.binary_search(&id).is_err(),
                None => true,
            });
        }
    }
    // Our `top_k_ascending` takes the per-superfile nested vec and flattens
    // internally (opann's takes a pre-flattened vec); pass it directly.
    Ok(super::vector::top_k_ascending(per_superfile, k))
}

fn map_vector_err(e: VectorError) -> QueryError {
    QueryError::Store(format!("vector leaf probe: {e}"))
}
