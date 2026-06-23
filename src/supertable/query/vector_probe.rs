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

use bytes::Bytes;
use futures::future::try_join_all;
use roaring::RoaringBitmap;

use super::{SuperfileHit, dispatch};
use crate::{
    get_meter::{GetPhaseGuard, GET_PHASE_LEAF_PROBE, GET_PHASE_VEC_OPEN},
    storage::{StorageError, StorageProvider},
    superfile::{
        LazyByteSource, LazySubSource, PrefetchedSource, VectorError,
        builder::{VectorConfig, vec_columns_json},
        vector::reader::{OpenOptions, VectorReader},
    },
    supertable::{
        error::QueryError,
        handle::SupertableReader,
        lazy_source::StorageRangeSource,
        manifest::{Manifest, SubsectionOffsets, SuperfileEntry, SuperfileUri},
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

    let candidates = PagedTree::new(source, root)
        .select_probes_where(query, routing.nprobe_max, &survives)
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
            return Ok(None);
        }
        out.push((leaf, dist, Arc::clone(entry)));
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

/// Open a [`VectorReader`] on the vector subsection only — manifest offsets
/// and optional `open_blob` supply open-time bytes; Parquet is never read.
async fn open_vector_reader_for_probe(
    storage: &Arc<dyn StorageProvider>,
    entry: &SuperfileEntry,
    vector_columns: &[VectorConfig],
) -> Result<Arc<VectorReader>, QueryError> {
    let offsets = entry
        .subsection_offsets
        .as_ref()
        .ok_or_else(|| QueryError::Store("vector probe needs subsection_offsets".into()))?;
    let (vec_off, vec_len) = offsets
        .vec
        .ok_or_else(|| QueryError::Store("vector probe needs vec offset".into()))?;
    let uri = entry.uri.storage_path();
    let overlay = build_vector_open_overlay(storage, &uri, offsets).await?;
    let sub: Arc<dyn LazyByteSource> = Arc::new(LazySubSource::new(overlay, vec_off, vec_len));
    let cols_json = vec_columns_json(vector_columns);
    let reader = VectorReader::open_lazy(
        sub,
        &cols_json,
        OpenOptions::for_object_store(),
    )
    .await
    .map_err(|e| QueryError::Store(format!("vector probe open: {e}")))?;
    Ok(Arc::new(reader))
}

async fn build_vector_open_overlay(
    storage: &Arc<dyn StorageProvider>,
    uri: &str,
    offsets: &SubsectionOffsets,
) -> Result<Arc<dyn LazyByteSource>, QueryError> {
    let inner: Arc<dyn LazyByteSource> = Arc::new(StorageRangeSource::with_known_size(
        Arc::clone(storage),
        uri.to_owned(),
        offsets.total_size,
    ));
    let mut overlay = PrefetchedSource::new(inner);
    for (off, bytes) in &offsets.open_blob {
        overlay.install(*off, Bytes::copy_from_slice(bytes));
    }
    let mut missing: Vec<(u64, u64)> = Vec::new();
    for &(off, len) in &offsets.vec_open_ranges {
        if len == 0 {
            continue;
        }
        if overlay.try_get_range_sync(off, len).is_none() {
            missing.push((off, len));
        }
    }
    if !missing.is_empty() {
        let _phase = GetPhaseGuard::new(GET_PHASE_VEC_OPEN);
        let fetched = try_join_all(missing.iter().map(|&(off, len)| {
            let storage = Arc::clone(storage);
            let uri = uri.to_owned();
            async move {
                storage
                    .get_range(&uri, off..off + len)
                    .await
                    .map(|bytes| (off, bytes))
                    .map_err(map_storage_err)
            }
        }))
        .await?;
        for (off, bytes) in fetched {
            overlay.install(off, bytes);
        }
    }
    Ok(Arc::new(overlay))
}

fn map_storage_err(e: StorageError) -> QueryError {
    QueryError::Store(format!("vector probe range GET: {e}"))
}

/// Fan out direct leaf probes: one [`VectorReader`] open per superfile (zero
/// GET when `open_blob` is present), then parallel `probe_leaf_async` per
/// admitted leaf.
pub(super) async fn fanout_opann_leaf_probes(
    reader: &SupertableReader,
    leaves: Vec<(LeafRef, f32, Arc<SuperfileEntry>)>,
    column: &str,
    query: &[f32],
    k: usize,
    options: VectorSearchOptions,
    allow: Option<HashMap<SuperfileUri, Arc<RoaringBitmap>>>,
) -> Result<Vec<SuperfileHit>, QueryError> {
    let manifest = reader.manifest();
    let storage = manifest
        .options
        .storage
        .as_ref()
        .ok_or_else(|| QueryError::Store("vector probe needs storage".into()))?;
    let filtered = allow.is_some();
    let rerank_mult = options.resolve(filtered).1;

    let whole_cell = manifest.options.is_hidden_vector_index;

    let mut by_superfile: HashMap<u128, (Arc<SuperfileEntry>, Vec<(u32, u32)>)> = HashMap::new();
    for (leaf, _, entry) in leaves {
        by_superfile
            .entry(leaf.superfile_id)
            .or_insert_with(|| (Arc::clone(&entry), Vec::new()))
            .1
            .push((leaf.doc_off, leaf.count));
    }

    // Legacy manifests may still carry per-cluster hidden leaves; collapse to
    // one whole-cell probe per superfile until re-ingested with write-time
    // `(0, 0)` stamping.
    if whole_cell {
        for (_id, (_entry, probes)) in by_superfile.iter_mut() {
            *probes = vec![(0, 0)];
        }
    }

    let column = Arc::new(column.to_owned());
    let query = Arc::new(query.to_vec());
    let vector_columns = manifest.options.vector_columns.clone();

    let superfile_jobs: Vec<_> = by_superfile.into_iter().collect();
    let _leaf_phase = GetPhaseGuard::new(GET_PHASE_LEAF_PROBE);
    let per_superfile = try_join_all(superfile_jobs.into_iter().map(
        |(_superfile_id, (entry, probe_jobs))| {
            let storage = Arc::clone(storage);
            let column = Arc::clone(&column);
            let query = Arc::clone(&query);
            let allow = allow.clone();
            let vector_columns = vector_columns.clone();
            async move {
                let vec_reader =
                    open_vector_reader_for_probe(&storage, &entry, &vector_columns).await?;
                let bitmap = allow.as_ref().and_then(|m| m.get(&entry.uri).cloned());
                let leaf_hits = try_join_all(probe_jobs.into_iter().map(|(doc_off, count)| {
                    let vec_reader = Arc::clone(&vec_reader);
                    let column = Arc::clone(&column);
                    let query = Arc::clone(&query);
                    let bitmap = bitmap.clone();
                    let entry = Arc::clone(&entry);
                    async move {
                        let hits = vec_reader
                            .probe_leaf_async(
                                &column,
                                &query,
                                k,
                                doc_off,
                                count,
                                rerank_mult,
                                bitmap,
                            )
                            .await
                            .map_err(map_vector_err)?;
                        Ok::<_, QueryError>(dispatch::tag_hits(&entry, hits))
                    }
                }))
                .await?;
                let mut merged: Vec<SuperfileHit> = Vec::new();
                for batch in leaf_hits {
                    merged.extend(batch);
                }
                Ok(merged)
            }
        },
    ))
    .await?;

    let mut all: Vec<SuperfileHit> = Vec::new();
    for batch in per_superfile {
        all.extend(batch);
    }
    Ok(super::vector::top_k_ascending(all, k))
}

fn map_vector_err(e: VectorError) -> QueryError {
    QueryError::Store(format!("vector leaf probe: {e}"))
}
