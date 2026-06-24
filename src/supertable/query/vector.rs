// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Vector kNN fan-out on [`Supertable`](super::super::Supertable).
//!
//! ## Public API
//!
//! The sync, user-facing entry points live on
//! [`Supertable`](super::super::Supertable):
//!
//! ```ignore
//! let opts = VectorSearchOptions::new();
//! // Bare call: `_id` + `score` only — no scalar decode.
//! let ids: Vec<RecordBatch> = table.vector_search("emb", &query_vec, 10, opts, None, None)?;
//! // Materialize row data by naming the columns to decode.
//! let rows: Vec<RecordBatch> = table.vector_search(
//!     "emb",
//!     &query_vec,
//!     10,
//!     opts,
//!     None,
//!     Some(&["_id", "title", "score"]),
//! )?;
//! ```
//!
//! Internally these drive the async kernel on the snapshot-pinned
//! [`SupertableReader`], whose `vector_search` (rows) / `vector_hits`
//! ([`SuperfileHit`], superfile-local) methods are the engine-facing
//! surface. Results are sorted by distance *ascending* — smaller is
//! closer (cosine: `1 - dot`, L2-sq: squared distance).
//!
//! ## Strategy
//!
//! Internally pins a snapshot reader and drives the async
//! kernel to completion via the sync→async bridge. The reader
//! holds a pinned `Arc<Manifest>`; for each visible superfile we:
//!
//!   1. Fetch the superfile's `SuperfileReader` from the store.
//!   2. Delegate to `SuperfileReader::vector_search`
//!      (cluster-aware IVF + 1-bit RaBitQ shortlist + full-precision
//!      rerank, all inside one superfile).
//!   3. Tag each `(local_doc_id, distance)` with the superfile URI.
//!   4. Concatenate across superfiles and global-top-k by distance.
//!
//! Unlike BM25, vector distances are inherently comparable across
//! superfiles — both cosine and L2-sq are functions of the query
//! and the per-doc vector only, not of superfile-scoped statistics.
//! So the per-superfile top-k → concatenate → global top-k pattern
//! recovers exact recall (modulo each per-superfile IVF's nprobe-
//! driven recall tradeoff, which is identical to the single-
//! superfile case).
//!
//! Fan-out uses centroid pruning:
//!
//!   1. **Score & sort** — compute `distance(query, centroid)`
//!      for each superfile (SIMD-accelerated: AVX-512 / AVX2 /
//!      NEON). Derive a lower bound per superfile:
//!      `max(0, centroid_dist − radius)`. Sort ascending.
//!      This is free — centroids are manifest metadata, no
//!      S3 GETs.
//!   2. **Search closest** — search the top `k*2` (min 3)
//!      superfiles in parallel (`tokio::spawn` per superfile).
//!      Merge results via bounded heap.
//!
//! Every skipped superfile is a batch of GET requests the
//! object-store-native engine never issues. For cold queries
//! this is the difference between seconds and milliseconds.

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
    future::Future,
    sync::Arc,
    time::Instant,
};

use arrow::record_batch::RecordBatch;
use arrow_array::{Array, Decimal128Array};
use roaring::RoaringBitmap;

use super::{
    SuperfileHit,
    candidate::CandidatePlan,
    dispatch,
    exec::common::{id_score_batch, resolve_hits_named, take_rows_object_store},
    hierarchical_iter,
    prune::{PruneLeaf, select_superfiles},
};
pub use crate::superfile::reader::VectorSearchOptions;
use crate::{
    storage::StorageProvider,
    superfile::{SuperfileReader, fts::reader::BoolMode, vector::distance::Metric},
    supertable::{
        error::QueryError,
        handle::{INCOMING_VECTOR_CELL, Supertable, SupertableReader},
        manifest::{
            Manifest, SuperfileEntry, SuperfileUri,
            list::{CellRoutingParams, PartitionStrategy},
        },
        tombstones::SidecarCache,
    },
};

/// An optional text-predicate filter for vector kNN search. When
/// supplied, kNN is ranked only among rows matching the predicate
/// (pushdown, not post-filter). Built from an FTS-indexed column, a
/// query string, and a [`BoolMode`].
pub struct VectorFilter<'a> {
    /// FTS-indexed column the predicate applies to.
    pub column: &'a str,
    /// Query string — tokenized with the index tokenizer.
    pub query: &'a str,
    /// Token matching mode (AND / OR).
    pub mode: BoolMode,
}

enum Probe {
    Clusters(Vec<u32>),
    Nprobe,
}

/// Apply query-time diagnostic overrides to the persisted cell-routing params.
/// `INFINO_CELL_NPROBE_MAX` caps (or sets) the adaptive probe ceiling without
/// rebuilding the index — set it equal to the nprobe floor to disable adaptive
/// expansion ("use the hint"), or sweep it to trade fan-out against recall.
/// Read once and cached.
fn routing_with_env_overrides(mut routing: CellRoutingParams) -> CellRoutingParams {
    use std::sync::OnceLock;
    static NPROBE_MAX: OnceLock<Option<usize>> = OnceLock::new();
    let override_max = *NPROBE_MAX.get_or_init(|| {
        std::env::var("INFINO_CELL_NPROBE_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
    });
    if let Some(n) = override_max {
        routing.nprobe_max = n.max(routing.nprobe_min);
    }
    routing
}

fn filter_superfiles_by_cells(
    superfiles: &[Arc<SuperfileEntry>],
    routed_cells: &[u32],
) -> Vec<Arc<SuperfileEntry>> {
    if routed_cells.is_empty() {
        return superfiles.to_vec();
    }
    let routed_keys: HashSet<[u8; 4]> = routed_cells.iter().map(|c| c.to_le_bytes()).collect();
    superfiles
        .iter()
        .filter(|sf| {
            if sf.partition_key.len() == 4 {
                let mut key = [0u8; 4];
                key.copy_from_slice(&sf.partition_key);
                routed_keys.contains(&key)
            } else if let Some(cell) = sf.partition_hint {
                routed_keys.contains(&cell.to_le_bytes())
            } else {
                false
            }
        })
        .cloned()
        .collect()
}

/// Whether every hit already references a superfile in the user manifest
/// (flat view or list parts — same source as query fan-out).
async fn hits_reference_user_superfiles(
    reader: &SupertableReader,
    hits: &[SuperfileHit],
) -> Result<bool, QueryError> {
    if hits.is_empty() {
        return Ok(true);
    }
    let manifest = reader.manifest();
    for hit in hits {
        if manifest
            .lookup_superfile_entry(hit.superfile)
            .await
            .map_err(QueryError::ManifestLoad)?
            .is_none()
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Extract the `_id` column (column 0, Decimal128) of `batch` as `Vec<i128>`.
fn id_values_from_batch(batch: &RecordBatch) -> Result<Vec<i128>, QueryError> {
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .map(|a| a.values().to_vec())
        .ok_or_else(|| QueryError::Execute("_id column missing".into()))
}

/// Resolve a stable row id from manifest span arithmetic when the superfile
/// body stores rows in contiguous id order. `None` when the id span is gapped
/// (not a single contiguous append), so the caller must read the `_id` column.
pub(crate) fn row_id_from_manifest_entry(
    entry: &SuperfileEntry,
    local_doc_id: u32,
) -> Option<i128> {
    let n_docs = i128::from(entry.n_docs);
    let span = entry.id_max.checked_sub(entry.id_min)?.checked_add(1)?;
    if n_docs == 0 || span != n_docs {
        return None;
    }
    Some(entry.id_min + i128::from(local_doc_id))
}

/// Sync stable-id resolution for incoming drain when superfile bytes are
/// resident (in-memory store, disk-cache mmap, or whole-object open).
pub(crate) fn stable_ids_for_incoming_drain_resident(
    entry: &SuperfileEntry,
    reader: &SuperfileReader,
) -> Result<Vec<i128>, QueryError> {
    if row_id_from_manifest_entry(entry, 0).is_some() {
        return Ok((0..entry.n_docs as u32)
            .map(|local| entry.id_min + i128::from(local))
            .collect());
    }
    let locals: Vec<u32> = (0..reader.n_docs() as u32).collect();
    if let Some(ids) = reader
        .vec()
        .and_then(|v| v.inline_stable_ids_for_locals(&locals))
    {
        return Ok(ids);
    }
    let id_column = reader.id_column();
    if reader.parquet_bytes().is_some() {
        let batch = reader
            .take_by_local_doc_ids(&locals, &[id_column])
            .map_err(|e| QueryError::Execute(e.to_string()))?;
        return id_values_from_batch(&batch);
    }
    Err(QueryError::Execute(
        "incoming drain requires resident superfile bytes".into(),
    ))
}

/// Stable `_id` for every row in `entry` (`local` → `id_min + local` when the
/// manifest span is contiguous, else targeted column reads). Same tier order as
/// [`hidden_hits_user_ids`]: span arithmetic → resident `take_by_local_doc_ids`
/// → [`read_ids_for_locals`].
pub(crate) async fn stable_ids_by_local_for_routing(
    manifest: &Manifest,
    entry: &SuperfileEntry,
    reader: &SuperfileReader,
) -> Result<Vec<i128>, QueryError> {
    if row_id_from_manifest_entry(entry, 0).is_some() {
        return Ok((0..entry.n_docs as u32)
            .map(|local| entry.id_min + i128::from(local))
            .collect());
    }
    let locals: Vec<u32> = (0..reader.n_docs() as u32).collect();
    // Hidden cell superfiles inline the stable `_id` in the IVF blob — resolve
    // straight from it (resident; no scalar `_id` column read) before falling
    // back to the column.
    if let Some(ids) = reader
        .vec()
        .and_then(|v| v.inline_stable_ids_for_locals(&locals))
    {
        return Ok(ids);
    }
    let id_column = reader.id_column();
    if reader.parquet_bytes().is_some() {
        let batch = reader
            .take_by_local_doc_ids(&locals, &[id_column])
            .map_err(|e| QueryError::Execute(e.to_string()))?;
        return id_values_from_batch(&batch);
    }
    read_ids_for_locals(manifest, entry, &locals, id_column).await
}

/// Read the `_id` column values at `local_ids` (in caller order) from one
/// superfile. Routed through the disk cache as a resident (mmap) read when a
/// cache is attached; falls back to object-store range GETs on lazy readers.
async fn read_ids_for_locals(
    manifest: &Manifest,
    entry: &SuperfileEntry,
    local_ids: &[u32],
    id_column: &str,
) -> Result<Vec<i128>, QueryError> {
    let storage = manifest
        .options
        .storage
        .as_ref()
        .ok_or_else(|| QueryError::Execute("id remap needs a storage backend".into()))?;
    let store = Arc::clone(&manifest.options.store);
    let disk_cache = manifest.options.disk_cache.as_ref();
    let reader = dispatch::open_reader(&store, disk_cache, Some(storage), entry).await?;
    // Hidden cell superfiles inline the stable `_id` in the IVF blob — resolve
    // straight from it (resident; no scalar `_id` column read) when available.
    if let Some(ids) = reader
        .vec()
        .and_then(|v| v.inline_stable_ids_for_locals(local_ids))
    {
        return Ok(ids);
    }
    // Cold path: the region is present but not resident (the search fetched
    // centroids/cluster_idx + blocks, never the region). Fetch it async and
    // index it — far cheaper than the scalar `_id` decode below, and the path
    // the inline-`_id` region was built for. (Warm hits the sync branch above.)
    if let Some(v) = reader.vec() {
        if let Some(ids) = v
            .inline_stable_ids_for_locals_async(local_ids)
            .await
            .map_err(|e| QueryError::Execute(e.to_string()))?
        {
            return Ok(ids);
        }
    }
    if reader.parquet_bytes().is_some() {
        let batch = reader
            .take_by_local_doc_ids(local_ids, &[id_column])
            .map_err(|e| QueryError::Execute(e.to_string()))?;
        return id_values_from_batch(&batch);
    }
    let batch = read_ids_batch_object_store(entry, local_ids, id_column, storage, &reader).await?;
    id_values_from_batch(&batch)
}

/// Read the `_id` column at `local_ids` from a superfile via object-store range
/// GETs. Works regardless of whether a reader is eager or lazy, so it serves
/// both the no-disk-cache path and the fallback when a cached reader is still a
/// lazy foreground reader (pre-mmap-promotion).
async fn read_ids_batch_object_store(
    entry: &SuperfileEntry,
    local_ids: &[u32],
    id_column: &str,
    storage: &Arc<dyn StorageProvider>,
    reader: &SuperfileReader,
) -> Result<arrow_array::RecordBatch, QueryError> {
    let (obj_store, path) = storage
        .object_store_handle(&entry.uri.storage_path())
        .ok_or_else(|| QueryError::Execute("no object_store handle for superfile".into()))?;
    let file_size = entry.subsection_offsets.as_ref().map(|o| o.total_size);
    take_rows_object_store(
        obj_store,
        path,
        file_size,
        reader.schema(),
        reader.n_docs(),
        local_ids,
        &[id_column],
    )
    .await
    .map_err(|e| QueryError::Execute(e.to_string()))
}

/// Remap step 1 (deduped): resolve the user `_id` that dual-write stamped into
/// each hidden-index hit, returned in `hidden_hits` order.
///
/// Arithmetic when a hidden superfile's id span is contiguous. The hidden index
/// is cell-partitioned, so a cell aggregates scattered user ids and its id
/// range is usually gapped — `id_min + local` rarely holds — so the common case
/// is a column read. Hits are grouped by hidden superfile so each gapped
/// superfile's `_id` column is read **once** (resident via the disk cache),
/// reading only the rows the hits touch — versus the previous per-hit
/// object-store read that dominated warm latency.
async fn hidden_hits_user_ids(
    hidden_manifest: &Manifest,
    hidden_hits: &[SuperfileHit],
    id_column: &str,
) -> Result<Vec<i128>, QueryError> {
    let mut ids = vec![0i128; hidden_hits.len()];
    let mut by_superfile: HashMap<SuperfileUri, Vec<usize>> = HashMap::new();
    for (i, hit) in hidden_hits.iter().enumerate() {
        // Piggyback fast path: the search already resolved the user `_id`
        // from the inline region (prefetched in the fan-out wave) and stamped
        // it here — reuse it and skip this superfile's region/scalar read
        // entirely. Hits without it (incoming superfiles have no inline
        // region) fall through to the grouped read below.
        if let Some(id) = hit.stable_id {
            ids[i] = id;
            continue;
        }
        by_superfile.entry(hit.superfile).or_default().push(i);
    }
    for (uri, idxs) in by_superfile {
        let entry = hidden_manifest
            .lookup_superfile_entry(uri)
            .await
            .map_err(QueryError::ManifestLoad)?
            .ok_or_else(|| {
                QueryError::Execute(format!("hidden superfile {uri:?} missing from manifest"))
            })?;
        // Contiguous span → arithmetic, no read.
        if row_id_from_manifest_entry(&entry, 0).is_some() {
            for &i in &idxs {
                ids[i] = entry.id_min + i128::from(hidden_hits[i].local_doc_id);
            }
            continue;
        }
        // Gapped span → one resident read of just the rows these hits touch.
        let locals: Vec<u32> = idxs.iter().map(|&i| hidden_hits[i].local_doc_id).collect();
        let vals = read_ids_for_locals(hidden_manifest, &entry, &locals, id_column).await?;
        for (j, &i) in idxs.iter().enumerate() {
            ids[i] = vals[j];
        }
    }
    Ok(ids)
}

/// `_id`-only hidden-index fast path. The caller wants only `_id` + `score`,
/// which remap step 1 already yields — so resolve the stable `_id` per hidden
/// hit and synthesize the batch directly, skipping remap steps 2/3 and the
/// user-superfile column resolve entirely (no user-table data-page read). Hits
/// are already in global rank order; the batch preserves it.
async fn hidden_hits_id_score_batch(
    user_reader: &SupertableReader,
    hidden_hits: &[SuperfileHit],
) -> Result<RecordBatch, QueryError> {
    let vit = user_reader
        .vector_index_table()
        .ok_or_else(|| QueryError::Execute("hidden vector index missing".into()))?;
    let vit_reader = vit.reader();
    let hidden_manifest: &Manifest = vit_reader.manifest();
    let id_column = user_reader.options().id_column.as_str();

    let ids = hidden_hits_user_ids(hidden_manifest, hidden_hits, id_column).await?;
    let scores: Vec<f32> = hidden_hits.iter().map(|h| h.score).collect();
    id_score_batch(user_reader, &ids, &scores).map_err(|e| QueryError::Execute(e.to_string()))
}

/// Map hidden-index `(superfile, local_doc_id)` hits to user-table hits
/// using aligned `_id` values stamped during dual-write.
async fn remap_hidden_hits_to_user_hits(
    user_reader: &SupertableReader,
    hidden_hits: &[SuperfileHit],
) -> Result<Vec<SuperfileHit>, QueryError> {
    if hidden_hits.is_empty() {
        return Ok(Vec::new());
    }
    let vit = user_reader
        .vector_index_table()
        .ok_or_else(|| QueryError::Execute("hidden vector index missing".into()))?;
    let vit_reader = vit.reader();
    let hidden_manifest = vit_reader.manifest();
    let user_manifest = user_reader.manifest();
    let id_column = user_reader.options().id_column.as_str();

    // Step 1: hidden hit → stable user `_id` (deduped, resident).
    let user_ids = hidden_hits_user_ids(hidden_manifest, hidden_hits, id_column).await?;

    // Step 2: user `_id` → (user superfile, local row). Resolve the owning
    // superfile by id range; arithmetic when its span is contiguous, else
    // binary-search the superfile's `_id` column — read once per superfile
    // (resident via the disk cache), grouped so a gapped superfile is never
    // re-read per hit. A future per-superfile id-run index would make the
    // gapped case O(1) and drop the column read entirely.
    let mut remapped: Vec<Option<SuperfileHit>> = vec![None; hidden_hits.len()];
    let mut gapped: HashMap<SuperfileUri, Vec<usize>> = HashMap::new();
    for (i, &user_row_id) in user_ids.iter().enumerate() {
        let user_entry = user_manifest
            .superfiles
            .iter()
            .find(|e| user_row_id >= e.id_min && user_row_id <= e.id_max)
            .ok_or_else(|| {
                QueryError::Execute(format!("no user superfile owns id {user_row_id}"))
            })?;
        if row_id_from_manifest_entry(user_entry, 0).is_some() {
            // Contiguous span (single-append): invert `id_min + local`.
            let local = u32::try_from(user_row_id - user_entry.id_min).map_err(|_| {
                QueryError::Execute(format!("local_doc_id out of range for id {user_row_id}"))
            })?;
            remapped[i] = Some(SuperfileHit {
                superfile: user_entry.uri,
                local_doc_id: local,
                score: hidden_hits[i].score,
                stable_id: None,
            });
        } else {
            gapped.entry(user_entry.uri).or_default().push(i);
        }
    }
    for (uri, idxs) in gapped {
        let user_entry = user_manifest
            .lookup_superfile_entry(uri)
            .await
            .map_err(QueryError::ManifestLoad)?
            .ok_or_else(|| {
                QueryError::Execute(format!("user superfile {uri:?} missing from manifest"))
            })?;
        let n = user_entry.n_docs as usize;
        let all_locals: Vec<u32> = (0..n as u32).collect();
        // Column is monotonic (ids minted in row order) → binary-searchable.
        let id_col =
            read_ids_for_locals(user_manifest, &user_entry, &all_locals, id_column).await?;
        for &i in &idxs {
            let user_row_id = user_ids[i];
            let pos = id_col.binary_search(&user_row_id).map_err(|_| {
                QueryError::Execute(format!("no row with id {user_row_id} in user superfile"))
            })?;
            remapped[i] = Some(SuperfileHit {
                superfile: uri,
                local_doc_id: pos as u32,
                score: hidden_hits[i].score,
                stable_id: None,
            });
        }
    }
    remapped
        .into_iter()
        .map(|h| h.ok_or_else(|| QueryError::Execute("hit remap incomplete".into())))
        .collect()
}

impl SupertableReader {
    /// Global cross-superfile cluster selection + waved fan-out. Shared
    /// by the user-table path and the hidden vector-index path.
    async fn fanout_vector_clusters(
        &self,
        superfiles: &[Arc<SuperfileEntry>],
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }
        self.vector_fanout_over_superfiles(superfiles.to_vec(), column, query, k, options, None)
            .await
    }

    async fn vector_fanout_over_superfiles(
        &self,
        superfiles: Vec<Arc<SuperfileEntry>>,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        allow: Option<HashMap<SuperfileUri, Arc<RoaringBitmap>>>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let filtered = allow.is_some();
        let (nprobe, _) = options.resolve(filtered);
        let manifest = self.manifest();

        // ---- Global cross-superfile cluster selection.
        //
        // Each kept superfile's manifest summary carries its per-cluster
        // (Sq8) centroids. Rank every (superfile, cluster) by centroid
        // distance to the query and probe only the globally-closest
        // clusters — so a query touches just the superfiles that own a
        // near cluster, instead of running `nprobe` in every superfile.
        // (A single per-superfile centroid can't do this: a time-ordered
        // superfile is a broad mix, so its mean sits near the global
        // centroid. Per-cluster centroids are fine-grained enough to
        // rank.) A superfile whose summary has no cluster centroids falls
        // back to a normal per-superfile `nprobe` probe — never dropped.
        let metric = manifest
            .options
            .vector_columns
            .iter()
            .find(|vc| vc.column == column)
            .map(|vc| vc.metric)
            .unwrap_or(Metric::L2Sq);

        let mut scored: Vec<(usize, u32, f32)> = Vec::new();
        let mut fallback: Vec<usize> = Vec::new();
        // Folded Sq8+residual scoring (`ClusterCentroids::score_clusters_into`):
        // one kernel per query, then one distance per cluster over stored bytes.
        for (si, entry) in superfiles.iter().enumerate() {
            // Filtered search: a superfile whose predicate matched no row
            // (absent from `allow`) is dropped here — it never scores a
            // cluster, never enters the fan-out, and issues zero GETs.
            if allow.as_ref().is_some_and(|m| !m.contains_key(&entry.uri)) {
                continue;
            }
            match entry.vector_summary.get(column) {
                Some(vs) if !vs.clusters.is_empty() && vs.clusters.dim as usize == query.len() => {
                    vs.clusters
                        .score_clusters_into(metric, query, |c, score| {
                            scored.push((si, c, score));
                        });
                }
                _ => fallback.push(si),
            }
        }

        // Global probe budget: the closest `nprobe × (eligible superfiles)`
        // clusters — the same total probe count as the old per-superfile
        // `nprobe`, but selected globally, so near superfiles get more
        // probes and far superfiles are skipped entirely. (Stage-4 recall
        // tuning may lower this.) Eligible = superfiles that actually
        // scored a cluster (filtered-out superfiles produce neither a
        // scored cluster nor a fallback, so they don't inflate the
        // budget).
        let n_eligible = {
            let mut segs: Vec<usize> = scored.iter().map(|&(si, _, _)| si).collect();
            segs.sort_unstable();
            segs.dedup();
            segs.len()
        };
        let budget = nprobe.saturating_mul(n_eligible.max(1)).max(nprobe);
        if scored.len() > budget {
            scored.select_nth_unstable_by(budget, |a, b| {
                a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal)
            });
            scored.truncate(budget);
        }
        let mut per_seg: HashMap<usize, Vec<u32>> = HashMap::new();
        for (si, c, _) in scored {
            per_seg.entry(si).or_default().push(c);
        }

        // Build fan-out units: selected superfiles probe their chosen
        // clusters; fallback superfiles probe `nprobe` normally; superfiles
        // with centroids but no globally-selected cluster are skipped
        // (the cross-superfile win). For filtered search each unit also
        // carries its per-superfile allow-set (a superfile reaching here
        // is guaranteed present in `allow` — empties were dropped above).
        let fallback: HashSet<usize> = fallback.into_iter().collect();
        // Look the allow-set up only for a superfile that is actually
        // selected (scored a kept cluster, or is a fallback) — a superfile
        // that survived vector pruning but whose predicate matched no row
        // is absent from `allow`, and must never be probed. Resolving the
        // bitmap eagerly for every entry would `expect`-panic on exactly
        // those filtered-out superfiles; gating it behind the selection
        // guards keeps the lookup on the path where presence is invariant.
        let mut units: Vec<(Arc<SuperfileEntry>, (Probe, Option<Arc<RoaringBitmap>>))> = Vec::new();
        for (si, entry) in superfiles.iter().enumerate() {
            let probe = if let Some(ids) = per_seg.remove(&si) {
                Probe::Clusters(ids)
            } else if fallback.contains(&si) {
                Probe::Nprobe
            } else {
                continue;
            };
            let bitmap = match allow.as_ref() {
                Some(m) => match m.get(&entry.uri) {
                    Some(bm) => Some(Arc::clone(bm)),
                    None => continue,
                },
                None => None,
            };
            units.push((Arc::clone(entry), (probe, bitmap)));
        }
        if units.is_empty() {
            return Ok(Vec::new());
        }

        // Fan out through the shared [`query::dispatch::fanout`] (also
        // used by FTS), but in waves capped by the configured reader
        // pool width. A cold vector kernel can hold large selected-cluster
        // `[codes][doc_ids]` prefix blocks while it builds its shortlist;
        // capping the number of concurrent superfiles keeps that transient
        // memory bounded by instance configuration instead of table size.
        // Skipped superfiles issue zero GETs.
        let column_arc = Arc::new(column.to_owned());
        let query_arc = Arc::new(query.to_vec());
        let kernel =
            move |reader: Arc<SuperfileReader>,
                  (probe, bitmap): (Probe, Option<Arc<RoaringBitmap>>)| {
                let column = Arc::clone(&column_arc);
                let query = Arc::clone(&query_arc);
                async move {
                    let res = match probe {
                        Probe::Clusters(ids) => {
                            reader
                                .vector_search_clusters_filtered(
                                    &column, &query, k, &ids, options, bitmap,
                                )
                                .await
                        }
                        Probe::Nprobe => {
                            reader
                                .vector_hits_filtered_async(&column, &query, k, options, bitmap)
                                .await
                        }
                    };
                    res.map_err(|e| QueryError::Parquet(e.to_string()))
                }
            };
        // Filtered search holds a per-superfile RoaringBitmap while the
        // kernel builds its shortlist; wave-cap the fan-out by reader-pool
        // width so transient memory stays bounded. The unfiltered path
        // carries no bitmaps and fans out all units at once (matching
        // main's concurrency — every superfile GET overlaps on tokio).
        let per_superfile = if allow.is_some() {
            let fanout_width = manifest.options.reader_pool.current_num_threads().max(1);
            let mut collected = Vec::new();
            while !units.is_empty() {
                let n = fanout_width.min(units.len());
                let wave: Vec<_> = units.drain(..n).collect();
                collected.extend(dispatch::fanout(self, wave, kernel.clone()).await?);
            }
            collected
        } else {
            dispatch::fanout(self, units, kernel).await?
        };

        Ok(top_k_ascending(per_superfile, k))
    }

    /// Filtered single-column vector kNN: the k-nearest rows **among
    /// those matching a text predicate**, by pushdown.
    ///
    /// The predicate is `filter_col` contains `filter_query`'s tokens
    /// under `mode` (the same unranked token match as
    /// [`Self::token_match_async`]). It is resolved per superfile into an
    /// allow-set of `local_doc_id`s, and each superfile's vector kernel
    /// ranks distance **only among its allowed doc-ids** — so the result
    /// is the true k-nearest among matching rows, with no over-fetch and
    /// no post-filter underflow. Superfiles whose predicate matches
    /// nothing are skipped (zero vector GETs).
    ///
    /// An empty `filter_query` (tokenizes to nothing) or a predicate
    /// that matches no row anywhere returns an empty `Vec`.
    ///
    /// `pub(crate)` async kernel — the public surface is the sync
    /// `vector_search` with a filter; this drives the cross-superfile fan-out.
    pub(crate) async fn vector_hits_filtered_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        filter: VectorFilter<'_>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let manifest = self.manifest();
        // Tokenize the predicate once with the index tokenizer (the same
        // tokenizer used at build time, so the terms match the postings AND
        // the manifest term blooms). No tokens (empty / punctuation-only) ⇒
        // nothing matches.
        let Some(tokenizer) = manifest.options.tokenizer.as_ref() else {
            return Ok(Vec::new());
        };
        let tokens: Vec<String> = tokenizer.tokenize(filter.query).collect();
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Manifest-only leaf survival: part-tier term bloom / range, then
        // per-superfile summaries — no superfile reads. Intersect with the
        // vector centroid prune so `token_match` opens only superfiles that
        // could match the predicate *and* might hold vector-near rows.
        let prune_leaves = [PruneLeaf::TermPresence {
            column: filter.column.to_owned(),
            terms: tokens.clone(),
            mode: filter.mode,
        }];
        let surviving: HashSet<u128> = select_superfiles(&manifest, &prune_leaves)
            .await?
            .iter()
            .map(|e| e.superfile_id.as_u128())
            .collect();
        if surviving.is_empty() {
            return Ok(Vec::new());
        }
        let superfiles = self
            .vector_pruned_superfiles_intersect(&manifest, column, query, &surviving)
            .await?;
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }

        // Resolve the exact per-superfile allow-set (`token_match` postings)
        // over the survivors; superfiles whose predicate matched no row are
        // dropped so they never fan out.
        let allow = self
            .candidate_bitmaps(&superfiles, filter.column, &tokens, filter.mode)
            .await?;
        if allow.is_empty() {
            return Ok(Vec::new());
        }

        self.vector_fanout_over_superfiles(superfiles, column, query, k, options, Some(allow))
            .await
    }

    /// Vector centroid prune intersected with a manifest-only survival set.
    async fn vector_pruned_superfiles_intersect(
        &self,
        manifest: &Manifest,
        column: &str,
        query: &[f32],
        surviving: &HashSet<u128>,
    ) -> Result<Vec<Arc<SuperfileEntry>>, QueryError> {
        Ok(manifest
            .get_pruned_superfiles_for_vector(column, query)
            .await
            .map_err(QueryError::ManifestLoad)?
            .into_iter()
            .filter(|e| surviving.contains(&e.superfile_id.as_u128()))
            .collect())
    }

    /// Resolve the text predicate (`filter_col` contains `tokens` under
    /// `mode`) to a per-superfile allow-set of matching `local_doc_id`s,
    /// over exactly the given vector-pruned `superfiles`.
    ///
    /// One `SuperfileReader::token_match` per superfile (postings-only,
    /// the leaf [`crate::supertable::query::candidate::CandidatePlan`]
    /// also uses), fanned out concurrently. Superfiles whose predicate
    /// matches no row are omitted from the returned map, so the caller
    /// skips them entirely.
    async fn candidate_bitmaps(
        &self,
        superfiles: &[Arc<SuperfileEntry>],
        filter_col: &str,
        tokens: &[String],
        mode: BoolMode,
    ) -> Result<HashMap<SuperfileUri, Arc<RoaringBitmap>>, QueryError> {
        let filter_col_arc = Arc::new(filter_col.to_owned());
        let tokens_arc: Arc<Vec<String>> = Arc::new(tokens.to_vec());
        self.fanout_candidate_bitmaps(superfiles, move |r, _entry| {
            let filter_col_arc = Arc::clone(&filter_col_arc);
            let tokens_arc = Arc::clone(&tokens_arc);
            async move {
                let refs: Vec<&str> = tokens_arc.iter().map(String::as_str).collect();
                r.token_match(&filter_col_arc, &refs, mode)
                    .await
                    .map_err(|e| QueryError::Parquet(e.to_string()))
                    .map(|docs| docs.into_iter().collect::<RoaringBitmap>())
            }
        })
        .await
    }

    /// Filtered vector kNN driven by a SQL `WHERE` [`CandidatePlan`] — the
    /// pushdown path for the `vector_search` table-valued function — rather
    /// than the single text-predicate shape of
    /// [`Self::vector_hits_filtered_async`].
    ///
    /// `plan` must be a **bounded** plan (not [`CandidatePlan::Unbounded`]):
    /// the caller routes `Unbounded` to the unfiltered
    /// [`Self::vector_search_async`], where DataFusion's `FilterExec`
    /// re-applies the predicate. For a bounded plan, each superfile's vector
    /// kernel ranks distance only among the `local_doc_id`s the plan admits,
    /// so the result is the true k-nearest among matching rows.
    ///
    /// Manifest-only leaf survival runs before any superfile opens: bounded
    /// FTS leaves are lowered to term-bloom prunes and intersected with the
    /// vector centroid prune. The per-superfile allow-set (`plan.evaluate`)
    /// then runs only over that intersection.
    pub(crate) async fn vector_hits_filtered_by_plan(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        plan: &CandidatePlan,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let manifest = self.manifest();
        let superfiles = match plan.surviving_superfile_ids(&manifest).await? {
            None => manifest
                .get_pruned_superfiles_for_vector(column, query)
                .await
                .map_err(QueryError::ManifestLoad)?,
            Some(surviving) if surviving.is_empty() => return Ok(Vec::new()),
            Some(surviving) => self
                .vector_pruned_superfiles_intersect(&manifest, column, query, &surviving)
                .await?,
        };
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }
        let allow = self.candidate_bitmaps_from_plan(&superfiles, plan).await?;
        if allow.is_empty() {
            return Ok(Vec::new());
        }
        self.vector_fanout_over_superfiles(superfiles, column, query, k, options, Some(allow))
            .await
    }

    /// Test/bench-only bitmap-filtered vector kNN. `allow_global` uses the
    /// same global row numbering as the bench corpus and is translated to
    /// per-superfile `local_doc_id` bitmaps before entering the normal filtered
    /// fan-out. This lets the supertable bench mirror the superfile filtered
    /// recall probe without requiring an FTS predicate on the vector-only
    /// fixture.
    #[cfg(feature = "test-helpers")]
    pub async fn vector_hits_global_allow_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        allow_global: Arc<RoaringBitmap>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 || allow_global.is_empty() {
            return Ok(Vec::new());
        }
        let manifest = self.manifest();
        let superfiles = manifest
            .get_pruned_superfiles_for_vector(column, query)
            .await
            .map_err(QueryError::ManifestLoad)?;
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }

        let mut allow_by_uri: HashMap<SuperfileUri, RoaringBitmap> = HashMap::new();
        let mut allowed = allow_global.iter().peekable();
        let mut base = 0u64;
        for entry in manifest.superfiles.iter() {
            let end = base.saturating_add(entry.n_docs);
            while allowed.peek().is_some_and(|&id| (id as u64) < base) {
                allowed.next();
            }
            let mut local = RoaringBitmap::new();
            while let Some(id) = allowed.peek().copied() {
                let id = id as u64;
                if id >= end {
                    break;
                }
                local.insert((id - base) as u32);
                allowed.next();
            }
            if !local.is_empty() {
                allow_by_uri.insert(entry.uri, local);
            }
            base = end;
        }

        if allow_by_uri.is_empty() {
            return Ok(Vec::new());
        }
        let allow = allow_by_uri
            .into_iter()
            .map(|(uri, bm)| (uri, Arc::new(bm)))
            .collect();
        self.vector_fanout_over_superfiles(superfiles, column, query, k, options, Some(allow))
            .await
    }

    /// Resolve a [`CandidatePlan`] to a per-superfile allow-set of matching
    /// `local_doc_id`s over the given vector-pruned `superfiles` — the
    /// boolean-plan analog of [`Self::candidate_bitmaps`] (which evaluates a
    /// single term match). `token_match` leaves are combined by `AND`/`OR`;
    /// superfiles whose plan matches no row are omitted so the caller skips
    /// them. Tombstoned rows are dropped by the shared `fanout` (a deleted
    /// row must never be a kNN candidate).
    ///
    /// The caller passes only a bounded plan, so `evaluate` returns
    /// `Some(bitmap)` per superfile; a defensive `None` (unbounded) is
    /// treated as the empty set, skipping that superfile.
    async fn candidate_bitmaps_from_plan(
        &self,
        superfiles: &[Arc<SuperfileEntry>],
        plan: &CandidatePlan,
    ) -> Result<HashMap<SuperfileUri, Arc<RoaringBitmap>>, QueryError> {
        let plan_arc = Arc::new(plan.clone());
        self.fanout_candidate_bitmaps(superfiles, move |r, _entry| {
            let plan = Arc::clone(&plan_arc);
            async move {
                plan.evaluate(r.as_ref())
                    .await
                    .map_err(|e| QueryError::Parquet(e.to_string()))?
                    .ok_or_else(|| {
                        QueryError::Execute(
                            "bounded CandidatePlan evaluated to Unbounded — planner bug".into(),
                        )
                    })
            }
        })
        .await
    }

    /// Fan out over `superfiles`, resolve matching `local_doc_id`s per
    /// superfile via `doc_ids`, subtract tombstones, and drop empties.
    async fn fanout_candidate_bitmaps<F, Fut>(
        &self,
        superfiles: &[Arc<SuperfileEntry>],
        doc_ids: F,
    ) -> Result<HashMap<SuperfileUri, Arc<RoaringBitmap>>, QueryError>
    where
        F: Fn(Arc<SuperfileReader>, Arc<SuperfileEntry>) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = Result<RoaringBitmap, QueryError>> + Send,
    {
        let units: Vec<(Arc<SuperfileEntry>, ())> =
            superfiles.iter().map(|e| (Arc::clone(e), ())).collect();
        let body = move |r: Arc<SuperfileReader>,
                         entry: Arc<SuperfileEntry>,
                         tombstone_cache: Option<Arc<SidecarCache>>,
                         now: Instant,
                         _: ()| {
            let doc_ids = doc_ids.clone();
            async move {
                let mut bm = doc_ids(r, Arc::clone(&entry)).await?;
                subtract_tombstones(&mut bm, &entry, tombstone_cache.as_deref(), now)?;
                Ok((entry.uri, bm))
            }
        };
        let pairs: Vec<(SuperfileUri, RoaringBitmap)> =
            dispatch::fanout_with(self, units, body).await?;
        Ok(pairs
            .into_iter()
            .filter(|(_, bm)| !bm.is_empty())
            .map(|(uri, bm)| (uri, Arc::new(bm)))
            .collect())
    }
    pub(crate) async fn vector_search_user_table_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let manifest = self.manifest();
        let superfiles = manifest
            .get_pruned_superfiles_for_vector(column, query)
            .await
            .map_err(QueryError::ManifestLoad)?;
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }
        self.fanout_vector_clusters(&superfiles, column, query, k, options)
            .await
    }

    /// Global-index vector kNN: hidden manifest + cell filter + fan-out at
    /// the hidden storage prefix. Falls back to the user table when the
    /// hidden index is absent or empty.
    pub(crate) async fn vector_search_global_index_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        if let Some(vit) = self.vector_index_table() {
            let vit_reader = vit.reader();
            let vit_manifest = vit_reader.manifest();
            let has_data = !vit_manifest.superfiles.is_empty() || vit_manifest.get_num_parts() > 0;
            if has_data {
                let vit_metric = vit_manifest
                    .options
                    .vector_columns
                    .iter()
                    .find(|vc| vc.column == column)
                    .map(|vc| vc.metric)
                    .unwrap_or(Metric::L2Sq);
                let selected = match vit_manifest.get_partition_strategy() {
                    PartitionStrategy::VectorCell {
                        clusters, routing, ..
                    } => {
                        let mut routed = clusters.select_cells_adaptive(
                            vit_metric,
                            query,
                            options.resolve(false).0,
                            routing_with_env_overrides(routing),
                        );
                        // Always scan the "incoming" append region in addition
                        // to the nprobe-routed cells: those rows have not been
                        // distributed into cells by maintenance yet, so routing
                        // by centroid can't see them.
                        routed.push(INCOMING_VECTOR_CELL);
                        if vit_manifest.superfiles.is_empty() {
                            vit_manifest
                                .superfiles_for_routed_cells(&routed)
                                .await
                                .map_err(|e| QueryError::Execute(e.to_string()))?
                        } else {
                            filter_superfiles_by_cells(&vit_manifest.superfiles, &routed)
                        }
                    }
                    _ => {
                        if vit_manifest.superfiles.is_empty() {
                            let part_ids: Vec<_> = vit_manifest
                                .get_all_list_entries()
                                .iter()
                                .map(|e| e.part_id)
                                .collect();
                            hierarchical_iter::load_and_flatten(vit_manifest, &part_ids)
                                .await
                                .map_err(|e| QueryError::Execute(e.to_string()))?
                        } else {
                            vit_manifest.superfiles.to_vec()
                        }
                    }
                };
                if !selected.is_empty() {
                    // Already top-k ascending (same as the user-table path
                    // below) — no second `top_k_ascending` pass needed.
                    return vit_reader
                        .fanout_vector_clusters(&selected, column, query, k, options)
                        .await;
                }
            }
        }
        self.vector_search_user_table_async(column, query, k, options)
            .await
    }

    /// Default async vector kernel — routes through the global hidden index
    /// when present (`vector_hits`, bare `vector_search` TVF).
    pub(crate) async fn vector_search_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.vector_search_global_index_async(column, query, k, options)
            .await
    }
}

impl SupertableReader {
    pub fn vector_search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        filter: Option<VectorFilter<'_>>,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, QueryError> {
        // Mark a foreground query in flight so background cache-fills yield
        // S3 bandwidth to it; released when this query returns.
        let _fg = crate::supertable::reader_cache::disk::ForegroundQueryGuard::enter();
        self.block_on(async {
            let hits = match filter {
                None => {
                    let hits = self
                        .vector_search_global_index_async(column, query, k, options)
                        .await?;
                    let on_user_table = hits.is_empty()
                        || hits_reference_user_superfiles(self, &hits).await?;
                    if projection.is_none() && !on_user_table {
                        let batch = hidden_hits_id_score_batch(self, &hits).await?;
                        return Ok(vec![batch]);
                    }
                    if on_user_table {
                        hits
                    } else {
                        remap_hidden_hits_to_user_hits(self, &hits).await?
                    }
                }
                Some(f) => {
                    self.vector_hits_filtered_async(column, query, k, options, f)
                        .await?
                }
            };
            let batch = resolve_hits_named(self, &hits, projection, "vector_search")
                .await
                .map_err(|e| QueryError::Execute(e.to_string()))?;
            Ok(vec![batch])
        })
    }

    pub fn vector_hits(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        filter: Option<VectorFilter<'_>>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        // Mark a foreground query in flight so background cache-fills yield
        // S3 bandwidth to it; released when this query returns.
        let _fg = crate::supertable::reader_cache::disk::ForegroundQueryGuard::enter();
        match filter {
            None => self.block_on(async {
                let hits = self
                    .vector_search_global_index_async(column, query, k, options)
                    .await?;
                if hits_reference_user_superfiles(self, &hits).await? {
                    Ok(hits)
                } else {
                    remap_hidden_hits_to_user_hits(self, &hits).await
                }
            }),
            Some(f) => self.block_on(self.vector_hits_filtered_async(column, query, k, options, f)),
        }
    }
}

fn subtract_tombstones(
    bm: &mut RoaringBitmap,
    entry: &SuperfileEntry,
    tombstone_cache: Option<&SidecarCache>,
    now: Instant,
) -> Result<(), QueryError> {
    if let Some(cache) = tombstone_cache {
        let deleted = cache
            .bitmap_for(entry.superfile_id, now)
            .map_err(|e| QueryError::Store(format!("tombstone cache: {e}")))?;
        if !deleted.is_empty() {
            *bm -= &*deleted;
        }
    }
    Ok(())
}

/// Merge per-superfile hits and return the top-k by *ascending*
/// distance (smallest = closest). Uses a max-heap of size k so
/// we never sort more than k elements — O(S·k·log k) instead of
/// O(S·k·log(S·k)) for the full-sort approach.
fn top_k_ascending(per_superfile: Vec<Vec<SuperfileHit>>, k: usize) -> Vec<SuperfileHit> {
    #[derive(PartialEq)]
    struct MaxByScore(SuperfileHit);
    impl Eq for MaxByScore {}
    impl PartialOrd for MaxByScore {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for MaxByScore {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0
                .score
                .partial_cmp(&other.0.score)
                .unwrap_or(Ordering::Equal)
        }
    }

    let mut heap = BinaryHeap::with_capacity(k + 1);
    for hit in per_superfile.into_iter().flatten() {
        if heap.len() < k {
            heap.push(MaxByScore(hit));
        } else if let Some(worst) = heap.peek()
            && hit.score < worst.0.score
        {
            heap.pop();
            heap.push(MaxByScore(hit));
        }
    }
    let mut result: Vec<SuperfileHit> = heap.into_iter().map(|m| m.0).collect();
    result.sort_unstable_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal));
    result
}

impl Supertable {
    /// Single-column vector kNN search over the current snapshot,
    /// returning Arrow rows nearest-first (distance score, smaller is
    /// nearer).
    ///
    /// Pins a fresh reader (applying the read-consistency policy), runs
    /// the IVF fan-out, and resolves the top-`k` nearest hits to Arrow
    /// rows.
    ///
    /// `projection` selects output columns by name (any of `_id`, the
    /// visible scalar columns, or the trailing `score`); `None` returns
    /// the engine-native result — `_id` + `score` only. Only the
    /// projected scalar columns are decoded — kNN is usually a
    /// retrieval step, so materializing row data is an explicit opt-in
    /// by column name for the hits you keep.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch};
    /// # use arrow_array::types::Float32Type;
    /// # use arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, IndexSpec, Metric, VectorSearchOptions};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new(
    /// #     "emb",
    /// #     DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 16),
    /// #     false,
    /// # )]));
    /// # let vecs = db.create_table("vecs", schema.clone(), IndexSpec::new().vector("emb", 16, 1, Metric::Cosine))?;
    /// # let mut data = vec![0.0f32; 16]; data[0] = 1.0;
    /// # let col = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vec![Some(data.iter().copied().map(Some).collect::<Vec<_>>())], 16);
    /// # vecs.append(&RecordBatch::try_new(schema, vec![Arc::new(col)])?)?;
    /// # let mut query = vec![0.0f32; 16]; query[0] = 1.0;
    /// // Bare call → `_id` + `score`, no scalar decode:
    /// let hits = vecs.vector_search("emb", &query, 10, VectorSearchOptions::new(), None, None)?;
    /// assert_eq!(hits[0].num_columns(), 2);
    /// // Explicit projection names the same columns (scalar columns,
    /// // when present, materialize row data):
    /// let rows = vecs.vector_search(
    ///     "emb",
    ///     &query,
    ///     10,
    ///     VectorSearchOptions::new(),
    ///     None,
    ///     Some(&["_id", "score"]),
    /// )?;
    /// assert!(rows.iter().map(|b| b.num_rows()).sum::<usize>() >= 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn vector_search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        filter: Option<VectorFilter<'_>>,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, crate::InfinoError> {
        self.reader()
            .vector_search(column, query, k, options, filter, projection)
            .map_err(crate::InfinoError::from)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc};

    use arrow::array::Array;
    use arrow_array::{FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    use super::VectorSearchOptions;
    use crate::{
        superfile::{
            builder::{FtsConfig, SuperfileBuilder, VectorConfig},
            vector::distance::Metric,
        },
        supertable::{Supertable, SupertableOptions, error::QueryError},
        test_helpers::default_tokenizer as tok,
    };

    /// Drive an async future to completion on a throwaway current-thread
    /// runtime. Used only for the single-superfile `SuperfileReader`
    /// oracle, whose search surface is async-only; the supertable
    /// reader's own search methods are sync and need no runtime here.
    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime")
            .block_on(fut)
    }

    fn fixed_list_f32(dim: usize) -> DataType {
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        )
    }

    /// Schema with id + title (FTS) + emb (vector). The supertable
    /// writer strips `emb` at commit time; vectors live in the
    /// embedded vector blob.
    fn schema_with_vector(dim: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", fixed_list_f32(dim), false),
        ]))
    }

    fn options_one_superfile_per_commit(dim: usize) -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        SupertableOptions::new(
            schema_with_vector(dim),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: crate::superfile::vector::rerank_codec::RerankCodec::Fp32,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// Construct a planted vector batch. Each doc gets a vector
    /// with one "active" component at dim `(global_id % dim)` set
    /// to 1.0 — keeps directions clearly separable so cosine
    /// distance from a query targeting a specific dim has only
    /// one cluster of close neighbors.
    fn build_vector_batch(start: u64, n: usize, dim: usize, schema: Arc<Schema>) -> RecordBatch {
        let titles = LargeStringArray::from((0..n).map(|i| format!("doc {i}")).collect::<Vec<_>>());
        let mut flat = Vec::<f32>::with_capacity(n * dim);
        for i in 0..n {
            let global = (start as usize) + i;
            for d in 0..dim {
                flat.push(if d == global % dim { 1.0 } else { 0.0 });
            }
        }
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Float32Array::from(flat);
        let fsl = FixedSizeListArray::try_new(
            item_field,
            dim as i32,
            Arc::new(values) as Arc<dyn Array>,
            None,
        )
        .expect("FSL");
        RecordBatch::try_new(schema, vec![Arc::new(titles), Arc::new(fsl)]).expect("batch")
    }

    /// Build a single-superfile oracle with the same `(id, title,
    /// emb)` rows. Note the separate `(scalar_batch, &[vector])`
    /// argument shape that `SuperfileBuilder::add_batch` takes —
    /// the supertable's writer wraps this for callers via
    /// `vector_split`, but for the oracle we plumb it manually.
    fn build_oracle_superfile(
        n_total: usize,
        dim: usize,
    ) -> Arc<crate::superfile::SuperfileReader> {
        // Oracle path goes through SuperfileBuilder directly,
        // so we mimic the supertable's effective schema by hand:
        // `_id` is `Decimal128(38, 0)`, ids are 0..n.
        let scalar_schema = Arc::new(Schema::new(vec![
            Field::new(
                "_id",
                DataType::Decimal128(
                    crate::supertable::options::DECIMAL128_PRECISION,
                    crate::supertable::options::DECIMAL128_SCALE,
                ),
                false,
            ),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = crate::superfile::builder::BuilderOptions::new(
            scalar_schema.clone(),
            "_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: crate::superfile::vector::rerank_codec::RerankCodec::Fp32,
            }],
            Some(tok()),
        );
        let mut b = SuperfileBuilder::new(opts).expect("builder");

        let ids = arrow_array::Decimal128Array::from((0..n_total as i128).collect::<Vec<_>>())
            .with_precision_and_scale(
                crate::supertable::options::DECIMAL128_PRECISION,
                crate::supertable::options::DECIMAL128_SCALE,
            )
            .expect("decimal128");
        let titles =
            LargeStringArray::from((0..n_total).map(|i| format!("doc {i}")).collect::<Vec<_>>());
        let scalar_batch =
            RecordBatch::try_new(scalar_schema, vec![Arc::new(ids), Arc::new(titles)])
                .expect("scalar batch");

        let mut flat = Vec::<f32>::with_capacity(n_total * dim);
        for i in 0..n_total {
            for d in 0..dim {
                flat.push(if d == i % dim { 1.0 } else { 0.0 });
            }
        }
        b.add_batch(&scalar_batch, &[flat.as_slice()])
            .expect("add_batch");
        let bytes = bytes::Bytes::from(b.finish().expect("finish"));
        Arc::new(crate::superfile::SuperfileReader::open(bytes).expect("open"))
    }

    #[test]
    fn vector_search_empty_supertable_returns_empty() {
        let st = Supertable::create(options_one_superfile_per_commit(16)).expect("create");
        let r = st.reader();
        let q = vec![0.1f32; 16];
        let hits = r
            .vector_hits("emb", &q, 5, VectorSearchOptions::new(), None)
            .expect("query");
        assert!(hits.is_empty());
    }

    #[test]
    fn vector_search_k_zero_short_circuits() {
        let st = Supertable::create(options_one_superfile_per_commit(16)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        w.append(&build_vector_batch(0, 8, 16, schema)).expect("a");
        w.commit().expect("c");
        let r = st.reader();
        let q = vec![0.1f32; 16];
        let hits = r
            .vector_hits("emb", &q, 0, VectorSearchOptions::new(), None)
            .expect("query");
        assert!(hits.is_empty());
    }

    #[test]
    fn vector_search_returns_ascending_distance_order() {
        let dim = 16;
        let st = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        w.append(&build_vector_batch(0, 8, dim, schema)).expect("a");
        w.commit().expect("c");
        let r = st.reader();
        // Query vector resembling row 0's pattern.
        let mut q = vec![0.0f32; dim];
        for (d, x) in q.iter_mut().enumerate() {
            *x = (d as f32) / 100.0 + 0.001;
        }
        let hits = r
            .vector_hits("emb", &q, 5, VectorSearchOptions::new(), None)
            .expect("query");
        assert!(!hits.is_empty());
        for w in hits.windows(2) {
            assert!(
                w[0].score <= w[1].score,
                "expected ascending: {:?} then {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn vector_search_top_k_caps_at_k() {
        let dim = 16;
        let st = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        // Three commits → three superfiles × 8 docs = 24 docs.
        for chunk in 0..3u64 {
            w.append(&build_vector_batch(chunk * 8, 8, dim, schema.clone()))
                .expect("a");
            w.commit().expect("c");
        }
        let r = st.reader();
        let q = vec![0.1f32; dim];
        let hits = r
            .vector_hits("emb", &q, 7, VectorSearchOptions::new(), None)
            .expect("query");
        assert_eq!(hits.len(), 7);
    }

    #[test]
    fn vector_search_global_selection_recovers_neighbors_under_low_budget() {
        // 10 superfiles × 16 one-hot docs. Query e_0's true neighbors are
        // the 10 docs with id % dim == 0 (one per superfile) at cosine
        // distance 0; every other doc is orthogonal (distance 1). With
        // nprobe = 1 the global budget is only 10 clusters across all 10
        // superfiles — so this exercises real cross-superfile cluster
        // pruning (most of the 10 × n_cent clusters are skipped), and
        // recall@10 must still recover the concentrated neighbors.
        let dim = 16;
        let st = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        let n_seg = 10u64;
        for chunk in 0..n_seg {
            w.append(&build_vector_batch(chunk * 16, 16, dim, schema.clone()))
                .expect("append");
            w.commit().expect("commit");
        }
        assert_eq!(st.reader().n_superfiles(), n_seg as usize);

        let mut q = vec![0f32; dim];
        q[0] = 1.0;
        let opts = VectorSearchOptions::new().with_nprobe(1);
        let hits = st
            .reader()
            .vector_hits("emb", &q, 10, opts, None)
            .expect("query");

        let exact_neighbors = hits.iter().filter(|h| h.score < 1e-3).count();
        assert!(
            exact_neighbors >= 9,
            "recall@10 ≥ 0.90 under aggressive global cluster pruning; \
             recovered {exact_neighbors}/10 exact neighbors"
        );
    }

    #[test]
    fn vector_search_carries_superfile_uris_for_multi_superfile_results() {
        let dim = 16;
        let st = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        for chunk in 0..3u64 {
            w.append(&build_vector_batch(chunk * 8, 8, dim, schema.clone()))
                .expect("a");
            w.commit().expect("c");
        }
        let r = st.reader();
        let q = vec![0.1f32; dim];
        let hits = r
            .vector_hits("emb", &q, 24, VectorSearchOptions::new(), None)
            .expect("query");
        let superfile_uris: HashSet<_> = hits.iter().map(|h| h.superfile).collect();
        // All three superfiles should contribute (high k pulls from
        // each).
        assert_eq!(superfile_uris.len(), 3);
    }

    #[test]
    fn vector_search_oracle_top_k_set_matches_single_superfile() {
        // Vector distances are superfile-independent — cosine /
        // L2-sq are functions of the query + per-doc vector only.
        // So the per-superfile-top-k → global-top-k pattern recovers
        // the same set as a single-superfile search, modulo each
        // IVF's nprobe-driven recall (we use a high-recall config).
        let dim = 16;
        let st = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        // 24 docs across 3 superfiles.
        for chunk in 0..3u64 {
            w.append(&build_vector_batch(chunk * 8, 8, dim, schema.clone()))
                .expect("a");
            w.commit().expect("c");
        }
        let oracle = build_oracle_superfile(24, dim);

        // High-recall config: full nprobe + plenty of rerank.
        let opts = VectorSearchOptions::new().with_nprobe(4);

        // Query targets dim 0 — closest neighbors are docs whose
        // global id is 0 mod dim (i.e. 0 and 16 in 24 docs at
        // dim=16). Other docs have orthogonal vectors and contribute
        // cosine distance = 1.0.
        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;

        // The oracle is a single-superfile `SuperfileReader` whose search
        // is async-only; drive it on a throwaway runtime. The supertable
        // reader below uses its sync public API.
        let oracle_hits =
            block_on(oracle.vector_hits_async("emb", &q, 2, opts)).expect("oracle query");
        let oracle_globals: HashSet<u32> = oracle_hits.iter().map(|(d, _)| *d).collect();
        assert_eq!(oracle_globals, [0u32, 16].iter().copied().collect());

        let st_reader = st.reader();
        let st_hits = st_reader
            .vector_hits("emb", &q, 2, opts, None)
            .expect("supertable query");
        let manifest = st_reader.manifest();
        let st_globals: HashSet<u32> = st_hits
            .iter()
            .map(|h| {
                let seg_idx = manifest
                    .superfiles
                    .iter()
                    .position(|e| e.uri == h.superfile)
                    .expect("superfile in manifest");
                (seg_idx as u32) * 8 + h.local_doc_id
            })
            .collect();
        assert_eq!(st_hits.len(), oracle_hits.len());
        assert_eq!(st_globals, oracle_globals);
    }

    #[test]
    fn vector_search_unknown_column_errors() {
        let dim = 16;
        let st = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        w.append(&build_vector_batch(0, 8, dim, schema)).expect("a");
        w.commit().expect("c");
        let r = st.reader();
        let q = vec![0.1f32; dim];
        let err = r
            .vector_hits("nope", &q, 5, VectorSearchOptions::new(), None)
            .expect_err("expected error");
        assert!(matches!(err, QueryError::Parquet(_)), "got {err:?}");
    }

    // ---- Tombstone filter helper: direct-call coverage --------------
    //
    // Exercises `apply_tombstone_filter` against a synthesized
    // bitmap + hit list without going through the full IVF +
    // lazy-source vector search path. The hook logic is identical
    // to the FTS path (both drop hits whose `local_doc_id` is in
    // the per-superfile bitmap); this direct test pins the
    // contract for the vector side.

    use tempfile::TempDir;
    use uuid::Uuid;

    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{
            manifest::{SuperfileEntry, SuperfileUri},
            query::SuperfileHit,
            tombstones::{SidecarCache, cache::DEFAULT_REFRESH_TTL},
            wal::{WalStore, tombstones_codec::TombstonesSidecar},
        },
    };

    fn synthetic_entry(superfile_id: Uuid) -> SuperfileEntry {
        SuperfileEntry {
            superfile_id,
            uri: SuperfileUri(superfile_id),
            n_docs: 100,
            id_min: 0,
            id_max: 99,
            scalar_stats: std::collections::HashMap::new(),
            fts_summary: std::collections::HashMap::new(),
            vector_summary: std::collections::HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            vector_layout: crate::superfile::vector::layout::VectorLayout::Ivf,
            subsection_offsets: None,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_tombstone_filter_drops_set_bits() {
        // Build a SidecarCache backed by a real (LocalFs) storage so
        // the hook exercises the same cache machinery that the
        // production query path uses.
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let ws = WalStore::new(Arc::clone(&storage));
        let cache = Arc::new(SidecarCache::new(ws.clone(), DEFAULT_REFRESH_TTL));

        let sf_id = Uuid::from_u128(0xFEEDFACE);
        // Pre-populate a sidecar with doc-ids 1, 3, 5 set.
        let mut bitmap = roaring::RoaringBitmap::new();
        bitmap.insert(1);
        bitmap.insert(3);
        bitmap.insert(5);
        ws.put_tombstones(sf_id, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("put sidecar");

        let entry = synthetic_entry(sf_id);
        let mut hits: Vec<SuperfileHit> = (0..8u32)
            .map(|d| SuperfileHit {
                superfile: entry.uri,
                local_doc_id: d,
                score: d as f32,
                stable_id: None,
            })
            .collect();

        crate::supertable::query::dispatch::apply_tombstone_filter(
            Some(&cache),
            &entry,
            &mut hits,
            std::time::Instant::now(),
        )
        .expect("filter");

        let remaining: Vec<u32> = hits.iter().map(|h| h.local_doc_id).collect();
        assert_eq!(remaining, vec![0u32, 2, 4, 6, 7]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_tombstone_filter_is_no_op_without_cache() {
        let entry = synthetic_entry(Uuid::from_u128(0xABCD));
        let mut hits: Vec<SuperfileHit> = (0..4u32)
            .map(|d| SuperfileHit {
                superfile: entry.uri,
                local_doc_id: d,
                score: 0.0,
                stable_id: None,
            })
            .collect();
        let original = hits.clone();
        crate::supertable::query::dispatch::apply_tombstone_filter(
            None,
            &entry,
            &mut hits,
            std::time::Instant::now(),
        )
        .expect("no-cache");
        assert_eq!(hits, original);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_tombstone_filter_short_circuits_on_empty_bitmap() {
        // No sidecar at all → cache populates the "known 404"
        // sentinel and `bitmap.is_empty()` short-circuits the
        // filter loop. Hit list is unchanged.
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let ws = WalStore::new(Arc::clone(&storage));
        let cache = Arc::new(SidecarCache::new(ws, DEFAULT_REFRESH_TTL));

        let entry = synthetic_entry(Uuid::from_u128(0x1111));
        let mut hits: Vec<SuperfileHit> = (0..4u32)
            .map(|d| SuperfileHit {
                superfile: entry.uri,
                local_doc_id: d,
                score: 0.0,
                stable_id: None,
            })
            .collect();
        let original = hits.clone();
        crate::supertable::query::dispatch::apply_tombstone_filter(
            Some(&cache),
            &entry,
            &mut hits,
            std::time::Instant::now(),
        )
        .expect("filter");
        assert_eq!(hits, original);
    }
    #[test]
    fn hybrid_vector_leg_uses_user_superfiles_not_hidden() {
        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let opts = options_one_superfile_per_commit(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let opts = opts.with_storage(storage);
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 32, dim, schema.clone()))
            .expect("append");
        w.commit().expect("commit");

        let reader = st.reader();
        let user_uris: HashSet<_> = reader.manifest().superfiles.iter().map(|e| e.uri).collect();
        assert!(
            reader.vector_index_table().is_some(),
            "hidden index must exist"
        );

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let hits = reader
            .hybrid_search(
                "title",
                "doc",
                crate::superfile::fts::reader::BoolMode::Or,
                "emb",
                &q,
                VectorSearchOptions::new(),
                5,
            )
            .expect("hybrid");
        assert!(!hits.is_empty());
        for hit in &hits {
            assert!(
                user_uris.contains(&hit.superfile),
                "hybrid vector leg must fan out on user superfiles, got {:?}",
                hit.superfile
            );
        }
    }

    #[test]
    fn vector_search_row_return_resolves_through_hidden_index() {
        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let opts = options_one_superfile_per_commit(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let opts = opts.with_storage(storage);
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 16, dim, schema.clone()))
            .expect("append");
        w.commit().expect("commit");

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let batches = st
            .reader()
            .vector_search(
                "emb",
                &q,
                5,
                VectorSearchOptions::new(),
                None,
                Some(&["_id", "score"]),
            )
            .expect("vector_search rows");
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert!(
            rows >= 1,
            "row-returning vector_search must resolve user rows"
        );
    }
}
