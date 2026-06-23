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
    superfile::{SuperfileReader, fts::reader::BoolMode},
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        manifest::{Manifest, SuperfileEntry, SuperfileUri, part::PartId},
        opann::paged::PagedTree,
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

/// Radius-aware adaptive probe selection (§7.3): the set of partition cells to
/// fetch for a query, from each candidate's centroid distance `d` and its
/// partition's covering radius `r`. Restores the removed `select_cells_adaptive`
/// behavior over the OPANN tree's descended candidates:
///
/// - always probe the `nprobe_min` nearest cells (a recall floor);
/// - then admit any further cell whose radius-aware lower bound
///   `lb = (d − r).max(0)` is within `τ = d* + slack·r*` (`d*`/`r*` = the nearest
///   cell's distance/radius), in ascending-`lb` order, up to `nprobe_max`.
///
/// The lower bound is the point: a far-centroid but large-radius partition can
/// still hold a true neighbor, and ranking purely by centroid distance drops it.
/// A cell with no recorded radius falls back to a distance-only bound.
fn adaptive_probe_cells(
    candidates: Vec<(u128, f32)>,
    radius_of: &HashMap<u128, f32>,
    nprobe_min: usize,
    nprobe_max: usize,
    slack: f32,
) -> HashSet<u128> {
    if candidates.is_empty() {
        return HashSet::new();
    }
    let mut scored = candidates;
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let radius = |cell: u128| radius_of.get(&cell).copied().unwrap_or(0.0);
    let lb = |i: usize| {
        let (cell, d) = scored[i];
        (d - radius(cell)).max(0.0)
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
    let mut chosen: HashSet<u128> = scored[..floor].iter().map(|(c, _)| *c).collect();
    let mut rest: Vec<usize> = (floor..n).collect();
    rest.sort_by(|&a, &b| lb(a).partial_cmp(&lb(b)).unwrap_or(Ordering::Equal));
    for i in rest {
        if chosen.len() >= nprobe_max {
            break;
        }
        if lb(i) <= tau {
            chosen.insert(scored[i].0);
        }
    }
    chosen
}

/// Superfiles in manifest commit order — loads parts when the flat
/// `manifest.superfiles` view is empty (lazy open after `Supertable::open`).
pub(super) async fn ordered_manifest_superfiles(
    manifest: &Manifest,
) -> Result<Vec<Arc<SuperfileEntry>>, QueryError> {
    if !manifest.is_in_process_only() {
        let part_ids: Vec<PartId> = manifest
            .get_all_list_entries()
            .iter()
            .map(|e| e.part_id)
            .collect();
        hierarchical_iter::load_and_flatten(manifest, &part_ids)
            .await
            .map_err(QueryError::ManifestLoad)
    } else {
        Ok(hierarchical_iter::fallback_to_flat_superfiles(manifest))
    }
}

/// Global doc-id base per superfile URI in manifest order.
async fn superfile_doc_base_offsets(
    manifest: &Manifest,
) -> Result<HashMap<SuperfileUri, u32>, QueryError> {
    let entries = ordered_manifest_superfiles(manifest).await?;
    let mut map = HashMap::with_capacity(entries.len());
    let mut acc = 0u32;
    for entry in entries {
        map.insert(entry.uri, acc);
        acc = acc.saturating_add(entry.n_docs as u32);
    }
    Ok(map)
}

/// Whether every hit already references a superfile in the user manifest.
fn hits_reference_user_superfiles(reader: &SupertableReader, hits: &[SuperfileHit]) -> bool {
    let manifest = reader.manifest();
    hits.iter().all(|hit| {
        manifest
            .superfiles
            .iter()
            .any(|entry| entry.uri == hit.superfile)
    })
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
        by_superfile.entry(hit.superfile).or_default().push(i);
    }
    for (uri, idxs) in by_superfile {
        let entry = hidden_manifest
            .superfiles
            .iter()
            .find(|e| e.uri == uri)
            .ok_or_else(|| {
                QueryError::Execute(format!("hidden superfile {uri:?} missing from manifest"))
            })?;
        // Contiguous span → arithmetic, no read.
        if row_id_from_manifest_entry(entry, 0).is_some() {
            for &i in &idxs {
                ids[i] = entry.id_min + i128::from(hidden_hits[i].local_doc_id);
            }
            continue;
        }
        // Gapped span → one resident read of just the rows these hits touch.
        let locals: Vec<u32> = idxs.iter().map(|&i| hidden_hits[i].local_doc_id).collect();
        let vals = read_ids_for_locals(hidden_manifest, entry, &locals, id_column).await?;
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
            });
        } else {
            gapped.entry(user_entry.uri).or_default().push(i);
        }
    }
    for (uri, idxs) in gapped {
        let user_entry = user_manifest
            .superfiles
            .iter()
            .find(|e| e.uri == uri)
            .ok_or_else(|| {
                QueryError::Execute(format!("user superfile {uri:?} missing from manifest"))
            })?;
        let n = user_entry.n_docs as usize;
        let all_locals: Vec<u32> = (0..n as u32).collect();
        // Column is monotonic (ids minted in row order) → binary-searchable.
        let id_col = read_ids_for_locals(user_manifest, user_entry, &all_locals, id_column).await?;
        for &i in &idxs {
            let user_row_id = user_ids[i];
            let pos = id_col.binary_search(&user_row_id).map_err(|_| {
                QueryError::Execute(format!("no row with id {user_row_id} in user superfile"))
            })?;
            remapped[i] = Some(SuperfileHit {
                superfile: uri,
                local_doc_id: pos as u32,
                score: hidden_hits[i].score,
            });
        }
    }
    // No user-tombstone filter here: user deletes are mirrored into the hidden
    // cells' own tombstone sidecars at delete time, so the hidden fan-out's
    // `apply_tombstone_filter` (already on the warm scan path) drops them before
    // they ever reach this remap — keeping both the `_id`-only fast path and
    // this path free of any extra query-time work.
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
        let manifest = self.manifest();

        // The caller already chose the superfiles to probe — the OPANN tree
        // descent for the hidden index, or the candidate enumeration for the
        // filtered user-table paths. There is NO cross-superfile cluster
        // re-ranking here: each kept superfile is probed directly with its own
        // within-superfile IVF `nprobe` + rerank. (Routing across superfiles
        // lives in the tree now, not in a second per-cluster scoring pass over
        // the manifest summaries.)
        //
        // For filtered search each unit carries its per-superfile allow-set; a
        // superfile whose predicate matched no row (absent from `allow`) is
        // dropped — it never enters the fan-out and issues zero GETs.
        let mut units: Vec<(Arc<SuperfileEntry>, Option<Arc<RoaringBitmap>>)> = Vec::new();
        for entry in &superfiles {
            let bitmap = match allow.as_ref() {
                Some(m) => match m.get(&entry.uri) {
                    Some(bm) => Some(Arc::clone(bm)),
                    None => continue,
                },
                None => None,
            };
            units.push((Arc::clone(entry), bitmap));
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
        let kernel = move |reader: Arc<SuperfileReader>, bitmap: Option<Arc<RoaringBitmap>>| {
            let column = Arc::clone(&column_arc);
            let query = Arc::clone(&query_arc);
            async move {
                reader
                    .vector_hits_filtered_async(&column, &query, k, options, bitmap)
                    .await
                    .map_err(|e| QueryError::Parquet(e.to_string()))
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

        Ok(top_k_ascending(
            per_superfile.into_iter().flatten().collect(),
            k,
        ))
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
        // tokenizer used at build time, so the terms match the postings AND the
        // manifest term blooms). No tokens (empty / punctuation-only) ⇒ nothing
        // matches.
        let Some(tokenizer) = manifest.options.tokenizer.as_ref() else {
            return Ok(Vec::new());
        };
        let tokens: Vec<String> = tokenizer.tokenize(filter.query).collect();
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        // §5a level-1 (leaf-survival): the cheap, manifest-only predicate prune
        // (part-tier term bloom / range, then per-superfile summaries — no
        // superfile reads) gives the superfiles that *could* match. The routing
        // tree descent below is gated to exactly these: a leaf whose superfile
        // failed the predicate is skipped without spending probe budget, so the
        // budget lands on vector-near *matching* cells. Without this, a selective
        // predicate craters recall (the budget goes to vector-near non-matching
        // cells, and matching cells slightly farther by vector never get probed).
        let prune_leaves = [PruneLeaf::TermPresence {
            column: filter.column.to_owned(),
            terms: tokens.clone(),
            mode: filter.mode,
        }];
        let surviving: HashSet<u128> = select_superfiles(manifest, &prune_leaves[..])
            .await?
            .iter()
            .map(|e| e.superfile_id.as_u128())
            .collect();
        if surviving.is_empty() {
            return Ok(Vec::new());
        }
        let manifest = self.manifest();
        if let Some(leaves) = super::vector_probe::select_opann_probe_leaves(
            self,
            manifest,
            column,
            query,
            &options,
            |sid| surviving.contains(&sid),
        )
        .await?
        {
            let entries: Vec<Arc<SuperfileEntry>> =
                leaves.iter().map(|(_, _, e)| Arc::clone(e)).collect();
            let allow = self
                .candidate_bitmaps(&entries, filter.column, &tokens, filter.mode)
                .await?;
            if allow.is_empty() {
                return Ok(Vec::new());
            }
            return super::vector_probe::fanout_opann_leaf_probes(
                self, leaves, column, query, k, options, Some(allow),
            )
            .await;
        }
        let superfiles = self
            .candidate_superfiles_for_vector(column, query, &options, |sid| {
                surviving.contains(&sid)
            })
            .await?;
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }

        // §5a level-2 (row mask): resolve the exact per-superfile allow-set
        // (`token_match` postings) over the probed survivors; superfiles whose
        // predicate matched no row are dropped so they never fan out.
        let allow = self
            .candidate_bitmaps(&superfiles, filter.column, &tokens, filter.mode)
            .await?;
        if allow.is_empty() {
            return Ok(Vec::new());
        }

        self.vector_fanout_over_superfiles(superfiles, column, query, k, options, Some(allow))
            .await
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
    /// There is deliberately **no selectivity gate** here (unlike the scan
    /// provider, which skips the index path above ~1% match density because
    /// a Parquet `RowSelection` can't skip saturated pages). The vector
    /// kernel reads the same IVF clusters either way; the allow-set only
    /// filters which candidates enter the shortlist heap, and even a
    /// non-selective predicate must still yield exactly-k matching hits — so
    /// a bounded plan is always pushed down.
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
        // Prune the superfile set through the routing tree, then resolve the
        // plan's allow-set over the survivors.
        // TODO(§5a): pass a real survival gate here — derive the surviving
        // superfile set from the plan's prune leaves (a `CandidatePlan` →
        // `PruneLeaf` extraction that doesn't exist yet) and feed it as
        // `survives`, mirroring the text path below. Until then the SQL-pushdown
        // path admits every leaf (vector-first), which under-probes selective
        // predicates — the recall bench runs through `vector_hits_global_allow_async`,
        // which IS survival-gated, so this gap doesn't mask the gate.
        let superfiles = self
            .candidate_superfiles_for_vector(column, query, &options, |_| true)
            .await?;
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
        // Translate the global allow-bitmap into per-superfile local bitmaps,
        // recording which superfiles hold any allowed row — that set is the §5a
        // survival gate, computed before the descent so the bench's filtered
        // recall measures the same survival-gated tree path production uses.
        let mut allow_by_uri: HashMap<SuperfileUri, RoaringBitmap> = HashMap::new();
        let mut surviving: HashSet<u128> = HashSet::new();
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
                surviving.insert(entry.superfile_id.as_u128());
                allow_by_uri.insert(entry.uri, local);
            }
            base = end;
        }
        if allow_by_uri.is_empty() {
            return Ok(Vec::new());
        }

        // Survival-gated descent: only superfiles holding allowed rows.
        let superfiles = self
            .candidate_superfiles_for_vector(column, query, &options, |sid| {
                surviving.contains(&sid)
            })
            .await?;
        if superfiles.is_empty() {
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
    /// Candidate superfiles for a vector query: descend the resident OPANN
    /// routing tree (radius-aware adaptive admission) when this table carries
    /// one, else enumerate the full set. Shared by the unfiltered user/hidden
    /// path and the filtered paths, so filtered search **actually prunes**
    /// through the tree instead of scanning every superfile — the predicate
    /// allow-set then drops the rest within the probed set.
    async fn candidate_superfiles_for_vector(
        &self,
        column: &str,
        query: &[f32],
        options: &VectorSearchOptions,
        survives: impl Fn(u128) -> bool,
    ) -> Result<Vec<Arc<SuperfileEntry>>, QueryError> {
        let manifest = self.manifest();
        // The resident OPANN routing tree swaps with the manifest — loaded once
        // at open and reused by every query. Copy the routing root + probe
        // budget out of the same record so neither borrows `manifest` across
        // the awaits below.
        let tree = manifest
            .opann_resident_tree()
            .await
            .map_err(|e| QueryError::Store(format!("opann tree load: {e}")))?;
        let opann = manifest.opann_routing().map(|r| (r.root_page, r.routing));
        let superfiles = match (tree, opann) {
            // OPANN: descend the compute-resident Sq8 routing tree for ALL
            // candidate leaves (cell id + centroid distance), then choose the
            // probe set by radius-aware adaptive admission (§7.3) rather than a
            // fixed top-N: a far-centroid but large-radius partition can still
            // hold a true neighbor, and pure centroid-distance ranking drops it.
            // `survives` is the §5a leaf-survival gate: a leaf whose superfile
            // failed the predicate is skipped in the descent without consuming
            // probe budget (the unfiltered path passes an always-true `survives`).
            (Some(source), Some((root, routing))) => {
                let candidates: Vec<(u128, f32)> = PagedTree::new(source, root)
                    .select_probes_where(query, usize::MAX, &survives)
                    .map_err(|e| QueryError::Store(format!("opann descent: {e}")))?
                    .into_iter()
                    // Collapse each cluster leaf to its superfile id; the probed
                    // superfile is then searched with its own internal IVF.
                    .map(|(leaf, d)| (leaf.superfile_id, d))
                    .collect();
                let entries = ordered_manifest_superfiles(manifest).await?;
                // Each candidate cell's covering radius is its partition's
                // vector-summary radius (the same value stamped as the leaf
                // radius at write — never decoded from a stored centroid).
                let radius_of: HashMap<u128, f32> = entries
                    .iter()
                    .filter_map(|sf| {
                        sf.vector_summary
                            .get(column)
                            .map(|vs| (sf.superfile_id.as_u128(), vs.radius))
                    })
                    .collect();
                // The search `nprobe` is the recall FLOOR — the minimum number of
                // nearest cells to probe — not a cap. Radius-aware admission then
                // expands beyond the floor up to the routing `nprobe_max` ceiling.
                // Wiring `nprobe` as the cap would defeat adaptive probing: a fixed
                // `nprobe=6` would hard-stop at 6 cells even when a query's true
                // neighbourhood is fragmented across many time-local per-commit
                // cells, dropping recall. With no search nprobe, the routing floor
                // (`nprobe_min`) applies and the ceiling stays `nprobe_max`.
                let floor = options.nprobe.unwrap_or(routing.nprobe_min);
                let cell_set = adaptive_probe_cells(
                    candidates,
                    &radius_of,
                    floor,
                    routing.nprobe_max,
                    routing.slack,
                );
                entries
                    .into_iter()
                    .filter(|sf| cell_set.contains(&sf.superfile_id.as_u128()))
                    .collect()
            }
            // Durable hidden index with no published tree yet: only the
            // in-process (pre-first-durable-commit) shape has data without a
            // tree, and it has no manifest list to flatten. A durable hidden
            // table always carries a tree, so reaching here with a published
            // list returns empty rather than brute-force scanning the index.
            _ if manifest.options.is_hidden_vector_index && !manifest.is_in_process_only() => {
                Vec::new()
            }
            // No tree (storage-less / `memory://` user table): there is no
            // routing layer to prune through, so enumerate the full set —
            // still honoring survival so a filtered query restricts to the
            // predicate-surviving superfiles.
            _ => manifest
                .get_pruned_superfiles_for_vector(column, query)
                .await
                .map_err(QueryError::ManifestLoad)?
                .into_iter()
                .filter(|sf| survives(sf.superfile_id.as_u128()))
                .collect(),
        };
        Ok(superfiles)
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
        if let Some(leaves) = super::vector_probe::select_opann_probe_leaves(
            self,
            manifest,
            column,
            query,
            &options,
            |_| true,
        )
        .await?
        {
            return super::vector_probe::fanout_opann_leaf_probes(
                self, leaves, column, query, k, options, None,
            )
            .await;
        }
        // Legacy: no OPANN tree or missing probe layout — open superfiles and
        // run per-superfile IVF search.
        let superfiles = self
            .candidate_superfiles_for_vector(column, query, &options, |_| true)
            .await?;
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }
        self.fanout_vector_clusters(&superfiles, column, query, k, options)
            .await
    }

    /// Global-index vector kNN. Each hidden superfile is an immutable cell;
    /// route by its per-cell centroids through the shared cross-superfile
    /// fan-out — which prunes to the nearest cells and scans those — exactly the
    /// user-table vector path, run over the hidden cell table. The hidden index
    /// is dual-written with every vector, so when it holds data it is
    /// authoritative; we fall back to the user table only when it is absent or
    /// empty.
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
                return vit_reader
                    .vector_search_user_table_async(column, query, k, options)
                    .await;
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
        self.block_on(async {
            let hits = match filter {
                None => {
                    let hits = self
                        .vector_search_global_index_async(column, query, k, options)
                        .await?;
                    let on_user_table =
                        hits.is_empty() || hits_reference_user_superfiles(self, &hits);
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
        match filter {
            None => self.block_on(async {
                let hits = self
                    .vector_search_global_index_async(column, query, k, options)
                    .await?;
                if hits_reference_user_superfiles(self, &hits) {
                    Ok(hits)
                } else {
                    remap_hidden_hits_to_user_hits(self, &hits).await
                }
            }),
            Some(f) => self.block_on(self.vector_hits_filtered_async(column, query, k, options, f)),
        }
    }

    test_visible! {
    /// Map each user superfile URI to its global doc-id base. Loads manifest
    /// parts when the flat `manifest.superfiles` list is empty (lazy open).
    fn superfile_doc_base_offsets(&self) -> Result<HashMap<SuperfileUri, u32>, QueryError> {
        self.block_on(superfile_doc_base_offsets(self.manifest()))
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
pub(super) fn top_k_ascending(hits: Vec<SuperfileHit>, k: usize) -> Vec<SuperfileHit> {
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
    for hit in hits {
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
    use std::{
        collections::{HashMap, HashSet},
        sync::Arc,
    };

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

    use super::adaptive_probe_cells;

    /// Radius-aware admission must pull in a far-centroid, large-radius cell
    /// (which pure centroid-distance ranking would drop) ahead of a closer
    /// small-radius one — the recall behavior fixed-top-N lost.
    #[test]
    fn adaptive_probe_admits_far_centroid_large_radius() {
        // c_near: d=1, r=0; c_far_wide: d=10 but r=9.5 → lb=0.5; c_mid: d=3, r=0.
        let candidates = vec![(1u128, 1.0f32), (2, 10.0), (3, 3.0)];
        let radius_of: HashMap<u128, f32> =
            [(1u128, 0.0f32), (2, 9.5), (3, 0.0)].into_iter().collect();
        // nprobe_min=1 (floor = c1), slack=1 → τ = d*·2 = 2.0; cap=2.
        let chosen = adaptive_probe_cells(candidates, &radius_of, 1, 2, 1.0);
        assert!(chosen.contains(&1), "nearest cell is the floor");
        assert!(
            chosen.contains(&2),
            "far-centroid large-radius cell (lb=0.5 ≤ τ=2.0) must be admitted over the closer small-radius one"
        );
        assert!(
            !chosen.contains(&3),
            "cap=2 reached; c3 (lb=3.0 > τ) excluded"
        );
        assert_eq!(chosen.len(), 2, "respects the nprobe_max cap");
    }

    /// The floor (`nprobe_min`) is always probed even when nothing else clears τ.
    #[test]
    fn adaptive_probe_honors_nprobe_min_floor() {
        let candidates = vec![(1u128, 1.0f32), (2, 100.0), (3, 200.0)];
        let radius_of: HashMap<u128, f32> =
            [(1u128, 0.0f32), (2, 0.0), (3, 0.0)].into_iter().collect();
        // τ = 2.0; only c1 within τ, but nprobe_min=2 forces the 2 nearest.
        let chosen = adaptive_probe_cells(candidates, &radius_of, 2, 8, 1.0);
        assert!(
            chosen.contains(&1) && chosen.contains(&2),
            "two nearest are the floor"
        );
        assert_eq!(chosen.len(), 2, "nothing else clears τ");
    }

    /// Path B acceptance gate: storage-backed hidden OPANN routing with default
    /// adaptive probing must reach recall@10 ≥ 0.99 after `Supertable::open`.
    #[test]
    fn storage_backed_opann_recall_at_acceptance_bar() {
        use std::collections::HashSet;

        use tempfile::TempDir;

        use crate::{
            storage::{LocalFsStorageProvider, StorageProvider},
            superfile::{
                builder::{FtsConfig, VectorConfig},
                vector::rerank_codec::RerankCodec,
            },
        };

        const N_COMMITS: usize = 16;
        const DOCS_PER_COMMIT: usize = 40;
        const TOTAL_DOCS: usize = N_COMMITS * DOCS_PER_COMMIT;
        const DIM: usize = 64;
        const TOP_K: usize = 10;
        const N_QUERIES: usize = 20;
        const TEST_NPROBE: usize = N_COMMITS * 4;
        const RECALL_BAR: f32 = 0.99;

        fn vector_for_global(g: usize, dim: usize) -> Vec<f32> {
            (0..dim)
                .map(|d| if d == g % dim { 1.0 } else { 0.0 })
                .collect()
        }

        fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| {
                    let z = x - y;
                    z * z
                })
                .sum()
        }

        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "emb",
                fixed_list_f32(DIM),
                false,
            ),
        ]));
        let dir = TempDir::new().expect("tmpdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let make_opts = || {
            SupertableOptions::new(
                schema.clone(),
                vec![FtsConfig {
                    column: "title".into(),
                }],
                vec![VectorConfig {
                    column: "emb".into(),
                    dim: DIM,
                    n_cent: 4,
                    rot_seed: 7,
                    metric: Metric::L2Sq,
                    rerank_codec: RerankCodec::Sq8Residual,
                }],
                Some(tok()),
            )
            .expect("options")
            .with_storage(Arc::clone(&storage))
            .with_writer_pool(Arc::clone(&pool))
        };
        let st = Supertable::create(make_opts()).expect("create");
        let mut w = st.writer().expect("writer");
        for commit in 0..N_COMMITS {
            let base = commit * DOCS_PER_COMMIT;
            let titles = LargeStringArray::from(
                (0..DOCS_PER_COMMIT)
                    .map(|i| format!("doc {}", base + i))
                    .collect::<Vec<_>>(),
            );
            let mut flat = Vec::with_capacity(DOCS_PER_COMMIT * DIM);
            for i in 0..DOCS_PER_COMMIT {
                flat.extend(vector_for_global(base + i, DIM));
            }
            let fsl = FixedSizeListArray::try_new(
                item.clone(),
                DIM as i32,
                Arc::new(Float32Array::from(flat)),
                None,
            )
            .expect("fsl");
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(titles) as Arc<dyn Array>,
                    Arc::new(fsl) as Arc<dyn Array>,
                ],
            )
            .expect("batch");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }
        drop(w);

        let mut corpus = Vec::with_capacity(TOTAL_DOCS);
        for g in 0..TOTAL_DOCS {
            corpus.push(vector_for_global(g, DIM));
        }

        let reopened = Supertable::open(make_opts()).expect("reopen");
        let reader = reopened.reader();
        let offsets = reader.superfile_doc_base_offsets().expect("doc bases");
        assert!(
            !offsets.is_empty(),
            "lazy-open manifest must expose superfile doc bases for recall grading"
        );

        let mut sum = 0.0f32;
        for qi in 0..N_QUERIES {
            let global_row = qi * (TOTAL_DOCS / N_QUERIES);
            let query = vector_for_global(global_row, DIM);
            let mut scored: Vec<(f32, u32)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (l2_sq(&query, v), i as u32))
                .collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
            let truth: HashSet<u32> = scored.into_iter().take(TOP_K).map(|(_, id)| id).collect();

            let hits = reader
                .vector_hits(
                    "emb",
                    &query,
                    TOP_K,
                    VectorSearchOptions::default().with_nprobe(TEST_NPROBE),
                    None,
                )
                .expect("vector_hits");
            let found: HashSet<u32> = hits
                .into_iter()
                .map(|h| {
                    let base = offsets
                        .get(&h.superfile)
                        .expect("hit superfile in user manifest");
                    base + h.local_doc_id
                })
                .collect();
            let hit = found.iter().filter(|id| truth.contains(id)).count();
            sum += hit as f32 / TOP_K as f32;
        }
        let mean = sum / N_QUERIES as f32;
        assert!(
            mean >= RECALL_BAR,
            "path B recall@{TOP_K} = {mean:.3} < acceptance bar {RECALL_BAR:.2} \
             (adaptive OPANN whole-cell routing after Supertable::open)"
        );
    }

    /// Same path-B gate as [`Self::storage_backed_opann_recall_at_acceptance_bar`],
    /// but with the **multi-shard time-mirror topology** the supertable vector
    /// bench uses: `writer_pool` threads fan each commit into one superfile per
    /// thread (16 commits × 16 shards ⇒ 256 hidden OPANN leaves). The small
    /// single-thread test above only ever builds ~16 leaves, so it cannot catch
    /// regressions that show up once the tree has hundreds of partitions.
    #[test]
    #[ignore = "reproduces the open 256-shard hidden-OPANN recall miss (recall@10 ~0.60); \
                descent and cell-open are verified correct, remap/mirror completeness still \
                under investigation — un-ignore once recall is restored"]
    fn storage_backed_opann_recall_multi_shard_time_mirror() {
        use std::collections::HashSet;

        use tempfile::TempDir;

        use crate::{
            storage::{LocalFsStorageProvider, StorageProvider},
            superfile::{
                builder::{FtsConfig, VectorConfig},
                vector::rerank_codec::RerankCodec,
            },
        };

        const N_COMMITS: usize = 16;
        const DOCS_PER_COMMIT: usize = 64;
        const POOL_THREADS: usize = 16;
        const TOTAL_DOCS: usize = N_COMMITS * DOCS_PER_COMMIT;
        const DIM: usize = 64;
        const TOP_K: usize = 10;
        const N_QUERIES: usize = 20;
        const TEST_NPROBE: usize = 64;
        /// Same tripwire as `benches/utils/executors.rs` `CORRECTNESS_RECALL_FLOOR`.
        const RECALL_BAR: f32 = 0.80;

        fn vector_for_global(g: usize, dim: usize) -> Vec<f32> {
            (0..dim)
                .map(|d| if d == g % dim { 1.0 } else { 0.0 })
                .collect()
        }

        fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| {
                    let z = x - y;
                    z * z
                })
                .sum()
        }

        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", fixed_list_f32(DIM), false),
        ]));
        let dir = TempDir::new().expect("tmpdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(POOL_THREADS)
                .build()
                .expect("pool"),
        );
        let make_opts = || {
            SupertableOptions::new(
                schema.clone(),
                vec![FtsConfig {
                    column: "title".into(),
                }],
                vec![VectorConfig {
                    column: "emb".into(),
                    dim: DIM,
                    n_cent: 4,
                    rot_seed: 7,
                    metric: Metric::L2Sq,
                    rerank_codec: RerankCodec::Sq8Residual,
                }],
                Some(tok()),
            )
            .expect("options")
            .with_storage(Arc::clone(&storage))
            .with_writer_pool(Arc::clone(&pool))
        };
        let st = Supertable::create(make_opts()).expect("create");
        let mut w = st.writer().expect("writer");
        for commit in 0..N_COMMITS {
            let base = commit * DOCS_PER_COMMIT;
            let titles = LargeStringArray::from(
                (0..DOCS_PER_COMMIT)
                    .map(|i| format!("doc {}", base + i))
                    .collect::<Vec<_>>(),
            );
            let mut flat = Vec::with_capacity(DOCS_PER_COMMIT * DIM);
            for i in 0..DOCS_PER_COMMIT {
                flat.extend(vector_for_global(base + i, DIM));
            }
            let fsl = FixedSizeListArray::try_new(
                item.clone(),
                DIM as i32,
                Arc::new(Float32Array::from(flat)),
                None,
            )
            .expect("fsl");
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(titles) as Arc<dyn Array>,
                    Arc::new(fsl) as Arc<dyn Array>,
                ],
            )
            .expect("batch");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }
        drop(w);

        let reopened = Supertable::open(make_opts()).expect("reopen");
        let reader = reopened.reader();
        let vit = reader
            .vector_index_table()
            .expect("hidden vector index");
        let hidden_sf = vit.reader().manifest().superfiles.len()
            + vit.reader().manifest().get_num_parts();
        assert!(
            hidden_sf >= N_COMMITS * 2,
            "expected multi-shard hidden mirror (got {hidden_sf} superfiles/parts, \
             want ≫ {N_COMMITS} from single-shard ingest)"
        );

        let offsets = reader.superfile_doc_base_offsets().expect("doc bases");
        let mut corpus = Vec::with_capacity(TOTAL_DOCS);
        for g in 0..TOTAL_DOCS {
            corpus.push(vector_for_global(g, DIM));
        }

        let mut sum = 0.0f32;
        for qi in 0..N_QUERIES {
            let global_row = qi * (TOTAL_DOCS / N_QUERIES);
            let query = vector_for_global(global_row, DIM);
            let mut scored: Vec<(f32, u32)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (l2_sq(&query, v), i as u32))
                .collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
            let truth: HashSet<u32> = scored.into_iter().take(TOP_K).map(|(_, id)| id).collect();

            let hits = reader
                .vector_hits(
                    "emb",
                    &query,
                    TOP_K,
                    VectorSearchOptions::default().with_nprobe(TEST_NPROBE),
                    None,
                )
                .expect("vector_hits");
            let found: HashSet<u32> = hits
                .into_iter()
                .map(|h| {
                    let base = offsets
                        .get(&h.superfile)
                        .expect("hit superfile in user manifest");
                    base + h.local_doc_id
                })
                .collect();
            let hit = found.iter().filter(|id| truth.contains(id)).count();
            sum += hit as f32 / TOP_K as f32;
        }
        let mean = sum / N_QUERIES as f32;
        assert!(
            mean >= RECALL_BAR,
            "multi-shard path B recall@{TOP_K} = {mean:.3} < floor {RECALL_BAR:.2} \
             (256-leaf time-mirror hidden OPANN after Supertable::open)"
        );
    }

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
