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
//! let rows: Vec<RecordBatch> =
//!     table.vector_search("emb", &query_vec, 10, opts, None, Some(&["_id", "title", "score"]))?;
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
use roaring::RoaringBitmap;
use tracing::debug;

use super::{SuperfileHit, candidate::CandidatePlan, dispatch, exec::common::resolve_hits_named};
pub use crate::superfile::reader::VectorSearchOptions;
use crate::{
    superfile::{
        SuperfileReader,
        error::{ReadError, VectorError},
        fts::reader::BoolMode,
        vector::distance::Metric,
    },
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        manifest::{SuperfileEntry, SuperfileUri},
        tombstones::SidecarCache,
    },
};

/// Map a per-superfile vector-search error to a query error. A budget refusal
/// (the kernel's `VectorError::OverBudget`, boxed in `ReadError::Vector`) keeps
/// its own variant so it surfaces as the public `InfinoError::OverBudget`;
/// anything else is a generic query error.
fn vector_read_query_error(e: ReadError) -> QueryError {
    match e {
        ReadError::Vector(v) => match *v {
            VectorError::OverBudget(m) => QueryError::OverBudget(m),
            other => QueryError::Parquet(other.to_string()),
        },
        other => QueryError::Parquet(other.to_string()),
    }
}

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

/// How to probe one superfile in the vector fan-out: the globally-selected
/// cluster ids for that superfile, or — for a superfile whose manifest
/// summary carries no per-cluster centroids — a normal per-superfile
/// `nprobe` probe (fallback, never silently dropped).
enum Probe {
    Clusters(Vec<u32>),
    Nprobe,
}

impl SupertableReader {
    /// Single-column vector kNN search across the pinned
    /// manifest's superfiles.
    ///
    /// Returns up to `k` lowest-distance hits, sorted ascending.
    /// `query` must match the column's declared `dim`.
    ///
    /// `options` (see [`VectorSearchOptions`]) controls per-
    /// superfile recall-vs-latency knobs (`nprobe`, `rerank_mult`).
    /// Defaults recover ≥0.9 recall@10 on typical IVF setups.
    ///
    /// Empty supertable (no superfiles) and `k == 0` short-circuit
    /// to an empty `Vec`.
    ///
    /// `pub(crate)` async kernel — the public surface is the sync
    /// [`SupertableReader::vector_search`], which drives this via the
    /// sync→async bridge.
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column, k = k, dim = query.len()))
    )]
    pub(crate) async fn vector_search_async(
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
        // Unfiltered: no per-superfile allow-set.
        self.vector_fanout_over_superfiles(superfiles, column, query, k, options, None)
            .await
    }

    /// Shared cluster-selection + fan-out body for both the unfiltered
    /// [`Self::vector_search_async`] and the filtered
    /// [`Self::vector_hits_filtered_async`].
    ///
    /// `superfiles` is the already-vector-pruned candidate list.
    /// `allow` (when `Some`) maps each superfile to its predicate
    /// allow-set of `local_doc_id`s; a superfile absent from the map is
    /// skipped (its predicate matched nothing), and each present
    /// superfile's kernel ranks distance only among its allowed doc-ids.
    /// `None` is the unfiltered path — every pruned superfile fans out
    /// with no allow-set.
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
        // Folded Sq8-domain scoring (`ClusterCentroids::score_clusters_into`):
        // Σq / ‖q‖² once per query, then one SIMD Sq8 dot per cluster over
        // the contiguous code rows — no per-cluster dequantize, no scratch.
        let sum_q: f32 = query.iter().sum();
        let norm_q_sq: f32 = query.iter().map(|v| v * v).sum();
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
                        .score_clusters_into(metric, query, sum_q, norm_q_sq, |c, score| {
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

        // The filtered path fans out in waves capped by the reader-pool
        // width: a cold vector kernel can hold large `[codes][doc_ids]`
        // cluster blocks while building its shortlist, so capping
        // concurrent superfiles keeps that transient memory bounded by
        // config, not table size. Skipped superfiles issue zero GETs.
        let column_arc = Arc::new(column.to_owned());
        let query_arc = Arc::new(query.to_vec());
        let reader_pool = Arc::clone(&manifest.options.reader_pool);
        // Per-connection memory budget: gates each superfile's cold cluster-block fetch.
        let budget = Some(Arc::clone(&manifest.options.connection_memory_budget));

        // `fanout_with`, not `fanout`: the body needs each superfile's  tombstone bitmap *before* its kernel
        // to push it in as a deny set (`fanout` only drops tombstones post-rank, which underflows).
        let body = move |reader: Arc<SuperfileReader>,
                         entry: Arc<SuperfileEntry>,
                         tombstone_cache: Option<Arc<SidecarCache>>,
                         now: Instant,
                         (probe, bitmap): (Probe, Option<Arc<RoaringBitmap>>)| {
            let column = Arc::clone(&column_arc);
            let query = Arc::clone(&query_arc);
            let reader_pool = Arc::clone(&reader_pool);
            let budget = budget.clone();
            async move {
                // For unfiltered path:
                //  - it resolve this superfile's tombstone  bitmap once (a warm cache hit after the orchestrator's
                //    prefetch) and push it down as the deny set.
                //  - A clean superfile resolves to `None`, leaving the hot path unchanged.
                //
                // Filtered search leaves it `None` throughout — its allow-set already excludes tombstones.
                let deny = match tombstone_cache.as_ref() {
                    Some(cache) if bitmap.is_none() => {
                        dispatch::tombstone_deny_set(cache, entry.superfile_id, now)?
                    }
                    _ => None,
                };
                let pool = Some(Arc::clone(&reader_pool));
                let hits = match probe {
                    Probe::Clusters(ids) => {
                        reader
                            .vector_search_clusters_filtered(
                                &column, &query, k, &ids, options, bitmap, deny, pool, budget,
                            )
                            .await
                    }
                    Probe::Nprobe => {
                        reader
                            .vector_hits_filtered_async(
                                &column, &query, k, options, bitmap, deny, pool, budget,
                            )
                            .await
                    }
                }
                .map_err(vector_read_query_error)?;

                Ok::<Vec<SuperfileHit>, QueryError>(dispatch::tag_hits(&entry, hits))
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
                collected.extend(dispatch::fanout_with(self, wave, body.clone()).await?);
            }
            collected
        } else {
            dispatch::fanout_with(self, units, body).await?
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
    /// [`Self::vector_hits_filtered`].
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column, k = k, dim = query.len()))
    )]
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
        let superfiles = manifest
            .get_pruned_superfiles_for_vector(column, query)
            .await
            .map_err(QueryError::ManifestLoad)?;
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize the predicate once with the index tokenizer (the same
        // tokenizer used at build time, so the terms match the postings).
        // No tokens (e.g. empty / punctuation-only) ⇒ nothing matches.
        let Some(tokenizer) = manifest.options.tokenizer.as_ref() else {
            return Ok(Vec::new());
        };
        let tokens: Vec<String> = tokenizer.tokenize(filter.query).collect();
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Build the per-superfile allow-set: one `token_match` per
        // candidate superfile (postings only — the same retrieval
        // `CandidatePlan::TermsAll` uses), grouped to a
        // `superfile → RoaringBitmap` map. Empty bitmaps are dropped so
        // their superfile never fans out. Reuses the shared `fanout`
        // orchestrator (concurrent reader opens) but keeps each unit's
        // result keyed to its superfile URI.
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
        let manifest = self.manifest();
        let superfiles = manifest
            .get_pruned_superfiles_for_vector(column, query)
            .await
            .map_err(QueryError::ManifestLoad)?;
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
}

fn subtract_tombstones(
    bm: &mut RoaringBitmap,
    entry: &SuperfileEntry,
    tombstone_cache: Option<&SidecarCache>,
    now: Instant,
) -> Result<(), QueryError> {
    if let Some(cache) = tombstone_cache
        && let Some(deleted) = dispatch::tombstone_deny_set(cache, entry.superfile_id, now)?
    {
        *bm -= &*deleted;
    }
    Ok(())
}

impl SupertableReader {
    /// Single-column vector kNN search over this reader's pinned
    /// snapshot, materialized as Arrow rows.
    ///
    /// This is the user-facing row-returning path. It runs the same
    /// vector hit kernel the SQL TVF uses, then resolves those top-k hits
    /// through the shared row materializer. Returned batches include
    /// `_id`, every visible scalar column, and a trailing `score` column
    /// containing the distance (smaller is better).
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
                None => self.vector_search_async(column, query, k, options).await?,
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

    /// Low-level vector kNN search over this reader's pinned snapshot.
    ///
    /// When `filter` is `Some`, kNN is ranked only among rows matching
    /// the text predicate (pushdown, not post-filter). `None` is the
    /// unfiltered path.
    ///
    /// Drives the internal async kernel to completion via the
    /// sync→async bridge ([`SupertableReader::block_on`]). Returns up
    /// to `k` hits sorted by distance *ascending*.
    pub fn vector_hits(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        filter: Option<VectorFilter<'_>>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        match filter {
            None => self.block_on(self.vector_search_async(column, query, k, options)),
            Some(f) => self.block_on(self.vector_hits_filtered_async(column, query, k, options, f)),
        }
    }
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
    /// `filter` optionally restricts the search to rows matching a text
    /// predicate (see [`VectorFilter`]): kNN is ranked only among the
    /// matching rows — a pushdown into the ranking, not a post-filter over
    /// the global top-`k`, so it still returns the true `k` nearest
    /// *matching* rows even when nearer non-matching rows exist. The filter
    /// column must be FTS-indexed; `None` searches all rows.
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
    /// # use infino::arrow_array::{FixedSizeListArray, Float32Array, RecordBatch};
    /// # use infino::arrow_array::types::Float32Type;
    /// # use infino::arrow_schema::{DataType, Field, Schema};
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
    /// let rows = vecs.vector_search("emb", &query, 10, VectorSearchOptions::new(), None, Some(&["_id", "score"]))?;
    /// assert!(rows.iter().map(|b| b.num_rows()).sum::<usize>() >= 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column, k = k, dim = query.len()))
    )]
    pub fn vector_search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        filter: Option<VectorFilter<'_>>,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, crate::InfinoError> {
        debug!(column, k, dim = query.len(), "vector_search");
        self.reader()
            .vector_search(column, query, k, options, filter, projection)
            .map_err(crate::InfinoError::from)
            .map_err(|e| e.with_context("vector_search", None))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cmp::Ordering,
        collections::{HashMap, HashSet},
        future::Future,
        sync::Arc,
        time::Instant,
    };

    use arrow::array::Array;
    use arrow_array::{
        Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch,
    };
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use datafusion::prelude::{Expr, col, lit};
    use roaring::RoaringBitmap;
    use tokio::runtime;

    use super::{VectorFilter, VectorSearchOptions, vector_read_query_error};
    use crate::{
        InfinoError,
        superfile::{
            SuperfileReader,
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder, VectorConfig},
            error::{ReadError, VectorError},
            vector::{distance::Metric, rerank_codec::RerankCodec},
        },
        supertable::{
            Supertable, SupertableOptions,
            error::QueryError,
            options::{DECIMAL128_PRECISION, DECIMAL128_SCALE},
            query::dispatch::apply_tombstone_filter,
        },
        test_helpers::default_tokenizer as tok,
    };

    /// Drive an async future to completion on a throwaway current-thread
    /// runtime. Used only for the single-superfile `SuperfileReader`
    /// oracle, whose search surface is async-only; the supertable
    /// reader's own search methods are sync and need no runtime here.
    fn block_on<F: Future>(fut: F) -> F::Output {
        runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime")
            .block_on(fut)
    }

    #[test]
    fn over_budget_vector_error_surfaces_as_infino_over_budget() {
        // A cold vector search that crosses the budget returns
        // `VectorError::OverBudget` (see the vector reader tests). Confirm it
        // routes all the way to the public `InfinoError::OverBudget` and isn't
        // flattened to a generic query error.
        let read_err = ReadError::Vector(Box::new(VectorError::OverBudget("gate".into())));
        let q = vector_read_query_error(read_err);

        assert!(matches!(q, QueryError::OverBudget(_)), "got {q:?}");
        assert!(matches!(
            InfinoError::from(QueryError::OverBudget("x".into())),
            InfinoError::OverBudget(_)
        ));

        // A non-budget read error stays a generic query error.
        assert!(matches!(
            vector_read_query_error(ReadError::MissingKv("k")),
            QueryError::Parquet(_)
        ));
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
                rerank_codec: RerankCodec::Fp32,
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
    fn build_oracle_superfile(n_total: usize, dim: usize) -> Arc<SuperfileReader> {
        // Oracle path goes through SuperfileBuilder directly,
        // so we mimic the supertable's effective schema by hand:
        // `_id` is `Decimal128(38, 0)`, ids are 0..n.
        let scalar_schema = Arc::new(Schema::new(vec![
            Field::new(
                "_id",
                DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE),
                false,
            ),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(
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
                rerank_codec: RerankCodec::Fp32,
            }],
            Some(tok()),
        );
        let mut b = SuperfileBuilder::new(opts).expect("builder");

        let ids = Decimal128Array::from((0..n_total as i128).collect::<Vec<_>>())
            .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
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
        let bytes = Bytes::from(b.finish().expect("finish"));
        Arc::new(SuperfileReader::open(bytes).expect("open"))
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

    // Deleting the rows nearest a query must not shrink an unfiltered
    // result below the live count — the delete-then-kNN underflow.
    //  - one superfile, so a hit's `local_doc_id` is its row index.
    //  - rank all rows, then tombstone the nearest `d` (well past the
    //    small-k candidate pool, so survivors fall outside any post-rank
    //    backfill).
    //  - `k=1` must return the nearest *live* row, never empty.
    //  - a full sweep returns exactly the `n - d` live rows.
    // Pre-fix the filter ran on the already-truncated top-k with no
    // backfill, so a deleted row in the top-k just vanished.
    #[test]
    fn vector_search_returns_k_live_rows_after_deletes() {
        let dim = 16;
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(options_one_superfile_per_commit(dim).with_storage(storage))
            .expect("create");

        // Spread pseudo-random vectors (integer hash → [0, 1)) give a
        // strict nearest ordering, unlike one-hot rows that tie.
        let n = 64usize;
        let pseudo = |i: usize, d: usize| -> f32 {
            let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(
                (d as u64)
                    .wrapping_add(1)
                    .wrapping_mul(0x2545_F491_4F6C_DD1D),
            );
            h ^= h >> 33;
            ((h >> 40) & 0xFFFF) as f32 / 65536.0
        };
        let schema = st.options().schema.clone();
        let titles = LargeStringArray::from((0..n).map(|i| format!("doc {i}")).collect::<Vec<_>>());
        let mut flat = Vec::<f32>::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push(pseudo(i, d));
            }
        }
        let fsl = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            Arc::new(Float32Array::from(flat)) as Arc<dyn Array>,
            None,
        )
        .expect("fsl");
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(titles), Arc::new(fsl)]).expect("batch");
        {
            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        let q: Vec<f32> = (0..dim).map(|d| pseudo(0, d)).collect();

        let baseline = st
            .reader()
            .vector_hits("emb", &q, n, VectorSearchOptions::new(), None)
            .expect("baseline");
        assert_eq!(baseline.len(), n);
        let d = 56usize;
        let deleted_ids: HashSet<u32> = baseline[..d].iter().map(|h| h.local_doc_id).collect();
        let deleted: Vec<Expr> = deleted_ids
            .iter()
            .map(|id| lit(format!("doc {id}")))
            .collect();
        let stats = st
            .delete(col("title").in_list(deleted, false))
            .expect("delete");
        assert_eq!(stats.n_tombstoned(), d);

        // k=1 must return the nearest *live* row, not an empty result.
        let hits = st
            .reader()
            .vector_hits("emb", &q, 1, VectorSearchOptions::new(), None)
            .expect("search");
        assert_eq!(hits.len(), 1, "k=1 returned no live row after deletes");
        assert!(
            !deleted_ids.contains(&hits[0].local_doc_id),
            "returned a deleted row: local_doc_id={}",
            hits[0].local_doc_id
        );

        // And a larger k returns exactly the live remainder.
        let live = n - d;
        let many = st
            .reader()
            .vector_hits("emb", &q, n, VectorSearchOptions::new(), None)
            .expect("search many");
        assert_eq!(many.len(), live, "expected all {live} live rows");
        assert!(many.iter().all(|h| !deleted_ids.contains(&h.local_doc_id)));
    }

    // The deny set is per-superfile, so the fix must hold across the
    // fan-out, not just within one file.
    //  - 3 superfiles × 16 one-hot docs; titles repeat per file, so
    //    `doc 0` names row 0 (the dim-0-active, distance-0 row) in each.
    //  - query e_0's only distance-0 neighbors are those three `doc 0`s.
    //  - deleting `doc 0` tombstones all three at once.
    // k=1 must then return a live distance-1 row from some superfile, not
    // empty — the global merge backfills past the per-file tombstones.
    #[test]
    fn vector_search_after_deletes_holds_across_superfiles() {
        let dim = 16;
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(options_one_superfile_per_commit(dim).with_storage(storage))
            .expect("create");
        let schema = st.options().schema.clone();
        {
            let mut w = st.writer().expect("writer");
            for chunk in 0..3u64 {
                w.append(&build_vector_batch(chunk * 16, 16, dim, schema.clone()))
                    .expect("append");
                w.commit().expect("commit");
            }
        }
        assert_eq!(st.reader().n_superfiles(), 3);

        let mut q = vec![0f32; dim];
        q[0] = 1.0;
        // The three distance-0 neighbors (one per superfile) all carry the
        // title `doc 0`; one delete tombstones every one of them.
        let stats = st
            .delete(col("title").in_list(vec![lit("doc 0")], false))
            .expect("delete");
        assert_eq!(stats.n_tombstoned(), 3);

        let hits = st
            .reader()
            .vector_hits("emb", &q, 1, VectorSearchOptions::new(), None)
            .expect("search");
        assert_eq!(hits.len(), 1, "k=1 underflowed across the fan-out");
        // Survivors are orthogonal to e_0 → cosine distance 1, never the
        // deleted distance-0 rows.
        assert!(hits[0].score > 0.5, "returned a deleted distance-0 row");
    }

    // Deleting every row must return empty, not panic or wrap — the deny
    // set excludes the whole superfile and the heap stays empty.
    #[test]
    fn vector_search_all_rows_deleted_returns_empty() {
        let dim = 16;
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(options_one_superfile_per_commit(dim).with_storage(storage))
            .expect("create");
        let schema = st.options().schema.clone();
        let n = 8usize;
        {
            let mut w = st.writer().expect("writer");
            w.append(&build_vector_batch(0, n, dim, schema)).expect("a");
            w.commit().expect("c");
        }
        let titles: Vec<Expr> = (0..n).map(|i| lit(format!("doc {i}"))).collect();
        let stats = st
            .delete(col("title").in_list(titles, false))
            .expect("delete");
        assert_eq!(stats.n_tombstoned(), n);

        let q = vec![0.1f32; dim];
        let hits = st
            .reader()
            .vector_hits("emb", &q, 5, VectorSearchOptions::new(), None)
            .expect("search");
        assert!(hits.is_empty(), "expected no live rows, got {}", hits.len());
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

    // ---- Filtered vector search (pushdown) --------------------------
    //
    // The acceptance test for the feature: vector kNN restricted to
    // rows matching a text predicate, where the predicate is pushed
    // *into* the per-superfile coarse shortlist. The strong assertion
    // is that the result equals the brute-force k-nearest among ONLY
    // the matching rows — proving it's k-nearest-among-matching, not a
    // post-filter over a global top-k (which would underflow / return
    // the wrong set whenever nearer non-matching rows exist).

    use super::BoolMode;

    /// Rows per commit (= per superfile) in the filtered-search corpus.
    const FILTER_DOCS_PER_SEG: usize = 30;
    /// Number of commits / superfiles in the filtered-search corpus.
    const FILTER_N_SEG: usize = 4;
    /// Vector dimensionality for the filtered-search corpus.
    const FILTER_DIM: usize = 64;

    /// Deterministic, reproducible scalar in `[-1, 1)` from two indices —
    /// a tiny splitmix64-style hash, so the test corpus is fixed across
    /// runs without pulling in an RNG dependency.
    fn pseudo(global_id: usize, d: usize) -> f32 {
        let mut x = (global_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (d as u64 + 1);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        // Map the high bits to [0, 1) then to [-1, 1).
        let unit = (x >> 11) as f32 / (1u64 << 53) as f32;
        unit * 2.0 - 1.0
    }

    /// The (deterministic, L2-normalized) vector for a global id —
    /// shared by the corpus builder and the brute-force oracle so the
    /// stored bytes and the ground-truth distances agree exactly.
    fn filter_vec(global_id: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..FILTER_DIM).map(|d| pseudo(global_id, d)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    /// The filter token for a global id. Interleaves alpha/beta by a
    /// rule (`global_id % 3 == 0` ⇒ beta, else alpha) that is
    /// independent of the vector geometry, so the query's nearest
    /// neighbors are a mix of both — making post-filtering observably
    /// wrong versus true pushdown.
    fn filter_token(global_id: usize) -> &'static str {
        if global_id.is_multiple_of(3) {
            "beta"
        } else {
            "alpha"
        }
    }

    /// Build one superfile's batch for the filtered-search corpus.
    fn build_filter_batch(start: usize, n: usize, schema: Arc<Schema>) -> RecordBatch {
        let titles = LargeStringArray::from(
            (0..n)
                .map(|i| format!("row {} {}", start + i, filter_token(start + i)))
                .collect::<Vec<_>>(),
        );
        let mut flat = Vec::<f32>::with_capacity(n * FILTER_DIM);
        for i in 0..n {
            flat.extend_from_slice(&filter_vec(start + i));
        }
        let fsl = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            FILTER_DIM as i32,
            Arc::new(Float32Array::from(flat)) as Arc<dyn Array>,
            None,
        )
        .expect("FSL");
        RecordBatch::try_new(schema, vec![Arc::new(titles), Arc::new(fsl)]).expect("batch")
    }

    /// Create + populate the filtered-search supertable (Fp32 rerank so
    /// the rerank distances are exact — no quantization slack in the
    /// ground-truth comparison) and return it.
    fn build_filter_supertable() -> Supertable {
        let st = Supertable::create(options_one_superfile_per_commit(FILTER_DIM)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        for seg in 0..FILTER_N_SEG {
            let start = seg * FILTER_DOCS_PER_SEG;
            w.append(&build_filter_batch(
                start,
                FILTER_DOCS_PER_SEG,
                schema.clone(),
            ))
            .expect("append");
            w.commit().expect("commit");
        }
        st
    }

    /// Map a hit back to its global id via manifest superfile order
    /// (each commit is one superfile of `FILTER_DOCS_PER_SEG` docs in
    /// append order — same convention as the unfiltered oracle test).
    fn hit_global_id(reader: &SupertableReader, h: &SuperfileHit) -> usize {
        let manifest = reader.manifest();
        let seg = manifest
            .superfiles
            .iter()
            .position(|e| e.uri == h.superfile)
            .expect("superfile in manifest");
        seg * FILTER_DOCS_PER_SEG + h.local_doc_id as usize
    }

    /// Brute-force k-nearest *among rows carrying `token`* by exact
    /// cosine distance (`1 - dot` on unit vectors). Returns the global
    /// ids, nearest first.
    fn brute_force_filtered_topk(query: &[f32], token: &str, k: usize) -> Vec<usize> {
        let total = FILTER_N_SEG * FILTER_DOCS_PER_SEG;
        let mut scored: Vec<(usize, f32)> = (0..total)
            .filter(|&g| filter_token(g) == token)
            .map(|g| {
                let v = filter_vec(g);
                let dot: f32 = query.iter().zip(&v).map(|(a, b)| a * b).sum();
                (g, 1.0 - dot)
            })
            .collect();
        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.into_iter().take(k).map(|(g, _)| g).collect()
    }

    use crate::supertable::handle::SupertableReader;

    #[test]
    fn vector_search_filtered_returns_knn_among_matching_rows_only() {
        let st = build_filter_supertable();
        let reader = st.reader();

        // Query = a fixed corpus vector's direction, so it has a dense
        // neighborhood of both alpha and beta rows around it.
        let query = filter_vec(7);
        let k = 8;
        // Full nprobe + generous rerank so the IVF stage has ~1.0 recall
        // on this small corpus: any miss would be the IVF's, not the
        // filter's, and we want to isolate the filter's correctness.
        let opts = VectorSearchOptions::new()
            .with_nprobe(64)
            .with_rerank_mult(64);

        let hits = reader
            .vector_hits(
                "emb",
                &query,
                k,
                opts,
                Some(VectorFilter {
                    column: "title",
                    query: "alpha",
                    mode: BoolMode::Or,
                }),
            )
            .expect("filtered query");

        // (a) Hard constraint: EVERY returned hit is an alpha row.
        for h in &hits {
            let g = hit_global_id(&reader, h);
            assert_eq!(
                filter_token(g),
                "alpha",
                "hit global_id={g} is not an alpha row (filter must be a hard constraint)"
            );
        }

        // (b) Exactness: the returned set equals the brute-force
        // k-nearest among ALPHA rows only. This is the proof that the
        // predicate is pushed into the ranking — a post-filter over the
        // global top-k would drop nearer beta rows and return a
        // different (and short) set.
        let got: HashSet<usize> = hits.iter().map(|h| hit_global_id(&reader, h)).collect();
        let truth: HashSet<usize> = brute_force_filtered_topk(&query, "alpha", k)
            .into_iter()
            .collect();
        assert_eq!(
            got.len(),
            k,
            "filtered kNN must return exactly k matching hits"
        );
        assert_eq!(
            got, truth,
            "filtered kNN set must equal brute-force k-nearest among alpha rows;\n got   = {got:?}\n truth = {truth:?}"
        );

        // Sanity: a naive post-filter of the *global* top-k would have
        // underflowed here (the unfiltered top-k contains beta rows that
        // are nearer than the k-th alpha row), so this corpus actually
        // exercises the pushdown path rather than coincidentally
        // agreeing with a post-filter.
        let global = reader
            .vector_hits("emb", &query, k, opts, None)
            .expect("unfiltered query");
        let global_alpha = global
            .iter()
            .filter(|h| filter_token(hit_global_id(&reader, h)) == "alpha")
            .count();
        assert!(
            global_alpha < k,
            "test corpus is mis-tuned: the global top-{k} already had {global_alpha} alpha rows, \
             so a post-filter wouldn't underflow and the test wouldn't distinguish pushdown"
        );
    }

    #[test]
    fn vector_search_filtered_results_are_distance_ascending() {
        let st = build_filter_supertable();
        let reader = st.reader();
        let query = filter_vec(11);
        let opts = VectorSearchOptions::new()
            .with_nprobe(64)
            .with_rerank_mult(64);
        let hits = reader
            .vector_hits(
                "emb",
                &query,
                6,
                opts,
                Some(VectorFilter {
                    column: "title",
                    query: "alpha",
                    mode: BoolMode::Or,
                }),
            )
            .expect("filtered query");
        assert!(!hits.is_empty());
        for w in hits.windows(2) {
            assert!(
                w[0].score <= w[1].score,
                "expected ascending distance: {:?} then {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn vector_search_filtered_empty_match_returns_empty() {
        let st = build_filter_supertable();
        let reader = st.reader();
        let query = filter_vec(3);
        let opts = VectorSearchOptions::new().with_nprobe(64);
        // No row's title contains this token, so the predicate matches
        // nothing in any superfile → empty result (no fan-out GETs).
        let hits = reader
            .vector_hits(
                "emb",
                &query,
                10,
                opts,
                Some(VectorFilter {
                    column: "title",
                    query: "nonexistenttoken",
                    mode: BoolMode::Or,
                }),
            )
            .expect("filtered query");
        assert!(
            hits.is_empty(),
            "empty-match filter must return empty: {hits:?}"
        );
    }

    #[test]
    fn vector_search_filtered_rows_resolve_and_carry_score() {
        // The row-returning `Supertable::vector_search_filtered` path:
        // bare projection is `_id` + `score`; a named projection
        // materializes the scalar column. Also confirms the resolved
        // rows are exactly the matching (alpha) rows.
        let st = build_filter_supertable();
        let query = filter_vec(5);
        let opts = VectorSearchOptions::new()
            .with_nprobe(64)
            .with_rerank_mult(64);

        let bare = st
            .vector_search(
                "emb",
                &query,
                5,
                opts,
                Some(VectorFilter {
                    column: "title",
                    query: "alpha",
                    mode: BoolMode::Or,
                }),
                None,
            )
            .expect("filtered rows bare");
        let n: usize = bare.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(n, 5, "five matching nearest rows");
        assert_eq!(bare[0].num_columns(), 2, "_id + score");

        let projected = st
            .vector_search(
                "emb",
                &query,
                5,
                opts,
                Some(VectorFilter {
                    column: "title",
                    query: "alpha",
                    mode: BoolMode::Or,
                }),
                Some(&["_id", "title", "score"]),
            )
            .expect("filtered rows projected");
        let titles = projected[0]
            .column(1)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title col");
        for i in 0..titles.len() {
            assert!(
                titles.value(i).contains("alpha"),
                "resolved row {} is not an alpha row: {:?}",
                i,
                titles.value(i)
            );
        }
    }

    #[test]
    fn vector_search_filtered_k_zero_short_circuits() {
        let st = build_filter_supertable();
        let reader = st.reader();
        let query = filter_vec(1);
        let hits = reader
            .vector_hits(
                "emb",
                &query,
                0,
                VectorSearchOptions::new(),
                Some(VectorFilter {
                    column: "title",
                    query: "alpha",
                    mode: BoolMode::Or,
                }),
            )
            .expect("k=0");
        assert!(hits.is_empty());
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
            SuperfileUri,
            manifest::SuperfileEntry,
            query::SuperfileHit,
            tombstones::{SidecarCache, TombstoneSeqView, cache::DEFAULT_SEAL_TTL},
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
            scalar_stats: HashMap::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
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
        let sf_id = Uuid::from_u128(0xFEEDFACE);
        let cache = Arc::new(SidecarCache::new(
            ws.clone(),
            DEFAULT_SEAL_TTL,
            Arc::new(TombstoneSeqView {
                manifest_id: 1,
                seqs: [(sf_id, 1u64)].into_iter().collect(),
            }),
        ));
        // Pre-populate a sidecar with doc-ids 1, 3, 5 set.
        let mut bitmap = RoaringBitmap::new();
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

        apply_tombstone_filter(Some(&cache), &entry, &mut hits, Instant::now()).expect("filter");

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
        apply_tombstone_filter(None, &entry, &mut hits, Instant::now()).expect("no-cache");
        assert_eq!(hits, original);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_tombstone_filter_short_circuits_on_empty_bitmap() {
        // Superfile absent from the seq map → the cache answers
        // "no tombstones" authoritatively (zero GETs) and
        // `bitmap.is_empty()` short-circuits the filter loop.
        // Hit list is unchanged.
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let ws = WalStore::new(Arc::clone(&storage));
        let cache = Arc::new(SidecarCache::new(
            ws,
            DEFAULT_SEAL_TTL,
            Arc::new(TombstoneSeqView::default()),
        ));

        let entry = synthetic_entry(Uuid::from_u128(0x1111));
        let mut hits: Vec<SuperfileHit> = (0..4u32)
            .map(|d| SuperfileHit {
                superfile: entry.uri,
                local_doc_id: d,
                score: 0.0,
            })
            .collect();
        let original = hits.clone();
        apply_tombstone_filter(Some(&cache), &entry, &mut hits, Instant::now()).expect("filter");
        assert_eq!(hits, original);
    }
}
