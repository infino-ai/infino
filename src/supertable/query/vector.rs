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
//! let ids: Vec<RecordBatch> = table.vector_search("emb", &query_vec, 10, opts, None)?;
//! // Materialize row data by naming the columns to decode.
//! let rows: Vec<RecordBatch> =
//!     table.vector_search("emb", &query_vec, 10, opts, Some(&["_id", "title", "score"]))?;
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

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use crate::superfile::SuperfileReader;
pub use crate::superfile::reader::VectorSearchOptions;
use crate::superfile::vector::distance::Metric;
use crate::supertable::error::QueryError;
use crate::supertable::handle::{Supertable, SupertableReader};
use crate::supertable::manifest::SuperfileEntry;
use arrow::record_batch::RecordBatch;

use super::SuperfileHit;
use super::exec::common::resolve_hits_named;

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

        let superfiles: Vec<Arc<SuperfileEntry>> = match manifest.list.as_ref() {
            Some(list) => {
                let kept = crate::supertable::manifest::list_prune::prune_parts_for_vector(
                    list,
                    column,
                    query,
                    f32::INFINITY,
                );
                crate::supertable::query::hierarchical_iter::load_and_flatten(
                    manifest.as_ref(),
                    &kept,
                )
                .await?
            }
            None => crate::supertable::query::hierarchical_iter::fallback_to_flat_superfiles(
                manifest.as_ref(),
            ),
        };
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }

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
        // tuning may lower this.)
        let n_eligible = superfiles.len().saturating_sub(fallback.len());
        let budget = options
            .nprobe
            .saturating_mul(n_eligible.max(1))
            .max(options.nprobe);
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
        // (the cross-superfile win).
        let fallback: std::collections::HashSet<usize> = fallback.into_iter().collect();
        let mut units: Vec<(Arc<SuperfileEntry>, Probe)> = Vec::new();
        for (si, entry) in superfiles.iter().enumerate() {
            if let Some(ids) = per_seg.remove(&si) {
                units.push((Arc::clone(entry), Probe::Clusters(ids)));
            } else if fallback.contains(&si) {
                units.push((Arc::clone(entry), Probe::Nprobe));
            }
        }
        if units.is_empty() {
            return Ok(Vec::new());
        }

        // Fan out through the shared [`query::dispatch::fanout`] (also
        // used by FTS): one tokio task per probed superfile opens the
        // reader and runs the kNN kernel — cold GETs across superfiles are
        // concurrent (tokio owns I/O), shortlist + rerank stay on rayon.
        // Skipped superfiles issue zero GETs.
        let column_arc = Arc::new(column.to_owned());
        let query_arc = Arc::new(query.to_vec());
        let kernel = move |reader: Arc<SuperfileReader>, probe: Probe| {
            let column = Arc::clone(&column_arc);
            let query = Arc::clone(&query_arc);
            async move {
                let res = match probe {
                    Probe::Clusters(ids) => {
                        reader
                            .vector_search_clusters(&column, &query, k, &ids, options)
                            .await
                    }
                    Probe::Nprobe => reader.vector_hits_async(&column, &query, k, options).await,
                };
                res.map_err(|e| QueryError::Parquet(e.to_string()))
            }
        };
        let per_superfile = crate::supertable::query::dispatch::fanout(self, units, kernel).await?;

        Ok(top_k_ascending(per_superfile, k))
    }
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
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, QueryError> {
        self.block_on(async {
            let hits = self.vector_search_async(column, query, k, options).await?;
            // `projection` selects output columns by name (`_id`, the
            // visible scalar columns, or the trailing `score`); `None`
            // returns `_id` + `score` only. The shared resolver decodes
            // only the projected columns.
            let batch = resolve_hits_named(self, &hits, projection, "vector_search")
                .await
                .map_err(|e| QueryError::Execute(e.to_string()))?;
            Ok(vec![batch])
        })
    }

    /// Low-level vector kNN search over this reader's pinned snapshot.
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
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.block_on(self.vector_search_async(column, query, k, options))
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
    /// let hits = vecs.vector_search("emb", &query, 10, VectorSearchOptions::new(), None)?;
    /// assert_eq!(hits[0].num_columns(), 2);
    /// // Explicit projection names the same columns (scalar columns,
    /// // when present, materialize row data):
    /// let rows = vecs.vector_search("emb", &query, 10, VectorSearchOptions::new(), Some(&["_id", "score"]))?;
    /// assert!(rows.iter().map(|b| b.num_rows()).sum::<usize>() >= 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn vector_search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, crate::InfinoError> {
        self.reader()
            .vector_search(column, query, k, options, projection)
            .map_err(crate::InfinoError::from)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Array;
    use arrow_array::{FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    use crate::superfile::builder::{FtsConfig, SuperfileBuilder, VectorConfig};

    use crate::superfile::vector::distance::Metric;
    use crate::supertable::error::QueryError;
    use crate::supertable::{Supertable, SupertableOptions};

    use super::VectorSearchOptions;

    use crate::test_helpers::default_tokenizer as tok;

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
            .vector_hits("emb", &q, 5, VectorSearchOptions::new())
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
            .vector_hits("emb", &q, 0, VectorSearchOptions::new())
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
            .vector_hits("emb", &q, 5, VectorSearchOptions::new())
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
            .vector_hits("emb", &q, 7, VectorSearchOptions::new())
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
        let hits = st.reader().vector_hits("emb", &q, 10, opts).expect("query");

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
            .vector_hits("emb", &q, 24, VectorSearchOptions::new())
            .expect("query");
        let superfile_uris: std::collections::HashSet<_> =
            hits.iter().map(|h| h.superfile).collect();
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
        let oracle_globals: std::collections::HashSet<u32> =
            oracle_hits.iter().map(|(d, _)| *d).collect();
        assert_eq!(oracle_globals, [0u32, 16].iter().copied().collect());

        let st_reader = st.reader();
        let st_hits = st_reader
            .vector_hits("emb", &q, 2, opts)
            .expect("supertable query");
        let manifest = st_reader.manifest();
        let st_globals: std::collections::HashSet<u32> = st_hits
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
            .vector_hits("nope", &q, 5, VectorSearchOptions::new())
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

    use crate::storage::{LocalFsStorageProvider, StorageProvider};
    use crate::supertable::SuperfileUri;
    use crate::supertable::manifest::SuperfileEntry;
    use crate::supertable::query::SuperfileHit;
    use crate::supertable::tombstones::SidecarCache;
    use crate::supertable::tombstones::cache::DEFAULT_REFRESH_TTL;
    use crate::supertable::wal::WalStore;
    use crate::supertable::wal::tombstones_codec::TombstonesSidecar;
    use tempfile::TempDir;
    use uuid::Uuid;

    fn synthetic_entry(superfile_id: Uuid) -> SuperfileEntry {
        SuperfileEntry {
            superfile_id,
            uri: SuperfileUri(superfile_id),
            n_docs: 100,
            id_min: 0,
            id_max: 99,
            scalar_stats: crate::supertable::manifest::ScalarStatsTable::default(),
            fts_summary: std::collections::HashMap::new(),
            vector_summary: std::collections::HashMap::new(),
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
}
