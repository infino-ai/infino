//! Vector kNN fan-out on [`Supertable`](super::super::Supertable).
//!
//! ## Public API
//!
//! ```ignore
//! let opts = VectorSearchOptions::new();
//! let hits: Vec<SuperfileHit> =
//!     supertable.vector_search("emb", &query_vec, 10, opts)?;
//! ```
//!
//! Returns [`SuperfileHit`]s sorted by distance *ascending* —
//! smaller distance is closer (cosine: `1 - dot`, L2-sq: squared
//! distance). `local_doc_id` is the row offset within `segment`;
//! doc-id space is local to a segment in v1.
//!
//! ## Strategy
//!
//! Internally pins a snapshot reader and drives the async
//! kernel to completion via the sync→async bridge. The reader
//! holds a pinned `Arc<Manifest>`; for each visible segment we:
//!
//!   1. Fetch the segment's `SuperfileReader` from the store.
//!   2. Delegate to `SuperfileReader::vector_search`
//!      (cluster-aware IVF + 1-bit RaBitQ shortlist + full-precision
//!      rerank, all inside one segment).
//!   3. Tag each `(local_doc_id, distance)` with the segment URI.
//!   4. Concatenate across superfiles and global-top-k by distance.
//!
//! Unlike BM25, vector distances are inherently comparable across
//! superfiles — both cosine and L2-sq are functions of the query
//! and the per-doc vector only, not of segment-scoped statistics.
//! So the per-segment top-k → concatenate → global top-k pattern
//! recovers exact recall (modulo each per-segment IVF's nprobe-
//! driven recall tradeoff, which is identical to the single-
//! superfile case).
//!
//! Fan-out uses centroid pruning:
//!
//!   1. **Score & sort** — compute `distance(query, centroid)`
//!      for each segment (SIMD-accelerated: AVX-512 / AVX2 /
//!      NEON). Derive a lower bound per segment:
//!      `max(0, centroid_dist − radius)`. Sort ascending.
//!      This is free — centroids are manifest metadata, no
//!      S3 GETs.
//!   2. **Search closest** — search the top `k*2` (min 3)
//!      segments in parallel (`tokio::spawn` per segment).
//!      Merge results via bounded heap.
//!
//! Every skipped segment is a batch of GET requests the
//! object-store-native engine never issues. For cold queries
//! this is the difference between seconds and milliseconds.

use std::sync::Arc;

use rayon::prelude::*;

use crate::superfile::SuperfileReader;
pub use crate::superfile::reader::VectorSearchOptions;
use crate::supertable::error::QueryError;
use crate::supertable::handle::{Supertable, SupertableReader};
use crate::supertable::manifest::SuperfileEntry;
use crate::supertable::reader_cache::SuperfileReaderCache;

use super::SuperfileHit;

impl SupertableReader {
    /// Single-column vector kNN search across the pinned
    /// manifest's superfiles.
    ///
    /// Returns up to `k` lowest-distance hits, sorted ascending.
    /// `query` must match the column's declared `dim`.
    ///
    /// `options` (see [`VectorSearchOptions`]) controls per-
    /// segment recall-vs-latency knobs (`nprobe`, `rerank_mult`).
    /// Defaults recover ≥0.9 recall@10 on typical IVF setups.
    ///
    /// Empty supertable (no superfiles) and `k == 0` short-circuit
    /// to an empty `Vec`.
    ///
    /// `pub(crate)` async kernel — the public surface is the sync
    /// [`Supertable::vector_search`], which drives this via the
    /// sync→async bridge after applying the read-consistency policy.
    pub(crate) async fn vector_search(
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
        let store = Arc::clone(&manifest.options.store);
        let disk_cache = manifest.options.disk_cache.as_ref().map(Arc::clone);
        let storage = manifest.options.storage.as_ref().map(Arc::clone);
        let tombstone_cache = self.tombstone_cache.clone();
        let now = std::time::Instant::now();

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
            None => crate::supertable::query::hierarchical_iter::fallback_to_flat_segments(
                manifest.as_ref(),
            ),
        };
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }

        // Fan out to every kept segment. The part-level pruner
        // above (`prune_parts_for_vector`) already dropped parts
        // that cannot contain the query's neighbors; within the
        // kept set we must search ALL segments.
        //
        // NB: a previous version skipped segments here via a
        // centroid-lower-bound cutoff (`best_lb * 2.0`). That was
        // NOT a correctness-preserving bound — on a corpus where
        // neighbors are spread across segments it silently dropped
        // true top-k results (recall collapsed to ~0.5). Any
        // segment-level skip must compare a real lower bound
        // against the running k-th best distance; a static
        // heuristic cutoff is wrong. Correct first, fast second.
        //
        // Fan-out is split by concern, mirroring the SQL resolve
        // path (`query::exec::common`):
        //   1. **Open** every kept segment's reader **concurrently**
        //      on the tokio runtime — opens are async I/O (in-memory
        //      cache hits / disk-cache cold fetches), so overlapping
        //      them wide is the right model and warm opens cost
        //      microseconds.
        //   2. **Search** every segment **in parallel** on
        //      `options.reader_pool` (rayon). The per-segment kNN
        //      search is CPU-bound over resident bytes on the hot
        //      path (centroid + 1-bit code scoring, then rerank);
        //      driving it on the reader pool keeps ALL per-segment
        //      CPU in one work-stealing pool instead of
        //      oversubscribing the tokio workers with the inner
        //      `par_iter` on the global pool. Bridged back to the
        //      async caller via a oneshot so no runtime worker is
        //      blocked. Cold lazy misses inside the search bridge
        //      through `Source::get_range` internally.
        let fanout_t0 = std::time::Instant::now();
        let open_t0 = std::time::Instant::now();
        let opened: Vec<Arc<SuperfileReader>> = futures::future::try_join_all(
            superfiles
                .iter()
                .map(|entry| open_reader(&store, disk_cache.as_ref(), storage.as_ref(), entry)),
        )
        .await?;
        let open_us = open_t0.elapsed().as_micros() as u64;

        // Pure-CPU phase on the reader pool. `into_par_iter`'s ordered
        // collect preserves segment order, so `per_segment[i]` lines up
        // with `superfiles[i]` for the post-phase tombstone filter.
        let pool = Arc::clone(&manifest.options.reader_pool);
        let column_owned = column.to_owned();
        let query_owned = query.to_vec();
        let timing_enabled = fanout_timing_enabled();
        let inputs: Vec<Arc<SuperfileReader>> = opened;
        let (tx, rx) = tokio::sync::oneshot::channel();
        pool.spawn(move || {
            let result: Result<Vec<(Vec<(u32, f32)>, u64)>, QueryError> = inputs
                .into_par_iter()
                .map(|reader| {
                    let search_t0 = std::time::Instant::now();
                    let hits = reader
                        .vector_search(&column_owned, &query_owned, k, options)
                        .map_err(|e| QueryError::Parquet(e.to_string()))?;
                    Ok((hits, search_t0.elapsed().as_micros() as u64))
                })
                .collect();
            let _ = tx.send(result);
        });
        let per_segment = rx.await.map_err(|_| {
            QueryError::Store("vector fan-out: reader pool dropped result".into())
        })??;

        // Tombstone filtering stays on the tokio side: a cache miss
        // refreshes the sidecar from storage (blocking I/O via the
        // sync→async bridge), which does not belong on the CPU pool.
        // On the hot path every `bitmap_for` is a cache hit, so this
        // is a cheap serial pass over the already-scored segments.
        let mut results: Vec<Vec<SuperfileHit>> = Vec::with_capacity(per_segment.len());
        let mut searches: Vec<u64> = Vec::with_capacity(per_segment.len());
        for (entry, (hits, search_us)) in superfiles.iter().zip(per_segment) {
            let mut tagged = tag_hits(entry.as_ref(), hits);
            apply_tombstone_filter(tombstone_cache.as_ref(), entry.as_ref(), &mut tagged, now)?;
            results.push(tagged);
            searches.push(search_us);
        }

        if timing_enabled {
            summarize_fanout_timing(open_us, &searches, fanout_t0.elapsed().as_micros() as u64);
        }

        Ok(top_k_ascending(results, k))
    }
}

impl Supertable {
    /// Single-column vector kNN search over the current snapshot.
    ///
    /// Pins a reader at call entry, applies the read-consistency
    /// policy, and drives the internal async kernel to completion
    /// via the sync→async bridge ([`Supertable::block_on_query`]).
    /// Returns up to `k` hits sorted by distance *ascending*.
    pub fn vector_search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.ensure_fresh();
        let reader = self.reader();
        self.block_on_query(reader.vector_search(column, query, k, options))
    }
}

/// `INFINO_FANOUT_TIMING=1` enables per-segment fan-out timing.
fn fanout_timing_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EN.get_or_init(|| {
        std::env::var("INFINO_FANOUT_TIMING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Print the aggregate open wall + per-segment search wall
/// distribution next to the total fan-out wall. With the
/// two-phase fan-out (concurrent opens on tokio, then parallel
/// search on the reader pool) the key signal is `total /
/// (open + max_search)`: ~1.0 means the search phase is fully
/// parallel (cost is the slowest segment); ≫1.0 means the reader
/// pool is serializing the per-segment CPU.
fn summarize_fanout_timing(open_us: u64, search_us: &[u64], total_us: u64) {
    let n = search_us.len();
    if n == 0 {
        eprintln!("[fanout-timing] no segments recorded; total={total_us}us");
        return;
    }
    let mut searches: Vec<u64> = search_us.to_vec();
    searches.sort_unstable();
    let pct = |v: &[u64], p: f64| -> u64 {
        let idx = (((v.len() - 1) as f64) * p).round() as usize;
        v[idx]
    };
    let max_search = searches.last().copied().unwrap_or(0);
    let ratio = total_us as f64 / (open_us + max_search).max(1) as f64;
    eprintln!(
        "[fanout-timing] n_seg={n} total={:.1}ms | opens(concurrent)={:.1}ms | \
         per-seg search: p50={:.1} p90={:.1} max={:.1}ms | \
         total/(open+max_search)={:.1}x  ({})",
        total_us as f64 / 1000.0,
        open_us as f64 / 1000.0,
        pct(&searches, 0.5) as f64 / 1000.0,
        pct(&searches, 0.9) as f64 / 1000.0,
        max_search as f64 / 1000.0,
        ratio,
        if ratio <= 1.5 {
            "PARALLEL: cost is open + slowest-segment search"
        } else {
            "SERIALIZED: reader-pool search is being serialized"
        },
    );
}

async fn open_reader(
    store: &Arc<dyn SuperfileReaderCache>,
    disk_cache: Option<&Arc<crate::supertable::reader_cache::DiskCacheStore>>,
    storage: Option<&Arc<dyn crate::storage::StorageProvider>>,
    entry: &SuperfileEntry,
) -> Result<Arc<SuperfileReader>, QueryError> {
    crate::supertable::query::superfile_reader::superfile_reader(
        store,
        disk_cache,
        storage,
        &entry.uri,
        entry.subsection_offsets.as_ref(),
    )
    .await
    .map_err(|e| QueryError::Store(e.to_string()))
}

fn tag_hits(entry: &SuperfileEntry, hits: Vec<(u32, f32)>) -> Vec<SuperfileHit> {
    hits.into_iter()
        .map(|(local_doc_id, score)| SuperfileHit {
            segment: entry.uri,
            local_doc_id,
            score,
        })
        .collect()
}

/// Drop tombstoned `local_doc_id`s from one superfile's vector hits.
/// Same shape + perf properties as the FTS path's filter — see
/// `query::fts::apply_tombstone_filter` for the design rationale.
fn apply_tombstone_filter(
    cache: Option<&Arc<crate::supertable::tombstones::SidecarCache>>,
    entry: &SuperfileEntry,
    hits: &mut Vec<SuperfileHit>,
    now: std::time::Instant,
) -> Result<(), QueryError> {
    let Some(cache) = cache else {
        return Ok(());
    };
    let bitmap = cache
        .bitmap_for(entry.superfile_id, now)
        .map_err(|e| QueryError::Store(format!("tombstone cache: {e}")))?;
    if bitmap.is_empty() {
        return Ok(());
    }
    hits.retain(|h| !bitmap.contains(h.local_doc_id));
    Ok(())
}

/// Merge per-segment hits and return the top-k by *ascending*
/// distance (smallest = closest). Uses a max-heap of size k so
/// we never sort more than k elements — O(S·k·log k) instead of
/// O(S·k·log(S·k)) for the full-sort approach.
fn top_k_ascending(per_segment: Vec<Vec<SuperfileHit>>, k: usize) -> Vec<SuperfileHit> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

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
    for hit in per_segment.into_iter().flatten() {
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

    fn options_one_segment_per_commit(dim: usize) -> SupertableOptions {
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

    #[tokio::test]
    async fn vector_search_empty_supertable_returns_empty() {
        let st = Supertable::create(options_one_segment_per_commit(16)).expect("create");
        let r = st.reader();
        let q = vec![0.1f32; 16];
        let hits = r
            .vector_search("emb", &q, 5, VectorSearchOptions::new())
            .await
            .expect("query");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn vector_search_k_zero_short_circuits() {
        let st = Supertable::create(options_one_segment_per_commit(16)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        w.append(&build_vector_batch(0, 8, 16, schema)).expect("a");
        w.commit().expect("c");
        let r = st.reader();
        let q = vec![0.1f32; 16];
        let hits = r
            .vector_search("emb", &q, 0, VectorSearchOptions::new())
            .await
            .expect("query");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn vector_search_returns_ascending_distance_order() {
        let dim = 16;
        let st = Supertable::create(options_one_segment_per_commit(dim)).expect("create");
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
            .vector_search("emb", &q, 5, VectorSearchOptions::new())
            .await
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

    #[tokio::test]
    async fn vector_search_top_k_caps_at_k() {
        let dim = 16;
        let st = Supertable::create(options_one_segment_per_commit(dim)).expect("create");
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
            .vector_search("emb", &q, 7, VectorSearchOptions::new())
            .await
            .expect("query");
        assert_eq!(hits.len(), 7);
    }

    #[tokio::test]
    async fn vector_search_carries_segment_uris_for_multi_segment_results() {
        let dim = 16;
        let st = Supertable::create(options_one_segment_per_commit(dim)).expect("create");
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
            .vector_search("emb", &q, 24, VectorSearchOptions::new())
            .await
            .expect("query");
        let segment_uris: std::collections::HashSet<_> = hits.iter().map(|h| h.segment).collect();
        // All three superfiles should contribute (high k pulls from
        // each).
        assert_eq!(segment_uris.len(), 3);
    }

    #[tokio::test]
    async fn vector_search_oracle_top_k_set_matches_single_superfile() {
        // Vector distances are segment-independent — cosine /
        // L2-sq are functions of the query + per-doc vector only.
        // So the per-segment-top-k → global-top-k pattern recovers
        // the same set as a single-superfile search, modulo each
        // IVF's nprobe-driven recall (we use a high-recall config).
        let dim = 16;
        let st = Supertable::create(options_one_segment_per_commit(dim)).expect("create");
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

        let oracle_hits = oracle
            .vector_search("emb", &q, 2, opts)
            .expect("oracle query");
        let oracle_globals: std::collections::HashSet<u32> =
            oracle_hits.iter().map(|(d, _)| *d).collect();
        assert_eq!(oracle_globals, [0u32, 16].iter().copied().collect());

        let st_reader = st.reader();
        let st_hits = st_reader
            .vector_search("emb", &q, 2, opts)
            .await
            .expect("supertable query");
        let manifest = st_reader.manifest();
        let st_globals: std::collections::HashSet<u32> = st_hits
            .iter()
            .map(|h| {
                let seg_idx = manifest
                    .superfiles
                    .iter()
                    .position(|e| e.uri == h.segment)
                    .expect("segment in manifest");
                (seg_idx as u32) * 8 + h.local_doc_id
            })
            .collect();
        assert_eq!(st_hits.len(), oracle_hits.len());
        assert_eq!(st_globals, oracle_globals);
    }

    #[tokio::test]
    async fn vector_search_unknown_column_errors() {
        let dim = 16;
        let st = Supertable::create(options_one_segment_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        w.append(&build_vector_batch(0, 8, dim, schema)).expect("a");
        w.commit().expect("c");
        let r = st.reader();
        let q = vec![0.1f32; dim];
        let err = r
            .vector_search("nope", &q, 5, VectorSearchOptions::new())
            .await
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
                segment: entry.uri,
                local_doc_id: d,
                score: d as f32,
            })
            .collect();

        super::apply_tombstone_filter(Some(&cache), &entry, &mut hits, std::time::Instant::now())
            .expect("filter");

        let remaining: Vec<u32> = hits.iter().map(|h| h.local_doc_id).collect();
        assert_eq!(remaining, vec![0u32, 2, 4, 6, 7]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_tombstone_filter_is_no_op_without_cache() {
        let entry = synthetic_entry(Uuid::from_u128(0xABCD));
        let mut hits: Vec<SuperfileHit> = (0..4u32)
            .map(|d| SuperfileHit {
                segment: entry.uri,
                local_doc_id: d,
                score: 0.0,
            })
            .collect();
        let original = hits.clone();
        super::apply_tombstone_filter(None, &entry, &mut hits, std::time::Instant::now())
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
                segment: entry.uri,
                local_doc_id: d,
                score: 0.0,
            })
            .collect();
        let original = hits.clone();
        super::apply_tombstone_filter(Some(&cache), &entry, &mut hits, std::time::Instant::now())
            .expect("filter");
        assert_eq!(hits, original);
    }
}
