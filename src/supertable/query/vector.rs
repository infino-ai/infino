//! Vector kNN fan-out method on
//! [`super::super::SupertableReader`].
//!
//! ## Public API
//!
//! ```ignore
//! let reader = supertable.reader();
//! let opts = VectorSearchOptions::new();
//! let hits: Vec<SuperfileHit> =
//!     reader.vector_search("emb", &query_vec, 10, opts)?;
//! ```
//!
//! Returns [`SuperfileHit`]s sorted by distance *ascending* —
//! smaller distance is closer (cosine: `1 - dot`, L2-sq: squared
//! distance). `local_doc_id` is the row offset within `segment`;
//! doc-id space is local to a segment in v1.
//!
//! ## Strategy
//!
//! Vector search is a method on the reader. The reader
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
//! Rayon fan-out runs on `options.reader_pool`. No skip pruning
//! is wired into this path in v1: every segment is searched.
//! `query::skip::vector_centroid_skip` is implemented as
//! all-keep, and `superfiles_sorted_by_centroid_distance` is
//! available as a fan-out ordering hint.
//!
//! TODO: wire incremental cutoff-driven skip — track the
//! running `k`-th-best distance during fan-out and drop superfiles
//! whose `(centroid, radius)` summary proves they can't reach
//! the cutoff. Cluster-aware cutoff pruning needs the running
//! top-k distance from the in-flight fan-out, which only
//! becomes available once at least one segment has been
//! searched — so it has to be paired with an incremental
//! top-k merge that produces the cutoff and a skip pre-pass
//! that consumes it.

use std::sync::Arc;

use rayon::prelude::*;

use crate::superfile::SuperfileReader;
pub use crate::superfile::reader::VectorSearchOptions;
use crate::supertable::error::QueryError;
use crate::supertable::handle::SupertableReader;
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
    pub fn vector_search(
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
        let pool = Arc::clone(&manifest.options.reader_pool);
        let column_owned = column.to_owned();
        let query_owned: Vec<f32> = query.to_vec();

        // M15c: hierarchical pruning. Vector list-prune
        // (`prune_parts_for_vector`) needs an upper-bound
        // distance cutoff to be useful — for top-k vector
        // search the cutoff is *dynamic* (only known after
        // some superfiles have been scanned + a running
        // top-k is in hand). A static all-load is the
        // conservative correctness-first choice for now;
        // iterative-cutoff list-prune is left for when the
        // measurement justifies its complexity. When no
        // list (in-process-only supertable), fall back to
        // the flat superfiles view directly.
        let superfiles: Vec<Arc<SuperfileEntry>> = match manifest.list.as_ref() {
            Some(list) => {
                let kept: Vec<_> = list.parts.iter().map(|p| p.part_id).collect();
                crate::supertable::query::hierarchical_iter::load_and_flatten(
                    manifest.as_ref(),
                    &kept,
                )?
            }
            None => crate::supertable::query::hierarchical_iter::fallback_to_flat_segments(
                manifest.as_ref(),
            ),
        };
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }

        let per_segment: Result<Vec<Vec<SuperfileHit>>, QueryError> = pool.install(|| {
            superfiles
                .par_iter()
                .map(|entry| {
                    let r = open_reader(&store, disk_cache.as_ref(), entry)?;
                    let hits = r
                        .vector_search(&column_owned, &query_owned, k, options)
                        .map_err(|e| QueryError::Parquet(e.to_string()))?;
                    Ok(tag_hits(entry, hits))
                })
                .collect()
        });

        Ok(top_k_ascending(per_segment?, k))
    }
}

fn open_reader(
    store: &Arc<dyn SuperfileReaderCache>,
    disk_cache: Option<&Arc<crate::supertable::reader_cache::DiskCacheStore>>,
    entry: &SuperfileEntry,
) -> Result<Arc<SuperfileReader>, QueryError> {
    crate::supertable::query::superfile_reader::superfile_reader(store, disk_cache, &entry.uri)
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

/// Concatenate per-segment hits and return the top-k by *ascending*
/// distance (smallest distance = closest neighbor). Defensively
/// treats NaN as worst (sorted to the bottom) — distance kernels
/// shouldn't emit NaN given finite inputs, but cosine is undefined
/// at zero-norm so NaN-defense matters more here than for BM25.
fn top_k_ascending(per_segment: Vec<Vec<SuperfileHit>>, k: usize) -> Vec<SuperfileHit> {
    let mut all: Vec<SuperfileHit> = per_segment.into_iter().flatten().collect();
    // pdqsort: cross-segment top-k merge runs on every query; the
    // tie-break inside `partial_cmp` plus the (segment, local_doc_id)
    // composite already gives a total order so stability isn't needed.
    all.sort_unstable_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all.truncate(k);
    all
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
        let st = Supertable::create(options_one_segment_per_commit(16));
        let r = st.reader();
        let q = vec![0.1f32; 16];
        let hits = r
            .vector_search("emb", &q, 5, VectorSearchOptions::new())
            .expect("query");
        assert!(hits.is_empty());
    }

    #[test]
    fn vector_search_k_zero_short_circuits() {
        let st = Supertable::create(options_one_segment_per_commit(16));
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        w.append(&build_vector_batch(0, 8, 16, schema)).expect("a");
        w.commit().expect("c");
        let r = st.reader();
        let q = vec![0.1f32; 16];
        let hits = r
            .vector_search("emb", &q, 0, VectorSearchOptions::new())
            .expect("query");
        assert!(hits.is_empty());
    }

    #[test]
    fn vector_search_returns_ascending_distance_order() {
        let dim = 16;
        let st = Supertable::create(options_one_segment_per_commit(dim));
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
        let st = Supertable::create(options_one_segment_per_commit(dim));
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
            .expect("query");
        assert_eq!(hits.len(), 7);
    }

    #[test]
    fn vector_search_carries_segment_uris_for_multi_segment_results() {
        let dim = 16;
        let st = Supertable::create(options_one_segment_per_commit(dim));
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
            .expect("query");
        let segment_uris: std::collections::HashSet<_> = hits.iter().map(|h| h.segment).collect();
        // All three superfiles should contribute (high k pulls from
        // each).
        assert_eq!(segment_uris.len(), 3);
    }

    #[test]
    fn vector_search_oracle_top_k_set_matches_single_superfile() {
        // Vector distances are segment-independent — cosine /
        // L2-sq are functions of the query + per-doc vector only.
        // So the per-segment-top-k → global-top-k pattern recovers
        // the same set as a single-superfile search, modulo each
        // IVF's nprobe-driven recall (we use a high-recall config).
        let dim = 16;
        let st = Supertable::create(options_one_segment_per_commit(dim));
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
        let opts = VectorSearchOptions::new()
            .with_nprobe(4)
            .with_rerank_mult(50);

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

    #[test]
    fn vector_search_unknown_column_errors() {
        let dim = 16;
        let st = Supertable::create(options_one_segment_per_commit(dim));
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        w.append(&build_vector_batch(0, 8, dim, schema)).expect("a");
        w.commit().expect("c");
        let r = st.reader();
        let q = vec![0.1f32; dim];
        let err = r
            .vector_search("nope", &q, 5, VectorSearchOptions::new())
            .expect_err("expected error");
        assert!(matches!(err, QueryError::Parquet(_)), "got {err:?}");
    }
}
