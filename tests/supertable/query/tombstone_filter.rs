// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Reader-side tombstone-filter integration tests.
//!
//! Verifies that a tombstoned row is invisible to subsequent FTS, vector, and
//! SQL queries on the same supertable handle. FTS and SQL drive the WAL
//! tombstone phase directly to pin sidecar-cache invalidation. Vector tests use
//! the public delete orchestration, which additionally publishes the hidden
//! index's stable-id delete set before querying through that index.
//!
//! 1. Real writer commit (`writer().append + commit`) publishes
//!    a superfile.
//! 2. A DELETE lands a bit in the per-superfile sidecar and, for vector
//!    tables, publishes the hidden-index delete set.
//! 3. A query runs against the same supertable handle; the
//!    tombstoned row is absent.
//!
//! The cache invalidation hook in `run_tombstone_phase` makes
//! the freshly-landed bit visible to the next query without
//! waiting for the `SidecarCache` TTL window to close — these
//! tests pin that behaviour.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use chrono::Utc;
use datafusion::prelude::{Expr, col, lit};
use infino::{
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::{
        builder::FtsConfig,
        fts::{reader::BoolMode, tokenize::Tokenizer},
    },
    supertable::{
        Supertable, SupertableOptions,
        query::vector::VectorSearchOptions,
        wal::{
            WalStore,
            pipeline::run_tombstone_phase,
            state_doc::{
                OpKind, RowId, SCHEMA_VERSION, TombstoneEntry, TombstoneOutcome, WalId, WalState,
                WalStateDoc,
            },
        },
    },
    test_helpers::{
        build_title_batch, default_supertable_options, default_tokenizer, default_vector_config,
    },
};
use tempfile::TempDir;

/// BM25 top-k for the tombstone-filtered FTS query.
const BM25_TOP_K: usize = 10;
/// Single-thread rayon pool for deterministic tombstone filtering.
const RAYON_POOL_THREADS: usize = 1;
/// Random-rotation seed for the tombstone fixture's vector index.
const VECTOR_ROT_SEED: u64 = 42;
/// Vector-search top-k for the tombstone-filtered ANN query.
const VECTOR_SEARCH_K: usize = 5;

fn build_delete_wal(target_id: i128, wal_id_value: i128) -> WalStateDoc {
    WalStateDoc {
        wal_id: WalId(wal_id_value),
        schema_version: SCHEMA_VERSION,
        op_kind: OpKind::Delete,
        state: WalState::Intent,
        created_at: Utc::now(),
        lease: None,
        predicate_repr: "integration test".into(),
        target_ids: vec![RowId(target_id)],
        new_row_count: None,
        new_row_content_hash: None,
        preallocated_superfile_id: None,
        minted_id_spans: Vec::new(),
        tombstone_progress: vec![TombstoneEntry {
            target_id: RowId(target_id),
            outcome: TombstoneOutcome::Pending,
            tombstoned_in_superfile: None,
        }],
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fts_query_excludes_tombstoned_row() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    // Three rows; all contain "alpha" so the BM25 search hits
    // every one of them. The middle row carries "bravo" too —
    // we'll tombstone it so the query drops from 3 hits to 2.
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&[
        "alpha solo",
        "alpha bravo",
        "alpha delta",
    ]))
    .expect("append");
    w.commit().expect("commit");
    drop(w);

    // Resolve the middle row's `_id`. The producer assigned ids
    // contiguously starting at `id_min`, so middle = id_min + 1.
    let manifest = st.reader().manifest().clone();
    let entry = manifest
        .get_all_superfiles()
        .first()
        .expect("at least one superfile");
    let target = entry.id_min + 1;

    // Drive the tombstone phase.
    let ws = WalStore::new(Arc::clone(&storage));
    let wal = build_delete_wal(target, 9_000_001);
    let etag = ws.create(&wal).await.expect("wal create");
    run_tombstone_phase(&st, &ws, &wal, &etag)
        .await
        .expect("tombstone phase");

    // Before tombstones the FTS query would return 3 hits;
    // post-tombstone we expect 2, and the dropped one is the
    // middle row.
    let hits = st
        .reader()
        .bm25_hits("title", "alpha", BM25_TOP_K, BoolMode::Or)
        .expect("fts");
    assert_eq!(hits.len(), 2, "tombstoned row must be excluded");
    for hit in &hits {
        assert_ne!(
            hit.local_doc_id, 1,
            "the tombstoned row's local doc_id (1) must not appear"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sql_query_excludes_tombstoned_row() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["aa", "bb", "cc", "dd"]))
        .expect("append");
    w.commit().expect("commit");
    drop(w);

    let manifest = st.reader().manifest().clone();
    let entry = manifest
        .get_all_superfiles()
        .first()
        .expect("at least one superfile");
    // Tombstone two of the four rows: id_min and id_min+2.
    let target_a = entry.id_min;
    let target_b = entry.id_min + 2;

    let ws = WalStore::new(Arc::clone(&storage));
    let wal_a = build_delete_wal(target_a, 9_000_011);
    let etag_a = ws.create(&wal_a).await.expect("wal create");
    run_tombstone_phase(&st, &ws, &wal_a, &etag_a)
        .await
        .expect("phase a");

    let wal_b = build_delete_wal(target_b, 9_000_012);
    let etag_b = ws.create(&wal_b).await.expect("wal create");
    run_tombstone_phase(&st, &ws, &wal_b, &etag_b)
        .await
        .expect("phase b");

    // `SELECT COUNT(*)` should now report 2 (4 minus the two
    // tombstoned rows).
    let batches = st
        .reader()
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("sql");
    assert_eq!(batches.len(), 1);
    let arr = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("count column");
    assert_eq!(arr.value(0), 2);

    // `SELECT title` should return only the un-tombstoned rows.
    let batches = st
        .reader()
        .query_sql("SELECT title FROM supertable ORDER BY title")
        .expect("sql");
    let titles: Vec<&str> = batches
        .iter()
        .flat_map(|b| {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .expect("title column");
            (0..col.len()).map(move |i| col.value(i))
        })
        .collect();
    assert_eq!(titles, vec!["bb", "dd"]);
}

// Deleting the row nearest a query must not shrink an unfiltered result
// — the delete-then-kNN underflow, end to end through the public delete
// orchestration (sidecar + hidden stable-id delete publication).
//  - one superfile here: the underflow bites *within* a superfile (a
//    multi-superfile merge backfills from other shards and hides it —
//    that case is the next test). A single shard also makes a hit's
//    `local_doc_id` its unambiguous row index.
//  - tombstone the query's nearest row via `Supertable::delete`.
//  - `k=1` must return the next-nearest *live* row, never empty.
// Pre-fix the filter ran on the already-truncated top-k with no
// backfill, so a deleted row in the top-k just vanished.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn vector_query_excludes_tombstoned_row() {
    use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use infino::{
        superfile::fts::tokenize::Tokenizer,
        supertable::query::vector::VectorSearchOptions,
        test_helpers::{default_tokenizer, default_vector_config},
    };

    // The bench-tier default vector config is 16-dim cosine. Stick
    // with the same dim here so the test reuses the well-trodden
    // fixture-style config without re-tuning n_cent or codec.
    const DIM: usize = 16;

    fn schema_with_vec() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    DIM as i32,
                ),
                false,
            ),
        ]))
    }

    fn vec_batch(titles: &[&str], rows: &[[f32; DIM]]) -> RecordBatch {
        let titles_arr: ArrayRef = Arc::new(LargeStringArray::from(titles.to_vec()));
        let mut flat: Vec<f32> = Vec::with_capacity(rows.len() * DIM);
        for r in rows {
            flat.extend_from_slice(r);
        }
        let values = Arc::new(Float32Array::from(flat));
        let vec_arr: ArrayRef = Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                DIM as i32,
                values,
                None,
            )
            .expect("FixedSizeList"),
        );
        RecordBatch::try_new(schema_with_vec(), vec![titles_arr, vec_arr]).expect("batch")
    }

    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("rayon"),
    );
    // Single-thread writer pool → one superfile (the why is in the test
    // note above). Distinct from `worker_threads = 2`, which is the tokio
    // runtime the WAL tombstone phase needs — not a writer pool.
    let writer_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("rayon writer"),
    );
    let tk: Arc<dyn Tokenizer> = default_tokenizer();
    let opts = SupertableOptions::new(
        schema_with_vec(),
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![default_vector_config("embedding", VECTOR_ROT_SEED)],
        Some(tk),
    )
    .expect("opts")
    .with_reader_pool(pool)
    .with_writer_pool(writer_pool)
    .with_storage(Arc::clone(&storage));

    let st = Supertable::create(opts).expect("create");

    // 16 unit-norm rows; rotating which lane is "hot" so the IVF
    // training has enough samples (n_cent=4 by default). The query
    // is closest to row 0 (lane-0 unit vector). Tombstoning row 0
    // makes row 1 (also lane-0 with a small perturbation) the
    // nearest visible neighbour.
    const N: usize = 16;
    let titles_owned: Vec<String> = (0..N).map(|i| format!("row-{i}")).collect();
    let titles: Vec<&str> = titles_owned.iter().map(|s| s.as_str()).collect();
    let mut rows: Vec<[f32; DIM]> = vec![[0.0; DIM]; N];
    for (i, row) in rows.iter_mut().enumerate() {
        row[i % DIM] = 1.0;
    }
    // Place a second row in cluster-0 so removing row 0 still
    // leaves a near neighbour in cluster 0. Slightly perturbed so
    // row 0 strictly beats it before tombstoning.
    rows[1] = [0.0; DIM];
    rows[1][0] = 0.99;
    rows[1][1] = 0.01;

    let mut w = st.writer().expect("writer");
    w.append(&vec_batch(&titles, &rows)).expect("append");
    w.commit().expect("commit");
    drop(w);

    let manifest = st.reader().manifest().clone();
    let entry = manifest
        .get_all_superfiles()
        .first()
        .expect("at least one superfile");
    let target = entry.id_min; // stable `_id` for row 0

    let stats = st.delete(col("title").eq(lit("row-0"))).expect("delete");
    assert_eq!(stats.n_tombstoned(), 1);

    // Query is the lane-0 unit vector — row 0 (now tombstoned) was its
    // exact nearest, row 1 (lane-0, lightly perturbed) is the nearest
    // live neighbour.
    let mut q = [0.0f32; DIM];
    q[0] = 1.0;

    // k=1 is the underflow repro: the single nearest is the deleted row,
    // so a no-backfill filter returns empty. It must return row 1.
    let top1 = st
        .reader()
        .vector_hits("embedding", &q, 1, VectorSearchOptions::new(), None)
        .expect("vector k=1");
    assert_eq!(
        top1.len(),
        1,
        "k=1 underflowed after deleting the nearest row"
    );
    assert_eq!(
        top1[0].stable_id,
        Some(target + 1),
        "k=1 must return the nearest live row, not the tombstoned one"
    );

    // A wider k must never surface the tombstoned row either.
    let hits = st
        .reader()
        .vector_hits(
            "embedding",
            &q,
            VECTOR_SEARCH_K,
            VectorSearchOptions::new(),
            None,
        )
        .expect("vector");
    assert!(!hits.is_empty(), "expected at least one un-tombstoned hit");
    for hit in &hits {
        assert_ne!(
            hit.stable_id,
            Some(target),
            "the tombstoned row's stable _id must not appear"
        );
    }
}

// The realistic-write-path counterpart to the single-superfile test
// above. The default writer pool shards a commit across many superfiles,
// where the underflow is normally hidden — the global merge backfills a
// deleted near-row from another shard. This pins that backfill end to
// end through the public `vector_search` surface:
//  - delete the k nearest rows to a query (spread across superfiles),
//  - re-query top-k: it must still return k rows (the next-nearest live
//    ones), and none of the deleted rows.
// Identity is the unique `title` / stable `_id`, never the per-shard
// `local_doc_id` — so the assertion is well-defined across superfiles.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn vector_query_backfills_across_superfiles_after_deletes() {
    const DIM: usize = 16;
    const N: usize = 90;
    const COMMITS: usize = 3;
    // Delete the k nearest, then ask for k again — every returned row must
    // be a live backfill.
    const K: usize = 10;

    // Deterministic spread vectors: an integer hash mixed to [0, 1), so
    // distances are distinct (no one-hot ties) and the nearest set is
    // well-defined.
    fn pseudo(i: usize, d: usize) -> f32 {
        let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(
            (d as u64)
                .wrapping_add(1)
                .wrapping_mul(0x2545_F491_4F6C_DD1D),
        );
        h ^= h >> 33;
        ((h >> 40) & 0xFFFF) as f32 / 65536.0
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    DIM as i32,
                ),
                false,
            ),
        ]))
    }

    fn batch(start: usize, end: usize) -> RecordBatch {
        let titles: ArrayRef = Arc::new(LargeStringArray::from(
            (start..end).map(|i| format!("doc {i}")).collect::<Vec<_>>(),
        ));
        let mut flat = Vec::<f32>::with_capacity((end - start) * DIM);
        for i in start..end {
            for d in 0..DIM {
                flat.push(pseudo(i, d));
            }
        }
        let emb: ArrayRef = Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                DIM as i32,
                Arc::new(Float32Array::from(flat)),
                None,
            )
            .expect("FixedSizeList"),
        );
        RecordBatch::try_new(schema(), vec![titles, emb]).expect("batch")
    }

    // Read the `title` column out of a result batch set.
    fn titles_of(batches: &[RecordBatch]) -> Vec<String> {
        batches
            .iter()
            .flat_map(|b| {
                let c = b
                    .column_by_name("title")
                    .expect("title column")
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("LargeStringArray");
                (0..c.len())
                    .map(|i| c.value(i).to_string())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let tk: Arc<dyn Tokenizer> = default_tokenizer();
    // Default writer pool — the realistic path that shards the commit.
    let st = Supertable::create(
        SupertableOptions::new(
            schema(),
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![default_vector_config("embedding", VECTOR_ROT_SEED)],
            Some(tk),
        )
        .expect("opts")
        .with_storage(Arc::clone(&storage)),
    )
    .expect("create");

    // Commit in chunks so the table spans several superfiles.
    let chunk = N.div_ceil(COMMITS);
    {
        let mut w = st.writer().expect("writer");
        for c in 0..COMMITS {
            let (start, end) = (c * chunk, ((c + 1) * chunk).min(N));
            if start >= end {
                break;
            }
            w.append(&batch(start, end)).expect("append");
            w.commit().expect("commit");
        }
    }
    assert!(
        st.reader().n_superfiles() > 1,
        "fixture must span multiple superfiles to exercise the merge backfill"
    );

    // Project `title` so the result is identified by its stable, unique
    // title rather than a per-superfile `local_doc_id`.
    let q: Vec<f32> = (0..DIM).map(|d| pseudo(0, d)).collect();
    let opts = VectorSearchOptions::new().with_nprobe(DIM);
    let proj = ["title", "score"];

    let before = st
        .reader()
        .vector_search("embedding", &q, K, opts, None, Some(&proj))
        .expect("search before");
    let deleted: Vec<String> = titles_of(&before);
    assert_eq!(deleted.len(), K, "expected k nearest before delete");

    // Delete exactly those k nearest rows.
    let preds: Vec<Expr> = deleted.iter().map(|t| lit(t.as_str())).collect();
    let stats = st
        .delete(col("title").in_list(preds, false))
        .expect("delete");
    assert_eq!(stats.n_tombstoned(), K, "all k nearest must tombstone");

    // Re-query: the merge must backfill k live rows from the surviving
    // ranks across superfiles, none of them the deleted ones.
    let after = st
        .reader()
        .vector_search("embedding", &q, K, opts, None, Some(&proj))
        .expect("search after");
    let survivors = titles_of(&after);
    assert_eq!(
        survivors.len(),
        K,
        "result underflowed instead of backfilling past tombstones"
    );
    let deleted_set: std::collections::HashSet<&String> = deleted.iter().collect();
    for title in &survivors {
        assert!(
            !deleted_set.contains(title),
            "a tombstoned row ({title}) resurfaced after delete"
        );
    }
}
