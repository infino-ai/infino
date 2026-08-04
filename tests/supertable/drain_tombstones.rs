// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Tombstoned vectors must be excluded from the hidden per-cell index at
//! DRAIN time — build-time exclusion, not query-time filtering.
//!
//! The drain materializes each user superfile's IVF rows and drops rows
//! whose `_id` is tombstoned before routing them into cells. A leak here
//! bakes deleted vectors into the derived index permanently: they occupy
//! cell space and rerank work forever, and any consumer trusting the
//! hidden index's membership (the drained ranges say this data is
//! routed) would resurface deleted rows. Asserting on the hidden table's
//! own document count pins the exclusion at the build layer, where
//! query-time tombstone filtering can't mask a regression.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{col, lit};
use infino::{
    VectorSearchOptions,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
    test_helpers::{default_tokenizer, default_vector_config},
};
use tempfile::TempDir;

/// Matches `default_vector_config`'s dimension.
const DIM: usize = 16;
/// Random-rotation seed for the fixture's vector index.
const VECTOR_ROT_SEED: u64 = 37;
/// Top-K above corpus size, so ranking decides the assertions.
const TOP_K: usize = 8;
/// The corpus; doc `i` carries a one-hot embedding at dim `i`.
const TITLES: &[&str] = &["alpha", "bravo", "charlie", "delta"];
/// The doc deleted before the drain.
const DELETED: usize = 1;
/// Rows the delete predicate tombstones: exactly one.
const DELETED_COUNT: usize = 1;

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

fn vector_options() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(DIM), false),
    ]));
    SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![default_vector_config("emb", VECTOR_ROT_SEED)],
        Some(default_tokenizer()),
    )
    .expect("valid options")
}

fn one_hot_batch(schema: Arc<Schema>) -> RecordBatch {
    let n = TITLES.len();
    let mut flat = Vec::<f32>::with_capacity(n * DIM);
    for i in 0..n {
        for d in 0..DIM {
            flat.push(if d == i % DIM { 1.0 } else { 0.0 });
        }
    }
    let fsl = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )
    .expect("FSL");
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(LargeStringArray::from(TITLES.to_vec())),
            Arc::new(fsl),
        ],
    )
    .expect("batch")
}

fn one_hot(dim_index: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[dim_index % DIM] = 1.0;
    v
}

/// Titles across all returned batches, in hit order.
fn hit_titles(batches: &[RecordBatch]) -> Vec<String> {
    let mut titles = Vec::new();
    for batch in batches {
        let idx = batch.schema().index_of("title").expect("title projected");
        let column = batch
            .column(idx)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title column")
            .clone();
        for row in 0..batch.num_rows() {
            titles.push(column.value(row).to_string());
        }
    }
    titles
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn drain_excludes_tombstoned_vectors_from_hidden_cells() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st =
        Supertable::create(vector_options().with_storage(Arc::clone(&storage))).expect("create");

    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    w.append(&one_hot_batch(schema)).expect("append");
    w.commit().expect("commit");

    // Tombstone one row BEFORE the drain runs.
    let pending = w
        .delete(col("title").eq(lit(TITLES[DELETED])))
        .expect("delete");
    assert_eq!(pending.matched, 1);
    let outcome = w.commit().expect("commit delete");
    assert_eq!(outcome.outcomes[0].n_tombstoned(), 1);
    drop(w);

    st.drain_vectors_to_cells_sync().expect("drain");

    // Build-time exclusion: the hidden index's own membership holds only
    // the survivors — the deleted vector never entered a cell.
    let hidden = st.vector_index_table().expect("hidden index table");
    assert_eq!(
        hidden.reader().expect("hidden reader").n_docs_total(),
        (TITLES.len() - DELETED_COUNT) as u64,
        "the tombstoned vector must not be routed into the hidden cells"
    );

    // Post-drain search under the deleted row's own embedding never
    // surfaces it, and a survivor still ranks itself first.
    let ghost = st
        .vector_search(
            "emb",
            &one_hot(DELETED),
            TOP_K,
            VectorSearchOptions::new(),
            None,
            Some(&["_id", "title", "score"]),
        )
        .expect("search deleted embedding");
    assert!(
        !hit_titles(&ghost).iter().any(|t| t == TITLES[DELETED]),
        "deleted row must not resurface post-drain"
    );

    let survivor = st
        .vector_search(
            "emb",
            &one_hot(0),
            TOP_K,
            VectorSearchOptions::new(),
            None,
            Some(&["_id", "title", "score"]),
        )
        .expect("search survivor embedding");
    assert_eq!(hit_titles(&survivor)[0], TITLES[0]);
}

/// Rows per superfile in the multi-superfile pairing fixture.
const PAIRING_ROWS_PER_BATCH: usize = 5;
/// Superfiles (commits) in the pairing fixture.
const PAIRING_BATCHES: usize = 3;
/// Deleted rows as (batch, row-within-batch) — distinct bitmaps on
/// distinct superfiles, so any cross-pairing changes the survivor set.
const PAIRING_DELETED: &[(usize, usize)] = &[(0, 1), (2, 3)];

fn pairing_title(batch: usize, row: usize) -> String {
    format!("b{batch}r{row}")
}

fn pairing_batch(schema: Arc<Schema>, batch: usize) -> RecordBatch {
    let mut flat = Vec::<f32>::with_capacity(PAIRING_ROWS_PER_BATCH * DIM);
    let mut titles = Vec::with_capacity(PAIRING_ROWS_PER_BATCH);
    for row in 0..PAIRING_ROWS_PER_BATCH {
        let global = batch * PAIRING_ROWS_PER_BATCH + row;
        for d in 0..DIM {
            flat.push(if d == global % DIM { 1.0 } else { 0.0 });
        }
        titles.push(pairing_title(batch, row));
    }
    let fsl = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )
    .expect("FSL");
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(LargeStringArray::from(titles)) as ArrayRef,
            Arc::new(fsl),
        ],
    )
    .expect("batch")
}

/// Multi-superfile drains must pair each superfile's rows with ITS OWN
/// tombstone bitmap. Before the `buffered` fix (#520), the batch-open
/// fan-out collected readers in I/O-completion order and zipped them
/// positionally against `batch_sources` — under reordering, one
/// superfile's rows were filtered by ANOTHER superfile's bitmap (wrong
/// rows dropped, deleted rows kept). Pinning the EXACT surviving
/// membership across three superfiles with distinct per-file tombstones
/// makes any cross-pairing visible; under `buffer_unordered` this fails
/// whenever completion order scrambles.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn drain_pairs_each_superfile_with_its_own_tombstones() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st =
        Supertable::create(vector_options().with_storage(Arc::clone(&storage))).expect("create");

    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    for batch in 0..PAIRING_BATCHES {
        w.append(&pairing_batch(schema.clone(), batch))
            .expect("append");
        w.commit().expect("commit");
    }
    for &(batch, row) in PAIRING_DELETED {
        let pending = w
            .delete(col("title").eq(lit(pairing_title(batch, row))))
            .expect("delete");
        assert_eq!(pending.matched, 1);
        w.commit().expect("commit delete");
    }
    drop(w);

    st.drain_vectors_to_cells_sync().expect("drain");

    let n_total = PAIRING_BATCHES * PAIRING_ROWS_PER_BATCH;
    let hidden = st.vector_index_table().expect("hidden index table");
    assert_eq!(
        hidden.reader().expect("hidden reader").n_docs_total(),
        (n_total - PAIRING_DELETED.len()) as u64,
        "exactly the tombstoned rows must be excluded"
    );

    // Exact membership: every survivor still ranks itself first under its
    // own one-hot embedding; no deleted title surfaces anywhere. A
    // mispaired bitmap fails both directions at once.
    for batch in 0..PAIRING_BATCHES {
        for row in 0..PAIRING_ROWS_PER_BATCH {
            let global = batch * PAIRING_ROWS_PER_BATCH + row;
            let title = pairing_title(batch, row);
            let deleted = PAIRING_DELETED.contains(&(batch, row));
            let hits = st
                .vector_search(
                    "emb",
                    &one_hot(global),
                    TOP_K,
                    VectorSearchOptions::new(),
                    None,
                    Some(&["_id", "title", "score"]),
                )
                .expect("pairing search");
            let titles = hit_titles(&hits);
            if deleted {
                assert!(
                    !titles.contains(&title),
                    "deleted {title} resurfaced: {titles:?}"
                );
            } else {
                assert_eq!(
                    titles.first().map(String::as_str),
                    Some(title.as_str()),
                    "survivor {title} must rank itself first"
                );
            }
        }
    }
}
