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
    BoolMode, CompactionSettings, OptimizeOptions, VectorSearchOptions,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::{builder::FtsConfig, vector::rerank_codec::RerankCodec},
    supertable::{Supertable, SupertableOptions},
    test_helpers::default_vector_config,
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
        vec![FtsConfig::new("title")],
        vec![default_vector_config("emb", VECTOR_ROT_SEED)],
    )
    .expect("valid options")
}

/// [`vector_options`] with the engine-default rerank codec, which catalog
/// tables get and which `optimize` builds cells from; `default_vector_config`
/// hands out the legacy Fp32 codec, which the cell build refuses.
fn production_vector_options() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(DIM), false),
    ]));
    let mut vector_config = default_vector_config("emb", VECTOR_ROT_SEED);
    vector_config.rerank_codec = RerankCodec::default();
    SupertableOptions::new(schema, vec![FtsConfig::new("title")], vec![vector_config])
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

/// The doc an update replaces after the drain.
const UPDATED: usize = 2;
/// The replacement row's title.
const UPDATED_TITLE: &str = "charlie v2";

/// One row titled `title` with a one-hot embedding at `dim_index`.
fn one_hot_row(schema: Arc<Schema>, title: &str, dim_index: usize) -> RecordBatch {
    let fsl = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM as i32,
        Arc::new(Float32Array::from(one_hot(dim_index))) as ArrayRef,
        None,
    )
    .expect("FSL");
    RecordBatch::try_new(
        schema,
        vec![Arc::new(LargeStringArray::from(vec![title])), Arc::new(fsl)],
    )
    .expect("batch")
}

/// Settings under which `optimize` merges the drained base with an update's
/// replacement superfile, so the replaced row is dropped from the user table.
fn merge_optimize_opts() -> OptimizeOptions {
    OptimizeOptions::compact(CompactionSettings {
        target_superfile_size_mb: 1,
        min_fill_percent: 1,
        min_superfiles_for_merge: 2,
        ..CompactionSettings::default()
    })
}

/// An update after the drain retires its old row in the hidden index too.
///
/// The hidden cells are not rewritten on a user mutation: a vector search
/// leaves a retired row out by the hidden index's deleted set. A delete
/// records its ids there; an update, which tombstones the rows it replaces,
/// must as well. Otherwise the old row's vector stays searchable: before a
/// compaction its hit is the replaced row, and once a compaction has dropped
/// that row from the user table, its `_id` is in no user superfile and the
/// search fails.
#[test]
fn an_update_after_the_drain_leaves_no_ghost_of_the_row_it_replaced() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(production_vector_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let schema = st.options().schema.clone();
    {
        let mut w = st.writer().expect("writer");
        w.append(&one_hot_batch(Arc::clone(&schema)))
            .expect("append");
        w.commit().expect("commit");
    }
    st.drain_vectors_to_cells_sync().expect("drain");

    let stats = st
        .update(
            col("title").eq(lit(TITLES[UPDATED])),
            &one_hot_row(Arc::clone(&schema), UPDATED_TITLE, UPDATED),
        )
        .expect("update");
    assert_eq!(stats.n_tombstoned(), 1);

    let search = |what: &str| {
        let hits = st
            .vector_search(
                "emb",
                &one_hot(UPDATED),
                TOP_K,
                VectorSearchOptions::new(),
                None,
                Some(&["_id", "title", "score"]),
            )
            .unwrap_or_else(|e| panic!("vector search {what}: {e}"));
        hit_titles(&hits)
    };

    let before = search("before the compaction");
    assert!(
        !before.iter().any(|t| t == TITLES[UPDATED]),
        "the replaced row must not come back before a compaction: {before:?}"
    );
    assert_eq!(before[0], UPDATED_TITLE);

    st.optimize(&merge_optimize_opts()).expect("optimize");
    let after = search("after the compaction");
    assert!(
        !after.iter().any(|t| t == TITLES[UPDATED]),
        "the replaced row must not come back after a compaction: {after:?}"
    );
    assert_eq!(after[0], UPDATED_TITLE);

    // A hybrid search's vector half reads the same hidden index.
    let hybrid = st
        .hybrid_search(
            "title",
            TITLES[UPDATED],
            BoolMode::Or,
            "emb",
            &one_hot(UPDATED),
            VectorSearchOptions::new(),
            TOP_K,
            Some(&["_id", "title", "score"]),
        )
        .expect("hybrid search after the compaction");
    let titles = hit_titles(&hybrid);
    assert!(
        !titles.iter().any(|t| t == TITLES[UPDATED]),
        "the replaced row must not come back in a hybrid search: {titles:?}"
    );
    assert!(titles.iter().any(|t| t == UPDATED_TITLE));
}

/// A row committed after an older reader was taken.
const LATER_TITLE: &str = "echo";
/// Its embedding's hot dimension, which no earlier row has.
const LATER_DIM: usize = 4;

/// A vector hit the query's snapshot has no row for is left out, not an
/// error.
///
/// The hidden index a search reads is the table's current one, which can
/// hold rows the query's own user-table snapshot does not: here, rows
/// committed and drained after an older reader was taken. Such a hit's
/// `_id` is in no user superfile of that snapshot. It is not a row the
/// snapshot shows, so the search leaves it out, as it leaves out a deleted
/// row, and answers with the rows it does show.
#[test]
fn a_search_through_an_older_reader_leaves_out_rows_its_snapshot_lacks() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(production_vector_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let schema = st.options().schema.clone();
    {
        let mut w = st.writer().expect("writer");
        w.append(&one_hot_batch(Arc::clone(&schema)))
            .expect("append");
        w.commit().expect("commit");
    }
    st.drain_vectors_to_cells_sync().expect("drain");
    let older = st.reader().expect("older reader");

    {
        let mut w = st.writer().expect("writer");
        w.append(&one_hot_row(Arc::clone(&schema), LATER_TITLE, LATER_DIM))
            .expect("append");
        w.commit().expect("commit");
    }
    st.drain_vectors_to_cells_sync().expect("drain");

    let hits = older
        .vector_search(
            "emb",
            &one_hot(LATER_DIM),
            TOP_K,
            VectorSearchOptions::new(),
            None,
            Some(&["_id", "title", "score"]),
        )
        .expect("a search through the older reader");
    let titles = hit_titles(&hits);
    assert!(
        !titles.iter().any(|t| t == LATER_TITLE),
        "a row the older snapshot lacks must not come back: {titles:?}"
    );
    assert_eq!(
        titles.len(),
        TITLES.len(),
        "the snapshot's own rows still answer: {titles:?}"
    );

    // A reader taken now has the row.
    let current = st
        .vector_search(
            "emb",
            &one_hot(LATER_DIM),
            TOP_K,
            VectorSearchOptions::new(),
            None,
            Some(&["_id", "title", "score"]),
        )
        .expect("a search now");
    assert_eq!(hit_titles(&current)[0], LATER_TITLE);
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
