// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! How many superfiles a vector table's commit writes. A commit with a
//! vector column packs its rows by cell, and the number of files the cells
//! pack into follows the commit's buffered bytes — one file per
//! `superfile_buffer_split_mb` of them, rounded up — capped by the writer
//! pool, exactly as a commit without vectors is cut. A small commit is one
//! superfile however wide the pool is. Every row stays readable and
//! searchable whichever count the commit took, on the collected publish
//! (no storage) and on the pipelined one (storage-backed).

#![deny(clippy::unwrap_used)]

use std::{collections::HashSet, sync::Arc};

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use infino::{
    VectorSearchOptions,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
    test_helpers::default_vector_config,
};
use rayon::ThreadPoolBuilder;
use tempfile::TempDir;

/// Dimension of the vector column; matches `default_vector_config`.
const DIM: usize = 16;
/// Random-rotation seed for the vector column.
const ROT_SEED: u64 = 29;
/// A writer pool wider than any commit here asks for, so the pool never
/// decides the count unless a test narrows it.
const WIDE_POOL: usize = 8;
/// A writer pool narrower than the sized commit's byte count asks for, so
/// the pool cap is the binding term.
const NARROW_POOL: usize = 2;
/// Rows in the small commit: a few hundred kibibytes, far under one split.
const SMALL_ROWS: usize = 512;
/// Split target for the sized commit, in MiB: small enough that a test
/// batch spans several of them.
const SIZED_SPLIT_MB: u64 = 1;
/// Rows in the sized commit. Each row buffers its 64-byte vector, its
/// 16-byte `_id`, an 8-byte title offset and about ten bytes of title, so
/// these rows buffer about 2.5 MiB: past two split targets and short of
/// three, with room either side for Arrow's allocation rounding.
const SIZED_ROWS: usize = 26_000;
/// Superfiles the sized commit's bytes call for at [`SIZED_SPLIT_MB`].
const SIZED_SHARDS: usize = 3;
/// Probes for the readback search: past the table's cell count, so the
/// search visits every cell and can return every row.
const PROBE_EVERY_CELL: usize = 4_096;
/// Xorshift seed for the vectors.
const VECTOR_SEED: u64 = 0x2545_F491_4F6C_DD1D;
/// Divisor that maps a 64-bit draw onto `[0, 1)`.
const DRAW_SCALE: f32 = (1u64 << 24) as f32;
/// Bits a draw keeps: the top 24, a full f32 mantissa.
const DRAW_SHIFT: u32 = 40;

/// The table's schema: a title and the vector column.
fn schema() -> SchemaRef {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", DataType::FixedSizeList(item, DIM as i32), false),
    ]))
}

/// A vector table on a writer pool of `pool` threads, cut at `split_mb`
/// per superfile, storage-backed under `dir` when one is given.
fn vector_table(pool: usize, split_mb: Option<u64>, dir: Option<&TempDir>) -> Supertable {
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(pool)
            .build()
            .expect("writer pool"),
    );
    let mut options = SupertableOptions::new(
        schema(),
        vec![FtsConfig::new("title")],
        vec![default_vector_config("emb", ROT_SEED)],
    )
    .expect("valid options")
    .with_writer_pool(pool);
    if let Some(split_mb) = split_mb {
        options = options.with_superfile_buffer_split_mb(split_mb);
    }
    if let Some(dir) = dir {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        options = options.with_storage(storage);
    }
    Supertable::create(options).expect("create")
}

/// `rows` rows titled `row000000`.., with pseudo-random vectors spread
/// over many cells, so a commit's cells fill every shard it is cut into.
fn batch(rows: usize) -> RecordBatch {
    let titles: Vec<String> = (0..rows).map(|i| format!("row{i:06}")).collect();
    let mut seed = VECTOR_SEED;
    let flat: Vec<f32> = (0..rows * DIM)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> DRAW_SHIFT) as f32 / DRAW_SCALE - 0.5
        })
        .collect();
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let fsl = FixedSizeListArray::try_new(
        item,
        DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )
    .expect("FSL");
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(LargeStringArray::from(titles)) as ArrayRef,
            Arc::new(fsl),
        ],
    )
    .expect("batch")
}

/// Append and commit `rows` rows, then check the commit wrote `superfiles`
/// superfiles and that a vector search reads every row back by title.
fn commit_and_check(st: &Supertable, rows: usize, superfiles: usize, label: &str) {
    let mut writer = st.writer().expect("writer");
    writer.append(&batch(rows)).expect("append");
    writer.commit().expect("commit");

    let reader = st.reader().expect("reader");
    assert_eq!(
        reader.n_superfiles(),
        superfiles,
        "{label}: superfiles the commit wrote"
    );
    assert_eq!(
        reader.n_docs_total(),
        rows as u64,
        "{label}: rows committed"
    );

    let query = vec![1.0f32; DIM];
    let hits = reader
        .vector_search(
            "emb",
            &query,
            rows,
            VectorSearchOptions::new().with_nprobe(PROBE_EVERY_CELL),
            None,
            Some(&["title"]),
        )
        .expect("vector search");
    let mut titles = HashSet::new();
    for hit in &hits {
        let column = hit
            .column_by_name("title")
            .and_then(|c| c.as_any().downcast_ref::<LargeStringArray>())
            .expect("title column");
        for row in 0..column.len() {
            titles.insert(column.value(row).to_owned());
        }
    }
    let expected: HashSet<String> = (0..rows).map(|i| format!("row{i:06}")).collect();
    assert_eq!(
        titles, expected,
        "{label}: the vector search reads back every committed row once"
    );
}

/// A commit far under one split target writes one superfile on a pool of
/// any width, rather than one per pool thread.
#[test]
fn a_small_vector_commit_writes_one_superfile() {
    let st = vector_table(WIDE_POOL, None, None);
    commit_and_check(&st, SMALL_ROWS, 1, "in memory");

    let dir = TempDir::new().expect("tempdir");
    let st = vector_table(WIDE_POOL, None, Some(&dir));
    commit_and_check(&st, SMALL_ROWS, 1, "storage-backed");
}

/// A commit of several split targets writes one superfile per target, and
/// a pool narrower than that caps it.
#[test]
fn a_vector_commit_splits_by_its_bytes_capped_by_the_pool() {
    for (pool, superfiles) in [(WIDE_POOL, SIZED_SHARDS), (NARROW_POOL, NARROW_POOL)] {
        let st = vector_table(pool, Some(SIZED_SPLIT_MB), None);
        commit_and_check(
            &st,
            SIZED_ROWS,
            superfiles,
            &format!("in memory, pool {pool}"),
        );

        let dir = TempDir::new().expect("tempdir");
        let st = vector_table(pool, Some(SIZED_SPLIT_MB), Some(&dir));
        commit_and_check(
            &st,
            SIZED_ROWS,
            superfiles,
            &format!("storage-backed, pool {pool}"),
        );
    }
}
