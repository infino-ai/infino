// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Cosine ingest must normalize on EVERY path (#512).
//!
//! The fixed cosine rerank grid spans `[-1, 1]`, so a non-unit component
//! saturates at ENCODE — silently, with no error and no clamp counter (the
//! drain's transcode tripwire only observes re-encodes). Measured cost when
//! this last shipped: about −10 points of recall@10 on raw Cohere input, and
//! the failure presents as a bad experiment rather than bad data, so the
//! time is lost doubting the wrong thing.
//!
//! #520 normalized at `VectorBuilder::add`, described there as "the single
//! seam every downstream consumer reads through". True for that builder,
//! false for the engine: the supertable buffers the caller's Arrow vector
//! buffers zero-copy and the commit-time hidden-index pack views them
//! directly (`VectorColumnView` → `PackRow::Fp32` →
//! `build_merged_subsection_from_fp32`), bypassing that seam. The user table
//! stored unit rows while the hidden index — what serves vector search —
//! stored clamped ones.
//!
//! So this guard is deliberately end-to-end rather than a unit test on a
//! seam: it drives the public `append()` path with vectors whose norms are
//! far from 1 and asserts served recall matches the same corpus submitted
//! pre-normalized. Any path that skips normalization fails it, whichever
//! seam it bypassed.

#![deny(clippy::unwrap_used)]

use std::{collections::HashMap, sync::Arc};

use arrow_array::{
    ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    Metric, VectorSearchOptions,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::{FtsConfig, VectorConfig},
    supertable::{Supertable, SupertableOptions},
    test_helpers::default_tokenizer,
};
use tempfile::TempDir;

/// Fixture dimension.
const DIM: usize = 16;
/// Random-rotation seed for the fixture's vector index.
const ROT_SEED: u64 = 41;
/// Corpus size: enough rows that clamped components reorder top-k, small
/// enough to stay a fast integration test.
const N_ROWS: usize = 512;
/// Queries per arm.
const N_QUERIES: usize = 32;
/// Top-k under comparison.
const K: usize = 10;
/// Per-row magnitude. Real embedding norms are O(1..100); at this scale
/// every component lands well outside the fixed `[-1, 1]` cosine grid, so an
/// unnormalized path clamps hard rather than marginally.
const RAW_MAGNITUDE: f32 = 12.0;
/// Recall agreement required between the two ingest representations. Cosine
/// declares magnitude irrelevant, so only fp-noise tie reordering is
/// legitimate; the failure this guards is tens of points wide.
const RECALL_EQUIVALENCE_TOLERANCE: f32 = 0.02;
/// The pre-normalized arm must itself recall well, or the equivalence
/// assertion could pass on two equally broken tables.
const FIXTURE_RECALL_FLOOR: f32 = 0.90;

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

/// Cosine options on the engine's DEFAULT rerank codec — the fixed-grid one
/// that clamps. Pinning `Fp32` here would pass on broken data, since an
/// exact rerank plane has nothing to saturate.
fn cosine_options() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(DIM), false),
    ]));
    SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
            stored: true,
        }],
        vec![VectorConfig::new(
            "emb".into(),
            DIM,
            ROT_SEED,
            Metric::Cosine,
        )],
        Some(default_tokenizer()),
    )
    .expect("valid options")
}

/// Deterministic row `i`, scaled to a deliberately non-unit magnitude and
/// spread across directions so top-k is a real ranking, not a tie.
fn raw_row(i: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..DIM)
        .map(|d| {
            let a = ((i * 37 + d * 11) % 101) as f32 / 101.0 - 0.5;
            let b = ((i * 13 + d * 29) % 71) as f32 / 71.0 - 0.5;
            a + 0.5 * b
        })
        .collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let scale = RAW_MAGNITUDE / norm.max(f32::MIN_POSITIVE);
    for x in &mut v {
        *x *= scale;
    }
    v
}

fn unit(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

fn batch(schema: Arc<Schema>, rows: &[Vec<f32>]) -> RecordBatch {
    let mut flat = Vec::<f32>::with_capacity(rows.len() * DIM);
    let mut titles = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        flat.extend_from_slice(row);
        titles.push(format!("row{i:04}"));
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

/// Engine `_id` → dense corpus row, the same way the bench harness does it
/// (`corpus::engine_id_to_dense`): one ordered `SELECT _id` scan, enumerated
/// in ingest order. Ids are generated per table, so two independently built
/// tables never share them — this map is what makes their results
/// comparable against one corpus-indexed ground truth.
fn engine_id_to_dense(st: &Supertable) -> HashMap<i128, u32> {
    let batches = st
        .reader()
        .expect("reader")
        .query_sql("SELECT _id FROM supertable ORDER BY _id")
        .expect("SELECT _id");
    let mut ids = Vec::with_capacity(N_ROWS);
    for b in &batches {
        let col = b
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id is Decimal128");
        ids.extend((0..col.len()).map(|i| col.value(i)));
    }
    assert_eq!(ids.len(), N_ROWS, "id scan must cover the corpus");
    ids.into_iter()
        .enumerate()
        .map(|(dense, id)| (id, dense as u32))
        .collect()
}

/// Served hits as dense corpus rows, in rank order.
fn hit_rows(batches: &[RecordBatch], id_to_dense: &HashMap<i128, u32>) -> Vec<u32> {
    let mut rows = Vec::new();
    for b in batches {
        let idx = b.schema().index_of("_id").expect("_id projected");
        let col = b
            .column(idx)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id is Decimal128");
        for i in 0..b.num_rows() {
            rows.push(
                *id_to_dense
                    .get(&col.value(i))
                    .expect("served _id present in the id map"),
            );
        }
    }
    rows
}

/// Brute-force cosine top-K over the unit corpus, as dense rows. Cosine
/// ignores magnitude, so this ground truth is identical for both arms —
/// exactly the property the engine must preserve.
fn brute_force_rows(pre: &[Vec<f32>], query: &[f32]) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = pre
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let dot: f32 = row.iter().zip(query).map(|(a, b)| a * b).sum();
            (i as u32, dot)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.into_iter().take(K).map(|(i, _)| i).collect()
}

/// Build a drained table from `rows`, then serve `queries` and return each
/// query's hits as dense corpus rows.
fn served_rows(dir: &TempDir, rows: &[Vec<f32>], queries: &[Vec<f32>]) -> Vec<Vec<u32>> {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(cosine_options().with_storage(storage)).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    w.append(&batch(schema, rows)).expect("append");
    w.commit().expect("commit");
    drop(w);
    st.drain_vectors_to_cells_sync().expect("drain");

    let id_to_dense = engine_id_to_dense(&st);
    queries
        .iter()
        .map(|q| {
            let batches = st
                .vector_search("emb", q, K, VectorSearchOptions::new(), None, None)
                .expect("search");
            hit_rows(&batches, &id_to_dense)
        })
        .collect()
}

fn mean_recall(hits: &[Vec<u32>], pre: &[Vec<f32>], queries: &[Vec<f32>]) -> f32 {
    let mut found = 0usize;
    for (qi, got) in hits.iter().enumerate() {
        let truth = brute_force_rows(pre, &queries[qi]);
        found += got.iter().filter(|r| truth.contains(r)).count();
    }
    found as f32 / (hits.len() * K) as f32
}

/// Non-unit cosine input through `append()` must serve like the same corpus
/// submitted pre-normalized. A gap means an ingest path skipped the
/// unit-normalize seam and the fixed cosine grid clamped (#512).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unnormalized_cosine_append_serves_like_prenormalized() {
    let raw: Vec<Vec<f32>> = (0..N_ROWS).map(raw_row).collect();
    let pre: Vec<Vec<f32>> = raw.iter().map(|r| unit(r)).collect();
    // Queries are unit on both arms; the corpus representation is the variable.
    let queries: Vec<Vec<f32>> = (0..N_QUERIES).map(|i| unit(&raw_row(i * 7 + 3))).collect();

    let raw_dir = TempDir::new().expect("tempdir");
    let pre_dir = TempDir::new().expect("tempdir");
    let raw_recall = mean_recall(&served_rows(&raw_dir, &raw, &queries), &pre, &queries);
    let pre_recall = mean_recall(&served_rows(&pre_dir, &pre, &queries), &pre, &queries);

    assert!(
        pre_recall >= FIXTURE_RECALL_FLOOR,
        "fixture too hard to detect clamping: pre-normalized recall@{K} \
         {pre_recall:.4} < {FIXTURE_RECALL_FLOOR}"
    );
    assert!(
        (pre_recall - raw_recall).abs() <= RECALL_EQUIVALENCE_TOLERANCE,
        "un-normalized cosine ingest recall@{K} {raw_recall:.4} vs pre-normalized \
         {pre_recall:.4} — an ingest path skipped the unit-normalize seam and the \
         fixed cosine grid clamped the out-of-range components (#512)"
    );
}
