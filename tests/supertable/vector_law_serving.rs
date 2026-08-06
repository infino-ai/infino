// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The stamped rerank law must actually SERVE — regression guard for the
//! scope bug fixed in #520: the law's `options` rebind once died inside
//! the admit arm's brace, and every law-stamped table silently served
//! the `rerank_mult = 256` constant. Recall assertions cannot catch a
//! recurrence (the constant budget only ADDS survivors — recall stays
//! equal-or-better; only latency regresses), so this test asserts the
//! EFFECTIVE budget the pooled warm arm hands to the global shortlist,
//! via the `served_shortlist_probe` test-helpers hook.
//!
//! The same probe pins the floor scoping from the same PR: law-served
//! defaults run floor-free; an explicit caller `nprobe` arms the floor at
//! the full per-cell budget (`k x rerank_mult`, so per-cell depth never
//! shrinks as the caller widens — #537).

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    VectorSearchOptions,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
    test_helpers::{default_tokenizer, default_vector_config, served_shortlist_probe},
};
use tempfile::TempDir;

/// Matches `default_vector_config`'s dimension.
const DIM: usize = 16;
/// Random-rotation seed for the fixture's vector index.
const VECTOR_ROT_SEED: u64 = 41;
/// Corpus size: small enough that any measured rerank budget is far below
/// the constant (`k * 256 = 2048`), so law-vs-constant separates cleanly.
const N_ROWS: usize = 64;
/// Query k for every probe assertion below.
const K: usize = 8;
/// A caller-set rerank multiplier used to pin override precedence with an
/// exact expected budget (`K * CALLER_RM`, replica overhead 0 by default).
const CALLER_RM: usize = 3;

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

/// Deterministic non-degenerate vector for row `i`.
fn row_vec(i: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| ((i * 31 + d * 17 + 7) % 97) as f32 / 97.0 + 0.05)
        .collect()
}

fn corpus_batch(schema: Arc<Schema>) -> RecordBatch {
    let mut flat = Vec::<f32>::with_capacity(N_ROWS * DIM);
    let mut titles = Vec::with_capacity(N_ROWS);
    for i in 0..N_ROWS {
        flat.extend(row_vec(i));
        titles.push(format!("row{i:03}"));
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn law_stamped_table_serves_the_law_not_the_constant() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st =
        Supertable::create(vector_options().with_storage(Arc::clone(&storage))).expect("create");

    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    w.append(&corpus_batch(schema)).expect("append");
    w.commit().expect("commit");
    drop(w);
    st.drain_vectors_to_cells_sync().expect("drain");

    let query = row_vec(3);

    // 1) Law-served default: the effective budget must be the measured
    //    law, not the constant. Any re-shadowing regression makes EVERY
    //    default record `K * 256 = 2048`, so the contains-check below
    //    fails even under concurrent recorders. Floor must be 0 —
    //    defaults run floor-free by design (#520).
    served_shortlist_probe::drain();
    st.vector_search("emb", &query, K, VectorSearchOptions::new(), None, None)
        .expect("default search");
    let recs = served_shortlist_probe::drain();
    assert!(
        recs.iter()
            .any(|&(limit, floor)| limit > 0 && limit < K * 256 && floor == 0),
        "default must serve a law budget below the K*256 constant with no \
         floor; recorded: {recs:?}"
    );

    // 2) Caller override wins with an exact budget: K * CALLER_RM.
    st.vector_search(
        "emb",
        &query,
        K,
        VectorSearchOptions::new().with_rerank_mult(CALLER_RM),
        None,
        None,
    )
    .expect("caller-rm search");
    let recs = served_shortlist_probe::drain();
    assert!(
        recs.contains(&(K * CALLER_RM, 0)),
        "caller rerank_mult must override the law exactly; recorded: {recs:?}"
    );

    // 3) Explicit caller nprobe re-arms the per-cell floor — at the full
    //    per-cell budget (#537: k x rerank_mult, width-independent), so
    //    widening the probe adds cells at constant depth instead of
    //    diluting a shared pool (the >=5M recall inversion). The floor
    //    must EQUAL the pooled limit (replica overhead is 0 here, so both
    //    are k x rerank_mult): per-cell depth == full budget is exactly
    //    #538's behavior, and a revert to the old `floor = k` fails this
    //    (the law-stamped budget on this fixture is above k).
    st.vector_search(
        "emb",
        &query,
        K,
        VectorSearchOptions::new().with_nprobe(2),
        None,
        None,
    )
    .expect("pinned-nprobe search");
    let recs = served_shortlist_probe::drain();
    assert!(
        recs.iter()
            .any(|&(limit, floor)| floor == limit && floor > K),
        "explicit nprobe must arm the per-cell floor at the full \
         k x rerank_mult budget (== the pooled limit, > k on this \
         fixture); recorded: {recs:?}"
    );
}
