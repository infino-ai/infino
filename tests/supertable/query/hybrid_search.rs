//! Public-API integration coverage for the `hybrid_search` SQL TVF.
//!
//! `hybrid_exec.rs` carries the in-crate unit tests (RRF math, single
//! segment). This file exercises the function the way a consumer does:
//! through the published `infino::supertable::Supertable::query_sql`
//! surface, over a **multi-segment** corpus so the BM25 + vector
//! retrievers fan out across segments and fuse the cross-segment
//! results. It pins three contracts the plan promises:
//!
//!   1. `SELECT *` over `hybrid_search(...)` yields the scalar schema
//!      (`_id`, `title`) plus an appended `score` column — the vector
//!      column is never exposed.
//!   2. The fused identity set is the **union** of the BM25 and vector
//!      sub-searches at the same `k` (RRF unions candidates, it does
//!      not intersect).
//!   3. A doc that ranks #1 in *both* retrievers fuses to the top, and
//!      the emitted `score` is descending (higher RRF = better).

#![deny(clippy::unwrap_used)]

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};

use infino::superfile::builder::FtsConfig;
use infino::supertable::{Supertable, SupertableOptions};
use infino::test_helpers::{default_tokenizer, default_vector_config};

/// `default_vector_config` is dim=16, cosine, n_cent=4.
const DIM: usize = 16;

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

/// Schema `[title (FTS), emb (vector)]`. The vector column is stripped
/// from the SQL scalar schema at commit, so SQL only ever sees
/// `_id` + `title`.
fn options_title_emb() -> SupertableOptions {
    let writer_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("writer pool"),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(DIM), false),
    ]));
    SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![default_vector_config("emb", 7)],
        Some(default_tokenizer()),
    )
    .expect("valid options")
    .with_writer_pool(writer_pool)
}

/// Doc `i` (within the batch) gets `titles[i]` and a one-hot embedding
/// at global dim `base_dim + i`.
fn build_batch(titles: &[&str], base_dim: usize, schema: Arc<Schema>) -> RecordBatch {
    let n = titles.len();
    let title_arr = LargeStringArray::from(titles.to_vec());
    let mut flat = Vec::<f32>::with_capacity(n * DIM);
    for i in 0..n {
        let active = base_dim + i;
        for d in 0..DIM {
            flat.push(if d == active { 1.0 } else { 0.0 });
        }
    }
    let fsl = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )
    .expect("FSL");
    RecordBatch::try_new(schema, vec![Arc::new(title_arr), Arc::new(fsl)]).expect("batch")
}

/// Two-segment corpus (docs 0-7, then 8-15). `rust` is sprinkled
/// across both segments; `async` is unique to doc 0. Doc `i`'s
/// embedding is one-hot at dim `i`, so a one-hot query at dim 0 is the
/// exact nearest neighbour of doc 0.
fn demo_two_segments() -> Supertable {
    let st = Supertable::create(options_title_emb()).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    w.append(&build_batch(
        &[
            "rust async",   // 0
            "python data",  // 1
            "java spring",  // 2
            "go rust",      // 3
            "ruby rails",   // 4
            "scala akka",   // 5
            "kotlin flow",  // 6
            "rust systems", // 7
        ],
        0,
        schema.clone(),
    ))
    .expect("append seg1");
    w.commit().expect("commit seg1");
    w.append(&build_batch(
        &[
            "swift ui",      // 8
            "rust web",      // 9
            "haskell pure",  // 10
            "elixir beam",   // 11
            "rust embedded", // 12
            "clojure lisp",  // 13
            "erlang otp",    // 14
            "rust macro",    // 15
        ],
        8,
        schema,
    ))
    .expect("append seg2");
    w.commit().expect("commit seg2");
    drop(w);
    st
}

fn csv_one_hot(active: usize) -> String {
    (0..DIM)
        .map(|d| if d == active { "1" } else { "0" })
        .collect::<Vec<_>>()
        .join(",")
}

fn id_set(batches: &[RecordBatch]) -> HashSet<i128> {
    let mut out = HashSet::new();
    for b in batches {
        let idx = b.schema().index_of("_id").expect("_id column");
        let a = b
            .column(idx)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal128 _id");
        for i in 0..a.len() {
            out.insert(a.value(i));
        }
    }
    out
}

fn col_str<'a>(batch: &'a RecordBatch, name: &str) -> &'a LargeStringArray {
    let idx = batch.schema().index_of(name).expect("column present");
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .expect("large utf8 column")
}

fn scores(batches: &[RecordBatch]) -> Vec<f32> {
    let mut out = Vec::new();
    for b in batches {
        let idx = b.schema().index_of("score").expect("score column");
        let c = b
            .column(idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("f32 score");
        out.extend((0..c.len()).map(|i| c.value(i)));
    }
    out
}

#[test]
fn hybrid_search_star_projection_exposes_scalar_schema_plus_score() {
    let st = demo_two_segments();
    let batches = st
        .query_sql(&format!(
            "SELECT * FROM hybrid_search('title', 'rust', 'emb', '{}', 5)",
            csv_one_hot(0)
        ))
        .expect("query_sql");
    let b = &batches[0];
    // Scalar schema is (_id, title); vector column `emb` is stripped.
    assert_eq!(b.num_columns(), 3, "expected _id, title, score");
    assert_eq!(b.schema().field(0).name(), "_id");
    assert_eq!(b.schema().field(1).name(), "title");
    assert_eq!(b.schema().field(2).name(), "score");
    assert!(
        b.schema().index_of("emb").is_err(),
        "vector column must never be exposed to SQL"
    );
}

#[test]
fn hybrid_search_identity_set_is_union_of_subsearches_across_segments() {
    let st = demo_two_segments();
    let qv = csv_one_hot(0);
    // k ≥ the total doc count (16) so RRF never truncates the fused
    // list — the union equality only holds when |bm25 ∪ vector| ≤ k.
    let k = 16;

    let hybrid = id_set(
        &st.query_sql(&format!(
            "SELECT _id FROM hybrid_search('title', 'rust', 'emb', '{qv}', {k})"
        ))
        .expect("hybrid query_sql"),
    );
    let bm25 = id_set(
        &st.query_sql(&format!(
            "SELECT _id FROM bm25_search('title', 'rust', {k})"
        ))
        .expect("bm25 query_sql"),
    );
    let vector = id_set(
        &st.query_sql(&format!(
            "SELECT _id FROM vector_search('emb', '{qv}', {k})"
        ))
        .expect("vector query_sql"),
    );

    // Both retrievers must actually return hits from each segment for
    // this to be a meaningful cross-segment union (guards corpus drift).
    assert!(bm25.len() >= 4, "'rust' should match across both segments");
    assert!(!vector.is_empty(), "vector retriever returned nothing");

    let expected: HashSet<i128> = bm25.union(&vector).copied().collect();
    assert_eq!(
        hybrid, expected,
        "hybrid identity set must equal bm25 ∪ vector at the same k"
    );
}

#[test]
fn hybrid_search_doc_top_in_both_retrievers_ranks_first() {
    let st = demo_two_segments();
    // `async` is unique to doc 0 (BM25 rank 1); a one-hot query at dim
    // 0 makes doc 0 the exact vector match (rank 1). Top in both →
    // highest RRF → emitted first.
    let res = st
        .query_sql(&format!(
            "SELECT title, score FROM hybrid_search('title', 'async', 'emb', '{}', 8)",
            csv_one_hot(0)
        ))
        .expect("query_sql");
    assert_eq!(
        col_str(&res[0], "title").value(0),
        "rust async",
        "doc top-ranked in both retrievers must fuse to #1"
    );

    let s = scores(&res);
    assert!(!s.is_empty(), "expected fused hits");
    for w in s.windows(2) {
        assert!(w[0] >= w[1], "fused RRF scores must be descending: {s:?}");
    }
}
