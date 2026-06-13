// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Infino reference implementation of [`SqlEngine`].
//!
//! The canonical `write` builds one in-memory `Supertable` through the
//! public writer API (`append` + `commit`) and retains the handle. Reads
//! go through the public `Supertable::query_sql`. The table carries the
//! FTS (`title`) and vector (`emb`) indexes in addition to the scalar
//! columns, so the full SQL surface — plain scalar SQL plus the
//! `bm25_search` / `vector_search` / `hybrid_search` table functions — is
//! all reachable through the one `query_sql` read path. No internal query
//! plumbing is touched.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int64Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use infino::superfile::builder::{FtsConfig, VectorConfig};
use infino::superfile::vector::distance::Metric;
use infino::superfile::vector::rerank_codec::RerankCodec;
use infino::supertable::{Supertable, SupertableOptions};
use infino::test_helpers::default_tokenizer;
use rayon::prelude::*;

use super::{Capabilities, SqlEngine, SqlOutput, SqlRow};

const TITLE_COLUMN: &str = "title";
// Low-cardinality bucket label (`b0`..`b9` by `doc_id % 10`), so an
// equality like `bucket = 'b0'` selects exactly 10% of rows. FTS-indexed,
// so the WHERE pushdown resolves it through `token_match`.
const BUCKET_COLUMN: &str = "bucket";
// High-cardinality key whose value is **uncorrelated with row (doc_id)
// order** — a multiplicative hash — so consecutive rows get values spread
// across the whole domain, defeating min/max page pruning. FTS-indexed, so
// the lookup resolves the one row's page directly.
const KEY_COLUMN: &str = "key";
const CATEGORY_COLUMN: &str = "category";
// Named `rating` (not `score`) so it never collides with the `score`
// column the bm25 / vector / hybrid TVFs append to their output schema.
const SCORE_COLUMN: &str = "rating";
const VECTOR_COLUMN: &str = "emb";
const WRITE_CHUNK: usize = 65_536;

/// Small vector dimension — this is a SQL/hybrid query-latency bench, not
/// a vector-recall bench, so a tiny dim keeps the vector index cheap.
pub const SQL_DIM: usize = 16;
const ROT_SEED: u64 = 7;

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

/// High-cardinality lookup key for `doc_id`, deliberately **uncorrelated
/// with row order** (Knuth multiplicative hash into a 1e8 domain ≫ the
/// row count, so values are near-unique and scattered). Used so a
/// `key = scatter_key(d)` lookup is selective *and* defeats min/max page
/// pruning. Shared by the harness (column data) and the bench (query
/// literal) so both agree on the value.
pub fn scatter_key(doc_id: u64) -> String {
    format!("k{:08}", doc_id.wrapping_mul(2_654_435_761) % 100_000_000)
}

pub fn sql_schema() -> Arc<Schema> {
    schema()
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(TITLE_COLUMN, DataType::LargeUtf8, false),
        Field::new(BUCKET_COLUMN, DataType::LargeUtf8, false),
        Field::new(KEY_COLUMN, DataType::LargeUtf8, false),
        Field::new(CATEGORY_COLUMN, DataType::LargeUtf8, false),
        Field::new(SCORE_COLUMN, DataType::Int64, false),
        Field::new(VECTOR_COLUMN, fixed_list_f32(SQL_DIM), false),
    ]))
}

/// Deterministic unit-norm embedding for `doc_id` — no RNG, so it is
/// reproducible and cheap. Used both to populate the `emb` column and
/// (via [`sample_query_csv`]) to form a query vector for the
/// `vector_search` / `hybrid_search` TVFs.
pub fn emb_for(doc_id: u64) -> [f32; SQL_DIM] {
    let mut v = [0f32; SQL_DIM];
    for (d, slot) in v.iter_mut().enumerate() {
        *slot = ((doc_id.wrapping_mul(31).wrapping_add(d as u64 * 7) % 97) as f32) + 1.0;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for slot in &mut v {
            *slot /= norm;
        }
    }
    v
}

/// CSV form of doc 0's embedding — a deterministic query vector for the
/// vector / hybrid TVFs.
pub fn sample_query_csv() -> String {
    emb_for(0)
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// `n_cent` for the vector index, clamped so tiny inputs (unit tests)
/// don't request more clusters than rows.
fn n_cent_for(n_rows: usize) -> usize {
    crate::corpus::n_cent(n_rows).min(n_rows.max(1))
}

/// Options for the SQL benchmark table shape.
pub fn sql_options(n_rows: usize) -> SupertableOptions {
    SupertableOptions::new(
        schema(),
        vec![
            FtsConfig {
                column: TITLE_COLUMN.into(),
            },
            FtsConfig {
                column: BUCKET_COLUMN.into(),
            },
            FtsConfig {
                column: KEY_COLUMN.into(),
            },
        ],
        vec![VectorConfig {
            column: VECTOR_COLUMN.into(),
            dim: SQL_DIM,
            n_cent: n_cent_for(n_rows),
            rot_seed: ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8Residual,
        }],
        Some(default_tokenizer()),
    )
    .expect("supertable sql options")
}

/// Build one in-memory supertable from `rows` via the public writer API.
fn build_supertable(rows: &[SqlRow<'_>]) -> Supertable {
    build_supertable_with_options(rows, sql_options(rows.len()), WRITE_CHUNK)
}

/// Build one supertable from `rows` via the public writer API with caller
/// supplied options and chunking. Bench cold tiers use this to build the
/// same SQL shape on object storage, then reopen it through a fresh cache.
pub fn build_supertable_with_options(
    rows: &[SqlRow<'_>],
    opts: SupertableOptions,
    write_chunk: usize,
) -> Supertable {
    let schema = schema();
    let st = Supertable::create(opts).expect("create supertable");
    let mut writer = st.writer().expect("writer");
    for chunk in rows.chunks(write_chunk.max(1)) {
        let titles = LargeStringArray::from(chunk.iter().map(|r| r.title).collect::<Vec<_>>());
        // 10-bucket label by `doc_id % 10`.
        let bucket_vals: Vec<String> = chunk
            .iter()
            .map(|r| format!("b{}", r.doc_id % 10))
            .collect();
        let buckets =
            LargeStringArray::from(bucket_vals.iter().map(String::as_str).collect::<Vec<_>>());
        // High-cardinality, order-uncorrelated key.
        let key_vals: Vec<String> = chunk.iter().map(|r| scatter_key(r.doc_id)).collect();
        let keys = LargeStringArray::from(key_vals.iter().map(String::as_str).collect::<Vec<_>>());
        let categories =
            LargeStringArray::from(chunk.iter().map(|r| r.category).collect::<Vec<_>>());
        let scores = Int64Array::from(chunk.iter().map(|r| r.score).collect::<Vec<_>>());
        let mut flat = Vec::with_capacity(chunk.len() * SQL_DIM);
        for r in chunk {
            flat.extend_from_slice(&emb_for(r.doc_id));
        }
        let emb = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            SQL_DIM as i32,
            Arc::new(Float32Array::from(flat)) as ArrayRef,
            None,
        )
        .expect("emb FixedSizeList");
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(titles),
                Arc::new(buckets),
                Arc::new(keys),
                Arc::new(categories),
                Arc::new(scores),
                Arc::new(emb),
            ],
        )
        .expect("sql record batch");
        writer.append(&batch).expect("append");
        writer.commit().expect("commit");
    }
    drop(writer);
    st
}

/// infino as a SQL engine.
pub struct InfinoSqlEngine;

/// Sealed infino SQL index: the in-memory supertable handle built by
/// `write`, queried through `query_sql`.
pub struct InfinoSqlIndex {
    table: Option<Supertable>,
}

impl InfinoSqlIndex {
    /// The supertable handle built by the measured 1-writer build.
    pub fn table(&self) -> &Supertable {
        self.table.as_ref().expect("table requested before write")
    }
}

impl SqlEngine for InfinoSqlEngine {
    type Index = InfinoSqlIndex;

    fn name() -> &'static str {
        "infino"
    }

    fn capabilities() -> Capabilities {
        Capabilities {
            fts: true,
            vector: true,
            sql: true,
            hybrid: true,
        }
    }

    fn create() -> Self::Index {
        InfinoSqlIndex { table: None }
    }

    fn write(index: &mut Self::Index, rows: &[SqlRow<'_>]) {
        index.table = Some(build_supertable(rows));
    }

    fn parallel_write(rows: &[SqlRow<'_>], writers: usize) {
        let writers = writers.max(1);
        if writers == 1 {
            std::hint::black_box(build_supertable(rows));
            return;
        }
        // Parallel build: shard the rows across `writers` builders, each
        // producing its own in-memory table concurrently (rayon `par_chunks`,
        // mirroring the FTS/vector engines). Build-only — handles dropped.
        let rows_per = rows.len().div_ceil(writers);
        let built: Vec<Supertable> = rows.par_chunks(rows_per).map(build_supertable).collect();
        std::hint::black_box(built);
    }

    fn read(index: &Self::Index, sql: &str) -> SqlOutput {
        let batches = index.table().reader().query_sql(sql).expect("query_sql");
        SqlOutput {
            rows: batches.iter().map(RecordBatch::num_rows).sum(),
        }
    }

    fn close(index: &mut Self::Index) {
        index.table = None;
    }

    fn delete(_index: Self::Index) {
        // Dropping the in-memory supertable releases the artifact.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> Vec<SqlRow<'static>> {
        vec![
            SqlRow {
                doc_id: 0,
                title: "rust async runtime",
                category: "rust",
                score: 10,
            },
            SqlRow {
                doc_id: 1,
                title: "python data pipeline",
                category: "python",
                score: 20,
            },
            SqlRow {
                doc_id: 2,
                title: "rust web framework",
                category: "rust",
                score: 30,
            },
        ]
    }

    #[test]
    fn scalar_sql_roundtrip() {
        let mut idx = InfinoSqlEngine::create();
        InfinoSqlEngine::write(&mut idx, &rows());

        let total = InfinoSqlEngine::read(&idx, "SELECT * FROM supertable");
        assert_eq!(total.rows, 3, "all rows visible; got {}", total.rows);

        let rust =
            InfinoSqlEngine::read(&idx, "SELECT title FROM supertable WHERE category = 'rust'");
        assert_eq!(rust.rows, 2, "two rust rows; got {}", rust.rows);
    }

    #[test]
    fn tvf_sql_options_resolve() {
        let mut idx = InfinoSqlEngine::create();
        InfinoSqlEngine::write(&mut idx, &rows());
        let qv = sample_query_csv();

        // bm25_search, vector_search, and hybrid_search are all just SQL
        // options through the same query_sql read path.
        let bm25 = InfinoSqlEngine::read(&idx, "SELECT _id FROM bm25_search('title', 'rust', 10)");
        assert!(
            bm25.rows >= 1,
            "bm25 should match 'rust'; got {}",
            bm25.rows
        );

        let vector = InfinoSqlEngine::read(
            &idx,
            &format!("SELECT _id FROM vector_search('emb', '{qv}', 3)"),
        );
        assert!(
            vector.rows >= 1,
            "vector should return hits; got {}",
            vector.rows
        );

        let hybrid = InfinoSqlEngine::read(
            &idx,
            &format!("SELECT _id FROM hybrid_search('title', 'rust', 'emb', '{qv}', 3)"),
        );
        assert!(
            hybrid.rows >= 1,
            "hybrid should fuse hits; got {}",
            hybrid.rows
        );
    }
}
