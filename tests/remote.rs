// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Hermetic tests for the remote (hosted) transport.
//!
//! A `wiremock` server stands in for the hosted endpoint: each test asserts the
//! request the sync client emits (method, path, auth header, body) and returns
//! a canned response, so the transport is exercised end-to-end with no real
//! service. The client runs on a blocking thread (`spawn_blocking`) so its
//! synchronous HTTP call never blocks the mock server's async runtime.

#![cfg(feature = "remote")]

use std::sync::Arc;

use arrow::ipc::writer::StreamWriter;
use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::prelude::{col, lit};
use infino::{
    Bm25SearchOptions, BoolMode, ConnectOptions, IndexSpec, InfinoError, OptimizeError,
    OptimizeOptions, VectorFilter, VectorSearchOptions,
};
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, header, method, path, query_param},
};

const KEY: &str = "ik_test";
const ARROW_CT: &str = "application/vnd.apache.arrow.stream";

fn id_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]))
}

/// A one-column `id` batch, and its Arrow-IPC bytes (a canned search response).
fn id_batch(ids: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(id_schema(), vec![Arc::new(Int32Array::from(ids))]).expect("batch")
}

fn ipc_bytes(batch: &RecordBatch) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut w = StreamWriter::try_new(&mut out, &batch.schema()).expect("ipc writer");
        w.write(batch).expect("ipc write");
        w.finish().expect("ipc finish");
    }
    out
}

/// Connect to the mock endpoint on a blocking thread and run `f`, returning its
/// result. Keeps the synchronous client off the async runtime.
async fn with_connection<T, F>(uri: String, f: F) -> T
where
    T: Send + 'static,
    F: FnOnce(infino::Connection) -> T + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let db = infino::connect_with(
            format!("{uri}/mydb"),
            ConnectOptions::new().with_api_key(KEY),
        )
        .expect("connect");
        f(db)
    })
    .await
    .expect("blocking task")
}

#[tokio::test]
async fn create_table_posts_expected_shape() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/create_table/mydb"))
        .and(header("authorization", format!("Bearer {KEY}").as_str()))
        .and(body_partial_json(json!({
            "table_name": "posts",
            "schema": [{"name": "id", "type": "i32", "nullable": false}],
            "indexes": {"fts": ["id"]},
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&server)
        .await;

    with_connection(server.uri(), |db| {
        db.create_table("posts", id_schema(), IndexSpec::new().fts("id"))
            .expect("create_table");
    })
    .await;
}

#[tokio::test]
async fn append_streams_arrow_body_with_table_query() {
    let server = MockServer::start().await;
    // open_table fetches the schema first.
    Mock::given(method("POST"))
        .and(path("/v1/schema/mydb"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {"name": "id", "type": "i32", "nullable": false}
        ])))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/append/mydb"))
        .and(query_param("table", "posts"))
        .and(header("content-type", ARROW_CT))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "rows": 3 })))
        .expect(1)
        .mount(&server)
        .await;

    with_connection(server.uri(), |db| {
        let table = db.open_table("posts").expect("open_table");
        table.append(&id_batch(vec![1, 2, 3])).expect("append");
    })
    .await;
}

#[tokio::test]
async fn bm25_search_sends_json_and_decodes_arrow() {
    let server = MockServer::start().await;
    // open_table fetches the schema first.
    Mock::given(method("POST"))
        .and(path("/v1/schema/mydb"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {"name": "id", "type": "i32", "nullable": false}
        ])))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/bm25_search/mydb"))
        .and(header("accept", ARROW_CT))
        .and(body_partial_json(json!({
            "table_name": "posts",
            "field_name": "id",
            "query": "hello",
            "k": 10,
            "mode": "or",
            "stats": "per_superfile",
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(ipc_bytes(&id_batch(vec![1, 2, 3])), ARROW_CT),
        )
        .expect(1)
        .mount(&server)
        .await;

    let rows = with_connection(server.uri(), |db| {
        let table = db.open_table("posts").expect("open_table");
        table
            .bm25_search("id", "hello", 10, Bm25SearchOptions::new(), None)
            .expect("bm25_search")
    })
    .await;
    let total: usize = rows.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(total, 3, "decoded the canned Arrow response into 3 rows");
}

#[tokio::test]
async fn query_sql_sends_json_and_decodes_arrow() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/query_sql/mydb"))
        .and(body_partial_json(
            json!({ "query": "SELECT id FROM posts" }),
        ))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(ipc_bytes(&id_batch(vec![7, 8])), ARROW_CT),
        )
        .expect(1)
        .mount(&server)
        .await;

    let rows = with_connection(server.uri(), |db| {
        db.query_sql("SELECT id FROM posts").expect("query_sql")
    })
    .await;
    let total: usize = rows.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(total, 2);
}

#[tokio::test]
async fn list_tables_parses_json_array() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/list_tables/mydb"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!(["a", "b"])))
        .mount(&server)
        .await;

    let names = with_connection(server.uri(), |db| db.list_tables().expect("list_tables")).await;
    assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
}

#[tokio::test]
async fn open_table_missing_maps_to_not_found() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/schema/mydb"))
        .respond_with(ResponseTemplate::new(404).set_body_string("no such table"))
        .mount(&server)
        .await;

    let err = with_connection(server.uri(), |db| {
        db.open_table("ghost")
            .expect_err("missing table must error")
    })
    .await;
    assert!(matches!(err, InfinoError::NotFound(_)), "got {err:?}");
}

/// Mount the schema endpoint so `open_table("posts")` succeeds — the other
/// table ops fetch the schema on open.
async fn mount_schema(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/schema/mydb"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {"name": "id", "type": "i32", "nullable": false}
        ])))
        .mount(server)
        .await;
}

#[tokio::test]
async fn token_match_sends_json_and_decodes_arrow() {
    let server = MockServer::start().await;
    mount_schema(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/token_match/mydb"))
        .and(body_partial_json(json!({
            "table_name": "posts", "field_name": "id", "query": "a b", "mode": "and",
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(ipc_bytes(&id_batch(vec![1, 2])), ARROW_CT),
        )
        .expect(1)
        .mount(&server)
        .await;

    let rows = with_connection(server.uri(), |db| {
        db.open_table("posts")
            .expect("open")
            .token_match("id", "a b", BoolMode::And, None)
            .expect("token_match")
    })
    .await;
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
}

#[tokio::test]
async fn exact_match_sends_json_and_decodes_arrow() {
    let server = MockServer::start().await;
    mount_schema(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/exact_match/mydb"))
        .and(body_partial_json(json!({
            "table_name": "posts", "field_name": "id", "value": "7",
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(ipc_bytes(&id_batch(vec![7])), ARROW_CT),
        )
        .expect(1)
        .mount(&server)
        .await;

    let rows = with_connection(server.uri(), |db| {
        db.open_table("posts")
            .expect("open")
            .exact_match("id", "7", None)
            .expect("exact_match")
    })
    .await;
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
}

#[tokio::test]
async fn count_parses_json_count() {
    let server = MockServer::start().await;
    mount_schema(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/count/mydb"))
        .and(body_partial_json(json!({
            "table_name": "posts", "field_name": "id", "query": "x", "mode": "or",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "count": 42 })))
        .expect(1)
        .mount(&server)
        .await;

    let n = with_connection(server.uri(), |db| {
        db.open_table("posts")
            .expect("open")
            .count("id", "x", BoolMode::Or)
            .expect("count")
    })
    .await;
    assert_eq!(n, 42);
}

#[tokio::test]
async fn vector_search_sends_query_filter_and_decodes_arrow() {
    let server = MockServer::start().await;
    mount_schema(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/vector_search/mydb"))
        .and(body_partial_json(json!({
            "table_name": "posts",
            "field_name": "emb",
            "query": [1.0, 0.0],
            "k": 5,
            "filter": {"field_name": "id", "query": "1", "mode": "or"},
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(ipc_bytes(&id_batch(vec![9])), ARROW_CT),
        )
        .expect(1)
        .mount(&server)
        .await;

    let rows = with_connection(server.uri(), |db| {
        let table = db.open_table("posts").expect("open");
        let filter = VectorFilter {
            column: "id",
            query: "1",
            mode: BoolMode::Or,
        };
        table
            .vector_search(
                "emb",
                &[1.0, 0.0],
                5,
                VectorSearchOptions::new(),
                Some(filter),
                None,
            )
            .expect("vector_search")
    })
    .await;
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
}

#[tokio::test]
async fn hybrid_search_sends_text_and_vector_fields() {
    let server = MockServer::start().await;
    mount_schema(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/hybrid_search/mydb"))
        .and(body_partial_json(json!({
            "table_name": "posts",
            "text_field": "id",
            "text_query": "hi",
            "mode": "or",
            "vector_field": "emb",
            "vector_query": [1.0, 0.0],
            "k": 5,
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(ipc_bytes(&id_batch(vec![3])), ARROW_CT),
        )
        .expect(1)
        .mount(&server)
        .await;

    let rows = with_connection(server.uri(), |db| {
        db.open_table("posts")
            .expect("open")
            .hybrid_search(
                "id",
                "hi",
                BoolMode::Or,
                "emb",
                &[1.0, 0.0],
                VectorSearchOptions::new(),
                5,
                None,
            )
            .expect("hybrid_search")
    })
    .await;
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
}

#[tokio::test]
async fn update_unparses_predicate_and_returns_stats() {
    let server = MockServer::start().await;
    mount_schema(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/update/mydb"))
        .and(query_param("table", "posts"))
        .and(header("content-type", ARROW_CT))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "matched": 1, "n_tombstoned": 1, "n_not_found": 0,
        })))
        .expect(1)
        .mount(&server)
        .await;

    let stats = with_connection(server.uri(), |db| {
        db.open_table("posts")
            .expect("open")
            .update(col("id").gt(lit(1_i32)), &id_batch(vec![7]))
            .expect("update")
    })
    .await;
    assert_eq!(stats.matched(), 1);
    assert_eq!(stats.n_tombstoned(), 1);
}

#[tokio::test]
async fn delete_unparses_predicate_and_returns_stats() {
    let server = MockServer::start().await;
    mount_schema(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/delete/mydb"))
        .and(query_param("table", "posts"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "matched": 2, "n_tombstoned": 2, "n_not_found": 0,
        })))
        .expect(1)
        .mount(&server)
        .await;

    let stats = with_connection(server.uri(), |db| {
        db.open_table("posts")
            .expect("open")
            .delete(col("id").lt(lit(5_i32)))
            .expect("delete")
    })
    .await;
    assert_eq!(stats.n_tombstoned(), 2);
}

#[tokio::test]
async fn optimize_is_client_unsupported_without_a_request() {
    let server = MockServer::start().await;
    mount_schema(&server).await;
    // No /v1/optimize mock: optimize is a server-side operation on a hosted
    // table, so it must short-circuit client-side and never send a request.
    let err = with_connection(server.uri(), |db| {
        db.open_table("posts")
            .expect("open")
            .optimize(&OptimizeOptions::default())
            .expect_err("optimize is server-side for a hosted table")
    })
    .await;
    assert!(matches!(err, OptimizeError::NoStorage), "got {err:?}");
}

#[tokio::test]
async fn create_database_posts_name_to_account_scoped_endpoint() {
    let server = MockServer::start().await;
    // The endpoint is account-scoped: no `/mydb` path segment, and the target
    // database travels in the body as `name`.
    Mock::given(method("POST"))
        .and(path("/v1/databases"))
        .and(header("authorization", format!("Bearer {KEY}").as_str()))
        .and(body_partial_json(json!({ "name": "mydb" })))
        .respond_with(ResponseTemplate::new(201))
        .expect(1)
        .mount(&server)
        .await;

    with_connection(server.uri(), |db| {
        db.create_database().expect("create_database");
    })
    .await;
}

#[tokio::test]
async fn create_database_conflict_maps_to_already_exists() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/databases"))
        .respond_with(ResponseTemplate::new(409).set_body_string("database exists"))
        .mount(&server)
        .await;

    let err = with_connection(server.uri(), |db| {
        db.create_database()
            .expect_err("a duplicate database must error")
    })
    .await;
    assert!(matches!(err, InfinoError::AlreadyExists(_)), "got {err:?}");
}
