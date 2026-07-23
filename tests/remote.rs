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
use infino::{BoolMode, ConnectOptions, IndexSpec, InfinoError};
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
            .bm25_search("id", "hello", 10, BoolMode::Or, None)
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

#[tokio::test]
async fn unsupported_op_errors_without_a_request() {
    let server = MockServer::start().await;
    // schema fetch for open_table; no vector_search mock — a request would 501.
    Mock::given(method("POST"))
        .and(path("/v1/schema/mydb"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {"name": "id", "type": "i32", "nullable": false}
        ])))
        .mount(&server)
        .await;

    let err = with_connection(server.uri(), |db| {
        let table = db.open_table("posts").expect("open_table");
        table
            .vector_search("v", &[0.0], 5, Default::default(), None, None)
            .expect_err("vector_search is not supported remotely")
    })
    .await;
    assert!(matches!(err, InfinoError::Backend(_)), "got {err:?}");
}
