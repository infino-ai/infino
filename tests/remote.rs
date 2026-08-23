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

use std::{collections::BTreeSet, io::Cursor, sync::Arc, time::Duration};

use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
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

/// A schema's Arrow-IPC bytes: a schema message and nothing else, which is what
/// a describe response carries.
fn schema_ipc_bytes(schema: &Schema) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut w = StreamWriter::try_new(&mut out, schema).expect("ipc schema writer");
        w.finish().expect("ipc schema finish");
    }
    out
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

    // The schema travels as base64 Arrow IPC. Assert on what it decodes to
    // rather than on exact bytes, so the test pins the contract and not the
    // encoder's byte layout.
    let requests = server
        .received_requests()
        .await
        .expect("request recording is enabled");
    let body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("json request body");
    let encoded = body["schema_ipc"]
        .as_str()
        .expect("schema_ipc is a base64 string");
    let bytes = BASE64.decode(encoded).expect("valid base64");
    let reader = StreamReader::try_new(Cursor::new(bytes), None).expect("an arrow ipc stream");
    assert_eq!(
        reader.schema().as_ref(),
        id_schema().as_ref(),
        "the posted schema decodes back to the declared one"
    );
    assert!(
        body.get("schema").is_none(),
        "the JSON descriptor form is no longer sent"
    );
}

#[tokio::test]
async fn append_streams_arrow_body_with_table_query() {
    let server = MockServer::start().await;
    // open_table fetches the schema first.
    Mock::given(method("POST"))
        .and(path("/v1/schema/mydb"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(schema_ipc_bytes(&id_schema()), ARROW_CT),
        )
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
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(schema_ipc_bytes(&id_schema()), ARROW_CT),
        )
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
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(schema_ipc_bytes(&id_schema()), ARROW_CT),
        )
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
            .vector_search("emb", &[1.0, 0.0], 5, Some(filter), None)
            .expect("vector_search")
    })
    .await;
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
}

#[tokio::test]
async fn vector_tuning_overrides_are_rejected_on_the_remote_transport() {
    // Options never cross the wire; the test-only `_with_options` variants
    // must fail loudly against a hosted table rather than silently serve
    // engine defaults (a sweep or oracle run would record wrong numbers).
    // No search endpoint is mounted: the rejection must precede any wire I/O.
    let server = MockServer::start().await;
    mount_schema(&server).await;

    let err = with_connection(server.uri(), |db| {
        let table = db.open_table("posts").expect("open");
        table
            .vector_search_with_options(
                "emb",
                &[1.0, 0.0],
                5,
                VectorSearchOptions::new().with_nprobe(2),
                None,
                None,
            )
            .expect_err("override must be refused")
    })
    .await;
    assert!(err.to_string().contains("remote transport"), "got: {err}");
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
            .hybrid_search("id", "hi", BoolMode::Or, "emb", &[1.0, 0.0], 5, None)
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

/// Collect the sorted, deduped `<op>` segment from each `/v1/<op>/…` path.
fn path_segments(paths: impl Iterator<Item = String>) -> Vec<String> {
    let mut segs: Vec<String> = paths
        .filter_map(|p| {
            p.strip_prefix("/v1/")
                .map(|rest| rest.split('/').next().unwrap_or_default().to_string())
        })
        .collect();
    segs.sort();
    segs.dedup();
    segs
}

/// The request-body schema name the spec defines for `(method, <op>)`, if the
/// operation carries a JSON body. `None` for bodyless ops (binary/query-param).
fn request_schema_name(spec: &serde_json::Value, method: &str, seg: &str) -> Option<String> {
    for (path, methods) in spec["paths"].as_object()? {
        if path.split('/').nth(2) != Some(seg) {
            continue;
        }
        let Some(op) = methods.get(method) else {
            continue;
        };
        if let Some(r) = op
            .pointer("/requestBody/content/application~1json/schema/$ref")
            .and_then(serde_json::Value::as_str)
        {
            return Some(r.rsplit('/').next().unwrap_or_default().to_string());
        }
    }
    None
}

/// The remote client's wire calls must match the published data-plane API spec.
///
/// The spec (`fixtures/hosted-openapi.json`) is the source of truth, refreshed
/// from the deployed `/openapi.json` by the `refresh-hosted-openapi` workflow.
/// This drives every public method against a mock server, captures the requests
/// the client emits, and checks them against the spec at two levels:
///
/// 1. **Operations** — the set of `/v1/<op>` paths called equals the spec's
///    paths. A hosted operation added or removed fails here.
/// 2. **Request signatures** — for each operation with a JSON body, the fields
///    the client sends conform to the spec's request schema: every required
///    field is present, and no field is undefined (the server's
///    `deny_unknown_fields` would 400 an undefined one). A renamed, newly
///    required, or removed field fails here.
///
/// The Rust remote transport is the single place these requests are built; the
/// node and python bindings call through it, so this one check covers all three
/// bindings. A failure means: the hosted API changed — update the transport.
///
/// `#[ignore]`d on purpose: this checks conformance to an external, moving
/// contract (the hosted API), so it is deliberately kept out of `make ci` — a
/// hosted-service change must never gate an OSS engine release. It is run
/// explicitly by the `hosted-api-drift` workflow (on spec/transport changes)
/// via `cargo test … -- --ignored`.
#[tokio::test]
#[ignore = "hosted-API conformance; run by the hosted-api-drift workflow, not make ci"]
async fn remote_client_matches_the_published_api_spec() {
    let server = MockServer::start().await;
    mount_schema(&server).await;

    // Drive every operation. Only `open_table` needs a parseable response; the
    // rest go unmatched (404) but their requests are still recorded — we assert
    // on what the client sends, not on the responses. Searches pass a projection
    // so the required field is present (the projection-optionality question is
    // tracked separately). `optimize`/`gc` short-circuit and send nothing.
    with_connection(server.uri(), |db| {
        let _ = db.create_database();
        let _ = db.create_table("posts", id_schema(), IndexSpec::new());
        let _ = db.list_tables();
        let _ = db.drop_table("posts", false);
        let _ = db.query_sql("SELECT id FROM posts");
        if let Ok(table) = db.open_table("posts") {
            let _ = table.append(&id_batch(vec![1]));
            let _ = table.update(col("id").gt(lit(0_i32)), &id_batch(vec![1]));
            let _ = table.delete(col("id").lt(lit(0_i32)));
            let _ = table.bm25_search("id", "x", 1, Bm25SearchOptions::new(), Some(&["_id"]));
            let _ = table.token_match("id", "x", BoolMode::Or, Some(&["_id"]));
            let _ = table.exact_match("id", "x", Some(&["_id"]));
            let _ = table.count("id", "x", BoolMode::Or);
            let _ = table.vector_search("id", &[1.0], 1, None, Some(&["_id"]));
            let _ = table.hybrid_search("id", "x", BoolMode::Or, "id", &[1.0], 1, Some(&["_id"]));
            let _ = table.optimize(&OptimizeOptions::default());
            let _ = table.gc(Duration::from_secs(0));
        }
    })
    .await;

    let spec: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/hosted-openapi.json"))
            .expect("valid hosted API spec fixture");
    let requests = server
        .received_requests()
        .await
        .expect("request recording is enabled");

    // Tier 1: operation-set parity.
    let client_ops = path_segments(requests.iter().map(|r| r.url.path().to_string()));
    let spec_ops = path_segments(spec["paths"].as_object().expect("paths").keys().cloned());
    assert_eq!(
        client_ops, spec_ops,
        "the remote client's operations drifted from the published API spec \
         (a hosted operation was added or removed); reconcile the transport"
    );

    // Tier 2: request-signature conformance for JSON-body operations.
    for req in &requests {
        let method = req.method.to_string().to_lowercase();
        let Some(seg) = req
            .url
            .path()
            .strip_prefix("/v1/")
            .map(|rest| rest.split('/').next().unwrap_or_default())
        else {
            continue;
        };
        let Some(schema_name) = request_schema_name(&spec, &method, seg) else {
            continue; // bodyless op (binary/query-param) — nothing to conform.
        };
        let schema = &spec["components"]["schemas"][&schema_name];
        let required: BTreeSet<&str> = schema["required"]
            .as_array()
            .map(|a| a.iter().filter_map(serde_json::Value::as_str).collect())
            .unwrap_or_default();
        let properties: BTreeSet<&str> = schema["properties"]
            .as_object()
            .map(|o| o.keys().map(String::as_str).collect())
            .unwrap_or_default();

        let body: serde_json::Value =
            serde_json::from_slice(&req.body).unwrap_or(serde_json::Value::Null);
        let Some(obj) = body.as_object() else {
            continue; // non-JSON body — not a schema-typed request.
        };
        let sent: BTreeSet<&str> = obj.keys().map(String::as_str).collect();

        let missing: Vec<&&str> = required.difference(&sent).collect();
        assert!(
            missing.is_empty(),
            "{method} /v1/{seg}: client omits required field(s) {missing:?} that \
             {schema_name} requires — the request signature drifted from the spec"
        );
        let unknown: Vec<&&str> = sent.difference(&properties).collect();
        assert!(
            unknown.is_empty(),
            "{method} /v1/{seg}: client sends field(s) {unknown:?} not defined by \
             {schema_name} — the request signature drifted (the server would reject these)"
        );
    }
}
