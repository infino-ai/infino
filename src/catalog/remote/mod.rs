// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Remote (hosted-service) transport.
//!
//! A [`RemoteCatalog`] forwards catalog operations to a hosted endpoint over
//! HTTP with a synchronous client, so a `connect("https://host/db", api_key)`
//! target serves the same [`Connection`](crate::Connection) /
//! [`Supertable`](crate::Supertable) surface as a local one. Compiled only
//! under the `remote` feature.

pub(crate) mod table;
pub(crate) mod wire;

use std::{
    io::{self, Read, copy, sink},
    sync::Arc,
    time::Duration,
};

use arrow_array::RecordBatch;
use arrow_schema::{Schema, SchemaRef};
use serde_json::{Value, from_reader, json};
use ureq::{Agent, Body, Error as HttpError, http};

use crate::{
    IndexSpec, InfinoError, Supertable,
    supertable::schema::{TableSchema, change::SchemaPatch},
};

/// Environment variable consulted for the API key when `ConnectOptions` does
/// not carry one.
const API_KEY_ENV: &str = "INFINO_API_KEY";

/// The `encoding` a describe names to get the schema document rather than
/// the column-descriptor array the endpoint answers by default.
const DOCUMENT_ENCODING: &str = "document";

/// Longest wait to look up the endpoint's address, and then to open the TCP
/// and TLS connection to it.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Longest wait for the endpoint to send the next bytes of a response. The
/// response to a write comes after the endpoint commits it, so this also
/// bounds the commit; it is set well above the endpoint's own request limit,
/// so it fires only on a connection that has stopped.
const READ_TIMEOUT: Duration = Duration::from_secs(300);

/// Longest wait for the connection to take more of a request body. It counts
/// from the last progress, not from the start, so a slow upload that keeps
/// moving is not cut off by it.
const WRITE_TIMEOUT: Duration = Duration::from_secs(60);

/// Longest a whole request may take, from the address lookup to the last byte
/// of the response. The service ends a request it is still working on after
/// 600 s (its load balancer's limit), so past that no answer can come; this
/// sits one connect timeout above it. A request is never sent twice, so this
/// is also the bound of the call.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(610);

/// The bounds of one hosted request.
#[derive(Clone, Copy)]
struct Timeouts {
    connect: Duration,
    read: Duration,
    write: Duration,
    total: Duration,
}

/// The bounds every hosted request runs under.
const TIMEOUTS: Timeouts = Timeouts {
    connect: CONNECT_TIMEOUT,
    read: READ_TIMEOUT,
    write: WRITE_TIMEOUT,
    total: REQUEST_TIMEOUT,
};

/// A response from the endpoint.
pub(crate) type Response = http::Response<Body>;

/// The HTTP client for a hosted connection. Every bound applies to every
/// request, on a new connection and on one reused from the pool alike, so a
/// request on a connection that stops making progress, or on one that keeps
/// trickling bytes, ends with an error instead of holding its caller.
fn agent(timeouts: Timeouts) -> Agent {
    let config = Agent::config_builder()
        .timeout_resolve(Some(timeouts.connect))
        .timeout_connect(Some(timeouts.connect))
        .timeout_per_read(Some(timeouts.read))
        .timeout_per_write(Some(timeouts.write))
        .timeout_global(Some(timeouts.total))
        // An error status comes back as a response, so its body can name the
        // cause (see `map_send`).
        .http_status_as_error(false)
        // No proxy is taken from the environment.
        .proxy(None)
        .build();
    Agent::new_with_config(config)
}

/// A connection to a hosted endpoint. Holds the sync HTTP client, the base URL
/// (scheme + host), the target database, and the bearer credential.
pub(crate) struct RemoteCatalog {
    agent: Agent,
    base_url: String,
    database: String,
    api_key: String,
}

impl RemoteCatalog {
    /// Build a hosted connection. The key comes from `api_key` or, when unset,
    /// the `INFINO_API_KEY` environment variable; a hosted connection without a
    /// key is a configuration error.
    pub(crate) fn new(
        base_url: String,
        database: String,
        api_key: Option<String>,
    ) -> Result<Self, InfinoError> {
        let api_key = api_key
            .or_else(|| std::env::var(API_KEY_ENV).ok())
            .filter(|k| !k.is_empty())
            .ok_or_else(|| {
                InfinoError::Config(format!(
                    "a hosted connection needs an API key (pass ConnectOptions::with_api_key or set {API_KEY_ENV})"
                ))
            })?;
        Ok(Self {
            agent: agent(TIMEOUTS),
            base_url,
            database,
            api_key,
        })
    }

    /// The endpoint URL for operation `op` against this connection's database.
    fn url(&self, op: &str) -> String {
        format!("{}/v1/{op}/{}", self.base_url, self.database)
    }

    fn bearer(&self) -> String {
        format!("Bearer {}", self.api_key)
    }

    /// POST a JSON body, requesting an Arrow-stream response (read endpoints
    /// honor it; JSON endpoints ignore it and return JSON).
    pub(crate) fn post_json(&self, op: &str, body: Value) -> Result<Response, InfinoError> {
        let request = self
            .agent
            .post(&self.url(op))
            .header("Authorization", &self.bearer())
            .header("Accept", wire::ARROW_STREAM_CONTENT_TYPE);
        map_send(op, request.send_json(body))
    }

    /// POST an Arrow-IPC body with query parameters — the `append` path.
    pub(crate) fn post_arrow(
        &self,
        op: &str,
        query: &[(&str, &str)],
        body: Vec<u8>,
    ) -> Result<Response, InfinoError> {
        let mut request = self
            .agent
            .post(&self.url(op))
            .header("Authorization", &self.bearer())
            .header("Content-Type", wire::ARROW_STREAM_CONTENT_TYPE);
        for (key, value) in query {
            request = request.query(key, value);
        }
        map_send(op, request.send(body))
    }

    /// Register the database this connection targets on the hosted service.
    /// `POST /v1/databases` with `{name}` — this endpoint is account-scoped
    /// (the account is identified by the API key), so unlike the per-database
    /// operations its path carries no database segment. A `201` is success; a
    /// `409` (already registered) surfaces as [`InfinoError::AlreadyExists`].
    pub(crate) fn create_database(&self) -> Result<(), InfinoError> {
        let url = format!("{}/v1/databases", self.base_url);
        let request = self
            .agent
            .post(&url)
            .header("Authorization", &self.bearer());
        let response = map_send(
            "create_database",
            request.send_json(json!({ "name": self.database })),
        )?;
        finish("create_database", response)
    }

    pub(crate) fn create_table(
        self: &Arc<Self>,
        name: &str,
        schema: SchemaRef,
        indexes: IndexSpec,
    ) -> Result<Supertable, InfinoError> {
        let body = json!({
            "table_name": name,
            "schema_ipc": wire::schema_to_ipc_base64(&schema)?,
            "indexes": wire::index_spec_to_json(&indexes),
        });
        finish("create_table", self.post_json("create_table", body)?)?;
        Ok(Supertable::from_table(Arc::new(table::RemoteTable::new(
            Arc::clone(self),
            name.to_string(),
            schema,
        ))))
    }

    pub(crate) fn open_table(self: &Arc<Self>, name: &str) -> Result<Supertable, InfinoError> {
        // Fetch the schema: this validates the table exists (a missing table is
        // a 404 → NotFound) and caches the schema so `schema()` is infallible.
        let response = self.post_json(
            "schema",
            json!({ "table_name": name, "encoding": wire::IPC_ENCODING }),
        )?;
        let schema = Arc::new(read_arrow_schema("schema", response)?);
        Ok(Supertable::from_table(Arc::new(table::RemoteTable::new(
            Arc::clone(self),
            name.to_string(),
            schema,
        ))))
    }

    pub(crate) fn list_tables(&self) -> Result<Vec<String>, InfinoError> {
        let response = self.post_json("list_tables", json!({}))?;
        let value = read_json("list_tables", response)?;
        let names = value.as_array().ok_or_else(|| {
            InfinoError::Backend("list_tables response was not a JSON array".to_string())
        })?;
        names
            .iter()
            .map(|v| {
                v.as_str().map(str::to_owned).ok_or_else(|| {
                    InfinoError::Backend("list_tables entry was not a string".to_string())
                })
            })
            .collect()
    }

    pub(crate) fn drop_table(&self, name: &str, purge: bool) -> Result<(), InfinoError> {
        let response =
            self.post_json("drop_table", json!({ "table_name": name, "purge": purge }))?;
        finish("drop_table", response)
    }

    pub(crate) fn query_sql(&self, sql: &str) -> Result<Vec<RecordBatch>, InfinoError> {
        let response = self.post_json("query_sql", json!({ "query": sql }))?;
        read_arrow("query_sql", response)
    }

    /// The schema document of `name`: `POST /v1/schema` with
    /// `encoding: "document"`. The endpoint serves the document in the
    /// engine's own JSON (what [`TableSchema::to_json`] writes), so it reads
    /// back with the same codec and nothing is translated.
    pub(crate) fn schema_document(&self, name: &str) -> Result<TableSchema, InfinoError> {
        let response = self.post_json(
            "schema",
            json!({ "table_name": name, "encoding": DOCUMENT_ENCODING }),
        )?;
        read_schema_document("schema", response)
    }

    /// Merge `patch` into the schema of `name` (creating the table when
    /// there is none) and read back the document that resulted: `POST
    /// /v1/schema` with `patch`, and `expected_schema_id` when the caller
    /// compares. The endpoint answers a compare-and-set that lost with 409,
    /// which on this one route means the schema moved — the local handle's
    /// [`InfinoError::Conflict`] — and never "already exists".
    pub(crate) fn apply_schema(
        &self,
        name: &str,
        patch: &SchemaPatch,
        expected: Option<u32>,
    ) -> Result<TableSchema, InfinoError> {
        let mut body = json!({ "table_name": name, "patch": patch.to_json() });
        if let Some(expected) = expected {
            body["expected_schema_id"] = json!(expected);
        }
        let response = match self.post_json("schema", body) {
            Ok(response) => response,
            // `status_to_error` reads every 409 as AlreadyExists, which is
            // what it means on the create routes. This route merges into the
            // table or creates it, so it never refuses a table for existing:
            // its 409 is only ever a lost compare-and-set, the schema moved.
            Err(InfinoError::AlreadyExists(message)) => {
                return Err(InfinoError::Conflict(message));
            }
            Err(e) => return Err(e),
        };
        read_schema_document("apply_schema", response)
    }
}

/// Read a schema document off a JSON response body.
fn read_schema_document(op: &str, response: Response) -> Result<TableSchema, InfinoError> {
    let value = read_json(op, response)?;
    TableSchema::from_json(&value)
        .map_err(|e| InfinoError::Backend(format!("{op}: parsing the schema document: {e}")))
}

/// Turn a send result into an [`InfinoError`]: an HTTP error status maps
/// through [`wire::status_to_error`]; a failure to get an answer at all
/// through [`transport_error`].
fn map_send(op: &str, result: Result<Response, HttpError>) -> Result<Response, InfinoError> {
    let response = result.map_err(|e| transport_error(op, e))?;
    let status = response.status();
    if !(status.is_client_error() || status.is_server_error()) {
        return Ok(response);
    }
    let body = response.into_body().read_to_string().unwrap_or_default();
    Err(wire::status_to_error(op, status.as_u16(), &body))
}

/// A request that got no answer: [`InfinoError::Unavailable`] when the
/// endpoint could not be reached, the connection failed or broke, or a bound
/// passed, any of which a later try may get past; else a `Backend` error (a
/// request the client could not build, a TLS certificate refused).
fn transport_error(op: &str, e: HttpError) -> InfinoError {
    let message = format!("{op}: transport error: {e}");
    match e {
        HttpError::Timeout(_)
        | HttpError::Io(_)
        | HttpError::HostNotFound
        | HttpError::ConnectionFailed
        | HttpError::Protocol(_) => InfinoError::Unavailable(message),
        _ => InfinoError::Backend(message),
    }
}

/// A response body that stopped before its end: the connection broke, or a
/// bound passed while it was read.
fn body_error(op: &str, e: io::Error) -> InfinoError {
    InfinoError::Unavailable(format!("{op}: reading response: {e}"))
}

/// Read the rest of a response whose body says nothing the caller needs. A
/// connection goes back to the pool for the next request only once its
/// response is read to the end; one left unread is closed.
pub(crate) fn finish(op: &str, response: Response) -> Result<(), InfinoError> {
    copy(&mut response.into_body().into_reader(), &mut sink()).map_err(|e| body_error(op, e))?;
    Ok(())
}

/// Read a JSON response body.
pub(crate) fn read_json(op: &str, response: Response) -> Result<Value, InfinoError> {
    from_reader(response.into_body().into_reader()).map_err(|e| {
        if e.is_io() {
            body_error(op, e.into())
        } else {
            InfinoError::Backend(format!("{op}: parsing response: {e}"))
        }
    })
}

/// The whole body of a response.
fn read_body(op: &str, response: Response) -> Result<Vec<u8>, InfinoError> {
    let mut buf = Vec::new();
    response
        .into_body()
        .into_reader()
        .read_to_end(&mut buf)
        .map_err(|e| body_error(op, e))?;
    Ok(buf)
}

/// Read an Arrow-IPC response body into record batches.
pub(crate) fn read_arrow(op: &str, response: Response) -> Result<Vec<RecordBatch>, InfinoError> {
    wire::ipc_to_batches(&read_body(op, response)?)
}

/// Read an Arrow-IPC response body into the schema it carries. The body is a
/// schema message with no batches, so this cannot go through
/// [`read_arrow`], which expects at least one batch to describe.
pub(crate) fn read_arrow_schema(op: &str, response: Response) -> Result<Schema, InfinoError> {
    wire::ipc_to_schema(&read_body(op, response)?)
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write, copy, sink},
        iter::repeat,
        net::{TcpListener, TcpStream},
        thread,
        time::{Duration, Instant},
    };

    use serde_json::json;

    use super::{RemoteCatalog, Timeouts, agent, finish, read_json};
    use crate::InfinoError;

    /// The per-read and per-write bound of the test client.
    const TEST_READ_TIMEOUT: Duration = Duration::from_millis(300);
    /// The connect bound of the test client, longer than the test needs.
    const TEST_CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
    /// The whole-request bound of the test client.
    const TEST_TOTAL_TIMEOUT: Duration = Duration::from_secs(1);
    /// How long a stalling endpoint holds the connection before closing it.
    /// A client with no bound waits this long, past [`LATEST_RETURN`].
    const SILENT_HOLD: Duration = Duration::from_secs(5);
    /// The latest a request may return for a bound to be what ended it.
    const LATEST_RETURN: Duration = Duration::from_secs(4);
    /// The gap between the bytes of a response that never ends: shorter than
    /// the per-read bound, so only the whole-request bound can end it.
    const TRICKLE_GAP: Duration = Duration::from_millis(100);
    /// A request body larger than the socket buffers on both ends, so its
    /// send waits on an endpoint that stopped reading.
    const LARGE_BODY_BYTES: usize = 64 << 20;
    /// What the endpoint answers a request it serves.
    const OK_RESPONSE: &[u8] =
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n[]";
    /// The head of a response whose body never comes in full.
    const ENDLESS_HEAD: &[u8] =
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000000\r\n\r\n";

    /// A catalog of the test endpoint at `addr`, under the test bounds.
    fn test_catalog(addr: &str) -> RemoteCatalog {
        let mut catalog = RemoteCatalog::new(
            format!("http://{addr}"),
            "db".to_string(),
            Some("ik_test".to_string()),
        )
        .expect("catalog");
        catalog.agent = agent(Timeouts {
            connect: TEST_CONNECT_TIMEOUT,
            read: TEST_READ_TIMEOUT,
            write: TEST_READ_TIMEOUT,
            total: TEST_TOTAL_TIMEOUT,
        });
        catalog
    }

    /// Read one request, head and body, off `stream`.
    fn read_request(stream: &mut TcpStream) {
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte).expect("request head");
            head.push(byte[0]);
        }
        let head = String::from_utf8_lossy(&head).to_lowercase();
        let length: u64 = head
            .lines()
            .find_map(|l| l.strip_prefix("content-length:"))
            .map_or(0, |n| n.trim().parse().expect("content-length"));
        let mut body = stream.take(length);
        copy(&mut body, &mut sink()).expect("request body");
    }

    /// An endpoint that answers the first request of its one connection,
    /// then hands the connection to `then`.
    fn serve_one_then(
        then: impl FnOnce(TcpStream) + Send + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr").to_string();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            read_request(&mut stream);
            stream.write_all(OK_RESPONSE).expect("answer");
            then(stream);
        });
        (addr, server)
    }

    /// Time `call` and check it failed as unavailable after `at_least` and
    /// before [`LATEST_RETURN`].
    fn assert_unavailable_within<T>(
        call: impl FnOnce() -> Result<T, InfinoError>,
        at_least: Duration,
    ) {
        let start = Instant::now();
        let result = call();
        let waited = start.elapsed();
        match result {
            Err(InfinoError::Unavailable(message)) => {
                assert!(
                    message.contains("transport error") || message.contains("reading response"),
                    "{message}"
                );
            }
            Err(other) => panic!("expected the service to be unavailable, got {other:?}"),
            Ok(_) => panic!("a stalled endpoint answered"),
        }
        assert!(waited >= at_least, "returned before the bound: {waited:?}");
        assert!(waited < LATEST_RETURN, "waited {waited:?}, past the bound");
    }

    /// A request to an endpoint that accepts the connection and then sends
    /// nothing fails once the read bound passes, instead of waiting forever.
    #[test]
    fn a_request_to_a_silent_endpoint_times_out() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr").to_string();
        // Accept and hold the connection open without reading or answering.
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept");
            thread::sleep(SILENT_HOLD);
            drop(stream);
        });
        let catalog = test_catalog(&addr);
        assert_unavailable_within(|| catalog.post_json("append", json!({})), TEST_READ_TIMEOUT);
        server.join().expect("server thread");
    }

    /// The read bound holds on a connection reused from the pool too: the
    /// endpoint answers one request, then reads the next and never answers.
    #[test]
    fn a_request_on_a_reused_connection_to_a_silent_endpoint_times_out() {
        let (addr, server) = serve_one_then(|mut stream| {
            read_request(&mut stream);
            thread::sleep(SILENT_HOLD);
        });
        let catalog = test_catalog(&addr);
        finish(
            "list_tables",
            catalog
                .post_json("list_tables", json!({}))
                .expect("first request"),
        )
        .expect("first answer");
        assert_unavailable_within(|| catalog.post_json("append", json!({})), TEST_READ_TIMEOUT);
        server.join().expect("server thread");
    }

    /// The write bound holds on a reused connection: the endpoint answers
    /// one request, then stops reading, and a large append body cannot be
    /// sent.
    #[test]
    fn a_large_body_on_a_reused_connection_to_an_endpoint_that_stops_reading_times_out() {
        let (addr, server) = serve_one_then(|stream| {
            thread::sleep(SILENT_HOLD);
            drop(stream);
        });
        let catalog = test_catalog(&addr);
        finish(
            "list_tables",
            catalog
                .post_json("list_tables", json!({}))
                .expect("first request"),
        )
        .expect("first answer");
        let body = vec![0u8; LARGE_BODY_BYTES];
        assert_unavailable_within(
            || catalog.post_arrow("append", &[("table", "t")], body),
            TEST_READ_TIMEOUT,
        );
        server.join().expect("server thread");
    }

    /// A response that keeps coming a byte at a time, each within the read
    /// bound, ends at the whole-request bound.
    #[test]
    fn a_response_that_never_ends_stops_at_the_request_bound() {
        let (addr, server) = serve_one_then(|mut stream| {
            read_request(&mut stream);
            let started = Instant::now();
            let mut bytes = ENDLESS_HEAD.iter().copied().chain(repeat(b' '));
            while started.elapsed() < SILENT_HOLD {
                thread::sleep(TRICKLE_GAP);
                let byte = bytes.next().expect("endless");
                if stream.write_all(&[byte]).is_err() {
                    return;
                }
            }
        });
        let catalog = test_catalog(&addr);
        finish(
            "list_tables",
            catalog
                .post_json("list_tables", json!({}))
                .expect("first request"),
        )
        .expect("first answer");
        assert_unavailable_within(
            || {
                catalog
                    .post_json("query_sql", json!({}))
                    .and_then(|r| read_json("query_sql", r))
            },
            TEST_TOTAL_TIMEOUT,
        );
        server.join().expect("server thread");
    }

    /// An answer read to its end leaves its connection for the next request,
    /// as an append's answer is read: the endpoint serves both requests over
    /// the one connection it accepts.
    #[test]
    fn an_answer_read_to_its_end_leaves_its_connection_for_the_next_request() {
        let (addr, server) = serve_one_then(|mut stream| {
            read_request(&mut stream);
            stream.write_all(OK_RESPONSE).expect("second answer");
        });
        let catalog = test_catalog(&addr);
        for op in ["append", "list_tables"] {
            let response = catalog.post_json(op, json!({})).expect("request");
            finish(op, response).expect("answer");
        }
        server
            .join()
            .expect("both requests came over one connection");
    }

    /// An endpoint that reads the request and closes the connection is
    /// unavailable, not a failure of the request.
    #[test]
    fn a_connection_closed_before_the_answer_is_unavailable() {
        let (addr, server) = serve_one_then(|mut stream| {
            read_request(&mut stream);
        });
        let catalog = test_catalog(&addr);
        finish(
            "list_tables",
            catalog
                .post_json("list_tables", json!({}))
                .expect("first request"),
        )
        .expect("first answer");
        assert_unavailable_within(|| catalog.post_json("append", json!({})), Duration::ZERO);
        server.join().expect("server thread");
    }
}
