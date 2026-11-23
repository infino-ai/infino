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

use std::{io::Read, sync::Arc, time::Duration};

use arrow_array::RecordBatch;
use arrow_schema::{Schema, SchemaRef};
use serde_json::{Value, json};

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

/// Longest wait to open the TCP and TLS connection to the endpoint.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Default deadline for a request, from connecting to the last byte of the
/// answer; `ConnectOptions::with_request_timeout` overrides it.
/// A hosted request can run up to 600 s, and an append, update or delete
/// commits inside its request and answers after the commit. The client waits
/// a little longer than that, so it never gives up on a write that still
/// commits: a retry of that write would apply it twice.
/// ureq 2 uses what is left of it as the timeout of each read and write, in
/// place of `timeout_read` and `timeout_write`. It does not cover the DNS
/// lookup.
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(620);

/// The HTTP client for a hosted connection. Without these bounds, a request on
/// a connection that stops making progress waits forever, and so does every
/// caller behind it. ureq 2 applies the deadline to a request body only on a
/// new connection, so an agent built with `pooled` false opens one for each
/// request.
fn agent(connect: Duration, total: Duration, pooled: bool) -> ureq::Agent {
    let builder = ureq::AgentBuilder::new()
        .timeout_connect(connect)
        .timeout(total);
    if pooled {
        builder.build()
    } else {
        builder.max_idle_connections(0).build()
    }
}

/// A connection to a hosted endpoint. Holds the sync HTTP client, the base URL
/// (scheme + host), the target database, and the bearer credential.
pub(crate) struct RemoteCatalog {
    agent: ureq::Agent,
    /// For Arrow bodies, which can be large: a new connection per request.
    arrow_agent: ureq::Agent,
    base_url: String,
    database: String,
    api_key: String,
}

impl RemoteCatalog {
    /// Build a hosted connection. The key comes from `api_key` or, when unset,
    /// the `INFINO_API_KEY` environment variable; a hosted connection without a
    /// key is a configuration error. `request_timeout` is the deadline of each
    /// request ([`DEFAULT_REQUEST_TIMEOUT`] when unset); zero is a
    /// configuration error.
    pub(crate) fn new(
        base_url: String,
        database: String,
        api_key: Option<String>,
        request_timeout: Option<Duration>,
    ) -> Result<Self, InfinoError> {
        let request_timeout = request_timeout.unwrap_or(DEFAULT_REQUEST_TIMEOUT);
        if request_timeout.is_zero() {
            return Err(InfinoError::Config(
                "the request timeout of a hosted connection must be greater than zero".to_string(),
            ));
        }
        let api_key = api_key
            .or_else(|| std::env::var(API_KEY_ENV).ok())
            .filter(|k| !k.is_empty())
            .ok_or_else(|| {
                InfinoError::Config(format!(
                    "a hosted connection needs an API key (pass ConnectOptions::with_api_key or set {API_KEY_ENV})"
                ))
            })?;
        Ok(Self {
            agent: agent(CONNECT_TIMEOUT, request_timeout, true),
            arrow_agent: agent(CONNECT_TIMEOUT, request_timeout, false),
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
    pub(crate) fn post_json(&self, op: &str, body: Value) -> Result<ureq::Response, InfinoError> {
        let request = self
            .agent
            .post(&self.url(op))
            .set("Authorization", &self.bearer())
            .set("Accept", wire::ARROW_STREAM_CONTENT_TYPE);
        map_send(op, request.send_json(body))
    }

    /// POST an Arrow-IPC body with query parameters — the `append` path.
    pub(crate) fn post_arrow(
        &self,
        op: &str,
        query: &[(&str, &str)],
        body: Vec<u8>,
    ) -> Result<ureq::Response, InfinoError> {
        self.post_arrow_with_headers(op, query, &[], body)
    }

    /// [`Self::post_arrow`] with extra request headers.
    pub(crate) fn post_arrow_with_headers(
        &self,
        op: &str,
        query: &[(&str, &str)],
        headers: &[(&str, &str)],
        body: Vec<u8>,
    ) -> Result<ureq::Response, InfinoError> {
        let mut request = self
            .arrow_agent
            .post(&self.url(op))
            .set("Authorization", &self.bearer())
            .set("Content-Type", wire::ARROW_STREAM_CONTENT_TYPE);
        for (key, value) in query {
            request = request.query(key, value);
        }
        for (name, value) in headers {
            request = request.set(name, value);
        }
        map_send(op, request.send_bytes(&body))
    }

    /// Register the database this connection targets on the hosted service.
    /// `POST /v1/databases` with `{name}` — this endpoint is account-scoped
    /// (the account is identified by the API key), so unlike the per-database
    /// operations its path carries no database segment. A `201` is success; a
    /// `409` (already registered) surfaces as [`InfinoError::AlreadyExists`].
    pub(crate) fn create_database(&self) -> Result<(), InfinoError> {
        let url = format!("{}/v1/databases", self.base_url);
        let request = self.agent.post(&url).set("Authorization", &self.bearer());
        map_send(
            "create_database",
            request.send_json(json!({ "name": self.database })),
        )?;
        Ok(())
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
        self.post_json("create_table", body)?;
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
        self.post_json("drop_table", json!({ "table_name": name, "purge": purge }))?;
        Ok(())
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
fn read_schema_document(op: &str, response: ureq::Response) -> Result<TableSchema, InfinoError> {
    let value = read_json(op, response)?;
    TableSchema::from_json(&value)
        .map_err(|e| InfinoError::Backend(format!("{op}: parsing the schema document: {e}")))
}

/// Turn a `ureq` send result into an [`InfinoError`]: an HTTP error status maps
/// through [`wire::status_to_error`]; a transport failure (a timeout, a
/// connection that failed or broke) is `Unavailable`.
fn map_send(
    op: &str,
    result: Result<ureq::Response, ureq::Error>,
) -> Result<ureq::Response, InfinoError> {
    match result {
        Ok(response) => Ok(response),
        Err(ureq::Error::Status(code, response)) => {
            let body = response.into_string().unwrap_or_default();
            Err(wire::status_to_error(op, code, &body))
        }
        Err(ureq::Error::Transport(transport)) => Err(InfinoError::Unavailable(format!(
            "{op}: transport error: {transport}"
        ))),
    }
}

/// Read a JSON response body.
pub(crate) fn read_json(op: &str, response: ureq::Response) -> Result<Value, InfinoError> {
    response
        .into_json::<Value>()
        .map_err(|e| InfinoError::Backend(format!("{op}: parsing response: {e}")))
}

/// Read an Arrow-IPC response body into record batches.
pub(crate) fn read_arrow(
    op: &str,
    response: ureq::Response,
) -> Result<Vec<RecordBatch>, InfinoError> {
    let mut buf = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut buf)
        .map_err(|e| InfinoError::Backend(format!("{op}: reading response: {e}")))?;
    wire::ipc_to_batches(&buf)
}

/// Read an Arrow-IPC response body into the schema it carries. The body is a
/// schema message with no batches, so this cannot go through
/// [`read_arrow`], which expects at least one batch to describe.
pub(crate) fn read_arrow_schema(op: &str, response: ureq::Response) -> Result<Schema, InfinoError> {
    let mut buf = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut buf)
        .map_err(|e| InfinoError::Backend(format!("{op}: reading response: {e}")))?;
    wire::ipc_to_schema(&buf)
}

#[cfg(test)]
mod tests {
    use std::{
        fmt::Debug,
        io::{Read, Write, copy, sink},
        iter::repeat,
        net::{TcpListener, TcpStream},
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };

    use serde_json::json;

    use super::{RemoteCatalog, read_json};
    use crate::InfinoError;

    /// The request deadline of the test client.
    const TEST_TIMEOUT: Duration = Duration::from_millis(300);
    /// How long a stalled endpoint holds the connection before closing it.
    /// A client with no bound waits this long, past [`LATEST_RETURN`].
    const HOLD: Duration = Duration::from_secs(5);
    /// The latest a request may return for its deadline to be what ended it.
    /// A stalled body write can take twice the deadline: one write that
    /// sends part of its bytes, then one that sends none.
    const LATEST_RETURN: Duration = Duration::from_secs(4);
    /// A complete answer to any request.
    const OK: &[u8] =
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n[]";
    /// The head of an answer whose body is far longer than the test lasts.
    const ENDLESS_HEAD: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 1000000\r\n\r\n";
    /// The pause between the bytes of the answer that never ends.
    const TRICKLE_GAP: Duration = Duration::from_millis(100);
    /// An append body larger than the socket buffers can hold, so the write
    /// blocks once the endpoint stops reading.
    const LARGE_BODY: usize = 64 << 20;

    fn catalog(addr: &str) -> RemoteCatalog {
        RemoteCatalog::new(
            format!("http://{addr}"),
            "db".to_string(),
            Some("ik_test".to_string()),
            Some(TEST_TIMEOUT),
        )
        .expect("catalog")
    }

    /// A zero deadline would fail every request at once; it is refused when
    /// the connection is built.
    #[test]
    fn a_zero_request_timeout_is_refused() {
        let result = RemoteCatalog::new(
            "http://127.0.0.1:1".to_string(),
            "db".to_string(),
            Some("ik_test".to_string()),
            Some(Duration::ZERO),
        );
        assert!(matches!(result, Err(InfinoError::Config(_))));
    }

    /// Read one request: the head, then the body its Content-Length names.
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
            .find_map(|line| line.strip_prefix("content-length:"))
            .map_or(0, |n| n.trim().parse().expect("content-length"));
        copy(&mut stream.take(length), &mut sink()).expect("request body");
    }

    /// Answer the first request on the first connection, then hand that
    /// connection to `then`. The listener stays open while `then` runs, so a
    /// new connection is taken by the system but nobody reads it.
    fn serve_one_then(then: impl FnOnce(TcpStream) + Send + 'static) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr").to_string();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            read_request(&mut stream);
            stream.write_all(OK).expect("answer");
            then(stream);
            drop(listener);
        });
        (addr, server)
    }

    /// `call` fails as `Unavailable`, after the deadline and well before the
    /// endpoint lets go.
    fn assert_unavailable_in_time<T: Debug>(call: impl FnOnce() -> Result<T, InfinoError>) {
        let start = Instant::now();
        let result = call();
        let waited = start.elapsed();
        assert!(waited < LATEST_RETURN, "waited {waited:?}: {result:?}");
        assert!(
            waited >= TEST_TIMEOUT,
            "returned after {waited:?}: {result:?}"
        );
        assert!(
            matches!(result, Err(InfinoError::Unavailable(_))),
            "{result:?}"
        );
    }

    /// A request to an endpoint that accepts the connection and then sends
    /// nothing ends at the deadline.
    #[test]
    fn a_request_to_a_silent_endpoint_times_out() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr").to_string();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept");
            thread::sleep(HOLD);
            drop(stream);
        });
        let catalog = catalog(&addr);
        assert_unavailable_in_time(|| catalog.post_json("append", json!({})));
        server.join().expect("server thread");
    }

    /// ureq 2 clears the timeouts of a connection it puts back in the pool;
    /// the deadline still ends a request that reuses it and gets no answer.
    #[test]
    fn a_silent_answer_on_a_reused_connection_times_out() {
        let (addr, server) = serve_one_then(|mut stream| {
            read_request(&mut stream);
            thread::sleep(HOLD);
        });
        let catalog = catalog(&addr);
        read_json(
            "list_tables",
            catalog.post_json("list_tables", json!({})).expect("first"),
        )
        .expect("first answer");
        assert_unavailable_in_time(|| catalog.post_json("append", json!({})));
        server.join().expect("server thread");
    }

    /// A body write is bounded only on a new connection. An append after an
    /// append opens one, so a large body to an endpoint that stopped reading
    /// ends in time.
    #[test]
    fn a_large_append_to_an_endpoint_that_stopped_reading_times_out() {
        let (addr, server) = serve_one_then(|stream| {
            thread::sleep(HOLD);
            drop(stream);
        });
        let catalog = catalog(&addr);
        catalog
            .post_arrow("append", &[("table", "t")], Vec::new())
            .expect("first append");
        assert_unavailable_in_time(|| {
            catalog.post_arrow("append", &[("table", "t")], vec![0u8; LARGE_BODY])
        });
        server.join().expect("server thread");
    }

    /// An answer that keeps sending a byte at a time never trips a read
    /// timeout; the deadline ends it.
    #[test]
    fn an_answer_that_never_ends_times_out() {
        let (addr, server) = serve_one_then(|mut stream| {
            read_request(&mut stream);
            let start = Instant::now();
            let mut bytes = ENDLESS_HEAD.iter().copied().chain(repeat(b' '));
            while start.elapsed() < HOLD {
                thread::sleep(TRICKLE_GAP);
                if stream.write_all(&[bytes.next().expect("byte")]).is_err() {
                    return;
                }
            }
        });
        let catalog = catalog(&addr);
        read_json(
            "list_tables",
            catalog.post_json("list_tables", json!({})).expect("first"),
        )
        .expect("first answer");
        assert_unavailable_in_time(|| catalog.post_json("query_sql", json!({})));
        server.join().expect("server thread");
    }
}
