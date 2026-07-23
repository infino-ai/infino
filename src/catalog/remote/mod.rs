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

use std::{io::Read, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use serde_json::{Value, json};

use crate::{IndexSpec, InfinoError, Supertable};

/// Environment variable consulted for the API key when `ConnectOptions` does
/// not carry one.
const API_KEY_ENV: &str = "INFINO_API_KEY";

/// A connection to a hosted endpoint. Holds the sync HTTP client, the base URL
/// (scheme + host), the target database, and the bearer credential.
pub(crate) struct RemoteCatalog {
    agent: ureq::Agent,
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
            agent: ureq::agent(),
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
        let mut request = self
            .agent
            .post(&self.url(op))
            .set("Authorization", &self.bearer())
            .set("Content-Type", wire::ARROW_STREAM_CONTENT_TYPE);
        for (key, value) in query {
            request = request.query(key, value);
        }
        map_send(op, request.send_bytes(&body))
    }

    pub(crate) fn create_table(
        self: &Arc<Self>,
        name: &str,
        schema: SchemaRef,
        indexes: IndexSpec,
    ) -> Result<Supertable, InfinoError> {
        let body = json!({
            "table_name": name,
            "schema": wire::schema_to_json(&schema)?,
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
        let response = self.post_json("schema", json!({ "table_name": name }))?;
        let value = read_json("schema", response)?;
        let fields = value.as_array().ok_or_else(|| {
            InfinoError::Backend("schema response was not a JSON array".to_string())
        })?;
        let schema = Arc::new(wire::json_to_schema(fields)?);
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
}

/// Turn a `ureq` send result into an [`InfinoError`]: an HTTP error status maps
/// through [`wire::status_to_error`]; a transport failure is a `Backend` error.
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
        Err(ureq::Error::Transport(transport)) => Err(InfinoError::Backend(format!(
            "{op}: transport error: {transport}"
        ))),
    }
}

/// Read a JSON response body.
fn read_json(op: &str, response: ureq::Response) -> Result<Value, InfinoError> {
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
