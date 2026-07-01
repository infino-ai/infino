// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared fake-gcs-server helpers for the GCS wire-protocol smoke test.
//!
//! Assumes fake-gcs-server is reachable at [`EMULATOR_ENDPOINT`]. Start it:
//!   docker run -d --rm -p 4443:4443 fsouza/fake-gcs-server \
//!     -scheme http -public-host 127.0.0.1:4443
//!
//! Buckets are created explicitly via the storage JSON API so an
//! empty-bucket open behaves like real GCS. The request body is a JSON
//! string literal (reqwest is built without its `json` feature here).

/// fake-gcs-server HTTP endpoint (plain HTTP; the smoke skips signing).
pub const EMULATOR_ENDPOINT: &str = "http://127.0.0.1:4443";

/// Create `bucket` via fake-gcs-server's storage JSON API. Idempotent:
/// a 409 (already exists) is treated as success. Panics with a start-it
/// hint if the emulator is unreachable.
pub async fn ensure_emulator_bucket(bucket: &str) {
    let client = reqwest::Client::new();
    let url = format!("{EMULATOR_ENDPOINT}/storage/v1/b?project=infino-test");
    let body = format!(r#"{{"name":"{bucket}"}}"#);
    let resp = client
        .post(&url)
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap_or_else(|e| {
            panic!(
                "fake-gcs-server not reachable at {EMULATOR_ENDPOINT}. Start it with:\n  \
                 docker run -d --rm -p 4443:4443 fsouza/fake-gcs-server \
                 -scheme http -public-host 127.0.0.1:4443\ncause: {e}"
            )
        });
    let status = resp.status();
    assert!(
        status.is_success() || status.as_u16() == 409,
        "create bucket {bucket} failed: {status}"
    );
}

/// Delete `bucket` on success; best-effort (the emulator is disposable).
pub async fn delete_emulator_bucket(bucket: &str) {
    let client = reqwest::Client::new();
    let _ = client
        .delete(format!("{EMULATOR_ENDPOINT}/storage/v1/b/{bucket}"))
        .send()
        .await;
}
