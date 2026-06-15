// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared Azure emulator helpers used by the Azurite-backed integration
//! tests.
//!
//! These utilities sign and issue the two container-level management
//! requests (create / delete) that the `object_store` crate does not
//! expose.  Everything here is specific to the fixed
//! `devstoreaccount1` emulator credentials that Azurite ships with.

use base64::Engine;
use sha2::{Digest, Sha256};

/// Azurite blob endpoint for the well-known `devstoreaccount1` account.
pub const EMULATOR_ENDPOINT: &str = "http://127.0.0.1:10000/devstoreaccount1";
pub const EMULATOR_ACCOUNT: &str = "devstoreaccount1";
// ggignore: this is Azurite's public, documented emulator key, not a secret.
pub const EMULATOR_KEY: &str =
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="; // ggignore
pub const STORAGE_API_VERSION: &str = "2021-08-06";

/// HMAC-SHA256 over `msg` with `key`, built directly on `Sha256` to
/// avoid a dependency on the pre-release `hmac` crate.
fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut block_key = if key.len() > BLOCK {
        Sha256::digest(key).to_vec()
    } else {
        key.to_vec()
    };
    block_key.resize(BLOCK, 0);

    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] ^= block_key[i];
        opad[i] ^= block_key[i];
    }

    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(msg);
    let inner = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner);
    outer.finalize().into()
}

/// Issue a SharedKey-signed container-level REST request against
/// Azurite. `object_store` has no container create/delete call and the
/// Azurite image ships no CLI, so tests sign these by hand.
async fn signed_container_request(
    method: reqwest::Method,
    container: &str,
) -> reqwest::Result<reqwest::Response> {
    let date = httpdate::fmt_http_date(std::time::SystemTime::now());
    let canonical_headers = format!("x-ms-date:{date}\nx-ms-version:{STORAGE_API_VERSION}\n");
    // The emulator URL path is `/devstoreaccount1/<container>`, and the
    // canonicalized resource is `/<account>` + that path — so the
    // account name appears twice.
    let canonical_resource =
        format!("/{EMULATOR_ACCOUNT}/{EMULATOR_ACCOUNT}/{container}\nrestype:container");
    // VERB + 11 empty header fields (Content-*, Date, If-*, Range),
    // then the canonicalized headers and resource.
    let string_to_sign = format!(
        "{method}\n{}{canonical_headers}{canonical_resource}",
        "\n".repeat(11)
    );

    let key = base64::engine::general_purpose::STANDARD
        .decode(EMULATOR_KEY)
        .expect("decode emulator key");
    let signature = base64::engine::general_purpose::STANDARD
        .encode(hmac_sha256(&key, string_to_sign.as_bytes()));

    let url = format!("{EMULATOR_ENDPOINT}/{container}?restype=container");
    reqwest::Client::new()
        .request(method, &url)
        .header("x-ms-date", &date)
        .header("x-ms-version", STORAGE_API_VERSION)
        .header("content-length", "0")
        .header(
            "authorization",
            format!("SharedKey {EMULATOR_ACCOUNT}:{signature}"),
        )
        .send()
        .await
}

/// Create the run's container. Treats 201 Created and 409 Conflict
/// (already exists) as success.
pub async fn ensure_emulator_container(container: &str) {
    let resp = match signed_container_request(reqwest::Method::PUT, container).await {
        Ok(resp) => resp,
        // A connect failure almost always means Azurite is not running.
        Err(e) if e.is_connect() => panic!(
            "Azurite is not reachable at {EMULATOR_ENDPOINT}. Start it with:\n  \
             docker run -d --rm -p 10000:10000 \
             mcr.microsoft.com/azure-storage/azurite \
             azurite-blob --blobHost 0.0.0.0\n\
             then re-run with INFINO_TEST_AZURE=1. (underlying error: {e})"
        ),
        Err(e) => panic!("create-container request failed: {e}"),
    };

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    assert!(
        status.is_success() || status.as_u16() == 409,
        "create container {container} failed: {status}\n{body}"
    );
}

/// Best-effort teardown of the run's container (202 Accepted, or 404
/// if already gone). Runs only on the success path — a failing run
/// leaves the container for inspection, and Azurite is disposable.
pub async fn delete_emulator_container(container: &str) {
    match signed_container_request(reqwest::Method::DELETE, container).await {
        Ok(resp) => {
            let status = resp.status();
            assert!(
                status.is_success() || status.as_u16() == 404,
                "delete container {container} failed: {status}"
            );
        }
        Err(e) => eprintln!("[azure] container cleanup skipped: {e}"),
    }
}
