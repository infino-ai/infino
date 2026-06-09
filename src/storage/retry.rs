// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared transient-retry + range-completion helpers for the
//! object-store-backed providers. One copy so retry semantics can't
//! drift between backends; each backend keeps its own error
//! translation and feeds already-translated results in.

use std::ops::Range;
use std::time::Duration;

use bytes::{Bytes, BytesMut};

use super::StorageError;

/// `object_store` retry depth. Deeper than the library default, which
/// exhausts before a flaky/high-latency connection recovers.
pub(crate) const MAX_RETRIES: usize = 20;

/// Overall `object_store` retry window, paired with [`MAX_RETRIES`].
pub(crate) const RETRY_TIMEOUT: Duration = Duration::from_secs(300);

/// Transient re-issue backoff: `BASE × 2^min(attempt, MAX_SHIFT)` ms, capped.
const BACKOFF_BASE_MS: u64 = 50;
const BACKOFF_MAX_SHIFT: u32 = 5;
const BACKOFF_CAP_MS: u64 = 2000;

/// App-level re-issue budget for transient transport failures that
/// `object_store` won't retry itself (e.g. "error sending request" on a
/// socket the service dropped under us).
const MAX_TRANSIENT_RETRIES: u32 = 8;

/// Retry budget applied to a store builder via `.with_retry(...)`.
pub(crate) fn config() -> object_store::RetryConfig {
    object_store::RetryConfig {
        max_retries: MAX_RETRIES,
        retry_timeout: RETRY_TIMEOUT,
        ..Default::default()
    }
}

/// Transient flakiness worth re-issuing an idempotent op for. Stable
/// errors (NotFound / PreconditionFailed / Permanent) are not.
fn is_retryable(err: &StorageError) -> bool {
    matches!(err, StorageError::TransientExhausted { .. })
}

/// Exponential backoff (50ms→2s) to drain a dead pooled connection
/// before a fresh dial.
fn backoff(attempt: u32) -> Duration {
    let ms = BACKOFF_BASE_MS.saturating_mul(1 << attempt.min(BACKOFF_MAX_SHIFT));
    Duration::from_millis(ms.min(BACKOFF_CAP_MS))
}

/// Permanent error: the object returned fewer bytes than requested and
/// made no progress. Stable, so callers don't retry.
fn short_read(uri: &str, start: u64, requested: u64, got: u64) -> StorageError {
    let source: Box<dyn std::error::Error + Send + Sync> = format!(
        "get_range short read: object returned {got} of {requested} bytes from offset {start}"
    )
    .into();
    StorageError::Permanent {
        uri: uri.into(),
        source,
    }
}

/// Re-issue an idempotent whole-object op (get / tail) through the
/// app-level transient budget. `op` must already map to `StorageError`.
pub(crate) async fn with_reissue<T, F, Fut>(mut op: F) -> Result<T, StorageError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, StorageError>>,
{
    let mut attempt = 0u32;
    loop {
        match op().await {
            Ok(v) => return Ok(v),
            Err(e) if is_retryable(&e) && attempt < MAX_TRANSIENT_RETRIES => {
                tokio::time::sleep(backoff(attempt)).await;
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Range-fetch with short-read completion + transient re-issue.
///
/// A GET can return short (truncated body) or fail transiently without
/// `object_store` retrying it. Both corrupt callers (over-slice /
/// zero-gap cache fill), so re-issue the still-missing tail; a fresh
/// dial also drops the dead socket. `fetch` performs one range GET,
/// already translated to `StorageError`.
pub(crate) async fn complete_range<F, Fut>(
    uri: &str,
    range: Range<u64>,
    mut fetch: F,
) -> Result<Bytes, StorageError>
where
    F: FnMut(Range<u64>) -> Fut,
    Fut: std::future::Future<Output = Result<Bytes, StorageError>>,
{
    let want = range.end.saturating_sub(range.start);
    if want == 0 {
        return Ok(Bytes::new());
    }
    let mut cursor = range.start;
    let mut filled: u64 = 0;
    let mut parts: Vec<Bytes> = Vec::new();
    let mut attempt = 0u32;
    loop {
        let chunk = match fetch(cursor..range.end).await {
            Ok(c) => c,
            Err(e) if is_retryable(&e) && attempt < MAX_TRANSIENT_RETRIES => {
                tokio::time::sleep(backoff(attempt)).await;
                attempt += 1;
                continue;
            }
            Err(e) => return Err(e),
        };
        if chunk.is_empty() {
            // Empty body for an in-bounds range is a transport glitch,
            // not end-of-object (that surfaces as a typed error).
            if attempt < MAX_TRANSIENT_RETRIES {
                tokio::time::sleep(backoff(attempt)).await;
                attempt += 1;
                continue;
            }
            return Err(short_read(uri, range.start, want, filled));
        }
        let take = (chunk.len() as u64).min(want - filled);
        filled += take;
        cursor += take;
        if take as usize == chunk.len() {
            parts.push(chunk);
        } else {
            parts.push(chunk.slice(0..take as usize));
        }
        if filled >= want {
            break;
        }
        // Short non-empty chunks are normal for a large range; `filled`
        // advances each iteration so the loop is bounded.
    }
    if parts.len() == 1 {
        return Ok(parts.pop().expect("len checked == 1"));
    }
    let mut out = BytesMut::with_capacity(want as usize);
    for p in &parts {
        out.extend_from_slice(p);
    }
    Ok(out.freeze())
}
