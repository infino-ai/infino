// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared transient-retry + range-completion helpers for the
//! object-store-backed providers. One copy so retry semantics can't
//! drift between backends; each backend keeps its own error
//! translation and feeds already-translated results in.

use std::{error::Error, future::Future, ops::Range, time::Duration};

use bytes::{Bytes, BytesMut};
use object_store::{BackoffConfig, RetryConfig};
use tokio::time;
use tracing::warn;

use super::{ObjectMeta, StorageError};

/// `object_store` retry depth. Deeper than the library default, which
/// exhausts before a flaky/high-latency connection recovers.
pub(crate) const MAX_RETRIES: usize = 20;

/// Overall `object_store` retry window, paired with [`MAX_RETRIES`].
///
/// This is the budget the retry loop spends across *all* attempts, so it
/// must stay strictly larger than any single backend's per-request timeout
/// (`AZURE_REQUEST_TIMEOUT`, `GCS_REQUEST_TIMEOUT`). If a request's own
/// timeout equalled this budget, one stalled request would consume the
/// entire window and no retry would ever fire — the stall would surface as
/// a fatal error after a single attempt. Paired with a 60s per-request
/// timeout this leaves room for roughly five attempts while capping the
/// worst-case total hang at 300s.
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
pub(crate) fn config() -> RetryConfig {
    RetryConfig {
        max_retries: MAX_RETRIES,
        retry_timeout: RETRY_TIMEOUT,
        // Spelled out rather than left to `..Default::default()` so the
        // curve is visible at the call site. These values match
        // object_store's own defaults (100ms initial, doubling, capped at
        // 15s), so this is behaviour-preserving. object_store applies
        // jitter on top: each sleep is a random value in
        // `[prev, prev * base)`, so retries don't synchronize into a
        // thundering herd.
        backoff: BackoffConfig {
            init_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(15),
            base: 2.0,
        },
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
    let source: Box<dyn Error + Send + Sync> =
        format!("short read: object returned {got} of {requested} bytes from offset {start}")
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
    Fut: Future<Output = Result<T, StorageError>>,
{
    let mut attempt = 0u32;
    loop {
        match op().await {
            Ok(v) => return Ok(v),
            Err(e) if is_retryable(&e) && attempt < MAX_TRANSIENT_RETRIES => {
                warn!(attempt, error = %e, "transient object-store error; re-issuing");
                time::sleep(backoff(attempt)).await;
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Re-issue a create-only PUT through the app-level transient budget,
/// recognising an earlier issue of its own that landed.
///
/// A create is not idempotent: a request that timed out after the store
/// accepted it has landed, and its re-issue then fails the precondition
/// against its own object. (`object_store` declines to retry a create on a
/// timeout for exactly this reason; this layer re-issues it anyway.) So once
/// an issue has failed transiently, a `PreconditionFailed` on a later issue
/// is ambiguous, and `read_back` settles it: an object holding exactly
/// `bytes` is the landed write, reported as created with its own CAS token;
/// anything else is a real lost race and surfaces as `PreconditionFailed`.
///
/// A `PreconditionFailed` on the first issue surfaces without a read-back:
/// nothing of ours can be there yet, so the key belongs to someone else even
/// when they wrote the same bytes, and the contention path costs no GET.
/// `create` performs one create-only PUT and `read_back` one whole-object
/// GET, both already translated to `StorageError`.
pub(crate) async fn create_with_reissue<C, CFut, R, RFut>(
    uri: &str,
    bytes: &Bytes,
    mut create: C,
    read_back: R,
) -> Result<Option<String>, StorageError>
where
    C: FnMut() -> CFut,
    CFut: Future<Output = Result<Option<String>, StorageError>>,
    R: FnOnce() -> RFut,
    RFut: Future<Output = Result<(Bytes, ObjectMeta), StorageError>>,
{
    let mut issued = 0u32;
    let created = with_reissue(|| {
        issued += 1;
        create()
    })
    .await;
    let lost = match created {
        Err(lost @ StorageError::PreconditionFailed { .. }) if issued > 1 => lost,
        other => return other,
    };
    match read_back().await {
        Ok((existing, meta)) if existing == *bytes => Ok(meta.etag),
        Ok(_) => Err(lost),
        Err(e) => {
            // Without the object we cannot tell our write from a peer's, so
            // report what the store said rather than claim a create.
            warn!(uri, error = %e, "could not read back a re-issued create; reporting the precondition");
            Err(lost)
        }
    }
}

/// Whole-object GET with truncated-body detection + transient re-issue.
///
/// `object_store`'s streamed body can come back short: a socket dropped
/// mid-transfer surfaces as a *successful* but truncated `Bytes`, not an
/// error — the same hazard [`complete_range`] guards the range path against.
/// A truncated whole-object body silently corrupts every caller that hashes
/// or parses it (an OPANN routing page fails its content-hash check; a
/// manifest part or footer fails to parse). `fetch` performs one GET and
/// returns the body paired with the object's declared size from the *same*
/// response; this re-issues a fresh GET (a new dial also drops the dead
/// socket) whenever the body is shorter than the declared size — or a
/// transient error fires — and fails with a stable short-read error once the
/// budget is spent, rather than hand back a partial object as success.
pub(crate) async fn complete_get<F, Fut>(
    uri: &str,
    mut fetch: F,
) -> Result<(Bytes, ObjectMeta), StorageError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<(Bytes, ObjectMeta), StorageError>>,
{
    let mut attempt = 0u32;
    loop {
        let (bytes, meta) = match fetch().await {
            Ok(v) => v,
            Err(e) if is_retryable(&e) && attempt < MAX_TRANSIENT_RETRIES => {
                time::sleep(backoff(attempt)).await;
                attempt += 1;
                continue;
            }
            Err(e) => return Err(e),
        };
        let got = bytes.len() as u64;
        if got == meta.size {
            return Ok((bytes, meta));
        }
        // A body shorter than the response's own declared size is a truncated
        // read delivered as success. Re-issue rather than return it — a
        // partial object would fail a downstream content-hash check or parse.
        if attempt < MAX_TRANSIENT_RETRIES {
            time::sleep(backoff(attempt)).await;
            attempt += 1;
            continue;
        }
        return Err(short_read(uri, 0, meta.size, got));
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
    Fut: Future<Output = Result<Bytes, StorageError>>,
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
                warn!(uri, attempt, error = %e, "transient range GET error; re-issuing tail");
                time::sleep(backoff(attempt)).await;
                attempt += 1;
                continue;
            }
            Err(e) => return Err(e),
        };
        if chunk.is_empty() {
            // Empty body for an in-bounds range is a transport glitch,
            // not end-of-object (that surfaces as a typed error).
            if attempt < MAX_TRANSIENT_RETRIES {
                time::sleep(backoff(attempt)).await;
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

#[cfg(test)]
mod tests {
    use std::{cell::Cell, sync::Arc};

    use tempfile::TempDir;

    use super::*;
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        test_helpers::fault_storage::{FaultKind, FaultOp, FaultStorage},
    };

    fn transient() -> StorageError {
        StorageError::TransientExhausted {
            uri: "u".into(),
            source: "boom".into(),
        }
    }

    fn not_found() -> StorageError {
        StorageError::NotFound { uri: "u".into() }
    }

    #[test]
    fn backoff_grows_then_caps_at_max_shift() {
        assert_eq!(backoff(0), Duration::from_millis(BACKOFF_BASE_MS));
        assert_eq!(backoff(1), Duration::from_millis(BACKOFF_BASE_MS * 2));
        // attempt >= BACKOFF_MAX_SHIFT saturates the shift.
        let capped = Duration::from_millis(BACKOFF_BASE_MS * (1 << BACKOFF_MAX_SHIFT));
        assert_eq!(backoff(BACKOFF_MAX_SHIFT), capped);
        assert_eq!(backoff(100), capped);
    }

    #[test]
    fn config_uses_our_budget() {
        let c = config();
        assert_eq!(c.max_retries, MAX_RETRIES);
        assert_eq!(c.retry_timeout, RETRY_TIMEOUT);
    }

    #[test]
    fn is_retryable_only_for_transient() {
        assert!(is_retryable(&transient()));
        assert!(!is_retryable(&not_found()));
        assert!(!is_retryable(&StorageError::PreconditionFailed {
            uri: "u".into()
        }));
    }

    #[test]
    fn short_read_builds_permanent_error() {
        let e = short_read("u", 0, 100, 10);
        assert!(matches!(e, StorageError::Permanent { .. }));
        assert!(e.to_string().contains("short read"));
    }

    #[tokio::test(start_paused = true)]
    async fn reissue_ok_first_try() {
        let calls = Cell::new(0u32);
        let r: Result<u8, StorageError> = with_reissue(|| {
            calls.set(calls.get() + 1);
            async { Ok(7u8) }
        })
        .await;
        assert_eq!(r.expect("test"), 7);
        assert_eq!(calls.get(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn reissue_retries_transient_then_succeeds() {
        let calls = Cell::new(0u32);
        let r: Result<u8, StorageError> = with_reissue(|| {
            let c = calls.get();
            calls.set(c + 1);
            async move { if c < 2 { Err(transient()) } else { Ok(7u8) } }
        })
        .await;
        assert_eq!(r.expect("test"), 7);
        assert_eq!(calls.get(), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn reissue_exhausts_budget_then_errors() {
        let calls = Cell::new(0u32);
        let r: Result<u8, StorageError> = with_reissue(|| {
            calls.set(calls.get() + 1);
            async { Err(transient()) }
        })
        .await;
        assert!(r.is_err());
        assert_eq!(calls.get(), MAX_TRANSIENT_RETRIES + 1);
    }

    #[tokio::test(start_paused = true)]
    async fn reissue_non_retryable_returns_immediately() {
        let calls = Cell::new(0u32);
        let r: Result<u8, StorageError> = with_reissue(|| {
            calls.set(calls.get() + 1);
            async { Err(not_found()) }
        })
        .await;
        assert!(matches!(r, Err(StorageError::NotFound { .. })));
        assert_eq!(calls.get(), 1);
    }

    // ---- create_with_reissue ---------------------------------------------

    /// The key every create below targets.
    const CREATE_URI: &str = "data/seg-00000001.sf.parquet";
    /// The bytes the create writes.
    const OURS: &[u8] = b"the bytes this create writes";
    /// A peer's different bytes at the same key.
    const PEERS: &[u8] = b"a peer's bytes";

    /// Fault storage over a fresh local directory.
    fn fault_store(dir: &TempDir) -> Arc<FaultStorage> {
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local provider"));
        FaultStorage::wrap(local)
    }

    /// Create [`OURS`] at [`CREATE_URI`] the way the object-store providers
    /// do, counting read-backs in `read_backs`.
    async fn create_ours(
        faults: &FaultStorage,
        read_backs: &Cell<u32>,
    ) -> Result<Option<String>, StorageError> {
        let bytes = Bytes::from_static(OURS);
        create_with_reissue(
            CREATE_URI,
            &bytes,
            || faults.put_atomic(CREATE_URI, bytes.clone()),
            || {
                read_backs.set(read_backs.get() + 1);
                faults.get(CREATE_URI)
            },
        )
        .await
    }

    /// The hazard `create_with_reissue` exists for: a create that timed out
    /// after landing, re-issued as a plain transient retry, meets its own
    /// object and reports a lost race.
    #[tokio::test]
    async fn a_bare_reissue_of_a_landed_create_meets_its_own_object() {
        let dir = TempDir::new().expect("tempdir");
        let faults = fault_store(&dir);
        faults.fail_with(
            FaultKind::TimedOutAfterLanding,
            FaultOp::PutAtomic,
            CREATE_URI,
            1,
        );
        let err = with_reissue(|| faults.put_atomic(CREATE_URI, Bytes::from_static(OURS)))
            .await
            .expect_err("the re-issue meets the first write");
        assert!(
            matches!(err, StorageError::PreconditionFailed { .. }),
            "expected PreconditionFailed, got {err:?}"
        );
    }

    /// A create that timed out after landing is reported created, with the
    /// landed object's token, so a caller can chain it into a CAS.
    #[tokio::test]
    async fn create_that_timed_out_after_landing_reports_its_own_write() {
        let dir = TempDir::new().expect("tempdir");
        let faults = fault_store(&dir);
        faults.fail_with(
            FaultKind::TimedOutAfterLanding,
            FaultOp::PutAtomic,
            CREATE_URI,
            1,
        );
        let read_backs = Cell::new(0);
        let token = create_ours(&faults, &read_backs)
            .await
            .expect("the landed write is ours");
        assert_eq!(faults.fired(), 1, "the first issue timed out after landing");
        assert_eq!(
            read_backs.get(),
            1,
            "the ambiguous precondition was read back"
        );
        let (bytes, meta) = faults.get(CREATE_URI).await.expect("get");
        assert_eq!(bytes.as_ref(), OURS);
        assert_eq!(token, meta.etag, "the token is the landed object's");
    }

    /// After a transient failure, a key holding a peer's bytes is a lost race.
    #[tokio::test]
    async fn create_that_timed_out_surfaces_a_peers_object() {
        let dir = TempDir::new().expect("tempdir");
        let faults = fault_store(&dir);
        faults
            .put_atomic(CREATE_URI, Bytes::from_static(PEERS))
            .await
            .expect("peer create");
        faults.fail(FaultOp::PutAtomic, CREATE_URI, 1);
        let read_backs = Cell::new(0);
        let err = create_ours(&faults, &read_backs)
            .await
            .expect_err("a peer's object is a lost race");
        assert!(
            matches!(err, StorageError::PreconditionFailed { .. }),
            "expected PreconditionFailed, got {err:?}"
        );
        assert_eq!(
            read_backs.get(),
            1,
            "the ambiguous precondition was read back"
        );
        let (bytes, _) = faults.get(CREATE_URI).await.expect("get");
        assert_eq!(bytes.as_ref(), PEERS, "the peer's object is untouched");
    }

    /// A first issue that finds the key taken never landed anything, so it
    /// is a lost race even against identical bytes, and costs no read-back.
    #[tokio::test]
    async fn create_whose_first_issue_finds_the_key_taken_never_reads_back() {
        let dir = TempDir::new().expect("tempdir");
        let faults = fault_store(&dir);
        faults
            .put_atomic(CREATE_URI, Bytes::from_static(OURS))
            .await
            .expect("earlier create");
        let read_backs = Cell::new(0);
        let err = create_ours(&faults, &read_backs)
            .await
            .expect_err("create-only on an existing key");
        assert!(
            matches!(err, StorageError::PreconditionFailed { .. }),
            "expected PreconditionFailed, got {err:?}"
        );
        assert_eq!(read_backs.get(), 0, "no read-back on the first issue");
    }

    /// A read-back that fails cannot prove the write ours, so the
    /// precondition stands.
    #[tokio::test]
    async fn create_whose_read_back_fails_reports_the_precondition() {
        let dir = TempDir::new().expect("tempdir");
        let faults = fault_store(&dir);
        faults.fail_with(
            FaultKind::TimedOutAfterLanding,
            FaultOp::PutAtomic,
            CREATE_URI,
            1,
        );
        faults.fail(FaultOp::Get, CREATE_URI, 1);
        let read_backs = Cell::new(0);
        let err = create_ours(&faults, &read_backs)
            .await
            .expect_err("an unread object is not claimed");
        assert!(
            matches!(err, StorageError::PreconditionFailed { .. }),
            "expected PreconditionFailed, got {err:?}"
        );
        assert_eq!(
            faults.fired(),
            2,
            "the landing timeout and the failed read-back"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn range_zero_length_is_empty() {
        let r = complete_range("u", 5..5, |_| async { Ok(Bytes::new()) })
            .await
            .expect("test");
        assert!(r.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn range_single_full_chunk() {
        let r = complete_range("u", 0..3, |_| async { Ok(Bytes::from_static(b"abc")) })
            .await
            .expect("test");
        assert_eq!(&r[..], b"abc");
    }

    #[tokio::test(start_paused = true)]
    async fn range_truncates_overlong_chunk() {
        let r = complete_range("u", 0..3, |_| async { Ok(Bytes::from_static(b"abcdef")) })
            .await
            .expect("test");
        assert_eq!(&r[..], b"abc");
    }

    #[tokio::test(start_paused = true)]
    async fn range_assembles_short_chunks() {
        let r = complete_range("u", 0..5, |range| async move {
            let data = b"abcde";
            let start = range.start as usize;
            let end = (start + 2).min(data.len());
            Ok(Bytes::copy_from_slice(&data[start..end]))
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abcde");
    }

    #[tokio::test(start_paused = true)]
    async fn range_retries_transient_then_completes() {
        let calls = Cell::new(0u32);
        let r = complete_range("u", 0..3, |_| {
            let c = calls.get();
            calls.set(c + 1);
            async move {
                if c == 0 {
                    Err(transient())
                } else {
                    Ok(Bytes::from_static(b"abc"))
                }
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abc");
    }

    #[tokio::test(start_paused = true)]
    async fn range_empty_body_then_full() {
        let calls = Cell::new(0u32);
        let r = complete_range("u", 0..3, |_| {
            let c = calls.get();
            calls.set(c + 1);
            async move {
                if c == 0 {
                    Ok(Bytes::new())
                } else {
                    Ok(Bytes::from_static(b"abc"))
                }
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abc");
    }

    #[tokio::test(start_paused = true)]
    async fn range_persistent_empty_body_is_short_read() {
        let r = complete_range("u", 0..3, |_| async { Ok(Bytes::new()) }).await;
        assert!(matches!(r, Err(StorageError::Permanent { .. })));
    }

    #[tokio::test(start_paused = true)]
    async fn range_propagates_non_retryable() {
        let r = complete_range("u", 0..3, |_| async { Err::<Bytes, _>(not_found()) }).await;
        assert!(matches!(r, Err(StorageError::NotFound { .. })));
    }

    fn meta(size: u64) -> ObjectMeta {
        ObjectMeta {
            size,
            etag: None,
            last_modified: std::time::UNIX_EPOCH,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn complete_get_returns_full_body_first_try() {
        let calls = Cell::new(0u32);
        let (bytes, _m) = complete_get("u", || {
            calls.set(calls.get() + 1);
            async { Ok((Bytes::from_static(b"hello"), meta(5))) }
        })
        .await
        .expect("full body");
        assert_eq!(&bytes[..], b"hello");
        assert_eq!(calls.get(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn complete_get_reissues_a_truncated_body() {
        // First response is a body of 3 bytes while the response's own meta
        // declares 5 (object_store's short-read-as-success hazard). The helper
        // must NOT return it — it re-issues and returns the full object.
        let calls = Cell::new(0u32);
        let (bytes, _m) = complete_get("u", || {
            let c = calls.get();
            calls.set(c + 1);
            async move {
                if c == 0 {
                    Ok((Bytes::from_static(b"hel"), meta(5)))
                } else {
                    Ok((Bytes::from_static(b"hello"), meta(5)))
                }
            }
        })
        .await
        .expect("reissue then full");
        assert_eq!(&bytes[..], b"hello");
        assert_eq!(calls.get(), 2, "must have re-issued exactly once");
    }

    #[tokio::test(start_paused = true)]
    async fn complete_get_persistent_truncation_is_short_read() {
        // A body that stays short past the budget surfaces a stable short-read
        // error rather than a silently-truncated object.
        let r = complete_get("u", || async { Ok((Bytes::from_static(b"hel"), meta(5))) }).await;
        let e = r.expect_err("persistent truncation must error");
        assert!(matches!(e, StorageError::Permanent { .. }));
        assert!(e.to_string().contains("short read"));
    }

    #[tokio::test(start_paused = true)]
    async fn complete_get_propagates_non_retryable() {
        let r = complete_get("u", || async { Err::<(Bytes, ObjectMeta), _>(not_found()) }).await;
        assert!(matches!(r, Err(StorageError::NotFound { .. })));
    }
}
