// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared transient-retry + range-completion helpers for the
//! object-store-backed providers. One copy so retry semantics can't
//! drift between backends; each backend keeps its own error
//! translation and feeds already-translated results in.

use std::{error::Error, future::Future, ops::Range, time::Duration};

use bytes::{Bytes, BytesMut};
use object_store::{
    Attribute, Attributes, BackoffConfig, GetOptions, PutMode, PutOptions, RetryConfig,
};
use tokio::time;
use tracing::warn;
use uuid::Uuid;

use super::{ObjectMeta, StorageError};

/// The user-metadata key a create-only PUT stamps with its [`CreateId`]
/// (`x-amz-meta-`, `x-goog-meta-` or `x-ms-meta-` on the wire). Lowercase
/// letters only: Azure takes only identifier-shaped metadata names (no
/// `-`), and some proxies drop header names that carry `_`.
const CREATE_ID_KEY: &str = "infinocreateid";

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

/// The id one create-only PUT stamps on its object, the same on every issue
/// of it, so a later HEAD can tell that call's own landed write from
/// anyone else's object at the key.
pub(crate) struct CreateId(String);

impl CreateId {
    /// A fresh id for one `put_atomic` call.
    pub(crate) fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Options for one issue of the create: create-only, with this id in the
    /// object's user metadata.
    pub(crate) fn put_options(&self) -> PutOptions {
        let mut attributes = Attributes::new();
        attributes.insert(
            Attribute::Metadata(CREATE_ID_KEY.into()),
            self.0.clone().into(),
        );
        PutOptions {
            mode: PutMode::Create,
            attributes,
            ..Default::default()
        }
    }

    /// Whether an object's `attributes` (a HEAD's) carry this id.
    fn stamped_on(&self, attributes: &Attributes) -> bool {
        attributes
            .get(&Attribute::Metadata(CREATE_ID_KEY.into()))
            .is_some_and(|id| id.as_ref() == self.0)
    }
}

/// Options for a HEAD: the object's metadata and user attributes, no body.
pub(crate) fn head_options() -> GetOptions {
    GetOptions {
        head: true,
        ..Default::default()
    }
}

/// Run a create-only PUT through the app-level transient budget, and settle
/// a `PreconditionFailed` by whose object holds the key.
///
/// A create is not idempotent. A request the store accepted can still come
/// back as a failure — it timed out and this layer re-issues it, or it got a
/// 5xx and `object_store` retried it — and the next issue then fails the
/// precondition against its own object. Every issue carries `id` in the
/// object's metadata, so on any `PreconditionFailed`, the first issue's
/// included, one HEAD settles it: an object stamped with `id` is this call's
/// own write, reported as created with its CAS token; anything else is a
/// real lost race and surfaces as `PreconditionFailed`, including a peer
/// that wrote the same bytes. The HEAD carries no body, so it costs one
/// request whatever the object's size, and only on the precondition path.
///
/// `create` performs one create-only PUT built with [`CreateId::put_options`];
/// `head` one HEAD built with [`head_options`], answering the object's
/// attributes and CAS token. Both are already translated to `StorageError`.
pub(crate) async fn create_with_reissue<C, CFut, H, HFut>(
    uri: &str,
    id: &CreateId,
    create: C,
    head: H,
) -> Result<Option<String>, StorageError>
where
    C: FnMut() -> CFut,
    CFut: Future<Output = Result<Option<String>, StorageError>>,
    H: FnMut() -> HFut,
    HFut: Future<Output = Result<(Attributes, Option<String>), StorageError>>,
{
    let lost = match with_reissue(create).await {
        Err(lost @ StorageError::PreconditionFailed { .. }) => lost,
        other => return other,
    };
    match with_reissue(head).await {
        Ok((attributes, token)) if id.stamped_on(&attributes) => Ok(token),
        Ok(_) => Err(lost),
        Err(e) => {
            // Without the object we cannot tell our write from a peer's, so
            // report what the store said rather than claim a create.
            warn!(uri, error = %e, "could not read back a failed create; reporting the precondition");
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
    use std::cell::Cell;

    use object_store::{
        Error as ObjError, ObjectStore, ObjectStoreExt, PutPayload, memory::InMemory,
        path::Path as ObjPath,
    };

    use super::*;

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
    /// The bytes every create below writes, a peer's included.
    const BYTES: &[u8] = b"the bytes every create writes";

    /// How a create's first issue goes before its result reaches the caller.
    #[derive(Clone, Copy)]
    enum FirstIssue {
        /// The store answers it.
        Answered,
        /// It lands, then the caller sees a timeout, so this layer re-issues
        /// it.
        TimedOutAfterLanding,
        /// It lands, then the store client retries it after a 5xx, and the
        /// retry meets the landed object.
        RetriedAfterLanding,
    }

    /// The in-memory store's errors, translated as the providers translate.
    fn translate(e: ObjError) -> StorageError {
        match e {
            ObjError::AlreadyExists { .. } | ObjError::Precondition { .. } => {
                StorageError::PreconditionFailed {
                    uri: CREATE_URI.into(),
                }
            }
            ObjError::NotFound { .. } => StorageError::NotFound {
                uri: CREATE_URI.into(),
            },
            other => StorageError::Permanent {
                uri: CREATE_URI.into(),
                source: Box::new(other),
            },
        }
    }

    /// An in-memory object store, which keeps user metadata as S3, GCS and
    /// Azure do, counting the PUTs and HEADs of the creates below.
    struct Store {
        inner: InMemory,
        path: ObjPath,
        puts: Cell<u32>,
        heads: Cell<u32>,
        head_fails: Cell<bool>,
    }

    impl Store {
        fn new() -> Self {
            Self {
                inner: InMemory::new(),
                path: ObjPath::from(CREATE_URI),
                puts: Cell::new(0),
                heads: Cell::new(0),
                head_fails: Cell::new(false),
            }
        }

        /// One create-only PUT of [`BYTES`] stamped with `id`.
        async fn put(&self, id: &CreateId) -> Result<Option<String>, StorageError> {
            self.inner
                .put_opts(&self.path, PutPayload::from_static(BYTES), id.put_options())
                .await
                .map(|r| r.e_tag)
                .map_err(translate)
        }

        /// Create [`BYTES`] at [`CREATE_URI`] stamped with `id`, the way the
        /// object-store providers do, its first issue going as `first` says.
        async fn create(
            &self,
            id: &CreateId,
            first: FirstIssue,
        ) -> Result<Option<String>, StorageError> {
            create_with_reissue(
                CREATE_URI,
                id,
                || {
                    let issue = self.puts.get();
                    self.puts.set(issue + 1);
                    async move {
                        let answered = self.put(id).await;
                        match (issue, first) {
                            (0, FirstIssue::TimedOutAfterLanding) => Err(transient()),
                            (0, FirstIssue::RetriedAfterLanding) => self.put(id).await,
                            _ => answered,
                        }
                    }
                },
                || {
                    self.heads.set(self.heads.get() + 1);
                    async {
                        if self.head_fails.get() {
                            return Err(not_found());
                        }
                        let head = self
                            .inner
                            .get_opts(&self.path, head_options())
                            .await
                            .map_err(translate)?;
                        Ok((head.attributes, head.meta.e_tag))
                    }
                },
            )
            .await
        }

        /// The CAS token of the object at the key.
        async fn token(&self) -> Option<String> {
            self.inner.head(&self.path).await.expect("head").e_tag
        }

        /// Whether the object at the key carries `id`.
        async fn stamped_with(&self, id: &CreateId) -> bool {
            let head = self
                .inner
                .get_opts(&self.path, head_options())
                .await
                .expect("head");
            id.stamped_on(&head.attributes)
        }
    }

    fn assert_lost_race(result: Result<Option<String>, StorageError>) {
        let err = result.expect_err("a lost race");
        assert!(
            matches!(err, StorageError::PreconditionFailed { .. }),
            "expected PreconditionFailed, got {err:?}"
        );
    }

    /// The hazard `create_with_reissue` exists for: a create that timed out
    /// after landing, re-issued as a plain transient retry, meets its own
    /// object and reports a lost race.
    #[tokio::test(start_paused = true)]
    async fn a_bare_reissue_of_a_landed_create_meets_its_own_object() {
        let store = Store::new();
        let id = CreateId::new();
        let issue = Cell::new(0u32);
        assert_lost_race(
            with_reissue(|| {
                let n = issue.get();
                issue.set(n + 1);
                let (store, id) = (&store, &id);
                async move {
                    let answered = store.put(id).await;
                    if n == 0 { Err(transient()) } else { answered }
                }
            })
            .await,
        );
    }

    /// A create the store answers needs no HEAD.
    #[tokio::test(start_paused = true)]
    async fn an_answered_create_makes_no_head() {
        let store = Store::new();
        let token = store
            .create(&CreateId::new(), FirstIssue::Answered)
            .await
            .expect("created");
        assert_eq!(token, store.token().await);
        assert_eq!(store.heads.get(), 0);
    }

    /// A create that timed out after landing is reported created, with the
    /// landed object's token, so a caller can chain it into a CAS.
    #[tokio::test(start_paused = true)]
    async fn a_create_that_timed_out_after_landing_reports_its_own_write() {
        let store = Store::new();
        let token = store
            .create(&CreateId::new(), FirstIssue::TimedOutAfterLanding)
            .await
            .expect("the landed write is ours");
        assert_eq!(token, store.token().await, "the landed object's token");
        assert_eq!((store.puts.get(), store.heads.get()), (2, 1));
    }

    /// A first issue the store client retried after it landed meets its own
    /// object; the HEAD settles it on that first issue.
    #[tokio::test(start_paused = true)]
    async fn a_create_the_store_client_retried_after_landing_reports_its_own_write() {
        let store = Store::new();
        let token = store
            .create(&CreateId::new(), FirstIssue::RetriedAfterLanding)
            .await
            .expect("the landed write is ours");
        assert_eq!(token, store.token().await, "the landed object's token");
        assert_eq!((store.puts.get(), store.heads.get()), (1, 1));
    }

    /// A peer's object is a lost race even when it holds the same bytes: the
    /// id is the peer's, not ours.
    #[tokio::test(start_paused = true)]
    async fn a_peer_that_wrote_the_same_bytes_is_a_lost_race() {
        let store = Store::new();
        let peer = CreateId::new();
        store
            .create(&peer, FirstIssue::Answered)
            .await
            .expect("the peer's create");
        assert_lost_race(store.create(&CreateId::new(), FirstIssue::Answered).await);
        assert_eq!(store.heads.get(), 1, "one HEAD settled it");
        assert!(store.stamped_with(&peer).await, "the peer's object stands");
    }

    /// An object no create stamped is never ours.
    #[tokio::test(start_paused = true)]
    async fn an_object_with_no_create_id_is_a_lost_race() {
        let store = Store::new();
        store
            .inner
            .put(&store.path, PutPayload::from_static(BYTES))
            .await
            .expect("an unstamped write");
        assert_lost_race(store.create(&CreateId::new(), FirstIssue::Answered).await);
    }

    /// A HEAD that fails cannot prove the object ours, so the precondition
    /// stands.
    #[tokio::test(start_paused = true)]
    async fn a_create_whose_head_fails_reports_the_precondition() {
        let store = Store::new();
        store.head_fails.set(true);
        assert_lost_race(
            store
                .create(&CreateId::new(), FirstIssue::TimedOutAfterLanding)
                .await,
        );
        assert_eq!(store.heads.get(), 1);
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
