// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared retry and read helpers for the object-store-backed providers: transient retries, and
//! reads that recover from a stuck store (a second request when headers are late, and a fresh
//! request for the missing bytes when a body goes quiet). One copy so the rules can't drift between
//! backends; each backend keeps its own error translation and feeds translated results in.

use std::{error::Error, future::Future, ops::Range, pin::pin, time::Duration};

use bytes::{Bytes, BytesMut};
use futures::{FutureExt, StreamExt, TryStreamExt, stream::BoxStream};
use object_store::{
    Attribute, Attributes, BackoffConfig, Error as ObjError, GetOptions, GetResult,
    ObjectMeta as StoreMeta, PutMode, PutOptions, RetryConfig,
};
use tokio::{sync::Semaphore, time};
use tracing::warn;
use uuid::Uuid;

use super::StorageError;
use crate::{runtime_metrics::io::UsageMeter, utils::trace::record};

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

/// How long an object-store request runs before it is taken for stuck and
/// a second one is sent beside it, whatever the bytes (for a read, how long
/// it waits for its response's headers). Far above a normal
/// request (on GCS a GET is about 45 ms at p50 and under 140 ms at p90, an
/// 8 MB PUT under half a second), far below a stalled one (one attempt can
/// run to the request timeout of a minute): a hedge costs nothing
/// normally, and bounds how long one stuck request holds what waits on it.
const STUCK_AFTER: Duration = Duration::from_secs(2);

/// A request moving bytes slower than this is taken for stuck: it is
/// hedged once it has run [`STUCK_AFTER`] plus the time its bytes take at
/// this rate, so a big request that is only slow is not sent twice. One
/// request on GCS moves tens of MB/s even with many others in flight.
const STUCK_BYTES_PER_SEC: u64 = 16 << 20;

/// How long a request of `bytes` runs before a hedge is sent beside it:
/// [`STUCK_AFTER`], plus the time those bytes take at
/// [`STUCK_BYTES_PER_SEC`].
pub(crate) fn stuck_after(bytes: u64) -> Duration {
    STUCK_AFTER + Duration::from_millis(bytes.saturating_mul(1000) / STUCK_BYTES_PER_SEC)
}

/// Run `attempt` for a request of `bytes`, starting a second one beside it
/// once the first has run [`stuck_after`] those bytes. The first attempt to
/// succeed wins and the other is dropped. One that fails waits for the
/// other, so a quick failure of the second cannot cancel a slow first that
/// would have succeeded, and the request fails only if both do, with the
/// later error. A first attempt that fails before the delay is the answer:
/// no second is sent. For requests that are safe to send twice.
///
/// With a `hedges` bound, the second attempt is sent only under one of its
/// permits, held until this request settles: a request past its delay
/// while every permit is out waits for one, still racing its first attempt.
/// A link slow enough that every request passes its delay then sends as
/// many second attempts as the bound has permits, not one per request,
/// while a request that is truly stuck still gets its second attempt once
/// a permit frees.
pub(crate) async fn hedged<T, E, F, Fut>(
    bytes: u64,
    hedges: Option<&Semaphore>,
    attempt: F,
) -> Result<T, E>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let mut first = pin!(attempt());
    tokio::select! {
        answer = &mut first => return answer,
        () = time::sleep(stuck_after(bytes)) => {}
    }
    let _permit = match hedges {
        None => None,
        Some(hedges) => tokio::select! {
            answer = &mut first => return answer,
            // A closed bound sends no second attempt; the first answers.
            permit = hedges.acquire() => match permit {
                Ok(permit) => Some(permit),
                Err(_) => return first.await,
            },
        },
    };
    let mut second = pin!(attempt());
    tokio::select! {
        answer = &mut first => match answer {
            Ok(value) => Ok(value),
            Err(_) => second.await,
        },
        answer = &mut second => match answer {
            Ok(value) => Ok(value),
            Err(_) => first.await,
        },
    }
}

/// Transient re-issue backoff: `BASE × 2^min(attempt, MAX_SHIFT)` ms, capped.
const BACKOFF_BASE_MS: u64 = 50;
const BACKOFF_MAX_SHIFT: u32 = 5;
const BACKOFF_CAP_MS: u64 = 2000;

/// App-level re-issue budget for transient transport failures that
/// `object_store` won't retry itself (e.g. "error sending request" on a
/// socket the service dropped under us).
const MAX_TRANSIENT_RETRIES: u32 = 8;

/// How long the HEAD that settles a transient create failure may run before
/// the create is re-issued without its answer. A HEAD is a safe request, so
/// the store client retries its timeouts for up to [`RETRY_TIMEOUT`]; without
/// this bound a stalled store would make each re-issue wait out that window
/// first. A healthy HEAD answers in well under a second.
const SETTLE_HEAD_TIMEOUT: Duration = Duration::from_secs(10);

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

/// Permanent error: a body carried more bytes than its headers stated, so it is not the object
/// they describe. Stable, so callers don't retry.
fn long_read(uri: &str, stated: u64, got: u64) -> StorageError {
    StorageError::Permanent {
        uri: uri.into(),
        source: format!("long read: body carried {got} bytes where its headers stated {stated}")
            .into(),
    }
}

/// Re-issue an idempotent write or delete through the app-level transient
/// budget. `op` must already map to `StorageError`.
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

    /// Who holds the key, from one HEAD of it.
    fn settle(&self, head: Result<(Attributes, Option<String>), StorageError>) -> Holder {
        match head {
            Ok((attributes, token)) if self.stamped_on(&attributes) => Holder::Ours(token),
            Ok(_) => Holder::Other,
            Err(e) => Holder::Unknown(e),
        }
    }
}

/// Who holds a create's key, as a HEAD of it answers.
enum Holder {
    /// The object is this call's own write, with its CAS token.
    Ours(Option<String>),
    /// Another writer's object holds the key: a lost race.
    Other,
    /// The HEAD failed, so the holder is unknown.
    Unknown(StorageError),
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
/// A store can also refuse a create against a held key by closing the
/// connection before it has read the body, which arrives as a transient
/// failure on every issue. So a transient failure is settled by the same
/// HEAD before it is re-issued: a held key answers now, as created when the
/// object is this call's and as `PreconditionFailed` otherwise, and only a
/// free key is re-issued. That HEAD is bounded by [`SETTLE_HEAD_TIMEOUT`]; one
/// that fails or runs past it re-issues the create, as before it existed.
///
/// `create` performs one create-only PUT built with [`CreateId::put_options`];
/// `head` one HEAD built with [`head_options`], answering the object's
/// attributes and CAS token. Both are already translated to `StorageError`.
pub(crate) async fn create_with_reissue<C, CFut, H, HFut>(
    uri: &str,
    id: &CreateId,
    mut create: C,
    mut head: H,
) -> Result<Option<String>, StorageError>
where
    C: FnMut() -> CFut,
    CFut: Future<Output = Result<Option<String>, StorageError>>,
    H: FnMut() -> HFut,
    HFut: Future<Output = Result<(Attributes, Option<String>), StorageError>>,
{
    let mut attempt = 0u32;
    let lost = loop {
        match create().await {
            Ok(token) => return Ok(token),
            Err(lost @ StorageError::PreconditionFailed { .. }) => break lost,
            Err(e) if is_retryable(&e) && attempt < MAX_TRANSIENT_RETRIES => {
                if let Ok(answer) = time::timeout(SETTLE_HEAD_TIMEOUT, head()).await {
                    match id.settle(answer) {
                        Holder::Ours(token) => return Ok(token),
                        Holder::Other => {
                            return Err(StorageError::PreconditionFailed { uri: uri.into() });
                        }
                        Holder::Unknown(_) => {}
                    }
                }
                warn!(attempt, error = %e, "transient object-store error; re-issuing");
                time::sleep(backoff(attempt)).await;
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    };
    match id.settle(with_reissue(&mut head).await) {
        Holder::Ours(token) => Ok(token),
        Holder::Other => Err(lost),
        Holder::Unknown(e) => {
            // Without the object we cannot tell our write from a peer's, so
            // report what the store said rather than claim a create.
            warn!(uri, error = %e, "could not read back a failed create; reporting the precondition");
            Err(lost)
        }
    }
}

/// Run a DELETE through the app-level transient budget.
///
/// A delete is idempotent: a re-issue of one that landed meets no object,
/// and a missing object counts as deleted, so re-issuing is safe. A
/// timeout or dropped connection on a DELETE reaches this layer as
/// `TransientExhausted` (`object_store` retries a timeout only for an
/// idempotent request, and the GCS client marks a DELETE as not
/// idempotent), so without a re-issue one such failure fails the whole
/// operation — a table purge removing thousands of objects, say.
///
/// `delete` performs one DELETE; `translate` maps the store's errors as the
/// provider does. Only `TransientExhausted` is re-issued; `NotFound` is
/// success and every other error is returned as it is.
pub(crate) async fn delete_with_reissue<D, DFut, T>(
    mut delete: D,
    translate: T,
) -> Result<(), StorageError>
where
    D: FnMut() -> DFut,
    DFut: Future<Output = Result<(), ObjError>>,
    T: Fn(ObjError) -> StorageError,
{
    with_reissue(|| {
        let issued = delete();
        let translate = &translate;
        async move { settle_delete(issued.await, translate) }
    })
    .await
}

/// One DELETE, settled as [`delete_with_reissue`] settles each issue but
/// never re-issued. For best-effort cleanup that a background sweep
/// finishes if this fails: its caller ignores the result, so it must not
/// wait through the re-issue backoff for it.
pub(crate) async fn delete_once<DFut, T>(delete: DFut, translate: T) -> Result<(), StorageError>
where
    DFut: Future<Output = Result<(), ObjError>>,
    T: Fn(ObjError) -> StorageError,
{
    settle_delete(delete.await, &translate)
}

/// A DELETE's answer as the caller sees it: a missing object is deleted.
fn settle_delete(
    answer: Result<(), ObjError>,
    translate: &impl Fn(ObjError) -> StorageError,
) -> Result<(), StorageError> {
    match answer {
        Ok(()) | Err(ObjError::NotFound { .. }) => Ok(()),
        Err(e) => Err(translate(e)),
    }
}

/// How long a read's response may go without a body frame before it is taken for stalled: it is
/// dropped and the bytes still missing are asked for again. Far above the gap between frames of a
/// body that is moving, even one sharing a slow link with hundreds of others (a read of the socket
/// returns whatever has arrived, and a TLS record holds at most 16 KiB, so a link moving 16 KiB/s
/// still delivers a frame a second), and far below the request timeout a stalled body would
/// otherwise wait out (30 s on S3, 60 s on GCS and Azure).
const BODY_IDLE: Duration = Duration::from_secs(5);

/// How long after it last heard from the store (headers or a body frame, or its start before
/// either) a read that stopped short may still be asked for again. A request that fails later has
/// been through `object_store`'s own retries, up to [`RETRY_TIMEOUT`], so asking again would repeat
/// them: the read fails instead, not after one such window per re-issue. A read that keeps hearing
/// from the store keeps its budget.
const REISSUE_WITHIN: Duration = Duration::from_secs(60);

/// Most second requests for reads' headers out at once across the process. Each holds its permit
/// until it or its read's first request answers. One as stuck as its first holds it until
/// `object_store`'s own retries end one of them: usually the first request's retry after the
/// client's timeout, at worst about six minutes, as `object_store` checks [`RETRY_TIMEOUT`] only
/// after a try fails, so its last try can start just inside it and run a full client timeout.
/// Reads waiting for a permit keep racing their first request, so they are never slower than
/// without hedging. A healthy store answers headers far below [`STUCK_AFTER`] and sends none; one
/// so slow that every read passes it gets this many at a time instead of one per read.
const READ_HEDGES_IN_FLIGHT: usize = 16;

/// The permits [`READ_HEDGES_IN_FLIGHT`] counts.
static READ_HEDGES: Semaphore = Semaphore::const_new(READ_HEDGES_IN_FLIGHT);

/// How a read hedges a request whose headers are late: the permits its second requests take, and
/// the meter that counts them.
pub(crate) struct Hedging<'a> {
    permits: &'a Semaphore,
    meter: &'a UsageMeter,
}

impl<'a> Hedging<'a> {
    /// Hedge under the process-wide read bound ([`READ_HEDGES_IN_FLIGHT`]), counting each second
    /// request on `meter`.
    pub(crate) fn reads(meter: &'a UsageMeter) -> Self {
        Self {
            permits: &READ_HEDGES,
            meter,
        }
    }

    /// A second request went out: counted on the meter and marked on the read's span.
    fn sent(&self) {
        self.meter.record_hedge();
        record("hedged", true);
    }
}

/// One read request's response whose headers have arrived: the object's metadata and the bytes
/// its body carries, as the headers state them, and the body still to stream.
pub(crate) struct Response<'a> {
    meta: StoreMeta,
    range: Range<u64>,
    body: BodyStream<'a>,
}

impl<'a> Response<'a> {
    /// `result`'s headers and body, its body's errors translated by `translate`.
    pub(crate) fn new(
        result: GetResult,
        translate: impl Fn(ObjError) -> StorageError + Send + 'a,
    ) -> Self {
        let meta = result.meta.clone();
        let range = result.range.clone();
        let body = result.into_stream().map_err(translate).boxed();
        Self { meta, range, body }
    }
}

/// When a read that stopped short may be asked for again: at most [`MAX_TRANSIENT_RETRIES`] times
/// in a row without the read keeping bytes it lacked, and only within [`REISSUE_WITHIN`] of the
/// last time it heard from the store. A range read keeps the bytes each request brings, so a big
/// one on a link that stalls now and then is bounded by its bytes, not by the count; a whole read
/// starts over on every request and keeps none, so its count never starts again.
struct Reissue {
    since: time::Instant,
    attempt: u32,
}

impl Reissue {
    fn new() -> Self {
        Self {
            since: time::Instant::now(),
            attempt: 0,
        }
    }

    /// Which re-issue comes next (from 0) and the backoff to sleep before it, or `None` once the
    /// read may not be asked for again.
    fn next(&mut self) -> Option<(u32, Duration)> {
        if self.attempt >= MAX_TRANSIENT_RETRIES || self.since.elapsed() >= REISSUE_WITHIN {
            return None;
        }
        let attempt = self.attempt;
        self.attempt += 1;
        Some((attempt, backoff(attempt)))
    }

    /// Headers or a frame arrived: the window starts again.
    fn heard(&mut self) {
        self.since = time::Instant::now();
    }

    /// The read kept bytes it lacked: the count starts again.
    fn advanced(&mut self) {
        self.attempt = 0;
    }
}

/// The bytes a read has gathered toward `want`, from one response or several. Copied at most once,
/// as `object_store` collects a body: a lone frame is kept as it is, and the buffer is allocated
/// at its full size only when a second frame arrives.
struct Gathered {
    want: u64,
    first: Bytes,
    rest: BytesMut,
    /// Bytes past `want` that frames carried and were cut off.
    cut: u64,
}

impl Gathered {
    fn new(want: u64) -> Self {
        Self {
            want,
            first: Bytes::new(),
            rest: BytesMut::new(),
            cut: 0,
        }
    }

    fn len(&self) -> u64 {
        (self.first.len() + self.rest.len()) as u64
    }

    fn done(&self) -> bool {
        self.len() >= self.want
    }

    /// Add `frame`, cut to the bytes still wanted.
    fn push(&mut self, mut frame: Bytes) {
        let room = (self.want - self.len()) as usize;
        if frame.len() > room {
            self.cut += (frame.len() - room) as u64;
            frame.truncate(room);
        }
        if frame.is_empty() {
            return;
        }
        if self.len() == 0 {
            self.first = frame;
            return;
        }
        if self.rest.is_empty() {
            self.rest.reserve(self.want as usize);
            self.rest.extend_from_slice(&self.first);
            self.first = Bytes::new();
        }
        self.rest.extend_from_slice(&frame);
    }

    fn into_bytes(self) -> Bytes {
        if self.rest.is_empty() {
            self.first
        } else {
            self.rest.freeze()
        }
    }
}

/// A response's body, its errors translated.
type BodyStream<'a> = BoxStream<'a, Result<Bytes, StorageError>>;

/// Why a response stopped before the read had all its bytes.
enum Stop {
    /// The body ended at the wrong length: short (`object_store` hands back a body cut short as if
    /// it were whole), or, for a whole read, longer than its headers state.
    Ended,
    /// No frame for [`BODY_IDLE`].
    Idle,
    /// The response came from another version of the object than the bytes already gathered.
    Changed,
    Failed(StorageError),
}

/// Open a read's response: `open` sends one request and answers once its headers arrive. While the
/// first request has no headers, a second request goes out beside it [`STUCK_AFTER`] after it was
/// sent, under one of `hedging`'s permits, and runs until one of the two answers: the first headers
/// win. It is not cut short, so on a store whose every answer takes longer than [`STUCK_AFTER`] a
/// stuck read still finishes through it. A read has one second request at a time; after one fails,
/// the next goes out [`STUCK_AFTER`] later.
/// A request that loses is dropped before any of its body is read, so a hedge costs one request,
/// never a second body, though the store may already have sent its first bytes.
///
/// The first request's failure is the answer, and a hedge in flight is dropped: a request that
/// fails is not stuck, and the read's own re-issue sends a fresh one, so a read failing on every
/// request gives up when it would without hedging. A hedge the store refused (not found, say)
/// answers for both unless the first has already answered; one that failed otherwise leaves the
/// first to answer. An answer ready at the same time as a timer is taken first.
async fn race_headers<T, F, Fut>(hedging: &Hedging<'_>, open: F) -> Result<T, StorageError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, StorageError>>,
{
    let mut first = pin!(open());
    loop {
        tokio::select! {
            biased;
            answer = &mut first => return answer,
            () = time::sleep(STUCK_AFTER) => {}
        }
        let _permit = tokio::select! {
            biased;
            answer = &mut first => return answer,
            permit = hedging.permits.acquire() => match permit {
                Ok(permit) => permit,
                // A closed bound sends no second request; the first answers.
                Err(_) => return first.await,
            },
        };
        hedging.sent();
        let second = open();
        tokio::select! {
            biased;
            answer = second => match answer {
                Ok(value) => return Ok(value),
                Err(e) if !is_retryable(&e) => {
                    return (&mut first).now_or_never().unwrap_or(Err(e));
                }
                // A hedge that failed in a way a retry may fix leaves the first running; the next
                // goes out `STUCK_AFTER` later.
                Err(_) => {}
            },
            answer = &mut first => return answer,
        }
    }
}

/// Stream `body` into `got` until the read has all its bytes (`None`), or the body ends, fails or
/// goes [`BODY_IDLE`] without a frame. Every frame restarts `reissue`'s window. The caller has just
/// heard the response's headers, so the idle time counts from them until the first frame.
///
/// One timer serves the whole body, not one per frame: a frame only records when it came (the
/// window's restart), and when the timer fires it either moves to [`BODY_IDLE`] past the last
/// frame or ends the body as stalled. A timer per frame cost a register and remove in the runtime's
/// timer wheel for every frame, a few percent of a fast large read.
async fn drain(
    body: &mut BodyStream<'_>,
    got: &mut Gathered,
    reissue: &mut Reissue,
) -> Option<Stop> {
    let mut idle = pin!(time::sleep_until(reissue.since + BODY_IDLE));
    while !got.done() {
        tokio::select! {
            biased;
            frame = body.next() => match frame {
                None => return Some(Stop::Ended),
                Some(Err(e)) => return Some(Stop::Failed(e)),
                Some(Ok(frame)) => {
                    reissue.heard();
                    got.push(frame);
                }
            },
            () = &mut idle => {
                let deadline = reissue.since + BODY_IDLE;
                if deadline <= time::Instant::now() {
                    return Some(Stop::Idle);
                }
                idle.as_mut().reset(deadline);
            }
        }
    }
    None
}

/// Whether `body`, its stated bytes read, ends there (`None`); if not, why, with how many more
/// bytes it sent. A body longer than its headers state is not the object they describe; one that
/// goes quiet once its stated bytes are in is whole.
async fn ended(body: &mut BodyStream<'_>) -> Option<(Stop, u64)> {
    loop {
        match time::timeout(BODY_IDLE, body.next()).await {
            Err(_) => return None,
            Ok(None) => return None,
            Ok(Some(Err(e))) => return Some((Stop::Failed(e), 0)),
            Ok(Some(Ok(frame))) if frame.is_empty() => {}
            Ok(Some(Ok(frame))) => return Some((Stop::Ended, frame.len() as u64)),
        }
    }
}

/// Sleep before a read's next request after `stop`, or return the error that ends the read: a
/// refused request's own at once, or once `reissue` allows no more, the last failure, a stall, or
/// `short` for a body that ended short.
async fn again(
    uri: &str,
    stop: Stop,
    reissue: &mut Reissue,
    short: impl FnOnce() -> StorageError,
) -> Result<(), StorageError> {
    let stop = match stop {
        Stop::Failed(e) if !is_retryable(&e) => return Err(e),
        stop => stop,
    };
    let Some((attempt, wait)) = reissue.next() else {
        return Err(match stop {
            Stop::Failed(e) => e,
            Stop::Idle => transient(uri, format!("no response bytes for {BODY_IDLE:?}")),
            Stop::Changed => transient(uri, "the object changed while it was read".into()),
            Stop::Ended => short(),
        });
    };
    match &stop {
        Stop::Failed(e) => warn!(uri, attempt, error = %e, "transient read error; asking again"),
        Stop::Idle => warn!(
            uri,
            attempt, "read stalled; asking again for its missing bytes"
        ),
        Stop::Changed => warn!(
            uri,
            attempt, "object changed under a read; reading it again"
        ),
        Stop::Ended => {}
    }
    time::sleep(wait).await;
    Ok(())
}

/// Transient error for a read that ran out of requests for `reason`.
fn transient(uri: &str, reason: String) -> StorageError {
    StorageError::TransientExhausted {
        uri: uri.into(),
        source: reason.into(),
    }
}

/// Read `range` of an object, each request's headers raced as [`race_headers`] does and each body
/// streamed as [`drain`] does. A body that stops short (it ended, failed or stalled) is dropped and
/// only the bytes still missing are asked for again, so a stall costs one more request, not a
/// second copy of the bytes already read. An empty or cut-short body for an in-bounds range is a
/// transport glitch, not the object's end (that surfaces as a typed error), and is asked for again
/// the same way. Bytes from two versions of the object are never joined: a response whose ETag
/// differs from the one the gathered bytes came with starts the range again. A failed request is
/// issued again while [`Reissue`] allows; one the store refused (not found, say) fails the read at
/// once. `open` sends one range GET, already translated.
pub(crate) async fn read_range<'a, F, Fut>(
    uri: &str,
    range: Range<u64>,
    hedging: Hedging<'_>,
    open: F,
) -> Result<Bytes, StorageError>
where
    F: Fn(Range<u64>) -> Fut,
    Fut: Future<Output = Result<Response<'a>, StorageError>>,
{
    let want = range.end.saturating_sub(range.start);
    let mut got = Gathered::new(want);
    let mut reissue = Reissue::new();
    // The ETag of the response the gathered bytes came with.
    let mut version = None;
    // The most bytes the read has held: only passing it is progress, so a read that keeps starting
    // over on an object that keeps changing still runs out of re-issues.
    let mut most = 0;
    while !got.done() {
        let missing = range.start + got.len()..range.end;
        let stop = match race_headers(&hedging, || open(missing.clone())).await {
            Ok(mut response) => {
                reissue.heard();
                if got.len() > 0 && response.meta.e_tag != version {
                    got = Gathered::new(want);
                    Some(Stop::Changed)
                } else {
                    version = response.meta.e_tag;
                    drain(&mut response.body, &mut got, &mut reissue).await
                }
            }
            Err(e) => Some(Stop::Failed(e)),
        };
        let advanced = got.len() > most;
        if advanced {
            most = got.len();
            reissue.advanced();
        }
        let Some(stop) = stop else { break };
        // A body cut short after bringing bytes asks for the rest at once: the store was answering.
        if advanced && matches!(stop, Stop::Ended) {
            continue;
        }
        again(uri, stop, &mut reissue, || {
            short_read(uri, range.start, want, got.len())
        })
        .await?;
    }
    Ok(got.into_bytes())
}

/// Read a response's whole body the way [`read_range`] reads a range, for a request whose bytes
/// can't be asked for in part (a whole object, its last bytes, a conditional read): a body that
/// stops short is dropped and the request sent again whole, and the bytes must be exactly what the
/// headers of the response that carried them state, no fewer and no more. `open` sends the
/// request, already translated; `None` is an answer without a body to read (not modified),
/// returned as it is.
pub(crate) async fn read_whole<'a, F, Fut>(
    uri: &str,
    hedging: Hedging<'_>,
    open: F,
) -> Result<Option<(Bytes, StoreMeta)>, StorageError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<Option<Response<'a>>, StorageError>>,
{
    let mut reissue = Reissue::new();
    // The offset, length and bytes sent of the last body whose length was wrong.
    let mut short = (0, 0, 0);
    loop {
        let stop = match race_headers(&hedging, &open).await {
            Ok(None) => return Ok(None),
            Ok(Some(Response {
                meta,
                range,
                mut body,
            })) => {
                reissue.heard();
                let mut got = Gathered::new(range.end.saturating_sub(range.start));
                match drain(&mut body, &mut got, &mut reissue).await {
                    None => match ended(&mut body).await {
                        None if got.cut == 0 => return Ok(Some((got.into_bytes(), meta))),
                        None => {
                            short = (range.start, got.want, got.len() + got.cut);
                            Stop::Ended
                        }
                        Some((stop, more)) => {
                            short = (range.start, got.want, got.len() + got.cut + more);
                            stop
                        }
                    },
                    Some(stop) => {
                        short = (range.start, got.want, got.len());
                        stop
                    }
                }
            }
            Err(e) => Stop::Failed(e),
        };
        let (start, want, got) = short;
        again(uri, stop, &mut reissue, || {
            if got > want {
                long_read(uri, want, got)
            } else {
                short_read(uri, start, want, got)
            }
        })
        .await?;
    }
}

/// [`read_whole`] for a request whose every answer has a body.
pub(crate) async fn read_body<'a, F, Fut>(
    uri: &str,
    hedging: Hedging<'_>,
    open: F,
) -> Result<(Bytes, StoreMeta), StorageError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<Response<'a>, StorageError>>,
{
    let read = read_whole(uri, hedging, || {
        let opened = open();
        async move { opened.await.map(Some) }
    })
    .await?;
    Ok(read.expect("an answer with a body reads one"))
}

#[cfg(test)]
mod tests {
    use std::{
        cell::{Cell, RefCell},
        future::pending,
    };

    use futures::{future, stream};
    use object_store::{
        Error as ObjError, ObjectStore, ObjectStoreExt, PutPayload, memory::InMemory,
        path::Path as ObjPath,
    };
    use tokio::spawn;

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
        /// It fails before reaching the store, which holds nothing.
        LostBeforeLanding,
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
        /// Every HEAD fails as a dropped connection does.
        head_fails_transiently: Cell<bool>,
        /// Every HEAD hangs and never answers.
        head_hangs: Cell<bool>,
        /// A create against a held key fails as a dropped connection does,
        /// on every issue, instead of answering the precondition.
        drops_creates_on_held_key: Cell<bool>,
    }

    impl Store {
        fn new() -> Self {
            Self {
                inner: InMemory::new(),
                path: ObjPath::from(CREATE_URI),
                puts: Cell::new(0),
                heads: Cell::new(0),
                head_fails: Cell::new(false),
                head_fails_transiently: Cell::new(false),
                head_hangs: Cell::new(false),
                drops_creates_on_held_key: Cell::new(false),
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
                        if self.drops_creates_on_held_key.get()
                            && self.inner.head(&self.path).await.is_ok()
                        {
                            return Err(transient());
                        }
                        if matches!((issue, first), (0, FirstIssue::LostBeforeLanding)) {
                            return Err(transient());
                        }
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
                        if self.head_hangs.get() {
                            pending::<()>().await;
                        }
                        if self.head_fails_transiently.get() {
                            return Err(transient());
                        }
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
        assert_eq!((store.puts.get(), store.heads.get()), (1, 1));
    }

    /// A store that drops the connection of a create against a held key is
    /// settled by one HEAD, not re-issued through the whole budget.
    #[tokio::test(start_paused = true)]
    async fn a_dropped_create_against_a_held_key_is_a_lost_race_after_one_issue() {
        let store = Store::new();
        let peer = CreateId::new();
        store
            .create(&peer, FirstIssue::Answered)
            .await
            .expect("the peer's create");
        store.puts.set(0);
        store.drops_creates_on_held_key.set(true);
        assert_lost_race(store.create(&CreateId::new(), FirstIssue::Answered).await);
        assert_eq!((store.puts.get(), store.heads.get()), (1, 1));
        assert!(store.stamped_with(&peer).await, "the peer's object stands");
    }

    /// A create that failed before reaching the store finds the key free
    /// and is re-issued.
    #[tokio::test(start_paused = true)]
    async fn a_create_lost_before_landing_is_reissued() {
        let store = Store::new();
        store.head_fails.set(true);
        let id = CreateId::new();
        let token = store
            .create(&id, FirstIssue::LostBeforeLanding)
            .await
            .expect("created on the second issue");
        assert_eq!(token, store.token().await);
        assert_eq!((store.puts.get(), store.heads.get()), (2, 1));
        assert!(store.stamped_with(&id).await);
    }

    /// A HEAD that fails transiently says nothing about the key, so the create
    /// is re-issued rather than reported as a lost race.
    #[tokio::test(start_paused = true)]
    async fn a_create_whose_settling_head_fails_transiently_is_reissued() {
        let store = Store::new();
        store.head_fails_transiently.set(true);
        let id = CreateId::new();
        let token = store
            .create(&id, FirstIssue::LostBeforeLanding)
            .await
            .expect("created on the second issue");
        assert_eq!(token, store.token().await);
        assert_eq!((store.puts.get(), store.heads.get()), (2, 1));
        assert!(store.stamped_with(&id).await);
    }

    /// A settling HEAD that hangs is abandoned after [`SETTLE_HEAD_TIMEOUT`]
    /// and the create re-issued, so a stalled store does not hold each
    /// re-issue for the HEAD's own retry window.
    #[tokio::test(start_paused = true)]
    async fn a_create_whose_settling_head_hangs_is_reissued_after_the_bound() {
        let store = Store::new();
        store.head_hangs.set(true);
        let started = time::Instant::now();
        let token = store
            .create(&CreateId::new(), FirstIssue::LostBeforeLanding)
            .await
            .expect("created on the second issue");
        assert_eq!(token, store.token().await);
        assert_eq!((store.puts.get(), store.heads.get()), (2, 1));
        assert!(
            started.elapsed() < SETTLE_HEAD_TIMEOUT + backoff(0) + Duration::from_secs(1),
            "waited {:?}",
            started.elapsed()
        );
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
        assert_eq!(store.heads.get(), 2);
    }

    /// The CAS token a create below answers once a PUT succeeds.
    const CREATED_TOKEN: &str = "created";

    /// Run [`create_with_reissue`] against a free key whose `n`th PUT answers
    /// `issue(n)`. Answers the result with the PUTs and HEADs it made.
    async fn create_on_free_key(
        issue: impl Fn(u32) -> Result<Option<String>, StorageError>,
    ) -> (Result<Option<String>, StorageError>, u32, u32) {
        let (puts, heads) = (Cell::new(0u32), Cell::new(0u32));
        let result = create_with_reissue(
            CREATE_URI,
            &CreateId::new(),
            || {
                let n = puts.get();
                puts.set(n + 1);
                let answer = issue(n);
                async move { answer }
            },
            || {
                heads.set(heads.get() + 1);
                async { Err(not_found()) }
            },
        )
        .await;
        (result, puts.get(), heads.get())
    }

    /// A create that fails for a reason other than a transient one is
    /// neither settled by a HEAD nor re-issued.
    #[tokio::test(start_paused = true)]
    async fn a_create_with_a_non_transient_failure_is_not_reissued() {
        let (result, puts, heads) = create_on_free_key(|_| Err(not_found())).await;
        assert!(matches!(result, Err(StorageError::NotFound { .. })));
        assert_eq!((puts, heads), (1, 0));
    }

    /// A create whose first two issues fail transiently against a free key is
    /// re-issued until the third lands.
    #[tokio::test(start_paused = true)]
    async fn a_create_recovers_after_several_transient_failures() {
        let (result, puts, heads) = create_on_free_key(|n| {
            if n < 2 {
                Err(transient())
            } else {
                Ok(Some(CREATED_TOKEN.into()))
            }
        })
        .await;
        assert_eq!(result.expect("created"), Some(CREATED_TOKEN.into()));
        assert_eq!((puts, heads), (3, 2));
    }

    /// A create that fails transiently on every issue spends the re-issue
    /// budget and reports the transient failure.
    #[tokio::test(start_paused = true)]
    async fn a_create_that_keeps_failing_transiently_spends_the_budget() {
        let (result, puts, heads) = create_on_free_key(|_| Err(transient())).await;
        assert!(matches!(
            result,
            Err(StorageError::TransientExhausted { .. })
        ));
        assert_eq!(
            (puts, heads),
            (MAX_TRANSIENT_RETRIES + 1, MAX_TRANSIENT_RETRIES)
        );
    }

    // ---- delete_with_reissue ---------------------------------------------

    /// The key every delete below targets.
    const DELETE_URI: &str = "data/seg-00000002.sf.parquet";

    /// How one issue of a delete goes.
    #[derive(Clone, Copy)]
    enum DeleteIssue {
        /// The store answers it.
        Answered,
        /// It lands, then its answer is lost and the caller sees a timeout.
        LandedAnswerLost,
        /// It times out before reaching the store.
        TimedOut,
        /// The store refuses the credentials.
        Refused,
    }

    /// The store's errors as the providers translate them.
    fn translate_delete(e: ObjError) -> StorageError {
        match e {
            ObjError::PermissionDenied { .. } => StorageError::PermissionDenied {
                uri: DELETE_URI.into(),
            },
            ObjError::Generic { source, .. } => StorageError::TransientExhausted {
                uri: DELETE_URI.into(),
                source,
            },
            other => StorageError::Permanent {
                uri: DELETE_URI.into(),
                source: Box::new(other),
            },
        }
    }

    /// An in-memory store holding one object, its deletes going as `issues`
    /// says in turn and answered once they run out, each issue counted.
    struct DeleteStore {
        inner: InMemory,
        path: ObjPath,
        issues: Vec<DeleteIssue>,
        issued: Cell<usize>,
    }

    impl DeleteStore {
        async fn holding_one(issues: &[DeleteIssue]) -> Self {
            let store = Self {
                inner: InMemory::new(),
                path: ObjPath::from(DELETE_URI),
                issues: issues.to_vec(),
                issued: Cell::new(0),
            };
            store
                .inner
                .put(&store.path, PutPayload::from_static(BYTES))
                .await
                .expect("put");
            store
        }

        /// One issue of the delete.
        async fn issue(&self) -> Result<(), ObjError> {
            let n = self.issued.get();
            self.issued.set(n + 1);
            let timed_out = || ObjError::Generic {
                store: "InMemory",
                source: "operation timed out".into(),
            };
            match self.issues.get(n).copied().unwrap_or(DeleteIssue::Answered) {
                DeleteIssue::Answered => self.answer().await,
                DeleteIssue::LandedAnswerLost => {
                    self.answer().await?;
                    Err(timed_out())
                }
                DeleteIssue::TimedOut => Err(timed_out()),
                DeleteIssue::Refused => Err(ObjError::PermissionDenied {
                    path: DELETE_URI.into(),
                    source: "refused".into(),
                }),
            }
        }

        /// The store's answer to a delete. `InMemory` answers a delete of a
        /// missing object with success where GCS and Azure answer 404, so
        /// this answers `NotFound` for one as they do.
        async fn answer(&self) -> Result<(), ObjError> {
            self.inner.head(&self.path).await?;
            self.inner.delete(&self.path).await
        }

        async fn delete_with_reissue(&self) -> Result<(), StorageError> {
            delete_with_reissue(|| self.issue(), translate_delete).await
        }

        async fn holds_object(&self) -> bool {
            self.inner.head(&self.path).await.is_ok()
        }
    }

    /// A delete that timed out before reaching the store is re-issued, and
    /// the re-issue deletes the object.
    #[tokio::test(start_paused = true)]
    async fn a_timed_out_delete_is_reissued() {
        let store = DeleteStore::holding_one(&[DeleteIssue::TimedOut]).await;
        store
            .delete_with_reissue()
            .await
            .expect("the re-issue deletes");
        assert_eq!(store.issued.get(), 2);
        assert!(!store.holds_object().await);
    }

    /// A delete that landed but whose answer was lost is re-issued, and the
    /// re-issue's 404 is the delete's success.
    #[tokio::test(start_paused = true)]
    async fn a_reissued_delete_that_meets_no_object_succeeds() {
        let store = DeleteStore::holding_one(&[DeleteIssue::LandedAnswerLost]).await;
        store
            .delete_with_reissue()
            .await
            .expect("a 404 on the re-issue is success");
        assert_eq!(store.issued.get(), 2);
        assert!(!store.holds_object().await);
    }

    /// Refused credentials are not re-issued: the same call fails the same
    /// way, and the caller sees the refusal at once.
    #[tokio::test(start_paused = true)]
    async fn a_refused_delete_is_not_reissued() {
        let store = DeleteStore::holding_one(&[DeleteIssue::Refused]).await;
        let err = store.delete_with_reissue().await.expect_err("refused");
        assert!(
            matches!(err, StorageError::PermissionDenied { .. }),
            "expected PermissionDenied, got {err:?}"
        );
        assert_eq!(store.issued.get(), 1);
        assert!(store.holds_object().await);
    }

    /// A delete whose every issue times out spends the re-issue budget and
    /// reports the transient failure.
    #[tokio::test(start_paused = true)]
    async fn a_delete_that_keeps_timing_out_spends_the_budget() {
        let issues = [DeleteIssue::TimedOut; MAX_TRANSIENT_RETRIES as usize + 1];
        let store = DeleteStore::holding_one(&issues).await;
        let err = store.delete_with_reissue().await.expect_err("budget spent");
        assert!(matches!(err, StorageError::TransientExhausted { .. }));
        assert_eq!(store.issued.get(), issues.len());
    }

    /// The best-effort delete is one issue: a timeout is reported, not
    /// re-issued, and a 404 is success.
    #[tokio::test(start_paused = true)]
    async fn a_delete_once_is_not_reissued() {
        let store = DeleteStore::holding_one(&[DeleteIssue::TimedOut]).await;
        let err = delete_once(store.issue(), translate_delete)
            .await
            .expect_err("a timeout is reported");
        assert!(matches!(err, StorageError::TransientExhausted { .. }));
        assert_eq!(store.issued.get(), 1);
        assert!(store.holds_object().await);

        store.inner.delete(&store.path).await.expect("delete");
        delete_once(store.issue(), translate_delete)
            .await
            .expect("a 404 is success");
    }

    // ---- read_range / read_whole -------------------------------------------

    /// The object the read tests serve: no byte repeats, so a slice served from the wrong offset
    /// doesn't match.
    const OBJECT: &[u8] = b"abcdefghijklm";

    /// One-byte frames the moving-body test sends.
    const TRICKLE_FRAMES: u64 = 20;
    /// Reads the hedge-bound test keeps stuck at once.
    const STUCK_READS: usize = 3;
    /// How long the hedge-bound test keeps its reads stuck.
    const STUCK_FOR: Duration = Duration::from_secs(30);
    /// A store answer slower than `STUCK_AFTER`: only a hedge that runs until it answers gets it.
    const SLOW_ANSWER: Duration = Duration::from_secs(3);
    /// How long a failing hedge takes to fail.
    const HEDGE_FAILS_AFTER: Duration = Duration::from_secs(1);
    /// How long past a deadline a test's request fails or answers: inside every other delay.
    const PAST_DEADLINE: Duration = Duration::from_secs(1);

    /// Slow frames the cut-timing test sends before its body goes quiet: enough to move the body's
    /// timer more than once.
    const FRAMES_BEFORE_CUT: u64 = 3;

    /// How many re-issues past the count the progress test needs: more than the count allows
    /// without progress.
    const PROGRESS_REISSUES: u64 = MAX_TRANSIENT_RETRIES as u64 + 4;

    /// Virtual time a test read may take before it is taken for looping: far past any read here.
    const TEST_READ_BOUND: Duration = Duration::from_secs(3600);

    type Body = BoxStream<'static, Result<Bytes, StorageError>>;

    /// `OBJECT`'s bytes in `range`.
    fn served(range: &Range<u64>) -> Bytes {
        Bytes::copy_from_slice(&OBJECT[range.start as usize..range.end as usize])
    }

    /// A body of `frames`, then its end.
    fn body(frames: Vec<Result<Bytes, StorageError>>) -> Body {
        stream::iter(frames).boxed()
    }

    /// A body of `frames`, then no frame ever again.
    fn stalling(frames: Vec<Bytes>) -> Body {
        stream::iter(frames.into_iter().map(Ok))
            .chain(stream::pending())
            .boxed()
    }

    /// A response for `range` of `OBJECT` whose body is `body`.
    fn response(range: Range<u64>, body: Body) -> Response<'static> {
        Response {
            meta: StoreMeta {
                location: ObjPath::from("u"),
                last_modified: Default::default(),
                size: OBJECT.len() as u64,
                e_tag: None,
                version: None,
            },
            range,
            body,
        }
    }

    /// `response` as sent by the object's version `tag`.
    fn tagged(tag: &str, mut response: Response<'static>) -> Response<'static> {
        response.meta.e_tag = Some(tag.into());
        response
    }

    /// A response serving `range` of `OBJECT` in one frame.
    fn whole(range: Range<u64>) -> Response<'static> {
        let frame = served(&range);
        response(range, body(vec![Ok(frame)]))
    }

    /// A bound with `permits` permits, for one test.
    fn bound(permits: usize) -> &'static Semaphore {
        Box::leak(Box::new(Semaphore::new(permits)))
    }

    /// Read `range` under a bound of its own with one permit, noting second requests in `hedges`.
    async fn read<F, Fut>(
        range: Range<u64>,
        hedges: &UsageMeter,
        open: F,
    ) -> Result<Bytes, StorageError>
    where
        F: Fn(Range<u64>) -> Fut,
        Fut: Future<Output = Result<Response<'static>, StorageError>>,
    {
        let hedging = Hedging {
            permits: bound(1),
            meter: hedges,
        };
        time::timeout(TEST_READ_BOUND, read_range("u", range, hedging, open))
            .await
            .expect("the read ends")
    }

    /// Read the whole of what `open` answers under a bound of its own with one permit.
    async fn read_all<F, Fut>(
        hedges: &UsageMeter,
        open: F,
    ) -> Result<Option<(Bytes, StoreMeta)>, StorageError>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<Option<Response<'static>>, StorageError>>,
    {
        let hedging = Hedging {
            permits: bound(1),
            meter: hedges,
        };
        time::timeout(TEST_READ_BOUND, read_whole("u", hedging, open))
            .await
            .expect("the read ends")
    }

    /// The requests a test's read sent, in order.
    struct Opens(RefCell<Vec<Range<u64>>>);

    impl Opens {
        fn new() -> Self {
            Self(RefCell::new(Vec::new()))
        }

        /// Note a request for `range`; returns how many came before it.
        fn note(&self, range: &Range<u64>) -> usize {
            let mut opens = self.0.borrow_mut();
            opens.push(range.clone());
            opens.len() - 1
        }

        fn count(&self) -> usize {
            self.0.borrow().len()
        }

        fn ranges(&self) -> Vec<Range<u64>> {
            self.0.borrow().clone()
        }
    }

    /// Counts a request in flight down when its future is dropped.
    struct Live<'a>(&'a Cell<usize>);

    impl Drop for Live<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() - 1);
        }
    }

    /// Sets its flag when dropped: shows a request's future was dropped.
    struct Dropped<'a>(&'a Cell<bool>);

    impl Drop for Dropped<'_> {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn range_zero_length_is_empty() {
        let hedges = UsageMeter::new();
        // A request would fail the read: an empty range sends none.
        let r = read(5..5, &hedges, |_| async { Err(not_found()) })
            .await
            .expect("test");
        assert!(r.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn range_single_full_frame() {
        let hedges = UsageMeter::new();
        let r = read(0..3, &hedges, |range| async move { Ok(whole(range)) })
            .await
            .expect("test");
        assert_eq!(&r[..], b"abc");
    }

    #[tokio::test(start_paused = true)]
    async fn range_truncates_overlong_body() {
        let hedges = UsageMeter::new();
        let r = read(0..3, &hedges, |range| async move {
            Ok(response(range, body(vec![Ok(served(&(0..6)))])))
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abc");
    }

    #[tokio::test(start_paused = true)]
    async fn range_gathers_many_frames() {
        let hedges = UsageMeter::new();
        let r = read(0..5, &hedges, |range| async move {
            let frames = (range.start..range.end)
                .map(|i| Ok(served(&(i..i + 1))))
                .collect();
            Ok(response(range, body(frames)))
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abcde");
    }

    /// A body that ends short is asked for again from where it stopped, not from the start.
    #[tokio::test(start_paused = true)]
    async fn a_short_body_is_completed_by_asking_for_its_missing_tail() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let r = read(0..5, &hedges, |range| {
            let first = opens.note(&range) == 0;
            async move {
                if first {
                    Ok(response(range, body(vec![Ok(served(&(0..2)))])))
                } else {
                    Ok(whole(range))
                }
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abcde");
        assert_eq!(opens.ranges(), vec![0..5, 2..5]);
    }

    #[tokio::test(start_paused = true)]
    async fn range_retries_transient_then_completes() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let r = read(0..3, &hedges, |range| {
            let first = opens.note(&range) == 0;
            async move {
                if first {
                    Err(transient())
                } else {
                    Ok(whole(range))
                }
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abc");
        assert_eq!(opens.count(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn range_empty_body_then_full() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let r = read(0..3, &hedges, |range| {
            let first = opens.note(&range) == 0;
            async move {
                if first {
                    Ok(response(range, body(vec![])))
                } else {
                    Ok(whole(range))
                }
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abc");
    }

    #[tokio::test(start_paused = true)]
    async fn range_persistent_empty_body_is_short_read() {
        let hedges = UsageMeter::new();
        let r = read(0..3, &hedges, |range| async move {
            Ok(response(range, body(vec![])))
        })
        .await;
        assert!(matches!(r, Err(StorageError::Permanent { .. })));
    }

    #[tokio::test(start_paused = true)]
    async fn range_propagates_non_retryable() {
        let hedges = UsageMeter::new();
        let r = read(0..3, &hedges, |_| async { Err(not_found()) }).await;
        assert!(matches!(r, Err(StorageError::NotFound { .. })));
        let r = read(0..3, &hedges, |range| async move {
            Ok(response(range, body(vec![Err(not_found())])))
        })
        .await;
        assert!(matches!(r, Err(StorageError::NotFound { .. })));
    }

    /// A transient failure with no bytes ever arriving is asked for again only as many times as
    /// the count allows.
    #[tokio::test(start_paused = true)]
    async fn a_read_without_progress_stops_after_its_count() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let r = read(0..3, &hedges, |range| {
            opens.note(&range);
            async { Err(transient()) }
        })
        .await;
        assert!(matches!(r, Err(StorageError::TransientExhausted { .. })));
        assert_eq!(opens.count(), 1 + MAX_TRANSIENT_RETRIES as usize);
    }

    /// A body that stops sending is dropped after `BODY_IDLE`, and only its missing bytes are asked
    /// for again: no second request beside it, no bytes read twice.
    #[tokio::test(start_paused = true)]
    async fn a_stalled_body_is_dropped_and_only_its_tail_asked_for() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let started = time::Instant::now();
        let r = read(0..5, &hedges, |range| {
            let first = opens.note(&range) == 0;
            async move {
                if first {
                    Ok(response(range, stalling(vec![served(&(0..2))])))
                } else {
                    Ok(whole(range))
                }
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abcde");
        assert_eq!(opens.ranges(), vec![0..5, 2..5]);
        assert_eq!(started.elapsed(), BODY_IDLE + backoff(0));
        assert_eq!(
            hedges.hedge_count(),
            0,
            "a body stall needs no second request"
        );
    }

    /// A body that keeps moving, however slowly, is never dropped or hedged: one frame just under
    /// `BODY_IDLE` apart, for far longer than any delay.
    #[tokio::test(start_paused = true)]
    async fn a_body_that_keeps_moving_is_never_asked_for_again() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let gap = BODY_IDLE - Duration::from_millis(1);
        let started = time::Instant::now();
        let r = read(0..TRICKLE_FRAMES, &hedges, |range| {
            opens.note(&range);
            async move {
                let frames = stream::iter(range).then(move |i| async move {
                    time::sleep(gap).await;
                    Ok(Bytes::from(vec![i as u8]))
                });
                Ok(response(0..TRICKLE_FRAMES, frames.boxed()))
            }
        })
        .await
        .expect("test");
        assert_eq!(r.len() as u64, TRICKLE_FRAMES);
        assert_eq!(started.elapsed(), gap * TRICKLE_FRAMES as u32);
        assert_eq!((opens.count(), hedges.hedge_count()), (1, 0));
    }

    /// A body cut after slow frames is cut exactly `BODY_IDLE` after its last frame, not after its
    /// headers or an earlier frame: the idle time always counts from the latest frame.
    #[tokio::test(start_paused = true)]
    async fn a_body_is_cut_exactly_its_idle_time_after_its_last_frame() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let gap = BODY_IDLE - Duration::from_millis(1);
        let sent = FRAMES_BEFORE_CUT;
        let started = time::Instant::now();
        let r = read(0..OBJECT.len() as u64, &hedges, |range| {
            let first = opens.note(&range) == 0;
            async move {
                if !first {
                    return Ok(whole(range));
                }
                let frames = stream::iter(0..sent)
                    .then(move |i| async move {
                        time::sleep(gap).await;
                        Ok(served(&(i..i + 1)))
                    })
                    .chain(stream::pending());
                Ok(response(range, frames.boxed()))
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], OBJECT);
        assert_eq!(
            opens.ranges(),
            vec![0..OBJECT.len() as u64, sent..OBJECT.len() as u64]
        );
        assert_eq!(
            started.elapsed(),
            gap * sent as u32 + BODY_IDLE + backoff(0)
        );
    }

    /// A body stalled on every request ends the read with a transient error that names the stall.
    #[tokio::test(start_paused = true)]
    async fn a_stall_on_every_request_ends_in_a_transient_error() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let r = read(0..3, &hedges, |range| {
            opens.note(&range);
            async move { Ok(response(range, stalling(vec![]))) }
        })
        .await;
        let e = r.expect_err("stalled every time");
        assert!(matches!(e, StorageError::TransientExhausted { .. }));
        assert!(e.to_string().contains("no response bytes"), "{e}");
        assert_eq!(opens.count(), 1 + MAX_TRANSIENT_RETRIES as usize);
    }

    /// Headers that have not arrived after `STUCK_AFTER` get a second request, which answers.
    #[tokio::test(start_paused = true)]
    async fn late_headers_get_a_second_request() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let started = time::Instant::now();
        let r = read(0..3, &hedges, |range| {
            let first = opens.note(&range) == 0;
            async move {
                if first {
                    pending::<()>().await;
                }
                Ok(whole(range))
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abc");
        assert_eq!(started.elapsed(), STUCK_AFTER);
        assert_eq!((opens.count(), hedges.hedge_count()), (2, 1));
    }

    /// Headers that arrive inside `STUCK_AFTER` send nothing more, however long the body takes.
    #[tokio::test(start_paused = true)]
    async fn headers_in_time_send_no_second_request() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let r = read(0..3, &hedges, |range| {
            opens.note(&range);
            async move {
                time::sleep(STUCK_AFTER - Duration::from_millis(1)).await;
                let frames = stream::iter(range).then(|i| async move {
                    time::sleep(STUCK_AFTER).await;
                    Ok(served(&(i..i + 1)))
                });
                Ok(response(0..3, frames.boxed()))
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abc");
        assert_eq!((opens.count(), hedges.hedge_count()), (1, 0));
    }

    /// The request that loses the header race is dropped as soon as the other's headers arrive,
    /// before any of its body is read.
    #[tokio::test(start_paused = true)]
    async fn the_loser_of_a_header_race_is_dropped() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let dropped = Cell::new(false);
        let r = read(0..3, &hedges, |range| {
            let first = opens.note(&range) == 0;
            let dropped = &dropped;
            async move {
                if first {
                    let _guard = Dropped(dropped);
                    pending::<()>().await;
                }
                Ok(whole(range))
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abc");
        assert!(dropped.get(), "the stuck request was dropped");
    }

    /// A read past its delay while every permit is out waits for one, still racing its first
    /// request, and sends its second once one frees.
    #[tokio::test(start_paused = true)]
    async fn a_header_hedge_waits_for_a_permit() {
        let permits = bound(1);
        let held = permits.acquire().await.expect("permit");
        let freed_at = STUCK_AFTER * 2;
        spawn(async move {
            time::sleep(freed_at).await;
            drop(held);
        });
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let started = time::Instant::now();
        let hedging = Hedging {
            permits,
            meter: &hedges,
        };
        let r = time::timeout(
            TEST_READ_BOUND,
            read_range("u", 0..3, hedging, |range| {
                let first = opens.note(&range) == 0;
                async move {
                    if first {
                        pending::<()>().await;
                    }
                    Ok(whole(range))
                }
            }),
        )
        .await
        .expect("the read ends")
        .expect("test");
        assert_eq!(&r[..], b"abc");
        assert_eq!(started.elapsed(), freed_at);
        assert_eq!(hedges.hedge_count(), 1);
    }

    /// A hedge that fails frees its permit at once, so another read's stuck request gets its hedge,
    /// though the first read never finishes.
    #[tokio::test(start_paused = true)]
    async fn a_failed_hedge_frees_its_permit_for_another_read() {
        let permits = bound(1);
        let hedges = UsageMeter::new();
        let stuck_opens = Opens::new();
        let stuck = read_range(
            "u",
            0..3,
            Hedging {
                permits,
                meter: &hedges,
            },
            |range| {
                let first = stuck_opens.note(&range) == 0;
                async move {
                    if first {
                        pending::<()>().await;
                    }
                    time::sleep(HEDGE_FAILS_AFTER).await;
                    Err::<Response<'static>, _>(transient())
                }
            },
        );
        let opens = Opens::new();
        let started = time::Instant::now();
        let other = read_range(
            "u",
            0..3,
            Hedging {
                permits,
                meter: &hedges,
            },
            |range| {
                let first = opens.note(&range) == 0;
                async move {
                    if first {
                        pending::<()>().await;
                    }
                    Ok(whole(range))
                }
            },
        );
        let r = time::timeout(TEST_READ_BOUND, async {
            tokio::select! {
                biased;
                _ = stuck => panic!("every request of this read is stuck or fails"),
                r = other => r,
            }
        })
        .await
        .expect("the other read ends")
        .expect("test");
        assert_eq!(&r[..], b"abc");
        assert_eq!(started.elapsed(), STUCK_AFTER + HEDGE_FAILS_AFTER);
        assert_eq!(hedges.hedge_count(), 2);
    }

    /// A hedge whose answer comes later than `STUCK_AFTER` is not cut short: it answers the read
    /// the first request never would.
    #[tokio::test(start_paused = true)]
    async fn a_hedge_slower_than_the_delay_still_answers() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let started = time::Instant::now();
        let r = read(0..3, &hedges, |range| {
            let first = opens.note(&range) == 0;
            async move {
                if first {
                    pending::<()>().await;
                }
                time::sleep(SLOW_ANSWER).await;
                Ok(whole(range))
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abc");
        assert_eq!(started.elapsed(), STUCK_AFTER + SLOW_ANSWER);
        assert_eq!(hedges.hedge_count(), 1);
    }

    /// The first request failing ends the race at once: the second is dropped, not waited for.
    #[tokio::test(start_paused = true)]
    async fn the_first_requests_failure_ends_the_race() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let fails_at = STUCK_AFTER + PAST_DEADLINE;
        let started = time::Instant::now();
        let r = read(0..3, &hedges, |range| {
            let first = opens.note(&range) == 0;
            async move {
                if first {
                    time::sleep(fails_at).await;
                    Err(not_found())
                } else {
                    pending().await
                }
            }
        })
        .await;
        assert!(matches!(r, Err(StorageError::NotFound { .. })));
        assert_eq!(started.elapsed(), fails_at);
        assert_eq!(hedges.hedge_count(), 1);
    }

    /// A second request that fails leaves the first to answer.
    #[tokio::test(start_paused = true)]
    async fn a_failed_second_request_leaves_the_first_to_answer() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let answers_at = STUCK_AFTER * 2;
        let r = read(0..3, &hedges, |range| {
            let first = opens.note(&range) == 0;
            async move {
                if first {
                    time::sleep(answers_at).await;
                    Ok(whole(range))
                } else {
                    Err(transient())
                }
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abc");
        assert_eq!(opens.count(), 2);
    }

    /// A read whose every request fails slowly takes as long to give up as it would without
    /// hedging: each first request's failure ends its race, so the re-issues run back to back.
    #[tokio::test(start_paused = true)]
    async fn a_dead_read_gives_up_as_soon_as_without_hedging() {
        let hedges = UsageMeter::new();
        let fails_after = STUCK_AFTER + PAST_DEADLINE;
        let started = time::Instant::now();
        let r = read(0..3, &hedges, |_| async move {
            time::sleep(fails_after).await;
            Err::<Response<'static>, _>(transient())
        })
        .await;
        assert!(matches!(r, Err(StorageError::TransientExhausted { .. })));
        let backoffs: Duration = (0..MAX_TRANSIENT_RETRIES).map(backoff).sum();
        assert_eq!(
            started.elapsed(),
            fails_after * (1 + MAX_TRANSIENT_RETRIES) + backoffs
        );
        assert_eq!(hedges.hedge_count(), u64::from(1 + MAX_TRANSIENT_RETRIES));
    }

    /// A read that has gone `REISSUE_WITHIN` without a byte is not asked for again: its failure
    /// came after `object_store`'s own retries, which another issue would only repeat.
    #[tokio::test(start_paused = true)]
    async fn a_read_failing_after_its_window_is_not_asked_again() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let r = read(0..3, &hedges, |range| {
            let first = opens.note(&range) == 0;
            async move {
                if first {
                    time::sleep(REISSUE_WITHIN).await;
                    Err(transient())
                } else {
                    pending().await
                }
            }
        })
        .await;
        assert!(matches!(r, Err(StorageError::TransientExhausted { .. })));
        assert_eq!(
            opens.count() as u64,
            1 + hedges.hedge_count(),
            "the first request and its hedges, no re-issue"
        );
    }

    /// A range read keeps the bytes each request brings, so each frame starts the window and the
    /// count again: a long read whose requests keep failing after bringing bytes is asked for
    /// again past both and completes.
    #[tokio::test(start_paused = true)]
    async fn progress_starts_the_window_and_the_count_again() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let gap = BODY_IDLE - Duration::from_millis(1);
        let started = time::Instant::now();
        let r = read(0..PROGRESS_REISSUES, &hedges, |range| {
            opens.note(&range);
            async move {
                let start = range.start;
                let frames =
                    stream::iter([Ok(start), Err(transient())]).then(move |item| async move {
                        time::sleep(gap).await;
                        item.map(|i| Bytes::from(vec![i as u8]))
                    });
                Ok(response(range, frames.boxed()))
            }
        })
        .await
        .expect("test");
        assert_eq!(r.len() as u64, PROGRESS_REISSUES);
        assert!(started.elapsed() > REISSUE_WITHIN);
        assert_eq!(opens.count() as u64, PROGRESS_REISSUES);
    }

    /// A body that keeps sending frames past the window, then fails, is still asked again: each
    /// frame starts the window over, not only the response's headers.
    #[tokio::test(start_paused = true)]
    async fn frames_start_the_window_again() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let gap = BODY_IDLE - Duration::from_millis(1);
        let all = 0..OBJECT.len() as u64;
        let started = time::Instant::now();
        let r = read(all.clone(), &hedges, |range| {
            let first = opens.note(&range) == 0;
            async move {
                if !first {
                    return Ok(response(range.clone(), body(vec![Ok(served(&range))])));
                }
                let frames = (range.start..range.end - 1)
                    .map(|i| Ok(served(&(i..i + 1))))
                    .chain([Err(transient())]);
                let frames = stream::iter(frames).then(move |item| async move {
                    time::sleep(gap).await;
                    item
                });
                Ok(response(range, frames.boxed()))
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], OBJECT);
        assert!(started.elapsed() > REISSUE_WITHIN);
        assert_eq!(opens.count(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn read_whole_returns_the_body_and_its_meta() {
        let hedges = UsageMeter::new();
        let (bytes, meta) = read_all(&hedges, || async { Ok(Some(whole(0..5))) })
            .await
            .expect("test")
            .expect("a body");
        assert_eq!(&bytes[..], b"abcde");
        assert_eq!(meta.size, OBJECT.len() as u64);
    }

    #[tokio::test(start_paused = true)]
    async fn read_whole_returns_not_modified_without_asking_again() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let r = read_all(&hedges, || {
            opens.note(&(0..0));
            async { Ok(None) }
        })
        .await
        .expect("test");
        assert!(r.is_none());
        assert_eq!(opens.count(), 1);
    }

    /// A whole body that ends short of what its headers state is asked for again whole, and the
    /// complete one returned.
    #[tokio::test(start_paused = true)]
    async fn read_whole_asks_again_after_a_truncated_body() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let (bytes, _) = read_all(&hedges, || {
            let first = opens.note(&(0..5)) == 0;
            async move {
                if first {
                    Ok(Some(response(0..5, body(vec![Ok(served(&(0..3)))]))))
                } else {
                    Ok(Some(whole(0..5)))
                }
            }
        })
        .await
        .expect("test")
        .expect("a body");
        assert_eq!(&bytes[..], b"abcde");
        assert_eq!(opens.count(), 2);
    }

    /// A whole body that stalls is dropped after `BODY_IDLE` and asked for again whole.
    #[tokio::test(start_paused = true)]
    async fn read_whole_asks_again_after_a_stalled_body() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let started = time::Instant::now();
        let (bytes, _) = read_all(&hedges, || {
            let first = opens.note(&(0..5)) == 0;
            async move {
                if first {
                    Ok(Some(response(0..5, stalling(vec![served(&(0..2))]))))
                } else {
                    Ok(Some(whole(0..5)))
                }
            }
        })
        .await
        .expect("test")
        .expect("a body");
        assert_eq!(&bytes[..], b"abcde");
        assert_eq!(started.elapsed(), BODY_IDLE + backoff(0));
        assert_eq!(hedges.hedge_count(), 0);
    }

    /// A whole body that stays short past the budget is a stable short-read error, never a
    /// silently truncated object.
    #[tokio::test(start_paused = true)]
    async fn read_whole_persistent_truncation_is_short_read() {
        let hedges = UsageMeter::new();
        let r = read_all(&hedges, || async {
            Ok(Some(response(0..5, body(vec![Ok(served(&(0..3)))]))))
        })
        .await;
        let e = r.expect_err("persistent truncation must error");
        assert!(matches!(e, StorageError::Permanent { .. }));
        assert!(e.to_string().contains("short read"));
    }

    #[tokio::test(start_paused = true)]
    async fn read_whole_propagates_non_retryable() {
        let hedges = UsageMeter::new();
        let r = read_all(&hedges, || async { Err(not_found()) }).await;
        assert!(matches!(r, Err(StorageError::NotFound { .. })));
    }

    /// Late headers on a whole read get a second request too.
    #[tokio::test(start_paused = true)]
    async fn read_whole_hedges_late_headers() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let (bytes, _) = read_all(&hedges, || {
            let first = opens.note(&(0..5)) == 0;
            async move {
                if first {
                    pending::<()>().await;
                }
                Ok(Some(whole(0..5)))
            }
        })
        .await
        .expect("test")
        .expect("a body");
        assert_eq!(&bytes[..], b"abcde");
        assert_eq!(hedges.hedge_count(), 1);
    }

    /// A tail whose ETag differs from the bytes already gathered is never joined to them: the
    /// read starts the range again and returns the new version whole.
    #[tokio::test(start_paused = true)]
    async fn a_changed_object_restarts_the_range() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let r = read(0..5, &hedges, |range| {
            let n = opens.note(&range);
            async move {
                Ok(match n {
                    0 => tagged("old", response(range, stalling(vec![served(&(0..2))]))),
                    _ => tagged(
                        "new",
                        response(
                            range.clone(),
                            body(vec![Ok(Bytes::from(served(&range).to_ascii_uppercase()))]),
                        ),
                    ),
                })
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"ABCDE");
        assert_eq!(opens.ranges(), vec![0..5, 2..5, 0..5]);
    }

    /// A tail from the same version is joined to the bytes before it.
    #[tokio::test(start_paused = true)]
    async fn a_tail_from_the_same_version_is_kept() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let r = read(0..5, &hedges, |range| {
            let first = opens.note(&range) == 0;
            async move {
                Ok(if first {
                    tagged("v", response(range, stalling(vec![served(&(0..2))])))
                } else {
                    tagged("v", whole(range))
                })
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abcde");
        assert_eq!(opens.ranges(), vec![0..5, 2..5]);
    }

    /// A whole body longer than its headers state is not the object they describe: it is asked
    /// for again, and one that stays too long is an error, never cut down and returned.
    #[tokio::test(start_paused = true)]
    async fn read_whole_rejects_a_body_longer_than_its_headers() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let (bytes, _) = read_all(&hedges, || {
            let first = opens.note(&(0..3)) == 0;
            async move {
                Ok(Some(if first {
                    response(0..3, body(vec![Ok(served(&(0..3))), Ok(served(&(3..5)))]))
                } else {
                    whole(0..3)
                }))
            }
        })
        .await
        .expect("test")
        .expect("a body");
        assert_eq!(&bytes[..], b"abc");
        assert_eq!(opens.count(), 2);
        let r = read_all(&hedges, || async {
            Ok(Some(response(0..3, body(vec![Ok(served(&(0..5)))]))))
        })
        .await;
        let e = r.expect_err("a body that stays too long");
        assert!(matches!(e, StorageError::Permanent { .. }));
        assert!(e.to_string().contains("long read"), "{e}");
    }

    /// A second request the store refused answers for both: the stalled first is not waited for.
    #[tokio::test(start_paused = true)]
    async fn a_refused_second_request_answers_at_once() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let started = time::Instant::now();
        let r = read(0..3, &hedges, |range| {
            let first = opens.note(&range) == 0;
            async move {
                if first {
                    pending().await
                } else {
                    Err(not_found())
                }
            }
        })
        .await;
        assert!(matches!(r, Err(StorageError::NotFound { .. })));
        assert_eq!(started.elapsed(), STUCK_AFTER);
    }

    /// A read waiting for a permit still races its first request: headers that arrive meanwhile
    /// answer, and no second request is sent or counted.
    #[tokio::test(start_paused = true)]
    async fn headers_that_arrive_while_waiting_for_a_permit_answer() {
        let permits = bound(1);
        let _held = permits.acquire().await.expect("permit");
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let answers_at = STUCK_AFTER * 3 / 2;
        let started = time::Instant::now();
        let hedging = Hedging {
            permits,
            meter: &hedges,
        };
        let r = time::timeout(
            TEST_READ_BOUND,
            read_range("u", 0..3, hedging, |range| {
                opens.note(&range);
                async move {
                    time::sleep(answers_at).await;
                    Ok(whole(range))
                }
            }),
        )
        .await
        .expect("the read ends")
        .expect("test");
        assert_eq!(&r[..], b"abc");
        assert_eq!(started.elapsed(), answers_at);
        assert_eq!((opens.count(), hedges.hedge_count()), (1, 0));
    }

    /// A read dropped while its second request is out drops both requests and frees the permit.
    #[tokio::test(start_paused = true)]
    async fn a_read_dropped_mid_race_frees_its_permit() {
        let permits = bound(1);
        let hedges = UsageMeter::new();
        let read = read_range(
            "u",
            0..3,
            Hedging {
                permits,
                meter: &hedges,
            },
            |_| pending::<Result<Response<'static>, StorageError>>(),
        );
        let dropped_at = STUCK_AFTER * 3 / 2;
        assert!(
            time::timeout(dropped_at, read).await.is_err(),
            "every request is stuck"
        );
        assert_eq!((hedges.hedge_count(), permits.available_permits()), (1, 1));
    }

    /// A second request that wins gives its permit back before its body is read, so a slow body
    /// never holds a permit another read's hedge needs.
    #[tokio::test(start_paused = true)]
    async fn a_winning_hedge_frees_its_permit_before_its_body() {
        let permits = bound(1);
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let body_after = STUCK_AFTER / 2;
        let read = read_range(
            "u",
            0..3,
            Hedging {
                permits,
                meter: &hedges,
            },
            |range| {
                let first = opens.note(&range) == 0;
                async move {
                    if first {
                        pending::<()>().await;
                    }
                    let frame = served(&range);
                    let late = stream::once(async move {
                        time::sleep(body_after).await;
                        Ok(frame)
                    });
                    Ok(response(range, late.boxed()))
                }
            },
        );
        let during_body = async {
            time::sleep(STUCK_AFTER + body_after / 2).await;
            permits.available_permits()
        };
        let (r, free) = tokio::join!(time::timeout(TEST_READ_BOUND, read), during_body);
        assert_eq!(&r.expect("the read ends").expect("test")[..], b"abc");
        assert_eq!((hedges.hedge_count(), free), (1, 1));
    }

    /// A read whose object keeps changing between its requests still runs out of re-issues: bytes
    /// gathered again after starting over are no progress until they pass what it held before.
    #[tokio::test(start_paused = true)]
    async fn a_read_of_an_object_that_keeps_changing_ends() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let r = read(0..5, &hedges, |range| {
            let tag = if opens.note(&range).is_multiple_of(2) {
                "v1"
            } else {
                "v2"
            };
            let one = range.start..range.start + 1;
            async move { Ok(tagged(tag, response(range, body(vec![Ok(served(&one))])))) }
        })
        .await;
        assert!(r.is_err(), "a read that never gathers its bytes fails");
        assert!(
            opens.count() <= 2 * (1 + MAX_TRANSIENT_RETRIES as usize),
            "{}",
            opens.count()
        );
    }

    /// A closed bound sends no second request; the first answers.
    #[tokio::test(start_paused = true)]
    async fn a_closed_bound_sends_no_second_request() {
        let permits = bound(1);
        permits.close();
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let answers_at = STUCK_AFTER * 2;
        let hedging = Hedging {
            permits,
            meter: &hedges,
        };
        let r = time::timeout(
            TEST_READ_BOUND,
            read_range("u", 0..3, hedging, |range| {
                opens.note(&range);
                async move {
                    time::sleep(answers_at).await;
                    Ok(whole(range))
                }
            }),
        )
        .await
        .expect("the read ends")
        .expect("test");
        assert_eq!(&r[..], b"abc");
        assert_eq!((opens.count(), hedges.hedge_count()), (1, 0));
    }

    /// Headers arriving restart the window: a read whose headers came after `REISSUE_WITHIN` (the
    /// store retrying a throttle, say) still asks again for a body that then stalls.
    #[tokio::test(start_paused = true)]
    async fn late_headers_restart_the_window() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let answered = Cell::new(false);
        let late = REISSUE_WITHIN + PAST_DEADLINE;
        let r = read(0..3, &hedges, |range| {
            let first = opens.note(&range) == 0;
            let answered = &answered;
            async move {
                if first {
                    time::sleep(late).await;
                    answered.set(true);
                    Ok(response(range, stalling(vec![])))
                } else if answered.get() {
                    Ok(whole(range))
                } else {
                    pending().await
                }
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abc");
        assert_eq!(
            opens.count() as u64,
            2 + hedges.hedge_count(),
            "the late first, its hedges, and one re-issue"
        );
    }

    /// After a hedge fails, the next goes out `STUCK_AFTER` later, and answers.
    #[tokio::test(start_paused = true)]
    async fn a_failed_hedge_is_followed_by_the_next() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let started = time::Instant::now();
        let r = read(0..3, &hedges, |range| {
            let n = opens.note(&range);
            async move {
                match n {
                    0 => pending::<()>().await,
                    1 => {
                        time::sleep(HEDGE_FAILS_AFTER).await;
                        return Err(transient());
                    }
                    _ => {}
                }
                Ok(whole(range))
            }
        })
        .await
        .expect("test");
        assert_eq!(&r[..], b"abc");
        assert_eq!(started.elapsed(), STUCK_AFTER * 2 + HEDGE_FAILS_AFTER);
        assert_eq!(hedges.hedge_count(), 2);
    }

    /// Hedges hold their permits for their whole lives, so however long reads stay stuck, no more
    /// are in flight at once than the bound has permits.
    #[tokio::test(start_paused = true)]
    async fn hedges_in_flight_never_pass_the_bound() {
        let permits = bound(1);
        let hedges = UsageMeter::new();
        let live = Cell::new(0usize);
        let peak = Cell::new(0usize);
        let stuck = || {
            read_range(
                "u",
                0..3,
                Hedging {
                    permits,
                    meter: &hedges,
                },
                |_| {
                    live.set(live.get() + 1);
                    peak.set(peak.get().max(live.get()));
                    let guard = Live(&live);
                    async move {
                        let _guard = guard;
                        pending::<Result<Response<'static>, StorageError>>().await
                    }
                },
            )
        };
        let all = future::join3(stuck(), stuck(), stuck());
        assert!(time::timeout(STUCK_FOR, all).await.is_err());
        assert_eq!(
            peak.get(),
            STUCK_READS + 1,
            "every first request and one hedge"
        );
        assert_eq!(
            hedges.hedge_count(),
            1,
            "a stuck hedge holds its permit until it answers"
        );
    }

    /// A success ready in the same poll as a hedge's refusal wins.
    #[tokio::test(start_paused = true)]
    async fn a_success_ready_with_a_refusal_wins() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let at = STUCK_AFTER * 3 / 2;
        let r = read(0..3, &hedges, |range| {
            let first = opens.note(&range) == 0;
            async move {
                if first {
                    time::sleep(at).await;
                    Ok(whole(range))
                } else {
                    time::sleep(at - STUCK_AFTER).await;
                    Err(not_found())
                }
            }
        })
        .await
        .expect("the first request's answer");
        assert_eq!(&r[..], b"abc");
    }

    /// A whole body that goes quiet once all its stated bytes are in is whole.
    #[tokio::test(start_paused = true)]
    async fn read_whole_takes_a_full_body_that_then_goes_quiet() {
        let hedges = UsageMeter::new();
        let opens = Opens::new();
        let started = time::Instant::now();
        let (bytes, _) = read_all(&hedges, || {
            opens.note(&(0..5));
            async { Ok(Some(response(0..5, stalling(vec![served(&(0..5))])))) }
        })
        .await
        .expect("test")
        .expect("a body");
        assert_eq!(&bytes[..], b"abcde");
        assert_eq!(opens.count(), 1);
        // It waits one idle time for the end that never comes; a real body ends at its length.
        assert_eq!(started.elapsed(), BODY_IDLE);
    }

    /// A lone frame is returned as it came, without a copy; several are copied once into a buffer
    /// of the full size.
    #[test]
    fn gathered_copies_only_when_a_second_frame_arrives() {
        let frame = served(&(0..5));
        let mut lone = Gathered::new(5);
        lone.push(frame.clone());
        assert_eq!(lone.into_bytes().as_ptr(), frame.as_ptr());
        let mut many = Gathered::new(5);
        many.push(served(&(0..2)));
        many.push(served(&(2..5)));
        assert!(many.rest.capacity() >= 5);
        assert_eq!(&many.into_bytes()[..], b"abcde");
    }
}
