// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A plain-HTTP object endpoint for tests whose first GET stalls, before its headers or partway
//! through its body, and which answers every later GET at once. The S3, GCS and Azure clients all
//! read a range with a `Range` header (`bytes=a-b`, or `bytes=-n` for the last bytes) and accept a
//! 206 with `Content-Range`, `ETag` and `Last-Modified`, or a 200 for a whole object, so one server
//! stands in for all three and shows each store's reads recovering a stalled GET through its real
//! client.

use std::{
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    ops::Range,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use crate::{runtime_metrics::io::UsageMeter, storage::StorageProvider};

/// Bytes of the object the endpoint serves.
const STALL_OBJECT_BYTES: usize = 4096;

/// The range a [`Read::Range`] reads, and the length a [`Read::Tail`] reads from the end.
const READ_BYTES: usize = 1024;

/// Bytes of the first GET's body the endpoint sends before a [`Stall::MidBody`] stall.
const SENT_BEFORE_STALL: usize = 512;

/// How long the endpoint holds its first GET: far past a hedge's delay and a body's idle timeout.
const STALL_HOLD: Duration = Duration::from_secs(60);

/// The read's bound in a stalled-GET test: under every client's request timeout (30 s on S3,
/// 60 s on GCS and Azure), so only the read's own recovery can meet it.
const RECOVERED_READ_BOUND: Duration = Duration::from_secs(10);

/// The endpoint's bytes repeat with this period, a prime, so a slice served from the wrong offset
/// doesn't match.
const STALL_PATTERN_PERIOD: usize = 251;

/// Where the endpoint's first GET stalls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Stall {
    /// Before any byte of the response: its headers never arrive.
    BeforeHeaders,
    /// After its headers and [`SENT_BEFORE_STALL`] bytes of its body.
    MidBody,
}

/// Which provider read a stalled-GET test makes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Read {
    /// `get_range` of the object's first [`READ_BYTES`].
    Range,
    /// `tail` of its last [`READ_BYTES`].
    Tail,
    /// `get` of the whole object.
    Whole,
}

impl Read {
    /// The bytes of the object this read returns.
    fn bytes(self) -> Range<usize> {
        match self {
            Self::Range => 0..READ_BYTES,
            Self::Tail => STALL_OBJECT_BYTES - READ_BYTES..STALL_OBJECT_BYTES,
            Self::Whole => 0..STALL_OBJECT_BYTES,
        }
    }
}

/// A running endpoint: its base URL, and the byte ranges of the GETs it has been sent, in order.
pub(crate) struct Endpoint {
    pub(crate) url: String,
    gets: Arc<Mutex<Vec<Range<usize>>>>,
}

/// The byte the endpoint serves at `offset`.
fn stall_byte(offset: usize) -> u8 {
    (offset % STALL_PATTERN_PERIOD) as u8
}

/// Start an endpoint whose first GET stalls as `stall` says, on a free port.
pub(crate) fn stalling_endpoint(stall: Stall) -> Endpoint {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let gets = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&gets);
    thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(conn) = conn else { return };
            let seen = Arc::clone(&seen);
            thread::spawn(move || serve(conn, stall, &seen));
        }
    });
    Endpoint {
        url: format!("http://{addr}"),
        gets,
    }
}

/// The bytes a `Range` header's value (after `bytes=`) asks for.
fn requested(spec: &str) -> Range<usize> {
    let (start, end) = spec.trim().split_once('-').expect("a byte range");
    if start.is_empty() {
        let len: usize = end.parse().expect("suffix length");
        return STALL_OBJECT_BYTES - len..STALL_OBJECT_BYTES;
    }
    let end: usize = end.parse().expect("range end");
    start.parse().expect("range start")..end + 1
}

/// Serve the requests on `conn` as [`stalling_endpoint`] describes, until it closes.
fn serve(mut conn: TcpStream, stall: Stall, gets: &Mutex<Vec<Range<usize>>>) {
    let mut reader = BufReader::new(conn.try_clone().expect("clone"));
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let mut range = None;
        loop {
            let mut header = String::new();
            if reader.read_line(&mut header).unwrap_or(0) == 0 {
                return;
            }
            if header == "\r\n" {
                break;
            }
            if let Some(spec) = header.to_ascii_lowercase().strip_prefix("range: bytes=") {
                range = Some(requested(spec));
            }
        }
        let partial = range.is_some();
        let range = range.unwrap_or(0..STALL_OBJECT_BYTES);
        let first = {
            let mut gets = gets.lock().expect("gets");
            gets.push(range.clone());
            gets.len() == 1
        };
        if first && stall == Stall::BeforeHeaders {
            thread::sleep(STALL_HOLD);
            return;
        }
        let body: Vec<u8> = range.clone().map(stall_byte).collect();
        let status = if partial {
            format!(
                "206 Partial Content\r\nContent-Range: bytes {}-{}/{STALL_OBJECT_BYTES}",
                range.start,
                range.end - 1
            )
        } else {
            "200 OK".to_string()
        };
        let head = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nETag: \"stall\"\r\n\
             Last-Modified: Wed, 21 Oct 2015 07:28:00 GMT\r\n\r\n",
            body.len(),
        );
        if conn.write_all(head.as_bytes()).is_err() {
            return;
        }
        if first {
            let _ = conn.write_all(&body[..SENT_BEFORE_STALL]);
            let _ = conn.flush();
            thread::sleep(STALL_HOLD);
            return;
        }
        if conn.write_all(&body).is_err() {
            return;
        }
    }
}

/// Make `read` through `provider`, whose store is `endpoint` stalling as `stall` says, and check
/// it recovered well inside the client's request timeout with the right bytes and one more GET:
/// beside the first when its headers stall (a hedge); when its body stalls, for the missing bytes
/// alone on a range read, or for the whole request again on a tail or whole read. `meter` is the
/// provider's: one read, and one hedge only for late headers.
pub(crate) async fn assert_stalled_read_recovers(
    provider: &dyn StorageProvider,
    meter: &UsageMeter,
    endpoint: &Endpoint,
    stall: Stall,
    read: Read,
) {
    let started = Instant::now();
    let bytes = match read {
        Read::Range => provider.get_range("obj", 0..READ_BYTES as u64).await,
        Read::Tail => provider
            .tail("obj", READ_BYTES as u64)
            .await
            .map(|(bytes, size)| {
                assert_eq!(size, STALL_OBJECT_BYTES as u64);
                bytes
            }),
        Read::Whole => provider.get("obj").await.map(|(bytes, meta)| {
            assert_eq!(meta.size, STALL_OBJECT_BYTES as u64);
            bytes
        }),
    }
    .expect("the read recovers");
    assert!(
        started.elapsed() < RECOVERED_READ_BOUND,
        "recovered in {:?}",
        started.elapsed()
    );
    let want = read.bytes();
    assert_eq!(bytes, want.clone().map(stall_byte).collect::<Vec<u8>>());
    let second = match (stall, read) {
        (Stall::MidBody, Read::Range) => want.start + SENT_BEFORE_STALL..want.end,
        _ => want.clone(),
    };
    assert_eq!(*endpoint.gets.lock().expect("gets"), vec![want, second]);
    let hedges = u64::from(stall == Stall::BeforeHeaders);
    assert_eq!(
        (meter.snapshot().get_count, meter.hedge_count()),
        (1, hedges)
    );
}
