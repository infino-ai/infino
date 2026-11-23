// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared test fixtures for the `reader/` submodule tests: blob builders that
//! plant small, known corpora so each test asserts against a fixed layout.

use std::{
    ops::Range,
    sync::{
        OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use tokio::{sync::RwLock, time::sleep};

use crate::superfile::{
    fts::builder::FtsBuilder,
    lazy_source::{LazyByteSource, LazyByteSourceError},
};

/// A one-column (`body`) `standard`-analyzer blob holding `docs`, doc `i`
/// being `docs[i]`.
pub(super) fn build_standard_blob(docs: &[&str]) -> (Bytes, String) {
    build_standard_blob_with(docs, None)
}

/// [`build_standard_blob`], when `order` is given storing its documents in
/// that order: blob position `i` holds `docs[order[i]]`, the way a merge
/// that reorders writes a blob.
pub(super) fn build_standard_blob_with(docs: &[&str], order: Option<&[u32]>) -> (Bytes, String) {
    let mut b = FtsBuilder::new();
    b.register_column("body".into(), false)
        .expect("register column");
    let rows: Vec<u32> = match order {
        Some(order) => order.to_vec(),
        None => (0..docs.len() as u32).collect(),
    };
    for (position, &row) in rows.iter().enumerate() {
        b.add_doc(0, position as u32, docs[row as usize])
            .expect("add doc");
    }
    if order.is_some() {
        b.doc_map = Some(rows);
    }
    let bytes = b.finish().expect("finish");
    let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
    (Bytes::from(bytes), json.to_string())
}

/// A `standard`-analyzer corpus whose vocabulary carries the two letters
/// Unicode case folding widens past lowercasing: `ſ` (long s, kept by
/// `to_lowercase`) and the Kelvin sign U+212A (lowercased to `k` at index).
/// Terms: k, kelvin, riſe, rise, set, sun, sunset, ſun.
pub(super) fn build_standard_fold_blob() -> (Bytes, String) {
    build_standard_blob(&["ſun riſe", "SUN set", "sunset rise", "Kelvin \u{212A}"])
}

pub(super) fn build_blob() -> (Bytes, String) {
    // 3 docs, 1 column.
    let mut b = FtsBuilder::new();
    b.register_column("body".into(), false)
        .expect("register column");
    b.add_doc(0, 0, "rust async runtime").expect("add doc");
    b.add_doc(0, 1, "tokio is a rust runtime").expect("add doc");
    b.add_doc(0, 2, "java spring boot").expect("add doc");
    let bytes = b.finish().expect("finish");
    let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
    (Bytes::from(bytes), json.to_string())
}

/// Build a corpus that exercises both the df=1 inline-encoded
/// path and the df ≥ 2 PFOR path side-by-side.
pub(super) fn build_mixed_df_blob() -> (Bytes, String) {
    let mut b = FtsBuilder::new();
    b.register_column("body".into(), false)
        .expect("register column");
    // `common`     → df = 3 (PFOR form)
    // `rust`       → df = 2 (PFOR form)
    // `uniqzero`  → df = 1 (inline form)
    // `uniqtwo`  → df = 1 (inline form)
    b.add_doc(0, 0, "common rust uniqzero").expect("add doc");
    b.add_doc(0, 1, "common rust").expect("add doc");
    b.add_doc(0, 2, "common uniqtwo").expect("add doc");
    let bytes = b.finish().expect("finish");
    let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
    (Bytes::from(bytes), json.to_string())
}

// ---- phrase atoms ----

/// Corpus with controlled adjacency for "new york": docs 0, 2
/// match (doc 4 twice); docs 1, 3 contain both words but never
/// adjacent in order.
pub(super) fn build_phrase_blob() -> (Bytes, &'static str) {
    use crate::superfile::fts::builder::FtsBuilder;
    let mut b = FtsBuilder::new();
    b.register_column("title".into(), true).expect("register");
    let docs = [
        "new york city",
        "york new haven",
        "the new york times",
        "new haven york",
        "new york new york",
    ];
    for (i, d) in docs.iter().enumerate() {
        b.add_doc(0, i as u32, d).expect("add doc");
    }
    (
        Bytes::from(b.finish().expect("finish")),
        r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#,
    )
}

/// A deterministic xorshift64 stream from `seed`, for tests that draw
/// reproducible inputs.
pub(super) fn xorshift(mut x: u64) -> impl FnMut() -> u64 {
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

/// A source that only serves bytes asynchronously, the way an
/// object store or a cold disk cache does.
///
/// `BytesLazyByteSource` answers a synchronous read from memory, so
/// a region left out of the open-time prefetch is still served and
/// the omission never shows. This one refuses, which is what turns
/// "the prefetch is missing a region the open needs" into a failure
/// rather than a slower path.
pub(crate) struct AsyncOnlySource(pub(crate) Bytes);

#[async_trait]
impl LazyByteSource for AsyncOnlySource {
    fn size(&self) -> u64 {
        self.0.len() as u64
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        let s = start as usize;
        let e = s + len as usize;
        match e <= self.0.len() {
            true => Ok(self.0.slice(s..e)),
            false => Err(LazyByteSourceError::OutOfBounds {
                start,
                len,
                size: self.0.len() as u64,
            }),
        }
    }

    fn try_get_range_sync(&self, _start: u64, _len: u64) -> Option<Bytes> {
        None
    }
}

/// [`AsyncOnlySource`] whose reads wait while a test holds `gate` for writing: a query can be
/// held mid-read, then let go or dropped. `started` counts the reads begun.
pub(super) struct GatedSource {
    pub(super) blob: Bytes,
    pub(super) gate: RwLock<()>,
    pub(super) started: AtomicUsize,
}

impl GatedSource {
    pub(super) fn new(blob: Bytes) -> Self {
        Self {
            blob,
            gate: RwLock::new(()),
            started: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl LazyByteSource for GatedSource {
    fn size(&self) -> u64 {
        self.blob.len() as u64
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        self.started.fetch_add(1, Ordering::Relaxed);
        let _open = self.gate.read().await;
        AsyncOnlySource(self.blob.clone()).range(start, len).await
    }
}

/// [`AsyncOnlySource`] that refuses any read overlapping `refused` once a
/// test sets it, so a query that must not touch a region (such as the
/// dictionary) fails if it does.
pub(super) struct RefusingSource {
    pub(super) blob: Bytes,
    pub(super) refused: OnceLock<Range<usize>>,
}

#[async_trait]
impl LazyByteSource for RefusingSource {
    fn size(&self) -> u64 {
        self.blob.len() as u64
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        let (s, e) = (start as usize, (start + len) as usize);
        if let Some(refused) = self.refused.get()
            && s < refused.end
            && e > refused.start
        {
            // Reported as out of bounds: the region is outside what the
            // test lets the query read.
            return Err(LazyByteSourceError::OutOfBounds {
                start,
                len,
                size: self.size(),
            });
        }
        AsyncOnlySource(self.blob.clone()).range(start, len).await
    }

    fn try_get_range_sync(&self, _start: u64, _len: u64) -> Option<Bytes> {
        None
    }
}

/// [`AsyncOnlySource`] whose reads take longer the earlier they start in the blob, so reads in
/// flight together come back last first.
pub(super) struct LaterFirstSource(pub(super) Bytes);

#[async_trait]
impl LazyByteSource for LaterFirstSource {
    fn size(&self) -> u64 {
        self.0.len() as u64
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        sleep(Duration::from_micros(self.size().saturating_sub(start))).await;
        AsyncOnlySource(self.0.clone()).range(start, len).await
    }
}
