// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Object-storage I/O for OPANN routing-tree pages.
//!
//! Pages are content-addressed immutable objects — the same discipline as
//! manifest parts: the blake3 of a page's bytes is its name, so a re-encoded
//! subtree that didn't change reuses its existing object, and a racing writer
//! that PUTs the same content collides benignly. Pages live under the hidden
//! vector-index supertable's storage (the caller passes that already-prefixed
//! [`StorageProvider`]); within it they sit in an [`OPANN_PAGES_DIR`] subdir.
//!
//! The page *write* side rides the commit: the changed pages of a copy-on-write
//! tree update travel in [`crate::supertable::writer::OpannRoutingCommit`] and
//! are PUT in the commit's parallel pre-pointer wave via
//! [`crate::supertable::manifest::commit::put_immutable_blob`] — the same
//! content-addressed blob writer manifest parts use. This module owns only the
//! page object name ([`page_uri`]) and the *read* side:
//! - [`load_resident`] walks the page graph from the root and returns a
//!   [`ResidentPageSource`] holding the whole tree in memory — the warm routing
//!   layer that descent then runs against with zero further object I/O.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bytes::Bytes;
use futures::future::try_join_all;

use crate::storage::{StorageError, StorageProvider};
use crate::supertable::manifest::part::ContentHash;
use crate::supertable::reader_cache::DiskCacheStore;
use crate::supertable::reader_cache::disk::DiskCacheError;

use super::page::{Page, PageError};
use super::paged::ResidentPageSource;

/// Subdirectory (under the hidden vector-index prefix) that holds OPANN pages.
pub(crate) const OPANN_PAGES_DIR: &str = "opann-pages";

/// Storage URI for the page with content hash `hash`. Mirrors the manifest
/// part scheme (`manifests/part-<hash>.…`): a fixed dir plus the hex hash.
/// Pages are written through the commit's content-addressed blob path
/// ([`crate::supertable::manifest::commit::put_immutable_blob`], the same one
/// manifest parts use); this just names the object.
pub(crate) fn page_uri(hash: &ContentHash) -> String {
    format!("{OPANN_PAGES_DIR}/page-{}.opann", hash.to_hex())
}

/// Failures writing or loading OPANN pages.
#[derive(Debug, thiserror::Error)]
pub(crate) enum OpannStoreError {
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
    #[error("page error: {0}")]
    Page(#[from] PageError),
    #[error("disk cache error: {0}")]
    Cache(#[from] DiskCacheError),
}

/// Load the whole routing tree reachable from `root` into memory and return it
/// as a [`ResidentPageSource`]. Walks the page graph (each page names its child
/// pages by hash), fetching and **hash-verifying** every page exactly once.
/// This is the warm-up: after it returns, descent runs entirely in memory.
///
/// Fetches go level-by-level: every distinct page of one tree level is fetched
/// concurrently (`try_join_all`), then its children seed the next level. A page
/// already resident (loaded as a shared child of an earlier level) is skipped,
/// so each page is fetched and verified exactly once even when the graph is a
/// DAG rather than a strict tree.
pub(crate) async fn load_resident(
    cache: Option<&Arc<DiskCacheStore>>,
    storage: &dyn StorageProvider,
    root: ContentHash,
) -> Result<ResidentPageSource, OpannStoreError> {
    let mut pages: HashMap<ContentHash, Bytes> = HashMap::new();
    let mut frontier: Vec<ContentHash> = vec![root];
    while !frontier.is_empty() {
        // Dedup this level against what's already resident and against itself,
        // preserving first-seen order, so we fetch each distinct hash once.
        let mut seen: HashMap<ContentHash, ()> = HashMap::new();
        let mut to_fetch: Vec<ContentHash> = Vec::new();
        for hash in frontier {
            if pages.contains_key(&hash) {
                continue;
            }
            if seen.insert(hash, ()).is_none() {
                to_fetch.push(hash);
            }
        }
        if to_fetch.is_empty() {
            break;
        }
        // Fetch every distinct page of this level concurrently. With a disk
        // cache attached each page rides `blob_bytes` (mmap-backed, evictable);
        // otherwise it falls back to a direct storage GET (heap bytes).
        let fetched = try_join_all(to_fetch.iter().map(|h| {
            let h = *h;
            async move { Ok::<_, OpannStoreError>((h, fetch_page(cache, storage, h).await?)) }
        }))
        .await?;
        let mut next_frontier: Vec<ContentHash> = Vec::new();
        for (hash, bytes) in fetched {
            // `blob_bytes` already content-verified on the cache path; the
            // re-check here is a cheap blake3 over already-resident bytes that
            // also covers the no-cache GET path.
            let actual = ContentHash::of(bytes.as_ref());
            if actual != hash {
                return Err(PageError::ContentHashMismatch {
                    expected: hash.to_hex(),
                    actual: actual.to_hex(),
                }
                .into());
            }
            // Parse to discover the page's child pages, then keep the (possibly
            // mmap-backed) bytes — descent re-parses from them through the
            // PageSource.
            let page = Page::parse(bytes.as_ref())?;
            for child in page.referenced_pages() {
                if !pages.contains_key(&child) {
                    next_frontier.push(child);
                }
            }
            pages.insert(hash, bytes);
        }
        frontier = next_frontier;
    }
    Ok(ResidentPageSource::from_byte_pages(pages))
}

/// Fetch one routing-tree page's bytes. With a disk cache, the page is pulled
/// through [`DiskCacheStore::blob_bytes`] — mmap-backed, content-verified, and
/// counted against the shared budget so old-version pages evict like superfile
/// blobs. Without one (e.g. the GC reachability walk), it's a direct GET.
async fn fetch_page(
    cache: Option<&Arc<DiskCacheStore>>,
    storage: &dyn StorageProvider,
    hash: ContentHash,
) -> Result<Bytes, OpannStoreError> {
    match cache {
        Some(c) => Ok(c.blob_bytes(hash, page_uri(&hash), storage).await?),
        None => {
            let (bytes, _meta) = storage.get(&page_uri(&hash)).await?;
            Ok(bytes)
        }
    }
}

/// Storage URIs of every routing-tree page reachable from `root` — the live
/// page set for GC. Walks the page graph (reusing [`load_resident`]) and maps
/// each reachable page hash to its object URI. GC adds these to its live set so
/// it sweeps only orphaned pages (superseded copy-on-write versions) and never
/// a page the current root still references.
pub(crate) async fn reachable_page_uris(
    storage: &dyn StorageProvider,
    root: ContentHash,
) -> Result<HashSet<String>, OpannStoreError> {
    // GC only needs the reachable hash set, not a warm cache — pass no cache so
    // a background reachability walk doesn't churn the query cache with pages.
    let resident = load_resident(None, storage, root).await?;
    Ok(resident.page_hashes().map(|h| page_uri(&h)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LocalFsStorageProvider;
    use crate::superfile::vector::distance::Metric;
    use crate::supertable::manifest::commit::put_immutable_blob;
    use crate::supertable::opann::paged::{PagedTree, SplitPages};
    use crate::supertable::opann::test_util::{build_tree, synth_cells};
    use bytes::Bytes;

    /// Write a tree's pages to `storage` through the shared content-addressed
    /// blob primitive (the production commit path), so the store round-trip
    /// tests exercise the same writer manifest parts use.
    async fn put_test_pages(storage: &dyn StorageProvider, split: &SplitPages) {
        for (hash, bytes) in &split.pages {
            put_immutable_blob(storage, &page_uri(hash), Bytes::from(bytes.clone()))
                .await
                .expect("put page");
        }
    }

    #[tokio::test]
    async fn storage_round_trip_descends_like_in_memory() {
        // Build a tree, split into pages, PUT them to a real (local-fs) store,
        // load them back by walking the page graph from the root, and confirm
        // descent off the loaded pages matches the in-memory tree exactly.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage = LocalFsStorageProvider::new(dir.path()).expect("local fs");
        let (dim, n) = (24usize, 200usize);
        let cells = synth_cells(n, dim);
        for metric in [Metric::L2Sq, Metric::Cosine, Metric::NegDot] {
            let tree = build_tree(metric, dim, &cells).expect("tree");
            let split = tree.to_pages(8);
            let root = split.root;
            let n_pages = split.pages.len();

            put_test_pages(&storage, &split).await;
            let source = load_resident(None, &storage, root)
                .await
                .expect("load pages");
            assert_eq!(
                source.len(),
                n_pages,
                "{metric:?}: load must reach every page from the root"
            );
            // The GC live-page set is exactly the reachable pages' URIs.
            let live = reachable_page_uris(&storage, root)
                .await
                .expect("live uris");
            assert_eq!(
                live.len(),
                n_pages,
                "{metric:?}: reachable_page_uris must cover every live page"
            );
            assert!(
                split.pages.keys().all(|h| live.contains(&page_uri(h))),
                "{metric:?}: every written page URI must be marked live"
            );

            let paged = PagedTree::new(source, root);
            for &target in &[0usize, 57, 199] {
                let q = &cells[target].0;
                for &k in &[1usize, 16, n] {
                    assert_eq!(
                        tree.select_probes(q, k),
                        paged.select_probes(q, k).expect("descend"),
                        "{metric:?} target {target} k {k}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn write_pages_is_idempotent() {
        // Re-writing the same pages (commit retry) collides benignly.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage = LocalFsStorageProvider::new(dir.path()).expect("local fs");
        let cells = synth_cells(64, 16);
        let tree = build_tree(Metric::L2Sq, 16, &cells).expect("tree");
        let split = tree.to_pages(8);
        put_test_pages(&storage, &split).await;
        put_test_pages(&storage, &split).await;
    }

    #[tokio::test]
    async fn multi_version_pages_load_consistently() {
        // The bench writes a fresh tree on every commit into the same store
        // (16 commits). Pages are content-addressed, so versions share unchanged
        // pages and accumulate the changed ones; loading the *latest* root must
        // still descend cleanly — no content-hash mismatch, no missing page.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage = LocalFsStorageProvider::new(dir.path()).expect("local fs");
        let (dim, n) = (24usize, 200usize);
        let mut latest: Option<(crate::supertable::manifest::part::ContentHash, usize)> = None;
        for round in 0..16u32 {
            let mut cells = synth_cells(n, dim);
            // Perturb so each round's tree (and its pages) differ, like the
            // shifting cell centroids/radii across successive commits.
            for (i, cell) in cells.iter_mut().enumerate() {
                cell.0[i % dim] += round as f32 * 0.017;
            }
            let tree = build_tree(Metric::L2Sq, dim, &cells).expect("tree");
            let split = tree.to_pages(8);
            put_test_pages(&storage, &split).await;
            latest = Some((split.root, split.pages.len()));
        }
        let (root, n_pages) = latest.expect("root");
        let source = load_resident(None, &storage, root)
            .await
            .expect("load latest root from accumulated multi-version store");
        assert_eq!(
            source.len(),
            n_pages,
            "latest root must reach exactly its own page set"
        );
        let paged = PagedTree::new(source, root);
        for &t in &[0usize, 37, 99, 199] {
            let q: Vec<f32> = (0..dim)
                .map(|d| (((t * 31 + d * 7 + 3) % 101) as f32) / 50.0 - 1.0)
                .collect();
            paged
                .select_probes(&q, n)
                .expect("descend without content-hash mismatch");
        }
    }
}
