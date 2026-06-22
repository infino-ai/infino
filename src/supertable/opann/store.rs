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
//! Two operations:
//! - [`write_pages`] persists a [`SplitPages`] (write side, run at commit).
//! - [`load_resident`] walks the page graph from the root and returns a
//!   [`ResidentPageSource`] holding the whole tree in memory — the warm routing
//!   layer that descent then runs against with zero further object I/O.
//!
//! This module does not touch the manifest, commit, or query paths; stamping
//! the root hash into the manifest and routing `vector_search` through it are
//! later increments.

use std::collections::HashMap;

use bytes::Bytes;

use crate::storage::{StorageError, StorageProvider};
use crate::supertable::manifest::part::ContentHash;

use super::page::{Page, PageError};
use super::paged::{ResidentPageSource, SplitPages};

/// Subdirectory (under the hidden vector-index prefix) that holds OPANN pages.
const OPANN_PAGES_DIR: &str = "opann-pages";

/// Storage URI for the page with content hash `hash`. Mirrors the manifest
/// part scheme (`manifest-parts/part-<hash>.…`): a fixed dir plus the hex hash.
fn page_uri(hash: &ContentHash) -> String {
    format!("{OPANN_PAGES_DIR}/page-{}.opann", hash.to_hex())
}

/// Failures writing or loading OPANN pages.
#[derive(Debug, thiserror::Error)]
pub(crate) enum OpannStoreError {
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
    #[error("page error: {0}")]
    Page(#[from] PageError),
}

/// Persist every page of `pages` to object storage, content-addressed. A page
/// whose object already exists (identical content, racing or retried writer)
/// is a benign collision and is treated as success — exactly as manifest parts
/// handle [`StorageError::PreconditionFailed`].
///
/// Pages are PUT sequentially here; the routing tree is small and this runs off
/// the query path at commit time. (Overlapping the PUTs is a later refinement.)
pub(crate) async fn write_pages(
    storage: &dyn StorageProvider,
    pages: &SplitPages,
) -> Result<(), OpannStoreError> {
    for (hash, bytes) in &pages.pages {
        match storage
            .put_atomic(&page_uri(hash), Bytes::from(bytes.clone()))
            .await
        {
            Ok(_) => {}
            // Content-addressed: same hash → same bytes already there.
            Err(StorageError::PreconditionFailed { .. }) => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

/// Load the whole routing tree reachable from `root` into memory and return it
/// as a [`ResidentPageSource`]. Walks the page graph (each page names its child
/// pages by hash), fetching and **hash-verifying** every page exactly once.
/// This is the warm-up: after it returns, descent runs entirely in memory.
///
/// Fetches are sequential for now (the tree is small and this is a one-time,
/// amortized warm-up); level-parallel fetching is a later refinement.
pub(crate) async fn load_resident(
    storage: &dyn StorageProvider,
    root: ContentHash,
) -> Result<ResidentPageSource, OpannStoreError> {
    let mut pages: HashMap<ContentHash, Vec<u8>> = HashMap::new();
    let mut frontier: Vec<ContentHash> = vec![root];
    while let Some(hash) = frontier.pop() {
        if pages.contains_key(&hash) {
            continue;
        }
        let (bytes, _meta) = storage.get(&page_uri(&hash)).await?;
        let actual = ContentHash::of(bytes.as_ref());
        if actual != hash {
            return Err(PageError::ContentHashMismatch {
                expected: hash.to_hex(),
                actual: actual.to_hex(),
            }
            .into());
        }
        // Parse to discover the page's child pages, then keep the raw bytes
        // (descent re-parses from them through the PageSource).
        let page = Page::parse(bytes.as_ref())?;
        for child in page.referenced_pages() {
            if !pages.contains_key(&child) {
                frontier.push(child);
            }
        }
        pages.insert(hash, bytes.to_vec());
    }
    Ok(ResidentPageSource::from_pages(pages))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LocalFsStorageProvider;
    use crate::superfile::vector::distance::Metric;
    use crate::supertable::opann::paged::PagedTree;
    use crate::supertable::opann::test_util::{build_tree, synth_cells};

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

            write_pages(&storage, &split).await.expect("write pages");
            let source = load_resident(&storage, root).await.expect("load pages");
            assert_eq!(
                source.len(),
                n_pages,
                "{metric:?}: load must reach every page from the root"
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
        write_pages(&storage, &split).await.expect("first write");
        write_pages(&storage, &split).await.expect("idempotent rewrite");
    }
}
