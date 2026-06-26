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
//! - [`load_resident`] loads the routing tree through the disk cache (one GET
//!   when [`OpannRouting::resident_uri`] + [`OpannRouting::resident_content_hash`]
//!   are stamped, via [`crate::supertable::manifest::commit::load_verified_blob`],
//!   otherwise a per-page walk for legacy manifests) and returns a
//!   [`ResidentPageSource`] whose pages are mmap slices.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use bytes::Bytes;
use futures::future::try_join_all;

use crate::get_meter::{GetPhaseGuard, GET_PHASE_OPANN};
use crate::storage::{StorageError, StorageProvider};
use crate::supertable::ManifestLoadError;
use crate::supertable::manifest::commit::{content_addressed_uri, load_verified_blob};
use crate::supertable::manifest::list::OpannRouting;
use crate::supertable::manifest::part::ContentHash;
use crate::supertable::reader_cache::DiskCacheStore;

use super::page::{Page, PageError};
use super::paged::{PageSource, ResidentPageSource};

/// Subdirectory (under the hidden vector-index prefix) that holds OPANN pages.
pub(crate) const OPANN_PAGES_DIR: &str = "opann-pages";

/// Magic prefix on packed resident bundles.
const RESIDENT_BUNDLE_MAGIC: &[u8; 4] = b"OPRS";

/// Wire format version for [`RESIDENT_BUNDLE_MAGIC`] bundles.
const RESIDENT_BUNDLE_VERSION: u8 = 1;

/// Header: magic (4) + version (1) + root hash (32) + page count (4).
const RESIDENT_BUNDLE_HEADER_LEN: usize = 4 + 1 + 32 + 4;

/// One directory entry: page hash (32) + payload offset (4) + payload len (4).
const RESIDENT_BUNDLE_DIR_ENTRY_LEN: usize = 32 + 4 + 4;

/// Storage URI for the page with content hash `hash`.
pub(crate) fn page_uri(hash: &ContentHash) -> String {
    content_addressed_uri(OPANN_PAGES_DIR, "page", hash, "opann")
}

/// Storage URI for the packed resident snapshot with content hash `hash`.
pub(crate) fn resident_uri(hash: &ContentHash) -> String {
    content_addressed_uri(OPANN_PAGES_DIR, "resident", hash, "opann")
}

/// Magic prefix on a packed deleted-user-`_id` blob.
const DELETED_IDS_MAGIC: &[u8; 4] = b"OPDS";

/// Wire-format version for [`DELETED_IDS_MAGIC`] blobs.
const DELETED_IDS_VERSION: u8 = 1;

/// Header: magic (4) + version (1) + count (4).
const DELETED_IDS_HEADER_LEN: usize = 4 + 1 + 4;

/// Bytes per serialized `_id` (a little-endian `i128`).
const DELETED_ID_LEN: usize = 16;

/// Storage URI for a packed deleted-`_id` blob with content hash `hash`.
pub(crate) fn deleted_ids_uri(hash: &ContentHash) -> String {
    content_addressed_uri(OPANN_PAGES_DIR, "deleted-ids", hash, "opann")
}

/// Serialize the consolidated deleted user-`_id` set. The caller passes a
/// sorted, deduplicated slice so the on-disk order is canonical — the same set
/// yields byte-identical blobs (content-addressed dedup) and the resident
/// reader can `binary_search` it directly.
pub(crate) fn encode_deleted_ids(sorted_ids: &[i128]) -> Vec<u8> {
    let mut out = Vec::with_capacity(DELETED_IDS_HEADER_LEN + sorted_ids.len() * DELETED_ID_LEN);
    out.extend_from_slice(DELETED_IDS_MAGIC);
    out.push(DELETED_IDS_VERSION);
    out.extend_from_slice(&(sorted_ids.len() as u32).to_le_bytes());
    for id in sorted_ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

/// Parse a deleted-`_id` blob written by [`encode_deleted_ids`]. Returns the
/// `_id`s in stored (ascending) order.
pub(crate) fn decode_deleted_ids(bytes: &[u8]) -> Result<Vec<i128>, PageError> {
    if bytes.len() < DELETED_IDS_HEADER_LEN {
        return Err(PageError::Truncated);
    }
    if &bytes[0..4] != DELETED_IDS_MAGIC {
        return Err(PageError::BadMagic);
    }
    let version = bytes[4];
    if version != DELETED_IDS_VERSION {
        return Err(PageError::UnsupportedVersion(version));
    }
    let count = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) as usize;
    let body = &bytes[DELETED_IDS_HEADER_LEN..];
    if body.len() != count * DELETED_ID_LEN {
        return Err(PageError::Truncated);
    }
    let mut ids = Vec::with_capacity(count);
    for chunk in body.chunks_exact(DELETED_ID_LEN) {
        let mut buf = [0u8; DELETED_ID_LEN];
        buf.copy_from_slice(chunk);
        ids.push(i128::from_le_bytes(buf));
    }
    Ok(ids)
}

/// Load the hidden index's consolidated deleted user-`_id` set for `routing`,
/// routed through the disk cache (mmap-backed, so a warm load is zero object
/// I/O — the set stays resident across queries, exactly like the routing
/// bundle). Returns an empty set when the manifest stamps no blob (legacy
/// manifests, or no deletes pending since the last drain).
pub(crate) async fn load_deleted_ids(
    routing: &OpannRouting,
    storage: &dyn StorageProvider,
    disk_cache: Option<&Arc<DiskCacheStore>>,
) -> Result<Vec<i128>, OpannStoreError> {
    let (Some(uri), Some(hash)) = (
        routing.deleted_ids_uri.as_deref(),
        routing.deleted_ids_content_hash,
    ) else {
        return Ok(Vec::new());
    };
    let _guard = GetPhaseGuard::new(GET_PHASE_OPANN);
    let bytes = load_verified_blob(hash, uri, storage, disk_cache).await?;
    Ok(decode_deleted_ids(&bytes)?)
}

/// Failures writing or loading OPANN pages.
#[derive(Debug, thiserror::Error)]
pub(crate) enum OpannStoreError {
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
    #[error("page error: {0}")]
    Page(#[from] PageError),
    #[error("verified blob load: {0}")]
    VerifiedBlob(#[from] ManifestLoadError),
}

/// Load the routing tree for `routing` through the disk cache and return a
/// [`ResidentPageSource`]. When a resident snapshot is stamped (`uri` +
/// `content_hash`, same fields as a manifest part), one object GET loads the
/// whole tree; legacy manifests without it fall back to a level-by-level page walk.
pub(crate) async fn load_resident(
    cache: Option<&Arc<DiskCacheStore>>,
    storage: &dyn StorageProvider,
    routing: &OpannRouting,
) -> Result<ResidentPageSource, OpannStoreError> {
    if let (Some(uri), Some(hash)) = (
        routing.resident_uri.as_deref(),
        routing.resident_content_hash,
    ) {
        return load_resident_from_snapshot(cache, storage, routing.root_page, hash, uri).await;
    }
    load_resident_page_walk(cache, storage, routing.root_page).await
}

/// Walk the page graph from `root`, fetching each page from object storage.
async fn load_resident_page_walk(
    cache: Option<&Arc<DiskCacheStore>>,
    storage: &dyn StorageProvider,
    root: ContentHash,
) -> Result<ResidentPageSource, OpannStoreError> {
    let pages = collect_reachable_pages_from_storage(cache, storage, root).await?;
    Ok(ResidentPageSource::from_byte_pages(pages))
}

/// One GET via [`load_verified_blob`], then slice pages out of the mmap.
async fn load_resident_from_snapshot(
    cache: Option<&Arc<DiskCacheStore>>,
    storage: &dyn StorageProvider,
    root: ContentHash,
    content_hash: ContentHash,
    uri: &str,
) -> Result<ResidentPageSource, OpannStoreError> {
    let _phase = GetPhaseGuard::new(GET_PHASE_OPANN);
    let bytes = load_verified_blob(content_hash, uri, storage, cache).await?;
    Ok(decode_resident_bundle(bytes, root)?)
}

/// Collect every page reachable from `root` via object storage (level-by-level
/// concurrent GETs). Used for legacy open paths and GC reachability.
async fn collect_reachable_pages_from_storage(
    cache: Option<&Arc<DiskCacheStore>>,
    storage: &dyn StorageProvider,
    root: ContentHash,
) -> Result<HashMap<ContentHash, Bytes>, OpannStoreError> {
    let mut pages: HashMap<ContentHash, Bytes> = HashMap::new();
    let mut frontier: Vec<ContentHash> = vec![root];
    while !frontier.is_empty() {
        // Dedup this level against what's already resident and against itself,
        // preserving first-seen order, so we fetch each distinct hash once.
        let mut seen: HashMap<ContentHash, ()> = HashMap::new();
        let mut to_fetch: Vec<(ContentHash, String)> = Vec::new();
        for hash in frontier {
            if pages.contains_key(&hash) {
                continue;
            }
            if seen.insert(hash, ()).is_none() {
                to_fetch.push((hash, page_uri(&hash)));
            }
        }
        if to_fetch.is_empty() {
            break;
        }
        let _phase = GetPhaseGuard::new(GET_PHASE_OPANN);
        let fetched = try_join_all(to_fetch.iter().map(|(h, uri)| {
            let hash = *h;
            let uri = uri.clone();
            async move {
                Ok::<_, OpannStoreError>((
                    hash,
                    load_verified_blob(hash, &uri, storage, cache).await?,
                ))
            }
        }))
        .await?;
        let mut next_frontier: Vec<ContentHash> = Vec::new();
        for (hash, bytes) in fetched {
            for child in unvisited_child_pages(&pages, bytes.as_ref())? {
                next_frontier.push(child);
            }
            pages.insert(hash, bytes);
        }
        frontier = next_frontier;
    }
    Ok(pages)
}

/// Walk `root` through `source` and pack every reachable page into one blob.
/// The returned bytes are content-addressed and PUT alongside individual pages
/// at commit so open can load the whole tree in one GET.
pub(crate) fn pack_resident_bundle(
    source: &dyn PageSource,
    root: ContentHash,
) -> Result<Vec<u8>, PageError> {
    let pages = collect_reachable_pages(source, root)?;
    encode_resident_bundle(root, &pages)
}

/// BFS over the page graph through an abstract [`PageSource`].
fn collect_reachable_pages(
    source: &dyn PageSource,
    root: ContentHash,
) -> Result<HashMap<ContentHash, Bytes>, PageError> {
    let mut pages: HashMap<ContentHash, Bytes> = HashMap::new();
    let mut queue: VecDeque<ContentHash> = VecDeque::from([root]);
    while let Some(hash) = queue.pop_front() {
        if pages.contains_key(&hash) {
            continue;
        }
        let bytes = source.fetch(&hash)?;
        let actual = ContentHash::of(bytes.as_ref());
        if actual != hash {
            return Err(PageError::ContentHashMismatch {
                expected: hash.to_hex(),
                actual: actual.to_hex(),
            });
        }
        for child in unvisited_child_pages(&pages, bytes.as_ref())? {
            queue.push_back(child);
        }
        pages.insert(hash, bytes);
    }
    Ok(pages)
}

/// Child page hashes referenced by `page_bytes` that are not already in `pages`.
fn unvisited_child_pages(
    pages: &HashMap<ContentHash, Bytes>,
    page_bytes: &[u8],
) -> Result<Vec<ContentHash>, PageError> {
    let page = Page::parse(page_bytes)?;
    Ok(page
        .referenced_pages()
        .into_iter()
        .filter(|child| !pages.contains_key(child))
        .collect())
}

/// Encode reachable pages into a deterministic bundle blob.
fn encode_resident_bundle(
    root: ContentHash,
    pages: &HashMap<ContentHash, Bytes>,
) -> Result<Vec<u8>, PageError> {
    let mut sorted: Vec<(ContentHash, Bytes)> = pages.iter().map(|(h, b)| (*h, b.clone())).collect();
    sorted.sort_by_key(|(h, _)| h.0);
    let n_pages = sorted.len();
    let header_len = RESIDENT_BUNDLE_HEADER_LEN;
    let dir_len = n_pages * RESIDENT_BUNDLE_DIR_ENTRY_LEN;
    let payload_len: usize = sorted.iter().map(|(_, b)| b.len()).sum();
    let mut out = Vec::with_capacity(header_len + dir_len + payload_len);
    out.extend_from_slice(RESIDENT_BUNDLE_MAGIC);
    out.push(RESIDENT_BUNDLE_VERSION);
    out.extend_from_slice(&root.0);
    out.extend_from_slice(&(n_pages as u32).to_le_bytes());
    let mut payload_off = 0u32;
    for (hash, bytes) in &sorted {
        out.extend_from_slice(&hash.0);
        out.extend_from_slice(&payload_off.to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        payload_off = payload_off
            .checked_add(bytes.len() as u32)
            .ok_or(PageError::BundleEncodeOverflow)?;
    }
    for (_, bytes) in &sorted {
        out.extend_from_slice(bytes);
    }
    Ok(out)
}

/// Slice pages out of a mmap-backed bundle blob.
fn decode_resident_bundle(bytes: Bytes, expected_root: ContentHash) -> Result<ResidentPageSource, PageError> {
    if bytes.len() < RESIDENT_BUNDLE_HEADER_LEN {
        return Err(PageError::BundleDecode("bundle too short for header".into()));
    }
    if bytes[..4] != RESIDENT_BUNDLE_MAGIC[..] {
        return Err(PageError::BundleDecode("bad bundle magic".into()));
    }
    if bytes[4] != RESIDENT_BUNDLE_VERSION {
        return Err(PageError::BundleDecode(format!(
            "unsupported bundle version {}",
            bytes[4]
        )));
    }
    let mut root_bytes = [0u8; 32];
    root_bytes.copy_from_slice(&bytes[5..37]);
    let root = ContentHash(root_bytes);
    if root != expected_root {
        return Err(PageError::BundleDecode(format!(
            "bundle root {} does not match manifest root {}",
            root.to_hex(),
            expected_root.to_hex()
        )));
    }
    let n_pages = u32::from_le_bytes(
        bytes[37..41]
            .try_into()
            .map_err(|_| PageError::BundleDecode("bad n_pages field".into()))?,
    );
    let dir_start = RESIDENT_BUNDLE_HEADER_LEN;
    let dir_end = dir_start
        .checked_add(n_pages as usize * RESIDENT_BUNDLE_DIR_ENTRY_LEN)
        .ok_or(PageError::BundleDecodeOverflow)?;
    if bytes.len() < dir_end {
        return Err(PageError::BundleDecode("bundle too short for directory".into()));
    }
    let payload_base = dir_end;
    let mut pages: HashMap<ContentHash, Bytes> = HashMap::with_capacity(n_pages as usize);
    for i in 0..n_pages as usize {
        let entry_off = dir_start + i * RESIDENT_BUNDLE_DIR_ENTRY_LEN;
        let mut hash_bytes = [0u8; 32];
        hash_bytes.copy_from_slice(&bytes[entry_off..entry_off + 32]);
        let hash = ContentHash(hash_bytes);
        let rel_off = u32::from_le_bytes(
            bytes[entry_off + 32..entry_off + 36]
                .try_into()
                .map_err(|_| PageError::BundleDecode("bad page offset".into()))?,
        );
        let len = u32::from_le_bytes(
            bytes[entry_off + 36..entry_off + 40]
                .try_into()
                .map_err(|_| PageError::BundleDecode("bad page length".into()))?,
        );
        let start = payload_base
            .checked_add(rel_off as usize)
            .ok_or(PageError::BundleDecodeOverflow)?;
        let end = start
            .checked_add(len as usize)
            .ok_or(PageError::BundleDecodeOverflow)?;
        if end > bytes.len() {
            return Err(PageError::BundleDecode(format!(
                "page {} slice [{start}, {end}) exceeds bundle len {}",
                hash.to_hex(),
                bytes.len()
            )));
        }
        let slice = bytes.slice(start..end);
        let actual = ContentHash::of(slice.as_ref());
        if actual != hash {
            return Err(PageError::ContentHashMismatch {
                expected: hash.to_hex(),
                actual: actual.to_hex(),
            });
        }
        pages.insert(hash, slice);
    }
    Ok(ResidentPageSource::from_byte_pages(pages))
}

/// Storage URIs of every routing-tree page reachable from `routing` — the live
/// page set for GC. Includes the resident bundle when stamped, plus every
/// individual page reachable from the root (copy-on-write orphans vs live).
pub(crate) async fn reachable_page_uris(
    storage: &dyn StorageProvider,
    routing: &OpannRouting,
) -> Result<HashSet<String>, OpannStoreError> {
    let mut live: HashSet<String> = HashSet::new();
    if let Some(uri) = &routing.resident_uri {
        live.insert(uri.clone());
    }
    let resident = load_resident_page_walk(None, storage, routing.root_page).await?;
    live.extend(resident.page_hashes().map(|h| page_uri(&h)));
    Ok(live)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LocalFsStorageProvider;
    use crate::superfile::vector::distance::Metric;
    use crate::supertable::manifest::commit::put_immutable_blob;
    use crate::supertable::manifest::list::{CellRoutingParams, OpannRouting};
    use crate::supertable::opann::paged::{PagedTree, SplitPages};
    use crate::supertable::opann::test_util::{build_tree, synth_cells};
    use bytes::Bytes;

    fn test_routing(root: ContentHash, resident_hash: Option<ContentHash>) -> OpannRouting {
        OpannRouting {
            root_page: root,
            routing: CellRoutingParams::default(),
            resident_uri: resident_hash.map(|h| resident_uri(&h)),
            resident_content_hash: resident_hash,
            deleted_ids_uri: None,
            deleted_ids_content_hash: None,
        }
    }

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

    #[test]
    fn deleted_ids_encode_decode_roundtrip() {
        // Canonical (sorted, deduped) set, including negatives and extremes —
        // `_id`s are signed 128-bit snowflakes.
        let ids: Vec<i128> = vec![i128::MIN, -1, 0, 1, 42, 1 << 100, i128::MAX];
        let bytes = encode_deleted_ids(&ids);
        assert_eq!(decode_deleted_ids(&bytes).expect("decode"), ids);

        // Empty set round-trips to an empty vec (header only).
        let empty = encode_deleted_ids(&[]);
        assert!(decode_deleted_ids(&empty).expect("decode empty").is_empty());

        // Same set → byte-identical blob (content-addressed dedup relies on it).
        assert_eq!(encode_deleted_ids(&ids), bytes);
    }

    #[test]
    fn deleted_ids_decode_rejects_corruption() {
        let good = encode_deleted_ids(&[1, 2, 3]);
        // Bad magic.
        let mut bad_magic = good.clone();
        bad_magic[0] ^= 0xff;
        assert!(matches!(
            decode_deleted_ids(&bad_magic),
            Err(PageError::BadMagic)
        ));
        // Truncated body (drop the last id's bytes).
        let truncated = &good[..good.len() - DELETED_ID_LEN];
        assert!(matches!(
            decode_deleted_ids(truncated),
            Err(PageError::Truncated)
        ));
        // Too-short for even the header.
        assert!(matches!(decode_deleted_ids(&[0u8; 3]), Err(PageError::Truncated)));
    }

    #[test]
    fn bundle_pack_roundtrip_matches_page_walk() {
        let (dim, n) = (24usize, 200usize);
        let cells = synth_cells(n, dim);
        let tree = build_tree(Metric::L2Sq, dim, &cells).expect("tree");
        let split = tree.to_pages(8);
        let source = ResidentPageSource::from_pages(split.pages.clone());
        let packed = pack_resident_bundle(&source, split.root).expect("pack");
        let bundle_hash = ContentHash::of(packed.as_ref());
        let from_bundle =
            decode_resident_bundle(Bytes::from(packed), split.root).expect("decode bundle");
        assert_eq!(from_bundle.len(), split.pages.len());
        let paged_bundle = PagedTree::new(from_bundle, split.root);
        let paged_direct = PagedTree::new(source, split.root);
        let q = &cells[57].0;
        assert_eq!(
            paged_direct.select_probes(q, 16).expect("direct"),
            paged_bundle.select_probes(q, 16).expect("bundle")
        );
        let routing = test_routing(split.root, Some(bundle_hash));
        assert!(routing.resident_content_hash.is_some());
        assert!(routing.resident_uri.is_some());
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
            let routing = test_routing(root, None);
            let source = load_resident(None, &storage, &routing)
                .await
                .expect("load pages");
            assert_eq!(
                source.len(),
                n_pages,
                "{metric:?}: load must reach every page from the root"
            );
            // The GC live-page set is exactly the reachable pages' URIs.
            let live = reachable_page_uris(&storage, &routing)
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
    async fn bundle_open_uses_one_get() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage = LocalFsStorageProvider::new(dir.path()).expect("local fs");
        let cells = synth_cells(128, 24);
        let tree = build_tree(Metric::L2Sq, 24, &cells).expect("tree");
        let split = tree.to_pages(8);
        put_test_pages(&storage, &split).await;
        let source = ResidentPageSource::from_pages(split.pages.clone());
        let packed = pack_resident_bundle(&source, split.root).expect("pack");
        let bundle_hash = ContentHash::of(packed.as_ref());
        put_immutable_blob(&storage, &resident_uri(&bundle_hash), Bytes::from(packed))
            .await
            .expect("put bundle");
        let routing = test_routing(split.root, Some(bundle_hash));
        let loaded = load_resident(None, &storage, &routing)
            .await
            .expect("load bundle");
        assert_eq!(loaded.len(), split.pages.len());
        let live = reachable_page_uris(&storage, &routing)
            .await
            .expect("live");
        assert!(live.contains(&resident_uri(&bundle_hash)));
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
        let mut latest: Option<(ContentHash, usize)> = None;
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
        let routing = test_routing(root, None);
        let source = load_resident(None, &storage, &routing)
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
