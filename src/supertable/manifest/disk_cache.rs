// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`ManifestDiskCache`] — a content-addressed on-disk cache for the
//! compressed (Avro+zstd) bytes of manifest parts.
//!
//! Manifest parts are immutable and addressed by the blake3 hash of
//! their compressed bytes (see [`crate::supertable::manifest::part`]).
//! That makes them an ideal disk-cache target: the content hash is the
//! key, so a cached file can never be stale — if the logical content
//! changes, the hash changes, and the new content lands under a new
//! filename. No invalidation logic is needed; old files simply become
//! eviction candidates.
//!
//! ## What this buys
//!
//! [`ManifestPartLoader`](super::ManifestPartLoader) consults this
//! cache before issuing a `StorageProvider::get`. On a hit the part
//! bytes come off local disk instead of object storage, eliminating
//! the network round-trip (the dominant cost on a cold part load over
//! S3). The cache also survives process restarts: the on-disk files
//! persist and the in-memory accounting index is rebuilt by a
//! directory scan at construction.
//!
//! ## What this does NOT cache
//!
//! Superfile *byte content* is cached one layer over, by
//! [`DiskCacheStore`](crate::supertable::reader_cache::DiskCacheStore).
//! This cache holds only manifest-part bytes. The two caches are
//! independent and use separate byte budgets.

use std::{
    fs,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use dashmap::{DashMap, Entry};

use crate::supertable::manifest::part::ContentHash;

/// File-name prefix for a cached manifest part. The blake3 hex of the
/// part's compressed bytes follows; the `.avro.zst` suffix mirrors the
/// storage-side layout so the on-disk file is self-describing.
const CACHE_FILE_PREFIX: &str = "part-";
/// File-name suffix for a cached manifest part. Matches the
/// storage-side part object naming.
const CACHE_FILE_SUFFIX: &str = ".avro.zst";
/// File-name suffix for an in-flight cache write, atomically renamed
/// onto the final name once the bytes are durable.
const CACHE_TMP_SUFFIX: &str = ".tmp";

/// Per-entry bookkeeping for the in-memory accounting index. The bytes
/// themselves live on disk; this only tracks what eviction needs.
struct CacheEntry {
    size_bytes: u64,
    /// Microseconds-since-construction of the last access. Monotonic
    /// per [`ManifestDiskCache`] instance; drives LRU eviction.
    last_access_us: AtomicU64,
}

/// Snapshot of the manifest cache's load. Surfaced via
/// [`ManifestDiskCache::stats`] for tests and observability.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ManifestCacheStats {
    pub n_entries: u64,
    pub current_bytes: u64,
    pub budget_bytes: u64,
    pub n_hits: u64,
    pub n_misses: u64,
    pub n_evictions: u64,
}

/// Content-addressed disk cache for manifest-part bytes.
///
/// Construction is sync (a directory scan); `get` / `put` are async
/// because file I/O is handed to `tokio::fs` so tokio workers aren't
/// blocked on the syscall. Verification (blake3 over the cached bytes)
/// runs inline, matching the loader's existing hash check.
pub struct ManifestDiskCache {
    cache_root: PathBuf,
    budget_bytes: u64,
    started_at: Instant,
    /// Accounting index: content hash → size + last-access. The bytes
    /// are on disk under [`Self::cache_path`]; this map exists only so
    /// eviction can pick victims without stat-ing the directory.
    entries: DashMap<ContentHash, CacheEntry>,
    current_bytes: AtomicU64,
    n_hits: AtomicU64,
    n_misses: AtomicU64,
    n_evictions: AtomicU64,
}

impl ManifestDiskCache {
    /// Construct a cache rooted at `cache_root` (created if absent)
    /// with a byte budget of `budget_bytes`. Scans `cache_root` for
    /// pre-existing `part-<hex>.avro.zst` files and seeds the
    /// accounting index from them, so a freshly-opened process reuses
    /// the parts a prior run cached (restart survival).
    pub fn new(cache_root: PathBuf, budget_bytes: u64) -> std::io::Result<Arc<Self>> {
        fs::create_dir_all(&cache_root)?;
        let cache = Self {
            cache_root,
            budget_bytes,
            started_at: Instant::now(),
            entries: DashMap::new(),
            current_bytes: AtomicU64::new(0),
            n_hits: AtomicU64::new(0),
            n_misses: AtomicU64::new(0),
            n_evictions: AtomicU64::new(0),
        };
        cache.scan_existing();
        Ok(Arc::new(cache))
    }

    /// Walk the cache directory and register every well-formed cache
    /// file in the accounting index. A leftover `.tmp` file (an
    /// interrupted write) is removed. Unparseable names are ignored.
    fn scan_existing(&self) {
        let Ok(read_dir) = fs::read_dir(&self.cache_root) else {
            return;
        };
        let now = self.now_us();
        for dirent in read_dir.flatten() {
            let path = dirent.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name.ends_with(CACHE_TMP_SUFFIX) {
                let _ = fs::remove_file(&path);
                continue;
            }
            let Some(hash) = parse_cache_filename(name) else {
                continue;
            };
            let Ok(meta) = dirent.metadata() else {
                continue;
            };
            let size = meta.len();
            self.entries.insert(
                hash,
                CacheEntry {
                    size_bytes: size,
                    last_access_us: AtomicU64::new(now),
                },
            );
            self.current_bytes.fetch_add(size, Ordering::AcqRel);
        }
    }

    /// Look up a part's compressed bytes by content hash.
    ///
    /// Returns `Some(bytes)` on a verified hit and `None` on a miss.
    /// The cached bytes are re-hashed and compared against the
    /// requested `hash`; a mismatch (disk corruption) is treated as a
    /// miss and the bad file is removed, so the loader transparently
    /// falls back to storage.
    pub async fn get(&self, hash: &ContentHash) -> Option<Vec<u8>> {
        // Fast negative: not in the accounting index ⇒ not cached.
        if !self.entries.contains_key(hash) {
            self.n_misses.fetch_add(1, Ordering::AcqRel);
            return None;
        }
        let path = self.cache_path(hash);
        let bytes = match tokio::fs::read(&path).await {
            Ok(b) => b,
            Err(_) => {
                // File vanished out from under the index (manual
                // delete, external eviction). Drop the stale entry.
                self.drop_entry(hash);
                self.n_misses.fetch_add(1, Ordering::AcqRel);
                return None;
            }
        };
        // Content-addressing guarantee: the bytes must hash to the key.
        // A mismatch means on-disk corruption — discard and miss.
        if ContentHash::of(&bytes) != *hash {
            let _ = tokio::fs::remove_file(&path).await;
            self.drop_entry(hash);
            self.n_misses.fetch_add(1, Ordering::AcqRel);
            return None;
        }
        if let Some(entry) = self.entries.get(hash) {
            entry.last_access_us.store(self.now_us(), Ordering::Release);
        }
        self.n_hits.fetch_add(1, Ordering::AcqRel);
        Some(bytes)
    }

    /// Insert a part's compressed bytes under its content hash.
    ///
    /// Best-effort: the cache is a pure optimization, so any failure
    /// (budget exhausted with no evictable victims, I/O error) is
    /// swallowed — the caller already holds the bytes and can proceed.
    /// Idempotent: an already-cached hash is a no-op. `hash` must be
    /// the blake3 of `bytes`; the caller (the loader) has already
    /// verified this.
    pub async fn put(&self, hash: ContentHash, bytes: &[u8]) {
        if self.entries.contains_key(&hash) {
            return;
        }
        let size = bytes.len() as u64;
        // A single part larger than the whole budget can never fit;
        // don't bother evicting everything for a doomed insert.
        if size > self.budget_bytes {
            return;
        }
        if !self.reserve(size) {
            return;
        }

        let final_path = self.cache_path(&hash);
        let tmp_path = self.tmp_path(&hash);
        let write_result = async {
            tokio::fs::write(&tmp_path, bytes).await?;
            tokio::fs::rename(&tmp_path, &final_path).await
        }
        .await;

        if write_result.is_err() {
            // Roll back the reservation; leave no partial tmp behind.
            self.current_bytes.fetch_sub(size, Ordering::Release);
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return;
        }

        // Commit into the accounting index. If a concurrent put raced
        // us to the same hash, release our reservation — the file on
        // disk is byte-identical either way (content addressing).
        match self.entries.entry(hash) {
            Entry::Vacant(v) => {
                v.insert(CacheEntry {
                    size_bytes: size,
                    last_access_us: AtomicU64::new(self.now_us()),
                });
            }
            Entry::Occupied(_) => {
                self.current_bytes.fetch_sub(size, Ordering::Release);
            }
        }
    }

    /// Reserve `size` bytes of budget, evicting LRU victims as needed.
    /// Returns `false` if the budget can't be made to fit (every entry
    /// got evicted and there still isn't room — only possible under a
    /// race, since oversized parts are rejected before calling this).
    fn reserve(&self, size: u64) -> bool {
        loop {
            let cur = self.current_bytes.load(Ordering::Acquire);
            if cur + size <= self.budget_bytes {
                if self
                    .current_bytes
                    .compare_exchange_weak(cur, cur + size, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return true;
                }
                continue;
            }
            let needed = (cur + size).saturating_sub(self.budget_bytes);
            if !self.evict_at_least(needed) {
                return false;
            }
        }
    }

    /// Evict least-recently-accessed entries until at least
    /// `bytes_needed` is freed. Returns `false` if there were no
    /// entries left to evict before reaching the target.
    fn evict_at_least(&self, bytes_needed: u64) -> bool {
        let mut candidates: Vec<(ContentHash, u64, u64)> = self
            .entries
            .iter()
            .map(|e| {
                (
                    *e.key(),
                    e.value().size_bytes,
                    e.value().last_access_us.load(Ordering::Acquire),
                )
            })
            .collect();
        // Oldest-access first.
        candidates.sort_by_key(|(_, _, last)| *last);

        let mut freed = 0u64;
        for (hash, size, _) in candidates {
            if freed >= bytes_needed {
                break;
            }
            // Atomic gate: only the caller that wins the remove runs
            // the unlink + decrement, so concurrent evictions of the
            // same victim can't double-count.
            if self.entries.remove(&hash).is_some() {
                let _ = fs::remove_file(self.cache_path(&hash));
                self.current_bytes.fetch_sub(size, Ordering::Release);
                self.n_evictions.fetch_add(1, Ordering::AcqRel);
                freed = freed.saturating_add(size);
            }
        }
        freed >= bytes_needed
    }

    /// Remove an entry from the accounting index and release its
    /// reserved bytes. The on-disk file is left to the caller.
    fn drop_entry(&self, hash: &ContentHash) {
        if let Some((_, entry)) = self.entries.remove(hash) {
            self.current_bytes
                .fetch_sub(entry.size_bytes, Ordering::Release);
        }
    }

    /// Snapshot of the cache's load. Cheap: reads atomics + a
    /// `DashMap::len`.
    pub fn stats(&self) -> ManifestCacheStats {
        ManifestCacheStats {
            n_entries: self.entries.len() as u64,
            current_bytes: self.current_bytes.load(Ordering::Acquire),
            budget_bytes: self.budget_bytes,
            n_hits: self.n_hits.load(Ordering::Acquire),
            n_misses: self.n_misses.load(Ordering::Acquire),
            n_evictions: self.n_evictions.load(Ordering::Acquire),
        }
    }

    fn now_us(&self) -> u64 {
        self.started_at.elapsed().as_micros() as u64
    }

    fn cache_path(&self, hash: &ContentHash) -> PathBuf {
        self.cache_root.join(cache_filename(hash))
    }

    fn tmp_path(&self, hash: &ContentHash) -> PathBuf {
        self.cache_root
            .join(format!("{}{CACHE_TMP_SUFFIX}", cache_filename(hash)))
    }
}

impl std::fmt::Debug for ManifestDiskCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManifestDiskCache")
            .field("cache_root", &self.cache_root)
            .field("budget_bytes", &self.budget_bytes)
            .field("current_bytes", &self.current_bytes.load(Ordering::Acquire))
            .field("n_entries", &self.entries.len())
            .finish()
    }
}

/// Build the on-disk file name for a cached part: `part-<hex>.avro.zst`.
fn cache_filename(hash: &ContentHash) -> String {
    format!("{CACHE_FILE_PREFIX}{}{CACHE_FILE_SUFFIX}", hash.to_hex())
}

/// Parse a cache file name back into its [`ContentHash`]. Returns
/// `None` for any name that doesn't match the `part-<hex>.avro.zst`
/// shape (so unrelated files in the directory are ignored).
fn parse_cache_filename(name: &str) -> Option<ContentHash> {
    let hex = name
        .strip_prefix(CACHE_FILE_PREFIX)?
        .strip_suffix(CACHE_FILE_SUFFIX)?;
    ContentHash::from_hex(hex)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic content hash for a payload (mirrors the loader's
    /// own `ContentHash::of`).
    fn hash_of(bytes: &[u8]) -> ContentHash {
        ContentHash::of(bytes)
    }

    fn tmp_root(tag: &str) -> PathBuf {
        // Unique-enough per test: the test name plus the entries this
        // cache will hold. No Date/random in scripts, but std tests
        // can use the process + a counter via the tag.
        std::env::temp_dir().join(format!("infino-manifest-cache-test-{tag}"))
    }

    #[tokio::test]
    async fn put_then_get_roundtrips() {
        let root = tmp_root("roundtrip");
        let _ = fs::remove_dir_all(&root);
        let cache = ManifestDiskCache::new(root.clone(), 1 << 20).expect("cache");

        let bytes = b"compressed-part-bytes".to_vec();
        let h = hash_of(&bytes);
        cache.put(h, &bytes).await;

        let got = cache.get(&h).await.expect("hit");
        assert_eq!(got, bytes);
        let stats = cache.stats();
        assert_eq!(stats.n_entries, 1);
        assert_eq!(stats.n_hits, 1);
        assert_eq!(stats.current_bytes, bytes.len() as u64);
        let _ = fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn miss_on_unknown_hash() {
        let root = tmp_root("miss");
        let _ = fs::remove_dir_all(&root);
        let cache = ManifestDiskCache::new(root.clone(), 1 << 20).expect("cache");
        let h = hash_of(b"never-inserted");
        assert!(cache.get(&h).await.is_none());
        assert_eq!(cache.stats().n_misses, 1);
        let _ = fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn put_is_idempotent() {
        let root = tmp_root("idempotent");
        let _ = fs::remove_dir_all(&root);
        let cache = ManifestDiskCache::new(root.clone(), 1 << 20).expect("cache");
        let bytes = b"abc".to_vec();
        let h = hash_of(&bytes);
        cache.put(h, &bytes).await;
        cache.put(h, &bytes).await;
        assert_eq!(cache.stats().n_entries, 1);
        assert_eq!(cache.stats().current_bytes, bytes.len() as u64);
        let _ = fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn oversized_part_is_not_cached() {
        let root = tmp_root("oversized");
        let _ = fs::remove_dir_all(&root);
        let cache = ManifestDiskCache::new(root.clone(), 4).expect("cache");
        let bytes = b"way too big for a 4 byte budget".to_vec();
        let h = hash_of(&bytes);
        cache.put(h, &bytes).await;
        assert_eq!(cache.stats().n_entries, 0);
        assert!(cache.get(&h).await.is_none());
        let _ = fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn lru_eviction_frees_room() {
        let root = tmp_root("lru");
        let _ = fs::remove_dir_all(&root);
        // Budget fits two 10-byte parts but not three.
        let cache = ManifestDiskCache::new(root.clone(), 25).expect("cache");
        let a = vec![1u8; 10];
        let b = vec![2u8; 10];
        let c = vec![3u8; 10];
        let (ha, hb, hc) = (hash_of(&a), hash_of(&b), hash_of(&c));
        cache.put(ha, &a).await;
        cache.put(hb, &b).await;
        // Touch `a` so `b` becomes the LRU victim.
        assert!(cache.get(&ha).await.is_some());
        cache.put(hc, &c).await;

        assert!(cache.get(&ha).await.is_some(), "a kept (recently used)");
        assert!(cache.get(&hc).await.is_some(), "c kept (just inserted)");
        assert!(cache.get(&hb).await.is_none(), "b evicted (LRU)");
        assert!(cache.stats().n_evictions >= 1);
        let _ = fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn corrupt_file_is_treated_as_miss() {
        let root = tmp_root("corrupt");
        let _ = fs::remove_dir_all(&root);
        let cache = ManifestDiskCache::new(root.clone(), 1 << 20).expect("cache");
        let bytes = b"genuine-bytes".to_vec();
        let h = hash_of(&bytes);
        cache.put(h, &bytes).await;
        // Overwrite the on-disk file with garbage that won't hash to h.
        let path = cache.cache_path(&h);
        tokio::fs::write(&path, b"tampered")
            .await
            .expect("write tampered file");
        assert!(cache.get(&h).await.is_none(), "corruption ⇒ miss");
        // The bad file is gone and the entry dropped.
        assert!(!path.exists());
        assert_eq!(cache.stats().n_entries, 0);
        let _ = fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn restart_scan_rebuilds_index() {
        let root = tmp_root("restart");
        let _ = fs::remove_dir_all(&root);
        let bytes = b"survives-restart".to_vec();
        let h = hash_of(&bytes);
        {
            let cache = ManifestDiskCache::new(root.clone(), 1 << 20).expect("cache");
            cache.put(h, &bytes).await;
        }
        // New cache over the same root: index rebuilt from disk.
        let reopened = ManifestDiskCache::new(root.clone(), 1 << 20).expect("reopen");
        assert_eq!(reopened.stats().n_entries, 1);
        assert_eq!(reopened.stats().current_bytes, bytes.len() as u64);
        assert_eq!(reopened.get(&h).await.expect("hit after restart"), bytes);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn filename_roundtrips_through_parse() {
        let h = hash_of(b"some-part");
        let name = cache_filename(&h);
        assert!(name.starts_with(CACHE_FILE_PREFIX));
        assert!(name.ends_with(CACHE_FILE_SUFFIX));
        assert_eq!(parse_cache_filename(&name), Some(h));
        // Unrelated names are rejected.
        assert_eq!(parse_cache_filename("not-a-part.txt"), None);
        assert_eq!(parse_cache_filename("part-xyz.avro.zst"), None);
    }
}
