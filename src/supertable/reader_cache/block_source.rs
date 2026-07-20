// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`BlockCachedSource`] — block-granular NVMe retention for lazy
//! (range-GET-backed) superfile reads.
//!
//! The disk cache historically had exactly two states per superfile: a lazy
//! reader whose every `range()` was a fresh object-store GET (nothing
//! retained), or a fully-promoted mmap of the whole object. Between "first
//! touch" and "full promotion" the same byte ranges were re-fetched on every
//! query — and full promotion of everything cannot work once the table
//! outgrows local disk (a 1B-row index is TBs).
//!
//! This source is the missing middle state: reads through it land in a
//! sparse local file at fixed block granularity. A miss fetches one
//! block-aligned GET per contiguous missing run, writes it into the sparse
//! file, and marks the blocks filled; every later read of those bytes — from
//! any query on the shared cached reader — is a local `pread`, zero GETs.
//! Disk (and budget) cost is proportional to the *touched* working set, not
//! the object size, which is what lets the cache serve tables far larger
//! than local disk.
//!
//! Budget integration: each newly filled run reserves its bytes against the
//! owning [`DiskCacheStore`]'s budget (with LRU eviction pressure) *before*
//! fetching, and the entry's shared `size_bytes` counter grows as blocks
//! land — so eviction sees a lazy entry's true footprint. On budget
//! exhaustion, or once this source's cache entry has been replaced (eviction
//! / mmap promotion), reads degrade to plain uncached passthrough instead of
//! failing. The source releases its accounted bytes and unlinks its file on
//! `Drop` (i.e. when the last in-flight reader over it goes away).

use std::{
    fs,
    os::unix::fs::FileExt,
    path::PathBuf,
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use async_trait::async_trait;
use bytes::Bytes;
use roaring::RoaringBitmap;

use super::disk::DiskCacheStore;
use crate::{
    superfile::{LazyByteSource, LazyByteSourceError},
    supertable::manifest::SuperfileUri,
};

/// Cache block size. Misses fetch block-aligned runs, so this bounds both
/// the read amplification of a small scan (a request pays at most one
/// leading + one trailing partial block of overhead) and the bitmap size
/// (a 32 MiB cell superfile is 64 blocks; a 27 GiB one at 1B scale is
/// ~55K). Post-drain vector queries read ~0.25–2 MiB scan ranges, so
/// 512 KiB keeps first-touch overshoot well under 2× while still
/// coalescing a multi-MiB scan into a handful of blocks.
const CACHE_BLOCK_BYTES: u64 = 512 * 1024;

/// Lazily-initialized sparse backing file. Created on the first cached read
/// (the object size may only be known after the open-time `tail()` on
/// unknown-size sources). `None` means creation failed once — the source
/// then serves plain passthrough reads forever (cache disabled, not broken).
struct BlockFile {
    file: fs::File,
    size: u64,
}

/// Block-caching wrapper around a network-backed [`LazyByteSource`].
/// See the module docs for semantics.
pub(crate) struct BlockCachedSource {
    inner: Arc<dyn LazyByteSource>,
    store: Weak<DiskCacheStore>,
    uri: SuperfileUri,
    path: PathBuf,
    /// Distinguishes this source from a replacement entry for the same URI.
    entry_token: Arc<()>,
    /// Whether this source reserves and releases touched-block bytes itself.
    owns_accounting: bool,
    /// Virtual hole: `(offset, len)` of a subsection whose reads bypass the
    /// block cache and fetch exact ranges from the inner source. Set to the
    /// FTS subsection: posting reads are ~KiB-sized and scattered, so
    /// rounding each to a 512 KiB block over-fetches ~200× per read on a
    /// wide OR (the block size is tuned for vector's 0.25–2 MiB scans).
    /// Their warm locality comes from the background-fill mmap promotion,
    /// not from this cache. Reads only partially overlapping the hole keep
    /// block semantics.
    passthrough: Option<(u64, u64)>,
    state: OnceLock<Option<BlockFile>>,
    /// Filled-block set. Guarded by a sync mutex; never held across await.
    filled: Mutex<RoaringBitmap>,
    /// Bytes of filled blocks — shared with the owning `CachedEntry`'s
    /// `size_bytes`, so eviction candidates report a lazy entry's real
    /// footprint as it grows.
    filled_bytes: Arc<AtomicU64>,
}

impl BlockCachedSource {
    #[cfg(test)]
    pub(crate) fn new(
        inner: Arc<dyn LazyByteSource>,
        store: Weak<DiskCacheStore>,
        uri: SuperfileUri,
        path: PathBuf,
    ) -> Arc<Self> {
        Self::new_with_accounting(inner, store, uri, path, true, None)
    }

    /// Construct a sparse source whose owning entry has already reserved the
    /// complete object size. `passthrough` is the optional exact-read hole
    /// (see the field docs) — the FTS subsection on the cold-open path.
    pub(crate) fn new_pre_reserved(
        inner: Arc<dyn LazyByteSource>,
        store: Weak<DiskCacheStore>,
        uri: SuperfileUri,
        path: PathBuf,
        passthrough: Option<(u64, u64)>,
    ) -> Arc<Self> {
        Self::new_with_accounting(inner, store, uri, path, false, passthrough)
    }

    fn new_with_accounting(
        inner: Arc<dyn LazyByteSource>,
        store: Weak<DiskCacheStore>,
        uri: SuperfileUri,
        path: PathBuf,
        owns_accounting: bool,
        passthrough: Option<(u64, u64)>,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner,
            store,
            uri,
            path,
            entry_token: Arc::new(()),
            owns_accounting,
            passthrough,
            state: OnceLock::new(),
            filled: Mutex::new(RoaringBitmap::new()),
            filled_bytes: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Whether `[start, start + len)` lies fully inside the passthrough hole.
    fn in_passthrough(&self, start: u64, len: u64) -> bool {
        match self.passthrough {
            Some((off, hole_len)) => start >= off && start + len <= off + hole_len,
            None => false,
        }
    }

    /// Identity token installed on the cache entry that owns this source.
    pub(crate) fn entry_token(&self) -> Arc<()> {
        Arc::clone(&self.entry_token)
    }

    /// Shared filled-bytes counter, installed as the cache entry's
    /// `size_bytes` so accounting and eviction see live growth.
    #[cfg(test)]
    pub(crate) fn filled_bytes_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.filled_bytes)
    }

    /// The sparse file, created on first use once the object size is known.
    fn block_file(&self) -> Option<&BlockFile> {
        let size = self.inner.size();
        if size == 0 {
            // Size not discovered yet (pre-`tail()` on an unknown-size
            // source) — don't latch the OnceLock; try again next read.
            return None;
        }
        self.state
            .get_or_init(|| {
                let file = fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&self.path)
                    .ok()?;
                file.set_len(size).ok()?;
                Some(BlockFile { file, size })
            })
            .as_ref()
    }

    /// Byte length of block `b` (the trailing block may be partial).
    fn block_len(size: u64, b: u32) -> u64 {
        let start = u64::from(b) * CACHE_BLOCK_BYTES;
        (size - start).min(CACHE_BLOCK_BYTES)
    }

    /// Inclusive block index range covering `[start, start + len)`.
    fn block_span(start: u64, len: u64) -> (u32, u32) {
        let b0 = start / CACHE_BLOCK_BYTES;
        let b1 = (start + len - 1) / CACHE_BLOCK_BYTES;
        (b0 as u32, b1 as u32)
    }

    fn all_filled(&self, b0: u32, b1: u32) -> bool {
        let filled = self.filled.lock().expect("filled bitmap mutex poisoned");
        (b0..=b1).all(|b| filled.contains(b))
    }

    /// Contiguous runs of not-yet-filled blocks within `[b0, b1]`.
    fn missing_runs(&self, b0: u32, b1: u32) -> Vec<(u32, u32)> {
        let filled = self.filled.lock().expect("filled bitmap mutex poisoned");
        let mut runs = Vec::new();
        let mut run_start: Option<u32> = None;
        for b in b0..=b1 {
            if filled.contains(b) {
                if let Some(s) = run_start.take() {
                    runs.push((s, b - 1));
                }
            } else if run_start.is_none() {
                run_start = Some(b);
            }
        }
        if let Some(s) = run_start {
            runs.push((s, b1));
        }
        runs
    }

    /// Mark `[b0, b1]` filled; returns the byte count of blocks that were
    /// NEWLY marked (a concurrent filler may have raced us on some).
    fn mark_filled(&self, size: u64, b0: u32, b1: u32) -> u64 {
        let mut filled = self.filled.lock().expect("filled bitmap mutex poisoned");
        let mut newly = 0u64;
        for b in b0..=b1 {
            if filled.insert(b) {
                newly += Self::block_len(size, b);
            }
        }
        newly
    }

    /// Serve `[start, start+len)` from the sparse file. `None` on a read
    /// error (caller degrades to passthrough).
    fn read_local(&self, bf: &BlockFile, start: u64, len: u64) -> Option<Bytes> {
        let mut out = vec![0u8; len as usize];
        bf.file.read_exact_at(&mut out, start).ok()?;
        Some(Bytes::from(out))
    }

    /// Fill every missing block covering the request, reserving budget per
    /// run and settling duplicate-fill accounting. Returns `false` if the
    /// read should degrade to passthrough (budget exhausted, entry replaced,
    /// store gone, or local file I/O failed).
    async fn fill_missing(
        &self,
        bf: &BlockFile,
        b0: u32,
        b1: u32,
    ) -> Result<bool, LazyByteSourceError> {
        let Some(store) = self.store.upgrade() else {
            return Ok(false);
        };
        // Only the source installed in the live cache entry accounts bytes:
        // after eviction or mmap promotion replaced the entry, keep serving
        // already-filled blocks but stop growing the footprint.
        if !store.lazy_block_entry_is_current(&self.uri, &self.entry_token) {
            return Ok(false);
        }
        for (rb0, rb1) in self.missing_runs(b0, b1) {
            let run_start = u64::from(rb0) * CACHE_BLOCK_BYTES;
            let run_end = (u64::from(rb1) + 1) * CACHE_BLOCK_BYTES;
            let run_len = run_end.min(bf.size) - run_start;
            if self.owns_accounting && store.reserve_block_bytes(run_len).await.is_err() {
                // Budget pressure with no evictable victims: serve uncached.
                return Ok(false);
            }
            let bytes = match self.inner.range(run_start, run_len).await {
                Ok(b) => b,
                Err(e) => {
                    if self.owns_accounting {
                        store.release_block_bytes(run_len);
                    }
                    return Err(e);
                }
            };
            if bf.file.write_all_at(&bytes, run_start).is_err() {
                if self.owns_accounting {
                    store.release_block_bytes(run_len);
                }
                return Ok(false);
            }
            let newly = self.mark_filled(bf.size, rb0, rb1);
            self.filled_bytes.fetch_add(newly, Ordering::AcqRel);
            if self.owns_accounting && newly < run_len {
                // A concurrent filler beat us to some blocks; its accounting
                // stands, ours is released.
                store.release_block_bytes(run_len - newly);
            }
        }
        Ok(true)
    }
}

impl Drop for BlockCachedSource {
    fn drop(&mut self) {
        // Last reader over this source is gone (entry evicted/replaced and
        // no in-flight queries): release the accounted bytes and remove the
        // sparse file. The store may already be gone at process teardown.
        if self.owns_accounting
            && let Some(store) = self.store.upgrade()
        {
            let filled = self.filled_bytes.load(Ordering::Acquire);
            if filled > 0 {
                store.release_block_bytes(filled);
            }
        }
        if self.state.get().is_some_and(|s| s.is_some()) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[async_trait]
impl LazyByteSource for BlockCachedSource {
    fn size(&self) -> u64 {
        self.inner.size()
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        if len == 0 {
            return Ok(Bytes::new());
        }
        if self.in_passthrough(start, len) {
            return self.inner.range(start, len).await;
        }
        let Some(bf) = self.block_file() else {
            return self.inner.range(start, len).await;
        };
        if start.saturating_add(len) > bf.size {
            // Out-of-bounds: let the inner source surface its typed error.
            return self.inner.range(start, len).await;
        }
        let (b0, b1) = Self::block_span(start, len);
        if !self.all_filled(b0, b1) && !self.fill_missing(bf, b0, b1).await? {
            return self.inner.range(start, len).await;
        }
        match self.read_local(bf, start, len) {
            Some(bytes) => Ok(bytes),
            None => self.inner.range(start, len).await,
        }
    }

    fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
        if len == 0 {
            return Some(Bytes::new());
        }
        if self.in_passthrough(start, len) {
            return self.inner.try_get_range_sync(start, len);
        }
        let bf = self.block_file()?;
        if start.saturating_add(len) > bf.size {
            return None;
        }
        let (b0, b1) = Self::block_span(start, len);
        if !self.all_filled(b0, b1) {
            return self.inner.try_get_range_sync(start, len);
        }
        self.read_local(bf, start, len)
    }

    async fn tail(&self, len: u64) -> Result<(Bytes, u64), LazyByteSourceError> {
        // Pass through: `tail` both fetches and (on unknown-size sources)
        // discovers the object size, which the inner source caches. Tail
        // bytes are open-time metadata already retained by the open-blob /
        // prefetch overlay above this source, so caching them here as
        // (mostly partial) blocks buys nothing.
        self.inner.tail(len).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use tempfile::tempdir;

    use super::*;
    use crate::supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, LruPolicy};

    /// In-memory fake source that counts `range` calls.
    struct CountingSource {
        blob: Bytes,
        calls: AtomicUsize,
    }

    impl CountingSource {
        fn new(n: usize) -> Self {
            let blob: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
            Self {
                blob: Bytes::from(blob),
                calls: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Acquire)
        }
    }

    #[async_trait]
    impl LazyByteSource for CountingSource {
        fn size(&self) -> u64 {
            self.blob.len() as u64
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            let (s, e) = (start as usize, (start + len) as usize);
            if e > self.blob.len() {
                return Err(LazyByteSourceError::OutOfBounds {
                    start,
                    len,
                    size: self.blob.len() as u64,
                });
            }
            Ok(self.blob.slice(s..e))
        }
    }

    /// A store whose budget admits everything; `noop_storage` is never hit
    /// because the block source's inner fake serves all reads.
    fn test_store(dir: &std::path::Path, budget: u64) -> Arc<DiskCacheStore> {
        use std::{ops::Range, time::SystemTime};

        use object_store::MultipartUpload;

        use crate::storage::{ObjectMeta, StorageError, StorageProvider};

        #[derive(Debug)]
        struct NoopStorage;

        fn unimplemented_err(uri: &str) -> StorageError {
            StorageError::Permanent {
                uri: uri.into(),
                source: "noop storage".into(),
            }
        }

        #[async_trait]
        impl StorageProvider for NoopStorage {
            async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
                let _ = uri;
                Ok(ObjectMeta {
                    size: 0,
                    etag: None,
                    last_modified: SystemTime::UNIX_EPOCH,
                })
            }
            async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn get_range(&self, uri: &str, _r: Range<u64>) -> Result<Bytes, StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn put_atomic(
                &self,
                uri: &str,
                _b: Bytes,
            ) -> Result<Option<String>, StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn put_if_match(
                &self,
                uri: &str,
                _b: Bytes,
                _etag: Option<&str>,
            ) -> Result<Option<String>, StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn put_multipart(
                &self,
                uri: &str,
            ) -> Result<Box<dyn MultipartUpload>, StorageError> {
                Err(unimplemented_err(uri))
            }
            async fn delete(&self, _uri: &str) -> Result<(), StorageError> {
                Ok(())
            }
        }

        let cfg = DiskCacheConfig {
            cache_root: dir.to_path_buf(),
            disk_budget_bytes: budget,
            cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
            eviction: Box::new(LruPolicy::new()),
            ..DiskCacheConfig::default()
        };
        DiskCacheStore::new_unpinned(Arc::new(NoopStorage), cfg).expect("test store")
    }

    /// Second read of the same bytes never touches the inner source, the
    /// returned bytes are identical, and accounting matches the touched
    /// block footprint (not the object size).
    #[tokio::test]
    async fn repeat_reads_are_served_locally_and_accounted_by_blocks() {
        const OBJ: usize = 3 * CACHE_BLOCK_BYTES as usize + 1000;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let inner = Arc::new(CountingSource::new(OBJ));
        let src = BlockCachedSource::new(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("t.blocks"),
        );
        // Install the source as current for its (synthetic) entry.
        store.install_block_entry_for_test(uri, src.filled_bytes_handle(), src.entry_token());

        // Read spanning blocks 0..=2 (one contiguous missing run → 1 GET).
        let start = 100u64;
        let len = 2 * CACHE_BLOCK_BYTES + 500;
        let first = src.range(start, len).await.expect("first read");
        assert_eq!(first, inner.blob.slice(100..(start + len) as usize));
        assert_eq!(inner.calls(), 1, "one block-run GET for the miss");

        // Identical read → zero inner calls.
        let second = src.range(start, len).await.expect("second read");
        assert_eq!(second, first);
        assert_eq!(inner.calls(), 1, "repeat read must not touch the source");

        // Sub-range and sync reads also come from local blocks.
        let sub = src.range(start + 10, 100).await.expect("sub read");
        assert_eq!(sub, inner.blob.slice(110..210));
        let sync = src
            .try_get_range_sync(start + 10, 100)
            .expect("sync read of filled blocks");
        assert_eq!(sync, sub);
        assert_eq!(inner.calls(), 1);

        // Accounting = 3 whole blocks (0..=2), not the object size.
        let expected = 3 * CACHE_BLOCK_BYTES;
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), expected);
        assert_eq!(store.stats().current_bytes, expected);

        // Touch the trailing partial block: its length is size - 3*B.
        let tail_start = 3 * CACHE_BLOCK_BYTES + 10;
        let t = src.range(tail_start, 50).await.expect("tail block read");
        assert_eq!(
            t,
            inner
                .blob
                .slice(tail_start as usize..tail_start as usize + 50)
        );
        assert_eq!(inner.calls(), 2);
        assert_eq!(
            src.filled_bytes_handle().load(Ordering::Acquire),
            expected + 1000,
            "trailing partial block accounts its real length"
        );

        // Drop the source: accounting released, blocks file unlinked.
        let path = dir.path().join("t.blocks");
        assert!(path.exists());
        store.remove_block_entry_for_test(&uri);
        drop(src);
        assert_eq!(store.stats().current_bytes, 0);
        assert!(!path.exists());
    }

    /// Two disjoint missing runs in one request → one GET per run.
    #[tokio::test]
    async fn disjoint_missing_runs_fetch_separately() {
        const OBJ: usize = 6 * CACHE_BLOCK_BYTES as usize;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let inner = Arc::new(CountingSource::new(OBJ));
        let src = BlockCachedSource::new(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("runs.blocks"),
        );
        store.install_block_entry_for_test(uri, src.filled_bytes_handle(), src.entry_token());

        // Fill block 2 first.
        let b = CACHE_BLOCK_BYTES;
        let _ = src.range(2 * b, 10).await.expect("fill middle block");
        assert_eq!(inner.calls(), 1);

        // Read blocks 1..=3: blocks 1 and 3 are missing → two run GETs.
        let got = src.range(b, 3 * b).await.expect("spanning read");
        assert_eq!(got, inner.blob.slice(b as usize..(4 * b) as usize));
        assert_eq!(inner.calls(), 3, "two missing runs around the filled block");

        store.remove_block_entry_for_test(&uri);
    }

    /// When the entry is no longer current (evicted/promoted), reads still
    /// succeed as passthrough and accounting stops growing.
    #[tokio::test]
    async fn stale_entry_degrades_to_passthrough() {
        const OBJ: usize = 2 * CACHE_BLOCK_BYTES as usize;
        let dir = tempdir().expect("tempdir");
        let store = test_store(dir.path(), u64::MAX);
        let uri = SuperfileUri::new_v4();
        let inner = Arc::new(CountingSource::new(OBJ));
        let src = BlockCachedSource::new(
            Arc::clone(&inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.path().join("stale.blocks"),
        );
        // Never installed as current → every miss is passthrough.
        let a = src.range(0, 64).await.expect("passthrough read");
        let bb = src.range(0, 64).await.expect("passthrough read again");
        assert_eq!(a, bb);
        assert_eq!(inner.calls(), 2, "uncached passthrough on both reads");
        assert_eq!(src.filled_bytes_handle().load(Ordering::Acquire), 0);
        assert_eq!(store.stats().current_bytes, 0);
    }
}
