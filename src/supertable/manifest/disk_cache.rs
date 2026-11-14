// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`MetadataDiskCache`] — a table's content-addressed on-disk cache for
//! the metadata objects it reads back from storage: manifest parts, the
//! summaries sections beside them (see
//! [`crate::supertable::manifest::summaries`]), term index objects, and a
//! hidden vector index's state blob and centroid section. Every object is
//! named by the blake3 hash of its bytes, so one cache serves all of them
//! under the same hash-named files, and a new kind of object needs no cache
//! of its own.
//!
//! The content hash is the key, so a cached file can never be stale: if
//! the logical content changes, the hash changes, and the new content
//! lands under a new filename. No invalidation logic is needed; old files
//! simply become eviction candidates.
//!
//! Each table has its own cache. A hidden vector index is a table of its
//! own, so it gets one in [`HIDDEN_INDEX_CACHE_SUBDIR`] beside its user
//! table's, with its own budget: its objects are 100 MB or more, and
//! sharing a budget would let them evict the user table's manifest parts.
//!
//! ## What this buys
//!
//! Readers consult this cache before issuing a `StorageProvider::get`. On
//! a hit the bytes come off local disk instead of object storage,
//! eliminating the network round-trip (the dominant cost on a cold load
//! over S3). The cache also survives process restarts: the on-disk files
//! persist and the in-memory accounting index is rebuilt by a directory
//! scan at construction.
//!
//! ## What this does NOT cache
//!
//! Superfile *byte content* is cached one layer over, by
//! [`DiskCacheStore`](crate::supertable::reader_cache::DiskCacheStore).
//! The two caches are independent and use separate byte budgets.
//!
//! ## Objects read in place
//!
//! A summaries section is not read whole: a handle keeps the section's
//! file here and reads its groups back from it piecewise, for as long as
//! the handle lives. Such an object is pinned ([`MetadataDiskCache::pin`])
//! while a handle reads from it, so eviction passes over it whatever else
//! fills the budget, and each piece is checked against a hash taken when
//! the whole object was verified ([`MetadataDiskCache::read_range_verified`]).

use std::{
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
    path::PathBuf,
    process,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use bytes::Bytes;
use dashmap::{DashMap, Entry};
use tokio::task::spawn_blocking;

use crate::{supertable::manifest::part::ContentHash, utils::trace::record};

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
/// Directory, beside a user table's cache, of its hidden vector index's
/// cache.
pub(crate) const HIDDEN_INDEX_CACHE_SUBDIR: &str = "hidden-index-metadata";
/// How much of a cached object [`MetadataDiskCache::verify`] reads at a time.
const VERIFY_CHUNK_BYTES: usize = 1 << 20;
/// Headroom over an object a cache holds whole ([`MetadataDiskCache::keep_whole`]),
/// in divisor form (`bytes + bytes / this`), for the smaller objects that
/// share the cache with it (term-index slices, routing siblings).
const KEPT_WHOLE_HEADROOM_DIVISOR: u64 = 10;

/// An object a table's metadata cache keeps whole, whatever its configured
/// budget: a load reads all of it, so a budget smaller than it would evict
/// each piece before the next load read it back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeptWhole {
    /// Every part, summaries section and rollup of the table's manifest.
    Manifest,
    /// A hidden vector index's state blob.
    VectorStateBlob,
    /// A hidden vector index's centroid section.
    CentroidSection,
}

/// What a `get` found on disk for an indexed hash.
enum CachedRead {
    /// The file is gone.
    Missing,
    /// The bytes do not hash to the key; the file has been removed.
    Corrupt,
    /// The bytes, verified against the key.
    Verified(Vec<u8>),
}

/// Per-entry bookkeeping for the in-memory accounting index. The bytes
/// themselves live on disk; this only tracks what eviction needs.
struct CacheEntry {
    size_bytes: u64,
    /// Microseconds-since-construction of the last access. Monotonic
    /// per [`MetadataDiskCache`] instance; drives LRU eviction.
    last_access_us: AtomicU64,
}

/// Snapshot of the metadata cache's load. Surfaced via
/// [`MetadataDiskCache::stats`] for tests and observability.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetadataCacheStats {
    pub n_entries: u64,
    pub current_bytes: u64,
    pub budget_bytes: u64,
    pub n_hits: u64,
    pub n_misses: u64,
    pub n_evictions: u64,
}

/// Content-addressed disk cache for a table's metadata objects.
///
/// Construction is sync (a directory scan); `get` / `put` are async
/// because their file I/O runs off the tokio workers, each as one
/// blocking task. A `get` verifies the bytes (blake3) in that same task.
pub struct MetadataDiskCache {
    cache_root: PathBuf,
    /// The budget the cache was opened with: working room for the objects
    /// fetched a piece at a time (the term index's slices), on top of what
    /// it holds whole.
    configured_bytes: u64,
    /// Room for the manifest kept whole ([`KeptWhole::Manifest`]), set to
    /// its latest size, so the budget follows it as it grows and shrinks.
    manifest_whole: AtomicU64,
    /// Room for [`KeptWhole::VectorStateBlob`], likewise.
    state_blob_whole: AtomicU64,
    /// Room for [`KeptWhole::CentroidSection`], likewise.
    centroid_section_whole: AtomicU64,
    started_at: Instant,
    /// Accounting index: content hash → size + last-access. The bytes
    /// are on disk under [`Self::cache_path`]; this map exists only so
    /// eviction can pick victims without stat-ing the directory.
    entries: DashMap<ContentHash, CacheEntry>,
    /// How many live [`CachePin`]s hold each object: an object a handle
    /// reads in place, which eviction passes over. Kept apart from
    /// `entries`, so a pin outlives its entry being removed and put again.
    pins: DashMap<ContentHash, usize>,
    current_bytes: AtomicU64,
    n_hits: AtomicU64,
    n_misses: AtomicU64,
    n_evictions: AtomicU64,
}

/// Numbers each manifest-part write's temp file in this process. Process
/// wide, as each table open builds its own cache over the same directory, so
/// two writes of the same part (a cancelled query's still landing while its
/// rerun writes again, or two opens of one table) never share one; the temp
/// name also carries the pid, for writes from another process.
static NEXT_TMP: AtomicU64 = AtomicU64::new(0);

impl MetadataDiskCache {
    /// Construct a cache rooted at `cache_root` (created if absent)
    /// with a byte budget of `budget_bytes`. Scans `cache_root` for
    /// pre-existing `part-<hex>.avro.zst` files and seeds the
    /// accounting index from them, so a freshly-opened process reuses
    /// the parts a prior run cached (restart survival).
    pub fn new(cache_root: PathBuf, budget_bytes: u64) -> io::Result<Arc<Self>> {
        fs::create_dir_all(&cache_root)?;
        let cache = Self {
            cache_root,
            configured_bytes: budget_bytes,
            manifest_whole: AtomicU64::new(0),
            state_blob_whole: AtomicU64::new(0),
            centroid_section_whole: AtomicU64::new(0),
            started_at: Instant::now(),
            entries: DashMap::new(),
            pins: DashMap::new(),
            current_bytes: AtomicU64::new(0),
            n_hits: AtomicU64::new(0),
            n_misses: AtomicU64::new(0),
            n_evictions: AtomicU64::new(0),
        };
        cache.scan_existing();
        Ok(Arc::new(cache))
    }

    /// The cache for this table's hidden vector index: its own directory,
    /// [`HIDDEN_INDEX_CACHE_SUBDIR`], beside this one, opened with the same
    /// configured budget. What this table holds is not carried over; the
    /// hidden index holds its own objects.
    pub(crate) fn for_hidden_index(&self) -> io::Result<Arc<Self>> {
        let parent = self.cache_root.parent().unwrap_or(&self.cache_root);
        Self::new(
            parent.join(HIDDEN_INDEX_CACHE_SUBDIR),
            self.configured_bytes,
        )
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
    // A hit is not free: it reads the file and re-hashes it end to end, so
    // on a large part this is real I/O plus a blake3 pass. `outcome`
    // separates the cheap index miss from the expensive verified hit.
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(
            name = "manifest.part_cache_get",
            skip_all,
            fields(outcome = tracing::field::Empty, bytes = tracing::field::Empty)
        )
    )]
    pub async fn get(&self, hash: &ContentHash) -> Option<Vec<u8>> {
        // Fast negative: not in the accounting index ⇒ not cached.
        if !self.entries.contains_key(hash) {
            record("outcome", "miss");
            self.n_misses.fetch_add(1, Ordering::AcqRel);
            return None;
        }
        let path = self.cache_path(hash);
        let expected = *hash;
        // The read and the hash run as one blocking task: a cached object can
        // be hundreds of MB, too much to hash on a runtime worker.
        let read = spawn_blocking(move || match fs::read(&path) {
            Err(_) => CachedRead::Missing,
            Ok(bytes) if ContentHash::of(&bytes) != expected => {
                let _ = fs::remove_file(&path);
                CachedRead::Corrupt
            }
            Ok(bytes) => CachedRead::Verified(bytes),
        })
        .await
        .unwrap_or(CachedRead::Missing);
        let bytes = match read {
            CachedRead::Verified(bytes) => bytes,
            CachedRead::Missing => {
                // File vanished out from under the index (manual
                // delete, external eviction). Drop the stale entry.
                record("outcome", "file_missing");
                self.drop_entry(hash);
                self.n_misses.fetch_add(1, Ordering::AcqRel);
                return None;
            }
            CachedRead::Corrupt => {
                // Content-addressing guarantee: the bytes must hash to the
                // key. A mismatch means on-disk corruption, already removed.
                record("outcome", "corrupt");
                self.drop_entry(hash);
                self.n_misses.fetch_add(1, Ordering::AcqRel);
                return None;
            }
        };
        if let Some(entry) = self.entries.get(hash) {
            entry.last_access_us.store(self.now_us(), Ordering::Release);
        }
        record("outcome", "hit");
        record("bytes", bytes.len() as u64);
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
    ///
    /// The write, the rename and the index insert run as one blocking task,
    /// which settles the budget on every outcome. A caller dropped while it
    /// runs (a cancelled query) therefore leaves the cache as if it had
    /// waited: the part lands indexed and charged, never as a file the
    /// budget and eviction cannot see.
    ///
    /// Returns whether the bytes are in the cache afterwards, so a caller
    /// that can read them back from here ([`Self::read_range`]) knows it
    /// may let go of its own copy.
    pub async fn put(self: &Arc<Self>, hash: ContentHash, bytes: Bytes) -> bool {
        if self.entries.contains_key(&hash) {
            return true;
        }
        let size = bytes.len() as u64;
        // A single part larger than the whole budget can never fit;
        // don't bother evicting everything for a doomed insert.
        if size > self.budget_bytes() {
            return false;
        }
        let cache = Arc::clone(self);
        // Reserving can evict many files at once (the budget drops when the
        // table's manifest shrinks), so it runs in the blocking task too.
        // A join error is a panic or a runtime shutting down. The cache is
        // best-effort either way, so the caller carries on.
        spawn_blocking(move || cache.reserve(size) && cache.write_reserved(hash, &bytes))
            .await
            .unwrap_or(false)
    }

    /// Write `bytes` under `hash` and index it, with `bytes.len()` already
    /// reserved. The reservation is given back when the write fails, or when
    /// a concurrent put of the same hash indexed it first. Returns whether
    /// the bytes are indexed afterwards.
    fn write_reserved(&self, hash: ContentHash, bytes: &[u8]) -> bool {
        let size = bytes.len() as u64;
        let final_path = self.cache_path(&hash);
        let tmp_path = self.tmp_path(&hash);
        if write_synced(&tmp_path, bytes)
            .and_then(|()| fs::rename(&tmp_path, &final_path))
            .is_err()
        {
            // Roll back the reservation; leave no partial tmp behind.
            self.current_bytes.fetch_sub(size, Ordering::Release);
            let _ = fs::remove_file(&tmp_path);
            return false;
        }

        // Commit into the accounting index. If a concurrent put raced us to
        // the same hash, release our reservation: the file on disk is
        // byte-identical either way (content addressing).
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
        true
    }

    /// Pin the object cached under `hash` for as long as the returned pin
    /// lives: eviction passes over it, whatever else fills the budget, so a
    /// handle that reads the object in place ([`Self::read_range_verified`])
    /// does not find it gone midway. `None` when the object is not cached.
    ///
    /// A pinned object still counts against the budget, so pins that fill it
    /// make later puts fail rather than evict them; a caller whose put fails
    /// keeps its own copy. [`Self::remove`] and a corrupt read still take a
    /// pinned object out: the first is an explicit erase, and a corrupt file
    /// is of no use to its readers.
    pub(crate) fn pin(self: &Arc<Self>, hash: ContentHash) -> Option<CachePin> {
        // The pin is counted before the entry is looked up, and eviction
        // checks it under the entry's lock (see `evict_at_least`), so an
        // object is either pinned before eviction looks or gone when the
        // lookup runs, never evicted pinned.
        *self.pins.entry(hash).or_insert(0) += 1;
        let pin = CachePin {
            cache: Arc::clone(self),
            hash,
        };
        self.entries.contains_key(&hash).then_some(pin)
    }

    /// Release one pin of `hash`.
    fn unpin(&self, hash: &ContentHash) {
        if let Entry::Occupied(mut pins) = self.pins.entry(*hash) {
            *pins.get_mut() -= 1;
            if *pins.get() == 0 {
                pins.remove();
            }
        }
    }

    /// Whether a pin holds `hash`.
    fn is_pinned(&self, hash: &ContentHash) -> bool {
        self.pins.contains_key(hash)
    }

    /// The size of the object cached under `hash`, when it is indexed.
    pub(crate) fn size_of(&self, hash: &ContentHash) -> Option<u64> {
        self.entries.get(hash).map(|entry| entry.size_bytes)
    }

    /// Read `len` bytes at `offset` of the object cached under `hash`,
    /// without verifying them. What a reader of an object's head uses
    /// before it [verifies](Self::verify) the object and learns the
    /// hashes of the pieces it will read with
    /// [`Self::read_range_verified`].
    ///
    /// Blocking: it reads the file on the calling thread.
    ///
    /// An object that is not indexed, or whose file is gone, is an error of
    /// kind `NotFound`, and a gone file leaves the index as a `get` of it
    /// would. A range past the object's end is `UnexpectedEof`.
    pub(crate) fn read_range(
        &self,
        hash: &ContentHash,
        offset: u64,
        len: usize,
    ) -> io::Result<Vec<u8>> {
        let out = self.read_at(hash, offset, len)?;
        self.touch(hash);
        self.n_hits.fetch_add(1, Ordering::AcqRel);
        Ok(out)
    }

    /// [`Self::read_range`], checked against `expected`, the hash of the
    /// range taken when the whole object was verified ([`Self::verify`]).
    /// For an object a reader keeps open for a handle's life and reads
    /// piecewise, the summaries sections beside the manifest parts, so the
    /// handle holds the pieces it has read, not the object, and still never
    /// takes a piece the object did not hold.
    ///
    /// Blocking: it reads and hashes the range on the calling thread. A
    /// caller on the runtime wraps it in a blocking task; the section
    /// readers call it where they decode, which is already CPU on that
    /// thread.
    ///
    /// A range that does not hash to `expected` is an error of kind
    /// `InvalidData`, and the file is removed, as a `get` removes a corrupt
    /// one: whatever reads the object next fetches it again. Otherwise as
    /// [`Self::read_range`].
    pub(crate) fn read_range_verified(
        &self,
        hash: &ContentHash,
        offset: u64,
        len: usize,
        expected: &ContentHash,
    ) -> io::Result<Vec<u8>> {
        let out = self.read_at(hash, offset, len)?;
        if ContentHash::of(&out) != *expected {
            self.remove(hash);
            self.n_misses.fetch_add(1, Ordering::AcqRel);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "a range of a cached object does not hash to what it held when verified",
            ));
        }
        self.touch(hash);
        self.n_hits.fetch_add(1, Ordering::AcqRel);
        Ok(out)
    }

    /// Read the object cached under `hash` end to end, check that it hashes
    /// to `hash`, and return the hash of each of `ranges` (`(offset, len)`)
    /// within it, for [`Self::read_range_verified`] to check the ranges
    /// against later. The object is streamed, never held whole.
    ///
    /// Blocking: it reads and hashes the file on the calling thread.
    ///
    /// An object that does not hash to `hash`, such as a file a crash left
    /// torn, is an error of kind `InvalidData`, and the file is removed, as
    /// a `get` removes one. An object not indexed, or whose file is gone,
    /// is `NotFound`. A range past the object's end is `UnexpectedEof`.
    pub(crate) fn verify(
        &self,
        hash: &ContentHash,
        ranges: &[(usize, usize)],
    ) -> io::Result<Vec<ContentHash>> {
        let mut file = self.open_indexed(hash)?;
        let mut whole = blake3::Hasher::new();
        let mut pieces: Vec<blake3::Hasher> =
            ranges.iter().map(|_| blake3::Hasher::new()).collect();
        let mut buffer = vec![0u8; VERIFY_CHUNK_BYTES];
        let mut at = 0usize;
        loop {
            let n = match file.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            };
            let chunk = &buffer[..n];
            whole.update(chunk);
            for ((offset, len), piece) in ranges.iter().zip(&mut pieces) {
                let start = (*offset).max(at);
                let end = offset.saturating_add(*len).min(at + n);
                if start < end {
                    piece.update(&chunk[start - at..end - at]);
                }
            }
            at += n;
        }
        if ContentHash(*whole.finalize().as_bytes()) != *hash {
            self.remove(hash);
            self.n_misses.fetch_add(1, Ordering::AcqRel);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "a cached object does not hash to its key",
            ));
        }
        if ranges
            .iter()
            .any(|(offset, len)| offset.saturating_add(*len) > at)
        {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "a range reaches past the cached object",
            ));
        }
        self.touch(hash);
        Ok(pieces
            .into_iter()
            .map(|piece| ContentHash(*piece.finalize().as_bytes()))
            .collect())
    }

    /// Read `len` bytes at `offset` of the object cached under `hash`.
    fn read_at(&self, hash: &ContentHash, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let mut file = self.open_indexed(hash)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut out = vec![0u8; len];
        file.read_exact(&mut out)?;
        Ok(out)
    }

    /// Open the file of the object cached under `hash`. `NotFound` when the
    /// object is not indexed, or its file is gone, which drops the entry as
    /// a `get` of it would.
    fn open_indexed(&self, hash: &ContentHash) -> io::Result<File> {
        if !self.entries.contains_key(hash) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "object is not in the metadata cache",
            ));
        }
        match File::open(self.cache_path(hash)) {
            Ok(file) => Ok(file),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // File vanished out from under the index (manual delete,
                // external eviction). Drop the stale entry.
                self.drop_entry(hash);
                self.n_misses.fetch_add(1, Ordering::AcqRel);
                Err(e)
            }
            Err(e) => Err(e),
        }
    }

    /// Mark `hash` as just used, for LRU eviction.
    fn touch(&self, hash: &ContentHash) {
        if let Some(entry) = self.entries.get(hash) {
            entry.last_access_us.store(self.now_us(), Ordering::Release);
        }
    }

    /// Reserve `size` bytes of budget, evicting LRU victims as needed.
    /// Returns `false` if the budget can't be made to fit (every entry
    /// got evicted and there still isn't room — only possible under a
    /// race, since oversized parts are rejected before calling this).
    fn reserve(&self, size: u64) -> bool {
        loop {
            let cur = self.current_bytes.load(Ordering::Acquire);
            let budget = self.budget_bytes();
            if cur + size <= budget {
                if self
                    .current_bytes
                    .compare_exchange_weak(cur, cur + size, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return true;
                }
                continue;
            }
            let needed = (cur + size).saturating_sub(budget);
            if !self.evict_at_least(needed) {
                return false;
            }
        }
    }

    /// Evict least-recently-accessed entries until at least
    /// `bytes_needed` is freed. Returns `false` if there were no
    /// entries left to evict before reaching the target.
    // Runs inline under `put`, so a full cache makes the caller pay for the
    // unlink sweep. `freed` vs `bytes_needed` shows when that is happening.
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(
            name = "manifest.part_cache_evict",
            skip_all,
            fields(bytes_needed = bytes_needed, freed = tracing::field::Empty)
        )
    )]
    fn evict_at_least(&self, bytes_needed: u64) -> bool {
        let mut candidates: Vec<(ContentHash, u64, u64)> = self
            .entries
            .iter()
            .filter(|e| !self.is_pinned(e.key()))
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
            // same victim can't double-count. The pin is checked under the
            // entry's lock, which `pin` takes after counting its pin, so a
            // pin taken since the candidates were listed is seen.
            if self
                .entries
                .remove_if(&hash, |hash, _| !self.is_pinned(hash))
                .is_some()
            {
                let _ = fs::remove_file(self.cache_path(&hash));
                self.current_bytes.fetch_sub(size, Ordering::Release);
                self.n_evictions.fetch_add(1, Ordering::AcqRel);
                freed = freed.saturating_add(size);
            }
        }
        record("freed", freed);
        freed >= bytes_needed
    }

    /// Remove the entry for `hash` and its file. Blocking: it unlinks the file.
    pub(crate) fn remove(&self, hash: &ContentHash) {
        if let Some((_, entry)) = self.entries.remove(hash) {
            let _ = fs::remove_file(self.cache_path(hash));
            self.current_bytes
                .fetch_sub(entry.size_bytes, Ordering::Release);
        }
    }

    /// Whether `hash` is in the cache.
    #[cfg(test)]
    pub(crate) fn contains(&self, hash: &ContentHash) -> bool {
        self.entries.contains_key(hash)
    }

    /// Remove an entry from the accounting index and release its
    /// reserved bytes. The on-disk file is left to the caller.
    fn drop_entry(&self, hash: &ContentHash) {
        if let Some((_, entry)) = self.entries.remove(hash) {
            self.current_bytes
                .fetch_sub(entry.size_bytes, Ordering::Release);
        }
    }

    /// The current byte budget: what the held objects need ([`Self::keep_whole`])
    /// plus the configured budget, which stays free for objects fetched a
    /// piece at a time however large the held ones are.
    pub fn budget_bytes(&self) -> u64 {
        [
            &self.manifest_whole,
            &self.state_blob_whole,
            &self.centroid_section_whole,
        ]
        .iter()
        .map(|room| room.load(Ordering::Acquire))
        .fold(self.configured_bytes, u64::saturating_add)
    }

    /// Keep `what`, now `bytes` long, whole: the budget covers it plus
    /// headroom. Replaces what was held for `what` before, so when the
    /// object shrinks the budget shrinks with it, and the next write evicts
    /// what is no longer read, least recently used first (the parts and
    /// sections of older versions, which no load touches any more).
    pub(crate) fn keep_whole(&self, what: KeptWhole, bytes: u64) {
        let with_headroom = bytes.saturating_add(bytes / KEPT_WHOLE_HEADROOM_DIVISOR);
        let room = match what {
            KeptWhole::Manifest => &self.manifest_whole,
            KeptWhole::VectorStateBlob => &self.state_blob_whole,
            KeptWhole::CentroidSection => &self.centroid_section_whole,
        };
        room.store(with_headroom, Ordering::Release);
    }

    /// Snapshot of the cache's load. Cheap: reads atomics + a
    /// `DashMap::len`.
    pub fn stats(&self) -> MetadataCacheStats {
        MetadataCacheStats {
            n_entries: self.entries.len() as u64,
            current_bytes: self.current_bytes.load(Ordering::Acquire),
            budget_bytes: self.budget_bytes(),
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

    /// A temp file for one write of `hash`, distinct from every other write's.
    fn tmp_path(&self, hash: &ContentHash) -> PathBuf {
        let n = NEXT_TMP.fetch_add(1, Ordering::Relaxed);
        self.cache_root.join(format!(
            "{}.{}.{n:x}{CACHE_TMP_SUFFIX}",
            cache_filename(hash),
            process::id()
        ))
    }
}

impl std::fmt::Debug for MetadataDiskCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetadataDiskCache")
            .field("cache_root", &self.cache_root)
            .field("budget_bytes", &self.budget_bytes())
            .field("current_bytes", &self.current_bytes.load(Ordering::Acquire))
            .field("n_entries", &self.entries.len())
            .finish()
    }
}

/// A pin of one cached object ([`MetadataDiskCache::pin`]): while it lives,
/// eviction passes over the object.
pub(crate) struct CachePin {
    cache: Arc<MetadataDiskCache>,
    hash: ContentHash,
}

impl CachePin {
    /// The cache holding the object.
    pub(crate) fn cache(&self) -> &MetadataDiskCache {
        &self.cache
    }

    /// The object's content hash.
    pub(crate) fn hash(&self) -> &ContentHash {
        &self.hash
    }
}

impl Drop for CachePin {
    fn drop(&mut self) {
        self.cache.unpin(&self.hash);
    }
}

impl std::fmt::Debug for CachePin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachePin")
            .field("hash", &self.hash.to_hex())
            .finish()
    }
}

/// Write `bytes` to a new file at `path` and flush them to disk before
/// returning, so a rename that publishes the file never publishes it
/// ahead of its data: a crash leaves the old name or the whole file,
/// never a torn one under the final name.
fn write_synced(path: &std::path::Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
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
    use std::time::Duration;

    use futures::FutureExt;
    use tokio::time::{sleep, timeout};

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
        let cache = MetadataDiskCache::new(root.clone(), 1 << 20).expect("cache");

        let bytes = b"compressed-part-bytes".to_vec();
        let h = hash_of(&bytes);
        cache.put(h, Bytes::copy_from_slice(&bytes)).await;

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
        let cache = MetadataDiskCache::new(root.clone(), 1 << 20).expect("cache");
        let h = hash_of(b"never-inserted");
        assert!(cache.get(&h).await.is_none());
        assert_eq!(cache.stats().n_misses, 1);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_budget_follows_what_the_cache_keeps_whole() {
        let root = tmp_root("kept-whole-budget");
        let _ = fs::remove_dir_all(&root);
        let cache = MetadataDiskCache::new(root.clone(), 100).expect("cache");
        assert_eq!(cache.budget_bytes(), 100);
        // What is held comes on top of the configured budget, each hold with
        // a tenth of headroom.
        cache.keep_whole(KeptWhole::Manifest, 1_000);
        assert_eq!(cache.budget_bytes(), 1_200);
        cache.keep_whole(KeptWhole::CentroidSection, 2_000);
        assert_eq!(cache.budget_bytes(), 3_400);
        // A smaller manifest lowers the budget again.
        cache.keep_whole(KeptWhole::Manifest, 0);
        assert_eq!(cache.budget_bytes(), 2_300);
        cache.keep_whole(KeptWhole::CentroidSection, 0);
        assert_eq!(cache.budget_bytes(), 100);
        let _ = fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn a_smaller_manifest_evicts_what_no_load_reads_any_more() {
        let root = tmp_root("kept-whole-shrinks");
        let _ = fs::remove_dir_all(&root);
        let cache = MetadataDiskCache::new(root.clone(), 0).expect("cache");
        let part = |tag: u8| Bytes::from(vec![tag; 100]);
        // An old manifest of four 100-byte parts.
        cache.keep_whole(KeptWhole::Manifest, 400);
        for tag in [1, 2, 3, 4] {
            assert!(cache.put(hash_of(&part(tag)), part(tag)).await);
        }
        // Its successor is smaller: parts 3 and 4 survive and part 5 is new.
        cache.keep_whole(KeptWhole::Manifest, 300);
        assert!(cache.get(&hash_of(&part(3))).await.is_some());
        assert!(cache.get(&hash_of(&part(4))).await.is_some());
        assert!(cache.put(hash_of(&part(5)), part(5)).await);
        // Parts 1 and 2, which no load reads any more, made room.
        for tag in [1, 2] {
            assert!(!cache.contains(&hash_of(&part(tag))), "part {tag} evicted");
        }
        for tag in [3, 4, 5] {
            assert!(cache.contains(&hash_of(&part(tag))), "part {tag} kept");
        }
        assert!(cache.stats().current_bytes <= cache.budget_bytes());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_hidden_index_cache_starts_from_the_configured_budget() {
        let parent = tmp_root("hidden-budget");
        let _ = fs::remove_dir_all(&parent);
        let user = MetadataDiskCache::new(parent.join("manifest-parts"), 100).expect("cache");
        user.keep_whole(KeptWhole::Manifest, 10_000);
        let hidden = user.for_hidden_index().expect("hidden cache");
        assert_eq!(
            hidden.budget_bytes(),
            100,
            "the user manifest is not the hidden index's"
        );
        let _ = fs::remove_dir_all(&parent);
    }

    #[tokio::test]
    async fn a_hidden_index_cache_keeps_its_own_entries() {
        let parent = tmp_root("hidden-index");
        let _ = fs::remove_dir_all(&parent);
        let cache = MetadataDiskCache::new(parent.join("parts"), 1 << 20).expect("cache");
        let hidden = cache.for_hidden_index().expect("hidden index cache");
        assert_eq!(hidden.budget_bytes(), cache.budget_bytes());

        let bytes = b"hidden-bytes".to_vec();
        let h = hash_of(&bytes);
        hidden.put(h, Bytes::copy_from_slice(&bytes)).await;
        assert!(
            parent
                .join(HIDDEN_INDEX_CACHE_SUBDIR)
                .join(cache_filename(&h))
                .exists()
        );
        assert_eq!(cache.get(&h).await, None);
        let _ = fs::remove_dir_all(&parent);
    }

    #[tokio::test]
    async fn put_is_idempotent() {
        let root = tmp_root("idempotent");
        let _ = fs::remove_dir_all(&root);
        let cache = MetadataDiskCache::new(root.clone(), 1 << 20).expect("cache");
        let bytes = b"abc".to_vec();
        let h = hash_of(&bytes);
        cache.put(h, Bytes::copy_from_slice(&bytes)).await;
        cache.put(h, Bytes::copy_from_slice(&bytes)).await;
        assert_eq!(cache.stats().n_entries, 1);
        assert_eq!(cache.stats().current_bytes, bytes.len() as u64);
        let _ = fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn oversized_part_is_not_cached() {
        let root = tmp_root("oversized");
        let _ = fs::remove_dir_all(&root);
        let cache = MetadataDiskCache::new(root.clone(), 4).expect("cache");
        let bytes = b"way too big for a 4 byte budget".to_vec();
        let h = hash_of(&bytes);
        cache.put(h, Bytes::copy_from_slice(&bytes)).await;
        assert_eq!(cache.stats().n_entries, 0);
        assert!(cache.get(&h).await.is_none());
        let _ = fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn lru_eviction_frees_room() {
        let root = tmp_root("lru");
        let _ = fs::remove_dir_all(&root);
        // Budget fits two 10-byte parts but not three.
        let cache = MetadataDiskCache::new(root.clone(), 25).expect("cache");
        let a = vec![1u8; 10];
        let b = vec![2u8; 10];
        let c = vec![3u8; 10];
        let (ha, hb, hc) = (hash_of(&a), hash_of(&b), hash_of(&c));
        cache.put(ha, Bytes::copy_from_slice(&a)).await;
        cache.put(hb, Bytes::copy_from_slice(&b)).await;
        // Touch `a` so `b` becomes the LRU victim.
        assert!(cache.get(&ha).await.is_some());
        cache.put(hc, Bytes::copy_from_slice(&c)).await;

        assert!(cache.get(&ha).await.is_some(), "a kept (recently used)");
        assert!(cache.get(&hc).await.is_some(), "c kept (just inserted)");
        assert!(cache.get(&hb).await.is_none(), "b evicted (LRU)");
        assert!(cache.stats().n_evictions >= 1);
        let _ = fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn a_removed_entry_leaves_the_disk_and_the_budget() {
        let root = tmp_root("remove");
        let _ = fs::remove_dir_all(&root);
        let cache = MetadataDiskCache::new(root.clone(), 1 << 20).expect("cache");
        let bytes = b"superseded-bytes".to_vec();
        let h = hash_of(&bytes);
        cache.put(h, Bytes::copy_from_slice(&bytes)).await;
        let path = cache.cache_path(&h);
        assert!(path.exists());

        cache.remove(&h);

        assert!(!path.exists(), "the file is unlinked");
        assert!(!cache.contains(&h));
        assert_eq!(cache.stats().current_bytes, 0);
        assert!(cache.get(&h).await.is_none());
        let _ = fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn corrupt_file_is_treated_as_miss() {
        let root = tmp_root("corrupt");
        let _ = fs::remove_dir_all(&root);
        let cache = MetadataDiskCache::new(root.clone(), 1 << 20).expect("cache");
        let bytes = b"genuine-bytes".to_vec();
        let h = hash_of(&bytes);
        cache.put(h, Bytes::copy_from_slice(&bytes)).await;
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
            let cache = MetadataDiskCache::new(root.clone(), 1 << 20).expect("cache");
            cache.put(h, Bytes::copy_from_slice(&bytes)).await;
        }
        // New cache over the same root: index rebuilt from disk.
        let reopened = MetadataDiskCache::new(root.clone(), 1 << 20).expect("reopen");
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

    /// Two writes of one part never share a temp file, so a cancelled query's
    /// write still landing cannot interleave with its rerun's and leave torn
    /// bytes under the part's name. That holds across the caches two opens of
    /// one table build over the same directory too.
    #[test]
    fn writes_of_one_part_use_distinct_temp_files() {
        let root = tmp_root("distinct-tmp");
        let _ = fs::remove_dir_all(&root);
        let cache = MetadataDiskCache::new(root.clone(), 1 << 20).expect("cache");
        let other = MetadataDiskCache::new(root.clone(), 1 << 20).expect("other cache");
        let h = hash_of(b"part");
        assert_ne!(cache.tmp_path(&h), cache.tmp_path(&h));
        assert_ne!(cache.tmp_path(&h), other.tmp_path(&h));
        let _ = fs::remove_dir_all(&root);
    }

    /// How long the dropped-put test waits for the write it handed off to land.
    const ORPHANED_WRITE_DEADLINE: Duration = Duration::from_secs(5);
    /// Poll cadence while it waits.
    const ORPHANED_WRITE_POLL: Duration = Duration::from_millis(5);

    /// A put dropped after it reserved budget, as a cancelled query drops it,
    /// still lands the part indexed and charged, with no temp file left, as
    /// if the caller had waited. A part on disk outside the index would sit
    /// past the budget where eviction cannot see it.
    #[tokio::test]
    async fn a_put_dropped_midway_still_lands_indexed_and_charged() {
        let root = tmp_root("dropped-put");
        let _ = fs::remove_dir_all(&root);
        let cache = MetadataDiskCache::new(root.clone(), 1 << 20).expect("cache");
        let bytes = b"compressed-part-bytes".to_vec();
        let h = hash_of(&bytes);

        // One poll reserves and hands the write off; dropping it is the cancel.
        let _ = cache.put(h, Bytes::copy_from_slice(&bytes)).now_or_never();
        let landed = timeout(ORPHANED_WRITE_DEADLINE, async {
            while cache.stats().n_entries == 0 {
                sleep(ORPHANED_WRITE_POLL).await;
            }
        })
        .await;
        assert!(landed.is_ok(), "the dropped put lands in the index");
        assert_eq!(
            cache.stats().current_bytes,
            bytes.len() as u64,
            "charged once"
        );
        let leftover_tmp = fs::read_dir(&root)
            .expect("cache dir")
            .flatten()
            .any(|e| e.file_name().to_string_lossy().ends_with(CACHE_TMP_SUFFIX));
        assert!(!leftover_tmp, "no temp file is left behind");
        assert_eq!(cache.get(&h).await.expect("hit"), bytes);
        let _ = fs::remove_dir_all(&root);
    }
}
