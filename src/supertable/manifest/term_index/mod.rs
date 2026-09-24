// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Table-level term index: for every `(column, term)`, the superfiles that
//! contain it, with the term's `df` in each, an upper bound on the score it
//! can reach there, and where its postings sit in that superfile.
//!
//! One artifact answers three questions the manifest used to answer with
//! three structures — *which superfiles hold this term* (the per-part and
//! per-entry term blooms), *how often it occurs table-wide* (the term-stats
//! sidecar's `df` sums), and *where its postings are* (the dictionary
//! inlined into every manifest entry's open blob). Both blooms saturate on
//! a large table and prune nothing; the inlined dictionary was most of a
//! decoded manifest's bytes. This index replaces all three with something
//! that is looked into, not loaded.
//!
//! **Shape.** A small *root* stays resident: the covered superfiles (postings
//! name them by ordinal) and, per segment, the key range and content hash of
//! every *slice*. A slice is one contiguous range of `column \x1F term` keys,
//! a few MB, holding a front-coded block dictionary (`utils::terms`) over a
//! postings region; a lookup binary-searches the root for the one slice that
//! can hold the key, fetches it, and reads one block. Prefix scans touch one
//! slice or a few adjacent ones. See `format` for the byte layout.
//!
//! **Validity.** A posting is followed only if its superfile is live in the
//! current manifest; postings for removed superfiles are simply ignored. So
//! a removal never invalidates the artifact — unlike the term-stats sidecar,
//! whose *sums* could not be attributed back to a departed superfile — and
//! the reference carries forward across every commit. A superfile with no
//! postings in any segment is uncovered and is probed directly.
//!
//! **Content addressing.** Root and slices are named by the blake3 of their
//! bytes (`term-index/root-<hash>.bin`, `term-index/slice-<hash>.bin`), so a
//! write is idempotent, a manifest pins an exact version, and GC keeps every
//! object a manifest inside the safety gap still names.

pub(crate) mod build;
pub(crate) mod format;

use std::{
    collections::{HashMap, HashSet},
    io,
    sync::Arc,
};

pub(crate) use build::{
    BuildPolicy, Built, Contribution, ContributionWriter, build, build_segment,
};
use bytes::Bytes;
pub(crate) use format::{Location, Posting, Root, Slice};
use thiserror::Error;
use uuid::Uuid;

use crate::{
    storage::{StorageError, StorageProvider},
    superfile::fts::{bm25::idf as bm25_idf, reader::BoolMode},
    supertable::{
        manifest::{RoutingRef, SuperfileEntry, disk_cache::ManifestDiskCache, part::ContentHash},
        query::prune::PruneLeaf,
    },
    utils::terms::make_key,
};

/// Object-store directory prefix for term-index objects, sibling to the
/// superfile data, manifest-parts and term-stats prefixes.
pub(crate) const STORAGE_PREFIX: &str = "term-index/";

/// Bytes of fetched slices kept resident per loaded index, least recently
/// used first out. Sized so a query burst over a large table's whole
/// vocabulary stays resident: at ~8 MiB a slice this holds ~64 slices, and a
/// slice is re-read from the manifest disk cache (a local read) only after
/// it has fallen out.
const RESIDENT_SLICE_BUDGET_BYTES: usize = 512 * 1024 * 1024;

/// A stored bound is exact, but a ceiling reaches the query through a
/// handful of roundings — the idf ratio, its product with the bound, a
/// phrase's idf recovery — that the score itself does not go through, so
/// the two can differ by a few ulps in either direction. A ceiling an ulp
/// below a real score would skip the superfile holding it; widening by
/// this factor keeps every ceiling an upper bound and prunes no less in
/// practice.
const CEILING_SLACK: f32 = 1.0 + 8.0 * f32::EPSILON;

/// Decoded posting runs kept per loaded index, keyed by `column \x1F term`.
/// A ranked query asks for the same term's postings several times — to
/// choose parts, to select superfiles, for ceilings, for locations — and
/// the second and later asks are answered here without reopening a slice.
/// Runs are small (one posting per containing superfile), so this is a
/// count bound, not a byte budget.
const RESIDENT_RUNS: usize = 4096;

/// Errors from building, storing or reading the term index.
#[derive(Debug, Error)]
pub(crate) enum TermIndexError {
    /// Object storage failed.
    #[error("term-index storage error: {0}")]
    Storage(StorageError),
    /// Bytes did not parse as the layout `format` describes.
    #[error("term-index artifact malformed: {0}")]
    Malformed(String),
    /// Fetched bytes do not hash to what the manifest references.
    #[error("term-index artifact hash mismatch")]
    HashMismatch,
    /// The build's inputs were inconsistent.
    #[error("term-index build error: {0}")]
    Build(String),
    /// A spill file could not be written or read.
    #[error("term-index spill I/O: {0}")]
    Io(#[from] io::Error),
}

impl From<StorageError> for TermIndexError {
    fn from(e: StorageError) -> Self {
        Self::Storage(e)
    }
}

impl TermIndexError {
    /// Whether this error says the object is gone or unusable — absent,
    /// unparseable, or not the bytes its hash promises — as opposed to a
    /// read that failed and may well succeed next time. A commit that
    /// cannot load the prior root restarts the index only in the first
    /// case; in the second it keeps the reference and waits.
    pub(crate) fn object_is_gone_or_corrupt(&self) -> bool {
        matches!(
            self,
            Self::Storage(StorageError::NotFound { .. }) | Self::Malformed(_) | Self::HashMismatch
        )
    }
}

fn object_uri(kind: &str, hash: &ContentHash) -> String {
    format!("{STORAGE_PREFIX}{kind}-{}.bin", hash.to_hex())
}

/// Storage URI of the slice with this content hash.
pub(crate) fn slice_uri(hash: &ContentHash) -> String {
    object_uri("slice", hash)
}

async fn put_content_addressed(
    storage: &dyn StorageProvider,
    kind: &str,
    bytes: Vec<u8>,
) -> Result<RoutingRef, TermIndexError> {
    Ok(crate::supertable::writer::put_content_addressed(
        storage,
        |hash| object_uri(kind, hash),
        bytes,
    )
    .await?)
}

/// Persist one slice; idempotent by content hash.
pub(crate) async fn write_slice(
    storage: &dyn StorageProvider,
    bytes: Vec<u8>,
) -> Result<RoutingRef, TermIndexError> {
    put_content_addressed(storage, "slice", bytes).await
}

/// Persist the root; idempotent by content hash. Call after every slice it
/// names is written, so a manifest never references a root whose slices
/// are not all present.
pub(crate) async fn write_root(
    storage: &dyn StorageProvider,
    root: &Root,
) -> Result<RoutingRef, TermIndexError> {
    put_content_addressed(storage, "root", root.encode()).await
}

/// Persist every slice of a build; idempotent by content hash.
async fn write_slices(
    storage: &dyn StorageProvider,
    slices: Vec<(ContentHash, Vec<u8>)>,
) -> Result<(), TermIndexError> {
    for (hash, bytes) in slices {
        let reference = write_slice(storage, bytes).await?;
        debug_assert_eq!(reference.content_hash, hash);
    }
    Ok(())
}

/// Persist a finished build — slices first, then the root — and return
/// the root's reference for the manifest.
pub(crate) async fn write_built(
    storage: &dyn StorageProvider,
    built: Built,
) -> Result<RoutingRef, TermIndexError> {
    write_slices(storage, built.slices).await?;
    write_root(storage, &built.root).await
}

/// Publish this commit's superfiles as a delta segment appended to `prior`
/// (the root the current manifest references, or none): write the new
/// slices, then a new root naming the prior segments plus this one, and
/// return the root's reference for the manifest CAS. Ordinals continue
/// from the prior root's superfile count, so earlier postings keep their
/// meaning. Content-addressed throughout: a retry after a lost CAS
/// re-derives the same slice hashes and re-PUTs them as no-ops.
pub(crate) async fn append_delta(
    storage: &dyn StorageProvider,
    prior: Option<Root>,
    contributions: &[Contribution],
    policy: &BuildPolicy,
) -> Result<RoutingRef, TermIndexError> {
    let mut root = prior.unwrap_or_default();
    let built = build_segment(contributions, policy, root.superfiles.len() as u32)?;
    write_slices(storage, built.slices).await?;
    root.superfiles.extend(built.superfiles);
    root.id_mins.extend(built.id_mins);
    root.segments.push(built.segment);
    write_root(storage, &root).await
}

/// Fetch a content-addressed object, through the manifest disk cache when
/// one is attached: a hit is served from local disk, already verified by
/// the cache; a miss is fetched, hash-verified, and written back
/// best-effort. The hash check is the only integrity check either tier
/// gets.
async fn fetch_verified(
    storage: &dyn StorageProvider,
    disk_cache: Option<&ManifestDiskCache>,
    uri: &str,
    hash: &ContentHash,
) -> Result<Bytes, TermIndexError> {
    if let Some(cache) = disk_cache
        && let Some(cached) = cache.get(hash).await
    {
        return Ok(Bytes::from(cached));
    }
    let (bytes, _meta) = storage.get(uri).await?;
    // Hashing a slice of several megabytes on the async worker would
    // stall every other unit that worker polls, so it runs off-thread.
    let expected = *hash;
    let to_check = bytes.clone();
    let matches =
        tokio::task::spawn_blocking(move || ContentHash::of(to_check.as_ref()) == expected)
            .await
            .map_err(|e| TermIndexError::Build(format!("hash task: {e}")))?;
    if !matches {
        return Err(TermIndexError::HashMismatch);
    }
    if let Some(cache) = disk_cache {
        cache.put(*hash, bytes.as_ref()).await;
    }
    Ok(bytes)
}

/// Fetch, hash-verify and parse the root a manifest references.
pub(crate) async fn load_root(
    storage: &dyn StorageProvider,
    reference: &RoutingRef,
) -> Result<Root, TermIndexError> {
    let bytes = fetch_verified(storage, None, &reference.uri, &reference.content_hash).await?;
    Root::decode(&bytes)
}

/// The resident half of the index plus a cache of fetched slices.
///
/// A slice is fetched whole on first use and kept; slices are immutable
/// and content-addressed, so the cache never goes stale. Bounding the cache
/// is the disk-cache layer's job once slices route through it; here it is
/// simply a map, sized by what the process has looked up.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the routing path, which lands next")
)]
pub(crate) struct TermIndex {
    root: Root,
    /// The root's storage URI — the identity a manifest pins, and what a
    /// cache compares to decide whether a loaded index is still current.
    root_uri: String,
    /// Every superfile the root lists, with its smallest doc id. A live
    /// superfile absent from it was committed before the index existed
    /// and is routed the old way.
    indexed: HashMap<Uuid, i128>,
    storage: Arc<dyn StorageProvider>,
    disk_cache: Option<Arc<ManifestDiskCache>>,
    /// Fetched slices by content hash; see [`RESIDENT_SLICE_BUDGET_BYTES`].
    slices: tokio::sync::Mutex<Resident<ContentHash, Bytes>>,
    /// Decoded runs by key; see [`RESIDENT_RUNS`]. A std mutex: nothing
    /// awaits while it is held.
    runs: std::sync::Mutex<Resident<Vec<u8>, Arc<Vec<Posting>>>>,
}

/// A bounded resident set, least recently used first out: `weigh` gives
/// each value's cost against `budget`, and a read refreshes recency, so a
/// burst that cycles through more than fits keeps what it keeps touching.
struct Resident<K, V> {
    map: HashMap<K, (V, u64)>,
    by_use: std::collections::BTreeMap<u64, K>,
    tick: u64,
    total: usize,
    budget: usize,
    weigh: fn(&V) -> usize,
}

impl<K: Eq + std::hash::Hash + Clone, V: Clone> Resident<K, V> {
    fn new(budget: usize, weigh: fn(&V) -> usize) -> Self {
        Self {
            map: HashMap::new(),
            by_use: std::collections::BTreeMap::new(),
            tick: 0,
            total: 0,
            budget,
            weigh,
        }
    }

    fn get(&mut self, key: &K) -> Option<V> {
        let (value, used) = self.map.get_mut(key)?;
        self.tick += 1;
        self.by_use.remove(used);
        *used = self.tick;
        self.by_use.insert(self.tick, key.clone());
        Some(value.clone())
    }

    /// Insert unless resident, evicting least recently used until it fits.
    fn insert(&mut self, key: K, value: V) {
        if self.map.contains_key(&key) {
            return;
        }
        let weight = (self.weigh)(&value);
        while self.total + weight > self.budget
            && let Some((_, old)) = self.by_use.pop_first()
            && let Some((gone, _)) = self.map.remove(&old)
        {
            self.total -= (self.weigh)(&gone);
        }
        self.total += weight;
        self.tick += 1;
        self.by_use.insert(self.tick, key.clone());
        self.map.insert(key, (value, self.tick));
    }
}

#[cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the routing path, which lands next")
)]
impl TermIndex {
    /// Wrap a loaded root.
    pub(crate) fn new(
        root: Root,
        root_uri: String,
        storage: Arc<dyn StorageProvider>,
        disk_cache: Option<Arc<ManifestDiskCache>>,
    ) -> Self {
        let indexed = root
            .superfiles
            .iter()
            .copied()
            .zip(root.id_mins.iter().copied())
            .collect();
        Self {
            root,
            root_uri,
            indexed,
            storage,
            disk_cache,
            slices: tokio::sync::Mutex::new(Resident::new(RESIDENT_SLICE_BUDGET_BYTES, Bytes::len)),
            runs: std::sync::Mutex::new(Resident::new(RESIDENT_RUNS, |_| 1)),
        }
    }

    /// Fetch, verify and parse the root a manifest references, through the
    /// manifest disk cache when one is attached.
    pub(crate) async fn load(
        storage: Arc<dyn StorageProvider>,
        disk_cache: Option<Arc<ManifestDiskCache>>,
        reference: &RoutingRef,
    ) -> Result<Self, TermIndexError> {
        let bytes = fetch_verified(
            storage.as_ref(),
            disk_cache.as_deref(),
            &reference.uri,
            &reference.content_hash,
        )
        .await?;
        let root = Root::decode(&bytes)?;
        Ok(Self::new(root, reference.uri.clone(), storage, disk_cache))
    }

    /// The root URI this index was loaded from.
    pub(crate) fn root_uri(&self) -> &str {
        &self.root_uri
    }

    /// Whether the root lists `superfile` — i.e. whether its postings are
    /// in this index at all.
    pub(crate) fn is_indexed(&self, superfile: &Uuid) -> bool {
        self.indexed.contains_key(superfile)
    }

    /// The smallest doc id of a listed superfile — the key that finds its
    /// manifest part from the part's recorded id range.
    pub(crate) fn id_min_of(&self, superfile: &Uuid) -> Option<i128> {
        self.indexed.get(superfile).copied()
    }

    /// The superfiles a term or prefix leaf routes to, or `None` when the
    /// leaf is not one the index answers — a scalar leaf, an empty term
    /// list, a non-UTF-8 prefix — or the lookup failed; the caller then
    /// keeps its summary-based answer.
    pub(crate) async fn route_leaf(&self, leaf: &PruneLeaf) -> Option<HashSet<Uuid>> {
        match leaf {
            PruneLeaf::TermPresence {
                column,
                terms,
                mode,
            } if !terms.is_empty() => {
                let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
                self.route(column, &refs, *mode).await.ok()
            }
            PruneLeaf::Prefix { column, prefix } => {
                let prefix = std::str::from_utf8(prefix).ok()?;
                self.route_prefix(column, prefix).await.ok()
            }
            _ => None,
        }
    }

    /// The superfiles that can match `terms` in `column` under `mode`:
    /// the union of the terms' posting sets for `Or`, their intersection
    /// for `And`. Exact for indexed superfiles; the caller decides what to
    /// do with live superfiles the root does not list. Ordinals that no
    /// longer resolve are skipped.
    pub(crate) async fn route(
        &self,
        column: &str,
        terms: &[&str],
        mode: BoolMode,
    ) -> Result<HashSet<Uuid>, TermIndexError> {
        let mut out: Option<HashSet<Uuid>> = None;
        for term in terms {
            let set: HashSet<Uuid> = self
                .postings(column, term)
                .await?
                .iter()
                .filter_map(|p| self.superfile_id(p.superfile))
                .collect();
            out = Some(match (out, mode) {
                (None, _) => set,
                (Some(acc), BoolMode::Or) => acc.union(&set).copied().collect(),
                (Some(acc), BoolMode::And) => acc.intersection(&set).copied().collect(),
            });
        }
        Ok(out.unwrap_or_default())
    }

    /// Per-superfile score ceilings for a query: for each entry, the sum
    /// over `terms` of the term's bound in that superfile, rescaled from
    /// the superfile's own idf to the idf the query scores with, plus for
    /// each phrase the cursor's own phrase ceiling — the members' idf sum
    /// times the smallest member bound in idf-scaled form — so a phrase is
    /// bounded by its rarest member. A superfile lacking a term contributes
    /// nothing for it; one the root does not list gets `+∞`, so it is
    /// opened first and unconditionally. `idf_used(term, local_idf)` is the
    /// idf the query scores `term` with given the superfile's own.
    ///
    /// **Precondition: the query scores with each column's declared `k1`
    /// and `b`.** The stored bounds were baked at those parameters and are
    /// rescaled here by idf alone, never by the factor an override would
    /// need (`bound_scale`), so under an override they are not upper
    /// bounds and must not order or skip anything. The ranked path keeps
    /// an override on the unordered fan-out for exactly this reason; a new
    /// caller must do the same or rescale first.
    pub(crate) async fn query_ceilings(
        &self,
        column: &str,
        terms: &[&str],
        phrases: &[Vec<&str>],
        entries: &[Arc<SuperfileEntry>],
        idf_used: &(dyn Fn(&str, f32) -> f32 + Sync),
    ) -> Result<HashMap<Uuid, f32>, TermIndexError> {
        let scored_docs: HashMap<Uuid, u64> = entries
            .iter()
            .map(|e| {
                let n = e
                    .fts_summary
                    .get(column)
                    .and_then(|s| s.length_stats.as_ref().map(|l| l.n_scored_docs))
                    .unwrap_or(e.n_docs);
                (e.superfile_id, n)
            })
            .collect();
        // Per term: superfile → (bound rescaled to the query's idf, bound / local idf).
        let mut per_term: HashMap<&str, HashMap<Uuid, (f32, f32)>> = HashMap::new();
        let mut all_terms: Vec<&str> = terms.to_vec();
        all_terms.extend(phrases.iter().flatten().copied());
        all_terms.sort_unstable();
        all_terms.dedup();
        for term in all_terms {
            let mut by_sf = HashMap::new();
            for p in self.postings(column, term).await?.iter() {
                let Some(id) = self.superfile_id(p.superfile) else {
                    continue;
                };
                let Some(&n) = scored_docs.get(&id) else {
                    continue;
                };
                let local_idf = bm25_idf(n, p.df.min(n));
                let ratio = if local_idf > 0.0 {
                    idf_used(term, local_idf) / local_idf
                } else {
                    1.0
                };
                let scaled = if local_idf > 0.0 {
                    p.bound / local_idf
                } else {
                    p.bound
                };
                by_sf.insert(id, (p.bound * ratio, scaled));
            }
            per_term.insert(term, by_sf);
        }
        let mut out: HashMap<Uuid, f32> = HashMap::with_capacity(entries.len());
        for e in entries {
            let id = e.superfile_id;
            if !self.is_indexed(&id) {
                out.insert(id, f32::INFINITY);
                continue;
            }
            let mut ceiling = 0.0f32;
            for term in terms {
                if let Some((rescaled, _)) = per_term.get(term).and_then(|m| m.get(&id)) {
                    ceiling += rescaled;
                }
            }
            for phrase in phrases {
                let mut idf_sum = 0.0f32;
                let mut min_scaled = f32::INFINITY;
                let mut complete = true;
                for member in phrase {
                    match per_term.get(member).and_then(|m| m.get(&id)) {
                        Some((rescaled, scaled)) => {
                            // rescaled = bound × (idf_used / local_idf); recover idf_used
                            // from the pair without a second idf call.
                            let local_scaled = *scaled;
                            let idf = if local_scaled > 0.0 {
                                rescaled / local_scaled
                            } else {
                                0.0
                            };
                            idf_sum += idf;
                            min_scaled = min_scaled.min(local_scaled);
                        }
                        None => {
                            complete = false;
                            break;
                        }
                    }
                }
                // Every member carries the placeholder that bounds nothing:
                // the phrase is unbounded here, not absent.
                if complete {
                    ceiling += match min_scaled.is_finite() {
                        true => idf_sum * min_scaled,
                        false => f32::INFINITY,
                    };
                }
            }
            out.insert(id, ceiling * CEILING_SLACK);
        }
        Ok(out)
    }

    /// For each indexed superfile among `entries`, the `(term, df,
    /// location)` of every one of `terms` it holds — what a reader needs
    /// to build its cursors without reading the superfile's dictionary.
    pub(crate) async fn locations(
        &self,
        column: &str,
        terms: &[&str],
        entries: &[Arc<SuperfileEntry>],
    ) -> Result<HashMap<Uuid, Vec<(String, u64, Location)>>, TermIndexError> {
        let live: HashSet<Uuid> = entries
            .iter()
            .map(|e| e.superfile_id)
            .filter(|id| self.is_indexed(id))
            .collect();
        let mut out: HashMap<Uuid, Vec<(String, u64, Location)>> = HashMap::new();
        for term in terms {
            for p in self.postings(column, term).await?.iter() {
                let Some(id) = self.superfile_id(p.superfile) else {
                    continue;
                };
                if live.contains(&id) {
                    out.entry(id)
                        .or_default()
                        .push(((*term).to_owned(), p.df, p.location));
                }
            }
        }
        Ok(out)
    }

    /// The superfiles holding any term with `prefix` in `column`.
    pub(crate) async fn route_prefix(
        &self,
        column: &str,
        prefix: &str,
    ) -> Result<HashSet<Uuid>, TermIndexError> {
        let mut out = HashSet::new();
        self.for_each_prefix(column, prefix, |_, run| {
            out.extend(run.iter().filter_map(|p| self.superfile_id(p.superfile)));
            true
        })
        .await?;
        Ok(out)
    }

    /// The resident root.
    pub(crate) fn root(&self) -> &Root {
        &self.root
    }

    /// The superfile a posting's ordinal names.
    pub(crate) fn superfile_id(&self, ordinal: u32) -> Option<Uuid> {
        self.root.superfiles.get(ordinal as usize).copied()
    }

    async fn slice_bytes(&self, hash: &ContentHash) -> Result<Bytes, TermIndexError> {
        if let Some(b) = self.slices.lock().await.get(hash) {
            return Ok(b);
        }
        let bytes = fetch_verified(
            self.storage.as_ref(),
            self.disk_cache.as_deref(),
            &slice_uri(hash),
            hash,
        )
        .await?;
        self.slices.lock().await.insert(*hash, bytes.clone());
        Ok(bytes)
    }

    /// Every posting for `term` in `column`, across all segments, in the
    /// order the segments were written. Empty when no segment holds the
    /// term. The caller filters to superfiles live in its manifest.
    pub(crate) async fn postings(
        &self,
        column: &str,
        term: &str,
    ) -> Result<Arc<Vec<Posting>>, TermIndexError> {
        let key = make_key(column, term);
        if let Some(run) = self.runs.lock().expect("resident runs lock").get(&key) {
            return Ok(run);
        }
        let mut out = Vec::new();
        let refs: Vec<_> = self.root.slices_for_key(&key).cloned().collect();
        for r in refs {
            let bytes = self.slice_bytes(&r.content_hash).await?;
            let slice = Slice::open(&bytes)?;
            if let Some(run) = slice.postings(&key)? {
                out.extend(run);
            }
        }
        let run = Arc::new(out);
        self.runs
            .lock()
            .expect("resident runs lock")
            .insert(key, Arc::clone(&run));
        Ok(run)
    }

    /// Visit every term in `column` with `prefix`, with its postings, until
    /// `visit` returns `false`. Terms arrive in key order within a segment.
    pub(crate) async fn for_each_prefix(
        &self,
        column: &str,
        prefix: &str,
        mut visit: impl FnMut(&[u8], Vec<Posting>) -> bool,
    ) -> Result<(), TermIndexError> {
        let key_prefix = make_key(column, prefix);
        let refs: Vec<_> = self.root.slices_for_prefix(&key_prefix).cloned().collect();
        let mut keep_going = true;
        for r in refs {
            if !keep_going {
                break;
            }
            let bytes = self.slice_bytes(&r.content_hash).await?;
            let slice = Slice::open(&bytes)?;
            slice.for_each_prefix(&key_prefix, |k, run| {
                keep_going = visit(k, run);
                keep_going
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use tempfile::TempDir;

    use super::*;
    use crate::{
        storage::LocalFsStorageProvider,
        supertable::query::prune::select_superfiles,
        utils::terms::{FstValue, make_key},
    };

    fn contribution(dir: &TempDir, id: u128, terms: &[(&str, &str, u64)]) -> Contribution {
        let mut w = ContributionWriter::create(dir.path(), Uuid::from_u128(id), id as i128 * 1000)
            .expect("create");
        let mut keyed: Vec<(Vec<u8>, u64)> = terms
            .iter()
            .map(|(c, t, df)| (make_key(c, t), *df))
            .collect();
        keyed.sort();
        for (i, (key, df)) in keyed.iter().enumerate() {
            let location = match *df {
                1 => Location::Inline {
                    doc_id: i as u32,
                    tf: 1,
                },
                d if d <= 128 => Location::Short {
                    offset: i as u64 * 100,
                    len: 50,
                },
                _ => Location::Pfor {
                    offset: i as u64 * 1000,
                    len: 800,
                },
            };
            w.push(key, *df, f32::INFINITY, location).expect("push");
        }
        w.finish().expect("finish")
    }

    #[test]
    fn contribution_rejects_out_of_order_keys() {
        let dir = TempDir::new().expect("tempdir");
        let mut w = ContributionWriter::create(dir.path(), Uuid::from_u128(1), 0).expect("create");
        w.push(b"b", 1, 1.0, Location::None).expect("first");
        assert!(matches!(
            w.push(b"a", 1, 1.0, Location::None),
            Err(TermIndexError::Build(m)) if m.contains("ascending")
        ));
        assert!(
            w.push(b"b", 1, 1.0, Location::None).is_err(),
            "equal keys are not ascending"
        );
    }

    /// The merge assigns superfile ordinals by contribution order, gathers
    /// every superfile's posting for a term into one run in ascending
    /// ordinal, and applies the location policy above the threshold.
    #[test]
    fn build_merges_contributions_into_runs_by_ordinal() {
        let dir = TempDir::new().expect("tempdir");
        let a = contribution(
            &dir,
            1,
            &[
                ("body", "alpha", 3),
                ("body", "beta", 1),
                ("title", "zed", 500),
            ],
        );
        let b = contribution(&dir, 2, &[("body", "alpha", 7), ("body", "gamma", 2)]);
        let c = contribution(&dir, 3, &[("body", "alpha", 1)]);
        let policy = BuildPolicy {
            slice_target_bytes: usize::MAX,
            range_max_superfiles: 2,
        };
        let built = build(&[a, b, c], &policy).expect("build");
        assert_eq!(
            built.root.superfiles,
            vec![Uuid::from_u128(1), Uuid::from_u128(2), Uuid::from_u128(3)]
        );
        assert_eq!(built.root.segments.len(), 1);
        assert_eq!(built.slices.len(), 1, "everything fits one slice");
        let slice = Slice::open(&built.slices[0].1).expect("open");

        let alpha = slice
            .postings(&make_key("body", "alpha"))
            .expect("ok")
            .expect("present");
        assert_eq!(
            alpha.iter().map(|p| p.superfile).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            alpha.iter().map(|p| p.df).collect::<Vec<_>>(),
            vec![3, 7, 1]
        );
        assert!(
            alpha.iter().all(|p| p.location == Location::None),
            "three superfiles exceeds the range threshold of two: locations dropped"
        );

        let beta = slice
            .postings(&make_key("body", "beta"))
            .expect("ok")
            .expect("present");
        assert_eq!(beta.len(), 1);
        assert_eq!(beta[0].superfile, 0);
        assert!(
            matches!(beta[0].location, Location::Inline { .. }),
            "below the threshold: location kept"
        );

        let zed = slice
            .postings(&make_key("title", "zed"))
            .expect("ok")
            .expect("present");
        assert!(matches!(
            zed[0].location,
            Location::Pfor {
                offset: 2000,
                len: 800
            }
        ));

        assert_eq!(
            slice.postings(&make_key("body", "delta")).expect("ok"),
            None
        );
        let s = &built.root.segments[0].slices[0];
        assert_eq!(s.first_key, make_key("body", "alpha"));
        assert_eq!(s.last_key, make_key("title", "zed"));
        assert_eq!(s.content_hash, ContentHash::of(&built.slices[0].1));
        assert_eq!(s.len as usize, built.slices[0].1.len());
    }

    /// Slices are cut at term boundaries once the target is reached, so
    /// every slice is a contiguous key range and the root routes each key
    /// to exactly one of them.
    #[test]
    fn build_cuts_contiguous_slices_the_root_routes_to() {
        let dir = TempDir::new().expect("tempdir");
        let terms: Vec<(String, String, u64)> = (0..200)
            .map(|i| ("body".to_owned(), format!("term{i:04}"), 1 + (i % 3) as u64))
            .collect();
        let refs: Vec<(&str, &str, u64)> = terms
            .iter()
            .map(|(c, t, d)| (c.as_str(), t.as_str(), *d))
            .collect();
        let a = contribution(&dir, 1, &refs);
        let policy = BuildPolicy {
            slice_target_bytes: 600,
            range_max_superfiles: 64,
        };
        let built = build(&[a], &policy).expect("build");
        let slices = &built.root.segments[0].slices;
        assert!(
            slices.len() > 3,
            "small target must cut several slices, got {}",
            slices.len()
        );
        for w in slices.windows(2) {
            assert!(
                w[0].last_key < w[1].first_key,
                "slices are disjoint and ordered"
            );
        }
        let by_hash: HashMap<_, _> = built.slices.iter().cloned().collect();
        for (c, t, d) in &terms {
            let key = make_key(c, t);
            let hits: Vec<_> = built.root.slices_for_key(&key).collect();
            assert_eq!(hits.len(), 1, "exactly one slice can hold {t}");
            let slice = Slice::open(&by_hash[&hits[0].content_hash]).expect("open");
            let run = slice.postings(&key).expect("ok").expect("present");
            assert_eq!(run[0].df, *d);
        }
    }

    /// End to end through storage: write, reference, load, look up, scan a
    /// prefix across a slice boundary — and refuse a tampered object.
    #[tokio::test]
    async fn write_load_lookup_and_prefix_through_storage() {
        let dir = TempDir::new().expect("tempdir");
        let terms: Vec<(String, String, u64)> = (0..120)
            .map(|i| ("body".to_owned(), format!("ab{i:03}"), 2))
            .collect();
        let refs: Vec<(&str, &str, u64)> = terms
            .iter()
            .map(|(c, t, d)| (c.as_str(), t.as_str(), *d))
            .collect();
        let a = contribution(&dir, 7, &refs);
        let b = contribution(&dir, 8, &[("body", "ab005", 9), ("body", "zz", 1)]);
        let policy = BuildPolicy {
            slice_target_bytes: 800,
            range_max_superfiles: 64,
        };
        let built = build(&[a, b], &policy).expect("build");
        let n_slices = built.root.segments[0].slices.len();
        assert!(n_slices > 1);

        let store_dir = TempDir::new().expect("store dir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local fs"));
        let reference = write_built(storage.as_ref(), built).await.expect("write");
        assert!(reference.uri.starts_with("term-index/root-"));
        let root = load_root(storage.as_ref(), &reference).await.expect("load");
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        assert_eq!(
            index.root().superfiles,
            vec![Uuid::from_u128(7), Uuid::from_u128(8)]
        );

        let run = index.postings("body", "ab005").await.expect("ok");
        assert_eq!(run.len(), 2, "both superfiles hold ab005");
        assert_eq!((run[0].superfile, run[0].df), (0, 2));
        assert_eq!((run[1].superfile, run[1].df), (1, 9));
        assert_eq!(index.superfile_id(1), Some(Uuid::from_u128(8)));
        assert!(index.postings("body", "nope").await.expect("ok").is_empty());
        assert!(
            index
                .postings("title", "ab005")
                .await
                .expect("ok")
                .is_empty(),
            "column is part of the key"
        );

        let mut seen = 0usize;
        index
            .for_each_prefix("body", "ab", |_, _| {
                seen += 1;
                true
            })
            .await
            .expect("scan");
        assert_eq!(
            seen, 120,
            "a prefix spanning several slices visits every term once"
        );

        let mut seen = 0usize;
        index
            .for_each_prefix("body", "ab", |_, _| {
                seen += 1;
                seen < 5
            })
            .await
            .expect("scan");
        assert_eq!(seen, 5, "the visitor can stop the scan");

        // Tamper with the root object: the hash check refuses it.
        let mut bad = reference.clone();
        bad.content_hash = ContentHash::of(b"other");
        assert!(matches!(
            load_root(storage.as_ref(), &bad).await,
            Err(TermIndexError::HashMismatch)
        ));
    }
    const DOCS_PER_SEGMENT: usize = 40;
    const SEGMENTS: usize = 3;

    fn title_schema() -> Arc<arrow_schema::Schema> {
        use arrow_schema::{DataType, Field, Schema};
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]))
    }

    /// Options for an FTS table on `storage`, with a small writer pool.
    fn fresh_options(storage: &Arc<dyn StorageProvider>) -> crate::supertable::SupertableOptions {
        use crate::{superfile::builder::FtsConfig, supertable::SupertableOptions};
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool"),
        );
        // Positions on, so phrase queries have something to verify against.
        SupertableOptions::new(
            title_schema(),
            vec![FtsConfig::new("title").positions(true)],
            Vec::new(),
        )
        .expect("options")
        .with_writer_pool(pool)
        .with_storage(Arc::clone(storage))
    }

    /// An empty FTS table on local-filesystem storage, its options adjusted
    /// by `customize`.
    fn table_with(
        customize: impl FnOnce(
            crate::supertable::SupertableOptions,
        ) -> crate::supertable::SupertableOptions,
    ) -> (
        TempDir,
        Arc<dyn StorageProvider>,
        crate::supertable::Supertable,
    ) {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs"));
        let options = customize(fresh_options(&storage));
        (
            dir,
            storage,
            crate::supertable::Supertable::create(options).expect("create"),
        )
    }

    /// An empty FTS table on local-filesystem storage.
    fn fresh_table() -> (
        TempDir,
        Arc<dyn StorageProvider>,
        crate::supertable::Supertable,
    ) {
        table_with(|o| o)
    }

    /// Commit one segment: every title holds `shared`; segment `s` holds
    /// `alpha` in the titles whose index is a multiple of `s + 2`. Returns
    /// the segment's `alpha` count.
    fn commit_segment(st: &crate::supertable::Supertable, segment: usize) -> u64 {
        use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
        let titles: Vec<String> = (0..DOCS_PER_SEGMENT)
            .map(|i| {
                let topic = if i % (segment + 2) == 0 {
                    "alpha"
                } else {
                    "beta"
                };
                format!("{topic} shared s{segment}d{i:02}")
            })
            .collect();
        let alpha = titles.iter().filter(|t| t.starts_with("alpha")).count() as u64;
        let arr: ArrayRef = Arc::new(LargeStringArray::from(
            titles.iter().map(String::as_str).collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(title_schema(), vec![arr]).expect("batch");
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
        alpha
    }

    /// Compaction-free optimize: the maintenance passes alone.
    fn stats_only_optimize(st: &crate::supertable::Supertable) {
        use crate::{CompactionSettings, OptimizeOptions};
        st.optimize(&OptimizeOptions::compact(CompactionSettings {
            min_fill_percent: 100,
            min_superfiles_for_merge: u64::MAX,
            ..CompactionSettings::default()
        }))
        .expect("optimize");
    }

    /// The live superfile ids and the root's covered set, for comparison.
    fn live_and_covered(
        st: &crate::supertable::Supertable,
        storage: &Arc<dyn StorageProvider>,
        rt: &tokio::runtime::Runtime,
    ) -> (std::collections::HashSet<Uuid>, Root) {
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let live = manifest
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect();
        let reference = manifest
            .term_index_ref()
            .cloned()
            .expect("a term-index reference");
        let root = rt
            .block_on(load_root(storage.as_ref(), &reference))
            .expect("root");
        (live, root)
    }

    /// A fragmented three-segment FTS table, optimized with compaction
    /// disabled so only the maintenance passes run. Returns the storage
    /// root, the table, and the per-segment `alpha` counts.
    fn optimized_fragmented_table() -> (
        TempDir,
        Arc<dyn StorageProvider>,
        crate::supertable::Supertable,
        Vec<u64>,
    ) {
        let (dir, storage, st) = fresh_table();
        let alpha_per_segment: Vec<u64> = (0..SEGMENTS).map(|s| commit_segment(&st, s)).collect();
        assert!(
            st.reader().expect("reader").n_superfiles() >= SEGMENTS,
            "fixture must stay fragmented"
        );
        stats_only_optimize(&st);
        (dir, storage, st, alpha_per_segment)
    }

    /// Every commit publishes its superfiles' postings in the same manifest
    /// as the entries: after each commit the root covers exactly the live
    /// set, one delta segment per commit, with per-superfile `df` right —
    /// and a later optimize folds the deltas into one base segment that
    /// answers identically.
    #[test]
    fn commit_publishes_postings_with_the_manifest() {
        let (_dir, storage, st) = fresh_table();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let mut alphas = Vec::new();
        for segment in 0..SEGMENTS {
            alphas.push(commit_segment(&st, segment));
            let (live, root) = live_and_covered(&st, &storage, &rt);
            let covered: std::collections::HashSet<Uuid> =
                root.superfiles.iter().copied().collect();
            assert_eq!(
                covered, live,
                "after commit {segment}: every visible superfile has postings, and only those"
            );
            assert_eq!(
                root.segments.len(),
                segment + 1,
                "one delta segment per commit"
            );
            let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
            let shared = rt
                .block_on(index.postings("title", "shared"))
                .expect("lookup");
            assert_eq!(
                shared.len(),
                live.len(),
                "`shared` posts once per live superfile"
            );
            let n_docs: HashMap<Uuid, u64> = st
                .reader()
                .expect("reader")
                .manifest()
                .get_all_superfiles()
                .iter()
                .map(|e| (e.superfile_id, e.n_docs))
                .collect();
            for p in shared.iter() {
                let id = index
                    .superfile_id(p.superfile)
                    .expect("ordinal resolves across deltas");
                assert_eq!(p.df, n_docs[&id]);
            }
        }
        // Optimize folds every delta into one base segment with the same answers.
        let before: Vec<u64> = {
            let (_, root) = live_and_covered(&st, &storage, &rt);
            let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
            let mut v: Vec<u64> = rt
                .block_on(index.postings("title", "alpha"))
                .expect("lookup")
                .iter()
                .map(|p| p.df)
                .collect();
            v.sort_unstable();
            v
        };
        stats_only_optimize(&st);
        let (live, root) = live_and_covered(&st, &storage, &rt);
        assert_eq!(root.segments.len(), 1, "optimize rebuilds one base segment");
        assert_eq!(
            root.superfiles
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>(),
            live
        );
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let mut after: Vec<u64> = rt
            .block_on(index.postings("title", "alpha"))
            .expect("lookup")
            .iter()
            .map(|p| p.df)
            .collect();
        after.sort_unstable();
        alphas.sort_unstable();
        assert_eq!(after, before, "the fold changes layout, not answers");
        assert_eq!(after, alphas, "and the answers are the fixture's");
    }

    /// Optimize publishes a term index over every live superfile, one
    /// contribution per superfile, with the artifact's `df` per superfile
    /// equal to what the fixture put there — checked without opening a
    /// superfile.
    #[test]
    fn optimize_publishes_a_term_index_over_every_superfile() {
        let (_dir, storage, st, mut alpha_per_segment) = optimized_fragmented_table();
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let reference = manifest
            .term_index_ref()
            .cloned()
            .expect("optimize must publish a term-index reference");
        assert!(reference.uri.starts_with(STORAGE_PREFIX));

        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (live, root) = live_and_covered(&st, &storage, &rt);
        let covered: std::collections::HashSet<Uuid> = root.superfiles.iter().copied().collect();
        assert_eq!(
            covered, live,
            "every live superfile contributes, and nothing else"
        );
        assert_eq!(
            root.segments.len(),
            1,
            "a maintenance build is one base segment"
        );

        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let shared = rt
            .block_on(index.postings("title", "shared"))
            .expect("lookup");
        assert_eq!(shared.len(), live.len(), "`shared` is in every superfile");
        let n_docs_by_id: HashMap<Uuid, u64> = manifest
            .get_all_superfiles()
            .iter()
            .map(|e| (e.superfile_id, e.n_docs))
            .collect();
        for p in shared.iter() {
            let id = index.superfile_id(p.superfile).expect("ordinal resolves");
            assert_eq!(
                p.df, n_docs_by_id[&id],
                "`shared` df is that superfile's doc count"
            );
            assert!(
                p.bound.is_finite() && p.bound > 0.0,
                "every posting carries a real ceiling"
            );
            assert_ne!(
                p.location,
                Location::None,
                "few superfiles: locations are carried"
            );
        }
        let mut alpha: Vec<u64> = rt
            .block_on(index.postings("title", "alpha"))
            .expect("lookup")
            .iter()
            .map(|p| p.df)
            .collect();
        alpha.sort_unstable();
        alpha_per_segment.sort_unstable();
        assert_eq!(
            alpha, alpha_per_segment,
            "per-superfile df for `alpha` matches the fixture"
        );
        assert!(
            rt.block_on(index.postings("title", "absent"))
                .expect("lookup")
                .is_empty()
        );
    }
    /// GC keeps the referenced root and every slice it names, and sweeps
    /// a slice nothing references. The live set is read from the root, so
    /// the slices survive even though the manifest never lists them.
    #[test]
    fn gc_keeps_the_term_index_and_sweeps_an_orphan_slice() {
        use std::{fs, time::Duration};

        let (dir, storage, st, _) = optimized_fragmented_table();
        let reference = st
            .reader()
            .expect("reader")
            .manifest()
            .term_index_ref()
            .cloned()
            .expect("reference");
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let root = rt
            .block_on(load_root(storage.as_ref(), &reference))
            .expect("root");
        let slice_paths: Vec<_> = root
            .segments
            .iter()
            .flat_map(|s| s.slices.iter())
            .map(|s| dir.path().join(slice_uri(&s.content_hash)))
            .collect();
        assert!(!slice_paths.is_empty());
        let orphan = dir
            .path()
            .join(slice_uri(&ContentHash::of(b"nothing references me")));
        fs::write(&orphan, b"stray slice bytes").expect("plant orphan");

        let report = st.gc(Duration::ZERO).expect("gc");

        assert!(
            dir.path().join(&reference.uri).exists(),
            "the referenced root survives"
        );
        for p in &slice_paths {
            assert!(
                p.exists(),
                "a slice the root names survives: {}",
                p.display()
            );
        }
        assert!(!orphan.exists(), "an unreferenced slice is swept");
        assert!(report.objects_deleted >= 1);
    }
    /// Compaction commits through the same path: the merged superfile's
    /// postings publish as a delta in the commit that removes its inputs.
    /// The root's superfile list is append-only, so the removed inputs stay
    /// listed — a reader ignores postings for superfiles no longer live —
    /// while every live superfile, the merged one included, is covered.
    #[test]
    fn compaction_publishes_the_merged_superfile_as_a_delta() {
        use crate::CompactionSettings;

        let (_dir, storage, st) = fresh_table();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        for segment in 0..SEGMENTS {
            commit_segment(&st, segment);
        }
        let (live_before, root_before) = live_and_covered(&st, &storage, &rt);
        assert!(live_before.len() >= SEGMENTS);
        let segments_before = root_before.segments.len();
        // `s0d00` names one title in segment 0 — a superfile compaction
        // removes; `alpha` spans every segment.
        let before: Vec<Vec<String>> = ["alpha", "s0d00"]
            .iter()
            .map(|q| ranked_titles(&st, q))
            .collect();
        assert_eq!(before[1].len(), 1);

        st.compact(&CompactionSettings {
            min_fill_percent: 1,
            min_superfiles_for_merge: 2,
            ..CompactionSettings::default()
        })
        .expect("compact");

        let (live_after, root_after) = live_and_covered(&st, &storage, &rt);
        assert!(
            live_after.len() < live_before.len(),
            "compaction must have merged"
        );
        let covered: std::collections::HashSet<Uuid> =
            root_after.superfiles.iter().copied().collect();
        assert!(
            live_after.is_subset(&covered),
            "every live superfile — the merged one included — has postings"
        );
        assert!(
            covered.is_superset(&live_before),
            "removed inputs stay listed; readers filter them by liveness"
        );
        assert_eq!(
            root_after.segments.len(),
            segments_before + 1,
            "the compaction commit appended one delta"
        );

        // The merged superfile's postings are right, and the removed
        // inputs' postings are still there to be filtered out by liveness.
        let index = TermIndex::new(root_after, String::new(), Arc::clone(&storage), None);
        let shared = rt
            .block_on(index.postings("title", "shared"))
            .expect("lookup");
        let n_docs: HashMap<Uuid, u64> = st
            .reader()
            .expect("reader")
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|e| (e.superfile_id, e.n_docs))
            .collect();
        let live_postings: Vec<_> = shared
            .iter()
            .filter(|p| live_after.contains(&index.superfile_id(p.superfile).expect("ordinal")))
            .collect();
        assert_eq!(live_postings.len(), live_after.len());
        for p in live_postings {
            assert_eq!(
                p.df,
                n_docs[&index.superfile_id(p.superfile).expect("ordinal")]
            );
        }
        assert!(
            shared.len() > live_after.len(),
            "the removed inputs' postings remain until the next fold"
        );

        // Through the query path, a posting is followed only if its
        // superfile is live: the removed inputs' stale postings route
        // nowhere, the merged superfile's fresh ones route to it, and
        // every row comes back.
        let after: Vec<Vec<String>> = ["alpha", "s0d00"]
            .iter()
            .map(|q| ranked_titles(&st, q))
            .collect();
        assert_eq!(after, before, "the same rows rank after compaction");
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let routed = rt
            .block_on(select_superfiles(
                manifest,
                &[PruneLeaf::TermPresence {
                    column: "title".into(),
                    terms: vec!["s0d00".into()],
                    mode: BoolMode::Or,
                }],
            ))
            .expect("select");
        assert_eq!(
            routed.len(),
            1,
            "a removed input's token routes to one superfile"
        );
        assert!(
            live_after.contains(&routed[0].superfile_id),
            "and that superfile is the live, merged one"
        );
    }

    /// Titles ranked for `query`, in result order, over the whole table.
    fn ranked_titles(st: &crate::supertable::Supertable, query: &str) -> Vec<String> {
        use arrow_array::{Array, LargeStringArray};

        use crate::Bm25SearchOptions;
        let reader = st.reader().expect("reader");
        let batches = reader
            .bm25_search(
                "title",
                query,
                DOCS_PER_SEGMENT * SEGMENTS,
                Bm25SearchOptions::new(),
                Some(&["title"]),
            )
            .expect("search");
        let mut out = Vec::new();
        for b in &batches {
            let titles = b
                .column(0)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("title");
            for i in 0..b.num_rows() {
                out.push(titles.value(i).to_owned());
            }
        }
        out
    }
    /// A row update replaces its rows with a fresh superfile through the
    /// update pipeline rather than the append path. Its postings publish
    /// with its entry all the same, so on a lazily loaded table — where
    /// part selection trusts a complete index — the replacement rows are
    /// found through it.
    #[test]
    fn updates_publish_the_replacement_superfile_into_the_index() {
        use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
        use datafusion::prelude::{col, lit};

        use crate::{Bm25SearchOptions, superfile::fts::reader::Bm25Stats, supertable::Supertable};

        let (_dir, storage, st) = table_with(|o| {
            o.with_eager_load_threshold(0)
                .with_target_superfiles_per_part(1)
        });
        for segment in 0..SEGMENTS {
            commit_segment(&st, segment);
        }
        assert!(
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_complete()
        );
        let replacement: ArrayRef = Arc::new(LargeStringArray::from(vec!["zeta shared s0d00"]));
        let batch = RecordBatch::try_new(title_schema(), vec![replacement]).expect("batch");
        let stats = st
            .update(col("title").eq(lit("alpha shared s0d00")), &batch)
            .expect("update");
        assert_eq!(stats.matched(), 1);

        let consumer =
            Supertable::open(fresh_options(&storage).with_eager_load_threshold(0)).expect("open");
        let reader = consumer.reader().expect("reader");
        let manifest = reader.manifest();
        assert!(
            manifest.term_index_complete(),
            "a replacement with postings keeps the index complete"
        );
        let index = reader
            .block_on(manifest.term_index())
            .expect("the index loads");
        let unindexed: Vec<Uuid> = reader
            .block_on(manifest.get_all_superfiles_loaded())
            .expect("entries")
            .iter()
            .map(|e| e.superfile_id)
            .filter(|id| !index.is_indexed(id))
            .collect();
        assert!(
            unindexed.is_empty(),
            "every live superfile is indexed: {unindexed:?}"
        );
        let batches = reader
            .bm25_search(
                "title",
                "zeta",
                10,
                Bm25SearchOptions::new().with_stats(Bm25Stats::PerSuperfile),
                Some(&["_id", "score"]),
            )
            .expect("search");
        assert_eq!(
            hits_of(&batches).len(),
            1,
            "the replacement row is found through the index"
        );
    }

    /// A commit that publishes a superfile without postings — one whose
    /// path built no contribution — leaves the index unable to route to
    /// it, so the index must stop claiming to list every live superfile.
    #[test]
    fn a_superfile_published_without_postings_marks_the_index_incomplete() {
        use crate::supertable::{
            manifest::{SuperfileUri, VectorLayout},
            writer::{CommitListMetadata, persist_commit_async},
        };

        let (_dir, storage, st) = fresh_table();
        commit_segment(&st, 0);
        assert!(
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_complete()
        );
        let entry = Arc::new(SuperfileEntry {
            stem: None,
            birth_version: 0,
            superfile_id: Uuid::new_v4(),
            uri: SuperfileUri::new_v4(),
            n_docs: 1,
            id_min: 1_000_000,
            id_max: 1_000_000,
            scalar_stats: Default::default(),
            fts_summary: Default::default(),
            vector_summary: Default::default(),
            partition_key: Vec::new(),
            partition_hint: None,
            vector_layout: VectorLayout::Ivf,
            subsection_offsets: None,
        });
        let committed = st
            .block_on_query(persist_commit_async(
                st.inner(),
                Arc::clone(&storage),
                vec![entry],
                &[],
                Vec::new(),
                Vec::new(),
                CommitListMetadata::empty(),
                Vec::new(),
            ))
            .expect("commit");
        assert!(
            committed.term_index_ref().is_some(),
            "the prior root is carried forward"
        );
        assert!(
            !committed.term_index_complete(),
            "an unindexed live superfile makes the index incomplete"
        );
    }

    /// Every ceiling the index computes is an upper bound on the score any
    /// document actually receives — for single terms, multi-term unions
    /// and phrases, under per-superfile statistics and under table-wide
    /// statistics, where the stored bound is rescaled from the superfile's
    /// own idf to the query's.
    #[test]
    fn query_ceilings_bound_real_scores_for_terms_and_phrases_under_both_stats() {
        use crate::{Bm25SearchOptions, superfile::fts::reader::Bm25Stats};

        let (_dir, storage, st) = fresh_table();
        for segment in 0..SEGMENTS {
            commit_segment(&st, segment);
        }
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let reader = st.reader().expect("reader");
        let entries = reader.manifest().get_all_superfiles().to_vec();
        let ranges: Vec<(Uuid, i128, i128)> = entries
            .iter()
            .map(|e| (e.superfile_id, e.id_min, e.id_max))
            .collect();
        let superfile_of = |id: i128| -> Uuid {
            ranges
                .iter()
                .find(|(_, lo, hi)| *lo <= id && id <= *hi)
                .map(|(sf, _, _)| *sf)
                .expect("every hit falls in one superfile's id range")
        };
        // Table-wide idf per term, as the query scores with under
        // `Bm25Stats::Global`: N is the scored-document total, df the sum
        // over every live superfile's posting.
        let scored_total: u64 = entries
            .iter()
            .map(|e| {
                e.fts_summary
                    .get("title")
                    .and_then(|s| s.length_stats.as_ref().map(|l| l.n_scored_docs))
                    .unwrap_or(e.n_docs)
            })
            .sum();
        let global_idf: HashMap<&str, f32> = ["alpha", "beta", "shared", "s1d02"]
            .into_iter()
            .map(|term| {
                let df: u64 = rt
                    .block_on(index.postings("title", term))
                    .expect("postings")
                    .iter()
                    .map(|p| p.df)
                    .sum();
                (term, bm25_idf(scored_total, df))
            })
            .collect();
        // (query, plain terms, phrases)
        let queries: [(&str, &[&str], &[&[&str]]); 5] = [
            ("alpha", &["alpha"], &[]),
            ("alpha shared", &["alpha", "shared"], &[]),
            ("\"alpha shared\"", &[], &[&["alpha", "shared"]]),
            (
                "shared \"alpha shared\"",
                &["shared"],
                &[&["alpha", "shared"]],
            ),
            ("beta \"shared s1d02\"", &["beta"], &[&["shared", "s1d02"]]),
        ];
        for stats in [Bm25Stats::PerSuperfile, Bm25Stats::Global] {
            for (query, terms, phrases) in &queries {
                let phrases: Vec<Vec<&str>> = phrases.iter().map(|p| p.to_vec()).collect();
                let idf_used = |term: &str, local: f32| match stats {
                    Bm25Stats::PerSuperfile => local,
                    Bm25Stats::Global => global_idf[term],
                };
                let ceilings = rt
                    .block_on(index.query_ceilings("title", terms, &phrases, &entries, &idf_used))
                    .expect("ceilings");
                let batches = reader
                    .bm25_search(
                        "title",
                        query,
                        DOCS_PER_SEGMENT * SEGMENTS,
                        Bm25SearchOptions::new().with_stats(stats),
                        Some(&["_id", "score"]),
                    )
                    .expect("search");
                let hits = hits_of(&batches);
                assert!(!hits.is_empty(), "{query}: the fixture has hits");
                for (id, score) in hits {
                    let sf = superfile_of(id);
                    let ceiling = ceilings[&sf];
                    assert!(
                        score <= ceiling,
                        "{query} under {stats:?} in {sf}: score {score} exceeds ceiling {ceiling}"
                    );
                    assert!(
                        ceiling.is_finite(),
                        "{query}: a real ceiling, not the placeholder"
                    );
                }
            }
        }
    }

    /// A phrase whose members all carry the placeholder bound is unbounded
    /// in that superfile, not absent: its ceiling is `+∞`, so the superfile
    /// is opened unconditionally rather than skipped.
    #[test]
    fn a_phrase_of_unbounded_members_has_an_unbounded_ceiling() {
        use crate::supertable::manifest::{SuperfileUri, VectorLayout};

        let dir = TempDir::new().expect("tempdir");
        // The `contribution` helper records the `+inf` placeholder bound.
        let c = contribution(&dir, 1, &[("title", "alpha", 5), ("title", "shared", 9)]);
        let built = build(&[c], &BuildPolicy::default()).expect("build");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs"));
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let reference = rt
            .block_on(write_built(storage.as_ref(), built))
            .expect("write");
        let index = rt
            .block_on(TermIndex::load(Arc::clone(&storage), None, &reference))
            .expect("load");
        let entry = Arc::new(SuperfileEntry {
            stem: None,
            birth_version: 0,
            superfile_id: Uuid::from_u128(1),
            uri: SuperfileUri::new_v4(),
            n_docs: 10,
            id_min: 0,
            id_max: 9,
            scalar_stats: Default::default(),
            fts_summary: Default::default(),
            vector_summary: Default::default(),
            partition_key: Vec::new(),
            partition_hint: None,
            vector_layout: VectorLayout::Ivf,
            subsection_offsets: None,
        });
        let entries = vec![entry];
        let local = |_: &str, idf: f32| idf;
        let ceilings = rt
            .block_on(index.query_ceilings(
                "title",
                &[],
                &[vec!["alpha", "shared"]],
                &entries,
                &local,
            ))
            .expect("ceilings");
        assert_eq!(ceilings[&Uuid::from_u128(1)], f32::INFINITY);
        let ceilings = rt
            .block_on(index.query_ceilings("title", &["alpha"], &[], &entries, &local))
            .expect("ceilings");
        assert_eq!(ceilings[&Uuid::from_u128(1)], f32::INFINITY);
    }

    /// A superfile whose best document ties the running k-th score is still
    /// opened: the stable `_id` order decides ties, so skipping it would
    /// return the wrong document. Two things keep it open, and this test
    /// exercises both together: the skip only fires for a ceiling strictly
    /// below the floor (`ceiling_can_compete`, pinned at exact equality by
    /// its own unit test), and the ceiling the query sees is widened by
    /// [`CEILING_SLACK`], so a real score that rounding put an ulp above
    /// its stored bound still cannot fall below it.
    #[test]
    fn a_superfile_whose_ceiling_ties_the_floor_is_still_opened() {
        use crate::{
            Bm25SearchOptions,
            runtime_metrics::op_stats::{self, with_op_stats},
        };

        let (_dir, _storage, st) = fresh_table_with_open_window(1);
        // Every title is three tokens, so length normalization is identical
        // everywhere and `alpha shared x` scores the same in both superfiles
        // under table-wide statistics. The second superfile's tf-2 title
        // gives it the higher ceiling, so it opens first; the first
        // superfile's ceiling then equals the k-th score exactly.
        commit_titles(&st, &["alpha shared x".to_owned()]);
        commit_titles(
            &st,
            &["alpha alpha x".to_owned(), "alpha shared x".to_owned()],
        );
        let run = |k: usize| -> (Vec<(i128, f32)>, u64) {
            with_op_stats(|| {
                let reader = st.reader().expect("reader");
                let batches = reader
                    .bm25_search(
                        "title",
                        "alpha",
                        k,
                        Bm25SearchOptions::new(),
                        Some(&["_id", "score"]),
                    )
                    .expect("search");
                let opened = op_stats::current().expect("metered").superfiles_opened();
                (hits_of(&batches), opened)
            })
            .0
        };
        let (all, _) = run(3);
        assert_eq!(all.len(), 3);
        assert_eq!(
            all[1].1, all[2].1,
            "the two `alpha shared x` titles tie on score"
        );
        assert!(all[1].0 < all[2].0, "ties resolve to the lower id");
        let (top2, opened) = run(2);
        assert_eq!(opened, 2, "the tying superfile is opened");
        assert_eq!(
            top2,
            all[..2].to_vec(),
            "and its lower-id document wins the tie"
        );
    }

    /// Ceiling-ordered opening changes only which superfiles are opened,
    /// never the answer: a multi-term query with a phrase returns the same
    /// `(id, score)` list at every window width, and the same as the
    /// unordered path (which a scoring override selects).
    #[test]
    fn ceiling_ordered_results_match_at_every_window_and_the_unordered_path() {
        use crate::{Bm25SearchOptions, superfile::fts::bm25::Bm25Params};

        let query = "alpha beta \"alpha shared\"";
        let k = 7;
        let mut results: Vec<Vec<(i128, f32)>> = Vec::new();
        for window in [1usize, 2, 64] {
            let (_dir, _storage, st) = fresh_table_with_open_window(window);
            for segment in 0..SEGMENTS {
                commit_segment(&st, segment);
            }
            let reader = st.reader().expect("reader");
            let ordered = reader
                .bm25_search(
                    "title",
                    query,
                    k,
                    Bm25SearchOptions::new(),
                    Some(&["_id", "score"]),
                )
                .expect("search");
            let defaults = Bm25Params::default();
            let unordered = reader
                .bm25_search(
                    "title",
                    query,
                    k,
                    Bm25SearchOptions::new().with_bm25(defaults.k1, defaults.b),
                    Some(&["_id", "score"]),
                )
                .expect("search");
            let ordered = hits_of(&ordered);
            assert_eq!(ordered.len(), k);
            assert_eq!(
                ordered,
                hits_of(&unordered),
                "window {window}: the ordered and unordered paths agree"
            );
            results.push(ordered);
        }
        // Ids are minted per table, so across tables the scores are what
        // must agree; within a table the `(id, score)` lists already did.
        let scores: Vec<Vec<f32>> = results
            .iter()
            .map(|r| r.iter().map(|(_, s)| *s).collect())
            .collect();
        assert!(
            scores.iter().all(|s| *s == scores[0]),
            "every window returns the same ranking: {results:?}"
        );
    }

    /// A memo that covers only some of a query's terms serves those and
    /// leaves the rest to the dictionary: the hits are identical to a run
    /// with no memo and to a run whose memo covers every term.
    #[test]
    fn a_partial_memo_falls_back_to_the_dictionary_for_the_terms_it_lacks() {
        use crate::superfile::{SuperfileReader, fts::reader::ClauseLists};

        let (dir, storage, st) = fresh_table();
        commit_segment(&st, 0);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let entries = st
            .reader()
            .expect("reader")
            .manifest()
            .get_all_superfiles()
            .to_vec();
        assert_eq!(entries.len(), 1);
        let bytes = std::fs::read(dir.path().join(entries[0].uri.storage_path())).expect("bytes");
        let sf = SuperfileReader::open(Bytes::from(bytes)).expect("open");
        let memo_for = |terms: &[&str]| {
            let by_sf = rt
                .block_on(index.locations("title", terms, &entries))
                .expect("locations");
            let pairs: Vec<(&str, u64, FstValue)> = by_sf[&entries[0].superfile_id]
                .iter()
                .filter_map(|(t, df, l)| l.to_dict_value().map(|v| (t.as_str(), *df, v)))
                .collect();
            assert_eq!(pairs.len(), terms.len(), "every asked term has a location");
            rt.block_on(sf.term_memo_from_dict_values(&pairs))
                .expect("memo")
        };
        let run = |memo: Option<&crate::superfile::fts::reader::FetchedTermMemo>| {
            let prep = rt
                .block_on(sf.prepare_clauses(
                    "title",
                    ClauseLists {
                        musts: &["alpha", "shared"],
                        shoulds: &[],
                        negatives: &[],
                        must_phrases: &[],
                        should_phrases: &[],
                        negative_phrases: &[],
                        global_idf: None,
                        prefetched: memo,
                        live_floor: None,
                        allow: None,
                    },
                    DOCS_PER_SEGMENT,
                    f32::NEG_INFINITY,
                    None,
                ))
                .expect("prepare");
            let mut hits = sf.run_prepared(prep, None).expect("run");
            hits.sort_by_key(|hit| hit.0);
            hits
        };
        let plain = run(None);
        assert!(!plain.is_empty());
        let partial = memo_for(&["shared"]);
        assert_eq!(
            run(Some(&partial)),
            plain,
            "a memo missing `alpha` still finds it"
        );
        let full = memo_for(&["alpha", "shared"]);
        assert_eq!(run(Some(&full)), plain, "a memo covering both terms agrees");
    }

    /// A lazily loaded handle reloads the index when its own commit
    /// publishes a new root: rows committed after the first query are
    /// found through the index, which lists the new superfile.
    #[test]
    fn a_lazy_handle_reloads_the_index_after_its_own_commit() {
        use crate::Bm25SearchOptions;

        let (_dir, _storage, st) = table_with(|o| {
            o.with_eager_load_threshold(0)
                .with_target_superfiles_per_part(1)
        });
        for segment in 0..SEGMENTS {
            commit_segment(&st, segment);
        }
        let search = |term: &str| -> usize {
            let reader = st.reader().expect("reader");
            let batches = reader
                .bm25_search(
                    "title",
                    term,
                    10,
                    Bm25SearchOptions::new(),
                    Some(&["_id", "score"]),
                )
                .expect("search");
            hits_of(&batches).len()
        };
        assert!(search("alpha") > 0, "the first query loads the index");
        let first_root = st
            .reader()
            .expect("reader")
            .manifest()
            .term_index_ref()
            .cloned()
            .expect("reference");
        commit_titles(&st, &["zeta shared extra".to_owned()]);
        assert_eq!(search("zeta"), 1, "the new row is found after the commit");
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let reference = manifest.term_index_ref().cloned().expect("reference");
        assert_ne!(reference, first_root, "the commit published a new root");
        let index = reader
            .block_on(manifest.term_index())
            .expect("the index loads");
        assert_eq!(
            index.root_uri(),
            reference.uri,
            "the loaded index is the new root's"
        );
        let newest = manifest
            .get_all_superfiles()
            .iter()
            .max_by_key(|e| e.id_min)
            .expect("entries")
            .superfile_id;
        assert!(index.is_indexed(&newest), "the new superfile is listed");
    }

    /// A slice fetched once is served from the manifest disk cache
    /// afterwards: with the object gone from storage, a fresh index over
    /// the same cache still answers, and one without the cache does not.
    #[test]
    fn slices_are_served_from_the_manifest_disk_cache_once_fetched() {
        use std::fs;

        use crate::supertable::manifest::disk_cache::ManifestDiskCache;

        let (dir, storage, st) = fresh_table();
        commit_segment(&st, 0);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let cache_dir = TempDir::new().expect("cache dir");
        let cache = ManifestDiskCache::new(cache_dir.path().to_path_buf(), 1 << 30).expect("cache");
        let warm = TermIndex::new(
            root.clone(),
            String::new(),
            Arc::clone(&storage),
            Some(Arc::clone(&cache)),
        );
        let first = rt
            .block_on(warm.postings("title", "shared"))
            .expect("fetch through storage");
        assert!(!first.is_empty());
        for slice in root.segments.iter().flat_map(|s| s.slices.iter()) {
            fs::remove_file(dir.path().join(slice_uri(&slice.content_hash))).expect("remove slice");
        }
        let cached = TermIndex::new(
            root.clone(),
            String::new(),
            Arc::clone(&storage),
            Some(cache),
        );
        let again = rt
            .block_on(cached.postings("title", "shared"))
            .expect("served from the disk cache");
        assert_eq!(*again, *first);
        let uncached = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        assert!(
            matches!(
                rt.block_on(uncached.postings("title", "shared")),
                Err(TermIndexError::Storage(_))
            ),
            "without the cache the missing object is a storage error"
        );
    }

    /// A commit against a table whose pointer is gone refuses before it
    /// writes any term-index object: the pointer fence runs first, so a
    /// purged table gains no orphans.
    #[test]
    fn a_purged_table_refuses_the_commit_before_writing_index_objects() {
        use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};

        use crate::supertable::manifest::commit::POINTER_PATH;

        let (dir, storage, st) = fresh_table();
        commit_segment(&st, 0);
        let objects = |dir: &TempDir| -> usize {
            std::fs::read_dir(dir.path().join(STORAGE_PREFIX))
                .expect("term-index dir")
                .count()
        };
        let before = objects(&dir);
        assert!(before >= 2, "a root and at least one slice");
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        rt.block_on(storage.delete(POINTER_PATH))
            .expect("delete pointer");
        let arr: ArrayRef = Arc::new(LargeStringArray::from(vec!["gamma shared late"]));
        let batch = RecordBatch::try_new(title_schema(), vec![arr]).expect("batch");
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        assert!(w.commit().is_err(), "the commit refuses");
        assert_eq!(objects(&dir), before, "no term-index object was written");
    }

    /// A prior root that cannot be read does not fail the commit: the
    /// index restarts from this commit's superfiles and is marked
    /// incomplete, and the next maintenance rebuild makes it whole again.
    #[test]
    fn an_unreadable_prior_root_restarts_the_index_incomplete() {
        let (dir, storage, st) = fresh_table();
        commit_segment(&st, 0);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (live_before, _) = live_and_covered(&st, &storage, &rt);
        let reference = st
            .reader()
            .expect("reader")
            .manifest()
            .term_index_ref()
            .cloned()
            .expect("reference");
        std::fs::remove_file(dir.path().join(&reference.uri)).expect("remove root");

        commit_segment(&st, 1);
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        assert!(
            !manifest.term_index_complete(),
            "the restarted index does not list the earlier superfile"
        );
        let (live, root) = live_and_covered(&st, &storage, &rt);
        let covered: std::collections::HashSet<Uuid> = root.superfiles.iter().copied().collect();
        assert!(
            covered.is_disjoint(&live_before),
            "the earlier superfile is not listed"
        );
        assert_eq!(
            covered.len(),
            live.len() - live_before.len(),
            "this commit's superfiles are"
        );

        stats_only_optimize(&st);
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        assert!(
            manifest.term_index_complete(),
            "a maintenance rebuild makes it whole"
        );
        let (live, root) = live_and_covered(&st, &storage, &rt);
        let covered: std::collections::HashSet<Uuid> = root.superfiles.iter().copied().collect();
        assert_eq!(covered, live);
    }

    /// GC with an unreadable root sweeps nothing: the live set cannot be
    /// derived, so the sweep errors rather than deleting slices it can no
    /// longer prove referenced.
    #[test]
    fn gc_refuses_to_sweep_when_the_root_is_unreadable() {
        use std::{fs, time::Duration};

        let (dir, storage, st, _) = optimized_fragmented_table();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let reference = st
            .reader()
            .expect("reader")
            .manifest()
            .term_index_ref()
            .cloned()
            .expect("reference");
        let orphan = dir
            .path()
            .join(slice_uri(&ContentHash::of(b"nothing references me")));
        fs::write(&orphan, b"stray slice bytes").expect("plant orphan");
        fs::remove_file(dir.path().join(&reference.uri)).expect("remove root");

        assert!(st.gc(Duration::ZERO).is_err(), "the sweep refuses");
        assert!(orphan.exists(), "nothing was deleted, the orphan included");
        for slice in root.segments.iter().flat_map(|s| s.slices.iter()) {
            assert!(dir.path().join(slice_uri(&slice.content_hash)).exists());
        }
    }

    /// A commit's delta publishes a new root and leaves the previous one
    /// unreferenced. GC keeps it while it is younger than the safety gap,
    /// then sweeps it; the slices the new root still names survive both.
    #[test]
    fn gc_sweeps_a_superseded_root_and_keeps_the_slices_the_new_root_names() {
        use std::time::Duration;

        let (dir, storage, st) = fresh_table();
        commit_segment(&st, 0);
        let first = st
            .reader()
            .expect("reader")
            .manifest()
            .term_index_ref()
            .cloned()
            .expect("first root");
        commit_segment(&st, 1);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let second = st
            .reader()
            .expect("reader")
            .manifest()
            .term_index_ref()
            .cloned()
            .expect("second root");
        assert_ne!(first, second);
        let first_path = dir.path().join(&first.uri);
        assert!(first_path.exists());

        st.gc(Duration::from_secs(3600)).expect("gc within the gap");
        assert!(first_path.exists(), "younger than the safety gap: kept");

        st.gc(Duration::ZERO).expect("gc");
        assert!(!first_path.exists(), "the superseded root is swept");
        assert!(dir.path().join(&second.uri).exists());
        for slice in root.segments.iter().flat_map(|s| s.slices.iter()) {
            assert!(
                dir.path().join(slice_uri(&slice.content_hash)).exists(),
                "a slice the current root names survives"
            );
        }
    }

    /// Optimize on an empty table publishes no index, and a repeat over an
    /// unchanged membership publishes nothing new — the same reference is
    /// found and no successor manifest is written.
    #[test]
    fn optimize_on_an_empty_table_publishes_no_index_and_a_repeat_is_a_no_op() {
        let (_dir, _storage, st) = fresh_table();
        stats_only_optimize(&st);
        assert!(
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_ref()
                .is_none(),
            "nothing to index"
        );
        commit_segment(&st, 0);
        stats_only_optimize(&st);
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let reference = manifest.term_index_ref().cloned().expect("reference");
        let id = manifest.get_manifest_id();
        stats_only_optimize(&st);
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        assert_eq!(manifest.term_index_ref(), Some(&reference));
        assert_eq!(
            manifest.get_manifest_id(),
            id,
            "no successor for an unchanged index"
        );
    }

    /// A slice with bytes past its declared sections is refused loudly,
    /// never read as a shorter but valid slice.
    #[test]
    fn a_slice_with_trailing_bytes_is_refused() {
        let dir = TempDir::new().expect("tempdir");
        let c = contribution(&dir, 1, &[("title", "alpha", 5)]);
        let built = build(&[c], &BuildPolicy::default()).expect("build");
        let mut bytes = built.slices[0].1.clone();
        assert!(Slice::open(&bytes).is_ok());
        bytes.push(0);
        assert!(
            matches!(Slice::open(&bytes), Err(TermIndexError::Malformed(m)) if m.contains("trailing")),
            "trailing bytes are malformed"
        );
    }

    /// An exact-match query resolves each superfile's terms from the
    /// index's postings locations: the docs are the ones the dictionary
    /// would give, and the table-level query plans exactly the reads the
    /// memo-fed superfile call plans — one fewer per superfile than a
    /// dictionary-first resolution.
    #[test]
    fn exact_match_reads_no_dictionary_when_the_index_knows_the_locations() {
        use crate::{
            runtime_metrics::op_stats::with_op_stats,
            superfile::{SuperfileReader, fts::reader::BoolMode},
        };

        let (dir, storage, st) = fresh_table();
        for segment in 0..SEGMENTS {
            commit_segment(&st, segment);
        }
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let reader = st.reader().expect("reader");
        let entries = reader.manifest().get_all_superfiles().to_vec();
        let terms = ["alpha", "shared"];
        let by_sf = rt
            .block_on(index.locations("title", &terms, &entries))
            .expect("locations");

        let mut planned_with_memo = 0u64;
        let mut docs_total = 0usize;
        for e in &entries {
            let bytes =
                std::fs::read(dir.path().join(e.uri.storage_path())).expect("superfile bytes");
            let sf = SuperfileReader::open(Bytes::from(bytes)).expect("open");
            let pairs: Vec<(&str, u64, FstValue)> = by_sf[&e.superfile_id]
                .iter()
                .filter_map(|(t, df, l)| l.to_dict_value().map(|v| (t.as_str(), *df, v)))
                .collect();
            let memo = rt
                .block_on(sf.term_memo_from_dict_values(&pairs))
                .expect("memo");
            let (plain, plain_work) = rt
                .block_on(sf.token_match("title", &terms, BoolMode::And))
                .expect("dictionary path");
            let (memoed, memo_work) = rt
                .block_on(sf.token_match_prefetched("title", &terms, BoolMode::And, Some(&memo)))
                .expect("memo path");
            assert_eq!(memoed, plain, "same docs either way");
            assert_eq!(
                memo_work.planned_ranges + 1,
                plain_work.planned_ranges,
                "the memo path plans one read fewer: the dictionary"
            );
            planned_with_memo += memo_work.planned_ranges;
            docs_total += plain.len();
        }

        let ((hits, planned), _) = with_op_stats(|| {
            let reader = st.reader().expect("reader");
            let hits = reader
                .token_match("title", "alpha shared", BoolMode::And)
                .expect("token_match");
            let planned = crate::runtime_metrics::op_stats::current()
                .expect("metered")
                .snapshot()
                .planned_read_ranges;
            (hits, planned)
        });
        assert_eq!(hits.len(), docs_total);
        assert_eq!(
            planned, planned_with_memo,
            "the table-level query resolved every superfile from the index's locations"
        );
    }

    /// A candidate plan — what a SQL `WHERE` on a text column and a filtered
    /// vector search resolve — reads no dictionary for the terms the index
    /// located: same rows as the dictionary path, one planned read fewer
    /// per superfile, and nothing changes for a column the memos lack.
    #[test]
    fn candidate_plans_resolve_from_the_index_locations() {
        use crate::{
            superfile::SuperfileReader,
            supertable::query::{
                candidate::{CandidatePlan, TermMemos},
                fts::{memos_from_plan_locations, plan_locations_for},
            },
        };

        let (dir, _storage, st) = fresh_table();
        for segment in 0..SEGMENTS {
            commit_segment(&st, segment);
        }
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let entries = st
            .reader()
            .expect("reader")
            .manifest()
            .get_all_superfiles()
            .to_vec();
        let plan = CandidatePlan::And(vec![
            CandidatePlan::TermsAll {
                column: "title".into(),
                tokens: vec!["alpha".into(), "shared".into()],
            },
            CandidatePlan::TermsAny {
                column: "title".into(),
                terms: vec!["s0d00".into(), "s1d02".into(), "absent".into()],
            },
        ]);
        let requests = plan.term_requests();
        assert_eq!(
            requests["title"],
            vec!["absent", "alpha", "s0d00", "s1d02", "shared"],
            "every exact-match term, once, per column"
        );
        // The real helpers: locations per column from the table's index,
        // then one memo per column per superfile — with the terms the index
        // shows absent from a superfile recorded as resolved misses.
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let locations = rt.block_on(plan_locations_for(manifest, &plan, &entries));
        assert_eq!(locations.len(), 1, "one column");
        let mut total_rows = 0u64;
        for e in &entries {
            let bytes =
                std::fs::read(dir.path().join(e.uri.storage_path())).expect("superfile bytes");
            let sf = SuperfileReader::open(Bytes::from(bytes)).expect("open");
            let memos = rt.block_on(memos_from_plan_locations(&sf, &locations, e.superfile_id));
            assert!(
                memos.contains_key("title"),
                "an indexed superfile gets a memo"
            );
            let (plain, plain_work) = rt
                .block_on(plan.evaluate(&sf, None, &TermMemos::new()))
                .expect("dictionary path");
            let (memoed, memo_work) = rt
                .block_on(plan.evaluate(&sf, None, &memos))
                .expect("memo path");
            assert_eq!(memoed, plain, "same rows either way");
            // Two exact-match nodes, each spared its dictionary read.
            assert_eq!(memo_work.planned_ranges + 2, plain_work.planned_ranges);
            total_rows += plain.map_or(0, |b| b.len());
        }
        assert!(total_rows > 0, "the fixture matches the plan");
    }

    /// Routing is exact: `Or` is the union of the terms' posting sets and
    /// `And` their intersection, term by term, on random corpora.
    #[test]
    fn routing_is_exact_on_random_corpora() {
        use proptest::prelude::*;

        let alphabet: Vec<String> = (0..12).map(|i| format!("t{i:02}")).collect();
        proptest!(ProptestConfig::with_cases(64), |(
            corpus in prop::collection::vec(prop::collection::btree_set(0usize..12, 0..8), 1..6),
        )| {
            let dir = TempDir::new().expect("tempdir");
            let contributions: Vec<Contribution> = corpus
                .iter()
                .enumerate()
                .map(|(i, terms)| {
                    let mut w = ContributionWriter::create(dir.path(), Uuid::from_u128(i as u128 + 1), i as i128).expect("create");
                    for t in terms {
                        w.push(&make_key("body", &alphabet[*t]), 1 + *t as u64, f32::INFINITY, Location::None).expect("push");
                    }
                    w.finish().expect("finish")
                })
                .collect();
            let policy = BuildPolicy { slice_target_bytes: 512, range_max_superfiles: 3 };
            let built = build(&contributions, &policy).expect("build");
            let store_dir = TempDir::new().expect("store dir");
            let storage: Arc<dyn StorageProvider> =
                Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local fs"));
            let rt = tokio::runtime::Runtime::new().expect("runtime");
            let reference = rt.block_on(write_built(storage.as_ref(), built)).expect("write");
            let index = rt.block_on(TermIndex::load(Arc::clone(&storage), None, &reference)).expect("load");
            let holders = |t: usize| -> HashSet<Uuid> {
                corpus.iter().enumerate().filter(|(_, s)| s.contains(&t)).map(|(i, _)| Uuid::from_u128(i as u128 + 1)).collect()
            };
            for a in 0..12 {
                let single = rt.block_on(index.route("body", &[&alphabet[a]], BoolMode::Or)).expect("route");
                prop_assert_eq!(&single, &holders(a));
                for b in 0..12 {
                    let pair = [alphabet[a].as_str(), alphabet[b].as_str()];
                    let or = rt.block_on(index.route("body", &pair, BoolMode::Or)).expect("route");
                    let and = rt.block_on(index.route("body", &pair, BoolMode::And)).expect("route");
                    prop_assert_eq!(&or, &holders(a).union(&holders(b)).copied().collect::<HashSet<_>>());
                    prop_assert_eq!(&and, &holders(a).intersection(&holders(b)).copied().collect::<HashSet<_>>());
                }
            }
            let prefixed = rt.block_on(index.route_prefix("body", "t0")).expect("prefix");
            let expect: HashSet<Uuid> = (0..10).flat_map(holders).collect();
            prop_assert_eq!(prefixed, expect);
            prop_assert!(rt.block_on(index.route("body", &["nope"], BoolMode::Or)).expect("route").is_empty());
        });
    }

    /// Through the query path: superfile selection for a term is exactly
    /// the set of live superfiles whose dictionary holds it — no more (the
    /// bloom's false positives are gone) and no less.
    #[test]
    fn selection_routes_exactly_through_the_index() {
        use crate::supertable::query::prune::{PruneLeaf, select_superfiles};

        let (_dir, _storage, st) = fresh_table();
        for segment in 0..SEGMENTS {
            commit_segment(&st, segment);
        }
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        assert!(
            rt.block_on(manifest.term_index()).is_some(),
            "the snapshot exposes its index"
        );
        let select = |terms: &[&str], mode: BoolMode| -> HashSet<Uuid> {
            let leaf = PruneLeaf::TermPresence {
                column: "title".to_owned(),
                terms: terms.iter().map(|t| (*t).to_owned()).collect(),
                mode,
            };
            rt.block_on(select_superfiles(
                manifest.as_ref(),
                std::slice::from_ref(&leaf),
            ))
            .expect("select")
            .iter()
            .map(|e| e.superfile_id)
            .collect()
        };
        let live: HashSet<Uuid> = manifest
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect();
        // Every title holds `shared` and `alpha` is in every segment.
        assert_eq!(select(&["shared"], BoolMode::Or), live);
        assert_eq!(select(&["alpha", "shared"], BoolMode::And), live);
        // A term in no superfile: exact routing returns nothing, where a
        // summary could only say "maybe".
        assert!(select(&["absent"], BoolMode::Or).is_empty());
        assert!(select(&["absent", "shared"], BoolMode::And).is_empty());
        assert_eq!(select(&["absent", "shared"], BoolMode::Or), live);
        // A segment-specific token (`s1d00` is only in segment 1's titles).
        let s1: HashSet<Uuid> = select(&["s1d00"], BoolMode::Or);
        assert_eq!(s1.len(), 1, "one superfile holds the token");
        let prefix = PruneLeaf::Prefix {
            column: "title".to_owned(),
            prefix: b"s1d".to_vec(),
        };
        let by_prefix: HashSet<Uuid> = rt
            .block_on(select_superfiles(
                manifest.as_ref(),
                std::slice::from_ref(&prefix),
            ))
            .expect("select")
            .iter()
            .map(|e| e.superfile_id)
            .collect();
        assert_eq!(
            by_prefix, s1,
            "a prefix routes through the slices to the same superfile"
        );
    }
    /// Every posting's bound is a true ceiling: for each term, the highest
    /// score any document in that superfile actually receives under the
    /// superfile's own statistics does not exceed the artifact's bound for
    /// it. The oracle is the public search itself, run with per-superfile
    /// statistics so its scores are in the scale the bounds were baked in;
    /// hits map to superfiles through the entries' id ranges.
    #[test]
    fn bounds_are_upper_bounds_on_real_scores() {
        use arrow_array::{Array, Decimal128Array, Float32Array, Int64Array};

        use crate::{Bm25SearchOptions, superfile::fts::reader::Bm25Stats};

        let (_dir, storage, st) = fresh_table();
        for segment in 0..SEGMENTS {
            commit_segment(&st, segment);
        }
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let reader = st.reader().expect("reader");
        let ranges: Vec<(Uuid, i128, i128)> = reader
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|e| (e.superfile_id, e.id_min, e.id_max))
            .collect();
        let superfile_of = |id: i128| -> Uuid {
            ranges
                .iter()
                .find(|(_, lo, hi)| *lo <= id && id <= *hi)
                .map(|(sf, _, _)| *sf)
                .expect("every hit falls in one superfile's id range")
        };
        for term in ["shared", "alpha", "beta", "s1d00", "s2d04"] {
            let postings = rt.block_on(index.postings("title", term)).expect("lookup");
            assert!(!postings.is_empty(), "{term} is indexed");
            let bounds: HashMap<Uuid, f32> = postings
                .iter()
                .map(|p| (index.superfile_id(p.superfile).expect("ordinal"), p.bound))
                .collect();
            for (sf, b) in &bounds {
                assert!(
                    b.is_finite(),
                    "{term} in {sf}: bound is a real ceiling, not the +inf placeholder"
                );
            }
            let batches = reader
                .bm25_search(
                    "title",
                    term,
                    DOCS_PER_SEGMENT * SEGMENTS,
                    Bm25SearchOptions::new().with_stats(Bm25Stats::PerSuperfile),
                    Some(&["_id", "score"]),
                )
                .expect("search");
            let mut observed_max: HashMap<Uuid, f32> = HashMap::new();
            for b in &batches {
                let scores = b
                    .column(1)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("score");
                let ids = b.column(0);
                for i in 0..b.num_rows() {
                    let id: i128 = if let Some(a) = ids.as_any().downcast_ref::<Decimal128Array>() {
                        a.value(i)
                    } else {
                        ids.as_any()
                            .downcast_ref::<Int64Array>()
                            .expect("_id")
                            .value(i) as i128
                    };
                    let sf = superfile_of(id);
                    let e = observed_max.entry(sf).or_insert(0.0);
                    *e = e.max(scores.value(i));
                }
            }
            assert!(!observed_max.is_empty());
            for (sf, observed) in observed_max {
                let bound = bounds
                    .get(&sf)
                    .copied()
                    .unwrap_or_else(|| panic!("{term}: a superfile with hits has a posting"));
                assert!(
                    observed <= bound,
                    "{term} in {sf}: observed max {observed} exceeds bound {bound}"
                );
            }
        }
    }
    /// Like [`fresh_table`] with a ceiling-ordered open window of `window`
    /// superfiles (1 = strictly sequential, so skipping is observable).
    fn fresh_table_with_open_window(
        window: usize,
    ) -> (
        TempDir,
        Arc<dyn StorageProvider>,
        crate::supertable::Supertable,
    ) {
        table_with(|o| o.with_bound_ordered_open_window(window))
    }

    fn commit_titles(st: &crate::supertable::Supertable, titles: &[String]) {
        use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
        let arr: ArrayRef = Arc::new(LargeStringArray::from(
            titles.iter().map(String::as_str).collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(title_schema(), vec![arr]).expect("batch");
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }

    /// `(_id, score)` per hit, in result order.
    fn hits_of(batches: &[arrow_array::RecordBatch]) -> Vec<(i128, f32)> {
        use arrow_array::{Array, Decimal128Array, Float32Array, Int64Array};
        let mut out = Vec::new();
        for b in batches {
            let scores = b
                .column(1)
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("score");
            let ids = b.column(0);
            for i in 0..b.num_rows() {
                let id: i128 = if let Some(a) = ids.as_any().downcast_ref::<Decimal128Array>() {
                    a.value(i)
                } else {
                    ids.as_any()
                        .downcast_ref::<Int64Array>()
                        .expect("_id")
                        .value(i) as i128
                };
                out.push((id, scores.value(i)));
            }
        }
        out
    }

    /// With superfiles opened in ceiling order and a strictly sequential
    /// window, a top-k query opens only the superfiles that can still place
    /// a document: the first open sets the floor, and a superfile whose
    /// ceiling is below it is never opened. Results are identical to the
    /// full walk (the same query with k large enough that nothing is
    /// skipped), and opening every superfile is what a large k still does.
    #[test]
    fn bound_ordered_opening_skips_superfiles_that_cannot_compete() {
        use crate::{
            Bm25SearchOptions,
            runtime_metrics::op_stats::{self, with_op_stats},
            superfile::fts::reader::Bm25Stats,
        };

        let (_dir, _storage, st) = fresh_table_with_open_window(1);
        // Segment 0 carries `alpha` three times per title — a clearly higher
        // ceiling than the single occurrence in segments 1 and 2.
        for segment in 0..3 {
            let word = if segment == 0 {
                "alpha alpha alpha"
            } else {
                "alpha"
            };
            let titles: Vec<String> = (0..DOCS_PER_SEGMENT)
                .map(|i| format!("{word} shared s{segment}d{i:02}"))
                .collect();
            commit_titles(&st, &titles);
        }
        let run = |k: usize| -> (Vec<(i128, f32)>, u64) {
            with_op_stats(|| {
                let reader = st.reader().expect("reader");
                let batches = reader
                    .bm25_search(
                        "title",
                        "alpha",
                        k,
                        Bm25SearchOptions::new().with_stats(Bm25Stats::PerSuperfile),
                        Some(&["_id", "score"]),
                    )
                    .expect("search");
                let opened = op_stats::current().expect("metered").superfiles_opened();
                (hits_of(&batches), opened)
            })
            .0
        };
        let (all, opened_all) = run(3 * DOCS_PER_SEGMENT);
        assert_eq!(
            opened_all, 3,
            "a k that needs every document opens every superfile"
        );
        assert_eq!(all.len(), 3 * DOCS_PER_SEGMENT);

        let (top1, opened_1) = run(1);
        assert_eq!(top1, all[..1].to_vec(), "identical to the full walk");
        assert_eq!(
            opened_1, 1,
            "the highest-ceiling superfile alone decides the top 1"
        );

        let (top5, opened_5) = run(5);
        assert_eq!(top5, all[..5].to_vec());
        assert_eq!(
            opened_5, 1,
            "five hits all sit in the tf-3 superfile; the others' ceilings stay below the floor"
        );
    }
    /// A memo built from the index's locations resolves exactly what the
    /// superfile's own dictionary would: same `df`, same form, same postings
    /// bytes — with the dictionary never consulted.
    #[test]
    fn memo_from_index_locations_matches_the_dictionary() {
        use crate::superfile::{SuperfileReader, fts::reader::FetchedTermSlot};

        let (dir, storage, st) = fresh_table();
        for segment in 0..SEGMENTS {
            commit_segment(&st, segment);
        }
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let reader = st.reader().expect("reader");
        let entries = reader.manifest().get_all_superfiles().to_vec();
        let terms = ["shared", "alpha", "beta", "s1d00", "absent"];
        let by_sf = rt
            .block_on(index.locations("title", &terms, &entries))
            .expect("locations");
        assert_eq!(
            by_sf.len(),
            entries.len(),
            "every indexed superfile has locations"
        );
        for e in &entries {
            let bytes =
                std::fs::read(dir.path().join(e.uri.storage_path())).expect("superfile bytes");
            let sf = SuperfileReader::open(Bytes::from(bytes)).expect("open");
            let locs = &by_sf[&e.superfile_id];
            assert!(
                locs.iter().all(|(t, _, _)| t != "absent"),
                "an absent term has no location"
            );
            let pairs: Vec<(&str, u64, FstValue)> = locs
                .iter()
                .filter_map(|(t, df, l)| l.to_dict_value().map(|v| (t.as_str(), *df, v)))
                .collect();
            let memo = rt
                .block_on(sf.term_memo_from_dict_values(&pairs))
                .expect("memo");
            let names: Vec<&str> = pairs.iter().map(|(t, _, _)| *t).collect();
            let (dfs, _) = rt.block_on(sf.term_dfs("title", &names)).expect("dfs");
            let facts = rt
                .block_on(sf.term_index_facts("title", &names))
                .expect("facts");
            for ((name, df), fact) in names.iter().zip(dfs).zip(facts) {
                let fact = fact.expect("dictionary has it");
                assert_eq!(fact.df, df, "{name}: the cursor's df is the header's df");
                assert_eq!(
                    memo.df(name),
                    df,
                    "{name}: df from the index equals the dictionary's"
                );
                let slot = memo.lookup(name).expect("in memo").expect("present");
                match (slot, fact.entry) {
                    (
                        FetchedTermSlot::Inline { doc_id, tf },
                        FstValue::Inline { doc_id: d, tf: t },
                    ) => {
                        assert_eq!((doc_id, tf), (d, t));
                    }
                    (
                        FetchedTermSlot::Pfor { bytes, short, .. },
                        FstValue::Pfor {
                            postings_length_hint,
                            short: s,
                            ..
                        },
                    ) => {
                        assert_eq!(short, s);
                        assert_eq!(
                            Some(bytes.len() as u32),
                            postings_length_hint,
                            "{name}: fetched exactly the postings range"
                        );
                    }
                    (_, value) => panic!("{name}: form mismatch against {value:?}"),
                }
            }
        }
    }
    /// A superfile committed with the term index carries no per-superfile
    /// term bloom, and the list carries no per-part union of them; term and
    /// prefix selection is still exact through the index, and with the index
    /// unavailable the absent bloom keeps every superfile — conservative,
    /// never wrong.
    #[test]
    fn committed_superfiles_carry_no_term_bloom_and_still_route_exactly() {
        use crate::supertable::query::{
            prune::{PruneLeaf, select_superfiles},
            skip::fts_bloom_skip,
        };

        let (_dir, _storage, st) = fresh_table();
        for segment in 0..SEGMENTS {
            commit_segment(&st, segment);
        }
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let entries = manifest.get_all_superfiles();
        for e in entries {
            let summary = e.fts_summary.get("title").expect("summary");
            assert!(
                summary.term_bloom.is_none(),
                "no per-superfile bloom is written"
            );
            assert!(
                summary.term_range.is_some(),
                "the term range is still recorded"
            );
            assert!(
                summary.length_stats.is_some(),
                "scoring statistics are still recorded"
            );
        }
        for part in manifest.get_all_list_entries() {
            if let Some(agg) = part.fts_summary_agg.get("title") {
                assert!(agg.term_bloom.is_none(), "no per-part union bloom either");
            }
        }
        // The manifest-summary answer alone keeps everything (no information).
        let all_kept = fts_bloom_skip(entries, "title", &["absent"], BoolMode::Or);
        assert!(all_kept.iter().all(|k| *k));
        // The index makes it exact.
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let leaf = PruneLeaf::TermPresence {
            column: "title".to_owned(),
            terms: vec!["absent".to_owned()],
            mode: BoolMode::Or,
        };
        assert!(
            rt.block_on(select_superfiles(
                manifest.as_ref(),
                std::slice::from_ref(&leaf)
            ))
            .expect("select")
            .is_empty()
        );
    }
    /// A committed superfile's open blob carries the parquet tail plus the
    /// two tiny FTS ranges a cold open reads — the header and the
    /// doc-lengths directory — and never the dictionary or the length
    /// arrays, which only a query that needs them reads.
    #[test]
    fn open_blob_inlines_the_fts_header_and_directory_but_not_the_dictionary() {
        let (_dir, _storage, st) = fresh_table();
        commit_segment(&st, 0);
        let reader = st.reader().expect("reader");
        for e in reader.manifest().get_all_superfiles() {
            let offsets = e.subsection_offsets.as_ref().expect("offsets recorded");
            assert_eq!(
                offsets.fts_open_ranges.len(),
                2,
                "the header and the directory: {:?}",
                offsets.fts_open_ranges
            );
            for &(off, len) in &offsets.fts_open_ranges {
                assert!(
                    len < 1024,
                    "an FTS open range is a few hundred bytes, never a dictionary: {len}"
                );
                assert!(
                    offsets
                        .open_blob
                        .iter()
                        .any(|(b_off, b)| *b_off == off && b.len() as u64 == len),
                    "each FTS open range is inlined"
                );
            }
            assert_eq!(
                offsets.open_blob.len(),
                1 + offsets.fts_open_ranges.len(),
                "the parquet tail plus the FTS open ranges"
            );
        }
    }
    /// The old format keeps working under the new reader, and mixes with
    /// the new format without changing an answer. The fixture is a table
    /// written by the engine before the term index existed: blooms in its
    /// parts, bloom unions in its list, a term-stats sidecar, no index.
    ///
    /// Four states of the same rows must answer every query identically:
    /// the fixture as written (blooms route); the fixture after the current
    /// writer appends a segment (index present but incomplete, so parts
    /// still route by summaries and only the new superfile is indexed);
    /// the fixture after a maintenance rebuild (index complete, blooms
    /// ignored); and a fresh table holding the same rows written entirely
    /// by the current writer.
    /// Recursive copy of a checked-in fixture into a scratch directory.
    fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
        use std::fs;
        fs::create_dir_all(to).expect("mkdir");
        for entry in fs::read_dir(from).expect("read_dir") {
            let entry = entry.expect("entry");
            let dest = to.join(entry.file_name());
            if entry.file_type().expect("type").is_dir() {
                copy_dir(&entry.path(), &dest);
            } else {
                fs::copy(entry.path(), dest).expect("copy");
            }
        }
    }

    /// Open a table written by the engine before the term index existed
    /// (the fixture's schema: one FTS column, no positions), with its options
    /// adjusted by `customize`.
    fn open_old_format(
        dir: &std::path::Path,
        customize: impl FnOnce(
            crate::supertable::SupertableOptions,
        ) -> crate::supertable::SupertableOptions,
    ) -> (Arc<dyn StorageProvider>, crate::supertable::Supertable) {
        use crate::{
            superfile::builder::FtsConfig,
            supertable::{Supertable, SupertableOptions},
        };
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir).expect("local fs"));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool"),
        );
        let options = customize(
            SupertableOptions::new(title_schema(), vec![FtsConfig::new("title")], Vec::new())
                .expect("options")
                .with_writer_pool(pool)
                .with_storage(Arc::clone(&storage)),
        );
        (storage, Supertable::open(options).expect("open"))
    }

    /// An upgraded table whose manifest is loaded lazily: the parts that
    /// mix superfiles written before the index (bloom-bearing) and after it
    /// (bloom-less) must keep no part-level bloom, or the part prune would
    /// treat the old superfiles' bloom as authoritative and drop terms that
    /// live only in the new superfiles.
    #[test]
    fn upgraded_lazy_tables_find_terms_that_only_new_superfiles_hold() {
        use std::path::Path;

        use crate::Bm25SearchOptions;

        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/old_format_fts_table");
        let dir = TempDir::new().expect("tempdir");
        copy_dir(&fixture, dir.path());
        let (_storage, st) = open_old_format(dir.path(), |o| o.with_eager_load_threshold(0));
        for segment in 3..6 {
            commit_segment(&st, segment);
        }
        // Reopen lazily so part selection is what finds the rows.
        let (_storage, consumer) = open_old_format(dir.path(), |o| o.with_eager_load_threshold(0));
        let reader = consumer.reader().expect("reader");
        for token in ["s3d00", "s4d01", "s5d02"] {
            let batches = reader
                .bm25_search(
                    "title",
                    token,
                    10,
                    Bm25SearchOptions::new(),
                    Some(&["_id", "score"]),
                )
                .expect("search");
            assert_eq!(
                hits_of(&batches).len(),
                1,
                "{token} lives only in a superfile written after the index and must be found"
            );
        }
    }

    /// A read of the prior root that fails for a reason other than the
    /// object being gone or corrupt — a transient storage error — must not
    /// restart the index: the commit keeps the reference it had, marks the
    /// index incomplete (this commit's superfiles are not listed), and the
    /// next maintenance rebuild makes it whole. Only an absent or corrupt
    /// root restarts it.
    #[test]
    fn a_transient_read_of_the_prior_root_keeps_the_index_and_marks_it_incomplete() {
        use crate::test_helpers::fault_storage::{FaultOp, FaultStorage};

        let dir = TempDir::new().expect("tempdir");
        let inner: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs"));
        let faults = FaultStorage::wrap(inner);
        let storage: Arc<dyn StorageProvider> = Arc::clone(&faults) as Arc<dyn StorageProvider>;
        let st = crate::supertable::Supertable::create(fresh_options(&storage)).expect("create");
        commit_segment(&st, 0);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (live_before, root_before) = live_and_covered(&st, &storage, &rt);
        let first = st
            .reader()
            .expect("reader")
            .manifest()
            .term_index_ref()
            .cloned()
            .expect("reference");

        faults.fail(FaultOp::Get, "term-index/root-", 1);
        commit_segment(&st, 1);
        assert_eq!(faults.fired(), 1, "the prior-root read hit the fault");
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        assert_eq!(
            manifest.term_index_ref(),
            Some(&first),
            "the reference is carried forward, not restarted"
        );
        assert!(
            !manifest.term_index_complete(),
            "this commit's superfiles are not listed"
        );
        let (_, root_after) = live_and_covered(&st, &storage, &rt);
        assert_eq!(
            root_after, root_before,
            "the accumulated segments are intact"
        );
        let covered: std::collections::HashSet<Uuid> =
            root_after.superfiles.iter().copied().collect();
        assert_eq!(covered, live_before);

        stats_only_optimize(&st);
        let reader = st.reader().expect("reader");
        assert!(
            reader.manifest().term_index_complete(),
            "maintenance makes it whole"
        );
        let (live, root) = live_and_covered(&st, &storage, &rt);
        let covered: std::collections::HashSet<Uuid> = root.superfiles.iter().copied().collect();
        assert_eq!(covered, live);
    }

    #[test]
    fn old_format_tables_read_and_mix_with_the_new_format() {
        use std::path::Path;

        use crate::{Bm25SearchOptions, superfile::fts::reader::Bm25Stats, supertable::Supertable};

        fn open(dir: &Path) -> (Arc<dyn StorageProvider>, Supertable) {
            open_old_format(dir, |o| o)
        }
        let queries: [(&str, BoolMode); 6] = [
            ("shared", BoolMode::Or),
            ("alpha", BoolMode::Or),
            ("alpha shared", BoolMode::And),
            ("beta s1d00", BoolMode::Or),
            ("s2d04", BoolMode::Or),
            ("absent", BoolMode::Or),
        ];
        // Rows are matched by title, not `_id`: ids are minted at write time,
        // so the fresh table's differ from the fixture's by construction.
        fn titled_hits(batches: &[arrow_array::RecordBatch]) -> Vec<(String, f32)> {
            use arrow_array::{Array, Float32Array, LargeStringArray};
            let mut out = Vec::new();
            for b in batches {
                let titles = b
                    .column(0)
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("title");
                let scores = b
                    .column(1)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("score");
                for i in 0..b.num_rows() {
                    out.push((titles.value(i).to_owned(), scores.value(i)));
                }
            }
            out
        }
        let answers = |st: &Supertable| -> Vec<Vec<(String, f32)>> {
            let reader = st.reader().expect("reader");
            queries
                .iter()
                .map(|(q, mode)| {
                    let batches = reader
                        .bm25_search(
                            "title",
                            q,
                            DOCS_PER_SEGMENT * (SEGMENTS + 1),
                            Bm25SearchOptions::new()
                                .with_mode(*mode)
                                .with_stats(Bm25Stats::Global),
                            Some(&["title", "score"]),
                        )
                        .expect("search");
                    let mut hits = titled_hits(&batches);
                    hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    hits
                })
                .collect()
        };

        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/old_format_fts_table");
        let work = TempDir::new().expect("tempdir");
        copy_dir(&fixture, work.path());

        // Cell 1: the old table as written. Blooms present, no index.
        let (_storage, st) = open(work.path());
        {
            let reader = st.reader().expect("reader");
            let manifest = reader.manifest();
            assert_eq!(manifest.get_all_superfiles().len(), SEGMENTS);
            assert!(
                manifest.term_index_ref().is_none(),
                "the fixture predates the index"
            );
            assert!(
                manifest.term_stats_blob().is_some(),
                "the fixture carries the term-stats sidecar"
            );
            for e in manifest.get_all_superfiles() {
                assert!(
                    e.fts_summary["title"].term_bloom.is_some(),
                    "old entries carry blooms"
                );
            }
        }
        let old_answers = answers(&st);
        assert!(
            old_answers[0].len() == SEGMENTS * DOCS_PER_SEGMENT,
            "`shared` hits every row"
        );
        assert!(old_answers[5].is_empty());

        // Cell 2: the current writer appends a segment. An index appears,
        // covering only the new superfile; the list marks it incomplete;
        // the old superfiles keep routing by their blooms.
        commit_segment(&st, SEGMENTS);
        {
            let reader = st.reader().expect("reader");
            let manifest = reader.manifest();
            assert_eq!(manifest.get_all_superfiles().len(), SEGMENTS + 1);
            assert!(
                manifest.term_index_ref().is_some(),
                "the commit published an index"
            );
            assert!(
                !manifest.term_index_complete(),
                "an upgraded table's first index is incomplete"
            );
            let rt = tokio::runtime::Runtime::new().expect("runtime");
            let index = rt.block_on(manifest.term_index()).expect("index loads");
            let indexed = manifest
                .get_all_superfiles()
                .iter()
                .filter(|e| index.is_indexed(&e.superfile_id))
                .count();
            assert_eq!(indexed, 1, "only the new superfile is indexed");
        }
        let mixed_answers = answers(&st);

        // Cell 3: a maintenance rebuild covers everything and flips the flag.
        stats_only_optimize(&st);
        {
            let manifest = st.reader().expect("reader").manifest().clone();
            assert!(
                manifest.term_index_complete(),
                "the rebuild lists every live superfile"
            );
        }
        let rebuilt_answers = answers(&st);

        // Cell 4: the same rows written entirely by the current writer.
        let (_d, _s, fresh) = fresh_table();
        for segment in 0..=SEGMENTS {
            commit_segment(&fresh, segment);
        }
        let fresh_answers = answers(&fresh);

        // The old three segments answer identically in every state; the
        // four-segment states answer identically to each other and to the
        // fresh table. Scores compare bitwise: the same rows, the same
        // global statistics, the same arithmetic.
        for (i, (q, _)) in queries.iter().enumerate() {
            assert_eq!(
                mixed_answers[i], rebuilt_answers[i],
                "{q}: mixed vs rebuilt"
            );
            assert_eq!(
                rebuilt_answers[i], fresh_answers[i],
                "{q}: rebuilt vs fresh"
            );
            let old_titles: Vec<&str> = old_answers[i].iter().map(|(t, _)| t.as_str()).collect();
            let surviving = mixed_answers[i]
                .iter()
                .filter(|(t, _)| old_titles.contains(&t.as_str()))
                .count();
            assert_eq!(
                surviving,
                old_titles.len(),
                "{q}: every old hit survives the append"
            );
        }
    }
    /// The resident set evicts least recently used against its budget —
    /// bytes for slices, a count for runs — and a read refreshes recency,
    /// so a burst that cycles through more than fits keeps what it keeps
    /// touching. Re-inserting a resident key is a no-op.
    #[test]
    fn resident_sets_evict_least_recently_used_within_budget() {
        let h = |n: u8| ContentHash([n; 32]);
        // Two of these fit the byte budget; three do not.
        let big = RESIDENT_SLICE_BUDGET_BYTES / 3 + 1;
        let mut slices: Resident<ContentHash, Bytes> =
            Resident::new(RESIDENT_SLICE_BUDGET_BYTES, Bytes::len);
        slices.insert(h(1), Bytes::from(vec![0u8; big]));
        slices.insert(h(2), Bytes::from(vec![0u8; big]));
        assert!(
            slices.get(&h(1)).is_some(),
            "touch 1: it is now most recent"
        );
        slices.insert(h(3), Bytes::from(vec![0u8; big]));
        assert!(
            slices.get(&h(2)).is_none(),
            "2 was least recently used and went first"
        );
        assert!(slices.get(&h(1)).is_some(), "1 was refreshed and survives");
        assert!(slices.get(&h(3)).is_some());
        assert!(slices.total <= RESIDENT_SLICE_BUDGET_BYTES);
        slices.insert(h(3), Bytes::from(vec![0u8; big]));
        assert_eq!(
            slices.total,
            2 * big,
            "re-inserting a resident slice is a no-op"
        );

        // Counted: the map never exceeds the bound, the oldest untouched
        // key goes first, and a fresh key is always admitted.
        let bound = 4;
        let mut runs: Resident<Vec<u8>, Arc<Vec<Posting>>> = Resident::new(bound, |_| 1);
        for n in 0..bound {
            runs.insert(vec![n as u8], Arc::new(Vec::new()));
        }
        assert!(runs.get(&vec![0u8]).is_some(), "touch 0");
        runs.insert(vec![bound as u8], Arc::new(Vec::new()));
        assert_eq!(runs.map.len(), bound, "the bound holds");
        assert!(
            runs.get(&vec![1u8]).is_none(),
            "1 was the least recently used"
        );
        assert!(runs.get(&vec![0u8]).is_some(), "0 was refreshed");
        assert!(
            runs.get(&vec![bound as u8]).is_some(),
            "the newest key is resident"
        );
        runs.insert(
            vec![0u8],
            Arc::new(vec![Posting {
                superfile: 0,
                df: 1,
                bound: 1.0,
                location: Location::None,
            }]),
        );
        assert!(
            runs.get(&vec![0u8]).expect("resident").is_empty(),
            "re-inserting a resident key keeps the first value"
        );
    }
    /// A term's decoded run is served from the resident set on later asks:
    /// the same allocation comes back, so the several consultations a
    /// ranked query makes cost one slice open, not four.
    #[test]
    fn repeated_postings_lookups_share_one_decoded_run() {
        let (_dir, storage, st) = fresh_table();
        commit_segment(&st, 0);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let a = rt
            .block_on(index.postings("title", "shared"))
            .expect("first");
        let b = rt
            .block_on(index.postings("title", "shared"))
            .expect("second");
        assert!(Arc::ptr_eq(&a, &b), "the second ask is the resident run");
        let absent = rt
            .block_on(index.postings("title", "absent"))
            .expect("absent");
        assert!(absent.is_empty());
        let again = rt
            .block_on(index.postings("title", "absent"))
            .expect("absent again");
        assert!(
            Arc::ptr_eq(&absent, &again),
            "an absent term is remembered too"
        );
    }
}
