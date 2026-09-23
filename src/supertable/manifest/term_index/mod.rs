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
    supertable::manifest::{
        RoutingRef, SuperfileEntry, disk_cache::ManifestDiskCache, part::ContentHash,
    },
    utils::terms::make_key,
};

/// Object-store directory prefix for term-index objects, sibling to the
/// superfile data, manifest-parts and term-stats prefixes.
pub(crate) const STORAGE_PREFIX: &str = "term-index/";

/// Objects at or above this size go through the multipart upload path.
const MULTIPART_THRESHOLD_BYTES: u64 = 100 * 1024 * 1024;

/// Bytes of fetched slices kept resident per loaded index, least recently
/// used first out. Sized so a query burst over a large table's whole
/// vocabulary stays resident: at ~8 MiB a slice this holds ~64 slices, and a
/// slice is re-read from the manifest disk cache (a local read, no hash)
/// only after it has fallen out.
const RESIDENT_SLICE_BUDGET_BYTES: usize = 512 * 1024 * 1024;

/// Errors from building, storing or reading the term index.
#[derive(Debug, Error)]
pub(crate) enum TermIndexError {
    /// Object storage failed.
    #[error("term-index storage error: {0}")]
    Storage(String),
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
        Self::Storage(e.to_string())
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
    let content_hash = ContentHash::of(&bytes);
    let uri = object_uri(kind, &content_hash);
    match crate::supertable::writer::put_bytes_multipart_or_atomic(
        storage,
        &uri,
        Bytes::from(bytes),
        MULTIPART_THRESHOLD_BYTES,
    )
    .await
    {
        // An object that already exists under its hash name is the same
        // bytes.
        Ok(()) | Err(StorageError::PreconditionFailed { .. }) => {}
        Err(e) => return Err(e.into()),
    }
    Ok(RoutingRef { uri, content_hash })
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

/// Persist a finished build — slices first, then the root — and return
/// the root's reference for the manifest.
pub(crate) async fn write_built(
    storage: &dyn StorageProvider,
    built: Built,
) -> Result<RoutingRef, TermIndexError> {
    for (hash, bytes) in built.slices {
        let reference = write_slice(storage, bytes).await?;
        debug_assert_eq!(reference.content_hash, hash);
    }
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
    for (hash, bytes) in built.slices {
        let reference = write_slice(storage, bytes).await?;
        debug_assert_eq!(reference.content_hash, hash);
    }
    root.superfiles.extend(built.superfiles);
    root.id_mins.extend(built.id_mins);
    root.segments.push(built.segment);
    write_root(storage, &root).await
}

/// Fetch a content-addressed object, through the manifest disk cache when
/// one is attached: a hit is served from local disk; a miss is fetched,
/// hash-verified, and written back best-effort. The hash check is the
/// only integrity check either tier gets.
async fn fetch_verified(
    storage: &dyn StorageProvider,
    disk_cache: Option<&ManifestDiskCache>,
    uri: &str,
    hash: &ContentHash,
) -> Result<Bytes, TermIndexError> {
    // The disk cache is keyed by content hash and verifies on insert, so a
    // hit needs no second hash; re-hashing a multi-megabyte slice on every
    // read was a per-query cost on the order of milliseconds.
    if let Some(cache) = disk_cache
        && let Some(cached) = cache.get(hash).await
    {
        return Ok(Bytes::from(cached));
    }
    let (bytes, _meta) = storage.get(uri).await?;
    if ContentHash::of(bytes.as_ref()) != *hash {
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
    /// Every superfile the root lists. A live superfile absent from it was
    /// committed before the index existed and is routed the old way.
    indexed: HashSet<Uuid>,
    storage: Arc<dyn StorageProvider>,
    disk_cache: Option<Arc<ManifestDiskCache>>,
    slices: tokio::sync::Mutex<ResidentSlices>,
}

/// The resident slice set: insertion-ordered so eviction is least recently
/// used, bounded by bytes.
#[derive(Default)]
struct ResidentSlices {
    order: std::collections::VecDeque<ContentHash>,
    bytes: HashMap<ContentHash, Bytes>,
    total: usize,
}

impl ResidentSlices {
    fn get(&mut self, hash: &ContentHash) -> Option<Bytes> {
        let b = self.bytes.get(hash)?.clone();
        // Move to the back: most recently used.
        if let Some(i) = self.order.iter().position(|h| h == hash) {
            self.order.remove(i);
            self.order.push_back(*hash);
        }
        Some(b)
    }

    fn insert(&mut self, hash: ContentHash, b: Bytes) {
        if self.bytes.contains_key(&hash) {
            return;
        }
        while self.total + b.len() > RESIDENT_SLICE_BUDGET_BYTES && !self.order.is_empty() {
            if let Some(old) = self.order.pop_front()
                && let Some(gone) = self.bytes.remove(&old)
            {
                self.total -= gone.len();
            }
        }
        self.total += b.len();
        self.order.push_back(hash);
        self.bytes.insert(hash, b);
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
        let indexed = root.superfiles.iter().copied().collect();
        Self {
            root,
            root_uri,
            indexed,
            storage,
            disk_cache,
            slices: tokio::sync::Mutex::new(ResidentSlices::default()),
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
        self.indexed.contains(superfile)
    }

    /// The smallest doc id of a listed superfile — the key that finds its
    /// manifest part from the part's recorded id range.
    pub(crate) fn id_min_of(&self, superfile: &Uuid) -> Option<i128> {
        let ordinal = self.root.superfiles.iter().position(|id| id == superfile)?;
        self.root.id_mins.get(ordinal).copied()
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
            for p in self.postings(column, term).await? {
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
                if complete && min_scaled.is_finite() {
                    ceiling += idf_sum * min_scaled;
                }
            }
            out.insert(id, ceiling);
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
            for p in self.postings(column, term).await? {
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
    ) -> Result<Vec<Posting>, TermIndexError> {
        let key = make_key(column, term);
        let mut out = Vec::new();
        let refs: Vec<_> = self.root.slices_for_key(&key).cloned().collect();
        for r in refs {
            let bytes = self.slice_bytes(&r.content_hash).await?;
            let slice = Slice::open(&bytes)?;
            if let Some(run) = slice.postings(&key)? {
                out.extend(run);
            }
        }
        Ok(out)
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

    /// An empty FTS table on local-filesystem storage.
    fn fresh_table() -> (
        TempDir,
        Arc<dyn StorageProvider>,
        crate::supertable::Supertable,
    ) {
        use crate::{
            superfile::builder::FtsConfig,
            supertable::{Supertable, SupertableOptions},
        };
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs"));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool"),
        );
        let options =
            SupertableOptions::new(title_schema(), vec![FtsConfig::new("title")], Vec::new())
                .expect("options")
                .with_writer_pool(pool)
                .with_storage(Arc::clone(&storage));
        (dir, storage, Supertable::create(options).expect("create"))
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
            for p in &shared {
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
        let root = rt
            .block_on(load_root(storage.as_ref(), &reference))
            .expect("root loads and verifies");
        let live: std::collections::HashSet<Uuid> = manifest
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect();
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
        for p in &shared {
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
    /// Like [`fresh_table`] with a reader pool of `threads`, which is also the
    /// width of the bound-ordered open window.
    fn fresh_table_with_threads(
        threads: usize,
    ) -> (
        TempDir,
        Arc<dyn StorageProvider>,
        crate::supertable::Supertable,
    ) {
        use crate::{
            superfile::builder::FtsConfig,
            supertable::{Supertable, SupertableOptions},
        };
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs"));
        let writer_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool"),
        );
        let reader_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool"),
        );
        let options =
            SupertableOptions::new(title_schema(), vec![FtsConfig::new("title")], Vec::new())
                .expect("options")
                .with_writer_pool(writer_pool)
                .with_reader_pool(reader_pool)
                .with_storage(Arc::clone(&storage));
        (dir, storage, Supertable::create(options).expect("create"))
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

        let (_dir, _storage, st) = fresh_table_with_threads(1);
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
            let values = rt
                .block_on(sf.term_locations("title", &names))
                .expect("values");
            for ((name, df), value) in names.iter().zip(dfs).zip(values) {
                assert_eq!(
                    memo.df(name),
                    df,
                    "{name}: df from the index equals the dictionary's"
                );
                let slot = memo.lookup(name).expect("in memo").expect("present");
                match (slot, value.expect("dictionary has it")) {
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
    /// A committed superfile's open blob holds the parquet tail only: the
    /// FTS open ranges are still recorded, so a cold open knows what to
    /// fetch, but their bytes are no longer copied into the manifest.
    #[test]
    fn open_blob_no_longer_inlines_the_dictionary() {
        let (_dir, _storage, st) = fresh_table();
        commit_segment(&st, 0);
        let reader = st.reader().expect("reader");
        for e in reader.manifest().get_all_superfiles() {
            let offsets = e.subsection_offsets.as_ref().expect("offsets recorded");
            assert!(
                !offsets.fts_open_ranges.is_empty(),
                "the ranges are still recorded"
            );
            // An inlined range is its own blob entry starting at the range's
            // offset. (On a fixture this small the parquet tail spans the
            // whole file, so "covered by some entry" would not distinguish.)
            for &(off, _) in &offsets.fts_open_ranges {
                assert!(
                    !offsets.open_blob.iter().any(|(b_off, _)| *b_off == off),
                    "an FTS open range must not be inlined into the manifest"
                );
            }
            assert_eq!(
                offsets.open_blob.len(),
                1,
                "only the parquet tail is inlined"
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
    #[test]
    fn old_format_tables_read_and_mix_with_the_new_format() {
        use std::{fs, path::Path};

        use crate::{
            Bm25SearchOptions,
            superfile::{builder::FtsConfig, fts::reader::Bm25Stats},
            supertable::{Supertable, SupertableOptions},
        };

        fn copy_dir(from: &Path, to: &Path) {
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
        fn open(dir: &Path) -> (Arc<dyn StorageProvider>, Supertable) {
            let storage: Arc<dyn StorageProvider> =
                Arc::new(LocalFsStorageProvider::new(dir).expect("local fs"));
            let pool = Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(2)
                    .build()
                    .expect("pool"),
            );
            let options =
                SupertableOptions::new(title_schema(), vec![FtsConfig::new("title")], Vec::new())
                    .expect("options")
                    .with_writer_pool(pool)
                    .with_storage(Arc::clone(&storage));
            (storage, Supertable::open(options).expect("open"))
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
    /// The resident slice set evicts least recently used by bytes, and a
    /// read refreshes recency — so a query burst that cycles through more
    /// slices than fit keeps the ones it keeps touching.
    #[test]
    fn resident_slices_evict_least_recently_used_by_bytes() {
        let mut r = ResidentSlices::default();
        // Two of these fit the budget; three do not.
        let big = RESIDENT_SLICE_BUDGET_BYTES / 3 + 1;
        let h = |n: u8| ContentHash([n; 32]);
        r.insert(h(1), Bytes::from(vec![0u8; big]));
        r.insert(h(2), Bytes::from(vec![0u8; big]));
        assert!(r.get(&h(1)).is_some(), "touch 1: it is now most recent");
        r.insert(h(3), Bytes::from(vec![0u8; big]));
        assert!(
            r.get(&h(2)).is_none(),
            "2 was least recently used and went first"
        );
        assert!(r.get(&h(1)).is_some(), "1 was refreshed and survives");
        assert!(r.get(&h(3)).is_some());
        assert!(r.total <= RESIDENT_SLICE_BUDGET_BYTES);
        r.insert(h(3), Bytes::from(vec![0u8; big]));
        assert_eq!(r.total, 2 * big, "re-inserting a resident slice is a no-op");
    }
}
