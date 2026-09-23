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

use std::{collections::HashMap, io, sync::Arc};

pub(crate) use build::{
    BuildPolicy, Built, Contribution, ContributionWriter, build, build_segment,
};
use bytes::Bytes;
pub(crate) use format::{Location, Posting, Root, Slice};
use thiserror::Error;
use uuid::Uuid;

use crate::{
    storage::{StorageError, StorageProvider},
    supertable::manifest::{RoutingRef, part::ContentHash},
    utils::terms::make_key,
};

/// Object-store directory prefix for term-index objects, sibling to the
/// superfile data, manifest-parts and term-stats prefixes.
pub(crate) const STORAGE_PREFIX: &str = "term-index/";

/// Objects at or above this size go through the multipart upload path.
const MULTIPART_THRESHOLD_BYTES: u64 = 100 * 1024 * 1024;

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
    root.segments.push(built.segment);
    write_root(storage, &root).await
}

/// Fetch, hash-verify and parse the root a manifest references.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the routing path, which lands next")
)]
pub(crate) async fn load_root(
    storage: &dyn StorageProvider,
    reference: &RoutingRef,
) -> Result<Root, TermIndexError> {
    let (bytes, _meta) = storage.get(&reference.uri).await?;
    if ContentHash::of(bytes.as_ref()) != reference.content_hash {
        return Err(TermIndexError::HashMismatch);
    }
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
    storage: Arc<dyn StorageProvider>,
    slices: tokio::sync::Mutex<HashMap<ContentHash, Bytes>>,
}

#[cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the routing path, which lands next")
)]
impl TermIndex {
    /// Wrap a loaded root.
    pub(crate) fn new(root: Root, storage: Arc<dyn StorageProvider>) -> Self {
        Self {
            root,
            storage,
            slices: tokio::sync::Mutex::new(HashMap::new()),
        }
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
            return Ok(b.clone());
        }
        let (bytes, _meta) = self.storage.get(&slice_uri(hash)).await?;
        if ContentHash::of(bytes.as_ref()) != *hash {
            return Err(TermIndexError::HashMismatch);
        }
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
    use crate::{storage::LocalFsStorageProvider, utils::terms::make_key};

    fn contribution(dir: &TempDir, id: u128, terms: &[(&str, &str, u64)]) -> Contribution {
        let mut w = ContributionWriter::create(dir.path(), Uuid::from_u128(id)).expect("create");
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
        let mut w = ContributionWriter::create(dir.path(), Uuid::from_u128(1)).expect("create");
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
        let index = TermIndex::new(root, Arc::clone(&storage));
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
            let index = TermIndex::new(root, Arc::clone(&storage));
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
            let index = TermIndex::new(root, Arc::clone(&storage));
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
        let index = TermIndex::new(root, Arc::clone(&storage));
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

        let index = TermIndex::new(root, Arc::clone(&storage));
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
                p.bound.is_infinite(),
                "this build writes the trivially valid bound"
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
        let index = TermIndex::new(root_after, Arc::clone(&storage));
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
}
