// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Table-level term index: for every `(column, term)`, the superfiles that
//! contain it, with the term's `df` in each, an upper bound on the score it
//! can reach there, and where its postings sit in that superfile.
//!
//! One artifact answers three questions — *which superfiles hold this term*,
//! *how often it occurs table-wide*, and *where its postings are* — and is
//! looked into, not loaded.
//!
//! **Shape.** A small *root* stays resident: the covered superfiles (postings
//! name them by ordinal) and, per layer, the key range and content hash of
//! every *slice*. A slice is one contiguous range of `field_id \x1F term`
//! keys ([`FieldId::term_key`]; columns are named by id so a rename leaves
//! the index valid),
//! a few MB, holding a front-coded block dictionary (`utils::terms`) over a
//! postings region; a lookup binary-searches the root for the one slice that
//! can hold the key, fetches it, and reads one block. Prefix scans touch one
//! slice or a few adjacent ones. See `format` for the byte layout.
//!
//! **Validity.** A posting is followed only if its superfile is live in the
//! current manifest; postings for removed superfiles are simply ignored. So
//! a removal never invalidates the artifact, and the reference carries
//! forward across every commit. A superfile with no postings in any layer
//! is uncovered and is probed directly.
//!
//! **Content addressing.** Root and slices are named by the blake3 of their
//! bytes (`term-index/root-<hash>.bin`, `term-index/slice-<hash>.bin`), so a
//! write is idempotent, a manifest pins an exact version, and GC keeps every
//! object a manifest inside the safety gap still names.

pub(crate) mod build;
#[cfg(test)]
mod fold_tool;
pub(crate) mod format;
mod layer_store;
mod rebase;
pub(crate) mod tier;

use std::{
    collections::{HashMap, HashSet},
    fs,
    future::Future,
    io,
    iter::repeat_n,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};

use build::build_layer;
#[cfg(test)]
pub(crate) use build::first_contribution_difference;
pub(crate) use build::{
    BuildPolicy, Built, BuiltBytes, Contribution, ContributionWriter, HEAD_DEPTH, HEAD_DEPTH_MAX,
    HeadSink, SliceFile, build, build_onto, merge_layers, scratch_dir, tier_path,
};
use bytes::Bytes;
use format::Layer;
#[cfg(test)]
pub(crate) use format::TierRef;
pub(crate) use format::{Location, Posting, Root, Run, Slice, legacy_bound_as_key};
use futures::{StreamExt, TryStreamExt, stream};
use layer_store::{SLICE_EXTENSION, TIER_EXTENSION};
pub(crate) use layer_store::{keep_built, prune_layer_store};
use rayon::ThreadPool;
use rebase::{OrdinalShift, rebase_layer, remove_spilled};
use thiserror::Error;
pub(crate) use tier::{BLOCK_ENTRIES, ColumnScoring, TierChunkRef, TierDirectory, TierEntry};
use tokio::{fs as async_fs, io::AsyncWriteExt, task, task::JoinSet};
use tracing::{error, warn};
use uuid::Uuid;

use crate::{
    runtime_bridge::run_on_pool,
    storage::{StorageError, StorageProvider, permission_denied_in_chain},
    superfile::{FtsError, fts::reader::BoolMode},
    supertable::{
        error::QueryError,
        manifest::{
            ManifestSnapshot, RoutingRef, SuperfileEntry, disk_cache::ManifestDiskCache,
            part::ContentHash,
        },
        query::prune::PruneLeaf,
        schema::FieldId,
    },
    utils::trace::detail_span,
};

/// Object-store directory prefix for term-index objects, sibling to the
/// superfile data and manifest-parts prefixes.
pub(crate) const STORAGE_PREFIX: &str = "term-index/";

/// Bytes of fetched slices kept resident per loaded index, least recently
/// used first out. Sized so a query burst over a large table's whole
/// vocabulary stays resident: at ~8 MiB a slice this holds ~64 slices, and a
/// slice is re-read from the manifest disk cache (a local read) only after
/// it has fallen out.
const RESIDENT_SLICE_BUDGET_BYTES: usize = 512 * 1024 * 1024;

/// Most bytes one ranged read of a tier takes: a term's chunks lie back to
/// back in the tier object and a merge wants them all, so a miss reads a
/// run of them at once; a longer run is read in pieces this big.
const TIER_READ_MAX_BYTES: u64 = 16 * 1024 * 1024;

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

/// Slices one batched lookup fetches at once. A slice is several MB, so
/// this bounds both the requests in flight and the slice bytes held
/// beyond the resident set while their terms decode.
const SLICE_FETCH_CONCURRENCY: usize = 8;

/// Objects a merge downloads at once. A rebuild at a billion documents
/// reads over a hundred gigabytes of slices and tier objects, and at eight
/// in flight the transfer ran at a fraction of the host's bandwidth; the
/// client keeps far more warm connections than this. Each task holds one
/// object at a time, so this many of the largest tier object bounds the
/// transfer's memory.
const LAYER_FETCH_CONCURRENCY: usize = 16;

/// Objects one build writes at once; the write-side twin of
/// [`LAYER_FETCH_CONCURRENCY`], with the same memory bound: a spilled
/// object is read back only as its upload starts.
const SLICE_WRITE_CONCURRENCY: usize = 16;

/// Ordinals a reserved layer leaves free past the root it was built
/// against, for the deltas other commits append while its own commit
/// retries. Ingest commits land a few times a second and a retrying commit
/// gives up within a minute, so this is rarely outgrown; outgrowing it
/// costs one rebuild of the layer, never correctness. Each unused slot
/// is a nil entry (32 bytes) in the root until the next full rebuild.
const RESERVED_ORDINAL_SLACK: u32 = 256;

/// Fan-out of the tiered fold a commit applies to its root's layers (see
/// [`fold_tail`]). A lookup opens one slice per layer, so a root that only
/// ever appended would cost one slice per commit since the last optimize;
/// folding every run of this many layers of one size level into one keeps
/// fewer than this many per level, a logarithmic number in all, while a
/// commit pays a fold only once per this many commits at each level. Eight
/// keeps a 200-commit root to about twenty layers and a 10k-commit root to
/// about forty.
const FOLD_FAN_OUT: usize = 8;

/// How far below the largest layer of a fold's group a layer may lie and
/// still fold with it, in levels, a level being a factor of [`FOLD_FAN_OUT`]
/// in size. Three quarters of a level, the span Lucene's LogMergePolicy
/// groups its levels by: a fold of [`FOLD_FAN_OUT`] layers of one size
/// lands a whole level up, past the span, so it does not fold again with the
/// group it came from.
const FOLD_LEVEL_SPAN: f64 = 0.75;

/// Errors from building, storing or reading the term index.
#[derive(Debug, Error)]
pub enum TermIndexError {
    /// Object storage failed.
    #[error("term-index storage error: {0}")]
    Storage(StorageError),
    /// Bytes did not parse as the layout `format` describes.
    #[error("term-index artifact malformed: {0}")]
    Malformed(String),
    /// The artifact was written in a layout version this engine does not
    /// read. Maintenance rebuilds it from the superfiles.
    #[error(
        "the table's term index ({what}) is at layout version {found} and this engine reads \
         version {expected}; run optimize() or reindex() on the table to rebuild it"
    )]
    UnsupportedVersion {
        what: &'static str,
        found: u32,
        expected: u32,
    },
    /// Fetched bytes do not hash to what the manifest references.
    #[error("term-index artifact hash mismatch")]
    HashMismatch,
    /// The build's inputs were inconsistent.
    #[error("term-index build error: {0}")]
    Build(String),
    /// Opening a superfile to read its terms failed.
    #[error("term-index open: {0}")]
    Open(#[source] QueryError),
    /// Reading a superfile's dictionary failed; `what` names the read.
    #[error("term-index {what} failed: {source}")]
    Read {
        what: &'static str,
        source: FtsError,
    },
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
    /// True when the backend refused the credentials in use.
    pub(crate) fn is_permission_denied(&self) -> bool {
        match self {
            Self::Storage(e) => e.is_permission_denied(),
            Self::Open(e) => e.is_permission_denied(),
            other => permission_denied_in_chain(other),
        }
    }

    /// Whether the object will never load as it is: absent, unparseable,
    /// not the bytes its hash promises, or in a layout this engine no
    /// longer reads. Only these are repaired by rebuilding; a commit that
    /// meets any other failure keeps the reference and waits.
    pub(crate) fn calls_for_rebuild(&self) -> bool {
        self.as_permanent().is_some()
    }

    /// Whether the artifact is in a layout this engine no longer reads.
    pub(crate) fn is_unsupported_version(&self) -> bool {
        matches!(self, Self::UnsupportedVersion { .. })
    }

    /// This error as a [`PermanentLoadFailure`], when it is one.
    pub(crate) fn as_permanent(&self) -> Option<PermanentLoadFailure> {
        match self {
            Self::Storage(StorageError::NotFound { uri }) => {
                Some(PermanentLoadFailure::NotFound(uri.clone()))
            }
            Self::Malformed(m) => Some(PermanentLoadFailure::Malformed(m.clone())),
            Self::UnsupportedVersion {
                what,
                found,
                expected,
            } => Some(PermanentLoadFailure::UnsupportedVersion {
                what,
                found: *found,
                expected: *expected,
            }),
            Self::HashMismatch => Some(PermanentLoadFailure::HashMismatch),
            _ => None,
        }
    }
}

/// A load failure that every later attempt would repeat: the object is
/// absent, unparseable, not the bytes its hash promises, or in a layout this
/// engine no longer reads. Kept so a broken root is fetched once per
/// handle.
#[derive(Debug, Clone)]
pub(crate) enum PermanentLoadFailure {
    NotFound(String),
    Malformed(String),
    UnsupportedVersion {
        what: &'static str,
        found: u32,
        expected: u32,
    },
    HashMismatch,
}

impl From<PermanentLoadFailure> for TermIndexError {
    fn from(f: PermanentLoadFailure) -> Self {
        match f {
            PermanentLoadFailure::NotFound(uri) => Self::Storage(StorageError::NotFound { uri }),
            PermanentLoadFailure::Malformed(m) => Self::Malformed(m),
            PermanentLoadFailure::UnsupportedVersion {
                what,
                found,
                expected,
            } => Self::UnsupportedVersion {
                what,
                found,
                expected,
            },
            PermanentLoadFailure::HashMismatch => Self::HashMismatch,
        }
    }
}

fn object_uri(kind: &str, hash: &ContentHash) -> String {
    format!("{STORAGE_PREFIX}{kind}-{}.bin", hash.to_hex())
}

/// Storage URI of the slice with this content hash.
pub(crate) fn slice_uri(hash: &ContentHash) -> String {
    object_uri("slice", hash)
}

/// Storage URI of a slice's tier object with this content hash.
/// One chunk of a tier as a merge reads it: its directory and every
/// block's body, in block order.
pub(crate) struct TierChunkData {
    pub(crate) dir: Arc<TierDirectory>,
    pub(crate) bodies: Vec<Bytes>,
}

impl TierChunkData {
    /// Bytes the chunk holds resident: its bodies (the directory is small
    /// beside them).
    fn weight(data: &Arc<Self>) -> usize {
        data.bodies.iter().map(Bytes::len).sum()
    }
}

/// The resident tier chunks, keyed by tier object and offset. Tier objects
/// are content-addressed, so the cache is shared across the roots a table
/// goes through: a delta commit replaces the root, not the tiers already
/// read.
pub(crate) type TierChunkCache = Arc<tokio::sync::Mutex<TierChunks>>;

/// The tier chunks in memory, and the ones a read in flight is about to
/// put there: a query of a term whose chunks another query is reading
/// treats them as resident, and its merge then waits for that read rather
/// than overlapping its opens.
pub(crate) struct TierChunks {
    resident: Resident<(ContentHash, u64), Arc<TierChunkData>>,
    in_flight: HashSet<(ContentHash, u64)>,
}

/// A tier chunk cache of the resident byte budget.
pub(crate) fn new_tier_chunk_cache() -> TierChunkCache {
    Arc::new(tokio::sync::Mutex::new(TierChunks {
        resident: Resident::new(RESIDENT_SLICE_BUDGET_BYTES, TierChunkData::weight),
        in_flight: HashSet::new(),
    }))
}

pub(crate) fn tier_uri(hash: &ContentHash) -> String {
    object_uri("tier", hash)
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

/// Persist the root; idempotent by content hash. Call after every slice it
/// names is written, so a manifest never references a root whose slices
/// are not all present.
#[cfg_attr(
    feature = "detailed-tracing",
    tracing::instrument(skip_all, fields(superfiles = root.superfiles.len()))
)]
pub(crate) async fn write_root(
    storage: &dyn StorageProvider,
    root: &Root,
) -> Result<RoutingRef, TermIndexError> {
    put_content_addressed(storage, "root", root.encode()).await
}

/// Persist every slice of a build; idempotent by content hash.
#[cfg_attr(
    feature = "detailed-tracing",
    tracing::instrument(skip_all, fields(slices = slices.len()))
)]
pub(crate) async fn write_slices(
    storage: &Arc<dyn StorageProvider>,
    slices: Vec<(ContentHash, BuiltBytes)>,
) -> Result<(), TermIndexError> {
    write_objects(storage, "slice", slices).await
}

/// Persist every tier chunk and tier list of a build; idempotent by
/// content hash. Written before the slices that name them, like slices
/// before the root.
#[cfg_attr(
    feature = "detailed-tracing",
    tracing::instrument(skip_all, fields(tiers = tiers.len()))
)]
pub(crate) async fn write_tiers(
    storage: &Arc<dyn StorageProvider>,
    tiers: Vec<(ContentHash, BuiltBytes)>,
) -> Result<(), TermIndexError> {
    write_objects(storage, "tier", tiers).await
}

async fn write_objects(
    storage: &Arc<dyn StorageProvider>,
    kind: &'static str,
    objects: Vec<(ContentHash, BuiltBytes)>,
) -> Result<(), TermIndexError> {
    let jobs = objects.into_iter().map(|(hash, built)| {
        let storage = Arc::clone(storage);
        async move {
            let bytes = match built {
                BuiltBytes::Resident(bytes) => bytes,
                BuiltBytes::Spilled { path, .. } => async_fs::read(&path).await?,
            };
            let reference = put_content_addressed(storage.as_ref(), kind, bytes).await?;
            debug_assert_eq!(reference.content_hash, hash);
            Ok(())
        }
    });
    run_bounded(SLICE_WRITE_CONCURRENCY, "object write", jobs).await
}

/// Run `jobs` as spawned tasks, at most `limit` at once, so they spread
/// over the runtime's workers: polled from the one thread that blocks on
/// a merge's future, as a stream of them would be, they moved bytes at
/// half the rate. The first error ends the run: no further job starts and
/// the tasks still running are stopped before it returns, so none writes
/// into a scratch directory the caller is about to remove.
async fn run_bounded<I, F>(limit: usize, what: &'static str, jobs: I) -> Result<(), TermIndexError>
where
    I: IntoIterator<Item = F>,
    F: Future<Output = Result<(), TermIndexError>> + Send + 'static,
{
    let mut running: JoinSet<Result<(), TermIndexError>> = JoinSet::new();
    let outcome = async {
        for job in jobs {
            while running.len() >= limit {
                join_one(&mut running, what).await?;
            }
            running.spawn(job);
        }
        while !running.is_empty() {
            join_one(&mut running, what).await?;
        }
        Ok(())
    }
    .await;
    if outcome.is_err() {
        running.shutdown().await;
    }
    outcome
}

/// Wait for one task of `running` and surface its result; a task that
/// panicked is reported as a build error.
async fn join_one(
    running: &mut JoinSet<Result<(), TermIndexError>>,
    what: &'static str,
) -> Result<(), TermIndexError> {
    match running.join_next().await {
        Some(Ok(result)) => result,
        Some(Err(e)) => Err(TermIndexError::Build(format!("{what}: {e}"))),
        None => Ok(()),
    }
}

/// Write one fetched object to `path` through the runtime's file I/O,
/// chunk by chunk, so a task stopped by `run_bounded` stops between
/// chunks rather than finishing the write on a blocking thread after the
/// caller has removed the directory. `write_all` hands the bytes over in
/// pieces; nothing copies the object whole first.
async fn write_object(path: PathBuf, bytes: Bytes) -> Result<(), TermIndexError> {
    let mut file = async_fs::File::create(&path).await?;
    file.write_all(&bytes).await?;
    file.flush().await?;
    Ok(())
}

/// Persist a finished build — slices first, then the root — and return
/// the root's reference for the manifest.
#[cfg(test)]
pub(crate) async fn write_built(
    storage: &Arc<dyn StorageProvider>,
    built: Built,
) -> Result<RoutingRef, TermIndexError> {
    write_tiers(storage, built.tiers).await?;
    write_slices(storage, built.slices).await?;
    write_root(storage.as_ref(), &built.root).await
}

/// Publish this commit's superfiles as a delta layer appended to `prior`
/// (the root the current manifest references, or none): write the new
/// slices, then a new root naming the prior layers plus this one, and
/// return the root's reference for the manifest CAS. Ordinals continue
/// from the prior root's superfile count, so earlier postings keep their
/// meaning. Content-addressed throughout: a retry after a lost CAS
/// re-derives the same slice hashes and re-PUTs them as no-ops. A fold the
/// root's layers call for merges on `pool` ([`fold_tail`]).
pub(crate) async fn append_delta(
    storage: &Arc<dyn StorageProvider>,
    prior: Option<Root>,
    contributions: &[Contribution],
    policy: &BuildPolicy,
    pool: Option<&ThreadPool>,
) -> Result<RoutingRef, TermIndexError> {
    let built = build_onto(prior.unwrap_or_default(), contributions, policy)?;
    write_tiers(storage, built.tiers).await?;
    write_slices(storage, built.slices).await?;
    publish_root(storage, built.root, policy, pool).await
}

/// A delta layer built and written once, numbered from a reserved
/// ordinal past the root it was built against. A commit that loses its race
/// appends the same layer to the winner's root, padding up to its first
/// ordinal, instead of building and uploading it again. When the winner's
/// root has grown past that ordinal, the layer is moved to new ordinals
/// from the bytes it keeps (see [`rebase`]) rather than built again.
pub(crate) struct ReservedLayer {
    /// The ordinal its first superfile takes. Its postings name
    /// `base + i` for the `i`-th of `superfiles`.
    base: u32,
    superfiles: Vec<Uuid>,
    id_mins: Vec<i128>,
    layer: Layer,
    /// The layer's slice and tier bytes, kept to move it.
    bytes: Arc<LayerBytes>,
    /// Appended only at exactly the root's length, and moved there
    /// otherwise, so the root is never padded on its account.
    exact: bool,
}

/// The slice and tier objects of a layer, by content hash.
struct LayerBytes {
    slices: Vec<(ContentHash, BuiltBytes)>,
    tiers: Vec<(ContentHash, BuiltBytes)>,
}

impl ReservedLayer {
    /// The ordinal just past the layer's last superfile.
    pub(crate) fn end(&self) -> u32 {
        self.base + self.superfiles.len() as u32
    }

    /// Whether it was built from `contributions`, in their order.
    fn built_from(&self, contributions: &[Contribution]) -> bool {
        self.superfiles.len() == contributions.len()
            && self
                .superfiles
                .iter()
                .zip(contributions)
                .all(|(id, c)| *id == c.superfile_id)
    }
}

/// Build `contributions` into a layer numbered from `base` and write its
/// slices, for a later [`append_reserved`].
pub(crate) async fn reserve_layer(
    storage: &Arc<dyn StorageProvider>,
    contributions: &[Contribution],
    policy: &BuildPolicy,
    base: u32,
    exact: bool,
) -> Result<ReservedLayer, TermIndexError> {
    let built = detail_span!(
        "term_index.build_layer",
        inputs = contributions.len(),
        ordinal_base = base,
    )
    .in_scope(|| build_layer(contributions, policy, base))?;
    write_tiers(storage, built.tiers.clone()).await?;
    write_slices(storage, built.slices.clone()).await?;
    Ok(ReservedLayer {
        base,
        superfiles: built.superfiles,
        id_mins: built.id_mins,
        layer: built.layer,
        bytes: Arc::new(LayerBytes {
            slices: built.slices,
            tiers: built.tiers,
        }),
        exact,
    })
}

/// `reserved` moved to start at `base` and its moved objects written, or
/// `None` when it cannot be moved in place (see [`OrdinalShift::new`]) or
/// the move failed, for the caller to build it again. The CPU work runs on
/// `pool`, one slice per task.
async fn move_reserved(
    storage: &Arc<dyn StorageProvider>,
    reserved: &ReservedLayer,
    base: u32,
    pool: Option<&ThreadPool>,
) -> Result<Option<ReservedLayer>, TermIndexError> {
    let n = reserved.superfiles.len() as u32;
    let Some(shift) = OrdinalShift::new(reserved.base, base, n) else {
        return Ok(None);
    };
    let span = detail_span!(
        "term_index.rebase_layer",
        from = reserved.base,
        to = base,
        superfiles = n,
        slices = reserved.layer.slices.len(),
    );
    let (layer, bytes) = (reserved.layer.clone(), Arc::clone(&reserved.bytes));
    let moved = run_on_pool(pool, "term-index layer rebase", move || {
        span.in_scope(|| rebase_layer(&layer, &bytes.slices, &bytes.tiers, &shift))
    })
    .await
    .map_err(|e| TermIndexError::Build(e.to_string()))?;
    let moved = match moved {
        Ok(moved) => moved,
        // Bytes this build wrote that the move cannot parse: the writer and
        // the move disagree on the format, which is a bug.
        Err(error @ TermIndexError::Malformed(_)) => {
            error!(%error, "term index: a reserved layer did not parse for a move; building it again");
            return Ok(None);
        }
        Err(error) => {
            warn!(%error, "term index: moving a reserved layer failed; building it again");
            return Ok(None);
        }
    };
    write_tiers(storage, moved.tiers.clone()).await?;
    write_slices(storage, moved.slices.clone()).await?;
    Ok(Some(ReservedLayer {
        base,
        superfiles: reserved.superfiles.clone(),
        id_mins: reserved.id_mins.clone(),
        layer: moved.layer,
        bytes: Arc::new(LayerBytes {
            slices: moved.slices,
            tiers: moved.tiers,
        }),
        exact: reserved.exact,
    }))
}

/// [`append_delta`] for a commit that may retry: the layer is built once,
/// [`RESERVED_ORDINAL_SLACK`] ordinals past `prior`'s length, and kept in
/// `reserved`. Each call pads `prior` with nil entries up to the reserved
/// ordinal and appends it, writing only the root unless the root's layers
/// make a run [`fold_tail`] merges, on `pool`. When `prior` has grown past
/// the reserved ordinal, the layer is moved to new ordinals, the slack past
/// `prior`'s length, and its moved slices written; it is built again only
/// when it cannot be moved, or the contributions are not the ones it was
/// built from. An `exact` layer is never padded: it is reused only at
/// `prior`'s exact length and otherwise moved there, without the slack.
///
/// Like every delta, it names only ordinals at or past the length of the
/// root it extends, which `layers_since` relies on. Optimize's rebuild
/// drops the padding.
pub(crate) async fn append_reserved(
    storage: &Arc<dyn StorageProvider>,
    prior: Option<Root>,
    contributions: &[Contribution],
    policy: &BuildPolicy,
    reserved: &mut Option<ReservedLayer>,
    pool: Option<&ThreadPool>,
) -> Result<RoutingRef, TermIndexError> {
    let mut root = prior.unwrap_or_default();
    let len = root.superfiles.len();
    let exact = reserved.as_ref().is_some_and(|r| r.exact);
    let reusable = reserved.as_ref().is_some_and(|r| {
        (r.base as usize == len || (!exact && r.base as usize > len)) && r.built_from(contributions)
    });
    if !reusable {
        let slack = if exact { 0 } else { RESERVED_ORDINAL_SLACK };
        let base = len as u32 + slack;
        let moved = match reserved
            .as_ref()
            .filter(|kept| kept.built_from(contributions))
        {
            Some(kept) => move_reserved(storage, kept, base, pool).await?,
            None => None,
        };
        let next = match moved {
            Some(moved) => moved,
            None => reserve_layer(storage, contributions, policy, base, exact).await?,
        };
        // The replaced layer's objects are written; its spilled copies are
        // read by nothing again.
        if let Some(replaced) = reserved.replace(next) {
            remove_spilled(&replaced.bytes.slices);
            remove_spilled(&replaced.bytes.tiers);
        }
    }
    let layer = reserved.as_ref().expect("reserved above");
    let padding = layer.base as usize - len;
    root.superfiles.extend(repeat_n(Uuid::nil(), padding));
    root.superfiles.extend_from_slice(&layer.superfiles);
    root.id_mins.extend(repeat_n(0, padding));
    root.id_mins.extend_from_slice(&layer.id_mins);
    root.layers.push(layer.layer.clone());
    publish_root(storage, root, policy, pool).await
}

/// Fold `root`'s layers where they have piled up, then write it and
/// return its reference. What every delta writer ends with. A fold is only
/// an optimisation, so one that fails is logged and the root published as
/// far as the folds before it took it, every layer still in it: the
/// commit it is part of never fails on its account.
async fn publish_root(
    storage: &Arc<dyn StorageProvider>,
    mut root: Root,
    policy: &BuildPolicy,
    pool: Option<&ThreadPool>,
) -> Result<RoutingRef, TermIndexError> {
    // A commit's policy knows its superfiles' scoring by id, not by
    // ordinal — the ordinals are this root's; the fold's merge keys the
    // tiers it rewrites by ordinal, so the table is laid out by the root
    // here. A policy that already carries the ordinal table keeps it.
    let fold_policy = match policy.scoring_by_ordinal.is_empty() {
        true => BuildPolicy {
            scoring_by_ordinal: Arc::new(
                root.superfiles
                    .iter()
                    .map(|id| {
                        policy
                            .scoring_by_superfile
                            .get(id)
                            .cloned()
                            .unwrap_or_default()
                    })
                    .collect(),
            ),
            ..policy.clone()
        },
        false => policy.clone(),
    };
    if let Err(error) = fold_tail(storage, &mut root, &fold_policy, pool).await {
        warn!(%error, "term index: folding layers at commit failed; publishing the root unfolded");
    }
    write_root(storage.as_ref(), &root).await
}

/// Merge each run of layers [`next_fold`] picks into one, the merge on
/// `pool`, until it picks none. Sizes are the slice bytes the root records,
/// so no slice is read to decide. Ordinals are unchanged: a fold is the
/// same postings in fewer slices, and its run is consecutive, so every
/// reader of the folded root answers as it did, in the same order.
///
/// This is what keeps a lookup cheap on a table taking a stream of
/// commits: each commit appends one layer, a lookup opens one slice per
/// layer, and only optimize rebuilds the index whole. Folded this way the
/// count stays logarithmic in the bytes committed since the last optimize,
/// whatever mix of sizes the commits come in; the merge work a commit pays
/// is amortized to the same logarithm, as each posting is rewritten about
/// once per level it climbs.
///
/// `root` changes only once a fold's merged slices are written, so a fold
/// that fails leaves it as the last one that succeeded made it.
///
/// A rebuild racing these commits stacks the layers committed since its
/// prior root (`layers_since`) only while that root is still a prefix of
/// the current one: a fold wholly above it leaves it one and is stacked as
/// a layer like any other, while a fold that reached into it does not,
/// and the rebuild catches up by reading the superfiles it lacks instead,
/// its existing fallback.
pub(crate) async fn fold_tail(
    storage: &Arc<dyn StorageProvider>,
    root: &mut Root,
    policy: &BuildPolicy,
    pool: Option<&ThreadPool>,
) -> Result<(), TermIndexError> {
    while let Some(run) = next_fold(root, policy) {
        let scratch = scratch_dir(policy.scratch_root.as_deref())?;
        let sources =
            download_layers(storage, &root.layers[run.clone()], scratch.path(), None).await?;
        // Every ordinal keeps its place: the fold renumbers nothing.
        let ordinals: Vec<Option<u32>> = (0..root.superfiles.len() as u32).map(Some).collect();
        let policy = policy.clone();
        let merged = run_on_pool(pool, "term-index fold at commit", move || {
            let merged = merge_layers(&sources, &ordinals, &policy);
            // The slice files are read; the scratch directory can go with
            // the task rather than outlive the fold.
            drop(scratch);
            merged
        })
        .await
        .map_err(|e| TermIndexError::Build(e.to_string()))??;
        write_slices(storage, merged.slices).await?;
        write_tiers(storage, merged.tiers).await?;
        root.layers.drain(run.clone());
        root.layers.insert(run.start, merged.layer);
    }
    Ok(())
}

/// The run of layers [`fold_tail`] merges next: [`FOLD_FAN_OUT`]
/// consecutive layers of one size level whose bytes together fit one
/// slice of `policy`'s target size. `None` when there is no such run.
///
/// Levels group as Lucene's LogMergePolicy groups them, so a mixed stream
/// of commit sizes folds as a uniform one does. From the oldest layer on,
/// the largest level left sets a group that runs to the last layer within
/// [`FOLD_LEVEL_SPAN`] of it, the smaller layers between included; the
/// group folds [`FOLD_FAN_OUT`] at a time from its oldest, and the next
/// group starts after it. Once folded, each group holds fewer than the
/// fan-out, and the groups fall by more than the span, so the count is
/// logarithmic.
///
/// The bound keeps a fold, which downloads, merges and uploads inside the
/// commit before its fence check, to the slice a cold lookup reads: a
/// larger fold would be commit latency and a wider window for a concurrent
/// writer to win, so a run past it is left to optimize's rebuild, as every
/// layer was before folding.
///
/// The newest layer is never in a run. Its slices were written by the
/// publish that is still under way and no root names them yet; a fold that
/// retired them would leave objects no root ever named, which the sweep
/// could not find. Every layer beneath it was named by the root this
/// publish replaces, so the slices a fold retires are exactly the ones the
/// sweep reclaims from that root.
fn next_fold(root: &Root, policy: &BuildPolicy) -> Option<Range<usize>> {
    let beneath = root.layers.len().checked_sub(1)?;
    let layers = &root.layers[..beneath];
    let sizes: Vec<u64> = layers.iter().map(layer_bytes).collect();
    let levels: Vec<f64> = layers.iter().map(layer_level).collect();
    let bound = policy.slice_target_bytes as u64;
    let mut start = 0;
    while start < beneath {
        let top = levels[start..]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let end = 1
            + (start..beneath)
                .rev()
                .find(|&i| levels[i] >= top - FOLD_LEVEL_SPAN)?;
        let mut first = start;
        while first + FOLD_FAN_OUT <= end {
            let run = first..first + FOLD_FAN_OUT;
            let bytes = sizes[run.clone()]
                .iter()
                .fold(0u64, |sum, &size| sum.saturating_add(size));
            if bytes <= bound {
                return Some(run);
            }
            first += FOLD_FAN_OUT;
        }
        start = end;
    }
    None
}

/// The slice bytes `layer` holds, as the root records them.
fn layer_bytes(layer: &Layer) -> u64 {
    layer.slices.iter().map(|slice| slice.len).sum()
}

/// `layer`'s size level: its bytes as a power of [`FOLD_FAN_OUT`].
fn layer_level(layer: &Layer) -> f64 {
    (layer_bytes(layer).max(1) as f64).log(FOLD_FAN_OUT as f64)
}

/// The delta layers committed on top of `prior` to reach `current`, or
/// `None` when `current` was not built on `prior` (e.g. another rebuild
/// replaced it).
///
/// Those deltas name superfiles only by ordinals at or past `prior`'s
/// length. The checks below cannot see posting ordinals; it holds because
/// the delta writers, `append_delta` and `append_reserved`, number from at
/// or past the length of the root they extend. A new root writer must keep
/// it.
pub(crate) fn layers_since<'a>(prior: &Root, current: &'a Root) -> Option<&'a [Layer]> {
    (current.superfiles.starts_with(&prior.superfiles)
        && current.id_mins.starts_with(&prior.id_mins)
        && current.layers.starts_with(&prior.layers))
    .then(|| &current.layers[prior.layers.len()..])
}

/// `base`, a rebuild made from `prior`, with the delta layers committed on
/// top of `prior` to reach `current` appended as they are. For a rebuild
/// that lost its publish to those commits: no slice is read or written, so
/// a retry costs one root however fast commits land.
///
/// The deltas name ordinals at or past `prior`'s length (see
/// [`layers_since`]), so `base` is padded with nil entries up to that
/// length and `current`'s superfiles past it are carried over. `None` when
/// `current` was not built on `prior`, or when `base` lists more superfiles
/// than `prior` and its ordinals would collide with the deltas'.
pub(crate) fn rebase(base: &Root, prior: &Root, current: &Root) -> Option<Root> {
    let deltas = layers_since(prior, current)?;
    let len = prior.superfiles.len();
    if base.superfiles.len() > len {
        return None;
    }
    // Nothing landed on top of `prior`: the rebuild stands alone, unpadded.
    if deltas.is_empty() && current.superfiles.len() == len {
        return Some(base.clone());
    }
    let padding = len - base.superfiles.len();
    let mut root = base.clone();
    root.superfiles.extend(repeat_n(Uuid::nil(), padding));
    root.superfiles
        .extend_from_slice(&current.superfiles[len..]);
    root.id_mins.extend(repeat_n(0, padding));
    root.id_mins.extend_from_slice(&current.id_mins[len..]);
    root.layers.extend_from_slice(deltas);
    Some(root)
}

/// Download every slice of `layers` into `dir`, verified against its
/// hash, [`LAYER_FETCH_CONCURRENCY`] at a time, and return each layer's
/// slice files in key order. A slice two layers share is fetched once.
/// Each slice's tier object lands in `dir` too, named by [`tier_path`],
/// so a merge finds it beside the slice. The bytes skip the manifest
/// disk cache: a merge reads each object once, and filling the cache with
/// the whole index would evict what queries use. With a `store`, an object
/// it holds is linked from it, and one downloaded is kept in it.
pub(crate) async fn download_layers(
    storage: &Arc<dyn StorageProvider>,
    layers: &[Layer],
    dir: &Path,
    store: Option<&Path>,
) -> Result<Vec<Vec<SliceFile>>, TermIndexError> {
    let mut unique: HashSet<ContentHash> = HashSet::new();
    let jobs = layers
        .iter()
        .flat_map(|layer| &layer.slices)
        .filter(|slice| unique.insert(slice.content_hash))
        .map(|slice| {
            let storage = Arc::clone(storage);
            let store = store.map(Path::to_path_buf);
            let path = slice_file_path(dir, &slice.content_hash);
            let tier_file = tier_path(dir, &slice.content_hash);
            let (hash, tier) = (slice.content_hash, slice.tier);
            async move {
                // One object resident at a time: the slice is on disk
                // before its tier object is fetched.
                let store = store.as_deref();
                let uri = slice_uri(&hash);
                fetch_object(storage.as_ref(), store, &uri, &hash, SLICE_EXTENSION, path).await?;
                if let Some(tier) = &tier {
                    let uri = tier_uri(tier);
                    fetch_object(
                        storage.as_ref(),
                        store,
                        &uri,
                        tier,
                        TIER_EXTENSION,
                        tier_file,
                    )
                    .await?;
                }
                Ok(())
            }
        });
    run_bounded(LAYER_FETCH_CONCURRENCY, "layer fetch", jobs).await?;
    if let Some(store) = store {
        layer_store::sync_dir(store).await;
    }
    Ok(slice_files(dir, layers))
}

/// Put the object `hash` at `path`: linked from `store` when it holds it,
/// else downloaded, verified, and kept in `store`.
async fn fetch_object(
    storage: &dyn StorageProvider,
    store: Option<&Path>,
    uri: &str,
    hash: &ContentHash,
    extension: &'static str,
    path: PathBuf,
) -> Result<(), TermIndexError> {
    if let Some(store) = store
        && layer_store::link_kept(store, hash, extension, &path).await
    {
        return Ok(());
    }
    let bytes = fetch_verified(storage, None, uri, hash).await?;
    write_object(path.clone(), bytes).await?;
    if let Some(store) = store {
        layer_store::keep_file(store.to_path_buf(), path, *hash, extension).await;
    }
    Ok(())
}

/// Where `download_layers` puts the slice named by `hash` under `dir`.
fn slice_file_path(dir: &Path, hash: &ContentHash) -> PathBuf {
    dir.join(format!("{}.slice", hash.to_hex()))
}

/// `layers`' slices as the files `download_layers` places under
/// `dir`, each with the key range its layer entry records.
pub(crate) fn slice_files(dir: &Path, layers: &[Layer]) -> Vec<Vec<SliceFile>> {
    layers
        .iter()
        .map(|layer| {
            layer
                .slices
                .iter()
                .map(|slice| SliceFile {
                    path: slice_file_path(dir, &slice.content_hash),
                    first_key: slice.first_key.clone(),
                    last_key: slice.last_key.clone(),
                })
                .collect()
        })
        .collect()
}

/// Write freshly built slices into `dir`, each with the key range its
/// layer entry in `refs` records, so a merge finds them where
/// [`download_layers`] would have put them.
pub(crate) fn spill_slices(
    dir: &Path,
    refs: &[format::SliceRef],
    slices: Vec<(ContentHash, BuiltBytes)>,
) -> Result<Vec<SliceFile>, TermIndexError> {
    let keys_of: HashMap<ContentHash, (&[u8], &[u8])> = refs
        .iter()
        .map(|r| {
            (
                r.content_hash,
                (r.first_key.as_slice(), r.last_key.as_slice()),
            )
        })
        .collect();
    slices
        .into_iter()
        .map(|(hash, built)| {
            let (first_key, last_key) = keys_of.get(&hash).ok_or_else(|| {
                TermIndexError::Build("a built slice its layer does not list".into())
            })?;
            let path = dir.join(format!("{}.slice", hash.to_hex()));
            place_built(&path, built)?;
            Ok(SliceFile {
                path,
                first_key: first_key.to_vec(),
                last_key: last_key.to_vec(),
            })
        })
        .collect()
}

/// Put `built` at `path`: a resident object is written, a spilled one is
/// linked or copied from where the build left it.
fn place_built(path: &Path, built: BuiltBytes) -> Result<(), TermIndexError> {
    match built {
        BuiltBytes::Resident(bytes) => fs::write(path, bytes)?,
        BuiltBytes::Spilled { path: from, .. } => {
            if from != path && fs::hard_link(&from, path).is_err() {
                fs::copy(&from, path)?;
            }
        }
    }
    Ok(())
}

/// One ranged read of a tier object: the chunks of `refs`, which lie in
/// `start..end` back to back, parsed and every block verified on the pool.
/// Returns the bytes fetched and the chunks, each with its place among
/// the run's chunks.
async fn read_span(
    storage: Arc<dyn StorageProvider>,
    uri: &str,
    start: u64,
    end: Option<u64>,
    refs: Vec<(usize, TierChunkRef)>,
) -> Result<(u64, Vec<(usize, Arc<TierChunkData>)>), TermIndexError> {
    let end = end.ok_or_else(|| TermIndexError::Malformed("tier chunk range".into()))?;
    let bytes = storage.get_range(uri, start..end).await?;
    if bytes.len() as u64 != end - start {
        return Err(TermIndexError::Malformed("short tier read".into()));
    }
    let fetched = bytes.len() as u64;
    let chunks = run_on_pool(None, "tier chunk verify", move || {
        let mut out = Vec::with_capacity(refs.len());
        for (i, chunk) in refs {
            let at = (chunk.offset - start) as usize;
            let chunk_bytes = bytes.slice(at..at + chunk.len as usize);
            let dir = Arc::new(TierDirectory::parse(&chunk_bytes, chunk.n_blocks)?);
            if dir.chunk_len() as u64 != chunk.len {
                return Err(TermIndexError::Malformed(
                    "tier chunk directory does not match the run's chunk length".into(),
                ));
            }
            let mut bodies = Vec::with_capacity(dir.n_blocks());
            for b in 0..dir.n_blocks() {
                let body = chunk_bytes.slice(dir.block_range(b));
                dir.verify_block(b, &body)?;
                bodies.push(body);
            }
            out.push((i, Arc::new(TierChunkData { dir, bodies })));
        }
        Ok::<_, TermIndexError>(out)
    })
    .await
    .map_err(|e| TermIndexError::Open(QueryError::Internal(e.to_string())))??;
    Ok((fetched, chunks))
}

/// Write freshly built tier objects into `dir`, each beside the spilled
/// slice of `slices` that names it, so a merge finds them where
/// [`download_layers`] would have put them.
pub(crate) fn spill_tiers(
    dir: &Path,
    slices: &[format::SliceRef],
    tiers: Vec<(ContentHash, BuiltBytes)>,
) -> Result<(), TermIndexError> {
    let slice_of: HashMap<ContentHash, ContentHash> = slices
        .iter()
        .filter_map(|s| Some((s.tier?, s.content_hash)))
        .collect();
    for (hash, built) in tiers {
        let slice = slice_of
            .get(&hash)
            .ok_or_else(|| TermIndexError::Build("a built tier object no slice names".into()))?;
        place_built(&tier_path(dir, slice), built)?;
    }
    Ok(())
}

/// Fetch a content-addressed object, through the manifest disk cache when
/// one is attached: a hit is served from local disk, already verified by
/// the cache; a miss is fetched, hash-verified, and written back
/// best-effort. The hash check is the only integrity check either tier
/// gets.
async fn fetch_verified(
    storage: &dyn StorageProvider,
    disk_cache: Option<&Arc<ManifestDiskCache>>,
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
        cache.put(*hash, bytes.clone()).await;
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

/// A term's run from one slice, with the slice's tier object when the
/// slice has one: what a run's chunk offsets are relative to.
#[derive(Debug, Clone)]
pub(crate) struct SliceRun {
    pub(crate) tier_object: Option<ContentHash>,
    pub(crate) run: Run,
}

/// A term's postings across its runs, in layer order.
pub(crate) fn postings_of(runs: &[SliceRun]) -> impl Iterator<Item = &Posting> {
    runs.iter().flat_map(|r| r.run.postings.iter())
}

/// The resident half of the index plus a cache of fetched slices.
///
/// A slice is fetched whole on first use and kept; slices are immutable
/// and content-addressed, so the cache never goes stale. Bounding the cache
/// is the disk-cache layer's job once slices route through it; here it is
/// simply a map, sized by what the process has looked up.
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
    /// Fetched tier chunks — directory and bodies — by `(tier object, chunk
    /// offset)`, under the same byte budget as the slices. A chunk is the
    /// unit so a merge's reads cost the set a lookup per region, not per
    /// block.
    tier_chunks: TierChunkCache,
    /// Decoded runs by key; see [`RESIDENT_RUNS`]. A std mutex: nothing
    /// awaits while it is held.
    runs: std::sync::Mutex<Resident<Vec<u8>, Arc<Vec<SliceRun>>>>,
}

/// A bounded resident set, least recently used first out: `weigh` gives
/// each value's cost against `budget`, and a read refreshes recency, so a
/// burst that cycles through more than fits keeps what it keeps touching.
pub(crate) struct Resident<K, V> {
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

    /// Whether `key` is held, without counting a use.
    fn holds(&self, key: &K) -> bool {
        self.map.contains_key(key)
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
            tier_chunks: new_tier_chunk_cache(),
            runs: std::sync::Mutex::new(Resident::new(RESIDENT_RUNS, |_| 1)),
        }
    }

    /// This index reading tier chunks through `cache`, shared with the
    /// indexes of the roots before and after it.
    pub(crate) fn with_tier_chunk_cache(mut self, cache: TierChunkCache) -> Self {
        self.tier_chunks = cache;
        self
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
            disk_cache.as_ref(),
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
    /// list, a non-UTF-8 prefix, a column the table does not have; the
    /// caller then keeps its summary-based answer. A lookup that fails is
    /// an error.
    pub(crate) async fn route_leaf(
        &self,
        leaf: &PruneLeaf,
        snapshot: &ManifestSnapshot,
    ) -> Result<Option<HashSet<Uuid>>, TermIndexError> {
        match leaf {
            PruneLeaf::TermPresence {
                column,
                terms,
                mode,
            } if !terms.is_empty() => {
                let Some(column) = snapshot.field_id(column) else {
                    return Ok(None);
                };
                let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
                self.route(column, &refs, *mode).await.map(Some)
            }
            PruneLeaf::Prefix { column, prefix } => {
                let Some(column) = snapshot.field_id(column) else {
                    return Ok(None);
                };
                let Ok(prefix) = std::str::from_utf8(prefix) else {
                    return Ok(None);
                };
                self.route_prefix(column, prefix).await.map(Some)
            }
            _ => Ok(None),
        }
    }

    /// The superfiles that can match `terms` in `column` under `mode`:
    /// the union of the terms' posting sets for `Or`, their intersection
    /// for `And`. Exact for indexed superfiles; the caller decides what to
    /// do with live superfiles the root does not list. Ordinals that no
    /// longer resolve are skipped.
    pub(crate) async fn route(
        &self,
        column: FieldId,
        terms: &[&str],
        mode: BoolMode,
    ) -> Result<HashSet<Uuid>, TermIndexError> {
        let mut out: Option<HashSet<Uuid>> = None;
        for runs in self.runs_many(column, terms).await? {
            let set: HashSet<Uuid> = postings_of(&runs)
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
        column_id: FieldId,
        terms: &[&str],
        phrases: &[Vec<&str>],
        entries: &[Arc<SuperfileEntry>],
        idf_used: &(dyn Fn(&str) -> f32 + Sync),
    ) -> Result<HashMap<Uuid, f32>, TermIndexError> {
        // A layer of the layout before tiers stores a bound as a score
        // at the superfile's own idf: each superfile's scored document
        // count turns it into the key the current layout stores.
        let scored_docs = |id: Uuid| -> u64 {
            entries
                .iter()
                .find(|e| e.superfile_id == id)
                .map_or(0, |e| {
                    e.fts_summary
                        .get(&column_id)
                        .and_then(|s| s.length_stats.as_ref().map(|l| l.n_scored_docs))
                        .unwrap_or(e.n_docs)
                })
        };
        // Per term: superfile → the key of its best document there.
        let mut per_term: HashMap<&str, HashMap<Uuid, f32>> = HashMap::new();
        let mut all_terms: Vec<&str> = terms.to_vec();
        all_terms.extend(phrases.iter().flatten().copied());
        all_terms.sort_unstable();
        all_terms.dedup();
        let runs = self.runs_many(column_id, &all_terms).await?;
        for (term, runs) in all_terms.into_iter().zip(runs) {
            let mut by_sf = HashMap::new();
            for slice_run in runs.iter() {
                for p in &slice_run.run.postings {
                    let Some(id) = self.superfile_id(p.superfile) else {
                        continue;
                    };
                    let key = match slice_run.run.legacy_bound {
                        true => legacy_bound_as_key(p.bound, p.df, scored_docs(id)),
                        false => p.bound,
                    };
                    by_sf.insert(id, key);
                }
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
                if let Some(bound) = per_term.get(term).and_then(|m| m.get(&id)) {
                    ceiling += bound * idf_used(term);
                }
            }
            for phrase in phrases {
                let mut idf_sum = 0.0f32;
                let mut min_scaled = f32::INFINITY;
                let mut complete = true;
                for member in phrase {
                    match per_term.get(member).and_then(|m| m.get(&id)) {
                        Some(bound) => {
                            idf_sum += idf_used(member);
                            min_scaled = min_scaled.min(*bound);
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
        column: FieldId,
        terms: &[&str],
        entries: &[Arc<SuperfileEntry>],
    ) -> Result<HashMap<Uuid, Vec<(String, u64, Location)>>, TermIndexError> {
        let live: HashSet<Uuid> = entries
            .iter()
            .map(|e| e.superfile_id)
            .filter(|id| self.is_indexed(id))
            .collect();
        let mut out: HashMap<Uuid, Vec<(String, u64, Location)>> = HashMap::new();
        let runs = self.runs_many(column, terms).await?;
        for (term, runs) in terms.iter().zip(runs) {
            for p in postings_of(&runs) {
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
        column: FieldId,
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
            self.disk_cache.as_ref(),
            &slice_uri(hash),
            hash,
        )
        .await?;
        self.slices.lock().await.insert(*hash, bytes.clone());
        Ok(bytes)
    }

    /// Every posting for `term` in `column`, across all layers, in the
    /// order the layers were written. Empty when no layer holds the
    /// term. The caller filters to superfiles live in its manifest.
    pub(crate) async fn postings(
        &self,
        column: FieldId,
        term: &str,
    ) -> Result<Vec<Posting>, TermIndexError> {
        let mut runs = self.runs_many(column, &[term]).await?;
        Ok(postings_of(&runs.pop().expect("one run per term asked"))
            .cloned()
            .collect())
    }

    /// [`Self::postings`] for each of `terms`, in the same order.
    ///
    /// Terms the resident runs do not answer are grouped by the slice that
    /// holds them, and each distinct slice is fetched once, up to
    /// [`LAYER_FETCH_CONCURRENCY`] at a time. Looking terms up one by one
    /// instead costs one whole-slice fetch per term, back to back: a query
    /// over hundreds of high-cardinality keys spread across a large index
    /// then spends its time waiting on those fetches in series, and refetches
    /// a slice shared by two terms if the resident set evicted it between
    /// them. Each slice's bytes are dropped once its terms are decoded, so
    /// the batch holds at most the in-flight slices beyond the resident set.
    pub(crate) async fn runs_many(
        &self,
        column: FieldId,
        terms: &[&str],
    ) -> Result<Vec<Arc<Vec<SliceRun>>>, TermIndexError> {
        let keys: Vec<Vec<u8>> = terms.iter().map(|t| column.term_key(t)).collect();
        let mut out: Vec<Option<Arc<Vec<SliceRun>>>> = {
            let mut resident = self.runs.lock().expect("resident runs lock");
            keys.iter().map(|k| resident.get(k)).collect()
        };
        // Per missing term, the slices that can hold it in layer order,
        // each with its tier object; per distinct slice, the missing terms
        // it can hold.
        let mut wanted: Vec<(usize, Vec<ContentHash>)> = Vec::new();
        let mut by_slice: HashMap<ContentHash, (Option<ContentHash>, Vec<usize>)> = HashMap::new();
        for (i, key) in keys.iter().enumerate() {
            if out[i].is_some() {
                continue;
            }
            let mut hashes = Vec::new();
            for r in self.root.slices_for_key(key) {
                by_slice
                    .entry(r.content_hash)
                    .or_insert_with(|| (r.tier, Vec::new()))
                    .1
                    .push(i);
                hashes.push(r.content_hash);
            }
            wanted.push((i, hashes));
        }
        if wanted.is_empty() {
            // Every term was resident: nothing to fetch, decode or cache.
            return Ok(out
                .into_iter()
                .map(|run| run.expect("every term was resident"))
                .collect());
        }
        let keys = &keys;
        let mut decoded: HashMap<(usize, ContentHash), SliceRun> = stream::iter(by_slice)
            .map(|(hash, (tier_object, terms_here))| async move {
                let bytes = self.slice_bytes(&hash).await?;
                let slice = Slice::open(&bytes)?;
                let mut found = Vec::new();
                for i in terms_here {
                    if let Some(run) = slice.run(&keys[i])? {
                        let chunked = run.tier.as_ref().is_some_and(|t| !t.chunks.is_empty());
                        if chunked && tier_object.is_none() {
                            return Err(TermIndexError::Malformed(
                                "a run has tier chunks but its slice names no tier object".into(),
                            ));
                        }
                        found.push(((i, hash), SliceRun { tier_object, run }));
                    }
                }
                Ok::<_, TermIndexError>(found)
            })
            .buffer_unordered(SLICE_FETCH_CONCURRENCY)
            .try_fold(HashMap::new(), |mut acc, found| async move {
                acc.extend(found);
                Ok(acc)
            })
            .await?;
        let mut resident = self.runs.lock().expect("resident runs lock");
        for (i, hashes) in wanted {
            let runs: Vec<SliceRun> = hashes
                .into_iter()
                .filter_map(|hash| decoded.remove(&(i, hash)))
                .collect();
            let runs = Arc::new(runs);
            resident.insert(keys[i].clone(), Arc::clone(&runs));
            out[i] = Some(runs);
        }
        Ok(out
            .into_iter()
            .map(|run| run.expect("every term resolved above"))
            .collect())
    }

    /// Every entry of a tier, block by block, in tier order (descending
    /// impact by block). A query reads a prefix instead; this is the whole
    /// thing, for tests.
    #[cfg(test)]
    pub(crate) async fn tier_entries(
        &self,
        object: &ContentHash,
        tier: &TierRef,
    ) -> Result<Vec<TierEntry>, TermIndexError> {
        let mut out = tier.inline.clone();
        let mut block = Vec::new();
        let (chunks, _) = self.tier_chunks(object, &tier.chunks).await?;
        for chunk in &chunks {
            for (i, body) in chunk.bodies.iter().enumerate() {
                chunk.dir.decode_block(i, body, &mut block)?;
                out.extend_from_slice(&block);
            }
        }
        Ok(out)
    }

    /// The chunks of one term's tier — each its parsed directory and every
    /// block's body — with the bytes the call fetched. Chunks not resident
    /// are read in as few ranged reads as their layout allows: the chunks
    /// of a term lie back to back, so a whole tier is one read of up to
    /// [`TIER_READ_MAX_BYTES`]. Every directory is verified by its own CRC
    /// and every block by its own, and all of it is resident afterwards.
    /// Whether every one of `chunks` in `object` is in memory or being
    /// read into it, so a read of them costs no storage round trip of this
    /// query's own.
    pub(crate) async fn tier_chunks_resident(
        &self,
        object: &ContentHash,
        chunks: &[TierChunkRef],
    ) -> bool {
        let cache = self.tier_chunks.lock().await;
        chunks.iter().all(|c| {
            let key = (*object, c.offset);
            cache.resident.holds(&key) || cache.in_flight.contains(&key)
        })
    }

    pub(crate) async fn tier_chunks(
        &self,
        object: &ContentHash,
        chunks: &[TierChunkRef],
    ) -> Result<(Vec<Arc<TierChunkData>>, u64), TermIndexError> {
        let mut out: Vec<Option<Arc<TierChunkData>>> = Vec::with_capacity(chunks.len());
        {
            let mut cache = self.tier_chunks.lock().await;
            for chunk in chunks {
                out.push(cache.resident.get(&(*object, chunk.offset)));
            }
        }
        if out.iter().all(Option::is_some) {
            return Ok((out.into_iter().flatten().collect(), 0));
        }
        // Spans of missing chunks that are contiguous in the object.
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            if out[i].is_some() {
                continue;
            }
            let end = chunk
                .offset
                .checked_add(chunk.len)
                .ok_or_else(|| TermIndexError::Malformed("tier chunk past its object".into()))?;
            match spans.last_mut() {
                Some((first, last))
                    if *last + 1 == i
                        && chunks[*last].offset.checked_add(chunks[*last].len)
                            == Some(chunk.offset)
                        && end - chunks[*first].offset <= TIER_READ_MAX_BYTES =>
                {
                    *last = i;
                }
                _ => spans.push((i, i)),
            }
        }
        // Each span: one ranged read, then its chunks parsed and every
        // block verified on the pool, off this task and off the lock. The
        // read runs as its own task and lands in the cache when it is done,
        // whether or not the query that asked still waits: a query whose
        // opens all finished first moves on, and the next query of the
        // term finds its chunks in memory.
        let uri = tier_uri(object);
        {
            let mut cache = self.tier_chunks.lock().await;
            for &(first, last) in &spans {
                for c in &chunks[first..=last] {
                    cache.in_flight.insert((*object, c.offset));
                }
            }
        }
        let fetches: Vec<_> = spans
            .into_iter()
            .map(|(first, last)| {
                let uri = uri.clone();
                let refs: Vec<(usize, TierChunkRef)> = chunks
                    .iter()
                    .enumerate()
                    .take(last + 1)
                    .skip(first)
                    .map(|(i, c)| (i, c.clone()))
                    .collect();
                let start = chunks[first].offset;
                let end = chunks[last].offset.checked_add(chunks[last].len);
                let storage = Arc::clone(&self.storage);
                let cache = Arc::clone(&self.tier_chunks);
                let object = *object;
                let keys: Vec<(ContentHash, u64)> =
                    refs.iter().map(|(_, c)| (object, c.offset)).collect();
                task::spawn(async move {
                    let read = read_span(storage, &uri, start, end, refs).await;
                    let mut cache = cache.lock().await;
                    for key in &keys {
                        cache.in_flight.remove(key);
                    }
                    let (fetched, chunks) = read?;
                    for (key, (_, data)) in keys.iter().zip(&chunks) {
                        cache.resident.insert(*key, Arc::clone(data));
                    }
                    Ok::<_, TermIndexError>((fetched, chunks))
                })
            })
            .collect();
        let mut fetched = 0u64;
        for handle in fetches {
            let (bytes, span_chunks) = handle
                .await
                .map_err(|e| TermIndexError::Open(QueryError::Internal(e.to_string())))??;
            fetched += bytes;
            for (i, data) in span_chunks {
                out[i] = Some(data);
            }
        }
        Ok((out.into_iter().flatten().collect(), fetched))
    }

    /// Visit every term in `column` with `prefix`, with its postings, until
    /// `visit` returns `false`. Terms arrive in key order within a layer.
    pub(crate) async fn for_each_prefix(
        &self,
        column: FieldId,
        prefix: &str,
        mut visit: impl FnMut(&[u8], Vec<Posting>) -> bool,
    ) -> Result<(), TermIndexError> {
        let key_prefix = column.term_key(prefix);
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
    use std::{
        collections::HashMap,
        fs,
        ops::Range,
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        thread,
        time::Duration,
    };

    use tempfile::TempDir;

    use super::{
        build::{TIER_INLINE_MAX, key_ranges},
        format::{Layer, ROOT_MAGIC, SliceRef},
        *,
    };
    use crate::{
        Bm25SearchOptions, CompactionSettings, InfinoError, OptimizeOptions, OptimizeSteps,
        storage::LocalFsStorageProvider,
        superfile::fts::bm25,
        supertable::{
            Supertable,
            manifest::{
                commit::{POINTER_PATH, read_pointer},
                list,
                options_hash::compute_options_hash,
            },
            query::prune::select_superfiles,
            schema::{LegacyNames, TableSchema},
            writer::stamp_with_retries,
        },
        test_helpers::{
            copy_dir_recursive,
            fault_storage::{FaultOp, FaultStorage},
            fid, old_format_fts_fixture, open_old_format_fts_fixture,
            slow_storage::SlowStorage,
        },
        utils::terms::{DictEntry, TERM_BLOCK_SIZE},
    };

    fn contribution(dir: &TempDir, id: u128, terms: &[(&str, &str, u64)]) -> Contribution {
        let mut w = ContributionWriter::create(dir.path(), Uuid::from_u128(id), id as i128 * 1000)
            .expect("create");
        let mut keyed: Vec<(Vec<u8>, u64)> = terms
            .iter()
            .map(|(c, t, df)| (fid(c).term_key(t), *df))
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
            w.push(key, *df, f32::INFINITY, location, 0.0, &[])
                .expect("push");
        }
        w.finish().expect("finish")
    }

    /// A contribution whose terms carry heads: `(column, term, docs)` with
    /// each doc `(row, weight)`; df is the doc count and the head is
    /// complete when `complete_up_to` covers it.
    /// The facts a fixture document of weight `w` is planted with: a term
    /// frequency of a hundred times the weight at length bucket zero, so
    /// the key orders like the weight.
    fn facts_of(w: f32) -> (u32, u8) {
        ((w * 100.0).round() as u32, 0)
    }

    /// The tier key of a fixture document of weight `w`, at the scoring
    /// the fixtures plant their heads at.
    fn key_of(w: f32) -> f32 {
        let (tf, len) = facts_of(w);
        ColumnScoring::default().key(tf, len)
    }

    fn contribution_with_heads(
        dir: &TempDir,
        id: u128,
        terms: &[(&str, &str, Vec<(u32, f32)>)],
        complete_up_to: usize,
    ) -> Contribution {
        let mut w = ContributionWriter::create(dir.path(), Uuid::from_u128(id), id as i128 * 1000)
            .expect("create");
        let mut keyed: Vec<(Vec<u8>, &str, &Vec<(u32, f32)>)> = terms
            .iter()
            .map(|(c, t, docs)| (fid(c).term_key(t), *c, docs))
            .collect();
        keyed.sort_by(|a, b| a.0.cmp(&b.0));
        for (key, column, docs) in keyed {
            w.set_scoring(fid(column), ColumnScoring::default());
            let mut sorted = docs.clone();
            sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
            let bound = sorted.first().map(|d| key_of(d.1)).unwrap_or(0.0);
            let complete = sorted.len() <= complete_up_to;
            let kept: Vec<(u32, u32, u8)> = sorted
                .iter()
                .take(complete_up_to)
                .map(|&(row, w)| {
                    let (tf, len) = facts_of(w);
                    (row, tf, len)
                })
                .collect();
            let tail = if complete {
                0.0
            } else {
                key_of(sorted[complete_up_to].1)
            };
            w.push(&key, docs.len() as u64, bound, Location::None, tail, &kept)
                .expect("push");
        }
        w.finish().expect("finish")
    }

    /// A built object's bytes, wherever the build kept them.
    fn bytes_of(built: &BuiltBytes) -> Vec<u8> {
        built.read().expect("built bytes")
    }

    /// The entries of `tier`: inline, or read from the first slice's tier
    /// object among `built_tiers` (these fixtures build one slice).
    fn tier_entries_of(
        layer: &Layer,
        built_tiers: &[(ContentHash, BuiltBytes)],
        tier: &format::TierRef,
    ) -> Vec<tier::TierEntry> {
        if tier.chunks.is_empty() {
            return tier.inline.clone();
        }
        let object = layer.slices[0]
            .tier
            .as_ref()
            .expect("the slice has a tier object");
        let bytes = bytes_of(
            &built_tiers
                .iter()
                .find(|(h, _)| h == object)
                .expect("the tier object is among the built")
                .1,
        );
        let mut out = Vec::new();
        for chunk in &tier.chunks {
            let range = chunk.offset as usize..(chunk.offset + chunk.len) as usize;
            tier::TierChunk::verify(&bytes[range.clone()], chunk.n_blocks).expect("crc");
            tier::TierChunk::open(&bytes[range], chunk.n_blocks)
                .expect("open chunk")
                .entries(&mut out)
                .expect("decode");
        }
        out
    }

    #[test]
    fn a_build_pools_the_heads_into_a_tier_and_raises_tails_when_it_truncates() {
        let dir = TempDir::new().expect("tempdir");
        // Three superfiles; "alpha" has three docs in each, "beta" one.
        let docs = |base: f32| vec![(1, base + 0.3), (5, base + 0.2), (9, base + 0.1)];
        let a = contribution_with_heads(
            &dir,
            1,
            &[
                ("body", "alpha", docs(1.0)),
                ("body", "beta", vec![(2, 0.5)]),
            ],
            8,
        );
        let b = contribution_with_heads(&dir, 2, &[("body", "alpha", docs(2.0))], 8);
        let c = contribution_with_heads(&dir, 3, &[("body", "alpha", docs(3.0))], 8);
        let policy = BuildPolicy {
            slice_target_bytes: usize::MAX,
            tier_entries_per_term: 4,
            ..BuildPolicy::default()
        };
        let built = build(&[a, b, c], &policy).expect("build");
        let slice_bytes = bytes_of(&built.slices[0].1);
        let slice = Slice::open(&slice_bytes).expect("open");
        let alpha = slice
            .run(&fid("body").term_key("alpha"))
            .expect("run")
            .expect("alpha is indexed");
        let tier = alpha.tier.expect("alpha has a tier");
        // Nine entries over three superfiles against a budget of four: the
        // depth drops to one for every superfile alike, so each keeps its
        // best document and the tier holds three.
        assert_eq!(tier.inline.len(), 3, "one per superfile under the budget");
        assert!(
            alpha.postings.iter().all(|p| p.tail > 0.0),
            "a truncated tier leaves every superfile a tail"
        );
        let entries = tier_entries_of(&built.root.layers[0], &built.tiers, &tier);
        let mut ids: Vec<(u32, u32)> = entries.iter().map(|e| (e.superfile, e.doc_id)).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![(0, 1), (1, 1), (2, 1)], "each superfile's best");
        // Each superfile's tail is the key of the best document it dropped.
        let tail = |sf: u32| {
            alpha
                .postings
                .iter()
                .find(|p| p.superfile == sf)
                .expect("posting")
                .tail
        };
        assert_eq!(tail(0), key_of(1.2));
        assert_eq!(tail(1), key_of(2.2));
        assert_eq!(tail(2), key_of(3.2));
        let beta = slice
            .run(&fid("body").term_key("beta"))
            .expect("run")
            .expect("beta is indexed");
        let beta_tier = beta.tier.expect("beta has a tier");
        assert!(
            beta.postings.iter().all(|p| p.tail == 0.0),
            "one doc under the budget leaves no tail"
        );
        assert_eq!(
            tier_entries_of(&built.root.layers[0], &built.tiers, &beta_tier).len(),
            1
        );
        // Both tiers are a few entries: they ride in their runs, and the
        // slice has no tier object.
        assert!(tier.chunks.is_empty() && beta_tier.chunks.is_empty());
        let layer = &built.root.layers[0];
        assert_eq!(layer.slices.len(), 1);
        assert!(layer.slices[0].tier.is_none(), "nothing to chunk");
        assert!(built.tiers.is_empty());
    }

    /// Six hundred terms in one contribution, docs scored off `id`: with
    /// slices cut small there are many boundaries to range at, and heads
    /// both inline and chunked.
    fn range_contribution(dir: &TempDir, id: u128) -> Contribution {
        let terms: Vec<String> = (0..600).map(|i| format!("t{i:04}")).collect();
        let docs = |base: f32, n: u32| {
            (0..n)
                .map(|i| (i * 3, base + i as f32 * 0.01))
                .collect::<Vec<_>>()
        };
        let entries: Vec<(&str, &str, Vec<(u32, f32)>)> = terms
            .iter()
            .enumerate()
            .map(|(i, t)| {
                (
                    "body",
                    t.as_str(),
                    docs(id as f32 + i as f32 * 0.001, 2 + (i % 20) as u32),
                )
            })
            .collect();
        contribution_with_heads(dir, id, &entries, 64)
    }

    /// The policy the range tests build and merge under: slices cut small.
    fn range_policy() -> BuildPolicy {
        BuildPolicy {
            slice_target_bytes: 2048,
            tier_entries_per_term: 100_000,
            ..BuildPolicy::default()
        }
    }

    /// Every term of a merged output: its postings and its tier's entries.
    fn merged_terms(
        m: &build::MergedLayer,
    ) -> Vec<(Vec<u8>, Vec<Posting>, Vec<(u32, u32, u32, u8)>)> {
        let mut out = Vec::new();
        for (i, (_, bytes)) in m.slices.iter().enumerate() {
            let bytes = bytes_of(bytes);
            let slice = Slice::open(&bytes).expect("open");
            for (key, run) in slice.terms_after(None, usize::MAX).expect("terms") {
                let mut entries: Vec<tier::TierEntry> = match &run.tier {
                    None => Vec::new(),
                    Some(t) if t.chunks.is_empty() => t.inline.clone(),
                    Some(t) => {
                        let object = m.layer.slices[i].tier.expect("a tier object");
                        let object = bytes_of(
                            &m.tiers
                                .iter()
                                .find(|(h, _)| *h == object)
                                .expect("the tier object is among the built")
                                .1,
                        );
                        let mut all = Vec::new();
                        for chunk in &t.chunks {
                            let range = chunk.offset as usize..(chunk.offset + chunk.len) as usize;
                            let mut block = Vec::new();
                            tier::TierChunk::open(&object[range], chunk.n_blocks)
                                .expect("chunk")
                                .entries(&mut block)
                                .expect("entries");
                            all.extend(block);
                        }
                        all
                    }
                };
                let mut entries: Vec<(u32, u32, u32, u8)> = entries
                    .drain(..)
                    .map(|e| (e.superfile, e.doc_id, e.tf, e.len))
                    .collect();
                entries.sort_unstable();
                out.push((key, run.postings, entries));
            }
        }
        out
    }

    /// A merge of `sources` in four ranges writes the terms the one-writer
    /// merge writes, each with the same postings and tier entries, in
    /// slices that follow each other in key order.
    fn assert_ranges_match_one_writer(
        sources: &[Vec<SliceFile>],
        ordinals: &[Option<u32>],
        policy: &BuildPolicy,
    ) {
        let merge = |ranges: usize| {
            let policy = BuildPolicy {
                merge_ranges: Some(ranges),
                ..policy.clone()
            };
            merge_layers(sources, ordinals, &policy).expect("merge")
        };
        let (one, many) = (merge(1), merge(4));
        assert!(
            many.layer.slices.len() >= 4,
            "{} slices from the ranges",
            many.layer.slices.len()
        );
        let ranges: Vec<(&[u8], &[u8])> = many
            .layer
            .slices
            .iter()
            .map(|s| (s.first_key.as_slice(), s.last_key.as_slice()))
            .collect();
        assert!(
            ranges.windows(2).all(|w| w[0].1 < w[1].0),
            "the ranges' slices follow each other in key order"
        );
        let (a, b) = (merged_terms(&one), merged_terms(&many));
        assert_eq!(a.len(), b.len(), "the same terms");
        assert_eq!(a, b, "the same postings and entries per term");
    }

    /// A build cut into key ranges at the contributions' marks writes the
    /// terms one writer writes: every key's postings and tier entries
    /// agree, and the ranges' slices follow each other in key order.
    #[test]
    fn a_build_in_key_ranges_writes_the_terms_one_writer_writes() {
        let dir = TempDir::new().expect("tempdir");
        let contributions: Vec<Contribution> =
            (1..=3).map(|id| range_contribution(&dir, id)).collect();
        assert!(
            contributions
                .iter()
                .all(|c| c.marks.first().map(|m| m.1) == Some(0) && c.marks.len() >= 8),
            "every contribution marks its first record, and enough others to cut four ranges"
        );
        let build_in = |ranges: usize| {
            let policy = BuildPolicy {
                merge_ranges: Some(ranges),
                ..range_policy()
            };
            let built = build_layer(&contributions, &policy, 0).expect("build");
            build::MergedLayer {
                layer: built.layer,
                slices: built.slices,
                tiers: built.tiers,
            }
        };
        let (one, many) = (build_in(1), build_in(4));
        let bounds: Vec<(&[u8], &[u8])> = many
            .layer
            .slices
            .iter()
            .map(|s| (s.first_key.as_slice(), s.last_key.as_slice()))
            .collect();
        assert!(
            bounds.windows(2).all(|w| w[0].1 < w[1].0),
            "the ranges' slices follow each other in key order"
        );
        assert_ne!(
            one.layer.slices.len(),
            0,
            "the one-writer build writes slices"
        );
        let (a, b) = (merged_terms(&one), merged_terms(&many));
        assert!(!a.is_empty());
        assert_eq!(a.len(), b.len(), "the same terms");
        assert_eq!(a, b, "the same postings and entries per term");
    }

    /// A merge cut into key ranges writes the terms one writer over
    /// everything writes: every key's postings and tier entries agree, and
    /// the ranges' slices follow each other in key order.
    #[test]
    fn a_merge_in_key_ranges_writes_the_terms_one_writer_writes() {
        let dir = TempDir::new().expect("tempdir");
        let policy = range_policy();
        let built = build(
            &[range_contribution(&dir, 1), range_contribution(&dir, 2)],
            &policy,
        )
        .expect("build");
        let layer = built.root.layers[0].clone();
        assert!(layer.slices.len() >= 8, "{} slices", layer.slices.len());
        let scratch = TempDir::new().expect("scratch");
        spill_tiers(scratch.path(), &layer.slices, built.tiers).expect("place tiers");
        let sources =
            vec![spill_slices(scratch.path(), &layer.slices, built.slices).expect("place slices")];
        assert_ranges_match_one_writer(&sources, &[Some(0), Some(1)], &policy);
    }

    /// The same over two source layers, a base and a delta built onto
    /// it, whose slices are cut at different keys: each range takes the
    /// slices of each layer that reach into it and interleaves them.
    #[test]
    fn a_merge_in_key_ranges_over_two_layers_writes_the_terms_one_writer_writes() {
        let dir = TempDir::new().expect("tempdir");
        let policy = range_policy();
        let base = build(&[range_contribution(&dir, 1)], &policy).expect("base");
        let delta = build_onto(
            base.root.clone(),
            &[range_contribution(&dir, 2), range_contribution(&dir, 3)],
            &policy,
        )
        .expect("delta");
        let (base_refs, delta_refs) = (&base.root.layers[0].slices, &delta.root.layers[1].slices);
        assert!(base_refs.len() >= 4 && delta_refs.len() >= 4);
        assert_ne!(
            base_refs.iter().map(|s| &s.first_key).collect::<Vec<_>>(),
            delta_refs.iter().map(|s| &s.first_key).collect::<Vec<_>>(),
            "the layers' slices are cut at different keys"
        );
        let scratch = TempDir::new().expect("scratch");
        spill_tiers(scratch.path(), base_refs, base.tiers).expect("place base tiers");
        spill_tiers(scratch.path(), delta_refs, delta.tiers).expect("place delta tiers");
        let sources = vec![
            spill_slices(scratch.path(), base_refs, base.slices).expect("place base"),
            spill_slices(scratch.path(), delta_refs, delta.slices).expect("place delta"),
        ];
        assert_ranges_match_one_writer(&sources, &[Some(0), Some(1), Some(2)], &policy);
    }

    /// A tier of more than a few entries is chunked into the slice's tier
    /// object; one of a few rides in the run. A merge keeps each where it
    /// belongs and renumbers the inline entries like the chunked ones.
    #[test]
    fn a_tier_of_a_few_entries_rides_in_the_run() {
        let dir = TempDir::new().expect("tempdir");
        let docs = |base: f32, n: u32| {
            (0..n)
                .map(|i| (i * 3, base + i as f32 * 0.01))
                .collect::<Vec<_>>()
        };
        let per_file = (TIER_INLINE_MAX / 2 + 1) as u32;
        let make = |id: u128| {
            contribution_with_heads(
                &dir,
                id,
                &[
                    ("body", "common", docs(id as f32, per_file)),
                    ("body", "rare", docs(0.5, 2)),
                ],
                1000,
            )
        };
        let policy = BuildPolicy {
            slice_target_bytes: usize::MAX,
            tier_entries_per_term: 1_000,
            ..BuildPolicy::default()
        };
        let built = build(&[make(1), make(2)], &policy).expect("build");
        let slice_bytes = bytes_of(&built.slices[0].1);
        let slice = Slice::open(&slice_bytes).expect("open");
        let run_of = |term: &str| {
            slice
                .run(&fid("body").term_key(term))
                .expect("run")
                .expect("indexed")
        };
        let common = run_of("common").tier.expect("tier");
        let rare = run_of("rare").tier.expect("tier");
        assert!(
            !common.chunks.is_empty() && common.inline.is_empty(),
            "more than a few entries are chunked"
        );
        assert!(
            rare.chunks.is_empty() && rare.inline.len() == 4,
            "a few entries ride in the run: {rare:?}"
        );
        assert!(
            run_of("rare")
                .postings
                .iter()
                .chain(run_of("common").postings.iter())
                .all(|p| p.tail == 0.0),
            "whole heads leave no tail"
        );
        let layer = &built.root.layers[0];
        assert_eq!(
            built.tiers.len(),
            1,
            "one tier object, for the chunked tier"
        );
        assert_eq!(
            tier_entries_of(layer, &built.tiers, &common).len(),
            2 * per_file as usize
        );
        let mut inline_ids: Vec<(u32, u32)> = rare
            .inline
            .iter()
            .map(|e| (e.superfile, e.doc_id))
            .collect();
        inline_ids.sort_unstable();
        assert_eq!(inline_ids, vec![(0, 0), (0, 3), (1, 0), (1, 3)]);

        // A merge that drops the first superfile renumbers the second to
        // ordinal 0 in the inline entries as in the chunked ones.
        let scratch = TempDir::new().expect("scratch");
        spill_tiers(scratch.path(), &layer.slices, built.tiers).expect("place tiers");
        let sources =
            vec![spill_slices(scratch.path(), &layer.slices, built.slices).expect("place slices")];
        let merged = merge_layers(&sources, &[None, Some(0)], &policy).expect("merge");
        let merged_bytes = bytes_of(&merged.slices[0].1);
        let merged_slice = Slice::open(&merged_bytes).expect("open");
        let rare = merged_slice
            .run(&fid("body").term_key("rare"))
            .expect("run")
            .expect("indexed")
            .tier
            .expect("tier");
        let mut ids: Vec<(u32, u32)> = rare
            .inline
            .iter()
            .map(|e| (e.superfile, e.doc_id))
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![(0, 0), (0, 3)]);
        let common = merged_slice
            .run(&fid("body").term_key("common"))
            .expect("run")
            .expect("indexed")
            .tier
            .expect("tier");
        assert_eq!(
            common.chunks.is_empty(),
            per_file as usize <= TIER_INLINE_MAX,
            "the merged tier is placed by its merged size"
        );
        let entries = tier_entries_of(&merged.layer, &merged.tiers, &common);
        assert!(entries.iter().all(|e| e.superfile == 0));
        assert_eq!(entries.len(), per_file as usize);
    }

    /// A build that spills writes each slice and tier object to a scratch
    /// file as it is cut and keeps none of their bytes; what it spills is
    /// byte for byte what a resident build keeps, and a merge over the
    /// spilled files reads them where it would find downloaded ones.
    #[test]
    fn a_spilling_build_writes_what_a_resident_build_keeps() {
        let dir = TempDir::new().expect("tempdir");
        let docs = |base: f32, n: u32| {
            (0..n)
                .map(|i| (i * 3, base + i as f32 * 0.01))
                .collect::<Vec<_>>()
        };
        let make = |id: u128, n: u32| {
            contribution_with_heads(
                &dir,
                id,
                &[
                    ("body", "alpha", docs(id as f32, n)),
                    ("body", "beta", docs(0.5, 2)),
                    ("body", "gamma", docs(1.5, n)),
                    ("body", "delta", docs(2.5, 3)),
                ],
                1000,
            )
        };
        let resident = BuildPolicy {
            slice_target_bytes: 48,
            tier_entries_per_term: 1_000,
            ..BuildPolicy::default()
        };
        let spilling = BuildPolicy {
            spill_to_disk: true,
            ..resident.clone()
        };
        let kept = build(&[make(1, 60), make(2, 60)], &resident).expect("resident build");
        let spilled = build(&[make(1, 60), make(2, 60)], &spilling).expect("spilling build");
        assert!(kept.slices.len() > 1, "several slices, so several files");
        assert_eq!(kept.slices, spilled.slices, "same slices, byte for byte");
        assert_eq!(
            kept.tiers, spilled.tiers,
            "same tier objects, byte for byte"
        );
        assert!(
            spilled
                .slices
                .iter()
                .chain(&spilled.tiers)
                .all(|(_, b)| matches!(b, BuiltBytes::Spilled { .. })),
            "every object of the spilling build is on disk"
        );
        assert!(
            kept.slices
                .iter()
                .chain(&kept.tiers)
                .all(|(_, b)| matches!(b, BuiltBytes::Resident(_))),
            "every object of the resident build is in memory"
        );
        // The spilled files feed a merge as downloaded ones would.
        let scratch = TempDir::new().expect("scratch");
        spill_tiers(
            scratch.path(),
            &spilled.root.layers[0].slices,
            spilled.tiers,
        )
        .expect("place tiers");
        let sources = vec![
            spill_slices(
                scratch.path(),
                &spilled.root.layers[0].slices,
                spilled.slices,
            )
            .expect("place slices"),
        ];
        let merged = merge_layers(&sources, &[Some(0), Some(1)], &resident).expect("merge");
        assert_eq!(
            merged.slices, kept.slices,
            "a merge of the spilled files rebuilds the same slices"
        );
    }

    #[test]
    fn merging_layers_with_tiers_writes_what_a_fresh_build_writes() {
        let dir = TempDir::new().expect("tempdir");
        let docs = |base: f32, n: u32| {
            (0..n)
                .map(|i| (i * 3, base + i as f32 * 0.01))
                .collect::<Vec<_>>()
        };
        let make = |id: u128, n: u32| {
            contribution_with_heads(
                &dir,
                id,
                &[
                    ("body", "alpha", docs(id as f32, n)),
                    ("body", "gamma", docs(0.5, 2)),
                ],
                1000,
            )
        };
        // No source truncates at this budget, so the merge sees every entry
        // a fresh build would and must write the same bytes.
        let policy = BuildPolicy {
            slice_target_bytes: 4096,
            tier_entries_per_term: 1_000,
            ..BuildPolicy::default()
        };
        let base = build(&[make(1, 60), make(2, 60)], &policy).expect("base");
        let delta = build_onto(base.root.clone(), &[make(3, 90)], &policy).expect("delta");
        spill_tiers(dir.path(), &base.root.layers[0].slices, base.tiers.clone())
            .expect("spill base tiers");
        spill_tiers(
            dir.path(),
            &delta.root.layers[1].slices,
            delta.tiers.clone(),
        )
        .expect("spill delta tiers");
        let sources = vec![
            spill_slices(dir.path(), &base.root.layers[0].slices, base.slices).expect("spill base"),
            spill_slices(dir.path(), &delta.root.layers[1].slices, delta.slices)
                .expect("spill delta"),
        ];
        // Drop the middle superfile; the merge renumbers the survivors.
        let merged = merge_layers(&sources, &[Some(0), None, Some(1)], &policy).expect("merge");
        let expected = build(&[make(1, 60), make(3, 90)], &policy).expect("survivors");
        assert_eq!(merged.slices, expected.slices, "slices are byte-identical");
        assert_eq!(
            merged.tiers, expected.tiers,
            "the tier objects are byte-identical"
        );
        assert_eq!(merged.layer, expected.root.layers[0]);
    }

    /// A source that already truncated cannot give a merge the entries it
    /// dropped, so a merge of truncated sources is not a fresh build; what
    /// it must still be is sound: every posting's tail bounds the keys of
    /// that superfile's documents the merged tier does not hold.
    #[test]
    fn merging_truncated_tiers_keeps_every_tail_an_upper_bound() {
        let dir = TempDir::new().expect("tempdir");
        let docs = |base: f32, n: u32| {
            (0..n)
                .map(|i| (i * 3, base + i as f32 * 0.01))
                .collect::<Vec<_>>()
        };
        let heads: Vec<(u128, Vec<(u32, f32)>)> =
            vec![(1, docs(1.0, 60)), (2, docs(2.0, 60)), (3, docs(1.5, 90))];
        let make = |id: u128| {
            let d = &heads.iter().find(|h| h.0 == id).expect("heads").1;
            contribution_with_heads(&dir, id, &[("body", "alpha", d.clone())], 1000)
        };
        let policy = BuildPolicy {
            slice_target_bytes: 4096,
            tier_entries_per_term: 40,
            ..BuildPolicy::default()
        };
        let base = build(&[make(1), make(2)], &policy).expect("base");
        let delta = build_onto(base.root.clone(), &[make(3)], &policy).expect("delta");
        spill_tiers(dir.path(), &base.root.layers[0].slices, base.tiers.clone())
            .expect("spill base tiers");
        spill_tiers(
            dir.path(),
            &delta.root.layers[1].slices,
            delta.tiers.clone(),
        )
        .expect("spill delta tiers");
        let sources = vec![
            spill_slices(dir.path(), &base.root.layers[0].slices, base.slices).expect("spill base"),
            spill_slices(dir.path(), &delta.root.layers[1].slices, delta.slices)
                .expect("spill delta"),
        ];
        let merged = merge_layers(&sources, &[Some(0), None, Some(1)], &policy).expect("merge");
        let slice_bytes = bytes_of(&merged.slices[0].1);
        let slice = Slice::open(&slice_bytes).expect("open");
        let run = slice
            .run(&fid("body").term_key("alpha"))
            .expect("run")
            .expect("alpha");
        let tier = run.tier.expect("tier");
        assert!(
            run.postings.iter().any(|p| p.tail > 0.0),
            "a truncated tier leaves a tail"
        );
        let held: HashSet<(u32, u32)> = tier_entries_of(&merged.layer, &merged.tiers, &tier)
            .iter()
            .map(|e| (e.superfile, e.doc_id))
            .collect();
        assert_eq!(held.len(), 40);
        // New ordinal 0 is superfile 1, new ordinal 1 is superfile 3.
        for (ordinal, id) in [(0u32, 1u128), (1, 3)] {
            let posting = run
                .postings
                .iter()
                .find(|p| p.superfile == ordinal)
                .expect("posting");
            for &(row, w) in &heads.iter().find(|h| h.0 == id).expect("heads").1 {
                if !held.contains(&(ordinal, row)) {
                    assert!(
                        posting.tail >= key_of(w),
                        "superfile {id} row {row}: key {} above tail {}",
                        key_of(w),
                        posting.tail
                    );
                }
            }
        }
    }

    #[test]
    fn contribution_rejects_out_of_order_keys() {
        let dir = TempDir::new().expect("tempdir");
        let mut w = ContributionWriter::create(dir.path(), Uuid::from_u128(1), 0).expect("create");
        w.push(b"b", 1, 1.0, Location::None, 0.0, &[])
            .expect("first");
        assert!(matches!(
            w.push(b"a", 1, 1.0, Location::None, 0.0, &[]),
            Err(TermIndexError::Build(m)) if m.contains("ascending")
        ));
        assert!(
            w.push(b"b", 1, 1.0, Location::None, 0.0, &[]).is_err(),
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
            ..BuildPolicy::default()
        };
        let built = build(&[a, b, c], &policy).expect("build");
        assert_eq!(
            built.root.superfiles,
            vec![Uuid::from_u128(1), Uuid::from_u128(2), Uuid::from_u128(3)]
        );
        assert_eq!(built.root.layers.len(), 1);
        assert_eq!(built.slices.len(), 1, "everything fits one slice");
        let slice_bytes = bytes_of(&built.slices[0].1);
        let slice = Slice::open(&slice_bytes).expect("open");

        let alpha = slice
            .postings(&fid("body").term_key("alpha"))
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
            alpha.iter().all(|p| p.location != Location::None),
            "a term in every superfile keeps its location in each"
        );

        let beta = slice
            .postings(&fid("body").term_key("beta"))
            .expect("ok")
            .expect("present");
        assert_eq!(beta.len(), 1);
        assert_eq!(beta[0].superfile, 0);
        assert!(
            matches!(beta[0].location, Location::Inline { .. }),
            "below the threshold: location kept"
        );

        // `contribution` places a term's postings by its rank among the
        // contribution's keys, so the expectation ranks the same keys.
        let mut a_keys = vec![
            fid("body").term_key("alpha"),
            fid("body").term_key("beta"),
            fid("title").term_key("zed"),
        ];
        a_keys.sort();
        let zed_key = fid("title").term_key("zed");
        let zed_rank = a_keys.iter().position(|k| *k == zed_key).expect("zed") as u64;
        let zed = slice.postings(&zed_key).expect("ok").expect("present");
        assert!(matches!(
            zed[0].location,
            Location::Pfor { offset, len: 800 } if offset == zed_rank * 1000
        ));

        assert_eq!(
            slice.postings(&fid("body").term_key("delta")).expect("ok"),
            None
        );
        let s = &built.root.layers[0].slices[0];
        let mut all_keys = a_keys;
        all_keys.push(fid("body").term_key("gamma"));
        all_keys.sort();
        assert_eq!(s.first_key, all_keys[0]);
        assert_eq!(s.last_key, all_keys[3]);
        assert_eq!(
            s.content_hash,
            ContentHash::of(&bytes_of(&built.slices[0].1))
        );
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
            ..BuildPolicy::default()
        };
        let built = build(&[a], &policy).expect("build");
        let slices = &built.root.layers[0].slices;
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
        let by_hash: HashMap<_, _> = built
            .slices
            .iter()
            .map(|(h, b)| (*h, bytes_of(b)))
            .collect();
        for (c, t, d) in &terms {
            let key = fid(c).term_key(t);
            let hits: Vec<_> = built.root.slices_for_key(&key).collect();
            assert_eq!(hits.len(), 1, "exactly one slice can hold {t}");
            let slice = Slice::open(&by_hash[&hits[0].content_hash]).expect("open");
            let run = slice.postings(&key).expect("ok").expect("present");
            assert_eq!(run[0].df, *d);
        }
    }

    /// Merging layers while dropping a superfile and renumbering the rest
    /// writes exactly what building the survivors directly writes, slice
    /// cuts included.
    #[test]
    fn merging_layers_matches_building_the_survivors() {
        let dir = TempDir::new().expect("tempdir");
        let terms = |range: Range<u64>, step: usize| -> Vec<(String, u64)> {
            range
                .step_by(step)
                .map(|i| (format!("t{i:03}"), i % 200 + 1))
                .collect()
        };
        let make = |id: u128, owned: Vec<(String, u64)>| {
            let terms: Vec<(&str, &str, u64)> = owned
                .iter()
                .map(|(t, df)| ("body", t.as_str(), *df))
                .collect();
            contribution(&dir, id, &terms)
        };
        let first = make(1, terms(0..60, 1));
        let gone = make(2, terms(30..90, 1));
        let last = make(3, terms(0..90, 2));
        // Small slices, so the merge has to cut where the build cuts.
        let policy = BuildPolicy {
            slice_target_bytes: 256,
            ..BuildPolicy::default()
        };

        let base = build(&[first, gone], &policy).expect("base");
        let delta = build_onto(base.root.clone(), &[last], &policy).expect("delta");
        let sources = vec![
            spill_slices(dir.path(), &base.root.layers[0].slices, base.slices).expect("spill base"),
            spill_slices(dir.path(), &delta.root.layers[1].slices, delta.slices)
                .expect("spill delta"),
        ];
        let merged = merge_layers(&sources, &[Some(0), None, Some(1)], &policy).expect("merge");
        let (layer, slices) = (merged.layer, merged.slices);

        let first = make(1, terms(0..60, 1));
        let last = make(3, terms(0..90, 2));
        let expected = build(&[first, last], &policy).expect("survivors");
        assert!(
            expected.slices.len() > 1,
            "the fixture spans several slices"
        );
        assert_eq!(layer, expected.root.layers[0]);
        assert_eq!(slices, expected.slices);
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
            ..BuildPolicy::default()
        };
        let built = build(&[a, b], &policy).expect("build");
        let n_slices = built.root.layers[0].slices.len();
        assert!(n_slices > 1);

        let store_dir = TempDir::new().expect("store dir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local fs"));
        let reference = write_built(&storage, built).await.expect("write");
        assert!(reference.uri.starts_with("term-index/root-"));
        let root = load_root(storage.as_ref(), &reference).await.expect("load");
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        assert_eq!(
            index.root().superfiles,
            vec![Uuid::from_u128(7), Uuid::from_u128(8)]
        );

        let run = index.postings(fid("body"), "ab005").await.expect("ok");
        assert_eq!(run.len(), 2, "both superfiles hold ab005");
        assert_eq!((run[0].superfile, run[0].df), (0, 2));
        assert_eq!((run[1].superfile, run[1].df), (1, 9));
        assert_eq!(index.superfile_id(1), Some(Uuid::from_u128(8)));
        assert!(
            index
                .postings(fid("body"), "nope")
                .await
                .expect("ok")
                .is_empty()
        );
        assert!(
            index
                .postings(fid("title"), "ab005")
                .await
                .expect("ok")
                .is_empty(),
            "column is part of the key"
        );

        let mut seen = 0usize;
        index
            .for_each_prefix(fid("body"), "ab", |_, _| {
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
            .for_each_prefix(fid("body"), "ab", |_, _| {
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
    const DOCS_PER_LAYER: usize = 40;
    const LAYERS: usize = 3;

    fn title_schema() -> Arc<arrow_schema::Schema> {
        use arrow_schema::{DataType, Field, Schema};
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]))
    }

    /// The id the table mints for `title_schema`'s one column.
    fn title_id() -> FieldId {
        TableSchema::from_user_schema(&title_schema())
            .id_of("title")
            .expect("title is a table column")
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

    /// Commit one layer: every title holds `shared`; layer `s` holds
    /// `alpha` in the titles whose index is a multiple of `s + 2`. Returns
    /// the layer's `alpha` count.
    fn commit_layer(st: &crate::supertable::Supertable, layer: usize) -> u64 {
        use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
        let titles: Vec<String> = (0..DOCS_PER_LAYER)
            .map(|i| {
                let topic = if i % (layer + 2) == 0 {
                    "alpha"
                } else {
                    "beta"
                };
                format!("{topic} shared s{layer}d{i:02}")
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

    /// A term index written by the layout before tiers — a root at its
    /// version naming slices at theirs — serves ceilings as that layout
    /// did: each stored score over the superfile's own idf, times the
    /// query's idf. Nothing opens more than it did before the upgrade.
    #[test]
    fn a_term_index_of_the_layout_before_tiers_serves_ceilings_as_before() {
        let (_dir, storage, st) = fresh_table();
        commit_layer(&st, 0);
        commit_layer(&st, 1);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let reader = st.reader().expect("reader");
        let entries: Vec<Arc<SuperfileEntry>> = reader.manifest().get_all_superfiles().to_vec();
        assert_eq!(entries.len(), 2);
        let key = title_id().term_key("shared");
        let slice = format::encode_legacy_slice(&[(
            key.clone(),
            vec![(0, 5, 2.5, Location::None), (1, 7, 1.9, Location::None)],
        )]);
        let hash = ContentHash::of(&slice);
        rt.block_on(put_content_addressed(
            storage.as_ref(),
            "slice",
            slice.clone(),
        ))
        .expect("slice stored");
        let root = Root {
            superfiles: entries.iter().map(|e| e.superfile_id).collect(),
            id_mins: entries.iter().map(|e| e.id_min).collect(),
            layers: vec![Layer {
                slices: vec![format::SliceRef {
                    first_key: key.clone(),
                    last_key: key.clone(),
                    content_hash: hash,
                    tier: None,
                    len: 0,
                }],
            }],
        };
        let root = Root::decode(&format::encode_legacy_root(&root, &[slice.len() as u64]))
            .expect("a legacy root decodes");
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let idf_used = |_: &str| 1.7f32;
        let ceilings = rt
            .block_on(index.query_ceilings(title_id(), &["shared"], &[], &entries, &idf_used))
            .expect("ceilings");
        for (e, (df, bound)) in entries.iter().zip([(5u64, 2.5f32), (7, 1.9)]) {
            let n = e
                .fts_summary
                .get(&title_id())
                .and_then(|s| s.length_stats.as_ref().map(|l| l.n_scored_docs))
                .unwrap_or(e.n_docs);
            let expected = bound / bm25::idf(n, df.min(n)) * 1.7 * CEILING_SLACK;
            let got = ceilings[&e.superfile_id];
            assert!(
                (got - expected).abs() <= expected * 1e-5,
                "superfile {}: ceiling {got}, expected {expected}",
                e.superfile_id
            );
        }
        // Its runs serve postings and are not mistaken for tiers.
        let runs = rt
            .block_on(index.runs_many(title_id(), &["shared"]))
            .expect("runs")
            .remove(0);
        assert!(runs.len() == 1 && runs[0].run.legacy_bound && runs[0].tier_object.is_none());
    }

    /// A merge that rewrites a layer of the layout before tiers writes
    /// its bounds as the keys the current layout stores, through each
    /// superfile's scored document count.
    #[test]
    fn a_merge_writes_a_legacy_layers_bounds_as_keys() {
        let key = title_id().term_key("shared");
        let slice = format::encode_legacy_slice(&[(
            key.clone(),
            vec![(0, 5, 2.5, Location::None), (1, 7, 1.9, Location::None)],
        )]);
        let scratch = TempDir::new().expect("scratch");
        let path = scratch.path().join("legacy.slice");
        fs::write(&path, &slice).expect("place slice");
        let sources = vec![vec![build::SliceFile {
            path,
            first_key: key.clone(),
            last_key: key.clone(),
        }]];
        let scoring = |n: u64| {
            std::collections::BTreeMap::from([(
                title_id(),
                tier::ColumnScoring {
                    scored_docs: n,
                    ..tier::ColumnScoring::default()
                },
            )])
        };
        let policy = BuildPolicy {
            scoring_by_ordinal: Arc::new(vec![scoring(100), scoring(1000)]),
            ..BuildPolicy::default()
        };
        let merged = merge_layers(&sources, &[Some(0), Some(1)], &policy).expect("merge");
        let bytes = bytes_of(&merged.slices[0].1);
        let run = Slice::open(&bytes)
            .expect("open")
            .run(&key)
            .expect("run")
            .expect("indexed");
        assert!(!run.legacy_bound, "written on in the current layout");
        let keys: Vec<f32> = run.postings.iter().map(|p| p.bound).collect();
        assert_eq!(
            keys,
            vec![
                legacy_bound_as_key(2.5, 5, 100),
                legacy_bound_as_key(1.9, 7, 1000)
            ]
        );
        assert!(
            run.postings.iter().all(|p| p.tail == f32::INFINITY),
            "nothing below a legacy bound is known"
        );
    }

    /// Optimize options that merge nothing: the maintenance passes alone.
    fn maintenance_only_options() -> OptimizeOptions {
        OptimizeOptions::compact(CompactionSettings {
            min_fill_percent: 100,
            min_superfiles_for_merge: u64::MAX,
            ..CompactionSettings::default()
        })
    }

    /// Compaction-free optimize: the maintenance passes alone.
    fn maintenance_only_optimize(st: &crate::supertable::Supertable) {
        st.optimize(&maintenance_only_options()).expect("optimize");
    }

    /// The live superfile ids and the root's covered set, for comparison.
    /// The list the table's pointer names right now, decoded.
    fn current_list(
        storage: &Arc<dyn StorageProvider>,
        legacy: &LegacyNames,
        rt: &tokio::runtime::Runtime,
    ) -> list::Manifest {
        rt.block_on(async {
            let (pointer, _) = read_pointer(storage.as_ref())
                .await
                .expect("pointer")
                .expect("the table has a pointer");
            let (bytes, _) = storage
                .get(&pointer.manifest_uri)
                .await
                .expect("list bytes");
            list::decode(&bytes, legacy).expect("decode")
        })
    }

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

    /// A fragmented three-layer FTS table, optimized with compaction
    /// disabled so only the maintenance passes run. Returns the storage
    /// root, the table, and the per-layer `alpha` counts.
    fn optimized_fragmented_table() -> (
        TempDir,
        Arc<dyn StorageProvider>,
        crate::supertable::Supertable,
        Vec<u64>,
    ) {
        let (dir, storage, st) = fresh_table();
        let alpha_per_layer: Vec<u64> = (0..LAYERS).map(|s| commit_layer(&st, s)).collect();
        assert!(
            st.reader().expect("reader").n_superfiles() >= LAYERS,
            "fixture must stay fragmented"
        );
        maintenance_only_optimize(&st);
        (dir, storage, st, alpha_per_layer)
    }

    /// Every commit publishes its superfiles' postings in the same manifest
    /// as the entries: after each commit the root covers exactly the live
    /// set, one delta layer per commit, with per-superfile `df` right —
    /// and a later optimize folds the deltas into one base layer that
    /// answers identically.
    #[test]
    fn commit_publishes_postings_with_the_manifest() {
        let (_dir, storage, st) = fresh_table();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let mut alphas = Vec::new();
        for layer in 0..LAYERS {
            alphas.push(commit_layer(&st, layer));
            let (live, root) = live_and_covered(&st, &storage, &rt);
            let covered: std::collections::HashSet<Uuid> =
                root.superfiles.iter().copied().collect();
            assert_eq!(
                covered, live,
                "after commit {layer}: every visible superfile has postings, and only those"
            );
            assert_eq!(root.layers.len(), layer + 1, "one delta layer per commit");
            let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
            let shared = rt
                .block_on(index.postings(title_id(), "shared"))
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
        // Optimize folds every delta into one base layer with the same answers.
        let before: Vec<u64> = {
            let (_, root) = live_and_covered(&st, &storage, &rt);
            let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
            let mut v: Vec<u64> = rt
                .block_on(index.postings(title_id(), "alpha"))
                .expect("lookup")
                .iter()
                .map(|p| p.df)
                .collect();
            v.sort_unstable();
            v
        };
        maintenance_only_optimize(&st);
        let (live, root) = live_and_covered(&st, &storage, &rt);
        assert_eq!(root.layers.len(), 1, "optimize rebuilds one base layer");
        assert_eq!(
            root.superfiles
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>(),
            live
        );
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let mut after: Vec<u64> = rt
            .block_on(index.postings(title_id(), "alpha"))
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
        let (_dir, storage, st, mut alpha_per_layer) = optimized_fragmented_table();
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
            root.layers.len(),
            1,
            "a maintenance build is one base layer"
        );

        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let shared = rt
            .block_on(index.postings(title_id(), "shared"))
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
            .block_on(index.postings(title_id(), "alpha"))
            .expect("lookup")
            .iter()
            .map(|p| p.df)
            .collect();
        alpha.sort_unstable();
        alpha_per_layer.sort_unstable();
        assert_eq!(
            alpha, alpha_per_layer,
            "per-superfile df for `alpha` matches the fixture"
        );
        assert!(
            rt.block_on(index.postings(title_id(), "absent"))
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
            .layers
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
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        let (live_before, root_before) = live_and_covered(&st, &storage, &rt);
        assert!(live_before.len() >= LAYERS);
        let layers_before = root_before.layers.len();
        // `s0d00` names one title in layer 0 — a superfile compaction
        // removes; `alpha` spans every layer.
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
            root_after.layers.len(),
            layers_before + 1,
            "the compaction commit appended one delta"
        );

        // The merged superfile's postings are right, and the removed
        // inputs' postings are still there to be filtered out by liveness.
        let index = TermIndex::new(root_after, String::new(), Arc::clone(&storage), None);
        let shared = rt
            .block_on(index.postings(title_id(), "shared"))
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
                DOCS_PER_LAYER * LAYERS,
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

        use crate::{Bm25SearchOptions, supertable::Supertable};

        let (_dir, storage, st) = table_with(|o| {
            o.with_eager_load_threshold(0)
                .with_target_superfiles_per_part(1)
        });
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
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
            .expect("the index loads")
            .expect("the table has an index");
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
                Bm25SearchOptions::new(),
                Some(&["_id", "score"]),
            )
            .expect("search");
        assert_eq!(
            hits_of(&batches).len(),
            1,
            "the replacement row is found through the index"
        );
    }

    /// A manifest entry for a superfile no commit built postings for.
    fn unindexed_entry() -> Arc<SuperfileEntry> {
        use crate::supertable::manifest::{SuperfileUri, VectorLayout};

        Arc::new(SuperfileEntry {
            physical_schema: None,
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
        })
    }

    /// Commit a membership change that carries no postings: `add` joins
    /// the table, `remove` leaves it.
    fn commit_without_postings(
        st: &crate::supertable::Supertable,
        storage: &Arc<dyn StorageProvider>,
        add: Vec<Arc<SuperfileEntry>>,
        remove: &[Arc<SuperfileEntry>],
    ) -> Arc<crate::supertable::manifest::ManifestSnapshot> {
        use crate::supertable::writer::{CommitListMetadata, persist_commit_async};

        st.block_on_query(persist_commit_async(
            st.inner(),
            Arc::clone(storage),
            add,
            remove,
            Vec::new(),
            Vec::new(),
            CommitListMetadata::empty(),
            Vec::new(),
        ))
        .expect("commit")
    }

    /// A commit that publishes a superfile without postings — one whose
    /// path built no contribution — leaves the index unable to route to
    /// it, so the index must stop claiming to list every live superfile.
    #[test]
    fn a_superfile_published_without_postings_marks_the_index_incomplete() {
        let (_dir, storage, st) = fresh_table();
        commit_layer(&st, 0);
        assert!(
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_complete()
        );
        let committed = commit_without_postings(&st, &storage, vec![unindexed_entry()], &[]);
        assert!(
            committed.term_index_ref().is_some(),
            "the prior root is carried forward"
        );
        assert!(
            !committed.term_index_complete(),
            "an unindexed live superfile makes the index incomplete"
        );
    }

    /// An index marked incomplete that already lists every live superfile
    /// is marked complete by the next maintenance rebuild, even though the
    /// rebuild produces the very root the manifest already references.
    ///
    /// The rebuild's root is content-addressed, so "nothing changed" and
    /// "the flag is still wrong" look the same from the reference alone. A
    /// rebuild that stopped there would leave the table routing by part
    /// summaries, and gathering global idf by opening superfiles, for as
    /// long as its membership stayed put.
    #[test]
    fn a_rebuild_matching_the_current_root_marks_the_index_complete() {
        let (_dir, storage, st) = fresh_table();
        commit_layer(&st, 0);
        let reference = st
            .reader()
            .expect("reader")
            .manifest()
            .term_index_ref()
            .cloned()
            .expect("the first commit publishes a root");

        // An unindexed superfile marks the index incomplete; removing it
        // leaves the index listing every live superfile while still marked
        // incomplete, with the root unchanged throughout.
        let entry = unindexed_entry();
        commit_without_postings(&st, &storage, vec![Arc::clone(&entry)], &[]);
        let stuck = commit_without_postings(&st, &storage, Vec::new(), &[entry]);
        // A direct commit does not advance the handle; adopt it, as a later
        // open would.
        st.block_on_query(st.refresh()).expect("refresh");
        assert_eq!(stuck.term_index_ref(), Some(&reference));
        assert!(!stuck.term_index_complete());
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (live, root) = live_and_covered(&st, &storage, &rt);
        let covered: std::collections::HashSet<Uuid> = root.superfiles.iter().copied().collect();
        assert_eq!(
            covered, live,
            "the index already lists every live superfile"
        );

        maintenance_only_optimize(&st);
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        assert_eq!(
            manifest.term_index_ref(),
            Some(&reference),
            "the rebuild reproduces the root it found"
        );
        assert!(
            manifest.term_index_complete(),
            "a rebuild over the whole membership marks the index complete"
        );
    }

    /// An empty FTS table whose storage can fail chosen operations.
    fn fault_table() -> (
        TempDir,
        Arc<FaultStorage>,
        Arc<dyn StorageProvider>,
        crate::supertable::Supertable,
    ) {
        let dir = TempDir::new().expect("tempdir");
        let inner: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs"));
        let faults = FaultStorage::wrap(inner);
        let storage: Arc<dyn StorageProvider> = Arc::clone(&faults) as Arc<dyn StorageProvider>;
        let st = crate::supertable::Supertable::create(fresh_options(&storage)).expect("create");
        (dir, faults, storage, st)
    }

    /// Run a compaction-free optimize with term-index writes failing, and
    /// return whether it tried to write one, i.e. whether it rebuilt the
    /// index. A rebuild always writes, even when it reproduces the same
    /// content-addressed objects.
    fn optimize_rebuilds_term_index(
        st: &crate::supertable::Supertable,
        faults: &FaultStorage,
    ) -> bool {
        let fired_before = faults.fired();
        faults.fail(FaultOp::PutAtomic, STORAGE_PREFIX, 1);
        let result = st.optimize(&maintenance_only_options());
        faults.clear();
        let rebuilt = faults.fired() > fired_before;
        if !rebuilt {
            result.expect("optimize");
        }
        rebuilt
    }

    /// A fragmented table, optimized once so its index is one layer over
    /// exactly the live superfiles.
    fn optimized_fault_table() -> (
        TempDir,
        Arc<FaultStorage>,
        Arc<dyn StorageProvider>,
        crate::supertable::Supertable,
    ) {
        let (dir, faults, storage, st) = fault_table();
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        maintenance_only_optimize(&st);
        (dir, faults, storage, st)
    }

    /// A second optimize over an unchanged table finds the index already in
    /// the shape a rebuild would produce, and writes nothing.
    #[test]
    fn a_repeat_optimize_skips_the_term_index_rebuild() {
        let (_dir, faults, _storage, st) = optimized_fault_table();
        let reference = st
            .reader()
            .expect("reader")
            .manifest()
            .term_index_ref()
            .cloned();
        assert!(
            !optimize_rebuilds_term_index(&st, &faults),
            "nothing changed, so nothing is rebuilt"
        );
        assert_eq!(
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_ref()
                .cloned(),
            reference
        );
    }

    /// An append after the last rebuild adds a delta layer, which the next
    /// optimize folds back in.
    #[test]
    fn an_appended_delta_layer_triggers_the_rebuild() {
        let (_dir, faults, storage, st) = optimized_fault_table();
        commit_layer(&st, LAYERS);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        assert_eq!(root.layers.len(), 2, "the append added a delta");
        assert!(optimize_rebuilds_term_index(&st, &faults));
    }

    #[test]
    fn optimize_keeps_the_delta_layer_when_the_refresh_step_is_off() {
        let (_dir, _faults, storage, st) = optimized_fault_table();
        commit_layer(&st, LAYERS);
        let steps = OptimizeSteps {
            refresh_term_index: false,
            ..OptimizeSteps::default()
        };
        st.optimize(&maintenance_only_options().with_steps(steps))
            .expect("optimize");
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        assert_eq!(root.layers.len(), 2, "the delta must not be folded in");
    }

    /// A one-slice layer, told apart by `tag`.
    fn tagged_layer(tag: u8) -> Layer {
        Layer {
            slices: vec![SliceRef {
                first_key: vec![tag],
                last_key: vec![tag],
                content_hash: ContentHash::of(&[tag]),
                tier: None,
                len: 0,
            }],
        }
    }

    /// A root listing `ids` (each its own `id_min`; `0` is padding) over
    /// one tagged layer per `tags`.
    fn tagged_root(ids: &[u128], tags: &[u8]) -> Root {
        Root {
            superfiles: ids.iter().map(|&id| Uuid::from_u128(id)).collect(),
            id_mins: ids.iter().map(|&id| id as i128).collect(),
            layers: tags.iter().map(|&tag| tagged_layer(tag)).collect(),
        }
    }

    /// The layers committed on top of `prior` are the ones `current` adds;
    /// none when nothing was committed.
    #[test]
    fn layers_since_are_the_deltas_committed_on_top_of_prior() {
        let prior = tagged_root(&[1, 2, 3, 4], &[0, 1]);
        // Two commits since: 5 at ordinal 4, 6 at ordinal 5.
        let current = tagged_root(&[1, 2, 3, 4, 5, 6], &[0, 1, 2, 3]);
        assert_eq!(
            layers_since(&prior, &current),
            Some(&[tagged_layer(2), tagged_layer(3)][..])
        );
        assert_eq!(layers_since(&prior, &prior), Some(&[][..]));
    }

    /// A root not built on `prior` has no deltas on top of it.
    #[test]
    fn layers_since_refuses_a_root_not_built_on_prior() {
        let prior = tagged_root(&[1, 2], &[0]);
        let rebuilt_elsewhere = tagged_root(&[1, 3], &[7, 8]);
        assert_eq!(layers_since(&prior, &rebuilt_elsewhere), None);

        let mut id_mins_differ = tagged_root(&[1, 2, 4], &[0, 1]);
        id_mins_differ.id_mins[0] += 1;
        assert_eq!(layers_since(&prior, &id_mins_differ), None);
    }

    /// The rebuild is padded to `prior`'s length and the deltas committed
    /// since go on top as they are, so their ordinals still name their
    /// superfiles.
    #[test]
    fn rebase_stacks_the_deltas_on_the_padded_rebuild() {
        let prior = tagged_root(&[1, 2, 3, 4], &[0, 1]);
        // The rebuild dropped 1 and 4, which were removed.
        let base = tagged_root(&[2, 3], &[9]);
        // Two commits since: 5 at ordinal 4, 6 at ordinal 5.
        let current = tagged_root(&[1, 2, 3, 4, 5, 6], &[0, 1, 2, 3]);
        assert_eq!(
            rebase(&base, &prior, &current),
            Some(tagged_root(&[2, 3, 0, 0, 5, 6], &[9, 2, 3]))
        );
    }

    /// With nothing committed on top of `prior`, the rebuild stands as it is.
    #[test]
    fn rebase_without_new_commits_is_the_rebuild() {
        let prior = tagged_root(&[1, 2, 3], &[0, 1]);
        let base = tagged_root(&[2, 3], &[9]);
        assert_eq!(rebase(&base, &prior, &prior), Some(base));
    }

    /// No rebase onto a root not built on `prior`, nor of a rebuild listing
    /// more superfiles than `prior`, whose ordinals would collide with the
    /// deltas'.
    #[test]
    fn rebase_refuses_what_it_cannot_stack() {
        let prior = tagged_root(&[1, 2], &[0]);
        let base = tagged_root(&[1, 2], &[9]);
        let rebuilt_elsewhere = tagged_root(&[1, 3], &[7, 8]);
        assert_eq!(rebase(&base, &prior, &rebuilt_elsewhere), None);

        let longer = tagged_root(&[1, 2, 7], &[9]);
        let current = tagged_root(&[1, 2, 5], &[0, 1]);
        assert_eq!(rebase(&longer, &prior, &current), None);
    }

    /// A commit that lands while optimize rebuilds the index is stacked on
    /// the rebuild as it was committed, not merged in: the retry writes no
    /// slice, the delta's ordinals still name its superfile, and the
    /// removed superfile leaves padding until the next rebuild drops it.
    #[test]
    fn a_commit_during_the_rebuild_is_rebased_onto() {
        let (dir, faults, storage, st) = fault_table();
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        // A removed superfile stays listed, so the rebuild is shorter than
        // the index it replaces.
        let reader = st.reader().expect("reader");
        let entries = reader.manifest().get_all_superfiles();
        commit_without_postings(&st, &storage, Vec::new(), &entries[..1]);
        st.block_on_query(st.refresh()).expect("refresh");
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (before, _) = live_and_covered(&st, &storage, &rt);

        // Another process commits just before the rebuild publishes. From
        // then on a slice write fails, so the retry must not write one.
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs"));
        let other = crate::supertable::Supertable::open(fresh_options(&local)).expect("open");
        let peer_ref = Arc::new(Mutex::new(None));
        let hook_peer_ref = Arc::clone(&peer_ref);
        let hook_faults = Arc::clone(&faults);
        faults.before(FaultOp::PutAtomic, "term-index/root-", move || {
            let reference = thread::spawn(move || {
                commit_layer(&other, LAYERS);
                other
                    .reader()
                    .expect("reader")
                    .manifest()
                    .term_index_ref()
                    .cloned()
            })
            .join()
            .expect("peer commit");
            *hook_peer_ref.lock().expect("peer ref") = reference;
            hook_faults.fail(FaultOp::PutAtomic, "term-index/slice-", usize::MAX);
        });
        maintenance_only_optimize(&st);
        faults.clear();
        assert_eq!(faults.fired(), 0, "the retry wrote no slice");

        let (live, root) = live_and_covered(&st, &storage, &rt);
        let listed: HashSet<Uuid> = root
            .superfiles
            .iter()
            .copied()
            .filter(|id| !id.is_nil())
            .collect();
        assert_eq!(listed, live, "every live superfile, and nothing removed");
        assert_eq!(
            root.superfiles.len(),
            live.len() + 1,
            "the removed superfile's ordinal is padded"
        );
        assert_eq!(root.layers.len(), 2, "the rebuild, then the commit's delta");
        let peer_ref = peer_ref
            .lock()
            .expect("peer ref")
            .clone()
            .expect("the peer published a root");
        let peer_root = rt
            .block_on(load_root(storage.as_ref(), &peer_ref))
            .expect("peer root");
        assert_eq!(
            root.layers.last(),
            peer_root.layers.last(),
            "the delta is stacked as the peer committed it"
        );
        assert!(
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_complete()
        );
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let found: HashSet<Uuid> = rt
            .block_on(index.postings(title_id(), "shared"))
            .expect("lookup")
            .iter()
            .map(|p| index.superfile_id(p.superfile).expect("ordinal resolves"))
            .collect();
        assert_eq!(found, live, "`shared` is found in every live superfile");
        // A term only the peer's superfile holds resolves to it alone, so
        // its delta's ordinal names the right superfile.
        let peer: HashSet<Uuid> = live.difference(&before).copied().collect();
        assert_eq!(peer.len(), 1, "the peer added one superfile");
        let peer_term = format!("s{LAYERS}d00");
        let found: HashSet<Uuid> = rt
            .block_on(index.postings(title_id(), &peer_term))
            .expect("lookup")
            .iter()
            .map(|p| index.superfile_id(p.superfile).expect("ordinal resolves"))
            .collect();
        assert_eq!(found, peer, "the peer's own term resolves to its superfile");

        maintenance_only_optimize(&st);
        let (live, root) = live_and_covered(&st, &storage, &rt);
        assert_eq!(root.layers.len(), 1, "the next rebuild is one layer");
        assert_eq!(root.superfiles.len(), live.len());
    }

    /// Arm `faults` so that a peer commits layer `next`, then `next + 1`,
    /// and so on, before each of the next `rounds` root writes: every publish
    /// in that window loses its race. A slice write after the first of them
    /// fails.
    fn commit_before_each_root_write(
        faults: &Arc<FaultStorage>,
        peer: crate::supertable::Supertable,
        next: usize,
        rounds: usize,
    ) {
        if rounds == 0 {
            return;
        }
        let hook_faults = Arc::clone(faults);
        faults.before(FaultOp::PutAtomic, "term-index/root-", move || {
            let committer = peer.clone();
            thread::spawn(move || commit_layer(&committer, next))
                .join()
                .expect("peer commit");
            hook_faults.fail(FaultOp::PutAtomic, "term-index/slice-", usize::MAX);
            commit_before_each_root_write(&hook_faults, peer, next + 1, rounds - 1);
        });
    }

    /// Commits landing before every publish do not starve the rebuild: each
    /// retry stacks what landed and writes only a root, and the index it
    /// finally publishes resolves every commit's terms.
    #[test]
    fn a_rebuild_publishes_while_commits_keep_landing() {
        const ROUNDS: usize = 4;
        let (dir, faults, storage, st) = fault_table();
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        // A removed superfile stays listed, so every rebase pads.
        let reader = st.reader().expect("reader");
        let entries = reader.manifest().get_all_superfiles();
        commit_without_postings(&st, &storage, Vec::new(), &entries[..1]);
        st.block_on_query(st.refresh()).expect("refresh");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs"));
        let other = crate::supertable::Supertable::open(fresh_options(&local)).expect("open");
        commit_before_each_root_write(&faults, other, LAYERS, ROUNDS);
        maintenance_only_optimize(&st);
        faults.clear();
        assert_eq!(faults.fired(), 0, "no retry wrote a slice");

        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (live, root) = live_and_covered(&st, &storage, &rt);
        assert_eq!(
            root.layers.len(),
            1 + ROUNDS,
            "the rebuild, then each commit's delta"
        );
        assert_eq!(
            root.superfiles.len(),
            live.len() + 1,
            "the removed superfile's ordinal is padded"
        );
        assert!(
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_complete()
        );
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let resolve = |term: &str| -> HashSet<Uuid> {
            rt.block_on(index.postings(title_id(), term))
                .expect("lookup")
                .iter()
                .map(|p| index.superfile_id(p.superfile).expect("ordinal resolves"))
                .collect()
        };
        assert_eq!(
            resolve("shared"),
            live,
            "`shared` is in every live superfile"
        );
        for layer in LAYERS..LAYERS + ROUNDS {
            let found = resolve(&format!("s{layer}d00"));
            assert_eq!(
                found.len(),
                1,
                "layer {layer}'s own term is in one superfile"
            );
            assert!(found.is_subset(&live), "and it is a live one");
        }
    }

    /// An incomplete index is rebuilt by walking only the superfiles it
    /// does not list, and a commit during the rebuild is folded in on top
    /// of it rather than everything being rebuilt again.
    #[test]
    fn a_commit_during_the_rebuild_of_an_incomplete_index_is_caught_up() {
        let (dir, faults, storage, st) = fault_table();
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        // With its root gone, the next commit restarts the index from its
        // own superfile alone.
        let reference = st
            .reader()
            .expect("reader")
            .manifest()
            .term_index_ref()
            .cloned()
            .expect("the commits publish a root");
        st.block_on_query(storage.delete(&reference.uri))
            .expect("delete root");
        commit_layer(&st, LAYERS);
        assert!(
            !st.reader()
                .expect("reader")
                .manifest()
                .term_index_complete()
        );

        // Another process commits just before the rebuild publishes.
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs"));
        let other = crate::supertable::Supertable::open(fresh_options(&local)).expect("open");
        faults.before(FaultOp::PutAtomic, "term-index/root-", move || {
            thread::spawn(move || commit_layer(&other, LAYERS + 1))
                .join()
                .expect("peer commit");
        });
        maintenance_only_optimize(&st);

        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (live, root) = live_and_covered(&st, &storage, &rt);
        assert_eq!(
            root.layers.len(),
            2,
            "the rebuild, then the caught-up commit"
        );
        assert_eq!(root.superfiles.len(), live.len(), "no padding");
        assert!(
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_complete()
        );
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let found: HashSet<Uuid> = rt
            .block_on(index.postings(title_id(), "shared"))
            .expect("lookup")
            .iter()
            .map(|p| index.superfile_id(p.superfile).expect("ordinal resolves"))
            .collect();
        assert_eq!(found, live, "`shared` is found in every live superfile");
    }

    /// Superfile objects end with this, whatever their stem.
    const SUPERFILE_SUFFIX: &str = ".sf.parquet";

    /// Optimize rebuilds the index from its own layers and reads no
    /// superfile: with every superfile read failing, it still publishes one
    /// layer over exactly the live superfiles.
    #[test]
    fn optimize_rebuilds_the_index_without_reading_superfiles() {
        let (_dir, faults, storage, st) = fault_table();
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        let reader = st.reader().expect("reader");
        let entries = reader.manifest().get_all_superfiles();
        commit_without_postings(&st, &storage, Vec::new(), &entries[..1]);
        drop(reader);
        drop(st);
        // A fresh handle holds no superfile in memory, so a walk would have
        // to read them from storage.
        let st = crate::supertable::Supertable::open(fresh_options(&storage)).expect("open");
        faults.fail(FaultOp::Get, SUPERFILE_SUFFIX, usize::MAX);
        faults.fail(FaultOp::GetRange, SUPERFILE_SUFFIX, usize::MAX);
        maintenance_only_optimize(&st);
        faults.clear();
        assert_eq!(faults.fired(), 0, "no superfile was read");

        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (live, root) = live_and_covered(&st, &storage, &rt);
        assert_eq!(root.layers.len(), 1);
        assert_eq!(root.superfiles.len(), live.len());
        assert!(
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_complete()
        );
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let found: HashSet<Uuid> = rt
            .block_on(index.postings(title_id(), "shared"))
            .expect("lookup")
            .iter()
            .map(|p| index.superfile_id(p.superfile).expect("ordinal resolves"))
            .collect();
        assert_eq!(found, live, "`shared` is found in every live superfile");
    }

    /// The merged index is byte for byte the one walking every superfile
    /// builds. The table has been compacted (so the merge copied the
    /// compaction's delta), appended to and had a superfile removed; the
    /// walk, forced by deleting the published root, publishes that same
    /// root again.
    #[test]
    fn the_merged_index_is_the_one_walking_every_superfile_builds() {
        let (dir, storage, st) = fresh_table();
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        let before = st.reader().expect("reader").n_superfiles();
        st.optimize(&OptimizeOptions::compact(CompactionSettings {
            min_fill_percent: 1,
            min_superfiles_for_merge: 2,
            ..CompactionSettings::default()
        }))
        .expect("compacting optimize");
        assert!(
            st.reader().expect("reader").n_superfiles() < before,
            "compaction must have merged"
        );
        commit_layer(&st, LAYERS);
        commit_layer(&st, LAYERS + 1);
        let reader = st.reader().expect("reader");
        let entries = reader.manifest().get_all_superfiles();
        commit_without_postings(&st, &storage, Vec::new(), &entries[entries.len() - 1..]);
        drop(reader);
        st.block_on_query(st.refresh()).expect("refresh");
        maintenance_only_optimize(&st);
        let merged = st
            .reader()
            .expect("reader")
            .manifest()
            .term_index_ref()
            .cloned()
            .expect("the merge publishes a root");
        drop(st);

        let root_path = dir.path().join(&merged.uri);
        fs::remove_file(&root_path).expect("remove root");
        let st = crate::supertable::Supertable::open(fresh_options(&storage)).expect("open");
        maintenance_only_optimize(&st);
        assert!(root_path.exists(), "the walk wrote its root");
        assert_eq!(
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_ref()
                .cloned(),
            Some(merged),
            "walking every superfile builds the same root"
        );
    }

    /// A slice the root names that is gone cannot be merged, so the rebuild
    /// walks every superfile instead and still publishes a whole index.
    #[test]
    fn a_missing_slice_falls_back_to_walking_every_superfile() {
        let (dir, storage, st) = fresh_table();
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let slice = &root.layers[0].slices[0];
        fs::remove_file(dir.path().join(slice_uri(&slice.content_hash))).expect("remove slice");

        maintenance_only_optimize(&st);
        let (live, root) = live_and_covered(&st, &storage, &rt);
        assert_eq!(root.layers.len(), 1);
        assert_eq!(root.superfiles.len(), live.len());
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let found: HashSet<Uuid> = rt
            .block_on(index.postings(title_id(), "shared"))
            .expect("lookup")
            .iter()
            .map(|p| index.superfile_id(p.superfile).expect("ordinal resolves"))
            .collect();
        assert_eq!(found, live, "`shared` is found in every live superfile");
    }

    /// When the retry cannot read the current root, it falls back to the
    /// last build. If that already lists every live superfile, it is
    /// published as it is, without an empty layer.
    #[test]
    fn a_retry_with_nothing_to_catch_up_adds_no_layer() {
        let (dir, faults, storage, st) = fault_table();
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }

        // Just before the rebuild publishes, another process removes a
        // superfile, and the root then fails to load twice: once for the
        // retry's rebuild check, once for the fold.
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs"));
        let other = crate::supertable::Supertable::open(fresh_options(&local)).expect("open");
        let hook_faults = Arc::clone(&faults);
        faults.before(FaultOp::PutAtomic, "term-index/root-", move || {
            thread::spawn(move || {
                let reader = other.reader().expect("reader");
                let entries = reader.manifest().get_all_superfiles();
                commit_without_postings(&other, &local, Vec::new(), &entries[..1]);
            })
            .join()
            .expect("peer commit");
            hook_faults.fail(FaultOp::Get, "term-index/root-", 2);
        });
        maintenance_only_optimize(&st);
        // The retry loads the root twice, for the rebuild check and the
        // fold, because a failure that may pass is not cached.
        assert_eq!(faults.fired(), 2, "both root loads failed");

        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (live, root) = live_and_covered(&st, &storage, &rt);
        assert_eq!(root.layers.len(), 1, "the build alone, no empty layer");
        let listed: HashSet<Uuid> = root.superfiles.iter().copied().collect();
        assert!(listed.is_superset(&live), "every live superfile is listed");
        assert!(
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_complete()
        );
    }

    /// A commit that only removes superfiles appends no layer and leaves
    /// the index complete, but the root still lists the removed superfile.
    /// Only the live-set check sees it.
    #[test]
    fn a_removal_only_commit_triggers_the_rebuild() {
        let (_dir, faults, storage, st) = optimized_fault_table();
        let reader = st.reader().expect("reader");
        let entries = reader.manifest().get_all_superfiles();
        commit_without_postings(&st, &storage, Vec::new(), &entries[..1]);
        // A direct commit does not advance the handle; adopt it.
        st.block_on_query(st.refresh()).expect("refresh");
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (live, root) = live_and_covered(&st, &storage, &rt);
        assert!(
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_complete()
        );
        assert_eq!(root.layers.len(), 1, "a removal appends no delta");
        assert_eq!(
            root.superfiles.len(),
            live.len() + 1,
            "the removed one is still listed"
        );

        assert!(optimize_rebuilds_term_index(&st, &faults));
        maintenance_only_optimize(&st);
        let (live, root) = live_and_covered(&st, &storage, &rt);
        let covered: HashSet<Uuid> = root.superfiles.iter().copied().collect();
        assert_eq!(covered, live, "the rebuild drops the removed superfile");
    }

    /// An index marked incomplete is rebuilt even when its root already has
    /// one layer over exactly the live superfiles.
    #[test]
    fn an_incomplete_index_triggers_the_rebuild() {
        let (_dir, faults, storage, st) = fault_table();
        commit_layer(&st, 0);
        // Adding then removing an unindexed superfile leaves the root as it
        // was but the index marked incomplete.
        let entry = unindexed_entry();
        commit_without_postings(&st, &storage, vec![Arc::clone(&entry)], &[]);
        commit_without_postings(&st, &storage, Vec::new(), &[entry]);
        st.block_on_query(st.refresh()).expect("refresh");
        assert!(
            !st.reader()
                .expect("reader")
                .manifest()
                .term_index_complete()
        );
        assert!(optimize_rebuilds_term_index(&st, &faults));
    }

    /// A root that is gone is rebuilt.
    #[test]
    fn an_unloadable_root_triggers_the_rebuild() {
        let (dir, faults, storage, st) = optimized_fault_table();
        let reference = st
            .reader()
            .expect("reader")
            .manifest()
            .term_index_ref()
            .cloned()
            .expect("reference");
        drop(st);
        fs::remove_file(dir.path().join(&reference.uri)).expect("remove root");
        // A fresh handle, so no loaded index is cached in memory.
        let st = crate::supertable::Supertable::open(fresh_options(&storage)).expect("open");
        assert!(optimize_rebuilds_term_index(&st, &faults));
    }

    /// Every ceiling the index computes is an upper bound on the score any
    /// document actually receives — for single terms, multi-term unions
    /// and phrases under table-wide statistics, where the stored bound is
    /// rescaled from the superfile's own idf to the query's.
    #[test]
    fn query_ceilings_bound_real_scores_for_terms_and_phrases() {
        use crate::Bm25SearchOptions;

        let (_dir, storage, st) = fresh_table();
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let reader = st.reader().expect("reader");
        let title = reader.manifest().field_id("title").expect("title id");
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
        // Table-wide idf per term, as the query scores with: N is the
        // scored-document total, df the sum
        // over every live superfile's posting.
        let scored_total: u64 = entries
            .iter()
            .map(|e| {
                e.fts_summary
                    .get(&title)
                    .and_then(|s| s.length_stats.as_ref().map(|l| l.n_scored_docs))
                    .unwrap_or(e.n_docs)
            })
            .sum();
        let global_idf: HashMap<&str, f32> = ["alpha", "beta", "shared", "s1d02"]
            .into_iter()
            .map(|term| {
                let df: u64 = rt
                    .block_on(index.postings(title_id(), term))
                    .expect("postings")
                    .iter()
                    .map(|p| p.df)
                    .sum();
                (term, bm25::idf(scored_total, df))
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
        for (query, terms, phrases) in &queries {
            let phrases: Vec<Vec<&str>> = phrases.iter().map(|p| p.to_vec()).collect();
            let idf_used = |term: &str| global_idf[term];
            let ceilings = rt
                .block_on(index.query_ceilings(title, terms, &phrases, &entries, &idf_used))
                .expect("ceilings");
            let batches = reader
                .bm25_search(
                    "title",
                    query,
                    DOCS_PER_LAYER * LAYERS,
                    Bm25SearchOptions::new(),
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
                    "{query} in {sf}: score {score} exceeds ceiling {ceiling}"
                );
                assert!(
                    ceiling.is_finite(),
                    "{query}: a real ceiling, not the placeholder"
                );
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
        let reference = rt.block_on(write_built(&storage, built)).expect("write");
        let index = rt
            .block_on(TermIndex::load(Arc::clone(&storage), None, &reference))
            .expect("load");
        let entry = Arc::new(SuperfileEntry {
            physical_schema: None,
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
        let local = |_: &str| 1.0f32;
        // The contribution keyed `title` by the test's stable id; the entry
        // has no summary, so that id reads the same as any other there.
        let title = fid("title");
        let ceilings = rt
            .block_on(index.query_ceilings(
                title,
                &[],
                &[vec!["alpha", "shared"]],
                &entries,
                &local,
            ))
            .expect("ceilings");
        assert_eq!(ceilings[&Uuid::from_u128(1)], f32::INFINITY);
        let ceilings = rt
            .block_on(index.query_ceilings(title, &["alpha"], &[], &entries, &local))
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

        let (_dir, _storage, st) = fresh_table_with_open_window_no_tiers(1);
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
                let opened = op_stats::current()
                    .expect("metered")
                    .score_pruning_survived();
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
            for layer in 0..LAYERS {
                commit_layer(&st, layer);
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
        use crate::superfile::{
            SuperfileReader,
            fts::reader::{ClauseLists, QueryIdf, ScoringView},
        };

        let (dir, storage, st) = fresh_table();
        commit_layer(&st, 0);
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
                .block_on(index.locations(title_id(), terms, &entries))
                .expect("locations");
            let pairs: Vec<(&str, u64, DictEntry)> = by_sf[&entries[0].superfile_id]
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
                        idf: QueryIdf::ThisFile,
                        prefetched: memo,
                        live_floor: None,
                        allow: None,
                        exact_count: false,
                    },
                    DOCS_PER_LAYER,
                    f32::NEG_INFINITY,
                    &ScoringView::default(),
                ))
                .expect("prepare");
            let mut hits = sf.run_prepared(prep, &ScoringView::default()).expect("run");
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
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
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
            .expect("the index loads")
            .expect("the table has an index");
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
        commit_layer(&st, 0);
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
            .block_on(warm.postings(title_id(), "shared"))
            .expect("fetch through storage");
        assert!(!first.is_empty());
        for slice in root.layers.iter().flat_map(|s| s.slices.iter()) {
            fs::remove_file(dir.path().join(slice_uri(&slice.content_hash))).expect("remove slice");
        }
        let cached = TermIndex::new(
            root.clone(),
            String::new(),
            Arc::clone(&storage),
            Some(cache),
        );
        let again = rt
            .block_on(cached.postings(title_id(), "shared"))
            .expect("served from the disk cache");
        assert_eq!(*again, *first);
        let uncached = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        assert!(
            matches!(
                rt.block_on(uncached.postings(title_id(), "shared")),
                Err(TermIndexError::Storage(_))
            ),
            "without the cache the missing object is a storage error"
        );
    }

    /// The tier objects under `dir`, by name.
    fn tier_objects(dir: &TempDir) -> HashSet<String> {
        fs::read_dir(dir.path().join(STORAGE_PREFIX))
            .map(|entries| {
                entries
                    .map(|e| {
                        e.expect("dir entry")
                            .file_name()
                            .to_string_lossy()
                            .into_owned()
                    })
                    .filter(|name| name.starts_with("tier-"))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// What the current root names: every slice, tier list and tier chunk.
    fn objects_named_by(root: &Root) -> HashSet<String> {
        let mut named: HashSet<String> = root
            .layers
            .iter()
            .flat_map(|s| s.slices.iter())
            .map(|s| {
                slice_uri(&s.content_hash)
                    .trim_start_matches(STORAGE_PREFIX)
                    .to_string()
            })
            .collect();
        named.extend(
            root.layers
                .iter()
                .flat_map(|s| s.slices.iter())
                .filter_map(|s| s.tier.as_ref())
                .map(|t| format!("tier-{}.bin", t.to_hex())),
        );
        named
    }

    /// Every commit writes its layer's tier chunks and names them from the
    /// root; the rebuild carries them into the base; a sweep past the gap
    /// keeps what the current root names and nothing a replaced root named.
    #[test]
    fn commits_write_tiers_the_rebuild_keeps_and_the_sweep_respects() {
        let (dir, storage, st) = fresh_table();
        commit_layer(&st, 0);
        commit_layer(&st, 1);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let current_ref = || {
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_ref()
                .cloned()
                .expect("a term-index reference")
        };
        let root = rt
            .block_on(load_root(storage.as_ref(), &current_ref()))
            .expect("root");
        assert_eq!(root.layers.len(), 2, "two deltas");
        assert!(
            root.layers
                .iter()
                .all(|s| s.slices.iter().any(|slice| slice.tier.is_some())),
            "each delta has a tier object"
        );
        let after_commits = tier_objects(&dir);
        let named = objects_named_by(&root);
        let with_tier = root
            .layers
            .iter()
            .flat_map(|s| s.slices.iter())
            .filter(|s| s.tier.is_some())
            .count();
        assert!(
            after_commits.iter().all(|o| named.contains(o)) && after_commits.len() == with_tier,
            "every tier object on disk is a slice's and named: {after_commits:?}"
        );
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let runs = rt
            .block_on(index.runs_many(title_id(), &["shared"]))
            .expect("runs")
            .remove(0);
        assert_eq!(runs.len(), 2, "one run per delta");
        let mut held = 0;
        for SliceRun { tier_object, run } in runs.iter() {
            let tier = run.tier.as_ref().expect("a term in every title has a tier");
            let object = tier_object.unwrap_or_else(|| ContentHash::of(b""));
            held += rt
                .block_on(index.tier_entries(&object, tier))
                .expect("tier entries")
                .len() as u64;
            assert!(
                run.postings.iter().all(|p| p.tail == 0.0),
                "complete heads leave no tail"
            );
        }
        assert_eq!(
            held,
            2 * DOCS_PER_LAYER as u64,
            "the deltas' tiers cover every document"
        );

        st.optimize(&maintenance_only_options()).expect("optimize");
        let rebuilt = rt
            .block_on(load_root(storage.as_ref(), &current_ref()))
            .expect("root");
        assert_eq!(rebuilt.layers.len(), 1, "the rebuild folds the deltas");
        assert!(
            rebuilt.layers[0]
                .slices
                .iter()
                .any(|slice| slice.tier.is_some())
        );
        let index = TermIndex::new(rebuilt.clone(), String::new(), Arc::clone(&storage), None);
        let runs = rt
            .block_on(index.runs_many(title_id(), &["shared"]))
            .expect("runs")
            .remove(0);
        assert_eq!(runs.len(), 1, "one run in the rebuilt base");
        let object = runs[0].tier_object.expect("the slice has a tier object");
        let tier = runs[0]
            .run
            .tier
            .as_ref()
            .expect("the merged run has a tier");
        let entries = rt
            .block_on(index.tier_entries(&object, tier))
            .expect("tier entries");
        assert_eq!(
            entries.len(),
            2 * DOCS_PER_LAYER,
            "the base tier covers every document"
        );
        assert!(
            entries
                .iter()
                .all(|e| index.superfile_id(e.superfile).is_some()),
            "every tier entry names a listed superfile"
        );

        st.gc(Duration::ZERO).expect("gc");
        let kept = tier_objects(&dir);
        let named: HashSet<String> = rt
            .block_on(async { objects_named_by(&rebuilt) })
            .into_iter()
            .filter(|o| o.starts_with("tier-"))
            .collect();
        assert_eq!(
            kept, named,
            "the sweep keeps exactly the tier objects the current root names"
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
        commit_layer(&st, 0);
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
        commit_layer(&st, 0);
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

        commit_layer(&st, 1);
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

        maintenance_only_optimize(&st);
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
        for slice in root.layers.iter().flat_map(|s| s.slices.iter()) {
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
        commit_layer(&st, 0);
        let first = st
            .reader()
            .expect("reader")
            .manifest()
            .term_index_ref()
            .cloned()
            .expect("first root");
        commit_layer(&st, 1);
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
        for slice in root.layers.iter().flat_map(|s| s.slices.iter()) {
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
        maintenance_only_optimize(&st);
        assert!(
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_ref()
                .is_none(),
            "nothing to index"
        );
        commit_layer(&st, 0);
        maintenance_only_optimize(&st);
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let reference = manifest.term_index_ref().cloned().expect("reference");
        let id = manifest.get_manifest_id();
        maintenance_only_optimize(&st);
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
        let mut bytes = bytes_of(&built.slices[0].1);
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
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let reader = st.reader().expect("reader");
        let entries = reader.manifest().get_all_superfiles().to_vec();
        let terms = ["alpha", "shared"];
        let by_sf = rt
            .block_on(index.locations(title_id(), &terms, &entries))
            .expect("locations");

        let mut planned_with_memo = 0u64;
        let mut docs_total = 0usize;
        for e in &entries {
            let bytes =
                std::fs::read(dir.path().join(e.uri.storage_path())).expect("superfile bytes");
            let sf = SuperfileReader::open(Bytes::from(bytes)).expect("open");
            let pairs: Vec<(&str, u64, DictEntry)> = by_sf[&e.superfile_id]
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

    /// On a table wider than sixty-four superfiles whose index one
    /// maintenance rebuild covers in a single layer, a term in every
    /// superfile still resolves from the index's locations: each
    /// superfile's memo covers every query term, so the ranked walk never
    /// reads a dictionary.
    #[test]
    fn a_wide_folded_index_resolves_every_superfile_from_locations() {
        use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};

        use crate::{
            superfile::SuperfileReader,
            supertable::query::fts::{index_locations_for, memo_from_locations},
        };

        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(8)
                .build()
                .expect("pool"),
        );
        let (dir, _storage, st) =
            table_with(|o| o.with_writer_pool(pool).with_superfile_buffer_split_mb(0));
        // Nine commits of eight superfiles each: 72, past the width at
        // which locations were once dropped.
        for c in 0..9 {
            let titles: Vec<String> = (0..400)
                .map(|i| format!("shared common{} c{c}d{i:03}", i % 3))
                .collect();
            let arr: ArrayRef = Arc::new(LargeStringArray::from(
                titles.iter().map(String::as_str).collect::<Vec<_>>(),
            ));
            let batch = RecordBatch::try_new(title_schema(), vec![arr]).expect("batch");
            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }
        st.refresh_term_index_sync().expect("maintenance rebuild");
        let rt = tokio::runtime::Runtime::new().expect("rt");
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let entries = manifest.get_all_superfiles().to_vec();
        assert!(entries.len() > 64, "wide table: {}", entries.len());
        let index = rt
            .block_on(manifest.term_index())
            .expect("index")
            .expect("indexed");
        assert_eq!(index.root().layers.len(), 1, "one folded layer");
        let run = rt
            .block_on(index.postings(title_id(), "shared"))
            .expect("postings");
        assert_eq!(run.len(), entries.len());
        assert!(
            run.iter().all(|p| p.location != Location::None),
            "a term in every superfile keeps every location"
        );
        let terms = ["shared", "common1", "absent"];
        let locations = rt
            .block_on(index_locations_for(manifest, "title", &terms, &entries))
            .expect("locations");
        for e in &entries {
            let bytes =
                std::fs::read(dir.path().join(e.uri.storage_path())).expect("superfile bytes");
            let sf = SuperfileReader::open(Bytes::from(bytes)).expect("open");
            let memo = rt
                .block_on(memo_from_locations(&sf, &locations, e.superfile_id))
                .expect("a memo for an indexed superfile");
            assert!(
                memo.covers(&terms),
                "every query term is resolved, present or absent, without the dictionary"
            );
        }
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
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
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
        let locations = rt
            .block_on(plan_locations_for(manifest, &plan, &entries))
            .expect("locations");
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

    /// A term in every one of many superfiles keeps a location in each:
    /// there is no width past which locations are dropped. Without them a
    /// query on such a term reads every opened superfile's whole dictionary
    /// — megabytes each — for postings that are kilobytes.
    #[test]
    fn locations_are_kept_however_many_superfiles_hold_the_term() {
        let dir = TempDir::new().expect("tempdir");
        let n = 130;
        let contributions: Vec<Contribution> = (1..=n)
            .map(|i| {
                let mut w =
                    ContributionWriter::create(dir.path(), Uuid::from_u128(i as u128), i as i128)
                        .expect("create");
                w.push(
                    &fid("body").term_key("common"),
                    7,
                    1.5,
                    Location::Pfor {
                        offset: 100 * i as u64,
                        len: 64,
                    },
                    0.0,
                    &[],
                )
                .expect("push");
                w.finish().expect("finish")
            })
            .collect();
        let built = build(&contributions, &BuildPolicy::default()).expect("build");
        let run: Vec<Posting> = built
            .slices
            .iter()
            .filter_map(|(_, built)| {
                Slice::open(&bytes_of(built))
                    .expect("open")
                    .postings(&fid("body").term_key("common"))
                    .expect("ok")
            })
            .flatten()
            .collect();
        assert_eq!(run.len(), n);
        for (i, p) in run.iter().enumerate() {
            assert_eq!(
                p.location,
                Location::Pfor {
                    offset: 100 * (i as u64 + 1),
                    len: 64
                },
                "superfile {} keeps its location",
                i + 1
            );
        }
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
                        w.push(&fid("body").term_key(&alphabet[*t]), 1 + *t as u64, f32::INFINITY, Location::None, 0.0, &[]).expect("push");
                    }
                    w.finish().expect("finish")
                })
                .collect();
            let policy = BuildPolicy {
            slice_target_bytes: 512 ,
            ..BuildPolicy::default()
        };
            let built = build(&contributions, &policy).expect("build");
            let store_dir = TempDir::new().expect("store dir");
            let storage: Arc<dyn StorageProvider> =
                Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local fs"));
            let rt = tokio::runtime::Runtime::new().expect("runtime");
            let reference = rt.block_on(write_built(&storage, built)).expect("write");
            let index = rt.block_on(TermIndex::load(Arc::clone(&storage), None, &reference)).expect("load");
            let holders = |t: usize| -> HashSet<Uuid> {
                corpus.iter().enumerate().filter(|(_, s)| s.contains(&t)).map(|(i, _)| Uuid::from_u128(i as u128 + 1)).collect()
            };
            for a in 0..12 {
                let single = rt.block_on(index.route(fid("body"), &[&alphabet[a]], BoolMode::Or)).expect("route");
                prop_assert_eq!(&single, &holders(a));
                for b in 0..12 {
                    let pair = [alphabet[a].as_str(), alphabet[b].as_str()];
                    let or = rt.block_on(index.route(fid("body"), &pair, BoolMode::Or)).expect("route");
                    let and = rt.block_on(index.route(fid("body"), &pair, BoolMode::And)).expect("route");
                    prop_assert_eq!(&or, &holders(a).union(&holders(b)).copied().collect::<HashSet<_>>());
                    prop_assert_eq!(&and, &holders(a).intersection(&holders(b)).copied().collect::<HashSet<_>>());
                }
            }
            let prefixed = rt.block_on(index.route_prefix(fid("body"), "t0")).expect("prefix");
            let expect: HashSet<Uuid> = (0..10).flat_map(holders).collect();
            prop_assert_eq!(prefixed, expect);
            prop_assert!(rt.block_on(index.route(fid("body"), &["nope"], BoolMode::Or)).expect("route").is_empty());
        });
    }

    /// Through the query path: superfile selection for a term is exactly
    /// the set of live superfiles whose dictionary holds it — no more (the
    /// bloom's false positives are gone) and no less.
    #[test]
    fn selection_routes_exactly_through_the_index() {
        use crate::supertable::query::prune::{PruneLeaf, select_superfiles};

        let (_dir, _storage, st) = fresh_table();
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        assert!(
            rt.block_on(manifest.term_index()).expect("loads").is_some(),
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
        // Every title holds `shared` and `alpha` is in every layer.
        assert_eq!(select(&["shared"], BoolMode::Or), live);
        assert_eq!(select(&["alpha", "shared"], BoolMode::And), live);
        // A term in no superfile: exact routing returns nothing, where a
        // summary could only say "maybe".
        assert!(select(&["absent"], BoolMode::Or).is_empty());
        assert!(select(&["absent", "shared"], BoolMode::And).is_empty());
        assert_eq!(select(&["absent", "shared"], BoolMode::Or), live);
        // A layer-specific token (`s1d00` is only in layer 1's titles).
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
    /// it. The oracle is each superfile searched on its own, so its scores
    /// are in the scale the bounds were baked in.
    #[test]
    fn bounds_are_upper_bounds_on_real_scores() {
        use crate::superfile::SuperfileReader;

        let (dir, storage, st) = fresh_table();
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let reader = st.reader().expect("reader");
        let superfiles: Vec<(Uuid, SuperfileReader)> = reader
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|e| {
                let bytes =
                    std::fs::read(dir.path().join(e.uri.storage_path())).expect("superfile bytes");
                let sf = SuperfileReader::open(Bytes::from(bytes)).expect("open");
                (e.superfile_id, sf)
            })
            .collect();
        let n_docs: HashMap<Uuid, u64> = reader
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|e| (e.superfile_id, e.n_docs))
            .collect();
        for term in ["shared", "alpha", "beta", "s1d00", "s2d04"] {
            let postings = rt
                .block_on(index.postings(title_id(), term))
                .expect("lookup");
            assert!(!postings.is_empty(), "{term} is indexed");
            // A posting's bound is a key; at the superfile's own idf it
            // bounds the superfile's own scores, an ulp of rounding aside.
            let bounds: HashMap<Uuid, f32> = postings
                .iter()
                .map(|p| {
                    let sf = index.superfile_id(p.superfile).expect("ordinal");
                    let ceiling = p.bound * bm25::idf(n_docs[&sf], p.df) * CEILING_SLACK;
                    (sf, ceiling)
                })
                .collect();
            for (sf, b) in &bounds {
                assert!(
                    b.is_finite(),
                    "{term} in {sf}: bound is a real ceiling, not the +inf placeholder"
                );
            }
            let mut observed_any = false;
            for (sf, superfile) in &superfiles {
                let hits = rt
                    .block_on(superfile.bm25_search_pretokenized(
                        "title",
                        &[term],
                        DOCS_PER_LAYER,
                        BoolMode::Or,
                    ))
                    .expect("search");
                let Some(observed) = hits.iter().map(|(_, s)| *s).reduce(f32::max) else {
                    continue;
                };
                observed_any = true;
                let bound = bounds
                    .get(sf)
                    .copied()
                    .unwrap_or_else(|| panic!("{term}: a superfile with hits has a posting"));
                assert!(
                    observed <= bound,
                    "{term} in {sf}: observed max {observed} exceeds bound {bound}"
                );
            }
            assert!(observed_any, "{term}: the fixture has hits");
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

    /// A table whose index keeps no tiers, so a ranked query takes the
    /// ceiling-ordered path and opens superfiles: what the open-count tests
    /// of that path exercise. With tiers the same queries are answered from
    /// the index without opening anything.
    fn fresh_table_with_open_window_no_tiers(
        window: usize,
    ) -> (
        TempDir,
        Arc<dyn StorageProvider>,
        crate::supertable::Supertable,
    ) {
        table_with(|o| {
            o.with_bound_ordered_open_window(window)
                .with_term_tier_entries_per_term(0)
        })
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
        };

        let (_dir, _storage, st) = fresh_table_with_open_window_no_tiers(1);
        // Layer 0 carries `alpha` three times per title — a clearly higher
        // ceiling than the single occurrence in layers 1 and 2.
        for layer in 0..3 {
            let word = if layer == 0 {
                "alpha alpha alpha"
            } else {
                "alpha"
            };
            let titles: Vec<String> = (0..DOCS_PER_LAYER)
                .map(|i| format!("{word} shared s{layer}d{i:02}"))
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
                        Bm25SearchOptions::new(),
                        Some(&["_id", "score"]),
                    )
                    .expect("search");
                let opened = op_stats::current()
                    .expect("metered")
                    .score_pruning_survived();
                (hits_of(&batches), opened)
            })
            .0
        };
        let (all, opened_all) = run(3 * DOCS_PER_LAYER);
        assert_eq!(
            opened_all, 3,
            "a k that needs every document opens every superfile"
        );
        assert_eq!(all.len(), 3 * DOCS_PER_LAYER);

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
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let reader = st.reader().expect("reader");
        let entries = reader.manifest().get_all_superfiles().to_vec();
        let terms = ["shared", "alpha", "beta", "s1d00", "absent"];
        let by_sf = rt
            .block_on(index.locations(title_id(), &terms, &entries))
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
            let pairs: Vec<(&str, u64, DictEntry)> = locs
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
                        DictEntry::Inline { doc_id: d, tf: t },
                    ) => {
                        assert_eq!((doc_id, tf), (d, t));
                    }
                    (
                        FetchedTermSlot::Pfor { bytes, short, .. },
                        DictEntry::Pfor {
                            postings_length,
                            short: s,
                            ..
                        },
                    ) => {
                        assert_eq!(short, s);
                        assert_eq!(
                            bytes.len() as u32,
                            postings_length,
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
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let entries = manifest.get_all_superfiles();
        let title = manifest.field_id("title").expect("title id");
        for e in entries {
            let summary = e.fts_summary.get(&title).expect("summary");
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
            if let Some(agg) = part.fts_summary_agg.get(&title) {
                assert!(agg.term_bloom.is_none(), "no per-part union bloom either");
            }
        }
        // The manifest-summary answer alone keeps everything (no information).
        let all_kept = fts_bloom_skip(entries, title, &["absent"], BoolMode::Or);
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
        commit_layer(&st, 0);
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
    /// parts, bloom unions in its list, no index.
    ///
    /// Four states of the same rows must answer every query identically:
    /// the fixture as written (blooms route); the fixture after the current
    /// writer appends a layer (index present but incomplete, so parts
    /// still route by summaries and only the new superfile is indexed);
    /// the fixture after a maintenance rebuild (index complete, blooms
    /// ignored); and a fresh table holding the same rows written entirely
    /// by the current writer.
    /// Recursive copy of a checked-in fixture into a scratch directory.
    /// Open a table written by the engine before the term index existed
    /// (the fixture's schema: one FTS column, no positions), with its options
    /// adjusted by `customize`.
    fn open_old_format(
        dir: &std::path::Path,
        customize: impl FnOnce(
            crate::supertable::SupertableOptions,
        ) -> crate::supertable::SupertableOptions,
    ) -> (Arc<dyn StorageProvider>, crate::supertable::Supertable) {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool"),
        );
        open_old_format_fts_fixture(dir, |o| customize(o.with_writer_pool(pool)))
    }

    /// An upgraded table whose manifest is loaded lazily: the parts that
    /// mix superfiles written before the index (bloom-bearing) and after it
    /// (bloom-less) must keep no part-level bloom, or the part prune would
    /// treat the old superfiles' bloom as authoritative and drop terms that
    /// live only in the new superfiles.
    /// A superfile written before field ids names its columns only by
    /// name. The index keys by id, so the build resolves each such column
    /// through the table's schema: the index built over the old fixture
    /// answers for `title`'s id.
    #[test]
    fn superfiles_without_field_ids_are_keyed_by_the_id_their_name_resolves_to() {
        let fixture = old_format_fts_fixture();
        let dir = TempDir::new().expect("tempdir");
        copy_dir_recursive(&fixture, dir.path());
        let (storage, st) = open_old_format(dir.path(), |o| o);
        maintenance_only_optimize(&st);

        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let title = manifest.field_id("title").expect("title is a table column");
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (live, root) = live_and_covered(&st, &storage, &rt);
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let shared = rt
            .block_on(index.postings(title, "shared"))
            .expect("lookup");
        assert_eq!(
            shared.len(),
            live.len(),
            "`shared` is in every fixture superfile, filed under `title`'s id"
        );
    }

    /// A table written before the list carried the schema leaves that
    /// state on its first commit under this engine: the list it writes is
    /// at the current format, carries the schema document (the fixture's
    /// one column, minted id 1), bears a commit token, and is stamped with
    /// the identity hash rather than the creation-record hash it opened
    /// under.
    #[test]
    fn an_old_table_s_first_commit_writes_the_schema_into_the_list() {
        let fixture = old_format_fts_fixture();
        let dir = TempDir::new().expect("tempdir");
        copy_dir_recursive(&fixture, dir.path());
        let (storage, st) = open_old_format(dir.path(), |o| o);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let options = Arc::clone(st.reader().expect("reader").options());
        let legacy = options.legacy_names();

        let before = current_list(&storage, &legacy, &rt);
        assert!(
            before.format_version.starts_with("1."),
            "the fixture's list predates the schema slot: {}",
            before.format_version
        );
        assert!(before.schema.is_none());
        assert!(before.commit_token.is_nil());

        commit_layer(&st, 3);

        let after = current_list(&storage, &legacy, &rt);
        assert_eq!(after.format_version, list::FORMAT_VERSION);
        let schema = after
            .schema
            .as_ref()
            .expect("the first commit fills the schema slot");
        assert_eq!(schema.id_of("title"), Some(FieldId(1)));
        assert_eq!(schema.schema_id(), 1);
        assert!(!after.commit_token.is_nil());
        assert_eq!(
            after.options_hash,
            compute_options_hash(&options, &after.partition_strategy)
        );
        assert_ne!(
            after.options_hash, before.options_hash,
            "the identity hash replaces the creation-record hash"
        );
    }

    #[test]
    fn upgraded_lazy_tables_find_terms_that_only_new_superfiles_hold() {
        use crate::Bm25SearchOptions;

        let fixture = old_format_fts_fixture();
        let dir = TempDir::new().expect("tempdir");
        copy_dir_recursive(&fixture, dir.path());
        let (_storage, st) = open_old_format(dir.path(), |o| o.with_eager_load_threshold(0));
        for layer in 3..6 {
            commit_layer(&st, layer);
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
        let (_dir, faults, storage, st) = fault_table();
        commit_layer(&st, 0);
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
        commit_layer(&st, 1);
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
        assert_eq!(root_after, root_before, "the accumulated layers are intact");
        let covered: std::collections::HashSet<Uuid> =
            root_after.superfiles.iter().copied().collect();
        assert_eq!(covered, live_before);

        maintenance_only_optimize(&st);
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

        use crate::{Bm25SearchOptions, supertable::Supertable};

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
                            DOCS_PER_LAYER * (LAYERS + 1),
                            Bm25SearchOptions::new().with_mode(*mode),
                            Some(&["title", "score"]),
                        )
                        .expect("search");
                    let mut hits = titled_hits(&batches);
                    hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    hits
                })
                .collect()
        };

        let fixture = old_format_fts_fixture();
        let work = TempDir::new().expect("tempdir");
        copy_dir_recursive(&fixture, work.path());

        // Cell 1: the old table as written. Blooms present, no index.
        let (_storage, st) = open(work.path());
        {
            let reader = st.reader().expect("reader");
            let manifest = reader.manifest();
            assert_eq!(manifest.get_all_superfiles().len(), LAYERS);
            assert!(
                manifest.term_index_ref().is_none(),
                "the fixture predates the index"
            );
            for e in manifest.get_all_superfiles() {
                assert!(
                    e.fts_summary[&manifest.field_id("title").expect("title id")]
                        .term_bloom
                        .is_some(),
                    "old entries carry blooms"
                );
            }
        }
        let old_answers = answers(&st);
        assert!(
            old_answers[0].len() == LAYERS * DOCS_PER_LAYER,
            "`shared` hits every row"
        );
        assert!(old_answers[5].is_empty());

        // Cell 2: the current writer appends a layer. An index appears,
        // covering only the new superfile; the list marks it incomplete;
        // the old superfiles keep routing by their blooms.
        commit_layer(&st, LAYERS);
        {
            let reader = st.reader().expect("reader");
            let manifest = reader.manifest();
            assert_eq!(manifest.get_all_superfiles().len(), LAYERS + 1);
            assert!(
                manifest.term_index_ref().is_some(),
                "the commit published an index"
            );
            assert!(
                !manifest.term_index_complete(),
                "an upgraded table's first index is incomplete"
            );
            let rt = tokio::runtime::Runtime::new().expect("runtime");
            let index = rt
                .block_on(manifest.term_index())
                .expect("index loads")
                .expect("indexed");
            let indexed = manifest
                .get_all_superfiles()
                .iter()
                .filter(|e| index.is_indexed(&e.superfile_id))
                .count();
            assert_eq!(indexed, 1, "only the new superfile is indexed");
        }
        let mixed_answers = answers(&st);
        // At a pruning k: an index that does not list every superfile
        // serves no tier, and the rebuilt one answers the same. (The old
        // superfiles record no baked average length, so the merge declines
        // them even then: a bound needs the average a head was keyed at.)
        let probe = |st: &Supertable| {
            crate::runtime_metrics::op_stats::with_op_stats(|| {
                st.reader()
                    .expect("reader")
                    .bm25_hits(
                        "title",
                        "alpha shared",
                        10,
                        Bm25SearchOptions::new().with_mode(BoolMode::And),
                    )
                    .expect("search")
                    .iter()
                    .map(|h| (h.superfile, h.local_doc_id, h.score))
                    .collect::<Vec<_>>()
            })
        };
        let (mixed_top, mixed_stats) = probe(&st);
        assert!(!mixed_top.is_empty());
        assert_eq!(
            mixed_stats.tier_entries, 0,
            "an index that does not cover the table serves no tier"
        );

        // Cell 3: a maintenance rebuild covers everything and flips the flag.
        maintenance_only_optimize(&st);
        {
            let manifest = st.reader().expect("reader").manifest().clone();
            assert!(
                manifest.term_index_complete(),
                "the rebuild lists every live superfile"
            );
        }
        let rebuilt_answers = answers(&st);
        let (rebuilt_top, _) = probe(&st);
        assert_eq!(rebuilt_top, mixed_top, "the same answer at a pruning k");

        // Cell 4: the same rows written entirely by the current writer.
        let (_d, _s, fresh) = fresh_table();
        for layer in 0..=LAYERS {
            commit_layer(&fresh, layer);
        }
        let fresh_answers = answers(&fresh);

        // The old three layers answer identically in every state; the
        // four-layer states answer identically to each other and to the
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
                tail: 1.0,
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
        commit_layer(&st, 0);
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let ask = |term: &str| {
            rt.block_on(index.runs_many(title_id(), &[term]))
                .expect("runs")
                .remove(0)
        };
        let (a, b) = (ask("shared"), ask("shared"));
        assert!(Arc::ptr_eq(&a, &b), "the second ask is the resident run");
        let absent = ask("absent");
        assert!(absent.is_empty());
        let again = ask("absent");
        assert!(
            Arc::ptr_eq(&absent, &again),
            "an absent term is remembered too"
        );
    }

    /// How long the slow store holds each read open, so reads issued
    /// together are seen overlapping.
    const SLOW_GET_DELAY: Duration = Duration::from_millis(20);
    /// Terms in the batched-lookup test, enough to span many small slices.
    const BATCH_TERMS: usize = 400;
    /// Slice target for the batched-lookup test: small, so the terms cut
    /// into many slices.
    const BATCH_SLICE_TARGET_BYTES: usize = 300;
    /// Random-key slice test: keys per build, and the slice target they cut
    /// against.
    const RANDOM_KEYS: usize = 20_000;
    const RANDOM_SLICE_TARGET_BYTES: usize = 64 * 1024;
    /// How far from the target a finished random-key slice may land: one
    /// term is under a hundred bytes, so this is generous, yet far below
    /// the 2x overshoot of an estimated cut.
    const RANDOM_SLICE_TOLERANCE_BYTES: usize = 4 * 1024;
    /// Hex digits in a random key — a 32-byte digest, the shape of a
    /// content-hash record key.
    const RANDOM_KEY_HEX_DIGITS: usize = 64;

    /// A batched lookup answers exactly what one-by-one lookups do — runs
    /// concatenated across layers in layer order, absent terms empty,
    /// a repeated term answered twice — while fetching each slice it needs
    /// once and several at a time. Looked up one by one, a query over many
    /// keys waited on one whole-slice fetch per key in series.
    #[tokio::test]
    async fn batched_postings_match_single_lookups_and_fetch_slices_together() {
        let dir = TempDir::new().expect("tempdir");
        let terms: Vec<String> = (0..BATCH_TERMS).map(|i| format!("k{i:04}")).collect();
        let base: Vec<(&str, &str, u64)> = terms.iter().map(|t| ("body", t.as_str(), 2)).collect();
        // The delta holds every third term again, so those runs span both
        // layers.
        let delta: Vec<(&str, &str, u64)> = terms
            .iter()
            .step_by(3)
            .map(|t| ("body", t.as_str(), 5))
            .collect();
        let policy = BuildPolicy {
            slice_target_bytes: BATCH_SLICE_TARGET_BYTES,
            ..BuildPolicy::default()
        };
        let store_dir = TempDir::new().expect("store dir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local fs"));
        let built = build(&[contribution(&dir, 1, &base)], &policy).expect("build");
        let prior = built.root.clone();
        write_built(&local, built).await.expect("write base");
        let reference = append_delta(
            &local,
            Some(prior),
            &[contribution(&dir, 2, &delta)],
            &policy,
            None,
        )
        .await
        .expect("append delta");
        let root = load_root(local.as_ref(), &reference).await.expect("load");
        assert_eq!(root.layers.len(), 2);

        // Interleave the terms so neighbours in the ask sit in different
        // slices, and add an absent term and a repeat.
        let mut asked: Vec<&str> = terms.iter().step_by(2).map(String::as_str).collect();
        asked.extend(terms.iter().skip(1).step_by(2).map(String::as_str));
        asked.push("absent");
        asked.push(terms[0].as_str());

        let slow = SlowStorage::wrap(Arc::clone(&local), SLOW_GET_DELAY);
        let batched_index = TermIndex::new(root.clone(), String::new(), slow.clone(), None);
        let batched = batched_index
            .runs_many(fid("body"), &asked)
            .await
            .expect("batched");
        let single_index = TermIndex::new(root.clone(), String::new(), Arc::clone(&local), None);
        for (term, runs) in asked.iter().zip(&batched) {
            let one = single_index
                .postings(fid("body"), term)
                .await
                .expect("single");
            assert_eq!(
                postings_of(runs).cloned().collect::<Vec<_>>(),
                one,
                "{term}"
            );
        }
        let spans_both = batched_index
            .postings(fid("body"), &terms[0])
            .await
            .expect("ok");
        assert_eq!(
            spans_both.iter().map(|p| p.df).collect::<Vec<_>>(),
            vec![2, 5],
            "base posting first, then the delta's"
        );
        assert!(batched[asked.len() - 2].is_empty(), "absent term");

        let needed: HashSet<ContentHash> = asked
            .iter()
            .flat_map(|t| {
                root.slices_for_key(&fid("body").term_key(t))
                    .map(|r| r.content_hash)
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(
            needed.len() > SLICE_FETCH_CONCURRENCY,
            "test spans many slices"
        );
        assert_eq!(slow.reads(), needed.len(), "each needed slice fetched once");
        assert!(
            slow.max_in_flight() > 1,
            "slices are fetched together, not one after another"
        );
        assert!(slow.max_in_flight() <= SLICE_FETCH_CONCURRENCY);
    }

    /// Random keys share almost no prefix, so front-coding keeps nearly all
    /// of each key and the dictionary is most of a slice. Slices are cut by
    /// their measured size, so they still land at the target rather than
    /// at the multiple of it a fixed per-key guess produced.
    #[test]
    fn random_keys_cut_slices_at_the_target_size() {
        let dir = TempDir::new().expect("tempdir");
        let keys: Vec<String> = (0..RANDOM_KEYS as u64)
            .map(|i| {
                let hex = ContentHash::of(&i.to_le_bytes()).to_hex();
                hex[..RANDOM_KEY_HEX_DIGITS].to_owned()
            })
            .collect();
        let terms: Vec<(&str, &str, u64)> =
            keys.iter().map(|k| ("record_key", k.as_str(), 1)).collect();
        let policy = BuildPolicy {
            slice_target_bytes: RANDOM_SLICE_TARGET_BYTES,
            ..BuildPolicy::default()
        };
        let built = build(&[contribution(&dir, 1, &terms)], &policy).expect("build");
        let slices = &built.root.layers[0].slices;
        assert!(slices.len() > 2, "the keys fill several slices");
        // Every slice but the last was cut when the next term would not
        // fit, so it is within one term of the target, above or below.
        let bytes_by_hash = |s: &SliceRef| {
            bytes_of(
                &built
                    .slices
                    .iter()
                    .find(|(h, _)| *h == s.content_hash)
                    .expect("slice bytes")
                    .1,
            )
            .len()
        };
        for s in &slices[..slices.len() - 1] {
            let len = bytes_by_hash(s);
            assert!(
                len <= RANDOM_SLICE_TARGET_BYTES + RANDOM_SLICE_TOLERANCE_BYTES,
                "slice of {len} bytes overshoots the {RANDOM_SLICE_TARGET_BYTES}-byte target"
            );
            assert!(
                len >= RANDOM_SLICE_TARGET_BYTES - RANDOM_SLICE_TOLERANCE_BYTES,
                "slice of {len} bytes was cut well short of the target"
            );
        }
    }

    /// The slice objects under `dir`, by name.
    fn slice_objects(dir: &TempDir) -> HashSet<String> {
        fs::read_dir(dir.path().join(STORAGE_PREFIX))
            .map(|entries| {
                entries
                    .map(|e| {
                        e.expect("dir entry")
                            .file_name()
                            .to_string_lossy()
                            .into_owned()
                    })
                    .filter(|name| name.starts_with("slice-"))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Superfile ids whose postings for `term` the root names, in order.
    async fn posted_superfiles(
        root: &Root,
        storage: &Arc<dyn StorageProvider>,
        term: &str,
    ) -> Vec<Uuid> {
        let index = TermIndex::new(root.clone(), String::new(), Arc::clone(storage), None);
        index
            .postings(fid("body"), term)
            .await
            .expect("lookup")
            .iter()
            .map(|p| index.superfile_id(p.superfile).expect("ordinal resolves"))
            .collect()
    }

    /// A local store with one base layer over superfile 1.
    async fn store_with_base(
        dir: &TempDir,
        policy: &BuildPolicy,
    ) -> (TempDir, Arc<dyn StorageProvider>, Root) {
        let store_dir = TempDir::new().expect("store dir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local fs"));
        let built =
            build(&[contribution(dir, 1, &[("body", "shared", 2)])], policy).expect("build");
        let root = built.root.clone();
        write_built(&local, built).await.expect("write base");
        (store_dir, local, root)
    }

    /// Terms a large commit holds beyond the shared one and its own: enough
    /// to make its layer tens of times a small commit's, about two
    /// levels up.
    const LARGE_COMMIT_TERMS: usize = 200;
    /// Commits in a stream of mixed sizes: past the fan-out squared, so the
    /// folds fold again.
    const MIXED_COMMITS: u128 = 80;
    /// A slice target below any slice's size, so the bound on what a commit
    /// folds refuses every run.
    const TINY_SLICE_TARGET_BYTES: usize = 64;
    /// Large commits that make a prior root's layers a level of their own.
    const LARGE_PRIOR_COMMITS: u128 = 3;
    /// Commit ids beyond any stream's, for a reserved layer's superfile.
    const RESERVED_ID: u128 = 99;

    /// Append superfile `id`'s delta to `root` through `storage` and read
    /// back the root it published: every commit holds `shared` and its own
    /// term, and a large one [`LARGE_COMMIT_TERMS`] more of its own.
    async fn commit_delta(
        dir: &TempDir,
        storage: &Arc<dyn StorageProvider>,
        root: Root,
        id: u128,
        large: bool,
        policy: &BuildPolicy,
    ) -> Root {
        let own = format!("own{id}");
        let extra: Vec<String> = match large {
            true => (0..LARGE_COMMIT_TERMS)
                .map(|t| format!("big{id}x{t}"))
                .collect(),
            false => Vec::new(),
        };
        let mut terms: Vec<(&str, &str, u64)> =
            vec![("body", "shared", 2), ("body", own.as_str(), 1)];
        terms.extend(extra.iter().map(|t| ("body", t.as_str(), 1)));
        let reference = append_delta(
            storage,
            Some(root),
            &[contribution(dir, id, &terms)],
            policy,
            None,
        )
        .await
        .expect("append");
        load_root(storage.as_ref(), &reference).await.expect("root")
    }

    /// The most layers a root folded to rest can hold, as the fold
    /// guarantees it: fewer than the fan-out in each level group beneath the
    /// newest layer, and each group's largest level more than
    /// [`FOLD_LEVEL_SPAN`] below the one before, so no more groups than the
    /// span fits into the levels the layers cover. One layer per commit,
    /// what appending alone leaves, is far past it.
    fn folded_count_bound(root: &Root) -> usize {
        let (lowest, highest) = root
            .layers
            .iter()
            .map(layer_level)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), level| {
                (lo.min(level), hi.max(level))
            });
        let groups = ((highest - lowest) / FOLD_LEVEL_SPAN).floor() as usize + 1;
        (FOLD_FAN_OUT - 1) * groups + 1
    }

    /// `root` routes every term as appending alone would have: `shared` to
    /// superfiles `1..=last` in commit order, each commit's own term to that
    /// commit alone.
    async fn assert_routes_every_commit(
        root: &Root,
        storage: &Arc<dyn StorageProvider>,
        last: u128,
    ) {
        assert_eq!(
            root.superfiles.len(),
            last as usize,
            "ordinals are untouched"
        );
        let expected: Vec<Uuid> = (1..=last).map(Uuid::from_u128).collect();
        assert_eq!(
            posted_superfiles(root, storage, "shared").await,
            expected,
            "a term in every commit routes to every superfile in commit order"
        );
        for id in 2..=last {
            assert_eq!(
                posted_superfiles(root, storage, &format!("own{id}")).await,
                [Uuid::from_u128(id)],
                "a term of one commit routes to that superfile alone"
            );
        }
    }

    /// A stream of like-sized delta commits folds its layers as they pile
    /// up, so the root holds a logarithmic number of them rather than one
    /// per commit, and every term still routes to exactly the superfiles
    /// that hold it, in commit order. Folds happen at the fan-out, so the
    /// count is checked just before and just after one.
    #[tokio::test]
    async fn delta_commits_fold_their_layers_and_route_the_same() {
        /// Enough commits for a fold of folds: past the fan-out squared.
        const COMMITS: u128 = (FOLD_FAN_OUT * FOLD_FAN_OUT + 3) as u128;
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy::default();
        let (_store_dir, local, base) = store_with_base(&dir, &policy).await;
        let mut root = base;
        let mut layers_before_first_fold = 0;
        for id in 2..=COMMITS {
            root = commit_delta(&dir, &local, root, id, false, &policy).await;
            let appended = id as usize; // layers so far: the base plus the deltas
            if appended == FOLD_FAN_OUT {
                layers_before_first_fold = root.layers.len();
            }
            if appended == FOLD_FAN_OUT + 1 {
                assert_eq!(
                    layers_before_first_fold, FOLD_FAN_OUT,
                    "nothing folds before a run of the fan-out lies beneath the newest layer"
                );
                assert_eq!(
                    root.layers.len(),
                    2,
                    "the first run folds into one layer beneath the newest"
                );
            }
        }
        assert!(
            root.layers.len() <= folded_count_bound(&root),
            "{COMMITS} commits left {} layers",
            root.layers.len()
        );
        assert_routes_every_commit(&root, &local, COMMITS).await;
    }

    /// A stream of commits of mixed sizes folds as a uniform one does:
    /// small and large commits alternating, and one large commit in four,
    /// each keep the root to a few layers per level, where folding only
    /// runs of like-sized neighbours left one layer per commit.
    #[tokio::test]
    async fn mixed_commit_sizes_fold_as_uniform_ones_do() {
        /// One large commit in this many, for each stream.
        const LARGE_EVERY: [u128; 2] = [2, 4];
        for every in LARGE_EVERY {
            let dir = TempDir::new().expect("tempdir");
            let policy = BuildPolicy::default();
            let (_store_dir, local, base) = store_with_base(&dir, &policy).await;
            let mut root = base;
            for id in 2..=MIXED_COMMITS {
                let large = id % every == 0;
                root = commit_delta(&dir, &local, root, id, large, &policy).await;
            }
            assert!(
                root.layers.len() <= folded_count_bound(&root),
                "one large commit in {every}: {MIXED_COMMITS} commits left {} layers, past {}",
                root.layers.len(),
                folded_count_bound(&root)
            );
            assert_routes_every_commit(&root, &local, MIXED_COMMITS).await;
        }
    }

    /// A run whose bytes pass one slice of the build's target size is not
    /// folded at commit: the commit appends its layer as it did before
    /// folding, and the run is left for optimize's rebuild.
    #[tokio::test]
    async fn a_run_past_one_slice_is_left_for_optimize() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy {
            slice_target_bytes: TINY_SLICE_TARGET_BYTES,
            ..BuildPolicy::default()
        };
        let (_store_dir, local, base) = store_with_base(&dir, &policy).await;
        let mut root = base;
        let commits = (2 * FOLD_FAN_OUT) as u128;
        for id in 2..=commits {
            root = commit_delta(&dir, &local, root, id, false, &policy).await;
        }
        assert_eq!(
            root.layers.len(),
            commits as usize,
            "no run fits one slice, so every commit keeps its layer"
        );
        assert_routes_every_commit(&root, &local, commits).await;
    }

    /// A fold that cannot read the slices it would merge does not fail the
    /// commit: the root is published unfolded, every layer in it, and
    /// routes as before.
    #[tokio::test]
    async fn a_fold_that_fails_publishes_the_root_unfolded() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy::default();
        let (_store_dir, local, base) = store_with_base(&dir, &policy).await;
        let mut root = base;
        let unfolded = FOLD_FAN_OUT as u128;
        for id in 2..=unfolded {
            root = commit_delta(&dir, &local, root, id, false, &policy).await;
        }
        assert_eq!(root.layers.len(), FOLD_FAN_OUT, "nothing folded yet");
        // The next commit makes a run, and every slice read it would merge
        // fails.
        let faults = FaultStorage::wrap(Arc::clone(&local));
        faults.fail(FaultOp::Get, "slice-", usize::MAX);
        let faulty = Arc::clone(&faults) as Arc<dyn StorageProvider>;
        let root = commit_delta(&dir, &faulty, root, unfolded + 1, false, &policy).await;
        assert!(faults.fired() > 0, "the fold tried to read its run");
        assert_eq!(
            root.layers.len(),
            FOLD_FAN_OUT + 1,
            "the root is published with every layer unfolded"
        );
        assert_routes_every_commit(&root, &local, unfolded + 1).await;
    }

    /// A retry that appends its reserved layer onto a winner's root folds
    /// the run beneath it like any delta: the reserved layer stays as
    /// built and newest, the padding up to its ordinal is kept, and every
    /// term routes to its superfile.
    #[tokio::test]
    async fn a_fold_on_the_retry_path_keeps_the_reserved_layer_and_padding() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy::default();
        let (_store_dir, local, base) = store_with_base(&dir, &policy).await;
        // Two short of a run beneath the newest, so neither the first
        // attempt nor the winner folds, and the retry does.
        let mut prior = base;
        let last_prior = (FOLD_FAN_OUT - 1) as u128;
        for id in 2..=last_prior {
            prior = commit_delta(&dir, &local, prior, id, false, &policy).await;
        }
        let reserved_own = format!("own{RESERVED_ID}");
        let retried = [contribution(
            &dir,
            RESERVED_ID,
            &[("body", "shared", 3), ("body", reserved_own.as_str(), 1)],
        )];
        let mut reserved = None;
        append_reserved(
            &local,
            Some(prior.clone()),
            &retried,
            &policy,
            &mut reserved,
            None,
        )
        .await
        .expect("first attempt");
        let winner_id = last_prior + 1;
        let winner = commit_delta(&dir, &local, prior, winner_id, false, &policy).await;
        assert_eq!(winner.layers.len(), FOLD_FAN_OUT, "the winner did not fold");

        let reference = append_reserved(
            &local,
            Some(winner.clone()),
            &retried,
            &policy,
            &mut reserved,
            None,
        )
        .await
        .expect("retry");
        let root = load_root(local.as_ref(), &reference).await.expect("root");
        let layer = &reserved.as_ref().expect("reserved").layer;
        assert_eq!(root.layers.len(), 2, "the run beneath the retry folded");
        assert_eq!(
            root.layers.last(),
            Some(layer),
            "the reserved layer is appended as it was built"
        );
        let padding = root.superfiles.iter().filter(|id| id.is_nil()).count();
        let ordinal = reserved.as_ref().expect("reserved").base as usize;
        assert_eq!(padding, ordinal - winner.superfiles.len());
        let mut expected: Vec<Uuid> = (1..=winner_id).map(Uuid::from_u128).collect();
        expected.push(Uuid::from_u128(RESERVED_ID));
        assert_eq!(posted_superfiles(&root, &local, "shared").await, expected);
        assert_eq!(
            posted_superfiles(&root, &local, &reserved_own).await,
            [Uuid::from_u128(RESERVED_ID)]
        );
    }

    /// A rebuild made from a prior root stacks the layers committed since
    /// it when a fold stayed wholly above it: a prior of large commits is a
    /// level of its own, the small commits after it fold among themselves,
    /// and the prior is still a prefix of the folded root.
    #[tokio::test]
    async fn a_rebuild_stacks_a_fold_wholly_above_its_prior() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy::default();
        let (_store_dir, local, base) = store_with_base(&dir, &policy).await;
        let mut prior = base;
        let last_prior = 1 + LARGE_PRIOR_COMMITS;
        for id in 2..=last_prior {
            prior = commit_delta(&dir, &local, prior, id, true, &policy).await;
        }
        let mut current = prior.clone();
        let last = last_prior + FOLD_FAN_OUT as u128 + 1;
        for id in last_prior + 1..=last {
            current = commit_delta(&dir, &local, current, id, false, &policy).await;
        }
        assert_eq!(
            current.layers.len(),
            prior.layers.len() + 2,
            "the small commits folded into one layer beneath the newest"
        );
        assert!(layers_since(&prior, &current).is_some());
        let stacked = rebase(&prior, &prior, &current).expect("stacked onto the rebuild");
        assert_routes_every_commit(&stacked, &local, last).await;
    }

    /// A fold that reached into a rebuild's prior root leaves no prefix to
    /// stack on: the rebuild is told so, and catches up instead.
    #[tokio::test]
    async fn a_rebuild_catches_up_when_a_fold_reached_into_its_prior() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy::default();
        let (_store_dir, local, base) = store_with_base(&dir, &policy).await;
        let mut prior = base;
        let last_prior = 3;
        for id in 2..=last_prior {
            prior = commit_delta(&dir, &local, prior, id, false, &policy).await;
        }
        let mut current = prior.clone();
        let last = FOLD_FAN_OUT as u128 + 1;
        for id in last_prior + 1..=last {
            current = commit_delta(&dir, &local, current, id, false, &policy).await;
        }
        assert_eq!(current.layers.len(), 2, "the fold took the prior's layers");
        assert_eq!(layers_since(&prior, &current), None);
        assert_eq!(rebase(&prior, &prior, &current), None);
        assert_routes_every_commit(&current, &local, last).await;
    }

    /// A retry appends the layer it already built onto the winner's root:
    /// no slice is written again, the root is padded up to the reserved
    /// ordinal, and every posting names the right superfile.
    #[tokio::test]
    async fn append_reserved_reuses_its_layer_on_a_longer_root() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy::default();
        let (store_dir, local, prior) = store_with_base(&dir, &policy).await;
        let merged = [contribution(
            &dir,
            9,
            &[("body", "shared", 3), ("body", "merged", 1)],
        )];
        let mut reserved = None;
        append_reserved(
            &local,
            Some(prior.clone()),
            &merged,
            &policy,
            &mut reserved,
            None,
        )
        .await
        .expect("first attempt");

        // Another commit publishes a delta first.
        let winner = append_delta(
            &local,
            Some(prior.clone()),
            &[contribution(&dir, 2, &[("body", "shared", 4)])],
            &policy,
            None,
        )
        .await
        .expect("winner's delta");
        let winner = load_root(local.as_ref(), &winner)
            .await
            .expect("winner root");
        let slices_before_retry = slice_objects(&store_dir);

        let reference = append_reserved(
            &local,
            Some(winner.clone()),
            &merged,
            &policy,
            &mut reserved,
            None,
        )
        .await
        .expect("retry");

        assert_eq!(
            slice_objects(&store_dir),
            slices_before_retry,
            "the retry writes no slice"
        );
        let root = load_root(local.as_ref(), &reference).await.expect("root");
        let base = prior.superfiles.len() + RESERVED_ORDINAL_SLACK as usize;
        assert_eq!(root.superfiles.len(), base + 1);
        assert_eq!(root.superfiles[base], Uuid::from_u128(9));
        let padding = root.superfiles.iter().filter(|id| id.is_nil()).count();
        assert_eq!(padding, base - winner.superfiles.len());
        assert_eq!(
            posted_superfiles(&root, &local, "shared").await,
            [1, 2, 9].map(Uuid::from_u128),
            "base, winner, then the reserved layer"
        );
        assert_eq!(
            posted_superfiles(&root, &local, "merged").await,
            [Uuid::from_u128(9)]
        );
    }

    /// A root grown past the reserved ordinal gets the layer moved, other
    /// contributions get one built again; either way its postings name the
    /// right superfile.
    #[tokio::test]
    async fn append_reserved_moves_or_rebuilds_when_it_cannot_reuse() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy::default();
        let (store_dir, local, prior) = store_with_base(&dir, &policy).await;
        let merged = [contribution(&dir, 9, &[("body", "merged", 1)])];
        let mut reserved = None;
        append_reserved(
            &local,
            Some(prior.clone()),
            &merged,
            &policy,
            &mut reserved,
            None,
        )
        .await
        .expect("first attempt");

        // More commits than the reservation leaves room for.
        let mut outgrown = prior.clone();
        let extra = RESERVED_ORDINAL_SLACK as usize + 1;
        outgrown.superfiles.extend(repeat_n(Uuid::nil(), extra));
        outgrown.id_mins.extend(repeat_n(0, extra));
        let slices_before = slice_objects(&store_dir);
        let reference = append_reserved(
            &local,
            Some(outgrown.clone()),
            &merged,
            &policy,
            &mut reserved,
            None,
        )
        .await
        .expect("outgrown");
        assert_ne!(
            slice_objects(&store_dir),
            slices_before,
            "moved slices written"
        );
        let root = load_root(local.as_ref(), &reference).await.expect("root");
        assert_eq!(
            root.superfiles.len(),
            outgrown.superfiles.len() + RESERVED_ORDINAL_SLACK as usize + 1
        );
        assert_eq!(
            posted_superfiles(&root, &local, "merged").await,
            [Uuid::from_u128(9)]
        );

        // A batch that lost a job hands over other contributions.
        let other = [contribution(&dir, 10, &[("body", "other", 1)])];
        let reference = append_reserved(&local, Some(prior), &other, &policy, &mut reserved, None)
            .await
            .expect("other contributions");
        let root = load_root(local.as_ref(), &reference).await.expect("root");
        assert_eq!(
            posted_superfiles(&root, &local, "other").await,
            [Uuid::from_u128(10)]
        );
        assert!(posted_superfiles(&root, &local, "merged").await.is_empty());
    }

    /// A slice target small enough that a few terms make several slices.
    const SMALL_SLICE_TARGET_BYTES: usize = 64;
    /// Documents a superfile holds of the term every tier shape shares: past
    /// one block, and past the head the fixtures keep.
    const COMMON_DOCS: u32 = 130;
    /// Documents a fixture head keeps of that term.
    const COMMON_HEAD: usize = 100;

    /// Three superfiles whose terms take every shape a run has: a tier in
    /// chunks of several blocks with tails, an inline tier, no tier, and
    /// every location kind.
    fn mixed_contributions(dir: &TempDir) -> Vec<Contribution> {
        let common = |first: u32| -> Vec<(u32, f32)> {
            (0..COMMON_DOCS)
                .map(|i| (first + i, 1.0 + i as f32 / 10.0))
                .collect()
        };
        vec![
            contribution_with_heads(
                dir,
                1,
                &[
                    ("body", "common", common(0)),
                    ("body", "rare", vec![(5, 2.0)]),
                ],
                COMMON_HEAD,
            ),
            contribution_with_heads(
                dir,
                2,
                &[
                    ("body", "common", common(1_000)),
                    ("body", "rare", vec![(7, 3.0)]),
                ],
                COMMON_HEAD,
            ),
            contribution(
                dir,
                3,
                &[
                    ("body", "common", 400),
                    ("body", "plain", 1),
                    ("body", "short", 50),
                ],
            ),
        ]
    }

    /// `objects` in content-hash order, for comparing sets of objects.
    fn by_hash(mut objects: Vec<(ContentHash, BuiltBytes)>) -> Vec<(ContentHash, BuiltBytes)> {
        objects.sort_by_key(|(hash, _)| hash.0);
        objects
    }

    /// A layer moved to other ordinals is, byte for byte, what a build at
    /// those ordinals writes, up and down, resident and spilled: every
    /// posting, inline tier entry and chunk directory renumbered, nothing
    /// else changed.
    #[test]
    fn a_moved_layer_equals_a_build_at_its_new_ordinals() {
        let dir = TempDir::new().expect("tempdir");
        let contributions = mixed_contributions(&dir);
        let n = contributions.len() as u32;
        let resident = BuildPolicy {
            slice_target_bytes: SMALL_SLICE_TARGET_BYTES,
            ..BuildPolicy::default()
        };
        let spilling = BuildPolicy {
            spill_to_disk: true,
            ..resident.clone()
        };
        for policy in [&resident, &spilling] {
            for (from, to) in [(300, 5_000), (5_000, 300), (70_000, 900_000)] {
                let at_from = build_layer(&contributions, policy, from).expect("build at from");
                assert!(at_from.layer.slices.len() > 1, "several slices");
                assert!(!at_from.tiers.is_empty(), "a chunked tier");
                let shift = OrdinalShift::new(from, to, n).expect("same widths");
                let moved = rebase_layer(&at_from.layer, &at_from.slices, &at_from.tiers, &shift)
                    .expect("move");
                let at_to = build_layer(&contributions, policy, to).expect("build at to");
                assert_eq!(moved.layer, at_to.layer, "{from} → {to}: slice list");
                assert_eq!(moved.slices, at_to.slices, "{from} → {to}: slices");
                assert_eq!(
                    by_hash(moved.tiers),
                    by_hash(at_to.tiers),
                    "{from} → {to}: tier objects"
                );
            }
        }
    }

    /// A move refuses a layer that names an ordinal it does not move, and
    /// leaves the kept bytes as they were.
    #[test]
    fn a_move_refuses_an_ordinal_outside_the_layer() {
        let dir = TempDir::new().expect("tempdir");
        let contributions = mixed_contributions(&dir);
        let policy = BuildPolicy {
            slice_target_bytes: SMALL_SLICE_TARGET_BYTES,
            ..BuildPolicy::default()
        };
        let built = build_layer(&contributions, &policy, 300).expect("build");
        let before = built.slices.clone();
        // One ordinal short: the layer's last superfile is not moved.
        let shift = OrdinalShift::new(300, 5_000, contributions.len() as u32 - 1).expect("shift");
        assert!(rebase_layer(&built.layer, &built.slices, &built.tiers, &shift).is_err());
        assert_eq!(built.slices, before, "kept bytes untouched");
    }

    /// A retry whose root outgrew the reservation moves the layer it built
    /// rather than building it again: it succeeds with the contributions'
    /// spill gone, and publishes the layer a build at the new ordinals
    /// would.
    #[tokio::test]
    async fn append_reserved_moves_its_layer_without_its_contributions() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy::default();
        let (_store_dir, local, prior) = store_with_base(&dir, &policy).await;
        let merged = mixed_contributions(&dir);
        let mut reserved = None;
        append_reserved(
            &local,
            Some(prior.clone()),
            &merged,
            &policy,
            &mut reserved,
            None,
        )
        .await
        .expect("first attempt");

        let mut outgrown = prior;
        let extra = RESERVED_ORDINAL_SLACK as usize + 1;
        outgrown.superfiles.extend(repeat_n(Uuid::nil(), extra));
        outgrown.id_mins.extend(repeat_n(0, extra));
        let base = outgrown.superfiles.len() as u32 + RESERVED_ORDINAL_SLACK;
        let expected = build_layer(&merged, &policy, base).expect("build at the new base");
        for c in &merged {
            fs::remove_file(&c.path).expect("drop the contribution spill");
        }

        let reference = append_reserved(
            &local,
            Some(outgrown),
            &merged,
            &policy,
            &mut reserved,
            None,
        )
        .await
        .expect("moved without the contributions");
        let root = load_root(local.as_ref(), &reference).await.expect("root");
        assert_eq!(root.layers.last(), Some(&expected.layer));
        assert_eq!(
            posted_superfiles(&root, &local, "rare").await,
            [1, 2].map(Uuid::from_u128)
        );
    }

    /// A move that would change how many bytes an ordinal takes is not made
    /// in place: the layer is built again from its contributions.
    #[tokio::test]
    async fn append_reserved_builds_again_across_a_varint_width() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy::default();
        let (_store_dir, local, prior) = store_with_base(&dir, &policy).await;
        let merged = mixed_contributions(&dir);
        let mut reserved = None;
        append_reserved(
            &local,
            Some(prior.clone()),
            &merged,
            &policy,
            &mut reserved,
            None,
        )
        .await
        .expect("first attempt");

        // Grown far enough that the new base's ordinals take a byte more.
        let mut outgrown = prior;
        let extra = 8_200;
        outgrown.superfiles.extend(repeat_n(Uuid::nil(), extra));
        outgrown.id_mins.extend(repeat_n(0, extra));
        let reference = append_reserved(
            &local,
            Some(outgrown.clone()),
            &merged,
            &policy,
            &mut reserved,
            None,
        )
        .await
        .expect("built again");
        let root = load_root(local.as_ref(), &reference).await.expect("root");
        let base = outgrown.superfiles.len() as u32 + RESERVED_ORDINAL_SLACK;
        let expected = build_layer(&merged, &policy, base).expect("build at the new base");
        assert_eq!(root.layers.last(), Some(&expected.layer));

        // The same crossing without the contributions: the build, and so
        // the retry, fails.
        let mut fresh = None;
        append_reserved(
            &local,
            Some(prior_of(&outgrown, extra)),
            &merged,
            &policy,
            &mut fresh,
            None,
        )
        .await
        .expect("first attempt");
        for c in &merged {
            fs::remove_file(&c.path).expect("drop the contribution spill");
        }
        assert!(
            append_reserved(&local, Some(outgrown), &merged, &policy, &mut fresh, None)
                .await
                .is_err(),
            "no move across the width, so a build was tried"
        );
    }

    /// `root` without its last `n` superfiles: the root a commit started
    /// from before `n` more landed.
    fn prior_of(root: &Root, n: usize) -> Root {
        let mut prior = root.clone();
        prior.superfiles.truncate(root.superfiles.len() - n);
        prior.id_mins.truncate(root.id_mins.len() - n);
        prior
    }

    /// A compaction that loses its commit to an ingest commit appends the
    /// term-index layer it already built: every slice on storage is one
    /// the root names, the index stays complete, and the merged superfile's
    /// terms route to it. Optimize's rebuild then drops the padding.
    #[test]
    fn a_compaction_retry_reuses_its_term_index_layer() {
        let (dir, faults, storage, st) = fault_table();
        for layer in 0..LAYERS {
            commit_layer(&st, layer);
        }
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (before, _) = live_and_covered(&st, &storage, &rt);
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs"));
        let peer = crate::supertable::Supertable::open(fresh_options(&local)).expect("open");
        faults.before(FaultOp::PutIfMatch, POINTER_PATH, move || {
            thread::spawn(move || commit_layer(&peer, LAYERS))
                .join()
                .expect("peer commit");
        });

        st.compact(&CompactionSettings {
            min_fill_percent: 1,
            min_superfiles_for_merge: 2,
            ..CompactionSettings::default()
        })
        .expect("compact");

        let (live, root) = live_and_covered(&st, &storage, &rt);
        // The peer published first, so the compaction's first pointer swap
        // lost and its commit landed on a retry.
        assert_eq!(live.len(), 2, "the merged superfile and the peer's");
        assert!(live.is_disjoint(&before), "every input was merged");
        let named: HashSet<String> = root
            .layers
            .iter()
            .flat_map(|s| &s.slices)
            .map(|s| {
                slice_uri(&s.content_hash)
                    .trim_start_matches(STORAGE_PREFIX)
                    .to_string()
            })
            .collect();
        assert_eq!(
            slice_objects(&dir),
            named,
            "the retry wrote no slice of its own"
        );
        assert!(
            root.superfiles.iter().any(Uuid::is_nil),
            "the root is padded up to the reserved ordinal"
        );
        assert!(
            st.reader()
                .expect("reader")
                .manifest()
                .term_index_complete()
        );
        let index = TermIndex::new(root, String::new(), Arc::clone(&storage), None);
        let found: HashSet<Uuid> = rt
            .block_on(index.postings(title_id(), "shared"))
            .expect("lookup")
            .iter()
            .map(|p| index.superfile_id(p.superfile).expect("ordinal resolves"))
            .filter(|id| live.contains(id))
            .collect();
        assert_eq!(found, live, "`shared` routes to every live superfile");

        maintenance_only_optimize(&st);
        let (live, root) = live_and_covered(&st, &storage, &rt);
        assert_eq!(root.layers.len(), 1);
        assert!(
            root.superfiles.iter().all(|id| !id.is_nil()),
            "the rebuild drops the padding"
        );
        assert_eq!(root.superfiles.len(), live.len());
    }

    /// A root layout version older than the one this engine reads.
    const OLD_ROOT_VERSION: u32 = 2;
    /// Root GETs a test counts before it stops counting.
    const ROOT_GET_PROBES: usize = 4;
    /// The URI fragment every root object carries.
    const ROOT_URI_FRAGMENT: &str = "term-index/root-";

    /// Point the table at a copy of its root stamped with layout
    /// `version`, as an engine that wrote that layout would have left it.
    fn plant_root_at_version(st: &Supertable, storage: &Arc<dyn StorageProvider>, version: u32) {
        let reference = st
            .reader()
            .expect("reader")
            .manifest()
            .term_index_ref()
            .cloned()
            .expect("reference");
        let (bytes, _) = st
            .block_on_query(storage.get(&reference.uri))
            .expect("root bytes");
        let mut bytes = bytes.to_vec();
        let at = ROOT_MAGIC.len();
        bytes[at..at + size_of::<u32>()].copy_from_slice(&version.to_le_bytes());
        st.block_on_query(stamp_with_retries(
            st.inner(),
            storage,
            "old-layout root",
            |old| {
                let bytes = bytes.clone();
                async move {
                    let reference = put_content_addressed(storage.as_ref(), "root", bytes).await?;
                    Ok(Some(old.with_term_index(reference)))
                }
            },
        ))
        .expect("publish the old root");
    }

    /// A term leaf on `title`, as SQL pruning plans one.
    fn title_leaf(term: &str) -> PruneLeaf {
        PruneLeaf::TermPresence {
            column: "title".to_owned(),
            terms: vec![term.to_owned()],
            mode: BoolMode::Or,
        }
    }

    /// A root in a layout this engine no longer reads fails every read that
    /// routes through it, naming the repair. Optimize rebuilds it and the
    /// reads work again. Both handle kinds are covered: an eager one routes
    /// per superfile, a lazy one also chooses manifest parts through the
    /// index.
    #[test]
    fn a_root_in_an_old_layout_fails_reads_until_optimize_rebuilds_it() {
        for eager_load_threshold in [u32::MAX, 0] {
            let (_dir, storage, st, _) = optimized_fragmented_table();
            plant_root_at_version(&st, &storage, OLD_ROOT_VERSION);
            drop(st);
            let options =
                || fresh_options(&storage).with_eager_load_threshold(eager_load_threshold);
            let st = Supertable::open(options()).expect("open");
            let reader = st.reader().expect("reader");
            let failures = [
                reader
                    .bm25_search("title", "alpha", 10, Bm25SearchOptions::new(), None)
                    .err(),
                reader.token_match("title", "alpha", BoolMode::Or).err(),
                reader.count("title", "alpha", BoolMode::Or).err(),
                reader.exact_match("title", "alpha").err(),
                st.block_on_query(select_superfiles(reader.manifest(), &[title_leaf("alpha")]))
                    .err(),
            ];
            for failure in failures {
                let e = InfinoError::from(failure.expect("the read fails"));
                assert!(matches!(e, InfinoError::Unsupported(_)), "{e}");
                assert!(e.to_string().contains("optimize()"), "{e}");
            }
            assert!(
                st.gc(Duration::ZERO).is_err(),
                "gc cannot derive the live set either"
            );

            maintenance_only_optimize(&st);
            let reader = st.reader().expect("reader");
            assert!(reader.count("title", "alpha", BoolMode::Or).expect("count") > 0);
            assert!(
                st.block_on_query(reader.manifest().term_index())
                    .expect("the rebuilt index loads")
                    .is_some()
            );
        }
    }

    /// A root GET that fails for now fails the read as retryable I/O and is
    /// not remembered: the next read loads the root.
    #[test]
    fn a_transient_root_failure_fails_the_read_and_the_next_one_loads_it() {
        let (_dir, faults, storage, st) = optimized_fault_table();
        drop(st);
        let st = Supertable::open(fresh_options(&storage)).expect("open");
        let reader = st.reader().expect("reader");
        faults.fail(FaultOp::Get, ROOT_URI_FRAGMENT, 1);
        let e = InfinoError::from(
            reader
                .count("title", "alpha", BoolMode::Or)
                .expect_err("the read fails"),
        );
        assert!(matches!(e, InfinoError::Io(_)), "{e}");
        assert_eq!(faults.fired(), 1, "the failure is the injected one");
        assert!(reader.count("title", "alpha", BoolMode::Or).expect("count") > 0);
    }

    /// A root that can never load is fetched once per handle, not once per
    /// query.
    #[test]
    fn a_root_that_never_loads_is_fetched_once() {
        let (_dir, faults, storage, st) = optimized_fault_table();
        plant_root_at_version(&st, &storage, OLD_ROOT_VERSION);
        drop(st);
        let st = Supertable::open(fresh_options(&storage)).expect("open");
        let gets = Arc::new(AtomicUsize::new(0));
        for _ in 0..ROOT_GET_PROBES {
            let gets = Arc::clone(&gets);
            faults.before(FaultOp::Get, ROOT_URI_FRAGMENT, move || {
                gets.fetch_add(1, Ordering::SeqCst);
            });
        }
        let reader = st.reader().expect("reader");
        for _ in 0..ROOT_GET_PROBES {
            assert!(reader.count("title", "alpha", BoolMode::Or).is_err());
        }
        assert_eq!(gets.load(Ordering::SeqCst), 1, "one GET for every query");
    }

    /// A root that fails to load for now fails the optimize and writes no
    /// index object; the next optimize, with the root readable, folds the
    /// index as usual.
    #[test]
    fn a_transient_root_failure_fails_optimize_without_rebuilding() {
        let (_dir, faults, storage, st) = optimized_fault_table();
        // An appended delta, so the next optimize has to load the root to
        // decide whether to rebuild.
        commit_layer(&st, LAYERS);
        drop(st);
        let st = Supertable::open(fresh_options(&storage)).expect("open");
        faults.fail(FaultOp::Get, ROOT_URI_FRAGMENT, 1);
        faults.fail(FaultOp::PutAtomic, STORAGE_PREFIX, 1);
        let fired_before = faults.fired();
        assert!(st.optimize(&maintenance_only_options()).is_err());
        faults.clear();
        assert_eq!(
            faults.fired(),
            fired_before + 1,
            "the root GET failed and no index object was written"
        );
        assert!(optimize_rebuilds_term_index(&st, &faults));
    }

    /// A slice the root names that is gone fails the read.
    #[test]
    fn a_missing_slice_fails_the_read() {
        let (dir, storage, st, _) = optimized_fragmented_table();
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let (_, root) = live_and_covered(&st, &storage, &rt);
        drop(st);
        for slice in root.layers.iter().flat_map(|s| s.slices.iter()) {
            fs::remove_file(dir.path().join(slice_uri(&slice.content_hash))).expect("remove slice");
        }
        let st = Supertable::open(fresh_options(&storage)).expect("open");
        let reader = st.reader().expect("reader");
        let e = InfinoError::from(
            reader
                .token_match("title", "alpha", BoolMode::Or)
                .expect_err("the read fails"),
        );
        assert!(matches!(e, InfinoError::Backend(_)), "{e}");
    }

    /// A commit on a table whose root is in an old layout succeeds: the
    /// index restarts from the commit's own superfiles, marked incomplete,
    /// and reads work again before any optimize.
    #[test]
    fn a_commit_over_an_old_layout_root_restarts_the_index() {
        let (_dir, storage, st, _) = optimized_fragmented_table();
        plant_root_at_version(&st, &storage, OLD_ROOT_VERSION);
        drop(st);
        let st = Supertable::open(fresh_options(&storage)).expect("open");
        commit_layer(&st, LAYERS);
        let reader = st.reader().expect("reader");
        assert!(
            !reader.manifest().term_index_complete(),
            "the restarted index lists only the new superfiles"
        );
        assert!(reader.count("title", "alpha", BoolMode::Or).expect("count") > 0);
    }

    /// Heads for `n` documents at descending keys from `base`.
    fn head_docs(base: f32, n: u32) -> Vec<(u32, f32)> {
        (0..n).map(|i| (i * 3, base + i as f32 * 0.01)).collect()
    }

    /// One of the three superfiles of [`tier_shapes`].
    fn tier_shape(dir: &TempDir, id: u128, alpha: u32, beta: u32, delta: u32) -> Contribution {
        contribution_with_heads(
            dir,
            id,
            &[
                ("body", "alpha", head_docs(id as f32, alpha)),
                ("body", "beta", head_docs(0.3, beta)),
                ("body", "gamma", head_docs(0.5, 2)),
                ("body", "delta", head_docs(0.7, delta)),
            ],
            1000,
        )
    }

    /// Three superfiles whose terms reach every shape a merge's tier can
    /// take: chunked groups of one and two ranks (`alpha`), exact multiples
    /// of a block (`delta`), one superfile inline beside one chunked
    /// (`beta`), and a term small enough to stay inline (`gamma`). Returns
    /// the merge sources — the base build over superfiles 1 and 2 and the
    /// delta over 3 — spilled under `dir`.
    fn tier_shapes(dir: &TempDir, policy: &BuildPolicy) -> Vec<Vec<SliceFile>> {
        let base = build(
            &[tier_shape(dir, 1, 60, 5, 64), tier_shape(dir, 2, 60, 2, 2)],
            policy,
        )
        .expect("base");
        let delta = build_onto(
            base.root.clone(),
            &[tier_shape(dir, 3, 90, 30, 128)],
            policy,
        )
        .expect("delta");
        spill_tiers(dir.path(), &base.root.layers[0].slices, base.tiers).expect("spill base tiers");
        spill_tiers(dir.path(), &delta.root.layers[1].slices, delta.tiers)
            .expect("spill delta tiers");
        vec![
            spill_slices(dir.path(), &base.root.layers[0].slices, base.slices).expect("spill base"),
            spill_slices(dir.path(), &delta.root.layers[1].slices, delta.slices)
                .expect("spill delta"),
        ]
    }

    /// The merge of `sources` under `policy`, with and without splicing,
    /// must agree byte for byte: the splice is a shortcut for the
    /// decoding path, never a different result.
    fn assert_splice_matches_decoding(
        sources: &[Vec<SliceFile>],
        ordinals: &[Option<u32>],
        policy: &BuildPolicy,
    ) -> build::MergedLayer {
        let spliced = merge_layers(sources, ordinals, policy).expect("spliced merge");
        let decoding = merge_layers(
            sources,
            ordinals,
            &BuildPolicy {
                splice_tiers: false,
                ..policy.clone()
            },
        )
        .expect("decoding merge");
        assert_eq!(spliced.slices, decoding.slices, "slices agree");
        assert_eq!(spliced.tiers, decoding.tiers, "tier objects agree");
        assert_eq!(spliced.layer, decoding.layer, "layers agree");
        spliced
    }

    /// The entries of `term` in a merged output.
    fn entries_of(merged: &build::MergedLayer, term: &str) -> usize {
        merged_terms(merged)
            .iter()
            .find(|(key, _, _)| key == &fid("body").term_key(term))
            .map(|(_, _, entries)| entries.len())
            .expect("the term is in the merge")
    }

    /// Over every tier shape, a merge that splices writes what the
    /// decoding merge writes, which is what a fresh build over the
    /// surviving superfiles writes when no source was truncated.
    #[test]
    fn a_spliced_merge_writes_what_the_decoding_merge_writes() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy {
            slice_target_bytes: 4096,
            tier_entries_per_term: 1_000,
            ..BuildPolicy::default()
        };
        let sources = tier_shapes(&dir, &policy);
        let merged = assert_splice_matches_decoding(&sources, &[Some(0), None, Some(1)], &policy);
        let fresh = build(
            &[
                tier_shape(&dir, 1, 60, 5, 64),
                tier_shape(&dir, 3, 90, 30, 128),
            ],
            &policy,
        )
        .expect("fresh");
        assert_eq!(merged.slices, fresh.slices, "a fresh build's slices");
        assert_eq!(merged.tiers, fresh.tiers, "a fresh build's tier objects");
        assert_eq!(
            entries_of(&merged, "beta"),
            35,
            "the inline head of one superfile sits beside the chunked head of the other"
        );
    }

    /// A budget below what the groups hold, or a depth below a group's
    /// size, sends the term down the decoding path; the two paths still
    /// agree, and the merge cuts the tier to the policy.
    #[test]
    fn a_splice_declines_a_cut_and_the_paths_still_agree() {
        let wide = BuildPolicy {
            slice_target_bytes: 4096,
            tier_entries_per_term: 1_000,
            ..BuildPolicy::default()
        };
        for (policy, cap) in [
            (
                BuildPolicy {
                    tier_entries_per_term: 100,
                    ..wide.clone()
                },
                100,
            ),
            (
                BuildPolicy {
                    head_depth_per_superfile: 64,
                    ..wide.clone()
                },
                3 * 64,
            ),
        ] {
            let dir = TempDir::new().expect("tempdir");
            let sources = tier_shapes(&dir, &wide);
            let merged =
                assert_splice_matches_decoding(&sources, &[Some(0), Some(1), Some(2)], &policy);
            let held = entries_of(&merged, "alpha");
            assert!(
                held <= cap && held < 60 + 60 + 90,
                "the merge cut alpha's tier to the policy: {held} entries, cap {cap}"
            );
        }
    }

    /// A term chunked in its sources can hold eight entries or fewer once
    /// the merge drops superfiles; the decoding path then keeps the tier
    /// inline in the run, and the splice must step aside rather than write
    /// chunks for it.
    #[test]
    fn a_splice_declines_a_tier_that_shrinks_to_inline() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy {
            slice_target_bytes: 4096,
            tier_entries_per_term: 1_000,
            ..BuildPolicy::default()
        };
        let sources = tier_shapes(&dir, &policy);
        // Only superfile 2 survives: `delta` falls from 64 + 2 entries,
        // chunked in the base layer, to its 2.
        let merged = assert_splice_matches_decoding(&sources, &[None, Some(0), None], &policy);
        assert_eq!(entries_of(&merged, "delta"), 2);
        let bytes = bytes_of(&merged.slices[0].1);
        let run = Slice::open(&bytes)
            .expect("open")
            .run(&fid("body").term_key("delta"))
            .expect("run")
            .expect("delta is in the merge");
        assert!(
            run.tier
                .as_ref()
                .is_some_and(|t| t.chunks.is_empty() && t.inline.len() == 2),
            "the merged tier sits inline in the run: {:?}",
            run.tier
        );
    }

    /// A tier read in windows smaller than a chunk crosses a window on
    /// nearly every term and writes what the default window writes.
    #[test]
    fn tier_windows_of_any_size_read_the_same_bytes() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy {
            slice_target_bytes: 4096,
            tier_entries_per_term: 1_000,
            ..BuildPolicy::default()
        };
        let sources = tier_shapes(&dir, &policy);
        let ordinals = [Some(0), None, Some(1)];
        let wide = merge_layers(&sources, &ordinals, &policy).expect("merge");
        for window in [1u64, 64, 4096] {
            let narrow = merge_layers(
                &sources,
                &ordinals,
                &BuildPolicy {
                    tier_read_window_bytes: window,
                    ..policy.clone()
                },
            )
            .expect("merge");
            assert_eq!(narrow.slices, wide.slices, "window {window}: slices");
            assert_eq!(narrow.tiers, wide.tiers, "window {window}: tiers");
        }
    }

    /// A slice's block marks name every dictionary block in key order,
    /// with run and chunk offsets that rise with the keys.
    #[test]
    fn block_marks_cover_every_block_in_order() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy {
            tier_entries_per_term: 100_000,
            ..BuildPolicy::default()
        };
        let built = build(&[range_contribution(&dir, 1)], &policy).expect("build");
        assert_eq!(built.slices.len(), 1, "one slice at the default target");
        let bytes = bytes_of(&built.slices[0].1);
        let slice = Slice::open(&bytes).expect("open");
        let terms = slice.terms_after(None, usize::MAX).expect("terms");
        let marks = slice.block_marks().expect("marks");
        assert_eq!(marks.len(), terms.len().div_ceil(TERM_BLOCK_SIZE));
        for (i, mark) in marks.iter().enumerate() {
            assert_eq!(
                mark.key,
                terms[i * TERM_BLOCK_SIZE].0,
                "mark {i} names its block's first key"
            );
        }
        assert!(
            marks
                .windows(2)
                .all(|w| w[0].postings_at < w[1].postings_at),
            "runs sit in key order"
        );
        let chunked: Vec<u64> = marks.iter().filter_map(|m| m.tier_at).collect();
        assert!(!chunked.is_empty(), "some blocks start with a chunked tier");
        assert!(
            chunked.windows(2).all(|w| w[0] < w[1]),
            "chunks sit in key order"
        );
        assert!(
            marks
                .last()
                .is_some_and(|m| m.postings_at < slice.postings_len())
        );
    }

    /// Ranges are cut by the bytes they would merge, so a hot key range —
    /// terms carrying most of the tier bytes — is spread over ranges that
    /// hold fewer terms each, cut inside a slice, and the merge in key ranges
    /// still writes the terms the one-writer merge writes.
    #[test]
    fn range_cuts_fall_by_weight_not_by_slice() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy {
            slice_target_bytes: 1 << 20,
            tier_entries_per_term: 100_000,
            merge_ranges: Some(4),
            ..BuildPolicy::default()
        };
        let names: Vec<String> = (0..600).map(|i| format!("t{i:04}")).collect();
        let hot = |id: u128| {
            let entries: Vec<(&str, &str, Vec<(u32, f32)>)> = names
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let n = if (200..400).contains(&i) { 60 } else { 20 };
                    (
                        "body",
                        t.as_str(),
                        head_docs(id as f32 + i as f32 * 0.001, n),
                    )
                })
                .collect();
            contribution_with_heads(&dir, id, &entries, 1000)
        };
        let built = build(&[hot(1), hot(2), hot(3)], &policy).expect("build");
        spill_tiers(dir.path(), &built.root.layers[0].slices, built.tiers).expect("spill tiers");
        let sources = vec![
            spill_slices(dir.path(), &built.root.layers[0].slices, built.slices)
                .expect("spill slices"),
        ];
        let ranges = key_ranges(&sources, &policy).expect("ranges");
        assert_eq!(ranges.len(), 4, "four ranges: {ranges:?}");
        let slice_starts: Vec<&[u8]> = sources[0].iter().map(|f| f.first_key.as_slice()).collect();
        let cuts: Vec<&[u8]> = ranges.iter().filter_map(|(s, _)| s.as_deref()).collect();
        assert!(
            cuts.iter().any(|c| !slice_starts.contains(c)),
            "a cut falls inside a slice: {cuts:?}"
        );
        let in_range = |t: &str, (s, e): &(Option<Vec<u8>>, Option<Vec<u8>>)| {
            let k = fid("body").term_key(t);
            s.as_ref().is_none_or(|s| *s <= k) && e.as_ref().is_none_or(|e| k < *e)
        };
        let counts: Vec<usize> = ranges
            .iter()
            .map(|r| names.iter().filter(|t| in_range(t, r)).count())
            .collect();
        let hot_range = ranges
            .iter()
            .position(|r| in_range("t0300", r))
            .expect("the hot range is in a range");
        let most = counts.iter().copied().max().unwrap_or(0);
        assert!(
            counts[hot_range] < most && counts[hot_range] * 2 < names.len(),
            "a hot range holds fewer terms than a cold one: {counts:?}"
        );
        let ordinals = [Some(0), Some(1), Some(2)];
        let many = merge_layers(&sources, &ordinals, &policy).expect("merge");
        let one = merge_layers(
            &sources,
            &ordinals,
            &BuildPolicy {
                merge_ranges: Some(1),
                ..policy.clone()
            },
        )
        .expect("merge");
        assert_eq!(merged_terms(&many), merged_terms(&one));
    }

    /// One failed object fails the whole transfer, download and upload
    /// alike, rather than a partial result.
    #[test]
    fn one_failed_object_fails_the_transfer() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy {
            slice_target_bytes: 2048,
            tier_entries_per_term: 100_000,
            ..BuildPolicy::default()
        };
        let built = build(&[range_contribution(&dir, 1)], &policy).expect("build");
        assert!(built.slices.len() >= 4, "{} slices", built.slices.len());
        let store_dir = TempDir::new().expect("store");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));
        let faults = FaultStorage::wrap(Arc::clone(&local));
        let faulty = Arc::clone(&faults) as Arc<dyn StorageProvider>;
        let rt = tokio::runtime::Runtime::new().expect("runtime");

        faults.fail(FaultOp::PutAtomic, "term-index/slice-", 1);
        let put = rt.block_on(write_slices(&faulty, built.slices.clone()));
        assert!(put.is_err(), "a failed put fails the upload");
        assert_eq!(faults.fired(), 1);

        let reference = rt.block_on(write_built(&local, built)).expect("write");
        let root = rt
            .block_on(load_root(local.as_ref(), &reference))
            .expect("root");
        let scratch = TempDir::new().expect("scratch");
        faults.fail(FaultOp::Get, "term-index/slice-", 1);
        let got = rt.block_on(download_layers(&faulty, &root.layers, scratch.path(), None));
        assert!(got.is_err(), "a failed fetch fails the download");
        assert_eq!(faults.fired(), 2);
        let good = rt.block_on(download_layers(&local, &root.layers, scratch.path(), None));
        assert!(good.is_ok(), "the same download succeeds without the fault");
    }

    /// Every file `download_layers` placed under `dir` for `layers`: each
    /// slice and its tier object.
    fn downloaded_bytes(dir: &Path, layers: &[Layer]) -> Vec<Vec<u8>> {
        layers
            .iter()
            .flat_map(|layer| &layer.slices)
            .flat_map(|slice| {
                let tier = slice.tier.map(|_| tier_path(dir, &slice.content_hash));
                [Some(slice_file_path(dir, &slice.content_hash)), tier]
            })
            .flatten()
            .map(|path| fs::read(path).expect("downloaded file"))
            .collect()
    }

    /// A store that kept a download serves the next one with no fetch, the
    /// same bytes, and a prune to an empty root clears it.
    #[test]
    fn a_kept_download_serves_the_next_one_without_a_fetch() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy {
            slice_target_bytes: 2048,
            tier_entries_per_term: 100_000,
            ..BuildPolicy::default()
        };
        let built = build(&[range_contribution(&dir, 1)], &policy).expect("build");
        let objects = TempDir::new().expect("objects");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(objects.path()).expect("local"));
        let faults = FaultStorage::wrap(Arc::clone(&local));
        let faulty = Arc::clone(&faults) as Arc<dyn StorageProvider>;
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let reference = rt.block_on(write_built(&local, built)).expect("write");
        let root = rt
            .block_on(load_root(local.as_ref(), &reference))
            .expect("root");
        assert!(root.layers[0].slices.iter().any(|s| s.tier.is_some()));
        let store = TempDir::new().expect("store");

        let first = TempDir::new().expect("first");
        rt.block_on(download_layers(
            &local,
            &root.layers,
            first.path(),
            Some(store.path()),
        ))
        .expect("download");
        faults.fail(FaultOp::Get, STORAGE_PREFIX, usize::MAX);
        let second = TempDir::new().expect("second");
        rt.block_on(download_layers(
            &faulty,
            &root.layers,
            second.path(),
            Some(store.path()),
        ))
        .expect("served from the store");
        assert_eq!(faults.fired(), 0, "no object was fetched");
        assert_eq!(
            downloaded_bytes(first.path(), &root.layers),
            downloaded_bytes(second.path(), &root.layers)
        );

        rt.block_on(prune_layer_store(store.path(), &Root::default()));
        assert_eq!(fs::read_dir(store.path()).expect("store").count(), 0);
    }

    /// A build kept before it is written serves the download of the root
    /// it publishes with no fetch.
    #[test]
    fn a_kept_build_serves_the_download_of_its_root_without_a_fetch() {
        let dir = TempDir::new().expect("tempdir");
        let policy = BuildPolicy {
            slice_target_bytes: 2048,
            tier_entries_per_term: 100_000,
            spill_to_disk: true,
            scratch_root: Some(dir.path().to_path_buf()),
            ..BuildPolicy::default()
        };
        let built = build(&[range_contribution(&dir, 1)], &policy).expect("build");
        let objects = TempDir::new().expect("objects");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(objects.path()).expect("local"));
        let faults = FaultStorage::wrap(Arc::clone(&local));
        let faulty = Arc::clone(&faults) as Arc<dyn StorageProvider>;
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let store = TempDir::new().expect("store");

        rt.block_on(keep_built(store.path(), &built.slices, &built.tiers));
        let reference = rt.block_on(write_built(&local, built)).expect("write");
        let root = rt
            .block_on(load_root(local.as_ref(), &reference))
            .expect("root");
        faults.fail(FaultOp::Get, STORAGE_PREFIX, usize::MAX);
        let scratch = TempDir::new().expect("scratch");
        rt.block_on(download_layers(
            &faulty,
            &root.layers,
            scratch.path(),
            Some(store.path()),
        ))
        .expect("served from the store");
        assert_eq!(faults.fired(), 0, "no object was fetched");
    }
}
