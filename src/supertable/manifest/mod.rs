// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! In-memory manifest types: `ManifestSnapshot`, `SuperfileEntry`,
//! `VectorSummary`. Per-column skip stats live on `SuperfileEntry` as
//! `HashMap<String, ScalarStatsAgg>` (scalar) and
//! `HashMap<String, FtsSummaryAgg>` (FTS).
//!
//! `ManifestSnapshot` is the single immutable point-in-time view of which
//! superfiles exist. `Supertable` holds the current manifest behind
//! an `ArcSwap<ManifestSnapshot>`; commits build a new `ManifestSnapshot` (superfiles:
//! old + new) and atomically swap it in. Readers
//! `ArcSwap::load_full` once at construction to pin a snapshot for
//! the lifetime of their queries.
//!
//! ## Construction is copy-on-write
//!
//! `ManifestSnapshot::with_appended` clones the outer `Vec` and shares each
//! existing `Arc<SuperfileEntry>` between the old and new manifests,
//! so the only per-commit allocation is the new entries plus the
//! `Vec` header. `ManifestSnapshot` itself is immutable — never mutated in
//! place — which is what makes lock-free reader-writer isolation
//! possible.

pub mod aggregates;
pub mod bloom;
pub mod commit;
pub mod disk_cache;
pub mod encoding;
pub mod hll;
pub mod list;
pub mod list_prune;
pub mod options_hash;
pub mod part;
pub mod partition;
pub mod term_range;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
    ops::Deref,
    sync::{Arc, OnceLock},
};

use arrow::compute::kernels::aggregate as agg;
use arrow_array::*;
use arrow_schema::{DataType, TimeUnit};
use dashmap::DashMap;
use futures::future;
/// Re-export the per-column skip aggregates so callers can refer to them as
/// `manifest::ScalarStatsAgg` / `manifest::FtsSummaryAgg` (the value types of
/// `SuperfileEntry.scalar_stats` / `SuperfileEntry.fts_summary`).
pub use list::{FtsSummaryAgg, ScalarStatsAgg};
use tokio::sync::OnceCell;
use uuid::Uuid;
use xxhash_rust::xxh3::xxh3_64;

use super::options::SupertableOptions;
use crate::{
    storage::{StorageError, StorageProvider},
    superfile::vector::distance::{
        COSINE_DISTANCE_BASE, L2_CROSS_TERM_COEFF, Metric, sq8_dot, u8_sum_sumsq,
    },
    supertable::{
        CommitError,
        error::ManifestError,
        manifest::{
            commit::{
                EncodedPart, PointerFile, frame_content_size, part_uri, read_pointer,
                translate_contention, write_manifest, write_part_bytes, write_pointer,
            },
            disk_cache::ManifestDiskCache,
            list::{
                FORMAT_VERSION as LIST_FORMAT_VERSION, Manifest, ManifestPartEntry,
                PartitionStrategy,
            },
            part::{ContentHash, ManifestPart, PartId},
            partition::{assign_partition, encode_partition_key},
        },
        query::{hierarchical_iter, prune::PruneLeaf},
    },
};

/// Zstd compression level for manifest parts and the manifest list.
/// Level 3 is zstd's own default — a balanced ratio/speed point that
/// keeps commit latency low while compressing the Avro-encoded
/// manifest well. (Valid range is 1..=22.)
pub const MANIFEST_ZSTD_LEVEL: i32 = 3;

/// One immutable point-in-time view of the supertable.
///
/// **Construction is copy-on-write.** Adding a superfile via
/// [`ManifestSnapshot::with_appended`] returns a new `ManifestSnapshot` whose
/// `superfiles` is `Vec::clone()` + new entries appended; the original
/// `ManifestSnapshot`'s `superfiles` is unchanged. `Arc<SuperfileEntry>` shares
/// the underlying entries between the old and new manifests so the
/// only per-commit allocation is the outer `Vec` and the new
/// entries themselves.
///
/// **Reader isolation.** Readers `ArcSwap::load_full` an
/// `Arc<ManifestSnapshot>` at construction and hold it for their lifetime.
/// New commits don't affect them. Old manifests are dropped
/// automatically once no reader holds an Arc to them.
///
/// `ManifestSnapshot` is the outer hierarchical wrapper (it adds the
/// `list` / `parts` / `loader` persistence-side fields);
/// `SuperfileList` is the flat in-process view that `ManifestSnapshot`
/// derefs to, so callers can access `.manifest_id`,
/// `.superfiles[i]`, `.n_docs_total()` etc. directly through a
/// `ManifestSnapshot`.
#[derive(Debug, Clone)]
pub struct SuperfileList {
    /// Monotonic point-in-time identifier. Starts at 0 (empty
    /// initial manifest from `Supertable::create`); each commit
    /// derives `manifest_id = old.manifest_id + 1`. With a single
    /// writer at a time, no separate counter or atomic is needed —
    /// the read-then-store sequence is exclusive by construction.
    pub manifest_id: u64,
    /// Pointer back to the immutable per-supertable configuration.
    /// Same Arc across all manifests of one supertable.
    pub options: Arc<SupertableOptions>,
    /// Append-only list of superfile entries. Each entry's `Arc`-share
    /// is what makes the copy-on-write per-commit construction
    /// cheap.
    pub superfiles: Vec<Arc<SuperfileEntry>>,
}

impl SuperfileList {
    /// Empty initial state at `manifest_id = 0`.
    pub fn empty(options: Arc<SupertableOptions>) -> Self {
        Self {
            manifest_id: 0,
            options,
            superfiles: Vec::new(),
        }
    }

    /// Build a successor SuperfileList with `new_entries` appended to
    /// the end of `superfiles`. Original is unchanged. `manifest_id`
    /// of the result is `self.manifest_id + 1`.
    pub fn with_appended(&self, new_entries: Vec<Arc<SuperfileEntry>>) -> Self {
        let mut superfiles = self.superfiles.clone();
        superfiles.extend(new_entries);
        Self {
            manifest_id: self.manifest_id + 1,
            options: self.options.clone(),
            superfiles,
        }
    }

    /// Total documents across all superfiles.
    pub fn n_docs_total(&self) -> u64 {
        self.superfiles.iter().map(|s| s.n_docs).sum()
    }
}

/// The hierarchical manifest. Outer wrapper around the
/// [`SuperfileList`] (flat in-process view) plus the
/// persistence-side metadata:
///
/// - `list`: the [`Manifest`] when this manifest was loaded
///   from / persisted to storage. `None` for in-process-only
///   supertables (no storage attached).
/// - `parts`: per-part lazy-load cache. `OnceCell` per part
///   coalesces concurrent `part(id)` calls into a single
///   `StorageProvider::get` — 100 query tasks on a cold part
///   issue exactly one load.
/// - `loader`: pulls part bytes through the storage provider
///   and verifies content hash. `None` when no storage is
///   attached (the in-process-only path).
///
/// `Deref` exposes the [`SuperfileList`] fields directly so
/// `manifest.manifest_id`, `manifest.superfiles[i]`,
/// `manifest.n_docs_total()` etc. work through a `ManifestSnapshot`
/// reference.
///
/// [`Manifest`]: list::Manifest
pub struct ManifestSnapshot {
    superfile_list: SuperfileList,
    list: Option<Manifest>,
    parts: DashMap<PartId, Arc<OnceCell<Arc<ManifestPart>>>>,
    loader: Option<Arc<ManifestPartLoader>>,
}

impl fmt::Debug for ManifestSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ManifestSnapshot")
            .field("manifest_id", &self.superfile_list.manifest_id)
            .field("n_superfiles", &self.superfile_list.superfiles.len())
            .field("has_list", &self.list.is_some())
            .field(
                "n_parts",
                &self.list.as_ref().map(|l| l.parts.len()).unwrap_or(0),
            )
            .field("n_parts_loaded", &self.parts.len())
            .field("has_loader", &self.loader.is_some())
            .finish()
    }
}

impl Deref for ManifestSnapshot {
    type Target = SuperfileList;
    fn deref(&self) -> &Self::Target {
        &self.superfile_list
    }
}

impl ManifestSnapshot {
    pub fn new(
        manifest_id: u64,
        options: Arc<SupertableOptions>,
        superfile_list: Vec<Arc<SuperfileEntry>>,
        storage: Option<Arc<dyn StorageProvider>>,
        list: Option<Manifest>,
    ) -> Self {
        let superfile_list = SuperfileList {
            manifest_id,
            options,
            superfiles: superfile_list,
        };
        if let Some(storage) = storage
            && let Some(list) = list
        {
            let manifest_cache = superfile_list.options.manifest_disk_cache.clone();
            let loader = Arc::new(ManifestPartLoader::new_with_cache(
                Arc::clone(&storage),
                &list,
                manifest_cache,
            ));
            Self {
                superfile_list,
                list: Some(list),
                parts: DashMap::new(),
                loader: Some(loader),
            }
        } else {
            Self {
                superfile_list,
                list: None,
                parts: DashMap::new(),
                loader: None,
            }
        }
    }

    #[cfg(test)]
    pub fn new_from_superfiles(
        opts: Arc<SupertableOptions>,
        superfiles: Vec<Arc<SuperfileEntry>>,
    ) -> Self {
        ManifestSnapshot::empty(opts).with_appended(superfiles)
    }

    /// Empty initial manifest at `manifest_id = 0`. Used by
    /// `Supertable::create` when no storage is attached.
    pub fn empty(options: Arc<SupertableOptions>) -> Self {
        Self {
            superfile_list: SuperfileList::empty(options),
            list: None,
            parts: DashMap::new(),
            loader: None,
        }
    }

    /// A *materialized* empty manifest at `manifest_id = 0`, ready to persist.
    /// Unlike [`Self::empty`] (which is in-process-only, `list: None`), this
    /// carries a `Some(list)`, so [`Self::write`] emits the initial (empty)
    /// manifest list + pointer. `Supertable::create` persists this on durable
    /// storage so the table is openable immediately — before its first append,
    /// after a reopen, or from another process — without shifting the
    /// `manifest_id` sequence (the first append still commits `manifest_id 1`).
    pub(crate) fn materialized_empty(options: Arc<SupertableOptions>) -> Self {
        let strategy = options.effective_partition_strategy();
        let list = Self::build_list(&options, strategy, 0, Vec::new(), BTreeMap::new());
        let loader = options.storage.as_ref().map(|storage| {
            Arc::new(ManifestPartLoader::new_with_cache(
                storage.clone(),
                &list,
                options.manifest_disk_cache.clone(),
            ))
        });
        Self {
            superfile_list: SuperfileList::empty(options),
            list: Some(list),
            parts: DashMap::new(),
            loader,
        }
    }

    /// Build a manifest list from the supertable `options` at `manifest_id`,
    /// carrying `parts` and the tombstone-seq state. Shared by the commit path
    /// ([`Self::update`]) and the initial empty-manifest materialization
    /// ([`Self::materialized_empty`]) so the options→list field mapping lives
    /// in one place.
    fn build_list(
        options: &SupertableOptions,
        strategy: PartitionStrategy,
        manifest_id: u64,
        parts: Vec<ManifestPartEntry>,
        tombstone_seqs: BTreeMap<Uuid, u64>,
    ) -> Manifest {
        Manifest {
            format_version: LIST_FORMAT_VERSION.into(),
            manifest_id,
            options_hash: options_hash::compute_options_hash(options, &strategy),
            schema: Vec::new(),
            id_column: options.id_column.clone(),
            fts_columns: options
                .fts_columns
                .iter()
                .map(|f| list::FtsColumnInfo {
                    column: f.column.clone(),
                })
                .collect(),
            vector_columns: options
                .vector_columns
                .iter()
                .map(|v| list::VectorColumnInfo {
                    column: v.column.clone(),
                    dim: v.dim,
                    n_cent: v.n_cent,
                    rot_seed: v.rot_seed,
                    metric: format!("{:?}", v.metric).to_lowercase(),
                })
                .collect(),
            partition_strategy: strategy,
            parts,
            tombstone_seqs,
        }
    }

    pub fn get_manifest_id(&self) -> u64 {
        self.superfile_list.manifest_id
    }

    pub fn get_next_manifest_id(&self) -> u64 {
        self.get_manifest_id() + 1
    }

    pub fn get_opts(&self) -> Arc<SupertableOptions> {
        self.superfile_list.options.clone()
    }

    pub fn get_partition_strategy(&self) -> PartitionStrategy {
        self.list
            .as_ref()
            .map(|l| l.partition_strategy.clone())
            .unwrap_or(self.superfile_list.options.effective_partition_strategy())
    }

    pub fn get_num_parts(&self) -> usize {
        self.list.as_ref().map(|l| l.parts.len()).unwrap_or(0)
    }

    pub fn get_num_parts_loaded(&self) -> usize {
        self.parts.len()
    }

    pub fn is_in_process_only(&self) -> bool {
        self.list.is_none()
    }

    pub fn get_cached_part_by_id(&self, part_id: &PartId) -> Option<Arc<ManifestPart>> {
        self.parts
            .get(part_id)
            .and_then(|cell| cell.value().get().cloned())
    }

    pub fn get_cached_part_by_list_idx(&self, idx: usize) -> Option<Arc<ManifestPart>> {
        let Some(list) = &self.list else {
            return None;
        };
        let part_id = list.parts[idx].part_id;
        self.get_cached_part_by_id(&part_id)
    }

    /// Load the committed manifest from storage.
    ///
    /// A genuinely absent pointer is [`ManifestLoadError::PointerNotFound`]:
    /// `Supertable::create` persists the initial (empty) pointer, so a
    /// registered table always has one. A missing pointer is therefore the
    /// open-or-create trigger for a never-created table, or a *lost* pointer
    /// on a created one — either way an error the caller sees, never a
    /// silently-empty table (which would mask committed-then-lost data). A
    /// *corrupt* pointer is a different error variant (`PointerParse`) and
    /// also propagates, so corruption is never masked.
    pub(crate) async fn load(
        current_manifest: Option<Arc<Self>>,
        storage: Arc<dyn StorageProvider>,
        options: Option<Arc<SupertableOptions>>,
    ) -> Result<Arc<Self>, ManifestLoadError> {
        // 1. Read the pointer file.
        let (pointer, _) = match read_pointer(storage.as_ref()).await? {
            Some(p) => p,
            None => return Err(ManifestLoadError::PointerNotFound),
        };
        Self::load_with_pointer(current_manifest, storage, options, pointer).await
    }

    /// [`Self::load`] with the pointer already in hand. Split out so
    /// the refresh path can read the pointer itself (conditionally,
    /// via [`probe_pointer`]) and still share the list + parts
    /// loading below.
    pub(crate) async fn load_with_pointer(
        current_manifest: Option<Arc<Self>>,
        storage: Arc<dyn StorageProvider>,
        options: Option<Arc<SupertableOptions>>,
        pointer: PointerFile,
    ) -> Result<Arc<Self>, ManifestLoadError> {
        if let Some(current_manifest) = &current_manifest
            && current_manifest.superfile_list.manifest_id >= pointer.manifest_id
        {
            // Pointer hasn't advanced past our in-memory state —
            return Err(ManifestLoadError::AlreadyLoaded);
        }

        // 2. Load + parse the manifest list.
        let (list_bytes, _) = storage
            .get(&pointer.manifest_uri)
            .await
            .map_err(ManifestLoadError::Storage)?;
        let list = list::decode(&list_bytes).map_err(ManifestLoadError::ListParse)?;

        let options = if let Some(options) = options {
            options
        } else if let Some(current) = &current_manifest {
            current.options.clone()
        } else {
            return Err(ManifestLoadError::ContentHashMismatch {
                expected: "valid options".to_string(),
                actual: "None options".to_string(),
            });
        };

        // Verify the caller's options match the
        // manifest's stamped digest. The all-zero stored
        // hash bypasses validation (legacy + synthetic
        // fixtures).
        let expected_hash = options_hash::compute_options_hash(&options, &list.partition_strategy);
        if let Err(mismatch) = options_hash::verify_options_hash(expected_hash, list.options_hash) {
            return Err(ManifestLoadError::ContentHashMismatch {
                expected: mismatch.expected,
                actual: mismatch.actual,
            });
        }

        // 3. Build the loader, superfiles & parts
        let loader = Arc::new(ManifestPartLoader::new_with_cache(
            Arc::clone(&storage),
            &list,
            options.manifest_disk_cache.clone(),
        ));
        let parts: DashMap<_, _> = DashMap::new();
        let mut all_superfiles: Vec<Arc<SuperfileEntry>> = Vec::new();
        if let Some(current_manifest) = &current_manifest {
            // If we have an existing manifest, populate `parts` with
            // existing entries and track missing part IDs for lazy-load.
            let mut missing_part_ids = Vec::new();
            for entry in &list.parts {
                if let Some(existing) = current_manifest.parts.get(&entry.part_id) {
                    parts.insert(entry.part_id, existing.value().clone());
                } else {
                    missing_part_ids.push(entry.part_id);
                }
            }

            let threshold = options.eager_load_threshold_parts as usize;
            let eager = list.parts.len() <= threshold;

            if eager {
                let load_futs = missing_part_ids
                    .iter()
                    .map(|id| {
                        let loader = Arc::clone(&loader);
                        let pid = *id;
                        async move { loader.load(pid).await }
                    })
                    .collect::<Vec<_>>();
                let loaded = future::join_all(load_futs).await;
                for (pid, result) in missing_part_ids.iter().zip(loaded) {
                    let part = result?;
                    let cell = OnceCell::new();
                    cell.set(part).expect("fresh cell");
                    parts.insert(*pid, Arc::new(cell));
                }
                for entry in &list.parts {
                    let cell = parts.get(&entry.part_id).expect("part inserted above");
                    let part = cell
                        .value()
                        .get()
                        .expect("eager-fetched or inherited; must be set");
                    all_superfiles.extend(part.superfiles.iter().cloned());
                }
            } else {
                for pid in &missing_part_ids {
                    parts.insert(*pid, Arc::new(OnceCell::new()));
                }
            }
        } else {
            let n_parts = list.parts.len();
            let threshold = options.eager_load_threshold_parts as usize;
            let eager = n_parts <= threshold;
            if eager {
                // eager-fetching every part (small manifests — fast first query)
                // parallel-fetch every part + populate
                // the flat superfile_list.superfiles view so the
                // iteration-style query paths (`bm25_search`,
                // `vector_search`, `query_sql`) see all superfiles
                // without going through the hierarchical iterator.
                let part_ids: Vec<_> = list.parts.iter().map(|p| p.part_id).collect();
                let load_futs = part_ids
                    .iter()
                    .map(|id| {
                        let loader = Arc::clone(&loader);
                        let pid = *id;
                        async move { loader.load(pid).await }
                    })
                    .collect::<Vec<_>>();
                let loaded = future::join_all(load_futs).await;
                for (pid, result) in part_ids.iter().zip(loaded) {
                    let part = result?;
                    all_superfiles.extend(part.superfiles.iter().cloned());
                    let cell = OnceCell::new();
                    cell.set(part).expect("fresh OnceCell");
                    parts.insert(*pid, Arc::new(cell));
                }
            } else {
                // Lazy path: each part gets an empty
                // `OnceCell`; first `Manifest::part(id).await`
                // triggers a single storage GET for that part.
                // `superfile_list.superfiles` stays empty — legacy
                // flat-iteration queries return zero results
                // until the hierarchical query path lands.
                // Callers in lazy mode today drive
                // `Manifest::part().await` directly.
                for entry in &list.parts {
                    parts.insert(entry.part_id, Arc::new(OnceCell::new()));
                }
            }
        }

        let mut new_superfile_list = SuperfileList::empty(options.clone());
        new_superfile_list.manifest_id = pointer.manifest_id;
        new_superfile_list.superfiles = all_superfiles;
        let new_manifest = ManifestSnapshot {
            superfile_list: new_superfile_list,
            list: Some(list),
            parts,
            loader: Some(loader),
        };

        Ok(Arc::new(new_manifest))
    }

    /// Commit a new manifest version.
    ///
    /// Orchestrates the four-step sequence:
    ///
    /// 1. **In parallel** — write each new manifest part + write
    ///    the new manifest list. Independent of each other; the
    ///    list references parts by URI (= blake3 of bytes,
    ///    computed before any I/O). Issued via
    ///    [`futures::future::join_all`].
    /// 2. Await all of the above (visibility barrier #1: parts
    ///    and list must be durable before the pointer publishes).
    /// 3. Build the new pointer file (manifest_id, list_uri,
    ///    list_content_hash).
    /// 4. Conditional pointer-PUT (visibility barrier #2: the
    ///    rename is the only thing readers observe).
    ///
    /// `parts_to_write` should contain **only the parts that need
    /// to be persisted** (i.e., new + changed). Each element is the
    /// pre-encoded (Avro+zstd) bytes produced by [`rebuild_part_and_entry`]
    /// — passing them directly avoids a second encode cycle.
    /// Reused parts from the previous manifest version are not in this
    /// list — their URIs are already in `new_list.parts[i].uri`.
    pub async fn write(
        &self,
        storage: &dyn StorageProvider,
        expected_prev_etag: Option<&str>,
        parts_to_write: &[&[u8]],
    ) -> Result<(), CommitError> {
        let Some(list_to_write) = self.list.as_ref() else {
            return Ok(());
        };
        // Step 1+2: parallel write of (list, parts).
        //
        // Both futures are independent — the list's references to
        // each part's URI are content-addressable from the
        // in-memory bytes before any I/O, so there's no
        // happens-before edge between them.
        let list_fut = write_manifest(storage, list_to_write);
        let part_futs = parts_to_write
            .iter()
            .map(|encoded| write_part_bytes(storage, encoded));
        let part_join = future::join_all(part_futs);

        let (list_res, part_results) = tokio::join!(list_fut, part_join);
        // Translate `Storage(PreconditionFailed)` from sub-writes
        // into `WriteContentionExhausted` so callers (and the
        // writer's OCC retry loop) can match on one variant
        // regardless of which CAS lost the race — list or pointer.
        let list_res = list_res.map_err(translate_contention)?;
        for part_result in part_results {
            part_result.map_err(translate_contention)?;
        }

        // Step 3: build pointer.
        let pointer = PointerFile {
            manifest_id: self.get_manifest_id(),
            manifest_uri: list_res.uri,
            content_hash: list_res.content_hash,
        };

        // Step 4: conditional pointer write — the visibility
        // barrier. Until this succeeds, no reader sees the new
        // manifest version.
        write_pointer(storage, &pointer, expected_prev_etag).await?;
        Ok(())
    }

    pub fn get_all_superfiles(&self) -> &[Arc<SuperfileEntry>] {
        &self.superfile_list.superfiles
    }

    pub(crate) async fn get_pruned_superfiles(
        &self,
        leaves: &[PruneLeaf],
    ) -> Result<Vec<Arc<SuperfileEntry>>, ManifestLoadError> {
        match &self.list {
            Some(list) => {
                // Intersect each constraining leaf's kept-part set. A leaf
                // with no part pruner (`None`) imposes no constraint.
                let mut kept: Option<HashSet<PartId>> = None;
                for leaf in leaves {
                    if let Some(part_ids) = leaf.keep_parts(list) {
                        let set: HashSet<PartId> = part_ids.into_iter().collect();
                        kept = Some(match kept {
                            None => set,
                            Some(existing) => existing.intersection(&set).copied().collect(),
                        });
                    }
                }
                // Preserve manifest (time) order of the surviving parts.
                let ordered: Vec<PartId> = match kept {
                    Some(set) => list
                        .parts
                        .iter()
                        .map(|p| p.part_id)
                        .filter(|id| set.contains(id))
                        .collect(),
                    None => list.parts.iter().map(|p| p.part_id).collect(),
                };
                hierarchical_iter::load_and_flatten(self, &ordered).await
            }
            None => Ok(hierarchical_iter::fallback_to_flat_superfiles(self)),
        }
    }

    pub(crate) async fn get_pruned_superfiles_for_vector(
        &self,
        column: &str,
        query: &[f32],
    ) -> Result<Vec<Arc<SuperfileEntry>>, ManifestLoadError> {
        match &self.list {
            Some(list) => {
                let kept = list_prune::prune_parts_for_vector(list, column, query, f32::INFINITY);
                hierarchical_iter::load_and_flatten(self, &kept).await
            }
            None => Ok(hierarchical_iter::fallback_to_flat_superfiles(self)),
        }
    }

    pub fn get_all_list_entries(&self) -> &[ManifestPartEntry] {
        match &self.list {
            Some(list) => &list.parts,
            None => &[],
        }
    }

    /// Build a successor manifest with `new_entries` appended.
    /// Preserves the persistence-side metadata (`list`, `loader`)
    /// from the predecessor; the per-part cache is fresh (an empty
    /// `DashMap`) because the parts referenced by the new version
    /// may differ. Cross-version part inheritance via content-
    /// addressed `Arc::clone` lives in `Supertable::refresh`.
    pub fn with_appended(&self, new_entries: Vec<Arc<SuperfileEntry>>) -> Self {
        Self {
            superfile_list: self.superfile_list.with_appended(new_entries),
            list: self.list.clone(),
            parts: DashMap::new(),
            loader: self.loader.clone(),
        }
    }

    /// The persisted list's per-superfile tombstone-seq map. `None`
    /// for in-process-only manifests (no persisted list ⇒ no sidecars
    /// can exist).
    pub fn get_tombstone_seqs(&self) -> Option<&BTreeMap<Uuid, u64>> {
        self.list.as_ref().map(|l| &l.tombstone_seqs)
    }

    /// Build a successor manifest identical to `self` except that every
    /// superfile in `touched` has its tombstone seq set to the successor's
    /// `manifest_id`. This is the mutation pipeline's post-sidecar stamp:
    /// no superfile entries or parts change, so persisting the successor
    /// is a list + pointer write only (empty `parts_to_write`).
    ///
    /// Returns `None` for in-process-only manifests (no persisted list —
    /// nothing to stamp, and no cross-process readers to inform).
    pub(crate) fn with_tombstone_seqs_bumped(&self, touched: &[Uuid]) -> Option<Self> {
        let list = self.list.as_ref()?;
        let next_id = self.get_next_manifest_id();
        let mut new_list = list.clone();
        new_list.manifest_id = next_id;
        for id in touched {
            new_list.tombstone_seqs.insert(*id, next_id);
        }
        let mut superfile_list = self.superfile_list.clone();
        superfile_list.manifest_id = next_id;
        // Same parts as the predecessor — inherit the loaded-part cache
        // wholesale so the stamp never forces a part re-fetch.
        let parts = DashMap::new();
        for kv in self.parts.iter() {
            parts.insert(*kv.key(), Arc::clone(kv.value()));
        }
        Some(Self {
            superfile_list,
            list: Some(new_list),
            parts,
            loader: self.loader.clone(),
        })
    }

    /// Lazy-load entry point for manifest parts.
    ///
    /// Concurrent callers on the same not-yet-loaded `part_id`
    /// share a single `StorageProvider::get` via the per-part
    /// `tokio::sync::OnceCell` — 100 concurrent queries on a
    /// cold part see exactly one load.
    ///
    /// Errors:
    /// - `OpenError::Build(BuildError::Store(...))` if no loader
    ///   is attached (in-process-only manifest).
    /// - `OpenError::ContentHashMismatch` if the loaded part's
    ///   blake3 doesn't match the manifest list's recorded hash.
    /// - `OpenError::ManifestPartParse { … }` for Avro / zstd
    ///   decode failures.
    pub async fn get_part_by_id(
        &self,
        part_id: PartId,
    ) -> Result<Arc<ManifestPart>, ManifestLoadError> {
        let loader = self
            .loader
            .as_ref()
            .ok_or(ManifestLoadError::NoLoaderAttached)?;
        let cell = self
            .parts
            .entry(part_id)
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();
        let loaded = cell.get_or_try_init(|| loader.load(part_id)).await?;
        Ok(Arc::clone(loaded))
    }

    /// Returns the new ManifestPartEntries when `new_entries` are added to `old` manifest. This
    /// operation may create new ManifestParts. The function also returns the new ManifestParts that
    /// the caller can decide to write to storage.
    pub async fn update(
        &self,
        new_entries: &[Arc<SuperfileEntry>],
        entries_to_remove: &[Arc<SuperfileEntry>],
    ) -> Result<(ManifestSnapshot, Vec<EncodedPart>), ManifestError> {
        // 1. Resolve the effective partition strategy. Locked at
        //    first commit: read from the existing manifest list
        //    if present, else use the options default.
        let opts = self.get_opts();
        let strategy = self.get_partition_strategy();

        // 2. Stamp each new entry with its partition key — this also validates
        //    against the strategy (surfaces SuperfileSpansPartition /
        //    unsupported-column-type / missing-partition_hint at commit). The
        //    partition lives on the ENTRY, not the part: parts are size-bucketed
        //    at the table level (below), so a part spans partitions and carries
        //    no key of its own. A reader recovers each superfile's partition
        //    from its entry in the part, with no data-file open. Assignment
        //    runs at commit time, so e.g. IngestionTime resolves to the current
        //    day bucket.
        //
        //    Entries must arrive unstamped (empty partition_key): the key is
        //    derived here, and every source of new entries — the writer,
        //    compaction (a merged superfile is a fresh entry), WAL replay —
        //    builds them with an empty key. A non-empty key would mean an
        //    earlier stage stamped it; committing would silently re-derive and
        //    overwrite that assignment (e.g. shifting an IngestionTime entry to
        //    the current day), so reject it instead.
        let new_entries: Vec<Arc<SuperfileEntry>> = new_entries
            .iter()
            .map(|e| {
                if !e.partition_key.is_empty() {
                    return Err(ManifestError::EntryAlreadyPartitioned {
                        detail: format!(
                            "superfile {} arrived with a partition_key already set",
                            e.superfile_id
                        ),
                    });
                }
                let pk = assign_partition(e, &strategy)?;
                Ok(Arc::new(SuperfileEntry {
                    partition_key: encode_partition_key(&pk),
                    ..(**e).clone()
                }))
            })
            .collect::<Result<_, ManifestError>>()?;

        // 3. One table-level lineage: new entries append to the last (latest)
        //    part — rewriting it, or splitting into a fresh part when it would
        //    exceed target_superfiles_per_part — while earlier parts carry over
        //    unchanged (same content-hash + URI, no re-encode / PUT). When there
        //    is no prior part, the new entries form the first one.
        let list_entries = self.get_all_list_entries();
        let latest_idx = list_entries.len().checked_sub(1);
        let mut out_list_entries: Vec<ManifestPartEntry> = Vec::new();
        let mut parts_to_write: Vec<EncodedPart> = Vec::new();
        let mut pending_new = new_entries.to_vec();

        for (i, entry) in list_entries.iter().enumerate() {
            if Some(i) != latest_idx || pending_new.is_empty() {
                out_list_entries.push(entry.clone());
                continue;
            }
            let new_for_part = std::mem::take(&mut pending_new);
            let combined_n = entry.n_superfiles + new_for_part.len() as u64;
            if combined_n > self.superfile_list.options.target_superfiles_per_part {
                // Split: keep the existing part, emit a fresh part for the new
                // superfiles.
                out_list_entries.push(entry.clone());
                let (fresh_entry, fresh_part, fresh_encoded) =
                    rebuild_part_and_entry(opts.clone(), vec![], new_for_part, None);
                out_list_entries.push(fresh_entry);
                parts_to_write.push(EncodedPart {
                    part: fresh_part,
                    encoded: fresh_encoded,
                });
            } else {
                // Rewrite the latest part = its existing superfiles + the new.
                let existing_part = self.get_part_by_id(entry.part_id).await?;
                let (rebuilt_entry, rebuilt_part, rebuilt_encoded) = rebuild_part_and_entry(
                    opts.clone(),
                    existing_part.superfiles.clone(),
                    new_for_part,
                    Some(entry),
                );
                out_list_entries.push(rebuilt_entry);
                parts_to_write.push(EncodedPart {
                    part: rebuilt_part,
                    encoded: rebuilt_encoded,
                });
            }
        }

        // Cold start: no prior parts, so the new entries form the first part.
        if !pending_new.is_empty() {
            let (fresh_entry, fresh_part, fresh_encoded) =
                rebuild_part_and_entry(opts.clone(), vec![], pending_new, None);
            out_list_entries.push(fresh_entry);
            parts_to_write.push(EncodedPart {
                part: fresh_part,
                encoded: fresh_encoded,
            });
        }
        let mut out_list_entries_after_removal = Vec::new();
        if entries_to_remove.is_empty() {
            out_list_entries_after_removal = out_list_entries;
        } else {
            // 4. Apply removals across every part: drop the removed superfile_ids
            //    wherever they live; a part with no match is left untouched.
            let removal_ids = entries_to_remove
                .iter()
                .map(|r| r.superfile_id)
                .collect::<HashSet<_>>();
            for entry in out_list_entries {
                // TODO: Handle merging 2 parts into one if their sum is within threshold

                // Fetch the part's current superfiles — from parts_to_write (freshly
                // rebuilt this commit) or the prior manifest.
                let (superfile_entries_in_part, existing_part_to_update) = if let Some(existing) =
                    parts_to_write
                        .iter_mut()
                        .find(|ep| ep.part.part_id == entry.part_id)
                {
                    (existing.part.superfiles.clone(), Some(existing))
                } else if let Ok(existing_part) = self.get_part_by_id(entry.part_id).await {
                    (existing_part.superfiles.clone(), None)
                } else {
                    return Err(ManifestError::UnknownPartId(entry.part_id));
                };
                let final_superfile_entries = superfile_entries_in_part
                    .iter()
                    .filter(|s| !removal_ids.contains(&s.superfile_id))
                    .cloned()
                    .collect::<Vec<_>>();

                // No superfile removed from this part → keep it unchanged.
                if final_superfile_entries.len() == superfile_entries_in_part.len() {
                    out_list_entries_after_removal.push(entry);
                    continue;
                }

                let (fresh_entry, fresh_part, fresh_encoded) =
                    rebuild_part_and_entry(opts.clone(), vec![], final_superfile_entries, None);

                if let Some(existing) = existing_part_to_update {
                    *existing = EncodedPart {
                        part: fresh_part,
                        encoded: fresh_encoded,
                    };
                } else {
                    parts_to_write.push(EncodedPart {
                        part: fresh_part,
                        encoded: fresh_encoded,
                    });
                }

                out_list_entries_after_removal.push(fresh_entry);
            }
        }

        let ids_to_remove = entries_to_remove
            .iter()
            .map(|e| e.superfile_id)
            .collect::<HashSet<_>>();

        // Carry the tombstone-seq map forward, dropping entries for
        // superfiles this commit removes — their sidecars leave the
        // manifest with them.
        let mut tombstone_seqs = self
            .list
            .as_ref()
            .map(|list| list.tombstone_seqs.clone())
            .unwrap_or_default();
        tombstone_seqs.retain(|id, _| !ids_to_remove.contains(id));

        let new_list = Self::build_list(
            opts.as_ref(),
            strategy,
            self.get_next_manifest_id(),
            out_list_entries_after_removal,
            tombstone_seqs,
        );
        let mut new_superfile_list = self
            .get_all_superfiles()
            .iter()
            .chain(new_entries.iter())
            .map(Arc::clone)
            .collect::<Vec<_>>();
        new_superfile_list.retain(|e| !ids_to_remove.contains(&e.superfile_id));

        let new_superfile_list = SuperfileList {
            manifest_id: self.get_next_manifest_id(),
            options: self.get_opts(),
            superfiles: new_superfile_list,
        };
        let loader = opts.storage.as_ref().map(|storage| {
            Arc::new(ManifestPartLoader::new_with_cache(
                storage.clone(),
                &new_list,
                opts.manifest_disk_cache.clone(),
            ))
        });
        // Inherit only the cached parts the new list still
        // references — entries for rewritten/removed parts are
        // dropped rather than carried forward, so the in-memory
        // parts cache can't grow without bound across commits.
        // Surviving parts keep their warm cache entry (no refetch);
        // the freshly-written parts are seeded below.
        let live_part_ids: HashSet<_> = new_list.parts.iter().map(|e| e.part_id).collect();
        let parts = DashMap::new();
        for kv in self.parts.iter() {
            if live_part_ids.contains(kv.key()) {
                parts.insert(*kv.key(), kv.value().clone());
            }
        }
        for part in parts_to_write.iter() {
            let part = part.part.clone();
            parts.insert(
                part.part_id,
                Arc::new(OnceCell::new_with(Some(Arc::new(part)))),
            );
        }

        let new_manifest = ManifestSnapshot {
            superfile_list: new_superfile_list,
            list: Some(new_list),
            parts,
            loader,
        };

        Ok((new_manifest, parts_to_write))
    }
}

/// build one `ManifestPart` from `superfiles` + the
/// matching `ManifestListEntry`. Encodes the part once,
/// content-hashes it, and computes the list-level aggregate
/// skip summaries that `list_prune` reads at query time.
/// If base_part is Some, the superfiles MUST only include the new superfiles to be added.
fn rebuild_part_and_entry(
    opts: Arc<SupertableOptions>,
    old_superfiles: Vec<Arc<SuperfileEntry>>,
    new_superfiles: Vec<Arc<SuperfileEntry>>,
    base_part: Option<&ManifestPartEntry>,
) -> (
    ManifestPartEntry,
    ManifestPart,
    Vec<u8>, // pre-encoded compressed bytes — reused by write path, no second encode
) {
    let _ = opts; // reserved for future per-options encoding tweaks (zstd level, etc.)

    let aggregates = crate::supertable::manifest::aggregates::compute(&new_superfiles, base_part);
    let superfiles = old_superfiles
        .into_iter()
        .chain(new_superfiles)
        .collect::<Vec<_>>();
    let part = ManifestPart {
        format_version: part::FORMAT_VERSION.into(),
        part_id: PartId::new_v4(),
        superfiles,
    };
    let compressed = part::encode(&part, MANIFEST_ZSTD_LEVEL);
    let size_compressed = compressed.len() as u64;
    let content_hash = ContentHash::of(&compressed);
    let size_uncompressed = frame_content_size(&compressed, size_compressed);
    let entry = ManifestPartEntry {
        part_id: part.part_id,
        uri: part_uri(&content_hash),
        n_superfiles: part.superfiles.len() as u64,
        size_bytes_compressed: size_compressed,
        size_bytes_uncompressed: size_uncompressed,
        content_hash,
        id_range: aggregates.id_range,
        scalar_stats_agg: aggregates.scalar_stats_agg,
        fts_summary_agg: aggregates.fts_summary_agg,
        vector_summary_agg: aggregates.vector_summary_agg,
    };
    (entry, part, compressed)
}

/// Pulls manifest parts through a [`StorageProvider`] and verifies
/// content-hash on load.
///
/// One `ManifestPartLoader` per `Manifest`. The same `Arc<dyn
/// StorageProvider>` is shared with the `DiskCacheStore` —
/// one auth handshake, one connection pool.
///
/// An optional [`ManifestDiskCache`] short-circuits the storage GET
/// when the part's compressed bytes are already on local disk. Because
/// parts are content-addressed, a cache hit can never be stale.
pub struct ManifestPartLoader {
    storage: Arc<dyn StorageProvider>,
    /// Maps `PartId → (expected content_hash, uri)`. Built from
    /// the manifest list at construction; immutable per-`Manifest`.
    parts_index: HashMap<PartId, (ContentHash, String)>,
    /// On-disk cache for compressed part bytes. `None` disables the
    /// cache (in-process-only supertables, tests, or storage attached
    /// without a `disk_cache_root` configured).
    manifest_disk_cache: Option<Arc<ManifestDiskCache>>,
}

impl ManifestPartLoader {
    pub fn new(storage: Arc<dyn StorageProvider>, list: &Manifest) -> Self {
        Self::new_with_cache(storage, list, None)
    }

    /// Like [`Self::new`] but attaches an on-disk part-bytes cache.
    pub fn new_with_cache(
        storage: Arc<dyn StorageProvider>,
        list: &Manifest,
        manifest_disk_cache: Option<Arc<ManifestDiskCache>>,
    ) -> Self {
        let mut idx = HashMap::with_capacity(list.parts.len());
        for entry in &list.parts {
            idx.insert(entry.part_id, (entry.content_hash, entry.uri.clone()));
        }
        Self {
            storage,
            parts_index: idx,
            manifest_disk_cache,
        }
    }

    /// Fetch + verify + decode one part. Returns the parsed
    /// `Arc<ManifestPart>`.
    ///
    /// Consults the on-disk cache first (a hit skips the storage GET);
    /// on a miss the freshly-fetched bytes are written back to the
    /// cache (best-effort) before decoding.
    pub async fn load(&self, part_id: PartId) -> Result<Arc<ManifestPart>, ManifestLoadError> {
        let (expected_hash, uri) = self
            .parts_index
            .get(&part_id)
            .ok_or(ManifestLoadError::PartNotInList { part_id })?;

        // Disk-cache hit: bytes are verified against `expected_hash`
        // inside `get`, so they're known-good here.
        if let Some(cache) = &self.manifest_disk_cache
            && let Some(bytes) = cache.get(expected_hash).await
        {
            let parsed = part::decode(&bytes)?;
            return Ok(Arc::new(parsed));
        }

        let (bytes, _) = self
            .storage
            .get(uri)
            .await
            .map_err(ManifestLoadError::Storage)?;
        let actual_hash = ContentHash::of(&bytes);
        if actual_hash != *expected_hash {
            return Err(ManifestLoadError::ContentHashMismatch {
                expected: expected_hash.to_hex(),
                actual: actual_hash.to_hex(),
            });
        }
        // Populate the cache for next time (best-effort; the hash is
        // already verified, satisfying `put`'s contract).
        if let Some(cache) = &self.manifest_disk_cache {
            cache.put(actual_hash, &bytes).await;
        }
        let parsed = part::decode(&bytes)?;
        Ok(Arc::new(parsed))
    }
}

/// Errors raised by [`Manifest::part`] and [`ManifestPartLoader::load`].
///
/// Standalone (not folded into the supertable-level
/// `OpenError`) so the per-part load surface stays narrowly
/// testable in isolation.
#[derive(Debug, thiserror::Error)]
pub enum ManifestLoadError {
    /// Pointer not found in storage.
    #[error("pointer not found in storage")]
    PointerNotFound,
    #[error("already loaded")]
    AlreadyLoaded,
    /// Pointer parse error.
    #[error("pointer parse error: {0}")]
    PointerParse(String),
    /// Caller invoked `Manifest::part(...)` on an in-process-only
    /// manifest (no storage attached). The hierarchical manifest
    /// has no on-disk parts to load from.
    #[error("no storage / loader attached to this manifest")]
    NoLoaderAttached,

    #[error("list parse error: {0}")]
    ListParse(#[source] list::ListParseError),
    /// `part_id` isn't in this manifest's list. Either the caller
    /// passed a stale id (pre-refresh) or the manifest list is
    /// missing an entry.
    #[error("part_id not in manifest list: {part_id}")]
    PartNotInList { part_id: PartId },
    /// Storage backend returned an error.
    #[error("storage error during part load: {0}")]
    Storage(#[source] StorageError),
    /// Computed blake3 of the loaded bytes didn't match the
    /// manifest list's recorded `content_hash`. The bad bytes
    /// are **not** auto-refetched — a mismatch indicates
    /// corruption, not a transient race, so it's surfaced as
    /// a caller-visible failure rather than papered over.
    #[error("content-hash mismatch: expected {expected}, got {actual}")]
    ContentHashMismatch { expected: String, actual: String },
    /// Avro / zstd / version-incompat parse failure.
    #[error("part parse failed")]
    Parse(#[from] part::PartParseError),
}

/// One superfile's metadata + skip-pruning summaries. The bytes that
/// back the superfile live in the superfile store keyed by `uri` —
/// `superfile_id` is for debugging / observability, `uri` is for
/// store routing.
#[derive(Debug, Clone)]
pub struct SuperfileEntry {
    /// Globally unique identifier (UUID v4) for debugging /
    /// observability. Distinct from `uri` so the store routing key
    /// can evolve independently of identity.
    pub superfile_id: Uuid,
    /// Opaque key into the `SuperfileReaderCache`. v1 wraps a UUID; the
    /// trait doesn't care about the internal shape.
    pub uri: SuperfileUri,
    /// Row count.
    pub n_docs: u64,
    /// id-column min and max (the supertable-injected
    /// `Decimal128(38, 0)` id column). Stored as `i128` to
    /// carry the 128-bit Snowflake-shaped values produced by
    /// the supertable's `IdGenerator`. Signed-int comparison
    /// gives time-ordered skip-pruning because the high bit
    /// stays 0 for any plausible current-era timestamp.
    pub id_min: i128,
    pub id_max: i128,
    /// Per-scalar-column aggregate (min/max + null count, exact sum, HLL),
    /// keyed by column name, for skip pruning of SQL filters. An absent
    /// column means "no usable stats" (the pruner keeps the superfile).
    pub scalar_stats: HashMap<String, ScalarStatsAgg>,
    /// Per-FTS-column term-presence bloom + lex range. The bloom
    /// drives exact-term skip; the term-range drives prefix-query
    /// skip via `[prefix, prefix_upper_bound)` overlap. Keyed by
    /// FTS column name. Same per-column [`FtsSummaryAgg`] shape the
    /// list-level aggregate uses; built per superfile via
    /// [`FtsSummaryAgg::from_superfile`].
    pub fts_summary: HashMap<String, FtsSummaryAgg>,
    /// Per-vector-column centroid + radius. Drives vector skip via
    /// triangle-inequality against the bounding sphere. Keyed by
    /// vector column name.
    pub vector_summary: HashMap<String, VectorSummary>,
    /// Partition assignment, encoded opaquely per the strategy
    /// (time_range = 8-byte LE u64 bucket index; hash = 4-byte LE
    /// u32 bucket id; column_range = 2-byte LE u16 boundary index).
    /// Empty (decoded as "unpartitioned") when no real partition
    /// strategy is configured; otherwise filled by the writer
    /// from the configured strategy at commit time.
    pub partition_key: Vec<u8>,
    /// Hash partitioning operates per-row, but at commit time we
    /// only have per-superfile summaries. Hash strategy requires
    /// superfiles to be pre-sharded — each builder-shard stamps the
    /// resulting bucket here on ingest. `None` under non-hash
    /// strategies and under the single-bucket Hash default.
    pub partition_hint: Option<u32>,
    /// precomputed superfile layout offsets so the
    /// cold-open path can fire the parquet-footer, vector
    /// subsection, and FTS subsection GETs **in parallel** in a
    /// single round-trip, without first reading the parquet KV
    /// metadata to learn where each subsection lives.
    ///
    /// Populated by the writer at commit time from the
    /// `ParquetParts` returned by `splice_index_blobs` (so
    /// the values are by construction consistent with what the
    /// parquet KV metadata would later say).
    ///
    /// `None` on superfiles produced by older writers that did not
    /// stamp this field; the cold open path falls back to the
    /// 2-RTT shape (parquet tail
    /// then vec/fts in parallel) — see
    /// `DiskCacheStore::reader_with_hints`.
    pub subsection_offsets: Option<SubsectionOffsets>,
}

/// superfile layout offsets cached on the manifest.
///
/// Knowing these up-front lets the cold-open path issue every
/// subsection GET in parallel against the same superfile object,
/// turning the canonical 2-RTT cold open (parquet tail → vec+fts
/// in parallel) into a single round-trip.
///
/// All offsets are absolute byte positions within the superfile
/// blob (matching `inf.vec.offset` / `inf.fts.offset` parquet KV
/// values), and `total_size` matches what an S3 `HEAD` would
/// return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubsectionOffsets {
    /// Total byte count of the superfile blob. Lets the cold-open
    /// path skip the upfront `HEAD` round-trip too — the same
    /// information the suffix-range tail would otherwise return,
    /// but available without any I/O.
    pub total_size: u64,
    /// Absolute `(offset, length)` of the vector subsection. `None`
    /// when the superfile carries no vector subsection.
    pub vec: Option<(u64, u64)>,
    /// Absolute `(offset, length)` of the FTS subsection. `None`
    /// when the superfile carries no FTS subsection.
    pub fts: Option<(u64, u64)>,
    /// Absolute ranges that fully cover vector open-time metadata.
    /// The hinted cache path prefetches these in the first network
    /// batch so `VectorReader::open_lazy` can resolve header,
    /// directory, subheaders, and codec metadata from the overlay.
    pub vec_open_ranges: Vec<(u64, u64)>,
    /// Absolute ranges that fully cover FTS open-time metadata:
    /// header+dictionary and doc-length tables. Query-time postings
    /// stay lazy.
    pub fts_open_ranges: Vec<(u64, u64)>,
    /// the actual bytes covering the superfile's
    /// open-time batch (parquet footer tail + the
    /// `vec_open_ranges` + the `fts_open_ranges`), carried inline
    /// in the manifest part.
    ///
    /// When non-empty, the cold-fetch path installs these directly
    /// into the reader's prefetch overlay and issues **zero**
    /// open-time GETs against the superfile object — the bytes
    /// already arrived in the single part GET that `cold_open`
    /// performs. The genuine first-touch per-superfile cost then
    /// collapses from 2 RTT-batches (open metadata + cluster
    /// postings) to 1 (postings only).
    ///
    /// Each tuple is `(absolute_offset, bytes)`. Empty on superfiles
    /// produced by older writers that did not capture it, or when
    /// blob capture is disabled
    /// — the path then falls back to fetching `vec_open_ranges` /
    /// `fts_open_ranges` over the wire.
    pub open_blob: Vec<(u64, Vec<u8>)>,
}

/// Opaque store key — wraps a UUID v4. The superfile store treats
/// this as a hash-eq token and doesn't peek inside. An
/// object-store-backed variant could swap to a path-shaped URI
/// without changing any caller, since the trait shape stays the
/// same.
#[derive(Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct SuperfileUri(pub Uuid);

impl SuperfileUri {
    /// Generate a fresh URI. Called by the writer at commit time
    /// when assigning a key for a new superfile's bytes.
    pub fn new_v4() -> Self {
        Self(Uuid::new_v4())
    }

    /// Object-store / LocalFS path for committed superfile bytes.
    /// `.sf.parquet` double suffix — on disk this is still valid
    /// Parquet (row groups + optional embedded FTS/vector blobs +
    /// footer), while the `.sf` marker flags it as a Superfile
    /// superfile without making the file look non-standard.
    pub fn storage_path(self) -> String {
        format!("data/seg-{}.sf.parquet", self.0)
    }

    /// Disk-cache filename for a promoted superfile.
    pub fn cache_filename(self) -> String {
        format!("seg-{}.sf.parquet", self.0)
    }

    /// Disk-cache tempfile while a cold fetch is in flight.
    pub fn cache_tmp_filename(self) -> String {
        format!("seg-{}.sf.parquet.tmp", self.0)
    }
}

/// Merge min/max arrays by comparing values and keeping the actual min and max.
///
/// Takes existing (min, max) and other (min, max) arrays and returns the
/// merged (min, max) where min is the smaller value and max is the larger.
/// Both arrays are assumed to be length-1 and of the same type.
pub(crate) fn merge_min_max_arrays(
    existing_min: &ArrayRef,
    other_min: &ArrayRef,
    existing_max: &ArrayRef,
    other_max: &ArrayRef,
) -> Option<(ArrayRef, ArrayRef)> {
    // Merge two optional bounds into the surviving one — smaller for a min,
    // larger for a max. A `None` (the column is all-null on that side)
    // yields to a present value; both `None` stays `None`, so an all-null
    // column stays all-null through the fold, while a manifest part with any
    // populated superfile keeps its real bound.
    #[inline]
    fn merge_opt<T: PartialOrd>(a: Option<T>, b: Option<T>, keep_min: bool) -> Option<T> {
        match (a, b) {
            (Some(a), Some(b)) => Some(if keep_min == (a <= b) { a } else { b }),
            (Some(v), None) | (None, Some(v)) => Some(v),
            (None, None) => None,
        }
    }

    macro_rules! prim_merge {
        ($array_ty:ty) => {{
            let exn = existing_min.as_any().downcast_ref::<$array_ty>()?;
            let otn = other_min.as_any().downcast_ref::<$array_ty>()?;
            let exx = existing_max.as_any().downcast_ref::<$array_ty>()?;
            let otx = other_max.as_any().downcast_ref::<$array_ty>()?;
            let at = |a: &$array_ty| (!a.is_null(0)).then(|| a.value(0));
            Some((
                Arc::new(<$array_ty>::from(vec![merge_opt(at(exn), at(otn), true)])) as ArrayRef,
                Arc::new(<$array_ty>::from(vec![merge_opt(at(exx), at(otx), false)])) as ArrayRef,
            ))
        }};
    }

    // Same fold, re-attaching the column's timezone (the constructor drops
    // it) so the merged bound keeps the exact type its inputs carried.
    macro_rules! ts_merge {
        ($array_ty:ty, $tz:expr) => {{
            let exn = existing_min.as_any().downcast_ref::<$array_ty>()?;
            let otn = other_min.as_any().downcast_ref::<$array_ty>()?;
            let exx = existing_max.as_any().downcast_ref::<$array_ty>()?;
            let otx = other_max.as_any().downcast_ref::<$array_ty>()?;
            let at = |a: &$array_ty| (!a.is_null(0)).then(|| a.value(0));
            Some((
                Arc::new(
                    <$array_ty>::from(vec![merge_opt(at(exn), at(otn), true)])
                        .with_timezone_opt($tz.clone()),
                ) as ArrayRef,
                Arc::new(
                    <$array_ty>::from(vec![merge_opt(at(exx), at(otx), false)])
                        .with_timezone_opt($tz.clone()),
                ) as ArrayRef,
            ))
        }};
    }

    match existing_min.data_type() {
        DataType::UInt8 => prim_merge!(UInt8Array),
        DataType::UInt16 => prim_merge!(UInt16Array),
        DataType::UInt32 => prim_merge!(UInt32Array),
        DataType::UInt64 => prim_merge!(UInt64Array),
        DataType::Int8 => prim_merge!(Int8Array),
        DataType::Int16 => prim_merge!(Int16Array),
        DataType::Int32 => prim_merge!(Int32Array),
        DataType::Int64 => prim_merge!(Int64Array),
        DataType::Float32 => prim_merge!(Float32Array),
        DataType::Float64 => prim_merge!(Float64Array),
        // `false < true`, so min folds like AND and max like OR.
        DataType::Boolean => prim_merge!(BooleanArray),
        DataType::Utf8 => {
            let exn = existing_min.as_any().downcast_ref::<StringArray>()?;
            let otn = other_min.as_any().downcast_ref::<StringArray>()?;
            let exx = existing_max.as_any().downcast_ref::<StringArray>()?;
            let otx = other_max.as_any().downcast_ref::<StringArray>()?;
            let min = merge_opt(
                (!exn.is_null(0)).then(|| exn.value(0)),
                (!otn.is_null(0)).then(|| otn.value(0)),
                true,
            );
            let max = merge_opt(
                (!exx.is_null(0)).then(|| exx.value(0)),
                (!otx.is_null(0)).then(|| otx.value(0)),
                false,
            );
            Some((
                Arc::new(StringArray::from(vec![min])),
                Arc::new(StringArray::from(vec![max])),
            ))
        }
        DataType::LargeUtf8 => {
            let exn = existing_min.as_any().downcast_ref::<LargeStringArray>()?;
            let otn = other_min.as_any().downcast_ref::<LargeStringArray>()?;
            let exx = existing_max.as_any().downcast_ref::<LargeStringArray>()?;
            let otx = other_max.as_any().downcast_ref::<LargeStringArray>()?;
            let min = merge_opt(
                (!exn.is_null(0)).then(|| exn.value(0)),
                (!otn.is_null(0)).then(|| otn.value(0)),
                true,
            );
            let max = merge_opt(
                (!exx.is_null(0)).then(|| exx.value(0)),
                (!otx.is_null(0)).then(|| otx.value(0)),
                false,
            );
            Some((
                Arc::new(LargeStringArray::from(vec![min])),
                Arc::new(LargeStringArray::from(vec![max])),
            ))
        }
        DataType::Decimal128(precision, scale) => {
            let exn = existing_min.as_any().downcast_ref::<Decimal128Array>()?;
            let otn = other_min.as_any().downcast_ref::<Decimal128Array>()?;
            let exx = existing_max.as_any().downcast_ref::<Decimal128Array>()?;
            let otx = other_max.as_any().downcast_ref::<Decimal128Array>()?;
            let at = |a: &Decimal128Array| (!a.is_null(0)).then(|| a.value(0));
            let min = merge_opt(at(exn), at(otn), true);
            let max = merge_opt(at(exx), at(otx), false);
            Some((
                Arc::new(
                    Decimal128Array::from(vec![min])
                        .with_precision_and_scale(*precision, *scale)
                        .ok()?,
                ),
                Arc::new(
                    Decimal128Array::from(vec![max])
                        .with_precision_and_scale(*precision, *scale)
                        .ok()?,
                ),
            ))
        }

        // Mirror `column_min_max`'s temporal set; without these arms a
        // multi-superfile (or compacted) temporal column errors the merge and
        // drops its stat, silently regressing the fold and range prune.
        DataType::Date32 => prim_merge!(Date32Array),
        DataType::Date64 => prim_merge!(Date64Array),
        DataType::Time32(TimeUnit::Second) => prim_merge!(Time32SecondArray),
        DataType::Time32(TimeUnit::Millisecond) => prim_merge!(Time32MillisecondArray),
        DataType::Time64(TimeUnit::Microsecond) => prim_merge!(Time64MicrosecondArray),
        DataType::Time64(TimeUnit::Nanosecond) => prim_merge!(Time64NanosecondArray),
        DataType::Timestamp(TimeUnit::Second, tz) => ts_merge!(TimestampSecondArray, tz),
        DataType::Timestamp(TimeUnit::Millisecond, tz) => ts_merge!(TimestampMillisecondArray, tz),
        DataType::Timestamp(TimeUnit::Microsecond, tz) => ts_merge!(TimestampMicrosecondArray, tz),
        DataType::Timestamp(TimeUnit::Nanosecond, tz) => ts_merge!(TimestampNanosecondArray, tz),
        _ => None,
    }
}

/// Compute (min, max) for one Arrow array as length-1 `ArrayRef`s.
///
/// Returns `None` only for unsupported types. An all-null input of a
/// supported type yields length-1 *null* min/max arrays (not `None`), so
/// its null count is still recorded and `IS [NOT] NULL` can prune on it.
/// Supported set: integer (signed + unsigned, all widths), float
/// (f32, f64), boolean, Utf8, LargeUtf8, Decimal128, and temporal
/// (Date32/64, Time32/64, Timestamp). The supertable schema
/// rejects vector columns up at the SupertableOptions layer, so
/// `FixedSizeList<Float32>` won't appear here in practice.
/// Exact column sum as a length-1 array typed to match SQL `SUM`'s
/// result for the column (signed → `Int64`, unsigned → `UInt64`,
/// floats → `Float64`). `None` for non-summable types (utf8, bool,
/// decimal) or when the exact total overflows the result type —
/// consumers treat missing as "no statistics".
pub(crate) fn column_sum(col: &ArrayRef) -> Option<ArrayRef> {
    macro_rules! signed {
        ($array_ty:ty) => {{
            let a = col.as_any().downcast_ref::<$array_ty>()?;
            let total: i128 = a.iter().flatten().map(i128::from).sum();
            let v = i64::try_from(total).ok()?;
            Some(Arc::new(Int64Array::from(vec![v])) as ArrayRef)
        }};
    }
    macro_rules! unsigned {
        ($array_ty:ty) => {{
            let a = col.as_any().downcast_ref::<$array_ty>()?;
            let total: u128 = a.iter().flatten().map(u128::from).sum();
            let v = u64::try_from(total).ok()?;
            Some(Arc::new(UInt64Array::from(vec![v])) as ArrayRef)
        }};
    }
    macro_rules! float {
        ($array_ty:ty) => {{
            let a = col.as_any().downcast_ref::<$array_ty>()?;
            let total: f64 = a.iter().flatten().map(f64::from).sum();
            Some(Arc::new(Float64Array::from(vec![total])) as ArrayRef)
        }};
    }

    match col.data_type() {
        DataType::Int8 => signed!(Int8Array),
        DataType::Int16 => signed!(Int16Array),
        DataType::Int32 => signed!(Int32Array),
        DataType::Int64 => signed!(Int64Array),
        DataType::UInt8 => unsigned!(UInt8Array),
        DataType::UInt16 => unsigned!(UInt16Array),
        DataType::UInt32 => unsigned!(UInt32Array),
        DataType::UInt64 => unsigned!(UInt64Array),
        DataType::Float32 => float!(Float32Array),
        DataType::Float64 => float!(Float64Array),
        _ => None,
    }
}

/// Add two length-1 sum arrays of the same type (see [`column_sum`]).
/// `None` on type mismatch or `Int64`/`UInt64` overflow. Shared with
/// the SQL provider's cross-segment statistics fold.
pub(crate) fn add_sum_arrays(a: &ArrayRef, b: &ArrayRef) -> Option<ArrayRef> {
    match (a.data_type(), b.data_type()) {
        (DataType::Int64, DataType::Int64) => {
            let x = a.as_any().downcast_ref::<Int64Array>()?.value(0);
            let y = b.as_any().downcast_ref::<Int64Array>()?.value(0);
            Some(Arc::new(Int64Array::from(vec![x.checked_add(y)?])) as ArrayRef)
        }
        (DataType::UInt64, DataType::UInt64) => {
            let x = a.as_any().downcast_ref::<UInt64Array>()?.value(0);
            let y = b.as_any().downcast_ref::<UInt64Array>()?.value(0);
            Some(Arc::new(UInt64Array::from(vec![x.checked_add(y)?])) as ArrayRef)
        }
        (DataType::Float64, DataType::Float64) => {
            let x = a.as_any().downcast_ref::<Float64Array>()?.value(0);
            let y = b.as_any().downcast_ref::<Float64Array>()?.value(0);
            Some(Arc::new(Float64Array::from(vec![x + y])) as ArrayRef)
        }
        _ => None,
    }
}

/// HyperLogLog distinct sketch over a column's non-null values.
/// `None` for types the sketch doesn't cover. Values hash by their
/// canonical byte representation (little-endian for numerics, raw
/// bytes for strings, IEEE bits for floats).
pub(crate) fn column_hll(col: &ArrayRef) -> Option<hll::HllSketch> {
    let mut sketch = hll::HllSketch::new();
    macro_rules! ints {
        ($array_ty:ty) => {{
            let a = col.as_any().downcast_ref::<$array_ty>()?;
            for v in a.iter().flatten() {
                sketch.insert_hash(xxh3_64(&v.to_le_bytes()));
            }
        }};
    }
    match col.data_type() {
        DataType::Int8 => ints!(Int8Array),
        DataType::Int16 => ints!(Int16Array),
        DataType::Int32 => ints!(Int32Array),
        DataType::Int64 => ints!(Int64Array),
        DataType::UInt8 => ints!(UInt8Array),
        DataType::UInt16 => ints!(UInt16Array),
        DataType::UInt32 => ints!(UInt32Array),
        DataType::UInt64 => ints!(UInt64Array),
        DataType::Float32 => {
            let a = col.as_any().downcast_ref::<Float32Array>()?;
            for v in a.iter().flatten() {
                sketch.insert_hash(xxh3_64(&v.to_bits().to_le_bytes()));
            }
        }
        DataType::Float64 => {
            let a = col.as_any().downcast_ref::<Float64Array>()?;
            for v in a.iter().flatten() {
                sketch.insert_hash(xxh3_64(&v.to_bits().to_le_bytes()));
            }
        }
        DataType::Utf8 => {
            let a = col.as_any().downcast_ref::<StringArray>()?;
            for v in a.iter().flatten() {
                sketch.insert_hash(xxh3_64(v.as_bytes()));
            }
        }
        DataType::LargeUtf8 => {
            let a = col.as_any().downcast_ref::<LargeStringArray>()?;
            for v in a.iter().flatten() {
                sketch.insert_hash(xxh3_64(v.as_bytes()));
            }
        }
        _ => return None,
    }
    Some(sketch)
}

pub(crate) fn column_min_max(col: &ArrayRef) -> Option<(ArrayRef, ArrayRef)> {
    // An all-null column yields *null* min/max (not no-stat), so its null
    // count is still recorded for `IS [NOT] NULL` pruning — hence the
    // `Option` min/max are kept rather than `?`-unwrapped away.
    macro_rules! prim {
        ($array_ty:ty) => {{
            let a = col.as_any().downcast_ref::<$array_ty>()?;
            let mn_arr: ArrayRef = Arc::new(<$array_ty>::from(vec![agg::min(a)]));
            let mx_arr: ArrayRef = Arc::new(<$array_ty>::from(vec![agg::max(a)]));
            Some((mn_arr, mx_arr))
        }};
    }

    // Timestamps are the same primitive fold, but the `from(vec![..])`
    // constructor builds a zone-less array; re-attach the column's zone so
    // the bound keeps its exact type (a naive-vs-zoned mismatch would fail
    // the cross-superfile merge and stat reconstruction).
    macro_rules! ts {
        ($array_ty:ty, $tz:expr) => {{
            let a = col.as_any().downcast_ref::<$array_ty>()?;
            let mn_arr: ArrayRef =
                Arc::new(<$array_ty>::from(vec![agg::min(a)]).with_timezone_opt($tz.clone()));
            let mx_arr: ArrayRef =
                Arc::new(<$array_ty>::from(vec![agg::max(a)]).with_timezone_opt($tz.clone()));
            Some((mn_arr, mx_arr))
        }};
    }

    match col.data_type() {
        DataType::UInt8 => prim!(UInt8Array),
        DataType::UInt16 => prim!(UInt16Array),
        DataType::UInt32 => prim!(UInt32Array),
        DataType::UInt64 => prim!(UInt64Array),
        DataType::Int8 => prim!(Int8Array),
        DataType::Int16 => prim!(Int16Array),
        DataType::Int32 => prim!(Int32Array),
        DataType::Int64 => prim!(Int64Array),
        DataType::Float32 => prim!(Float32Array),
        DataType::Float64 => prim!(Float64Array),
        DataType::Boolean => {
            let a = col.as_any().downcast_ref::<BooleanArray>()?;
            Some((
                Arc::new(BooleanArray::from(vec![agg::min_boolean(a)])),
                Arc::new(BooleanArray::from(vec![agg::max_boolean(a)])),
            ))
        }
        DataType::Utf8 => {
            let a = col.as_any().downcast_ref::<StringArray>()?;
            Some((
                Arc::new(StringArray::from(vec![agg::min_string(a)])),
                Arc::new(StringArray::from(vec![agg::max_string(a)])),
            ))
        }
        DataType::LargeUtf8 => {
            let a = col.as_any().downcast_ref::<LargeStringArray>()?;
            Some((
                Arc::new(LargeStringArray::from(vec![agg::min_string(a)])),
                Arc::new(LargeStringArray::from(vec![agg::max_string(a)])),
            ))
        }
        DataType::Decimal128(precision, scale) => {
            let a = col.as_any().downcast_ref::<Decimal128Array>()?;
            Some((
                Arc::new(
                    Decimal128Array::from(vec![agg::min(a)])
                        .with_precision_and_scale(*precision, *scale)
                        .ok()?,
                ),
                Arc::new(
                    Decimal128Array::from(vec![agg::max(a)])
                        .with_precision_and_scale(*precision, *scale)
                        .ok()?,
                ),
            ))
        }

        // Temporal columns are numeric-backed and orderable, so min/max fold
        // (DataFusion's aggregate fast-path) and prune the same as integers.
        DataType::Date32 => prim!(Date32Array),
        DataType::Date64 => prim!(Date64Array),
        DataType::Time32(TimeUnit::Second) => prim!(Time32SecondArray),
        DataType::Time32(TimeUnit::Millisecond) => prim!(Time32MillisecondArray),
        DataType::Time64(TimeUnit::Microsecond) => prim!(Time64MicrosecondArray),
        DataType::Time64(TimeUnit::Nanosecond) => prim!(Time64NanosecondArray),
        DataType::Timestamp(TimeUnit::Second, tz) => ts!(TimestampSecondArray, tz),
        DataType::Timestamp(TimeUnit::Millisecond, tz) => ts!(TimestampMillisecondArray, tz),
        DataType::Timestamp(TimeUnit::Microsecond, tz) => ts!(TimestampMicrosecondArray, tz),
        DataType::Timestamp(TimeUnit::Nanosecond, tz) => ts!(TimestampNanosecondArray, tz),
        _ => None,
    }
}

/// Per-vector-column summary: cluster centroid + bounding radius.
/// Already produced by the superfile vector builder (per-column,
/// inside the vector blob's outer header KV metadata); the writer
/// copies them into the manifest at commit time. Vector skip uses
/// centroid + radius for triangle-inequality pruning of superfiles
/// whose bounding sphere is too far from a query to contain any
/// possible top-k hit.
#[derive(Debug, Clone)]
pub struct VectorSummary {
    /// Cluster centroid; length matches the vector column's `dim`
    /// declared in `SupertableOptions::vector_columns`.
    pub centroid: Vec<f32>,
    /// Maximum distance from any indexed vector in this superfile to
    /// `centroid`, in the same metric the column was built with.
    pub radius: f32,
    /// Per-cluster IVF centroids (Sq8, per-cluster calibration) for
    /// cross-superfile global cluster selection. Empty when the superfile
    /// has no vector index for this column.
    pub clusters: ClusterCentroids,
}

/// Maximum Sq8 code value. The manifest's per-cluster centroid
/// summary quantizes each component to a single unsigned byte, so
/// the per-cluster scale maps `[min, max]` onto `[0, SQ8_CODE_MAX]`.
const SQ8_CODE_MAX: f32 = 255.0;

/// Per-cluster IVF centroids for one vector column, Sq8-quantized with
/// per-cluster calibration. Carried in the manifest so a query can rank
/// every superfile's clusters globally — without opening the superfile —
/// and probe only the globally-closest clusters. The 1-bit shortlist +
/// rerank still run on the superfile's on-disk compressed vectors; these
/// drive cluster *selection* only.
///
/// Quantization is value-only (no metric); the selector applies the
/// column's metric when scoring a dequantized centroid against a query.
#[derive(Debug, Clone, Default)]
pub struct ClusterCentroids {
    pub n_cent: u32,
    pub dim: u32,
    /// `n_cent * dim` Sq8 codes, cluster-major.
    pub codes: Vec<u8>,
    /// Per-cluster dequant base (min component); length `n_cent`.
    pub mins: Vec<f32>,
    /// Per-cluster dequant step `(max - min) / 255`; length `n_cent`.
    pub scales: Vec<f32>,
    /// Per-cluster indexed doc count; length `n_cent`. Count-0 clusters
    /// are skipped by the selector.
    pub counts: Vec<u32>,
    /// Lazily-computed per-cluster `(Σcode, Σcode²)` — the
    /// query-independent moments the folded L2 scoring needs to
    /// reconstruct `‖centroid‖²` without dequantizing. Populated on
    /// first L2 query (one pass over `codes`), 8 bytes per cluster;
    /// never serialized (decode starts it empty).
    pub code_moments: OnceLock<Vec<(f32, f32)>>,
}

impl ClusterCentroids {
    /// The "no cluster centroids" value — a superfile without a vector
    /// index for the column.
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.n_cent == 0
    }

    /// Sq8-quantize fp32 cluster centroids (`centroids` is cluster-major,
    /// `n_cent * dim` floats) with per-cluster calibration: each cluster
    /// centroid spans the full 8-bit range against its own component
    /// min/max. `counts` is the per-cluster indexed doc count.
    pub fn from_fp32(n_cent: u32, dim: u32, centroids: &[f32], counts: Vec<u32>) -> Self {
        let nc = n_cent as usize;
        let d = dim as usize;
        let mut codes = vec![0u8; nc * d];
        let mut mins = vec![0f32; nc];
        let mut scales = vec![0f32; nc];
        for c in 0..nc {
            let src = &centroids[c * d..(c + 1) * d];
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for &v in src {
                mn = mn.min(v);
                mx = mx.max(v);
            }
            if !mn.is_finite() {
                mn = 0.0;
            }
            if !mx.is_finite() {
                mx = 0.0;
            }
            let scale = if mx > mn {
                (mx - mn) / SQ8_CODE_MAX
            } else {
                0.0
            };
            mins[c] = mn;
            scales[c] = scale;
            let dst = &mut codes[c * d..(c + 1) * d];
            for (o, &v) in dst.iter_mut().zip(src) {
                *o = if scale > 0.0 {
                    ((v - mn) / scale).round().clamp(0.0, SQ8_CODE_MAX) as u8
                } else {
                    0
                };
            }
        }
        Self {
            n_cent,
            dim,
            codes,
            mins,
            scales,
            counts,
            code_moments: OnceLock::new(),
        }
    }

    /// Score every populated cluster against `query` directly in the
    /// Sq8 code domain — no per-cluster dequantization, no scratch
    /// buffer. The affine dequant (`v_j = min + scale·code_j`) folds
    /// out of every metric:
    ///
    /// ```text
    /// dot(q, centroid) = min·Σq + scale·(q · codes)
    /// ‖centroid‖²      = d·min² + 2·min·scale·Σcode + scale²·Σcode²
    /// L2²(q, centroid) = ‖q‖² − 2·dot(q, centroid) + ‖centroid‖²
    /// ```
    ///
    /// so the only O(dim) work per cluster is one Sq8 dot product over
    /// the already-contiguous `codes` row — the same AVX-512 / AVX2 /
    /// `wide` kernel the rerank path uses. `sum_q` is `Σ query_j`;
    /// `norm_q_sq` is `‖query‖²` (read only for L2). Calls
    /// `emit(cluster_id, score)` for each cluster with a nonzero
    /// indexed count. Scores equal `dequantize_into` + `distance` up
    /// to f32 association order (gated by
    /// `score_clusters_into_matches_dequantized_distance`).
    pub fn score_clusters_into(
        &self,
        metric: Metric,
        query: &[f32],
        sum_q: f32,
        norm_q_sq: f32,
        mut emit: impl FnMut(u32, f32),
    ) {
        let d = self.dim as usize;
        debug_assert_eq!(query.len(), d);
        // L2 needs each cluster's query-independent code moments;
        // computed once per `ClusterCentroids` (first L2 query) so the
        // per-query, per-cluster O(dim) work stays a single Sq8 dot.
        let moments = matches!(metric, Metric::L2Sq).then(|| {
            self.code_moments.get_or_init(|| {
                (0..self.n_cent as usize)
                    .map(|c| u8_sum_sumsq(&self.codes[c * d..(c + 1) * d]))
                    .collect()
            })
        });
        for c in 0..self.n_cent as usize {
            if self.counts[c] == 0 {
                continue;
            }
            let codes = &self.codes[c * d..(c + 1) * d];
            let dot_qc = self.mins[c] * sum_q + self.scales[c] * sq8_dot(query, codes, d);
            let score = match metric {
                Metric::Cosine => COSINE_DISTANCE_BASE - dot_qc,
                Metric::NegDot => -dot_qc,
                Metric::L2Sq => {
                    let (sum_c, sumsq_c) = moments.expect("moments built for L2 above")[c];
                    let centroid_norm_sq = d as f32 * self.mins[c] * self.mins[c]
                        + L2_CROSS_TERM_COEFF * self.mins[c] * self.scales[c] * sum_c
                        + self.scales[c] * self.scales[c] * sumsq_c;
                    norm_q_sq - L2_CROSS_TERM_COEFF * dot_qc + centroid_norm_sq
                }
            };
            emit(c as u32, score);
        }
    }

    /// Dequantize cluster `c`'s centroid into `out` (length `dim`).
    pub fn dequantize_into(&self, c: usize, out: &mut [f32]) {
        let d = self.dim as usize;
        let codes = &self.codes[c * d..(c + 1) * d];
        let mn = self.mins[c];
        let scale = self.scales[c];
        for (o, &code) in out.iter_mut().zip(codes) {
            *o = mn + code as f32 * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{hint::black_box, slice::from_ref, sync::Arc, time::Instant};

    use arrow_array::{
        Array, Date32Array, Date64Array, Int64Array, Time64MicrosecondArray,
        TimestampMicrosecondArray,
    };
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use dashmap::DashMap;
    use datafusion::scalar::ScalarValue;
    use tempfile::TempDir;
    use tokio::sync::OnceCell;

    use super::*;
    use crate::{
        storage::LocalFsStorageProvider,
        superfile::{builder::FtsConfig, vector::distance::distance},
        supertable::manifest::{
            commit::{PartWriteResult, write_manifest_part},
            list::{Manifest, PartitionStrategy},
        },
        test_helpers::default_tokenizer,
    };

    /// Deterministic synthetic fp32 centroids for the folded-scoring
    /// tests: distinct per-cluster ranges so per-cluster Sq8
    /// calibration (min/scale) actually varies.
    fn synth_clusters(n_cent: u32, dim: u32, seed: u64) -> (ClusterCentroids, Vec<f32>) {
        let (nc, d) = (n_cent as usize, dim as usize);
        let mut centroids = vec![0f32; nc * d];
        for c in 0..nc {
            for j in 0..d {
                let v = ((seed + (c * d + j) as u64 * 2_654_435_761) % 1000) as f32 / 250.0 - 2.0
                    + c as f32 * 0.1;
                centroids[c * d + j] = v;
            }
        }
        let counts: Vec<u32> = (0..nc).map(|c| if c == nc / 2 { 0 } else { 10 }).collect();
        let cc = ClusterCentroids::from_fp32(n_cent, dim, &centroids, counts);
        (cc, centroids)
    }

    #[test]
    fn min_max_stats_record_and_fold_all_null_columns() {
        let arr = |vals: Vec<Option<i64>>| Arc::new(Int64Array::from(vals)) as ArrayRef;
        let scalar = |a: &ArrayRef| ScalarValue::try_from_array(a, 0).expect("decode");

        // All-null column still yields a stat, with null min/max (so the
        // null count is recorded and `IS [NOT] NULL` can prune on it).
        let (mn, mx) = column_min_max(&arr(vec![None, None])).expect("all-null stat");
        assert!(mn.is_null(0) && mx.is_null(0));

        // Populated column → real min/max, nulls ignored.
        let (mn, mx) = column_min_max(&arr(vec![Some(5), Some(2), None])).expect("stat");
        assert_eq!(scalar(&mn), ScalarValue::Int64(Some(2)));
        assert_eq!(scalar(&mx), ScalarValue::Int64(Some(5)));

        // Fold: a real bound wins over a null bound; both-null stays null.
        let null1 = arr(vec![None]);
        let (mn, mx) =
            merge_min_max_arrays(&null1, &arr(vec![Some(2)]), &null1, &arr(vec![Some(9)]))
                .expect("merge real over null");
        assert_eq!(scalar(&mn), ScalarValue::Int64(Some(2)));
        assert_eq!(scalar(&mx), ScalarValue::Int64(Some(9)));
        let (mn, mx) = merge_min_max_arrays(&null1, &null1, &null1, &null1).expect("merge null");
        assert!(mn.is_null(0) && mx.is_null(0));
    }

    #[test]
    fn min_max_stats_cover_temporal_columns() {
        let scalar = |a: &ArrayRef| ScalarValue::try_from_array(a, 0).expect("decode");

        // Date32 (the ClickBench `EventDate` case that used to carry no stat):
        // numeric-backed, so min/max record and fold like an integer.
        let d: ArrayRef = Arc::new(Date32Array::from(vec![Some(20100), Some(19000), None]));
        let (mn, mx) = column_min_max(&d).expect("date stat");
        assert_eq!(scalar(&mn), ScalarValue::Date32(Some(19000)));
        assert_eq!(scalar(&mx), ScalarValue::Date32(Some(20100)));

        // Two superfiles' date bounds fold to the outer extremes.
        let lo: ArrayRef = Arc::new(Date32Array::from(vec![Some(19000)]));
        let hi: ArrayRef = Arc::new(Date32Array::from(vec![Some(20100)]));
        let olo: ArrayRef = Arc::new(Date32Array::from(vec![Some(18000)]));
        let ohi: ArrayRef = Arc::new(Date32Array::from(vec![Some(21000)]));
        let (mn, mx) = merge_min_max_arrays(&lo, &olo, &hi, &ohi).expect("date merge");
        assert_eq!(scalar(&mn), ScalarValue::Date32(Some(18000)));
        assert_eq!(scalar(&mx), ScalarValue::Date32(Some(21000)));

        // Timestamp keeps its timezone through both build and merge. A
        // naive-vs-zoned mismatch would fail the merge and silently drop the
        // stat, so assert the type survives, not just the value.
        let tz = "+05:30";
        let ts: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![Some(200i64), Some(100)]).with_timezone(tz),
        );
        let zoned = DataType::Timestamp(TimeUnit::Microsecond, Some(tz.into()));
        let (mn, mx) = column_min_max(&ts).expect("ts stat");
        assert_eq!(mn.data_type(), &zoned);
        assert_eq!(
            scalar(&mn),
            ScalarValue::TimestampMicrosecond(Some(100), Some(tz.into()))
        );
        assert_eq!(
            scalar(&mx),
            ScalarValue::TimestampMicrosecond(Some(200), Some(tz.into()))
        );
        let (mmn, _mmx) = merge_min_max_arrays(&mn, &mn, &mx, &mx).expect("ts merge keeps tz");
        assert_eq!(mmn.data_type(), &zoned);

        // Date64 (ms-since-epoch) and Time64 ride the same `prim!` arm as
        // Date32; spot-check that each records and reconstructs to its type.
        let d64: ArrayRef = Arc::new(Date64Array::from(vec![Some(9i64), Some(2)]));
        assert_eq!(
            scalar(&column_min_max(&d64).expect("date64 stat").0),
            ScalarValue::Date64(Some(2))
        );
        let t64: ArrayRef = Arc::new(Time64MicrosecondArray::from(vec![Some(7i64), Some(3)]));
        assert_eq!(
            scalar(&column_min_max(&t64).expect("time64 stat").0),
            ScalarValue::Time64Microsecond(Some(3))
        );
    }

    /// Folded Sq8-domain scoring must equal dequantize-then-distance
    /// (the prior selection path) up to f32 association order, for all
    /// three metrics, and must skip count-0 clusters.
    #[test]
    fn score_clusters_into_matches_dequantized_distance() {
        let (n_cent, dim) = (17u32, 96u32);
        let (cc, _) = synth_clusters(n_cent, dim, 7);
        let query: Vec<f32> = (0..dim)
            .map(|j| ((j as u64 * 40_503 + 11) % 997) as f32 / 500.0 - 1.0)
            .collect();
        let sum_q: f32 = query.iter().sum();
        let norm_q_sq: f32 = query.iter().map(|v| v * v).sum();

        for metric in [Metric::Cosine, Metric::L2Sq, Metric::NegDot] {
            let mut folded: Vec<(u32, f32)> = Vec::new();
            cc.score_clusters_into(metric, &query, sum_q, norm_q_sq, |c, s| {
                folded.push((c, s));
            });

            // Reference: the old dequantize + distance loop.
            let mut deq = vec![0f32; dim as usize];
            let mut reference: Vec<(u32, f32)> = Vec::new();
            for c in 0..n_cent as usize {
                if cc.counts[c] == 0 {
                    continue;
                }
                cc.dequantize_into(c, &mut deq);
                reference.push((c as u32, distance(metric, &query, &deq)));
            }

            assert_eq!(
                folded.len(),
                reference.len(),
                "{metric:?}: cluster sets differ (count-0 skip)"
            );
            for ((fc, fs), (rc, rs)) in folded.iter().zip(&reference) {
                assert_eq!(fc, rc, "{metric:?}: cluster order");
                let tol = 1e-3 * (1.0 + rs.abs());
                assert!(
                    (fs - rs).abs() <= tol,
                    "{metric:?} cluster {fc}: folded {fs} vs dequantized {rs} (tol {tol})"
                );
            }
        }
    }

    /// Microbench: folded Sq8 scoring vs the old dequantize+distance
    /// loop over a supertable-scale cluster set. Gated `#[ignore]`;
    /// run via `cargo test --release --lib
    /// score_clusters_microbench -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    fn score_clusters_microbench() {
        let (n_cent, dim) = (4096u32, 384u32);
        let iters = 50usize;
        let (cc, _) = synth_clusters(n_cent, dim, 99);
        let query: Vec<f32> = (0..dim).map(|j| (j as f32).sin()).collect();
        let sum_q: f32 = query.iter().sum();
        let norm_q_sq: f32 = query.iter().map(|v| v * v).sum();

        for metric in [Metric::Cosine, Metric::L2Sq] {
            let t0 = Instant::now();
            for _ in 0..iters {
                let mut acc = 0f32;
                cc.score_clusters_into(metric, &query, sum_q, norm_q_sq, |_, s| acc += s);
                black_box(acc);
            }
            let folded_us = t0.elapsed().as_micros() as f64 / iters as f64;

            let mut deq = vec![0f32; dim as usize];
            let t0 = Instant::now();
            for _ in 0..iters {
                let mut acc = 0f32;
                for c in 0..n_cent as usize {
                    if cc.counts[c] == 0 {
                        continue;
                    }
                    cc.dequantize_into(c, &mut deq);
                    acc += distance(metric, &query, &deq);
                }
                black_box(acc);
            }
            let dequant_us = t0.elapsed().as_micros() as f64 / iters as f64;
            println!(
                "score_clusters {metric:?}: folded {folded_us:.0} µs vs dequantize {dequant_us:.0} µs ({:.1}×)",
                dequant_us / folded_us
            );
        }
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]))
    }

    fn opts() -> Arc<SupertableOptions> {
        let tk = default_tokenizer();
        Arc::new(
            SupertableOptions::new(
                schema(),
                vec![FtsConfig {
                    column: "title".into(),
                }],
                vec![],
                Some(tk),
            )
            .expect("valid options"),
        )
    }

    fn seg_entry(uuid: Uuid, n_docs: u64) -> Arc<SuperfileEntry> {
        Arc::new(SuperfileEntry {
            superfile_id: uuid,
            uri: SuperfileUri(uuid),
            n_docs,
            id_min: 0,
            id_max: n_docs.saturating_sub(1) as i128,
            scalar_stats: HashMap::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    #[test]
    fn empty_manifest_starts_at_zero() {
        let m = ManifestSnapshot::empty(opts());
        assert_eq!(m.manifest_id, 0);
        assert_eq!(m.superfiles.len(), 0);
        assert_eq!(m.n_docs_total(), 0);
    }

    #[test]
    fn with_appended_increments_manifest_id_and_extends_superfiles() {
        let m0 = ManifestSnapshot::empty(opts());
        let entry = seg_entry(Uuid::new_v4(), 100);
        let m1 = m0.with_appended(vec![entry.clone()]);
        assert_eq!(m1.manifest_id, 1);
        assert_eq!(m1.superfiles.len(), 1);
        assert_eq!(m1.n_docs_total(), 100);
        // Original m0 unchanged — the immutability invariant.
        assert_eq!(m0.manifest_id, 0);
        assert_eq!(m0.superfiles.len(), 0);
        assert_eq!(m0.n_docs_total(), 0);
    }

    #[test]
    fn with_appended_chains_to_higher_manifest_ids() {
        let m0 = ManifestSnapshot::empty(opts());
        let m1 = m0.with_appended(vec![seg_entry(Uuid::new_v4(), 50)]);
        let m2 = m1.with_appended(vec![seg_entry(Uuid::new_v4(), 75)]);
        assert_eq!(m0.manifest_id, 0);
        assert_eq!(m1.manifest_id, 1);
        assert_eq!(m2.manifest_id, 2);
        assert_eq!(m0.superfiles.len(), 0);
        assert_eq!(m1.superfiles.len(), 1);
        assert_eq!(m2.superfiles.len(), 2);
        assert_eq!(m2.n_docs_total(), 50 + 75);
    }

    #[test]
    fn with_appended_shares_old_superfiles_via_arc() {
        // The new manifest's superfiles[0] should be the SAME Arc as
        // the original's superfiles[0] — copy-on-write doesn't
        // re-allocate per-superfile. (Verified by Arc::ptr_eq.)
        let entry = seg_entry(Uuid::new_v4(), 1);
        let m0 = ManifestSnapshot::empty(opts()).with_appended(vec![entry.clone()]);
        let m1 = m0.with_appended(vec![seg_entry(Uuid::new_v4(), 2)]);
        assert!(Arc::ptr_eq(&m0.superfiles[0], &m1.superfiles[0]));
    }

    #[test]
    fn with_appended_empty_input_still_bumps_manifest_id() {
        // Edge case: with_appended(vec![]) is a no-op for superfiles
        // but should still produce a new manifest_id. (Whether this
        // is a "should" decision or "ok behavior" is fine here —
        // the writer won't call it with empty input in practice;
        // the test pins the current behavior.)
        let m0 = ManifestSnapshot::empty(opts());
        let m1 = m0.with_appended(vec![]);
        assert_eq!(m1.manifest_id, 1);
        assert_eq!(m1.superfiles.len(), 0);
    }

    #[test]
    fn new_from_superfiles_builds_manifest_at_id_one_with_entries() {
        // `new_from_superfiles` is `empty(opts).with_appended(...)`:
        // one append hop off the empty manifest, so manifest_id lands
        // at 1 and the manifest carries exactly the entries handed in.
        let a = seg_entry(Uuid::new_v4(), 10);
        let b = seg_entry(Uuid::new_v4(), 20);
        let m = ManifestSnapshot::new_from_superfiles(opts(), vec![a.clone(), b.clone()]);
        assert_eq!(m.manifest_id, 1);
        assert_eq!(m.superfiles.len(), 2);
        assert_eq!(m.n_docs_total(), 30);
        // Copy-on-write shares the passed-in Arcs rather than
        // re-allocating per-superfile.
        assert!(Arc::ptr_eq(&m.superfiles[0], &a));
        assert!(Arc::ptr_eq(&m.superfiles[1], &b));
        // No storage attached, so it's an in-process-only manifest
        // (no ManifestList / loader).
        assert!(m.is_in_process_only());
    }

    #[test]
    fn new_from_superfiles_with_empty_input_is_empty_at_id_one() {
        // Mirrors `with_appended(vec![])`: no superfiles, but the
        // single append hop still advances manifest_id to 1.
        let m = ManifestSnapshot::new_from_superfiles(opts(), vec![]);
        assert_eq!(m.manifest_id, 1);
        assert_eq!(m.superfiles.len(), 0);
        assert_eq!(m.n_docs_total(), 0);
    }

    #[test]
    fn get_next_manifest_id_is_current_plus_one() {
        let m0 = ManifestSnapshot::empty(opts());
        assert_eq!(m0.get_manifest_id(), 0);
        assert_eq!(m0.get_next_manifest_id(), 1);

        let m1 = m0.with_appended(vec![seg_entry(Uuid::new_v4(), 1)]);
        assert_eq!(m1.get_manifest_id(), 1);
        assert_eq!(m1.get_next_manifest_id(), 2);
    }

    #[test]
    fn get_next_manifest_id_is_a_pure_read() {
        // Querying the successor id is side-effect-free: the
        // manifest's own id is untouched and repeat calls are stable.
        let m = ManifestSnapshot::empty(opts());
        let _ = m.get_next_manifest_id();
        assert_eq!(m.get_manifest_id(), 0, "current id unchanged");
        assert_eq!(m.get_next_manifest_id(), m.get_next_manifest_id());
    }

    #[test]
    fn superfile_uri_is_distinct_per_call() {
        let a = SuperfileUri::new_v4();
        let b = SuperfileUri::new_v4();
        assert_ne!(a, b);
    }

    // ============================================================
    // In-memory `Manifest` with lazy-load parts — content-hash-
    // verified per-part fetch through an injected
    // `StorageProvider`, OnceCell coalescing on cold cells,
    // typed errors for missing loader / missing part / hash
    // mismatch.
    // ============================================================

    mod lazy_load {
        use std::{
            collections::HashMap,
            error::Error,
            ops::Range,
            slice::from_ref,
            sync::{
                Arc,
                atomic::{AtomicUsize, Ordering},
            },
            time::SystemTime,
        };

        use arrow_schema::{DataType, Field, Schema};
        use async_trait::async_trait;
        use bytes::Bytes;
        use dashmap::DashMap;
        use tokio::spawn;
        use uuid::Uuid;

        use super::super::*;
        use crate::{
            storage::{ObjectMeta, StorageError, StorageProvider},
            supertable::{
                SupertableOptions,
                manifest::{
                    list::{FORMAT_VERSION as LIST_FORMAT_VERSION, PartitionStrategy},
                    part::{self as part_mod, ContentHash, ManifestPart, PartId},
                },
            },
        };

        #[derive(Debug)]
        struct CountingMockStorage {
            objects: HashMap<String, Bytes>,
            get_calls: AtomicUsize,
        }

        impl CountingMockStorage {
            fn new(objects: HashMap<String, Bytes>) -> Self {
                Self {
                    objects,
                    get_calls: AtomicUsize::new(0),
                }
            }

            fn get_call_count(&self) -> usize {
                self.get_calls.load(Ordering::Acquire)
            }
        }

        #[async_trait]
        impl StorageProvider for CountingMockStorage {
            async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
                match self.objects.get(uri) {
                    Some(b) => Ok(ObjectMeta {
                        size: b.len() as u64,
                        etag: Some("mock-etag".into()),
                        last_modified: SystemTime::UNIX_EPOCH,
                    }),
                    None => Err(StorageError::NotFound { uri: uri.into() }),
                }
            }

            async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
                self.get_calls.fetch_add(1, Ordering::AcqRel);
                match self.objects.get(uri) {
                    Some(b) => Ok((
                        b.clone(),
                        ObjectMeta {
                            size: b.len() as u64,
                            etag: Some("mock-etag".into()),
                            last_modified: SystemTime::UNIX_EPOCH,
                        },
                    )),
                    None => Err(StorageError::NotFound { uri: uri.into() }),
                }
            }

            async fn get_range(
                &self,
                uri: &str,
                _range: Range<u64>,
            ) -> Result<Bytes, StorageError> {
                Err(permanent(uri, "get_range unimplemented for mock"))
            }

            async fn put_atomic(
                &self,
                uri: &str,
                _bytes: Bytes,
            ) -> Result<Option<String>, StorageError> {
                Err(permanent(uri, "put_atomic unimplemented for mock"))
            }

            async fn put_if_match(
                &self,
                uri: &str,
                _bytes: Bytes,
                _expected_etag: Option<&str>,
            ) -> Result<Option<String>, StorageError> {
                Err(permanent(uri, "put_if_match unimplemented for mock"))
            }

            async fn put_multipart(
                &self,
                uri: &str,
            ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
                Err(permanent(uri, "put_multipart unimplemented for mock"))
            }

            async fn delete(&self, _uri: &str) -> Result<(), StorageError> {
                Ok(())
            }
        }

        fn permanent(uri: &str, msg: &'static str) -> StorageError {
            let boxed: Box<dyn Error + Send + Sync> = msg.into();
            StorageError::Permanent {
                uri: uri.into(),
                source: boxed,
            }
        }

        fn make_test_part(seed: u8) -> ManifestPart {
            ManifestPart {
                format_version: part_mod::FORMAT_VERSION.into(),
                part_id: PartId(Uuid::from_bytes([seed; 16])),
                superfiles: vec![],
            }
        }

        fn encode_and_index(
            parts: &[ManifestPart],
        ) -> (HashMap<String, Bytes>, Vec<ManifestPartEntry>) {
            let mut objects = HashMap::new();
            let mut entries = Vec::new();
            for p in parts {
                let bytes = part_mod::encode(p, 3);
                let hash = ContentHash::of(&bytes);
                let uri = format!("manifests/part-{}.avro.zst", hash.to_hex());
                let size_compressed = bytes.len() as u64;
                objects.insert(uri.clone(), Bytes::from(bytes));
                entries.push(ManifestPartEntry {
                    part_id: p.part_id,
                    uri,
                    n_superfiles: p.superfiles.len() as u64,
                    size_bytes_compressed: size_compressed,
                    size_bytes_uncompressed: size_compressed,
                    content_hash: hash,
                    id_range: (0, 0),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                });
            }
            (objects, entries)
        }

        fn fresh_list(entries: Vec<ManifestPartEntry>) -> list::Manifest {
            list::Manifest {
                tombstone_seqs: Default::default(),
                format_version: LIST_FORMAT_VERSION.into(),
                manifest_id: 1,
                options_hash: ContentHash([0u8; 32]),
                schema: Vec::new(),
                id_column: "doc_id".into(),
                fts_columns: vec![],
                vector_columns: vec![],
                partition_strategy: PartitionStrategy::Hash {
                    column: "doc_id".into(),
                    n_buckets: 64,
                },
                parts: entries,
            }
        }

        fn options_for_test() -> Arc<SupertableOptions> {
            let s = Arc::new(Schema::new(vec![Field::new(
                "title",
                DataType::LargeUtf8,
                false,
            )]));
            Arc::new(SupertableOptions::new(s, vec![], vec![], None).expect("opts"))
        }

        fn build_manifest_with_loader(
            list: Manifest,
            storage: Arc<dyn StorageProvider>,
        ) -> ManifestSnapshot {
            let loader = Arc::new(ManifestPartLoader::new(Arc::clone(&storage), &list));
            ManifestSnapshot {
                superfile_list: SuperfileList::empty(options_for_test()),
                list: Some(list),
                parts: DashMap::new(),
                loader: Some(loader),
            }
        }

        #[tokio::test]
        async fn part_first_touch_loads_and_caches() {
            let part = make_test_part(7);
            let (objects, entries) = encode_and_index(from_ref(&part));
            let storage = Arc::new(CountingMockStorage::new(objects));
            let list = fresh_list(entries);
            let manifest =
                build_manifest_with_loader(list, Arc::clone(&storage) as Arc<dyn StorageProvider>);

            let loaded = manifest.get_part_by_id(part.part_id).await.expect("load");
            assert_eq!(loaded.part_id, part.part_id);
            assert_eq!(storage.get_call_count(), 1, "exactly one storage.get");
        }

        #[tokio::test]
        async fn second_touch_hits_cache_zero_additional_gets() {
            let part = make_test_part(11);
            let (objects, entries) = encode_and_index(from_ref(&part));
            let storage = Arc::new(CountingMockStorage::new(objects));
            let list = fresh_list(entries);
            let manifest =
                build_manifest_with_loader(list, Arc::clone(&storage) as Arc<dyn StorageProvider>);

            let a = manifest
                .get_part_by_id(part.part_id)
                .await
                .expect("first load");
            let b = manifest
                .get_part_by_id(part.part_id)
                .await
                .expect("second load");
            assert!(Arc::ptr_eq(&a, &b), "second touch must return cached Arc");
            assert_eq!(storage.get_call_count(), 1, "cache hit ⇒ no extra get");
        }

        #[tokio::test]
        async fn concurrent_loaders_coalesce_to_one_get() {
            let part = make_test_part(13);
            let (objects, entries) = encode_and_index(from_ref(&part));
            let storage = Arc::new(CountingMockStorage::new(objects));
            let list = fresh_list(entries);
            let manifest = Arc::new(build_manifest_with_loader(
                list,
                Arc::clone(&storage) as Arc<dyn StorageProvider>,
            ));

            // 100 concurrent tasks on the same cold cell.
            let mut handles = Vec::with_capacity(100);
            for _ in 0..100 {
                let m = Arc::clone(&manifest);
                let pid = part.part_id;
                handles.push(spawn(async move { m.get_part_by_id(pid).await }));
            }
            let mut first: Option<Arc<ManifestPart>> = None;
            for h in handles {
                let p = h.await.expect("join").expect("load");
                match &first {
                    None => first = Some(p),
                    Some(f) => assert!(
                        Arc::ptr_eq(f, &p),
                        "all concurrent loaders must share the same Arc"
                    ),
                }
            }
            assert_eq!(
                storage.get_call_count(),
                1,
                "100 concurrent loaders on cold cell ⇒ exactly one storage.get"
            );
        }

        #[tokio::test]
        async fn content_hash_mismatch_surfaces_typed_error_without_refetch() {
            let part = make_test_part(17);
            let (mut objects, entries) = encode_and_index(from_ref(&part));
            // Tamper with the stored bytes — content_hash on
            // the list entry no longer matches.
            let bytes = objects.values().next().expect("one obj").clone();
            let mut tampered = bytes.to_vec();
            let last = tampered.len() - 1;
            tampered[last] ^= 0xff;
            let uri = entries[0].uri.clone();
            objects.insert(uri, Bytes::from(tampered));
            let (_, fresh_entries) = encode_and_index(from_ref(&part));
            let list = fresh_list(fresh_entries);

            let storage = Arc::new(CountingMockStorage::new(objects));
            let manifest =
                build_manifest_with_loader(list, Arc::clone(&storage) as Arc<dyn StorageProvider>);

            let err = manifest
                .get_part_by_id(part.part_id)
                .await
                .expect_err("must reject tampered bytes");
            assert!(
                matches!(err, ManifestLoadError::ContentHashMismatch { .. }),
                "expected ContentHashMismatch, got {err:?}"
            );
            // Bad bytes are NOT auto-refetched. Retry returns
            // the same error. OnceCell behavior on Err futures
            // is implementation-defined (cached vs re-issued);
            // load-bearing assertion is just that retry does
            // not magically succeed.
            let _pre = storage.get_call_count();
            let err2 = manifest
                .get_part_by_id(part.part_id)
                .await
                .expect_err("must reject on retry too");
            assert!(matches!(
                err2,
                ManifestLoadError::ContentHashMismatch { .. }
            ));
        }

        #[tokio::test]
        async fn part_id_not_in_list_surfaces_typed_error() {
            let part = make_test_part(19);
            let (objects, entries) = encode_and_index(&[part]);
            let storage = Arc::new(CountingMockStorage::new(objects));
            let list = fresh_list(entries);
            let manifest =
                build_manifest_with_loader(list, Arc::clone(&storage) as Arc<dyn StorageProvider>);

            let stranger = PartId(Uuid::from_bytes([0xff; 16]));
            let err = manifest
                .get_part_by_id(stranger)
                .await
                .expect_err("must reject");
            assert!(
                matches!(err, ManifestLoadError::PartNotInList { .. }),
                "expected PartNotInList, got {err:?}"
            );
            assert_eq!(
                storage.get_call_count(),
                0,
                "missing-id check happens before any storage.get"
            );
        }

        #[tokio::test]
        async fn disk_cache_hit_serves_second_loader_without_storage_get() {
            // Two independent loaders sharing one on-disk manifest cache
            // (models a fresh manifest snapshot, or a process restart):
            // the first populates the cache from storage, the second
            // reads the part bytes off local disk with zero storage GETs.
            let part = make_test_part(23);
            let (objects, entries) = encode_and_index(from_ref(&part));
            let storage = Arc::new(CountingMockStorage::new(objects));
            let storage_dyn = Arc::clone(&storage) as Arc<dyn StorageProvider>;
            let list = fresh_list(entries);

            let cache_root = std::env::temp_dir()
                .join("infino-manifest-cache-loader-test-disk_cache_hit_second_loader");
            let _ = std::fs::remove_dir_all(&cache_root);
            let cache = ManifestDiskCache::new(cache_root.clone(), 1 << 20).expect("cache");

            // Loader A: cold — one storage GET, cache populated.
            let loader_a = ManifestPartLoader::new_with_cache(
                Arc::clone(&storage_dyn),
                &list,
                Some(Arc::clone(&cache)),
            );
            let a = loader_a.load(part.part_id).await.expect("first load");
            assert_eq!(a.part_id, part.part_id);
            assert_eq!(storage.get_call_count(), 1, "first loader fetches once");
            assert_eq!(cache.stats().n_entries, 1, "part bytes cached on disk");

            // Loader B: fresh loader, same cache — disk hit, no new GET.
            let loader_b = ManifestPartLoader::new_with_cache(
                Arc::clone(&storage_dyn),
                &list,
                Some(Arc::clone(&cache)),
            );
            let b = loader_b.load(part.part_id).await.expect("second load");
            assert_eq!(b.part_id, part.part_id);
            assert_eq!(
                storage.get_call_count(),
                1,
                "disk-cache hit ⇒ no additional storage.get"
            );
            assert!(cache.stats().n_hits >= 1, "recorded a cache hit");

            let _ = std::fs::remove_dir_all(&cache_root);
        }

        #[tokio::test]
        async fn loader_without_cache_always_hits_storage() {
            // Sanity: with no cache attached, each loader load is a
            // storage GET — confirms the cache is what removes them.
            let part = make_test_part(29);
            let (objects, entries) = encode_and_index(from_ref(&part));
            let storage = Arc::new(CountingMockStorage::new(objects));
            let storage_dyn = Arc::clone(&storage) as Arc<dyn StorageProvider>;
            let list = fresh_list(entries);

            let loader = ManifestPartLoader::new(Arc::clone(&storage_dyn), &list);
            loader.load(part.part_id).await.expect("load 1");
            loader.load(part.part_id).await.expect("load 2");
            assert_eq!(
                storage.get_call_count(),
                2,
                "no cache ⇒ every load round-trips to storage"
            );
        }

        #[tokio::test]
        async fn no_loader_attached_surfaces_typed_error() {
            // In-process-only manifest — ManifestSnapshot::empty has
            // no loader. Calling part() must error cleanly,
            // not panic.
            let manifest = ManifestSnapshot::empty(options_for_test());
            let err = manifest
                .get_part_by_id(PartId(Uuid::nil()))
                .await
                .expect_err("must error");
            assert!(
                matches!(err, ManifestLoadError::NoLoaderAttached),
                "expected NoLoaderAttached, got {err:?}"
            );
        }
    }

    // ============================================================
    // SuperfileUri path helpers, Debug formatters, and the
    // `add_sum_arrays` additive-sum helper (the scalar-stats build /
    // merge logic itself is tested on `ScalarStatsAgg` in `list.rs`).
    // ============================================================

    #[test]
    fn superfile_uri_path_helpers_share_the_same_uuid() {
        let uri = SuperfileUri(Uuid::from_u128(0x1234_5678));
        let id = uri.0;
        assert_eq!(uri.storage_path(), format!("data/seg-{id}.sf.parquet"));
        assert_eq!(uri.cache_filename(), format!("seg-{id}.sf.parquet"));
        assert_eq!(uri.cache_tmp_filename(), format!("seg-{id}.sf.parquet.tmp"));
    }

    #[test]
    fn manifest_debug_reports_counts() {
        let m = ManifestSnapshot::empty(opts()).with_appended(vec![seg_entry(Uuid::new_v4(), 3)]);
        let dbg = format!("{m:?}");
        assert!(dbg.contains("Manifest"));
        assert!(dbg.contains("manifest_id"));
        assert!(dbg.contains("n_superfiles"));
        // No storage attached ⇒ has_loader false, has_list false.
        assert!(dbg.contains("has_loader"));
    }

    #[test]
    fn manifest_debug_with_list_reports_part_count() {
        // A Manifest carrying a `list` exercises the Some-arm of the
        // `n_parts` closure in Debug (the empty-Manifest test above
        // only hits the `unwrap_or(0)` None-arm).
        use list::PartitionStrategy;
        let entry = part::PartId::new_v4();
        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 1,
            options_hash: part::ContentHash([0u8; 32]),
            schema: Vec::new(),
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![list::ManifestPartEntry {
                part_id: entry,
                uri: "manifests/part-x".into(),
                n_superfiles: 0,
                size_bytes_compressed: 0,
                size_bytes_uncompressed: 0,
                content_hash: part::ContentHash([0u8; 32]),
                id_range: (0, 0),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let m = ManifestSnapshot {
            superfile_list: SuperfileList::empty(opts()),
            list: Some(list),
            parts: DashMap::new(),
            loader: None,
        };
        let dbg = format!("{m:?}");
        assert!(dbg.contains("n_parts: 1"), "{dbg}");
        assert!(dbg.contains("has_list: true"), "{dbg}");
    }

    #[test]
    fn cluster_centroids_empty_is_empty_and_default_matches() {
        let cc = ClusterCentroids::empty();
        assert!(cc.is_empty());
        assert_eq!(cc.n_cent, 0);
        // A populated one is not empty.
        let cc = ClusterCentroids::from_fp32(2, 4, &[0.0; 8], vec![1, 1]);
        assert!(!cc.is_empty());
        assert_eq!(cc.n_cent, 2);
        assert_eq!(cc.dim, 4);
    }

    #[test]
    fn add_sum_arrays_handles_each_type_and_overflow() {
        use arrow_array::{Float64Array, Int64Array, UInt64Array};
        // Int64 + Int64.
        let r = add_sum_arrays(
            &(Arc::new(Int64Array::from(vec![3])) as ArrayRef),
            &(Arc::new(Int64Array::from(vec![4])) as ArrayRef),
        )
        .expect("int sum");
        assert_eq!(
            r.as_any()
                .downcast_ref::<Int64Array>()
                .expect("test")
                .value(0),
            7
        );
        // UInt64 + UInt64.
        let r = add_sum_arrays(
            &(Arc::new(UInt64Array::from(vec![3u64])) as ArrayRef),
            &(Arc::new(UInt64Array::from(vec![4u64])) as ArrayRef),
        )
        .expect("uint sum");
        assert_eq!(
            r.as_any()
                .downcast_ref::<UInt64Array>()
                .expect("test")
                .value(0),
            7
        );
        // Float64 + Float64.
        let r = add_sum_arrays(
            &(Arc::new(Float64Array::from(vec![1.5])) as ArrayRef),
            &(Arc::new(Float64Array::from(vec![2.5])) as ArrayRef),
        )
        .expect("float sum");
        assert!(
            (r.as_any()
                .downcast_ref::<Float64Array>()
                .expect("test")
                .value(0)
                - 4.0)
                .abs()
                < 1e-9
        );
        // Overflow → None.
        let r = add_sum_arrays(
            &(Arc::new(Int64Array::from(vec![i64::MAX])) as ArrayRef),
            &(Arc::new(Int64Array::from(vec![1])) as ArrayRef),
        );
        assert!(r.is_none(), "i64 overflow drops the stat");
        // Type mismatch → None.
        let r = add_sum_arrays(
            &(Arc::new(Int64Array::from(vec![1])) as ArrayRef),
            &(Arc::new(UInt64Array::from(vec![1u64])) as ArrayRef),
        );
        assert!(r.is_none(), "type mismatch drops the stat");
    }

    // ---- Manifest::update-------------------------------------------
    /// Base builder for a synthetic superfile entry. `pk` is the on-disk
    /// `partition_key`: pass a stamped key to model a prior-commit entry (as
    /// stored in an existing part), or `Vec::new()` for an unstamped entry
    /// destined for `update()`, which derives and stamps the key itself.
    fn make_entry(docs: u64, pk: Vec<u8>, hint: Option<u32>) -> Arc<SuperfileEntry> {
        Arc::new(SuperfileEntry {
            superfile_id: uuid::Uuid::new_v4(),
            uri: SuperfileUri::new_v4(),
            n_docs: docs,
            id_min: 0,
            id_max: docs as i128 - 1,
            scalar_stats: Default::default(),
            fts_summary: Default::default(),
            vector_summary: Default::default(),
            partition_key: pk,
            partition_hint: hint,
            subsection_offsets: None,
        })
    }

    /// Stamped prior-commit entry (non-empty `partition_key`), for placing
    /// into existing on-disk parts.
    fn make_superfile_entry(docs: u64, pk: Vec<u8>) -> Arc<SuperfileEntry> {
        make_entry(docs, pk, None)
    }

    /// Builds an UNSTAMPED entry (empty partition_key) for passing to
    /// `update()`, which derives and stamps the key. Entries that model a
    /// prior commit (placed into existing parts) keep `make_superfile_entry*`,
    /// which carries the already-stamped key.
    fn make_new_entry(docs: u64) -> Arc<SuperfileEntry> {
        make_entry(docs, Vec::new(), None)
    }

    fn hash_bucket_0_pk() -> Vec<u8> {
        // Hash partition with n_buckets=1 encodes to [0, 0, 0, 0] in little-endian
        vec![0, 0, 0, 0]
    }

    fn simple_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "text",
            DataType::LargeUtf8,
            false,
        )]))
    }

    fn make_opts() -> Arc<SupertableOptions> {
        SupertableOptions::new(simple_schema(), vec![], vec![], None)
            .map(Arc::new)
            .expect("valid options")
    }

    fn empty_manifest(opts: &Arc<SupertableOptions>) -> Arc<ManifestSnapshot> {
        Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList::empty(opts.clone()),
            list: Some(Manifest {
                tombstone_seqs: Default::default(),
                format_version: list::FORMAT_VERSION.into(),
                manifest_id: 0,
                options_hash: ContentHash([0u8; 32]),
                schema: vec![],
                id_column: "_id".into(),
                fts_columns: vec![],
                vector_columns: vec![],
                partition_strategy: PartitionStrategy::Hash {
                    column: "_id".into(),
                    n_buckets: 1,
                },
                parts: vec![],
            }),
            parts: DashMap::new(),
            loader: None,
        })
    }

    #[tokio::test]
    async fn update_fresh_start_cold_partition_should_create_entry() {
        let opts = make_opts();
        let old_manifest = empty_manifest(&opts);

        let new_entry = make_new_entry(100);
        let new_entries = vec![new_entry];

        let (new_manifest, parts) = old_manifest
            .update(&new_entries, &[])
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(list_entries[0].n_superfiles, 1);
        assert_eq!(parts[0].part.superfiles.len(), 1);
        assert_eq!(parts[0].part.superfiles[0].n_docs, 100);
    }

    #[tokio::test]
    async fn update_fresh_start_multiple_cold_partitions_should_create_entries() {
        // Multiple new entries in a single commit all land in one table-level
        // part (well under the default target).
        let opts = make_opts();
        let old_manifest = empty_manifest(&opts);

        let entry1 = make_new_entry(100);
        let entry2 = make_new_entry(200);
        let new_entries = vec![entry1, entry2];

        let (new_manifest, parts) = old_manifest
            .update(&new_entries, &[])
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(list_entries[0].n_superfiles, 2);
        assert_eq!(parts[0].part.superfiles.len(), 2);
        let total_docs: u64 = parts[0].part.superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 300);
    }

    fn local_storage() -> (TempDir, Arc<dyn StorageProvider>) {
        let dir = TempDir::new().expect("tempdir");
        let store: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local"));
        (dir, store)
    }

    #[tokio::test]
    async fn update_add_to_existing_partition_rewrites_part() {
        // Adding a new entry to the single existing part rewrites it in place.
        let opts = make_opts();
        let pk_untouched = hash_bucket_0_pk();

        let (_dir, storage) = local_storage();

        let old_superfile = make_superfile_entry(100, pk_untouched.clone());
        let existing_part = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![old_superfile.clone()],
        };
        let pw = write_manifest_part(storage.as_ref(), &existing_part, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part");

        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![ManifestPartEntry {
                part_id: pw.part_id,
                uri: pw.uri,
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: 1,
                id_range: (0, 99),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let loader = ManifestPartLoader::new(storage, &list);

        let parts = DashMap::new();
        parts.insert(
            pw.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(existing_part)))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![old_superfile],
            },
            list: Some(list),
            parts,
            loader: Some(Arc::new(loader)),
        });

        // Add new entry to the SAME partition (not a new/cold partition)
        let new_entry = make_new_entry(50);
        let new_entries = vec![new_entry];

        let (new_manifest, parts) = old_manifest
            .update(&new_entries, &[])
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        // Should have 1 list entry (rewritten old one)
        assert_eq!(list_entries.len(), 1);
        // Should have 1 new part (the rewritten one)
        assert_eq!(parts.len(), 1);

        assert_eq!(list_entries[0].n_superfiles, 2);

        // Part should have combined superfiles
        assert_eq!(parts[0].part.superfiles.len(), 2);
        let total_docs: u64 = parts[0].part.superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 150);
    }

    #[tokio::test]
    async fn update_leaves_unchanged_parts_untouched() {
        // Start with three parts, two superfiles each, forming one table-level
        // lineage in list order: [part_0, part_1, part_2]. New entries append to
        // the LAST part (the latest), which here has room for one more superfile
        // (target = 3, so 2 + 1 = 3 stays within target → rewrite, no split). We
        // commit a single new superfile. After update ONLY the last part
        // changes; the two earlier parts must carry over byte-for-byte — same
        // part_id, uri, and content_hash — and must NOT be re-emitted into
        // `parts_to_write` (no re-encode, no PUT).
        const SUPERFILES_PER_PART: u64 = 2;
        const TARGET_SUPERFILES_PER_PART: u64 = 3;

        let pk_a = hash2_pk(0);
        let pk_b = hash2_pk(1);
        let (_dir, storage) = local_storage();

        // Attach storage so the manifests `update` derives also carry
        // a loader — the second (removal) phase loads carried-over parts
        // (A_old, B) back from storage.
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_part = TARGET_SUPERFILES_PER_PART;
        let opts = Arc::new(base_opts.with_storage(storage.clone()));

        // Helper: build a 2-superfile part for a partition and persist it.
        async fn two_superfile_part(
            storage: &dyn StorageProvider,
            pk: &[u8],
            hint: u32,
            docs: [u64; 2],
        ) -> (ManifestPart, PartWriteResult) {
            let part = ManifestPart {
                format_version: part::FORMAT_VERSION.into(),
                part_id: PartId::new_v4(),
                superfiles: vec![
                    make_superfile_entry_hinted(docs[0], pk.to_vec(), hint),
                    make_superfile_entry_hinted(docs[1], pk.to_vec(), hint),
                ],
            };
            let pw = write_manifest_part(storage, &part, MANIFEST_ZSTD_LEVEL)
                .await
                .expect("write part");
            (part, pw)
        }

        let (part_a_old, pw_a_old) =
            two_superfile_part(storage.as_ref(), &pk_a, 0, [100, 110]).await;
        let (_part_a_latest, pw_a_latest) =
            two_superfile_part(storage.as_ref(), &pk_a, 0, [120, 130]).await;
        let (part_b, pw_b) = two_superfile_part(storage.as_ref(), &pk_b, 1, [200, 210]).await;

        // Build a list entry mirroring a persisted part.
        let entry_for = |pw: &PartWriteResult| -> ManifestPartEntry {
            ManifestPartEntry {
                part_id: pw.part_id,
                uri: pw.uri.clone(),
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: SUPERFILES_PER_PART,
                id_range: (0, 0),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }
        };

        // List order: [part_0, part_1, part_2]. The last part is the latest
        // (the rewrite candidate); the two earlier parts carry over unchanged.
        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 2,
            },
            parts: vec![
                entry_for(&pw_a_old),
                entry_for(&pw_a_latest),
                entry_for(&pw_b),
            ],
        };
        let loader = ManifestPartLoader::new(storage, &list);

        // Only the latest (last) part is needed in-cache for the rewrite to
        // load + combine; the loader serves the rest from storage.
        let parts_map = DashMap::new();
        parts_map.insert(
            part_b.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_b.clone())))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: part_a_old
                    .superfiles
                    .iter()
                    .chain(part_b.superfiles.iter())
                    .cloned()
                    .collect(),
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        // Commit one new superfile. Keep `new_entry` around — the second
        // phase below removes it again.
        let new_entry = make_new_entry_hinted(140, 0);
        let (new_manifest, parts_to_write) = old_manifest
            .update(from_ref(&new_entry), &[])
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        // Three list entries remain (part_0 carried over, part_1 carried over,
        // part_2 rewritten in place), and only ONE part is re-emitted for
        // writing — the rewritten last part.
        assert_eq!(list_entries.len(), 3, "list entry count");
        assert_eq!(
            parts_to_write.len(),
            1,
            "only the rewritten last part should be re-emitted; \
             unchanged parts must not be re-encoded/PUT",
        );

        // Locate the carried-over entries by their original part_id and
        // assert they are byte-for-byte identical to what was persisted.
        let find = |part_id: PartId| {
            list_entries
                .iter()
                .find(|e| e.part_id == part_id)
                .unwrap_or_else(|| panic!("entry for part {part_id:?} missing after update"))
        };

        // part_0 carries over unchanged.
        let part0_after = find(pw_a_old.part_id);
        assert_eq!(part0_after.uri, pw_a_old.uri, "carried-over part_0 uri");
        assert_eq!(
            part0_after.content_hash, pw_a_old.content_hash,
            "carried-over part_0 content_hash",
        );
        assert_eq!(part0_after.n_superfiles, SUPERFILES_PER_PART);

        // part_1 carries over unchanged too.
        let part1_after = find(pw_a_latest.part_id);
        assert_eq!(part1_after.uri, pw_a_latest.uri, "carried-over part_1 uri");
        assert_eq!(
            part1_after.content_hash, pw_a_latest.content_hash,
            "carried-over part_1 content_hash",
        );
        assert_eq!(part1_after.n_superfiles, SUPERFILES_PER_PART);

        // The one re-emitted part is the rewritten last part: it now holds the
        // original two superfiles plus the new one.
        assert_eq!(
            parts_to_write[0].part.superfiles.len(),
            (SUPERFILES_PER_PART + 1) as usize,
            "rewritten last part should hold its 2 superfiles + the new one",
        );
        // And the original last part_id is gone from the list (it was
        // rewritten, not carried over).
        assert!(
            !list_entries.iter().any(|e| e.part_id == pw_b.part_id),
            "the rewritten last part is replaced, so its old part_id must not survive",
        );
        // The rewritten (last) entry is the one that is neither of the two
        // carried-over parts — it holds the combined superfiles.
        let rewritten_after = list_entries
            .iter()
            .find(|e| e.part_id != pw_a_old.part_id && e.part_id != pw_a_latest.part_id)
            .expect("rewritten last entry present after the add");
        assert_eq!(rewritten_after.n_superfiles, SUPERFILES_PER_PART + 1);

        // ---- Second phase: remove the superfile we just added --------
        //
        // The new superfile lives in the rewritten last part. Remove it. Only
        // that part should change. The two earlier parts never held the removed
        // superfile — both must carry over byte-for-byte.
        //
        // Capture the rewritten last part's identity (the part the removal will
        // legitimately rebuild).
        let rewritten_v1_part_id = rewritten_after.part_id;

        let (after_removal, removal_parts) = new_manifest
            .update(&[], from_ref(&new_entry))
            .await
            .expect("update removal");
        let entries_after = after_removal.get_all_list_entries();

        assert_eq!(entries_after.len(), 3, "list entry count after removal");

        // The part we removed from MUST change: its v1 part_id is gone.
        assert!(
            !entries_after
                .iter()
                .any(|e| e.part_id == rewritten_v1_part_id),
            "the part we removed a superfile from must be rebuilt (new part_id)",
        );

        // part_0 held none of the removed superfile — same part identity.
        let part0_after_removal = entries_after
            .iter()
            .find(|e| e.part_id == pw_a_old.part_id)
            .expect("part_0 must survive the removal unchanged");
        assert_eq!(
            part0_after_removal.uri, pw_a_old.uri,
            "part_0 uri after removal",
        );
        assert_eq!(
            part0_after_removal.content_hash, pw_a_old.content_hash,
            "part_0 content_hash after removal",
        );

        // part_1 held none of the removed superfile — same part identity.
        let part1_after_removal = entries_after
            .iter()
            .find(|e| e.part_id == pw_a_latest.part_id)
            .expect("part_1 must survive the removal unchanged");
        assert_eq!(
            part1_after_removal.uri, pw_a_latest.uri,
            "part_1 uri after removal",
        );
        assert_eq!(
            part1_after_removal.content_hash, pw_a_latest.content_hash,
            "part_1 content_hash after removal",
        );

        // Only the part that actually lost a superfile should be
        // re-emitted for writing.
        assert_eq!(
            removal_parts.len(),
            1,
            "only the part we removed from should be rewritten; unchanged parts \
             must not be re-encoded/PUT",
        );
    }

    #[tokio::test]
    async fn update_rewrite_partition_within_target() {
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_part = 3;
        let opts = Arc::new(base_opts);

        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        let sf1 = make_superfile_entry(100, pk.clone());
        let sf2 = make_superfile_entry(150, pk.clone());

        let existing_part = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf1.clone(), sf2.clone()],
        };
        let pw = write_manifest_part(storage.as_ref(), &existing_part, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part");

        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![ManifestPartEntry {
                part_id: pw.part_id,
                uri: pw.uri,
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: 2,
                id_range: (0, 149),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let loader = ManifestPartLoader::new(storage, &list);

        let parts = DashMap::new();
        parts.insert(
            pw.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(existing_part)))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf1, sf2],
            },
            list: Some(list),
            parts,
            loader: Some(Arc::new(loader)),
        });

        // Add 1 new superfile to same partition (2 + 1 = 3, within target)
        let new_entry = make_new_entry(75);
        let new_entries = vec![new_entry];

        let (new_manifest, parts) = old_manifest
            .update(&new_entries, &[])
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        // Rewrite case: 1 list entry (old entry replaced), 1 new part
        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts.len(), 1);

        assert_eq!(list_entries[0].n_superfiles, 3);

        // Part should have all 3 superfiles combined
        let part = &parts[0];
        assert_eq!(part.part.superfiles.len(), 3);
        // Verify combined doc count
        let total_docs: u64 = part.part.superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 325); // 100 + 150 + 75
    }

    #[tokio::test]
    async fn update_split_partition_exceeds_target() {
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_part = 2;
        let opts = Arc::new(base_opts);

        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        let sf1 = make_superfile_entry(100, pk.clone());
        let sf2 = make_superfile_entry(150, pk.clone());

        let existing_part = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf1.clone(), sf2.clone()],
        };
        let pw = write_manifest_part(storage.as_ref(), &existing_part, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part");

        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![ManifestPartEntry {
                part_id: pw.part_id,
                uri: pw.uri,
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: 2,
                id_range: (0, 149),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let loader = ManifestPartLoader::new(storage, &list);

        let parts = DashMap::new();
        parts.insert(
            pw.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(existing_part)))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf1, sf2],
            },
            list: Some(list),
            parts,
            loader: Some(Arc::new(loader)),
        });

        // Add 2 new superfiles to same partition (2 + 2 = 4, exceeds target of 2)
        let new_entry1 = make_new_entry(75);
        let new_entry2 = make_new_entry(80);
        let new_entries = vec![new_entry1, new_entry2];

        let (new_manifest, parts) = old_manifest
            .update(&new_entries, &[])
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        // Split case: 2 list entries (old + fresh for split), 1 new part (fresh)
        assert_eq!(list_entries.len(), 2);
        assert_eq!(parts.len(), 1);

        // First entry is the carried-over existing part; the second is the
        // fresh split part holding the new superfiles.

        // First entry (old) should still have original superfiles
        assert_eq!(list_entries[0].n_superfiles, 2);

        // Second entry (fresh) should have the new superfiles
        assert_eq!(list_entries[1].n_superfiles, 2);

        // The one new part should have exactly the 2 new superfiles
        let part = &parts[0];
        assert_eq!(part.part.superfiles.len(), 2);
        let total_docs: u64 = part.part.superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 155); // 75 + 80
    }

    /// Hinted counterpart to [`make_superfile_entry`] — stamped, with a
    /// routing `partition_hint`.
    fn make_superfile_entry_hinted(docs: u64, pk: Vec<u8>, hint: u32) -> Arc<SuperfileEntry> {
        make_entry(docs, pk, Some(hint))
    }

    /// Hinted counterpart to [`make_new_entry`] — UNSTAMPED (empty
    /// partition_key), carrying a routing `partition_hint` for `update()` to
    /// derive the key from.
    fn make_new_entry_hinted(docs: u64, hint: u32) -> Arc<SuperfileEntry> {
        make_entry(docs, Vec::new(), Some(hint))
    }

    fn hash2_pk(bucket: u32) -> Vec<u8> {
        bucket.to_le_bytes().to_vec()
    }

    #[tokio::test]
    async fn update_older_entry_preserved_when_latest_rewritten() {
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_part = 2;
        let opts = Arc::new(base_opts);

        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        let sf_old = make_superfile_entry(100, pk.clone());
        let sf_latest = make_superfile_entry(150, pk.clone());

        let part_old = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_old.clone()],
        };
        let pw_old = write_manifest_part(storage.as_ref(), &part_old, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_old");

        let part_latest = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_latest.clone()],
        };
        let pw_latest = write_manifest_part(storage.as_ref(), &part_latest, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_latest");

        // Old manifest with TWO entries for same partition (result of prior split)
        // Second one is the "latest" for that partition
        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![
                ManifestPartEntry {
                    part_id: pw_old.part_id,
                    uri: pw_old.uri.clone(),
                    content_hash: pw_old.content_hash,
                    size_bytes_compressed: pw_old.size_bytes_compressed,
                    size_bytes_uncompressed: pw_old.size_bytes_uncompressed,
                    n_superfiles: 1,
                    id_range: (0, 99),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestPartEntry {
                    part_id: pw_latest.part_id,
                    uri: pw_latest.uri,
                    content_hash: pw_latest.content_hash,
                    size_bytes_compressed: pw_latest.size_bytes_compressed,
                    size_bytes_uncompressed: pw_latest.size_bytes_uncompressed,
                    n_superfiles: 1,
                    id_range: (0, 149),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
            ],
        };
        let loader = ManifestPartLoader::new(storage, &list);

        let parts = DashMap::new();
        parts.insert(
            part_latest.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_latest)))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf_old, sf_latest],
            },
            list: Some(list),
            parts,
            loader: Some(Arc::new(loader)),
        });

        // Add one new entry for the partition
        let new_entries = vec![make_new_entry(75)];

        let (new_manifest, parts) = old_manifest
            .update(&new_entries, &[])
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        // Expect: old entry (preserved) + latest entry (rewritten) = 2 list entries
        // Expect: 1 new part (latest rewrite)
        assert_eq!(list_entries.len(), 2);
        assert_eq!(parts.len(), 1);

        // First entry should carry over the old one unchanged
        assert_eq!(list_entries[0].n_superfiles, 1);
        // URI should be exactly the same as the original written part
        assert_eq!(list_entries[0].uri, pw_old.uri);

        // Second entry should be the rewritten latest (1 + 1 = 2 superfiles)
        assert_eq!(list_entries[1].n_superfiles, 2);

        // New part should have the combined latest + new
        assert_eq!(parts[0].part.superfiles.len(), 2);
        let total_docs: u64 = parts[0].part.superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 225); // 150 + 75
    }

    // ---- cross-partition tests --------------------------------------------

    #[tokio::test]
    async fn update_two_partitions_both_touched() {
        // Two existing parts hold superfiles tagged with different partition
        // hints. New entries carrying both hints are added in one commit. Parts
        // are a single table-level lineage now, so the new entries all append to
        // the LAST existing part (the latest); the earlier part carries over
        // unchanged. Each superfile keeps its own partition_hint/partition_key.
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_part = 3;
        let opts = Arc::new(base_opts);

        let pk_a = hash2_pk(0);
        let pk_b = hash2_pk(1);
        let (_dir, storage) = local_storage();

        let sf_a = make_superfile_entry_hinted(100, pk_a.clone(), 0);
        let part_a = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a.clone()],
        };
        let pw_a = write_manifest_part(storage.as_ref(), &part_a, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_a");

        let sf_b = make_superfile_entry_hinted(200, pk_b.clone(), 1);
        let part_b = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_b.clone()],
        };
        let pw_b = write_manifest_part(storage.as_ref(), &part_b, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_b");

        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 2,
            },
            parts: vec![
                ManifestPartEntry {
                    part_id: pw_a.part_id,
                    uri: pw_a.uri,
                    content_hash: pw_a.content_hash,
                    size_bytes_compressed: pw_a.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a.size_bytes_uncompressed,
                    n_superfiles: 1,
                    id_range: (0, 99),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestPartEntry {
                    part_id: pw_b.part_id,
                    uri: pw_b.uri,
                    content_hash: pw_b.content_hash,
                    size_bytes_compressed: pw_b.size_bytes_compressed,
                    size_bytes_uncompressed: pw_b.size_bytes_uncompressed,
                    n_superfiles: 1,
                    id_range: (0, 199),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
            ],
        };
        let loader = ManifestPartLoader::new(storage, &list);
        let parts_map = DashMap::new();
        parts_map.insert(
            part_a.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a)))),
        );
        parts_map.insert(
            part_b.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_b)))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf_a, sf_b],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let new_entries = vec![make_new_entry_hinted(50, 0), make_new_entry_hinted(80, 1)];

        let (new_manifest, parts) = old_manifest
            .update(&new_entries, &[])
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        // Single lineage: the earlier part carries over, the latest part is
        // rewritten with both new entries appended. 2 list entries, 1 new part.
        assert_eq!(list_entries.len(), 2);
        assert_eq!(parts.len(), 1);

        // [0] The earlier part carries over unchanged — original part_id
        // preserved, still its single original superfile.
        assert_eq!(list_entries[0].part_id, pw_a.part_id);
        assert_eq!(list_entries[0].n_superfiles, 1);
        assert_eq!(list_entries[0].content_hash, pw_a.content_hash);

        // [1] The latest part is rewritten: its 1 existing superfile + both new
        // ones = 3 superfiles.
        assert_eq!(list_entries[1].n_superfiles, 3);
        assert_eq!(parts[0].part.superfiles.len(), 3);
        let total_docs: u64 = parts[0].part.superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 330); // 200 (existing B) + 50 + 80

        // Each new superfile kept its own partition_hint (the routing tag is
        // independent of the part's table-level grouping key).
        let hints: Vec<_> = parts[0]
            .part
            .superfiles
            .iter()
            .map(|s| s.partition_hint)
            .collect();
        assert!(hints.contains(&Some(0)), "hint-0 new entry preserved");
        assert!(hints.contains(&Some(1)), "hint-1 new entry preserved");
    }

    #[tokio::test]
    async fn update_two_partitions_one_touched_exact_carry_over() {
        // One new entry is committed. Parts are a single table-level lineage, so
        // it appends to the LAST existing part (the latest), which is rewritten;
        // the earlier part carries over with the exact URI and content_hash that
        // were written — no re-encode, no PUT.
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_part = 3;
        let opts = Arc::new(base_opts);

        let pk_a = hash2_pk(0);
        let pk_b = hash2_pk(1);
        let (_dir, storage) = local_storage();

        let sf_a = make_superfile_entry_hinted(100, pk_a.clone(), 0);
        let part_a = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a.clone()],
        };
        let pw_a = write_manifest_part(storage.as_ref(), &part_a, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_a");

        let sf_b = make_superfile_entry_hinted(200, pk_b.clone(), 1);
        let part_b = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_b.clone()],
        };
        let pw_b = write_manifest_part(storage.as_ref(), &part_b, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_b");

        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 2,
            },
            parts: vec![
                ManifestPartEntry {
                    part_id: pw_a.part_id,
                    uri: pw_a.uri,
                    content_hash: pw_a.content_hash,
                    size_bytes_compressed: pw_a.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a.size_bytes_uncompressed,
                    n_superfiles: 1,
                    id_range: (0, 99),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestPartEntry {
                    part_id: pw_b.part_id,
                    uri: pw_b.uri.clone(),
                    content_hash: pw_b.content_hash,
                    size_bytes_compressed: pw_b.size_bytes_compressed,
                    size_bytes_uncompressed: pw_b.size_bytes_uncompressed,
                    n_superfiles: 1,
                    id_range: (0, 199),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
            ],
        };
        let loader = ManifestPartLoader::new(storage, &list);
        let parts_map = DashMap::new();
        parts_map.insert(
            part_a.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a)))),
        );
        parts_map.insert(
            part_b.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_b)))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf_a, sf_b],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        // Only touch partition A
        let new_entries = vec![make_new_entry_hinted(50, 0)];

        let (new_manifest, parts) = old_manifest
            .update(&new_entries, &[])
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        // 2 list entries (earlier part carried over, latest rewritten), 1 new part
        assert_eq!(list_entries.len(), 2);
        assert_eq!(parts.len(), 1);

        // [0] Earlier part: exact carry-over — part_id and content_hash unchanged.
        assert_eq!(list_entries[0].part_id, pw_a.part_id);
        assert_eq!(list_entries[0].n_superfiles, 1);
        assert_eq!(list_entries[0].content_hash, pw_a.content_hash);

        // [1] Latest part: rewritten with its existing superfile + the new one =
        // 2 superfiles, 250 docs.
        assert_eq!(list_entries[1].n_superfiles, 2);
        assert_eq!(parts[0].part.superfiles.len(), 2);
        let docs: u64 = parts[0].part.superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(docs, 250); // 200 (existing B) + 50 (new)
    }

    #[tokio::test]
    async fn update_two_partitions_each_with_prior_split() {
        // Four existing parts from prior splits, in one table-level lineage:
        // [p0, p1, p2, p3]. Two new entries (different partition hints) are
        // committed. They append to the LAST part (p3, the latest); p3 already
        // holds 1 superfile so 1 + 2 = 3 exceeds the target of 2 — a split. The
        // split keeps p3 as-is and emits a fresh part holding just the 2 new
        // entries. p0..p2 carry over unchanged. Each superfile keeps its own
        // partition_hint.
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_part = 2;
        let opts = Arc::new(base_opts);

        let pk_a = hash2_pk(0);
        let pk_b = hash2_pk(1);
        let (_dir, storage) = local_storage();

        // Partition A: two parts
        let sf_a_old = make_superfile_entry_hinted(100, pk_a.clone(), 0);
        let part_a_old = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a_old.clone()],
        };
        let pw_a_old = write_manifest_part(storage.as_ref(), &part_a_old, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_a_old");

        let sf_a_latest = make_superfile_entry_hinted(150, pk_a.clone(), 0);
        let part_a_latest = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a_latest.clone()],
        };
        let pw_a_latest =
            write_manifest_part(storage.as_ref(), &part_a_latest, MANIFEST_ZSTD_LEVEL)
                .await
                .expect("write part_a_latest");

        // Partition B: two parts
        let sf_b_old = make_superfile_entry_hinted(200, pk_b.clone(), 1);
        let part_b_old = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_b_old.clone()],
        };
        let pw_b_old = write_manifest_part(storage.as_ref(), &part_b_old, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_b_old");

        let sf_b_latest = make_superfile_entry_hinted(250, pk_b.clone(), 1);
        let part_b_latest = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_b_latest.clone()],
        };
        let pw_b_latest =
            write_manifest_part(storage.as_ref(), &part_b_latest, MANIFEST_ZSTD_LEVEL)
                .await
                .expect("write part_b_latest");

        // List order: [a_old, a_latest, b_old, b_latest]
        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 2,
            },
            parts: vec![
                ManifestPartEntry {
                    part_id: pw_a_old.part_id,
                    uri: pw_a_old.uri.clone(),
                    content_hash: pw_a_old.content_hash,
                    size_bytes_compressed: pw_a_old.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a_old.size_bytes_uncompressed,
                    n_superfiles: 1,
                    id_range: (0, 99),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestPartEntry {
                    part_id: pw_a_latest.part_id,
                    uri: pw_a_latest.uri,
                    content_hash: pw_a_latest.content_hash,
                    size_bytes_compressed: pw_a_latest.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a_latest.size_bytes_uncompressed,
                    n_superfiles: 1,
                    id_range: (0, 149),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestPartEntry {
                    part_id: pw_b_old.part_id,
                    uri: pw_b_old.uri.clone(),
                    content_hash: pw_b_old.content_hash,
                    size_bytes_compressed: pw_b_old.size_bytes_compressed,
                    size_bytes_uncompressed: pw_b_old.size_bytes_uncompressed,
                    n_superfiles: 1,
                    id_range: (0, 199),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestPartEntry {
                    part_id: pw_b_latest.part_id,
                    uri: pw_b_latest.uri,
                    content_hash: pw_b_latest.content_hash,
                    size_bytes_compressed: pw_b_latest.size_bytes_compressed,
                    size_bytes_uncompressed: pw_b_latest.size_bytes_uncompressed,
                    n_superfiles: 1,
                    id_range: (0, 249),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
            ],
        };
        let loader = ManifestPartLoader::new(storage, &list);
        let parts_map = DashMap::new();
        parts_map.insert(
            part_a_latest.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a_latest)))),
        );
        parts_map.insert(
            part_b_latest.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_b_latest)))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf_a_old, sf_a_latest, sf_b_old, sf_b_latest],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let new_entries = vec![make_new_entry_hinted(75, 0), make_new_entry_hinted(90, 1)];

        let (new_manifest, parts) = old_manifest
            .update(&new_entries, &[])
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        // 5 list entries: the 4 existing parts all carry over, plus 1 fresh
        // split part holding the 2 new entries.
        assert_eq!(list_entries.len(), 5);
        // 1 new part: the fresh split.
        assert_eq!(parts.len(), 1);

        // [0..=3] The four existing parts carry over unchanged — original
        // part_ids preserved, one superfile each.
        assert_eq!(list_entries[0].part_id, pw_a_old.part_id);
        assert_eq!(list_entries[0].uri, pw_a_old.uri);
        assert_eq!(list_entries[0].content_hash, pw_a_old.content_hash);

        assert_eq!(list_entries[1].part_id, pw_a_latest.part_id);

        assert_eq!(list_entries[2].part_id, pw_b_old.part_id);
        assert_eq!(list_entries[2].uri, pw_b_old.uri);
        assert_eq!(list_entries[2].content_hash, pw_b_old.content_hash);

        assert_eq!(list_entries[3].part_id, pw_b_latest.part_id);

        for e in &list_entries[0..4] {
            assert_eq!(e.n_superfiles, 1);
        }

        // [4] Fresh split part: the 2 new entries. 165 docs (75 + 90).
        assert_eq!(list_entries[4].n_superfiles, 2);
        assert_eq!(parts[0].part.superfiles.len(), 2);
        let docs: u64 = parts[0].part.superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(docs, 165); // 75 (hint 0) + 90 (hint 1)

        // The new superfiles kept their own partition hints.
        let hints: Vec<_> = parts[0]
            .part
            .superfiles
            .iter()
            .map(|s| s.partition_hint)
            .collect();
        assert!(hints.contains(&Some(0)), "hint-0 new entry preserved");
        assert!(hints.contains(&Some(1)), "hint-1 new entry preserved");
    }

    #[tokio::test]
    async fn update_multiple_partitions_land_in_one_lineage() {
        // A single commit of several new superfiles carrying DIFFERENT
        // partition hints (well under the target) produces ONE table-level part
        // holding ALL of them. Each superfile keeps its own partition_hint —
        // the routing tag is independent of the part's table-level grouping.
        // Under the default single-bucket Hash strategy the commit-time
        // partition_key stamped on every entry is bucket 0, regardless of hint.
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_part = 8;
        let opts = Arc::new(base_opts);

        let old_manifest = empty_manifest(&opts);

        let hints = [0u32, 1, 2, 3];
        let new_entries: Vec<_> = hints
            .iter()
            .enumerate()
            .map(|(i, &h)| make_new_entry_hinted(100 + i as u64, h))
            .collect();

        let (new_manifest, parts) = old_manifest
            .update(&new_entries, &[])
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        // One part holding all four.
        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(list_entries[0].n_superfiles, hints.len() as u64);
        assert_eq!(parts[0].part.superfiles.len(), hints.len());

        // Every original superfile is present, each kept its own
        // partition_hint, and each carries the commit-time partition_key
        // stamped by `update` — bucket 0 under the single-bucket Hash default.
        for (&h, expected) in hints.iter().zip(new_entries.iter()) {
            let landed = parts[0]
                .part
                .superfiles
                .iter()
                .find(|s| s.superfile_id == expected.superfile_id)
                .unwrap_or_else(|| panic!("superfile with hint {h} landed in the part"));
            assert_eq!(landed.partition_hint, Some(h), "partition_hint preserved");
            assert_eq!(
                landed.partition_key,
                hash2_pk(0),
                "single-bucket Hash stamps bucket 0 on every entry"
            );
        }
    }

    // ---- removal tests ---------------------------------------------------

    #[tokio::test]
    async fn update_remove_one_superfile_from_partition() {
        // Partition has 2 superfiles; remove one. Verifies the part is
        // rewritten containing only the superfile that was not removed.
        let opts = make_opts();
        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        let sf_keep = make_superfile_entry(100, pk.clone());
        let sf_remove = make_superfile_entry(150, pk.clone());

        let existing_part = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_keep.clone(), sf_remove.clone()],
        };
        let pw = write_manifest_part(storage.as_ref(), &existing_part, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part");

        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![ManifestPartEntry {
                part_id: pw.part_id,
                uri: pw.uri,
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: 2,
                id_range: (0, 149),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let loader = ManifestPartLoader::new(storage, &list);
        let parts_map = DashMap::new();
        parts_map.insert(
            existing_part.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(existing_part)))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf_keep.clone(), sf_remove.clone()],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let (new_manifest, parts) = old_manifest
            .update(&[], from_ref(&sf_remove))
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        // Part rewritten with 1 superfile; no cold entries.
        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(list_entries[0].n_superfiles, 1);
        assert_eq!(parts[0].part.superfiles.len(), 1);
        assert_eq!(
            parts[0].part.superfiles[0].superfile_id,
            sf_keep.superfile_id
        );
        let total_docs: u64 = parts[0].part.superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 100);
    }

    #[tokio::test]
    async fn update_add_and_remove_in_same_partition() {
        // One new superfile is added while one existing superfile is removed
        // in the same partition. The resulting part should contain the
        // surviving existing superfile plus the new one — not the removed one.
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_part = 3;
        let opts = Arc::new(base_opts);

        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        let sf_keep = make_superfile_entry(100, pk.clone());
        let sf_remove = make_superfile_entry(150, pk.clone());

        let existing_part = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_keep.clone(), sf_remove.clone()],
        };
        let pw = write_manifest_part(storage.as_ref(), &existing_part, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part");

        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![ManifestPartEntry {
                part_id: pw.part_id,
                uri: pw.uri,
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: 2,
                id_range: (0, 149),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let loader = ManifestPartLoader::new(storage, &list);
        let parts_map = DashMap::new();
        parts_map.insert(
            existing_part.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(existing_part)))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf_keep.clone(), sf_remove.clone()],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let sf_new = make_new_entry(75);
        let new_entries = vec![sf_new.clone()];

        let (new_manifest, parts) = old_manifest
            .update(&new_entries, from_ref(&sf_remove))
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        // Net result: 1 list entry, 1 part — sf_keep + sf_new, sf_remove absent
        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(list_entries[0].n_superfiles, 2);
        assert_eq!(parts[0].part.superfiles.len(), 2);

        let ids: Vec<_> = parts[0]
            .part
            .superfiles
            .iter()
            .map(|s| s.superfile_id)
            .collect();
        assert!(ids.contains(&sf_keep.superfile_id));
        assert!(ids.contains(&sf_new.superfile_id));
        assert!(!ids.contains(&sf_remove.superfile_id));

        let total_docs: u64 = parts[0].part.superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 175); // 100 + 75
    }

    #[tokio::test]
    async fn update_remove_from_one_partition_other_carried_over_exactly() {
        // Two parts. The removed superfile lives only in the first part. Removals
        // are checked against every part: the first part matches and is
        // rewritten; the second holds no matching id and carries over with the
        // exact URI and content_hash — no re-encode, no PUT.
        let opts = make_opts();
        let pk_a = hash2_pk(0);
        let pk_b = hash2_pk(1);
        let (_dir, storage) = local_storage();

        let sf_a_keep = make_superfile_entry_hinted(100, pk_a.clone(), 0);
        let sf_a_remove = make_superfile_entry_hinted(150, pk_a.clone(), 0);
        let part_a = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a_keep.clone(), sf_a_remove.clone()],
        };
        let pw_a = write_manifest_part(storage.as_ref(), &part_a, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_a");

        let sf_b = make_superfile_entry_hinted(200, pk_b.clone(), 1);
        let part_b = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_b.clone()],
        };
        let pw_b = write_manifest_part(storage.as_ref(), &part_b, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_b");

        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 2,
            },
            parts: vec![
                ManifestPartEntry {
                    part_id: pw_a.part_id,
                    uri: pw_a.uri,
                    content_hash: pw_a.content_hash,
                    size_bytes_compressed: pw_a.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a.size_bytes_uncompressed,
                    n_superfiles: 2,
                    id_range: (0, 149),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestPartEntry {
                    part_id: pw_b.part_id,
                    uri: pw_b.uri.clone(),
                    content_hash: pw_b.content_hash,
                    size_bytes_compressed: pw_b.size_bytes_compressed,
                    size_bytes_uncompressed: pw_b.size_bytes_uncompressed,
                    n_superfiles: 1,
                    id_range: (0, 199),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
            ],
        };
        let loader = ManifestPartLoader::new(storage, &list);
        let parts_map = DashMap::new();
        parts_map.insert(
            part_a.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a)))),
        );
        parts_map.insert(
            part_b.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_b)))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf_a_keep.clone(), sf_a_remove.clone(), sf_b.clone()],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let (new_manifest, parts) = old_manifest
            .update(&[], from_ref(&sf_a_remove))
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        // 2 list entries, 1 new part (only the first part was rewritten)
        assert_eq!(list_entries.len(), 2);
        assert_eq!(parts.len(), 1);

        // First part: rewritten with 1 surviving superfile.
        assert_eq!(list_entries[0].n_superfiles, 1);
        assert_eq!(parts[0].part.superfiles.len(), 1);
        assert_eq!(
            parts[0].part.superfiles[0].superfile_id,
            sf_a_keep.superfile_id
        );
        let docs_a: u64 = parts[0].part.superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(docs_a, 100);

        // Second part: exact carry-over — URI and content_hash unchanged.
        assert_eq!(list_entries[1].n_superfiles, 1);
        assert_eq!(list_entries[1].uri, pw_b.uri);
        assert_eq!(list_entries[1].content_hash, pw_b.content_hash);
    }

    #[tokio::test]
    async fn update_remove_from_latest_part_in_split_partition() {
        // Two parts from a prior split: part_a_old (1 sf) and part_a_latest
        // (2 sfs). We remove sf_a_latest_remove, which lives in the SECOND
        // (latest) part. The removal set is checked against every part:
        // part_a_old holds no matching id and carries over unchanged, while
        // part_a_latest matches and is rewritten with only its surviving
        // superfile.
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_part = 2;
        let opts = Arc::new(base_opts);

        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        // part_a_old: frozen entry from a prior split
        let sf_a_old = make_superfile_entry(100, pk.clone());
        let part_a_old = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a_old.clone()],
        };
        let pw_a_old = write_manifest_part(storage.as_ref(), &part_a_old, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_a_old");

        // part_a_latest: current mutable entry; contains the sf to remove
        let sf_a_latest_keep = make_superfile_entry(150, pk.clone());
        let sf_a_latest_remove = make_superfile_entry(200, pk.clone());
        let part_a_latest = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a_latest_keep.clone(), sf_a_latest_remove.clone()],
        };
        let pw_a_latest =
            write_manifest_part(storage.as_ref(), &part_a_latest, MANIFEST_ZSTD_LEVEL)
                .await
                .expect("write part_a_latest");

        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![
                ManifestPartEntry {
                    part_id: pw_a_old.part_id,
                    uri: pw_a_old.uri.clone(),
                    content_hash: pw_a_old.content_hash,
                    size_bytes_compressed: pw_a_old.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a_old.size_bytes_uncompressed,
                    n_superfiles: 1,
                    id_range: (0, 99),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestPartEntry {
                    part_id: pw_a_latest.part_id,
                    uri: pw_a_latest.uri.clone(),
                    content_hash: pw_a_latest.content_hash,
                    size_bytes_compressed: pw_a_latest.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a_latest.size_bytes_uncompressed,
                    n_superfiles: 2,
                    id_range: (0, 199),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
            ],
        };
        let loader = ManifestPartLoader::new(storage, &list);
        let parts_map = DashMap::new();
        parts_map.insert(
            part_a_old.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a_old)))),
        );
        parts_map.insert(
            part_a_latest.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a_latest)))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![
                    sf_a_old.clone(),
                    sf_a_latest_keep.clone(),
                    sf_a_latest_remove.clone(),
                ],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let (new_manifest, parts_to_write) = old_manifest
            .update(&[], from_ref(&sf_a_latest_remove))
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        assert_eq!(list_entries.len(), 2);
        // Only the part that actually held the removed superfile is rewritten;
        // the other carries over untouched.
        assert_eq!(parts_to_write.len(), 1);

        // [0] part_a_old carried over. [1] part_a_latest rewritten.

        // sf_a_old survives (in one of the output parts)
        // sf_a_latest_keep survives (in one of the output parts)
        // sf_a_latest_remove is absent from every output part
        let all_ids: Vec<_> = parts_to_write
            .iter()
            .flat_map(|ep| ep.part.superfiles.iter())
            .map(|s| s.superfile_id)
            .collect();
        assert!(
            all_ids.contains(&sf_a_latest_keep.superfile_id),
            "sf_a_latest_keep must survive"
        );
        assert!(
            !all_ids.contains(&sf_a_latest_remove.superfile_id),
            "sf_a_latest_remove must be absent"
        );

        // Each rewritten part has exactly 1 superfile
        assert_eq!(list_entries[0].n_superfiles, 1);
        assert_eq!(list_entries[1].n_superfiles, 1);
    }

    #[tokio::test]
    async fn update_remove_all_superfiles_empties_partition() {
        // All superfiles in a partition are removed. Documents the current
        // behavior: the list entry survives with n_superfiles=0 and the
        // part has no superfiles (empty partition).
        let opts = make_opts();
        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        let sf1 = make_superfile_entry(100, pk.clone());
        let sf2 = make_superfile_entry(150, pk.clone());

        let existing_part = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf1.clone(), sf2.clone()],
        };
        let pw = write_manifest_part(storage.as_ref(), &existing_part, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part");

        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![ManifestPartEntry {
                part_id: pw.part_id,
                uri: pw.uri,
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: 2,
                id_range: (0, 149),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let loader = ManifestPartLoader::new(storage, &list);
        let parts_map = DashMap::new();
        parts_map.insert(
            existing_part.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(existing_part)))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf1.clone(), sf2.clone()],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let (new_manifest, parts) = old_manifest
            .update(&[], &[sf1.clone(), sf2.clone()])
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        // Both superfiles removed: list entry remains with n_superfiles=0.
        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(list_entries[0].n_superfiles, 0);
        assert_eq!(parts[0].part.superfiles.len(), 0);
    }

    #[tokio::test]
    async fn update_remove_nonexistent_superfile_id_is_noop() {
        // entries_to_remove contains a superfile_id that is not present in any
        // part. The filter matches nothing and both original superfiles survive.
        // The part is still rewritten (the removal loop doesn't skip parts where
        // no removal matched), so n_superfiles stays at 2.
        let opts = make_opts();
        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        let sf1 = make_superfile_entry(100, pk.clone());
        let sf2 = make_superfile_entry(150, pk.clone());

        let existing_part = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf1.clone(), sf2.clone()],
        };
        let pw = write_manifest_part(storage.as_ref(), &existing_part, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part");

        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![ManifestPartEntry {
                part_id: pw.part_id,
                uri: pw.uri,
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: 2,
                id_range: (0, 149),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let loader = ManifestPartLoader::new(storage, &list);
        let parts_map = DashMap::new();
        parts_map.insert(
            existing_part.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(existing_part)))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf1.clone(), sf2.clone()],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        // sf_ghost was never added to any part; its superfile_id won't match anything
        let sf_ghost = make_superfile_entry(50, pk.clone());

        let (new_manifest, parts_to_write) = old_manifest
            .update(&[], from_ref(&sf_ghost))
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts_to_write.len(), 0);
        assert_eq!(list_entries[0].n_superfiles, 2);
    }

    #[tokio::test]
    async fn update_remove_from_older_frozen_part_in_split_partition() {
        // Two parts from a prior split: part_a_old (2 sfs: sf_a_old_keep +
        // sf_a_old_remove) and part_a_latest (1 sf). We remove sf_a_old_remove,
        // which lives in the FIRST (older) part. The removal set is checked
        // against every part: part_a_old matches and is rewritten without the
        // removed superfile; part_a_latest holds no matching id and carries over
        // unchanged. sf_a_old_keep and sf_a_latest survive.
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_part = 2;
        let opts = Arc::new(base_opts);

        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        // part_a_old: frozen entry — contains the sf to remove
        let sf_a_old_keep = make_superfile_entry(100, pk.clone());
        let sf_a_old_remove = make_superfile_entry(150, pk.clone());
        let part_a_old = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a_old_keep.clone(), sf_a_old_remove.clone()],
        };
        let pw_a_old = write_manifest_part(storage.as_ref(), &part_a_old, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_a_old");

        // part_a_latest: mutable entry — does not contain the sf to remove
        let sf_a_latest = make_superfile_entry(200, pk.clone());
        let part_a_latest = ManifestPart {
            format_version: part::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a_latest.clone()],
        };
        let pw_a_latest =
            write_manifest_part(storage.as_ref(), &part_a_latest, MANIFEST_ZSTD_LEVEL)
                .await
                .expect("write part_a_latest");

        let list = Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![
                ManifestPartEntry {
                    part_id: pw_a_old.part_id,
                    uri: pw_a_old.uri,
                    content_hash: pw_a_old.content_hash,
                    size_bytes_compressed: pw_a_old.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a_old.size_bytes_uncompressed,
                    n_superfiles: 2,
                    id_range: (0, 149),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestPartEntry {
                    part_id: pw_a_latest.part_id,
                    uri: pw_a_latest.uri,
                    content_hash: pw_a_latest.content_hash,
                    size_bytes_compressed: pw_a_latest.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a_latest.size_bytes_uncompressed,
                    n_superfiles: 1,
                    id_range: (0, 199),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
            ],
        };
        let loader = ManifestPartLoader::new(storage, &list);
        let parts_map = DashMap::new();
        parts_map.insert(
            part_a_old.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a_old)))),
        );
        parts_map.insert(
            part_a_latest.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a_latest)))),
        );
        let old_manifest = Arc::new(ManifestSnapshot {
            superfile_list: SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![
                    sf_a_old_keep.clone(),
                    sf_a_old_remove.clone(),
                    sf_a_latest.clone(),
                ],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let (new_manifest, parts_to_write) = old_manifest
            .update(&[], from_ref(&sf_a_old_remove))
            .await
            .expect("update");
        let list_entries = new_manifest.get_all_list_entries();

        assert_eq!(list_entries.len(), 2);
        // Only the part that held the removed superfile is rewritten; the other
        // carries over untouched.
        assert_eq!(parts_to_write.len(), 1);

        // [0] part_a_old rewritten. [1] part_a_latest carried over.

        // sf_a_old_keep and sf_a_latest survive; sf_a_old_remove is absent
        let all_ids: Vec<_> = parts_to_write
            .iter()
            .flat_map(|ep| ep.part.superfiles.iter())
            .map(|s| s.superfile_id)
            .collect();
        assert!(
            all_ids.contains(&sf_a_old_keep.superfile_id),
            "sf_a_old_keep must survive"
        );
        assert!(
            !all_ids.contains(&sf_a_old_remove.superfile_id),
            "sf_a_old_remove must be absent"
        );

        // Old part now has 1 sf (sf_a_old_remove was removed)
        assert_eq!(list_entries[0].n_superfiles, 1);
        // Latest part still has 1 sf (removal did not touch it)
        assert_eq!(list_entries[1].n_superfiles, 1);
    }

    /// Build a single-part `ManifestList` carrying `n_parts` placeholder
    /// entries — enough to exercise the list-aware `ManifestSnapshot` accessors
    /// without attaching storage.
    fn list_with_parts(n_parts: usize) -> list::Manifest {
        use list::{ManifestPartEntry, PartitionStrategy};
        let parts = (0..n_parts)
            .map(|i| ManifestPartEntry {
                part_id: part::PartId(Uuid::from_u128(i as u128 + 1)),
                uri: format!("manifests/part-{i}"),
                n_superfiles: 0,
                size_bytes_compressed: 0,
                size_bytes_uncompressed: 0,
                content_hash: part::ContentHash([0u8; 32]),
                id_range: (0, 0),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            })
            .collect();
        list::Manifest {
            tombstone_seqs: Default::default(),
            format_version: list::FORMAT_VERSION.into(),
            manifest_id: 1,
            options_hash: part::ContentHash([0u8; 32]),
            schema: Vec::new(),
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts,
        }
    }

    fn manifest_with_list(list: list::Manifest) -> ManifestSnapshot {
        ManifestSnapshot {
            superfile_list: SuperfileList::empty(opts()),
            list: Some(list),
            parts: DashMap::new(),
            loader: None,
        }
    }

    /// `get_num_parts` / `get_all_list_entries` read straight off the
    /// attached `ManifestList` (the Some-arm of both accessors).
    #[test]
    fn list_accessors_read_from_attached_list() {
        let m = manifest_with_list(list_with_parts(3));
        assert_eq!(m.get_num_parts(), 3);
        assert_eq!(m.get_all_list_entries().len(), 3);
        assert_eq!(m.get_num_parts_loaded(), 0, "nothing eagerly loaded");
        assert!(!m.is_in_process_only(), "a list is attached");

        // No-list manifest takes the None-arms.
        let empty = ManifestSnapshot::empty(opts());
        assert_eq!(empty.get_num_parts(), 0);
        assert!(empty.get_all_list_entries().is_empty());
        assert!(empty.is_in_process_only());
    }

    /// `get_cached_part_by_id` / `get_cached_part_by_list_idx` return
    /// `None` before any part is fetched into the per-part cache; the
    /// list-index variant resolves the index to a `PartId` first.
    #[test]
    fn cached_part_lookups_miss_before_load() {
        let m = manifest_with_list(list_with_parts(2));
        let known_id = part::PartId(Uuid::from_u128(1));
        assert!(m.get_cached_part_by_id(&known_id).is_none());
        assert!(m.get_cached_part_by_list_idx(0).is_none());
        assert!(m.get_cached_part_by_list_idx(1).is_none());

        // A manifest with no list has no parts to resolve by index.
        let empty = ManifestSnapshot::empty(opts());
        assert!(empty.get_cached_part_by_list_idx(0).is_none());
    }

    /// `ManifestSnapshot::new` with no storage/list takes the in-process-only
    /// constructor branch (loader + list both `None`).
    #[test]
    fn manifest_new_without_storage_is_in_process_only() {
        let m = ManifestSnapshot::new(7, opts(), vec![seg_entry(Uuid::new_v4(), 4)], None, None);
        assert_eq!(m.get_manifest_id(), 7);
        assert!(m.is_in_process_only());
        assert_eq!(m.get_num_parts(), 0);
        assert_eq!(m.superfiles.len(), 1);
    }

    /// `ClusterCentroids::from_fp32` clamps a non-finite component
    /// min/max to zero (the `is_finite` guard branches).
    #[test]
    fn from_fp32_handles_non_finite_components() {
        let centroids = [f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0];
        let cc = ClusterCentroids::from_fp32(1, 4, &centroids, vec![1]);
        // Degenerate (non-finite) min/max collapse to a zero scale, so
        // every code in the cluster is 0 — no NaN/inf leaks through.
        assert_eq!(cc.scales[0], 0.0);
        assert!(cc.mins[0].is_finite());
        assert!(cc.codes.iter().all(|&c| c == 0));
    }
}
