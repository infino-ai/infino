//! `Supertable` + `SupertableReader` — the in-memory handle.
//!
//! `Supertable::create(opts).expect("create")` returns a clone-shared handle holding
//! an empty initial manifest behind `ArcSwap<Manifest>`.
//! `Supertable::reader()` does `ArcSwap::load_full` once and pins
//! the resulting `Arc<Manifest>` for the reader's lifetime, so a
//! reader captured before a commit keeps seeing pre-commit state
//! even after the writer has swapped in a new manifest.
//!
//! `SupertableInner.writer_outstanding: AtomicBool` is the
//! single-writer slot — the writer flips it true on acquisition
//! and (via `Drop`) flips it false on release.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock};

use arc_swap::ArcSwap;
use tokio::runtime::Runtime;

use super::error::{BuildError, OpenError};
use super::manifest::Manifest;
use super::options::SupertableOptions;
use crate::runtime_bridge::bridge_sync_to_async;

/// Top-level handle. Cheap to clone (one `Arc::clone`); all clones
/// share the same `SupertableInner`. Hand a clone to each thread
/// that wants to read or to acquire the writer.
#[derive(Clone)]
pub struct Supertable {
    inner: Arc<SupertableInner>,
}

/// Internal shared state. Every `Supertable` clone holds one Arc
/// pointing at the same `SupertableInner`. The writer module
/// reaches in to mutate `manifest` (via `ArcSwap::store`) on
/// commit and to manipulate `writer_outstanding` for the
/// single-writer slot enforcement.
pub(super) struct SupertableInner {
    /// Schema, FTS columns, vector columns, tokenizer, thread
    /// pools, segment store, commit threshold. Immutable for
    /// the supertable's lifetime; shared via Arc so readers,
    /// the writer, and rayon shard workers all see the same
    /// instances without copying.
    pub(super) options: Arc<SupertableOptions>,
    /// The current point-in-time view of which superfiles exist.
    /// Each commit publishes a new Manifest via ArcSwap::store;
    /// readers do ArcSwap::load_full at construction to pin a
    /// snapshot for the duration of their queries.
    pub(super) manifest: ArcSwap<Manifest>,
    /// Single-writer slot: the writer flips this true on
    /// acquisition (via compare-exchange) and (via Drop) flips
    /// it false on release. Atomic flag, not a lock — never
    /// blocks; never starves; the slot simply rejects a second
    /// concurrent `Supertable::writer()` call until the first
    /// writer is dropped.
    pub(super) writer_outstanding: AtomicBool,
    /// Generator for the supertable-injected `_id` column.
    /// Each `append()` locks the mutex once, mints
    /// `batch.num_rows()` ids, and unlocks. The
    /// writer-slot lock already serializes `append()` per
    /// supertable handle, so this mutex is uncontended in
    /// practice; it's present only because ferroid's
    /// `BasicSnowflakeGenerator` is `!Sync` by design (it
    /// uses interior-mutable `Cell`). One generator per
    /// supertable, constructed fresh on `create()` /
    /// `open()` with a 40-bit random worker_id.
    pub(super) id_generator: Mutex<crate::supertable::utils::idgen::IdGenerator>,
    /// Lazily-initialized tokio Runtime that drives DataFusion
    /// plans for `query_sql`. Tokio is single-worker here — it
    /// runs the async I/O state machine, not CPU-bound work
    /// (that lives on `options.reader_pool`). One Runtime per
    /// supertable, shared across all SQL queries; allocated on
    /// first use rather than at `create()` so supertables that
    /// never run SQL don't pay the runtime cost.
    pub(super) sql_runtime: OnceLock<Arc<Runtime>>,
    /// Per-process reader-side cache of per-superfile tombstone
    /// bitmaps. `Some` when storage is attached (the cache
    /// fetches sidecars from `superfiles/<id>.tombstones`);
    /// `None` for in-memory-only supertables where no sidecars
    /// can exist. Query paths read through this cache before
    /// returning per-superfile hits; writers invalidate cached
    /// entries after each successful sidecar CAS-PUT.
    pub(super) tombstone_cache: Option<Arc<crate::supertable::tombstones::SidecarCache>>,
    /// Fresh `supertable_handle_id` minted at handle
    /// construction. Used as the `lease.owner` identifier on
    /// every WAL this process drives. Not the OS PID — we need
    /// uniqueness across restarts on the same PID AND across
    /// multiple handles within one process (a process that
    /// opens five supertables holds five distinct ids). Minted
    /// via `IdGenerator::next_id()` once at create / open.
    pub(super) handle_id: crate::supertable::wal::state_doc::SupertableHandleId,
}

impl SupertableInner {
    /// Get (or lazily build) the SQL Runtime.
    pub(super) fn sql_runtime(&self) -> Arc<Runtime> {
        Arc::clone(self.sql_runtime.get_or_init(|| {
            Arc::new(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .enable_all()
                    .thread_name("supertable-sql")
                    .build()
                    .expect(
                        "invariant: tokio Runtime build only fails on \
                         catastrophic OS resource exhaustion",
                    ),
            )
        }))
    }
}

impl Supertable {
    /// Create-or-open from validated options.
    ///
    /// Behaviour:
    ///
    /// - **No storage attached** → fresh in-memory handle, no
    ///   I/O. Empty manifest; recovery is a no-op.
    /// - **Storage attached, no pointer file** → fresh
    ///   storage-backed handle. Empty manifest; recovery sweep
    ///   runs in case prior peer processes left stray WALs.
    /// - **Storage attached, pointer file present** →
    ///   transparently delegates to [`Supertable::open`]. Loads
    ///   the existing manifest list + parts and runs the
    ///   recovery sweep. This closes the "create silently
    ///   shadows existing committed state" footgun.
    ///
    /// Sync API. Internally bridges to async I/O for the
    /// pointer probe + the open delegation via the same
    /// `Handle::try_current() + block_in_place` pattern the
    /// rest of the supertable's sync paths use. Works from
    /// sync `#[test]` contexts and from multi-thread
    /// `#[tokio::test]` contexts; calling from a
    /// `current_thread` runtime panics (use the async
    /// [`Supertable::open`] directly in that case).
    pub fn create(options: SupertableOptions) -> Result<Self, OpenError> {
        // Pointer-probe pass. When storage is attached AND a
        // pointer file already exists, we want open's load path
        // — never silently shadow committed state with an empty
        // manifest.
        if let Some(storage) = options.storage.as_ref() {
            let probe = Arc::clone(storage);
            let probe_result = bridge_sync_to_async(async move {
                crate::supertable::manifest::commit::read_pointer(&*probe).await
            });
            match probe_result {
                Ok(Some(_pointer)) => {
                    return bridge_sync_to_async(Self::open(options));
                }
                Ok(None) => {
                    // No pointer → fall through to fresh-create.
                }
                Err(e) => {
                    return Err(OpenError::Storage(
                        crate::storage::StorageError::Permanent {
                            uri: "_supertable/current".into(),
                            source: Box::new(std::io::Error::other(format!("{e}"))),
                        },
                    ));
                }
            }
        }

        let options = Arc::new(options);
        let initial = Manifest::empty(options.clone());
        let tombstone_cache = build_tombstone_cache(&options);
        let id_generator = crate::supertable::utils::idgen::IdGenerator::new();
        let handle_id =
            crate::supertable::wal::state_doc::SupertableHandleId(id_generator.next_id());
        let inner = Arc::new(SupertableInner {
            options,
            manifest: ArcSwap::new(Arc::new(initial)),
            writer_outstanding: AtomicBool::new(false),
            id_generator: Mutex::new(id_generator),
            sql_runtime: OnceLock::new(),
            tombstone_cache,
            handle_id,
        });
        install_disk_cache_pinning(&inner);
        let st = Self { inner };
        // Open-time recovery + gc sweeps — no-op when no
        // storage is attached. Best-effort: a sweep failure
        // here doesn't fail handle construction; the next
        // sweep gets another shot.
        let _ = st.run_recovery_sweep_once_blocking();
        let _ = bridge_sync_to_async(async { st.run_gc_sweep_once().await.map_err(|_| ()) });
        Ok(st)
    }

    /// Open an existing persisted supertable.
    ///
    /// Reads the pointer file at
    /// `<root>/_supertable/current` via the storage provider
    /// attached on `options`, parses the manifest list, and
    /// eager-fetches manifest parts when the part count is
    /// below `options.eager_load_threshold_parts`. The returned
    /// `Supertable` is ready to serve queries from the
    /// snapshot at the pointer's `manifest_id`.
    ///
    /// Errors:
    /// - [`OpenError::PointerUnreadable`] if the pointer
    ///   doesn't exist (open-or-create trigger).
    /// - [`OpenError::Build`] if `options.storage` is `None`
    ///   (open requires a storage backend).
    /// - [`OpenError::Storage`], [`OpenError::ManifestListParse`],
    ///   [`OpenError::ContentHashMismatch`],
    ///   [`OpenError::ManifestPartLoad`] for fetch / parse
    ///   failures.
    pub async fn open(options: SupertableOptions) -> Result<Self, OpenError> {
        use crate::supertable::ManifestPartLoader;
        use crate::supertable::manifest::commit::read_pointer;
        use crate::supertable::manifest::list as list_mod;
        use crate::supertable::manifest::{Manifest, SuperfileList};

        let storage = options
            .storage
            .as_ref()
            .ok_or_else(|| {
                OpenError::Build(BuildError::Store(
                    "Supertable::open requires options.storage; \
                     attach via .with_storage(...) before calling open"
                        .into(),
                ))
            })?
            .clone();

        // 1. Read the pointer file.
        let pointer = match read_pointer(&*storage).await? {
            Some(p) => p,
            None => {
                // No pointer → no supertable at this location.
                // Map to OpenError::PointerUnreadable so the
                // open-or-create caller can pattern-match.
                return Err(OpenError::PointerUnreadable(
                    crate::storage::StorageError::NotFound {
                        uri: "_supertable/current".into(),
                    },
                ));
            }
        };

        // 2. Load + parse the manifest list.
        let (list_bytes, _) = storage
            .get(&pointer.manifest_list_uri)
            .await
            .map_err(OpenError::Storage)?;
        let list = list_mod::decode(&list_bytes)
            .map_err(|e| OpenError::ManifestListParse(format!("{e}")))?;

        // D15: verify the caller's options match the
        // manifest's stamped digest. The all-zero stored
        // hash bypasses validation (legacy + synthetic
        // fixtures).
        let expected_hash = crate::supertable::manifest::options_hash::compute_options_hash(
            &options,
            &list.partition_strategy,
        );
        if let Err(mismatch) = crate::supertable::manifest::options_hash::verify_options_hash(
            expected_hash,
            list.options_hash,
        ) {
            return Err(OpenError::OptionsHashMismatch {
                expected: mismatch.expected,
                actual: mismatch.actual,
            });
        }

        // 3. Build the loader. Then either eager-fetch every
        //    part (small manifests — fast first query) or
        //    populate empty `OnceCell`s for lazy-load (large
        //    manifests pay no upfront cost; parts hydrate on
        //    first `Manifest::part(id).await`).
        let loader = Arc::new(ManifestPartLoader::new(Arc::clone(&storage), &list));
        let n_parts = list.parts.len();
        let threshold = options.eager_load_threshold_parts as usize;
        let eager = n_parts <= threshold;

        let parts_map = dashmap::DashMap::new();
        let mut all_segments: Vec<Arc<crate::supertable::SuperfileEntry>> = Vec::new();
        if eager {
            // Eager path: parallel-fetch every part + populate
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
            let loaded = futures::future::join_all(load_futs).await;
            for (pid, result) in part_ids.iter().zip(loaded) {
                let part = result.map_err(|e| OpenError::ManifestPartLoad {
                    part_id: pid.0.to_string(),
                    source: Box::new(e),
                })?;
                all_segments.extend(part.superfiles.iter().cloned());
                let cell = tokio::sync::OnceCell::new();
                cell.set(part).expect("fresh OnceCell");
                parts_map.insert(*pid, Arc::new(cell));
            }
        } else {
            // Lazy path (M15b): each part gets an empty
            // `OnceCell`; first `Manifest::part(id).await`
            // triggers a single storage GET for that part.
            // `superfile_list.superfiles` stays empty — legacy
            // flat-iteration queries return zero results
            // until M15c's hierarchical query path lands.
            // Callers in lazy mode today drive
            // `Manifest::part().await` directly.
            for entry in &list.parts {
                parts_map.insert(entry.part_id, Arc::new(tokio::sync::OnceCell::new()));
            }
        }

        // 4. Build the in-memory hierarchical Manifest.
        //    `manifest_id` mirrors the pointer. The flat
        //    `superfile_list.superfiles` is populated only in
        //    eager mode (see above); lazy mode leaves it
        //    empty pending M15c.
        let options_arc = Arc::new(options);
        let mut superfile_list = SuperfileList::empty(options_arc.clone());
        superfile_list.manifest_id = pointer.manifest_id;
        superfile_list.superfiles = all_segments;

        let manifest = Manifest {
            superfile_list,
            list: Some(list),
            parts: parts_map,
            loader: Some(loader),
        };

        let tombstone_cache = build_tombstone_cache(&options_arc);
        // Fresh generator per open. The 64-bit ms timestamp
        // prefix advances naturally across process restarts, so
        // re-opened supertables never re-mint values that already
        // live in storage — no resume-from-id_max-on-open logic
        // needed. The worker_id is also fresh, further insulating
        // restarts from collisions.
        let id_generator = crate::supertable::utils::idgen::IdGenerator::new();
        let handle_id =
            crate::supertable::wal::state_doc::SupertableHandleId(id_generator.next_id());
        let inner = Arc::new(SupertableInner {
            options: options_arc,
            manifest: ArcSwap::new(Arc::new(manifest)),
            writer_outstanding: AtomicBool::new(false),
            id_generator: Mutex::new(id_generator),
            sql_runtime: OnceLock::new(),
            tombstone_cache,
            handle_id,
        });
        install_disk_cache_pinning(&inner);
        let st = Self { inner };
        // Open-time recovery sweep — drives every Intent /
        // Appended WAL discovered in `wal/mutations/` to
        // Complete (or skips lease-conflicted ones for a peer
        // to drive). Best-effort: a sweep failure doesn't fail
        // `open` because the supertable is still functional —
        // the next sweep gets another shot.
        let _ = st.run_recovery_sweep_once().await;
        // GC sweep follows recovery on the same LIST: reaps
        // Complete WALs past `T_wal_grace` and orphan arrow
        // sidecars past `T_sidecar_grace`. Best-effort; same
        // sweep budget.
        let _ = st.run_gc_sweep_once().await;
        Ok(st)
    }

    /// Re-read the manifest pointer from storage.
    /// If the pointer names a newer `manifest_id` than this
    /// supertable's current in-memory state, load the new
    /// list, **inherit** unchanged parts from the current
    /// `Manifest` via content-addressed lookup, eager-fetch
    /// the newly-referenced parts, and `ArcSwap` the new
    /// `Manifest` into place. Pre-refresh `SupertableReader`s
    /// keep their pinned snapshot — the swap is invisible to
    /// them.
    ///
    /// Returns `Ok(true)` iff a newer manifest was loaded.
    /// `Ok(false)` if the pointer hasn't advanced (the cheap
    /// no-op refresh path).
    pub async fn refresh(&self) -> Result<bool, OpenError> {
        use crate::supertable::ManifestPartLoader;
        use crate::supertable::manifest::commit::read_pointer;
        use crate::supertable::manifest::list as list_mod;
        use crate::supertable::manifest::{Manifest, SuperfileList};

        let storage = self
            .inner
            .options
            .storage
            .as_ref()
            .ok_or_else(|| {
                OpenError::Build(BuildError::Store(
                    "Supertable::refresh requires options.storage".into(),
                ))
            })?
            .clone();

        // 1. Read the current pointer. If it's not newer than
        //    our in-memory manifest_id, no-op.
        let pointer = match read_pointer(&*storage).await? {
            Some(p) => p,
            None => return Ok(false),
        };
        let current = self.inner.manifest.load_full();
        if pointer.manifest_id <= current.superfile_list.manifest_id {
            return Ok(false);
        }

        // 2. Load + parse the new manifest list.
        let (list_bytes, _) = storage
            .get(&pointer.manifest_list_uri)
            .await
            .map_err(OpenError::Storage)?;
        let new_list = list_mod::decode(&list_bytes)
            .map_err(|e| OpenError::ManifestListParse(format!("{e}")))?;

        // 3. Inherit unchanged parts via content-addressed
        //    lookup. For each part in the new list whose
        //    PartId is also in the current Manifest's
        //    parts cache, Arc::clone the OnceCell — same
        //    bytes, no re-fetch, no re-parse. Parts in the
        //    new list that aren't in the current cache are
        //    eager-fetched.
        let new_loader = Arc::new(ManifestPartLoader::new(Arc::clone(&storage), &new_list));
        let new_parts: dashmap::DashMap<_, _> = dashmap::DashMap::new();
        let mut missing_part_ids = Vec::new();
        for entry in &new_list.parts {
            if let Some(existing) = current.parts.get(&entry.part_id) {
                new_parts.insert(entry.part_id, existing.value().clone());
            } else {
                missing_part_ids.push(entry.part_id);
            }
        }

        // Eager-fetch the missing ones in parallel — but
        // only when the total post-refresh part count is at
        // or under the eager-load threshold (M15b). Above
        // it, leave missing parts as empty `OnceCell`s for
        // lazy-load on first access, matching the lazy-open
        // semantics. Inherited parts (Arc::clone'd above)
        // keep whatever state they had — already-loaded
        // stays loaded; lazy stays lazy.
        let threshold = self.inner.options.eager_load_threshold_parts as usize;
        let eager = new_list.parts.len() <= threshold;
        if eager {
            let load_futs = missing_part_ids
                .iter()
                .map(|id| {
                    let loader = Arc::clone(&new_loader);
                    let pid = *id;
                    async move { loader.load(pid).await }
                })
                .collect::<Vec<_>>();
            let loaded = futures::future::join_all(load_futs).await;
            for (pid, result) in missing_part_ids.iter().zip(loaded) {
                let part = result.map_err(|e| OpenError::ManifestPartLoad {
                    part_id: pid.0.to_string(),
                    source: Box::new(e),
                })?;
                let cell = tokio::sync::OnceCell::new();
                cell.set(part).expect("fresh cell");
                new_parts.insert(*pid, Arc::new(cell));
            }
        } else {
            for pid in &missing_part_ids {
                new_parts.insert(*pid, Arc::new(tokio::sync::OnceCell::new()));
            }
        }

        // 4. Rebuild the flat superfile_list from all parts in
        //    the new manifest — eager mode only. In lazy
        //    mode the flat view stays empty pending M15c.
        let mut all_segments: Vec<Arc<crate::supertable::SuperfileEntry>> = Vec::new();
        if eager {
            for entry in &new_list.parts {
                let cell = new_parts.get(&entry.part_id).expect("part inserted above");
                let part = cell
                    .value()
                    .get()
                    .expect("eager-fetched or inherited; must be set");
                all_segments.extend(part.superfiles.iter().cloned());
            }
        }

        // 5. Build + ArcSwap the new Manifest.
        let mut new_segment_list = SuperfileList::empty(self.inner.options.clone());
        new_segment_list.manifest_id = pointer.manifest_id;
        new_segment_list.superfiles = all_segments;
        let new_manifest = Manifest {
            superfile_list: new_segment_list,
            list: Some(new_list),
            parts: new_parts,
            loader: Some(new_loader),
        };
        self.inner.manifest.store(Arc::new(new_manifest));
        Ok(true)
    }

    /// Pinned reader. Captures the current manifest at construction
    /// and holds it for its lifetime. New commits don't affect a
    /// live reader; closing + reopening picks up later commits.
    pub fn reader(&self) -> SupertableReader {
        SupertableReader {
            manifest: self.inner.manifest.load_full(),
            tombstone_cache: self.inner.tombstone_cache.clone(),
        }
    }

    /// Per-supertable configuration (schema, FTS / vector columns,
    /// tokenizer). Immutable for the supertable's lifetime.
    pub fn options(&self) -> &Arc<SupertableOptions> {
        &self.inner.options
    }

    /// This handle's lease-owner id. Stamped on every WAL the
    /// handle's recovery sweep / commit pipeline acquires.
    /// Minted once at handle construction via `IdGenerator`;
    /// distinct from every other handle in the process
    /// (different `worker_id`) and from every prior process
    /// (different `ms` timestamp).
    pub fn handle_id(&self) -> crate::supertable::wal::state_doc::SupertableHandleId {
        self.inner.handle_id
    }

    /// Construct a [`Supertable`] handle wrapping an existing
    /// `SupertableInner` arc. Internal-only: used by the writer
    /// to hand a `Supertable` to the WAL pipeline functions
    /// without re-running the full create-or-open flow. Skips
    /// the open-time recovery sweep on purpose — the inner has
    /// already been initialized.
    pub(super) fn from_inner(inner: Arc<SupertableInner>) -> Self {
        Self { inner }
    }

    /// Operator hatch: run one WAL recovery sweep against this
    /// supertable's storage prefix. Useful for long-lived
    /// handles that want bounded recovery latency without
    /// restarting the process, and for integration tests that
    /// pre-seed half-finished WALs and verify the sweep
    /// completes them.
    ///
    /// Returns `Ok(report)` with the per-outcome counts on
    /// success; `Err(NoStorageAttached)` for in-memory-only
    /// supertables (no WALs can exist there).
    pub async fn run_recovery_sweep_once(
        &self,
    ) -> Result<
        crate::supertable::wal::recovery::RecoveryReport,
        crate::supertable::wal::recovery::RecoveryError,
    > {
        crate::supertable::wal::recovery::scan_and_recover(
            self,
            self.inner.handle_id,
            crate::supertable::wal::lease::DEFAULT_LEASE_DURATION,
        )
        .await
    }

    /// Sync-bridged version of [`run_recovery_sweep_once`]. Used
    /// by [`Supertable::create`] to drive an open-time sweep
    /// from a sync entry point. Same sync→async pattern the
    /// writer's `persist_commit` uses: ride the ambient tokio
    /// runtime when present, lazy-init the supertable's owned
    /// runtime otherwise.
    pub fn run_recovery_sweep_once_blocking(
        &self,
    ) -> Result<
        crate::supertable::wal::recovery::RecoveryReport,
        crate::supertable::wal::recovery::RecoveryError,
    > {
        let drive = self.run_recovery_sweep_once();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(drive)),
            Err(_) => self.inner.sql_runtime().block_on(drive),
        }
    }

    /// Operator hatch: run one GC sweep over this supertable's
    /// `wal/mutations/` prefix. Reaps `Complete` WALs older
    /// than the wal-grace window + orphan `.arrow` sidecars
    /// older than the sidecar-grace window. Tests pass custom
    /// grace windows via [`run_gc_sweep_with_grace`].
    pub async fn run_gc_sweep_once(
        &self,
    ) -> Result<crate::supertable::wal::gc::GcReport, crate::supertable::wal::gc::GcError> {
        crate::supertable::wal::gc::run_sweep(
            self,
            chrono::Utc::now(),
            crate::supertable::wal::gc::DEFAULT_WAL_GRACE,
            crate::supertable::wal::gc::DEFAULT_SIDECAR_GRACE,
        )
        .await
    }

    /// Same as [`run_gc_sweep_once`] but with caller-supplied
    /// `now` + grace windows. Useful for deterministic tests
    /// that simulate grace-window crossings without sleeping.
    pub async fn run_gc_sweep_with_grace(
        &self,
        now: chrono::DateTime<chrono::Utc>,
        wal_grace: std::time::Duration,
        sidecar_grace: std::time::Duration,
    ) -> Result<crate::supertable::wal::gc::GcReport, crate::supertable::wal::gc::GcError> {
        crate::supertable::wal::gc::run_sweep(self, now, wal_grace, sidecar_grace).await
    }

    /// Current manifest's id, without pinning a reader. Useful for
    /// observability + tests that want to assert "a commit
    /// happened" without holding a snapshot.
    pub fn manifest_id(&self) -> u64 {
        self.inner.manifest.load().manifest_id
    }

    /// Observability snapshot of the supertable's load.
    /// Cheap to call: one RSS syscall + an `ArcSwap::load` + a couple of
    /// length reads on the in-memory manifest. See
    /// [`crate::supertable::SupertableStats`] for the field-level contract.
    pub fn stats(&self) -> crate::supertable::SupertableStats {
        let manifest = self.inner.manifest.load();
        let n_manifest_parts = manifest.list.as_ref().map(|l| l.parts.len());
        let cache = self.inner.options.disk_cache.as_ref();
        let mmap_resident_bytes = cache.map(|c| c.current_mmap_size_bytes());
        // One `cache.stats()` call covers four fields. Cache
        // counters are atomic loads, so the snapshot is
        // self-consistent for each counter but not coherent
        // across counters under heavy concurrent activity —
        // adequate for observability.
        let cache_snapshot = cache.map(|c| c.stats());
        crate::supertable::SupertableStats {
            manifest_id: manifest.superfile_list.manifest_id,
            n_superfiles: manifest.superfile_list.superfiles.len(),
            n_manifest_parts,
            n_manifest_parts_loaded: manifest.parts.len(),
            process_rss_bytes: crate::supertable::stats::process_rss_bytes(),
            mmap_resident_bytes,
            memory_budget_bytes: self.inner.options.memory_budget_bytes,
            n_cold_fetches: cache_snapshot.as_ref().map(|s| s.n_cold_fetches),
            n_cache_evictions: cache_snapshot.as_ref().map(|s| s.n_evictions),
            n_cache_madvise_calls: cache_snapshot.as_ref().map(|s| s.n_madvise_calls),
            n_cache_entries: cache_snapshot.as_ref().map(|s| s.n_entries),
        }
    }

    /// Internal accessor used by the writer module. Not part of
    /// the public API.
    pub(super) fn inner(&self) -> &Arc<SupertableInner> {
        &self.inner
    }

    /// SQL Runtime accessor, exposed within the crate for the
    /// `query::sql` module's `block_on`. Lazy: first call
    /// allocates a single-worker tokio Runtime cached on
    /// `SupertableInner`; subsequent calls clone the `Arc`.
    pub(crate) fn sql_runtime(&self) -> Arc<Runtime> {
        self.inner.sql_runtime()
    }
}

/// M14b.1 — install a manifest-aware pinned-URI callback on
/// the attached `DiskCacheStore`. Called from
/// [`Supertable::create`] and [`Supertable::open`] right
/// after the `Arc<SupertableInner>` is built; before the
/// supertable is exposed to any concurrent user.
///
/// The closure captures a `Weak<SupertableInner>` (not a
/// strong `Arc`) — without that, the cache holds the
/// supertable alive and the supertable holds the cache
/// alive, leaking both at drop. The `Weak::upgrade` is
/// cheap and bounded: on each eviction sweep, returns the
/// current `Manifest`'s segment URI set; if the supertable
/// has already dropped (cache outlived it), returns the
/// empty set — eviction proceeds without pinning, which is
/// the safe fallback.
/// Build the tombstone-sidecar cache when storage is attached.
/// Returns `None` for in-memory-only supertables — no sidecars
/// can exist there, so the query paths skip the filter hook
/// entirely.
fn build_tombstone_cache(
    options: &Arc<SupertableOptions>,
) -> Option<Arc<crate::supertable::tombstones::SidecarCache>> {
    let storage = options.storage.as_ref()?.clone();
    let wal_store = crate::supertable::wal::WalStore::new(storage);
    Some(Arc::new(crate::supertable::tombstones::SidecarCache::new(
        wal_store,
        crate::supertable::tombstones::cache::DEFAULT_REFRESH_TTL,
    )))
}

fn install_disk_cache_pinning(inner: &Arc<SupertableInner>) {
    let cache = match inner.options.disk_cache.as_ref() {
        Some(c) => c,
        None => return,
    };
    let weak = Arc::downgrade(inner);
    let pinned_fn: Arc<
        dyn Fn() -> std::collections::HashSet<crate::supertable::SuperfileUri> + Send + Sync,
    > = Arc::new(move || {
        let strong = match weak.upgrade() {
            Some(s) => s,
            // Supertable already dropped; nothing to pin.
            None => return std::collections::HashSet::new(),
        };
        strong
            .manifest
            .load()
            .superfile_list
            .superfiles
            .iter()
            .map(|e| e.uri)
            .collect()
    });
    cache.set_pinned_fn(pinned_fn);
}

impl std::fmt::Debug for Supertable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = self.inner.manifest.load();
        f.debug_struct("Supertable")
            .field("manifest_id", &m.manifest_id)
            .field("n_superfiles", &m.superfiles.len())
            .field("id_column", &self.inner.options.id_column)
            .finish()
    }
}

/// Snapshot-pinned reader. Captures `Arc<Manifest>` at construction
/// and holds it through query lifetime — new commits to the parent
/// `Supertable` don't affect this reader's view. Query methods
/// (`bm25_search`, `vector_search`, etc.) are added by the query
/// modules on top of this handle.
pub struct SupertableReader {
    manifest: Arc<Manifest>,
    /// Per-process tombstone-bitmap cache shared with the parent
    /// `Supertable`. Query paths read through this before
    /// returning per-superfile hits so tombstoned rows never
    /// reach callers. `None` for in-memory-only supertables.
    pub(crate) tombstone_cache: Option<Arc<crate::supertable::tombstones::SidecarCache>>,
}

impl SupertableReader {
    /// Manifest id pinned at construction. Useful for asserting
    /// reader-vs-writer visibility ordering in tests.
    pub fn manifest_id(&self) -> u64 {
        self.manifest.manifest_id
    }

    /// Number of superfiles visible to this reader.
    pub fn n_superfiles(&self) -> usize {
        self.manifest.superfiles.len()
    }

    /// Total documents across all superfiles visible to this reader.
    pub fn n_docs_total(&self) -> u64 {
        self.manifest.n_docs_total()
    }

    /// Pinned manifest. Exposed for query-side machinery
    /// (skip helpers, fan-out, etc.) to read the segment list
    /// + summaries directly.
    pub fn manifest(&self) -> &Arc<Manifest> {
        &self.manifest
    }
}

impl std::fmt::Debug for SupertableReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SupertableReader")
            .field("manifest_id", &self.manifest.manifest_id)
            .field("n_superfiles", &self.manifest.superfiles.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema};
    use uuid::Uuid;

    use crate::superfile::builder::FtsConfig;

    use crate::supertable::manifest::{ScalarStatsTable, SuperfileEntry, SuperfileUri};

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]))
    }

    fn opts() -> SupertableOptions {
        let tk = crate::test_helpers::default_tokenizer();
        SupertableOptions::new(
            schema(),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(tk),
        )
        .expect("valid options")
    }

    fn entry(n_docs: u64) -> Arc<SuperfileEntry> {
        let id = Uuid::new_v4();
        Arc::new(SuperfileEntry {
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs,
            id_min: 0,
            id_max: n_docs.saturating_sub(1) as i128,
            scalar_stats: ScalarStatsTable::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
        })
    }

    /// Test-only helper: publish a successor manifest by appending
    /// superfiles and ArcSwap'ing the result into place. Equivalent
    /// to what the writer will do at commit time, exposed here so
    /// the manifest-swap behavior can be exercised in tests
    /// without depending on writer machinery.
    fn publish_appended(st: &Supertable, entries: Vec<Arc<SuperfileEntry>>) {
        let old = st.inner.manifest.load();
        let new = old.with_appended(entries);
        st.inner.manifest.store(Arc::new(new));
    }

    #[test]
    fn create_returns_handle_with_empty_initial_manifest() {
        let st = Supertable::create(opts()).expect("create");
        assert_eq!(st.manifest_id(), 0);
        let r = st.reader();
        assert_eq!(r.manifest_id(), 0);
        assert_eq!(r.n_superfiles(), 0);
        assert_eq!(r.n_docs_total(), 0);
    }

    #[test]
    fn supertable_clone_shares_inner_state() {
        let st1 = Supertable::create(opts()).expect("create");
        let st2 = st1.clone();
        // Same Arc<SupertableInner> behind both clones — verify
        // by mutating through one and observing through the other.
        publish_appended(&st1, vec![entry(50)]);
        assert_eq!(st2.manifest_id(), 1);
    }

    #[test]
    fn options_accessor_returns_arc_to_validated_options() {
        let st = Supertable::create(opts()).expect("create");
        let opts_arc = st.options();
        assert_eq!(opts_arc.id_column, "_id");
        assert_eq!(opts_arc.fts_columns.len(), 1);
    }

    #[test]
    fn reader_pins_manifest_across_subsequent_commits() {
        // The load-bearing reader-isolation invariant: a reader
        // captured before a commit must keep seeing the pre-commit
        // manifest, even after the writer has ArcSwap::store'd a
        // new one.
        let st = Supertable::create(opts()).expect("create");

        // Pin reader at manifest_id = 0.
        let pinned = st.reader();
        assert_eq!(pinned.manifest_id(), 0);
        assert_eq!(pinned.n_superfiles(), 0);

        // Publish 2 superfiles → manifest_id = 1.
        publish_appended(&st, vec![entry(10), entry(20)]);
        assert_eq!(st.manifest_id(), 1);

        // Pinned reader still sees the OLD manifest.
        assert_eq!(pinned.manifest_id(), 0);
        assert_eq!(pinned.n_superfiles(), 0);

        // Fresh reader sees the NEW manifest.
        let fresh = st.reader();
        assert_eq!(fresh.manifest_id(), 1);
        assert_eq!(fresh.n_superfiles(), 2);
        assert_eq!(fresh.n_docs_total(), 30);
    }

    #[test]
    fn manifest_immutability_property() {
        // Property: every successor manifest is structurally
        // independent of its predecessors. After several commits,
        // each prior reader's pinned manifest reports its
        // construction-time state, not the latest.
        let st = Supertable::create(opts()).expect("create");

        let r0 = st.reader();
        publish_appended(&st, vec![entry(1)]);
        let r1 = st.reader();
        publish_appended(&st, vec![entry(2)]);
        let r2 = st.reader();
        publish_appended(&st, vec![entry(3)]);
        let r3 = st.reader();

        // Each reader's manifest_id matches the one published at
        // its capture time.
        assert_eq!(r0.manifest_id(), 0);
        assert_eq!(r1.manifest_id(), 1);
        assert_eq!(r2.manifest_id(), 2);
        assert_eq!(r3.manifest_id(), 3);

        // Segment counts are monotonic across capture times.
        assert_eq!(r0.n_superfiles(), 0);
        assert_eq!(r1.n_superfiles(), 1);
        assert_eq!(r2.n_superfiles(), 2);
        assert_eq!(r3.n_superfiles(), 3);

        // Doc counts add up correctly per pinned snapshot.
        assert_eq!(r0.n_docs_total(), 0);
        assert_eq!(r1.n_docs_total(), 1);
        assert_eq!(r2.n_docs_total(), 1 + 2);
        assert_eq!(r3.n_docs_total(), 1 + 2 + 3);
    }

    #[test]
    fn reader_manifest_arc_outlives_supertable_drop() {
        // The reader's pinned Arc<Manifest> must keep the manifest
        // alive even after the parent Supertable is dropped. This
        // is the "snapshot pinned past the supertable's lifetime"
        // guarantee — the underlying superfiles stay reachable.
        let r = {
            let st = Supertable::create(opts()).expect("create");
            publish_appended(&st, vec![entry(5)]);
            st.reader()
            // st dropped here; reader survives.
        };
        assert_eq!(r.manifest_id(), 1);
        assert_eq!(r.n_superfiles(), 1);
        assert_eq!(r.n_docs_total(), 5);
    }

    #[test]
    fn many_concurrent_readers_share_one_manifest() {
        // Two readers issued at the same point should pin the SAME
        // Arc<Manifest>. The Arc-share is what makes "thousands of
        // concurrent readers" cheap: one allocation, N+1 ref count.
        let st = Supertable::create(opts()).expect("create");
        publish_appended(&st, vec![entry(7)]);
        let r1 = st.reader();
        let r2 = st.reader();
        assert!(Arc::ptr_eq(r1.manifest(), r2.manifest()));
    }

    #[test]
    fn debug_format_doesnt_explode() {
        let st = Supertable::create(opts()).expect("create");
        let s = format!("{:?}", st);
        assert!(s.contains("Supertable"));

        let r = st.reader();
        let s = format!("{:?}", r);
        assert!(s.contains("SupertableReader"));
    }
}
