// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

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

use std::future::Future;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use arc_swap::ArcSwap;
use datafusion::execution::context::SessionContext;
use tokio::runtime::Runtime;

use super::error::{BuildError, OpenError};
use super::manifest::Manifest;
use super::options::SupertableOptions;
use crate::runtime_bridge::{bridge_on_runtime, bridge_sync_to_async};
use crate::supertable::ManifestLoadError;

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
    /// pools, superfile store, commit threshold. Immutable for
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
    pub(super) query_runtime: OnceLock<Arc<Runtime>>,
    /// Cached `SessionContext` for `query_sql`, keyed on the
    /// manifest `Arc` it was built against. Building one is
    /// ~1.5 ms (default optimizer rules + 3 TVF re-registrations
    /// + provider register), so reusing it across queries on the
    /// same snapshot is a large speedup for warm BM25 / vector
    /// SQL where the kernel itself runs in microseconds.
    ///
    /// Invalidation is automatic: every commit publishes a new
    /// `Arc<Manifest>` via `manifest.store(...)`, so on the next
    /// `query_sql` the `Arc::ptr_eq` check fails and the cache
    /// is rebuilt against the fresh snapshot.
    pub(super) sql_session_cache: Mutex<Option<(Arc<Manifest>, SessionContext)>>,
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
    /// Last time the read path checked the storage manifest pointer
    /// for freshness, under [`Consistency::BoundedStaleness`]. `None`
    /// until the first check (so the first query always refreshes).
    /// Unused for [`Consistency::Strong`] (always checks) and
    /// [`Consistency::Snapshot`] (never checks).
    pub(super) last_pointer_check: Mutex<Option<std::time::Instant>>,
}

impl Drop for SupertableInner {
    /// Tear down the lazily-built query runtime without tripping
    /// tokio's "cannot drop a runtime from within an async context"
    /// guard.
    ///
    /// The public API is sync, but it explicitly supports being
    /// called from inside a caller's own multi-thread runtime (the
    /// sync→async bridge uses `block_in_place` there). In that mode a
    /// sync query lazily builds the owned `query_runtime`. If the
    /// caller then drops their last `Supertable` handle while still
    /// inside their runtime, the default `Arc<Runtime>` drop would
    /// panic. `shutdown_background` consumes the runtime without
    /// blocking, so it is safe from any context. The `try_unwrap`
    /// guard ensures we only shut it down when this is the last
    /// owner; otherwise an outstanding transient clone (never the
    /// last reference) just decrements normally.
    fn drop(&mut self) {
        if let Some(rt) = self.query_runtime.take()
            && let Ok(rt) = Arc::try_unwrap(rt)
        {
            rt.shutdown_background();
        }
    }
}

impl SupertableInner {
    /// Get (or lazily build) the runtime that drives the public sync
    /// API's async kernels when the caller is not already on a Tokio
    /// runtime (queries, SQL, writer commits). Sized to the host's
    /// parallelism: the cold read path fans a query out across every
    /// superfile via `tokio::spawn` + `spawn_blocking` (range GETs,
    /// CRC verification, zstd decode), so a single worker would
    /// serialize that fan-out and inflate cold latency. One worker per
    /// CPU lets those overlap, matching what an async caller gets.
    pub(super) fn query_runtime(&self) -> Arc<Runtime> {
        Arc::clone(self.query_runtime.get_or_init(|| {
            // Fallback worker count when the host won't report its
            // parallelism; small but multi-threaded so the cold-read
            // fan-out still overlaps rather than serializing.
            const FALLBACK_QUERY_RUNTIME_WORKERS: usize = 4;
            let workers = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(FALLBACK_QUERY_RUNTIME_WORKERS);
            Arc::new(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(workers)
                    .enable_all()
                    .thread_name("supertable-query")
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
    // Interim options-based constructor — not on the curated public surface
    // (the catalog `create_table` supersedes it). `pub` under `test-helpers`
    // so tests/benches reach it directly; `pub(crate)` otherwise, where the
    // catalog `Connection` calls it internally.
    test_visible! {
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
    /// `#[tokio::test]` contexts. In-memory creates avoid the
    /// open-time sweep bridge entirely because no WAL/GC I/O can
    /// exist without attached storage.
    fn create(options: SupertableOptions) -> Result<Self, OpenError> {
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
                    return Self::open(options);
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
            query_runtime: OnceLock::new(),
            sql_session_cache: Mutex::new(None),
            tombstone_cache,
            handle_id,
            last_pointer_check: Mutex::new(None),
        });
        install_disk_cache_pinning(&inner);
        let st = Self { inner };
        // Open-time recovery + GC sweeps need storage. For in-memory
        // supertables they are guaranteed no-ops, so skip the async
        // bridge; this keeps `Supertable::create` usable inside
        // current-thread `#[tokio::test]` contexts for pure in-memory
        // unit tests.
        if st.inner.options.storage.is_some() {
            // Best-effort: a sweep failure here doesn't fail handle
            // construction; the next sweep gets another shot.
            let _ = st.run_recovery_sweep_once_blocking();
            let _ = bridge_sync_to_async(async { st.run_gc_sweep_once().await.map_err(|_| ()) });
        }
        Ok(st)
    }
    }

    // Interim options-based open — internal counterpart of `create`; the
    // catalog `Connection` calls it internally, tests/benches reach it via
    // `test-helpers`.
    test_visible! {
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
    /// - [`OpenError::ManifestLoadError`] for manifest load failures.
    /// - [`OpenError::Build`] if `options.storage` is `None`
    ///   (open requires a storage backend).
    /// - [`OpenError::Storage`], [`OpenError::ManifestListParse`],
    ///   [`OpenError::ContentHashMismatch`],
    ///   [`OpenError::ManifestPartLoad`] for fetch / parse
    ///   failures.
    ///
    /// Sync public API. Internally bridges to the async storage I/O
    /// via the same `Handle::try_current() + block_in_place` pattern
    /// as the rest of the supertable's sync surface.
    fn open(options: SupertableOptions) -> Result<Self, OpenError> {
        bridge_sync_to_async(Self::open_async(options))
    }
    }

    /// Async open kernel. `pub(crate)` — the public surface is the
    /// sync [`Supertable::open`]; this is the I/O implementation it
    /// (and the open-time create path) drive on the ambient runtime.
    pub(crate) async fn open_async(options: SupertableOptions) -> Result<Self, OpenError> {
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
        let options_arc = Arc::new(options);

        let manifest = Manifest::load(None, storage, Some(options_arc.clone())).await?;
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
            manifest: ArcSwap::new(manifest),
            writer_outstanding: AtomicBool::new(false),
            id_generator: Mutex::new(id_generator),
            query_runtime: OnceLock::new(),
            sql_session_cache: Mutex::new(None),
            tombstone_cache,
            handle_id,
            last_pointer_check: Mutex::new(None),
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
    ///
    /// `pub(crate)` — not a public verb. Freshness is engine-driven
    /// via [`Supertable::ensure_fresh`] on the read path, governed by
    /// [`crate::supertable::options::Consistency`]. This is the
    /// mechanism that drives the pointer re-check.
    pub(crate) async fn refresh(&self) -> Result<bool, OpenError> {
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

        let current = self.inner.manifest.load_full();
        let manifest = match Manifest::load(Some(current), storage, None).await {
            Ok(manifest) => manifest,
            Err(ManifestLoadError::PointerNotFound) => return Ok(false),
            Err(ManifestLoadError::AlreadyLoaded) => return Ok(false),
            Err(err) => return Err(OpenError::ManifestLoadError(err)),
        };
        self.inner.manifest.store(manifest);
        Ok(true)
    }

    /// Current manifest's id, without pinning a reader. Useful for
    /// observability + tests that want to assert "a commit
    /// happened" without holding a snapshot.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn manifest_id(&self) -> u64 {
        self.inner.manifest.load().manifest_id
    }

    test_visible! {
    /// Pinned reader. Captures the current manifest at construction
    /// and holds it for its lifetime. New commits don't affect a
    /// live reader; closing + reopening picks up later commits.
    ///
    /// Applies the read-consistency policy ([`Supertable::ensure_fresh`])
    /// before pinning, so the reader observes the freshest manifest
    /// the configured
    /// [`Consistency`](crate::supertable::options::Consistency) allows.
    /// No-op for an in-memory supertable and under `Snapshot`.
    fn reader(&self) -> SupertableReader {
        self.ensure_fresh();
        SupertableReader {
            manifest: self.inner.manifest.load_full(),
            tombstone_cache: self.inner.tombstone_cache.clone(),
            inner: Arc::clone(&self.inner),
        }
    }
    }

    /// Engine-driven read-path freshness. Applies
    /// `options.read_consistency` ([`crate::supertable::options::Consistency`]):
    /// re-checks the storage manifest pointer and advances the
    /// in-memory snapshot when a newer `manifest_id` is published, so
    /// the next [`Supertable::reader`] sees committed data without the
    /// application ever calling refresh by hand.
    ///
    /// Called at the head of every public query method. No-op for an
    /// in-memory supertable (no storage pointer) and for
    /// [`Consistency::Snapshot`](crate::supertable::options::Consistency::Snapshot).
    /// Best-effort: a failed pointer read leaves the current snapshot
    /// in place rather than failing the query.
    pub(crate) fn ensure_fresh(&self) {
        use crate::supertable::options::Consistency;
        if self.inner.options.storage.is_none() {
            return;
        }
        match self.inner.options.read_consistency {
            Consistency::Snapshot => {}
            Consistency::Strong => {
                let _ = bridge_sync_to_async(self.refresh());
            }
            Consistency::BoundedStaleness(window) => {
                // Decide whether a check is due under the lock, stamp
                // "now" optimistically so concurrent queries don't all
                // stampede the pointer, then release the lock *before*
                // the (blocking) pointer read.
                let due = {
                    let mut last = self
                        .inner
                        .last_pointer_check
                        .lock()
                        .expect("last_pointer_check mutex poisoned");
                    let due = last.map(|t| t.elapsed() >= window).unwrap_or(true);
                    if due {
                        *last = Some(std::time::Instant::now());
                    }
                    due
                };
                if due {
                    let _ = bridge_sync_to_async(self.refresh());
                }
            }
        }
    }

    test_visible! {
    /// Per-supertable configuration (schema, FTS / vector columns,
    /// tokenizer). Immutable for the supertable's lifetime.
    fn options(&self) -> &Arc<SupertableOptions> {
        &self.inner.options
    }
    }

    /// The user-facing Arrow schema — the columns the caller supplied.
    /// The auto-injected `_id` is not part of this schema.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, IndexSpec};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # let posts = db.create_table("posts", schema, IndexSpec::new().fts("body"))?;
    /// assert_eq!(posts.schema().field(0).name(), "body");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn schema(&self) -> arrow_schema::SchemaRef {
        self.inner.options.user_schema()
    }

    /// Sync→async bridge for the public query surface. Mirrors the
    /// runtime handling in [`Supertable::query_sql`]: when a caller is
    /// already on a `multi_thread` runtime, reuse it via
    /// `block_in_place`; otherwise drive the future on the lazily-built
    /// `query_runtime`. Lets `vector_search` / `bm25_search` /
    /// `bm25_search_prefix` present a sync public API over the async
    /// `SupertableReader` kernels without spinning a throwaway runtime
    /// per call.
    pub(crate) fn block_on_query<F: Future>(&self, fut: F) -> F::Output {
        bridge_on_runtime(fut, &self.query_runtime())
    }

    /// Block until the on-disk cache has fully promoted every superfile
    /// in the current manifest to an mmap-backed reader, or `timeout`
    /// elapses for one of them. This is the public "warm-readiness"
    /// primitive: once it returns `Ok(())`, subsequent searches read
    /// from resident mmap pages instead of issuing object-store range
    /// GETs through the lazy foreground source, so latency drops from
    /// the cold/lazy path (hundreds of ms — seconds against real S3) to
    /// the in-memory steady state (single-digit ms).
    ///
    /// A real serving node calls this on startup, after `open`, to take
    /// traffic only once its cache is hot. No-op when no disk cache is
    /// attached, and a short-circuit when background fill is disabled
    /// (`INFINO_DISABLE_BG_FILL`) — nothing is ever promoted then, so
    /// there is nothing to wait for and blocking until `timeout` would
    /// be pointless.
    ///
    /// Crucially, requesting promotion here is also what *drives* it to
    /// completion: registering a promotion waiter releases the
    /// background full-superfile fill that otherwise idles behind
    /// foreground lazy readers under steady query load. Warming purely
    /// by replaying queries does not register that waiter, so the
    /// superfiles can stay lazy/S3-backed indefinitely.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn wait_until_warm(
        &self,
        timeout: std::time::Duration,
    ) -> Result<(), crate::supertable::reader_cache::disk::DiskCacheError> {
        let Some(cache) = self.inner.options.disk_cache.as_ref() else {
            return Ok(());
        };
        if crate::supertable::reader_cache::disk::skip_background_fill() {
            return Ok(());
        }
        let cache = Arc::clone(cache);
        let manifest = self.inner.manifest.load_full();
        self.block_on_query(async move {
            for entry in manifest.superfiles.iter() {
                cache.wait_until_mmap_promoted(&entry.uri, timeout).await?;
            }
            Ok(())
        })
    }

    /// This handle's lease-owner id. Stamped on every WAL the
    /// handle's recovery sweep / commit pipeline acquires.
    /// Minted once at handle construction via `IdGenerator`;
    /// distinct from every other handle in the process
    /// (different `worker_id`) and from every prior process
    /// (different `ms` timestamp). Test-only accessor — production
    /// code reads `inner.handle_id` directly.
    #[cfg(test)]
    pub(crate) fn handle_id(&self) -> crate::supertable::wal::state_doc::SupertableHandleId {
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
    /// Not public API: WAL recovery is engine-driven — it runs
    /// automatically on [`Supertable::open`]. This manual hook is a
    /// crate internal used only by in-crate unit tests that pre-seed
    /// half-finished WALs and assert the sweep completes them.
    pub(crate) async fn run_recovery_sweep_once(
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
    pub(crate) fn run_recovery_sweep_once_blocking(
        &self,
    ) -> Result<
        crate::supertable::wal::recovery::RecoveryReport,
        crate::supertable::wal::recovery::RecoveryError,
    > {
        let drive = self.run_recovery_sweep_once();
        bridge_on_runtime(drive, &self.inner.query_runtime())
    }

    /// Operator hatch: run one GC sweep over this supertable's
    /// `wal/mutations/` prefix. Reaps `Complete` WALs older
    /// than the wal-grace window + orphan `.arrow` sidecars
    /// older than the sidecar-grace window. Tests that need custom
    /// grace windows call `crate::supertable::wal::gc::run_sweep`
    /// directly.
    /// Not public API: WAL/sidecar GC is engine-driven — it runs
    /// automatically on [`Supertable::open`] and (production) on a
    /// background cadence. This manual hook is a crate internal used
    /// only by in-crate unit tests.
    pub(crate) async fn run_gc_sweep_once(
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

    /// Observability snapshot of the supertable's load.
    /// Cheap to call: one RSS syscall + an `ArcSwap::load` + a couple of
    /// length reads on the in-memory manifest. See
    /// [`crate::supertable::SupertableStats`] for the field-level contract.
    #[cfg(any(test, feature = "test-helpers"))]
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
    pub(crate) fn query_runtime(&self) -> Arc<Runtime> {
        self.inner.query_runtime()
    }

    /// Crate-internal accessor for the cached `SessionContext`
    /// keyed on the manifest `Arc`. Used by `query_sql` to
    /// reuse the registered provider + TVFs across queries on
    /// the same snapshot.
    pub(crate) fn sql_session_cache(&self) -> &Mutex<Option<(Arc<Manifest>, SessionContext)>> {
        &self.inner.sql_session_cache
    }

    /// Diagnostic-only: returns the cached `SessionContext`
    /// (building it on miss), bypassing the run-and-collect
    /// path. Lets benchmarks decompose `query_sql` cost into
    /// `ctx.sql()` (parse + analyze + logical/physical plan)
    /// vs `DataFrame::collect()` (execute) to find where the
    /// remaining dispatch time goes after the cache hit.
    #[doc(hidden)]
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn __debug_cached_session(&self) -> SessionContext {
        // Reuses the same fast path as `query_sql` — see the
        // doc-comment on `sql_session_cache` for invalidation.
        self.reader().query_sql("SELECT 1 WHERE 1=0").ok();
        let guard = self
            .sql_session_cache()
            .lock()
            .expect("sql_session_cache mutex poisoned");
        guard
            .as_ref()
            .map(|(_, ctx)| ctx.clone())
            .expect("session cache must be populated after warm-up call")
    }
}

/// Install the eviction-pinning policy on the attached
/// `DiskCacheStore`. Called from [`Supertable::create`] and
/// [`Supertable::open`] right after the `Arc<SupertableInner>`
/// is built; before the supertable is exposed to any
/// concurrent user.
///
/// Policy: **pin nothing.** The cache is a bounded LRU and must
/// be free to evict any superfile to stay under its budget — an
/// index larger than the cache budget has to be able to
/// stream/evict through it. (Previously this pinned the entire
/// live manifest, which made the index *required* to fit inside
/// the budget: once the cache filled, every entry was pinned,
/// eviction found "no eligible victims", and the next admit
/// hard-failed with `BudgetExceeded`.)
///
/// Pinning the live index was never needed for in-flight
/// correctness: a query holds an `Arc<SuperfileReader>` over an
/// mmap, and the cache can evict + unlink the backing file while
/// that mapping stays valid (POSIX keeps the inode alive until
/// the last reference drops). So eviction during a read is
/// already safe without pinning.
///
/// Left as a function (rather than inlined) so a future
/// genuinely-in-flight pin set (URIs a query is actively
/// holding) can be wired here if a workload ever needs it —
/// but that is a *bounded* set, never the whole manifest.
fn install_disk_cache_pinning(inner: &Arc<SupertableInner>) {
    let cache = match inner.options.disk_cache.as_ref() {
        Some(c) => c,
        None => return,
    };
    let pinned_fn: Arc<
        dyn Fn() -> std::collections::HashSet<crate::supertable::SuperfileUri> + Send + Sync,
    > = Arc::new(std::collections::HashSet::new);
    cache.set_pinned_fn(pinned_fn);
}

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
/// `Supertable` don't affect this reader's view. The public read
/// methods (`bm25_search`, `bm25_search_prefix`, `vector_search`,
/// `query_sql`) live on this handle; each drives its async kernel to
/// completion via the sync→async bridge ([`SupertableReader::block_on`]),
/// mirroring the way [`SupertableWriter`](crate::supertable::SupertableWriter)
/// drives `commit`.
#[derive(Clone)]
pub struct SupertableReader {
    manifest: Arc<Manifest>,
    /// Per-process tombstone-bitmap cache shared with the parent
    /// `Supertable`. Query paths read through this before
    /// returning per-superfile hits so tombstoned rows never
    /// reach callers. `None` for in-memory-only supertables.
    pub(crate) tombstone_cache: Option<Arc<crate::supertable::tombstones::SidecarCache>>,
    /// Shared inner state, held only so the reader's sync read
    /// methods can drive their async kernels on the supertable's
    /// `query_runtime` — the same `Arc<SupertableInner>` the writer
    /// holds. One `Arc::clone` per `reader()`; keeping it alive also
    /// keeps the runtime alive for the reader's lifetime, so a reader
    /// captured before its parent `Supertable` drops can still query.
    inner: Arc<SupertableInner>,
}

/// A non-owning handle to a pinned reader snapshot, held by the SQL
/// search TVFs that live inside a cached `SessionContext`.
///
/// Caching the `SessionContext` on `SupertableInner` while its TVFs
/// held a strong `Arc<SupertableReader>` formed a reference cycle
/// (`SupertableInner` → cached `SessionContext` → TVF →
/// `Arc<SupertableReader>` → `SupertableInner`), which leaked the
/// entire consumer on every reopen. `WeakReader` breaks it: it holds a
/// `Weak<SupertableInner>` plus the pinned `Arc<Manifest>` (a manifest
/// never points back at the inner, so it adds no cycle) and rebuilds
/// the strong reader on demand. The upgrade always succeeds while a
/// query is executing, because the live consumer keeps the inner alive.
#[derive(Clone)]
pub(crate) struct WeakReader {
    inner: Weak<SupertableInner>,
    manifest: Arc<Manifest>,
    tombstone_cache: Option<Arc<crate::supertable::tombstones::SidecarCache>>,
}

impl std::fmt::Debug for WeakReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeakReader").finish_non_exhaustive()
    }
}

impl WeakReader {
    /// Capture a reader's snapshot without keeping its inner alive.
    pub(crate) fn from_reader(reader: &SupertableReader) -> Self {
        Self {
            inner: Arc::downgrade(reader.inner_arc()),
            manifest: Arc::clone(reader.manifest()),
            tombstone_cache: reader.tombstone_cache.clone(),
        }
    }

    /// Reconstruct the strong pinned reader, or `None` if the owning
    /// consumer has already been dropped.
    pub(crate) fn upgrade(&self) -> Option<Arc<SupertableReader>> {
        let inner = self.inner.upgrade()?;
        Some(Arc::new(SupertableReader::from_inner_pinned(
            inner,
            Arc::clone(&self.manifest),
            self.tombstone_cache.clone(),
        )))
    }
}

impl SupertableReader {
    /// Manifest id pinned at construction. Useful for asserting
    /// reader-vs-writer visibility ordering in tests.
    pub fn manifest_id(&self) -> u64 {
        self.manifest.manifest_id
    }

    /// Sync→async bridge for this reader's public query surface.
    /// Reuses an ambient `multi_thread` runtime via `block_in_place`
    /// when present, otherwise drives on the supertable's lazily-built
    /// `query_runtime`. Same bridge the writer's `commit` uses.
    pub(crate) fn block_on<F: Future>(&self, fut: F) -> F::Output {
        bridge_on_runtime(fut, &self.inner.query_runtime())
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
    /// (skip helpers, fan-out, etc.) to read the superfile list
    /// + summaries directly.
    pub fn manifest(&self) -> &Arc<Manifest> {
        &self.manifest
    }

    /// The shared `Arc<SupertableInner>` backing this reader. Used to
    /// build a [`WeakReader`] that retains the snapshot without an
    /// owning cycle through a cached `SessionContext`. Module-private:
    /// `SupertableInner` is module-private, and the only caller is
    /// [`WeakReader::from_reader`] in this file.
    fn inner_arc(&self) -> &Arc<SupertableInner> {
        &self.inner
    }

    /// Rebuild a pinned reader from its parts. Pairs with
    /// [`WeakReader::upgrade`]: the SQL search TVFs cache a weak inner
    /// plus the pinned manifest, then reconstruct the strong reader at
    /// `call()` time (the consumer is always alive while a query runs).
    /// Module-private (takes the module-private `SupertableInner`); the
    /// only caller is [`WeakReader::upgrade`] in this file.
    fn from_inner_pinned(
        inner: Arc<SupertableInner>,
        manifest: Arc<Manifest>,
        tombstone_cache: Option<Arc<crate::supertable::tombstones::SidecarCache>>,
    ) -> Self {
        Self {
            manifest,
            tombstone_cache,
            inner,
        }
    }

    /// Per-supertable configuration for this reader's snapshot.
    pub(crate) fn options(&self) -> &Arc<SupertableOptions> {
        &self.inner.options
    }

    /// Cached `SessionContext` keyed on the manifest `Arc`, reused by
    /// [`SupertableReader::query_sql`] across queries on this snapshot.
    pub(crate) fn sql_session_cache(&self) -> &Mutex<Option<(Arc<Manifest>, SessionContext)>> {
        &self.inner.sql_session_cache
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
            subsection_offsets: None,
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

        // Superfile counts are monotonic across capture times.
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
