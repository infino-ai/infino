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

use std::{
    collections::HashSet,
    fmt,
    future::Future,
    sync::{Arc, Mutex, OnceLock, Weak, atomic::AtomicBool},
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use arrow_schema::SchemaRef;
use chrono::Utc;
use datafusion::execution::context::SessionContext;
use tokio::runtime::Runtime;

use super::{
    error::{BuildError, OpenError},
    manifest::Manifest,
    options::SupertableOptions,
};
use crate::{
    config::CompactionSettings,
    runtime_bridge::{
        bridge_on_runtime, bridge_sync_to_async, get_or_init_query_runtime,
        shutdown_query_runtime_on_drop,
    },
    storage::PrefixedStorageProvider,
    supertable::{
        ManifestLoadError, SuperfileUri, SupertableStats,
        opann::live::undrained_user_vector_entries,
        options::Consistency,
        reader_cache::disk::{DiskCacheError, skip_background_fill},
        stats::process_rss_bytes,
        tombstones::{SidecarCache, cache::DEFAULT_REFRESH_TTL},
        utils::idgen::IdGenerator,
        wal::{
            WalStore, gc,
            lease::DEFAULT_LEASE_DURATION,
            recovery::{RecoveryError, RecoveryReport, scan_and_recover},
        },
    },
};

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
    /// Single-compaction slot. Same acquire/release pattern as
    /// `writer_outstanding`. Prevents concurrent `compact()` calls
    /// within the same process from racing on seals and manifest
    /// writes. Cross-process coordination happens at the sidecar-seal
    /// level.
    pub(super) compaction_outstanding: AtomicBool,
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
    pub(super) id_generator: Mutex<IdGenerator>,
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
    pub(super) tombstone_cache: Option<Arc<SidecarCache>>,
    /// Fresh `supertable_handle_id` minted at handle
    /// construction. Used as the `lease.owner` identifier on
    /// every WAL this process drives. Not the OS PID — we need
    /// uniqueness across restarts on the same PID AND across
    /// multiple handles within one process (a process that
    /// opens five supertables holds five distinct ids). Minted
    /// via `IdGenerator::next_id()` once at create / open.
    pub(super) handle_id: crate::supertable::wal::state_doc::SupertableHandleId,
    /// Hidden sibling supertable storing vectors only, partitioned by
    /// global centroids so unfiltered search can route by nearest cell.
    pub(super) vector_index_table: Option<Arc<Supertable>>,
    /// Last time the read path checked the storage manifest pointer
    /// for freshness, under [`Consistency::BoundedStaleness`]. `None`
    /// until the first check (so the first query always refreshes).
    /// Unused for [`Consistency::Strong`] (always checks) and
    /// [`Consistency::Snapshot`] (never checks).
    pub(super) last_pointer_check: Mutex<Option<std::time::Instant>>,
    /// Back-reference to the parent user table (hidden vector-index sibling only).
    pub(super) user_table: std::sync::OnceLock<std::sync::Weak<SupertableInner>>,
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
        shutdown_query_runtime_on_drop(&mut self.query_runtime);
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
        get_or_init_query_runtime(&self.query_runtime, "supertable-query")
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
        bridge_sync_to_async(Self::create_async(options))
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

    /// Async open kernel. Sync [`Supertable::open`] bridges here.
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
        let vector_index_table = if let Some(hidden_opts) =
            build_vector_index_options(options_arc.as_ref(), Some(manifest.as_ref()), None)
        {
            let hidden_storage = hidden_opts.storage.clone().ok_or_else(|| {
                OpenError::Build(BuildError::Store(
                    "VectorIndexSuperTable requires options.storage".into(),
                ))
            })?;
            match crate::supertable::manifest::commit::read_pointer(&*hidden_storage).await {
                Ok(Some(_)) => {
                    let hidden_arc = Arc::new(hidden_opts);
                    match Manifest::load(None, hidden_storage, Some(hidden_arc.clone())).await {
                        Ok(hidden_manifest) => open_table_async(hidden_arc, hidden_manifest, None)
                            .await
                            .ok()
                            .map(Arc::new),
                        Err(e) => {
                            tracing::warn!(
                                "supertable: hidden vector-index table unavailable: {e}"
                            );
                            None
                        }
                    }
                }
                Ok(None) => create_table_async(hidden_opts, None, None)
                    .await
                    .ok()
                    .map(Arc::new),
                Err(e) => {
                    tracing::warn!("supertable: hidden vector-index table unavailable: {e}");
                    None
                }
            }
        } else {
            None
        };
        open_table_async(options_arc, manifest, vector_index_table).await
    }

    /// Async create kernel. Sync [`Supertable::create`] bridges here.
    pub(crate) async fn create_async(options: SupertableOptions) -> Result<Self, OpenError> {
        if let Some(storage) = options.storage.as_ref() {
            let probe = Arc::clone(storage);
            match crate::supertable::manifest::commit::read_pointer(&*probe).await {
                Ok(Some(_pointer)) => return Self::open_async(options).await,
                Ok(None) => {}
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
        let vector_index_storage_prefix = if options.vector_columns.is_empty() {
            None
        } else {
            Some(generate_vector_index_storage_prefix())
        };
        let vector_index_table = if let Some(ref prefix) = vector_index_storage_prefix {
            if let Some(hidden_opts) =
                build_vector_index_options(&options, None, Some(prefix.as_str()))
            {
                Some(Arc::new(
                    create_table_async(hidden_opts, None, Some(prefix.clone())).await?,
                ))
            } else {
                None
            }
        } else {
            None
        };
        create_table_async(options, vector_index_table, vector_index_storage_prefix).await
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

    test_visible! {
    fn vector_index_table(&self) -> Option<&Arc<Supertable>> {
        self.inner.vector_index_table.as_ref()
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
                        *last = Some(Instant::now());
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
    pub fn schema(&self) -> SchemaRef {
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

    test_visible! {
    /// Route every accumulated INCOMING IVF superfile into per-cell delta
    /// superfiles. Not part of the public API — [`Supertable::optimize`]
    /// calls this on the hidden index before compact; tests and benches may
    /// invoke it directly on the hidden table handle.
    fn drain(&self) -> Result<(), BuildError> {
        self.block_on_query(super::writer::drain_incoming_to_cells(Arc::clone(
            &self.inner,
        )))
    }
    }

    test_visible! {
    /// Build or load the hidden index OPANN routing tree so the first vector
    /// query on this handle does not pay genesis / page-load on the hot path.
    fn prewarm_hidden_opann_tree(&self) -> Result<(), super::error::QueryError> {
        let Some(hidden) = self.inner.vector_index_table.as_ref() else {
            return Ok(());
        };
        self.block_on_query(async {
            hidden
                .reader()
                .ensure_opann_resident_tree()
                .await
                .map(|_| ())
        })
    }
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
    pub fn wait_until_warm(&self, timeout: Duration) -> Result<(), DiskCacheError> {
        let Some(cache) = self.inner.options.disk_cache.as_ref() else {
            return Ok(());
        };
        if skip_background_fill() {
            return Ok(());
        }
        let cache = Arc::clone(cache);
        let manifest = self.inner.manifest.load_full();
        let hidden_manifest = self
            .inner
            .vector_index_table
            .as_ref()
            .map(|hidden| hidden.inner.manifest.load_full());
        self.block_on_query(async move {
            for entry in manifest.superfiles.iter() {
                if cache.is_cached(&entry.uri) {
                    cache.wait_until_mmap_promoted(&entry.uri, timeout).await?;
                }
            }
            if let Some(hidden) = hidden_manifest {
                for entry in hidden.superfiles.iter() {
                    if cache.is_cached(&entry.uri) {
                        cache.wait_until_mmap_promoted(&entry.uri, timeout).await?;
                    }
                }
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
    pub(crate) async fn run_recovery_sweep_once(&self) -> Result<RecoveryReport, RecoveryError> {
        scan_and_recover(self, self.inner.handle_id, DEFAULT_LEASE_DURATION).await
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
    pub(crate) async fn run_gc_sweep_once(&self) -> Result<gc::GcReport, gc::GcError> {
        gc::run_sweep(
            self,
            Utc::now(),
            gc::DEFAULT_WAL_GRACE,
            gc::DEFAULT_SIDECAR_GRACE,
        )
        .await
    }

    /// Observability snapshot of the supertable's load.
    /// Cheap to call: one RSS syscall + an `ArcSwap::load` + a couple of
    /// length reads on the in-memory manifest. See
    /// [`crate::supertable::SupertableStats`] for the field-level contract.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn stats(&self) -> SupertableStats {
        let manifest = self.inner.manifest.load();
        let n_manifest_parts = manifest.get_num_parts();
        let cache = self.inner.options.disk_cache.as_ref();
        let mmap_resident_bytes = cache.map(|c| c.current_mmap_size_bytes());
        // One `cache.stats()` call covers four fields. Cache
        // counters are atomic loads, so the snapshot is
        // self-consistent for each counter but not coherent
        // across counters under heavy concurrent activity —
        // adequate for observability.
        let cache_snapshot = cache.map(|c| c.stats());
        SupertableStats {
            manifest_id: manifest.get_manifest_id(),
            n_superfiles: manifest.get_all_superfiles().len(),
            n_manifest_parts,
            n_manifest_parts_loaded: manifest.get_num_parts_loaded(),
            process_rss_bytes: process_rss_bytes(),
            mmap_resident_bytes,
            memory_budget_bytes: self.inner.options.memory_budget_bytes,
            n_cold_fetches: cache_snapshot.as_ref().map(|s| s.n_cold_fetches),
            n_cache_evictions: cache_snapshot.as_ref().map(|s| s.n_evictions),
            n_cache_madvise_calls: cache_snapshot.as_ref().map(|s| s.n_madvise_calls),
            n_cache_entries: cache_snapshot.as_ref().map(|s| s.n_entries),
        }
    }

    #[cfg(any(test, feature = "test-helpers"))]
    test_visible! {
    /// Force-open every user + hidden vector superfile reader on the
    /// pinned snapshot — the cold-open phase before a timed search.
    /// Hidden IVF superfiles use their prefixed storage provider.
    fn open_all_superfiles(&self) {
        let reader = self.reader();
        let manifest = reader.manifest();
        let store = manifest.options.store.clone();
        let disk_cache = manifest.options.disk_cache.clone();
        let user_storage = manifest
            .options
            .storage
            .clone()
            .expect("open_all_superfiles: user table needs storage");
        let mut targets: Vec<(
            crate::supertable::manifest::SuperfileUri,
            Option<crate::supertable::manifest::SubsectionOffsets>,
            std::sync::Arc<dyn crate::storage::StorageProvider>,
        )> = manifest
            .superfiles
            .iter()
            .map(|e| {
                (
                    e.uri,
                    e.subsection_offsets.clone(),
                    std::sync::Arc::clone(&user_storage),
                )
            })
            .collect();
        if let Some(hidden) = self.inner.vector_index_table.as_ref() {
            let hidden_manifest = hidden.inner.manifest.load_full();
            let hidden_storage = hidden_manifest
                .options
                .storage
                .clone()
                .expect("open_all_superfiles: hidden vector index needs storage");
            for entry in hidden_manifest.superfiles.iter() {
                targets.push((
                    entry.uri,
                    entry.subsection_offsets.clone(),
                    std::sync::Arc::clone(&hidden_storage),
                ));
            }
        }
        self.block_on_query(async move {
            let handles: Vec<_> = targets
                .into_iter()
                .map(|(uri, offsets, storage)| {
                    let store = store.clone();
                    let disk_cache = disk_cache.clone();
                    tokio::spawn(async move {
                        crate::supertable::query::superfile_reader::superfile_reader(
                            &store,
                            disk_cache.as_ref(),
                            Some(&storage),
                            &uri,
                            offsets.as_ref(),
                        )
                        .await
                    })
                })
                .collect();
            for h in handles {
                h.await
                    .expect("open_all_superfiles: join superfile open task")
                    .expect("open_all_superfiles: open superfile readers");
            }
            Ok::<(), crate::supertable::reader_cache::disk::DiskCacheError>(())
        })
        .expect("open_all_superfiles");
    }
    }

    #[cfg(any(test, feature = "test-helpers"))]
    test_visible! {
    /// Diagnostic: `(total_hidden_superfiles, max_superfiles_in_one_cell)` for
    /// the hidden vector-index table, or `None` when there is no hidden table.
    /// Used by benches to observe how compacted the hidden cell index is.
    fn hidden_vector_superfile_stats(&self) -> Option<(usize, usize)> {
        let hidden = self.inner.vector_index_table.as_ref()?;
        let reader = hidden.reader();
        let manifest = reader.manifest();
        let mut by_cell: std::collections::HashMap<Vec<u8>, usize> =
            std::collections::HashMap::new();
        for entry in manifest.superfiles.iter() {
            *by_cell.entry(entry.partition_key.clone()).or_default() += 1;
        }
        let total = manifest.superfiles.len();
        let max_per_cell = by_cell.values().copied().max().unwrap_or(0);
        Some((total, max_per_cell))
    }
    }

    #[cfg(any(test, feature = "test-helpers"))]
    test_visible! {
    /// User superfiles not yet routed into hidden per-cell IVF (bench / test diagnostics).
    fn undrained_user_superfile_count(&self) -> Option<usize> {
        let hidden = self.inner.vector_index_table.as_ref()?;
        let vec_col = hidden.inner.options.vector_columns.first()?;
        let user_manifest = self.inner.manifest.load_full();
        let hidden_manifest = hidden.inner.manifest.load();
        Some(
            undrained_user_vector_entries(
                &user_manifest,
                hidden_manifest.opann_routing(),
                &vec_col.column,
            )
            .len(),
        )
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
/// Initial number of global vector-index cells. `optimize()` splits any cell
/// past the 50k overflow cap into more cells as the corpus grows, so the cell
/// count grows with the data. The OPANN routing tree routes to cells (cheap,
/// sublinear); within each routed cell the query selects clusters by their
/// resident centroids and range-GETs only those.
pub(crate) const GLOBAL_VECTOR_CELL_COUNT: usize = 64;

/// Reserved VectorCell partition id for the hidden index's "incoming" append
/// region. Each commit writes one IVF superfile under this sentinel
/// partition holding that whole batch (all cells mixed, unsorted). Queries
/// route via the OPANN tree to admitted cluster leaves. Call [`Supertable::optimize`]
/// (or the internal [`Supertable::drain`] hook on the hidden table) to route
/// INCOMING rows into per-cell IVF superfiles.
/// `u32::MAX` is out of the
/// valid cell range `0..n_cent`, so it never collides with a real cell.
pub(crate) const INCOMING_VECTOR_CELL: u32 = u32::MAX;

/// Lloyd iterations when folding per-superfile cluster centroids into the
/// global cell grid at open/create time.
pub(crate) const GLOBAL_VECTOR_KMEANS_ITERS: usize = 8;

/// Fixed PRNG seed for global centroid training.
pub(crate) const GLOBAL_VECTOR_KMEANS_SEED: u64 = 0x51ED_2A11;

/// Eager-load every part of a hidden vector-index manifest entry whose part
/// count is at or below this — the hidden index's per-cell superfiles are small
/// and almost always touched together, so the lazy-open round-trips don't pay off.
const HIDDEN_VECTOR_INDEX_EAGER_LOAD_THRESHOLD: u32 = 128;

/// Hidden vector-index compaction: target packed per-cell superfile size. Smaller
/// than the user table's default — cell superfiles are many and individually small.
pub(crate) const HIDDEN_VECTOR_INDEX_TARGET_SUPERFILE_SIZE_MB: u64 = 8;

/// Hidden vector-index compaction: merge a superfile once it drops below this
/// fraction (percent) of the target size.
pub(crate) const HIDDEN_VECTOR_INDEX_MIN_FILL_PERCENT: u8 = 40;

/// Hidden vector-index compaction: per-pass memory ceiling.
pub(crate) const HIDDEN_VECTOR_INDEX_MAX_MEMORY_MB: u64 = 512;

/// True for the derived hidden vector-index sibling (VectorCell routing, no FTS).
pub(crate) fn is_hidden_vector_index_table(opts: &SupertableOptions) -> bool {
    !opts.vector_columns.is_empty()
        && opts.fts_columns.is_empty()
        && matches!(
            opts.partition_strategy,
            Some(crate::supertable::manifest::list::PartitionStrategy::VectorCell { .. })
        )
}

pub(crate) fn legacy_vector_index_storage_prefix() -> &'static str {
    "_vector_index"
}

fn generate_vector_index_storage_prefix() -> String {
    format!("_infino_{}_vector_index", uuid::Uuid::new_v4())
}

fn resolve_vector_index_storage_prefix(
    user_opts: &SupertableOptions,
    user_manifest: Option<&super::manifest::Manifest>,
    create_prefix: Option<&str>,
) -> Option<String> {
    if user_opts.vector_columns.is_empty() {
        return None;
    }
    if let Some(prefix) = create_prefix {
        return Some(prefix.to_string());
    }
    if let Some(manifest) = user_manifest
        && let Some(prefix) = manifest.vector_index_storage_prefix()
    {
        return Some(prefix.to_string());
    }
    Some(legacy_vector_index_storage_prefix().to_string())
}

fn build_vector_index_options(
    user_opts: &SupertableOptions,
    user_manifest: Option<&super::manifest::Manifest>,
    create_prefix: Option<&str>,
) -> Option<SupertableOptions> {
    let storage_prefix =
        resolve_vector_index_storage_prefix(user_opts, user_manifest, create_prefix)?;
    let storage = user_opts.storage.as_ref()?;
    let sub_storage: Arc<dyn crate::storage::StorageProvider> = Arc::new(
        PrefixedStorageProvider::new(Arc::clone(storage), storage_prefix.as_str()),
    );
    let mut fields: Vec<arrow_schema::FieldRef> = Vec::new();
    for vc in &user_opts.vector_columns {
        let item_field = Arc::new(arrow_schema::Field::new(
            "item",
            arrow_schema::DataType::Float32,
            true,
        ));
        fields.push(Arc::new(arrow_schema::Field::new(
            &vc.column,
            arrow_schema::DataType::FixedSizeList(item_field, vc.dim as i32),
            false,
        )));
    }
    let hidden_schema = Arc::new(arrow_schema::Schema::new(fields));
    // Hidden index reads Sq8+ε rerank rows without fp32 reconstruction.
    let hidden_vector_columns: Vec<crate::superfile::builder::VectorConfig> = user_opts
        .vector_columns
        .iter()
        .map(|vc| crate::superfile::builder::VectorConfig {
            rerank_codec: crate::superfile::vector::rerank_codec::RerankCodec::Sq8Residual,
            ..vc.clone()
        })
        .collect();
    let mut hidden_opts = SupertableOptions::new(
        hidden_schema,
        vec![],
        hidden_vector_columns,
        user_opts.tokenizer.clone(),
    )
    .ok()?;
    hidden_opts = hidden_opts
        .with_storage(Arc::clone(&sub_storage))
        .with_vector_layout(crate::superfile::vector::layout::VectorLayout::Ivf)
        .with_eager_load_threshold(HIDDEN_VECTOR_INDEX_EAGER_LOAD_THRESHOLD)
        .with_compaction(CompactionSettings {
            target_superfile_size_mb: HIDDEN_VECTOR_INDEX_TARGET_SUPERFILE_SIZE_MB,
            min_fill_percent: HIDDEN_VECTOR_INDEX_MIN_FILL_PERCENT,
            max_memory_mb: HIDDEN_VECTOR_INDEX_MAX_MEMORY_MB,
        });
    if let Some(cache) = user_opts.disk_cache.as_ref() {
        hidden_opts = hidden_opts.with_disk_cache(Arc::clone(cache));
    }
    Some(hidden_opts)
}

/// Build one supertable handle. Leaf — never creates a hidden sibling.
async fn build_handle(
    options: Arc<SupertableOptions>,
    manifest: Arc<Manifest>,
    vector_index_table: Option<Arc<Supertable>>,
) -> Result<Supertable, OpenError> {
    let tombstone_cache = build_tombstone_cache(&options);
    let id_generator = crate::supertable::utils::idgen::IdGenerator::new();
    let handle_id = crate::supertable::wal::state_doc::SupertableHandleId(id_generator.next_id());
    let inner = Arc::new(SupertableInner {
        options,
        manifest: ArcSwap::new(manifest),
        writer_outstanding: AtomicBool::new(false),
        compaction_outstanding: AtomicBool::new(false),
        id_generator: Mutex::new(id_generator),
        query_runtime: OnceLock::new(),
        sql_session_cache: Mutex::new(None),
        tombstone_cache,
        handle_id,
        vector_index_table,
        // Open already loaded the manifest from a fresh pointer read, so the
        // BoundedStaleness clock starts now — the first query is within the
        // window and skips a redundant pointer GET (the cold-search wave). Both
        // the user table and the hidden index go through here, so neither pays a
        // first-query freshness round-trip.
        last_pointer_check: Mutex::new(Some(Instant::now())),
        user_table: std::sync::OnceLock::new(),
    });
    install_disk_cache_pinning(&inner);
    let st = Supertable { inner: Arc::clone(&inner) };
    if let Some(vit) = st.inner.vector_index_table.as_ref() {
        let _ = vit
            .inner()
            .user_table
            .set(Arc::downgrade(&st.inner));
    }
    if st.inner.options.storage.is_some() {
        let _ = st.run_recovery_sweep_once().await;
        let _ = st.run_gc_sweep_once().await;
        // Prewarm the OPANN routing tree at open so the first cold *search*
        // doesn't pay its descent-page load. Only the hidden vector-index
        // manifest carries a routing tree (a no-op on the user table); the open
        // already does I/O (the sweeps above), so this folds the one-time tree
        // load into the open-latency window instead of the search critical path.
        // After the sweeps so the manifest is final — a recovery swap would
        // otherwise warm a tree that's about to be replaced.
        let m = st.inner.manifest.load_full();
        if m.opann_routing().is_some() {
            if let Err(e) = m.opann_resident_tree().await {
                tracing::warn!("supertable: OPANN routing-tree prewarm at open failed: {e}");
            }
        }
        if let Some(vit) = st.inner.vector_index_table.as_ref() {
            let vit_reader = vit.reader();
            let hm = vit_reader.manifest();
            // Post-drain: warm persisted routing pages. Pre-drain: incoming centroids
            // live in the user manifest summaries — no tree to warm.
            let warm_hidden = hm.opann_routing().is_some();
            if warm_hidden {
                if let Err(e) = vit_reader.ensure_opann_resident_tree().await {
                    tracing::warn!(
                        "supertable: hidden-index OPANN routing-tree prewarm at open failed: {e}"
                    );
                }
            }
        }
    }
    Ok(st)
}

/// Create one supertable handle (empty manifest). Leaf — never creates a sibling.
async fn create_table_async(
    options: SupertableOptions,
    vector_index_table: Option<Arc<Supertable>>,
    vector_index_storage_prefix: Option<String>,
) -> Result<Supertable, OpenError> {
    let options = Arc::new(options);
    let manifest = Arc::new(Manifest::empty_with_vector_index_prefix(
        options.clone(),
        vector_index_storage_prefix,
    ));
    build_handle(options, manifest, vector_index_table).await
}

/// Open one supertable handle from a loaded manifest. Leaf — never creates a sibling.
async fn open_table_async(
    options: Arc<SupertableOptions>,
    manifest: Arc<Manifest>,
    vector_index_table: Option<Arc<Supertable>>,
) -> Result<Supertable, OpenError> {
    build_handle(options, manifest, vector_index_table).await
}

fn install_disk_cache_pinning(inner: &Arc<SupertableInner>) {
    let cache = match inner.options.disk_cache.as_ref() {
        Some(c) => c,
        None => return,
    };
    let pinned_fn: Arc<dyn Fn() -> HashSet<SuperfileUri> + Send + Sync> = Arc::new(HashSet::new);
    cache.set_pinned_fn(pinned_fn);
}

/// Build the tombstone-sidecar cache when storage is attached.
/// Returns `None` for in-memory-only supertables — no sidecars
/// can exist there, so the query paths skip the filter hook
/// entirely.
fn build_tombstone_cache(options: &Arc<SupertableOptions>) -> Option<Arc<SidecarCache>> {
    let storage = options.storage.as_ref()?.clone();
    let wal_store = WalStore::new(storage);
    Some(Arc::new(SidecarCache::new(wal_store, DEFAULT_REFRESH_TTL)))
}

impl fmt::Debug for Supertable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    pub(crate) tombstone_cache: Option<Arc<SidecarCache>>,
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
    tombstone_cache: Option<Arc<SidecarCache>>,
}

impl fmt::Debug for WeakReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        tombstone_cache: Option<Arc<SidecarCache>>,
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

    pub(crate) fn vector_index_table(&self) -> Option<&Arc<Supertable>> {
        self.inner.vector_index_table.as_ref()
    }

    /// Parent user table for the hidden vector-index sibling (if any).
    pub(crate) fn hidden_parent_user_manifest(&self) -> Option<Arc<Manifest>> {
        self.inner
            .user_table
            .get()
            .and_then(|w| w.upgrade())
            .map(|u| u.manifest.load_full())
    }

    /// Load persisted OPANN pages when post-drain; pre-drain incoming centroids
    /// are built live from the user manifest at query time.
    pub(crate) async fn ensure_opann_resident_tree(
        &self,
    ) -> Result<Option<Arc<crate::supertable::opann::paged::ResidentPageSource>>, crate::supertable::error::QueryError>
    {
        self.manifest()
            .opann_resident_tree()
            .await
            .map_err(|e| crate::supertable::error::QueryError::Store(format!("opann tree load: {e}")))
    }

    /// Resident routing metadata only — persisted page graph + live incoming
    /// genesis tree. Does not open or fetch vector IVF blobs. Test-only warming
    /// helper (no production caller); `pub` under `test-helpers` so the
    /// integration tests can prime the cold-GET measurement.
    #[cfg(feature = "test-helpers")]
    pub async fn warm_opann_routing_metadata(
        &self,
    ) -> Result<(), crate::supertable::error::QueryError> {
        // Only the persisted base tree's pages need warming (post-drain). The
        // incoming buffer is a manifest-resident SIMD scan — already in memory,
        // no GET to prime.
        let _ = self.ensure_opann_resident_tree().await?;
        Ok(())
    }
}

impl fmt::Debug for SupertableReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SupertableReader")
            .field("manifest_id", &self.manifest.manifest_id)
            .field("n_superfiles", &self.manifest.superfiles.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow_schema::{DataType, Field, Schema};
    use tempfile::TempDir;
    use uuid::Uuid;

    use super::*;
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::builder::FtsConfig,
        supertable::{
            manifest::{SuperfileEntry, SuperfileUri},
            options::Consistency,
        },
        test_helpers::default_tokenizer,
    };

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]))
    }

    fn opts() -> SupertableOptions {
        let tk = default_tokenizer();
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
            arrival_ordinal: 0,
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs,
            id_min: 0,
            id_max: n_docs.saturating_sub(1) as i128,
            scalar_stats: HashMap::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            vector_layout: crate::superfile::vector::layout::VectorLayout::Ivf,
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

    #[test]
    fn schema_returns_user_schema_without_injected_id() {
        let st = Supertable::create(opts()).expect("create");
        let sch = st.schema();
        // The user-facing schema is exactly the column the test fixture
        // declared — the auto-injected `_id` is not part of it.
        assert_eq!(sch.fields().len(), 1);
        assert_eq!(sch.field(0).name(), "title");
    }

    #[test]
    fn manifest_accessor_matches_reader_manifest_id() {
        let st = Supertable::create(opts()).expect("create");
        assert_eq!(st.manifest_id(), 0);
        publish_appended(&st, vec![entry(3)]);
        // The handle-level `manifest_id` advances with the swap, and a
        // fresh reader pins the same value.
        assert_eq!(st.manifest_id(), 1);
        assert_eq!(st.reader().manifest_id(), 1);
    }

    #[test]
    fn handle_id_is_stable_for_a_handle_and_distinct_across_handles() {
        let st1 = Supertable::create(opts()).expect("create");
        let st2 = Supertable::create(opts()).expect("create");
        // Stable within one handle (and its clones).
        assert_eq!(st1.handle_id(), st1.clone().handle_id());
        // Distinct across independently-created handles.
        assert_ne!(st1.handle_id(), st2.handle_id());
    }

    #[test]
    fn query_runtime_is_lazily_built_and_cached() {
        let st = Supertable::create(opts()).expect("create");
        let rt1 = st.query_runtime();
        let rt2 = st.query_runtime();
        // Second call returns the same cached runtime, not a fresh one.
        assert!(Arc::ptr_eq(&rt1, &rt2));
    }

    #[test]
    fn block_on_query_drives_a_future_to_completion() {
        let st = Supertable::create(opts()).expect("create");
        let out = st.block_on_query(async { 7_u32 + 35 });
        assert_eq!(out, 42);
    }

    #[test]
    fn stats_reports_in_memory_snapshot() {
        let st = Supertable::create(opts()).expect("create");
        publish_appended(&st, vec![entry(10), entry(20)]);
        let s = st.stats();
        assert_eq!(s.manifest_id, 1);
        assert_eq!(s.n_superfiles, 2);
        // In-memory supertable has no manifest list / disk cache.
        assert_eq!(s.n_manifest_parts, 0);
        assert_eq!(s.mmap_resident_bytes, None);
        assert_eq!(s.n_cold_fetches, None);
    }

    #[test]
    fn wait_until_warm_is_noop_without_disk_cache() {
        let st = Supertable::create(opts()).expect("create");
        // No disk cache attached → returns Ok immediately.
        st.wait_until_warm(Duration::from_millis(1))
            .expect("warm no-op");
    }

    #[test]
    fn debug_cached_session_populates_the_session_cache() {
        let st = Supertable::create(opts()).expect("create");
        // Building the diagnostic session forces a SessionContext to be
        // built and cached on the inner.
        let _ctx = st.__debug_cached_session();
        let guard = st
            .sql_session_cache()
            .lock()
            .expect("sql_session_cache mutex");
        assert!(guard.is_some(), "session cache populated after warm-up");
    }

    #[test]
    fn weak_reader_round_trips_and_debug() {
        let st = Supertable::create(opts()).expect("create");
        publish_appended(&st, vec![entry(4)]);
        let reader = st.reader();
        let weak = WeakReader::from_reader(&reader);
        // Debug is non-exhaustive but must not explode.
        assert!(format!("{weak:?}").contains("WeakReader"));
        // While the parent + reader are alive, upgrade succeeds and
        // observes the same pinned snapshot.
        let upgraded = weak.upgrade().expect("upgrade while inner alive");
        assert_eq!(upgraded.manifest_id(), reader.manifest_id());
        assert_eq!(upgraded.n_superfiles(), 1);
    }

    #[test]
    fn weak_reader_upgrade_fails_after_inner_dropped() {
        let weak = {
            let st = Supertable::create(opts()).expect("create");
            let reader = st.reader();
            let weak = WeakReader::from_reader(&reader);
            drop(reader);
            drop(st);
            weak
        };
        // The owning inner is gone, so upgrade yields None.
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn reader_options_match_handle_options() {
        let st = Supertable::create(opts()).expect("create");
        let r = st.reader();
        // The reader's options accessor reaches the same validated
        // options the handle exposes.
        assert_eq!(r.options().id_column, st.options().id_column);
        assert_eq!(r.options().fts_columns.len(), 1);
    }

    #[test]
    fn hidden_vector_index_visible_after_commit() {
        use std::sync::Arc;

        use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray};
        use arrow_schema::{DataType, Field, Schema};

        use crate::superfile::{
            builder::{FtsConfig, VectorConfig},
            reader::VectorSearchOptions,
            vector::{distance::Metric, rerank_codec::RerankCodec},
        };

        let dim = 16usize;
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(item_field.clone(), dim as i32),
                false,
            ),
        ]));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(
            schema.clone(),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool);
        let st = Supertable::create(options).expect("create");
        assert!(
            st.reader().vector_index_table().is_some(),
            "vector columns + storage must create hidden index sibling"
        );

        let titles = LargeStringArray::from(vec!["a", "b", "c"]);
        let flat = Float32Array::from(vec![1.0f32; 3 * dim]);
        let fsl = FixedSizeListArray::new(item_field, dim as i32, Arc::new(flat), None);
        let batch = arrow_array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(titles) as Arc<dyn Array>,
                Arc::new(fsl) as Arc<dyn Array>,
            ],
        )
        .expect("batch");

        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        w.commit().expect("commit");

        assert!(st.reader().n_superfiles() > 0);
        let reader = st.reader();
        let hidden = reader.vector_index_table().expect("hidden index");
        assert!(
            hidden.reader().n_superfiles() > 0,
            "dual-write must land in hidden table before commit returns"
        );

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let hits = st
            .reader()
            .vector_hits("emb", &q, 3, VectorSearchOptions::new(), None)
            .expect("vector search");
        assert!(!hits.is_empty(), "search should find committed vectors");
    }

    /// The hidden IVF superfiles must be made *resident* in the
    /// disk cache by a vector query, and a warm re-query must serve from
    /// that resident mmap without re-fetching from storage.
    ///
    /// Regression guard: the hidden-index read path used to `get_range`
    /// straight from object storage, bypassing the cache entirely — so the
    /// hidden superfiles were never resident and every (incl. warm) vector
    /// query paid an object-store round-trip. The fix routes the read
    /// through `reader_synchronous_with_storage`, cold-fetching through the
    /// hidden table's *prefixed* storage (the shared cache is keyed to the
    /// user storage and can't resolve the hidden prefix on its own).
    #[test]
    fn hidden_ivf_superfiles_become_resident_in_cache() {
        use std::{collections::HashSet, sync::Arc};

        use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray};
        use arrow_schema::{DataType, Field, Schema};

        use crate::{
            superfile::{
                builder::{FtsConfig, VectorConfig},
                reader::VectorSearchOptions,
                vector::{distance::Metric, rerank_codec::RerankCodec},
            },
            supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
        };

        let dim = 16usize;
        // A few hundred vectors across several cells. Hidden IVF
        // superfiles are never inlined into the manifest open_blob, so the
        // query reads each probed cell's vec blob from storage through the
        // disk cache regardless of size.
        let n_rows = 512usize;
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(item_field.clone(), dim as i32),
                false,
            ),
        ]));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );

        let storage_dir = TempDir::new().expect("storage tempdir");
        let cache_dir = TempDir::new().expect("cache tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));

        let make_options = || {
            SupertableOptions::new(
                schema.clone(),
                vec![FtsConfig {
                    column: "title".into(),
                }],
                vec![VectorConfig {
                    column: "emb".into(),
                    dim,
                    n_cent: 4,
                    rot_seed: 7,
                    metric: Metric::Cosine,
                    rerank_codec: RerankCodec::Fp32,
                }],
                Some(crate::test_helpers::default_tokenizer()),
            )
            .expect("valid options")
            .with_storage(Arc::clone(&storage))
        };

        // ---- Producer: create + commit, then drop. The producer's own
        // post-commit cache pre-population is irrelevant here — we test a
        // *fresh* consumer process (cold cache), as on a real deployment.
        {
            let producer =
                Supertable::create(make_options().with_writer_pool(pool)).expect("create");

            // Diverse vectors so the hidden IVF index has real content.
            let titles =
                LargeStringArray::from((0..n_rows).map(|i| format!("doc {i}")).collect::<Vec<_>>());
            let mut flat = Vec::<f32>::with_capacity(n_rows * dim);
            for i in 0..n_rows {
                for d in 0..dim {
                    flat.push(if d == i % dim { 1.0 } else { 0.0 });
                }
            }
            let fsl = FixedSizeListArray::new(
                item_field,
                dim as i32,
                Arc::new(Float32Array::from(flat)),
                None,
            );
            let batch = arrow_array::RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(titles) as Arc<dyn Array>,
                    Arc::new(fsl) as Arc<dyn Array>,
                ],
            )
            .expect("batch");
            let mut w = producer.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
            if let Some(hidden) = producer.reader().vector_index_table() {
                hidden.drain().expect("route hidden incoming into manifest cells");
            }
        }

        // ---- Consumer: open fresh with a brand-new empty disk cache,
        // keyed (as in production) to the *user* storage. The hidden index
        // lives behind a prefixed provider over the same storage and shares
        // this cache instance.
        let cfg = DiskCacheConfig {
            cache_root: cache_dir.path().to_path_buf(),
            disk_budget_bytes: 1 << 30,
            cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
            cold_fetch_streams: 4,
            cold_fetch_chunk_bytes: 1 << 20,
            mmap_cold_threshold_secs: 0,
            mmap_sweep_interval_secs: 0,
            eviction: Box::new(LruPolicy::new()),
            verify_crc_on_open: true,
            ..Default::default()
        };
        let pinned_fn: Arc<dyn Fn() -> HashSet<SuperfileUri> + Send + Sync> =
            Arc::new(HashSet::new);
        let cache = DiskCacheStore::new(Arc::clone(&storage), cfg, pinned_fn).expect("cache");

        let st =
            Supertable::open(make_options().with_disk_cache(Arc::clone(&cache))).expect("open");

        // Collect the hidden IVF superfile URIs.
        let reader = st.reader();
        let hidden = reader.vector_index_table().expect("hidden index");
        let hidden_uris: Vec<SuperfileUri> = hidden
            .reader()
            .manifest()
            .superfiles
            .iter()
            .map(|e| e.uri)
            .collect();
        assert!(
            !hidden_uris.is_empty(),
            "hidden IVF index must have superfiles after commit"
        );

        // Cold: none of the hidden superfiles are resident yet.
        for uri in &hidden_uris {
            assert!(
                !cache.is_cached(uri),
                "hidden superfile {uri:?} unexpectedly resident before any query"
            );
        }

        // First vector query routes through the hidden IVF index.
        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let hits = st
            .reader()
            .vector_hits("emb", &q, 5, VectorSearchOptions::new(), None)
            .expect("vector search");
        assert!(!hits.is_empty(), "search should find committed vectors");

        // Every probed hidden IVF superfile must now be resident
        // (mmap-backed), proving the read went through the disk cache via
        // the hidden prefixed storage — not a bare object-store get_range.
        let resident: Vec<&SuperfileUri> =
            hidden_uris.iter().filter(|u| cache.is_cached(u)).collect();
        assert!(
            !resident.is_empty(),
            "vector query must make at least one hidden IVF superfile \
             resident in the cache; none of {hidden_uris:?} are cached"
        );
        for uri in &resident {
            assert!(
                cache.is_cached(uri),
                "resident hidden IVF superfile {uri:?} must be in disk cache"
            );
        }

        // Warm re-query: the resident superfiles serve locally — no new
        // cold-fetch. This is the warm-latency regression guard.
        let cold_before = cache.stats().n_cold_fetches;
        let hits2 = st
            .reader()
            .vector_hits("emb", &q, 5, VectorSearchOptions::new(), None)
            .expect("warm vector search");
        assert!(!hits2.is_empty());
        let cold_after = cache.stats().n_cold_fetches;
        assert_eq!(
            cold_before, cold_after,
            "warm vector query must hit the resident cache; cold-fetches grew \
             from {cold_before} to {cold_after}"
        );
    }

    /// A storage-backed handle under `Consistency::Strong` drives
    /// `ensure_fresh`'s Strong arm, which calls `refresh`. With no
    /// commit yet there is no manifest pointer, so `refresh` reports
    /// "nothing newer" and the snapshot stays at the empty manifest.
    #[test]
    fn hidden_ivf_append_only_emits_multiple_files_per_cell() {
        use std::{collections::HashMap, sync::Arc};

        use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray};
        use arrow_schema::{DataType, Field, Schema};

        use crate::superfile::{
            builder::{FtsConfig, VectorConfig},
            vector::{distance::Metric, rerank_codec::RerankCodec},
        };

        let dim = 16usize;
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(item_field.clone(), dim as i32),
                false,
            ),
        ]));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(
            schema.clone(),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool);
        let st = Supertable::create(options).expect("create");

        for commit in 0..4 {
            let titles = LargeStringArray::from(vec![format!("doc-{commit}")]);
            let flat = Float32Array::from(vec![1.0f32; dim]);
            let fsl = FixedSizeListArray::new(item_field.clone(), dim as i32, Arc::new(flat), None);
            let batch = arrow_array::RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(titles) as Arc<dyn Array>,
                    Arc::new(fsl) as Arc<dyn Array>,
                ],
            )
            .expect("batch");

            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden vector index")
            .clone();
        let hidden_reader = hidden.reader();
        let hidden_manifest = hidden_reader.manifest();
        let mut by_cell = HashMap::<Vec<u8>, usize>::new();
        for entry in hidden_manifest.superfiles.iter() {
            *by_cell.entry(entry.partition_key.clone()).or_default() += 1;
        }
        let max_visible = by_cell.values().copied().max().unwrap_or(0);
        assert!(
            max_visible >= 2,
            "append-only hidden path should emit multiple visible files per cell, got {max_visible}"
        );
    }

    #[test]
    fn hidden_ivf_compaction_collapses_per_cell() {
        use std::{collections::HashMap, sync::Arc};

        use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray};
        use arrow_schema::{DataType, Field, Schema};

        use crate::{
            config::CompactionSettings,
            superfile::{
                builder::{FtsConfig, VectorConfig},
                vector::{distance::Metric, layout::VectorLayout, rerank_codec::RerankCodec},
            },
        };

        let dim = 128usize;
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(item_field.clone(), dim as i32),
                false,
            ),
        ]));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(
            schema.clone(),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool);
        let st = Supertable::create(options).expect("create");

        let rows_per_commit = 8usize;
        let commits_per_drain = 5usize;
        for commit in 0..10 {
            let titles = LargeStringArray::from(
                (0..rows_per_commit)
                    .map(|row| format!("doc-{commit}-{row}"))
                    .collect::<Vec<_>>(),
            );
            let flat = Float32Array::from(vec![1.0f32; rows_per_commit * dim]);
            let fsl = FixedSizeListArray::new(item_field.clone(), dim as i32, Arc::new(flat), None);
            let batch = arrow_array::RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(titles) as Arc<dyn Array>,
                    Arc::new(fsl) as Arc<dyn Array>,
                ],
            )
            .expect("batch");

            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
            if (commit + 1) % commits_per_drain == 0 {
                st.reader()
                    .vector_index_table()
                    .expect("hidden vector index")
                    .drain()
                    .expect("route accumulated INCOMING into cells");
            }
        }

        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden vector index")
            .clone();
        let count_by_cell = |manifest: &crate::supertable::manifest::Manifest| -> usize {
            let mut by_cell = HashMap::<Vec<u8>, usize>::new();
            for entry in manifest.superfiles.iter() {
                if entry.vector_layout != VectorLayout::Ivf {
                    continue;
                }
                *by_cell.entry(entry.partition_key.clone()).or_default() += 1;
            }
            by_cell.values().copied().max().unwrap_or(0)
        };
        let before = count_by_cell(hidden.reader().manifest());
        assert!(
            before >= 2,
            "need multiple append-only superfiles per cell before compaction, got {before}"
        );

        let cfg = CompactionSettings {
            target_superfile_size_mb: 1,
            min_fill_percent: 1,
            ..CompactionSettings::default()
        };
        hidden.compact(&cfg).expect("hidden compact");

        let after_reader = hidden.reader();
        let after_manifest = after_reader.manifest();
        let after = count_by_cell(after_manifest);
        assert!(
            after < before,
            "compaction should collapse per-cell superfiles: before={before} after={after}"
        );
        for entry in &after_manifest.superfiles {
            assert_eq!(
                entry.vector_layout,
                crate::superfile::vector::layout::VectorLayout::Ivf
            );
            assert!(
                entry
                    .subsection_offsets
                    .as_ref()
                    .and_then(|o| o.vec)
                    .is_some(),
                "compacted hidden IVF entry {:?} missing vec subsection",
                entry.uri
            );
        }
        let hits = st
            .reader()
            .vector_hits(
                "emb",
                &vec![1.0f32; dim],
                3,
                crate::superfile::reader::VectorSearchOptions::new(),
                None,
            )
            .expect("vector search after hidden compaction");
        assert!(
            !hits.is_empty(),
            "vector search should still work after hidden compaction"
        );
    }

    #[test]
    fn ensure_fresh_under_strong_consistency_refreshes_against_storage() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = opts()
            .with_storage(storage)
            .with_read_consistency(Consistency::Strong);
        let st = Supertable::create(options).expect("create storage-backed handle");
        // `reader()` calls `ensure_fresh`, which under Strong drives a
        // blocking `refresh` against the storage pointer. No pointer is
        // published yet, so the pinned snapshot remains the empty
        // manifest.
        let r = st.reader();
        assert_eq!(r.n_superfiles(), 0);
        // A direct refresh likewise reports no newer manifest.
        let advanced = bridge_sync_to_async(st.refresh()).expect("refresh against empty store");
        assert!(!advanced, "no commit yet ⇒ refresh finds nothing newer");
    }
}
