// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `Supertable` + `SupertableReader` — the in-memory handle.
//!
//! `Supertable::create(opts).expect("create")` returns a clone-shared handle holding
//! an empty initial manifest behind `ArcSwap<ManifestSnapshot>`.
//! `Supertable::reader()` does `ArcSwap::load_full` once and pins
//! the resulting `Arc<ManifestSnapshot>` for the reader's lifetime, so a
//! reader captured before a commit keeps seeing pre-commit state
//! even after the writer has swapped in a new manifest.
//!
//! `SupertableInner.writer_outstanding: AtomicBool` is the
//! single-writer slot — the writer flips it true on acquisition
//! and (via `Drop`) flips it false on release.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    future::Future,
    sync::{Arc, Mutex, OnceLock, Weak, atomic::AtomicBool},
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use arrow_schema::SchemaRef;
use chrono::Utc;
use datafusion::{execution::context::SessionContext, logical_expr::LogicalPlan};
use tokio::runtime::Runtime;
use tracing::{debug, warn};

use super::{
    error::{BuildError, CommitError, OpenError},
    hidden_deleted::{self, HiddenDeletedError},
    manifest::{ManifestSnapshot, list::CellRoutingParams},
    options::SupertableOptions,
};
use crate::{
    config,
    runtime_bridge::{bridge_on_runtime, bridge_sync_to_async, shared_io_runtime},
    storage::{PrefixedStorageProvider, StorageError},
    superfile::{
        builder::VectorConfig,
        vector::{kmeans::kmeans, rerank_codec::RerankCodec},
    },
    supertable::{
        ManifestLoadError, SuperfileUri, SupertableStats,
        manifest::commit::{PointerProbe, probe_pointer, read_pointer},
        options::Consistency,
        query::{
            scalar_cache::DecodedScalarCache,
            sql::{SqlSchemas, build_sql_schemas},
        },
        reader_cache::disk::{DiskCacheError, skip_background_fill},
        stats::process_rss_bytes,
        tombstones::{SidecarCache, TombstoneSeqView, cache::DEFAULT_SEAL_TTL},
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
    /// Each commit publishes a new ManifestSnapshot via ArcSwap::store;
    /// readers do ArcSwap::load_full at construction to pin a
    /// snapshot for the duration of their queries.
    pub(super) manifest: ArcSwap<ManifestSnapshot>,
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
    /// Cached `SessionContext` for `query_sql`, keyed on the
    /// manifest `Arc` it was built against. Building one is
    /// ~1.5 ms (default optimizer rules + 3 TVF re-registrations
    /// + provider register), so reusing it across queries on the
    /// same snapshot is a large speedup for warm BM25 / vector
    /// SQL where the kernel itself runs in microseconds.
    ///
    /// Invalidation is automatic: every commit publishes a new
    /// `Arc<ManifestSnapshot>` via `manifest.store(...)`, so on the next
    /// `query_sql` the `Arc::ptr_eq` check fails and the cache
    /// is rebuilt against the fresh snapshot.
    pub(super) sql_session_cache: Mutex<Option<(Arc<ManifestSnapshot>, SessionContext)>>,
    /// Deterministic scalar SQL logical plans keyed by statement text and
    /// manifest identity. Physical plans are intentionally rebuilt so fresh
    /// tombstone overlays and query-stable functions retain their semantics.
    pub(super) sql_logical_plan_cache:
        Mutex<Option<(Arc<ManifestSnapshot>, HashMap<String, LogicalPlan>)>>,
    /// Bounded decoded-row cache shared by all readers of this immutable
    /// supertable handle.
    pub(super) decoded_scalar_cache: DecodedScalarCache,
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
    /// Set once at open when the hidden vector index is **configured and
    /// materialized** (its storage pointer exists) but fails to load/open — i.e.
    /// present-but-broken, distinct from never-configured or not-yet-drained
    /// (both leave `vector_index_table = None` with this unset). Vector search
    /// errors on a broken index rather than silently brute-scanning the user
    /// table; a genuinely absent index still falls back. Unset for the hidden
    /// table's own inner (no nested index).
    pub(super) hidden_index_open_error: std::sync::OnceLock<String>,
    /// Last time the read path checked the storage manifest pointer
    /// for freshness, under [`Consistency::BoundedStaleness`]. `None`
    /// until the first check (so the first query always refreshes).
    /// Unused for [`Consistency::Strong`] (always checks) and
    /// [`Consistency::Snapshot`] (never checks).
    pub(super) last_pointer_check: Mutex<Option<Instant>>,
    /// Etag of the manifest pointer from this handle's last storage
    /// probe. Powers the conditional (`If-None-Match`) freshness
    /// probe in [`Supertable::refresh`]: an unchanged pointer answers
    /// as a bodyless 304 instead of a full read. `None` until the
    /// first probe, and stale right after this process's own commits
    /// (which rewrite the pointer without capturing its new etag) —
    /// the next probe then takes the full-read path and re-seeds it.
    pub(super) last_pointer_etag: Mutex<Option<String>>,
    /// Cached SQL schemas, built once from the immutable `options` (lock-free
    /// lazy init). A pure function of the schema, so no snapshot invalidation
    /// (unlike `sql_session_cache`). See [`SqlSchemas`].
    pub(super) sql_schemas: OnceLock<Arc<SqlSchemas>>,
    /// Decoded hidden deleted-`_id` set, cached per hidden manifest version.
    /// The set is a deliberate duplicate of the user-table tombstones, carried
    /// INLINE in the hidden manifest so hidden vector search drops deleted rows
    /// from resident bytes instead of GETting the user table's tombstones on
    /// every query. Caching only adds the `SidecarCache`-style discipline on
    /// top: decode the inline bytes once per manifest version, not once per
    /// query. Keyed by `manifest_id`, which bumps on every deleted-id stamp.
    pub(super) hidden_deleted_cache: Mutex<Option<(u64, Arc<Vec<i128>>)>>,
}

impl SupertableInner {
    /// Runtime driving the sync API's async kernels when the caller
    /// isn't already on a tokio runtime. Process-wide — see
    /// [`shared_query_runtime`].
    pub(super) fn query_runtime(&self) -> Arc<Runtime> {
        shared_io_runtime()
    }

    /// The table's cached SQL schemas, built once from the immutable options.
    /// Cheap `Arc` clone on every call after the first.
    pub(super) fn sql_schemas(&self) -> Arc<SqlSchemas> {
        Arc::clone(
            self.sql_schemas
                .get_or_init(|| Arc::new(build_sql_schemas(&self.options))),
        )
    }

    /// Push the current manifest's tombstone-seq view into the
    /// sidecar cache. Called wherever a newer manifest is swapped
    /// into `self.manifest` (refresh, commit, mutation stamp) so the
    /// cache's freshness authority tracks the snapshot readers pin.
    /// No-op when the cache's view is already at (or past) the
    /// current manifest — the common every-query case, kept clone-free.
    pub(crate) fn reconcile_tombstone_seqs(&self) {
        let Some(cache) = self.tombstone_cache.as_ref() else {
            return;
        };
        let manifest = self.manifest.load();
        if manifest.manifest_id <= cache.view_manifest_id() {
            return;
        }
        cache.reconcile(tombstone_seq_view(&manifest));
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
    /// Open a persisted supertable.
    ///
    /// Reads the pointer file at
    /// `<root>/_supertable/current` via the storage provider
    /// attached on `options`, parses the manifest list, and
    /// eager-fetches every manifest part in parallel (the default;
    /// `options.eager_load_threshold_parts` below the part count opts
    /// into lazy loading). Open therefore scales with manifest size,
    /// and queries on the returned handle pay no serial manifest GETs.
    /// The returned `Supertable` is ready to serve queries from the
    /// snapshot at the pointer's `manifest_id`.
    ///
    /// A genuinely absent pointer is a [`ManifestLoadError::PointerNotFound`]
    /// error, not an empty table: `create` persists the initial pointer, so
    /// a registered table always has one, and a missing pointer is the
    /// open-or-create trigger (or a lost pointer) — surfaced, never masked
    /// as a silently-empty table.
    ///
    /// Errors:
    /// - [`OpenError::ManifestLoadError`] for manifest load failures,
    ///   including a missing pointer (`PointerNotFound`), parse, corruption,
    ///   or fetch.
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
        let manifest = ManifestSnapshot::load(None, storage, Some(options_arc.clone())).await?;
        // Resolve the hidden vector index into one of three states:
        //  * Present  — configured, materialized, opened → `Some(handle)`.
        //  * Absent   — not configured, or configured but no pointer yet
        //               (pre-first-drain) → `None`, no error. Queries fall back
        //               to the user table (its rows are the source of truth).
        //  * Broken   — configured and materialized (pointer exists) but the
        //               manifest/table won't load/open → `None` + an error.
        //               Vector queries surface the error instead of silently
        //               brute-scanning the user table.
        let (vector_index_table, hidden_index_broken) = if let Some(hidden_opts) =
            build_vector_index_options(options_arc.as_ref(), Some(manifest.as_ref()), None)
        {
            let hidden_storage = hidden_opts.storage.clone().ok_or_else(|| {
                OpenError::Build(BuildError::Store(
                    "VectorIndexSuperTable requires options.storage".into(),
                ))
            })?;
            match read_pointer(&*hidden_storage).await {
                Ok(Some(_)) => {
                    let hidden_arc = Arc::new(hidden_opts);
                    match ManifestSnapshot::load(None, hidden_storage, Some(hidden_arc.clone()))
                        .await
                    {
                        Ok(hidden_manifest) => {
                            match open_table_async(hidden_arc, hidden_manifest, None).await {
                                Ok(t) => (Some(Arc::new(t)), None),
                                Err(e) => {
                                    warn!(
                                        "supertable: hidden vector-index table failed to open: {e}"
                                    );
                                    (None, Some(e.to_string()))
                                }
                            }
                        }
                        Err(e) => {
                            warn!("supertable: hidden vector-index manifest failed to load: {e}");
                            (None, Some(e.to_string()))
                        }
                    }
                }
                // A consumer-memory-mode handle must not bootstrap-create the
                // sibling: its user summaries hydrate stripped, so the grid
                // can't be trained here and the create would durably stamp a
                // default (non-VectorCell) partition strategy. Absent is
                // correct — queries fall back to the user fan until a writer
                // handle materializes the index.
                Ok(None) if options_arc.summary_centroids_from_superfiles => (None, None),
                Ok(None) => match create_table_async(hidden_opts, None, None).await {
                    Ok(table) => (Some(Arc::new(table)), None),
                    Err(e) => {
                        // Surface a genuine bootstrap failure as Broken (carry the
                        // error) rather than Absent, matching the sibling arms —
                        // otherwise a storage fault silently degrades to full scan.
                        warn!("supertable: hidden vector-index bootstrap-create failed: {e}");
                        (None, Some(e.to_string()))
                    }
                },
                Err(e) => {
                    warn!("supertable: hidden vector-index pointer unreadable: {e}");
                    (None, Some(e.to_string()))
                }
            }
        } else {
            (None, None)
        };
        let handle = open_table_async(options_arc, manifest, vector_index_table).await?;
        if let Some(err) = hidden_index_broken {
            let _ = handle.inner.hidden_index_open_error.set(err);
        }
        // The manifests are loaded now, so the attached disk cache can be
        // sized against the real footprint (user + hidden index) instead
        // of whatever fixed default it was constructed with.
        handle.reconcile_cache_budget();
        debug!(
            manifest_id = handle.inner.manifest.load().manifest_id,
            "opened supertable"
        );
        Ok(handle)
    }

    /// Async create kernel. Sync [`Supertable::create`] bridges here.
    pub(crate) async fn create_async(options: SupertableOptions) -> Result<Self, OpenError> {
        if let Some(storage) = options.storage.as_ref() {
            let probe = Arc::clone(storage);
            match read_pointer(&*probe).await {
                Ok(Some(_pointer)) => return Self::open_async(options).await,
                Ok(None) => {}
                Err(e) => {
                    return Err(OpenError::Storage(StorageError::Permanent {
                        uri: "_supertable/current".into(),
                        source: Box::new(std::io::Error::other(format!("{e}"))),
                    }));
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
    /// `ManifestSnapshot` via content-addressed lookup, eager-fetch
    /// the newly-referenced parts, and `ArcSwap` the new
    /// `ManifestSnapshot` into place. Pre-refresh `SupertableReader`s
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

        // Conditional pointer probe: with the last-seen etag in hand,
        // an unchanged pointer answers as a bodyless 304 — the
        // steady-state cost of the consistency check is one
        // roundtrip, no transfer, no parse.
        let prev_etag = self
            .inner
            .last_pointer_etag
            .lock()
            .expect("last_pointer_etag mutex poisoned")
            .clone();
        let probe = probe_pointer(storage.as_ref(), prev_etag.as_deref())
            .await
            .map_err(OpenError::ManifestLoadError)?;
        let (pointer, meta) = match probe {
            PointerProbe::Absent | PointerProbe::NotModified => return Ok(false),
            PointerProbe::Read(pointer, meta) => (pointer, meta),
        };
        *self
            .inner
            .last_pointer_etag
            .lock()
            .expect("last_pointer_etag mutex poisoned") = meta.etag.clone();

        let current = self.inner.manifest.load_full();
        let manifest = match ManifestSnapshot::load_with_pointer(
            Some(current),
            storage,
            None,
            pointer,
        )
        .await
        {
            Ok(manifest) => manifest,
            // Pointer changed but our in-memory state already
            // covers it (e.g. this process's own commit rewrote
            // the pointer) — nothing newer to load, and the etag
            // captured above makes the next probe a 304.
            Err(ManifestLoadError::AlreadyLoaded) => return Ok(false),
            Err(err) => return Err(OpenError::ManifestLoadError(err)),
        };
        self.inner.manifest.store(manifest);
        self.inner.reconcile_tombstone_seqs();
        debug!(
            manifest_id = self.inner.manifest.load().manifest_id,
            "refreshed manifest"
        );
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
        self.pinned_reader()
    }
    }

    test_visible! {
    /// Pin the current in-memory manifest without a storage freshness check.
    /// Hidden vector queries use this for slow-state residency while their
    /// fast delete/watermark refresh runs concurrently with data I/O.
    fn pinned_reader(&self) -> SupertableReader {
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

    test_visible! {
    /// Hidden vector-index storage prefix recorded on the user manifest.
    /// Exposed only to tests/benches that explicitly administer derived state.
    fn vector_index_storage_prefix(&self) -> Option<String> {
        self.inner
            .manifest
            .load()
            .vector_index_storage_prefix()
            .map(str::to_owned)
    }
    }

    /// Decide whether this handle's consistency policy requires a pointer read
    /// now. Bounded-staleness callers share the timestamp so concurrent query
    /// paths cannot stampede storage.
    fn pointer_refresh_due(&self) -> bool {
        if self.inner.options.storage.is_none() {
            return false;
        }
        match self.inner.options.read_consistency {
            Consistency::Snapshot => false,
            Consistency::Strong => true,
            Consistency::BoundedStaleness(window) => {
                // Decide whether a check is due under the lock, stamp
                // "now" optimistically so concurrent queries don't all
                // stampede the pointer, then release the lock before I/O.
                {
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
                }
            }
        }
    }

    /// Async form used when freshness is one branch of a query I/O wave.
    /// Best-effort: a failed pointer read leaves the current snapshot in place.
    pub(crate) async fn ensure_fresh_async(&self) {
        if self.pointer_refresh_due() {
            let _ = self.refresh().await;
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
        if self.pointer_refresh_due()
            && let Err(e) = bridge_sync_to_async(self.refresh())
        {
            debug!(error = %e, "manifest refresh failed; serving current snapshot");
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
    /// # use infino::arrow_schema::{DataType, Field, Schema};
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

    /// Cached per-table SQL schemas (scan view + scalar schema).
    pub(crate) fn sql_schemas(&self) -> Arc<SqlSchemas> {
        self.inner.sql_schemas()
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

    /// Route undrained user superfiles into the hidden per-cell index. Not part
    /// of the public API — [`Supertable::optimize`] calls this before compact;
    /// tests and benches may invoke it directly via
    /// [`Supertable::drain_vectors_to_cells_sync`].
    pub(crate) fn drain_hidden_vector_cells_sync(&self) -> Result<(), BuildError> {
        let Some(hidden) = self.inner.vector_index_table.as_ref() else {
            return Ok(());
        };
        bridge_on_runtime(
            super::writer::drain_user_superfiles_to_hidden_cells(
                Arc::clone(&self.inner),
                Arc::clone(&hidden.inner),
            ),
            &self.query_runtime(),
        )?;
        // The drain writes the hidden per-cell index — roughly a second
        // copy of the vector payload — so the cache budget floor moves.
        self.reconcile_cache_budget();
        Ok(())
    }

    /// Total on-storage bytes of the committed superfiles across the user
    /// table and the hidden vector-index table, from the currently loaded
    /// manifest views (lazy, not-yet-loaded manifest parts contribute 0 —
    /// the reconcile below is raise-only, so an undercount is safe).
    pub(crate) fn on_storage_footprint_bytes(&self) -> u64 {
        let table_bytes = |inner: &SupertableInner| -> u64 {
            inner
                .manifest
                .load_full()
                .superfiles
                .iter()
                .filter_map(|e| e.subsection_offsets.as_ref())
                .map(|o| o.total_size)
                .sum()
        };
        let user = table_bytes(&self.inner);
        let hidden = self
            .inner
            .vector_index_table
            .as_ref()
            .map(|h| table_bytes(&h.inner))
            .unwrap_or(0);
        user.saturating_add(hidden)
    }

    /// Reconcile the attached disk cache's budget with the table's current
    /// on-storage footprint (user + hidden index + headroom). Called after
    /// open — once the manifests are loaded — and again after the drain
    /// grows the hidden index. Raise-only for engine-managed (auto-sized)
    /// budgets; an explicit budget is never changed, but gets a one-shot
    /// warning when the footprint exceeds it (steady-state reads would
    /// churn the cache).
    pub(crate) fn reconcile_cache_budget(&self) {
        let Some(cache) = self.inner.options.disk_cache.as_ref() else {
            return;
        };
        let footprint = self.on_storage_footprint_bytes();
        if footprint == 0 {
            return;
        }
        let floor = footprint.saturating_add(footprint / CACHE_BUDGET_HEADROOM_DIVISOR);
        cache.reconcile_budget_floor(floor, footprint);
    }

    #[cfg(any(test, feature = "test-helpers"))]
    test_visible! {
    /// No-staging drain: build the hidden per-cell index by routing + splicing
    /// the **user** superfiles' local clusters into cells (multi-cluster
    /// fragments — inner pruning preserved). Called on the user-facing table
    /// (it owns the hidden `vector_index_table`); benches invoke it between the
    /// pre-drain and post-drain search phases.
    fn drain_vectors_to_cells_sync(&self) -> Result<(), BuildError> {
        self.drain_hidden_vector_cells_sync()
    }
    }

    /// Block until the disk cache has settled every background fill the
    /// caller's own queries kicked off, or `timeout` elapses.
    ///
    /// Scoped to in-flight work only: superfiles never opened, and opens
    /// that don't spawn fills (vector), are not waited on — so a query on
    /// a small working set settles fast regardless of table size.
    /// Registering this waiter lets pending fills proceed even if another
    /// handle still holds a lazy reader.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn wait_until_warm(&self, timeout: Duration) -> Result<(), DiskCacheError> {
        let Some(cache) = self.inner.options.disk_cache.as_ref() else {
            return Ok(());
        };
        if skip_background_fill() {
            return Ok(());
        }
        let cache = Arc::clone(cache);
        self.block_on_query(async move { cache.wait_until_fills_settled(timeout).await })
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

    /// Run one GC sweep over this supertable's `wal/mutations/` prefix.
    /// Reaps `Complete` WALs older than the wal-grace window + orphan
    /// `.arrow` sidecars older than the sidecar-grace window. Runs at
    /// `Supertable::open`/`create` and again on every `optimize()` call.
    /// Not public API: exposed only as a manual hook for in-crate tests
    /// that need custom grace windows via `wal::gc::run_sweep` directly.
    pub(crate) async fn run_gc_sweep_once(&self) -> Result<gc::GcReport, gc::GcError> {
        gc::run_sweep(
            self,
            Utc::now(),
            gc::DEFAULT_WAL_GRACE,
            gc::DEFAULT_SIDECAR_GRACE,
        )
        .await
    }

    /// Sync-bridged version of [`run_gc_sweep_once`], for callers (like
    /// [`Supertable::optimize`]) that aren't already inside an async
    /// context.
    pub(crate) fn run_gc_sweep_once_blocking(&self) -> Result<gc::GcReport, gc::GcError> {
        bridge_on_runtime(self.run_gc_sweep_once(), &self.inner.query_runtime())
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
                            true,
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
    /// Diagnostic: for every packed hidden cell, the stable `_id`s stored in
    /// it (merged across shard superfiles, sorted by cell id). `None` when
    /// there is no hidden table or a hidden blob has no packed cells. Used by
    /// tests/benches to audit drain assignment against the global cell grid.
    fn hidden_cell_stable_id_sets(&self) -> Option<Vec<(u32, Vec<i128>)>> {
        let hidden = self.inner.vector_index_table.as_ref()?;
        let hidden_manifest = hidden.inner.manifest.load_full();
        let store = hidden_manifest.options.store.clone();
        let disk_cache = hidden_manifest.options.disk_cache.clone();
        let storage = hidden_manifest.options.storage.clone()?;
        let targets: Vec<_> = hidden_manifest
            .superfiles
            .iter()
            .map(|e| (e.uri, e.subsection_offsets.clone()))
            .collect();
        let merged = self
            .block_on_query(async move {
                let mut by_cell: HashMap<u32, Vec<i128>> = HashMap::new();
                for (uri, offsets) in targets {
                    let reader = crate::supertable::query::superfile_reader::superfile_reader(
                        &store,
                        disk_cache.as_ref(),
                        Some(&storage),
                        &uri,
                        offsets.as_ref(),
                        true,
                    )
                    .await
                    .map_err(|e| e.to_string())?;
                    let Some(vec_reader) = reader.vec() else {
                        continue;
                    };
                    let Some(cells) = vec_reader
                        .packed_cell_stable_ids_async()
                        .await
                        .map_err(|e| e.to_string())?
                    else {
                        continue;
                    };
                    for (cell_id, ids) in cells {
                        by_cell.entry(cell_id).or_default().extend(ids);
                    }
                }
                Ok::<_, String>(by_cell)
            })
            .ok()?;
        let mut out: Vec<(u32, Vec<i128>)> = merged.into_iter().collect();
        out.sort_unstable_by_key(|(cell, _)| *cell);
        Some(out)
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
        // Parts are table-level size buckets; per-cell identity lives on each
        // superfile entry's partition key.
        let mut by_cell: HashMap<Vec<u8>, usize> = HashMap::new();
        let flat_superfiles = manifest.get_all_superfiles();
        for entry in flat_superfiles {
            *by_cell.entry(entry.partition_key.clone()).or_default() += 1;
        }
        let total = flat_superfiles.len();
        if total == 0 {
            return Some((0, 0));
        }
        let max_per_cell = by_cell.values().copied().max().unwrap_or(0);
        Some((total, max_per_cell))
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
    pub(crate) fn sql_session_cache(
        &self,
    ) -> &Mutex<Option<(Arc<ManifestSnapshot>, SessionContext)>> {
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
/// Cell count for the **user** table's grid, trained at the first commit —
/// used to cell-pack user superfiles and route the pre-drain query. The
/// per-table option wins when set; otherwise `vector.user_cell_count`
/// from the YAML config (no env override).
pub(crate) fn user_vector_cell_count(options: &SupertableOptions) -> usize {
    options
        .user_cell_count
        .unwrap_or_else(|| config::global().vector.user_cell_count)
        .max(1)
}

/// Cell count for the **hidden** vector index: `global_vector_index.grid` is
/// trained at this count and the drain reads it verbatim; post-drain routing
/// runs at this granularity. The per-table option wins when set; otherwise
/// `vector.hidden_cell_count` from the YAML config (no env override).
pub(crate) fn hidden_vector_cell_count(options: &SupertableOptions) -> usize {
    options
        .hidden_cell_count
        .unwrap_or_else(|| config::global().vector.hidden_cell_count)
        .max(1)
}

/// Reserved VectorCell partition id for the hidden index's "incoming" append
/// region. Each hidden commit writes one IVF superfile under this sentinel
/// partition holding that whole batch (all cells mixed, unsorted). Queries
/// always scan the incoming superfiles in addition to the nprobe-routed cell
/// superfiles; background OPANN maintenance later routes incoming into the
/// per-cell IVF superfiles and deletes it. `u32::MAX` is out of the
/// valid cell range `0..n_cent`, so it never collides with a real cell.
pub(crate) const INCOMING_VECTOR_CELL: u32 = u32::MAX;

/// Lloyd iterations when folding per-superfile cluster centroids into the
/// global cell grid at open/create time.
pub(crate) const GLOBAL_VECTOR_KMEANS_ITERS: usize = 8;

/// Fixed PRNG seed for global centroid training.
pub(crate) const GLOBAL_VECTOR_KMEANS_SEED: u64 = 0x51ED_2A11;

/// Headroom an engine-managed (auto-sized) cache budget keeps over the
/// table's on-storage footprint, in divisor form (`footprint +
/// footprint / this`). Slack for in-flight cold-fetch reservations while
/// the full working set stays resident.
const CACHE_BUDGET_HEADROOM_DIVISOR: u64 = 10;

/// Train global VectorCell centroids from the user manifest and queue them
/// on the hidden index table for its next commit.
/// Aggressive compaction profile for the hidden vector-index table: keep
/// ~one compact packed shard object per partition key instead of many
/// small delta files.
/// True for the derived hidden vector-index sibling (VectorCell routing, no FTS).
pub(crate) fn is_hidden_vector_index_table(opts: &SupertableOptions) -> bool {
    !opts.vector_columns.is_empty()
        && opts.fts_columns.is_empty()
        && matches!(
            opts.partition_strategy,
            Some(crate::supertable::manifest::list::PartitionStrategy::VectorCell { .. })
        )
}

pub(crate) fn hidden_vector_index_compaction_settings() -> crate::config::CompactionSettings {
    let vector = &crate::config::global().vector;
    crate::config::CompactionSettings {
        target_superfile_size_mb: vector.compaction_target_mb,
        min_fill_percent: vector.compaction_min_fill_percent,
        max_memory_mb: vector.compaction_max_memory_mb,
        ..Default::default()
    }
}

/// Open-time bootstrap only: derive initial global centroids from an
/// existing user-table IVF summary. Hidden commits use
/// [`super::opann`] MVCC maintenance — never call this per commit.
pub(crate) fn train_global_centroids(
    user_opts: &SupertableOptions,
    manifest: &super::manifest::ManifestSnapshot,
    n_cells: usize,
) -> Option<super::manifest::ClusterCentroids> {
    let vc = user_opts.vector_columns.first()?;
    let mut all_centroids = Vec::new();
    let mut dim = 0usize;
    for entry in manifest.superfiles.iter() {
        let Some(vs) = entry.vector_summary.get(&vc.column) else {
            continue;
        };
        for cell in &vs.cells {
            let clusters = &cell.clusters;
            // Stripped summaries (read-only consumer memory mode) carry no
            // fp32 to train from; grid bootstrap is a writer-side concern.
            if clusters.is_empty() || !clusters.vectors_resident() {
                continue;
            }
            dim = clusters.dim as usize;
            for c in 0..clusters.n_cent as usize {
                if clusters.counts[c] == 0 {
                    continue;
                }
                all_centroids.extend_from_slice(clusters.centroid(c));
            }
        }
    }
    if all_centroids.is_empty() || dim == 0 {
        return None;
    }
    let n_src = all_centroids.len() / dim;
    let n = n_cells.min(n_src).max(1);
    let centroids = kmeans(
        &all_centroids,
        dim,
        n,
        GLOBAL_VECTOR_KMEANS_ITERS,
        GLOBAL_VECTOR_KMEANS_SEED,
    );
    Some(super::manifest::ClusterCentroids::from_fp32(
        n as u32,
        dim as u32,
        &centroids,
        vec![1u32; n],
    ))
}

pub(crate) fn legacy_vector_index_storage_prefix() -> &'static str {
    super::manifest::DEFAULT_VECTOR_INDEX_PREFIX
}

fn generate_vector_index_storage_prefix() -> String {
    format!("_infino_{}_vector_index", uuid::Uuid::new_v4())
}

fn resolve_vector_index_storage_prefix(
    user_opts: &SupertableOptions,
    user_manifest: Option<&super::manifest::ManifestSnapshot>,
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
    user_manifest: Option<&super::manifest::ManifestSnapshot>,
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
    // Hidden maintenance reads residual-family rows without fp32
    // reconstruction. Preserve a fixed/local residual user codec; non-residual
    // user codecs retain the existing local-residual hidden representation.
    let hidden_vector_columns: Vec<VectorConfig> = user_opts
        .vector_columns
        .iter()
        .map(|vc| VectorConfig {
            rerank_codec: if vc.rerank_codec.is_sq8_residual_family() {
                vc.rerank_codec
            } else {
                RerankCodec::Sq8Residual
            },
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
        .with_reader_pool(Arc::clone(&user_opts.reader_pool))
        .with_writer_pool(Arc::clone(&user_opts.writer_pool))
        .with_read_consistency(user_opts.read_consistency);
    hidden_opts.connection_memory_budget = Arc::clone(&user_opts.connection_memory_budget);
    // Hidden-manifest summaries hydrate stripped unconditionally (the fp32
    // wire home is the slow-CAS centroid section), so the consumer memory
    // mode has nothing left to gate there. Keep it off on the derived
    // options: the flag marks a handle read-only, and hidden maintenance
    // (drain, split, compaction) must stay writable regardless of how the
    // user handle was opened.
    hidden_opts.summary_centroids_from_superfiles = false;
    // Per-table cell-count overrides ride along too: the hidden handle's
    // paths resolve counts through its own options.
    hidden_opts.user_cell_count = user_opts.user_cell_count;
    hidden_opts.hidden_cell_count = user_opts.hidden_cell_count;
    if let Some(cache) = user_opts.disk_cache.as_ref() {
        hidden_opts = hidden_opts.with_disk_cache(Arc::clone(cache));
    }
    if let Some(manifest) = user_manifest
        && let Some(clusters) =
            train_global_centroids(user_opts, manifest, hidden_vector_cell_count(user_opts))
    {
        hidden_opts = hidden_opts.with_partition_strategy(
            crate::supertable::manifest::list::PartitionStrategy::VectorCell {
                column: user_opts.vector_columns[0].column.clone(),
                clusters,
                routing: CellRoutingParams::default(),
            },
        );
    }
    Some(hidden_opts)
}

/// Build one supertable handle. Leaf — never creates a hidden sibling.
async fn build_handle(
    options: Arc<SupertableOptions>,
    manifest: Arc<ManifestSnapshot>,
    vector_index_table: Option<Arc<Supertable>>,
) -> Result<Supertable, OpenError> {
    let tombstone_cache = build_tombstone_cache(&options, &manifest);
    let id_generator = crate::supertable::utils::idgen::IdGenerator::new();
    let handle_id = crate::supertable::wal::state_doc::SupertableHandleId(id_generator.next_id());
    let inner = Arc::new(SupertableInner {
        options,
        manifest: ArcSwap::new(manifest),
        writer_outstanding: AtomicBool::new(false),
        compaction_outstanding: AtomicBool::new(false),
        id_generator: Mutex::new(id_generator),
        sql_session_cache: Mutex::new(None),
        sql_logical_plan_cache: Mutex::new(None),
        decoded_scalar_cache: DecodedScalarCache::default(),
        tombstone_cache,
        handle_id,
        vector_index_table,
        hidden_index_open_error: std::sync::OnceLock::new(),
        last_pointer_check: Mutex::new(None),
        last_pointer_etag: Mutex::new(None),
        hidden_deleted_cache: Mutex::new(None),
        sql_schemas: OnceLock::new(),
    });
    install_disk_cache_pinning(&inner);
    let st = Supertable { inner };
    if st.inner.options.storage.is_some() {
        // Best-effort: a sweep failure here doesn't fail handle
        // construction; the next sweep gets another shot.
        if let Err(e) = st.run_recovery_sweep_once().await {
            warn!(error = %e, "open-time recovery sweep failed (best-effort)");
        }
        if let Err(e) = st.run_gc_sweep_once().await {
            warn!(error = %e, "open-time gc sweep failed (best-effort)");
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
    // A durable create *persists* the initial empty manifest — its list plus
    // the pointer at `manifest_id 0` — so the freshly created table is
    // openable right away: before any append, after a reopen, and from
    // another process (`open` requires a pointer). This doesn't shift the id
    // sequence: the first append still commits `manifest_id 1`. An in-memory
    // table keeps the lighter in-process-only empty snapshot.
    let (manifest, vector_index_table) = if let Some(storage) = options.storage.clone() {
        let materialized = Arc::new(
            ManifestSnapshot::materialized_empty_with_vector_index_prefix(
                options.clone(),
                vector_index_storage_prefix,
            ),
        );
        // `expected_prev_etag = None` is the initial-commit shape: no prior
        // pointer to fence on.
        match materialized.write(storage.as_ref(), None, &[]).await {
            Ok(()) => (materialized, vector_index_table),
            // Lost the initial-pointer race to a concurrent creator on the
            // same storage: adopt their committed manifest rather than
            // failing — `create` is create-or-open, and a pointer that
            // appeared between the caller's probe and this write is the same
            // as "pointer already present". Drop the loser's pre-built
            // hidden handle (it used a freshly generated prefix the durable
            // manifest does not track) and reopen against the winner's
            // stamped prefix.
            Err(CommitError::WriteContentionExhausted) => {
                let adopted = ManifestSnapshot::load(None, storage, Some(options.clone())).await?;
                let reconciled =
                    reconcile_vector_index_table_to_manifest(options.as_ref(), &adopted).await?;
                (adopted, reconciled)
            }
            Err(e) => return Err(e.into()),
        }
    } else {
        (
            Arc::new(ManifestSnapshot::empty_with_vector_index_prefix(
                options.clone(),
                vector_index_storage_prefix,
            )),
            vector_index_table,
        )
    };
    build_handle(options, manifest, vector_index_table).await
}

/// After a lost create-race, open (or bootstrap) the hidden vector-index
/// table at the prefix stamped in `adopted` — never keep the loser's
/// process-local UUID prefix.
async fn reconcile_vector_index_table_to_manifest(
    user_opts: &SupertableOptions,
    adopted: &ManifestSnapshot,
) -> Result<Option<Arc<Supertable>>, OpenError> {
    let Some(hidden_opts) = build_vector_index_options(user_opts, Some(adopted), None) else {
        return Ok(None);
    };
    let hidden_storage = hidden_opts.storage.clone().ok_or_else(|| {
        OpenError::Build(BuildError::Store(
            "VectorIndexSuperTable requires options.storage".into(),
        ))
    })?;
    match read_pointer(&*hidden_storage).await {
        Ok(Some(_)) => {
            let hidden_arc = Arc::new(hidden_opts);
            let hidden_manifest =
                ManifestSnapshot::load(None, hidden_storage, Some(hidden_arc.clone())).await?;
            Ok(Some(Arc::new(
                open_table_async(hidden_arc, hidden_manifest, None).await?,
            )))
        }
        Ok(None) => {
            // Leaf create at the winner's already-prefixed storage — do not
            // recurse through `create_table_async` (that path reconciles
            // again and would form an infinitely sized future).
            let hidden_arc = Arc::new(hidden_opts);
            let manifest = if let Some(storage) = hidden_arc.storage.clone() {
                let materialized = Arc::new(
                    ManifestSnapshot::materialized_empty_with_vector_index_prefix(
                        Arc::clone(&hidden_arc),
                        None,
                    ),
                );
                match materialized.write(storage.as_ref(), None, &[]).await {
                    Ok(()) => materialized,
                    Err(CommitError::WriteContentionExhausted) => {
                        ManifestSnapshot::load(None, storage, Some(Arc::clone(&hidden_arc))).await?
                    }
                    Err(e) => return Err(e.into()),
                }
            } else {
                Arc::new(ManifestSnapshot::empty_with_vector_index_prefix(
                    Arc::clone(&hidden_arc),
                    None,
                ))
            };
            Ok(Some(Arc::new(
                build_handle(hidden_arc, manifest, None).await?,
            )))
        }
        Err(e) => Err(OpenError::Storage(StorageError::Permanent {
            uri: "_supertable/current".into(),
            source: Box::new(std::io::Error::other(format!("{e}"))),
        })),
    }
}

/// Open one supertable handle from a loaded manifest. Leaf — never creates a sibling.
async fn open_table_async(
    options: Arc<SupertableOptions>,
    manifest: Arc<ManifestSnapshot>,
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
/// entirely. The cache is born with the seq view of `manifest`
/// (the snapshot the handle opens with), so it is authoritative
/// from the first query.
fn build_tombstone_cache(
    options: &Arc<SupertableOptions>,
    manifest: &ManifestSnapshot,
) -> Option<Arc<SidecarCache>> {
    let storage = options.storage.as_ref()?.clone();
    let wal_store = WalStore::new(storage);
    Some(Arc::new(SidecarCache::new(
        wal_store,
        DEFAULT_SEAL_TTL,
        tombstone_seq_view(manifest),
    )))
}

/// The tombstone-seq view of `manifest`, in the shape the sidecar
/// cache validates against. An in-process-only manifest (no
/// persisted list) has no sidecars, so its view is empty.
fn tombstone_seq_view(manifest: &ManifestSnapshot) -> Arc<TombstoneSeqView> {
    Arc::new(TombstoneSeqView {
        manifest_id: manifest.manifest_id,
        seqs: manifest.get_tombstone_seqs().cloned().unwrap_or_default(),
    })
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

/// Snapshot-pinned reader. Captures `Arc<ManifestSnapshot>` at construction
/// and holds it through query lifetime — new commits to the parent
/// `Supertable` don't affect this reader's view. The public read
/// methods (`bm25_search`, `bm25_search_prefix`, `vector_search`,
/// `hybrid_search`, `query_sql`) live on this handle; each drives its async kernel to
/// completion via the sync→async bridge ([`SupertableReader::block_on`]),
/// mirroring the way [`SupertableWriter`](crate::supertable::SupertableWriter)
/// drives `commit`.
#[derive(Clone)]
pub struct SupertableReader {
    manifest: Arc<ManifestSnapshot>,
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
/// `Weak<SupertableInner>` plus the pinned `Arc<ManifestSnapshot>` (a manifest
/// never points back at the inner, so it adds no cycle) and rebuilds
/// the strong reader on demand. The upgrade always succeeds while a
/// query is executing, because the live consumer keeps the inner alive.
#[derive(Clone)]
pub(crate) struct WeakReader {
    inner: Weak<SupertableInner>,
    manifest: Arc<ManifestSnapshot>,
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
    /// ManifestSnapshot id pinned at construction. Useful for asserting
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

    #[cfg(any(test, feature = "test-helpers"))]
    test_visible! {
    /// Load every lazy manifest part and return `(superfiles, index bytes)`.
    /// Benchmarks use this to size a cache when reopening a retained table.
    fn load_superfile_storage_stats(&self) -> Result<(usize, u64), ManifestLoadError> {
        let entries = self.block_on(self.manifest.get_all_superfiles_loaded())?;
        let total_index_bytes = entries
            .iter()
            .filter_map(|entry| entry.subsection_offsets.as_ref())
            .map(|offsets| offsets.total_size)
            .sum();
        Ok((entries.len(), total_index_bytes))
    }
    }

    /// Total documents across all superfiles visible to this reader.
    pub fn n_docs_total(&self) -> u64 {
        self.manifest.n_docs_total()
    }

    /// Pinned manifest. Exposed for query-side machinery
    /// (skip helpers, fan-out, etc.) to read the superfile list
    /// + summaries directly.
    pub fn manifest(&self) -> &Arc<ManifestSnapshot> {
        &self.manifest
    }

    pub(crate) fn decoded_scalar_cache(&self) -> &DecodedScalarCache {
        &self.inner.decoded_scalar_cache
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
        manifest: Arc<ManifestSnapshot>,
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

    /// Cached per-table SQL schemas (scan view + scalar schema).
    pub(crate) fn sql_schemas(&self) -> Arc<SqlSchemas> {
        self.inner.sql_schemas()
    }

    /// Cached `SessionContext` keyed on the manifest `Arc`, reused by
    /// [`SupertableReader::query_sql`] across queries on this snapshot.
    pub(crate) fn sql_session_cache(
        &self,
    ) -> &Mutex<Option<(Arc<ManifestSnapshot>, SessionContext)>> {
        &self.inner.sql_session_cache
    }

    /// Cached deterministic scalar SQL plans for this reader's manifest.
    pub(crate) fn sql_logical_plan_cache(
        &self,
    ) -> &Mutex<Option<(Arc<ManifestSnapshot>, HashMap<String, LogicalPlan>)>> {
        &self.inner.sql_logical_plan_cache
    }

    pub(crate) fn vector_index_table(&self) -> Option<&Arc<Supertable>> {
        self.inner.vector_index_table.as_ref()
    }

    /// `Some(reason)` when a **configured and materialized** hidden vector index
    /// failed to load/open at table-open (present-but-broken). `None` when the
    /// index is present (usable) or genuinely absent. Vector search uses this to
    /// fail loud on a broken index instead of falling back to a user-table scan.
    pub(crate) fn hidden_index_open_error(&self) -> Option<&str> {
        self.inner.hidden_index_open_error.get().map(String::as_str)
    }

    /// Decoded hidden deleted-`_id` set for this reader's pinned manifest,
    /// cached per manifest version so the inline bytes are decoded once per
    /// version rather than once per query (the `SidecarCache` discipline).
    ///
    /// The set itself is a deliberate duplicate of the user-table tombstones,
    /// carried inline in the hidden manifest: hidden vector search drops
    /// deleted rows from these resident bytes instead of GETting the user
    /// table's per-superfile tombstones on every query.
    pub(crate) fn hidden_deleted_ids(&self) -> Result<Arc<Vec<i128>>, HiddenDeletedError> {
        let version = self.manifest.get_manifest_id();
        {
            let guard = self
                .inner
                .hidden_deleted_cache
                .lock()
                .expect("hidden deleted-set cache mutex poisoned");
            if let Some((cached_version, ids)) = guard.as_ref()
                && *cached_version == version
            {
                return Ok(Arc::clone(ids));
            }
        }
        let ids = hidden_deleted::deleted_user_ids(&self.manifest)?;
        *self
            .inner
            .hidden_deleted_cache
            .lock()
            .expect("hidden deleted-set cache mutex poisoned") = Some((version, Arc::clone(&ids)));
        Ok(ids)
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
    use std::{
        collections::{HashMap, HashSet},
        sync::Arc,
    };

    use arrow_schema::{DataType, Field, Schema};
    use tempfile::TempDir;
    use uuid::Uuid;

    use super::*;
    use crate::{
        config::OptimizeOptions,
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::{builder::FtsConfig, vector::layout::VectorLayout},
        supertable::{
            manifest::{SuperfileEntry, SuperfileUri},
            options::Consistency,
            query::dispatch::open_reader,
        },
        test_helpers::default_tokenizer,
    };

    fn rerank_payloads_by_stable_id(table: &Supertable) -> HashMap<i128, Vec<u8>> {
        let table_reader = table.reader();
        let manifest = table_reader.manifest();
        let entries = bridge_sync_to_async(manifest.get_all_superfiles_loaded())
            .expect("load superfile entries");
        let mut payloads = HashMap::new();
        for entry in entries {
            let reader = bridge_sync_to_async(open_reader(
                &manifest.options.store,
                manifest.options.disk_cache.as_ref(),
                manifest.options.storage.as_ref(),
                &entry,
                true,
            ))
            .expect("open superfile");
            let vector = reader.vec().expect("vector reader");
            let rows = bridge_sync_to_async(vector.materialized_index_rows_async("emb"))
                .expect("materialized residual rows");
            for row in rows {
                let mut bytes = row.encoded.codes;
                bytes.extend_from_slice(&row.encoded.residuals);
                if let Some(previous) = payloads.insert(row.stable_id, bytes.clone()) {
                    assert_eq!(previous, bytes, "replica payload changed");
                }
            }
        }
        payloads
    }

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
                positions: false,
            }],
            vec![],
            Some(tk),
        )
        .expect("valid options")
    }

    fn entry(n_docs: u64) -> Arc<SuperfileEntry> {
        let id = Uuid::new_v4();
        Arc::new(SuperfileEntry {
            birth_version: 0,
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
        // The reader's pinned Arc<ManifestSnapshot> must keep the manifest
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
        // Arc<ManifestSnapshot>. The Arc-share is what makes "thousands of
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
    fn query_runtime_is_process_shared() {
        let st1 = Supertable::create(opts()).expect("create");
        let st2 = Supertable::create(opts()).expect("create");
        // Every handle sees the one process-level query runtime — repeated
        // calls and independent handles never build extra tokio workers.
        assert!(Arc::ptr_eq(&st1.query_runtime(), &st1.query_runtime()));
        assert!(Arc::ptr_eq(&st1.query_runtime(), &st2.query_runtime()));
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
    fn vector_search_works_after_commit_and_drain() {
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
        let make_options = || {
            SupertableOptions::new(
                schema.clone(),
                vec![FtsConfig {
                    column: "title".into(),
                    positions: false,
                }],
                vec![VectorConfig {
                    column: "emb".into(),
                    dim,
                    n_cent: 4,
                    rot_seed: 7,
                    metric: Metric::Cosine,
                    rerank_codec: RerankCodec::Sq8FixedResidual,
                    provided_centroids: None,
                }],
                Some(crate::test_helpers::default_tokenizer()),
            )
            .expect("valid options")
            .with_storage(Arc::clone(&storage))
            .with_writer_pool(Arc::clone(&pool))
        };
        let st = Supertable::create(make_options()).expect("create");
        assert!(
            st.reader().vector_index_table().is_some(),
            "vector columns + storage must create hidden index sibling"
        );

        let titles = LargeStringArray::from(vec!["a", "b", "c"]);
        let flat = Float32Array::from(vec![1.0f32; 3 * dim]);
        let fsl = FixedSizeListArray::new(item_field, dim as i32, Arc::new(flat), None);
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

        assert!(st.reader().n_superfiles() > 0);
        let user_payloads = rerank_payloads_by_stable_id(&st);
        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden index")
            .clone();
        assert_eq!(
            hidden.options().vector_columns[0].rerank_codec,
            RerankCodec::Sq8FixedResidual,
            "hidden table must inherit the fixed residual codec"
        );
        // Phase B: the commit does NOT dual-write into the hidden table. It only
        // bootstraps the global cell grid into the hidden manifest; the cell
        // superfiles are drained from the user superfiles on demand.
        assert_eq!(
            hidden.reader().n_superfiles(),
            0,
            "commit must not dual-write into the hidden table"
        );
        assert!(
            st.reader()
                .manifest()
                .get_global_vector_index()
                .is_some_and(|g| g.grid.n_cent > 0 && g.grid.dim > 0),
            "commit must bootstrap the global cell grid into the user manifest"
        );
        // The finer user-side grid is trained exactly when the two counts
        // differ; with the default single grid (`user_cell_count` ==
        // `hidden_cell_count`) it stays `None` and the user side falls back
        // to `grid` via `into_user_grid`.
        let user_grid_trained = st
            .reader()
            .manifest()
            .get_global_vector_index()
            .is_some_and(|g| {
                g.user_grid
                    .as_ref()
                    .is_some_and(|u| u.n_cent > 0 && u.dim > 0)
            });
        assert_eq!(
            user_grid_trained,
            user_vector_cell_count(st.options()) != hidden_vector_cell_count(st.options()),
            "user-side grid must be trained exactly when the cell counts differ"
        );
        assert_eq!(
            st.reader().manifest().superfiles[0].vector_layout,
            VectorLayout::MultiCellIvf,
            "grid commit must emit packed user MultiCellIvf superfiles"
        );

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        // Pre-drain: with empty cells the query falls back to the user superfiles.
        let hits = st
            .reader()
            .vector_hits("emb", &q, 3, VectorSearchOptions::new(), None)
            .expect("vector search");
        assert!(
            !hits.is_empty(),
            "pre-drain search must fall back to the user superfiles"
        );

        // Reopen before drain: the user flat view is lazy/empty and its
        // superfile entries live in manifest parts. Drain must hydrate those
        // authoritative parts rather than treating the table as empty.
        drop(w);
        drop(hidden);
        drop(st);
        let st = Supertable::open(make_options()).expect("reopen before drain");
        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden index after reopen")
            .clone();

        // Drain the parts-backed user superfiles into hidden cells; the query
        // is now served by the hidden cell index.
        st.drain_vectors_to_cells_sync().expect("drain to cells");
        let hidden_payloads = rerank_payloads_by_stable_id(&hidden);
        assert_eq!(
            hidden_payloads, user_payloads,
            "fixed residual payloads must survive default k-means drain"
        );
        assert!(
            hidden.reader().n_superfiles() > 0,
            "drain must populate the hidden cell index"
        );
        let hits2 = st
            .reader()
            .vector_hits("emb", &q, 3, VectorSearchOptions::new(), None)
            .expect("post-drain vector search");
        assert!(
            !hits2.is_empty(),
            "post-drain search must hit the hidden cells"
        );

        let user_uris: HashSet<_> = st
            .reader()
            .manifest()
            .superfiles
            .iter()
            .map(|entry| entry.uri)
            .collect();
        let in_process_drained = hidden.reader().manifest().get_drained_ranges();
        assert!(
            st.reader()
                .manifest()
                .superfiles
                .iter()
                .all(|entry| in_process_drained.contains(entry.birth_version)),
            "drain must cover every in-process user birth version"
        );

        // A fresh consumer must recover the same drained watermark. Otherwise
        // the tiered query misclassifies fully-drained user files as deltas and
        // reads both user and hidden vector data.
        drop(hidden);
        drop(st);
        let reopened = Supertable::open(make_options()).expect("reopen after drain");
        let reopened_hidden = reopened
            .reader()
            .vector_index_table()
            .expect("hidden index after drained reopen")
            .clone();
        let reopened_drained = reopened_hidden.reader().manifest().get_drained_ranges();
        assert!(
            reopened
                .reader()
                .manifest()
                .superfiles
                .iter()
                .all(|entry| reopened_drained.contains(entry.birth_version)),
            "cold reopen must retain every drained user birth version"
        );
        let cold_hits = reopened
            .reader()
            .vector_hits("emb", &q, 3, VectorSearchOptions::new(), None)
            .expect("cold post-drain vector search");
        assert!(
            cold_hits
                .iter()
                .all(|hit| !user_uris.contains(&hit.superfile)),
            "cold post-drain search must not read user superfiles"
        );

        reopened.append(&batch).expect("append fixed delta");
        reopened
            .drain_vectors_to_cells_sync()
            .expect("drain fixed delta");
        let hidden = reopened
            .reader()
            .vector_index_table()
            .expect("hidden after second drain")
            .clone();
        let before_compaction = rerank_payloads_by_stable_id(&hidden);
        reopened
            .optimize(&OptimizeOptions::default())
            .expect("compact fixed hidden index");
        let after_compaction = rerank_payloads_by_stable_id(&hidden);
        assert_eq!(
            after_compaction, before_compaction,
            "fixed residual payloads must survive compaction"
        );
    }

    /// The per-table cell-count overrides win over the YAML config; the
    /// `.max(1)` floor still applies. No assertion on the config-backed
    /// default value itself — the test environment's YAML may override it.
    #[test]
    fn vector_cell_count_options_override_config() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "text",
            DataType::LargeUtf8,
            false,
        )]));
        let base = SupertableOptions::new(schema, vec![], vec![], None).expect("options");
        let overridden = base.with_vector_cell_counts(7, 9);
        assert_eq!(user_vector_cell_count(&overridden), 7);
        assert_eq!(hidden_vector_cell_count(&overridden), 9);
        let floored = overridden.with_vector_cell_counts(0, 0);
        assert_eq!(user_vector_cell_count(&floored), 1);
        assert_eq!(hidden_vector_cell_count(&floored), 1);
    }

    /// Read-only consumer memory mode (`summary_centroids_from_superfiles`):
    /// after a drain, a consumer reopened with the mode on must (a) hold
    /// the hidden summaries without resident fp32 centroids and (b) return
    /// exactly the hits of a mode-off consumer — the deferred admit rescore
    /// reads centroid regions from the superfiles through the reader cache.
    #[test]
    fn stripped_summary_consumer_matches_resident_hits() {
        use std::sync::Arc;

        use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray};
        use arrow_schema::{DataType, Field, Schema};

        use crate::{
            superfile::{
                builder::{FtsConfig, VectorConfig},
                reader::VectorSearchOptions,
                vector::{distance::Metric, rerank_codec::RerankCodec},
            },
            supertable::query::SuperfileHit,
        };

        // Rows across the planted one-hot directions (three per direction
        // at dim=16), so each cosine cluster has a few members.
        const N_DOCS: usize = 48;
        // Top-k compared across the two consumer modes.
        const TOP_K: usize = 5;
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
        let make_options = |from_superfiles: bool| {
            SupertableOptions::new(
                schema.clone(),
                vec![FtsConfig {
                    column: "title".into(),
                    positions: false,
                }],
                vec![VectorConfig {
                    column: "emb".into(),
                    dim,
                    n_cent: 4,
                    rot_seed: 7,
                    metric: Metric::Cosine,
                    rerank_codec: RerankCodec::Sq8FixedResidual,
                    provided_centroids: None,
                }],
                Some(crate::test_helpers::default_tokenizer()),
            )
            .expect("valid options")
            .with_storage(Arc::clone(&storage))
            .with_writer_pool(Arc::clone(&pool))
            .with_summary_centroids_from_superfiles(from_superfiles)
        };

        // One-hot planted directions (i % dim) — separable cosine clusters.
        let st = Supertable::create(make_options(false)).expect("create");
        let titles =
            LargeStringArray::from((0..N_DOCS).map(|i| format!("doc {i}")).collect::<Vec<_>>());
        let mut flat = Vec::with_capacity(N_DOCS * dim);
        for i in 0..N_DOCS {
            for d in 0..dim {
                flat.push(if d == i % dim { 1.0f32 } else { 0.0 });
            }
        }
        let fsl = FixedSizeListArray::new(
            item_field.clone(),
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
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
        drop(w);

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        q[1] = 0.05;
        let ids = |hits: &[SuperfileHit]| {
            hits.iter()
                .map(|h| (h.superfile, h.local_doc_id, h.stable_id))
                .collect::<Vec<_>>()
        };

        // Pre-drain: the user wave serves the query, and a mode-on consumer
        // hydrates the user manifest from routing parts (no fp32 download) —
        // the deferred rescore reads user superfile centroid regions.
        let pre_baseline = Supertable::open(make_options(false)).expect("pre-drain mode-off");
        let pre_base_hits = pre_baseline
            .reader()
            .vector_hits("emb", &q, TOP_K, VectorSearchOptions::new(), None)
            .expect("pre-drain baseline hits");
        assert!(!pre_base_hits.is_empty(), "pre-drain baseline returns hits");
        drop(pre_baseline);
        let pre_stripped = Supertable::open(make_options(true)).expect("pre-drain mode-on");
        let pre_stripped_reader = pre_stripped.reader();
        let user_manifest = pre_stripped_reader.manifest();
        let user_part_entries = user_manifest.get_all_list_entries();
        assert!(
            !user_part_entries.is_empty(),
            "pre-drain user manifest must carry parts (routing hydration under test)"
        );
        assert!(
            user_part_entries
                .iter()
                .all(|entry| entry.routing.is_some()),
            "commits must stamp a routing sibling on every user part"
        );
        // Routing-part decode must have dropped the user fp32 — otherwise
        // the parity check below never exercises the full-part rescore.
        let saw_stripped_user_cell = user_manifest.superfiles.iter().any(|entry| {
            entry.vector_summary.values().any(|vs| {
                vs.cells
                    .iter()
                    .any(|cell| cell.clusters.n_cent > 0 && !cell.clusters.vectors_resident())
            })
        });
        assert!(
            saw_stripped_user_cell,
            "mode-on consumer must hydrate stripped user summaries from routing parts"
        );
        let pre_stripped_hits = pre_stripped
            .reader()
            .vector_hits("emb", &q, TOP_K, VectorSearchOptions::new(), None)
            .expect("pre-drain stripped hits");
        assert_eq!(
            ids(&pre_stripped_hits),
            ids(&pre_base_hits),
            "pre-drain: routing-part hydration must reproduce resident hits"
        );
        drop(pre_stripped);

        st.drain_vectors_to_cells_sync().expect("drain to cells");
        drop(st);

        let baseline = Supertable::open(make_options(false)).expect("reopen mode-off");
        let base_hits = baseline
            .reader()
            .vector_hits("emb", &q, TOP_K, VectorSearchOptions::new(), None)
            .expect("baseline hits");
        assert!(!base_hits.is_empty(), "baseline must return hits");
        drop(baseline);

        let stripped = Supertable::open(make_options(true)).expect("reopen mode-on");
        let hidden = stripped
            .reader()
            .vector_index_table()
            .expect("hidden index")
            .clone();
        let hidden_reader = hidden.reader();
        let hidden_manifest = hidden_reader.manifest();
        assert!(
            hidden_manifest.slow_vector_state_blob().is_some(),
            "drain must publish the slow-state ref (routing-shaped blob)"
        );
        assert!(
            hidden_manifest.slow_vector_state_centroids_blob().is_some(),
            "drain must publish the centroid-section ref — exact rescores read it"
        );
        let mut saw_stripped_summary = false;
        for entry in hidden_manifest.superfiles.iter() {
            for vs in entry.vector_summary.values() {
                for cell in &vs.cells {
                    if cell.clusters.n_cent > 0 {
                        assert!(
                            !cell.clusters.vectors_resident(),
                            "hidden summary fp32 must be dropped in mode-on consumers"
                        );
                        saw_stripped_summary = true;
                    }
                }
            }
        }
        assert!(
            saw_stripped_summary,
            "drained hidden manifest must carry cell summaries"
        );

        let stripped_hits = stripped
            .reader()
            .vector_hits("emb", &q, TOP_K, VectorSearchOptions::new(), None)
            .expect("stripped-mode hits");
        assert_eq!(
            ids(&stripped_hits),
            ids(&base_hits),
            "deferred rescore must reproduce the resident-mode hit set"
        );
        for (a, b) in stripped_hits.iter().zip(&base_hits) {
            assert!(
                (a.score - b.score).abs() <= 1e-5 * (1.0 + b.score.abs()),
                "scores must agree: {} vs {}",
                a.score,
                b.score
            );
        }
    }

    /// Plan contract: splice-mode drain is a separate identity path. Routing
    /// keeps each local cluster verbatim (no re-kmeans); fixed residual
    /// payloads must match the user-side bytes by stable `_id`.
    #[test]
    fn splice_drain_preserves_fixed_residual_payloads() {
        use std::sync::Arc;

        use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray};
        use arrow_schema::{DataType, Field, Schema};

        use crate::{
            config::DrainConsolidate,
            superfile::{
                builder::{FtsConfig, VectorConfig},
                vector::{distance::Metric, rerank_codec::RerankCodec},
            },
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
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8FixedResidual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool)
        .with_drain_consolidate(DrainConsolidate::Splice)
        .with_drain_batch_superfiles(-1);

        let st = Supertable::create(options).expect("create");
        let titles = LargeStringArray::from(vec!["a", "b", "c", "d"]);
        // Distinct axis-aligned vectors so local clusters are non-trivial.
        let mut flat_vals = vec![0.0f32; 4 * dim];
        for (row, axis) in [0usize, 1, 2, 3].into_iter().enumerate() {
            flat_vals[row * dim + axis] = 1.0;
        }
        let flat = Float32Array::from(flat_vals);
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
        drop(w);

        let user_payloads = rerank_payloads_by_stable_id(&st);
        assert!(!user_payloads.is_empty(), "user rows must have payloads");
        st.drain_vectors_to_cells_sync().expect("splice drain");
        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden index")
            .clone();
        assert!(
            hidden.reader().n_superfiles() > 0,
            "splice drain must populate hidden cells"
        );
        let hidden_payloads = rerank_payloads_by_stable_id(&hidden);
        assert_eq!(
            hidden_payloads, user_payloads,
            "fixed residual payloads must survive splice drain byte-for-byte"
        );
    }

    /// A Kmeans-consolidate drain re-clusters the user rows through
    /// `materialized_ivf_rows_in_doc_order`; every doc's stable id must survive
    /// into the hidden index. Payloads may be re-quantized under the new
    /// centroids, so compare the id set rather than bytes.
    #[test]
    fn kmeans_drain_preserves_all_stable_ids() {
        use std::sync::Arc;

        use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray};
        use arrow_schema::{DataType, Field, Schema};

        use crate::{
            config::DrainConsolidate,
            superfile::{
                builder::{FtsConfig, VectorConfig},
                vector::{distance::Metric, rerank_codec::RerankCodec},
            },
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
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool)
        .with_drain_consolidate(DrainConsolidate::Kmeans)
        .with_drain_batch_superfiles(-1);

        let st = Supertable::create(options).expect("create");
        let titles = LargeStringArray::from(vec!["a", "b", "c", "d"]);
        let mut flat_vals = vec![0.0f32; 4 * dim];
        for (row, axis) in [0usize, 1, 2, 3].into_iter().enumerate() {
            flat_vals[row * dim + axis] = 1.0;
        }
        let flat = Float32Array::from(flat_vals);
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
        drop(w);

        let user_payloads = rerank_payloads_by_stable_id(&st);
        assert_eq!(user_payloads.len(), 4, "four user rows before drain");
        st.drain_vectors_to_cells_sync().expect("kmeans drain");

        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden index")
            .clone();
        assert!(
            hidden.reader().n_superfiles() > 0,
            "kmeans drain must populate hidden cells"
        );
        let hidden_payloads = rerank_payloads_by_stable_id(&hidden);
        let mut user_ids: Vec<i128> = user_payloads.keys().copied().collect();
        let mut hidden_ids: Vec<i128> = hidden_payloads.keys().copied().collect();
        user_ids.sort_unstable();
        hidden_ids.sort_unstable();
        assert_eq!(
            hidden_ids, user_ids,
            "kmeans drain preserves every doc's stable id"
        );
    }

    /// The default k-means drain over an Sq8Residual index materializes user
    /// rows in doc order (the `materialized_ivf_rows_in_doc_order` path, which
    /// only runs for a residual codec) and populates the hidden cell index.
    #[test]
    fn kmeans_drain_over_residual_index_materializes_and_populates_cells() {
        use std::sync::Arc;

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
        // Sq8Residual (not Fixed) + default consolidate (k-means) is the combo
        // that routes the drain through materialized_ivf_rows_in_doc_order.
        let options = SupertableOptions::new(
            schema.clone(),
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(Arc::clone(&storage))
        .with_writer_pool(pool);

        let st = Supertable::create(options).expect("create");
        let titles = LargeStringArray::from(vec!["a", "b", "c", "d"]);
        let mut flat = vec![0.0f32; 4 * dim];
        for row in 0..4 {
            flat[row * dim + row] = 1.0;
        }
        let fsl = FixedSizeListArray::new(
            item_field,
            dim as i32,
            Arc::new(Float32Array::from(flat)),
            None,
        );
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
        drop(w);

        st.drain_vectors_to_cells_sync()
            .expect("k-means residual drain");

        // The residual-materialized rows landed in the hidden cell index.
        let (total, max_per_cell) = st
            .hidden_vector_superfile_stats()
            .expect("hidden stats after drain");
        assert!(total > 0, "k-means drain must populate hidden cells");
        assert!(
            max_per_cell > 0 && max_per_cell <= total,
            "max-per-cell {max_per_cell} in 1..={total}",
        );
    }

    /// After a splice drain into the hidden vector index, the reader's derived-
    /// state accessors report a live hidden index: a storage prefix is stamped,
    /// the hidden-superfile stats are non-empty, the user superfiles carry real
    /// index bytes, and the disk cache warms without timing out.
    #[test]
    fn drained_reader_reports_hidden_index_and_warms() {
        use std::{sync::Arc, time::Duration};

        use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray};
        use arrow_schema::{DataType, Field, Schema};

        use crate::{
            config::DrainConsolidate,
            superfile::{
                builder::{FtsConfig, VectorConfig},
                vector::{distance::Metric, rerank_codec::RerankCodec},
            },
            test_helpers::default_disk_cache,
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
        let storage_dir = TempDir::new().expect("storage tempdir");
        let cache_dir = TempDir::new().expect("cache tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));
        let disk_cache = default_disk_cache(Arc::clone(&storage), cache_dir.path());

        let options = SupertableOptions::new(
            schema.clone(),
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8FixedResidual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(Arc::clone(&storage))
        .with_writer_pool(pool)
        .with_disk_cache(Arc::clone(&disk_cache))
        .with_drain_consolidate(DrainConsolidate::Splice)
        .with_drain_batch_superfiles(-1);

        let st = Supertable::create(options).expect("create");
        let titles = LargeStringArray::from(vec!["a", "b", "c", "d"]);
        let mut flat_vals = vec![0.0f32; 4 * dim];
        for (row, axis) in [0usize, 1, 2, 3].into_iter().enumerate() {
            flat_vals[row * dim + axis] = 1.0;
        }
        let flat = Float32Array::from(flat_vals);
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
        drop(w);

        st.drain_vectors_to_cells_sync().expect("splice drain");

        // A hidden index now exists → a storage prefix is stamped on the
        // user manifest.
        let prefix = st
            .vector_index_storage_prefix()
            .expect("hidden vector index prefix present after drain");
        assert!(!prefix.is_empty(), "storage prefix must be non-empty");

        // The hidden table carries at least one cell superfile, and the
        // busiest cell holds at least one.
        let (total, max_per_cell) = st
            .hidden_vector_superfile_stats()
            .expect("hidden vector stats present after drain");
        assert!(
            total > 0,
            "drain must populate hidden superfiles, got {total}"
        );
        assert!(
            max_per_cell > 0 && max_per_cell <= total,
            "max-per-cell {max_per_cell} must be in 1..={total}"
        );

        // The user superfiles load and report real per-superfile index bytes.
        let (n_superfiles, index_bytes) = st
            .reader()
            .load_superfile_storage_stats()
            .expect("load superfile storage stats");
        assert!(n_superfiles > 0, "user table has committed superfiles");
        assert!(
            index_bytes > 0,
            "committed vector superfiles carry index bytes"
        );

        // The disk cache warms within the timeout.
        st.wait_until_warm(Duration::from_secs(5))
            .expect("disk cache warms without timing out");
    }

    /// An engine-managed (auto-sized) cache budget must be raised at open
    /// to the table's real on-storage footprint — user superfiles plus the
    /// hidden vector index — while an explicit budget is never changed.
    #[test]
    fn open_reconciles_auto_sized_cache_budget_with_footprint() {
        use arrow_array::{Array, FixedSizeListArray, Float32Array};

        use crate::{
            superfile::{
                builder::VectorConfig,
                vector::{distance::Metric, rerank_codec::RerankCodec},
            },
            supertable::reader_cache::{DiskCacheConfig, DiskCacheStore},
        };

        /// Deliberately smaller than any committed superfile, so an
        /// unreconciled budget is distinguishable from a raised one.
        const TINY_BUDGET_BYTES: u64 = 4;

        let dim = 16usize;
        let n_rows = 64usize;
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vec_schema = Arc::new(Schema::new(vec![Field::new(
            "emb",
            DataType::FixedSizeList(item_field.clone(), dim as i32),
            false,
        )]));
        let storage_dir = TempDir::new().expect("storage tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));
        let make_options = || {
            SupertableOptions::new(
                vec_schema.clone(),
                vec![],
                vec![VectorConfig {
                    column: "emb".into(),
                    dim,
                    n_cent: 4,
                    rot_seed: 7,
                    metric: Metric::Cosine,
                    rerank_codec: RerankCodec::Sq8Residual,
                    provided_centroids: None,
                }],
                None,
            )
            .expect("valid options")
            .with_storage(Arc::clone(&storage))
        };

        // Producer: commit vectors and drain them into the hidden index so
        // the on-storage footprint spans both tables.
        {
            let producer = Supertable::create(make_options()).expect("create");
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
                vec_schema.clone(),
                vec![Arc::new(fsl) as Arc<dyn Array>],
            )
            .expect("batch");
            let mut w = producer.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
            producer.drain_vectors_to_cells_sync().expect("drain");
        }

        // Auto-sized consumer: a tiny engine-managed budget must be raised
        // to at least the footprint by the open-time reconcile.
        let auto_cache_dir = TempDir::new().expect("cache tempdir");
        let auto_cache = DiskCacheStore::new_unpinned(
            Arc::clone(&storage),
            DiskCacheConfig {
                cache_root: auto_cache_dir.path().to_path_buf(),
                disk_budget_bytes: TINY_BUDGET_BYTES,
                mmap_cold_threshold_secs: 0,
                mmap_sweep_interval_secs: 0,
                ..Default::default()
            },
        )
        .expect("auto cache");
        auto_cache.mark_budget_auto_sized();
        let st = Supertable::open(make_options().with_disk_cache(Arc::clone(&auto_cache)))
            .expect("open with auto-sized cache");
        let footprint = st.on_storage_footprint_bytes();
        assert!(footprint > 0, "committed + drained table has a footprint");
        assert!(
            auto_cache.disk_budget_bytes() >= footprint,
            "auto-sized budget {} must cover the footprint {footprint}",
            auto_cache.disk_budget_bytes(),
        );
        drop(st);

        // Explicit-budget consumer: the same open leaves the budget alone.
        let explicit_cache_dir = TempDir::new().expect("cache tempdir");
        let explicit_cache = DiskCacheStore::new_unpinned(
            Arc::clone(&storage),
            DiskCacheConfig {
                cache_root: explicit_cache_dir.path().to_path_buf(),
                disk_budget_bytes: TINY_BUDGET_BYTES,
                mmap_cold_threshold_secs: 0,
                mmap_sweep_interval_secs: 0,
                ..Default::default()
            },
        )
        .expect("explicit cache");
        let st = Supertable::open(make_options().with_disk_cache(Arc::clone(&explicit_cache)))
            .expect("open with explicit cache");
        assert_eq!(
            explicit_cache.disk_budget_bytes(),
            TINY_BUDGET_BYTES,
            "explicit budgets are warned about, never changed"
        );
        drop(st);
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
                    positions: false,
                }],
                vec![VectorConfig {
                    column: "emb".into(),
                    dim,
                    n_cent: 4,
                    rot_seed: 7,
                    metric: Metric::Cosine,
                    rerank_codec: RerankCodec::Sq8Residual,
                    provided_centroids: None,
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
            // Phase B: drain the user superfiles into the hidden cells (no
            // dual-write), so the consumer below has real cell superfiles to
            // make resident.
            producer
                .drain_vectors_to_cells_sync()
                .expect("drain user superfiles into hidden cells");
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

    /// Build a vector-only supertable, append `commits` batches of
    /// `rows_per_commit` unit vectors (draining into hidden cells after each
    /// commit when `drain_each`), then reopen a fresh consumer whose disk
    /// cache is in lazy-foreground mode — the exact reader state a query
    /// fan-out leaves behind, which compaction must still read eagerly.
    /// Returns the temp dirs (kept alive by the caller) and the consumer.
    fn vector_consumer_with_lazy_cache(
        dim: usize,
        rows_per_commit: usize,
        commits: usize,
        drain_each: bool,
    ) -> (TempDir, TempDir, Supertable) {
        use arrow_array::{Array, FixedSizeListArray, Float32Array};

        use crate::{
            superfile::{
                builder::VectorConfig,
                vector::{distance::Metric, rerank_codec::RerankCodec},
            },
            supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
        };

        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "emb",
            DataType::FixedSizeList(item_field.clone(), dim as i32),
            false,
        )]));
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
                vec![],
                vec![VectorConfig {
                    column: "emb".into(),
                    dim,
                    n_cent: 4,
                    rot_seed: 7,
                    metric: Metric::Cosine,
                    rerank_codec: RerankCodec::Sq8Residual,
                    provided_centroids: None,
                }],
                None,
            )
            .expect("valid options")
            .with_storage(Arc::clone(&storage))
            .with_writer_pool(Arc::clone(&pool))
        };

        {
            let producer = Supertable::create(make_options()).expect("create");
            for _ in 0..commits {
                let flat = vec![1.0f32; rows_per_commit * dim];
                let fsl = FixedSizeListArray::new(
                    item_field.clone(),
                    dim as i32,
                    Arc::new(Float32Array::from(flat)),
                    None,
                );
                let batch = arrow_array::RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(fsl) as Arc<dyn Array>],
                )
                .expect("batch");
                let mut w = producer.writer().expect("writer");
                w.append(&batch).expect("append");
                w.commit().expect("commit");
                if drain_each {
                    producer.drain_vectors_to_cells_sync().expect("drain");
                }
            }
        }

        let cfg = DiskCacheConfig {
            cache_root: cache_dir.path().to_path_buf(),
            disk_budget_bytes: 1 << 30,
            cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
            cold_fetch_streams: 4,
            cold_fetch_chunk_bytes: 1 << 20,
            mmap_cold_threshold_secs: 0,
            mmap_sweep_interval_secs: 0,
            eviction: Box::new(LruPolicy::new()),
            verify_crc_on_open: true,
            ..Default::default()
        };
        let pinned_fn: Arc<dyn Fn() -> std::collections::HashSet<SuperfileUri> + Send + Sync> =
            Arc::new(std::collections::HashSet::new);
        let cache = DiskCacheStore::new(Arc::clone(&storage), cfg, pinned_fn).expect("cache");
        let consumer =
            Supertable::open(make_options().with_disk_cache(Arc::clone(&cache))).expect("open");
        (storage_dir, cache_dir, consumer)
    }

    const COMPACTION_TEST_SETTINGS: crate::config::CompactionSettings =
        crate::config::CompactionSettings {
            target_superfile_size_mb: 1,
            min_fill_percent: 1,
            max_memory_mb: 64,
            stale_seal_timeout_ms: crate::config::DEFAULT_STALE_SEAL_TIMEOUT_MS,
        };

    /// Regression guard for optimize/compaction on the hidden vector index:
    /// after lazy hidden-index reads, hidden compaction must still open every
    /// input as an eager reader and merge without `RecordBatch` read failures.
    #[test]
    fn hidden_compaction_succeeds_after_lazy_hidden_reads() {
        use crate::superfile::reader::VectorSearchOptions;

        const DIM: usize = 16;
        let (_storage_dir, _cache_dir, consumer) =
            vector_consumer_with_lazy_cache(DIM, 5_000, 3, true);

        let hidden = consumer
            .reader()
            .vector_index_table()
            .expect("hidden index")
            .clone();
        let mut per_cell: HashMap<Vec<u8>, usize> = HashMap::new();
        for entry in &hidden.reader().manifest().superfiles {
            *per_cell.entry(entry.partition_key.clone()).or_insert(0) += 1;
        }
        assert!(
            per_cell.values().copied().max().unwrap_or(0) >= 2,
            "expected >=2 hidden superfiles in at least one cell"
        );

        // Populate hidden readers through the lazy query path before compaction.
        let query = vec![1.0f32; DIM];
        let hits = consumer
            .reader()
            .vector_hits("emb", &query, 10, VectorSearchOptions::new(), None)
            .expect("vector search");
        assert!(!hits.is_empty(), "hidden index should return vector hits");

        hidden
            .compact(&COMPACTION_TEST_SETTINGS)
            .expect("hidden compaction should succeed after lazy reads");
    }

    /// Regression guard for user-table compaction after pre-drain lazy reads.
    /// The pre-drain vector query path opens user superfiles lazily through the
    /// disk cache; subsequent compaction must still read full record batches.
    #[test]
    fn user_compaction_succeeds_after_pre_drain_lazy_reads() {
        use crate::superfile::reader::VectorSearchOptions;

        const DIM: usize = 1024;
        let (_storage_dir, _cache_dir, consumer) =
            vector_consumer_with_lazy_cache(DIM, 512, 4, false);

        // Pre-drain query path: user superfiles are read lazily.
        let query = vec![1.0f32; DIM];
        let hits = consumer
            .reader()
            .vector_hits("emb", &query, 10, VectorSearchOptions::new(), None)
            .expect("vector search");
        assert!(!hits.is_empty(), "pre-drain user search should return hits");

        consumer
            .compact(&COMPACTION_TEST_SETTINGS)
            .expect("user compaction should succeed after lazy pre-drain reads");
    }

    /// SQL-shaped tables (multi-column FTS + Sq8 vector) must survive
    /// `optimize()` after lazy disk-cache reads — the same path the SQL
    /// supertable bench takes (pre-compact warm/cold → optimize).
    #[test]
    fn sql_shaped_optimize_after_lazy_reads() {
        use arrow_array::{
            Array, FixedSizeListArray, Float32Array, Int64Array, LargeStringArray, RecordBatch,
        };

        use crate::{
            superfile::{
                builder::VectorConfig,
                vector::{distance::Metric, rerank_codec::RerankCodec},
            },
            supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
        };

        const DIM: usize = 16;
        const ROWS: usize = 64;
        const COMMITS: usize = 4;

        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("bucket", DataType::LargeUtf8, false),
            Field::new("key", DataType::LargeUtf8, false),
            Field::new("category", DataType::LargeUtf8, false),
            Field::new("rating", DataType::Int64, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(item_field.clone(), DIM as i32),
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
                vec![
                    FtsConfig {
                        column: "title".into(),
                        positions: false,
                    },
                    FtsConfig {
                        column: "bucket".into(),
                        positions: false,
                    },
                    FtsConfig {
                        column: "key".into(),
                        positions: false,
                    },
                    FtsConfig {
                        column: "category".into(),
                        positions: false,
                    },
                ],
                vec![VectorConfig {
                    column: "emb".into(),
                    dim: DIM,
                    n_cent: 4,
                    rot_seed: 7,
                    metric: Metric::Cosine,
                    rerank_codec: RerankCodec::Sq8Residual,
                    provided_centroids: None,
                }],
                Some(default_tokenizer()),
            )
            .expect("valid options")
            .with_storage(Arc::clone(&storage))
            .with_writer_pool(Arc::clone(&pool))
        };

        {
            let producer = Supertable::create(make_options()).expect("create");
            for c in 0..COMMITS {
                let titles: Vec<String> = (0..ROWS).map(|i| format!("doc {c} {i}")).collect();
                let buckets: Vec<String> = (0..ROWS).map(|i| format!("b{}", i % 10)).collect();
                let keys: Vec<String> = (0..ROWS).map(|i| format!("k{c}_{i}")).collect();
                let cats: Vec<String> = (0..ROWS)
                    .map(|i| if i % 2 == 0 { "cat" } else { "dog" }.to_string())
                    .collect();
                let ratings: Vec<i64> = (0..ROWS).map(|i| i as i64).collect();
                let flat = vec![1.0f32; ROWS * DIM];
                let fsl = FixedSizeListArray::new(
                    item_field.clone(),
                    DIM as i32,
                    Arc::new(Float32Array::from(flat)),
                    None,
                );
                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(LargeStringArray::from(
                            titles.iter().map(String::as_str).collect::<Vec<_>>(),
                        )) as Arc<dyn Array>,
                        Arc::new(LargeStringArray::from(
                            buckets.iter().map(String::as_str).collect::<Vec<_>>(),
                        )) as Arc<dyn Array>,
                        Arc::new(LargeStringArray::from(
                            keys.iter().map(String::as_str).collect::<Vec<_>>(),
                        )) as Arc<dyn Array>,
                        Arc::new(LargeStringArray::from(
                            cats.iter().map(String::as_str).collect::<Vec<_>>(),
                        )) as Arc<dyn Array>,
                        Arc::new(Int64Array::from(ratings)) as Arc<dyn Array>,
                        Arc::new(fsl) as Arc<dyn Array>,
                    ],
                )
                .expect("batch");
                let mut w = producer.writer().expect("writer");
                w.append(&batch).expect("append");
                w.commit().expect("commit");
            }
        }

        let cfg = DiskCacheConfig {
            cache_root: cache_dir.path().to_path_buf(),
            disk_budget_bytes: 1 << 30,
            cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
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
        let consumer =
            Supertable::open(make_options().with_disk_cache(Arc::clone(&cache))).expect("open");

        // Exercise the lazy FTS path before optimize (mirrors the SQL bench's
        // pre-compact warm/cold queries against a disk-cache consumer).
        use crate::superfile::fts::reader::BoolMode;
        let hits = consumer
            .bm25_search("title", "doc", 5, BoolMode::Or, None)
            .expect("bm25 pre-optimize");
        assert!(!hits.is_empty(), "pre-optimize FTS should return hits");

        consumer
            .optimize(&OptimizeOptions::compact(COMPACTION_TEST_SETTINGS))
            .expect("sql-shaped optimize after lazy reads");

        let hits_after = consumer
            .bm25_search("title", "doc", 5, BoolMode::Or, None)
            .expect("bm25 post-optimize");
        assert!(
            !hits_after.is_empty(),
            "FTS must remain searchable after Sq8+FTS optimize"
        );
    }

    /// Each drain APPENDS packed shard object(s) to the hidden manifest (no
    /// removals — the user superfiles stay as the durable source). Draining
    /// across successive commits accumulates multiple files under the same
    /// partition key, which compaction later collapses.
    #[test]
    fn drain_appends_multiple_files_per_cell() {
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
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool);
        let st = Supertable::create(options).expect("create");

        for commit in 0..2 {
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
            // Phase B: drain after each commit; each drain appends packed shard files.
            st.drain_vectors_to_cells_sync().expect("drain to cells");
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
            "each drain should append a packed shard file, got {max_visible}"
        );
    }

    /// After a drain, `open_all_superfiles` force-opens every user + hidden
    /// reader without error, and `hidden_cell_stable_id_sets` audits the drained
    /// cells — the sum of per-cell stable ids equals the ingested doc count.
    #[test]
    fn open_all_superfiles_and_hidden_stable_id_audit() {
        use std::sync::Arc;

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
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool);
        let st = Supertable::create(options).expect("create");

        const N: usize = 6;
        let titles = LargeStringArray::from((0..N).map(|i| format!("doc-{i}")).collect::<Vec<_>>());
        let flat = Float32Array::from(vec![1.0f32; N * dim]);
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
        drop(w);
        st.drain_vectors_to_cells_sync().expect("drain");

        // Cold-open every reader (user + hidden); must not error.
        st.open_all_superfiles();

        // Audit: the drained cells' inline stable ids cover every doc.
        let sets = st
            .hidden_cell_stable_id_sets()
            .expect("post-drain hidden cells expose stable-id sets");
        let total: usize = sets.iter().map(|(_, ids)| ids.len()).sum();
        assert_eq!(
            total, N,
            "every drained doc carries a stable id in some cell"
        );
    }

    /// Splitting a cell whose packed shard also holds *other* cells must
    /// republish those neighbours intact (the keep-cells branch). Ingest two
    /// vector directions (→ two cells in one shard), split the busiest, and
    /// confirm both directions still resolve afterward.
    #[test]
    fn split_overflow_cell_republishes_neighbour_cells() {
        use std::sync::Arc;

        use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray};
        use arrow_schema::{DataType, Field, Schema};

        use crate::{
            superfile::{
                builder::{FtsConfig, VectorConfig},
                vector::{distance::Metric, rerank_codec::RerankCodec},
            },
            supertable::{manifest::list::PartitionStrategy, writer::split_overflow_cell},
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
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool);
        let st = Supertable::create(options).expect("create");

        // 8 rows at e_0 and 8 at e_1 → two distinct cells packed in one shard.
        const N: usize = 16;
        let titles = LargeStringArray::from((0..N).map(|i| format!("doc-{i}")).collect::<Vec<_>>());
        let mut flat = vec![0.0f32; N * dim];
        for r in 0..N {
            flat[r * dim + usize::from(r >= N / 2)] = 1.0; // first half e_0, second half e_1
        }
        let fsl = FixedSizeListArray::new(
            item_field.clone(),
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
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
        st.drain_vectors_to_cells_sync().expect("drain to cells");

        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden index")
            .clone();
        // Two populated cells (the two directions). Split the busiest; the
        // other populated cell is the neighbour whose count must survive.
        let (busiest, neighbour, neighbour_count, n_cent_before) =
            match hidden.reader().manifest().get_partition_strategy() {
                PartitionStrategy::VectorCell { clusters, .. } => {
                    let mut populated: Vec<u32> = (0..clusters.n_cent)
                        .filter(|&c| clusters.counts[c as usize] > 0)
                        .collect();
                    assert!(
                        populated.len() >= 2,
                        "two directions must drain into two cells, got {:?}",
                        clusters.counts
                    );
                    populated.sort_by_key(|&c| std::cmp::Reverse(clusters.counts[c as usize]));
                    let busiest = populated[0];
                    let neighbour = populated[1];
                    (
                        busiest,
                        neighbour,
                        clusters.counts[neighbour as usize],
                        clusters.n_cent,
                    )
                }
                other => panic!("hidden must be VectorCell after drain, got {other:?}"),
            };

        hidden
            .block_on_query(split_overflow_cell(hidden.inner().clone(), busiest))
            .expect("split");

        // The split grows the grid by one sub-cell and republishes the
        // neighbour cell untouched — its doc count is unchanged.
        match hidden.reader().manifest().get_partition_strategy() {
            PartitionStrategy::VectorCell { clusters, .. } => {
                assert_eq!(
                    clusters.n_cent,
                    n_cent_before + 1,
                    "split adds one sub-cell"
                );
                assert_eq!(
                    clusters.counts[neighbour as usize], neighbour_count,
                    "the republished neighbour cell keeps its docs"
                );
            }
            other => panic!("still VectorCell, got {other:?}"),
        }
    }

    /// Directly exercises the over-cap cell split (`split_overflow_cell`). The
    /// normal `optimize` path only reaches it once a cell passes the 500k
    /// `cell_split_doc_cap`; calling the inner routine on a drained cell covers
    /// the extract → split → rebuild → atomic-swap chain without that volume. The
    /// split must add a sub-cell to the grid and lose no doc.
    #[test]
    fn split_overflow_cell_grows_grid_and_preserves_docs() {
        use std::sync::Arc;

        use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray};
        use arrow_schema::{DataType, Field, Schema};

        use crate::{
            superfile::{
                builder::{FtsConfig, VectorConfig},
                reader::VectorSearchOptions,
                vector::{distance::Metric, rerank_codec::RerankCodec},
            },
            supertable::{manifest::list::PartitionStrategy, writer::split_overflow_cell},
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
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool);
        let st = Supertable::create(options).expect("create");

        // Identical embeddings route every doc into one global cell; drain them
        // into the hidden per-cell index so a single real cell holds all N.
        const N: usize = 6;
        let titles = LargeStringArray::from((0..N).map(|i| format!("doc-{i}")).collect::<Vec<_>>());
        let flat = Float32Array::from(vec![1.0f32; N * dim]);
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
        st.drain_vectors_to_cells_sync().expect("drain to cells");

        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden index")
            .clone();

        // The most-populated cell in the hidden grid holds all N docs.
        let (split_cell, n_cent_before, docs_in_cell) =
            match hidden.reader().manifest().get_partition_strategy() {
                PartitionStrategy::VectorCell { clusters, .. } => {
                    let cell = (0..clusters.n_cent)
                        .max_by_key(|&c| clusters.counts.get(c as usize).copied().unwrap_or(0))
                        .expect("at least one cell");
                    (cell, clusters.n_cent, clusters.counts[cell as usize])
                }
                other => panic!("hidden index must be VectorCell after drain, got {other:?}"),
            };
        assert!(docs_in_cell >= 2, "the split cell needs at least two docs");

        // Sanity: the drained docs are retrievable before the split.
        let q = vec![1.0f32; dim];
        let hits_before = st
            .reader()
            .vector_hits("emb", &q, N, VectorSearchOptions::new(), None)
            .expect("pre-split search");
        assert!(!hits_before.is_empty(), "docs retrievable before split");

        // Split the over-cap cell directly (bypasses the 500k cap gate).
        let split_outcome = hidden
            .block_on_query(split_overflow_cell(hidden.inner().clone(), split_cell))
            .expect("split");
        assert!(
            split_outcome.is_some(),
            "live rows present, split must commit"
        );

        // The grid gained a sub-cell, and the two sub-cells together account for
        // exactly the live docs (`split_cell` keeps its id; the new cell is
        // appended at the old `n_cent`). Routing-independent — it reads the
        // counts the split re-derives from the actual live rows, which also
        // corrects the pre-split grid count (that count can lag the true total).
        match hidden.reader().manifest().get_partition_strategy() {
            PartitionStrategy::VectorCell { clusters, .. } => {
                assert_eq!(
                    clusters.n_cent,
                    n_cent_before + 1,
                    "split inserts one sub-centroid into the grid"
                );
                let kept = clusters.counts[split_cell as usize];
                let moved = clusters.counts[n_cent_before as usize];
                assert_eq!(
                    kept + moved,
                    N as u32,
                    "the two sub-cells must account for every live doc"
                );
            }
            other => panic!("still VectorCell after split, got {other:?}"),
        }
    }

    /// With writer_pool=N>1 and multiple touched cells, drain publishes at most
    /// N packed shard objects and stamps partition_hint = shard_id (cell % N).
    #[test]
    fn drain_packs_cells_into_at_most_writer_pool_shards() {
        use std::{collections::HashSet, sync::Arc};

        use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray};
        use arrow_schema::{DataType, Field, Schema};

        use crate::superfile::{
            builder::{FtsConfig, VectorConfig},
            vector::{distance::Metric, layout::VectorLayout, rerank_codec::RerankCodec},
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
        const POOL: usize = 2;
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(POOL)
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
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool);
        let st = Supertable::create(options).expect("create");

        // Train the initial grid from distinct directions in ONE commit. Using
        // one row per commit would train the immutable grid from the first
        // single row, leaving every centroid identical and failing to exercise
        // multi-cell shard packing.
        let titles =
            LargeStringArray::from((0..8usize).map(|i| format!("doc{i}")).collect::<Vec<_>>());
        let mut vectors = vec![0.0f32; 8 * dim];
        for i in 0..8usize {
            vectors[i * dim + (i % dim)] = 1.0;
        }
        let flat = Float32Array::from(vectors);
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
        st.drain_vectors_to_cells_sync().expect("drain");

        let hidden = st.reader().vector_index_table().expect("hidden").clone();
        assert_eq!(
            hidden.options().writer_pool.current_num_threads(),
            POOL,
            "hidden drain must inherit the user table's configured writer pool"
        );
        let hidden_reader = hidden.reader();
        let manifest = hidden_reader.manifest();
        let n_objects = manifest.superfiles.len();
        assert_eq!(
            n_objects, POOL,
            "five populated cells span both cell % {POOL} worker shards"
        );
        let mut hints = HashSet::new();
        for entry in manifest.superfiles.iter() {
            assert_eq!(entry.vector_layout, VectorLayout::MultiCellIvf);
            let hint = entry.partition_hint.expect("shard partition_hint");
            assert!(
                (hint as usize) < POOL,
                "partition_hint={hint} must be a shard id in 0..{POOL}"
            );
            hints.insert(hint);
            // Each packed object should open with a non-empty cell directory.
            assert!(
                !entry
                    .vector_summary
                    .get("emb")
                    .map(|v| v.cells.iter().all(|cell| cell.clusters.is_empty()))
                    .unwrap_or(true),
                "packed shard missing cluster summary"
            );
        }
        assert_eq!(
            hints.len(),
            n_objects,
            "each shard object has a distinct hint"
        );
    }

    /// Bounded-batch drain: `drain_batch_superfiles` is a memory bound and must
    /// never change the published layout. One drain over N user superfiles
    /// publishes ≤ writer_pool packed shard objects (here pool=1 ⇒ one shard),
    /// whether the budget forces N batches (`1`) or a single merge (`-1`).
    #[test]
    fn bounded_drain_batches_by_superfile_count() {
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

        let make = |batch_sf: i64| {
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
                    positions: false,
                }],
                vec![VectorConfig {
                    column: "emb".into(),
                    dim,
                    n_cent: 4,
                    rot_seed: 7,
                    metric: Metric::Cosine,
                    rerank_codec: RerankCodec::Sq8Residual,
                    provided_centroids: None,
                }],
                Some(crate::test_helpers::default_tokenizer()),
            )
            .expect("valid options")
            .with_storage(storage)
            .with_writer_pool(pool)
            .with_drain_batch_superfiles(batch_sf);
            let st = Supertable::create(options).expect("create");
            // Three commits → three user superfiles (identical vectors → one cell).
            for commit in 0..3 {
                let titles = LargeStringArray::from(vec![format!("doc-{commit}")]);
                let flat = Float32Array::from(vec![1.0f32; dim]);
                let fsl =
                    FixedSizeListArray::new(item_field.clone(), dim as i32, Arc::new(flat), None);
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
            // ONE drain call — the batching happens inside it.
            st.drain_vectors_to_cells_sync().expect("drain");
            st
        };

        let max_files_per_cell = |st: &Supertable| -> usize {
            let hidden = st
                .reader()
                .vector_index_table()
                .expect("hidden index")
                .clone();
            let reader = hidden.reader();
            let manifest = reader.manifest();
            let mut by_cell = HashMap::<Vec<u8>, usize>::new();
            for entry in manifest.superfiles.iter() {
                *by_cell.entry(entry.partition_key.clone()).or_default() += 1;
            }
            by_cell.values().copied().max().unwrap_or(0)
        };

        // batch=1: 3 user superfiles -> 3 memory batches, but still one packed
        // shard object (writer_pool=1 ⇒ N=1; identical vectors ⇒ one cell).
        let st1 = make(1);
        assert_eq!(
            max_files_per_cell(&st1),
            1,
            "batch=1 over 3 user superfiles must still publish one packed shard"
        );

        // batch=-1 (unbounded): all 3 in one merge -> identical layout.
        let st_unb = make(-1);
        assert_eq!(
            max_files_per_cell(&st_unb),
            1,
            "unbounded drain must merge all user superfiles into one packed shard"
        );

        // batch=0: drain skipped → hidden index stays empty.
        let st0 = make(0);
        assert_eq!(
            st0.reader()
                .vector_index_table()
                .expect("hidden index")
                .reader()
                .n_superfiles(),
            0,
            "batch=0 must skip the drain"
        );
    }

    /// The drain batch budget is a memory bound (how many user superfiles are
    /// materialized at once) and must not change the published layout.
    #[test]
    fn drain_batch_budget_never_changes_cell_layout() {
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
        let mut options = SupertableOptions::new(
            schema.clone(),
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool);
        options.drain_batch_superfiles = 1;
        let st = Supertable::create(options).expect("create");

        const N_COMMITS: usize = 3;
        for commit in 0..N_COMMITS {
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

        st.drain_vectors_to_cells_sync().expect("drain");

        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden index")
            .clone();
        let reader = hidden.reader();
        let manifest = reader.manifest();
        let mut per_cell = HashMap::<Vec<u8>, usize>::new();
        let mut total_rows = 0u64;
        for e in manifest.superfiles.iter() {
            *per_cell.entry(e.partition_key.clone()).or_default() += 1;
            total_rows += e.n_docs;
        }
        let max_per_cell = per_cell.values().copied().max().unwrap_or(0);
        assert_eq!(
            max_per_cell, 1,
            "one drain run must publish one packed shard object (got {max_per_cell})"
        );
        assert_eq!(
            total_rows, N_COMMITS as u64,
            "every drained row lands exactly once"
        );
        let fine_clusters: u32 = manifest
            .superfiles
            .iter()
            .flat_map(|entry| {
                entry
                    .vector_summary
                    .get("emb")
                    .into_iter()
                    .flat_map(|summary| summary.cells.iter())
            })
            .map(|cell| cell.clusters.n_cent)
            .sum();
        assert_eq!(
            fine_clusters, 1,
            "three one-row source batches must form one complete-cell fine IVF, not three batch fragments"
        );
        assert!(
            !manifest.get_drained_ranges().is_empty(),
            "single publish must advance drained watermark"
        );

        st.drain_vectors_to_cells_sync().expect("re-drain no-op");
        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden index")
            .clone();
        let n_after = hidden.reader().manifest().superfiles.len();
        assert_eq!(
            n_after,
            per_cell.len(),
            "no-op re-drain must not append shard files"
        );
    }

    /// Residency under churn — the manifest-split invariant, end to end.
    /// Drain publishes the slow-CAS entry blob and stamps its ref; a USER
    /// DELETE (which records hidden deleted-ids and bumps the HIDDEN
    /// pointer — the linked-manifest churn path) must preserve the ref and
    /// the resident entries (same `Arc`s). Only the next drain (membership
    /// change) replaces the blob and swaps the entries.
    #[test]
    fn hidden_slow_state_survives_user_delete_churn() {
        use std::sync::Arc;

        use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray};
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::prelude::{col, lit};

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
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool);
        let st = Supertable::create(options).expect("create");

        let append_one = |title: &str| {
            let titles = LargeStringArray::from(vec![title.to_owned()]);
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
        };
        append_one("alpha");
        append_one("beta");
        st.drain_vectors_to_cells_sync().expect("drain");

        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden vector index")
            .clone();
        let manifest_a = Arc::clone(hidden.reader().manifest());
        let (uri_a, _) = manifest_a
            .slow_vector_state_blob()
            .expect("drain must publish + stamp the slow-CAS ref");
        let uri_a = uri_a.to_owned();
        assert!(!manifest_a.superfiles.is_empty());

        // Churn: a USER delete records hidden deleted-ids — list-only churn
        // on the HIDDEN manifest (linked manifests). Ref + entries survive.
        let stats = st.delete(col("title").eq(lit("alpha"))).expect("delete");
        assert_eq!(stats.n_tombstoned(), 1, "delete must tombstone one row");
        let manifest_b = Arc::clone(hidden.reader().manifest());
        assert!(
            manifest_b.get_manifest_id() > manifest_a.get_manifest_id(),
            "user delete must bump the hidden manifest (deleted-ids stamp)"
        );
        assert!(
            manifest_b.deleted_user_ids_inline().is_some(),
            "delete must stamp hidden deleted ids inline"
        );
        let (uri_b, _) = manifest_b
            .slow_vector_state_blob()
            .expect("delete churn must PRESERVE the slow-CAS ref");
        assert_eq!(uri_b, uri_a, "ref unchanged by list-only churn");
        assert_eq!(manifest_b.superfiles.len(), manifest_a.superfiles.len());
        for (b, a) in manifest_b
            .superfiles
            .iter()
            .zip(manifest_a.superfiles.iter())
        {
            assert!(
                Arc::ptr_eq(b, a),
                "residency: the entries must be the SAME Arcs across delete churn"
            );
        }

        // Membership change: another commit + drain republishes the blob —
        // the ONLY invalidation the slow state accepts.
        append_one("gamma");
        st.drain_vectors_to_cells_sync().expect("second drain");
        let manifest_c = Arc::clone(hidden.reader().manifest());
        let (uri_c, _) = manifest_c
            .slow_vector_state_blob()
            .expect("drain must restamp the ref");
        assert_ne!(uri_c, uri_a, "new membership ⇒ new content-addressed blob");
    }

    /// The hidden deleted-`_id` set is decoded from the resident inline
    /// manifest bytes ONCE per manifest version and cached on the handle:
    /// repeated reads on the same version return the same `Arc` (no
    /// re-decode), and a user delete that bumps the hidden manifest
    /// re-decodes the updated set. This is the only discipline the
    /// GET-free inline set needs.
    #[test]
    fn hidden_deleted_ids_decoded_once_per_manifest_version() {
        use std::sync::Arc;

        use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray};
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::prelude::{col, lit};

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
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool);
        let st = Supertable::create(options).expect("create");

        let append_one = |title: &str| {
            let titles = LargeStringArray::from(vec![title.to_owned()]);
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
        };
        append_one("alpha");
        append_one("beta");
        st.drain_vectors_to_cells_sync().expect("drain");

        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden vector index")
            .clone();

        // No deletes yet: the resident set is empty, and two reads on the
        // same manifest version return the SAME cached `Arc` (decoded once).
        let empty_a = hidden.reader().hidden_deleted_ids().expect("decode");
        let empty_b = hidden.reader().hidden_deleted_ids().expect("cached");
        assert!(empty_a.is_empty(), "no deletes ⇒ empty resident set");
        assert!(
            Arc::ptr_eq(&empty_a, &empty_b),
            "same manifest version must reuse the decoded set (no per-query decode)"
        );

        // A user delete bumps the hidden manifest and stamps the id inline.
        let stats = st.delete(col("title").eq(lit("alpha"))).expect("delete");
        assert_eq!(stats.n_tombstoned(), 1, "delete tombstones one row");

        // New manifest version ⇒ re-decode the updated set; then cached again.
        let ids_a = hidden
            .reader()
            .hidden_deleted_ids()
            .expect("decode after delete");
        let ids_b = hidden
            .reader()
            .hidden_deleted_ids()
            .expect("cached after delete");
        assert_eq!(ids_a.len(), 1, "one deleted id resident after delete");
        assert!(
            Arc::ptr_eq(&ids_a, &ids_b),
            "post-delete version must also reuse its decoded set"
        );
        assert!(
            !Arc::ptr_eq(&empty_a, &ids_a),
            "a manifest bump must re-decode the updated set, not serve the stale one"
        );
    }

    /// Every drain-built hidden cell superfile must carry a usable
    /// `vector_summary` (summary centroid + non-empty per-cluster centroids,
    /// correct dim). An entry without one would silently degrade cluster
    /// selection — the fan-out hard-errors on it now, so the build path must
    /// never produce such an entry.
    #[test]
    fn drain_built_entries_carry_vector_summaries() {
        use std::sync::Arc;

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
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool);
        let st = Supertable::create(options).expect("create");

        // Distinct directions so the drain builds more than one cell.
        for i in 0..8usize {
            let titles = LargeStringArray::from(vec![format!("doc{i}")]);
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            let flat = Float32Array::from(v);
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
        st.drain_vectors_to_cells_sync().expect("drain");

        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden vector index")
            .clone();
        let manifest = Arc::clone(hidden.reader().manifest());
        assert!(!manifest.superfiles.is_empty(), "drain built cell files");
        for entry in manifest.superfiles.iter() {
            let vs = entry.vector_summary.get("emb").unwrap_or_else(|| {
                panic!(
                    "drain-built hidden superfile {} has NO vector_summary",
                    entry.superfile_id
                )
            });
            assert_eq!(vs.centroid.len(), dim, "summary centroid dim");
            assert!(
                vs.cells.iter().any(|cell| !cell.clusters.is_empty()),
                "drain-built hidden superfile {} has EMPTY cluster centroids",
                entry.superfile_id
            );
            assert!(
                vs.cells
                    .iter()
                    .all(|cell| cell.clusters.dim as usize == dim),
                "cluster centroid dim"
            );
        }
    }

    /// Raw pointer object ceiling for the thin-pointer assertions: three
    /// short text lines (id, list URI, hash) — generously bounded.
    const MAX_POINTER_OBJECT_BYTES: usize = 512;

    /// Storage contract of the fast/slow split, end to end:
    /// (1) once the drainer stamps the slow-CAS ref, the pointer object is
    ///     TINY — no payload rides the hot-CAS write;
    /// (2) `optimize` (whose membership `update`s clear the ref) ends
    ///     re-stamped with a durable, non-empty blob — the state a
    ///     post-maintenance footprint reads;
    /// (3) a fresh process open hydrates the flat view FROM the blob —
    ///     proven by deleting every hidden manifest part first, so nothing
    ///     else can serve the entries.
    #[test]
    fn slow_state_thin_pointer_and_blob_serves_fresh_open() {
        use std::sync::Arc;

        use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray};
        use arrow_schema::{DataType, Field, Schema};

        use crate::{
            config::OptimizeOptions,
            superfile::{
                builder::{FtsConfig, VectorConfig},
                reader::VectorSearchOptions,
                vector::{distance::Metric, rerank_codec::RerankCodec},
            },
            supertable::manifest::commit::{MANIFEST_PARTS_DIR, POINTER_PATH},
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
        let make_options = || {
            let storage: Arc<dyn StorageProvider> =
                Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
            SupertableOptions::new(
                schema.clone(),
                vec![FtsConfig {
                    column: "title".into(),
                    positions: false,
                }],
                vec![VectorConfig {
                    column: "emb".into(),
                    dim,
                    n_cent: 4,
                    rot_seed: 7,
                    metric: Metric::Cosine,
                    rerank_codec: RerankCodec::Sq8Residual,
                    provided_centroids: None,
                }],
                Some(crate::test_helpers::default_tokenizer()),
            )
            .expect("valid options")
            .with_storage(storage)
            .with_writer_pool(Arc::clone(&pool))
        };
        let st = Supertable::create(make_options()).expect("create");

        let append_one = |title: &str| {
            let titles = LargeStringArray::from(vec![title.to_owned()]);
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
        };
        append_one("alpha");
        append_one("beta");
        st.drain_vectors_to_cells_sync().expect("drain");

        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden vector index")
            .clone();
        let hidden_storage = hidden
            .reader()
            .manifest()
            .options
            .storage
            .clone()
            .expect("hidden storage");

        // (1) Ref stamped ⇒ pointer tiny (raw object bytes bounded — no
        // payload of any kind rides the hot-CAS write), blob durable and
        // non-empty.
        let (uri_a, _) = hidden
            .reader()
            .manifest()
            .slow_vector_state_blob()
            .map(|(u, h)| (u.to_owned(), h))
            .expect("drain must stamp the slow-CAS ref");
        let (ptr_bytes, _) = hidden
            .block_on_query(hidden_storage.get(POINTER_PATH))
            .expect("read pointer object");
        assert!(
            ptr_bytes.len() <= MAX_POINTER_OBJECT_BYTES,
            "pointer object must stay tiny (id + list uri + hash); got {} bytes",
            ptr_bytes.len()
        );
        let (blob, _) = hidden
            .block_on_query(hidden_storage.get(&uri_a))
            .expect("slow blob durable");
        assert!(!blob.is_empty(), "slow blob carries the entry payload");

        // (2) optimize (drain no-op + compaction membership updates clear the
        // ref) must END re-stamped, thin-pointered, with a durable blob.
        st.optimize(&OptimizeOptions::default()).expect("optimize");
        let manifest_after = Arc::clone(hidden.reader().manifest());
        let (uri_b, _) = manifest_after
            .slow_vector_state_blob()
            .map(|(u, h)| (u.to_owned(), h))
            .expect("optimize must end with the ref re-stamped");
        let (blob_b, _) = hidden
            .block_on_query(hidden_storage.get(&uri_b))
            .expect("slow blob durable after optimize");
        assert!(!blob_b.is_empty());
        let (ptr_bytes_b, _) = hidden
            .block_on_query(hidden_storage.get(POINTER_PATH))
            .expect("read pointer object");
        assert!(
            ptr_bytes_b.len() <= MAX_POINTER_OBJECT_BYTES,
            "post-optimize pointer must stay tiny; got {} bytes",
            ptr_bytes_b.len()
        );
        let n_entries = manifest_after.superfiles.len();
        assert!(n_entries > 0, "hidden flat view populated");

        // (3) Hidden membership never writes manifest parts. Fresh open must
        // hydrate exclusively from the slow blob.
        let parts = hidden
            .block_on_query(hidden_storage.list_with_prefix(MANIFEST_PARTS_DIR))
            .expect("list hidden parts");
        assert!(
            parts.is_empty(),
            "hidden table must not write manifest parts"
        );
        drop(hidden);
        drop(st);

        let st2 = Supertable::open(make_options()).expect("reopen");
        let hidden2 = st2
            .reader()
            .vector_index_table()
            .expect("hidden vector index on reopen")
            .clone();
        assert_eq!(
            hidden2.reader().manifest().superfiles.len(),
            n_entries,
            "fresh open hydrated the flat view from the blob (parts deleted)"
        );
        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let hits = st2
            .reader()
            .vector_hits("emb", &q, 2, VectorSearchOptions::new(), None)
            .expect("vector search on blob-hydrated manifest");
        assert!(!hits.is_empty(), "search serves from the hydrated view");
    }

    /// Incremental drain: each drain consumes only user commits not already in
    /// the hidden manifest's `drained_ranges`, and a drain with no new commits
    /// is a no-op (no re-drive, no duplicate cells). The distinguishing signal
    /// is the *third* drain: with incrementality it adds nothing; without it,
    /// it would re-drain everything and append another per-cell file.
    #[test]
    fn incremental_drain_skips_already_drained_commits() {
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
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool);
        let st = Supertable::create(options).expect("create");

        let commit = |tag: &str| {
            let titles = LargeStringArray::from(vec![format!("doc-{tag}")]);
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
        };
        let cell_files = || -> usize {
            let hidden = st
                .reader()
                .vector_index_table()
                .expect("hidden index")
                .clone();
            let reader = hidden.reader();
            let manifest = reader.manifest();
            let mut by_cell = HashMap::<Vec<u8>, usize>::new();
            for e in manifest.superfiles.iter() {
                *by_cell.entry(e.partition_key.clone()).or_default() += 1;
            }
            by_cell.values().copied().max().unwrap_or(0)
        };

        // Commit A, drain → one cell file; the commit's version is now drained.
        commit("a");
        st.drain_vectors_to_cells_sync().expect("drain 1");
        assert_eq!(cell_files(), 1, "first drain populates the cell");
        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden index")
            .clone();
        assert!(
            !hidden.reader().manifest().get_drained_ranges().is_empty(),
            "drain must record progress in drained_ranges"
        );

        // Commit B, drain → only B is new, so exactly one more cell file.
        commit("b");
        st.drain_vectors_to_cells_sync().expect("drain 2");
        assert_eq!(cell_files(), 2, "second drain consumes only the new commit");

        // No new commit: the third drain is a NO-OP (incrementality).
        st.drain_vectors_to_cells_sync().expect("drain 3 (no-op)");
        assert_eq!(
            cell_files(),
            2,
            "drain with nothing new must not re-drive already-drained commits"
        );
        // Watermark stays a single genesis-anchored interval (contiguous commits).
        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden index")
            .clone();
        assert_eq!(
            hidden
                .reader()
                .manifest()
                .get_drained_ranges()
                .intervals()
                .len(),
            1,
            "contiguous commits must leave drained_ranges as one interval"
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
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(crate::test_helpers::default_tokenizer()),
        )
        .expect("valid options")
        .with_storage(storage)
        .with_writer_pool(pool);
        let st = Supertable::create(options).expect("create");

        let rows_per_commit = 8usize;
        for commit in 0..3 {
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
            // Phase B: drain after each commit; each drain appends a file per
            // cell, accumulating the per-cell superfiles compaction collapses.
            st.drain_vectors_to_cells_sync().expect("drain to cells");
        }

        let hidden = st
            .reader()
            .vector_index_table()
            .expect("hidden vector index")
            .clone();
        let count_by_shard = |manifest: &crate::supertable::manifest::ManifestSnapshot| -> usize {
            let mut by_shard = HashMap::<Vec<u8>, usize>::new();
            for entry in manifest.superfiles.iter() {
                if entry.vector_layout != VectorLayout::MultiCellIvf
                    && entry.vector_layout != VectorLayout::Ivf
                {
                    continue;
                }
                *by_shard.entry(entry.partition_key.clone()).or_default() += 1;
            }
            by_shard.values().copied().max().unwrap_or(0)
        };
        let before = count_by_shard(hidden.reader().manifest());
        assert!(
            before >= 2,
            "need multiple drained packed shards before compaction, got {before}"
        );

        let cfg = CompactionSettings {
            target_superfile_size_mb: 1,
            min_fill_percent: 1,
            ..CompactionSettings::default()
        };
        hidden.compact(&cfg).expect("hidden compact");

        let after_reader = hidden.reader();
        let after_manifest = after_reader.manifest();
        let after = count_by_shard(after_manifest);
        assert!(
            after < before,
            "compaction should collapse packed shards: before={before} after={after}"
        );
        for entry in &after_manifest.superfiles {
            assert!(
                entry.vector_layout == VectorLayout::MultiCellIvf
                    || entry.vector_layout == VectorLayout::Ivf,
                "unexpected layout {:?}",
                entry.vector_layout
            );
            assert!(
                entry
                    .subsection_offsets
                    .as_ref()
                    .and_then(|o| o.vec)
                    .is_some(),
                "compacted hidden entry {:?} missing vec subsection",
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
