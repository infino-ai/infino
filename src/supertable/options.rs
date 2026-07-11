// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `SupertableOptions` — the immutable per-supertable configuration.
//!
//! ## Shape
//!
//! `SupertableOptions` is the supertable layer's analogue of
//! `superfile::builder::BuilderOptions`, with one structural
//! difference: **the supertable's schema includes vector columns as
//! `FixedSizeList<Float32, dim>` fields**. Callers append a single
//! `RecordBatch` carrying both scalar columns and vector columns;
//! [`utils::vector_split::split_vectors`] pulls the vector columns out at
//! commit time and forwards `(scalar_only_batch, &[&[f32]])` to the
//! underlying `SuperfileBuilder` (whose schema must NOT include
//! vector columns — vectors live only in the embedded vector blob,
//! never in Parquet).
//!
//! ## Validation
//!
//! `SupertableOptions::new` validates everything statically derivable
//! from the schema + config:
//!
//! - `id_column` exists in schema and is `UInt64`.
//! - Each FTS column exists in schema and is `LargeUtf8`.
//! - Each vector column exists in schema as
//!   `FixedSizeList<Float32, dim>` with `list_size == dim`.
//! - `dim` is in [16, 4096] (mirrors the superfile vector range).
//! - No `\x1F` or `inf.` reserved-name characters.
//! - Logical names unique across `fts_columns` and `vector_columns`.
//! - FTS columns require a tokenizer (and vice versa: a tokenizer
//!   without FTS columns is rejected as inconsistent input — caller
//!   bug).
//!
//! Per-row validation (vectors are non-null, schema matches) lives
//! in [`utils::vector_split`](super::utils::vector_split) since it's a runtime
//! check against each input batch.
//!
//! [`utils::vector_split::split_vectors`]: super::utils::vector_split::split_vectors

use std::{
    collections::HashSet,
    fmt,
    sync::{Arc, OnceLock},
    time::Duration,
};

use arrow_schema::{DataType, Field, Schema};
use rayon::{ThreadPool, ThreadPoolBuilder};

use super::{
    error::BuildError,
    reader_cache::{
        ColdFetchMode, DiskCacheConfig, DiskCacheStore, InMemoryReaderCache, LruPolicy,
        SuperfileReaderCache,
    },
};
use crate::{
    config::{Config, StorageBackend, StorageColdFetchMode, ThreadCount},
    memory::ConnectionMemoryBudget,
    storage::{
        AzureStorageProvider, GcsStorageProvider, LocalFsStorageProvider, S3StorageProvider,
        StorageProvider,
    },
    superfile::{
        OpenOptions,
        builder::{BuilderOptions, FtsConfig, VectorConfig},
        fts::tokenize::Tokenizer,
    },
    supertable::manifest::{disk_cache::ManifestDiskCache, list::PartitionStrategy},
};

/// Vector column dim must be in this inclusive range. Mirrors
/// `superfile::error::BuildError::VectorDimOutOfRange`.
const VECTOR_DIM_MIN: usize = 16;
const VECTOR_DIM_MAX: usize = 4096;

/// Reserved separator inside FTS FST keys (`<col>\x1F<term>`); user
/// column names must not contain it. Mirrors superfile's
/// `check_user_column_name`.
const RESERVED_SEPARATOR: char = '\x1F';

/// Reserved KV-metadata prefix; user column names must not start
/// with it. Mirrors superfile's `check_user_column_name`.
const RESERVED_PREFIX: &str = "inf.";

/// Default writer-pool size: half the host's logical cores
/// (rounded up, minimum 1). Read-mostly workloads keep reader p99
/// stable under writer load; ETL-shaped overrides via
/// [`SupertableOptions::with_writer_pool`].
fn default_writer_thread_count() -> usize {
    num_cpus::get().div_ceil(2).max(1)
}

/// Default reader-pool size: every logical core. Reader fan-out
/// across non-pruned superfiles saturates this pool.
fn default_reader_thread_count() -> usize {
    num_cpus::get().max(1)
}

/// Process-wide reader pool, shared by every supertable that doesn't
/// inject its own (via [`SupertableOptions::with_reader_pool`] or a
/// `Fixed` count in [`Config`]) — M open tables cost N reader threads,
/// not M×N.
static SHARED_READER_POOL: OnceLock<Arc<ThreadPool>> = OnceLock::new();

/// Process-wide writer pool; same sharing contract at half the cores.
static SHARED_WRITER_POOL: OnceLock<Arc<ThreadPool>> = OnceLock::new();

fn shared_reader_pool() -> Arc<ThreadPool> {
    Arc::clone(SHARED_READER_POOL.get_or_init(|| {
        Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(default_reader_thread_count())
                .thread_name(|i| format!("supertable-reader-{i}"))
                .build()
                .expect("invariant: rayon pool build only fails on thread-spawn failure"),
        )
    }))
}

fn shared_writer_pool() -> Arc<ThreadPool> {
    Arc::clone(SHARED_WRITER_POOL.get_or_init(|| {
        Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(default_writer_thread_count())
                .thread_name(|i| format!("supertable-writer-{i}"))
                .build()
                .expect("invariant: rayon pool build only fails on thread-spawn failure"),
        )
    }))
}

/// Default name of the supertable-injected primary-key column.
/// Tracks `Config::supertable::id_column`'s default; kept as a
/// free function so the validation in
/// [`SupertableOptions::new`] doesn't need a `Config` snapshot.
fn default_id_column() -> String {
    "_id".to_string()
}

/// Arrow / Parquet decimal precision for the id column. 38 is
/// the maximum precision a `Decimal128` can carry — covers
/// every possible 128-bit value without truncation, lets
/// Parquet annotate the column as `DECIMAL(38, 0)`, and
/// preserves byte-comparable / numerically-comparable sort
/// order against the underlying `i128`.
pub(crate) const DECIMAL128_PRECISION: u8 = 38;

/// Scale of zero — the id column carries integers, not
/// fractions. With precision 38 and scale 0, the column
/// behaves as a signed 128-bit integer for both Arrow's
/// comparison kernels and Parquet's stats encoding.
pub(crate) const DECIMAL128_SCALE: i8 = 0;

/// Default bounded-staleness window for read consistency — how long
/// a hot query may reuse a manifest pointer before re-checking.
const DEFAULT_READ_STALENESS_SECS: u64 = 1;
/// Default soft cap on superfiles per manifest part; exceeding it
/// triggers a part split on commit.
const DEFAULT_TARGET_SUPERFILES_PER_PART: u64 = 10_000;
/// Default soft cap on a manifest part's compressed size (10 MiB).
const DEFAULT_PART_SIZE_THRESHOLD_BYTES: u64 = 10 * (1 << 20);
/// Default: eager-load manifest parts at open when there are at most
/// this many (open latency vs memory trade-off).
const DEFAULT_EAGER_LOAD_THRESHOLD_PARTS: u32 = 4;
/// Subdirectory under the disk-cache root that holds the
/// content-addressed manifest-part byte cache. Kept separate from the
/// superfile cache files so the two budgets and eviction sets don't
/// interfere.
const MANIFEST_CACHE_SUBDIR: &str = "manifest-parts";
/// Default optimistic-commit retry budget under contention.
const DEFAULT_MAX_COMMIT_RETRIES: u32 = 10;
/// Default writer auto-flush threshold (1 GiB, in MiB units).
const DEFAULT_COMMIT_THRESHOLD_SIZE_MB: u64 = 1024;
/// Default object size (100 MiB) above which uploads route through
/// multipart.
const DEFAULT_PUT_MULTIPART_THRESHOLD_BYTES: u64 = 100 * (1 << 20);

/// Read-path freshness policy — how an open handle picks up superfiles
/// committed (by this or another process) after it opened.
///
/// Modeled on the same knob every object-store-native engine exposes
/// (turbopuffer's per-query consistency level, LanceDB's
/// `read_consistency_interval`): the *engine* re-checks the manifest
/// pointer for the caller; the application never refreshes by hand.
/// A same-process writer's commit is always visible immediately
/// (read-your-writes) regardless of this setting — the policy only
/// governs picking up *other* processes' commits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Consistency {
    /// Re-check the manifest pointer on every query (turbopuffer's
    /// default). One cheap pointer read per query; strongest freshness.
    Strong,
    /// Re-check the pointer at most once per interval, serving the
    /// pinned snapshot in between. Trades a bounded staleness window
    /// for not paying the pointer read on every (sub-millisecond) hot
    /// query — the speed-per-dollar default.
    BoundedStaleness(Duration),
    /// Snapshot fixed at open. Only same-process commits advance it;
    /// other processes' new data requires a fresh `open`. For pure
    /// read replicas / time-bounded scans that never want surprise
    /// pointer reads.
    ///
    /// Do not hold a `Snapshot` handle open longer than the GC safety
    /// gap (default 24 hours). GC removes old manifests once they age
    /// past that threshold; a handle whose pinned manifest has been
    /// collected will fail on the next query.
    Snapshot,
}

impl Default for Consistency {
    fn default() -> Self {
        // Bounded staleness with a 1s window: the conditional pointer
        // read (~10ms on S3) is negligible against a cold query but
        // dominates a hot one, so amortize it rather than pay it per
        // hot query. Strong/Snapshot are opt-in via
        // `with_read_consistency`.
        Consistency::BoundedStaleness(Duration::from_secs(DEFAULT_READ_STALENESS_SECS))
    }
}

/// All knobs needed to construct a supertable.
///
/// Holds both the immutable per-supertable configuration (schema,
/// FTS / vector columns, tokenizer) and the runtime resources the
/// writer / reader paths use (thread pools, superfile store,
/// commit-flush threshold). Held by `SupertableInner` as
/// `Arc<SupertableOptions>` so readers, the writer, and rayon
/// shard workers all see the same instances without copying.
pub struct SupertableOptions {
    /// User-declared Arrow schema. Contains every
    /// `fts_columns[i].column` (LargeUtf8) and every
    /// `vector_columns[i].column` (FixedSizeList<Float32, dim>).
    /// Must NOT contain a field named [`Self::id_column`] — the
    /// supertable injects that column at append time.
    pub schema: Arc<Schema>,
    /// Name of the system-managed primary-key column the
    /// supertable injects on every `append()`. Defaults to
    /// `"_id"`; override via [`Self::with_id_column`] or by
    /// applying a [`Config`] whose `supertable.id_column`
    /// differs. The column type is fixed at `UInt64`.
    pub id_column: String,
    /// FTS columns. May be empty.
    pub fts_columns: Vec<FtsConfig>,
    /// Vector columns. Each must appear in `schema` as
    /// `FixedSizeList<Float32, dim>` with matching `list_size`.
    /// May be empty.
    pub vector_columns: Vec<VectorConfig>,
    /// Shared tokenizer for all FTS columns. Required iff
    /// `fts_columns` is non-empty.
    pub tokenizer: Option<Arc<dyn Tokenizer>>,
    /// Pool used by reader fan-out (skip + per-superfile fan-out +
    /// top-k merge). Default: every logical core.
    pub reader_pool: Arc<ThreadPool>,
    /// Pool used by writer commit-time rayon-shard. Default:
    /// half the logical cores.
    pub writer_pool: Arc<ThreadPool>,
    /// Where superfile bytes live. Shared across reader threads
    /// and the writer. Default: `InMemoryReaderCache`.
    pub store: Arc<dyn SuperfileReaderCache>,
    /// Object-storage backend. When `Some`, writer commits
    /// persist superfiles + manifest to storage via
    /// [`commit_manifest`](crate::supertable::manifest::commit::commit_manifest);
    /// when `None`, the supertable is in-memory-only.
    ///
    /// Reads go through `store` (the in-memory
    /// `SuperfileReaderCache`) unless a `disk_cache` is attached,
    /// in which case the reader path routes through the cache.
    pub storage: Option<Arc<dyn StorageProvider>>,
    /// Disk cache for storage-backed superfile reads.
    /// When attached together with `storage`, the supertable's
    /// reader path routes superfile-bytes lookups through this
    /// cache instead of relying solely on the in-memory `store`
    /// — the load-bearing change that lets a cross-process
    /// `Supertable::open` answer queries on a 100GB index
    /// without pulling every superfile into RAM.
    ///
    /// Construction stays user-managed (`Arc<DiskCacheStore>`)
    /// so callers retain full control over cache root, budget,
    /// eviction policy, and (currently) the `pinned_fn` shape.
    /// Auto-wiring the `pinned_fn` to the supertable's current
    /// manifest is deferred — eviction during use is safe
    /// today (an `Arc<SuperfileReader>` keeps the `Arc<Mmap>`
    /// alive after the on-disk file is unlinked, so in-flight
    /// queries finish correctly), so pinning is purely a
    /// reclaim-and-refetch perf optimization that lands on
    /// measured need.
    ///
    /// Independent of `storage`: attaching a cache without
    /// storage is a configuration error caught at
    /// [`Supertable::create`] / [`Supertable::open`] time.
    pub disk_cache: Option<Arc<DiskCacheStore>>,
    /// On-disk cache for compressed manifest-part bytes. When set,
    /// [`ManifestPartLoader`](crate::supertable::manifest::ManifestPartLoader)
    /// reads a part's bytes from local disk on a hit instead of
    /// round-tripping to object storage. Parts are content-addressed,
    /// so cached files are never stale and the cache survives process
    /// restarts. Independent of `disk_cache` (which caches superfile
    /// content) and uses its own byte budget. `None` disables it.
    pub manifest_disk_cache: Option<Arc<ManifestDiskCache>>,
    /// Best-effort memory budget for the disk cache's mmap
    /// working set, in bytes. When set together with
    /// `disk_cache`, the supertable triggers
    /// [`DiskCacheStore::sweep_for_budget`] after each
    /// commit (and on demand via the bench's memory-pressure
    /// loop): if the cache's mmap-resident size exceeds the
    /// budget, the oldest entries get `madvise(MADV_DONTNEED)`
    /// until the working set is back under the cap. Pages
    /// re-fault from the backing file on next access — the
    /// on-disk cache and the entry-set are unchanged.
    ///
    /// "Best-effort": not a hard cgroup limit. RSS stays
    /// within ±10% of budget under sustained query load on
    /// the laptop bench; strict enforcement requires a
    /// custom allocator on top.
    ///
    /// `None`-equivalent shape: don't call
    /// [`Self::with_memory_budget`]. The cache then runs the
    /// idle-threshold madvise sweep on its existing schedule
    /// but does NOT proactively bound the RSS.
    pub memory_budget_bytes: Option<u64>,
    /// Per-connection heap budget, shared (cloned `Arc`) by every supertable
    /// the owning connection opens. The query and ingest paths reserve against
    /// it before allocating; `measured()` by default, so it tracks usage but
    /// never refuses until a limit is set. Distinct from `memory_budget_bytes`:
    /// that bounds the mmap resident set, this bounds anonymous heap. See
    /// [`crate::memory`].
    pub(crate) connection_memory_budget: Arc<ConnectionMemoryBudget>,
    /// When `true` (default), each commit pre-populates the
    /// attached `disk_cache` with the superfile bytes it just
    /// wrote (so the producer's own next query skips the
    /// cold-fetch round-trip). Set `false` for a write-only
    /// producer that drops the cache right after ingest: the
    /// pre-population is then pure wasted work (and, when the
    /// superfile set exceeds the cache budget, floods the log
    /// with "budget exceeded" warnings). No effect without
    /// `disk_cache` attached.
    pub prepopulate_cache_on_commit: bool,
    /// Partition strategy. Stamped into the manifest list
    /// on the first commit; immutable thereafter (changes
    /// require external compaction).
    ///
    /// When `None` at [`Supertable::create`] time, resolved
    /// to `Hash { column: id_column, n_buckets: 1 }` — a
    /// single-bucket strategy that's observationally
    /// equivalent to "no partitioning" (every superfile lands
    /// in the one bucket → one `ManifestPart` per commit).
    /// Callers wanting real partitioning set this via
    /// [`Self::with_partition_strategy`].
    ///
    /// At [`Supertable::open`] time, this field is read from
    /// the persisted manifest list — config changes after
    /// creation have no effect.
    pub partition_strategy: Option<PartitionStrategy>,
    /// Soft cap on superfiles per `ManifestPart`.
    /// When a partition's existing part reaches this count,
    /// the next commit's superfiles for that partition go into
    /// a fresh part instead of rewriting the existing one.
    /// Default `10_000`.
    pub target_superfiles_per_part: u64,
    /// Soft cap on a `ManifestPart`'s compressed Avro+zstd
    /// size in bytes. Triggers part-split alongside
    /// `target_superfiles_per_partition`. Default `10 * (1 << 20)`
    /// (10 MiB).
    pub part_size_threshold_bytes: u64,
    /// Eager-load threshold for manifest parts at
    /// [`Supertable::open`] time. When the manifest list
    /// references this many parts or fewer, open parallel-
    /// fetches all parts up front + populates the
    /// `Manifest.parts` cache. Above the threshold, parts are
    /// left in empty `OnceCell`s — the first
    /// `Manifest::part(id).await` lazy-loads on demand.
    ///
    /// Default `4`. Set to
    /// `0` to force lazy-load even for tiny manifests
    /// (useful for tests that want to verify the lazy path).
    ///
    /// **Eager mode** populates `Manifest.superfile_list.superfiles`
    /// with the flat union of all loaded parts' superfiles —
    /// the legacy query paths (`bm25_search`,
    /// `vector_search`, `query_sql`) iterate this flat view.
    ///
    /// **Lazy mode** leaves `Manifest.superfile_list.superfiles`
    /// empty until the hierarchical query path lands.
    /// Until then, callers using lazy mode must drive
    /// `Manifest::part(id).await` directly; legacy
    /// flat-iteration queries return empty results.
    pub eager_load_threshold_parts: u32,
    /// Max OCC retry attempts before `writer.commit()` surfaces
    /// [`CommitError::WriteContentionExhausted`](crate::supertable::CommitError::WriteContentionExhausted).
    /// Each retry refreshes the in-memory state from storage,
    /// rebuilds the commit on top, and re-issues with jittered
    /// exponential backoff. Default `10` — enough for typical
    /// concurrent-writer contention; writer-heavy workloads can
    /// raise via [`Self::with_max_commit_retries`].
    pub max_commit_retries: u32,
    /// Auto-flush threshold for the writer's in-memory buffer,
    /// in MiB of accumulated raw payload (Arrow scalar columns +
    /// f32 vector slices). When the buffer crosses this
    /// threshold during `append`, the writer triggers an
    /// internal `commit()`. `0` disables auto-flush — only
    /// caller-driven `commit()` produces superfiles.
    /// Default: 1024 (1 GiB).
    pub commit_threshold_size_mb: u64,
    /// Superfile size (in bytes) at or above which the writer
    /// routes the storage write through
    /// [`StorageProvider::put_multipart`] instead of
    /// [`StorageProvider::put_atomic`]. The single-PUT path
    /// pins the whole superfile in `Bytes` at issue time and
    /// re-uploads everything on retry; the multipart path
    /// splits the upload into 8-MiB chunks driven in
    /// parallel, lowering both peak RSS during the put and
    /// the cost of a transient backend failure mid-upload.
    ///
    /// Default: `100 * (1 << 20)` (100 MiB) — matches the
    /// standard S3 SDK multipart threshold. Set to
    /// `u64::MAX` to disable multipart routing entirely;
    /// set to a tiny value (e.g. `1`) to force every
    /// superfile through the multipart path (useful for tests).
    pub put_multipart_threshold_bytes: u64,
    /// Whether the supertable's read-side opens of
    /// `SuperfileReader` should verify the trailing whole-
    /// blob CRC and per-subsection CRCs. Default `true`. The
    /// supertable threads this through
    /// `SuperfileReader::open_with`'s
    /// `OpenOptions { verify_crc }` on every open it
    /// performs (writer post-commit summary extraction, disk
    /// cache cold-fetch finalize). Set `false` when the
    /// underlying storage already validates checksums —
    /// content-addressed object stores, ZFS, etc. — to skip
    /// the scan.
    pub verify_crc_on_open: bool,
    /// Read-path freshness policy. The query path checks the manifest
    /// pointer for the caller per this setting (see [`Consistency`]);
    /// the application never refreshes by hand. Default:
    /// [`Consistency::BoundedStaleness`] with a 1s window.
    pub read_consistency: Consistency,
}

impl SupertableOptions {
    /// Construct + validate. Returns `BuildError::*` on any
    /// inconsistency between schema, fts_columns, and
    /// vector_columns. The id column name defaults to `"_id"`;
    /// override via [`Self::with_id_column`] or by applying a
    /// [`Config`] whose `supertable.id_column` differs.
    ///
    /// The schema must NOT contain a field named the same as
    /// the configured id column — the supertable injects that
    /// column at append time.
    pub fn new(
        schema: Arc<Schema>,
        fts_columns: Vec<FtsConfig>,
        vector_columns: Vec<VectorConfig>,
        tokenizer: Option<Arc<dyn Tokenizer>>,
    ) -> Result<Self, BuildError> {
        let id_column = default_id_column();

        // 1. User schema must NOT contain the id column —
        //    that's the supertable's responsibility to inject
        //    at append time. Surfacing a typed error here lets
        //    callers catch the conflict at construction
        //    instead of getting a confusing duplicate-column
        //    error from Arrow at first append.
        if schema.fields().iter().any(|f| f.name() == &id_column) {
            return Err(BuildError::IdColumnReserved(id_column.clone()));
        }

        // 2. Each FTS column must exist in schema and be LargeUtf8.
        for fc in &fts_columns {
            check_user_column_name(&fc.column)?;
            let idx = schema
                .index_of(&fc.column)
                .map_err(|_| BuildError::FtsColumnMissing {
                    column: fc.column.clone(),
                })?;
            let f = schema.field(idx);
            if f.data_type() != &DataType::LargeUtf8 {
                return Err(BuildError::FtsColumnMustBeLargeUtf8 {
                    column: fc.column.clone(),
                    actual: format!("{:?}", f.data_type()),
                });
            }
        }

        // 3. Each vector column must exist in schema as
        //    FixedSizeList<Float32, dim> with list_size == dim.
        for vc in &vector_columns {
            check_user_column_name(&vc.column)?;
            if vc.dim < VECTOR_DIM_MIN || vc.dim > VECTOR_DIM_MAX {
                return Err(BuildError::VectorDimOutOfRange {
                    column: vc.column.clone(),
                    dim: vc.dim,
                });
            }
            let idx = schema
                .index_of(&vc.column)
                .map_err(|_| BuildError::VectorColumnMissing {
                    column: vc.column.clone(),
                })?;
            let f = schema.field(idx);
            match f.data_type() {
                DataType::FixedSizeList(item_field, list_size) => {
                    if item_field.data_type() != &DataType::Float32 {
                        return Err(BuildError::VectorColumnNotFixedSizeList {
                            column: vc.column.clone(),
                            dim: vc.dim,
                            actual: format!("{:?}", f.data_type()),
                        });
                    }
                    let list_size = usize::try_from(*list_size).unwrap_or(usize::MAX);
                    if list_size != vc.dim {
                        return Err(BuildError::VectorColumnDimMismatch {
                            column: vc.column.clone(),
                            expected: vc.dim,
                            actual: list_size,
                        });
                    }
                }
                other => {
                    return Err(BuildError::VectorColumnNotFixedSizeList {
                        column: vc.column.clone(),
                        dim: vc.dim,
                        actual: format!("{:?}", other),
                    });
                }
            }
        }

        // 4. Logical names unique across fts_columns + vector_columns.
        //    (Each was checked for presence in `schema` above; here
        //    we ensure no duplicates between the two role lists.)
        let mut seen_logical: HashSet<&str> = HashSet::new();
        for fc in &fts_columns {
            if !seen_logical.insert(fc.column.as_str()) {
                return Err(BuildError::DuplicateLogicalName(fc.column.clone()));
            }
        }
        for vc in &vector_columns {
            if !seen_logical.insert(vc.column.as_str()) {
                return Err(BuildError::DuplicateLogicalName(vc.column.clone()));
            }
        }

        // 5. FTS columns require a tokenizer.
        if !fts_columns.is_empty() && tokenizer.is_none() {
            return Err(BuildError::MissingTokenizer);
        }

        // 6. Shared thread pools + a fresh store.
        let reader_pool = shared_reader_pool();
        let writer_pool = shared_writer_pool();
        let store: Arc<dyn SuperfileReaderCache> = Arc::new(InMemoryReaderCache::new());

        Ok(Self {
            schema,
            id_column,
            fts_columns,
            vector_columns,
            tokenizer,
            reader_pool,
            writer_pool,
            store,
            storage: None,
            disk_cache: None,
            manifest_disk_cache: None,
            memory_budget_bytes: None,
            // Placeholder: standalone options (tests, direct callers) get an unshared measure-only budget.
            // The catalog's `build_options` overwrites this with the connection's shared budget, and
            // `apply_config` replaces it from `config.yaml`.
            connection_memory_budget: ConnectionMemoryBudget::measured(),
            prepopulate_cache_on_commit: true,
            partition_strategy: None,
            target_superfiles_per_part: DEFAULT_TARGET_SUPERFILES_PER_PART,
            part_size_threshold_bytes: DEFAULT_PART_SIZE_THRESHOLD_BYTES,
            eager_load_threshold_parts: DEFAULT_EAGER_LOAD_THRESHOLD_PARTS,
            max_commit_retries: DEFAULT_MAX_COMMIT_RETRIES,
            commit_threshold_size_mb: DEFAULT_COMMIT_THRESHOLD_SIZE_MB,
            put_multipart_threshold_bytes: DEFAULT_PUT_MULTIPART_THRESHOLD_BYTES,
            verify_crc_on_open: true,
            read_consistency: Consistency::default(),
        })
    }

    /// Schema as the user supplied it — the shape that
    /// `Supertable::append`'s callers build their batches
    /// against. By contract the user schema never contains
    /// the id column; the supertable injects it at append
    /// time.
    pub fn user_schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    /// Schema with the id column prepended — what the writer's
    /// commit path hands to `SuperfileBuilder` and what Parquet
    /// stores.
    pub fn effective_schema(&self) -> Arc<Schema> {
        let mut fields = vec![Arc::new(Field::new(
            &self.id_column,
            DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE),
            false,
        ))];
        fields.extend(self.schema.fields().iter().cloned());
        Arc::new(Schema::new(fields))
    }

    /// Resolve the effective partition strategy for this
    /// supertable. Called at [`Supertable::create`] time
    /// when nothing's been persisted yet. The default —
    /// `Hash { column: id_column, n_buckets: 1 }` — is
    /// observationally equivalent to "no partitioning";
    /// callers wanting real partitioning set
    /// [`Self::partition_strategy`] via
    /// [`Self::with_partition_strategy`].
    pub fn effective_partition_strategy(&self) -> PartitionStrategy {
        self.partition_strategy
            .clone()
            .unwrap_or(PartitionStrategy::IngestionTime {
                granularity_secs: 86_400,
            })
    }

    /// Override the name of the supertable-injected id
    /// column. Useful when `_id` collides with a business
    /// field name; the column type and generation semantics
    /// stay fixed regardless of the name.
    ///
    /// Rejects names that already appear in the user schema
    /// (same check as construction) by returning
    /// [`BuildError::IdColumnReserved`].
    pub fn with_id_column(mut self, name: impl Into<String>) -> Result<Self, BuildError> {
        let name = name.into();
        if self.schema.fields().iter().any(|f| f.name() == &name) {
            return Err(BuildError::IdColumnReserved(name));
        }
        self.id_column = name;
        Ok(self)
    }

    /// Override the reader thread pool. Useful for tests
    /// (deterministic single-thread pool) or for callers wiring
    /// up a shared pool across subsystems.
    pub fn with_reader_pool(mut self, pool: Arc<ThreadPool>) -> Self {
        self.reader_pool = pool;
        self
    }

    /// Override the writer thread pool.
    pub fn with_writer_pool(mut self, pool: Arc<ThreadPool>) -> Self {
        self.writer_pool = pool;
        self
    }

    /// Set the read-path freshness policy. See [`Consistency`].
    /// Default: [`Consistency::BoundedStaleness`] with a 1s window.
    pub fn with_read_consistency(mut self, consistency: Consistency) -> Self {
        self.read_consistency = consistency;
        self
    }

    /// Override the superfile store. Default is
    /// [`InMemoryReaderCache`]; tests + production deployments
    /// with persistence swap this for an mmap- or object-store-
    /// backed implementation.
    pub fn with_store(mut self, store: Arc<dyn SuperfileReaderCache>) -> Self {
        self.store = store;
        self
    }

    /// Attach an object-store backend. Engages the
    /// write-through path: each successful commit persists
    /// superfile bytes + the new manifest (parts + list +
    /// pointer) to storage via the `commit_manifest`
    /// primitive.
    ///
    /// `None`-equivalent shape: don't call this method —
    /// the supertable then runs in-memory only.
    ///
    /// Reads still go through `store` unless a `disk_cache`
    /// is also attached.
    pub fn with_storage(mut self, storage: Arc<dyn StorageProvider>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Attach a disk cache for storage-backed reads.
    /// Must be paired with [`Self::with_storage`]; attaching a
    /// cache without storage is caught at create / open time.
    ///
    /// When attached:
    ///   - The writer's commit path **skips** the in-memory
    ///     `store.put` — superfile bytes go to object storage
    ///     only, and the cache hydrates lazily on first query.
    ///     This removes the OOM trap at 100GB scale (the
    ///     in-memory `SuperfileReaderCache` doesn't evict, so a
    ///     long-running writer would otherwise accumulate every
    ///     superfile's bytes in RAM forever).
    ///   - Reader paths route superfile-byte lookups through the
    ///     cache (in-memory tier checked first for hot writes
    ///     made in this process, then disk cache, then
    ///     cold-fetch from object storage).
    ///
    /// Cache construction stays user-managed. Build the
    /// [`DiskCacheStore`] yourself with whatever `pinned_fn` /
    /// budget / eviction policy fits the deployment; pass the
    /// resulting `Arc<DiskCacheStore>` here.
    pub fn with_disk_cache(mut self, cache: Arc<DiskCacheStore>) -> Self {
        self.disk_cache = Some(cache);
        self
    }

    /// Attach an on-disk cache for compressed manifest-part bytes.
    /// On a cache hit the part loader reads bytes from local disk
    /// instead of round-tripping to object storage. Construction stays
    /// user-managed — build the [`ManifestDiskCache`] with the cache
    /// root and byte budget that fit the deployment and pass the
    /// resulting `Arc` here.
    pub fn with_manifest_disk_cache(mut self, cache: Arc<ManifestDiskCache>) -> Self {
        self.manifest_disk_cache = Some(cache);
        self
    }

    /// Attach a best-effort memory budget for the disk
    /// cache's mmap working set. See
    /// [`Self::memory_budget_bytes`] for semantics. No-op
    /// without [`Self::with_disk_cache`] attached — the
    /// budget only steers the cache's sweep behavior.
    pub fn with_memory_budget(mut self, budget_bytes: u64) -> Self {
        self.memory_budget_bytes = Some(budget_bytes);
        self
    }

    /// Enable/disable post-commit disk-cache pre-population. See
    /// [`Self::prepopulate_cache_on_commit`]. Pass `false` for a
    /// write-only producer (ingest then drop) to skip the wasted
    /// warm-fill and its "budget exceeded" log spam.
    pub fn with_cache_prepopulation(mut self, enabled: bool) -> Self {
        self.prepopulate_cache_on_commit = enabled;
        self
    }

    /// Set the partition strategy. Stamped into the manifest
    /// list at first commit; immutable thereafter (changes
    /// require external compaction). Without this call,
    /// [`Self::effective_partition_strategy`] returns the
    /// single-bucket Hash default.
    pub fn with_partition_strategy(mut self, strategy: PartitionStrategy) -> Self {
        self.partition_strategy = Some(strategy);
        self
    }

    /// Override the soft cap on superfiles per manifest part.
    /// Default `10_000`.
    pub fn with_target_superfiles_per_part(mut self, n: u64) -> Self {
        self.target_superfiles_per_part = n;
        self
    }

    /// Override the soft cap on a manifest part's compressed
    /// size in bytes. Default `10 MiB`.
    pub fn with_part_size_threshold_bytes(mut self, n: u64) -> Self {
        self.part_size_threshold_bytes = n;
        self
    }

    /// Override the eager-load threshold for manifest parts
    /// at [`Supertable::open`] time. See
    /// [`Self::eager_load_threshold_parts`] for semantics.
    /// Default `4`. Set to `0` to force lazy-load even on
    /// tiny manifests (test-friendly).
    pub fn with_eager_load_threshold(mut self, n: u32) -> Self {
        self.eager_load_threshold_parts = n;
        self
    }

    /// Override the max OCC retry attempts before
    /// `writer.commit()` surfaces
    /// `CommitError::WriteContentionExhausted`. See
    /// [`Self::max_commit_retries`] for semantics. Default
    /// `10`.
    pub fn with_max_commit_retries(mut self, n: u32) -> Self {
        self.max_commit_retries = n;
        self
    }

    /// Override the auto-flush threshold (MiB).
    pub fn with_commit_threshold_size_mb(mut self, mb: u64) -> Self {
        self.commit_threshold_size_mb = mb;
        self
    }

    /// Override the superfile-size threshold (bytes) at which
    /// the writer routes through `put_multipart` instead of
    /// `put_atomic`. See [`Self::put_multipart_threshold_bytes`].
    /// Default `100 MiB`.
    pub fn with_put_multipart_threshold_bytes(mut self, n: u64) -> Self {
        self.put_multipart_threshold_bytes = n;
        self
    }

    /// Override whether the supertable's `SuperfileReader::open`
    /// calls verify CRC on the embedded vector blob. Default
    /// `true`. See [`Self::verify_crc_on_open`].
    pub fn with_verify_crc_on_open(mut self, v: bool) -> Self {
        self.verify_crc_on_open = v;
        self
    }

    /// Build the `SuperfileReader::open_with` options that
    /// match the current `verify_crc_on_open` setting. Used
    /// by every supertable-internal callsite that opens a
    /// superfile so the global config knob applies uniformly.
    pub(crate) fn superfile_open_options(&self) -> OpenOptions {
        OpenOptions {
            verify_crc: self.verify_crc_on_open,
        }
    }

    test_visible! {
        /// The connection memory budget these options carry. Exposed for
        /// integration tests that assert budget accounting (peak / denials)
        /// after a query, confirming the budget was wired to the query path.
        fn connection_budget(&self) -> &Arc<ConnectionMemoryBudget> {
            &self.connection_memory_budget
        }
    }

    /// Apply system [`Config`] to this `SupertableOptions`.
    /// Rebuilds the reader / writer thread pools, copies
    /// supertable knobs, and attaches configured persistent
    /// storage + disk cache.
    ///
    /// `auto` thread counts resolve to `num_cpus` (reader) and
    /// `max(1, num_cpus / 2)` (writer). Explicit integers are used
    /// as-is (clamped to ≥ 1).
    ///
    /// The schema, FTS / vector configuration, tokenizer, and
    /// in-memory superfile store are preserved. If
    /// `cfg.storage.backend` is not `none`, this method attaches
    /// the requested storage provider; if
    /// `cfg.storage.disk_cache_root` is set, it also attaches a
    /// `DiskCacheStore` configured from the same storage section.
    ///
    /// Rejects an id-column name from config that conflicts with
    /// a user-schema field — same check as
    /// [`Self::with_id_column`].
    pub fn apply_config(mut self, cfg: &Config) -> Result<Self, BuildError> {
        // `Auto` keeps the shared pool; `Fixed` builds a dedicated one
        // (the config analogue of `with_reader_pool` / `with_writer_pool`).
        self.reader_pool = match cfg.supertable.reader_threads {
            ThreadCount::Auto => shared_reader_pool(),
            ThreadCount::Fixed(n) => Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(n.max(1))
                    .thread_name(|i| format!("supertable-reader-{i}"))
                    .build()
                    .map_err(|e| BuildError::ThreadPoolCreation(e.to_string()))?,
            ),
        };
        self.writer_pool = match cfg.supertable.writer_threads {
            ThreadCount::Auto => shared_writer_pool(),
            ThreadCount::Fixed(n) => Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(n.max(1))
                    .thread_name(|i| format!("supertable-writer-{i}"))
                    .build()
                    .map_err(|e| BuildError::ThreadPoolCreation(e.to_string()))?,
            ),
        };
        self.commit_threshold_size_mb = cfg.supertable.commit_threshold_size_mb;
        self.verify_crc_on_open = cfg.supertable.verify_crc_on_open;
        // The `config.yaml` source for the connection budget; the connect path
        // uses `ConnectOptions` instead. 0, the shipped default, is measure-only.
        // Note this replaces the budget outright: don't call `apply_config` on
        // options that already carry a shared connection budget, or the sharing
        // is silently dropped.
        self.connection_memory_budget =
            ConnectionMemoryBudget::from_budget_bytes(cfg.memory.connection_budget_bytes);
        if cfg.supertable.id_column != self.id_column {
            if self
                .schema
                .fields()
                .iter()
                .any(|f| f.name() == &cfg.supertable.id_column)
            {
                return Err(BuildError::IdColumnReserved(
                    cfg.supertable.id_column.clone(),
                ));
            }
            self.id_column = cfg.supertable.id_column.clone();
        }
        self.apply_storage_config(cfg)?;
        Ok(self)
    }

    fn apply_storage_config(&mut self, cfg: &Config) -> Result<(), BuildError> {
        let storage: Option<Arc<dyn StorageProvider>> = match cfg.storage.backend {
            StorageBackend::None => None,
            StorageBackend::LocalFs => {
                let root = cfg.storage.local_root.as_ref().ok_or_else(|| {
                    BuildError::Store("storage.backend=local_fs requires storage.local_root".into())
                })?;
                Some(Arc::new(LocalFsStorageProvider::new(root)?) as Arc<dyn StorageProvider>)
            }
            StorageBackend::S3 => {
                let bucket = cfg.storage.bucket.as_ref().ok_or_else(|| {
                    BuildError::Store("storage.backend=s3 requires storage.bucket".into())
                })?;
                Some(Arc::new(S3StorageProvider::new_with_prefix(
                    bucket,
                    &cfg.storage.prefix,
                    &cfg.storage.storage_options,
                )?) as Arc<dyn StorageProvider>)
            }
            StorageBackend::Azure => {
                let container = cfg.storage.bucket.as_ref().ok_or_else(|| {
                    BuildError::Store("storage.backend=azure requires storage.bucket".into())
                })?;
                Some(Arc::new(AzureStorageProvider::new_with_prefix(
                    container,
                    &cfg.storage.prefix,
                    &cfg.storage.storage_options,
                )?) as Arc<dyn StorageProvider>)
            }
            StorageBackend::Gcs => {
                let bucket = cfg.storage.bucket.as_ref().ok_or_else(|| {
                    BuildError::Store("storage.backend=gcs requires storage.bucket".into())
                })?;
                Some(Arc::new(GcsStorageProvider::new_with_prefix(
                    bucket,
                    &cfg.storage.prefix,
                    &cfg.storage.storage_options,
                )?) as Arc<dyn StorageProvider>)
            }
        };

        let Some(storage) = storage else {
            return Ok(());
        };

        if let Some(cache_root) = cfg.storage.disk_cache_root.as_ref() {
            let cold_fetch_mode = match cfg.storage.cold_fetch_mode {
                StorageColdFetchMode::HybridWithPrefetch => ColdFetchMode::HybridWithPrefetch,
                StorageColdFetchMode::RangeOnly => ColdFetchMode::RangeOnly,
                StorageColdFetchMode::LazyForegroundWithBackgroundFill => {
                    ColdFetchMode::LazyForegroundWithBackgroundFill
                }
            };
            let disk_cfg = DiskCacheConfig {
                cache_root: cache_root.clone(),
                disk_budget_bytes: cfg.storage.disk_budget_bytes,
                cold_fetch_mode,
                cold_fetch_streams: cfg.storage.cold_fetch_streams.max(1),
                cold_fetch_chunk_bytes: cfg.storage.cold_fetch_chunk_bytes.max(1),
                prefetch_concurrency: cfg.storage.prefetch_concurrency.max(1),
                mmap_cold_threshold_secs: cfg.storage.mmap_cold_threshold_secs,
                mmap_sweep_interval_secs: cfg.storage.mmap_sweep_interval_secs,
                eviction: Box::new(LruPolicy::new()),
                verify_crc_on_open: cfg.supertable.verify_crc_on_open,
            };
            let cache = DiskCacheStore::new_unpinned(Arc::clone(&storage), disk_cfg)
                .map_err(|e| BuildError::Store(format!("disk cache construction: {e}")))?;
            self.disk_cache = Some(cache);

            // Manifest-part bytes get their own content-addressed cache
            // under a sibling subdirectory, with an independent budget.
            let manifest_cache_root = cache_root.join(MANIFEST_CACHE_SUBDIR);
            let manifest_cache =
                ManifestDiskCache::new(manifest_cache_root, cfg.storage.manifest_disk_budget_bytes)
                    .map_err(|e| {
                        BuildError::Store(format!("manifest disk cache construction: {e}"))
                    })?;
            self.manifest_disk_cache = Some(manifest_cache);
        }

        self.storage = Some(storage);
        Ok(())
    }

    /// Construct a `superfile::BuilderOptions` for one rayon
    /// shard worker at commit time. The shard worker constructs
    /// its own `SuperfileBuilder` from this and feeds its slice
    /// of buffered batches into it.
    ///
    /// Differences from a "natural" superfile config:
    /// - Schema is the **scalar-only** schema
    ///   ([`SupertableOptions::scalar_schema`]) — vector columns
    ///   live in the embedded vector blob, never in Parquet.
    ///   The id column is prepended (Decimal128(38, 0)).
    pub fn builder_options(&self) -> BuilderOptions {
        BuilderOptions::new(
            self.scalar_schema(),
            self.id_column.clone(),
            self.fts_columns.clone(),
            self.vector_columns.clone(),
            self.tokenizer.clone(),
        )
    }

    /// Effective scalar-only schema — the user's columns with
    /// vector columns projected out *and* the supertable-
    /// injected id column prepended. This is what the
    /// underlying `SuperfileBuilder` sees and what Parquet
    /// stores.
    ///
    /// Vectors live in the embedded vector blob, never in
    /// Parquet, so they don't appear here. The id column is
    /// always first.
    ///
    /// Cost is one schema-walk + one `Vec::clone` of the
    /// surviving fields per call. Caching on first call is a
    /// future optimization if benches show this on the hot
    /// path.
    pub fn scalar_schema(&self) -> Arc<Schema> {
        let vector_names: HashSet<&str> = self
            .vector_columns
            .iter()
            .map(|vc| vc.column.as_str())
            .collect();
        let mut kept: Vec<Arc<Field>> = Vec::with_capacity(self.schema.fields().len() + 1);
        kept.push(Arc::new(Field::new(
            &self.id_column,
            DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE),
            false,
        )));
        kept.extend(
            self.schema
                .fields()
                .iter()
                .filter(|f| !vector_names.contains(f.name().as_str()))
                .cloned(),
        );
        Arc::new(Schema::new(kept))
    }
}

impl fmt::Debug for SupertableOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SupertableOptions")
            .field("schema_fields", &self.schema.fields().len())
            .field("id_column", &self.id_column)
            .field("n_fts_columns", &self.fts_columns.len())
            .field("n_vector_columns", &self.vector_columns.len())
            .field("has_tokenizer", &self.tokenizer.is_some())
            .finish()
    }
}

/// Reject user-supplied column names that would collide with
/// infino's internal byte-protocol or KV-key conventions.
/// Same rules as `superfile::builder::check_user_column_name`
/// — mirrored here so the typed error surfaces at the
/// supertable-options layer before any `SuperfileBuilder` is
/// constructed downstream.
fn check_user_column_name(name: &str) -> Result<(), BuildError> {
    if name.contains(RESERVED_SEPARATOR) {
        return Err(BuildError::ReservedSeparatorInColumnName(name.to_string()));
    }
    if name.starts_with(RESERVED_PREFIX) {
        return Err(BuildError::ReservedPrefixInColumnName(name.to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{env, fs, sync::Arc};

    use arrow_schema::{DataType, Field};
    use uuid::Uuid;

    use super::*;
    use crate::superfile::vector::{distance::Metric, rerank_codec::RerankCodec};

    fn fixed_list_f32(dim: usize) -> DataType {
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        )
    }

    /// Minimal user schema with one scalar + one vector column.
    /// The user schema must NOT contain the id column — the
    /// supertable injects it at append time.
    fn schema_with_vector(dim: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", fixed_list_f32(dim), false),
        ]))
    }

    fn vc(name: &str, dim: usize) -> VectorConfig {
        VectorConfig {
            column: name.into(),
            dim,
            n_cent: 4,
            rot_seed: 0,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Fp32,
        }
    }

    fn fc(name: &str) -> FtsConfig {
        FtsConfig {
            column: name.into(),
        }
    }

    use crate::test_helpers::default_tokenizer as tok;

    #[test]
    fn valid_options_with_fts_and_vector_succeeds() {
        let s = schema_with_vector(16);
        let opts = SupertableOptions::new(s, vec![fc("title")], vec![vc("emb", 16)], Some(tok()))
            .expect("valid options should succeed");
        assert_eq!(opts.id_column, "_id");
        assert_eq!(opts.fts_columns.len(), 1);
        assert_eq!(opts.vector_columns.len(), 1);
    }

    #[test]
    fn schema_that_contains_id_column_is_rejected() {
        // The user schema must NOT contain a field named the
        // same as the configured id column. The supertable
        // injects that column at append time; a collision
        // would produce a duplicate-column error at first
        // append, so we surface a typed error at construction
        // instead.
        let s = Arc::new(Schema::new(vec![
            Field::new("_id", DataType::UInt64, false),
            Field::new("emb", fixed_list_f32(16), false),
        ]));
        let err = SupertableOptions::new(s, vec![], vec![vc("emb", 16)], None)
            .expect_err("expected error");
        assert!(matches!(err, BuildError::IdColumnReserved(c) if c == "_id"));
    }

    #[test]
    fn fts_column_missing_from_schema_rejected() {
        let s = schema_with_vector(16);
        let err = SupertableOptions::new(s, vec![fc("absent")], vec![], Some(tok()))
            .expect_err("expected error");
        assert!(matches!(err, BuildError::FtsColumnMissing { column } if column == "absent"));
    }

    #[test]
    fn fts_column_wrong_type_rejected() {
        // `body` is Utf8 not LargeUtf8 — must reject.
        let s = Arc::new(Schema::new(vec![Field::new("body", DataType::Utf8, false)]));
        let err = SupertableOptions::new(s, vec![fc("body")], vec![], Some(tok()))
            .expect_err("expected error");
        assert!(
            matches!(err, BuildError::FtsColumnMustBeLargeUtf8 { column, .. } if column == "body")
        );
    }

    #[test]
    fn vector_column_missing_from_schema_rejected() {
        let s = Arc::new(Schema::new(vec![Field::new(
            "category",
            DataType::Utf8,
            false,
        )]));
        let err = SupertableOptions::new(s, vec![], vec![vc("emb", 16)], None)
            .expect_err("expected error");
        assert!(matches!(err, BuildError::VectorColumnMissing { column } if column == "emb"));
    }

    #[test]
    fn vector_column_not_fixed_size_list_rejected() {
        // emb is Float32 scalar instead of FixedSizeList.
        let s = Arc::new(Schema::new(vec![Field::new(
            "emb",
            DataType::Float32,
            false,
        )]));
        let err = SupertableOptions::new(s, vec![], vec![vc("emb", 16)], None)
            .expect_err("expected error");
        assert!(matches!(
            err,
            BuildError::VectorColumnNotFixedSizeList { column, .. } if column == "emb"
        ));
    }

    #[test]
    fn vector_column_wrong_inner_type_rejected() {
        // FixedSizeList<Float64> instead of Float32.
        let s = Arc::new(Schema::new(vec![Field::new(
            "emb",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 16),
            false,
        )]));
        let err = SupertableOptions::new(s, vec![], vec![vc("emb", 16)], None)
            .expect_err("expected error");
        assert!(matches!(
            err,
            BuildError::VectorColumnNotFixedSizeList { column, .. } if column == "emb"
        ));
    }

    #[test]
    fn vector_column_dim_mismatch_rejected() {
        // Schema declares list_size=8 but config asks for dim=16.
        let s = Arc::new(Schema::new(vec![Field::new(
            "emb",
            fixed_list_f32(8),
            false,
        )]));
        let err = SupertableOptions::new(s, vec![], vec![vc("emb", 16)], None)
            .expect_err("expected error");
        assert!(matches!(
            err,
            BuildError::VectorColumnDimMismatch { expected: 16, actual: 8, column } if column == "emb"
        ));
    }

    #[test]
    fn vector_dim_below_min_rejected() {
        let s = Arc::new(Schema::new(vec![Field::new(
            "emb",
            fixed_list_f32(8),
            false,
        )]));
        let err = SupertableOptions::new(s, vec![], vec![vc("emb", 8)], None)
            .expect_err("expected error");
        assert!(matches!(
            err,
            BuildError::VectorDimOutOfRange { column, dim: 8 } if column == "emb"
        ));
    }

    #[test]
    fn vector_dim_above_max_rejected() {
        let s = Arc::new(Schema::new(vec![Field::new(
            "emb",
            fixed_list_f32(8192),
            false,
        )]));
        let err = SupertableOptions::new(s, vec![], vec![vc("emb", 8192)], None)
            .expect_err("expected error");
        assert!(matches!(
            err,
            BuildError::VectorDimOutOfRange { column, dim: 8192 } if column == "emb"
        ));
    }

    #[test]
    fn duplicate_logical_name_across_fts_and_vector_rejected() {
        let s = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", fixed_list_f32(16), false),
        ]));
        // Duplicate `emb` between two vector_columns entries —
        // hits the cross-list dedup check.
        let err =
            SupertableOptions::new(s.clone(), vec![], vec![vc("emb", 16), vc("emb", 16)], None)
                .expect_err("expected error");
        assert!(matches!(err, BuildError::DuplicateLogicalName(n) if n == "emb"));
    }

    #[test]
    fn reserved_separator_in_fts_column_name_rejected() {
        let s = Arc::new(Schema::new(vec![Field::new(
            "ti\u{1F}tle",
            DataType::LargeUtf8,
            false,
        )]));
        let err = SupertableOptions::new(s, vec![fc("ti\u{1F}tle")], vec![], Some(tok()))
            .expect_err("expected error");
        assert!(matches!(err, BuildError::ReservedSeparatorInColumnName(_)));
    }

    #[test]
    fn reserved_prefix_in_vector_column_name_rejected() {
        let s = Arc::new(Schema::new(vec![Field::new(
            "inf.emb",
            fixed_list_f32(16),
            false,
        )]));
        let err = SupertableOptions::new(s, vec![], vec![vc("inf.emb", 16)], None)
            .expect_err("expected error");
        assert!(matches!(err, BuildError::ReservedPrefixInColumnName(_)));
    }

    #[test]
    fn fts_columns_without_tokenizer_rejected() {
        let s = Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]));
        let err =
            SupertableOptions::new(s, vec![fc("title")], vec![], None).expect_err("expected error");
        assert!(matches!(err, BuildError::MissingTokenizer));
    }

    #[test]
    fn empty_fts_and_vector_succeeds_without_tokenizer() {
        let s = Arc::new(Schema::new(vec![Field::new(
            "category",
            DataType::Utf8,
            false,
        )]));
        // No FTS, no vectors, no tokenizer — the supertable becomes
        // a thin wrapper over scalar Parquet data. Must succeed.
        SupertableOptions::new(s, vec![], vec![], None).expect("empty fts + vector should succeed");
    }

    #[test]
    fn scalar_schema_drops_vector_columns_and_prepends_id() {
        let s = schema_with_vector(16);
        let opts = SupertableOptions::new(s, vec![fc("title")], vec![vc("emb", 16)], Some(tok()))
            .expect("valid options");
        let scalar = opts.scalar_schema();
        let names: Vec<_> = scalar.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["_id", "title"]);
        // _id field is `Decimal128(38, 0)`.
        let id_field = scalar.field(0);
        assert_eq!(
            id_field.data_type(),
            &DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE)
        );
    }

    #[test]
    fn scalar_schema_no_vector_columns_still_prepends_id() {
        let s = Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]));
        let opts = SupertableOptions::new(s, vec![fc("title")], vec![], Some(tok()))
            .expect("valid options");
        let scalar = opts.scalar_schema();
        let names: Vec<_> = scalar.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["_id", "title"]);
    }

    #[test]
    fn effective_schema_prepends_id_keeps_vector_columns() {
        let s = schema_with_vector(16);
        let opts = SupertableOptions::new(s, vec![fc("title")], vec![vc("emb", 16)], Some(tok()))
            .expect("valid options");
        let eff = opts.effective_schema();
        let names: Vec<_> = eff.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["_id", "title", "emb"]);
    }

    #[test]
    fn user_schema_returns_input_schema_unchanged() {
        let s = schema_with_vector(16);
        let opts = SupertableOptions::new(
            Arc::clone(&s),
            vec![fc("title")],
            vec![vc("emb", 16)],
            Some(tok()),
        )
        .expect("valid options");
        let us = opts.user_schema();
        assert_eq!(us.fields().len(), s.fields().len());
        for (a, b) in us.fields().iter().zip(s.fields().iter()) {
            assert_eq!(a.name(), b.name());
        }
    }

    #[test]
    fn with_id_column_overrides_default() {
        let s = schema_with_vector(16);
        let opts = SupertableOptions::new(s, vec![fc("title")], vec![vc("emb", 16)], Some(tok()))
            .expect("valid options")
            .with_id_column("row_id")
            .expect("override accepted");
        assert_eq!(opts.id_column, "row_id");
        // effective_schema now uses the new name.
        let eff = opts.effective_schema();
        assert_eq!(eff.field(0).name(), "row_id");
    }

    #[test]
    fn with_id_column_rejects_name_that_collides_with_user_schema() {
        let s = schema_with_vector(16);
        let opts = SupertableOptions::new(s, vec![fc("title")], vec![vc("emb", 16)], Some(tok()))
            .expect("valid options");
        let err = opts.with_id_column("title").expect_err("collision");
        assert!(matches!(err, BuildError::IdColumnReserved(c) if c == "title"));
    }

    #[test]
    fn apply_config_sets_writer_pool_size_to_fixed_value() {
        use figment::{
            Figment,
            providers::{Format, Yaml},
        };

        let yaml = r#"
supertable:
  reader_threads: 3
  writer_threads: 5
  commit_threshold_size_mb: 7
"#;
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(yaml))).expect("parse config");

        let s = schema_with_vector(16);
        let opts = SupertableOptions::new(s, vec![fc("title")], vec![vc("emb", 16)], Some(tok()))
            .expect("valid options")
            .apply_config(&cfg)
            .expect("apply_config");

        assert_eq!(opts.commit_threshold_size_mb, 7);
        assert_eq!(opts.reader_pool.current_num_threads(), 3);
        assert_eq!(opts.writer_pool.current_num_threads(), 5);
    }

    #[test]
    fn apply_config_sets_connection_memory_budget_from_config() {
        use figment::{
            Figment,
            providers::{Format, Yaml},
        };

        // A positive config value -> bounded budget, gated at 90%.
        let yaml = "memory:\n  connection_budget_bytes: 1000\n";
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(yaml))).expect("parse config");
        let opts = plain_opts().apply_config(&cfg).expect("apply_config");
        assert_eq!(opts.connection_memory_budget.limit(), Some(900));
    }

    #[test]
    fn apply_config_default_leaves_connection_memory_budget_measured() {
        // The shipped default (0) is measure-only.
        let cfg = Config::defaults().expect("embedded default");
        let opts = plain_opts().apply_config(&cfg).expect("apply_config");
        assert_eq!(opts.connection_memory_budget.limit(), None);
    }

    #[test]
    fn apply_config_auto_resolves_to_num_cpus_defaults() {
        // `auto` is the embedded default; verify resolution clamps
        // ≥ 1 and uses num_cpus-derived defaults.
        let cfg = Config::defaults().expect("embedded default");

        let s = schema_with_vector(16);
        let opts = SupertableOptions::new(s, vec![fc("title")], vec![vc("emb", 16)], Some(tok()))
            .expect("valid options")
            .apply_config(&cfg)
            .expect("apply_config");

        let reader_default = num_cpus::get().max(1);
        let writer_default = num_cpus::get().div_ceil(2).max(1);
        assert_eq!(opts.reader_pool.current_num_threads(), reader_default);
        assert_eq!(opts.writer_pool.current_num_threads(), writer_default);
        assert_eq!(opts.commit_threshold_size_mb, 1024);
        assert_eq!(opts.id_column, "_id");
    }

    #[test]
    fn debug_format_doesnt_explode() {
        let s = schema_with_vector(16);
        let opts = SupertableOptions::new(s, vec![fc("title")], vec![vc("emb", 16)], Some(tok()))
            .expect("valid options");
        let s = format!("{:?}", opts);
        assert!(s.contains("SupertableOptions"));
        assert!(s.contains("_id"));
    }

    /// Helper: a minimal valid options instance (no FTS / vector) for
    /// exercising the builder methods that don't touch the schema.
    fn plain_opts() -> SupertableOptions {
        let s = Arc::new(Schema::new(vec![Field::new(
            "category",
            DataType::Utf8,
            false,
        )]));
        SupertableOptions::new(s, vec![], vec![], None).expect("valid options")
    }

    #[test]
    fn consistency_default_is_bounded_staleness_one_sec() {
        assert_eq!(
            Consistency::default(),
            Consistency::BoundedStaleness(Duration::from_secs(DEFAULT_READ_STALENESS_SECS))
        );
    }

    #[test]
    fn new_sets_documented_defaults() {
        let opts = plain_opts();
        assert!(opts.prepopulate_cache_on_commit);
        assert!(opts.verify_crc_on_open);
        assert!(opts.storage.is_none());
        assert!(opts.disk_cache.is_none());
        assert!(opts.memory_budget_bytes.is_none());
        assert!(opts.partition_strategy.is_none());
        assert_eq!(
            opts.target_superfiles_per_part,
            DEFAULT_TARGET_SUPERFILES_PER_PART
        );
        assert_eq!(
            opts.part_size_threshold_bytes,
            DEFAULT_PART_SIZE_THRESHOLD_BYTES
        );
        assert_eq!(
            opts.eager_load_threshold_parts,
            DEFAULT_EAGER_LOAD_THRESHOLD_PARTS
        );
        assert_eq!(opts.max_commit_retries, DEFAULT_MAX_COMMIT_RETRIES);
        assert_eq!(
            opts.commit_threshold_size_mb,
            DEFAULT_COMMIT_THRESHOLD_SIZE_MB
        );
        assert_eq!(
            opts.put_multipart_threshold_bytes,
            DEFAULT_PUT_MULTIPART_THRESHOLD_BYTES
        );
        assert_eq!(opts.read_consistency, Consistency::default());
    }

    #[test]
    fn effective_partition_strategy_defaults_to_ingestion_time() {
        let opts = plain_opts();
        match opts.effective_partition_strategy() {
            PartitionStrategy::IngestionTime { granularity_secs } => {
                assert_eq!(granularity_secs, 86_400);
            }
            other => panic!("expected IngestionTime with 1-day granularity, got {other:?}"),
        }
    }

    #[test]
    fn effective_partition_strategy_returns_configured_strategy() {
        let strat = PartitionStrategy::Hash {
            column: "category".into(),
            n_buckets: 64,
        };
        let opts = plain_opts().with_partition_strategy(strat.clone());
        assert_eq!(opts.effective_partition_strategy(), strat);
    }

    #[test]
    fn scalar_threshold_builders_set_their_fields() {
        let opts = plain_opts()
            .with_target_superfiles_per_part(42)
            .with_part_size_threshold_bytes(4096)
            .with_eager_load_threshold(0)
            .with_max_commit_retries(99)
            .with_commit_threshold_size_mb(7)
            .with_put_multipart_threshold_bytes(1)
            .with_memory_budget(1 << 30)
            .with_cache_prepopulation(false)
            .with_verify_crc_on_open(false);
        assert_eq!(opts.target_superfiles_per_part, 42);
        assert_eq!(opts.part_size_threshold_bytes, 4096);
        assert_eq!(opts.eager_load_threshold_parts, 0);
        assert_eq!(opts.max_commit_retries, 99);
        assert_eq!(opts.commit_threshold_size_mb, 7);
        assert_eq!(opts.put_multipart_threshold_bytes, 1);
        assert_eq!(opts.memory_budget_bytes, Some(1 << 30));
        assert!(!opts.prepopulate_cache_on_commit);
        assert!(!opts.verify_crc_on_open);
    }

    #[test]
    fn with_read_consistency_overrides_default() {
        let opts = plain_opts().with_read_consistency(Consistency::Strong);
        assert_eq!(opts.read_consistency, Consistency::Strong);
        let opts = opts.with_read_consistency(Consistency::Snapshot);
        assert_eq!(opts.read_consistency, Consistency::Snapshot);
    }

    #[test]
    fn with_reader_and_writer_pool_override_pools() {
        let reader = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("reader pool"),
        );
        let writer = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(3)
                .build()
                .expect("writer pool"),
        );
        let opts = plain_opts()
            .with_reader_pool(Arc::clone(&reader))
            .with_writer_pool(Arc::clone(&writer));
        assert_eq!(opts.reader_pool.current_num_threads(), 2);
        assert_eq!(opts.writer_pool.current_num_threads(), 3);
        assert!(Arc::ptr_eq(&opts.reader_pool, &reader));
        assert!(Arc::ptr_eq(&opts.writer_pool, &writer));
    }

    #[test]
    fn with_store_replaces_default_store() {
        let store: Arc<dyn SuperfileReaderCache> = Arc::new(InMemoryReaderCache::new());
        let opts = plain_opts().with_store(Arc::clone(&store));
        // The Arc held by opts is the one we passed in.
        let opts_store: Arc<dyn SuperfileReaderCache> = Arc::clone(&opts.store);
        assert!(Arc::ptr_eq(&opts_store, &store));
    }

    #[test]
    fn with_storage_attaches_provider() {
        let dir = env::temp_dir().join(format!("infino-opts-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("mkdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
        let opts = plain_opts().with_storage(storage);
        assert!(opts.storage.is_some());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn superfile_open_options_track_verify_crc_flag() {
        let opts = plain_opts();
        assert!(opts.superfile_open_options().verify_crc);
        let opts = opts.with_verify_crc_on_open(false);
        assert!(!opts.superfile_open_options().verify_crc);
    }

    #[test]
    fn builder_options_use_scalar_schema_and_id_column() {
        let s = schema_with_vector(16);
        let opts = SupertableOptions::new(s, vec![fc("title")], vec![vc("emb", 16)], Some(tok()))
            .expect("valid options");
        let bo = opts.builder_options();
        // builder_options carries the scalar-only schema (vectors
        // dropped, id prepended) and the same id column + role configs.
        let names: Vec<_> = bo
            .schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, vec!["_id", "title"]);
        assert_eq!(bo.id_column, "_id");
        assert_eq!(bo.fts_columns.len(), 1);
        assert_eq!(bo.vector_columns.len(), 1);
    }

    #[test]
    fn apply_config_overrides_id_column_when_no_collision() {
        use figment::{
            Figment,
            providers::{Format, Yaml},
        };

        let yaml = r#"
supertable:
  id_column: row_pk
"#;
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(yaml))).expect("parse config");
        let opts = plain_opts().apply_config(&cfg).expect("apply_config");
        assert_eq!(opts.id_column, "row_pk");
    }

    #[test]
    fn apply_config_rejects_id_column_that_collides_with_schema() {
        use figment::{
            Figment,
            providers::{Format, Yaml},
        };

        // Config id column collides with the user-schema field
        // `category`.
        let yaml = r#"
supertable:
  id_column: category
"#;
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(yaml))).expect("parse config");
        let err = plain_opts().apply_config(&cfg).expect_err("collision");
        assert!(matches!(err, BuildError::IdColumnReserved(c) if c == "category"));
    }

    #[test]
    fn apply_config_with_none_backend_leaves_storage_unattached() {
        // Default config has storage.backend = none → no storage /
        // disk cache attached, exercising the early-return arm of
        // apply_storage_config.
        let cfg = Config::defaults().expect("defaults");
        let opts = plain_opts().apply_config(&cfg).expect("apply_config");
        assert!(opts.storage.is_none());
        assert!(opts.disk_cache.is_none());
    }

    #[test]
    fn with_disk_cache_attaches_cache() {
        use crate::supertable::reader_cache::{DiskCacheConfig, DiskCacheStore, LruPolicy};

        let dir = env::temp_dir().join(format!("infino-opts-dc-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("mkdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
        let cache = DiskCacheStore::new_unpinned(
            Arc::clone(&storage),
            DiskCacheConfig {
                cache_root: dir.join("cache"),
                mmap_cold_threshold_secs: 0,
                eviction: Box::new(LruPolicy::new()),
                ..Default::default()
            },
        )
        .expect("disk cache");

        let opts = plain_opts().with_storage(storage).with_disk_cache(cache);
        assert!(opts.disk_cache.is_some());
        assert!(opts.storage.is_some());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_config_attaches_local_fs_storage() {
        use figment::{
            Figment,
            providers::{Format, Yaml},
        };

        // `storage.backend = local_fs` with a local_root exercises
        // the LocalFs arm of apply_storage_config; no disk_cache_root
        // means the cache stays unattached.
        let dir = env::temp_dir().join(format!("infino-opts-localfs-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("mkdir");
        let yaml = format!(
            "storage:\n  backend: local_fs\n  local_root: {}\n",
            dir.display()
        );
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(&yaml))).expect("parse config");
        let opts = plain_opts().apply_config(&cfg).expect("apply_config");
        assert!(opts.storage.is_some(), "local_fs backend attaches storage");
        assert!(
            opts.disk_cache.is_none(),
            "no disk_cache_root ⇒ no cache attached"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_config_local_fs_with_disk_cache_root_attaches_both() {
        use figment::{
            Figment,
            providers::{Format, Yaml},
        };

        // local_fs backend + a disk_cache_root drives the disk-cache
        // construction branch of apply_storage_config (cold-fetch
        // mode mapping, DiskCacheConfig build, new_unpinned).
        let dir = env::temp_dir().join(format!("infino-opts-dcroot-{}", Uuid::new_v4()));
        let cache_root = dir.join("cache");
        fs::create_dir_all(&dir).expect("mkdir");
        let yaml = format!(
            "storage:\n  backend: local_fs\n  local_root: {}\n  disk_cache_root: {}\n",
            dir.display(),
            cache_root.display()
        );
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(&yaml))).expect("parse config");
        let opts = plain_opts().apply_config(&cfg).expect("apply_config");
        assert!(opts.storage.is_some());
        assert!(
            opts.disk_cache.is_some(),
            "disk_cache_root ⇒ cache attached"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_config_disk_cache_root_with_range_only_is_rejected() {
        use figment::{
            Figment,
            providers::{Format, Yaml},
        };

        let dir = env::temp_dir().join(format!("infino-opts-ro-cfg-{}", Uuid::new_v4()));
        let cache_root = dir.join("cache");
        fs::create_dir_all(&dir).expect("mkdir");
        let yaml = format!(
            "storage:\n  backend: local_fs\n  local_root: {}\n  disk_cache_root: {}\n  cold_fetch_mode: range_only\n",
            dir.display(),
            cache_root.display()
        );
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(&yaml))).expect("parse config");
        let err = plain_opts()
            .apply_config(&cfg)
            .expect_err("range_only + disk_cache_root must be rejected");
        assert!(
            matches!(err, BuildError::Store(_)),
            "expected BuildError::Store, got: {err:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_config_local_fs_without_root_is_rejected() {
        use figment::{
            Figment,
            providers::{Format, Yaml},
        };

        // local_fs backend but no local_root → the typed Store error
        // arm of apply_storage_config.
        let yaml = "storage:\n  backend: local_fs\n";
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(yaml))).expect("parse config");
        let err = plain_opts()
            .apply_config(&cfg)
            .expect_err("missing local_root");
        assert!(matches!(err, BuildError::Store(_)), "{err:?}");
    }

    #[test]
    fn apply_config_attaches_s3_storage_from_bucket() {
        use figment::{
            Figment,
            providers::{Format, Yaml},
        };

        // `storage.backend = s3` with a bucket drives the S3 arm of
        // apply_storage_config. `S3StorageProvider::new` only builds a
        // client object from the AWS credential chain — it makes no
        // network call — so the arm is exercised offline.
        let yaml = "storage:\n  backend: s3\n  bucket: example-bucket\n  prefix: tbl/example\n";
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(yaml))).expect("parse config");
        let opts = plain_opts().apply_config(&cfg).expect("apply_config");
        assert!(opts.storage.is_some(), "s3 backend attaches storage");
        assert!(
            opts.disk_cache.is_none(),
            "no disk_cache_root ⇒ no cache attached"
        );
    }

    #[test]
    fn apply_config_s3_without_bucket_is_rejected() {
        use figment::{
            Figment,
            providers::{Format, Yaml},
        };

        // s3 backend but no bucket → the typed Store error arm of
        // apply_storage_config.
        let yaml = "storage:\n  backend: s3\n";
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(yaml))).expect("parse config");
        let err = plain_opts().apply_config(&cfg).expect_err("missing bucket");
        assert!(matches!(err, BuildError::Store(_)), "{err:?}");
    }

    #[test]
    fn apply_config_azure_without_bucket_is_rejected() {
        use figment::{
            Figment,
            providers::{Format, Yaml},
        };

        // azure backend but no container → the typed Store error arm
        // of apply_storage_config.
        let yaml = "storage:\n  backend: azure\n";
        let cfg =
            Config::from_figment(Figment::new().merge(Yaml::string(yaml))).expect("parse config");
        let err = plain_opts()
            .apply_config(&cfg)
            .expect_err("missing container");
        assert!(matches!(err, BuildError::Store(_)), "{err:?}");
    }
}
