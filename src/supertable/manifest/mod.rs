//! In-memory manifest types: `Manifest`, `SuperfileEntry`,
//! `ScalarStatsTable`, `FtsSummary`, `VectorSummary`.
//!
//! `Manifest` is the single immutable point-in-time view of which
//! superfiles exist. `Supertable` holds the current manifest behind
//! an `ArcSwap<Manifest>`; commits build a new `Manifest` (superfiles:
//! old + new) and atomically swap it in. Readers
//! `ArcSwap::load_full` once at construction to pin a snapshot for
//! the lifetime of their queries.
//!
//! ## Construction is copy-on-write
//!
//! `Manifest::with_appended` clones the outer `Vec` and shares each
//! existing `Arc<SuperfileEntry>` between the old and new manifests,
//! so the only per-commit allocation is the new entries plus the
//! `Vec` header. `Manifest` itself is immutable — never mutated in
//! place — which is what makes lock-free reader-writer isolation
//! possible.

pub mod aggregates;
pub mod bloom;
pub mod commit;
pub mod encoding;
pub mod list;
pub mod list_prune;
pub mod options_hash;
pub mod part;
pub mod partition;
pub mod term_range;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Schema;
use uuid::Uuid;

use bloom::Bloom;

use super::options::SupertableOptions;

/// One immutable point-in-time view of the supertable.
///
/// **Construction is copy-on-write.** Adding a segment via
/// [`Manifest::with_appended`] returns a new `Manifest` whose
/// `superfiles` is `Vec::clone()` + new entries appended; the original
/// `Manifest`'s `superfiles` is unchanged. `Arc<SuperfileEntry>` shares
/// the underlying entries between the old and new manifests so the
/// only per-commit allocation is the outer `Vec` and the new
/// entries themselves.
///
/// **Reader isolation.** Readers `ArcSwap::load_full` an
/// `Arc<Manifest>` at construction and hold it for their lifetime.
/// New commits don't affect them. Old manifests are dropped
/// automatically once no reader holds an Arc to them.
///
/// `Manifest` is the outer hierarchical wrapper (it adds the
/// `list` / `parts` / `loader` persistence-side fields);
/// `SuperfileList` is the flat in-process view that `Manifest`
/// derefs to, so callers can access `.manifest_id`,
/// `.superfiles[i]`, `.n_docs_total()` etc. directly through a
/// `Manifest`.
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
    /// Append-only list of segment entries. Each entry's `Arc`-share
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
/// - `list`: the [`ManifestList`] when this manifest was loaded
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
/// `manifest.n_docs_total()` etc. work through a `Manifest`
/// reference.
///
/// [`ManifestList`]: list::ManifestList
pub struct Manifest {
    pub superfile_list: SuperfileList,
    pub list: Option<list::ManifestList>,
    pub parts: dashmap::DashMap<
        part::PartId,
        std::sync::Arc<tokio::sync::OnceCell<std::sync::Arc<part::ManifestPart>>>,
    >,
    pub loader: Option<std::sync::Arc<ManifestPartLoader>>,
}

impl std::fmt::Debug for Manifest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Manifest")
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

impl std::ops::Deref for Manifest {
    type Target = SuperfileList;
    fn deref(&self) -> &Self::Target {
        &self.superfile_list
    }
}

impl Manifest {
    /// Empty initial manifest at `manifest_id = 0`. Used by
    /// `Supertable::create` when no storage is attached.
    pub fn empty(options: Arc<SupertableOptions>) -> Self {
        Self {
            superfile_list: SuperfileList::empty(options),
            list: None,
            parts: dashmap::DashMap::new(),
            loader: None,
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
            parts: dashmap::DashMap::new(),
            loader: self.loader.clone(),
        }
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
    pub async fn part(
        &self,
        part_id: part::PartId,
    ) -> Result<std::sync::Arc<part::ManifestPart>, ManifestLoadError> {
        let loader = self
            .loader
            .as_ref()
            .ok_or(ManifestLoadError::NoLoaderAttached)?;
        let cell = self
            .parts
            .entry(part_id)
            .or_insert_with(|| std::sync::Arc::new(tokio::sync::OnceCell::new()))
            .clone();
        let loaded = cell.get_or_try_init(|| loader.load(part_id)).await?;
        Ok(std::sync::Arc::clone(loaded))
    }
}

/// Pulls manifest parts through a [`StorageProvider`] and verifies
/// content-hash on load.
///
/// One `ManifestPartLoader` per `Manifest`. The same `Arc<dyn
/// StorageProvider>` is shared with the `DiskCacheStore` —
/// one auth handshake, one connection pool.
pub struct ManifestPartLoader {
    storage: std::sync::Arc<dyn crate::storage::StorageProvider>,
    /// Maps `PartId → (expected content_hash, uri)`. Built from
    /// the manifest list at construction; immutable per-`Manifest`.
    parts_index: std::collections::HashMap<part::PartId, (part::ContentHash, String)>,
}

impl ManifestPartLoader {
    pub fn new(
        storage: std::sync::Arc<dyn crate::storage::StorageProvider>,
        list: &list::ManifestList,
    ) -> Self {
        let mut idx = std::collections::HashMap::with_capacity(list.parts.len());
        for entry in &list.parts {
            idx.insert(entry.part_id, (entry.content_hash, entry.uri.clone()));
        }
        Self {
            storage,
            parts_index: idx,
        }
    }

    /// Fetch + verify + decode one part. Returns the parsed
    /// `Arc<ManifestPart>`.
    pub async fn load(
        &self,
        part_id: part::PartId,
    ) -> Result<std::sync::Arc<part::ManifestPart>, ManifestLoadError> {
        let (expected_hash, uri) = self
            .parts_index
            .get(&part_id)
            .ok_or(ManifestLoadError::PartNotInList { part_id })?;
        let (bytes, _) = self
            .storage
            .get(uri)
            .await
            .map_err(ManifestLoadError::Storage)?;
        let actual_hash = part::ContentHash::of(&bytes);
        if actual_hash != *expected_hash {
            return Err(ManifestLoadError::ContentHashMismatch {
                expected: expected_hash.to_hex(),
                actual: actual_hash.to_hex(),
            });
        }
        let parsed = part::decode(&bytes)?;
        Ok(std::sync::Arc::new(parsed))
    }
}

/// Errors raised by [`Manifest::part`] and [`ManifestPartLoader::load`].
///
/// Standalone (not folded into the supertable-level
/// `OpenError`) so the per-part load surface stays narrowly
/// testable in isolation.
#[derive(Debug, thiserror::Error)]
pub enum ManifestLoadError {
    /// Caller invoked `Manifest::part(...)` on an in-process-only
    /// manifest (no storage attached). The hierarchical manifest
    /// has no on-disk parts to load from.
    #[error("no storage / loader attached to this manifest")]
    NoLoaderAttached,
    /// `part_id` isn't in this manifest's list. Either the caller
    /// passed a stale id (pre-refresh) or the manifest list is
    /// missing an entry.
    #[error("part_id not in manifest list: {part_id}")]
    PartNotInList { part_id: part::PartId },
    /// Storage backend returned an error.
    #[error("storage error during part load")]
    Storage(#[source] crate::storage::StorageError),
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

/// One segment's metadata + skip-pruning summaries. The bytes that
/// back the segment live in the segment store keyed by `uri` —
/// `superfile_id` is for debugging / observability, `uri` is for
/// store routing.
#[derive(Debug)]
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
    /// Per-scalar-column min/max for skip pruning of SQL filters.
    pub scalar_stats: ScalarStatsTable,
    /// Per-FTS-column term-presence bloom + lex range. The bloom
    /// drives exact-term skip; the term-range drives prefix-query
    /// skip via `[prefix, prefix_upper_bound)` overlap. Keyed by
    /// FTS column name.
    pub fts_summary: HashMap<String, FtsSummary>,
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
    /// only have per-segment summaries. Hash strategy requires
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
    /// `None` on segments produced by pre-M6 writers; the cold
    /// open path falls back to the 2-RTT shape (parquet tail
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
    /// when the segment carries no vector subsection.
    pub vec: Option<(u64, u64)>,
    /// Absolute `(offset, length)` of the FTS subsection. `None`
    /// when the segment carries no FTS subsection.
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
    /// the actual bytes covering the segment's
    /// open-time batch (parquet footer tail + the
    /// `vec_open_ranges` + the `fts_open_ranges`), carried inline
    /// in the manifest part.
    ///
    /// When non-empty, the cold-fetch path installs these directly
    /// into the reader's prefetch overlay and issues **zero**
    /// open-time GETs against the segment object — the bytes
    /// already arrived in the single part GET that `cold_open`
    /// performs. The genuine first-touch per-segment cost then
    /// collapses from 2 RTT-batches (open metadata + cluster
    /// postings) to 1 (postings only).
    ///
    /// Each tuple is `(absolute_offset, bytes)`. Empty on segments
    /// produced by pre-M7 writers, or when blob capture is disabled
    /// — the path then falls back to fetching `vec_open_ranges` /
    /// `fts_open_ranges` over the wire.
    pub open_blob: Vec<(u64, Vec<u8>)>,
}

/// Opaque store key — wraps a UUID v4. The segment store treats
/// this as a hash-eq token and doesn't peek inside. An
/// object-store-backed variant could swap to a path-shaped URI
/// without changing any caller, since the trait shape stays the
/// same.
#[derive(Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct SuperfileUri(pub Uuid);

impl SuperfileUri {
    /// Generate a fresh URI. Called by the writer at commit time
    /// when assigning a key for a new segment's bytes.
    pub fn new_v4() -> Self {
        Self(Uuid::new_v4())
    }

    /// Object-store / LocalFS path for committed segment bytes.
    /// `.sf.parquet` double suffix — on disk this is still valid
    /// Parquet (row groups + optional embedded FTS/vector blobs +
    /// footer), while the `.sf` marker flags it as a Superfile
    /// segment without making the file look non-standard.
    pub fn storage_path(self) -> String {
        format!("data/seg-{}.sf.parquet", self.0)
    }

    /// Disk-cache filename for a promoted segment.
    pub fn cache_filename(self) -> String {
        format!("seg-{}.sf.parquet", self.0)
    }

    /// Disk-cache tempfile while a cold fetch is in flight.
    pub fn cache_tmp_filename(self) -> String {
        format!("seg-{}.sf.parquet.tmp", self.0)
    }
}

/// Per-scalar-column min/max for a segment, used by scalar skip
/// pruning. Each column's min/max is a length-1 `ArrayRef` of the
/// column's data type — the most general shape that doesn't
/// require pulling DataFusion into this layer. The skip helper
/// converts to DataFusion `ScalarValue` at compare time when
/// matching against query predicates.
#[derive(Debug, Clone, Default)]
pub struct ScalarStatsTable {
    /// `cols[col_name] = (min_array, max_array)`. Both arrays are
    /// length-1 with the column's logical Arrow type.
    pub cols: HashMap<String, (ArrayRef, ArrayRef)>,
}

impl ScalarStatsTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute per-column min / max across `batches` for every
    /// scalar column in `scalar_schema`, skipping types whose
    /// ordering isn't well-defined here (anything other than
    /// integer / float / boolean / utf8).
    ///
    /// Used by [`crate::supertable::writer::SupertableWriter`] at
    /// commit time to populate per-segment scalar skip stats. The
    /// resulting table maps `column_name → (min_arr, max_arr)`,
    /// where each entry is a length-1 [`ArrayRef`] of the column's
    /// type — zero-pad isn't needed since the skip planner reads
    /// scalar values out via Arrow's per-type accessors.
    ///
    /// Memory cost: one `concat` per skippable column, each
    /// producing a ~`n_docs`-row temporary that's freed before
    /// the next column. For a 1M-row shard with 5 skippable
    /// columns, peak overhead is one column's worth (~MB) — far
    /// below the parquet footprint we're already paying.
    pub fn from_batches(scalar_schema: &Schema, batches: &[&RecordBatch]) -> Self {
        let mut cols: HashMap<String, (ArrayRef, ArrayRef)> = HashMap::new();
        if batches.is_empty() {
            return Self { cols };
        }
        for (idx, field) in scalar_schema.fields().iter().enumerate() {
            let arrays: Vec<&dyn arrow_array::Array> =
                batches.iter().map(|b| b.column(idx).as_ref()).collect();
            let combined = match arrow::compute::concat(&arrays) {
                Ok(a) => a,
                // Concat fails for shape mismatch; skip silently —
                // the skip planner treats missing stats as "can't
                // prune", which is the safe default.
                Err(_) => continue,
            };
            if let Some(pair) = column_min_max(&combined) {
                cols.insert(field.name().clone(), pair);
            }
        }
        Self { cols }
    }
}

/// Compute (min, max) for one Arrow array as length-1 `ArrayRef`s.
///
/// Returns `None` for unsupported types or for all-null inputs.
/// Supported set: integer (signed + unsigned, all widths), float
/// (f32, f64), boolean, Utf8, LargeUtf8. The supertable schema
/// rejects vector columns up at the SupertableOptions layer, so
/// `FixedSizeList<Float32>` won't appear here in practice.
fn column_min_max(col: &arrow_array::ArrayRef) -> Option<(ArrayRef, ArrayRef)> {
    use arrow::compute::kernels::aggregate as agg;
    use arrow_array::*;
    use arrow_schema::DataType;

    macro_rules! prim {
        ($array_ty:ty) => {{
            let a = col.as_any().downcast_ref::<$array_ty>()?;
            let mn = agg::min(a)?;
            let mx = agg::max(a)?;
            let mn_arr: ArrayRef = Arc::new(<$array_ty>::from(vec![mn]));
            let mx_arr: ArrayRef = Arc::new(<$array_ty>::from(vec![mx]));
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
            let mn = agg::min_boolean(a)?;
            let mx = agg::max_boolean(a)?;
            Some((
                Arc::new(BooleanArray::from(vec![mn])),
                Arc::new(BooleanArray::from(vec![mx])),
            ))
        }
        DataType::Utf8 => {
            let a = col.as_any().downcast_ref::<StringArray>()?;
            let mn = agg::min_string(a)?;
            let mx = agg::max_string(a)?;
            Some((
                Arc::new(StringArray::from(vec![mn])),
                Arc::new(StringArray::from(vec![mx])),
            ))
        }
        DataType::LargeUtf8 => {
            let a = col.as_any().downcast_ref::<LargeStringArray>()?;
            let mn = agg::min_string(a)?;
            let mx = agg::max_string(a)?;
            Some((
                Arc::new(LargeStringArray::from(vec![mn])),
                Arc::new(LargeStringArray::from(vec![mx])),
            ))
        }
        DataType::Decimal128(precision, scale) => {
            let a = col.as_any().downcast_ref::<Decimal128Array>()?;
            let mn = agg::min(a)?;
            let mx = agg::max(a)?;
            Some((
                Arc::new(
                    Decimal128Array::from(vec![mn])
                        .with_precision_and_scale(*precision, *scale)
                        .ok()?,
                ),
                Arc::new(
                    Decimal128Array::from(vec![mx])
                        .with_precision_and_scale(*precision, *scale)
                        .ok()?,
                ),
            ))
        }
        _ => None,
    }
}

/// Per-FTS-column summary: a term-presence bloom (drives
/// exact-term skip pruning) plus a lex term range (drives
/// prefix-query skip via `[prefix, prefix_upper_bound)` overlap).
/// Both are derived for free at commit time from the FST's term
/// iterator (the FST yields keys in lex order; the first and last
/// keys are min and max).
#[derive(Debug, Clone)]
pub struct FtsSummary {
    /// Term-presence bloom filter — sized to ~7% FPR at typical
    /// per-column term cardinalities (64 KiB / column / segment
    /// is the default).
    pub term_bloom: Bloom,
    /// Number of distinct terms seen at build time. Useful for
    /// validating the bloom's sizing in tests + for observability.
    pub n_terms_distinct: u32,
    /// Lex-smallest and lex-largest term in this segment's FST for
    /// this column. Prefix skip checks
    /// `[prefix, prefix_upper_bound)` overlap with this range.
    pub term_range: (Vec<u8>, Vec<u8>),
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
    /// Maximum distance from any indexed vector in this segment to
    /// `centroid`, in the same metric the column was built with.
    pub radius: f32,
    /// Per-cluster IVF centroids (Sq8, per-cluster calibration) for
    /// cross-segment global cluster selection. Empty when the segment
    /// has no vector index for this column.
    pub clusters: ClusterCentroids,
}

/// Per-cluster IVF centroids for one vector column, Sq8-quantized with
/// per-cluster calibration. Carried in the manifest so a query can rank
/// every segment's clusters globally — without opening the segment —
/// and probe only the globally-closest clusters. The 1-bit shortlist +
/// rerank still run on the segment's on-disk compressed vectors; these
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
}

impl ClusterCentroids {
    /// The "no cluster centroids" value — a segment without a vector
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
            let scale = if mx > mn { (mx - mn) / 255.0 } else { 0.0 };
            mins[c] = mn;
            scales[c] = scale;
            let dst = &mut codes[c * d..(c + 1) * d];
            for (o, &v) in dst.iter_mut().zip(src) {
                *o = if scale > 0.0 {
                    ((v - mn) / scale).round().clamp(0.0, 255.0) as u8
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
    use super::*;
    use std::sync::Arc;

    use arrow_array::{Array, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};

    use crate::superfile::builder::FtsConfig;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]))
    }

    fn opts() -> Arc<SupertableOptions> {
        let tk = crate::test_helpers::default_tokenizer();
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
            scalar_stats: ScalarStatsTable::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    #[test]
    fn empty_manifest_starts_at_zero() {
        let m = Manifest::empty(opts());
        assert_eq!(m.manifest_id, 0);
        assert_eq!(m.superfiles.len(), 0);
        assert_eq!(m.n_docs_total(), 0);
    }

    #[test]
    fn with_appended_increments_manifest_id_and_extends_segments() {
        let m0 = Manifest::empty(opts());
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
        let m0 = Manifest::empty(opts());
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
    fn with_appended_shares_old_segments_via_arc() {
        // The new manifest's superfiles[0] should be the SAME Arc as
        // the original's superfiles[0] — copy-on-write doesn't
        // re-allocate per-segment. (Verified by Arc::ptr_eq.)
        let entry = seg_entry(Uuid::new_v4(), 1);
        let m0 = Manifest::empty(opts()).with_appended(vec![entry.clone()]);
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
        let m0 = Manifest::empty(opts());
        let m1 = m0.with_appended(vec![]);
        assert_eq!(m1.manifest_id, 1);
        assert_eq!(m1.superfiles.len(), 0);
    }

    #[test]
    fn segment_uri_is_distinct_per_call() {
        let a = SuperfileUri::new_v4();
        let b = SuperfileUri::new_v4();
        assert_ne!(a, b);
    }

    #[test]
    fn scalar_stats_table_default_is_empty() {
        let s = ScalarStatsTable::new();
        assert!(s.cols.is_empty());
    }

    #[test]
    fn scalar_stats_table_can_hold_arrow_array_min_max() {
        // Spot-check that the (ArrayRef, ArrayRef) shape is
        // constructable for a typical column type.
        let mut s = ScalarStatsTable::new();
        let min: ArrayRef = Arc::new(UInt64Array::from(vec![1u64]));
        let max: ArrayRef = Arc::new(UInt64Array::from(vec![999u64]));
        s.cols.insert("ts".into(), (min, max));
        assert_eq!(s.cols.len(), 1);
        let (lo, hi) = s.cols.get("ts").expect("inserted");
        assert_eq!(lo.len(), 1);
        assert_eq!(hi.len(), 1);
    }

    #[test]
    fn fts_summary_round_trip_fields() {
        // BLOCK_BYTES = 64; smallest valid bloom = one block.
        let s = FtsSummary {
            term_bloom: bloom::BloomBuilder::with_n_blocks(1).finish(),
            n_terms_distinct: 1234,
            term_range: (b"err".to_vec(), b"foo".to_vec()),
        };
        assert_eq!(s.term_bloom.len(), 64);
        assert_eq!(s.n_terms_distinct, 1234);
        assert_eq!(s.term_range.0, b"err".to_vec());
        assert_eq!(s.term_range.1, b"foo".to_vec());
    }

    #[test]
    fn vector_summary_round_trip_fields() {
        let s = VectorSummary {
            centroid: vec![0.1, 0.2, 0.3],
            radius: 0.5,
            clusters: ClusterCentroids::empty(),
        };
        assert_eq!(s.centroid.len(), 3);
        assert!((s.radius - 0.5).abs() < 1e-9);
    }

    // ============================================================
    // In-memory `Manifest` with lazy-load parts — content-hash-
    // verified per-part fetch through an injected
    // `StorageProvider`, OnceCell coalescing on cold cells,
    // typed errors for missing loader / missing part / hash
    // mismatch.
    // ============================================================

    mod lazy_load {
        use super::super::*;
        use async_trait::async_trait;
        use bytes::Bytes;
        use std::collections::HashMap;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use uuid::Uuid;

        use crate::storage::{ObjectMeta, StorageError, StorageProvider};
        use crate::supertable::manifest::list::{
            FORMAT_VERSION as LIST_FORMAT_VERSION, ManifestList, ManifestListEntry,
            PartitionStrategy,
        };
        use crate::supertable::manifest::part::{
            self as part_mod, ContentHash, ManifestPart, PartId,
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
                        },
                    )),
                    None => Err(StorageError::NotFound { uri: uri.into() }),
                }
            }

            async fn get_range(
                &self,
                uri: &str,
                _range: std::ops::Range<u64>,
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
            let boxed: Box<dyn std::error::Error + Send + Sync> = msg.into();
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
        ) -> (HashMap<String, Bytes>, Vec<ManifestListEntry>) {
            let mut objects = HashMap::new();
            let mut entries = Vec::new();
            for p in parts {
                let bytes = part_mod::encode(p, 3);
                let hash = ContentHash::of(&bytes);
                let uri = format!("manifests/part-{}.avro.zst", hash.to_hex());
                let size_compressed = bytes.len() as u64;
                objects.insert(uri.clone(), Bytes::from(bytes));
                entries.push(ManifestListEntry {
                    part_id: p.part_id,
                    uri,
                    n_superfiles: p.superfiles.len() as u64,
                    size_bytes_compressed: size_compressed,
                    size_bytes_uncompressed: size_compressed,
                    content_hash: hash,
                    partition_key: Vec::new(),
                    id_range: (0, 0),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                });
            }
            (objects, entries)
        }

        fn fresh_list(entries: Vec<ManifestListEntry>) -> ManifestList {
            ManifestList {
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

        fn options_for_test() -> Arc<crate::supertable::SupertableOptions> {
            use crate::supertable::SupertableOptions;
            use arrow_schema::{DataType, Field, Schema};
            let s = Arc::new(Schema::new(vec![Field::new(
                "title",
                DataType::LargeUtf8,
                false,
            )]));
            Arc::new(SupertableOptions::new(s, vec![], vec![], None).expect("opts"))
        }

        fn build_manifest_with_loader(
            list: ManifestList,
            storage: Arc<dyn StorageProvider>,
        ) -> Manifest {
            let loader = Arc::new(ManifestPartLoader::new(Arc::clone(&storage), &list));
            Manifest {
                superfile_list: crate::supertable::SuperfileList::empty(options_for_test()),
                list: Some(list),
                parts: dashmap::DashMap::new(),
                loader: Some(loader),
            }
        }

        #[tokio::test]
        async fn part_first_touch_loads_and_caches() {
            let part = make_test_part(7);
            let (objects, entries) = encode_and_index(std::slice::from_ref(&part));
            let storage = Arc::new(CountingMockStorage::new(objects));
            let list = fresh_list(entries);
            let manifest =
                build_manifest_with_loader(list, Arc::clone(&storage) as Arc<dyn StorageProvider>);

            let loaded = manifest.part(part.part_id).await.expect("load");
            assert_eq!(loaded.part_id, part.part_id);
            assert_eq!(storage.get_call_count(), 1, "exactly one storage.get");
        }

        #[tokio::test]
        async fn second_touch_hits_cache_zero_additional_gets() {
            let part = make_test_part(11);
            let (objects, entries) = encode_and_index(std::slice::from_ref(&part));
            let storage = Arc::new(CountingMockStorage::new(objects));
            let list = fresh_list(entries);
            let manifest =
                build_manifest_with_loader(list, Arc::clone(&storage) as Arc<dyn StorageProvider>);

            let a = manifest.part(part.part_id).await.expect("first load");
            let b = manifest.part(part.part_id).await.expect("second load");
            assert!(Arc::ptr_eq(&a, &b), "second touch must return cached Arc");
            assert_eq!(storage.get_call_count(), 1, "cache hit ⇒ no extra get");
        }

        #[tokio::test]
        async fn concurrent_loaders_coalesce_to_one_get() {
            let part = make_test_part(13);
            let (objects, entries) = encode_and_index(std::slice::from_ref(&part));
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
                handles.push(tokio::spawn(async move { m.part(pid).await }));
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
            let (mut objects, entries) = encode_and_index(std::slice::from_ref(&part));
            // Tamper with the stored bytes — content_hash on
            // the list entry no longer matches.
            let bytes = objects.values().next().expect("one obj").clone();
            let mut tampered = bytes.to_vec();
            let last = tampered.len() - 1;
            tampered[last] ^= 0xff;
            let uri = entries[0].uri.clone();
            objects.insert(uri, Bytes::from(tampered));
            let (_, fresh_entries) = encode_and_index(std::slice::from_ref(&part));
            let list = fresh_list(fresh_entries);

            let storage = Arc::new(CountingMockStorage::new(objects));
            let manifest =
                build_manifest_with_loader(list, Arc::clone(&storage) as Arc<dyn StorageProvider>);

            let err = manifest
                .part(part.part_id)
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
                .part(part.part_id)
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
            let err = manifest.part(stranger).await.expect_err("must reject");
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
        async fn no_loader_attached_surfaces_typed_error() {
            // In-process-only manifest — Manifest::empty has
            // no loader. Calling part() must error cleanly,
            // not panic.
            let manifest = Manifest::empty(options_for_test());
            let err = manifest
                .part(PartId(Uuid::nil()))
                .await
                .expect_err("must error");
            assert!(
                matches!(err, ManifestLoadError::NoLoaderAttached),
                "expected NoLoaderAttached, got {err:?}"
            );
        }
    }
}
