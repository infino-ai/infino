//! Vector blob reader. Multi-column kNN search via IVF + 1-bit RaBitQ
//! shortlist + full-precision rerank.
//!
//! Opens the unified-blob byte layout produced by
//! [`super::builder::VectorBuilder::finish`] and exposes per-column
//! kNN search.
//!
//! Self-contained: owns its `Bytes`. Per-column metadata is parsed
//! eagerly at `open()`; per-query work happens on demand.

use crate::superfile::format::checksum::crc32c;
use crate::superfile::format::{self};
use crate::superfile::lazy_source::{LazyByteSource, LazyByteSourceError};
use crate::superfile::vector::distance::{Metric, distance_bytes};
use crate::superfile::vector::quant::BitQuantizer;
use crate::superfile::vector::rotation::RandomRotation;
use crate::superfile::{ReadError, error::VectorError};
use bytes::Bytes;
use rayon::prelude::*;
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

const OUTER_HEADER_SIZE: usize = 32;
const DIR_ENTRY_SIZE: usize = 64;
const SUB_HEADER_SIZE: usize = 56;

/// JSON-deserialized form of one entry in `inf.vec.columns`. The KV
/// value is a JSON array of these in declaration order.
#[derive(Debug, Clone, Deserialize)]
pub struct VectorColumnConfig {
    pub name: String,
    pub dim: usize,
    pub n_cent: usize,
    pub rot_seed: u64,
    /// `"l2sq"`, `"cosine"`, or `"negdot"`.
    pub metric: String,
}

/// Per-column reader state; cached at open time.
#[derive(Debug)]
pub struct ColumnReader {
    pub name: String,
    pub dim: usize,
    pub n_cent: u32,
    pub n_docs: u32,
    pub metric: Metric,
    pub rot_seed: u64,
    /// Byte range of this column's subsection within the outer blob.
    subsection_range: Range<usize>,
    /// Offsets relative to the subsection start.
    summary_off: usize,
    summary_radius: f32,
    centroids_off: usize,
    cluster_idx_off: usize,
    codes_off: usize,
    full_off: usize,
    doc_ids_off: usize,
    /// `local_doc_id → cluster-position`. Plan 011 M2: built lazily
    /// on first rerank (or eagerly at open when
    /// [`OpenOptions::prefetch_eager`] is on). ~4 MB at 1 M docs
    /// per column when populated. Concurrent first-rerank queries
    /// may each build the table; one wins the `set` race, the
    /// other drops their copy — no `Mutex` on the hot path.
    doc_to_pos: OnceLock<Vec<u32>>,
    quant: BitQuantizer,
    /// Cached random rotation built once at open from `(dim, rot_seed)`.
    /// Construction is `O(dim³)` for Gram-Schmidt — at dim=384 that's
    /// ~7.9 ms, dominant over every other per-query stage if rebuilt
    /// per `search()`. Build once, reuse forever.
    rot: RandomRotation,
}

/// Per-open knobs for [`VectorReader::open_with`]. `Default` is the
/// safe + lazy choice (CRC verification on, no eager prefetch). The
/// argumentless [`VectorReader::open`] takes the default.
///
/// Plan 011 consolidates open-time knobs here. Today: `verify_crc`
/// (CRC pre-pass) and `prefetch_eager` (eager `doc_to_pos` build).
/// Plan 013 may add object-storage-native knobs (e.g. `range_fetch_
/// concurrency`) under the same struct.
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Verify the per-subsection CRC during open. Each subsection is
    /// scanned in full (≈1.5 GiB at 1M × 384, single column), so this
    /// dominates cold-open time when on. Defaults to `true`; the
    /// argumentless [`VectorReader::open`] uses this default.
    /// Flip to `false` when storage is already trusted (content-
    /// addressed object store, checksummed filesystem) to skip
    /// the scan.
    pub verify_crc: bool,
    /// If `true`, build the per-column `doc_to_pos` lookup table at
    /// open time by scanning each column's `doc_ids` region (today's
    /// pre-011 behaviour). Costs ~2-3 ms at 1M × 384 + the resident
    /// `n_docs × 4` bytes per column (4 MB at 1M, 40 MB at 10M).
    ///
    /// If `false` (default), `doc_to_pos` is left empty at open and
    /// built lazily on the first `search()` that reaches the rerank
    /// stage on that column — via [`std::sync::OnceLock::set`] under
    /// concurrent searches, so two simultaneous first-rerank queries
    /// may each build the table; one wins, the other drops their
    /// copy. The build itself uses [`Source::try_get_range_sync`],
    /// so on `Source::InMemory` it's a zero-copy walk over the
    /// already-resident bytes.
    ///
    /// Bench harnesses + tests that want today's "first-search is
    /// hot" semantics flip this to `true`. The supertable reader
    /// path leaves it `false` (the lazy default) — first query pays
    /// the build cost, every subsequent one is unchanged.
    pub prefetch_eager: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            verify_crc: true,
            prefetch_eager: false,
        }
    }
}

/// Backing for a [`VectorReader`]. Plan 011 M1.
///
/// Two variants today, plumbed through every byte-fetch point:
///
/// - `InMemory(Bytes)`: the legacy path — caller materialised
///   the full subsection before opening. Every byte fetch is a
///   zero-copy `Bytes::slice` against the buffer. The sync
///   `search()` path always works.
/// - `Lazy(Arc<dyn LazyByteSource>)`: a range-fetching source
///   (mmap, S3 range GET, broadcast subscription). M1 wires
///   the enum and the access pattern; M2 lands `open_lazy` and
///   the lazy-friendly `open_with` shape. Sync `search()`
///   returns [`VectorError::NeedsAsyncEntry`] on this variant
///   (M3) unless the caller pre-populated the relevant ranges.
///
/// `Source: Clone` so `Arc`-shared instances can be handed to
/// multiple readers / supertable segments without forking the
/// underlying state. Lazy variant clones the `Arc`; in-memory
/// clones the `Bytes` (refcount bump).
#[derive(Clone)]
pub enum Source {
    InMemory(Bytes),
    Lazy(Arc<dyn LazyByteSource>),
}

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InMemory(b) => f.debug_tuple("InMemory").field(&b.len()).finish(),
            Self::Lazy(_) => f.debug_struct("Lazy").finish_non_exhaustive(),
        }
    }
}

impl Source {
    /// Total backing size in bytes — matches what a single
    /// `get_range(0..len())` would cover.
    pub fn len(&self) -> usize {
        match self {
            Self::InMemory(b) => b.len(),
            Self::Lazy(s) => s.size() as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sync best-effort fetch. Always succeeds on
    /// `Source::InMemory` (zero-copy `Bytes::slice`); on
    /// `Source::Lazy` returns `Some` only if the range is
    /// already resident in the source's in-process cache.
    ///
    /// Returns `None` for out-of-bounds ranges so callers can
    /// distinguish "not available sync" from a hard error;
    /// callers that need a typed error should fall through to
    /// [`Self::get_range`].
    pub fn try_get_range_sync(&self, range: Range<usize>) -> Option<Bytes> {
        let start = range.start as u64;
        let len = range.len() as u64;
        match self {
            Self::InMemory(b) => {
                if range.end > b.len() {
                    return None;
                }
                Some(b.slice(range))
            }
            Self::Lazy(s) => s.try_get_range_sync(start, len),
        }
    }

    /// Async fetch — always succeeds for in-bounds ranges.
    /// On `Source::InMemory` resolves immediately (zero-copy
    /// `Bytes::slice` wrapped in an already-ready future).
    pub async fn get_range(&self, range: Range<usize>) -> Result<Bytes, LazyByteSourceError> {
        match self {
            Self::InMemory(b) => {
                if range.end > b.len() {
                    return Err(LazyByteSourceError::OutOfBounds {
                        start: range.start as u64,
                        len: range.len() as u64,
                        size: b.len() as u64,
                    });
                }
                Ok(b.slice(range))
            }
            Self::Lazy(s) => s.range(range.start as u64, range.len() as u64).await,
        }
    }
}

/// Multi-column vector blob reader. `Send + Sync`; concurrent
/// searches share the underlying [`Source`] (refcount-shared via
/// `Bytes` / `Arc<dyn LazyByteSource>`).
#[derive(Debug)]
pub struct VectorReader {
    source: Source,
    n_docs: u64,
    columns: Vec<ColumnReader>,
    column_id_by_name: HashMap<String, u32>,
}

impl VectorReader {
    /// Open the reader. `columns_json` is the value of the
    /// `inf.vec.columns` Parquet KV key (a JSON array of
    /// [`VectorColumnConfig`]).
    /// Open the reader with default options (CRC verification on).
    pub fn open(blob: Bytes, columns_json: &str) -> Result<Self, VectorError> {
        Self::open_with(blob, columns_json, OpenOptions::default())
    }

    /// Open with explicit options. The fast path is
    /// `OpenOptions { verify_crc: false }` which skips both the
    /// outer (whole-blob) CRC and the per-subsection CRC scans —
    /// at 1M × 384 cold open drops from ~132 ms to ~2 ms. Use this
    /// when the underlying storage is trusted (e.g. local disk after
    /// a successful file integrity check) or when CRC verification
    /// is performed elsewhere in the stack.
    pub fn open_with(
        blob: Bytes,
        columns_json: &str,
        opts: OpenOptions,
    ) -> Result<Self, VectorError> {
        // M1: every byte fetch routes through `Source::try_get_range_sync`
        // so a future lazy variant can intercept the same call sites
        // without a second rewrite. `InMemory` returns zero-copy
        // `Bytes::slice` views; refcount bumps only.
        let source = Source::InMemory(blob);

        if source.len() < OUTER_HEADER_SIZE + 4 {
            return Err(VectorError::Read(ReadError::MissingKv(
                "vector blob header",
            )));
        }

        // Pull the fixed-size outer header in one fetch — five small
        // reads collapse into one `Bytes::slice`.
        let header = fetch_sync(&source, 0..OUTER_HEADER_SIZE, "outer header")?;
        if &header[0..8] != format::vec::OUTER_MAGIC {
            return Err(VectorError::Read(ReadError::BadMagic {
                section: "vector",
                expected: format::vec::OUTER_MAGIC,
                actual: header[0..8].to_vec(),
            }));
        }

        let version = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
        if version != format::vec::VERSION {
            return Err(VectorError::Read(ReadError::UnsupportedVersion(format!(
                "vector blob version {version}"
            ))));
        }

        let n_columns =
            u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;
        let n_docs = read_u64_le(&header[16..24]);
        let dir_offset = read_u64_le(&header[24..32]) as usize;

        // Verify directory CRC.
        let dir_size = n_columns * DIR_ENTRY_SIZE;
        if dir_offset + dir_size + 4 > source.len() {
            return Err(VectorError::Read(ReadError::MalformedVersion(
                "vector directory runs past blob".into(),
            )));
        }
        let dir_bytes = fetch_sync(&source, dir_offset..dir_offset + dir_size, "directory")?;
        let dir_crc_bytes = fetch_sync(
            &source,
            dir_offset + dir_size..dir_offset + dir_size + 4,
            "directory crc",
        )?;
        let dir_crc_expected = read_u32_le(&dir_crc_bytes);
        let dir_crc_actual = crc32c(&dir_bytes);
        if dir_crc_expected != dir_crc_actual {
            return Err(VectorError::Read(ReadError::ChecksumMismatch {
                section: "vector/directory",
                column: String::new(),
            }));
        }

        if opts.verify_crc {
            verify_vector_crcs(&source, &dir_bytes, n_columns)?;
        }

        // Parse JSON.
        let cols_json: Vec<VectorColumnConfig> =
            serde_json::from_str(columns_json).map_err(|e| {
                VectorError::Read(ReadError::MalformedVersion(format!(
                    "inf.vec.columns JSON: {e}"
                )))
            })?;
        if cols_json.len() != n_columns {
            return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                "inf.vec.columns has {} entries, header says {n_columns}",
                cols_json.len()
            ))));
        }

        // Parse each directory entry, build ColumnReader.
        let mut columns = Vec::with_capacity(n_columns);
        let mut column_id_by_name = HashMap::with_capacity(n_columns);
        for (i, cfg) in cols_json.iter().enumerate() {
            let entry_off = i * DIR_ENTRY_SIZE;
            let column_id = u32::from_le_bytes([
                dir_bytes[entry_off],
                dir_bytes[entry_off + 1],
                dir_bytes[entry_off + 2],
                dir_bytes[entry_off + 3],
            ]);
            if column_id != i as u32 {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "vector dir entry {i} has column_id {column_id}"
                ))));
            }
            let dim = u32::from_le_bytes([
                dir_bytes[entry_off + 4],
                dir_bytes[entry_off + 5],
                dir_bytes[entry_off + 6],
                dir_bytes[entry_off + 7],
            ]) as usize;
            let n_cent = u32::from_le_bytes([
                dir_bytes[entry_off + 8],
                dir_bytes[entry_off + 9],
                dir_bytes[entry_off + 10],
                dir_bytes[entry_off + 11],
            ]);
            let metric_id = u32::from_le_bytes([
                dir_bytes[entry_off + 12],
                dir_bytes[entry_off + 13],
                dir_bytes[entry_off + 14],
                dir_bytes[entry_off + 15],
            ]);
            let rot_seed = read_u64_le(&dir_bytes[entry_off + 16..entry_off + 24]);
            let subsection_off = read_u64_le(&dir_bytes[entry_off + 24..entry_off + 32]) as usize;
            let subsection_len = read_u64_le(&dir_bytes[entry_off + 32..entry_off + 40]) as usize;
            // bytes [40..48] = summary_offset (absolute), [48..52] = summary_length, then padding
            let _summary_off_abs = read_u64_le(&dir_bytes[entry_off + 40..entry_off + 48]);

            // Validate against JSON.
            if dim != cfg.dim {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' dim mismatch: dir={dim} json={}",
                    cfg.name, cfg.dim
                ))));
            }
            if rot_seed != cfg.rot_seed {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' rot_seed mismatch",
                    cfg.name
                ))));
            }
            let metric = match metric_id {
                0 => Metric::L2Sq,
                1 => Metric::Cosine,
                2 => Metric::NegDot,
                _ => {
                    return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                        "unknown metric_id {metric_id} for column '{}'",
                        cfg.name
                    ))));
                }
            };

            // Validate subsection bounds + magic. Subsection CRCs are
            // verified above in the parallel CRC pre-pass when requested.
            let sub_end = subsection_off + subsection_len;
            if sub_end > source.len() {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} runs past blob"
                ))));
            }
            let sub = fetch_sync(&source, subsection_off..sub_end, "subsection")?;
            if sub.len() < SUB_HEADER_SIZE + 4 {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} too short"
                ))));
            }
            if &sub[0..8] != format::vec::SUB_MAGIC {
                return Err(VectorError::Read(ReadError::BadMagic {
                    section: "vector/subsection",
                    expected: format::vec::SUB_MAGIC,
                    actual: sub[0..8].to_vec(),
                }));
            }
            let sub_crc_pos = sub.len() - 4;

            // Sub-header parse (SUB_HEADER_SIZE = 56 bytes):
            //   [8..12]  version  (cross-checked against outer header)
            //   [12..16] reserved
            //   [16..24] summary_centroid_offset (relative to sub start)
            //   [24..28] summary_radius_x100
            //   [28..32] reserved
            //   [32..40] centroids_offset
            //   [40..48] cluster_idx_offset
            //   [48..52] codes_offset
            //   [52..56] full_offset
            let summary_off = read_u64_le(&sub[16..24]) as usize;
            let summary_radius_x100 = read_u32_le(&sub[24..28]);
            let centroids_off = read_u64_le(&sub[32..40]) as usize;
            let cluster_idx_off = read_u64_le(&sub[40..48]) as usize;
            let codes_off = read_u32_le(&sub[48..52]) as usize;
            let full_off = read_u32_le(&sub[52..56]) as usize;

            let summary_radius = (summary_radius_x100 as f32) / 100.0;

            // Compute n_docs for this column and doc_ids_off.
            // doc_ids start at end of full vectors. Total subsection
            // bytes (excluding CRC) = SUB_HEADER + summary + centroids +
            // cluster_idx + codes + full + doc_ids.
            let quant = BitQuantizer::new(dim);
            let code_bytes = quant.code_bytes();
            // We can derive n_docs from the cluster index: sum of counts
            // across clusters. Or from the layout: doc_ids region size
            // / 4. Let's compute from doc_ids region:
            //   doc_ids_size = sub.len() - 4 - doc_ids_off
            // But we need doc_ids_off first. Use full_off + full_size:
            // that requires n_docs, circular. Instead derive from
            // codes region: codes region size = full_off - codes_off,
            // and codes region size = n_docs * code_bytes.
            let codes_size = full_off - codes_off;
            if code_bytes == 0 || !codes_size.is_multiple_of(code_bytes) {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' codes size {codes_size} not divisible by {code_bytes}",
                    cfg.name
                ))));
            }
            let col_n_docs = (codes_size / code_bytes) as u32;

            let full_size = (col_n_docs as usize) * dim * 4;
            let doc_ids_off = full_off + full_size;

            // Bounds-check the cluster_idx + doc_ids regions without
            // reading them. Plan 011 M2: the per-cluster walk that
            // builds `doc_to_pos` is gated on `opts.prefetch_eager`
            // — without it, open never touches the `doc_ids` region
            // (the table is built lazily on first rerank). The whole-
            // region bounds checks below are pure offset math, so
            // they're cheap and run unconditionally.
            let cluster_idx_size = (n_cent as usize) * 8;
            let cluster_idx_end = cluster_idx_off + cluster_idx_size;
            if cluster_idx_end > sub_crc_pos {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' cluster index runs past subsection",
                    cfg.name
                ))));
            }
            let doc_ids_size = (col_n_docs as usize) * 4;
            if doc_ids_off + doc_ids_size > sub_crc_pos {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' doc_ids region runs past subsection",
                    cfg.name
                ))));
            }

            let doc_to_pos: OnceLock<Vec<u32>> = OnceLock::new();
            if opts.prefetch_eager {
                // Eager path: walk every cluster's doc_ids slice now
                // and seed the OnceLock. Matches today's pre-011
                // behaviour for callers (bench harnesses, eager
                // tests) that want first-search to be hot.
                let table = build_doc_to_pos(
                    &sub,
                    n_cent,
                    cluster_idx_off,
                    doc_ids_off,
                    col_n_docs,
                    sub_crc_pos,
                    &cfg.name,
                )?;
                let _ = doc_to_pos.set(table);
            }

            // Soft cross-check: cfg.metric matches blob's metric.
            let cfg_metric = match cfg.metric.as_str() {
                "l2sq" => Some(Metric::L2Sq),
                "cosine" => Some(Metric::Cosine),
                "negdot" => Some(Metric::NegDot),
                _ => None,
            };
            if let Some(m) = cfg_metric
                && m != metric
            {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' metric mismatch: dir={metric:?} json={}",
                    cfg.name, cfg.metric
                ))));
            }

            columns.push(ColumnReader {
                name: cfg.name.clone(),
                dim,
                n_cent,
                n_docs: col_n_docs,
                metric,
                rot_seed,
                subsection_range: subsection_off..sub_end,
                summary_off,
                summary_radius,
                centroids_off,
                cluster_idx_off,
                codes_off,
                full_off,
                doc_ids_off,
                doc_to_pos,
                quant,
                rot: RandomRotation::new(dim, rot_seed),
            });
            column_id_by_name.insert(cfg.name.clone(), i as u32);
        }

        Ok(VectorReader {
            source,
            n_docs,
            columns,
            column_id_by_name,
        })
    }

    pub fn n_docs(&self) -> u64 {
        self.n_docs
    }

    pub fn vector_columns(&self) -> impl Iterator<Item = &str> {
        self.columns.iter().map(|c| c.name.as_str())
    }

    /// Per-column summary centroid + radius, used by the storage plan
    /// for cross-segment skip pruning.
    pub fn summary(&self, column: &str) -> Option<(Vec<f32>, f32)> {
        let cid = *self.column_id_by_name.get(column)?;
        let col = &self.columns[cid as usize];
        // M1: byte access routed through `Source::try_get_range_sync`
        // — zero-copy on `InMemory`, M2/M3 wires the lazy path.
        let sub = self
            .source
            .try_get_range_sync(col.subsection_range.clone())?;
        let off = col.summary_off;
        let dim = col.dim;
        let centroid: Vec<f32> = (0..dim)
            .map(|i| {
                let s = off + i * 4;
                f32::from_le_bytes([sub[s], sub[s + 1], sub[s + 2], sub[s + 3]])
            })
            .collect();
        Some((centroid, col.summary_radius))
    }

    /// Single-column kNN search. Returns `(local_doc_id, distance)`
    /// sorted ascending by distance (smaller = closer for every metric).
    pub fn search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank_mult: usize,
    ) -> Result<Vec<(u32, f32)>, VectorError> {
        let cid = *self
            .column_id_by_name
            .get(column)
            .ok_or_else(|| VectorError::UnknownColumn(column.to_string()))?;
        let col = &self.columns[cid as usize];

        if query.len() != col.dim {
            return Err(VectorError::DimensionMismatch {
                expected: col.dim,
                got: query.len(),
            });
        }
        if k == 0 || col.n_docs == 0 {
            return Ok(Vec::new());
        }

        // M1: subsection bytes obtained via `Source::try_get_range_sync`
        // — zero-copy on `InMemory` (the only Source path on the
        // current sync `search` contract). M3 lands the async
        // counterpart + a `NeedsAsyncEntry` fast-fail on the
        // lazy / unpopulated path. On out-of-bounds the source
        // returns `None`; surface that as a typed read error
        // rather than panicking on the implicit bounds check.
        let sub = self
            .source
            .try_get_range_sync(col.subsection_range.clone())
            .ok_or_else(|| {
                VectorError::Read(ReadError::MalformedVersion(format!(
                    "vector source missing subsection bytes for column '{}' \
                     (range {:?})",
                    col.name, col.subsection_range
                )))
            })?;

        // 1. Score query vs every centroid (cheap; n_cent is small).
        //
        // Zero-copy `f32x8` over the centroid bytes in `sub` via
        // `distance_bytes` — `bytemuck::try_cast_slice` borrows the
        // 4-aligned region (common case for our layout), falling
        // back to a per-chunk LE decode if alignment is off. At
        // `n_cent = 1024, dim = 384` this is 1024 zero-copy SIMD
        // dot/L2² calls per query; no per-centroid heap allocation.
        let dim = col.dim;
        let dim_bytes = dim * 4;
        let mut centroid_scores: Vec<(usize, f32)> = (0..col.n_cent as usize)
            .map(|c| {
                let start = col.centroids_off + c * dim_bytes;
                let bytes = &sub[start..start + dim_bytes];
                (c, distance_bytes(col.metric, query, bytes))
            })
            .collect();
        centroid_scores.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let nprobe_eff = nprobe.min(col.n_cent as usize).max(1);
        centroid_scores.truncate(nprobe_eff);

        // 2. Rotate query for the 1-bit code estimator. The rotation
        // matrix was built once at open and is cached on `col` —
        // rebuilding it per `search()` costs `O(dim³)` Gram-Schmidt
        // (~7.9 ms at dim=384), which dominates every other per-query
        // stage. The `apply` itself is a cheap `O(dim²)` matvec.
        let mut q_rot = vec![0f32; dim];
        col.rot.apply(query, &mut q_rot);

        // 3. Scan codes within probed clusters → shortlist.
        let cb = col.quant.code_bytes();
        let mut shortlist: Vec<(u32, f32)> = Vec::new();
        for &(c, _) in &centroid_scores {
            let idx_start = col.cluster_idx_off + c * 8;
            let off = u32::from_le_bytes([
                sub[idx_start],
                sub[idx_start + 1],
                sub[idx_start + 2],
                sub[idx_start + 3],
            ]);
            let cnt = u32::from_le_bytes([
                sub[idx_start + 4],
                sub[idx_start + 5],
                sub[idx_start + 6],
                sub[idx_start + 7],
            ]);
            if cnt == 0 {
                continue;
            }
            for i in 0..cnt as usize {
                let code_start = col.codes_off + (off as usize + i) * cb;
                let code = &sub[code_start..code_start + cb];
                let est = col.quant.estimate_dot_rotated(&q_rot, code);
                let did_start = col.doc_ids_off + (off as usize + i) * 4;
                let did = u32::from_le_bytes([
                    sub[did_start],
                    sub[did_start + 1],
                    sub[did_start + 2],
                    sub[did_start + 3],
                ]);
                shortlist.push((did, est));
            }
        }

        // 4. Take top `k * rerank_mult` by descending estimate
        //    (higher est = closer for cosine / negdot; for l2sq it's
        //    reversed but the rerank step uses true distance anyway,
        //    so a slightly looser shortlist is fine).
        let want = (k.saturating_mul(rerank_mult)).min(shortlist.len());
        if want < shortlist.len() {
            shortlist.select_nth_unstable_by(want.saturating_sub(1), |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
            });
            shortlist.truncate(want);
        }

        // 5. Full-precision rerank using the true metric.
        //
        // Score each candidate's full vector directly from its byte
        // slice in the blob — `distance_bytes` zero-copies into
        // `f32x8` when 4-aligned (the common case for our layout)
        // and falls back to a per-chunk LE decode otherwise. Same
        // zero-copy pattern as the centroid probe above; no
        // per-candidate heap allocation.
        //
        // Re-fetch the subsection slice via the Source. On
        // `InMemory` this is a refcount-only `Bytes::slice`;
        // on a future warm `Lazy`, the same range is served
        // from the in-process cache populated by step 3 / 4.
        let sub = self
            .source
            .try_get_range_sync(col.subsection_range.clone())
            .ok_or_else(|| {
                VectorError::Read(ReadError::MalformedVersion(format!(
                    "vector source dropped subsection bytes mid-search for column '{}'",
                    col.name
                )))
            })?;
        // Plan 011 M2: lazy `doc_to_pos` build on first rerank.
        // The first search through this branch on this column
        // walks the cluster index + doc_ids region and seeds the
        // `OnceLock`; every subsequent search hits the populated
        // table via `OnceLock::get`. Two concurrent first-rerank
        // queries may both build the table; one wins the `set`
        // race, the other drops its copy — no `Mutex` on the
        // hot path.
        let doc_to_pos = ensure_doc_to_pos(col, &sub)?;
        let dim_bytes = col.dim * 4;
        let mut reranked: Vec<(u32, f32)> = shortlist
            .iter()
            .map(|&(did, _)| {
                let pos = doc_to_pos[did as usize] as usize;
                let start = col.full_off + pos * dim_bytes;
                let bytes = &sub[start..start + dim_bytes];
                let d = distance_bytes(col.metric, query, bytes);
                (did, d)
            })
            .collect();
        reranked.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        reranked.truncate(k);
        Ok(reranked)
    }
}

/// Return the column's `doc_to_pos` table, building it on first
/// access. The `sub` slice is the column's subsection bytes —
/// caller already fetched it via `Source::try_get_range_sync`.
///
/// Free function (not a method on `VectorReader`) because the
/// returned `&[u32]` borrow is tied to the `col: &ColumnReader`
/// argument, not to `&self`. Keeping the function shape narrow
/// keeps the lifetime trivial.
#[inline]
fn ensure_doc_to_pos<'a>(col: &'a ColumnReader, sub: &[u8]) -> Result<&'a [u32], VectorError> {
    if let Some(t) = col.doc_to_pos.get() {
        return Ok(t.as_slice());
    }
    let sub_crc_pos = sub.len() - 4;
    let table = build_doc_to_pos(
        sub,
        col.n_cent,
        col.cluster_idx_off,
        col.doc_ids_off,
        col.n_docs,
        sub_crc_pos,
        &col.name,
    )?;
    // Race-safe: if another thread `set` first, our table is
    // dropped and we return that thread's view. Either way the
    // post-condition holds.
    let _ = col.doc_to_pos.set(table);
    Ok(col
        .doc_to_pos
        .get()
        .expect("doc_to_pos OnceLock set above")
        .as_slice())
}

#[inline]
fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline]
fn read_u64_le(b: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&b[0..8]);
    u64::from_le_bytes(buf)
}

#[inline]
fn verify_vector_crcs(
    source: &Source,
    dir_bytes: &[u8],
    n_columns: usize,
) -> Result<(), VectorError> {
    // `Bytes` instead of `&'a [u8]` so the par_iter doesn't need a
    // lifetime parameter — each job owns a refcount-shared view into
    // the source, which is itself shared via the outer `Source` for
    // the duration of `open_with`.
    struct CrcJob {
        idx: i32,
        bytes: Bytes,
        expected: u32,
    }

    let mut jobs: Vec<CrcJob> = Vec::with_capacity(n_columns + 1);

    let outer_total = source.len();
    let outer_crc_pos = outer_total - 4;
    let outer_body = fetch_sync(source, 0..outer_crc_pos, "outer body")?;
    let outer_crc_bytes = fetch_sync(source, outer_crc_pos..outer_total, "outer crc")?;
    jobs.push(CrcJob {
        idx: -1,
        bytes: outer_body,
        expected: read_u32_le(&outer_crc_bytes),
    });

    for i in 0..n_columns {
        let entry_off = i * DIR_ENTRY_SIZE;
        let subsection_off = read_u64_le(&dir_bytes[entry_off + 24..entry_off + 32]) as usize;
        let subsection_len = read_u64_le(&dir_bytes[entry_off + 32..entry_off + 40]) as usize;
        let sub_end = subsection_off + subsection_len;
        if sub_end > source.len() {
            return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                "subsection {i} runs past blob"
            ))));
        }
        let sub = fetch_sync(source, subsection_off..sub_end, "subsection")?;
        if sub.len() < SUB_HEADER_SIZE + 4 {
            return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                "subsection {i} too short"
            ))));
        }
        let sub_crc_pos = sub.len() - 4;
        // `Bytes::slice` is zero-copy — no second `try_get_range_sync`
        // round-trip needed since we already hold the subsection.
        let sub_body = sub.slice(0..sub_crc_pos);
        let sub_crc_bytes = sub.slice(sub_crc_pos..sub.len());
        jobs.push(CrcJob {
            idx: i as i32,
            bytes: sub_body,
            expected: read_u32_le(&sub_crc_bytes),
        });
    }

    // The outer-blob scan and per-subsection scans each touch ~1.5 GiB
    // at 1M x 384. They are independent, so run them as parallel jobs
    // and let the checksum module's CLMUL implementation handle each
    // byte stream quickly.
    let mismatch = jobs.par_iter().find_map_any(|job| {
        if crc32c(&job.bytes) != job.expected {
            Some(job.idx)
        } else {
            None
        }
    });
    if let Some(idx) = mismatch {
        if idx < 0 {
            return Err(VectorError::Read(ReadError::ChecksumMismatch {
                section: "vector",
                column: String::new(),
            }));
        }
        let i = idx as usize;
        return Err(VectorError::Read(ReadError::ChecksumMismatch {
            section: "vector/subsection",
            column: format!(" (column index {i})"),
        }));
    }

    Ok(())
}

/// Best-effort sync byte fetch with a typed error. Used throughout
/// `open_with` so every byte access goes through the `Source`
/// abstraction — the lazy variant (M2) will plumb the eager-prefetch
/// path through the same call sites without a second rewrite.
///
/// Failure mode here means the range is out-of-bounds or not
/// present in the sync cache. M1 only opens `Source::InMemory`, where
/// any in-bounds range succeeds zero-copy; this only fires on a
/// malformed blob today.
#[inline]
fn fetch_sync(source: &Source, range: Range<usize>, what: &str) -> Result<Bytes, VectorError> {
    let start = range.start;
    let end = range.end;
    source.try_get_range_sync(range).ok_or_else(|| {
        VectorError::Read(ReadError::MalformedVersion(format!(
            "vector {what} range {start}..{end} past blob"
        )))
    })
}

/// Walk a column's `cluster_idx + doc_ids` regions and produce the
/// `local_doc_id → cluster-position` lookup table that powers the
/// rerank fetch.
///
/// Pulled out of the per-column open loop in plan 011 M2 so it can
/// also run lazily on first rerank. `sub` is the subsection bytes
/// (relative offsets); `sub_crc_pos` is the trailing-CRC boundary
/// inside that slice (`sub.len() - 4`). Per-cluster `doc_ids` slice
/// bounds are checked here — the per-cluster check used to live in
/// the open loop; with the open loop now skipping this walk in the
/// lazy path, the bounds check moves with it.
fn build_doc_to_pos(
    sub: &[u8],
    n_cent: u32,
    cluster_idx_off: usize,
    doc_ids_off: usize,
    n_docs: u32,
    sub_crc_pos: usize,
    column_name: &str,
) -> Result<Vec<u32>, VectorError> {
    let mut doc_to_pos = vec![u32::MAX; n_docs as usize];
    for c in 0..n_cent as usize {
        let idx_start = cluster_idx_off + c * 8;
        let off = u32::from_le_bytes([
            sub[idx_start],
            sub[idx_start + 1],
            sub[idx_start + 2],
            sub[idx_start + 3],
        ]);
        let cnt = u32::from_le_bytes([
            sub[idx_start + 4],
            sub[idx_start + 5],
            sub[idx_start + 6],
            sub[idx_start + 7],
        ]);
        let did_start = doc_ids_off + (off as usize) * 4;
        let did_end = did_start + (cnt as usize) * 4;
        if did_end > sub_crc_pos {
            return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                "column '{column_name}' doc_ids slice {did_start}..{did_end} past subsection"
            ))));
        }
        for i in 0..cnt as usize {
            let s = did_start + i * 4;
            let d = u32::from_le_bytes([sub[s], sub[s + 1], sub[s + 2], sub[s + 3]]);
            if (d as usize) < doc_to_pos.len() {
                doc_to_pos[d as usize] = off + i as u32;
            }
        }
    }
    Ok(doc_to_pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::vector::builder::{VectorBuilder, VectorConfig};

    fn build_blob(n_docs: u32, dim: usize, n_cent: usize, metric: Metric) -> (Bytes, String) {
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            name: "embedding".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric,
        })
        .expect("register column");
        for i in 0..n_docs {
            // Deterministic "random" vector.
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i.wrapping_mul(31) + j as u32) % 100) as f32 * 0.01)
                .collect();
            b.add(0, &v).expect("add to vector builder");
        }
        let bytes = b.finish().expect("finish vector builder");
        let metric_s = match metric {
            Metric::L2Sq => "l2sq",
            Metric::Cosine => "cosine",
            Metric::NegDot => "negdot",
        };
        let json = format!(
            r#"[{{"name":"embedding","dim":{dim},"n_cent":{n_cent},"rot_seed":7,"metric":"{metric_s}"}}]"#
        );
        (Bytes::from(bytes), json)
    }

    #[test]
    fn open_accepts_valid_blob() {
        let (blob, json) = build_blob(64, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open should succeed");
        assert_eq!(r.n_docs(), 64);
        let cols: Vec<&str> = r.vector_columns().collect();
        assert_eq!(cols, vec!["embedding"]);
    }

    #[test]
    fn open_rejects_bad_magic() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let mut bytes = blob.to_vec();
        bytes[0] = b'X';
        let err = VectorReader::open(Bytes::from(bytes), &json).expect_err("expected error");
        assert!(matches!(err, VectorError::Read(ReadError::BadMagic { .. })));
    }

    #[test]
    fn open_rejects_short_blob() {
        let err = VectorReader::open(Bytes::from(vec![0u8; 8]), "[]").expect_err("expected error");
        assert!(matches!(err, VectorError::Read(_)));
    }

    #[test]
    fn open_detects_corruption_via_outer_crc() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let mut bytes = blob.to_vec();
        // Flip a byte in the middle of the directory area.
        let pos = OUTER_HEADER_SIZE + 10;
        bytes[pos] ^= 0xFF;
        let err = VectorReader::open(Bytes::from(bytes), &json).expect_err("expected error");
        assert!(matches!(
            err,
            VectorError::Read(ReadError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn open_with_skip_crc_accepts_corrupted_directory_bytes() {
        // The fast-open path explicitly skips CRC verification — so
        // a flipped byte in the directory area opens cleanly. The
        // caller is responsible for upstream integrity.
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let mut bytes = blob.to_vec();
        let pos = OUTER_HEADER_SIZE + 10;
        bytes[pos] ^= 0xFF;
        let r = VectorReader::open_with(
            Bytes::from(bytes),
            &json,
            OpenOptions {
                verify_crc: false,
                ..OpenOptions::default()
            },
        );
        // Open succeeds; the corruption may surface later as a
        // bad-magic / bounds error or be silently absorbed depending
        // on which byte got flipped. The contract is "we don't
        // verify"; not "we'll always read sensibly."
        let _ = r;
    }

    #[test]
    fn open_with_default_options_matches_open() {
        // Sanity: default opts produce identical results to `open`.
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r1 = VectorReader::open(blob.clone(), &json).expect("open VectorReader");
        let r2 = VectorReader::open_with(blob, &json, OpenOptions::default())
            .expect("open VectorReader");
        assert_eq!(r1.n_docs(), r2.n_docs());
    }

    #[test]
    fn search_self_query_returns_self_as_top1() {
        let dim = 16;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            name: "embedding".into(),
            dim,
            n_cent: 4,
            rot_seed: 7,
            metric: Metric::L2Sq,
        })
        .expect("register column");
        let mut all_vecs = Vec::new();
        for i in 0..64u32 {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i.wrapping_mul(13) + j as u32 * 5) % 100) as f32)
                .collect();
            b.add(0, &v).expect("add to vector builder");
            all_vecs.push(v);
        }
        let bytes = b.finish().expect("finish vector builder");
        let json = r#"[{"name":"embedding","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#;
        let r = VectorReader::open(Bytes::from(bytes), json).expect("open VectorReader");

        // Pick a doc, query with its own vector → top-1 is self with distance 0.
        let target = 17;
        let hits = r
            .search("embedding", &all_vecs[target], 5, 4, 5)
            .expect("FTS search");
        assert!(!hits.is_empty(), "search should return hits");
        assert_eq!(hits[0].0, target as u32, "self should be nearest");
        assert!(
            hits[0].1 < 1e-3,
            "self distance should be ~0, got {}",
            hits[0].1
        );
    }

    #[test]
    fn search_unknown_column_errors() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let err = r
            .search("nonexistent", &[0.0; 16], 5, 4, 5)
            .expect_err("expected error");
        assert!(matches!(err, VectorError::UnknownColumn(_)));
    }

    #[test]
    fn search_dim_mismatch_errors() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let err = r
            .search("embedding", &[0.0; 8], 5, 4, 5)
            .expect_err("expected error");
        assert!(matches!(err, VectorError::DimensionMismatch { .. }));
    }

    #[test]
    fn search_zero_k_returns_empty() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let hits = r
            .search("embedding", &[0.0; 16], 0, 4, 5)
            .expect("FTS search");
        assert!(hits.is_empty());
    }

    #[test]
    fn search_results_sorted_ascending_by_distance() {
        let (blob, json) = build_blob(64, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let q = vec![0.5; 16];
        let hits = r.search("embedding", &q, 10, 4, 5).expect("FTS search");
        for w in hits.windows(2) {
            assert!(w[0].1 <= w[1].1, "distances should be ascending");
        }
    }

    #[test]
    fn summary_returns_dim_centroid_and_radius() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let (centroid, radius) = r.summary("embedding").expect("vector summary");
        assert_eq!(centroid.len(), 16);
        assert!(radius >= 0.0);
        assert!(r.summary("nonexistent").is_none());
    }

    // -----------------------------------------------------------------
    // Plan 011 M1 — Source enum sanity tests
    // -----------------------------------------------------------------
    //
    // M1 only adds the enum + reroutes runtime byte access through
    // it; the public open path still takes a `Bytes` (M2 introduces
    // `open_lazy`). These tests directly exercise the `Source`
    // contract so any future refactor that breaks the InMemory
    // zero-copy invariant or mis-implements the Lazy wrapper fails
    // here rather than at the wider Lance oracle gate.

    #[test]
    fn source_in_memory_try_get_range_sync_zero_copy() {
        let payload = Bytes::from(vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let src = Source::InMemory(payload.clone());
        let slice = src
            .try_get_range_sync(3..7)
            .expect("in-bounds InMemory sync must succeed");
        assert_eq!(slice.as_ref(), &payload[3..7]);
        // Zero-copy invariant: returned Bytes points into the
        // same allocation as the source.
        let expected_ptr = unsafe { payload.as_ptr().add(3) };
        assert_eq!(slice.as_ptr(), expected_ptr);
    }

    #[test]
    fn source_in_memory_try_get_range_sync_out_of_bounds_returns_none() {
        let src = Source::InMemory(Bytes::from(vec![0u8; 4]));
        assert!(src.try_get_range_sync(0..100).is_none());
        assert!(src.try_get_range_sync(8..10).is_none());
    }

    #[tokio::test]
    async fn source_in_memory_get_range_round_trips_through_async() {
        let payload = Bytes::from(vec![100u8, 101, 102, 103, 104, 105]);
        let src = Source::InMemory(payload.clone());
        let got = src
            .get_range(1..5)
            .await
            .expect("InMemory async always succeeds");
        assert_eq!(got.as_ref(), &payload[1..5]);
    }

    #[test]
    fn source_lazy_try_get_range_sync_falls_through_to_trait_default_or_impl() {
        // Wrap an in-memory blob in the trait-shaped
        // `BytesLazyByteSource`, then in `Source::Lazy`. The lazy
        // adapter's `try_get_range_sync` impl returns `Some` for
        // in-bounds ranges; we exercise the full enum dispatch
        // path here so the Lazy arm of `Source::try_get_range_sync`
        // doesn't drift apart from the InMemory arm.
        use crate::superfile::lazy_source::BytesLazyByteSource;
        let payload = Bytes::from(vec![7u8, 8, 9, 10, 11, 12, 13, 14]);
        let lazy: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(payload.clone()));
        let src = Source::Lazy(lazy);
        let slice = src
            .try_get_range_sync(2..6)
            .expect("BytesLazyByteSource always serves sync");
        assert_eq!(slice.as_ref(), &payload[2..6]);
    }

    #[tokio::test]
    async fn source_lazy_get_range_dispatches_to_trait_range() {
        use crate::superfile::lazy_source::BytesLazyByteSource;
        let payload = Bytes::from(vec![21u8, 22, 23, 24, 25, 26, 27]);
        let lazy: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(payload.clone()));
        let src = Source::Lazy(lazy);
        let got = src.get_range(1..5).await.expect("async range");
        assert_eq!(got.as_ref(), &payload[1..5]);
    }

    /// `Source::Clone` lets readers share the underlying
    /// state cheaply (refcount bump). Clones must observe
    /// identical bytes — no fork between paths.
    #[test]
    fn source_clone_observes_identical_bytes() {
        let payload = Bytes::from(vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let a = Source::InMemory(payload.clone());
        let b = a.clone();
        let sa = a.try_get_range_sync(2..6).expect("sa");
        let sb = b.try_get_range_sync(2..6).expect("sb");
        assert_eq!(sa.as_ref(), sb.as_ref());
        assert_eq!(sa.as_ptr(), sb.as_ptr());
    }

    #[test]
    fn open_rejects_columns_json_mismatch() {
        let (blob, _) = build_blob(32, 16, 4, Metric::L2Sq);
        // header says 1 column; pass 2-column JSON.
        let bad_json = r#"[{"name":"a","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"},{"name":"b","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#;
        let err = VectorReader::open(blob, bad_json).expect_err("expected error");
        assert!(matches!(
            err,
            VectorError::Read(ReadError::MalformedVersion(_))
        ));
    }

    // -----------------------------------------------------------------
    // Plan 011 M2 — lazy `doc_to_pos` + `OpenOptions::prefetch_eager`
    // -----------------------------------------------------------------
    //
    // These tests pin the M2 contract: `open_with` no longer touches
    // the `doc_ids` region when `prefetch_eager: false` (default),
    // the table populates on first rerank via `OnceLock`, and the
    // search results are bit-for-bit identical to the eager path.
    // The memory-ceiling guarantee is asserted as a structural
    // post-condition: `doc_to_pos.get() == None` immediately after
    // a lazy open, `Some(vec.len() == n_docs)` after the first
    // rerank.

    /// Build the same shape used by the search-correctness tests
    /// elsewhere in this module, with a non-trivial `n_docs` so
    /// the `doc_to_pos` allocation is observable (≥ n_cent so
    /// the cluster walk has work to do).
    fn build_search_corpus() -> (Bytes, String, Vec<Vec<f32>>) {
        let dim = 16usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            name: "embedding".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::L2Sq,
        })
        .expect("register column");
        let mut all = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i.wrapping_mul(13) + j as u32 * 5) % 100) as f32)
                .collect();
            b.add(0, &v).expect("add to vector builder");
            all.push(v);
        }
        let bytes = b.finish().expect("finish vector builder");
        let json = r#"[{"name":"embedding","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#
            .to_string();
        (Bytes::from(bytes), json, all)
    }

    /// Default open uses `prefetch_eager: false` — the per-column
    /// `OnceLock<Vec<u32>>` must be empty right after open. This
    /// is the memory-ceiling pre-state: zero bytes allocated for
    /// `doc_to_pos` until a rerank touches it.
    #[test]
    fn open_lazy_default_leaves_doc_to_pos_empty() {
        let (blob, json, _) = build_search_corpus();
        let r = VectorReader::open(blob, &json).expect("open");
        for col in &r.columns {
            assert!(
                col.doc_to_pos.get().is_none(),
                "lazy open must leave doc_to_pos empty for column '{}', \
                 got Some({} entries)",
                col.name,
                col.doc_to_pos.get().map(|v| v.len()).unwrap_or(0)
            );
        }
    }

    /// `prefetch_eager: true` runs the cluster walk at open time
    /// (today's pre-011 semantics). The `OnceLock` is populated
    /// before any `search()` is called, with exactly `n_docs`
    /// entries per column.
    #[test]
    fn open_eager_populates_doc_to_pos_at_open() {
        let (blob, json, _) = build_search_corpus();
        let r = VectorReader::open_with(
            blob,
            &json,
            OpenOptions {
                verify_crc: true,
                prefetch_eager: true,
            },
        )
        .expect("open");
        for col in &r.columns {
            let table = col.doc_to_pos.get().unwrap_or_else(|| {
                panic!(
                    "eager open must populate doc_to_pos for column '{}', got None",
                    col.name
                )
            });
            assert_eq!(
                table.len(),
                col.n_docs as usize,
                "doc_to_pos length for column '{}' should equal n_docs",
                col.name
            );
        }
    }

    /// On the lazy path, the first `search()` that reaches the
    /// rerank stage must populate the `OnceLock`. Captures the
    /// transition pre → post first-rerank that plan 011 calls out
    /// in its memory-ceiling table.
    #[test]
    fn first_search_on_lazy_path_populates_doc_to_pos() {
        let (blob, json, all) = build_search_corpus();
        let r = VectorReader::open(blob, &json).expect("open");

        let col = &r.columns[0];
        assert!(
            col.doc_to_pos.get().is_none(),
            "pre-state: doc_to_pos must start empty"
        );

        let hits = r
            .search("embedding", &all[17], 5, 4, 5)
            .expect("search must succeed on lazy InMemory");
        assert_eq!(hits[0].0, 17, "self-query must recover self");

        let table = r.columns[0]
            .doc_to_pos
            .get()
            .expect("post-state: doc_to_pos must be populated after first rerank");
        assert_eq!(
            table.len(),
            r.columns[0].n_docs as usize,
            "populated table length must equal n_docs"
        );
    }

    /// Bit-for-bit parity between `prefetch_eager: true` and
    /// `prefetch_eager: false` paths. The lazy build runs the
    /// exact same `build_doc_to_pos` function as the eager open;
    /// any drift would surface as different search results on
    /// identical input.
    #[test]
    fn lazy_vs_eager_search_results_bit_for_bit_identical() {
        let (blob, json, all) = build_search_corpus();

        let r_eager = VectorReader::open_with(
            blob.clone(),
            &json,
            OpenOptions {
                verify_crc: true,
                prefetch_eager: true,
            },
        )
        .expect("eager open");
        let r_lazy = VectorReader::open_with(
            blob,
            &json,
            OpenOptions {
                verify_crc: true,
                prefetch_eager: false,
            },
        )
        .expect("lazy open");

        for &q_idx in &[0usize, 17, 31, 63] {
            let hits_eager = r_eager
                .search("embedding", &all[q_idx], 5, 4, 5)
                .expect("eager search");
            let hits_lazy = r_lazy
                .search("embedding", &all[q_idx], 5, 4, 5)
                .expect("lazy search");
            assert_eq!(
                hits_eager, hits_lazy,
                "eager vs lazy results must match for query {q_idx}"
            );
        }
    }

    /// `OnceLock::set` is single-shot. Two back-to-back searches
    /// on the lazy path must observe the same allocation (proves
    /// the second call doesn't rebuild the table). We compare
    /// the underlying `Vec`'s data pointer, which only stays
    /// stable across calls if the lookup table is reused.
    #[test]
    fn second_search_reuses_lazy_built_doc_to_pos() {
        let (blob, json, all) = build_search_corpus();
        let r = VectorReader::open(blob, &json).expect("open");

        let _ = r
            .search("embedding", &all[3], 5, 4, 5)
            .expect("first search");
        let ptr_after_first = r.columns[0]
            .doc_to_pos
            .get()
            .expect("populated after first search")
            .as_ptr();

        let _ = r
            .search("embedding", &all[40], 5, 4, 5)
            .expect("second search");
        let ptr_after_second = r.columns[0]
            .doc_to_pos
            .get()
            .expect("still populated after second search")
            .as_ptr();

        assert_eq!(
            ptr_after_first, ptr_after_second,
            "OnceLock::set is single-shot; second search must reuse the same allocation"
        );
    }

    /// Memory-ceiling proxy at the structural level: per column,
    /// the lazy `doc_to_pos` allocation is bounded by
    /// `n_docs × 4` bytes when populated and is exactly 0 bytes
    /// when empty. The `Vec<u32>::capacity()` ≥ length invariant
    /// means `capacity * 4 ≥ resident` for the lookup table.
    /// We pin the upper bound at `2 × n_docs × 4` to absorb the
    /// Vec's reserve slack without letting it grow unbounded.
    #[test]
    fn doc_to_pos_lazy_allocation_is_bounded_by_n_docs() {
        let (blob, json, all) = build_search_corpus();

        let r_lazy = VectorReader::open(blob.clone(), &json).expect("lazy open");
        let col = &r_lazy.columns[0];
        let n_docs = col.n_docs as usize;
        assert_eq!(
            col.doc_to_pos.get().map(|v| v.capacity()).unwrap_or(0),
            0,
            "lazy open: zero bytes for doc_to_pos"
        );

        let _ = r_lazy
            .search("embedding", &all[0], 5, 4, 5)
            .expect("first search to trigger lazy build");
        let cap = r_lazy.columns[0]
            .doc_to_pos
            .get()
            .expect("populated after rerank")
            .capacity();
        assert!(
            cap >= n_docs && cap <= 2 * n_docs,
            "post-rerank: doc_to_pos capacity {cap} should be in [n_docs, 2 × n_docs] (n_docs={n_docs})"
        );
    }
}
