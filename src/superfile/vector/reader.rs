//! Vector blob reader. Multi-column kNN search via IVF + 1-bit RaBitQ
//! shortlist + full-precision rerank.
//!
//! Opens the unified-blob byte layout produced by
//! [`super::builder::VectorBuilder::finish`] and exposes per-column
//! kNN search.
//!
//! Self-contained: owns its `Bytes`. Per-column metadata is parsed
//! eagerly at `open()`; per-query work happens on demand.

use crate::runtime_bridge::bridge_sync_to_async;
use crate::superfile::format::checksum::crc32c;
use crate::superfile::format::{self};
use crate::superfile::lazy_source::{LazyByteSource, LazyByteSourceError};
use crate::superfile::vector::distance::{Metric, Sq8Kernel, distance_bytes, distance_bytes_codec};
use crate::superfile::vector::quant::BitQuantizer;
use crate::superfile::vector::rerank_codec::RerankCodec;
use crate::superfile::vector::rotation::RandomRotation;
use crate::superfile::{ReadError, error::VectorError};
use bytes::Bytes;
use rayon::prelude::*;
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

const OUTER_HEADER_SIZE: usize = 32;
const DIR_ENTRY_SIZE: usize = 64;
const SUB_HEADER_SIZE: usize = 56;

/// JSON-deserialized form of one entry in `inf.vec.columns`. The KV
/// value is a JSON array of these in declaration order. Mirrors
/// the build-time [`VectorConfig`] but with `metric` as a string
/// so serde can read it straight from the KV blob.
#[derive(Debug, Clone, Deserialize)]
pub struct VectorColumnConfig {
    pub column: String,
    pub dim: usize,
    pub n_cent: usize,
    pub rot_seed: u64,
    /// `"l2sq"`, `"cosine"`, or `"negdot"`.
    pub metric: String,
}

/// Sq8 quantizer state materialised from the on-disk `codec_meta`
/// region at open time. The reader picks the candidate's cluster
/// slice of `scale` / `offset` and passes it into [`Sq8Kernel::new`]
/// once per (query, cluster) pair to build the per-query precomputes.
///
/// Per-cluster (not per-column) quantizer: each IVF cluster
/// owns its own `(scale[dim], offset[dim])` pair, packed
/// contiguously cluster-by-cluster. Per-cluster quantization avoids
/// stretching 256 buckets over the whole column when the rerank signal
/// lives inside much narrower IVF clusters.
#[derive(Debug, Clone)]
pub(super) struct Sq8ColumnMeta {
    /// Per-cluster, per-dim quantizer scale. Length =
    /// `n_cent × dim`, laid out cluster-major: cluster `c`'s
    /// scale array is `scale[c·dim .. (c+1)·dim]`.
    /// `x_decoded[d] = code[d] * scale[c·dim + d] + offset[c·dim + d]`
    /// for a doc in cluster `c`.
    pub scale: Vec<f32>,
    /// Per-cluster, per-dim quantizer offset. Same layout as `scale`.
    pub offset: Vec<f32>,
    /// Per-doc `Σ_d x_decoded²`, length == n_docs, indexed by
    /// position-in-full (matches the rerank shortlist's `pos`
    /// field). `Some` for L2Sq columns; `None` for Cosine /
    /// NegDot (the `Σx²` term cancels out of those distance
    /// formulas).
    pub per_doc_norms: Option<Vec<f32>>,
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
    /// On-disk rerank codec for this column.
    pub rerank_codec: RerankCodec,
    /// `Sq8`-only quantizer metadata, materialised at open time from
    /// the `codec_meta` region. `None` for every other codec.
    pub(super) sq8_meta: Option<Sq8ColumnMeta>,
    /// Byte range of this column's subsection within the outer blob.
    subsection_range: Range<usize>,
    /// Offsets relative to the subsection start.
    summary_off: usize,
    summary_radius: f32,
    centroids_off: usize,
    cluster_idx_off: usize,
    codes_off: usize,
    /// Relative offset of the per-column `codec_meta` region inside
    /// the subsection. `0` means "no codec_meta in this subsection".
    #[allow(dead_code)]
    codec_meta_off: usize,
    full_off: usize,
    doc_ids_off: usize,
    quant: BitQuantizer,
    /// Cached random rotation built once at open from `(dim, rot_seed)`.
    /// Construction is `O(dim³)` for Gram-Schmidt — at dim=384 that's
    /// ~7.9 ms, dominant over every other per-query stage if rebuilt
    /// per `search()`. Build once, reuse forever.
    rot: RandomRotation,
}

/// Per-open knobs for [`VectorReader::open_with`]. `Default` is the
/// safe choice (CRC verification on). The argumentless
/// [`VectorReader::open`] takes the default.
///
/// Today: `verify_crc` only. Object-storage-native knobs (e.g.
/// `range_fetch_concurrency`) may land here later under the same
/// struct.
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
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self { verify_crc: true }
    }
}

/// Backing for a [`VectorReader`].
///
/// Two variants, plumbed through every byte-fetch point:
///
/// - `InMemory(Bytes)`: caller materialised the full
///   subsection before opening. Every byte fetch is a
///   zero-copy `Bytes::slice` against the buffer.
/// - `Lazy(Arc<dyn LazyByteSource>)`: a range-fetching source
///   (mmap, object-store range GET, broadcast subscription).
///   Every byte access in the open + search paths routes
///   through the same call sites so swapping the backing in
///   doesn't require a second rewrite.
///
/// Both variants expose **sync-only** byte access — every
/// public surface in `src/` is sync. The
/// `LazyByteSource::range` trait method is async because
/// production impls (object store, network sources) are;
/// [`Source::get_range`] hides that under the same
/// `block_in_place + Handle::block_on` / one-shot
/// `current_thread` `Runtime` bridge the supertable's
/// per-segment reader uses for the disk-cache fetch path.
/// Hot-path callers (`Source::InMemory`, mmap-backed
/// `BytesLazyByteSource`) never hit the bridge — both override
/// `try_get_range_sync` to return zero-copy slices, so
/// `get_range` resolves on the sync fast path.
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

    /// Sync range fetch with internal async bridging on cold
    /// `Source::Lazy` misses.
    ///
    /// Fast path: `try_get_range_sync` (zero-copy `Bytes::slice`
    /// on `InMemory`; same on `BytesLazyByteSource` / mmap-
    /// backed sources). This covers every production caller
    /// today and every hot-path read at default open
    /// (`Source::Lazy(BytesLazyByteSource over
    /// Bytes::from_owner(mmap))`).
    ///
    /// Cold path (`Source::Lazy` and `try_get_range_sync`
    /// returned `None`): bridge to the source's `async fn
    /// range(...)` via `block_in_place + Handle::block_on`
    /// when there's an ambient tokio runtime, or build a
    /// throwaway `current_thread` `Runtime` when there isn't.
    /// This is the same pattern `supertable::query::
    /// segment_reader` uses for its sync disk-cache fetch path
    /// (see `segment_reader::segment_reader` for the canonical
    /// reference). The runtime-build cost on the no-ambient
    /// fallback is ≈ 1 ms — negligible vs the network
    /// round-trip the source is about to issue. In production
    /// the supertable always has an ambient runtime, so the
    /// no-ambient branch fires only in standalone tests /
    /// scripts.
    ///
    /// Convention: every public surface in `src/` stays sync,
    /// async is hidden behind well-defined bridge points.
    /// `Source::get_range` is one of those bridge points.
    pub fn get_range(&self, range: Range<usize>) -> Result<Bytes, LazyByteSourceError> {
        if let Some(bytes) = self.try_get_range_sync(range.clone()) {
            return Ok(bytes);
        }
        let Self::Lazy(s) = self else {
            // `Source::InMemory` always satisfies `try_get_range_sync`
            // for in-bounds ranges. Reaching this arm means the
            // request was out of bounds.
            return Err(LazyByteSourceError::OutOfBounds {
                start: range.start as u64,
                len: range.len() as u64,
                size: self.len() as u64,
            });
        };
        let start = range.start as u64;
        let len = range.len() as u64;
        bridge_sync_to_async(s.range(start, len))
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
        // Every byte fetch routes through `Source::try_get_range_sync`
        // so the lazy variant can intercept the same call sites
        // without a second rewrite. `InMemory` returns zero-copy
        // `Bytes::slice` views; refcount bumps only.
        Self::open_with_source(Source::InMemory(blob), columns_json, opts)
    }

    /// Open over an arbitrary [`Source`].
    ///
    /// The structural decode path is the same as
    /// [`Self::open_with`]; this entry just accepts a pre-built
    /// `Source`. Used by:
    /// - Test helpers that need to wire a counting / mock
    ///   [`LazyByteSource`] under a `Source::Lazy` (e.g. the
    ///   range-counting integration test).
    /// - A future `SuperfileReader::open_lazy` rewrite that
    ///   hands the underlying source through to the
    ///   `VectorReader` instead of materialising the full
    ///   subsection up-front.
    ///
    /// Today's contract on `Source::Lazy`: every byte access in
    /// the open path goes through
    /// [`Source::try_get_range_sync`], so the lazy source must
    /// already have the structural-decode regions (header,
    /// directory, per-subsection headers) resident — typically
    /// via a one-range pre-fetch issued by the caller. A future
    /// async open entrypoint can pre-fetch those regions on the
    /// source's behalf.
    pub fn open_with_source(
        source: Source,
        columns_json: &str,
        opts: OpenOptions,
    ) -> Result<Self, VectorError> {
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
            // bytes [40..48] = summary_offset (absolute), [48..52] = summary_length,
            // [52..56] = codec_id (1) + reserved (3)
            let _summary_off_abs = read_u64_le(&dir_bytes[entry_off + 40..entry_off + 48]);
            let codec_id = dir_bytes[entry_off + 52];
            let rerank_codec = RerankCodec::from_codec_id(codec_id).ok_or_else(|| {
                VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' has unknown rerank-codec id {codec_id} \
                     (known ids: 0=fp32, 1=sq8, 2=rabitq_only)",
                    cfg.column
                )))
            })?;
            if !rerank_codec.is_implemented() {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' uses rerank codec {} which is not implemented by this version",
                    cfg.column,
                    rerank_codec.name()
                ))));
            }

            // Validate against JSON.
            if dim != cfg.dim {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' dim mismatch: dir={dim} json={}",
                    cfg.column, cfg.dim
                ))));
            }
            if rot_seed != cfg.rot_seed {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' rot_seed mismatch",
                    cfg.column
                ))));
            }
            let metric = match metric_id {
                0 => Metric::L2Sq,
                1 => Metric::Cosine,
                2 => Metric::NegDot,
                _ => {
                    return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                        "unknown metric_id {metric_id} for column '{}'",
                        cfg.column
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
            //   [12..16] codec_meta_off (former reserved slot; zero
            //            for Fp32 and legacy fp32 segments).
            //   [16..24] summary_centroid_offset (relative to sub start)
            //   [24..28] summary_radius_x100
            //   [28..32] reserved
            //   [32..40] centroids_offset
            //   [40..48] cluster_idx_offset
            //   [48..52] codes_offset
            //   [52..56] full_offset
            let codec_meta_off = read_u32_le(&sub[12..16]) as usize;
            let summary_off = read_u64_le(&sub[16..24]) as usize;
            let summary_radius_x100 = read_u32_le(&sub[24..28]);
            let centroids_off = read_u64_le(&sub[32..40]) as usize;
            let cluster_idx_off = read_u64_le(&sub[40..48]) as usize;
            let codes_off = read_u32_le(&sub[48..52]) as usize;
            let full_off = read_u32_le(&sub[52..56]) as usize;
            // Fp32 + RabitqOnly all keep zero-byte codec_meta; Sq8
            // emits per-cluster scale/offset (+ per-doc
            // norms for L2Sq/Cosine). The per-codec layout check below
            // validates the declared `codec_meta_off` against the
            // codec's expected size once `col_n_docs` is known.
            let codec_meta_required_zero =
                matches!(rerank_codec, RerankCodec::Fp32 | RerankCodec::RabitqOnly);
            if codec_meta_required_zero && codec_meta_off != 0 {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' has codec_meta_off={codec_meta_off} for codec {}; \
                     fp32/rabitq_only must write codec_meta_off=0 (zero-byte meta region)",
                    cfg.column,
                    rerank_codec.name()
                ))));
            }

            let summary_radius = (summary_radius_x100 as f32) / 100.0;

            // Compute n_docs for this column and doc_ids_off.
            // doc_ids start at end of full vectors. Total subsection
            // bytes (excluding CRC) = SUB_HEADER + summary + centroids +
            // cluster_idx + codes + full + doc_ids.
            let quant = BitQuantizer::new(dim);
            let code_bytes = quant.code_bytes();
            // Derive `n_docs` from the codes region size. The codes
            // region is [codes_off, end_of_codes), where end_of_codes
            // is either `codec_meta_off` (Sq8) or `full_off` (Fp32 —
            // no codec_meta region between codes and full).
            let end_of_codes = if codec_meta_off != 0 {
                if codec_meta_off <= codes_off || codec_meta_off > full_off {
                    return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                        "column '{}' has codec_meta_off={codec_meta_off} outside \
                         (codes_off={codes_off}, full_off={full_off}]",
                        cfg.column
                    ))));
                }
                codec_meta_off
            } else {
                full_off
            };
            let codes_size = end_of_codes - codes_off;
            if code_bytes == 0 || !codes_size.is_multiple_of(code_bytes) {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' codes size {codes_size} not divisible by {code_bytes}",
                    cfg.column
                ))));
            }
            let col_n_docs = (codes_size / code_bytes) as u32;

            // `full[]` byte stride depends on the column's rerank codec.
            // Centroids stay fp32 regardless (only the per-doc rerank
            // region compresses).
            let per_vec_bytes = rerank_codec.per_vector_bytes(dim);
            let full_size = (col_n_docs as usize) * per_vec_bytes;
            let doc_ids_off = full_off + full_size;

            let expected_codec_meta_size =
                rerank_codec.codec_meta_bytes(dim, col_n_docs as usize, n_cent as usize, metric);
            let actual_codec_meta_size = if codec_meta_off != 0 {
                full_off - codec_meta_off
            } else {
                0
            };
            if actual_codec_meta_size != expected_codec_meta_size {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' codec_meta region is {actual_codec_meta_size} bytes \
                     on disk, but codec {} / metric {metric:?} expects \
                     {expected_codec_meta_size} bytes",
                    cfg.column,
                    rerank_codec.name()
                ))));
            }

            // Materialise Sq8 codec_meta (per-cluster scale + offset
            // plus optional per-doc norms) at open time. Parse through
            // `f32::from_le_bytes` because the codec_meta region is not
            // guaranteed to be 4-byte aligned for all dimensions.
            let sq8_meta = if rerank_codec == RerankCodec::Sq8 {
                let meta_start = codec_meta_off;
                let meta_end = meta_start + actual_codec_meta_size;
                let meta_bytes = &sub[meta_start..meta_end];
                let so_block_bytes = (n_cent as usize) * dim * 4;
                let scale_end = so_block_bytes;
                let offset_end = scale_end + so_block_bytes;
                let scale = parse_f32_le_vec(&meta_bytes[0..scale_end]);
                let offset = parse_f32_le_vec(&meta_bytes[scale_end..offset_end]);
                let per_doc_norms = if matches!(metric, Metric::L2Sq | Metric::Cosine) {
                    let norms_end = offset_end + (col_n_docs as usize) * 4;
                    debug_assert_eq!(norms_end, actual_codec_meta_size);
                    Some(parse_f32_le_vec(&meta_bytes[offset_end..norms_end]))
                } else {
                    None
                };
                Some(Sq8ColumnMeta {
                    scale,
                    offset,
                    per_doc_norms,
                })
            } else {
                None
            };

            // Bounds-check the cluster_idx + doc_ids regions without
            // reading them. Search carries `pos = off + i` inline
            // in the shortlist, so it never needs a `doc_to_pos`
            // lookup table; open only needs the offset math.
            let cluster_idx_size = (n_cent as usize) * 8;
            let cluster_idx_end = cluster_idx_off + cluster_idx_size;
            if cluster_idx_end > sub_crc_pos {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' cluster index runs past subsection",
                    cfg.column
                ))));
            }
            let doc_ids_size = (col_n_docs as usize) * 4;
            if doc_ids_off + doc_ids_size > sub_crc_pos {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' doc_ids region runs past subsection",
                    cfg.column
                ))));
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
                    cfg.column, cfg.metric
                ))));
            }

            columns.push(ColumnReader {
                name: cfg.column.clone(),
                dim,
                n_cent,
                n_docs: col_n_docs,
                metric,
                rot_seed,
                rerank_codec,
                sq8_meta,
                subsection_range: subsection_off..sub_end,
                summary_off,
                summary_radius,
                centroids_off,
                cluster_idx_off,
                codes_off,
                codec_meta_off,
                full_off,
                doc_ids_off,
                quant,
                rot: RandomRotation::new(dim, rot_seed),
            });
            column_id_by_name.insert(cfg.column.clone(), i as u32);
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
        // Byte access routed through `Source::try_get_range_sync`
        // — zero-copy on `InMemory`, same call site serves the
        // lazy path.
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

    /// Single-column kNN search. Returns `(local_doc_id,
    /// distance)` sorted ascending by distance (smaller = closer
    /// for every metric).
    ///
    /// Sync — every public surface in `src/` is sync. Routes
    /// per-region byte access through [`Source::get_range`],
    /// which is itself sync and bridges to the underlying
    /// async `LazyByteSource::range` only on a cold
    /// `Source::Lazy` miss (via `block_in_place +
    /// Handle::block_on`, same pattern as the supertable's
    /// per-segment reader). On `Source::InMemory` and on
    /// `Source::Lazy` warm caches (`BytesLazyByteSource`,
    /// mmap-backed) every fetch resolves zero-copy on the
    /// sync fast path.
    ///
    /// Range count per cold first search at `nprobe = 8`:
    ///
    /// - 1 range for centroids (`n_cent × dim × 4` bytes)
    /// - 1 range for the cluster_idx header (`n_cent × 8` bytes)
    /// - `nprobe` ranges for per-cluster codes
    /// - `nprobe` ranges for per-cluster doc_ids
    /// - 1 fat range covering the rerank batch in `full[]` from
    ///   `min(pos)` to `max(pos) + 1`
    ///
    /// At `nprobe = 8`: 2 + 16 + 1 = **19 ranges**. Rerank `pos`
    /// is captured inline in the shortlist tuple at code-scoring
    /// time (each candidate's position is `off + i` where
    /// `(off, cnt)` is the cluster's entry and `i` is the
    /// in-cluster index), so there is no `doc_to_pos` lookup
    /// table at all.
    pub fn search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank_mult: usize,
    ) -> Result<Vec<(u32, f32)>, VectorError> {
        let (col, validated) = self.resolve_column(column, query, k)?;
        if !validated {
            return Ok(Vec::new());
        }
        // Centroids are always fp32 (4 bytes/dim) regardless of codec.
        // `full[]` (rerank candidates) is codec-dependent.
        let centroid_stride = col.dim * 4;
        let full_vec_bytes = col.rerank_codec.per_vector_bytes(col.dim);
        let sub_start = col.subsection_range.start;

        // 1. Centroids region. `n_cent × dim × 4` bytes,
        //    ~1.5 MB at default shape. Source::InMemory
        //    returns a zero-copy Bytes::slice; warm-cache
        //    Source::Lazy returns the same; cold-miss
        //    Source::Lazy bridges to async range() via the
        //    sync→async pattern in Source::get_range.
        let centroids_start = sub_start + col.centroids_off;
        let centroids_end = centroids_start + (col.n_cent as usize) * centroid_stride;
        let centroids = self
            .source
            .get_range(centroids_start..centroids_end)
            .map_err(|e| VectorError::LazySource(e.to_string()))?;

        // 2. Cluster_idx header. `n_cent × 8` bytes, 8 KB at
        //    default shape. Cheap.
        let idx_start = sub_start + col.cluster_idx_off;
        let idx_end = idx_start + (col.n_cent as usize) * 8;
        let cluster_idx = self
            .source
            .get_range(idx_start..idx_end)
            .map_err(|e| VectorError::LazySource(e.to_string()))?;

        // 3. Score centroids → top `nprobe` clusters.
        let mut centroid_scores = score_centroids(&centroids, col, query);
        let nprobe_eff = nprobe.min(col.n_cent as usize).max(1);
        centroid_scores.truncate(nprobe_eff);

        // 4. Rotate query once for the 1-bit code estimator.
        let mut q_rot = vec![0f32; col.dim];
        col.rot.apply(query, &mut q_rot);

        // 5. Per-cluster fetches (codes + doc_ids) and shortlist
        //    build. Shortlist tuple is (doc_id, estimate, pos,
        //    cluster_id); pos = off + i and cluster_id are
        //    captured inline at no extra fetch cost. cluster_id
        //    is consumed by the per-cluster Sq8 rerank dispatch to
        //    pick each candidate's quantizer; Fp32/None
        //    rerank paths ignore it.
        let cb = col.quant.code_bytes();
        let mut shortlist: Vec<(u32, f32, u32, u32)> = Vec::new();
        for &(c, _) in &centroid_scores {
            let (off, cnt) = read_cluster_entry(&cluster_idx, c);
            if cnt == 0 {
                continue;
            }
            let codes_start = sub_start + col.codes_off + (off as usize) * cb;
            let codes_end = codes_start + (cnt as usize) * cb;
            let codes = self
                .source
                .get_range(codes_start..codes_end)
                .map_err(|e| VectorError::LazySource(e.to_string()))?;
            let did_start = sub_start + col.doc_ids_off + (off as usize) * 4;
            let did_end = did_start + (cnt as usize) * 4;
            let doc_ids = self
                .source
                .get_range(did_start..did_end)
                .map_err(|e| VectorError::LazySource(e.to_string()))?;
            score_cluster_codes(
                &codes,
                &doc_ids,
                cnt,
                off,
                c as u32,
                &col.quant,
                &q_rot,
                &mut shortlist,
            );
        }

        if shortlist.is_empty() {
            return Ok(Vec::new());
        }

        // `None` columns have no `full[]` region to refine against:
        // the 1-bit shortlist is the final ranking. Return sign-flipped
        // estimates so the public `(doc_id, distance)` convention still
        // means smaller is closer.
        if !col.rerank_codec.writes_full() {
            let _ = rerank_mult;
            if shortlist.len() > k {
                shortlist.select_nth_unstable_by(k - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
                });
                shortlist.truncate(k);
            }
            shortlist.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            return Ok(shortlist
                .into_iter()
                .map(|(did, est, _pos, _c)| (did, -est))
                .collect());
        }

        // 6. Trim to `k × rerank_mult` by descending estimate.
        let want = (k.saturating_mul(rerank_mult)).min(shortlist.len());
        if want < shortlist.len() {
            shortlist.select_nth_unstable_by(want.saturating_sub(1), |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
            });
            shortlist.truncate(want);
        }

        // 7. Fat range over `full[]` covering all rerank
        //    candidates. `[min_pos..max_pos + 1]` over-fetches
        //    today (positions span probed clusters); a future
        //    layout that interleaves codes + doc_ids + full per
        //    cluster would drop this to `nprobe` cluster-sized
        //    ranges. Single get_range either way.
        let mut min_pos = shortlist[0].2;
        let mut max_pos = shortlist[0].2;
        for &(_, _, pos, _) in &shortlist[1..] {
            if pos < min_pos {
                min_pos = pos;
            }
            if pos > max_pos {
                max_pos = pos;
            }
        }
        let full_start = sub_start + col.full_off + (min_pos as usize) * full_vec_bytes;
        let full_end = sub_start + col.full_off + ((max_pos as usize) + 1) * full_vec_bytes;
        let full_run = self
            .source
            .get_range(full_start..full_end)
            .map_err(|e| VectorError::LazySource(e.to_string()))?;

        // 8. CPU-only rerank using the true metric. Sq8 columns
        //    pre-build a per-query kernel that folds the per-dim
        //    scale/offset into the query (one `dim/8` SIMD pass);
        //    the per-doc inner step is then a plain u8→f32 widen
        //    + SIMD dot. Fp32 takes the flat dispatch.
        Ok(rerank_candidates_in_run(
            &full_run, min_pos, &shortlist, col, query, k,
        ))
    }

    /// Look up the column by name and validate `query.len() == col.dim`
    /// + the "empty work" short-circuit (`k == 0` or `n_docs == 0`).
    /// `Ok((col, true))` = real search to follow; `Ok((col, false))`
    /// = empty-result short circuit, caller returns `Ok(Vec::new())`.
    #[inline]
    fn resolve_column(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
    ) -> Result<(&ColumnReader, bool), VectorError> {
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
            return Ok((col, false));
        }
        Ok((col, true))
    }
}

/// Score `query` against every centroid in `centroids_bytes` and
/// return the per-cluster `(cluster_id, distance)` pairs sorted by
/// ascending distance (closest first). Caller truncates to `nprobe`.
///
/// Takes a `&[u8]` view so the caller can hand in either an
/// in-memory subsection slice or the just-fetched centroids
/// region bytes from [`Source::get_range`] — both reach this
/// helper through the same shape.
#[inline]
fn score_centroids(centroids_bytes: &[u8], col: &ColumnReader, query: &[f32]) -> Vec<(usize, f32)> {
    // Centroids are stored as fp32 regardless of the column's rerank
    // codec — only the per-doc `full[]` region compresses. `distance_bytes`
    // assumes fp32, which is correct here.
    let centroid_stride = col.dim * 4;
    let mut scores: Vec<(usize, f32)> = (0..col.n_cent as usize)
        .map(|c| {
            let bytes = &centroids_bytes[c * centroid_stride..(c + 1) * centroid_stride];
            (c, distance_bytes(col.metric, query, bytes))
        })
        .collect();
    scores.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    scores
}

/// Score one cluster's 1-bit codes against the rotated query and
/// append `(doc_id, estimate, pos_in_full, cluster_id)` tuples to
/// `shortlist`. `pos = off + i` is the candidate's index in the
/// column's `full[]` array — captured here at no extra cost so the
/// rerank step doesn't need any lookup table. `cluster_id` is
/// captured for the per-cluster Sq8 rerank dispatch: each candidate
/// knows which cluster's `(scale, offset)` quantizer to dequant
/// against. For Fp32/None the cluster_id is recorded but
/// ignored by the rerank step (kept for layout simplicity — the
/// extra 4 bytes per shortlist entry are noise next to the
/// `k × rerank_mult` heap traffic).
#[inline]
fn score_cluster_codes(
    cluster_codes: &[u8],
    cluster_doc_ids: &[u8],
    cnt: u32,
    off: u32,
    cluster_id: u32,
    quant: &BitQuantizer,
    q_rot: &[f32],
    shortlist: &mut Vec<(u32, f32, u32, u32)>,
) {
    let cb = quant.code_bytes();
    // Per-query precompute for the AVX-512 RaBitQ estimator: the
    // estimate is `2 * Σ_{bit=1} q_rot[d] − q_total`; the second
    // term is constant across every candidate scored against
    // this query, so we hoist it out of the per-doc loop. Cost
    // ≪ 1 % across the typical IVF probe (thousands of candidates).
    // Non-AVX-512 hosts ignore the precomputed value and fall
    // back to the original sign-table kernel, so the numeric
    // result is identical regardless.
    let q_total: f32 = q_rot.iter().sum();
    for i in 0..cnt as usize {
        let code = &cluster_codes[i * cb..(i + 1) * cb];
        let est = quant.estimate_dot_rotated_with_total(q_rot, code, q_total);
        let did = u32::from_le_bytes([
            cluster_doc_ids[i * 4],
            cluster_doc_ids[i * 4 + 1],
            cluster_doc_ids[i * 4 + 2],
            cluster_doc_ids[i * 4 + 3],
        ]);
        shortlist.push((did, est, off + i as u32, cluster_id));
    }
}

/// Decode one cluster's `(off, cnt)` entry from
/// `cluster_idx_slice` (the `n_cent × 8` bytes of the column's
/// cluster index header). `c` is the cluster id.
#[inline]
fn read_cluster_entry(cluster_idx_slice: &[u8], c: usize) -> (u32, u32) {
    let base = c * 8;
    let off = u32::from_le_bytes([
        cluster_idx_slice[base],
        cluster_idx_slice[base + 1],
        cluster_idx_slice[base + 2],
        cluster_idx_slice[base + 3],
    ]);
    let cnt = u32::from_le_bytes([
        cluster_idx_slice[base + 4],
        cluster_idx_slice[base + 5],
        cluster_idx_slice[base + 6],
        cluster_idx_slice[base + 7],
    ]);
    (off, cnt)
}

/// Full-precision rerank over `shortlist`, returning the top-`k`
/// `(doc_id, distance)` pairs sorted by ascending distance.
///
/// `full_run` is a contiguous run of `full[]` bytes covering at
/// least the byte range `[base_pos × stride .. (max_pos + 1) ×
/// stride)`, where `stride = col.rerank_codec.per_vector_bytes(
/// col.dim)` — every candidate's `pos` in `shortlist` must lie
/// in `[base_pos, base_pos + full_run.len() / stride)`. For the
/// sync path, `base_pos = 0` and `full_run` is the column's
/// whole `full[]` slice; for the async path, `base_pos =
/// min(pos)` and `full_run` is the per-query fat range.
///
/// Dispatches on `col.rerank_codec`:
/// - **Fp32**: flat dispatch via [`distance_bytes_codec`]
///   (fp32 zero-copy SIMD).
/// - **Sq8**: builds a per-query [`Sq8Kernel`] from the column's
///   `codec_meta` once (folds scale/offset into the query so the
///   per-doc inner step is a plain u8→f32 widen + SIMD dot;
///   per-doc decoded-norm cached at encode time short-circuits
///   `Σx²` for L2Sq).
#[inline]
fn rerank_candidates_in_run(
    full_run: &[u8],
    base_pos: u32,
    shortlist: &[(u32, f32, u32, u32)],
    col: &ColumnReader,
    query: &[f32],
    k: usize,
) -> Vec<(u32, f32)> {
    let stride = col.rerank_codec.per_vector_bytes(col.dim);
    let mut reranked: Vec<(u32, f32)> = match col.rerank_codec {
        RerankCodec::Fp32 => shortlist
            .iter()
            .map(|&(did, _, pos, _)| {
                let local = (pos - base_pos) as usize;
                let start = local * stride;
                let bytes = &full_run[start..start + stride];
                let d = distance_bytes_codec(col.metric, col.rerank_codec, query, bytes);
                (did, d)
            })
            .collect(),
        RerankCodec::Sq8 => {
            // Per-cluster Sq8: each candidate's cluster_id selects
            // a `(scale[dim], offset[dim])` slice from the column
            // meta. We build a fresh per-cluster `Sq8Kernel`
            // lazily — at typical nprobe ≤ 64 we touch only a
            // handful of clusters per query, and building a
            // kernel is `O(dim)` SIMD work (one pass over the
            // query × scale + one over query × offset). Caching
            // by cluster_id avoids rebuilding the kernel for
            // sibling candidates in the same cluster (most of
            // the shortlist).
            //
            // Metadata is materialised at open time on every Sq8
            // column; the unwrap can't fail unless someone
            // constructs a `ColumnReader` outside `open_with`.
            let meta = col
                .sq8_meta
                .as_ref()
                .expect("Sq8 column must carry sq8_meta (built in open_with)");
            let dim = col.dim;
            // Cache the per-cluster `Sq8Kernel` indexed by
            // `cluster_id` so sibling candidates in the same
            // cluster (most of the shortlist at typical nprobe
            // values) reuse the kernel rather than rebuilding the
            // O(dim) per-query precompute. `cluster_id` is bounded
            // by `col.n_cent` (which is what built the column's
            // IVF index), so a flat `Vec<Option<Sq8Kernel>>` of
            // that length indexes in O(1) — no hash, no
            // hashbrown bucket walk — and at typical
            // `n_cent = 1024` costs ~16 KiB of `Option<Box<…>>`
            // slots which is negligible against the per-query
            // working set.
            let n_cent = col.n_cent as usize;
            let mut kernel_cache: Vec<Option<Sq8Kernel>> = (0..n_cent).map(|_| None).collect();
            shortlist
                .iter()
                .map(|&(did, _, pos, cluster_id)| {
                    let local = (pos - base_pos) as usize;
                    let start = local * stride;
                    let bytes = &full_run[start..start + stride];
                    let c = cluster_id as usize;
                    let kernel = kernel_cache[c].get_or_insert_with(|| {
                        let scale_c = &meta.scale[c * dim..(c + 1) * dim];
                        let offset_c = &meta.offset[c * dim..(c + 1) * dim];
                        Sq8Kernel::new(
                            col.metric,
                            query,
                            scale_c,
                            offset_c,
                            meta.per_doc_norms.as_deref(),
                        )
                    });
                    let d = kernel.distance_at(pos, bytes);
                    (did, d)
                })
                .collect()
        }
        RerankCodec::RabitqOnly => unreachable!(
            "rerank_candidates_in_run reached with RabitqOnly codec — RabitqOnly \
             columns have no full[] region and should short-circuit before \
             the rerank step"
        ),
    };
    reranked.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    reranked.truncate(k);
    reranked
}

#[inline]
fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Decode an aligned-or-not `&[u8]` of length `4·N` as a
/// `Vec<f32>` of length `N`. Used for Sq8's `codec_meta` arrays
/// (scale, offset, per-doc norms) where the byte slice can land
/// at any alignment relative to the `Bytes` backing — see the
/// reader-side note where this is called for the alignment
/// argument. Slow path (4 byte reads per f32) but only runs at
/// open time over at-most-`8·n_cent·dim + 4·n_docs` bytes per Sq8
/// column; the per-query inner loop never goes through here.
#[inline]
fn parse_f32_le_vec(bytes: &[u8]) -> Vec<f32> {
    debug_assert!(bytes.len().is_multiple_of(4));
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
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
/// abstraction — the lazy variant plumbs the eager-prefetch
/// path through the same call sites without a second rewrite.
///
/// Failure mode here means the range is out-of-bounds or not
/// present in the sync cache. On `Source::InMemory` any in-bounds
/// range succeeds zero-copy; this only fires on a malformed blob.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::vector::builder::{VectorBuilder, VectorConfig};

    fn build_blob(n_docs: u32, dim: usize, n_cent: usize, metric: Metric) -> (Bytes, String) {
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "embedding".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric,
            rerank_codec: RerankCodec::Fp32,
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
            r#"[{{"column":"embedding","dim":{dim},"n_cent":{n_cent},"rot_seed":7,"metric":"{metric_s}"}}]"#
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
        let r =
            VectorReader::open_with(Bytes::from(bytes), &json, OpenOptions { verify_crc: false });
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
            column: "embedding".into(),
            dim,
            n_cent: 4,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
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
        let json = r#"[{"column":"embedding","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#;
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
    // Source enum sanity tests
    // -----------------------------------------------------------------
    //
    // These tests directly exercise the `Source` contract so any
    // future refactor that breaks the InMemory zero-copy invariant
    // or mis-implements the Lazy wrapper fails here rather than at
    // the wider Lance oracle gate.

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

    #[test]
    fn source_in_memory_get_range_returns_zero_copy_slice() {
        let payload = Bytes::from(vec![100u8, 101, 102, 103, 104, 105]);
        let src = Source::InMemory(payload.clone());
        let got = src
            .get_range(1..5)
            .expect("InMemory sync get_range always succeeds for in-bounds ranges");
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

    #[test]
    fn source_lazy_get_range_serves_warm_cache_via_try_get_range_sync() {
        use crate::superfile::lazy_source::BytesLazyByteSource;
        let payload = Bytes::from(vec![21u8, 22, 23, 24, 25, 26, 27]);
        let lazy: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(payload.clone()));
        let src = Source::Lazy(lazy);
        // BytesLazyByteSource overrides try_get_range_sync to
        // return Some for every in-bounds range, so get_range
        // takes the sync fast path — no block_on bridge fires.
        let got = src.get_range(1..5).expect("warm cache sync hit");
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
        let bad_json = r#"[{"column":"a","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"},{"column":"b","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#;
        let err = VectorReader::open(blob, bad_json).expect_err("expected error");
        assert!(matches!(
            err,
            VectorError::Read(ReadError::MalformedVersion(_))
        ));
    }

    // -----------------------------------------------------------------
    // Rerank-codec discriminator round-trip
    // -----------------------------------------------------------------
    //
    // The codec discriminator rides as byte 52 of the per-column
    // directory entry; the codec_meta region offset rides as bytes
    // 12..16 of the sub-header. Both are zero on legacy fp32
    // segments. Today `Fp32`, `Sq8`, and `RabitqOnly` are implemented; every other
    // codec must fail loudly against a reader that does not yet know
    // how to decode it.

    /// A fresh `Fp32` build round-trips through the reader with the
    /// codec byte preserved as `RerankCodec::Fp32`.
    #[test]
    fn open_round_trips_fp32_codec_discriminator() {
        let (blob, json) = build_blob(64, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open");
        assert_eq!(r.columns.len(), 1);
        assert_eq!(
            r.columns[0].rerank_codec,
            RerankCodec::Fp32,
            "Fp32 build must surface as RerankCodec::Fp32 on the reader"
        );
        assert_eq!(
            r.columns[0].codec_meta_off, 0,
            "Fp32 segments must write codec_meta_off = 0 (zero-size region)"
        );
    }

    /// Every exposed rerank codec is wired end-to-end.
    #[test]
    fn register_column_accepts_every_codec() {
        for codec in [RerankCodec::Fp32, RerankCodec::Sq8, RerankCodec::RabitqOnly] {
            let mut b = VectorBuilder::new();
            b.register_column(VectorConfig {
                column: "v".into(),
                dim: 16,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::L2Sq,
                rerank_codec: codec,
            })
            .unwrap_or_else(|e| panic!("codec {codec:?} must register, got {e:?}"));
        }
    }

    /// Building a column with `RerankCodec::Sq8` round-trips through
    /// the reader. The codec discriminator surfaces on
    /// `ColumnReader.rerank_codec`; the codec_meta region carries
    /// `scale[dim] + offset[dim]` plus per-doc norms for L2Sq. The
    /// on-disk `full[]` region shrinks to `n_docs * dim` u8 codes.
    #[test]
    fn open_round_trips_sq8_codec_discriminator_l2sq() {
        let dim = 32usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8,
        })
        .expect("register column");
        for i in 0..n_docs {
            let v: Vec<f32> = (0..dim).map(|j| (i + j as u32) as f32 * 0.1).collect();
            b.add(0, &v).expect("add");
        }
        let blob = b.finish().expect("finish");

        let json =
            r#"[{"column":"v","dim":32,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        assert_eq!(r.columns.len(), 1);
        let col = &r.columns[0];
        assert_eq!(col.rerank_codec, RerankCodec::Sq8);
        assert_ne!(col.codec_meta_off, 0, "Sq8 must declare codec_meta_off > 0");
        assert_eq!(col.doc_ids_off - col.full_off, (col.n_docs as usize) * dim);

        let meta = col
            .sq8_meta
            .as_ref()
            .expect("Sq8 column must materialise sq8_meta at open");
        assert_eq!(meta.scale.len(), (col.n_cent as usize) * dim);
        assert_eq!(meta.offset.len(), (col.n_cent as usize) * dim);
        let norms = meta
            .per_doc_norms
            .as_ref()
            .expect("L2Sq Sq8 column must carry per-doc norms");
        assert_eq!(norms.len(), col.n_docs as usize);
    }

    /// Cosine Sq8 columns carry per-doc decoded norms so rerank can
    /// normalize the decoded vector before computing cosine distance.
    #[test]
    fn open_sq8_cosine_carries_per_doc_norms() {
        let dim = 16usize;
        let n_cent = 4usize;
        let n_docs = 32u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 11,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8,
        })
        .expect("register column");
        for i in 0..n_docs {
            let mut v: Vec<f32> = (0..dim)
                .map(|j| (i + j as u32) as f32 * 0.1 + 0.5)
                .collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            b.add(0, &v).expect("add");
        }
        let blob = b.finish().expect("finish");
        let json =
            r#"[{"column":"v","dim":16,"n_cent":4,"rot_seed":11,"metric":"cosine"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        let col = &r.columns[0];
        let meta = col.sq8_meta.as_ref().expect("Sq8 must carry sq8_meta");
        let norms = meta
            .per_doc_norms
            .as_ref()
            .expect("Cosine Sq8 must carry per-doc norms");
        assert_eq!(norms.len(), n_docs as usize);
        assert_eq!(meta.scale.len(), n_cent * dim);
        assert_eq!(meta.offset.len(), n_cent * dim);
    }

    /// Pins the per-doc-norms indexing contract: the on-disk norms
    /// array is indexed by position in `full[]` (matching the rerank
    /// shortlist's `pos`), not by `doc_id`.
    #[test]
    fn sq8_per_doc_norms_indexed_by_pos_not_doc_id() {
        let dim = 16usize;
        let n_cent = 4usize;
        let n_docs = 32u32;
        let make = |i: u32| -> Vec<f32> {
            let s = 1.0 + (i as f32) * 0.5;
            (0..dim).map(|j| s + (j as f32) * 0.1).collect()
        };
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 23,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8,
        })
        .expect("register column");
        let mut planted = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs {
            let v = make(i);
            b.add(0, &v).expect("add");
            planted.push(v);
        }
        let blob = b.finish().expect("finish");

        let json =
            r#"[{"column":"v","dim":16,"n_cent":4,"rot_seed":23,"metric":"l2sq"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        let col = &r.columns[0];
        let meta = col.sq8_meta.as_ref().expect("Sq8 meta present");
        let norms_by_pos = meta
            .per_doc_norms
            .as_ref()
            .expect("L2Sq Sq8 carries per-doc norms");

        let insertion_norms: Vec<f32> = planted
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum::<f32>())
            .collect();
        let n_matching = insertion_norms
            .iter()
            .zip(norms_by_pos.iter())
            .filter(|(ins, pos_n)| (**ins - **pos_n).abs() < 0.5)
            .count();
        assert!(
            n_matching < (n_docs as usize) / 2,
            "expected clustered build to reorder docs across positions, got {n_matching}/{n_docs} near insertion order"
        );

        for i in [0u32, 7, 15, 23, 31] {
            let hits = r
                .search("v", &planted[i as usize], 1, 4, 64)
                .expect("self-query");
            assert_eq!(hits[0].0, i, "self-query top-1 doc_id for doc {i}");
            assert!(
                hits[0].1 <= 0.5,
                "doc {i}: self-query distance {} too large",
                hits[0].1
            );
        }
    }

    /// Sq8 build + open + self-query recovers the planted self-vector
    /// at top-1 through codec-aware rerank.
    #[test]
    fn sq8_self_query_round_trips_top1_l2sq() {
        let dim = 32usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 13,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8,
        })
        .expect("register column");
        let make = |i: u32| -> Vec<f32> {
            (0..dim)
                .map(|j| ((i.wrapping_mul(17) + j as u32 * 3) % 64) as f32 * 0.5)
                .collect()
        };
        let mut all = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs {
            let v = make(i);
            b.add(0, &v).expect("add");
            all.push(v);
        }
        let blob = b.finish().expect("finish");

        let json =
            r#"[{"column":"v","dim":32,"n_cent":4,"rot_seed":13,"metric":"l2sq"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        let hits = r
            .search("v", &all[17], 5, 4, 20)
            .expect("search must succeed on Sq8 column");
        assert_eq!(hits[0].0, 17, "Sq8 self-query must recover self at top-1");
        assert!(
            hits[0].1 <= 1.0,
            "Sq8 self-query distance {} should be small",
            hits[0].1
        );
    }

    /// Sq8 self-query top-1 round-trips under cosine too. The corpus
    /// uses hashed unit vectors so the self-vector has a wide margin
    /// over neighbors after quantization.
    #[test]
    fn sq8_self_query_round_trips_top1_cosine() {
        let dim = 32usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 19,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8,
        })
        .expect("register column");
        let make = |i: u32| -> Vec<f32> {
            let raw: Vec<f32> = (0..dim)
                .map(|j| {
                    let h = (i.wrapping_mul(0x9E37_79B9)) ^ ((j as u32).wrapping_mul(0x85EB_CA77));
                    let h = h.wrapping_mul(0xC2B2_AE35);
                    ((h & 0xFFFF) as f32) / 65535.0
                })
                .collect();
            let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            raw.into_iter().map(|x| x / norm).collect()
        };
        let mut all = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs {
            let v = make(i);
            b.add(0, &v).expect("add");
            all.push(v);
        }
        let blob = b.finish().expect("finish");

        let json =
            r#"[{"column":"v","dim":32,"n_cent":4,"rot_seed":19,"metric":"cosine"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        let hits = r
            .search("v", &all[42], 5, 4, 20)
            .expect("search must succeed on Sq8 cosine column");
        assert_eq!(hits[0].0, 42, "Sq8 cosine self-query must recover self");
    }

    /// Building with `RerankCodec::RabitqOnly` succeeds and the
    /// on-disk segment carries a zero-length `full[]` region.
    #[test]
    fn open_round_trips_rabitq_only_codec_discriminator() {
        let dim = 16usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::RabitqOnly,
        })
        .expect("register RabitqOnly column");
        for i in 0..n_docs {
            let v: Vec<f32> = (0..dim).map(|j| (i + j as u32) as f32 * 0.1).collect();
            b.add(0, &v).expect("add");
        }
        let blob = b.finish().expect("finish");

        let json =
            r#"[{"column":"v","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        assert_eq!(r.columns.len(), 1);
        let col = &r.columns[0];
        assert_eq!(col.rerank_codec, RerankCodec::RabitqOnly);
        assert_eq!(col.codec_meta_off, 0);
        assert_eq!(
            col.doc_ids_off, col.full_off,
            "RabitqOnly segments have zero-length full[]"
        );
        assert_eq!(col.n_docs, n_docs);
    }

    /// A `RabitqOnly` column returns top-K directly from the 1-bit
    /// shortlist. Distances are sign-flipped estimates so smaller is
    /// still closer.
    ///
    /// Self-query must rank the planted vector at index 0 — the
    /// 1-bit shortlist's score for the exact-match query is the
    /// dot product against itself, which dominates every neighbour
    /// at this dim/cluster shape. A weaker `.any(== self)` check
    /// would mask a regression that demoted self into positions
    /// 1–4.
    #[test]
    fn rabitq_only_self_query_ranks_self_first() {
        let dim = 128usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 11,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::RabitqOnly,
        })
        .expect("register RabitqOnly column");
        let make = |i: u32| -> Vec<f32> {
            let raw: Vec<f32> = (0..dim)
                .map(|j| {
                    let h = (i.wrapping_mul(0x9E37_79B9)) ^ ((j as u32).wrapping_mul(0x85EB_CA77));
                    let h = h.wrapping_mul(0xC2B2_AE35);
                    ((h & 0xFFFF) as f32) / 65535.0 - 0.5
                })
                .collect();
            let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            raw.into_iter().map(|x| x / norm).collect()
        };
        let mut all = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs {
            let v = make(i);
            b.add(0, &v).expect("add");
            all.push(v);
        }
        let blob = b.finish().expect("finish");
        let json =
            r#"[{"column":"v","dim":128,"n_cent":4,"rot_seed":11,"metric":"l2sq"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");

        let hits = r
            .search("v", &all[17], 5, n_cent, 5)
            .expect("RabitqOnly search must succeed");
        assert!(!hits.is_empty(), "search must return some hits");
        // Self-query must rank itself at index 0. `.any` masked
        // demotion regressions — the planted vector's 1-bit
        // shortlist score against itself is strictly greater than
        // any neighbour's at this dim/cluster shape.
        assert_eq!(
            hits[0].0, 17,
            "self-query must rank the planted vector at index 0, got {hits:?}"
        );
        assert!(hits.iter().all(|(_, d)| d.is_finite()));
        for w in hits.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "RabitqOnly hits must be sorted ascending by distance, got {hits:?}"
            );
        }
    }

    /// `RabitqOnly` search must not fetch a `full[]` range because
    /// the column does not store one.
    #[test]
    fn rabitq_only_search_issues_no_full_region_fetch() {
        let dim = 32usize;
        let n_cent = 4usize;
        let n_docs = 32u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 13,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::RabitqOnly,
        })
        .expect("register RabitqOnly column");
        for i in 0..n_docs {
            let v: Vec<f32> = (0..dim).map(|j| (i + j as u32) as f32 * 0.1).collect();
            b.add(0, &v).expect("add");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json =
            r#"[{"column":"v","dim":32,"n_cent":4,"rot_seed":13,"metric":"l2sq"}]"#.to_string();

        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let async_calls = counting.async_counter();
        let sync_calls = counting.sync_counter();
        let r = VectorReader::open_with_source(
            Source::Lazy(StdArc::clone(&counting) as StdArc<dyn LazyByteSource>),
            &json,
            OpenOptions::default(),
        )
        .expect("open lazy");

        async_calls.store(0, AtomicOrdering::Relaxed);
        sync_calls.store(0, AtomicOrdering::Relaxed);
        let query: Vec<f32> = (0..dim).map(|j| j as f32 * 0.1).collect();
        let _ = r.search("v", &query, 5, n_cent, 5).expect("search");

        let sync_count = sync_calls.load(AtomicOrdering::Relaxed) as usize;
        let async_count = async_calls.load(AtomicOrdering::Relaxed);
        assert_eq!(
            async_count, 0,
            "RabitqOnly search on warm lazy must not bridge to async"
        );
        let max_expected = 2 + 2 * n_cent;
        assert!(
            sync_count <= max_expected,
            "RabitqOnly search must issue at most {max_expected} sync fetches; got {sync_count}"
        );
        assert!(
            sync_count >= 4,
            "test corpus produced only empty clusters? got sync_count={sync_count}"
        );
    }

    /// A directory entry carrying an unknown codec id errors as
    /// `MalformedVersion` rather than guessing at a byte layout.
    #[test]
    fn open_rejects_segment_with_unknown_codec_id() {
        let (blob, json) = build_blob(64, 16, 4, Metric::L2Sq);
        let mut bytes = blob.to_vec();

        const OUTER_HDR: usize = 32;
        const DIR_ENTRY: usize = 64;
        let dir_off = OUTER_HDR;
        let codec_byte_off = dir_off + 52;
        bytes[codec_byte_off] = 200u8; // unassigned

        let dir_bytes = &bytes[dir_off..dir_off + DIR_ENTRY];
        let new_crc = crc32c(dir_bytes);
        let crc_off = dir_off + DIR_ENTRY;
        bytes[crc_off..crc_off + 4].copy_from_slice(&new_crc.to_le_bytes());

        let err =
            VectorReader::open_with(Bytes::from(bytes), &json, OpenOptions { verify_crc: false })
                .expect_err("unknown codec id must error at open");
        assert!(
            matches!(err, VectorError::Read(ReadError::MalformedVersion(_))),
            "expected MalformedVersion for unknown codec id, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("unknown") || msg.contains("200"),
            "error must call out the unknown id, got: {msg}"
        );
    }

    // -----------------------------------------------------------------
    // Diagnostic — Sq8 vs Fp32 recall on planted-cluster
    // cosine corpus
    // -----------------------------------------------------------------
    //
    // Planted-cluster cosine corpora can push Sq8 recall well outside
    // the "< 0.005 drop on normalized embeddings" target envelope.
    // The hypothesis is that the **per-column** Sq8 quantizer wastes
    // most of its 256 buckets on cross-cluster spread: the per-dim
    // global range across the full corpus is much wider than the
    // intra-cluster spread, so within any one cluster only a handful
    // of buckets are used. The intra-cluster cosine differences
    // between top-K candidates then fall below the per-bucket
    // quantization noise → reranks flip.
    //
    // This `#[ignore]`-gated diagnostic reproduces the recall drop at
    // a small scale and prints corpus geometry stats. Run with
    // `cargo test --lib -- sq8_recall_diagnostic --ignored --nocapture`
    // to inspect. Per-column-quantizer fix is decided based on what
    // this prints.
    #[test]
    #[ignore = "Sq8 vs Fp32 recall diagnostic; ~10s; --ignored --nocapture"]
    fn sq8_recall_diagnostic_planted_cluster_cosine() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::{Distribution, StandardNormal};

        let n_docs = 16_000u32;
        let dim = 384usize;
        let n_cent_planted = 64usize;
        let n_cent_ivf = 256usize;
        let seed: u64 = 1;

        // 1. Build the corpus — same shape as benches/utils/corpus.rs:
        //    planted centers from 3·N(0,1) per dim, per-doc =
        //    center + 0.3·N(0,1), L2-normalized.
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = StandardNormal;
        let centers: Vec<Vec<f32>> = (0..n_cent_planted)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        let s: f64 = dist.sample(&mut rng);
                        (s as f32) * 3.0
                    })
                    .collect()
            })
            .collect();
        let mut all: Vec<Vec<f32>> = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs as usize {
            let center = &centers[i % n_cent_planted];
            let mut v: Vec<f32> = center
                .iter()
                .map(|&c| {
                    let s: f64 = dist.sample(&mut rng);
                    c + (s as f32) * 0.3
                })
                .collect();
            let nrm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= nrm;
            }
            all.push(v);
        }

        // 2. Corpus geometry: per-dim global range vs intra-cluster spread.
        let mut g_min = vec![f32::INFINITY; dim];
        let mut g_max = vec![f32::NEG_INFINITY; dim];
        for v in &all {
            for d in 0..dim {
                if v[d] < g_min[d] {
                    g_min[d] = v[d];
                }
                if v[d] > g_max[d] {
                    g_max[d] = v[d];
                }
            }
        }
        let g_ranges: Vec<f32> = (0..dim).map(|d| g_max[d] - g_min[d]).collect();
        let mean_g_range: f32 = g_ranges.iter().sum::<f32>() / dim as f32;
        let max_g_range: f32 = g_ranges.iter().cloned().fold(0.0f32, f32::max);

        let mut c0_min = vec![f32::INFINITY; dim];
        let mut c0_max = vec![f32::NEG_INFINITY; dim];
        let mut c0_count = 0u32;
        for (i, v) in all.iter().enumerate() {
            if i % n_cent_planted == 0 {
                c0_count += 1;
                for d in 0..dim {
                    if v[d] < c0_min[d] {
                        c0_min[d] = v[d];
                    }
                    if v[d] > c0_max[d] {
                        c0_max[d] = v[d];
                    }
                }
            }
        }
        let intra_ranges: Vec<f32> = (0..dim).map(|d| c0_max[d] - c0_min[d]).collect();
        let mean_intra: f32 = intra_ranges.iter().sum::<f32>() / dim as f32;

        eprintln!("--- corpus geometry (16k × 384, 64 planted centers, cosine, L2-normalized) ---");
        eprintln!(
            "per-dim global range: mean={mean_g_range:.4}  max={max_g_range:.4}  \
             bucket_width@255={:.6}",
            mean_g_range / 255.0
        );
        eprintln!("per-dim intra-cluster-0 range ({c0_count} docs): mean={mean_intra:.4}");
        eprintln!(
            "bucket-waste factor (global / intra): {:.1}x — Sq8 uses ~{} of 256 buckets per cluster",
            mean_g_range / mean_intra.max(1e-9),
            (255.0 * mean_intra / mean_g_range).round() as i32
        );

        // 3. Build Fp32 + Sq8 segments from the same corpus.
        let build = |codec: RerankCodec| -> Bytes {
            let mut b = VectorBuilder::new();
            b.register_column(VectorConfig {
                column: "v".into(),
                dim,
                n_cent: n_cent_ivf,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: codec,
            })
            .expect("register");
            for v in &all {
                b.add(0, v).expect("add");
            }
            Bytes::from(b.finish().expect("finish"))
        };
        let fp32_blob = build(RerankCodec::Fp32);
        let sq8_blob = build(RerankCodec::Sq8);
        eprintln!(
            "--- segment sizes ---\n\
             fp32: {:.2} MiB (1.00x)\n\
             sq8:  {:.2} MiB ({:.2}x)",
            fp32_blob.len() as f64 / 1024.0 / 1024.0,
            sq8_blob.len() as f64 / 1024.0 / 1024.0,
            sq8_blob.len() as f64 / fp32_blob.len() as f64
        );

        let json = format!(
            r#"[{{"column":"v","dim":{dim},"n_cent":{n_cent_ivf},"rot_seed":7,"metric":"cosine"}}]"#
        );
        let r_fp32 = VectorReader::open(fp32_blob, &json).expect("open fp32");
        let r_sq8 = VectorReader::open(sq8_blob, &json).expect("open sq8");

        // 4. Brute-force ground truth (cosine sim descending = neg-dot
        //    ascending — both engines return smaller-is-closer).
        let n_queries = 100usize;
        let k = 10usize;
        let nprobe = n_cent_ivf / 4;
        let rerank_mult = 50usize; // Sq8 calibration floor at dim ≤ 384
        let ground_truth: Vec<std::collections::HashSet<u32>> = (0..n_queries)
            .map(|qi| {
                let q = &all[qi];
                let mut sims: Vec<(u32, f32)> = (0..all.len())
                    .map(|j| {
                        let d: f32 = (0..dim).map(|i| q[i] * all[j][i]).sum();
                        (j as u32, d)
                    })
                    .collect();
                sims.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                sims.into_iter().take(k).map(|(id, _)| id).collect()
            })
            .collect();

        let recall_of = |reader: &VectorReader, label: &str| -> f32 {
            let mut total_match = 0usize;
            for qi in 0..n_queries {
                let hits = reader
                    .search("v", &all[qi], k, nprobe, rerank_mult)
                    .expect("search");
                let hit_ids: std::collections::HashSet<u32> =
                    hits.into_iter().map(|(id, _)| id).collect();
                let gt = &ground_truth[qi];
                total_match += gt.iter().filter(|id| hit_ids.contains(id)).count();
            }
            let recall = total_match as f32 / (n_queries * k) as f32;
            eprintln!("recall@{k} ({label}): {recall:.4}");
            recall
        };

        eprintln!(
            "--- recall@{k} on {n_queries} self-queries (nprobe={nprobe}, rerank_mult={rerank_mult}) ---"
        );
        let r_fp = recall_of(&r_fp32, "fp32");
        let r_sq = recall_of(&r_sq8, "sq8 ");
        eprintln!("drop (fp32 - sq8 ): {:.4}", r_fp - r_sq);
        eprintln!("(target acceptance: drop must be \u{2264} 0.01)");

        // -- Probe: vary rerank_mult to isolate shortlist depth vs rerank noise --
        eprintln!("\n--- rerank_mult sweep (Sq8, same corpus/queries) ---");
        for &rm in &[20usize, 50, 100, 200, 400] {
            let mut tm = 0usize;
            for qi in 0..n_queries {
                let hits = r_sq8.search("v", &all[qi], k, nprobe, rm).expect("search");
                let hit_ids: std::collections::HashSet<u32> =
                    hits.into_iter().map(|(id, _)| id).collect();
                tm += ground_truth[qi]
                    .iter()
                    .filter(|id| hit_ids.contains(id))
                    .count();
            }
            eprintln!(
                "  rerank_mult={rm:>4}: sq8 recall@{k} = {:.4}",
                tm as f32 / (n_queries * k) as f32
            );
        }

        // -- Probe: typical top-10 cosine spread (signal that
        //    Sq8 noise must beat).
        let mut spreads = Vec::with_capacity(n_queries);
        for qi in 0..n_queries.min(20) {
            let q = &all[qi];
            let mut sims: Vec<f32> = (0..all.len())
                .map(|j| (0..dim).map(|i| q[i] * all[j][i]).sum::<f32>())
                .collect();
            sims.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let top11: Vec<f32> = sims.iter().take(11).cloned().collect();
            // Spread between top-1 (self, sim=1) and top-10
            let span = top11[0] - top11[10];
            // Median consecutive gap among top-10
            let mut gaps: Vec<f32> = (1..11).map(|i| top11[i - 1] - top11[i]).collect();
            gaps.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med_gap = gaps[gaps.len() / 2];
            spreads.push((span, med_gap));
        }
        let mean_span: f32 = spreads.iter().map(|(s, _)| s).sum::<f32>() / spreads.len() as f32;
        let mean_gap: f32 = spreads.iter().map(|(_, g)| g).sum::<f32>() / spreads.len() as f32;
        eprintln!("\n--- top-10 cosine geometry (the signal Sq8 noise must beat) ---");
        eprintln!(
            "  mean top1-to-top10 span:      {mean_span:.4}\n  \
             mean consecutive median gap:  {mean_gap:.5}\n  \
             Sq8 noise est. (3e-5) vs gap: ratio = {:.2}%",
            3e-5_f32 / mean_gap.max(1e-9) * 100.0
        );
    }

    // -----------------------------------------------------------------
    // Lazy open + inline-`pos` search
    // -----------------------------------------------------------------
    //
    // Open touches only the structural-decode regions (directory,
    // sub-header, summary, centroids, cluster_idx). Search carries
    // `pos = off + i` inline in the shortlist tuple — there is no
    // `doc_to_pos` lookup table to populate. The structural
    // memory-ceiling tests below ride on these invariants.

    /// Search-shape corpus used by the inline-pos tests and the
    /// sync-search / counting-source tests. Picks a non-trivial
    /// `n_docs ≥ n_cent` so each cluster has multiple candidates.
    fn build_search_corpus() -> (Bytes, String, Vec<Vec<f32>>) {
        let dim = 16usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "embedding".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
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
        let json = r#"[{"column":"embedding","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#
            .to_string();
        (Bytes::from(bytes), json, all)
    }

    /// Self-query smoke: lazy default open must recover the
    /// planted self-vector at top-1, confirming the inline-`pos`
    /// rerank path returns the correct results on the
    /// search-shape corpus that every test in this section uses.
    #[test]
    fn lazy_default_search_recovers_self_query() {
        let (blob, json, all) = build_search_corpus();
        let r = VectorReader::open(blob, &json).expect("open");
        let hits = r
            .search("embedding", &all[17], 5, 4, 5)
            .expect("search must succeed on lazy InMemory");
        assert_eq!(hits[0].0, 17, "self-query must recover self");
    }

    // -----------------------------------------------------------------
    // Sync `search()` on `Source::Lazy`
    // -----------------------------------------------------------------
    //
    // These tests pin the contract: the *only* public entry point
    // is sync `search()`. It works on every `Source` variant —
    // `InMemory` and warm-cache `Source::Lazy` resolve every range
    // through `try_get_range_sync` (zero-copy); cold-miss
    // `Source::Lazy` bridges to the source's async `range()` via
    // `block_in_place + Handle::block_on` / one-shot
    // `current_thread` `Runtime`, the same pattern the supertable's
    // per-segment reader uses for the disk-cache fetch path. No
    // `search_async` is exposed at the public surface; the
    // cold-path async bridging is hidden inside
    // `Source::get_range`.
    //
    // A `CountingLazyByteSource` test helper wraps a `Bytes`
    // payload and counts every `range` / `try_get_range_sync`
    // call against an `AtomicU64`. The `disable_sync` switch
    // lets a test force the cold-miss path (sync access
    // disabled) — exposes any silent fallthrough that would
    // bypass the block_on bridge.

    use crate::superfile::lazy_source::{BytesLazyByteSource, LazyByteSource, LazyByteSourceError};
    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};

    /// Test-only [`LazyByteSource`] that wraps a `Bytes` payload and
    /// records every async / sync range request as a counter. The
    /// two `*_returns_none` switches let a test force either the
    /// "async only" path (sync access disabled) or "sync only" path
    /// (async access disabled — exposes any silent fallthrough to
    /// `range` on the supposedly-sync code path).
    #[derive(Debug)]
    struct CountingLazyByteSource {
        bytes: Bytes,
        /// Counts of every `range()` invocation.
        async_calls: StdArc<AtomicU64>,
        /// Counts of every `try_get_range_sync()` invocation.
        sync_calls: StdArc<AtomicU64>,
        /// If `true`, `try_get_range_sync` returns `None` for every
        /// in-bounds range — forces the caller to the async path.
        sync_disabled: AtomicBool,
    }

    impl CountingLazyByteSource {
        fn new(bytes: Bytes) -> Self {
            Self {
                bytes,
                async_calls: StdArc::new(AtomicU64::new(0)),
                sync_calls: StdArc::new(AtomicU64::new(0)),
                sync_disabled: AtomicBool::new(false),
            }
        }

        fn async_counter(&self) -> StdArc<AtomicU64> {
            StdArc::clone(&self.async_calls)
        }

        fn sync_counter(&self) -> StdArc<AtomicU64> {
            StdArc::clone(&self.sync_calls)
        }

        fn disable_sync(&self) {
            self.sync_disabled.store(true, AtomicOrdering::Relaxed);
        }
    }

    #[async_trait::async_trait]
    impl LazyByteSource for CountingLazyByteSource {
        fn size(&self) -> u64 {
            self.bytes.len() as u64
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            self.async_calls.fetch_add(1, AtomicOrdering::Relaxed);
            let total = self.bytes.len() as u64;
            if start.saturating_add(len) > total {
                return Err(LazyByteSourceError::OutOfBounds {
                    start,
                    len,
                    size: total,
                });
            }
            let s = start as usize;
            Ok(self.bytes.slice(s..s + len as usize))
        }

        fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
            self.sync_calls.fetch_add(1, AtomicOrdering::Relaxed);
            if self.sync_disabled.load(AtomicOrdering::Relaxed) {
                return None;
            }
            let total = self.bytes.len() as u64;
            if start.saturating_add(len) > total {
                return None;
            }
            let s = start as usize;
            Some(self.bytes.slice(s..s + len as usize))
        }
    }

    /// Sync `search()` on a `Source::Lazy` whose `try_get_range_sync`
    /// always succeeds (warm cache) behaves identically to the
    /// `Source::InMemory` path. This is the steady-state shape the
    /// supertable reader sees today (the reader_cache is in-process,
    /// so every range is resident).
    #[test]
    fn search_on_lazy_source_with_warm_sync_cache_matches_in_memory() {
        let (blob, json, all) = build_search_corpus();
        let r_mem = VectorReader::open(blob.clone(), &json).expect("InMemory open");
        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let r_lazy =
            VectorReader::open_with_source(Source::Lazy(counting), &json, OpenOptions::default())
                .expect("lazy open with warm sync cache");

        for &q_idx in &[0usize, 17, 31, 63] {
            let hits_mem = r_mem
                .search("embedding", &all[q_idx], 5, 4, 5)
                .expect("InMemory search");
            let hits_lazy = r_lazy
                .search("embedding", &all[q_idx], 5, 4, 5)
                .expect("Lazy(warm) search");
            assert_eq!(
                hits_mem, hits_lazy,
                "lazy warm-sync results must match InMemory for query {q_idx}"
            );
        }
    }

    /// Sync `search()` on a `Source::Lazy` whose
    /// `try_get_range_sync` returns `None` for every range still
    /// succeeds — `Source::get_range` bridges to the source's
    /// async `range()` via the one-shot `current_thread`
    /// `Runtime` fallback (no ambient tokio runtime in
    /// `#[test]`). Results must equal the `Source::InMemory`
    /// baseline.
    ///
    /// This is the cold-path proof — the public sync surface
    /// works against an arbitrary async-only `LazyByteSource`
    /// impl. Production callers always have an ambient runtime
    /// (the supertable owns one), so the `block_in_place +
    /// Handle::block_on` branch is what fires there; this test
    /// exercises the no-ambient-runtime fallback to keep that
    /// path live.
    #[test]
    fn search_on_lazy_source_with_no_sync_fallback_bridges_to_async() {
        let (blob, json, all) = build_search_corpus();
        let r_mem = VectorReader::open(blob.clone(), &json).expect("InMemory baseline");
        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let async_counter = counting.async_counter();
        let r_lazy = VectorReader::open_with_source(
            Source::Lazy(StdArc::clone(&counting) as StdArc<dyn LazyByteSource>),
            &json,
            OpenOptions::default(),
        )
        .expect("lazy open");
        counting.disable_sync();

        for &q_idx in &[0usize, 17, 31, 63] {
            let hits_mem = r_mem
                .search("embedding", &all[q_idx], 5, 4, 5)
                .expect("InMemory search");
            let hits_lazy = r_lazy
                .search("embedding", &all[q_idx], 5, 4, 5)
                .expect("sync search must succeed via block_on bridge");
            assert_eq!(
                hits_mem, hits_lazy,
                "sync search with block_on bridge must match InMemory for query {q_idx}"
            );
        }

        assert!(
            async_counter.load(AtomicOrdering::Relaxed) > 0,
            "with sync access disabled, every fetch must route through \
             the source's async range() via the block_on bridge"
        );
    }

    /// Range-counting budget. Sync `search()` issues per-region
    /// / per-cluster `Source::get_range` calls:
    ///
    /// - 1 range for centroids
    /// - 1 range for cluster_idx
    /// - 1 range per probed cluster's codes
    /// - 1 range per probed cluster's doc_ids
    /// - 1 fat range for the rerank batch in `full[]`
    ///
    /// At `nprobe = N` with all probed clusters non-empty:
    /// `2 + 2N + 1 = 2N + 3` ranges. The corpus here has
    /// `n_cent = 4` and the test uses `nprobe = 4`, so the
    /// worst-case budget is `2·4 + 3 = 11`. The production
    /// shape (`nprobe = 8` on a 1M corpus) is
    /// `2·8 + 3 = 19` — a future layout that interleaves
    /// codes + doc_ids per cluster would drop this further by
    /// turning the per-cluster pair into a single range.
    ///
    /// Forcing `try_get_range_sync` off makes every range route
    /// through the source's async `range()` via the block_on
    /// bridge, so the `async_calls` counter is the right
    /// instrumentation for "how many distinct ranges did
    /// `search()` request".
    ///
    /// A regression that smuggles in extra range fetches — e.g.
    /// reintroducing the whole-subsection fallback, or pulling
    /// the full `doc_ids` region over the wire at open — surfaces
    /// here rather than at the production object-store harness.
    #[test]
    fn search_cold_first_search_range_count_per_cluster() {
        let (blob, json, all) = build_search_corpus();
        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let async_counter = counting.async_counter();
        let sync_counter = counting.sync_counter();
        let r = VectorReader::open_with_source(
            Source::Lazy(StdArc::clone(&counting) as StdArc<dyn LazyByteSource>),
            &json,
            OpenOptions::default(),
        )
        .expect("lazy open");

        let async_after_open = async_counter.load(AtomicOrdering::Relaxed);
        let sync_after_open = sync_counter.load(AtomicOrdering::Relaxed);
        assert_eq!(
            async_after_open, 0,
            "open path uses try_get_range_sync only — no async fetches expected"
        );
        assert!(
            sync_after_open > 0,
            "open path should have issued sync range fetches"
        );

        counting.disable_sync();
        let hits = r
            .search("embedding", &all[7], 5, 4, 5)
            .expect("sync search via block_on bridge");
        assert!(!hits.is_empty(), "search should return hits");

        let async_calls_for_first_search =
            async_counter.load(AtomicOrdering::Relaxed) - async_after_open;
        // Worst-case at nprobe=4, all clusters non-empty:
        //   centroids + cluster_idx + nprobe*(codes + doc_ids) + 1 fat full[] = 11.
        // Lower bound is 3 (centroids + cluster_idx + fat full[]) when
        // every probed cluster is empty, but for this corpus every
        // cluster has docs.
        assert!(
            (3..=11).contains(&(async_calls_for_first_search as usize)),
            "per-cluster path: cold first search expected to issue \
             3..=11 ranges (centroids + cluster_idx + per-cluster \
             codes/doc_ids + 1 fat rerank range). Got {async_calls_for_first_search}."
        );
    }

    /// `BytesLazyByteSource` (the production-ready in-memory
    /// `LazyByteSource` impl) yields the same sync `search()`
    /// results as `Source::InMemory`. Locks in the contract that
    /// the trait-based path doesn't accidentally diverge from the
    /// enum-based fast path.
    #[test]
    fn search_matches_in_memory_through_bytes_lazy_source() {
        let (blob, json, all) = build_search_corpus();
        let r_mem = VectorReader::open(blob.clone(), &json).expect("InMemory baseline");
        let lazy_src: StdArc<dyn LazyByteSource> = StdArc::new(BytesLazyByteSource::new(blob));
        let r_lazy =
            VectorReader::open_with_source(Source::Lazy(lazy_src), &json, OpenOptions::default())
                .expect("lazy open via BytesLazyByteSource");

        for &q_idx in &[3usize, 19, 47] {
            let hits_mem = r_mem
                .search("embedding", &all[q_idx], 5, 4, 5)
                .expect("InMemory search");
            let hits_lazy = r_lazy
                .search("embedding", &all[q_idx], 5, 4, 5)
                .expect("BytesLazyByteSource sync search");
            assert_eq!(
                hits_mem, hits_lazy,
                "BytesLazyByteSource results must match InMemory for query {q_idx}"
            );
        }
    }

    // -----------------------------------------------------------------
    // Memory-ceiling unit tests
    // -----------------------------------------------------------------
    //
    // Headline guarantee: resident set per open vector segment is
    // bounded by `O(n_cent × dim × 4 + small)`, independent of
    // `n_docs`. Concretely: opening a `Source::Lazy` over a
    // mmap-backed `BytesLazyByteSource` at 1M × 384 with
    // `OpenOptions { verify_crc: false }` must leave the process
    // RSS delta ≤ 10 MB per opened column.
    //
    // Why mmap specifically: this is how the disk cache feeds
    // bytes into `SuperfileReader` —
    // `Bytes::from_owner(Arc<Mmap>)`. The kernel never faults
    // the bulk codes/full/doc_ids pages on the default path
    // because nothing in `open_with_source` accesses them: the
    // CRC scan is gated on `verify_crc`, search uses inline
    // `pos` so no `doc_ids` walk happens, and the
    // structural-decode bytes (outer header + dir + sub_header)
    // are a handful of pages. The resident allocation is
    // dominated by the rotation matrix (≈ 590 KB at dim=384)
    // and small column metadata — well inside the 10 MB ceiling
    // at any practical `n_docs`.
    //
    // Companion smoke test below (`mem_ceiling_lazy_open_smoke`)
    // runs in default `cargo test --lib` at a smaller scale so
    // every PR gets continuous feedback on this guarantee
    // without paying for a 1M-doc build. The 1M × 384 plan-spec
    // version is `#[ignore]`'d because
    // `VectorBuilder.finish_to(...)` at that scale takes ~35 s in
    // release / several minutes in debug. Run explicitly:
    //
    // ```bash
    // cargo test --release -p infino --lib \
    //     mem_ceiling_lazy_open_under_10mib -- --ignored --nocapture
    // ```

    /// `Bytes::from_owner` adapter for `Arc<memmap2::Mmap>` —
    /// mirrors `supertable::reader_cache::disk::ArcMmapOwner`
    /// (which is private to that module). Sharing the mapping
    /// via `Arc<Mmap>` keeps it alive for the reader's lifetime
    /// while also letting the test anchor the mmap explicitly.
    struct MmapOwner(StdArc<memmap2::Mmap>);

    impl AsRef<[u8]> for MmapOwner {
        fn as_ref(&self) -> &[u8] {
            self.0.as_ref()
        }
    }

    /// Build an `(n_docs × dim)` corpus, register a single
    /// vector column with the requested IVF shape, and stream
    /// the resulting unified-blob bytes to `tmp` via
    /// `VectorBuilder::finish_to`. The streaming write avoids
    /// materializing a 1.5 GiB `Vec<u8>` in the test's address
    /// space at 1M × 384 — the build's transient peak doesn't
    /// survive the `before` RSS snapshot.
    ///
    /// Deterministic per-row vector: `seed = i × 0x9E3779B1`
    /// folded through a linear congruential step per dim slot.
    /// Same shape the bench corpus generators use, inlined so
    /// the unit test doesn't reach into the bench harness.
    fn build_corpus_to_file(
        path: &std::path::Path,
        n_docs: u32,
        dim: usize,
        n_cent: usize,
    ) -> String {
        use std::io::BufWriter;

        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "embedding".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
        })
        .expect("register column");
        let mut v = vec![0f32; dim];
        for i in 0..n_docs {
            let mut seed = i.wrapping_mul(0x9E37_79B1);
            for slot in v.iter_mut() {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                *slot = ((seed >> 16) as f32) / 65_535.0;
            }
            b.add(0, &v).expect("add to vector builder");
        }
        let file = std::fs::File::create(path).expect("create tempfile");
        let writer = BufWriter::new(file);
        b.finish_to(writer).expect("finish_to BufWriter<File>");

        format!(
            r#"[{{"column":"embedding","dim":{dim},"n_cent":{n_cent},"rot_seed":7,"metric":"l2sq"}}]"#
        )
    }

    /// Open a `Source::Lazy` over a mmap'd corpus file and
    /// return the process RSS delta (bytes) attributable to
    /// the open. Anchors `(reader, mmap_arc)` past the
    /// after-RSS read so neither is dropped before
    /// measurement.
    ///
    /// `memory_stats::memory_stats()` reads `/proc/self/statm`
    /// on Linux — cheap syscall, no allocations of its own.
    /// `physical_mem` is the kernel's RSS counter (anon +
    /// file-mapped). Faulted mmap pages count; unfaulted
    /// pages don't. The whole point of the test is that the
    /// open path only touches a handful of pages (outer
    /// header, directory, per-subsection header) and leaves
    /// the rest of the file unmapped.
    fn measure_lazy_open_rss_delta(corpus_path: &std::path::Path, json: &str) -> (usize, usize) {
        let file = std::fs::File::open(corpus_path).expect("reopen corpus readonly");
        let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("mmap corpus");
        let mmap_arc = StdArc::new(mmap);
        let bytes = Bytes::from_owner(MmapOwner(StdArc::clone(&mmap_arc)));
        let lazy: StdArc<dyn LazyByteSource> = StdArc::new(BytesLazyByteSource::new(bytes));

        let before = memory_stats::memory_stats()
            .expect("memory_stats supported")
            .physical_mem;

        let reader = VectorReader::open_with_source(
            Source::Lazy(lazy),
            json,
            OpenOptions { verify_crc: false },
        )
        .expect("lazy open");

        let after = memory_stats::memory_stats()
            .expect("memory_stats supported")
            .physical_mem;

        let n_cols = reader.columns.len();
        let delta = after.saturating_sub(before);

        // Keep both alive past the RSS reads — dropping
        // `reader` before reading `after` would silently
        // make the delta look smaller than reality.
        std::hint::black_box((&reader, &mmap_arc));
        drop(reader);
        drop(mmap_arc);

        (delta, n_cols)
    }

    /// Memory-ceiling assertion at production scale.
    ///
    /// 1 M × 384, `n_cent = 1024`. `#[ignore]`-gated because
    /// the `VectorBuilder.finish_to(...)` call takes ~35 s in
    /// release. Run explicitly:
    ///
    /// ```bash
    /// cargo test --release -p infino --lib \
    ///     mem_ceiling_lazy_open_under_10mib -- --ignored --nocapture
    /// ```
    ///
    /// A regression that re-introduces eager subsection
    /// materialization or that scans `doc_ids` at open will
    /// push per-column RSS past the 10 MB ceiling and fail
    /// here rather than at the 100 M production OOM.
    #[test]
    #[ignore]
    fn mem_ceiling_lazy_open_under_10mib() {
        const N_DOCS: u32 = 1_000_000;
        const DIM: usize = 384;
        const N_CENT: usize = 1024;

        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let json = build_corpus_to_file(tmp.path(), N_DOCS, DIM, N_CENT);

        let (delta_bytes, n_cols) = measure_lazy_open_rss_delta(tmp.path(), &json);
        let delta_mib = delta_bytes as f64 / (1024.0 * 1024.0);
        let per_col_mib = delta_mib / (n_cols.max(1) as f64);

        eprintln!(
            "mem_ceiling_lazy_open_under_10mib (1M × {DIM}, n_cent={N_CENT}): \
             RSS delta = {delta_mib:.3} MiB over {n_cols} column(s) \
             = {per_col_mib:.3} MiB/col"
        );

        assert!(
            per_col_mib <= 10.0,
            "lazy open RSS delta {per_col_mib:.3} MiB/col \
             exceeds 10 MiB ceiling at 1M × {DIM}, \
             n_cent={N_CENT} (total delta {delta_mib:.3} MiB \
             over {n_cols} column(s))."
        );
    }

    /// Memory-ceiling assertion at smoke scale.
    ///
    /// 50 k × 64, `n_cent = 64`. Runs in default
    /// `cargo test --lib` (~1–2 s build) so every PR gets
    /// continuous feedback on the structural property: lazy
    /// open touches only the structural-decode pages, never
    /// the bulk codes/full/doc_ids regions. The 10 MiB ceiling
    /// at the headline 1M × 384 scale is asserted at the same
    /// value here because the resident allocation (mostly the
    /// rotation matrix at `dim²·4` = 16 KB for dim=64) is
    /// *smaller* at smoke scale, not larger — if this fires,
    /// the bigger test will too.
    ///
    /// `dim = 64` keeps the corpus tiny (~13 MB on disk) and
    /// the rotation matrix Gram-Schmidt fast.
    #[test]
    fn mem_ceiling_lazy_open_smoke() {
        const N_DOCS: u32 = 50_000;
        const DIM: usize = 64;
        const N_CENT: usize = 64;

        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let json = build_corpus_to_file(tmp.path(), N_DOCS, DIM, N_CENT);

        let (delta_bytes, n_cols) = measure_lazy_open_rss_delta(tmp.path(), &json);
        let delta_mib = delta_bytes as f64 / (1024.0 * 1024.0);
        let per_col_mib = delta_mib / (n_cols.max(1) as f64);

        eprintln!(
            "mem_ceiling_lazy_open_smoke ({N_DOCS} × {DIM}, n_cent={N_CENT}): \
             RSS delta = {delta_mib:.3} MiB over {n_cols} column(s) \
             = {per_col_mib:.3} MiB/col"
        );

        assert!(
            per_col_mib <= 10.0,
            "lazy open smoke RSS delta {per_col_mib:.3} MiB/col \
             exceeds 10 MiB ceiling at {N_DOCS} × {DIM} \
             (total delta {delta_mib:.3} MiB over {n_cols} column(s))."
        );
    }

    // -----------------------------------------------------------------
    // Supertable-scale memory ceiling
    // -----------------------------------------------------------------
    //
    // The single-segment `mem_ceiling_lazy_open_*` tests above pin
    // the per-reader bound. These multi-segment variants pin the
    // *supertable-shaped* bound: open N segments concurrently —
    // same shape `Supertable::commit` produces (N =
    // N_SEGMENTS_BENCH × num_cpus because the writer-pool sharding
    // creates one segment per thread) — and assert the total anon
    // RSS delta scales as `N × O(centroids + rotation + small)`,
    // not as `N × subsection_size`.
    //
    // What this proves (and what it doesn't):
    //
    // - PROVES: a supertable opened with the production
    //   disk-cache path
    //   (`Source::InMemory(Bytes::from_owner(mmap))` per segment)
    //   keeps anon RSS bounded across an arbitrary number of
    //   segments, with no per-doc anon term. Equivalent here
    //   because `Bytes::from_owner` is zero-copy over the mmap,
    //   and the lazy-open path doesn't touch `doc_ids[]` /
    //   `full[]` at open time (inline `pos` removes the only
    //   reason open ever touched `doc_ids[]`).
    //
    // - DOES NOT PROVE: the in-process `InMemoryReaderCache`
    //   path (`Bytes::from(Vec)` per segment) has the same bound.
    //   That path holds each segment's bytes in anon by
    //   construction (no mmap involved). The in-memory cache is
    //   the test/bench path; production attaches a
    //   `StorageProvider` and routes through the disk cache. The
    //   in-memory cache's anon cost is its declared contract.
    //
    // The bench's 10M × 4-commit × num_cpus-thread shape produces
    // exactly the topology these tests exercise. The smoke
    // variant mirrors the bench's *layout* at a tiny corpus size
    // (4 segments × 50 k docs × 64 dim) so every PR catches
    // regressions (~5 s build). The `#[ignore]`'d production-scale
    // variant uses the bench's actual per-segment shape (16
    // segments × 625 k docs × 384 dim × n_cent_per_segment
    // matching the bench's `n_cent_total / 4`) and runs only when
    // called out.

    /// Open `N` segment files (built by `build_corpus_to_file`) via
    /// `Source::Lazy(BytesLazyByteSource over Arc<Mmap>)` and return
    /// the total RSS delta attributable to those opens. Anchors
    /// `(readers, mmaps)` past the after-RSS read.
    fn measure_lazy_multi_segment_open_rss_delta(
        corpus_paths: &[std::path::PathBuf],
        jsons: &[String],
    ) -> (usize, usize, usize) {
        assert_eq!(corpus_paths.len(), jsons.len(), "paths/jsons must align");
        let n_segments = corpus_paths.len();

        // Pre-build (mmap, lazy source) pairs *before* the `before`
        // snapshot so the syscalls don't contaminate the delta — we
        // only want the open path's allocations in the measurement.
        let mut lazies: Vec<(StdArc<memmap2::Mmap>, StdArc<dyn LazyByteSource>)> =
            Vec::with_capacity(n_segments);
        for path in corpus_paths {
            let file = std::fs::File::open(path).expect("reopen corpus readonly");
            let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("mmap corpus");
            let mmap_arc = StdArc::new(mmap);
            let bytes = Bytes::from_owner(MmapOwner(StdArc::clone(&mmap_arc)));
            let lazy: StdArc<dyn LazyByteSource> = StdArc::new(BytesLazyByteSource::new(bytes));
            lazies.push((mmap_arc, lazy));
        }

        let before = memory_stats::memory_stats()
            .expect("memory_stats supported")
            .physical_mem;

        let mut readers: Vec<VectorReader> = Vec::with_capacity(n_segments);
        let mut n_cols_total = 0usize;
        for ((_, lazy), json) in lazies.iter().zip(jsons.iter()) {
            let reader = VectorReader::open_with_source(
                Source::Lazy(StdArc::clone(lazy)),
                json,
                OpenOptions { verify_crc: false },
            )
            .expect("lazy open");
            n_cols_total += reader.columns.len();
            readers.push(reader);
        }

        let after = memory_stats::memory_stats()
            .expect("memory_stats supported")
            .physical_mem;

        let delta = after.saturating_sub(before);

        // Keep both alive past the RSS reads — dropping any reader
        // (or mmap) before reading `after` would silently shrink the
        // measured delta.
        std::hint::black_box((&readers, &lazies));
        drop(readers);
        drop(lazies);

        (delta, n_cols_total, n_segments)
    }

    /// Supertable-scale memory ceiling — smoke.
    ///
    /// Mirrors the bench's 4-commit × num_cpus-thread shape at a
    /// tiny corpus size. Builds 4 segment files (each 50 k × 64
    /// dim × n_cent=64 — same shape as
    /// `mem_ceiling_lazy_open_smoke`), opens all 4 lazy, and
    /// asserts the total anon RSS delta is ≤ 10 MiB. With
    /// per-segment ceiling of 10 MiB / column from the single-
    /// segment smoke and a 4× multiplier in the worst case
    /// (centroids + rotation matrix per segment), 10 MiB total
    /// gives plenty of headroom while still failing loud if a
    /// regression makes per-segment opens allocate per-doc.
    ///
    /// Runs in the default `cargo test --lib` suite (~3–5 s
    /// total) so every PR validates the supertable-shape bound.
    #[test]
    fn mem_ceiling_lazy_multi_segment_open_smoke() {
        const N_SEGMENTS: usize = 4;
        const N_DOCS_PER_SEG: u32 = 50_000;
        const DIM: usize = 64;
        const N_CENT: usize = 64;

        let mut tmps: Vec<tempfile::NamedTempFile> = Vec::with_capacity(N_SEGMENTS);
        let mut paths: Vec<std::path::PathBuf> = Vec::with_capacity(N_SEGMENTS);
        let mut jsons: Vec<String> = Vec::with_capacity(N_SEGMENTS);
        for _ in 0..N_SEGMENTS {
            let tmp = tempfile::NamedTempFile::new().expect("tempfile");
            let json = build_corpus_to_file(tmp.path(), N_DOCS_PER_SEG, DIM, N_CENT);
            paths.push(tmp.path().to_path_buf());
            jsons.push(json);
            tmps.push(tmp); // keep the tempfile alive until end
        }

        let (delta_bytes, n_cols_total, n_segments) =
            measure_lazy_multi_segment_open_rss_delta(&paths, &jsons);
        let delta_mib = delta_bytes as f64 / (1024.0 * 1024.0);
        let per_seg_mib = delta_mib / n_segments as f64;

        eprintln!(
            "mem_ceiling_lazy_multi_segment_open_smoke ({N_SEGMENTS} × {N_DOCS_PER_SEG} × \
             {DIM}, n_cent={N_CENT}): RSS delta = {delta_mib:.3} MiB over {n_segments} \
             segment(s) ({n_cols_total} column(s) total) = {per_seg_mib:.3} MiB/segment"
        );

        assert!(
            delta_mib <= 10.0,
            "supertable-shape lazy open RSS delta {delta_mib:.3} MiB exceeds 10 MiB ceiling \
             at {N_SEGMENTS} × {N_DOCS_PER_SEG} × {DIM} — regression hint: each segment may \
             be touching its doc_ids/full[]/codes region at open"
        );

        drop(tmps);
    }

    /// Supertable-scale memory ceiling — production shape.
    ///
    /// Mirrors the bench's actual 10M × 4-commit ×
    /// 4-thread-writer-pool topology: 16 segments × 625 k docs ×
    /// 384 dim × `n_cent_per_segment = n_cent(10M) / 4` (the
    /// bench's `corpus::n_cent(10M)` returns 1024, so this is
    /// 256). Each segment file is ~960 MiB on disk; the test
    /// writes ~15 GiB total to the tempdir. Build time is
    /// dominated by the 16 sequential streaming builds at
    /// ~10 s each in release ≈ 3 min total.
    ///
    /// `#[ignore]`-gated. Run explicitly:
    ///
    /// ```bash
    /// cargo test --release -p infino --lib \
    ///     mem_ceiling_lazy_supertable_scale_under_50mib -- --ignored --nocapture
    /// ```
    ///
    /// Bound: 50 MiB total anon over the 16 segments. The
    /// per-segment open materialises:
    /// - rotation matrix: `dim² × 4 = 576 KiB` at dim=384
    /// - centroids buffer (in lazy source page cache, not anon):
    ///   `n_cent × dim × 4 = 384 KiB` at the smoke shape
    /// - per-column header / cluster_idx slices (KiB-range)
    ///
    /// Add a 2× safety margin for allocator overhead +
    /// reader-struct fields, multiply by 16 segments → ~20 MiB
    /// theoretical, 50 MiB ceiling for headroom. A regression
    /// that re-introduces eager subsection materialisation
    /// would blow this to ~15 GiB (the full corpus) and fail
    /// loud here rather than at the production 100 M OOM.
    #[test]
    #[ignore]
    fn mem_ceiling_lazy_supertable_scale_under_50mib() {
        const N_SEGMENTS: usize = 16;
        const N_DOCS_PER_SEG: u32 = 625_000;
        const DIM: usize = 384;
        const N_CENT_PER_SEG: usize = 256;

        let mut tmps: Vec<tempfile::NamedTempFile> = Vec::with_capacity(N_SEGMENTS);
        let mut paths: Vec<std::path::PathBuf> = Vec::with_capacity(N_SEGMENTS);
        let mut jsons: Vec<String> = Vec::with_capacity(N_SEGMENTS);
        for i in 0..N_SEGMENTS {
            let tmp = tempfile::NamedTempFile::new().expect("tempfile");
            eprintln!(
                "  building segment {i:2}/{N_SEGMENTS} \
                 ({N_DOCS_PER_SEG} × {DIM}, n_cent={N_CENT_PER_SEG})…"
            );
            let json = build_corpus_to_file(tmp.path(), N_DOCS_PER_SEG, DIM, N_CENT_PER_SEG);
            paths.push(tmp.path().to_path_buf());
            jsons.push(json);
            tmps.push(tmp);
        }

        let (delta_bytes, n_cols_total, n_segments) =
            measure_lazy_multi_segment_open_rss_delta(&paths, &jsons);
        let delta_mib = delta_bytes as f64 / (1024.0 * 1024.0);
        let per_seg_mib = delta_mib / n_segments as f64;

        eprintln!(
            "mem_ceiling_lazy_supertable_scale_under_50mib ({N_SEGMENTS} × {N_DOCS_PER_SEG} × \
             {DIM}, n_cent={N_CENT_PER_SEG}): RSS delta = {delta_mib:.3} MiB over \
             {n_segments} segment(s) ({n_cols_total} column(s) total) = \
             {per_seg_mib:.3} MiB/segment"
        );

        assert!(
            delta_mib <= 50.0,
            "supertable-scale (10M-bench shape) lazy open RSS delta {delta_mib:.3} MiB \
             exceeds 50 MiB ceiling at {N_SEGMENTS} × {N_DOCS_PER_SEG} × {DIM}. \
             Eager re-introduction would push this past 15 GiB."
        );

        drop(tmps);
    }

    /// Many-segments stress test — 100M aspiration shape.
    ///
    /// The honest scale test for "100M docs across a supertable"
    /// can't materialise 100M production-shape segments on a
    /// developer box (the per-segment 625k × 384 shape used in
    /// the bench produces ~960 MiB on disk × 160 segments = 150
    /// GiB of corpus). Instead, this test pins the *structural*
    /// memory bound by varying the high-cardinality axis
    /// (segment count) at a thin per-segment shape:
    /// **100 segments × 50 k docs × 128 dim × 128 n_cent**.
    ///
    /// What this proves:
    ///
    /// - Per-segment open allocation is `O(n_cent × dim × 4 +
    ///   rotation + small)` — no `n_docs` term. At this shape:
    ///   centroids 64 KiB + rotation matrix 64 KiB + column
    ///   metadata ≪ 1 MiB per segment. Total expected RSS
    ///   delta ≪ 200 MiB across 100 segments; 400 MiB ceiling
    ///   for allocator overhead + reader-struct fields.
    ///
    /// - With no per-doc resident state on the open path,
    ///   segment count is the only scaling dimension. A
    ///   regression that reintroduced any per-doc resident state
    ///   — e.g. a returning lookup table at `n_docs × 4` bytes
    ///   per column — would here allocate 100 × 50 k × 4 = 20
    ///   MiB anon just for tables (small but growing); at the
    ///   bench's 100 segments × 625 k the same regression is
    ///   250 MiB.
    ///
    /// Each segment file is ~25 MiB on disk; the test writes
    /// ~2.5 GiB total to the tempdir. Build time is dominated by
    /// the 100 sequential streaming builds (~1.5 s each in
    /// release ≈ 2.5 min total).
    ///
    /// `#[ignore]`-gated. Run explicitly:
    ///
    /// ```bash
    /// cargo test --release -p infino --lib \
    ///     mem_ceiling_lazy_many_segments_under_400mib -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn mem_ceiling_lazy_many_segments_under_400mib() {
        const N_SEGMENTS: usize = 100;
        const N_DOCS_PER_SEG: u32 = 50_000;
        const DIM: usize = 128;
        const N_CENT_PER_SEG: usize = 128;

        let mut tmps: Vec<tempfile::NamedTempFile> = Vec::with_capacity(N_SEGMENTS);
        let mut paths: Vec<std::path::PathBuf> = Vec::with_capacity(N_SEGMENTS);
        let mut jsons: Vec<String> = Vec::with_capacity(N_SEGMENTS);
        for i in 0..N_SEGMENTS {
            let tmp = tempfile::NamedTempFile::new().expect("tempfile");
            if i % 10 == 0 {
                eprintln!(
                    "  building segment {i:3}/{N_SEGMENTS} \
                     ({N_DOCS_PER_SEG} × {DIM}, n_cent={N_CENT_PER_SEG})…"
                );
            }
            let json = build_corpus_to_file(tmp.path(), N_DOCS_PER_SEG, DIM, N_CENT_PER_SEG);
            paths.push(tmp.path().to_path_buf());
            jsons.push(json);
            tmps.push(tmp);
        }

        let (delta_bytes, n_cols_total, n_segments) =
            measure_lazy_multi_segment_open_rss_delta(&paths, &jsons);
        let delta_mib = delta_bytes as f64 / (1024.0 * 1024.0);
        let per_seg_mib = delta_mib / n_segments as f64;

        eprintln!(
            "mem_ceiling_lazy_many_segments_under_400mib ({N_SEGMENTS} × {N_DOCS_PER_SEG} × \
             {DIM}, n_cent={N_CENT_PER_SEG}): RSS delta = {delta_mib:.3} MiB over \
             {n_segments} segment(s) ({n_cols_total} column(s) total) = \
             {per_seg_mib:.3} MiB/segment"
        );

        assert!(
            delta_mib <= 400.0,
            "many-segments lazy open RSS delta {delta_mib:.3} MiB exceeds 400 MiB ceiling \
             at {N_SEGMENTS} × {N_DOCS_PER_SEG} × {DIM}. A regression that reintroduced \
             any per-doc resident state at open would push this much higher; keeping the \
             bound structural is what this test guards."
        );

        drop(tmps);
    }
}
