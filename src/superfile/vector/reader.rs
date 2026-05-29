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
use crate::superfile::lazy_source::{LazyByteSource, LazyByteSourceError, PrefetchedSource};
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
/// value is a JSON array of these in declaration order.
#[derive(Debug, Clone, Deserialize)]
pub struct VectorColumnConfig {
    pub column: String,
    pub dim: usize,
    pub n_cent: usize,
    pub rot_seed: u64,
    /// `"l2sq"`, `"cosine"`, or `"negdot"`.
    pub metric: String,
}

/// Plan 012 M3 + Sq8PerCluster — Sq8 quantizer state
/// materialised from the on-disk `codec_meta` region at open
/// time. The reader picks the candidate's cluster slice of
/// `scale` / `offset` and passes it into [`Sq8Kernel::new`]
/// once per (query, cluster) pair to build the per-query
/// precomputes.
///
/// Per-cluster (not per-column) quantizer: each IVF cluster
/// owns its own `(scale[dim], offset[dim])` pair, packed
/// contiguously cluster-by-cluster. Per-column was the
/// original M3 design but collapsed recall on highly clustered
/// cosine corpora (intra-cluster spread is 5–10× narrower than
/// cross-cluster, so 256 buckets stretched over the global
/// range left only ~25–50 usable buckets per cluster; ranking
/// noise dominated intra-cluster cosine differences). See
/// `RerankCodec::codec_meta_bytes` doc for the failure-mode
/// analysis. Per-cluster recovers recall at +`n_cent × dim × 8`
/// bytes of codec_meta (3 MiB at the 1M × 1024 × 384 shape).
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
    /// field). `Some` for L2Sq (short-circuits the `Σx²` term)
    /// and Cosine (normalizes the decoded vector at rerank
    /// time); `None` for NegDot, where the `Σx²` term cancels
    /// out of the distance formula.
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
    /// Plan 012 — on-disk rerank codec for this column. Today
    /// admits Fp32 (M1), Bf16 (M2), Sq8 (M3); the parser rejects
    /// every other codec at open time with a `MalformedVersion`
    /// until the corresponding milestone lands (None: M4).
    pub rerank_codec: RerankCodec,
    /// Plan 012 M3 — `Sq8`-only quantizer metadata, materialised
    /// at open time from the `codec_meta` region. `None` for
    /// every other codec (Fp32 / Bf16 / None). At dim=384 the
    /// scale + offset arrays are 3 KB total; for L2Sq columns
    /// the per-doc norms add `n_docs × 4` bytes (4 MB at 1M
    /// docs / column). Materialising here amortizes the parse
    /// across every search call.
    pub(super) sq8_meta: Option<Sq8ColumnMeta>,
    /// Byte range of this column's subsection within the outer blob.
    subsection_range: Range<usize>,
    /// Offsets relative to the subsection start.
    summary_off: usize,
    summary_radius: f32,
    centroids_off: usize,
    cluster_idx_off: usize,
    /// Plan 012 M1 — relative offset of the per-column
    /// `codec_meta` region inside the subsection. `0` means
    /// "no codec_meta" (Fp32 / Bf16 / None); non-zero is only
    /// produced by codecs whose `codec_meta_bytes(...) > 0`
    /// (`Sq8` is the only one today). In the 013 layout
    /// `codec_meta` sits between `cluster_idx` and the
    /// per-cluster blocks (inside the open-time region).
    #[allow(dead_code)]
    codec_meta_off: usize,
    /// Relative offset of the per-cluster blocks region. Each
    /// cluster `c` lives at
    /// `per_cluster_blocks_off + doc_off[c] * (code_bytes + 4)`
    /// for `count[c] * (code_bytes + 4)` bytes, formatted as
    /// `[codes_chunk: count*code_bytes][doc_ids_chunk: count*4]`.
    per_cluster_blocks_off: usize,
    full_off: usize,
    quant: BitQuantizer,
    /// Cached random rotation built once at open from `(dim, rot_seed)`.
    /// Construction is `O(dim³)` for Gram-Schmidt — at dim=384 that's
    /// ~7.9 ms, dominant over every other per-query stage if rebuilt
    /// per `search()`. Build once, reuse forever.
    rot: RandomRotation,
}

impl ColumnReader {
    /// Plan 013 M3 — byte range covering one cluster's
    /// `[codes_chunk + doc_ids_chunk]` block as a single
    /// contiguous span. Pulled in **one** range fetch per
    /// probed cluster; the cold-first-search budget collapses
    /// to `nprobe + 1` range GETs (nprobe cluster blocks + 1
    /// rerank run) on a freshly-opened lazy reader, down from
    /// `2 × nprobe + 1` on the 011-era split-range path.
    ///
    /// 013 layout: each cluster's block is
    /// `count * (code_bytes + 4)` bytes formatted as
    /// `[codes: count*code_bytes][doc_ids: count*4]`. The
    /// per-cluster `(doc_off, count)` entry recorded in
    /// `cluster_idx` addresses both halves with no extra
    /// lookup: byte offset = `per_cluster_blocks_off +
    /// doc_off * (code_bytes + 4)`.
    pub(super) fn cluster_block_range(
        &self,
        cluster_doc_off: u32,
        cluster_count: u32,
    ) -> Range<usize> {
        let sub_start = self.subsection_range.start;
        let cb = self.quant.code_bytes();
        let start = sub_start + self.per_cluster_blocks_off + (cluster_doc_off as usize) * (cb + 4);
        let len = (cluster_count as usize) * (cb + 4);
        start..start + len
    }

    /// Byte range of the full[] (rerank) vectors covering
    /// `[min_pos, max_pos]` inclusive — used by the rerank-fetch
    /// fat-range path.
    pub(super) fn rerank_run_range(
        &self,
        min_pos: u32,
        max_pos: u32,
        per_vec_bytes: usize,
    ) -> Range<usize> {
        let sub_start = self.subsection_range.start;
        let start = sub_start + self.full_off + (min_pos as usize) * per_vec_bytes;
        let end = sub_start + self.full_off + ((max_pos as usize) + 1) * per_vec_bytes;
        start..end
    }
}

/// Per-open knobs for [`VectorReader::open_with`] and
/// [`VectorReader::open_lazy`]. `Default` is the safe choice
/// (CRC verification on). The argumentless [`VectorReader::open`]
/// takes the default; the lazy path uses
/// [`Self::for_object_store`] which turns CRC off (a full-blob
/// scan would defeat every byte-budget number in plan 013).
///
/// Plan 011 added `verify_crc`. Plan 013 M2 added
/// `open_time_speculative_bytes` for the lazy path's GET 2.
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
    /// Plan 013 M2 — speculative tail length, in bytes, fetched
    /// past the directory's end on the lazy-open GET 2.
    ///
    /// The default (2 MiB) covers the open-time region for the
    /// 1M × 384 sq8 / `n_cent = 1024` single-column shape in one
    /// GET. Larger segments (≥ 10M × 1024) trigger a targeted
    /// follow-up range for the subsection-0 codec_meta / centroids
    /// tail. Multi-column segments always issue one follow-up
    /// range per additional column (subsections 2..N's open-time
    /// regions are non-contiguous with subsection 0's). Increase
    /// to tighten the cold-open round-trip count at the cost of
    /// wasted bandwidth when the speculation misses.
    pub open_time_speculative_bytes: usize,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            verify_crc: true,
            open_time_speculative_bytes: DEFAULT_OPEN_TIME_SPECULATIVE_BYTES,
        }
    }
}

impl OpenOptions {
    /// Plan 013 M2 — defaults tuned for an object-store-backed
    /// `Source::Lazy` open: `verify_crc = false` (a full-blob
    /// scan would defeat every cold-open byte-budget number in
    /// the plan; deployments that need CRC verification opt
    /// back in and accept the cost). `open_time_speculative_bytes`
    /// stays at the default.
    pub fn for_object_store() -> Self {
        Self {
            verify_crc: false,
            ..Self::default()
        }
    }
}

/// Default speculative tail length for `open_lazy`'s GET 2.
/// 2 MiB at one S3 in-region GET is ≈ 60-100 ms TTFB + transfer
/// (vs ≈ 50 ms for a small range), and it covers the open-time
/// region for the 1M × 384 sq8 single-column shape outright.
const DEFAULT_OPEN_TIME_SPECULATIVE_BYTES: usize = 6 * 1024 * 1024;

/// Backing for a [`VectorReader`]. Plan 011 M1.
///
/// Two variants today, plumbed through every byte-fetch point:
///
/// - `InMemory(Bytes)`: the legacy path — caller materialised
///   the full subsection before opening. Every byte fetch is a
///   zero-copy `Bytes::slice` against the buffer.
/// - `Lazy(Arc<dyn LazyByteSource>)`: a range-fetching source
///   (mmap, S3 range GET, broadcast subscription). M1 wires
///   the enum + every access site through it; M2 lands the
///   lazy-friendly `open_with_source` shape.
///
/// Both variants expose **sync-only** byte access matching
/// plan 002 Q9 (resolved as commit `2e351ba`) — every public
/// surface in `src/` is sync. The `LazyByteSource::range`
/// trait method is async because production impls (S3 / object
/// store) are; `Source::get_range` hides that under the same
/// `block_in_place + Handle::block_on` / one-shot
/// `current_thread` `Runtime` bridge `supertable::query::
/// segment_reader` uses for the disk-cache fetch path. Hot-
/// path callers (`Source::InMemory`, mmap-backed
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
    /// Plan 002 Q9 (commit `2e351ba`) resolved the project's
    /// sync-vs-async convention: every public surface stays
    /// sync, async is hidden behind well-defined bridge points.
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
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(s.range(start, len))),
            Err(_) => {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| {
                        LazyByteSourceError::Storage(crate::storage::StorageError::Permanent {
                            uri: "lazy-source://vector-reader".to_string(),
                            source: Box::new(std::io::Error::other(format!(
                                "tokio runtime build for lazy source fetch: {e}"
                            ))),
                        })
                    })?;
                rt.block_on(s.range(start, len))
            }
        }
    }

    /// Plan 013 M5 — concurrent multi-range fetch.
    ///
    /// Semantically equivalent to calling
    /// [`Self::get_range`] once per input range, but on
    /// `Source::Lazy` the lazy ranges fire **in parallel** under
    /// a single sync→async bridge instead of one bridge-per-call.
    /// That collapses N sequential round-trips into one batch
    /// whose wall-clock is `max(per-range RTT)` rather than
    /// `sum(per-range RTT)`. This is the hot lever for the cold
    /// first-search budget: today's `search()` issues `nprobe`
    /// per-cluster block GETs serially via [`Self::get_range`];
    /// after this lands it issues them once concurrently.
    ///
    /// In-memory and overlay-hit ranges still resolve via the
    /// sync zero-copy fast path (`try_get_range_sync`); only the
    /// genuine cold misses are dispatched through the async
    /// bridge. The output `Vec<Bytes>` preserves input order.
    ///
    /// Errors short-circuit: the first lazy fetch to fail
    /// returns its error and any sibling fetches still in flight
    /// are aborted (`futures::future::try_join_all` semantics).
    pub fn get_ranges_parallel(
        &self,
        ranges: &[Range<usize>],
    ) -> Result<Vec<Bytes>, LazyByteSourceError> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }

        // Resolve overlay/in-memory hits sync first; queue the
        // rest for one batched async dispatch.
        let mut out: Vec<Option<Bytes>> = Vec::with_capacity(ranges.len());
        let mut pending: Vec<(usize, u64, u64)> = Vec::new();
        for (i, r) in ranges.iter().enumerate() {
            if let Some(b) = self.try_get_range_sync(r.clone()) {
                out.push(Some(b));
                continue;
            }
            if !matches!(self, Self::Lazy(_)) {
                // InMemory + missed sync = out-of-bounds for this range.
                return Err(LazyByteSourceError::OutOfBounds {
                    start: r.start as u64,
                    len: r.len() as u64,
                    size: self.len() as u64,
                });
            }
            pending.push((i, r.start as u64, r.len() as u64));
            out.push(None);
        }

        if !pending.is_empty() {
            let Self::Lazy(s) = self else {
                unreachable!("pending non-empty implies Source::Lazy (sync-miss guard above)");
            };
            let src = Arc::clone(s);
            let order: Vec<usize> = pending.iter().map(|(i, _, _)| *i).collect();
            let fut = async move {
                let futs = pending
                    .into_iter()
                    .map(|(_i, start, len)| {
                        let s = Arc::clone(&src);
                        async move { s.range(start, len).await }
                    })
                    .collect::<Vec<_>>();
                futures::future::try_join_all(futs).await
            };
            let bytes: Vec<Bytes> = match tokio::runtime::Handle::try_current() {
                Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fut))?,
                Err(_) => {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| {
                            LazyByteSourceError::Storage(crate::storage::StorageError::Permanent {
                                uri: "lazy-source://vector-reader-parallel".to_string(),
                                source: Box::new(std::io::Error::other(format!(
                                    "tokio runtime build for parallel lazy fetch: {e}"
                                ))),
                            })
                        })?;
                    rt.block_on(fut)?
                }
            };
            for (slot, b) in order.into_iter().zip(bytes) {
                out[slot] = Some(b);
            }
        }

        Ok(out
            .into_iter()
            .map(|b| b.expect("every slot filled by sync or async path"))
            .collect())
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
        Self::open_with_source(Source::InMemory(blob), columns_json, opts)
    }

    /// Plan 013 M5 — async open against a [`LazyByteSource`]
    /// that drives a **1-range** cold open in the typical case.
    ///
    /// Pulls only the bytes the open path actually reads:
    ///   - GET 1 (combined): `[0..open_time_speculative_bytes]`
    ///     — outer header (32 B), directory + CRC, and the
    ///     subsection-0 open-time region in **one** round-trip.
    ///     Default 2 MiB speculation covers the
    ///     1M × 384 sq8 / `n_cent = 1024` single-column shape.
    ///     Pre-M5 this was two sequential GETs (32-byte probe
    ///     to learn `dir_offset`, then a `dir_offset`-anchored
    ///     range); folding them saves one S3 RTT on the cold
    ///     open critical path.
    ///   - GET 2 (fallback): when GET 1's speculation didn't
    ///     reach `dir_end + open_time_region`, an explicit
    ///     `dir_offset`-anchored range covers the residual
    ///     directory + speculation. Same shape as the legacy
    ///     pre-M5 GET 2.
    ///   - GET 3+: targeted follow-up range per subsection whose
    ///     open-time region overflowed the speculative tail
    ///     (large-segment subsection 0; every subsection 1..N
    ///     of a multi-column segment).
    ///
    /// All fetched ranges land in a [`PrefetchedSource`] overlay;
    /// the subsequent structural decode (via
    /// [`Self::open_with_source`]) resolves every sub-header /
    /// codec_meta read from the overlay without touching the
    /// underlying source again.
    ///
    /// `opts.verify_crc = true` is honored, but it forces every
    /// subsection to be fetched in full and defeats the cold-open
    /// byte-budget goal of plan 013 — only set it when the
    /// underlying storage is untrusted and CRC verification is
    /// load-bearing. The convenience constructor
    /// [`OpenOptions::for_object_store`] sets it to `false`
    /// (the load-bearing default discussed in the plan's
    /// "verify_crc trade-off" section).
    pub async fn open_lazy(
        source: Arc<dyn LazyByteSource>,
        columns_json: &str,
        opts: OpenOptions,
    ) -> Result<Self, VectorError> {
        let blob_size = source.size() as usize;
        if blob_size < OUTER_HEADER_SIZE + 4 {
            return Err(VectorError::Read(ReadError::MissingKv(
                "vector blob header",
            )));
        }

        // GET 1 (combined) — Plan 013 M5: single speculative
        // head GET that covers outer header + directory + the
        // typical open-time region (including codec_meta tail
        // for sq8) in **one** round-trip.
        //
        // Pre-M5 this was two sequential GETs: a 32-byte outer
        // header to learn `dir_offset`, then a `dir_offset`-
        // anchored range covering the directory + speculation.
        // The 32-byte fetch is mostly RTT — saving it shaves
        // an entire S3 round-trip off the cold-open critical
        // path (~25-50 ms per RTT in-region).
        //
        // `open_time_speculative_bytes` defaults to 6 MiB —
        // covers the sq8 / n_cent=1024 / dim=384 layout's
        // open-time region (~5 MiB end-to-end through codec_meta
        // tail) with margin for typical shapes. Larger segments
        // (oversized dir / oversized codec_meta) fall back to
        // GET 2 / GET 3+ below.
        let head_prefetch_end = opts
            .open_time_speculative_bytes
            .max(OUTER_HEADER_SIZE + 4)
            .min(blob_size);
        let head_prefetch = source
            .range(0, head_prefetch_end as u64)
            .await
            .map_err(|e| {
                VectorError::Read(ReadError::MalformedVersion(format!(
                    "lazy open: combined head prefetch: {e}"
                )))
            })?;

        let header_bytes = head_prefetch.slice(0..OUTER_HEADER_SIZE);
        if &header_bytes[0..8] != format::vec::OUTER_MAGIC {
            return Err(VectorError::Read(ReadError::BadMagic {
                section: "vector",
                expected: format::vec::OUTER_MAGIC,
                actual: header_bytes[0..8].to_vec(),
            }));
        }
        let version = read_u32_le(&header_bytes[8..12]);
        if version != format::vec::VERSION {
            return Err(VectorError::Read(ReadError::UnsupportedVersion(format!(
                "vector blob version {version}"
            ))));
        }
        let n_columns = read_u32_le(&header_bytes[12..16]) as usize;
        let dir_offset = read_u64_le(&header_bytes[24..32]) as usize;
        let dir_size = n_columns * DIR_ENTRY_SIZE;
        let dir_end = dir_offset + dir_size + 4 /* dir CRC */;
        if dir_end > blob_size {
            return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                "lazy open: directory end {dir_end} exceeds blob size {blob_size}",
            ))));
        }

        // Resolve dir + open-time bytes. Three cases ordered by
        // frequency:
        //
        //   A. head_prefetch already covers everything past the
        //      directory up to `head_prefetch_end` (the common
        //      case for single-column / small-dir segments;
        //      `head_prefetch_end` is bounded by `spec` so for
        //      large blobs we lean on the codec_meta-tail
        //      fallback below to cover whatever's past the
        //      speculation rather than re-fetching the dir).
        //   B. head_prefetch covers the dir + dir_crc but not
        //      the full open-time region. Slice what we have;
        //      per-subsection codec_meta tail GETs below cover
        //      any uncovered open-time bytes.
        //   C. head_prefetch_end < dir_end — the speculation
        //      didn't even reach the directory. This requires
        //      `spec < dir_end` (i.e. `spec` smaller than
        //      `n_columns * 64 + 36`), which is only
        //      reachable on adversarial / many-column blobs.
        //      Issue an explicit dir-anchored fallback covering
        //      `[dir_offset..dir_end + spec]` — same shape as
        //      the pre-M5 GET 2.
        //
        // No-op for the common path: `head_prefetch_end >= dir_end`
        // is the rule, so the fallback GET virtually never
        // fires. The earlier (and buggy) Plan 013 M5 logic
        // gated on `head_prefetch.len() >= dir_end + spec`,
        // which mis-fired by ~`dir_end` bytes for every
        // single-column cold open and spuriously paid an
        // extra dir-anchored GET on the critical path.
        let speculative_end = dir_end
            .saturating_add(opts.open_time_speculative_bytes)
            .min(blob_size);
        let (prefetch_bytes_start, prefetch_bytes) = if head_prefetch_end >= dir_end {
            // Cases A + B — slice from the head prefetch.
            let end = head_prefetch_end.max(dir_end);
            (dir_offset, head_prefetch.slice(dir_offset..end))
        } else {
            // Case C — dir-anchored fallback covering
            // [dir_offset..speculative_end].
            let fallback = source
                .range(dir_offset as u64, (speculative_end - dir_offset) as u64)
                .await
                .map_err(|e| {
                    VectorError::Read(ReadError::MalformedVersion(format!(
                        "lazy open: directory+open-time prefetch: {e}"
                    )))
                })?;
            (dir_offset, fallback)
        };
        // The right end of what's now covered in the overlay,
        // in absolute blob coordinates. Used below to decide
        // whether per-subsection codec_meta tail GETs need to
        // fire.
        let coverage_end = prefetch_bytes_start + prefetch_bytes.len();

        // Validate directory CRC against the prefetched bytes
        // before walking subsection metadata. A directory-CRC
        // mismatch on the lazy path is the closest thing we
        // have to an end-to-end integrity check when
        // `verify_crc = false`.
        let dir_bytes_slice = &prefetch_bytes[0..dir_size];
        let dir_crc_expected = read_u32_le(&prefetch_bytes[dir_size..dir_size + 4]);
        let dir_crc_actual = crc32c(dir_bytes_slice);
        if dir_crc_expected != dir_crc_actual {
            return Err(VectorError::Read(ReadError::ChecksumMismatch {
                section: "vector/directory",
                column: String::new(),
            }));
        }

        // Stage the overlay with the combined head prefetch
        // and (case C only) the dir-anchored fallback.
        let mut overlay = PrefetchedSource::new(Arc::clone(&source));
        overlay.install(0, head_prefetch.clone());
        if prefetch_bytes.as_ptr() != head_prefetch.as_ptr() {
            // Case C — we issued a dir-anchored fallback GET.
            // Install its bytes so subsequent overlay lookups
            // resolve from memory.
            overlay.install(prefetch_bytes_start as u64, prefetch_bytes.clone());
        }

        // GET 3+ — for each column whose open-time region
        // overflowed the speculative window, fetch the tail.
        // The tail covers `[sub_header_end..per_cluster_blocks_off]`
        // and pulls the sub-header itself in the same range so
        // the overlay has one contiguous slice per subsection.
        for i in 0..n_columns {
            let entry_off = i * DIR_ENTRY_SIZE;
            let subsection_off =
                read_u64_le(&dir_bytes_slice[entry_off + 24..entry_off + 32]) as usize;
            let subsection_len =
                read_u64_le(&dir_bytes_slice[entry_off + 32..entry_off + 40]) as usize;
            if subsection_len < SUB_HEADER_SIZE + 4 {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} too short ({subsection_len} bytes)"
                ))));
            }
            let sub_end = subsection_off + subsection_len;
            if sub_end > blob_size {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} runs past blob",
                ))));
            }

            // Pull the sub-header. On subsection 0 it usually lives
            // inside the GET 2 prefetch; on subsections 1..N or
            // when GET 2's speculation didn't reach subsection 0's
            // sub-header (rare; would mean speculation < n_cent*8 +
            // codec_meta + ~56 B), the overlay misses and the
            // async `range` call below issues a real GET against
            // the underlying source.
            let sub_header = overlay
                .range(subsection_off as u64, SUB_HEADER_SIZE as u64)
                .await
                .map_err(|e| {
                    VectorError::Read(ReadError::MalformedVersion(format!(
                        "lazy open: subsection {i} sub-header fetch: {e}"
                    )))
                })?;
            if &sub_header[0..8] != format::vec::SUB_MAGIC {
                return Err(VectorError::Read(ReadError::BadMagic {
                    section: "vector/subsection",
                    expected: format::vec::SUB_MAGIC,
                    actual: sub_header[0..8].to_vec(),
                }));
            }
            let per_cluster_blocks_off = read_u64_le(&sub_header[48..56]) as usize;
            let open_time_abs_end = subsection_off + per_cluster_blocks_off;
            if open_time_abs_end > sub_end {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} per_cluster_blocks_off {per_cluster_blocks_off} \
                     exceeds subsection length {subsection_len}",
                ))));
            }
            let codec_meta_size = read_u32_le(&sub_header[12..16]) as usize;

            // Codec_meta lives at `[cluster_idx_off + n_cent*8 ..
            // per_cluster_blocks_off]`. We only need it for Sq8
            // columns (non-Sq8 declares codec_meta_size = 0).
            //
            // Plan 013 M5 — decide whether to fetch using actual
            // overlay coverage rather than `speculative_end`.
            // The combined head prefetch already covers
            // `[0..head_prefetch_end]`; whatever's past that
            // (and past the dir-anchored fallback's end in
            // case C) needs a per-subsection tail GET.
            if codec_meta_size > 0 {
                let cluster_idx_off = read_u64_le(&sub_header[40..48]) as usize;
                let n_cent = read_u32_le(&dir_bytes_slice[entry_off + 8..entry_off + 12]) as usize;
                let codec_meta_abs_off = subsection_off + cluster_idx_off + n_cent * 8;
                let already_covered =
                    open_time_abs_end <= head_prefetch_end || open_time_abs_end <= coverage_end;
                if !already_covered {
                    let codec_meta_len = (open_time_abs_end - codec_meta_abs_off) as u64;
                    let tail = source
                        .range(codec_meta_abs_off as u64, codec_meta_len)
                        .await
                        .map_err(|e| {
                            VectorError::Read(ReadError::MalformedVersion(format!(
                                "lazy open: subsection {i} codec_meta fetch: {e}"
                            )))
                        })?;
                    overlay.install(codec_meta_abs_off as u64, tail);
                }
            }
        }

        Self::open_with_source(Source::Lazy(Arc::new(overlay)), columns_json, opts)
    }

    /// Plan 011 M3 — open over an arbitrary [`Source`].
    ///
    /// The structural decode path is the same as
    /// [`Self::open_with`]; this entry just accepts a pre-built
    /// `Source`. Used by:
    /// - Test helpers that need to wire a counting / mock
    ///   [`LazyByteSource`] under a `Source::Lazy` (e.g. the
    ///   range-counting integration test).
    /// - [`Self::open_lazy`] (013 M2), which pre-fetches the
    ///   open-time region into a [`PrefetchedSource`] overlay
    ///   and hands the overlay through as `Source::Lazy`.
    ///
    /// Contract on `Source::Lazy`: the lazy source's
    /// `try_get_range_sync` must resolve every range request
    /// the structural decode path issues — sub-header (56 B per
    /// column) and codec_meta tail (Sq8 columns only). M2's
    /// `open_lazy` guarantees this via the overlay; callers
    /// constructing a `Source::Lazy` directly (tests, mmap-
    /// backed sources) must arrange equivalent residency.
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

        // Verify directory CRC (cheap, needed before we can parallelize
        // subsection CRCs since we walk dir entries to find them).
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

        // CRC verification: the outer-blob scan and per-subsection scans
        // each touch ~1.5 GiB at 1M × 384; together they're the bulk of
        // cold-open cost when `verify_crc=true`. Two stacked optimizations:
        //
        // 1. CLMUL-vectorized CRC32C via `crc-fast` in the checksum
        //    module — folds 8 lanes in vector regs instead of one
        //    serial dependency chain on `_mm_crc32_u64`, ~10× single-
        //    thread throughput on the boxes we measure.
        // 2. Run outer + per-subsection scans in parallel via rayon —
        //    each subsection's CRC is independent. The outer scan
        //    overlaps with the largest subsection on its own thread.
        //
        // Skip the whole pass via `OpenOptions { verify_crc: false }`
        // if upstream storage is trusted (content-addressed object
        // store, etc.); that path is ~12 ms regardless.
        if opts.verify_crc {
            // `Bytes` instead of `&'a [u8]` so the par_iter doesn't need
            // a lifetime parameter — each job owns a refcount-shared view
            // into the source, which is itself shared via the outer
            // `Source` for the duration of `open_with`.
            struct CrcJob {
                idx: i32,
                bytes: Bytes,
                expected: u32,
            }

            let mut jobs: Vec<CrcJob> = Vec::with_capacity(n_columns + 1);

            let outer_total = source.len();
            let outer_crc_pos = outer_total - 4;
            let outer_body = fetch_sync(&source, 0..outer_crc_pos, "outer body")?;
            let outer_crc_bytes = fetch_sync(&source, outer_crc_pos..outer_total, "outer crc")?;
            jobs.push(CrcJob {
                idx: -1,
                bytes: outer_body,
                expected: read_u32_le(&outer_crc_bytes),
            });

            for i in 0..n_columns {
                let entry_off = i * DIR_ENTRY_SIZE;
                let subsection_off =
                    read_u64_le(&dir_bytes[entry_off + 24..entry_off + 32]) as usize;
                let subsection_len =
                    read_u64_le(&dir_bytes[entry_off + 32..entry_off + 40]) as usize;
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
                } else {
                    let i = idx as usize;
                    return Err(VectorError::Read(ReadError::ChecksumMismatch {
                        section: "vector/subsection",
                        column: format!(" (column index {i})"),
                    }));
                }
            }
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
            // [52..56] = codec_id (1) + reserved (3) — plan 012 M1
            let _summary_off_abs = read_u64_le(&dir_bytes[entry_off + 40..entry_off + 48]);
            let codec_id = dir_bytes[entry_off + 52];
            let rerank_codec = RerankCodec::from_codec_id(codec_id).ok_or_else(|| {
                VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' has unknown rerank-codec id {codec_id} \
                     (plan 012: 0=fp32, 1=bf16, 2=sq8, 3=none)",
                    cfg.column
                )))
            })?;
            if !rerank_codec.is_implemented() {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' uses rerank codec {} which is not implemented yet \
                     (plan 012 — `fp32`, `bf16`, `sq8`, `none` are wired through M4)",
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

            // Validate subsection bounds + magic.
            //
            // Open-time region fetch — Plan 013 M2. The reader's
            // open path only reads the sub-header + (when present)
            // codec_meta from the subsection. Per-cluster blocks,
            // full[], and the trailing CRC are search-time concerns.
            //
            // To minimize cold-open byte volume against an object-
            // store-backed `Source::Lazy`, fetch in two phases:
            //   Phase A — sub-header (56 B) at `[subsection_off..
            //     subsection_off + SUB_HEADER_SIZE]`. Carries
            //     codec_meta_size and per_cluster_blocks_off, which
            //     together fix the open-time region's end offset.
            //   Phase B — codec_meta tail at `[subsection_off +
            //     cluster_idx_off + n_cent*8 .. subsection_off +
            //     per_cluster_blocks_off]` (Sq8 columns only;
            //     skipped when codec_meta_size == 0).
            //
            // On `Source::InMemory` both fetches are zero-copy
            // `Bytes::slice` views — identical cost to the previous
            // single full-subsection slice. On `Source::Lazy` they
            // resolve through the `PrefetchedSource` overlay
            // installed by `open_lazy` (zero underlying GETs) when
            // the caller pre-fetched the open-time region; on a
            // bare `Source::Lazy` they fall through to the
            // underlying async `range` and round-trip per fetch.
            let sub_end = subsection_off + subsection_len;
            if sub_end > source.len() {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} runs past blob"
                ))));
            }
            if subsection_len < SUB_HEADER_SIZE + 4 {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} too short"
                ))));
            }
            let sub_header = fetch_sync(
                &source,
                subsection_off..subsection_off + SUB_HEADER_SIZE,
                "sub_header",
            )?;
            if &sub_header[0..8] != format::vec::SUB_MAGIC {
                return Err(VectorError::Read(ReadError::BadMagic {
                    section: "vector/subsection",
                    expected: format::vec::SUB_MAGIC,
                    actual: sub_header[0..8].to_vec(),
                }));
            }
            // CRC was either already verified above in the parallel
            // pre-pass (when `opts.verify_crc` is true) or skipped on
            // the lazy fast path. Either way `sub_crc_pos` is derived
            // from `subsection_len` (directory entry), not from a
            // resident buffer.
            let sub_crc_pos = subsection_len - 4;

            // Sub-header parse. Only one layout supported
            // (new-service-only; no pre-013 segments to keep
            // readable). See `format::vec::SUBSECTION_VERSION`
            // for the byte-level spec.
            //   [ 8..12] SUBSECTION_VERSION
            //   [12..16] codec_meta_size (u32 LE)
            //   [16..24] summary_centroid_offset (u64 LE)
            //   [24..28] summary_radius_x100 (u32 LE)
            //   [28..32] reserved (u32)
            //   [32..40] centroids_off (u64 LE)
            //   [40..48] cluster_idx_off (u64 LE)
            //   [48..56] per_cluster_blocks_off (u64 LE)
            let subsection_version = read_u32_le(&sub_header[8..12]);
            if subsection_version != format::vec::SUBSECTION_VERSION {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' has unsupported subsection layout version \
                     {subsection_version}; this build supports only {}",
                    cfg.column,
                    format::vec::SUBSECTION_VERSION
                ))));
            }

            let quant = BitQuantizer::new(dim);
            let code_bytes = quant.code_bytes();
            if code_bytes == 0 {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' dim={dim} yields code_bytes=0",
                    cfg.column
                ))));
            }
            let per_vec_bytes = rerank_codec.per_vector_bytes(dim);
            let codec_meta_required_zero = matches!(
                rerank_codec,
                RerankCodec::Fp32 | RerankCodec::Bf16 | RerankCodec::RabitqOnly
            );

            let codec_meta_size = read_u32_le(&sub_header[12..16]) as usize;
            let summary_off = read_u64_le(&sub_header[16..24]) as usize;
            let summary_radius_x100 = read_u32_le(&sub_header[24..28]);
            let centroids_off = read_u64_le(&sub_header[32..40]) as usize;
            let cluster_idx_off = read_u64_le(&sub_header[40..48]) as usize;
            let per_cluster_blocks_off = read_u64_le(&sub_header[48..56]) as usize;

            if codec_meta_required_zero && codec_meta_size != 0 {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' has codec_meta_size={codec_meta_size} for codec {}; \
                     fp32/bf16/none must write codec_meta_size=0",
                    cfg.column,
                    rerank_codec.name()
                ))));
            }

            // codec_meta sits immediately after cluster_idx (when
            // non-empty); 0 means "no codec_meta" and skips the
            // sq8_meta parse below.
            let cluster_idx_size = (n_cent as usize) * 8;
            let codec_meta_off = if codec_meta_size == 0 {
                0
            } else {
                let off = cluster_idx_off + cluster_idx_size;
                // codec_meta must immediately precede the
                // per-cluster blocks region by exactly its
                // declared size. Any gap is a malformed segment.
                if off + codec_meta_size != per_cluster_blocks_off {
                    return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                        "column '{}' codec_meta region [{off}..{}) does not abut \
                         per_cluster_blocks_off={per_cluster_blocks_off}",
                        cfg.column,
                        off + codec_meta_size
                    ))));
                }
                off
            };

            // Per-cluster blocks total = n_docs * (code_bytes + 4);
            // full[] = n_docs * per_vec_bytes; together they fill
            // [per_cluster_blocks_off..sub_crc_pos). Solve for
            // n_docs.
            let blocks_plus_full_size = sub_crc_pos - per_cluster_blocks_off;
            let per_doc_blocks_plus_full = code_bytes + 4 + per_vec_bytes;
            if per_doc_blocks_plus_full == 0
                || !blocks_plus_full_size.is_multiple_of(per_doc_blocks_plus_full)
            {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' per_cluster_blocks + full region \
                     {blocks_plus_full_size} bytes not divisible by per-doc stride \
                     {per_doc_blocks_plus_full}",
                    cfg.column
                ))));
            }
            let col_n_docs = (blocks_plus_full_size / per_doc_blocks_plus_full) as u32;
            let per_cluster_blocks_size = (col_n_docs as usize) * (code_bytes + 4);
            let full_off = per_cluster_blocks_off + per_cluster_blocks_size;
            let actual_codec_meta_size = codec_meta_size;

            // Sq8 + L2Sq adds the per-doc norms tail to codec_meta
            // (`n_docs * 4` bytes); now that `col_n_docs` is known
            // we can validate the declared size against the codec's
            // exact expectation.
            let expected_codec_meta_size =
                rerank_codec.codec_meta_bytes(dim, col_n_docs as usize, n_cent as usize, metric);
            if actual_codec_meta_size != expected_codec_meta_size {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' codec_meta_size={actual_codec_meta_size} on disk but \
                     codec {} / metric {metric:?} expects {expected_codec_meta_size} bytes",
                    cfg.column,
                    rerank_codec.name()
                ))));
            }

            let summary_radius = (summary_radius_x100 as f32) / 100.0;

            // Materialise Sq8 codec_meta (scale + offset + optional
            // per-doc norms) at open time. The arrays are small —
            // ~3 KB scale+offset at dim=384, ~4 MB per-doc norms at
            // 1M docs — and reused across every search call.
            //
            // Parse via `f32::from_le_bytes` rather than
            // `bytemuck::cast_slice<u8, f32>` — the codec_meta
            // region sits at `cluster_idx_off + n_cent × 8` and
            // the f32 slices inside aren't guaranteed to be
            // 4-aligned (the `cast_slice` form would panic on
            // a misaligned slice).
            let sq8_meta = if rerank_codec == RerankCodec::Sq8 {
                // Per-cluster layout (Sq8PerCluster): the codec_meta
                // region is `n_cent × dim` f32s of scale followed by
                // `n_cent × dim` f32s of offset, optionally followed
                // by `n_docs` f32s of decoded-norm cache for
                // L2Sq / Cosine.
                //
                // Phase B fetch (Plan 013 M2) — codec_meta lives at
                // `[subsection_off + codec_meta_off ..
                //   subsection_off + codec_meta_off + codec_meta_size]`
                // inside the open-time region. Fetched separately
                // from the sub-header so non-Sq8 columns pay only
                // 56 B per subsection at open time.
                let meta_abs_start = subsection_off + codec_meta_off;
                let meta_abs_end = meta_abs_start + actual_codec_meta_size;
                let meta_bytes = fetch_sync(&source, meta_abs_start..meta_abs_end, "codec_meta")?;
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

            // Structural bounds. cluster_idx fits before the
            // per-cluster blocks region; full[] is the last
            // region before the CRC. The
            // `blocks_plus_full_size.is_multiple_of(...)` check
            // above already pinned n_docs; this check guards an
            // off-by-one in the cluster_idx slot.
            let cluster_idx_end = cluster_idx_off + cluster_idx_size;
            if cluster_idx_end > sub_crc_pos {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' cluster index runs past subsection",
                    cfg.column
                ))));
            }
            let full_size = (col_n_docs as usize) * per_vec_bytes;
            let full_end = full_off + full_size;
            if full_end != sub_crc_pos {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' full region ends at {full_end} but subsection body \
                     ends at {sub_crc_pos}",
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
                codec_meta_off,
                per_cluster_blocks_off,
                full_off,
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

    /// Single-column kNN search. Returns `(local_doc_id,
    /// distance)` sorted ascending by distance (smaller = closer
    /// for every metric).
    ///
    /// Sync — matches plan 002 Q9's convention (every public
    /// surface in `src/` is sync). Routes per-region byte
    /// access through [`Source::get_range`], which is itself
    /// sync and bridges to the underlying async
    /// `LazyByteSource::range` only on a cold `Source::Lazy`
    /// miss (via `block_in_place + Handle::block_on`, same
    /// pattern as `supertable::query::segment_reader`). On
    /// `Source::InMemory` and on `Source::Lazy` warm caches
    /// (`BytesLazyByteSource`, mmap-backed) every fetch resolves
    /// zero-copy on the sync fast path.
    ///
    /// Range count per cold first search at `nprobe = 8` on the
    /// v0 layout:
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
    /// table at all — that 4 MB / 1M-doc allocation was deleted
    /// in plan 011 M4 once the audit confirmed zero external
    /// readers. See `claude_plans/011_lazy_reader_loads.md`
    /// § Search path for the contract.
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
        // `full[]` (rerank candidates) is codec-dependent — fp32 today,
        // bf16 from M2, sq8 from M3.
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

        // 5. Per-cluster fetches and shortlist build. Shortlist
        //    tuple is (doc_id, estimate, pos, cluster_id);
        //    pos = off + i and cluster_id are captured inline at
        //    no extra fetch cost. cluster_id is consumed by the
        //    Sq8PerCluster rerank dispatch to pick each
        //    candidate's quantizer; Fp32/Bf16/None rerank paths
        //    ignore it.
        //
        //    Plan 013 M3 — codes and doc_ids per cluster live in
        //    one contiguous block on disk (`per-cluster blocks`
        //    region under the v1 layout), so each cluster pulls
        //    in **one** `get_range` call. Plan 013 M5 — those
        //    `nprobe` per-cluster GETs fire **concurrently**
        //    via [`Source::get_ranges_parallel`] instead of
        //    serially via per-call [`Source::get_range`]. On a
        //    `Source::Lazy` backed by object storage the cold
        //    first-search wall-clock collapses from
        //    `sum_c RTT(c)` to `max_c RTT(c)` (one HTTP/2
        //    multiplexed batch). On warm/in-memory paths the
        //    requests resolve through the sync zero-copy
        //    fast path with no extra cost.
        let _ = sub_start; // retained for downstream offset math below
        let cb = col.quant.code_bytes();
        let mut cluster_meta: Vec<(usize, u32, u32)> = Vec::with_capacity(nprobe_eff);
        let mut cluster_ranges: Vec<Range<usize>> = Vec::with_capacity(nprobe_eff);
        for &(c, _) in &centroid_scores {
            let (off, cnt) = read_cluster_entry(&cluster_idx, c);
            if cnt == 0 {
                continue;
            }
            cluster_ranges.push(col.cluster_block_range(off, cnt));
            cluster_meta.push((c, off, cnt));
        }
        let cluster_blocks = self
            .source
            .get_ranges_parallel(&cluster_ranges)
            .map_err(|e| VectorError::LazySource(e.to_string()))?;
        debug_assert_eq!(cluster_blocks.len(), cluster_meta.len());

        let mut shortlist: Vec<(u32, f32, u32, u32)> = Vec::new();
        for ((c, off, cnt), block) in cluster_meta.into_iter().zip(cluster_blocks) {
            let codes_len = (cnt as usize) * cb;
            let doc_ids_len = (cnt as usize) * 4;
            debug_assert_eq!(block.len(), codes_len + doc_ids_len);
            let codes = block.slice(0..codes_len);
            let doc_ids = block.slice(codes_len..codes_len + doc_ids_len);
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

        // Plan 012 M4: `None` codec short-circuit. The 1-bit
        // shortlist *is* the final ranking — there's no `full[]`
        // region on disk and no rerank step. We:
        //   * partial-sort the shortlist to land the top-K by
        //     descending estimate (higher dot estimate = better
        //     candidate),
        //   * fully sort the retained k for a stable output
        //     ordering, and
        //   * flip the sign of the estimate so the returned
        //     `(doc_id, distance)` pairs follow the standard
        //     "smaller = closer" convention for the caller. The
        //     value is a 1-bit-derived score, not a true metric
        //     distance; for `None` columns recall is the
        //     contract, not numerical agreement with fp32.
        //
        // `rerank_mult` is intentionally ignored here — there's
        // nothing to refine. Storage shrinks by ~30×; recall
        // drops 0.05-0.15 vs rerank-equipped codecs (corpus-
        // dependent). M5 will surface the bench numbers.
        if matches!(col.rerank_codec, RerankCodec::RabitqOnly) {
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
        //    when positions span probed clusters; grouping
        //    consecutive runs into multiple smaller ranges is a
        //    013 M3 follow-up. The `full[]` region itself sits
        //    at the same per-vector stride in both v0 and v1
        //    (only its absolute offset differs).
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
        let full_range = col.rerank_run_range(min_pos, max_pos, full_vec_bytes);
        let full_start = full_range.start;
        let full_end = full_range.end;
        let full_run = self
            .source
            .get_range(full_start..full_end)
            .map_err(|e| VectorError::LazySource(e.to_string()))?;

        // 8. CPU-only rerank using the true metric. Sq8 columns
        //    pre-build a per-query kernel that folds the per-dim
        //    scale/offset into the query (one `dim/8` SIMD pass);
        //    the per-doc inner step is then a plain u8→f32 widen
        //    + SIMD dot. Fp32/Bf16 take the flat dispatch.
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
/// captured for the Sq8PerCluster rerank dispatch: each candidate
/// knows which cluster's `(scale, offset)` quantizer to dequant
/// against. For Fp32/Bf16/None the cluster_id is recorded but
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
/// - **Fp32 / Bf16**: flat dispatch via [`distance_bytes_codec`]
///   (fp32 zero-copy SIMD or bf16-widen SIMD).
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
        RerankCodec::Fp32 | RerankCodec::Bf16 => shortlist
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
            // Sq8PerCluster: each candidate's cluster_id selects
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
            let mut kernel_cache: HashMap<u32, Sq8Kernel> = HashMap::new();
            shortlist
                .iter()
                .map(|&(did, _, pos, cluster_id)| {
                    let local = (pos - base_pos) as usize;
                    let start = local * stride;
                    let bytes = &full_run[start..start + stride];
                    let kernel = kernel_cache.entry(cluster_id).or_insert_with(|| {
                        let c = cluster_id as usize;
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
            "rerank_candidates_in_run reached with None codec — None columns \
             have no full[] region and should short-circuit before the rerank step"
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
/// open time over at-most-`8·dim + 4·n_docs` bytes per Sq8
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
    // Plan 011 M1 — Source enum sanity tests
    // -----------------------------------------------------------------
    //
    // M1 only adds the enum + reroutes runtime byte access through
    // it; the public open path still takes a `Bytes` (M2 introduces
    // `open_lazy`). These tests directly exercise the `Source`
    // contract so any future refactor that breaks the InMemory
    // zero-copy invariant or mis-implements the Lazy wrapper fails
    // here rather than at the wider recall oracle gate.

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
    // Plan 012 M1 — rerank-codec discriminator round-trip
    // -----------------------------------------------------------------
    //
    // The codec discriminator rides as byte 52 of the per-column
    // directory entry; the codec_meta region offset rides as bytes
    // 12..16 of the sub-header. Both are zero on pre-012 fp32
    // segments. M2 wires `Fp32` + `Bf16` end-to-end — `Sq8` / `None`
    // must still round-trip as a typed `MalformedVersion` at open
    // time so a future segment built by an M3+ binary fails loud
    // against an M2 binary rather than mis-decoding.

    use crate::superfile::format::checksum::crc32c;

    /// Plan 012 M1: a fresh `Fp32` build round-trips through the
    /// reader with `ColumnReader.rerank_codec == Fp32` — the
    /// directory-entry codec byte makes it back out of the on-disk
    /// representation unchanged. The structural assertion pins the
    /// on-disk discriminator contract.
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

    /// Plan 012 M4: every codec the enum exposes is now wired end-
    /// to-end (`Fp32` M1, `Bf16` M2, `Sq8` M3, `None` M4), so
    /// `register_column` must accept all of them. The check exists
    /// so adding a *new* unimplemented variant in the future
    /// surfaces here loud and fast.
    #[test]
    fn register_column_accepts_every_codec() {
        for codec in [
            RerankCodec::Fp32,
            RerankCodec::Bf16,
            RerankCodec::Sq8,
            RerankCodec::RabitqOnly,
        ] {
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

    /// Plan 012 M2: building a column with `RerankCodec::Bf16`
    /// round-trips through the reader. The codec discriminator
    /// surfaces on `ColumnReader.rerank_codec`, the
    /// codec_meta region stays zero-bytes (bf16 has no per-column
    /// metadata), and the on-disk `full[]` region halves to
    /// `n_docs × dim × 2` bytes.
    #[test]
    fn open_round_trips_bf16_codec_discriminator() {
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
            rerank_codec: RerankCodec::Bf16,
        })
        .expect("register column");
        for i in 0..n_docs {
            let v: Vec<f32> = (0..dim).map(|j| (i + j as u32) as f32 * 0.1).collect();
            b.add(0, &v).expect("add");
        }
        let blob = b.finish().expect("finish");

        let json =
            r#"[{"column":"v","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        assert_eq!(r.columns.len(), 1);
        assert_eq!(
            r.columns[0].rerank_codec,
            RerankCodec::Bf16,
            "Bf16 build must surface as RerankCodec::Bf16 on the reader"
        );
        assert_eq!(
            r.columns[0].codec_meta_off, 0,
            "Bf16 segments must write codec_meta_off = 0 (zero-byte meta region)"
        );
        // bf16 = 2 bytes/dim, so full[] = n_docs × dim × 2 bytes.
        // full[] is the last region before the trailing 4-byte
        // CRC, so its on-disk size is
        // `(subsection_size - 4) - full_off`.
        let col = &r.columns[0];
        let expected_full_size = (col.n_docs as usize) * dim * 2;
        let actual_full_size = (col.subsection_range.len() - 4) - col.full_off;
        assert_eq!(
            actual_full_size, expected_full_size,
            "Bf16 full[] region must be n_docs × dim × 2 bytes",
        );
    }

    /// Plan 012 M2: a Bf16 build + open + self-query recovers the
    /// planted self-vector at top-1, end-to-end through the
    /// codec-aware rerank dispatch. Confirms the encode (build) +
    /// widen-to-fp32 (search) paths agree on the bf16 layout —
    /// any byte-order or stride mismatch would surface as
    /// wrong-doc-id or out-of-bounds.
    #[test]
    fn bf16_self_query_round_trips_top1() {
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
            rerank_codec: RerankCodec::Bf16,
        })
        .expect("register column");
        // Use values that survive bf16 rounding exactly so the rerank
        // distance is 0.0 for the self-query — sidesteps tolerance
        // noise on the assertion.
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
            .search("v", &all[17], 5, 4, 5)
            .expect("search must succeed on Bf16 column");
        assert_eq!(hits[0].0, 17, "Bf16 self-query must recover self at top-1");
        // The L2Sq distance back to self is bounded by the bf16
        // round-trip error of the planted vector (zero on
        // half-integers below 64, so this is exact).
        assert!(
            hits[0].1.abs() <= 1e-3,
            "Bf16 self-query distance {} should be ~0",
            hits[0].1
        );
    }

    /// Plan 012 M3: building a column with `RerankCodec::Sq8`
    /// round-trips through the reader. The codec discriminator
    /// surfaces on `ColumnReader.rerank_codec`; the codec_meta
    /// region carries `scale[dim] + offset[dim]` (always) plus
    /// per-doc norms (L2Sq only). The on-disk `full[]` region
    /// shrinks to `n_docs × dim` u8 codes (4× smaller than fp32).
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

        // codec_meta_off must be non-zero for Sq8 — codec_meta
        // sits inside the open-time region between cluster_idx
        // and the per-cluster blocks.
        assert_ne!(col.codec_meta_off, 0, "Sq8 must declare codec_meta_off > 0");
        // full[] is n_docs × dim u8 codes — the final region
        // before the trailing CRC.
        let actual_full_size = (col.subsection_range.len() - 4) - col.full_off;
        assert_eq!(actual_full_size, (col.n_docs as usize) * dim);

        // sq8_meta materialised at open: per-cluster scale +
        // offset (Sq8PerCluster layout — n_cent × dim floats
        // each), per-doc norms present for L2Sq.
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

    /// Plan 012 M3 + Sq8PerCluster: cosine Sq8 columns carry the
    /// per-doc decoded-norm cache — the rerank kernel normalizes
    /// the decoded vector with it (`1 − dot / |x_decoded|`). Only
    /// negdot drops the norms (its `Σx²` term cancels out),
    /// shrinking codec_meta from `8·n_cent·dim + 4·n_docs` to
    /// `8·n_cent·dim`.
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
            // Pre-normalised vectors so cosine has a meaningful
            // reference; the test only checks the codec_meta shape,
            // not the recall.
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
        let norms = meta.per_doc_norms.as_ref().expect(
            "Cosine Sq8 must carry per-doc norms to normalize the decoded vector at rerank",
        );
        assert_eq!(norms.len(), n_docs as usize);
        assert_eq!(meta.scale.len(), n_cent * dim);
        assert_eq!(meta.offset.len(), n_cent * dim);
    }

    /// Plan 012 M3: pins the per-doc-norms indexing contract —
    /// the on-disk norms array is indexed by **position in
    /// `full[]`** (matching the rerank shortlist's `pos`),
    /// not by `doc_id`. The two diverge whenever the writer
    /// pool's cluster-contiguous order differs from insertion
    /// order, which it does in practice (rows get scattered
    /// across clusters by the k-means assignment, so pos ≠ id
    /// for almost every doc).
    ///
    /// Pin: insert N docs whose decoded norms strictly increase
    /// with insertion order, build, open, and assert that the
    /// open-time norms array — read in pos order — does **not**
    /// equal the insertion-order norms. If it does, we're
    /// silently indexing the wrong way; an L2Sq distance lookup
    /// would then return some other doc's norm and corrupt the
    /// rerank ordering.
    ///
    /// We also recompute each `norms[pos]` from the planted
    /// vectors via the per-pos `doc_id` and confirm it matches
    /// — proving the pos-indexed lookup actually resolves to
    /// "this doc's decoded L2 norm".
    #[test]
    fn sq8_per_doc_norms_indexed_by_pos_not_doc_id() {
        let dim = 16usize;
        let n_cent = 4usize;
        let n_docs = 32u32;
        // Vectors whose L2 norm grows monotonically with doc_id.
        // Scattered across clusters by the k-means / random
        // rotation, so insertion order ≠ pos order — making the
        // mis-index easy to detect.
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

        // Insertion-order norms (computed against the fp32
        // originals; these monotonically increase by design).
        let insertion_norms: Vec<f32> = planted
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum::<f32>())
            .collect();

        // If norms were indexed by doc_id, `norms_by_pos[i]`
        // would equal `insertion_norms[i]` up to quantization
        // (a few percent). Cluster-scattered builds reorder
        // docs across positions, so the two sequences should
        // disagree on most slots — this asserts the reorder
        // actually happened (the pin would be vacuous if every
        // doc landed at `pos = doc_id`).
        let n_matching = insertion_norms
            .iter()
            .zip(norms_by_pos.iter())
            .filter(|(ins, pos_n)| (**ins - **pos_n).abs() < 0.5)
            .count();
        assert!(
            n_matching < (n_docs as usize) / 2,
            "expected k-means + rotation to scatter docs across positions, \
             but norms_by_pos matches insertion_norms in {n_matching}/{n_docs} \
             slots — test corpus may have landed all docs in pos == doc_id order, \
             defeating the indexing pin"
        );

        // For every pos, confirm `norms_by_pos[pos]` equals the
        // decoded L2 norm of the doc at that pos. We don't know
        // the pos↔doc_id mapping from outside the reader, but a
        // self-query against `planted[i]` should return doc_id=i
        // at top-1; the returned distance should be ~0 (matches
        // the quantized doc to itself). That same distance,
        // recomputed via the kernel using doc_i's planted
        // values, requires `norms_by_pos[pos_of(i)]` to be doc_i's
        // decoded norm — exactly the contract we're pinning.
        // A mis-index would surface as a non-zero self-distance
        // larger than the quantization error tolerance.
        for i in [0u32, 7, 15, 23, 31] {
            // rerank_mult=64 → refine=64 ≥ n_docs=32 → every
            // candidate is reranked. Removes the 1-bit shortlist
            // as a confounding variable: any miss here is a real
            // norms-indexing bug, not a Hamming-recall artifact.
            let hits = r
                .search("v", &planted[i as usize], 1, 4, 64)
                .expect("self-query");
            assert_eq!(hits[0].0, i, "self-query top-1 doc_id for doc {i}");
            // Quantization noise bound: per-dim error ≤ scale/2
            // ≈ span/510. For our corpus, dim spans are ~ 16, so
            // |q-x|² ≤ dim · (span/510)² ≈ 16 · 0.001 ≈ 0.016.
            // A norms-table mis-index would push this to the
            // order of the other docs' norms (≥ 1 unit).
            assert!(
                hits[0].1 <= 0.5,
                "doc {i}: self-query distance {} too large — likely norms \
                 mis-indexed (pos vs doc_id swap)",
                hits[0].1
            );
        }
    }

    /// Plan 012 M3: an Sq8 build + open + self-query recovers the
    /// planted self-vector at top-1. End-to-end through the
    /// codec-aware rerank dispatch + Sq8Kernel — any layout drift
    /// (codec_meta order, code stride, per-doc-norm indexing)
    /// would surface as wrong-doc or out-of-bounds.
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
        // Exhaustive rerank (rerank_mult=20 → refine=100 ≥ n_docs=64)
        // so the test pins Sq8 codec correctness independently of
        // the 1-bit shortlist's recall ceiling.
        let hits = r
            .search("v", &all[17], 5, 4, 20)
            .expect("search must succeed on Sq8 column");
        assert_eq!(hits[0].0, 17, "Sq8 self-query must recover self at top-1");
        // Sq8 round-trip error: per-dim quantization step is
        // `scale ≈ (max-min)/255`. For this corpus, dim values
        // sit in [0, 31.5] so per-dim error ≲ 0.06, |q-x|² over
        // 32 dims ≲ 32 × 0.06² ≈ 0.12. Pinning a generous bound
        // to keep the test robust to RNG quirks.
        assert!(
            hits[0].1 <= 1.0,
            "Sq8 self-query distance {} should be small (≤ 1.0)",
            hits[0].1
        );
    }

    /// Plan 012 M3: Sq8 self-query top-1 round-trips under Cosine
    /// too. Exercises the Cosine branch of `Sq8Kernel::distance_at`
    /// (no per-doc-norm lookup, `dist = 1 − dot`).
    ///
    /// Corpus design (matters!): unit-norm vectors drawn from
    /// hashed-uniform values per (doc, dim) so neighbor pairs land
    /// at `dot ≈ 1/√dim ≈ 0.18` — gap to self of ~0.82, well above
    /// the Sq8 quantization noise floor (~0.005 for this corpus).
    /// An earlier draft used `((i·23 + j·5) % 50 + 1)` which made
    /// adjacent docs near-parallel (dot ≈ 0.99) and triggered a
    /// quantization-driven swap of doc 5 ↔ doc 42 on self-query —
    /// real Sq8+Cosine behaviour on pathological inputs, not a
    /// kernel bug, but not a useful pin for codec correctness.
    /// Real cosine workloads (semantic embeddings) look like the
    /// current corpus, not the pathological one.
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
                    // Per-(doc, dim) hash → uniform u16 → fp32 in
                    // [0, 1). Two docs from this generator have
                    // expected dot product ≈ 1/√dim ≈ 0.18 after
                    // L2-normalization.
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
        // Exhaustive rerank (rerank_mult=20 → refine=100 ≥ n_docs=64)
        // so any failure here pins Sq8 codec correctness rather than
        // 1-bit shortlist recall.
        let hits = r
            .search("v", &all[42], 5, 4, 20)
            .expect("search must succeed on Sq8 cosine column");
        assert_eq!(hits[0].0, 42, "Sq8 cosine self-query must recover self");
    }

    // -----------------------------------------------------------------
    // Plan 012 M4 — `None` codec (no rerank column)
    // -----------------------------------------------------------------
    //
    // The `None` codec drops the `full[]` region entirely. The
    // 1-bit shortlist *is* the final ranking; the on-disk
    // segment shrinks by ~30× at 1M × 384. Distance values
    // returned from `search()` are `-estimate` (1-bit dot
    // estimate, sign-flipped so smaller = closer holds) — not a
    // true metric distance.

    /// Plan 012 M4: building with `RerankCodec::RabitqOnly` succeeds
    /// and the on-disk segment carries a zero-length `full[]`
    /// region. Also pins the directory-entry discriminator
    /// (`codec_id = 3`) and the zero-byte codec_meta invariant
    /// (`codec_meta_off = 0`).
    #[test]
    fn open_round_trips_none_codec_discriminator() {
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
        .expect("register None column");
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
        assert_eq!(
            col.rerank_codec,
            RerankCodec::RabitqOnly,
            "None build must surface as RerankCodec::RabitqOnly on the reader"
        );
        assert_eq!(
            col.codec_meta_off, 0,
            "None segments must write codec_meta_off = 0 (zero-byte meta region)"
        );
        // `None` segments have zero-length full[]. Since full is
        // the last region before the trailing CRC, `full_off`
        // lands at the subsection body's end — i.e.
        // `subsection_size - 4`.
        let body_end_off = col.subsection_range.len() - 4;
        assert_eq!(
            col.full_off, body_end_off,
            "None segments have zero-length full[] — full_off must coincide \
             with the end of the subsection body"
        );
        assert_eq!(col.n_docs, n_docs);
    }

    /// Plan 012 M4: a `None`-codec column's self-query returns
    /// the planted vector inside the top-K of the 1-bit
    /// shortlist. At dim=128 / n_docs=64 with a well-separated
    /// corpus the 1-bit estimator's top-K reliably contains the
    /// self-vector even without rerank — exactly the contract
    /// `None` opts into. Returned distances are `-estimate`
    /// (sign-flipped so smaller = closer holds).
    #[test]
    fn none_self_query_in_top_k_via_shortlist_only() {
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
        .expect("register None column");
        // Angularly diverse corpus — hashed-uniform vectors,
        // L2-normalized. Two docs from this generator have
        // expected dot ≈ 1/√dim ≈ 0.09, so 1-bit estimates
        // separate cleanly and the self-vector dominates the
        // shortlist for its own query.
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

        // nprobe = n_cent so every cluster contributes to the
        // shortlist — the test asserts the 1-bit shortlist's
        // top-K contract, not the cluster-probing one. rerank_mult
        // is intentionally ignored by the None path (asserted
        // here by passing a value that would otherwise oversample).
        let hits = r
            .search("v", &all[17], 5, n_cent, 5)
            .expect("None-codec search must succeed");
        assert!(
            !hits.is_empty(),
            "None-codec search must return at least one hit"
        );
        assert!(
            hits.iter().any(|(did, _)| *did == 17),
            "self-query must surface the planted vector in top-K, got {hits:?}"
        );
        // Distances are `-estimate` — non-positive for a
        // self-query (the 1-bit dot estimate of a vector with
        // itself is strictly positive after the random rotation).
        assert!(
            hits.iter().all(|(_, d)| d.is_finite()),
            "all None-codec distances must be finite, got {hits:?}"
        );
        // Top-1's distance must be ≤ any other hit's (ascending
        // sort contract).
        for w in hits.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "None-codec hits must be sorted ascending by distance, got {hits:?}"
            );
        }
    }

    /// Plan 012 M4: a `None`-codec search over a counting
    /// lazy source must not perform any range fetch past the
    /// `doc_ids` region — proven indirectly via the total
    /// range count: 2 centroids-region + 2 cluster-idx-region
    /// + `2 × nprobe` (codes + doc_ids per probed cluster). A
    /// regression that left the fat `full[]` `get_range` in
    /// for None columns would surface as one extra range
    /// request, which this asserts away. The structural
    /// invariant (full[] is zero-length on disk) is pinned by
    /// `open_round_trips_none_codec_discriminator`; this test
    /// pins the read-path side.
    #[test]
    fn none_search_issues_no_full_region_fetch() {
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
        .expect("register None column");
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

        // Reset counters: open() touches the directory + every
        // sub-header. We only want to count search-time fetches.
        async_calls.store(0, AtomicOrdering::Relaxed);
        sync_calls.store(0, AtomicOrdering::Relaxed);
        let query: Vec<f32> = (0..dim).map(|j| j as f32 * 0.1).collect();
        let _ = r.search("v", &query, 5, n_cent, 5).expect("search");

        // Upper-bound sync fetches for None / nprobe = n_cent:
        //   centroids (1) + cluster_idx (1)
        // + per-cluster codes (≤ n_cent)
        // + per-cluster doc_ids (≤ n_cent)
        // = at most 2 + 2·n_cent = 10
        //
        // The Fp32/Bf16/Sq8 paths would add one more fat
        // `full[]` get_range on top — that's the leak this
        // test guards against. Empty clusters reduce the
        // upper bound (per-cluster fetches skip on cnt == 0)
        // but never raise it. Async should stay at 0 —
        // warm-cache lazy never falls through to the async
        // bridge for in-memory blobs.
        let sync_count = sync_calls.load(AtomicOrdering::Relaxed) as usize;
        let async_count = async_calls.load(AtomicOrdering::Relaxed);
        assert_eq!(
            async_count, 0,
            "None-codec search on warm lazy must not bridge to async"
        );
        let max_expected = 2 + 2 * n_cent;
        assert!(
            sync_count <= max_expected,
            "None-codec search must issue at most 2 + 2·nprobe = {max_expected} \
             sync fetches (centroids + cluster_idx + per-cluster codes + \
             per-cluster doc_ids); got {sync_count} — any extra is a leaked \
             full[] fetch"
        );
        // A search that probed at least one non-empty cluster
        // must issue ≥ 4 fetches (centroids + idx + ≥1 cluster's
        // codes + doc_ids); below that and we'd be testing a
        // pathological corpus, not the None codec.
        assert!(
            sync_count >= 4,
            "test corpus produced only empty clusters? got sync_count={sync_count}"
        );
    }

    /// Plan 012 M2: a directory entry carrying an unknown codec id
    /// (anything outside `0..=3` — e.g. `255` from a corrupted /
    /// future-format segment) errors as `MalformedVersion`. The
    /// safety net catches both forward-compat reads (future codec
    /// ids land in the gap) and on-disk corruption.
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

        let err = VectorReader::open_with(
            Bytes::from(bytes),
            &json,
            OpenOptions {
                verify_crc: false,
                ..Default::default()
            },
        )
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
    // Plan 011 M3.b / M4 — lazy open + inline-`pos` search
    // -----------------------------------------------------------------
    //
    // Open touches only the structural-decode regions (directory,
    // sub-header, summary, centroids, cluster_idx). Search carries
    // `pos = off + i` inline in the shortlist tuple — there is no
    // `doc_to_pos` lookup table to populate (deleted in M4 after
    // the audit confirmed zero external readers). The structural
    // memory-ceiling tests below ride on these invariants.

    // -----------------------------------------------------------------
    // Plan 012 M5 diagnostic — Sq8 vs Fp32 recall on planted-cluster
    // cosine corpus
    // -----------------------------------------------------------------
    //
    // The 1M × 384 bench measured Sq8 recall@10 = 0.860 vs Fp32 = 0.964
    // — well outside the plan's "< 0.005 drop on normalized embeddings"
    // envelope. The hypothesis is that the **per-column** Sq8 quantizer
    // wastes most of its 256 buckets on cross-cluster spread: per-dim
    // global range across 1M docs ≈ 0.4, intra-cluster spread ≈ 0.015,
    // so within any one cluster only ~10 buckets are used. The intra-
    // cluster cosine differences between top-K candidates are smaller
    // than the per-bucket quantization noise → reranks flip.
    //
    // This `#[ignore]`-gated diagnostic reproduces the recall drop at
    // 16k × 384 (1/64 scale) and prints corpus geometry stats. Run
    // with `cargo test --lib -- sq8_recall_diagnostic --ignored
    // --nocapture` to inspect. Per-column-quantizer fix (or fallback
    // to Bf16 default) is decided based on what this prints.
    #[test]
    #[ignore = "Plan 012 M5 recall diagnostic; ~10s; --ignored --nocapture"]
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
        let bf16_blob = build(RerankCodec::Bf16);
        let sq8_blob = build(RerankCodec::Sq8);
        eprintln!(
            "--- segment sizes ---\n\
             fp32: {:.2} MiB (1.00x)\n\
             bf16: {:.2} MiB ({:.2}x)\n\
             sq8:  {:.2} MiB ({:.2}x)",
            fp32_blob.len() as f64 / 1024.0 / 1024.0,
            bf16_blob.len() as f64 / 1024.0 / 1024.0,
            bf16_blob.len() as f64 / fp32_blob.len() as f64,
            sq8_blob.len() as f64 / 1024.0 / 1024.0,
            sq8_blob.len() as f64 / fp32_blob.len() as f64
        );

        let json = format!(
            r#"[{{"column":"v","dim":{dim},"n_cent":{n_cent_ivf},"rot_seed":7,"metric":"cosine"}}]"#
        );
        let r_fp32 = VectorReader::open(fp32_blob, &json).expect("open fp32");
        let r_bf16 = VectorReader::open(bf16_blob, &json).expect("open bf16");
        let r_sq8 = VectorReader::open(sq8_blob, &json).expect("open sq8");

        // 4. Brute-force ground truth (cosine sim descending = neg-dot
        //    ascending — both engines return smaller-is-closer).
        let n_queries = 100usize;
        let k = 10usize;
        let nprobe = n_cent_ivf / 4;
        let rerank_mult = 50usize; // plan-doc Sq8 floor at dim ≤ 384
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
        let r_bf = recall_of(&r_bf16, "bf16");
        let r_sq = recall_of(&r_sq8, "sq8 ");
        eprintln!(
            "drop (fp32 - bf16): {:.4}\ndrop (fp32 - sq8 ): {:.4}",
            r_fp - r_bf,
            r_fp - r_sq
        );
        eprintln!(
            "(plan acceptance #2: drop must be \u{2264} 0.01; bench measured 0.10 drop at 1M scale)"
        );

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

    /// Search-shape corpus used by the M3.b inline-pos tests and the
    /// M3 sync-search / counting-source tests. Picks a non-trivial
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

    /// Plan 011 M3.b self-query smoke: lazy default open must
    /// recover the planted self-vector at top-1, confirming the
    /// inline-`pos` rerank path returns the correct results on
    /// the search-shape corpus that every M3/M4 test uses.
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
    // Plan 011 M3 — sync `search()` on `Source::Lazy`
    // -----------------------------------------------------------------
    //
    // These tests pin the M3 contract per plan 002 Q9 (commit
    // `2e351ba`): the *only* public entry point is sync
    // `search()`. It works on every `Source` variant — `InMemory`
    // and warm-cache `Source::Lazy` resolve every range through
    // `try_get_range_sync` (zero-copy); cold-miss `Source::Lazy`
    // bridges to the source's async `range()` via
    // `block_in_place + Handle::block_on` / one-shot
    // `current_thread` `Runtime`, the same pattern
    // `supertable::query::segment_reader` uses for the disk-cache
    // fetch path. No `search_async` is exposed at the public
    // surface; the cold-path async bridging is hidden inside
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
        /// Current in-flight `range()` futures (entry-bumped,
        /// drop-decremented). Plan 013 M5 — pairs with
        /// `max_in_flight` to pin that
        /// [`Source::get_ranges_parallel`] dispatches its cold
        /// fetches concurrently rather than serially.
        in_flight: StdArc<AtomicU64>,
        max_in_flight: StdArc<AtomicU64>,
        /// Per-`range()` artificial latency. Defaults to zero
        /// (legacy callers); the parallel-dispatch test sets it
        /// to a small delay so concurrent futures actually
        /// overlap in wall-clock instead of completing in the
        /// trivial sync slice path inside `range`.
        async_latency_us: AtomicU64,
    }

    impl CountingLazyByteSource {
        fn new(bytes: Bytes) -> Self {
            Self {
                bytes,
                async_calls: StdArc::new(AtomicU64::new(0)),
                sync_calls: StdArc::new(AtomicU64::new(0)),
                sync_disabled: AtomicBool::new(false),
                in_flight: StdArc::new(AtomicU64::new(0)),
                max_in_flight: StdArc::new(AtomicU64::new(0)),
                async_latency_us: AtomicU64::new(0),
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

        /// Max-concurrent observer — sampled at every `range()`
        /// entry. Concurrent fetches will produce a value `> 1`;
        /// serial fetches stay at `1`.
        fn max_in_flight_counter(&self) -> StdArc<AtomicU64> {
            StdArc::clone(&self.max_in_flight)
        }

        /// Set per-`range()` artificial latency. Used by the
        /// parallel-dispatch test to ensure concurrent futures
        /// overlap in wall-clock (without latency, the trivial
        /// `bytes.slice(...)` body of `range()` resolves
        /// instantaneously and in-flight peaks at 1 even when
        /// many futures were spawned together).
        fn set_async_latency(&self, latency: std::time::Duration) {
            self.async_latency_us
                .store(latency.as_micros() as u64, AtomicOrdering::Relaxed);
        }
    }

    /// RAII guard: bumps `in_flight` on construct, decrements
    /// on drop, and bumps `max_in_flight` if the new in-flight
    /// count exceeds the previous max. Pairs with
    /// [`CountingLazyByteSource::max_in_flight_counter`] to give
    /// the parallel-dispatch test a single observable for
    /// "fetches actually overlapped."
    struct InFlightGuard {
        in_flight: StdArc<AtomicU64>,
        max_in_flight: StdArc<AtomicU64>,
    }

    impl InFlightGuard {
        fn enter(in_flight: StdArc<AtomicU64>, max_in_flight: StdArc<AtomicU64>) -> Self {
            let now = in_flight.fetch_add(1, AtomicOrdering::AcqRel) + 1;
            // Bump max_in_flight monotonically.
            max_in_flight.fetch_max(now, AtomicOrdering::AcqRel);
            Self {
                in_flight,
                max_in_flight,
            }
        }
    }

    impl Drop for InFlightGuard {
        fn drop(&mut self) {
            self.in_flight.fetch_sub(1, AtomicOrdering::AcqRel);
            // max_in_flight is monotonic by design; nothing to
            // unwind on drop.
            let _ = &self.max_in_flight;
        }
    }

    #[async_trait::async_trait]
    impl LazyByteSource for CountingLazyByteSource {
        fn size(&self) -> u64 {
            self.bytes.len() as u64
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            self.async_calls.fetch_add(1, AtomicOrdering::Relaxed);
            let _guard = InFlightGuard::enter(
                StdArc::clone(&self.in_flight),
                StdArc::clone(&self.max_in_flight),
            );
            let latency_us = self.async_latency_us.load(AtomicOrdering::Relaxed);
            if latency_us > 0 {
                tokio::time::sleep(std::time::Duration::from_micros(latency_us)).await;
            }
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
    /// exercises the no-ambient-runtime fallback branch to
    /// keep that path live.
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

    /// Range-counting test (plan 011 M3 budget). Sync `search()`
    /// issues per-region / per-cluster `Source::get_range`
    /// calls:
    ///
    /// - 1 range for centroids
    /// - 1 range for cluster_idx
    /// - 1 range per probed cluster's codes
    /// - 1 range per probed cluster's doc_ids
    /// - 1 fat range for the rerank batch in `full[]`
    ///
    /// On v0 layout at `nprobe = N` with all probed clusters
    /// non-empty: `2 + 2N + 1 = 2N + 3` ranges. The corpus here
    /// has `n_cent = 4` and the test uses `nprobe = 4`, so the
    /// worst-case budget is `2·4 + 3 = 11`. The plan's
    /// production budget (`nprobe = 8` on a 1M corpus) is
    /// `2·8 + 3 = 19` — and 013 M1's v1 layout drops this further
    /// by interleaving codes + doc_ids per cluster (one range
    /// per cluster instead of two).
    ///
    /// Forcing `try_get_range_sync` off makes every range route
    /// through the source's async `range()` via the block_on
    /// bridge, so the `async_calls` counter is the right
    /// instrumentation for "how many distinct ranges did
    /// `search()` request".
    ///
    /// A regression that smuggles in extra range fetches — e.g.
    /// reintroducing the whole-subsection fallback, or pulling the
    /// full `doc_ids` region over the wire at open — surfaces here
    /// rather than at the production S3 harness in 013.
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
    // Plan 011 § Acceptance #2 — memory-ceiling unit test
    // -----------------------------------------------------------------
    //
    // Plan 011's headline guarantee is "resident set per open
    // vector segment is bounded by O(n_cent × dim × 4 + small)",
    // independent of `n_docs`. Acceptance criterion #2 spells it
    // out: opening a `Source::Lazy` over a mmap-backed
    // `BytesLazyByteSource` at 1M × 384 with
    // `OpenOptions { verify_crc: false }` must leave the process
    // RSS delta ≤ 10 MB per opened column.
    //
    // Why mmap specifically: this is exactly how the disk cache
    // (003 M5) feeds bytes into `SuperfileReader` —
    // `Bytes::from_owner(Arc<Mmap>)`. The kernel never faults the
    // bulk codes/full/doc_ids pages on the default path because
    // nothing in `open_with_source` accesses them: the CRC scan
    // is gated on `verify_crc`, search uses inline `pos` (M3.b)
    // so no `doc_ids` walk happens, and the structural-decode
    // bytes (outer header + dir + sub_header) are a handful of
    // pages. The resident allocation is dominated by the rotation
    // matrix (≈ 590 KB at dim=384) and small column metadata —
    // well inside the 10 MB ceiling at any practical
    // `n_docs`.
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
    /// `VectorBuilder::finish_to` (plan 010 M5). The streaming
    /// write avoids materializing a 1.5 GiB `Vec<u8>` in the
    /// test's address space at 1M × 384 — the build's transient
    /// peak doesn't survive the `before` RSS snapshot.
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
            OpenOptions {
                verify_crc: false,
                ..Default::default()
            },
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

    /// **Plan 011 acceptance criterion #2 (plan-spec scale).**
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
    /// materialization (the pre-011 behaviour) or that scans
    /// `doc_ids` at open will push per-column RSS past the
    /// 10 MB ceiling and fail here rather than at the 100 M
    /// production OOM.
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
            "Plan 011 acceptance #2: lazy open RSS delta \
             {per_col_mib:.3} MiB/col exceeds 10 MiB ceiling \
             at 1M × {DIM}, n_cent={N_CENT} (total delta \
             {delta_mib:.3} MiB over {n_cols} column(s))."
        );
    }

    /// **Plan 011 acceptance criterion #2 (smoke scale).**
    ///
    /// 50 k × 64, `n_cent = 64`. Runs in default
    /// `cargo test --lib` (~1–2 s build) so every PR gets
    /// continuous feedback on the structural property: lazy
    /// open touches only the structural-decode pages, never
    /// the bulk codes/full/doc_ids regions. The 10 MiB ceiling
    /// at the plan's headline 1M × 384 scale is asserted at
    /// the same value here because the resident allocation
    /// (mostly the rotation matrix at `dim²·4` = 16 KB for
    /// dim=64) is *smaller* at smoke scale, not larger — if
    /// this fires, the bigger test will too.
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
    // Plan 011 — supertable-scale memory ceiling
    // -----------------------------------------------------------------
    //
    // The single-segment `mem_ceiling_lazy_open_*` tests above pin the
    // per-reader bound. These multi-segment variants pin the
    // *supertable-shaped* bound: open N segments concurrently — same
    // shape `Supertable::commit` produces (N = N_SEGMENTS_BENCH × num_cpus
    // because `split_buffer_into_row_shards` shards each commit's
    // buffer into one segment per writer-pool thread) — and assert the
    // total anon RSS delta scales as `N × O(centroids + rotation +
    // small)`, not as `N × subsection_size`.
    //
    // What this proves (and what it doesn't):
    //
    // - PROVES: a supertable opened with the production disk-cache
    //   path (`Source::InMemory(Bytes::from_owner(mmap))` per segment —
    //   see `supertable::reader_cache::disk::insert`) keeps anon
    //   RSS bounded across an arbitrary number of segments, with no
    //   per-doc anon term. Equivalent here because
    //   `Bytes::from_owner` is zero-copy over the mmap, and the
    //   lazy-open path doesn't touch `doc_ids[]` / `full[]` at
    //   open time (M3.b's inline `pos` removes the only reason
    //   open ever touched `doc_ids[]`).
    //
    // - DOES NOT PROVE: the in-process `InMemoryReaderCache` path
    //   (`Bytes::from(Vec)` per segment — see
    //   `supertable::reader_cache::in_memory::insert`) has the same
    //   bound. That path holds each segment's bytes in anon by
    //   construction (no mmap involved). The in-memory cache is the
    //   test/bench path; production attaches a `StorageProvider` and
    //   routes through the disk cache. A separate test for the
    //   in-memory cache path isn't a 011 deliverable — that path's
    //   anon cost is its declared contract.
    //
    // The bench's 10M × 4-commit × num_cpus-thread shape produces
    // exactly the topology these tests exercise. The smoke variant
    // mirrors the bench's *layout* at a tiny corpus size (4 segments
    // × 50 k docs × 64 dim) so every PR catches regressions
    // (~5 s build). The `#[ignore]`'d plan-spec variant uses the
    // bench's actual per-segment shape (16 segments × 625 k docs ×
    // 384 dim × n_cent_per_segment matching the bench's
    // `n_cent_total / 4`) and runs only when called out.

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
                OpenOptions {
                    verify_crc: false,
                    ..Default::default()
                },
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

    /// **Plan 011 supertable-scale memory ceiling (smoke).**
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

    /// **Plan 011 supertable-scale memory ceiling (plan-spec).**
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

    /// **Plan 011 M4 — many-segments stress test (100M
    /// aspiration shape).**
    ///
    /// The honest scale test for "100M docs across a supertable"
    /// can't materialise 100M production-shape segments on a
    /// developer box (the per-segment 625k × 384 shape used in
    /// the bench produces ~960 MiB on disk × 160 segments = 150
    /// GiB of corpus). Instead, this test pins the *structural*
    /// memory bound by varying the high-cardinality axis (segment
    /// count) at a thin per-segment shape: **100 segments × 50 k
    /// docs × 128 dim × 128 n_cent**.
    ///
    /// What this proves:
    ///
    /// - Per-segment open allocation is `O(n_cent × dim × 4 +
    ///   rotation + small)` — no `n_docs` term. At this shape:
    ///   centroids 64 KiB + rotation matrix 64 KiB + column
    ///   metadata ≪ 1 MiB per segment. Total expected RSS delta
    ///   ≪ 200 MiB across 100 segments; 400 MiB ceiling for
    ///   allocator overhead + reader-struct fields.
    ///
    /// - The deletion of `doc_to_pos` (M4) made segment-count
    ///   the only scaling dimension. A regression that reintroduced
    ///   any per-doc resident state — e.g. a returning lookup
    ///   table at `n_docs × 4` bytes per column — would here
    ///   allocate 100 × 50 k × 4 = 20 MiB anon just for tables
    ///   (small but growing); at the bench's 100 segments × 625 k
    ///   the same regression is 250 MiB.
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
             any per-doc resident state would push this much higher; M4's deletion of \
             doc_to_pos is what keeps the bound structural."
        );

        drop(tmps);
    }

    // -----------------------------------------------------------------
    // Plan 013 M2 — VectorReader::open_lazy cold-open range budget +
    // round-trip parity. The lazy open path issues a 1-GET combined
    // head prefetch (outer header + dir + speculative open-time tail,
    // all in one range) against the underlying LazyByteSource, plus
    // optional per-column follow-ups for shapes where the speculation
    // didn't cover the open-time region. Every fetched range is
    // installed into a PrefetchedSource overlay; the subsequent
    // structural decode resolves every sub-header / codec_meta read
    // from the overlay without touching the underlying source again,
    // so the underlying source sees only the open_lazy-issued GETs.
    //
    // Plan 013 M5 — pre-M5 the budget was 2-3 GETs: a 32-byte outer
    // header followed by a `dir_offset`-anchored range. M5 folds
    // both into a single `[0..open_time_speculative_bytes]` GET so
    // the typical small-segment, single-column cold open issues
    // exactly **1** async range call instead of 2.
    // -----------------------------------------------------------------

    fn build_small_segment(
        dim: usize,
        n_cent: usize,
        n_docs: u32,
        codec: RerankCodec,
        metric: Metric,
    ) -> (Bytes, String, Vec<Vec<f32>>) {
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 41,
            metric,
            rerank_codec: codec,
        })
        .expect("register column");
        let mut all = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs {
            let v: Vec<f32> = (0..dim).map(|j| (i + j as u32) as f32 * 0.1).collect();
            b.add(0, &v).expect("add");
            all.push(v);
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let metric_str = match metric {
            Metric::L2Sq => "l2sq",
            Metric::Cosine => "cosine",
            Metric::NegDot => "negdot",
        };
        let json = format!(
            r#"[{{"column":"v","dim":{dim},"n_cent":{n_cent},"rot_seed":41,"metric":"{metric_str}"}}]"#,
        );
        (blob, json, all)
    }

    /// Plan 013 M5 — single-column small-segment cold open
    /// against a `LazyByteSource` issues exactly **1**
    /// underlying async `range` call: the combined head
    /// prefetch covering outer header + directory + open-time
    /// speculation. Every sub-header / codec_meta read
    /// resolves from the `PrefetchedSource` overlay; the
    /// underlying source sees no follow-up GETs.
    ///
    /// This is the headline budget for a typical superfile vector
    /// segment (≤ 1 MiB open-time region, single column). Larger
    /// segments (≥ 10M × 1024) or multi-column segments trigger
    /// follow-up GETs covered by `open_lazy_fp32_segment_*` below.
    ///
    /// Pre-M5 this was 2 GETs (header probe + dir-anchored
    /// fetch); M5 folds the two into one.
    #[tokio::test]
    async fn open_lazy_small_sq8_segment_issues_one_async_range() {
        let (blob, json, _) = build_small_segment(32, 4, 64, RerankCodec::Sq8, Metric::L2Sq);
        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let async_counter = counting.async_counter();

        let _reader = VectorReader::open_lazy(
            StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
            &json,
            OpenOptions::for_object_store(),
        )
        .await
        .expect("open_lazy small Sq8");

        let n_calls = async_counter.load(AtomicOrdering::Relaxed);
        assert_eq!(
            n_calls, 1,
            "small-segment open_lazy must issue exactly 1 async range call \
             (combined head prefetch: outer header + dir + speculative \
             open-time tail); observed {n_calls}",
        );
    }

    /// Plan 013 M5 — Fp32 / Bf16 / None codecs declare
    /// `codec_meta_size = 0`, so the combined head prefetch
    /// covers every needed byte. Confirms the 1-range budget
    /// holds across every non-Sq8 codec.
    #[tokio::test]
    async fn open_lazy_small_segment_issues_one_async_range_every_codec() {
        for codec in [
            RerankCodec::Fp32,
            RerankCodec::Bf16,
            RerankCodec::RabitqOnly,
        ] {
            let (blob, json, _) = build_small_segment(32, 4, 64, codec, Metric::L2Sq);
            let counting = StdArc::new(CountingLazyByteSource::new(blob));
            let async_counter = counting.async_counter();

            let _reader = VectorReader::open_lazy(
                StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
                &json,
                OpenOptions::for_object_store(),
            )
            .await
            .unwrap_or_else(|e| panic!("open_lazy {codec:?}: {e:?}"));

            let n_calls = async_counter.load(AtomicOrdering::Relaxed);
            assert_eq!(
                n_calls, 1,
                "open_lazy ({codec:?}) must issue exactly 1 async range call; \
                 observed {n_calls}",
            );
        }
    }

    /// Plan 013 M5 — when the combined head prefetch is too
    /// small to even reach `dir_end`, `open_lazy` falls back
    /// to (a) a dir-anchored fallback GET covering
    /// `[dir_offset..dir_end + spec]`, plus (b) a
    /// per-subsection codec_meta tail GET when neither prefetch
    /// covers the open-time region's end. Pins the budget at
    /// exactly **3** range calls for the most adversarial shape
    /// (64-byte speculation — below both `dir_end` and the
    /// codec_meta region).
    ///
    /// On the common path (spec ≥ dir_end) the dir-anchored
    /// fallback never fires — see
    /// `open_lazy_small_sq8_segment_issues_one_async_range`
    /// for the headline 1-GET budget.
    #[tokio::test]
    async fn open_lazy_undersized_speculation_issues_follow_up_range() {
        let (blob, json, _) = build_small_segment(32, 4, 64, RerankCodec::Sq8, Metric::L2Sq);
        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let async_counter = counting.async_counter();

        // 64-byte speculation; head_prefetch_end = 64 bytes
        // (max(64, OUTER_HEADER_SIZE + 4 = 36) = 64), well
        // below `dir_end = 32 + 1*64 + 4 = 100`. Forces case C
        // dir-anchored fallback and a codec_meta tail GET.
        let opts = OpenOptions {
            verify_crc: false,
            open_time_speculative_bytes: 64,
        };
        let _reader = VectorReader::open_lazy(
            StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
            &json,
            opts,
        )
        .await
        .expect("open_lazy with undersized speculation");

        let n_calls = async_counter.load(AtomicOrdering::Relaxed);
        assert_eq!(
            n_calls, 3,
            "undersized speculation must issue exactly 3 range calls \
             (combined head + dir-anchored fallback + codec_meta tail); \
             observed {n_calls}",
        );
    }

    /// Plan 013 M2 — round-trip parity. A search against an
    /// `open_lazy` reader returns the same `(doc_id, distance)`
    /// hits as the eager `open()` path. Confirms the open-path
    /// refactor (Phase A sub-header + Phase B codec_meta) and
    /// the overlay round-trip preserve every search-critical
    /// metadata field.
    #[tokio::test]
    async fn open_lazy_search_matches_eager_open_per_codec() {
        for codec in [
            RerankCodec::Fp32,
            RerankCodec::Bf16,
            RerankCodec::Sq8,
            RerankCodec::RabitqOnly,
        ] {
            let (blob, json, all) = build_small_segment(32, 4, 64, codec, Metric::L2Sq);
            let r_eager = VectorReader::open(blob.clone(), &json)
                .unwrap_or_else(|e| panic!("eager open {codec:?}: {e:?}"));
            let counting = StdArc::new(CountingLazyByteSource::new(blob));
            let r_lazy = VectorReader::open_lazy(
                StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
                &json,
                OpenOptions::for_object_store(),
            )
            .await
            .unwrap_or_else(|e| panic!("open_lazy {codec:?}: {e:?}"));

            for &q_idx in &[0usize, 7, 17, 31] {
                let hits_eager = r_eager
                    .search("v", &all[q_idx], 5, 4, 5)
                    .unwrap_or_else(|e| panic!("eager search {codec:?}: {e:?}"));
                let hits_lazy = r_lazy
                    .search("v", &all[q_idx], 5, 4, 5)
                    .unwrap_or_else(|e| panic!("lazy search {codec:?}: {e:?}"));
                assert_eq!(
                    hits_eager, hits_lazy,
                    "search results must match between eager and lazy open \
                     (codec {codec:?}, query {q_idx})",
                );
            }
        }
    }

    /// Plan 013 M3 — cold first search after `open_lazy` issues
    /// at most `nprobe + 1` underlying async range GETs against
    /// the LazyByteSource (one combined codes+doc_ids block per
    /// probed cluster + one fat rerank range). The pre-search
    /// reads (centroids + cluster_idx) all resolve from the
    /// `PrefetchedSource` overlay installed by `open_lazy` —
    /// they live inside the open-time region and don't touch
    /// the underlying source again.
    ///
    /// Headline budget for the 013 plan's "First-search phase"
    /// (≤ 12 ranges, ≤ 5 MB at 1M × 384 sq8, nprobe = 8). The
    /// small-segment test here pins the structural shape; M5's
    /// s3s-fs bench measures the real wall-clock against AWS-
    /// shape RTTs.
    ///
    /// The test sizes the open_time_speculative_bytes knob to
    /// exactly cover the open-time region. Without that, the
    /// default 2 MiB speculation would overlay the entire small
    /// test segment and search would resolve everything from
    /// the overlay — masking the cold-fetch range budget the
    /// test is meant to pin.
    ///
    /// "At most" because some probed clusters can be empty
    /// (zero-count entries skip the block fetch entirely); for a
    /// well-distributed corpus the budget is hit exactly.
    ///
    /// Runs on the multi-thread tokio runtime so the sync
    /// `search()` path's `block_in_place + Handle::block_on`
    /// bridge can fire (the current-thread runtime that
    /// `#[tokio::test]` uses by default rejects `block_in_place`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cold_first_search_after_open_lazy_within_nprobe_plus_one_ranges() {
        let (blob, json, all) = build_small_segment(32, 8, 128, RerankCodec::Sq8, Metric::L2Sq);

        // Inspect the segment via the eager path to learn its
        // open-time region size; size the lazy open's speculative
        // tail to cover exactly that.
        let r_eager = VectorReader::open(blob.clone(), &json).expect("eager open");
        let col_eager = &r_eager.columns[0];
        let open_time_region_bytes = col_eager.per_cluster_blocks_off;
        drop(r_eager);

        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let async_counter = counting.async_counter();
        // Disable BytesLazyByteSource's zero-copy sync path so
        // every non-overlay read is forced through the async
        // `range` bridge — that's what an object-store-backed
        // source actually pays per region.
        counting.disable_sync();

        let opts = OpenOptions {
            verify_crc: false,
            open_time_speculative_bytes: open_time_region_bytes,
        };
        let r_lazy = VectorReader::open_lazy(
            StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
            &json,
            opts,
        )
        .await
        .expect("open_lazy");

        let after_open = async_counter.load(AtomicOrdering::Relaxed);
        // Plan 013 M5 — open_lazy issues 1-2 GETs here:
        // the combined head prefetch (always), plus an
        // optional dir-anchored fallback if the speculation
        // didn't reach the open-time region end. With the
        // speculation knob sized to `per_cluster_blocks_off`
        // (= the open-time region end), the head prefetch
        // captures everything and exactly 1 GET fires.
        // The accepted budget is 1 or 2.
        assert!(
            (1..=2).contains(&after_open),
            "open_lazy must issue 1-2 async ranges before search starts; \
             observed {after_open}",
        );

        let nprobe = 4usize;
        let _hits = r_lazy
            .search("v", &all[0], 5, nprobe, 5)
            .expect("cold first search");

        let after_search = async_counter.load(AtomicOrdering::Relaxed);
        let search_calls = after_search - after_open;
        let max_expected = (nprobe + 1) as u64;
        assert!(
            search_calls <= max_expected,
            "cold first search at nprobe={nprobe} must issue ≤ {max_expected} async \
             range GETs (one per probed cluster + one fat rerank range); observed \
             {search_calls}",
        );
        assert!(
            search_calls >= 2,
            "cold first search must issue at least 2 async range GETs (≥1 cluster \
             block + 1 rerank range); observed {search_calls} suggests search \
             accidentally short-circuited the cluster / rerank fetch paths",
        );
    }

    /// Plan 013 M5 — cold first search must dispatch its
    /// per-cluster block fetches **concurrently**, not
    /// serially. The total range-GET count was already
    /// pinned by the M3 budget test above; this test pins
    /// the round-trip count.
    ///
    /// Each `range()` call holds an in-flight slot (RAII
    /// guard); peak in-flight ≥ 2 proves the cluster fetches
    /// overlapped. We pad `range()` with a small artificial
    /// latency so a serial implementation completes each
    /// future before the next is awaited — without the
    /// latency, the trivial `bytes.slice(...)` body
    /// resolves instantly and even a serial caller looks
    /// concurrent (in-flight peaks at 1 indistinguishably).
    ///
    /// Runs on the multi-thread runtime for the same
    /// `block_in_place` reason as the M3 test above.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cold_first_search_dispatches_cluster_gets_concurrently() {
        let (blob, json, all) = build_small_segment(32, 8, 256, RerankCodec::Sq8, Metric::L2Sq);

        // Size open-time speculation to cover subsection 0
        // exactly so the speculation doesn't bleed into the
        // per-cluster region; otherwise the cluster fetches
        // would hit the overlay and skip the async path.
        let r_eager = VectorReader::open(blob.clone(), &json).expect("eager open");
        let open_time_region_bytes = r_eager.columns[0].per_cluster_blocks_off;
        drop(r_eager);

        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let max_in_flight = counting.max_in_flight_counter();
        counting.disable_sync();
        counting.set_async_latency(std::time::Duration::from_millis(5));

        let opts = OpenOptions {
            verify_crc: false,
            open_time_speculative_bytes: open_time_region_bytes,
        };
        let r_lazy = VectorReader::open_lazy(
            StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
            &json,
            opts,
        )
        .await
        .expect("open_lazy");

        // Reset max_in_flight after open (we only want to
        // pin the search-side dispatch shape; open is its
        // own M2 budget exercise).
        max_in_flight.store(0, AtomicOrdering::Release);

        let nprobe = 8usize;
        let hits = tokio::task::spawn_blocking({
            let r_lazy = std::sync::Arc::new(r_lazy);
            let q = all[0].clone();
            move || r_lazy.search("v", &q, 5, nprobe, 5)
        })
        .await
        .expect("spawn_blocking join")
        .expect("cold first search");
        assert!(!hits.is_empty(), "self-query should return ≥ 1 hit");

        let peak = max_in_flight.load(AtomicOrdering::Acquire);
        // With nprobe=8 well-populated clusters the cluster
        // fetches must overlap; allow margin (≥ 2) so an
        // unusually busy scheduler doesn't false-fail. Pre-M5
        // (serial dispatch) this value is exactly 1.
        assert!(
            peak >= 2,
            "cold first search per-cluster fetches must overlap (peak in-flight \
             ≥ 2); observed {peak} — looks like the parallel-dispatch path \
             regressed to serial",
        );
    }

    /// Plan 013 M3 — round-trip parity for the unified
    /// codes+doc_ids per-cluster fetch path. The combined block
    /// gets sliced into a `codes` prefix and `doc_ids` suffix
    /// inside the search hot loop; this test pins that the
    /// slice boundaries land at exactly `count * code_bytes`
    /// (i.e. the bit-identical results survive the refactor
    /// from two separate ranges to one combined block).
    #[tokio::test]
    async fn m3_combined_cluster_fetch_matches_eager_open_per_codec() {
        for codec in [
            RerankCodec::Fp32,
            RerankCodec::Bf16,
            RerankCodec::Sq8,
            RerankCodec::RabitqOnly,
        ] {
            let (blob, json, all) = build_small_segment(32, 4, 64, codec, Metric::L2Sq);
            let r_eager = VectorReader::open(blob.clone(), &json)
                .unwrap_or_else(|e| panic!("eager open {codec:?}: {e:?}"));
            let counting = StdArc::new(CountingLazyByteSource::new(blob));
            let r_lazy = VectorReader::open_lazy(
                StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
                &json,
                OpenOptions::for_object_store(),
            )
            .await
            .unwrap_or_else(|e| panic!("open_lazy {codec:?}: {e:?}"));

            for &q_idx in &[0usize, 7, 17, 31] {
                let hits_eager = r_eager
                    .search("v", &all[q_idx], 5, 4, 5)
                    .unwrap_or_else(|e| panic!("eager search {codec:?}: {e:?}"));
                let hits_lazy = r_lazy
                    .search("v", &all[q_idx], 5, 4, 5)
                    .unwrap_or_else(|e| panic!("lazy search {codec:?}: {e:?}"));
                assert_eq!(
                    hits_eager, hits_lazy,
                    "M3 combined cluster fetch must produce bit-identical search \
                     results vs eager (codec {codec:?}, query {q_idx})",
                );
            }
        }
    }

    /// Plan 013 M3 — pins the `cluster_block_range` address math
    /// against the 013 layout's per-cluster block spec
    /// (`[codes: cnt*cb][doc_ids: cnt*4]`). Walks every non-
    /// empty cluster and checks the block range size matches
    /// `cnt × (cb + 4)` exactly, the start aligns with
    /// `per_cluster_blocks_off + doc_off × (cb + 4)`, and the
    /// codes/doc_ids halves slice in at the expected boundary
    /// inside the fetched block.
    #[test]
    fn cluster_block_range_matches_v1_layout_invariant() {
        let (blob, json, _) = build_small_segment(32, 4, 64, RerankCodec::Sq8, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open");
        let col = &r.columns[0];
        let cb = col.quant.code_bytes();

        let cluster_idx_bytes = r
            .source
            .try_get_range_sync(
                col.subsection_range.start + col.cluster_idx_off
                    ..col.subsection_range.start + col.cluster_idx_off + (col.n_cent as usize) * 8,
            )
            .expect("cluster_idx must be resident in InMemory source");

        let mut n_non_empty = 0usize;
        for c in 0..col.n_cent {
            let (off, cnt) = read_cluster_entry(&cluster_idx_bytes, c as usize);
            if cnt == 0 {
                continue;
            }
            n_non_empty += 1;
            let block = col.cluster_block_range(off, cnt);
            let expected_start =
                col.subsection_range.start + col.per_cluster_blocks_off + (off as usize) * (cb + 4);
            let expected_len = (cnt as usize) * (cb + 4);
            assert_eq!(
                block.start, expected_start,
                "cluster {c} block start must equal \
                 per_cluster_blocks_off + doc_off × (cb + 4)",
            );
            assert_eq!(
                block.len(),
                expected_len,
                "cluster {c} block size must equal cnt × (cb + 4)",
            );
            // Inside the fetched block, `[0..cnt*cb)` is codes
            // and `[cnt*cb..cnt*(cb+4))` is doc_ids — the exact
            // boundary the search() hot path slices at.
            let codes_end = (cnt as usize) * cb;
            assert!(
                codes_end < block.len(),
                "codes prefix must precede doc_ids suffix"
            );
            assert_eq!(
                block.len() - codes_end,
                (cnt as usize) * 4,
                "doc_ids suffix must be cnt × 4 bytes",
            );
        }
        assert!(
            n_non_empty > 0,
            "test corpus must populate at least one cluster"
        );
    }

    /// Plan 013 M2 — verify the `Source::Lazy` reader constructed
    /// by `open_lazy` exposes the same column metadata as the
    /// eager reader (dim, n_cent, n_docs, codec, sq8_meta shape).
    /// The structural decode that produces `ColumnReader` runs
    /// against the overlay; this test pins that every parsed
    /// field surfaces unchanged.
    #[tokio::test]
    async fn open_lazy_column_metadata_matches_eager_open() {
        let (blob, json, _) = build_small_segment(32, 4, 64, RerankCodec::Sq8, Metric::L2Sq);
        let r_eager = VectorReader::open(blob.clone(), &json).expect("eager open");
        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let r_lazy = VectorReader::open_lazy(
            StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
            &json,
            OpenOptions::for_object_store(),
        )
        .await
        .expect("open_lazy");

        assert_eq!(r_eager.columns.len(), r_lazy.columns.len());
        let col_eager = &r_eager.columns[0];
        let col_lazy = &r_lazy.columns[0];
        assert_eq!(col_eager.name, col_lazy.name);
        assert_eq!(col_eager.dim, col_lazy.dim);
        assert_eq!(col_eager.n_cent, col_lazy.n_cent);
        assert_eq!(col_eager.n_docs, col_lazy.n_docs);
        assert_eq!(col_eager.rerank_codec, col_lazy.rerank_codec);
        assert_eq!(col_eager.metric, col_lazy.metric);

        let meta_eager = col_eager.sq8_meta.as_ref().expect("eager Sq8 meta");
        let meta_lazy = col_lazy.sq8_meta.as_ref().expect("lazy Sq8 meta");
        assert_eq!(meta_eager.scale, meta_lazy.scale);
        assert_eq!(meta_eager.offset, meta_lazy.offset);
        assert_eq!(meta_eager.per_doc_norms, meta_lazy.per_doc_norms);
    }
}
