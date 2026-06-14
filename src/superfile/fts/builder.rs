// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS blob builder. Multi-column FTS index assembly.
//!
//! `FtsBuilder` accumulates posting records across all FTS-indexed
//! columns and on `finish_to<W>` emits the on-disk FTS blob:
//!
//! ```text
//!   header (48 bytes)
//!   FST term dictionary  + CRC32C
//!   postings region      + CRC32C
//!   doc-lengths directory   + CRC32C
//!   per-column doc-lengths arrays  (each + its own CRC32C)
//! ```
//!
//! See `docs/architecture/superfile.md` for the full byte-level spec.
//!
//! ## Build architecture
//!
//! Two-mode accumulator, threshold-based, identical in shape to
//! `VectorBuilder`:
//!
//! - **In-RAM mode**: each column holds a `FxHashMap<term, Vec<(doc,
//!   tf)>>` until total accumulated bytes cross
//!   `spill_threshold_bytes` (default 256 MiB). Small builds never
//!   touch the disk during `add_doc`.
//! - **Spill mode**: once the threshold is crossed, that column's
//!   terms are interned into a per-column `term_to_id`/`id_to_term`
//!   pair, the in-memory map is drained into per-column hash-
//!   partitioned spill files holding **fixed-size 12-byte
//!   `(term_id_le, doc_id_le, tf_le)` triples**, and from then on
//!   `add_doc` writes one triple per posting straight to the spill
//!   files via buffered file IO. Same shape as vector's spill: no
//!   per-record framing, no variable-length payload, no per-record
//!   allocation on read.
//!
//! `finish_to<W: Write>` correspondingly has two paths:
//!
//! - **In-RAM finish**: no column spilled. Per-column maps are
//!   drained, sorted, encoded into a posting-region scratch file,
//!   and the FST is built in RAM (small).
//! - **Spilled finish**: at least one column has spilled. The
//!   spilled column's `id_to_term` builds a lex-rank lookup
//!   (`term_id → rank in lex order`, one `Vec<u32>` per column,
//!   bounded by vocab so small even at 10M docs). Partition files
//!   are read as fixed-size triples, sorted by
//!   `(lex_rank[term_id], doc_id)` (pdqsort over `[(u32, u32,
//!   u32)]` — pure u32 compares, no `&[u8]` chasing), then
//!   k-way-merged into global lex order. The FST is built
//!   *streaming* via [`StreamingDictBuilder`] writing to a scratch
//!   file, using `id_to_term[term_id]` to recover the term bytes
//!   per emission. Final blob assembly is `header → FST scratch →
//!   posting scratch → doc-lengths`, all streamed through `W`.
//!
//! Mirror of vector: vector spills its input corpus as raw f32
//! bytes past 256 MiB and streams its centroid+code layout to
//! scratch; FTS spills its posting accumulator as fixed 12-byte
//! triples past 256 MiB and streams its FST + posting region to
//! scratch. Both bound peak resident memory by a formula that does
//! not include `n_docs`, and both use fixed-size, no-framing record
//! formats so the spill IO is allocator-free on the read side.
//!
//! ## Builder lifecycle
//!
//! 1. `FtsBuilder::new(tokenizer)` — empty builder.
//! 2. `register_column(name)` per FTS column, in declaration order.
//! 3. `add_doc(column_id, local_doc_id, text)` per `(doc, column)` pair.
//!    Caller passes monotonically-increasing `local_doc_id`s.
//! 4. `finish()` (returns `Vec<u8>`) or `finish_to(impl Write)`
//!    (streams the blob progressively to any sink).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::hash::{BuildHasher, Hash};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use hashbrown::hash_map::{HashMap as HbHashMap, RawEntryMut};
use memmap2::Mmap;
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::superfile::BuildError;
use crate::superfile::format::checksum::{crc32c, crc32c_append};
use crate::superfile::format::fts::HEADER_SIZE as FTS_HEADER_SIZE;
use crate::superfile::format::{self, FST_SEPARATOR};
use crate::superfile::fts::dict::{DictBuilder, StreamingDictBuilder};
use crate::superfile::fts::fst_value::FstValue;
use crate::superfile::fts::posting::{BLOCK_LEN, Block, EncodedBlock, encode_block};
use crate::superfile::fts::tokenize::{AsciiLowerTokenizer, Tokenizer};

/// Per-column term interner table.
///
/// Keys are `&'static str` slices that **actually borrow from the
/// per-column `term_arena: bumpalo::Bump`** stored alongside the
/// map in the [`ColumnPostings::Spilled`] variant. The `'static`
/// lifetime tag is a lie maintained by `unsafe` `std::mem::
/// transmute` at insertion (see [`intern_term_id_arena`]) and
/// kept sound by two structural invariants:
///
/// 1. The arena outlives the map. The `Spilled` variant declares
///    `term_arena` *after* `term_to_id` and `id_to_term` so Rust
///    drops it last (struct fields drop in declaration order, so
///    the last-declared field is the last-dropped). The
///    `finish_to` per-column block that destructures `Spilled`
///    likewise puts `term_arena` last in the pattern and adds an
///    explicit `drop(term_arena)` after the last use of
///    `id_to_term` for the same reason.
/// 2. No method on the map's keys does a deref during the map's
///    own `Drop`. `&str` has no `Drop` impl, so dropping the map
///    just deallocates the bucket table without touching the
///    key bytes. The arena can therefore safely outlast the
///    map's body without the map ever needing arena bytes
///    during its own teardown.
///
/// Switching from `Box<str>` keys to arena `&str` keys:
///
///   * removes one `Box<str>` heap allocation per intern miss
///     (was: one for the map key + one cloned for `id_to_term`;
///     now: one bump-arena copy reused by both);
///   * packs all term bytes densely in the bump arena instead
///     of scattering them across the global heap, which keeps
///     the byte-compare fast path inside `raw_entry_mut` cache-
///     hot on the per-token spill-arm hot loop (the bench
///     vocab is ~10K terms × ~5-10 byte average = ~75 KiB
///     total, comfortably L2-resident as one contiguous arena
///     instead of ~10K separate heap allocations chasing
///     pointers through unrelated allocator buckets).
type TermIdMap = HbHashMap<&'static str, u32, FxBuildHasher>;

#[derive(Default)]
struct FinishProfile {
    enabled: bool,
    encode_calls: u64,
    encode_df1: u64,
    encode_pfor: u64,
    encode_total: Duration,
    encode_block_build: Duration,
    encode_meta_write: Duration,
    encode_skip_write: Duration,
    encode_block_write: Duration,
    fst_insert: Duration,
    // Per-column phase totals (summed across columns; printed in the
    // [fts-finish] summary line at the end of finish_to).
    partition_flush: Duration,
    lex_rank_build: Duration,
    partition_sort: Duration,
    mmap_open: Duration,
    scratch_cleanup: Duration,
    // Whole-finish phase totals (printed once at end of finish_to).
    fst_close: Duration,
    postings_close: Duration,
    doc_lengths_emit: Duration,
    blob_copy: Duration,
}

impl FinishProfile {
    fn from_env() -> Self {
        Self {
            enabled: std::env::var_os("INFINO_FTS_PROFILE").is_some(),
            ..Self::default()
        }
    }
}

/// Per-(column, term) metadata header — 20 bytes, written immediately
/// before the term's skip table + posting blocks in the postings region.
/// `term_metadata_offset` (referenced from the FST value) points at the
/// start of this struct.
///
/// Layout:
///   off  0 ..  4 : df (u32) — bounded by n_docs per superfile
///   off  4 .. 12 : postings_offset (u64) — equals the term's metadata_offset;
///                  self-describing. u64 supports superfiles past 4 GiB
///                  (e.g. the 16 GB target).
///   off 12 .. 16 : postings_length (u32) — single term's bytes, well under
///                  4 G even at high df (≤ ~1 MB for the most common term in
///                  a 16 GB superfile).
///   off 16 .. 20 : num_blocks (u32)
///
/// `df`, `postings_length`, and `num_blocks` stay u32; only the absolute
/// offset into the postings region needs the full u64 range.
pub(crate) const TERM_META_SIZE: usize = 20;

/// Skip-table entry size in bytes.
pub(crate) const SKIP_ENTRY_SIZE: usize = 16;

/// Doc-lengths directory entry size in bytes (per column).
///
/// Layout:
///   off  0 ..  4 : column_id (u32)
///   off  4 .. 12 : doc_lengths_offset (u64) — absolute offset of this column's
///                  doc-lengths array in the FTS blob. u64 supports superfiles
///                  past 4 GiB.
///   off 12 .. 16 : avgdl_x1000 (u32) — avgdl × 1000, as an integer
///
/// Only the absolute offset needs u64; column_id and avgdl_x1000 stay
/// u32 (bounded by column count and doc length respectively).
pub(crate) const DOC_LENGTHS_ENTRY_SIZE: usize = 16;

/// Default per-column in-RAM accumulator budget before a column
/// flushes to spill files. Mirrors `VectorBuilder::spill_threshold_bytes`
/// (also 256 MiB by default). Builds whose every column stays below
/// this never touch the disk during `add_doc`.
///
/// Overridable per-builder via `FtsBuilder::set_spill_threshold_bytes(b)`.
pub const DEFAULT_SPILL_THRESHOLD_BYTES: usize = 256 * 1024 * 1024;

/// Default hash-partition count for the spill-backed postings accumulator.
///
/// In spill mode each `(term, doc_id, tf)` record is written to one
/// partition during `add_doc`; `finish_to` sorts and k-way-merges
/// partitions in global lex order. Higher values shrink the expected
/// per-partition size at the cost of more file handles.
///
/// Overridable per-builder via `FtsBuilder::set_spill_partitions(n)`.
pub const DEFAULT_SPILL_PARTITIONS: usize = 128;

/// Default in-memory budget per partition during the finish-time
/// sort pass. Partitions whose on-disk size exceeds this value are
/// sorted via external merge (chunked sort + k-way merge over
/// sorted spill files) rather than being fully materialised in RAM.
///
/// Overridable per-builder via `FtsBuilder::set_max_partition_bytes(b)`.
pub const DEFAULT_MAX_PARTITION_BYTES: u64 = 256 * 1024 * 1024;

/// Per-partition write buffer. 64 KiB matches the vector builder's
/// bucket writer budget and amortizes syscall cost without pinning
/// meaningful RAM.
const PARTITION_BUF_SIZE: usize = 64 * 1024;

/// Approximate per-record byte overhead in the in-RAM posting
/// accumulator. Used to drive the spill threshold; intentionally
/// rough — the threshold is a soft budget, not a hard cap.
///
/// - `~24 B`: per new term: FxHashMap entry header + `Vec<(u32,u32)>`
///   header + `Box<str>` header + small-alloc rounding.
/// - `+ term.len()`: term bytes.
/// - `+ 8 B`: per added posting: `(doc_id: u32, tf: u32)`.
const ACCUM_NEW_TERM_FIXED_BYTES: usize = 24;
const ACCUM_POSTING_BYTES: usize = 8;

/// Partition size (in triples) below which the lex-rank sort uses
/// `sort_unstable_by` instead of the counting/radix variant: under
/// this count the histogram allocation outweighs the algorithmic
/// savings.
const RADIX_SORT_MIN_TRIPLES: usize = 256;

/// Upper bound on the initial in-RAM chunk capacity (in triples)
/// during external merge sort. Caps the up-front `Vec` reservation
/// (~12 MiB of triples) when `max_partition_bytes` is large.
const EXTERNAL_MERGE_CHUNK_CAP_TRIPLES: usize = 1024 * 1024;

/// Number of triples buffered before each `write_all` when streaming
/// a sorted partition to disk. Amortizes the syscall cost (~48 KiB
/// per flush).
const SORT_OUTPUT_BATCH_TRIPLES: usize = 4096;

/// Per-column build-time state (scalar accounting only).
struct ColumnState {
    name: String,
    /// One u32 per doc (token count for this column), push order
    /// matches local_doc_id order.
    doc_lengths: Vec<u32>,
    /// Total token count across every doc in this column. Used for
    /// `avgdl = total_tokens / n_docs`.
    total_tokens: u64,
}

/// Per-column posting accumulator. Starts in `InRam` mode; transitions
/// to `Spilled` exactly once when this column's accumulated bytes
/// cross the builder's `spill_threshold_bytes`.
enum ColumnPostings {
    /// In-RAM term → posting list map. Small builds stay here forever.
    InRam {
        terms: FxHashMap<Box<str>, Vec<(u32, u32)>>,
        /// Estimated bytes held by `terms` — used to drive the spill
        /// threshold check. Approximate (see `ACCUM_*_BYTES`).
        bytes: usize,
    },
    /// Hash-partitioned spill files plus the per-column term
    /// interner. Records on disk are fixed-size 12-byte
    /// `(term_id_le, doc_id_le, tf_le)` triples — same shape as
    /// vector's raw-f32 spill, no per-record framing.
    ///
    /// `term_to_id` assigns a fresh `u32` ID to each distinct term
    /// the first time it's seen (during the threshold flush, then
    /// during subsequent `add_doc` calls). `id_to_term` is the
    /// reverse map used at `finish_to` time to recover the term
    /// bytes for FST emission. Both are bounded by the column's
    /// vocabulary, which is typically O(10^4 - 10^6) even on 10M-
    /// doc corpora — millions of bytes, not gigabytes.
    Spilled {
        partitions: Vec<SpillPartition>,
        term_to_id: TermIdMap,
        /// Per-id reverse lookup used by `finish_to` to recover term
        /// bytes for FST emit. Entries are `&'static str` slices
        /// borrowing from `term_arena` (see [`TermIdMap`] doc for
        /// the lifetime-extension invariant — same arena, same
        /// drop-order rules).
        id_to_term: Vec<&'static str>,
        /// Per-doc tf accumulator, indexed by `term_id`. Replaces the
        /// previous `FxHashMap<u32, u32>` scratch: term_ids are dense
        /// `0..vocab_size` (vocab ~10K on the bench Zipfian corpus, ≤
        /// few M on 10M-doc corpora), so a dense `Vec<u32>` keyed on
        /// `term_id` is a single array store per token (~5ns) vs a
        /// hashmap probe + entry update (~25-30ns). On the 1M-doc
        /// spill bench that's ~150M token-hits, so the per-call delta
        /// is the dominant `add_doc` cost saving.
        ///
        /// Memory: 4 bytes × vocab. 40 KiB at 10K vocab; a few MiB
        /// even at multi-M vocab — well inside the `O(max_partition_
        /// bytes)` budget. The Vec grows monotonically as `intern_
        /// term_id` mints new IDs; only ever read at the indices
        /// listed in `updated_terms`, so stale entries past the
        /// current `add_doc`'s set are ignored.
        dense_doc_tf: Vec<u32>,
        /// Set of `term_id`s that were written in the current
        /// `add_doc` call. Drained at the end of each call to emit
        /// triples and zero out `dense_doc_tf` slots — bounded by
        /// the per-doc unique-term count (≤ doc length), so a tiny
        /// `Vec<u32>` reused across calls.
        updated_terms: Vec<u32>,
        /// Per-column bump arena owning the term bytes. Backing
        /// store for the `&'static str` keys in `term_to_id` and
        /// entries in `id_to_term` (see [`TermIdMap`] doc).
        ///
        /// **Declared LAST in this variant so it drops LAST.**
        /// Rust drops struct fields in declaration order; the map
        /// + reverse vec drop first, both of which leave their
        ///   `&str` keys / entries bitwise-deallocated without
        ///   dereferencing, and only then does the arena's
        ///   `Drop` free the term bytes.
        term_arena: bumpalo::Bump,
    },
}

impl ColumnPostings {
    fn new() -> Self {
        Self::InRam {
            terms: FxHashMap::default(),
            bytes: 0,
        }
    }
    fn is_spilled(&self) -> bool {
        matches!(self, Self::Spilled { .. })
    }
}

/// Per-partition batch buffer used in front of `BufWriter` on the
/// hot `add_doc` spill path. Sized so it holds 341 fixed-12-byte
/// triples (≈ 4 KiB). The flush path is one `Vec::extend_from_slice
/// (&[u8; 12])` per posting plus one `BufWriter::write_all` per
/// full batch, vs the previous "one `BufWriter::write_all([u8; 12])`
/// per posting". On the 1M-doc bench this is ~440K BufWriter
/// branches instead of ~150M.
const SPILL_BATCH_TRIPLES: usize = 341;

struct SpillPartition {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    /// In-memory batch buffer flushed into `writer` when full.
    /// `add_doc` appends triples here in the hot path; `finish_to`'s
    /// flush-stage drains any partial buffer once per partition before
    /// the merge starts.
    ///
    /// Stored as `Vec<Triple>` rather than `Vec<u8>` so the hot-path
    /// `push_triple_batched` is a single `Vec::push` (one len-bump +
    /// 12-byte store) instead of `extend_from_slice` of three
    /// per-field `copy_from_slice` writes. On the 1M-doc spill bench
    /// that's ~135M call-sites in the hot loop; the per-call delta
    /// adds up.
    ///
    /// On flush the buffer is cast to `&[u8]` via `bytemuck::cast_slice`
    /// (LE hosts) — zero copy. BE hosts pay a per-triple byte-swap
    /// path. Capacity reserved to `SPILL_BATCH_TRIPLES` up front so
    /// the steady-state push is branch-free w.r.t. capacity.
    batch: Vec<Triple>,
}

/// Fixed on-disk record size in the spill files: 4 bytes `term_id`
/// + 4 bytes `doc_id` + 4 bytes `tf`, all little-endian.
///
/// This matches vector's "raw f32 bytes" spill strategy in shape:
/// fixed-size records, no framing, no variable-length payload.
/// `read_triples` reads N×12 bytes in one syscall-amortised batch
/// and reinterprets as `&[[u32; 3]]` via `bytemuck` — zero per-
/// record allocation, no UTF-8 validation, no `Box<str>` round-
/// trips.
const TRIPLE_BYTES: usize = std::mem::size_of::<Triple>();

/// Sortable + heap-mergeable posting triple. Matches the on-disk
/// layout (`[term_id_le, doc_id_le, tf_le]`) exactly so a partition
/// file's bytes can be reinterpreted as `&[Triple]` without copying
/// on little-endian hosts.
type Triple = [u32; 3];

#[inline(always)]
fn triple_term_id(t: &Triple) -> u32 {
    t[0]
}
#[inline(always)]
fn triple_doc_id(t: &Triple) -> u32 {
    t[1]
}
#[inline(always)]
fn triple_tf(t: &Triple) -> u32 {
    t[2]
}

/// Write one triple as little-endian bytes via a single 12-byte
/// `write_all`. Replaces the old four-call
/// `write_posting_record` — function-call overhead is ~4× lower
/// on `BufWriter` and the syscall amortisation through the 64-KiB
/// buffer is unchanged. Reserved for callers without a per-
/// partition batch buffer (the `flush_in_ram_to_partitions`
/// streaming path uses the buffered `push_triple_batched` path
/// instead — see below).
/// On big-endian hosts, the bulk `bytemuck::cast_slice` write
/// path in `write_triples_sorted` is replaced with a per-triple
/// scalar write to preserve the little-endian on-disk format.
/// On little-endian hosts (x86_64, aarch64) the bulk path is
/// the only one compiled, so this function isn't built at all.
#[cfg(not(target_endian = "little"))]
#[inline(always)]
fn write_triple<W: Write>(w: &mut W, term_id: u32, doc_id: u32, tf: u32) -> Result<(), BuildError> {
    let mut buf = [0u8; TRIPLE_BYTES];
    buf[0..4].copy_from_slice(&term_id.to_le_bytes());
    buf[4..8].copy_from_slice(&doc_id.to_le_bytes());
    buf[8..12].copy_from_slice(&tf.to_le_bytes());
    w.write_all(&buf)?;
    Ok(())
}

/// Append one fixed-12-byte triple to a `SpillPartition`'s in-
/// memory batch buffer, flushing the batch to the partition's
/// `BufWriter` only when it reaches `SPILL_BATCH_BYTES` (4 KiB).
///
/// This is the hot path on `add_doc` spill: ~150M calls at 1M
/// docs / 1500 tokens/doc. Each call is one `extend_from_slice`
/// of 12 bytes onto a `Vec<u8>` (the Vec's capacity is reserved
/// up-front, so the extend is a pure memcpy + len bump — no
/// branch on capacity); every 341st call also pays one
/// `BufWriter::write_all` + buffer clear. Replaces the old
/// "one `BufWriter::write_all([u8; 12])` per posting" pattern
/// which paid the `BufWriter`'s "does this fit in the inline
/// buffer?" branch on every single posting.
#[inline(always)]
fn push_triple_batched(
    partition: &mut SpillPartition,
    term_id: u32,
    doc_id: u32,
    tf: u32,
) -> Result<(), BuildError> {
    partition.batch.push([term_id, doc_id, tf]);
    if partition.batch.len() >= SPILL_BATCH_TRIPLES {
        flush_partition_batch(partition)?;
    }
    Ok(())
}

/// Drain any pending bytes in a partition's batch buffer into its
/// `BufWriter`. Called from the hot path when the batch fills and
/// from `finish_to`'s flush stage so partial buffers reach disk
/// before the merge starts.
#[inline]
fn flush_partition_batch(partition: &mut SpillPartition) -> Result<(), BuildError> {
    if partition.batch.is_empty() {
        return Ok(());
    }
    let writer = partition
        .writer
        .as_mut()
        .expect("partition writer is open before finish");
    // LE host: zero-copy cast from `&[Triple]` to `&[u8]`. BE host:
    // per-triple swap path keeps the on-disk file little-endian so
    // the LE fast read path stays valid cross-arch.
    #[cfg(target_endian = "little")]
    {
        writer.write_all(bytemuck::cast_slice::<Triple, u8>(&partition.batch))?;
    }
    #[cfg(not(target_endian = "little"))]
    {
        for t in &partition.batch {
            write_triple(writer, t[0], t[1], t[2])?;
        }
    }
    partition.batch.clear();
    Ok(())
}

/// Read every triple in `path` into a `Vec<Triple>`. The file is
/// laid out as a contiguous run of 12-byte LE triples, so on a
/// little-endian host the read is a single batched `read_to_end`
/// followed by a `bytemuck` reinterpretation (zero per-record
/// allocation, zero UTF-8 validation, no `Box<str>` chasing).
///
/// On a big-endian host the same bytes are read but each triple
/// is byte-swapped on the way in — kept behind a `cfg` so x86_64
/// and arm64 hit the fast cast path.
fn read_partition_triples(path: &Path) -> Result<Vec<Triple>, BuildError> {
    let mut bytes = Vec::new();
    let mut f = File::open(path)?;
    f.read_to_end(&mut bytes)?;
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    if bytes.len() % TRIPLE_BYTES != 0 {
        return Err(BuildError::Io(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "spill partition {path:?} length {} not a multiple of {}",
                bytes.len(),
                TRIPLE_BYTES
            ),
        )));
    }
    #[cfg(target_endian = "little")]
    {
        // Zero-copy bytes → triples cast on LE hosts. `bytemuck`
        // gates this on `Pod` alignment; `Vec<u8>::read_to_end`
        // returns a `Vec` whose buffer is aligned at least to
        // `align_of::<usize>()` (8 bytes on x86_64), so the cast
        // to `[u32; 3]` (alignment 4) is sound.
        let triples: &[Triple] = bytemuck::try_cast_slice(&bytes).map_err(|_| {
            BuildError::Io(std::io::Error::new(
                ErrorKind::InvalidData,
                "bytemuck: spill bytes failed alignment for &[Triple]",
            ))
        })?;
        Ok(triples.to_vec())
    }
    #[cfg(not(target_endian = "little"))]
    {
        let n = bytes.len() / TRIPLE_BYTES;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * TRIPLE_BYTES;
            let t = [
                u32::from_le_bytes(
                    bytes[off..off + 4]
                        .try_into()
                        .expect("invariant: 4-byte triple field"),
                ),
                u32::from_le_bytes(
                    bytes[off + 4..off + 8]
                        .try_into()
                        .expect("invariant: 4-byte triple field"),
                ),
                u32::from_le_bytes(
                    bytes[off + 8..off + 12]
                        .try_into()
                        .expect("invariant: 4-byte triple field"),
                ),
            ];
            out.push(t);
        }
        Ok(out)
    }
}

/// Build the two lex-order permutations between `term_id` and
/// lex rank over `id_to_term`:
///
/// * `lex_rank[term_id] = rank` (forward map, used for sort keys
///   in the per-partition sort).
/// * `term_id_in_lex_order[rank] = term_id` (inverse map; the
///   sorted permutation itself, used by the finish-time merge to
///   walk term_ids in global lex order without any heap
///   arbitration).
///
/// Both come from the same `sort_unstable_by` pass over `[0..n)`,
/// so producing both is essentially free. Bounded by the column's
/// vocabulary; tiny even at 10M docs.
fn build_lex_rank(id_to_term: &[&str]) -> (Vec<u32>, Vec<u32>) {
    let n = id_to_term.len();
    let mut by_lex: Vec<u32> = (0..n as u32).collect();
    by_lex.sort_unstable_by(|&a, &b| {
        id_to_term[a as usize]
            .as_bytes()
            .cmp(id_to_term[b as usize].as_bytes())
    });
    let mut rank = vec![0u32; n];
    for (r, id) in by_lex.iter().enumerate() {
        rank[*id as usize] = r as u32;
    }
    (rank, by_lex)
}

#[inline(always)]
fn compute_hash<Q: Hash + ?Sized, S: BuildHasher>(hash_builder: &S, key: &Q) -> u64 {
    hash_builder.hash_one(key)
}

/// Intern `term` in the spill-mode column vocabulary and return its
/// stable `term_id`.
///
/// This is the hot spill `add_doc` lookup. Use hashbrown's raw entry
/// API so miss handling reuses the hash computed for lookup instead
/// of doing `get(term)` (probe once) followed by `insert(boxed, id)`
/// (rehash + probe again). Hit behaviour remains one borrowed lookup
/// with no allocation; misses copy the term bytes into the per-column
/// `term_arena: bumpalo::Bump` (one bump-pointer advance + memcpy, no
/// heap allocator round-trip) and store the resulting `&str` slice in
/// both the map and `id_to_term` (a reference copy, no second heap
/// alloc — replaces the prior `Box<str>` + `boxed.clone()` pattern
/// which paid two heap allocations per miss).
///
/// The on-hit cost benefit is the more important change at scale:
/// every intern hit's byte-compare inside `raw_entry_mut` previously
/// dereferenced an arbitrary `Box<str>` whose backing bytes sat
/// wherever the global allocator placed it (typically scattered
/// across cache lines, often L2-cold for vocabularies >~1 KB worth
/// of total key bytes). Arena-backed keys live densely in a single
/// contiguous `bumpalo::Bump` so the byte-compare hits L1/L2 the
/// vast majority of the time, regardless of vocab size.
///
/// Returns `(term_id, is_new)`. `is_new == true` iff this call minted
/// a brand-new id (`term_to_id` previously had no entry for `term`).
/// Callers that keep a per-id parallel array (e.g. `dense_doc_tf` on
/// the spill path) use this to gate the grow check off the hot path:
/// after the InRam→Spilled transition pre-grows the parallel array
/// to `id_to_term.len()`, the array only needs to be extended when a
/// fresh id is minted, which is once per novel term per column —
/// not once per token.
///
/// SAFETY (invariants enforced by callers and the `Spilled` variant
/// struct layout, see [`TermIdMap`] doc): the `'static` lifetime tag
/// on the keys is a lie — the real lifetime is that of `arena`. The
/// transmute below is sound iff (a) `arena` outlives both
/// `term_to_id` and `id_to_term`, and (b) no key dereference happens
/// after `arena`'s `Drop` runs. Both hold by construction: `arena`
/// is declared *after* the map and reverse-vec in `Spilled` so it
/// drops *last*, and the `Drop` impls for `HbHashMap<&'static str,
/// u32, _>` and `Vec<&'static str>` never dereference their
/// `&str` keys/entries (no `Drop` to run on a reference).
#[inline(always)]
fn intern_term_id(
    term_to_id: &mut TermIdMap,
    id_to_term: &mut Vec<&'static str>,
    arena: &bumpalo::Bump,
    term: &str,
) -> (u32, bool) {
    let hash = compute_hash(term_to_id.hasher(), term);
    match term_to_id
        .raw_entry_mut()
        .from_hash(hash, |existing| *existing == term)
    {
        RawEntryMut::Occupied(entry) => (*entry.get(), false),
        RawEntryMut::Vacant(entry) => {
            let id = id_to_term.len() as u32;
            // Bump-arena copy: one pointer advance + memcpy of the
            // term bytes, returning a `&str` whose lifetime is tied
            // to `&arena`'s borrow scope.
            let arena_str: &str = arena.alloc_str(term);
            // SAFETY: see function-level doc. The `'static` tag is
            // a lie; the real lifetime is `arena`'s, which the
            // `Spilled` variant guarantees outlives the map +
            // id_to_term via field drop order.
            let static_str: &'static str = unsafe { std::mem::transmute(arena_str) };
            id_to_term.push(static_str);
            entry.insert_hashed_nocheck(hash, static_str, id);
            (id, true)
        }
    }
}

/// Min-heap entry for the k-way merge over sorted partition chunks.
/// Sort key is a packed `u64 = (lex_rank as u64) << 32 | doc_id as
/// u64` — natural u64 ordering matches `(lex_rank, doc_id)` lex
/// order, so the heap comparator is a single u64 compare per pop.
/// Ordering is inverted (heap returns the *smallest* sort key
/// first) by implementing `Ord` reversed; `BinaryHeap` is a max-
/// heap.
struct MergeEntry {
    /// `(lex_rank as u64) << 32 | doc_id as u64`.
    sort_key: u64,
    /// Original term_id (used at emit time to look up the term
    /// bytes via `id_to_term`).
    term_id: u32,
    tf: u32,
    reader_idx: usize,
}

impl PartialEq for MergeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for MergeEntry {}
impl PartialOrd for MergeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MergeEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so the largest "smallest" wins on pop — gives
        // BinaryHeap min-heap behaviour over (lex_rank, doc_id).
        other
            .sort_key
            .cmp(&self.sort_key)
            .then(other.reader_idx.cmp(&self.reader_idx))
    }
}

#[inline(always)]
fn pack_sort_key(lex_rank: u32, doc_id: u32) -> u64 {
    ((lex_rank as u64) << 32) | (doc_id as u64)
}

/// Iterator producing sorted triples (by `(lex_rank, doc_id)`) for
/// one partition.
///
/// `InMemory` is the small-partition path: the whole partition fits
/// in `max_partition_bytes` of RAM, so it's read once, sorted in
/// place, and drained.
///
/// `Merge` is the over-budget path: the partition is streamed in
/// chunks of `max_partition_bytes`, each chunk sorted in RAM and
/// spilled to a sorted-chunk side file under the scratch directory,
/// and then k-way-merged via a `BinaryHeap` of cursors so the
/// finish-time sort never holds more than one chunk plus one
/// record per chunk file at a time.
enum PartitionIter {
    InMemory(std::vec::IntoIter<Triple>),
    Merge {
        readers: Vec<BufReader<File>>,
        heap: BinaryHeap<MergeEntry>,
        /// Sorted-chunk files; kept alive so their inodes don't
        /// get reaped before iteration finishes.
        _chunk_paths: Vec<PathBuf>,
    },
}

impl PartitionIter {
    /// Pull the next sorted triple from this partition, looking up
    /// the sort key via `lex_rank` when refilling a merge cursor
    /// (so the heap stays minimal — only sort_key + tf + term_id
    /// + reader_idx).
    fn next_with(&mut self, lex_rank: &[u32]) -> Option<Result<Triple, BuildError>> {
        match self {
            PartitionIter::InMemory(it) => it.next().map(Ok),
            PartitionIter::Merge { readers, heap, .. } => {
                let MergeEntry {
                    sort_key,
                    term_id,
                    tf,
                    reader_idx,
                } = heap.pop()?;
                // Low 32 bits of the packed key carry doc_id.
                let popped: Triple = [term_id, sort_key as u32, tf];
                match read_one_triple(&mut readers[reader_idx]) {
                    Ok(Some(next_t)) => {
                        let next_id = triple_term_id(&next_t);
                        let next_doc = triple_doc_id(&next_t);
                        let key = pack_sort_key(lex_rank[next_id as usize], next_doc);
                        heap.push(MergeEntry {
                            sort_key: key,
                            term_id: next_id,
                            tf: triple_tf(&next_t),
                            reader_idx,
                        });
                    }
                    Ok(None) => { /* chunk drained */ }
                    Err(e) => return Some(Err(e)),
                }
                Some(Ok(popped))
            }
        }
    }
}

/// Read a single 12-byte triple from a sorted-chunk file. Returns
/// `Ok(None)` on clean EOF.
fn read_one_triple<R: Read>(r: &mut R) -> Result<Option<Triple>, BuildError> {
    let mut buf = [0u8; TRIPLE_BYTES];
    match r.read_exact(&mut buf) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(BuildError::Io(e)),
    }
    Ok(Some([
        u32::from_le_bytes(buf[0..4].try_into().expect("slice len 4")),
        u32::from_le_bytes(buf[4..8].try_into().expect("slice len 4")),
        u32::from_le_bytes(buf[8..12].try_into().expect("slice len 4")),
    ]))
}

/// Write a slice of triples to a sorted-chunk file. Single
/// `write_all` per chunk on LE hosts via `bytemuck` byte-cast.
fn write_triples_sorted(triples: &[Triple], path: &Path) -> Result<(), BuildError> {
    let mut w = BufWriter::with_capacity(PARTITION_BUF_SIZE, File::create(path)?);
    #[cfg(target_endian = "little")]
    {
        let bytes: &[u8] = bytemuck::cast_slice(triples);
        w.write_all(bytes)?;
    }
    #[cfg(not(target_endian = "little"))]
    {
        for t in triples {
            write_triple(&mut w, t[0], t[1], t[2])?;
        }
    }
    w.flush()?;
    Ok(())
}

fn spill_sorted_chunk(
    chunk: &mut Vec<Triple>,
    scratch_dir: &Path,
    partition_label: &str,
    chunk_idx: usize,
    lex_rank: &[u32],
    out_paths: &mut Vec<PathBuf>,
) -> Result<(), BuildError> {
    radix_sort_triples_by_lex_rank(chunk, lex_rank);
    let path = scratch_dir.join(format!("{partition_label}_sorted{chunk_idx}.bin"));
    write_triples_sorted(chunk, &path)?;
    chunk.clear();
    #[cfg(test)]
    finish_debug::record_chunk_path(&path);
    out_paths.push(path);
    Ok(())
}

/// Test-only observer for the external-merge chunk-file path.
///
/// Used by `external_merge_path_matches_in_memory_path_byte_for_
/// byte` to gate that the over-budget partition branch in
/// `open_partition_sorted` actually runs (and therefore that the
/// reviewer's "is the external merge exercised?" question is
/// answered by a positive test, not just blob-equality). Lives in
/// `#[cfg(test)]` so production binaries pay zero runtime cost.
#[cfg(test)]
mod finish_debug {
    use std::cell::RefCell;
    use std::path::{Path, PathBuf};

    thread_local! {
        // Thread-local so concurrent `cargo test` workers do not
        // cross-pollute each other's observed chunk lists. Tests
        // that drive the external-merge path build + finish on
        // their own worker thread, so this stays isolated.
        static OBSERVED_CHUNKS: RefCell<Vec<PathBuf>> = const { RefCell::new(Vec::new()) };
    }

    /// Clear the observed-chunks log. Tests call this before the
    /// build whose external-merge activity they want to inspect.
    pub fn reset() {
        OBSERVED_CHUNKS.with(|c| c.borrow_mut().clear());
    }

    /// Called by `spill_sorted_chunk` for every sorted-chunk file
    /// written during external-merge.
    pub fn record_chunk_path(path: &Path) {
        OBSERVED_CHUNKS.with(|c| c.borrow_mut().push(path.to_path_buf()));
    }

    /// Snapshot of the observed-chunk-path list.
    pub fn observed() -> Vec<PathBuf> {
        OBSERVED_CHUNKS.with(|c| c.borrow().clone())
    }
}

/// Sort `triples` in place by `(lex_rank[term_id], doc_id)` using a
/// single-pass O(n) stable counting sort over `lex_rank`.
///
/// **Why a counting sort suffices** (and beats every comparison /
/// LSB-radix variant we tried):
///
/// 1. The sort key is *bounded*. `lex_rank[term_id]` is in
///    `0..vocab_size`. The bench Zipfian column has ~10K vocab; even
///    a 10M-doc supertable column tops out around a few million —
///    the counts table fits comfortably in L2 either way.
/// 2. The secondary key is *free*. Within a partition file, all
///    triples for a fixed `term_id` are appended in strictly
///    increasing `doc_id` order (`add_doc` is called with monotonic
///    `local_doc_id` per `column_id`, and each call emits one
///    triple per unique term in iteration order over
///    `updated_terms`, with all triples for that doc emitted
///    contiguously). So a *stable* sort on `lex_rank[term_id]`
///    leaves the within-rank order as the original `doc_id` order
///    — exactly the `(lex_rank, doc_id)` order the finish-time
///    lex-order partition traversal needs.
/// 3. Counting sort is **one pass to histogram + one pass to
///    scatter**. No `O(log n)` compare chain (pdqsort), no 5–8
///    LSB-byte passes (radix), no comparator chasing `lex_rank`
///    twice per call. Two reads of every triple and one write.
///
/// **Memory shape**: `counts: Vec<u32>` of length
/// `vocab_size + 1` (~40 KiB at 10K vocab), plus the `out: Vec
/// <Triple>` of the same length as `triples` — the same size as the
/// radix variant's parallel-array workspace, just a single
/// allocation instead of three.
///
/// Falls back to `sort_unstable_by` for tiny inputs where the counts
/// allocation outweighs the algorithmic savings.
fn radix_sort_triples_by_lex_rank(triples: &mut Vec<Triple>, lex_rank: &[u32]) {
    let n = triples.len();
    if n < RADIX_SORT_MIN_TRIPLES {
        triples.sort_unstable_by(|a, b| {
            lex_rank[triple_term_id(a) as usize]
                .cmp(&lex_rank[triple_term_id(b) as usize])
                .then(triple_doc_id(a).cmp(&triple_doc_id(b)))
        });
        return;
    }

    let vocab_size = lex_rank.len();
    // Histogram + prefix-sum table. `+1` so the final entry holds
    // `n` after prefix-sum, which simplifies the bounds check below.
    let mut offsets: Vec<u32> = vec![0u32; vocab_size + 1];

    // Pass 1: count triples per `lex_rank`. The hot inner step is
    // `lex_rank[term_id as usize]` — at `target-cpu=x86-64-v3` LLVM
    // lowers the strided load to an AVX2 `vpgatherdd` (8 ranks per
    // instruction). The histogram `offsets[rank] += 1` has a self-
    // dependency that prevents auto-vectorisation, but the loop
    // body is short enough that out-of-order issue absorbs the
    // latency.
    //
    // SAFETY (every `get_unchecked` in this function): term ids are
    // dense in `[0, vocab_size)` and `lex_rank` has `vocab_size`
    // entries, so `t[0] < lex_rank.len()`. `rank < vocab_size` indexes
    // `offsets` (length `vocab_size + 1`); the prefix sum keeps every
    // scatter `dst < n` (`out.len()`), and each `offsets[rank]` is
    // bumped at most its histogram count, so all accesses are in bounds.
    for t in triples.iter() {
        let rank = unsafe { *lex_rank.get_unchecked(t[0] as usize) } as usize;
        offsets[rank] = offsets[rank].wrapping_add(1);
    }

    // Pass 2 (cheap): convert counts into starting-offset table by
    // exclusive prefix sum. Runs over `vocab_size + 1` entries
    // (~10K on the bench) — well below the partition sort cost.
    let mut sum: u32 = 0;
    for c in offsets.iter_mut() {
        let tmp = *c;
        *c = sum;
        sum = sum.wrapping_add(tmp);
    }
    debug_assert_eq!(sum as usize, n, "histogram total != triple count");

    // Pass 3: scatter into `out`. Each triple lands at
    // `offsets[rank]`, then we bump that slot so the next triple
    // for the same rank lands immediately after. Because we walk
    // `triples` in arrival order and arrival order is `(doc_id,
    // term_id_within_doc)`, the within-rank order in `out` is the
    // partition's `doc_id` order — i.e. `(lex_rank, doc_id)`.
    let mut out: Vec<Triple> = vec![[0u32; 3]; n];
    for t in triples.iter() {
        let rank = unsafe { *lex_rank.get_unchecked(t[0] as usize) } as usize;
        let dst = unsafe { *offsets.get_unchecked(rank) } as usize;
        unsafe {
            *out.get_unchecked_mut(dst) = *t;
            *offsets.get_unchecked_mut(rank) = (dst as u32).wrapping_add(1);
        }
    }

    *triples = out;
}

/// Open a partition as a sorted-triple iterator. Picks the in-
/// memory path when the on-disk partition is at or below
/// `max_partition_bytes` and the external-merge path when it
/// isn't.
fn open_partition_sorted(
    partition_path: &Path,
    max_partition_bytes: u64,
    scratch_dir: &Path,
    partition_label: &str,
    lex_rank: &[u32],
) -> Result<PartitionIter, BuildError> {
    let len = std::fs::metadata(partition_path)?.len();
    if len <= max_partition_bytes {
        let mut triples = read_partition_triples(partition_path)?;
        radix_sort_triples_by_lex_rank(&mut triples, lex_rank);
        return Ok(PartitionIter::InMemory(triples.into_iter()));
    }

    // External merge: stream the partition in
    // `max_partition_bytes`-sized triple chunks, sort each chunk
    // in RAM, write sorted-chunk spill, then k-way merge. The
    // resident peak during this path is one chunk's triples plus
    // one triple per chunk file in the heap.
    let chunk_triples = (max_partition_bytes as usize) / TRIPLE_BYTES;
    let mut sorted_chunk_paths: Vec<PathBuf> = Vec::new();
    let mut r = BufReader::with_capacity(PARTITION_BUF_SIZE, File::open(partition_path)?);
    let mut chunk: Vec<Triple> =
        Vec::with_capacity(chunk_triples.min(EXTERNAL_MERGE_CHUNK_CAP_TRIPLES));
    let mut chunk_idx: usize = 0;
    while let Some(t) = read_one_triple(&mut r)? {
        chunk.push(t);
        if chunk.len() >= chunk_triples {
            spill_sorted_chunk(
                &mut chunk,
                scratch_dir,
                partition_label,
                chunk_idx,
                lex_rank,
                &mut sorted_chunk_paths,
            )?;
            chunk_idx += 1;
        }
    }
    if !chunk.is_empty() {
        spill_sorted_chunk(
            &mut chunk,
            scratch_dir,
            partition_label,
            chunk_idx,
            lex_rank,
            &mut sorted_chunk_paths,
        )?;
    }

    let mut readers: Vec<BufReader<File>> = Vec::with_capacity(sorted_chunk_paths.len());
    for p in &sorted_chunk_paths {
        readers.push(BufReader::with_capacity(PARTITION_BUF_SIZE, File::open(p)?));
    }
    let mut heap: BinaryHeap<MergeEntry> = BinaryHeap::with_capacity(readers.len());
    for (idx, reader) in readers.iter_mut().enumerate() {
        if let Some(t) = read_one_triple(reader)? {
            let term_id = triple_term_id(&t);
            let doc_id = triple_doc_id(&t);
            heap.push(MergeEntry {
                sort_key: pack_sort_key(lex_rank[term_id as usize], doc_id),
                term_id,
                tf: triple_tf(&t),
                reader_idx: idx,
            });
        }
    }
    Ok(PartitionIter::Merge {
        readers,
        heap,
        _chunk_paths: sorted_chunk_paths,
    })
}

pub struct FtsBuilder {
    tokenizer: Arc<dyn Tokenizer>,
    columns: Vec<ColumnState>,
    /// Per-column posting accumulator. Each entry starts in
    /// `ColumnPostings::InRam` and transitions to `Spilled` exactly
    /// once when its accumulated bytes cross `spill_threshold_bytes`.
    /// Mirror of `VectorBuilder`'s `pre_spill_buffer` + `spill`.
    postings: Vec<ColumnPostings>,
    /// Scratch directory that owns all posting + FST spill files.
    /// Lazily populated — small builds (every column stays in RAM)
    /// never write here. Dropped after `finish_to` copies its
    /// contents into the output writer.
    scratch_dir: tempfile::TempDir,
    /// Per-column in-RAM accumulator budget. When a column's `InRam`
    /// state's `bytes` would cross this on an `add_doc`, that column
    /// is flushed to spill files and transitions to `Spilled` for the
    /// rest of the build. Default: `DEFAULT_SPILL_THRESHOLD_BYTES`.
    spill_threshold_bytes: usize,
    /// Number of hash partitions used in spill mode. Must be ≥ 1.
    /// Default: `DEFAULT_SPILL_PARTITIONS`.
    spill_partitions: usize,
    /// Per-partition in-RAM sort budget at finish time. Partitions
    /// exceeding this size on disk are sorted via external merge.
    /// Default: `DEFAULT_MAX_PARTITION_BYTES`.
    max_partition_bytes: u64,
    /// Tracks the number of distinct local_doc_ids ever seen by add_doc.
    /// Used as `n_docs` for the FTS blob header.
    n_docs: u32,
    /// Per-shard bump arena reused across every `add_doc` call.
    /// Holds the transient `&str` keys of the per-doc tf hashmap.
    /// Reset at the top of each `add_doc` so the leftover bytes are
    /// invalidated before the next allocation; `Bump::reset` keeps
    /// the largest chunk so subsequent docs allocate in-place
    /// without going back to the system allocator.
    ///
    /// Only the in-RAM arm of `add_doc` consumes this — the spill
    /// arm interns tokens straight into the column's `term_to_id`
    /// and dedupes via a dense `Vec<u32>` (kept inside `ColumnPostings
    /// ::Spilled`) keyed by `term_id` instead.
    bump: bumpalo::Bump,
}

impl FtsBuilder {
    /// Construct a builder with the default scratch directory
    /// (under `$TMPDIR` via `tempfile::tempdir()`) and the default
    /// 256 MiB spill threshold. Mirror of `VectorBuilder::new`.
    ///
    /// Panics if creating the scratch tempdir fails — same policy
    /// as `VectorBuilder::new` for the same reason (no realistic
    /// recovery at construction time, preserves existing public
    /// API). Operators running large builds should prefer
    /// [`Self::with_scratch`] pointing at an instance-store NVMe
    /// partition.
    pub fn new(tokenizer: Arc<dyn Tokenizer>) -> Self {
        let scratch_dir = tempfile::tempdir().expect("create FtsBuilder scratch tempdir");
        Self::from_parts(tokenizer, scratch_dir)
    }

    /// Construct a builder with `scratch` as the scratch root. The
    /// directory must already exist and be writable. Used for
    /// benchmarks + production deployments that want to pin scratch
    /// to instance-store NVMe (`/mnt/nvme0/infino-build`, etc.)
    /// instead of the default `$TMPDIR` (which on EC2 is typically
    /// EBS-backed `/tmp`).
    ///
    /// Mirror of `VectorBuilder::with_scratch`, same return type
    /// (`Result<Self, BuildError>`).
    pub fn with_scratch(
        tokenizer: Arc<dyn Tokenizer>,
        scratch: PathBuf,
    ) -> Result<Self, BuildError> {
        let scratch_dir = tempfile::Builder::new()
            .prefix("infino-fts-")
            .tempdir_in(&scratch)?;
        Ok(Self::from_parts(tokenizer, scratch_dir))
    }

    fn from_parts(tokenizer: Arc<dyn Tokenizer>, scratch_dir: tempfile::TempDir) -> Self {
        Self {
            tokenizer,
            columns: Vec::new(),
            postings: Vec::new(),
            scratch_dir,
            spill_threshold_bytes: DEFAULT_SPILL_THRESHOLD_BYTES,
            spill_partitions: DEFAULT_SPILL_PARTITIONS,
            max_partition_bytes: DEFAULT_MAX_PARTITION_BYTES,
            n_docs: 0,
            bump: bumpalo::Bump::new(),
        }
    }

    /// Override the per-column in-RAM accumulator budget. Once a
    /// column's accumulated posting bytes cross this threshold, that
    /// column flushes its in-memory map to spill files and runs the
    /// remainder of the build in spill mode.
    ///
    /// Mirror of `VectorBuilder::set_spill_threshold_bytes`, same
    /// `&mut self` setter shape. Threshold is read live at
    /// `add_doc` time, so changes after `register_column` *do*
    /// apply — unlike vector, which snapshots into each
    /// `ColumnState` at registration time.
    pub fn set_spill_threshold_bytes(&mut self, threshold: usize) {
        assert!(
            threshold > 0,
            "FtsBuilder: spill_threshold_bytes must be > 0"
        );
        self.spill_threshold_bytes = threshold;
    }

    /// Override the hash-partition count used when a column spills.
    /// Must be called *before* the first `register_column` — today
    /// partition files are created lazily on first spill, so a
    /// post-registration call would also work, but we require the
    /// pre-registration call for forward-compat with eager-create
    /// modes.
    pub fn set_spill_partitions(&mut self, n: usize) -> Result<(), BuildError> {
        if !self.columns.is_empty() {
            return Err(BuildError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "FtsBuilder::set_spill_partitions must be called before any register_column",
            )));
        }
        if n == 0 {
            return Err(BuildError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "FtsBuilder: spill_partitions must be ≥ 1",
            )));
        }
        // Partition selection is `term_id & (n - 1)` on the hot
        // spill path; that's only correct (uniform partitioning)
        // when `n` is a power of two. Reject non-PO2 values so the
        // hot path stays branch-free instead of falling back to
        // modulo.
        if !n.is_power_of_two() {
            return Err(BuildError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("FtsBuilder: spill_partitions must be a power of two; got {n}"),
            )));
        }
        self.spill_partitions = n;
        Ok(())
    }

    /// Override the per-partition in-RAM sort budget. Partitions
    /// whose on-disk size exceeds this value are sorted via external
    /// merge at finish time. Safe to call at any point before
    /// `finish`.
    pub fn set_max_partition_bytes(&mut self, bytes: u64) {
        assert!(bytes > 0, "FtsBuilder: max_partition_bytes must be > 0");
        self.max_partition_bytes = bytes;
    }

    /// Register an FTS column up-front. Returns its `column_id` (its
    /// index in declaration order).
    pub fn register_column(&mut self, name: String) -> Result<u32, BuildError> {
        if name.as_bytes().contains(&FST_SEPARATOR) {
            return Err(BuildError::ReservedSeparatorInColumnName(name));
        }
        if name.starts_with(format::RESERVED_PREFIX) {
            return Err(BuildError::ReservedPrefixInColumnName(name));
        }
        if self.columns.iter().any(|c| c.name == name) {
            return Err(BuildError::DuplicateColumnName(name));
        }
        let column_id = self.columns.len() as u32;
        self.columns.push(ColumnState {
            name,
            doc_lengths: Vec::new(),
            total_tokens: 0,
        });
        self.postings.push(ColumnPostings::new());
        Ok(column_id)
    }

    /// Open spill partition files for a column and return them.
    /// Called the first time a column's in-RAM accumulator crosses
    /// `spill_threshold_bytes`.
    fn open_partitions_for_column(
        scratch_dir: &Path,
        column_id: u32,
        n_partitions: usize,
    ) -> Result<Vec<SpillPartition>, BuildError> {
        let mut partitions = Vec::with_capacity(n_partitions);
        for partition in 0..n_partitions {
            let path = scratch_dir.join(format!("fts_col{column_id}_part{partition}.bin"));
            let file = File::create(&path)?;
            partitions.push(SpillPartition {
                path,
                writer: Some(BufWriter::with_capacity(PARTITION_BUF_SIZE, file)),
                batch: Vec::with_capacity(SPILL_BATCH_TRIPLES),
            });
        }
        Ok(partitions)
    }

    /// Drain an in-RAM term → postings map into spill partitions,
    /// assigning a fresh `term_id` per term as it's first seen.
    /// Used once per column at the moment that column crosses the
    /// spill threshold. After this returns, the map is empty
    /// (already dropped by the caller) and all records live in the
    /// partition files as fixed-size 12-byte triples.
    ///
    /// `term_to_id` + `id_to_term` are populated with this
    /// column's vocabulary; subsequent `add_doc` calls reuse them
    /// to intern any new terms they see.
    fn flush_in_ram_to_partitions(
        terms: FxHashMap<Box<str>, Vec<(u32, u32)>>,
        partitions: &mut [SpillPartition],
        term_to_id: &mut TermIdMap,
        id_to_term: &mut Vec<&'static str>,
        arena: &bumpalo::Bump,
    ) -> Result<(), BuildError> {
        let n_part = partitions.len();
        debug_assert!(
            n_part.is_power_of_two(),
            "spill_partitions must be a power of 2; got {n_part}"
        );
        let mask = n_part - 1;
        for (term, postings) in terms {
            // Drain-time intern: every drained term is by construction
            // a fresh id (term_to_id was empty when the transition
            // started). The `is_new` flag is unused here; the dense-
            // array sizing is handled by the caller pre-growing
            // `dense_doc_tf` to `id_to_term.len()` after this returns.
            let (term_id, _is_new) = intern_term_id(term_to_id, id_to_term, arena, &term);
            let p = (term_id as usize) & mask;
            for (doc_id, tf) in postings {
                push_triple_batched(&mut partitions[p], term_id, doc_id, tf)?;
            }
        }
        Ok(())
    }

    /// Index `text` for `(column_id, local_doc_id)`.
    ///
    /// Caller must call this once per (doc, registered FTS column) pair,
    /// with monotonically increasing `local_doc_id` per column. Multiple
    /// occurrences of the same term in `text` increment the term-frequency
    /// for that doc.
    pub fn add_doc(
        &mut self,
        column_id: u32,
        local_doc_id: u32,
        text: &str,
    ) -> Result<(), BuildError> {
        let col_idx = column_id as usize;
        if col_idx >= self.columns.len() {
            return Err(BuildError::FtsColumnTypeInvalid {
                column: format!("(unregistered column_id {column_id})"),
                actual: "n/a".to_string(),
            });
        }

        // The contract is that `local_doc_id` increments by 1 per
        // (per-column) call, starting at 0. `finish()` indexes
        // `col.doc_lengths[doc_id]` with a doc_id from the posting list,
        // so the doc_lengths vec must be in sync with the local_doc_id
        // axis. Catch contract violations early in debug builds;
        // release skips the check.
        debug_assert!(
            local_doc_id as usize == self.columns[col_idx].doc_lengths.len(),
            "FtsBuilder::add_doc: local_doc_id ({local_doc_id}) must equal \
             this column's next index ({}); doc_ids must be consecutive \
             from 0 within a column",
            self.columns[col_idx].doc_lengths.len(),
        );

        // Dispatch by variant. Re-destructured inside each helper
        // so the helper can field-split-borrow `self.postings`,
        // `self.columns`, and (for in-RAM) `self.bump` without
        // routing through an enum field reference held across the
        // call.
        if self.postings[col_idx].is_spilled() {
            self.add_doc_spilled(col_idx, local_doc_id, text)
        } else {
            self.add_doc_inram(col_idx, local_doc_id, text)
        }
    }

    /// Spilled-mode hot path. Per-token cost: intern lookup + dense-
    /// array tf bump; per-doc cost: drain `updated_terms` into the
    /// partition writers as 12-byte triples. Invariant maintained:
    /// `dense_doc_tf.len() == id_to_term.len()` and
    /// `dense_doc_tf[tid] != 0 iff tid in updated_terms`. See the
    /// inline `SAFETY` comments for the bounds-check elisions.
    #[inline(always)]
    fn add_doc_spilled(
        &mut self,
        col_idx: usize,
        local_doc_id: u32,
        text: &str,
    ) -> Result<(), BuildError> {
        // Downcast once per call so the tokenize_each_inline fast
        // path is reachable. See the original `add_doc` for the
        // ~150M dyn-dispatch savings this buys on the 1M-doc bench.
        let tokenizer = &self.tokenizer;
        let ascii_tok = tokenizer
            .as_ref()
            .as_any()
            .downcast_ref::<AsciiLowerTokenizer>();
        let mut tokens_in_doc: u64 = 0;

        let col_post = &mut self.postings[col_idx];
        let (partitions, term_to_id, id_to_term, dense_doc_tf, updated_terms, term_arena) =
            match col_post {
                ColumnPostings::Spilled {
                    partitions,
                    term_to_id,
                    id_to_term,
                    dense_doc_tf,
                    updated_terms,
                    term_arena,
                } => (
                    partitions,
                    term_to_id,
                    id_to_term,
                    dense_doc_tf,
                    updated_terms,
                    term_arena,
                ),
                ColumnPostings::InRam { .. } => {
                    unreachable!("add_doc_spilled called on InRam column")
                }
            };

        let n_part = partitions.len();
        debug_assert!(
            n_part.is_power_of_two(),
            "spill_partitions must be a power of 2"
        );
        let mask = n_part - 1;

        updated_terms.clear();
        let mut on_token = |tok: &str| {
            tokens_in_doc += 1;
            let (term_id, is_new) = intern_term_id(term_to_id, id_to_term, term_arena, tok);
            let idx = term_id as usize;
            if is_new {
                debug_assert_eq!(idx, dense_doc_tf.len());
                dense_doc_tf.push(0);
            }
            // SAFETY: lockstep invariant between `dense_doc_tf`
            // and `id_to_term` (see type-level docs above) holds
            // here: `intern_term_id` returns `term_id <
            // id_to_term.len()` and we just `push(0)` on `is_new`
            // so `idx < dense_doc_tf.len()`.
            let slot = unsafe { dense_doc_tf.get_unchecked_mut(idx) };
            if *slot == 0 {
                updated_terms.push(term_id);
            }
            *slot += 1;
        };
        if let Some(ascii) = ascii_tok {
            ascii.tokenize_each_inline(text, &mut on_token);
        } else {
            tokenizer.tokenize_each(text, &mut on_token);
        }

        // `self.columns[col_idx]` is a disjoint field from
        // `self.postings[col_idx]`, so split-borrow legal.
        let col = &mut self.columns[col_idx];
        let dl_clamped: u32 = tokens_in_doc.min(u32::MAX as u64) as u32;
        col.doc_lengths.push(dl_clamped);
        col.total_tokens = col.total_tokens.saturating_add(tokens_in_doc);
        let docs_now = local_doc_id.saturating_add(1);
        if docs_now > self.n_docs {
            self.n_docs = docs_now;
        }

        // Per-doc drain: emit one 12-byte triple per touched
        // term, then zero the dense slot to restore the
        // `dense_doc_tf[tid] != 0 iff in updated_terms` invariant.
        //
        // SAFETY: `p = term_id & mask` where
        // `mask = partitions.len() - 1` (power-of-two enforced),
        // so `p < partitions.len()`; the dense-slot index `idx`
        // was already validated above.
        for &term_id in updated_terms.iter() {
            let idx = term_id as usize;
            let slot = unsafe { dense_doc_tf.get_unchecked_mut(idx) };
            let tf = *slot;
            *slot = 0;
            let p = (term_id as usize) & mask;
            let partition = unsafe { partitions.get_unchecked_mut(p) };
            push_triple_batched(partition, term_id, local_doc_id, tf)?;
        }

        Ok(())
    }

    /// In-RAM hot path. Stages tokens into a per-doc `tf_per_term`
    /// hashbrown map (`&str` keys backed by `self.bump`), then
    /// drains into the column's `terms: HashMap<Box<str>, …>`
    /// posting accumulator. On a Zipfian corpus ~80-95% of tokens
    /// are repeats within the same doc, so the bump-arena-keyed
    /// raw-entry probe pays off vs a `Box<str>`-keyed map.
    ///
    /// After the drain, if the column's accumulated bytes cross
    /// `self.spill_threshold_bytes`, transitions it to
    /// `ColumnPostings::Spilled` (one-shot — every subsequent
    /// `add_doc` for this column routes through
    /// [`Self::add_doc_spilled`]).
    #[inline(always)]
    fn add_doc_inram(
        &mut self,
        col_idx: usize,
        local_doc_id: u32,
        text: &str,
    ) -> Result<(), BuildError> {
        let tokenizer = &self.tokenizer;
        let ascii_tok = tokenizer
            .as_ref()
            .as_any()
            .downcast_ref::<AsciiLowerTokenizer>();
        let mut tokens_in_doc: u64 = 0;

        // Reset the per-shard bump arena: leftover token bytes
        // from the prior `add_doc` call are invalidated before we
        // reuse the chunk. `Bump::reset` keeps the largest chunk
        // (no system-allocator round trip on the steady-state
        // doc) and frees any extra chunks the pathological-long
        // doc grew.
        self.bump.reset();
        let bump = &self.bump;

        // Per-doc dedup map. `hashbrown::raw_entry_mut` probes
        // with the *borrowed* `&str` token from the tokenizer's
        // input slice and only pays the `bump.alloc_str` copy on
        // the miss path. Hit-path cost: hash + load + increment.
        let mut tf_per_term: HbHashMap<&'static str, u32, FxBuildHasher> =
            HbHashMap::with_hasher(FxBuildHasher);
        let mut on_token = |tok: &str| {
            tokens_in_doc += 1;
            let hash = compute_hash(tf_per_term.hasher(), tok);
            match tf_per_term
                .raw_entry_mut()
                .from_hash(hash, |existing| *existing == tok)
            {
                RawEntryMut::Occupied(mut e) => {
                    *e.get_mut() += 1;
                }
                RawEntryMut::Vacant(e) => {
                    // Miss path: copy borrowed token bytes into
                    // the bump arena so the key outlives the
                    // tokenizer's input lifetime, then widen the
                    // bump ref to `'static` tied to the HashMap's
                    // lifetime. The HashMap drops at the end of
                    // this call — well before `self.bump` is
                    // reset on the next call.
                    let bumped: &str = bump.alloc_str(tok);
                    // SAFETY: the `'static` tag is a lie — the real
                    // lifetime is `self.bump`, which is reset only on the
                    // next call, after this HashMap (and `extended`) has
                    // been dropped at the end of the current call.
                    let extended: &'static str = unsafe { std::mem::transmute(bumped) };
                    e.insert_hashed_nocheck(hash, extended, 1);
                }
            }
        };
        if let Some(ascii) = ascii_tok {
            ascii.tokenize_each_inline(text, &mut on_token);
        } else {
            tokenizer.tokenize_each(text, &mut on_token);
        }

        let col = &mut self.columns[col_idx];
        let dl_clamped: u32 = tokens_in_doc.min(u32::MAX as u64) as u32;
        col.doc_lengths.push(dl_clamped);
        col.total_tokens = col.total_tokens.saturating_add(tokens_in_doc);
        let docs_now = local_doc_id.saturating_add(1);
        if docs_now > self.n_docs {
            self.n_docs = docs_now;
        }

        let column_id = col_idx as u32;
        let col_post = &mut self.postings[col_idx];
        let (terms, bytes) = match col_post {
            ColumnPostings::InRam { terms, bytes } => (terms, bytes),
            ColumnPostings::Spilled { .. } => {
                unreachable!("add_doc_inram called on Spilled column")
            }
        };
        // Lookup with `get_mut(&str)` first — borrowed probe
        // against a `Box<str>` key, no allocation on hits. Only
        // the miss path constructs a `Box::<str>::from(term)` for
        // insertion. Bytes accounting folds into the same loop.
        let mut new_bytes: usize = 0;
        for (term, tf) in tf_per_term {
            let term_len = term.len();
            match terms.get_mut(term) {
                Some(acc) => {
                    acc.push((local_doc_id, tf));
                    new_bytes = new_bytes.saturating_add(ACCUM_POSTING_BYTES);
                }
                None => {
                    terms.insert(Box::<str>::from(term), vec![(local_doc_id, tf)]);
                    new_bytes = new_bytes.saturating_add(
                        ACCUM_NEW_TERM_FIXED_BYTES + term_len + ACCUM_POSTING_BYTES,
                    );
                }
            }
        }
        let new_total = bytes.saturating_add(new_bytes);

        if new_total > self.spill_threshold_bytes {
            // Crossed the threshold. Drain the in-RAM map into a
            // fresh set of spill files for this column, build the
            // interner from the drained vocab, transition to
            // `Spilled` so subsequent `add_doc` calls route to
            // `add_doc_spilled`.
            let drained = std::mem::take(terms);
            let mut partitions = Self::open_partitions_for_column(
                self.scratch_dir.path(),
                column_id,
                self.spill_partitions,
            )?;
            let term_arena = bumpalo::Bump::new();
            let mut term_to_id: TermIdMap = TermIdMap::default();
            // Pre-size to the exact post-flush vocab.
            let mut id_to_term: Vec<&'static str> = Vec::with_capacity(drained.len());
            Self::flush_in_ram_to_partitions(
                drained,
                &mut partitions,
                &mut term_to_id,
                &mut id_to_term,
                &term_arena,
            )?;
            let dense_doc_tf = vec![0u32; id_to_term.len()];
            let updated_terms: Vec<u32> = Vec::new();
            *col_post = ColumnPostings::Spilled {
                partitions,
                term_to_id,
                id_to_term,
                dense_doc_tf,
                updated_terms,
                // Declared last in the variant so it drops last;
                // see `TermIdMap` doc for the lifetime invariant.
                term_arena,
            };
        } else {
            *bytes = new_total;
        }

        Ok(())
    }

    /// Finalise and emit the FTS blob bytes. Consumes the builder.
    ///
    /// Returns `BuildError::Io` for scratch IO failures that can
    /// occur on the spill path (partition write/read, streaming-FST
    /// scratch file, posting region scratch file). Mirror of
    /// `VectorBuilder::finish`, which has the same return type for
    /// the same reason.
    pub fn finish(self) -> Result<Vec<u8>, BuildError> {
        let mut blob = Vec::new();
        self.finish_to(&mut blob)?;
        Ok(blob)
    }

    /// Streaming variant: write the final FTS blob progressively to `w`.
    ///
    /// Two finish paths, picked automatically by inspecting whether
    /// any column has spilled:
    ///
    /// - **In-RAM finish** (no column spilled): per-column term maps
    ///   are sorted and drained term-by-term, encoded postings flow
    ///   to a posting-region scratch file, and the FST is built in
    ///   RAM via `DictBuilder`. Small-build path; mirrors the
    ///   in-RAM `VectorBuilder` finish.
    ///
    /// - **Spilled finish** (any column spilled): every column's
    ///   posting source — in-RAM map (for columns that stayed below
    ///   threshold) or partition files (for spilled columns) — is
    ///   normalised into a sorted record stream, then column-by-
    ///   column those streams feed a *streaming* FST builder writing
    ///   to a scratch file. The FST never lives entirely in RAM.
    ///   Mirrors the spilled `VectorBuilder` finish.
    ///
    /// Final blob assembly is byte-identical between the two paths
    /// (regression-gated by
    /// `build_above_threshold_spills_and_matches_in_ram_byte_for_byte`).
    pub fn finish_to<W: Write>(self, w: W) -> Result<(), BuildError> {
        // Dispatch to the path that matches the builder's current
        // state. The two paths share `assemble_and_write_blob` as
        // their tail, so the output blob is byte-identical for
        // builds that *could* be served by either (regression-
        // gated by `build_above_threshold_spills_and_matches_in_
        // ram_byte_for_byte`).
        if self.postings.iter().any(|c| c.is_spilled()) {
            self.finish_to_spilled(w)
        } else {
            self.finish_to_inram(w)
        }
    }

    /// In-RAM finish path: every column's accumulated `terms` map
    /// stayed under `spill_threshold_bytes`, so the FST is built in
    /// one shot via [`DictBuilder`] (no scratch FST file), no
    /// partition flush runs, and the per-column emit loop only ever
    /// sees the `InRam` variant.
    ///
    /// This is the production rayon-sharded path for builds whose
    /// per-shard data fits below the spill threshold — the case the
    /// 1M-doc Zipfian bench measures.
    fn finish_to_inram<W: Write>(self, mut w: W) -> Result<(), BuildError> {
        let FtsBuilder {
            tokenizer: _,
            columns,
            postings,
            scratch_dir,
            spill_threshold_bytes: _,
            spill_partitions: _,
            max_partition_bytes: _,
            n_docs,
            bump: _,
        } = self;

        let n_columns = columns.len() as u32;
        let mut n_terms_total_usize: usize = 0;

        // Build the per-column work list, sorted by lex name. Drained
        // by value below so each column's in-RAM map is dropped the
        // instant its terms have been emitted. Mirror of vector's
        // `columns.into_iter().enumerate()` pattern.
        let mut work: Vec<(usize, ColumnState, ColumnPostings)> = columns
            .into_iter()
            .zip(postings)
            .enumerate()
            .map(|(orig_idx, (state, posting_state))| (orig_idx, state, posting_state))
            .collect();
        work.sort_unstable_by(|a, b| a.1.name.cmp(&b.1.name));

        let mut avgdl_per_col: Vec<f32> = vec![0.0; n_columns as usize];
        for (orig_idx, state, _) in &work {
            let n = state.doc_lengths.len() as u64;
            avgdl_per_col[*orig_idx] = if n == 0 {
                0.0
            } else {
                (state.total_tokens as f32) / (n as f32)
            };
        }

        let scratch_path = scratch_dir.path().to_path_buf();
        // Posting body scratch file. Encoded posting blocks for every
        // (column, term) flow here in lex order, then get streamed
        // through to `w` at assembly time.
        let postings_path = scratch_path.join("infino_fts_postings.bin");
        let mut postings_writer = BufWriter::new(File::create(&postings_path)?);
        let mut postings_len: u64 = 0;
        let mut postings_crc_acc: u32 = 0;
        let mut key_buf: Vec<u8> = Vec::with_capacity(64);
        let mut term_scratch = TermScratch::default();
        let mut finish_profile = FinishProfile::from_env();

        // The in-RAM path's FST sink: collect (key, value) into a
        // `DictBuilder` and serialise once at assembly time. No
        // scratch file, no streaming.
        let mut fst_inram = DictBuilder::new();

        let mut doc_lengths_by_orig_col: Vec<Option<Vec<u32>>> =
            (0..n_columns as usize).map(|_| None).collect();

        for (orig_col_idx, col_state, posting_state) in work.drain(..) {
            let ColumnState {
                name: col_name,
                doc_lengths: col_doc_lengths_owned,
                total_tokens: _,
            } = col_state;
            let col_name_bytes = col_name.as_bytes();
            let avgdl = avgdl_per_col[orig_col_idx];
            let col_doc_lengths: &[u32] = &col_doc_lengths_owned;

            // In-RAM path invariant: dispatcher checked
            // `!any_spilled`, so every column is `InRam`.
            let terms = match posting_state {
                ColumnPostings::InRam { terms, bytes: _ } => terms,
                ColumnPostings::Spilled { .. } => unreachable!(
                    "finish_to_inram dispatched on !any_spilled; \
                     Spilled column cannot appear here"
                ),
            };

            // Sort term keys; per-term doc lists are already in
            // insertion order (monotonically increasing local_doc_id
            // per the `add_doc` contract), so no per-list sort
            // needed. pdqsort: dictionary entries for one in-RAM
            // column can run into millions; stability is unnecessary
            // because keys are unique.
            type InRamEntries = Vec<(Box<str>, Vec<(u32, u32)>)>;
            let mut entries: InRamEntries = terms.into_iter().collect();
            entries.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            for (term, postings) in entries {
                encode_and_emit_term(
                    &term,
                    &postings,
                    col_name_bytes,
                    col_doc_lengths,
                    avgdl,
                    n_docs,
                    &mut key_buf,
                    &mut postings_writer,
                    &mut postings_crc_acc,
                    &mut postings_len,
                    Some(&mut fst_inram),
                    None,
                    &mut finish_profile,
                    &mut term_scratch,
                )?;
                n_terms_total_usize += 1;
            }

            doc_lengths_by_orig_col[orig_col_idx] = Some(col_doc_lengths_owned);
        }

        assemble_and_write_blob(
            BlobAssemblyInputs {
                postings_writer,
                postings_path,
                postings_crc_acc,
                postings_len,
                fst_sink: FstSinkFinish::InRam(fst_inram),
                n_columns,
                n_docs,
                n_terms_total_usize,
                avgdl_per_col,
                doc_lengths_by_orig_col,
                scratch_dir,
                finish_profile,
            },
            &mut w,
        )
    }

    /// Spilled finish path: at least one column transitioned to
    /// `Spilled` during `add_doc`. Uses a streaming `MapBuilder`
    /// (FST bytes go to a scratch file as we go, never resident in
    /// RAM in full), drains every spilled column's per-partition
    /// batch buffers + closes their `BufWriter`s before the merge
    /// reads them, and the per-column emit loop handles both
    /// variants (a spilled build can still have InRam columns whose
    /// accumulator stayed under threshold).
    fn finish_to_spilled<W: Write>(self, mut w: W) -> Result<(), BuildError> {
        let FtsBuilder {
            tokenizer: _,
            columns,
            postings,
            scratch_dir,
            spill_threshold_bytes: _,
            spill_partitions: _,
            max_partition_bytes,
            n_docs,
            bump: _,
        } = self;

        let n_columns = columns.len() as u32;
        let mut n_terms_total_usize: usize = 0;

        let mut work: Vec<(usize, ColumnState, ColumnPostings)> = columns
            .into_iter()
            .zip(postings)
            .enumerate()
            .map(|(orig_idx, (state, posting_state))| (orig_idx, state, posting_state))
            .collect();
        work.sort_unstable_by(|a, b| a.1.name.cmp(&b.1.name));

        let mut avgdl_per_col: Vec<f32> = vec![0.0; n_columns as usize];
        for (orig_idx, state, _) in &work {
            let n = state.doc_lengths.len() as u64;
            avgdl_per_col[*orig_idx] = if n == 0 {
                0.0
            } else {
                (state.total_tokens as f32) / (n as f32)
            };
        }

        let scratch_path = scratch_dir.path().to_path_buf();
        let postings_path = scratch_path.join("infino_fts_postings.bin");
        let mut postings_writer = BufWriter::new(File::create(&postings_path)?);
        let mut postings_len: u64 = 0;
        let mut postings_crc_acc: u32 = 0;
        let mut key_buf: Vec<u8> = Vec::with_capacity(64);
        let mut term_scratch = TermScratch::default();
        let mut finish_profile = FinishProfile::from_env();

        // Streaming FST: bytes flow to a scratch file as we insert
        // sorted keys, so the FST never lives in RAM in full.
        // Assembly reopens the file, CRCs it, and copies it into the
        // output writer.
        let fst_streaming_path = scratch_path.join("infino_fts_dict.bin");
        let mut fst_streaming = {
            let fst_file = File::create(&fst_streaming_path)?;
            let bw = BufWriter::new(fst_file);
            StreamingDictBuilder::new(bw).map_err(map_fst_err)?
        };

        // Drain every spilled column's per-partition batch buffer
        // into the partition's `BufWriter`, then flush + close it
        // so the merge phase reads a complete file. InRam columns
        // in this work list are no-ops.
        let partition_flush_start = finish_profile.enabled.then(Instant::now);
        for (_, _, cp) in &mut work {
            if let ColumnPostings::Spilled { partitions, .. } = cp {
                for partition in partitions {
                    flush_partition_batch(partition)?;
                    if let Some(mut writer) = partition.writer.take() {
                        writer.flush()?;
                    }
                }
            }
        }
        if let Some(t) = partition_flush_start {
            finish_profile.partition_flush += t.elapsed();
        }

        let mut doc_lengths_by_orig_col: Vec<Option<Vec<u32>>> =
            (0..n_columns as usize).map(|_| None).collect();

        // Drain in lex order, consuming each entry by value so the
        // ColumnPostings (partition writers, in-RAM map) is dropped
        // before we touch the next column.
        for (orig_col_idx, col_state, posting_state) in work.drain(..) {
            let ColumnState {
                name: col_name,
                doc_lengths: col_doc_lengths_owned,
                total_tokens: _,
            } = col_state;
            let col_name_bytes = col_name.as_bytes();
            let avgdl = avgdl_per_col[orig_col_idx];
            let col_doc_lengths: &[u32] = &col_doc_lengths_owned;

            match posting_state {
                ColumnPostings::InRam { terms, bytes: _ } => {
                    // Sort term keys; per-term doc lists are already
                    // in insertion order which is monotonically
                    // increasing local_doc_id per the add_doc
                    // contract — no per-list sort needed.
                    type InRamEntries = Vec<(Box<str>, Vec<(u32, u32)>)>;
                    let mut entries: InRamEntries = terms.into_iter().collect();
                    // pdqsort: posting-table dictionary entries for
                    // one in-RAM column can run into millions of
                    // terms; stability is unnecessary because keys
                    // are unique.
                    entries.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
                    for (term, postings) in entries {
                        encode_and_emit_term(
                            &term,
                            &postings,
                            col_name_bytes,
                            col_doc_lengths,
                            avgdl,
                            n_docs,
                            &mut key_buf,
                            &mut postings_writer,
                            &mut postings_crc_acc,
                            &mut postings_len,
                            None,
                            Some(&mut fst_streaming),
                            &mut finish_profile,
                            &mut term_scratch,
                        )?;
                        n_terms_total_usize += 1;
                    }
                }
                ColumnPostings::Spilled {
                    partitions,
                    term_to_id,
                    id_to_term,
                    dense_doc_tf: _,
                    updated_terms: _,
                    term_arena,
                } => {
                    // Term interner is finished being written to;
                    // drop the forward map (`term_to_id`) immediately
                    // — the rest of the spilled finish only needs
                    // the reverse map (`id_to_term`) for FST emit and
                    // the lex-rank table built from it.
                    //
                    // Lifetime sanity: `term_to_id` and `id_to_term`
                    // hold `&'static str` keys/entries that actually
                    // borrow from `term_arena`. We must keep
                    // `term_arena` alive until the **last
                    // dereference** of those keys/entries, which is
                    // the FST emit loop's `id_to_term[term_id]`
                    // index below. End-of-scope `Drop` order does
                    // not matter for soundness — neither `&str` nor
                    // its container's `Drop` impls dereference the
                    // borrowed bytes — but any **use** of `id_to_
                    // term[i].as_bytes()` must occur before
                    // `term_arena` goes out of scope. Both this
                    // arm's body and the helper functions it calls
                    // observe that.
                    drop(term_to_id);
                    let lex_rank_start = finish_profile.enabled.then(Instant::now);
                    let (lex_rank, term_id_in_lex_order) = build_lex_rank(&id_to_term);
                    if let Some(t) = lex_rank_start {
                        finish_profile.lex_rank_build += t.elapsed();
                    }

                    // Pre-sort every partition to a sorted-triple
                    // file under scratch. Sorting partition-at-a-
                    // time bounds the in-RAM sort working set to
                    // `max_partition_bytes` (one partition at a
                    // time), then the k-way merge across the
                    // resulting sorted files runs with
                    // O(n_partitions) cursors each holding one
                    // triple + a small read buffer.
                    let sort_start = finish_profile.enabled.then(Instant::now);
                    let mut sorted_files: Vec<PathBuf> = Vec::with_capacity(partitions.len());
                    for (partition_idx, partition) in partitions.iter().enumerate() {
                        let sorted_path = scratch_path.join(format!(
                            "fts_col{orig_col_idx}_part{partition_idx}.sorted.bin"
                        ));
                        sort_partition_to_file(
                            &partition.path,
                            &sorted_path,
                            max_partition_bytes,
                            &scratch_path,
                            &format!("c{orig_col_idx}_p{partition_idx}"),
                            &lex_rank,
                        )?;
                        sorted_files.push(sorted_path);
                    }
                    if let Some(t) = sort_start {
                        finish_profile.partition_sort += t.elapsed();
                    }

                    // Lex-order partition traversal. Replaces the
                    // earlier `BinaryHeap`-based k-way merge: since
                    // partition assignment is `partition =
                    // term_id & (n_part - 1)` (enforced
                    // power-of-two in `set_spill_partitions`),
                    // every posting for a given `term_id` lives
                    // in exactly one partition. Within that
                    // partition, the sort-partition phase has
                    // already arranged triples in
                    // `(lex_rank[term_id], doc_id)` order, so all
                    // triples for one `term_id` are contiguous
                    // there and they emerge in `doc_id` order
                    // when scanned forward.
                    //
                    // The merge therefore reduces to: walk
                    // `term_id_in_lex_order` (the
                    // sort-once-globally permutation we already
                    // produced above) and, for each `term_id`,
                    // drain the contiguous matching run from
                    // `sorted_slices[term_id & mask]` starting at
                    // that partition's cursor. Cost is O(n_postings
                    // + n_terms) sequential mmap reads + one u32
                    // compare per posting, versus the heap path's
                    // O(n_postings · log n_part) compares + heap
                    // pushes/pops.
                    //
                    // We still mmap each sorted partition file
                    // (zero-copy `&[Triple]` over page-cache-hot
                    // bytes), so the per-posting access is pointer
                    // arithmetic against contiguous memory.
                    let mmap_start = finish_profile.enabled.then(Instant::now);
                    let mut mmaps: Vec<Mmap> = Vec::with_capacity(sorted_files.len());
                    for p in &sorted_files {
                        let f = File::open(p)?;
                        // SAFETY: the sorted-partition scratch file is
                        // owned by this builder's `scratch_dir`; no
                        // other process truncates or appends to it
                        // for the lifetime of the `Mmap`.
                        let mmap = unsafe { Mmap::map(&f)? };
                        mmaps.push(mmap);
                    }
                    if let Some(t) = mmap_start {
                        finish_profile.mmap_open += t.elapsed();
                    }
                    let sorted_slices: Vec<&[Triple]> = mmaps
                        .iter()
                        .map(|m| {
                            if m.is_empty() {
                                &[][..]
                            } else {
                                bytemuck::cast_slice::<u8, Triple>(&m[..])
                            }
                        })
                        .collect();
                    let mask = (sorted_slices.len() - 1) as u32;
                    // Per-partition next-index into its sorted
                    // slice. Walked forward only — each posting
                    // is read exactly once.
                    let mut cursors: Vec<usize> = vec![0usize; sorted_slices.len()];

                    // Per-term posting buffer. Reused across all
                    // terms in this column so we pay one
                    // `Vec` growth schedule instead of one per
                    // term.
                    let mut group: Vec<(u32, u32)> = Vec::new();
                    let merge_profile_start = Instant::now();
                    let encode_calls_before = finish_profile.encode_calls;
                    let encode_df1_before = finish_profile.encode_df1;
                    let encode_pfor_before = finish_profile.encode_pfor;
                    let encode_total_before = finish_profile.encode_total;
                    let encode_block_build_before = finish_profile.encode_block_build;
                    let encode_meta_write_before = finish_profile.encode_meta_write;
                    let encode_skip_write_before = finish_profile.encode_skip_write;
                    let encode_block_write_before = finish_profile.encode_block_write;
                    let fst_insert_before = finish_profile.fst_insert;
                    for &term_id in &term_id_in_lex_order {
                        let p = (term_id & mask) as usize;
                        let slice = sorted_slices[p];
                        let mut pos = cursors[p];
                        group.clear();
                        // Drain the contiguous run for this term.
                        // Termination: either the partition runs
                        // out, or the next triple's term_id
                        // differs (next term in this partition's
                        // lex-rank order, which can only be a
                        // strictly higher `lex_rank` and so a
                        // different `term_id`).
                        while pos < slice.len() {
                            let t = &slice[pos];
                            if triple_term_id(t) != term_id {
                                break;
                            }
                            group.push((triple_doc_id(t), triple_tf(t)));
                            pos += 1;
                        }
                        cursors[p] = pos;
                        if group.is_empty() {
                            // Term registered in `id_to_term` but
                            // no postings landed for it — only
                            // possible if `flush_in_ram_to_partitions`
                            // ran with an empty postings vec.
                            // Defensive: skip without emitting.
                            continue;
                        }
                        let term_bytes: &str = id_to_term[term_id as usize];
                        encode_and_emit_term(
                            term_bytes,
                            &group,
                            col_name_bytes,
                            col_doc_lengths,
                            avgdl,
                            n_docs,
                            &mut key_buf,
                            &mut postings_writer,
                            &mut postings_crc_acc,
                            &mut postings_len,
                            None,
                            Some(&mut fst_streaming),
                            &mut finish_profile,
                            &mut term_scratch,
                        )?;
                        n_terms_total_usize += 1;
                    }
                    // Sanity: every partition should now be fully
                    // drained. If not, we lost or mis-ordered
                    // triples somewhere upstream.
                    debug_assert!(
                        cursors
                            .iter()
                            .zip(sorted_slices.iter())
                            .all(|(c, s)| *c == s.len()),
                        "lex-order partition traversal did not drain all triples; \
                         partition assignment or sort invariant violated"
                    );
                    if finish_profile.enabled {
                        let merge_total = merge_profile_start.elapsed();
                        let encode_total = finish_profile.encode_total - encode_total_before;
                        let non_encode = merge_total.saturating_sub(encode_total);
                        eprintln!(
                            "[fts-profile] col='{}' merge_total={:.3}s non_encode_merge={:.3}s encode_total={:.3}s calls={} df1={} pfor={} block_build={:.3}s meta_write={:.3}s skip_write={:.3}s block_write={:.3}s fst_insert={:.3}s",
                            col_name,
                            merge_total.as_secs_f64(),
                            non_encode.as_secs_f64(),
                            encode_total.as_secs_f64(),
                            finish_profile.encode_calls - encode_calls_before,
                            finish_profile.encode_df1 - encode_df1_before,
                            finish_profile.encode_pfor - encode_pfor_before,
                            (finish_profile.encode_block_build - encode_block_build_before)
                                .as_secs_f64(),
                            (finish_profile.encode_meta_write - encode_meta_write_before)
                                .as_secs_f64(),
                            (finish_profile.encode_skip_write - encode_skip_write_before)
                                .as_secs_f64(),
                            (finish_profile.encode_block_write - encode_block_write_before)
                                .as_secs_f64(),
                            (finish_profile.fst_insert - fst_insert_before).as_secs_f64(),
                        );
                    }

                    // Sorted-partition scratch files are scoped to
                    // this column and only consumed by the k-way
                    // merge above. Drop the mmap views first
                    // (releases the page-cache references), then
                    // remove the files so the next spilled column
                    // doesn't see their disk residency. (Original
                    // partition files are owned by `partitions`
                    // and dropped at the next iteration boundary.)
                    let cleanup_start = finish_profile.enabled.then(Instant::now);
                    drop(sorted_slices);
                    drop(mmaps);
                    for p in &sorted_files {
                        let _ = std::fs::remove_file(p);
                    }
                    // The raw spill partition files are also no
                    // longer needed once the merge finishes — the
                    // tempdir cleanup will reap them at scope exit
                    // but doing it here keeps peak resident bytes
                    // on disk bounded to one column.
                    drop(partitions);
                    // Explicit drop sequence: `id_to_term` first
                    // (its `&'static str` entries borrow from
                    // `term_arena`; we've finished the last read
                    // at the FST emit loop above), then
                    // `term_arena` itself releases the term-byte
                    // backing store. Soundness does not strictly
                    // require this order (neither `Vec<&str>::
                    // drop` nor `Bump::drop` interacts with the
                    // other), but spelling it out leaves the
                    // intent unambiguous to future readers.
                    drop(id_to_term);
                    drop(term_arena);
                    drop(lex_rank);
                    if let Some(t) = cleanup_start {
                        finish_profile.scratch_cleanup += t.elapsed();
                    }
                }
            }

            // Hand this column's doc-lengths off for the
            // final-assembly pass. `posting_state` (with its
            // partition writers if any) and `col_name` are dropped
            // here at scope exit before the next iteration starts,
            // bounding peak per-column resident state to one column.
            // Mirror of vector's `columns.into_iter().enumerate()`
            // lifetime in `VectorBuilder::finish_to`.
            doc_lengths_by_orig_col[orig_col_idx] = Some(col_doc_lengths_owned);
        }

        assemble_and_write_blob(
            BlobAssemblyInputs {
                postings_writer,
                postings_path,
                postings_crc_acc,
                postings_len,
                fst_sink: FstSinkFinish::Streaming {
                    builder: fst_streaming,
                    path: fst_streaming_path,
                },
                n_columns,
                n_docs,
                n_terms_total_usize,
                avgdl_per_col,
                doc_lengths_by_orig_col,
                scratch_dir,
                finish_profile,
            },
            &mut w,
        )
    }
}

/// Inputs threaded into [`assemble_and_write_blob`]. Bundled into a
/// struct rather than a 12-arg call so the two finish paths can read
/// like "build everything, then assemble" instead of routing through
/// a long positional argument list. All fields are consumed by
/// assembly.
struct BlobAssemblyInputs {
    /// Open scratch writer holding every encoded posting block in
    /// lex order. Assembly closes it (CRC trailer + flush) before
    /// streaming the file's contents into the output.
    postings_writer: BufWriter<File>,
    /// On-disk path of the posting body. Reopened for the streaming
    /// copy into the output writer.
    postings_path: PathBuf,
    /// Running CRC32C over every byte written to `postings_writer`.
    /// Assembly appends the little-endian trailer.
    postings_crc_acc: u32,
    /// Bytes written so far to `postings_writer` (excluding trailer).
    /// Assembly grows this by 4 when it appends the CRC.
    postings_len: u64,
    /// Whichever FST sink was used during the per-column emit loop
    /// (exactly one of the two variants).
    fst_sink: FstSinkFinish,
    n_columns: u32,
    n_docs: u32,
    /// Pre-cast checked downstream against `u32::MAX`.
    n_terms_total_usize: usize,
    /// Per-original-column avgdl (declaration order, not lex order).
    avgdl_per_col: Vec<f32>,
    /// Per-original-column doc-lengths, moved out of `work` by each
    /// emit-loop iteration.
    doc_lengths_by_orig_col: Vec<Option<Vec<u32>>>,
    /// Scratch dir owning every spill file. Dropped after the
    /// streamed regions (FST + postings) have been copied into `w`.
    scratch_dir: tempfile::TempDir,
    /// Profile accumulator — final block of `[fts-finish]` timings
    /// is emitted at the bottom of assembly.
    finish_profile: FinishProfile,
}

/// FST emission sink picked by the active finish path.
enum FstSinkFinish {
    /// In-RAM build: hand the populated `DictBuilder` to assembly,
    /// which calls `finish()` to produce the FST bytes in one shot.
    InRam(DictBuilder),
    /// Spilled build: hand the open `StreamingDictBuilder` (and the
    /// scratch path it's been writing to) to assembly, which finishes
    /// the builder, computes the file's CRC by streaming, and copies
    /// the file into the output.
    Streaming {
        builder: StreamingDictBuilder<BufWriter<File>>,
        path: PathBuf,
    },
}

/// Common tail of every finish path: close the posting body, finalise
/// the FST, build the doc-lengths directory + arrays, and write
/// `[header | fst | postings | dir | arrays]` to `w`. Lifted out of
/// `FtsBuilder::finish_to` so the in-RAM and spilled paths share one
/// regression target instead of two — every byte the reader observes
/// passes through here.
fn assemble_and_write_blob<W: Write>(
    inputs: BlobAssemblyInputs,
    w: &mut W,
) -> Result<(), BuildError> {
    let BlobAssemblyInputs {
        mut postings_writer,
        postings_path,
        postings_crc_acc,
        mut postings_len,
        fst_sink,
        n_columns,
        n_docs,
        n_terms_total_usize,
        avgdl_per_col,
        mut doc_lengths_by_orig_col,
        scratch_dir,
        mut finish_profile,
    } = inputs;

    debug_assert!(
        n_terms_total_usize <= u32::MAX as usize,
        "term count overflows u32"
    );
    let n_terms_total = n_terms_total_usize as u32;

    // Close the posting body file (CRC trailer + flush).
    let postings_close_start = finish_profile.enabled.then(Instant::now);
    let postings_crc = postings_crc_acc;
    let postings_crc_le = postings_crc.to_le_bytes();
    postings_writer.write_all(&postings_crc_le)?;
    postings_writer.flush()?;
    drop(postings_writer);
    postings_len += postings_crc_le.len() as u64;
    if let Some(t) = postings_close_start {
        finish_profile.postings_close += t.elapsed();
    }

    // Finalise the FST. Either path produces "FST bytes followed
    // by 4 trailing CRC bytes"; the source differs.
    enum FstSource {
        InRam(Vec<u8>),
        Streamed { path: PathBuf, len: u64, crc: u32 },
    }
    let fst_close_start = finish_profile.enabled.then(Instant::now);
    let fst_source = match fst_sink {
        FstSinkFinish::InRam(db) => {
            let mut bytes = db.finish();
            let crc = crc32c(&bytes);
            bytes.extend_from_slice(&crc.to_le_bytes());
            FstSource::InRam(bytes)
        }
        FstSinkFinish::Streaming {
            builder,
            path: fst_streaming_path,
        } => {
            let mut bw = builder.finish().map_err(map_fst_err)?;
            bw.flush()?;
            // Close the write side of the FST scratch file. The
            // returned `File` is `File::create`-opened (write-only),
            // so we must reopen for reading to compute the CRC and
            // later stream into `w`.
            let write_file = bw
                .into_inner()
                .map_err(|e| BuildError::Io(e.into_error()))?;
            drop(write_file);

            // Stream the FST scratch file with bounded memory to
            // compute its CRC.
            let mut read_file = File::open(&fst_streaming_path)?;
            let fst_body_len = read_file.metadata()?.len();
            read_file.seek(SeekFrom::Start(0))?;
            let mut reader = BufReader::with_capacity(PARTITION_BUF_SIZE, read_file);
            let mut crc: u32 = 0;
            let mut buf = vec![0u8; PARTITION_BUF_SIZE];
            loop {
                let n = match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                    Err(e) => return Err(BuildError::Io(e)),
                };
                crc = crc32c_append(crc, &buf[..n]);
            }
            drop(reader);
            FstSource::Streamed {
                path: fst_streaming_path,
                len: fst_body_len + 4, /* trailing CRC */
                crc,
            }
        }
    };
    if let Some(t) = fst_close_start {
        finish_profile.fst_close += t.elapsed();
    }

    // Compute final-blob offsets now that both region lengths are known.
    let dl_emit_start = finish_profile.enabled.then(Instant::now);
    let fst_total_len: u64 = match &fst_source {
        FstSource::InRam(bytes) => bytes.len() as u64,
        FstSource::Streamed { len, .. } => *len,
    };
    let header_size: u64 = FTS_HEADER_SIZE as u64;
    let fst_offset: u64 = header_size;
    let postings_offset: u64 = fst_offset + fst_total_len;
    let doc_lengths_table_offset: u64 = postings_offset + postings_len;
    let mut doc_lengths_array_offset: u64 =
        doc_lengths_table_offset + (n_columns as u64) * (DOC_LENGTHS_ENTRY_SIZE as u64) + 4 /* dir CRC */;

    let mut dir_buf: Vec<u8> = Vec::with_capacity(n_columns as usize * DOC_LENGTHS_ENTRY_SIZE);
    let mut arrays_buf: Vec<u8> = Vec::new();
    for i in 0..n_columns as usize {
        let avgdl_x1000 = (avgdl_per_col[i] * format::fts::AVGDL_FIXED_POINT_SCALE)
            .max(0.0)
            .min(u32::MAX as f32) as u32;
        dir_buf.extend_from_slice(&(i as u32).to_le_bytes());
        dir_buf.extend_from_slice(&doc_lengths_array_offset.to_le_bytes());
        dir_buf.extend_from_slice(&avgdl_x1000.to_le_bytes());

        let col_dls = doc_lengths_by_orig_col[i]
            .take()
            .expect("doc_lengths recorded for every registered column");
        let array_start = arrays_buf.len();
        // x86_64 is little-endian and the format spec is
        // little-endian u32 — so a raw byte-cast over the
        // `Vec<u32>` slice is the wire encoding. `bytemuck`
        // gates this on `Pod` so a non-LE host would fail
        // compilation rather than silently emit wrong bytes;
        // the SIMD memcpy that `extend_from_slice` lowers to
        // is materially faster than the per-u32 `to_le_bytes`
        // + push loop, especially at the 10M-doc / column
        // scale where this writes 40 MB per column.
        #[cfg(target_endian = "little")]
        arrays_buf.extend_from_slice(bytemuck::cast_slice::<u32, u8>(&col_dls));
        #[cfg(not(target_endian = "little"))]
        for &dl in &col_dls {
            arrays_buf.extend_from_slice(&dl.to_le_bytes());
        }
        let array_bytes = &arrays_buf[array_start..];
        let array_crc = crc32c(array_bytes);
        arrays_buf.extend_from_slice(&array_crc.to_le_bytes());
        doc_lengths_array_offset += (col_dls.len() as u64) * 4 + 4;
    }
    let dir_crc = crc32c(&dir_buf);
    dir_buf.extend_from_slice(&dir_crc.to_le_bytes());
    if let Some(t) = dl_emit_start {
        finish_profile.doc_lengths_emit += t.elapsed();
    }

    // Final assembly. Bytes flow scratch → small streaming buffer
    // → `w`, never re-materialising the full blob in RAM.
    let blob_copy_start = finish_profile.enabled.then(Instant::now);
    let mut header = Vec::with_capacity(header_size as usize);
    header.extend_from_slice(format::fts::MAGIC); // 8
    header.extend_from_slice(&format::fts::VERSION.to_le_bytes()); // 4
    header.extend_from_slice(&n_columns.to_le_bytes()); // 4
    header.extend_from_slice(&n_docs.to_le_bytes()); // 4
    header.extend_from_slice(&n_terms_total.to_le_bytes()); // 4
    header.extend_from_slice(&fst_offset.to_le_bytes()); // 8
    header.extend_from_slice(&postings_offset.to_le_bytes()); // 8
    header.extend_from_slice(&doc_lengths_table_offset.to_le_bytes()); // 8
    debug_assert_eq!(header.len(), header_size as usize, "header size mismatch");

    w.write_all(&header)?;
    match fst_source {
        FstSource::InRam(bytes) => w.write_all(&bytes)?,
        FstSource::Streamed { path, crc, .. } => {
            let mut reader = BufReader::with_capacity(PARTITION_BUF_SIZE, File::open(&path)?);
            std::io::copy(&mut reader, w)?;
            w.write_all(&crc.to_le_bytes())?;
        }
    }
    let mut postings_reader =
        BufReader::with_capacity(PARTITION_BUF_SIZE, File::open(&postings_path)?);
    std::io::copy(&mut postings_reader, w)?;
    drop(postings_reader);

    // Drop the scratch tempdir as soon as the streamed source
    // files (FST + posting body) have been copied into `w`. The
    // remaining writes (`dir_buf`, `arrays_buf`) are
    // already-resident `Vec<u8>` and don't touch the disk.
    // Mirror of vector's `drop(scratch_dir);` at the bottom of
    // `VectorBuilder::finish_to`.
    drop(scratch_dir);

    w.write_all(&dir_buf)?;
    w.write_all(&arrays_buf)?;
    if let Some(t) = blob_copy_start {
        finish_profile.blob_copy += t.elapsed();
    }

    if finish_profile.enabled {
        eprintln!(
            "[fts-finish] partition_flush={:.3}s lex_rank={:.3}s partition_sort={:.3}s mmap_open={:.3}s scratch_cleanup={:.3}s postings_close={:.3}s fst_close={:.3}s doc_lengths_emit={:.3}s blob_copy={:.3}s",
            finish_profile.partition_flush.as_secs_f64(),
            finish_profile.lex_rank_build.as_secs_f64(),
            finish_profile.partition_sort.as_secs_f64(),
            finish_profile.mmap_open.as_secs_f64(),
            finish_profile.scratch_cleanup.as_secs_f64(),
            finish_profile.postings_close.as_secs_f64(),
            finish_profile.fst_close.as_secs_f64(),
            finish_profile.doc_lengths_emit.as_secs_f64(),
            finish_profile.blob_copy.as_secs_f64(),
        );
    }

    Ok(())
}

#[inline]
fn map_fst_err(e: fst::Error) -> BuildError {
    BuildError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Reusable per-term scratch buffers threaded through
/// `encode_and_emit_term`. One instance is created at the top of
/// `finish_to` and re-used across every term encoded in the column,
/// turning ~3M+ per-term `Vec::new` allocations on the 1M-doc
/// forced-spill bench into ~5 reused allocations (one per buffer,
/// once per column). The Block's `doc_ids` / `tfs` Vecs are
/// `mem::take`'d into a Block, encoded, then swapped back via
/// `mem::take` so the underlying buffer is reused for the next
/// chunk (same allocation, just `Vec::clear` between iterations).
#[derive(Default)]
struct TermScratch {
    /// Per-block doc_id column passed to `encode_block`. Re-used by
    /// taking it into a Block, encoding, and taking it back. Capacity
    /// stabilises at `BLOCK_LEN` after the first dense term.
    doc_ids: Vec<u32>,
    /// Per-block tf column, same lifecycle as `doc_ids`.
    tfs: Vec<u32>,
    /// Per-term minimum doc-length per block, used to compute the
    /// per-block BM25 upper bound for the skip table.
    min_dl_per_block: Vec<u32>,
    /// Per-term list of encoded blocks held across the meta + skip-
    /// table + block-bytes emit stages.
    encoded_blocks: Vec<EncodedBlock>,
    /// Per-term contiguous byte buffer holding meta + skip table +
    /// concatenated block bytes; written to `postings_writer` as one
    /// `write_counted` call.
    term_buf: Vec<u8>,
}

/// Encode one term's posting list and emit the resulting FST entry
/// into whichever sink the finish path uses. Exactly one of
/// `fst_entries_inram` / `fst_streaming` is `Some`; the function
/// dispatches accordingly.
///
/// Owns the per-term encoding policy (df=1 inline value, df≥2 PFOR
/// blocks via `encode_posting_group`).
#[allow(clippy::too_many_arguments)]
fn encode_and_emit_term<W: Write>(
    term: &str,
    pairs: &[(u32, u32)],
    col_name_bytes: &[u8],
    col_doc_lengths: &[u32],
    avgdl: f32,
    n_docs: u32,
    key_buf: &mut Vec<u8>,
    postings_writer: &mut W,
    postings_crc_acc: &mut u32,
    postings_len: &mut u64,
    fst_entries_inram: Option<&mut DictBuilder>,
    mut fst_streaming: Option<&mut StreamingDictBuilder<BufWriter<File>>>,
    profile: &mut FinishProfile,
    scratch: &mut TermScratch,
) -> Result<(), BuildError> {
    let encode_start = profile.enabled.then(Instant::now);
    profile.encode_calls += 1;
    // Build the FST key once; reused regardless of in-RAM vs spilled
    // emit policy.
    key_buf.clear();
    key_buf.extend_from_slice(col_name_bytes);
    key_buf.push(FST_SEPARATOR);
    key_buf.extend_from_slice(term.as_bytes());

    debug_assert!(
        pairs.windows(2).all(|w| w[0].0 < w[1].0),
        "posting list not sorted by doc_id"
    );

    let df = pairs.len() as u64;

    let fst_value: u64 = if df == 1 {
        profile.encode_df1 += 1;
        let (doc_id, tf) = pairs[0];
        FstValue::pack_inline(doc_id, tf)
    } else {
        profile.encode_pfor += 1;
        let idf_t = crate::superfile::fts::bm25::idf(n_docs as u64, df);
        // Reuse `scratch.encoded_blocks` / `scratch.min_dl_per_block`
        // across every dense term in the column (one allocation amortised
        // over ~5K terms / ~1M blocks at 1M docs, vs `Vec::new` per term).
        let encoded_blocks = &mut scratch.encoded_blocks;
        let min_dl_per_block = &mut scratch.min_dl_per_block;
        encoded_blocks.clear();
        min_dl_per_block.clear();
        let block_build_start = profile.enabled.then(Instant::now);
        // Build each block by moving the reusable `doc_ids` / `tfs`
        // buffers into a `Block` (Vec move = pointer swap, no copy),
        // encoding, then moving them back so the next chunk reuses the
        // same heap allocation. `encode_block` only borrows, but the
        // public signature takes `&Block` whose fields are owned Vecs,
        // hence the take/put dance. The take/put preserves capacity at
        // `BLOCK_LEN`, so the steady-state per-chunk cost is
        // `clear` + `extend_from_slice` of ≤128 u32s with no realloc.
        let mut block_doc_ids = std::mem::take(&mut scratch.doc_ids);
        let mut block_tfs = std::mem::take(&mut scratch.tfs);
        if block_doc_ids.capacity() < BLOCK_LEN {
            block_doc_ids.reserve(BLOCK_LEN - block_doc_ids.capacity());
        }
        if block_tfs.capacity() < BLOCK_LEN {
            block_tfs.reserve(BLOCK_LEN - block_tfs.capacity());
        }
        for chunk in pairs.chunks(BLOCK_LEN) {
            block_doc_ids.clear();
            block_tfs.clear();
            block_doc_ids.extend(chunk.iter().map(|&(d, _)| d));
            block_tfs.extend(chunk.iter().map(|&(_, t)| t));
            let min_dl = block_doc_ids
                .iter()
                .map(|d| col_doc_lengths[*d as usize])
                .min()
                .unwrap_or(0);
            min_dl_per_block.push(min_dl);
            let block = Block {
                doc_ids: std::mem::take(&mut block_doc_ids),
                tfs: std::mem::take(&mut block_tfs),
            };
            encoded_blocks.push(encode_block(&block));
            // Reclaim the underlying allocations for the next chunk.
            block_doc_ids = block.doc_ids;
            block_tfs = block.tfs;
        }
        scratch.doc_ids = block_doc_ids;
        scratch.tfs = block_tfs;
        if let Some(start) = block_build_start {
            profile.encode_block_build += start.elapsed();
        }
        let num_blocks = encoded_blocks.len() as u32;
        let metadata_offset = *postings_len;
        let skip_table_size = encoded_blocks.len() * SKIP_ENTRY_SIZE;
        let blocks_total_size: usize = encoded_blocks.iter().map(|b| b.bytes.len()).sum();
        let postings_length = (TERM_META_SIZE + skip_table_size + blocks_total_size) as u64;

        debug_assert!(df <= u32::MAX as u64, "df overflows u32");
        debug_assert!(
            postings_length <= u32::MAX as u64,
            "single-term posting > 4 GiB"
        );

        // Reuse `scratch.term_buf` across every dense term. `clear`
        // keeps the existing allocation; only a true grow (a term
        // larger than any previously seen) reallocs.
        let term_buf = &mut scratch.term_buf;
        term_buf.clear();
        if term_buf.capacity() < postings_length as usize {
            term_buf.reserve(postings_length as usize - term_buf.capacity());
        }
        let meta_write_start = profile.enabled.then(Instant::now);
        term_buf.extend_from_slice(&(df as u32).to_le_bytes());
        term_buf.extend_from_slice(&metadata_offset.to_le_bytes());
        term_buf.extend_from_slice(&(postings_length as u32).to_le_bytes());
        term_buf.extend_from_slice(&num_blocks.to_le_bytes());
        debug_assert_eq!(term_buf.len(), TERM_META_SIZE);
        if let Some(start) = meta_write_start {
            profile.encode_meta_write += start.elapsed();
        }

        let mut block_offset: u32 = (TERM_META_SIZE + skip_table_size) as u32;
        let skip_write_start = profile.enabled.then(Instant::now);
        for (i, blk) in encoded_blocks.iter().enumerate() {
            let max_bm25 = crate::superfile::fts::bm25::block_upper_bound(
                idf_t,
                blk.max_tf,
                min_dl_per_block[i],
                avgdl,
            );
            // ceil(): the stored fixed-point value must stay a true
            // UPPER bound after quantization — truncation would round
            // it below the real block max and let BMW / floor skips
            // drop blocks that still hold qualifying docs. (The reader
            // additionally adds one step on decode to cover files
            // written before this rounding fix.)
            let max_bm25_x1000 = (max_bm25 * format::fts::BLOCK_MAX_BM25_FIXED_POINT_SCALE)
                .ceil()
                .max(0.0)
                .min(u32::MAX as f32) as u32;
            term_buf.extend_from_slice(&blk.last_doc_id.to_le_bytes());
            term_buf.extend_from_slice(&block_offset.to_le_bytes());
            term_buf.extend_from_slice(&max_bm25_x1000.to_le_bytes());
            term_buf.extend_from_slice(&0u32.to_le_bytes());
            block_offset += blk.bytes.len() as u32;
        }
        if let Some(start) = skip_write_start {
            profile.encode_skip_write += start.elapsed();
        }

        let block_write_start = profile.enabled.then(Instant::now);
        for blk in encoded_blocks.iter() {
            term_buf.extend_from_slice(&blk.bytes);
        }
        debug_assert_eq!(term_buf.len(), postings_length as usize);
        write_counted(postings_writer, postings_crc_acc, postings_len, term_buf)?;
        if let Some(start) = block_write_start {
            profile.encode_block_write += start.elapsed();
        }

        FstValue::pack_pfor(metadata_offset, postings_length as u32)
    };

    let fst_insert_start = profile.enabled.then(Instant::now);
    if let Some(db) = fst_entries_inram {
        db.insert(key_buf, fst_value);
    } else if let Some(sb) = fst_streaming.as_mut() {
        sb.insert_sorted(key_buf, fst_value).map_err(map_fst_err)?;
    }
    if let Some(start) = fst_insert_start {
        profile.fst_insert += start.elapsed();
    }

    if let Some(start) = encode_start {
        profile.encode_total += start.elapsed();
    }

    Ok(())
}

fn write_counted<W: Write>(
    w: &mut W,
    crc_acc: &mut u32,
    len: &mut u64,
    bytes: &[u8],
) -> Result<(), BuildError> {
    w.write_all(bytes)?;
    *crc_acc = crc32c_append(*crc_acc, bytes);
    *len += bytes.len() as u64;
    Ok(())
}

/// Sort one partition file to a sorted-triple file at `out_path`.
/// Uses an in-memory sort when the partition is at or below
/// `max_partition_bytes`, otherwise external merge over chunked
/// sorted spills (partition-skew defense: a single pathologically
/// large partition would otherwise blow the in-memory sort budget).
///
/// Both the input and the output are runs of fixed 12-byte triples
/// `(term_id_le, doc_id_le, tf_le)`. The ordering written to
/// `out_path` is `(lex_rank[term_id], doc_id)` so a downstream k-way
/// merge over multiple sorted partitions produces global lex order
/// in one pass.
fn sort_partition_to_file(
    in_path: &Path,
    out_path: &Path,
    max_partition_bytes: u64,
    scratch_dir: &Path,
    partition_label: &str,
    lex_rank: &[u32],
) -> Result<(), BuildError> {
    let mut iter = open_partition_sorted(
        in_path,
        max_partition_bytes,
        scratch_dir,
        partition_label,
        lex_rank,
    )?;
    let mut w = BufWriter::with_capacity(PARTITION_BUF_SIZE, File::create(out_path)?);
    let mut batch: Vec<Triple> = Vec::with_capacity(SORT_OUTPUT_BATCH_TRIPLES);
    while let Some(triple) = iter.next_with(lex_rank) {
        let t = triple?;
        batch.push(t);
        if batch.len() == SORT_OUTPUT_BATCH_TRIPLES {
            #[cfg(target_endian = "little")]
            {
                w.write_all(bytemuck::cast_slice::<Triple, u8>(&batch))?;
            }
            #[cfg(not(target_endian = "little"))]
            for t in &batch {
                write_triple(&mut w, t[0], t[1], t[2])?;
            }
            batch.clear();
        }
    }
    if !batch.is_empty() {
        #[cfg(target_endian = "little")]
        {
            w.write_all(bytemuck::cast_slice::<Triple, u8>(&batch))?;
        }
        #[cfg(not(target_endian = "little"))]
        for t in &batch {
            write_triple(&mut w, t[0], t[1], t[2])?;
        }
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::default_tokenizer as tokenizer;

    #[test]
    fn register_column_returns_sequential_ids() {
        let mut b = FtsBuilder::new(tokenizer());
        assert_eq!(
            b.register_column("title".into()).expect("register column"),
            0
        );
        assert_eq!(
            b.register_column("body".into()).expect("register column"),
            1
        );
        assert_eq!(b.register_column("tag".into()).expect("register column"), 2);
    }

    #[test]
    fn register_column_rejects_separator_byte() {
        let mut b = FtsBuilder::new(tokenizer());
        let bad = String::from("ti\x1Ftle");
        let err = b.register_column(bad).expect_err("expected error");
        assert!(matches!(err, BuildError::ReservedSeparatorInColumnName(_)));
    }

    #[test]
    fn register_column_rejects_reserved_prefix() {
        let mut b = FtsBuilder::new(tokenizer());
        let err = b
            .register_column("inf.title".into())
            .expect_err("expected error");
        assert!(matches!(err, BuildError::ReservedPrefixInColumnName(_)));
    }

    #[test]
    fn register_column_rejects_duplicates() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("title".into()).expect("register column");
        let err = b
            .register_column("title".into())
            .expect_err("expected error");
        assert!(matches!(err, BuildError::DuplicateColumnName(_)));
    }

    #[test]
    fn add_doc_unknown_column_id_errors() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("title".into()).expect("register column");
        let err = b.add_doc(99, 0, "text").expect_err("expected error");
        assert!(matches!(err, BuildError::FtsColumnTypeInvalid { .. }));
    }

    #[tokio::test]
    async fn add_doc_accumulates_tf_within_doc() {
        use crate::superfile::fts::reader::{BoolMode, FtsReader};
        use bytes::Bytes;

        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("title".into()).expect("register column");
        b.add_doc(0, 0, "rust rust rust async").expect("add doc");

        let blob = Bytes::from(b.finish().expect("finish"));
        let r =
            FtsReader::open(blob, r#"[{"name":"title","tokenizer":"ascii_lower"}]"#).expect("open");
        let rust_hits = r
            .search("title", &["rust"], 10, BoolMode::Or)
            .await
            .expect("rust search");
        let async_hits = r
            .search("title", &["async"], 10, BoolMode::Or)
            .await
            .expect("async search");
        assert_eq!(rust_hits.len(), 1);
        assert_eq!(rust_hits[0].0, 0);
        assert_eq!(async_hits.len(), 1);
        assert_eq!(async_hits[0].0, 0);
    }

    #[tokio::test]
    async fn cross_column_same_term_stays_isolated_through_round_trip() {
        // A term that appears in two different columns must keep
        // its posting lists scoped per column in the emitted FST +
        // posting region. This also exercises the spill-backed
        // accumulator: column id is implicit in the selected partition
        // set, while the final FST key remains `<col>\x1F<term>`.
        use crate::superfile::fts::reader::{BoolMode, FtsReader};
        use bytes::Bytes;

        let mut b = FtsBuilder::new(tokenizer());
        let title_id = b.register_column("title".into()).expect("register title");
        let body_id = b.register_column("body".into()).expect("register body");

        // Doc 0: "rust" + "tokio" in title, "rust" + "async" in body.
        // Doc 1: only in body — "rust".
        // Doc 2: only in title — "rust".
        b.add_doc(title_id, 0, "rust tokio")
            .expect("add title doc 0");
        b.add_doc(body_id, 0, "rust async").expect("add body doc 0");
        b.add_doc(body_id, 1, "rust").expect("add body doc 1");
        b.add_doc(title_id, 1, "rust").expect("add title doc 1");

        // Round-trip through finish() + FtsReader::search. The
        // reader looks up via `dict::make_key(column, term)`, so this
        // is the strict on-disk equivalent of "two columns share a
        // term — does each see its own postings?"
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"title","tokenizer":"ascii_lower"},{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        // "rust" in title returns title's docs (0, 1) and no others.
        let hits_t = r
            .search("title", &["rust"], 10, BoolMode::Or)
            .await
            .expect("title search");
        let ids_t: Vec<u32> = hits_t.iter().map(|(d, _)| *d).collect();
        assert_eq!(ids_t.len(), 2, "title 'rust' hit count");
        assert!(ids_t.contains(&0));
        assert!(ids_t.contains(&1));

        // "rust" in body also returns its own docs (0, 1). Same ids
        // by coincidence; what matters is the search is scoped to
        // body's posting list, not title's.
        let hits_b = r
            .search("body", &["rust"], 10, BoolMode::Or)
            .await
            .expect("body search");
        let ids_b: Vec<u32> = hits_b.iter().map(|(d, _)| *d).collect();
        assert_eq!(ids_b.len(), 2, "body 'rust' hit count");
        assert!(ids_b.contains(&0));
        assert!(ids_b.contains(&1));

        // Cross-leak negative: a term that lives only in body
        // (`async`) must NOT be findable in title, and vice versa
        // (`tokio` in body).
        let hits_async_in_title = r
            .search("title", &["async"], 10, BoolMode::Or)
            .await
            .expect("title async search");
        assert!(
            hits_async_in_title.is_empty(),
            "title must not return 'async' (lives only in body)"
        );
        let hits_tokio_in_body = r
            .search("body", &["tokio"], 10, BoolMode::Or)
            .await
            .expect("body tokio search");
        assert!(
            hits_tokio_in_body.is_empty(),
            "body must not return 'tokio' (lives only in title)"
        );
    }

    #[test]
    fn add_doc_tracks_doc_lengths_clamped() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("body".into()).expect("register column");
        b.add_doc(0, 0, "alpha beta gamma").expect("add doc");
        b.add_doc(0, 1, "").expect("add doc"); // zero-token doc
        b.add_doc(0, 2, "delta").expect("add doc");
        let col = &b.columns[0];
        assert_eq!(col.doc_lengths, vec![3, 0, 1]);
        assert_eq!(col.total_tokens, 4);
    }

    #[test]
    fn add_doc_updates_n_docs_per_call() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("body".into()).expect("register column");
        // Contract: local_doc_id is consecutive from 0 (per column).
        // n_docs ends up == max(local_doc_id) + 1 == call count.
        b.add_doc(0, 0, "a").expect("add doc");
        b.add_doc(0, 1, "b").expect("add doc");
        b.add_doc(0, 2, "c").expect("add doc");
        assert_eq!(b.n_docs, 3);
    }

    #[test]
    fn finish_emits_valid_header() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("title".into()).expect("register column");
        b.add_doc(0, 0, "hello world").expect("add doc");
        let blob = b.finish().expect("finish");

        // Magic.
        assert_eq!(&blob[0..8], format::fts::MAGIC);
        // Version.
        let version = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]);
        assert_eq!(version, format::fts::VERSION);
        // n_columns.
        let n_cols = u32::from_le_bytes([blob[12], blob[13], blob[14], blob[15]]);
        assert_eq!(n_cols, 1);
        // n_docs (u32 at 16..20).
        let n_docs = u32::from_le_bytes([blob[16], blob[17], blob[18], blob[19]]);
        assert_eq!(n_docs, 1);
        // n_terms_total = 2 ("hello", "world") (u32 at 20..24).
        let n_terms = u32::from_le_bytes([blob[20], blob[21], blob[22], blob[23]]);
        assert_eq!(n_terms, 2);
        // fst_offset == 48 (u64 at 24..32).
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&blob[24..32]);
        let fst_off = u64::from_le_bytes(buf);
        assert_eq!(fst_off, 48);
    }

    /// Determinism gate: two independent in-RAM builds over the
    /// same corpus must emit byte-identical blobs. Catches hasher
    /// / iteration-order regressions on the in-RAM finish path.
    #[test]
    fn finish_to_matches_finish_byte_for_byte() {
        fn build() -> FtsBuilder {
            let mut b = FtsBuilder::new(tokenizer());
            b.register_column("title".into()).expect("register title");
            for (i, text) in [
                "rust async rust",
                "tokio runtime",
                "rust search engine",
                "async search",
            ]
            .iter()
            .enumerate()
            {
                b.add_doc(0, i as u32, text).expect("add doc");
            }
            b
        }

        let via_finish = build().finish().expect("finish");
        let mut via_finish_to = Vec::new();
        build()
            .finish_to(&mut via_finish_to)
            .expect("finish_to Vec");
        assert_eq!(via_finish_to, via_finish);
    }

    #[tokio::test]
    async fn finish_to_temp_file_round_trips_through_reader() {
        use crate::superfile::fts::reader::{BoolMode, FtsReader};
        use bytes::Bytes;
        use std::io::BufWriter;

        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("title".into()).expect("register title");
        for i in 0..256u32 {
            b.add_doc(0, i, &format!("common term{i:03}"))
                .expect("add doc");
        }

        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("fts.blob");
        {
            let file = File::create(&path).expect("create blob");
            let writer = BufWriter::new(file);
            b.finish_to(writer).expect("finish_to file");
        }
        let blob = std::fs::read(&path).expect("read blob");
        let r = FtsReader::open(
            Bytes::from(blob),
            r#"[{"name":"title","tokenizer":"ascii_lower"}]"#,
        )
        .expect("open FTS reader");
        let hits = r
            .search("title", &["common"], 10, BoolMode::Or)
            .await
            .expect("search");
        assert_eq!(hits.len(), 10);
    }

    #[test]
    fn finish_with_no_docs_still_produces_valid_blob() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("title".into()).expect("register column");
        let blob = b.finish().expect("finish");
        assert_eq!(&blob[0..8], format::fts::MAGIC);
        // n_docs == 0 (u32 at 16..20), n_terms_total == 0 (u32 at 20..24).
        assert_eq!(
            u32::from_le_bytes([blob[16], blob[17], blob[18], blob[19]]),
            0
        );
        assert_eq!(
            u32::from_le_bytes([blob[20], blob[21], blob[22], blob[23]]),
            0
        );
    }

    #[test]
    fn small_build_stays_in_ram_no_spill_files_created() {
        // Mirror of vector's "small build never touches the disk
        // during add_doc" gate. With the default spill threshold
        // (256 MiB) a 100-doc build can never cross it.
        let parent = tempfile::tempdir().expect("parent");
        let mut b = FtsBuilder::with_scratch(tokenizer(), parent.path().to_path_buf())
            .expect("with_scratch");
        b.register_column("body".into()).expect("register col");
        for i in 0..100u32 {
            b.add_doc(0, i, &format!("alpha beta gamma{i}"))
                .expect("add doc");
        }
        // Every column must still be in InRam mode.
        for cp in &b.postings {
            assert!(
                !cp.is_spilled(),
                "small build must not have spilled to disk"
            );
        }
        // And the scratch tempdir under the override must contain no
        // posting partition files yet (FtsBuilder's scratch tempdir is
        // a *sub*directory of `parent`).
        let mut spill_files_found = 0usize;
        for entry in walkdir_files(parent.path()) {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("fts_col") && name.ends_with(".bin") {
                spill_files_found += 1;
            }
        }
        assert_eq!(
            spill_files_found, 0,
            "small build must not pre-create posting spill files"
        );
        // Sanity: finish_to still produces a working blob via the
        // in-RAM path.
        let _blob = b.finish().expect("finish");
    }

    #[test]
    fn build_above_threshold_spills_and_matches_in_ram_byte_for_byte() {
        // Threshold mode = real test: a low spill threshold forces
        // the same corpus to take the spilled finish_to path; result
        // must match the in-RAM finish byte-for-byte. This is the
        // streaming-FST regression gate — the spilled path uses
        // `StreamingDictBuilder` writing to a scratch file, while
        // the in-RAM path uses the in-memory `DictBuilder`. Both
        // must produce identical FST bytes.
        fn build_corpus(b: &mut FtsBuilder) {
            b.register_column("body".into()).expect("register col");
            // 1000 docs, each unique → 1000+ distinct terms forces
            // partitions to fill if threshold is low.
            for i in 0..1000u32 {
                b.add_doc(
                    0,
                    i,
                    &format!("common shared term{i:04} payload{i:04} extra word{i:04}"),
                )
                .expect("add doc");
            }
        }

        let mut baseline = FtsBuilder::new(tokenizer());
        build_corpus(&mut baseline);
        // Baseline must stay in RAM.
        for cp in &baseline.postings {
            assert!(!cp.is_spilled(), "baseline must stay in RAM");
        }
        let baseline_blob = baseline.finish().expect("finish baseline");

        // Force spill via low threshold. 16 KiB is well below the
        // corpus's accumulator size. Pin the scratch dir under a
        // tempdir we control so we can inspect on-disk partition
        // files mid-build (counterpart to the negative assertion in
        // `small_build_stays_in_ram_no_spill_files_created`).
        let parent = tempfile::tempdir().expect("parent");
        let mut spilled = FtsBuilder::with_scratch(tokenizer(), parent.path().to_path_buf())
            .expect("with_scratch");
        spilled.set_spill_threshold_bytes(16 * 1024);
        build_corpus(&mut spilled);
        let any_spilled = spilled.postings.iter().any(|c| c.is_spilled());
        assert!(any_spilled, "low threshold must force spill");
        // Positive walkdir assert: at least one `fts_col*.bin`
        // partition spill file must exist before `finish()`
        // consumes the builder and drops the scratch tempdir. This
        // gates against future refactors that keep the `Spilled`
        // variant flag but bypass the on-disk write path.
        let mut spill_files_found = 0usize;
        for entry in walkdir_files(parent.path()) {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("fts_col") && name.ends_with(".bin") {
                spill_files_found += 1;
            }
        }
        assert!(
            spill_files_found > 0,
            "spilled build must materialise at least one fts_col*.bin partition file on disk"
        );
        let spilled_blob = spilled.finish().expect("finish spilled");

        assert_eq!(
            spilled_blob, baseline_blob,
            "streaming-FST + spill path must produce byte-identical blob"
        );
    }

    /// Walk a directory recursively yielding only files.
    /// Local helper used by `small_build_stays_in_ram_no_spill_files_created`;
    /// avoids pulling in a dev-dep on `walkdir` for one test.
    fn walkdir_files(root: &std::path::Path) -> Vec<std::fs::DirEntry> {
        let mut out = Vec::new();
        let mut stack: Vec<std::path::PathBuf> = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let rd = match std::fs::read_dir(&dir) {
                Ok(r) => r,
                Err(_) => continue,
            };
            for entry in rd.flatten() {
                let ft = match entry.file_type() {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                if ft.is_dir() {
                    stack.push(entry.path());
                } else if ft.is_file() {
                    out.push(entry);
                }
            }
        }
        out
    }

    #[test]
    fn external_merge_path_matches_in_memory_path_byte_for_byte() {
        // Drive the over-budget branch of `open_partition_sorted`: a
        // single very common term gets hashed into one partition and
        // its on-disk records exceed an aggressively low
        // `max_partition_bytes`, forcing chunked sort + k-way merge.
        // The encoded blob must match a baseline build that used the
        // default (effectively unbounded) per-partition budget.
        fn build_corpus(builder: &mut FtsBuilder) {
            builder
                .register_column("body".into())
                .expect("register col");
            // `common` appears in every doc → one term dominates one
            // hash partition. ~600 docs is enough that the per-record
            // bytes for that partition pass a 4 KiB budget.
            for i in 0..600u32 {
                builder
                    .add_doc(0, i, &format!("common term{i:04} payload{i:04}"))
                    .expect("add doc");
            }
        }

        // Baseline: force the *spilled* finish path (low spill
        // threshold), but leave `max_partition_bytes` at its
        // effectively-unbounded default so every partition fits in
        // memory and `open_partition_sorted` takes the in-memory
        // branch. This isolates the variable under test (in-memory
        // partition sort vs external merge) from the baseline's
        // identity (the spilled finish path).
        let mut baseline = FtsBuilder::new(tokenizer());
        baseline.set_spill_threshold_bytes(1);
        build_corpus(&mut baseline);
        let baseline_blob = baseline.finish().expect("finish baseline");

        // Tight budget forces external merge. The column must
        // spill (`set_spill_threshold_bytes(1)`) so the spilled
        // finish path is even taken; then 1 KiB per partition is
        // well below the dominant partition's on-disk size, so the
        // merge path is exercised on at least one partition.
        finish_debug::reset();
        let mut tight = FtsBuilder::new(tokenizer());
        tight.set_spill_threshold_bytes(1);
        tight.set_max_partition_bytes(1024);
        build_corpus(&mut tight);
        let tight_blob = tight.finish().expect("finish tight");

        assert_eq!(
            tight_blob, baseline_blob,
            "external-merge path must produce identical blob bytes"
        );
        // Positive gate: at least one sorted-chunk file must have
        // been written during the tight build. Without this assert
        // the test would pass trivially if a future refactor made
        // every partition fit in budget.
        let chunks = finish_debug::observed();
        assert!(
            !chunks.is_empty(),
            "external-merge path must have written at least one sorted-chunk file; \
             observed chunks were empty (test no longer exercises the over-budget branch)"
        );
    }

    #[test]
    fn scratch_dir_under_with_scratch_is_removed_after_finish() {
        // `with_scratch(PathBuf)` lets operators pin spill files to
        // instance-store NVMe. The `tempfile::TempDir` produced under the
        // override must still be cleaned up when the builder is
        // consumed by `finish`; if it isn't, repeated builds leak
        // disk. This test asserts the directory the builder created
        // under the override path is gone after the build.
        let parent = tempfile::tempdir().expect("parent tempdir");
        let dir_count_before = std::fs::read_dir(parent.path())
            .expect("read parent")
            .count();

        let mut b = FtsBuilder::with_scratch(tokenizer(), parent.path().to_path_buf())
            .expect("with_scratch");
        b.register_column("body".into()).expect("register col");
        b.add_doc(0, 0, "alpha beta gamma").expect("add doc");
        let _blob = b.finish().expect("finish");

        let dir_count_after = std::fs::read_dir(parent.path())
            .expect("read parent")
            .count();
        assert_eq!(
            dir_count_after, dir_count_before,
            "FtsBuilder scratch tempdir leaked under override path"
        );
    }

    #[tokio::test]
    async fn configurable_spill_partitions_round_trips_through_reader() {
        use crate::superfile::fts::reader::{BoolMode, FtsReader};
        use bytes::Bytes;

        // Higher partition count: more files, smaller per-partition
        // working set. Must still produce a queryable blob.
        let mut b = FtsBuilder::new(tokenizer());
        b.set_spill_partitions(256).expect("set partitions");
        b.register_column("body".into()).expect("register col");
        for i in 0..50u32 {
            b.add_doc(0, i, &format!("alpha beta gamma{i:02}"))
                .expect("add doc");
        }
        let blob = b.finish().expect("finish");
        let r = FtsReader::open(
            Bytes::from(blob),
            r#"[{"name":"body","tokenizer":"ascii_lower"}]"#,
        )
        .expect("open reader");
        let hits = r
            .search("body", &["alpha"], 100, BoolMode::Or)
            .await
            .expect("search alpha");
        assert_eq!(hits.len(), 50, "alpha is in every doc");
    }

    #[test]
    fn finish_offsets_are_consistent() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("body".into()).expect("register column");
        for i in 0..10 {
            b.add_doc(0, i, &format!("term{i} common"))
                .expect("add doc");
        }
        let blob = b.finish().expect("finish");

        // Header layout post-u32-narrowing: fst_offset at 24..32,
        // postings_offset at 32..40, doc_lengths_table_offset at 40..48.
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&blob[24..32]);
        let fst_off = u64::from_le_bytes(buf) as usize;
        buf.copy_from_slice(&blob[32..40]);
        let postings_off = u64::from_le_bytes(buf) as usize;
        buf.copy_from_slice(&blob[40..48]);
        let dir_off = u64::from_le_bytes(buf) as usize;

        assert_eq!(fst_off, 48);
        assert!(postings_off > fst_off, "postings after FST");
        assert!(dir_off > postings_off, "directory after postings");
        assert!(dir_off <= blob.len(), "directory offset within blob");
    }
}
