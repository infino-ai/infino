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
//! ## Build architecture (plan 017, mirrors plan 010)
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

use crate::superfile::BuildError;
use crate::superfile::format::checksum::{crc32c, crc32c_append};
use crate::superfile::format::{self, FST_SEPARATOR};
use crate::superfile::fts::dict::{DictBuilder, StreamingDictBuilder};
use crate::superfile::fts::fst_value::FstValue;
use crate::superfile::fts::posting::{BLOCK_LEN, Block, encode_block};
use crate::superfile::fts::tokenize::Tokenizer;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use memmap2::Mmap;

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
///   off  0 ..  4 : df (u32) — bounded by n_docs per segment
///   off  4 .. 12 : postings_offset (u64) — equals the term's metadata_offset;
///                  self-describing. u64 supports segments past 4 GiB
///                  (e.g. the 16 GB target).
///   off 12 .. 16 : postings_length (u32) — single term's bytes, well under
///                  4 G even at high df (≤ ~1 MB for the most common term in
///                  a 16 GB segment).
///   off 16 .. 20 : num_blocks (u32)
///
/// `df`, `postings_length`, and `num_blocks` stay u32; only the absolute
/// offset into the postings region needs the full u64 range.
const TERM_META_SIZE: usize = 20;

/// Skip-table entry size in bytes.
const SKIP_ENTRY_SIZE: usize = 16;

/// Doc-lengths directory entry size in bytes (per column).
///
/// Layout:
///   off  0 ..  4 : column_id (u32)
///   off  4 .. 12 : doc_lengths_offset (u64) — absolute offset of this column's
///                  doc-lengths array in the FTS blob. u64 supports segments
///                  past 4 GiB.
///   off 12 .. 16 : avgdl_x1000 (u32) — avgdl × 1000, as an integer
///
/// Only the absolute offset needs u64; column_id and avgdl_x1000 stay
/// u32 (bounded by column count and doc length respectively).
const DOC_LENGTHS_ENTRY_SIZE: usize = 16;

/// Default per-column in-RAM accumulator budget before a column
/// flushes to spill files. Mirrors `VectorBuilder::spill_threshold_bytes`
/// (also 256 MiB by default; plan 010). Builds whose every column
/// stays below this never touch the disk during `add_doc`.
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
        term_to_id: FxHashMap<Box<str>, u32>,
        id_to_term: Vec<Box<str>>,
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
const SPILL_BATCH_BYTES: usize = SPILL_BATCH_TRIPLES * TRIPLE_BYTES;

struct SpillPartition {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    /// 4 KiB batch buffer flushed into `writer` when full.
    /// `add_doc` appends 12-byte triples here in the hot path;
    /// `finish_to`'s flush-stage drains any partial buffer once
    /// per partition before the merge starts.
    batch: Vec<u8>,
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
const TRIPLE_BYTES: usize = 12;

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
fn write_triple<W: Write>(
    w: &mut W,
    term_id: u32,
    doc_id: u32,
    tf: u32,
) -> Result<(), BuildError> {
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
    let mut buf = [0u8; TRIPLE_BYTES];
    buf[0..4].copy_from_slice(&term_id.to_le_bytes());
    buf[4..8].copy_from_slice(&doc_id.to_le_bytes());
    buf[8..12].copy_from_slice(&tf.to_le_bytes());
    partition.batch.extend_from_slice(&buf);
    if partition.batch.len() >= SPILL_BATCH_BYTES {
        flush_partition_batch(partition)?;
    }
    Ok(())
}

/// Drain any pending bytes in a partition's batch buffer into its
/// `BufWriter`. Called from the hot path when the batch fills and
/// from `finish_to`'s flush stage so partial buffers reach disk
/// before the merge starts.
#[inline]
fn flush_partition_batch(
    partition: &mut SpillPartition,
) -> Result<(), BuildError> {
    if partition.batch.is_empty() {
        return Ok(());
    }
    let writer = partition
        .writer
        .as_mut()
        .expect("partition writer is open before finish");
    writer.write_all(&partition.batch)?;
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
                u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()),
                u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()),
                u32::from_le_bytes(bytes[off + 8..off + 12].try_into().unwrap()),
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
fn build_lex_rank(id_to_term: &[Box<str>]) -> (Vec<u32>, Vec<u32>) {
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
    fn next_with(
        &mut self,
        lex_rank: &[u32],
    ) -> Option<Result<Triple, BuildError>> {
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
fn write_triples_sorted(
    triples: &[Triple],
    path: &Path,
) -> Result<(), BuildError> {
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
    out_paths.push(path);
    Ok(())
}

/// Sort `triples` in place by `(lex_rank[term_id], doc_id)`
/// using an 8-pass LSB byte-bucket radix sort over packed u64
/// keys (`(lex_rank << 32) | doc_id`). Active byte count is
/// detected per call so a corpus whose `(vocab × n_docs)`
/// product fits in 40 bits only pays 5 passes, etc.
///
/// Why this beats `sort_unstable_by` (pdqsort):
///
/// - **No comparator overhead.** pdqsort's branchy compare
///   chases `lex_rank[term_id]` twice per call; this radix
///   does that lookup *once* per element in the key-build
///   pass and never again. At 1M triples × O(log n) compares
///   that's ~25M lex_rank lookups eliminated.
/// - **Auto-vectorizable key build.** The first pass is a
///   tight `for t in triples` loop that loads
///   `lex_rank[t[0] as usize]` per element. With our
///   `target-cpu=x86-64-v3` config LLVM lowers this to an
///   AVX2 `vpgatherdd` gather (8 u32 lanes per instruction)
///   plus shifts/packs. Eight elements processed per
///   gathered batch; gather throughput on Sapphire Rapids is
///   ~8 elements / 5 cycles.
/// - **O(n) per pass instead of O(n log n) total.** For our
///   1M-doc bench (1.2M triples per partition, 46-bit
///   packed-key range → 6 passes) the radix does ~7.2M
///   scatter writes per partition vs pdqsort's ~25M
///   compare+swap operations.
/// - **Stable key ordering.** LSB byte radix is stable, so
///   ties on `lex_rank` resolve to insertion order; insertion
///   order here equals `doc_id` order from the in-RAM batch
///   buffer, so the final ordering is exactly
///   `(lex_rank, doc_id)`. (`pdqsort` is unstable; we
///   recovered the tiebreak via `.then(doc_id.cmp(...))` —
///   no longer needed.)
///
/// Falls back to `sort_unstable_by` for n < 256, where the
/// radix bookkeeping (allocate parallel arrays, 256-bucket
/// histograms × ≥4 passes) costs more than just sorting
/// scalar.
fn radix_sort_triples_by_lex_rank(triples: &mut Vec<Triple>, lex_rank: &[u32]) {
    let n = triples.len();
    if n < 256 {
        triples.sort_unstable_by(|a, b| {
            lex_rank[triple_term_id(a) as usize]
                .cmp(&lex_rank[triple_term_id(b) as usize])
                .then(triple_doc_id(a).cmp(&triple_doc_id(b)))
        });
        return;
    }

    // Parallel arrays: packed sort key (u64) + tf payload + term_id payload.
    // Single fused build pass — LLVM lowers
    // `lex_rank[t[0] as usize]` to AVX2 vpgatherdd at the
    // configured target-cpu.
    let mut keys: Vec<u64> = Vec::with_capacity(n);
    let mut tfs: Vec<u32> = Vec::with_capacity(n);
    let mut term_ids: Vec<u32> = Vec::with_capacity(n);
    let mut max_key: u64 = 0;
    for t in triples.iter() {
        let term_id = t[0];
        let doc_id = t[1];
        let tf = t[2];
        let rank = lex_rank[term_id as usize];
        let key = ((rank as u64) << 32) | (doc_id as u64);
        keys.push(key);
        tfs.push(tf);
        term_ids.push(term_id);
        if key > max_key {
            max_key = key;
        }
    }
    // Skip leading-zero byte passes.
    let n_passes = if max_key == 0 {
        1
    } else {
        ((64 - max_key.leading_zeros() + 7) / 8) as usize
    };

    let mut keys2: Vec<u64> = vec![0u64; n];
    let mut tfs2: Vec<u32> = vec![0u32; n];
    let mut term_ids2: Vec<u32> = vec![0u32; n];

    for pass in 0..n_passes {
        let shift = (pass * 8) as u32;

        // Histogram. Two-pass histogram intentionally — first
        // a write of zeros (`[0u32; 256]` initialiser, lowered
        // to AVX2 vmovaps), then the count loop. Hot inner
        // body is `hist[byte] += 1` which the AVX2 backend
        // can't vectorise (data dep on hist[byte]), but the
        // loop body is tight enough that branch prediction
        // and L1 hits make it ~1 cycle/element in practice.
        let mut hist = [0u32; 256];
        for &k in &keys {
            hist[((k >> shift) & 0xff) as usize] += 1;
        }

        // Prefix sum (in place). 256 elements, runs once per
        // pass — negligible.
        let mut sum: u32 = 0;
        for c in hist.iter_mut() {
            let tmp = *c;
            *c = sum;
            sum = sum.wrapping_add(tmp);
        }

        // Scatter. Carries the dominant cost; data dep on
        // hist[bucket]++ prevents vectorisation, but the 4
        // memory streams (keys2, tfs2, term_ids2, hist) all
        // sit in L1/L2 for typical partition sizes.
        for i in 0..n {
            let k = keys[i];
            let bucket = ((k >> shift) & 0xff) as usize;
            let dst = hist[bucket] as usize;
            keys2[dst] = k;
            tfs2[dst] = tfs[i];
            term_ids2[dst] = term_ids[i];
            hist[bucket] += 1;
        }

        std::mem::swap(&mut keys, &mut keys2);
        std::mem::swap(&mut tfs, &mut tfs2);
        std::mem::swap(&mut term_ids, &mut term_ids2);
    }

    // If `n_passes` was even, the sorted data is in
    // `keys`/`tfs`/`term_ids` (the originals after even-many
    // swaps). For odd `n_passes`, it's also there because the
    // final iteration's swap moved the sorted output back. So
    // we always read from the "primary" arrays here.
    triples.clear();
    triples.reserve(n);
    for i in 0..n {
        // Low 32 bits of the packed sort key carry doc_id.
        triples.push([term_ids[i], keys[i] as u32, tfs[i]]);
    }
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
    let mut chunk: Vec<Triple> = Vec::with_capacity(chunk_triples.min(1024 * 1024));
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
    /// and dedupes via `doc_tf_scratch` keyed by `term_id` instead.
    bump: bumpalo::Bump,
    /// Per-doc `term_id -> tf` dedup scratch, reused across every
    /// spill-mode `add_doc` call. Cleared (`.clear()`, retain
    /// capacity) on each entry instead of being freshly allocated,
    /// saving 1M `FxHashMap::default()` + grow + drop cycles on
    /// the 1M-doc bench. Only used by the spill arm; the in-RAM
    /// arm stages tokens through bumpalo + a local
    /// `HashMap<&str, u32>` instead (so `&str` keys can be cheaply
    /// promoted to `Box<str>` for the `terms` map without an
    /// extra copy).
    doc_tf_scratch: FxHashMap<u32, u32>,
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
    /// Mirror of `VectorBuilder::with_scratch` under plan 010, same
    /// return type (`Result<Self, BuildError>`).
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
            doc_tf_scratch: FxHashMap::default(),
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
                format!(
                    "FtsBuilder: spill_partitions must be a power of two; got {n}"
                ),
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
        let _ = column_id;
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
                batch: Vec::with_capacity(SPILL_BATCH_BYTES),
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
        term_to_id: &mut FxHashMap<Box<str>, u32>,
        id_to_term: &mut Vec<Box<str>>,
    ) -> Result<(), BuildError> {
        let n_part = partitions.len();
        debug_assert!(
            n_part.is_power_of_two(),
            "spill_partitions must be a power of 2; got {n_part}"
        );
        let mask = n_part - 1;
        for (term, postings) in terms {
            let term_id = match term_to_id.get(&term) {
                Some(&id) => id,
                None => {
                    let id = id_to_term.len() as u32;
                    id_to_term.push(term.clone());
                    term_to_id.insert(term, id);
                    id
                }
            };
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

        // Split borrows: `tokenize_each` calls into `self.tokenizer`
        // and each per-arm closure captures the per-arm accumulator.
        // Disjoint immutable borrow of `self.tokenizer` plus a
        // mutable borrow of one of `self.postings[col_idx]` or
        // `self.bump` — both ok by field-split borrow rules.
        let tokenizer = &self.tokenizer;
        let mut tokens_in_doc: u64 = 0;

        // Two-mode accumulator (mirror of vector). If this column
        // is already in spill mode, intern + write straight to its
        // partition files without going through the bumpalo +
        // `tf_per_term: HashMap<&str, u32>` intermediate. Otherwise
        // (in-RAM mode) stage tokens in the bumpalo + tf_per_term
        // pair (so we can stash `&str` keys into `terms: HashMap<
        // Box<str>, ...>` without per-token boxing), then check
        // whether this batch would cross the spill threshold and
        // transition if so.
        let column_id = col_idx as u32;
        // Split-borrow: spill arm needs `&mut self.doc_tf_scratch`
        // alongside `&mut self.postings[col_idx]`. Pre-borrow the
        // scratch here (disjoint field from `postings`) so the
        // `match col_post { ... }` arm can capture both. Cleared
        // (capacity retained) so the per-doc dedup map reuses its
        // hash buckets instead of allocating a fresh table per
        // call.
        let doc_tf = &mut self.doc_tf_scratch;
        doc_tf.clear();
        let col_post = &mut self.postings[col_idx];
        match col_post {
            ColumnPostings::Spilled {
                partitions,
                term_to_id,
                id_to_term,
            } => {
                let n_part = partitions.len();
                debug_assert!(
                    n_part.is_power_of_two(),
                    "spill_partitions must be a power of 2"
                );
                let mask = n_part - 1;

                // Per-doc dedup keyed by `term_id` (u32). No
                // `Box<str>` keys, no bumpalo, no `tf_per_term:
                // HashMap<&str, u32>` intermediate: we intern
                // per-token using the column's already-warm
                // `term_to_id`, sum per-doc term frequencies in
                // the reusable `doc_tf` scratch, then drain
                // straight to spill at the end of the doc.
                //
                // Saves on the spill hot path (1M docs × ~150
                // tokens/doc on the bench):
                //   - 150M `bumpalo::alloc_str` calls
                //   - 150M `Box<str>::from(&str)` boxings
                //   - 150M `tf_per_term.entry(&'static str)` probes
                //   - 1M  `FxHashMap::default()` allocations
                //     (scratch is reused across calls; cleared
                //     above)
                //
                // Term lookup is `term_to_id.get(tok)`: 1 probe
                // on hit (Zipfian vocab plateaus quickly so hit
                // dominates), 2 probes on miss (`get` + `insert`).
                // The `Box<str>::from(tok)` allocation only fires
                // on the miss branch.
                tokenizer.tokenize_each(text, &mut |tok| {
                    tokens_in_doc += 1;
                    let term_id = match term_to_id.get(tok) {
                        Some(&id) => id,
                        None => {
                            let id = id_to_term.len() as u32;
                            let boxed: Box<str> = tok.into();
                            id_to_term.push(boxed.clone());
                            term_to_id.insert(boxed, id);
                            id
                        }
                    };
                    *doc_tf.entry(term_id).or_insert(0) += 1;
                });

                // Update column doc-lengths + accounting. The
                // borrow of `self.columns[col_idx]` is disjoint
                // from the `&mut self.postings[col_idx]` we still
                // hold via `partitions` etc. (different fields of
                // `self`), so this is split-borrow legal.
                let col = &mut self.columns[col_idx];
                let dl_clamped: u32 = tokens_in_doc.min(u32::MAX as u64) as u32;
                col.doc_lengths.push(dl_clamped);
                col.total_tokens = col.total_tokens.saturating_add(tokens_in_doc);
                let docs_now = local_doc_id.saturating_add(1);
                if docs_now > self.n_docs {
                    self.n_docs = docs_now;
                }

                // Iterate by reference; the scratch is owned by
                // `self.doc_tf_scratch` and stays allocated for
                // the next `add_doc` call (cleared at the top).
                for (&term_id, &tf) in doc_tf.iter() {
                    let p = (term_id as usize) & mask;
                    push_triple_batched(
                        &mut partitions[p],
                        term_id,
                        local_doc_id,
                        tf,
                    )?;
                }
            }
            ColumnPostings::InRam { .. } => {
                // Re-borrow `terms` + `bytes` inside this arm via a
                // separate match below so we can split-borrow
                // `self.columns[col_idx]` (a disjoint field) in
                // between. The outer destructure is unused.
                //
                // Reset the per-shard bump arena so leftover token
                // bytes from the prior `add_doc` call are
                // invalidated before we reuse the chunk.
                // `Bump::reset` keeps the largest chunk (no
                // system-allocator round trip on the typical
                // steady-state doc) and frees any extra chunks the
                // pathological-long doc grew.
                //
                // We only reset (and use) the bump in the in-RAM
                // arm because the spill arm interns each token
                // straight into `term_to_id` and never needs a
                // backing `&str` allocation that outlives the
                // tokenizer callback.
                self.bump.reset();
                let bump = &self.bump;

                let mut tf_per_term: HashMap<&str, u32> = HashMap::new();
                tokenizer.tokenize_each(text, &mut |tok| {
                    tokens_in_doc += 1;
                    // alloc_str copies the borrowed token bytes
                    // into the bump. The returned `&str` outlives
                    // the next callback call (bump-arena
                    // lifetime), unlike the input `tok` which
                    // doesn't.
                    let bumped: &str = bump.alloc_str(tok);
                    // SAFETY-equivalent: widen the lifetime from
                    // the bump's borrow to a `'static` tag tied
                    // to the HashMap's lifetime. `tf_per_term` is
                    // a local that drops at the end of `add_doc`
                    // — well before `self.bump` is reset on the
                    // next call — so every key in the HashMap
                    // stays valid for the HashMap's full
                    // lifetime.
                    let extended: &'static str =
                        unsafe { std::mem::transmute(bumped) };
                    *tf_per_term.entry(extended).or_insert(0) += 1;
                });

                let col = &mut self.columns[col_idx];
                let dl_clamped: u32 = tokens_in_doc.min(u32::MAX as u64) as u32;
                col.doc_lengths.push(dl_clamped);
                col.total_tokens = col.total_tokens.saturating_add(tokens_in_doc);
                let docs_now = local_doc_id.saturating_add(1);
                if docs_now > self.n_docs {
                    self.n_docs = docs_now;
                }
                // Re-borrow the postings slot mutably after the
                // disjoint borrow of `self.columns[col_idx]` so
                // the InRam-mode insert + threshold-check block
                // below has the right `&mut` shape.
                let col_post = &mut self.postings[col_idx];
                let (terms, bytes) = match col_post {
                    ColumnPostings::InRam { terms, bytes } => (terms, bytes),
                    // We're already inside the `InRam` arm of the
                    // outer match; no other variant is reachable
                    // before we transition below.
                    ColumnPostings::Spilled { .. } => unreachable!(
                        "column was InRam at top of add_doc; cannot have spilled mid-call"
                    ),
                };
                // Insert via `Entry` API: one HashMap probe per
                // posting (vs the old two probes — `contains_key`
                // pre-check + `entry().or_default()`). At 100
                // terms/doc × 1M docs this halves the in-RAM
                // hot-path HashMap traffic and is the dominant
                // win on the rayon multi-thread path (where
                // segments are small enough to never spill but
                // were paying double-probe per posting). Bytes
                // accounting is folded into the same arm so the
                // threshold check is a single `>` after the
                // insert loop, no separate scan over
                // `tf_per_term.keys()`.
                let mut new_bytes: usize = 0;
                for (term, tf) in tf_per_term {
                    use std::collections::hash_map::Entry;
                    let term_len = term.len();
                    match terms.entry(term.into()) {
                        Entry::Vacant(e) => {
                            e.insert(vec![(local_doc_id, tf)]);
                            new_bytes = new_bytes.saturating_add(
                                ACCUM_NEW_TERM_FIXED_BYTES
                                    + term_len
                                    + ACCUM_POSTING_BYTES,
                            );
                        }
                        Entry::Occupied(mut e) => {
                            e.get_mut().push((local_doc_id, tf));
                            new_bytes =
                                new_bytes.saturating_add(ACCUM_POSTING_BYTES);
                        }
                    }
                }
                let new_total = bytes.saturating_add(new_bytes);

                if new_total > self.spill_threshold_bytes {
                    // We just crossed the threshold. Drain the
                    // (now-just-grown) in-RAM map into a fresh
                    // set of spill files for this column,
                    // building the term interner from the drained
                    // vocabulary; transition the column to
                    // `Spilled` so subsequent `add_doc` calls
                    // skip the in-RAM hashmap entirely. The
                    // current batch's postings are already in the
                    // map and flushed alongside everything else,
                    // so no separate batch-to-spill step is
                    // needed (same shape as
                    // VectorBuilder's pre_spill_buffer transition).
                    let drained = std::mem::take(terms);
                    let mut partitions = Self::open_partitions_for_column(
                        self.scratch_dir.path(),
                        column_id,
                        self.spill_partitions,
                    )?;
                    let mut term_to_id: FxHashMap<Box<str>, u32> =
                        FxHashMap::default();
                    let mut id_to_term: Vec<Box<str>> = Vec::new();
                    Self::flush_in_ram_to_partitions(
                        drained,
                        &mut partitions,
                        &mut term_to_id,
                        &mut id_to_term,
                    )?;
                    *col_post = ColumnPostings::Spilled {
                        partitions,
                        term_to_id,
                        id_to_term,
                    };
                } else {
                    *bytes = new_total;
                }
            }
        }

        Ok(())
    }

    /// Finalise and emit the FTS blob bytes. Consumes the builder.
    ///
    /// Returns `BuildError::Io` for scratch IO failures that can
    /// occur on the spill path (partition write/read, streaming-FST
    /// scratch file, posting region scratch file). Mirror of
    /// `VectorBuilder::finish`, which has the same return type for
    /// the same reason after plan 010.
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
    /// (regression-gated by `finish_to_matches_finish_byte_for_byte`).
    pub fn finish_to<W: Write>(self, mut w: W) -> Result<(), BuildError> {
        // Destructure `self` up front. Mirror of vector's
        // `let VectorBuilder { columns, scratch_dir, .. } = self;`
        // at the top of `VectorBuilder::finish_to`: makes per-field
        // ownership explicit, lets the loop body consume `columns`
        // and `postings` by value without partial-move bookkeeping,
        // and surfaces unused fields (`tokenizer`, `bump`,
        // `spill_threshold_bytes`, `spill_partitions`) as `_`
        // bindings rather than dead `self.field` references.
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
            doc_tf_scratch: _,
        } = self;

        let n_columns = columns.len() as u32;
        let mut n_terms_total_usize: usize = 0;

        // Decide which finish path to take. Any column in `Spilled`
        // mode forces the streaming-FST path; otherwise everything
        // fits in RAM and we use the in-RAM FST builder.
        let any_spilled = postings.iter().any(|c| c.is_spilled());

        // Build the per-column work list as `(original_column_id,
        // ColumnState, ColumnPostings)`, sorted by lex name. Draining
        // this `Vec` in the finish loop consumes one entry per
        // iteration *by value*, so each column's `ColumnPostings`
        // (its partition writers and any in-RAM term map) is dropped
        // the instant its terms have been emitted. Mirror of vector's
        // `columns.into_iter().enumerate()` pattern in
        // `VectorBuilder::finish_to`, which releases per-column
        // reservoir + pre-spill buf + spill file as each subsection
        // is built.
        let mut work: Vec<(usize, ColumnState, ColumnPostings)> = columns
            .into_iter()
            .zip(postings)
            .enumerate()
            .map(|(orig_idx, (state, posting_state))| (orig_idx, state, posting_state))
            .collect();
        // n_columns is small (typically < 16), so the sort itself
        // doesn't matter for wall time — `sort_unstable_by` here is
        // for style consistency with the hot posting-record sorts.
        work.sort_unstable_by(|a, b| a.1.name.cmp(&b.1.name));

        // Per-column avgdl (×1000 fixed-point per spec), keyed by
        // *original* column id so the doc-lengths directory below
        // can iterate in declaration order regardless of lex order.
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
        // through to `w` at assembly time. Lives under `scratch_dir`
        // so it's cleaned up by the tempdir drop at the end of
        // `finish_to`.
        let postings_path = scratch_path.join("infino_fts_postings.bin");
        let mut postings_writer = BufWriter::new(File::create(&postings_path)?);
        let mut postings_len: u64 = 0;
        let mut postings_crc_acc: u32 = 0;
        let mut key_buf: Vec<u8> = Vec::with_capacity(64);
        let mut finish_profile = FinishProfile::from_env();

        // Path-dependent FST destination:
        //   - In-RAM path: collect (key, value) into a `DictBuilder`
        //     and serialise once at the end (`fst_entries_inram`).
        //   - Spilled path: a `StreamingDictBuilder` writes FST bytes
        //     into a scratch file as we go (`fst_streaming`).
        // Exactly one is `Some`.
        let mut fst_entries_inram: Option<DictBuilder> =
            (!any_spilled).then(DictBuilder::new);
        let fst_streaming_path = scratch_path.join("infino_fts_dict.bin");
        let mut fst_streaming: Option<StreamingDictBuilder<BufWriter<File>>> = if any_spilled {
            let fst_file = File::create(&fst_streaming_path)?;
            let bw = BufWriter::new(fst_file);
            Some(StreamingDictBuilder::new(bw).map_err(map_fst_err)?)
        } else {
            None
        };

        // Drain every spilled column's per-partition batch buffer
        // into the partition's `BufWriter`, then flush + close
        // the `BufWriter` so reads see a complete file. In-RAM
        // columns are no-ops.
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

        // Per-column doc-lengths held back so the (avgdl + array)
        // assembly at the end can iterate by original column id
        // independent of work-list ordering. Each entry is moved out
        // of `work` as that column finishes, then consumed once at
        // assembly time.
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
                    let mut entries: Vec<(Box<str>, Vec<(u32, u32)>)> =
                        terms.into_iter().collect();
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
                            fst_entries_inram.as_mut(),
                            fst_streaming.as_mut(),
                            &mut finish_profile,
                        )?;
                        n_terms_total_usize += 1;
                    }
                }
                ColumnPostings::Spilled {
                    partitions,
                    term_to_id,
                    id_to_term,
                } => {
                    // Term interner is finished being written to;
                    // drop the forward map (`term_to_id`) immediately
                    // — the rest of the spilled finish only needs
                    // the reverse map (`id_to_term`) for FST emit and
                    // the lex-rank table built from it.
                    drop(term_to_id);
                    let (lex_rank, term_id_in_lex_order) =
                        build_lex_rank(&id_to_term);

                    // Pre-sort every partition to a sorted-triple
                    // file under scratch. Sorting partition-at-a-
                    // time bounds the in-RAM sort working set to
                    // `max_partition_bytes` (one partition at a
                    // time), then the k-way merge across the
                    // resulting sorted files runs with
                    // O(n_partitions) cursors each holding one
                    // triple + a small read buffer.
                    let mut sorted_files: Vec<PathBuf> =
                        Vec::with_capacity(partitions.len());
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
                        let term_bytes = id_to_term[term_id as usize].as_ref();
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
                            fst_entries_inram.as_mut(),
                            fst_streaming.as_mut(),
                            &mut finish_profile,
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
                            (finish_profile.encode_block_build - encode_block_build_before).as_secs_f64(),
                            (finish_profile.encode_meta_write - encode_meta_write_before).as_secs_f64(),
                            (finish_profile.encode_skip_write - encode_skip_write_before).as_secs_f64(),
                            (finish_profile.encode_block_write - encode_block_write_before).as_secs_f64(),
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
                    drop(id_to_term);
                    drop(lex_rank);
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

        debug_assert!(
            n_terms_total_usize <= u32::MAX as usize,
            "term count overflows u32"
        );
        let n_terms_total = n_terms_total_usize as u32;

        // Close the posting body file (CRC trailer + flush).
        let postings_crc = postings_crc_acc;
        let postings_crc_le = postings_crc.to_le_bytes();
        postings_writer.write_all(&postings_crc_le)?;
        postings_writer.flush()?;
        drop(postings_writer);
        postings_len += postings_crc_le.len() as u64;

        // Finalise the FST. Either path produces "FST bytes followed
        // by 4 trailing CRC bytes"; the source differs.
        enum FstSource {
            InRam(Vec<u8>),
            Streamed { path: PathBuf, len: u64, crc: u32 },
        }
        let fst_source = if let Some(db) = fst_entries_inram.take() {
            let mut bytes = db.finish();
            let crc = crc32c(&bytes);
            bytes.extend_from_slice(&crc.to_le_bytes());
            FstSource::InRam(bytes)
        } else {
            let sb = fst_streaming.take().expect("streaming FST must be present");
            let mut bw = sb.finish().map_err(map_fst_err)?;
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
        };

        // Compute final-blob offsets now that both region lengths are known.
        let fst_total_len: u64 = match &fst_source {
            FstSource::InRam(bytes) => bytes.len() as u64,
            FstSource::Streamed { len, .. } => *len,
        };
        let header_size: u64 = 48;
        let fst_offset: u64 = header_size;
        let postings_offset: u64 = fst_offset + fst_total_len;
        let doc_lengths_table_offset: u64 = postings_offset + postings_len;
        let mut doc_lengths_array_offset: u64 =
            doc_lengths_table_offset + (n_columns as u64) * (DOC_LENGTHS_ENTRY_SIZE as u64) + 4 /* dir CRC */;

        let mut dir_buf: Vec<u8> = Vec::with_capacity(n_columns as usize * DOC_LENGTHS_ENTRY_SIZE);
        let mut arrays_buf: Vec<u8> = Vec::new();
        for i in 0..n_columns as usize {
            let avgdl_x1000 = (avgdl_per_col[i] * 1000.0).max(0.0).min(u32::MAX as f32) as u32;
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

        // Final assembly. Bytes flow scratch → small streaming buffer
        // → `w`, never re-materialising the full blob in RAM.
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
                std::io::copy(&mut reader, &mut w)?;
                w.write_all(&crc.to_le_bytes())?;
            }
        }
        let mut postings_reader = BufReader::with_capacity(PARTITION_BUF_SIZE, File::open(&postings_path)?);
        std::io::copy(&mut postings_reader, &mut w)?;
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

        Ok(())
    }
}

#[inline]
fn map_fst_err(e: fst::Error) -> BuildError {
    BuildError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
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
        let mut encoded_blocks: Vec<crate::superfile::fts::posting::EncodedBlock> = Vec::new();
        let mut min_dl_per_block: Vec<u32> = Vec::new();
        let block_build_start = profile.enabled.then(Instant::now);
        for chunk in pairs.chunks(BLOCK_LEN) {
            let doc_ids: Vec<u32> = chunk.iter().map(|&(d, _)| d).collect();
            let tfs: Vec<u32> = chunk.iter().map(|&(_, t)| t).collect();
            let min_dl = doc_ids
                .iter()
                .map(|d| col_doc_lengths[*d as usize])
                .min()
                .unwrap_or(0);
            min_dl_per_block.push(min_dl);
            encoded_blocks.push(encode_block(&Block { doc_ids, tfs }));
        }
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

        let meta_write_start = profile.enabled.then(Instant::now);
        let mut term_buf: Vec<u8> = Vec::with_capacity(postings_length as usize);
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
            let max_bm25_x1000 = (max_bm25 * 1000.0).max(0.0).min(u32::MAX as f32) as u32;
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
        for blk in &encoded_blocks {
            term_buf.extend_from_slice(&blk.bytes);
        }
        debug_assert_eq!(term_buf.len(), postings_length as usize);
        write_counted(postings_writer, postings_crc_acc, postings_len, &term_buf)?;
        if let Some(start) = block_write_start {
            profile.encode_block_write += start.elapsed();
        }

        FstValue::pack_pfor(metadata_offset)
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
/// sorted spills (mirror of the partition-skew defense documented in
/// plan 017's "Skew control" section).
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
    let mut batch: Vec<Triple> = Vec::with_capacity(4096);
    while let Some(triple) = iter.next_with(lex_rank) {
        let t = triple?;
        batch.push(t);
        if batch.len() == 4096 {
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

    #[test]
    fn add_doc_accumulates_tf_within_doc() {
        use crate::superfile::fts::reader::{BoolMode, FtsReader};
        use bytes::Bytes;

        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("title".into()).expect("register column");
        b.add_doc(0, 0, "rust rust rust async").expect("add doc");

        let blob = Bytes::from(b.finish().expect("finish"));
        let r = FtsReader::open(blob, r#"[{"name":"title","tokenizer":"ascii_lower"}]"#)
            .expect("open");
        let rust_hits = r
            .search("title", &["rust"], 10, BoolMode::Or)
            .expect("rust search");
        let async_hits = r
            .search("title", &["async"], 10, BoolMode::Or)
            .expect("async search");
        assert_eq!(rust_hits.len(), 1);
        assert_eq!(rust_hits[0].0, 0);
        assert_eq!(async_hits.len(), 1);
        assert_eq!(async_hits[0].0, 0);
    }

    #[test]
    fn cross_column_same_term_stays_isolated_through_round_trip() {
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
            .expect("title async search");
        assert!(
            hits_async_in_title.is_empty(),
            "title must not return 'async' (lives only in body)"
        );
        let hits_tokio_in_body = r
            .search("body", &["tokio"], 10, BoolMode::Or)
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

    #[test]
    fn finish_to_temp_file_round_trips_through_reader() {
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
        let mut b =
            FtsBuilder::with_scratch(tokenizer(), parent.path().to_path_buf())
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
                    &format!(
                        "common shared term{i:04} payload{i:04} extra word{i:04}"
                    ),
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
        // corpus's accumulator size.
        let mut spilled = FtsBuilder::new(tokenizer());
        spilled.set_spill_threshold_bytes(16 * 1024);
        build_corpus(&mut spilled);
        let any_spilled = spilled.postings.iter().any(|c| c.is_spilled());
        assert!(any_spilled, "low threshold must force spill");
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

        let mut baseline = FtsBuilder::new(tokenizer());
        build_corpus(&mut baseline);
        let baseline_blob = baseline.finish().expect("finish baseline");

        // Tight budget forces external merge. 1 KiB is well below
        // the dominant partition's on-disk size, so the merge path
        // is exercised on at least one partition.
        let mut tight = FtsBuilder::new(tokenizer());
        tight.set_max_partition_bytes(1024);
        build_corpus(&mut tight);
        let tight_blob = tight.finish().expect("finish tight");

        assert_eq!(
            tight_blob, baseline_blob,
            "external-merge path must produce identical blob bytes"
        );
    }

    #[test]
    fn scratch_dir_under_with_scratch_is_removed_after_finish() {
        // `with_scratch(PathBuf)` lets operators pin spill files to
        // instance-store NVMe (see plan 017's "Scratch directory
        // placement"). The `tempfile::TempDir` produced under the
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

    #[test]
    fn configurable_spill_partitions_round_trips_through_reader() {
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
