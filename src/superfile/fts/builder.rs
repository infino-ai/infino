// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS blob builder. Multi-column FTS index assembly.
//!
//! `FtsBuilder` accumulates posting records across all FTS-indexed
//! columns and on `finish_to<W>` emits the on-disk FTS blob:
//!
//! ```text
//!   header (56 bytes)
//!   term dictionary                + CRC32C
//!   postings region                + CRC32C
//!   positions region               + CRC32C  (empty without positions)
//!   doc-id map                     + CRC32C  (V8 only)
//!   doc-lengths directory          + CRC32C
//!   per-column doc-lengths arrays  (each + its own CRC32C)
//! ```
//!
//! `format::fts` documents each region; `docs/architecture/superfile.md`
//! has the full byte-level spec.
//!
//! ## Build architecture
//!
//! Two-mode accumulator, threshold-based, identical in shape to
//! `VectorBuilder`:
//!
//! Every column interns its terms into a dense id space as they are
//! tokenized (`term_to_id` / `id_to_term`, bytes in a per-column arena)
//! and dedups a document's tokens through arrays indexed by that id.
//! Where the postings then go depends on the column's size:
//!
//! - **In-RAM mode**: each id's postings and position runs sit in
//!   per-id vectors until the column's estimated bytes cross
//!   `spill_threshold_bytes` (default 256 MiB; a supertable commit
//!   raises it to what the build reserved for the shard's text).
//!   The finish sorts the ids by term bytes once and encodes each
//!   list in place — no disk is touched during `add_doc` or the sort.
//! - **Spill mode**: once the threshold is crossed, the per-id lists
//!   are drained into per-column hash-partitioned spill files holding
//!   **fixed-size 12-byte `(term_id_le, doc_id_le, tf_le)` triples**
//!   (20-byte records with the run offset for a positional column),
//!   and from then on `add_doc` writes one record per posting straight
//!   to the spill files via buffered file IO. Same shape as vector's
//!   spill: no per-record framing, no variable-length payload, no
//!   per-record allocation on read.
//!
//! `finish_to<W: Write>` correspondingly has three paths:
//!
//! - **In-RAM finish**: no column spilled. Each column's ids are
//!   sorted by their term bytes once and every list is encoded in
//!   place into a posting-region scratch file; the term dictionary
//!   is built in RAM (small).
//! - **Lane finish**: a builder given a [`LaneView`] holds no
//!   postings of its own; the commit's frozen lanes are merged over
//!   this superfile's rows, term by term, through the spilled
//!   finish's streaming dictionary.
//! - **Spilled finish**: at least one column has spilled. The
//!   spilled column's `id_to_term` builds a lex-rank lookup
//!   (`term_id → rank in lex order`, one `Vec<u32>` per column,
//!   bounded by vocab so small even at 10M docs). Partition files
//!   are read as fixed-size triples, sorted by
//!   `(lex_rank[term_id], doc_id)` (pdqsort over `[(u32, u32,
//!   u32)]` — pure u32 compares, no `&[u8]` chasing), then
//!   k-way-merged into global lex order. The term dictionary is
//!   streamed by a `TermBlockWriter` to a scratch file, using
//!   `id_to_term[term_id]` to recover the term bytes per emission.
//!   Final blob assembly is `header → dictionary scratch → posting
//!   scratch → positions → doc-id map → doc-lengths`, all streamed
//!   through `W`.
//!
//! Mirror of vector: vector spills its input corpus as raw f32
//! bytes past 256 MiB and streams its centroid+code layout to
//! scratch; FTS spills its posting accumulator as fixed 12-byte
//! triples past 256 MiB and streams its term dictionary + posting
//! region to scratch. Both bound peak resident memory by a formula that does
//! not include `n_docs`, and both use fixed-size, no-framing record
//! formats so the spill IO is allocator-free on the read side.
//!
//! ## Builder lifecycle
//!
//! 1. `FtsBuilder::new()` — empty builder.
//! 2. `register_column(name, false)` per FTS column, in declaration order.
//! 3. `add_doc(column_id, local_doc_id, text)` per `(doc, column)` pair.
//!    Caller passes monotonically-increasing `local_doc_id`s.
//! 4. `finish()` (returns `Vec<u8>`) or `finish_to(impl Write)`
//!    (streams the blob progressively to any sink).
//!
//! A supertable commit fed as its rows arrive runs several builders in
//! lane mode instead (`set_lane_mode`, then `add_doc`, then `freeze`),
//! merges the frozen lanes once (`freeze_lanes`), and gives each
//! superfile's builder its rows of them (`set_lane_view`) before step 4.

use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
    fs,
    fs::File,
    io::{self, BufReader, BufWriter, Error, ErrorKind, Read, Seek, SeekFrom, Write},
    mem,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
    vec::IntoIter,
};

use bumpalo::Bump;
use memmap2::Mmap;
use rayon::ThreadPool;
use tempfile::{Builder, TempDir};
use tracing::debug;

use super::term_table::TermTable;

use crate::{
    config::scratch_root,
    superfile::{
        BuildError,
        bits::PackScratch,
        format::{
            self, KEY_SEPARATOR,
            checksum::{crc32c, crc32c_append},
            fts::{coarse_slot, skip_entry},
        },
        fts::{
            analysis::ChainTokenizer,
            bm25,
            positions::{TermRuns, encode_group, encode_run, skip_run},
            posting::{BLOCK_LEN, Block, EncodedBlock, block_lanes, encode_block_with},
            reader::{ColumnLengthStats, TopKEntry, finish_head},
            short::{SHORT_MAX_DF, encode_short},
            sorted_merge::{SortedInput, merge_column},
            tokenize::{StandardTokenizer, Tokenizer},
        },
        id_space::FtsDocId,
    },
    utils::{
        join::run_spawned,
        terms::{DictEntry, INLINE_TF_MAX, TermBlockWriter, TermDictBuilder, validate_column_name},
        trace::{detail_span, record},
        varint::{push_head, push_varint},
    },
};

/// Term bytes interned for a column, borrowed from its arena.
///
/// Entries of a [`Vocab`]'s `id_to_term` are `&'static str` slices that
/// **actually borrow from the per-column `term_arena: bumpalo::Bump`**
/// stored alongside them. The `'static` lifetime tag is a lie maintained
/// by `unsafe` `std::mem::transmute` at insertion (see
/// [`intern_term_id`]) and kept sound by two structural invariants:
///
/// 1. The arena outlives the list. [`Vocab`] declares `term_arena` last
///    so Rust drops it last (struct fields drop in declaration order).
///    The finish blocks that destructure a `Vocab` likewise add an
///    explicit `drop(term_arena)` after the last use of `id_to_term`.
/// 2. Dropping the list never dereferences an entry: `&str` has no
///    `Drop` impl, so the arena can outlast the list's body without the
///    list needing its bytes during teardown.
///
/// All of a column's term bytes sit densely in one arena, with one
/// bump-pointer copy per new term and no heap allocation of its own.
type ArenaStr = &'static str;

/// Chain terminator for the per-doc position chains
/// (`FtsBuilder::doc_pos_chain`): no previous occurrence.
const CHAIN_END: u32 = u32::MAX;

/// A lane records where a term's position runs end once every this many
/// postings (see [`FtsBuilder::set_lane_mode`]). Cutting a list at an
/// arbitrary posting then skips at most this many runs from the nearest
/// recorded end, for a table a quarter the size of one entry per posting.
const RUN_END_STRIDE: usize = 4;

#[derive(Default)]
struct FinishProfile {
    enabled: bool,
    encode_calls: u64,
    encode_df1: u64,
    encode_short: u64,
    encode_pfor: u64,
    /// Time spent encoding and writing terms. The merge encodes on many
    /// threads and sums their times, so there it can exceed wall time.
    encode_total: Duration,
    encode_block_build: Duration,
    encode_meta_write: Duration,
    encode_skip_write: Duration,
    encode_block_write: Duration,
    dict_insert: Duration,
    // Per-column phase totals (summed across columns; printed in the
    // [fts-finish] summary line at the end of finish_to).
    partition_flush: Duration,
    lex_rank_build: Duration,
    partition_sort: Duration,
    mmap_open: Duration,
    scratch_cleanup: Duration,
    // Whole-finish phase totals (printed once at end of finish_to).
    dict_close: Duration,
    postings_close: Duration,
    doc_lengths_emit: Duration,
    blob_copy: Duration,
}

impl FinishProfile {
    /// Add a worker thread's encode counts and times.
    fn absorb(&mut self, other: &FinishProfile) {
        self.encode_calls += other.encode_calls;
        self.encode_df1 += other.encode_df1;
        self.encode_short += other.encode_short;
        self.encode_pfor += other.encode_pfor;
        self.encode_total += other.encode_total;
        self.encode_block_build += other.encode_block_build;
        self.encode_meta_write += other.encode_meta_write;
        self.encode_skip_write += other.encode_skip_write;
        self.encode_block_write += other.encode_block_write;
    }

    fn from_config() -> Self {
        Self {
            enabled: crate::config::global().diagnostics.fts_profile,
            ..Self::default()
        }
    }
}

/// Per-(column, term) metadata header — 20 bytes, written immediately
/// before the term's skip table + posting blocks in the postings region.
/// `term_metadata_offset` (referenced from the dictionary entry) points at the
/// start of this struct.
///
/// Layout:
///   off  0 ..  4 : df (u32) — bounded by n_docs per superfile
///   off  4 .. 12 : postings_offset (u64) — equals the term's metadata_offset;
///                  self-describing. u64 supports superfiles past 4 GiB
///                  (e.g. the 16 GB target).
///   off 12 .. 16 : postings_length (u32) — this term's byte length, as
///                  the dictionary entry also records it.
///   off 16 .. 20 : num_blocks (u32)
///
/// `df`, `postings_length`, and `num_blocks` stay u32; only the absolute
/// offset into the postings region needs the full u64 range.
pub(crate) const TERM_META_SIZE: usize = 20;

/// Extended per-term metadata header for terms of a **positional**
/// column: the 20-byte layout plus `positions_offset` (u64, absolute
/// within the positions region) and `positions_length` (u32). The
/// column's positions flag selects the stride — the two layouts never
/// mix within one column.
pub(crate) const TERM_META_POSITIONAL_SIZE: usize = 32;

/// Where a long term's header holds its own offset in the postings
/// region. Filled in when the term is written, since only the writer knows
/// where it lands.
const HEADER_POSTINGS_OFFSET: Range<usize> = 4..12;

/// Where a positional long term's header holds its offset in the
/// positions region; filled in when the term is written, like
/// [`HEADER_POSTINGS_OFFSET`].
const HEADER_POSITIONS_OFFSET: Range<usize> = 20..28;

/// Largest document frequency at which a term's blocks may take the
/// patched encoding. A patched block decodes about 15 ns slower than a
/// plain one, and a common term's blocks are what every intersection
/// and union over it expands in bulk; a term below this bar is walked
/// in full only when it is itself a query term, a few dozen blocks at
/// most. On a Zipfian corpus the bands below the bar hold ~60% of what
/// the patched form saves, the bands above it the rest.
const PATCHED_MAX_DF: usize = 16 * 1024;

/// A term in at least this fraction of the column's scored documents is
/// written in blocks of two sub-blocks ([`BLOCK_LEN`] docs); a rarer one
/// in blocks of one, as every file was before blocks could be wider. A
/// wider block halves the per-block work a reader pays per document —
/// header, skip entry, position group, tf array — and coarsens the block
/// maxima a ranked walk prunes by. The common terms are the ones a walk
/// meets in every block it visits, so they carry the gain; a rare term
/// kept narrow keeps its pruning at the finer grain and spares a short
/// list the second sub-block's padding. A density rather than a count so
/// the rule carries across corpora; the measurements behind the value
/// are recorded with the benches.
const WIDE_BLOCK_MIN_DENSITY: f64 = 0.001;

/// The block length a term of `df` postings gets in a column of `n_docs`
/// scored documents (see [`WIDE_BLOCK_MIN_DENSITY`]).
fn block_len_for(df: usize, n_docs: usize) -> usize {
    let density = df as f64 / n_docs.max(1) as f64;
    block_lanes(density >= WIDE_BLOCK_MIN_DENSITY)
}

/// Smallest document frequency whose dense blocks are stored as bitsets
/// past the size-only break-even, by [`DENSE_BITSET_BIAS`].
const DENSE_BITSET_MIN_DF: usize = 256 * 1024;

/// How much larger than its packed deltas a dense term's block may be and
/// still be stored as a presence bitset.
const DENSE_BITSET_BIAS: f32 = 2.0;

/// The bitset bias a term with `df` documents is written with.
fn dense_bitset_bias(df: usize) -> f32 {
    match df >= DENSE_BITSET_MIN_DF {
        true => DENSE_BITSET_BIAS,
        false => 1.0,
    }
}

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

/// Doc-lengths directory entry for [`format::fts::VERSION_V9`], which widens
/// the entry to carry what a reader used to derive by walking the column:
///
/// ```text
///   off  0 ..  4 : column_id (u32)
///   off  4 .. 12 : doc_lengths_offset (u64)
///   off 12 .. 16 : avgdl_x1000 (u32)
///   off 16 .. 24 : total_tokens (u64)   — exact sum of the column's lengths
///   off 24 .. 32 : n_scored_docs (u64)  — documents carrying a token here
///   off 32 .. 33 : occupied_lo (u8)     — lowest bucket any document uses
///   off 33 .. 34 : occupied_hi (u8)     — highest, for the bound scale
/// ```
///
/// Without these a reader still has to touch every document to learn the
/// column's totals, which is the walk the stored buckets exist to remove.
pub(crate) const DOC_LENGTHS_ENTRY_SIZE_V9: usize = 34;

/// The doc-lengths directory entry width `version` uses.
pub(crate) const fn doc_lengths_entry_size(version: u32) -> usize {
    match format::fts::has_stored_buckets(version) {
        true => DOC_LENGTHS_ENTRY_SIZE_V9,
        false => DOC_LENGTHS_ENTRY_SIZE,
    }
}

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
/// - `~24 B`: per new term: the interner entry + the `Vec<(u32,u32)>`
///   header + small-alloc rounding.
/// - `+ term.len()`: term bytes.
/// - `+ 8 B`: per added posting: `(doc_id: u32, tf: u32)`.
const ACCUM_NEW_TERM_FIXED_BYTES: usize = 24;
const ACCUM_POSTING_BYTES: usize = 8;

/// Partition size (in triples) below which the lex-rank sort uses
/// `sort_unstable_by` instead of the counting/radix variant: under
/// this count the histogram allocation outweighs the algorithmic
/// savings.
pub(crate) const RADIX_SORT_MIN_TRIPLES: usize = 256;

/// Upper bound on the initial in-RAM chunk capacity (in triples)
/// during external merge sort. Caps the up-front `Vec` reservation
/// (~12 MiB of triples) when `max_partition_bytes` is large.
const EXTERNAL_MERGE_CHUNK_CAP_TRIPLES: usize = 1024 * 1024;

/// Number of triples buffered before each `write_all` when streaming
/// a sorted partition to disk. Amortizes the syscall cost (~48 KiB
/// per flush).
const SORT_OUTPUT_BATCH_TRIPLES: usize = 4096;

/// The version stamped on a finished blob, which is always
/// [`format::fts::VERSION_CURRENT`].
///
/// Reordered or not, a blob gets the same version: `V9` records the doc-id
/// map's length in the eight bytes before the doc-lengths directory, so one
/// version describes a blob with a map and one without. Older blobs encoded
/// that distinction in the version itself and are still read, never written.
///
/// A column's BM25 parameters do not move the version: they are recorded
/// in its `inf.fts.columns` entry and read back from there, so the stored
/// per-block bound is interpreted against the pair that entry names.
const fn blob_version() -> u32 {
    format::fts::VERSION_CURRENT
}

/// The per-document length as the blob stores it, saturated to the
/// stored width — what both the block-max bound and the reader's bucket
/// are computed from.
fn stored_doc_length(len: u32) -> u32 {
    len.min(format::fts::DOC_LENGTH_STORED_MAX)
}

struct ColumnState {
    name: String,
    /// One u32 per doc (token count for this column), push order
    /// matches local_doc_id order.
    doc_lengths: Vec<u32>,
    /// Total token count across every doc in this column. Used for
    /// `avgdl = total_tokens / documents with tokens`.
    total_tokens: u64,
    /// Each doc's token count before storage clamps it, so a stretch of
    /// a lane's documents can total its tokens exactly. Kept only by a
    /// lane (see [`FtsBuilder::set_lane_mode`]); empty otherwise.
    tokens: Vec<u32>,
    /// Record token positions for this column (phrase support). Set
    /// at registration from `FtsConfig::positions`; selects the
    /// positional capture path in `add_doc` and the extended
    /// per-term layout at emit.
    positions: bool,
    /// BM25 parameters for this column, from `FtsConfig::bm25`. The
    /// per-block score bounds in the skip table are the block's true
    /// max under this pair, and the pair itself is recorded in the
    /// column's KV entry so the reader knows what the bounds mean.
    params: bm25::Bm25Params,
    /// Length statistics of the table this column's superfile joins —
    /// every superfile already committed — so the average it bakes and
    /// scores at is the table-wide one as of this commit, not its own.
    /// Zero for a standalone build, which then averages over itself.
    corpus: ColumnLengthStats,
}

impl ColumnState {
    /// The average document length this column is written at: the
    /// table's token total over the documents that carry tokens, this
    /// superfile's `own` included, rounded to the fixed point the file
    /// stores so the bounds baked from it are exact at the value a
    /// reader recovers. A null or empty cell is indexed as a zero-length
    /// document so the per-doc arrays stay aligned with the row, but it
    /// is not a document this column has, and dividing by it would
    /// deflate the average by the column's fill rate.
    fn stored_average(&self, own: &ColumnLengthStats) -> f32 {
        let mut table = self.corpus;
        table.merge_with(own);
        bm25::stored_avgdl(table.avgdl())
    }

    /// This column's own statistics: token total over, and count of,
    /// the documents that carry tokens.
    fn length_stats(&self) -> ColumnLengthStats {
        ColumnLengthStats::from_lengths(self.doc_lengths.iter().copied())
    }
}

/// Per-column vocabulary: the term interner and the per-document scratch
/// `add_doc` dedups a document's tokens through. Both accumulators share
/// it, so the transition from the in-RAM store to the spilled one keeps
/// every id and rewrites no term.
struct Vocab {
    /// Term → dense id. Long terms are confirmed against `id_to_term`.
    term_to_id: TermTable,
    /// Id → term bytes, for the dictionary emit and the lex-rank table.
    /// Entries borrow from `term_arena` (see [`ArenaStr`]).
    id_to_term: Vec<ArenaStr>,
    /// Per-doc tf accumulator indexed by id. Dense ids make this one
    /// array store per token where a hash probe would cost several
    /// times more. `dense_doc_tf[id] != 0` iff `id` is in
    /// `updated_terms`; the per-doc drain restores the zero.
    dense_doc_tf: Vec<u32>,
    /// Per-id position state for the current document, indexed by id;
    /// empty for a positionless column. While the store is in RAM it is
    /// the term's last position in this document (read only while the
    /// term's tf is non-zero); once spilled it is the term's chain head
    /// (`CHAIN_END` = no occurrence in this doc).
    dense_doc_poshead: Vec<u32>,
    /// Ids the current document touched; drained at the end of each
    /// `add_doc`. Bounded by the doc's distinct terms, reused across docs.
    updated_terms: Vec<u32>,
    /// Owns the term bytes behind `term_to_id` and `id_to_term`.
    /// **Declared last so it drops last**: the table and the vec free
    /// their memory without dereferencing a term, and only then does the
    /// arena release the bytes (see [`ArenaStr`]).
    term_arena: Bump,
}

impl Vocab {
    fn new() -> Self {
        Self {
            term_to_id: TermTable::new(),
            id_to_term: Vec::new(),
            dense_doc_tf: Vec::new(),
            dense_doc_poshead: Vec::new(),
            updated_terms: Vec::new(),
            term_arena: Bump::new(),
        }
    }

    /// Distinct terms interned so far.
    fn len(&self) -> usize {
        self.id_to_term.len()
    }

    /// Intern `term`, growing the per-id arrays when it is new.
    #[inline(always)]
    fn intern(&mut self, term: &str, positional: bool) -> u32 {
        let (term_id, is_new) = intern_term_id(
            &mut self.term_to_id,
            &mut self.id_to_term,
            &self.term_arena,
            term,
        );
        if is_new {
            self.dense_doc_tf.push(0);
            if positional {
                self.dense_doc_poshead.push(CHAIN_END);
            }
        }
        term_id
    }
}

/// The in-RAM accumulator: every id's postings in arrival order (ascending
/// doc id, by the `add_doc` contract) and, for a positional column, its
/// concatenated position runs — one delta-varint run per `(doc, tf)`
/// pair, delimited by the pair's `tf`. Indexed by term id, so a posting
/// lands with one array index and one push; the finish walks the ids in
/// dictionary order and encodes each list in place, with no sort pass.
struct RamStore {
    /// One entry per id. A term's lists sit side by side, so a document's
    /// visit to a term — the run during the scan, the posting and the run
    /// end at the drain — reads one place rather than one table per list.
    lists: Vec<TermLists>,
    /// Estimated bytes held (see `ACCUM_*_BYTES`); drives the spill check.
    bytes: usize,
}

/// One id's accumulated lists in the in-RAM store.
#[derive(Default)]
struct TermLists {
    postings: Vec<(u32, u32)>,
    /// Empty for a positionless column.
    runs: Vec<u8>,
    /// Where the list's position runs end in `runs` after every
    /// [`RUN_END_STRIDE`] postings, so a stretch of the list's postings
    /// can take its runs by byte range. Kept only by a lane (see
    /// [`FtsBuilder::set_lane_mode`]); empty otherwise.
    run_ends: Vec<u32>,
}

impl RamStore {
    fn new() -> Self {
        Self {
            lists: Vec::new(),
            bytes: 0,
        }
    }

    /// Give every interned id its slot. Amortized O(1) per new id.
    #[inline(always)]
    fn grow_to(&mut self, n_terms: usize) {
        if self.lists.len() < n_terms {
            self.lists.resize_with(n_terms, TermLists::default);
        }
    }
}

/// Where a positional scan puts each token's position. The in-RAM store
/// takes the position as it arrives, gap-encoded straight onto the term's
/// run: the first occurrence of a term in a document opens its run with
/// the absolute position and later ones append the gap from the last
/// (`dense_doc_poshead` holds that last position while the store is in
/// RAM). A spilled store links the occurrence into the per-document chain
/// instead, since its run has to be written whole to the partition's blob
/// at the end of the document.
enum PositionSink<'a> {
    Runs {
        lists: &'a mut Vec<TermLists>,
        /// Run bytes appended during this document, for the accumulator's
        /// byte estimate.
        bytes: usize,
    },
    Chain(&'a mut Vec<(u32, u32)>),
}

/// Where a column's postings live: in RAM until the column's estimated
/// bytes cross the builder's `spill_threshold_bytes`, then in hash-
/// partitioned spill files for the rest of the build. The records in
/// the files are fixed-size `(term_id, doc_id, tf[, pos_lo, pos_hi])`
/// lanes — same shape as the vector builder's raw-f32 spill, no framing.
enum PostingStore {
    Ram(RamStore),
    Spilled(SpillStore),
}

/// Per-column posting accumulator: the vocabulary plus the store.
struct ColumnPostings {
    vocab: Vocab,
    store: PostingStore,
}

impl ColumnPostings {
    fn new() -> Self {
        Self {
            vocab: Vocab::new(),
            store: PostingStore::Ram(RamStore::new()),
        }
    }
    fn is_spilled(&self) -> bool {
        matches!(self.store, PostingStore::Spilled(_))
    }
    fn is_empty(&self) -> bool {
        self.vocab.id_to_term.is_empty()
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

struct SpillPartition<const N: usize> {
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
    batch: Vec<[u32; N]>,
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
/// Sortable + heap-mergeable posting triple. Matches the on-disk
/// layout (`[term_id_le, doc_id_le, tf_le]`) exactly so a partition
/// file's bytes can be reinterpreted as `&[Triple]` without copying
/// on little-endian hosts.
/// Lane count of the plain (positionless) spill record.
const PLAIN_RECORD_LANES: usize = 3;

/// Lane count of the positional spill record:
/// `[term_id, doc_id, tf, pos_off_lo, pos_off_hi]`. The last two
/// lanes carry the little-endian halves of the u64 byte offset of
/// this posting's position run in the partition's positions blob.
const POSITIONAL_RECORD_LANES: usize = 5;

/// Pack a positions-blob byte offset into the positional record's
/// two trailing lanes.
#[inline(always)]
fn pos_off_lanes(off: u64) -> (u32, u32) {
    (off as u32, (off >> 32) as u32)
}

/// Recover the positions-blob byte offset from a positional record.
#[inline(always)]
fn record_pos_off(rec: &[u32; POSITIONAL_RECORD_LANES]) -> u64 {
    (rec[3] as u64) | ((rec[4] as u64) << 32)
}

/// Per-partition positions blob for a positional column's spill: the
/// delta-varint runs of every posting routed to this partition, in
/// arrival order, addressed by the byte offsets riding in the
/// partition's 5-lane records. Written during `add_doc` (and the
/// in-RAM→spill transition), mmap'd read-only at merge time.
struct PartitionPositions {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    /// Bytes written so far — the next run's offset.
    len: u64,
}

/// A spilled column's partition set. A positionless column spills
/// plain 3-lane records; a positional column spills 5-lane records
/// plus one positions blob per partition (parallel vectors).
enum SpillStore {
    Plain(Vec<SpillPartition<PLAIN_RECORD_LANES>>),
    Positional {
        partitions: Vec<SpillPartition<POSITIONAL_RECORD_LANES>>,
        blobs: Vec<PartitionPositions>,
    },
}

impl SpillStore {
    fn n_partitions(&self) -> usize {
        match self {
            SpillStore::Plain(p) => p.len(),
            SpillStore::Positional { partitions, .. } => partitions.len(),
        }
    }
}

#[inline(always)]
fn triple_term_id<const N: usize>(t: &[u32; N]) -> u32 {
    t[0]
}
#[inline(always)]
fn triple_doc_id<const N: usize>(t: &[u32; N]) -> u32 {
    t[1]
}
#[inline(always)]
fn triple_tf<const N: usize>(t: &[u32; N]) -> u32 {
    t[2]
}

/// Write one triple as little-endian bytes via a single 12-byte
/// `write_all`. Replaces the old four-call
/// `write_posting_record` — function-call overhead is ~4× lower
/// on `BufWriter` and the syscall amortisation through the 64-KiB
/// buffer is unchanged. Reserved for callers without a per-
/// partition batch buffer (the spill transition streams through the
/// buffered `push_record_batched` path instead — see below).
/// On big-endian hosts, the bulk `bytemuck::cast_slice` write
/// path in `write_triples_sorted` is replaced with a per-triple
/// scalar write to preserve the little-endian on-disk format.
/// On little-endian hosts (x86_64, aarch64) the bulk path is
/// the only one compiled, so this function isn't built at all.
#[cfg(not(target_endian = "little"))]
#[inline(always)]
fn write_record<W: Write, const N: usize>(w: &mut W, rec: &[u32; N]) -> Result<(), BuildError> {
    let mut buf = [0u8; MAX_RECORD_BYTES];
    for (lane, v) in rec.iter().enumerate() {
        buf[lane * 4..lane * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    w.write_all(&buf[..mem::size_of::<[u32; N]>()])?;
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
fn push_record_batched<const N: usize>(
    partition: &mut SpillPartition<N>,
    rec: [u32; N],
) -> Result<(), BuildError> {
    partition.batch.push(rec);
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
fn flush_partition_batch<const N: usize>(
    partition: &mut SpillPartition<N>,
) -> Result<(), BuildError> {
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
        writer.write_all(bytemuck::cast_slice::<[u32; N], u8>(&partition.batch))?;
    }
    #[cfg(not(target_endian = "little"))]
    {
        for t in &partition.batch {
            write_record(writer, t)?;
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
fn read_partition_records<const N: usize>(path: &Path) -> Result<Vec<[u32; N]>, BuildError> {
    let mut bytes = Vec::new();
    let mut f = File::open(path)?;
    f.read_to_end(&mut bytes)?;
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let rec_bytes = mem::size_of::<[u32; N]>();
    if bytes.len() % rec_bytes != 0 {
        return Err(BuildError::Io(Error::new(
            ErrorKind::InvalidData,
            format!(
                "spill partition {path:?} length {} not a multiple of {}",
                bytes.len(),
                rec_bytes
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
        let records: &[[u32; N]] = bytemuck::try_cast_slice(&bytes).map_err(|_| {
            BuildError::Io(Error::new(
                ErrorKind::InvalidData,
                "bytemuck: spill bytes failed alignment for the record slice",
            ))
        })?;
        Ok(records.to_vec())
    }
    #[cfg(not(target_endian = "little"))]
    {
        let n = bytes.len() / rec_bytes;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * rec_bytes;
            let mut rec = [0u32; N];
            for (lane, slot) in rec.iter_mut().enumerate() {
                let at = off + lane * 4;
                *slot = u32::from_le_bytes(
                    bytes[at..at + 4]
                        .try_into()
                        .expect("invariant: 4-byte record lane"),
                );
            }
            out.push(rec);
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

/// The error every call on a builder whose `add_doc` failed partway
/// through a document returns.
fn failed_builder() -> BuildError {
    BuildError::Io(Error::other(
        "the fts builder is unusable after a document failed to index",
    ))
}

/// The error a lane returns for the document that takes its postings
/// past its threshold. A lane never spills — its reservation is what
/// bounds it — so the feed drops the lanes and the commit builds from the
/// buffer, where a column that outgrows the reservation spills.
fn lane_outgrown() -> BuildError {
    BuildError::Io(Error::other(
        "an fts lane outgrew its reservation; the commit builds from the buffer",
    ))
}

/// Intern `term` in a column's vocabulary and return its stable
/// `term_id`.
///
/// This is the hot `add_doc` lookup: one probe of the [`TermTable`],
/// which decides a hit from the slot alone for any term of up to 16
/// bytes. A miss copies the term bytes into the per-column `term_arena`
/// (one bump-pointer advance and a copy, no heap allocation) and
/// records the slice in `id_to_term` under the new id.
///
/// Returns `(term_id, is_new)`. `is_new == true` iff this call minted
/// a brand-new id (`term_to_id` previously had no entry for `term`).
/// Callers that keep a per-id parallel array (the [`Vocab`]'s
/// `dense_doc_tf` and `dense_doc_poshead`, and the in-RAM store's
/// per-id lists) use this to gate the grow check off the hot path:
/// an array only needs to be extended when a fresh id is minted,
/// which is once per novel term per column — not once per token.
///
/// SAFETY (invariants enforced by callers and the [`Vocab`] struct
/// layout, see [`ArenaStr`]): the `'static` lifetime tag on the
/// entries of `id_to_term` is a lie — the real lifetime is that of
/// `arena`. The transmute below is sound iff (a) `arena` outlives
/// `id_to_term` and the table that reads it, and (b) no entry is
/// dereferenced after `arena`'s `Drop` runs. Both hold by construction:
/// `arena` is declared last in [`Vocab`] so it drops last, and dropping
/// a `Vec<&'static str>` never dereferences its entries.
#[inline(always)]
fn intern_term_id(
    term_to_id: &mut TermTable,
    id_to_term: &mut Vec<ArenaStr>,
    arena: &Bump,
    term: &str,
) -> (u32, bool) {
    let (id, is_new) = term_to_id.intern(term.as_bytes(), id_to_term);
    if is_new {
        debug_assert_eq!(id as usize, id_to_term.len());
        // Bump-arena copy: one pointer advance + memcpy of the term
        // bytes, returning a `&str` whose lifetime is tied to `&arena`'s
        // borrow scope.
        let arena_str: &str = arena.alloc_str(term);
        // SAFETY: see function-level doc. The `'static` tag is a lie;
        // the real lifetime is `arena`'s, which [`Vocab`] guarantees
        // outlives `id_to_term` via field drop order.
        let static_str: ArenaStr = unsafe { std::mem::transmute(arena_str) };
        id_to_term.push(static_str);
    }
    (id, is_new)
}

/// Min-heap entry for the k-way merge over sorted partition chunks.
/// Sort key is a packed `u64 = (lex_rank as u64) << 32 | doc_id as
/// u64` — natural u64 ordering matches `(lex_rank, doc_id)` lex
/// order, so the heap comparator is a single u64 compare per pop.
/// Ordering is inverted (heap returns the *smallest* sort key
/// first) by implementing `Ord` reversed; `BinaryHeap` is a max-
/// heap.
struct MergeEntry<const N: usize> {
    /// `(lex_rank as u64) << 32 | doc_id as u64`.
    sort_key: u64,
    /// The full record — term_id/doc_id/tf lanes plus, on positional
    /// columns, the positions-blob offset lanes.
    rec: [u32; N],
    reader_idx: usize,
}

impl<const N: usize> PartialEq for MergeEntry<N> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<const N: usize> Eq for MergeEntry<N> {}
impl<const N: usize> PartialOrd for MergeEntry<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<const N: usize> Ord for MergeEntry<N> {
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
enum PartitionIter<const N: usize> {
    InMemory(IntoIter<[u32; N]>),
    Merge {
        readers: Vec<BufReader<File>>,
        heap: BinaryHeap<MergeEntry<N>>,
        /// Sorted-chunk files; kept alive so their inodes don't
        /// get reaped before iteration finishes.
        _chunk_paths: Vec<PathBuf>,
    },
}

impl<const N: usize> PartitionIter<N> {
    /// Pull the next sorted triple from this partition, looking up
    /// the sort key via `lex_rank` when refilling a merge cursor
    /// (so the heap stays minimal — only sort_key + tf + term_id
    /// + reader_idx).
    fn next_with(&mut self, lex_rank: &[u32]) -> Option<Result<[u32; N], BuildError>> {
        match self {
            PartitionIter::InMemory(it) => it.next().map(Ok),
            PartitionIter::Merge { readers, heap, .. } => {
                let MergeEntry {
                    rec, reader_idx, ..
                } = heap.pop()?;
                match read_one_record::<_, N>(&mut readers[reader_idx]) {
                    Ok(Some(next_t)) => {
                        let next_id = triple_term_id(&next_t);
                        let next_doc = triple_doc_id(&next_t);
                        let key = pack_sort_key(lex_rank[next_id as usize], next_doc);
                        heap.push(MergeEntry {
                            sort_key: key,
                            rec: next_t,
                            reader_idx,
                        });
                    }
                    Ok(None) => { /* chunk drained */ }
                    Err(e) => return Some(Err(e)),
                }
                Some(Ok(rec))
            }
        }
    }
}

/// Read a single 12-byte triple from a sorted-chunk file. Returns
/// `Ok(None)` on clean EOF.
fn read_one_record<R: Read, const N: usize>(r: &mut R) -> Result<Option<[u32; N]>, BuildError> {
    let mut buf = [0u8; MAX_RECORD_BYTES];
    let rec_bytes = mem::size_of::<[u32; N]>();
    match r.read_exact(&mut buf[..rec_bytes]) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(BuildError::Io(e)),
    }
    let mut rec = [0u32; N];
    for (lane, slot) in rec.iter_mut().enumerate() {
        let at = lane * 4;
        *slot = u32::from_le_bytes(buf[at..at + 4].try_into().expect("slice len 4"));
    }
    Ok(Some(rec))
}

/// Stack-buffer bound for [`read_one_record`]: the widest spill
/// record (the positional 5-lane form).
const MAX_RECORD_BYTES: usize = mem::size_of::<[u32; 5]>();

/// Write a slice of triples to a sorted-chunk file. Single
/// `write_all` per chunk on LE hosts via `bytemuck` byte-cast.
fn write_records_sorted<const N: usize>(
    triples: &[[u32; N]],
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
            write_record(&mut w, t)?;
        }
    }
    w.flush()?;
    Ok(())
}

fn spill_sorted_chunk<const N: usize>(
    chunk: &mut Vec<[u32; N]>,
    scratch_dir: &Path,
    partition_label: &str,
    chunk_idx: usize,
    lex_rank: &[u32],
    out_paths: &mut Vec<PathBuf>,
) -> Result<(), BuildError> {
    radix_sort_records_by_lex_rank(chunk, lex_rank);
    let path = scratch_dir.join(format!("{partition_label}_sorted{chunk_idx}.bin"));
    write_records_sorted(chunk, &path)?;
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
    use std::{
        cell::RefCell,
        path::{Path, PathBuf},
    };

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
/// 2. The secondary key is *almost* free. For the tokenizing ingest
///    feed, all triples for a fixed `term_id` are appended in
///    strictly increasing `doc_id` order (`add_doc` is called with
///    monotonic `local_doc_id` per `column_id`, and each call emits
///    one triple per unique term in iteration order over
///    `updated_terms`, with all triples for that doc emitted
///    contiguously), so the *stable* scatter on `lex_rank[term_id]`
///    already leaves the within-rank order as `doc_id` order. The
///    compaction carry paths may not arrive ascending — the
///    multi-cell merge remaps postings through the packed stable-id
///    row order — so a verification pass (pass 4 below) scans each
///    rank run and sorts only the runs with an inversion, restoring
///    the `(lex_rank, doc_id)` order the finish-time lex-order
///    partition traversal needs.
/// 3. Counting sort is **one pass to histogram + one pass to
///    scatter**. No `O(log n)` compare chain (pdqsort), no 5–8
///    LSB-byte passes (radix), no comparator chasing `lex_rank`
///    twice per call. Three reads of every triple and one write
///    (histogram, scatter, order verification) — the verification
///    read is sequential and measured within noise on the 1M
///    superfile build.
///
/// **Memory shape**: `counts: Vec<u32>` of length
/// `vocab_size + 1` (~40 KiB at 10K vocab), plus the `out: Vec
/// <Triple>` of the same length as `triples` — the same size as the
/// radix variant's parallel-array workspace, just a single
/// allocation instead of three.
///
/// Falls back to `sort_unstable_by` for tiny inputs where the counts
/// allocation outweighs the algorithmic savings.
fn radix_sort_records_by_lex_rank<const N: usize>(triples: &mut Vec<[u32; N]>, lex_rank: &[u32]) {
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
    // for the same rank lands immediately after. The scatter is
    // stable, so within a rank `out` holds arrival order — which is
    // `doc_id` order for the tokenizing ingest feed, but not
    // necessarily for the compaction carry feeds; pass 4 repairs
    // any run that needs it.
    let mut out: Vec<[u32; N]> = vec![[0u32; N]; n];
    for t in triples.iter() {
        let rank = unsafe { *lex_rank.get_unchecked(t[0] as usize) } as usize;
        let dst = unsafe { *offsets.get_unchecked(rank) } as usize;
        unsafe {
            *out.get_unchecked_mut(dst) = *t;
            *offsets.get_unchecked_mut(rank) = (dst as u32).wrapping_add(1);
        }
    }

    // Pass 4: repair within-rank doc order. The counting scatter above is
    // stable, so within a rank `out` holds arrival order. The tokenizing
    // ingest path always arrives in ascending doc order, but the
    // compaction carry paths can feed a term's docs out of order (the
    // multi-cell merge remaps postings through the packed row order), and
    // the sorted-chunk contract downstream — the external-merge heap key
    // and `encode_block`'s strictly-ascending doc ids — is `(lex_rank,
    // doc_id)`. Scan each rank run and sort only the runs that need it:
    // linear for the already-sorted ingest shape.
    let mut run_start = 0usize;
    let mut run_rank = lex_rank[triple_term_id(&out[0]) as usize];
    let mut run_sorted = true;
    for i in 1..=n {
        let rank_at = |t: &[u32; N]| lex_rank[triple_term_id(t) as usize];
        let boundary = i == n || rank_at(&out[i]) != run_rank;
        if boundary {
            if !run_sorted {
                out[run_start..i].sort_unstable_by_key(triple_doc_id);
            }
            if i < n {
                run_rank = rank_at(&out[i]);
            }
            run_start = i;
            run_sorted = true;
        } else if triple_doc_id(&out[i]) < triple_doc_id(&out[i - 1]) {
            run_sorted = false;
        }
    }

    *triples = out;
}

/// Open a partition as a sorted-triple iterator. Picks the in-
/// memory path when the on-disk partition is at or below
/// `max_partition_bytes` and the external-merge path when it
/// isn't.
fn open_partition_sorted<const N: usize>(
    partition_path: &Path,
    max_partition_bytes: u64,
    scratch_dir: &Path,
    partition_label: &str,
    lex_rank: &[u32],
) -> Result<PartitionIter<N>, BuildError> {
    let len = fs::metadata(partition_path)?.len();
    if len <= max_partition_bytes {
        let mut triples = read_partition_records::<N>(partition_path)?;
        radix_sort_records_by_lex_rank(&mut triples, lex_rank);
        return Ok(PartitionIter::InMemory(triples.into_iter()));
    }

    // External merge: stream the partition in
    // `max_partition_bytes`-sized triple chunks, sort each chunk
    // in RAM, write sorted-chunk spill, then k-way merge. The
    // resident peak during this path is one chunk's triples plus
    // one triple per chunk file in the heap.
    let chunk_triples = (max_partition_bytes as usize) / mem::size_of::<[u32; N]>();
    let mut sorted_chunk_paths: Vec<PathBuf> = Vec::new();
    let mut r = BufReader::with_capacity(PARTITION_BUF_SIZE, File::open(partition_path)?);
    let mut chunk: Vec<[u32; N]> =
        Vec::with_capacity(chunk_triples.min(EXTERNAL_MERGE_CHUNK_CAP_TRIPLES));
    let mut chunk_idx: usize = 0;
    while let Some(t) = read_one_record::<_, N>(&mut r)? {
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
    let mut heap: BinaryHeap<MergeEntry<N>> = BinaryHeap::with_capacity(readers.len());
    for (idx, reader) in readers.iter_mut().enumerate() {
        if let Some(t) = read_one_record::<_, N>(reader)? {
            let term_id = triple_term_id(&t);
            let doc_id = triple_doc_id(&t);
            heap.push(MergeEntry {
                sort_key: pack_sort_key(lex_rank[term_id as usize], doc_id),
                rec: t,
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
    /// Tokenizer for each registered column, indexed by column_id — the
    /// analyzer that column's text is tokenized with at index time.
    /// Grown in lockstep with `columns`.
    column_tokenizers: Vec<Arc<dyn Tokenizer>>,
    columns: Vec<ColumnState>,
    /// Per-column posting accumulator. Each entry's store starts in
    /// RAM and transitions to spill files exactly once, when its
    /// accumulated bytes cross `spill_threshold_bytes`.
    /// Mirror of `VectorBuilder`'s `pre_spill_buffer` + `spill`.
    postings: Vec<ColumnPostings>,
    /// Scratch directory that owns all posting + term dictionary spill files.
    /// Lazily populated — small builds (every column stays in RAM)
    /// never write here. Dropped after `finish_to` copies its
    /// contents into the output writer.
    scratch_dir: TempDir,
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
    /// Per-doc occurrence chain: `(position, prev_index)` per token,
    /// `CHAIN_END` terminating each term's chain. Cleared per doc;
    /// capacity persists.
    doc_pos_chain: Vec<(u32, u32)>,
    /// Scratch for one term's positions while draining a doc (chain
    /// walk emits them descending; reversed here before encoding).
    pos_scratch: Vec<u32>,
    /// Scratch for one (term, doc) encoded position run on the
    /// spilled positional drain path (the in-RAM path encodes
    /// straight into the accumulator's per-term buffer instead).
    run_scratch: Vec<u8>,
    /// The Parquet row each doc id stands for, when the caller fed
    /// documents in an order of its own rather than in row order.
    ///
    /// `Some` writes the map as its own region before the doc-lengths
    /// directory, with its length in the trailer just before the
    /// directory, and the reader translates hits through it. `None`, the
    /// default, writes no map and a zero trailer. The version is the same
    /// either way.
    pub(crate) doc_map: Option<Vec<u32>>,
    /// Prebuilt inputs whose postings the finish merges term by term
    /// instead of reading the accumulator (the compaction merge). Empty
    /// for every other build.
    sorted_inputs: Vec<SortedInput>,
    /// Where each written term's facts go — its document frequency, score
    /// bound, dictionary entry and best documents — when a caller wants
    /// them as the blob is written rather than read back from it.
    head_sink: Option<Box<dyn TermHeadSink>>,
    /// Best documents per term handed to the sink; `0` keeps none.
    head_depth: usize,
    /// Whether this builder is a lane of a build fed as its rows arrive
    /// (see [`Self::set_lane_mode`]): it then keeps what cutting its
    /// lists by row needs.
    lane_mode: bool,
    /// Lanes the finish merges instead of reading this builder's own
    /// accumulator, over a range of their rows; `None` for every other
    /// build. See [`Self::set_lane_view`].
    lane_view: Option<LaneView>,
    /// Set once an `add_doc` failed partway through a document. The
    /// store may then hold part of that document, so the builder refuses
    /// further documents and refuses to finish: a caller that retries on
    /// the same builder gets an error, not a blob with a torn document.
    failed: bool,
}

/// One column of a frozen lane: its terms, each id's postings with the
/// runs' end offsets, and its per-document lengths. The ids here are the
/// column's dictionary ranks: the freeze lays every table out in term
/// order, so a finish walking the lanes' terms in step reads each lane's
/// tables front to back instead of jumping about them in the order the
/// terms were first seen.
pub(crate) struct FrozenColumn {
    /// Every term's bytes, back to back, with each id's span.
    term_bytes: String,
    term_spans: Vec<(u32, u32)>,
    postings: Vec<Vec<(u32, u32)>>,
    run_ends: Vec<Vec<u32>>,
    pos_runs: Vec<Vec<u8>>,
    /// Exact token count per document; a document's stored length is
    /// derived from it.
    tokens: Vec<u32>,
    /// For a list that spans several pieces, where each piece's postings
    /// start in it: `cuts[cut_at[id] + p]..cuts[cut_at[id] + p + 1]` are
    /// piece `p`'s. `u32::MAX` for a list within one piece, whose whole
    /// extent the piece takes. Filled by [`freeze_lanes`].
    cut_at: Vec<u32>,
    cuts: Vec<u32>,
}

impl FrozenColumn {
    fn term(&self, id: u32) -> &str {
        let (start, len) = self.term_spans[id as usize];
        &self.term_bytes[start as usize..start as usize + len as usize]
    }

    /// How many terms the column holds.
    fn n_terms(&self) -> usize {
        self.term_spans.len()
    }

    /// The id of the first term not before `pivot`, in dictionary order.
    fn lower_bound(&self, pivot: &str) -> usize {
        let (mut lo, mut hi) = (0usize, self.n_terms());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.term(mid as u32) < pivot {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// A run of a lane's documents that are consecutive rows of the commit:
/// documents `doc..doc + len` are rows `row..row + len`. A lane's
/// documents of one batch are one such run, so a lane's whole row map is
/// a few hundred of these rather than an entry per document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stretch {
    pub(crate) doc: u32,
    pub(crate) row: u32,
    pub(crate) len: u32,
}

/// The first document of `stretches` at or past row `row`, as a document
/// index; the document count when none is. The stretches ascend in both
/// documents and rows.
fn docs_before_row(stretches: &[Stretch], row: u32) -> usize {
    let i = stretches.partition_point(|s| s.row + s.len <= row);
    match stretches.get(i) {
        None => stretches.last().map_or(0, |s| (s.doc + s.len) as usize),
        Some(s) if s.row >= row => s.doc as usize,
        Some(s) => (s.doc + (row - s.row)) as usize,
    }
}

/// The row document `doc` is, from `*cursor`, which is left at the stretch
/// holding it. The stretch at the cursor is tried first, so ascending
/// documents within one stretch cost a compare each; a document further
/// on is found by binary search over the stretches after it, so a term
/// seen in few batches does not walk every stretch in between — every
/// term's walk starts at the first stretch, and a lane has one stretch
/// per batch fed.
#[inline]
fn row_of(stretches: &[Stretch], cursor: &mut usize, doc: u32) -> u32 {
    let past = |s: &Stretch| doc >= s.doc + s.len;
    if past(&stretches[*cursor]) {
        *cursor += stretches[*cursor..].partition_point(past);
    }
    let s = &stretches[*cursor];
    s.row + (doc - s.doc)
}

/// A lane of a build fed as its rows arrived, frozen for the finish: the
/// accumulator's lists, now read-only, and the commit rows its documents
/// are, as stretches. Any stretch of the commit's rows can be cut out of
/// a lane by row, which is what lets a commit decide how many superfiles
/// to write once it knows how many rows it holds.
pub(crate) struct FrozenLane {
    columns: Vec<FrozenColumn>,
    shape: Vec<(String, bool)>,
    stretches: Vec<Stretch>,
}

/// One term of one lane, in the merged dictionary order of a column; the
/// entries of one term across the lanes are adjacent, the first flagged.
/// The first and last document of the lane's list ride along, so a
/// piece can pass over a list that misses its rows without a look at
/// the list itself.
#[derive(Clone, Copy)]
struct OrderedTerm {
    id: u32,
    lane: u8,
    first: bool,
    first_doc: u32,
    last_doc: u32,
}

/// A commit's frozen lanes with, per column, every lane's terms in one
/// merged dictionary order. The merge is done once for the commit; the
/// pieces then walk it with no term comparison of their own.
pub(crate) struct FrozenLanes {
    lanes: Vec<FrozenLane>,
    order: Vec<Vec<OrderedTerm>>,
    /// The rows of each superfile the commit writes, in order.
    pieces: Vec<Range<u32>>,
}

/// The most lanes a commit may be fed through: a merged order names each
/// entry's lane in a byte.
pub(crate) const MAX_LANES: usize = u8::MAX as usize;

/// Merge `lanes`' dictionaries, column by column, into the order the
/// pieces walk. The lanes must hold the same columns. The term space is
/// cut into as many ranges as the pool has threads, at terms taken from
/// the longest lane's dictionary, and the ranges merge side by side: a
/// term is one value in every lane, so no term straddles a cut.
///
/// For a caller outside `pool`: the cuts and the range merges go to the
/// pool as spawned tasks the caller waits for, so no pool thread waits for
/// another's work. Run under `install`, the merge's own waits ran other
/// commits' seals meanwhile, and a merge of two seconds' work took
/// seventeen.
pub(crate) fn freeze_lanes(
    pool: &ThreadPool,
    lanes: Vec<FrozenLane>,
    pieces: Vec<Range<u32>>,
) -> Result<FrozenLanes, BuildError> {
    fn lost() -> BuildError {
        BuildError::Io(Error::other("a lane merge task ended without a result"))
    }
    check_lanes(&lanes)?;
    let n_columns = lanes.first().map_or(0, |lane| lane.columns.len());
    let shared_pieces = Arc::new(pieces.clone());
    let lanes = run_spawned(
        pool,
        lanes,
        move |_, mut lane| {
            cut_lane(&mut lane, &shared_pieces);
            Ok(lane)
        },
        lost,
    )?;
    let lanes = Arc::new(lanes);
    let ranges = pool.current_num_threads().max(1);
    let mut order = Vec::with_capacity(n_columns);
    for column in 0..n_columns {
        let spans = merge_spans(&lanes, column, ranges);
        let shared = Arc::clone(&lanes);
        let merged = run_spawned(
            pool,
            spans,
            move |_, spans| Ok(merge_term_range(&shared, column, &spans)),
            lost,
        )?;
        order.push(merged.concat());
    }
    let lanes = Arc::try_unwrap(lanes)
        .map_err(|_| BuildError::Io(Error::other("a lane merge task still holds the lanes")))?;
    Ok(FrozenLanes {
        lanes,
        order,
        pieces,
    })
}

/// The lanes a commit may seal: no more than a merged order can name,
/// all holding the same columns.
fn check_lanes(lanes: &[FrozenLane]) -> Result<(), BuildError> {
    if lanes.len() > MAX_LANES {
        return Err(BuildError::Io(Error::other("too many fts lanes")));
    }
    if let Some(first) = lanes.first()
        && lanes.iter().any(|lane| lane.shape != first.shape)
    {
        return Err(BuildError::Io(Error::other(
            "fts lanes must hold the same columns",
        )));
    }
    Ok(())
}

/// Where each piece's postings start in every list of `lane` that spans
/// more than one piece: one pass over the list, in step with the pieces,
/// in place of a search per piece at the seal.
fn cut_lane(lane: &mut FrozenLane, pieces: &[Range<u32>]) {
    let bounds: Vec<usize> = pieces
        .iter()
        .map(|piece| docs_before_row(&lane.stretches, piece.start))
        .chain(
            pieces
                .last()
                .map(|last| docs_before_row(&lane.stretches, last.end)),
        )
        .collect();
    let piece_of = |doc: u32| bounds.partition_point(|&b| b <= doc as usize) - 1;
    for col in &mut lane.columns {
        col.cut_at = vec![u32::MAX; col.postings.len()];
        col.cuts.clear();
        for (id, list) in col.postings.iter().enumerate() {
            let (Some(&(first, _)), Some(&(last, _))) = (list.first(), list.last()) else {
                continue;
            };
            if piece_of(first) == piece_of(last) {
                continue;
            }
            col.cut_at[id] = col.cuts.len() as u32;
            let mut at = 0usize;
            for &bound in &bounds {
                while at < list.len() && (list[at].0 as usize) < bound {
                    at += 1;
                }
                col.cuts.push(at as u32);
            }
        }
    }
}

/// The term space of `column` cut into up to `ranges` ranges at terms
/// taken from the longest lane's dictionary — a term is one value in
/// every lane, so no term straddles a cut — as each lane's dictionary
/// span per range. Empty when no lane has the column.
fn merge_spans(lanes: &[FrozenLane], column: usize, ranges: usize) -> Vec<Vec<Range<usize>>> {
    let longest = lanes
        .iter()
        .map(|lane| &lane.columns[column])
        .max_by_key(|col| col.n_terms());
    let Some(longest) = longest else {
        return Vec::new();
    };
    // The terms the ranges start at, after the first.
    let pivots: Vec<&str> = (1..ranges)
        .map(|r| longest.n_terms() * r / ranges)
        .filter(|&at| at < longest.n_terms())
        .map(|at| longest.term(at as u32))
        .collect();
    // Each lane's dictionary index at every cut.
    let cuts: Vec<Vec<usize>> = lanes
        .iter()
        .map(|lane| {
            let col = &lane.columns[column];
            let mut cuts = vec![0usize];
            cuts.extend(pivots.iter().map(|pivot| col.lower_bound(pivot)));
            cuts.push(col.n_terms());
            cuts
        })
        .collect();
    (0..pivots.len() + 1)
        .map(|range| cuts.iter().map(|c| c[range]..c[range + 1]).collect())
        .collect()
}

/// Merge the dictionary entries `spans[lane]` of every lane's `column`
/// into one order, ties adjacent in lane order with the first flagged.
fn merge_term_range(
    lanes: &[FrozenLane],
    column: usize,
    spans: &[Range<usize>],
) -> Vec<OrderedTerm> {
    // A min-heap of each lane's next term; ties pop in lane order.
    let mut heap: BinaryHeap<Reverse<(&str, u8, u32)>> = BinaryHeap::new();
    let mut next: Vec<usize> = spans.iter().map(|span| span.start).collect();
    for (i, lane) in lanes.iter().enumerate() {
        let col = &lane.columns[column];
        if next[i] < spans[i].end {
            let id = next[i] as u32;
            heap.push(Reverse((col.term(id), i as u8, id)));
        }
    }
    let mut order = Vec::with_capacity(spans.iter().map(ExactSizeIterator::len).sum());
    let mut last: Option<&str> = None;
    while let Some(Reverse((term, lane, id))) = heap.pop() {
        let list = &lanes[lane as usize].columns[column].postings[id as usize];
        order.push(OrderedTerm {
            id,
            lane,
            first: last != Some(term),
            first_doc: list.first().map_or(u32::MAX, |&(d, _)| d),
            last_doc: list.last().map_or(0, |&(d, _)| d),
        });
        last = Some(term);
        let i = lane as usize;
        let col = &lanes[i].columns[column];
        next[i] += 1;
        if next[i] < spans[i].end {
            let id = next[i] as u32;
            heap.push(Reverse((col.term(id), lane, id)));
        }
    }
    order
}

/// What a build finishes from in place of its own accumulator: the
/// commit's lanes, restricted to piece `piece` of them — the superfile's
/// own rows, numbered from the piece's first row.
#[derive(Clone)]
pub(crate) struct LaneView {
    pub(crate) lanes: Arc<FrozenLanes>,
    pub(crate) piece: usize,
}

impl LaneView {
    /// The commit rows this view's superfile holds.
    fn rows(&self) -> &Range<u32> {
        &self.lanes.pieces[self.piece]
    }
}

/// Receives each term's facts as the blob is written: what a table-level
/// term index records per `(term, superfile)`, produced in the pass that
/// writes the postings instead of a second pass that reads them back.
/// Terms arrive in dictionary order within a column, columns in the
/// builder's order.
pub(crate) trait TermHeadSink: Send {
    fn term(
        &mut self,
        column: &str,
        term: &str,
        facts: &TermHeadFacts<'_>,
    ) -> Result<(), BuildError>;
    /// Every term has been handed over.
    fn finish(&mut self) -> Result<(), BuildError>;
}

/// One term's facts for a [`TermHeadSink`].
pub(crate) struct TermHeadFacts<'a> {
    /// Documents carrying the term in this blob.
    pub(crate) df: u64,
    /// The largest key — the length-normalized term frequency at this
    /// blob's average length and parameters, the factor a score multiplies
    /// the idf by — of a document of the term in this blob.
    pub(crate) bound: f32,
    /// The dictionary entry written for the term.
    pub(crate) entry: DictEntry,
    /// The best documents by impact as `(row, term frequency, length
    /// bucket)`, in descending impact then ascending blob document id, at
    /// most the sink's depth, encoded by [`push_head`].
    pub(crate) head: &'a [u8],
    /// The length-normalized term frequency — `tf / (tf + norm)` at this
    /// blob's average length and parameters, the factor a score multiplies
    /// the idf by — of the best document `head` left out; `0.0` when it
    /// holds them all.
    pub(crate) tail: f32,
    /// The average length this blob scores the column at, and so the
    /// average `tail` and the heads' keys are at.
    pub(crate) avgdl: f32,
    /// The column's parameters, likewise.
    pub(crate) params: bm25::Bm25Params,
}

/// A term's head as the encoder gathers it.
#[derive(Default, Clone)]
struct HeadOut {
    df: u64,
    /// The head's documents as `(row, term frequency, length bucket)`,
    /// encoded by [`push_head`] where the term is encoded, so the thread
    /// that writes terms in order only copies them.
    facts: Vec<u8>,
    bound: f32,
    tail: f32,
    /// The average length and parameters `tail` is keyed at.
    avgdl: f32,
    params: bm25::Bm25Params,
}

/// Keep a term's best `depth` documents by `scores` (one per posting, as
/// a reader scores them), the way a reader's single-term search ranks
/// them: `depth + 1` are kept so the first document left out is the tail,
/// exactly. The heap entry is the reader's own, so a tie keeps the smaller
/// document id as a reader's top-k does.
fn term_head(
    pairs: &[(u32, u32)],
    scores: &[f32],
    enc: &TermEncoding<'_>,
    heap: &mut BinaryHeap<TopKEntry<(u32, u32)>>,
    kept: &mut Vec<((u32, u32), f32)>,
    out: &mut HeadOut,
) {
    let depth = enc.head_depth;
    heap.clear();
    out.df = pairs.len() as u64;
    for (&(doc, tf), &score) in pairs.iter().zip(scores) {
        if heap.len() <= depth {
            heap.push(TopKEntry(score, (doc, tf)));
        } else if let Some(mut worst) = heap.peek_mut()
            && score > worst.0
        {
            *worst = TopKEntry(score, (doc, tf));
        }
    }
    // The key of a document: what the term index orders its tiers by.
    let key = |(doc, tf): (u32, u32)| {
        bm25::tf_factor(
            tf,
            enc.params
                .dl_norm_k1(bm25::stored_len(enc.doc_lengths[doc as usize]), enc.avgdl),
        )
    };
    (out.bound, out.tail) = finish_head(heap, depth, kept, key);
    out.facts.clear();
    push_head(
        &mut out.facts,
        kept.iter().map(|&((doc, tf), _)| {
            let row = enc.rows.map_or(doc, |rows| rows[doc as usize]);
            (row, tf, bm25::quantize_len(enc.doc_lengths[doc as usize]))
        }),
    );
    out.avgdl = enc.avgdl;
    out.params = enc.params;
}

impl Default for FtsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FtsBuilder {
    /// Construct a builder with the scratch directory at `storage.scratch_root`
    /// (defaults to the system temp dir when unset) and the
    /// default 256 MiB spill threshold. Mirror of `VectorBuilder::new`.
    ///
    /// Panics if creating the scratch tempdir fails — same policy
    /// as `VectorBuilder::new` for the same reason (no realistic
    /// recovery at construction time, preserves existing public
    /// API). Operators running large builds should prefer
    /// [`Self::with_scratch`] pointing at an instance-store NVMe
    /// partition.
    pub fn new() -> Self {
        let scratch_root = scratch_root();

        let scratch_dir = Builder::new()
            .prefix("infino-fts-")
            .tempdir_in(&scratch_root)
            .expect("create FtsBuilder scratch tempdir");

        Self::from_parts(scratch_dir)
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
    pub fn with_scratch(scratch: PathBuf) -> Result<Self, BuildError> {
        let scratch_dir = Builder::new().prefix("infino-fts-").tempdir_in(&scratch)?;
        Ok(Self::from_parts(scratch_dir))
    }

    fn from_parts(scratch_dir: TempDir) -> Self {
        Self {
            column_tokenizers: Vec::new(),
            columns: Vec::new(),
            postings: Vec::new(),
            scratch_dir,
            spill_threshold_bytes: DEFAULT_SPILL_THRESHOLD_BYTES,
            spill_partitions: DEFAULT_SPILL_PARTITIONS,
            max_partition_bytes: DEFAULT_MAX_PARTITION_BYTES,
            n_docs: 0,
            lane_mode: false,
            lane_view: None,
            failed: false,
            doc_pos_chain: Vec::new(),
            pos_scratch: Vec::new(),
            run_scratch: Vec::new(),
            doc_map: None,
            sorted_inputs: Vec::new(),
            head_sink: None,
            head_depth: 0,
        }
    }

    /// Have the finish merge these inputs' postings term by term instead of
    /// reading the accumulator, which must stay empty. Inputs are in output
    /// row order; their doc lengths are appended separately.
    /// Hand every written term's facts to `sink`, with its best `depth`
    /// documents by impact.
    pub(crate) fn set_term_head_sink(&mut self, sink: Box<dyn TermHeadSink>, depth: usize) {
        self.head_sink = Some(sink);
        self.head_depth = depth;
    }

    pub(crate) fn set_sorted_inputs(&mut self, inputs: Vec<SortedInput>) {
        self.sorted_inputs = inputs;
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
            return Err(BuildError::Io(Error::new(
                ErrorKind::InvalidInput,
                "FtsBuilder::set_spill_partitions must be called before any register_column",
            )));
        }
        if n == 0 {
            return Err(BuildError::Io(Error::new(
                ErrorKind::InvalidInput,
                "FtsBuilder: spill_partitions must be ≥ 1",
            )));
        }
        // Partition selection is `term_id & (n - 1)` on the hot
        // spill path; that's only correct (uniform partitioning)
        // when `n` is a power of two. Reject non-PO2 values so the
        // hot path stays branch-free instead of falling back to
        // modulo.
        if !n.is_power_of_two() {
            return Err(BuildError::Io(Error::new(
                ErrorKind::InvalidInput,
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

    /// Register an FTS column up-front with the standard analyzer and
    /// BM25 pair. Returns its `column_id` (its index in declaration
    /// order). Use [`FtsBuilder::register_column_with_tokenizer`] to
    /// declare a column's filters or BM25 pair.
    pub fn register_column(&mut self, name: String, positions: bool) -> Result<u32, BuildError> {
        self.register_column_with_tokenizer(
            name,
            positions,
            Arc::new(StandardTokenizer),
            bm25::Bm25Params::STANDARD,
        )
    }

    /// Register an FTS column tokenized with an explicit `tokenizer`.
    /// Returns its `column_id`. Lets each column carry its own analyzer
    /// (per-field analysis).
    pub fn register_column_with_tokenizer(
        &mut self,
        name: String,
        positions: bool,
        tokenizer: Arc<dyn Tokenizer>,
        params: bm25::Bm25Params,
    ) -> Result<u32, BuildError> {
        if !validate_column_name(&name) {
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
            tokens: Vec::new(),
            positions,
            params,
            corpus: ColumnLengthStats::default(),
        });
        self.postings.push(ColumnPostings::new());
        self.column_tokenizers.push(tokenizer);
        Ok(column_id)
    }

    /// Open spill partition files for a column and return them.
    /// Called the first time a column's in-RAM accumulator crosses
    /// `spill_threshold_bytes`.
    fn open_partitions_for_column<const N: usize>(
        scratch_dir: &Path,
        column_id: u32,
        n_partitions: usize,
    ) -> Result<Vec<SpillPartition<N>>, BuildError> {
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

    /// Open one positions blob per spill partition for a positional
    /// column. Sibling of
    /// [`open_partitions_for_column`](Self::open_partitions_for_column);
    /// the blob at index `p` carries the position runs of every record
    /// routed to partition `p`.
    fn open_position_blobs_for_column(
        scratch_dir: &Path,
        column_id: u32,
        n_partitions: usize,
    ) -> Result<Vec<PartitionPositions>, BuildError> {
        let mut blobs = Vec::with_capacity(n_partitions);
        for partition in 0..n_partitions {
            let path = scratch_dir.join(format!("fts_col{column_id}_part{partition}.pos.bin"));
            let file = File::create(&path)?;
            blobs.push(PartitionPositions {
                path,
                writer: Some(BufWriter::with_capacity(PARTITION_BUF_SIZE, file)),
                len: 0,
            });
        }
        Ok(blobs)
    }

    /// Move an in-RAM column's postings into fresh spill partitions,
    /// keyed by the ids the vocabulary already assigned, so the column
    /// can run the rest of the build in spill mode. Called once per
    /// column, at the moment its estimated bytes cross the threshold.
    /// A positional column's runs go to the partition's positions blob
    /// and each record carries its run's byte offset in the trailing
    /// lanes, exactly as the spilled `add_doc` writes them.
    fn spill_ram_store(
        scratch_dir: &Path,
        column_id: u32,
        n_partitions: usize,
        positional: bool,
        ram: RamStore,
    ) -> Result<SpillStore, BuildError> {
        debug_assert!(
            n_partitions.is_power_of_two(),
            "spill_partitions must be a power of 2; got {n_partitions}"
        );
        let mask = n_partitions - 1;
        let RamStore { lists, .. } = ram;
        if !positional {
            let mut partitions =
                Self::open_partitions_for_column(scratch_dir, column_id, n_partitions)?;
            for (term_id, TermLists { postings: list, .. }) in lists.into_iter().enumerate() {
                let p = term_id & mask;
                for (doc_id, tf) in list {
                    push_record_batched(&mut partitions[p], [term_id as u32, doc_id, tf])?;
                }
            }
            return Ok(SpillStore::Plain(partitions));
        }
        let mut partitions =
            Self::open_partitions_for_column(scratch_dir, column_id, n_partitions)?;
        let mut blobs = Self::open_position_blobs_for_column(scratch_dir, column_id, n_partitions)?;
        for (
            term_id,
            TermLists {
                postings: list,
                runs,
                ..
            },
        ) in lists.into_iter().enumerate()
        {
            let p = term_id & mask;
            let blob = &mut blobs[p];
            let writer = blob
                .writer
                .as_mut()
                .expect("positions blob writer is open before finish");
            let mut at: usize = 0;
            for (doc_id, tf) in list {
                let run_start = at;
                skip_run(&runs, &mut at, tf).expect("builder-encoded runs are well-formed");
                writer.write_all(&runs[run_start..at])?;
                let pos_off = blob.len;
                blob.len += (at - run_start) as u64;
                let (lo, hi) = pos_off_lanes(pos_off);
                push_record_batched(&mut partitions[p], [term_id as u32, doc_id, tf, lo, hi])?;
            }
            debug_assert_eq!(at, runs.len(), "runs must cover exactly the pairs");
        }
        Ok(SpillStore::Positional { partitions, blobs })
    }

    /// Index `text` for `(column_id, local_doc_id)`.
    ///
    /// Caller must call this once per (doc, registered FTS column) pair,
    /// with monotonically increasing `local_doc_id` per column. Multiple
    /// occurrences of the same term in `text` increment the term-frequency
    /// for that doc.
    /// Give a column the length statistics of the table its superfile
    /// joins, so `finish` bakes — and the file declares — the table-wide
    /// average document length rather than this superfile's own. Without
    /// it the column averages over itself, which is right for the first
    /// superfile of a table and for a standalone file.
    pub fn set_corpus_length_stats(&mut self, column_id: u32, corpus: ColumnLengthStats) {
        self.columns[column_id as usize].corpus = corpus;
    }

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
        if self.failed {
            return Err(failed_builder());
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

        // Downcast once per call so the tokenize_each_inline fast
        // path is reachable: through the trait object every token
        // costs an indirect call and the interning closure cannot
        // inline into the scan loop.
        let tokenizer = &self.column_tokenizers[col_idx];
        let standard_tok = tokenizer
            .as_ref()
            .as_any()
            .downcast_ref::<StandardTokenizer>();
        // A column with a stopword set or a stemmer tokenizes through
        // the chain, which wraps `standard`. It gets its own
        // monomorphized arm for the same reason, and it must be reached
        // through the *chain's* scan — the base's would index the
        // unfiltered tokens.
        let chain_tok = tokenizer.as_ref().as_any().downcast_ref::<ChainTokenizer>();
        let positional = self.columns[col_idx].positions;
        let lane_mode = self.lane_mode;
        let mut tokens_in_doc: u64 = 0;

        let ColumnPostings { vocab, store } = &mut self.postings[col_idx];
        let Vocab {
            term_to_id,
            id_to_term,
            dense_doc_tf,
            dense_doc_poshead,
            updated_terms,
            term_arena,
        } = vocab;
        updated_terms.clear();
        let mut pos_overflow = false;
        // Run bytes the in-RAM sink appended during the scan.
        let mut run_bytes: usize = 0;

        // Per-token cost: intern lookup + dense-array tf bump; the
        // positional scan also links each occurrence into the per-doc
        // chain. Both stores share this scan and differ only in where
        // the per-doc drain below puts the postings.
        if !positional {
            let mut on_token = |tok: &str| {
                tokens_in_doc += 1;
                let (term_id, is_new) = intern_term_id(term_to_id, id_to_term, term_arena, tok);
                let idx = term_id as usize;
                if is_new {
                    debug_assert_eq!(idx, dense_doc_tf.len());
                    dense_doc_tf.push(0);
                }
                // SAFETY: lockstep invariant between `dense_doc_tf`
                // and `id_to_term` (see [`Vocab`]) holds here:
                // `intern_term_id` returns `term_id < id_to_term.len()`
                // and we just `push(0)` on `is_new`, so `idx <
                // dense_doc_tf.len()`.
                let slot = unsafe { dense_doc_tf.get_unchecked_mut(idx) };
                if *slot == 0 {
                    updated_terms.push(term_id);
                }
                *slot += 1;
            };
            if let Some(standard) = standard_tok {
                standard.tokenize_each_inline(text, &mut on_token);
            } else if let Some(chain) = chain_tok {
                chain.tokenize_each_inline(text, &mut on_token);
            } else {
                tokenizer.tokenize_each(text, &mut on_token);
            }
        } else {
            self.doc_pos_chain.clear();
            let mut sink = match store {
                PostingStore::Ram(ram) => PositionSink::Runs {
                    lists: &mut ram.lists,
                    bytes: 0,
                },
                PostingStore::Spilled(_) => PositionSink::Chain(&mut self.doc_pos_chain),
            };
            // `record` interns the token, bumps its tf, and hands its
            // `position` to the sink. `position` is the gap-inclusive
            // ordinal — a token the tokenizer drops still advances it, so
            // a phrase never treats the tokens on either side of a dropped
            // word as adjacent. The doc length (`tokens_in_doc`) counts
            // only *emitted* tokens, bumped by the caller below, so it
            // stays identical to the positionless build; the two coincide
            // only when nothing is dropped.
            let mut record = |tok: &str, position: u64| {
                let (term_id, is_new) = intern_term_id(term_to_id, id_to_term, term_arena, tok);
                let idx = term_id as usize;
                if is_new {
                    debug_assert_eq!(idx, dense_doc_tf.len());
                    dense_doc_tf.push(0);
                    dense_doc_poshead.push(CHAIN_END);
                    if let PositionSink::Runs { lists, .. } = &mut sink {
                        lists.push(TermLists::default());
                    }
                }
                if position > u32::MAX as u64 {
                    // Positions are u32 ordinals; checked after the
                    // scan and surfaced as a hard build error.
                    pos_overflow = true;
                    return;
                }
                let position = position as u32;
                // SAFETY: same lockstep invariant as the positionless
                // closure; `dense_doc_poshead` (and the in-RAM store's
                // per-id vectors) are pushed in the same `is_new` branch
                // so they share the bound.
                let slot = unsafe { dense_doc_tf.get_unchecked_mut(idx) };
                let first = *slot == 0;
                if first {
                    updated_terms.push(term_id);
                }
                *slot += 1;
                let state = unsafe { dense_doc_poshead.get_unchecked_mut(idx) };
                match &mut sink {
                    PositionSink::Runs { lists, bytes } => {
                        let run = unsafe { &mut lists.get_unchecked_mut(idx).runs };
                        let before = run.len();
                        push_varint(run, if first { position } else { position - *state });
                        *bytes += run.len() - before;
                        *state = position;
                    }
                    PositionSink::Chain(doc_pos_chain) => {
                        let chain_idx = doc_pos_chain.len() as u32;
                        doc_pos_chain.push((position, *state));
                        *state = chain_idx;
                    }
                }
            };
            if let Some(standard) = standard_tok {
                // `standard` drops nothing — every segment carrying an
                // alphanumeric is emitted — so an emission ordinal *is*
                // the gap-inclusive position and no gap bookkeeping is
                // needed. Monomorphized for the same reason as above.
                standard.tokenize_each_inline(text, |tok| {
                    record(tok, tokens_in_doc);
                    tokens_in_doc += 1;
                });
            } else if let Some(chain) = chain_tok {
                // Gap-aware: a token the chain's stopword filter removes
                // advances the position ordinal but emits nothing, and
                // the doc length counts only what is emitted — the token
                // count Lucene's norms are built from too.
                chain.tokenize_each_inline_positioned(text, |tok, position| {
                    record(tok, position);
                    tokens_in_doc += 1;
                });
            } else {
                // A custom tokenizer reports its own gap-inclusive
                // positions through the trait; the default numbering is
                // consecutive, which is correct for one that drops
                // nothing (see the `Tokenizer` trait docs).
                tokenizer.tokenize_each_positioned(text, &mut |tok, position| {
                    record(tok, position);
                    tokens_in_doc += 1;
                });
            }
            run_bytes = match sink {
                PositionSink::Runs { bytes, .. } => bytes,
                PositionSink::Chain(_) => 0,
            };
        }
        if pos_overflow {
            // The document's counts and position heads are cleared, as the
            // drain below would have cleared them, and the builder is marked
            // unusable: the in-RAM store already holds part of this
            // document's position runs, which nothing can take back out.
            for &term_id in updated_terms.iter() {
                dense_doc_tf[term_id as usize] = 0;
                if positional {
                    dense_doc_poshead[term_id as usize] = CHAIN_END;
                }
            }
            updated_terms.clear();
            self.failed = true;
            return Err(BuildError::PositionOverflow {
                column: self.columns[col_idx].name.clone(),
            });
        }

        // `self.columns[col_idx]` is a disjoint field from
        // `self.postings[col_idx]`, so split-borrow legal.
        let col = &mut self.columns[col_idx];
        let tokens_clamped = u32::try_from(tokens_in_doc).unwrap_or(u32::MAX);
        col.doc_lengths.push(stored_doc_length(tokens_clamped));
        col.total_tokens = col.total_tokens.saturating_add(tokens_in_doc);
        if lane_mode {
            col.tokens.push(tokens_clamped);
        }
        let docs_now = local_doc_id.saturating_add(1);
        if docs_now > self.n_docs {
            self.n_docs = docs_now;
        }

        // Per-doc drain: one posting per touched term, then zero the
        // dense slot to restore the `dense_doc_tf[tid] != 0 iff in
        // updated_terms` invariant. A positional column that has spilled
        // additionally walks the term's chain (positions come out
        // descending — the chain heads at the LAST occurrence), reverses
        // it, and delta-varint encodes the run; in RAM the run was
        // written during the scan.
        //
        // SAFETY (every `get_unchecked` below): `idx` was validated in
        // the scan above; `p = term_id & mask` where `mask =
        // n_partitions - 1` (power-of-two enforced), so `p` indexes the
        // partition vec.
        let doc_pos_chain = &self.doc_pos_chain;
        let pos_scratch = &mut self.pos_scratch;
        let run_buf = &mut self.run_scratch;
        match store {
            PostingStore::Ram(ram) => {
                ram.grow_to(id_to_term.len());
                let mut new_bytes: usize = 0;
                for &term_id in updated_terms.iter() {
                    let idx = term_id as usize;
                    let slot = unsafe { dense_doc_tf.get_unchecked_mut(idx) };
                    let tf = *slot;
                    *slot = 0;
                    let lists = &mut ram.lists[idx];
                    let list = &mut lists.postings;
                    if list.is_empty() {
                        new_bytes = new_bytes
                            .saturating_add(ACCUM_NEW_TERM_FIXED_BYTES + id_to_term[idx].len());
                    }
                    list.push((local_doc_id, tf));
                    new_bytes = new_bytes.saturating_add(ACCUM_POSTING_BYTES);
                    // A positional column's run for this (term, doc) was
                    // written by the scan, gap by gap, as the tokens came.
                    if lane_mode && positional && list.len().is_multiple_of(RUN_END_STRIDE) {
                        lists.run_ends.push(lists.runs.len() as u32);
                    }
                }
                ram.bytes = ram
                    .bytes
                    .saturating_add(new_bytes)
                    .saturating_add(run_bytes);
                if ram.bytes > self.spill_threshold_bytes {
                    if self.lane_mode {
                        self.failed = true;
                        return Err(lane_outgrown());
                    }
                    // Crossed the threshold: move the postings to spill
                    // files, keyed by the ids already assigned, and run
                    // the rest of the build in spill mode.
                    // The store is taken before the spill can fail, so a
                    // failed spill leaves the builder unusable, not empty.
                    let spilled = match Self::spill_ram_store(
                        self.scratch_dir.path(),
                        column_id,
                        self.spill_partitions,
                        positional,
                        mem::replace(ram, RamStore::new()),
                    ) {
                        Ok(spilled) => spilled,
                        Err(error) => {
                            self.failed = true;
                            return Err(error);
                        }
                    };
                    *store = PostingStore::Spilled(spilled);
                    // In RAM the per-id state held last positions; the
                    // spilled scan reads it as chain heads.
                    dense_doc_poshead.fill(CHAIN_END);
                }
            }
            PostingStore::Spilled(SpillStore::Plain(partitions)) => {
                let mask = partitions.len() - 1;
                for &term_id in updated_terms.iter() {
                    let idx = term_id as usize;
                    let slot = unsafe { dense_doc_tf.get_unchecked_mut(idx) };
                    let tf = *slot;
                    *slot = 0;
                    let p = (term_id as usize) & mask;
                    let partition = unsafe { partitions.get_unchecked_mut(p) };
                    push_record_batched(partition, [term_id, local_doc_id, tf])?;
                }
            }
            PostingStore::Spilled(SpillStore::Positional { partitions, blobs }) => {
                let mask = partitions.len() - 1;
                for &term_id in updated_terms.iter() {
                    let idx = term_id as usize;
                    let slot = unsafe { dense_doc_tf.get_unchecked_mut(idx) };
                    let tf = *slot;
                    *slot = 0;
                    let head_slot = unsafe { dense_doc_poshead.get_unchecked_mut(idx) };
                    let head = *head_slot;
                    *head_slot = CHAIN_END;
                    pos_scratch.clear();
                    let mut at = head;
                    while at != CHAIN_END {
                        let (pos, prev) = doc_pos_chain[at as usize];
                        pos_scratch.push(pos);
                        at = prev;
                    }
                    pos_scratch.reverse();
                    debug_assert_eq!(pos_scratch.len() as u32, tf, "chain length must equal tf");
                    run_buf.clear();
                    encode_run(run_buf, pos_scratch);
                    let p = (term_id as usize) & mask;
                    let blob = &mut blobs[p];
                    let pos_off = blob.len;
                    blob.writer
                        .as_mut()
                        .expect("positions blob writer is open before finish")
                        .write_all(run_buf)?;
                    blob.len += run_buf.len() as u64;
                    let (lo, hi) = pos_off_lanes(pos_off);
                    let partition = unsafe { partitions.get_unchecked_mut(p) };
                    push_record_batched(partition, [term_id, local_doc_id, tf, lo, hi])?;
                }
            }
        }

        Ok(())
    }

    /// Push one prebuilt posting `(term, doc_id, tf, positions)` into the
    /// column's accumulator without tokenizing — the FTS-compaction-merge
    /// counterpart of [`add_doc`]. `positions` is empty for a non-positional
    /// column; otherwise it holds the `tf` token offsets for this `(term,
    /// doc)`. The concatenating merges feed each term's docs in ascending
    /// doc-id order (sequential inputs, dense remap), matching what
    /// `add_doc` produces. The multi-cell merge does not — it remaps
    /// postings through the packed stable-id row order — so it forces the
    /// column into spill mode first, where the finish restores per-term
    /// doc order (see `radix_sort_records_by_lex_rank`); the in-RAM
    /// accumulator preserves insertion order and must only be fed
    /// ascending.
    ///
    /// Doc lengths are fed separately via [`append_prebuilt_doc_lengths`] — the
    /// merge carries the inputs' stored lengths rather than recomputing them.
    ///
    /// Memory-bounded: while in-RAM, a push that crosses `spill_threshold_bytes`
    /// transitions the column to the spilled accumulator (the same
    /// [`add_doc`]-time transition), after which each posting is written
    /// straight to the partition/position spill files.
    ///
    /// [`add_doc`]: Self::add_doc
    /// [`append_prebuilt_doc_lengths`]: Self::append_prebuilt_doc_lengths
    pub(crate) fn add_prebuilt_term_posting(
        &mut self,
        column_id: u32,
        term: &str,
        doc_id: FtsDocId,
        tf: u32,
        positions: &[u32],
    ) -> Result<(), BuildError> {
        // The accumulators below are the blob's own id space from end to
        // end, so the id comes off here, once, at the entry point that
        // decides what space the caller had to be in.
        let doc_id = doc_id.get();
        let col_idx = column_id as usize;
        let positional = self.columns[col_idx].positions;
        let ColumnPostings { vocab, store } = &mut self.postings[col_idx];
        let term_id = vocab.intern(term, positional);
        let idx = term_id as usize;
        match store {
            PostingStore::Ram(ram) => {
                ram.grow_to(vocab.len());
                let mut new_bytes: usize = 0;
                let lists = &mut ram.lists[idx];
                let list = &mut lists.postings;
                if list.is_empty() {
                    new_bytes = new_bytes.saturating_add(ACCUM_NEW_TERM_FIXED_BYTES + term.len());
                }
                list.push((doc_id, tf));
                new_bytes = new_bytes.saturating_add(ACCUM_POSTING_BYTES);
                if positional {
                    let run = &mut lists.runs;
                    let before = run.len();
                    encode_run(run, positions);
                    new_bytes = new_bytes.saturating_add(run.len() - before);
                }
                ram.bytes = ram.bytes.saturating_add(new_bytes);
                if ram.bytes > self.spill_threshold_bytes {
                    if self.lane_mode {
                        self.failed = true;
                        return Err(lane_outgrown());
                    }
                    // The store is taken before the spill can fail, so a
                    // failed spill leaves the builder unusable, not empty.
                    let spilled = match Self::spill_ram_store(
                        self.scratch_dir.path(),
                        column_id,
                        self.spill_partitions,
                        positional,
                        mem::replace(ram, RamStore::new()),
                    ) {
                        Ok(spilled) => spilled,
                        Err(error) => {
                            self.failed = true;
                            return Err(error);
                        }
                    };
                    *store = PostingStore::Spilled(spilled);
                    vocab.dense_doc_poshead.fill(CHAIN_END);
                }
            }
            PostingStore::Spilled(SpillStore::Plain(parts)) => {
                let p = idx & (parts.len() - 1);
                push_record_batched(&mut parts[p], [term_id, doc_id, tf])?;
            }
            PostingStore::Spilled(SpillStore::Positional {
                partitions: parts,
                blobs,
            }) => {
                let p = idx & (parts.len() - 1);
                let blob = &mut blobs[p];
                let writer = blob
                    .writer
                    .as_mut()
                    .expect("positions blob writer open before finish");
                let run = &mut self.run_scratch;
                run.clear();
                encode_run(run, positions);
                writer.write_all(run)?;
                let pos_off = blob.len;
                blob.len += run.len() as u64;
                let (lo, hi) = pos_off_lanes(pos_off);
                push_record_batched(&mut parts[p], [term_id, doc_id, tf, lo, hi])?;
            }
        }
        Ok(())
    }

    /// Append a run of per-doc lengths to a column (the compaction merge feeds
    /// the inputs' already-clamped stored lengths rather than recomputing them
    /// from text) and advance the builder's doc count so `finish` sizes the
    /// doc-lengths table and `n_docs` correctly. Append (not replace) so the
    /// prebuilt feed composes: with lengths a prior input already carried
    /// across, and with lengths `add_doc` accumulated for docs tokenized into
    /// the same builder — the merged lengths axis stays aligned with the
    /// column's ascending local-doc-id axis either way.
    ///
    /// Also accumulates `total_tokens` as the sum of the lengths. `finish`
    /// derives the stored average from it, and a doc length *is*
    /// its token count (`add_doc` clamps only at `u32::MAX`, never reached in
    /// practice), so this sum equals what re-indexing the same corpus would
    /// accumulate — keeping merged BM25 scores identical to a fresh build.
    pub(crate) fn append_prebuilt_doc_lengths(&mut self, column_id: u32, doc_lengths: &[u32]) {
        let total_tokens: u64 = doc_lengths.iter().map(|&dl| u64::from(dl)).sum();
        let col = &mut self.columns[column_id as usize];
        col.total_tokens += total_tokens;
        col.doc_lengths
            .extend(doc_lengths.iter().map(|&dl| stored_doc_length(dl)));
        self.n_docs = self.n_docs.max(col.doc_lengths.len() as u32);
    }

    /// Whether every column's postings are still in RAM.
    pub(crate) fn is_in_ram(&self) -> bool {
        self.postings.iter().all(|cp| !cp.is_spilled())
    }

    /// Make this builder a lane of a build fed as its rows arrive. A lane
    /// keeps, beside its lists, where each posting's position run ends and
    /// each document's exact token count, so that once frozen any stretch
    /// of its rows can be cut out by row (see [`FrozenLane`]). A lane never
    /// spills: it holds a share of one commit's rows, and the finish reads
    /// it out of RAM.
    pub(crate) fn set_lane_mode(&mut self) {
        self.lane_mode = true;
    }

    /// Freeze this lane for the finish: `stretches` are the commit rows
    /// its documents are, covering documents `0..n_docs` in order with the
    /// rows ascending. The vocabulary's map and arena go; the lists stay as
    /// they are.
    pub(crate) fn freeze(self, stretches: Vec<Stretch>) -> Result<FrozenLane, BuildError> {
        if !self.lane_mode {
            return Err(BuildError::Io(Error::other(
                "only a builder in lane mode can be frozen",
            )));
        }
        if self.failed {
            return Err(failed_builder());
        }
        if !self.is_in_ram() {
            return Err(BuildError::Io(Error::other(
                "an fts lane that spilled cannot be frozen",
            )));
        }
        let mut docs = 0u32;
        let mut row_end = 0u32;
        for s in &stretches {
            if s.doc != docs || s.len == 0 || s.row < row_end {
                return Err(BuildError::Io(Error::other(
                    "an fts lane's stretches must cover its documents in order, rows ascending",
                )));
            }
            docs += s.len;
            row_end = s.row + s.len;
        }
        if docs != self.n_docs {
            return Err(BuildError::Io(Error::other(
                "an fts lane's stretches must cover exactly its documents",
            )));
        }
        let shape = self.shape();
        let columns = self
            .columns
            .into_iter()
            .zip(self.postings)
            .map(|(col, cp)| {
                let ColumnPostings { vocab, store } = cp;
                let PostingStore::Ram(ram) = store else {
                    unreachable!("checked to be in RAM above");
                };
                let Vocab {
                    term_to_id,
                    id_to_term,
                    term_arena,
                    ..
                } = vocab;
                drop(term_to_id);
                // Every table in dictionary order: the merged walk at the
                // finish then reads each lane front to back.
                let (_, by_lex) = build_lex_rank(&id_to_term);
                let mut term_bytes =
                    String::with_capacity(id_to_term.iter().map(|t| t.len()).sum());
                let term_spans = by_lex
                    .iter()
                    .map(|&id| {
                        let t = id_to_term[id as usize];
                        let start = term_bytes.len() as u32;
                        term_bytes.push_str(t);
                        (start, t.len() as u32)
                    })
                    .collect();
                // The arena outlives every read of `id_to_term` above.
                drop(id_to_term);
                drop(term_arena);
                // Split the per-id lists into the frozen column's tables,
                // in dictionary order. A positionless column keeps no runs
                // and so no run ends: those tables stay empty.
                let mut lists = ram.lists;
                let mut postings = Vec::with_capacity(by_lex.len());
                let mut run_ends = Vec::new();
                let mut pos_runs = Vec::new();
                if col.positions {
                    run_ends.reserve(by_lex.len());
                    pos_runs.reserve(by_lex.len());
                }
                for &id in by_lex.iter() {
                    let l = mem::take(&mut lists[id as usize]);
                    postings.push(l.postings);
                    if col.positions {
                        run_ends.push(l.run_ends);
                        pos_runs.push(l.runs);
                    }
                }
                drop(lists);
                FrozenColumn {
                    term_bytes,
                    term_spans,
                    postings,
                    run_ends,
                    pos_runs,
                    tokens: col.tokens,
                    cut_at: Vec::new(),
                    cuts: Vec::new(),
                }
            })
            .collect();
        Ok(FrozenLane {
            columns,
            shape,
            stretches,
        })
    }

    /// Finish from `view` rather than from this builder's own accumulator:
    /// the lanes tokenized the commit's rows between them, and this
    /// builder's superfile holds the rows `view.rows` of it. The finish
    /// merges the lanes' lists over those rows, term by term, so the blob
    /// is the one this builder would write had it been fed those rows in
    /// order itself. The lanes must hold this builder's columns, and this
    /// builder must not have accumulated postings of its own.
    pub(crate) fn set_lane_view(&mut self, view: LaneView) {
        self.lane_view = Some(view);
    }

    /// The builder's columns as a lane records them: each name and
    /// whether it keeps positions.
    fn shape(&self) -> Vec<(String, bool)> {
        self.columns
            .iter()
            .map(|c| (c.name.clone(), c.positions))
            .collect()
    }

    /// Fill this builder's document lengths and count from its lane view:
    /// each lane's lengths placed at the rows its documents are, numbered
    /// from the view's first row, the token totals summed exactly.
    fn take_lane_lengths(&mut self) -> Result<(), BuildError> {
        let view = self.lane_view.as_ref().expect("the caller checked");
        let shape = self.shape();
        let rows = view.rows().clone();
        let n_docs = rows.len();
        for lane in view.lanes.lanes.iter() {
            if lane.shape != shape {
                return Err(BuildError::Io(Error::other(
                    "fts lanes must hold the builder's columns",
                )));
            }
        }
        for (c, col) in self.columns.iter_mut().enumerate() {
            let mut lengths = vec![0u32; n_docs];
            let mut total_tokens = 0u64;
            for lane in view.lanes.lanes.iter() {
                let (dlo, dhi) = lane_doc_range(lane, &rows);
                let lane_col = &lane.columns[c];
                let mut cursor = 0usize;
                for d in dlo..dhi {
                    let row = row_of(&lane.stretches, &mut cursor, d as u32);
                    let tokens = lane_col.tokens[d];
                    lengths[(row - rows.start) as usize] = stored_doc_length(tokens);
                    total_tokens += u64::from(tokens);
                }
            }
            col.doc_lengths = lengths;
            col.total_tokens = total_tokens;
        }
        self.n_docs = n_docs as u32;
        Ok(())
    }

    /// Finalise and emit the FTS blob bytes. Consumes the builder.
    ///
    /// Returns `BuildError::Io` for scratch IO failures that can
    /// occur on the spill path (partition write/read, streaming-dictionary
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
    /// Three finish paths, picked automatically:
    ///
    /// - **In-RAM finish** (no column spilled): each column's ids are
    ///   sorted by their term bytes once and drained term by term,
    ///   encoded postings flow to a posting-region scratch file, and
    ///   the term dictionary is built in RAM via `TermDictBuilder`.
    ///   Small-build path; mirrors the in-RAM `VectorBuilder` finish.
    ///
    /// - **Lane finish** (a [`LaneView`] was set): the builder's own
    ///   accumulator is empty; the commit's frozen lanes are merged
    ///   over this superfile's rows, term by term, through the spilled
    ///   finish below.
    ///
    /// - **Spilled finish** (any column spilled): every column's
    ///   posting source — in-RAM map (for columns that stayed below
    ///   threshold) or partition files (for spilled columns) — is
    ///   normalised into a sorted record stream, then column-by-
    ///   column those streams feed a `TermBlockWriter` streaming the
    ///   term dictionary to a scratch file, so it never lives entirely
    ///   in RAM.
    ///   Mirrors the spilled `VectorBuilder` finish.
    ///
    /// Final blob assembly is byte-identical between the paths
    /// (regression-gated by
    /// `build_above_threshold_spills_and_matches_in_ram_byte_for_byte` and
    /// `lanes_cut_at_the_finish_write_the_blobs_single_builders_write`).
    pub fn finish_to<W: Write>(mut self, w: W) -> Result<(), BuildError> {
        if self.failed {
            return Err(failed_builder());
        }
        // Dispatch to the path that matches the builder's current
        // state. The two paths share `assemble_and_write_blob` as
        // their tail, so the output blob is byte-identical for
        // builds that *could* be served by either (regression-
        // gated by `build_above_threshold_spills_and_matches_in_
        // ram_byte_for_byte`).
        // The sorted merge supplies every posting; any the accumulator also
        // holds would be dropped from the output.
        if !self.sorted_inputs.is_empty() && self.postings.iter().any(|c| !c.is_empty()) {
            return Err(BuildError::Io(Error::other(
                "fts: sorted merge inputs set on a builder that also accumulated postings",
            )));
        }
        if self.lane_view.is_some() {
            if self.postings.iter().any(|c| !c.is_empty()) {
                return Err(BuildError::Io(Error::other(
                    "fts: a lane view set on a builder that also accumulated postings",
                )));
            }
            self.take_lane_lengths()?;
            return self.finish_to_spilled(w);
        }
        if !self.sorted_inputs.is_empty() || self.postings.iter().any(|c| c.is_spilled()) {
            self.finish_to_spilled(w)
        } else {
            self.finish_to_inram(w)
        }
    }

    /// In-RAM finish path: every column's accumulated `terms` map
    /// stayed under `spill_threshold_bytes`, so the term dictionary is
    /// built in one shot via [`TermDictBuilder`] (no scratch file), no
    /// partition flush runs, and the per-column emit loop only ever
    /// sees the `InRam` variant.
    ///
    /// This is the production rayon-sharded path for builds whose
    /// per-shard data fits below the spill threshold — the case the
    /// 1M-doc Zipfian bench measures.
    fn finish_to_inram<W: Write>(self, mut w: W) -> Result<(), BuildError> {
        let FtsBuilder {
            column_tokenizers: _,
            columns,
            postings,
            scratch_dir,
            spill_threshold_bytes: _,
            spill_partitions: _,
            max_partition_bytes: _,
            n_docs,
            doc_pos_chain: _,
            pos_scratch: _,
            run_scratch: _,
            doc_map,
            sorted_inputs: _,
            mut head_sink,
            head_depth,
            lane_mode: _,
            lane_view: _,
            failed: _,
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
        let mut n_scored_per_col: Vec<u32> = vec![0; n_columns as usize];
        for (orig_idx, state, _) in &work {
            let own = state.length_stats();
            avgdl_per_col[*orig_idx] = state.stored_average(&own);
            n_scored_per_col[*orig_idx] = own.n_scored_docs as u32;
        }
        let scratch_path = scratch_dir.path().to_path_buf();
        // Posting body scratch file. Encoded posting blocks for every
        // (column, term) flow here in lex order, then get streamed
        // through to `w` at assembly time.
        let postings_path = scratch_path.join("infino_fts_postings.bin");
        let mut postings_writer = BufWriter::new(File::create(&postings_path)?);
        let mut postings_len: u64 = 0;
        let mut postings_crc_acc: u32 = 0;
        // Every blob carries a positions region, empty when no column
        // records positions.
        let mut positions_sink = PositionsSink::create(&scratch_path)?;
        let mut key_buf: Vec<u8> = Vec::with_capacity(64);
        let mut term_scratch = TermScratch::default();
        let mut finish_profile = FinishProfile::from_config();

        // The in-RAM path's dictionary sink: collect (key, value) into a
        // `TermDictBuilder` and serialise once at assembly time. No
        // scratch file, no streaming.
        let mut dict_inram = TermDictBuilder::new();

        let mut doc_lengths_by_orig_col: Vec<Option<Vec<u32>>> =
            (0..n_columns as usize).map(|_| None).collect();

        for (orig_col_idx, col_state, posting_state) in work.drain(..) {
            let ColumnState {
                name: col_name,
                doc_lengths: col_doc_lengths_owned,
                total_tokens: _,
                tokens: _,
                positions: col_positions,
                params,
                corpus: _,
            } = col_state;
            let col_name_bytes = col_name.as_bytes();
            let avgdl = avgdl_per_col[orig_col_idx];
            let col_doc_lengths: &[u32] = &col_doc_lengths_owned;
            // The collection size idf is baked with: the documents that
            // carry tokens, matching what the reader divides by.
            let n_scored = n_scored_per_col[orig_col_idx];
            let enc = TermEncoding {
                doc_lengths: col_doc_lengths,
                avgdl,
                params,
                n_scored_docs: n_scored,
                heads: head_sink.is_some(),
                head_depth,
                rows: doc_map.as_deref(),
            };

            // In-RAM path invariant: the dispatcher checked that no column
            // spilled, so every store is the in-RAM one.
            let ColumnPostings { vocab, store } = posting_state;
            let ram = match store {
                PostingStore::Ram(ram) => ram,
                PostingStore::Spilled(_) => unreachable!(
                    "finish_to_inram dispatched on !any_spilled; \
                     a spilled column cannot appear here"
                ),
            };
            let mut sink = TermSink {
                col_name: &col_name,
                col_name_bytes,
                col_positions,
                key_buf: &mut key_buf,
                postings_writer: &mut postings_writer,
                postings_crc_acc: &mut postings_crc_acc,
                postings_len: &mut postings_len,
                positions_sink: &mut positions_sink,
                dict: DictSink::InRam(&mut dict_inram),
                head_sink: &mut head_sink,
            };
            n_terms_total_usize += emit_ram_column(
                vocab,
                ram,
                &enc,
                &mut finish_profile,
                &mut term_scratch,
                &mut sink,
            )?;

            doc_lengths_by_orig_col[orig_col_idx] = Some(col_doc_lengths_owned);
        }

        if let Some(sink) = head_sink.as_mut() {
            sink.finish()?;
        }
        assemble_and_write_blob(
            BlobAssemblyInputs {
                postings_writer,
                postings_path,
                postings_crc_acc,
                postings_len,
                positions_sink,
                dict_sink: DictSinkFinish::InRam(dict_inram),
                n_columns,
                n_docs,
                n_terms_total_usize,
                avgdl_per_col,
                doc_lengths_by_orig_col,
                scratch_dir,
                finish_profile,
                doc_map,
            },
            &mut w,
        )
    }

    /// Spilled finish path: at least one column transitioned to
    /// `Spilled` during `add_doc`. Uses a streaming `MapBuilder`
    /// (dictionary bytes go to a scratch file as we go, never resident in
    /// RAM in full), drains every spilled column's per-partition
    /// batch buffers + closes their `BufWriter`s before the merge
    /// reads them, and the per-column emit loop handles both
    /// variants (a spilled build can still have InRam columns whose
    /// accumulator stayed under threshold).
    #[cfg_attr(feature = "detailed-tracing", tracing::instrument(skip_all))]
    fn finish_to_spilled<W: Write>(self, mut w: W) -> Result<(), BuildError> {
        let FtsBuilder {
            column_tokenizers: _,
            columns,
            postings,
            scratch_dir,
            spill_threshold_bytes: _,
            spill_partitions: _,
            max_partition_bytes,
            n_docs,
            doc_pos_chain: _,
            pos_scratch: _,
            run_scratch: _,
            doc_map,
            sorted_inputs,
            mut head_sink,
            head_depth,
            lane_mode: _,
            lane_view,
            failed: _,
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
        let mut n_scored_per_col: Vec<u32> = vec![0; n_columns as usize];
        for (orig_idx, state, _) in &work {
            let own = state.length_stats();
            avgdl_per_col[*orig_idx] = state.stored_average(&own);
            n_scored_per_col[*orig_idx] = own.n_scored_docs as u32;
        }

        let scratch_path = scratch_dir.path().to_path_buf();
        let postings_path = scratch_path.join("infino_fts_postings.bin");
        let mut postings_writer = BufWriter::new(File::create(&postings_path)?);
        let mut postings_len: u64 = 0;
        let mut postings_crc_acc: u32 = 0;
        // Every blob carries a positions region, empty when no column
        // records positions.
        let mut positions_sink = PositionsSink::create(&scratch_path)?;
        let mut key_buf: Vec<u8> = Vec::with_capacity(64);
        let mut term_scratch = TermScratch::default();
        let mut finish_profile = FinishProfile::from_config();

        // Streaming term dictionary: bytes flow to a scratch file as we insert
        // sorted keys, so the term dictionary never lives in RAM in full.
        // Assembly reopens the file, CRCs it, and copies it into the
        // output writer.
        let dict_streaming_path = scratch_path.join("infino_fts_dict.bin");
        let mut dict_streaming = {
            let dict_file = File::create(&dict_streaming_path)?;
            let bw = BufWriter::new(dict_file);
            TermBlockWriter::new(bw)
        };

        // Drain every spilled column's per-partition batch buffer
        // into the partition's `BufWriter`, then flush + close it
        // so the merge phase reads a complete file. InRam columns
        // in this work list are no-ops.
        let partition_flush_start = finish_profile.enabled.then(Instant::now);
        for (_, _, cp) in &mut work {
            if let PostingStore::Spilled(partitions) = &mut cp.store {
                match partitions {
                    SpillStore::Plain(parts) => {
                        for partition in parts {
                            flush_partition_batch(partition)?;
                            if let Some(mut writer) = partition.writer.take() {
                                writer.flush()?;
                            }
                        }
                    }
                    SpillStore::Positional { partitions, blobs } => {
                        for partition in partitions.iter_mut() {
                            flush_partition_batch(partition)?;
                            if let Some(mut writer) = partition.writer.take() {
                                writer.flush()?;
                            }
                        }
                        for blob in blobs.iter_mut() {
                            if let Some(mut writer) = blob.writer.take() {
                                writer.flush()?;
                            }
                        }
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
                tokens: _,
                positions: col_positions,
                params,
                corpus: _,
            } = col_state;
            let col_name_bytes = col_name.as_bytes();
            let avgdl = avgdl_per_col[orig_col_idx];
            let col_doc_lengths: &[u32] = &col_doc_lengths_owned;
            // The collection size idf is baked with: the documents that
            // carry tokens, matching what the reader divides by.
            let n_scored = n_scored_per_col[orig_col_idx];
            let enc = TermEncoding {
                doc_lengths: col_doc_lengths,
                avgdl,
                params,
                n_scored_docs: n_scored,
                heads: head_sink.is_some(),
                head_depth,
                rows: doc_map.as_deref(),
            };

            let mut sink = TermSink {
                col_name: &col_name,
                col_name_bytes,
                col_positions,
                key_buf: &mut key_buf,
                postings_writer: &mut postings_writer,
                postings_crc_acc: &mut postings_crc_acc,
                postings_len: &mut postings_len,
                positions_sink: &mut positions_sink,
                dict: DictSink::Streaming(&mut dict_streaming),
                head_sink: &mut head_sink,
            };
            // A build fed by lanes: the accumulator is empty (checked in
            // `finish_to`) and the lanes are merged term by term.
            if let Some(view) = lane_view.as_ref() {
                n_terms_total_usize += emit_lanes_column(
                    view,
                    orig_col_idx,
                    &enc,
                    &mut finish_profile,
                    &mut term_scratch,
                    &mut sink,
                )?;
            } else if !sorted_inputs.is_empty() {
                // The compaction merge: the accumulator is empty (checked in
                // `finish_to`) and the inputs are merged term by term.
                let merge_span = detail_span!(
                    "fts_sorted_merge",
                    column = col_name.as_str(),
                    inputs = sorted_inputs.len(),
                    terms = tracing::field::Empty,
                    walk_wait_ms = tracing::field::Empty,
                    encode_ms = tracing::field::Empty,
                    write_ms = tracing::field::Empty,
                )
                .entered();
                let profiling = finish_profile.enabled;
                // Terms are encoded on many threads, each with its own
                // scratch and profile, and written here in term order.
                let mut n_emitted: usize = 0;
                let workers = merge_column(
                    &sorted_inputs,
                    orig_col_idx as u32,
                    || {
                        let profile = FinishProfile {
                            enabled: profiling,
                            ..FinishProfile::default()
                        };
                        (TermScratch::default(), profile)
                    },
                    |(scratch, profile), pairs, runs| {
                        let positions = col_positions.then_some(runs);
                        let encoded = encode_term_timed(pairs, positions, &enc, profile, scratch)?;
                        // A long term's buffers are moved out rather than
                        // copied, so a worker does not keep their size. A
                        // short body is small; copying it keeps the
                        // scratch's capacity for the next term.
                        let (body, positions) = match encoded {
                            EncodedTerm::Inline(_) => (Vec::new(), Vec::new()),
                            EncodedTerm::Short => (scratch.term_buf.clone(), Vec::new()),
                            EncodedTerm::Long => {
                                // Positions grow without a reserve; drop
                                // the spare capacity while the term waits.
                                let mut positions = mem::take(&mut scratch.pos_out);
                                positions.shrink_to_fit();
                                // Its blocks' bytes are in the body now.
                                scratch.encoded_blocks.clear();
                                (mem::take(&mut scratch.term_buf), positions)
                            }
                        };
                        Ok(MergedTerm {
                            encoded,
                            body,
                            positions,
                            head: scratch.head.clone(),
                        })
                    },
                    |term, mut merged| {
                        n_emitted += 1;
                        write_term(
                            term,
                            merged.encoded,
                            &mut merged.body,
                            &merged.positions,
                            &merged.head,
                            &mut finish_profile,
                            &mut sink,
                        )
                    },
                )?;
                for (_, profile) in &workers {
                    finish_profile.absorb(profile);
                }
                n_terms_total_usize += n_emitted;
                record("terms", n_emitted as u64);
                drop(merge_span);
            } else {
                let ColumnPostings { vocab, store } = posting_state;
                match store {
                    // A column that stayed under the threshold in a build
                    // where another spilled: emitted straight from RAM,
                    // into the streaming dictionary like its neighbours.
                    PostingStore::Ram(ram) => {
                        n_terms_total_usize += emit_ram_column(
                            vocab,
                            ram,
                            &enc,
                            &mut finish_profile,
                            &mut term_scratch,
                            &mut sink,
                        )?;
                    }
                    PostingStore::Spilled(partitions) => {
                        let Vocab {
                            term_to_id,
                            id_to_term,
                            term_arena,
                            ..
                        } = vocab;
                        // Term interner is finished being written to; drop the
                        // forward map (`term_to_id`) immediately — the rest of
                        // the spilled finish only needs the reverse map
                        // (`id_to_term`) for dictionary emit and the lex-rank
                        // table built from it.
                        //
                        // Lifetime sanity: `term_to_id` and `id_to_term`
                        // hold `&'static str` keys/entries that actually
                        // borrow from `term_arena`. We must keep
                        // `term_arena` alive until the **last
                        // dereference** of those keys/entries, which is
                        // the dictionary emit loop's `id_to_term[term_id]`
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
                        let mut sorted_files: Vec<PathBuf> =
                            Vec::with_capacity(partitions.n_partitions());
                        let partition_paths: Vec<PathBuf> = match &partitions {
                            SpillStore::Plain(parts) => {
                                parts.iter().map(|p| p.path.clone()).collect()
                            }
                            SpillStore::Positional {
                                partitions: parts, ..
                            } => parts.iter().map(|p| p.path.clone()).collect(),
                        };
                        let sort_span = detail_span!(
                            "fts_partition_sort",
                            column = col_name.as_str(),
                            partitions = partition_paths.len()
                        )
                        .entered();
                        for (partition_idx, partition_path) in partition_paths.iter().enumerate() {
                            let sorted_path = scratch_path.join(format!(
                                "fts_col{orig_col_idx}_part{partition_idx}.sorted.bin"
                            ));
                            match &partitions {
                                SpillStore::Plain(_) => {
                                    sort_partition_to_file::<PLAIN_RECORD_LANES>(
                                        partition_path,
                                        &sorted_path,
                                        max_partition_bytes,
                                        &scratch_path,
                                        &format!("c{orig_col_idx}_p{partition_idx}"),
                                        &lex_rank,
                                    )?
                                }
                                SpillStore::Positional { .. } => {
                                    sort_partition_to_file::<POSITIONAL_RECORD_LANES>(
                                        partition_path,
                                        &sorted_path,
                                        max_partition_bytes,
                                        &scratch_path,
                                        &format!("c{orig_col_idx}_p{partition_idx}"),
                                        &lex_rank,
                                    )?
                                }
                            }
                            sorted_files.push(sorted_path);
                        }
                        drop(sort_span);
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
                        let merge_profile_start = Instant::now();
                        let encode_calls_before = finish_profile.encode_calls;
                        let encode_df1_before = finish_profile.encode_df1;
                        let encode_pfor_before = finish_profile.encode_pfor;
                        let encode_total_before = finish_profile.encode_total;
                        let encode_block_build_before = finish_profile.encode_block_build;
                        let encode_meta_write_before = finish_profile.encode_meta_write;
                        let encode_skip_write_before = finish_profile.encode_skip_write;
                        let encode_block_write_before = finish_profile.encode_block_write;
                        let dict_insert_before = finish_profile.dict_insert;
                        let emit_span = detail_span!(
                            "fts_emit",
                            column = col_name.as_str(),
                            terms = tracing::field::Empty,
                            encode_ms = tracing::field::Empty,
                            gather_ms = tracing::field::Empty,
                            block_build_ms = tracing::field::Empty,
                            block_write_ms = tracing::field::Empty,
                            dict_insert_ms = tracing::field::Empty,
                        )
                        .entered();
                        let n_emitted = match &partitions {
                            SpillStore::Plain(_) => merge_sorted_spill::<PLAIN_RECORD_LANES, _>(
                                &sorted_files,
                                None,
                                &term_id_in_lex_order,
                                &id_to_term,
                                &enc,
                                &mut finish_profile,
                                &mut term_scratch,
                                &mut sink,
                            )?,
                            SpillStore::Positional { blobs, .. } => {
                                // mmap each partition's positions blob so
                                // run assembly is pointer arithmetic; an
                                // empty blob maps to an empty slice.
                                let mut blob_mmaps: Vec<Option<Mmap>> =
                                    Vec::with_capacity(blobs.len());
                                for blob in blobs {
                                    let f = File::open(&blob.path)?;
                                    // SAFETY: the blob scratch file is owned
                                    // by this builder's `scratch_dir`; its
                                    // writer was flushed + closed in the
                                    // partition-flush stage and nothing
                                    // mutates it for the Mmap's lifetime.
                                    let mmap = match blob.len {
                                        0 => None,
                                        _ => Some(unsafe { Mmap::map(&f)? }),
                                    };
                                    blob_mmaps.push(mmap);
                                }
                                merge_sorted_spill::<POSITIONAL_RECORD_LANES, _>(
                                    &sorted_files,
                                    Some(&blob_mmaps),
                                    &term_id_in_lex_order,
                                    &id_to_term,
                                    &enc,
                                    &mut finish_profile,
                                    &mut term_scratch,
                                    &mut sink,
                                )?
                            }
                        };
                        n_terms_total_usize += n_emitted;
                        record("terms", n_emitted as u64);
                        if finish_profile.enabled {
                            let merge_total = merge_profile_start.elapsed();
                            let encode_total = finish_profile.encode_total - encode_total_before;
                            let non_encode = merge_total.saturating_sub(encode_total);
                            record("encode_ms", encode_total.as_millis() as u64);
                            record("gather_ms", non_encode.as_millis() as u64);
                            record(
                                "block_build_ms",
                                (finish_profile.encode_block_build - encode_block_build_before)
                                    .as_millis() as u64,
                            );
                            record(
                                "block_write_ms",
                                (finish_profile.encode_block_write - encode_block_write_before)
                                    .as_millis() as u64,
                            );
                            record(
                                "dict_insert_ms",
                                (finish_profile.dict_insert - dict_insert_before).as_millis()
                                    as u64,
                            );
                            debug!(
                                "[fts-profile] col='{}' merge_total={:.3}s non_encode_merge={:.3}s encode_total={:.3}s calls={} df1={} pfor={} block_build={:.3}s meta_write={:.3}s skip_write={:.3}s block_write={:.3}s dict_insert={:.3}s",
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
                                (finish_profile.dict_insert - dict_insert_before).as_secs_f64(),
                            );
                        }
                        drop(emit_span);

                        // Sorted-partition scratch files are scoped to
                        // this column and only consumed by the k-way
                        // merge above. Drop the mmap views first
                        // (releases the page-cache references), then
                        // remove the files so the next spilled column
                        // doesn't see their disk residency. (Original
                        // partition files are owned by `partitions`
                        // and dropped at the next iteration boundary.)
                        let cleanup_start = finish_profile.enabled.then(Instant::now);
                        for p in &sorted_files {
                            let _ = fs::remove_file(p);
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
                        // at the dictionary emit loop above), then
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

        if let Some(sink) = head_sink.as_mut() {
            sink.finish()?;
        }
        assemble_and_write_blob(
            BlobAssemblyInputs {
                postings_writer,
                postings_path,
                postings_crc_acc,
                postings_len,
                positions_sink,
                dict_sink: DictSinkFinish::Streaming {
                    builder: dict_streaming,
                    path: dict_streaming_path,
                },
                n_columns,
                n_docs,
                n_terms_total_usize,
                avgdl_per_col,
                doc_lengths_by_orig_col,
                scratch_dir,
                finish_profile,
                doc_map,
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
/// Streaming sink for the positions region, mirroring the postings
/// scratch-file quartet (writer, path, running CRC, running length).
/// Created by a finish path iff any registered column is positional;
/// its running `len` doubles as the next term's `positions_offset`
/// (offsets in term metadata are relative to the region start).
struct PositionsSink {
    writer: BufWriter<File>,
    path: PathBuf,
    crc_acc: u32,
    len: u64,
}

impl PositionsSink {
    fn create(scratch_path: &Path) -> Result<Self, BuildError> {
        let path = scratch_path.join("infino_fts_positions.bin");
        Ok(Self {
            writer: BufWriter::new(File::create(&path)?),
            path,
            crc_acc: 0,
            len: 0,
        })
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), BuildError> {
        write_counted(&mut self.writer, &mut self.crc_acc, &mut self.len, bytes)
    }
}

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
    /// Positions region sink. Always present: the region (possibly just
    /// its CRC-of-empty trailer) sits between the postings and the
    /// doc-lengths directory.
    positions_sink: PositionsSink,
    /// Whichever dictionary sink was used during the per-column emit loop
    /// (exactly one of the two variants).
    dict_sink: DictSinkFinish,
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
    /// streamed regions (term dictionary + postings) have been copied into `w`.
    scratch_dir: TempDir,
    /// Profile accumulator — final block of `[fts-finish]` timings
    /// is emitted at the bottom of assembly.
    finish_profile: FinishProfile,
    /// See [`FtsBuilder::doc_map`]. `Some` makes this a `V8` blob.
    doc_map: Option<Vec<u32>>,
}

/// Dictionary emission sink picked by the active finish path.
enum DictSinkFinish {
    /// In-RAM build: hand the populated `TermDictBuilder` to assembly,
    /// which calls `finish()` to produce the dictionary bytes in one shot.
    InRam(TermDictBuilder),
    /// Spilled build: hand the open `TermBlockWriter` (and the
    /// scratch path it's been writing to) to assembly, which finishes
    /// the builder, computes the file's CRC by streaming, and copies
    /// the file into the output.
    Streaming {
        builder: TermBlockWriter<BufWriter<File>>,
        path: PathBuf,
    },
}

/// Common tail of every finish path: close the posting body, finalise
/// the term dictionary, build the doc-lengths directory + arrays, and
/// write `[header | dict | postings | positions | doc map | dir |
/// arrays]` to `w`. Lifted out of `FtsBuilder::finish_to` so the in-RAM
/// and spilled paths share one regression target instead of two — every
/// byte the reader observes passes through here.
fn assemble_and_write_blob<W: Write>(
    inputs: BlobAssemblyInputs,
    w: &mut W,
) -> Result<(), BuildError> {
    let BlobAssemblyInputs {
        mut postings_writer,
        postings_path,
        postings_crc_acc,
        mut postings_len,
        positions_sink,
        dict_sink,
        n_columns,
        n_docs,
        n_terms_total_usize,
        avgdl_per_col,
        mut doc_lengths_by_orig_col,
        scratch_dir,
        mut finish_profile,
        doc_map,
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

    // Close the positions region file the same way (CRC trailer +
    // flush). A build with no positional column (or whose positional
    // terms all inlined) leaves a body of zero bytes — the region is
    // then just the 4-byte CRC-of-empty, the same legal shape an
    // empty postings region takes.
    let positions_region = {
        let mut sink = positions_sink;
        let crc_le = sink.crc_acc.to_le_bytes();
        sink.writer.write_all(&crc_le)?;
        sink.writer.flush()?;
        drop(sink.writer);
        (sink.path, sink.len + crc_le.len() as u64)
    };

    // Finalise the term dictionary. Either path produces "dictionary bytes
    // followed by 4 trailing CRC bytes"; the source differs.
    enum DictSource {
        InRam(Vec<u8>),
        Streamed { path: PathBuf, len: u64, crc: u32 },
    }
    let dict_close_start = finish_profile.enabled.then(Instant::now);
    let dict_source = match dict_sink {
        DictSinkFinish::InRam(db) => {
            let mut bytes = db.finish();
            let crc = crc32c(&bytes);
            bytes.extend_from_slice(&crc.to_le_bytes());
            DictSource::InRam(bytes)
        }
        DictSinkFinish::Streaming {
            builder,
            path: dict_streaming_path,
        } => {
            let mut bw = builder.finish()?;
            bw.flush()?;
            // Close the write side of the dictionary scratch file. The
            // returned `File` is `File::create`-opened (write-only),
            // so we must reopen for reading to compute the CRC and
            // later stream into `w`.
            let write_file = bw
                .into_inner()
                .map_err(|e| BuildError::Io(e.into_error()))?;
            drop(write_file);

            // Stream the dictionary scratch file with bounded memory to
            // compute its CRC.
            let mut read_file = File::open(&dict_streaming_path)?;
            let dict_body_len = read_file.metadata()?.len();
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
            DictSource::Streamed {
                path: dict_streaming_path,
                len: dict_body_len + 4, /* trailing CRC */
                crc,
            }
        }
    };
    if let Some(t) = dict_close_start {
        finish_profile.dict_close += t.elapsed();
    }

    // Compute final-blob offsets now that both region lengths are known.
    let dl_emit_start = finish_profile.enabled.then(Instant::now);
    let dict_total_len: u64 = match &dict_source {
        DictSource::InRam(bytes) => bytes.len() as u64,
        DictSource::Streamed { len, .. } => *len,
    };
    let fts_version = blob_version();
    let header_size = format::fts::HEADER_SIZE as u64;
    let dict_offset: u64 = header_size;
    let postings_offset: u64 = dict_offset + dict_total_len;
    // The positions region sits between the postings and the
    // doc-lengths directory (keeping the lazy-open doc-lengths tail
    // fetch small); absent, the directory follows postings directly.
    let positions_offset: u64 = postings_offset + postings_len;
    // One `u32` per document plus the region CRC, or nothing at all.
    let doc_map_offset: u64 = positions_offset + positions_region.1;
    let doc_map_len: u64 = match &doc_map {
        Some(map) => (map.len() * format::fts::U32_BYTES + format::CRC_BYTES) as u64,
        None => 0,
    };
    // The map's length as a trailer the reader finds by stepping back from
    // the directory, so one version describes a blob with a map and one
    // without while the header keeps its single width.
    let doc_map_len_trailer = format::fts::doc_map_len_trailer(fts_version) as u64;
    let doc_lengths_table_offset: u64 = doc_map_offset + doc_map_len + doc_map_len_trailer;
    let entry_size = doc_lengths_entry_size(fts_version);
    let mut doc_lengths_array_offset: u64 =
        doc_lengths_table_offset + (n_columns as u64) * (entry_size as u64) + 4 /* dir CRC */;

    let mut dir_buf: Vec<u8> = Vec::with_capacity(n_columns as usize * entry_size);
    let mut arrays_buf: Vec<u8> = Vec::new();
    for i in 0..n_columns as usize {
        let avgdl_x1000 = bm25::avgdl_x1000(avgdl_per_col[i]);
        dir_buf.extend_from_slice(&(i as u32).to_le_bytes());
        dir_buf.extend_from_slice(&doc_lengths_array_offset.to_le_bytes());
        dir_buf.extend_from_slice(&avgdl_x1000.to_le_bytes());

        let col_dls = doc_lengths_by_orig_col[i]
            .take()
            .expect("doc_lengths recorded for every registered column");
        let array_start = arrays_buf.len();
        // `add_doc` / the prebuilt carry already saturated every length to
        // the stored width.
        for &dl in &col_dls {
            arrays_buf.extend_from_slice(&(dl as u16).to_le_bytes());
        }
        // The buckets the scorer indexes by, written beside the lengths they
        // quantize so a reader scores without deriving them. The totals and
        // the occupied range come from the same pass and go in the entry, so
        // nothing about this column costs a reader a walk.
        let mut stats = ColumnLengthStats::default();
        let mut occupied_lo = u8::MAX;
        let mut occupied_hi = u8::MIN;
        for &dl in &col_dls {
            let bucket = bm25::quantize_len(dl);
            occupied_lo = occupied_lo.min(bucket);
            occupied_hi = occupied_hi.max(bucket);
            arrays_buf.push(bucket);
            stats.add(dl);
        }
        if col_dls.is_empty() {
            occupied_lo = 0;
            occupied_hi = 0;
        }
        dir_buf.extend_from_slice(&stats.total_tokens.to_le_bytes());
        dir_buf.extend_from_slice(&stats.n_scored_docs.to_le_bytes());
        dir_buf.push(occupied_lo);
        dir_buf.push(occupied_hi);

        // One CRC over both arrays: they are written together and a reader
        // that trusts one must trust the other.
        let array_bytes = &arrays_buf[array_start..];
        let array_crc = crc32c(array_bytes);
        arrays_buf.extend_from_slice(&array_crc.to_le_bytes());
        doc_lengths_array_offset += (col_dls.len()
            * (format::fts::DOC_LENGTH_BYTES + format::fts::DOC_LENGTH_BUCKET_BYTES))
            as u64
            + format::CRC_BYTES as u64;
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
    header.extend_from_slice(&fts_version.to_le_bytes()); // 4
    header.extend_from_slice(&n_columns.to_le_bytes()); // 4
    header.extend_from_slice(&n_docs.to_le_bytes()); // 4
    header.extend_from_slice(&n_terms_total.to_le_bytes()); // 4
    header.extend_from_slice(&dict_offset.to_le_bytes()); // 8
    header.extend_from_slice(&postings_offset.to_le_bytes()); // 8
    header.extend_from_slice(&doc_lengths_table_offset.to_le_bytes()); // 8
    header.extend_from_slice(&positions_offset.to_le_bytes()); // 8
    // The doc-id map needs no header field: its length is the trailer written
    // immediately before the doc-lengths directory, which a reader already
    // fetches. That is what keeps every version's header one width, and a
    // cold open to one read.
    debug_assert_eq!(header.len(), header_size as usize, "header size mismatch");
    debug_assert!(
        doc_map.as_ref().is_none_or(|m| doc_lengths_table_offset
            == doc_map_offset
                + (m.len() * format::fts::U32_BYTES + format::CRC_BYTES) as u64
                + doc_map_len_trailer),
        "the doc-id map and its length trailer must be the last things before \
         the doc-lengths directory, or a reader cannot find where the map starts"
    );

    w.write_all(&header)?;
    match dict_source {
        DictSource::InRam(bytes) => w.write_all(&bytes)?,
        DictSource::Streamed { path, crc, .. } => {
            let mut reader = BufReader::with_capacity(PARTITION_BUF_SIZE, File::open(&path)?);
            io::copy(&mut reader, w)?;
            w.write_all(&crc.to_le_bytes())?;
        }
    }
    let mut postings_reader =
        BufReader::with_capacity(PARTITION_BUF_SIZE, File::open(&postings_path)?);
    io::copy(&mut postings_reader, w)?;
    drop(postings_reader);
    {
        let mut positions_reader =
            BufReader::with_capacity(PARTITION_BUF_SIZE, File::open(&positions_region.0)?);
        io::copy(&mut positions_reader, w)?;
    }

    // Drop the scratch tempdir as soon as the streamed source
    // files (term dictionary + posting body) have been copied into `w`. The
    // remaining writes (`dir_buf`, `arrays_buf`) are
    // already-resident `Vec<u8>` and don't touch the disk.
    // Mirror of vector's `drop(scratch_dir);` at the bottom of
    // `VectorBuilder::finish_to`.
    drop(scratch_dir);

    if let Some(map) = &doc_map {
        debug_assert_eq!(
            map.len() as u32,
            n_docs,
            "doc-id map must carry one row per document"
        );
        let mut map_buf: Vec<u8> = Vec::with_capacity(map.len() * format::fts::U32_BYTES);
        for row in map {
            map_buf.extend_from_slice(&row.to_le_bytes());
        }
        let map_crc = crc32c(&map_buf);
        w.write_all(&map_buf)?;
        w.write_all(&map_crc.to_le_bytes())?;
    }
    // The trailer the reader steps back to, written whether or not there is a
    // map: zero says there is none.
    if doc_map_len_trailer > 0 {
        w.write_all(&doc_map_len.to_le_bytes())?;
    }

    w.write_all(&dir_buf)?;
    w.write_all(&arrays_buf)?;
    if let Some(t) = blob_copy_start {
        finish_profile.blob_copy += t.elapsed();
    }

    if finish_profile.enabled {
        debug!(
            "[fts-finish] partition_flush={:.3}s lex_rank={:.3}s partition_sort={:.3}s mmap_open={:.3}s scratch_cleanup={:.3}s postings_close={:.3}s dict_close={:.3}s doc_lengths_emit={:.3}s blob_copy={:.3}s",
            finish_profile.partition_flush.as_secs_f64(),
            finish_profile.lex_rank_build.as_secs_f64(),
            finish_profile.partition_sort.as_secs_f64(),
            finish_profile.mmap_open.as_secs_f64(),
            finish_profile.scratch_cleanup.as_secs_f64(),
            finish_profile.postings_close.as_secs_f64(),
            finish_profile.dict_close.as_secs_f64(),
            finish_profile.doc_lengths_emit.as_secs_f64(),
            finish_profile.blob_copy.as_secs_f64(),
        );
    }

    Ok(())
}

/// Reusable per-term scratch buffers threaded through `encode_term`.
/// Each thread that encodes terms owns one and re-uses it across every
/// term it encodes, turning ~3M+ per-term `Vec::new` allocations on the
/// 1M-doc forced-spill bench into ~5 reused allocations (one per buffer,
/// once per column). The compaction merge moves a long term's `term_buf`
/// and `pos_out` out, so those two are allocated again per long term. The
/// Block's `doc_ids` / `tfs` Vecs are
/// `mem::take`'d into a Block, encoded, then swapped back via
/// `mem::take` so the underlying buffer is reused for the next
/// chunk (same allocation, just `Vec::clear` between iterations).
#[derive(Default)]
struct TermScratch {
    /// The term's head for the sink, when one is attached, and each
    /// posting's score, computed once for the head and the block maxima.
    head: HeadOut,
    head_heap: BinaryHeap<TopKEntry<(u32, u32)>>,
    /// The documents the head keeps, drained from the heap: `(doc, tf)`
    /// with the score, in descending score.
    head_kept: Vec<((u32, u32), f32)>,
    scores: Vec<f32>,
    /// Per-block doc_id column passed to `encode_block`. Re-used by
    /// taking it into a Block, encoding, and taking it back. Capacity
    /// stabilises at `BLOCK_LEN` after the first dense term.
    doc_ids: Vec<u32>,
    /// Per-block tf column, same lifecycle as `doc_ids`.
    tfs: Vec<u32>,
    /// Per-block BM25 upper bound for the skip table — the maximum per-doc
    /// contribution over the block's docs, scored with the same quantized
    /// document length the reader uses (not the looser `score(max_tf,
    /// min_dl)`, whose best-case tf and best-case dl need not co-occur in
    /// one doc). Scoring against the quantized length keeps the bound a
    /// true upper bound over query-time scores; a tighter bound lets the
    /// reader's block-skip fire more often.
    block_maxes: Vec<f32>,
    /// Per-term list of encoded blocks held across the meta + skip-
    /// table + block-bytes emit stages.
    encoded_blocks: Vec<EncodedBlock>,
    /// Per-term contiguous byte buffer holding meta + skip table +
    /// concatenated block bytes; written to `postings_writer` as one
    /// `write_counted` call.
    term_buf: Vec<u8>,
    /// Per-term positions-block offsets (one per PFOR block, byte
    /// offset of the block's first run within the term's positions
    /// bytes) for positional columns. Reused like the other buffers.
    pos_block_offsets: Vec<u32>,
    /// A long term's positions region: every block's group back to back.
    /// Reused across terms.
    pos_out: Vec<u8>,
    /// The patched encoders' planning buffers.
    pack: PackScratch,
    /// One block's run values (first position absolute per doc, then
    /// gaps) decoded from the accumulator's LEB128 runs for regrouping.
    pos_vals: Vec<u32>,
}

/// Merge one spilled column's sorted partition files and emit every
/// term. Shared by the plain and positional stores: the lex-order
/// walk, contiguous-run drain, and emit sequence are identical; a
/// positional store (`blob_mmaps = Some`) additionally recovers each
/// posting's position run from its partition's blob — sliced at the
/// offset riding in the record's trailing lanes, delimited by the
/// posting's `tf` — and hands the term's assembled runs to the
/// encoder. Returns the number of terms emitted.
#[allow(clippy::too_many_arguments)]
fn merge_sorted_spill<const N: usize, W: Write>(
    sorted_files: &[PathBuf],
    blob_mmaps: Option<&[Option<Mmap>]>,
    term_id_in_lex_order: &[u32],
    id_to_term: &[&'static str],
    enc: &TermEncoding<'_>,
    finish_profile: &mut FinishProfile,
    term_scratch: &mut TermScratch,
    sink: &mut TermSink<'_, W>,
) -> Result<usize, BuildError> {
    let mmap_start = finish_profile.enabled.then(Instant::now);
    let mut mmaps: Vec<Mmap> = Vec::with_capacity(sorted_files.len());
    for p in sorted_files {
        let f = File::open(p)?;
        // SAFETY: the sorted-partition scratch file is owned by this
        // builder's `scratch_dir`; no other process truncates or
        // appends to it for the lifetime of the `Mmap`.
        let mmap = unsafe { Mmap::map(&f)? };
        mmaps.push(mmap);
    }
    if let Some(t) = mmap_start {
        finish_profile.mmap_open += t.elapsed();
    }
    let sorted_slices: Vec<&[[u32; N]]> = mmaps
        .iter()
        .map(|m| {
            if m.is_empty() {
                &[][..]
            } else {
                bytemuck::cast_slice::<u8, [u32; N]>(&m[..])
            }
        })
        .collect();
    let blob_slices: Vec<&[u8]> = match blob_mmaps {
        Some(ms) => ms
            .iter()
            .map(|m| m.as_ref().map_or(&[][..], |m| &m[..]))
            .collect(),
        None => Vec::new(),
    };
    let mask = (sorted_slices.len() - 1) as u32;
    // Per-partition next-index into its sorted slice. Walked forward
    // only — each posting is read exactly once.
    let mut cursors: Vec<usize> = vec![0usize; sorted_slices.len()];

    // Per-term buffers, reused across all terms in this column so we
    // pay one growth schedule instead of one per term.
    let mut group: Vec<(u32, u32)> = Vec::new();
    let mut group_pos: Vec<u64> = Vec::new();
    let mut term_run: Vec<u8> = Vec::new();
    let mut n_emitted = 0usize;
    for &term_id in term_id_in_lex_order {
        let p = (term_id & mask) as usize;
        let slice = sorted_slices[p];
        let mut pos = cursors[p];
        group.clear();
        group_pos.clear();
        // Drain the contiguous run for this term. Termination: either
        // the partition runs out, or the next record's term_id differs
        // (next term in this partition's lex-rank order, which can
        // only be a strictly higher `lex_rank` and so a different
        // `term_id`).
        while pos < slice.len() {
            let t = &slice[pos];
            if triple_term_id(t) != term_id {
                break;
            }
            group.push((triple_doc_id(t), triple_tf(t)));
            if blob_mmaps.is_some() {
                // The generic walk serves both widths; the offset
                // lanes exist only on the positional record, which is
                // the only case with `blob_mmaps` present.
                debug_assert_eq!(N, POSITIONAL_RECORD_LANES);
                let rec: &[u32; POSITIONAL_RECORD_LANES] =
                    t[..].try_into().expect("positional record width");
                group_pos.push(record_pos_off(rec));
            }
            pos += 1;
        }
        cursors[p] = pos;
        if group.is_empty() {
            // Term registered in `id_to_term` but no postings landed
            // for it — only possible if the in-RAM flush ran with an
            // empty postings vec. Defensive: skip without emitting.
            continue;
        }
        let term_positions = match blob_mmaps.is_some() {
            true => {
                // Assemble the term's position runs in merged (doc)
                // order: slice each posting's run out of the blob,
                // its end found by skipping `tf` varints.
                term_run.clear();
                let blob = blob_slices[p];
                for (i, &(_, tf)) in group.iter().enumerate() {
                    let start = group_pos[i] as usize;
                    let mut at = start;
                    skip_run(blob, &mut at, tf).expect("builder-encoded blob runs are well-formed");
                    term_run.extend_from_slice(&blob[start..at]);
                }
                Some(TermRuns::Encoded(term_run.as_slice()))
            }
            false => None,
        };
        let term_bytes: &str = id_to_term[term_id as usize];
        encode_and_emit_term(
            term_bytes,
            &group,
            term_positions,
            enc,
            finish_profile,
            term_scratch,
            sink,
        )?;
        n_emitted += 1;
    }
    // Sanity: every partition should now be fully drained. If not, we
    // lost or mis-ordered records somewhere upstream.
    debug_assert!(
        cursors
            .iter()
            .zip(sorted_slices.iter())
            .all(|(c, s)| *c == s.len()),
        "lex-order partition traversal did not drain all records; \
         partition assignment or sort invariant violated"
    );
    Ok(n_emitted)
}

/// The documents of `lane` whose rows fall in `rows`, as a range of its
/// document indexes: its stretches ascend, so two binary searches find it.
fn lane_doc_range(lane: &FrozenLane, rows: &Range<u32>) -> (usize, usize) {
    (
        docs_before_row(&lane.stretches, rows.start),
        docs_before_row(&lane.stretches, rows.end),
    )
}

/// One lane's share of the term being emitted: postings `pos..hi` of
/// list `id` still to take, and the lane's row stretch the next one is in.
struct Holder<'a> {
    lane: usize,
    col: &'a FrozenColumn,
    id: u32,
    pos: usize,
    hi: usize,
    /// Where posting `pos`'s position run begins, in a positional column.
    read: usize,
    cursor: usize,
}

impl Holder<'_> {
    /// Advance `read` to posting `to`'s run: a short way is walked from the
    /// current run, a long way is looked up from the recorded run ends.
    fn read_to(&mut self, to: usize) {
        if to - self.pos <= RUN_END_STRIDE {
            skip_runs(self.col, self.id, self.pos..to, &mut self.read);
        } else {
            self.read = run_offset(self.col, self.id, to);
        }
    }
}

/// Advance `at`, a byte offset into list `id`'s position runs, past the
/// runs of postings `span`.
fn skip_runs(col: &FrozenColumn, id: u32, span: Range<usize>, at: &mut usize) {
    let runs = &col.pos_runs[id as usize];
    for &(_, tf) in &col.postings[id as usize][span] {
        skip_run(runs, at, tf).expect("builder-encoded run");
    }
}

/// Where posting `p` of list `id`'s position runs begins: the nearest
/// recorded end before it, then the runs between skipped. A stretch's
/// bytes are located this way instead of by walking its postings, so a
/// long stretch of a common term costs a lookup, not a scan.
fn run_offset(col: &FrozenColumn, id: u32, p: usize) -> usize {
    if p == col.postings[id as usize].len() {
        return col.pos_runs[id as usize].len();
    }
    let block = p / RUN_END_STRIDE;
    let mut at = match block {
        0 => 0,
        _ => col.run_ends[id as usize][block - 1] as usize,
    };
    skip_runs(col, id, block * RUN_END_STRIDE..p, &mut at);
    at
}

/// Emit one column from a lane view, merged term by term over the view's
/// rows, walking the commit's merged dictionary order: a term held by one
/// lane is encoded from the stretch of that lane's list that falls in the
/// rows, its documents mapped to their rows; a term several lanes hold
/// has their stretches merged by row, which is a merge of ascending runs
/// since every lane's documents are in row order, with each posting's
/// positions brought along. A list whose first and last documents miss
/// the rows is passed over at once, one within them is taken whole, and
/// one spanning several pieces is cut where [`freeze_lanes`] recorded;
/// its runs come by byte range.
fn emit_lanes_column<W: Write>(
    view: &LaneView,
    column: usize,
    enc: &TermEncoding<'_>,
    finish_profile: &mut FinishProfile,
    term_scratch: &mut TermScratch,
    sink: &mut TermSink<'_, W>,
) -> Result<usize, BuildError> {
    let col_positions = sink.col_positions;
    let rows = view.rows();
    let first_row = rows.start;
    let lanes = &view.lanes.lanes;
    // Each lane's documents in the view, as document indexes.
    let docs: Vec<Range<usize>> = lanes
        .iter()
        .map(|lane| {
            let (lo, hi) = lane_doc_range(lane, rows);
            lo..hi
        })
        .collect();
    let order = &view.lanes.order[column];

    let mut pairs: Vec<(u32, u32)> = Vec::new();
    // The runs of a term several lanes hold, in merged posting order: a
    // byte range of one lane's runs per stretch taken, copied as they are.
    let mut merged_runs: Vec<u8> = Vec::new();
    // The lanes holding the current term: the list, the postings of it in
    // the view, how far the merge has read, and the lane's row stretch its
    // next document is in.
    let mut holders: Vec<Holder> = Vec::new();
    let mut n_emitted = 0usize;
    let mut at = 0usize;
    while at < order.len() {
        // The entries of one term: from a first-flagged entry to the next.
        let group_end = order[at + 1..]
            .iter()
            .position(|entry| entry.first)
            .map_or(order.len(), |n| at + 1 + n);
        holders.clear();
        for entry in &order[at..group_end] {
            let lane = entry.lane as usize;
            let range = &docs[lane];
            let (first, last) = (entry.first_doc as usize, entry.last_doc as usize);
            if first >= range.end || last < range.start {
                continue;
            }
            let col = &lanes[lane].columns[column];
            // A list within the piece is taken whole; one that spans pieces
            // has its cuts recorded.
            let (lo, hi) = if first >= range.start && last < range.end {
                (0, col.postings[entry.id as usize].len())
            } else {
                let at = col.cut_at[entry.id as usize] as usize + view.piece;
                (col.cuts[at] as usize, col.cuts[at + 1] as usize)
            };
            if lo == hi {
                continue;
            }
            holders.push(Holder {
                lane,
                col,
                id: entry.id,
                pos: lo,
                hi,
                read: match col_positions {
                    true => run_offset(col, entry.id, lo),
                    false => 0,
                },
                cursor: 0,
            });
        }
        at = group_end;
        if holders.is_empty() {
            continue;
        }
        let term = {
            let entry = order[at - 1];
            lanes[entry.lane as usize].columns[column].term(entry.id)
        };
        pairs.clear();
        let runs: Option<TermRuns<'_>> = if holders.len() == 1 {
            let Holder {
                lane,
                col,
                id,
                pos: lo,
                hi,
                ..
            } = holders[0];
            let stretches = &lanes[lane].stretches;
            let mut cursor = 0usize;
            pairs.extend(
                col.postings[id as usize][lo..hi]
                    .iter()
                    .map(|&(doc, tf)| (row_of(stretches, &mut cursor, doc) - first_row, tf)),
            );
            // Only a positional column has runs to slice.
            let from = holders[0].read;
            col_positions.then(|| {
                TermRuns::Encoded(&col.pos_runs[id as usize][from..run_offset(col, id, hi)])
            })
        } else {
            // A merge of ascending runs. A lane's documents of a batch are
            // one stretch of rows, so the holder with the lowest next row
            // is taken in a stretch, up to the next row any other holder
            // has, before the holders are compared again; its runs for the
            // stretch are one byte range, located by the recorded run
            // ends and copied without decoding.
            merged_runs.clear();
            loop {
                let mut lowest: Option<(usize, u32)> = None;
                let mut bound = u32::MAX;
                for (h, holder) in holders.iter_mut().enumerate() {
                    if holder.pos < holder.hi {
                        let (doc, _) = holder.col.postings[holder.id as usize][holder.pos];
                        let row = row_of(&lanes[holder.lane].stretches, &mut holder.cursor, doc);
                        match lowest {
                            Some((_, r)) if row >= r => bound = bound.min(row),
                            Some((_, r)) => {
                                bound = r;
                                lowest = Some((h, row));
                            }
                            None => lowest = Some((h, row)),
                        }
                    }
                }
                let Some((h, _)) = lowest else {
                    break;
                };
                let holder = &mut holders[h];
                let stretches = &lanes[holder.lane].stretches;
                let list = &holder.col.postings[holder.id as usize];
                let mut to = holder.pos;
                while to < holder.hi {
                    let (doc, tf) = list[to];
                    let row = row_of(stretches, &mut holder.cursor, doc);
                    if row >= bound {
                        break;
                    }
                    pairs.push((row - first_row, tf));
                    to += 1;
                }
                if col_positions {
                    let from = holder.read;
                    holder.read_to(to);
                    merged_runs.extend_from_slice(
                        &holder.col.pos_runs[holder.id as usize][from..holder.read],
                    );
                }
                holder.pos = to;
            }
            col_positions.then_some(TermRuns::Encoded(&merged_runs))
        };
        encode_and_emit_term(term, &pairs, runs, enc, finish_profile, term_scratch, sink)?;
        n_emitted += 1;
    }
    Ok(n_emitted)
}

/// Emit one in-RAM column's terms in dictionary order: the ids are
/// sorted by their term bytes once, then each id's postings and runs
/// are encoded straight out of the store — the lists are already in
/// ascending doc order, so no per-list or partition sort runs. Each
/// list is taken out of the store as it is emitted, so the column's
/// resident bytes shrink as the emit proceeds. Returns the number of
/// terms written.
fn emit_ram_column<W: Write>(
    vocab: Vocab,
    ram: RamStore,
    enc: &TermEncoding<'_>,
    finish_profile: &mut FinishProfile,
    term_scratch: &mut TermScratch,
    sink: &mut TermSink<'_, W>,
) -> Result<usize, BuildError> {
    let col_positions = sink.col_positions;
    let Vocab {
        term_to_id,
        id_to_term,
        term_arena,
        ..
    } = vocab;
    // The forward map is finished with; only the reverse map is read
    // below. Its keys borrow from `term_arena`, which outlives every
    // read of `id_to_term` here (dropped explicitly at the end).
    drop(term_to_id);
    let RamStore { mut lists, .. } = ram;
    let (_, term_id_in_lex_order) = build_lex_rank(&id_to_term);
    let mut n_emitted = 0usize;
    for &term_id in &term_id_in_lex_order {
        let idx = term_id as usize;
        let TermLists {
            postings: pairs,
            runs,
            ..
        } = mem::take(&mut lists[idx]);
        if pairs.is_empty() {
            // Interned but never given a posting — only a prebuilt feed
            // that failed mid-push could leave this; nothing to write.
            continue;
        }
        let runs = col_positions.then_some(TermRuns::Encoded(runs.as_slice()));
        encode_and_emit_term(
            id_to_term[idx],
            &pairs,
            runs,
            enc,
            finish_profile,
            term_scratch,
            sink,
        )?;
        n_emitted += 1;
    }
    drop(id_to_term);
    drop(term_arena);
    Ok(n_emitted)
}

/// Where a finish writes one column's terms: the postings region with
/// its running CRC and length, the positions region, the dictionary
/// (built in RAM or streamed to scratch), the term-index sink, and the key
/// scratch the writes share. Built per column, in place of a dozen
/// arguments on every emit; the finish's profile travels beside it.
struct TermSink<'a, W> {
    col_name: &'a str,
    col_name_bytes: &'a [u8],
    /// Whether the column keeps positions: only then does a term carry
    /// runs into the positions region.
    col_positions: bool,
    key_buf: &'a mut Vec<u8>,
    postings_writer: &'a mut W,
    postings_crc_acc: &'a mut u32,
    postings_len: &'a mut u64,
    positions_sink: &'a mut PositionsSink,
    dict: DictSink<'a>,
    head_sink: &'a mut Option<Box<dyn TermHeadSink>>,
}

/// The dictionary a finish adds its terms to.
enum DictSink<'a> {
    /// Built in RAM, for a build no column of which spilled.
    InRam(&'a mut TermDictBuilder),
    /// Streamed to a scratch file in term order.
    Streaming(&'a mut TermBlockWriter<BufWriter<File>>),
}

/// Encode one term's posting list and emit the resulting dictionary entry
/// into the finish's sink. `runs` is `Some` exactly for a positional
/// column.
///
/// Owns the per-term encoding policy (df=1 inline value, df≥2 PFOR
/// blocks via `encode_posting_group`): [`encode_term`] builds the term's
/// bytes and [`write_term`] places them.
fn encode_and_emit_term<W: Write>(
    term: &str,
    pairs: &[(u32, u32)],
    runs: Option<TermRuns<'_>>,
    enc: &TermEncoding<'_>,
    profile: &mut FinishProfile,
    scratch: &mut TermScratch,
    sink: &mut TermSink<'_, W>,
) -> Result<(), BuildError> {
    let encoded = encode_term_timed(pairs, runs, enc, profile, scratch)?;
    write_term(
        term,
        encoded,
        &mut scratch.term_buf,
        &scratch.pos_out,
        &scratch.head,
        profile,
        sink,
    )
}

/// [`encode_term`], counted and timed in `profile`.
fn encode_term_timed(
    pairs: &[(u32, u32)],
    term_positions: Option<TermRuns<'_>>,
    enc: &TermEncoding<'_>,
    profile: &mut FinishProfile,
    scratch: &mut TermScratch,
) -> Result<EncodedTerm, BuildError> {
    let encode_start = profile.enabled.then(Instant::now);
    profile.encode_calls += 1;
    let encoded = encode_term(pairs, term_positions, enc, profile, scratch)?;
    if let Some(start) = encode_start {
        profile.encode_total += start.elapsed();
    }
    Ok(encoded)
}

/// What a column's terms are encoded against, besides the terms.
#[derive(Clone, Copy)]
struct TermEncoding<'a> {
    doc_lengths: &'a [u32],
    avgdl: f32,
    params: bm25::Bm25Params,
    n_scored_docs: u32,
    /// Whether a head is gathered for each term, and how deep.
    heads: bool,
    head_depth: usize,
    /// The row each blob document id stands for, when they differ.
    rows: Option<&'a [u32]>,
}

/// A term encoded on a worker thread, its bytes taken out of that
/// worker's scratch so the scratch can take the next term.
struct MergedTerm {
    encoded: EncodedTerm,
    body: Vec<u8>,
    positions: Vec<u8>,
    head: HeadOut,
}

/// What [`encode_term`] made of one term. A body sits in the scratch's
/// `term_buf`; a long term's positions sit in its `pos_out`.
#[derive(Clone, Copy)]
enum EncodedTerm {
    /// The whole posting fits the dictionary value; no bytes.
    Inline(DictEntry),
    /// A short-form body, positions inside it.
    Short,
    /// A long-form body whose header still needs its two offsets.
    Long,
}

/// Build one term's bytes. Depends on nothing outside the term, so terms
/// can be encoded in any order; [`write_term`] then places them in term
/// order. `term_positions` is `Some` exactly for a positional column.
fn encode_term(
    pairs: &[(u32, u32)],
    term_positions: Option<TermRuns<'_>>,
    enc: &TermEncoding<'_>,
    profile: &mut FinishProfile,
    scratch: &mut TermScratch,
) -> Result<EncodedTerm, BuildError> {
    let TermEncoding {
        doc_lengths: col_doc_lengths,
        avgdl,
        params,
        n_scored_docs,
        ..
    } = *enc;
    scratch.term_buf.clear();
    scratch.pos_out.clear();

    debug_assert!(
        pairs.windows(2).all(|w| w[0].0 < w[1].0),
        "posting list not sorted by doc_id"
    );

    let df = pairs.len() as u64;
    // Every posting's score, as a reader scores it: the heads rank by it
    // and a long list's blocks take their maxima from it. A short list
    // without a head to gather is not scored.
    let idf_t = bm25::idf(n_scored_docs as u64, df);
    let score_all = |scratch: &mut TermScratch| {
        scratch.scores.clear();
        scratch.scores.extend(pairs.iter().map(|&(doc, tf)| {
            bm25::score(
                idf_t,
                tf,
                bm25::stored_len(col_doc_lengths[doc as usize]),
                avgdl,
                params,
            )
        }));
    };
    if enc.heads {
        score_all(scratch);
        term_head(
            pairs,
            &scratch.scores,
            enc,
            &mut scratch.head_heap,
            &mut scratch.head_kept,
            &mut scratch.head,
        );
    }

    // A df=1 posting inlines into the dictionary entry when the WHOLE
    // posting fits: a positionless column always does (doc_id + tf);
    // a positional column only when tf == 1 and the single position
    // fits the 30-bit slot — the position rides where tf normally
    // lives, tf implied 1 (one position is exactly what a phrase
    // check needs). Otherwise even a df=1 term takes the PFOR form so
    // its positions land in the region.
    let inline_value: Option<DictEntry> = if df == 1 {
        let (doc_id, tf) = pairs[0];
        match &term_positions {
            None => Some(DictEntry::Inline { doc_id, tf }),
            Some(runs) if tf == 1 => {
                let pos = runs.first_value();
                (pos <= INLINE_TF_MAX).then_some(DictEntry::Inline { doc_id, tf: pos })
            }
            Some(_) => None,
        }
    } else {
        None
    };

    let encoded = if let Some(v) = inline_value {
        profile.encode_df1 += 1;
        EncodedTerm::Inline(v)
    } else if pairs.len() <= SHORT_MAX_DF {
        // Single-block term: the short form (`fts::short`) — no header,
        // skip entry, coarse slot or block header, and no lane padding. Its positions sit inline in the body.
        profile.encode_short += 1;
        let positions = match term_positions.as_ref() {
            Some(runs) => {
                // The whole term is one position group, inline in the body:
                // its run values, packed or LEB128, whichever is smaller.
                let vals = &mut scratch.pos_vals;
                let out = &mut scratch.pos_out;
                vals.clear();
                out.clear();
                let mut at = 0usize;
                for (i, &(_, tf)) in pairs.iter().enumerate() {
                    runs.push_run(i, tf, &mut at, vals);
                }
                debug_assert!(
                    runs.encoded().is_none_or(|bytes| at == bytes.len()),
                    "runs must cover exactly the pairs"
                );
                let tfs = &mut scratch.tfs;
                tfs.clear();
                tfs.extend(pairs.iter().map(|&(_, tf)| tf));
                encode_group(out, tfs, vals, true, &mut scratch.pack);
                Some(out.as_slice())
            }
            None => None,
        };
        encode_short(&mut scratch.term_buf, pairs, positions);
        // The body carries its positions; nothing goes to the region.
        scratch.pos_out.clear();
        EncodedTerm::Short
    } else {
        profile.encode_pfor += 1;
        if !enc.heads {
            score_all(scratch);
        }
        // Reuse `scratch.encoded_blocks` / `scratch.min_dl_per_block`
        // across every dense term in the column (one allocation amortised
        // over ~5K terms / ~1M blocks at 1M docs, vs `Vec::new` per term).
        let encoded_blocks = &mut scratch.encoded_blocks;
        let block_maxes = &mut scratch.block_maxes;
        encoded_blocks.clear();
        block_maxes.clear();
        let block_len = block_len_for(pairs.len(), n_scored_docs as usize);
        let block_build_start = profile.enabled.then(Instant::now);
        // Build each block by moving the reusable `doc_ids` / `tfs`
        // buffers into a `Block` (Vec move = pointer swap, no copy),
        // encoding, then moving them back so the next chunk reuses the
        // same heap allocation. `encode_block` only borrows, but the
        // public signature takes `&Block` whose fields are owned Vecs,
        // hence the take/put dance. The take/put preserves capacity at
        // `BLOCK_LEN`, so the steady-state per-chunk cost is
        // `clear` + `extend_from_slice` of a block's u32s with no realloc.
        let mut block_doc_ids = mem::take(&mut scratch.doc_ids);
        let mut block_tfs = mem::take(&mut scratch.tfs);
        if block_doc_ids.capacity() < BLOCK_LEN {
            block_doc_ids.reserve(BLOCK_LEN - block_doc_ids.capacity());
        }
        if block_tfs.capacity() < BLOCK_LEN {
            block_tfs.reserve(BLOCK_LEN - block_tfs.capacity());
        }
        for (at, chunk) in pairs.chunks(block_len).enumerate() {
            block_doc_ids.clear();
            block_tfs.clear();
            block_doc_ids.extend(chunk.iter().map(|&(d, _)| d));
            block_tfs.extend(chunk.iter().map(|&(_, t)| t));
            // Tight per-block BM25 upper bound: the max per-doc contribution
            // over the block. score() rises with tf and falls with dl, so the
            // looser `score(max_tf, min_dl)` over-estimates when the
            // highest-tf doc isn't also the shortest. Computing the real max
            // here (build-time, off the query path) tightens the skip-table
            // bound so the reader skips more blocks.
            //
            // Score against the *quantized* length the reader scores with —
            // the resident norm table stores each length as a one-byte
            // bucket and scores from its dequantized representative, which
            // truncates the length downward. A shorter length means a
            // smaller norm and thus a *higher* BM25 score, so bounding
            // against the exact length would leave the stored max below the
            // query-time score of that same doc and let the block-max skip
            // drop a qualifying document.
            let block_max = scratch.scores[at * block_len..at * block_len + chunk.len()]
                .iter()
                .fold(0.0f32, |m, &s| m.max(s));
            block_maxes.push(block_max);
            let block = Block {
                doc_ids: mem::take(&mut block_doc_ids),
                tfs: mem::take(&mut block_tfs),
            };
            let prev_last_doc_id = encoded_blocks.last().map(|b: &EncodedBlock| b.last_doc_id);
            encoded_blocks.push(encode_block_with(
                &block,
                prev_last_doc_id,
                pairs.len() <= PATCHED_MAX_DF,
                dense_bitset_bias(pairs.len()),
                &mut scratch.pack,
            ));
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
        let skip_table_size = encoded_blocks.len() * skip_entry::bytes(term_positions.is_some());
        let blocks_total_size: usize = encoded_blocks.iter().map(|b| b.bytes.len()).sum();
        let term_meta_size = match term_positions {
            Some(_) => TERM_META_POSITIONAL_SIZE,
            None => TERM_META_SIZE,
        };
        // Coarse block-max table at the tail of the term region: one slot
        // per `COARSE_BLOCK_MAX_SPAN` blocks bounding the whole span, giving
        // the ranked walk a second skip level. Appended last so no existing
        // block offset moves.
        let num_coarse = (num_blocks as usize).div_ceil(format::fts::COARSE_BLOCK_MAX_SPAN);
        let coarse_table_size = num_coarse * coarse_slot::BYTES;
        let postings_length =
            (term_meta_size + skip_table_size + blocks_total_size + coarse_table_size) as u64;

        // Regroup the runs block by block, recording where each block's
        // group starts (the skip table's per-block positions offset). Every
        // long-form group is packed, so the phrase decode reads it whole and
        // indexes it by the block's tf prefix sums.
        let pos_block_offsets = &mut scratch.pos_block_offsets;
        pos_block_offsets.clear();
        let pos_out = &mut scratch.pos_out;
        if let Some(runs) = &term_positions {
            let mut at: usize = 0;
            let vals = &mut scratch.pos_vals;
            for (c, chunk) in pairs.chunks(block_len).enumerate() {
                pos_block_offsets.push(pos_out.len() as u32);
                vals.clear();
                for (k, &(_, tf)) in chunk.iter().enumerate() {
                    runs.push_run(c * block_len + k, tf, &mut at, vals);
                }
                let tfs = &mut scratch.tfs;
                tfs.clear();
                tfs.extend(chunk.iter().map(|&(_, tf)| tf));
                encode_group(pos_out, tfs, vals, false, &mut scratch.pack);
            }
            debug_assert!(
                runs.encoded().is_none_or(|bytes| at == bytes.len()),
                "runs must cover exactly the pairs"
            );
        }

        debug_assert!(df <= u32::MAX as u64, "df overflows u32");
        // The header's postings_length field is a stored u32; a
        // silent truncation on cast would write a file whose header
        // understates its own postings length.
        if postings_length > u32::MAX as u64 {
            return Err(BuildError::Io(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "single term's postings are {postings_length} bytes, past the 4 GiB \
                     limit of the metadata header's postings_length field"
                ),
            )));
        }

        // `term_buf` is empty here, so this reserves exactly the term. It
        // reuses its allocation across terms, except where the compaction
        // merge moves a long term's buffer out.
        let term_buf = &mut scratch.term_buf;
        term_buf.reserve(postings_length as usize);
        let meta_write_start = profile.enabled.then(Instant::now);
        term_buf.extend_from_slice(&(df as u32).to_le_bytes());
        // The two offsets are placeholders until `write_term` fills them.
        debug_assert_eq!(term_buf.len(), HEADER_POSTINGS_OFFSET.start);
        term_buf.extend_from_slice(&0u64.to_le_bytes());
        term_buf.extend_from_slice(&(postings_length as u32).to_le_bytes());
        term_buf.extend_from_slice(&num_blocks.to_le_bytes());
        if term_positions.is_some() {
            debug_assert_eq!(term_buf.len(), HEADER_POSITIONS_OFFSET.start);
            term_buf.extend_from_slice(&0u64.to_le_bytes());
            term_buf.extend_from_slice(&(pos_out.len() as u32).to_le_bytes());
        }
        debug_assert_eq!(term_buf.len(), term_meta_size);
        if let Some(start) = meta_write_start {
            profile.encode_meta_write += start.elapsed();
        }

        // Blocks follow the meta and the skip table, so their offsets start
        // past both.
        let mut block_offset: u32 = (term_meta_size + skip_table_size) as u32;
        // Coarse slots, filled alongside the per-block skip entries and
        // appended after the blocks below. Each holds the span's max of
        // the per-block `f32` maxes, stored as `f32` bits — a true upper
        // bound over the span — and the byte offset of the span's first
        // block.
        let mut coarse_slots: Vec<u8> = Vec::with_capacity(coarse_table_size);
        let mut span_max: f32 = 0.0;
        let mut span_start: u32 = block_offset;
        let coarse_span = format::fts::COARSE_BLOCK_MAX_SPAN;
        let skip_write_start = profile.enabled.then(Instant::now);
        for (i, blk) in encoded_blocks.iter().enumerate() {
            let max_bm25 = block_maxes[i];
            // The block-max slot stores the exact `f32` bits: it equals the
            // reader's per-doc score for the block's max doc (same
            // quantized-length scoring), so it is an exact upper bound.
            if i.is_multiple_of(coarse_span) {
                span_start = block_offset;
            }
            let len = u16::try_from(blk.bytes.len()).expect("a block is under 64 KiB");
            term_buf.extend_from_slice(&blk.last_doc_id.to_le_bytes());
            term_buf.extend_from_slice(&len.to_le_bytes());
            term_buf.extend_from_slice(&max_bm25.to_bits().to_le_bytes());
            if term_positions.is_some() {
                term_buf.extend_from_slice(&pos_block_offsets[i].to_le_bytes());
            }
            block_offset += blk.bytes.len() as u32;

            span_max = span_max.max(max_bm25);
            if (i + 1).is_multiple_of(coarse_span) || i + 1 == encoded_blocks.len() {
                coarse_slots.extend_from_slice(&span_max.to_bits().to_le_bytes());
                coarse_slots.extend_from_slice(&span_start.to_le_bytes());
                span_max = 0.0;
            }
        }
        debug_assert_eq!(coarse_slots.len(), coarse_table_size);
        if let Some(start) = skip_write_start {
            profile.encode_skip_write += start.elapsed();
        }

        let block_write_start = profile.enabled.then(Instant::now);
        for blk in encoded_blocks.iter() {
            term_buf.extend_from_slice(&blk.bytes);
        }
        // Coarse block-max table: the term region's tail.
        term_buf.extend_from_slice(&coarse_slots);
        debug_assert_eq!(term_buf.len(), postings_length as usize);
        if let Some(start) = block_write_start {
            profile.encode_block_write += start.elapsed();
        }
        EncodedTerm::Long
    };
    Ok(encoded)
}

/// Place an encoded term: fill in its header's offsets, append its bytes
/// to the postings and, for a long positional term, the positions region,
/// and add its dictionary entry. Terms must arrive in term order.
fn write_term<W: Write>(
    term: &str,
    encoded: EncodedTerm,
    body: &mut [u8],
    positions: &[u8],
    head: &HeadOut,
    profile: &mut FinishProfile,
    sink: &mut TermSink<'_, W>,
) -> Result<(), BuildError> {
    let write_start = profile.enabled.then(Instant::now);
    sink.key_buf.clear();
    sink.key_buf.extend_from_slice(sink.col_name_bytes);
    sink.key_buf.push(KEY_SEPARATOR);
    sink.key_buf.extend_from_slice(term.as_bytes());

    let dict_entry = match encoded {
        EncodedTerm::Inline(value) => value,
        EncodedTerm::Short | EncodedTerm::Long => {
            let metadata_offset = *sink.postings_len;
            let long = matches!(encoded, EncodedTerm::Long);
            if long {
                body[HEADER_POSTINGS_OFFSET].copy_from_slice(&metadata_offset.to_le_bytes());
                if sink.col_positions {
                    body[HEADER_POSITIONS_OFFSET]
                        .copy_from_slice(&sink.positions_sink.len.to_le_bytes());
                }
            }
            let block_write_start = (profile.enabled && long).then(Instant::now);
            write_counted(
                sink.postings_writer,
                sink.postings_crc_acc,
                sink.postings_len,
                body,
            )?;
            if long && sink.col_positions {
                sink.positions_sink.write(positions)?;
            }
            if let Some(start) = block_write_start {
                profile.encode_block_write += start.elapsed();
            }
            DictEntry::Pfor {
                metadata_offset,
                postings_length: body.len() as u32,
                short: !long,
            }
        }
    };

    if let Some(head_sink) = sink.head_sink.as_mut() {
        head_sink.term(
            sink.col_name,
            term,
            &TermHeadFacts {
                df: head.df,
                bound: head.bound,
                entry: dict_entry,
                head: &head.facts,
                tail: head.tail,
                avgdl: head.avgdl,
                params: head.params,
            },
        )?;
    }
    let dict_insert_start = profile.enabled.then(Instant::now);
    match &mut sink.dict {
        DictSink::InRam(dict) => dict.insert(sink.key_buf, dict_entry),
        DictSink::Streaming(dict) => dict.insert_sorted(sink.key_buf, dict_entry)?,
    }
    if let Some(start) = dict_insert_start {
        profile.dict_insert += start.elapsed();
    }
    if let Some(start) = write_start {
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
fn sort_partition_to_file<const N: usize>(
    in_path: &Path,
    out_path: &Path,
    max_partition_bytes: u64,
    scratch_dir: &Path,
    partition_label: &str,
    lex_rank: &[u32],
) -> Result<(), BuildError> {
    let mut iter = open_partition_sorted::<N>(
        in_path,
        max_partition_bytes,
        scratch_dir,
        partition_label,
        lex_rank,
    )?;
    let mut w = BufWriter::with_capacity(PARTITION_BUF_SIZE, File::create(out_path)?);
    let mut batch: Vec<[u32; N]> = Vec::with_capacity(SORT_OUTPUT_BATCH_TRIPLES);
    while let Some(triple) = iter.next_with(lex_rank) {
        let t = triple?;
        batch.push(t);
        if batch.len() == SORT_OUTPUT_BATCH_TRIPLES {
            #[cfg(target_endian = "little")]
            {
                w.write_all(bytemuck::cast_slice::<[u32; N], u8>(&batch))?;
            }
            #[cfg(not(target_endian = "little"))]
            for t in &batch {
                write_record(&mut w, t)?;
            }
            batch.clear();
        }
    }
    if !batch.is_empty() {
        #[cfg(target_endian = "little")]
        {
            w.write_all(bytemuck::cast_slice::<[u32; N], u8>(&batch))?;
        }
        #[cfg(not(target_endian = "little"))]
        for t in &batch {
            write_record(&mut w, t)?;
        }
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use bytes::Bytes;
    use rayon::ThreadPoolBuilder;
    use tempfile::tempdir;

    use super::*;
    use crate::{
        superfile::fts::{
            posting::SUB_BLOCK_LEN,
            reader::{BoolMode, FtsReader},
        },
        utils::varint::read_head,
    };

    /// The block length is the term's density. In a column of enough docs
    /// that one in a thousand is past a sub-block, a term just over that
    /// share is written wide (its postings in one block) and one just
    /// under it narrow (two blocks), while a term in every doc is wide
    /// throughout; the three mix in one column, and a query over the two
    /// around the floor finds every doc.
    #[tokio::test]
    async fn block_length_follows_the_terms_density() {
        /// Docs in the column: one in a thousand is a sub-block and two.
        const N_DOCS: u32 = ((SUB_BLOCK_LEN + 2) as f64 / WIDE_BLOCK_MIN_DENSITY) as u32;
        /// Docs holding a term at the floor; the two terms sit a doc
        /// either side of it.
        const AT_FLOOR: u32 = (N_DOCS as f64 * WIDE_BLOCK_MIN_DENSITY) as u32;
        let stride = N_DOCS / AT_FLOOR;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let text = match (i % stride == 0, i) {
                (true, 0) | (false, 1) => "every wide",
                (true, _) => "every wide narrow",
                (false, _) => "every",
            };
            b.add_doc(0, i, text).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let layout = async |term| {
            r.term_layout("body", term)
                .await
                .expect("layout")
                .expect("present")
        };
        let (wide, narrow, every) = (
            layout("wide").await,
            layout("narrow").await,
            layout("every").await,
        );
        assert_eq!(
            (wide.df, narrow.df),
            (u64::from(AT_FLOOR) + 1, u64::from(AT_FLOOR) - 1)
        );
        assert_eq!(wide.num_blocks, 1, "{wide:?}");
        assert_eq!(narrow.num_blocks, 2, "{narrow:?}");
        assert_eq!(
            every.num_blocks,
            (N_DOCS as usize).div_ceil(BLOCK_LEN),
            "{every:?}"
        );
        for (mode, want) in [(BoolMode::Or, wide.df), (BoolMode::And, narrow.df)] {
            let hits = r
                .search("body", &["wide", "narrow"], N_DOCS as usize, mode)
                .await
                .expect("search");
            assert_eq!(hits.len() as u64, want, "{mode:?}");
        }
    }

    /// The radix path (n >= `RADIX_SORT_MIN_TRIPLES`) must deliver
    /// `(lex_rank, doc_id)` order even when a term's docs arrive out of
    /// order — the compaction carry paths feed postings remapped through
    /// a packed row order, unlike the always-ascending ingest path. A
    /// term-only stable scatter once preserved the unsorted arrival
    /// order, and `encode_block` then indexed a bitset block out of
    /// bounds on the merged output.
    #[test]
    fn radix_sort_orders_docs_within_term_for_unsorted_feeds() {
        // 4 terms × enough triples to clear the radix threshold, docs
        // deliberately fed in descending order per term.
        let n_terms = 4u32;
        let per_term = RADIX_SORT_MIN_TRIPLES as u32;
        // Identity lex ranks (term ids already lexicographic).
        let lex_rank: Vec<u32> = (0..n_terms).collect();
        let mut triples: Vec<[u32; 3]> = Vec::new();
        for doc in (0..per_term).rev() {
            for term in 0..n_terms {
                triples.push([term, doc, 1]);
            }
        }
        assert!(triples.len() >= RADIX_SORT_MIN_TRIPLES);
        radix_sort_records_by_lex_rank(&mut triples, &lex_rank);
        let sorted = triples.windows(2).all(|w| {
            let (a, b) = (&w[0], &w[1]);
            let ka = (lex_rank[triple_term_id(a) as usize], triple_doc_id(a));
            let kb = (lex_rank[triple_term_id(b) as usize], triple_doc_id(b));
            ka < kb
        });
        assert!(sorted, "triples must come out in (lex_rank, doc_id) order");
    }

    /// Lanes that tokenized interleaved shares of a corpus, frozen and
    /// merged at the finish over a range of rows, write the blob one
    /// builder fed those rows in order writes, on a positional and a plain
    /// column: over the whole corpus, and cut into two superfiles' worth
    /// of rows, with terms the lanes share, terms only one holds, a term
    /// of one document each, and a term that straddles the cut.
    fn two_column_builder() -> FtsBuilder {
        let mut b = FtsBuilder::new();
        b.register_column("pos".into(), true).expect("register");
        b.register_column("flat".into(), false).expect("register");
        b
    }

    /// The blob one builder writes for rows `first..past` of `text_of`,
    /// both columns.
    fn single_blob(first: u32, past: u32, text_of: &dyn Fn(u32) -> String) -> Vec<u8> {
        let mut b = two_column_builder();
        for i in first..past {
            b.add_doc(0, i - first, &text_of(i)).expect("add pos");
            b.add_doc(1, i - first, &text_of(i)).expect("add flat");
        }
        b.finish().expect("finish single")
    }

    /// The blob piece `piece` of `pieces` gets from `lanes` lanes fed rows
    /// `0..n_docs` of `text_of`: row `i` goes to lane `(i / slice) % lanes`,
    /// so each lane holds runs of `slice` rows interleaved with the other
    /// lanes', in row order, and consecutive rows of a lane fold into one
    /// stretch.
    fn lanes_blob(
        n_docs: u32,
        lanes: u32,
        slice: u32,
        text_of: &dyn Fn(u32) -> String,
        pieces: Vec<Range<u32>>,
        piece: usize,
    ) -> Vec<u8> {
        let stretches_of = |rows: &[u32]| -> Vec<Stretch> {
            let mut out: Vec<Stretch> = Vec::new();
            for (doc, &row) in rows.iter().enumerate() {
                match out.last_mut() {
                    Some(s) if s.row + s.len == row => s.len += 1,
                    _ => out.push(Stretch {
                        doc: doc as u32,
                        row,
                        len: 1,
                    }),
                }
            }
            out
        };
        let mut fed: Vec<(FtsBuilder, Vec<u32>)> = (0..lanes)
            .map(|_| {
                let mut b = two_column_builder();
                b.set_lane_mode();
                (b, Vec::new())
            })
            .collect();
        for i in 0..n_docs {
            let (lane, rows) = &mut fed[((i / slice) % lanes) as usize];
            let doc = rows.len() as u32;
            lane.add_doc(0, doc, &text_of(i)).expect("add pos");
            lane.add_doc(1, doc, &text_of(i)).expect("add flat");
            rows.push(i);
        }
        let frozen: Vec<FrozenLane> = fed
            .into_iter()
            .map(|(lane, rows)| lane.freeze(stretches_of(&rows)).expect("freeze"))
            .collect();
        let pool = ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("pool");
        let frozen = Arc::new(freeze_lanes(&pool, frozen, pieces).expect("merge the lanes"));
        let mut b = two_column_builder();
        b.set_lane_view(LaneView {
            lanes: frozen,
            piece,
        });
        b.finish().expect("finish view")
    }

    /// A term held by several lanes but missing from some lanes' batches:
    /// the merge takes each lane's stretch only up to the next row any
    /// other lane holds, not up to that lane's own next document. Here
    /// `t` is in rows 0 and 21 (lane 0), 7 (lane 1) and 35 (lane 2), so
    /// lane 0's two postings must not come out together.
    #[test]
    fn a_term_some_lanes_miss_in_a_batch_merges_in_row_order() {
        const N_DOCS: u32 = 42;
        let text_of = |i: u32| match i {
            0 | 7 | 21 | 35 => format!("u{i} t t"),
            _ => format!("u{i}"),
        };
        assert_eq!(
            lanes_blob(N_DOCS, 3, 7, &text_of, vec![0..N_DOCS; 1], 0),
            single_blob(0, N_DOCS, &text_of)
        );
    }

    /// A lane that outgrows its threshold fails the document that crosses
    /// it rather than spilling, and can then neither take more nor be
    /// frozen: the feed drops the lanes and the commit builds from the
    /// buffer.
    #[test]
    fn a_lane_that_outgrows_its_threshold_fails_and_cannot_be_frozen() {
        let mut lane = FtsBuilder::new();
        lane.register_column("body".into(), false)
            .expect("register");
        lane.set_lane_mode();
        lane.set_spill_threshold_bytes(1);
        assert!(
            lane.add_doc(0, 0, "alpha beta").is_err(),
            "the lane is over"
        );
        assert!(lane.add_doc(0, 1, "gamma").is_err(), "no more documents");
        assert!(
            lane.freeze(vec![Stretch {
                doc: 0,
                row: 0,
                len: 1,
            }])
            .is_err(),
            "no frozen lane"
        );
    }

    #[test]
    fn lanes_cut_at_the_finish_write_the_blobs_single_builders_write() {
        const N_DOCS: u32 = 500;
        /// Lane counts the rows are fed through: one, a few, and more lanes
        /// than slices of rows, since how many lanes an append opens depends
        /// on how many appends are open at once.
        const LANE_COUNTS: [u32; 4] = [1, 2, 3, 80];
        const SLICE: u32 = 7;
        const CUT: u32 = 213;
        let text_of = |i: u32| {
            let mut text = format!("every every u{i} shared{}", i % 7);
            if i.is_multiple_of(3) {
                text.push_str(" some some");
            }
            if i.is_multiple_of(50) {
                text.push_str(" rare");
            }
            if (CUT - 1..=CUT).contains(&i) {
                text.push_str(" straddle straddle");
            }
            text
        };
        let single = |first: u32, past: u32| single_blob(first, past, &text_of);
        for lanes in LANE_COUNTS {
            let from_view = |pieces: Vec<Range<u32>>, piece: usize| {
                lanes_blob(N_DOCS, lanes, SLICE, &text_of, pieces, piece)
            };
            assert_eq!(
                from_view(vec![0..N_DOCS; 1], 0),
                single(0, N_DOCS),
                "whole, {lanes} lanes"
            );
            assert_eq!(
                from_view(vec![0..CUT, CUT..N_DOCS], 0),
                single(0, CUT),
                "first piece, {lanes} lanes"
            );
            assert_eq!(
                from_view(vec![0..CUT, CUT..N_DOCS], 1),
                single(CUT, N_DOCS),
                "second piece, {lanes} lanes"
            );
        }

        // A lane whose stretches do not cover its documents is refused.
        let mut lane = two_column_builder();
        lane.set_lane_mode();
        lane.add_doc(0, 0, "a").expect("add");
        lane.add_doc(1, 0, "a").expect("add");
        assert!(
            lane.freeze(vec![Stretch {
                doc: 0,
                row: 5,
                len: 2
            }])
            .is_err()
        );
        // A builder not in lane mode cannot be frozen.
        assert!(two_column_builder().freeze(Vec::new()).is_err());
    }

    #[test]
    fn register_column_returns_sequential_ids() {
        let mut b = FtsBuilder::new();
        assert_eq!(
            b.register_column("title".into(), false)
                .expect("register column"),
            0
        );
        assert_eq!(
            b.register_column("body".into(), false)
                .expect("register column"),
            1
        );
        assert_eq!(
            b.register_column("tag".into(), false)
                .expect("register column"),
            2
        );
    }

    #[test]
    fn register_column_rejects_separator_byte() {
        let mut b = FtsBuilder::new();
        let bad = String::from("ti\x1Ftle");
        let err = b.register_column(bad, false).expect_err("expected error");
        assert!(matches!(err, BuildError::ReservedSeparatorInColumnName(_)));
    }

    #[test]
    fn register_column_rejects_reserved_prefix() {
        let mut b = FtsBuilder::new();
        let err = b
            .register_column("inf.title".into(), false)
            .expect_err("expected error");
        assert!(matches!(err, BuildError::ReservedPrefixInColumnName(_)));
    }

    #[test]
    fn register_column_rejects_duplicates() {
        let mut b = FtsBuilder::new();
        b.register_column("title".into(), false)
            .expect("register column");
        let err = b
            .register_column("title".into(), false)
            .expect_err("expected error");
        assert!(matches!(err, BuildError::DuplicateColumnName(_)));
    }

    #[test]
    fn add_doc_unknown_column_id_errors() {
        let mut b = FtsBuilder::new();
        b.register_column("title".into(), false)
            .expect("register column");
        let err = b.add_doc(99, 0, "text").expect_err("expected error");
        assert!(matches!(err, BuildError::FtsColumnTypeInvalid { .. }));
    }

    #[test]
    fn add_doc_tracks_doc_lengths_clamped() {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false)
            .expect("register column");
        b.add_doc(0, 0, "alpha beta gamma").expect("add doc");
        b.add_doc(0, 1, "").expect("add doc"); // zero-token doc
        b.add_doc(0, 2, "delta").expect("add doc");
        let col = &b.columns[0];
        assert_eq!(col.doc_lengths, vec![3, 0, 1]);
        assert_eq!(col.total_tokens, 4);
    }

    #[test]
    fn add_doc_updates_n_docs_per_call() {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false)
            .expect("register column");
        // Contract: local_doc_id is consecutive from 0 (per column).
        // n_docs ends up == max(local_doc_id) + 1 == call count.
        b.add_doc(0, 0, "a").expect("add doc");
        b.add_doc(0, 1, "b").expect("add doc");
        b.add_doc(0, 2, "c").expect("add doc");
        assert_eq!(b.n_docs, 3);
    }

    /// The version a build without a doc-id map stamps must be the one the
    /// staleness check treats as current. Raising `VERSION_CURRENT` without
    /// writing it would mark every file stale, including ones this engine
    /// just wrote, and a reindex would rewrite the whole table into files it
    /// still considered stale. This fails first instead.
    #[test]
    fn current_version_matches_the_written_version() {
        assert_eq!(blob_version(), format::fts::VERSION_CURRENT);
    }

    #[test]
    fn finish_emits_valid_header() {
        let mut b = FtsBuilder::new();
        b.register_column("title".into(), false)
            .expect("register column");
        b.add_doc(0, 0, "hello world").expect("add doc");
        let blob = b.finish().expect("finish");

        // Magic.
        assert_eq!(&blob[0..8], format::fts::MAGIC);
        // Version — new code always writes the current version.
        let version = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]);
        assert_eq!(version, format::fts::VERSION_CURRENT);
        // n_columns.
        let n_cols = u32::from_le_bytes([blob[12], blob[13], blob[14], blob[15]]);
        assert_eq!(n_cols, 1);
        // n_docs (u32 at 16..20).
        let n_docs = u32::from_le_bytes([blob[16], blob[17], blob[18], blob[19]]);
        assert_eq!(n_docs, 1);
        // n_terms_total = 2 ("hello", "world") (u32 at 20..24).
        let n_terms = u32::from_le_bytes([blob[20], blob[21], blob[22], blob[23]]);
        assert_eq!(n_terms, 2);
        // dict_offset == the header size (u64 at 24..32).
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&blob[24..32]);
        let dict_off = u64::from_le_bytes(buf);
        assert_eq!(dict_off, format::fts::HEADER_SIZE as u64);
    }

    /// Determinism gate: two independent in-RAM builds over the
    /// same corpus must emit byte-identical blobs. Catches hasher
    /// / iteration-order regressions on the in-RAM finish path.
    #[test]
    fn finish_to_matches_finish_byte_for_byte() {
        fn build() -> FtsBuilder {
            let mut b = FtsBuilder::new();
            b.register_column("title".into(), false)
                .expect("register title");
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
        use std::io::BufWriter;

        use bytes::Bytes;

        use crate::superfile::fts::reader::{BoolMode, FtsReader};

        let mut b = FtsBuilder::new();
        b.register_column("title".into(), false)
            .expect("register title");
        for i in 0..256u32 {
            b.add_doc(0, i, &format!("common term{i:03}"))
                .expect("add doc");
        }

        let tmp = tempdir().expect("tempdir");
        let path = tmp.path().join("fts.blob");
        {
            let file = File::create(&path).expect("create blob");
            let writer = BufWriter::new(file);
            b.finish_to(writer).expect("finish_to file");
        }
        let blob = fs::read(&path).expect("read blob");
        let r = FtsReader::open(
            Bytes::from(blob),
            r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75}]"#,
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
        let mut b = FtsBuilder::new();
        b.register_column("title".into(), false)
            .expect("register column");
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
        let parent = tempdir().expect("parent");
        let mut b = FtsBuilder::with_scratch(parent.path().to_path_buf()).expect("with_scratch");
        b.register_column("body".into(), false)
            .expect("register col");
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

    /// A positional column that crosses the spill threshold partway
    /// through the corpus writes the blob the in-RAM build writes, byte
    /// for byte: the run the scan wrote gap by gap onto the term's list
    /// and the run the spilled drain writes whole from the document's
    /// chain are the same bytes, and the dictionary and positions
    /// regions assemble the same either way.
    #[test]
    fn a_positional_build_that_spills_partway_matches_in_ram_byte_for_byte() {
        fn build_corpus(b: &mut FtsBuilder) {
            b.register_column("body".into(), true)
                .expect("register col");
            for i in 0..1000u32 {
                // Repeats within a document give runs with gaps; the
                // unique terms keep the vocabulary growing so the
                // threshold is crossed in the middle of the corpus.
                b.add_doc(
                    0,
                    i,
                    &format!(
                        "common shared common term{i:04} payload{} common extra word{i:04} shared",
                        i % 37
                    ),
                )
                .expect("add doc");
            }
        }
        let mut baseline = FtsBuilder::new();
        build_corpus(&mut baseline);
        assert!(baseline.is_in_ram(), "baseline must stay in RAM");
        let baseline_blob = baseline.finish().expect("finish baseline");

        for threshold in [4 * 1024, 16 * 1024, 64 * 1024] {
            let mut spilled = FtsBuilder::new();
            spilled.set_spill_threshold_bytes(threshold);
            build_corpus(&mut spilled);
            assert!(
                !spilled.is_in_ram(),
                "threshold {threshold} must force a spill"
            );
            let blob = spilled.finish().expect("finish spilled");
            assert_eq!(blob, baseline_blob, "threshold {threshold}");
        }
    }

    /// A document whose positions overflow leaves the builder unusable
    /// rather than holding a torn document.
    #[test]
    fn a_failed_document_poisons_the_builder() {
        #[derive(Debug)]
        struct Overflowing;
        impl Tokenizer for Overflowing {
            fn name(&self) -> &'static str {
                "overflowing_test"
            }
            fn tokenize<'a>(&'a self, _text: &'a str) -> Box<dyn Iterator<Item = String> + 'a> {
                Box::new(["a".to_string(), "b".to_string()].into_iter())
            }
            fn tokenize_each_positioned(&self, _text: &str, f: &mut dyn FnMut(&str, u64)) {
                f("a", 0);
                f("b", u64::from(u32::MAX) + 1);
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
        let mut b = FtsBuilder::new();
        b.register_column_with_tokenizer(
            "body".into(),
            true,
            Arc::new(Overflowing),
            bm25::Bm25Params::default(),
        )
        .expect("register");
        assert!(matches!(
            b.add_doc(0, 0, "x"),
            Err(BuildError::PositionOverflow { .. })
        ));
        assert!(b.add_doc(0, 1, "x").is_err(), "no more documents");
        assert!(b.finish().is_err(), "no blob");

        // A lane fails the same way, and cannot be frozen.
        let mut lane = FtsBuilder::new();
        lane.register_column_with_tokenizer(
            "body".into(),
            true,
            Arc::new(Overflowing),
            bm25::Bm25Params::default(),
        )
        .expect("register");
        lane.set_lane_mode();
        assert!(lane.add_doc(0, 0, "x").is_err());
        assert!(
            lane.freeze(vec![Stretch {
                doc: 0,
                row: 0,
                len: 1,
            }])
            .is_err(),
            "no frozen lane"
        );
    }

    /// A spill that fails leaves the builder unusable, not emptied: the
    /// in-RAM store was taken before the spill could fail, so a later
    /// finish must refuse rather than write a column with no postings.
    #[test]
    fn a_failed_spill_poisons_the_builder() {
        let scratch = tempdir().expect("scratch");
        let mut b = FtsBuilder::with_scratch(scratch.path().to_path_buf()).expect("builder");
        b.register_column("body".into(), false).expect("register");
        // Spill on the first document's drain.
        b.set_spill_threshold_bytes(1);
        // The spill's partition files then have nowhere to go.
        fs::remove_dir_all(b.scratch_dir.path()).expect("remove the scratch directory");
        assert!(
            b.add_doc(0, 0, "alpha beta gamma").is_err(),
            "the spill fails"
        );
        assert!(b.add_doc(0, 1, "delta").is_err(), "no more documents");
        assert!(b.finish().is_err(), "no blob");
    }

    #[tokio::test]
    async fn add_doc_accumulates_tf_within_doc() {
        let mut b = FtsBuilder::new();
        b.register_column("title".into(), false)
            .expect("register column");
        b.add_doc(0, 0, "rust rust rust async").expect("add doc");

        let blob = Bytes::from(b.finish().expect("finish"));
        let r = FtsReader::open(
            blob,
            r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75}]"#,
        )
        .expect("open");
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
        // A term that appears in two different columns must keep its
        // posting lists scoped per column in the emitted term dictionary +
        // posting region: each column has its own vocabulary and ids, and
        // the final dictionary key is `<col>\x1F<term>`.
        let mut b = FtsBuilder::new();
        let title_id = b
            .register_column("title".into(), false)
            .expect("register title");
        let body_id = b
            .register_column("body".into(), false)
            .expect("register body");

        // Doc 0: "rust" + "tokio" in title, "rust" + "async" in body.
        // Doc 1: "rust" in both.
        b.add_doc(title_id, 0, "rust tokio")
            .expect("add title doc 0");
        b.add_doc(body_id, 0, "rust async").expect("add body doc 0");
        b.add_doc(body_id, 1, "rust").expect("add body doc 1");
        b.add_doc(title_id, 1, "rust").expect("add title doc 1");

        // Round-trip through finish() + FtsReader::search. The reader looks
        // up via `dict::make_key(column, term)`, so this is the strict
        // on-disk equivalent of "two columns share a term — does each see
        // its own postings?"
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75},{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        // "rust" in title returns title's docs (0, 1) and no others.
        let hits_t = r
            .search("title", &["rust"], 10, BoolMode::Or)
            .await
            .expect("title search");
        let ids_t: Vec<u32> = hits_t.iter().map(|(d, _)| d.get()).collect();
        assert_eq!(ids_t.len(), 2, "title 'rust' hit count");
        assert!(ids_t.contains(&0));
        assert!(ids_t.contains(&1));

        // "rust" in body also returns its own docs (0, 1). Same ids by
        // coincidence; what matters is the search is scoped to body's
        // posting list, not title's.
        let hits_b = r
            .search("body", &["rust"], 10, BoolMode::Or)
            .await
            .expect("body search");
        let ids_b: Vec<u32> = hits_b.iter().map(|(d, _)| d.get()).collect();
        assert_eq!(ids_b.len(), 2, "body 'rust' hit count");
        assert!(ids_b.contains(&0));
        assert!(ids_b.contains(&1));

        // Cross-leak negative: a term that lives only in body (`async`)
        // must NOT be findable in title, and vice versa (`tokio` in body).
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
    fn build_above_threshold_spills_and_matches_in_ram_byte_for_byte() {
        // Threshold mode = real test: a low spill threshold forces
        // the same corpus to take the spilled finish_to path; result
        // must match the in-RAM finish byte-for-byte. This is the
        // streaming-dictionary regression gate — the spilled path uses
        // `TermBlockWriter` writing to a scratch file, while
        // the in-RAM path uses the in-memory `TermDictBuilder`. Both
        // must produce identical dictionary bytes.
        fn build_corpus(b: &mut FtsBuilder) {
            b.register_column("body".into(), false)
                .expect("register col");
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

        let mut baseline = FtsBuilder::new();
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
        let parent = tempdir().expect("parent");
        let mut spilled =
            FtsBuilder::with_scratch(parent.path().to_path_buf()).expect("with_scratch");
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
            "streaming-dictionary + spill path must produce byte-identical blob"
        );
    }

    /// Walk a directory recursively yielding only files.
    /// Local helper used by `small_build_stays_in_ram_no_spill_files_created`;
    /// avoids pulling in a dev-dep on `walkdir` for one test.
    fn walkdir_files(root: &Path) -> Vec<fs::DirEntry> {
        let mut out = Vec::new();
        let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let rd = match fs::read_dir(&dir) {
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
                .register_column("body".into(), false)
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
        let mut baseline = FtsBuilder::new();
        baseline.set_spill_threshold_bytes(1);
        build_corpus(&mut baseline);
        let baseline_blob = baseline.finish().expect("finish baseline");

        // Tight budget forces external merge. The column must
        // spill (`set_spill_threshold_bytes(1)`) so the spilled
        // finish path is even taken; then 1 KiB per partition is
        // well below the dominant partition's on-disk size, so the
        // merge path is exercised on at least one partition.
        finish_debug::reset();
        let mut tight = FtsBuilder::new();
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
        let parent = tempdir().expect("parent tempdir");
        let dir_count_before = fs::read_dir(parent.path()).expect("read parent").count();

        let mut b = FtsBuilder::with_scratch(parent.path().to_path_buf()).expect("with_scratch");
        b.register_column("body".into(), false)
            .expect("register col");
        b.add_doc(0, 0, "alpha beta gamma").expect("add doc");
        let _blob = b.finish().expect("finish");

        let dir_count_after = fs::read_dir(parent.path()).expect("read parent").count();
        assert_eq!(
            dir_count_after, dir_count_before,
            "FtsBuilder scratch tempdir leaked under override path"
        );
    }

    #[tokio::test]
    async fn configurable_spill_partitions_round_trips_through_reader() {
        use bytes::Bytes;

        use crate::superfile::fts::reader::{BoolMode, FtsReader};

        // Higher partition count: more files, smaller per-partition
        // working set. Must still produce a queryable blob.
        let mut b = FtsBuilder::new();
        b.set_spill_partitions(256).expect("set partitions");
        b.register_column("body".into(), false)
            .expect("register col");
        for i in 0..50u32 {
            b.add_doc(0, i, &format!("alpha beta gamma{i:02}"))
                .expect("add doc");
        }
        let blob = b.finish().expect("finish");
        let r = FtsReader::open(
            Bytes::from(blob),
            r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#,
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
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false)
            .expect("register column");
        for i in 0..10 {
            b.add_doc(0, i, &format!("term{i} common"))
                .expect("add doc");
        }
        let blob = b.finish().expect("finish");

        // Header layout post-u32-narrowing: dict_offset at 24..32,
        // postings_offset at 32..40, doc_lengths_table_offset at 40..48.
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&blob[24..32]);
        let dict_off = u64::from_le_bytes(buf) as usize;
        buf.copy_from_slice(&blob[32..40]);
        let postings_off = u64::from_le_bytes(buf) as usize;
        buf.copy_from_slice(&blob[40..48]);
        let dir_off = u64::from_le_bytes(buf) as usize;

        assert_eq!(dict_off, format::fts::HEADER_SIZE);
        assert!(postings_off > dict_off, "postings after term dictionary");
        assert!(dir_off > postings_off, "directory after postings");
        assert!(dir_off <= blob.len(), "directory offset within blob");
        buf.copy_from_slice(&blob[48..56]);
        let positions_off = u64::from_le_bytes(buf) as usize;
        assert!(positions_off > postings_off, "positions after postings");
        assert!(dir_off > positions_off, "directory after positions");
    }

    #[test]
    fn set_spill_partitions_rejects_after_register_column() {
        // Must be called before the first `register_column`.
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false)
            .expect("register col");
        let err = b.set_spill_partitions(16).expect_err("expected error");
        match err {
            BuildError::Io(e) => {
                assert!(e.to_string().contains("before any register_column"));
            }
            other => panic!("expected Io error, got {:?}", other),
        }
    }

    #[test]
    fn set_spill_partitions_rejects_zero() {
        let mut b = FtsBuilder::new();
        let err = b.set_spill_partitions(0).expect_err("expected error");
        match err {
            BuildError::Io(e) => assert!(e.to_string().contains("must be ≥ 1")),
            other => panic!("expected Io error, got {:?}", other),
        }
    }

    #[test]
    fn set_spill_partitions_rejects_non_power_of_two() {
        // Partition selection is `term_id & (n - 1)`, only correct for
        // power-of-two `n`.
        const NON_PO2: usize = 7;
        let mut b = FtsBuilder::new();
        let err = b.set_spill_partitions(NON_PO2).expect_err("expected error");
        match err {
            BuildError::Io(e) => assert!(e.to_string().contains("power of two")),
            other => panic!("expected Io error, got {:?}", other),
        }
    }

    #[test]
    fn read_partition_triples_empty_file_is_empty() {
        // An empty spill partition file decodes to zero triples.
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("empty.part");
        fs::write(&path, []).expect("write empty file");
        let triples = read_partition_records::<PLAIN_RECORD_LANES>(&path).expect("read empty");
        assert!(triples.is_empty());
    }

    #[test]
    fn read_partition_triples_round_trips_le_bytes() {
        // Spill files are a contiguous run of 12-byte little-endian
        // `[term_id, doc_id, tf]` triples; decoding restores them.
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("good.part");
        let mut bytes = Vec::new();
        for field in [3u32, 4, 5, 6, 7, 8] {
            bytes.extend_from_slice(&field.to_le_bytes());
        }
        fs::write(&path, &bytes).expect("write triples");
        let triples = read_partition_records::<PLAIN_RECORD_LANES>(&path).expect("read triples");
        assert_eq!(triples, vec![[3u32, 4, 5], [6u32, 7, 8]]);
    }

    #[test]
    fn read_partition_triples_rejects_non_multiple_length() {
        // A file whose byte length isn't a multiple of the 12-byte
        // triple size is malformed and must error.
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("ragged.part");
        // One full triple plus a stray byte.
        fs::write(
            &path,
            vec![0u8; mem::size_of::<[u32; PLAIN_RECORD_LANES]>() + 1],
        )
        .expect("write ragged file");
        let err = read_partition_records::<PLAIN_RECORD_LANES>(&path).expect_err("expected error");
        match err {
            BuildError::Io(e) => {
                assert_eq!(e.kind(), ErrorKind::InvalidData);
                assert!(e.to_string().contains("not a multiple"));
            }
            other => panic!("expected Io error, got {:?}", other),
        }
    }

    #[test]
    fn merge_entry_orders_by_sort_key_reversed() {
        // `MergeEntry` is a min-heap entry: `Ord` is reversed so a
        // BinaryHeap (max-heap) pops the smallest sort_key first. This
        // exercises the derived `PartialEq` / `PartialOrd` bridges too.
        let small = MergeEntry {
            sort_key: 10,
            rec: [1u32, 10, 1],
            reader_idx: 0,
        };
        let large = MergeEntry {
            sort_key: 20,
            rec: [2u32, 20, 1],
            reader_idx: 1,
        };
        // Reversed ordering: the smaller sort_key is the "greater"
        // entry so the max-heap surfaces it first.
        assert_eq!(small.cmp(&large), Ordering::Greater);
        assert_eq!(small.partial_cmp(&large), Some(Ordering::Greater));
        assert!(small != large);

        let small_dup = MergeEntry {
            sort_key: 10,
            rec: [9u32, 10, 9],
            reader_idx: 0,
        };
        // Equality is keyed on the comparator (sort_key then
        // reader_idx); the record lanes are payload and don't
        // affect it.
        assert!(small == small_dup);

        let mut heap: BinaryHeap<MergeEntry<PLAIN_RECORD_LANES>> = BinaryHeap::new();
        heap.push(large);
        heap.push(small);
        assert_eq!(heap.pop().expect("non-empty heap").sort_key, 10);
    }

    // ---- positional format ----

    /// Spill-forcing variant of [`build_title_blob`]: a 1-byte
    /// threshold pushes the column through the in-RAM → spill
    /// transition on the first doc; `max_partition_bytes` under the
    /// partition size additionally forces the chunked external merge.
    fn build_title_blob_spilled(
        docs: &[String],
        positional: bool,
        max_partition_bytes: Option<u64>,
    ) -> bytes::Bytes {
        let mut b = FtsBuilder::new();
        b.set_spill_threshold_bytes(1);
        if let Some(m) = max_partition_bytes {
            b.set_max_partition_bytes(m);
        }
        b.register_column("title".into(), positional)
            .expect("register column");
        for (i, text) in docs.iter().enumerate() {
            b.add_doc(0, i as u32, text).expect("add doc");
        }
        bytes::Bytes::from(b.finish().expect("finish"))
    }

    /// The full positional query battery must agree bit-for-bit
    /// across build paths: spilled positional vs in-RAM positional vs
    /// in-RAM positionless. Covers the spilled capture (5-lane
    /// records + positions blobs), the in-RAM → spill transition
    /// carrying accumulated runs, the lex-order merge's run assembly,
    /// and the df=1 inline packing fed from the spill path.
    async fn assert_title_blobs_agree(
        a: bytes::Bytes,
        a_json: &str,
        b: bytes::Bytes,
        b_json: &str,
        k: usize,
    ) {
        use crate::superfile::fts::reader::{BoolMode, FtsReader};
        let ra = FtsReader::open(a, a_json).expect("open a");
        let rb = FtsReader::open(b, b_json).expect("open b");
        let queries: &[(&[&str], BoolMode)] = &[
            (&["common"], BoolMode::Or),
            (&["uniqueonce"], BoolMode::Or),
            (&["dupdup"], BoolMode::Or),
            (&["medium", "uniqueonce"], BoolMode::Or),
            (&["common", "medium"], BoolMode::And),
        ];
        for (terms, mode) in queries {
            let ha = ra.search("title", terms, k, *mode).await.expect("search a");
            let hb = rb.search("title", terms, k, *mode).await.expect("search b");
            assert_eq!(ha, hb, "results diverged for {terms:?} ({mode:?})");
            assert!(!ha.is_empty(), "corpus sanity: {terms:?} matches");
        }
        for term in ["common", "medium", "uniqueonce", "dupdup"] {
            assert_eq!(
                ra.term_df("title", term).await.expect("df a").0,
                rb.term_df("title", term).await.expect("df b").0,
                "df diverged for {term}"
            );
        }
    }

    #[tokio::test]
    async fn spilled_positional_build_searches_identically() {
        let docs = positional_corpus();
        let k = docs.len();
        let spilled_pos = build_title_blob_spilled(&docs, true, None);
        assert_eq!(
            u32::from_le_bytes(spilled_pos[8..12].try_into().expect("version bytes")),
            format::fts::VERSION_CURRENT
        );
        let inram_pos = build_title_blob(&docs, true);
        let inram_plain = build_title_blob(&docs, false);
        assert_title_blobs_agree(
            spilled_pos.clone(),
            title_json(true),
            inram_pos,
            title_json(true),
            k,
        )
        .await;
        assert_title_blobs_agree(
            spilled_pos,
            title_json(true),
            inram_plain,
            title_json(false),
            k,
        )
        .await;
    }

    #[tokio::test]
    async fn spilled_positional_external_merge_searches_identically() {
        // A tiny per-partition sort budget forces the chunked
        // external-merge path over the 5-lane records.
        let docs = positional_corpus();
        let k = docs.len();
        let spilled = build_title_blob_spilled(&docs, true, Some(64));
        let inram = build_title_blob(&docs, true);
        assert_title_blobs_agree(spilled, title_json(true), inram, title_json(true), k).await;
    }

    /// Column-json for the single "title" column, with or without the
    /// positions flag — matching what `fts_columns_json` emits.
    fn title_json(positional: bool) -> &'static str {
        match positional {
            true => {
                r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#
            }
            false => r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75}]"#,
        }
    }

    /// Build a one-column blob from `docs`, positional or not.
    fn build_title_blob(docs: &[String], positional: bool) -> bytes::Bytes {
        let mut b = FtsBuilder::new();
        b.register_column("title".into(), positional)
            .expect("register column");
        for (i, text) in docs.iter().enumerate() {
            b.add_doc(0, i as u32, text).expect("add doc");
        }
        bytes::Bytes::from(b.finish().expect("finish"))
    }

    /// Corpus exercising every positional emit shape: a multi-block
    /// common term (> 2 × BLOCK_LEN docs), a df=1 tf=1 term (inline
    /// position packing), a df=1 tf>1 term (PFOR-forced df=1), and a
    /// medium term with within-doc repeats (multi-varint runs).
    fn positional_corpus() -> Vec<String> {
        let n_docs = 3 * BLOCK_LEN + 7;
        let mut docs = Vec::with_capacity(n_docs);
        for i in 0..n_docs {
            let mut t = String::from("common filler");
            if i % 5 == 0 {
                t.push_str(" medium medium");
            }
            if i == 42 {
                t.push_str(" uniqueonce");
            }
            if i == 43 {
                t.push_str(" dupdup dupdup dupdup");
            }
            docs.push(t);
        }
        docs
    }

    #[test]
    fn positionless_region_is_empty_and_positions_follow_postings() {
        let docs = positional_corpus();
        let plain = build_title_blob(&docs, false);
        let positional = build_title_blob(&docs, true);

        let version_of = |blob: &bytes::Bytes| {
            u32::from_le_bytes(blob[8..12].try_into().expect("4 header bytes"))
        };
        assert_eq!(version_of(&plain), format::fts::VERSION_CURRENT);
        assert_eq!(version_of(&positional), format::fts::VERSION_CURRENT);

        // A positionless build's region is just the CRC-of-empty. The
        // doc-id map's length trailer sits between it and the directory, so
        // the span from the positions offset to the directory covers both.
        let read_u64_plain =
            |at: usize| u64::from_le_bytes(plain[at..at + 8].try_into().expect("8 header bytes"));
        let trailer = format::fts::doc_map_len_trailer(version_of(&plain)) as u64;
        let region_len = read_u64_plain(40) - read_u64_plain(48) - trailer;
        assert_eq!(region_len, 4, "positionless region = 4-byte CRC only");

        // The positions-region offset points between the postings region
        // and the doc-lengths directory.
        let read_u64 = |blob: &bytes::Bytes, at: usize| {
            u64::from_le_bytes(blob[at..at + 8].try_into().expect("8 header bytes"))
        };
        let postings_off = read_u64(&positional, 32);
        let doc_lengths_off = read_u64(&positional, 40);
        let positions_off = read_u64(&positional, 48);
        assert!(postings_off < positions_off, "positions follow postings");
        assert!(
            positions_off < doc_lengths_off,
            "doc-lengths directory follows positions"
        );
        // The region is non-empty: the corpus has multi-doc terms
        // whose runs must land in it (only df=1 tf=1 inlines skip it).
        assert!(doc_lengths_off - positions_off > 4, "region has a body");
    }

    #[tokio::test]
    async fn positional_build_searches_identically_to_positionless() {
        use crate::superfile::fts::reader::{BoolMode, FtsReader};

        let docs = positional_corpus();
        let plain =
            FtsReader::open(build_title_blob(&docs, false), title_json(false)).expect("plain");
        let positional =
            FtsReader::open(build_title_blob(&docs, true), title_json(true)).expect("positional");

        // Positions must never change matching or scoring: identical
        // (doc, score) lists for every query shape — multi-block
        // common term, inline df=1 (tf=1), PFOR-forced df=1 (tf>1),
        // AND / OR multi-term.
        let queries: &[(&[&str], BoolMode)] = &[
            (&["common"], BoolMode::Or),
            (&["uniqueonce"], BoolMode::Or),
            (&["dupdup"], BoolMode::Or),
            (&["medium", "uniqueonce"], BoolMode::Or),
            (&["common", "medium"], BoolMode::And),
        ];
        let k = docs.len();
        for (terms, mode) in queries {
            let a = plain
                .search("title", terms, k, *mode)
                .await
                .expect("plain search");
            let b = positional
                .search("title", terms, k, *mode)
                .await
                .expect("positional search");
            assert_eq!(a, b, "results diverged for {terms:?} ({mode:?})");
            assert!(!a.is_empty(), "corpus sanity: {terms:?} matches");
        }

        // Count + df fast paths agree too.
        for term in ["common", "medium", "uniqueonce", "dupdup"] {
            let a = plain.term_df("title", term).await.expect("plain df").0;
            let b = positional
                .term_df("title", term)
                .await
                .expect("positional df")
                .0;
            assert_eq!(a, b, "df diverged for {term}");
            let ca = plain
                .token_match_count("title", &[term], BoolMode::Or)
                .await
                .expect("plain count")
                .0;
            let cb = positional
                .token_match_count("title", &[term], BoolMode::Or)
                .await
                .expect("positional count")
                .0;
            assert_eq!(ca, cb, "count diverged for {term}");
        }
    }

    #[tokio::test]
    async fn mixed_columns_only_positional_column_pays() {
        use crate::superfile::fts::reader::{BoolMode, FtsReader};

        // Two columns, one positional: both keep answering queries, each
        // with its own term-meta stride.
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        b.register_column("title".into(), true).expect("register");
        for i in 0..(BLOCK_LEN as u32 + 9) {
            b.add_doc(0, i, "shared bodyterm").expect("body doc");
            b.add_doc(1, i, "shared titleterm titleterm")
                .expect("title doc");
        }
        let blob = bytes::Bytes::from(b.finish().expect("finish"));
        assert_eq!(
            u32::from_le_bytes(blob[8..12].try_into().expect("version bytes")),
            format::fts::VERSION_CURRENT
        );
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75},{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let body_hits = r
            .search("body", &["bodyterm"], 10, BoolMode::Or)
            .await
            .expect("body search");
        let title_hits = r
            .search("title", &["titleterm"], 10, BoolMode::Or)
            .await
            .expect("title search");
        assert_eq!(body_hits.len(), 10);
        assert_eq!(title_hits.len(), 10);
        // The same term string scoped to each column stays isolated.
        let shared_body = r
            .token_match_count("body", &["shared"], BoolMode::Or)
            .await
            .expect("shared body")
            .0;
        assert_eq!(shared_body, BLOCK_LEN as u64 + 9);
    }

    /// A term's head names each document it keeps by the row it stands
    /// for, with the document's own term frequency and its own length,
    /// quantized: in a blob whose documents are renumbered, the length is
    /// looked up by document, not by row. It keeps the best-scoring
    /// documents, as many as the depth.
    #[test]
    fn a_head_names_its_documents_by_row_with_their_own_lengths() {
        /// Documents holding the term.
        const DOCS: u32 = 12;
        /// Documents the head keeps.
        const DEPTH: usize = 4;
        /// Length step between documents, so no two share a length.
        const LEN_STEP: u32 = 37;
        /// Term frequencies cycle through this many values.
        const TF_CYCLE: u32 = 5;
        let doc_lengths: Vec<u32> = (0..DOCS).map(|d| 1 + d * LEN_STEP).collect();
        let avgdl = doc_lengths.iter().sum::<u32>() as f32 / DOCS as f32;
        // Blob document d stands for row DOCS - 1 - d.
        let rows: Vec<u32> = (0..DOCS).rev().collect();
        let pairs: Vec<(u32, u32)> = (0..DOCS).map(|d| (d, 1 + d % TF_CYCLE)).collect();
        let params = bm25::Bm25Params::default();
        let enc = TermEncoding {
            doc_lengths: &doc_lengths,
            avgdl,
            params,
            n_scored_docs: DOCS,
            heads: true,
            head_depth: DEPTH,
            rows: Some(&rows),
        };
        let idf = bm25::idf(u64::from(DOCS), pairs.len() as u64);
        let scores: Vec<f32> = pairs
            .iter()
            .map(|&(doc, tf)| {
                bm25::score(
                    idf,
                    tf,
                    bm25::stored_len(doc_lengths[doc as usize]),
                    avgdl,
                    params,
                )
            })
            .collect();
        let mut best: Vec<u32> = (0..DOCS).collect();
        best.sort_by(|&a, &b| scores[b as usize].total_cmp(&scores[a as usize]));
        assert!(
            scores[best[DEPTH - 1] as usize] > scores[best[DEPTH] as usize],
            "the kept documents are unambiguous"
        );
        let mut scratch = TermScratch::default();
        term_head(
            &pairs,
            &scores,
            &enc,
            &mut scratch.head_heap,
            &mut scratch.head_kept,
            &mut scratch.head,
        );
        let head = read_head(&scratch.head.facts, &mut 0).expect("a head");
        assert_eq!(head.len(), DEPTH);
        let mut kept: Vec<u32> = head
            .iter()
            .map(|&(row, tf, len)| {
                let doc = DOCS - 1 - row;
                assert_eq!(tf, pairs[doc as usize].1, "row {row}'s own tf");
                assert_eq!(
                    len,
                    bm25::quantize_len(doc_lengths[doc as usize]),
                    "row {row}'s own length"
                );
                doc
            })
            .collect();
        kept.sort_unstable();
        let mut expected = best[..DEPTH].to_vec();
        expected.sort_unstable();
        assert_eq!(kept, expected, "the best-scoring documents");
    }
}
