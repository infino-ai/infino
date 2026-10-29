// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! PFOR-delta block codec for posting lists.
//!
//! Postings are encoded in blocks of up to **256 docs**: one or two
//! **sub-blocks** of 128 lanes, the unit `BitPacker4x` packs with SIMD,
//! at one shared bit width. Each block stores a sorted run of `doc_ids`
//! (delta-encoded relative to a base doc id) plus per-doc `tfs`, both
//! bit-packed at the minimum width needed for that block. A block is
//! decodable given its bytes and the previous block's last doc id
//! (which the skip table carries), so the skip table can jump straight
//! to a block. A block of more than 128 docs halves the per-block work a
//! reader pays per document — header, skip entry, position group, tf
//! array — which is why the writer prefers it; a block of at most 128
//! docs is one sub-block, which is also every block of a file written
//! before blocks could be wider.
//!
//! See `docs/architecture/superfile.md` for the overall posting region
//! layout and how blocks chain into the BM25 + BlockMaxWAND query loop.
//!
//! ## On-disk block layout
//!
//! The header is one 32-bit little-endian word:
//!
//! ```text
//!   bits    field
//!   ─────────────────────────────────────────────────────────────────
//!   0..7    doc_count - 1, low seven bits
//!   7..13   delta_bits          (0..=32, bit-width for deltas)
//!   13..19  tf_bits             (0..=32, bit-width for tfs)
//!   19..24  n_delta_exceptions  (patched blocks; 0 otherwise)
//!   24..26  encoding            (byte 3's low two bits)
//!   26..31  n_tf_exceptions     (patched blocks; 0 otherwise)
//!   31      from blob version 11: doc_count - 1, bit seven (so a count
//!           up to 256); before it: tfs stored minus one (set by every
//!           block written from version 10, clear on older blocks,
//!           which store tfs as is). A version-11 block always stores
//!           its tfs minus one.
//! ```
//!
//! The reader is told which reading of bit 31 a file takes when it
//! parses a header; everything else about a block follows from the
//! count, which fixes the lane count: 128 lanes (one sub-block) for a
//! count up to 128, 256 (two) above it.
//!
//! A packed or patched block's base doc id is not stored: it is the
//! previous block's last doc id, or zero for a term's first block, so
//! `delta[0] = doc[0] - base` is at least one from the second block on.
//! A bitset block's word-aligned origin follows the word as a `u32`
//! (its presence words start at the block's own first doc, not at the
//! previous block's last).
//!
//! After the header:
//!
//! ```text
//!   lanes × delta_bits / 8  packed deltas (always every lane)
//!   lanes × tf_bits / 8     packed tfs    (always every lane)
//! ```
//!
//! A two-sub-block stream is the first sub-block's packing followed by
//! the second's, both at the block's width; the second sub-block's
//! sorted deltas are relative to the first's last doc. `lanes × bits / 8`
//! is always an integer because a sub-block is 128 lanes.
//!
//! Two further encodings share the header and put the tfs last: a
//! **bitset** block ([`ENCODING_BITSET`]) stores dense doc ids as
//! presence words, and a **patched** block ([`ENCODING_PATCHED`]) packs
//! deltas and tfs at the width most lanes fit and lists the outliers as
//! exceptions — see each constant's documentation for its layout.
//!
//! ## Partial last block
//!
//! The last block in a posting list may have fewer docs than its lanes.
//! The encoder pads `doc_ids` with the last real value (delta = 0) and
//! pads `tfs` with zero before bit-packing — this keeps both arrays at
//! the bitpacker's required sub-block length without inflating the
//! bit width. The header's `doc_count` tells the consumer how many of
//! the decoded slots are real.
//!
//! ## Precondition-check convention
//!
//! - **`assert!`** for O(1) checks (slice length, type fits, header
//!   field in range). Cost is a few cycles per call; runs in release.
//!   Worth the safety net at function boundaries — especially before
//!   handing slices to the bitpacking crate, whose SIMD path uses
//!   `unsafe` writes that assume sufficient destination length.
//! - **`debug_assert!`** for O(n)-or-worse checks (sorted-ness,
//!   per-element content validation). Stripped in release; the cheap
//!   `assert!` preconditions plus the caller's contract carry safety
//!   in production.

use std::ops::Range;

use bitpacking::{BitPacker, BitPacker4x};
use wide::u32x8;

use crate::{
    superfile::bits::{
        EXPAND_SLACK, ExceptionPlan, PackScratch, expand_bitset, expand_bitset_scalar,
        payload_bytes, plan_exceptions,
    },
    utils::varint::{CONTINUATION_BIT, push_varint, read_varint, varint_len},
};

/// Lanes one bit-packer call covers: a **sub-block**, and the most docs a
/// block holds in a file written before blocks could be wider.
pub const SUB_BLOCK_LEN: usize = BitPacker4x::BLOCK_LEN;
/// Most `(doc_id, tf)` pairs in one block: two sub-blocks. Every buffer a
/// block decodes into is sized by it; a block's real lane count is
/// [`BlockHeader::lanes`].
pub const BLOCK_LEN: usize = 2 * SUB_BLOCK_LEN;

/// Header byte offset of the block `encoding` field: that byte's low two
/// bits ([`ENCODING_MASK`]).
pub const ENCODING_OFF: usize = 3;
/// Mask selecting the encoding within header byte [`ENCODING_OFF`].
pub const ENCODING_MASK: u8 = 0b11;

/// Block `encoding`: doc ids stored as PFOR-delta packing.
pub const ENCODING_PACKED: u8 = 0;
/// Block `encoding`: **patched** packing. Doc-id deltas
/// and tfs are each bit-packed at a width most lanes fit, and the few
/// lanes that do not — the outliers that would otherwise set the width
/// for every lane — store their high bits as **exceptions**. Layout after
/// the header, whose word carries the two exception counts:
///
/// ```text
///   lanes × delta_bits / 8  packed low bits of the deltas
///   ...                     per delta exception: lane (u8), high bits (LEB128 u32)
///   ...                     per tf exception:    lane (u8), high bits (LEB128 u32)
///   lanes × tf_bits / 8     packed low bits of the tfs (trailing, as always)
/// ```
///
/// Deltas here are explicit (`doc[0] - base`, then `doc[i] -
/// doc[i-1]`, padding lanes 0), prefix-summed by the reader after the
/// exceptions are patched in — the sorted bit-packer's fused prefix sum
/// cannot see a patch — about 15 ns more per block than the plain
/// decode. Taken only when the writer allows it for the term (see
/// [`encode_block_with`]'s `patchable`), when it is smaller than plain packing
/// and when the bitset did not claim the block, so a block with uniform
/// widths stays [`ENCODING_PACKED`] and decodes exactly as before.
pub const ENCODING_PATCHED: u8 = 2;
/// Most exception lanes a patched stream may carry: the header's count
/// field is five bits, and past this the stream is better off wider.
const PATCHED_MAX_EXCEPTIONS: usize = 31;

/// Block `encoding`: a **run** of consecutive doc ids. The header's origin
/// word holds the first doc; the block stores no doc payload at all, and
/// the doc ids are `origin..origin + count`. Tfs follow, packed as in
/// [`ENCODING_PACKED`].
pub const ENCODING_RUN: u8 = 3;

/// Block `encoding`: doc ids stored as a **presence bitset** over
/// `[origin, last_doc_id]`, `origin` aligned down to a 64-bit word so
/// the union count can OR it in word-aligned. Chosen only when it does
/// not grow the block (dense blocks). Tfs follow, packed identically to
/// PACKED.
pub const ENCODING_BITSET: u8 = 1;

/// Align a doc id down to the 64-bit word that contains it — the origin of
/// a [`ENCODING_BITSET`] block's presence bitset, so its words line up
/// with the union bitset and the OR needs no per-word bit shift.
#[inline]
pub fn bitset_block_base(doc_id: u32) -> u32 {
    doc_id & !63
}

/// Bytes the header takes for a block of `encoding`.
#[inline]
fn header_bytes(encoding: u8) -> usize {
    match encoding {
        ENCODING_BITSET | ENCODING_RUN => COMPACT_HEADER_SIZE + ORIGIN_BYTES,
        _ => COMPACT_HEADER_SIZE,
    }
}

/// The header: one `u32` word.
const COMPACT_HEADER_SIZE: usize = 4;
/// A bitset block's stored origin (`u32` LE) after the word.
const ORIGIN_BYTES: usize = 4;
/// Header field positions and widths.
const HDR_COUNT_BITS: u32 = 7;
const HDR_DELTA_BITS_SHIFT: u32 = 7;
const HDR_TF_BITS_SHIFT: u32 = 13;
const HDR_WIDTH_BITS: u32 = 6;
const HDR_DELTA_EXC_SHIFT: u32 = 19;
const HDR_ENCODING_SHIFT: u32 = 24;
const HDR_TF_EXC_SHIFT: u32 = 26;
/// Header bit 31: in a file whose blocks may be wide, the count's bit
/// seven; before that, set when the block stores its tfs minus one.
const HDR_TOP_SHIFT: u32 = 31;
const HDR_EXC_BITS: u32 = 5;

/// Whether a block of `encoding` answers presence and rank without a
/// decode: a bitset by bit test and popcount, a run by arithmetic.
#[inline]
pub fn is_dense_encoding(encoding: u8) -> bool {
    encoding == ENCODING_BITSET || encoding == ENCODING_RUN
}

/// The encoding of the block at `bytes` — one byte read.
#[inline]
pub fn block_encoding(bytes: &[u8]) -> u8 {
    bytes[ENCODING_OFF] & ENCODING_MASK
}

/// A block's header, decoded: what every decode needs before it touches
/// the payload. Parsed from the bytes plus the previous block's last doc
/// id. Twelve bytes, so the cursor's per-block
/// cache of it is one store and one compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockHeader {
    /// Real `(doc_id, tf)` pairs in the block, `1..=BLOCK_LEN`, stored
    /// minus one so a full wide block's 256 fits the byte.
    count_minus_one: u8,
    pub delta_bits: u8,
    pub tf_bits: u8,
    pub encoding: u8,
    n_delta_exc: u8,
    n_tf_exc: u8,
    /// Where the doc-id payload (deltas or presence words) begins.
    payload: u8,
    /// The value the first delta is relative to (packed and patched
    /// blocks), or the presence bitset's word-aligned origin.
    pub base: u32,
    /// What a stored tf is short of the real one: `1` on blocks that store
    /// tfs minus one, `0` on blocks written before they did.
    pub tf_bias: u8,
}

impl BlockHeader {
    /// Parse the header at the start of `bytes`. `prev_last_doc_id` is
    /// the previous block's last doc id, `None` for a term's first block.
    /// `wide` says the file was written with blocks of up to two
    /// sub-blocks (blob version 11 on), where bit 31 is the count's bit
    /// seven and every block stores its tfs minus one; in an older file
    /// bit 31 is the minus-one flag and a block is one sub-block.
    ///
    /// # Panics
    ///
    /// `bytes` is shorter than the header, or a bit width is past 32
    /// (the packer's unsafe kernels take the width on trust). The other
    /// fields cannot leave their range — the header word gives them no
    /// room to — and the postings region is CRC-validated at open, so
    /// nothing else is checked on this per-block path.
    #[inline]
    pub fn parse(bytes: &[u8], prev_last_doc_id: Option<u32>, wide: bool) -> Self {
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let field = |shift: u32, bits: u32| ((word >> shift) & ((1 << bits) - 1)) as u8;
        let top = field(HDR_TOP_SHIFT, 1);
        let (count_minus_one, tf_bias) = match wide {
            true => (field(0, HDR_COUNT_BITS) | (top << HDR_COUNT_BITS), 1),
            false => (field(0, HDR_COUNT_BITS), top),
        };
        let encoding = field(HDR_ENCODING_SHIFT, 2);
        let (base, payload) = match encoding {
            ENCODING_BITSET | ENCODING_RUN => (
                u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                (COMPACT_HEADER_SIZE + ORIGIN_BYTES) as u8,
            ),
            _ => (prev_last_doc_id.unwrap_or(0), COMPACT_HEADER_SIZE as u8),
        };
        let header = Self {
            count_minus_one,
            delta_bits: field(HDR_DELTA_BITS_SHIFT, HDR_WIDTH_BITS),
            tf_bits: field(HDR_TF_BITS_SHIFT, HDR_WIDTH_BITS),
            encoding,
            base,
            n_delta_exc: field(HDR_DELTA_EXC_SHIFT, HDR_EXC_BITS),
            n_tf_exc: field(HDR_TF_EXC_SHIFT, HDR_EXC_BITS),
            payload,
            tf_bias,
        };
        // Widths are validated where they drive an unpacker
        // (`check_widths`), not on every probe that only needs the
        // encoding and origin.
        debug_assert!(
            header.count() <= BLOCK_LEN,
            "block header: doc_count > BLOCK_LEN"
        );
        debug_assert!(
            header.encoding <= ENCODING_RUN,
            "block header: unknown encoding"
        );
        header
    }

    /// Reject a corrupt header before its widths reach an unpacker.
    #[inline]
    pub fn check_widths(&self) {
        assert!(
            self.delta_bits <= 32 && self.tf_bits <= 32,
            "block header: bit width > 32"
        );
    }

    /// Real `(doc_id, tf)` pairs in the block.
    #[inline]
    pub fn count(&self) -> usize {
        usize::from(self.count_minus_one) + 1
    }

    /// Lanes the block's packed streams hold: one sub-block for a count up
    /// to [`SUB_BLOCK_LEN`], two above it.
    #[inline]
    pub fn lanes(&self) -> usize {
        lanes_for(self.count())
    }

    /// Where the doc-id payload (deltas or presence words) begins.
    #[inline]
    pub fn payload(&self) -> usize {
        usize::from(self.payload)
    }

    /// Exceptions in the delta stream (patched blocks).
    #[inline]
    pub fn n_delta_exc(&self) -> usize {
        usize::from(self.n_delta_exc)
    }

    /// Exceptions in the tf stream (patched blocks).
    #[inline]
    pub fn n_tf_exc(&self) -> usize {
        usize::from(self.n_tf_exc)
    }

    /// Bytes the trailing packed tfs take.
    #[inline]
    pub fn tfs_size(&self) -> usize {
        packed_bytes(self.lanes(), self.tf_bits)
    }

    /// Bytes the packed deltas take (packed and patched blocks).
    #[inline]
    fn deltas_size(&self) -> usize {
        packed_bytes(self.lanes(), self.delta_bits)
    }
}

/// One block of postings — sorted-ascending `doc_ids` plus per-doc
/// `tfs`. Both vectors must have the same length, ≤ [`BLOCK_LEN`].
pub struct Block {
    pub doc_ids: Vec<u32>,
    pub tfs: Vec<u32>,
}

/// Encoded form of one block. `bytes` is the on-disk byte layout
/// described in the module docs; `last_doc_id` and `max_tf` are
/// duplicated out of the block body for skip-table / BMW use without
/// re-decoding.
pub struct EncodedBlock {
    pub bytes: Vec<u8>,
    pub last_doc_id: u32,
    pub max_tf: u32,
}

/// The cheapest patched packing of a block's lanes: the packed low bits
/// plus, per exception, a lane byte and its high bits as a varint.
fn plan_patched(lanes: &[u32], plan: &mut ExceptionPlan, candidates: &mut Vec<(u32, u32)>) {
    plan_exceptions(
        lanes,
        PATCHED_MAX_EXCEPTIONS,
        |width| packed_bytes(lanes.len(), width),
        |_, hi| 1 + varint_len(hi),
        plan,
        candidates,
    );
}

/// Pack `lanes` at `width` bits, keeping only each lane's low `width`
/// bits (the exceptions carry the rest), into `out`.
fn compress_low_bits(bp: &BitPacker4x, lanes: &[u32], width: u8, out: &mut [u8]) {
    if width == 0 {
        return;
    }
    let mask: u32 = if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    };
    let mut low = [0u32; BLOCK_LEN];
    for (l, &v) in low.iter_mut().zip(lanes) {
        *l = v & mask;
    }
    pack(bp, &low[..lanes.len()], out, width);
}

/// Bytes a stream of `lanes` lanes takes at `width` bits: a whole number
/// of sub-blocks, so [`payload_bytes`] has no rounding to do.
#[inline]
fn packed_bytes(lanes: usize, width: u8) -> usize {
    payload_bytes(lanes, width)
}

/// Docs a full block holds: two sub-blocks when `wide`, one otherwise.
/// The one rule behind a header's lane count, a writer's block length
/// and a reader's estimate of a term's docs per block.
#[inline]
pub fn block_lanes(wide: bool) -> usize {
    match wide {
        true => BLOCK_LEN,
        false => SUB_BLOCK_LEN,
    }
}

/// Lanes a block of `count` docs packs: one sub-block up to
/// [`SUB_BLOCK_LEN`] docs, two above it.
#[inline]
fn lanes_for(count: usize) -> usize {
    block_lanes(count > SUB_BLOCK_LEN)
}

/// The sub-blocks of a stream of `lanes` (a whole number of them), each
/// with the value its sorted deltas are relative to: `initial` for the
/// first, the last lane of the one before for the rest.
fn sub_blocks(initial: u32, lanes: &[u32]) -> impl Iterator<Item = (u32, &[u32])> {
    lanes
        .chunks_exact(SUB_BLOCK_LEN)
        .enumerate()
        .map(move |(h, sub)| {
            let init = match h {
                0 => initial,
                _ => lanes[h * SUB_BLOCK_LEN - 1],
            };
            (init, sub)
        })
}

/// The widest lane of `lanes` (a whole number of sub-blocks).
fn num_bits(bp: &BitPacker4x, lanes: &[u32]) -> u8 {
    lanes
        .chunks_exact(SUB_BLOCK_LEN)
        .map(|sub| bp.num_bits(sub))
        .max()
        .unwrap_or(0)
}

/// The widest delta of the ascending `lanes`, each sub-block relative to
/// its starting value ([`sub_blocks`]).
fn num_bits_sorted(bp: &BitPacker4x, initial: u32, lanes: &[u32]) -> u8 {
    sub_blocks(initial, lanes)
        .map(|(init, sub)| bp.num_bits_sorted(init, sub))
        .max()
        .unwrap_or(0)
}

/// Pack `lanes` (a whole number of sub-blocks) at `width` into `out`, one
/// sub-block after another.
fn pack(bp: &BitPacker4x, lanes: &[u32], out: &mut [u8], width: u8) {
    let step = packed_bytes(SUB_BLOCK_LEN, width);
    for (h, sub) in lanes.chunks_exact(SUB_BLOCK_LEN).enumerate() {
        bp.compress(sub, &mut out[h * step..(h + 1) * step], width);
    }
}

/// Pack the ascending `lanes` as deltas at `width` into `out`, each
/// sub-block relative to its starting value ([`sub_blocks`]).
fn pack_sorted(bp: &BitPacker4x, initial: u32, lanes: &[u32], out: &mut [u8], width: u8) {
    let step = packed_bytes(SUB_BLOCK_LEN, width);
    for (h, (init, sub)) in sub_blocks(initial, lanes).enumerate() {
        bp.compress_sorted(init, sub, &mut out[h * step..(h + 1) * step], width);
    }
}

/// Unpack `dest.len()` lanes (a whole number of sub-blocks) at `width`
/// from `bytes`.
fn unpack(bytes: &[u8], dest: &mut [u32], width: u8) {
    let bp = BitPacker4x::new();
    let step = packed_bytes(SUB_BLOCK_LEN, width);
    for (h, sub) in dest.chunks_exact_mut(SUB_BLOCK_LEN).enumerate() {
        bp.decompress(&bytes[h * step..(h + 1) * step], sub, width);
    }
}

/// Unpack `dest.len()` ascending lanes from their deltas at `width`, the
/// first relative to `initial` and each later sub-block to the last lane
/// of the one before it.
fn unpack_sorted(initial: u32, bytes: &[u8], dest: &mut [u32], width: u8) {
    let bp = BitPacker4x::new();
    let step = packed_bytes(SUB_BLOCK_LEN, width);
    let mut init = initial;
    for (h, sub) in dest.chunks_exact_mut(SUB_BLOCK_LEN).enumerate() {
        bp.decompress_sorted(init, &bytes[h * step..(h + 1) * step], sub, width);
        init = sub[SUB_BLOCK_LEN - 1];
    }
}

/// The compact header word for a block.
fn compact_header(
    count: usize,
    delta_bits: u8,
    tf_bits: u8,
    encoding: u8,
    n_delta_exc: usize,
    n_tf_exc: usize,
) -> [u8; COMPACT_HEADER_SIZE] {
    debug_assert!((1..=BLOCK_LEN).contains(&count));
    debug_assert!(n_delta_exc <= PATCHED_MAX_EXCEPTIONS && n_tf_exc <= PATCHED_MAX_EXCEPTIONS);
    // The count's low seven bits in the count field and its bit seven at
    // the top: a block this crate writes always stores tfs minus one, so
    // the top bit is free to carry it (see the module docs).
    let count_minus_one = count as u32 - 1;
    let word = (count_minus_one & ((1 << HDR_COUNT_BITS) - 1))
        | (u32::from(delta_bits) << HDR_DELTA_BITS_SHIFT)
        | (u32::from(tf_bits) << HDR_TF_BITS_SHIFT)
        | ((n_delta_exc as u32) << HDR_DELTA_EXC_SHIFT)
        | (u32::from(encoding) << HDR_ENCODING_SHIFT)
        | ((n_tf_exc as u32) << HDR_TF_EXC_SHIFT)
        | ((count_minus_one >> HDR_COUNT_BITS) << HDR_TOP_SHIFT);
    word.to_le_bytes()
}

/// Encode one block. `prev_last_doc_id` is the previous block's last
/// doc id (`None` for a term's first block), which the base doc id is
/// derived from. `patchable`
/// lets the block take the patched form when that is smaller; the
/// writer grants it per term, withholding it from the common terms
/// whose blocks every intersection and union walks in bulk. `scratch`
/// holds the planners' buffers, reused from block to block.
///
/// A run of consecutive doc ids always takes [`ENCODING_RUN`]. Otherwise
/// the presence bitset is taken whenever it is at most `bitset_bias` times
/// the size of the packed deltas (`1.0`: only when it is no larger). A
/// bitset block is folded into a count a word at a time and probed by a
/// bit test, where a packed one is decoded doc by doc, so the writer pays
/// a few bytes for it on the terms that unions and intersections read in
/// bulk.
///
/// # Panics
///
/// - `b.doc_ids.is_empty()` — can't encode an empty block.
/// - `b.doc_ids.len() != b.tfs.len()` — mismatched parallel vectors.
/// - `b.doc_ids.len() > BLOCK_LEN`.
/// - (debug) `b.doc_ids` not strictly ascending, or not above
///   `prev_last_doc_id`.
pub fn encode_block_with(
    b: &Block,
    prev_last_doc_id: Option<u32>,
    patchable: bool,
    bitset_bias: f32,
    scratch: &mut PackScratch,
) -> EncodedBlock {
    let count = b.doc_ids.len();
    assert!(count > 0, "encode_block: empty block");
    assert_eq!(
        count,
        b.tfs.len(),
        "encode_block: doc_ids/tfs length mismatch"
    );
    assert!(
        count <= BLOCK_LEN,
        "encode_block: doc_count {count} > BLOCK_LEN {BLOCK_LEN}"
    );
    debug_assert!(
        b.doc_ids.windows(2).all(|w| w[0] < w[1]),
        "encode_block: doc_ids must be strictly ascending"
    );
    debug_assert!(
        prev_last_doc_id.is_none_or(|p| p < b.doc_ids[0]),
        "encode_block: doc_ids must follow the previous block"
    );

    let last_doc_id = b.doc_ids[count - 1];
    let max_tf = b.tfs.iter().copied().max().unwrap_or(0);
    let lanes = lanes_for(count);

    // Pad both arrays to the block's lanes. doc_ids: pad with the last
    // real value so the padded delta is 0. tfs: pad with 0 (default fill).
    let mut padded_doc_ids = [0u32; BLOCK_LEN];
    padded_doc_ids[..count].copy_from_slice(&b.doc_ids);
    for slot in &mut padded_doc_ids[count..lanes] {
        *slot = last_doc_id;
    }
    let padded_doc_ids = &padded_doc_ids[..lanes];
    // Tfs are stored minus one (every tf is at least one), so a block
    // whose docs each hold the term once packs its tfs at width zero.
    // Padding lanes stay zero.
    let mut padded_tfs = [0u32; BLOCK_LEN];
    for (l, &t) in padded_tfs.iter_mut().zip(&b.tfs) {
        debug_assert!(t >= 1, "encode_block: a posting's tf is at least one");
        *l = t - 1;
    }
    let padded_tfs = &padded_tfs[..lanes];

    // The value the decoder recovers doc_ids[0] from (decompressed[0] =
    // base + delta[0]): the previous block's last doc id, which the skip
    // table already holds, or zero for the first block — whose first
    // delta is then the doc id itself, an outlier the patched form absorbs
    // as one exception. A bitset's origin is stored.
    let base_doc_id = prev_last_doc_id.unwrap_or(0);
    let aligned_base = bitset_block_base(b.doc_ids[0]);

    let bp = BitPacker4x::new();
    let delta_bits = num_bits_sorted(&bp, base_doc_id, padded_doc_ids);
    let tf_bits = num_bits(&bp, padded_tfs);

    let deltas_size = packed_bytes(lanes, delta_bits);
    let tfs_size = packed_bytes(lanes, tf_bits);
    let header_size = header_bytes(ENCODING_PACKED);

    // Store the doc ids as a presence bitset instead of PFOR deltas when
    // that does not grow the block (a dense block — a common term's full
    // near-consecutive docs). The bitset origin is word-aligned so the
    // union count can OR it in without a per-word shift. Measured
    // against the tight deltas from the block's own first doc, so the
    // decision does not depend on where the previous block ended.
    let bitset_words = (last_doc_id - aligned_base) as usize / 64 + 1;
    let bitset_size = bitset_words * 8;
    let tight_delta_bits = num_bits_sorted(&bp, b.doc_ids[0].saturating_sub(1), padded_doc_ids);
    let use_bitset =
        bitset_size as f32 <= packed_bytes(lanes, tight_delta_bits) as f32 * bitset_bias;

    // A run of consecutive doc ids stores none of them: the origin word
    // holds the first, and the rest follow from the count.
    if last_doc_id - b.doc_ids[0] == count as u32 - 1 {
        let mut bytes = Vec::with_capacity(header_bytes(ENCODING_RUN) + tfs_size);
        bytes.extend_from_slice(&compact_header(count, 0, tf_bits, ENCODING_RUN, 0, 0));
        bytes.extend_from_slice(&b.doc_ids[0].to_le_bytes());
        let tfs_start = bytes.len();
        bytes.resize(tfs_start + tfs_size, 0);
        pack(&bp, padded_tfs, &mut bytes[tfs_start..], tf_bits);
        return EncodedBlock {
            bytes,
            last_doc_id,
            max_tf,
        };
    }

    // Patched packing: explicit deltas and tfs,
    // each at the width most lanes fit plus an exception list for the
    // rest. Considered only for a block the bitset did not claim: a dense
    // block's presence words are what the count kernels bit-test and
    // rank into, and that O(1) probe is worth more than the few bytes
    // patching a dense partial block would save.
    if patchable && !use_bitset {
        let mut explicit_deltas = [0u32; BLOCK_LEN];
        explicit_deltas[0] = b.doc_ids[0] - base_doc_id;
        for (slot, pair) in explicit_deltas[1..count]
            .iter_mut()
            .zip(b.doc_ids.windows(2))
        {
            *slot = pair[1] - pair[0];
        }
        let explicit_deltas = &explicit_deltas[..lanes];
        plan_patched(
            explicit_deltas,
            &mut scratch.plan_a,
            &mut scratch.candidates,
        );
        plan_patched(padded_tfs, &mut scratch.plan_b, &mut scratch.candidates);
        let (delta_plan, tf_plan) = (&scratch.plan_a, &scratch.plan_b);
        let patched_size = header_size + delta_plan.bytes + tf_plan.bytes;
        let plain_size = header_size + deltas_size + tfs_size;
        if patched_size < plain_size {
            let mut bytes = Vec::with_capacity(patched_size);
            bytes.extend_from_slice(&compact_header(
                count,
                delta_plan.width,
                tf_plan.width,
                ENCODING_PATCHED,
                delta_plan.exceptions.len(),
                tf_plan.exceptions.len(),
            ));
            let deltas_start = bytes.len();
            let deltas_packed = packed_bytes(lanes, delta_plan.width);
            bytes.resize(deltas_start + deltas_packed, 0);
            compress_low_bits(
                &bp,
                explicit_deltas,
                delta_plan.width,
                &mut bytes[deltas_start..],
            );
            for &(lane, hi) in delta_plan.exceptions.iter().chain(&tf_plan.exceptions) {
                bytes.push(lane as u8);
                push_varint(&mut bytes, hi);
            }
            let tfs_start = bytes.len();
            let tfs_packed = packed_bytes(lanes, tf_plan.width);
            bytes.resize(tfs_start + tfs_packed, 0);
            compress_low_bits(&bp, padded_tfs, tf_plan.width, &mut bytes[tfs_start..]);
            debug_assert_eq!(bytes.len(), patched_size);
            return EncodedBlock {
                bytes,
                last_doc_id,
                max_tf,
            };
        }
    }

    let doc_ids_size = if use_bitset { bitset_size } else { deltas_size };
    let encoding = if use_bitset {
        ENCODING_BITSET
    } else {
        ENCODING_PACKED
    };
    let mut bytes = Vec::with_capacity(header_bytes(encoding) + doc_ids_size + tfs_size);

    // Header.
    let stored_delta_bits = if use_bitset { 0 } else { delta_bits };
    bytes.extend_from_slice(&compact_header(
        count,
        stored_delta_bits,
        tf_bits,
        encoding,
        0,
        0,
    ));
    if use_bitset {
        bytes.extend_from_slice(&aligned_base.to_le_bytes());
    }

    // Doc ids.
    let doc_ids_start = bytes.len();
    bytes.resize(doc_ids_start + doc_ids_size, 0);
    if use_bitset {
        let words = &mut bytes[doc_ids_start..doc_ids_start + bitset_size];
        for &d in &b.doc_ids {
            let bit = (d - aligned_base) as usize;
            let w = (bit / 64) * 8;
            let lane = bit % 64;
            let mut word = u64::from_le_bytes(words[w..w + 8].try_into().expect("8 bytes"));
            word |= 1u64 << lane;
            words[w..w + 8].copy_from_slice(&word.to_le_bytes());
        }
    } else {
        pack_sorted(
            &bp,
            base_doc_id,
            padded_doc_ids,
            &mut bytes[doc_ids_start..doc_ids_start + deltas_size],
            delta_bits,
        );
    }

    // Packed tfs — identical in both encodings, in doc order.
    let tfs_start = bytes.len();
    bytes.resize(tfs_start + tfs_size, 0);
    pack(
        &bp,
        padded_tfs,
        &mut bytes[tfs_start..tfs_start + tfs_size],
        tf_bits,
    );

    EncodedBlock {
        bytes,
        last_doc_id,
        max_tf,
    }
}

/// Decode one block whose header is `hdr`. `dest_doc_ids` and
/// `dest_tfs` must each have at least [`BLOCK_LEN`] elements; the
/// decoder writes every lane of the block (`hdr.lanes()`, a run block
/// all `BLOCK_LEN`) for SIMD reasons, and the slots past that keep
/// whatever an earlier block left. The returned `doc_count` tells the
/// caller how many slots are real; the lanes past it are padding (zero
/// deltas for doc_ids, zero tfs), and nothing past the count may be read.
///
/// # Panics
///
/// - `bytes` is shorter than the header claims.
/// - `dest_doc_ids.len() < BLOCK_LEN` or `dest_tfs.len() < BLOCK_LEN`.
pub fn decode_block(
    bytes: &[u8],
    hdr: &BlockHeader,
    dest_doc_ids: &mut [u32],
    dest_tfs: &mut [u32],
) -> usize {
    hdr.check_widths();
    if hdr.encoding == ENCODING_PATCHED {
        // Both halves in one pass over the exception lists: the delta
        // pass ends where the tf exceptions begin.
        let tf_exc_start = decode_patched_doc_ids(bytes, hdr, dest_doc_ids);
        unpack_tfs(bytes, hdr, dest_tfs);
        patch_lanes(bytes, tf_exc_start, hdr.n_tf_exc(), hdr.tf_bits, dest_tfs);
        restore_tfs(dest_tfs, hdr);
        return hdr.count();
    }
    let count = decode_block_doc_ids(bytes, hdr, dest_doc_ids);
    decode_block_tfs(bytes, hdr, dest_tfs);
    count
}

/// Decode a patched block's doc ids: unpack the low delta bits, OR the
/// exceptions' high bits into their lanes, then prefix-sum from the
/// base. The prefix sum is the SIMD log-step form ([`prefix_sum_block`]);
/// the sorted bit-packer's fused one cannot see a patch. Returns the
/// offset where the tf exceptions begin.
fn decode_patched_doc_ids(bytes: &[u8], hdr: &BlockHeader, dest_doc_ids: &mut [u32]) -> usize {
    assert!(
        dest_doc_ids.len() >= BLOCK_LEN,
        "decode_block_doc_ids: dest_doc_ids must have at least {BLOCK_LEN} slots"
    );
    let deltas_size = hdr.deltas_size();
    assert!(
        bytes.len() >= hdr.payload() + deltas_size,
        "decode_block_doc_ids: bytes ({}) shorter than header+deltas ({})",
        bytes.len(),
        hdr.payload() + deltas_size
    );
    let lanes = hdr.lanes();
    unpack(
        &bytes[hdr.payload()..hdr.payload() + deltas_size],
        &mut dest_doc_ids[..lanes],
        hdr.delta_bits,
    );
    let tf_exc_start = patch_lanes(
        bytes,
        hdr.payload() + deltas_size,
        hdr.n_delta_exc(),
        hdr.delta_bits,
        dest_doc_ids,
    );
    prefix_sum_block(&mut dest_doc_ids[..lanes], hdr.base);
    tf_exc_start
}

/// Lanes per SIMD word of the prefix sum.
const PREFIX_LANES: usize = 8;

/// Turn a block's deltas into doc ids in place: `dest[i] = base + Σ
/// dest[..=i]`. Eight lanes at a time in log steps — each lane adds the
/// lane one, two, then four back — then the running carry; the lane
/// shifts are array rebuilds the compiler turns into register shuffles.
/// On the reference box this is a third faster than the packer's own
/// fused prefix sum and twice as fast as a serial scalar chain, where a
/// four-chain scalar form auto-vectorized into strided stores and was
/// slower than serial.
fn prefix_sum_block(dest: &mut [u32], base: u32) {
    debug_assert_eq!(dest.len() % PREFIX_LANES, 0);
    let mut carry = u32x8::splat(base);
    for chunk in dest.chunks_exact_mut(PREFIX_LANES) {
        let lanes: [u32; PREFIX_LANES] = chunk.try_into().expect("eight lanes");
        let v = u32x8::from(lanes);
        let a = v.to_array();
        let v = v + u32x8::from([0, a[0], a[1], a[2], a[3], a[4], a[5], a[6]]);
        let a = v.to_array();
        let v = v + u32x8::from([0, 0, a[0], a[1], a[2], a[3], a[4], a[5]]);
        let a = v.to_array();
        let v = v + u32x8::from([0, 0, 0, 0, a[0], a[1], a[2], a[3]]) + carry;
        let out = v.to_array();
        chunk.copy_from_slice(&out);
        carry = u32x8::splat(out[PREFIX_LANES - 1]);
    }
}

/// One exception at `at`: its lane, its high bits, and the offset past
/// it. The one-byte high value most exceptions carry skips the general
/// varint decode.
#[inline]
fn read_exception(bytes: &[u8], at: usize) -> (usize, u32, usize) {
    let lane = bytes[at] as usize;
    let first = bytes[at + 1];
    if first < CONTINUATION_BIT {
        return (lane, u32::from(first), at + 2);
    }
    let mut next = at + 1;
    let hi = read_varint(bytes, &mut next).expect("patched block exception within block");
    (lane, hi, next)
}

/// OR `n` exceptions starting at `at` — each a lane byte and its high
/// bits as a LEB128 `u32` — into their lanes of `dest` at `width`.
/// Returns the offset just past the list. One pass: a block is patched
/// and measured in the same walk, and the one-byte high bits most
/// exceptions carry skip the general varint decode.
#[inline]
fn patch_lanes(bytes: &[u8], mut at: usize, n: usize, width: u8, dest: &mut [u32]) -> usize {
    for _ in 0..n {
        let (lane, hi, next) = read_exception(bytes, at);
        at = next;
        dest[lane] |= hi << width;
    }
    at
}

/// Advance past `n` exceptions starting at `at` without applying them.
#[inline]
fn skip_lanes(bytes: &[u8], mut at: usize, n: usize) -> usize {
    for _ in 0..n {
        at += 1;
        read_varint(bytes, &mut at).expect("patched block exception within block");
    }
    at
}

/// Unpack a block's trailing packed tfs (the low bits, for a patched
/// block).
fn unpack_tfs(bytes: &[u8], hdr: &BlockHeader, dest_tfs: &mut [u32]) {
    hdr.check_widths();
    assert!(
        dest_tfs.len() >= BLOCK_LEN,
        "decode_block_tfs: dest_tfs must have at least {BLOCK_LEN} slots"
    );
    let tfs_size = hdr.tfs_size();
    assert!(
        bytes.len() >= hdr.payload() + tfs_size,
        "decode_block_tfs: bytes shorter than header+tfs"
    );
    let tfs_start = bytes.len() - tfs_size;
    unpack(
        &bytes[tfs_start..tfs_start + tfs_size],
        &mut dest_tfs[..hdr.lanes()],
        hdr.tf_bits,
    );
}

/// Byte ranges of a patched block's delta and tf exception lists.
///
/// # Panics
///
/// `bytes` is not a well-formed patched block (the CRC-validated
/// postings region is the caller's guarantee).
pub(crate) fn patched_exception_ranges(
    bytes: &[u8],
    hdr: &BlockHeader,
) -> (Range<usize>, Range<usize>) {
    let delta_exc_start = hdr.payload() + hdr.deltas_size();
    let tf_exc_start = skip_lanes(bytes, delta_exc_start, hdr.n_delta_exc());
    let end = skip_lanes(bytes, tf_exc_start, hdr.n_tf_exc());
    (delta_exc_start..tf_exc_start, tf_exc_start..end)
}

/// Decode only the tf array of a block (the trailing tf-packed bytes) into
/// `dest_tfs`, in doc order, skipping the doc-id half. The ranked-OR
/// membership probe locates a bitset-block doc by bit-test + popcount-rank
/// and needs only that doc's tf, never the expanded doc ids — so it decodes
/// the tfs (which are BitPacker4x-packed and can't be single-value-indexed)
/// once per block and reads the rank-th one, avoiding the doc-id expansion.
pub fn decode_block_tfs(bytes: &[u8], hdr: &BlockHeader, dest_tfs: &mut [u32]) {
    unpack_tfs(bytes, hdr, dest_tfs);
    if hdr.encoding == ENCODING_PATCHED {
        let tf_exc_start = skip_lanes(bytes, hdr.payload() + hdr.deltas_size(), hdr.n_delta_exc());
        patch_lanes(bytes, tf_exc_start, hdr.n_tf_exc(), hdr.tf_bits, dest_tfs);
    }
    restore_tfs(dest_tfs, hdr);
}

/// Undo the minus-one a block's tfs are stored with, on its real lanes;
/// padding lanes stay zero. A block written before tfs were stored that
/// way has a zero bias and is left as decoded.
#[inline]
fn restore_tfs(dest_tfs: &mut [u32], hdr: &BlockHeader) {
    if hdr.tf_bias == 0 {
        return;
    }
    let bias = u32::from(hdr.tf_bias);
    for t in &mut dest_tfs[..hdr.count()] {
        *t += bias;
    }
}

/// The first doc of a dense block (see [`is_dense_encoding`]) at or after
/// `from`, as `(doc, rank)` where `rank` is the doc's index in the block's
/// tf array. `None` when no doc of the block is `>= from`.
///
/// A probe that skips into a dense block needs one doc, not the block's
/// worth the full expansion writes. A run block answers by arithmetic; a bitset
/// block scans its words from the target's word with a masked
/// `trailing_zeros`, at most a handful of u64s, `rank` being the number of
/// set bits before the doc.
pub fn dense_next_doc(bytes: &[u8], hdr: &BlockHeader, from: u32) -> Option<(u32, usize)> {
    if hdr.encoding == ENCODING_RUN {
        // Every id from the origin on, `count` of them: arithmetic.
        let doc = from.max(hdr.base);
        let rank = (doc - hdr.base) as usize;
        return (rank < hdr.count()).then_some((doc, rank));
    }
    debug_assert_eq!(hdr.encoding, ENCODING_BITSET);
    hdr.check_widths();
    let tfs_size = hdr.tfs_size();
    assert!(
        bytes.len() >= hdr.payload() + tfs_size,
        "dense_next_doc: bytes shorter than header+tfs"
    );
    let words = &bytes[hdr.payload()..bytes.len() - tfs_size];
    let bit = from.saturating_sub(hdr.base) as usize;
    let first_word = bit / 64;
    let mut rank = 0usize;
    for (wi, chunk) in words.chunks_exact(8).enumerate() {
        let word = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
        if wi < first_word {
            rank += word.count_ones() as usize;
            continue;
        }
        let masked = match wi == first_word {
            true => word & (u64::MAX << (bit % 64)),
            false => word,
        };
        if masked != 0 {
            let tz = masked.trailing_zeros();
            rank += (word & !(u64::MAX << tz)).count_ones() as usize;
            return Some((hdr.base + (wi as u32) * 64 + tz, rank));
        }
        rank += word.count_ones() as usize;
    }
    None
}

/// One tf out of a dense block (see [`is_dense_encoding`]) without
/// unpacking the others: the tf of the doc at index `rank` (its set-bit
/// index in a bitset block, its offset from the origin in a run). Dense
/// blocks carry plain packed tfs (never patched). Within a sub-block
/// `BitPacker4x` interleaves four 32-bit lanes: value `i` lives in lane
/// `i % 4` at bit `(i / 4) * width` of that lane's stream, whose 32-bit
/// words sit at every fourth word of the sub-block's payload; a second
/// sub-block's payload follows the first's.
///
/// For a *walk* over a block the full unpack into the tf array wins (one
/// unpack serves every doc); this is for a *probe* that touches one doc.
pub fn dense_tf_at(bytes: &[u8], hdr: &BlockHeader, rank: usize) -> u32 {
    debug_assert!(hdr.encoding == ENCODING_BITSET || hdr.encoding == ENCODING_RUN);
    debug_assert!(rank < hdr.count());
    hdr.check_widths();
    let width = usize::from(hdr.tf_bits);
    if width == 0 {
        return u32::from(hdr.tf_bias);
    }
    let tfs_size = hdr.tfs_size();
    assert!(
        bytes.len() >= hdr.payload() + tfs_size,
        "dense_tf_at: bytes shorter than header+tfs"
    );
    let packed = &bytes[bytes.len() - tfs_size..];
    let packed = &packed[(rank / SUB_BLOCK_LEN) * packed_bytes(SUB_BLOCK_LEN, hdr.tf_bits)..];
    let rank = rank % SUB_BLOCK_LEN;
    let lane = rank % 4;
    let bit = (rank / 4) * width;
    let word = |w: usize| -> u64 {
        let at = (w * 4 + lane) * 4;
        u64::from(u32::from_le_bytes(
            packed[at..at + 4].try_into().expect("4 bytes"),
        ))
    };
    let shift = bit % 32;
    let mut v = word(bit / 32) >> shift;
    if shift + width > 32 {
        v |= word(bit / 32 + 1) << (32 - shift);
    }
    (v & ((1u64 << width) - 1)) as u32 + u32::from(hdr.tf_bias)
}

/// Decode only the doc ids of a block into `dest_doc_ids`, skipping the
/// tf half. Returns the block's doc count. The unranked count kernels
/// never read a tf, so they take this instead of [`decode_block`].
///
/// # Panics
///
/// As [`decode_block`], minus the `dest_tfs` checks, and when a run
/// block's ids would pass `u32::MAX`.
pub fn decode_block_doc_ids(bytes: &[u8], hdr: &BlockHeader, dest_doc_ids: &mut [u32]) -> usize {
    hdr.check_widths();
    assert!(
        dest_doc_ids.len() >= BLOCK_LEN,
        "decode_block_doc_ids: dest_doc_ids must have at least {BLOCK_LEN} slots"
    );
    let count = hdr.count();

    if hdr.encoding == ENCODING_RUN {
        // `origin..origin + count`; padding lanes repeat the last doc, as a
        // packed block's zero deltas do. A corrupt origin could put the
        // last id past `u32::MAX`, which would wrap to small ids.
        let last = hdr.base.checked_add(count as u32 - 1);
        assert!(
            last.is_some(),
            "decode_block_doc_ids: run block passes u32::MAX"
        );
        for (i, d) in dest_doc_ids[..BLOCK_LEN].iter_mut().enumerate() {
            *d = hdr.base + (i as u32).min(count as u32 - 1);
        }
        return count;
    }

    if hdr.encoding == ENCODING_BITSET {
        // Doc ids are a presence bitset over `[base, ...]`; the tfs are
        // the trailing `tf_bits`-packed bytes, so the bitset is everything
        // between the header and them. Emit the set bits in ascending
        // order (= ascending doc id) — the sorted order every consumer
        // expects.
        let tfs_size = hdr.tfs_size();
        assert!(
            bytes.len() >= hdr.payload() + tfs_size,
            "decode_block_doc_ids: bytes shorter than header+tfs"
        );
        let words = &bytes[hdr.payload()..bytes.len() - tfs_size];
        // Bounds safety: `j` advances once per set bit, so it is bounded by
        // `popcount(words)`, and the `dest_doc_ids[j]` writes carry no per-bit
        // bounds check on this hot decode loop because two invariants keep that
        // popcount ≤ `BLOCK_LEN`:
        //   1. The builder sets exactly `doc_count` bits (≤ `BLOCK_LEN`, the
        //      per-block cap) when it encodes a bitset block, so a well-formed
        //      block has `popcount == count ≤ BLOCK_LEN`.
        //   2. Decode only ever runs on CRC-validated bytes: the postings
        //      region's checksum is verified in `SuperfileReader::open` before
        //      any block is decoded, so a corrupted bitmap — which could carry
        //      extra set bits — is rejected at open and never reaches here.
        // The `debug_assert_eq!(j, count)` below is the test/debug tripwire that
        // fires if a future builder change ever breaks invariant 1. Both
        // expansions bound their writes themselves (a corrupt block panics
        // rather than overruns), so invariant 2 keeps that panic unreachable.
        let j = match expand_bitset(words, hdr.base, dest_doc_ids) {
            Some(j) => j,
            None => {
                let mut out = [0u32; BLOCK_LEN + EXPAND_SLACK];
                let j = expand_bitset_scalar(words, hdr.base, &mut out);
                dest_doc_ids[..count].copy_from_slice(&out[..count]);
                j
            }
        };
        debug_assert_eq!(j, count, "bitset set-bit count must equal doc_count");
        return count;
    }

    if hdr.encoding == ENCODING_PATCHED {
        // Explicit deltas at the narrow width, exceptions patched in,
        // then the prefix sum the sorted packer would have fused. The tf
        // exceptions are never touched.
        decode_patched_doc_ids(bytes, hdr, dest_doc_ids);
        return count;
    }

    let deltas_size = hdr.deltas_size();
    assert!(
        bytes.len() >= hdr.payload() + deltas_size,
        "decode_block_doc_ids: bytes ({}) shorter than header+deltas ({})",
        bytes.len(),
        hdr.payload() + deltas_size
    );
    unpack_sorted(
        hdr.base,
        &bytes[hdr.payload()..hdr.payload() + deltas_size],
        &mut dest_doc_ids[..hdr.lanes()],
        hdr.delta_bits,
    );

    count
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode with fresh scratch.
    fn encode_one(b: &Block, prev_last_doc_id: Option<u32>, patchable: bool) -> EncodedBlock {
        encode_block_with(
            b,
            prev_last_doc_id,
            patchable,
            1.0,
            &mut PackScratch::default(),
        )
    }

    /// Build a Block from parallel slices.
    fn block(doc_ids: &[u32], tfs: &[u32]) -> Block {
        Block {
            doc_ids: doc_ids.to_vec(),
            tfs: tfs.to_vec(),
        }
    }

    /// Encode as a term's first block and decode it back.
    fn roundtrip_with(b: &Block, prev: Option<u32>) -> EncodedBlock {
        let enc = encode_one(b, prev, true);
        let hdr = BlockHeader::parse(&enc.bytes, prev, true);
        assert_eq!(hdr.count(), b.doc_ids.len());
        let mut got_doc_ids = vec![0u32; BLOCK_LEN];
        let mut got_tfs = vec![0u32; BLOCK_LEN];
        let count = decode_block(&enc.bytes, &hdr, &mut got_doc_ids, &mut got_tfs);
        assert_eq!(count, b.doc_ids.len(), "doc_count round-trip");
        assert_eq!(
            &got_doc_ids[..count],
            b.doc_ids.as_slice(),
            "doc_ids round-trip (prev {prev:?})"
        );
        assert_eq!(&got_tfs[..count], b.tfs.as_slice(), "tfs round-trip");
        // The two half decodes agree with the whole one.
        let mut ids = vec![0u32; BLOCK_LEN];
        assert_eq!(decode_block_doc_ids(&enc.bytes, &hdr, &mut ids), count);
        assert_eq!(&ids[..count], b.doc_ids.as_slice());
        let mut tfs = vec![0u32; BLOCK_LEN];
        decode_block_tfs(&enc.bytes, &hdr, &mut tfs);
        assert_eq!(&tfs[..count], b.tfs.as_slice());
        enc
    }

    /// Round-trip as a term's first block.
    fn roundtrip(b: &Block) -> EncodedBlock {
        roundtrip_with(b, None)
    }

    /// Encode as a term's first block.
    fn encode_first(b: &Block) -> EncodedBlock {
        encode_one(b, None, true)
    }

    // --- Basic round-trips ----------------------------------------------

    #[test]
    fn outliers_take_the_patched_encoding_and_shrink_the_block() {
        // 127 docs one apart and one 1M gap: plain packing needs 20-bit
        // deltas for every lane (320 B); patched packs 1-bit deltas (16 B)
        // and one exception. Same for a lone tf of 900 among ones.
        let mut doc_ids: Vec<u32> = (1000..1127).collect();
        doc_ids.push(1_126 + 1_000_000);
        let mut tfs = vec![1u32; 128];
        tfs[40] = 900;
        // As a later block: the base is the previous block's last doc.
        let enc = roundtrip_with(&block(&doc_ids, &tfs), Some(999));
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PATCHED);
        assert!(enc.bytes.len() < 64, "got {} bytes", enc.bytes.len());
        let hdr = BlockHeader::parse(&enc.bytes, Some(999), true);
        assert_eq!((hdr.n_delta_exc(), hdr.n_tf_exc()), (1, 1));
        assert_eq!(hdr.base, 999);
    }

    #[test]
    fn a_first_block_absorbs_its_doc_id_as_one_exception() {
        // A packed block has no stored base, so a first block's first
        // delta is the doc id itself — patched away as one exception
        // rather than widening every lane.
        let doc_ids: Vec<u32> = (0..128).map(|i| 500_000 + 3 * i).collect();
        let enc = roundtrip(&block(&doc_ids, &[1; 128]));
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PATCHED);
        let hdr = BlockHeader::parse(&enc.bytes, None, true);
        assert_eq!(hdr.base, 0);
        assert_eq!(hdr.delta_bits, 2);
        assert_eq!(hdr.n_delta_exc(), 1);
    }

    #[test]
    fn a_block_the_writer_may_not_patch_stays_plain() {
        // The same outlier block, with patching withheld for its term:
        // plain packing at the outlier's width, decoded by the fused
        // kernel — what a common term's blocks get.
        let mut doc_ids: Vec<u32> = (1000..1127).collect();
        doc_ids.push(1_126 + 1_000_000);
        let b = block(&doc_ids, &[1; 128]);
        let plain = encode_one(&b, Some(999), false);
        assert_eq!(block_encoding(&plain.bytes), ENCODING_PACKED);
        let patched = encode_one(&b, Some(999), true);
        assert_eq!(block_encoding(&patched.bytes), ENCODING_PATCHED);
        assert!(plain.bytes.len() > patched.bytes.len());
        let hdr = BlockHeader::parse(&plain.bytes, Some(999), true);
        let mut d = vec![0u32; BLOCK_LEN];
        assert_eq!(decode_block_doc_ids(&plain.bytes, &hdr, &mut d), 128);
        assert_eq!(&d[..128], doc_ids.as_slice());
    }

    #[test]
    fn uniform_blocks_stay_packed_byte_for_byte() {
        // No outlier ⇒ the patched plan cannot beat plain, so the block
        // is the PACKED layout.
        let doc_ids: Vec<u32> = (0..128).map(|i| 10 + 37 * i).collect();
        let tfs: Vec<u32> = (0..128).map(|i| 1 + i % 4).collect();
        let enc = roundtrip_with(&block(&doc_ids, &tfs), Some(9));
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PACKED);
        // 6-bit deltas (37) and 2-bit tfs (stored minus one, so up to 3),
        // nothing else.
        assert_eq!(enc.bytes.len(), COMPACT_HEADER_SIZE + 16 * 6 + 16 * 2);
    }

    #[test]
    fn partial_patched_block_round_trips() {
        let doc_ids = vec![5u32, 6, 7, 5_000_000];
        let tfs = vec![1u32, 1, 1, 1];
        let enc = roundtrip(&block(&doc_ids, &tfs));
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PATCHED);
    }

    #[test]
    fn roundtrip_full_block_dense() {
        // 128 sequential docs, all tf=1: the densest possible posting
        // (1-bit deltas, 1-bit tfs).
        let doc_ids: Vec<u32> = (1000..1128).collect();
        let tfs = vec![1u32; 128];
        let enc = roundtrip(&block(&doc_ids, &tfs));
        assert_eq!(enc.last_doc_id, 1127);
        assert_eq!(enc.max_tf, 1);
    }

    #[test]
    fn roundtrip_dense_block_uses_bitset_encoding() {
        // 128 docs over 129 ids pack to 2-bit deltas (32 B); the bitset is
        // 3 words (24 B), so the bitset wins.
        let doc_ids: Vec<u32> = (64..193).filter(|&d| d != 100).collect();
        let tfs = vec![1u32; 128];
        let enc = roundtrip_with(&block(&doc_ids, &tfs), Some(40));
        assert_eq!(block_encoding(&enc.bytes), ENCODING_BITSET);
        let hdr = BlockHeader::parse(&enc.bytes, Some(40), true);
        assert_eq!(hdr.base, 64, "origin");
        assert_eq!(hdr.delta_bits, 0);
    }

    #[test]
    fn blocks_written_before_tfs_were_stored_minus_one_decode_unchanged() {
        // A block from before this format stores each tf as is and leaves
        // header bit 31 clear. Encoding `tf + 1` stores exactly the old
        // values; clearing the bit then makes the old block, which every tf
        // reader must return unchanged, in each encoding.
        let shapes: [(Vec<u32>, bool); 4] = [
            ((0..128).map(|i| 10 + 37 * i).collect(), false), // packed
            ((64..193).filter(|&d| d != 100).collect(), false), // bitset
            (
                (0..128)
                    .map(|i| 3 * i + if i >= 77 { 9_000_000 } else { 0 })
                    .collect(),
                true,
            ), // patched
            ((500..628).collect(), false),                    // run
        ];
        for (docs, patchable) in shapes {
            let tfs: Vec<u32> = (0..docs.len() as u32).map(|i| 1 + i % 5).collect();
            let shifted: Vec<u32> = tfs.iter().map(|t| t + 1).collect();
            let mut enc = encode_one(&block(&docs, &shifted), None, patchable);
            enc.bytes[3] &= 0x7F;
            // A file from before wide blocks reads bit 31 as the flag.
            let hdr = BlockHeader::parse(&enc.bytes, None, false);
            assert_eq!(hdr.tf_bias, 0, "legacy block");
            let mut d = vec![0u32; BLOCK_LEN];
            let mut t = vec![0u32; BLOCK_LEN];
            let n = decode_block(&enc.bytes, &hdr, &mut d, &mut t);
            assert_eq!(&d[..n], docs.as_slice(), "encoding {}", hdr.encoding);
            assert_eq!(
                &t[..n],
                tfs.as_slice(),
                "decode_block, encoding {}",
                hdr.encoding
            );
            let mut t2 = vec![0u32; BLOCK_LEN];
            decode_block_tfs(&enc.bytes, &hdr, &mut t2);
            assert_eq!(
                &t2[..n],
                tfs.as_slice(),
                "decode_block_tfs, encoding {}",
                hdr.encoding
            );
            if is_dense_encoding(hdr.encoding) {
                for (rank, &tf) in tfs.iter().enumerate() {
                    assert_eq!(dense_tf_at(&enc.bytes, &hdr, rank), tf, "rank {rank}");
                }
            }
            // The same block as this crate writes it decodes identically.
            let enc = encode_one(&block(&docs, &tfs), None, patchable);
            let hdr = BlockHeader::parse(&enc.bytes, None, true);
            assert_eq!(hdr.tf_bias, 1, "current block");
            let n = decode_block(&enc.bytes, &hdr, &mut d, &mut t);
            assert_eq!(
                &t[..n],
                tfs.as_slice(),
                "current, encoding {}",
                hdr.encoding
            );
        }
    }

    #[test]
    fn a_bitset_bias_takes_the_bitset_past_the_size_break_even() {
        // Docs two or three apart: 2-bit deltas pack to 32 B, the presence
        // bitset over ~320 ids is 6 words (48 B). The size-only rule packs;
        // a bias of 2 takes the bitset. Both decode to the same block.
        let docs: Vec<u32> = (0..128u32).map(|i| 64 + i * 2 + i / 2).collect();
        let tfs: Vec<u32> = (0..128).map(|i| 1 + i % 3).collect();
        let b = block(&docs, &tfs);
        let plain = encode_block_with(&b, Some(63), false, 1.0, &mut PackScratch::default());
        let biased = encode_block_with(&b, Some(63), false, 2.0, &mut PackScratch::default());
        assert_eq!(block_encoding(&plain.bytes), ENCODING_PACKED);
        assert_eq!(block_encoding(&biased.bytes), ENCODING_BITSET);
        assert!(biased.bytes.len() > plain.bytes.len());
        for enc in [&plain, &biased] {
            let hdr = BlockHeader::parse(&enc.bytes, Some(63), true);
            let (mut d, mut t) = (vec![0u32; BLOCK_LEN], vec![0u32; BLOCK_LEN]);
            let n = decode_block(&enc.bytes, &hdr, &mut d, &mut t);
            assert_eq!(&d[..n], docs.as_slice());
            assert_eq!(&t[..n], tfs.as_slice());
        }
    }

    #[test]
    fn consecutive_docs_store_no_doc_payload() {
        // A run of consecutive ids, full or partial, first block or not:
        // the origin word holds the first doc and nothing else is stored
        // for the doc ids. All-ones tfs add nothing either.
        for (ids, prev) in [(64u32..192, Some(40)), (5..9, None), (7..8, Some(3))] {
            let doc_ids: Vec<u32> = ids.collect();
            let ones = vec![1u32; doc_ids.len()];
            let enc = roundtrip_with(&block(&doc_ids, &ones), prev);
            assert_eq!(block_encoding(&enc.bytes), ENCODING_RUN);
            let hdr = BlockHeader::parse(&enc.bytes, prev, true);
            assert_eq!(hdr.base, doc_ids[0], "origin is the first doc");
            assert_eq!(enc.bytes.len(), COMPACT_HEADER_SIZE + ORIGIN_BYTES);
            let twos = vec![2u32; doc_ids.len()];
            let enc = roundtrip_with(&block(&doc_ids, &twos), prev);
            assert_eq!(block_encoding(&enc.bytes), ENCODING_RUN);
            assert_eq!(enc.bytes.len(), COMPACT_HEADER_SIZE + ORIGIN_BYTES + 16);
        }
    }

    #[test]
    fn bitset_probe_helpers_agree_with_the_full_expansion() {
        // A dense block with a gap pattern: every target between the base
        // and past the last doc must resolve to the same (doc, rank) the
        // expanded arrays give, and the lane read must equal the unpacked tf
        // for every width the packer produces.
        let doc_ids: Vec<u32> = (0..128u32).map(|i| 1000 + i * 3 + (i % 5)).collect();
        let doc_ids: Vec<u32> = {
            let mut v = doc_ids;
            v.sort_unstable();
            v.dedup();
            v
        };
        for width in 0..=20u32 {
            let cap = (1u64 << width) as u32;
            let tfs: Vec<u32> = (0..doc_ids.len() as u32)
                .map(|i| match width {
                    0 => 1,
                    _ => 1 + (i.wrapping_mul(2_654_435_761).wrapping_add(i * 7) % (cap - 1).max(1)),
                })
                .collect();
            let enc = encode_one(&block(&doc_ids, &tfs), Some(900), true);
            let hdr = BlockHeader::parse(&enc.bytes, Some(900), true);
            if hdr.encoding != ENCODING_BITSET {
                continue;
            }
            let mut ids = vec![0u32; BLOCK_LEN];
            let mut got_tfs = vec![0u32; BLOCK_LEN];
            let n = decode_block(&enc.bytes, &hdr, &mut ids, &mut got_tfs);
            for target in (hdr.base - 5)..=(doc_ids[n - 1] + 3) {
                let want = ids[..n].iter().position(|&d| d >= target);
                let got = dense_next_doc(&enc.bytes, &hdr, target);
                match want {
                    None => assert!(got.is_none(), "target {target}"),
                    Some(r) => {
                        let (doc, rank) = got.expect("a doc >= target");
                        assert_eq!((doc, rank), (ids[r], r), "target {target}");
                        assert_eq!(dense_tf_at(&enc.bytes, &hdr, rank), got_tfs[r]);
                    }
                }
            }
        }
    }

    #[test]
    fn run_probe_helpers_agree_with_the_full_expansion() {
        // A run block answers the one-doc probes by arithmetic: every
        // target around and inside the run lands where the expansion says.
        let doc_ids: Vec<u32> = (500..560).collect();
        let tfs: Vec<u32> = (0..doc_ids.len() as u32).map(|i| 1 + i % 6).collect();
        let enc = encode_one(&block(&doc_ids, &tfs), Some(10), true);
        let hdr = BlockHeader::parse(&enc.bytes, Some(10), true);
        assert_eq!(hdr.encoding, ENCODING_RUN);
        for target in 490..570 {
            let want = doc_ids.iter().position(|&d| d >= target);
            let got = dense_next_doc(&enc.bytes, &hdr, target);
            assert_eq!(got, want.map(|r| (doc_ids[r], r)), "target {target}");
            if let Some(r) = want {
                assert_eq!(dense_tf_at(&enc.bytes, &hdr, r), tfs[r], "rank {r}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "run block passes u32::MAX")]
    fn a_run_block_whose_ids_pass_u32_max_is_refused() {
        // A corrupt origin four short of `u32::MAX` under a five-doc run:
        // the last id does not fit, and must not wrap to small ids.
        let doc_ids: Vec<u32> = (5..10).collect();
        let mut enc = encode_one(&block(&doc_ids, &[1; 5]), None, false);
        assert_eq!(block_encoding(&enc.bytes), ENCODING_RUN);
        enc.bytes[COMPACT_HEADER_SIZE..COMPACT_HEADER_SIZE + ORIGIN_BYTES]
            .copy_from_slice(&(u32::MAX - 3).to_le_bytes());
        let hdr = BlockHeader::parse(&enc.bytes, None, true);
        let mut ids = vec![0u32; BLOCK_LEN];
        decode_block_doc_ids(&enc.bytes, &hdr, &mut ids);
    }

    #[test]
    #[should_panic(expected = "bit width > 32")]
    fn bitset_probe_refuses_a_header_width_past_32() {
        // A bitset block whose header claims tf width 33: the probe must
        // fail the width check, not underflow the tf-size arithmetic.
        let doc_ids: Vec<u32> = (256..385).filter(|&d| d != 300).collect();
        let tfs = vec![1u32; 128];
        let mut enc = encode_one(&block(&doc_ids, &tfs), Some(200), false);
        assert_eq!(block_encoding(&enc.bytes), ENCODING_BITSET);
        let mut word = u32::from_le_bytes(enc.bytes[..4].try_into().expect("4 bytes"));
        let mask = ((1u32 << HDR_WIDTH_BITS) - 1) << HDR_TF_BITS_SHIFT;
        word = (word & !mask) | (33 << HDR_TF_BITS_SHIFT);
        enc.bytes[..4].copy_from_slice(&word.to_le_bytes());
        let hdr = BlockHeader::parse(&enc.bytes, Some(200), true);
        assert_eq!(hdr.tf_bits, 33);
        let _ = dense_next_doc(&enc.bytes, &hdr, 256);
    }

    #[test]
    #[should_panic(expected = "bit width > 32")]
    fn decoders_refuse_a_header_width_past_32() {
        // The width check moved from every header parse to the decoders
        // that consume the widths; a corrupt compact header still cannot
        // reach an unpacker.
        let doc_ids: Vec<u32> = (0..128u32).map(|i| 10 + i * 7).collect();
        let tfs = vec![1u32; 128];
        let mut enc = encode_one(&block(&doc_ids, &tfs), None, false);
        let mut word = u32::from_le_bytes(enc.bytes[..4].try_into().expect("4 bytes"));
        let mask = ((1u32 << HDR_WIDTH_BITS) - 1) << HDR_DELTA_BITS_SHIFT;
        word = (word & !mask) | (33 << HDR_DELTA_BITS_SHIFT);
        enc.bytes[..4].copy_from_slice(&word.to_le_bytes());
        let hdr = BlockHeader::parse(&enc.bytes, None, true);
        assert_eq!(hdr.delta_bits, 33, "the parse itself no longer rejects it");
        let mut ids = vec![0u32; BLOCK_LEN];
        decode_block_doc_ids(&enc.bytes, &hdr, &mut ids);
    }

    #[test]
    fn roundtrip_bitset_block_nonzero_aligned_base() {
        // A dense run far from zero: the origin is the word holding the
        // block's first doc, stored whether the block is
        // a term's first or follows another far behind it.
        let doc_ids: Vec<u32> = (960..1089).filter(|&d| d != 1000).collect();
        let tfs = vec![2u32; 128];
        for prev in [None, Some(17)] {
            let enc = roundtrip_with(&block(&doc_ids, &tfs), prev);
            assert_eq!(block_encoding(&enc.bytes), ENCODING_BITSET);
            let hdr = BlockHeader::parse(&enc.bytes, prev, true);
            assert_eq!(hdr.base, 960);
            assert_eq!(hdr.payload(), 8);
        }
    }

    #[test]
    fn roundtrip_sparse_block_stays_packed() {
        let doc_ids: Vec<u32> = (0..128).map(|i| 1 + i * 100_000).collect();
        let tfs = vec![1u32; 128];
        let enc = roundtrip(&block(&doc_ids, &tfs));
        assert_ne!(block_encoding(&enc.bytes), ENCODING_BITSET);
    }

    #[test]
    fn roundtrip_partial_block_single_doc() {
        roundtrip(&block(&[42], &[7]));
    }

    #[test]
    fn roundtrip_partial_block_50_docs() {
        let doc_ids: Vec<u32> = (0..50).map(|i| 3 + i * 9).collect();
        let tfs: Vec<u32> = (0..50).map(|i| 1 + i % 3).collect();
        roundtrip(&block(&doc_ids, &tfs));
    }

    #[test]
    fn roundtrip_first_doc_is_zero() {
        roundtrip(&block(&[0, 1, 5], &[1, 1, 1]));
    }

    #[test]
    fn roundtrip_first_doc_is_u32_max_minus_n() {
        let doc_ids: Vec<u32> = (0..10).map(|i| u32::MAX - 20 + i).collect();
        let tfs = vec![1u32; 10];
        roundtrip(&block(&doc_ids, &tfs));
        roundtrip_with(&block(&doc_ids, &tfs), Some(u32::MAX - 21));
    }

    #[test]
    fn a_later_block_follows_the_previous_last_doc() {
        // Consecutive blocks of one term, each decoded from the previous
        // block's last doc id.
        let all: Vec<u32> = (0..300u32).map(|i| 7 + i * 5).collect();
        let mut prev = None;
        let mut recovered = Vec::new();
        for chunk in all.chunks(BLOCK_LEN) {
            let enc = roundtrip_with(&block(chunk, &vec![1; chunk.len()]), prev);
            let hdr = BlockHeader::parse(&enc.bytes, prev, true);
            let mut d = vec![0u32; BLOCK_LEN];
            let n = decode_block_doc_ids(&enc.bytes, &hdr, &mut d);
            recovered.extend_from_slice(&d[..n]);
            prev = Some(enc.last_doc_id);
        }
        assert_eq!(recovered, all);
    }

    // --- Bit-width edge cases ---------------------------------------------

    #[test]
    fn bit_width_1_for_dense_postings() {
        // Tfs are stored minus one: tfs of 1 need no bits, tfs of 2 one.
        let doc_ids: Vec<u32> = (100..228).collect();
        for (tf, want) in [(1u32, 0u8), (2, 1)] {
            let tfs = vec![tf; 128];
            let hdr = BlockHeader::parse(
                &encode_one(&block(&doc_ids, &tfs), Some(99), true).bytes,
                Some(99),
                true,
            );
            assert_eq!(hdr.tf_bits, want, "tf {tf}");
        }
    }

    #[test]
    fn bit_width_7_just_below_byte_boundary() {
        let doc_ids: Vec<u32> = (0..128).map(|i| i * 127).collect(); // deltas of 127 → 7 bits
        let tfs = vec![127u32; 128];
        let enc = roundtrip(&block(&doc_ids, &tfs));
        let hdr = BlockHeader::parse(&enc.bytes, None, true);
        assert_eq!((hdr.delta_bits, hdr.tf_bits), (7, 7));
    }

    #[test]
    fn bit_width_8_at_byte_boundary() {
        let doc_ids: Vec<u32> = (0..128).map(|i| i * 255).collect(); // deltas of 255 → 8 bits
        let tfs = vec![255u32; 128];
        let enc = roundtrip(&block(&doc_ids, &tfs));
        let hdr = BlockHeader::parse(&enc.bytes, None, true);
        assert_eq!((hdr.delta_bits, hdr.tf_bits), (8, 8));
    }

    #[test]
    fn widths_31_and_32_round_trip_losslessly() {
        // The widest deltas and tfs the codec stores: plain packing takes
        // them at full width, and the patched path packs the same values.
        for (last, want_bits) in [((1u32 << 31) - 1, 31u8), (u32::MAX, 32)] {
            let b = block(&[0, last], &[u32::MAX, 1]);
            let plain = encode_one(&b, None, false);
            let hdr = BlockHeader::parse(&plain.bytes, None, true);
            assert_eq!((hdr.delta_bits, hdr.tf_bits), (want_bits, 32));
            roundtrip_with(&b, None);
        }
        // A full block of 32-bit tfs stays plain.
        let doc_ids: Vec<u32> = (0..128).collect();
        let enc = roundtrip(&block(&doc_ids, &[u32::MAX - 5; 128]));
        let hdr = BlockHeader::parse(&enc.bytes, None, true);
        assert_eq!(hdr.tf_bits, 32);
    }

    #[test]
    fn bit_width_0_for_all_one_tfs() {
        // Every posting's tf is at least one and is stored minus one, so a
        // block whose docs each hold the term once packs at width 0 (no tf
        // bytes at all).
        let doc_ids: Vec<u32> = (0..128).map(|i| i * 2).collect();
        let tfs = vec![1u32; 128];
        let enc = roundtrip(&block(&doc_ids, &tfs));
        let hdr = BlockHeader::parse(&enc.bytes, None, true);
        assert_eq!(hdr.tf_bits, 0);
        assert_eq!(hdr.tfs_size(), 0);
    }

    // --- Header layout / metadata --------------------------------------

    #[test]
    fn encoded_block_carries_last_doc_id_and_max_tf() {
        let doc_ids: Vec<u32> = vec![5, 10, 15, 20, 25];
        let tfs: Vec<u32> = vec![1, 4, 2, 9, 3];
        let enc = encode_first(&block(&doc_ids, &tfs));
        assert_eq!(enc.last_doc_id, 25);
        assert_eq!(enc.max_tf, 9);
    }

    #[test]
    fn header_doc_count_round_trips() {
        for count in [1usize, 2, 31, 32, 33, 63, 64, 65, 127, 128] {
            let doc_ids: Vec<u32> = (1..=count as u32).collect();
            let tfs = vec![1u32; count];
            let enc = encode_one(&block(&doc_ids, &tfs), None, true);
            assert_eq!(
                BlockHeader::parse(&enc.bytes, None, true).count(),
                count,
                "header.doc_count for n={count}"
            );
        }
    }

    #[test]
    fn header_doc_count_round_trips_past_a_sub_block() {
        for count in [129usize, 130, 191, 192, 193, 255, 256] {
            let doc_ids: Vec<u32> = (1..=count as u32).collect();
            let tfs = vec![1u32; count];
            let enc = encode_one(&block(&doc_ids, &tfs), None, true);
            let hdr = BlockHeader::parse(&enc.bytes, None, true);
            assert_eq!(hdr.count(), count, "header.doc_count for n={count}");
            assert_eq!(hdr.lanes(), BLOCK_LEN, "two sub-blocks for n={count}");
        }
    }

    /// A block of more than one sub-block round-trips in every encoding,
    /// and its probes (one tf, next doc) answer past the first sub-block.
    #[test]
    fn wide_blocks_round_trip_in_every_encoding() {
        for count in [129usize, 200, 255, 256] {
            let n = count as u32;
            let shapes: Vec<(Vec<u32>, Vec<u32>, bool, u8)> = vec![
                // Packed: a uniform gap no bitset or patch beats.
                (
                    (0..n).map(|i| 1000 + 37 * i).collect(),
                    (0..n).map(|i| 1 + i % 5).collect(),
                    false,
                    ENCODING_PACKED,
                ),
                // Patched: one outlier gap and one outlier tf.
                (
                    (0..n)
                        .map(|i| 10 + 3 * i + if i >= n - 10 { 9_000_000 } else { 0 })
                        .collect(),
                    (0..n)
                        .map(|i| if i == 7 { 40 } else { 1 + i % 3 })
                        .collect(),
                    true,
                    ENCODING_PATCHED,
                ),
                // Bitset: dense with every eleventh doc missing.
                (
                    (64u32..).filter(|d| d % 11 != 5).take(count).collect(),
                    (0..n).map(|i| 1 + i % 7).collect(),
                    false,
                    ENCODING_BITSET,
                ),
                // Run: consecutive.
                (
                    (500..500 + n).collect(),
                    (0..n).map(|i| 1 + i % 4).collect(),
                    false,
                    ENCODING_RUN,
                ),
            ];
            for (doc_ids, tfs, patchable, want) in shapes {
                let prev = Some(doc_ids[0] - 7);
                let enc = encode_one(&block(&doc_ids, &tfs), prev, patchable);
                let hdr = BlockHeader::parse(&enc.bytes, prev, true);
                assert_eq!(hdr.encoding, want, "n={count}: premise, the encoding");
                assert_eq!(hdr.count(), count, "n={count}");
                assert_eq!(hdr.lanes(), BLOCK_LEN, "n={count}");
                let mut d = vec![0u32; BLOCK_LEN];
                let mut t = vec![0u32; BLOCK_LEN];
                let got = decode_block(&enc.bytes, &hdr, &mut d, &mut t);
                assert_eq!(got, count, "n={count} enc {want}");
                assert_eq!(&d[..count], doc_ids.as_slice(), "n={count} enc {want}: ids");
                assert_eq!(&t[..count], tfs.as_slice(), "n={count} enc {want}: tfs");
                let mut d2 = vec![0u32; BLOCK_LEN];
                assert_eq!(decode_block_doc_ids(&enc.bytes, &hdr, &mut d2), count);
                assert_eq!(
                    &d2[..count],
                    doc_ids.as_slice(),
                    "n={count} enc {want}: ids alone"
                );
                let mut t2 = vec![0u32; BLOCK_LEN];
                decode_block_tfs(&enc.bytes, &hdr, &mut t2);
                assert_eq!(
                    &t2[..count],
                    tfs.as_slice(),
                    "n={count} enc {want}: tfs alone"
                );
                if is_dense_encoding(want) {
                    for rank in [0usize, 127, 128, 129, count - 1]
                        .into_iter()
                        .filter(|&r| r < count)
                    {
                        assert_eq!(
                            dense_tf_at(&enc.bytes, &hdr, rank),
                            tfs[rank],
                            "n={count} enc {want}: tf at rank {rank}"
                        );
                    }
                    for &doc in [doc_ids[0], doc_ids[127], doc_ids[128], doc_ids[count - 1]].iter()
                    {
                        let rank = doc_ids.iter().position(|&x| x == doc).expect("present");
                        assert_eq!(
                            dense_next_doc(&enc.bytes, &hdr, doc),
                            Some((doc, rank)),
                            "n={count} enc {want}: next doc at {doc}"
                        );
                    }
                    assert_eq!(
                        dense_next_doc(&enc.bytes, &hdr, doc_ids[count - 1] + 1),
                        None
                    );
                }
            }
        }
    }

    /// A header from a file written before wide blocks reads bit 31 as the
    /// tfs-minus-one flag, and a wide file's reader reads the same bit as
    /// the count's eighth.
    #[test]
    fn bit_31_reads_by_the_file_not_the_block() {
        let doc_ids: Vec<u32> = (0..100).map(|i| 5 + 9 * i).collect();
        let tfs: Vec<u32> = (0..100).map(|i| 2 + i % 3).collect();
        let mut enc = encode_one(&block(&doc_ids, &tfs), None, false);
        // 100 docs: bit 31 is clear as written.
        assert_eq!(enc.bytes[3] & 0x80, 0);
        enc.bytes[3] |= 0x80;
        let old = BlockHeader::parse(&enc.bytes, None, false);
        assert_eq!((old.count(), old.tf_bias), (100, 1), "legacy: the flag");
        let wide = BlockHeader::parse(&enc.bytes, None, true);
        assert_eq!(
            (wide.count(), wide.tf_bias),
            (228, 1),
            "wide: count bit seven"
        );
    }

    /// A block from a version-10 file: one sub-block, bit 31 the flag that
    /// its tfs are stored minus one. Read by the old reading it decodes as
    /// written.
    #[test]
    fn a_version_ten_block_decodes_by_the_old_reading() {
        let mut dense = 0;
        for (doc_ids, patchable) in [
            ((0..100u32).map(|i| 5 + 9 * i).collect::<Vec<_>>(), false),
            ((500..628u32).collect::<Vec<_>>(), false),
            (
                (0..90u32)
                    .map(|i| 7 + 3 * i + if i >= 80 { 9_000_000 } else { 0 })
                    .collect::<Vec<_>>(),
                true,
            ),
        ] {
            let n = doc_ids.len();
            let tfs: Vec<u32> = (0..n as u32).map(|i| 1 + i % 4).collect();
            let mut enc = encode_one(&block(&doc_ids, &tfs), Some(2), patchable);
            // This writer sets bit 31 only for a count past one sub-block.
            assert_eq!(enc.bytes[3] & 0x80, 0);
            enc.bytes[3] |= 0x80;
            let hdr = BlockHeader::parse(&enc.bytes, Some(2), false);
            assert_eq!(
                (hdr.count(), hdr.lanes(), hdr.tf_bias),
                (n, SUB_BLOCK_LEN, 1)
            );
            let mut d = vec![0u32; BLOCK_LEN];
            let mut t = vec![0u32; BLOCK_LEN];
            assert_eq!(decode_block(&enc.bytes, &hdr, &mut d, &mut t), n);
            assert_eq!(&d[..n], doc_ids.as_slice());
            assert_eq!(&t[..n], tfs.as_slice());
            let mut ids = vec![0u32; BLOCK_LEN];
            assert_eq!(decode_block_doc_ids(&enc.bytes, &hdr, &mut ids), n);
            assert_eq!(&ids[..n], doc_ids.as_slice());
            if is_dense_encoding(hdr.encoding) {
                dense += 1;
                for (r, &tf) in tfs.iter().enumerate() {
                    assert_eq!(dense_tf_at(&enc.bytes, &hdr, r), tf, "rank {r}");
                }
            }
        }
        assert!(dense > 0, "premise: a dense block is among the cases");
    }

    #[test]
    fn header_encoding_bits_name_the_encoding() {
        // Byte 3's low two bits are the encoding: 0 =
        // PACKED (a full block of uniform wide deltas, where nothing beats
        // plain packing), 1 = BITSET (dense), 2 = PATCHED (a sparse partial
        // block: the padding lanes pack at width 0 and the real deltas
        // ride as exceptions).
        let uniform: Vec<u32> = (0..128).map(|i| 1 + 100_000 * i).collect();
        let packed = encode_one(&block(&uniform, &[1; 128]), Some(0), true);
        assert_eq!(
            block_encoding(&packed.bytes),
            ENCODING_PACKED,
            "uniform ⇒ PACKED"
        );
        let bitset = encode_first(&block(&[1, 2, 4], &[1, 1, 1]));
        assert_eq!(
            block_encoding(&bitset.bytes),
            ENCODING_BITSET,
            "dense ⇒ BITSET"
        );
        let run = encode_first(&block(&[1, 2, 3], &[1, 1, 1]));
        assert_eq!(
            block_encoding(&run.bytes),
            ENCODING_RUN,
            "consecutive ⇒ RUN"
        );
        let patched = encode_first(&block(&[1, 100_000], &[1, 1]));
        assert_eq!(
            block_encoding(&patched.bytes),
            ENCODING_PATCHED,
            "sparse partial ⇒ PATCHED"
        );
        for enc in [&packed, &bitset, &patched, &run] {
            assert_eq!(
                BlockHeader::parse(&enc.bytes, Some(0), true).encoding,
                block_encoding(&enc.bytes)
            );
        }
    }

    #[test]
    fn compact_header_holds_every_field_at_its_extreme() {
        // Widths of 32, a full block, and the most exceptions either
        // stream may carry all fit the word and read back.
        let bytes = compact_header(128, 32, 32, ENCODING_PATCHED, 31, 31);
        let hdr = BlockHeader::parse(&bytes, Some(7), true);
        assert_eq!(
            (
                hdr.count(),
                hdr.delta_bits,
                hdr.tf_bits,
                hdr.encoding,
                hdr.n_delta_exc(),
                hdr.n_tf_exc(),
                hdr.base
            ),
            (128, 32, 32, ENCODING_PATCHED, 31, 31, 7)
        );
        assert_eq!(block_encoding(&bytes), ENCODING_PATCHED);
        let bytes = compact_header(1, 0, 0, ENCODING_PACKED, 0, 0);
        let hdr = BlockHeader::parse(&bytes, None, true);
        assert_eq!(
            (hdr.count(), hdr.delta_bits, hdr.tf_bits, hdr.base),
            (1, 0, 0, 0)
        );
    }

    // --- Mixed bit widths between deltas and tfs ------------------------

    #[test]
    fn delta_and_tf_use_independent_bit_widths() {
        // Wide deltas, narrow tfs — a full block of uniform lanes, so the
        // header carries the plain widths.
        let doc_ids: Vec<u32> = (0..128).map(|i| i * 1024).collect(); // delta = 1024 → 11 bits
        let tfs: Vec<u32> = (0..128).map(|_| 2).collect();
        let enc = roundtrip(&block(&doc_ids, &tfs));
        let hdr = BlockHeader::parse(&enc.bytes, None, true);
        assert!(
            (10..=12).contains(&hdr.delta_bits),
            "expected ~11 delta bits, got {}",
            hdr.delta_bits
        );
        // A tf of 2 is stored as 1: one bit.
        assert_eq!(hdr.tf_bits, 1);
    }

    // --- Panic surface for invalid input -------------------------------

    #[test]
    #[should_panic(expected = "empty block")]
    fn encode_block_panics_on_empty() {
        encode_first(&Block {
            doc_ids: vec![],
            tfs: vec![],
        });
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn encode_block_panics_on_length_mismatch() {
        encode_first(&Block {
            doc_ids: vec![1, 2, 3],
            tfs: vec![1, 2],
        });
    }

    #[test]
    #[should_panic(expected = "> BLOCK_LEN")]
    fn encode_block_panics_on_oversize() {
        let doc_ids: Vec<u32> = (0..BLOCK_LEN as u32 + 1).collect();
        let tfs = vec![1u32; BLOCK_LEN + 1];
        encode_first(&block(&doc_ids, &tfs));
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn header_parse_panics_on_short_input() {
        BlockHeader::parse(&[0u8; 3], None, true);
    }

    #[test]
    #[should_panic(expected = "shorter than header")]
    fn decode_block_panics_on_truncated_payload() {
        let enc = encode_first(&block(&[1, 100, 200], &[1, 1, 1]));
        let hdr = BlockHeader::parse(&enc.bytes, None, true);
        let mut d = vec![0u32; BLOCK_LEN];
        let mut t = vec![0u32; BLOCK_LEN];
        let _ = decode_block(&enc.bytes[..COMPACT_HEADER_SIZE + 1], &hdr, &mut d, &mut t);
    }

    #[test]
    #[should_panic(expected = "must have at least")]
    fn decode_block_panics_on_undersized_dest() {
        let enc = encode_first(&block(&[1, 2, 3], &[1, 1, 1]));
        let hdr = BlockHeader::parse(&enc.bytes, None, true);
        let mut d = vec![0u32; 10];
        let mut t = vec![0u32; BLOCK_LEN];
        let _ = decode_block(&enc.bytes, &hdr, &mut d, &mut t);
    }

    // --- Cross-block independence --------------------------------------

    #[test]
    fn blocks_with_disjoint_doc_id_ranges_decode_independently() {
        // Two blocks, each self-contained given its predecessor's last doc.
        let b1 = block(&[1, 2, 3, 4, 5], &[1, 2, 1, 2, 1]);
        let b2 = block(&[1000, 1001, 1010, 1100], &[5, 1, 3, 9]);
        let enc1 = encode_first(&b1);
        let enc2 = encode_one(&b2, Some(5), true);
        // Decode in opposite order to confirm zero shared state.
        let h2 = BlockHeader::parse(&enc2.bytes, Some(5), true);
        let mut d2 = vec![0u32; BLOCK_LEN];
        let mut t2 = vec![0u32; BLOCK_LEN];
        let n2 = decode_block(&enc2.bytes, &h2, &mut d2, &mut t2);
        let h1 = BlockHeader::parse(&enc1.bytes, None, true);
        let mut d1 = vec![0u32; BLOCK_LEN];
        let mut t1 = vec![0u32; BLOCK_LEN];
        let n1 = decode_block(&enc1.bytes, &h1, &mut d1, &mut t1);
        assert_eq!(&d1[..n1], &b1.doc_ids[..]);
        assert_eq!(&d2[..n2], &b2.doc_ids[..]);
        assert_eq!(&t1[..n1], &b1.tfs[..]);
        assert_eq!(&t2[..n2], &b2.tfs[..]);
    }

    // --- Stress: many blocks decode in order --------------------------

    #[test]
    fn decoding_many_blocks_in_sequence_recovers_full_list() {
        // Simulate a posting list of ~1000 docs split into ~8 blocks.
        let all_doc_ids: Vec<u32> = (0..1000u32).map(|i| i * 3 + 7).collect();
        let all_tfs: Vec<u32> = (0..1000u32).map(|i| (i % 5) + 1).collect();

        let mut encoded: Vec<EncodedBlock> = Vec::new();
        let mut prev = None;
        for (d, t) in all_doc_ids.chunks(BLOCK_LEN).zip(all_tfs.chunks(BLOCK_LEN)) {
            let enc = encode_one(&block(d, t), prev, true);
            prev = Some(enc.last_doc_id);
            encoded.push(enc);
        }

        let mut recovered_doc_ids = Vec::with_capacity(all_doc_ids.len());
        let mut recovered_tfs = Vec::with_capacity(all_tfs.len());
        let mut buf_d = vec![0u32; BLOCK_LEN];
        let mut buf_t = vec![0u32; BLOCK_LEN];
        let mut prev = None;
        for enc in &encoded {
            let hdr = BlockHeader::parse(&enc.bytes, prev, true);
            let n = decode_block(&enc.bytes, &hdr, &mut buf_d, &mut buf_t);
            recovered_doc_ids.extend_from_slice(&buf_d[..n]);
            recovered_tfs.extend_from_slice(&buf_t[..n]);
            prev = Some(enc.last_doc_id);
        }
        assert_eq!(recovered_doc_ids, all_doc_ids);
        assert_eq!(recovered_tfs, all_tfs);
    }

    // ---- Property tests ----
    //
    // Random sorted-ascending `Vec<u32>` and matching `tfs`
    // round-trip losslessly through `encode_block` /
    // `decode_block`. Covers bit widths the explicit-value
    // tests above only pin at specific points.

    use proptest::prelude::*;

    /// Sorted, strictly-ascending `Vec<u32>` of length `1..=BLOCK_LEN`.
    fn sorted_doc_ids() -> impl Strategy<Value = Vec<u32>> {
        (1usize..=BLOCK_LEN).prop_flat_map(|n| {
            // Cap each delta at u32::MAX / n so the cumulative
            // sum can't overflow.
            let max_delta = (u32::MAX / n.max(1) as u32).max(1);
            let deltas = prop::collection::vec(1u32..=max_delta, n);
            (0u32..1024, deltas).prop_map(|(start, ds)| {
                let mut v = Vec::with_capacity(ds.len());
                let mut acc = start;
                for d in ds {
                    acc = acc.saturating_add(d);
                    v.push(acc);
                }
                v
            })
        })
    }

    proptest! {
        #[test]
        fn prop_roundtrip(
            doc_ids in sorted_doc_ids(),
            tf_seed in any::<u64>(),
            as_later_block in any::<bool>(),
        ) {
            // Matching tfs from a seeded xorshift so length
            // matches doc_ids; tf in 1..=4096 to bound bit
            // width.
            let mut rng = tf_seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let tfs: Vec<u32> = (0..doc_ids.len())
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    (rng & 0xFFF) as u32 + 1
                })
                .collect();
            // A later block follows some doc below its first.
            let prev = match (as_later_block, doc_ids[0]) {
                (true, first) if first > 0 => Some((first - 1) / 2),
                _ => None,
            };

            let block = Block { doc_ids: doc_ids.clone(), tfs: tfs.clone() };
            let enc = encode_one(&block, prev, true);

            prop_assert_eq!(enc.last_doc_id, *doc_ids.last().expect("last element"));
            prop_assert_eq!(enc.max_tf, *tfs.iter().max().expect("iter max"));

            let hdr = BlockHeader::parse(&enc.bytes, prev, true);
            let mut got_doc_ids = vec![0u32; BLOCK_LEN];
            let mut got_tfs = vec![0u32; BLOCK_LEN];
            let count = decode_block(&enc.bytes, &hdr, &mut got_doc_ids, &mut got_tfs);
            prop_assert_eq!(count, doc_ids.len());
            prop_assert_eq!(&got_doc_ids[..count], doc_ids.as_slice());
            prop_assert_eq!(&got_tfs[..count], tfs.as_slice());
        }

        /// On-disk byte length is determined by header bit
        /// widths — locks the contract so layout can't change
        /// silently.
        #[test]
        fn prop_byte_length_matches_header_widths(
            doc_ids in sorted_doc_ids(),
            tf_seed in any::<u64>(),
        ) {
            let mut rng = tf_seed | 1;
            let tfs: Vec<u32> = (0..doc_ids.len())
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    (rng & 0xFFFF) as u32 + 1
                })
                .collect();

            let enc = encode_first(&Block { doc_ids, tfs });
            let hdr = BlockHeader::parse(&enc.bytes, None, true);
            let tfs_size = hdr.tfs_size();
            match hdr.encoding {
                ENCODING_BITSET => {
                    // Doc ids are a whole number of 64-bit words; tfs trail.
                    prop_assert_eq!(hdr.delta_bits, 0, "bitset block has delta_bits 0");
                    let bitset_bytes = enc.bytes.len() - hdr.payload() - tfs_size;
                    prop_assert!(bitset_bytes >= 8 && bitset_bytes.is_multiple_of(8));
                }
                ENCODING_RUN => {
                    // Header and origin, no doc payload; tfs trail.
                    prop_assert_eq!(hdr.delta_bits, 0, "run block has delta_bits 0");
                    prop_assert_eq!(enc.bytes.len(), COMPACT_HEADER_SIZE + ORIGIN_BYTES + tfs_size);
                }
                ENCODING_PATCHED => {
                    let (delta_exc, tf_exc) = patched_exception_ranges(&enc.bytes, &hdr);
                    prop_assert_eq!(
                        enc.bytes.len(),
                        COMPACT_HEADER_SIZE
                            + (hdr.lanes() * hdr.delta_bits as usize) / 8
                            + delta_exc.len()
                            + tf_exc.len()
                            + tfs_size
                    );
                }
                _ => {
                    prop_assert_eq!(
                        enc.bytes.len(),
                        COMPACT_HEADER_SIZE + (hdr.lanes() * hdr.delta_bits as usize) / 8 + tfs_size
                    );
                }
            }
        }
    }

    // --- Patched exceptions at their limits -----------------------------

    /// Gap and tf the outlier lanes take in the exception-limit blocks.
    const LIMIT_OUTLIER_GAP: u32 = 1 << 20;
    const LIMIT_OUTLIER_TF: u32 = 900;
    /// The doc the exception-limit blocks follow: one below their first
    /// doc, so lane 0's delta is 1 like every non-outlier lane.
    const LIMIT_PREV_LAST_DOC: u32 = 1000;
    /// A word-aligned base for the bitset-edge block.
    const WORD_ALIGNED_BASE: u32 = 1024;

    /// A block of [`BLOCK_LEN`] lanes one apart and tf 1, except the listed
    /// outlier lanes (a wide gap, a large tf, or both on one lane).
    fn outlier_block(delta_lanes: &[usize], tf_lanes: &[usize]) -> Block {
        let mut doc_ids = Vec::with_capacity(BLOCK_LEN);
        let mut d = 1000u32;
        for lane in 0..BLOCK_LEN {
            // Non-outlier docs three apart: neither a run nor dense enough
            // for a bitset, so the delta and tf streams both meet the
            // patched planner.
            d += if delta_lanes.contains(&lane) {
                LIMIT_OUTLIER_GAP
            } else {
                3
            };
            doc_ids.push(d);
        }
        let tfs: Vec<u32> = (0..BLOCK_LEN)
            .map(|lane| {
                if tf_lanes.contains(&lane) {
                    LIMIT_OUTLIER_TF
                } else {
                    1
                }
            })
            .collect();
        block(&doc_ids, &tfs)
    }

    /// Exceptions in both streams at once, on shared and distinct lanes,
    /// round-trip and are counted separately in the header. A decoder
    /// that patched one stream's exceptions into the other, or read the
    /// tf exceptions from where the delta exceptions end without
    /// skipping them, would corrupt the lanes after the first exception.
    #[test]
    fn delta_and_tf_exceptions_coexist_in_one_block() {
        let b = outlier_block(&[10, 50, 100], &[5, 50, 120]);
        let enc = roundtrip_with(&b, Some(LIMIT_PREV_LAST_DOC));
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PATCHED);
        let hdr = BlockHeader::parse(&enc.bytes, Some(LIMIT_PREV_LAST_DOC), true);
        assert_eq!((hdr.n_delta_exc(), hdr.n_tf_exc()), (3, 3));
        let (delta_exc, tf_exc) = patched_exception_ranges(&enc.bytes, &hdr);
        assert!(!delta_exc.is_empty() && !tf_exc.is_empty());
    }

    /// The exception count is a five-bit header field, so a stream may
    /// carry at most `PATCHED_MAX_EXCEPTIONS` outliers: a block with
    /// exactly that many patches, one more falls back to plain packing
    /// at the outliers' width. Both decode to the planted lanes; an encoder
    /// that admitted a 32nd exception would overflow the header field.
    #[test]
    fn patched_exceptions_stop_at_the_header_field_limit() {
        let at_limit: Vec<usize> = (0..PATCHED_MAX_EXCEPTIONS).map(|i| 2 + 4 * i).collect();
        let enc = roundtrip_with(&outlier_block(&at_limit, &[]), Some(LIMIT_PREV_LAST_DOC));
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PATCHED);
        let hdr = BlockHeader::parse(&enc.bytes, Some(LIMIT_PREV_LAST_DOC), true);
        assert_eq!(hdr.n_delta_exc(), PATCHED_MAX_EXCEPTIONS);

        let past: Vec<usize> = (0..=PATCHED_MAX_EXCEPTIONS).map(|i| 2 + 4 * i).collect();
        let enc = roundtrip_with(&outlier_block(&past, &[]), Some(LIMIT_PREV_LAST_DOC));
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PACKED);

        // The same limit on the tf stream.
        let enc = roundtrip_with(&outlier_block(&[], &at_limit), Some(LIMIT_PREV_LAST_DOC));
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PATCHED);
        let hdr = BlockHeader::parse(&enc.bytes, Some(LIMIT_PREV_LAST_DOC), true);
        assert_eq!(hdr.n_tf_exc(), PATCHED_MAX_EXCEPTIONS);
        let enc = roundtrip_with(&outlier_block(&[], &past), Some(LIMIT_PREV_LAST_DOC));
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PACKED);
    }

    // --- Bitset probes at word edges ------------------------------------

    /// The bitset probes at the docs that sit on 64-bit word edges: the
    /// first and last bit of a word, and the two bits either side of a
    /// word boundary. `dense_next_doc` must not lose the last bit of a
    /// word to the mask of the next, and the rank (the tf index) must
    /// count every set bit of the earlier words. Absent docs on the
    /// other side of each edge must resolve to the next present one.
    #[test]
    fn bitset_probes_resolve_docs_on_word_edges() {
        // Present: a whole sub-block's range except four absent docs chosen
        // next to the edges, so every edge has a present and an absent
        // neighbour.
        let absent = [
            WORD_ALIGNED_BASE + 1,
            WORD_ALIGNED_BASE + 62,
            WORD_ALIGNED_BASE + 65,
            WORD_ALIGNED_BASE + 126,
        ];
        let doc_ids: Vec<u32> = (WORD_ALIGNED_BASE..WORD_ALIGNED_BASE + SUB_BLOCK_LEN as u32)
            .filter(|d| !absent.contains(d))
            .collect();
        let tfs: Vec<u32> = (0..doc_ids.len() as u32).map(|i| 1 + i % 7).collect();
        let enc = encode_one(&block(&doc_ids, &tfs), Some(WORD_ALIGNED_BASE - 24), true);
        let hdr = BlockHeader::parse(&enc.bytes, Some(WORD_ALIGNED_BASE - 24), true);
        assert_eq!(
            hdr.encoding, ENCODING_BITSET,
            "premise: a dense block is a bitset"
        );
        assert_eq!(
            hdr.base, WORD_ALIGNED_BASE,
            "premise: the origin is the first doc's word"
        );
        let rank_of = |doc: u32| doc_ids.iter().position(|&d| d == doc).expect("present");
        for edge in [
            WORD_ALIGNED_BASE,
            WORD_ALIGNED_BASE + 63,
            WORD_ALIGNED_BASE + 64,
            WORD_ALIGNED_BASE + 127,
        ] {
            let rank = rank_of(edge);
            assert_eq!(
                dense_next_doc(&enc.bytes, &hdr, edge),
                Some((edge, rank)),
                "edge doc {edge}"
            );
            assert_eq!(
                dense_tf_at(&enc.bytes, &hdr, rank),
                tfs[rank],
                "tf at edge doc {edge}"
            );
        }
        // An absent doc just before an edge resolves to the edge doc.
        for (from, next) in [
            (WORD_ALIGNED_BASE + 62, WORD_ALIGNED_BASE + 63),
            (WORD_ALIGNED_BASE + 126, WORD_ALIGNED_BASE + 127),
        ] {
            assert_eq!(
                dense_next_doc(&enc.bytes, &hdr, from),
                Some((next, rank_of(next))),
                "absent doc {from} resolves to {next}"
            );
        }
        // An absent doc just after an edge resolves to the one after it.
        assert_eq!(
            dense_next_doc(&enc.bytes, &hdr, WORD_ALIGNED_BASE + 65),
            Some((WORD_ALIGNED_BASE + 66, rank_of(WORD_ALIGNED_BASE + 66)))
        );
        // Past the last bit: nothing.
        assert_eq!(
            dense_next_doc(&enc.bytes, &hdr, WORD_ALIGNED_BASE + 128),
            None
        );
    }
}
