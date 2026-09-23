// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! PFOR-delta block codec for posting lists.
//!
//! Postings are encoded in fixed-size **128-doc blocks** (matching
//! `BitPacker4x::BLOCK_LEN` for SIMD-friendly bit-packing). Each block
//! stores a sorted run of `doc_ids` (delta-encoded relative to a base
//! doc id) plus per-doc `tfs`, both bit-packed at the minimum width
//! needed for that block. A block is decodable given its bytes and the
//! previous block's last doc id (which the skip table carries), so the
//! skip table can jump straight to a block.
//!
//! See `docs/architecture/superfile.md` for the overall posting region
//! layout and how blocks chain into the BM25 + BlockMaxWAND query loop.
//!
//! ## On-disk block layout
//!
//! Two header layouts exist, chosen by the blob version
//! (`format::fts::BlockLayout`). The **compact** header (current) is one 32-bit
//! little-endian word:
//!
//! ```text
//!   bits    field
//!   ─────────────────────────────────────────────────────────────────
//!   0..7    doc_count - 1       (1..=128 docs)
//!   7..13   delta_bits          (0..=32, bit-width for deltas)
//!   13..19  tf_bits             (0..=32, bit-width for tfs)
//!   19..24  n_delta_exceptions  (patched blocks; 0 otherwise)
//!   24..26  encoding            (so byte 3's low two bits are the
//!                                encoding in both layouts)
//!   26..31  n_tf_exceptions     (patched blocks; 0 otherwise)
//!   31      spare (0)
//! ```
//!
//! A packed or patched block's base doc id is not stored: it is the
//! previous block's last doc id, or zero for a term's first block, so
//! `delta[0] = doc[0] - base` is at least one from the second block on.
//! A bitset block's word-aligned origin follows the word as a `u32`
//! (its presence words start at the block's own first doc, not at the
//! previous block's last). The **wide** header (`V1`–`V6`) is the older
//! 8-byte form — `doc_count`, `delta_bits`, `tf_bits`, `encoding` as
//! bytes, then the base or origin as a `u32` — and never carries a
//! patched block.
//!
//! After the header:
//!
//! ```text
//!   16 × delta_bits  packed deltas (always BLOCK_LEN values)
//!   16 × tf_bits     packed tfs    (always BLOCK_LEN values)
//! ```
//!
//! `BLOCK_LEN * delta_bits / 8` is always an integer because
//! `BLOCK_LEN == 128`, so `128 * num_bits` is divisible by 8 for every
//! valid `num_bits` value.
//!
//! Two further encodings share the header and put the tfs last: a
//! **bitset** block ([`ENCODING_BITSET`]) stores dense doc ids as
//! presence words, and a **patched** block ([`ENCODING_PATCHED`]) packs
//! deltas and tfs at the width most lanes fit and lists the outliers as
//! exceptions — see each constant's documentation for its layout.
//!
//! ## Partial last block
//!
//! The last block in a posting list may have `doc_count < BLOCK_LEN`.
//! The encoder pads `doc_ids` with the last real value (delta = 0) and
//! pads `tfs` with zero before bit-packing — this keeps both arrays at
//! the bitpacker's required `BLOCK_LEN` length without inflating the
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
    superfile::{
        bits::{ExceptionPlan, PackScratch, plan_exceptions},
        format::fts::BlockLayout,
    },
    utils::varint::{CONTINUATION_BIT, push_varint, read_varint, varint_len},
};

/// Number of `(doc_id, tf)` pairs per encoded block. Fixed at 128 to
/// match `BitPacker4x::BLOCK_LEN`.
pub const BLOCK_LEN: usize = BitPacker4x::BLOCK_LEN;

/// Header byte offset of the block `encoding` field. In the compact
/// header the field is that byte's low two bits ([`ENCODING_MASK`]); in
/// the wide header the whole byte.
pub const ENCODING_OFF: usize = 3;
/// Mask selecting the encoding within header byte [`ENCODING_OFF`].
pub const ENCODING_MASK: u8 = 0b11;

/// Block `encoding`: doc ids stored as PFOR-delta packing. Every
/// `V1`–`V3` block is this.
pub const ENCODING_PACKED: u8 = 0;
/// Block `encoding`: **patched** packing (`VERSION_V7`). Doc-id deltas
/// and tfs are each bit-packed at a width most lanes fit, and the few
/// lanes that do not — the outliers that would otherwise set the width
/// for all 128 — store their high bits as **exceptions**. Layout after
/// the compact header, whose word carries the two exception counts:
///
/// ```text
///   16 × delta_bits  packed low bits of the deltas
///   ...              per delta exception: lane (u8), high bits (LEB128 u32)
///   ...              per tf exception:    lane (u8), high bits (LEB128 u32)
///   16 × tf_bits     packed low bits of the tfs (trailing, as always)
/// ```
///
/// Deltas here are explicit (`doc[0] - base`, then `doc[i] -
/// doc[i-1]`, padding lanes 0), prefix-summed by the reader after the
/// exceptions are patched in — the sorted bit-packer's fused prefix sum
/// cannot see a patch — about 15 ns more per block than the plain
/// decode. Taken only when the writer allows it for the term (see
/// `encode_block`'s `patchable`), when it is smaller than plain packing
/// and when the bitset did not claim the block, so a block with uniform
/// widths stays [`ENCODING_PACKED`] and decodes exactly as before.
pub const ENCODING_PATCHED: u8 = 2;
/// Most exception lanes a patched stream may carry: the header's count
/// field is five bits, and past this the stream is better off wider.
const PATCHED_MAX_EXCEPTIONS: usize = 31;

/// Block `encoding`: doc ids stored as a **presence bitset** over
/// `[origin, last_doc_id]`, `origin` aligned down to a 64-bit word so
/// the union count can OR it in word-aligned. Chosen only when it does
/// not grow the block (dense blocks). Tfs follow, packed identically to
/// PACKED. `VERSION_V4` and later blobs contain these.
pub const ENCODING_BITSET: u8 = 1;

/// Align a doc id down to the 64-bit word that contains it — the origin of
/// a [`ENCODING_BITSET`] block's presence bitset, so its words line up
/// with the union bitset and the OR needs no per-word bit shift.
#[inline]
pub fn bitset_block_base(doc_id: u32) -> u32 {
    doc_id & !63
}

impl BlockLayout {
    /// Bytes the header takes for a block of `encoding`.
    #[inline]
    pub fn header_bytes(self, encoding: u8) -> usize {
        match (self, encoding) {
            (Self::Wide, _) => WIDE_HEADER_SIZE,
            (Self::Compact, ENCODING_BITSET) => COMPACT_HEADER_SIZE + ORIGIN_BYTES,
            (Self::Compact, _) => COMPACT_HEADER_SIZE,
        }
    }
}

/// The wide header: doc_count, delta_bits, tf_bits, encoding, base.
const WIDE_HEADER_SIZE: usize = 8;
/// The compact header: one `u32` word.
const COMPACT_HEADER_SIZE: usize = 4;
/// A compact bitset block's stored origin (`u32` LE) after the word.
const ORIGIN_BYTES: usize = 4;
/// Compact header field positions and widths.
const HDR_COUNT_BITS: u32 = 7;
const HDR_DELTA_BITS_SHIFT: u32 = 7;
const HDR_TF_BITS_SHIFT: u32 = 13;
const HDR_WIDTH_BITS: u32 = 6;
const HDR_DELTA_EXC_SHIFT: u32 = 19;
const HDR_ENCODING_SHIFT: u32 = 24;
const HDR_TF_EXC_SHIFT: u32 = 26;
const HDR_EXC_BITS: u32 = 5;

/// The encoding of the block at `bytes` — one byte read, both layouts.
#[inline]
pub fn block_encoding(bytes: &[u8]) -> u8 {
    bytes[ENCODING_OFF] & ENCODING_MASK
}

/// A block's header, decoded: what every decode needs before it touches
/// the payload. Parsed from the bytes plus, for the compact layout, the
/// previous block's last doc id. Twelve bytes, so the cursor's per-block
/// cache of it is one store and one compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockHeader {
    /// Real `(doc_id, tf)` pairs in the block, `1..=BLOCK_LEN`.
    count: u8,
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
}

impl BlockHeader {
    /// Parse the header at the start of `bytes`. `prev_last_doc_id` is
    /// the previous block's last doc id, `None` for a term's first block;
    /// the wide layout ignores it.
    ///
    /// # Panics
    ///
    /// `bytes` is shorter than the header, or a bit width is past 32
    /// (the packer's unsafe kernels take the width on trust). The other
    /// fields cannot leave their range — the compact word gives them no
    /// room to — and the postings region is CRC-validated at open, so
    /// nothing else is checked on this per-block path.
    #[inline]
    pub fn parse(bytes: &[u8], layout: BlockLayout, prev_last_doc_id: Option<u32>) -> Self {
        let header = match layout {
            BlockLayout::Wide => {
                let encoding = bytes[ENCODING_OFF];
                assert!(
                    encoding != ENCODING_PATCHED,
                    "wide header with a patched block"
                );
                Self {
                    count: bytes[0],
                    delta_bits: bytes[1],
                    tf_bits: bytes[2],
                    encoding,
                    base: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                    n_delta_exc: 0,
                    n_tf_exc: 0,
                    payload: WIDE_HEADER_SIZE as u8,
                }
            }
            BlockLayout::Compact => {
                let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                let field = |shift: u32, bits: u32| ((word >> shift) & ((1 << bits) - 1)) as u8;
                let encoding = field(HDR_ENCODING_SHIFT, 2);
                let (base, payload) = match encoding {
                    ENCODING_BITSET => (
                        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                        (COMPACT_HEADER_SIZE + ORIGIN_BYTES) as u8,
                    ),
                    _ => (prev_last_doc_id.unwrap_or(0), COMPACT_HEADER_SIZE as u8),
                };
                Self {
                    count: field(0, HDR_COUNT_BITS) + 1,
                    delta_bits: field(HDR_DELTA_BITS_SHIFT, HDR_WIDTH_BITS),
                    tf_bits: field(HDR_TF_BITS_SHIFT, HDR_WIDTH_BITS),
                    encoding,
                    base,
                    n_delta_exc: field(HDR_DELTA_EXC_SHIFT, HDR_EXC_BITS),
                    n_tf_exc: field(HDR_TF_EXC_SHIFT, HDR_EXC_BITS),
                    payload,
                }
            }
        };
        // Widths are validated where they drive an unpacker
        // (`check_widths`), not on every probe that only needs the
        // encoding and origin.
        debug_assert!(
            usize::from(header.count) <= BLOCK_LEN,
            "block header: doc_count > BLOCK_LEN"
        );
        debug_assert!(
            header.encoding <= ENCODING_PATCHED,
            "block header: unknown encoding"
        );
        header
    }

    /// Real `(doc_id, tf)` pairs in the block.
    /// Reject a corrupt header before its widths reach an unpacker.
    #[inline]
    pub fn check_widths(&self) {
        assert!(
            self.delta_bits <= 32 && self.tf_bits <= 32,
            "block header: bit width > 32"
        );
    }

    #[inline]
    pub fn count(&self) -> usize {
        usize::from(self.count)
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
        BLOCK_LEN * self.tf_bits as usize / 8
    }

    /// Bytes the packed deltas take (packed and patched blocks).
    #[inline]
    fn deltas_size(&self) -> usize {
        BLOCK_LEN * self.delta_bits as usize / 8
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
fn plan_patched(
    lanes: &[u32; BLOCK_LEN],
    plan: &mut ExceptionPlan,
    candidates: &mut Vec<(u32, u32)>,
) {
    plan_exceptions(
        lanes,
        PATCHED_MAX_EXCEPTIONS,
        |width| BLOCK_LEN * width as usize / 8,
        |_, hi| 1 + varint_len(hi),
        plan,
        candidates,
    );
}

/// Pack `lanes` at `width` bits, keeping only each lane's low `width`
/// bits (the exceptions carry the rest), into `out`.
fn compress_low_bits(bp: &BitPacker4x, lanes: &[u32; BLOCK_LEN], width: u8, out: &mut [u8]) {
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
    bp.compress(&low, out, width);
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
    let word = (count as u32 - 1)
        | (u32::from(delta_bits) << HDR_DELTA_BITS_SHIFT)
        | (u32::from(tf_bits) << HDR_TF_BITS_SHIFT)
        | ((n_delta_exc as u32) << HDR_DELTA_EXC_SHIFT)
        | (u32::from(encoding) << HDR_ENCODING_SHIFT)
        | ((n_tf_exc as u32) << HDR_TF_EXC_SHIFT);
    word.to_le_bytes()
}

/// Encode one block. `prev_last_doc_id` is the previous block's last
/// doc id (`None` for a term's first block): the compact layout derives
/// the base doc id from it, the wide layout stores its own. `patchable`
/// lets the block take the patched form when that is smaller; the
/// writer grants it per term, withholding it from the common terms
/// whose blocks every intersection and union walks in bulk. `scratch`
/// holds the planners' buffers, reused from block to block.
///
/// # Panics
///
/// - `b.doc_ids.is_empty()` — can't encode an empty block.
/// - `b.doc_ids.len() != b.tfs.len()` — mismatched parallel vectors.
/// - `b.doc_ids.len() > BLOCK_LEN`.
/// - (debug) `b.doc_ids` not strictly ascending, or not above
///   `prev_last_doc_id`.
pub fn encode_block(
    b: &Block,
    layout: BlockLayout,
    prev_last_doc_id: Option<u32>,
    patchable: bool,
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

    // Pad both arrays to BLOCK_LEN. doc_ids: pad with the last real value
    // so the padded delta is 0. tfs: pad with 0 (default fill).
    let mut padded_doc_ids = [0u32; BLOCK_LEN];
    padded_doc_ids[..count].copy_from_slice(&b.doc_ids);
    for slot in &mut padded_doc_ids[count..] {
        *slot = last_doc_id;
    }
    let mut padded_tfs = [0u32; BLOCK_LEN];
    padded_tfs[..count].copy_from_slice(&b.tfs);

    // The value the decoder recovers doc_ids[0] from (decompressed[0] =
    // base + delta[0]). Wide: `doc_ids[0] - 1`, so the smallest delta is
    // 1 and the width is tight (clamped at 0 for doc 0). Compact: the
    // previous block's last doc id, which the skip table already holds,
    // or zero for the first block — whose first delta is then the doc
    // id itself, an outlier the patched form absorbs as one exception.
    // A bitset's origin is stored in both layouts.
    let base_doc_id = match layout {
        BlockLayout::Wide => b.doc_ids[0].saturating_sub(1),
        BlockLayout::Compact => prev_last_doc_id.unwrap_or(0),
    };
    let aligned_base = bitset_block_base(b.doc_ids[0]);

    let bp = BitPacker4x::new();
    let delta_bits = bp.num_bits_sorted(base_doc_id, &padded_doc_ids);
    let tf_bits = bp.num_bits(&padded_tfs);

    let deltas_size = BLOCK_LEN * delta_bits as usize / 8;
    let tfs_size = BLOCK_LEN * tf_bits as usize / 8;
    let header_size = layout.header_bytes(ENCODING_PACKED);

    // Store the doc ids as a presence bitset instead of PFOR deltas when
    // that does not grow the block (a dense block — a common term's ~128
    // near-consecutive docs). The bitset origin is word-aligned so the
    // union count can OR it in without a per-word shift. Measured
    // against the tight deltas from the block's own first doc in both
    // layouts, so the decision does not depend on where the previous
    // block ended.
    let bitset_words = (last_doc_id - aligned_base) as usize / 64 + 1;
    let bitset_size = bitset_words * 8;
    let tight_delta_bits = bp.num_bits_sorted(b.doc_ids[0].saturating_sub(1), &padded_doc_ids);
    let use_bitset = bitset_size <= BLOCK_LEN * tight_delta_bits as usize / 8;

    // Patched packing (compact layout only): explicit deltas and tfs,
    // each at the width most lanes fit plus an exception list for the
    // rest. Considered only for a block the bitset did not claim: a dense
    // block's presence words are what the count kernels bit-test and
    // rank into, and that O(1) probe is worth more than the few bytes
    // patching a dense partial block would save.
    if patchable && layout == BlockLayout::Compact && !use_bitset {
        let mut explicit_deltas = [0u32; BLOCK_LEN];
        explicit_deltas[0] = b.doc_ids[0] - base_doc_id;
        for (slot, pair) in explicit_deltas[1..count]
            .iter_mut()
            .zip(b.doc_ids.windows(2))
        {
            *slot = pair[1] - pair[0];
        }
        plan_patched(
            &explicit_deltas,
            &mut scratch.plan_a,
            &mut scratch.candidates,
        );
        plan_patched(&padded_tfs, &mut scratch.plan_b, &mut scratch.candidates);
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
            let deltas_packed = BLOCK_LEN * delta_plan.width as usize / 8;
            bytes.resize(deltas_start + deltas_packed, 0);
            compress_low_bits(
                &bp,
                &explicit_deltas,
                delta_plan.width,
                &mut bytes[deltas_start..],
            );
            for &(lane, hi) in delta_plan.exceptions.iter().chain(&tf_plan.exceptions) {
                bytes.push(lane as u8);
                push_varint(&mut bytes, hi);
            }
            let tfs_start = bytes.len();
            let tfs_packed = BLOCK_LEN * tf_plan.width as usize / 8;
            bytes.resize(tfs_start + tfs_packed, 0);
            compress_low_bits(&bp, &padded_tfs, tf_plan.width, &mut bytes[tfs_start..]);
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
    let mut bytes = Vec::with_capacity(layout.header_bytes(encoding) + doc_ids_size + tfs_size);

    // Header.
    let stored_delta_bits = if use_bitset { 0 } else { delta_bits };
    match layout {
        BlockLayout::Wide => {
            bytes.push(count as u8);
            bytes.push(stored_delta_bits);
            bytes.push(tf_bits);
            bytes.push(encoding);
            bytes.extend_from_slice(
                &if use_bitset {
                    aligned_base
                } else {
                    base_doc_id
                }
                .to_le_bytes(),
            );
        }
        BlockLayout::Compact => {
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
        }
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
        bp.compress_sorted(
            base_doc_id,
            &padded_doc_ids,
            &mut bytes[doc_ids_start..doc_ids_start + deltas_size],
            delta_bits,
        );
    }

    // Packed tfs — identical in both encodings, in doc order.
    let tfs_start = bytes.len();
    bytes.resize(tfs_start + tfs_size, 0);
    bp.compress(
        &padded_tfs,
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
/// decoder writes all `BLOCK_LEN` slots for SIMD reasons. The returned
/// `doc_count` tells the caller how many of those slots are real (the
/// rest are padding values — zero deltas for doc_ids, zero tfs).
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
    BitPacker4x::new().decompress(
        &bytes[hdr.payload()..hdr.payload() + deltas_size],
        &mut dest_doc_ids[..BLOCK_LEN],
        hdr.delta_bits,
    );
    let tf_exc_start = patch_lanes(
        bytes,
        hdr.payload() + deltas_size,
        hdr.n_delta_exc(),
        hdr.delta_bits,
        dest_doc_ids,
    );
    prefix_sum_block(&mut dest_doc_ids[..BLOCK_LEN], hdr.base);
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
    BitPacker4x::new().decompress(
        &bytes[tfs_start..tfs_start + tfs_size],
        &mut dest_tfs[..BLOCK_LEN],
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
}

/// Decode only the doc ids of a block into `dest_doc_ids`, skipping the
/// tf half. Returns the block's doc count. The unranked count kernels
/// never read a tf, so they take this instead of [`decode_block`].
///
/// # Panics
///
/// As [`decode_block`], minus the `dest_tfs` checks.
/// The first set bit of a bitset block at or after `from`, as `(doc, rank)`
/// where `rank` is the number of set bits before it — the doc's index in
/// the block's tf array. `None` when no doc of the block is `>= from`.
///
/// A probe that skips into a bitset block needs one doc, not the 128 the
/// full expansion writes; this scans the block's words from the target's
/// word with a masked `trailing_zeros`, at most a handful of u64s.
pub fn bitset_next_doc(bytes: &[u8], hdr: &BlockHeader, from: u32) -> Option<(u32, usize)> {
    debug_assert_eq!(hdr.encoding, ENCODING_BITSET);
    hdr.check_widths();
    let tfs_size = hdr.tfs_size();
    assert!(
        bytes.len() >= hdr.payload() + tfs_size,
        "bitset_next_doc: bytes shorter than header+tfs"
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

/// One tf out of a bitset block without unpacking the other 127: the tf of
/// the doc with set-bit index `rank`. Bitset blocks carry plain packed tfs
/// (never patched). `BitPacker4x` interleaves four 32-bit lanes: value `i`
/// lives in lane `i % 4` at bit `(i / 4) * width` of that lane's stream,
/// whose 32-bit words sit at every fourth word of the payload.
///
/// For a *walk* over a block the full unpack into the tf array wins (one
/// unpack serves every doc); this is for a *probe* that touches one doc.
pub fn bitset_tf_at(bytes: &[u8], hdr: &BlockHeader, rank: usize) -> u32 {
    debug_assert_eq!(hdr.encoding, ENCODING_BITSET);
    debug_assert!(rank < BLOCK_LEN);
    hdr.check_widths();
    let width = usize::from(hdr.tf_bits);
    if width == 0 {
        return 0;
    }
    let tfs_size = hdr.tfs_size();
    assert!(
        bytes.len() >= hdr.payload() + tfs_size,
        "bitset_tf_at: bytes shorter than header+tfs"
    );
    let packed = &bytes[bytes.len() - tfs_size..];
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
    (v & ((1u64 << width) - 1)) as u32
}

pub fn decode_block_doc_ids(bytes: &[u8], hdr: &BlockHeader, dest_doc_ids: &mut [u32]) -> usize {
    hdr.check_widths();
    assert!(
        dest_doc_ids.len() >= BLOCK_LEN,
        "decode_block_doc_ids: dest_doc_ids must have at least {BLOCK_LEN} slots"
    );
    let count = hdr.count();

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
        // fires if a future builder change ever breaks invariant 1. This bound
        // depends on invariant 2: if a path is ever added that decodes blocks
        // before validating their CRC, a `j < BLOCK_LEN` bound becomes mandatory.
        let mut j = 0usize;
        for (wi, chunk) in words.chunks_exact(8).enumerate() {
            let mut word = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
            while word != 0 {
                dest_doc_ids[j] = hdr.base + (wi as u32 * 64 + word.trailing_zeros());
                j += 1;
                word &= word - 1;
            }
        }
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
    BitPacker4x::new().decompress_sorted(
        hdr.base,
        &bytes[hdr.payload()..hdr.payload() + deltas_size],
        &mut dest_doc_ids[..BLOCK_LEN],
        hdr.delta_bits,
    );

    count
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAYOUTS: [BlockLayout; 2] = [BlockLayout::Wide, BlockLayout::Compact];

    /// Encode with fresh scratch.
    fn encode_one(
        b: &Block,
        layout: BlockLayout,
        prev_last_doc_id: Option<u32>,
        patchable: bool,
    ) -> EncodedBlock {
        encode_block(
            b,
            layout,
            prev_last_doc_id,
            patchable,
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
    fn roundtrip_with(b: &Block, layout: BlockLayout, prev: Option<u32>) -> EncodedBlock {
        let enc = encode_one(b, layout, prev, true);
        let hdr = BlockHeader::parse(&enc.bytes, layout, prev);
        assert_eq!(hdr.count(), b.doc_ids.len());
        let mut got_doc_ids = vec![0u32; BLOCK_LEN];
        let mut got_tfs = vec![0u32; BLOCK_LEN];
        let count = decode_block(&enc.bytes, &hdr, &mut got_doc_ids, &mut got_tfs);
        assert_eq!(count, b.doc_ids.len(), "doc_count round-trip");
        assert_eq!(
            &got_doc_ids[..count],
            b.doc_ids.as_slice(),
            "doc_ids round-trip ({layout:?}, prev {prev:?})"
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

    /// Round-trip under both layouts as a first block; return the
    /// compact encoding.
    fn roundtrip(b: &Block) -> EncodedBlock {
        roundtrip_with(b, BlockLayout::Wide, None);
        roundtrip_with(b, BlockLayout::Compact, None)
    }

    fn compact(b: &Block) -> EncodedBlock {
        encode_one(b, BlockLayout::Compact, None, true)
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
        let enc = roundtrip_with(&block(&doc_ids, &tfs), BlockLayout::Compact, Some(999));
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PATCHED);
        assert!(enc.bytes.len() < 64, "got {} bytes", enc.bytes.len());
        let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Compact, Some(999));
        assert_eq!((hdr.n_delta_exc(), hdr.n_tf_exc()), (1, 1));
        assert_eq!(hdr.base, 999);
    }

    #[test]
    fn a_first_block_absorbs_its_doc_id_as_one_exception() {
        // The compact layout has no stored base, so a first block's first
        // delta is the doc id itself — patched away as one exception
        // rather than widening every lane.
        let doc_ids: Vec<u32> = (0..128).map(|i| 500_000 + 3 * i).collect();
        let enc = roundtrip(&block(&doc_ids, &[1; 128]));
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PATCHED);
        let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Compact, None);
        assert_eq!(hdr.base, 0);
        assert_eq!(hdr.delta_bits, 2);
        assert_eq!(hdr.n_delta_exc(), 1);
        // The wide header stores `first - 1` instead and packs plain; the
        // compact block is no larger for lacking a stored base.
        let wide = encode_one(&block(&doc_ids, &[1; 128]), BlockLayout::Wide, None, true);
        assert_eq!(block_encoding(&wide.bytes), ENCODING_PACKED);
        assert_eq!(
            BlockHeader::parse(&wide.bytes, BlockLayout::Wide, None).base,
            499_999
        );
        assert!(
            enc.bytes.len() <= wide.bytes.len(),
            "{} vs {}",
            enc.bytes.len(),
            wide.bytes.len()
        );
    }

    #[test]
    fn a_block_the_writer_may_not_patch_stays_plain() {
        // The same outlier block, with patching withheld for its term:
        // plain packing at the outlier's width, decoded by the fused
        // kernel — what a common term's blocks get.
        let mut doc_ids: Vec<u32> = (1000..1127).collect();
        doc_ids.push(1_126 + 1_000_000);
        let b = block(&doc_ids, &[1; 128]);
        let plain = encode_one(&b, BlockLayout::Compact, Some(999), false);
        assert_eq!(block_encoding(&plain.bytes), ENCODING_PACKED);
        let patched = encode_one(&b, BlockLayout::Compact, Some(999), true);
        assert_eq!(block_encoding(&patched.bytes), ENCODING_PATCHED);
        assert!(plain.bytes.len() > patched.bytes.len());
        let hdr = BlockHeader::parse(&plain.bytes, BlockLayout::Compact, Some(999));
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
        let enc = roundtrip_with(&block(&doc_ids, &tfs), BlockLayout::Compact, Some(9));
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PACKED);
        // 6-bit deltas (37) and 3-bit tfs (up to 4), nothing else.
        assert_eq!(enc.bytes.len(), COMPACT_HEADER_SIZE + 16 * 6 + 16 * 3);
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
        // 128 consecutive docs pack to 1-bit deltas (16 B); the bitset is
        // 2 words (16 B), so the tie goes to the bitset in both layouts.
        let doc_ids: Vec<u32> = (64..192).collect();
        let tfs = vec![1u32; 128];
        for layout in LAYOUTS {
            let enc = roundtrip_with(&block(&doc_ids, &tfs), layout, Some(40));
            assert_eq!(block_encoding(&enc.bytes), ENCODING_BITSET, "{layout:?}");
            let hdr = BlockHeader::parse(&enc.bytes, layout, Some(40));
            assert_eq!(hdr.base, 64, "{layout:?} origin");
            assert_eq!(hdr.delta_bits, 0);
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
            for layout in LAYOUTS {
                let enc = encode_one(&block(&doc_ids, &tfs), layout, Some(900), true);
                let hdr = BlockHeader::parse(&enc.bytes, layout, Some(900));
                if hdr.encoding != ENCODING_BITSET {
                    continue;
                }
                let mut ids = vec![0u32; BLOCK_LEN];
                let mut got_tfs = vec![0u32; BLOCK_LEN];
                let n = decode_block(&enc.bytes, &hdr, &mut ids, &mut got_tfs);
                for target in (hdr.base - 5)..=(doc_ids[n - 1] + 3) {
                    let want = ids[..n].iter().position(|&d| d >= target);
                    let got = bitset_next_doc(&enc.bytes, &hdr, target);
                    match want {
                        None => assert!(got.is_none(), "{layout:?} target {target}"),
                        Some(r) => {
                            let (doc, rank) = got.expect("a doc >= target");
                            assert_eq!((doc, rank), (ids[r], r), "{layout:?} target {target}");
                            assert_eq!(bitset_tf_at(&enc.bytes, &hdr, rank), got_tfs[r]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "bit width > 32")]
    fn bitset_probe_refuses_a_header_width_past_32() {
        // A bitset block whose header claims tf width 33: the probe must
        // fail the width check, not underflow the tf-size arithmetic.
        let doc_ids: Vec<u32> = (256..384).collect();
        let tfs = vec![1u32; 128];
        let mut enc = encode_one(
            &block(&doc_ids, &tfs),
            BlockLayout::Compact,
            Some(200),
            false,
        );
        assert_eq!(block_encoding(&enc.bytes), ENCODING_BITSET);
        let mut word = u32::from_le_bytes(enc.bytes[..4].try_into().expect("4 bytes"));
        let mask = ((1u32 << HDR_WIDTH_BITS) - 1) << HDR_TF_BITS_SHIFT;
        word = (word & !mask) | (33 << HDR_TF_BITS_SHIFT);
        enc.bytes[..4].copy_from_slice(&word.to_le_bytes());
        let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Compact, Some(200));
        assert_eq!(hdr.tf_bits, 33);
        let _ = bitset_next_doc(&enc.bytes, &hdr, 256);
    }

    #[test]
    #[should_panic(expected = "bit width > 32")]
    fn decoders_refuse_a_header_width_past_32() {
        // The width check moved from every header parse to the decoders
        // that consume the widths; a corrupt compact header still cannot
        // reach an unpacker.
        let doc_ids: Vec<u32> = (0..128u32).map(|i| 10 + i * 7).collect();
        let tfs = vec![1u32; 128];
        let mut enc = encode_one(&block(&doc_ids, &tfs), BlockLayout::Compact, None, false);
        let mut word = u32::from_le_bytes(enc.bytes[..4].try_into().expect("4 bytes"));
        let mask = ((1u32 << HDR_WIDTH_BITS) - 1) << HDR_DELTA_BITS_SHIFT;
        word = (word & !mask) | (33 << HDR_DELTA_BITS_SHIFT);
        enc.bytes[..4].copy_from_slice(&word.to_le_bytes());
        let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Compact, None);
        assert_eq!(hdr.delta_bits, 33, "the parse itself no longer rejects it");
        let mut ids = vec![0u32; BLOCK_LEN];
        decode_block_doc_ids(&enc.bytes, &hdr, &mut ids);
    }

    #[test]
    fn roundtrip_bitset_block_nonzero_aligned_base() {
        // A dense run far from zero: the origin is the word holding the
        // block's first doc, stored in both layouts, whether the block is
        // a term's first or follows another far behind it.
        let doc_ids: Vec<u32> = (960..1088).collect();
        let tfs = vec![2u32; 128];
        for (layout, prev) in [
            (BlockLayout::Wide, None),
            (BlockLayout::Compact, None),
            (BlockLayout::Compact, Some(17)),
        ] {
            let enc = roundtrip_with(&block(&doc_ids, &tfs), layout, prev);
            assert_eq!(block_encoding(&enc.bytes), ENCODING_BITSET, "{layout:?}");
            let hdr = BlockHeader::parse(&enc.bytes, layout, prev);
            assert_eq!(hdr.base, 960, "{layout:?}");
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
        roundtrip_with(
            &block(&doc_ids, &tfs),
            BlockLayout::Compact,
            Some(u32::MAX - 21),
        );
    }

    #[test]
    fn a_later_block_follows_the_previous_last_doc() {
        // Consecutive blocks of one term, each decoded from the previous
        // block's last doc id.
        let all: Vec<u32> = (0..300u32).map(|i| 7 + i * 5).collect();
        let mut prev = None;
        let mut recovered = Vec::new();
        for chunk in all.chunks(BLOCK_LEN) {
            let enc = roundtrip_with(
                &block(chunk, &vec![1; chunk.len()]),
                BlockLayout::Compact,
                prev,
            );
            let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Compact, prev);
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
        // Deltas of 1 need exactly 1 bit; tfs of 1 too.
        let doc_ids: Vec<u32> = (100..228).collect();
        let tfs = vec![1u32; 128];
        for layout in LAYOUTS {
            let hdr = BlockHeader::parse(
                &encode_one(&block(&doc_ids, &tfs), layout, Some(99), true).bytes,
                layout,
                Some(99),
            );
            assert_eq!(hdr.tf_bits, 1, "{layout:?}");
        }
    }

    #[test]
    fn bit_width_7_just_below_byte_boundary() {
        let doc_ids: Vec<u32> = (0..128).map(|i| i * 127).collect(); // deltas of 127 → 7 bits
        let tfs = vec![127u32; 128];
        let enc = roundtrip_with(&block(&doc_ids, &tfs), BlockLayout::Wide, None);
        let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Wide, None);
        assert_eq!((hdr.delta_bits, hdr.tf_bits), (7, 7));
    }

    #[test]
    fn bit_width_8_at_byte_boundary() {
        let doc_ids: Vec<u32> = (0..128).map(|i| i * 255).collect(); // deltas of 255 → 8 bits
        let tfs = vec![255u32; 128];
        let enc = roundtrip_with(&block(&doc_ids, &tfs), BlockLayout::Wide, None);
        let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Wide, None);
        assert_eq!((hdr.delta_bits, hdr.tf_bits), (8, 8));
    }

    #[test]
    fn widths_31_and_32_round_trip_losslessly() {
        // The widest deltas and tfs the codec stores. The wide header
        // never patches, so its widths are the plain ones; the compact
        // block packs the same values through the patched path.
        for (last, want_bits) in [((1u32 << 31) - 1, 31u8), (u32::MAX, 32)] {
            let b = block(&[0, last], &[u32::MAX, 1]);
            let enc = roundtrip_with(&b, BlockLayout::Wide, None);
            let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Wide, None);
            assert_eq!((hdr.delta_bits, hdr.tf_bits), (want_bits, 32));
            roundtrip_with(&b, BlockLayout::Compact, None);
        }
        // A full block of 32-bit tfs stays plain in both layouts.
        let doc_ids: Vec<u32> = (0..128).collect();
        let enc = roundtrip(&block(&doc_ids, &[u32::MAX - 5; 128]));
        let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Compact, None);
        assert_eq!(hdr.tf_bits, 32);
    }

    #[test]
    fn bit_width_0_for_all_zero_tfs() {
        // A tf of 0 is not meaningful for BM25 but the codec must not
        // choke on it; it packs at width 0 (no tf bytes at all).
        let doc_ids: Vec<u32> = (0..128).map(|i| i * 2).collect();
        let tfs = vec![0u32; 128];
        let enc = roundtrip(&block(&doc_ids, &tfs));
        let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Compact, None);
        assert_eq!(hdr.tf_bits, 0);
        assert_eq!(hdr.tfs_size(), 0);
    }

    // --- Header layout / metadata --------------------------------------

    #[test]
    fn encoded_block_carries_last_doc_id_and_max_tf() {
        let doc_ids: Vec<u32> = vec![5, 10, 15, 20, 25];
        let tfs: Vec<u32> = vec![1, 4, 2, 9, 3];
        let enc = compact(&block(&doc_ids, &tfs));
        assert_eq!(enc.last_doc_id, 25);
        assert_eq!(enc.max_tf, 9);
    }

    #[test]
    fn header_doc_count_round_trips() {
        for count in [1usize, 2, 31, 32, 33, 63, 64, 65, 127, 128] {
            let doc_ids: Vec<u32> = (1..=count as u32).collect();
            let tfs = vec![1u32; count];
            for layout in LAYOUTS {
                let enc = encode_one(&block(&doc_ids, &tfs), layout, None, true);
                assert_eq!(
                    BlockHeader::parse(&enc.bytes, layout, None).count(),
                    count,
                    "header.doc_count for n={count} ({layout:?})"
                );
            }
        }
    }

    #[test]
    fn header_encoding_byte_marks_the_layout() {
        // Byte 3's low two bits are the encoding in both layouts: 0 =
        // PACKED (a full block of uniform wide deltas, where nothing beats
        // plain packing), 1 = BITSET (dense), 2 = PATCHED (a sparse partial
        // block: the padding lanes pack at width 0 and the real deltas
        // ride as exceptions).
        let uniform: Vec<u32> = (0..128).map(|i| 1 + 100_000 * i).collect();
        let packed = encode_one(
            &block(&uniform, &[1; 128]),
            BlockLayout::Compact,
            Some(0),
            true,
        );
        assert_eq!(
            block_encoding(&packed.bytes),
            ENCODING_PACKED,
            "uniform ⇒ PACKED"
        );
        let bitset = compact(&block(&[1, 2, 3], &[1, 1, 1]));
        assert_eq!(
            block_encoding(&bitset.bytes),
            ENCODING_BITSET,
            "dense ⇒ BITSET"
        );
        let patched = compact(&block(&[1, 100_000], &[1, 1]));
        assert_eq!(
            block_encoding(&patched.bytes),
            ENCODING_PATCHED,
            "sparse partial ⇒ PATCHED"
        );
        for enc in [&packed, &bitset, &patched] {
            assert_eq!(
                BlockHeader::parse(&enc.bytes, BlockLayout::Compact, Some(0)).encoding,
                block_encoding(&enc.bytes)
            );
        }
        let wide = encode_one(
            &block(&[1, 2, 3], &[1, 1, 1]),
            BlockLayout::Wide,
            None,
            true,
        );
        assert_eq!(wide.bytes[ENCODING_OFF], ENCODING_BITSET);
    }

    #[test]
    fn compact_header_holds_every_field_at_its_extreme() {
        // Widths of 32, a full block, and the most exceptions either
        // stream may carry all fit the word and read back.
        let bytes = compact_header(128, 32, 32, ENCODING_PATCHED, 31, 31);
        let hdr = BlockHeader::parse(&bytes, BlockLayout::Compact, Some(7));
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
        let hdr = BlockHeader::parse(&bytes, BlockLayout::Compact, None);
        assert_eq!(
            (hdr.count(), hdr.delta_bits, hdr.tf_bits, hdr.base),
            (1, 0, 0, 0)
        );
    }

    #[test]
    fn wide_header_base_doc_id_is_first_minus_one() {
        // A wide non-bitset block stores base = first doc - 1.
        let enc = encode_one(
            &block(&[100, 100_000, 200_000], &[1, 1, 1]),
            BlockLayout::Wide,
            None,
            true,
        );
        assert_ne!(block_encoding(&enc.bytes), ENCODING_BITSET);
        assert_eq!(
            BlockHeader::parse(&enc.bytes, BlockLayout::Wide, None).base,
            99
        );
        let enc = encode_one(
            &block(&[0, 1, 2], &[1, 1, 1]),
            BlockLayout::Wide,
            None,
            true,
        );
        assert_eq!(
            BlockHeader::parse(&enc.bytes, BlockLayout::Wide, None).base,
            0,
            "saturating_sub at 0"
        );
    }

    // --- Mixed bit widths between deltas and tfs ------------------------

    #[test]
    fn delta_and_tf_use_independent_bit_widths() {
        // Wide deltas, narrow tfs — a full block of uniform lanes, so the
        // header carries the plain widths.
        let doc_ids: Vec<u32> = (0..128).map(|i| i * 1024).collect(); // delta = 1024 → 11 bits
        let tfs: Vec<u32> = (0..128).map(|_| 1).collect();
        let enc = roundtrip_with(&block(&doc_ids, &tfs), BlockLayout::Wide, None);
        let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Wide, None);
        assert!(
            (10..=12).contains(&hdr.delta_bits),
            "expected ~11 delta bits, got {}",
            hdr.delta_bits
        );
        assert_eq!(hdr.tf_bits, 1);
    }

    // --- Panic surface for invalid input -------------------------------

    #[test]
    #[should_panic(expected = "empty block")]
    fn encode_block_panics_on_empty() {
        compact(&Block {
            doc_ids: vec![],
            tfs: vec![],
        });
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn encode_block_panics_on_length_mismatch() {
        compact(&Block {
            doc_ids: vec![1, 2, 3],
            tfs: vec![1, 2],
        });
    }

    #[test]
    #[should_panic(expected = "> BLOCK_LEN")]
    fn encode_block_panics_on_oversize() {
        let doc_ids: Vec<u32> = (0..129).collect();
        let tfs = vec![1u32; 129];
        compact(&block(&doc_ids, &tfs));
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn header_parse_panics_on_short_input() {
        BlockHeader::parse(&[0u8; 3], BlockLayout::Compact, None);
    }

    #[test]
    #[should_panic(expected = "shorter than header")]
    fn decode_block_panics_on_truncated_payload() {
        let enc = compact(&block(&[1, 100, 200], &[1, 1, 1]));
        let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Compact, None);
        let mut d = vec![0u32; BLOCK_LEN];
        let mut t = vec![0u32; BLOCK_LEN];
        let _ = decode_block(&enc.bytes[..COMPACT_HEADER_SIZE + 1], &hdr, &mut d, &mut t);
    }

    #[test]
    #[should_panic(expected = "must have at least")]
    fn decode_block_panics_on_undersized_dest() {
        let enc = compact(&block(&[1, 2, 3], &[1, 1, 1]));
        let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Compact, None);
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
        let enc1 = compact(&b1);
        let enc2 = encode_one(&b2, BlockLayout::Compact, Some(5), true);
        // Decode in opposite order to confirm zero shared state.
        let h2 = BlockHeader::parse(&enc2.bytes, BlockLayout::Compact, Some(5));
        let mut d2 = vec![0u32; BLOCK_LEN];
        let mut t2 = vec![0u32; BLOCK_LEN];
        let n2 = decode_block(&enc2.bytes, &h2, &mut d2, &mut t2);
        let h1 = BlockHeader::parse(&enc1.bytes, BlockLayout::Compact, None);
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

        for layout in LAYOUTS {
            let mut encoded: Vec<EncodedBlock> = Vec::new();
            let mut prev = None;
            for (d, t) in all_doc_ids.chunks(BLOCK_LEN).zip(all_tfs.chunks(BLOCK_LEN)) {
                let enc = encode_one(&block(d, t), layout, prev, true);
                prev = Some(enc.last_doc_id);
                encoded.push(enc);
            }

            let mut recovered_doc_ids = Vec::with_capacity(all_doc_ids.len());
            let mut recovered_tfs = Vec::with_capacity(all_tfs.len());
            let mut buf_d = vec![0u32; BLOCK_LEN];
            let mut buf_t = vec![0u32; BLOCK_LEN];
            let mut prev = None;
            for enc in &encoded {
                let hdr = BlockHeader::parse(&enc.bytes, layout, prev);
                let n = decode_block(&enc.bytes, &hdr, &mut buf_d, &mut buf_t);
                recovered_doc_ids.extend_from_slice(&buf_d[..n]);
                recovered_tfs.extend_from_slice(&buf_t[..n]);
                prev = Some(enc.last_doc_id);
            }
            assert_eq!(recovered_doc_ids, all_doc_ids, "{layout:?}");
            assert_eq!(recovered_tfs, all_tfs, "{layout:?}");
        }
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
            // matches doc_ids; tf in 0..=4095 to bound bit
            // width.
            let mut rng = tf_seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let tfs: Vec<u32> = (0..doc_ids.len())
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    (rng & 0xFFF) as u32
                })
                .collect();
            // A later block follows some doc below its first.
            let prev = match (as_later_block, doc_ids[0]) {
                (true, first) if first > 0 => Some((first - 1) / 2),
                _ => None,
            };

            let block = Block { doc_ids: doc_ids.clone(), tfs: tfs.clone() };
            for layout in LAYOUTS {
                let enc = encode_one(&block, layout, prev, true);

                prop_assert_eq!(enc.last_doc_id, *doc_ids.last().expect("last element"));
                prop_assert_eq!(enc.max_tf, *tfs.iter().max().expect("iter max"));

                let hdr = BlockHeader::parse(&enc.bytes, layout, prev);
                let mut got_doc_ids = vec![0u32; BLOCK_LEN];
                let mut got_tfs = vec![0u32; BLOCK_LEN];
                let count = decode_block(&enc.bytes, &hdr, &mut got_doc_ids, &mut got_tfs);
                prop_assert_eq!(count, doc_ids.len());
                prop_assert_eq!(&got_doc_ids[..count], doc_ids.as_slice());
                prop_assert_eq!(&got_tfs[..count], tfs.as_slice());
            }
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
                    (rng & 0xFFFF) as u32
                })
                .collect();

            let enc = compact(&Block { doc_ids, tfs });
            let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Compact, None);
            let tfs_size = hdr.tfs_size();
            match hdr.encoding {
                ENCODING_BITSET => {
                    // Doc ids are a whole number of 64-bit words; tfs trail.
                    prop_assert_eq!(hdr.delta_bits, 0, "bitset block has delta_bits 0");
                    let bitset_bytes = enc.bytes.len() - hdr.payload() - tfs_size;
                    prop_assert!(bitset_bytes >= 8 && bitset_bytes.is_multiple_of(8));
                }
                ENCODING_PATCHED => {
                    let (delta_exc, tf_exc) = patched_exception_ranges(&enc.bytes, &hdr);
                    prop_assert_eq!(
                        enc.bytes.len(),
                        COMPACT_HEADER_SIZE
                            + (BLOCK_LEN * hdr.delta_bits as usize) / 8
                            + delta_exc.len()
                            + tf_exc.len()
                            + tfs_size
                    );
                }
                _ => {
                    prop_assert_eq!(
                        enc.bytes.len(),
                        COMPACT_HEADER_SIZE + (BLOCK_LEN * hdr.delta_bits as usize) / 8 + tfs_size
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

    /// A block of 128 lanes one apart and tf 1, except the listed
    /// outlier lanes (a wide gap, a large tf, or both on one lane).
    fn outlier_block(delta_lanes: &[usize], tf_lanes: &[usize]) -> Block {
        let mut doc_ids = Vec::with_capacity(BLOCK_LEN);
        let mut d = 1000u32;
        for lane in 0..BLOCK_LEN {
            d += if delta_lanes.contains(&lane) {
                LIMIT_OUTLIER_GAP
            } else {
                1
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
        let enc = roundtrip_with(&b, BlockLayout::Compact, Some(LIMIT_PREV_LAST_DOC));
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PATCHED);
        let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Compact, Some(LIMIT_PREV_LAST_DOC));
        assert_eq!((hdr.n_delta_exc(), hdr.n_tf_exc()), (3, 3));
        let (delta_exc, tf_exc) = patched_exception_ranges(&enc.bytes, &hdr);
        assert!(!delta_exc.is_empty() && !tf_exc.is_empty());
        // The wide layout never patches; the same block stays plain there.
        let wide = roundtrip_with(&b, BlockLayout::Wide, Some(LIMIT_PREV_LAST_DOC));
        assert_eq!(block_encoding(&wide.bytes), ENCODING_PACKED);
    }

    /// The exception count is a five-bit header field, so a stream may
    /// carry at most `PATCHED_MAX_EXCEPTIONS` outliers: a block with
    /// exactly that many patches, one more falls back to plain packing
    /// at the outliers' width. Both decode to the planted lanes; an encoder
    /// that admitted a 32nd exception would overflow the header field.
    #[test]
    fn patched_exceptions_stop_at_the_header_field_limit() {
        let at_limit: Vec<usize> = (0..PATCHED_MAX_EXCEPTIONS).map(|i| 2 + 4 * i).collect();
        let enc = roundtrip_with(
            &outlier_block(&at_limit, &[]),
            BlockLayout::Compact,
            Some(LIMIT_PREV_LAST_DOC),
        );
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PATCHED);
        let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Compact, Some(LIMIT_PREV_LAST_DOC));
        assert_eq!(hdr.n_delta_exc(), PATCHED_MAX_EXCEPTIONS);

        let past: Vec<usize> = (0..=PATCHED_MAX_EXCEPTIONS).map(|i| 2 + 4 * i).collect();
        let enc = roundtrip_with(
            &outlier_block(&past, &[]),
            BlockLayout::Compact,
            Some(LIMIT_PREV_LAST_DOC),
        );
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PACKED);

        // The same limit on the tf stream.
        let enc = roundtrip_with(
            &outlier_block(&[], &at_limit),
            BlockLayout::Compact,
            Some(LIMIT_PREV_LAST_DOC),
        );
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PATCHED);
        let hdr = BlockHeader::parse(&enc.bytes, BlockLayout::Compact, Some(LIMIT_PREV_LAST_DOC));
        assert_eq!(hdr.n_tf_exc(), PATCHED_MAX_EXCEPTIONS);
        let enc = roundtrip_with(
            &outlier_block(&[], &past),
            BlockLayout::Compact,
            Some(LIMIT_PREV_LAST_DOC),
        );
        assert_eq!(block_encoding(&enc.bytes), ENCODING_PACKED);
    }

    // --- Bitset probes at word edges ------------------------------------

    /// The bitset probes at the docs that sit on 64-bit word edges: the
    /// first and last bit of a word, and the two bits either side of a
    /// word boundary. `bitset_next_doc` must not lose the last bit of a
    /// word to the mask of the next, and the rank (the tf index) must
    /// count every set bit of the earlier words. Absent docs on the
    /// other side of each edge must resolve to the next present one.
    #[test]
    fn bitset_probes_resolve_docs_on_word_edges() {
        // Present: the whole 128-doc range except four absent docs chosen
        // next to the edges, so every edge has a present and an absent
        // neighbour.
        let absent = [
            WORD_ALIGNED_BASE + 1,
            WORD_ALIGNED_BASE + 62,
            WORD_ALIGNED_BASE + 65,
            WORD_ALIGNED_BASE + 126,
        ];
        let doc_ids: Vec<u32> = (WORD_ALIGNED_BASE..WORD_ALIGNED_BASE + BLOCK_LEN as u32)
            .filter(|d| !absent.contains(d))
            .collect();
        let tfs: Vec<u32> = (0..doc_ids.len() as u32).map(|i| 1 + i % 7).collect();
        let enc = encode_one(
            &block(&doc_ids, &tfs),
            BlockLayout::Compact,
            Some(WORD_ALIGNED_BASE - 24),
            true,
        );
        let hdr = BlockHeader::parse(
            &enc.bytes,
            BlockLayout::Compact,
            Some(WORD_ALIGNED_BASE - 24),
        );
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
                bitset_next_doc(&enc.bytes, &hdr, edge),
                Some((edge, rank)),
                "edge doc {edge}"
            );
            assert_eq!(
                bitset_tf_at(&enc.bytes, &hdr, rank),
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
                bitset_next_doc(&enc.bytes, &hdr, from),
                Some((next, rank_of(next))),
                "absent doc {from} resolves to {next}"
            );
        }
        // An absent doc just after an edge resolves to the one after it.
        assert_eq!(
            bitset_next_doc(&enc.bytes, &hdr, WORD_ALIGNED_BASE + 65),
            Some((WORD_ALIGNED_BASE + 66, rank_of(WORD_ALIGNED_BASE + 66)))
        );
        // Past the last bit: nothing.
        assert_eq!(
            bitset_next_doc(&enc.bytes, &hdr, WORD_ALIGNED_BASE + 128),
            None
        );
    }
}
