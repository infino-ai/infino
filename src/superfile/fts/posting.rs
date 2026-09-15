// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! PFOR-delta block codec for posting lists.
//!
//! Postings are encoded in fixed-size **128-doc blocks** (matching
//! `BitPacker4x::BLOCK_LEN` for SIMD-friendly bit-packing). Each block
//! stores a sorted run of `doc_ids` (delta-encoded relative to a stored
//! `base_doc_id`) plus per-doc `tfs`, both bit-packed at the minimum
//! width needed for that block. Blocks are independently decodable —
//! no shared state between blocks — so the skip-table can jump straight
//! to a block by byte offset.
//!
//! See `docs/architecture/superfile.md` for the overall posting region
//! layout and how blocks chain into the BM25 + BlockMaxWAND query loop.
//!
//! ## On-disk block layout
//!
//! ```text
//!   offset  bytes   field
//!   ─────────────────────────────────────────────────────────────────
//!   0       1       doc_count           (u8; 1..=128)
//!   1       1       delta_bits          (u8; 0..=32, bit-width for deltas)
//!   2       1       tf_bits             (u8; 0..=32, bit-width for tfs)
//!   3       1       reserved            (must be 0)
//!   4       4       base_doc_id (LE u32; passed as `initial` to the
//!                   bitpacker so deltas are computed relative to it)
//!   8       16 × delta_bits  packed deltas (always BLOCK_LEN values)
//!   ...     16 × tf_bits     packed tfs    (always BLOCK_LEN values)
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

use crate::superfile::fts::positions::{push_varint, read_varint, varint_len};

/// Number of `(doc_id, tf)` pairs per encoded block. Fixed at 128 to
/// match `BitPacker4x::BLOCK_LEN`.
pub const BLOCK_LEN: usize = BitPacker4x::BLOCK_LEN;

/// Header size in bytes (doc_count + delta_bits + tf_bits + encoding +
/// base_doc_id).
pub const HEADER_SIZE: usize = 8;

/// Header byte offset of the block `encoding` field (formerly reserved).
pub const ENCODING_OFF: usize = 3;

/// Block `encoding` (header byte 3): doc ids stored as PFOR-delta packing
/// (today's layout). Every `V1`–`V3` block is this.
pub const ENCODING_PACKED: u8 = 0;
/// Block `encoding`: **patched** packing (`VERSION_V7`). Doc-id deltas
/// and tfs are each bit-packed at a width most lanes fit, and the few
/// lanes that do not — the outliers that would otherwise set the width
/// for all 128 — store their high bits as **exceptions**. Layout after
/// the 8-byte header:
///
/// ```text
///   8       1       n_delta_exceptions (u8)
///   9       1       n_tf_exceptions    (u8)
///   10      16 × delta_bits  packed low bits of the deltas
///   ...     per delta exception: lane (u8), high bits (LEB128 u32)
///   ...     per tf exception:    lane (u8), high bits (LEB128 u32)
///   ...     16 × tf_bits     packed low bits of the tfs (trailing, as always)
/// ```
///
/// Deltas here are explicit (`doc[0] - base_doc_id`, then `doc[i] -
/// doc[i-1]`, padding lanes 0), prefix-summed by the reader after the
/// exceptions are patched in — the sorted bit-packer's fused prefix sum
/// cannot see a patch. Chosen only when it is the smallest of the three
/// encodings for the block, so a block with uniform widths stays
/// [`ENCODING_PACKED`] and decodes exactly as before.
pub const ENCODING_PATCHED: u8 = 2;
/// Bytes after the header that hold the two exception counts of a
/// patched block.
pub const PATCHED_COUNTS_SIZE: usize = 2;
/// Most exception lanes a patched stream may carry. Bounds the patch
/// loop the decoder runs; past this the stream is better off wider.
const PATCHED_MAX_EXCEPTIONS: usize = 32;

/// Block `encoding`: doc ids stored as a **presence bitset** over
/// `[base_doc_id, last_doc_id]`, `base_doc_id` aligned down to a 64-bit
/// word so the union count can OR it in word-aligned. Chosen only when it
/// does not grow the block (dense blocks). Tfs follow, packed identically
/// to PACKED. Only `VERSION_V4` blobs contain these.
pub const ENCODING_BITSET: u8 = 1;

/// Align a doc id down to the 64-bit word that contains it — the origin of
/// a [`ENCODING_BITSET`] block's presence bitset, so its words line up
/// with the union bitset and the OR needs no per-word bit shift.
#[inline]
pub fn bitset_block_base(doc_id: u32) -> u32 {
    doc_id & !63
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

/// A patched stream's plan: the width most lanes fit and the lanes that
/// do not, with their high bits.
struct PatchPlan {
    width: u8,
    exceptions: Vec<(u8, u32)>,
    /// Bytes the packed low bits plus the exception list take.
    bytes: usize,
}

/// Plan the cheapest patched packing of `lanes` (all `BLOCK_LEN` of
/// them). Tries every width below the plain width and keeps the one
/// whose packed bits plus exception list is smallest, subject to
/// [`PATCHED_MAX_EXCEPTIONS`]. Returns the plain width's plan when
/// nothing beats it.
fn plan_patched(lanes: &[u32; BLOCK_LEN]) -> PatchPlan {
    let plain_bits = width_bits(lanes.iter().copied().max().unwrap_or(0));
    let mut best = PatchPlan {
        width: plain_bits,
        exceptions: Vec::new(),
        bytes: BLOCK_LEN * plain_bits as usize / 8,
    };
    for width in 0..plain_bits {
        let mut exceptions = Vec::new();
        let mut bytes = BLOCK_LEN * width as usize / 8;
        for (i, &v) in lanes.iter().enumerate() {
            let hi = if width == 0 { v } else { v >> width };
            if hi != 0 {
                exceptions.push((i as u8, hi));
                bytes += 1 + varint_len(hi);
                if exceptions.len() > PATCHED_MAX_EXCEPTIONS || bytes >= best.bytes {
                    break;
                }
            }
        }
        if exceptions.len() <= PATCHED_MAX_EXCEPTIONS && bytes < best.bytes {
            best = PatchPlan {
                width,
                exceptions,
                bytes,
            };
        }
    }
    best
}

/// Bits needed to hold `v` (zero for zero).
#[inline]
fn width_bits(v: u32) -> u8 {
    (u32::BITS - v.leading_zeros()) as u8
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

/// Encode one block.
///
/// # Panics
///
/// - `b.doc_ids.is_empty()` — can't encode an empty block.
/// - `b.doc_ids.len() != b.tfs.len()` — mismatched parallel vectors.
/// - `b.doc_ids.len() > BLOCK_LEN` — caller must split into 128-doc
///   chunks before calling.
/// - `b.doc_ids` not strictly increasing — the codec assumes sorted
///   unique doc_ids (debug-only check).
pub fn encode_block(b: &Block) -> EncodedBlock {
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

    // `initial` is the value the decoder uses to recover doc_ids[0]
    // (decompressed[0] = initial + delta[0]). Choose `doc_ids[0] - 1`
    // so the smallest delta is 1 and bit-width is tight; clamp at 0
    // for the doc_ids[0] == 0 case (delta[0] = 0).
    let base_doc_id = b.doc_ids[0].saturating_sub(1);

    let bp = BitPacker4x::new();
    let delta_bits = bp.num_bits_sorted(base_doc_id, &padded_doc_ids);
    let tf_bits = bp.num_bits(&padded_tfs);

    let deltas_size = BLOCK_LEN * delta_bits as usize / 8;
    let tfs_size = BLOCK_LEN * tf_bits as usize / 8;

    // Store the doc ids as a presence bitset instead of PFOR deltas when
    // that does not grow the block (a dense block — a common term's ~128
    // near-consecutive docs). The bitset origin is word-aligned so the
    // union count can OR it in without a per-word shift.
    let aligned_base = bitset_block_base(b.doc_ids[0]);
    let bitset_words = (last_doc_id - aligned_base) as usize / 64 + 1;
    let bitset_size = bitset_words * 8;
    let use_bitset = bitset_size <= deltas_size;

    // Patched packing: explicit deltas and tfs, each at the width most
    // lanes fit plus an exception list for the rest. Considered only for a
    // block the bitset did not claim: a dense block's presence words are
    // what the count kernels bit-test and rank into, and that O(1) probe
    // is worth more than the few bytes patching a dense partial block
    // would save.
    let mut explicit_deltas = [0u32; BLOCK_LEN];
    explicit_deltas[0] = b.doc_ids[0] - base_doc_id;
    for (slot, pair) in explicit_deltas[1..count]
        .iter_mut()
        .zip(b.doc_ids.windows(2))
    {
        *slot = pair[1] - pair[0];
    }
    let delta_plan = plan_patched(&explicit_deltas);
    let tf_plan = plan_patched(&padded_tfs);
    let patched_size = HEADER_SIZE + PATCHED_COUNTS_SIZE + delta_plan.bytes + tf_plan.bytes;
    let plain_size = HEADER_SIZE + deltas_size + tfs_size;
    if !use_bitset && patched_size < plain_size {
        let mut bytes = Vec::with_capacity(patched_size);
        bytes.push(count as u8);
        bytes.push(delta_plan.width);
        bytes.push(tf_plan.width);
        bytes.push(ENCODING_PATCHED);
        bytes.extend_from_slice(&base_doc_id.to_le_bytes());
        bytes.push(delta_plan.exceptions.len() as u8);
        bytes.push(tf_plan.exceptions.len() as u8);
        let deltas_start = bytes.len();
        let deltas_packed = BLOCK_LEN * delta_plan.width as usize / 8;
        bytes.resize(deltas_start + deltas_packed, 0);
        compress_low_bits(
            &bp,
            &explicit_deltas,
            delta_plan.width,
            &mut bytes[deltas_start..],
        );
        for &(lane, hi) in &delta_plan.exceptions {
            bytes.push(lane);
            push_varint(&mut bytes, hi);
        }
        for &(lane, hi) in &tf_plan.exceptions {
            bytes.push(lane);
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

    let doc_ids_size = if use_bitset { bitset_size } else { deltas_size };
    let mut bytes = Vec::with_capacity(HEADER_SIZE + doc_ids_size + tfs_size);

    // Header.
    bytes.push(count as u8);
    bytes.push(if use_bitset { 0 } else { delta_bits });
    bytes.push(tf_bits);
    bytes.push(if use_bitset {
        ENCODING_BITSET
    } else {
        ENCODING_PACKED
    });
    bytes.extend_from_slice(
        &if use_bitset {
            aligned_base
        } else {
            base_doc_id
        }
        .to_le_bytes(),
    );

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

/// Decode one block. `dest_doc_ids` and `dest_tfs` must each have at
/// least [`BLOCK_LEN`] elements; the decoder writes all `BLOCK_LEN`
/// slots for SIMD reasons. The returned `doc_count` tells the caller
/// how many of those slots are real (the rest are padding values
/// — zero deltas for doc_ids, zero tfs).
///
/// # Panics
///
/// - `bytes.len() < HEADER_SIZE`.
/// - `bytes` is shorter than the header claims.
/// - `dest_doc_ids.len() < BLOCK_LEN` or `dest_tfs.len() < BLOCK_LEN`.
/// - Header reports `delta_bits > 32` or `tf_bits > 32`.
pub fn decode_block(bytes: &[u8], dest_doc_ids: &mut [u32], dest_tfs: &mut [u32]) -> usize {
    let count = decode_block_doc_ids(bytes, dest_doc_ids);

    // Term frequencies are always the trailing `tfs_size` bytes of the
    // block, regardless of how the doc ids ahead of them are encoded
    // (PACKED deltas or a BITSET). The count paths skip this half (see
    // `decode_block_doc_ids`); only scoring needs it.
    assert!(
        dest_tfs.len() >= BLOCK_LEN,
        "decode_block: dest_tfs must have at least {BLOCK_LEN} slots"
    );
    let tf_bits = bytes[2];
    assert!(tf_bits <= 32, "decode_block: tf_bits {tf_bits} > 32");
    let tfs_size = BLOCK_LEN * tf_bits as usize / 8;
    assert!(
        bytes.len() >= HEADER_SIZE + tfs_size,
        "decode_block: bytes ({}) shorter than header+tfs ({})",
        bytes.len(),
        HEADER_SIZE + tfs_size
    );
    let tfs_start = bytes.len() - tfs_size;
    BitPacker4x::new().decompress(
        &bytes[tfs_start..tfs_start + tfs_size],
        &mut dest_tfs[..BLOCK_LEN],
        tf_bits,
    );
    if bytes[ENCODING_OFF] == ENCODING_PATCHED {
        let (_, tf_exc) = patched_exception_ranges(bytes);
        apply_exceptions(&bytes[tf_exc], tf_bits, dest_tfs);
    }

    count
}

/// Byte ranges of a patched block's delta and tf exception lists.
///
/// # Panics
///
/// `bytes` is not a well-formed patched block (the CRC-validated
/// postings region is the caller's guarantee).
pub(crate) fn patched_exception_ranges(bytes: &[u8]) -> (Range<usize>, Range<usize>) {
    let n_delta_exc = bytes[HEADER_SIZE] as usize;
    let n_tf_exc = bytes[HEADER_SIZE + 1] as usize;
    let delta_bits = bytes[1] as usize;
    let delta_exc_start = HEADER_SIZE + PATCHED_COUNTS_SIZE + BLOCK_LEN * delta_bits / 8;
    let mut at = delta_exc_start;
    for _ in 0..n_delta_exc {
        at += 1;
        read_varint(bytes, &mut at).expect("patched block exception within block");
    }
    let tf_exc_start = at;
    for _ in 0..n_tf_exc {
        at += 1;
        read_varint(bytes, &mut at).expect("patched block exception within block");
    }
    (delta_exc_start..tf_exc_start, tf_exc_start..at)
}

/// OR each exception's high bits back into its lane of `dest`.
#[inline]
fn apply_exceptions(exceptions: &[u8], width: u8, dest: &mut [u32]) {
    let mut at = 0usize;
    while at < exceptions.len() {
        let lane = exceptions[at] as usize;
        at += 1;
        let hi = read_varint(exceptions, &mut at).expect("patched block exception within block");
        dest[lane] |= hi << width;
    }
}

/// Decode only the tf array of a block (the trailing tf-packed bytes) into
/// `dest_tfs`, in doc order, skipping the doc-id half. The ranked-OR
/// membership probe locates a bitset-block doc by bit-test + popcount-rank
/// and needs only that doc's tf, never the expanded doc ids — so it decodes
/// the tfs (which are BitPacker4x-packed and can't be single-value-indexed)
/// once per block and reads the rank-th one, avoiding the doc-id expansion.
pub fn decode_block_tfs(bytes: &[u8], dest_tfs: &mut [u32]) {
    assert!(
        dest_tfs.len() >= BLOCK_LEN,
        "decode_block_tfs: dest_tfs must have at least {BLOCK_LEN} slots"
    );
    let tf_bits = bytes[2];
    assert!(tf_bits <= 32, "decode_block_tfs: tf_bits {tf_bits} > 32");
    let tfs_size = BLOCK_LEN * tf_bits as usize / 8;
    assert!(
        bytes.len() >= HEADER_SIZE + tfs_size,
        "decode_block_tfs: bytes shorter than header+tfs"
    );
    let tfs_start = bytes.len() - tfs_size;
    BitPacker4x::new().decompress(
        &bytes[tfs_start..tfs_start + tfs_size],
        &mut dest_tfs[..BLOCK_LEN],
        tf_bits,
    );
    if bytes[ENCODING_OFF] == ENCODING_PATCHED {
        let (_, tf_exc) = patched_exception_ranges(bytes);
        apply_exceptions(&bytes[tf_exc], tf_bits, dest_tfs);
    }
}

/// Decode only the doc ids of a posting block, skipping the term-frequency
/// half entirely. Unranked counts (union, intersection) never look at
/// `tf` — they tally doc ids — so decoding + reading the packed tfs is
/// pure waste there; this halves the per-block decode work on the count
/// path. Returns the doc count. Shared by [`decode_block`], which appends
/// the tf decode for the scoring path.
pub fn decode_block_doc_ids(bytes: &[u8], dest_doc_ids: &mut [u32]) -> usize {
    assert!(
        dest_doc_ids.len() >= BLOCK_LEN,
        "decode_block_doc_ids: dest_doc_ids must have at least {BLOCK_LEN} slots"
    );
    assert!(
        bytes.len() >= HEADER_SIZE,
        "decode_block_doc_ids: bytes too short for header"
    );

    let count = bytes[0] as usize;
    let delta_bits = bytes[1];
    let tf_bits = bytes[2];
    let encoding = bytes[3];
    assert!(
        delta_bits <= 32,
        "decode_block_doc_ids: delta_bits {delta_bits} > 32"
    );
    assert!(
        count <= BLOCK_LEN,
        "decode_block_doc_ids: doc_count {count} > BLOCK_LEN"
    );
    let base_doc_id = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

    if encoding == ENCODING_BITSET {
        // Doc ids are a presence bitset over `[base_doc_id, ...]`; the tfs
        // are the trailing `tf_bits`-packed bytes, so the bitset is
        // everything between the header and them. Emit the set bits in
        // ascending order (= ascending doc id) — the sorted order every
        // consumer expects.
        let tfs_size = BLOCK_LEN * tf_bits as usize / 8;
        assert!(
            bytes.len() >= HEADER_SIZE + tfs_size,
            "decode_block_doc_ids: bytes shorter than header+tfs"
        );
        let words = &bytes[HEADER_SIZE..bytes.len() - tfs_size];
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
                dest_doc_ids[j] = base_doc_id + (wi as u32 * 64 + word.trailing_zeros());
                j += 1;
                word &= word - 1;
            }
        }
        debug_assert_eq!(j, count, "bitset set-bit count must equal doc_count");
        return count;
    }

    let deltas_size = BLOCK_LEN * delta_bits as usize / 8;
    if encoding == ENCODING_PATCHED {
        // Explicit deltas at the narrow width, exceptions patched in,
        // then the prefix sum the sorted packer would have fused.
        let deltas_start = HEADER_SIZE + PATCHED_COUNTS_SIZE;
        assert!(
            bytes.len() >= deltas_start + deltas_size,
            "decode_block_doc_ids: bytes shorter than header+deltas"
        );
        BitPacker4x::new().decompress(
            &bytes[deltas_start..deltas_start + deltas_size],
            &mut dest_doc_ids[..BLOCK_LEN],
            delta_bits,
        );
        let (delta_exc, _) = patched_exception_ranges(bytes);
        apply_exceptions(&bytes[delta_exc], delta_bits, dest_doc_ids);
        let mut prev = base_doc_id;
        for d in dest_doc_ids[..BLOCK_LEN].iter_mut() {
            prev = prev.wrapping_add(*d);
            *d = prev;
        }
        return count;
    }
    assert!(
        bytes.len() >= HEADER_SIZE + deltas_size,
        "decode_block_doc_ids: bytes ({}) shorter than header+deltas ({})",
        bytes.len(),
        HEADER_SIZE + deltas_size
    );

    BitPacker4x::new().decompress_sorted(
        base_doc_id,
        &bytes[HEADER_SIZE..HEADER_SIZE + deltas_size],
        &mut dest_doc_ids[..BLOCK_LEN],
        delta_bits,
    );

    count
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a Block from parallel slices.
    fn block(doc_ids: &[u32], tfs: &[u32]) -> Block {
        Block {
            doc_ids: doc_ids.to_vec(),
            tfs: tfs.to_vec(),
        }
    }

    /// Encode then decode; assert round-trip.
    fn roundtrip(b: &Block) -> EncodedBlock {
        let enc = encode_block(b);
        let mut got_doc_ids = vec![0u32; BLOCK_LEN];
        let mut got_tfs = vec![0u32; BLOCK_LEN];
        let count = decode_block(&enc.bytes, &mut got_doc_ids, &mut got_tfs);
        assert_eq!(count, b.doc_ids.len(), "doc_count round-trip");
        assert_eq!(
            &got_doc_ids[..count],
            b.doc_ids.as_slice(),
            "doc_ids round-trip"
        );
        assert_eq!(&got_tfs[..count], b.tfs.as_slice(), "tfs round-trip");
        enc
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
        let enc = roundtrip(&block(&doc_ids, &tfs));
        assert_eq!(enc.bytes[ENCODING_OFF], ENCODING_PATCHED);
        assert!(enc.bytes.len() < 64, "got {} bytes", enc.bytes.len());
        // The tf-only decode patches too.
        let mut got_tfs = vec![0u32; BLOCK_LEN];
        decode_block_tfs(&enc.bytes, &mut got_tfs);
        assert_eq!(&got_tfs[..128], tfs.as_slice());
        // The doc-id-only decode patches and prefix-sums.
        let mut got_ids = vec![0u32; BLOCK_LEN];
        assert_eq!(decode_block_doc_ids(&enc.bytes, &mut got_ids), 128);
        assert_eq!(&got_ids[..128], doc_ids.as_slice());
    }

    #[test]
    fn uniform_blocks_stay_packed_byte_for_byte() {
        // No outlier ⇒ the patched plan cannot beat plain, so the block
        // is the PACKED layout older readers know.
        let doc_ids: Vec<u32> = (0..128).map(|i| 10 + 37 * i).collect();
        let tfs: Vec<u32> = (0..128).map(|i| 1 + i % 4).collect();
        let enc = roundtrip(&block(&doc_ids, &tfs));
        assert_eq!(enc.bytes[ENCODING_OFF], ENCODING_PACKED);
    }

    #[test]
    fn partial_patched_block_round_trips() {
        let doc_ids = vec![5u32, 6, 7, 5_000_000];
        let tfs = vec![1u32, 1, 1, 1];
        let enc = roundtrip(&block(&doc_ids, &tfs));
        assert_eq!(enc.bytes[ENCODING_OFF], ENCODING_PATCHED);
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
        // Sanity on byte layout: header + 16*1 + 16*1 = 40 bytes.
        assert_eq!(enc.bytes.len(), HEADER_SIZE + 16 + 16);
        // 1-bit deltas beat a bitset here, so it stays PACKED.
        assert_eq!(enc.bytes[3], ENCODING_PACKED);
    }

    #[test]
    fn roundtrip_dense_block_uses_bitset_encoding() {
        // 128 docs at stride 2 (span 254): 2-bit deltas, so the bitset
        // (≤ the packed deltas) is chosen. Round-trips through the bitset
        // decode path and its set-bit ordering.
        let doc_ids: Vec<u32> = (0..128).map(|i| i * 2).collect();
        let tfs: Vec<u32> = (0..128).map(|i| (i % 5) + 1).collect();
        let enc = roundtrip(&block(&doc_ids, &tfs));
        assert_eq!(enc.bytes[3], ENCODING_BITSET, "dense block must be bitset");
        assert_eq!(enc.last_doc_id, 254);
        // base aligned down to a 64-bit boundary (0 here).
        assert_eq!(
            u32::from_le_bytes(enc.bytes[4..8].try_into().expect("4 bytes")),
            0
        );
    }

    #[test]
    fn roundtrip_bitset_block_nonzero_aligned_base() {
        // Dense block far from the origin: base aligns down to a word.
        let doc_ids: Vec<u32> = (0..100).map(|i| 10_000 + i * 2).collect();
        let tfs = vec![1u32; 100];
        let enc = roundtrip(&block(&doc_ids, &tfs));
        assert_eq!(enc.bytes[3], ENCODING_BITSET);
        assert_eq!(
            u32::from_le_bytes(enc.bytes[4..8].try_into().expect("4 bytes")),
            10_000 & !63,
            "base is word-aligned"
        );
    }

    #[test]
    fn roundtrip_sparse_block_stays_packed() {
        // Widely-scattered docs: a bitset would be huge, so PACKED wins.
        let doc_ids: Vec<u32> = (0..64).map(|i| i * 100_000).collect();
        let tfs = vec![1u32; 64];
        let enc = roundtrip(&block(&doc_ids, &tfs));
        assert_eq!(
            enc.bytes[3], ENCODING_PACKED,
            "sparse block must stay packed"
        );
    }

    #[test]
    fn roundtrip_partial_block_single_doc() {
        let enc = roundtrip(&block(&[42], &[1]));
        assert_eq!(enc.last_doc_id, 42);
        assert_eq!(enc.max_tf, 1);
    }

    #[test]
    fn roundtrip_partial_block_50_docs() {
        let doc_ids: Vec<u32> = (0..50).map(|i| i * 3).collect();
        let tfs: Vec<u32> = (0..50).map(|i| (i % 7) + 1).collect();
        let enc = roundtrip(&block(&doc_ids, &tfs));
        assert_eq!(enc.last_doc_id, 49 * 3);
        assert_eq!(enc.max_tf, 7);
    }

    #[test]
    fn roundtrip_first_doc_is_zero() {
        // doc_ids[0] = 0 forces base_doc_id = 0 (saturating_sub).
        let enc = roundtrip(&block(&[0, 1, 5, 10], &[1, 1, 2, 1]));
        assert_eq!(enc.last_doc_id, 10);
    }

    #[test]
    fn roundtrip_first_doc_is_u32_max_minus_n() {
        // Far-end of the u32 range to exercise the upper bits.
        let max = u32::MAX;
        let doc_ids = vec![max - 200, max - 100, max - 50, max - 1];
        let tfs = vec![1u32, 2, 3, 4];
        let enc = roundtrip(&block(&doc_ids, &tfs));
        assert_eq!(enc.last_doc_id, max - 1);
        assert_eq!(enc.max_tf, 4);
    }

    // --- Bit-width edge cases (1, 7, 8, 31, 32) -------------------------

    /// Build a **full** block whose every delta is exactly `max_delta` and
    /// every tf exactly `max_tf`. Uniform lanes, so the patched encoding
    /// cannot beat plain packing and the header pins the plain widths.
    fn block_with_max_delta_and_tf(count: usize, max_delta: u32, max_tf: u32) -> Block {
        let count = BLOCK_LEN.min(count.max(1));
        let mut doc_ids = Vec::with_capacity(count);
        let mut acc: u32 = 1;
        for _ in 0..count {
            let delta = max_delta.max(1);
            acc = acc.checked_add(delta).expect("overflow in test setup");
            doc_ids.push(acc);
        }
        let tfs = vec![max_tf; count];
        Block { doc_ids, tfs }
    }

    #[test]
    fn bit_width_1_for_dense_postings() {
        // Deltas all 1, tfs all 1 → bit_width 1 for both.
        let doc_ids: Vec<u32> = (10..138).collect();
        let tfs = vec![1u32; 128];
        let enc = roundtrip(&block(&doc_ids, &tfs));
        // delta_bits + tf_bits = 1 + 1 = 2; payload = 16 + 16 = 32 bytes
        assert_eq!(enc.bytes[1], 1, "delta_bits");
        assert_eq!(enc.bytes[2], 1, "tf_bits");
    }

    #[test]
    fn bit_width_7_just_below_byte_boundary() {
        let b = block_with_max_delta_and_tf(64, 0x7F /* 127 */, 0x7F);
        let enc = roundtrip(&b);
        assert_eq!(enc.bytes[1], 7, "delta_bits should be 7 for max delta 127");
        assert_eq!(enc.bytes[2], 7, "tf_bits should be 7 for max tf 127");
    }

    #[test]
    fn bit_width_8_at_byte_boundary() {
        let b = block_with_max_delta_and_tf(64, 0x80 /* 128 */, 0x80);
        let enc = roundtrip(&b);
        assert_eq!(enc.bytes[1], 8);
        assert_eq!(enc.bytes[2], 8);
    }

    #[test]
    fn widths_31_and_32_round_trip_losslessly() {
        // Values needing 31 and 32 bits — a full block of them would
        // overflow a u32 doc id, so these are partial blocks, which take
        // the patched form (padding lanes narrow, the wide lanes as
        // exceptions). What is pinned is that nothing is lost at the top
        // of the range, in either stream.
        roundtrip(&block(
            &[1 << 30, (1 << 30) + (1 << 30)],
            &[1 << 30, 1 << 30],
        ));
        roundtrip(&block(&[7, 7 + (1 << 31)], &[1 << 31, u32::MAX]));
        roundtrip(&block(&[u32::MAX - 1, u32::MAX], &[u32::MAX, 1]));
    }

    // --- All-zero tfs (bit_width 0) -------------------------------------

    #[test]
    fn bit_width_0_for_all_zero_tfs() {
        // Defensive: even though tf=0 is not produced by the FTS
        // pipeline, the codec must handle it cleanly (bit_width 0
        // means the packed-tfs region is zero bytes).
        let doc_ids: Vec<u32> = (1..=128).collect();
        let tfs = vec![0u32; 128];
        let enc = roundtrip(&block(&doc_ids, &tfs));
        assert_eq!(enc.bytes[2], 0, "tf_bits should be 0 for all-zero tfs");
        assert_eq!(enc.max_tf, 0);
        // Payload: header + 16*1 (deltas) + 0 (tfs) = 24 bytes.
        assert_eq!(enc.bytes.len(), HEADER_SIZE + 16);
    }

    // --- Header layout / metadata --------------------------------------

    #[test]
    fn encoded_block_carries_last_doc_id_and_max_tf() {
        let doc_ids: Vec<u32> = vec![5, 10, 15, 20, 25];
        let tfs: Vec<u32> = vec![1, 4, 2, 9, 3];
        let enc = encode_block(&block(&doc_ids, &tfs));
        assert_eq!(enc.last_doc_id, 25);
        assert_eq!(enc.max_tf, 9);
    }

    #[test]
    fn header_doc_count_round_trips() {
        for count in [1usize, 2, 31, 32, 33, 63, 64, 65, 127, 128] {
            let doc_ids: Vec<u32> = (1..=count as u32).collect();
            let tfs = vec![1u32; count];
            let enc = encode_block(&block(&doc_ids, &tfs));
            assert_eq!(
                enc.bytes[0] as usize, count,
                "header.doc_count for n={count}"
            );
        }
    }

    #[test]
    fn header_encoding_byte_marks_the_layout() {
        // Byte 3 (formerly reserved) is the encoding: 0 = PACKED (a full
        // block of uniform wide deltas, where nothing beats plain packing),
        // 1 = BITSET (dense), 2 = PATCHED (a sparse partial block: the
        // padding lanes pack at width 0 and the real deltas ride as
        // exceptions).
        let uniform: Vec<u32> = (0..128).map(|i| 1 + 100_000 * i).collect();
        let packed = encode_block(&block(&uniform, &[1; 128]));
        assert_eq!(packed.bytes[3], ENCODING_PACKED, "uniform ⇒ PACKED");
        let bitset = encode_block(&block(&[1, 2, 3], &[1, 1, 1]));
        assert_eq!(bitset.bytes[3], ENCODING_BITSET, "dense ⇒ BITSET");
        let patched = encode_block(&block(&[1, 100_000], &[1, 1]));
        assert_eq!(
            patched.bytes[3], ENCODING_PATCHED,
            "sparse partial ⇒ PATCHED"
        );
    }

    #[test]
    fn header_base_doc_id_is_first_minus_one() {
        // A non-bitset block stores base = first doc - 1, whichever of the
        // two packings it takes.
        let enc = encode_block(&block(&[100, 100_000, 200_000], &[1, 1, 1]));
        assert_ne!(enc.bytes[3], ENCODING_BITSET);
        let base_le = u32::from_le_bytes([enc.bytes[4], enc.bytes[5], enc.bytes[6], enc.bytes[7]]);
        assert_eq!(base_le, 99);
    }

    #[test]
    fn header_base_doc_id_clamps_at_zero() {
        let enc = encode_block(&block(&[0, 1, 2], &[1, 1, 1]));
        let base_le = u32::from_le_bytes([enc.bytes[4], enc.bytes[5], enc.bytes[6], enc.bytes[7]]);
        assert_eq!(base_le, 0, "saturating_sub at 0");
    }

    // --- Mixed bit widths between deltas and tfs ------------------------

    #[test]
    fn delta_and_tf_use_independent_bit_widths() {
        // Wide deltas, narrow tfs — a full block of uniform lanes, so the
        // header carries the plain widths.
        let doc_ids: Vec<u32> = (0..128).map(|i| i * 1024).collect(); // delta = 1024 → 11 bits
        let tfs: Vec<u32> = (0..128).map(|_| 1).collect();
        let enc = roundtrip(&block(&doc_ids, &tfs));
        let dbits = enc.bytes[1];
        let tbits = enc.bytes[2];
        assert!(
            (10..=12).contains(&dbits),
            "expected ~11 delta bits, got {dbits}"
        );
        assert_eq!(tbits, 1);
    }

    // --- Panic surface for invalid input -------------------------------

    #[test]
    #[should_panic(expected = "empty block")]
    fn encode_block_panics_on_empty() {
        let _ = encode_block(&Block {
            doc_ids: vec![],
            tfs: vec![],
        });
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn encode_block_panics_on_length_mismatch() {
        let _ = encode_block(&Block {
            doc_ids: vec![1, 2],
            tfs: vec![1],
        });
    }

    #[test]
    #[should_panic(expected = "BLOCK_LEN")]
    fn encode_block_panics_on_oversize() {
        let doc_ids: Vec<u32> = (1..=(BLOCK_LEN as u32 + 1)).collect();
        let tfs = vec![1u32; BLOCK_LEN + 1];
        let _ = encode_block(&Block { doc_ids, tfs });
    }

    #[test]
    #[should_panic(expected = "header")]
    fn decode_block_panics_on_short_input() {
        let mut d = vec![0u32; BLOCK_LEN];
        let mut t = vec![0u32; BLOCK_LEN];
        let _ = decode_block(&[0u8; 4], &mut d, &mut t);
    }

    #[test]
    #[should_panic(expected = "must have at least")]
    fn decode_block_panics_on_undersized_dest() {
        let enc = encode_block(&block(&[1, 2, 3], &[1, 1, 1]));
        let mut d = vec![0u32; BLOCK_LEN - 1];
        let mut t = vec![0u32; BLOCK_LEN];
        let _ = decode_block(&enc.bytes, &mut d, &mut t);
    }

    // --- Cross-block independence --------------------------------------

    #[test]
    fn blocks_with_disjoint_doc_id_ranges_decode_independently() {
        // Two blocks, each self-contained. Decoder doesn't need any
        // cross-block state.
        let b1 = block(&[1, 2, 3, 4, 5], &[1, 2, 1, 2, 1]);
        let b2 = block(&[1000, 1001, 1010, 1100], &[5, 1, 3, 9]);
        let enc1 = encode_block(&b1);
        let enc2 = encode_block(&b2);
        // Decode in opposite order to confirm zero shared state.
        let mut d2 = vec![0u32; BLOCK_LEN];
        let mut t2 = vec![0u32; BLOCK_LEN];
        let n2 = decode_block(&enc2.bytes, &mut d2, &mut t2);
        let mut d1 = vec![0u32; BLOCK_LEN];
        let mut t1 = vec![0u32; BLOCK_LEN];
        let n1 = decode_block(&enc1.bytes, &mut d1, &mut t1);
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
        for (d, t) in all_doc_ids.chunks(BLOCK_LEN).zip(all_tfs.chunks(BLOCK_LEN)) {
            encoded.push(encode_block(&block(d, t)));
        }

        let mut recovered_doc_ids = Vec::with_capacity(all_doc_ids.len());
        let mut recovered_tfs = Vec::with_capacity(all_tfs.len());
        let mut buf_d = vec![0u32; BLOCK_LEN];
        let mut buf_t = vec![0u32; BLOCK_LEN];
        for enc in &encoded {
            let n = decode_block(&enc.bytes, &mut buf_d, &mut buf_t);
            recovered_doc_ids.extend_from_slice(&buf_d[..n]);
            recovered_tfs.extend_from_slice(&buf_t[..n]);
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

            let block = Block { doc_ids: doc_ids.clone(), tfs: tfs.clone() };
            let enc = encode_block(&block);

            prop_assert_eq!(enc.last_doc_id, *doc_ids.last().expect("last element"));
            prop_assert_eq!(enc.max_tf, *tfs.iter().max().expect("iter max"));

            let mut got_doc_ids = vec![0u32; BLOCK_LEN];
            let mut got_tfs = vec![0u32; BLOCK_LEN];
            let count = decode_block(&enc.bytes, &mut got_doc_ids, &mut got_tfs);
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
                    (rng & 0xFFFF) as u32
                })
                .collect();

            let enc = encode_block(&Block { doc_ids, tfs });
            let tf_bits = enc.bytes[2] as usize;
            let tfs_size = (BLOCK_LEN * tf_bits) / 8;
            if enc.bytes[3] == ENCODING_BITSET {
                // Doc ids are a whole number of 64-bit words; tfs trail.
                prop_assert_eq!(enc.bytes[1], 0, "bitset block has delta_bits 0");
                let bitset_bytes = enc.bytes.len() - 8 - tfs_size;
                prop_assert!(bitset_bytes >= 8 && bitset_bytes.is_multiple_of(8));
            } else if enc.bytes[3] == ENCODING_PATCHED {
                let delta_bits = enc.bytes[1] as usize;
                let (delta_exc, tf_exc) = patched_exception_ranges(&enc.bytes);
                prop_assert_eq!(
                    enc.bytes.len(),
                    HEADER_SIZE
                        + PATCHED_COUNTS_SIZE
                        + (BLOCK_LEN * delta_bits) / 8
                        + delta_exc.len()
                        + tf_exc.len()
                        + tfs_size
                );
            } else {
                let delta_bits = enc.bytes[1] as usize;
                prop_assert_eq!(enc.bytes.len(), 8 + (BLOCK_LEN * delta_bits) / 8 + tfs_size);
            }
        }
    }
}
