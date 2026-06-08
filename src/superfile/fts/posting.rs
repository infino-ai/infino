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

use bitpacking::{BitPacker, BitPacker4x};

/// Number of `(doc_id, tf)` pairs per encoded block. Fixed at 128 to
/// match `BitPacker4x::BLOCK_LEN`.
pub const BLOCK_LEN: usize = BitPacker4x::BLOCK_LEN;

/// Header size in bytes (doc_count + delta_bits + tf_bits + reserved +
/// base_doc_id).
pub const HEADER_SIZE: usize = 8;

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
    let mut bytes = Vec::with_capacity(HEADER_SIZE + deltas_size + tfs_size);

    // Header.
    bytes.push(count as u8);
    bytes.push(delta_bits);
    bytes.push(tf_bits);
    bytes.push(0); // reserved
    bytes.extend_from_slice(&base_doc_id.to_le_bytes());

    // Packed deltas.
    let deltas_start = bytes.len();
    bytes.resize(deltas_start + deltas_size, 0);
    bp.compress_sorted(
        base_doc_id,
        &padded_doc_ids,
        &mut bytes[deltas_start..deltas_start + deltas_size],
        delta_bits,
    );

    // Packed tfs.
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
    assert!(
        dest_doc_ids.len() >= BLOCK_LEN,
        "decode_block: dest_doc_ids must have at least {BLOCK_LEN} slots"
    );
    assert!(
        dest_tfs.len() >= BLOCK_LEN,
        "decode_block: dest_tfs must have at least {BLOCK_LEN} slots"
    );
    assert!(
        bytes.len() >= HEADER_SIZE,
        "decode_block: bytes too short for header"
    );

    let count = bytes[0] as usize;
    let delta_bits = bytes[1];
    let tf_bits = bytes[2];
    // bytes[3] = reserved, ignored.
    assert!(
        delta_bits <= 32,
        "decode_block: delta_bits {delta_bits} > 32"
    );
    assert!(tf_bits <= 32, "decode_block: tf_bits {tf_bits} > 32");
    assert!(
        count <= BLOCK_LEN,
        "decode_block: doc_count {count} > BLOCK_LEN"
    );
    let base_doc_id = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

    let deltas_size = BLOCK_LEN * delta_bits as usize / 8;
    let tfs_size = BLOCK_LEN * tf_bits as usize / 8;
    assert!(
        bytes.len() >= HEADER_SIZE + deltas_size + tfs_size,
        "decode_block: bytes ({}) shorter than header+deltas+tfs ({})",
        bytes.len(),
        HEADER_SIZE + deltas_size + tfs_size
    );

    let bp = BitPacker4x::new();
    let deltas_start = HEADER_SIZE;
    bp.decompress_sorted(
        base_doc_id,
        &bytes[deltas_start..deltas_start + deltas_size],
        &mut dest_doc_ids[..BLOCK_LEN],
        delta_bits,
    );

    let tfs_start = deltas_start + deltas_size;
    bp.decompress(
        &bytes[tfs_start..tfs_start + tfs_size],
        &mut dest_tfs[..BLOCK_LEN],
        tf_bits,
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

    /// Build doc_ids whose deltas top out at exactly `max_delta` and tfs
    /// whose values top out at `max_tf`.
    fn block_with_max_delta_and_tf(count: usize, max_delta: u32, max_tf: u32) -> Block {
        let mut doc_ids = Vec::with_capacity(count);
        let mut acc: u32 = 1;
        for i in 0..count {
            // Vary deltas to span [1, max_delta]; force at least one
            // entry to hit max_delta exactly.
            let delta = if i == count / 2 {
                max_delta
            } else {
                1 + (i as u32 % max_delta.max(1))
            };
            acc = acc.checked_add(delta).expect("overflow in test setup");
            doc_ids.push(acc);
        }
        let tfs: Vec<u32> = (0..count)
            .map(|i| {
                if i == count / 3 {
                    max_tf
                } else {
                    (i as u32 % max_tf.max(1)) + 1
                }
            })
            .collect();
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
    fn bit_width_31_just_below_word_boundary() {
        // Max delta of 2^30: needs 31 bits.
        let b = block_with_max_delta_and_tf(8, 1 << 30, 1 << 30);
        let enc = roundtrip(&b);
        assert_eq!(enc.bytes[1], 31);
        assert_eq!(enc.bytes[2], 31);
    }

    #[test]
    fn bit_width_32_full_word() {
        // Maximum possible delta: 2^31 (still fits in u32, needs 32 bits).
        let b = block_with_max_delta_and_tf(4, 1 << 31, 1 << 31);
        let enc = roundtrip(&b);
        assert_eq!(enc.bytes[1], 32);
        assert_eq!(enc.bytes[2], 32);
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
    fn header_reserved_byte_is_zero() {
        let enc = encode_block(&block(&[1, 2, 3], &[1, 1, 1]));
        assert_eq!(enc.bytes[3], 0, "reserved byte must be 0");
    }

    #[test]
    fn header_base_doc_id_is_first_minus_one() {
        let enc = encode_block(&block(&[100, 102, 105], &[1, 1, 1]));
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
        // Wide deltas, narrow tfs.
        let doc_ids: Vec<u32> = (0..16).map(|i| i * 1024).collect(); // delta = 1024 → 11 bits
        let tfs: Vec<u32> = (0..16).map(|_| 1).collect();
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
            let delta_bits = enc.bytes[1] as usize;
            let tf_bits = enc.bytes[2] as usize;
            let expected_len = 8 /* header */
                + (BLOCK_LEN * delta_bits) / 8
                + (BLOCK_LEN * tf_bits) / 8;
            prop_assert_eq!(enc.bytes.len(), expected_len);
        }
    }
}
