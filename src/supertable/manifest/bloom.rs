// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS term-presence bloom filter.
//!
//! One bloom per (superfile, FTS column). Built once at commit time
//! by feeding the superfile's FST term iterator through a
//! [`BloomBuilder`]; queried at skip-prune time via
//! [`Bloom::contains`] to decide whether a superfile could contain
//! at least one of a query's terms. Returns `false` definitively
//! (skip the superfile) or `true` with the bloom's false-positive
//! rate (scan the superfile).
//!
//! # Algorithm: block bloom + XXH3-64 + portable SIMD bit-test
//!
//! - **Block bloom** (Putze et al., 2007): the bit array is split
//!   into fixed-size *blocks* (one cache line, 64 B = 512 bits).
//!   Each key is mapped to one block; both `insert` and `contains`
//!   touch only that block. Cache-line-bounded probes avoid the
//!   scattered random reads of a non-block bloom.
//! - **XXH3-64** as the hash backbone: ~3× faster than SipHash-1-3
//!   in the early-exit miss path that dominates skip-prune
//!   workloads. We don't need HashDoS resistance — terms come
//!   from a closed corpus, not adversarial input.
//! - **Portable SIMD via `wide::u64x4`**: `contains` builds a
//!   K-bit test mask once, then checks `(block & mask) == mask`
//!   in two SIMD AND-NOT-OR-reduce operations covering all 8
//!   block words. On x86_64 with AVX2 this lowers to two 256-bit
//!   `vpand` / `vptest` instructions; on aarch64 NEON it lowers
//!   to four 128-bit `bic`/`orr` instructions; elsewhere it
//!   falls back to scalar.
//! - **Kirsch-Mitzenmacher splitting with golden-ratio
//!   avalanche**: one 64-bit hash is split into a block index
//!   (high 32 bits, masked to `n_blocks`) and K position seeds
//!   derived from a multiplicative remix
//!   (`h * 0x9E37_79B9_7F4A_7C15`). The avalanche decorrelates
//!   the position seeds from the block index, which is what
//!   keeps the in-block bit pattern from clustering.
//!
//! # Sizing (default)
//!
//! - 64 KiB total → 1024 blocks of 64 B each (`n_blocks` is always
//!   a power of two so the modulo is a bit-AND).
//! - K = 4 hash functions per block.
//! - At ~100 K distinct terms (a typical Zipfian 1 M-doc superfile),
//!   that's ~100 items / block, ~7% FPR — meaningful skip without
//!   blowing manifest RAM.
//!
//! # Layout
//!
//! Bytes are stored as `Arc<[u64]>` of length `n_blocks * 8`
//! (each block is 8 × `u64` = 64 B). Storing as `u64` rather than
//! `u8` lets the inner loop test bits with one shift-and-mask per
//! probe, no byte-aligned addressing math. The Arc wrapper is so a
//! `FtsSummaryAgg` clone (and through it, a manifest clone) doesn't
//! duplicate the bloom payload.

use std::sync::Arc;

use wide::u64x4;
use xxhash_rust::xxh3::xxh3_64;

/// Block size in bytes — one cache line on x86_64 / aarch64.
pub const BLOCK_BYTES: usize = 64;
/// Block size in bits.
pub const BLOCK_BITS: usize = BLOCK_BYTES * 8; // 512
/// `u64` words per block.
const BLOCK_WORDS: usize = BLOCK_BYTES / 8; // 8
/// Hash functions per block (Kirsch-Mitzenmacher derived from one
/// 64-bit XXH3 result).
pub const K: usize = 4;

/// Default bloom size: 64 KiB / 1024 blocks. Sized so a typical
/// Zipfian 1 M-doc superfile with ~100 K distinct terms hits ~7%
/// FPR.
pub const DEFAULT_N_BLOCKS: usize = 1024;
/// Default bloom byte size (64 KiB).
pub const DEFAULT_BLOOM_BYTES: usize = DEFAULT_N_BLOCKS * BLOCK_BYTES;

/// 64-bit golden-ratio constant (⌊2^64 / φ⌋, odd). Used as a
/// multiplicative-avalanche mixer to decorrelate the in-block bit
/// positions from the block index, both derived from the same hash.
const GOLDEN_RATIO_U64: u64 = 0x9E37_79B9_7F4A_7C15;

/// Term-presence bloom filter for a single FTS column in a single
/// superfile.
///
/// Cheap to clone (`Arc::clone` on the underlying word buffer); a
/// `FtsSummaryAgg` clone shares this Arc with all manifest copies in
/// the supertable's snapshot history.
#[derive(Clone, Debug)]
pub struct Bloom {
    words: Arc<[u64]>,
    n_blocks_mask: u32,
}

impl Bloom {
    /// Reconstruct from a previously-emitted byte buffer. The
    /// buffer length must be exactly `n_blocks * BLOCK_BYTES` for
    /// some power-of-two `n_blocks`; passing anything else returns
    /// `None`.
    ///
    /// Useful for serialization round-trips when the manifest is
    /// loaded from a persistent backend.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if !bytes.len().is_multiple_of(BLOCK_BYTES) {
            return None;
        }
        let n_blocks = bytes.len() / BLOCK_BYTES;
        if n_blocks == 0 || !n_blocks.is_power_of_two() {
            return None;
        }
        let mut words = vec![0u64; n_blocks * BLOCK_WORDS];
        for (i, w) in words.iter_mut().enumerate() {
            let off = i * 8;
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[off..off + 8]);
            *w = u64::from_le_bytes(buf);
        }
        Some(Self {
            words: words.into(),
            n_blocks_mask: (n_blocks - 1) as u32,
        })
    }

    /// Serialize the bloom's bytes (little-endian `u64` words). The
    /// inverse of [`Bloom::from_bytes`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.words.len() * 8);
        for w in self.words.iter() {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Test whether `key` may have been inserted. `false` is
    /// definitive (the superfile doesn't contain `key`); `true` may
    /// be a false positive at the bloom's configured FPR.
    #[inline]
    pub fn contains(&self, key: &[u8]) -> bool {
        let h = xxh3_64(key);
        let (block_idx, mask) = block_and_mask(h, self.n_blocks_mask);
        let block_offset = block_idx * BLOCK_WORDS;
        let block: &[u64; BLOCK_WORDS] = (&self.words[block_offset..block_offset + BLOCK_WORDS])
            .try_into()
            .expect("BLOCK_WORDS-sized slice");
        contains_block(block, &mask)
    }

    /// Number of blocks. Always a power of two.
    pub fn n_blocks(&self) -> usize {
        (self.n_blocks_mask as usize) + 1
    }

    /// Total byte length.
    pub fn len(&self) -> usize {
        self.n_blocks() * BLOCK_BYTES
    }

    /// Whether the underlying word array is empty.
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }
}

/// Builder for a [`Bloom`]. Constructed with [`BloomBuilder::new`]
/// (default size) or [`BloomBuilder::with_n_blocks`] (custom);
/// fed with [`BloomBuilder::insert`]; finalized with
/// [`BloomBuilder::finish`].
///
/// One builder per (superfile, FTS column) at commit time.
pub struct BloomBuilder {
    words: Vec<u64>,
    n_blocks_mask: u32,
}

impl BloomBuilder {
    /// Builder sized at the default 64 KiB / 1024 blocks.
    pub fn new() -> Self {
        Self::with_n_blocks(DEFAULT_N_BLOCKS)
    }

    /// Builder with a caller-specified block count. Must be a
    /// power of two and ≥ 1; panics in debug otherwise (caller bug).
    pub fn with_n_blocks(n_blocks: usize) -> Self {
        debug_assert!(
            n_blocks.is_power_of_two() && n_blocks >= 1,
            "n_blocks must be a power of two ≥ 1; got {n_blocks}",
        );
        Self {
            words: vec![0u64; n_blocks * BLOCK_WORDS],
            n_blocks_mask: (n_blocks - 1) as u32,
        }
    }

    /// Insert `key` into the bloom. Idempotent (re-inserting the
    /// same key sets the same bits).
    pub fn insert(&mut self, key: &[u8]) {
        let h = xxh3_64(key);
        let (block_idx, mask) = block_and_mask(h, self.n_blocks_mask);
        let block_offset = block_idx * BLOCK_WORDS;
        // OR each mask word into the corresponding block word.
        // Insert is uncommon enough (per-superfile-build, not per-
        // query) that we don't bother SIMDing this path.
        let block = &mut self.words[block_offset..block_offset + BLOCK_WORDS];
        for (b, m) in block.iter_mut().zip(mask.iter()) {
            *b |= *m;
        }
    }

    /// Finalize into a shareable [`Bloom`].
    pub fn finish(self) -> Bloom {
        Bloom {
            words: self.words.into(),
            n_blocks_mask: self.n_blocks_mask,
        }
    }

    /// Number of blocks. Always a power of two.
    pub fn n_blocks(&self) -> usize {
        (self.n_blocks_mask as usize) + 1
    }
}

impl Default for BloomBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Block-level "all K positions present" check. `true` iff every bit
/// set in `mask` is also set in `block` (equivalently, `mask & !block == 0`
/// over the 512-bit lane).
///
/// Single-tier `wide::u64x4` AND-NOT-OR-reduce kernel. Lowers to two
/// 256-bit `vpandn` / `vpor` pairs on AVX2 x86_64 (the
/// `target-cpu=x86-64-v3` baseline), four 128-bit NEON ops on aarch64,
/// scalar elsewhere.
///
/// We deliberately do *not* dispatch to an AVX-512 kernel here even
/// on hosts that support it. The block is exactly 64 B = one ZMM
/// register, so AVX-512 only saves a single 256-bit iteration; at the
/// ~1 ns per-call regime this kernel runs in, that saving is
/// dominated by the per-instruction frequency-licensing cost on
/// Sapphire Rapids / Ice Lake (a 5% regression vs `wide` on the
/// `avx512_microbench_contains_block` benchmark before it was
/// removed). The AVX-512 win factor scales with bytes-per-call; this
/// kernel doesn't have enough of them to overcome the license cost
/// in isolation.
#[inline]
fn contains_block(block: &[u64; BLOCK_WORDS], mask: &[u64; BLOCK_WORDS]) -> bool {
    let block_lo = u64x4::new([block[0], block[1], block[2], block[3]]);
    let block_hi = u64x4::new([block[4], block[5], block[6], block[7]]);
    let mask_lo = u64x4::new([mask[0], mask[1], mask[2], mask[3]]);
    let mask_hi = u64x4::new([mask[4], mask[5], mask[6], mask[7]]);
    let r_lo = !block_lo & mask_lo;
    let r_hi = !block_hi & mask_hi;
    let combined = r_lo | r_hi;
    let parts = combined.to_array();
    (parts[0] | parts[1] | parts[2] | parts[3]) == 0
}

/// Derive (block_index, K-bit mask spread across BLOCK_WORDS u64s)
/// from a 64-bit hash.
///
/// Splitting strategy (avoids the K-position arithmetic-progression
/// correlation that's the standard failure mode of naive
/// Kirsch-Mitzenmacher):
///
/// - **`h1`** (the XXH3-64 result): high 32 bits drive the block
///   index. Low 32 bits aren't used here — we want the block
///   index decorrelated from the position seeds, since the K
///   positions form an arithmetic progression in their seeds and
///   any overlap between the index bits and the seed bits funnels
///   into clustered position patterns within a block.
/// - **`h2`** (golden-ratio-multiplicative remix of h1): provides
///   the position seeds. The multiplication
///   `h1 * 0x9E37_79B9_7F4A_7C15` is an avalanche function —
///   every output bit depends on every input bit — so h2's bits
///   are effectively independent of the high 32 bits of h1 that
///   drove block selection. h2's low 32 bits give h_low; h2's
///   high 32 bits give h_high; positions = h_low + i·h_high mod
///   BLOCK_BITS for i in 0..K.
///
/// Returns `(block_index, mask)` where `mask` has exactly K bits
/// set across its 8 u64s, matching the K positions.
#[inline]
fn block_and_mask(h: u64, n_blocks_mask: u32) -> (usize, [u64; BLOCK_WORDS]) {
    let block_idx = ((h >> 32) as u32 & n_blocks_mask) as usize;
    // Golden-ratio multiplicative avalanche to decorrelate
    // position seeds from the block index above.
    let h2 = h.wrapping_mul(GOLDEN_RATIO_U64);
    let h_low = h2 as u32;
    let h_high = (h2 >> 32) as u32;
    let block_bits_mask = (BLOCK_BITS as u32) - 1; // 511

    let mut mask = [0u64; BLOCK_WORDS];
    // Unrolled K=4 to give the compiler the cleanest trace into
    // a SIMD-friendly mask construction.
    let p0 = (h_low & block_bits_mask) as usize;
    let p1 = (h_low.wrapping_add(h_high) & block_bits_mask) as usize;
    let p2 = (h_low.wrapping_add(h_high.wrapping_mul(2)) & block_bits_mask) as usize;
    let p3 = (h_low.wrapping_add(h_high.wrapping_mul(3)) & block_bits_mask) as usize;
    mask[p0 / 64] |= 1u64 << (p0 % 64);
    mask[p1 / 64] |= 1u64 << (p1 % 64);
    mask[p2 / 64] |= 1u64 << (p2 % 64);
    mask[p3 / 64] |= 1u64 << (p3 % 64);

    (block_idx, mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a bloom from a small, deterministic set of keys.
    fn build_with(keys: &[&[u8]]) -> Bloom {
        let mut b = BloomBuilder::new();
        for k in keys {
            b.insert(k);
        }
        b.finish()
    }

    // ---- presence / absence correctness -----------------------------

    #[test]
    fn empty_bloom_contains_nothing() {
        let b = BloomBuilder::new().finish();
        assert!(!b.contains(b"alpha"));
        assert!(!b.contains(b"beta"));
        assert!(!b.contains(b""));
    }

    #[test]
    fn inserted_keys_are_definitely_present() {
        let keys: &[&[u8]] = &[
            b"alpha", b"beta", b"gamma", b"delta", b"epsilon", b"zeta", b"eta", b"theta",
        ];
        let b = build_with(keys);
        for k in keys {
            assert!(
                b.contains(k),
                "inserted key {:?} must be reported present",
                std::str::from_utf8(k).unwrap_or("<non-utf8>"),
            );
        }
    }

    #[test]
    fn insert_is_idempotent() {
        let mut b = BloomBuilder::new();
        b.insert(b"hello");
        let bytes_a = b.words.clone();
        b.insert(b"hello");
        let bytes_b = b.words.clone();
        assert_eq!(bytes_a, bytes_b);
    }

    // ---- false-positive rate ---------------------------------------

    /// At ~100K inserted keys / 64 KiB bloom, FPR should be ~7%.
    /// Pin a generous upper bound; a measured 0% or 50% would
    /// indicate a real bug.
    #[test]
    fn fpr_is_within_target_band_at_100k_keys() {
        let mut b = BloomBuilder::new();
        let n_inserted = 100_000usize;
        for i in 0..n_inserted {
            b.insert(format!("term{i}").as_bytes());
        }
        let bloom = b.finish();

        // Probe 100K keys NOT in the inserted set.
        let mut false_positives = 0usize;
        let n_probes = 100_000usize;
        for i in 0..n_probes {
            let probe = format!("absent_term_{i}");
            if bloom.contains(probe.as_bytes()) {
                false_positives += 1;
            }
        }
        let fpr = false_positives as f64 / n_probes as f64;
        // Theoretical: ~7%. Allow [3%, 12%] band.
        assert!(
            (0.03..=0.12).contains(&fpr),
            "FPR {} outside expected band [0.03, 0.12]",
            fpr,
        );
    }

    /// Inserted keys must still all return true at the same scale.
    /// Combined with the FPR test, this verifies "no false
    /// negatives" (a load-bearing bloom invariant).
    #[test]
    fn no_false_negatives_at_100k_keys() {
        let mut b = BloomBuilder::new();
        let n = 100_000usize;
        for i in 0..n {
            b.insert(format!("term{i}").as_bytes());
        }
        let bloom = b.finish();
        for i in 0..n {
            let key = format!("term{i}");
            assert!(
                bloom.contains(key.as_bytes()),
                "false negative on inserted key {key}",
            );
        }
    }

    // ---- sizing -----------------------------------------------------

    #[test]
    fn default_size_is_64kib_1024_blocks() {
        let b = BloomBuilder::new().finish();
        assert_eq!(b.len(), DEFAULT_BLOOM_BYTES);
        assert_eq!(b.n_blocks(), DEFAULT_N_BLOCKS);
    }

    #[test]
    fn custom_size_round_trips_through_builder() {
        for n_blocks in [1usize, 2, 4, 8, 256, 4096] {
            let b = BloomBuilder::with_n_blocks(n_blocks).finish();
            assert_eq!(b.n_blocks(), n_blocks);
            assert_eq!(b.len(), n_blocks * BLOCK_BYTES);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "n_blocks must be a power of two")]
    fn non_power_of_two_n_blocks_panics_in_debug() {
        let _ = BloomBuilder::with_n_blocks(3);
    }

    // ---- byte serialization ----------------------------------------

    #[test]
    fn to_bytes_from_bytes_round_trip_preserves_membership() {
        let keys: &[&[u8]] = &[b"alpha", b"beta", b"gamma", b"delta", b"epsilon"];
        let b1 = build_with(keys);
        let bytes = b1.to_bytes();

        let b2 = Bloom::from_bytes(&bytes).expect("valid bytes");
        assert_eq!(b2.n_blocks(), b1.n_blocks());

        for k in keys {
            assert!(b2.contains(k));
        }
        // Definite-absent keys (chosen to give scatter); a few might
        // collide as false positives, but not all of them.
        let probes: &[&[u8]] = &[b"never1", b"never2", b"never3", b"never4"];
        let n_collisions = probes.iter().filter(|p| b2.contains(p)).count();
        assert!(
            n_collisions < probes.len(),
            "all probes false-positive — bloom appears saturated",
        );
    }

    #[test]
    fn from_bytes_rejects_misaligned_buffer() {
        // Length not a multiple of BLOCK_BYTES.
        assert!(Bloom::from_bytes(&[0u8; BLOCK_BYTES + 1]).is_none());
        // Length is a multiple but block count isn't a power of two.
        assert!(Bloom::from_bytes(&[0u8; BLOCK_BYTES * 3]).is_none());
        // Empty.
        assert!(Bloom::from_bytes(&[]).is_none());
    }

    // ---- hash distribution -----------------------------------------

    /// Block index distribution should be reasonably uniform
    /// across the block array. With 100K random keys and 1024
    /// blocks the χ² test would be ~1023 d.f.; we just check
    /// that the max-block-load isn't pathologically skewed.
    #[test]
    fn block_load_is_roughly_uniform() {
        let n_blocks = DEFAULT_N_BLOCKS;
        let mut load = vec![0usize; n_blocks];
        for i in 0..100_000usize {
            let h = xxh3_64(format!("term{i}").as_bytes());
            let (block_idx, _) = block_and_mask(h, (n_blocks - 1) as u32);
            load[block_idx] += 1;
        }
        let max = *load.iter().max().expect("non-empty");
        let mean = load.iter().sum::<usize>() as f64 / n_blocks as f64;
        // 100K / 1024 ≈ 97.7 mean. Max should be within ~3× of
        // mean for a healthy hash. (A bad hash that funneled
        // everything into a few blocks would have max in the
        // thousands.)
        assert!(
            (max as f64) < 3.0 * mean,
            "max-block-load {max} more than 3× mean {mean}",
        );
    }

    /// Two different keys should very rarely produce the same
    /// (block_idx, mask). 1000 random keys; collisions should
    /// be a tiny fraction.
    #[test]
    fn different_keys_rarely_collide_on_same_signature() {
        let mut sigs: std::collections::HashMap<(usize, [u64; BLOCK_WORDS]), Vec<String>> =
            std::collections::HashMap::new();
        for i in 0..1000usize {
            let key = format!("a-{i}");
            let h = xxh3_64(key.as_bytes());
            let sig = block_and_mask(h, (DEFAULT_N_BLOCKS - 1) as u32);
            sigs.entry(sig).or_default().push(key);
        }
        let collisions = sigs.values().filter(|v| v.len() > 1).count();
        // For 1000 keys against 1024 blocks × C(512, 4) ≈ 2.8e9
        // signature space, the birthday-bound expected collision
        // count is essentially zero. Allow up to a few for safety.
        assert!(
            collisions < 5,
            "{collisions} signature collisions in 1000 keys"
        );
    }

    // ---- Arc sharing -----------------------------------------------

    #[test]
    fn clone_shares_words_via_arc() {
        let b1 = build_with(&[b"alpha", b"beta"]);
        let b2 = b1.clone();
        assert_eq!(b1.words.as_ptr(), b2.words.as_ptr());
    }

    #[test]
    fn debug_format_doesnt_explode() {
        let b = build_with(&[b"x"]);
        let s = format!("{:?}", b);
        assert!(s.contains("Bloom"));
    }

    /// Builder geometry (`n_blocks` on the *builder*, plus `Default`)
    /// and the finished bloom's `len` / `is_empty` agree across the
    /// build → finish boundary.
    #[test]
    fn builder_and_bloom_geometry_is_consistent() {
        const CUSTOM_BLOCKS: usize = 4;
        let mut b = BloomBuilder::with_n_blocks(CUSTOM_BLOCKS);
        assert_eq!(
            b.n_blocks(),
            CUSTOM_BLOCKS,
            "builder reports its block count"
        );
        b.insert(b"alpha");
        let bloom = b.finish();
        assert_eq!(bloom.n_blocks(), CUSTOM_BLOCKS);
        assert_eq!(bloom.len(), CUSTOM_BLOCKS * BLOCK_BYTES);
        assert!(!bloom.is_empty(), "a sized bloom's word array is non-empty");

        // `Default` builds the standard geometry, same as `new()`.
        assert_eq!(
            BloomBuilder::default().n_blocks(),
            BloomBuilder::new().n_blocks()
        );
    }
}
