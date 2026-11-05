// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The builder's term interner table: term bytes → dense term id.
//!
//! An open-addressing table with linear probing whose slot carries the
//! first 16 bytes of its term inline, zero-padded, next to the term's
//! length and id. Nearly every term a tokenizer emits is 16 bytes or
//! shorter, so a lookup reads one slot — usually one cache line — and
//! decides the match from the slot alone: no control-byte array to load
//! first, and no trip to the term bytes stored elsewhere. Longer terms
//! match on length and the inline prefix first and only then compare
//! their full bytes, which the caller keeps (see [`TermTable::intern`]).
//!
//! The table holds ids only; the caller owns the id → term list, so a
//! term's bytes are stored once.

use std::mem::replace;

/// One slot. `tag == 0` marks it empty; otherwise `tag` is the term's
/// byte length plus one (saturating), `words` its first 16 bytes
/// zero-padded little-endian, and `id` its term id.
#[derive(Clone, Copy, Default)]
struct Slot {
    words: [u64; 2],
    id: u32,
    tag: u32,
}

/// Terms up to this many bytes are held whole in their slot.
const INLINE_BYTES: usize = 16;

pub(crate) struct TermTable {
    /// Power-of-two length (or empty before the first insert).
    slots: Vec<Slot>,
    /// Occupied slots.
    len: usize,
    /// Grow before an insert once `len` reaches this (3/4 of capacity).
    grow_at: usize,
}

impl TermTable {
    pub(crate) fn new() -> Self {
        Self {
            slots: Vec::new(),
            len: 0,
            grow_at: 0,
        }
    }

    /// The id of `term`, or — when it is not in the table — insert it
    /// under the next id, `keys.len()`, and return that id with `true`.
    ///
    /// `keys` is the caller's id → term list: entry `id` must be the term
    /// interned under `id`. A term longer than [`INLINE_BYTES`] is
    /// compared against `keys` to confirm a match, and the table rehashes
    /// such terms from `keys` when it grows. On a `true` return the caller
    /// must push `term` onto `keys` before the next call.
    #[inline(always)]
    pub(crate) fn intern<K: AsRef<[u8]>>(&mut self, term: &[u8], keys: &[K]) -> (u32, bool) {
        if self.len >= self.grow_at {
            self.grow(keys);
        }
        let n = term.len();
        let words = load_words(&term[..n.min(INLINE_BYTES)]);
        let hash = hash_term(term, words);
        let tag = tag_of(n);
        let mask = self.slots.len() - 1;
        let mut i = hash as usize & mask;
        loop {
            let slot = &mut self.slots[i];
            if slot.tag == 0 {
                let id = keys.len() as u32;
                *slot = Slot { words, id, tag };
                self.len += 1;
                return (id, true);
            }
            if slot.tag == tag
                && slot.words == words
                && (n <= INLINE_BYTES || keys[slot.id as usize].as_ref() == term)
            {
                return (slot.id, false);
            }
            i = (i + 1) & mask;
        }
    }

    /// Double the capacity (16 slots at first) and reinsert every term.
    #[cold]
    #[inline(never)]
    fn grow<K: AsRef<[u8]>>(&mut self, keys: &[K]) {
        let cap = (self.slots.len() * 2).max(16);
        let old = replace(&mut self.slots, vec![Slot::default(); cap]);
        let mask = cap - 1;
        for slot in old.into_iter().filter(|s| s.tag != 0) {
            let hash = if slot.tag as usize - 1 <= INLINE_BYTES {
                hash_words(slot.words, slot.tag as usize - 1)
            } else {
                let term = keys[slot.id as usize].as_ref();
                hash_term(term, slot.words)
            };
            let mut i = hash as usize & mask;
            while self.slots[i].tag != 0 {
                i = (i + 1) & mask;
            }
            self.slots[i] = slot;
        }
        self.grow_at = cap / 4 * 3;
    }
}

#[inline(always)]
fn tag_of(n: usize) -> u32 {
    (n as u64 + 1).min(u32::MAX as u64) as u32
}

const K0: u64 = 0x243f_6a88_85a3_08d3;
const K1: u64 = 0x1319_8a2e_0370_7344;
const K2: u64 = 0xa409_3822_299f_31d0;
const K3: u64 = 0x082e_fa98_ec4e_6c89;

/// Multiply as 128 bits and fold the halves together.
#[inline(always)]
fn fold(a: u64, b: u64) -> u64 {
    let p = (a as u128).wrapping_mul(b as u128);
    (p as u64) ^ ((p >> 64) as u64)
}

/// Hash of a term of `n` bytes whose first 16, zero-padded, are `words`.
#[inline(always)]
fn hash_words(words: [u64; 2], n: usize) -> u64 {
    fold(words[0] ^ K0 ^ n as u64, words[1] ^ K1)
}

/// Hash of `term`, given its first 16 bytes' `words`: the inline hash,
/// with every further 16-byte chunk folded in.
#[inline(always)]
fn hash_term(term: &[u8], words: [u64; 2]) -> u64 {
    let mut h = hash_words(words, term.len());
    if term.len() > INLINE_BYTES {
        for chunk in term[INLINE_BYTES..].chunks(INLINE_BYTES) {
            let [a, b] = load_words(chunk);
            h = fold(a ^ h ^ K2, b ^ K3);
        }
    }
    h
}

/// Up to 16 bytes as two little-endian words, zero-padded. A few
/// overlapping loads instead of a variable-length copy.
#[inline(always)]
fn load_words(b: &[u8]) -> [u64; 2] {
    let n = b.len();
    debug_assert!(n <= INLINE_BYTES);
    if n >= 8 {
        let w0 = le_u64(&b[..8]);
        let w1 = if n > 8 {
            // Bytes `n - 8..n`; the top `n - 8` of them are bytes `8..n`.
            le_u64(&b[n - 8..n]) >> ((16 - n) * 8)
        } else {
            0
        };
        [w0, w1]
    } else if n >= 4 {
        let lo = le_u32(&b[..4]) as u64;
        // Bytes `n - 4..n`; the top `n - 4` of them are bytes `4..n`.
        let hi = (le_u32(&b[n - 4..n]) as u64) >> ((8 - n) * 8);
        [lo | (hi << 32), 0]
    } else if n > 0 {
        // n = 1, 2 or 3: bytes 0, n/2 and n-1 cover all of them.
        let w = b[0] as u64
            | ((b[n / 2] as u64) << ((n / 2) * 8))
            | ((b[n - 1] as u64) << ((n - 1) * 8));
        [w, 0]
    } else {
        [0, 0]
    }
}

/// `b`, exactly 8 bytes, as a little-endian word.
#[inline(always)]
fn le_u64(b: &[u8]) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(b);
    u64::from_le_bytes(w)
}

/// `b`, exactly 4 bytes, as a little-endian word.
#[inline(always)]
fn le_u32(b: &[u8]) -> u32 {
    let mut w = [0u8; 4];
    w.copy_from_slice(b);
    u32::from_le_bytes(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn padded(b: &[u8]) -> [u64; 2] {
        let mut buf = [0u8; 16];
        buf[..b.len()].copy_from_slice(b);
        [le_u64(&buf[..8]), le_u64(&buf[8..])]
    }

    #[test]
    fn load_words_is_the_zero_padded_prefix_at_every_length() {
        let src: Vec<u8> = (1..=16u8).map(|b| b.wrapping_mul(37) | 0x80).collect();
        for n in 0..=16 {
            assert_eq!(load_words(&src[..n]), padded(&src[..n]), "n = {n}");
        }
    }

    #[test]
    fn interns_every_length_and_finds_each_again_after_growth() {
        let mut keys: Vec<Vec<u8>> = Vec::new();
        let mut table = TermTable::new();
        let mut want: HashMap<Vec<u8>, u32> = HashMap::new();
        // Terms of every length 0..=40, sharing prefixes so long terms
        // collide on their inline words and must be told apart by their
        // full bytes, plus the same prefixes padded with NUL bytes so a
        // shorter term never matches a zero-padded longer one.
        let mut terms: Vec<Vec<u8>> = Vec::new();
        for i in 0..2_000u32 {
            let len = (i % 41) as usize;
            let mut t: Vec<u8> = b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH"[..len].to_vec();
            if len > 0 {
                let last = t.len() - 1;
                t[last] = (i / 41) as u8;
            }
            terms.push(t.clone());
            t.push(0);
            terms.push(t);
        }
        for round in 0..2 {
            for t in &terms {
                let (id, new) = table.intern(t, &keys);
                match want.get(t) {
                    Some(&prev) => {
                        assert!(!new, "{t:?} re-minted");
                        assert_eq!(id, prev);
                    }
                    None => {
                        assert_eq!(round, 0);
                        assert!(new);
                        assert_eq!(id as usize, keys.len());
                        keys.push(t.clone());
                        want.insert(t.clone(), id);
                    }
                }
            }
        }
        assert_eq!(table.len, want.len());
    }
}
