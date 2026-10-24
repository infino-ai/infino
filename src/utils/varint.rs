// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! LEB128 variable-length integers, little-endian base-128 with the high
//! bit of each byte as the continuation flag. One implementation for the
//! position runs, the short-form bodies, the patched-block exceptions
//! and the term dictionary.

/// LEB128 continuation flag: high bit set ⇒ another byte follows.
pub(crate) const CONTINUATION_BIT: u8 = 0x80;
/// Payload bits per LEB128 byte.
const PAYLOAD_BITS: u32 = 7;
/// Payload mask per LEB128 byte.
const PAYLOAD_MASK: u8 = 0x7f;

/// Append one `u64` as LEB128 to `out`.
#[inline]
pub(crate) fn push_u64_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v as u8) & PAYLOAD_MASK;
        v >>= PAYLOAD_BITS;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | CONTINUATION_BIT);
    }
}

/// Append one `u32` as LEB128 to `out`.
#[inline]
pub(crate) fn push_varint(out: &mut Vec<u8>, v: u32) {
    push_u64_varint(out, u64::from(v));
}

/// Decode one LEB128 `u64` from `bytes` starting at `*at`, advancing
/// `*at` past it. `None` on truncated input or a value that overflows
/// `u64` — both only reachable on corrupt bytes, which the caller
/// surfaces as a read error.
#[inline]
pub(crate) fn read_u64_varint(bytes: &[u8], at: &mut usize) -> Option<u64> {
    let mut v: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        let &b = bytes.get(*at)?;
        *at += 1;
        if shift >= u64::BITS {
            return None;
        }
        let payload = u64::from(b & PAYLOAD_MASK);
        if shift > 0 && payload >> (u64::BITS - shift) != 0 {
            // Payload bits past the 64-bit boundary ⇒ overflow.
            return None;
        }
        v |= payload << shift;
        if b & CONTINUATION_BIT == 0 {
            return Some(v);
        }
        shift += PAYLOAD_BITS;
    }
}

/// Decode one LEB128 `u32`, as [`read_u64_varint`] but refusing a value
/// past `u32::MAX`.
#[inline]
pub(crate) fn read_varint(bytes: &[u8], at: &mut usize) -> Option<u32> {
    u32::try_from(read_u64_varint(bytes, at)?).ok()
}

/// Encoded LEB128 length of `v`.
#[inline]
pub(crate) fn varint_len(v: u32) -> usize {
    match v {
        0..=0x7f => 1,
        0x80..=0x3fff => 2,
        0x4000..=0x1f_ffff => 3,
        0x20_0000..=0xfff_ffff => 4,
        _ => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Largest byte length one encoded `u32` can occupy (5 × 7 bits).
    const MAX_U32_VARINT_BYTES: usize = 5;

    #[test]
    fn varint_round_trips_boundaries() {
        for v in [0u32, 1, 127, 128, 16383, 16384, u32::MAX - 1, u32::MAX] {
            let mut buf = Vec::new();
            push_varint(&mut buf, v);
            assert!(buf.len() <= MAX_U32_VARINT_BYTES);
            assert_eq!(buf.len(), varint_len(v));
            let mut at = 0;
            assert_eq!(read_varint(&buf, &mut at), Some(v));
            assert_eq!(at, buf.len());
        }
        for v in [0u64, 1 << 32, u64::MAX >> 1, u64::MAX] {
            let mut buf = Vec::new();
            push_u64_varint(&mut buf, v);
            let mut at = 0;
            assert_eq!(read_u64_varint(&buf, &mut at), Some(v));
            assert_eq!(at, buf.len());
        }
    }

    #[test]
    fn read_varint_rejects_truncation() {
        let mut buf = Vec::new();
        push_varint(&mut buf, 300);
        let mut at = 0;
        assert_eq!(read_varint(&buf[..1], &mut at), None);
    }

    #[test]
    fn read_varint_rejects_overflow() {
        // A value past u32::MAX is refused by the u32 reader, accepted by
        // the u64 one; eleven continuation bytes overflow even u64.
        let mut buf = Vec::new();
        push_u64_varint(&mut buf, 1 << 32);
        assert_eq!(read_varint(&buf, &mut 0), None);
        assert_eq!(read_u64_varint(&buf, &mut 0), Some(1 << 32));
        let buf = [0xffu8; 11];
        assert_eq!(read_u64_varint(&buf, &mut 0), None);
    }
}
