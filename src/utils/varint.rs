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

/// The fewest bytes [`push_doc_facts`] writes: a byte of row, a byte of
/// term frequency and the length bucket.
const MIN_DOC_FACTS_BYTES: usize = 3;

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

/// Append a document's facts — row, term frequency, length bucket — as a
/// term's head carries them.
pub(crate) fn push_doc_facts(out: &mut Vec<u8>, row: u32, tf: u32, len: u8) {
    push_varint(out, row);
    push_varint(out, tf);
    out.push(len);
}

/// Read what [`push_doc_facts`] wrote at `at`, advancing it; `None` short.
pub(crate) fn read_doc_facts(bytes: &[u8], at: &mut usize) -> Option<(u32, u32, u8)> {
    let row = read_varint(bytes, at)?;
    let tf = read_varint(bytes, at)?;
    let len = *bytes.get(*at)?;
    *at += 1;
    Some((row, tf, len))
}

/// Append a term's head: its count, then each document's facts in order.
pub(crate) fn push_head(out: &mut Vec<u8>, head: impl ExactSizeIterator<Item = (u32, u32, u8)>) {
    push_varint(out, head.len() as u32);
    for (row, tf, len) in head {
        push_doc_facts(out, row, tf, len);
    }
}

/// Read what [`push_head`] wrote at `at`, advancing it; `None` short.
pub(crate) fn read_head(bytes: &[u8], at: &mut usize) -> Option<Vec<(u32, u32, u8)>> {
    let n = read_varint(bytes, at)? as usize;
    // A corrupt count cannot ask for more room than the bytes left could
    // fill.
    let fits = bytes.len().saturating_sub(*at) / MIN_DOC_FACTS_BYTES;
    let mut head = Vec::with_capacity(n.min(fits));
    for _ in 0..n {
        head.push(read_doc_facts(bytes, at)?);
    }
    Some(head)
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
/// past `u32::MAX`. A one-byte value, the common case, is read inline;
/// longer ones take the general decoder.
#[inline(always)]
pub(crate) fn read_varint(bytes: &[u8], at: &mut usize) -> Option<u32> {
    if let Some(&b) = bytes.get(*at)
        && b & CONTINUATION_BIT == 0
    {
        *at += 1;
        return Some(u32::from(b));
    }
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

    /// A head is its count, then each document's row and term frequency as
    /// LEB128 and its length bucket as one byte — the layout a contribution
    /// record has always carried — and reads back whole; cut anywhere short
    /// of its end, it reads as nothing.
    #[test]
    fn a_head_reads_back_as_written_and_short_reads_as_nothing() {
        let head = [
            (0u32, 1u32, 0u8),
            (127, 128, 7),
            (16_384, u32::MAX, u8::MAX),
        ];
        let mut buf = Vec::new();
        push_head(&mut buf, head.iter().copied());
        let mut expected = vec![head.len() as u8];
        for (row, tf, len) in head {
            push_varint(&mut expected, row);
            push_varint(&mut expected, tf);
            expected.push(len);
        }
        assert_eq!(buf, expected, "the layout");
        let mut at = 0;
        assert_eq!(read_head(&buf, &mut at).as_deref(), Some(&head[..]));
        assert_eq!(at, buf.len());
        for cut in 0..buf.len() {
            assert_eq!(read_head(&buf[..cut], &mut 0), None, "cut at {cut}");
        }

        let mut empty = Vec::new();
        push_head(&mut empty, [].into_iter());
        assert_eq!(empty, [0], "an empty head is its count alone");
        assert_eq!(read_head(&empty, &mut 0), Some(Vec::new()));
    }

    /// A head whose count no remaining bytes could fill reads as nothing,
    /// without first reserving room for that many documents: a corrupt
    /// count of `u32::MAX` would otherwise ask for ~50 GB before the first
    /// short read.
    #[test]
    fn a_head_count_past_its_bytes_reads_as_nothing_without_reserving_it() {
        let mut corrupt = Vec::new();
        push_varint(&mut corrupt, u32::MAX);
        push_doc_facts(&mut corrupt, 1, 1, 0);
        assert_eq!(read_head(&corrupt, &mut 0), None);
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
