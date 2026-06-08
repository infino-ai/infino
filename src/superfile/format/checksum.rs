// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! CRC32C wrapper.
//!
//! Every section of every blob ends with a 4-byte CRC32C of its body —
//! the format-spec invariant. This module is a thin shim over the
//! `crc-fast` crate (CLMUL-based, folds multiple streams in vector
//! registers on x86_64 and aarch64) so the rest of the codebase has
//! a single, named entry point and so we can swap implementations
//! without touching call sites.
//!
//! Earlier revisions used the `crc32c` crate, which dispatches to
//! the scalar SSE4.2 `_mm_crc32_u64` instruction. That is fast, but
//! bottlenecked by one dependency chain. The CLMUL path breaks that
//! chain by folding over parallel streams, which matters on the
//! 1.5 GiB vector cold-open path.
//!
//! Use the Castagnoli polynomial (0x1EDC6F41) — the same one used
//! by iSCSI, Btrfs, Parquet bloom filters, and most modern systems.
//! `crc-fast` exposes this as `CrcAlgorithm::Crc32Iscsi` /
//! `crc32_iscsi(...)`.

use crc_fast::CrcAlgorithm;

/// Compute CRC32C over `bytes`. Hardware-accelerated when the crate's
/// runtime feature detection finds SSE4.2 (x86) or v8.1 CRC (ARM); falls
/// back to a software implementation otherwise.
#[inline]
pub fn crc32c(bytes: &[u8]) -> u32 {
    crc_fast::crc32_iscsi(bytes) as u32
}

/// Streaming variant — extend a prior `prev` CRC by `bytes`. Implemented
/// via `checksum_combine` polynomial math (`crc-fast` doesn't expose a
/// seed-from-prev API on its `Digest`). Used by streaming builders and
/// kept here so callers don't reach for the upstream crate directly.
#[inline]
pub fn crc32c_append(prev: u32, bytes: &[u8]) -> u32 {
    let suffix = crc_fast::crc32_iscsi(bytes) as u64;
    crc_fast::checksum_combine(
        CrcAlgorithm::Crc32Iscsi,
        prev as u64,
        suffix,
        bytes.len() as u64,
    ) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    // Castagnoli reference vectors. These match RFC 3720 / iSCSI test data
    // and the published `crc32c` crate test fixtures.

    #[test]
    fn empty_input_is_zero() {
        // crc32c() of empty input is 0 by definition.
        assert_eq!(crc32c(b""), 0);
    }

    #[test]
    fn single_byte_known_vector() {
        // crc32c("a") = 0xC1D04330 (standard Castagnoli reference).
        assert_eq!(crc32c(b"a"), 0xC1D0_4330);
    }

    #[test]
    fn ascii_string_known_vector() {
        // crc32c("123456789") = 0xE3069283 — the classic CRC reference vector.
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn append_matches_one_shot() {
        // Streaming over arbitrary chunks must agree with a single-shot call.
        let bytes: &[u8] = b"the quick brown fox jumps over the lazy dog";
        let one_shot = crc32c(bytes);

        let mut acc = 0u32;
        for chunk in bytes.chunks(7) {
            acc = crc32c_append(acc, chunk);
        }
        assert_eq!(acc, one_shot);

        // Single-byte chunks too — most pathological case.
        let mut acc2 = 0u32;
        for chunk in bytes.chunks(1) {
            acc2 = crc32c_append(acc2, chunk);
        }
        assert_eq!(acc2, one_shot);
    }

    #[test]
    fn append_with_zero_seed_equals_one_shot() {
        // crc32c_append(0, x) === crc32c(x). Documents the "fresh stream
        // starts at 0" invariant for callers.
        assert_eq!(crc32c_append(0, b"hello"), crc32c(b"hello"));
    }

    #[test]
    fn alignment_does_not_affect_result() {
        // Hardware CRC instructions can be alignment-sensitive on some
        // microarchitectures; the crate's wrapper handles that. Verify
        // shifted slices give identical results.
        let buf: Vec<u8> = (0..1024u16).map(|i| i as u8).collect();
        let baseline = crc32c(&buf);
        for shift in 1..16 {
            let shifted: Vec<u8> = std::iter::repeat_n(0u8, shift)
                .chain(buf.iter().copied())
                .collect();
            assert_eq!(crc32c(&shifted[shift..]), baseline);
        }
    }

    #[test]
    fn different_inputs_give_different_outputs() {
        // Sanity: changing one bit changes the CRC. This is the whole
        // point of using a CRC vs nothing.
        let a = crc32c(b"AAAA");
        let b = crc32c(b"AAAB");
        assert_ne!(a, b);
    }

    #[test]
    fn large_input_does_not_panic_or_truncate() {
        // 4 MB input — exercise the streaming path inside the crate.
        let big: Vec<u8> = (0..4_000_000).map(|i| (i % 256) as u8).collect();
        let _ = crc32c(&big);
    }
}
