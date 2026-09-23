// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Little-endian fixed-width reads over a byte slice, bounds-checked.
//! Shared by the superfile format parser and the term dictionary.

/// Little-endian `u32` at `at`, `None` past the end of `bytes`.
#[inline]
pub(crate) fn u32_le_at(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().expect("4 bytes")))
}

/// Little-endian `u64` at `at`, `None` past the end of `bytes`.
#[inline]
pub(crate) fn u64_le_at(bytes: &[u8], at: usize) -> Option<u64> {
    bytes
        .get(at..at + 8)
        .map(|s| u64::from_le_bytes(s.try_into().expect("8 bytes")))
}
