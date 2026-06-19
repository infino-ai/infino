// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Hand-rolled binary format for one superfile's tombstone
//! sidecar object.
//!
//! Lives in `wal/` because the format is private to the
//! tombstone subsystem — readers and writers both go through
//! `encode_sidecar` / `decode_sidecar`. No serde, no third-party
//! binary format crate; the structure is two fixed-size scalars
//! plus an optional 24-byte `SealRecord` plus a `RoaringBitmap`
//! that self-serializes.
//!
//! ## Layout
//!
//! ```text
//! offset  size  field
//! ----------------------------------------------------------------
//! 0       8     magic: "INFTOMB\0"                       (literal)
//! 8       4     schema_version: u32 LE                   (= 1)
//! 12      1     seal_flag: u8                            (0|1)
//! 13      24    seal_payload, present iff seal_flag == 1:
//!                  [0..16]  compaction_id: Uuid (BE bytes)
//!                  [16..24] sealed_at_unix_millis: i64 LE
//! 13|37   4     bitmap_len: u32 LE
//! +4      var   bitmap_bytes: RoaringBitmap::serialize_into output
//! ```
//!
//! `i64 LE` (rather than RFC3339 string) for `sealed_at`: the
//! sidecar is a binary file already, and 8 bytes of unix millis
//! sort the same as the human-readable timestamp.
//!
//! ## Bitmap framing rationale
//!
//! `roaring::RoaringBitmap::serialize_into(W)` and
//! `RoaringBitmap::deserialize_from(R)` produce a portable
//! representation, but the on-disk size isn't visible to the
//! caller without invoking `serialized_size()`. We prefix with
//! an explicit `bitmap_len: u32 LE` to make the layout
//! self-describing — a reader can length-validate the bitmap
//! region against the file size before invoking the bitmap
//! deserialize, which lets us surface "trailing-garbage" or
//! "truncated" errors without depending on internal bitmap-
//! deserialize error variants.

use std::io::{Cursor, Read};

use chrono::{TimeZone, Utc};
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::state_doc::SealRecord;

/// 8-byte magic. Lets the codec reject foreign objects
/// (e.g. an old serde-encoded sidecar, a stray JSON file,
/// or a sidecar from a corrupt write) before parsing further.
const MAGIC: &[u8; 8] = b"INFTOMB\0";

/// Bumped when the on-disk layout changes. Older readers reject
/// anything they don't recognize so a format drift surfaces as
/// a typed error at decode time rather than producing garbage.
pub const SCHEMA_VERSION: u32 = 1;

/// On-disk byte widths of the fixed header fields. The header is
/// `MAGIC | version | seal_flag [| uuid | millis]`, followed by the
/// bitmap length prefix and the bitmap bytes.
const MAGIC_LEN: usize = 8;
/// `u32` little-endian schema version width.
const VERSION_LEN: usize = 4;
/// Single-byte seal-flag width (one of [`SEAL_FLAG_UNSEALED`] /
/// [`SEAL_FLAG_SEALED`]).
const SEAL_FLAG_LEN: usize = 1;
/// Compaction-id width in the seal payload (a 16-byte UUID).
const SEAL_UUID_LEN: usize = 16;
/// Sealed-at timestamp width in the seal payload (`i64` LE unix
/// millis).
const SEAL_TIMESTAMP_LEN: usize = 8;
/// `u32` little-endian bitmap-length prefix width, written before
/// the serialized RoaringBitmap so the region is self-describing.
const BITMAP_LEN_PREFIX_LEN: usize = 4;

/// Seal-flag byte meaning "no seal payload follows".
const SEAL_FLAG_UNSEALED: u8 = 0;
/// Seal-flag byte meaning "a 24-byte seal payload (uuid + millis)
/// follows".
const SEAL_FLAG_SEALED: u8 = 1;

/// Fixed header + flag length when the sidecar is unsealed. The
/// bitmap length prefix and bitmap bytes follow directly.
const HEADER_LEN_UNSEALED: usize = MAGIC_LEN + VERSION_LEN + SEAL_FLAG_LEN;

/// Header + flag + seal_payload length when sealed.
const HEADER_LEN_SEALED: usize = HEADER_LEN_UNSEALED + SEAL_UUID_LEN + SEAL_TIMESTAMP_LEN;

/// On-disk representation of one superfile's tombstones —
/// already-parsed shape. `decode_sidecar` produces this from
/// raw bytes; `encode_sidecar` consumes it.
#[derive(Debug, Clone)]
pub struct TombstonesSidecar {
    /// Compaction write-barrier. `Some` once a compactor has
    /// frozen this sidecar's bitmap. Tombstone writers that
    /// observe a sealed sidecar must NOT write to it; they
    /// re-resolve the target against a fresh manifest (which
    /// will either point at the merged target superfile, or
    /// surface that the compaction hasn't finished publishing
    /// yet, in which case the writer waits). Monotonic: once
    /// `Some`, never reverts to `None` — the sidecar is
    /// eventually deleted post-compaction, never un-sealed in
    /// place.
    pub seal: Option<SealRecord>,

    /// Tombstoned local doc_ids within this superfile.
    pub bitmap: RoaringBitmap,
}

/// Typed errors from the codec. Carry enough context to
/// localize a malformed sidecar (offset, observed value, etc.)
/// since the format is wire-stable and any failure here is a
/// corruption / version-skew event worth diagnosing.
#[derive(Debug, thiserror::Error)]
pub enum SidecarCodecError {
    #[error("sidecar truncated: need {needed} bytes, have {have}")]
    Truncated { needed: usize, have: usize },

    #[error("bad magic — expected {expected:?}, got {got:?}")]
    BadMagic { expected: [u8; 8], got: [u8; 8] },

    #[error("unsupported schema version {got}; this build supports up to {max}")]
    UnsupportedVersion { got: u32, max: u32 },

    #[error("invalid seal_flag byte {got}; must be 0 or 1")]
    InvalidSealFlag { got: u8 },

    #[error("invalid sealed_at_unix_millis {millis}: cannot represent as a UTC chrono::DateTime")]
    InvalidSealTimestamp { millis: i64 },

    #[error("bitmap length {declared} exceeds remaining bytes {remaining}")]
    BitmapTooLong { declared: u32, remaining: usize },

    #[error("trailing garbage after bitmap: {trailing} unexpected bytes")]
    TrailingBytes { trailing: usize },

    #[error("RoaringBitmap deserialize failed: {0}")]
    BitmapDecode(#[source] std::io::Error),

    #[error("RoaringBitmap serialize failed: {0}")]
    BitmapEncode(#[source] std::io::Error),
}

/// Encode the sidecar to its on-disk byte layout. Allocates a
/// single `Vec<u8>` sized from the bitmap's `serialized_size()`
/// so there's exactly one allocation per encode.
pub fn encode_sidecar(sidecar: &TombstonesSidecar) -> Result<Vec<u8>, SidecarCodecError> {
    let bitmap_size = sidecar.bitmap.serialized_size();
    let header_len = if sidecar.seal.is_some() {
        HEADER_LEN_SEALED
    } else {
        HEADER_LEN_UNSEALED
    };
    let total = header_len + BITMAP_LEN_PREFIX_LEN + bitmap_size;
    let mut out: Vec<u8> = Vec::with_capacity(total);

    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());

    match &sidecar.seal {
        None => {
            out.push(SEAL_FLAG_UNSEALED);
        }
        Some(seal) => {
            out.push(SEAL_FLAG_SEALED);
            out.extend_from_slice(seal.compaction_id.as_bytes());
            let millis = seal.sealed_at.timestamp_millis();
            out.extend_from_slice(&millis.to_le_bytes());
        }
    }

    // Reserve four bytes for the bitmap length prefix; fill in
    // after we know the actual encoded size. We pre-computed it
    // via `serialized_size()` but verifying-and-writing keeps
    // the format self-consistent if `roaring` ever changes its
    // sizing function.
    let len_prefix_pos = out.len();
    out.extend_from_slice(&[0u8; BITMAP_LEN_PREFIX_LEN]);

    let pre_bitmap_len = out.len();
    sidecar
        .bitmap
        .serialize_into(&mut out)
        .map_err(SidecarCodecError::BitmapEncode)?;
    let bitmap_actual = (out.len() - pre_bitmap_len) as u32;
    out[len_prefix_pos..len_prefix_pos + BITMAP_LEN_PREFIX_LEN]
        .copy_from_slice(&bitmap_actual.to_le_bytes());

    Ok(out)
}

/// Inverse of `encode_sidecar`. Validates the magic, version,
/// flag byte, bitmap length, AND that no trailing bytes follow
/// the bitmap.
pub fn decode_sidecar(bytes: &[u8]) -> Result<TombstonesSidecar, SidecarCodecError> {
    let mut cur = Cursor::new(bytes);

    // 1. Magic.
    let mut magic_buf = [0u8; MAGIC_LEN];
    read_exact(&mut cur, &mut magic_buf, MAGIC_LEN, bytes.len())?;
    if &magic_buf != MAGIC {
        return Err(SidecarCodecError::BadMagic {
            expected: *MAGIC,
            got: magic_buf,
        });
    }

    // 2. Schema version.
    let mut vbuf = [0u8; VERSION_LEN];
    read_exact(&mut cur, &mut vbuf, VERSION_LEN, bytes.len())?;
    let version = u32::from_le_bytes(vbuf);
    if version > SCHEMA_VERSION {
        return Err(SidecarCodecError::UnsupportedVersion {
            got: version,
            max: SCHEMA_VERSION,
        });
    }

    // 3. Seal flag + optional payload.
    let mut fbuf = [0u8; SEAL_FLAG_LEN];
    read_exact(&mut cur, &mut fbuf, SEAL_FLAG_LEN, bytes.len())?;
    let seal = match fbuf[0] {
        SEAL_FLAG_UNSEALED => None,
        SEAL_FLAG_SEALED => {
            let mut uuid_buf = [0u8; SEAL_UUID_LEN];
            read_exact(&mut cur, &mut uuid_buf, SEAL_UUID_LEN, bytes.len())?;
            let compaction_id = Uuid::from_bytes(uuid_buf);
            let mut tbuf = [0u8; SEAL_TIMESTAMP_LEN];
            read_exact(&mut cur, &mut tbuf, SEAL_TIMESTAMP_LEN, bytes.len())?;
            let millis = i64::from_le_bytes(tbuf);
            let sealed_at = Utc
                .timestamp_millis_opt(millis)
                .single()
                .ok_or(SidecarCodecError::InvalidSealTimestamp { millis })?;
            Some(SealRecord {
                compaction_id,
                sealed_at,
            })
        }
        other => return Err(SidecarCodecError::InvalidSealFlag { got: other }),
    };

    // 4. Bitmap length prefix.
    let mut lbuf = [0u8; BITMAP_LEN_PREFIX_LEN];
    read_exact(&mut cur, &mut lbuf, BITMAP_LEN_PREFIX_LEN, bytes.len())?;
    let bitmap_len = u32::from_le_bytes(lbuf);
    let remaining = bytes.len() - (cur.position() as usize);
    if (bitmap_len as usize) > remaining {
        return Err(SidecarCodecError::BitmapTooLong {
            declared: bitmap_len,
            remaining,
        });
    }

    // 5. Bitmap bytes. Bound the read to exactly bitmap_len
    // so a fluky roaring deserialize can't consume trailing
    // bytes silently.
    let bitmap_start = cur.position() as usize;
    let bitmap_end = bitmap_start + bitmap_len as usize;
    let bitmap = RoaringBitmap::deserialize_from(&bytes[bitmap_start..bitmap_end])
        .map_err(SidecarCodecError::BitmapDecode)?;

    // 6. Trailing-bytes check. A correctly-encoded sidecar has
    // exactly `bitmap_end == bytes.len()`; anything past that is
    // either format drift or storage corruption.
    let trailing = bytes.len() - bitmap_end;
    if trailing != 0 {
        return Err(SidecarCodecError::TrailingBytes { trailing });
    }

    let _ = version; // currently no version-specific branching, but pinned for the future

    Ok(TombstonesSidecar { seal, bitmap })
}

#[inline]
fn read_exact(
    cur: &mut Cursor<&[u8]>,
    dst: &mut [u8],
    needed: usize,
    total_len: usize,
) -> Result<(), SidecarCodecError> {
    cur.read_exact(dst)
        .map_err(|_| SidecarCodecError::Truncated {
            needed,
            have: total_len.saturating_sub(cur.position() as usize),
        })
}

#[cfg(test)]
mod tests {
    use chrono::DateTime;

    use super::*;

    fn sample_bitmap() -> RoaringBitmap {
        let mut b = RoaringBitmap::new();
        b.insert(1);
        b.insert(42);
        b.insert(1_000);
        b.insert(100_000);
        b
    }

    #[test]
    fn unsealed_roundtrip() {
        let sidecar = TombstonesSidecar {
            seal: None,
            bitmap: sample_bitmap(),
        };
        let bytes = encode_sidecar(&sidecar).expect("encode");
        let decoded = decode_sidecar(&bytes).expect("decode");
        assert!(decoded.seal.is_none());
        let expected: Vec<u32> = sidecar.bitmap.iter().collect();
        let got: Vec<u32> = decoded.bitmap.iter().collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn sealed_roundtrip() {
        let sealed_at: DateTime<Utc> = "2026-05-30T12:34:56.789Z".parse().expect("ts");
        let sidecar = TombstonesSidecar {
            seal: Some(SealRecord {
                compaction_id: Uuid::from_u128(0x1234_5678_90AB_CDEF_0000_1111_2222_3333),
                sealed_at,
            }),
            bitmap: sample_bitmap(),
        };
        let bytes = encode_sidecar(&sidecar).expect("encode");
        let decoded = decode_sidecar(&bytes).expect("decode");
        let s = decoded.seal.expect("seal preserved");
        assert_eq!(
            s.compaction_id,
            sidecar.seal.as_ref().expect("seal set above").compaction_id
        );
        // Sub-ms components are dropped by the unix-millis
        // encoding; assert ms-truncated equality.
        assert_eq!(s.sealed_at.timestamp_millis(), sealed_at.timestamp_millis());
    }

    #[test]
    fn empty_bitmap_roundtrip() {
        let sidecar = TombstonesSidecar {
            seal: None,
            bitmap: RoaringBitmap::new(),
        };
        let bytes = encode_sidecar(&sidecar).expect("encode");
        let decoded = decode_sidecar(&bytes).expect("decode");
        assert!(decoded.bitmap.is_empty());
    }

    #[test]
    fn rejects_short_input() {
        // Fewer bytes than the magic.
        let err = decode_sidecar(&[0, 1, 2]).expect_err("short");
        assert!(matches!(err, SidecarCodecError::Truncated { .. }));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = encode_sidecar(&TombstonesSidecar {
            seal: None,
            bitmap: RoaringBitmap::new(),
        })
        .expect("encode");
        bytes[0] = b'X';
        let err = decode_sidecar(&bytes).expect_err("bad magic");
        assert!(matches!(err, SidecarCodecError::BadMagic { .. }));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = encode_sidecar(&TombstonesSidecar {
            seal: None,
            bitmap: RoaringBitmap::new(),
        })
        .expect("encode");
        // Bump the LE u32 at offset 8 to a future version.
        let bumped = (SCHEMA_VERSION + 7).to_le_bytes();
        bytes[8..12].copy_from_slice(&bumped);
        let err = decode_sidecar(&bytes).expect_err("future version");
        assert!(matches!(
            err,
            SidecarCodecError::UnsupportedVersion { got, .. } if got == SCHEMA_VERSION + 7
        ));
    }

    #[test]
    fn rejects_invalid_seal_flag() {
        let mut bytes = encode_sidecar(&TombstonesSidecar {
            seal: None,
            bitmap: RoaringBitmap::new(),
        })
        .expect("encode");
        // Offset 12 = seal flag byte.
        bytes[12] = 9;
        let err = decode_sidecar(&bytes).expect_err("bad flag");
        assert!(matches!(err, SidecarCodecError::InvalidSealFlag { got: 9 }));
    }

    #[test]
    fn rejects_trailing_garbage() {
        let mut bytes = encode_sidecar(&TombstonesSidecar {
            seal: None,
            bitmap: sample_bitmap(),
        })
        .expect("encode");
        bytes.push(0xAA);
        bytes.push(0xBB);
        let err = decode_sidecar(&bytes).expect_err("trailing");
        assert!(matches!(
            err,
            SidecarCodecError::TrailingBytes { trailing: 2 }
        ));
    }

    #[test]
    fn rejects_bitmap_length_past_buffer() {
        let mut bytes = encode_sidecar(&TombstonesSidecar {
            seal: None,
            bitmap: sample_bitmap(),
        })
        .expect("encode");
        // The bitmap-len prefix sits at offset 13 (unsealed).
        let huge: u32 = 0xFFFF_FFFF;
        bytes[13..17].copy_from_slice(&huge.to_le_bytes());
        let err = decode_sidecar(&bytes).expect_err("over-length");
        assert!(matches!(err, SidecarCodecError::BitmapTooLong { .. }));
    }

    #[test]
    fn magic_offset_is_stable() {
        // Lock in that the magic occupies bytes [0..8) verbatim,
        // even with seal=Some + a non-empty bitmap. This guards
        // against an accidental reordering of fields in encode.
        let bytes = encode_sidecar(&TombstonesSidecar {
            seal: Some(SealRecord {
                compaction_id: Uuid::nil(),
                sealed_at: Utc.timestamp_millis_opt(0).single().expect("ts"),
            }),
            bitmap: sample_bitmap(),
        })
        .expect("encode");
        assert_eq!(&bytes[0..8], MAGIC);
        assert_eq!(&bytes[8..12], &SCHEMA_VERSION.to_le_bytes());
        assert_eq!(bytes[12], 1);
    }
}
