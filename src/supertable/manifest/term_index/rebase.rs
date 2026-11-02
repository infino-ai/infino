// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Moving a built layer to other ordinals without building it again.
//!
//! A layer names its superfiles by their ordinal in the root it is appended
//! to. A layer built for a commit that then loses its race to commits that
//! grew the root past the layer's first ordinal can no longer be appended as
//! it is. Every ordinal it names is one of its own superfiles', numbered
//! consecutively, so moving it is one constant shift of every ordinal: the
//! order of the superfiles, and with it every tier's grouping, cut and
//! encoding, stays as it was. The shift rewrites each ordinal where it sits:
//! a posting's and an inline tier entry's varint at its own width, and a
//! chunk directory's `u32`, with the directory's CRC. Nothing else in a
//! slice or a tier object changes, so a moved layer is the bytes a build at
//! the new ordinals would have written, at a fraction of the cost.

use std::{collections::HashMap, fs, io, sync::Arc};

use rayon::prelude::*;

use super::{
    BuiltBytes, TermIndexError,
    format::{Layer, SliceRef, rebase_slice},
};
use crate::{
    supertable::manifest::part::ContentHash,
    utils::varint::{CONTINUATION_BIT, read_varint, varint_len},
};

/// Payload bits per LEB128 byte.
const VARINT_PAYLOAD_BITS: u32 = 7;
/// Payload mask per LEB128 byte.
const VARINT_PAYLOAD_MASK: u32 = 0x7f;

/// The move of a layer's `n` ordinals from `from..from + n` to
/// `to..to + n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OrdinalShift {
    from: u32,
    to: u32,
    n: u32,
}

impl OrdinalShift {
    /// The shift, or `None` when it cannot be made in place: some ordinal
    /// would take another number of bytes as a posting writes it
    /// (`ordinal << 1 | tail`) or as an inline tier entry does (the ordinal
    /// alone), or would not fit. The layer is then built again.
    ///
    /// A varint's width only grows with its value, so the ordinals of a
    /// range share one width when its ends do.
    pub(crate) fn new(from: u32, to: u32, n: u32) -> Option<Self> {
        let shift = Self { from, to, n };
        let Some(last) = n.checked_sub(1) else {
            return Some(shift);
        };
        let (from_last, to_last) = (from.checked_add(last)?, to.checked_add(last)?);
        if from_last.max(to_last) > u32::MAX >> 1 {
            return None;
        }
        let one_width = |values: [u32; 4]| {
            values
                .iter()
                .all(|v| varint_len(*v) == varint_len(values[0]))
        };
        let posting = |o: u32| o << 1;
        let tagged = one_width([
            posting(from),
            posting(from_last) | 1,
            posting(to),
            posting(to_last) | 1,
        ]);
        let plain = one_width([from, from_last, to, to_last]);
        (tagged && plain).then_some(shift)
    }

    /// Where `ordinal` moves to; an error for an ordinal outside the
    /// layer's, which a layer built for one batch never names.
    pub(crate) fn apply(&self, ordinal: u32) -> Result<u32, TermIndexError> {
        ordinal
            .checked_sub(self.from)
            .filter(|i| *i < self.n)
            .map(|i| self.to + i)
            .ok_or_else(|| {
                TermIndexError::Malformed(format!(
                    "rebase: ordinal {ordinal} is not one of the layer's {}..{}",
                    self.from,
                    self.from + self.n
                ))
            })
    }
}

/// Rewrite the varint at `at` with `f` of its value, at the same width, and
/// advance past it. An error when the new value needs another width.
pub(crate) fn rewrite_varint(
    bytes: &mut [u8],
    at: &mut usize,
    f: impl FnOnce(u32) -> Result<u32, TermIndexError>,
) -> Result<(), TermIndexError> {
    let start = *at;
    let old = read_varint(bytes, at)
        .ok_or_else(|| TermIndexError::Malformed("rebase: truncated varint".into()))?;
    let mut value = f(old)?;
    let width = *at - start;
    if varint_len(value) != width {
        return Err(TermIndexError::Malformed(
            "rebase: an ordinal changed its encoded width".into(),
        ));
    }
    for (i, byte) in bytes[start..*at].iter_mut().enumerate() {
        let more = if i + 1 < width { CONTINUATION_BIT } else { 0 };
        *byte = (value & VARINT_PAYLOAD_MASK) as u8 | more;
        value >>= VARINT_PAYLOAD_BITS;
    }
    Ok(())
}

/// A layer moved to other ordinals: its slice list and the bytes to write.
pub(crate) struct RebasedLayer {
    pub(crate) layer: Layer,
    pub(crate) slices: Vec<(ContentHash, BuiltBytes)>,
    pub(crate) tiers: Vec<(ContentHash, BuiltBytes)>,
}

/// One slice moved: its new reference, its bytes and its tier object's.
type MovedSlice = (
    SliceRef,
    (ContentHash, BuiltBytes),
    Option<(ContentHash, BuiltBytes)>,
);

/// Move `layer`, whose bytes are `slices` and `tiers`, through `shift`. The
/// slices are moved in parallel on the current rayon pool, each from a copy
/// of its bytes, so the given ones are left as they were whatever happens.
/// A moved object is kept where its original was: in memory, or as a new
/// file beside a spilled one.
pub(crate) fn rebase_layer(
    layer: &Layer,
    slices: &[(ContentHash, BuiltBytes)],
    tiers: &[(ContentHash, BuiltBytes)],
    shift: &OrdinalShift,
) -> Result<RebasedLayer, TermIndexError> {
    let slice_bytes: HashMap<&ContentHash, &BuiltBytes> =
        slices.iter().map(|(h, b)| (h, b)).collect();
    let tier_bytes: HashMap<&ContentHash, &BuiltBytes> =
        tiers.iter().map(|(h, b)| (h, b)).collect();
    let moved = layer
        .slices
        .par_iter()
        .map(|slice_ref| -> Result<MovedSlice, TermIndexError> {
            let slice_source = kept(&slice_bytes, &slice_ref.content_hash)?;
            let tier_source = slice_ref
                .tier
                .as_ref()
                .map(|hash| kept(&tier_bytes, hash))
                .transpose()?;
            let mut slice = read(slice_source)?;
            let mut tier = tier_source.map(read).transpose()?;
            rebase_slice(&mut slice, tier.as_deref_mut(), shift)?;
            let tier = match (tier_source, tier) {
                (Some(source), Some(bytes)) => {
                    let hash = ContentHash::of(&bytes);
                    Some((hash, keep(source, bytes, &hash)?))
                }
                _ => None,
            };
            let hash = ContentHash::of(&slice);
            let moved_ref = SliceRef {
                content_hash: hash,
                tier: tier.as_ref().map(|(h, _)| *h),
                ..slice_ref.clone()
            };
            Ok((moved_ref, (hash, keep(slice_source, slice, &hash)?), tier))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut rebased = RebasedLayer {
        layer: Layer {
            slices: Vec::new(),
            ordinals: Some(shift.to..shift.to + shift.n),
        },
        slices: Vec::with_capacity(moved.len()),
        tiers: Vec::new(),
    };
    for (slice_ref, slice, tier) in moved {
        rebased.layer.slices.push(slice_ref);
        rebased.slices.push(slice);
        // Two slices whose tier objects were byte-identical moved them to
        // the same bytes; one copy is written.
        if let Some(tier) = tier
            && !rebased.tiers.iter().any(|(h, _)| *h == tier.0)
        {
            rebased.tiers.push(tier);
        }
    }
    Ok(rebased)
}

/// The kept bytes of the object named `hash`.
fn kept<'a>(
    bytes: &HashMap<&ContentHash, &'a BuiltBytes>,
    hash: &ContentHash,
) -> Result<&'a BuiltBytes, TermIndexError> {
    bytes.get(hash).copied().ok_or_else(|| {
        TermIndexError::Malformed("rebase: an object the layer names was not kept".into())
    })
}

/// Remove the files `objects` were spilled to, best effort: a moved layer
/// leaves its originals behind, and nothing reads them again.
pub(crate) fn remove_spilled(objects: &[(ContentHash, BuiltBytes)]) {
    for (_, bytes) in objects {
        if let BuiltBytes::Spilled { path, .. } = bytes {
            let _ = fs::remove_file(path);
        }
    }
}

/// The bytes of `built`, read back when spilled.
fn read(built: &BuiltBytes) -> Result<Vec<u8>, TermIndexError> {
    Ok(match built {
        BuiltBytes::Resident(bytes) => bytes.clone(),
        BuiltBytes::Spilled { path, .. } => fs::read(path)?,
    })
}

/// Keep `bytes`, hashing to `hash`, where `source` was kept: in memory, or
/// in a new file beside a spilled one, in the same scratch directory.
/// Two slices with byte-identical tier objects may write the same file at
/// once; the bytes are the same, so the file is correct either way.
fn keep(source: &BuiltBytes, bytes: Vec<u8>, hash: &ContentHash) -> io::Result<BuiltBytes> {
    Ok(match source {
        BuiltBytes::Resident(_) => BuiltBytes::Resident(bytes),
        BuiltBytes::Spilled { _dir, path } => {
            let path = path
                .parent()
                .ok_or_else(|| io::Error::other("rebase: a spilled object has no directory"))?
                .join(format!("rebased-{}", hash.to_hex()));
            fs::write(&path, bytes)?;
            BuiltBytes::Spilled {
                _dir: Arc::clone(_dir),
                path,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::varint::push_varint;

    /// Ordinals that end the one-, two- and three-byte widths of a
    /// posting's `ordinal << 1 | tail`.
    const ONE_BYTE_LAST: u32 = 63;
    const TWO_BYTE_LAST: u32 = 8_191;
    const THREE_BYTE_LAST: u32 = 1_048_575;

    #[test]
    fn a_shift_within_one_width_is_made_in_place_either_way() {
        assert!(OrdinalShift::new(300, 5_000, 3).is_some(), "up");
        assert!(OrdinalShift::new(5_000, 300, 3).is_some(), "down");
        assert!(OrdinalShift::new(70_000, 900_000, 32).is_some());
        assert!(OrdinalShift::new(7, 7, 0).is_some(), "an empty layer");
    }

    #[test]
    fn a_shift_across_a_width_is_refused() {
        // Every range end, posting and inline width alike, must stay put.
        assert!(OrdinalShift::new(10, ONE_BYTE_LAST + 1, 1).is_none());
        assert!(
            OrdinalShift::new(300, TWO_BYTE_LAST - 1, 3).is_none(),
            "the range straddles"
        );
        assert!(OrdinalShift::new(9_000, THREE_BYTE_LAST + 1, 1).is_none());
        assert!(
            OrdinalShift::new(0, u32::MAX >> 1, 2).is_none(),
            "past what a posting holds"
        );
    }

    #[test]
    fn a_shift_moves_only_the_layers_ordinals() {
        let shift = OrdinalShift::new(300, 5_000, 3).expect("shift");
        assert_eq!(shift.apply(300).expect("first"), 5_000);
        assert_eq!(shift.apply(302).expect("last"), 5_002);
        assert!(shift.apply(299).is_err());
        assert!(shift.apply(303).is_err());
    }

    #[test]
    fn a_varint_is_rewritten_in_its_own_bytes() {
        let mut bytes = vec![0xaa];
        push_varint(&mut bytes, 600);
        bytes.push(0xbb);
        let mut at = 1;
        rewrite_varint(&mut bytes, &mut at, |v| Ok(v + 9_000)).expect("rewrite");
        assert_eq!(at, 3, "past the varint");
        let mut expected = vec![0xaa];
        push_varint(&mut expected, 9_600);
        expected.push(0xbb);
        assert_eq!(bytes, expected);

        let mut at = 1;
        assert!(
            rewrite_varint(&mut bytes, &mut at, |_| Ok(1)).is_err(),
            "a narrower value"
        );
    }
}
