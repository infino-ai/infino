// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Short-form posting body for terms whose whole list fits one block
//! (`df <= BLOCK_LEN`) — the `VERSION_V7` layout for the rare-term tail.
//!
//! The long form charges a single-block term its metadata header, a
//! skip entry, a position sub-index row, a coarse slot and a block
//! header before the first posting, and the block codec then pads the
//! partial block to `BLOCK_LEN` lanes at the block's bit width — on a
//! Zipfian corpus that made the ~97% of terms with `df <= 128` more
//! than half of the postings region while holding 6% of the postings.
//! The short form is a few bytes per posting and nothing per block:
//!
//! ```text
//!   varint   df
//!   bitmap   ceil(df / 8) bytes; bit i set ⇔ tf_i == 1
//!   groupvar df doc-id deltas (first absolute, then doc − prev)
//!   varint   tf_i for every i with tf_i != 1, in posting order
//!   group    the term's position group (positional column only; see
//!            `positions::encode_group`) — inline, since a short term's
//!            body is fetched whole and read once
//! ```
//!
//! Group-varint packs four values behind one control byte (two bits per
//! value: its byte length minus one), the values following in
//! little-endian truncated form. A tail group with fewer than four
//! values still spends the control byte; its unused slots are ignored.
//! Folding `tf == 1` into a bitmap rather than into the delta's low bit
//! keeps every delta a plain `u32`, so a doc id anywhere in the `u32`
//! range encodes without a widening step.
//!
//! A short body is read once and whole: the reader decodes it straight
//! into the pre-filled single-block cursor the df=1 inline form already
//! uses, so nothing downstream distinguishes the two.

use crate::{
    superfile::fts::posting::BLOCK_LEN,
    utils::varint::{push_varint, read_varint},
};

/// Largest posting count the short form is used for — one block.
pub(crate) const SHORT_MAX_DF: usize = BLOCK_LEN;

/// Decoded short body: `n` postings in `doc_ids[..n]` / `tfs[..n]`, plus
/// where the inline position group starts when the column has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ShortDecoded {
    pub(crate) n: usize,
    /// Byte offset within the body of the term's position group.
    pub(crate) positions_at: Option<usize>,
}

/// Byte length (1..=4) of `v` in group-varint, minus one.
#[inline]
fn group_len_code(v: u32) -> u8 {
    match v {
        0..=0xff => 0,
        0x100..=0xffff => 1,
        0x1_0000..=0xff_ffff => 2,
        _ => 3,
    }
}

/// Append `values` as group-varint.
fn push_group_varint(out: &mut Vec<u8>, values: &[u32]) {
    for group in values.chunks(4) {
        let mut control: u8 = 0;
        for (i, &v) in group.iter().enumerate() {
            control |= group_len_code(v) << (2 * i);
        }
        out.push(control);
        for (i, &v) in group.iter().enumerate() {
            let len = ((control >> (2 * i)) & 3) as usize + 1;
            out.extend_from_slice(&v.to_le_bytes()[..len]);
        }
    }
}

/// Decode `n` group-varint values from `bytes` at `*at` into `out[..n]`.
/// `None` on truncation.
fn read_group_varint(bytes: &[u8], at: &mut usize, n: usize, out: &mut [u32]) -> Option<()> {
    let mut i = 0;
    while i < n {
        let control = *bytes.get(*at)?;
        *at += 1;
        let in_group = (n - i).min(4);
        for j in 0..in_group {
            let len = ((control >> (2 * j)) & 3) as usize + 1;
            let slice = bytes.get(*at..*at + len)?;
            let mut buf = [0u8; 4];
            buf[..len].copy_from_slice(slice);
            out[i + j] = u32::from_le_bytes(buf);
            *at += len;
        }
        i += in_group;
    }
    Some(())
}

/// Encode one term's postings (`pairs` = `(doc_id, tf)`, doc ids
/// strictly ascending, `1..=SHORT_MAX_DF` of them) into `out`.
/// `positions` is the term's encoded position group on a positional
/// column, appended inline.
pub(crate) fn encode_short(out: &mut Vec<u8>, pairs: &[(u32, u32)], positions: Option<&[u8]>) {
    let n = pairs.len();
    assert!(
        (1..=SHORT_MAX_DF).contains(&n),
        "short form takes 1..={SHORT_MAX_DF} postings, got {n}"
    );
    debug_assert!(
        pairs.windows(2).all(|w| w[0].0 < w[1].0),
        "short form needs strictly ascending doc ids"
    );
    push_varint(out, n as u32);
    let bitmap_start = out.len();
    out.resize(bitmap_start + n.div_ceil(8), 0);
    let mut deltas = [0u32; SHORT_MAX_DF];
    let mut prev = 0u32;
    for (i, &(doc, tf)) in pairs.iter().enumerate() {
        if tf == 1 {
            out[bitmap_start + i / 8] |= 1 << (i % 8);
        }
        deltas[i] = if i == 0 { doc } else { doc - prev };
        prev = doc;
    }
    push_group_varint(out, &deltas[..n]);
    for &(_, tf) in pairs {
        if tf != 1 {
            push_varint(out, tf);
        }
    }
    if let Some(group) = positions {
        out.extend_from_slice(group);
    }
}

/// The `df` a short body declares — its leading varint. `None` on a
/// malformed body.
#[inline]
pub(crate) fn short_df(body: &[u8]) -> Option<u32> {
    let mut at = 0;
    let df = read_varint(body, &mut at)?;
    (1..=SHORT_MAX_DF as u32).contains(&df).then_some(df)
}

/// Decode a short body into `doc_ids[..n]` / `tfs[..n]` (both at least
/// [`SHORT_MAX_DF`] long). `positional` selects whether a position group
/// follows the postings (its bytes are left to the phrase decode). `None`
/// on any malformed input — truncation, a `df` outside `1..=SHORT_MAX_DF`,
/// a non-ascending doc id, trailing garbage on a positionless body.
pub(crate) fn decode_short(
    body: &[u8],
    positional: bool,
    doc_ids: &mut [u32],
    tfs: &mut [u32],
) -> Option<ShortDecoded> {
    let mut at = 0usize;
    let n = read_varint(body, &mut at)? as usize;
    if !(1..=SHORT_MAX_DF).contains(&n) || doc_ids.len() < n || tfs.len() < n {
        return None;
    }
    let bitmap_len = n.div_ceil(8);
    let bitmap_at = at;
    body.get(at..at + bitmap_len)?;
    at += bitmap_len;
    read_group_varint(body, &mut at, n, doc_ids)?;
    // Prefix-sum the deltas; the first is absolute.
    for i in 1..n {
        doc_ids[i] = doc_ids[i - 1].checked_add(doc_ids[i])?;
        if doc_ids[i] <= doc_ids[i - 1] {
            return None;
        }
    }
    let bitmap = &body[bitmap_at..bitmap_at + bitmap_len];
    for i in 0..n {
        tfs[i] = if (bitmap[i / 8] >> (i % 8)) & 1 == 1 {
            1
        } else {
            let tf = read_varint(body, &mut at)?;
            if tf < 2 {
                return None;
            }
            tf
        };
    }
    match positional {
        true => (at < body.len()).then_some(ShortDecoded {
            n,
            positions_at: Some(at),
        }),
        false => (at == body.len()).then_some(ShortDecoded {
            n,
            positions_at: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(pairs: &[(u32, u32)], positions: Option<&[u8]>) -> Vec<u8> {
        let mut body = Vec::new();
        encode_short(&mut body, pairs, positions);
        let mut d = [0u32; SHORT_MAX_DF];
        let mut t = [0u32; SHORT_MAX_DF];
        let got = decode_short(&body, positions.is_some(), &mut d, &mut t).expect("decodes");
        assert_eq!(got.n, pairs.len());
        match positions {
            Some(group) => assert_eq!(&body[got.positions_at.expect("group")..], group),
            None => assert_eq!(got.positions_at, None),
        }
        for (i, &(doc, tf)) in pairs.iter().enumerate() {
            assert_eq!((d[i], t[i]), (doc, tf), "posting {i}");
        }
        assert_eq!(short_df(&body), Some(pairs.len() as u32));
        body
    }

    #[test]
    fn two_far_apart_docs_cost_a_few_bytes_not_hundreds() {
        // The case that motivated the form: delta_bits ≈ 20 padded to 128
        // lanes cost the long form 320 B of doc ids for two postings.
        let body = round_trip(&[(7, 1), (900_007, 1)], None);
        // df(1) + bitmap(1) + control(1) + 1 + 3 = 7 bytes.
        assert_eq!(body.len(), 7);
    }

    #[test]
    fn inline_position_group_round_trips() {
        round_trip(
            &[(0, 3), (1, 1), (2, 1), (1_000_000, 7)],
            Some(&[1, 2, 3, 4, 5, 6, 7]),
        );
    }

    #[test]
    fn full_block_and_single_posting_round_trip() {
        let full: Vec<(u32, u32)> = (0..SHORT_MAX_DF as u32)
            .map(|i| (i * 3 + 1, i % 5 + 1))
            .collect();
        round_trip(&full, None);
        round_trip(&[(u32::MAX, 2)], None);
        round_trip(&[(u32::MAX, 1)], Some(&[0u8, 9]));
    }

    #[test]
    fn every_group_varint_width_round_trips() {
        let pairs: Vec<(u32, u32)> = vec![
            (0, 1),
            (0xff, 1),
            (0xff + 0xffff, 2),
            (0xff + 0xffff + 0xff_ffff, 1),
            (u32::MAX, 300),
        ];
        round_trip(&pairs, None);
    }

    #[test]
    fn malformed_bodies_are_refused_not_panicked() {
        let mut d = [0u32; SHORT_MAX_DF];
        let mut t = [0u32; SHORT_MAX_DF];
        let mut body = Vec::new();
        encode_short(&mut body, &[(5, 1), (9, 4)], None);
        for cut in 0..body.len() {
            assert!(
                decode_short(&body[..cut], false, &mut d, &mut t).is_none(),
                "cut {cut}"
            );
        }
        let mut extra = body.clone();
        extra.push(0);
        assert!(
            decode_short(&extra, false, &mut d, &mut t).is_none(),
            "trailing byte"
        );
        assert!(
            decode_short(&body, true, &mut d, &mut t).is_none(),
            "a positional body needs its group"
        );
        assert!(short_df(&[0]).is_none(), "df 0");
        assert!(short_df(&[129]).is_none(), "df past a block");
        // A zero delta (duplicate doc) is refused.
        let mut dup = Vec::new();
        push_varint(&mut dup, 2);
        dup.push(0b11);
        push_group_varint(&mut dup, &[4, 0]);
        assert!(decode_short(&dup, false, &mut d, &mut t).is_none());
    }

    #[test]
    #[should_panic(expected = "short form takes")]
    fn more_than_a_block_panics() {
        let too_many: Vec<(u32, u32)> = (0..=SHORT_MAX_DF as u32).map(|i| (i, 1)).collect();
        encode_short(&mut Vec::new(), &too_many, None);
    }
}
