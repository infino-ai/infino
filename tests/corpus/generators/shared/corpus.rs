// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

// The documents every corpus generator writes, shared verbatim so tables
// written by different engine versions differ only in format shape.
//
// Included with `include!` rather than shared as a crate: each generator
// pins its own engine version and resolves its own dependency graph, and a
// common crate would have to satisfy all of them at once.

/// Documents per generated table. Sized so the `common` term clears
/// `BLOCK_LEN * COARSE_BLOCK_MAX_SPAN` (128 * 32 = 4096) postings and its
/// coarse block-max table holds many entries rather than one — the
/// structure a smaller corpus leaves untested.
pub const N_DOCS: u32 = 12_000;

/// Characters in the unbroken run planted in one document. Past any token
/// cap the analyzer applies, so the run indexes as a single unreachable
/// term before the tokenization fix and as findable pieces after it.
const LONG_RUN_CHARS: usize = 512;

/// Document carrying the over-cap run.
pub const LONG_RUN_DOC: u32 = 100;
/// Document carrying emoji, which older analyzers dropped entirely.
pub const EMOJI_DOC: u32 = 200;
/// Document carrying non-ASCII text.
pub const NON_ASCII_DOC: u32 = 300;

/// Run length and gap of the `burst` term, chosen so one posting block
/// holds a long consecutive run plus one large jump: the block's max delta
/// is then wide enough that the presence bitset encodes smaller than the
/// packed deltas, which is what makes the builder choose the bitset
/// encoding. A uniformly dense term does not — its deltas are all 1 and
/// pack to a single bit.
const BURST_RUN: u32 = 127;
const BURST_PERIOD: u32 = 327;

/// The `body` column: carries `common` in every document, so its posting
/// list is long enough to exercise multi-block skipping and the coarse
/// level.
pub fn body(i: u32) -> String {
    let mut s = String::from("common");
    if i % BURST_PERIOD < BURST_RUN {
        s.push_str(" burst");
    }
    match i % 4 {
        0 => s.push_str(" alpha shared"),
        1 => s.push_str(" beta shared pad pad"),
        2 => s.push_str(" gamma pad"),
        _ => s.push_str(" delta shared pad pad pad pad"),
    }
    match i {
        LONG_RUN_DOC => {
            s.push(' ');
            s.push_str(&"z".repeat(LONG_RUN_CHARS));
        }
        EMOJI_DOC => s.push_str(" 🔥 cat 🐈‍⬛ dog"),
        NON_ASCII_DOC => s.push_str(" café naïve 東京"),
        _ => {}
    }
    s.push_str(&format!(" d{i}"));
    s
}

/// The `title` column: shorter text, and the column generators make
/// positional where the engine version allows it.
pub fn title(i: u32) -> String {
    match i % 3 {
        0 => format!("quick brown fox t{i}"),
        1 => format!("lazy dog jumps t{i}"),
        _ => format!("brown dog t{i}"),
    }
}

/// The `notes` column: null for three rows in four, and null for every row
/// in `all_null`. A column's BM25 statistics must count the documents that
/// carry tokens rather than the table's rows, and a densely-filled corpus
/// cannot tell those apart.
pub fn notes(i: u32) -> Option<String> {
    match i % 4 {
        0 => Some(format!("note common sparse n{i}")),
        _ => None,
    }
}
