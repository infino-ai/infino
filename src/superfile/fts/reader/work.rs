// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Per-query FTS work accounting: [`MatchWork`] (the posting bytes /
//! planned ranges / kernel-CPU a walk performs, fed to op_stats) and the
//! cursor-metadata sums behind it. `MatchWork` is part of the
//! `fts::reader::*` surface; the summing helpers are `pub(super)`.

use super::{cursor::TermCursor, phrase::AnyCursor};

/// Sum of the posting-byte ranges a cursor set indexes into (each cursor's
/// term metadata + skip table + posting blocks). Feeds the per-query work
/// stats ([`crate::runtime_metrics::op_stats`]).
pub(super) fn term_cursor_bytes(cursors: &[TermCursor]) -> u64 {
    cursors.iter().map(|c| c.bytes.len() as u64).sum()
}

/// [`term_cursor_bytes`] for heterogeneous atoms: a phrase member counts
/// its posting bytes **and** its position runs — positional verification
/// is exactly the work that separates phrase cost from term cost.
pub(super) fn atom_cursor_bytes(atoms: &[AnyCursor]) -> u64 {
    atoms
        .iter()
        .map(|a| match a {
            AnyCursor::Term(c) => c.bytes.len() as u64,
            AnyCursor::Phrase(p) => p
                .members
                .iter()
                .map(|m| m.cursor.bytes.len() as u64 + m.positions.len() as u64)
                .sum(),
        })
        .sum()
}

/// Byte-source ranges a cursor set's build requested — one per PFOR
/// term. Inline (df=1) cursors plan no fetch (their `bytes` is empty),
/// matching the single-term arm's "bytes 0 implies ranges 0".
pub(super) fn term_cursor_ranges(cursors: &[TermCursor]) -> u64 {
    cursors
        .iter()
        .filter(|c| !c.bytes.is_empty())
        .map(|c| 1 + u64::from(c.header_probed))
        .sum()
}

/// Byte-source ranges the atoms' builds requested: one per PFOR term's
/// posting range; a phrase member adds one for its postings and one for
/// its position runs. Inline legs (empty buffers) plan no fetch.
pub(super) fn atom_planned_ranges(atoms: &[AnyCursor]) -> u64 {
    let term_ranges = |c: &TermCursor| {
        if c.bytes.is_empty() {
            0
        } else {
            1 + u64::from(c.header_probed)
        }
    };
    atoms
        .iter()
        .map(|a| match a {
            AnyCursor::Term(c) => term_ranges(c),
            AnyCursor::Phrase(p) => p
                .members
                .iter()
                .map(|m| term_ranges(&m.cursor) + u64::from(!m.positions.is_empty()))
                .sum(),
        })
        .sum()
}

/// Work tallies from one unranked match / dictionary walk — the posting
/// bytes indexed and the byte-source ranges the plan requested. Returned
/// alongside match results so the supertable flushes once per superfile;
/// `pub` because the carrying fns are the test-helpers-widened surface
/// (the module gate in `lib.rs` keeps it crate-private in normal builds).
#[derive(Debug, Default, Clone, Copy)]
pub struct MatchWork {
    /// Posting (and phrase-position / header) bytes the walk indexed into.
    pub postings_bytes: u64,
    /// Byte-source ranges the walk's build requested, pre-coalesce.
    pub planned_ranges: u64,
    /// Bracketed on-CPU ns of the walk's synchronous scoring/merge
    /// section (gated on `metering_active`; 0 when unmetered).
    pub kernel_cpu_ns: u64,
}

impl MatchWork {
    /// Tallies for a plain term-cursor set.
    pub(super) fn for_cursors(cursors: &[TermCursor]) -> Self {
        Self {
            postings_bytes: term_cursor_bytes(cursors),
            planned_ranges: term_cursor_ranges(cursors),
            kernel_cpu_ns: 0,
        }
    }

    /// Tallies for a heterogeneous atom set (terms + phrases).
    pub(super) fn for_atoms(atoms: &[AnyCursor]) -> Self {
        Self {
            postings_bytes: atom_cursor_bytes(atoms),
            planned_ranges: atom_planned_ranges(atoms),
            kernel_cpu_ns: 0,
        }
    }

    /// Fold another walk's tallies into this one (e.g. a negation set).
    pub fn merge(&mut self, other: MatchWork) {
        self.postings_bytes += other.postings_bytes;
        self.planned_ranges += other.planned_ranges;
        self.kernel_cpu_ns += other.kernel_cpu_ns;
    }
}
