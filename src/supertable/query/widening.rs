// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! How far a top-`k` search widens when too few of its hits survive.
//!
//! A search asks its kernel for `k` hits, then drops some of them: rows a
//! `WHERE` the index cannot answer rejects, or rows deleted since the index
//! was built. When fewer than `k` survive, the search runs again for more
//! hits. Each rerun pays the whole search again, so the number of rounds is
//! what a selective filter or a dense run of deletes costs. [`next_want`]
//! sizes each round from the pass rate the last one saw, for the full-text
//! and vector `WHERE` fills and the vector search's refill past deletes.

/// The least a widening round grows by: [`next_want`] never asks for fewer
/// than this many times the last round's hits. Growing by at least this much
/// bounds the total work at about twice the final round's.
pub(crate) const OVER_FETCH_GROWTH: usize = 2;

/// How many times a round grows `want` after a round in which no hit
/// survived: with no pass rate to size the next round by, it looks much
/// deeper than [`OVER_FETCH_GROWTH`] would.
pub(crate) const OVER_FETCH_GROWTH_NONE_PASSED: usize = 16;

/// The hits the next round asks for, after a round that asked for `want`
/// kept `kept` of them (passing the `WHERE`, or live past the deletes):
/// enough hits to find `k` at the pass rate seen, twice over,
/// never less than [`OVER_FETCH_GROWTH`] times `want` and never past
/// `ceiling`. Past the depth an index can answer from, a round costs about
/// the same whatever its `want`, so a round sized to the pass rate saves the
/// rounds doubling would spend getting there.
pub(crate) fn next_want(k: usize, want: usize, kept: usize, ceiling: usize) -> usize {
    let sized = match kept {
        0 => want.saturating_mul(OVER_FETCH_GROWTH_NONE_PASSED),
        kept => want
            .saturating_mul(k)
            .div_ceil(kept)
            .saturating_mul(OVER_FETCH_GROWTH),
    };
    sized
        .max(want.saturating_mul(OVER_FETCH_GROWTH))
        .min(ceiling)
}

#[cfg(test)]
mod tests {
    use crate::supertable::query::{exec::common::over_fetch_ceiling, widening::*};

    /// Rows in a table large enough that its row count never caps a fill.
    const HUGE_TABLE: usize = 100_000_000;

    /// The `k` of a typical top-`k` search.
    const SMALL_K: usize = 10;

    /// How many rounds a fill for [`SMALL_K`] rows takes when one hit in
    /// `.0` survives.
    const ROUNDS_BY_SURVIVAL: [(usize, usize); 5] = [(1, 1), (2, 2), (10, 2), (100, 3), (1_000, 4)];

    /// Rounds a fill for `k` rows takes when one hit in `one_in` survives,
    /// each round sized by [`next_want`] from the last.
    fn rounds_to_fill(k: usize, one_in: usize, ceiling: usize) -> usize {
        let mut want = k;
        let mut rounds = 1;
        while want / one_in < k && want < ceiling {
            want = next_want(k, want, want / one_in, ceiling);
            rounds += 1;
        }
        rounds
    }

    /// Each widening round reruns the whole search, so a selective filter
    /// pays per round. Sized from the last round's pass rate, one row in a
    /// hundred fills in three rounds and one in a thousand in four.
    #[test]
    fn a_selective_filter_fills_in_a_few_rounds() {
        let ceiling = over_fetch_ceiling(SMALL_K, HUGE_TABLE);
        for (one_in, rounds) in ROUNDS_BY_SURVIVAL {
            assert_eq!(
                rounds_to_fill(SMALL_K, one_in, ceiling),
                rounds,
                "one in {one_in}"
            );
        }
    }
}
