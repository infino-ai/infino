// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Query-option types for FTS search: the default-operator [`BoolMode`],
//! the [`Bm25Stats`] idf-source selector, and the [`Bm25SearchOptions`]
//! builder. Part of the `fts::reader::*` public surface.

use crate::superfile::fts::bm25::Bm25Params;

/// Default operator for a query's bare (sigil-less) terms. Terms
/// carrying an explicit clause sigil keep their polarity regardless
/// of mode: `+term` is a must (every hit contains it), `-term` a
/// must-not (hard exclusion).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub enum BoolMode {
    /// Bare terms are musts: all of them must match the doc.
    And,
    /// Bare terms are shoulds: any of them matching contributes to
    /// the doc's score. When the query also carries `+must` terms,
    /// the musts alone define the match set and bare terms become
    /// scoring-only. The default.
    #[default]
    Or,
}

/// Which BM25 collection statistics to score term rarity (idf) with
/// across the superfiles a query fans out over.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub enum Bm25Stats {
    /// Score each superfile against its own local document count and
    /// term document-frequencies. Fast (full fan-out, no extra pass),
    /// but a term's idf — and therefore a doc's score — depends on
    /// which superfile it lands in, so scores are only approximately
    /// comparable across superfiles and ranking drifts as the table
    /// fragments.
    PerSuperfile,
    /// Score every superfile against table-wide idf: the document count
    /// and per-term document-frequencies aggregated across all
    /// superfiles in the query's manifest snapshot. A term then has one
    /// idf for the whole table, so a fragmented table ranks like a
    /// single unified corpus, at the cost of a document-frequency
    /// gather before scoring. (Length normalization still uses each
    /// superfile's own average document length.) The default: ranking
    /// should not depend on how commits happened to shard the corpus.
    #[default]
    Global,
}

impl From<&str> for Bm25Stats {
    fn from(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "global" => Bm25Stats::Global,
            "per_superfile" => Bm25Stats::PerSuperfile,
            // Anything else is treated as "unspecified" and takes the
            // default, so this conversion and the enum default cannot
            // drift apart.
            _ => Bm25Stats::default(),
        }
    }
}

/// Options for a BM25 search: the boolean `mode` and the corpus-statistics
/// `stats`. Set fields with the `with_*` builders; [`Default`] is
/// [`BoolMode::Or`] with [`Bm25Stats::Global`].
///
/// ```ignore
/// // OR mode, global stats (the defaults):
/// Bm25SearchOptions::new()
/// // AND mode, per-superfile (segment-local) stats:
/// Bm25SearchOptions::new().with_mode(BoolMode::And).with_stats(Bm25Stats::PerSuperfile)
/// ```
/// `#[non_exhaustive]`: construct with [`Bm25SearchOptions::new`] and
/// the `with_*` setters. The attribute is what lets a further search
/// option be added without breaking callers — adding a field to a
/// struct that can be built with a literal is a breaking change no
/// matter what the field defaults to, because Rust literals must name
/// every field.
///
/// `Eq` is deliberately absent: [`Bm25SearchOptions::bm25`] holds
/// `f32`s, which have no total equality. `PartialEq` is derived.
#[derive(Debug, Copy, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct Bm25SearchOptions {
    /// Boolean mode for the query's bare terms (`Or` = should, `And` = must).
    pub mode: BoolMode,
    /// Which BM25 corpus statistics to score with.
    pub stats: Bm25Stats,
    /// Similarity parameters to score with, overriding whatever each
    /// column declared. `None` — the default — scores every column with
    /// its own declared pair, which is also the pair its stored bounds
    /// were baked at, so the query pays nothing for pruning.
    ///
    /// An override is corrected for: the reader inflates each column's
    /// stored bounds by the supremum of the ratio between the two
    /// parameter sets, so results stay exact and only pruning power is
    /// traded. A column whose declared pair already equals the override
    /// pays nothing either.
    pub bm25: Option<Bm25Params>,
}

impl Bm25SearchOptions {
    /// Default options: `Or` mode, global statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the boolean mode for the query's bare terms.
    pub fn with_mode(mut self, mode: BoolMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set which BM25 corpus statistics to score with.
    pub fn with_stats(mut self, stats: Bm25Stats) -> Self {
        self.stats = stats;
        self
    }

    /// Score with these similarity parameters instead of each column's
    /// declared pair — `k1` (term frequency saturation, `> 0`) and `b`
    /// (length normalization, in `[0, 1]`). Invalid values are rejected
    /// when the search runs.
    ///
    /// Intended for relevance experimentation, which is what a
    /// query-time knob is good for: it needs no rebuild, and the cost
    /// is bound looseness rather than wrong results. A pair you mean to
    /// keep belongs on the column (`FtsField::bm25`), where the bounds
    /// are built with it and the correction disappears.
    pub fn with_bm25(mut self, k1: f32, b: f32) -> Self {
        self.bm25 = Some(Bm25Params::new(k1, b));
        self
    }
}

impl From<&str> for BoolMode {
    fn from(s: &str) -> Self {
        match s {
            "and" => BoolMode::And,
            "or" => BoolMode::Or,
            _ => BoolMode::Or,
        }
    }
}
