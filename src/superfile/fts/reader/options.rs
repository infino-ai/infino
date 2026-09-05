// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Query-option types for FTS search: the default-operator [`BoolMode`],
//! the [`Bm25Stats`] idf-source selector, and the [`Bm25SearchOptions`]
//! builder. Part of the `fts::reader::*` public surface.

use std::sync::Arc;

use super::expansion::QueryExpansion;

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
    /// fragments. The default.
    #[default]
    PerSuperfile,
    /// Score every superfile against table-wide idf: the document count
    /// and per-term document-frequencies aggregated across all
    /// superfiles in the query's manifest snapshot. A term then has one
    /// idf for the whole table, so a fragmented table ranks like a
    /// single unified corpus, at the cost of a document-frequency
    /// gather before scoring. (Length normalization still uses each
    /// superfile's own average document length.)
    Global,
}

impl From<&str> for Bm25Stats {
    fn from(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "global" => Bm25Stats::Global,
            _ => Bm25Stats::PerSuperfile,
        }
    }
}

/// Options for a BM25 search: the boolean `mode`, the corpus-statistics
/// `stats`, and an optional per-call query `expansion`. Set fields with
/// the `with_*` builders; [`Default`] is [`BoolMode::Or`] with
/// [`Bm25Stats::PerSuperfile`] and no expansion.
///
/// ```ignore
/// // OR mode, per-superfile stats (the defaults):
/// Bm25SearchOptions::new()
/// // AND mode, global stats:
/// Bm25SearchOptions::new().with_mode(BoolMode::And).with_stats(Bm25Stats::Global)
/// // A vocabulary for this call only:
/// Bm25SearchOptions::new().with_expansion(Some(Arc::new(QueryExpansion::new().stop(["the"]))))
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct Bm25SearchOptions {
    /// Boolean mode for the query's bare terms (`Or` = should, `And` = must).
    pub mode: BoolMode,
    /// Which BM25 corpus statistics to score with.
    pub stats: Bm25Stats,
    /// Stop terms and term groups applied to this call's query. `None`
    /// (the default) applies whatever expansion is registered on the
    /// table for the searched column; `Some` overrides that registration
    /// for this call. See [`QueryExpansion`].
    pub expansion: Option<Arc<QueryExpansion>>,
}

impl Bm25SearchOptions {
    /// Default options: `Or` mode, per-superfile statistics, no
    /// per-call expansion.
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

    /// Apply a query-time expansion to this call only, overriding the
    /// column's registration; `None` (the default) falls back to it.
    pub fn with_expansion(mut self, expansion: Option<Arc<QueryExpansion>>) -> Self {
        self.expansion = expansion;
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
