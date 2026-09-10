// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`IndexSpec`] — declares which columns of a table are full-text
//! (BM25) indexed and which are vector (IVF kNN) indexed. Passed to
//! [`Connection::create_table`](crate::Connection::create_table) alongside
//! the Arrow schema.

use crate::superfile::{
    builder::FtsConfig,
    fts::{
        analysis::{Base, Stemmer, Stopwords, chain_name, parse_chain_name},
        bm25::Bm25Params,
        tokenize::STANDARD_TOKENIZER,
    },
    vector::{builder::VectorConfig, distance::Metric},
};

/// Default rotation-matrix RNG seed for vector columns. The seed only
/// has to be stable for a given table; the public API does not vary it.
const DEFAULT_ROT_SEED: u64 = 0x5EED_5EED_5EED_5EED;

/// A vector index declaration: column, dimensionality, and distance metric.
#[derive(Debug, Clone)]
struct VectorIndex {
    column: String,
    dim: usize,
    metric: Metric,
}

/// One full-text (BM25) indexed column, with its per-column options.
///
/// Passed to [`IndexSpec::fts`]. A plain column name converts with all
/// defaults (`standard` analyzer, stored text), so the common case
/// stays `.fts("body")`; build a `FtsField` to change an option:
///
/// ```
/// use infino::{FtsField, IndexSpec};
/// let spec = IndexSpec::new()
///     .fts("title")
///     .fts(FtsField::new("body").analyzer("ascii_lower").stored(false));
/// # let _ = spec;
/// ```
#[derive(Debug, Clone)]
pub struct FtsField {
    column: String,
    /// The base analyzer name. The stopword set and stemmer are held
    /// separately and composed onto it at lowering, so the two setters
    /// and [`FtsField::analyzer`] may be called in any order and the
    /// column still ends up with one canonical analyzer name.
    analyzer: String,
    stopwords: Stopwords,
    stemmer: Stemmer,
    positions: bool,
    stored: bool,
    bm25: Bm25Params,
}

impl FtsField {
    /// Declare `column` as full-text indexed with the defaults: the
    /// `standard` analyzer (Unicode UAX #29 word segmentation, full
    /// Unicode lowercasing) and the raw text stored. The column must be
    /// a UTF-8 string column in the table schema.
    pub fn new(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            analyzer: STANDARD_TOKENIZER.to_string(),
            stopwords: Stopwords::None,
            stemmer: Stemmer::None,
            positions: false,
            stored: true,
            bm25: Bm25Params::STANDARD,
        }
    }

    /// Pick the column's analyzer by name — `"standard"` (the default)
    /// or `"ascii_lower"`, which splits on ASCII alphanumerics and
    /// drops every non-ASCII token. The analyzer is per column: each
    /// FTS column is tokenized with its own, so columns in one table
    /// may use different analyzers. It is recorded with the table and
    /// cannot be changed afterwards.
    ///
    /// This names the *base* tokenizer. [`FtsField::stopwords`] and
    /// [`FtsField::stemmer`] add filters on top of it, and the column's
    /// recorded analyzer name is the whole chain
    /// (`"standard+stop=english+stem=english"`). A composite name is
    /// also accepted here, so a name read back from a table round-trips;
    /// a filter setter then replaces that component of it.
    ///
    /// Validated at `create_table`, with the column named in the error.
    pub fn analyzer(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        match parse_chain_name(&name) {
            // A composite name sets the components it *names* and
            // clears none, so the three setters commute: a base name
            // never undoes a `.stopwords()` that came before it, and a
            // filter setter after a composite name replaces just that
            // component. To turn a filter off, name its `None`.
            Some((base, stopwords, stemmer)) => {
                self.analyzer = base.name().to_string();
                if stopwords != Stopwords::None {
                    self.stopwords = stopwords;
                }
                if stemmer != Stemmer::None {
                    self.stemmer = stemmer;
                }
            }
            // Unresolvable: kept verbatim so `create_table`'s error
            // names the analyzer the caller actually wrote, rather than
            // some normalized form of it.
            None => self.analyzer = name,
        }
        self
    }

    /// Remove this column's stopwords — the very common words whose
    /// presence in a document says almost nothing about what it is
    /// about. Off by default.
    ///
    /// Applies to both sides: the words leave the index, and they leave
    /// a query too, so searching `"the climate policy"` searches
    /// `climate policy`. Each removed word leaves a **hole** in the
    /// token positions, so an exact phrase still knows the words it
    /// matched were not adjacent in the text: with the English set,
    /// `"new york"` does not match `new the york`, and
    /// `"end of the world"` matches only text with two words between
    /// `end` and `world`.
    ///
    /// The trade is index size and speed against the handful of queries
    /// that are *about* a stopword: once `the` is not indexed, no query
    /// can find it — `"to be or not to be"` matches everything, because
    /// nothing is left of it. Declare it on a column of prose where the
    /// common words carry no signal, not on one holding short identifiers
    /// or titles.
    ///
    /// Recorded with the table and cannot be changed afterwards — it
    /// decides what is in the index, and there is no migration. A
    /// filter's effect is not recoverable from the index it produced:
    /// the tokens it removed were never written, so changing it means
    /// building a new table from the source text and re-ingesting.
    ///
    /// Which makes one combination permanent. On a
    /// [`stored(false)`](FtsField::stored) column the source text is
    /// never kept, so there is nothing left to re-analyze — not even a
    /// full rebuild can undo the choice. Declare a filter with
    /// `stored(false)` only once you are sure of it.
    pub fn stopwords(mut self, stopwords: Stopwords) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Reduce this column's words to their stems, so a search for one
    /// inflection finds the others. Off by default.
    ///
    /// Applies to both sides — index and query — so
    /// [`Stemmer::English`] makes `running`, `runs` and `run` one term
    /// and any of them find all of them. Irregular forms it has no rule
    /// for (`ran`, `went`) stay distinct.
    ///
    /// The trade is precision: stemming conflates words that a reader
    /// would not, so a query for an exact word can return documents
    /// carrying a relative of it, and there is no way to ask for the
    /// unstemmed form on a stemmed column. It also shifts the scoring —
    /// folding inflections together raises the merged term's document
    /// frequency, and so lowers its idf.
    ///
    /// Recorded with the table and cannot be changed afterwards — it
    /// decides what is in the index, and there is no migration. A
    /// stem is not invertible: the index holds `run`, never the
    /// `running` it came from, so changing or removing the stemmer
    /// means building a new table from the source text and
    /// re-ingesting.
    ///
    /// Which makes one combination permanent. On a
    /// [`stored(false)`](FtsField::stored) column the source text is
    /// never kept, so there is nothing left to re-analyze — not even a
    /// full rebuild can undo the choice. Declare a filter with
    /// `stored(false)` only once you are sure of it.
    pub fn stemmer(mut self, stemmer: Stemmer) -> Self {
        self.stemmer = stemmer;
        self
    }

    /// Record token positions for this column, which is what exact
    /// phrase queries (`"climate policy"`) need. Off by default.
    ///
    /// The trade is index size: positions roughly double the column's
    /// full-text index footprint, so they are a per-column opt-in
    /// rather than something every column pays for. A column without
    /// them answers a phrase query with an error naming the column,
    /// never a silent bag-of-words fallback that would return the wrong
    /// documents.
    pub fn positions(mut self, positions: bool) -> Self {
        self.positions = positions;
        self
    }

    /// Keep the raw text in the table (the default). Pass `false` for
    /// an index-only column: the text is searchable (BM25, token and
    /// phrase matching) but never stored, so it cannot be read back —
    /// not in SQL results, not in a search projection, not in
    /// predicates. `append` and `update` batches still carry the
    /// column (the text has to arrive to be indexed); it is dropped at
    /// write time. The trade is storage: large text that is only ever
    /// searched skips the stored copy entirely.
    pub fn stored(mut self, stored: bool) -> Self {
        self.stored = stored;
        self
    }

    /// Set this column's BM25 similarity parameters — `k1` (term
    /// frequency saturation, `> 0`) and `b` (length normalization, in
    /// `[0, 1]`). Defaults to `1.2` / `0.75`, the standard pair.
    ///
    /// Per column, and recorded in the index: the stored block-max
    /// bounds are built with the pair declared here, and the pair
    /// itself is written alongside them, so a reader always scores with
    /// the parameters its bounds belong to. A search may score with a
    /// different pair — the bounds are corrected for the difference —
    /// so this is the column's default rather than a commitment.
    ///
    /// Validated at `create_table`, with the column named in the error.
    pub fn bm25(mut self, k1: f32, b: f32) -> Self {
        self.bm25 = Bm25Params::new(k1, b);
        self
    }

    /// This column's whole analysis chain as one canonical analyzer
    /// name — the single string that reaches the superfile's
    /// `inf.fts.columns` entry, the catalog record, the remote wire and
    /// the options-hash. A column with no filter yields its base name
    /// unchanged, so a default column is byte-identical everywhere to
    /// one declared before the chain existed.
    ///
    /// An analyzer name that does not resolve passes through untouched:
    /// validation at `create_table` is what reports it, and it must
    /// report the name as written.
    fn chain_analyzer(&self) -> String {
        match Base::from_name(&self.analyzer) {
            Some(base) => chain_name(base, self.stopwords, self.stemmer).to_string(),
            None => self.analyzer.clone(),
        }
    }
}

impl From<&str> for FtsField {
    fn from(column: &str) -> Self {
        Self::new(column)
    }
}

impl From<String> for FtsField {
    fn from(column: String) -> Self {
        Self::new(column)
    }
}

/// Declares the search indexes to build over a table's columns.
///
/// Built fluently; every column named here must exist in the table's
/// Arrow schema. Columns with no index are still stored and queryable
/// via SQL — they just have no BM25 / vector index.
///
/// ```
/// use infino::{IndexSpec, Metric};
/// let spec = IndexSpec::new()
///     .fts("body")
///     .vector("embedding", 384, Metric::Cosine);
/// # let _ = spec;
/// ```
#[derive(Debug, Clone, Default)]
pub struct IndexSpec {
    fts: Vec<FtsField>,
    vectors: Vec<VectorIndex>,
}

impl IndexSpec {
    /// An empty spec — no FTS, no vector indexes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a column as full-text (BM25) indexed. Takes a plain column
    /// name for the defaults, or an [`FtsField`] to set the analyzer
    /// and whether the raw text is stored:
    ///
    /// ```
    /// use infino::{FtsField, IndexSpec};
    /// let spec = IndexSpec::new()
    ///     .fts("title")
    ///     .fts(FtsField::new("body").analyzer("ascii_lower").stored(false));
    /// # let _ = spec;
    /// ```
    pub fn fts(mut self, field: impl Into<FtsField>) -> Self {
        self.fts.push(field.into());
        self
    }

    /// Mark `column` as vector (IVF kNN) indexed. `dim` is the vector
    /// dimensionality and `metric` the distance metric. The column must be a
    /// `FixedSizeList<Float32, dim>` column in the schema. The IVF centroid
    /// count is derived from the data at build time, not declared here.
    pub fn vector(mut self, column: impl Into<String>, dim: usize, metric: Metric) -> Self {
        self.vectors.push(VectorIndex {
            column: column.into(),
            dim,
            metric,
        });
        self
    }

    /// FTS column names, in declaration order.
    pub(crate) fn fts_columns(&self) -> Vec<String> {
        self.fts.iter().map(|f| f.column.clone()).collect()
    }

    /// FTS analyzer names, in declaration order (parallel to
    /// [`fts_columns`](Self::fts_columns)).
    ///
    /// Each is the column's **whole** analysis chain as one canonical
    /// composite name, so this is all the catalog record, the remote
    /// create-table wire and the options-hash have to carry for
    /// stopwords and stemming — see
    /// [`FtsField::chain_analyzer`].
    pub(crate) fn fts_analyzers(&self) -> Vec<String> {
        self.fts.iter().map(|f| f.chain_analyzer()).collect()
    }

    /// FTS positions flags, in declaration order (parallel to
    /// [`fts_columns`](Self::fts_columns)).
    pub(crate) fn fts_positions(&self) -> Vec<bool> {
        self.fts.iter().map(|f| f.positions).collect()
    }

    /// FTS stored flags, in declaration order (parallel to
    /// [`fts_columns`](Self::fts_columns)).
    pub(crate) fn fts_stored(&self) -> Vec<bool> {
        self.fts.iter().map(|f| f.stored).collect()
    }

    /// FTS BM25 parameters, in declaration order (parallel to
    /// [`fts_columns`](Self::fts_columns)).
    pub(crate) fn fts_bm25(&self) -> Vec<Bm25Params> {
        self.fts.iter().map(|f| f.bm25).collect()
    }

    /// Vector index declarations as `(column, dim, metric)`, in declaration
    /// order. Used by the remote transport to serialize the spec.
    #[cfg(feature = "remote")]
    pub(crate) fn vector_indexes(&self) -> impl Iterator<Item = (&str, usize, Metric)> {
        self.vectors
            .iter()
            .map(|v| (v.column.as_str(), v.dim, v.metric))
    }

    /// Lower to the internal `(FtsConfig, VectorConfig)` lists the
    /// supertable options take. `rot_seed` / `rerank_codec` are not part
    /// of the public spec — defaults are applied here.
    pub(crate) fn to_configs(&self) -> (Vec<FtsConfig>, Vec<VectorConfig>) {
        let fts = self
            .fts
            .iter()
            .map(|f| {
                FtsConfig::new(f.column.clone())
                    .analyzer(f.chain_analyzer())
                    .positions(f.positions)
                    .stored(f.stored)
                    .bm25(f.bm25.k1, f.bm25.b)
            })
            .collect();
        let vectors = self
            .vectors
            .iter()
            .map(|v| VectorConfig::new(v.column.clone(), v.dim, DEFAULT_ROT_SEED, v.metric))
            .collect();
        (fts, vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lowered analyzer name for a single declared column.
    fn analyzer_of(field: FtsField) -> String {
        IndexSpec::new().fts(field).fts_analyzers().remove(0)
    }

    /// The three analysis setters commute. `FtsField` holds the base
    /// name and the two filters separately and composes them only at
    /// lowering, precisely so a caller cannot lose a filter by
    /// declaring the base after it — which a naive "append to a string"
    /// implementation would do.
    #[test]
    fn the_analysis_setters_commute() {
        let want = "ascii_lower+stop=english+stem=english";
        assert_eq!(
            analyzer_of(
                FtsField::new("t")
                    .analyzer("ascii_lower")
                    .stopwords(Stopwords::English)
                    .stemmer(Stemmer::English)
            ),
            want
        );
        // Filters first, base last — the order that would drop them if
        // `.analyzer()` reset the chain.
        assert_eq!(
            analyzer_of(
                FtsField::new("t")
                    .stemmer(Stemmer::English)
                    .stopwords(Stopwords::English)
                    .analyzer("ascii_lower")
            ),
            want
        );
        // And interleaved.
        assert_eq!(
            analyzer_of(
                FtsField::new("t")
                    .stopwords(Stopwords::English)
                    .analyzer("ascii_lower")
                    .stemmer(Stemmer::English)
            ),
            want
        );
    }

    /// A composite name round-trips through `.analyzer()`, which is
    /// what `open_table` relies on: it rebuilds the spec by handing the
    /// recorded name straight back, so a chain that did not survive
    /// that would silently reopen a table with a different analyzer
    /// than its postings were built with.
    #[test]
    fn a_composite_name_round_trips_through_the_analyzer_setter() {
        for name in [
            "standard",
            "ascii_lower",
            "standard+stop=english",
            "standard+stem=english",
            "standard+stop=english+stem=english",
            "ascii_lower+stop=english+stem=english",
        ] {
            assert_eq!(analyzer_of(FtsField::new("t").analyzer(name)), name);
        }
    }

    /// A composite name sets the components it *names* and clears none,
    /// so a filter setter after one replaces just that component and a
    /// setter before one is not undone.
    #[test]
    fn a_composite_name_sets_only_what_it_names() {
        // The name carries no stopword set, so the earlier setter stands.
        assert_eq!(
            analyzer_of(
                FtsField::new("t")
                    .stopwords(Stopwords::English)
                    .analyzer("standard+stem=english")
            ),
            "standard+stop=english+stem=english"
        );
        // Turning a filter back off is explicit, never implied by a name.
        assert_eq!(
            analyzer_of(
                FtsField::new("t")
                    .analyzer("standard+stop=english+stem=english")
                    .stopwords(Stopwords::None)
            ),
            "standard+stem=english"
        );
    }

    /// An unresolvable analyzer passes through untouched, so
    /// `create_table`'s error can name what the caller actually wrote
    /// rather than a normalized form of it.
    #[test]
    fn an_unresolvable_analyzer_is_lowered_verbatim() {
        for name in ["nonesuch", "standard+stop=german", "STANDARD"] {
            assert_eq!(analyzer_of(FtsField::new("t").analyzer(name)), name);
        }
    }

    /// The defaults, asserted where they are declared: `standard`, no
    /// filters, no positions. A default column's analyzer name is the
    /// bare base name, which is what keeps it byte-identical on disk to
    /// one declared before chains existed.
    #[test]
    fn a_bare_declaration_takes_the_plain_standard_analyzer() {
        let spec = IndexSpec::new().fts("body");
        assert_eq!(spec.fts_analyzers(), vec![STANDARD_TOKENIZER.to_string()]);
        assert_eq!(spec.fts_positions(), vec![false]);
        assert_eq!(spec.fts_stored(), vec![true]);
    }
}
