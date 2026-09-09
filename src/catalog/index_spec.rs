// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`IndexSpec`] — declares which columns of a table are full-text
//! (BM25) indexed and which are vector (IVF kNN) indexed. Passed to
//! [`Connection::create_table`](crate::Connection::create_table) alongside
//! the Arrow schema.

use crate::superfile::{
    builder::FtsConfig,
    fts::{bm25::Bm25Params, tokenize::STANDARD_TOKENIZER},
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
    analyzer: String,
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
    pub fn analyzer(mut self, name: impl Into<String>) -> Self {
        self.analyzer = name.into();
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
    pub(crate) fn fts_analyzers(&self) -> Vec<String> {
        self.fts.iter().map(|f| f.analyzer.clone()).collect()
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
                    .analyzer(f.analyzer.clone())
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
