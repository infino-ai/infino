// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`IndexSpec`] — declares which columns of a table are full-text
//! (BM25) indexed and which are vector (IVF kNN) indexed. Passed to
//! [`Connection::create_table`](crate::Connection::create_table) alongside
//! the Arrow schema.

use crate::superfile::{
    builder::FtsConfig,
    fts::tokenize::ASCII_LOWER_TOKENIZER,
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

/// A full-text index declaration: the column and the analyzer
/// (tokenizer) name applied to it.
#[derive(Debug, Clone)]
struct FtsIndex {
    column: String,
    analyzer: String,
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
    fts: Vec<FtsIndex>,
    vectors: Vec<VectorIndex>,
}

impl IndexSpec {
    /// An empty spec — no FTS, no vector indexes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark `column` as full-text (BM25) indexed with the default
    /// `ascii_lower` analyzer (ASCII split + lowercase, non-ASCII
    /// dropped). The column must be a UTF-8 string column in the schema.
    /// Use [`fts_with_analyzer`](Self::fts_with_analyzer) to pick a
    /// different analyzer.
    pub fn fts(self, column: impl Into<String>) -> Self {
        self.fts_with_analyzer(column, ASCII_LOWER_TOKENIZER)
    }

    /// Mark `column` as full-text (BM25) indexed with a named analyzer
    /// (`"ascii_lower"` or `"standard"` — the Unicode-aware UAX #29
    /// tokenizer that keeps non-ASCII text). The analyzer is per column:
    /// each FTS column is tokenized with its own, so columns in one table
    /// may use different analyzers.
    pub fn fts_with_analyzer(
        mut self,
        column: impl Into<String>,
        analyzer: impl Into<String>,
    ) -> Self {
        self.fts.push(FtsIndex {
            column: column.into(),
            analyzer: analyzer.into(),
        });
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
    /// of the public spec — defaults are applied here. The analyzer
    /// choice rides on the options' shared tokenizer (resolved by
    /// `table_tokenizer`), not on `FtsConfig`.
    pub(crate) fn to_configs(&self) -> (Vec<FtsConfig>, Vec<VectorConfig>) {
        let fts = self
            .fts
            .iter()
            .map(|f| FtsConfig {
                column: f.column.clone(),
                positions: false,
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
