// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`IndexSpec`] — declares which columns of a table are full-text
//! (BM25) indexed and which are vector (IVF kNN) indexed. Passed to
//! [`Connection::create_table`](crate::Connection::create_table) alongside
//! the Arrow schema.

use std::{collections::HashSet, sync::Arc};

use arrow_schema::{Schema, SchemaRef};

use crate::{
    InfinoError,
    catalog::validate_schema,
    superfile::{
        builder::FtsConfig,
        fts::{
            analysis::{Stemmer, Stopwords},
            bm25::Bm25Params,
            tokenize::STANDARD_TOKENIZER,
        },
        range::KeyKind,
        vector::{builder::VectorConfig, distance::Metric, rerank_codec::RerankCodec},
    },
    supertable::{
        companion::{CompanionSpec, check_purpose},
        schema::{ColumnEncoding, ENCODING_META_KEY, SORT_META_KEY},
    },
};

/// Default rotation-matrix RNG seed for vector columns. The seed only
/// has to be stable for a given table; the public API does not vary it.
pub(crate) const DEFAULT_ROT_SEED: u64 = 0x5EED_5EED_5EED_5EED;

/// A vector index declaration: column, dimensionality, and distance metric.
#[derive(Debug, Clone)]
struct VectorIndex {
    column: String,
    dim: usize,
    metric: Metric,
    /// The rotation seed, which the public builder does not vary: it is
    /// [`DEFAULT_ROT_SEED`] unless a schema document being replayed onto a
    /// new table carried another, in which case the table has to be built
    /// with the one the document named.
    rot_seed: u64,
    /// Likewise the rerank codec; `None` leaves the engine's default.
    rerank_codec: Option<RerankCodec>,
}

/// One full-text (BM25) indexed column, with its per-column options.
///
/// Passed to [`IndexSpec::fts`]. A plain column name converts with all
/// defaults (the `standard` analyzer, stored text), so the common case
/// stays `.fts("body")`; build a `FtsField` to change an option:
///
/// ```
/// use infino::{FtsField, IndexSpec, Stemmer};
/// let spec = IndexSpec::new()
///     .fts("title")
///     .fts(FtsField::new("body").stemmer(Stemmer::English).stored(false));
/// # let _ = spec;
/// ```
#[derive(Debug, Clone)]
pub struct FtsField {
    column: String,
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

    /// Name the column's base tokenizer. `"standard"`, the default, is the
    /// only one; any other name is refused at `create_table`.
    pub fn analyzer(mut self, name: impl Into<String>) -> Self {
        self.analyzer = name.into();
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
    ranges: Vec<String>,
    encodings: Vec<(String, ColumnEncoding)>,
    sort: Vec<String>,
    companions: Vec<Companion>,
}

/// A hidden companion table declared on a table: its purpose, its Arrow
/// schema and the indexes over it. See [`IndexSpec::companion`].
#[derive(Debug, Clone)]
struct Companion {
    purpose: String,
    schema: SchemaRef,
    indexes: IndexSpec,
}

impl IndexSpec {
    /// An empty spec — no FTS, no vector indexes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a column as full-text (BM25) indexed. Takes a plain column
    /// name for the defaults, or an [`FtsField`] to set the analysis
    /// filters and whether the raw text is stored:
    ///
    /// ```
    /// use infino::{FtsField, IndexSpec, Stopwords};
    /// let spec = IndexSpec::new()
    ///     .fts("title")
    ///     .fts(FtsField::new("body").stopwords(Stopwords::English).stored(false));
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
            rot_seed: DEFAULT_ROT_SEED,
            rerank_codec: None,
        });
        self
    }

    /// Mark `column` as range indexed: each superfile keeps the column's
    /// rows in value order, so a SQL range or equality predicate on it
    /// (`price > 500`, `ts BETWEEN a AND b`, `qty = 3`) reads only the rows
    /// it matches instead of scanning the column.
    ///
    /// The per-file minimum and maximum the table keeps for every column
    /// already skip the files and pages a range cannot reach, which is
    /// enough when the rows arrive in the column's order (a timestamp on
    /// time-ordered data). A range index is for a column whose values are
    /// spread across every file — a price, a score, a duration — where
    /// those bounds skip nothing. The cost is the sorted copy: a few bytes
    /// per row of the column, in every superfile.
    ///
    /// The column must be an integer, float, date or timestamp column in the
    /// schema, and carry no other index.
    ///
    /// ```
    /// use infino::IndexSpec;
    /// let spec = IndexSpec::new().fts("title").range("price");
    /// # let _ = spec;
    /// ```
    pub fn range(mut self, column: impl Into<String>) -> Self {
        self.ranges.push(column.into());
        self
    }

    /// The range-indexed columns, in declaration order.
    pub(crate) fn range_columns(&self) -> &[String] {
        &self.ranges
    }

    /// Encode `column` in every superfile's Parquet body as `encoding`
    /// names: [`ColumnEncoding::Delta`] for an integer, date, time,
    /// timestamp or duration column, [`ColumnEncoding::ByteStreamSplit`]
    /// for an `f32` or `f64` one. Built for time series, where timestamps
    /// arrive in order a steady step apart and metric values change
    /// slowly: both shrink such a column well past what the dictionary or
    /// plain encoding does under the same codec.
    ///
    /// The encoding is stored on the column ([`ENCODING_META_KEY`]), so
    /// every append and compaction merge after writes the column the same
    /// way, and a column a later write adds can name one the same way in
    /// its field metadata. It changes how the body is written only: every
    /// query, index and reader works as before, and any Parquet reader
    /// decodes the body.
    ///
    /// ```
    /// use infino::{ColumnEncoding, IndexSpec};
    /// let spec = IndexSpec::new()
    ///     .encoding("time", ColumnEncoding::Delta)
    ///     .encoding("value", ColumnEncoding::ByteStreamSplit);
    /// # let _ = spec;
    /// ```
    pub fn encoding(mut self, column: impl Into<String>, encoding: ColumnEncoding) -> Self {
        self.encodings.push((column.into(), encoding));
        self
    }

    /// Refuses an encoding the table cannot carry: on a column `schema`
    /// does not have, of a type the encoding does not take, or declared
    /// twice.
    pub(crate) fn check_encodings(&self, schema: &Schema) -> Result<(), InfinoError> {
        let mut seen = HashSet::new();
        for (column, encoding) in &self.encodings {
            let refuse = |reason: String| {
                Err(InfinoError::Config(format!(
                    "{} encoding on {column:?}: {reason}",
                    encoding.as_str()
                )))
            };
            let Ok(field) = schema.field_with_name(column) else {
                return refuse("the schema has no such column".into());
            };
            if !encoding.accepts(field.data_type()) {
                return refuse(format!(
                    "the encoding does not take a `{}` column",
                    field.data_type()
                ));
            }
            if !seen.insert(column.as_str()) {
                return refuse("declared twice".into());
            }
        }
        Ok(())
    }

    /// `schema` with each declared encoding ([`ENCODING_META_KEY`]) and
    /// each sort key column's place ([`SORT_META_KEY`]) named on its column,
    /// where the table's schema document keeps them. Unchanged when the
    /// spec declares neither.
    pub(crate) fn stamp_encodings(&self, schema: SchemaRef) -> SchemaRef {
        if self.encodings.is_empty() && self.sort.is_empty() {
            return schema;
        }
        let fields: Vec<_> = schema
            .fields()
            .iter()
            .map(|field| {
                let encoding = self
                    .encodings
                    .iter()
                    .find(|(column, _)| column == field.name());
                let place = self.sort.iter().position(|column| column == field.name());
                if encoding.is_none() && place.is_none() {
                    return Arc::clone(field);
                }
                let mut metadata = field.metadata().clone();
                if let Some((_, encoding)) = encoding {
                    metadata.insert(ENCODING_META_KEY.to_string(), encoding.as_str().to_string());
                }
                if let Some(place) = place {
                    metadata.insert(SORT_META_KEY.to_string(), place.to_string());
                }
                Arc::new(field.as_ref().clone().with_metadata(metadata))
            })
            .collect();
        Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
    }

    /// Keep the table's rows ordered by `columns`, the first before the
    /// next: every compaction merge writes its output in that order, so the
    /// rows that share the first column's value sit together in each file
    /// it writes, and a filter on it reads only the row groups and pages
    /// whose Parquet statistics can hold the value. Built for time series:
    /// a series key first and the time second puts each series' points
    /// together, in time order, where a dashboard reads them and where its
    /// timestamps and values compress best.
    ///
    /// Rows are in that order from the first compaction that rewrites them;
    /// a file an append writes keeps the commit's order until then. The
    /// columns must be scalar columns of the schema (a number, a string, a
    /// date or a timestamp), each named once. A table with a vector index
    /// cannot have one. Declared at `create_table` only.
    ///
    /// ```
    /// use infino::IndexSpec;
    /// let spec = IndexSpec::new().sort_by(["series", "time"]);
    /// # let _ = spec;
    /// ```
    pub fn sort_by<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.sort = columns.into_iter().map(Into::into).collect();
        self
    }

    /// Refuses a sort key the table cannot keep: a column `schema` does not
    /// have, one that is not a scalar, one named twice, or any on a table
    /// with a vector index.
    pub(crate) fn check_sort(&self, schema: &Schema) -> Result<(), InfinoError> {
        if self.sort.is_empty() {
            return Ok(());
        }
        if !self.vectors.is_empty() {
            return Err(InfinoError::Config(
                "a sort key on a table with a vector index: the index orders its rows".into(),
            ));
        }
        let mut seen = HashSet::new();
        for column in &self.sort {
            let refuse = |reason: String| {
                Err(InfinoError::Config(format!(
                    "sort key column {column:?}: {reason}"
                )))
            };
            let Ok(field) = schema.field_with_name(column) else {
                return refuse("the schema has no such column".into());
            };
            if field.data_type().is_nested() {
                return refuse(format!(
                    "a `{}` column cannot be sorted on",
                    field.data_type()
                ));
            }
            if !seen.insert(column.as_str()) {
                return refuse("named twice".into());
            }
        }
        Ok(())
    }

    /// Refuses a range index the table cannot carry: a column `schema`
    /// does not have, one of a type the index does not key, one declared
    /// twice, or one that already carries a full-text or vector index.
    pub(crate) fn check_ranges(&self, schema: &Schema) -> Result<(), InfinoError> {
        let mut seen = HashSet::new();
        for column in &self.ranges {
            let refuse = |reason: String| {
                Err(InfinoError::Config(format!(
                    "range index on {column:?}: {reason}"
                )))
            };
            let Ok(field) = schema.field_with_name(column) else {
                return refuse("the schema has no such column".into());
            };
            if KeyKind::of(field.data_type()).is_none() {
                return refuse(format!(
                    "a range index needs an integer, float, date or timestamp column, not `{}`",
                    field.data_type()
                ));
            }
            if !seen.insert(column.as_str()) {
                return refuse("declared twice".into());
            }
            if self.fts.iter().any(|f| &f.column == column)
                || self.vectors.iter().any(|v| &v.column == column)
            {
                return refuse("the column carries another index".into());
            }
        }
        Ok(())
    }

    /// Create the table with a hidden companion table beside it: a second
    /// table with its own Arrow `schema` and its own full-text and vector
    /// `indexes`, for state the caller keeps about the table — a ledger of
    /// what has been loaded into it, say — that should live and die with
    /// the table but never be mistaken for one of the catalog's tables.
    ///
    /// `purpose` names the companion: reach it with
    /// [`Supertable::companion`] by the same name. It must be non-empty, at
    /// most 64 characters of `[a-z0-9_]`, unique on the table, and not
    /// `vector_index`, which the table's own hidden vector index uses.
    /// `indexes` may not declare companions of its own. Declare several
    /// companions with several calls.
    ///
    /// The companion is stored under the table's location and recorded on
    /// its manifest, is never listed by
    /// [`Connection::list_tables`](crate::Connection::list_tables), and is
    /// removed with the table by
    /// [`Connection::drop_table`](crate::Connection::drop_table) with
    /// `purge`. It is an ordinary table otherwise: append, update, delete,
    /// search and optimize it through the handle, and change its schema
    /// with [`Supertable::apply_schema`]. Companions are declared at
    /// `create_table` only; one cannot be added to an existing table.
    /// A `memory://` or hosted connection refuses a spec with companions.
    ///
    /// A companion is strictly optional, and the engine never interprets
    /// what it holds. The engine guarantees three things: the companion is
    /// hidden from the catalog, it shares the table's lifecycle, and it
    /// keeps the schema it was declared or patched with. Everything else
    /// belongs to the component that creates it: what its purpose means,
    /// what its rows say, when they are written, and any consistency it
    /// keeps with the table's own rows (writing the table's new rows before
    /// it removes old ones, say). No engine operation reads a companion or
    /// depends on its contents.
    ///
    /// Every engine that writes the table must be at least the version
    /// that created it. An older one does not know the manifest field
    /// naming the companions and drops it on its next commit; the
    /// companion's data is untouched, and [`Supertable::companion`]
    /// re-attaches it from its storage location.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use infino::arrow_schema::{DataType, Field, Schema};
    /// use infino::IndexSpec;
    ///
    /// let ledger = Arc::new(Schema::new(vec![
    ///     Field::new("key", DataType::LargeUtf8, false),
    ///     Field::new("rows", DataType::UInt64, true),
    /// ]));
    /// let spec = IndexSpec::new()
    ///     .fts("body")
    ///     .companion("ledger", ledger, IndexSpec::new().fts("key"));
    /// # let _ = spec;
    /// ```
    ///
    /// [`Supertable::companion`]: crate::Supertable::companion
    /// [`Supertable::apply_schema`]: crate::Supertable::apply_schema
    pub fn companion(
        mut self,
        purpose: impl Into<String>,
        schema: SchemaRef,
        indexes: IndexSpec,
    ) -> Self {
        self.companions.push(Companion {
            purpose: purpose.into(),
            schema,
            indexes,
        });
        self
    }

    /// Whether the spec declares any companion.
    pub(crate) fn has_companions(&self) -> bool {
        !self.companions.is_empty()
    }

    /// Refuses companions a table cannot be created with: a purpose
    /// [`check_purpose`] refuses, a purpose declared twice, a companion
    /// declaring companions of its own, or a companion whose schema or
    /// analyzers a table would be refused for.
    pub(crate) fn check_companions(&self) -> Result<(), InfinoError> {
        let mut purposes = HashSet::new();
        for companion in &self.companions {
            check_purpose(&companion.purpose).map_err(InfinoError::Config)?;
            if !purposes.insert(companion.purpose.as_str()) {
                return Err(InfinoError::Config(format!(
                    "companion purpose {:?} is declared twice",
                    companion.purpose
                )));
            }
            if companion.indexes.has_companions() {
                return Err(InfinoError::Config(format!(
                    "companion {:?} declares companions of its own; a companion cannot have any",
                    companion.purpose
                )));
            }
            if !companion.indexes.ranges.is_empty() {
                return Err(InfinoError::Config(format!(
                    "companion {:?} declares a range index; a companion takes full-text and \
                     vector indexes only",
                    companion.purpose
                )));
            }
            validate_schema(&companion.schema)?;
            companion.indexes.check_analyzers()?;
        }
        Ok(())
    }

    /// The declared companions, lowered to what the table options carry.
    pub(crate) fn companion_specs(&self) -> Vec<CompanionSpec> {
        self.companions
            .iter()
            .map(|companion| {
                let (fts_columns, vector_columns) = companion.indexes.to_configs();
                CompanionSpec {
                    purpose: companion.purpose.clone(),
                    schema: Arc::clone(&companion.schema),
                    fts_columns,
                    vector_columns,
                }
            })
            .collect()
    }

    /// The same declaration, carrying a seed and codec a schema document
    /// already fixed. Not public: a caller declaring a new table does not
    /// choose these, but a table created from another's document must be
    /// built with what that document said rather than silently taking the
    /// defaults.
    pub(crate) fn vector_as_recorded(
        mut self,
        column: impl Into<String>,
        dim: usize,
        metric: Metric,
        rot_seed: u64,
        rerank_codec: RerankCodec,
    ) -> Self {
        self.vectors.push(VectorIndex {
            column: column.into(),
            dim,
            metric,
            rot_seed,
            rerank_codec: Some(rerank_codec),
        });
        self
    }

    /// Refuses an FTS column that names an analyzer other than `standard`.
    pub(crate) fn check_analyzers(&self) -> Result<(), InfinoError> {
        match self.fts.iter().find(|f| f.analyzer != STANDARD_TOKENIZER) {
            Some(f) => Err(InfinoError::Config(format!(
                "FTS column {:?}: unsupported analyzer {:?}; the only analyzer is {STANDARD_TOKENIZER:?}",
                f.column, f.analyzer
            ))),
            None => Ok(()),
        }
    }

    /// FTS column names, in declaration order.
    pub(crate) fn fts_columns(&self) -> Vec<String> {
        self.fts.iter().map(|f| f.column.clone()).collect()
    }

    /// FTS stopword sets, in declaration order (parallel to
    /// [`fts_columns`](Self::fts_columns)).
    pub(crate) fn fts_stopwords(&self) -> Vec<Stopwords> {
        self.fts.iter().map(|f| f.stopwords).collect()
    }

    /// FTS stemmers, in declaration order (parallel to
    /// [`fts_columns`](Self::fts_columns)).
    pub(crate) fn fts_stemmers(&self) -> Vec<Stemmer> {
        self.fts.iter().map(|f| f.stemmer).collect()
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
    /// supertable options take. `rot_seed` / `rerank_codec` are not part of
    /// the public spec, so a column declared through [`Self::vector`]
    /// carries the defaults; one replayed from a schema document carries
    /// what that document recorded.
    pub(crate) fn to_configs(&self) -> (Vec<FtsConfig>, Vec<VectorConfig>) {
        let fts = self
            .fts
            .iter()
            .map(|f| {
                FtsConfig::new(f.column.clone())
                    .stopwords(f.stopwords)
                    .stemmer(f.stemmer)
                    .positions(f.positions)
                    .stored(f.stored)
                    .bm25(f.bm25.k1, f.bm25.b)
            })
            .collect();
        let vectors = self
            .vectors
            .iter()
            .map(|v| {
                let config = VectorConfig::new(v.column.clone(), v.dim, v.rot_seed, v.metric);
                match v.rerank_codec {
                    Some(codec) => config.with_rerank_codec(codec),
                    None => config,
                }
            })
            .collect();
        (fts, vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a single declared column lowers to on each analysis surface.
    fn analysis_of(field: FtsField) -> (Stopwords, Stemmer) {
        let spec = IndexSpec::new().fts(field);
        (
            spec.fts_stopwords().remove(0),
            spec.fts_stemmers().remove(0),
        )
    }

    /// The analysis setters are independent: each carries its own option
    /// through to its own persisted field, so they commute and neither
    /// can clobber the other.
    #[test]
    fn the_analysis_setters_commute() {
        let want = (Stopwords::English, Stemmer::English);
        assert_eq!(
            analysis_of(
                FtsField::new("t")
                    .stopwords(Stopwords::English)
                    .stemmer(Stemmer::English)
            ),
            want
        );
        assert_eq!(
            analysis_of(
                FtsField::new("t")
                    .stemmer(Stemmer::English)
                    .stopwords(Stopwords::English)
            ),
            want
        );
        // Turning a filter off is explicit.
        assert_eq!(
            analysis_of(
                FtsField::new("t")
                    .stopwords(Stopwords::English)
                    .stopwords(Stopwords::None)
            ),
            (Stopwords::None, Stemmer::None)
        );
    }

    /// The defaults, asserted where they are declared: no filters, no
    /// positions, text stored.
    #[test]
    fn a_bare_declaration_takes_the_plain_standard_analyzer() {
        let spec = IndexSpec::new().fts("body");
        assert_eq!(spec.fts_stopwords(), vec![Stopwords::None]);
        assert_eq!(spec.fts_stemmers(), vec![Stemmer::None]);
        assert_eq!(spec.fts_positions(), vec![false]);
        assert_eq!(spec.fts_stored(), vec![true]);
    }

    fn series_schema() -> Schema {
        Schema::new(vec![
            arrow_schema::Field::new(
                "time",
                arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, None),
                false,
            ),
            arrow_schema::Field::new("value", arrow_schema::DataType::Float64, false),
            arrow_schema::Field::new("host", arrow_schema::DataType::LargeUtf8, false),
        ])
    }

    /// A declared encoding is checked against the column it names: the
    /// column must exist, take the encoding, and be declared once.
    #[test]
    fn an_encoding_the_column_cannot_take_is_refused() {
        let schema = series_schema();
        let ok = IndexSpec::new()
            .encoding("time", ColumnEncoding::Delta)
            .encoding("value", ColumnEncoding::ByteStreamSplit);
        ok.check_encodings(&schema)
            .expect("both columns take their encodings");
        for (spec, why) in [
            (
                IndexSpec::new().encoding("host", ColumnEncoding::Delta),
                "a string",
            ),
            (
                IndexSpec::new().encoding("value", ColumnEncoding::Delta),
                "a float",
            ),
            (
                IndexSpec::new().encoding("time", ColumnEncoding::ByteStreamSplit),
                "a timestamp",
            ),
            (
                IndexSpec::new().encoding("missing", ColumnEncoding::Delta),
                "no such column",
            ),
            (
                IndexSpec::new()
                    .encoding("time", ColumnEncoding::Delta)
                    .encoding("time", ColumnEncoding::Delta),
                "declared twice",
            ),
        ] {
            assert!(
                matches!(spec.check_encodings(&schema), Err(InfinoError::Config(_))),
                "{why} must be refused"
            );
        }
    }

    /// The encodings ride on their columns' metadata; the other columns
    /// and the metadata a column already had are left as they were.
    #[test]
    fn declared_encodings_are_stamped_on_their_columns() {
        let schema = Arc::new(series_schema());
        let spec = IndexSpec::new().encoding("time", ColumnEncoding::Delta);
        let stamped = spec.stamp_encodings(Arc::clone(&schema));
        let meta = |name: &str| {
            stamped
                .field_with_name(name)
                .expect("column")
                .metadata()
                .get(ENCODING_META_KEY)
                .cloned()
        };
        assert_eq!(meta("time").as_deref(), Some("delta"));
        assert_eq!(meta("value"), None);
        assert!(Arc::ptr_eq(
            &IndexSpec::new().stamp_encodings(Arc::clone(&schema)),
            &schema
        ));
    }

    /// A sort key names scalar columns the schema has, each once, on a
    /// table with no vector index.
    #[test]
    fn a_sort_key_the_table_cannot_keep_is_refused() {
        let schema = series_schema();
        IndexSpec::new()
            .sort_by(["host", "time"])
            .check_sort(&schema)
            .expect("a string and a timestamp sort");
        for (spec, why) in [
            (IndexSpec::new().sort_by(["missing"]), "no such column"),
            (IndexSpec::new().sort_by(["host", "host"]), "named twice"),
            (
                IndexSpec::new()
                    .sort_by(["time"])
                    .vector("value", 2, Metric::Cosine),
                "a vector index",
            ),
        ] {
            assert!(
                matches!(spec.check_sort(&schema), Err(InfinoError::Config(_))),
                "{why} must be refused"
            );
        }
    }

    /// Each sort key column carries its place in the key.
    #[test]
    fn the_sort_key_is_stamped_on_its_columns_in_order() {
        let stamped = IndexSpec::new()
            .sort_by(["host", "time"])
            .stamp_encodings(Arc::new(series_schema()));
        let place = |name: &str| {
            stamped
                .field_with_name(name)
                .expect("column")
                .metadata()
                .get(SORT_META_KEY)
                .cloned()
        };
        assert_eq!(place("host").as_deref(), Some("0"));
        assert_eq!(place("time").as_deref(), Some("1"));
        assert_eq!(place("value"), None);
    }
}
