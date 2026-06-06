//! Typed error enums for the superfile crate.
//!
//! Variants live here so every module can name the same error
//! type without circular imports; the constructors are in the
//! modules that produce each error.

use thiserror::Error;

/// Errors that can occur while building a superfile.
#[derive(Debug, Error)]
pub enum BuildError {
    #[error("schema is missing the declared id_column {0:?}")]
    MissingIdColumn(String),

    #[error("id_column {0:?} must be Decimal128(38, 0); found {1:?}")]
    IdColumnWrongType(String, String),

    #[error("FTS column {column:?} must be LargeUtf8; found {actual:?}")]
    FtsColumnMustBeLargeUtf8 { column: String, actual: String },

    #[error("FTS column {column:?} has invalid type {actual:?}")]
    FtsColumnTypeInvalid { column: String, actual: String },

    #[error("duplicate column name {0:?}")]
    DuplicateColumnName(String),

    #[error("logical name {0:?} duplicated across fts_columns and vector_columns")]
    DuplicateLogicalName(String),

    #[error("user column name {0:?} contains reserved \\x1F separator")]
    ReservedSeparatorInColumnName(String),

    #[error("user column name {0:?} starts with reserved prefix 'inf.'")]
    ReservedPrefixInColumnName(String),

    #[error("vector column {column:?} declares dim={dim}; must be in [16, 4096]")]
    VectorDimOutOfRange { column: String, dim: usize },

    /// The column requested a rerank codec that this build of infino
    /// does not implement. Supported codecs today: `fp32`,
    /// `sq8`, `none` (see
    /// [`crate::superfile::vector::rerank_codec::RerankCodec`]).
    #[error(
        "vector column {column:?}: rerank codec {codec:?} is not supported; \
         supported codecs are fp32, sq8, none"
    )]
    VectorRerankCodecUnimplemented { column: String, codec: &'static str },

    #[error(
        "vector column {column:?}: query/added vector dim {actual} does not match declared dim {expected}"
    )]
    VectorDimMismatch {
        column: String,
        expected: usize,
        actual: usize,
    },

    #[error("vectors slice has {actual} entries but {expected} vector columns are declared")]
    VectorCountMismatch { expected: usize, actual: usize },

    #[error("row arrow_values has {actual} entries but schema has {expected} columns")]
    WrongRowShape { expected: usize, actual: usize },

    #[error("RecordBatch schema does not match builder schema")]
    BatchSchemaMismatch,

    #[error("FTS column {0:?} not found in schema")]
    FtsColumnMissing(String),

    #[error("io error during build: {0}")]
    Io(#[from] std::io::Error),

    #[error("parquet/arrow error during build: {0}")]
    Footer(#[from] crate::superfile::format::footer::FooterError),
}

/// Errors that can occur while opening / reading a superfile.
#[derive(Debug, Error)]
pub enum ReadError {
    #[error("required KV metadata key {0:?} is missing")]
    MissingKv(&'static str),

    #[error("bad magic bytes for {section}: expected {expected:?}, got {actual:?}")]
    BadMagic {
        section: &'static str,
        expected: &'static [u8],
        actual: Vec<u8>,
    },

    #[error("unsupported format version {0}")]
    UnsupportedVersion(String),

    #[error("CRC32C mismatch in section {section}{column}")]
    ChecksumMismatch {
        section: &'static str,
        column: String, // empty if not column-scoped
    },

    #[error("malformed format-version string {0:?}")]
    MalformedVersion(String),

    #[error("io error during read: {0}")]
    Io(#[from] std::io::Error),

    #[error("parquet/arrow error during read: {0}")]
    Footer(#[from] crate::superfile::format::footer::FooterError),

    #[error("malformed key/value metadata: {0}")]
    MalformedKv(String),

    #[error("schema unavailable in Parquet metadata")]
    MissingSchema,

    #[error("FTS error: {0}")]
    Fts(Box<FtsError>),

    #[error("vector error: {0}")]
    Vector(Box<VectorError>),

    #[error("column {0:?} not found in superfile schema")]
    UnknownColumn(String),

    #[error("local_doc_id {doc_id} out of range (segment has {n_docs} docs)")]
    DocIdOutOfRange { doc_id: u32, n_docs: u64 },

    #[error("take-by-doc-id requires eager bytes; reader was opened via open_lazy")]
    LazyReaderUnsupported,

    #[error("columnar read error: {0}")]
    Columnar(String),
}

impl From<FtsError> for ReadError {
    fn from(e: FtsError) -> Self {
        ReadError::Fts(Box::new(e))
    }
}

impl From<VectorError> for ReadError {
    fn from(e: VectorError) -> Self {
        ReadError::Vector(Box::new(e))
    }
}

/// Errors specific to FTS query execution.
#[derive(Debug, Error)]
pub enum FtsError {
    #[error("unknown FTS column {0:?}")]
    UnknownColumn(String),

    #[error("read error: {0}")]
    Read(#[from] ReadError),
}

/// Errors specific to vector query execution.
#[derive(Debug, Error)]
pub enum VectorError {
    #[error("unknown vector column {0:?}")]
    UnknownColumn(String),

    #[error("query dimension {got} does not match column dimension {expected}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("read error: {0}")]
    Read(#[from] ReadError),

    /// The underlying [`crate::superfile::LazyByteSource`]
    /// surfaced a typed error during a range fetch (storage failure,
    /// out-of-bounds range, …). Stringified for crate-boundary
    /// stability; callers that need the typed
    /// `LazyByteSourceError` should match on the source directly.
    #[error("lazy source error during vector search: {0}")]
    LazySource(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_error_renders_helpfully() {
        let e = BuildError::DuplicateColumnName("body".into());
        assert!(e.to_string().contains("body"));
    }

    #[test]
    fn read_error_renders_section_and_column() {
        let e = ReadError::ChecksumMismatch {
            section: "fts/postings",
            column: " (column 'title')".into(),
        };
        let s = e.to_string();
        assert!(s.contains("fts/postings"));
        assert!(s.contains("title"));
    }

    #[test]
    fn fts_error_can_wrap_read_error() {
        let inner = ReadError::MissingKv("inf.fts.columns");
        let outer: FtsError = inner.into();
        assert!(matches!(outer, FtsError::Read(_)));
    }

    #[test]
    fn vector_error_renders_dim_mismatch() {
        let e = VectorError::DimensionMismatch {
            expected: 768,
            got: 384,
        };
        let s = e.to_string();
        assert!(s.contains("768") && s.contains("384"));
    }
}
